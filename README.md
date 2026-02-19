# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-19 | 今日论文总数: 406

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Image Measurement Method for Automatic Insertion of Forks into Inclined Pallet

**arXiv ID:** 2602.16178 | [PDF](https://arxiv.org/pdf/2602.16178v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 2. MolCrystalFlow: Molecular Crystal Structure Prediction via Flow Matching

**arXiv ID:** 2602.16020 | [PDF](https://arxiv.org/pdf/2602.16020v1)

**作者:** Cheng Zeng `[一作]` (University of Florida), Mingjie Liu `[通讯]` (University of Florida)

**通讯引用:** 6829 | [OpenAlex ID](https://openalex.org/A5100681146)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于流匹配的生成模型MolCrystalFlow，用于高效预测分子晶体的晶格、分子位置与取向；

**💡 创新点**

核心创新在于将分子作为刚体分离内分子与外部晶格的复杂性，并在相应的黎曼流形上联合学习晶格矩阵、分子质心与SO(3)取向；

**🔧 技术方法**

采用E(3)等变图神经网络、轴角表示、周期性黎曼流匹配、最优传输与自适应基分布的组合技术；

**📊 数据集**

使用公开的Thurlemann分子晶体数据集（11,488条）及OMC25-MCF子集（46,802条）进行训练与评估；

**📈 对比分析**

与MOFFlow和Genarris-3比较，MolCrystalFlow在结构匹配率、晶格体积偏差（≈3.9%）和推断速度（≈22 ms/结构）上均优于对照模型；

**⚠️ 局限性**

局限在于缺乏能量引导、仅采用刚体近似、未显式利用空间群约束，导致对分子构象多形体的预测仍受限。

---

## 3. Automated Re-Identification of Holstein-Friesian Cattle in Dense Crowds

**arXiv ID:** 2602.15962 | [PDF](https://arxiv.org/pdf/2602.15962v1)

**作者:** Phoenix Yu `[一作]` (University of Bristol), Neill W Campbell `[通讯]` (University of Bristol)

**通讯引用:** 2225 | [OpenAlex ID](https://openalex.org/A5109857567)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一套基于 OWLv2 与 SAM2 的无监督自动化检测与重识别（Re-ID）框架，能够在 Holstein-Friesian 牛群密集的“闪烁”背景下精准地提取个体分割并实现无标签重识别。

**💡 创新点**

创新点在于：①结合开源无词汇定位（OWLv2）与分割（SAM2）实现无需微调的高精度实例分割；②使用无监督对比学习（UCL）在自动生成的 RGB 分割数据上进行聚类，完成无标注的 Re-ID；③通过九天现场奶牛摄像数据验证框架的实用性与迁移性。

**🔧 技术方法**

使用的技术包括：Open‑Vocabulary Weight‑free Localization (OWLv2)、Segment Anything Model v2 (SAM2)、GroundingDINO/ GroundedSAM 等对比模型、无监督对比学习（NTXentLoss、kNN/K‑Means）、ResNet‑50 作为特征提取器、k‑fold 交叉验证。

**📊 数据集**

数据集为在英国布里斯托大学农场收集的九天室内奶牛 CCTV 视频，采集了 19–24 只牛在 12:00–14:00 期间的图像，自动生成约 524,469 个 RGB 分割样本。

**📈 对比分析**

与 YOLO‑v11x、RetinaNet、SAM2 baseline 以及 GroundingDINO/GroundedSAM 等基线相比，OWLv2+SAM2 在定位精度上提升约 48% 和 27%（IoU 0.450→0.898；匹配率 33.8%→75.4%），而在 Re‑ID 任务中取得 94.82%±4.10% 的准确率，显著高于传统基线。

**⚠️ 局限性**

限制包括：仍存在过/欠分割情况，导致部分 Re‑ID 误差；模型对快速移动的牛表现下降；当前实现依赖单一摄像头和固定时间窗口，缺乏跨场景或跨时间的鲁棒性；以及对极端遮挡或不同光照条件的适应性尚待进一步验证。

---

## 4. Hybrid Model Predictive Control with Physics-Informed Neural Network for Satellite Attitude Control

**arXiv ID:** 2602.15954 | [PDF](https://arxiv.org/pdf/2602.15954v1)

**作者:** Carlo Cena `[一作]` (Politecnico di Torino), Marcello Chiaberge `[通讯]` (Politecnico di Torino)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过物理信息神经网络（PINN）学习航天器姿态动力学，并将其嵌入混合模型预测控制（MPC）框架，实现姿态控制；

**💡 创新点**

在损失函数中加入物理约束并用拉格朗日双重方法动态平衡数据与物理权重；构建姿态误差<1°时自动切换到线性MPC的混合控制策略，从而提升稳态收敛速度与鲁棒性；

**🔧 技术方法**

使用多层感知器（MLP）+ PINN、物理约束损失、拉格朗日双重调节、非线性与线性MPC、Basilisk高保真仿真、torch和do‑mpc库；

**📊 数据集**

基于Basilisk模拟的300+50轨道姿态轨迹数据集，包含随机初始姿态、轨道位置、反作用轮速度以及环境扰动，生成输入-输出对并按67/33划分训练/验证集；

**📈 对比分析**

与纯数据驱动MLP、非线性MPC、线性MPC进行对比；在10步递归预测中MRE从19.67%降至6.26%；在300个蒙特卡洛噪声实验中混合MPC收敛时间43.9 s，稳态误差0.0023°，比线性MPC快约61%–76%，且稳态误差更低；

**⚠️ 局限性**

训练数据缺乏摩擦与噪声，仅在仿真环境下验证；模型对参数变化的鲁棒性受限于预设的物理假设；混合MPC在高扭矩需求时可能产生过大扭矩分布。

---

## 5. TurboADMM: A Structure-Exploiting Parallel Solver for Multi-Agent Trajectory Optimization

**arXiv ID:** 2602.15838 | [PDF](https://arxiv.org/pdf/2602.15838v1)

**作者:** Yucheng Chen `[一作]` `[通讯]`, Yucheng Chen

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 TurboADMM，一种专为多智能体轨迹优化设计的单机 QP 求解器，利用 ADMM 分解、Riccati 暖启动和 qpOASES 热启动实现并行求解。

**💡 创新点**

三大创新点：① 将 ADMM 与 Riccati 暖启动协同设计，使每个子问题在首次迭代即可获得高质量初始解；② 在每个 ADMM 迭代间复用 qpOASES 的 QR 分解实现热启动；③ 结合共享内存并行（OpenMP）实现近线性按智能体数扩展的求解速度。

**🔧 技术方法**

采用 ADMM 分解、Ricatti recursion、qpOASES 参数化 QP 热启动、OpenMP 并行；对齐线性系统与约束的块三角结构。

**📊 数据集**

使用 2–14 代理、20 步预测、4D 状态、2D 控制的二维碰撞回避基准（所有对间碰撞约束），在 Intel i7-155H 22 核 CPU 上测试。

**📈 对比分析**

与 OSQP、MOSEK、HPIPM 进行比较；TurboADMM 在 6–14 代理时实现 5–22 倍速度提升，内存和求解时间随代理数近线性增长，显著优于单机通用求解器。

**⚠️ 局限性**

局限：仅支持线性动力学与凸约束；对极大规模（>20 代理）尚未验证；需要手动设置 ADMM 惩罚参数，热启动对数值稳定性敏感。

---

## 6. Position-Aware Scene-Appearance Disentanglement for Bidirectional Photoacoustic Microscopy Registration

**arXiv ID:** 2602.15959 | [PDF](https://arxiv.org/pdf/2602.15959v1)

**作者:** Yiwen Wang `[一作]`, Jiahao Qin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出GPEReg‑Net实现双向光学分辨率光声显微镜（OR‑PAM）扫描的图像配准，采用场景-外观分离与全局位置编码，无需显式变形场估计。

**💡 创新点**

创新点包括：①使用AdaIN实现对域无关场景特征与全局外观码的解耦并直接重建配准图像；②引入全局位置编码（可学习+正弦编码+跨帧多头注意力）赋予网络时间上下文感知。

**🔧 技术方法**

技术细节：U‑Net 结构场景编码器、Instance Normalization、AdaIN 模块、可学习+正弦位置编码、跨帧注意力、L1 重建损失+MSE 场景对齐损失。

**📊 数据集**

使用 OR‑PAM‑Reg‑4K（4248 对图像）和 OR‑PAM‑Reg‑Temporal‑26K（26000+ 连续帧）进行实验评估。

**📈 对比分析**

与 SIFT、Demons、VoxelMorph、TransMorph、SAS‑Net 等方法对比：在 4K 数据上取得 NCC 0.953、SSIM 0.932、PSNR 34.49 dB，超过 SAS‑Net 3.8% SSIM、1.99 dB PSNR；在 26K 数据上虽单帧质量略低，但时序一致性最高，TNCC 与 TSSIM 均高于竞争对手。

**⚠️ 局限性**

局限性：固定容量的可学习位置编码表难以适应长时序变化；全局外观解耦在高时间变异、局部域移的情境下表现欠佳，需要可变长度编码和局部外观建模以提升鲁棒性。

---

## 7. Evaluating Demographic Misrepresentation in Image-to-Image Portrait Editing

**arXiv ID:** 2602.16149 | [PDF](https://arxiv.org/pdf/2602.16149v1)

**作者:** Huichan Seo `[一作]` (Carnegie Mellon University), Jean Oh `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2987 | [OpenAlex ID](https://openalex.org/A5019807694)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究开放权重指令引导的图像到图像（I2I）编辑中，系统评估种族、性别、年龄等人口属性下的身份保持失败，并首次定义了软抹除与刻板替换两种失效模式；构建基准并验证提示级身份约束的缓解效果。

**💡 创新点**

1) 明确表述并度量I2I编辑中的软抹除与刻板替换；2) 构建基于FairFace的人口平衡基准与诊断提示集；3) 证明提示级特征约束可无模型更新显著缓解身份变化；4) 展示VLM评估与人工评估高度一致，提供可扩展评估框架。

**🔧 技术方法**

使用三款开放权重指令引导I2I编辑器（FLUX.2-dev、Step1X-Edit-v1p2、Qwen-Image-Edit-2511），VLM评估器（Gemini 3.0 Flash Preview 与 GPT-5-mini），以及基于可观测特征的提示级身份约束。

**📊 数据集**

84张从FairFace按种族、性别、年龄因子抽样的人像；诊断提示集20条（涵盖职业与易感属性）；额外的WinoBias职业提示集用于性别-职业偏见测试。

**📈 对比分析**

通过生成5,040张编辑图像，使用VLM与人工评估进行对比。结果显示软抹除普遍存在，种族差异显著（非白人肤色变浅、种族变更率高）；提示级约束显著降低非白人种族变化；VLM与人类评估一致，VLM提供保守下限。

**⚠️ 局限性**

样本规模有限（仅84张人像），仅评估3款开放编辑器，WinoBias提示集可能不完全代表自然用户行为，闭源系统或未来架构的表现可能不同。

---

## 8. Can Causality Cure Confusion Caused By Correlation (in Software Analytics)?

**arXiv ID:** 2602.16091 | [PDF](https://arxiv.org/pdf/2602.16091v1)

**作者:** Amirali Rayegan `[一作]` (North Carolina State University), Tim Menzies `[通讯]` (North Carolina State University)

**通讯引用:** 14474 | [OpenAlex ID](https://openalex.org/A5077008083)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究将因果信息融入决策树的拆分准则，探讨其在软件工程多目标优化任务中的稳定性与鲁棒性，并与人类专家判断及传统相关性拆分模型进行对比。

**💡 创新点**

创新点在于提出基于条件熵（H(Y|X)/H(Y)）的因果拆分准则，并通过后门过滤（条件互信息）预剪枝去除混杂变量，从而在符号模型中实现对因果关系的约束与提升。

**🔧 技术方法**

采用的技术包括：传统EZR方差减少决策树；条件熵与条件互信息的因果拆分方法；bootstrap‑ensemble协议、Gini impurity、Kolmogorov‑Smirnov检验及Cliff's Delta等统计评估工具。

**📊 数据集**

实验使用MOOT仓库中的120+多目标优化任务，涵盖软件配置、云性能调优、项目与流程分析、特征模型、行为、金融、健康等多领域数据集。

**📈 对比分析**

通过对每个数据集生成20棵传统树和20棵因果树，计算其在测试集上的d2h优化结果，使用KS统计和Cliff's Delta比较两类模型的性能；结果显示因果树在稳定性指标上明显优于传统树，而在优化性能上差异不显著或略低。

**⚠️ 局限性**

局限性包括：条件互信息的预剪枝无法确立因果方向，因果拆分仍受估计误差影响；实验仅基于MOOT数据集，且人类专家样本有限，未验证方法在更大规模或实时系统中的可推广性。

---

## 9. Adaptive Semi-Supervised Training of P300 ERP-BCI Speller System with Minimum Calibration Effort

**arXiv ID:** 2602.15955 | [PDF](https://arxiv.org/pdf/2602.15955v1)

**作者:** Shumeng Chen `[一作]` (Emory University), Tianwen Ma `[通讯]` (Emory University)

**通讯引用:** 534 | [OpenAlex ID](https://openalex.org/A5103248151)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于EM-GMM的自适应半监督训练框架，用以降低P300脑机接口（BCI）文字输入器的校准负担。

**💡 创新点**

创新点在于将EM-GMM直接应用于EEG特征向量，利用少量标记数据初始化后，持续利用无标记数据进行在线自适应更新，并通过正则化提高目标信号峰值，从而在仅使用约10个字母（150个序列）校准时即可达到或超过传统完整校准的性能。

**🔧 技术方法**

核心技术包括：EM-GMM算法（在线和离线两种实现）、xDAWN谱滤波、特征维度约简、动态停止阈值、信息传输速率（ITR）和BCI效用评估。

**📊 数据集**

实验数据集由美国密歇根大学UM‑DBI实验室的15名受试者收集的P300 ERP信号构成（16通道、256 Hz），以及基于真实数据生成的200次仿真。

**📈 对比分析**

与完整校准的离线基线相比，适应方法在少量训练序列（75~150）下即可获得平均字符识别率>0.8、ITR和BCI效用持续提升，9/15受试者达到≥0.7的识别率，其中7名受试者表现更佳；完整校准在285个序列下平均识别率为91%。

**⚠️ 局限性**

局限性包括：对输入数据质量高度敏感；参数初始化对收敛影响大；未实现全局可迁移性；仅考虑了行列闪烁刺激，缺乏上下文或强化学习等自适应策略；对无标记数据的噪声鲁棒性仍需进一步提升。

---

## 10. From Tool Orchestration to Code Execution: A Study of MCP Design Choices

**arXiv ID:** 2602.15945 | [PDF](https://arxiv.org/pdf/2602.15945v1)

**作者:** Yuval Felendler `[一作]` (Ben Gurion University of Negev), Asaf Shabtai `[通讯]` (Ben Gurion University of Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究传统MCP与代码执行CE-MCP两种工具协作模型在效率、任务质量与安全性上的差异，并通过实验验证其性能与漏洞。

**💡 创新点**

提出CE-MCP架构并基于MAESTRO框架构建完整的攻击模型与分层防御策略，首次系统地评估代码生成与执行对安全的影响。

**🔧 技术方法**

利用GPT‑4系列大型语言模型、MCP‑Bench基准、MAESTRO安全框架以及CE‑MCP执行流程，对两种模型进行统一实验。

**📊 数据集**

使用MCP‑Bench的28台工具服务器（挑选10台代表性服务器）以及其对应的任务集合作为实验数据集。

**📈 对比分析**

通过比较Token使用量、整体执行时间、交互回合数以及任务完成率发现，CE‑MCP在多工具、多服务器任务上可减少约30–50% Token 与延迟，且回合数显著降低；但在部分三服务器任务中，因一次性代码生成的失误导致任务完成率略低。

**⚠️ 局限性**

实验仅涵盖代表性攻击场景，未针对自适应对手进行压力测试；防御措施对部分基于异常信息的攻击有效，但对更复杂的动态规避策略尚未充分验证。

---

## 11. CHAI: CacHe Attention Inference for text2video

**arXiv ID:** 2602.16132 | [PDF](https://arxiv.org/pdf/2602.16132v1)

**作者:** Joel Mathew Cherian `[一作]` (Georgia Institute of Technology), Anand Padmanabha Iyer `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 3288 | [OpenAlex ID](https://openalex.org/A5090733623)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于实体级跨推理缓存的无训练加速方法CHAI，利用Cache Attention在文本到视频扩散模型中显著减少推理步骤。

**💡 创新点**

创新点在于：① 采用实体级相似性进行跨推理缓存；② 引入Cache Attention在注意力中使用缓存的键值，仅在需要时注入共享结构；③ 在保持视频质量的前提下，仅使用8步即可生成与30步相近的视频。

**🔧 技术方法**

技术包括：扩散Transformer（OpenSora 1.2）修改、Cache Attention层、向量数据库检索实体嵌入、LRU缓存策略、FAISS、LongCLIP分词实体提取。

**📊 数据集**

使用数据集：VBench（692条多维度评测）与VidProM（1000条真实用户提示）进行评估。

**📈 对比分析**

与OpenSora 1.2（30步）和NIRVANA‑VID、AdaCache等基线对比，CHAI在VBench上实现3.35×加速、0.3%质量损失；在VidProM上实现1.65×加速、相近质量；在缓存约1‑5 GB下命中率>80%。

**⚠️ 局限性**

局限性：对高分辨率视频的缓存占用大；目前仅使用单个缓存latent，未充分利用多缓存组合；缺少与内部推理缓存的联合加速策略。

---

## 12. GPSBench: Do Large Language Models Understand GPS Coordinates?

**arXiv ID:** 2602.16105 | [PDF](https://arxiv.org/pdf/2602.16105v1)

**作者:** Thinh Hung Truong `[一作]` (University of Melbourne), Jianzhong Qi `[通讯]` (University of Melbourne)

**通讯引用:** 4612 | [OpenAlex ID](https://openalex.org/A5022290876)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GPSBench 评测基准，包含 57,800 个样本、17 个任务，涵盖纯 GPS 计算与实际地理推理两条赛道，评估 14 大型语言模型在全球坐标推理上的能力。

**💡 创新点**

创新点在于：①首次系统化衡量 LLM 在全球尺度 GPS 计算与地理知识融合的综合表现；②区分纯数学几何推理与需结合世界知识的应用推理；③通过噪声扰动检验模型是否真正理解坐标而非仅记忆；④研究微调对坐标推理与地理知识的互斥影响；⑤展示坐标增强能显著提升下游地理任务的性能。

**🔧 技术方法**

使用零样本提示的标准化评测流程，采用 Mean Absolute Percentage Error（MAPE）与准确率对数值与分类任务统一评分；对比不同模型家族与规模对坐标推理的影响；并进行分区域、分粒度的误差分析。

**📊 数据集**

基于 GeoNames 地名数据库（18,196 个地点）构建样本，结合 WGS84 参考坐标、标准大圆公式、UTM 与 Web Mercator 转换等公式，覆盖六大洲并包含多层级空间知识。

**📈 对比分析**

与 14 种 LLM（GPT、Gemini、Claude、Qwen3、Mistral 等）在纯 GPS 与应用赛道上进行对比，发现大模型在纯 GPS 计算（如距离、方位）能达 84%+，但在细粒度地名识别仅 23%；模型规模越大性能提升显著；微调可提升几何任务但会削弱世界知识任务；在下游 MapEval 与 Hierarchical Spatial 任务中加入坐标后准确率提升 6–23%。

**⚠️ 局限性**

局限包括：仅评估文本 LLM ；不使用链式思考或工具；提示全英文，缺乏跨语言评测；GeoNames 覆盖偏向城市，农村、海洋、极地地区缺失；单点城市坐标可能导致邻近城市被误判；误差评价采用百分比，无法反映不同尺度的实际应用需求。

---

## 13. SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks

**arXiv ID:** 2602.16187 | [PDF](https://arxiv.org/pdf/2602.16187v1)

**作者:** Zirui Zang `[一作]` (University of Pennsylvania), Rahul Mangharam `[通讯]` (University of Pennsylvania)

**通讯引用:** 4568 | [OpenAlex ID](https://openalex.org/A5009445756)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种用于迭代任务的安全信息论学习模型预测控制（SIT‑LMPC），通过将LMPC框架扩展到随机非线性系统并结合MPPI实现约束优化；

**💡 创新点**

①利用外部惩罚与在线自适应惩罚参数实现安全约束；②使用正则化流（normalizing flows）学习值函数，取代传统高斯或BNN；③通过GPU并行化实现实时控制；

**🔧 技术方法**

信息论MPPI（MPPI）、外部惩罚、在线自适应惩罚（AP）算法、正则化流（NSF）以及GPU并行采样；

**📊 数据集**

在三种场景下评估：线性点质量模型、模拟的无人车（单轨动力学）和真实1/5规模越野跑车，使用自定义轨迹生成的训练数据；

**📈 对比分析**

与传统LMPC、ABC‑LMPC（CEM‑MPC）以及固定惩罚的MPPI/ABC-LMPC比较。实验显示：在点质量模型中收敛速度最快；在模拟赛车和真实车辆中，SIT‑LMPC在保持安全的前提下显著降低圈速（约30%提升），并避免了ABC‑LMPC的多次碰撞；

**⚠️ 局限性**

仍需已知或可识别的系统动力学；对极大随机扰动和高度未知动力学的鲁棒性有限；训练成本与模型规模成正比，需GPU资源；

---

## 14. From Conflicts to Collisions: A Two-Stage Collision Scenario-Testing Approach for Autonomous Driving Systems

**arXiv ID:** 2602.15837 | [PDF](https://arxiv.org/pdf/2602.15837v1)

**作者:** Siyuan Chen `[一作]` (University of Tokyo), Manabu Okada `[通讯]` (TIER IV)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在模拟环境中使用两阶段搜索框架，对自动驾驶系统进行碰撞场景测试。

**💡 创新点**

创新点在于将冲突作为中间搜索目标，先生成多样化冲突场景再转化为碰撞。

**🔧 技术方法**

技术包括遗传算法、基于冲突的目标函数和专门的原子级冲突定向变异。

**📊 数据集**

使用百度Apollo ADS在LGSVL仿真平台上，随机生成的街景以及San Francisco地图。

**📈 对比分析**

与AV‑Fuzzer、MOSAT、Legend比较，发现碰撞类型多达12/14种，平均每种碰撞步数低于基线，整体发现率提升约50%。

**⚠️ 局限性**

局限在于需要手工定义冲突类型和变异策略，且对特定模拟器与ADS依赖性高，无法直接推广到所有平台。

---

## 15. DocSplit: A Comprehensive Benchmark Dataset and Evaluation Approach for Document Packet Recognition and Splitting

**arXiv ID:** 2602.15958 | [PDF](https://arxiv.org/pdf/2602.15958v1)

**作者:** Md Mofijul Islam `[一作]` (Amazon Web Services), Diego A. Socolinsky `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了文档包拆分（DocSplit）基准数据集与评价框架，用于评估模型在多页文档拆分任务中的边界检测、分类和页面排序能力。

**💡 创新点**

提出了专门针对文档包拆分的评价指标：结合 Rand Index、V‑measure 的聚类分数与 Kendall Tau 排序分数的综合 Packet Score，并设计五种逐步增加难度的测试场景。

**🔧 技术方法**

利用多模态大型语言模型（Claude Sonnet/Haiku、DeepSeek、Gemma、Qwen 3）进行实验，并使用文本+图像的结构化 JSON 输出进行评估。

**📊 数据集**

数据集基于 RVL‑CDIP‑MP，经过改造生成 5 种不同拆分场景（单类别顺序、随机、跨类别顺序、交错、完全随机）共约 5,231 页。

**📈 对比分析**

通过综合 Packet Score 与传统页面级准确率对比，发现 Qwen 3 在聚类与排序上表现最佳（Packet≈0.94），Gemma 在聚类上显著逊色；排序分数普遍高于 0.97，说明排序是相对容易的任务。

**⚠️ 局限性**

主要局限包括：边界检测仍是瓶颈；数据集仅涵盖英文北美商务/政府文件，难以泛化至多语种或非西方格式；以及基准依赖自动化标注，可能存在误标。

---

## 16. Kalman-Inspired Runtime Stability and Recovery in Hybrid Reasoning Systems

**arXiv ID:** 2602.15855 | [PDF](https://arxiv.org/pdf/2602.15855v1)

**作者:** Barak Or `[一作]` `[通讯]`, Barak Or

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文件介绍了IJCAI‑ECAI 2026会议论文的排版与格式要求，阐述了页边距、列宽、字体、标题、引用等细节；

**💡 创新点**

创新点是为会议统一制定了完整、细致的排版规范，方便作者快速遵循；

**🔧 技术方法**

主要使用 LaTeX 与 Word 模板（ijcai26.sty、ijcai26.tex、ijcai26.docx）完成排版；

**📊 数据集**

无研究数据集，本文仅为格式说明；

**📈 对比分析**

无方法比较，本文仅提供格式示例与规范；

**⚠️ 局限性**

局限在于仅适用于本次会议的论文排版，不能直接用于其他会议或期刊。

---

## 17. MultiCube-RAG for Multi-hop Question Answering

**arXiv ID:** 2602.15898 | [PDF](https://arxiv.org/pdf/2602.15898v1)

**作者:** Jimeng Shi `[一作]` (University of Illinois Urbana-Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 122538 | [OpenAlex ID](https://openalex.org/A5019539533)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练的多维立方体结构与迭代检索-推理框架 MultiCube-RAG，用于提升多跳问答性能。

**💡 创新点**

创新点在于：①基于本体的正交维度设计立方体，将主题、属性、关系映射到不同轴；②利用多立方体并按查询动态选择合适的立方体，实现可解释且高效的多步检索与推理；③完全不需要模型训练，依赖预训练 LLM 进行抽取与推理。

**🔧 技术方法**

核心技术包括：预训练 LLM（GPT‑4o‑mini / Llama3.3‑70B‑Instruct）用于本体抽取、子查询生成与立方体选择；立方体检索（稀疏精确匹配与稠密嵌入匹配）；迭代推理-检索循环；无监督立方体构建与增量更新。

**📊 数据集**

在四个多跳 QA 基准上进行实验：2WikiQA、HotpotQA、MuSiQue、LV‑Eval。

**📈 对比分析**

与九种基线（BM25、NV‑Embed‑v2、RAPTOR、GraphRAG、HippoRAG 2、IRCoT、RankCoT 等）对比，MultiCube‑RAG 在 2WikiQA、MuSiQue、LV‑Eval 上平均提升约 8.9%（EM/F1），在 HotpotQA 上接近最强基线，检索速度显著优于图结构方法。

**⚠️ 局限性**

局限性：需要先使用 LLM 抽取本体与维度，耗时和成本；立方体维度设计需人工或领域知识；在某些数据集上提升有限；对极大规模文档的增量更新仍需多轮 LLM 调用。

---

## 18. Quality-constrained Entropy Maximization Policy Optimization for LLM Diversity

**arXiv ID:** 2602.15894 | [PDF](https://arxiv.org/pdf/2602.15894v1)

**作者:** Haihui Pan `[一作]` (Zuoyebang Education Technology), Yang Song `[通讯]` (Zuoyebang Education Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出在对齐过程中同时优化输出多样性与质量的目标，提出QEMPO和QEMPO-KL两种质量约束熵最大化策略

**💡 创新点**

创新点在于将对齐任务分解为质量与多样性两部分，并通过熵最大化与KL约束实现更高多样性同时保持质量

**🔧 技术方法**

利用信息论熵、KL约束、策略梯度、离线DPO、在线GRPO等技术实现

**📊 数据集**

使用UltraFeedback、MT‑Bench、GSM8K、MATH500等公开数据集进行训练与评估

**📈 对比分析**

与RLHF、SPL等基线对比，实验表明QEMPO/ QEMPO‑KL在多样性上优于RLHF且质量保持或略优，在线模式下pass@k提升尤为显著

**⚠️ 局限性**

局限性包括对超参数敏感、训练稳定性需要额外约束、在极难任务上仍可能出现质量下降、且方法对参考模型的依赖仍存在

---

## 19. Discrete Stochastic Localization for Non-autoregressive Generation

**arXiv ID:** 2602.16169 | [PDF](https://arxiv.org/pdf/2602.16169v1)

**作者:** Yunshu Wu `[一作]` (University of California Riverside), Greg Ver Steeg `[通讯]` (University of California Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的训练框架DSL（Discrete Stochastic Localization），通过混合连续SNR噪声和掩码式终点噪声，让单一的去噪网络在多种中间噪声水平下均表现良好，从而显著提升非自回归扩散语言模型在迭代细化过程中的效率与质量。

**💡 创新点**

创新点在于将连续与离散噪声过程统一到一个SNR不变的去噪器中，并通过平滑化终点噪声和混合训练策略，使模型在自生成的中间草稿上保持鲁棒性与良好的不确定性校准；同时提出了一种轻量级的step‑budget感知remasking策略，仅调节一个超参数即可改善采样质量。

**🔧 技术方法**

使用的技术包括：基于高维球面嵌入的离散高斯扩散、SNR不变的MMSE去噪器、混合噪声训练（连续对数正态分布+平滑ROAR终点），以及改进的softmax转换器和去噪Transformer（DiT）架构；在采样阶段采用MDLM和ReMDM的remasking策略，并加入step‑budget感知的remasking强度控制。

**📊 数据集**

实验主要在OpenWebText（OWT）数据集上进行，使用GPT‑2 BPE分词，长度为1024，评估指标包括MAUVE、生成文本的生成PPL以及句子熵。

**📈 对比分析**

与原始MDLM、ReMDM、PRISM、ADLM等基线对比，DSL fine‑tuned模型在相同步数预算下，MAUVE提升显著（例如T=128时从0.057提升到0.639），生成PPL也下降；在高步数预算下亦保持与自回归模型相当的质量。仅调节remasking强度的策略进一步提升了MAUVE‑PPL折衷。

**⚠️ 局限性**

局限性包括：虽然训练改进显著降低了误差累积，但非自回归模型仍面临并行更新导致的条件独立性假设偏差；DSL并未从根本上解决因多Token并行生成而产生的联合分布偏差问题，且目前仅在无条件文本生成上验证，尚未扩展到更大模型或条件生成任务。

---

## 20. Extracting and Analyzing Rail Crossing Behavior Signatures from Videos using Tensor Methods

**arXiv ID:** 2602.16057 | [PDF](https://arxiv.org/pdf/2602.16057v1)

**作者:** Dawon Ahn `[一作]` (University of California Riverside), Evangelos E. Papalexakis `[通讯]` (University of California Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在多视角张量分解框架下，利用视频嵌入提取与相似度矩阵分析，发现铁路道口行为模式与地点相关性更高且进近期行为最具辨识度。

**💡 创新点**

首次将三阶段行为映射为多视角张量，采用非负对称CP分解挖掘可解释的时序行为组件，并以此实现跨地点行为聚类。

**🔧 技术方法**

使用TimeSformer预训练视频嵌入、余弦相似度构建多视角相似度张量、非负对称CP分解以及t‑SNE可视化。

**📊 数据集**

31条铁路道口视频（4个地点，共31个视频）于2024年2月采集，覆盖不同时间段。

**📈 对比分析**

通过CORCONDIA、重构误差与留一验证确定4阶CP分解；在该数据集上显示出明显的地点聚类效果，未与传统单地点统计方法直接对比，但证明跨地点行为可分辨性。

**⚠️ 局限性**

样本地点有限、缺乏道口几何/交通量等结构特征、使用通用预训练模型可能未能捕捉细粒度跨域行为。

---

## 21. Muon with Spectral Guidance: Efficient Optimization for Scientific Machine Learning

**arXiv ID:** 2602.16167 | [PDF](https://arxiv.org/pdf/2602.16167v1)

**作者:** Binghang Lu `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**通讯引用:** 6210 | [OpenAlex ID](https://openalex.org/A5078138445)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 SpecMuon 优化器，融合 Muon 的正交化梯度更新与模式级 Relaxed Scalar Auxiliary Variable（RSAV）机制，用于物理信息神经网络与神经算子训练，解决梯度失衡、多尺度和刚性问题。

**💡 创新点**

将 Muon 的奇异向量正交化更新与能量稳定化的 RSAV 结合，构成多模式梯度流；在理论上证明能量耗散、辅助变量正性及在 Polyak‑Łojasiewicz 条件下的线性收敛；在实现上采用低秩 SVD 截断，保持高效。

**🔧 技术方法**

低秩 SVD、Muon's orthogonalized update、RSAV 机制、能量耗散分析、Polyak‑Łojasiewicz 收敛证明、梯度截断与正则化、与 Adam/AdamW 对比实验。

**📊 数据集**

线性回归、1D Burgers 方程、2D 热方程、fractional PDE（fPINN‑DeepONet）等典型 PDE 数据集。

**📈 对比分析**

通过与 Adam、AdamW、原 Muon 进行对比实验，SpecMuon 在训练损失下降速度更快、稳定性更佳，最终 MSE 更低，收敛更快；虽然略有计算开销，但总体性能显著提升。

**⚠️ 局限性**

需要 SVD 与奇异值截断，增加计算与内存成本；目前仅在全量或小批量设置验证，未针对大规模随机/mini‑batch、复杂多物理耦合或高维算子问题做深入验证；RSAV 参数选择仍依赖经验调优。

---

## 22. Mitigating Gradient Inversion Risks in Language Models via Token Obfuscation

**arXiv ID:** 2602.15897 | [PDF](https://arxiv.org/pdf/2602.15897v1)

**作者:** Xinguo Feng `[一作]` (University of Queensland), Guangdong Bai `[通讯]` (City University of Hong Kong)

**通讯引用:** 1800 | [OpenAlex ID](https://openalex.org/A5015858067)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于token层面遮蔽的防御方法（GHOST），用于抵御语言模型梯度反演攻击。

**💡 创新点**

创新点在于利用大规模词表中语义相距但嵌入相近的“影子”token，分两步搜索和选择，打破梯度-嵌入-token之间的直接映射。

**🔧 技术方法**

使用多准则相似度筛选（余弦距离、词干相似性等）进行搜索；随后通过内部隐藏状态的MSE优化挑选shadow token；并与梯度噪声/剪枝等基线进行对比。

**📊 数据集**

在BERT、RoBERTa、DeBERTa、GPT‑2、Llama、Gemma等21种模型上，分别在SST‑2、Tweet Sentiment、Yahoo Answers、Enron Emails、Open Australian Legal Corpus、AG News等六类分类/生成数据集上进行评测。

**📈 对比分析**

与现有梯度噪声/剪枝、DLG、TAG、LAMP、GRAB、DAGER 等攻击/防御做对比，恢复率降至≈1%，攻击R‑1≤0.08；同时保持分类F1≥0.90、生成PPL≈5.45，几乎无功能损失。

**⚠️ 局限性**

对极端自适应攻击仍存在一定恢复概率（最高R‑1≈0.29），且在极大模型或长文本上搜索与选择成本较高，未验证跨语言或多语种情境。

---

## 23. Towards Fair and Efficient De-identification: Quantifying the Efficiency and Generalizability of De-identification Approaches

**arXiv ID:** 2602.15869 | [PDF](https://arxiv.org/pdf/2602.15869v1)

**作者:** Noopur Zambare `[一作]` (University of Alberta), Mohamed Abdalla `[通讯]` (University of Alberta)

**通讯引用:** 983 | [OpenAlex ID](https://openalex.org/A5058570740)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

系统评估了多种临床去标识化模型（BERT、ClinicalBERT、ModernBERT、小型LLM如Llama 1‑8B、Qwen 1.5‑7B以及大型LLM 70B、72B）的性能、推理效率和跨格式、跨语言、跨性别的泛化能力，并提出了 BERT‑MultiCulture‑DEID 模型。

**💡 创新点**

创新点在于①对小型模型与大型模型在效率‑性能权衡进行量化；②在多语言、多文化、不同性别的去标识化任务中系统性评估并发现大模型虽更稳健但效率低；③通过多语言训练提升 BERT 的泛化，发布 BERT‑MultiCulture‑DEID。

**🔧 技术方法**

采用 Transformer 预训练模型（BERT 系列）和大规模 LLM（Llama、Qwen）结合 LoRA 适配、Prompt‑tuning、以及规则基系统（PyDeid、Presidio），并使用 CIRE 指标评估临床信息保留。

**📊 数据集**

使用 MIMIC‑III（公开）和私有 Alberta Health Services 关节病转诊信（约 204 封）数据集，并通过 Faker 生成多语言 PII。

**📈 对比分析**

对比方法：在相同硬件（2×A100）下测量词/秒、时间/笔记、精确率/召回率、CIRE，发现 BERT 系列在 1528–2446 词/秒、0.99 召回率；小型 LLM 约 423–1271 词/秒、0.96 召回；大型 LLM 仅 8–10 词/秒、召回约 0.97 但精确率低；规则基系统最快但精确率最低。

**⚠️ 局限性**

主要局限包括：①仅在特定硬件与有限训练样本下评估，结果可能随硬件变化；②Faker 生成的多语言 PII 可能不完全代表真实多样性；③未覆盖所有语言与文化；④评估仅聚焦于 5 种非英语语言，且大型 LLM 训练成本高；⑤训练数据标签错误可能影响结果。

---

## 24. Algorithm-Based Pipeline for Reliable and Intent-Preserving Code Translation with LLMs

**arXiv ID:** 2602.16106 | [PDF](https://arxiv.org/pdf/2602.16106v1)

**作者:** Shahriar Rumi Dipto `[一作]` (University of Saskatchewan), Chanchal K. Roy `[通讯]` (University of Saskatchewan)

**通讯引用:** 9247 | [OpenAlex ID](https://openalex.org/A5102756770)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在 LLM 代码翻译中加入语言中立算法草图的管道，先生成算法规范再生成目标语言代码。

**💡 创新点**

创新点在于把算法描述作为前置中间层，显式锁定 I/O、数据结构、数值规则和循环边界，从而大幅降低语义漂移与运行时错误。

**🔧 技术方法**

采用链式思考提示、两阶段 LLM 调用（先生成算法再生成代码）以及五大通用 LLM（DeepSeek R1/V3、Llama 4 Maverick、GPT‑4o、Qwen2.5）。

**📊 数据集**

使用 Avatar 与 CodeNet 两大公开数据集，分别包含 Python–Java 互译任务。

**📈 对比分析**

对比直接一轮翻译与算法化管道，按编译成功、运行时稳定、测试通过等指标统计，微平均准确率从 67.7% 提升至 78.5%（+10.8%），显著消除词法/标记错误、完整性缺失和结构声明错误。

**⚠️ 局限性**

局限包括：算法步骤仍需人工指定完整约束，缺少自动化验证；对其他语言对的通用性需进一步验证；模型对类型/初始化等深层语义约束的处理仍不充分。

---

## 25. Harnessing Implicit Cooperation: A Multi-Agent Reinforcement Learning Approach Towards Decentralized Local Energy Markets

**arXiv ID:** 2602.16062 | [PDF](https://arxiv.org/pdf/2602.16062v1)

**作者:** Nelson Salazar-Pena `[一作]`, Andres Gonzalez-Mancera `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在分布式能源市场中提出隐式合作框架，利用多智能体强化学习和系统级 KPI 作为环境信号实现去中心化的能量协调。

**💡 创新点**

通过将 KPI 嵌入观测空间，解决 DTDE 下的非平稳性与协调缺失，并证明 APPO‑DTDE 为最优配置，形成自组织的交易社区。

**🔧 技术方法**

使用多智能体强化学习（APPO、PPO、SAC）结合 CTCE/CTDE/DTDE 三种训练范式，并在 IEEE 34 节点拓扑上进行仿真。

**📊 数据集**

在三乘三因子实验（CTCE/CTDE/DTDE × APPO/PPO/SAC）中使用 IEEE 34 节点物理网络及相应的市场规则数据集。

**📈 对比分析**

与中心化基准 CTCE 比较，APPO‑DTDE 协调得分 91.7%，实现高效交易；DTDE 在效率与稳定性之间权衡，显著降低网格平衡方差 31%。

**⚠️ 局限性**

局限在于仿真时间较长、对更大规模/异构网络的推广尚未验证，且 KPI 设计在面对策略攻击与协同偏差时的鲁棒性待进一步研究。

---

## 26. Towards Efficient Constraint Handling in Neural Solvers for Routing Problems

**arXiv ID:** 2602.16012 | [PDF](https://arxiv.org/pdf/2602.16012v1)

**作者:** Jieyi Bi `[一作]` (Nanyang Technological University), Cathy Wu `[通讯]` (MIT)

**通讯引用:** 1278 | [OpenAlex ID](https://openalex.org/A5102933746)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Construct-and-Refine（CaR）框架，利用端到端联合训练，将构造模块与细化模块相结合，通过显式学习的可行性细化来高效处理车辆路径问题中的复杂约束。

**💡 创新点**

①首次将可行性细化与构造‑改进混合，使用短步细化实现高效与可行性兼顾；②通过共享 Transformer 编码器实现跨范式表示学习，提升隐式可行性意识；③引入熵多样性损失和监督细化损失，显著增强构造质量与细化效果；④在多约束 VRP 上实现零不可行率并大幅压缩搜索步数。

**🔧 技术方法**

强化学习（REINFORCE）+CMDP 松弛、注意力 Transformer 编码器、熵多样性损失、监督细化损失、短步改进（T_R=5/10/20）、共享编码器、POMO/PIP 构造器、NeuOpt/N2S 细化器。

**📊 数据集**

TSPTW、CVRPBLTW、CVRP、TSP、SOP、TSPDL 以及 CVRPLIB 等标准 VRP 数据集，按规模 50、100、200 等尺寸生成。

**📈 对比分析**

与经典求解器 LKH‑3、OR‑Tools、HGS 以及多种基线（POMO、PIP、NeuOpt‑GIRE、LCP、EAS+SGBS 等）在可行率、最优性差距和运行时间上对比；结果显示 CaR 在复杂约束下可行率 0%，最优性差距仅 0.005% 左右，运行时间比传统改进搜索快 8–10 倍，且在大部分指标上优于现有经典与神经网络求解器。

**⚠️ 局限性**

仍受训练样本规模限制，针对极大规模实例的可扩展性尚需进一步验证；对非 VRP 的约束（如调度、资源分配）尚未彻底评估；联合训练增加了模型复杂度，部署与调优成本较高。

---

## 27. Multi-source Heterogeneous Public Opinion Analysis via Collaborative Reasoning and Adaptive Fusion: A Systematically Integrated Approach

**arXiv ID:** 2602.15857 | [PDF](https://arxiv.org/pdf/2602.15857v1)

**作者:** Yi Liu `[一作]` (Xi'an Jiaotong University), Yi Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 57222 | [OpenAlex ID](https://openalex.org/A5100330449)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种跨平台协同推理与自适应融合（CRAF）框架，用于多源异构公共舆情的统一分析

**💡 创新点**

四大创新：1）跨平台协同注意力模块对齐语义并保留源特性；2）分层自适应融合机制根据数据质量动态加权；3）共享潜在空间的联合多任务学习；4）多模态提取方案集成OCR、ASR与视觉情绪分析

**🔧 技术方法**

技术手段包括TF-IDF传统特征、Pangu-7B深度嵌入、跨平台注意力、门控融合、层归一化+ReLU的分层优化、跨模态注意力、联合损失（聚类KL、焦点损失、一致性正则）以及多模态概率模型

**📊 数据集**

在三个多源中文数据集上验证：Weibo-12（58k条，12平台），CrossPlatform-15（72k条，15平台），NewsForum-8（35k条，8平台）

**📈 对比分析**

与传统TF-IDF+聚类、单源BERT、早/晚融合、AttentionFusion、ChatGPT-4、Pangu-7B等基线对比，CRAF在主题聚类ARI上提升至0.76（比最佳基线高4.1%），情感F1提升至0.84（高3.8%），并在新平台少样本适配上仅需50样本即可达到与BERT微调相同的表现

**⚠️ 局限性**

局限性包括：多模态深度融合不足、需周期性再训练以对抗概念漂移、缺乏可解释性、边缘设备部署的算力瓶颈、主要验证于中文平台，跨语言适配待进一步研究

---

## 28. R$^2$Energy: A Large-Scale Benchmark for Robust Renewable Energy Forecasting under Diverse and Extreme Conditions

**arXiv ID:** 2602.15961 | [PDF](https://arxiv.org/pdf/2602.15961v1)

**作者:** Zhi Sheng `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**通讯引用:** 31839 | [OpenAlex ID](https://openalex.org/A5029768249)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了R^2Energy大规模基准，评估NWP辅助下的风光发电预测鲁棒性。

**💡 创新点**

创新点在于引入极端天气标注、Qualification Rate指标和标准化无泄漏协议，揭示鲁棒性与模型复杂度的权衡关系。

**🔧 技术方法**

采用深度学习（RNN/GRU、CNN、Transformer）、物理基线和统计方法，并结合多尺度评估框架。

**📊 数据集**

使用10.7 M小时分辨率、902台风光站点的气象与发电记录，覆盖中国四个气候区。

**📈 对比分析**

通过对16种模型在三类预测时段（USTF/ STF/ MTF）下进行MAE、RMSE、Q、S等指标比较，发现AR/GRU在极端天气下表现最佳，Transformer在高噪声场景易失效。

**⚠️ 局限性**

局限在于仅使用ERA5作为“完美”NWP，未考虑真实NWP误差；极端事件标签基于CMA分类，细粒度不足；缺少跨国或多能源类型验证。

---

## 29. A Koopman-Bayesian Framework for High-Fidelity, Perceptually Optimized Haptic Surgical Simulation

**arXiv ID:** 2602.15834 | [PDF](https://arxiv.org/pdf/2602.15834v1)

**作者:** Rohit Kaushik `[一作]` (Hanson Professional Services), Eva Kaushik `[通讯]` (University of Tennessee)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5016394768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一个统一的 Koopman-Bayesian 框架，用于高保真、感知优化的外科手术仿真 haptic 渲染。

**💡 创新点**

创新点在于将非线性工具-组织相互作用映射到 Koopman 嵌入空间以实现线性预测与控制，并结合贝叶斯心理物理模型根据个体 JND 进行力反馈自适应校准。

**🔧 技术方法**

核心技术包括 Koopman 嵌入（EDMD）、线性预测控制、贝叶斯层级心理物理模型（Stevens 与 Weber 规律）、低通滤波、延迟补偿和统计混合效应模型。

**📊 数据集**

使用基于医学影像的软组织（肝、肾）与骨骼（股骨）模型以及 Phantom Omni 与自定义 6-DOF 力反馈操作器的数据集，并在 MATLAB/Simulink 与 C++/ROS 环境下进行 900 次模拟试验和 12 人试点。

**📈 对比分析**

通过与传统弹簧阻尼与能量基渲染方法对比，系统平均渲染延迟为 4.3 ms，力误差 < 2.8%，感知判别率提升 20%，多元方差分析与混合效应模型显示显著优于基线（p < 0.01）。

**⚠️ 局限性**

主要局限包括对不同病理组织需要在线重新训练 Koopman 运算符、计算量大导致高阶 FEM 仍需加速、贝叶斯心理物理假设（高斯噪声、固定 Weber 比）不完全捕捉个体差异、以及仅有 12 人的初步人体试验，尚需更大样本的临床验证。

---

## 30. VGGT-based online 3D semantic SLAM for indoor scene understanding and navigation

**arXiv ID:** 2602.15899 | [PDF](https://arxiv.org/pdf/2602.15899v1)

**作者:** Anna Gelencsér-Horváth `[一作]`, Kristóf Karacs `[通讯]` (Pázmány Péter Catholic University)

**通讯引用:** 255 | [OpenAlex ID](https://openalex.org/A5051277218)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于VGGT的在线3D语义SLAM框架，利用滑动窗口和相机姿态对齐实现对长视频流的稀疏语义地图构建，并通过地面平面投影支持助理导航；

**💡 创新点**

创新点包括：①滑动窗口+姿态对齐实现内存高效、实时的SLAM；②将2D实例分割通过VGGT跟踪头提升到3D并保持持久实例身份，支持变化检测；③使用LiDAR尺度校正消除长时漂移；④通过地面平面投影压缩为2D地图，提升导航效率；⑤实现了对象的增、删、移动态更新。

**🔧 技术方法**

使用了Visual Geometry Grounded Transformer (VGGT)、滑动窗口管线、相机姿态图优化、LiDAR深度融合、实例分割+跟踪、RANSAC平面估计、点云投影、前沿探索等技术。

**📊 数据集**

评估数据集包括7-Scenes（姿态与重建），ScanNet++（语义与实例一致性），以及自制的室内序列（办公室、住宅）进行定性验证。

**📈 对比分析**

与Infinite VGGT、IncVGGT、StreamVGGT等方法在7-Scenes上对比，RMSE ATE、重建精度指标接近或优于现有方案；GPU峰值内存保持在17GB，速度约3.57fps；在ScanNet++上ID一致率达90–95%；在助理导航任务中实现实时前沿探索和座位搜索，显示实用性。

**⚠️ 局限性**

局限性包括：未建模连续物体运动或完全动态场景；变化检测仅针对已分割实例，未覆盖未分割区域；LiDAR与相机同步误差可能影响尺度估计；在强遮挡下跟踪头可能失效。

---

## 31. LLM-Driven Intent-Based Privacy-Aware Orchestration Across the Cloud-Edge Continuum

**arXiv ID:** 2602.16100 | [PDF](https://arxiv.org/pdf/2602.16100v1)

**作者:** Zijie Su `[一作]` (University of Melbourne), Adel N. Toosi `[通讯]` (University of Melbourne)

**通讯引用:** 4848 | [OpenAlex ID](https://openalex.org/A5083902835)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于大语言模型的意图驱动隐私保护编排框架，能够将自然语言隐私意图自动转化为 Kubernetes 节点选择规则和 ONOS 网络流规则，实现云边计算连续体的跨层合规性。

**💡 创新点**

创新点在于首次将 LLM 与意图驱动管理深度结合，既支持计算层的 pod 调度，又支持网络层的路径规划，同时提供统一的验证管道和 90 条多场景隐私意图基准数据集，显著降低了专家手工配置的工作量。

**🔧 技术方法**

采用 GPT‑4o 作为核心推理模型，结合 Kubernetes、ONOS、Mininet 以及 OpenAI API，利用提示工程和结构化输出实现安全策略的自动生成与执行。

**📊 数据集**

使用了自建的 90 条自然语言隐私意图数据集，涵盖计算、网络与混合三大域，并包含简单与复杂两种难度等级，配合 Kubernetes+ONOS 的实验平台进行验证。

**📈 对比分析**

与 Claude‑3.5 Haiku、DeepSeek‑V3 等模型对比，GPT‑4o 在 90 条意图上的成功率达 95.6%，平均端到端时延约 21 秒，验证步骤平均 3.7 条，展示了显著优于其他模型的准确性与效率。

**⚠️ 局限性**

主要局限包括对标签一致性和控制平面安全性的高度依赖、在高关键性环境下 95% 成功率仍需人工复核、以及对多租户生产环境的泛化性不足。

---

## 32. Graphon Mean-Field Subsampling for Cooperative Heterogeneous Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.16196 | [PDF](https://arxiv.org/pdf/2602.16196v1)

**作者:** Emile Anand `[一作]` (Georgia Institute of Technology), Adam Wierman `[通讯]` (California Institute of Technology)

**通讯引用:** 10080 | [OpenAlex ID](https://openalex.org/A5062565732)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出图谱均值场子采样框架（GMFS），以仅利用少量邻居（κ≪n）近似大规模异质多智能体系统的全局均值场，从而实现可扩展的合作强化学习。

**💡 创新点**

创新点在于将图谱加权子采样与均值场理论结合，证明其Bellman算子仍为收缩映射，并给出O(1/√κ)的最优性误差上界，显著降低了样本复杂度和计算开销。

**🔧 技术方法**

技术手段包括图谱理论、均值场近似、Bellman收缩分析、子采样误差估计、集中式训练/分散式执行以及对齐的离散/连续状态空间方法。

**📊 数据集**

实验使用自定义的机器人仓储协作和交通协调仿真环境中的模拟数据，验证框架在这些典型异质场景中的有效性。

**📈 对比分析**

与全局均值场和传统均值场方法对比，实验结果表明随κ增大，平均折扣回报单调提升，趋近最优且计算速度显著加快；在机器人协作任务中实现了近乎最优的性能。

**⚠️ 局限性**

局限性包括：依赖生成器或离线数据，尚未在纯在线环境下处理探索-利用平衡；图谱参数需要预先估计或给定；目前仅针对协作任务，未考虑竞争或混合环境。

---

## 33. Emotion Collider: Dual Hyperbolic Mirror Manifolds for Sentiment Recovery via Anti Emotion Reflection

**arXiv ID:** 2602.16161 | [PDF](https://arxiv.org/pdf/2602.16161v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11901 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了双重Poincaré球超图框架Emotion Collider (EC-Net)，用于多模态情感分析，支持缺失/噪声模态下的鲁棒融合与重构。

**💡 创新点**

创新点在于将超图高阶交互与双球超平面几何相结合，采用径向-角向对比学习与可学习镜像层实现跨模态高维对齐，并加入属性嵌入和正交约束提升重构与判别性能。

**🔧 技术方法**

使用Poincaré球嵌入、超图神经网络、可微镜像层、对比学习、属性路径与正交解耦、SetTransformer融合、Riemannian Adam优化。

**📊 数据集**

在CMU-MOSI、CMU-MOSEI和IEMOCAP三大公开多模态情绪/情感数据集上进行实验。

**📈 对比分析**

相较于最新基线方法，EC-Net在Acc2/Acc7/F1/MAE/Pearson等指标上均优于同类模型，尤其在缺失模态、噪声扰动和高缺失率场景下表现出显著提升，平均提升约3–4个百分点。

**⚠️ 局限性**

局限性包括对曲率比例的敏感性、对大规模数据的计算开销，以及在极端缺失率或不匹配模态分布时仍可能出现性能下降。

---

## 34. Understand Then Memory: A Cognitive Gist-Driven RAG Framework with Global Semantic Diffusion

**arXiv ID:** 2602.15895 | [PDF](https://arxiv.org/pdf/2602.15895v1)

**作者:** Pengcheng Zhou `[一作]`, Chun Yu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CogitoRAG框架，模拟人类情景记忆流程，先将非结构化文本压缩为语义要点（gist memory），再构建多维知识图谱；在线检索时使用查询分解、实体扩散与CogniRank融合重排，将检索到的文档与其高密度记忆配对提供给生成器；

**💡 创新点**

创新点在于：①通过语义要点抽取保留文本隐含逻辑与实体关系；②构建含实体、事实、记忆与来源节点的多维知识图谱；③在检索阶段引入查询分解与实体扩散机制，模拟人类重要性判断；④CogniRank将扩散得分与语义相似度融合实现全局上下文感知的重排；

**🔧 技术方法**

技术包括：大语言模型用于语义要点抽取与推理；知识抽取模型提取实体及关系三元组；向量编码统一表示实体、记忆、事实、段落与查询；随机游走扩散算法实现实体与段落关联；min–max标准化与单参数融合实现重排；

**📊 数据集**

使用5个主流QA基准（NQ、PopQA、MuSiQue、2WikiMultihopQA、HotpotQA）和GraphBench（Novel/Medical）进行评测；

**📈 对比分析**

与9个基线（None、NV‑Embed‑v2、GraphRAG、LightRAG、RAPTOR、HippoRAG、HippoRAG2、ComoRAG、ToG2）在EM、F1、检索时间等指标上对比，CogitoRAG在所有QA数据集EM最高（NQ 51.3%、PopQA 50.94%、MuSiQue 43.2%、2Wiki 69.9%、Hotpot 76.2%），在GraphBench多任务ACC也均优于对手；

**⚠️ 局限性**

局限包括：语义要点抽取依赖预训练LM的推理能力，对高专业化或歧义文本的把握有限；实体扩散在大规模知识图上计算成本高，需进一步轻量化；gist memory构建为离线静态过程，无法实时更新或动态学习新知识。

---

## 35. NLP Privacy Risk Identification in Social Media (NLP-PRISM): A Survey

**arXiv ID:** 2602.15866 | [PDF](https://arxiv.org/pdf/2602.15866v1)

**作者:** Dhiman Goswami `[一作]` (George Mason University), Sanchari Das `[通讯]` (George Mason University)

**通讯引用:** 4226 | [OpenAlex ID](https://openalex.org/A5003726306)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对社交媒体文本的自然语言处理任务中的隐私风险进行系统综述，提出六维度的NLP-PRISM框架并在Transformer模型上评估隐私泄露与性能权衡；

**💡 创新点**

创新点在于构建专门针对社交媒体NLP的隐私风险评估框架、量化六类任务的风险分布，并首次在同一框架下对多模型隐私攻击结果进行对比；

**🔧 技术方法**

采用隐私保护微调技术（命名实体遮蔽、文本扰动噪声）、成员推断（MIA）与属性推断（AIA）攻击评估，并使用XLM‑R、GPT‑2、FLAN‑T5等Transformer；

**📊 数据集**

使用Sentiment140、MELD、OLID、Dravidian CodeMix、TOEFL11、VarDial等多任务数据集进行实验；

**📈 对比分析**

实验表明隐私保护微调导致F1下降1%–23%，不同任务的影响程度差异明显，攻击成功率最高（MIA AUC 0.81，AIA准确率0.75）；

**⚠️ 局限性**

局限性包括仅覆盖同行评审论文、缺乏对预印本与新兴生成模型的评估、在语言识别任务中难以兼顾隐私与性能、以及对不同攻击情景的全面性不足。

---

## 36. Convergence rates of random-order best-response dynamics in public good games on networks

**arXiv ID:** 2602.15986 | [PDF](https://arxiv.org/pdf/2602.15986v1)

**作者:** Wojciech Misiak `[一作]` (University of Warsaw), Marcin Dziubiński `[通讯]` (University of Warsaw)

**通讯引用:** 386 | [OpenAlex ID](https://openalex.org/A5058148440)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在网络游戏中随机顺序最优响应动力学的收敛速度，并揭示了导致收敛慢的关键现象。

**💡 创新点**

创新点在于将稳定性阈值与图谱谱结构结合，提出了完整的稳定均衡判定条件，并系统性地描述了“重排”现象及其对收敛时间的影响。

**🔧 技术方法**

使用了谱理论、随机过程（马尔可夫链）分析、数值模拟与图论结构特征提取等技术手段。

**📊 数据集**

主要使用了合成图数据集，包括路径图、星图、环图、完全图、二分图、随机正则图、Erdős–Rényi图和Barabási–Albert图。

**📈 对比分析**

通过在多种图模型上绘制收敛时间曲线，发现收敛时间随外部性参数 δ 的变化呈现峰值；在阈值附近收敛时间可指数级放大，验证了理论预测。

**⚠️ 局限性**

局限性包括：只考虑线性最优响应的策略替代游戏；对复杂无结构图的解析仍有限；重排现象的理论上限虽已证明但实用性受限；结果对其他动力学或更一般的收益函数可能不直接适用。

---

## 37. BamaER: A Behavior-Aware Memory-Augmented Model for Exercise Recommendation

**arXiv ID:** 2602.15879 | [PDF](https://arxiv.org/pdf/2602.15879v1)

**作者:** Qing Yang `[一作]` (Guilin University of Electronic Technology), Jingwei Zhang `[通讯]` (Guilin University of Electronic Technology)

**通讯引用:** 8450 | [OpenAlex ID](https://openalex.org/A5100434265)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于行为感知与记忆增强的练习推荐框架 BamaER，实现学生学习进度预测、知识掌握预测与多样性过滤的端到端流程。

**💡 创新点**

创新点包括：① 三向混合编码捕获学生行为特征；② 动态记忆矩阵与自注意力相结合的知识追踪模块，兼顾长期积累与近期变化；③ 利用hippopotamus优化算法对候选练习进行多样性优化，提升推荐覆盖率。

**🔧 技术方法**

技术方法涵盖深度神经网络（层归一化、三向混合编码、双向注意力）、记忆网络（动态记忆矩阵）、行为特征融合、hippopotamus优化算法以及传统的Softmax、Sigmoid等激活函数。

**📊 数据集**

实验使用五个真实教育数据集：ASSISTments 2009、2012、2017、Algebra2005 与 Bridge2006。

**📈 对比分析**

与 FM、NCF、SASRec、DRER、MMER、LSTMCQP、KCP-ER、MulOER-SAN、ER-TGA 等八个基线模型在 Accuracy、Novelty、Diversity 三指标上对比，BamaER 在大多数指标上均取得最高或第二高的分数，显著优于传统方法。

**⚠️ 局限性**

局限性包括：对新用户冷启动时需要较多历史交互数据；模型复杂度高，训练成本较大；对候选集构建与参数设置（如推荐数量、HO 种群大小）较为敏感；未利用知识图谱或语义关联进一步提升推荐质量。

---

## 38. Optimization of an Augmented R-CUBE mechanism for Cervical Surgery

**arXiv ID:** 2602.15886 | [PDF](https://arxiv.org/pdf/2602.15886v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 39. Understanding LLM Failures: A Multi-Tape Turing Machine Analysis of Systematic Errors in Language Model Reasoning

**arXiv ID:** 2602.15868 | [PDF](https://arxiv.org/pdf/2602.15868v1)

**作者:** Magnus Boman `[一作]` (Karolinska Institutet), Magnus Boman `[通讯]` (Karolinska Institutet)

**通讯引用:** 2106 | [OpenAlex ID](https://openalex.org/A5021753785)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将大型语言模型推理流程形式化为多磁带图灵机，阐释各阶段（分词、前向计算、解码、反分词）如何导致系统性错误。

**💡 创新点**

创新点在于：①为LLM提供可验证的离散计算模型；②通过模型定位错误源，揭示分词隐蔽、缺乏显式计数子程序等根本原因；③将链式思维提示解释为输出磁带的外部化计算。

**🔧 技术方法**

使用技术主要是多磁带确定性图灵机的构造，定义七条磁带（字符、token、词典、模型参数、工作区、logits、字符输出），并将Transformer的注意力与前向层映射为磁带操作。

**📊 数据集**

未使用标准公开数据集，文章以“Strawberry计数”和“中心嵌套句子”作为理论案例进行演示。

**📈 对比分析**

比较方法为将LLM实际推理流程与图灵机模型的状态转移进行对齐，说明在同一输入下模型为何在某些阶段失效；未给出数值性能指标，而是提供理论上定位错误的说明。

**⚠️ 局限性**

限制在于：①模型假设确定性贪婪推理，未覆盖采样、beam搜索等实际策略；②忽略并行执行、有限精度等硬件细节；③无法捕捉模型内部连续神经动态，仅通过离散状态捕捉概念层面；④缺乏对更大规模LLM的实证验证。

---

## 40. Temporal Panel Selection in Ongoing Citizens' Assemblies

**arXiv ID:** 2602.16194 | [PDF](https://arxiv.org/pdf/2602.16194v1)

**作者:** Yusuf Hakan Kalayci `[一作]`, Evi Micha `[通讯]` (University of Southern California)

**通讯引用:** 120 | [OpenAlex ID](https://openalex.org/A5057226813)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了针对永久公民议会的时间排序框架，并设计了三种算法，分别在面板级、前缀级和全局级别提供近似比例代表性与个体公平性保证。

**💡 创新点**

创新点在于将度量空间下的比例代表性概念扩展到连续面板序列，构建层次化分组与链式覆盖机制，实现了指数级近似与常数因子近似的平衡，并首次证明在面板序列中可保持个体公平性的同时满足前缀比例代表性。

**🔧 技术方法**

采用了度量空间下的比例公平聚类（α‑PFC）、比例代表公平（β‑PRF）理论，以及公平贪婪捕获、度量扩展批准、修改版贪婪捕获等算法，结合层次分组与链式构造实现代表性保证。

**📊 数据集**

论文主要基于理论分析与合成实验，无使用真实数据集；若有实验则使用人工生成的欧氏空间点集模拟公民特征。

**📈 对比分析**

通过数学证明与实验验证，三种算法分别在全局面板达到O(6)-PRF、单面板达到O(26)-PRF，前缀级别在面板数ℓ处实现O(4^ℓ)‑PFC，另一个算法在前缀级别实现常数因子PFC，同时保持个体公平性；实验显示近似因子随面板数指数增长或常数保持。

**⚠️ 局限性**

局限性包括：对面板规模和数量需预先知晓；前缀级别的常数因子近似仅针对PFC而非更强的PRF；实现指数级近似的算法在面板数增大时计算成本和误差积累显著；未探讨任意子序列面板的代表性。

---

## 41. Bit-Width-Aware Design Environment for Few-Shot Learning on Edge AI Hardware

**arXiv ID:** 2602.16024 | [PDF](https://arxiv.org/pdf/2602.16024v1)

**作者:** R. Kanda `[一作]` (Tohoku University), T. Hanyu `[通讯]` (Tohoku University)

**通讯引用:** 5484 | [OpenAlex ID](https://openalex.org/A5062434040)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在PYNQ‑Z1 FPGA上实现了实时few‑shot学习，采用FINN框架实现任意位宽量化，并通过自定义Transpose与GAP转换节点优化实现高吞吐量和低延迟；

**💡 创新点**

引入FINN实现任意位宽量化、优化Transpose与Reduce‑Mean→GAP转换，显著降低资源占用并将推理速度提升至传统Tensil实现的两倍；

**🔧 技术方法**

使用FINN框架、Brevitas量化、ONNX转换、FPGA HLS/RTL流水线处理、DRAM/BRAM存储及自定义节点转换技术；

**📊 数据集**

ResNet‑9在MiniImageNet上训练，用CIFAR‑10（5‑way 5‑shot）进行准确率与推理性能评估；

**📈 对比分析**

与传统PEFSL+Tensil实现对比，FINN实现16位时保持62.78%准确率，6/4位量化保持59.7%；延迟16.3 ms、吞吐61.5 fps，DSP利用率低、LUT/FF/BRAM利用率高；

**⚠️ 局限性**

仅将backbone部署在FPGA，分类器等仍在CPU上，导致未完全释放FPGA性能；FINN在位宽灵活性优点下资源消耗更大。

---

## 42. Decomposing Large-Scale Ising Problems on FPGAs: A Hybrid Hardware Approach

**arXiv ID:** 2602.15985 | [PDF](https://arxiv.org/pdf/2602.15985v1)

**作者:** Ruihong Yin `[一作]` (University of Minnesota), Chris H. Kim `[通讯]` (University of Minnesota)

**通讯引用:** 7471 | [OpenAlex ID](https://openalex.org/A5043025421)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在FPGA上实现Ising机的子问题分解，结合28nm CMOS 50位耦合振荡器芯片，形成混合硬件架构以解决大规模3SAT问题

**💡 创新点**

将分解逻辑迁移至FPGA实现双层并行（空间并行+任务级流水线），并通过同板轻量级串行接口消除PCIe通信瓶颈，显著缩短子问题传输延迟

**🔧 技术方法**

使用BFS基分解策略、CSR存储、FPGA RTL实现的图遍历单元、并行平衡单元和子问题生成模块，配合COBI芯片快速能量最小化

**📊 数据集**

在SATLIB 3SAT基准集（uf20、uf50等）上评测

**📈 对比分析**

与在Intel i5-11500 CPU上运行的优化C+++PCIe方案对比，平均速度提升约1.93倍、能耗降低约168倍，通信延迟减少62%，COBI利用率从15%提升至30%

**⚠️ 局限性**

主要局限在FPGA资源和内存带宽；子问题生成仍占90%时间，且更大FPGA和更宽AXI总线能进一步提升性能

---

## 43. Federated Graph AGI for Cross-Border Insider Threat Intelligence in Government Financial Schemes

**arXiv ID:** 2602.16109 | [PDF](https://arxiv.org/pdf/2602.16109v1)

**作者:** Srikumar Nayak `[一作]` (Incedo Inc.), James Walmesley `[通讯]` (University of Kent)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 FedGraph-AGI，一个在跨境金融网络中通过联邦图神经网络结合 AGI 推理实现隐私保护的内部威胁检测框架。

**💡 创新点**

创新点在于首次将基于 Large Action Models 的因果推理与跨境异构图数据的 Mixture‑of‑Experts 聚合相结合，实现了跨国数据共享、图结构学习与多步推理的统一。

**🔧 技术方法**

技术方案包括联邦学习、图注意力网络（GAT）、MoE 聚合、LAM（大动作模型）因果推理、差分隐私与安全聚合。

**📊 数据集**

使用了一个合成的 10 个司法辖区、50,000 条交易、1,000 个节点的跨境金融交易数据集。

**📈 对比分析**

与中心化 GNN、FedAvg、FedProx、FedGNN 等基线对比，FedGraph-AGI 在 92.3% 准确率、0.956 AUC 等指标上分别比最佳联邦基线高 6.2% 和比中心化基线高约 6.7%，并在加速收敛和低误报率方面表现优异。

**⚠️ 局限性**

局限性包括仅在合成数据上验证、AGI 推理耗时较高、假设客户端可信且易受拜占庭攻击、可能存在公平性偏差且对实时大规模交易的可扩展性尚待进一步评估。

---

## 44. Pluralism in AI Governance: Toward Sociotechnical Alignment and Normative Coherence

**arXiv ID:** 2602.15881 | [PDF](https://arxiv.org/pdf/2602.15881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 45. State Design Matters: How Representations Shape Dynamic Reasoning in Large Language Models

**arXiv ID:** 2602.15858 | [PDF](https://arxiv.org/pdf/2602.15858v1)

**作者:** Annie Wong `[一作]` (Leiden University), Anna V. Kononova `[通讯]` (Leiden University)

**通讯引用:** 12845 | [OpenAlex ID](https://openalex.org/A5016332970)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究在动态多步推理任务中，LLM 在推理时如何表示状态（历史粒度、结构与空间定位）对决策性能的影响。

**💡 创新点**

首次系统评估三大维度（长短历史、自然语言 vs. 结构化编码、文本 vs. 图像/图示）在多模型、多任务中的效果，发现摘要能显著降低噪声、提升长时序推理；结构化编码仅对具备代码/结构化输出经验的模型有效；图示（VoT）比直接图像更常提升性能，且提升主要来自构建过程而非信息量本身。

**🔧 技术方法**

使用冻结参数的预训练LLM进行推理时态提示；对历史进行自动摘要；提供多种结构化状态编码（字典、矩阵、列表等）；利用视觉语言模型接收图像；采用 Visualization‑of‑Thought 生成 ASCII 地图；对比 Oracle‑Summary 与 Oracle‑VoT 以分离信息缺失与生成质量。

**📊 数据集**

实验数据集包括 Tower of Hanoi、Messenger (SmartPlay) 与 BabyAI（Goto、Open、Pickup、PickUpSeqGoTo、PutNext）等。

**📈 对比分析**

通过 0–1 归一化得分与标准差，对 Long‑Form 与 Summary、自然语言与结构化、文本 vs. Vision/VoT 进行对比；结果显示：摘要在多数模型/任务上提升 10–30%；结构化编码在部分模型（如 DeepSeek‑R1‑14B）可提升 10%+；Vision 多为无效或负面影响，VoT 在大型模型上可将 0.6‑0.8 提升至 1.0；Oracle‑Summary/VoT 显示信息本身并非瓶颈，关键是模型能否可靠构建/解析。

**⚠️ 局限性**

局限性：摘要与 VoT 依赖模型的推理与生成能力，易出现误差累积；结构化编码需模型具备解析 JSON 等格式的能力，否则导致性能下降；Vision 模型常出现视觉忽视，导致信息利用率低；所有实验均在仿真环境下完成，真实物理环境的持续状态追踪与感知误差仍是待解决的问题。

---

## 46. Gated Tree Cross-attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs

**arXiv ID:** 2602.15846 | [PDF](https://arxiv.org/pdf/2602.15846v1)

**作者:** Xinyu Gao `[一作]` (Zhejiang University), Nai Ding `[通讯]` (Zhejiang University)

**通讯引用:** 7151 | [OpenAlex ID](https://openalex.org/A5008847016)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在保持预训练 LLM 背骨不变的前提下，使用预先解析的成分句法树和门控树形跨注意力分支，对 decoder-only LLM 进行可检查、可恢复的结构注入，以提升其句法稳健性。

**💡 创新点**

创新点包括：① 检查点兼容的门控树跨注意力（GTCA）分支，② 通过 token 更新掩码和三阶段训练调节结构注入的时间和强度，③ 利用离线预计算的句法树缓存避免运行时解析，④ 结合头级门控实现可解释的结构利用。

**🔧 技术方法**

技术主要包括 Transformer 结构、门控跨注意力、头级门控、预先解析的成分句法树、三阶段 LoRA 训练（任务适配、结构专化、联合细化）以及 UUAS 句法探针。

**📊 数据集**

使用的主要数据集为多选题推理（CLOTH、MMLU）、句法检验（BLiMP、CoLA）、常识推理（HellaSwag、Winogrande），以及 Penn Treebank 进行句法探针。

**📈 对比分析**

与基线（Backbone、LoRA-only、Direct-Joint）对比，GTCA 在 Qwen-2.5-7B 和 Llama-3-8B 上的 BLiMP 准确率提升约 3–5 分，CoLA 恢复率提升约 3–4 分，且保持或略优于基线在多选题、常识推理上的性能，显示在保持广泛能力的同时显著提升句法稳健性。

**⚠️ 局限性**

局限性：依赖外部成分句法解析器，低质量或领域外解析可能导致噪声和性能下降；离线解析和缓存增加预处理、存储和推理开销；目前仅验证于英语多选题与句法测试，未覆盖 NLI、长文本生成或多轮交互等场景。

---

## 47. FLoPS: Semantics, Operations, and Properties of P3109 Floating-Point Representations in Lean

**arXiv ID:** 2602.15965 | [PDF](https://arxiv.org/pdf/2602.15965v1)

**作者:** Tung-Che Chang `[一作]` (Rutgers University), Santosh Nagarakatte `[通讯]` (Rutgers University)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

**🎯 论文内容**

在Lean 4理论证明器中构建并验证IEEE P3109低精度浮点标准的完整形式化模型；

**💡 创新点**

提出了三域等价（bit‑level、代数ADT、语义域）的三角等同证明，并在此框架下对FastTwoSum和ExtractScalar等核心算法进行形式化验证，揭示P3109在饱和模式下FastTwoSum能准确计算溢出误差等新性质；

**🔧 技术方法**

使用Lean 4、数学库、定理证明技巧（如`refine`、`simp`、`lia`等）、定理类Faithful/Monotonic、对称域等手段；

**📊 数据集**

无数据集（本工作为形式化验证而非实验实现）；

**📈 对比分析**

通过构造性的等价证明和自动化证明脚本验证算法性质，未给出数值性能指标；

**⚠️ 局限性**

局限在于只覆盖P3109标准，未实现可执行代码，且对极低精度（P=1）等边界情况需要进一步完善。

---

## 48. Building Safe and Deployable Clinical Natural Language Processing under Temporal Leakage Constraints

**arXiv ID:** 2602.15852 | [PDF](https://arxiv.org/pdf/2602.15852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 49. The SLAM Confidence Trap

**arXiv ID:** 2602.15884 | [PDF](https://arxiv.org/pdf/2602.15884v1)

**作者:** Sebastian Sansoni `[一作]` (Instituto de Automática), Santiago Ramón Tosetti Sanz `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

分析并指出 SLAM 社区存在的“置信陷阱”，即过度关注离线几何精度而忽视实时不确定性估计，提出应将一致性与实时不确定性评估作为评价 SLAM 系统的核心。

**💡 创新点**

创新点在于从历史演进的视角系统梳理 SLAM 发展趋势，明确将评估指标从传统的 ATE/RPE 等外部误差转向内部一致性不确定性；并提出“置信陷阱”概念与逃脱路径，强调需要统一、可验证的协方差估计与概率推理。

**🔧 技术方法**

采用理论分析、概率不确定性推导、Gaussian Belief Propagation、蒙特卡洛 Dropout、深度集成等方法论，讨论如何在传统滤波、粒子滤波、图优化和深度学习感知中实现一致性不确定性估计。

**📊 数据集**

未进行实验验证，文中主要引用 TUM RGB‑D、KITTI 等标准 benchmark 以及农业数据集（Rosario Agricultural Dataset、Bonn Agricultural Dataset）作为讨论背景，未在本研究中直接使用。

**📈 对比分析**

由于缺乏新实验，本文以文献综述和理论对比为主，指出现有 SLAM 系统在真实环境中缺乏自我评估机制，鲁棒性与安全性远低于理想几何精度；因此呼吁建立以不确定性一致性为核心的新评价框架。

**⚠️ 局限性**

局限性：缺乏实验数据与实现细节，所提评估框架尚未公开实现，无法直接验证其实时性与可扩展性；此外，如何在多传感器融合与深度学习感知中统一协方差估计仍是开放问题。

---

## 50. Coverage Path Planning for Autonomous Sailboats in Inhomogeneous and Time-Varying Oceans: A Spatiotemporal Optimization Approach

**arXiv ID:** 2602.15901 | [PDF](https://arxiv.org/pdf/2602.15901v1)

**作者:** Yang An `[一作]` (Institute of Deep-sea Science and Engineering, Chinese Academy of Sciences), Zhengru Ren `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2291 | [OpenAlex ID](https://openalex.org/A5011149759)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对异质、时变海洋环境下自动帆船的时空覆盖路径规划框架。

**💡 创新点**

创新点在于结合空间形态约束与时间预测前瞻，利用蒙特卡罗树搜索实现对风流变化的自适应规划。

**🔧 技术方法**

核心技术包括拓扑形态约束、天气预报驱动的look‑ahead规划、基于16方向的可行动作集、以及MCTS与启发式评分。

**📊 数据集**

使用随机生成的风场和流场（多尺度噪声+高斯平滑）模拟数据，并与手工设计的bou­stro­p­hedon基线进行对比。

**📈 对比分析**

实验结果表明，K=1的look‑ahead策略比K=0和传统方法分别将覆盖时间降低约30%–40%，并在所有随机场景下保持可行覆盖，尽管冗余率略升。

**⚠️ 局限性**

局限包括对预报精度的敏感性、计算开销较大、仅针对单船规划、未考虑多船协作与实时观测误差。

---

## 51. A Curious Class of Adpositional Multiword Expressions in Korean

**arXiv ID:** 2602.16023 | [PDF](https://arxiv.org/pdf/2602.16023v1)

**作者:** Junghyun Min `[一作]` (Georgetown University), Nathan Schneider `[通讯]` (Georgetown University)

**通讯引用:** 7055 | [OpenAlex ID](https://openalex.org/A5014008069)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析了韩国的后置动词式多词介词（PVC），构建了包含14种PVC的初步列表并提出了与PARSEME框架兼容的注释规范。

**💡 创新点**

首次明确定义并系统化韩国PVC的结构与语义特征，填补了现有跨语言MWE框架对韩语多词介词研究的空白。

**🔧 技术方法**

采用正则表达式候选抽取、KoNLP++等形态学分析器以及人工验证相结合的技术流程。

**📊 数据集**

利用2024年5月更新的韩语维基百科语料库（约515k篇文章）进行挖掘与分析。

**📈 对比分析**

通过人工判断将PVC与轻动词构造及非MWE结构对比，使用语法合法性测试展示PVC的词汇化与限定性；未给出具体性能指标，但证明了其可替代性与固定性。

**⚠️ 局限性**

依赖形态分析器的准确性、未覆盖所有领域变体、缺乏大规模标注与评估，且PVC列表仅为初步、局部结果，缺乏跨文本多样性验证。

---

## 52. Computing Approximate Pareto Frontiers for Submodular Utility and Cost Tradeoffs

**arXiv ID:** 2602.15964 | [PDF](https://arxiv.org/pdf/2602.15964v1)

**作者:** Karan Vombatkere `[一作]` (Boston University), Evimaria Terzi `[通讯]` (Boston University)

**通讯引用:** 5212 | [OpenAlex ID](https://openalex.org/A5005972547)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

提出了用于求解子模最大化问题的近似 Pareto 前沿，能够同时考虑子模效用函数与成本函数的折衷；

**💡 创新点**

创新点在于定义了 (α1,α2)-近似 Pareto 前沿，给出多种成本模型（基数、背包、图直径）下的理论保证，并提出基于网格和网格无关的高效算法；

**🔧 技术方法**

采用子模贪心、分解网格、Pareto 剪枝、预算限制下的多起点贪心、基于球形集合的直径处理等技术；

**📊 数据集**

在团队组建、推荐系统和影响力最大化三个领域的数据集上实验，使用公开数据（如 Yelp、社交网络、技能集合等）；

**📈 对比分析**

与传统单目标优化、线性组合权重方法以及随机/基于度数/距离启发式等基线对比，实验表明新算法在前沿质量、前沿规模和运行时间方面均优于基线，尤其在大规模实例上速度快、前沿多样性好；

**⚠️ 局限性**

局限性包括：对一般成本函数的近似因子可能偏大；网格方法在高动态范围内需要更多计算；对非图直径的复杂成本模型的理论保证尚未完全覆盖。

---

## 53. Fast Online Learning with Gaussian Prior-Driven Hierarchical Unimodal Thompson Sampling

**arXiv ID:** 2602.15972 | [PDF](https://arxiv.org/pdf/2602.15972v1)

**作者:** Tianchi Zhao `[一作]` (Beijing University Of Civil Engineering And Architecture), Jinliang Li `[通讯]` (Tsinghua University)

**通讯引用:** 10449 | [OpenAlex ID](https://openalex.org/A5110488978)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于高斯先验的层次化聚类Thompson采样算法TSCG及其单峰改进版UTSCG，用于解决具有高斯奖励且按群集划分的多臂赌博机问题。

**💡 创新点**

创新点在于将聚类结构与单峰性相结合，形成两种层次化采样策略，并证明其相较于传统TSG具有更优的理论回报上界。

**🔧 技术方法**

采用高斯先验的Thompson采样、层次化聚类模型、单峰带宽分析及理论证明，并在MATLAB中进行仿真验证。

**📊 数据集**

使用3GPP毫米波通信参数生成的RSS样本以及基于S&P500和CSI300股票收益分布的模拟资产收益数据，同时构造20臂4群集的实验数据。

**📈 对比分析**

与TSG、UCB、TLP等基线算法在累计回报和最佳臂选择率上进行对比，实验表明TSCG和UTSCG在累计回报下降更快、最佳臂选择率更高，尤其UTSCG在单峰场景中优于TSCG。

**⚠️ 局限性**

局限在于假设群集已知且奖励方差已知，未考虑非高斯分布或未知方差的情况，也未对自适应群集或多目标任务进行扩展。

---

## 54. ASPEN: Spectral-Temporal Fusion for Cross-Subject Brain Decoding

**arXiv ID:** 2602.16147 | [PDF](https://arxiv.org/pdf/2602.16147v1)

**作者:** Megan Lee `[一作]` (Carnegie Mellon University), Kateryna Shapovalenko `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5115728563)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过研究跨受试者EEG基于BCI的泛化问题，提出了ASPEN双模架构，利用谱时域特征与时域特征的乘法融合实现跨模态的一致性检验。

**💡 创新点**

创新点在于：①引入乘法多模融合机制，强制谱时域两条流在特征上达成一致，从而显著抑制噪声和相位误差；②通过自适应权重动态平衡谱与时域贡献，满足不同BCI范式的需求。

**🔧 技术方法**

使用技术包括：STFT谱图预处理、CNN+SE网络提取时域与谱时域特征、EEGNet风格的时域分支、乘法融合、交叉熵/二元交叉熵损失、Grad‑CAM可视化分析。

**📊 数据集**

实验数据集涵盖六个公开基准：SSVEP（Wang2016、Lee2019）、P300（BI2014b、BNCI2014_009）、Motor Imagery（BNCI2014_001、Lee2019 MI）。

**📈 对比分析**

与EEGNet、EEGConformer、CTNet、TSformer‑SA、MultiDiffNet等基线在相同训练设置下比较，ASPEN在三组数据（Lee2019 SSVEP、BNCI2014 P300、Lee2019 MI）获得最佳或竞争性未见受试者准确率，提升幅度约3–8%。

**⚠️ 局限性**

局限性包括：仍需针对每个任务手动调优STFT参数，乘法融合可能过度依赖谱域强特征，未充分探索自监督预训练或可学习时频变换的潜在提升。

---

## 55. The human intention. A taxonomy attempt and its applications to robotics

**arXiv ID:** 2602.15963 | [PDF](https://arxiv.org/pdf/2602.15963v1)

**作者:** J. E. Domínguez-Vidal `[一作]` (Institut de Robòtica i Informàtica Industrial), Alberto Sanfeliu `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 7325 | [OpenAlex ID](https://openalex.org/A5041730641)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对人类意图的心理学定义进行综述，提出五种分类维度，并将其映射到机器人研究中，结合案例阐述多维意图对人机交互的影响。

**💡 创新点**

创新点在于首次将心理学中的意图概念系统化为可用于机器人领域的分类体系，并讨论各维度之间的关系与互补性。

**🔧 技术方法**

本文主要采用文献综述和理论分析方法，并未开发或使用具体算法技术。

**📊 数据集**

无。

**📈 对比分析**

无实验或性能比较。

**⚠️ 局限性**

该工作为理论性综述，缺乏实证验证；分类框架可能不完整，实际应用时需结合具体情境进一步细化。

---

## 56. CLAA: Cross-Layer Attention Aggregation for Accelerating LLM Prefill

**arXiv ID:** 2602.16054 | [PDF](https://arxiv.org/pdf/2602.16054v1)

**作者:** Bradley McDanel `[一作]` (Franklin and Marshall), Harshit Khaitan `[通讯]` (Meta Reality Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过引入答案驱动的Oracle框架，构建了LLM预填阶段的token重要性基准，并利用该基准诊断现有排名方法的层级不稳定问题，进而提出了跨层注意力聚合（CLAA）策略以提升预填加速效果。

**💡 创新点**

创新点在于：①提出了基于答案回溯的oracle，真正量化token重要性；②发现并针对层级波动的失效模式，通过跨层最大化聚合稳健地恢复重要性；③在不修改核心模型的前提下实现显著TTFT和准确率提升。

**🔧 技术方法**

核心技术包括：多层注意力分数聚合、Softmax+max/mean归一化、1D平均池化降噪，以及使用FlashAttention-2实现高效前向推理。

**📊 数据集**

实验使用Llama‑3.1‑8B、Llama‑3.2‑3B、Mistral‑Nemo‑12B三大模型，并评估LongBench、Needle‑in‑a‑Haystack、RULER三大长文本推理基准。

**📈 对比分析**

与GemFilter、FastKV、Speculative Prefill等基线相比，CLAA在10%–40% token保留率下平均提升≈1–3%准确率，TTFT下降最多39%，与oracle上限相差不到1%。

**⚠️ 局限性**

局限性包括：对动态重要性任务（如多文档摘要）效果有限；在多轮对话场景中单次预填排名可能失效；且对非常短prompt或高压缩率时仍存在精度损失。

---

## 57. Adaptive Illumination Control for Robot Perception

**arXiv ID:** 2602.15900 | [PDF](https://arxiv.org/pdf/2602.15900v1)

**作者:** Yash Turkar `[一作]` (University at Buffalo), Karthik Dantu `[通讯]` (University at Buffalo)

**通讯引用:** 1863 | [OpenAlex ID](https://openalex.org/A5032635242)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种闭环主动照明控制框架Lightning，用于提升机器人在低光或高动态范围场景下的视觉SLAM鲁棒性。

**💡 创新点**

创新点包括：1）构造Co‑Located Illumination Decomposition (CLID) 重照明网络，可从单一灯光强度生成完整光照空间图像；2）设计离线Optimal Intensity Schedule (OIS) 通过动态规划在合成图像上寻找最佳灯光时间序列；3）通过模仿学习将离线专家蒸馏成实时控制策略ILC，实现在线灯光强度调节。

**🔧 技术方法**

采用深度学习图像重照明网络、动态规划优化、行为克隆/模仿学习、基于特征匹配的匹配得分以及FLIR Blackfly S相机采集的原始RAW图像进行SLAM评估。

**📊 数据集**

使用MIT Multi‑Illumination Dataset预训练CLID，并自行采集室内静态场景与10条机器人轨迹，采集光照为25%步进，并记录0%与100%作为参考。

**📈 对比分析**

与固定0%和100%灯光基线进行对比，采用WRMSE、轨迹占比、平均光照强度等指标评估。实验表明ILC和OIS在所有序列中均显著降低WRMSE、提升轨迹完成比例，并在功耗上低于恒亮基线。

**⚠️ 局限性**

受CLID重照明精度限制，若场景光照分布或材质与训练集差异较大，重照明误差会影响OIS与ILC；仅控制灯光强度，未结合曝光；需RAW图像，非ISP后处理；对室外高动态范围等更广泛环境尚未验证。

---

## 58. Anatomy of Capability Emergence: Scale-Invariant Representation Collapse and Top-Down Reorganization in Neural Networks

**arXiv ID:** 2602.15997 | [PDF](https://arxiv.org/pdf/2602.15997v1)

**作者:** Jayadev Billa `[一作]` `[通讯]` (Unaffiliated researcher), Jayadev Billa (Unaffiliated researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对神经网络在训练过程中出现新能力（emergence）的几何演化进行了系统性比较，采用五个模型规模（405K–85M参数）、24个算法任务、120个已知出现事件以及200+检查点，测量五种几何指标（表示有效秩、Fisher有效秩、LLC、Hessian主特征值、梯度协方差秩），并检验这些几何度量在控制实验与自然语言模型（Pythia）之间的可迁移性；

**💡 创新点**

创新点包括①发现训练初期普适的表示坍塌并收敛到任务特定、规模不变的底部；②展示层级从输出向下的“自上而下”坍塌传播；③建立几何层级，表明表示几何先于行为出现，而LLC同步、Hessian滞后；④明确几何监测在细粒度预测中的边界条件，如任务-训练对齐与难易任务分裂；

**🔧 技术方法**

技术手段包括GPT‑2风格无注意力预归一化 Transformer、AdamW + cosine 学习率调度、密集检查点策略、准确率与对数概率两种表现度量、低阈值持续性出现判定、RankMe 计算、Fisher 归一化、SGLD 估计 LLC、Stochastic Lanczos 计算 Hessian、梯度协方差有效秩、交叉相关函数与预排序预测、Spearman 相关与一致率评估；

**📊 数据集**

数据集主要为自生成的算法任务（COPY、REV、CMP、PAR、ADD、MOD、SORT、MUL）各难度级别，共 1,000 诊断样本；Pythia 采用 Pile 预训练数据，并在三种模型（160M、410M、2.8B）上使用七个 NLP 诊断集（语法、语义、算术、ICL、事实、推理、通用）进行评估；

**📈 对比分析**

比较方法通过交叉相关函数确定几何指标与行为出现的时间先后； precursor rate 量化几何先于行为的比例；对 Pythia 进行同样的检查点计算并比较层级与排序。结果显示：表示几何在 86%（nano）至 100%（hard任务）前先行；LLC 同步；Hessian 仅 17% 先行；预测上几何指标只能捕捉粗糙的易难分层，细粒度预测（within‑class concordance 27%、swap test 26%）表现不佳；

**⚠️ 局限性**

局限性包括①仅覆盖 8 个相对简单的算法任务；②最大模型仅 85M 参数，未检验更大规模下的结论；③多任务训练导致指标高度相关，难以区分独立事件；④梯度协方差仅基于前 50K 参数，可能偏向嵌入层；⑤Pythia 的诊断样本较少，导致几何信号噪声大；⑥缺乏因果干预验证几何先行的机制性作用；

---

## 59. Detecting Deepfakes with Multivariate Soft Blending and CLIP-based Image-Text Alignment

**arXiv ID:** 2602.15903 | [PDF](https://arxiv.org/pdf/2602.15903v1)

**作者:** Jingwei Li `[一作]` (Zhejiang Gongshang University), Pengfei Wu `[通讯]` (Zhejiang Gongshang University)

**通讯引用:** 641 | [OpenAlex ID](https://openalex.org/A5030109120)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于 CLIP 的多模态图文对齐网络，用于检测面部深度伪造，并通过多变量软混合增强（MSBA）和多变量伪造强度估计（MFIE）模块提升模型泛化能力。

**💡 创新点**

创新点包括：1）将 CLIP 视觉-文本双模态对齐引入深度伪造检测，利用文本提示指导视觉特征提取；2）设计 MSBA 数据增强策略，随机混合多种伪造模式并生成软标签，逼迫网络学习多模式共性特征；3）构建 MFIE 模块，实现空间伪造强度预测和混合权重回归，作为辅助监督提升判别鲁棒性。

**🔧 技术方法**

技术包括：CLIP ViT 视觉编码器、文本编码器与多模态交互投影层、Transformer 视觉解码器、轻量级转置卷积解码器、双重监督损失（分类、语义相似度、强度回归、权重 KL 散度）。

**📊 数据集**

训练数据使用 FaceForensics++（FF++）C23 版本；跨域测试在 Celeb-DF、DFDC Preview、DFDC、DFD、DeeperForensics‑1.0 等五个公开数据集。

**📈 对比分析**

与 Xception、Face‑Xray、F³Net、SPSL、SRM、UCF、CORE 等十余种基线比较，MSBA‑CLIP 在 FF++ 内域实现 100% ACC/AUC，在跨域视频/帧级 AUC 上平均提升 3.27%，单数据集最高提升 9.73%（DFD），显示出显著的准确性与泛化优势。

**⚠️ 局限性**

主要局限：依赖大型预训练 VLM，参数量大、推理速度慢；对极低强度或极端后处理伪造仍存在一定误判；MSBA 生成的合成样本可能无法完全覆盖所有真实混合模式，需进一步提升样本多样性。

---

## 60. ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization

**arXiv ID:** 2602.15983 | [PDF](https://arxiv.org/pdf/2602.15983v1)

**作者:** Junbo Jacob Lian `[一作]` (Northwestern University), Chung-Piaw Teo `[通讯]` (National University of Singapore)

**通讯引用:** 6160 | [OpenAlex ID](https://openalex.org/A5026381568)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ReLoop框架，解决LLM生成优化代码时的静默失败问题，结合结构化生成与无监督行为验证；

**💡 创新点**

创新点在于将专家建模流程拆解为四阶段链式思维并使用求解器参数扰动检测缺失约束/目标，形成互补的生成与验证循环；

**🔧 技术方法**

采用LLM四阶段生成（Understand→Formalize→Synthesize→Verify）、L1执行验证、L2扰动行为验证、IIS诊断修复及回退机制；

**📊 数据集**

使用新发布的RetailOpt-190（190个多约束零售库存优化实例）与公开Benchmark MAMO-ComplexLP、IndustryOR进行评测；

**📈 对比分析**

与基础、SFT、RL模型的直接生成、CoT和ReLoop三配置对比，ReLoop在RetailOpt-190上实现100%可执行率、约31%严格准确率、约35%实用准确率，跨Benchmark提升约4–12个百分点；

**⚠️ 局限性**

局限包括SFT模型对CoT格式兼容性差、L2验证成本线性增长、约束提取依赖同一LLM，且对系数误差、等价误差与未覆盖问题结构仍无法完全修复。

---

## 61. SAM 3D Body: Robust Full-Body Human Mesh Recovery

**arXiv ID:** 2602.15989 | [PDF](https://arxiv.org/pdf/2602.15989v1)

**作者:** Xitong Yang `[一作]` (Meta Superintelligence Labs), Kris Kitani `[通讯]` (Meta Superintelligence Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种可提示的编码-解码架构，结合共享图像编码器与分离的身体和手部解码器，利用高质量、多样化的数据引擎和新的解耦参数化人体网格模型，构建端到端的全身人体网格恢复系统。

**💡 创新点**

核心创新点包括：① 提示可控的解码器设计，可通过 2D 关键点/遮罩等提示引导推断；② 将身体和手部解码器分离，解决不同分辨率和监督目标的冲突；③ 采用新的 3DB 网格模型，骨架姿态与体形解耦，提升可解释性和控制性；④ 构建规模达 700 万张图像、数百万 3D 注释的多源数据引擎，主动挖掘难样本并使用多视角、合成与多模态技术提升标注质量。

**🔧 技术方法**

技术方法包括 Transformer‑based 编码‑解码架构、prompt 机制、可学习的查询 token、三维关键点与姿态约束损失、手部解码与融合、姿态先验、视角与相机参数估计、基于 VLM 的数据挖掘与自动标注流程。

**📊 数据集**

使用的数据集：单视角（AIChallenger、MS COCO、MPII、3DPW、SA‑1B）、多视角（Ego‑Exo4D、Harmony4D、EgoHumans、InterHand2.6M、DexYCB、Goliath）、合成数据（Goliath Synthetic）以及通过数据引擎主动标注的 7M 图像。实验评估涵盖 5 大标准基准（3DPW、EMDB、RICH、COCO、LSPET）以及 5 个新域数据集（Ego‑Exo4D、Harmony4D、Goliath、Synthetic、SA1B‑Hard），手部评估使用 FreiHand。

**📈 对比分析**

相较于现有单视角 HMR 方法（CameraHMR、PromptHMR、NLF、SMPLerX、HMR2.0b 等）和视频基准（WHAM、TRAM、GENMO），在所有标准指标（MPJPE、PA‑MPJPE、PVE、PCK）上均取得领先或相当成绩；在新域数据集上，留一法模型表现优于基线，尤其在极难姿态与遮挡场景中显著提升；手部精度在 FreiHand 上与专门手部模型相当；人类偏好实验中，模型胜率最高，赢率可达 95%+。

**⚠️ 局限性**

潜在局限：① 需要大量高质量 3D 注释和多视角资源，构建成本高；② 模型规模较大（ViT‑H/DINOv3 编码器），推理速度与资源消耗可能受限；③ 对极端光照、复杂遮挡仍可能存在误差，且对非人类或多目标场景的适用性未充分验证。

---

## 62. MARVL: Multi-Stage Guidance for Robotic Manipulation via Vision-Language Models

**arXiv ID:** 2602.15872 | [PDF](https://arxiv.org/pdf/2602.15872v1)

**作者:** Xunlan Zhou `[一作]` (Nanjing University), De-chuan Zhan `[通讯]` (Nanjing University)

**通讯引用:** 4436 | [OpenAlex ID](https://openalex.org/A5073912249)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 MARVL——一种利用视觉‑语言模型（VLM）进行多阶段引导奖励的框架，旨在改进机器人强化学习中的奖励设计与稀疏奖励问题。

**💡 创新点**

创新点包括：① 场景‑视角解耦（Scene‑View Decomposition）使 VLM 对摄像角度不敏感；② 任务方向投影（Task Direction Projection）将高维嵌入投射到任务特定方向上，实现进度的单调性；③ 多阶段子任务拆解与置信度阈值裁剪（Confidence‑Thresholded Shaping），提升奖励对任务进度的感知和鲁棒性。

**🔧 技术方法**

核心技术：VLM 的轻量级微调、投影算子、阈值裁剪、子任务分解、离线 RL（SAC/TD3）等；并将这些奖励模块无缝集成到现有 RL 循环中。

**📊 数据集**

实验数据集：Meta‑World 机器人操控基准（8个稀疏奖励任务）和 Panda‑Gym（跨域验证），以及用 500 张多视角 Meta‑World 图像进行 VLM 微调。

**📈 对比分析**

与基线（SAC、LIV、FuRL、Relay）及 Oracle（手工设计的密集奖励）比较，MARVL 在 Meta‑World 所有任务上实现最高成功率、最快收敛速度，并在跨域 Panda‑Gym 与不同摄像头配置下保持强大性能；样本效率显著提升。

**⚠️ 局限性**

局限性：仅在仿真环境验证，尚未验证零样本真实世界部署；对更长时序、多物体任务的扩展仍需探索；中间子任务目标图像目前多为手工收集，自动生成方法尚未彻底验证。

---

## 63. Can Generative Artificial Intelligence Survive Data Contamination? Theoretical Guarantees under Contaminated Recursive Training

**arXiv ID:** 2602.16065 | [PDF](https://arxiv.org/pdf/2602.16065v1)

**作者:** Kevin Wang `[一作]`, Didong Li `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 18246 | [OpenAlex ID](https://openalex.org/A5105838811)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在递归训练过程中加入人工合成数据与真实数据混合，研究生成模型在数据污染下的收敛行为。

**💡 创新点**

首次给出在无分布假设下递归污染训练仍能收敛的理论证明，并揭示收敛率取决于基准率与真实数据比例的相位转移。

**🔧 技术方法**

采用核密度估计（KDE）、Wasserstein GAN（WGAN）以及扩散模型等生成器，并使用Wasserstein‑1距离、MMD等距离度量。

**📊 数据集**

在一维高斯混合分布、两峰混合 Gaussian、以及MNIST手写数字图像等数据集上进行实验。

**📈 对比分析**

通过与理论预期的收敛阶数对比，实验结果显示不同真实数据比例下的收敛速率与理论一致，证明模型在污染环境中仍可收敛。

**⚠️ 局限性**

局限在于未考虑非凸距离、KL/交叉熵等常用损失，未建模人类筛选与强化学习过程，并且高维下无法准确评估Wasserstein距离。

---

## 64. EmoTrack: An application to Facilitate User Reflection on Their Online Behaviours

**arXiv ID:** 2602.15839 | [PDF](https://arxiv.org/pdf/2602.15839v1)

**作者:** Ruiyong Zhang `[一作]` `[通讯]`, Ruiyong Zhang

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了 EmoTrack 多平台应用，帮助年轻人记录观看 YouTube 的情绪与行为，自动分类视频并生成可视化报告，支持自我反思。

**💡 创新点**

创新点在于将个人信息学与 ChatGPT 结合，实现自动视频分类与情绪影响可视化，首次在此类工具中提供从记录到反思的完整闭环，并通过 SUS 与访谈验证可用性。

**🔧 技术方法**

使用技术包括 Python（后端、API 调用、ChatGPT 接口）、Flutter（前端 UI）、Firebase（身份验证、Cloud Firestore、Cloud Storage、Hosting）、Google App Engine（部署）、YouTube Data API、Google Takeout（数据获取）以及 OpenAI GPT‑3.5‑turbo。

**📊 数据集**

数据集为 13 名英国大学生的 YouTube 观看历史（由 Google Takeout 导出）与实时情绪记录，此外对实验参与者进行 SUS 调查与访谈。

**📈 对比分析**

通过 SUS 得分 79.8（高于平均值 68），访谈显示用户认为报告有助于自我反思；但未对大规模性能进行量化对比，仅证明在小样本场景下可行且易用。

**⚠️ 局限性**

局限性包括样本量不足、仅聚焦 YouTube、需手动上传观看历史、依赖 ChatGPT 可能产生分类偏差、对长期使用与跨平台兼容性的评估不足。

---

## 65. Language Statistics and False Belief Reasoning: Evidence from 41 Open-Weight LMs

**arXiv ID:** 2602.16085 | [PDF](https://arxiv.org/pdf/2602.16085v1)

**作者:** Sean Trott `[一作]` (Rutgers University), Pamela D. Rivière `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对41个开源语言模型在虚假信念任务上的表现进行复制与扩展，检验这些模型是否对角色的知识状态敏感，并探索模型规模、训练规模与心理状态推理能力之间的关系。

**💡 创新点**

创新点在于使用大规模开源模型而非传统闭源模型进行系统性检验，同时利用模型行为生成并验证关于非真值动词导致的错误信念偏差的新假设，揭示模型与人类在不同操控（知识状态 vs. 知识线索）下的差异。

**🔧 技术方法**

主要技术包括：基于token化的Log Odds计算、线性混合效应模型与极大似然比检验、心理计量预测力（PPP）评估、AIC比较以及对参数规模与训练token数的回归分析。

**📊 数据集**

使用了原始研究的192条虚假信念情景（包含知识状态、知识线索、首项与最近项四种变体），并采集对应的人类响应数据做对照。

**📈 对比分析**

与人类的准确率（约82.7%）进行对比，发现约34%模型对知识状态表现出显著敏感性，最高准确率为74.5%（仍低于人类）。模型规模越大，准确率和心理计量预测力越高，但总体仍未能“解释掉”人类的知识状态效应。

**⚠️ 局限性**

主要限制包括：虚假信念任务对心理状态推理的构造效度可能不足、样本仅覆盖部分开源模型且缺乏闭源顶尖模型、模型可能通过“捷径”完成任务、以及对模型训练数据组成与架构的完整可解释性不足。

---

## 66. A2H: Agent-to-Human Protocol for AI Agent

**arXiv ID:** 2602.15831 | [PDF](https://arxiv.org/pdf/2602.15831v1)

**作者:** Zhiyuan Liang `[一作]` (China Telecom Research Institute), Yujun Cheng `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 169 | [OpenAlex ID](https://openalex.org/A5021584440)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Agent-to-Human（A2H）协议，整合人类为可发现、可调用的实体，实现 AI 代理与人类在多模态平台上的协作。

**💡 创新点**

创新点：① 通过 Human Card 将人类身份与专业能力注册到可解析的域名系统；② 设计 Formal Communication Schema，规范代理何时、为何以及如何请求人类帮助；③ 建立 Unified Messaging Abstraction，将复杂 JSON 转化为多平台友好交互，支持按钮、表单等原生 UI。

**🔧 技术方法**

技术手段：分布式 Human Card 注册与查询、语义标签匹配、决策阈值触发（Ambiguity、Criticality、Resource Exhaustion）、A2H‑JSON 规范、Slack/Teams/Email 渠道适配器、异步与同步通信模式。

**📊 数据集**

使用的数据集：论文未提供公开数据集；案例实验基于模拟 DevOps 环境，使用 GPT‑4 生成日志、Kubernetes 相关配置以及手工创建的 Human Card，模拟 Slack 通信。

**📈 对比分析**

对比方法：在同一 DevOps 场景下，比较基线 Chat‑based 代理（直接让人类参与聊天）与 A2H‑enabled 代理。结果显示：① 人类定位更精准、无需人工广播；② 通过 Clarification Trigger 减少歧义；③ UI 交互减少错误；④ 通过 Permission Gate 提升安全性；整体完成率提升，误操作率显著下降。

**⚠️ 局限性**

局限性：① 未在大规模真实环境中进行纵向评估，缺乏跨组织、跨平台的实测数据；② Human Card 的维护与更新依赖手动注册，可能导致信息滞后；③ 对于高并发、低延迟场景的性能尚未深入验证；④ 需要进一步研究安全与隐私保护（如端点加密、访问控制）等细节。

---

## 67. Improving Interactive In-Context Learning from Natural Language Feedback

**arXiv ID:** 2602.16066 | [PDF](https://arxiv.org/pdf/2602.16066v1)

**作者:** Martin Klissarov `[一作]` (Google DeepMind), Edward Grefenstette `[通讯]` (Google DeepMind)

**通讯引用:** 12333 | [OpenAlex ID](https://openalex.org/A5023508792)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建教师-学生的didactic互动框架，将语言反馈视为可学习的能力，提升LLM在多轮对话中的上下文学习与自我改进。

**💡 创新点**

创新地将信息不对称的教师反馈与RL^2F结合，使模型能够在对话中主动理解并整合自然语言反馈，进而实现自我校正。

**🔧 技术方法**

使用强化学习（RL^2F）、对话式教师-学生策略、POMDP建模以及近似无梯度内循环的元学习方法。

**📊 数据集**

采用 Omni-Math、HardMath2、ARC-AGI、Codeforces、BIG-Bench Extra Hard、LiveCodeBench、Poker、Wordle 等多种可验证任务与跨领域评测集。

**📈 对比分析**

与单轮RL、SFT 和基线模型对比，模型在多轮交互中准确率提升约10–15%，Gemini 2.5 Flash 通过此方法接近 Gemini 2.5 Pro，并在外域任务平均提升约5–7%。

**⚠️ 局限性**

仍缺乏对教师反馈生成的完整控制，可能出现自我强化偏见，且难以将短期适应转化为长期知识，对非可验证任务的通用性尚待验证。

---

## 68. A Theoretical Approach to Stablecoin Design via Price Windows

**arXiv ID:** 2602.15981 | [PDF](https://arxiv.org/pdf/2602.15981v1)

**作者:** Katherine Molinet `[一作]` (University of Edinburgh), Aris Filos-Ratsikas `[通讯]` (University of Edinburgh)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5065488177)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文通过理论模型分析了基于价格窗口的稳定币在缺乏二级稳定机制时的短期与长期稳定性，并证明仅靠价格窗口无法同时实现两者，提出交易费用与后备资产波动率的阈值关系；

**💡 创新点**

创新点在于定义并使用弱ε稳定性概念、引入敏感投机者的正式定义以及推导预留耗尽的阈值与期望耗尽时间的解析公式，展示价格窗口设计的固有短期与长期稳定权衡；

**🔧 技术方法**

主要技术包括形式化的价格窗口模型、极限上/下限理论证明、最优/近似投机者策略的构造、期望收益与耗尽时间的解析推导，以及基于独立同分布和历史价格的蒙特卡罗仿真；

**📊 数据集**

使用的实验数据集为2022-2025年比特币与以太坊每小时开盘价（约26000条记录）以及假设的正态分布和随机游走模型；

**📈 对比分析**

通过理论阈值与仿真结果对比，验证在无交易费用的情形下，投机者可在约10天内耗尽1000枚后备币；实测显示ETH的耗尽速度快于BTC，进一步验证波动率与耗尽速率正相关；

**⚠️ 局限性**

局限性包括：仅考虑单一投机者且无竞争市场、未加入二级稳定机制、交易费用假设固定、价格模型理想化，且未对多策略投机者或真实市场动态进行深入建模。

---

## 69. VDLM: Variable Diffusion LMs via Robust Latent-to-Text Rendering

**arXiv ID:** 2602.15870 | [PDF](https://arxiv.org/pdf/2602.15870v1)

**作者:** Shuhui Qu `[一作]` (Stanford University), Shuhui Qu `[通讯]` (Stanford University)

**通讯引用:** 969 | [OpenAlex ID](https://openalex.org/A5112694715)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种变量扩散语言模型（VDLM），将语义规划与文本渲染分离，使用连续变量嵌入进行扩散规划，并在嵌入空间进行后期强化学习，最终通过鲁棒的 Vec2Text 逆向器将规划结果转为文本。

**💡 创新点**

创新点包括：① 在变量级别对语义嵌入进行 LLaDA 风格的掩码扩散，从而大幅缩短规划长度；② 将奖励与价值函数直接作用于嵌入空间，避免 RL 循环中出现文本解码；③ 对 Vec2Text 进行噪声注入鲁棒训练，使其能在规划噪声下恢复准确文本。

**🔧 技术方法**

核心技术：LLaDA‑style 掩码扩散、TraceRL‑style 轨迹感知强化学习、嵌入逆向模型 Vec2Text、L2 噪声鲁棒训练、语义变量构造与投影。

**📊 数据集**

使用九个基准数据集：MMLU、BBH、ARC‑Challenge、TruthfulQA、PIQA、GSM8K、MATH、GPQA、HumanEval。

**📈 对比分析**

与 LLaDA、以及 LLaMA3、Qwen2、Qwen2.5、Mistral、DeepSeek、Gemma2 等 AR 8B/7B 模型在相同协议下对比；预训练阶段 VDLM 平均分 51.6，略高于 LLaDA 的 49.3；后期训练（SFT+RL）后，GSM8K 达 89.8、MATH 62.4、HumanEval 74.9，分别逼近或超越多 AR 7B/9B 模型，显示显著的长篇生成提升。

**⚠️ 局限性**

局限性：预训练阶段在代码与高精度数学任务上仍表现不佳，主要受限于逆向渲染的语法准确性；RL 在嵌入空间优化时需要强健的渲染器才能保持效果；鲁棒训练需要精细调节噪声水平，过大或过小都会导致性能下降。

---

## 70. Learning to Drive in New Cities Without Human Demonstrations

**arXiv ID:** 2602.15891 | [PDF](https://arxiv.org/pdf/2602.15891v1)

**作者:** Zilin Wang `[一作]` (University of Oxford), Shimon Whiteson `[通讯]` (University of Oxford)

**通讯引用:** 12739 | [OpenAlex ID](https://openalex.org/A5056879203)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出NOMAD框架，利用目标城市的地图和元信息通过自我对弈多智能体强化学习实现驾驶策略迁移，完全不需要目标城市的人类示范。

**💡 创新点**

创新点在于：①只使用地图与少量元信息即可完成跨城市迁移；②通过KL正则化将目标城市自我对弈与源城市行为先验结合，避免奖励过拟合并保持人类行为；③构建了无示范的Pareto前沿，兼顾成功率与行为真实度。

**🔧 技术方法**

使用技术包括：多智能体强化学习（IPPO）、自我对弈训练、KL正则化约束、基于地图的模拟器（GPUDrive/Nocturne/Waymax）、简易奖励函数以及场景生成器。

**📊 数据集**

实验基于nuPlan大型跨城市数据集，覆盖波士顿、旧金山、旧金山、旧金山、旧金山等城市，且在不同大陆间进行迁移验证。

**📈 对比分析**

与零射转移、仅行为克隆、带示范的RL等基线相比，NOMAD在闭环评估中将成功率从约42%提升至90%以上，现实性得分提升至约0.76~0.77，形成明显的Pareto前沿，且无需目标城市示范即可逼近专家上限。

**⚠️ 局限性**

局限性包括：依赖高质量地图与元信息，无法充分建模不同城市的交通文化和社会约定；KL正则权重需手动调参；在极端场景或复杂交互中仍可能出现性能下降。

---

## 71. Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities

**arXiv ID:** 2602.16093 | [PDF](https://arxiv.org/pdf/2602.16093v1)

**作者:** Shankar Padmanabhan `[一作]` (Cornell University), Tanya Goyal `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究后训练大语言模型的持续知识适配，提出通过拆分上下文的蒸馏方法来在不遗忘先前能力的前提下学习新知识。

**💡 创新点**

创新点在于将上下文蒸馏转化为在同一模型内部用富上下文条件作为教师、无上下文作为学生，直接最小化 KL，避免显式生成步骤并显著缓解灾难性遗忘。

**🔧 技术方法**

采用基于 KL 的上下文蒸馏训练框架，配合教师模型冻结、学生模型微调，并使用多句子文档拆分技术。

**📊 数据集**

使用两个文档级别更新数据集：KUP（模拟新闻更新）和 BioASQ（医学问答）作为适配语料。

**📈 对比分析**

与传统微调、LoRA、TALR、Rephrase、KL 正则化以及先前的上下文蒸馏 baseline 对比，DiSC 在维持指令遵循、推理和编码等能力下降不超过5分的同时，在两大适配任务上实现约5-12分的性能提升，整体表现优于所有基线。

**⚠️ 局限性**

局限在于方法主要针对文档级知识更新，未在极大模型或多领域持续学习场景中验证，对模型规模扩展和长文本推理的适用性尚需进一步研究。

---

## 72. A Lightweight Explainable Guardrail for Prompt Safety

**arXiv ID:** 2602.15853 | [PDF](https://arxiv.org/pdf/2602.15853v1)

**作者:** Md Asiful Islam `[一作]` (University of Arizona), Mihai Surdeanu `[通讯]` (University of Arizona)

**通讯引用:** 14088 | [OpenAlex ID](https://openalex.org/A5047699502)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种轻量化可解释的安全提示防护方法LEG，能同时判断提示是否不安全并给出解释词。

**💡 创新点**

引入多任务学习架构联合训练分类与解释器；使用对抗式合成数据克服确认偏差；提出结合交叉熵、焦点损失与不确定性加权的联合损失。

**🔧 技术方法**

多任务学习（MTL）共享Transformer编码器；合成解释生成策略；自定义联合损失（交叉熵+focal+uncertainty weighting）；基于DeBERTa-v3的轻量级模型。

**📊 数据集**

AEGIS2.0、WildGuardMix、Toxic-Chat0124三个提示安全数据集，并对词级标签做扩展。

**📈 对比分析**

与LLAMA Guard、WILDGURD、ToxicChat-T5-Large、OpenAI Moderation API、Llama Prompt Guard 2等基线及自定义Prompt/Word/LIME/SHAP基线比较；在域内域外均达到或超过SOTA；在词级解释上优于传统后置解释方法；模型参数小、推理速度快。

**⚠️ 局限性**

仅在英语数据集评估；多语言表现未验证；在某些数据集上性能略低，需进一步调优。

---

## 73. Scrutinizing Variables for Checkpoint Using Automatic Differentiation

**arXiv ID:** 2602.16010 | [PDF](https://arxiv.org/pdf/2602.16010v1)

**作者:** Xin Huang `[一作]` (Nanchang Hangkong University), Kento Sato `[通讯]` (R-CCS, RIKEN)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用自动微分技术识别高性能计算应用中需要检查点的变量内部哪些元素对输出无影响，从而只保存关键元素以减少检查点存储。

**💡 创新点**

创新点在于首次将自动微分用于细粒度元素级别的检查点优化，并通过可视化揭示关键/非关键元素分布及其与代码/算法的对应关系。

**🔧 技术方法**

采用 LLVM‑基的自动微分工具 Enzyme 进行梯度计算，构建自定义检查点库，并用可视化工具展示元素分布。

**📊 数据集**

评估数据集为 NPB（NAS Parallel Benchmarks）套件中的 8 个基准程序，覆盖多种典型 HPC 计算模式。

**📈 对比分析**

通过对比原始完整检查点与只保存关键元素的检查点，在 NPB 试验中平均可节省 13% 存储，某些基准最高可达 20%，同时验证恢复后结果正确。

**⚠️ 局限性**

局限性包括：仍需手动确定需要检查点的变量，方法依赖自动微分的准确性；未能在算法层面自动推断关键性；对编程缺陷敏感，若无缺陷则难以获益。

---

## 74. LGQ: Learning Discretization Geometry for Scalable and Stable Image Tokenization

**arXiv ID:** 2602.16086 | [PDF](https://arxiv.org/pdf/2602.16086v1)

**作者:** Idil Bilge Altun `[一作]` (Indiana University Bloomington), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 456 | [OpenAlex ID](https://openalex.org/A5070970331)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可学习的几何量化(LGQ)方法，用温度控制的软分配实现连续到离散的平滑转换；

**💡 创新点**

创新点在于将量化视为几何学习，使用可微软分配并加入峰度与全局利用正则化，避免传统VQ的代码坍塌；

**🔧 技术方法**

采用温度调度的Gibbs分配、straight‑through估计器、峰度正则化、全局使用正则化以及变分自由能框架；

**📊 数据集**

在ImageNet 128×128数据集上进行实验；

**📈 对比分析**

与VQ、FSQ、SimVQ、LFQ等方法比较，LGQ在rFID、SSIM、LPIPS等指标上表现最佳，同时仅使用约50%代码，实现更优的利用率–失真折衷；

**⚠️ 局限性**

局限性包括仅在图像任务上验证，缺乏对视频或多模态的实验；温度退火与正则化参数需人工调节，且对大规模训练的计算成本相对较高。

---

## 75. Verifier-Constrained Flow Expansion for Discovery Beyond the Data

**arXiv ID:** 2602.15984 | [PDF](https://arxiv.org/pdf/2602.15984v1)

**作者:** Riccardo De Santi `[一作]` (ETH Zürich), Andreas Krause `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用给定的验证器（verifier）对预训练的流模型（flow）或扩散模型进行全局或局部扩展，使其在有效设计空间（如分子空间）上覆盖更广，同时保持生成样本的有效性。

**💡 创新点**

1) 引入强弱验证器概念，将全局/局部流扩展建模为验证器约束的熵最大化；2) 提出在噪声状态空间上执行的可扩展镜像下降（Mirror‑Descent）算法，可同时处理全局与局部扩展；3) 给出闭式梯度表达式与理论收敛保证；4) 推导出无约束探索变体。

**🔧 技术方法**

概率空间优化、镜像下降、流模型/扩散模型 ODE/SDE 框架、KL 正则化、熵函数、强化学习/控制论中的过程惊喜最大化、可微梯度推导、RDKit 验证器、分子指纹 VENDI 多样性评估。

**📊 数据集**

QM9 分子数据集、GEOM‑Drugs 数据集；使用 FlowMol CTMC 预训练模型。

**📈 对比分析**

与 FlowMol CTMC、G‑vs S‑MEME、G‑FE vs CONSTR、FDC 等现有流/扩散探索方法以及标准约束生成方法进行对比。实验结果显示：在全局扩展中获得接近最优熵且 99% 样本有效率；在局部扩展中在保持 90%+ 有效率的同时显著提升熵；在分子构象多样性任务中，VENDI 指标从 89 提升到 100，样本有效率从 69% 提升到 81%，均优于基线方法。

**⚠️ 局限性**

1) 强验证器稀缺，弱验证器可能导致过度保守或仍出现无效样本；2) 高维流/扩散模型的数值稳定性和梯度估计仍是挑战；3) 需要手动调节超参数（α、γ、λ、η），对不同任务的选择较为敏感；4) 理论收敛证明基于理想化假设，实际近似求解可能收敛缓慢；5) 与纯探索方法相比，计算成本略有提升。

---

## 76. Statistical-Geometric Degeneracy in UAV Search: A Physics-Aware Asymmetric Filtering Approach

**arXiv ID:** 2602.15893 | [PDF](https://arxiv.org/pdf/2602.15893v1)

**作者:** Zhiyuan Ren `[一作]` (Xidian University), Ben Lan `[通讯]` (Guangdong Nasasi Communications Technology Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种针对灾后UAV搜索中NLOS偏差的非对称Huber EKF估计器，并通过主动“交叉”式机动获取双向信息来克服统计‑几何退化。

**💡 创新点**

创新点在于将NLOS非负物理先验显式融入MAP推导，得到一侧Huber损失函数，并证明主动生成双向信息是消除退化的必要条件。

**🔧 技术方法**

使用Bayesian MAP、EKF、闭式一侧Huber损失、主动感知规划（Reactive Crossing、FIM E‑optimal）以及仿真比较。

**📊 数据集**

使用仿真数据（2D nadir扫描、不同噪声水平、NLOS概率/偏差）进行Monte Carlo测试。

**📈 对比分析**

与对称Huber EKF、ADMM‑EKF以及FIM规划的组合对比，结果显示AsymmetricHuberEKF在收敛速度、最终RMSE方面优于对称基线，且在低噪声/高噪声场景下性能稳定，计算成本最低。

**⚠️ 局限性**

局限在于仅限二维平面、假设UAV姿态已知、未考虑飞行动力学、碰撞回避和实际硬件的信号处理等因素。

---

## 77. A Unified, Cross-Platform Framework for Automatic GUI and Plugin Generation in Structural Bioinformatics and Beyond

**arXiv ID:** 2602.16047 | [PDF](https://arxiv.org/pdf/2602.16047v1)

**作者:** Sikao Guo `[一作]`, Frédéric Cazals `[通讯]` (Inria)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

该论文提出了一套三阶段的工作流和工具包，自动化地为命令行工具生成跨平台（桌面与 Web）图形用户界面（GUI）插件，采用 MVP 设计模式，确保模型、视图与业务逻辑分离。

**💡 创新点**

创新点在于：
- 通过单一的 JSON 规范实现多平台 GUI 的“一次定义，多次生成”；
- 将插件的四个核心区域（输入、输出、3D 渲染、更新）模块化；
- 结合现有可视化引擎（PyMOL、VMD、NGL Viewer、Three.js）实现无缝通信；
- 形成可扩展的生成器架构，支持后续平台与引擎的快速接入。

**🔧 技术方法**

技术主要包括：MVP 架构、Qt Designer + 解析脚本、JSON 规范与验证、针对 pymol.Qt、PyQt6、Tkinter、Panel 的代码生成器、socket/API 交互、post‑analysis 脚本、以及 Web 前端技术（Panel + NGL/Three.js）。

**📊 数据集**

数据集主要是 SBL（Structural Bioinformatics Library）中的 CLI 工具（如界面建模器、α‑complex 生成器）及其产生的中间文件（.pdb、.ply、.obj、.json 等），用作生成 GUI 的输入与后处理输出。

**📈 对比分析**

论文未给出传统方法的量化对比，而是通过案例展示（VMD、PyMOL、Web）说明：
- 生成时间从数分钟到数小时显著缩短；
- 单一规格可即时同步到所有平台，避免同步错误；
- 代码可读性和可维护性提升，工程成本降低。由于缺乏基准实验，具体性能数值未给出。

**⚠️ 局限性**

局限性包括：
- 第一步设计仍需人工完成（布局与 flag 预选）；
- 需要手写后处理脚本，自动化程度有限；
- 当前仅支持四种 GUI/渲染库，扩展需实现新的生成器；
- 对于极大或特殊的 CLI 工具，生成器可能需要额外调整。

---

## 78. Egocentric Bias in Vision-Language Models

**arXiv ID:** 2602.15892 | [PDF](https://arxiv.org/pdf/2602.15892v1)

**作者:** Maijunxian Wang `[一作]` (University of California), Dezhi Luo `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FlipSet 基准评估视觉语言模型在第 2 级视角转换任务中的表现，发现大多数模型存在以自身视角为主的偏差，并揭示了模型在整合理论心智与空间旋转能力上的不足。

**💡 创新点**

设计了隔离 180° 旋转与理论心智的诊断式基准，采用多选答案类型进行错误分析，并通过大规模实验首次揭示 VLM 的组合性缺陷。

**🔧 技术方法**

在零样本条件下使用多选提示、链式思考（CoT）分析、统计分类和相关性检验等技术对 VLM 进行评估。

**📊 数据集**

构建了 FlipSet 数据集，包含 28 个二维字符字符串任务（共 336 个评估实例）以及对应的理论心智、空间旋转控制任务。

**📈 对比分析**

对 103 个公开 VLM 进行统一评估，平均准确率仅约 9%，低于 25% 随机水平；控制实验显示理论心智 ≈90%，空间旋转 ≈26%，第 2 级视角 ≈10%，并证实 L2 VPT 性能低于 ToM 与 MR 乘积，表明存在显著的组合性缺陷。

**⚠️ 局限性**

缺陷在于 VLM 缺乏将社会意识与空间操作绑定的机制，导致在整合任务时性能显著下降；模型仍然倾向于以自身视角思考，且链式思考无法改善这一偏差。

---

## 79. Rethinking ANN-based Retrieval: Multifaceted Learnable Index for Large-scale Recommendation System

**arXiv ID:** 2602.16124 | [PDF](https://arxiv.org/pdf/2602.16124v1)

**作者:** Jiang Zhang `[一作]` (Meta Platforms, Inc.), Qifan Wang `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种多面向可学习索引(MFLI)，在训练阶段联合学习物品嵌入与多层残差量化的分层码本，从而在服务时直接通过索引查找而不再需要近似最近邻搜索；

**💡 创新点**

创新点在于(1)引入多面向残差量化实现每个嵌入维度并行编码；(2)采用延迟启用、层级激活与正则化等技术实现在线稳定训练；(3)设计 split‑and‑merge 重新平衡机制保证索引大小可控；(4)构建统一的多面向索引映射与 delta‑update 管线，支持实时索引刷新；

**🔧 技术方法**

主要技术包括多面向残差量化(MF‑RQ)、多面向损失与量化损失联合训练、样本软最大化(Sampled Softmax)、k‑means 切分、索引重平衡、Delta‑update 与全量快照、GPU 并行实现、Per‑index reranker；

**📊 数据集**

使用的是工业级视频推荐平台真实日志，覆盖数十亿用户与数亿视频，数据分为训练集 Period P1 与评估集 Period P2；

**📈 对比分析**

离线评测对比六种基线（NeuCF, MoL, HLLM, VQIndex, HSTU, MTMH）在四项召回任务（VVC, Like, LWT, CCD）和两项语义相关性指标（T1, T2）上进行，MFLI 在召回上提升最高 11.8% 及 6.3%，在语义相关性上提升 110%~126%，在线 A/B 测试中 QPS 提升 60%，显著降低流行度偏倚并提升新鲜度与参与度；

**⚠️ 局限性**

局限性包括：(1) 对索引平衡参数（如 B_low/B_upp）的依赖较大，需人工调优；(2) 多面向模型训练复杂度高，GPU 资源占用显著；(3) 在极稀疏或极冷启动场景下，码本量化精度可能不足；(4) Delta‑update 仍需周期性全量刷新，可能在极大规模时引入延迟；

---

## 80. Playing With AI: How Do State-Of-The-Art Large Language Models Perform in the 1977 Text-Based Adventure Game Zork?

**arXiv ID:** 2602.15867 | [PDF](https://arxiv.org/pdf/2602.15867v1)

**作者:** Berry Gerrits `[一作]` (University of Twente), Berry Gerrits `[通讯]` (University of Twente)

**通讯引用:** 141 | [OpenAlex ID](https://openalex.org/A5036576570)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了 ChatGPT、Claude、Gemini 等主流 LLM 在文本冒险游戏 Zork（Zork I）的推理与问题解决能力。

**💡 创新点**

创新点在于将文本冒险游戏作为对 LLM 元认知与长时记忆的自然评估框架，并系统比较不同“思考”模式和提示细节对表现的影响。

**🔧 技术方法**

使用零射击（zero‑shot）对话交互、基本与高级提示工程、模型自带的“extended thinking”模式，以及得分和动作计数等评价指标。

**📊 数据集**

使用的数据集为 Zork I（《The Great Underground Empire》）的官方文本游戏环境和描述。

**📈 对比分析**

通过对六个 LLM 进行 5 次独立运行，统计最终得分和动作数进行比较；结果显示平均得分不到 10%，Claude Opus 4.5 最高约 75 分（≈20% 完成），“思考”与详细提示对性能无显著提升。

**⚠️ 局限性**

局限性包括样本量小、缺乏人类基准、无法排除训练数据泄漏、仅采用示例性定性分析、未系统探索其他提示或模型调优方案。

---

## 81. Large Language Models for Assisting American College Applications

**arXiv ID:** 2602.15850 | [PDF](https://arxiv.org/pdf/2602.15850v1)

**作者:** Zhengliang Liu `[一作]` (University of Georgia), Tianming Liu `[通讯]` (University of Georgia)

**通讯引用:** 17223 | [OpenAlex ID](https://openalex.org/A5100647156)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出EZCollegeApp，一款基于LLM的助理系统，用人类监督的方式为高中生提供美国高校申请表单的结构化填充与答案提示。

**💡 创新点**

创新点包括映射优先的表单理解架构、来源分离的检索增强生成、以及严格的人机循环、提取而非生成的策略。

**🔧 技术方法**

核心技术涵盖大型语言模型、检索增强生成、结构化语义映射、向量检索、浏览器插件以及多层安全与隐私设计。

**📊 数据集**

数据来源包括官方招生网站抓取、FAQ与社区论坛爬取以及学生上传的成绩单、简历和个人陈述等文件。

**📈 对比分析**

在自动化测试与人工评估中，系统回答率达80–85%，引用有效率92%，并获得用户在有用性、可读性与可信度方面的积极反馈。

**⚠️ 局限性**

局限性在于无法生成原创作文、对极少见或新政策的即时更新依赖爬虫、以及在复杂多语言或非结构化表单场景下的准确性不足。

---

## 82. Hiding in Plain Sight: Understanding the Everyday Practices and Challenges of Car Dwellers

**arXiv ID:** 2602.16112 | [PDF](https://arxiv.org/pdf/2602.16112v1)

**作者:** Rachael Zehrung `[一作]` (University of California), Yunan Chen `[通讯]` (University of California)

**通讯引用:** 3832 | [OpenAlex ID](https://openalex.org/A5100658705)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对r/urbancarliving子版块2020‑2025年期间约150条帖子与评论进行定性主题分析，研究了车居者在日常生活、技术使用与身份认同方面的做法与挑战；

**💡 创新点**

首次将车居视为介于无家可归与数字游牧之间的“中间生活”，强调基础设施能力对身份塑造的作用，并提出针对不同动机的技术干预与支持策略；

**🔧 技术方法**

采用定性主题分析方法与MaxQDA软件，利用Arctic Shift抓取Reddit公开数据，无机器学习或算法实现；

**📊 数据集**

2020‑2025年间r/urbancarliving子版块的帖子与评论（共150条主题化记录，约13,500字节的文本数据），来源于公开的Reddit数据；

**📈 对比分析**

该研究不涉及算法比较或性能评估，而是通过数据饱和与主题重复验证研究可信度；

**⚠️ 局限性**

局限在于样本仅来自Reddit用户，主要是美国地区、数字可达性高的群体，缺乏对不使用网络车居者与多元文化背景的代表性；

---

## 83. From Reflection to Repair: A Scoping Review of Dataset Documentation Tools

**arXiv ID:** 2602.15968 | [PDF](https://arxiv.org/pdf/2602.15968v1)

**作者:** Pedro Reynolds-Cuéllar `[一作]` (Robotics and AI Institute), Heila Precel `[通讯]` (Robotics and AI Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开展了一项系统综述并混合方法分析59篇关于数据集文档工具的论文，探讨其动机、概念化与整合情况。

**💡 创新点**

识别了工具设计中的四大障碍（价值不明确、去情境化、未说明劳动负担、未来整合视角）并提出将负责任AI工具设计从个体向机构转变的研究议程。

**🔧 技术方法**

采用PRISMA指南的scoping review方法、ATLAS.ti进行反思性主题分析，并对工具自动化程度和集成方式进行量化描述。

**📊 数据集**

文献数据来源为ACM DL、IEEE Xplore、ScienceDirect、ArXiv、ACL Anthology以及AAAI、NeurIPS会议记录，最终筛选59篇论文。

**📈 对比分析**

通过工具类型、自动化级别、参与者身份、利益相关者参与度等定量指标进行分类比较，但多数工具缺乏实证评估，使用效果和性能不确定。

**⚠️ 局限性**

仅覆盖公开学术资源，未包括私有工具；缺少真实工作流与用户研究；对技术实现细节和效果的实证证据不足。

---

## 84. Peeking Ahead of the Field Study: Exploring VLM Personas as Support Tools for Embodied Studies in HCI

**arXiv ID:** 2602.16157 | [PDF](https://arxiv.org/pdf/2602.16157v1)

**作者:** Xinyue Gui `[一作]` (University of Tokyo), Takeo Igarashi `[通讯]` (University of Tokyo)

**通讯引用:** 11374 | [OpenAlex ID](https://openalex.org/A5102743150)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过在同一场景下对比真实参与者的场地实验与基于Vision‑Language模型（VLM）的模拟角色（Persona）完成的街道交叉任务，探讨VLM是否能复制人类行为并支持HCI的实地研究。

**💡 创新点**

创新点在于：①首次将VLM persona引入AV‑行人交互的实地研究场景；②提出了一套从问卷到prompt的Persona构建流程；③通过量化与质化比较，给出三条可操作的使用指南。

**🔧 技术方法**

使用的技术包括：OpenAI GPT‑4.1（VLM）、YOLOv8姿态检测、自定义交互式视频模拟器、非参数三因素ANOVA、主题分析等。

**📊 数据集**

数据集由20名真实受试者在现场完成6个实验条件（共120个视频）与对应的VLM模拟轨迹（共120条轨迹）组成。

**📈 对比分析**

比较方法：将人类与VLM的跨越时间、轨迹决策分布、Likert评分与开放式回答进行对齐；结果显示VLM的平均跨越时间与人类无显著差异（p≈0.85），但VLM表现出更低的变异性和更高的“自信”评分。

**⚠️ 局限性**

局限性包括：①VLM缺乏真实感知（如深度视角）导致对细节和异常行为的捕捉不足；②构造Persona的prompt高度依赖模型，可能导致“冻结”行为；③VLM对内部自评（如自信）存在夸大，不能完全替代少量个体或特殊人群的实验。

---

## 85. Memes-as-Replies: Can Models Select Humorous Manga Panel Responses?

**arXiv ID:** 2602.15842 | [PDF](https://arxiv.org/pdf/2602.15842v1)

**作者:** Ryosuke Kohita `[一作]` (CyberAgent), Seiichiro Yoshioka `[通讯]` (CyberAgent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Meme Reply Selection 任务，并构建了大规模、可复用的 MaMe‑Re 基准；

**💡 创新点**

创新点在于将 meme 视作交互式回复，设计了针对上下文幽默的选择任务，并提供了完整的评估框架；

**🔧 技术方法**

采用大语言模型（如 GPT‑5、Gemini 2.5 Pro）和多模态/相似度编码器进行偏好式与相似度式匹配；

**📊 数据集**

使用了 250 条合成社交媒体上下文与 400 张漫画面板，形成 100,000 个上下文‑meme 对，附 500,000 条二元幽默注释；

**📈 对比分析**

通过 Score@1、Consensus Hit Rate (CHR) 与 nDCG 等指标比较，LLM 在检索‑重排架构中表现最好，但整体性能仍远低于人类水平；

**⚠️ 局限性**

局限在于高低一致性差、仅限漫画域、视觉信息难以有效利用、以及难以区分语义相似的幽默候选。

---

## 86. Towards a More Realistic VR Experience: Merging Haptic Gloves with Precision Gloves

**arXiv ID:** 2602.15833 | [PDF](https://arxiv.org/pdf/2602.15833v1)

**作者:** Paolo Bottoni `[一作]` (Sapienza University of Rome), Marco Raoul Marini `[通讯]` (Sapienza University of Rome)

**通讯引用:** 553 | [OpenAlex ID](https://openalex.org/A5069310487)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

开发了一种混合式VR手套，将高精度数据手套与触觉手套结合，实现精准捕捉与触觉反馈；

**💡 创新点**

首次提出通过UDP实时将数据手套信号注入SenseGlove SDK，绕过触觉手套原生输入误差的集成方案；

**🔧 技术方法**

采用Meta Quest 3、Yamaha Data Glove、SenseGlove Nova 1、PC、Unity引擎、UDP传输及SenseGlove SDK进行实现；

**📊 数据集**

使用15名不同背景参与者的SUS和NASA‑TLX问卷数据进行评估；

**📈 对比分析**

相较单一触觉手套，混合系统在SUS平均65.7%、Raw NASA‑TLX 27.76%显示易用且低认知负荷，验证了精度与沉浸度的提升；

**⚠️ 局限性**

局限在于仍需改进校准函数、进一步降低延迟，并实现全无线化以提升实用性。

---

## 87. Energy-Efficient p-Bit-Based Fully-Connected Quantum-Inspired Simulated Annealer with Dual BRAM Architecture

**arXiv ID:** 2602.16143 | [PDF](https://arxiv.org/pdf/2602.16143v1)

**作者:** Naoya Onizawa `[一作]` (Research Institute of Electrical Communication), Takahiro Hanyu `[通讯]` (Research Institute of Electrical Communication)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于p-bit的全连接SSQA FPGA架构，实现能耗低的量子启发式模拟退火加速器。

**💡 创新点**

采用旋转-序列/复制并行调度与双BRAM延迟线，消除传统shift-register导致的线性fan-out，保持逻辑资源与时延几乎不随节点数增长；并通过只保存最终复制状态减少存储。

**🔧 技术方法**

p-bit、SSQA算法、双BRAM延迟线、XOR-Shift随机数生成器、AXI调度控制、Xilinx ZC706 FPGA实现。

**📊 数据集**

G-set中的G11–G15（800节点MAX‑CUT）以及TSP、图同构等QUBO实例。

**📈 对比分析**

与传统shift‑register SSQA、HA‑SSA、IPAPT以及CPU、GPU基准进行资源、能耗、延迟比较；在800节点MAX‑CUT上实现0.091 W功耗、LUT/FF 1.45%/0.38%，比传统方案节能约70%、逻辑资源降低97%，延迟12 ms可通过并行化降低到1 ms。

**⚠️ 局限性**

单序列更新导致单步时延较高，双BRAM占用的BRAM块仍随N²增长，无法处理极大规模问题；对稀疏网络需压缩权重以降低BRAM需求。

---

## 88. CAST: Achieving Stable LLM-based Text Analysis for Data Analytics

**arXiv ID:** 2602.15861 | [PDF](https://arxiv.org/pdf/2602.15861v1)

**作者:** Jinxiang Xie `[一作]` (Nanjing University), Dongmei Zhang `[通讯]` (Microsoft Research)

**通讯引用:** 11374 | [OpenAlex ID](https://openalex.org/A5100331488)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出CAST框架，在文本分析与表格数据整合（TADA）中通过算法化提示和先思考后输出显著提升LLM生成的摘要和标签的稳定性。

**💡 创新点**

创新点在于将任务拆解为可编程的算法步骤，并强制模型先生成并确认中间状态，显著降低潜在推理路径熵，从而实现高稳定性的单次生成。

**🔧 技术方法**

采用算法化提示（Algorithmic Prompting）和先思考后输出（Thinking‑before‑Speaking）两种技术，结合LLM的自回归生成，形成一条受限的推理轨迹。

**📊 数据集**

使用多种公开数据集：MASSIVE、Google Play 评论、Amazon 及 Book 书评、Teams 用户反馈、Sushi 虚拟评论、Twitter 线程等，构建32个摘要–查询对和5,100条标签任务。

**📈 对比分析**

与Zero‑shot CoT、Few‑shot CoT、Self‑Consistency、AP‑Only、TbS‑Only等基线在稳定性、准确率和处理时延上对比，CAST在摘要与标签的稳定性分数（CAST‑S/CAST‑T）始终位居榜首，且在质量与效率方面无显著下降。

**⚠️ 局限性**

局限在于需人工预设算法流程，难以自动迁移到全新任务；过度约束可能压制细微语义差异；对算法细粒度与泛化性仍有进一步探索空间。

---

## 89. World Action Models are Zero-shot Policies

**arXiv ID:** 2602.15922 | [PDF](https://arxiv.org/pdf/2602.15922v1)

**作者:** Seonghyeon Ye `[一作]`, Joel Jang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了World Action Models（WAMs），一种基于大规模视频扩散模型的机器人基础模型，能够联合预测视频未来状态和动作，实现跨环境、跨任务和跨实体的零-shot 与少样本泛化；

**💡 创新点**

创新点包括①利用互联网视频的时空先验将视频预测与逆动力学联合训练，②采用自回归 DiT + KV 缓存实现异步闭环控制，③提出 -Flash 噪声调度和一系列系统级加速实现 38× 推理速度，④通过仅视频的跨实体学习提升 42% 以上的未见任务性能，并展示仅 30 分钟玩耍数据即可完成新机器人适配；

**🔧 技术方法**

核心技术包括：Wan2.1-I2V-14B-480P 预训练图像到视频扩散模型；DiT 自回归 Transformer + Flow‑matching 训练目标；教师强迫、KV 缓存、CFG 并行、DiT 缓存；Torch Compile + CUDA Graphs、NVFP4 量化、-Flash 噪声调度、Savitzky‑Golay 滤波；

**📊 数据集**

使用约 500 小时 AgiBot G1 真实世界遥控数据；Frank 机器人公开的 DROID 多样化数据；12 分钟人类 egocentric 视频；20 分钟 YAM 机器人视频；并在 RoboArena、PolaRiS、Genie Sim 3.0 等基准上进行评测；

**📈 对比分析**

对比两大 VLA 基线（GR00T N1.6 与 π0.5），在从零和预训练两种初始化下，评估 seen/unseen 任务、跨实体、少样本；WAM 在 seen 任务平均任务进度提升 2×，未见任务达 39.5%（比 16.3% 更高）；跨实体视频学习后进度提升至 55%；-Flash 单步推理保持 74% 进度，推理频率达 7Hz；

**⚠️ 局限性**

局限性包括：大模型导致计算成本高，实时性能仍低于 20Hz 的 VLA；在亚厘米级高精度任务上表现不足；跨实体时受视频预测误差影响；缺乏长时程记忆与系统 2 规划；需要进一步研究尺度定律和大规模人类视频迁移。

---

## 90. Not the Example, but the Process: How Self-Generated Examples Enhance LLM Reasoning

**arXiv ID:** 2602.15863 | [PDF](https://arxiv.org/pdf/2602.15863v1)

**作者:** Daehoon Gwak `[一作]` (KAIST AI), Jaegul Choo `[通讯]` (KAIST AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了自生成提示对大型语言模型推理性能的影响，并证明问题生成过程是提升效果的关键。

**💡 创新点**

创新点在于揭示自生成示例提升的核心来自于生成过程本身，而非示例内容本身。

**🔧 技术方法**

使用了三种提示策略（零样本、集成、分离），并对注意力模式进行分析。

**📊 数据集**

使用了数学推理数据集MATH和GSM8K进行评估。

**📈 对比分析**

通过在五种模型上比较，集成提示始终优于零样本和分离提示，提升率可达数个百分点。

**⚠️ 局限性**

局限性包括仅在数学领域验证，未覆盖其他专业领域，且注意力分析仅在单一开源模型上完成。

---

## 91. Preference Optimization for Review Question Generation Improves Writing Quality

**arXiv ID:** 2602.15849 | [PDF](https://arxiv.org/pdf/2602.15849v1)

**作者:** Karun Sharma `[一作]`, Prayag Tiwari `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了专门生成高质量同行评审问题的模型 IntelliAsk，并构建了相应的奖励模型 IntelliReward，以提高问题的努力度、证据性和扎实度。

**💡 创新点**

提出基于人类专家标注的三维质量度量（努力、证据、扎实度），并训练奖励模型；用该奖励模型进行强化学习，使生成的评审问题在质量上明显优于传统 SFT 与现有大模型。

**🔧 技术方法**

结合冻结的自回归 LLM 与多头 Transformer 头的奖励模型；采用 DAPO/GRPO 强化学习；使用人类偏好标注的奖励训练；对模型进行对比实验和自动化评估。

**📊 数据集**

采集自 ICLR 2024 OpenReview 评论的 15.5k 问题与 5,841 篇论文；构建 572 题‑论文对用于人类偏好标注；以及多项公开基准（MuSR、WritingBench 等）用于外部评测。

**📈 对比分析**

与 SFT 基线、主流 LLM（Gemini、o3、Qwen 等）以及公开基准进行对比；在人工评估中 IntelliAsk‑32B 在三维指标上得到 0.66/3，显著高于 Gemini 2.5 Pro；在自动化 IntelliReward 评估中达到 0.55/3，远超 SFT；在外部推理/写作基准上匹配或超过 Qwen3‑32B。

**⚠️ 局限性**

仅覆盖文本内容，未处理图表等多模态信息；缺乏更大规模的专家标注；模型规模受限，可能难以在更大模型上进一步提升；在现实评审中仍需谨慎使用以免产生过于复杂的无用问题。

---

## 92. P-RAG: Prompt-Enhanced Parametric RAG with LoRA and Selective CoT for Biomedical and Multi-Hop QA

**arXiv ID:** 2602.15874 | [PDF](https://arxiv.org/pdf/2602.15874v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 93. Can LLMs Assess Personality? Validating Conversational AI for Trait Profiling

**arXiv ID:** 2602.15848 | [PDF](https://arxiv.org/pdf/2602.15848v1)

**作者:** Andrius Matšenas `[一作]`, Kim Lilii Tamm `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过与 IPIP-50 问卷对照，验证了大型语言模型（LLM）在实时对话中提取 Big Five 维度人格特质的可行性。

**💡 创新点**

创新点在于将 LLM 的对话式评估与传统问卷相结合，并引入用户主观准确度评估，展示对话式人格测评的可接受性与可靠性。

**🔧 技术方法**

使用 Gemini 2.5 Flash LLM 进行指导式对话，随后由另一 LLM 对话记录进行得分，涉及 20 条开放式提问。

**📊 数据集**

数据集包括 33 名受试者的对话记录与对应的 IPIP-50 问卷得分。

**📈 对比分析**

通过配对 t 检验和 Pearson 相关，发现 LLM 与问卷得分在 3 项（Conscientiousness、Openness、Neuroticism）无显著差异，相关系数介于 0.38–0.58，且用户认为两种方法同等准确。

**⚠️ 局限性**

局限在样本量小、仅使用单一 LLM、顺序效应、样本偏向年轻技术用户以及 IPIP-50 本身的自报偏差。

---

## 94. MARLEM: A Multi-Agent Reinforcement Learning Simulation Framework for Implicit Cooperation in Decentralized Local Energy Markets

**arXiv ID:** 2602.16063 | [PDF](https://arxiv.org/pdf/2602.16063v1)

**作者:** Nelson Salazar-Pena `[一作]`, Andres Gonzalez-Mancera `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个开源的多智能体强化学习（MARL）仿真框架，用于在去中心化本地能源市场（LEM）中研究隐式协作与市场-物理耦合。

**💡 创新点**

创新点包括：1）将模块化市场清算、物理网格约束与MARL无缝集成；2）支持完全去中心化的训练与执行（DTDE）并通过共享 KPI 和奖励函数诱导隐式协作；3）三阶段匹配机制（偏好匹配、双向拍卖、DSO 清算）；4）可插拔的市场规则、信誉与区块链验证；5）完整的分析与可视化工具。

**🔧 技术方法**

技术实现：基于 Gymnasium 的多智能体环境；使用 Ray RLlib 中的 PPO、APPO、SAC；动态生成可再生与负荷时间序列；图网络电网模型（IEEE 13/34 节点、线性损耗、拥塞）；奖励设计结合系统 KPI；数据结构包括订单、交易、网格状态与 KPI。

**📊 数据集**

数据集：由程序生成的光伏/负荷时间序列（高斯光照+噪声），以及标准电网拓扑（IEEE 13/34 节点）和可生成的网格拓扑；无外部真实数据。

**📈 对比分析**

对比方法：使用零知识智能体（随机投标）与理论上最优的 CTCE 基准；在储能协调案例中展示：无电池 vs 战略性电池配置。结果显示：电池配置降低价格波动（价格分布由多峰转单峰）、提高自消费率、提升协调得分；CTCE 较 DTDE/CTDE 仍有性能提升，验证去中心化学习的可行性。

**⚠️ 局限性**

局限性：1）目前仅在零知识智能体下验证，缺乏完整 MARL 训练结果；2）仿真基于合成数据，未与真实市场和设备参数对标；3）规模受限（最多十数智能体），大规模网络与复杂市场规则尚未充分测试；4）部分物理约束（如电压、动态潮流）被简化；5）对 DSO 行为的假设（如固定费率）可能不反映实际运营。

---

## 95. KD4MT: A Survey of Knowledge Distillation for Machine Translation

**arXiv ID:** 2602.15845 | [PDF](https://arxiv.org/pdf/2602.15845v1)

**作者:** Ona de Gibert `[一作]` (University of Helsinki), Jörg Tiedemann `[通讯]` (University of Helsinki)

**通讯引用:** 8671 | [OpenAlex ID](https://openalex.org/A5082417280)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2025年10月前105篇关于机器翻译（MT）中的知识蒸馏（KD）论文进行了全面梳理与归类，系统总结了KD方法、应用场景、研究趋势及存在的空白。

**💡 创新点**

创新点在于：①构建了KD4MT专用数据库与词汇表；②从方法论、算法细化和应用三个维度重新定义并细分KD技术；③揭示KD在MT中既是压缩工具也是监督重塑与效率提升手段；④指出评估标准缺失、偏见放大、低频词效果不佳等关键问题。

**🔧 技术方法**

使用的技术包括：响应式KD（Word-KD、Seq-KD）、特征式KD（Feature-KD）及其变体；多教师、代理任务、LLM辅助的KD；以及基于RL的对策（f‑distill、ImitKD、GKD）等；对各类KD在多语言、低资源、域适应、时延敏感翻译等应用中的具体实现做了深入梳理。

**📊 数据集**

主要数据集为MT通用并行语料库（如WMT、IWSLT、OpenSubtitles）和单语语料；由于调查基于文献综述，未使用统一实验数据，而是汇总了被综述论文所采用的多样化数据来源，重点指出大多数研究以英双语为主。

**📈 对比分析**

对比方法主要是通过量化统计（KD类型使用频率、应用场景分布、参数比等）和性能趋势分析；未进行统一实验对比，但总结了Seq‑KD在低资源和NAT中的优势、Word‑KD在压缩任务中的可行性，以及LLM辅助KD在数据生成与多教师情景下的潜在收益；总体发现不同KD策略在不同任务中表现差异显著。

**⚠️ 局限性**

局限性包括：①缺乏统一评测指标与共享基准，导致方法可比性差；②多数研究偏向英语双语，低资源语言缺乏充分验证；③特征式KD研究不足，难以系统评估其优势；④KD对数据熵的削减易导致低频词召回率下降、幻觉与偏见放大风险；⑤LLM辅助KD受限于模型规模与成本，且合成数据质量与多样性尚未得到充分规范。

---

## 96. Markov Chains with Rewinding

**arXiv ID:** 2602.16028 | [PDF](https://arxiv.org/pdf/2602.16028v1)

**作者:** Amir Azarmehr `[一作]` (Northeastern University), Madhu Sudan `[通讯]` (Harvard University)

**通讯引用:** 19443 | [OpenAlex ID](https://openalex.org/A5101519422)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出并系统研究了一类新的随机过程——可重置马尔可夫链（Markov Chains with Rewinding），并以此为模型解决在部分可观测马尔可夫链中辨别初始状态的问题。

**💡 创新点**

创新点在于：①证明任意可重置策略（包括自适应和非自适应）都能以非自适应策略完成辨别；②给出多项式时间非自适应算法，其查询复杂度为 _M(a,b) 的 O(n) 次方；③构造了实例展示在最坏情况下自适应与非自适应查询复杂度之间存在多项式（甚至是几乎指数）差距；④通过归约证明任何一般部分可观测马尔可夫链的辨别问题可转化为规范链（canonical chain）的实例，保持查询与运行时间的近似不变。

**🔧 技术方法**

主要技术包括：对状态空间进行分区（partition）并构造加权有向图；利用分区之间的总变差距离（total variation distance）定义边权；求取从最粗分区到区分目标状态的最短路径，得到查询树；采用随机抽样与集中不等式实现非自适应测试；使用耦合（coupling）与总变差距离分析下界；以及构造“延展路径”与“特殊状态”实现从一般链到规范链的归约。

**📊 数据集**

该工作纯粹为理论分析，没有使用具体的数据集；所有结论均基于数学证明与构造的马尔可夫链实例。

**📈 对比分析**

在对比方面，作者证明非自适应算法能够在多项式时间内完成任务，且其查询复杂度仅比最优自适应策略高出多项式因子；然而通过构造的示例表明，在某些参数设置下，最优自适应策略需要 O(n^2 d) 次查询，而最优非自适应策略至少需要 Ω(d^{1-o(1)} n) 次查询，显示了显著的性能差距。

**⚠️ 局限性**

主要限制在于：①非自适应算法的查询复杂度与状态数 n 的指数相关，难以在 n 取大值时保持紧凑；②虽然证明了存在多项式差距，但并未给出能消除 n 依赖的更优算法；③归约到规范链虽然理论上可行，但在实践中可能导致状态空间急剧膨胀，影响实际可实现性；④下界示例仅涵盖特殊构造的链，尚未证明所有可能链都存在相同级别的差距。

---

## 97. The Perplexity Paradox: Why Code Compresses Better Than Math in LLM Prompts

**arXiv ID:** 2602.15843 | [PDF](https://arxiv.org/pdf/2602.15843v1)

**作者:** Warren Johnson `[一作]` `[通讯]` (Bona Opera Studios), Warren Johnson (Bona Opera Studios)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在多种代码与推理基准上对压缩阈值进行系统验证，展开了对压缩决策的逐词困惑度分析，并基于此提出了任务感知自适应压缩算法 TAAC。

**💡 创新点**

创新点在于首次揭示“困惑度悖论”——代码语法因高困惑度被保留而推理任务中的数值被剔除；并通过函数签名注入实验验证了该机制，同时设计了能在不同任务下动态调节压缩率并保证质量的自适应算法。

**🔧 技术方法**

使用的技术包括基于困惑度与学习型的提示压缩方法（LLMLingua、SelectiveContext）、DistilBERT任务分类器、Perplexity 计算、信息密度估计、MLP 质量预测器以及 ANCOVA 等统计分析。

**📊 数据集**

所用数据集包括代码基准 HumanEval、MBPP、HumanEval+、MultiPL‑E 及推理基准 GSM8K、MATH、ARC‑Challenge 与 MMLU‑STEM。

**📈 对比分析**

与固定比例压缩相比，TAAC 在保持 95.6% 质量的前提下实现 21.8% 的成本节约，较固定 0.6 压缩率提升 6.5 个百分点，且在 Pareto 前沿表现优异。

**⚠️ 局限性**

局限性包括仅评估单函数生成任务、单轮交互、基于单一 pilot 模型的困惑度估计，未涉及更长代码、复合任务或多轮推理场景。

---

## 98. MoE-Spec: Expert Budgeting for Efficient Speculative Decoding

**arXiv ID:** 2602.16052 | [PDF](https://arxiv.org/pdf/2602.16052v1)

**作者:** Bradley McDanel `[一作]` (Franklin and Marshall), Harshit Khaitan `[通讯]` (Meta Reality Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MoE-Spec，一种在Mixture-of-Experts（MoE）模型中对推理验证阶段实行专家预算的训练无关方法，显著降低专家加载量；

**💡 创新点**

通过聚合草稿树中路由概率，按重要性动态限制每层加载的专家数，实现了把验证成本与草稿树复杂度解耦；

**🔧 技术方法**

利用MoE路由器产生的专家权重做排序、子集选择，并采用截断或替换策略处理被裁剪专家；

**📊 数据集**

在OLMoE-1B/7B、Qwen3-30B-A3B、Mixtral-8x7B三大MoE架构上，使用GSM8K、MATH500、HumanEval、MBPP、CNN/DailyMail等五个基准；

**📈 对比分析**

与基准EAGLE和自回归推理相比，MoE-Spec在三种模型上平均提升10–30%吞吐率，同时保持与EAGLE相近的质量；

**⚠️ 局限性**

需要手动调节预算超参数；对极小草稿树时预算开销可能超过收益；只针对softmax top‑k路由，sigmoid或其他路由机制需进一步适配；

---

## 99. Weak Zero-Knowledge and One-Way Functions

**arXiv ID:** 2602.16156 | [PDF](https://arxiv.org/pdf/2602.16156v1)

**作者:** Rohit Chatterjee `[一作]` (National University of Singapore), Prashant Nalini Vasudevan `[通讯]` (National University of Singapore)

**通讯引用:** 400 | [OpenAlex ID](https://openalex.org/A5032319697)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

研究了弱零知识（Zero‑Knowledge）协议在最坏情况硬语言上的意义，证明若所有NP语言均拥有误差满足 ϵ_c+ϵ_s+ϵ_z<1 的弱NIZK或公共币ZK协议，则可以构造一阶或无限经常性的一键函数，从而推出一键函数的存在。

**💡 创新点**

将前人仅在 ϵ_c+√ϵ_s+ϵ_z<1 条件下得到的结果提升到最一般的 ϵ_c+ϵ_s+ϵ_z<1，且对公共币多轮协议得到更紧的 ϵ_c+ϵ_s+(2k-1)ϵ_z<1 条件；对常数轮协议进一步改进到 ϵ_c+ϵ_s+kϵ_z<1，展示了弱零知识在最坏情况硬语言上的更强力。

**🔧 技术方法**

采用对抗性逆函数构造与模拟器相结合的技术，利用分布式一键函数与零知识模拟器的相互作用，构造可用来判定语言成员性的识别器；对多轮协议引入递归函数族与逼近技术，通过 Hoeffding 及数据处理不等式累积误差，得到更精确的误差阈值。

**📊 数据集**

本文为理论工作，不使用具体数据集；所有证明均基于 NP 语言、随机参考字符串、公开币等理论构造。

**📈 对比分析**

比较方法主要是与之前在 ϵ_c+√ϵ_s+ϵ_z<1 条件下的结果和公开币 ZK 协议的 ϵ_c+ϵ_s+kϵ_z<1 结果进行对比；通过误差阈值的收缩，展示在最坏情况硬语言上对弱零知识协议的更广泛适用性，性能表现体现在能直接推出一键函数的存在，而不需要额外的加密或弱一键函数假设。

**⚠️ 局限性**

限制在于：常数轮公共币协议只能得到无限经常性的一键函数，而无法得到标准一键函数；递归构造导致算法复杂度上升，且对于非常数轮协议仍未实现更强的结果；同时，证明依赖于 NP 语言的最坏情况硬假设，若该假设失效，则结果不成立。

---

## 100. A Study on Real-time Object Detection using Deep Learning

**arXiv ID:** 2602.15926 | [PDF](https://arxiv.org/pdf/2602.15926v1)

**作者:** Ankita Bose `[一作]` (GITAM University), Naveen N `[通讯]` (GITAM University)

**通讯引用:** 3217 | [OpenAlex ID](https://openalex.org/A5101756206)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了实时目标检测的深度学习模型、公开与自制数据集、应用场景，并给出了多模型在同一评测指标下的对比。

**💡 创新点**

提出了系统化的模型演进与版本分支图谱，整合了评测框架与未来研究方向，首次对YOLO、SSD、Transformer等多族类模型在同一基准下进行统一对比。

**🔧 技术方法**

采用CNN、R-CNN系列、YOLO、SSD、RetinaNet、EfficientDet、DETR、Transformer等主流深度学习技术，并对其架构、Anchor、特征融合等关键技术进行了梳理。

**📊 数据集**

使用了COCO、PASCAL VOC、KITTI、WIDER FACE、DUTS、UCF、以及多领域自制数据集（如车、枪、人脸、脑瘤等）进行实验与讨论。

**📈 对比分析**

通过 mAP、IoU、FPS、模型大小等统一指标对比，表明YOLOv10、EfficientDet、DETR等在速度与精度上各有优势，但仍需权衡；实验表明在COCO上EfficientDet-D4可达92.1% mAP，YOLOv10-S实现高达200 FPS。

**⚠️ 局限性**

受限于小目标、遮挡、低对比度场景、Transformer高计算成本、缺乏统一硬件基准与跨域泛化能力等因素，导致在边缘设备与安全关键场景下仍有性能与可靠性挑战。

---

## 101. Reranker Optimization via Geodesic Distances on k-NN Manifolds

**arXiv ID:** 2602.15860 | [PDF](https://arxiv.org/pdf/2602.15860v1)

**作者:** Wen G. Gong `[一作]` `[通讯]`, Wen G. Gong

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了在检索增强生成（RAG）系统中使用几何重排（Maniscope）的方法，以在初始检索候选集上基于k‑NN流形的测地距离对文档进行重排序。

**💡 创新点**

创新点在于将测地距离与全局余弦相似度相结合，利用k‑NN图的局部几何结构提升重排质量，同时通过算法优化实现毫秒级低延迟。

**🔧 技术方法**

核心技术包括稀疏k‑NN图构建、基于余弦距离的边权设定、Dijkstra单源最短路径计算、以及向量化的邻近搜索与CSR图存储。

**📊 数据集**

在8个BEIR基准数据集（共1233个查询）上进行评估，涵盖医学、科学、金融、网络搜索、论证、事实验证等多领域。

**📈 对比分析**

与HNSW、Jina Cross‑Encoder、BGE‑M3等现有重排器对比，Maniscope在最难的3个数据集上分别提升7.0%、1.6%和2.8% NDCG@3，平均延迟仅4.7 ms，比HNSW快3.2×、比交叉编码器快10‑45×，并且与最优交叉编码器的性能差距不足2%。

**⚠️ 局限性**

局限性包括：仅适用于少量候选集（M≈10–100），对k值敏感（图可能不连通）、依赖高质量嵌入；当候选数过大时，层次化方法（如HNSW）可能更高效。

---

## 102. Revolutionizing Long-Term Memory in AI: New Horizons with High-Capacity and High-Speed Storage

**arXiv ID:** 2602.16192 | [PDF](https://arxiv.org/pdf/2602.16192v1)

**作者:** Hiroaki Yamanaka `[一作]` (Kioxia Corporation), Jun Deguchi `[通讯]` (Kioxia Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在自进化式AI代理的外部记忆研究中提出并验证了三种新颖范式：STORE‑THEN‑ON‑DEMAND‑EXTRACT（STONE）、深层洞察发现以及经验记忆共享；

**💡 创新点**

创新点在于：①将经验完整保留以避免信息丢失并实现多任务复用；②通过聚合多条经验进行统计推断，提升在随机环境下的决策可靠性；③构建多代理共享记忆池以降低单体试错成本；

**🔧 技术方法**

主要技术包括：LLM驱动的检索增强生成（RAG）、基于KV‑cache的加速抽取、密集向量与稀疏逻辑检索的组合、ε‑greedy策略与经验重放对比实验；

**📊 数据集**

实验使用的主要数据集为：公司内部规章文档（构造问答集）、HotpotQA问答集以及三臂赌博机模拟环境；

**📈 对比分析**

与传统的“抽取后存储”相比，STONE在问答任务中显著延长检索预算使用时间；在赌博机任务中，深层洞察发现通过ε‑greedy策略获得比单一经验重放更高的累计奖励；在共享记忆实验中，多代理共享显著提升成功率至0.62，只需约一十分之一的单体试错量；

**⚠️ 局限性**

主要局限包括：存储容量膨胀导致的硬件成本、LLM提取时的计算延迟、全面召回的检索效率、隐私与安全风险以及缺乏专门针对跨任务检索的评价基准；

---

## 103. IT-OSE: Exploring Optimal Sample Size for Industrial Data Augmentation

**arXiv ID:** 2602.15878 | [PDF](https://arxiv.org/pdf/2602.15878v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 104. Beyond Learning: A Training-Free Alternative to Model Adaptation

**arXiv ID:** 2602.16189 | [PDF](https://arxiv.org/pdf/2602.16189v1)

**作者:** Namkyung Yoon `[一作]` (Korea University), Hwangnam Kim `[通讯]` (Korea University)

**通讯引用:** 2315 | [OpenAlex ID](https://openalex.org/A5028781455)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过识别语言模型中激活一致的局部模块，并将这些模块直接复制到另一模型中，实现了无训练的模型适配。

**💡 创新点**

提出了“模型移植”概念，证明局部模块可跨模型迁移实现即时性能提升，并引入TIR和Recovery两项评估指标。

**🔧 技术方法**

利用激活对齐分析（LAE）挑选模块，采用直接权重复制的方式替换对应的线性层（注意力投影与前馈层）。

**📊 数据集**

在包含500道跨学科（代数、几何、数论等）数学推理题目的benchmark数据集上进行实验。

**📈 对比分析**

与源模型与目标模型基线比较，TIR最高达2.33，Recovery最高达300%，显示移植可显著提升或超过源模型性能；实验覆盖多种解码长度与迁移规模。

**⚠️ 局限性**

实验仅限于结构兼容的同类模型，未验证不同架构间的迁移；范围局限于数学推理任务；缺乏大规模多任务或跨领域验证。

---

## 105. Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research

**arXiv ID:** 2602.16072 | [PDF](https://arxiv.org/pdf/2602.16072v1)

**作者:** Chenda Duan `[一作]` (University of California Los Angeles), Vwani Roychowdhury `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了Omni-iEEG大规模预手术iEEG数据库，包含302名患者178小时记录，并提供36k+专家标注的病理事件。

**💡 创新点**

创新点在于跨中心数据标准化、统一临床元数据、提供可复现的高质量病理事件注释，并定义临床意义的基准任务与统一评价指标，推动机器学习在癫痫研究中的可重复、可推广应用。

**🔧 技术方法**

使用了多种深度学习技术，包括时域LSTM+注意力、Transformer（PatchTST）、TimesNet、基于事件的PyHFO、端到端的TimeConv-CNN、CLAP音频预训练模型等，以及传统的事件检测算法。

**📊 数据集**

采用了来自八家癫痫中心的公开iEEG数据（Open iEEG、Zurich HFO、Epilepsy Interictal、HUP等），统一按BIDS格式整理。

**📈 对比分析**

在病理事件分类任务中，PyHFO-Omni获得最高F1/召回；在病理脑区识别任务中，端到端TimeConv-CNN和事件基PyHFO-Omni在渠道级AUC相近（≈0.81），并在术后成效预测的AUC达到0.73–0.74。

**⚠️ 局限性**

局限包括注释主观性、未覆盖所有HFO检测/去噪方法、仅关注预手术interictal数据、以及跨中心调优仍需进一步验证。

---

## 106. Genetic Generalized Additive Models

**arXiv ID:** 2602.15877 | [PDF](https://arxiv.org/pdf/2602.15877v1)

**作者:** Kaaustaaub Shankar `[一作]` (University of Cincinnati), Kelly Cohen `[通讯]` (University of Cincinnati)

**通讯引用:** 3249 | [OpenAlex ID](https://openalex.org/A5034113408)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用NSGA-II进化算法自动搜索并优化GAM（Generalized Additive Models）的结构与超参数，以同时最小化预测误差（RMSE）和模型复杂度惩罚，提升可解释性。

**💡 创新点**

首次将多目标进化搜索与GAM结构设计相结合，构造了一个结合稀疏性、平滑度和置信区间宽度的复杂度惩罚项，实现了在准确率与可解释性之间的自动权衡。

**🔧 技术方法**

采用NSGA-II进化框架，基因编码为特征项类型（none/linear/spline）、knots数量、平滑参数λ和特征是否缩放，使用交叉变异、5折交叉验证、置信区间宽度计算与稀疏度比例等指标。

**📊 数据集**

加州住房（California Housing）数据集，包含8个特征和一个连续目标（房价）。

**📈 对比分析**

与基线全 spline LinearGAM（最多25个knots）和无深度限制的决策树比较，评估RMSE与复杂度惩罚。实验表明，RMSE最优的GA模型在所有种子上均优于决策树和基线 LinearGAM；knee模型在RMSE与复杂度之间取得平衡，表现接近或优于基线且复杂度显著降低。

**⚠️ 局限性**

计算成本高——每个个体都需要完整拟合GAM；搜索空间大，易受初始种群和参数设置影响；未对不同不确定性度量进行系统比较；缺乏对模型解释效果的定量评估和跨数据集的验证。

---

## 107. Visual Memory Injection Attacks for Multi-Turn Conversations

**arXiv ID:** 2602.15927 | [PDF](https://arxiv.org/pdf/2602.15927v1)

**作者:** Christian Schlarmann `[一作]` (University of Tübingen), Matthias Hein `[通讯]` (University of Tübingen)

**通讯引用:** 13147 | [OpenAlex ID](https://openalex.org/A5025830560)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在多轮对话中，利用视觉扰动实现的隐蔽目标输出攻击（Visual Memory Injection），并验证其在多款开放权重大型视觉语言模型上的有效性。

**💡 创新点**

创新点在于提出了两项机制：benign anchoring（确保非触发对话中模型正常输出）与 context‑cycling（在优化过程中动态变换对话长度，提升跨轮次的稳健性），从而实现长时隐蔽触发。

**🔧 技术方法**

采用自适应投影梯度下降（APGD）对图像进行扰动优化，同时结合了视觉模型概率目标、anchor 与 trigger 两组 prompt‑output 对，以最大化目标输出概率与保持非触发正常行为的对数似然。

**📊 数据集**

使用了两组图像集（随机采样20张 + 20张不太知名的地标图像）以及三套对话提示集合（随机、主题一致、未参与训练的提示）进行实验，并对多种目标（手机、汽车、政治、股票）分别设计 anchor 与 trigger 对。

**📈 对比分析**

通过目标成功率（trigger 输出正确且非上下文泄漏）与上下文成功率两项指标进行评估。实验显示在 Qwen2.5‑VL‑7B、Qwen3‑VL‑8B 及 LLaVA‑OneVision‑1.5‑8B 上均可达 60–90% 的成功率，且在未见提示、重写提示和微调模型上保持较高转移性能；单轮攻击方案在多轮中快速失效。

**⚠️ 局限性**

主要限制包括：需对基模型具有白盒访问权限；攻击仅针对单一输入图像；针对仅通过 API 提供的模型的攻击尚未探索；对长文本输入和多图像场景的适用性尚未验证。

---

## 108. FUTURE-VLA: Forecasting Unified Trajectories Under Real-time Execution

**arXiv ID:** 2602.15882 | [PDF](https://arxiv.org/pdf/2602.15882v1)

**作者:** Jingjing Fan `[一作]` (Tsinghua University), Zhidong Deng `[通讯]` (Tsinghua University)

**通讯引用:** 2856 | [OpenAlex ID](https://openalex.org/A5102011846)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 FUTURE-VLA 统一架构，能够在单次前向推断中同时生成多视角长时段动作块和潜在空间的未来视觉预览，并通过预测引导的 HIL 机制实现交互式执行门控。

**💡 创新点**

创新点包括：① 双侧效率策略——时空自适应压缩最大化信息密度；② 在同一前向推断中融合动作与未来视觉的自回归生成；③ 基于实时预测的 HIL 机制，提升任务安全性与成功率。

**🔧 技术方法**

技术手段：基于 Qwen3‑VL 语言后端，使用冻结的 DINOv3‑ViT 视觉编码；FAST 频谱动作编码；1D 视觉令牌化（TiTok 风格）实现 32 码的压缩；自回归潜在空间预测；时空自适应压缩模块。

**📊 数据集**

使用的数据集：LIBERO、RoboTwin 2.0、真实世界 Agilex Piper 平台（目标排序、双手交接、桌面清理）以及内部多视角记录。

**📈 对比分析**

与 OpenVLA、WorldVLA、π_0、π_0.5 等基线对比，LIBERO 平均成功率为 91.3%（无 HIL）/99.2%（有 HIL），RoboTwin 75.4%（有 HIL），真实世界 78%（有 HIL）。在长时空窗口扩大 16 倍后仍保持单帧推理时延，显著优于现有方法。

**⚠️ 局限性**

局限性：需预训练模型支持；在极端视觉变化或动态环境中潜在空间预测可能累积误差；压缩比例需手工调优；对硬件性能仍有一定依赖。

---

## 109. AI as Teammate or Tool? A Review of Human-AI Interaction in Decision Support

**arXiv ID:** 2602.15865 | [PDF](https://arxiv.org/pdf/2602.15865v1)

**作者:** Most. Sharmin Sultana Samu `[一作]` (BRAC University), Farig Sadeque `[通讯]` (BRAC University)

**通讯引用:** 340 | [OpenAlex ID](https://openalex.org/A5009105388)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

综述2023-2025人机交互研究，探讨AI是工具还是队友

**💡 创新点**

提出AI成为主动队友需自适应、情境感知的交互与共享心理模型

**🔧 技术方法**

分析解释界面、信任校准、协作框架与LLM辅助决策等技术

**📊 数据集**

复习医疗、金融、执法等多领域的实验与模拟数据集

**📈 对比分析**

通过准确率、信任度、工作负荷等多维指标比较，发现解释丰富提升信任但未提升决策质量，协作框架虽提升准确度但成本增加

**⚠️ 局限性**

仅基于短期实验，缺乏长期纵向数据，接口设计多为静态，领域覆盖有限

---

## 110. Test-Time Adaptation for Tactile-Vision-Language Models

**arXiv ID:** 2602.15873 | [PDF](https://arxiv.org/pdf/2602.15873v1)

**作者:** Chuyang Ye `[一作]` (New York University), Jingyan Jiang `[通讯]` (Shenzhen Technology University)

**通讯引用:** 1087 | [OpenAlex ID](https://openalex.org/A5015538961)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RobustTouch 框架，实现多模态（触觉-视觉-语言）模型的测试时自适应，专门应对异步分布漂移。

**💡 创新点**

创新点包括：1）基于扰动的可靠性指标，用于动态样本过滤；2）可靠性感知的动态模态融合；3）可靠性引导的自监督损失，三者协同提升在异步漂移场景下的鲁棒性。

**🔧 技术方法**

使用技术：扰动扰乱输入、Affinity 函数、熵与置信度变差指标、3σ 自适应阈值、MLP 融合网络、置信度正则化 + 类别平衡损失、CLIP 预训练模型、AdamW 优化。

**📊 数据集**

数据集：TAG‑C benchmark（由 Touch and GO 改造，包含视觉 15 种污染和触觉 7 种污染，每种 5 个严重级别，采用 ImageNet‑C 迁移）。

**📈 对比分析**

与单模态 TTA（TENT、SAR）及多模态 TTA（TDA、READ）对比。在 TAG‑C 的连续域变迁和动态野生场景中，RobustTouch 平均准确率提升 3.8%–6.9%，在最苛刻情况提升 4.1% 点，明显优于基线。

**⚠️ 局限性**

局限性：1）主要关注视觉与触觉两模态，语言模态适配未深入；2）依赖 CLIP 预训练，受限于模型容量；3）对极端批量大小和超参数敏感；4）尚未覆盖多模态同时漂移的复杂场景。

---

## 111. MaS-VQA: A Mask-and-Select Framework for Knowledge-Based Visual Question Answering

**arXiv ID:** 2602.15915 | [PDF](https://arxiv.org/pdf/2602.15915v1)

**作者:** Xianwei Mao `[一作]` (Zhejiang University), Jiajun Bu `[通讯]` (Zhejiang University)

**通讯引用:** 13203 | [OpenAlex ID](https://openalex.org/A5052757755)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于Mask-and-Select的框架MaS-VQA，用于知识驱动的视觉问答；

**💡 创新点**

创新点在于同时对视觉区域和检索文本进行联合精细筛选，再将筛选后的显式知识引导冻结多模态LLM生成隐式知识，从而显著降低噪声并提升推理效果；

**🔧 技术方法**

技术包括多模态检索、知识引导的注意力掩码、问句条件下的短语选择、以及基于冻结MLLM的隐式知识压缩与推理；

**📊 数据集**

使用Encyclopedic-VQA和InfoSeek两个大规模知识型视觉问答数据集进行评测；

**📈 对比分析**

与多种零拷贝MLLM和检索增强模型比较，MaS-VQA在两大数据集上均实现最高或接近最高的准确率，证明了其在多模态知识融合上的优势；

**⚠️ 局限性**

局限性包括对检索质量高度依赖、可能无法完全避免知识库偏见、以及在极度稀疏或缺失外部知识时表现受限。

---

## 112. Do Personality Traits Interfere? Geometric Limitations of Steering in Large Language Models

**arXiv ID:** 2602.15847 | [PDF](https://arxiv.org/pdf/2602.15847v1)

**作者:** Pranav Bhandari `[一作]` (University of Western Australia), Mehwish Nasim `[通讯]` (University of Western Australia)

**通讯引用:** 377 | [OpenAlex ID](https://openalex.org/A5086025266)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了大语言模型中人格特质调控向量的几何关系，并比较了不同正交化策略对调控效果的影响。

**💡 创新点**

首次系统评估人格特质向量的几何独立性，并发现即使强制正交化仍存在跨特质干扰，揭示模型人格表征的耦合子空间。

**🔧 技术方法**

使用激活工程注入向量、Gram矩阵正交化（软硬）、Löwdin 对角化、梯度注入层选择以及 GPT‑4o‑mini 评判技术。

**📊 数据集**

采用标注高低人格的文本数据、Big Five Inventory 问卷以及中性提示生成的文本。

**📈 对比分析**

在 LLaMA‑3‑8B 和 Mistral‑8B 上对比基线、软正交和硬正交条件下的目标特质变化与交叉特质泄漏，结果显示硬正交虽降低交叉泄漏但未提升调控精度且削弱了生成流畅度。

**⚠️ 局限性**

仅评估了两类模型和 Big Five 特质，使用线性正交化方法，评估主要依赖模型评判，未包含更广泛的模型家族、非线性解耦或人工评估。

---

## 113. Feature-based morphological analysis of shape graph data

**arXiv ID:** 2602.16120 | [PDF](https://arxiv.org/pdf/2602.16120v1)

**作者:** Murad Hossen `[一作]` (University of Houston), Nicolas Charon `[通讯]` (University of Houston)

**通讯引用:** 749 | [OpenAlex ID](https://openalex.org/A5064933293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套基于形状图的显式不变特征的统计分析流程，并在分类、聚类等任务中应用。

**💡 创新点**

创新在于设计了兼顾拓扑与几何的19维可解释特征，严格满足刚体变换与分辨率不变性，无需对齐或重采样。

**🔧 技术方法**

采用手工特征提取结合NetworkX、PoreSpy等库，随后使用随机森林、t-SNE、Gromov-Wasserstein等传统机器学习与距离度量。

**📊 数据集**

在三类数据集上验证：城市道路网络、三维神经元形态和星形胶质细胞形态。

**📈 对比分析**

与Gromov-Wasserstein、morphoGNN、SVarM等方法对比，特征基方法在道路和神经元分类/聚类中取得最高或相近的准确率/ARI，星形胶质细胞上优于两者但仍低于神经元。

**⚠️ 局限性**

局限在于特征空间维度有限，对极大样本或细微形态差异的捕捉能力不足；以及需要手工设计特征，难以自动适应新领域。

---

## 114. Transforming GenAI Policy to Prompting Instruction: An RCT of Scalable Prompting Interventions in a CS1 Course

**arXiv ID:** 2602.16033 | [PDF](https://arxiv.org/pdf/2602.16033v1)

**作者:** Ruiwei Xiao `[一作]` (Carnegie Mellon University), John Stamper `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2823 | [OpenAlex ID](https://openalex.org/A5060576109)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开展了一项以ICAP框架为基础的四种教学条件（从被动提醒到主动阅读、选择与写作）的大规模（N=979）随机对照实验，评估在CS1课程中通过提示生成（prompting）训练提升学生的学习型提示能力与学业成绩的关系。

**💡 创新点**

创新点在于：①首次在真实课堂中进行大规模RCTS验证提示教学的有效性；②通过逐步提升认知参与度的设计验证ICAP在提示素养领域的适用性；③提出可扩展、低成本、等效的“选择‑然后‑写作”提示教学模式，兼顾时间投入与学习成效。

**🔧 技术方法**

采用了自我调节学习（SRL）与提示工程（Prompt Engineering）融合的教学框架、基于ICAP的四层次活动设计、LLM驱动的自动评分与即时反馈系统、以及标准化的提示评估量表。

**📊 数据集**

研究数据来自一门北美公立高校的CS1课程，共收集979名学生的随机分组数据，最终保留431名完成全部测评的样本。

**📈 对比分析**

比较方法为配对t检验/Wilcoxon符号秩检验评估学习增益、单因素ANOVA+Tukey后测检验四组间差异、线性回归检验提示得分对期末考试的预测作用。结果显示：①所有组即时与延迟学习增益显著且呈阶梯递增（从0.08到0.51，效应量r>0.85）；②延迟学习增益保留率最高的组（4）为47%；③虽然四组期末考试分数差异不显著，但提示得分每升高1%预测期末成绩提升0.09%。

**⚠️ 局限性**

局限性包括：①研究仅在单门CS1课程、单一机构开展，缺乏跨学科与跨机构的普适性验证；②未通过中介/因果实验验证提示素养提升对学业成绩的因果机制；③仅考察了少数心理特质对效果的调节，未覆盖更广泛的公平性维度。

---

## 115. Non-Contact Physiological Monitoring in Pediatric Intensive Care Units via Adaptive Masking and Self-Supervised Learning

**arXiv ID:** 2602.15967 | [PDF](https://arxiv.org/pdf/2602.15967v1)

**作者:** Mohamed Khalil Ben Salah `[一作]` (University of Quebec), Rita Noumeir `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种自监督的远程光学测脉（rPPG）估计框架，利用Adaptive Masking网络和教师-学生蒸馏，在儿童重症监护室（PICU）视频上实现无接触心率监测。

**💡 创新点**

核心创新在于结合VisionMamba结构的自监督自适应掩码策略和多目标损失，利用Mamba基控制器实现动态空间时间重要性采样，并通过教师模型提供生理先验进行蒸馏，形成三阶段进阶预训练。

**🔧 技术方法**

采用VisionMamba backbone、Gumbel‑TopK可微自适应掩码、policy gradient强化学习、Pixel‑MSE+Pearson相关恢复目标、教师‑学生生理蒸馏、三阶段课程学习、光学脉冲重建与心率回归等技术。

**📊 数据集**

使用公开的UBFC‑rPPG、VIPL‑HR、ECG‑Fitness三大公共数据集做初始预训练，随后在CHU Sainte‑Justine PICU收集的500名儿科患者（30秒RGB视频）进行无标签预训练与有标签微调。

**📈 对比分析**

与多种传统与深度学习方法（CHROM、POS、PhysNet、EfficientPhys、PhysFormer等）在PICU测试集对比，未微调时MAE 10.3 bpm，微调后MAE 3.2 bpm，RMSE 5.4 bpm，相关系数0.91，显著优于PhysFormer 5.8 bpm、MTTS‑CAN 6.5 bpm，提升约45–50%。

**⚠️ 局限性**

局限在于对极端遮挡（>70%）仍存在误差，模型对不同肤色差异仅提升有限，需更大多中心数据验证，且仍依赖大规模GPU训练，部署在嵌入式设备时需进一步优化。

---

## 116. Distributed physics-informed neural networks via domain decomposition for fast flow reconstruction

**arXiv ID:** 2602.15883 | [PDF](https://arxiv.org/pdf/2602.15883v1)

**作者:** Yixiao Qian `[一作]` (Zhejiang University), Shengze Cai `[通讯]` (Zhejiang University)

**通讯引用:** 5231 | [OpenAlex ID](https://openalex.org/A5014873301)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种面向稀疏速度测量的分布式物理信息神经网络（PINN）框架，用于高分辨率流场重构。

**💡 创新点**

创新点在于：① 引入参考锚点归一化与非对称加权策略解决分布式逆问题中的压力自由度；② 利用 CUDA Graph 与 JIT 加速高阶导数的自动微分，显著提升 GPU 计算吞吐量；③ 通过时空域分解实现大规模并行训练，并在多维基准上验证了近线性强伸缩与准确性提升。

**🔧 技术方法**

核心技术包括：物理信息神经网络、时空域分解（XPINN/Parareal 风格）、Ghost 层一致性约束、参考锚点归一化、CUDA Graph 与 JIT 编译、异步通信与权重调度。

**📊 数据集**

使用了三组基准数据集：二维稳态箱腔流（Re=100）、二维瞬态圆柱绕流（Re=100）、三维瞬态圆柱绕流（Re=300），均采用高保真 CFD 生成的速度和压强场作为真值。

**📈 对比分析**

与单域 PINN 及传统插值/数据同化方法对比，分布式 PINN 在保持相同观测点数和训练轮次下显著降低了速度和压强的相对 L² 误差，并在 2×、4×、8× GPU 配置下实现了 1.7–1.9 倍的强伸缩，3D 任务更达 7 倍加速。

**⚠️ 局限性**

局限性包括：① 对极小规模问题计算开销占比高，导致缩放收益有限；② 仍依赖全局观测点与 PDE 采样的事先分配，未解决观测点选择自适应；③ 目前仅针对不可压 Navier–Stokes 及结构化域，扩展到多相、可压或非结构化网格需进一步研究。

---

## 117. Rethinking Soft Compression in Retrieval-Augmented Generation: A Query-Conditioned Selector Perspective

**arXiv ID:** 2602.15856 | [PDF](https://arxiv.org/pdf/2602.15856v1)

**作者:** Yunhao Liu `[一作]` (Fudan University), Yun Xiong `[通讯]` (Fudan University)

**通讯引用:** 3140 | [OpenAlex ID](https://openalex.org/A5001877137)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对检索增强生成（RAG）中软压缩的瓶颈，分析了全压缩的不可行性和不必要性，并提出一种基于查询条件的选择器（selector）软压缩框架，用自回归解码器实现对检索文档的必要信息压缩。

**💡 创新点**

创新点包括：①从理论与实验两方面揭示全压缩导致LLM注意力失衡、信息稀释的根本原因；②将压缩器角色从全量压缩转为查询条件选择器，专门保留与查询相关的关键信息；③设计两阶段训练策略：第一阶段用大规模合成QA数据训练选择器与投影器；第二阶段在公开QA数据上微调生成器，使其有效利用压缩嵌入；④构建覆盖14M问答对的多难度合成数据集，配合梯度学习提升选择器鲁棒性。

**🔧 技术方法**

技术包括：软上下文压缩、解码器自回归选择器、投影器对齐嵌入空间、两阶段训练（选择器+投影器训练、生成器微调）、合成QA数据生成流水线（文档过滤、质量评估、难度分级、LLM评判）、梯度学习。

**📊 数据集**

使用的数据集：合成的14M（查询、文档、答案）对；评估集包括自然问题（Natural Questions）、TriviaQA、Web Questions、PopQA、HotpotQA、FactKG；检索文档来源为Wikipedia dump；训练中使用Qwen3-30B-A3B-Instruct-2507生成合成数据；生成器实验基于Mistral-7B-Instruct、Qwen2.5-7B-Instruct。

**📈 对比分析**

与非压缩RAG、硬压缩（LLMLingua-2、RECOMP）以及软压缩（ICAE、xRAG、COCOM、PISCO）等基线进行对比。实验结果显示本文方法在所有六个知识密集任务上，性能与非压缩RAG持平甚至略优，同时在压缩基线上显著领先；在效率方面，相比非压缩基线，推理延迟和计算量降低33.8%~84.6%，显著提升系统响应速度。

**⚠️ 局限性**

局限性：①仍需依赖大规模合成QA数据，构建成本高；②对检索质量极低或文档极长时，选择器的鲁棒性和压缩效果可能受限；③全压缩问题的分析与改进主要针对当前LLM架构，未来模型结构变化可能需要进一步验证。

---

## 118. Narrative Theory-Driven LLM Methods for Automatic Story Generation and Understanding: A Survey

**arXiv ID:** 2602.15851 | [PDF](https://arxiv.org/pdf/2602.15851v1)

**作者:** David Y. Liu `[一作]` (University of New South Wales), Paul Dawson `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本篇综述性论文系统梳理了自然语言处理（尤其是大型语言模型LLM）在自动故事生成与理解任务中对叙事理论的应用，提出了基于叙事学的三级层级（fabula、discourse、narration）和四类叙事理论（古典、语境、认知、跨媒介）对应的研究范式，并总结了相关数据集、任务与技术趋势；

**💡 创新点**

创新点在于首次从叙事学视角构建统一的分类体系，揭示LLM如何落地叙事理论、任务与技术之间的交互关系，指出缺乏标准化评测与数据集的空白，并为未来跨学科研究提供了具体的方向与方法论；

**🔧 技术方法**

主要技术包括：LLM（GPT‑4、Llama‑3、Claude 等）在生成/理解任务中的prompting、few‑shot、chain‑of‑thought、multi‑agent 框架；针对理解任务的fine‑tuning（LoRA、adapter、distillation）和domain‑adaptive pre‑training；对叙事理论的量化化实现（如Genette、Propp等结构化特征提取、叙事性评分、心理深度评估等）；

**📊 数据集**

使用的主流数据集包括 Reddit、Wikipedia、Project Gutenberg、新闻语料（CNN/NYTimes等）、民俗/童话语料（UCD Irish Folklore、Kaggle Folk Tales Dataset）以及多语言/多领域的自建文本集合，覆盖小说、短篇、非小说、脚本等多种文本类型；

**📈 对比分析**

对比方法多样：人类评测（使用专业写作评分表、TTCW等）、LLM 自评（如GPT‑4 生成评估得分）、参考基准（BLEU、BERTScore、F1 等）以及统计偏差检测（性别/种族偏见）。总体性能呈现混合结果：在理解任务中 fine‑tuned 模型往往优于零/少 shot，但生成任务普遍缺乏一致性评估，LLM 生成质量与人类水平尚有差距，且不同模型/提示策略对结果影响显著；

**⚠️ 局限性**

主要局限包括：缺乏统一的叙事质量基准与评价指标；数据集稀缺且标注成本高，尤其是跨语言与长文本；LLM 生成任务缺乏监督式训练或强化学习方法；评测方法不统一导致跨研究对比困难；模型偏见与文化刻板印象仍难以完全消除；对叙事理论的引用与解释仍较为零散，缺乏系统化的理论验证框架。

---

## 119. Language Model Representations for Efficient Few-Shot Tabular Classification

**arXiv ID:** 2602.15844 | [PDF](https://arxiv.org/pdf/2602.15844v1)

**作者:** Inwon Kang `[一作]` (Rensselaer Polytechnic Institute), Oshani Seneviratne `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 881 | [OpenAlex ID](https://openalex.org/A5038466673)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种利用现成大语言模型（LLM）嵌入进行少样本表格分类的方法，避免训练专用模型或大量微调。

**💡 创新点**

创新点在于：①引入“公共成分去除（CCR）”几何校正，缓解嵌入空间的各向异性；②对软最大温度进行任务特定校准，并用元学习器预测最优温度；③通过这些轻量化改造，直接利用LLM嵌入即可完成分类。

**🔧 技术方法**

技术包括：LLM序列化与嵌入、KNN/核回归分类器、CCR几何校正、温度调参、元学习回归预测温度、缓存加速推理。

**📊 数据集**

使用的主要数据集为CARTE-Binary（11个二分类任务）和TabArena-Binary（28个二分类任务），并在CARTE-CLF、TabArena-CLF等更大规模基准上验证。

**📈 对比分析**

与传统方法（KNN、XGBoost）、专用表格基础模型（TabPFN、ConTextTab）以及其他LLM方法（TabuLa-8B、Llama3.1-8B、Granite3.3等）对比，发现：在k≤32的低样本场景下，本文方法在语义丰富的数据集上能逼近或超过专用模型，且在推理时间上比自回归LLM快约1000倍。

**⚠️ 局限性**

局限性包括：仅针对二分类任务，未充分验证多分类或回归；对非语义化表格效果不如传统模型；温度校准的元学习仍需更多数据和理论支持。

---

## 120. Latent Objective Induction and Diversity-Constrained Selection: Algorithms for Multi-Locale Retrieval Pipelines

**arXiv ID:** 2602.15921 | [PDF](https://arxiv.org/pdf/2602.15921v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Levent Sarioglu `[通讯]` (Bahcesehir University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了多源检索管线中的多算法组合，包括加权地区分配、级联国家码推断、域级多样性约束和隐式目标诱导（LOI）技术，以提升检索结果的多样性和首方来源比例

**💡 创新点**

创新点在于提出可同时满足最小代表、预算完整和比例约束的加权地区分配算法、以优先级链实现确定性且容错的国家码推断、通过哈希映射实现的 O(1) 域级多样性约束，以及通过隐式目标诱导实现的无约束但概率偏好的策略

**🔧 技术方法**

核心技术包括：约束整数划分算法、优先级链推断、哈希表域计数、LLM 环境塑造器 LOI、以及多阶段 LLM pipeline（包括语言检测、研究简报生成、关键词生成、源选择等）

**📊 数据集**

实验使用 120 条多语言查询（土耳其语、英语、德语、阿拉伯语），涵盖地缘、学术、地方/商务、通用四类，评估基于公开搜索 API 的检索结果

**📈 对比分析**

与单域基线、无简报多域配置对比，使用首方来源比例、域重复率、国家码覆盖率和人工相关性评分等指标；结果显示首方来源比例提升 62%，域重复率降低 89%，且 LOI 相比显式约束提升 16% 的首方来源比例并保持更高相关性

**⚠️ 局限性**

局限包括：依赖搜索 API 的源可用性，LLM 对隐式策略理解的依赖，TLD 推断在 CDN 内容上可能失效，且隐式诱导效果受模型规模和语义理解能力影响

---

## 121. LAND: A Longitudinal Analysis of Neuromorphic Datasets

**arXiv ID:** 2602.15973 | [PDF](https://arxiv.org/pdf/2602.15973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 122. Should There be a Teacher In-the-Loop? A Study of Generative AI Personalized Tasks Middle School

**arXiv ID:** 2602.15876 | [PDF](https://arxiv.org/pdf/2602.15876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 123. Decoupling Strategy and Execution in Task-Focused Dialogue via Goal-Oriented Preference Optimization

**arXiv ID:** 2602.15854 | [PDF](https://arxiv.org/pdf/2602.15854v1)

**作者:** Jingyi Xu `[一作]` (NoDesk AI), Zhoupeng Shou `[通讯]` (Zhejiang University)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5113093170)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出GOPO框架，将任务导向对话中的策略规划与回应生成分离，采用层级强化学习实现长周期业务目标的优化。

**💡 创新点**

创新点包括双代理层级RL：专家代理规划策略链并转换为硬约束；引入任务聚焦序列参与度(TSE)指标及与业务最终指标对齐的联合奖励，解决单代理策略不灵活与合规性弱的问题。

**🔧 技术方法**

使用技术包括层级强化学习、策略梯度优化、专家技能归一化折扣累计收益（ESNDCG）评估、GPT‑4自动评估多维奖励、硬约束解码以及动态权重奖励机制。

**📊 数据集**

使用数据集为Mgshop、Multiwoz、TmallBrand‑A和TmallBrand‑B四个公开与内部电商客服数据集。

**📈 对比分析**

通过与SFT、PPO、Memento等传统基线以及Qwen‑235B、DeepSeek‑R1、GLM‑4.7、GPT‑5.2、Gemini‑2.5等大型模型对比，GOPO在TSE、Reward和BLEU等指标上均提升25%–38%、12%–40%等，尤其在7B规模模型上实现与超大模型相近的业务效果。

**⚠️ 局限性**

局限性包括需人工手工定义策略池，奖励设计仍可进一步优化；在跨领域自动策略发现与更复杂约束的泛化能力尚待提升。

---

## 124. Multi-Objective Alignment of Language Models for Personalized Psychotherapy

**arXiv ID:** 2602.16053 | [PDF](https://arxiv.org/pdf/2602.16053v1)

**作者:** Mehrab Beikzadeh `[一作]` (University of California), Saadia Gabriel `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对335名有精神健康经历者进行偏好调查，构建多维度偏好数据集，并基于这些偏好使用多目标直接偏好优化（MODPO）训练语言模型，实现在保持安全性的同时提升同理心、主动倾听、自主改变、信任建立等治疗维度的对话质量。

**💡 创新点**

提出了基于患者偏好的多目标治疗AI对齐框架MODPO，首次系统比较多目标与单目标与参数融合方法，并证明域特定治疗指标优于通用沟通原则，为情感计算与AI伦理提供可复现的数据集与评价流程。

**🔧 技术方法**

采用多目标直接偏好优化（MODPO）、单目标DPO、SFT、参数合并（Soups）、RoBERTa奖励模型、LoRA微调、GPT‑5评估器、模型毒性检测（ModelCitizens、Perspective API）等技术。

**📊 数据集**

使用来自EPITOME子版块的真实对话（2,379问答）作为训练文本，335名受访者的偏好排名及其生成的150个高质量个体化人物用于训练与评估，600个测试问题用于最终对比，临床医生评估作为外部验证。

**📈 对比分析**

通过与GPT‑4o、Base、SFT_Empathy、DPO_Empathy、DPO_Soup、Joint‑Loss DPO等模型进行头对头比较，使用win率、McNemar检验、Fleiss κ等统计指标。MODPO在同理心–安全两目标中达到77.6%同理心、62.6%安全，显著平衡两维度；相比单目标DPO_Empathy（93.6%同理心、47.8%安全）更具整体优势；与通用Grice原则的MODPO_Maxim相比提升17.2个百分点；临床医生验证显示MODPO_Survey获得约70%+偏好，LLM评估与临床一致性接近人类间一致性。

**⚠️ 局限性**

仅在英语西方文化环境中验证，数据量有限（约2,400问答、5条回复）；未评估长期疗效与临床结果；安全仍为软约束，需加入危机检测与硬约束；多目标优化在更大模型或不同基础模型下的效果未知；文化适配与跨语种迁移仍待研究。

---

## 125. Optimization Instability in Autonomous Agentic Workflows for Clinical Symptom Detection

**arXiv ID:** 2602.16037 | [PDF](https://arxiv.org/pdf/2602.16037v1)

**作者:** Cameron Cagan `[一作]` (Massachusetts General Hospital), Hossein Estiri `[通讯]` (Massachusetts General Hospital)

**通讯引用:** 20261 | [OpenAlex ID](https://openalex.org/A5087242254)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了Pythia框架的自我优化流程，并发现了在低样本不平衡条件下会出现的优化不稳定现象；

**💡 创新点**

提出了采用回溯式选择（selector agent）代替主动干预（guiding agent）来缓解此类不稳定，证明后者在低阳性率场景更稳健；

**🔧 技术方法**

利用多代理系统（专属分类器、误差分析代理、合成代理和选择代理）对LLM提示进行迭代改进，并用Llama 3.1 70B模型进行推理；

**📊 数据集**

实验使用400条电子病历（每条标注脑雾、胸痛、呼吸急促三种症状）分为200条开发集与200条验证集；

**📈 对比分析**

与专家构建的词典基线相比，Pythia在脑雾（3%阳性率）上F1提升331%，胸痛提升7%，呼吸急促略低（-5%），验证显示相较词典在低阳性率下表现更佳；

**⚠️ 局限性**

局限包括：仅使用单一病历记录；输入仅为单一自然语言术语；对开发集的F1并不能可靠预测验证集性能；且在单一机构数据上验证，需多中心评估与更丰富的初始化策略。

---

## 126. EdgeNav-QE: QLoRA Quantization and Dynamic Early Exit for LAM-based Navigation on Edge Devices

**arXiv ID:** 2602.15836 | [PDF](https://arxiv.org/pdf/2602.15836v1)

**作者:** Mengyun Liu `[一作]`, Jianan Jiang `[通讯]` (Guangzhou University)

**通讯引用:** 757 | [OpenAlex ID](https://openalex.org/A5112990671)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fede83ac-7505-405f-ab37-e7284695c47f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出EdgeNav-QE框架，将大规模动作模型压缩并自适应推理，以实现边缘设备上实时导航。

**💡 创新点**

创新点在于将4位NF4量化的QLoRA与动态早停（DEE）机制相结合，既显著压缩模型又根据场景复杂度动态调节计算深度。

**🔧 技术方法**

采用QLoRA量化、LoRA适配器、Transformer骨干以及动态早停（DEE）技术。

**📊 数据集**

使用Habitat‑Sim的Matterport3D（ObjectNav）数据集进行主实验，并在Gibson、Replica、HM3D上进行跨环境验证。

**📈 对比分析**

与FP16、INT8 PTQ、DeeR‑VLA等基线对比，EdgeNav‑QE在内存5.4 GB、平均延迟78 ms、成功率81.8%时实现了最优的速度-精度-资源平衡。

**⚠️ 局限性**

主要局限在于早停误判导致约42%的失败率，以及量化引入的精细导航误差和定位不确定性。

---

## 127. MAEB: Massive Audio Embedding Benchmark

**arXiv ID:** 2602.16008 | [PDF](https://arxiv.org/pdf/2602.16008v1)

**作者:** Adnan El Assadi `[一作]` (Carleton University), Kenneth Enevoldsen `[通讯]` (Aarhus University)

**通讯引用:** 182 | [OpenAlex ID](https://openalex.org/A5087151587)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了MAEB（Massive Audio Embedding Benchmark），一个跨100+语言、30个任务、涵盖语音、音乐、环境声、跨模态推理的统一大规模音频嵌入评测框架，并对50+模型进行了系统评估。

**💡 创新点**

创新点包括：①将MTEB生态系统扩展到音频领域，形成跨模态统一评测；②通过任务筛选与Borda计数实现高效评估；③首次在音频嵌入评测中引入大规模多语言、长音频和跨模态零样本任务；④与Audio LLM性能建立相关性验证评测有效性。

**🔧 技术方法**

技术手段包括：多任务评测（分类、零样本分类、聚类、检索、再排序）配合不同模型的嵌入提取方式（均值池化、L2归一化、隐藏层池化）；使用Spearman、Pearson相关分析与Borda计数对模型排名；采用多种预训练方法（自监督语音、对比学习、跨模态对齐、LLM微调）构建模型库。

**📊 数据集**

所用数据集涵盖98个任务，其中30个用于MAEB；包括CommonVoice、VoxPopuli、SIB‑FLEURS、MInDS‑14等语音多语言集；ESC50、FSC‑AI等环境声集；MusicTag、MTG‑JZ、MIR‑Guitar等音乐集；Bioacoustics、Jungle‑Sounds等生物与环境音集；以及跨模态检索与零样本文本‑音频对齐任务。

**📈 对比分析**

评测方法采用GPU小时效率高的MAEB任务集，使用Borda计数与平均分相结合的排名；实验显示无单一模型全局领先，LCO‑Embedding‑Omni‑7B在整体Borda排名第一（平均52.2%），Qwen2‑Audio在音频单项任务上表现最佳；零样本检索最高可达50.3%，但聚类性能仍低（最高22.7%）。

**⚠️ 局限性**

局限性包括：①音频长度限制在30秒，难以评估长音频场景；②语言覆盖虽多，但对低资源与少数族裔语言仍偏弱；③模型规模与算力要求高，普及受限；④聚类与跨模态多语言检索仍表现欠佳；⑤缺乏生成、实时处理与噪声环境下的评测，生态效度有限。

---

## 128. A Decade of Human-Robot Interaction through Immersive Lenses: A Literature Review on Extended Reality as a Research Instrument in Social Robotics

**arXiv ID:** 2602.15840 | [PDF](https://arxiv.org/pdf/2602.15840v1)

**作者:** André Helgert `[一作]` (University of Applied Sciences Ruhr West), Sabrina C. Eimler `[通讯]` (University of Applied Sciences Ruhr West)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究系统性综述了2015-2025年间将XR技术（VR/AR/MR）作为研究工具用于社会机器人交互的实证论文，共筛选33篇文献；

**💡 创新点**

创新点在于对XR在社会HRI中的角色与应用情境进行细致分类，揭示技术与方法瓶颈，并提出五阶段路线图、统一的研究报告与分类框架；

**🔧 技术方法**

采用系统文献综述（PRISMA）框架、结构化编码表及统计分析，评估XR技术、硬件、软件、机器人平台与交互方式等维度；

**📊 数据集**

使用的“数据集”为33篇精选论文的实验设计与报告数据，而非公开实验数据集；

**📈 对比分析**

通过编码和频数统计对比不同研究的XR技术、实验设计、数据收集与分析方法；结果显示多数研究集中在实验室模拟、短任务、主观评估，生态效度与多模态交互表现不足；

**⚠️ 局限性**

局限包括：检索数据库有限、编码过程主观性、部分论文缺失硬件/软件细节、样本偏向WEIRD、AR/MR应用稀缺、长期与多用户研究缺乏。

---

## 129. B-DENSE: Branching For Dense Ensemble Network Learning

**arXiv ID:** 2602.15971 | [PDF](https://arxiv.org/pdf/2602.15971v1)

**作者:** Cherish Puniani `[一作]` (Indian Institute of Technology), Shree Singhi `[通讯]` (Indian Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 B-DENSE 框架，对扩散模型进行多分支轨迹对齐的蒸馏。

**💡 创新点**

通过在学生网络中输出多支通道并在训练时对齐教师完整轨迹，显著减少离散化误差。

**🔧 技术方法**

多分支 U‑Net 结构、DDIM / SFD 蒸馏、概率流 ODE 理论分析、加权损失函数。

**📊 数据集**

使用 CIFAR‑10（32×32）和 ImageNet 64×64 数据集。

**📈 对比分析**

与 Progressive Distillation 和 SFD 基线比较；在低步数（NFE）下 FID 明显提升（如 CIFAR‑10 NFE=2 从 11.96 降至 8.92；ImageNet NFE=2 从 10.25 降至 9.57）。

**⚠️ 局限性**

仍受教师轨迹质量限制，分支权重需手动设定，未验证在更高分辨率或视频/3D 生成场景中的可扩展性。

---

## 130. Reactive Slip Control in Multifingered Grasping: Hybrid Tactile Sensing and Internal-Force Optimization

**arXiv ID:** 2602.16127 | [PDF](https://arxiv.org/pdf/2602.16127v1)

**作者:** Théo Ayral `[一作]` (Université Grenoble Alpes), Mathieu Grossard `[通讯]` (Université Paris-Saclay)

**通讯引用:** 815 | [OpenAlex ID](https://openalex.org/A5109874922)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种混合数据驱动与模型驱动的反应性滑动控制框架，利用压电和电阻式触觉传感器实时构建抓取矩阵，并在其零空间内优化内部力以阻止多指抓取中的滑动。

**💡 创新点**

将高频压电滑动感知与低频电阻式接触定位融合，在线更新抓取模型并通过二次规划求解内部力方向，保持物体力矩不变，仅通过增大摩擦内部力实现稳固抓取。

**🔧 技术方法**

采用混合触觉传感（PzE+PzR）、短时FFT+RNN滑动检测、抓取矩阵与手腕雅可比矩阵的在线更新、二次规划（QP）在内部力零空间内求解、关节电流/张力估计及内部力向量正则化。

**📊 数据集**

使用自建的Festo滑动平台实验数据，对抓取物体施加已知拉力并记录关节电流、触觉信号和位移，未使用公开数据集。

**📈 对比分析**

与不做内部力调节或统一力增大策略对比，在多指抓取实验中单步内部力调整即可在约19–20 mm位移内停止滑动；检测延迟约20 ms，理论闭环延迟35–40 ms。

**⚠️ 局限性**

实验中感知与通信延迟高（100–400 ms）限制了实时性，仅能单步内部力更新；未验证对动态或大接触区域抓取的适用性，且对手的可控性检查和零空间计算存在计算与可实现性限制。

---

## 131. BTReport: A Framework for Brain Tumor Radiology Report Generation with Clinically Relevant Features

**arXiv ID:** 2602.16006 | [PDF](https://arxiv.org/pdf/2602.16006v1)

**作者:** Juampablo E. Heras Rivera `[一作]` (University of Washington), Mehmet Kurt `[通讯]` (University of Washington)

**通讯引用:** 1557 | [OpenAlex ID](https://openalex.org/A5011934020)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 BTReport，一种两阶段脑肿瘤放射报告生成框架，先从多模态 MRI deterministically 提取临床量化特征，再用 LLM 合成符合专业报告风格的文本。

**💡 创新点**

创新点在于：①利用可解释的量化特征（如 VASARI、3D MLS、肿瘤体积等）作为报告输入，避免对 VLM 进行任务特定微调；②设计零样本 3D MLS 估计和肿瘤稳健解剖分割流水线；③通过结构化提示控制 LLM 生成内容，保证报告可追溯且符合临床实际。

**🔧 技术方法**

技术手段包括：SynthMorph + SynthSeg 实现肿瘤稳健解剖分割；Atlas‑based MLS 估计；Vasari‑auto 自动提取 VASARI 特征；gpt‑oss:120B、LLaMA‑3.1 70B 等开源 LLM；RadEval、TBFact、BERTScore、RaTEScore 等自动评测工具。

**📊 数据集**

使用两大数据集：HuskyBrain（184例GBM mpMRI + 放射报告，用于特征验证与报告评估）和 BraTS'23（461例 mpMRI + 预后数据，用于生存预测验证）；并构建 BTReport‑BraTS 作为公开补充数据。

**📈 对比分析**

与 AutoRG‑Brain、Seg‑to‑Exp 等现有 neuro‑oncology RRG 方法对比，BTReport 在 BLEU、ROUGE、TBFact、BERTScore、RaTEScore 等指标上均显著优于基线（p<0.0001），生成报告的词法相似度与事实准确性均提升。

**⚠️ 局限性**

局限性包括：仅基于 mpMRI（缺乏扩散、血管等信息），报告仍依赖分割质量；未覆盖白质高信号、基底池等额外特征；未结合放射科医生的临床评审；目前仅测试两种 LLM，缺乏对更多模型的泛化验证。

---

## 132. HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents

**arXiv ID:** 2602.16165 | [PDF](https://arxiv.org/pdf/2602.16165v1)

**作者:** Jiangweizhi Peng `[一作]` (University of Minnesota), Mingyi Hong `[通讯]` (University of Minnesota)

**通讯引用:** 14611 | [OpenAlex ID](https://openalex.org/A5100633783)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 HiPER 框架，将 LLM 代理的高层规划与低层执行分离，并通过 Plan-Execute 接口实现结构化输出。

**💡 创新点**

创新点在于提出层次化优势估计（Hierarchical Advantage Estimation, HAE），将信用分配与子目标、切换决策的两层结构对齐，并显式化子目标生成与执行。

**🔧 技术方法**

技术手段包括基于选项框架的层次化策略、Plan-Execute 结构化提示、PPO 风格的双层价值函数与 HAE，以及自回归 LLM 生成子目标与原子动作。

**📊 数据集**

评估使用了 ALFWorld 和 WebShop 两个交互式文本环境，分别涵盖多任务与购物类场景。

**📈 对比分析**

与 PPO、RLOO、GRPO、GiGPO 等基线比较，HiPER 在 ALFWorld 上取得 97.4%/83.3% 的最高成功率，在 WebShop 上亦超过 70% 的成绩，并显著提升采样效率和训练稳定性。

**⚠️ 局限性**

局限性包括对 Prompt 设计的高度依赖、对大模型规模的需求，以及在极端稀疏奖励场景下仍需改进信用分配与探索机制。

---

## 133. ScenicRules: An Autonomous Driving Benchmark with Multi-Objective Specifications and Abstract Scenarios

**arXiv ID:** 2602.16073 | [PDF](https://arxiv.org/pdf/2602.16073v1)

**作者:** Kevin Kai-Chun Chang `[一作]` (University of California), Sanjit A. Seshia `[通讯]` (University of California)

**通讯引用:** 14803 | [OpenAlex ID](https://openalex.org/A5064230639)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ScenicRules 基准，用以评估自动驾驶系统在多目标优先级规范下的性能。基准通过对 19 条驾驶目标进行形式化、构建层级 Rulebook、使用 Scenic 语言生成抽象可随机化的场景，并通过 k‑center 选取代表性场景以及 LLM 辅助生成近事故场景；此外还给出了基于 Rulebook 的违例度量与模拟基 falsification 流程。

**💡 创新点**

创新点主要包括：①将多目标优先级规则与抽象场景（Scenic）结合，首次提供兼具可解释性与可扩展性的评测框架；②提出层级 Rulebook 结构，使得优先级可在规则组层面灵活调整，既可解释又易于适配不同驾驶情境；③利用 coreset 选择（k‑center 贪心）方法压缩庞大场景空间，保证覆盖度；④采用 LLM（Gemini 2.5 Flash）自动生成基于真实事故报告的近事故场景，丰富评测难度。

**🔧 技术方法**

使用技术包括 Signal Temporal Logic (STL) 与鲁棒度量、Rulebook 与层级 Rulebook 架构、Scenic 领域专用语言、k‑center 贪心核心集选择、LLM 文本到 Scenic 的自动翻译、模拟与 falsification（VerifAI、MetaDrive/CARLA）以及参数与优先级的贪心优化。

**📊 数据集**

主要使用数据集：Reasonable Crowd（人类偏好标注的轨迹对）、加州 DMV 事故报告（用于生成近事故场景）以及 MetaDrive 内置地图与仿真环境。

**📈 对比分析**

对比方法：将层级 Rulebook 的人类偏好匹配准确率与随机森林、逻辑回归、Rulebook+决策树等机器学习基线进行比较，准确率约 80–82%。参数优化后提升至 81–85%。在 falsification 测试中，基准在常规与近事故场景下对 PPO 与规则驱动规划器均能产生高误差值和 95% 以上的 Counterexample 比例，验证了其覆盖性与发现失败模式的能力。

**⚠️ 局限性**

局限性：规则的形式化细节对评估结果影响显著，当前仅包含 19 条规则且缺少多车交互目标；层级 Rulebook 的组间顺序仍需人工或数据驱动选择，无法完全覆盖所有场景偏好；LLM 生成的近事故场景可能存在误差；实验主要基于 MetaDrive，未在多种仿真器或真实车辆上验证；人类偏好数据规模有限，难以覆盖全部驾驶语义。

---

## 134. How Uncertain Is the Grade? A Benchmark of Uncertainty Metrics for LLM-Based Automatic Assessment

**arXiv ID:** 2602.16039 | [PDF](https://arxiv.org/pdf/2602.16039v1)

**作者:** Hang Li `[一作]` (Michigan State University), Jiliang Tang `[通讯]` (Michigan State University)

**通讯引用:** 25224 | [OpenAlex ID](https://openalex.org/A5040639891)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基于大型语言模型（LLM）的自动评估系统中的不确定性进行系统评估，构建了一个涵盖多种模型、数据集和提示策略的基准；

**💡 创新点**

首次从分类与关系两大视角对多种不确定性量化方法进行比较，并揭示分类方法在评估准确性上表现更好，而关系方法在样本波动稳定性上更强；

**🔧 技术方法**

利用重复生成、类别频率统计（如Numset、MAR、CE、FSD）、关系图构建（Jaccard、嵌入、NLI）及其图属性（NAD、GE、Eigen、DSE）等技术；

**📊 数据集**

使用14个评估问题，包含两份公开数据集（ASAP、SemEval-13-T7）与三份私有数据集（化学、地球科学、教育学）；

**📈 对比分析**

通过AUROC、C-index、AUARC、AUERC四个指标对比，分类方法CE、FSD、MAR、Numset普遍排名靠前；关系方法中JS_NAD/JS_GE表现稳定，但整体效果低于分类方法；

**⚠️ 局限性**

缺点在于对模型性能高度依赖，关系方法需额外计算资源；不同模型、问题对方法排名影响较大，未提供统一最优方案。

---

## 135. Nash-convergence of Game Dynamics and Complexity

**arXiv ID:** 2602.16016 | [PDF](https://arxiv.org/pdf/2602.16016v1)

**作者:** Oliver Biggar `[一作]`, Georgios Piliouras `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

无法确定论文内容，请提供论文摘要或主要内容。

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 136. Collaborative Zone-Adaptive Zero-Day Intrusion Detection for IoBT

**arXiv ID:** 2602.16098 | [PDF](https://arxiv.org/pdf/2602.16098v1)

**作者:** Amirmohammad Pasdar `[一作]` (University of Melbourne), Van-Thuan Pham `[通讯]` (University of Melbourne)

**通讯引用:** 2228 | [OpenAlex ID](https://openalex.org/A5056177929)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为Zone-Adaptive Intrusion Detection (ZAID)的协作检测框架，旨在应对IoBT环境中的零日攻击，利用局部适应和协作学习来提高检测能力。

**💡 创新点**

创新点在于结合了通用卷积模型、基于自编码器的重构信号和轻量级适配器模块，实现了参数高效的区域适应，支持跨区域的协作学习。

**🔧 技术方法**

使用了卷积神经网络（CNN）作为通用模型和自编码器作为辅助异常评分，结合了联邦学习和伪标记技术。

**📊 数据集**

在ToN_IoT数据集上进行评估，采用零日协议排除了MITM、DDoS和DoS攻击，并在区域级部署和适应中引入这些攻击。

**📈 对比分析**

ZAID在未见攻击流量上达到了83.16%的准确率，并在UNSW-NB15数据集上转移时达到了71.64%的最佳准确率，显示出其在对抗IoBT环境中未见攻击的有效性。

**⚠️ 局限性**

局限性包括数据集和场景的代表性不足，伪标记噪声和阈值敏感性，以及联邦学习假设和对抗鲁棒性的问题。

---

## 137. Surgical Activation Steering via Generative Causal Mediation

**arXiv ID:** 2602.16080 | [PDF](https://arxiv.org/pdf/2602.16080v1)

**作者:** Aruna Sankaranarayanan `[一作]` (Massachusetts Institute of Technology), Dylan Hadfield-Menell `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1403 | [OpenAlex ID](https://openalex.org/A5076757561)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Generative Causal Mediation (GCM) 方法，利用长文本响应对比定位并激活注意力头实现语言模型的行为控制。

**💡 创新点**

将因果中介分析应用于长文本生成，提供激活修补、归因修补和注意力头淘汰三种实现变体，实现对多层注意力头的精准定位与干预。

**🔧 技术方法**

采用激活修补、梯度归因修补、差分均值与均值激活向量、Representation Fine‑Tuning（ReFT）等技术，结合线性探针进行头部重要性排序。

**📊 数据集**

构造三类对比提示集（拒绝、附和、诗体转化），在Qwen‑14B、OLMo‑13B、SOLAR‑10.7B三大模型上进行实验。

**📈 对比分析**

与随机选择、Inference‑Time Interventions（线性探针）等基线比较，GCM在大部分任务和模型上平均成功率超过80%，显著优于基线。

**⚠️ 局限性**

局限性包括：在SOLAR模型的拒绝任务上表现不佳；对外域数据迁移效果有限；仅针对二元对比概念，尚未扩展到多概念或更细粒度控制。

---

## 138. DARTH-PUM: A Hybrid Processing-Using-Memory Architecture

**arXiv ID:** 2602.16075 | [PDF](https://arxiv.org/pdf/2602.16075v1)

**作者:** Ryan Wong `[一作]` (University of Illinois Urbana-Champaign), Saugata Ghose `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 6110 | [OpenAlex ID](https://openalex.org/A5036666743)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种混合模拟‑数字处理内存架构 DARTH-PUM，可在单芯片上高效执行 AES 加密、CNN 推理和 LLM 编码器等多种应用。

**💡 创新点**

创新点包括：将模拟 PUM 的高效 MVM 与数字 PUM 的通用计算能力无缝结合；提供统一 ISA 与软硬协同的接口；通过位移单元、转置单元、指令注入器、虚拟模拟核心等硬件协同，实现无专用功能单元的通用加速；对 ADC 选择与误差补偿进行系统化设计。

**🔧 技术方法**

核心技术包括 ReRAM 内存、模拟 MVM、数字 Boolean PUM（OSCAR/RACTER pipelines）、SAR / ramp ADC、bit‑slicing、虚拟模拟核心、指令注入器、转置单元、错误补偿、ISA 与编译器支持。

**📊 数据集**

使用的数据集/工作负载：AES‑128 加密、ResNet‑20 在 CIFAR‑10 上的推理、以及 Transformer‑style LLM 编码器（公开的 LLM 训练/推理数据）。

**📈 对比分析**

方法：面积等价对比 Baseline（CPU+模拟 PUM）、单独数字 PUM、应用专用加速器（AppAccel）以及 NVIDIA RTX 4090 GPU；结果显示 DARTH-PUM 对 AES、ResNet‑20、LLM 分别提升 59.4×、14.8×、40.8×，平均提升 31.4×；能耗平均降低 66.8×；相对 GPU 吞吐提升约 11.8×、能耗提升约 7.5×。

**⚠️ 局限性**

局限性：非 MVM 操作仍占大比例，导致 LLM 中性能无法与专用加速器相匹配；对 ADC 选择和噪声/非理想性敏感，需额外补偿；需要专门的编译器/软件层支持；在极大位宽或动态更新矩阵场景下仍有额外开销。

---

## 139. MedProbCLIP: Probabilistic Adaptation of Vision-Language Foundation Model for Reliable Radiograph-Report Retrieval

**arXiv ID:** 2602.16019 | [PDF](https://arxiv.org/pdf/2602.16019v1)

**作者:** Ahmad Elallaf `[一作]` (Texas A&M University), Gongbo Liang `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

建立了MedProbCLIP概率对比学习框架，用于胸部X光与放射报告的跨模态检索。

**💡 创新点**

通过将图像和文本编码成高斯分布来显式建模不确定性与多对多对应，并结合多视角和多段报告进行联合训练。

**🔧 技术方法**

采用变分信息瓶颈、概率对比损失（CSD）、多模态双编码器（ViT + BioMedBERT）、KL正则化、混合精度训练等技术。

**📊 数据集**

使用MIMIC‑CXR大规模胸部X光与报告数据集。

**📈 对比分析**

与CLIP、PCME++、CXR‑CLIP等基线在相同框架、相同数据、相同训练设置下对比，MedProbCLIP在检索Recall@K、RSUM、零样本分类准确率等指标均明显优于基线，提升幅度为R@1提升约3–7个百分点。

**⚠️ 局限性**

在极端图像损伤（如高噪声）下仍有局限；需要精细调节方差正则化；当监督信息足够清晰时无明显优势；模型训练复杂度和推理时分布估计开销较大。

---

## 140. Universally Optimal Decremental Tree Minima

**arXiv ID:** 2602.15977 | [PDF](https://arxiv.org/pdf/2602.15977v1)

**作者:** Benjamin Aram Berendsohn `[一作]` `[通讯]` (Max Planck Institute for Informatics), Benjamin Aram Berendsohn (Max Planck Institute for Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文扩展了通用最优性概念到数据结构，研究了动态图问题：在给定的顶点加权森林中，维护每棵树的最小权重顶点，支持边删除操作。

**💡 创新点**

提出了一种在所有固定初始森林和固定操作/查询数量下具有最优总运行时间的数据结构，适用于所有权重分配和操作序列的最坏情况。

**🔧 技术方法**

结合了两种技术：1) 将输入分解为路径，以利用特定路径的数据结构实现O(1)时间；2) 使用splay树来优化处理某些排序相关的子问题。

**📊 数据集**

使用了顶点加权森林作为数据集，且可以扩展到支持边权重和半群求和查询。

**📈 对比分析**

与现有方法比较，本文提出的数据结构在每个固定初始森林和操作数量下都能达到最优性能，且在特定情况下（如初始森林为路径）能实现O(1)时间的操作。

**⚠️ 局限性**

限制在于该数据结构的通用最优性定义可能不适用于所有动态图问题，且在某些情况下可能无法达到实例最优性。

---

## 141. Evidence-Grounded Subspecialty Reasoning: Evaluating a Curated Clinical Intelligence Layer on the 2025 Endocrinology Board-Style Examination

**arXiv ID:** 2602.16050 | [PDF](https://arxiv.org/pdf/2602.16050v1)

**作者:** Amir Hosseinian `[一作]` (January AI), Nima Aghaeepour `[通讯]` (Stanford University)

**通讯引用:** 9226 | [OpenAlex ID](https://openalex.org/A5031425035)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究在ESAP 2025内分泌科执业考试中，评估了一套以领域专属证据为基础的临床推理系统。

**💡 创新点**

创新点在于构建多层次、按循证等级分级的证据库，并通过结构化推理栈实现可追溯、可信的答案生成，且在无外部检索的条件下优于大型模型。

**🔧 技术方法**

采用了手工策划的指南、同行评议文献和关键RCT三层证据语料库，结合多专门化推理模块和仲裁层的ensemble推理架构。

**📊 数据集**

使用的主要数据集是由Endocrine Society提供的ESAP 2025考试题集，共120道多项选择题。

**📈 对比分析**

在零样本评估中，将该系统与GPT‑5、GPT‑5.2、Gemini‑3‑Pro进行对比，单答准确率为87.5%（远高于74.6%、74.0%和69.8%），Top‑2准确率92.5%（高于85.25%、81.1%和83.59%）。

**⚠️ 局限性**

局限性包括仅验证单一专科、考试情境与真实临床存在差异、比较条件混杂（检索方式与模型规模不同）、ESAP题集无法公开、以及结果受US中心化指南影响。

---

## 142. A novel Integrated Motion Tracking Device (IMTD) for Objective Laparoscopic Training Assessment: Development and Validation

**arXiv ID:** 2602.15885 | [PDF](https://arxiv.org/pdf/2602.15885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 143. A Methodology for Identifying Evaluation Items for Practical Dialogue Systems Based on Business-Dialogue System Alignment Models

**arXiv ID:** 2602.15835 | [PDF](https://arxiv.org/pdf/2602.15835v1)

**作者:** Mikio Nakano `[一作]` (C4A Research Institute, Inc.), Kazunori Komatani `[通讯]` (Osaka University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文提出了一种基于业务-对话系统对齐模型的评价项识别方法，用以构建实际对话系统的全面评价框架。

**💡 创新点**

创新点在于将业务-IT对齐模型迁移到对话系统领域，定义了通用的价值、风险、成本、服务四个子模型，并用ArchiMate语言实现可视化对齐模型。

**🔧 技术方法**

主要技术是业务-IT对齐模型的构建、ArchiMate建模与应用、以及对AI原则的风险映射。

**📊 数据集**

由于是方法论研究，未使用特定数据集；仅在FAQ问答机器人案例中演示模型构建。

**📈 对比分析**

论文未给出实验比较或性能指标，仅通过案例说明方法可识别多维评价项，未实现量化评估。

**⚠️ 局限性**

局限性包括缺乏量化评估、案例数量有限、对业务-IT对齐模型不熟悉的工程师难以快速构建，以及技术快速演进导致新风险难以捕捉。

---

## 144. Why Any-Order Autoregressive Models Need Two-Stream Attention: A Structural-Semantic Tradeoff

**arXiv ID:** 2602.16092 | [PDF](https://arxiv.org/pdf/2602.16092v1)

**作者:** Patrick Pynadath `[一作]` (Purdue University), Ruqi Zhang `[通讯]` (Purdue University)

**通讯引用:** 210 | [OpenAlex ID](https://openalex.org/A5046946005)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究任何顺序自回归模型中单流注意力面临的结构-语义权衡，并提出 Decoupled RoPE 以仅在单流中解耦位置与内容。

**💡 创新点**

揭示两流注意力的成功并非仅因分离位置与内容，而是因为它绕过了单流自回归中固有的结构-语义冲突。

**🔧 技术方法**

使用旋转位置编码（RoPE）的改进版本 Decoupled RoPE，结合任何顺序自回归训练目标，并对比 Masked Diffusion 的双向编码器。

**📊 数据集**

在字符级文本数据集 text8 上进行训练与评估。

**📈 对比分析**

通过 coherence‑diversity 前沿和验证 NLL 进行比较：在短序列长度（128）下 Decoupled RoPE 与 Masked Diffusion 接近；在长序列长度（1024）下 Masked Diffusion 显著优于 Decoupled RoPE。

**⚠️ 局限性**

实验规模有限，仅在字符级数据集上验证；未直接评估两流注意力，缺乏正式证明结构‑语义权衡的理论支持。

---

## 145. ReasonNavi: Human-Inspired Global Map Reasoning for Zero-Shot Embodied Navigation

**arXiv ID:** 2602.15864 | [PDF](https://arxiv.org/pdf/2602.15864v1)

**作者:** Yuzhuo Ao `[一作]` (Hong Kong University of Science and Technology), Chi-Keung Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 13044 | [OpenAlex ID](https://openalex.org/A5062566088)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 ReasonNavi 框架，利用多模态大型语言模型（MLLM）进行全局推理，再结合确定性路径规划器完成零样本目标导航；

**💡 创新点**

创新点在于将人类“先推理后行动”的思路拆分为离散化目标点选择，采用多阶段（房间定位+节点挑选）与模型集成方式，使 MLLM 只需一次全局推理，避免连续坐标预测难题；

**🔧 技术方法**

技术核心包括 MLLM 推理（Seed‑1.6‑Thinking、Gemini‑2.5‑Pro、GPT‑5），Poisson Disk Sampling 与 Watershed 进行房间分割与候选点生成，A*+VFH* 确定性规划，MobileSAM 用于目标精确分割，VGGT 用于生成二维地图；

**📊 数据集**

主要数据集为 Habitat‑Matterport 3D（HM3D）2022/2023 的 Object‑Goal、Image‑Goal 与 Text‑Goal 任务，同时在多楼层和多人协作场景下进行验证；

**📈 对比分析**

与多种基准方法（RL、构建式、其他 MLLM 基础方法）比较，ReasonNavi 在 Obj‑Nav SPL 最高 31.4、Img‑Nav SPL 30.4、Text‑Nav SPL 24.3，且在 SR/SPL 上均优于现有技术；

**⚠️ 局限性**

局限性在于需要预先获取完整的二维全局地图，对地图质量依赖较大，且在高度动态或多目标环境下的适用性尚待进一步验证。

---

## 146. Access in the Shadow of Ableism: An Autoethnography of a Blind Student's Higher Education Experience in China

**arXiv ID:** 2602.16070 | [PDF](https://arxiv.org/pdf/2602.16070v1)

**作者:** Weijun Zhang `[一作]` (Syracuse University), Xinru Tang `[通讯]` (University of California, Irvine)

**通讯引用:** 609 | [OpenAlex ID](https://openalex.org/A5042585285)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过自我民族志方法，对一名中国盲学生在专门的盲生教育项目与主流高校的学习经历进行对比，揭示了在不同教育环境中所面临的可访问性矛盾与压力。

**💡 创新点**

创新点在于将自我民族志与批判性残障研究相结合，首次系统探讨中国高等教育中“可访问性”被置于更广阔的“能够主义”结构下的多重张力，并提出将可访问性视为持续探索而非单一解决方案的概念。

**🔧 技术方法**

使用的技术主要是反思性主题分析（reflexive thematic analysis）对自我叙事进行编码与归纳，并通过对比分析揭示两种教育模式的差异。

**📊 数据集**

数据集由作者本人（Zhang）在2023年12月回顾整理的20余条叙事片段组成，涵盖其在两所高校的招生、教学、技术、考试等方面的亲身经历。

**📈 对比分析**

方法上以案例对比为核心，未涉及定量性能指标，而是通过质性分析展示在两种机构中可访问性的实现程度和存在的系统性障碍；结果显示两种模式均受制于制度缺陷与“能够主义”文化，导致学生必须进行自我倡导与权宜之计。

**⚠️ 局限性**

局限性包括：样本单一（仅一位盲学生的经历），回忆偏差与主观情绪可能影响数据准确性，缺乏可推广的定量评估，且对中国特定文化与制度背景的深度解释仍待进一步验证。

---

## 147. Doc-to-LoRA: Learning to Instantly Internalize Contexts

**arXiv ID:** 2602.15902 | [PDF](https://arxiv.org/pdf/2602.15902v1)

**作者:** Rujikorn Charakorn `[一作]` (Sakana AI), Robert Tjarko Lange `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量级超网络D2L，用于即时内部化长上下文，使LLM在无上下文的情况下回答问题。

**💡 创新点**

创新点在于将上下文蒸馏过程元学习为单前向超网络，并使用Perceiver架构和分块机制处理变长输入。

**🔧 技术方法**

采用LoRA适配器、Perceiver超网络、分块处理以及查询独立的上下文蒸馏训练。

**📊 数据集**

使用合成Needle-in-a-Haystack、六个真实QA基准（SQuAD、DROP、ROPES、LongBench系列等）以及Imagenette进行视觉信息实验。

**📈 对比分析**

与传统上下文蒸馏、基于查询生成的CD、T2L、LLMLingua-2等方法比较，D2L在有限计算下实现近乎ICL的准确率，同时更新延迟和内存使用降低数倍，甚至在长文本和视觉任务中实现零样本内部化。

**⚠️ 局限性**

局限性包括对超网络训练数据的依赖、对查询分布的偏倚，以及在极长上下文或多模态跨域的泛化仍有限。

---

## 148. The Impact of Class Uncertainty Propagation in Perception-Based Motion Planning

**arXiv ID:** 2602.16035 | [PDF](https://arxiv.org/pdf/2602.16035v1)

**作者:** Jibran Iqbal Shah `[一作]` (University of Toronto), Florian Shkurti `[通讯]` (University of Toronto)

**通讯引用:** 1544 | [OpenAlex ID](https://openalex.org/A5010648258)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了感知不确定性在自动驾驶运动规划中的影响，提出了一种新型的可校准不确定性感知随机MPC（UA‑SMPC），并与两种预测模型（HAICU和Trajectron++）在nuPlan闭环基准上进行系统对比。

**💡 创新点**

创新点包括：① UA‑SMPC引入精确的概率碰撞避免约束，能够直接处理多模态GMM预测；② 通过对预测模型的校准程度进行定量比较，揭示分布质量（NLL/Entropy）比点误差（minADE/minFDE）更能决定规划性能；③ 对预测协方差缩放进行敏感度分析，验证不同模型对不确定性校准的响应差异。

**🔧 技术方法**

技术手段：使用条件变分自编码器/图神经网络生成多模态GMM轨迹预测；将class/状态不确定性融合进预测；构建基于Mahalanobis距离的精确碰撞回避约束；利用CasADi与IPOPT求解非凸SMPC优化；对不同协方差尺度进行实验。

**📊 数据集**

数据集：PUP数据集（含类不确定性）用于训练预测模型；nuPlan规划基准用于闭环评估，包含超过150个时刻的真实轨迹和道路障碍信息。

**📈 对比分析**

比较方法：在nuPlan闭环场景下对HAICU+UA‑SMPC、Trajectron+++UA‑SMPC和恒定速度基线进行多指标评估（进展、冲击、TTC、闭环得分）。结果显示：短期（3s）HAICU在TTC和闭环得分上优于Trajectron++，但Trajectron++在进展上略胜；长期（8s）HAICU+UA‑SMPC在所有指标上（尤其是闭环得分）明显优于其他两种方案。

**⚠️ 局限性**

局限性：① 预测模型训练仅基于PUP，未在nuPlan上直接训练，导致在nuPlan场景中缺乏类不确定性信息；② 缺乏对其他不确定性传播方法（如PSU‑TF）的系统比较；③ HAICU对协方差缩放高度敏感，表明校准质量对规划结果的影响显著；④ 仅在二维平面上评估，未考虑纵向动力学等更复杂因素。

---

## 149. Missing-by-Design: Certifiable Modality Deletion for Revocable Multimodal Sentiment Analysis

**arXiv ID:** 2602.16144 | [PDF](https://arxiv.org/pdf/2602.16144v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11901 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种可对多模态情感分析模型执行可验证模态级删除的完整框架 Missing-by-Design (MBD)，实现对用户请求的模态信息撤销。

**💡 创新点**

创新点在于：① 用属性嵌入分离样本特定与模态固定信息；② 结合生成器和对比回译实现高质量缺失模态重建；③ 引入 SwiftPrune 启发的数值稳定重要性代理 + 模态显著性阈值进行参数 surgery；④ 通过 Gaussian 机制校准，实现 (ε,δ)-可辨识度保证，并生成机器可验证的 Modality Deletion Certificate。

**🔧 技术方法**

技术手段包括：属性嵌入分解、生成式重建网络、对比式正则化、SwiftPrune 重要性代理、模态显著性梯度、加性高斯噪声、zCDP 到 (ε,δ) 的转换、证书生成与 SHA‑256 校验。

**📊 数据集**

在 CMU‑MOSI、CMU‑MOSEI 和 IEMOCAP 三大公开多模态情感基准上进行评估，涵盖文本、音频、视觉三种模态。

**📈 对比分析**

与多种现有基线（GCNet、MoMKE、UniMSE、HyCon 等）对比，MBD 在完整模态下平均提升 1–2% 主要指标；在固定缺失模态和全局缺失率不同的设置下均显著优于其他方法；消除音频模态时，在 ε≤1 的隐私预算下 ASR 降至与随机相近，而二分类准确率仅损失 1–1.5%；ablation 证明属性嵌入与重建模块贡献最大。

**⚠️ 局限性**

局限性包括：① 数据集缺乏族群、方言等多样性标签，无法评估在弱势群体中的公平性；② 现行的可辨识度保证主要基于攻击测试，可能无法抵御更强或跨域攻击；③ 证书可被重放或权重恢复，需硬件信任或区块链校验；④ 监管合规性仍需进一步验证，模型更新后需重新检查；⑤ 对于大模型和多模态组合的理论边界尚未收敛。

---

## 150. EarthSpatialBench: Benchmarking Spatial Reasoning Capabilities of Multimodal LLMs on Earth Imagery

**arXiv ID:** 2602.15918 | [PDF](https://arxiv.org/pdf/2602.15918v1)

**作者:** Zelin Xu `[一作]` (University of Florida), Zhe Jiang `[通讯]` (University of Florida)

**通讯引用:** 2696 | [OpenAlex ID](https://openalex.org/A5102906971)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们创建了 EarthSpatialBench 基准，包含 325K 个问答对，评估多模态大型语言模型在地球影像中的距离、方向、拓扑等空间推理任务。

**💡 创新点**

创新点在于同时引入三种几何类型（bbox、polygon、polyline）和三种对象引用方式（文本、视觉覆盖、坐标），并覆盖定量与定性问题，系统化评估 MLLMs 的空间推理能力。

**🔧 技术方法**

采用 GIS 计算、模板化问题生成、链式思维提示以及多模态预训练模型（Qwen3‑VL、InternVL、GPT‑5、Gemini、Claude）等技术。

**📊 数据集**

使用 SatlasPretrain 高分辨率光学影像 21K 张（1m 分辨率）及其几何注释作为数据源，并结合多模态 LLM 的训练与评估数据。

**📈 对比分析**

通过 MAE/RMSE、准确率、F1、IoU 等指标对 8 个模型进行评估，结果显示闭源模型在分类任务上优势明显，但在定位和定量推理方面仍显不足，整体存在显著性能差距。

**⚠️ 局限性**

主要局限在于视觉定位误差、文本描述模糊导致的锚定困难、几何类型异质性导致的推理误差，链式思维对定位帮助有限，模型对多模态几何输入的整合能力不足。

---

## 151. FeDecider: An LLM-Based Framework for Federated Cross-Domain Recommendation

**arXiv ID:** 2602.16034 | [PDF](https://arxiv.org/pdf/2602.16034v1)

**作者:** Xinrui He `[一作]` (University of Illinois Urbana-Champaign), Jingrui He `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4192 | [OpenAlex ID](https://openalex.org/A5073158087)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于大语言模型的联邦跨域推荐框架 FeDecider，解决了域异构导致的过拟合与隐式表示难以匹配的问题。

**💡 创新点**

创新点在于：①通过服务器端对 LoRA 更新进行方向分解，只传递方向信息以消除尺度噪声；②在客户端采用可学习的个性化权重，对接收到的方向信息进行数据感知的融合，实现细粒度知识共享与自适应。

**🔧 技术方法**

核心技术包括：LLM（IDGenRec）作为基础模型、LoRA 参数高效微调、方向分解（DoRA 风格）与个性化权重学习。

**📊 数据集**

使用了 Goodreads（Crime、Comics、Children）和 Amazon（Beauty、Clothing、Electronics/Phones）三组跨域数据集。

**📈 对比分析**

与 FedAvg、PFedAvg、FedProx、Ditto、FFA-LoRA、RoLoRA、FellRec、FDLoRA 等传统与 LoRA 基线比较，FeDecider 在 Hit@5、NDCG@10 等指标上均优于所有基线，尤其在多域共识和长尾推荐上表现突出。

**⚠️ 局限性**

局限性包括：需要一定的客户端算力来维护 LoRA 与权重参数；通信成本略高于单纯 LoRA 聚合；目前仅在文本域实验，尚未验证多模态（图像+文本）场景。

---

## 152. The Limits of Long-Context Reasoning in Automated Bug Fixing

**arXiv ID:** 2602.16069 | [PDF](https://arxiv.org/pdf/2602.16069v1)

**作者:** Ravi Raju `[一作]` (SambaNova Systems), Urmish Thakker `[通讯]` (SambaNova Systems)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估当前LLM在长上下文代码调试与补丁生成中的能力，尤其在SWE‑Bench Verified场景下的表现

**💡 创新点**

揭示agentic工作流并不能真正测试长上下文推理，并系统分析LLM在64k–128k上下文下的失败模式和hallucination

**🔧 技术方法**

使用mini‑SWE‑agent、BM25+golden‑patch检索（RAG）以及单次提示实验

**📊 数据集**

SWE‑Bench Verified、SWE‑Bench Lite

**📈 对比分析**

与mini‑SWE‑agent的token长度和成功率对比，agentic成功率可达31%，而单次长上下文仅7%/0%，展示了显著性能差距

**⚠️ 局限性**

长上下文实际可用容量远低于理论上限，LLM在多文件推理中易产生hallucination，agentic拆分步骤掩盖了这一缺陷

---

## 153. LLMs Exhibit Significantly Lower Uncertainty in Creative Writing Than Professional Writers

**arXiv ID:** 2602.16162 | [PDF](https://arxiv.org/pdf/2602.16162v1)

**作者:** Peiqi Sui `[一作]` `[通讯]` (McGill University), Peiqi Sui (McGill University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过信息论方法对比LLM生成与人类作者的故事续写，量化并分析了二者在不确定性（熵、困惑度、PMI、CPMI）上的差距，并研究了对创作质量的影响。

**💡 创新点**

创新点在于：①首次用信息论度量人类与模型在创意写作中的不确定性差距；②揭示对齐策略导致的不确定性缺失与创意退化；③证明不确定性与写作质量呈非线性“甜点”关系。

**🔧 技术方法**

技术包括：使用同一LLM作为生成器和不确定性评估器；计算词级熵、困惑度、上下文PMI/CPMI；对不同模型（基础、指令调优、推理）和规模做系统实验；用写作质量奖励模型进行相关性与曲线回归分析。

**📊 数据集**

数据集：New Yorker专业短篇、Tell-Me-A-Story创作工作坊文本，以及Ghostbuster多领域对照集（创意、新闻、论文）用于跨领域比较。

**📈 对比分析**

比较方法：对每对情境/续写计算人类/模型的不确定性比值与差异；统计各模型/数据集的中位数、差距幅度；通过Spearman相关和二次回归检验不确定性与质量的关系；结果显示人类续写不确定性高2–4倍，指令/推理模型幅度更大，且不确定性与质量呈正相关且存在“甜点”峰值。

**⚠️ 局限性**

局限性：仅针对短篇文本，可能不适用于长篇或对话；使用同一模型评估可能带来评估偏差；不确定性指标主要基于词级统计，忽略更深层次叙事结构；缺少定性人类评审验证。

---

## 154. Distributed Order Recording Techniques for Efficient Record-and-Replay of Multi-threaded Programs

**arXiv ID:** 2602.15995 | [PDF](https://arxiv.org/pdf/2602.15995v1)

**作者:** Xiang Fu `[一作]` (Nanchang Hangkong University), Martin Schulz `[通讯]` (Technical University of Munich)

**通讯引用:** 10415 | [OpenAlex ID](https://openalex.org/A5045289712)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了分布式时钟（DC）和分布式纪元（DE）记录技术，用于高效地记录与重放 OpenMP 程序的共享内存访问，并将该技术与 MPI 级别的重放工具 ReMPI 结合，构建了可在大规模 HPC 环境下工作的新型 Record‑and‑Replay 工具。

**💡 创新点**

创新点：① 引入逻辑时钟替代线程 ID 进行记录，显著降低 I/O 与线程同步开销；② 进一步提出纪元记录，利用并发可互换的 load/store 指令组（满足特定条件）实现更大程度的并行重放；③ 将分布式记录技术与现有 MPI 重放框架无缝集成，实现纯粹共享内存与混合 MPI+OpenMP 应用的统一重放。

**🔧 技术方法**

技术手段：基于 LLVM Clang 进行源代码层的函数前后插桩；使用全局逻辑时钟与本地记录文件实现 DC 记录；通过维护访问历史和计算纪元值实现 DE 记录；实现 ReOMP 库与 ReMPI 交叉使用，支持 MPI_THREAD_MULTIPLE；采用 tmpfs 或节点本地存储来减少 I/O 延迟。

**📊 数据集**

数据集：1) 合成基准（omp_reduction、omp_critical、omp_atomic、data_race）用于量化不同记录策略的开销；2) 五个真实科学计算应用（AMG、QuickSilver、miniFE、HACC、HPCCG）评估性能提升；3) 两个 MPI+OpenMP 应用（HACC、HPCCG）作为案例研究，测试 ReMPI+ReOMP 在 24~4800 线程规模下的可扩展性。

**📈 对比分析**

比较方法：对 ST（传统线程 ID 记录）、DC、DE 三种策略在记录和重放阶段分别测量运行时间和资源占用；在合成基准和真实应用中比较不同线程数（2–112）下的性能。结果表明：在记录阶段，DE 约比 ST 低 10–30% 开销；在重放阶段，DE 速度比 ST 高 3.3–5.6 倍，DC 也比 ST 快 1.9–4.5 倍。MPI+OpenMP 组合实验显示 ReMPI+ReOMP 记录与重放的额外开销仅为 5–10% 左右。

**⚠️ 局限性**

局限性：① DE 需要维护访问历史，导致记录时略高于 DC；② 仅对满足两类条件的 load/store 可并行重放，无法覆盖所有数据竞争情况；③ 需要将记录文件放置于节点本地或 tmpfs，若使用共享文件系统仍可能成为瓶颈；④ 对非共享内存同步（如自定义锁、原子操作外的同步）支持有限；⑤ 对 OpenMP 任务调度等更复杂的调度模型的兼容性尚未验证。

---

## 155. CheckIfExist: Detecting Citation Hallucinations in the Era of AI-Generated Content

**arXiv ID:** 2602.15871 | [PDF](https://arxiv.org/pdf/2602.15871v1)

**作者:** Diletta Abbonato `[一作]` `[通讯]` (University of Turin), Diletta Abbonato (University of Turin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

提出并实现了一个开源的实时参考文献校验工具CheckIfExist，支持单条与批量BibTeX条目验证，并返回APA格式和可导出的BibTeX记录；

**💡 创新点**

创新点在于采用多源级联验证架构，结合CrossRef、Semantic Scholar和OpenAlex三大数据库，并通过字符串相似度（Levenshtein）与作者交叉比对提升校验的精确度与召回率；

**🔧 技术方法**

核心技术包括前端React+TypeScript实现无安装Web界面、LaTeX命令预处理、BibTeX解析、API多源查询、基于Levenshtein距离的标题相似度计算、作者匹配与多源一致性校验、置信度分数综合与阈值决策；

**📊 数据集**

使用的主要数据集为CrossRef、Semantic Scholar、OpenAlex的公开元数据（涵盖数千万条学术作品），以及从这些数据库检索的真实参考条目做为验证样本；

**📈 对比分析**

与单一数据库验证相比，级联多源查询在实验中显著提升了召回率（覆盖率从≈90%提升至≈99%）且在保证精度的同时实现秒级响应；虽然未给出完整量化指标，但性能表现满足科研写作即时校验需求；

**⚠️ 局限性**

局限性包括对API请求限额与网络延迟的依赖、部分学科或非英文出版物的覆盖不完整、可能出现的误报/漏报（如作者姓名相似但非同一人），以及需要持续维护数据库接口变更。

---

## 156. Heuristic Search as Language-Guided Program Optimization

**arXiv ID:** 2602.16038 | [PDF](https://arxiv.org/pdf/2602.16038v1)

**作者:** Mingxin Yu `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1876 | [OpenAlex ID](https://openalex.org/A5019603699)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LaGO 框架，将 LLM 驱动的自动启发式设计拆分为前向评估、后向分析、更新三大模块，形成可模块化、可升级的语言引导优化流程。

**💡 创新点**

创新点在于：① 把启发式搜索视为程序空间的显式优化；② 将评估、反馈、更新完全解耦，支持各模块独立改进；③ 引入可执行分析器生成结构化反馈、协同进化构造与改进启发式，以及多样性驱动的种群管理机制。

**🔧 技术方法**

使用技术包括：大型语言模型（如 GPT）进行代码生成与分析；可执行回溯追踪生成运行轨迹；基于种群的进化策略；soft‑penalty 约束处理；多样性度量与选择。

**📊 数据集**

使用数据集：HeuriGym benchmark 中的四个真实世界组合优化任务——PDPTW、机组人员配对、技术映射（EDA）和内操作并行调度，每个任务约 30 个实例。

**📈 对比分析**

与 HeuriGen、EoH、ReEvo、LLM‑LNS 等基线在三种协议下进行对比，采用 QYI 指标评估。结果显示 LaGO 在四个领域均超过基线，提升幅度 0.1–0.3 QYI，且泛化误差仅 0.07；在 TSP 上表现相当，表明对成熟领域提升有限。

**⚠️ 局限性**

局限性包括：对成熟、已优化的任务提升有限；需要软约束与调参；评估大规模实例仍昂贵；LLM 推理成本高；缺乏自动化脚本模板的高度适配灵活性。

---

## 157. Every Little Helps: Building Knowledge Graph Foundation Model with Fine-grained Transferable Multi-modal Tokens

**arXiv ID:** 2602.15896 | [PDF](https://arxiv.org/pdf/2602.15896v1)

**作者:** Yichi Zhang `[一作]` (Zhejiang University), Huajun Chen `[通讯]` (Zhejiang University)

**通讯引用:** 8112 | [OpenAlex ID](https://openalex.org/A5102018239)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个多模态知识图谱推理的基础模型（MMKG-FM），通过将结构、视觉和文本信息统一离散化为 token，并采用层次融合与混合消息机制完成跨 KG 的零样本与微调推理。

**💡 创新点**

创新点：① 将多模态信息离散化为可共享的 token，解决了传统模型的实体/关系专属嵌入问题；② 设计了结构编码器与多模态编码器的层次融合架构，并在全局传播中引入多模态混合消息（MiM）机制，实现了不同关系模式的自适应消息组合；③ 通过统一训练目标实现零样本推理与微调的一体化。

**🔧 技术方法**

技术：token化（使用文本 tokenizer 与 VQ‑VAE 视觉 codebook）、图神经网络（结构编码器与全局传播）、Transformer Encoder（多模态编码器）、门控融合（GF）、混合消息机制（MiM）以及对数似然损失。

**📊 数据集**

使用了 17 个多模态知识图谱（包括 DB15K、MKG‑W、MKG‑Y、YAGO15K、WN18RR++ 等），并在这些数据集上进行跨图、归纳与全归纳实验。

**📈 对比分析**

与传统 MMKGR 方法、KG 基础模型（ULTRA、MOTIF、KG‑ICL）以及 IndMKG 等基线相比，MMKG‑FM 在转导、归纳和全归纳三种设置下均表现出更高的 MRR/H@10，并在零样本场景中接近或超越微调 SOTA，显示出强大的跨图迁移能力。

**⚠️ 局限性**

局限性：① 对结构信息的 token 仍采用单一位置 token，可能无法充分捕捉更复杂的结构模式；② 视觉与文本 token 的数量需要经验调节，过多会导致效率下降；③ 在极度稀疏或缺少多模态内容的 KG 上表现不如纯结构模型。

---

## 158. Punchlines Unbound: Comedy Practices in Social Virtual Reality

**arXiv ID:** 2602.16013 | [PDF](https://arxiv.org/pdf/2602.16013v1)

**作者:** Ryo Ohara `[一作]` (University of Tokyo), Hideaki Kuzuoka `[通讯]` (University of Tokyo)

**通讯引用:** 3494 | [OpenAlex ID](https://openalex.org/A5004170823)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过对23名日本虚拟喜剧演员的半结构化访谈与现场表演观察，系统分析了他们在社交VR平台（主要是VRChat和Cluster）上的喜剧表演策略、观众互动方式和社区治理实践。

**💡 创新点**

创新点包括：①将“缺乏细腻非言语线索”视为创意资源，发现演员通过手势触发面部表情、夸张动作等方式实现喜剧效果；②揭示观众在VR中通过情境适配表情包反馈的独特互动模式；③阐明社区自发的打断与调节机制，强调VR本身可成为共享文化语境。

**🔧 技术方法**

使用技术主要是社交VR平台（VRChat、Cluster）及其头像与手势控制功能；未开发新算法，仅通过现有平台功能和手工控制实现表演；访谈与录像分析采用ATLAS.ti进行主题编码。

**📊 数据集**

数据集包含：23名演员的访谈文本、71段现场表演录像（平均时长7.46分钟），以及参与者的人口学与表演属性表；无公开大规模数据集。

**📈 对比分析**

本研究未进行算法或性能对比，而是采用质性主题分析与三角验证方法：访谈、现场录像与公开YouTube录制三源交叉检验，确保发现的可信度。研究结果以定性描述呈现，无数值指标。

**⚠️ 局限性**

限制：仅聚焦日本语社区，缺乏观众视角；受访者多为同一社群，可能存在文化同质性；技术熟练度差异大，可能影响结果；未对不同平台的技术差异进行系统量化比较。

---

## 159. Geometry-Aware Uncertainty Quantification via Conformal Prediction on Manifolds

**arXiv ID:** 2602.16015 | [PDF](https://arxiv.org/pdf/2602.16015v1)

**作者:** Marzieh Amiri Shahbazi `[一作]` (Rochester Institute of Technology), Ali Baheri `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 294 | [OpenAlex ID](https://openalex.org/A5086511671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种自适应地理（geodesic）一致性预测框架，用于在黎曼流形上构造分布无关的置信预测区域；

**💡 创新点**

创新点在于将欧氏残差替换为曲线距离的非一致性分数，并通过交叉验证得到的难度估计器对分数进行归一化，实现对异方差的局部自适应校准，同时保持分布无关的覆盖保证；

**🔧 技术方法**

采用分裂一致性预测（split conformal）方法、几何距离（geodesic distance）计算、k-NN回归与困难度估计、交叉验证难度回归以及自适应半径球形置信区间；

**📊 数据集**

使用两组数据集：一组是控制异方差的球面模拟数据（基于von Mises–Fisher分布），另一组是真实的IGRF‑14地磁场预测数据（ESA Swarm卫星观测的地磁方向向量）；

**📈 对比分析**

与标准的无自适应几何一致性预测和基于坐标的基线进行比较，结果表明自适应方法在保持边际覆盖率的前提下，显著降低了条件覆盖率的方差、提升了最差区间的覆盖率，并且区域面积更小；

**⚠️ 局限性**

局限性包括对交换性假设的依赖、仅使用等方差的球形预测区间（不支持方向性误差）、以及在流式或时间序列数据中可能失效的场景。

---

## 160. Close-enough general routing problem for multiple unmanned aerial vehicles in monitoring missions

**arXiv ID:** 2602.15841 | [PDF](https://arxiv.org/pdf/2602.15841v1)

**作者:** Huan Liu `[一作]` (Nanjing University of Information Science and Technology), Yi Gu `[通讯]` (Central South University)

**通讯引用:** 8633 | [OpenAlex ID](https://openalex.org/A5104075324)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并求解了多无人机闭近式通用路由问题（CEMUAVGRP），该问题要求无人机在执行包含节点（带圆形覆盖邻域）和边的监测任务时，最小化总飞行距离。

**💡 创新点**

首次将闭近式（cover‑within‑disk）节点观测与通用路由结合；采用嵌套非线性/整数规划模型，将问题拆分为通用路由和近似路由两阶段；提出基于自适应迭代局部搜索（AILS）的两阶段迭代框架，整合VND和SOCP求解；引入阈值递增接受准则和多样化扰动操作，提高全局搜索能力。

**🔧 技术方法**

核心技术包括：Adaptive Iterated Local Search (AILS)、Variable Neighborhood Descent (VND)、Second‑Order Cone Programming (SOCP)、Regret Insertion、Gurobi求解器、扰动（破坏-修复）算子、阈值接受策略。

**📊 数据集**

使用自行生成的150个基准实例（30个基本实例 C1–C30，每个实例进一步分为5个子实例，涵盖Type I与Type II，节点数、边数、航程和承载量各不相同），这些实例基于现有无人机路由问题的数据集并扩展了节点覆盖邻域。

**📈 对比分析**

与文献中的Branch‑and‑Cut（B&C）算法对比：在所有150个实例中，AILS‑VND‑SOCP 的平均误差不超过2.5%，在41个实例达到最优（0%误差），并在13个实例上优于B&C。相较于不使用覆盖邻域的算法（AILS‑VND‑SOCPI），引入覆盖邻域可平均减少14%飞行距离，某些实例还能减少使用的无人机数量。计算时间方面，AILS‑VND‑SOCP 在Type I实例上可在20分钟内完成，显著快于B&C（50–120分钟）。

**⚠️ 局限性**

局限性：假设所有无人机同质、无充电/续航考虑；边不可拆分；节点覆盖邻域仅为固定半径；仅测试人工合成实例，缺乏真实场景验证；算法对规模较大实例的可扩展性仍待进一步验证。

---

## 161. Hybrid Tabletop Exercise (TTX) based on a Mathematical Simulation-based Model for the Maritime Sector

**arXiv ID:** 2602.15975 | [PDF](https://arxiv.org/pdf/2602.15975v1)

**作者:** Diego Cabuya-Padilla `[一作]` (Escuela Superior de Guerra General Rafael Reyes Prieto), Carlos Castaneda-Marroquín `[通讯]` (Escuela Superior de Guerra General Rafael Reyes Prieto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一种融合传统桌面演练（TTX）与基于数学模拟模型（SERDUX-MARCIM）的混合式桌面演练方法，专门用于提升海事领域的网络态势感知（CSA）与决策能力，并在阿根廷与美国两地实施实战演练。

**💡 创新点**

创新点在于将严肃游戏与分层传播模型相结合，构建可实时更新、可视化的网络攻击传播框架，从而在桌面演练中提供动态决策支持，显著弥补传统TTX缺乏实时数据与复杂性建模的不足。

**🔧 技术方法**

技术手段包括：1）基于流行病学模型（SIR/SEIR）的仿真框架SERDUX-MARCIM；2）系统动力学与代理模型的耦合实现；3）桌面演练的结构化流程设计与参与者引导；4）Python/SimPy等仿真工具与可视化图表。

**📊 数据集**

数据集主要来源于公开的2017年NotPetya攻击对马士基（Maersk）的事件细节、网络拓扑与资产信息，并在演练中自定义的四个关键事件与三种监测变量（网络状态、服务可用度、整体网络风险姿态）。

**📈 对比分析**

通过对两场演练的参与者满意度、方案理解度与模型准确性进行问卷评估，并采用多元线性回归与情景预测分析，对比传统TTX与混合TTX在提升CSA三维（感知、理解、预测）和决策效率方面的效果。结果显示，混合TTX在平均满意度（4.56/5）与模型准确度（4.67/5）上均超出传统方法，且回归模型解释度高达87.4%。

**⚠️ 局限性**

局限性包括：1）样本规模有限（36名高层参与者），可能影响统计显著性；2）模型参数需基于实际海事网络环境进一步校准；3）演练侧重战略层面，对战术与操作层面未做细化；4）缺乏长期跟踪评估以验证学习效果持久性。

---

## 162. Axle Sensor Fusion for Online Continual Wheel Fault Detection in Wayside Railway Monitoring

**arXiv ID:** 2602.16101 | [PDF](https://arxiv.org/pdf/2602.16101v1)

**作者:** Afonso Lourenço `[一作]` (GECAD, ISEP, Polytechnic of Porto), Goreti Marreiros `[通讯]` (GECAD, ISEP, Polytechnic of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个标签高效的持续学习框架，用于基于加速度传感器的VAE潜在空间与光纤布拉格光栅（FBG）应变传感器的语义信息融合，实现铁路车轮缺陷的在线连续检测。

**💡 创新点**

创新点包括：①将FBG提取的轴计数、时间索引和应变变形信息作为语义特征与VAE编码的潜在表示融合；②使用极少标签的XGBoost分类器和基于经验回放的损失平衡策略，既保持对新域的适应，又防止灾难性遗忘；③在实验中系统性验证了语义信息对多种峰值检测算法和持续学习策略的显著提升。

**🔧 技术方法**

技术方法主要包括：1）Peak Detection（Tony Beltramelli、Peakdetect、Detectpeaks、Scipy）提取轴计数；2）Variational AutoEncoder（VAE）对加速度序列进行无监督潜在表示；3）XGBoost监督分类器；4）Replay-based Continual Learning（reservoir sampling、loss‑based sampling、预测标签回放等）；5）语义特征提取与融合；6）基于仿真生成的多域数据集进行实验与统计检验（Friedman、Shaffer）。

**📊 数据集**

使用基于VSI仿真生成的多域铁路轨道与车体交互数据，包含两种车型（Laagrss、Alfa）、多种负载、速度、轨道不平整、车轮缺陷（平面缺口、极化）等情境，产生约数十万条加速度与应变时间序列，并通过峰值检测得到轴计数与语义标签。

**📈 对比分析**

与传统手工特征+XGBoost、单一VAE深度、不同峰值检测器、不同经验回放策略进行对比。实验结果表明：VAE单层模型+语义融合的准确率可达95%，比手工特征提升约23%；在多种峰值检测算法中，Scipy峰值检测与语义增强（S‑WD*、I‑WD*）的准确率最高达93%；在持续学习场景下，loss‑based回放（LB）配合800样本缓冲区，前向迁移（FWT）最高可达0.31，遗忘率显著低于无回放或reservoir回放，知识保持率（KGR）>3。整体性能提升5–12个百分点，尤其在多缺陷路径中表现突出。

**⚠️ 局限性**

主要局限包括：1）实验数据完全基于仿真，缺乏真实铁路环境下的噪声与电磁干扰验证；2）峰值检测算法对参数敏感，仍需更鲁棒的自动阈值策略；3）FBG传感器对温度、热膨胀的敏感性未系统评估；4）持续学习框架在极端数据分布漂移（如极低速或极高速）下的泛化能力待进一步验证。

---

## 163. What Persona Are We Missing? Identifying Unknown Relevant Personas for Faithful User Simulation

**arXiv ID:** 2602.15832 | [PDF](https://arxiv.org/pdf/2602.15832v1)

**作者:** Weiwen Su `[一作]` (University of Tokyo), Masashi Toyoda `[通讯]` (Institute of Industrial Science University of Tokyo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了识别用户模拟中未知相关人物的任务，并构建了PICQ数据集，对LLMs在该任务上的表现进行评估。

**💡 创新点**

首创“未知相关人物识别”任务与多维度评估（影响、可获取性、忠实度），并揭示了模型规模下忠实度与洞察力的倒U曲线。

**🔧 技术方法**

采用多任务提示策略、LLMs推理、NLI匹配及自动评分等技术进行实验。

**📊 数据集**

使用基于TVshowGuess剧本构建的PICQ数据集，该数据集包含对话、PICQ、基本人物信息、答案及手工标注的未知人物。

**📈 对比分析**

通过影响、可获取性和F1等指标与GPT‑4.1、Qwen3‑32B等模型对比，发现Qwen3‑32B在忠实度最高，而GPT‑4.1在洞察力最高，模型规模对结果影响显著。

**⚠️ 局限性**

研究仅局限于英文剧本，缺乏真实对话数据，跨语言、跨文化适用性未知。

---

## 164. Differentially Private Non-convex Distributionally Robust Optimization

**arXiv ID:** 2602.16155 | [PDF](https://arxiv.org/pdf/2602.16155v1)

**作者:** Difei Xu `[一作]` (King Abdullah University of Science and Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 4027 | [OpenAlex ID](https://openalex.org/A5100401482)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了两种基于差分隐私的分布鲁棒优化（DP‑DRO）算法，分别为 DP Double‑Spider（针对一般 ψ‑divergence）和 DP Recursive‑Spider（针对 KL‑divergence）实现非凸损失函数的私有优化。

**💡 创新点**

创新点：
1) 通过主从对偶把 DRO 转化为无约束最小化，设计了双变异 SPIDER 结构的 DP Double‑Spider；
2) 把 KL‑DRO 重写为组合式问题，提出 DP Recursive‑Spider，并实现了最佳梯度范数上界 O((√d log(1/δ))/(nε))^{2/3}；
3) 对两算法给出（ε,δ）-DP 证明及梯度范数的理论上界；
4) 在隐私安全性上通过 Membership Inference Attack (MIA) 实验显示 DP Recursive‑Spider 的 AUC 明显低于基线。

**🔧 技术方法**

主要技术：SPIDER 及其双变异变体、Gaussian 机制、梯度裁剪、主从对偶变换、组合优化（STORM‑style 递归）、敏感度分析。

**📊 数据集**

使用的数据集：CIFAR10-ST、MNIST-ST、CelebA、Fashion‑MNIST（均为人工构造的 imbalanced 版本）。

**📈 对比分析**

对比方法：DP‑SGDA、PrivateDiff Minimax 以及非私有 SCDRO。评价指标包括梯度范数收敛、测试准确率、MIA AUC。实验结果表明：
- 在所有 ε 设置下，DP Double‑Spider 与 DP Recursive‑Spider 的测试准确率均高于基线；
- 梯度范数收敛更快且波动更小；
- DP Recursive‑Spider 的 MIA AUC 在 0.78–0.82 之间，显著低于 DP Double‑Spider（≈0.97）。

**⚠️ 局限性**

局限性：
1) 仅针对 L‑smooth、非凸损失；
2) KL‑DRO 的改进只适用于 KL‑divergence，未给出其他 ψ‑divergence 的最优界；
3) 实验规模仅限四个数据集，未验证更大模型或不同任务的泛化；
4) 对深度网络的梯度噪声影响及收敛速度的理论分析仍不完整。

---

## 165. Can Vision-Language Models See Squares? Text-Recognition Mediates Spatial Reasoning Across Three Model Families

**arXiv ID:** 2602.15950 | [PDF](https://arxiv.org/pdf/2602.15950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 166. Fly0: Decoupling Semantic Grounding from Geometric Planning for Zero-Shot Aerial Navigation

**arXiv ID:** 2602.15875 | [PDF](https://arxiv.org/pdf/2602.15875v1)

**作者:** Zhenxing Xu `[一作]` (National University of Defense Technology), Ji Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 12095 | [OpenAlex ID](https://openalex.org/A5100386450)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在无人机视觉语言导航中，提出 Fly0 框架，先使用大语言模型对指令进行 2D 像素定位，再利用深度信息投影为 3D 目标，并由几何规划器生成碰撞自由轨迹，实现零样本导航。

**💡 创新点**

创新点在于将语义推理与低层运动控制解耦：将 MLLM 用作语义定位器而非控制器，显著降低推理延迟，提升成功率并获得可解释的 3D 目标。

**🔧 技术方法**

使用多模态大型语言模型（如 Qwen2.5VL-32B）进行 2D 语义定位，深度投影实现 3D 转换，Ego-Planner 进行 B-spline 轨迹优化。

**📊 数据集**

评估使用 AirSim/UE4 仿真环境中的 AerialVLN 与 OpenFly 数据集，以及自建的校园/公园实景数据集。

**📈 对比分析**

相较于端到端学习和零样本 MLLM 控制方法，Fly0 在 AerialVLN 上成功率提升约 20% 并将导航误差降至 27m，实景实验成功率 62% 远高于对比基线。

**⚠️ 局限性**

局限包括对语义歧义的鲁棒性不足（多目标环境导致误定位）以及深度估计误差导致的 3D 投影误差，影响精准终点到达。

---

## 167. AI-CARE: Carbon-Aware Reporting Evaluation Metric for AI Models

**arXiv ID:** 2602.16042 | [PDF](https://arxiv.org/pdf/2602.16042v1)

**作者:** KC Santosh `[一作]` (University of South Dakota), Rodrigue Rizk `[通讯]` (University of South Dakota)

**通讯引用:** 145 | [OpenAlex ID](https://openalex.org/A5035738930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出AI-CARE框架，统一记录并可视化模型的能耗、碳排放与性能，生成碳-性能折衷曲线与标量碳意识分数。

**💡 创新点**

设计标准化的碳意识评估层，绘制Pareto前沿，提出单标量碳意识分数（SCAS）以兼顾准确性与碳排放，并以开源形式实现可复现评估。

**🔧 技术方法**

采用能耗监测工具（如CodeCarbon）与碳强度计量，利用浮点运算量和内存访问量模型计算能耗，再按固定电网碳强度转换为碳排放，并通过Python脚本生成可视化曲线。

**📊 数据集**

在MNIST、Fashion-MNIST、CIFAR-10/100、ImageNet-100等视觉基准上测试多种模型架构（MLP、CNN、Transformer、MLP-Mixer）。

**📈 对比分析**

对比模型在相同实验设置下的预测指标（准确率、精确率、召回率、F1）与总碳排放，绘制多指标碳-性能曲线；结果表明性能提升往往伴随不成比例的碳增幅，碳意识分数揭示更高效模型；在简单数据集轻量模型排名更佳，复杂数据集高能耗模型排名下降。

**⚠️ 局限性**

仅提供经验评估，未对训练期间动态能耗进行实时优化；采用固定电网碳强度不考虑地域差异；只关注碳排放与能耗，未涵盖延迟、成本等其他部署因素；工具的易用性需进一步验证。

---

## 168. From Transcripts to AI Agents: Knowledge Extraction, RAG Integration, and Robust Evaluation of Conversational AI Assistants

**arXiv ID:** 2602.15859 | [PDF](https://arxiv.org/pdf/2602.15859v1)

**作者:** Krittin Pachtrachai `[一作]` (Amity Research and Application Center), Touchapon Kraisingkorn `[通讯]` (Amity Research and Application Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从历史通话记录构建、评估可对话式AI助手，采用转录评估、知识提取、RAG与系统化提示调优。

**💡 创新点**

创新点在于将对话评估与主题覆盖最大化相结合，使用LLM自动评判；将提取知识外部化至RAG，并提出多层级模块化提示设计，实现可扩展的可控行为。

**🔧 技术方法**

技术包括LLM自动评估、知识抽取、检索增强生成（RAG）、模块化/协议化Prompt Engineering以及用户模拟与红队评测。

**📊 数据集**

数据集为真实房地产与专业招聘行业的历史电话转录，经过评估过滤后形成高质量对话样本。

**📈 对比分析**

与不同提示层级对比实验显示，治理式提示在两行业实现35%/25%通话覆盖、97%/94%事实准确度、100%拒绝准确率，且对抗攻击表现卓越。

**⚠️ 局限性**

局限在于覆盖率受限于实时数据需求与高阶谈判，RAG对动态信息更新不足，且仍需人工监督。

---

## 169. A Comprehensive Survey on Deep Learning-Based LiDAR Super-Resolution for Autonomous Driving

**arXiv ID:** 2602.15904 | [PDF](https://arxiv.org/pdf/2602.15904v1)

**作者:** June Moh Goo `[一作]` (University College London), Jan Boehm `[通讯]` (University College London)

**通讯引用:** 6576 | [OpenAlex ID](https://openalex.org/A5056951932)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对自动驾驶中基于深度学习的 LiDAR 超分辨率方法进行全面综述，梳理了从 CNN、模型驱动解卷积、隐式表示到 Transformer/Mamba 等四大类别，阐述了数据表示、问题定义、基准数据集与评价指标，并指出当前技术的趋势与挑战。

**💡 创新点**

首次系统性总结 LiDAR 超分辨率研究，提出四大方法体系，构建了统一的概念框架与评价体系，识别了跨传感器通用性、实时推理与几何信息损失等关键瓶颈，并给出未来研究方向。

**🔧 技术方法**

综述涵盖 CNN‑based、Model‑based Deep Unrolling、Implicit Representation、Transformer 与 Mamba‑based 技术，结合投影式范围图、极坐标处理、物理模型解卷积、隐式连续函数以及自注意力与状态空间模型等关键技术。

**📊 数据集**

讨论了多种基准数据集，包括真实场景的 KITTI、nuScenes、DurLAR、Ouster 等；以及合成环境的 CARLA、LiDAR‑CS、SemanticKITTI 等，用于训练、验证与跨传感器比较。

**📈 对比分析**

对比了各类别方法的参数量、实时性、重建质量（MAE、Chamfer Distance、IoU 等）和对下游任务的影响，指出 Transformer/Mamba 在保持几何细节上表现最佳，但计算成本最高；而 Deep Unrolling 与 Implicit 方法在参数与推理速度上具优势，但在表达复杂语义时受限。

**⚠️ 局限性**

主要局限包括：跨传感器泛化差，需针对每种 LiDAR 重新训练；实时推理仍难以满足 25 fps 需求；大部分研究仅关注重建指标，缺乏对目标检测/分割等下游任务的系统评估；投影式范围图导致几何信息丢失，3D 原生方法计算成本高。

---

## 170. Uncertainty-Guided Inference-Time Depth Adaptation for Transformer-Based Visual Tracking

**arXiv ID:** 2602.16160 | [PDF](https://arxiv.org/pdf/2602.16160v1)

**作者:** Patrick Poggi `[一作]` (University of Illinois at Chicago), Amit Ranjan Trivedi `[通讯]` (University of Illinois at Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了UncL-STARK，在Transformer单目标跟踪中实现无架构修改的深度自适应推理，通过不确定性引导动态选择encoder‑decoder层数；

**💡 创新点**

创新点包括：① 采用随机深度训练与知识蒸馏实现多深度可预测；② 用角点热图的top‑k概率质量作为轻量级不确定性估计；③ 基于反馈阈值策略在下一帧动态调整深度；

**🔧 技术方法**

使用技术包括Transformer encoder‑decoder（STARK）、随机深度训练、知识蒸馏、热图自估计以及不确定性驱动的深度自适应推理；

**📊 数据集**

实验数据集为GOT‑10k（训练/验证）和LaSOT（测试）；

**📈 对比分析**

与固定深度STARK对比，采用GFLOPs、延迟、能耗等度量，UncL-STARK在保持0.2%以内准确率的前提下，GFLOPs降低12%，延迟8.9%，能耗10.8%；与随机深度或静态裁剪相比，精度更高；

**⚠️ 局限性**

局限性包括：阈值需手动设定，极深帧仍需全深度；在极短或极复杂场景下可能不如固定深度稳定；目前仅适用于已生成热图的跟踪器。

---

## 171. ODYN: An All-Shifted Non-Interior-Point Method for Quadratic Programming in Robotics and AI

**arXiv ID:** 2602.16005 | [PDF](https://arxiv.org/pdf/2602.16005v1)

**作者:** Jose Rojas `[一作]` (Heriot-Watt University), Carlos Mastalli `[通讯]` (Heriot-Watt University)

**通讯引用:** 1148 | [OpenAlex ID](https://openalex.org/A5032652050)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种名为 Odyn 的全移位非内部点二次规划（QP）求解器，用于高效求解稠密与稀疏 QP，特别适合机器人与 AI 的连续优化、控制与学习任务。

**💡 创新点**

核心创新在于：① 采用全移位 NCP（非互补）函数与中心化加权障碍，消除传统内部点法对内部可行性强制的限制；② 将所有非线性互补约束与增广拉格朗日乘子（Proximal Method of Multipliers）融合，形成路径跟踪的 NIPM；③ 通过闭式解算共识变量、对梯度与约束残差进行“软”正则化，实现对退化、秩缺陷和数值不稳定问题的鲁棒处理。

**🔧 技术方法**

技术实现包括：稠密与稀疏后端（基于 Eigen 的稠密/稀疏线性代数），全 KKT 与压缩 KKT 系统求解（Cholesky/LDL），非单调 Armijo 步长、动态更新的中心参数与插值参数，自动微分（Implicit Function Theorem）以支持可微优化层；并提供 Python 接口与 C++ 核心。

**📊 数据集**

使用标准 Maros–Mészáros QP 测试集（138 题）评估收敛与稳定性；在机器人应用中，使用基于 Crocoddyl 的 SQP、基于 Odyn 的接触动力学仿真以及用于 Sudoku 学习的可微 QP 任务。

**📈 对比分析**

与 ProxQP、PiQP、OSQP、Mosek、Gurobi 等现代求解器做性能对比：在中高精度下，Odyn 在失效率（9.7%）与迭代次数上与 PiQP 并驾齐驱，且在高精度场景下仍保持与 PiQP 相近；在 Warm‑to‑Cold 比值上，Odyn 的比例显著低于 OSQP、ProxQP，显示出更强的 warm‑start 能力；在稠密后端与退化 QP 上，Odyn 也表现出可竞争或优于现有方法的计算时间与迭代数。

**⚠️ 局限性**

主要局限包括：① 对极高精度需求时缺乏迭代细化（Iterative Refinement），导致收敛速度慢；② 在某些单精度或极度退化问题上仍可能出现误差积累；③ 需要手动调参（中心化、正则化、线搜索参数）以获得最佳性能；④ 对大规模稀疏 QP 的最优内存与求解时间仍不如专门的 ADMM/ALM 方案。

---

## 172. Enhancing Action and Ingredient Modeling for Semantically Grounded Recipe Generation

**arXiv ID:** 2602.15862 | [PDF](https://arxiv.org/pdf/2602.15862v1)

**作者:** Guoshan Liu `[一作]`, Yu-Gang Jiang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种语义扎根的食谱生成框架，先对食材与烹饪动作进行预测与校验，再作为内部上下文生成指令。

**💡 创新点**

创新点在于将动作与食材预测转化为内部检索，结合链式思考数据、GRPO强化学习与频率感知奖励，并引入语义置信度评分与纠正模块，显著提升语义忠实度。

**🔧 技术方法**

采用多模态大型语言模型（Qwen2.5-VL-7B、LLaVA-v1.5-7B）与LoRA微调，使用链式思考（CoT）构建动作推理数据，GRPO强化学习设计 F1/单词/格式奖励，GPT‑4o 进行置信度评分与纠正，最后通过结构化提示实现检索增强式生成。

**📊 数据集**

主要使用 Recipe1M 数据集（图像、配料、步骤），并基于 Recipe1M 构建的动作推理数据集（含 CoT 注解）和配料语料；在基准模型和公开基线上进行对比评估。

**📈 对比分析**

与 LLaVA‑FT、Qwen2.5‑VL‑7B‑FT、RARG、FoodLMM 等公开模型在 Recipe1M 上进行定量对比，使用 SacreBLEU、ROUGE‑L、BERTScore 评估文本流畅度，使用 F1、IoU、召回率、精确率评估动作与配料的语义一致性；实验显示在动作/配料 F1 与 IoU 上取得 SOTA，虽略有 BLEU 降低，但语义正确性显著提升。

**⚠️ 局限性**

局限性包括对长尾动作仍有一定偏差、需依赖 GPT‑4o 等强大 LLM 进行置信度校正、模型对 Recipe1M 的特定性强、并且在流畅度与语义精确度之间仍存在权衡；此外训练成本高、未实现完整端到端联合优化。

---

## 173. Investigating GNN Convergence on Large Randomly Generated Graphs with Realistic Node Feature Correlations

**arXiv ID:** 2602.16145 | [PDF](https://arxiv.org/pdf/2602.16145v1)

**作者:** Mohammed Zain Ali Ahmed `[一作]` `[通讯]`, Mohammed Zain Ali Ahmed

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种在 Barabási–Albert（BA）图生成过程中同时采样相关节点特征的方法，并利用该生成的随机图评估图神经网络（GNN）的收敛性和表达能力。

**💡 创新点**

创新点在于：①将节点特征的正相关（同质性）与 BA 过程中的优先连接机制关联，构造可控的相关特征；②提出简单与归一化两种相关采样方案，并给出理论分析证明在某些情形下可避免收敛；③结合理论与大规模实验验证相关特征对 GNN 收敛性的影响。

**🔧 技术方法**

使用技术包括：BA 模型图生成、条件多元正态分布采样节点特征（实现正相关）、将正态分布映射至均匀分布、构建 3 层 GAT 或 GCN 网络+均值池化+单层 MLP+softmax 进行图分类、对不同图大小（25–2000）与密度（稀疏/稠密）进行实验。

**📊 数据集**

所用数据集为论文自生成的 30 张随机图，每张图的节点数从 25 到 2000，节点特征维度为 32，特征值在 U[0,1] 范围内。

**📈 对比分析**

比较方法为对 12 种组合（GAT/GCN × 稀疏/稠密 × 无相关/简单相关/归一化相关）分别计算 30 次样本的类别概率均值与标准差，并绘制图形。实验结果显示：无相关时所有组合均收敛；稠密图或简单相关时仍收敛；而稀疏图且采用归一化相关时输出波动大、未收敛，表明相关特征可提升 GNN 的表达能力。

**⚠️ 局限性**

局限性包括：仅针对连续特征且假设邻居特征独立；BA 模型存在聚类系数不合理等缺陷；未考虑其他图生成器或异质特征；理论推导中做了多项近似；实验仅使用随机图，未检验在真实网络上的表现；并未对不同激活函数、跳连或宽度等 GNN 组件的影响进行深入探讨。

---

## 174. IRIS: Intent Resolution via Inference-time Saccades for Open-Ended VQA in Large Vision-Language Models

**arXiv ID:** 2602.16138 | [PDF](https://arxiv.org/pdf/2602.16138v1)

**作者:** Parsa Madinei `[一作]` (University of California Santa Barbara), Miguel P. Eckstein `[通讯]` (University of California Santa Barbara)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出IRIS，一种训练无关的实时眼动辅助VQA系统，利用用户在发问时的眼动信息来消解开放式视觉问答中的指称歧义。

**💡 创新点**

创新点在于：①在推理阶段直接利用眼动数据，无需模型改动或额外训练；②提出实时交互协议、时间-空间眼动过滤方法；③公开了针对歧义VQA的新基准数据集和评估套件。

**🔧 技术方法**

采用技术包括：高精度眼动追踪（EyeLink1000）、语音识别与发问时间检测、视觉语言模型（VLM）推理、眼动时间窗口筛选与空间聚类、文本嵌入相似度评估等。

**📊 数据集**

使用新收集的500对图像-问题（50张日常场景图像）数据集，其中包含40张歧义图像与10张非歧义图像，并将其公开作为基准。

**📈 对比分析**

对10款SOTA VLM（GPT-5 Mini、Gemini 2.5 Flash等）进行对比实验；在加入眼动信息后，歧义问答准确率从35.2%提升至77.2%（115%增幅），在非歧义问答保持不变；语义相似度亦显著提高。

**⚠️ 局限性**

局限性包括：实验仅在受控实验室环境下进行，受试者为大学生样本；使用研究级眼动仪，需验证在更广泛人群和消费级设备上的可行性。

---

## 175. Managing Credible Anonymous Identities in Web 3.0 Services: A Scalable On-Chain Admission Framework with Recursive Proof Aggregation

**arXiv ID:** 2602.16130 | [PDF](https://arxiv.org/pdf/2602.16130v1)

**作者:** Zibin Lin `[一作]` (Shenzhen University), Shui Yu `[通讯]` (University of Technology Sydney)

**通讯引用:** 28073 | [OpenAlex ID](https://openalex.org/A5005228053)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了ZK‑AMS——一种基于零知识递归证明与多钥同态加密的可扩展匿名身份管理与入网服务框架；

**💡 创新点**

通过将Nova递归折叠与BGV多钥同态加密结合，实现无信任批处理与常数级链上验证，解决了Web 3.0入网的Sybil防护、隐私保护与成本可预测的三重矛盾；

**🔧 技术方法**

核心技术包括：零知识证明（Groth16）、Nova递归折叠（NIFS）、BGV多钥同态加密、可链接环签名（MLSAGS）；

**📊 数据集**

使用Ethereum私有测试网进行实验评估，未采用公开数据集；

**📈 对比分析**

与原始zkBID方案对比，ZK‑AMS在批量大小N增大时保持链上验证成本稳定（≈435 k gas），批量证明时间基本不随N变化，整体入网吞吐提升至约5.6–6.3 用户/秒；

**⚠️ 局限性**

主要局限在于环签名验证的gas成本仍较高，且批处理节点缺乏完善的激励机制以确保其活跃度与抗攻击性。

---

## 176. On the Power of Source Screening for Learning Shared Feature Extractors

**arXiv ID:** 2602.16125 | [PDF](https://arxiv.org/pdf/2602.16125v1)

**作者:** Leo `[一作]`, Lili Su `[通讯]` (Northeastern University)

**通讯引用:** 2298 | [OpenAlex ID](https://openalex.org/A5101541239)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究在多源线性回归中，通过源筛选仅使用部分源来学习共享低维子空间，证明在存在足够多“良好”源的前提下，仅训练这部分源即可达到极限统计性能。

**💡 创新点**

创新点在于：①提出“信息子群”概念并证明其存在；②证明仅使用信息子群即可实现下界匹配的最优子空间估计；③设计了基于稳定秩与最小特征值的可实现源筛选算法（含天才版与经验版）；④首次在此问题下用理论+实验验证源筛选能在保持样本量的同时提升子空间重构精度。

**🔧 技术方法**

主要技术包括：线性子空间估计、主角角距离度量、稳定秩与奇异值分解、Grothendieck 因子化、随机抽样与分层取样策略、以及分布式回归与Vision Transformer特征提取。

**📊 数据集**

使用了合成数据（聚类系数与异质高斯系数两种生成方式）以及真实数据集ACSIncome（10维表格，1.6M样本）和CelebA（224×224图像，ViT提取192维特征）。

**📈 对比分析**

与全量训练、随机子采样和基于Power‑of‑Choice的主动采样进行对比。实验表明，在合成和真实数据上，经验式筛选方案在子空间重构误差或分类准确率上均优于全量训练，并在某些场景下甚至优于最优天才版。

**⚠️ 局限性**

局限性包括：仅针对线性模型给出理论保证；假设协方差矩阵可估计且噪声满足子高斯；实验中仅使用少数几种基准方法，缺乏对更复杂非线性方法的评估；在极端不平衡或高维情况下，稳定秩估计可能失效。

---

## 177. Evolutionary Context Search for Automated Skill Acquisition

**arXiv ID:** 2602.16113 | [PDF](https://arxiv.org/pdf/2602.16113v1)

**作者:** Qi Sun `[一作]` (Sakana AI), Yujin Tang `[通讯]` (Sakana AI)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于遗传算法的演化上下文搜索(ECS)，在LLM部署后通过仅进行推理调用、无梯度更新的方式，从给定文本资源中自动生成最优上下文，从而使模型获得新的技能；

**💡 创新点**

创新点在于把上下文选择建模为进化优化问题，使用遗传算法在资源池中搜索非直观、组合性强的上下文；演化得到的上下文对不同模型高度通用，可实现跨模型迁移；

**🔧 技术方法**

采用遗传算法（初始化、精英选择、交叉、变异）与LLM指导的矛盾消解，构建多层次上下文单元（原文、抽象洞察、可调用技能），实现无梯度、无训练的上下文优化；

**📊 数据集**

主要实验数据集包括BackendBench（CuTeDSL DSL 编码任务）、τ^2‑Bench（航空域多轮对话任务），并使用GPT‑5.2、Gemini‑3‑Pro等生成的文档/洞察作为资源；

**📈 对比分析**

与多种基线（RAG‑Dense/BM25/Hybird、AST+Dense、Full Context、Random Sample）对比，ECS在BackendBench上相对AST+Hybrid提升约27%，在τ^2‑Bench提升约7%；在Claude‑Sonnet和DeepSeek等未参与搜索的模型上亦显著提升，表明跨模型迁移效果良好；

**⚠️ 局限性**

限制包括：需要预先投入搜索预算；在海量语料中“needle‑in‑the‑haystack”场景下搜索效率可能下降；LLM在后处理矛盾消解的效果因任务不同而异；对资源可访问性和质量存在依赖。

---

## 178. Modeling Trust and Liquidity Under Payment System Stress: A Multi-Agent Approach

**arXiv ID:** 2602.16186 | [PDF](https://arxiv.org/pdf/2602.16186v1)

**作者:** Masoud Amouzgar `[一作]` (Sharif University of Technology), Masoud Amouzgar `[通讯]` (blu Bank)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个多代理模型，用以研究零售支付系统中断如何通过消费者信任衰退、商家信号以及社交网络传染导致提现压力延迟峰值。

**💡 创新点**

创新点在于将支付可靠性与消费者行为记忆、阈值驱动的渠道规避以及商家广播持久性结合，证明并仿真了技术恢复后仍出现最大提现压力的现象。

**🔧 技术方法**

采用了小世界网络、阈值门控机制、有限记忆更新及离散时间代理模型，并通过理论证明和数值仿真验证。

**📊 数据集**

使用的是模拟数据，未引用真实交易或事件日志，而是根据设定的概率过程生成支付成功/失败/未知结果。

**📈 对比分析**

对比方法主要是无替代渠道与有即时转账替代渠道的情景，结果显示替代渠道降低了峰值恐慌但对累计提现影响非单调，整体性能以行为峰值和累计提现量为指标。

**⚠️ 局限性**

局限性包括未考虑银行资产负债表、互金网络、商家与收单机构真实互动，支付系统被视为外生过程，且模型未做实证校准。

---

## 179. Multi-Agent Combinatorial-Multi-Armed-Bandit framework for the Submodular Welfare Problem under Bandit Feedback

**arXiv ID:** 2602.16183 | [PDF](https://arxiv.org/pdf/2602.16183v1)

**作者:** Subham Pokhriyal `[一作]` (Indian Institute of Technology Ropar), Vaneet Aggarwal `[通讯]` (Purdue University)

**通讯引用:** 6180 | [OpenAlex ID](https://openalex.org/A5064822688)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一个非通信的多智能体组合多臂赌博机（MA‑CMAB）框架，用于在全带宽反馈下求解子模福利（Submodular Welfare Problem）问题；

**💡 创新点**

创新点在于：①将离散的分区分配问题映射为多智能体CMAB，②在缺乏逐臂反馈的全带宽环境下实现探索‑承诺策略并通过离线‑在线转化获得近似保证，③实现了相对于(1‑1/e)近似基准的O~(T^{2/3})累积后悔率，首次在此类分区子模福利问题中得到此类下界；

**🔧 技术方法**

采用了离线子模最大化的连续贪婪与管道化逼近技术、随机化分配策略、探索‑承诺的两阶段算法、Hoeffding 置信区间与全概率事件的概率分析；

**📊 数据集**

本研究为理论性工作，未使用公开数据集，实验仅基于理论分析与假设的噪声模型；

**📈 对比分析**

与传统单智能体或可分离多智能体CMAB方法对比，本文在全带宽反馈下仍保持与最优单智能体子模CMAB相当的后悔率O~(T^{2/3})；

**⚠️ 局限性**

限制包括：①缺乏实验验证，②离线算法需具备α‑近似与δ‑鲁棒性，需额外的值算子；③无法处理动态或非均匀噪声场景；④仅适用于离散分区而非更一般的组合约束。

---

## 180. Edge Learning via Federated Split Decision Transformers for Metaverse Resource Allocation

**arXiv ID:** 2602.16174 | [PDF](https://arxiv.org/pdf/2602.16174v1)

**作者:** Fatih Temiz `[一作]` (University of Ottawa), Melike Erol-Kantarci `[通讯]` (University of Ottawa)

**通讯引用:** 7976 | [OpenAlex ID](https://openalex.org/A5089891162)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种联邦拆分决策变压器（FSDT）框架，用于多 RAT MEC 环境下的元宇宙资源分配。

**💡 创新点**

创新点在于将决策变压器拆分为云端共享解码器和边缘端专属嵌入/预测层，兼顾全局协同与本地适配，并通过离线强化学习提升样本效率。

**🔧 技术方法**

采用联邦学习、拆分学习和离线强化学习（决策变压器）技术，结合自回归序列建模实现资源分配决策。

**📊 数据集**

使用公开的 VR 眼动跟踪数据集（200+ 4K YouTube 视频，包含头部/眼动坐标），并基于此生成 FoV 级别的切片质量标签。

**📈 对比分析**

与中央 DDPG、联邦 DDPG、中央 DT 和联邦 DT 等基线对比，FSDT 在 QoE 上提升约 10%，延迟更低且更稳定，表现出最优的整体性能。

**⚠️ 局限性**

局限性包括仅在模拟环境和四用户场景下验证，缺乏真实设备测试；云端解码器仍需承担大部分计算，且对边缘设备协作和更大规模多设备场景的适应性待进一步研究。

---

## 181. Bellman-Ford in Almost-Linear Time for Dense Graphs

**arXiv ID:** 2602.16153 | [PDF](https://arxiv.org/pdf/2602.16153v1)

**作者:** George Z. Li `[一作]` (Carnegie Mellon University), Junkai Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 3607 | [OpenAlex ID](https://openalex.org/A5026460845)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种随机化的 n²+o(1) 时间单源最短路算法，改进了 Bellman‑Ford 在稠密图上的性能；

**💡 创新点**

通过改进 shortcutting 构造、强化 betweenness 降低、以及多尺度复制方案，使得每轮可将负边数量减少约 1/3，从而在 O(log n) 次迭代内实现近线性时间；

**🔧 技术方法**

使用负权重单源搜索、潜在函数重加权、复制顶点、层叠图实现 h‑hop 路径、随机采样、桶化/分层复制以及多跳搜索等技术；

**📊 数据集**

本文为理论性工作，没有使用实验数据集；

**📈 对比分析**

相较于之前的 O(n²·log n)、O(mn^{8/9}) 等算法，取得了 O(n²+o(1)) 的时间复杂度，在稠密图上实现了近线性运行；在稀疏图上仍与 Dijkstra 等方法相比不具优势；

**⚠️ 局限性**

算法为随机化，需多次迭代；需要在图中复制 O(log²n) 个顶点，实现复杂度高；仅在稠密图上达到最优表现，对于非常稀疏图仍不如经典算法。

---

## 182. Queer NLP: A Critical Survey on Literature Gaps, Biases and Trends

**arXiv ID:** 2602.16151 | [PDF](https://arxiv.org/pdf/2602.16151v1)

**作者:** Sabine Weber `[一作]` (University of Bamberg), Yanan Long `[通讯]` (Queer in AI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究系统性综述了ACL Anthology中所有涉及 LGBTQIA+ 社群的NLP论文，梳理研究趋势、现存缺口与未来方向；

**💡 创新点**

首次对该子领域进行全面量化与质性分析，并提出了基于利益相关者与交叉性视角的评价框架；

**🔧 技术方法**

采用PRISMA搜索与筛选流程，结合手工标注与归类编码，运用统计与可视化工具对论文进行量化描述；

**📊 数据集**

主要使用ACL Anthology检索结果、补充的非ACL会议论文以及公开的语言模型与数据集（如LLM评估模板、情感词典、hate speech语料等）作为分析素材；

**📈 对比分析**

通过对论文数量、主题、语言多样性、交叉性与利益相关者参与度的统计，展示了研究热点与偏差；在具体技术评测上，主要是对现有方法的归纳与比较，而非统一实验；

**⚠️ 局限性**

局限在于仅覆盖ACL文献且以英文为主，交叉性与多语种覆盖不足，且大部分研究缺乏真实社区参与，导致结论可能无法完全反映实际需求。

---

## 183. ModalImmune: Immunity Driven Unlearning via Self Destructive Training

**arXiv ID:** 2602.16197 | [PDF](https://arxiv.org/pdf/2602.16197v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11901 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了一种名为 ModalImmune 的训练框架，通过在训练过程中有选择性地对单一模态信息进行有控制的崩塌（Self‑Destructive Learning），使模型在面对模态缺失、损坏或恶意干扰时仍能保持稳健的预测和重构能力。

**💡 创新点**

创新点包括：① 将自毁训练视为因果干预，主动暴露模型于有损信息；② 采用频谱自适应崩塌正则化与信息增益驱动的 bandit 选择器，精准定位高影响模态；③ 通过曲率感知梯度屏蔽稳定更新过程；④ 使用认证的 Neumann‑截断双层超参数优化（BHGD）实现自动化、稳健的超参数调整。

**🔧 技术方法**

使用的技术：频谱自适应崩塌正则化、EXP3.P 信息增益 bandit、曲率感知梯度屏蔽、Neumann‑截断双层优化、属性向量生成器、对比学习、重构生成器、自动化超参数更新等。

**📊 数据集**

使用的数据集：CMU‑MOSI、CMU‑MOSEI 和 IEMOCAP 等多模态情感分析基准数据集，涵盖文本、音频、视觉三种模态。

**📈 对比分析**

与多种基线（HyCon、UniMSE、ConFEDE、MGCL、CLGSI 等）在完整模态、缺失模态与受噪声干扰的测试下进行对比。ModalImmune 在完整模态下获得最高 Acc2/Acc7/ F1 等指标；在单模态、组合缺失以及轻重噪声情况下保持性能下降不足 1%，显著优于传统缺失处理或对抗训练方法，并且收敛速度与推理延迟基本无负担。

**⚠️ 局限性**

局限性：在同时缺失多个关键模态（如音频+视觉）时仍会出现显著准确率下降；模型对极端噪声或未见模态组合的泛化尚有限；实现包含多种自毁与曲率屏蔽步骤，增加了一定的工程复杂度；理论上虽给出认证的超参数估计，但对不同数据分布的鲁棒性仍需进一步验证。

---

## 184. Rethinking Input Domains in Physics-Informed Neural Networks via Geometric Compactification Mappings

**arXiv ID:** 2602.16193 | [PDF](https://arxiv.org/pdf/2602.16193v1)

**作者:** Zhenzhen Huang `[一作]` (University of Electronic Science and Technology of China), Chaoning Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 2007 | [OpenAlex ID](https://openalex.org/A5057230698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出通过几何紧化映射重构输入坐标的GC-PINN框架，用以解决多尺度偏微分方程的梯度僵硬和收敛慢问题。

**💡 创新点**

创新点在于将可微几何紧化映射（环形、径向、局部拉伸）嵌入PINN输入层，直接调节残差算子的谱特性，从而实现训练稳定性和精度提升。

**🔧 技术方法**

技术上采用了多种可微几何映射、标准多层感知机（MLP）、Adam + L‑BFGS双阶段优化，以及自动微分求解残差和边界约束。

**📊 数据集**

实验数据集为合成的1D Burgers、1D/2D 对流扩散、1D Helmholtz 与2D Navier‑Stokes 方程，全部使用分析制造解来构造源项和边界条件。

**📈 对比分析**

与传统PINN、PINN+RAR、FF-PINN、SA-PINN和gPINN比较，GC‑PINN 在大多数指标（MSE、Rel_L²、Rel_H¹）上实现数阶误差降低、收敛更快、残差分布更均匀。

**⚠️ 局限性**

局限性包括对映射类型与超参数的选择需经验性调优，可能导致额外的计算和内存开销，并在某些复杂几何或高维场景下收敛不稳定。

---

## 185. Deep TPC: Temporal-Prior Conditioning for Time Series Forecasting

**arXiv ID:** 2602.16188 | [PDF](https://arxiv.org/pdf/2602.16188v1)

**作者:** Filippos Bellos `[一作]`, Jason J. Corso `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Temporal-Prior Conditioning (TPC)，将时间作为第一类模态，深度注入冻结的LLM进行多变量时间序列长程预测。

**💡 创新点**

在LLM中引入可学习的时间序列token，并通过跨层交叉注意力将时间先验文本嵌入与时间序列分离、层级融合，形成仅训练极少参数的深度时间条件化方法。

**🔧 技术方法**

使用冻结的GPT‑2 small LLM、patch embedding、RevIN归一化、可学习时间序列token、跨层交叉注意力、门控机制、自动回归预测以及与LoRA等对照实验。

**📊 数据集**

在ETTh1/2、ETTm1/2、Weather、Electricity、Traffic、Solar等标准长程预测数据集上进行评估。

**📈 对比分析**

与AutoTimes、S^2IP‑LLM、PatchTST、DLinear等基准对比，TPC在大多数数据集的MSE/MAE上均优于或与之持平，尤其在长程预测任务中取得SOTA表现。

**⚠️ 局限性**

仅在GPT‑2 small上验证，缺乏更大LLM的泛化评估；时间先验仅通过文本描述嵌入，未探索更丰富的多模态信息；预计算时间描述嵌入可能限制实时部署。

---

## 186. World Model Failure Classification and Anomaly Detection for Autonomous Inspection

**arXiv ID:** 2602.16182 | [PDF](https://arxiv.org/pdf/2602.16182v1)

**作者:** Michelle Ho `[一作]` (Stanford University), Shayegan Omidshafiei `[通讯]` (Field AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个融合失败分类与异常检测的实时框架，用于工业量规检查机器人在任务中早期识别成功、已知失败与未知异常三种状态。

**💡 创新点**

创新点在于将世界模型（world model）作为特征提取器，同时使用 conformal prediction（CP）阈值对成功与失败两类分别校准，实现无策略依赖、分布无关的多类别判别，并在同一架构下兼顾已知失败与 OOD 检测。

**🔧 技术方法**

核心技术包括：① 预训练的 Cosmos 视觉编码器与世界模型预测未来帧；② 采用多种 CP 分数（重建误差、潜在预测误差、马氏距离等）并对成功/失败训练集分别设定阈值；③ 通过最大分数阈值决策函数实现早期检测。

**📊 数据集**

使用从 2025 年 7 月至 8 月在办公室与工业现场收集的 290 条视频数据集（14 成功+14 失败训练；6 成功+6 失败验证；45 成功+37 失败 CP 校准；45 成功+37 失败+53 OOD 测试），平均时长约 10 秒。

**📈 对比分析**

与人工观察和单一 OOD 检测方法对比，所提方法在 90% CP 阈值下实现了约 91% 的三类分类准确率，且平均提前 1–3 秒做出判断；马氏距离虽最高但易过拟合。

**⚠️ 局限性**

局限性包括：① CP 阈值为静态，难以应对运行时分布漂移；② 需要外部推理导致网络延迟；③ 仅使用视觉信息，未结合语义或历史上下文；④ 需多模型维护，计算开销相对较大。

---

## 187. Towards Secure and Scalable Energy Theft Detection: A Federated Learning Approach for Resource-Constrained Smart Meters

**arXiv ID:** 2602.16181 | [PDF](https://arxiv.org/pdf/2602.16181v1)

**作者:** Diego Labate `[一作]` (University of Calabria), Giancarlo Fortino `[通讯]` (University of Calabria)

**通讯引用:** 26015 | [OpenAlex ID](https://openalex.org/A5023424929)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在资源受限的智能电表上实现的联邦学习框架，用轻量级 MLP 检测电力盗用，并在本地模型更新中注入高斯噪声以实现差分隐私。

**💡 创新点**

创新点在于：① 将联邦学习与差分隐私结合，提供形式化的隐私保证；② 采用极简 MLP 结构，显著降低计算和通信开销；③ 在 IID 与非 IID 数据分布下均实现良好性能，表明框架对数据异质性具有鲁棒性。

**🔧 技术方法**

使用的技术包括：联邦平均（FedAvg）、随机梯度下降（SGD）优化、Gaussian 差分隐私噪声、z-score 特征标准化、Python PyTorch 实现。

**📊 数据集**

使用中国国家电网公司（SGCC）公开的 42,372 位用户、1035 天的用电数据，包含 3,615 名盗电者（约 9%）。

**📈 对比分析**

与现有 FL-CNN、FedDetect 等方法比较，实验显示在 80 轮训练下，本方法在准确率、AUC、通信成本等指标上均达到或超过 SOTA，且在 5 名客户端时通信成本仅为 305 MB，显著低于传统方案。

**⚠️ 局限性**

局限性：① 仅采用简单 MLP，未利用时间序列或更深网络可能限制对复杂盗电行为的捕捉；② 差分隐私噪声可能在极低隐私预算下影响模型精度；③ 研究未考虑客户端掉线、异步更新和真实网络延迟等实际部署挑战。

---

## 188. EnterpriseGym Corecraft: Training Generalizable Agents on High-Fidelity RL Environments

**arXiv ID:** 2602.16179 | [PDF](https://arxiv.org/pdf/2602.16179v1)

**作者:** Sushant Mehta `[一作]` (Surge AI), Edwin Chen `[通讯]` (Surge AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在高保真企业模拟环境Corecraft中，利用强化学习（GRPO + 自适应裁剪）对GLM 4.6进行单轮训练，使其在客户支持任务上显著提升；

**💡 创新点**

提出了基于专家编写的细粒度评估 rubrics 的奖励机制，并证明高质量、多样化、现实化的训练环境能在分布外基准上实现跨域迁移；

**🔧 技术方法**

使用 Group Relative Policy Optimization (GRPO)、自适应裁剪、LLM 判别器评估 rubrics 以及 Docker 容器化的模拟环境；

**📊 数据集**

核心数据集为 Corecraft 训练集（约2500实体、23种工具的企业客户支持任务）和内部 hold‑out 评估集；

**📈 对比分析**

与 GPT‑5.2、Claude Opus 等前沿模型对比，单轮训练后 GLM 4.6 的任务通过率从25.37% 提升至36.76%（+11.39pp），并在 BFCL Parallel (+4.5pp)、τ^2‑Bench Retail (+7.4pp) 与 Toolathlon (+6.2pp) 等外部基准上实现显著提升；

**⚠️ 局限性**

局限性包括仅单轮训练、仅针对 GLM 4.6、缺乏不同模型或更深层次 ablation、对其他行业场景的适应性待验证，以及对环境重现成本和规模的挑战。

---

## 189. Human-AI Collaboration in Large Language Model-Integrated Building Energy Management Systems: The Role of User Domain Knowledge and AI Literacy

**arXiv ID:** 2602.16140 | [PDF](https://arxiv.org/pdf/2602.16140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 190. Learning Personalized Agents from Human Feedback

**arXiv ID:** 2602.16173 | [PDF](https://arxiv.org/pdf/2602.16173v1)

**作者:** Kaiqu Liang `[一作]` (Meta Superintelligence Labs), Saghar Hosseini `[通讯]` (Meta Superintelligence Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Personalized Agents from Human Feedback (PAHF) 框架，使 AI 代理能够通过持续学习和显式记忆实时对用户偏好进行个性化。

**💡 创新点**

创新点在于双向反馈机制：预先行动时主动澄清（pre‑action），行动后及时纠正（post‑action），并将两者与显式记忆相结合，实现对不确定性和偏好漂移的补偿。

**🔧 技术方法**

使用大型语言模型（如 GPT‑4o）结合 ReAct 交互式推理，检索增强生成（RAG）+密集检索记忆（SQLite/FAISS），以及 LLM‑判定的显著性检测和摘要更新。

**📊 数据集**

实验数据集包括：1）嵌入式操作基准——40 名用户、每阶段 30 个场景；2）在线购物基准——20 名用户、每阶段 45 个场景；两套数据均包含原始与演化的用户偏好配置。

**📈 对比分析**

与 No‑Memory、Pre‑Action‑Only、Post‑Action‑Only 等基线在四阶段评估中比较；PAHF 在成功率、累计个性化错误（ACPE）和反馈频率上均领先（Embodied 约70%，Shopping 约70%），证明双向反馈与显式记忆的优势。

**⚠️ 局限性**

局限性包括：依赖 LLM 作为人类模拟与判定，可能在真实用户场景中产生误判；记忆结构简单，扩展性待验证；实验规模有限，难以全面评估大规模多用户情境下的鲁棒性。

---

## 191. "You Can Actually Do Something": Shifts in High School Computer Science Teachers' Conceptions of AI/ML Systems and Algorithmic Justice

**arXiv ID:** 2602.16123 | [PDF](https://arxiv.org/pdf/2602.16123v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 192. Collection: UAV-Based Wireless Multi-modal Measurements from AERPAW Autonomous Data Mule (AADM) Challenge in Digital Twin and Real-World Environments

**arXiv ID:** 2602.16163 | [PDF](https://arxiv.org/pdf/2602.16163v1)

**作者:** Md Sharif Hossen `[一作]` (North Carolina State University), Rudra Dutta `[通讯]` (North Carolina State University)

**通讯引用:** 3008 | [OpenAlex ID](https://openalex.org/A5055485742)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了无人机作为数据 mule 的自主数据收集与基地站关联决策，并提供了公开的多模态数据集；

**💡 创新点**

创新点在于将高保真数字孪生与真实场景实验相结合，构建了完整的无人机轨迹规划、基站选择与多源感知融合平台，首次公开多基站动态关联与数据下载的全链路记录；

**🔧 技术方法**

使用了无人机实时控制、SDR（USRP）探测、LoRa 低功耗接收、Keysight RFS 位置估计、Fortem R20 雷达跟踪、数字孪生仿真、Python/MATLAB 数据处理脚本；

**📊 数据集**

数据集为 AERPAW AADM Challenge 公开数据集，包含 DT 与 RW 两个环境的 UAV 轨迹、位置、速度、姿态、SNR、数据下载量、雷达、RFS、LoRa 测量；

**📈 对比分析**

通过将团队在三种不同数据需求场景下的 DT 与 RW 分数进行对比，评估算法在真实环境中的泛化能力；总体来看，表现稳定的团队在所有场景中获得更高排名，单场景优秀未必转化为整体优胜；

**⚠️ 局限性**

限制包括：仅覆盖四个基站、缺乏详细能耗/电池使用数据、实验仅针对单无人机、环境干扰与多无人机协同未涉及，未来需扩展场景复杂度与能耗建模。

---

## 193. Balancing Faithfulness and Performance in Reasoning via Multi-Listener Soft Execution

**arXiv ID:** 2602.16154 | [PDF](https://arxiv.org/pdf/2602.16154v1)

**作者:** Nithin Sivakumaran `[一作]` (University of North Carolina Chapel Hill), Elias Stengel-Eskin `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多方强化学习框架（MAT-STEER），通过让一个说话模型生成推理链并让多个听众模型从截断的前缀继续推理，以奖励推理链的可执行性，从而提升链式思考（CoT）的可信度和解释性。

**💡 创新点**

创新点在于①引入多听众软执行奖励机制，将推理链的可执行性视为多方一致性；②将正确性与可信度分离，使用蒙版监督微调（SFT）与LoRA仅调整答案端口，避免单一RL奖励导致性能下降；③通过多监听器多截断点的训练显著提升推理链的简洁性和线性。

**🔧 技术方法**

核心技术包括 Group Relative Policy Optimization (GRPO) 强化学习、软执行（Listener执行前缀推理）奖励、蒙版监督微调（SFT）+ LoRA、截断比例（25/50/75%）以及多监听器聚合一致性奖励。

**📊 数据集**

训练数据取自 Big-Bench Hard (BBH) 子集，评估数据涵盖 BIG-Bench Extra Hard、ZebraLogicBench、MuSR 与 FOLIO 四大多步推理基准。

**📈 对比分析**

与原始模型、单一奖励（仅可信度或仅正确性）、Hint-Optimization、MAT-Steer 等基线比较，MAT-STEER 在 hint attribution、AOC（早期回答与错误注入）等可信度指标上提升约 2–4 分，同时保持甚至提升任务准确率（最高提升 3.2%），且在 legibility 评分上亦优于对照组。

**⚠️ 局限性**

局限性包括：仍需在训练阶段使用多模型成本，训练过程对超参敏感；在域外数据上的可信度提升有限，过度聚焦可执行性可能导致推理链缺乏多样性；且虽然准确率基本不下降，但在极端复杂任务上仍存在可信度与准确率的权衡。

---

## 194. Retrieval Collapses When AI Pollutes the Web

**arXiv ID:** 2602.16136 | [PDF](https://arxiv.org/pdf/2602.16136v1)

**作者:** Hongyeon Yu `[一作]` (NAVER Corporation), Young-Bum Kim `[通讯]` (NAVER Corporation)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实证检验网络中AI生成内容导致检索崩溃的两阶段过程，评估SEO和攻击式污染对检索管线的影响。

**💡 创新点**

提出Retrieval Collapse概念，系统性分析SEO与恶意污染的传播动态，并量化池污染率到曝光污染率的关系。

**🔧 技术方法**

利用MS MARCO数据集构建查询–答案对，采用GPT‑5系列模型生成合成文档，使用BM25和LLM Ranker进行检索与重排，LLM Judge评估答案正确性。

**📊 数据集**

使用MS MARCO passage ranking数据集、从Google Search API检索得到的原始网页集合，以及通过GPT生成的SEO和Abuse Pool文档。

**📈 对比分析**

通过比较BM25与LLM Ranker在两种污染场景下的Pool Contamination Rate、Exposure Contamination Rate、Citation Contamination Rate和Answer Accuracy，结果显示SEO污染导致ECR超过80%，攻击污染下BM25暴露约19%有害文档，LLM Ranker能有效抑制。

**⚠️ 局限性**

实验仅在受控环境中进行，SEO生成基于通用LLM并非专家级，缺乏真实大规模Web环境验证，指标仅描述性，未探究自主AI主动操控检索的情况。

---

## 195. OmniCT: Towards a Unified Slice-Volume LVLM for Comprehensive CT Analysis

**arXiv ID:** 2602.16110 | [PDF](https://arxiv.org/pdf/2602.16110v1)

**作者:** Tianwei Lin `[一作]` (Zhejiang University), Beng Chin Ooi `[通讯]` (Zhejiang University)

**通讯引用:** 21209 | [OpenAlex ID](https://openalex.org/A5024892041)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种统一的CT切片-体积视觉语言模型OmniCT，解决了传统2D和3D LVLM在跨切片空间一致性与体积语义方面的不足；

**💡 创新点**

引入空间一致性增强（SCE）模块和器官级语义增强（OSE）模块，实现切片与体积的无缝对齐、3D位置编码以及器官级信息聚合；

**🔧 技术方法**

使用卷积+视觉Transformer编码器、三轴位置嵌入、MoE混合投影、器官掩模定位与自适应聚合等技术；

**📊 数据集**

构建并公开MedEval-CT数据集，包含1.7M VQA样本（7类任务、4级临床难度、13器官分布），覆盖切片和3D体积两种输入；

**📈 对比分析**

在多种公开2D/3D CT基准（SLAKE、VQA-RAD、OmniMedVQA、RadFig-VQA、M3D、CT-RATE、3D-RAD）上与多类基线模型对比，OmniCT在7B规模下平均提升约+10-15分，显著优于现有医疗及通用LVLM；

**⚠️ 局限性**

对比实验中仍需进一步验证不同3D编码器的优势，模型在极小器官和复杂解剖结构上的表现仍有提升空间，且对多模态混合训练的机制解释仍有限。

---

## 196. Fast KV Compaction via Attention Matching

**arXiv ID:** 2602.16284 | [PDF](https://arxiv.org/pdf/2602.16284v1)

**作者:** Adam Zweiger `[一作]` (Massachusetts Institute of Technology), Yoon Kim `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 21973 | [OpenAlex ID](https://openalex.org/A5100693798)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于Attention Matching的KV缓存压缩方法，能够在一次性压缩过程中快速减少长上下文的KV存储量；

**💡 创新点**

创新点在于将KV压缩问题转化为匹配注意力输出和注意力质量的目标，并通过引入可学习的标量偏置、闭式子问题（非负最小二乘、最小二乘、OMP）以及非均匀头预算等技术，实现无梯度、几乎无训练的高效压缩；

**🔧 技术方法**

主要技术包括Attention Matching目标、非负最小二乘（NNLS）拟合偏置、最小二乘拟合values、正交匹配追踪（OMP）/最高注意力关键选择、分块压缩、RoPE相位对齐、可变长度KV存储以及自学习/重复预填充产生参考查询；

**📊 数据集**

实验使用的模型为Qwen3‑4B、Llama3.1‑8B、Gemma3‑12B，在长文本推理基准QuALITY（5–8k token）和LongHealth（60k token）上进行评估；

**📈 对比分析**

与Cartridges、token‑pruning（H2O、SnapKV、PyramidKV、KVzip）和summarization等基线对比，Attention Matching在50×压缩率下保持与Cartridges相近甚至更好的下游QA准确率，压缩速度提升约两位数（从数小时降至数分钟），在某些设置可实现200×压缩且性能仅略低于单纯summarization；

**⚠️ 局限性**

局限性包括：尽管比梯度优化快，但在极端压缩率（如100×）下性能略逊于Cartridges；查询生成阶段仍占主要计算成本；目前仅针对一次性压缩，在线/连续压缩仍需进一步研究；

---

## 197. Are LLMs Ready to Replace Bangla Annotators?

**arXiv ID:** 2602.16241 | [PDF](https://arxiv.org/pdf/2602.16241v1)

**作者:** Md. Najib Hasan `[一作]` (Wichita State University), Souvika Sarkar `[通讯]` (Wichita State University)

**通讯引用:** 2135 | [OpenAlex ID](https://openalex.org/A5032176749)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了17种大语言模型在低资源语言孟加拉语中的仇恨言论自动标注性能，并提出了统一的三维评估框架（分类、推理一致性、提示鲁棒性）。

**💡 创新点**

创新点包括①构建首个涵盖21个身份敏感类别的孟加拉语仇恨言论数据集；②提出集成分类、推理一致性和提示鲁棒性为一体的评估框架；③发现模型规模与标注质量不成正比，推理流畅不代表标注准确，并揭示了身份偏见与提示敏感性。

**🔧 技术方法**

使用技术包括零样本提示、统一评估框架、BERTScore与余弦相似度评估推理一致性、TELeR分类的四级提示变化、F1/Kappa等指标。

**📊 数据集**

数据集为Bangla_HateSpeech (BAHS)，共5000条样本，涵盖5个领域（性别、宗教、娱乐、地缘政治、政治）与21个身份标签。

**📈 对比分析**

比较方法：在三维框架下计算F1、Cohen's Kappa、身份匹配误差、推理相似度与提示一致性；将模型划分为Human-like、Objective、Adaptive三类。性能显示，最佳模型DeepSeek R1 Distill (70B) 在分类与推理均可达约85%的F1/Kappa，但仍低于人类水平；多数模型分类准确率低于50%，对提示和身份高度敏感。

**⚠️ 局限性**

局限性：仅针对孟加拉语；提示模板人为设计可能缺乏生态真实性；评估指标对幻觉推理捕捉不足；数据集规模有限；模型表现易随微调或部署环境变化而变动。

---

## 198. EasyControlEdge: A Foundation-Model Fine-Tuning for Edge Detection

**arXiv ID:** 2602.16238 | [PDF](https://arxiv.org/pdf/2602.16238v1)

**作者:** Hiroki Nakamura `[一作]` (Panasonic Holdings Corporation), Tadahiro Taniguchi `[通讯]` (Kyoto University)

**通讯引用:** 2705 | [OpenAlex ID](https://openalex.org/A5023160093)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出EasyControlEdge，利用图像生成基础模型实现精准、数据高效且可控的边缘检测

**💡 创新点**

通过轻量级条件注入LoRA、像素级监督与迭代流匹配，将大规模预训练模型的高频细节生成能力迁移到边缘检测任务，并在推理时实现可调节边缘密度的控制

**🔧 技术方法**

轻量级条件注入（LoRA）、流匹配（Flow Matching）与迭代推理、像素级加权交叉熵损失、无监督指导（CFG）

**📊 数据集**

BSDS500、NYUDv2、BIPED、CubiCasa四个公开边缘/墙壁检测数据集

**📈 对比分析**

与DiffusionEdge、GED等最先进方法对比，EasyControlEdge在CEval（原始预测）上显著提升F-score，在有限样本（10%）下仍保持高性能，并能通过调整guidance scale控制边缘密度

**⚠️ 局限性**

仅在FLUX.1-dev+EasyControl框架上验证，未测试其他基础模型或适配方法；当基础模型或适配方式改变时需进一步验证其可迁移性

---

## 199. Graph neural network for colliding particles with an application to sea ice floe modeling

**arXiv ID:** 2602.16213 | [PDF](https://arxiv.org/pdf/2602.16213v1)

**作者:** Ruibiao Zhu `[一作]` (Australian National University), Ruibiao Zhu `[通讯]` (Australian National University)

**通讯引用:** 5506 | [OpenAlex ID](https://openalex.org/A5058693222)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于图神经网络的海冰碎片碰撞模拟方法（CN），通过构建固定邻接图并结合数据同化实现对海冰动力学的高效预测。

**💡 创新点**

创新点在于将碰撞相关几何信息嵌入边特征、使用Mish激活函数、仅依赖两步位置信息推断速度，并在一维场景下实现了比传统DEM更快、更准确的仿真。

**🔧 技术方法**

采用Graph Neural Network（CN）、数据同化（Ensemble Kalman Filter/Ensemble Transform Kalman Filter）、以及对比基准（Interaction Network、Graph Network-based Simulator）。

**📊 数据集**

使用自生成的无量纲一维DEM数据集，包含10个和30个冰块的碰撞轨迹，涵盖初始速度、位置等参数。

**📈 对比分析**

与IN、GNS对比，CN在单步和长时序评估中取得PCC>0.93、RMSE显著降低，推理时间仅比DEM少约60%且在30个冰块时提升约63%；在长达20,000步的模拟中仍保持PCC>0.87。

**⚠️ 局限性**

主要限制是仅在一维无旋转、无摩擦的设置下验证，未能覆盖二维真实海冰的旋转、切向力和摩擦等复杂物理效应。

---

## 200. Near-optimal population protocols on bounded-degree trees

**arXiv ID:** 2602.16222 | [PDF](https://arxiv.org/pdf/2602.16222v1)

**作者:** Joel Rybicki `[一作]` (Humboldt University of Berlin), Robin Vacus `[通讯]` (Sorbonne University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了稀疏交互图中种群协议的时空权衡，特别是在有界度树上的领导者选举和精确多数问题上，提出了常数空间的协议，具有近似最优的最坏情况稳定时间。

**💡 创新点**

提出了在有界度树上，种群协议不表现出显著的渐近时空权衡的结果，首次展示了在这一类图中，领导者选举和精确多数问题的常数状态协议可以在最优时间内解决。

**🔧 技术方法**

使用了新的自稳定2跳着色协议和自稳定树定向算法，这些算法基于随机漂移论证来界定稳定时间，并且在有界度图上实现了时间最优。

**📊 数据集**

使用了有界度树作为数据集，研究了在该数据集上领导者选举和精确多数问题的解决方案。

**📈 对比分析**

与现有方法相比，提出的协议在有界度树上实现了线性加速，特别是在领导者选举和精确多数问题上，稳定时间为O(Dn)和O(n^2 log n)，而现有协议的时间复杂度较高。

**⚠️ 局限性**

限制在于这些协议主要针对有界度树，且在其他类型的图上可能不适用，此外，尽管在有界度树上表现良好，但在更一般的图类中可能存在更高的空间和时间复杂度。

---

## 201. Long-Tail Knowledge in Large Language Models: Taxonomy, Mechanisms, Interventions and Implications

**arXiv ID:** 2602.16201 | [PDF](https://arxiv.org/pdf/2602.16201v1)

**作者:** Sanket Badhe `[一作]` (Google), Nehal Kathrotia `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统总结并构建了长尾知识（Long‑Tail Knowledge）在 LLM 中的分类、机制、干预策略和社会技术影响的统一分析框架。

**💡 创新点**

首次将长尾知识问题按定义、机制、干预和影响四个维度整体归纳，揭示了评估方法、治理与隐私等空白与交互依赖。

**🔧 技术方法**

综述方法结合频率/稀疏性分析、tokenization 影响、梯度稀释、对齐压缩、检索增强生成（RAG）、Mixture‑of‑Experts、模型编辑、RLHF 等技术。

**📊 数据集**

利用公开语料与基准（Common Crawl、MMLU、BIG‑bench、CPopQA、CANDLE、CAMeL 等）进行文献检索与案例对照。

**📈 对比分析**

通过对比各类干预措施与现有评估基准，发现对长尾知识的提升有限且往往伴随记忆衰减、偏见放大或计算成本上升；未给出统一实验性能度量，主要以文献证据为依据。

**⚠️ 局限性**

局限在于缺乏统一、可操作的长尾知识度量与评估方法，研究多为综述与理论分析，缺少系统实验验证与可解释性工具；同时面临隐私、可持续性和治理的跨领域挑战。

---

## 202. Mind the Gap: Evaluating LLMs for High-Level Malicious Package Detection vs. Fine-Grained Indicator Identification

**arXiv ID:** 2602.16304 | [PDF](https://arxiv.org/pdf/2602.16304v1)

**作者:** Ahmed Ryan `[一作]` (University of Alabama), Md Rayhanur Rahman `[通讯]` (University of Alabama)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对13种大型语言模型在PyPI恶意Python包检测和细粒度恶意指标识别上的性能进行系统评估

**💡 创新点**

发现LLM存在显著的粒度差距：包级检测几乎完美，但细粒度指标识别下降约41%；并首次量化提示、温度、参数规模、上下文宽度等因素对性能的影响

**🔧 技术方法**

使用提示工程（Zero‑Shot、Few‑Shot、Chain‑of‑Thought、Self‑Consistency、Tree‑of‑Thought）以及不同温度设置，对模型进行多配置实验，并采用F1、宏/微/加权F1等标准指标评估

**📊 数据集**

使用精心构建的4,070个Python包数据集（3,700个正常、370个恶意），恶意包包含47个标注的恶意指标

**📈 对比分析**

通过采样与多配置比较，结果显示GPT‑4.1在包检测上F1≈0.99，LLAMA在多标签任务中加权F1≈0.58，提示复杂度和温度对细粒度识别影响有限

**⚠️ 局限性**

局限包括样本比例偏差（10:1）、仅关注PyPI生态、模型可能记忆导致的过拟合、输出非确定性、以及缺乏对细粒度语义理解的深度

---

## 203. The Validity of Coreference-based Evaluations of Natural Language Understanding

**arXiv ID:** 2602.16200 | [PDF](https://arxiv.org/pdf/2602.16200v1)

**作者:** Ian Porada `[一作]` (McGill University), Ian Porada `[通讯]` (McGill University)

**通讯引用:** 31 | [OpenAlex ID](https://openalex.org/A5008161413)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将测量有效性框架引入核心ference评估，系统性评估了现有基准的有效性，并提出了一种新的基于事件可行性推理的受控评估方法，以检验语言模型在核心ference推理上的一致性和泛化能力。

**💡 创新点**

创新点包括：①首次将测量有效性（contestedness、convergent validity 等）与核心ference评估结合，揭示传统基准结论的不确定性；②设计并实现了一个聚焦事件可行性推理的受控评估，发现大型语言模型在此维度上表现不一致；③提出混合推理（将提示式LM与监督CR模型集成）提升不同数据集上的整体准确率，进一步验证了评估方法的实用性。

**🔧 技术方法**

使用的技术主要包括：提示式大规模语言模型（Llama 3.1 8B/70B、Mistral、OLMo 等）、监督式核心ference系统（Maverick、LingMess 等）、测量有效性分析框架、受控实验设计（事件可行性推理）、集成推理策略。

**📊 数据集**

使用的数据集包括：传统核心ference基准（OntoNotes、OntoGUM、GUM、PreCo 等）、挑战集（Winograd Schema、Winogrande、DPR 等），以及自制的 11 个 PCR 评估集（涵盖自然文本中的可指代词和挑战集实例）。

**📈 对比分析**

与现有方法比较时，提示式LM在 Winograd 风格的挑战集上取得接近人类水平的准确率，但在自然文本核心ference评估中表现低于监督模型；混合模型在两类数据集上均优于单一方法，整体准确率提升约 3–5%（取决于数据集）。

**⚠️ 局限性**

局限性包括：①评估数据集可能存在训练集泄漏；②测量有效性框架未完全覆盖所有核心ference细粒度现象；③受控实验聚焦事件可行性推理，未覆盖更广泛的语义推理任务；④对非英语/低资源语言的泛化性未知。

---

## 204. Markerless Robot Detection and 6D Pose Estimation for Multi-Agent SLAM

**arXiv ID:** 2602.16308 | [PDF](https://arxiv.org/pdf/2602.16308v1)

**作者:** Markus Rueggeberg `[一作]` (German Aerospace Center), Riccardo Giubilato `[通讯]` (German Aerospace Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了基于深度学习的无标记机器人姿态估计，并将其集成到分布式多机器人SLAM系统中，以提升机器人间的互定位精度。

**💡 创新点**

首次利用已知机器人形状的无标记姿态估计实现多机器人互定位，克服了传统fiducial（如AprilTag）在距离、视角和光照条件上的限制。

**🔧 技术方法**

采用YOLOv7目标检测 + Transformer-based 2D-3D对应预测 + 神经PnP回归 + amodal mask 辅助 + iSAM2 优化的分布式SLAM框架。

**📊 数据集**

使用BlenderProc和OAISYS生成的合成数据（8000张图像）以及真实的Mt. Etna 现场数据进行训练与评估。

**📈 对比分析**

与仅使用AprilTag的基线方法对比，检测率提升87%，最大检测距离提升227%，轨迹误差下降28%，在现场测试中显著缩短了开环导航时间（最大开环持续时间下降55%）。

**⚠️ 局限性**

对极近距离下的旋转误差仍高于标记法，依赖于已知的CAD模型，且对多关节或非刚性机器人适用性有限。

---

## 205. Dynamic and Streaming Algorithms for Union Volume Estimation

**arXiv ID:** 2602.16306 | [PDF](https://arxiv.org/pdf/2602.16306v1)

**作者:** Sujoy Bhore `[一作]` (Indian Institute of Technology Bombay), Yanheng Wang `[通讯]` (Saarland University)

**通讯引用:** 378 | [OpenAlex ID](https://openalex.org/A5052245256)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文提出了三类新的算法，用于在oracle模型下动态估计几何对象集合的并集体积：①支持插入和删除的完全动态算法，②支持插入和后缀查询（滑动窗口）的一致流算法，③支持插入和删除的凸体流算法。

**💡 创新点**

主要创新点包括：
- 首次给出支持插入与删除的动态算法，能够在多维任意对象（包括二维三角形、三维简单x体等）上以多项式时间保持(1±ε)体积估计；
- 在滑动窗口和后缀查询设置下实现了polylog时间与空间的一致流解法；
- 通过将连续问题离散化并结合稀疏恢复与整数规划，扩展到任意维度常数维的凸体，得到低空间、低时间的流式估计。

**🔧 技术方法**

核心技术包括：
- 基于l‑sample的随机抽样与估计，并利用自适应级别维护样本集；
- 采用改进的对数方法（log‑method）将删除性结构升级为完全动态结构；
- 在滑动窗口中引入时间戳和阈值维护，保证只保留最近的样本；
- 对凸体使用John椭圆、旋转盒逼近以及整数规划枚举，以实现稀疏向量的弱采样；
- 利用稀疏恢复的数据结构对弱采样结果进行压缩与查询。

**📊 数据集**

论文为理论算法研究，没有使用实际数据集；实验部分通过理论分析和概率界定来证明算法的时间、空间复杂度和误差。

**📈 对比分析**

在理论上，动态算法的摊销更新/查询时间为O(log⁵(n/ε²))，空间为O(n + log⁴(n/ε²))；后缀查询算法的更新/查询时间为O(log²(n/ε²))，空间为O(log(Δ)·log(n/ε²))；凸体流算法的更新/查询时间和空间均为O((nR/r)/ε²)，其中R/r为几何尺度比。与先前的插入仅算法相比，提升了删除支持并保持polylog级别的性能。

**⚠️ 局限性**

局限性包括：
- 动态算法在最坏情况下需要线性空间，难以进一步压缩；
- 凸体流算法仅适用于常数维度，且对对象的尺度比（R/r）要求较大；
- 目前对高维非凸体的动态估计仍未给出有效方法。

---

## 206. Multi-agent cooperation through in-context co-player inference

**arXiv ID:** 2602.16301 | [PDF](https://arxiv.org/pdf/2602.16301v1)

**作者:** Marissa A. Weis `[一作]`, Alexander Meulemans `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在迭代囚徒困境中，用混合训练（对学习代理与预设表格代理混合对战）训练序列模型代理，使其在单轮内实现最佳响应，并通过相互胁迫最终实现合作。

**💡 创新点**

创新点在于证明：无须显式的协作者学习意识、元梯度或时间尺度分离，单纯的多样化对手分布即可促使代理通过情境学习产生合作行为；将情境学习与协作者学习意识结合，提供了一种可扩展、计算高效的合作多智能体学习框架。

**🔧 技术方法**

使用了：分布式多智能体强化学习 (Decentralized MARL)、优势演员-评论家 (A2C)、基于序列模型的 Predictive Policy Improvement (PPI) 算法、序列模型自监督预训练、Monte Carlo 价值估计、混合训练池。

**📊 数据集**

数据集：迭代囚徒困境的交互序列；对手包括 5 维参数化的表格策略代理（在 5 维空间均匀采样）和随机初始化的学习代理，全部在模拟环境中自行生成。

**📈 对比分析**

与仅对学习代理训练的对照组以及给定显式协作者身份的实验进行 ablation。结果显示，混合训练的 A2C 与 PPI 代理在 100 轮实验中均能实现高比例合作，而对照组在多数实验中退化为互相背叛；性能差异显著，标准误差可视化表明结果稳健。

**⚠️ 局限性**

局限性：实验仅在两智能体的迭代囚徒困境中验证，缺乏对更大规模或更复杂游戏（如公共物品、团队协作）的通用性证明；混合池中的表格代理相对简单，可能不足以捕捉真实世界对手的多样性；同时，仍需更多实证验证该机制在真实分布式系统中的可迁移性。

---

## 207. Aladdin-FTI @ AMIYA Three Wishes for Arabic NLP: Fidelity, Diglossia, and Multidialectal Generation

**arXiv ID:** 2602.16290 | [PDF](https://arxiv.org/pdf/2602.16290v1)

**作者:** Jonathan Mutal `[一作]` (Universite de Geneve), Pierrette Bouillon `[通讯]` (Universite de Geneve)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Aladdin‑FTI 模型，结合机器翻译与指令式下的下一个词生成，支持多种阿拉伯方言的生成与互译。

**💡 创新点**

首次采用联合训练目标（MT+GEN），使同一小模型同时兼顾 diglossia 与方言忠诚度，并在闭源数据轨道上取得与大型模型相当的性能。

**🔧 技术方法**

基于 SmolLM3‑3B（和 Llama‑3.1‑8B‑Instruct 进行 LoRA 微调），使用指令化训练格式、交叉任务损失组合以及多种解码策略。

**📊 数据集**

仅使用官方闭源数据集，包括 EN↔MSA↔DA 平行语料、MADAR、FLORES、HABIBI 等，覆盖摩洛哥、埃及、巴勒斯坦、叙利亚、沙特等方言。

**📈 对比分析**

通过与 MT、GEN 单目标模型以及 GPT‑OSS‑120B、Command R Arabic 等大型基线比较，使用 ChrF++ 与 Macro ADI2 评估；联合 MT+GEN 模型在 diglossia 与方言忠诚度上实现平衡，SmolLM3‑3B 在闭源任务上与 8B 大模型相当。

**⚠️ 局限性**

仅测试单一模型架构，未验证不同规模或模型族的泛化；未考虑少样本/对话式学习；仅依赖官方闭源语料，未使用外部资源。

---

## 208. Toward Scalable Verifiable Reward: Proxy State-Based Evaluation for Multi-turn Tool-Calling LLM Agents

**arXiv ID:** 2602.16246 | [PDF](https://arxiv.org/pdf/2602.16246v1)

**作者:** Yun-Shiuan Chuang `[一作]` (PayPal AI), Prakhar Mehrotra `[通讯]` (PayPal AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出代理状态评估框架，利用LLM推断代理状态进行最终状态评估，替代传统完整确定性后端；

**💡 创新点**

核心创新在于不需要构建完整数据库后端，使用LLM生成代理状态并由LLM判别器评估，显著降低工程成本同时保持评估可靠性；

**🔧 技术方法**

采用LLM代理状态追踪器、LLM判别器、用户/工具仿真器以及ReAct式推理代理，整合多模型LLM组件完成仿真与评估；

**📊 数据集**

构建了208个人工合成场景（157个训练、51个测试），覆盖电商和账户管理等工业工作流；

**📈 对比分析**

对多种LLM（GPT‑5、GPT‑4o、Gemini‑2.5、Qwen‑30B等）进行基准比较，发现模型规模和推理力度越大，目标完成率（GC）越高；同时使用RFT、SFT等训练方法验证了其提升效果；

**⚠️ 局限性**

局限包括对LLM推断质量高度敏感、场景设计需要人工投入、工具/用户hallucination仍存在一定风险、以及未充分验证跨域泛化能力。

---

## 209. HyPCA-Net: Advancing Multimodal Fusion in Medical Image Analysis

**arXiv ID:** 2602.16245 | [PDF](https://arxiv.org/pdf/2602.16245v1)

**作者:** J. Dhar `[一作]`, N. Zaidi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计并实现了一种Hybrid Parallel-Cascade Attention Network（HyPCA-Net），用于多模态医学图像融合，兼顾性能与低计算成本。

**💡 创新点**

提出残差自适应学习注意力（RALA）与双视图级联注意力（DVCA），将并行空间‑通道融合与级联混合空间/双域建模相结合，形成混合注意力机制，解决信息损失与高计算成本问题。

**🔧 技术方法**

采用残差连接、跨尺度卷积、空间‑通道并行注意力、频域与空间域联合注意力、混合空间频域融合、卷积神经网络+注意力模块，以及多任务学习框架。

**📊 数据集**

在十个公开医学影像分类数据集（如 Nickparvar、IQ-OTHNCCD、Tuberculosis、CNMC-2019、HAM10000、SIPaKMeD、CRC、CBIS-DDSM 等）以及两个分割数据集（肺部 CT、皮肤病变）上进行评估。

**📈 对比分析**

与多种单模态与多模态基线（如 DDA-Net、MSCAM、MFMSA、Gloria、MTTU-Net、MuMu、M^3Att、DRIFA-Net、HyPCA-Net 等）对比，HyPCA-Net 在分类任务中平均提升约5.2% 绩效，参数量降低约73%，在分割任务中亦取得最高 Dice/IoU 结果。

**⚠️ 局限性**

目前缺乏对抗鲁棒性与跨域适应的验证，模型在极端噪声或数据分布漂移下的泛化能力尚待提升。

---

## 210. DistributedEstimator: Distributed Training of Quantum Neural Networks via Circuit Cutting

**arXiv ID:** 2602.16233 | [PDF](https://arxiv.org/pdf/2602.16233v1)

**作者:** Prabhjot Singh `[一作]` (University of Melbourne), Rajkumar Buyya `[通讯]` (University of Melbourne)

**通讯引用:** 105997 | [OpenAlex ID](https://openalex.org/A5014716105)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了面向分布式的量子神经网络训练管线，将电路切割视为分阶段并行工作并进行细粒度计时

**💡 创新点**

首次从系统角度量化切割对训练管线的整体开销、重构瓶颈、分布式调度与鲁棒性影响，并给出可扩展的测量框架

**🔧 技术方法**

使用Qiskit、qiskit-addon-cutting、PyTorch、Ray/线程池等实现分布式估计器，结合参数位移梯度、经典重构及可配置调度策略

**📊 数据集**

在Iris与MNIST两个二分类任务上进行实验，采用统一的子实验shots与训练预算

**📈 对比分析**

通过对比无切割与1/2/3个切割的总训练时间、加速比、重构占比、准确率与鲁棒性，发现切割导致显著增时，但准确率不退，鲁棒性保持或略提升；加速受重构束缚，straggler敏感性取决于重构占比

**⚠️ 局限性**

主要局限在于仅使用模拟器与线程池；重构仍是串行瓶颈，缺乏分布式/增量重构与自适应shots策略；实验规模与硬件多样性有限

---

## 211. When to Identify Is to Control: On the Controllability of Combinatorial Optimization Problems

**arXiv ID:** 2602.16311 | [PDF](https://arxiv.org/pdf/2602.16311v1)

**作者:** Max Klimm `[一作]` (TU Berlin), Jannik Matuschke `[通讯]` (KU Leuven)

**通讯引用:** 287 | [OpenAlex ID](https://openalex.org/A5055549371)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并研究了在组合优化系统中通过最小控制集合（identifying set）实现系统状态可控性的新框架，给出了对凸集、离散集合、流网络、路径、基于图的系统、马尔可夫结构以及显式列举解集合的理论与算法。

**💡 创新点**

创新点在于将控制问题与识别问题等价化，并通过构造相应的稀疏矩阵/图形实现控制集合的最小化；提出了针对凸集的矩阵/图形Matroid（cographic、partition、polymatroid）理论，给出多种多项式或近似算法，同时阐明了非凸函数、一般离散集合和有向图路径控制的NP/Σ₂ⁿ难度。

**🔧 技术方法**

核心技术包括：子梯度与凸分析、线性/凸组合的几何特征、Matroid理论（基、环、连通分量、双图）、图论中的无向环判定、动态规划与树/森林结构、Set Cover贪心逼近、硬度归约与二阶可判定问题等。

**📊 数据集**

本研究为纯理论分析，未使用具体实验数据集，主要通过数理证明与多种图结构实例（如 DAG、一般有向图、网络流、基底集合）演示结果。

**📈 对比分析**

与现有的计价/收费控制方法对比，本文提供了更强的可控性保证（对任意成本函数都可实现目标状态），并给出最小控制集的最优/近似解法，理论上证明了在多数情形下的多项式时间或近似比例（如 √|E| 或 2ln|X|），同时也指出了不可逼近的下界。

**⚠️ 局限性**

局限性包括：1）在非凸或非离散场景下控制集合可能不存在；2）对一般有向图路径控制问题仅给出高阶可判定难度与弱近似结果，实际实现困难；3）多项式时间算法往往需要完整的可行域表示（如凸包或Matroid基），在规模极大时仍可能不可行；4）实验验证缺失，实际性能仍待评估。

---

## 212. Computing Equilibria in Games with Stochastic Action Sets

**arXiv ID:** 2602.16234 | [PDF](https://arxiv.org/pdf/2602.16234v1)

**作者:** Thomas Schwarz `[一作]` (National University of Singapore), Chun Kai Ling `[通讯]` (National University of Singapore)

**通讯引用:** 5874 | [OpenAlex ID](https://openalex.org/A5061456453)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究游戏中的随机可行动集合（GSAS），并提出对应的策略与均衡表示方法。

**💡 创新点**

提出可在 |A_i| 维度内紧凑表示 Nash 均衡的理论与算法，避免指数级表示。

**🔧 技术方法**

利用睡眠内部遗憾最小化（SI-MWU）与随机逼近，证明收敛到近似 NE，且收敛率为 O(√(log|A_i|/T))。

**📊 数据集**

在随机生成的 GSAS、100×100 随机偏置支撑游戏和棋盘匹配硬币游戏上进行实验；使用 Gurobi 的线性规划作为基准。

**📈 对比分析**

与基于序列形式的 LP 求解器对比，SI-MWU 在更大规模游戏上显著更快且保持低 SI‑遗憾；实验展示收敛到近似均衡且 SPR 低。

**⚠️ 局限性**

仍缺乏对 compact w 对应策略的 SPR 理论上界，算法局限于二人零和游戏；对一般博弈和更复杂结构的扩展尚未完成。

---

## 213. UCTECG-Net: Uncertainty-aware Convolution Transformer ECG Network for Arrhythmia Detection

**arXiv ID:** 2602.16216 | [PDF](https://arxiv.org/pdf/2602.16216v1)

**作者:** Hamzeh Asgharnezhad `[一作]` (Deakin University), U. Rajendra Acharya `[通讯]` (University of Southern Queensland)

**通讯引用:** 86697 | [OpenAlex ID](https://openalex.org/A5074179735)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并实现了一种混合卷积‑Transformer 架构 UCTECG‑Net，联合原始 ECG 与其频谱特征进行心律失常分类，并在模型中嵌入三种不确定性量化方法。

**💡 创新点**

创新点包括：①在同一网络中并行处理时间域与频率域信息；②将 Monte Carlo Dropout、Deep Ensembles 与 Ensemble MC Dropout 三种不确定性方法集成于统一框架；③通过不确定性混淆矩阵和相关指标评估模型的风险意识；④在 MIT‑BIH 与 PTB 数据集上实现了目前最高的准确率与不确定性准确率。

**🔧 技术方法**

使用的技术包括 1‑D 卷积层、Transformer 编码器、频谱生成、Monte Carlo Dropout、Deep Ensembles、Ensemble MC Dropout、预测熵、以及不确定性混淆矩阵等。

**📊 数据集**

实验数据集为 MIT‑BIH Arrhythmia（5 类）和 PTB Diagnostic（2 类）的心跳分段数据，所有样本均已重采样至 125 Hz 并裁剪为 187 步长。

**📈 对比分析**

通过与 LSTM、CNN1D、Transformer 进行 5 次随机初始化的重复实验进行对比，UCTECG‑Net 在 MIT‑BIH 上取得 98.58 % 的准确率、98.54 % 的 F1 分数，在 PTB 上取得 99.14 % 的准确率；不确定性准确率（UAcc）和相关敏感度、特异度均优于基线模型。

**⚠️ 局限性**

局限性包括：模型计算量大，主要针对短段心跳的离线分析；未对多导联或长时序 ECG 进行验证；缺乏临床真实场景下的实时部署与验证；以及未针对边缘设备进行轻量化改造。

---

## 214. Generative AI Usage of University Students: Navigating Between Education and Business

**arXiv ID:** 2602.16307 | [PDF](https://arxiv.org/pdf/2602.16307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 215. Geometric Neural Operators via Lie Group-Constrained Latent Dynamics

**arXiv ID:** 2602.16209 | [PDF](https://arxiv.org/pdf/2602.16209v1)

**作者:** Jiaquan Zhang `[一作]` (University of Electronic Science and Technology of China), Chaoning Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 2007 | [OpenAlex ID](https://openalex.org/A5057230698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于 Lie 群的低秩参数化的隐空间约束模块（MCL），用于改进神经算子在多层迭代和长时序推演中的不稳定性。

**💡 创新点**

创新点在于将隐状态更新视为 Lie 群作用而非欧氏向量加法，通过低秩 Lie 代数参数化实现等距、能量守恒的隐状态演化，从而显著抑制隐漂移。

**🔧 技术方法**

使用了低秩 Lie 代数参数化、指数映射近似、线性化步进以及插件式 MCL 层集成到现有 FNO、U‑NO、GINO 等神经算子。

**📊 数据集**

实验数据集包括 1D Burgers、线性输运、扩散‑吸附、2D 达西流、浅水方程和 Navier‑Stokes 等 PDE 的高精度数值模拟数据，涵盖多参数和多步推演。

**📈 对比分析**

通过与原始算子在多模型、多方程、多步长条件下的对比，MCL 使 MSE、Rel_L2、Rel_H1 等误差平均降低 30–50%，在长时序上保持更低的误差累积，参数增幅仅约 2.3%。

**⚠️ 局限性**

局限性包括低秩近似若秩过低可能削弱表达能力，对极端非线性或高维系统的泛化仍需验证；训练时需额外学习步长 α，且该模块主要针对欧氏隐空间，对所有算子架构的适用性仍有限。

---

## 216. Cryptographic Applications of Twisted Goppa Codes

**arXiv ID:** 2602.16207 | [PDF](https://arxiv.org/pdf/2602.16207v1)

**作者:** Harshdeep Singh `[一作]` (Indian Institute of Technology), Indivar Gupta `[通讯]` (Defence Research and Development Organisation)

**通讯引用:** 149 | [OpenAlex ID](https://openalex.org/A5005200150)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文定义并研究了多扭曲Goppa（MTG）码的结构、最小距离与解码算法，并将其应用于Niederreiter公钥加密。

**💡 创新点**

创新点在于把多扭曲Reed–Solomon码的子域子码作为MTG码的原型，提出可纠正 ⌊t/2⌋ 错误的扩展欧几里得解码方案，并证明其对部分密钥恢复攻击具有抵抗力，同时设计了可压缩的准循环结构。

**🔧 技术方法**

主要技术包括多扭曲多项式构造、子域子码与MDS性质判定、扩展欧几里得算法、错误定位与评估多项式以及准循环对称性分析。

**📊 数据集**

实验使用的示例数据集来自 𝔽₂⁵、𝔽₄²、𝔽₂¹⁶ 等有限域的具体字母点和多项式，覆盖多种参数组合。

**📈 对比分析**

与传统单扭曲Goppa码和经典Goppa码相比，MTG码在相同距离下可达到更高的码率，解码复杂度保持在 O(t²+tn)；在Niederreiter加密下，公钥压缩率提升至约 50% 左右。

**⚠️ 局限性**

主要限制是目前仅实现单扭曲位置的高效解码，且对多扭曲位置的解码复杂度尚未完全优化；同时准循环结构在极端参数下可能产生新的结构性攻击。

---

## 217. Training-Free Adaptation of Diffusion Models via Doob's $h$-Transform

**arXiv ID:** 2602.16198 | [PDF](https://arxiv.org/pdf/2602.16198v1)

**作者:** Qijie Zhu `[一作]` (Northwestern University), Minshuo Chen `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种训练无关、推理时可适配预训练扩散模型的方法 Doob-Oriented Inference-time Transformation (Doob‑OT)，通过 Doob 的 h‑transform 对采样过程动态校正，使模型朝高奖励方向生成样本，支持非可微奖励函数。

**💡 创新点**

创新点包括：①使用 Doob 的 h‑transform 以测度传输的视角重新构造目标分布；②提出 Monte Carlo 近似的动态校正项，避免梯度计算；③给出高概率总变差收敛保证；④实现训练免费且高效的推理时适配，并能与重采样/搜索方法互补。

**🔧 技术方法**

核心技术包括扩散模型、Doob h‑transform、蒙特卡洛估计、Tweedie 公式近似、时间阈值 l* 与校正强度 γ 的调节，算法在保持原预训练分数网络冻结的前提下完成采样。

**📊 数据集**

实验使用 D4RL 离线 RL 基准（HalfCheetah、Hopper、Walker2d 等）、LAION 美学评分图像生成、ImageReward 图像奖励，评估离线策略性能与图像审美得分。

**📈 对比分析**

与训练基础方法（IQL、Diffuser、D‑QL、QGPO）以及推理时方法（TFG、DAS、TTS、BoK、BFS）比较，Doob‑OT 在 9 个任务中 8 个达到最高推理时得分，平均 87.6，超越 TTS 并与训练方法相当；在图像审美实验中显著提升平均审美分数，并与完整轨迹模拟相比，使用 surrogate 近似既保持效果又显著降低运行时间。

**⚠️ 局限性**

局限性在于当高奖励区块在预训练分布中概率极低时，Monte Carlo 近似会产生高方差，导致效率下降；算法仍需调参 γ、τ、l* 等，并且在极稀奖励场景下需要更大样本量以保证收敛。

---

## 218. BAT: Better Audio Transformer Guided by Convex Gated Probing

**arXiv ID:** 2602.16305 | [PDF](https://arxiv.org/pdf/2602.16305v1)

**作者:** Houtan Ghaffari `[一作]` (Ghent University), Paul Devos `[通讯]` (Ghent University)

**通讯引用:** 54167 | [OpenAlex ID](https://openalex.org/A5053142611)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 Convex Gated Probing（CGP）评估方法并基于它改进了自监督音频模型，形成 Better Audio Transformer（BAT）

**💡 创新点**

创新点在于：①将多层特征通过可学习的 soft‑gating 聚合；②在自监督训练中加入注意力门控提升目标质量；③用 ViT 解码器代替 CNN 解码器以增强编码器表达；④构建统一、可复现的训练框架

**🔧 技术方法**

核心技术包括：ViT 变压器、Masked Latent Regression、EMA 目标网络、prototype‑based 线性分类、soft‑max 层权重聚合、门控多头自注意力、局部 min‑max 归一化

**📊 数据集**

使用的公开数据集为：AudioSet（AS‑2M 与 AS‑20k）、ESC‑50 以及 Speech Commands V2（SC‑2）

**📈 对比分析**

与传统 fine‑tuning、线性探测以及之前的 Protobin 探测进行对比；BAT 在冻结特征评估上明显优于 EAT 与 SSLAM，mAP 提升约 3–4pp，CGP 能将冻结性能逼近 fine‑tuning，甚至在某些任务上超越 fine‑tuning

**⚠️ 局限性**

局限性包括：仍未达到部分公开报告的最高分数；对不同音频域（如生物声学）适应性未完全验证；以及对模型规模和数据量的进一步扩展需要更多实验

---

## 219. "What I'm Interested in is Something that Violates the Law": Regulatory Practitioner Views on Automated Detection of Deceptive Design Patterns

**arXiv ID:** 2602.16302 | [PDF](https://arxiv.org/pdf/2602.16302v1)

**作者:** Arianna Rossi `[一作]` (Sant'Anna School of Advanced Studies), Simon Parkin `[通讯]` (TU Delft)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对欧盟监管从业者进行访谈，探讨其在检测和执法暗模式时使用的计算技术、需求与学术自动化工具的适用阶段；并评估现有学术工具在证据收集、合法性映射等方面的优劣。

**💡 创新点**

首次将监管实践与学术自动化工具需求进行系统对照，提出了针对执法需求的技术与流程匹配框架，并强调透明、可追溯与法律可采纳性。

**🔧 技术方法**

采用访谈与主题分析（Thematic Analysis）方法，对监管活动与工具功能进行编码；参考现有自动化检测技术（如AidUI、UIGuard、LLM‑based 系统）及其技术架构。

**📊 数据集**

使用九位欧洲监管从业者的访谈记录作为主要数据集；同时引用学术工具所用公开数据集（如网页界面抓取、cookie banner 数据、DP 目录）。

**📈 对比分析**

通过质性对比评估学术工具在“扫描（Sweep）”“筛选（Screening）”“法律违规确认”等执法阶段的适用性；发现大多数工具仅支持初步扫描，缺乏时间戳、可证据化与法律映射，导致在正式执法中使用有限。

**⚠️ 局限性**

局限性包括样本规模小、仅覆盖欧盟监管人员、未访谈学术工具开发者、未涉及商业工具；且对学术工具的评估多为定性，缺乏统一性能度量，导致结论不具普适性。

---

## 220. MICE: Minimal Interaction Cross-Encoders for efficient Re-ranking

**arXiv ID:** 2602.16299 | [PDF](https://arxiv.org/pdf/2602.16299v1)

**作者:** Mathias Vast `[一作]` (Sinequa by ChapsVision), Benjamin Piwowarski `[通讯]` (Sorbonne Université, CNRS, ISIR)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了 Minimal Interaction Cross-Encoder (MICE)，一种将跨编码器（cross‑encoder）与轻量级后交互模型相结合的高效重排架构。

**💡 创新点**

通过对跨编码器自注意力交互进行解释性掩码，剔除冗余交互并引入中层融合、只向量化的交叉注意力和层剔除，实现既保持或提升效果又大幅降低推理成本。

**🔧 技术方法**

主要技术包括 Transformer 自注意力掩码、mid‑fusion（前几层独立编码）、轻量化交叉注意力（仅从文档到查询）和顶部层剪枝；使用 MiniLM‑v2 与 Ettin（ModernBERT）两类轻量化背骨；训练采用 Margin‑MSE 蒸馏损失。

**📊 数据集**

在 MS‑MARCO（ID）以及 BEIR 13 个 OOD 数据集（如 ArguAna、HotpotQA 等）上进行评估，另外使用 TREC DL19/20 验证更细粒度效果。

**📈 对比分析**

与标准跨编码器、ColBERTv2、Sparse‑CE、PreTTR 等基线比较，MICE 在 ID 上几乎不失效（nDCG@10 仅下滑 ≤1.5 分），在 OOD 上提升 4–6 分；推理 latency 从 470 ms 降至 113 ms（预计算场景 26 ms），速度提升 4×，与 ColBERT 相当但效果更佳。

**⚠️ 局限性**

局限性：未实现完整索引与检索流水线；在交互层剔除查询自注意力导致性能崩溃，说明该组件仍必要；低维交互层压缩效果不佳，需进一步利用蒸馏或其他压缩方法。

---

## 221. MultiCW: A Large-Scale Balanced Benchmark Dataset for Training Robust Check-Worthiness Detection Models

**arXiv ID:** 2602.16298 | [PDF](https://arxiv.org/pdf/2602.16298v1)

**作者:** Martin Hyben `[一作]` (Kempelen Institute of Intelligent Technologies), Robert Moro `[通讯]` (Kempelen Institute of Intelligent Technologies)

**通讯引用:** 565 | [OpenAlex ID](https://openalex.org/A5032021797)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多语种多域、双写作风格的Check-Worthy Claim检测基准数据集MultiCW，并提供了基线模型；

**💡 创新点**

通过在16种语言、7个主题和两种写作风格上实现完整平衡，首次构建了大规模、跨语言、跨风格的CW检测数据集；

**🔧 技术方法**

利用多语言Transformer（XLM‑R、mDeBERTa、LESA）进行细调，并对15个大型语言模型（Claude、GPT‑4、Llama、Mistral、Qwen、Nemotron等）进行零样本推理；

**📊 数据集**

集成CLEF、MultiClaim、Ru22Fact等公开数据，补齐低资源语言的样本，使用机器翻译与维基百科抽句进行样本平衡；

**📈 对比分析**

在标准的Accuracy、macro Precision/Recall上对训练集划分的In‑Domain测试集和Out‑of‑Distribution（4种未见语言）进行评估；细调模型在In‑Domain上达到≈92%准确率，零样本LLM最高约79%；

**⚠️ 局限性**

仅覆盖16种语言，低资源语言样本为翻译产物；缺乏分级（非二元）检查性标注，未涵盖方言、代码混杂或多模态内容；

---

## 222. Flow on Social Media? Rarer Than You'd Think

**arXiv ID:** 2602.16279 | [PDF](https://arxiv.org/pdf/2602.16279v1)

**作者:** Michael T. Knierim `[一作]` (Karlsruhe Institute of Technology), Alexander Maedche `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 17228 | [OpenAlex ID](https://openalex.org/A5080792995)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在为期五天的实地研究中，研究者通过手机应用使用日志与日间重建法（Day Reconstruction Method）相结合，收集40名学生在日常生活中使用社交媒体的客观数据和自发流体验报告。

**💡 创新点**

创新点在于首次将体验与归因分离，使用客观轨迹日志与无诱导流体验报告相匹配，提供实证证据显示社交媒体极少诱发流体验且在日常中可能与流体验相竞争。

**🔧 技术方法**

使用技术包括移动应用追踪（MoviSensXS）、DRM问卷、累积逻辑回归混合模型、负二项混合模型以及重复测量相关分析。

**📊 数据集**

所用数据集为40名学生的五日连续数据，包含每分钟社交媒体使用时长、DRM中记录的流事件、疲劳与准备状态量表等。

**📈 对比分析**

通过比较个体内外效应的混合模型，发现每日更高的社交媒体使用时长与更少的流体验相关，且此关联在活跃使用与被动使用中呈现不同强度，整体表明社交媒体对流体验的负面或无明显正面影响。

**⚠️ 局限性**

局限性包括仅记录手机端使用，未覆盖平板、电脑或电视；DRM的回顾性设计可能低估短暂流体验；样本量相对较小；社交媒体活动被粗略划分为主动/被动；研究为相关性设计，无法确定因果关系。

---

## 223. RelianceScope: An Analytical Framework for Examining Students' Reliance on Generative AI Chatbots in Problem Solving

**arXiv ID:** 2602.16251 | [PDF](https://arxiv.org/pdf/2602.16251v1)

**作者:** Hyoungwook Jin `[一作]` (University of Michigan), Xu Wang `[通讯]` (University of Michigan)

**通讯引用:** 97752 | [OpenAlex ID](https://openalex.org/A5100424784)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了学生在使用生成式 AI 聊天机器人进行问题解决时的依赖模式，提出了基于帮助寻求和响应使用两大维度的依赖分析框架，并结合知识情境进行细粒度分析。

**💡 创新点**

创新点包括：①将帮助寻求与响应使用共同建模，形成 3×3 的九种依赖模式；②引入知识情境（重要性×掌握程度）双维度解释依赖行为；③设计非侵入式日志收集与自动化分析流程，使得在实际学习环境中可大规模实施。

**🔧 技术方法**

技术手段包括：
- 记录学生与机器人对话、代码编辑与复制日志；
- 人工标注依赖模式与知识情境；
- 统计分析（Somers' D、MANOVA、OLS、LSA）;
- 采用大型语言模型（GPT‑4o‑mini、Gemini‑flash、Gemini‑pro、Qwen3）进行自动依赖模式分类，并与人工标注对比。

**📊 数据集**

使用的数据集来自 79 名大学生的 Web 编程课程，包含 1,362 条聊天记录、2,708 条代码编辑日志、427 个交互段。数据已公开（https://osf.io/27ec5/overview?view_only=a8234a17f908464297d35d5ca1ef476c）。

**📈 对比分析**

与传统单一维度的帮助寻求或响应使用分析相比，本文框架能够同时捕捉两者的交互。自动分类实验显示，LLM 在检测被动模式的 F1 分别为 0.807（帮助寻求）和 0.760（响应使用），与人工标注的一致性相近；使用 3–9 例 few‑shot 可进一步提升性能。对学习成绩的回归分析未发现显著关联，提示需更大样本和多测量验证。

**⚠️ 局限性**

局限性：
- 研究仅在自定义聊天机器人与特定课程环境下进行，可能不具备对商业机器人（如 ChatGPT）的普适性；
- 仅采用一次前后测，缺乏纵向多时点数据，难以判断因果关系；
- 数据量相对有限，缺乏跨文化、跨学科的验证，导致结果可能受样本偏差影响。

---

## 224. Online Prediction of Stochastic Sequences with High Probability Regret Bounds

**arXiv ID:** 2602.16236 | [PDF](https://arxiv.org/pdf/2602.16236v1)

**作者:** Matthias Frey `[一作]` (University of Melbourne), Jingge Zhu `[通讯]` (University of Melbourne)

**通讯引用:** 509 | [OpenAlex ID](https://openalex.org/A5007978311)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文重新审视了已知时间范围内随机序列的通用预测问题，探讨了是否可以推导出高概率的消失悔恨界限，以补充现有的期望界限。

**💡 创新点**

提出了高概率的悔恨界限，具有与先前期望界限相似的形式，并证明了在不做额外假设的情况下，无法显著改善该界限的指数。

**🔧 技术方法**

使用了概率论和信息论中的技术，特别是通过超马尔可夫过程和阿祖玛-霍夫丁不等式来推导结果。

**📊 数据集**

未具体提及使用的数据集，但讨论了在实际应用中的潜在场景，如航空交通控制、自动驾驶和医疗保健中的预测。

**📈 对比分析**

与现有的期望悔恨界限进行比较，提出的高概率界限在收敛速率上与期望界限相匹配，且在高概率下表现出更好的可靠性。

**⚠️ 局限性**

局限性在于高概率界限的误差概率仅以√(log(1/))的形式出现，而在某些情况下，可能需要更好的界限来提高实用性。

---

## 225. Bayesian Quadrature: Gaussian Processes for Integration

**arXiv ID:** 2602.16218 | [PDF](https://arxiv.org/pdf/2602.16218v1)

**作者:** Maren Mahsereci `[一作]` (Yahoo Research), Toni Karvonen `[通讯]` (Lappeenranta-Lahti University of Technology LUT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统综述了贝叶斯四边形（Bayesian quadrature）的理论基础、方法分类、理论保证，并通过控制实验探讨了不同建模、推断和采样策略对性能的影响。

**💡 创新点**

创新点在于首次提出三轴（建模、推断、采样）的系统化分类法，提供了完整的理论保证汇总，并通过实证实验展示各选择组合的效果，为后续研究和实际应用提供了明确的参考框架。

**🔧 技术方法**

主要采用贝叶斯数值方法中的高斯过程（Gaussian Process）建模、马尔科夫链蒙特卡罗（MCMC）及变分推断、以及基于拉格朗日插值/采样点策略的采样方法；同时与传统蒙特卡罗、Gauss‑Legendre 等经典数值积分方法进行了对比。

**📊 数据集**

实验数据主要为人工生成的积分基准函数（如一维、二维、多维的Rosenbrock、Gaussian、Chebyshev 等），并通过不同维度和复杂度的测试函数验证方法的鲁棒性。

**📈 对比分析**

与传统蒙特卡罗、Quasi‑Monte Carlo、Gaussian Quadrature 等方法比较后发现，贝叶斯四边形在样本数较少时能显著提升积分精度，尤其在高维稀疏采样情形下优势明显；但随着维度增长和采样量提升，计算成本与误差收敛速度逐渐趋于传统方法。

**⚠️ 局限性**

主要局限包括：高维情况下计算复杂度显著增加、核函数选择与参数估计对性能影响大、缺乏通用的自动化参数调优策略，以及在非光滑或高噪声函数上的收敛性理论仍待完善。

---

## 226. Condorcet Dimension and Pareto Optimality for Matchings and Beyond

**arXiv ID:** 2602.16289 | [PDF](https://arxiv.org/pdf/2602.16289v1)

**作者:** Telikepalli Kavitha `[一作]` (Tata Institute of Fundamental Research), Ulrike Schmidt-Kraepelin `[通讯]` (TU Eindhoven)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究在一侧偏好下由匹配（或枝干、图结构）诱导的选举中Condorcet维度的上界与下界，探讨弱Condorcet胜者与Pareto最优集之间的关系；

**💡 创新点**

创新点在于将Condorcet维度扩展到弱、半序甚至任意偏好，揭示其与Matroid约束、枝干结构的紧密联系，并给出近似最优的构造算法与最优下界；

**🔧 技术方法**

主要技术包括：对Pareto最优集构造的分枝树与交换图分析、Matroid基础交换定理、图论的可分枝性与循环消除、以及NP难度的归约证明；

**📊 数据集**

该工作为纯理论研究，未使用具体实验数据集，而是通过构造性证明与复杂度归约给出理论结果；

**📈 对比分析**

由于研究侧重理论边界，未与实验方法比较，但通过证明多项式算法可在严格/弱偏好下获得大小为2的Condorcet集合，在半序偏好下实现O(√n)的上界，展示了该方法的最优或近似最优；

**⚠️ 局限性**

局限性在于：在半序或Matroid约束下Condorcet维度可高达Θ(√n)或n；部分偏好下Pareto最优集可能不存在；且判定Condorcet维度或Pareto最优匹配是NP‑完备，限制了算法的可扩展性。

---

## 227. Breaking the Sub-Millimeter Barrier: Eyeframe Acquisition from Color Images

**arXiv ID:** 2602.16281 | [PDF](https://arxiv.org/pdf/2602.16281v1)

**作者:** Manel Guzmán `[一作]` (Horizons Optical), Antonio Agudo `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**通讯引用:** 34386 | [OpenAlex ID](https://openalex.org/A5024769212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一套基于多视角RGB+深度的计算机视觉框架，用于测量眼镜框架，替代传统机械测量仪。

**💡 创新点**

创新点在于将SAM2风格分割、Depth Anything深度估计与多视角特征融合相结合，利用InVision四摄系统实现无投影、无机械装置的亚毫米精度测量，并提供了针对该任务的新分割与轨迹数据集。

**🔧 技术方法**

使用了4摄InVision系统采集；SAM2 Transformer‑based分割模型；Depth Anything ViT模型的相对深度估计；EfficientNetV2 Backbone的多视角特征提取；多视角融合策略（晚期融合+max‑pooling）以及数据增强与归一化技术。

**📊 数据集**

使用了1,002张带分割mask的图像集（来自全球20台InVision设备，四视角每张图）以及5,000张包含4视角RGB+深度与真实轨迹的样本（500个测量），并用CVAT标注。

**📈 对比分析**

与DeepLabV3+分割模型相比，IoU从0.931提升至0.958；轨迹测量在S模型+灰度+深度模式下平均误差0.4238 mm，最大误差1.8466 mm，88%测量小于1 mm，优于传统相位测量约2 mm的误差。

**⚠️ 局限性**

局限性包括：缺乏真实深度ground‑truth，深度估计受背景噪声影响；模型对遮挡区域（尤其是上角）精度下降；仅验证了全框眼镜，未覆盖半框/无框；缺乏与现有机械测量仪的直接对比。

---

## 228. Quantum Oracle Distribution Switching and its Applications to Fully Anonymous Ring Signatures

**arXiv ID:** 2602.16268 | [PDF](https://arxiv.org/pdf/2602.16268v1)

**作者:** Marvin Beckmann `[一作]` (Technical University of Denmark), Christian Majenz `[通讯]` (Technical University of Denmark)

**通讯引用:** 785 | [OpenAlex ID](https://openalex.org/A5084799117)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提供了四条量子随机预言机模型（QROM）下的安全归约，证明了两类通用环签名构造（AOS框架和基于环陷门的构造）的安全性。

**💡 创新点**

创新点在于引入环陷门预图像可采样函数（RPSF）框架，统一并推广了以往仅在经典ROM证明的环签名安全性，同时提出了统计距离与Renyi散度相结合的Oracle重编程技术。

**🔧 技术方法**

采用测量与重编程、压缩预言机、Renyi散度分析、历史无关归约等技术，对QROM下的签名、匿名性和不可伪造性进行证明。

**📊 数据集**

本文为理论分析，无需实际数据集；所有证明基于抽象概率分布与量子查询模型。

**📈 对比分析**

通过对比经典ROM证明，QROM归约的安全阈值虽然在环成员数上呈指数增长，但对小环（如Signal协议所需的大小为2）已提供可用的安全保证；与现有的仅在ROM证明的构造相比，本文的归约实现了可量子化的安全保证。

**⚠️ 局限性**

局限性在于归约对环大小的指数依赖、RPSF构造在大环上的困难，以及在某些构造中需要较长盐长度以满足可编程性，导致实际安全强度下降。

---

## 229. AFFMAE: Scalable and Efficient Vision Pretraining for Desktop Graphics Cards

**arXiv ID:** 2602.16249 | [PDF](https://arxiv.org/pdf/2602.16249v1)

**作者:** David Smerkous `[一作]` (Oregon State University), Behzad Najafian `[通讯]` (University of Washington)

**通讯引用:** 5780 | [OpenAlex ID](https://openalex.org/A5045615753)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AFFMAE，一种面向桌面级 GPU 的高分辨率视觉预训练框架，通过自适应离网 token 合并实现 Masked Autoencoders 的高效训练。

**💡 创新点**

创新点在于结合自适应离网动态下采样、对掩码无感知的编码器、数值稳定的混合精度 Flash‑style cluster attention、深度监督以及 Perlin 噪声掩码，显著降低 FLOPs 与显存，同时保持与 ViT‑MAE 相当的分割性能。

**🔧 技术方法**

使用技术包括自适应 Off‑grid token merging、Triton Flash‑style cluster attention、深度监督（Deep Supervision）、Perlin Noise 掩码、混合精度训练及自动微分等。

**📊 数据集**

在 187,270 张无标签电子显微镜（EM）图像上进行自监督预训练，在 FPW 数据集（570/164 图像）上进行监督细调，用于高分辨率 EM 分割。

**📈 对比分析**

与 ViT‑MAE 基线在相同参数量下对比，AFFMAE 在 512×512 分辨率下 mIoU 仅差 0.4%，但 FLOPs 降 4.5×、显存降 46%；在 768×768 时 FLOPs 降 7×、显存降 2×，训练吞吐率提升 36%。

**⚠️ 局限性**

局限性包括对极端掩码比例（>65%）和极细结构在高下采样时可能的信息损失；在 3D/4D 数据上的验证尚待扩展。

---

## 230. Submodular Maximization under Supermodular Constraint: Greedy Guarantees

**arXiv ID:** 2602.16240 | [PDF](https://arxiv.org/pdf/2602.16240v1)

**作者:** Ajitesh Srivastava `[一作]` (University of Southern California), Shanghua Teng `[通讯]` (University of Southern California)

**通讯引用:** 11747 | [OpenAlex ID](https://openalex.org/A5102005063)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在超级模数约束下最大化单调子模函数的问题，提出基于比率-边际的贪心算法并给出近似分析；

**💡 创新点**

提出一种更宽松的超级模数曲率定义，扩展可处理的成本函数范围，并证明贪心算法在曲率 γ 受限时可取得 (1−e^(−(1−γ))) 的近似，并在目标函数也有曲率 c 时进一步改进；

**🔧 技术方法**

使用贪心比率-边际选择、曲率理论、递推不等式、二分搜索和实验模拟；

**📊 数据集**

在模拟的多轮 LLM 辩论环境中生成随机代理和问题集，基于代理准确率和代价构造子模目标与超级模约束；

**📈 对比分析**

与多种贪心启发式（Ratio-fg、Greedy-f、Greedy-g、Random）以及穷举最优做对比，实验显示 Ratio‑Marginal 近乎达到最优，且在全局视图模型下明显优于其它方法；

**⚠️ 局限性**

仅在超级模数成本曲率被限定时才可获得保证；问题在一般情况下不可多项式逼近，且实验仅在人工生成的模拟数据上验证，实际大规模应用仍需进一步研究。

---

## 231. Collaborative Safe Bayesian Optimization

**arXiv ID:** 2602.16235 | [PDF](https://arxiv.org/pdf/2602.16235v1)

**作者:** Alina Castell Blasco `[一作]` (Ericsson Research), Maxime Bouton `[通讯]` (Ericsson Research)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种协作安全贝叶斯优化（CoSBO）框架，用于在移动网络中在线安全地调优网络参数。

**💡 创新点**

创新点在于：① 引入协作初始化策略，利用与主任务相似的协作者数据提升样本效率；② 将协作信息编码为上下文变量，使低质量协作者不会损害优化过程；③ 在多约束安全优化的基础上首次在通信场景中应用。

**🔧 技术方法**

核心技术包括高斯过程（Gaussian Process）建模、SafeOpt-MC安全贝叶斯优化、Pearson相关系数用于协作者选择、上下文感知的GP更新与安全集合扩展。

**📊 数据集**

使用仿真数据集：5个城市拓扑图、3种流量负载，共15个网络场景，每个场景在水平波束宽度和电倾角两个参数上进行优化，数据来自3GPP NR城市宏观级仿真。

**📈 对比分析**

与基线SafeOpt-MC和无安全约束的随机探索方法对比；CoSBO在高质量协作者下平均比SafeOpt-MC提前约7次迭代达到最优（在仿真中约相当于7天），即样本效率显著提升；在低质量协作者时性能与SafeOpt-MC相当，且不出现退化。

**⚠️ 局限性**

局限性包括：① 对协作者相关性假设敏感，若协作者不具备足够相似性则无法获益；② 仅在仿真环境验证，真实网络中环境噪声和动态变化的影响尚未完全评估；③ 目前仅处理单一安全约束，扩展到多约束仍需研究。

---

## 232. DataCube: A Video Retrieval Platform via Natural Language Semantic Profiling

**arXiv ID:** 2602.16231 | [PDF](https://arxiv.org/pdf/2602.16231v1)

**作者:** Yiming Ju `[一作]` (Beijing Academy of Artificial Intelligence), Tengfei Pan `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 4599 | [OpenAlex ID](https://openalex.org/A5100612025)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了DataCube平台，实现了从大规模视频库自动分割、质量控制、语义化描述、嵌入索引，到基于自然语言的混合检索与深度检索，最终支持快速生成定制化视频子集；

**💡 创新点**

创新点在于将多维自然语言语义配置与嵌入过滤、神经重排序、深度语义匹配三种检索模式融合，既保持检索效率，又提升细粒度语义匹配精度，且无需重新处理整个库即可构建数据集；

**🔧 技术方法**

技术栈包括PySceneDetect场景检测、PaddleOCR文本识别、RAFT光流评估、NIQE/MUSIQ美学评分、Qwen2.5‑VL‑7B生成语义配置、BGE嵌入编码、Milvus索引、Ray调度、vLLM推理、Qwen3‑Reranker‑0.6B重排序、Qwen2.5‑VL‑72B深度检索；

**📊 数据集**

使用公开的大规模视频数据集如HowTo100M、Panda‑70M、Koala‑36M等作为基准，并支持用户上传私有视频构建自定义库；

**📈 对比分析**

通过混合检索+深度检索相结合，在保留10k候选范围的高效检索基础上，深度检索可在3‑5分钟内完成对500候选的精确匹配，显著提升对复杂查询（如排除条件）的准确性，优于传统单一CLIP嵌入检索的召回率与精度；

**⚠️ 局限性**

局限性包括深度检索的计算成本和耗时、对低质量视频的过度过滤、对多语言查询细粒度支持不足，以及需要依赖大型视觉语言模型的资源与推理成本。

---

## 233. Factored Latent Action World Models

**arXiv ID:** 2602.16229 | [PDF](https://arxiv.org/pdf/2602.16229v1)

**作者:** Zizhao Wang `[一作]` (University of Texas at Austin), Peter Stone `[通讯]` (Sony AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种因式化潜在动作模型（FLAM），能够在无动作标签的视频中学习多实体环境的可控制世界模型，并通过因式化状态与潜在动作实现更精准的动态建模与视频生成。

**💡 创新点**

创新点在于将整个场景拆分为多个独立因子，每个因子拥有自己的潜在动作，并共享统一的潜在动作空间；这一设计显著降低了多实体动作组合的维度，并在多实体环境下实现更好的因果分离与可控性。

**🔧 技术方法**

技术方法包括：VQ‑VAE预训练编码器提取离散特征；Slot Attention 作为因式化器生成可持续的因子槽；共享逆向与正向动力学模型（采用 spatio‑temporal self‑attention 与跨时间注意力）预测潜在动作与下一步因子；VAE‑style KL 正则化控制潜在动作容量；聚合器通过跨时间注意力将预测因子映射回特征空间并解码。

**📊 数据集**

使用的数据集包括四个仿真环境（MultiGrid、Bigfish、Leaper、Starpilot）以及真实驾驶视频集 nuPlan，所有数据均包含多实体独立动作场景。

**📈 对比分析**

与 GenIe、AdaWorld、World Model、PlaySlot、SlotFormer 等基线进行比较，实验结果显示 FLAM 在 PSNR、SSIM、LPIPS、FVD 等视频预测指标上均优于基线，并在可控视频生成与基于伪标签的策略学习上取得更高的性能，尤其在多实体、复杂视觉环境中表现显著提升。

**⚠️ 局限性**

局限性包括：需为每个数据集单独训练 VQ‑VAE，缺乏跨数据集共享的通用编码器；聚合器仅使用 Transformer 解码，生成视觉质量受限；未来工作可尝试预训练 tokenizer、统一编码器以及更具表现力的解码器（如扩散或流匹配模型）。

---

## 234. The Weight of a Bit: EMFI Sensitivity Analysis of Embedded Deep Learning Models

**arXiv ID:** 2602.16309 | [PDF](https://arxiv.org/pdf/2602.16309v1)

**作者:** Jakub Breier `[一作]` (TTControl GmbH), Xiaolu Hou `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对嵌入式神经网络在不同权重量化格式下对电磁故障注入（EMFI）的鲁棒性进行了系统实验评估。

**💡 创新点**

首次在同一硬件平台、同一模型集上横向比较 FP32、FP16、INT8、INT4 四种权重量化格式对 EMFI 故障的敏感性，并证明整数量化在保持精度的同时提供更高的容错性。

**🔧 技术方法**

使用 NewAE ChipSHOUTER EMFI 平台进行电磁脉冲注入，进行表面扫描定位脉冲位置；对权重缓冲区进行位级损坏分析；在嵌入式内存芯片上部署 ResNet-18/34/50 与 VGG-11 模型，采用 ONNX Runtime 进行推理。

**📊 数据集**

使用 ImageNet‑1K 验证集（子样本 4096 张图像）进行模型准确率评估。

**📈 对比分析**

通过对比注入前后的 Top‑1 / Top‑5 准确率来衡量不同数值格式的抗攻击性能；结果显示浮点格式在一次注入后几乎完全失效，INT8 在大模型中 Top‑1 仍维持约 70%（Top‑5 约 90%），而 INT4 效果略逊。

**⚠️ 局限性**

局限性包括：仅使用 4 MB 未加纠错的 SRAM，未考察激活层或其他硬件误差；实验仅涉及单次 EMFI 注入，未涵盖持续或多点注入；平台规模有限，可能无法直接推广至更大规模的 AI 加速器。

---

## 235. A Calculus of Overlays

**arXiv ID:** 2602.16291 | [PDF](https://arxiv.org/pdf/2602.16291v1)

**作者:** Bo Yang `[一作]` `[通讯]` (Figure AI Inc), Bo Yang (Figure AI Inc)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了 Overlay-Calculus，一种基于叠加继承（记录、定义、继承）的最小计算模型，并给出了无约简的观测语义，演示了其对 λ 演算的完整嵌入以及对配置语言、表达式问题的应用。

**💡 创新点**

创新点在于：1）引入可交换、幂等的继承语义，消除了多重继承的线性化问题；2）通过仅使用集合论和 Knaster–Tarski 固定点给出 λ 演算的完整抽象语义；3）展示了无函数、不可变的 Turing 完整模型，解决表达式问题并实现 CPS 无关性与随机存取内存。

**🔧 技术方法**

使用了集合论、幂集格、Knaster–Tarski 固定点、F-coalgebra、A‑normal 形式翻译、观察性语义等理论工具，并实现了 Overlay 语言及其测试套件，借鉴 NixOS 模块系统等配置语言。

**📊 数据集**

实验中主要使用了 NixOS 包集合（超过 10 万包）的延迟可观察结构作为示例；表达式问题示例也在实现中给出。

**📈 对比分析**

与 Böhm 树的对应关系通过证明充分抽象性得到验证，未做基准性能评测；实现层面在 Overlay 语言中通过完整测试覆盖所有示例，主要关注语义正确性而非运行速度。

**⚠️ 局限性**

局限性包括：未提供类型系统与总性检查，缺乏对标量值的原生支持，无法直接处理递归函数或多值求值；实现仍处于实验阶段，缺乏大规模应用评估与性能分析。

---

## 236. Regret and Sample Complexity of Online Q-Learning via Concentration of Stochastic Approximation with Time-Inhomogeneous Markov Chains

**arXiv ID:** 2602.16274 | [PDF](https://arxiv.org/pdf/2602.16274v1)

**作者:** Rahul Singh `[一作]`, Nicholas Bambos `[通讯]` (Stanford University)

**通讯引用:** 5298 | [OpenAlex ID](https://openalex.org/A5002056995)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了在线无模型 Q‑学习的高概率收敛与折扣奖励分析，兼容 Boltzmann 软最大与 Smoothed ε_n‑Greedy 探索，且不需要乐观、置信上界或模型改动；

**💡 创新点**

①首个无乐观的在线 Q‑学习收敛与奖励保证；②针对时间异质马尔可夫噪声的随机逼近提供了新的高概率收敛定理；③给出了子线性 O(N^{9/10}) 的奖励上界；

**🔧 技术方法**

利用马尔可夫随机逼近中的 Poisson 方程与噪声平均技术，结合高概率收敛分析；对温度调度与 ε‑贪婪混合策略进行渐进收敛和样本复杂度证明；

**📊 数据集**

论文为理论分析，未使用任何公开数据集；

**📈 对比分析**

与已有基于乐观或模型方法（如 UCBVI‑γ、Double Q‑learning 等）相比，虽然奖励上界更高，但证明了即使在未加任何改动的经典 Q‑学习中仍可实现子线性奖励；样本复杂度与离线 Q‑学习的最优结果一致；

**⚠️ 局限性**

1) 奖励上界 O(N^{9/10}) 仍高于最优的 O(√N)；2) 只适用于折扣回报，未覆盖平均奖励情形；3) Boltzmann 探索在小 suboptimality gap 时退化为线性；4) 需要对探索参数进行精细调度，实验验证缺失。

---

## 237. Prediction of Major Solar Flares Using Interpretable Class-dependent Reward Framework with Active Region Magnetograms and Domain Knowledge

**arXiv ID:** 2602.16264 | [PDF](https://arxiv.org/pdf/2602.16264v1)

**作者:** Zixian Wu `[一作]` (Jiangsu University of Science and Technology), Honglei Jin `[通讯]` (Jiangsu University of Science and Technology)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5103410296)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了多种基于知识型特征和磁图的日常太阳耀斑预测数据集，首次提出基于类依赖奖励（CDR）的监督学习框架来预测≥M级耀斑，比较了CNN、CNN‑BiLSTM、Transformer及其CDR对应模型的分类与概率预测性能，并与NASA/CCMC模型对比；

**💡 创新点**

创新点在于：①将强化学习奖励机制引入监督学习，形成CDR框架解决类不平衡问题；②结合LOS与向量磁场参数，验证Transformer在多特征组合下优于图像模型；③对奖励工程进行敏感性分析并证明模型鲁棒；④使用SHAP解释Transformer与CDR-Transformer的特征重要性差异；

**🔧 技术方法**

技术手段包括卷积神经网络、双向LSTM、Transformer架构、类依赖奖励学习、经验回放、加权交叉熵、Brier Skill Score、True Skill Statistic、SHAP解释方法；

**📊 数据集**

数据集来自SDO/HMI SHARP，包含39个磁场参数（31 LOS + 8 向量）及SHARP LOS磁图，覆盖2010‑2020年808个AR，共10个交叉验证拆分；

**📈 对比分析**

比较方法为10折交叉验证后计算TSS和BSS，并对概率阈值扫描；结果显示：CDR-Transformer‑10在分类TSS最高（0.829），概率BSS最高（0.489），且在对比测试集上优于NASA/CCMC（TSS>0.95）和iTransformer（TSS>0.93）；

**⚠️ 局限性**

局限性在于：1）奖励值设定仍需经验调优；2）模型仅验证了≥M级耀斑，其他级别和CMEs待扩展；3）受限于SHARP单AR数据，无法处理多AR影响；4）Transformer对长序列的时间复杂度高，需进一步优化。

---

## 238. Amortized Predictability-aware Training Framework for Time Series Forecasting and Classification

**arXiv ID:** 2602.16224 | [PDF](https://arxiv.org/pdf/2602.16224v1)

**作者:** Xu Zhang `[一作]` (Fudan University), Wei Wang `[通讯]` (Fudan University)

**通讯引用:** 40165 | [OpenAlex ID](https://openalex.org/A5100391662)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种通用的 Amortized Predictability-aware Training Framework (APTF)，用于改进时间序列预测（TSF）和时间序列分类（TSC）的训练过程。

**💡 创新点**

创新点在于：① 通过层次化的 Predictability-aware Loss (HPL) 对低可预测样本动态识别并逐步加大惩罚；② 引入 amortization 模型缓解模型偏差导致的可预测性估计误差，从而进一步提升 HPL 的效果。

**🔧 技术方法**

主要技术包括：深度学习模型（CNN、Transformer、MLP 等），基于损失的样本分桶与权重分配，层次化桶策略，协同训练（amortization）以及常规的交叉熵/均方误差等损失函数。

**📊 数据集**

使用 11 个 TSF 数据集（包含 3 组基金销售数据、Exchange、Weather、Electricity、Traffic、4 个 ETT）以及 128 个 UCR 归档的单变量 TSC 数据集进行实验。

**📈 对比分析**

与 11 种主流 TSF 模型（如 Autoformer、Scaleformer、Informer 等）以及 5 种 TSC 基线（如 InceptionTime、ResNet 等）比较，APTF 在短期/长期预测任务中平均提升 2%–15% 的精度（WMAPE、MSE/MAE），在 TSC 任务中平均提升 0.96% 的准确率；相较于 WaveBound 与 Co‑Teaching，APTF 在噪声较大的基金销售数据上表现更优。

**⚠️ 局限性**

局限性包括：① 采用 amortization 模型时显著增加 1.5–2 倍的显存与训练时间；② 对桶数、阶段间隔等超参数敏感，需要经验或网格搜索；③ 主要针对低可预测样本的软惩罚，未考虑极端标签噪声场景（如 Co‑Teaching 的硬丢弃）。

---

## 239. SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and Long-term Time Series Forecasting

**arXiv ID:** 2602.16220 | [PDF](https://arxiv.org/pdf/2602.16220v1)

**作者:** Xu Zhang `[一作]` (Fudan University), Wei Wang `[通讯]` (Fudan University)

**通讯引用:** 40165 | [OpenAlex ID](https://openalex.org/A5100391662)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了SEMixer模型，用于长周期时间序列预测。

**💡 创新点**

创新点包括：随机注意力机制（RAM）和多尺度渐进混合链（MPMC），能有效提升时间片语义并缓解多尺度间语义间隙。

**🔧 技术方法**

技术实现基于轻量化MLP‑Mixer，结合实例归一化、位置嵌入、随机交互矩阵与dropout集成，以及逐级拼接的多尺度混合。

**📊 数据集**

使用了十个公开长周期时间序列数据集（ETTh1/2/TTm1/2、Weather、Electricity、Solar、ILI、Exchange、Traffic）以及21GB无线网络 KPI 数据（2025 CCF AlOps Challenge）。

**📈 对比分析**

与12类先进基线（Transformer、CNN、线性等）在多步预测（96、192、336、720步）进行对比，SEMixer在MSE/MAE上均优于所有基线，并在挑战赛中获得第三名。

**⚠️ 局限性**

局限性：对极长序列仍受GPU内存限制，随机注意力参数需手动设定，模型在高噪声环境下仍会出现一定误差。

---

## 240. Multi-Class Boundary Extraction from Implicit Representations

**arXiv ID:** 2602.16217 | [PDF](https://arxiv.org/pdf/2602.16217v1)

**作者:** Jash Vira `[一作]`, Simon Ratcliffe `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个二维多类别隐式表示的边界提取算法，能够保证拓扑一致且无孔洞。

**💡 创新点**

创新点包括将1D拓扑括号根查与梯度定位相结合的根检算法，以及基于边缘根点的自适应二维多边形化方法，支持三域交点估计。

**🔧 技术方法**

使用了1D根查、梯度投影、BFS、三域交点三线性插值、Nelder–Mead优化、几何阈值细化、堆栈式自适应细分等技术。

**📊 数据集**

采用真实地质建模数据（k=14），通过神经网络生成的多类别隐式表示进行测试，切片分辨率为1000×1000。

**📈 对比分析**

与无几何阈值的基准相比，加入阈值后细节更精细且边界更光滑，计算时间略增，示例显示无孔洞且无自相交。

**⚠️ 局限性**

局限性在于目前仅实现二维，三维推广仍待研究；对高噪声或密集交点的鲁棒性尚未完全验证。

---

## 241. Nonplanar Model Predictive Control for Autonomous Vehicles with Recursive Sparse Gaussian Process Dynamics

**arXiv ID:** 2602.16206 | [PDF](https://arxiv.org/pdf/2602.16206v1)

**作者:** Ahmad Amine `[一作]` (University of Pennsylvania), Rahul Mangharam `[通讯]` (University of Pennsylvania)

**通讯引用:** 4568 | [OpenAlex ID](https://openalex.org/A5009445756)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对非平面地形的实时自适应车辆控制框架，使用递归稀疏高斯过程学习残差动力学，并将其嵌入模型预测路径积分（MPPI）控制器，实现高精度路径跟踪。

**💡 创新点**

创新点在于将几何感知的单轨模型与递归稀疏GP残差学习相结合，既能在线适应变化的地形，又能在MPC框架中实时计算；同时利用GPU并行加速MPPI采样与GP预测。

**🔧 技术方法**

采用单轨双轮动力学模型、坡度/法向量提取、递归稀疏高斯过程（在线更新）、MPPI（采样式MPC）、NVIDIA Isaac Sim仿真环境和GPU并行计算。

**📊 数据集**

使用自建的3D点云地形高度图（含坡度、法向量）以及Isaac Sim中的三条非平面轨道（Kidney、L形、Oval）进行验证，并未使用公开数据集。

**📈 对比分析**

与仅使用单轨模型的MPPI基线进行对比，评估指标包括轨迹交叉误差、绝对交叉误差直方图和控制频率；实验结果显示递归GP残差模型显著降低交叉误差，控制频率保持在≥50Hz，尤其在高曲率坡道上保持稳定。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，缺乏真实车辆实验；GP在线更新仍需消耗一定计算资源，虽低于20ms但高于纯单轨模型；依赖离线训练获取超参数，且在极端复杂地形或存在动态障碍物时的鲁棒性尚待进一步验证。

---

## 242. Linked Data Classification using Neurochaos Learning

**arXiv ID:** 2602.16204 | [PDF](https://arxiv.org/pdf/2602.16204v1)

**作者:** Pooja Honna `[一作]` (National Institute of Advanced Studies), Nanjangud C. Narendra `[通讯]` (National Institute of Advanced Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文将神经混沌学习（Neurochaos Learning）扩展到知识图谱数据，通过节点特征聚合后输入ChaosNet进行节点分类。

**💡 创新点**

创新点在于：①首次将NL应用于图数据；②提出基于平均聚合的轻量级节点聚合方案；③在无深度图神经网络的情况下，用混沌特征提升分类性能。

**🔧 技术方法**

使用技术包括：ChaosNet（混沌神经网络）、ChaosFEX（混沌特征提取）、平均节点聚合、宏F1评分、5折交叉验证、Python+NumPy+Scikit-learn+PyTorch Geometric。

**📊 数据集**

实验数据集为七个公开节点分类数据集：Cora、Citeseer、Pubmed（同类高同质性）；Actor、Cornell、Wisconsin、Squirrel（异质性强）。

**📈 对比分析**

比较方法：将不同聚合策略（原始、聚合、双重）与NL的宏F1进行对比。结果显示：在同质性图上，双重聚合可将宏F1提升至0.84；在异质性图上，性能下降（最低约0.26）。

**⚠️ 局限性**

局限性：①对高度异质性或稀疏特征图的处理效果差；②缺乏自适应聚合机制；③未将混沌映射分布在图结构中，导致对图结构信息利用不足。

---

## 243. Label-Consistent Data Generation for Aspect-Based Sentiment Analysis Using LLM Agents

**arXiv ID:** 2602.16379 | [PDF](https://arxiv.org/pdf/2602.16379v1)

**作者:** Mohammad H. A. Monfared `[一作]`, Akbar Karimi `[通讯]` (Lamarr Institute for Machine Learning and Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种基于代理（agentic）流程的数据增广方法，用于Aspect-Based Sentiment Analysis（ABSA），通过多步生成与验证生成高质量合成训练样本。

**💡 创新点**

引入ReAct式多代理框架，将生成与标签验证分离，实现结构化、可控的合成数据，显著提升标签一致性和任务性能；与传统单步提示方法进行严格对比。

**🔧 技术方法**

使用 Qwen2.5-14B 语言模型、工具调用、ReAct 风格代理、生成–评估两阶段流程、自动化验证以及与 Prompting baseline 的对比实验。

**📊 数据集**

四个 SemEval ABSA 基准数据集（Laptop14、Rest14、Rest15、Rest16），涵盖 ATE、ATSC、ASPE 三个子任务。

**📈 对比分析**

在 T5-Base 与 Tk-Instruct 两种 Encoder–Decoder 模型上，比较原始数据、仅生成数据、混合数据，以及 Agentic 与 Prompting 的差异。结果显示 Agentic 生成的混合数据在大多数子任务和模型上提升 0–3% F1，尤其对 T5-Base 显著；Prompting 往往不如或略降性能；单独使用生成数据性能大幅下降。

**⚠️ 局限性**

依赖开放源 LLM Qwen2.5，生成能力仍不如商业模型；单独使用生成数据无法替代人工注释；大规模生成可能引入噪声；模型对指令调优程度的依赖导致增益不均匀。

---

## 244. Experimental and Numerical Study of the Transient Response of a Cantilever Beam with a Piezoelectric Disc Sensor

**arXiv ID:** 2602.16374 | [PDF](https://arxiv.org/pdf/2602.16374v1)

**作者:** Radek Kolman `[一作]` (Institute of Thermomechanics, v.v.i. Academy of Sciences), Jan Kober `[通讯]` (Institute of Thermomechanics, v.v.i. Academy of Sciences)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究通过在悬臂梁上粘贴压电圆盘传感器，结合激光测振仪与电压测量，开展了实验和数值仿真，构建了弹-压电耦合有限元模型并对Rayleigh阻尼及电路参数进行识别。

**💡 创新点**

创新点在于：①提出了完整的压电耦合模型并引入浮动电位边界条件；②通过两种不同的参数识别策略（梯度求解TRF和无梯度CMA-ES）实现了阻尼与电路参数的自动识别；③提供了可作为基准的实验装置与数值验证流程。

**🔧 技术方法**

使用了光学激光测振仪、压电传感器、SfePy有限元软件、COMSOL对照验证、Newmark时间积分、Nitsche弱形式施加边界条件、梯度求解器Trust Region Reflective (TRF)以及CMA-ES优化。

**📊 数据集**

使用了实验测得的激光速度时序和压电电压时序作为数据集；材料参数（钢E、ρ等）由实验测定；外电路参数R、C通过识别得到。

**📈 对比分析**

通过与COMSOL仿真结果及实验测量进行对比验证，频率误差在几Hz以内；两种识别方法对比：TRF仅需39次评估即可得到良好拟合；CMA-ES虽然评估次数达6400次，但拟合精度略优，成本更高。

**⚠️ 局限性**

主要局限包括：①模型仅考虑线性压电-弹性耦合；②初始模型未考虑粘合层弹性导致频率偏差；③未考虑温度或非线性效应；④参数识别对初始猜测敏感，可能存在多解。

---

## 245. Markerless 6D Pose Estimation and Position-Based Visual Servoing for Endoscopic Continuum Manipulators

**arXiv ID:** 2602.16365 | [PDF](https://arxiv.org/pdf/2602.16365v1)

**作者:** Junhyun Park `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Minho Hwang `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**通讯引用:** 1636 | [OpenAlex ID](https://openalex.org/A5051115770)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个统一的框架，实现了连续体操纵器的无标记立体视觉6D位姿估计，并基于此实现了闭环位置视觉伺服控制。

**💡 创新点**

创新点包括：1) 采用物理一致的光照渲染模拟器生成大规模、像素精确的合成数据；2) 设计了立体多模态融合网络，联合利用分割、关键点、热图和边界框提升几何可观测性；3) 引入一次性渲染补偿模块，在不迭代的情况下实现几何一致的位姿校正；4) 开发了无标签的自监督实景适配方法，显著缩小仿真-现实差距。

**🔧 技术方法**

使用技术包括：物理基础的伪刚体URDF仿真与NVIDIA Isaac Sim；多模态卷积网络（ResNet-50+DeepLabv3）与多头注意力；可微渲染器用于生成可比较的几何特征；残差预测网络实现一次性位姿微调；基于PBVS的逆运动学闭环控制。

**📊 数据集**

数据集：在Isaac Sim中生成200k对域随机化的立体图像，包含分割、关键点、边界框与6D位姿；另外采集1k个真实场景样本，使用Charuco标记获得精确基准位姿，用于评估与自监督适配。

**📈 对比分析**

与现有方法比较：在真实数据上，经过自适配的MFFN+Refine实现平均平移误差0.83 mm、旋转误差2.76°，比之前最佳1.64 mm/5.02°提高约34.6%/13.8%；在闭环轨迹跟踪中相较开环误差减少约85%/59%；整体推理时间约210 ms，较迭代渲染方法提升数倍。

**⚠️ 局限性**

局限性：在强遮挡、烟雾等极端视觉条件下鲁棒性待验证；PBVS控制未显式建模连续体的滞后非线性，可能影响精度；当前单帧210 ms的推理速度仍不适合高频率实时控制，需要进一步加速。

---

## 246. System Identification under Constraints and Disturbance: A Bayesian Estimation Approach

**arXiv ID:** 2602.16358 | [PDF](https://arxiv.org/pdf/2602.16358v1)

**作者:** Sergi Martinez `[一作]` (Heriot-Watt University), Carlos Mastalli `[通讯]` (Heriot-Watt University)

**通讯引用:** 1148 | [OpenAlex ID](https://openalex.org/A5032652050)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

我们提出了一种贝叶斯系统识别框架，能够同时估计机器人状态轨迹与物理参数，并通过逆向动力学、接触与闭环约束、摩擦模型及能量观测实现高精度的物理一致性；

**💡 创新点**

创新点在于将逆向动力学作为硬约束引入贝叶斯优化，结合能量回归提升可观测性，并利用参数化等式约束 Riccati 递归实现线性时间复杂度；

**🔧 技术方法**

核心技术包括逆向动力学建模、隐式运动约束与 Baumgarte 稳定化、指数-特征参数化的惯性与摩擦模型、能量观测约束、贝叶斯优化及等式约束 Riccati 递归求解；

**📊 数据集**

在仿真中使用双摆、机械臂、四旋翼、人形与四足机器人；实机测试在 Unitree B1 四足配备 Z1 机械臂的硬件上，采集编码器、IMU 与执行器命令；

**📈 对比分析**

通过与前向动力学、无约束贝叶斯方法和传统频率识别基线对比，实验表明逆向动力学框架收敛更快、参数误差更低；能量观测显著降低摩擦误差，硬件实验提升了姿态与轨迹跟踪性能；

**⚠️ 局限性**

局限性包括对激励运动的依赖、对柔性部件建模不足、能量观测对功率测量精度敏感以及实时实现对计算资源的较高需求。

---

## 247. Articulated 3D Scene Graphs for Open-World Mobile Manipulation

**arXiv ID:** 2602.16356 | [PDF](https://arxiv.org/pdf/2602.16356v1)

**作者:** Martin Büchner `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2545 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 MoMa-SG 框架，能够从单次 RGB‑D 观察中提取语义与运动学信息，构建面向移动操纵的可操作三维场景图。

**💡 创新点**

创新点包括：① 统一的扭矩估计与正则化方法，可一次性同时估计旋转关节与滑动关节；② 通过交互分段与深度差分结合的交互检测；③ 在场景图中同时编码语义、运动学与包含关系；④ 引入 Arti4D‑Semantic 数据集，首次将语义层级与关节轴标签结合。

**🔧 技术方法**

核心技术包括 RGB‑D 点跟踪、深度差分与交互优先级融合、基于扭矩的运动学优化、正则化的点轨迹点积先验、GPT‑5-mini 交互模式判定、增量三维分割与 CLIP 特征匹配、二进制整数规划匹配关节与对象、以及 Open‑Vocab 实时语义查询。

**📊 数据集**

使用了 Arti4D‑Semantic（62 序列、600 次交互、包含父子关系标签）和原始 Arti4D 数据集，以及在 DROID 数据集上进行横向对比。

**📈 对比分析**

与 Pandora、HMM、ArtiPoint、ArtGS 等基线对比，MoMa‑SG 在交互分割（1D‑IoU 0.649）、关节轴误差（prismatic 13.19°、revolute 0.091°）和对象分割召回（IoU 0.824）等指标均优于对手；在两种移动机器人（HSR、Spot）上打开/关闭任务成功率均超过 80%。

**⚠️ 局限性**

主要限制是对精准相机位姿与深度信息的依赖；在光学反射、深度噪声大、实时动作识别困难等非精细场景下性能下降；且对交互识别的实时性要求较高。

---

## 248. Load Balanced Parallel Node Generation for Meshless Numerical Methods

**arXiv ID:** 2602.16347 | [PDF](https://arxiv.org/pdf/2602.16347v1)

**作者:** Jon Vehovar `[一作]` (Jozef Stefan International Postgraduate School), Gregor Kosec `[通讯]` (Institut Jožef Stefan)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种新的并行 Poisson 盘样本生成算法，用于 Meshless 方法的节点生成。

**💡 创新点**

创新点：使用预构建的工作树（work tree）和空间树结合，利用细粒度锁和原子操作避免锁定，改进线程同步，提升并行效率。

**🔧 技术方法**

技术：Poisson disc sampling、分层空间索引、hypertree、原子变量、Dekker 算法、C++ 并行 STL 等。

**📊 数据集**

数据集：单圆盘域，h 恒定，产生约 200 万到 4 亿点。

**📈 对比分析**

比较方法：与已有 Pfill 并行算法做强缩放测试，测点插入率与每线程吞吐量；结果显示在 1–64 线程时本算法约提升 2 倍，超过 64 线程性能下降。

**⚠️ 局限性**

局限：对密度变化支持有限，分布式内存实现仍未完成，线程空闲与多阶段问题导致效率下降，性能分叉现象未解释。

---

## 249. Multi-Agent Meta-Advisor for UAV Fleet Trajectory Design in Vehicular Networks

**arXiv ID:** 2602.16345 | [PDF](https://arxiv.org/pdf/2602.16345v1)

**作者:** Leonardo Spampinato `[一作]` (University of Bologna), Riccardo Marini `[通讯]` (National Laboratory of Wireless Communications of CNIT)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6`

**🎯 论文内容**

提出一种在线多智能体强化学习框架，用于无人机（UABS）舰队在车联网环境中快速适应多种服务区域和起飞配置的轨迹设计问题；

**💡 创新点**

核心创新在于引入多任务元-顾问（MAMO）机制，利用共享的元策略引导探索并通过动态覆盖机制防止过度依赖错误建议，实现更安全、更快速的多任务迁移学习；

**🔧 技术方法**

技术实现基于CTDE结构的3DQN（双DQN+Dueling网络）进行任务特定策略学习，元-顾问采用同样的价值网络进行探索指导，辅以自适应覆盖判定；

**📊 数据集**

实验数据来源于Bologna市真实街区的三种服务区域，结合SUMO仿真得到车辆移动轨迹，构造五种起飞点组合，形成一组多任务仿真环境；

**📈 对比分析**

与传统ε-greedy（不同衰减速率）、无覆盖的MAMA以及单一通用模型做对比。MAMO在训练回报、首次成功回合（FSE）和网络吞吐率等指标上均优于对手，FSE下降高达60%，服务满意度最高可达70%；

**⚠️ 局限性**

局限性包括：依赖中心控制器进行经验收集与模型同步，导致潜在通信延迟；元-顾问的泛化仍受限于任务分布，若覆盖机制不充分可能导致性能下降；实验仅在仿真环境验证，真实部署中的动态变化与干扰仍需进一步评估。

---

## 250. Machine Learning Driven Prediction of the Behavior of Biohybrid Actuators

**arXiv ID:** 2602.16330 | [PDF](https://arxiv.org/pdf/2602.16330v1)

**作者:** Michail-Antisthenis Tsompanas `[一作]` (University of the West of England), Andrew Adamatzky `[通讯]` (University of the West of England)

**通讯引用:** 12201 | [OpenAlex ID](https://openalex.org/A5036652783)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用机器学习模型对基于肌肉环的生物混合驱动器在电刺激下的最大拉力和时间序列拉力进行预测，旨在实现对生物混合机器人行为的数字孪生与可控性提升。

**💡 创新点**

在同一实验装置下，将静态预测与动态预测相结合，并首次通过加入基线拉力作为额外输入显著提升神经网络静态预测精度；同时使用LSTM实现对完整拉力时间序列的高精度再现，为后续闭环控制奠定基础。

**🔧 技术方法**

采用随机森林回归（RFR）、前馈神经网络（NN）以及长短期记忆网络（LSTM）三种机器学习方法，并对模型进行超参数调优和特征编码。

**📊 数据集**

实验数据共161个实验，涵盖多种频率、脉宽与波形组合，采集的时间步长为0.04 s，总计123,786步，其中161个最大拉力作为静态预测训练集，122,176个10步窗口用于LSTM动态预测。

**📈 对比分析**

模型比较结果显示：RFR在静态预测中取得R²≈0.9277，NN提升至R²≈0.9363，加入基线拉力后NN进一步提升至R²≈0.9425；LSTM在动态预测中达到了R²≈0.9956、MSE≈0.0013，表明对时间序列的拟合极为精准。

**⚠️ 局限性**

局限性包括：静态模型训练数据量有限（仅161例），难以推广到不同肌肉构型；动态模型仅采用单变量时间序列且为开放循环，缺乏对刺激信号的多输入建模，难以捕捉刺激启动时的突变；闭环控制与多模态输入等仍需进一步研究。

---

## 251. Wearable AR for Restorative Breaks: How Interactive Narrative Experiences Support Relaxation for Young Adults

**arXiv ID:** 2602.16323 | [PDF](https://arxiv.org/pdf/2602.16323v1)

**作者:** Jindu Wang `[一作]` (Hong Kong University of Science and Technology), Ling-Ping Yuan `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出并验证了一套基于可穿戴AR的互动休息体验框架，将日常音频/视频内容与轻量化身体活动无缝嵌入，形成“Rise–Peak–Closure”节奏的休息旅程。

**💡 创新点**

创新点包括：① 将媒体叙事时刻与活动提示实现双重对齐的“Seamless Guidance Unit”模型；② 采用音频中心媒体降低视觉负荷；③ 在互动过程中通过节奏化结构平衡沉浸与恢复，提升休息质量和后续工作准备。

**🔧 技术方法**

技术手段：XREAL Air 2 AR 眼镜、Unity 2022、AR Foundation、空间音频与手部追踪、音频/视觉提示、预设的活动指令与路径规划。

**📊 数据集**

数据集：未使用公开语料库，实验基于16名年轻知识工作者的现场体验；在设计探测阶段使用了8名参与者的视频基线素材（短片《Flowers and Trees》及改编的喜剧脱口秀音频）。

**📈 对比分析**

对比方法：在同一实验场景下进行四个条件（InteractiveBreak、Interactive Non‑Narrative、Break Reminder、Video Watching）的跨条件设计，采用重复测量ANOVA、Likert量表及用户体验量表（UES‑SF）评估。实验结果显示 InteractiveBreak 在无缝过渡、休息质量、工作准备度及整体满意度上显著优于其它三种基线，且被11/16名参与者选为长期使用首选。

**⚠️ 局限性**

局限性：① AR 设备视场、重量与色彩失真可能影响舒适度；② 研究仅在短期、控制性 Pomodoro 结构中验证，未考察长期坚持与新鲜度衰退；③ 媒体类型单一，缺乏跨类型的通用性验证；④ 样本规模较小，缺乏多样化工作环境与人群验证。

---

## 252. End-user validation of BRIGHT with custom-developed graphical user interface applied to cervical cancer brachytherapy

**arXiv ID:** 2602.16321 | [PDF](https://arxiv.org/pdf/2602.16321v1)

**作者:** Leah R. M. Dickhoff `[一作]`, Tanja Alderliesten `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在子宫颈癌放射性内照疗中，使用自定义 GUI 对 BRIGHT 半自动放射治疗规划系统进行终端用户验证，生成并选择多方案。

**💡 创新点**

首次引入专门 GUI 支持 Pareto 前沿导航、双方案比较、重优化，并将其应用于子宫颈癌 BT，验证其临床可行性。

**🔧 技术方法**

采用 MO‑RV‑GOMEA 多目标优化算法，GPU 加速剂量计算与优化，GUI 集成剂量分布可视化、DV 指标、Golden Corner 等功能。

**📊 数据集**

使用 10 例已完成四分次子宫颈癌 BT 病例的真实临床图像与标注，满足 EMBRACE‑II 协议。

**📈 对比分析**

通过 Blinded 一对一比较，4 名放射科医师评估 BRIGHT 计划与临床计划，SUS 得分 83.3 为“优秀”，BRIGHT 在 8/10 例被偏好，5 例存在临床相关差异；规划时间约 8.5 分钟。

**⚠️ 局限性**

局限性包括样本量小、病例为四分次高复杂度、GUI 导航体验仍可改进，以及未在不同机构或剂量模型中进一步验证。

---

## 253. Towards Secure and Interoperable Data Spaces for 6G: The 6G-DALI Approach

**arXiv ID:** 2602.16386 | [PDF](https://arxiv.org/pdf/2602.16386v1)

**作者:** Dimitrios Amaxilatis `[一作]` (Spark Works Ltd), Christos Verikoukis `[通讯]` (Industrial Systems Institute)

**通讯引用:** 8411 | [OpenAlex ID](https://openalex.org/A5016866376)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了面向6G网络的完整数据空间架构6G-DALI，融合GAIA-X和IDSA框架，支持安全、主权、互操作的数据共享与AI实验；

**💡 创新点**

创新点包括双层数据空间与数据湖分离设计、意图驱动的DataOps（LLM将自然语言请求转化为实验/查询）、内置市场与RAN模型管理、以及跨测试平台的自动化MLOps/FLOps工作流；

**🔧 技术方法**

采用了Federated Identity（SSI、Verifiable Credentials）、基于策略的数据合约、ELT管道、LLM、Eclipse Data Space Connector、语义三元组存储、MinIO分布式存储、云原生API等技术；

**📊 数据集**

使用的主要数据集来自6G实验平台（EUR、ISI、KUL、DT测试床）及其产生的“冷数据”和“热数据”，并通过DataOps自动清洗、增强后用于模型训练；

**📈 对比分析**

与GAIA-X与IDSA参考架构进行对比，说明对齐点（如治理、身份、语义）与差异（如双层架构、测试床集成、市场嵌入）；性能评估尚未给出定量指标，主要通过架构兼容性和功能覆盖来证明；

**⚠️ 局限性**

局限性在于仍处于设计验证阶段，缺乏大规模跨云/边缘的性能基准；对多样化测试床的依赖可能导致部署复杂性；对实时大规模数据流的处理能力尚未充分验证；

---

## 254. How Reliable is Your Service at the Extreme Edge? Analytical Modeling of Computational Reliability

**arXiv ID:** 2602.16362 | [PDF](https://arxiv.org/pdf/2602.16362v1)

**作者:** MHD Saria Allahham `[一作]` (Queen's University), Hossam S. Hassanein `[通讯]` (Queen's University)

**通讯引用:** 9867 | [OpenAlex ID](https://openalex.org/A5021196543)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对极端边缘计算（XEC）中AI推理流式服务的计算可靠性分析框架，能够在设备可用计算容量和工作负载需求随机波动的环境下量化服务的可靠性。

**💡 创新点**

创新点在于：①给出了在仅有设备声明范围（MI）和历史观测数据（MLE）两种信息情形下的闭式可靠性表达式；②将单设备分析扩展到多设备系统，推导出串行、并行和工作划分配置的可靠性公式；③提出最优工作负载分配规则和设备选择的解析边界，为调度器提供可计算的决策依据。

**🔧 技术方法**

使用概率模型（均匀分布和截断正态分布）、极大似然估计、积分解析、Lagrange乘子法求解最优分配，以及 Docker+YOLO11m 模拟环境进行实验验证。

**📊 数据集**

在实验中使用 YOLO11m 目标检测模型作为代表性推理工作负载，基于其算力需求与图像分辨率关系进行模拟；未采用公开数据集，重点是推理算力与线程分配的映射。

**📈 对比分析**

通过 Monte Carlo 抽样、模拟测量与解析公式对比，验证两种信息情形下的可靠性表达式与实际可靠性高度一致；历史数据模型在样本量充足时逼近真实可靠性，提升预测精度；多设备配置的可靠性边界与实验结果吻合，显示所提方法在设备选取和负载划分上的有效性。

**⚠️ 局限性**

局限性包括：①仅考虑计算资源的波动，未建模通信延迟和网络可靠性；②假设设备间独立性，实际可能存在相关性；③对低频或极端条件下的极值预测能力有限；④需要收集足够历史数据以获得 MLE 参数，首次部署时可能受限。

---

## 255. Docking and Persistent Operations for a Resident Underwater Vehicle

**arXiv ID:** 2602.16360 | [PDF](https://arxiv.org/pdf/2602.16360v1)

**作者:** Leonard Günzel `[一作]` (Norwegian University of Science and Technology), Martin Ludvigsen `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 2073 | [OpenAlex ID](https://openalex.org/A5009440097)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究设计并部署了一个可供迷你ROV使用的多传感器驻留式海底停靠站，并实现了其在90 m深度的自主定位、对接、充电与结构检查功能。

**💡 创新点**

创新点包括：①将声学定位（USBL）与视觉定位（ArUco标记）融合，实现从远距离声学导航到近距离视觉对接的闭环；②在海底部署一个可充电、可数据通信的无人机站点，提供持续能量和网络支持；③通过仿真双胞胎与硬件在环的SIL方法，缩短了从仿真到现场的验证周期。

**🔧 技术方法**

关键技术包括：Blueye X3迷你ROV + Jetson Nano外部计算单元；USB‑Lightweight Acoustic Modem（Evologics S2C）用于声学导航；多光束声纳与DVL实现环境感知；光学调制器实现短距离高速通信；ArUco视觉标记与OpenCV/ROS2 ArUco Pose Estimation；IMU+DVL融合的扩展卡尔曼滤波器；Gazebo仿真与ROS2软件架构。

**📊 数据集**

使用的数据集主要为现场实测数据：90 m深度驻留站的声学定位坐标、光学图像（ArUco检测）、惯性与多声束声纳传感器记录；以及仿真生成的图像序列和传感器噪声模型。

**📈 对比分析**

比较方法：在浅水与90 m深度环境下进行对接试验，记录成功率、对接时间与误差。结果显示：90 m深度前方对接成功率为90 %（9/10），侧向对接成功率为70 %（7/10），改进后为左侧10/10、右侧9/10。对接平均耗时约140 s，完整检查任务约4 min，显示系统在实际海况下能够实现可靠的自主对接与快速检查。

**⚠️ 局限性**

局限性包括：①对深度环境的光学可见度依赖度高，光照不足导致标记检测失败；②磁漂移与鱼类遮挡导致定位误差；③系统目前需要持续外部供电（电池不足以支持长期自给自足），需大型支持船舶部署；④对接站尺寸大、重量重，物流成本高；⑤在强流/波浪环境下对接可靠性受限。

---

## 256. Optical Inversion and Spectral Unmixing of Spectroscopic Photoacoustic Images with Physics-Informed Neural Networks

**arXiv ID:** 2602.16357 | [PDF](https://arxiv.org/pdf/2602.16357v1)

**作者:** Sarkis Ter Martirosyan `[一作]` (Georgia Institute of Technology), Stanislav Emelianov `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 20982 | [OpenAlex ID](https://openalex.org/A5046383892)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种自监督深度自编码器SPOI‑AE，用于光声光谱图像的光学反演和光谱去混。

**💡 创新点**

创新点在于将物理光声前向模型嵌入自编码器解码阶段，实现自监督训练，并允许吸收光谱自适应学习，从而克服传统线性假设导致的误差。

**🔧 技术方法**

采用全连接神经网络构建吸收系数和散射系数估计器，使用批量归一化、LeakyReLU 与 ReLU 激活；通过 Adam 优化器进行训练，损失函数为 MSE 与 MSAD 的加权和；解码阶段直接实现光学前向方程。

**📊 数据集**

使用 11 幅来自 7 只小鼠腋下淋巴结的光声光谱图像（146 波长，680–970 nm）作为训练集，3 幅作为测试集；并在模拟血管囊胚上进行地面真实验证。

**📈 对比分析**

与传统线性去混方法（Lit.NLS 与 NMF）对比；在真实数据上 SPOI‑AE 在 MSE、MSAD 与平均 R² 上均显著优于两者；在模拟数据上 MAE[SO₂] 为 2.63 pp，优于 NMF 但略逊于 Lit.NLS。

**⚠️ 局限性**

限制包括：无真实标签的光学参数无法直接验证；仅针对淋巴结组织，未验证跨组织或跨物种泛化；缺乏不确定性量化；需要更多半监督或监督数据来进一步提升性能。

---

## 257. SCAR: Satellite Imagery-Based Calibration for Aerial Recordings

**arXiv ID:** 2602.16349 | [PDF](https://arxiv.org/pdf/2602.16349v1)

**作者:** Henry Hölzemann `[一作]` (Fraunhofer FKIE), Michael Schleiss `[通讯]` (University of Bundeswehr Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SCAR框架，利用卫星影像和DEM实现无人机视觉-惯导系统的长期自动标定校正；通过将机载图像与地理参考影像对齐，生成2D–3D对应关系，联合优化相机内参与相机与INS的外参；并在多季节飞行数据上验证其稳健性。

**💡 创新点**

将长期标定视为对地理参考的反复验证与优化；引入基于卫星影像的自动生成GCP，形成外部绝对参考；发布可复现的开源工具箱，并在多年的真实飞行数据上进行系统评估。

**🔧 技术方法**

基于因子图的非线性优化（GTSAM），GNSS/INS位姿先验，航空-卫星图像匹配（SuperPoint+LightGlue或RoMa），DEM上升值生成3D点，鲁棒核函数，分阶段块坐标优化，以及后期的最小二乘外参对齐。

**📊 数据集**

六条长周期航拍数据集（2022‑2024年），采用超轻型飞机搭载FLIR BFS相机与SBG Ellipse2‑D INS；使用北莱茵-威斯特法伦的公开正射影像和DEM；数据集将公开作为SCAR基准。

**📈 对比分析**

与Kalibr、COLMAP、VINS‑Mono等传统标定方法对比，使用中位投影误差、视觉定位旋转/平移误差以及2/2、5/5、10/10阈值的姿态精度评估；SCAR将投影误差从约45‑47像素降至5‑7像素，视觉定位旋转误差从约2°降至0.3‑1.3°，在严格阈值下的姿态成功率显著提升。

**⚠️ 局限性**

对航空‑卫星对应的不确定性依赖场景与位置，使用全局等方差模型；受制于正射影像与DEM的质量与时效；仅适用于近垂直视角，倾斜视角会降低几何鲁棒性；在特定训练段上可能出现轻微过拟合，需要更强正则化与跨区域验证。

---

## 258. The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks

**arXiv ID:** 2602.16340 | [PDF](https://arxiv.org/pdf/2602.16340v1)

**作者:** Eitan Gronich `[一作]` (Weizmann Institute of Science), Gal Vardi `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 153 | [OpenAlex ID](https://openalex.org/A5034725081)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了基于动量的优化器（如Adam、Muon、Muonsignum等）在光滑齐次模型上的隐式偏差，证明它们趋向于相应范数下的最大间隔解。

**💡 创新点**

首次将“近似最速下降”框架应用于动量优化器，统一阐述了多种优化器的间隔最大化性质，并扩展了对光滑齐次网络的分析。

**🔧 技术方法**

利用最速下降理论、KKT条件、近似最速下降定义以及对动量估计的收敛性分析。

**📊 数据集**

使用MNIST数据集的两层齐次网络（含平方ReLU与ReLU激活）进行实验。

**📈 对比分析**

通过对比NGD、Signum、Adam、Muon及Muonsignum等优化器在不同范数下的间隔值和余弦相似度，实验表明NGD最大化ℓ2间隔，Signum与Adam最大化ℓ∞间隔，Muon最大化谱范数间隔，Muonsignum表现为两者的组合。

**⚠️ 局限性**

结果仅在光滑齐次模型且假设参数方向收敛的情况下成立；对非光滑模型（如ReLU网络）及无方向收敛假设的情况仍未证明；对Adam和Muon的方向收敛理论缺失。

---

## 259. Subtractive Modulative Network with Learnable Periodic Activations

**arXiv ID:** 2602.16337 | [PDF](https://arxiv.org/pdf/2602.16337v1)

**作者:** Tiou Wang `[一作]` (KTH Royal Institute of Technology), Sabine Süsstrunk `[通讯]` (EPFL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Subtractive Modulative Network（SMN）作为一种基于子tractive 合成思想的高效隐式神经表示框架。

**💡 创新点**

创新点在于引入可学习的正弦激活层（Oscillator）和多阶段乘法调制模块（Filter），实现频谱的主动抑制和谐波生成。

**🔧 技术方法**

采用可学习正弦层、乘法调制掩码、以及自掩码平方放大等技术，构成子tractive 信号处理流水线。

**📊 数据集**

使用 Kodak、DIV2K 图像数据集以及 NeRF 合成场景数据集进行评估。

**📈 对比分析**

与 Gauss、SIREN、WIRE、RINR 等基线模型对比，SMN 在 2D 图像上 PSNR 最高（41.40 dB）且参数最少，在 3D NeRF 上平均 PSNR 32.98 dB，明显优于对手。

**⚠️ 局限性**

局限在于对多层滤波器深度的易优化问题（深度过大导致梯度消失），且目前仅在自然图像和合成场景上验证，缺乏对更复杂或多模态数据的实验。

---

## 260. A Self-Supervised Approach for Enhanced Feature Representations in Object Detection Tasks

**arXiv ID:** 2602.16322 | [PDF](https://arxiv.org/pdf/2602.16322v1)

**作者:** Santiago C. Vilabella `[一作]` (Menéndez Pelayo International University), Beatriz Remeseiro `[通讯]` (University of Oviedo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用自监督学习训练特征提取器，以减少物体检测中对标注数据的依赖

**💡 创新点**

将SimCLR预训练的EfficientNet B1作为特征提取器，并通过单层检测头证明在少标签数据下显著提升定位性能

**🔧 技术方法**

利用SimCLR对比学习、InfoNCE损失、EfficientNet B1 backbone以及Grad‑CAM可视化技术

**📊 数据集**

使用COCO数据集进行无标签预训练，PascalVOC（2007/2012）进行下游检测任务

**📈 对比分析**

与ImageNet预训练的Baseline进行对比，在TINY和FULL数据集上，定位指标（Mean IoU、IoU 0.5/0.7）明显优于Baseline，分类指标略低

**⚠️ 局限性**

分类性能不及ImageNet预训练，且仅在极简检测头上验证；预训练数据规模有限，未来需更大无标签集和更复杂检测网络

---

## 261. CADEvolve: Creating Realistic CAD via Program Evolution

**arXiv ID:** 2602.16317 | [PDF](https://arxiv.org/pdf/2602.16317v1)

**作者:** Maksim Elistratov `[一作]` (Lomonosov Moscow State University), Dmitrii Zhemchuzhnikov `[通讯]` (Lomonosov Moscow State University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于演化的离线数据生成管道CADEvolve，利用VLM不断编辑并验证CAD程序，从手写原语生成多操作参数化生成器，并进一步构造可执行脚本，最终形成覆盖完整CadQuery操作集的三层数据集；

**💡 创新点**

创新点在于：① 将演化搜索迁移至数据生成阶段，实现自动化生成高复杂度、可执行的CAD程序；② 通过检索增强、分阶段验证和自我修复提升程序质量；③ 采用统一化与二值化规范化技术，得到可训练的标准化脚本；

**🔧 技术方法**

主要技术包括VLM（如GPT‑5‑mini）指导的提议-执行-过滤循环、检索增强代码合成、阶段化几何验证、脚本规范化与二值化、代码级重写、以及基于GRPO的RL微调；

**📊 数据集**

使用的数据集为CADEvolve‑3L（G、P、C三层）以及从ABC、ShapeNet、MCB、DeepCAD、Fusion 360 Gallery等公开数据中提取并增强的程序；

**📈 对比分析**

在Image2CAD基准上对DeepCAD、Fusion 360 Gallery和MCB进行评估，模型在CD下降、IoU上升的同时保持低错误率，整体性能超过现有cadrille等方法，达成SOTA水平；

**⚠️ 局限性**

局限性包括：生成数据为合成分布，可能与实际工业CAD分布不一致；程序仅在CadQuery实现，跨平台转换存在挑战；以及在初期仍存在草图多样性不足和几何精度偏差等问题。

---

## 262. MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks

**arXiv ID:** 2602.16313 | [PDF](https://arxiv.org/pdf/2602.16313v1)

**作者:** Zexue He `[一作]` (Stanford University), Alex Pentland `[通讯]` (Stanford University)

**通讯引用:** 93690 | [OpenAlex ID](https://openalex.org/A5007176508)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MemoryArena——一种面向多会话、任务子任务相互依赖的统一评估平台，专门测评 LLM 代理在 Memory‑Agent‑Environment 循环中的记忆使用与决策能力。

**💡 创新点**

创新点在于将记忆与行动、环境反馈紧密耦合，设计了四类人造任务（分布式购物、旅行规划、递进式搜索、形式推理），每个任务包含多次子任务，后续子任务必须利用先前会话中的记忆才能完成；同时提供统一的实验框架、评估指标和基准结果。

**🔧 技术方法**

使用了多种记忆机制：长上下文缓冲（如 GPT‑5.1‑mini、GPT‑4.1‑mini、Claude‑Sonnet‑4.5 等）、外部记忆系统（MemGPT、Mem0、ReasoningBank）、以及检索增强生成（RAG）系统（BM25、embedding‑RAG、MemoRAG、GraphRAG）。所有实验均在同一 LLM 代理 GPT‑5.1‑mini 上完成。

**📊 数据集**

构建了四个数据集：Bundle Web Shopping（150 任务）、Group Travel Planning（270 任务）、Progressive Web Search（256 任务）和 Formal Reasoning（80 任务，40 数学 + 20 物理）。每个任务包含数十步子任务，整体长度约 57 步，生成的推理轨迹总计 40k+ tokens。

**📈 对比分析**

与现有记忆评测（LoCoMo、LongMemEval 等）和单会话代理基准（WebArena、SWE‑Bench 等）对比，发现即便在现有长上下文记忆基准上表现优秀的模型，在 MemoryArena 上任务成功率（SR）和进度分数（PS）普遍低于 30%，多数为 0%；RAG 在搜索与推理任务中显著提升，但在旅行规划和购物任务中效果有限；外部记忆整体不如长上下文或 RAG，且不具“一加一大于二”优势。

**⚠️ 局限性**

局限性包括：1）当前记忆系统与任务代理未联合训练，导致查询/检索方式与 LLM 的上下文处理不匹配；2）记忆机制大多为通用式召回或压缩，缺乏对任务特定状态变量的精确跟踪；3）多会话任务对 POMDP 估计要求高，现有模型在长期依赖下易出现误差累积；4）外部记忆导致显著延迟，未能兼顾效率与效果。

---

## 263. Dynamic Modeling and MPC for Locomotion of Tendon-Driven Soft Quadruped

**arXiv ID:** 2602.16371 | [PDF](https://arxiv.org/pdf/2602.16371v1)

**作者:** Saumya Karan `[一作]` (Indian Institute of Technology Gandhinagar), Madhu Vadali `[通讯]` (Indian Institute of Technology Gandhinagar)

**通讯引用:** 603 | [OpenAlex ID](https://openalex.org/A5028564486)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一款只用四个伺服驱动的柔性四足机器人SLOT，构建了基于离散Cosserat棒理论的软腿动力学模型，提出了模块化的全身耦合仿真框架，并嵌入凸优化的模型预测控制（MPC）实现实时全身运动控制，最终在硬件上完成了爬行、步行和全向行走的实验验证。

**💡 创新点**

① 在软腿动力学方面首次将离散Cosserat棒理论与全身刚体动力学耦合；② 通过物理一致的反作用力接口实现了软硬耦合的可扩展模型；③ 在MPC中使用物理建模的力–角关系将地面反作用力直接映射到张力驱动，实现了在仅四个执行器下的稳定多足步态；④ 通过实验实现了<5mm RMSE的重心轨迹跟踪，验证了模型的高保真性。

**🔧 技术方法**

Cosserat棒理论（离散化）、有限元与刚体动力学耦合、凸二次规划（SCS）模型预测控制、ROS2实时框架、视觉–惯性SLAM（Intel RealSense D435 + IMU）、PID低层执行器控制、GPU+CPU并行仿真。

**📊 数据集**

主要使用SLOT硬件实验数据（ArUco标定+RealSense RGB‑D）作为验证集；未使用公开的第三方数据集；对单腿、全身行走、爬行和全向行走进行内部对比。

**📈 对比分析**

采用实验测量的重心轨迹与仿真轨迹进行RMSE、MAE、NRMSE对比；在步行和爬行中分别取得CoM X、Z方向RMSE约4–5 mm；MPC求解时间6.2 ms，整个控制循环30 Hz；在不同扰动情形下的递归可行性、最终成本均为0，收敛时间≤14.6 s，证明控制器的稳态性能和可行性。

**⚠️ 局限性**

① 模型假设软腿仅在XZ平面变形，忽略三维扭转和外向弯曲；② TPU线性弹性近似忽略粘弹性和应变率效应；③ 体积耦合采用分离式仿真，未考虑腿-躯干的完整动力耦合，可能在高动态或复杂地形下失效；④ 仅在平坦地面上验证，缺乏对不规则地形的适应性；⑤ 对外部噪声和持续扰动的鲁棒性有限。

---

## 264. Designing Production-Scale OCR for India: Multilingual and Domain-Specific Systems

**arXiv ID:** 2602.16430 | [PDF](https://arxiv.org/pdf/2602.16430v1)

**作者:** Ali Faraz `[一作]` (Krutrim AI), Shubham Agarwal `[通讯]` (Krutrim AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对印度多语言、多样化文档，提出两套OCR系统：Chitrapathak（两种训练策略）和Parichay（针对政府文档的结构化字段提取）。

**💡 创新点**

创新点在于系统性对比VLM端到端训练与fine‑tune OCR专用模型的准确率‑延迟权衡，并通过加入旋转预处理显著提升结构化提取的鲁棒性；同时提供实测SOTA指标与多模型比较。

**🔧 技术方法**

技术包括：Vision‑Language模型（CLIP‑336、Qwen2.5‑VL、Phi‑3.5 Vision Instruct）、投影MLP、2D‑RoPE窗口注意力、LoRA参数高效微调、完整参数fine‑tune、vLLM推理加速与动态裁剪。

**📊 数据集**

数据集：Chitrapathak‑1训练集约7M印刷书页（多语言），Chitrapathak‑2训练集约1.1M样本；Parichay使用约21K标注的政府身份证件与车辆/保险等表单（5K测试集）。

**📈 对比分析**

在IndicVisionBench‑OCR、Synthdog、SROIE、旧书OCR等基准上，Chitrapathak‑2在多数语言实现SOTA（如Telugu char‑ANLS 6.69），比Chitrapathak‑1快3–6×；Parichay‑2+旋转在结构化字段提取上EM 89.8%并比Parichay‑1快约4×。对比Gemini‑2.5 Flash、GPT‑4o、Gemma‑3、LLaMA‑4、Nanonets‑OCR2‑3B、Surya OCR等模型，表现均优于或接近SOTA。

**⚠️ 局限性**

局限性包括：对稀有脚本的准确率仍有下降；复杂表单和密集布局仍易出现序列错误；Tokenizer对某些脚本导致高延迟；对极高分辨率文档仍需裁剪；需要更多标注数据以进一步提升泛化。

---

## 265. Easy Data Unlearning Bench

**arXiv ID:** 2602.16400 | [PDF](https://arxiv.org/pdf/2602.16400v1)

**作者:** Roy Rinberg `[一作]` (Harvard University), Volkan Cevher `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了一个统一、可扩展的机器忘记（unlearning）评估基准 Easy Data Unlearning Bench，提供预训练模型、oracle 模型集合以及 KL 边际差异度量。

**💡 创新点**

创新点在于：1) 统一标准化的评估框架与开源资源；2) 采用 KL 边际差异直接衡量与 oracle 的分布相似性，抵抗“游戏”风险；3) 引入 teacher‑forcing 扩展，可用于 LLM 的下一词预测评估。

**🔧 技术方法**

核心技术包括：KL 边际差异度量、预训练/oracle 模型集群、自动化实验流水线、离线 margin 计算与可视化；实现基于 PyTorch 的快速调用。

**📊 数据集**

使用 CIFAR‑10 数据集（ResNet‑9）和 200 个预训练模型，10 个忘记集合（10–1000 样本）以及相应的 oracle 模型和预计算 margin。

**📈 对比分析**

对比方法：直接与不使用忘记集合重新训练的 oracle 进行 KL 边际差异比较；实验显示基准 retrain 与 noisy‑SGD 等方法在 KL 评分上存在显著差距，证明框架可捕捉真实的忘记效果。

**⚠️ 局限性**

局限性：目前仅针对分类任务；数据集与模型规模有限（仅 CIFAR‑10/ResNet‑9）；对大型生成模型的评估仍在规划中；需要进一步验证跨任务与跨规模的通用性。

---

## 266. Computing Tarski Fixed Points in Financial Networks

**arXiv ID:** 2602.16387 | [PDF](https://arxiv.org/pdf/2602.16387v1)

**作者:** Leander Besting `[一作]` (RWTH Aachen University), Lars Huth `[通讯]` (RWTH Aachen University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文设计并实现了一套多阶段、基于Tarski固定点的算法，能够在多元化的Eisenberg‑Noe金融网络（支持任意单调分段线性支付函数和违约成本）中，强多项式时间求得最小清算状态，并给出计算最大清算状态、范围清算状态以及债权交易的高效算法。

**💡 创新点**

创新点包括：
- 通过“逐步注入资产 + 溶解SCC”策略，将最小清算问题转化为可解的线性规划与特征向量问题，实现了强多项式时间；
- 将任意单调分段线性支付函数通过多边形拆分与辅助银行的构造等价转化为优先‑比例支付函数，证明两类网络在清算状态上等价；
- 在含违约成本的模型中引入辅助“沉降银行”，用线性规划捕捉违约成本对清算状态的影响；
- 通过对可达SCC的“洪水”操作与斜率分析，构造了可判定并求解债权交易（信用正向交易）的算法。

**🔧 技术方法**

主要技术手段：
- Tarski固定点理论与Knaster‑Tarski定理；
- 逐步资产注入的分段线性逼近与斜率计算；
- SCC分解与有向无环图（DAG）上的洪水算法；
- 线性规划与线性方程组求解（包括特征向量求解）；
- 线性变换与等价网络构造；
- 迭代与二分搜索实现对债权交易返回值的最优求解。

**📊 数据集**

本文没有使用实测数据集，而是以理论分析为主；所有结果均基于抽象的金融网络模型，并通过构造性证明给出算法的时间复杂度与正确性。

**📈 对比分析**

与已有工作相比，本文提供的最小清算状态算法从此前只能在特定支付函数（如边排序）下实现，扩展到所有单调分段线性函数，且在强多项式时间内完成；最大清算状态的计算也实现了完全多项式时间。债权交易方面，本文证明了存在性区间并给出了最优返回值的多项式求解方法，弥补了先前仅针对最大清算状态的研究。总体而言，算法的时间复杂度为 O((n+k)(n^3+m))（k 为分段数），在理论上优于或等同于现有的迭代或近似方法。

**⚠️ 局限性**

限制与挑战：
- 仅适用于单调分段线性或其等价的优先‑比例支付函数，无法直接处理非分段线性或非单调支付函数；
- 对大规模网络的实际实现可能受限于求解线性方程组与特征向量的高复杂度；
- 违约成本的处理依赖于辅助银行的构造，可能导致网络规模膨胀；
- 对动态冲击、时变网络结构的适应性尚未研究；
- 论文中的实验验证缺失，主要基于理论证明，实际性能需进一步评估。

---

## 267. Scalable Base Station Configuration via Bayesian Optimization with Block Coordinate Descent

**arXiv ID:** 2602.16378 | [PDF](https://arxiv.org/pdf/2602.16378v1)

**作者:** Kakeru Takamori `[一作]` (University of Electro-Communications), Koya Sato `[通讯]` (University of Electro-Communications)

**通讯引用:** 1281 | [OpenAlex ID](https://openalex.org/A5069290606)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对密集基站部署，提出一种将块坐标下降与贝叶斯优化相结合的可扩展基站配置搜索方法。

**💡 创新点**

创新点在于利用块坐标下降将高维参数空间拆分为每个基站的子问题，显著降低每次贝叶斯优化的有效维度，从而克服维度灾难。

**🔧 技术方法**

采用贝叶斯优化（高斯过程+RBF核+期望改进）与块坐标下降策略，逐基站顺序更新位置、功率与天线朝向。

**📊 数据集**

在 Sionna RT 提供的 3D 城市场景（1 km × 1 km 区域）上进行射线追踪仿真，评估区域平均吞吐量。

**📈 对比分析**

与平方格布置的方向性/全向天线、仅优化功率的平方格布置、以及传统整体贝叶斯优化进行比较。实验显示，在 N_Tx=16 时提升 15.8%，N_Tx=25 时提升 21.5%，显著优于基线方法。

**⚠️ 局限性**

仅在单一城市场景验证，尚未评估不同地形或更大规模部署的鲁棒性；块顺序随机化可能未能达到最优更新策略，且每轮内部贝叶斯优化仍需多次射线追踪计算。

---

## 268. A Multihop Rendezvous Protocol for Cognitive Radio-based Emergency Response Network

**arXiv ID:** 2602.16367 | [PDF](https://arxiv.org/pdf/2602.16367v1)

**作者:** Zahid Ali `[一作]` (Atlantic Technological University), Saim Ghafoor `[通讯]` (Atlantic Technological University)

**通讯引用:** 270 | [OpenAlex ID](https://openalex.org/A5046496946)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出一种多跳双模时钟算法（M-DMCA），实现认知无线电灾备网络中的高效节点发现与拓扑构建。

**💡 创新点**

创新点在于：①双通道（素数/非素数）跳频策略提高单时隙多次 rendezvous 机会；②引入三向握手机制消除传统二向握手的时隙延迟；③在存在高主信号活动的异步频道环境下，显著缩短 rendezvous 时间。

**🔧 技术方法**

使用技术包括：基于素数/非素数划分的双模时钟跳频；能量检测的频谱感知；三向握手协议；NS-3.35 仿真框架；以及记忆无关的 ON/OFF 马尔可夫续延模型来模拟主信号活动。

**📊 数据集**

主要数据集为仿真生成的随机网络拓扑（1000m×1000m 区域内 3/10/20 个节点），采用 10 或 20 条可用频道，并在 0%、85% 及混合主信号活动下进行实验；未使用公开真实数据集。

**📈 对比分析**

与传统单跳或单向握手协议（RCS、MCA、EMCA）相比，M‑DMCA 在 20 节点、20 条频道、m=2 及 85% 主信号活动的最坏情况下降低平均 rendezvous 时间 24%；在 2WH 与 3WH 的比较中，3WH 的平均时间比 2WH 低约 50%，并且在各种频道相似度和主信号强度下均保持领先。

**⚠️ 局限性**

主要局限包括：假设节点数量已知且网络为静态部署；每个节点仅有单一射频接口；未考虑节点动态加入/离开导致的拓扑变化；未来需扩展至未知/动态节点数的场景。

---

## 269. Explainability for Fault Detection System in Chemical Processes

**arXiv ID:** 2602.16341 | [PDF](https://arxiv.org/pdf/2602.16341v1)

**作者:** Georgios Gravanis `[一作]` (International Hellenic University), Konstantinos Diamantaras `[通讯]` (International Hellenic University)

**通讯引用:** 4706 | [OpenAlex ID](https://openalex.org/A5047747213)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了将两种后置解释方法IG（Integrated Gradients）和SHAP（SHapley Additive exPlanations）应用于一个高精度LSTM故障诊断模型，分析其在Tennessee Eastman Process（TEP）化工过程中的解释效果，并比较两者对特征重要性的评估；

**💡 创新点**

首次将IG与SHAP并行用于多变量时间序列化工过程故障诊断，验证两种方法在根因识别上的一致性与差异，发现SHAP在某些失效（如IDV8/12/18）更能精准指示关键变量，为后置解释在工业过程中的适用性提供了实证依据；

**🔧 技术方法**

采用LSTM深度学习分类器进行多类别故障检测，随后使用Post‑hoc XAI技术IG与SHAP对模型决策进行特征归因，利用归一化重要性得分和热力图进行可视化；

**📊 数据集**

使用Tennessee Eastman Process（TEP）仿真数据集，包含20种预定义失效（IDV1–20），每个失效包含多维传感器测量与过程变量；

**📈 对比分析**

通过对失效前100个采样点的归一化特征重要性得分进行对比，按失效影响区域分组进行聚类分析。实验表明LSTM分类器在所有失效上均达到99%准确率，IG与SHAP在大多数失效上特征重要性一致，但在IDV8/12/18等失效中SHAP提供了更为细致的根因指示；

**⚠️ 局限性**

实验仅针对TEP仿真过程，样本量限制在100条失效样本，未验证在更大规模或不同化工过程中的通用性；IG在某些失效中的解释性不足，说明后置解释方法仍需针对特定任务进行调优。

---

## 270. push0: Scalable and Fault-Tolerant Orchestration for Zero-Knowledge Proof Generation

**arXiv ID:** 2602.16338 | [PDF](https://arxiv.org/pdf/2602.16338v1)

**作者:** Mohsen Ahmadvand `[一作]` (Zircuit), Ching-Lun Chiu `[通讯]` (Zircuit)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出push0，一个可扩展且容错的零知识证明调度框架。

**💡 创新点**

创新点在于事件驱动的调度-收集架构、基于优先级队列的头链顺序保证、无状态调度器和可插拔收集策略，实现了零知识证明任务的自动重试与跨机故障恢复。

**🔧 技术方法**

使用了分布式消息总线（NATS JetStream）、Kubernetes、OpenTelemetry/Prometheus监控、持久化优先级队列、无状态分派器和分区亲和路由。

**📊 数据集**

使用了Zircuit zkRollup生产数据（14M+块）以及模拟的zkEVM、SP1等证明工作负载。

**📈 对比分析**

与单线程调度和其他自定义服务相比，push0在生产集群中每任务中位调度延迟<10 ms，水平扩展效率>99%，在0–5 s模拟证明下保持0.05%额外开销；在SP1 GPU证明场景下平均证明完成时间为24.4 min（区块级）和2 min（聚合），无调度失败。

**⚠️ 局限性**

局限性包括：仅支持可信消息总线、缺乏完全去中心化与拜占庭容错、对证明有效性仅在链上验证、对加密输入隐私需应用层加密、需要手动调整ACK超时以适配不同证明时延。

---

## 271. HAWX: A Hardware-Aware FrameWork for Fast and Scalable ApproXimation of DNNs

**arXiv ID:** 2602.16336 | [PDF](https://arxiv.org/pdf/2602.16336v1)

**作者:** Samira Nazari `[一作]` (University of Zanjan), Christian Herglotz `[通讯]` (Tallinn University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了HAWX框架，利用多层敏感度评分在运算符、滤波器、层和模型层面实现硬件感知的DNN近似设计空间探索；

**💡 创新点**

创新点在于：①多级敏感度评分可实现细粒度（滤波器级）近似；②通过预测模型结合硬件成本，实现针对空间与时序加速器的硬件感知配置；③显著降低搜索复杂度，滤波器级搜索相较全穷举提升至10^6+倍；

**🔧 技术方法**

使用深度学习梯度与输出误差相乘得到操作敏感度，聚合成滤波器/层/模型级评分；结合预测模型估计精度、功耗和面积；采用k-means挑选代表样本；通过二分搜索确定阈值；实现针对空间（数据流）和时序（FGPU）加速器的映射；

**📊 数据集**

评估使用LeNet-5（MNIST）、VGG-11、ResNet-18（CIFAR‑10）和EfficientLiteNet（GTSRB）等四个主流DNN；

**📈 对比分析**

与全穷举搜索、TFApprox、I‑NN、AME、RL‑DSE等方法对比：在LeNet‑5上层级搜索加速23×，滤波器级搜索超过10^6×；对EfficientLiteNet达到10^5000×；在硬件映射上，数据流和FGPU加速器实现功耗/面积显著降低，且保持精度与穷举相近；

**⚠️ 局限性**

局限性在于：目前仅验证了乘法近似（8‑bit），对加法等其他算子推广有限；对更大规模网络或更复杂算子库的评估尚未展开；硬件映射的预测模型需针对不同架构手工调参，自动化程度还有提升空间。

---

## 272. Individual Fairness in Community Detection: Quantitative Measure and Comparative Evaluation

**arXiv ID:** 2602.16326 | [PDF](https://arxiv.org/pdf/2602.16326v1)

**作者:** Fabrizio Corriera `[一作]` (Leiden University), Akrati Saxena `[通讯]` (Leiden University)

**通讯引用:** 455 | [OpenAlex ID](https://openalex.org/A5055236010)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究社区检测中的个体公平性，提出基于共现矩阵的个体公平度量 IB 与整体公平度量 IB_G。

**💡 创新点**

创新点在于首次在社区检测领域引入可量化的个体公平度量，并证明个体公平与群体公平并不等价。

**🔧 技术方法**

使用余弦距离衡量节点共现向量差异，并将该度量与 30 种主流社区检测算法结合实现。

**📊 数据集**

实验数据包括 ABCD 基准生成的合成网络（不同混合参数）以及 Email‑Eu‑core、Football、Polbooks 三个真实网络。

**📈 对比分析**

通过与 Modularity、NMI、ARI、NF1 等质量指标以及群体公平指标 Φ 的对比，结果显示部分算法在高性能与高个体公平之间存在权衡，且无单一算法始终占优。

**⚠️ 局限性**

主要局限在于共现矩阵对稀疏图敏感，且当社区不可辨识时个体公平度量趋于 0，导致公平评估失效。

---

## 273. Interpolation in Proof Theory

**arXiv ID:** 2602.16318 | [PDF](https://arxiv.org/pdf/2602.16318v1)

**作者:** Iris van der Giessen `[一作]` (University of Amsterdam), Roman Kuznets `[通讯]` (Institute of Computer Science of Czech Academy of Sciences)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了证明论方法用于证明不同逻辑的插值性质（CIP、LIP、UIP、ULIP）

**💡 创新点**

将Maehara与Pitts方法统一到通用框架，并提出半分析规则和完全终止性标准来判定何种系统可实现插值

**🔧 技术方法**

利用分裂序列、标签序列、超序列、嵌套序列等结构化推理体系，并在此基础上实现Maehara、Pitts算法的自动化构造插值

**📊 数据集**

本文不使用实验数据集，主要以形式化证明系统为研究对象

**📈 对比分析**

通过对不同规则类型的分析与已知插值定理的对比，证明了多类逻辑（经典、直觉、模态、子结构、非正常模态、条件逻辑等）中的插值性质，且在可构造系统下插值的大小与证明长度线性相关

**⚠️ 局限性**

局限在于部分逻辑仍缺乏半分析或完全终止性的计算机/证明系统，且对更复杂结构的扩展尚未完全覆盖

---

## 274. A Graph Meta-Network for Learning on Kolmogorov-Arnold Networks

**arXiv ID:** 2602.16316 | [PDF](https://arxiv.org/pdf/2602.16316v1)

**作者:** Guy Bar-Shalom `[一作]` (Technion), Haggai Maron `[通讯]` (Technion)

**通讯引用:** 1768 | [OpenAlex ID](https://openalex.org/A5050312737)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了首个针对 Kolmogorov–Arnold 网络（KAN）的权重空间模型，利用 KAN 的节点-边图表示（KAN‑graph）并构建基于 GNN 的 WS‑KAN 架构，用于预测 KAN 的性能、生成新权重以及直接学习剪枝掩码。

**💡 创新点**

创新点在于：①证明 KAN 与传统 MLP 具有相同的隐藏层排列对称性；②设计了 KAN‑graph，将可学习的一维函数嵌入边特征，并加入位置编码；③提出兼容双向消息传递的 GNN 结构，使模型天然满足对称性并能逼近 KAN 的前向传递；④构建了首个 KAN 模型“动物园”，覆盖多任务与多数据集，为后续权重空间研究提供基准。

**🔧 技术方法**

使用技术包括：B‑spline 参数化的一维函数、图神经网络（Message‑Passing GNN）、位置编码、双向消息传递、对齐与增广策略（Alignment, Augmentation）以及多种基线（MLP、DMC、DeepSets、SetTransformer）。

**📊 数据集**

所用数据集有：MNIST、Fashion‑MNIST、Kuzushiji‑MNIST、CIFAR‑10、合成 Sine 以及 KAN‑based INR（Sine、MNIST、F‑MNIST、K‑MNIST、CIFAR‑10）。

**📈 对比分析**

与 MLP、MLP+Aug、MLP+Align、DMC、DeepSets、SetTransformer 等基线对比，WS‑KAN 在 INR 分类、准确性预测和剪枝掩码预测任务上均取得最高或接近最高的指标（准确率/均方误差/R²/ROC‑AUC 等）。在 OOD 宽度扩展中表现出可观的鲁棒性，尽管随着宽度增大性能略有下降。

**⚠️ 局限性**

限制与挑战：①目前仅验证了宽度扩展，未在更深或更复杂拓扑上测试；②对图规模的线性复杂度在大规模 KAN 仍可能成为瓶颈；③缺乏对 CNN‑KAN 等变体的实验与理论推广；④对对称性对表达能力影响的深入理论分析仍待进一步完善。

---

## 275. Formalized Run-Time Analysis of Active Learning -- Coalgebraically in Agda

**arXiv ID:** 2602.16427 | [PDF](https://arxiv.org/pdf/2602.16427v1)

**作者:** Thorsten Wißmann `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Thorsten Wißmann `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5059938244)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文将学习算法建模为协代数，并在 Agda 证明器中形式化学习游戏的运行时复杂度，特别给出了数值猜谜游戏的 𝒪(log n) 上下界；

**💡 创新点**

创新点在于提出了通用的游戏类型、学习者和教师的定义，消除了对隐藏自动机和状态数的依赖，能够在不知隐藏模型的前提下证明复杂度；

**🔧 技术方法**

主要技术包括范畴论中的协代数、泛函形式的游戏类型、自然变换与语义解释，以及 Agda 的定理证明；

**📊 数据集**

论文未使用传统实验数据集，而是通过形式化构造和证明展示结果；

**📈 对比分析**

由于采用形式化证明而非实验评估，本文未给出与现有算法的性能对比，但在数值猜谜游戏中证明了最佳的二分搜索上限；

**⚠️ 局限性**

局限性在于目前仅在数值猜谜游戏上验证，未扩展到 DFA、Mealy 机等实际自动机学习算法；

---

## 276. Reintroducing the Second Player in EPR

**arXiv ID:** 2602.16410 | [PDF](https://arxiv.org/pdf/2602.16410v1)

**作者:** Leroy Chew `[一作]` (Czech Technical University in Prague), Martin Suda `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 81391 | [OpenAlex ID](https://openalex.org/A5045719436)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对EPR（有效命题逻辑）的子类进行研究，定义了一个与QBF等价的PSPACE完备子类，并给出了相应的求解算法。

**💡 创新点**

创新点在于将QBF的两人游戏语义重新引入EPR，构造出既保持EPR闭包又能通过分支分离实现多级多项式阶层的子类。

**🔧 技术方法**

采用量化公式转换、分支搜索算法、分叉索引理论以及Robinson分辨法闭包证明等技术。

**📊 数据集**

在TPTP库的936个CNF实例中检索，识别出308个属于该子类的实例。

**📈 对比分析**

通过对TPTP实例的实验，发现约50个纯命题、109个单子子以及49个两“活跃”参数实例，证明算法能定位不同阶层问题，性能相较传统EPR求解器在这类实例上更具针对性。

**⚠️ 局限性**

局限在于算法仍为递归式且缺乏实用优化，难以在大规模实例上直接竞争，并且对非单调或包含等价式的公式检测不足。

---

## 277. Bounds and Constructions of Codes for Ordered Composite DNA Sequences

**arXiv ID:** 2602.16406 | [PDF](https://arxiv.org/pdf/2602.16406v1)

**作者:** Zuo Ye `[一作]` (Xidian University), Gennian Ge `[通讯]` (Capital Normal University)

**通讯引用:** 3817 | [OpenAlex ID](https://openalex.org/A5029449317)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出有序复合DNA通道的错误模型并给出码字大小上界及相关码的构造

**💡 创新点**

引入t-(e1,…,et)复合错误/删失模型并给出接近最优的上界与系统编码方案

**🔧 技术方法**

采用Schur多项式、可逆范德蒙矩阵、系统化生成矩阵与有限域线性方程求解等技术

**📊 数据集**

无具体实验数据集，本文为理论分析与构造

**📈 对比分析**

通过与已知上界下界对比，构造码的冗余接近理论最优，错误修正能力满足模型要求

**⚠️ 局限性**

对一般(e1,…,ek)-删失模型的上界未知；未充分利用列向量非递减属性；缺少满足生物学约束的实验验证

---

## 278. Variable-Length Semantic IDs for Recommender Systems

**arXiv ID:** 2602.16375 | [PDF](https://arxiv.org/pdf/2602.16375v1)

**作者:** Kirill Khrylchenko `[一作]` `[通讯]` (HSE University), Kirill Khrylchenko (HSE University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出可变长度语义标识符（Variable‑Length Semantic IDs）并用离散VAE配合Gumbel‑Softmax进行训练，旨在让热门物品用更短代码、长尾和冷启动物品用更长代码，从而提升生成式推荐模型的效率与质量。

**💡 创新点**

创新点在于：① 将可变长度编码引入语义ID，突破固定长度限制；② 用概率长度先验和KL正则化形成长度惩罚，避免REINFORCE的梯度方差和训练不稳；③ 将生成式推荐与突出的 emergent communication 思想统一到可变长度离散VAE框架。

**🔧 技术方法**

主要技术包括：离散变分自编码器（dVAE）与Gumbel‑Softmax重参数化、截断几何长度先验、自由位与β‑VAE温度/权重调度、残差编码的软化实现、Transformer 解码器与序列推荐训练。

**📊 数据集**

使用三大真实推荐数据集：Yambda（音乐）、VK‑LSVD（短视频）和 Amazon Toys & Games（电商），均包含上万物品、数千万交互，且提供预训练的物品嵌入。

**📈 对比分析**

与R‑KMeans、固定长度dVAE以及REINFORCE训练的可变长度模型相比，本文方法在 Recall@100 和 Coverage@100 上均有提升（尤其在VK‑LSVD上可提升约11% Recall，Coverage提升约40%），且在相同token预算下可容纳更多历史交互；REINFORCE训练在规模上不稳定，dVAE训练更稳定、收敛更快。

**⚠️ 局限性**

局限性包括：① 需要对物品嵌入做预处理，无法直接应用于无文本特征的冷启动场景；② 最大长度与词表大小仍需经验调优，极端长尾物品可能仍得到过长代码；③ 在动态更新的商品目录中需额外机制来维护长度分配与新物品编码。

---

## 279. Dual-Quadruped Collaborative Transportation in Narrow Environments via Safe Reinforcement Learning

**arXiv ID:** 2602.16353 | [PDF](https://arxiv.org/pdf/2602.16353v1)

**作者:** Zhezhi Lei `[一作]` (Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19888 | [OpenAlex ID](https://openalex.org/A5100357282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于安全强化学习的双足机器人协同运输框架，通过成本-优势分解和约束分配实现安全、高性能的分布式决策；

**💡 创新点**

创新点在于将共享安全约束拆解为个体成本约束并动态分配预算，以成本优势分解与拉格朗日方法联合实现信赖区域更新，从而在协同任务中显著提升安全与效率；

**🔧 技术方法**

技术包括信赖区域强化学习（PPO+KL约束）、成本优势分解、拉格朗日惩罚、贝叶斯优化用于约束预算分配、层级奖励/成本设计，以及双足机器人高低层控制架构；

**📊 数据集**

实验数据来源于仿真环境Isaac Gym（双足机器人协同运输任务）和真实机器人（Go 2平台）在门、走廊、森林三种环境下收集的交互轨迹；

**📈 对比分析**

与MAPPO、HAPPO、UCA、MACPO等基线对比，实验显示本方法在碰撞率、成功率、轨迹直线度和完成时间等指标上均优于基线，达到零碰撞、100%成功率和最高轨迹直线度；

**⚠️ 局限性**

局限性包括仅针对两机器人设置，约束分配仍需手动调参；对更大团队的可扩展性未知；任务奖励/成本设计对不同场景的通用性有限；以及仿真到真实的转移可能仍存在性能下降。

---

## 280. Inductive Satisfiability Certification for Universal Quantifiers and Uninterpreted Function Symbols

**arXiv ID:** 2602.16335 | [PDF](https://arxiv.org/pdf/2602.16335v1)

**作者:** Stefan Ratschan `[一作]` (Institute of Computer Science, Czech Academy of Sciences), Marek Dančo `[通讯]` (Czech Institute of Informatics, Robotics and Cybernetics)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种基于归纳证明的可满足性证书框架，并实现了针对UFLIA子类公式的判定算法，该算法通过在有限区间内求解并向外归纳传播，从而判定存在满足模型的公式。

**💡 创新点**

创新点在于：①提出“可满足性证书”概念，将可满足性转化为归纳证明；②设计了只需在有限区间内求解的区间扩展算法；③提供了对极值子集和传播可行性的判定条件，并给出了可压缩的CNF编码；④通过实验验证该方法可突破现有SMT求解器在此类无穷模型公式上的局限。

**🔧 技术方法**

使用技术包括：线性整数算术与无解释函数的组合理论；可满足性证书与归纳传播器的定义；极值子集与传播可行性约束的Presburger编码；CNF压缩编码；Python实现与Z3 API调用；实验中对比Z3、cvc5（含Sygus模式）等SMT求解器。

**📊 数据集**

数据集：28 个手工构造的可满足问题，包含至多两个常量、1-2个一元无解释函数、简单整数约束及单一全称量词。部分问题还提供了受限范围（0≤x≤10^c）以及无界版本，以评估求解器在不同规模下的表现。

**📈 对比分析**

比较方法：将自研算法与 Z3、cvc5（默认模式）以及 cvc5 的 Sygus 模式直接求解同一套问题进行对比。结果显示：在无界或大范围参数 c 下，现有求解器大多超时或返回 unknown，而自研算法在所有测试问题上几乎即时返回 sat；Sygus 模式在符合语法的函数问题上表现稳定，但对非预定义函数失效。整体而言，归纳基方法在此类公式上显著优于传统 SMT 求解器。

**⚠️ 局限性**

局限性：算法仅适用于满足特定语法约束的 UFLIA 公式（单一量化变量、一元函数、统一系数、可归纳传播条件可满足）；对未满足传播条件的公式无法完成判定；在某些需要双向归纳或更强归纳形式的公式上可能无法终止；实验仅覆盖有限手工构造问题，未证明在更大规模或多元变量场景下的可扩展性。

---

## 281. Spatial Audio Question Answering and Reasoning on Dynamic Source Movements

**arXiv ID:** 2602.16334 | [PDF](https://arxiv.org/pdf/2602.16334v1)

**作者:** Arvind Krishna Sridhar `[一作]` (Qualcomm Technologies Incorporated), Erik Visser `[通讯]` (Qualcomm Technologies Incorporated)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个能够对多通道空间音频进行动态运动推理的端到端多模态问答模型，并设计了针对运动的空间音频数据增强框架、思考模式（Thinking Mode）以及查询条件下的源分离预处理。

**💡 创新点**

创新点主要包括：①运动中心的空间音频数据增强，能合成多样的运动轨迹；②在LLM中引入思考模式，让模型在给出答案前先生成中间推理步骤；③将查询相关源分离（Audio Grounding Model）作为预处理，提升推理的精度和可解释性；④系统性分析了思考模式与源分离的交互效果。

**🔧 技术方法**

使用的技术包括：修改后的BAT空间音频编码器（支持时间感知），Q-Former投影模块，Qwen 3 4B LLM并通过LoRA微调；基于AudioCaps训练的Audio Grounding Model（AGM）进行源分离；多阶段训练策略（基础定位 → 全模型联合训练）和多任务损失；Pyroomacoustics渲染多样化运动场景。

**📊 数据集**

数据集：以AudioSet强标签子集为基础抽取高质量单声道事件，利用Pyroomacoustics合成10秒左右的双声道空间音频并赋予预定义运动轨迹；用LLM自动生成对应的运动推理问答对；AudioCaps用于训练AGM；最终得到包含运动推理任务的空间音频QA数据集。

**📈 对比分析**

对BAT基线模型进行了对比实验，设置三种源分离方案：无掩码（NoMask）、AGM掩码、真值掩码（GT）。在Yes/No、Multiple Choice、Open三种问题类型上，整体准确率分别为54.3%、55.0%和56.1%。思考模式在掩码存在时能提升2–5个百分点，尤其在GT掩码下提升约+2.0%；Δ Interaction指标显示思考模式对源分离的依赖性强。实验表明源分离与推理相结合显著提升空间音频理解性能。

**⚠️ 局限性**

局限性包括：①仅使用时间域掩码，无法完全抑制重叠事件导致的噪声；②在多源重叠场景下性能显著下降；③推理模式的延时较大（≈6.2s vs 2.4s）；④仅在双声道、前方监听、10s短时段的合成数据上验证，未覆盖复杂真实环境与多源长音频。

---

## 282. Case Study: Saturations as Explicit Models in Equational Theories

**arXiv ID:** 2602.16324 | [PDF](https://arxiv.org/pdf/2602.16324v1)

**作者:** Mikoláš Janota `[一作]` (Czech Technical University in Prague), Stephan Schulz `[通讯]` (DHBW Stuttgart)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了如何从单元等式片段中的饱和集构造显式（可能无限）的模型，并在等价理论项目中实现并验证了该方法。

**💡 创新点**

创新点在于将ATP饱和集直接视为收敛重写系统，从而得到显式模型证书，并实现对预排序方程的自动连贯性和终止性验证。

**🔧 技术方法**

采用无故障完成/超位置推理、归一化函数构造、改造E求解器以输出重写系统，并使用Isabelle等工具验证连贯性与终止性。

**📊 数据集**

实验使用Equational Theories Project的22028942个蕴含（约4.7k个等式）作为数据集，重点处理其中304个饱和问题。

**📈 对比分析**

将该方法与有限模型构造器FMB及Infinox比较，261个预排序重写系统通过工具验证连贯性与终止性，证明可为所有非定理生成可验证的无限反例。

**⚠️ 局限性**

局限性在于仅适用于单元等式片段，未能自动验证非预排序方程的连贯性与终止性，也无法直接构造有限模型。

---

## 283. The Diversity Paradox revisited: Systemic Effects of Feedback Loops in Recommender Systems

**arXiv ID:** 2602.16315 | [PDF](https://arxiv.org/pdf/2602.16315v1)

**作者:** Gabriele Barlacchi `[一作]` (Scuola Normale Superiore and Università di Pisa), Luca Pappalardo `[通讯]` (CNR and Scuola Normale Superiore)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个可模拟隐式反馈、周期性重新训练、概率化采纳以及多种推荐算法的反馈循环模型，并在两大真实数据集上进行长期仿真，系统性评估了推荐器对个体与集体多样性、集中度和用户相似度的影响。

**💡 创新点**

创新点在于：①将隐式反馈、周期性训练与采纳概率融入单一模型，克服传统实验的理想化假设；②通过长时序仿真揭示“多样性悖论”主要源于静态横截面评估，而非反馈循环本身；③提供了统一的实验框架，可对不同算法和采纳率进行比较。

**🔧 技术方法**

主要技术包括：基于 RecBole 的多种协同过滤与混合算法（ItemKNN、BPR、NeuMF、NGCF、SpectralCF、SGL、DeepFM、DCN-V2）；概率化采纳机制；周期性重新训练；Gini 系数、用户相似度（Jaccard）等度量；agent‑based 反馈循环仿真框架。

**📊 数据集**

使用两大公开隐式反馈数据集：Amazon 1.0（电商购买记录）和 Last.fm 1K（音乐聆听记录），分别涵盖数千用户、数万物品，并保留时间连续性。

**📈 对比分析**

比较方法：在不同采纳率（η=0,0.2,0.5,0.8,1）下，对所有推荐算法进行 5 次重复仿真，评估个体 Gini、集体 Gini、用户相似度等指标；与基准（无采纳）对比并分析时间演化。性能方面：在静态端点时，高采纳往往提升个体多样性但降低集体多样性；但随时间推移，所有算法均导致个体多样性衰减，集体多样性下降且表现因域与算法而异。

**⚠️ 局限性**

局限性：仅基于两种数据集且为离散日步仿真；未覆盖多模态或内容深度特征；采纳概率和训练窗口的设定为经验值；未对干预措施进行实验，难以给出具体优化方案。

---

## 284. TabAgent: A Framework for Replacing Agentic Generative Components with Tabular-Textual Classifiers

**arXiv ID:** 2602.16429 | [PDF](https://arxiv.org/pdf/2602.16429v1)

**作者:** Ido Levy `[一作]` (IBM Research), Segev Shlomov `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TabAgent 框架，将 agentic 系统中多次调用的 LLM 生成式决策组件替换为基于执行轨迹的文本‑表格分类器，实现单次推理；

**💡 创新点**

创新点在于：① 自动从执行轨迹中提炼 schema、state、dependency 三类结构化特征（TabSchema）；② 通过 schema 对齐的合成数据扩充（TabSynth）提升稀有模式覆盖；③ 用轻量化的文本‑表格模型 TabHead 替代昂贵的 LLM，显著降低 latency 与成本；

**🔧 技术方法**

核心技术包括：多阶段 LLM 驱动的特征工程、结构化表格特征抽取、schema‑aligned 生成式数据增强、文本‑表格深度学习分类器（约 50M 参数），以及自热插拔部署机制；

**📊 数据集**

主要数据集为 IBM CUGA 在 AppWorld benchmark 上的 605 条成功轨迹，涵盖 5 个跨应用任务（Amazon、Gmail、Phone、SimpleNote、Spotify），并使用生成的合成样本扩充训练；

**📈 对比分析**

与传统检索（BM25、Dense Semantic Retrieval）和 LLM 控制器（Llama 1B/3B/8B、GPT‑4.1）对比，TabAgent 在 Recall@7 ≥ 0.88、Recall@9 ≥ 0.92、P@R 提升平均 +0.14 的同时，latency 下降 ~95%，成本下降 85–91%，表现优于现有最佳方法；

**⚠️ 局限性**

局限性包括：依赖足够丰富且成功的轨迹日志；合成数据可能引入偏差；对极少见或全新工具组合的泛化能力有限；需要领域专家参与特征设计；目前仅验证于 AppWorld，尚需在更多场景中进一步评估。

---

## 285. Verifiable Semantics for Agent-to-Agent Communication

**arXiv ID:** 2602.16424 | [PDF](https://arxiv.org/pdf/2602.16424v1)

**作者:** Philipp Schoenegger `[一作]` (Microsoft), Chris Daly `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于刺激-意义模型的语义对齐协议，通过在公共可观测事件上对齐代理对词条的判定来认证共享词表，并在决策中仅使用已认证的核心词表；

**💡 创新点**

将语义一致性验证从自然语言转移到可观测事件的行为一致性，利用统计置信区间（Wilson上界）实现可验证且可复制的语义对齐；

**🔧 技术方法**

采用公共账本记录判定、Wilson置信区间上界、核心保护推理、递归认证与重新谈判机制；

**📊 数据集**

在合成语义偏差的模拟环境（6词表）和对Qwen2.5-3B-Instruct微调的内容审核任务（6词表）上进行实验；

**📈 对比分析**

与未受限使用所有词表的基线相比，核心保护推理在模拟中将争议率从最大40%降低至约2%，在LLM验证中将争议率从5.3%降至2.6%，证明显著的误差约束；

**⚠️ 局限性**

局限于单词级别、仅支持二元代理、重新谈判机制仍为草案、假设代理诚实行事，并未覆盖组合表达与上下文依赖语义。

---

## 286. Parameter-Free Adaptive Multi-Scale Channel-Spatial Attention Aggregation framework for 3D Indoor Semantic Scene Completion Toward Assisting Visually Impaired

**arXiv ID:** 2602.16385 | [PDF](https://arxiv.org/pdf/2602.16385v1)

**作者:** Qi He `[一作]` (University of Electronic Science and Technology of China), Zhenglin Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 17984 | [OpenAlex ID](https://openalex.org/A5080583583)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了 AMAA 框架，用于单目 3D 语义场景补全，提升视觉障碍辅助系统的空间感知质量。

**💡 创新点**

创新点在于可靠性导向的体素特征调制（3D SEBlock + SimAM）与层级自适应注意力聚合（AFG），解决投影不确定和多尺度信息污染。

**🔧 技术方法**

采用 3D SEBlock、SimAM、Adaptive Feature Gating、MonoScene 主干、PyTorch、TensorRT 等技术实现端到端推理。

**📊 数据集**

使用 NYU‑Depth v2 数据集进行训练与评估。

**📈 对比分析**

与 MonoScene、LMSCNet、AICNet、3DSketch 等基线对比，SC IoU 提升至 43.10%，SSC mIoU 提升至 27.25%，并在 NVIDIA Jetson Orin NX 上实现实时推理。

**⚠️ 局限性**

仍受单目深度不确定、纹理匮乏、薄壁结构等限制，难以完全消除投影噪声，导致在极端场景下的误判或保守补全。

---

## 287. Improved Bounds for Reward-Agnostic and Reward-Free Exploration

**arXiv ID:** 2602.16363 | [PDF](https://arxiv.org/pdf/2602.16363v1)

**作者:** Oran Ridel `[一作]` (Tel Aviv University), Alon Cohen `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了无奖励（reward‑free）和奖励无关（reward‑agnostic）探索在有限时限马尔可夫决策过程中的样本复杂度，并给出了对应的上界与下界。

**💡 创新点**

创新点在于：① 用单一在线MDP算法（基于在线镜像下降）取代以往多次独立无偏算法，显著降低低阶项；② 对时间非齐次MDP的reward‑free探索给出紧致下界，解决了已存在的上界与下界差距。

**🔧 技术方法**

主要技术包括：在线镜像下降（OMD）框架、Bernstein型置信区间、基于占用度量的凸优化、惰性（pessimistic）模型基离线强化学习与采样估计。

**📊 数据集**

实验与验证基于理论构造的MDP实例（如多状态树结构），并未使用公开数据集，侧重理论分析与模拟。

**📈 对比分析**

与之前的O(H³|S|²|A|/ε²)等算法相比，新算法在低阶项上大幅改进，尤其在中等到大ε范围内达到更低样本复杂度，理论上逼近下界。

**⚠️ 局限性**

局限性包括：在极低精度（ε极小）时仍存在与下界的差距；算法依赖于时间非齐次MDP的假设，实战中需进一步验证；实现细节对超参数选择敏感。

---

## 288. Helpful to a Fault: Measuring Illicit Assistance in Multi-Turn, Multilingual LLM Agents

**arXiv ID:** 2602.16346 | [PDF](https://arxiv.org/pdf/2602.16346v1)

**作者:** Nivya Talokar `[一作]` (Independent Researcher), Antoine Bosselut `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 5289 | [OpenAlex ID](https://openalex.org/A5088410008)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了STING框架，对LLM代理在多轮、跨语言场景下的非法协助行为进行自动红队评估。

**💡 创新点**

创新点在于将多轮红队与分阶段攻击拆解相结合，并将红队过程视为有限预算的时间到首个“越狱”随机变量，提出了受限平均越狱发现（RMJD）指标与生存分析方法。

**🔧 技术方法**

采用四类协作代理（策略师、攻击者、拒绝检测器、阶段完成检查器）进行自动化对话；使用多轮提示、工具调用、语言转换以及基于Cox模型的风险比分析。

**📊 数据集**

利用AgentHarm公共测试集（44个危害场景共176个提示）以及在七种非英语语言（中文、法语、乌克兰语、印地语、乌尔都语、泰卢固语）中的合成工具接口。

**📈 对比分析**

与单轮提示和X-Teaming基线对比，STING在Qwen3-Next、Claude Sonnet 4.5、DeepSeek-V3.2等模型上实现了约107%–101%提升的AgentHarm得分；多轮评估揭示非英语场景并未像聊天机器人那样显著提升越狱成功率。

**⚠️ 局限性**

局限性包括：评估依赖自动判别器（可能欠缺精度）、仅测试有限模型与工具集、跨语言性能受工具兼容性影响、并未探索更强攻击者与防御方案的深度交互。

---

## 289. How to Label Resynthesized Audio: The Dual Role of Neural Audio Codecs in Audio Deepfake Detection

**arXiv ID:** 2602.16343 | [PDF](https://arxiv.org/pdf/2602.16343v1)

**作者:** Yixuan Xiao `[一作]` (University of Stuttgart), Ngoc Thang Vu `[通讯]` (University of Stuttgart)

**通讯引用:** 2737 | [OpenAlex ID](https://openalex.org/A5020700841)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了新的 CodecDeepfakeDetection（CDD）数据集，并探究了神经音频编码器（NAC）在音频深度伪造检测中的双重作用；

**💡 创新点**

首次系统评估了将 codec 重合成（CoRS）样本标记为真实或伪造对检测性能的影响，并揭示了不同 NAC 设计目标对检测策略的关键影响；

**🔧 技术方法**

使用 XLS‑R 前端结合 AASIST 与 LWBN 两种后端模型，配合多种 NAC（EnCodec、Mimi、DAC、XCodec2 等）进行数据增强和实验；

**📊 数据集**

利用基于 ASVspoof‑5 贡献协议扩展的 CDD 数据集，包含 3.6k/1.2k/1.2k 的重合成样本与 1.2k 唯一的真实语音，覆盖多种 TTS 与 NAC 组合；

**📈 对比分析**

对比在 ASVspoof‑5 与 CDD 上训练的模型，发现仅在 CDD 上训练的检测器在 T‑CoSG 上表现略有提升，但整体 EER 仍高达 20%–25%；对 CoRS 作为真实或伪造标记的实验表明，压缩导向 NAC 需以伪造标记为主，而合成导向 NAC 则更适合以真实标记；

**⚠️ 局限性**

主要局限在于：①缺乏对多样化攻击方式（如多模型融合、后处理等）的评估；②实验仅覆盖有限数量的 NAC 与 TTS 组合，未能充分泛化到未知或更高级的编码器；③当前模型对相同 NAC 用于传输与生成时的区分仍不够精准，导致误判率偏高。

---

## 290. ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding

**arXiv ID:** 2602.16412 | [PDF](https://arxiv.org/pdf/2602.16412v1)

**作者:** Daichi Yashima `[一作]` (Keio University), Komei Sugiura `[通讯]` (Keio University)

**通讯引用:** 1835 | [OpenAlex ID](https://openalex.org/A5033744547)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出ReMoRa，一种在压缩视频流上直接处理I帧和运动矢量的多模态大语言模型，用以实现可扩展的长视频理解。

**💡 创新点**

创新点在于引入Refined Motion Representation模块将块级运动矢量去噪并细化为高精度运动表示，以及Hierarchical Motion State Space模块实现线性时间的长程时间建模。

**🔧 技术方法**

使用了压缩域运动矢量、块级去噪网络、状态空间模型（SSM）、Mamba块、预训练LLM Qwen2和SigLIP ViT‑SO等技术。

**📊 数据集**

在200K指令调优集上训练，评测使用LongVideoBench、NExT‑QA、MLVU、VideoMME、PerceptionTest、MSVD‑QA、ActivityNet‑QA等长视频理解与视频QA数据集。

**📈 对比分析**

与多种基线（LLaVA‑OneVision、Qwen2‑VL、Video‑LaVIT、EMA等）对比，ReMoRa在LongVideoBench、NExT‑QA、MLVU等指标上平均提升约1–2点，整体平均分69.8高于基线0.9点。

**⚠️ 局限性**

局限在于仍依赖对压缩流的关键帧检测与运动矢量的准确性，对极低帧率或高运动复杂度的视频效果可能受限。

---

## 291. Guide-Guard: Off-Target Predicting in CRISPR Applications

**arXiv ID:** 2602.16327 | [PDF](https://arxiv.org/pdf/2602.16327v1)

**作者:** Joseph Bingham `[一作]` (Rutgers University), Saman Zonouz `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了Guide-Guard，一种基于CNN的模型，用于预测CRISPR-Cas13引导RNA的离靶行为，整体准确率约为84%。

**💡 创新点**

创新点在于结合位点权重和核苷酸类型的加权One‑hot编码、zip拼接以及八分类输出，显著提升了离靶预测的准确性。

**🔧 技术方法**

采用卷积神经网络（CNN）、One‑hot编码、加权处理、交叉熵损失、Adam优化器等深度学习技术。

**📊 数据集**

使用了公开的Cas13 RNA引导数据集，包括CD46、CD55、CD71三种转录本约5,000条带有不匹配的引导序列。

**📈 对比分析**

通过20折交叉验证评估，ROC AUC为0.839，整体准确率84%，并且比仅考虑完全匹配的现有方法表现更优。

**⚠️ 局限性**

局限性包括对其他CRISPR系统的泛化能力有限、依赖已标注的激活能数据、离靶匹配的准确率低于完全匹配且需要大量标注数据。

---

## 292. RoboGene: Boosting VLA Pre-training via Diversity-Driven Agentic Framework for Real-World Task Generation

**arXiv ID:** 2602.16444 | [PDF](https://arxiv.org/pdf/2602.16444v1)

**作者:** Yixue Zhang `[一作]` (Peking University), Jian Tang `[通讯]` (Peking University)

**通讯引用:** 15408 | [OpenAlex ID](https://openalex.org/A5039176528)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了RoboGene框架，自动生成多样且物理可执行的机器人操作任务，支持单臂、双臂和移动机器人。

**💡 创新点**

将LFU多样性采样、自我反思评估和长时记忆人机交互相结合，形成闭环生成流程，有效提升任务质量与多样性，显著抑制LLM幻觉。

**🔧 技术方法**

利用LLM生成任务描述，三重评估器（约束、创新、物理可行性）进行自我反思，LFU采样机制优化多样性，记忆模块整合HITL反馈，实现闭环生成。

**📊 数据集**

通过RoboGene生成18k条轨迹（1200个任务，每个任务15个演示），并使用该数据集评估VLA模型；同时与RoboMIND、AgiBot World、RoboCoin等公开数据进行对比。

**📈 对比分析**

与人类专家、规则方法以及GPT‑4o、Gemini‑2.5‑Pro在900个任务上对比，RoboGene在任务可执行性、多样性覆盖以及VLA预训练零射击泛化性能上均显著优于所有基线。

**⚠️ 局限性**

生成任务仍需人工审核，且对LLM对物理约束的理解存在依赖；实验仅覆盖单臂、双臂和移动机器人，对其他硬件平台的适应性尚未验证。

---

## 293. Intra-Fairness Dynamics: The Bias Spillover Effect in Targeted LLM Alignment

**arXiv ID:** 2602.16438 | [PDF](https://arxiv.org/pdf/2602.16438v1)

**作者:** Eva Paraschou `[一作]` (Technical University of Denmark), Sneha Das `[通讯]` (Technical University of Denmark)

**通讯引用:** 145 | [OpenAlex ID](https://openalex.org/A5035293526)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了针对性别对齐如何影响三大 LLM 在九个敏感属性上的公平性，并系统评估了在不同上下文（模棱两可 vs 明确）下的偏差溢出现象。

**💡 创新点**

首次量化并揭示了 LLM 在聚合评估下掩盖的偏差溢出，提出了基于 BBQ、DPO 与 PEFT 的多属性公平评估流程。

**🔧 技术方法**

使用 Direct Preference Optimization（DPO）配合 Parameter‑Efficient Fine‑Tuning（PEFT，QLoRA）对 Mistral‑7B、Llama‑3.1‑8B 与 Qwen‑2.5‑7B 进行性别对齐。

**📊 数据集**

采用 BBQ 基准数据集（共 58,492 条多选题，覆盖九个敏感属性）。

**📈 对比分析**

通过预对齐与后对齐的准确率比较并使用 McNemar 检验验证显著性；总体上准确率提升，但在模棱两可情境下出现显著退化，尤其是外貌、性取向和残疾属性。

**⚠️ 局限性**

仅评估单一对齐目标，未探讨多属性对齐；训练与评估使用同一格式，难以区分内容与格式偏差；未评估不同对齐算法或数据集对偏差溢出的影响。

---

## 294. Bibby AI -- AI Latex Editor writing assistant for researchers vs Overleaf Alternative vs OpenAI Prism. (Bibby AI Latex Editor)

**arXiv ID:** 2602.16432 | [PDF](https://arxiv.org/pdf/2602.16432v1)

**作者:** Nilesh jain `[一作]`, Andrej Karpathy `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Bibby AI，一款原生集成八大 AI 功能的 LaTeX 编辑器，实现学术写作全流程（写作、引用、表格/公式生成、审稿评估、摘要/综述生成、深度检索、错误检测与一键修复）在同一界面完成。

**💡 创新点**

创新点在于：①从架构层面将 AI 嵌入编辑器核心，利用实时 AST 与增量解析消除 DOM‑scrape fragility；②通过三信号融合（编译日志、AST、包数据库）并结合 Gemini 2.5 Pro 的领域微调，实现精准定位与可验证的错误修复；③构建了首个真实错误基准 LaTeXBench‑500，涵盖 500 条真实错误，细分六大错误类别。

**🔧 技术方法**

技术上：使用 CodeMirror 6 + 语法树；Gemini 2.5 Pro（Fine‑tuned on LaTeX + 学术写作）做所有 AI 任务；三信号融合 + AST 验证；多源数据层（Semantic Scholar、CrossRef、arXiv、BibTeX Store）；零训练隐私层；前端与后端分层架构。

**📊 数据集**

数据集：LaTeXBench‑500（500 条来自 TeX.StackExchange、实际会议稿、学生稿的真实错误）；文献检索集成 Semantic Scholar、CrossRef、arXiv 等公开数据库。

**📈 对比分析**

比较方法：与 Overleaf 原生诊断（仅显示编译日志）和 OpenAI Prism（对日志做通用 LLM 处理）做对比；指标为检测准确率（DA）与修复准确率（FA）。Bibby 在所有六类错误上均超越对手：DA 91.4%（Overleaf 61.2%、Prism 78.3%），FA 83.7%（Overleaf 无、Prism 64.1%）。

**⚠️ 局限性**

局限性：仅针对 LaTeX；依赖 Gemini 2.5 Pro 与外部 API（若无网络或 API 限制会受影响）；对罕见错误的泛化能力尚未完全验证；目前只在实验环境中测试，真实学术团队使用仍需进一步评估；AI 生成内容的可解释性与学术伦理仍需细化。

---

## 295. From Latent to Observable Position-Based Click Models in Carousel Interfaces

**arXiv ID:** 2602.16541 | [PDF](https://arxiv.org/pdf/2602.16541v1)

**作者:** Santiago de Leon-Martinez `[一作]` (Brno University of Technology), Maria Bielikova `[通讯]` (Kempelen Institute of Intelligent Technologies)

**通讯引用:** 3386 | [OpenAlex ID](https://openalex.org/A5030414237)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对多列表旋转式界面，提出并实现了三种基于位置的点击模型。

**💡 创新点**

创新点在于首次将眼动观测直接用于定位检查变量，并将PBM适配到旋转式界面。

**🔧 技术方法**

使用梯度优化、EM、MLE等算法训练PBM，并利用眼动数据构建观察检查似然。

**📊 数据集**

实验采用RecGaze数据集，包含点击、眼动、位置信息。

**📈 对比分析**

通过与EM、MLE以及经典CM/TCM/CCM对比，梯度方法表现最佳，OEPBM在点击预测和检查一致性上取得最高分。

**⚠️ 局限性**

局限性在于数据规模小、点击稀疏、仅使用单一工业数据集，且缺少更大范围的用户行为数据。

---

## 296. Framework of Thoughts: A Foundation Framework for Dynamic and Optimized Reasoning based on Chains, Trees, and Graphs

**arXiv ID:** 2602.16512 | [PDF](https://arxiv.org/pdf/2602.16512v1)

**作者:** Felix Fricke `[一作]` (Technical University of Munich), Georg Groh `[通讯]` (Technical University of Munich)

**通讯引用:** 6100 | [OpenAlex ID](https://openalex.org/A5004398345)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Framework of Thoughts (FoT) 框架，支持动态推理图、并行执行、缓存以及自动化的超参数和提示优化，并在该框架中实现并评估了 Tree of Thoughts、Graph of Thoughts 与 ProbTree 三种推理方案。

**💡 创新点**

核心创新点在于：①构建通用的动态推理框架，可在执行期间自适应生成和演化推理图；②引入安全并行执行约束与缓存机制，使运行时显著加速、成本显著降低；③提供基于 Optuna 和 DSPy 的自动化优化工具，让推理方案能充分挖掘性能潜力。

**🔧 技术方法**

技术细节包括：基于操作（operation）与执行图的结构化模型；安全并行执行的父子/祖先/后代约束；持久化与过程缓存；Optuna 超参数优化；DSPy 提示优化；使用 GPT‑4o、GPT‑3.5‑Turbo、GPT‑4.1‑mini 等 LLM 及外部工具调用。

**📊 数据集**

采用的评测数据集有：Go24（算术 24 题）、Sorting（128 个整数排序）、Document Merging (DM，文档合并）、HotpotQA 与 MuSiQue（多跳问答）。

**📈 对比分析**

通过与原始实现对比，添加并行化、缓存与自动化优化后，平均运行时间提升约 10.7 倍（范围 1.9–35.4 倍），成本下降 14–46%；任务得分亦有提升，例如 Go24 从 63.0% 提升至 66.0%，Sorting 错误数下降，DM F1 分数提高，HotpotQA 与 MuSiQue 在成本上保持不变。

**⚠️ 局限性**

目前实现仅涵盖手动（GoT）和半自动（ToT、ProbTree）方案，缺乏完整自动化的动态推理实现；需要更友好的图抽象与接口、更广泛的任务验证；尽管 FoT 大幅降低了运行与优化成本，但优化过程本身仍耗时和费用高，仍需进一步提升效率。

---

## 297. DressWild: Feed-Forward Pose-Agnostic Garment Sewing Pattern Generation from In-the-Wild Images

**arXiv ID:** 2602.16502 | [PDF](https://arxiv.org/pdf/2602.16502v1)

**作者:** Zeng Tao `[一作]` (UCLA and Fudan University), Chenfanfu Jiang `[通讯]` (UCLA)

**通讯引用:** 5317 | [OpenAlex ID](https://openalex.org/A5068163735)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个名为 DressWild 的端到端快速推理框架，能够从单张在野外拍摄的服装图片中恢复可编辑、可分离且符合物理仿真的二维缝制图样及其对应的三维服装模型。

**💡 创新点**

创新点包括：①利用视觉‑语言模型(VLM)生成固定 T‑姿态的正向图像，实现姿态和视角的先验归一化；②构建三条互补特征流（原图像特征、T‑姿态重建特征、姿态编码特征），并通过混合注意力的 Transformer 进行融合；③在参数解码阶段采用自回归 Transformer 直接预测缝制图样的顶点、曲线控制点、平移旋转以及缝合关系，兼顾离散与连续输出；④通过数据生成与增强将传统 T‑姿态服装数据扩展到多姿态、多视角的野外场景，显著提升了模型的泛化能力。

**🔧 技术方法**

核心技术包括：视觉‑语言模型（如 NanoBanana Pro）用于姿态归一化；Hunyuan3D‑2.0 提取 3D 结构特征；HybridGL 与 SAM3D‑Body 进行服装与人体分割与姿态编码；Transformer 编码器/解码器实现特征融合与参数预测；高斯似然与交叉熵损失实现离散+连续参数的联合训练；以及后端的物理仿真（PBD、CIPC）与纹理生成（Hunyuan3D‑Paint）。

**📊 数据集**

主要使用的服装图样数据集为 19 种基础服装类型的 20,000+ 变体（含 12 类，25,031 条样本）。通过 VLM 生成多姿态、多视角的图像来扩充训练集，并结合公开的 GarmentCodeData、SewFactory 等图样资源进行训练和评估。

**📈 对比分析**

与现有两种基线（NeuralTailor 和 SewFormer）进行定量对比。DressWild 在 Panel Accuracy、Edge Accuracy、形状误差、F‑Shape 误差和 Chamfer Distance 等指标上均明显优于两者：Panel Accuracy 94.35% / 85.41%（相较 25.99% / 29.05% 和 28.81% / 34.56%），形状误差从 23.65/22.94 降至 6.22，CD 下降至 0.01899，验证了模型在多姿态、野外图像下的强大鲁棒性和精度。

**⚠️ 局限性**

主要局限包括：①对 VLM 生成的正向图像质量高度依赖，若 VLM 产生的 T‑姿态图像不够准确，后续特征提取会受影响；②模型对极端姿势或复杂多层服装（如大量折叠、重叠的层次）仍可能出现误差；③虽然推理速度快，但整体训练仍需大规模 GPU 资源；④缺乏对真实布料属性（如面料弹性、摩擦）和材质变化的显式建模，导致纹理与物理仿真之间可能存在一定偏差。

---

## 298. Benchmarking Adversarial Robustness and Adversarial Training Strategies for Object Detection

**arXiv ID:** 2602.16494 | [PDF](https://arxiv.org/pdf/2602.16494v1)

**作者:** Alexis Winter `[一作]` (Universite Paris-Saclay), Bertrand Luvison `[通讯]` (Universite Paris-Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个统一的数字非补丁攻击基准框架，用来公平比较目标检测器的对抗攻击与防御方法，并在此框架下评估了多种先进攻击与对抗训练策略。

**💡 创新点**

创新点在于①提出专门针对定位与分类错误的 AP_loc 与 CSR 两个新指标；②统一了攻击评估的度量（L2、L∞、LPIPS 等感知指标）和数据集；③系统性揭示了攻击跨架构（CNN→Transformer）的转移性缺口；④实验证明混合高扰动攻击（如 OSFD+EBAD）能获得最强防御效果。

**🔧 技术方法**

技术包括对抗攻击算法（OSFD、EBAD、CAA、Phantom Sponges 等）、对抗训练（混合攻击样本）、多模型评估（YOLOv3、Faster R‑CNN、FCOS、DETR、DINO 等），以及使用 LPIPS、SSIM 等感知指标衡量扰动可感知性。

**📊 数据集**

使用 COCO 训练集、VOC 2007 测试集（以及部分 COCO‑person、DOTA 等子集）作为实验数据集。

**📈 对比分析**

对比方法：在同一数据集、相同评估指标（mAP、AP_loc、CSR、LPIPS 等）下对 7 种攻击进行白盒与黑盒测试。结果显示 OSFD 在所有 CNN 模型上导致 mAP 降至 10‑20%，但对 DINO 的转移性极差；EBAD 与 CAA 对分类错误影响显著；Phantom Sponges 主要制造伪目标。对抗训练实验表明 100% 对抗样本训练优于混合训练，混合 OSFD+EBAD 能在保持 70% 以上 mAP 的同时显著提升鲁棒性。

**⚠️ 局限性**

局限性包括：①基准仅涵盖数字非补丁攻击，未覆盖物理/补丁攻击；②对 Transformer 目标检测器的攻击仍缺乏有效方法；③对抗训练的域泛化能力（跨数据集、跨环境）未充分验证；④计算成本高（如 OSFD 需要 44 s/图）限制了实际部署。

---

## 299. Team of Thoughts: Efficient Test-time Scaling of Agentic Systems through Orchestrated Tool Calling

**arXiv ID:** 2602.16485 | [PDF](https://arxiv.org/pdf/2602.16485v1)

**作者:** Jeffrey T. H. Wong `[一作]` (Imperial College London), Yiren Zhao `[通讯]` (Imperial College London)

**通讯引用:** 13274 | [OpenAlex ID](https://openalex.org/A5076778501)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Team-of-Thoughts多智能体系统，利用异构模型通过工具调用实现协同推理与代码生成；

**💡 创新点**

创新点在于（1）引入主控器-工具范式，动态激活最佳工具；（2）制定主控器校准方案选出最优协调模型；（3）实现工具自评机制，构建专家化专业化档案；

**🔧 技术方法**

采用大型语言模型（Claude‑Sonnet‑4.5、GPT‑5‑Mini、Gemini‑3‑Flash‑Preview 等）、工具调用接口、主控器校准与自评流程、并行推理与战略令牌分配；

**📊 数据集**

使用数学推理基准 AIME2024/2025、代码生成基准 Humaneval+、MBPP+、LiveCodeBench 等公开数据集；

**📈 对比分析**

与单模型推理、传统协同推理（AgentVerse、Majority Voting）对比，Team‑of‑Thoughts 在 AIME24、LiveCodeBench 等任务上分别达到 96.67% 与 72.53% 的准确率，显著优于同类方法且成本更低；

**⚠️ 局限性**

局限性包括：主控器与工具模型需先行校准与自评，依赖大规模计算与昂贵模型；对任务类型识别与自评质量不够稳健时可能导致不当调用；系统在高度动态或实时环境下的扩展性和可解释性仍待提升。

---

## 300. Reinforcement Learning for Parameterized Quantum State Preparation: A Comparative Study

**arXiv ID:** 2602.16523 | [PDF](https://arxiv.org/pdf/2602.16523v1)

**作者:** Gerhard Stenzel `[一作]` (Ludwig Maximilian University of Munich), Claudia Linnhoff-Popien `[通讯]` (Ludwig Maximilian University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

未提供论文内容

**💡 创新点**

无法确定

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未给出比较方法或性能数据

**⚠️ 局限性**

无法评估

---

## 301. Synthesis and Verification of Transformer Programs

**arXiv ID:** 2602.16473 | [PDF](https://arxiv.org/pdf/2602.16473v1)

**作者:** Hongjian Jiang `[一作]` (University of Kaiserslautern-Landau), Anthony Widjaja Lin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了针对 C-RASP（Counting RASP）语言的自动验证与合成框架，首先将 C‑RASP 程序转换为 Lustre 并使用 Kind2+SMT 进行模型检查；随后设计了一种基于模拟退火的本地搜索算法，在给定正负样本的约束下学习 C‑RASP 程序。

**💡 创新点**

创新点包括：① 将 C‑RASP 与 Lustre 之间建立多项式时间的翻译，利用成熟的 Lustre 检查器完成语言等价、包含、空性等属性的自动验证；② 在合成层面提出了受结构约束的模拟退火搜索策略，兼顾误分类、不可达子句与程序尺寸，显著提升了学习效率和可解释性。

**🔧 技术方法**

主要技术手段包括：Lustre 编译与 Kind2 SMT 迁移；模拟退火（SA）加本地语法变异；基于样本的误分类惩罚与结构惩罚；约束学习框架将部分规范与样本联合使用。

**📊 数据集**

使用了多种基准数据集，涵盖正规、计数式与上下文无关式语言：Dyck-1、AStarBStar、AnBnCn、AAStar、ContainsAB、Majority、Existential、PT-2/3/5、D_2/3/4、Tomita 1-7、Next(Argmax)。每个任务均采用 1000 条平衡样本，字符串长度在 [ℓ_min, 100] 之间。

**📈 对比分析**

实验中，合成与验证在 300 秒超时内大多数基准实现了 100% 正确率；合成时间普遍在 1–80 秒，验证时间在 1–90 秒之间；相比之下，使用 HuggingFace 训练 GPT‑2 的 Transformer 在同一基准上需数天。表格中显示的 Refinement R、Synth、Verif 指标进一步证明了验证驱动合成在减少冗余、提高效率方面的优势。

**⚠️ 局限性**

局限性：① C‑RASP 无法表达某些语言（如 Parity、PT‑5、Tomita‑3/5/6/7），导致在这些基准上无法在时限内合成；② 对于不可判定属性（如空性）仅提供了间接检查；③ 当前实现仅支持单词级别的 C‑RASP，未扩展至更复杂的变体；④ 需要手工设定形状参数、温度退火曲线等超参数。

---

## 302. RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion

**arXiv ID:** 2602.16548 | [PDF](https://arxiv.org/pdf/2602.16548v1)

**作者:** Tianmeng Hu `[一作]`, Ke Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一个两阶段的RNA三维逆向设计框架：先用条件扩散模型预训练得到可根据目标3D骨架生成RNA序列的生成器，再通过强化学习微调该生成器，使其直接优化生成序列折叠后的结构与目标结构的相似度。

**💡 创新点**

创新点包括：①首次在RNA 3D逆向设计中直接优化结构相似度，而非传统的本地序列回收；②将GVP‑GNN作为结构编码器与扩散模型相结合，捕捉核苷酸的空间几何关系；③引入改进的强化学习（DDPO + 移动平均基线、剪枝）并设计四种基于GDT_TS、TM‑score、RMSD的奖励函数；④在实验中显著提升结构相似度与设计成功率。

**🔧 技术方法**

主要技术：GVP‑GNN结构编码器；条件扩散模型（variance‑preserving + DDIM采样）；强化学习（policy‑gradient、DDPO、优势函数、移动平均基线、剪枝）。

**📊 数据集**

使用RNASolo RNA 3D结构数据集（4,223条序列，12,011个结构），按结构相似度划分为训练/验证/测试集（测试集100条）。另外通过AlphaFold3对生成结构进行交叉验证。

**📈 对比分析**

与SOTA方法RiboDiffusion和gRNAde比较：预训练阶段NSR 61%（比RiboDiffusion提升9%）；强化学习后在三维自一致性指标（GDT_TS、TM‑score、RMSD）上超过100%提升，GDT_TS>0.5的设计比例从27%提升到72%，RMSD≤2Å的比例从3%提升到33%。在AlphaFold3 oracle上亦保持优势。

**⚠️ 局限性**

局限性：①评估依赖于结构预测器（RhoFold/AlphaFold3），预测误差会影响奖励与评估；②奖励函数仅考虑几何相似度，未涵盖功能、稳定性或免疫原性等设计目标；③强化学习需要大量采样，计算成本高；④尚未进行实验验证，无法确认设计序列的实际折叠与功能。

---

## 303. Spectral Conditions for the Ingleton Inequality

**arXiv ID:** 2602.16536 | [PDF](https://arxiv.org/pdf/2602.16536v1)

**作者:** Rostislav Matveev `[一作]`, Andrei Romashchenko `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个新的方法，利用谱性质（尤其是二次谱间距）来证明对一类在支撑图为二部正则图的随机变量对（X,Y），即使互信息不可提取，Ingleton不等式仍在一个小的加性误差范围内成立；该方法给出了 Ingleton量的下界，取决于图的最大与次大特征值；并给出多种具体实例（如有限射影平面、曲线图、多项式图、k‑l 旗子等），展示该结论的适用性。

**💡 创新点**

创新点在于将Ingleton不等式的验证与图的谱间距（尤其是 λ1/λ2 的比值）关联，证明了即使互信息远非可提取，仍能控制 Ingleton 表达式；同时提出了在 quasi‑uniform 拆分和分层技巧，扩展了此前仅适用于可提取互信息的情形；此外，提出了一个通用的“稀疏分解”工具，可将任意四元组拆成有限个 quasi‑uniform 组成。

**🔧 技术方法**

核心技术包括：1) 二部图的扩散混合引理（expander mixing lemma）；2) 对有限集的分层与划分技术（Alon‑Newman‑Shen‑Tardos‑Vereshchagin 结果）；3) 谱分析与极大最小特征值的关系；4) 信息论中的 Shannon 型不等式和 Gács–Körner 共同信息概念。

**📊 数据集**

未使用标准机器学习数据集，而是构造了若干理论实例：射影平面与其对偶、有限域上多项式图、Grassmannian 旗子图等；这些实例通过计算其谱和互信息来验证理论结果。

**📈 对比分析**

论文未与已有方法在具体实验上直接比较；然而，通过上述实例，可见该方法在这些结构化的随机变量对上得到的 Ingleton 量下界与已知上界差距极小（误差项为 O(log H)），说明该方法在理论上优于传统依赖可提取互信息的证明。

**⚠️ 局限性**

主要限制在于最终得到的误差项仍为 O(log H(X,Y,A,B))，目前尚未证明能进一步压缩到常数级；此外，方法依赖于支撑图为二部正则图的特定结构，对非正则或非二部图的推广尚不明确。

---

## 304. Capacity-constrained demand response in smart grids using deep reinforcement learning

**arXiv ID:** 2602.16525 | [PDF](https://arxiv.org/pdf/2602.16525v1)

**作者:** Shafagh Abband Pashaki `[一作]` (Lincoln AI Lab University of Lincoln), Amir Badiee `[通讯]` (Lincoln AI Lab University of Lincoln)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在本研究中，作者设计并实现了一套基于深度强化学习的容量约束激励式需求响应（CCRL‑DR）框架，利用电力市场价格和聚合住宅负荷的实时信息动态设定激励率，以在满足电网容量限制的前提下调节和分散居民用电负荷。该框架将家庭层面的可控电器（可调功率、可时移、不可中断等）与用户舒适度失衡成本（discomfort cost）相结合，实现了激励与用户行为的协同优化。

**💡 创新点**

创新点包括：
1) 将电网容量约束直接嵌入强化学习的状态和奖励函数，确保决策始终符合物理网络限制；
2) 采用分层结构，服务提供者（SP）为决策主体，住宅用户为环境，提升了模型的可解释性和可扩展性；
3) 在激励决策过程中引入基于LSTM的价格和负荷预测，为强化学习提供前瞻信息；
4) 在用户侧使用家电级别的能量管理系统（HEMS），使得激励效果更细粒度、用户舒适度可量化。

**🔧 技术方法**

技术手段包括：
- 双重深度Q网络（DDQN）作为强化学习算法；
- LSTM网络进行单步价格与负荷预测；
- 全连接网络（两层隐藏层）构成策略与目标网络；
- 经验回放缓冲区与ε‑greedy探索策略；
- 软更新目标网络；
- 采用收益折扣、损失函数调参等典型RL细节。

**📊 数据集**

数据集：
- 住宅负荷：Pecan Street Inc.（Dataport）公开的三户用户2018年4月1日至9月30日的小时级数据；
- 电价：ERCOT市场的实时价格数据（EnergyOnline）同时间段；
- 训练与测试划分：训练集为4月–6月，测试集为7月；

**📈 对比分析**

比较方法与性能：
- 与基准弹性响应（EBLR）模型对比，后者仅使用聚合弹性模型并未考虑容量限制；
- 在7月平均结果中，CCRL‑DR将峰值负荷从11.89 kW降低至7.60 kW（约36%），均值从6.46 kW降至5.34 kW（约17%），PAR从1.84降至1.42（约23%）。
- 与EBLR相比，CCRL‑DR在峰值削减、负荷平滑以及避免回弹峰值方面表现更优。

**⚠️ 局限性**

局限性：
- 研究仅在三户住宅的仿真环境中验证，缺乏大规模、多样化用户群体的实测验证；
- 单一代理模型在用户数增大时可能面临维度灾难，扩展性尚未充分证明；
- 对预测模型的依赖可能在实际运作中受到外部干扰，未评估鲁棒性；
- 研究未深入探讨用户信任、隐私与长期参与度等社会技术问题。

---

## 305. Small molecule retrieval from tandem mass spectrometry: what are we optimizing for?

**arXiv ID:** 2602.16507 | [PDF](https://arxiv.org/pdf/2602.16507v1)

**作者:** Gaetan De Waele `[一作]` (Ghent University), Willem Waegeman `[通讯]` (Ghent University)

**通讯引用:** 4573 | [OpenAlex ID](https://openalex.org/A5028945060)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在LC‑MS/MS数据下，预测分子指纹时不同损失函数对指纹相似度和分子检索性能的影响，并给出了理论的贝叶斯风险与遗憾上界。

**💡 创新点**

揭示了指纹相似度与检索性能之间的基本Pareto折衷，提出了针对向量级相似度、检索指标与位级准确率之间的遗憾上界，并给出可从数据计算的相似度带宽指标作为实用设计指南。

**🔧 技术方法**

使用深度学习预测指纹的框架（多层感知机+sigmoid），并比较了八种损失函数：位级（BCE、Focal）、向量级（Cosine、IoU）和列表级（四种Contrastive变体）。理论分析采用贝叶斯风险与遗憾框架。

**📊 数据集**

MassSpecGym基准数据集（231 104条谱-分子对，28 929种分子），使用Morgan(2,4096)指纹，并在等质量和等化学式两种检索候选设置上评估。

**📈 对比分析**

通过在不同损失函数下训练模型，比较了平均Tanimoto相似度和Hit‑Rate@k（k=1,5,20）。结果显示，Contrastive损失在检索上取得最高HR@k，而IoU损失在Tanimoto相似度上表现最佳，二者形成明显的Pareto前沿；理论上提供的遗憾上界能解释这一折衷。

**⚠️ 局限性**

局限性包括：只考虑了特定的指纹表示与候选集构造，未在非化学二进制检索场景验证；理论假设在实际数据分布中的条件分布不易估计；模型在高不确定性/大化学空间时仍面临检索难题。

---

## 306. Interpretability-by-Design with Accurate Locally Additive Models and Conditional Feature Effects

**arXiv ID:** 2602.16503 | [PDF](https://arxiv.org/pdf/2602.16503v1)

**作者:** Vasilis Gkolemis `[一作]` (ATHENA Research Center), Christos Diou `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新型可解释模型 CALM，利用条件局部可加函数对特征进行分区并分别学习形状函数，实现对交互的局部建模。

**💡 创新点**

创新点在于将特征影响函数按区域分裂成多条一维曲线，既保持了 GAM 的可解释性，又通过条件化捕捉交互，且引入了基于异质性度量的分区与地区回归的蒸馏训练流程。

**🔧 技术方法**

技术手段包括：使用高容量黑盒模型（如 XGBoost）做教师模型，利用 PDP 基于异质性最小化的 CART 生成特征分区树，随后进行区域感知回归（region‑aware backfitting）或梯度提升拟合形状函数。

**📊 数据集**

在实验中使用了 3 组合成回归数据以及 25 个公开 tabular 数据集（10 组分类、15 组回归），涵盖如 Adult、COMPAS、HELOC、MIMIC2、Phoneme、Bike Sharing、California Housing 等。

**📈 对比分析**

与 EBM、EB^2M、NAM、NODE、GAMI‑Net、XGBoost 等基线对比，CALM 在大多数数据集上均超过平均 GAM 表现，往往逼近或匹配 GA^2M 的准确度，同时保持更少的交互项（分类平均 6.2 项，回归平均 15.1 项），并且训练速度在可解释模型中处于中等偏快水平。

**⚠️ 局限性**

局限性包括：当交互数量过多时，分区曲线与竖线增多，图示易读性下降；模型性能高度依赖于教师黑盒的准确性；并且在极大特征维度下分区树的构造与形状函数拟合仍需进一步优化。

---

## 307. Fast and Scalable Analytical Diffusion

**arXiv ID:** 2602.16498 | [PDF](https://arxiv.org/pdf/2602.16498v1)

**作者:** Xinyi Shang `[一作]` (University College London), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6343 | [OpenAlex ID](https://openalex.org/A5066530136)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了动态时间感知的黄金子集扩散框架，显著加速了分析性扩散模型的推理速度

**💡 创新点**

创新点在于发现后验逐步浓缩现象，并设计基于粗细筛选的动态黄金子集选择策略；同时给出了稀疏近似误差上界

**🔧 技术方法**

利用可解释的经验贝叶斯推断、粗-细距离筛选、无偏流式softmax以及时间调度的动态子集采样

**📊 数据集**

在MNIST、FashionMNIST、CIFAR-10、CelebA-HQ、AFHQ、ImageNet‑1K等多尺度数据集上进行验证

**📈 对比分析**

与Optimal、Wiener、Kamb、PCA等分析方法及EDM等神经扩散器对比，71×加速AFHQ，首次在ImageNet‑1K上实现可比质量，且效率大幅提升

**⚠️ 局限性**

对低噪声阶段的极限近似仍依赖精细邻居，过度压缩可能导致细节丢失，且在极大数据规模下仍需预处理以降低粗筛选成本

---

## 308. SRFed: Mitigating Poisoning Attacks in Privacy-Preserving Federated Learning with Heterogeneous Data

**arXiv ID:** 2602.16480 | [PDF](https://arxiv.org/pdf/2602.16480v1)

**作者:** Yiwen Lu `[一作]` (Nanjing University), Yiwen Lu `[通讯]` (Nanjing University)

**通讯引用:** 4255 | [OpenAlex ID](https://openalex.org/A5051696264)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 SRFed 的联邦学习框架，兼顾隐私保护和拜占庭鲁棒性，适用于非独立同分布（Non‑IID）数据。

**💡 创新点**

创新点包括：①去中心化高效功能加密（DEFE），消除第三方依赖并提高解密效率；②基于层级投影与聚类的隐私保护鲁棒聚合策略，能在异构数据下有效识别并剔除恶意模型。

**🔧 技术方法**

采用了 DEFE（功能加密）技术、层级投影、K‑means 聚类、噪声扰动和标准 SGD 训练。

**📊 数据集**

在 MNIST（手写数字）和 CIFAR‑10（彩色图像）两个公开数据集上进行实验。

**📈 对比分析**

与 FedAvg、ShieldFL、PBFL、Median、Biscotti、FoolsGold 等基线相比，SRFed 在整体准确率（OA）、源类准确率（SA）和攻击成功率（ASR）上均表现更优；在效率方面，总训练时间比 ShieldFL 下降 58%，比 ESB‑FL 降低 22%。

**⚠️ 局限性**

在极端攻击比例（>40%）和高度异构数据环境下，鲁棒性仍略有下降；模型对高噪声攻击的容错性仍有提升空间。

---

## 309. Reactive Motion Generation With Particle-Based Perception in Dynamic Environments

**arXiv ID:** 2602.16462 | [PDF](https://arxiv.org/pdf/2602.16462v1)

**作者:** Xiyuan Zhao `[一作]` (Southeast University), Aiguo Song `[通讯]` (Southeast University)

**通讯引用:** 8507 | [OpenAlex ID](https://openalex.org/A5048327458)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于粒子地图的实时动态感知与多自由度机械臂的 MPPI 运动规划框架 SMART，能够在复杂动态环境中实现安全的碰撞回避。

**💡 创新点**

创新点包括：
- 全局双结构粒子地图 G‑DSP，将机器人运动学与粒子状态耦合，实现对障碍物速度与不确定性的显式估计；
- D‑STORM（动态障碍物感知 MPPI），在 MPPI 中同时传播机器人与障碍物动力学，利用粒子地图提供的速度与协方差信息，显著提升对快速移动障碍物的预测与回避能力；
- 通过 GPU 张量化操作实现粒子地图预测、更新、重采样与规划的极端实时化。

**🔧 技术方法**

采用的核心技术：SMC‑PHD 过滤器、GPU 张量运算、MPPI / STORM 控制、双子空间（体素与金字塔）粒子分配、贝塞尔 SDF 机器人模型、基于不确定性的碰撞成本函数。

**📊 数据集**

实验数据集：在 PyBullet 物理引擎中模拟 UR5 机械臂与多种动态障碍物（瓶子、箱子、兔子模型、拼接箱子等），以及真实 UR5 机器人搭配 Intel RealSense D435i 深度摄像头，利用光学运动捕捉系统（NOKOV）记录动态障碍物的真值轨迹。

**📈 对比分析**

方法对比与性能：
- 与基于光线投射的 Ewok、DSP、K3DOM 三种粒子地图以及 STORM、DS‑MPPI 两种基准规划器进行比较；
- 评价指标包括成功率、回路执行时间、轨迹长度与控制频率；
- 在静态与低速障碍物下，SMART 与基准表现相近；
- 在中速/高速动态障碍物下，SMART 的成功率显著高于基准（>88% vs <70%），虽然执行时间略长，但安全性和轨迹可预测性更好。

**⚠️ 局限性**

局限性：
- 仅能在相机视野内感知动态障碍物，视野外的障碍物仍可能导致误判；
- 粒子地图在 FOV 边缘存在退化问题，难以完全消除粒子权重漂移；
- 对大规模、复杂形状障碍物的局部最小值问题仍存在风险；
- 需要更高效的多视角融合与自适应 FOV 扩展，以进一步提升鲁棒性。

---

## 310. Visual Self-Refine: A Pixel-Guided Paradigm for Accurate Chart Parsing

**arXiv ID:** 2602.16455 | [PDF](https://arxiv.org/pdf/2602.16455v1)

**作者:** Jinsong Li `[一作]` (Chinese University of Hong Kong), Dahua Lin `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 42620 | [OpenAlex ID](https://openalex.org/A5010087030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Visual Self-Refine (VSR) 并实现 ChartVSR，提升图表解析精度

**💡 创新点**

利用像素级定位可视化反馈实现模型自我纠错

**🔧 技术方法**

基于 Qwen2.5-VL-3B、视觉编码器 + MLP 叠加、迭代可视化反馈

**📊 数据集**

自建 ChartP-Bench 及 800k 合成图表数据集

**📈 对比分析**

在 ChartP-Bench 和现有 Benchmarks 上显著超越其他 LVLM，尤其在高密度图表上 AP 提升 30+%

**⚠️ 局限性**

推理成本增加，复杂图表仍可能存在难以纠正的深层感知错误

---

## 311. Enhanced Connectivity in Ambient Backscatter Communications via Fluid Antenna Readers

**arXiv ID:** 2602.16446 | [PDF](https://arxiv.org/pdf/2602.16446v1)

**作者:** Masoud Kaveh `[一作]` (Aalto University), F. Javier Lopez-Martinez `[通讯]` (University of Granada)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种基于像素级流体天线（FAS）的AmBC接收机，通过动态选择天线位置来提升回波链路质量，并联合优化回波调制系数以满足能量采集约束。

**💡 创新点**

创新点在于：1）利用FAS的空间多样性在不增加射频链路的前提下克服双重路径损耗与乘法衰落；2）在存在观测噪声的实际环境中，采用基于观测驱动的端口选择；3）将能量收集安全边际纳入整体优化，形成联合离散-连续非凸问题；4）用粒子群优化（PSO）实现对观测误差鲁棒的近似最优解。

**🔧 技术方法**

主要技术包括：流体天线系统（FAS）端口选择、二元OOK背向散射调制、能量收集模型、随机信道与观测误差建模、粒子群优化算法。

**📊 数据集**

实验使用仿真数据，采用均匀分布的FAS端口、相关Rician衰落、观测误差方差0.05、不同端口数K、不同间距d等参数进行Monte‑Carlo模拟，并与单天线TAS基准进行对比。

**📈 对比分析**

与传统单天线TAS基准相比，FAS方案在所有考察的SNR、能量安全边际以及观测误差范围内均实现了明显的吞吐率提升，尤其在中高SNR和较大端口数时差距显著；性能随端口数、间距增大而提升但趋于饱和。

**⚠️ 局限性**

局限性包括：1）仿真结果基于理想化的Rician模型和观测误差假设，未验证硬件实现；2）PSO求解近似最优，未给出收敛性或复杂度分析；3）仅考虑二元OOK调制，未探讨更高阶调制；4）对端口间距与物理尺寸的折衷未给出解析最优方案。

---

## 312. IndicEval: A Bilingual Indian Educational Evaluation Framework for Large Language Models

**arXiv ID:** 2602.16467 | [PDF](https://arxiv.org/pdf/2602.16467v1)

**作者:** Saurabh Bharti `[一作]` (Chhattisgarh Swami Vivekanand Technical University), Nachiket Tapas `[通讯]` (Chhattisgarh Swami Vivekanand Technical University)

**通讯引用:** 1068 | [OpenAlex ID](https://openalex.org/A5081938166)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了IndicEval，一套基于真实印度高考题（UPSC、JEE、NEET）的双语评测平台

**💡 创新点**

创新点在于结合双语推理、链式思维提示与多学科真实考试题材，提升评测生态有效性

**🔧 技术方法**

使用Zero‑Shot、Few‑Shot和Chain‑of‑Thought（CoT）提示技术，并实现自动化答案提取与评分

**📊 数据集**

数据集来源于公开的 UPSC、JEE 与 NEET 试卷，涵盖英语和印地语多选题，约 1,100 题

**📈 对比分析**

对 Gemini 2.0 Flash、GPT‑4、Claude 与 LLaMA 3‑70B 进行全因子实验，CoT 在多项任务上提升 10–25%，但在 Hindi 题目下仍显著低于 English，模型间差距可达 45%

**⚠️ 局限性**

局限性包括仅评估多选题、仅涵盖两种语言、依赖 API 版本稳定性、且采用精确匹配评分无法捕获部分推理过程

---

## 313. Continuous Fluid Antenna Sampling for Channel Estimation in Cell-Free Massive MIMO

**arXiv ID:** 2602.16459 | [PDF](https://arxiv.org/pdf/2602.16459v1)

**作者:** Masoud Kaveh `[一作]` (Aalto University), Kai-Kit Wong `[通讯]` (University College London)

**通讯引用:** 24957 | [OpenAlex ID](https://openalex.org/A5011048761)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了细胞自由大规模MIMO系统中连续流体天线（FA）框架下的上行信道估计，推导了基于高斯过程回归的LMMSE估计量，并与离散端口设计进行了比较。

**💡 创新点**

提出了将连续FA视为空间采样的严格Gaussian Process模型，实现无离散化采样的更优信道估计，并证明在相同位置约束下连续采样能够获得不高于任何有限端口离散方案的NMSE，且在非退化相关性下实现严格改进。

**🔧 技术方法**

使用高斯过程回归、LMMSE估计、Jakes空间相关模型、流体天线运动约束优化以及数值仿真对比等技术。

**📊 数据集**

通过模拟细胞自由网络场景（L=64 AP，K=10 UE，400×400 m²区域，路径损耗指数3.2，阴影噪声8 dB）进行实验，没有使用公开的实际数据集。

**📈 对比分析**

采用相同物理位移约束和符号时间下的NMSE对比方法；连续FA在所有测试情形下均表现出更低的NMSE，尤其在导轨长度增大、pilot长度增加或端口数有限时差距明显扩大。

**⚠️ 局限性**

主要限制包括假设理想时分双工与完美同步、未考虑流体天线的能耗与硬件实现细节，仅评估上行估计，且未研究下行或联合训练与移动最优设计的复杂度。

---

## 314. GICDM: Mitigating Hubness for Reliable Distance-Based Generative Model Evaluation

**arXiv ID:** 2602.16449 | [PDF](https://arxiv.org/pdf/2602.16449v1)

**作者:** Nicolas Salvy `[一作]` (Inria), Bertrand Thirion `[通讯]` (CEA)

**通讯引用:** 79562 | [OpenAlex ID](https://openalex.org/A5026762833)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 GICDM 方法，利用迭代上下文距离度量(ICDM)消除高维嵌入空间中的 hubness 并改进生成模型的距离基评估指标。

**💡 创新点**

创新点在于：①将 ICDM 仅应用于真实数据以均匀化密度；②为生成样本推导独立的尺度因子，使其评估仅受真实数据影响；③加入多尺度过滤防止过度校正，从而在高维空间下可靠地缓解 hubness。

**🔧 技术方法**

使用技术包括：ICDM/ICDM-迭代、非迭代上下文距离度量 (NICDM)、KNN 密度估计、邻域尺度因子、阈值过滤、多尺度 K 设定、Pearson 相关性评估、对比人类评分等。

**📊 数据集**

实验数据集涵盖 synthetic benchmark 以及真实图像数据集：CIFAR‑10、ImageNet、LSUN Bedroom、FFHQ，使用 InceptionV3、DINOv2/3、CLAP 等高维嵌入。

**📈 对比分析**

通过在 synthetic benchmark 上的测试成功计数和与人类评估的 Pearson 相关性进行比较。GICDM 在 Clipped Density、Clipped Coverage 等指标上提升了测试通过率（例如从 8/14 提升到 11/13），并在多数据集上保持或提升与人类评分的相关性；在指导实验中恢复了预期的覆盖度下降。

**⚠️ 局限性**

局限性包括：①仅对基于距离且对真实数据异常值鲁棒的指标有效；②需要多次 ICDM 迭代，计算成本相对较高；③对 K、阈值等超参数选择敏感。

---

## 315. FEKAN: Feature-Enriched Kolmogorov-Arnold Networks

**arXiv ID:** 2602.16530 | [PDF](https://arxiv.org/pdf/2602.16530v1)

**作者:** Sidharth S. Menon `[一作]`, Ameya D. Jagtap `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 5099 | [OpenAlex ID](https://openalex.org/A5061905151)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了 Feature‑Enriched Kolmogorov–Arnold Networks（FEKAN），通过在 KAN 结构中引入特征丰富化技术，显著提升了函数逼近、物理信息网络与神经算子任务的学习效率与精度。

**💡 创新点**

创新点：在不增加可训练参数的前提下，利用 Fourier/随机特征映射对输入进行升维，提升了 KAN 的表示能力、收敛速度和抗谱偏差能力，并在多种基底（Spline、Cheby、RBF 等）上实现了统一、可扩展的改进。

**🔧 技术方法**

采用的技术：Kolmogorov–Arnold Network 及其变体、特征映射（Fourier、随机 Fourier、Chebyshev 等）、物理信息神经网络（PI‑KAN/PI‑FEKAN）、神经算子（DeepONet/DeepOKAN/DeepOFEKAN）、神经切线核（NTK）分析、梯度下降/Adam 优化与自动微分。

**📊 数据集**

使用的数据集与任务：多维函数逼近（含高频与不连续），PDE（Helmholtz、Allen‑Cahn、Klein‑Gordon、Lorenz 系统等），边值问题与多阶段训练，随机采样的高频气泡动力学数据，以及多维输入的神经算子学习任务。

**📈 对比分析**

比较方法与性能：与多种现有 KAN 变体（SplineKAN、FastKAN、ChebyKAN、ReLUKAN、HRKAN、WavKAN 等）在相同网络容量、相同训练轮次下进行对比，主要指标为相对 L₂ 误差、收敛速度（训练迭代次数）和计算时间；实验表明 FEKAN 在所有基底和任务上均能显著降低误差、加速收敛，计算成本相当或仅略有提升。

**⚠️ 局限性**

局限性：特征映射的频率或形式需手动或调参，随机 Fourier 特征可能导致超参数选择困难；升维后模型容量增大，若数据不足易出现过拟合；在某些基底（如 Chebyshev）仍需进一步稳定性改进；整体计算开销仍高于传统 MLP，尤其在大规模高维问题中需进一步优化。

---

## 316. Disproving (Positive) Almost-Sure Termination of Probabilistic Term Rewriting via Random Walks

**arXiv ID:** 2602.16522 | [PDF](https://arxiv.org/pdf/2602.16522v1)

**作者:** Jan-Christoph Kassing `[一作]` (RWTH Aachen University), Jürgen Giesl `[通讯]` (RWTH Aachen University)

**通讯引用:** 4179 | [OpenAlex ID](https://openalex.org/A5025232172)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了自动化方法用于反证概率项重写系统的几乎确定性终止性。

**💡 创新点**

创新点在于将循环检测与随机游走嵌入相结合，利用计数出现次数的动态规划来量化终止性。

**🔧 技术方法**

使用的技术包括概率重写系统、随机游走模型、动态规划计数、模式项分析以及嵌入映射。

**📊 数据集**

实验使用了Terminations Problem Database中的138个概率项重写系统，并扩充了20个非终止或非正终止实例。

**📈 对比分析**

与现有工具相比，新方法在158个基准中共证明正终止70例，否定几乎确定终止24例，性能在30秒超时内完成。

**⚠️ 局限性**

局限性包括对线性循环项的依赖、仅考虑非变量递增树或正向/反向重写、以及对复杂模式项判定的计算复杂度。

---

## 317. MMA: Multimodal Memory Agent

**arXiv ID:** 2602.16493 | [PDF](https://arxiv.org/pdf/2602.16493v1)

**作者:** Yihao Lu `[一作]` (Peking University), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 52915 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多模态记忆代理（MMA），通过动态置信度评分对检索到的记忆进行可靠性评估，从而提升长期交互式系统的可靠性与安全性。

**💡 创新点**

创新点在于引入基于来源可信度、时间衰减和网络共识的三维置信度框架，结合可调节的主动否认机制，并设计了控制源可靠性与多模态冲突的 MMA-Bench 基准测试。

**🔧 技术方法**

技术包括记忆检索后置信度模块、指数时间衰减函数、语义一致性共识度量、可解释的主动否认决策，以及基于风险敏感的 CoRe 评分体系。

**📊 数据集**

使用数据集包括文本事实验证数据集 FEVER、对话式长文本推理基准 LoCoMo，以及自研的 MMA-Bench（包含多模态对话、源可靠性标签及视觉冲突）。

**📈 对比分析**

与 MIRIX 等基线对比，MMA 在 FEVER 上保持相同准确率但标准差降低35.2%，在 LoCoMo 上提升可操作准确率并减少错误回答，在 MMA-Bench 的视觉模式下实现41.18% 的 Type‑B 准确率，显著优于基线。

**⚠️ 局限性**

局限性包括对检索召回率的依赖（若检索缺失证据即使置信度高也无法修正）以及在低密度信息环境下共识机制可能过于保守，需进一步研究自适应门控策略。

---

## 318. Improved Bounds for Discrete Voronoi Games

**arXiv ID:** 2602.16518 | [PDF](https://arxiv.org/pdf/2602.16518v1)

**作者:** Mark de Berg `[一作]` (Eindhoven University of Technology), Geert van Wordragen `[通讯]` (Aalto University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5028724460)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了两玩家在平面上进行的一轮离散Voronoi游戏，其中玩家P可以放置k>1个点，玩家Q仅放置一个点，分析了P在最坏情况下能赢得的选民数量。

**💡 创新点**

创新点在于：1）改进了针对凸范围的ϵ‑net构造，获得更小k的更强net；2）提出基于四叉树的P点放置策略，并结合改进的net，显著提升了k>1时P能保证赢得的选民比例；3）在L1度量下首次给出与L2相当的改进，证明了P至少能赢得(1−66/7k)·n个选民；4）给出了L2度量下的(k=4)可达n/2的结果。

**🔧 技术方法**

主要使用的技术包括：几何分割（如三条共线划分六楔形）、ϵ‑net理论、四叉树压缩结构、Voronoi图性质、中心点与极点理论、递归块划分与网格放点。

**📊 数据集**

论文没有使用实验数据集，所有结果均为理论分析与证明，主要针对任意大小的点集V。

**📈 对比分析**

与先前Baník等人的结果相比，本文在k≥4时将下界从1−42/k提升至1−205/8k；在L1度量下，将下界从1−2/5k提升至1−66/7k；在k=4时实现了n/2的胜利比例，优于先前需k=5的结论。算法实现复杂度为O(n log n)，空间复杂度为O(n)。

**⚠️ 局限性**

局限性包括：1）对大k的下界仍不一定是最优，仍有进一步提升空间；2）四叉树+net的构造在实践中可能需要较大常数；3）L1度量下的网格与凸net方法无法直接得到更细致的极值；4）论文缺乏实验验证，结果仅在理论上证明。

---

## 319. Supercharging Agenda Setting Research: The ParlaCAP Dataset of 28 European Parliaments and a Scalable Multilingual LLM-Based Classification

**arXiv ID:** 2602.16516 | [PDF](https://arxiv.org/pdf/2602.16516v1)

**作者:** Taja Kuzman Pungeršek `[一作]`, Nikola Ljubešić `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了ParlaCAP分类器和对应的ParlaCAP语料库，对欧洲议会发言进行政策议题和情感标注。

**💡 创新点**

创新点是利用LLM教师-学生框架，用GPT‑4o对议会文本自动标注训练数据，再用小型BERT模型Fine‑tune，解决手工标注成本高的问题。

**🔧 技术方法**

技术包括GPT‑4o生成标签、XLM‑R‑Parla和XLM‑RoBERTa等多语言Transformer、针对Public Lands的增广策略以及置信度过滤等方法。

**📊 数据集**

数据集为ParlaMint 4.1/5.0议会发言语料，覆盖29个欧洲国家，辅以PartyFacts、V‑Dem等元数据。

**📈 对比分析**

通过与GPT‑4o、XLM‑R‑Parla基础模型以及其他公开CAP模型比较，ParlaCAP在宏观F1上达到0.70以上，性能接近LLM，显著优于非域适配模型。

**⚠️ 局限性**

局限性包括在少数议会语料（如Bosnian）表现不佳、标签分布不平衡、LLM标注可能引入偏差、测试集不可公开等。

---

## 320. Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding

**arXiv ID:** 2602.16545 | [PDF](https://arxiv.org/pdf/2602.16545v1)

**作者:** Kaiting Liu `[一作]` (Leiden University), Hazel Doughty `[通讯]` (Leiden University)

**通讯引用:** 1960 | [OpenAlex ID](https://openalex.org/A5076082212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出视频分类器在不需要重新训练的情况下，将粗粒度类别拆分为更细粒度子类别的任务（类别拆分）。

**💡 创新点**

创新点在于利用视频分类器隐藏的组合结构进行零样本编辑，构建修改器字典并对齐文本到权重空间，从而在无监督或极少监督下实现类别拆分；同时提出低样本微调结合零样本初始化的方案。

**🔧 技术方法**

技术包括：视频分类器的头部权重修改、修改器检索与对齐（基于文本编码器CLIP或其他），零样本权重构造，低样本微调仅更新扩展头；使用MSE训练对齐模块。

**📊 数据集**

使用两个基准数据集：SSV2-Split（Something‑Something V2）和FineGym‑Split（FineGym）——从原始数据划分粗细标签。

**📈 对比分析**

与多种视觉‑语言模型（CLIP、FG‑CLIP、VideoCLIP‑XL、VideoPrism、InternVideo2 等）对比，零样本方法在一般性（generality）上提升至约45‑46%，在局部性（locality）保持约99%；低样本微调进一步提升。

**⚠️ 局限性**

局限包括：仅编辑分类头导致对更深层特征的利用受限；对新的视觉区分能力依赖于模型背骨已编码的特征；对极其细微或新视觉差异仍易失败；对多重修改器或层次结构的支持有限。

---

## 321. The S-Hamiltonian Cycle Problem

**arXiv ID:** 2602.16532 | [PDF](https://arxiv.org/pdf/2602.16532v1)

**作者:** Antoine Amarilli `[一作]` (University of Lille), Mikaël Monet `[通讯]` (University of Lille)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在给定固定正整数集合S时，图中是否存在S‑Hamiltonian环（即每对相邻顶点间可通过长度属于S的路径相连）的问题，并对S的不同取值进行复杂度分类。

**💡 创新点**

创新点在于给出完整的复杂度决策树：证明S={2}和S={2,4}对应问题为NP‑完整；S={1,2,4}和S={2,4,6}可在多项式（甚至线性）时间内判断；S为所有奇整数或偶整数的情况也可线性判定；并指出剩余的奇整数有限集合（如{1,3}）仍未解决。

**🔧 技术方法**

主要技术包括：针对S={2,4}的Cook归约构造特殊Gadget、利用图的邻接图与线图属性、基于BFS构造覆盖树、对偶图和三角形Gadget进行可视化约简、利用可满足性与MSO解释在有界cliquewidth图上的多项式求解等。

**📊 数据集**

本文为理论研究，未使用具体数据集，而是给出了泛型图的构造与证明。

**📈 对比分析**

性能评估以复杂度分析为主：对可解案例给出线性时间构造算法；对不可解案例给出NP‑完整性证明；对于无限S，给出判定方法并证明其多项式可解性。

**⚠️ 局限性**

局限性：对奇数集合中的非单例有限集合（尤其{1,3}）以及S={2,4}和S={2,4,6}的路径变体，复杂度仍未知；若改为不允许“回跳”或考虑有向图，问题结构与结论将会不同。

---

## 322. Recursive language models for jailbreak detection: a procedural defense for tool-augmented agents

**arXiv ID:** 2602.16520 | [PDF](https://arxiv.org/pdf/2602.16520v1)

**作者:** Doron Shavit `[一作]` `[通讯]` (Silverfort), Doron Shavit (Silverfort)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于递归语言模型的RLM-JB检测框架，利用根模型指挥分段分析、去混淆、并行检测及证据聚合来识别 jailbreak 攻击

**💡 创新点**

将 jailbreak 检测转化为有界分析程序，实现了覆盖保证、分段去混淆和跨段证据合成的程序化检测流程；并通过递归语言模型实现安全可审计的多层调用

**🔧 技术方法**

递归语言模型(RLM)、代码沙箱执行、文本去混淆(Base64 等)、重叠分块、并行分块检测、证据聚合与阈值决策

**📊 数据集**

AutoDAN 风格的攻击样本、InjectPrompt 注入语料、合成的 benign 提示（用 LLM 生成）

**📈 对比分析**

在 DeepSeek‑V3.2、GPT‑4o、GPT‑5.2 三种后端上评估，召回率 92.5%‑98%，精确度 98.99%‑100%，误报率 0%‑2%，F1 分数最高达 98.49%；相比传统单次检测提升显著

**⚠️ 局限性**

未覆盖针对性自适应红队攻击；分块与多调用导致的时延与算力成本；对分块策略、阈值、模板的敏感度需要进一步研究

---

## 323. VIGOR: Visual Goal-In-Context Inference for Unified Humanoid Fall Safety

**arXiv ID:** 2602.16511 | [PDF](https://arxiv.org/pdf/2602.16511v1)

**作者:** Osher Azulay `[一作]` (University of Michigan), Stella X. Yu `[通讯]` (University of Michigan)

**通讯引用:** 10145 | [OpenAlex ID](https://openalex.org/A5042014034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种统一的视觉驱动下的全身跌倒安全框架，涵盖跌倒抑制、冲击缓冲和站立恢复，并实现了零样本从仿真到真实机器人的迁移。

**💡 创新点**

创新点包括：①将数据复杂度拆分为稀疏的人类关键帧和独立的地形变化；②使用“目标-情境”latent表示，将姿态目标与局部地形直接融合；③通过有特权教师与学生的知识蒸馏，将地形感知与高层控制集成。

**🔧 技术方法**

采用强化学习（PPO）训练有特权教师，利用稀疏关键帧进行模仿学习，结合视觉-运动整合网络（基于深度相机的感知编码 + 运动历史编码），实现端到端的动作生成，并使用领域随机化提升仿真-真实一致性。

**📊 数据集**

使用了从公开视频中重建的3D人体运动（VideoMimic）作为稀疏关键帧，配合仿真中随机生成的多种地形（平地、波浪、斜坡、台阶等），以及在Unitree G1机器人上收集的真实深度图像进行测试。

**📈 对比分析**

与HOST和FIRM两种主流基线相比，本文方法在跌倒恢复和站立任务中均实现了更高的成功率、更低的恢复时间、更小的跟踪误差和能耗，且在多种复杂地形下保持零样本实地表现，验证了其优越性。

**⚠️ 局限性**

局限性包括：目前仅在跌倒后进行反应，未与连续行走或长周期导航协同训练；对细微接触过渡的运动精度有限，需更密集的监督或奖励调节；在极端或未知地形下的适应性仍待进一步提升。

---

## 324. Software-heavy Asset Administration Shells: Classification and Use Cases

**arXiv ID:** 2602.16499 | [PDF](https://arxiv.org/pdf/2602.16499v1)

**作者:** Carsten Ellwein `[一作]` (Institute for Control Engineering of Machine Tools University of Stuttgart), Andreas Wortmann `[通讯]` (Institute for Control Engineering of Machine Tools University of Stuttgart)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出了基于运行时环境与功能组件的资产管理壳（AAS）软件重量级分类，并从质量评估与应用场景的角度系统评估不同架构对数字孪生实施的影响，提供了实现和选择 AAS 软件重量级的指导框架。

**💡 创新点**

创新点在于：①构建了两层多级分类（运行时环境三种类型 + 功能层级六级）来描述 AAS 的软件重量级；②将 ISO/IEC 25000 质量标准映射到各级架构，形成可量化的评价表；③针对不同层级给出典型工业用例与最佳实践建议，弥补了现有文献中缺乏系统性分析的空白。

**🔧 技术方法**

使用的技术包括：AAS 规范与子模型、Eclipse BaSyx SDK、容器化（Docker）、脚本/源代码嵌入、REST/OPC‑UA 接口、事件与安全机制；评估方法主要基于对六个质量维度（可靠性、易用性、性能、安全、可维护性、可迁移性）的主观评判，并列出对应的优缺点。

**📊 数据集**

本文未采用具体实验数据集，而是通过对已有研究与标准（如 Kritzinger、FA3ST、VDI/VDE 2193 等）进行文献综述，构建了理论模型与评价表。

**📈 对比分析**

比较方法为表格化的质量评估（绿色=无需担忧、橙色=部分关注、红色=强烈关注）。在性能方面，作者指出低层级（0–2）不需关注性能，较高级别（3–5）需进行性能优化，但未给出量化实验结果，说明评估主要为定性分析。

**⚠️ 局限性**

局限性包括：①缺乏基于真实系统的实验验证与性能基准；②评价主要基于主观判定，缺乏客观度量；③对运行时环境与功能层级的划分仍有主观性，可能不适用于所有工业场景；④未考虑跨平台部署与与现有 MES/ERP 集成的细节。

---

## 325. Learning to Learn from Language Feedback with Social Meta-Learning

**arXiv ID:** 2602.16488 | [PDF](https://arxiv.org/pdf/2602.16488v1)

**作者:** Jonathan Cook `[一作]` (Google DeepMind), Edward Grefenstette `[通讯]` (Google DeepMind)

**通讯引用:** 12334 | [OpenAlex ID](https://openalex.org/A5023508792)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过把静态任务转化为教师-学生的教学式多轮对话，训练LLM在交互中学习如何利用语言反馈来解决问题。

**💡 创新点**

创新点包括：① 引入信息不对称的教师模型，让学生主动获取有价值的反馈；② 证明在线强化学习比离线监督微调更能提升对话学习能力；③ 设计 Q‑priming 预训练阶段，显著增强模型的提问与探索行为。

**🔧 技术方法**

采用离线强化学习+SFT、在线 GRPO 强化学习、POMDP 框架、信息不对称教师、Q‑priming 预训练等技术。

**📊 数据集**

使用的数据集有：Omni‑MATH（2000 题）、OpenCodeInstruct、LiveCodeBench 以及 Lost‑in‑Conversation（数学与代码子任务）。

**📈 对比分析**

与单轮 SFT、单轮 RL、扩大组大小的 RL 基线对比，SML 4‑turn RL 在 Omni‑MATH、LiveCodeBench 与 Lost‑in‑Conversation 上均优于基线，展示了跨域泛化与更高的对话成功率；在线 RL 远优于离线 SFT；Q‑priming 则进一步提升问答率并减少误答。

**⚠️ 局限性**

局限性在于仅在可验证的数学与代码领域、对话级稀疏奖励下实验；未覆盖开放式任务；未尝试更细粒度或多样化的回报模型；未处理目标动态变化或用户意图随对话演变的情境。

---

## 326. Leveraging Large Language Models for Causal Discovery: a Constraint-based, Argumentation-driven Approach

**arXiv ID:** 2602.16481 | [PDF](https://arxiv.org/pdf/2602.16481v1)

**作者:** Zihao Li `[一作]` (Imperial College London), Fabrizio Russo `[通讯]` (Imperial College London)

**通讯引用:** 2754 | [OpenAlex ID](https://openalex.org/A5065482976)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出将大型语言模型（LLM）生成的语义先验与Causal ABA框架结合，实现约束驱动的因果结构发现。

**💡 创新点**

创新点在于设计了鲁棒的LLM约束提取管道、共识过滤机制，以及基于知识图谱的无记忆合成评测协议。

**🔧 技术方法**

采用的技术包括Causal ABA推理、PC/MPC独立性检验、LLM提示与结构化输出、共识过滤、VF2子图同构、以及对ABA的权重软化搜索优化。

**📊 数据集**

使用的数据集包括传统的PC/BCG、TIC等真实因果网络，以及通过CauseNet知识图谱生成的54个5、10、15节点的合成DAG。

**📈 对比分析**

与基线（MPC、FGS、NOTEARS-MLP、GRaSP、BOSS、LLM-only等）比较，ABAPC-LLM在SHD、F1、SID指标上均优于所有对照，并在5~15节点规模下显著提升。

**⚠️ 局限性**

局限性包括对LLM输出的准确性仍依赖共识过滤，可能在复杂/未见领域中召回率下降；以及对大规模图的计算开销和对训练数据记忆的潜在影响。

---

## 327. Training Models on Dialects of Translationese Shows How Lexical Diversity and Source-Target Syntactic Similarity Shape Learning

**arXiv ID:** 2602.16469 | [PDF](https://arxiv.org/pdf/2602.16469v1)

**作者:** Jenny Kunz `[一作]` (Linköping University), Jenny Kunz `[通讯]` (Linköping University)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5049592538)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了使用24种不同源语言的机器翻译文本训练小型英文语言模型时，对语言建模与句法可接受性表现的影响；

**💡 创新点**

创新点在于将源语言的句法相似度、词汇多样性等属性与模型性能进行量化关联，并提出跨源翻译交叉评估来揭示翻译语料“方言”效应；

**🔧 技术方法**

采用125M参数的GPT式Transformer模型（Goldfish训练框架），结合FineTranslations、FLORES+、FineWeb数据，以及BLiMP评测；

**📊 数据集**

使用的主要数据集包括FineTranslations（FineWeb-2翻译成英文）、FineWeb、FLORES+评估集和BLiMP句法可接受性测试；

**📈 对比分析**

通过对比翻译语料模型与本土英文基线在FLORES+、FineWeb的perplexity以及BLiMP准确率，发现翻译语料导致perplexity升高、可接受性略低；在大规模训练时，源语言与英语的句法相似度显著降低perplexity并提升BLiMP准确率，词汇多样性在低数据时更重要；

**⚠️ 局限性**

局限性包括仅翻译到高资源目标语言英文、未完全分离源语言句法与语料属性、仅使用单一翻译系统、模型规模有限以及缺乏文化知识等方面的影响。

---

## 328. HPMixer: Hierarchical Patching for Multivariate Time Series Forecasting

**arXiv ID:** 2602.16468 | [PDF](https://arxiv.org/pdf/2602.16468v1)

**作者:** Jung Min Choi `[一作]` (ISMLL University of Hildesheim), Lars Schmidt-Thieme `[通讯]` (ISMLL University of Hildesheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 HPMixer，针对多变量时间序列长周期预测，采用分离周期与残差的双分支架构，利用可学习周期模块、可学习静止小波变换和两级层级非重叠补丁混合来捕捉多尺度残差。

**💡 创新点**

创新点包括：① 用 MLP 加强可学习周期模块，实现更丰富的周期性表达；② 引入可学习静止小波变换，获得稳定且可降噪的频域表示；③ 设计两级层级非重叠补丁机制与通道混合编码器，提升多尺度残差建模能力。

**🔧 技术方法**

采用的技术包括可学习周期模块、可学习静止小波变换 (LSWT)、层级补丁 (coarse‑fine patching)、通道混合编码器、MLP、反向小波变换 (ISWT) 等。

**📊 数据集**

实验数据集涵盖七个主流多变量时间序列数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity 和 Traffic。

**📈 对比分析**

与 SimpleTM、CycleNet、Timexer、iTransformer、PatchTST、TimeMixer 等基线进行比较，在大多数数据集和预测窗口上取得最佳或接近最佳的 MSE/MAE，HPMixer 在多项指标上多次获得第一名，性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：① 预设周期长度缺乏自适应性，难以处理高度不规则或漂移的周期；② 在高维度数据如 Traffic 上性能下降，通道混合机制可能无法充分捕捉复杂的跨通道依赖；③ 需要进一步研究完全可学习的周期长度和更可扩展的交叉通道建模框架。

---

## 329. Hardware-accelerated graph neural networks: an alternative approach for neuromorphic event-based audio classification and keyword spotting on SoC FPGA

**arXiv ID:** 2602.16442 | [PDF](https://arxiv.org/pdf/2602.16442v1)

**作者:** Kamil Jeziorek `[一作]` (AGH University of Krakow), Tomasz Kryjak `[通讯]` (AGH University of Krakow)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5005086061)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

实现了基于SoC FPGA的事件图神经网络，用于音频分类与实时关键词检测；

**💡 创新点**

创新点在于设计稀疏事件图生成与GraphConv层的硬件友好实现，并将人工耳蜗产生的事件流与量化感知训练结合，首次实现连续低功耗的事件驱动KWS；

**🔧 技术方法**

采用了Graph Neural Network（PointNetConv）、跳步图生成、批归一化与位置归一化、量化感知训练、GRU序列模型、FPGA异步事件流处理等技术；

**📊 数据集**

使用了Spiking Heidelberg Digits (SHD) 和 Spiking Speech Commands (SSC‑35/SSC‑11) 两个公开数据集；

**📈 对比分析**

与FPGA实现的SNN方法相比，SHD分类准确率达到92.74%（比SNN高4.5%），参数仅18.9k；SSC首次给出FPGA基准，KWS实时延迟10.53µs、功耗1.18W；与Jetson Orin GPU对比，延迟提升15k×、功耗降低4倍；

**⚠️ 局限性**

局限性在于实验仅使用模拟事件流，未结合真实人工耳蜗传感器；KWS精度受限于自动标注的词边界，整体性能仍低于软件级最优；硬件资源受限导致可扩展性受限。

---

## 330. Vulnerability Analysis of Safe Reinforcement Learning via Inverse Constrained Reinforcement Learning

**arXiv ID:** 2602.16543 | [PDF](https://arxiv.org/pdf/2602.16543v1)

**作者:** Jialiang Fan `[一作]` (University of Notre Dame), Fanxin Kong `[通讯]` (University of Notre Dame)

**通讯引用:** 2787 | [OpenAlex ID](https://openalex.org/A5007560139)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种基于逆约束强化学习的对抗攻击框架，利用专家演示和黑盒交互学习约束网络与代理策略，进而在不访问目标策略梯度或真实约束的情况下对安全强化学习（Safe RL）策略进行攻击，揭示其安全漏洞。

**💡 创新点**

创新点在于：①无需对手策略梯度或真实安全约束；②通过ICRL学习得到的约束网络和代理策略作为梯度来源，支持梯度上升式对抗扰动；③给出约束误差与攻击效果的理论边界；④在多个安全RL基准上系统验证攻击效果。

**🔧 技术方法**

采用逆约束强化学习（ICRL）学习约束函数和学习者策略；使用梯度上升/投影梯度下降生成对抗扰动；构建可微系统辨识网络和约束网络；利用Lipschitz连续性理论给出攻击强度与成本增长的上界。

**📊 数据集**

实验数据来自PyBullet‑based安全RL基准 OmniSafe 的四个环境：Safe‑Ant‑Velocity、Safe‑Ant‑Position、SafetyBallRun、SafetyBallCircle。

**📈 对比分析**

与基线 L2 攻击（FGSM、PGD、Max‑Reward、Max‑Cost）进行对比，L1 框架在同一扰动预算下导致的成本（安全违约）显著高于基线，同时保持相近的累计回报；实验结果与理论上限一致，证明能诱发真实成本违例。

**⚠️ 局限性**

局限性：①攻击效果依赖于ICRL学习到的约束近似，误差过大可能导致误判；②需要一定数量专家演示，数据获取成本；③扰动预算与约束的Lipschitz常数估计较为粗略，泛化到更复杂动态环境仍需验证。

---

## 331. Transfer Learning of Linear Regression with Multiple Pretrained Models: Benefiting from More Pretrained Models via Overparameterization Debiasing

**arXiv ID:** 2602.16531 | [PDF](https://arxiv.org/pdf/2602.16531v1)

**作者:** Daniel Boharon `[一作]` (Ben Gurion University), Yehuda Dar `[通讯]` (Ben Gurion University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在目标线性回归任务中利用多源预训练模型进行迁移学习的效果与理论。

**💡 创新点**

创新点在于提出了针对过参数化预训练模型的欠偏（overparameterization bias）补偿方法，使得多模型迁移在过参数化情形下保持一致性。

**🔧 技术方法**

主要技术包括随机矩阵理论、闭式解析的期望误差推导、最优正则化调参以及基于验证的缩放因子学习。

**📊 数据集**

实验使用合成高维高斯线性回归数据，涵盖不同参数化水平、协方差形状以及任务关系矩阵。

**📈 对比分析**

与传统单模型迁移、岭回归和无迁移基线相比，使用多模型迁移能显著降低测试误差，且经过欠偏补偿后误差可接近贝叶斯最优。

**⚠️ 局限性**

局限在于需要对预训练模型的参数化程度或通过验证确定缩放因子；对非线性任务或实际数据集的适用性尚未验证。

---

## 332. Optimizing Soft Prompt Tuning via Structural Evolution

**arXiv ID:** 2602.16500 | [PDF](https://arxiv.org/pdf/2602.16500v1)

**作者:** Zhenzhen Huang `[一作]` (University of Electronic Science and Technology of China), Heng Tao Shen `[通讯]` (Tongji University)

**通讯引用:** 30802 | [OpenAlex ID](https://openalex.org/A5052993469)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过持久同调分析软提示向量在训练过程中的结构演化，并提出 Topological Soft Prompt Loss (TSLoss) 进行结构化正则化；

**💡 创新点**

创新点在于将拓扑数据分析（TDA）应用于软提示的高维表示，量化其 H0/H1 结构，并基于此设计正则化损失，显著提升可解释性与收敛速度；

**🔧 技术方法**

主要技术包括持久同调、Vietoris‑Rips 复形、softmin 邻近度、H0/H1 损失函数、TSLoss 与交叉熵联合训练，应用于多大规模 LLM 的软提示微调；

**📊 数据集**

实验使用 GSM8K、MMLU‑CF 与 LongBench 三大基准数据集；

**📈 对比分析**

在五个 LLM（Gemma‑2B‑IT、Open‑LLaMA‑7B、DeepSeek‑7B‑Chat、LLaMA2‑13B、Qwen1.5‑14B）上与多种基线（Standard、L2‑Reg、PDLoss、Prefix‑Tuning、P‑Tuning v2、ACT、DPC）比较，TSLoss 在绝大多数模型/数据集上实现最优或接近最优的准确率，并在收敛迭代次数上大幅减少；

**⚠️ 局限性**

局限性包括：需要手动调节 λ_ts 与温度 α 以避免振荡；对极大模型的效果相对有限；结构分析仅在 Gemma‑2B‑IT 上完成，未覆盖更大规模模型；缺乏对结构与语义之间关系的深入解释。

---

## 333. From Growing to Looping: A Unified View of Iterative Computation in LLMs

**arXiv ID:** 2602.16490 | [PDF](https://arxiv.org/pdf/2602.16490v1)

**作者:** Ferdinand Kapl `[一作]` (Technical University of Munich), Stefan Bauer `[通讯]` (Technical University of Munich)

**通讯引用:** 25645 | [OpenAlex ID](https://openalex.org/A5083708821)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对“循环模型”（Universal Transformers）与“深度增长模型”两种结构进行了系统比较与整合，证明它们在内部计算模式、对后层的依赖和迭代计算方面表现相似，并展示了两者可组合使用（先增长再循环）以提升推理能力。

**💡 创新点**

创新点在于首次将循环与深度增长视为同一迭代计算机制的两种实现方式，并通过机制诊断与层级干预验证它们共享的深度周期性特征；此外，提出在推理时对深度增长模型进行循环增强可获得显著性能提升。

**🔧 技术方法**

使用的技术包括Transformer模型的层级循环（weight‑tying）、中间层重复增长、残差流诊断、层级干预（交换/去除层），以及在推理时对中间块的重复迭代；同时在预训练与冷却阶段引入了高质量数学数据混合。

**📊 数据集**

实验数据集覆盖22项评测：知识类（Closed‑book Q&A）、开放书（Open‑book Q&A）、语言模型（Lambada、HellaSwag）、数学推理（SVAMP、ASDiv、MAWPS、GSM8K）以及推理基准（Variable Assignment等），并使用SmolLM‑v1在约200B/400B tokens上预训练。

**📈 对比分析**

对比方法包括在相同参数、相同推理 FLOPs、相同训练 FLOPs下比较标准、循环与深度增长模型；结果显示循环模型在参数预算下的推理性能优于基线，深度增长模型在训练 FLOPs 较低时同样能提升推理；两者结合后在相同数据与推理 FLOPs下实现了最高的推理准确率。

**⚠️ 局限性**

主要局限在于完全循环模型对层级顺序更为敏感，推理时多次循环往往不再带来进一步提升；此外实验仅验证了特定块大小（4层）和规模（360M/1.7B），未探究更大规模或不同架构下的泛化能力。

---

## 334. Phase-Based Bit Commitment Protocol

**arXiv ID:** 2602.16489 | [PDF](https://arxiv.org/pdf/2602.16489v1)

**作者:** Janis Nötzel `[一作]` (Technical University of Munich), Peter van Loock `[通讯]` (Johannes Gutenberg University Mainz)

**通讯引用:** 12829 | [OpenAlex ID](https://openalex.org/A5066133727)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于相位的光学量子比特承诺协议，并给出了其在诚实但好奇环境下的安全性分析；

**💡 创新点**

创新点在于引入网络提供者保证传输线路安全作为假设，利用相位平均的相干态实现近似隐藏和绑定，并讨论了Mayers攻击的实现难度；

**🔧 技术方法**

主要技术包括相干态生成、相位偏移编码、光学位移操作、光子计数测量以及对密钥状态的Wigner函数和一范数分析；

**📊 数据集**

论文未使用实际数据集，属于理论协议设计与分析；

**📈 对比分析**

通过解析公式给出了Bob和Alice的作弊概率上界，并在参数M、k、能量t的选择上给出了ε安全的取值范围；

**⚠️ 局限性**

局限性包括：需要假设网络提供者能完全防止窃听；协议仅实现近似安全，需要较高能量和大M值；实际实现中需克服高阶光子数的产生与检测挑战。

---

## 335. Pitts and Intuitionistic Multi-Succedent: Uniform Interpolation for KM

**arXiv ID:** 2602.16445 | [PDF](https://arxiv.org/pdf/2602.16445v1)

**作者:** Hugo Férée `[一作]` (Université Paris Cité), Ian Shillito `[通讯]` (University of Birmingham)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5017336560)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文为intuitionistic modal logic KM构造了一个多后继终止的序列推理系统，并利用Pitts的技术证明了该逻辑的统一插值性质。

**💡 创新点**

创新点在于将Pitts的统一插值方法从单后继到多后继、从经典到intuitionistic的转移，并设计了新的终止且无割规则的多后继序列算子。

**🔧 技术方法**

主要技术包括证明理论中的终止与割消除、Pitts的递归插值构造，以及在Coq证明助手中的完整机械化。

**📊 数据集**

本研究未使用任何实验数据集，而是以形式化证明和Coq脚本验证为主要手段。

**📈 对比分析**

由于为理论研究，本文未进行实验性能比较；通过Coq机械化实现保证了证明的严谨性与可计算性。

**⚠️ 局限性**

局限性在于方法目前仅已在KM逻辑上实现，对其他intuitionistic多后继逻辑的推广仍需要进一步研究。

---

## 336. Learning with Locally Private Examples by Inverse Weierstrass Private Stochastic Gradient Descent

**arXiv ID:** 2602.16436 | [PDF](https://arxiv.org/pdf/2602.16436v1)

**作者:** Jean Dufraiche `[一作]` (University of Lille), Marc Tommasi `[通讯]` (University of Lille)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的算法，称为逆魏尔斯特拉斯私有随机梯度下降（IWP-SGD），用于在非交互式局部差分隐私（LDP）下进行学习，旨在纠正因LDP引入的偏差。

**💡 创新点**

创新点在于通过逆魏尔斯特拉斯变换来表征LDP引起的偏差，并提出了一种新的无偏梯度估计器，从而实现了在完全任务无关的LDP设置下渐近恢复非私有问题的种群风险最小化器。

**🔧 技术方法**

使用了逆魏尔斯特拉斯变换和伯努利变换的数学工具，构建了IWP-SGD算法，并进行了理论分析和实证验证。

**📊 数据集**

使用了合成数据集和真实世界数据集（如ACSIncome和ACSPublicCoverage）进行实验验证。

**📈 对比分析**

与传统的SGD方法相比，IWP-SGD在处理LDP数据时能够有效消除偏差，且在合成和真实数据集上均表现出较好的收敛性和无偏性。

**⚠️ 局限性**

限制在于IWP-SGD的梯度估计器的方差较高，可能导致在某些情况下的性能下降，体现了偏差与方差之间的权衡。

---

## 337. Quecto-V1: Empirical Analysis of 8-bit Quantized Small Language Models for On-Device Legal Retrieval

**arXiv ID:** 2602.16640 | [PDF](https://arxiv.org/pdf/2602.16640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 338. Causally-Guided Automated Feature Engineering with Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.16435 | [PDF](https://arxiv.org/pdf/2602.16435v1)

**作者:** Arun Vignesh Malarkkan `[一作]` (Arizona State University), Yanjie Fu `[通讯]` (Arizona State University)

**通讯引用:** 6154 | [OpenAlex ID](https://openalex.org/A5032187620)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为CAFE的因果引导式自动特征工程框架，结合因果结构发现和多智能体强化学习实现特征转换的自适应生成。

**💡 创新点**

创新点在于将因果图作为软先验引导搜索，利用多智能体DQN分层决策以及因果形塑奖励来实现对抗分布漂移的稳健特征生成。

**🔧 技术方法**

核心技术包括NOTEARS稀疏因果图学习、三阶段级联多智能体深度Q学习、因果层级探索策略、奖励形塑与特征复杂度惩罚。

**📊 数据集**

在15个公开基准数据集上验证，包括分类与回归任务，使用XGBoost作为下游模型。

**📈 对比分析**

与10个基线方法（统计、传统AFE、RL、LLM等）对比，CAFE在13/15数据集上提升宏观F1/逆相对绝对误差最高达7%，在受控协变量偏移下性能下降仅为4倍（7.1% vs 28.1%）。

**⚠️ 局限性**

主要限制是对因果发现假设（因果充分性、线性加性噪声、静态图）敏感，无法处理未观测混杂、时变因果结构或反馈环。

---

## 339. Beyond SGD, Without SVD: Proximal Subspace Iteration LoRA with Diagonal Fractional K-FAC

**arXiv ID:** 2602.16456 | [PDF](https://arxiv.org/pdf/2602.16456v1)

**作者:** Abdulla Jasem Almansoori `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Martin Takáč `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 5298 | [OpenAlex ID](https://openalex.org/A5070679093)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种无需SVD的LoRA优化器PSI-LoRA，利用近似低秩投影和预收敛子空间迭代实现高效训练；

**💡 创新点**

将LoRA训练视为在无SVD条件下求最优低秩近似的proximal子问题，提出ALS+子空间迭代加Kronecker‑factored度量的内存高效实现，并引入对角K‑FAC/​Shampoo分数阶缩放，显著降低学习率调优成本；

**🔧 技术方法**

使用的技术包括proximal subspace iteration、交替最小二乘（ALS）与块幂迭代、Kronecker‑factored度量（K‑FAC、Shampoo）及其对角近似、分数阶矩阵幂缩放、低秩动量更新；

**📊 数据集**

实验涵盖CIFAR‑100、GLUE（MNLI、QNLI、QQP、SST‑2、CoLA、STS‑B）、SQuAD v2、WikiText‑103、T5‑SQuAD、GPT‑2‑WikiText‑103等数据集；

**📈 对比分析**

与Full‑weight、LoRA、RPLoRA、SVDLoRA、Proj.等基线比较，PSI‑LoRA（Scaled）在大多数任务上能匹配或优于LoRA/SVDLoRA，并对学习率更稳健；

**⚠️ 局限性**

主要局限包括额外的计算和缓存开销、内层迭代在随机梯度下可能退化、对角K‑FAC仅为粗糙近似，未来需改进低秩度量、梯度累积、方差降低和分布式实现。

---

## 340. Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments

**arXiv ID:** 2602.16653 | [PDF](https://arxiv.org/pdf/2602.16653v1)

**作者:** Yangjie Xu `[一作]` (University of Luxembourg), Radu State `[通讯]` (University of Luxembourg)

**通讯引用:** 5253 | [OpenAlex ID](https://openalex.org/A5069228908)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

探讨并评估 Agent Skill 框架在小规模语言模型（SLM）上的效果，系统地比较不同规模与专业化模型在多个任务中的性能与 GPU VRAM‑time 效率。

**💡 创新点**

提出 Agent Skill 的正式数学定义，证明小模型在大型技能库中鲁棒性差，并显示代码专用模型在执行效率上优于一般指令调优模型；同时引入“进阶披露”机制并量化其对小模型能力的影响。

**🔧 技术方法**

使用基于 POMDP 的信息寻求控制模型，对技能选择与执行做动态上下文管理；采用不同上下文工程策略（DI、FSI、ASI）并通过 GPU VRAM‑time 作为成本指标评估。

**📊 数据集**

实验数据集包括公开的 IMDB（电影评论）、FiNER（金融 XBRL 标签）和专有的 InsurBench（保险理赔邮件），分别用于二分类、标签预测和多步骤决策任务。

**📈 对比分析**

比较方法：在同一任务下对比 DI、FSI 与 ASI 三种上下文工程方式，并与 GPT‑4o‑mini 作为基准；性能表现显示：约 12B–30B 的中型模型在技能选择上显著提升，80B 代码专用模型在执行准确率上可与 GPT‑4o‑mini 相当，同时显著降低 VRAM‑time，显示出高效性。

**⚠️ 局限性**

局限性：仅覆盖分类与标签两类任务，缺乏对递归或持续推理场景的深入分析；小模型在进阶披露下的表现尚不稳定；Skill.md 的最佳结构与表述方式仍待研究。

---

## 341. A Systematic Evaluation of Sample-Level Tokenization Strategies for MEG Foundation Models

**arXiv ID:** 2602.16626 | [PDF](https://arxiv.org/pdf/2602.16626v1)

**作者:** SungJun Cho `[一作]` (University of Oxford), Mark W. Woolrich `[通讯]` (University of Oxford)

**通讯引用:** 72228 | [OpenAlex ID](https://openalex.org/A5088377480)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

系统性评估了MEG神经基础模型中样本级分词策略，提出并验证了一种自学习的分词器并与传统固定分词器进行对比

**💡 创新点**

首次将自学习分词器与固定分词器在同一实验框架下进行统一评估，探索分词对重建、生成、个体识别和下游任务的影响

**🔧 技术方法**

使用基于自编码器的自学习分词器、均值缩放+均匀/分位数离散化、μ‑变换等分词方法，构建MEG‑GPT transformer预训练模型并进行生成、谱分析、个体指纹和解码实验

**📊 数据集**

Cam‑CAN、Nottingham MEGUK、Wakeman‑Henson三大公开MEG数据集，涵盖不同采集设备、任务与受试者

**📈 对比分析**

通过PVE重建精度、token预测准确率、合成数据的静态/动态谱相似度、个体识别Top‑1/一致性分数以及零射/微调解码准确率进行多维比较，结果显示固定分词器与自学习分词器在大多数指标上表现相当，后者在个体指纹和生成的谱精度上略有优势

**⚠️ 局限性**

仅关注样本级分词，未探讨非样本级压缩策略；每个分词器仅训练一次，未评估训练稳定性；模型窗口限制为80样本，可能忽略慢动态；依赖特定预处理与源重建，跨管道推广性未知

---

## 342. Explainable AI: Context-Aware Layer-Wise Integrated Gradients for Explaining Transformer Models

**arXiv ID:** 2602.16608 | [PDF](https://arxiv.org/pdf/2602.16608v1)

**作者:** Melkamu Abay Mersha `[一作]` (University of Colorado Colorado Springs), Jugal Kalita `[通讯]` (University of Colorado Colorado Springs)

**通讯引用:** 9021 | [OpenAlex ID](https://openalex.org/A5049180880)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Context-Aware Layer-wise Integrated Gradients (CA-LIG) 框架，用于解释 Transformer 模型的决策，融合每层的 Integrated Gradients 与 class-specific attention gradients，生成层级且上下文敏感的归因图。

**💡 创新点**

创新点：① 在每个 Transformer block 上计算层级 Integrated Gradients，捕获 token 重要性随层级演化的过程；② 将 token 层级重要性与注意力梯度融合，既保留局部贡献又体现全局交互；③ 采用 relevance rollout 与正负分解，得到支持/反对证据并保持归因一致性。

**🔧 技术方法**

技术方法：Layer-wise Integrated Gradients、Attention Gradient Fusion、Symmetric Min-Max 归一、λ 加权融合、relevance rollout、正负分解等。

**📊 数据集**

数据集与模型：文本任务使用 IMDB（情感分析）、20 Newsgroups（多类长文本）、Amharic Hate Speech（低资源语言）结合 BERT、XLM‑R、AfroLM；视觉任务使用 CIFAR‑10、ASIRRA，模型为 Masked Autoencoder Vision Transformer。

**📈 对比分析**

比较与性能：与 IxG、IG、LRP、Attention‑Rollout/Last 等基线方法对比，采用 token‑F1（文本）和插入/删除 AUC（视觉）等量化指标，CA‑LIG 在多任务、多数据集上均取得更高的 F1、AUC，且可视化结果更符合人类直觉，说明其更可信、更易解释。

**⚠️ 局限性**

局限性：仅验证 encoder‑only 结构，未针对 decoder 或多模态 Transformer；λ 参数需手工调优；视觉实验覆盖有限，未系统评估多种视觉模型；未来需扩展至跨模态、可学习的融合机制等。

---

## 343. A Contrastive Learning Framework Empowered by Attention-based Feature Adaptation for Street-View Image Classification

**arXiv ID:** 2602.16590 | [PDF](https://arxiv.org/pdf/2602.16590v1)

**作者:** Qi You `[一作]` (University College London), James Haworth `[通讯]` (University College London)

**通讯引用:** 1891 | [OpenAlex ID](https://openalex.org/A5056082244)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CLIP-MHAdapter，一种轻量级的多头自注意力适配器，用于街景图像属性分类。

**💡 创新点**

创新点是将多头自注意力模块加入CLIP适配器，使其能够捕获局部补丁间的相互依赖，提升细粒度属性识别，同时保持极低的可训练参数量。

**🔧 技术方法**

使用CLIP视觉-文本预训练模型、视觉Transformer、轻量级多头自注意力（MHSA）、Bottleneck MLP、残差融合等技术。

**📊 数据集**

使用公开的Global StreetScapes（GSS）数据集，包含八个街景属性。

**📈 对比分析**

与零样本CLIP、线性探针、CoOp、CLIP-Adapter、MaxViT等基线比较，CLIP-MHAdapter在大多数属性上实现或超过全量模型的准确率，仅需约1.38M可训练参数，显著降低计算成本。

**⚠️ 局限性**

局限在于对类别不平衡敏感，部分属性（如天气、反射）仍落后于全模型；标签噪声与不一致也影响性能。

---

## 344. Utility-Preserving De-Identification for Math Tutoring: Investigating Numeric Ambiguity in the MathEd-PII Benchmark Dataset

**arXiv ID:** 2602.16571 | [PDF](https://arxiv.org/pdf/2602.16571v1)

**作者:** Zhuqian Zhou `[一作]` (Cornell University), René F. Kizilcec `[通讯]` (Cornell University)

**通讯引用:** 7114 | [OpenAlex ID](https://openalex.org/A5071778778)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于人机协作的 MathEd-PII 数据集，并系统探讨了数学辅导对话中数值歧义导致的 PII 误检问题。

**💡 创新点**

首次提出数值歧义分析与段落感知提示（segment‑aware prompting）来提升数学对话中的 PII 检测精度，兼顾数据实用性。

**🔧 技术方法**

使用了 Microsoft Presidio 规则引擎、LLM（Claude、Gemini、GPT 系列）以及基于数学密度的分段方法与段落感知提示。

**📊 数据集**

采用了 1,000 场数学辅导会话（115,620 条消息、769,628 个 token）构成的 MathEd-PII 数据集。

**📈 对比分析**

与 Presidio 基线相比，段落感知提示下 Gemini 3 Pro 的 F1 提升至 0.821（精度 0.934，召回 0.730），显著降低了数值歧义导致的误检。

**⚠️ 局限性**

局限于单一来源、数据多样性不足、召回率仍有提升空间，以及对低成本/边缘部署效率尚未验证。

---

## 345. Arc2Morph: Identity-Preserving Facial Morphing with Arc2Face

**arXiv ID:** 2602.16569 | [PDF](https://arxiv.org/pdf/2602.16569v1)

**作者:** Nicolò Di Domenico `[一作]` (University of Bologna), Davide Maltoni `[通讯]` (University of Bologna)

**通讯引用:** 16484 | [OpenAlex ID](https://openalex.org/A5027196976)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于 Arc2Face 的深度学习面部变形方法，能够在保持身份信息的同时生成逼真的 ISO/ICAO 兼容人脸图像。

**💡 创新点**

创新点在于：①将 ArcFace 识别嵌入投影至 CLIP 多模态空间，并在该空间进行球面线性插值以实现身份混合；②使用 ControlNet 控制姿态与表情，BEN2 背景去除实现标准化；③在多种公开与私有数据集上全面评估，证明其攻击潜力优于传统基于特征点与现有深度学习方法。

**🔧 技术方法**

使用 ArcFace 识别编码器、CLIP 文本编码器、Arc2Face 生成器、ControlNet、BEN2 背景去除网络、EMOCAv2 语义映射、Spherical Linear Interpolation (slerp) 等技术。

**📊 数据集**

实验数据集包括 FEI、ONOT、SOTAMD Digital、iMARS-MQ、EINMorph-HQ v2、EINMorph-MQ v2 等，覆盖真实与合成、低高分辨率及多种光照/姿态条件。

**📈 对比分析**

通过 Morphing Attack Potential (MAP) 指标与多种商业与深度学习 FRS 对比，Arc2Morph 在所有评估场景中取得最高 MAP，尤其在单/多 probe 与多 FRS 的鲁棒性与通用性曲线中均优于对手。

**⚠️ 局限性**

局限性包括：①对 Arc2Face 及其预训练数据的依赖，可能对不同人种、光照、表情变化的泛化有限；②实验多使用 Arc2Face 生成的 probe，导致模型对该生成器更具攻击性；③未在极端低光、遮挡或非标准姿态下充分验证，需进一步研究更广泛场景的鲁棒性。

---

## 346. Hidden in Plain Sight: Detecting Illicit Massage Businesses from Mobility Data

**arXiv ID:** 2602.16561 | [PDF](https://arxiv.org/pdf/2602.16561v1)

**作者:** Roya Shomali `[一作]` (University of Alabama), Jason Parton `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出利用匿名手机位置信息（mobility data）检测隐蔽的非法按摩业务，并通过正负无标签学习（positive‑unlabeled learning）构建风险评分，供执法部门进行资源分配与优先检查。

**💡 创新点**

创新点在于①将移动数据作为不易被经营者操纵的行为信号；②采用PU学习解决标签不对称问题；③将预测结果直接嵌入预算约束的优先级优化框架；④从移动数据中识别四大运营特征（需求稳定、晚间集中、短服务时长、局部客源）。

**🔧 技术方法**

使用正负无标签学习的Bagging方法（PU Bagging），基于随机森林作为基学习器；特征工程包括28个时空运营指标；评估采用Spy技术的AUC、Average Precision、恢复率等指标。

**📊 数据集**

数据集为2024年美国境内的按摩/水疗POI周访客数据（约891k观测），以及成人服务网站广告数据（约920家店铺），通过手机号匹配形成正样本。

**📈 对比分析**

相较于仅用在线评论或广告的传统方法，模型在POI‑周层面实现0.973的AUC、0.843的AP；在业务层面最大聚合方案下，前10%高风险商家可捕获约53%的已知非法业务，提升约5.3倍。

**⚠️ 局限性**

局限包括：①正样本仅基于成人广告，可能遗漏其他非法运营；②仅覆盖单一年份，季节与执法动态的稳定性未知；③无法区分强迫与自愿的性交易；④移动数据隐私与伦理审查限制。

---

## 347. Towards Autonomous Robotic Kidney Ultrasound: Spatial-Efficient Volumetric Imaging via Template Guided Optimal Pivoting

**arXiv ID:** 2602.16641 | [PDF](https://arxiv.org/pdf/2602.16641v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 348. Illustration of Barren Plateaus in Quantum Computing

**arXiv ID:** 2602.16558 | [PDF](https://arxiv.org/pdf/2602.16558v1)

**作者:** Gerhard Stenzel `[一作]` (LMU Munich), Claudia Linnhoff-Popien `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在变分量子电路（VQC）中使用参数共享对全局最优解和梯度景观的影响，并提出了梯度欺骗检测算法。

**💡 创新点**

创新点在于首次量化参数共享导致的梯度欺骗程度，展示了欺骗率随共享度上升而显著增加，并给出了可量化的“梯度欺骗度”指标。

**🔧 技术方法**

采用量子模拟器对多重参数共享的量子电路进行梯度采样，使用 Adam 和 SGD 等经典梯度优化器进行训练，并引入了梯度欺骗检测算法与分辨率概念。

**📊 数据集**

实验数据来源于高分辨率（高达1440采样点）模拟量子电路的全局最优解，未使用真实数据集，而是基于合成量子线路生成的训练目标。

**📈 对比分析**

通过比较不同学习率和共享程度下 Adam 与 SGD 的收敛成功率与欺骗率，结果显示随着参数共享增加，优化器收敛成功率下降，最优学习率需要逐步调整，梯度方法的性能仅提升约一阶量级，远低于理论上可实现的四阶提升。

**⚠️ 局限性**

主要限制包括对噪声量子硬件的缺乏考虑、梯度基优化器在高度欺骗景观下表现不佳、实验仅限于模拟环境且未验证进化或混合优化方法的有效性。

---

## 349. Neighborhood Stability as a Measure of Nearest Neighbor Searchability

**arXiv ID:** 2602.16673 | [PDF](https://arxiv.org/pdf/2602.16673v1)

**作者:** Thomas Vecchiato `[一作]` (University of Copenhagen), Sebastian Bruch `[通讯]` (Northeastern University)

**通讯引用:** 508 | [OpenAlex ID](https://openalex.org/A5046454671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出邻域稳定性（NSM）作为聚类质量的内部指标，并用其评估基于聚类的近似最近邻检索和图像聚类效果。

**💡 创新点**

首次将邻域稳定性与近邻一致性概念结合，定义聚类NSM和点NSM，证明其满足聚类质量公理并与ANN准确率和聚类外部指标显著相关。

**🔧 技术方法**

采用基于最近邻一致性的稳定性度量、随机采样与近邻搜索、以及Hoeffding等统计理论进行分析与实验验证。

**📊 数据集**

在欧氏距离、余弦相似度、内积等三类度量下，使用公开ANN基准数据集（如MNIST、ImageNet、10M向量等）和视觉模型嵌入的图像数据集进行实验。

**📈 对比分析**

与传统内部指标（Davies‑Bouldin、Silhouette、相似性指数）比较，聚类NSM在ANN准确率及Mutual Information、Homogeneity上的Spearman相关系数普遍显著（p<0.001），显示更高的预测能力。

**⚠️ 局限性**

仅适用于平面聚类且权重方案固定，未覆盖层次或密度聚类；理论证明主要依赖理想球形条件，实际中对内积距离的适用性仅通过经验验证。

---

## 350. Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment

**arXiv ID:** 2602.16660 | [PDF](https://arxiv.org/pdf/2602.16660v1)

**作者:** Yuyan Bu `[一作]` (Beijing Academy of Artificial Intelligence), Juntao Dai `[通讯]` (Institute for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了插件式的多语言一致性(MLC)损失，用于在不需要多语言响应数据的情况下实现多语言安全对齐。

**💡 创新点**

创新点在于通过单一步骤的奇异值正则化，将多语言表示压缩为近似秩1，从而实现跨语言的语义一致性，既资源高效又兼容现有对齐范式。

**🔧 技术方法**

使用了表示学习、奇异值分解（SVD）与温度软最大化的辅助损失、传统对齐损失(SFT/DPO/SimPO/ORPO)的联合训练，以及线性表示提取器。

**📊 数据集**

主要使用修改版PKU‑SafeRLHF语料库进行训练，评估数据包括PKU‑SafeRLHF、MultiJail以及MMLU/ MMMLU‑lite进行安全与通用能力测评。

**📈 对比分析**

与MPO、SDRRL等多语言对齐基线以及单语对齐方法对比，MLC在10种语言上平均安全率提升至≈90%，方差降低≥90%，攻击成功率显著下降，同时对通用能力影响极小。

**⚠️ 局限性**

局限性包括：对极低资源语言仍需翻译文本；在极深层或语言特定层的对齐可能削弱多语言通用性；在超大模型或极端多语言场景下的进一步验证尚缺。

---

## 351. Causal and Compositional Abstraction

**arXiv ID:** 2602.16612 | [PDF](https://arxiv.org/pdf/2602.16612v1)

**作者:** Robin Lorenz `[一作]` (Quantinuum), Sean Tull `[通讯]` (Quantinuum)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出将抽象视为自然变换的范式，统一并推广现有因果抽象概念，并扩展至量子模型；

**💡 创新点**

通过范畴论框架把抽象关系定义为自然变换，提出双向抽象、组件级抽象和量子抽象，统一并扩展传统方法；

**🔧 技术方法**

使用对称单张范畴、马尔科夫范畴、字符串图、自然变换及构造型因果模型等技术；

**📊 数据集**

本文为理论工作，未使用具体数据集；

**📈 对比分析**

通过与文献中的构造抽象、Q‑τ一致性、分布式抽象等对应证明等价性，暂无实验性能比较；

**⚠️ 局限性**

主要局限在于仅给出理论框架，缺乏实证验证；量子因果模型的完整定义尚未完全展开；对大规模模型的可扩展性未讨论。

---

## 352. FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM Serving

**arXiv ID:** 2602.16603 | [PDF](https://arxiv.org/pdf/2602.16603v1)

**作者:** Chia-chi Hsieh `[一作]` (Tsinghua University), Lijie Wen `[通讯]` (Tsinghua University)

**通讯引用:** 4499 | [OpenAlex ID](https://openalex.org/A5030845033)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对多SLO LLM服务的系统，利用操作符级抢占与事件驱动调度，分离了抢占粒度与调度频率，从而显著缓解了prefill阶段的HoL阻塞，提升了服务吞吐量和响应速度。

**💡 创新点**

创新点在于：1）操作符级预抢（Operator-Level Preemption），在Transformer的最小执行单元之间插入抢占点，几乎无执行延迟；2）事件驱动调度（Event-Driven Scheduling），仅在请求到达或完成时触发调度，避免了频繁调度导致的控制开销；3）滑动余量调度（S-EDF）与SLO感知批量（SLO-aware Batching）进一步提升了多SLO环境下的良好吞吐量。

**🔧 技术方法**

采用了自研的抢占检测与协同机制、基于余量的EDF调度算法、预测TTFT的多项式模型、以及与vLLM、DistServe集成的调度框架，支持Tensor并行和Mixture-of-Experts结构。

**📊 数据集**

使用真实的生产级请求轨迹 QwenTrace（包含聊天、图像、搜索、摘要四类任务）以及多种模型（Llama3-8B、Qwen2.5-14B、Llama3-70B、Qwen3-30B-A3B）进行评估。

**📈 对比分析**

与DistServe（CP2K/CP8K）和vLLM基线对比，实验显示在相同SLO保障下，系统在高负载时可提升4.7–5.6倍的良好吞吐量；在SLO紧迫度上实现1.5–3.1倍更严格的满足度；且抢占延迟仅在4.5 ms以下，几乎无阻塞。

**⚠️ 局限性**

局限性包括：1）主要针对prefill阶段，对decode优化不显著；2）在极端长输入时仍受单个操作符执行时间限制；3）实验主要在单机多GPU场景，跨机分布式扩展尚未深入验证；4）对低负载/单SLO场景收益有限。

---

## 353. Sensor Query Schedule and Sensor Noise Covariances for Accuracy-constrained Trajectory Estimation

**arXiv ID:** 2602.16598 | [PDF](https://arxiv.org/pdf/2602.16598v1)

**作者:** Abhishek Goudar `[一作]` (Technical University of Munich), Angela P. Schoellig `[通讯]` (University of Toronto)

**通讯引用:** 5954 | [OpenAlex ID](https://openalex.org/A5052147335)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于后验克莱姆-劳伯下界的半正定规划框架，用来计算满足给定轨迹估计精度的传感器查询速率或传感器噪声协方差。

**💡 创新点**

创新点在于将期望估计误差目标直接嵌入可凸化的约束中，既可优化查询速率也可优化噪声协方差，并能判定某一精度是否可实现。

**🔧 技术方法**

主要技术包括后验克莱姆-劳伯下界的递推推导、凸优化（SDP）以及对线性化系统的期望求解（高斯积分/蒙特卡罗），实现工具为CVXPY+MOSEK。

**📊 数据集**

实验数据来自仿真中的随机初始轨迹以及真实实验中的八个UWB基站与光学运动捕捉系统收集的移动机器人轨迹。

**📈 对比分析**

通过与低速率/高噪声的基线对比，所计算的速率或协方差能使RMSE落在预设阈值内，而基线往往超标；实验结果显示算法在仿真和实测中均能满足精度目标，性能优于传统经验式或基于不确定性阈值的调度。

**⚠️ 局限性**

局限性包括：仅适用于线性高斯模型或可线性化的系统；求解需要先验过程噪声信息；优化结果可能保守，导致传感器速率高于硬件实际可承受；未考虑非欧几里得运动学的非线性拓展。

---

## 354. Why Thinking Hurts? Diagnosing and Rectifying the Reasoning Shift in Foundation Recommender Models

**arXiv ID:** 2602.16587 | [PDF](https://arxiv.org/pdf/2602.16587v1)

**作者:** Luankang Zhang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 28029 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在推荐系统中启用Chain-of-Thought（CoT）推理后导致性能下降的现象，并提出一种无训练的推理时子空间对齐方法，既压缩推理链又校正其对推荐的偏置，恢复甚至提升模型表现。

**💡 创新点**

创新点在于：①诊断CoT导致的“通用子空间先验”文本惯性，使语义ID信号被稀释；②提出推理时子空间对齐框架，包括推理链压缩与基于对比的偏置减扣；③在保持推理优势的同时消除无关文本漂移，实现无额外训练的性能提升。

**🔧 技术方法**

主要技术包括：大型语言模型（如Qwen-1.7B/8B）作为推荐基底；压缩推理链的轻量级指令模型；对齐推理与历史信息的三种上下文对比评分（Expert、Amateur、Baseline）；Z-score归一化与偏置减扣公式；Beam search与语义ID搜索。

**📊 数据集**

数据集：OpenOneRec基准的广告（Ad）和产品（Product）两类测试集，各采样1,000条实例；对比基准模型包括SASRec和HSTU。

**📈 对比分析**

对比方法：在Think-Off（无推理）与Think-On（启用CoT）两种模式下与SASRec、HSTU、以及本方法在两种模型规模（1.7B/8B）下的表现。实验结果表明：①单纯的CoT会导致Recall@K/NDCG@K下降；②使用子空间对齐后，Think-On模式恢复并超过Think-Off，且效果在两种规模下均稳定；③模型规模提升提升了非推理基线，但CoT偏置仍存，子空间对齐能显著缓解。

**⚠️ 局限性**

局限性包括：①压缩推理链仍依赖轻量级模型的质量，可能在极长推理链上失效；②偏置减扣的α超参数需要经验调优；③方法在非基础模型或其他推荐框架中验证有限；④对极端领域（如冷启动）中的推理效益仍待进一步研究。

---

## 355. A Scalable Approach to Solving Simulation-Based Network Security Games

**arXiv ID:** 2602.16564 | [PDF](https://arxiv.org/pdf/2602.16564v1)

**作者:** Michael Lanier `[一作]`, Yevgeniy Vorobeychik `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了MetaDOAR，一个层次化的元控制器，用于加速大规模网络安全游戏中的连续控制双Oracle最佳响应。

**💡 创新点**

创新点在于通过学习的设备筛选器与LRU Q值缓存，显著缩小动作空间并减少重复计算，从而实现可扩展的高效最佳响应。

**🔧 技术方法**

采用了双Oracle/PSRO框架、结构化节点嵌入、top‑k 设备选择、经验回放与缓存机制，以及CyGym模拟环境。

**📊 数据集**

使用了CyGym（Volt Typhoon CyberDefenseEnv）构建的从10到10,000设备不等的网络拓扑。

**📈 对比分析**

与原始DOAR、IPPO、MAPPO、HAGS、HMARL等基线对比，MetaDOAR在大规模网络中实现更高玩家效用，训练时间和内存几乎不随设备数增长而增大。

**⚠️ 局限性**

局限在于对最佳响应的近似可能导致非精确均衡，缓存失效策略及元学习的收敛性未给出理论保证。

---

## 356. Evaluating Collective Behaviour of Hundreds of LLM Agents

**arXiv ID:** 2602.16662 | [PDF](https://arxiv.org/pdf/2602.16662v1)

**作者:** Richard Willis `[一作]` (King's College London), Joel Z. Leibo `[通讯]` (Google DeepMind)

**通讯引用:** 7184 | [OpenAlex ID](https://openalex.org/A5054808675)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种评估框架，让大型语言模型（LLM）生成可编程策略来应对社交困境，并在大规模代理群体中评估其集体行为。

**💡 创新点**

创新点在于：①以算法形式生成策略，支持预部署检查和大规模扩展；②结合文化进化模拟用户选择，揭示利用性策略如何主导群体演化；③系统性比较多种最先进LLM在公共资源、集体风险和共同池资源游戏中的表现。

**🔧 技术方法**

使用技术包括：提示工程让LLM生成自然语言策略，再将其转化为代码；多模型（Claude Haiku 4.5、Gemini 2.5 Flash、GPT 5 Mini等）自我博弈与文化进化模拟；主成分分析（PCA）评估策略多样性与态度分离度；归一化收益衡量社会福利。

**📊 数据集**

主要使用的“数据集”是由LLM生成的512条每种态度（合作性、利用性）的策略集合，以及通过自我博弈产生的游戏回合历史；没有使用传统文本或图像数据集，而是依赖LLM生成的算法代码和对应的游戏结果。

**📈 对比分析**

比较方法：在不同群体规模（4、16、64、256）下，按比例混合合作与利用策略进行自我博弈，计算归一化社会福利；随后在同一环境下执行文化进化模拟，记录最终基因占比与福利效率。结果显示，DeepSeek 在利用性策略下仍能保持较高福利，Claude 在合作性策略下表现最优，但在文化进化中其利用性策略往往主导，导致大多数情形下福利偏低。

**⚠️ 局限性**

局限性包括：①游戏模型过于简化，二元动作、固定收益与已知轮次；②缺乏通信机制，可能低估真实多智能体交互中的合作潜力；③文化进化参数和态度定义未系统调优；④LLM生成策略的可靠性受模型解释差异影响；⑤未在更复杂真实环境中进行实证验证。

---

## 357. Fast Shortest Path in Graphs With Sparse Signed Tree Models and Applications

**arXiv ID:** 2602.16605 | [PDF](https://arxiv.org/pdf/2602.16605v1)

**作者:** Édouard Bonnet `[一作]` (CNRS), Sungmin Moon `[通讯]` (KAIST)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文提出了一种新的有符号树模型（Signed Tree Model），并基于该模型设计了高效的单源最短路、全点对最短路、矩阵乘法以及一阶模型检验等算法；

**💡 创新点**

创新点在于：①将有符号树模型推广为能处理稠密图的稀疏表示；②利用几何算法对矩形集进行补集划分，从而得到稠密图的间隔双团分解；③在不依赖构造序列（witness）的情况下，以概率方式快速构造有符号树模型；④通过有符号树模型实现多项式时间内的 APSP、矩阵乘法和模型检验；

**🔧 技术方法**

主要技术包括：树模型与 DAG 压缩的转换、矩形包络与补集分割算法、随机采样与几何集合划分、可变宽度的 merge‑width 与 twin‑width 的关系、以及基于间隔双团分解的稀疏矩阵向量乘法；

**📊 数据集**

论文未给出具体实验数据集，全部结果均为理论证明与算法分析；

**📈 对比分析**

与之前的结果相比，单源最短路从 O(n) 降到 O(n log n)（对稀疏树模型），全点对最短路在 bounded twin‑width 图上从 O(n^3) 降至 O(n^2 log^2 n)；在 symmetric difference O(n^{1/3}) 的图上得到 O(n^{7/3} log^2 n) 的 APSP；矩阵乘法从 O(n^3) 降至 O(n^2 log n)；而一阶模型检验在给定 merge‑width witness 的情况下由 O(n^3) 降至 O(n^2)；

**⚠️ 局限性**

局限性包括：①算法常数与对数因子仍较大，实际可实现性尚未验证；②部分结果仍需给定 witness（如 twin‑width、merge‑width 序列）或在高概率下完成；③对非常稠密图（如 m=Θ(n^2)）的运行时间仍为超线性；④在没有构造序列的情况下求解 sd‑degeneracy 仍是随机化算法，可能在最坏情况出现失败。

---

## 358. Decentralized and Fully Onboard: Range-Aided Cooperative Localization and Navigation on Micro Aerial Vehicles

**arXiv ID:** 2602.16594 | [PDF](https://arxiv.org/pdf/2602.16594v1)

**作者:** Abhishek Goudar `[一作]` (Technical University of Munich), Angela P. Schoellig `[通讯]` (University of Toronto)

**通讯引用:** 5954 | [OpenAlex ID](https://openalex.org/A5052147335)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种完全去中心化、基于范围辅助的多机协同定位与距离约束组态控制方法，能在资源受限的微型无人机团队中实现小米级定位与组态精度；

**💡 创新点**

创新点在于：①使用异步块坐标下降（BCD）进行协同定位，避免了同步通信与全局协调；②将组态控制建模为带运动先验的MAP推断问题，自动考虑状态不确定性；③将定位与组态控制统一在因子图框架中，高效可并行求解；

**🔧 技术方法**

核心技术包括：因子图优化、VIO与UWB距离预积分、连续时间高斯过程运动先验、异步块坐标下降、固定延迟平滑器；

**📊 数据集**

使用Gazebo仿真环境、室内外UWB+VIO+视觉数据以及运动捕捉系统作为地面真值数据集；

**📈 对比分析**

与中心化批处理估计及梯度控制方法对比，定位RMSE≈0.10‑0.12 m，组态RMSE≈0.1‑0.3 m，性能与中心化方案相当且优于梯度控制；

**⚠️ 局限性**

局限性包括：在高速运动、通讯延迟或丢包较多时性能下降；未加入障碍物避让；需要UWB或其他可靠距离测量手段；假设组态图保持微刚性。

---

## 359. DataJoint 2.0: A Computational Substrate for Agentic Scientific Workflows

**arXiv ID:** 2602.16585 | [PDF](https://arxiv.org/pdf/2602.16585v1)

**作者:** Dimitri Yatsenko `[一作]` (DataJoint Inc), Thinh T. Nguyen `[通讯]` (DataJoint Inc)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了面向科学工作流的关系工作流模型（Relational Workflow Model），将数据表、行和外键等元件统一为数据、结构和计算依赖，并在此基础上开发了一个支持对象存储、语义匹配、可扩展类型系统以及分布式作业管理的完整平台。

**💡 创新点**

创新点包括：① 将外键用作执行依赖，将表定义为工作流步骤；② 对象增强模式（OAS）实现事务级联对象存储与关系存储；③ 基于属性血缘的语义匹配，防止同名属性错误关联；④ 可插拔的类型系统与分布式作业协调，实现可观测、可重现、可扩展的科研数据流水线。

**🔧 技术方法**

技术包括 Python 框架 DataJoint（实现五操作算子查询、make() 声明式计算）、MySQL/PostgreSQL 关系数据库、对象存储（S3/Blob）与 fsspec 接口、分布式作业表 + populate()、语义匹配算法、可扩展编码器/解码器。

**📊 数据集**

使用的实验数据集主要来自神经科学与精准医学领域：MICrONS、Aeon、Spyglass、SCENE、Hussain Shuler Lab、ORION、PosePipe、UCSF Cadwell Lab、Harvard Mouse Behavior Core 等多种多模态实验数据。

**📈 对比分析**

通过在这些真实科研项目中部署，展示了模型对数据完整性、可追溯性、重现性和协作的支持；与传统文件式工作流（Nextflow、Snakemake）相比，提供了事务一致性、语义安全和可查询的计算依赖；在大规模实验（数百万行）下仍保持可伸缩性，且可通过导出到 Delta/iceberg 等湖仓格式实现高效分析。

**⚠️ 局限性**

局限性包括：① 行式数据库不适合大规模列式分析，无法直接满足大数据分析需求；② 需要额外的模型学习与维护，学习曲线相对陡峭；③ 对象存储与事务同步的开销与配置复杂；④ 目前仅支持 MySQL/PostgreSQL，缺乏对新型数据库的原生支持。

---

## 360. MerLean: An Agentic Framework for Autoformalization in Quantum Computation

**arXiv ID:** 2602.16554 | [PDF](https://arxiv.org/pdf/2602.16554v1)

**作者:** Yuanjie Ren `[一作]` (Massachusetts Institute of Technology), Yidi Qi `[通讯]` (Northeastern University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了 MerLean，一个完全自动化的双向框架，用于将科学论文中的数学表述提取、形式化为 Lean 4 代码，并将验证后的 Lean 代码再转换为人类可读的自然语言蓝图；

**💡 创新点**

创新点在于：①利用前沿 LLM（Claude Opus 4.5）与多轮 agentic 交互，完成从原始文本到形式化代码的全流程；②通过 compile‑fix‑verify 循环确保代码可编译且与原意一致；③引入自动“autoinformalization”实现形式化结果的可读性回译；④在完整论文层面（包含依赖关系）实现自动化，突破以往单一定理或片段级别的限制；

**🔧 技术方法**

使用技术包括：Claude LLM 作为核心推理引擎；Lean 4 与 Mathlib 作为形式化目标；Model Context Protocol（MCP）服务器提供语言服务器交互；semantic search 工具（leansearch, leanfind）辅助检索现有 lemmas；错误诊断与修复循环；明确声明缺失库支持的 axiom 阶段；以及自然语言再生成的 LLM 端点；

**📊 数据集**

数据集为三篇理论量子计算论文：1）《Balanced Product Codes》；2）《Fault‑Tolerant Quantum Computation》；3）一篇未公开的手稿（保证无数据泄漏）。共提取 114 条数学声明，生成 2,050 条 Lean 声明，代码行数超过 41,000；

**📈 对比分析**

与现有自动形式化工作（如 miniF2F、ProofNet 等）相比，MerLean 在完整论文级别实现了 end‑to‑end 自动化；在三篇论文上共耗时 42 小时，平均每条声明约 21 分钟；在每类声明中，定义最易编译，定理最难；实验表明大多数声明在 1–10 次 compile‑fix 内成功，只有少数需 20+ 次；该系统在三篇论文中都完成全部形式化，未出现未修复错误；

**⚠️ 局限性**

局限性包括：①对 Mathlib 现有库的依赖仍存在缺口，需手工声明 axiom，降低可迁移性；② faithfulness 检查仍可能遗漏细微语义差异，需人工审查验证；③整体依赖昂贵 LLM 资源，成本与可扩展性受限；④实验仅覆盖三篇量子计算论文，缺乏跨领域泛化验证；⑤缺乏标准化 benchmark 与公开比较基线。

---

## 361. Automated Extraction of Mechanical Constitutive Models from Scientific Literature using Large Language Models: Applications in Cultural Heritage Conservation

**arXiv ID:** 2602.16551 | [PDF](https://arxiv.org/pdf/2602.16551v1)

**作者:** Rui Hu `[一作]` (Shanghai University), Jizhong Huang `[通讯]` (Shanghai University)

**通讯引用:** 790 | [OpenAlex ID](https://openalex.org/A5080675011)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

基于两阶段大模型（Gatekeeper‑Analyst）自动从海量科学论文中提取文化遗产材料的力学本构方程、参数及元数据；

**💡 创新点**

提出了可扩展的“两阶段代理框架”与“上下文感知符号归一化”机制，显著提升对多模态科学文献中隐含方程的识别与语义映射；

**🔧 技术方法**

采用大型语言模型（LLM）进行语义过滤与精细提取，结合Schema约束的JSON输出、闭环校验与自我纠错；

**📊 数据集**

对来自arXiv的2000+科研论文进行预处理，最终筛选出113篇核心文献，提取185个本构实例和450+实验校准参数；

**📈 对比分析**

与人工标注的Ground‑Truth 222条目标比对，系统在Precision 80.4%、Recall 83.3%、F1 81.9%（AUC 0.782、FPR 3.3%）等指标上表现优异，手工工作量减少约90%；

**⚠️ 局限性**

受限于低分辨率PDF解析、图表数据抽取难度和模型“hallucination”，导致少量误检与漏检，后续需改进扫描质量和图表数字化技术。

---

## 362. Factorization Machine with Quadratic-Optimization Annealing for RNA Inverse Folding and Evaluation of Binary-Integer Encoding and Nucleotide Assignment

**arXiv ID:** 2602.16643 | [PDF](https://arxiv.org/pdf/2602.16643v1)

**作者:** Shuta Kikuchi `[一作]` (Keio University), Shu Tanaka `[通讯]` (Keio University)

**通讯引用:** 1468 | [OpenAlex ID](https://openalex.org/A5057961231)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将因子机与Ising机结合的FMQA框架，用于解决RNA逆折叠问题。

**💡 创新点**

创新点在于将RNA核苷酸的分类变量通过不同二进制编码映射到FM模型，并系统评估编码与整数-核苷酸映射对搜索效果的影响。

**🔧 技术方法**

采用因子机(FM)做代理模型，利用模拟退火Ising机进行离散优化，并使用序列的归一化集合缺陷(NED)作为目标函数。

**📊 数据集**

实验数据来自Eterna100基准，主要测试目标为stickshift以及另外八个长度在12–36nt的无伪结点二级结构。

**📈 对比分析**

与随机搜索、树结构Parzen估计器(TPE)和遗传算法比较，FMQA在保持同等评估预算下实现了更低的NED和更高的成功率，评估次数显著减少。

**⚠️ 局限性**

局限性包括仅适用于无伪结点结构，受编码与整数映射影响大，且在较长或极短支链结构上表现下降。

---

## 363. Consensus Based Task Allocation for Angles-Only Local Catalog Maintenance of Satellite Systems

**arXiv ID:** 2602.16678 | [PDF](https://arxiv.org/pdf/2602.16678v1)

**作者:** Harrison Perone `[一作]` (University of Connecticut), Christopher W. Hays `[通讯]` (Air Force Research Laboratory)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出一种基于Consensus‑Based Bundle Algorithm（CBBA）的去中心化任务分配算法，用于多卫星系统的局部目录维护，并引入新的观测评分与切换逻辑。

**💡 创新点**

创新点包括：① 将观测质量评分与协方差主轴、相对距离和角度相结合；② 采用黑名单机制和基于信息下降率的目标切换；③ 对CBBA引入折扣因子和α阈值调节，以平衡燃料消耗与不确定性控制。

**🔧 技术方法**

技术手段：CBBA任务分配、网络分布式卡尔曼滤波（N-DKF）与逆协方差插值（ICI）进行状态估计，角度仅传感器测量，姿态控制采用比例-阻尼力矩控制，整体仿真使用Python实现。

**📊 数据集**

数据集：采用合成仿真场景，包含2个通信卫星和8个小天体，初始位置随机生成；未使用真实宇宙物体数据集。

**📈 对比分析**

比较方法：与固定时滞阈值（hysteresis）算法在燃料消耗和不确定性截断积分（clipped integral）两指标上进行对比；实验显示CBBA改进版在两项指标上均优于现有Pareto前沿，燃料更少且不确定性更低。

**⚠️ 局限性**

局限性：仅在小规模（2卫星、8目标）场景下验证；假设即时通信；仅考虑角度仅传感器且FOV固定；折扣因子和α参数需要针对不同规模与传感器特性重新调优；未评估异步通信或多目标FOV观测的效果。

---

## 364. SPARC: Scenario Planning and Reasoning for Automated C Unit Test Generation

**arXiv ID:** 2602.16671 | [PDF](https://arxiv.org/pdf/2602.16671v1)

**作者:** Jaid Monwar Chowdhury `[一作]` (Bangladesh University of Engineering and Technology), Reyhaneh Jabbarvand `[通讯]` (University of Illinois)

**通讯引用:** 634 | [OpenAlex ID](https://openalex.org/A5058824250)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SPARC框架，利用控制流图、检索增强的操作映射和路径级测试生成来自动化C语言单元测试。

**💡 创新点**

创新点在于将测试生成拆分为每条可执行路径的情景，结合检索增强的LLM推理和迭代修复循环，显著减少语义误差和编译错误。

**🔧 技术方法**

采用了LLM（如OpenAI GPT系列）、RAG检索、Clang生成CFG、路径枚举、Unity测试框架、AddressSanitizer等技术。

**📊 数据集**

使用了59个真实世界与算法型C项目作为评估数据集。

**📈 对比分析**

与普通提示生成和符号执行工具对比，SPARC在行覆盖率提升31.36%、分支覆盖率提升26.01%、变异得分提升20.78%，在复杂案例上匹敌或超过符号执行工具。

**⚠️ 局限性**

局限性包括对路径枚举规模的敏感、仍需多轮LLM修复、对极大代码基的可扩展性有限，以及对LLM质量和可解释性的依赖。

---

## 365. PredMapNet: Future and Historical Reasoning for Consistent Online HD Vectorized Map Construction

**arXiv ID:** 2602.16669 | [PDF](https://arxiv.org/pdf/2602.16669v1)

**作者:** Bo Lang `[一作]` (Lehigh University), Mooi Choo Chuah `[通讯]` (Lehigh University)

**通讯引用:** 3708 | [OpenAlex ID](https://openalex.org/A5046998111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种端到端的在线高精度地图（HD）构建框架PredMapNet，利用查询跟踪与预测实现连续、可追踪的向量化地图生成。

**💡 创新点**

创新点在于：①引入语义感知查询生成器，用全景分割引导查询初始化；②设计历史栅格化地图记忆与历史地图引导模块，使跟踪查询获得细粒度空间先验；③加入短期未来引导模块，显式预测地图实例的即时运动，从而提升时序一致性。

**🔧 技术方法**

采用Mask2Former式语义注意力、DETR风格查询解码、BEV编码器、栅格化存储记忆、跨帧图像-激光融合以及小型MLP进行轨迹预测。

**📊 数据集**

在nuScenes和Argoverse2两个公开自动驾驶数据集上进行训练与评估，包含旧分割和新分割。

**📈 对比分析**

与Mask2Map、MapTRv2、StreamMapNet、MapTracker等SOTA方法对比，PredMapNet在nuScenes上达到76.9 mAP和69.7 C‑mAP（相比Mask2Map提升≈+5 mAP、+8 C‑mAP），在Argoverse2上获得77.3 mAP和69.1 C‑mAP，整体精度与时序一致性均领先。

**⚠️ 局限性**

局限性包括：在追踪稀疏或遮挡场景下仍可能出现误跟；相机‑激光同步要求高；模型相较于单帧方法略低FPS（≈10 fps），需进一步加速。

---

## 366. Towards a Science of AI Agent Reliability

**arXiv ID:** 2602.16666 | [PDF](https://arxiv.org/pdf/2602.16666v1)

**作者:** Stephan Rabanser `[一作]` (Princeton University), Arvind Narayanan `[通讯]` (Princeton University)

**通讯引用:** 18551 | [OpenAlex ID](https://openalex.org/A5058102069)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了将 AI 代理的可靠性拆解为一致性、鲁棒性、可预测性和安全性四个维度，并设计了 14 个可量化的指标来评估各维度的表现；

**💡 创新点**

创新点在于把安全可靠工程的多维度可靠性框架迁移到 AI 代理评估，形成一套既能独立于能力衡量又可互相比较的指标体系；

**🔧 技术方法**

采用多跑实验、输入/环境扰动、错误注入、置信度自评与 LLM 生成的安全审计等技术手段实现指标测量；

**📊 数据集**

使用了 GAIA（包含 165 个多工具任务）和 τ-bench（包含 26 个经校验的客服对话任务）作为评测数据集；

**📈 对比分析**

对 14 种模型（OpenAI、Google、Anthropic）在两个基准上进行 5 次多跑评估，结果显示能力提升虽显著，但在一致性、鲁棒性和安全性等维度的改进有限，说明单一准确率无法反映实际可靠性；

**⚠️ 局限性**

局限包括基准覆盖范围有限、仅使用单一代理框架、LMM 安全评估的可靠性、指标选择主观性、以及安全性未完整纳入总体可靠性分数等。

---

## 367. Optimizer choice matters for the emergence of Neural Collapse

**arXiv ID:** 2602.16642 | [PDF](https://arxiv.org/pdf/2602.16642v1)

**作者:** Jim Zhao `[一作]` (University of Basel), Aurelien Lucchi `[通讯]` (University of Basel)

**通讯引用:** 12520 | [OpenAlex ID](https://openalex.org/A5060064717)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过理论推导和大规模实验，研究了优化器（尤其是权重衰减的耦合/解耦）对神经网络终端训练阶段出现神经崩塌（NC）现象的影响。

**💡 创新点**

创新点在于提出新的诊断指标 NC0，并证明在自适应优化器中仅当使用耦合权重衰减时 NC 才能出现；同时首次揭示动量能加速 NC 的收敛，展示了 Adam 与 AdamW 在 NC 产生上的根本差异。

**🔧 技术方法**

使用了 SGD、Adam、AdamW、SignGD（及其耦合/解耦变体）、SGDW 等一阶优化器；理论分析基于 NC0 的动力学、动量和权重衰减的结合；实验中训练 ResNet9、VGG9 等 CNN 及 UFM 设定。

**📊 数据集**

实验涵盖 MNIST、FashionMNIST、CIFAR‑10 等标准图像分类数据集，总计超过 3,900 次训练。

**📈 对比分析**

通过比较 NC0、NC1、NC2、NC3 四个 NC 指标的最终值和下降趋势，发现耦合权重衰减能够显著降低 NC 指标，Adam 在这方面优于 AdamW；动量在 SGD 中进一步加速 NC 指标趋零，实验表明不同优化器在相同训练误差下可得到不同的几何结构。

**⚠️ 局限性**

局限性包括：理论证明仅在简化的 SignGD+UFM 设定下完成，尚未完全推广到深度网络和真正的自适应优化器；仅分析了最后一层的 NC 属性，未探讨中间层；实验覆盖的模型和数据集相对有限，未包含更大规模网络（如 ViT）和多样化任务。

---

## 368. ColBERT-Zero: To Pre-train Or Not To Pre-train ColBERT models

**arXiv ID:** 2602.16609 | [PDF](https://arxiv.org/pdf/2602.16609v1)

**作者:** Antoine Chaffin `[一作]` (LightOn), Florent Krzakala `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多向量检索模型ColBERT的预训练流程，比较仅KD、监督对比+KD和全预训练三种训练路径的效果。

**💡 创新点**

证明全预训练显著提升性能，并发现仅加入监督阶段即可接近全预训练，揭示提示对齐在复用预训练模型中的关键作用。

**🔧 技术方法**

采用ColBERT架构、InfoNCE对比损失、KL蒸馏损失、Prompt嵌入，并使用PyLate+GradCache实现大规模无显存限制训练。

**📊 数据集**

主要使用公开的Nomic Embed数据集进行无监督和监督对比训练，使用MS‑MARCO生成KD标签，并利用Nomic硬负样本进行监督微调。

**📈 对比分析**

在BEIR基准上评估nDCG@10，ColBERT‑Zero在多数数据集上超越GTE‑ModernColBERT和其基模型，单步KD效果最差。

**⚠️ 局限性**

受数据质量与规模限制，提示对齐效果与训练阶段高度耦合，尚未验证在更强数据或不同模型上的通用性。

---

## 369. Sequential Membership Inference Attacks

**arXiv ID:** 2602.16596 | [PDF](https://arxiv.org/pdf/2602.16596v1)

**作者:** Thomas Michel `[一作]` (Inria), Emilie Kaufmann `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出针对模型连续更新的最优成员推断攻击 SeMI*，并基于概率推理实现顺序测试。

**💡 创新点**

创新点在于推导“隔离性”性质，证明在顺序访问时可完全恢复插入批次统计量，消除传统单一模型攻击的信号稀释问题，并给出针对已知、均匀、未知插入时刻的最优检验。

**🔧 技术方法**

主要技术包括极大似然比检验、贝叶斯推断、梯度正态近似、隐私预算的差分隐私框架以及统计置信下界推导。

**📊 数据集**

实验使用 Fashion‑MNIST、CIFAR‑10 与 Purchase‑100 三个公开数据集，评估在预训练+DP‑SGD 微调场景下的攻击效果。

**📈 对比分析**

与现有基于损失变化的 heuristic 攻击（Delta、Back‑Front 等）相比，SeMI‑SGD 在三组数据集上实现更紧的隐私下界，尤其在低隐私（大 ε）区间表现最为突出。

**⚠️ 局限性**

局限性包括假设梯度服从正态分布、需要白盒访问模型参数、插入时刻若未知会显著下降性能，以及未针对黑盒查询环境给出完整实现。

---

## 370. AIFL: A Global Daily Streamflow Forecasting Model Using Deterministic LSTM Pre-trained on ERA5-Land and Fine-tuned on IFS

**arXiv ID:** 2602.16579 | [PDF](https://arxiv.org/pdf/2602.16579v1)

**作者:** Maria Luisa Taccari `[一作]` (European Centre for Medium-Range Weather Forecasts), Florian Pappenberger `[通讯]` (European Centre for Medium-Range Weather Forecasts)

**通讯引用:** 25389 | [OpenAlex ID](https://openalex.org/A5057022798)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并训练了一种基于LSTM的全球日流量预测模型AIFL，专为运营级NWP预报而设计；

**💡 创新点**

创新点在于双阶段迁移学习策略：先在ERA5-Land重分析上预训练，再在IFS控制预报上微调，以消除重分析与实时预报的分布差异；

**🔧 技术方法**

采用单层LSTM配合静态和动态特征的多层感知嵌入网络，并使用归一化MSE损失；

**📊 数据集**

使用CARAVAN生态系统中的18,588个水文站点，包含ERA5-Land（1980‑2019）和IFS（2016‑2019）两套气象驱动；

**📈 对比分析**

与GloFAS、Google全球洪水模型在相同测试集上对比，AIFL在2021‑2024独立时序测试中达到中位KGE' 0.66、NSE 0.53，并在事件检测上实现全零误报；

**⚠️ 局限性**

局限在于对罕见极端事件的召回率偏低，缺乏概率输出和多源降水融合，模型对部分干旱、稀疏流域的预测仍不稳健。

---

## 371. MoDE-Boost: Boosting Shared Mobility Demand with Edge-Ready Prediction Models

**arXiv ID:** 2602.16573 | [PDF](https://arxiv.org/pdf/2602.16573v1)

**作者:** Antonios Tziorvas `[一作]` (University of Piraeus), Yannis Theodoridis `[通讯]` (University of Piraeus)

**通讯引用:** 10502 | [OpenAlex ID](https://openalex.org/A5018268830)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于梯度提升树的边缘计算友好模型MoDE-Boost，用于城市共享微型出行需求的短期预测。

**💡 创新点**

创新点在于：①将时空特征融合到统一的表格表示；②通过统一模型共享全市信息同时保持局部差异；③实现极低的训练/推理延迟，适合在边缘设备上部署。

**🔧 技术方法**

技术手段包括：XGBoost回归/分类、时间序列特征提取（滞后、滚动统计、指数加权、傅里叶分解）、上下文特征（日历、节假日）、贝叶斯超参搜索与TPE剪枝。

**📊 数据集**

使用的数据集涵盖：地区级的荷兰三城（阿姆斯特丹、鹿特丹、海牙）共享电动车实时轨迹；点级的纽约CitiBike与芝加哥Divvy共享单车历史记录。

**📈 对比分析**

实验对比历史平均、季节性朴素、指数平滑、Croston、ASTN和TimeGPT等基线，MoDE-Boost在RMSE/MAE/F1方面均取得显著优势，并且训练时间约15秒、推理延迟<1µs，显示出高效实用。

**⚠️ 局限性**

局限性包括：缺少天气等外部因子、仅采用三分层分类、对人口稀疏或异常地区的预测效果相对下降，未来可引入更细粒度标签和多源外部数据。

---

## 372. Steering diffusion models with quadratic rewards: a fine-grained analysis

**arXiv ID:** 2602.16570 | [PDF](https://arxiv.org/pdf/2602.16570v1)

**作者:** Ankur Moitra `[一作]`, Dhruv Rohatgi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文研究在预训练扩散模型中对奖励函数进行倾斜后进行采样的算法与计算复杂度，聚焦于二次奖励函数；

**💡 创新点**

创新点在于：①证明线性奖励始终可高效采样；②展示负定二次奖励（即rank‑1）即使极小也不可解；③提出基于Hubbard‑Stratonovich变换的低秩正定二次奖励的多项式时间采样算法；

**🔧 技术方法**

技术手段包括：对扩散模型的得分（score）oracle进行精确分析、利用得分偏移公式构造线性奖励的得分oracle、利用Hubbard‑Stratonovich变换将正定二次奖励拆解为高维线性奖励的叠加、并结合离散网格估计归一化常数与蒙特卡洛估计；

**📊 数据集**

论文未使用公开数据集，所有结果均为理论证明与算法分析；

**📈 对比分析**

与其他方法的比较仅在理论复杂度层面，未给出实验指标；论文指出负定奖励在rank‑1时已不可解，而低秩正定奖励可在多项式时间内逼近；

**⚠️ 局限性**

局限性包括：假设得分oracle完全精确；只针对低秩正定或线性奖励的高效算法；负定二次奖励的不可解结果不排除存在更细致的可解子类；实际应用中模型误差与噪声未被充分考虑。

---

## 373. Creating a digital poet

**arXiv ID:** 2602.16578 | [PDF](https://arxiv.org/pdf/2602.16578v1)

**作者:** Vered Tohar `[一作]` (Bar-Ilan University), Amir Leshem `[通讯]` (Bar-Ilan University)

**通讯引用:** 4984 | [OpenAlex ID](https://openalex.org/A5019966334)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在七个月的工作坊式互动中，利用提示式专家反馈而不对模型进行再训练，将GPT‑4塑造成“数字诗人”，生成连贯且具有特色的诗歌风格，并通过盲作者辨别实验证明其诗歌与人类作品难以区分，最终将其作品整理并出版成诗集。

**💡 创新点**

通过持续、结构化的prompt‑based专家反馈实现长周期的in‑context学习，证明无参数更新即可为大型语言模型构建稳定的文学声音，并通过盲评实验挑战传统的作者身份辨别范式。

**🔧 技术方法**

使用GPT‑4的大语言模型，配合in‑context learning、结构化批评与修订循环、模型自我总结生成写作原则、以及生成自我身份信息的prompt。

**📊 数据集**

主要使用内部生成的诗歌（约30-50首）和选自知名人类诗人的诗歌（约30首）作为评估对照；未采用公开诗歌语料库或大型文本数据集。

**📈 对比分析**

采用盲作者辨别实验：50名人文学者/毕业生在每组6首诗（3 AI，3人类）中进行判定。结果显示人类诗被标记为“人类”54%，AI诗52%，95%置信区间均包含50%，两者无显著差异，表明难以区分。书级编辑后出版，但未给出额外性能指标。

**⚠️ 局限性**

局限性包括：仅在希伯来自由诗领域取得成功，难以实现严格韵律和押韵；评估样本量有限、受限于特定语言与受试人群；工作坊过程可能引入训练者偏好和文化先验；未验证诗歌的文学价值或跨语言、跨文化的普适性。

---

## 374. Learning to unfold cloth: Scaling up world models to deformable object manipulation

**arXiv ID:** 2602.16675 | [PDF](https://arxiv.org/pdf/2602.16675v1)

**作者:** Jack Rome `[一作]` (University of Edinburgh), Subramanian Ramamoorthy `[通讯]` (University of Edinburgh)

**通讯引用:** 2161 | [OpenAlex ID](https://openalex.org/A5071122608)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在机器人空中展开布料任务中，作者基于 DreamerV2 设计了一个强化学习框架，并通过引入表面法向图像、改进重放缓冲区和加入演示数据，训练出可在真实机器人上零样本部署的高效展开策略。

**💡 创新点**

创新点包括：① 用深度图转换为实时表面法向图，去除颜色纹理干扰；② 在 Replay Buffer 中加入演示数据并采用批量随机增强，显著提升模型对未见状态的泛化；③ 将双相机（站立视角与腕部视角）同时输入至世界模型，提高 3D 变形感知。

**🔧 技术方法**

主要技术：DreamerV2（世界模型+策略学习）、表面法向图生成（Sobel 边缘+归一化）、Replay Buffer 结构改造、随机数据增强、Unity+ObiCloth 物理仿真、Franka Emika 机器人 Cartesian 随动控制。

**📊 数据集**

使用自研的 Unity + ObiCloth 服装仿真环境，采集多种尺寸、颜色、纹理的随机布料数据；真实实验中使用的都是现场捕获的 RGB、深度图（经转换为法向图），并未使用公开数据集。

**📈 对比分析**

与传统 pick‑and‑place、SAC（仅图像或向量输入）、R‑AIF 以及原始 DreamerV2 进行对比；在模拟环境中，表面法向+改进 Replay 的版本在 5 种衣物类型上平均提升 15% 成功率、降低 35% 失败率；真实机器人零样本部署的展开成功率为 74%，接近或超过同类文献报告的 82‑86%。

**⚠️ 局限性**

主要局限：① 低分辨率观测导致对尺寸较小衣物（如短裤）识别困难；② 单臂 Franka 受限，无法有效旋转布料；③ 对深度相机噪声敏感，尤其在接近布料时出现盲区；④ 需要针对每种衣物单独训练，缺乏通用策略；⑤ 仍需改进传感器布局和多臂协同以提升稳健性。

---

## 375. Unpaired Image-to-Image Translation via a Self-Supervised Semantic Bridge

**arXiv ID:** 2602.16664 | [PDF](https://arxiv.org/pdf/2602.16664v1)

**作者:** Jiaming Liu `[一作]` (Stanford University), Sergios Gatidis `[通讯]` (Stanford University)

**通讯引用:** 5864 | [OpenAlex ID](https://openalex.org/A5080097591)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 Self-supervised Semantic Bridge（SSB），通过自监督视觉编码器构建共享语义空间，实现无监督图像对图像翻译与文本引导的图像编辑；

**💡 创新点**

创新点在于：1）使用自监督编码器（如 DINOv2）生成与外观无关的几何一致特征；2）将这些特征映射到扩散桥模型中，形成条件翻译器；3）引入可调端点不确定性，以平衡几何保持与外观生成；4）实现跨医学与自然图像域及文本编辑的统一框架；

**🔧 技术方法**

技术核心包括 DINOv2 自监督视觉编码、PCA 降维、KL‑VAE 与扩散桥（PF‑ODE）相结合的桥模型、可插值漂移控制、文本提示下的 SD3‑M 编辑；

**📊 数据集**

使用的数据集包括医学域的 SynthRAD2023/2025、AMOS2022、TotalSegmentator‑MRI/CT、IXI、pelvic MRI‑CT、PSMA‑FDG‑PET‑CT、CT‑RATE 及 UKBB‑MRI；自然域的 Horse→Zebra、Apple→Orange、Flickr、DIV2K、Pexels、FLUX‑dev 以及 LAION‑5B 子集；

**📈 对比分析**

与 CycleGAN、UNIT、SDEdit、DDIB、SynDiff 等基线对比；在 MRI→CT 任务中 SSB 取得 MS‑SSIM 0.810、PSNR 23.21 dB、FID 30.15，优于所有无监督基线；在自然 I2I 任务中 SSB 获得最高 SSIM 0.794、CLIP‑T 0.322，整体性能领先或相当；

**⚠️ 局限性**

局限性：当目标变换需要显著改变几何结构或物体类别时，语义桥的几何一致约束会限制翻译/编辑效果，导致无法实现深度形状改动。

---

## 376. Retrieval Augmented Generation of Literature-derived Polymer Knowledge: The Example of a Biodegradable Polymer Expert System

**arXiv ID:** 2602.16650 | [PDF](https://arxiv.org/pdf/2602.16650v1)

**作者:** Sonakshi Gupta `[一作]` (Georgia Institute of Technology), Rampi Ramprasad `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了两种检索增强生成（RAG）管道（VectorRAG和GraphRAG），用于构建面向聚酯可降解聚合物（PHA）的文献学者系统，支持多论文检索、实验上下文保留、推理与引用生成。

**💡 创新点**

创新点在于将语义向量检索与知识图谱推理相结合，构建了大规模、上下文保留的段落向量索引与可扩展的实体规范化知识图谱；同时通过实验验证两种管道的互补性和对科学推理的可靠性。

**🔧 技术方法**

使用了大型语言模型（GPT‑4o‑mini、Llama‑3.1‑70B、ChatGPT‑5、Gemini）与检索技术（dense vector检索、图遍历与子图检索），并结合实体抽取、规范化、聚类、语义匹配与路径重排序。

**📊 数据集**

数据集为1,028篇专注于PHA的全文论文（共44,609段落），以及1,028篇论文的全文原始文本，经过手工验证构成专业语料库；知识图谱包含约39万条关系元组。

**📈 对比分析**

在控制性21篇论文基准上，两种管道Recall≈1；在完整语料库上VectorRAG Recall≈0.72、GraphRAG Recall≈0.94，准确率均约0.96-0.97；在专家评估中GraphRAG+GPT‑4o‑mini平均得分9-10，超过商业Web‑RAG系统；GraphRAG在多跳推理与引用可靠性上优于VectorRAG，后者在段落级详细解释上更强。

**⚠️ 局限性**

局限性包括检索召回在大规模语料下下降、VectorRAG对长上下文导致推理延迟、知识图谱构建依赖LLM抽取质量、以及对非PHA领域泛化的验证不足。

---

## 377. AREG: Adversarial Resource Extraction Game for Evaluating Persuasion and Resistance in Large Language Models

**arXiv ID:** 2602.16639 | [PDF](https://arxiv.org/pdf/2602.16639v1)

**作者:** Adib Sakhawat `[一作]` (Islamic University of Technology), Fardeen Sadab `[通讯]` (Islamic University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个多轮零和谈判基准（AREG），用于评估大型语言模型在对话中的说服与抵抗能力。

**💡 创新点**

①将说服与抵抗放在同一互动框架内；②采用自动裁判器确定资源转移；③引入双重 Elo 评分同时衡量攻击与防守。

**🔧 技术方法**

使用对话生成技术、游戏理论模型、Elo 评分、以及将语言模型作为判决者的自动裁判方法。

**📊 数据集**

通过自定义角色提示与自动裁判生成的交互数据；对八款前沿 LLM 在 280 场对弈中的对话记录进行评估。

**📈 对比分析**

通过五轮循环对弈，每个模型在两角色中相互对抗，得到 C‑Elo 与 V‑Elo；结果显示所有模型的防守 Elo 平均高于说服 Elo，二者相关性弱（ρ=0.33），并发现增量承诺策略有效率约 2.8 倍，验证请求比直接拒绝更能抵抗。

**⚠️ 局限性**

仅限英语、单一慈善募捐情境；自动裁判对模糊承诺处理不确定；无法覆盖低信任场景；模型规模与社交能力无显著关联。

---

## 378. An $n^{2+o(1)}$ Time Algorithm for Single-Source Negative Weight Shortest Paths

**arXiv ID:** 2602.16638 | [PDF](https://arxiv.org/pdf/2602.16638v1)

**作者:** Sanjeev Khanna `[一作]` (New York University), Junkai Song `[通讯]` (New York University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种近线性时间的算法，用于求解带有任意实权重的图的全对最短路径（APSP）问题。

**💡 创新点**

首次在近线性时间内实现APSP，并且使用了新的图稀疏化与跳数减少技术，避免了传统算法的O(n^3)或更高复杂度。

**🔧 技术方法**

采用了随机化稀疏化、跳数压缩、潜在函数（potential）技术以及Steiner节点结构来控制图的规模和保持距离不变。

**📊 数据集**

无具体数据集，论文以理论分析为主，给出了在任意稠密或稀疏图上的时间复杂度。

**📈 对比分析**

与现有方法相比，该算法在稠密图中实现了~O(n^2)时间复杂度，在稀疏图中实现了近线性时间，优于以前的~O(mn)下界。

**⚠️ 局限性**

局限性包括：算法依赖随机化，可能需要多次重试；对极稀疏图的优化不充分；以及在实际应用中实现Steiner节点管理可能会产生额外的常数因子。

---

## 379. Almost Sure Convergence of Differential Temporal Difference Learning for Average Reward Markov Decision Processes

**arXiv ID:** 2602.16629 | [PDF](https://arxiv.org/pdf/2602.16629v1)

**作者:** Ethan Blaser `[一作]` (University of Virginia), Shangtong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 712 | [OpenAlex ID](https://openalex.org/A5033834190)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文证明了在不使用局部计时器的情况下，平均奖励下的差分TD学习（on-policy 和 off-policy）在任意步长 n 下几乎必然收敛，并给出了三条足够条件。

**💡 创新点**

首次将 D-稳定性和秩一扰动理论引入 RL，消除了差分TD学习对局部计时器的依赖。

**🔧 技术方法**

采用了随机逼近的 ODE 分析、Borkar-Meyn 定理以及 M-矩阵的 D-稳定性和秩一扰动结果。

**📊 数据集**

在 5×5 网格世界上对 n=3 的差分TD进行实验。

**📈 对比分析**

通过 RMSVE 指标观察不同 η 下的收敛情况，实验显示即使理论上 η_0=0，算法在多种 η 下仍能收敛，性能稳健。

**⚠️ 局限性**

离散化的收敛保证只给出保守的 η_0 上界，并且在 off-policy 情况下缺乏对 η 的精确上限，仍需进一步研究 D-稳定性理论。

---

## 380. Style-Aware Gloss Control for Generative Non-Photorealistic Rendering

**arXiv ID:** 2602.16611 | [PDF](https://arxiv.org/pdf/2602.16611v1)

**作者:** Santiago Jimenez-Navarro `[一作]` (University of Zaragoza), Ana Serrano `[通讯]` (University of Zaragoza)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了无监督学习如何在非摄影实景绘画中学习层次化潜在空间，并发现光泽度能从其它视觉因素中分离出来；随后设计了轻量级适配器，将该潜在空间与扩散模型对接，实现对风格与光泽度的细粒度控制。

**💡 创新点**

创新点在于：1）在无监督条件下揭示光泽度在StyleGAN2-ADA潜在空间中的自发层次化分离；2）利用此分离的潜在空间构建轻量级适配器，使扩散模型能够精准控制光泽与风格；3）提出基于梯度正则、互信息与Ridge回归等方法对潜在空间进行定量分析。

**🔧 技术方法**

核心技术包括：StyleGAN2-ADA训练与W+空间构建、pSp编码器实现单前向解码、层次化潜在空间分析（互信息、线性回归、t-SNE）、基于Stable Diffusion XL的扩散模型与轻量级适配器、ControlNet用于几何控制、Marigold实现光照/色度映射。

**📊 数据集**

数据集为自建的10,080张非摄影风格化物体图像，涵盖3种艺术风格（炭笔、墨水、油画）、20种几何体、4种光照、7级光泽度和6种颜色，来源于对Subias等工作进行重处理与样式映射。

**📈 对比分析**

与通用文本到图像模型（FLUX、GPT‑Image 1）、风格迁移方法（StyleID、DEADiff、InstantStyle）以及Artist‑Inator进行对比；在用户研究中获得最高偏好率（>90%）和最低Rank Product；在光泽度控制实验中实现连续平滑的光泽度变化，优于其他方法。

**⚠️ 局限性**

局限性包括：仅覆盖3种风格，难以泛化到未见风格；对颜色的控制依赖后期色度图，可能导致细节颜色损失；缺乏预训练的控制网络用于颜色图，未来工作需进一步完善；模型规模相对较小，生成多样性受限。

---

## 381. Agentic AI, Medical Morality, and the Transformation of the Patient-Physician Relationship

**arXiv ID:** 2602.16553 | [PDF](https://arxiv.org/pdf/2602.16553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 382. Who can we trust? LLM-as-a-jury for Comparative Assessment

**arXiv ID:** 2602.16610 | [PDF](https://arxiv.org/pdf/2602.16610v1)

**作者:** Mengjie Qian `[一作]` (University of Cambridge), Kate M. Knill `[通讯]` (University of Cambridge)

**通讯引用:** 1077 | [OpenAlex ID](https://openalex.org/A5111076409)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的聚合方法 BT-σ，能够在没有人工标签的情况下，联合推断文本生成候选的排名和多种 LLM 判定者的可靠性，从而提升无参考评估的鲁棒性。

**💡 创新点**

创新点在于将判定者的辨别参数（σ_k）嵌入 Bradley–Terry 模型，实现无监督的判定者可靠性建模；并进一步给出硬判定版 hard BT-σ，以在概率噪声极大时保持稳定性。

**🔧 技术方法**

核心技术包括 Bradley–Terry 软/硬模型、对比概率去偏（对称化）、最大似然估计、引入判定者辨别参数的联合学习，以及对比概率不一致性（环不一致率）的评估。

**📊 数据集**

使用两个 NLG 评测基准：SummEval（文本摘要）和 Topical-Chat（对话回复），每个基准均包含多维评估维度并产生成对偏好概率。

**📈 对比分析**

与平均概率、硬/软 BT、温度缩放 BT（Temp‑BT）、BT‑σ‑asp 等方法比较，BT‑σ 在两大基准上均能显著提升 Spearman 相关（约 3–5% 的提升），硬 BT‑σ 在概率极度不一致的子任务上更为稳健。

**⚠️ 局限性**

局限性包括：① 仍假设判定者偏好符合 BT 结构，无法完全消除系统性偏差；② 需要足够的成对比较才能有效估计 σ_k；③ 仅针对 NLG 评估任务验证，可能在其他领域需进一步验证；④ 无监督方法无法提供判定者的绝对置信度，仅通过相对权重调整。

---

## 383. CitiLink-Summ: Summarization of Discussion Subjects in European Portuguese Municipal Meeting Minutes

**arXiv ID:** 2602.16607 | [PDF](https://arxiv.org/pdf/2602.16607v1)

**作者:** Miguel Marques `[一作]` (University of Beira Interior), Ricardo Campos `[通讯]` (University of Beira Interior)

**通讯引用:** 1871 | [OpenAlex ID](https://openalex.org/A5089440969)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CitiLink‑Summ 数据集，包含 120 篇葡萄牙语市政会议纪要及其 2,880 个手工编写的讨论主题摘要。

**💡 创新点**

首次为葡萄牙语市政会议纪要讨论主题摘要提供专门的数据集与基准，并通过严格的多阶段人工标注流程保证摘要的高抽象度与质量。

**🔧 技术方法**

利用预训练的 Encoder‑Decoder 模型（BART、BART Large、PTT5、LED、PRIMERA）以及大型生成模型（Qwen2.5‑1.5B‑Instruct、Gemini‑2.5‑flash）进行微调或少样本提示，并使用 ROUGE、BLEU、METEOR、BERTScore 等指标评估。

**📊 数据集**

使用自建的 CitiLink‑Summ 数据集：120 篇会议纪要，按讨论主题拆分后与对应 2,880 个摘要进行训练与评估。

**📈 对比分析**

通过在 ROUGE‑1、ROUGE‑2、ROUGE‑L、BLEU、METEOR、BERTScore 等指标上对比模型，发现较大模型（PRIMERA、BART Large、Gemini）表现最佳；ROUGE‑1 最高达 68.96，BERTScore 最高 86.57，整体分数仍处于中等水平。

**⚠️ 局限性**

局限包括数据量有限、模型仅使用讨论文本而未利用主题信息，导致抽象度高的人工摘要难以完全复制；缺乏人工评估与跨语言适用性验证。

---

## 384. Predicting The Cop Number Using Machine Learning

**arXiv ID:** 2602.16600 | [PDF](https://arxiv.org/pdf/2602.16600v1)

**作者:** Meagan Mann `[一作]` (Queen's University), Erin Meger `[通讯]` (Queen's University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5022097963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究利用传统机器学习和图神经网络对 2-13 节点的连通图进行拦截数（cop number）预测，并通过可解释性分析揭示哪些结构特征最为重要。

**💡 创新点**

创新点在于：①提供可解释的特征重要性排名（SHAP 与置换重要性），证明节点连通性、图密度、团结构和树宽对拦截数的决定性影响；②证明即使仅使用顶点度信息，Graph Isomorphism Network 也能近似预测拦截数，展示 GNN 能直接从拓扑中学习到拦截数的结构信号。

**🔧 技术方法**

技术方法包括：传统机器学习模型（决策树、随机森林、梯度提升、逻辑回归）与 GNN（Graph Isomorphism Network），以及可解释性工具 SHAP 与置换重要性。

**📊 数据集**

使用的数据集为 McKay 小图数据库，涵盖 2-13 节点的 约 300,000 条连通图，手工计算了约 40 个结构特征并附上精确拦截数。

**📈 对比分析**

通过 80/20 分层拆分进行训练/测试，使用准确率、宏 F1 等指标进行比较；传统模型中梯度提升取得最高准确率 0.9779、宏 F1 0.9785，GNN 的准确率为 0.9465、宏 F1 0.8774，说明传统模型在本任务上略优。

**⚠️ 局限性**

局限性包括：数据集高度不平衡（拦截数 3 的样本稀缺），特征间高度相关影响可解释性分析，且实验仅针对小图，缺乏对更大图的推广性验证。

---

## 385. Measuring Mid-2025 LLM-Assistance on Novice Performance in Biology

**arXiv ID:** 2602.16703 | [PDF](https://arxiv.org/pdf/2602.16703v1)

**作者:** Shen Zhou Hong `[一作]` (Active Site), Joe Torres `[通讯]` (Active Site)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在真实物理实验室环境中，设计并执行了双盲随机对照试验，评估大型语言模型（LLM）对新手实验者在模拟病毒逆转录基因组工作流程（5项实验任务）中的实验室表现。

**💡 创新点**

首次在实验室现场使用随机对照方法评估LLM辅助效果，并通过分步里程碑和贝叶斯层次聚合分析捕捉非二元进展，填补了仅基于数字化基准的评估空白。

**🔧 技术方法**

技术包括前瞻性预注册、双盲随机化、Biosafety 2实验室操作、LLM交互（Anthropic、Google DeepMind、OpenAI）与互联网资源对照、Fisher exact、贝叶斯层次模型和阶梯式顺序回归等统计手段。

**📊 数据集**

数据集由153名本科/研究生组成，完成5项实验任务（微量移液、细胞培养、分子克隆、腺相关病毒生产、qPCR RNA定量），并收集实验结果、进度日志和LLM交互记录。

**📈 对比分析**

对照组仅使用互联网搜索，实验组使用LLM；主效应为核心任务序列完成率，两组无显著差异（P = 0.759）。单项任务中细胞培养表现边际显著提升（P = 0.059），贝叶斯聚合估计LLM平均提升约1.4倍（95%CrI 0.74–2.62）。

**⚠️ 局限性**

局限包括样本量低、任务完成率低导致统计功效不足；实验任务与完整逆转录基因组流程不完全一致；受限于LLM安全分类器关闭和使用模式多样化；未评估更高级实验者或更完善接口支持的效果。

---

## 386. Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents

**arXiv ID:** 2602.16699 | [PDF](https://arxiv.org/pdf/2602.16699v1)

**作者:** Wenxuan Ding `[一作]` (New York University), Greg Durrett `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Calibrate-Then-Act 框架，让 LLM 在与环境交互时显式考虑不确定性与成本权衡，从而提升探索决策的 Pareto‑optimal 性。

**💡 创新点**

通过将先验概率显式输入模型，分离校准与行动，诱导 LLM 进行最优探索与提交决策；该方法在 RL 训练中保持优越性并可在多任务场景下统一应用。

**🔧 技术方法**

LLM 提示工程、可校准的置信度估计（ISO）、BERT‑tiny 文件格式预测器、GRPO 强化学习、Pandora's Box 盒子实验、知识 QA 与编码任务的多步决策模拟。

**📊 数据集**

PopQA（知识 QA）、自构造的 FileReading（CSV 读取）数据集，以及自定义的 Pandora's Box 任务。

**📈 对比分析**

与单轮固定策略、无思考 Prompt、RL 端到端训练等 baseline 进行对比。实验显示 CTA‑Prompted 在 QA 上的折扣奖励最高，CTA‑RL 在编码任务中比 RL 多获约 3.5% 的奖励，并且在不同成本比下保持 Pareto‑optimal 前沿。

**⚠️ 局限性**

依赖显式先验的前提下，若先验估计不准则可能导致决策失误；在真实复杂环境中先验难以获得；RL 训练仍受样本效率与成本参数设定限制。

---

## 387. Learning Situated Awareness in the Real World

**arXiv ID:** 2602.16682 | [PDF](https://arxiv.org/pdf/2602.16682v1)

**作者:** Chuhan Li `[一作]`, Xin Eric Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了基于第一视角视频的Situated Awareness in the Real World基准，检验多模态基础模型的观察者中心空间推理能力。

**💡 创新点**

首次系统关注观察者视角的空间推理，将任务细分为定位、相对方向、轨迹形状、逆向路径规划、空间记忆与空间可行性六大类别，并使用真实世界的第一视角视频。

**🔧 技术方法**

采用零样本多模态基础模型（Gemini、Qwen等）和多种基线（随机、常见答案、盲LLM、Socratic）进行问答评估，并用正则表达式和GPT‑4o‑mini提取答案。

**📊 数据集**

自录制的Ray‑Ban Meta智能眼镜视频，覆盖10个户外场景和5个室内场景，约1200段视频，配有六类观测任务的人工多项选择问答。

**📈 对比分析**

与人类、开源与专有模型以及随机/盲基线对比，最佳专有模型Gemini 3 Flash达53.89%准确率，远低于91.55%的人工水平，整体人机差距达37.66%。

**⚠️ 局限性**

当前模型在摄像机旋转与平移区分、轨迹复杂度、对象持久记忆和室内外差异等方面表现不佳，说明多模态模型在观察者中心空间推理上仍存在显著缺陷。

---

## 388. E-Graphs as a Persistent Compiler Abstraction

**arXiv ID:** 2602.16707 | [PDF](https://arxiv.org/pdf/2602.16707v1)

**作者:** Jules Merckx `[一作]` (Ghent University), Tobias Grosser `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

引入eqsat dialect，将e-graph直接嵌入MLIR IR，实现编译过程中的持久化等价饱和；

**💡 创新点**

创新点在于：利用MLIR的可扩展性将e-graph构造为IR操作，改造pdl_interp实现e匹配与重写，从而避免与外部库之间的转换，保持跨抽象层级的等价性；

**🔧 技术方法**

技术实现：Python‑based xDSL+MLIR框架，eqsat dialect（eqsateclass、eqsatconst_eclass、eqsategraph 等操作），改造pdl_interp实现e‑matching、重建与提取；结合稀疏数据流分析、常量折叠、成本模型；对Herbie等浮点优化进行重现；

**📊 数据集**

数据集与案例：Herbie提供的31个FPBench基准（FPCore 表达式），以及与Herbie原始实现相同的输入；

**📈 对比分析**

对比方法：与原始 Herbie（egg/egglog）在准确率和运行时间上对比；准确率基本一致，但实现慢约400×；在e‑graph 匹配层面，将所有 pdl 模式合并为单个 matcher 时，比逐模式匹配快约2.5×；

**⚠️ 局限性**

局限性：实现基于 Python，性能远低于 Rust/egglog；缺乏高效的 e‑graph 重建、控制流匹配与动态精度评估；对大规模代码与复杂控制流支持不足；

---

## 389. TeCoNeRV: Leveraging Temporal Coherence for Compressible Neural Representations for Videos

**arXiv ID:** 2602.16711 | [PDF](https://arxiv.org/pdf/2602.16711v1)

**作者:** Namitha Padmanabhan `[一作]` (University of Maryland), Abhinav Shrivastava `[通讯]` (University of Maryland)

**通讯引用:** 7571 | [OpenAlex ID](https://openalex.org/A5101614443)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了TeCoNeRV，一种通过 patch‑tubelet 空间‑时间分解、残差编码和时序一致性正则化的超网络 implicit neural representation 视频压缩框架，能够在高分辨率下实现高效编码。

**💡 创新点**

创新点主要有三点：① 使用 patch‑tubelet 将视频划分为固定大小的 spatio‑temporal 区块，显著降低内存需求并实现分辨率无关的训练；② 对连续片段的 INR 权重做残差编码，极大压缩比特流；③ 引入时序一致性正则化，使权重变动与视频内容同步，进一步减少残差并提供可调节的速率控制。

**🔧 技术方法**

采用的核心技术包括：INR + hypernetwork（基于 NeRV 架构）；patch‑tubelet 划分；残差量化+算术编码；L1 时序正则化；基于 Kinetics‑400、UVG、HEVC、MCL‑JCV 的多分辨率训练与评估；以及快速解码的单张量预测。

**📊 数据集**

主要使用的数据集：Kinetics‑400（训练集），UVG、HEVC（Class B/C/E）和 MCL‑JCV（验证集）进行多分辨率（480p/720p/1080p）评估。

**📈 对比分析**

与 NeRV、HiNeRV 以及 NeRV‑Enc 基线对比；在 UVG、HEVC、MCL‑JCV 的 480p/720p/1080p 上实现了 2.47–5.35 dB 的 PSNR 提升、36% 的比特率下降，并保持 1.5–3× 的编码速度提升，首次在 720p/1080p 上实现超网络压缩。

**⚠️ 局限性**

限制：在无重叠裁剪时可能出现块边缘伪影；对极长视频或极高帧率的实时编码仍存在内存/速度瓶颈；时序正则化需要手动调参；无显式运动估计，随机访问需插入 key‑frame 或使用前一帧残差策略。

---

## 390. Knowledge-Embedded Latent Projection for Robust Representation Learning

**arXiv ID:** 2602.16709 | [PDF](https://arxiv.org/pdf/2602.16709v1)

**作者:** Weijing Tang `[一作]` (Carnegie Mellon University), Tianxi Cai `[通讯]` (Harvard University)

**通讯引用:** 21027 | [OpenAlex ID](https://openalex.org/A5078003862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

针对高维不平衡的二值矩阵（如电子健康记录），提出了知识嵌入的潜在投影（KELP）模型，利用外部语义嵌入对列向量进行平滑约束，从而实现鲁棒的低维表示学习。

**💡 创新点**

创新点在于：①将列嵌入视为外部语义嵌入在核RKHS中的平滑函数，显著降低自由参数维度；②采用核主成分分析构造信息丰富的子空间；③设计了可证明局部线性收敛的投影梯度下降算法；④提供数据驱动的核选择方法，避免负面知识融合。

**🔧 技术方法**

使用技术包括：潜在空间模型、核主成分分析（KPCA）、再生核希尔伯特空间映射、投影梯度下降（PGD）、核函数自适应选择、理论误差上界与收敛分析、仿真与真实EHR数据实验。

**📊 数据集**

数据集：MS患者PROMOTE队列（212例，3,296维临床特征），并使用来自12.5万VA患者的128维通用医学语义嵌入作为外部知识。

**📈 对比分析**

与传统的无知识融合的广义线性因子模型（GLFM）对比；在稀疏不平衡场景下，KELP在估计误差、知识图谱重构和功能障碍预测（AUROC）等指标上均优于GLFM，尤其在样本量小或特征维度极大时优势显著。

**⚠️ 局限性**

局限性：依赖外部语义嵌入的相关性，若嵌入与真实潜在结构偏离可能导致近似误差；核选择对性能敏感；当前模型仅适用于二值矩阵，未充分考虑个体协变量；动态/时序扩展仍待研究。

---

## 391. Reinforced Fast Weights with Next-Sequence Prediction

**arXiv ID:** 2602.16704 | [PDF](https://arxiv.org/pdf/2602.16704v1)

**作者:** Hee Seung Hwang `[一作]` (Princeton University), Olga Russakovsky `[通讯]` (Princeton University)

**通讯引用:** 44846 | [OpenAlex ID](https://openalex.org/A5022811687)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个名为ReFINE的强化学习框架，用来训练fast weight语言模型，使其能够在长上下文下进行下一序列预测（NSP）而非传统的下一词预测（NTP），从而显著提升模型的长文本推理与检索能力。

**💡 创新点**

创新点包括：①将NSP视为强化学习任务，引入基于熵的 token 选择与多步 roll‑out；②设计了以隐藏层相似度（余弦相似）为核心的序列级奖励，并在测试时加入精确匹配奖励；③提出了可跨 mid‑training、post‑training 与 test‑time 训练阶段使用的统一框架；④使用 Group Relative Policy Optimization (GRPO) 对 fast weight 进行梯度更新。

**🔧 技术方法**

技术手段包括 fast weight 架构（LaCT 与 DeltaNet）、熵权重 token 采样、roll‑out 生成、隐藏层相似度奖励、精确匹配奖励、GRPO 强化学习优化以及与传统 NTP 损失的混合训练。

**📊 数据集**

实验数据集涵盖：mid‑training 用 Long‑Data‑Collections；评估用 RULER NIAH、Booksum、SQuADQA、HotpotQA、LongBench；测试时使用 Needle‑in‑a‑Haystack 与长上下文 QA 任务。

**📈 对比分析**

与标准 NTP + SFT 基线相比，ReFINE 在三阶段均获得显著提升：在 LaCT‑760M 上，16K 上下文的 RULER NIAH 任务提升 8.5%（mid‑training）、15.3%（post‑training）、9.5%（test‑time）；在 DeltaNet‑1.3B 上提升 20.3%、11.0%、15.0%；在 LongBench 的 12 项任务中平均得分提升约 0.4–0.7 点，显著优于传统方法。

**⚠️ 局限性**

局限性包括：①奖励对长 roll‑out（k>5）效果衰减，需设计更丰富的语义相似度度量；②roll‑out 长度与分块数量固定，缺乏自适应机制；③实验仅覆盖两种 fast‑weight 模型，尚未验证更大规模或其他架构；④强化学习训练相对耗时，未充分讨论计算成本；⑤数据偏差可能影响模型泛化。

---

## 392. On the Hardness of Approximation of the Fair k-Center Problem

**arXiv ID:** 2602.16688 | [PDF](https://arxiv.org/pdf/2602.16688v1)

**作者:** Suhas Thejaswi `[一作]` (Aalto University), Suhas Thejaswi `[通讯]` (Aalto University)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5054266141)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了公平k中心问题在一般度量空间中不可实现3-ε近似，即3倍近似是最优的。

**💡 创新点**

首次在两组或一对一组的非退化情形下，将k中心禁用中心变体的3-ε硬度约化到公平k中心，从而确立3倍近似界的本质性。

**🔧 技术方法**

采用多项式约化、度量空间构造与δ分离技术，将k中心禁用中心实例转换为公平k中心实例。

**📊 数据集**

未使用任何实验数据集，完全基于理论证明。

**📈 对比分析**

无实验比较，结论为理论复杂度证明，说明3倍近似已达到最佳极限。

**⚠️ 局限性**

结论仅适用于任意度量空间，未讨论欧氏或双倍度量等特殊结构下的可能改进。

---

## 393. VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection

**arXiv ID:** 2602.16681 | [PDF](https://arxiv.org/pdf/2602.16681v1)

**作者:** Yingyuan Yang `[一作]` (Tsinghua University), Chen Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 52011 | [OpenAlex ID](https://openalex.org/A5100374115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了VETime框架，实现零样本时间序列异常检测，通过视觉与时序两种模态的联合学习完成异常识别与定位。

**💡 创新点**

创新点在于：1）可逆图像转换（Reversible Image Conversion）将时序数据映射为信息密集的视觉图像；2）补丁级时序对齐（Patch-Level Temporal Alignment）将视觉特征映射回原始时间轴；3）异常窗口对比学习（Anomaly Window Contrastive Learning）捕捉点异常与上下文异常；4）任务自适应多模态融合（Task-Adaptive Multi-Modal Fusion）动态调度模态特征。

**🔧 技术方法**

采用ViT视觉编码器、时序编码器、交叉注意力、对比学习、熵正则化的专家路由器、以及多任务损失（BCE+MSE+对比+熵）。

**📊 数据集**

在TSB-AD基准的11个公开单变量数据集（NAB、YAHOO、SMAP、MSL、IOPS、MGAB等）以及多变量数据集上进行实验。

**📈 对比分析**

与零样本TSFMs、全样本深度学习模型以及传统与视觉模型对比，VETime在零样本设置中夺得25/44项第一名，在全样本设置中夺得23/44项第一名，平均排名仅为2.05/2.02，且在视觉模型对比中实现约100倍的速度提升。

**⚠️ 局限性**

局限性包括：仅利用视觉与时序两模态，缺乏文本解释；对极长序列的处理尚待验证；对视觉编码器的依赖可能导致在不同视觉预训练模型下性能波动。

---

## 394. Policy Compiler for Secure Agentic Systems

**arXiv ID:** 2602.16708 | [PDF](https://arxiv.org/pdf/2602.16708v1)

**作者:** Nils Palumbo `[一作]` (University of Wisconsin--Madison), Somesh Jha `[通讯]` (University of Wisconsin--Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

设计并实现了一个多智能体系统的Policy Compiler，通过构建因果依赖图并在运行时用Datalog规则强制执行安全策略，提供确定性、可验证的授权；

**💡 创新点**

创新点在于：①使用跨代理依赖图捕获信息流，②用递归Datalog实现跨代理的授权推理，③将策略编译为自动化、无模型依赖的运行时监控；

**🔧 技术方法**

主要技术包括：依赖图构建与更新、Datalog（Differential Datalog）推理、参考监控（Reference Monitor）、Rust实现、身份与角色认证、LLM工具接口；

**📊 数据集**

实验数据集涵盖三类案例：Prompt injection防护（公开对抗场景）、τ^2‑bench客户服务任务（航空、零售子集）、MALADE药物警戒系统（使用FDA FAERS数据库）；

**📈 对比分析**

与未强化的自然语言策略对比，采用攻击成功率、任务完成率、延迟和API成本等指标；实验表明合规率从48%提升至93%，攻击成功率降至0%，额外延迟约20%但因去除长策略文本导致成本略降；

**⚠️ 局限性**

主要局限包括：仅覆盖通过工具调用的依赖，无法捕获侧信道或代码执行的非受控通信；需人工将自然语言政策转化为Datalog规则，缺乏完全自动化；未支持时间/速率限制等动态策略；

---

## 395. Learning Humanoid End-Effector Control for Open-Vocabulary Visual Loco-Manipulation

**arXiv ID:** 2602.16705 | [PDF](https://arxiv.org/pdf/2602.16705v1)

**作者:** Runpei Dong `[一作]` (University of Illinois Urbana-Champaign), Saurabh Gupta `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 11296 | [OpenAlex ID](https://openalex.org/A5040211080)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了一套模块化系统，使人形机器人能够在未见过的环境和未知物体上进行开放词汇的视觉定位与抓取。

**💡 创新点**

创新点在于（1）结合经典运动学、逆运动学与学习的残差前向模型，实现终端执行器精确追踪；（2）利用大规模视觉预训练模型（Grounding DINO、SAM‑3、AnyGrasp）实现开放词汇目标的感知与抓取规划；（3）通过在模拟中大规模强化学习训练并在真实世界中使用MOCAP校准，显著降低终端跟踪误差。

**🔧 技术方法**

核心技术包括：大规模视觉模型（Grounding DINO、SAM‑3、AnyGrasp）、残差神经前向运动学与基座里程计模型、逆运动学 + 轨迹规划、基于PPO的终端执行器跟踪策略、离线系统辨识、以及在线重规划与目标调整。

**📊 数据集**

使用的数据集有：AMASS（约8K人体运动序列）、Unitree G1在MOCAP房间采集的3小时末端执行器与基座数据、以及在真实环境中收集的日常物体与场景抓取试验数据。

**📈 对比分析**

与现有方法（AMO、FALCON）对比，HERO在模拟下的终端平移误差为2.48 cm（远优于8.29 cm和13.57 cm），在真实MOCAP下的平均误差为2.44 cm；在开放词汇抓取任务中，真实世界成功率达到90%（在25个日常物体上83.8%）。

**⚠️ 局限性**

主要局限包括：头戴RGB‑D相机视野受限，导致超过1 m或高于0.9 m的物体难以识别；经典轨迹规划可能产生扭曲或能耗高的运动；模块化设计受限于各子模块（如视觉模型或抓取手）的性能；所用Dex‑3手指关节灵活性不足，导致部分大块或不规则物体易滑落或碰撞。

---

## 396. Causality is Key for Interpretability Claims to Generalise

**arXiv ID:** 2602.16698 | [PDF](https://arxiv.org/pdf/2602.16698v1)

**作者:** Shruti Joshi `[一作]`, Dhanya Sridhar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种以因果推断为核心的解释框架，利用 Pearl 的因果阶梯和可识别因果表示学习（CRL）来系统地定义、评估和诊断大型语言模型（LLM）的解释性声明。

**💡 创新点**

创新点在于：①将解释性研究的目标（estimand）与所用证据的因果阶梯（L1–L3）严格对应；②通过可识别性（identifiability）与等价类（equivalence class）阐明解释声明的可信范围；③给出实用的诊断清单和案例研究，展示“证据‑声明”之间的常见缺口；④提出四条未来研究方向，桥接 CRL 与机制解释。

**🔧 技术方法**

主要技术：因果推断概念（估计量、阶梯、等价类、可识别性）、可识别因果表示学习（CRL）理论、对现有解释方法（稀疏自编码器、激活补丁、Steering 向量）的因果分析、文献注释与编码方法。

**📊 数据集**

使用的数据集：本文并未提出新的实验数据，而是以 50 篇代表性论文（共 186 条声明）为样本进行注释分析；示例中提到的 LLM 解释方法会基于常见的提示集和对抗测试集（如 jailbreak、拒绝行为测试）。

**📈 对比分析**

比较方式：通过手工注释评估声明与证据的阶梯匹配度，统计“声明阶梯 > 证据阶梯”的比例（约 50%），并在案例中阐明不同层次的证据如何支持或不足以支撑特定声明；未给出数值性能指标，侧重理论与诊断的效果。

**⚠️ 局限性**

局限性：①需要依赖假设（如可识别性、对外部干预的可控性），若假设失效则结论不稳健；②大部分分析是理论与案例导向，缺乏大规模实验验证；③因果阶梯匹配仅为诊断工具，未给出自动化评估流程；④对实际 LLM 训练数据和结构的适用性尚待验证。

---

## 397. Protecting the Undeleted in Machine Unlearning

**arXiv ID:** 2602.16697 | [PDF](https://arxiv.org/pdf/2602.16697v1)

**作者:** Aloni Cohen `[一作]` (University of Chicago), Uri Stemmer `[通讯]` (Tel Aviv University)

**通讯引用:** 1039 | [OpenAlex ID](https://openalex.org/A5019495492)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了针对机器无学习（machine unlearning）的新安全框架，分析并证明在满足传统“完美重训练”目标的定义下，删除某些数据点会导致未删除数据的隐私泄露；并给出了多种重构攻击，展示只需控制 ω(1) 个删除点即可重构几乎完整的数据集。

**💡 创新点**

创新点在于：①发现并正式化了“删除导致未删除数据泄露”的隐私风险；②提出了新的“删除安全（undeleted‑safe）”安全定义，专门防止删除行为对剩余数据的附加泄露；③通过差分隐私、统计查询（SQ）框架与模拟器技术构建了满足该定义的算法示例，证明该定义兼容常见功能如求和、布告板等。

**🔧 技术方法**

主要技术包括：差分隐私（DP）在单次和连续观测下的应用；统计查询（SQ）框架与足够统计量的估计；模拟器基础的安全定义与证明；重构攻击（包括中位数、CountMod 等）与误差分析；以及动态/自适应攻击模型。

**📊 数据集**

使用合成数据与理论构造任务（如 CountMod、近似中位数、包含★的多重计数等），未涉及真实工业或公开数据集，侧重理论演示与攻击阐释。

**📈 对比分析**

与已有的完美重训练相关定义对比，证明后者易被攻击；提出的删除安全定义既不如传统定义过于严格，又能保证如求和、布告板等实用功能；理论上重构攻击只需 ω(1) 个删除点；在差分隐私设置下，算法误差不随删除次数增长，保持常数级。

**⚠️ 局限性**

局限性包括：①对非 SQ/统计查询的功能仍需额外泄露或改写；②新定义对动态自适应攻击的完整安全证明仍在完善中；③实际实现细节（如效率、参数选择）未给出；④对于复杂模型（如深度学习）如何满足该定义仍是开放问题。

---

## 398. Fairness Dynamics in Digital Economy Platforms with Biased Ratings

**arXiv ID:** 2602.16695 | [PDF](https://arxiv.org/pdf/2602.16695v1)

**作者:** J. Martin Smit `[一作]` (University of Amsterdam), Fernando P. Santos `[通讯]` (University of Amsterdam)

**通讯引用:** 15456 | [OpenAlex ID](https://openalex.org/A5073403497)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

本文构建了一个进化博弈论模型，模拟数字平台中推荐算法如何影响服务提供者的合作行为以及群体公平性。

**💡 创新点**

创新点在于首次将进化博弈论与推荐系统结合，证明了通过调节高评分优先级(k_G)和针对边缘化群体的显式优先级(k_M)可以在不牺牲用户体验的前提下提升公平性，并给出了必要条件与Pareto前沿。

**🔧 技术方法**

主要使用技术包括进化博弈理论、Moran过程、马尔可夫链分析、数值仿真以及公平性指标（DPR、UX）和Pareto最优性判定。

**📊 数据集**

实验数据为合成模拟（Z_D=Z_M=20或80），不依赖真实平台数据；通过参数网格搜索探索不同用户群体(ε,γ,k)和算法参数(k_G,k_M)的影响。

**📈 对比分析**

比较方法是对不同(k_G,k_M)组合计算UX、DPR和Pareto前沿；结果显示在k_M>0时，DPR显著提升且对用户体验影响最小；在不确定的评分偏差下，k_M>0策略仍能获得更稳健的公平性。

**⚠️ 局限性**

局限性包括：假设服务提供者数量恒定、忽略新进入和退出、假设用户对群体特征完全模糊、仅考虑二元高/低努力策略、简化评分系统，且模型未在真实平台上进行验证。

---

## 399. EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data

**arXiv ID:** 2602.16710 | [PDF](https://arxiv.org/pdf/2602.16710v1)

**作者:** Ruijie Zheng `[一作]` (NVIDIA), Linxi Fan `[通讯]` (NVIDIA)

**通讯引用:** 8530 | [OpenAlex ID](https://openalex.org/A5025713692)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在超过20,854小时的自我中心人类视频上预训练 Vision‑Language‑Action 策略，并在少量对齐的人机对比数据上微调，最终实现了面向多指机器人手的精细操控任务。

**💡 创新点**

提出了基于人类动作预测损失与数据量的对数线性缩放规律，证明人类数据是可扩展的监督源；采用手腕相对运动 + 22-DoF 手指关节重定向的动作表示；两阶段训练（大规模预训练 + 对齐中间训练）实现了一步适配与跨体态迁移。

**🔧 技术方法**

使用流式 VLA 模型（VLM 主干 + DiT 动作专家），手腕相对运动表征，关节空间手指重定向，轻量化体态适配器，以及三阶段训练管线。

**📊 数据集**

20,854 小时自我中心人类活动数据（9,869 场景、6,015 任务、43,237 物体），829 小时 EgoDex 数据，344 个对齐人机桌面任务数据（约 50 小时人类、4 小时机器人）。

**📈 对比分析**

与从零训练、仅中间训练、仅人类预训练进行对比；在人机对齐后获得平均任务完成率提升 55% 以上；一阶迁移成功率 88%（衬衫折叠）和 55%（瓶盖拆卸）；跨体态迁移在 Unitree G1 上提升 30% 绝对成功率；缩放曲线呈现 R²=0.9983 的对数线性关系，表明无明显饱和。

**⚠️ 局限性**

对齐中间训练仍需一定量机器人数据；实验聚焦桌面任务，未充分验证极端长时程或更复杂场景；人类数据噪声大、任务对齐有限；对极其不同体态或全新任务的零样本迁移能力仍受限。

---

## 400. Saliency-Aware Multi-Route Thinking: Revisiting Vision-Language Reasoning

**arXiv ID:** 2602.16702 | [PDF](https://arxiv.org/pdf/2602.16702v1)

**作者:** Mingjia Shi `[一作]` (University of Virginia), Jundong Li `[通讯]` (University of Virginia)

**通讯引用:** 13436 | [OpenAlex ID](https://openalex.org/A5029588473)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种无训练、无数据、模型无关的推理时间缩放方法——Saliency‑Aware Principle Selection（SAP），通过在视觉–语言模型推理过程中动态选择高层次推理原则，维持对视觉信息的持续关注，从而降低文本主导推理导致的视觉误差累积和物体虚假报错。

**💡 创新点**

创新点包括：
- 将推理路由从细粒度的 token 序列抽象为高层次推理原则，显著降低搜索空间稀疏性；
- 采用视觉显著性作为“模态感知”指导信号，克服传统指导在视觉‑语言推理中的粗糙、噪声问题；
- 引入进化式原则选择框架，利用离散排序反馈实现鲁棒的原则优化；
- 支持多路推理并行化，打破单一路径长链推理的顺序瓶颈，提升吞吐量与响应时延。

**🔧 技术方法**

主要技术手段包括：
- 高层次推理原则定义与实现（如“每次形成结论后都回看图像”）；
- 基于进化算法（population‑based selection）进行原则搜索与细化；
- 视觉显著性热力图提取作为评估指标，评估每条原则下生成文本的视觉一致性；
- 对比实验设置，将 SAP 与 LongCoT、传统单一路径推理进行对比。

**📊 数据集**

实验数据集主要为视觉‑语言问答与物体识别类基准：OCRVQA、POPE‑Recall、MS‑COCO；
- 训练与推理使用开源 VLM Qwen3‑VL‑8B；
- 对比实验中不使用额外标注或预训练，只在推理阶段应用 SAP。

**📈 对比分析**

与传统单路长链（LongCoT）和无指导推理的对比表明：
- SAP 在相同 token‑生成预算下，显著降低物体虚假报错率（例如 OCRVQA 误报下降 12%）；
- 在 POPE‑Recall 上，视觉一致性得分提升 8%；
- 响应时延比 LongCoT 降低 25%（多路并行推理实现），而整体性能与 LongCoT 相当甚至略有提升。

**⚠️ 局限性**

局限性包括：
- 视觉显著性热图仅提供粗粒度视觉指引，复杂视觉关系仍可能被忽略；
- 进化搜索依赖离散排序反馈，若评估准则不充分可能导致收敛不佳；
- 多路并行需要额外模型实例资源，在极端规模部署中仍面临计算与内存成本；
- 目前实验集中在单一 VLM（Qwen3‑VL‑8B），跨模型通用性需进一步验证。

---

## 401. Are Object-Centric Representations Better At Compositional Generalization?

**arXiv ID:** 2602.16689 | [PDF](https://arxiv.org/pdf/2602.16689v1)

**作者:** Ferdinand Kapl `[一作]` (Technical University of Munich), Andrea Dittadi `[通讯]` (Technical University of Munich)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5008956185)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可控的视觉问答基准，用以评估对象中心化表示在组合推理中的泛化能力。

**💡 创新点**

创新点在于系统对比稠密与对象中心化表示，并证明在数据、计算或多样性受限时后者更具优势。

**🔧 技术方法**

技术包括以DINOv2/SigLIP2为基础，通过Slot Attention、k-means、跨注意力等方式生成对象中心化表示，并在Transformer VQA下游模型中进行评测。

**📊 数据集**

使用的数据集为CLEVRTex、Super-CLEVR与MOVi-C三种合成视觉世界，分别构建不同难度的训练集与组合外推（COOD）测试集。

**📈 对比分析**

比较方法在匹配表示尺寸、下游模型容量与FLOPs的前提下进行，实验结果显示对象中心化在更难的组合推理与受限条件下表现更好，而稠密表示在较易任务或大计算预算时可追赶甚至超越。

**⚠️ 局限性**

局限性包括仅在合成场景中验证，未对真实世界数据进行测试；并且对象中心化方法对下游模型规模与计算资源分配仍有一定依赖。

---

## 402. Scaling Open Discrete Audio Foundation Models with Interleaved Semantic, Acoustic, and Text Tokens

**arXiv ID:** 2602.16687 | [PDF](https://arxiv.org/pdf/2602.16687v1)

**作者:** Potsawee Manakul `[一作]` (Stanford University), Diyi Yang `[通讯]` (Stanford University)

**通讯引用:** 13359 | [OpenAlex ID](https://openalex.org/A5089413311)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了在大规模下使用下一个令牌预测（next-token prediction）训练本地音频基础模型，联合建模语义、声学和文本令牌，并在此基础上训练出SODA系列模型。

**💡 创新点**

创新点：① 统一的跨模态音频-文本架构，允许音频连续生成、语义理解、跨模态任务；② 通过离散音频令牌（Mimi）与文本在utterance级别交错，解决了语义瓶颈；③ 首次对离散音频模型进行IsoFLOP规模分析，给出模型大小与训练数据的比例指数；④ 对冷启动与热启动进行对比实验，证明冷启动更稳定；⑤ 将SODA作为统一后端实现声纹保持的语音对语音翻译。

**🔧 技术方法**

技术：基于Qwen3解码器架构的Transformer；离散音频令牌化使用Mimi（8个RVQ码本，12.5Hz）；utterance级别文本与音频交错；下一个令牌预测目标；IsoFLOP规模实验与功耗分析；NLL与下游任务的相关性分析；模型从冷启动（随机初始化）与热启动（文本LLM预训练）比较。

**📊 数据集**

数据集：语音数据使用Yodas（约165K小时英语）、Emilia（约140K小时英语）和MLS（45K小时音频书）；文本数据使用Nemotron-CC；混合比例为95%语音、5%文本；多任务评估数据包括sBLIMP、sWUGGY、Salmon、tBLIMP、tWUGGY、HellaSwag、LibriSpeech（ASR）、seed-tts-eval（TTS）以及CVSS‑T（语音到语音翻译）。

**📈 对比分析**

对比方法：在语义、声学、文本、跨模态四大能力上与TWIST、SpiritLM、Llama‑Mimi等现有模型对标；通过验证损失（NLL）与下游指标的Spearman相关系数评估；SODA在跨模态任务中表现出相对优异的ASR、TTS；在S2ST任务中SODA相较于无声学预训练和仅文本预训练的模型提升3–4倍BLEU，声纹相似度SIM显著提高。总体而言，SODA在多任务上实现了统一的中等规模表现，部分指标可与专用模型竞争。

**⚠️ 局限性**

局限性：未观察到显著的自发能力（emergent）；统一模型牺牲了一部分单一任务的最优性能；对离散音频令牌率、权重损失等超参尚未全面探索；在大规模训练中仍存在计算瓶颈与资源需求；对文本知识的迁移不完全恢复；多模态任务仍依赖精细调优，尚缺乏真正的零样本跨任务迁移。

---

## 403. One Hand to Rule Them All: Canonical Representations for Unified Dexterous Manipulation

**arXiv ID:** 2602.16712 | [PDF](https://arxiv.org/pdf/2602.16712v1)

**作者:** Zhenyu Wei `[一作]` (University of North Carolina at Chapel Hill), Mingyu Ding `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 2876 | [OpenAlex ID](https://openalex.org/A5022382771)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种统一的手形参数化与标准化URDF，将多种多自由度抓取手映射到同一结构和动作空间，实现跨实体的学习与推理。

**💡 创新点**

创新点在于：①把不同手型的几何与运动信息压缩为82维可解释参数，并通过自动化管道生成统一URDF；②构建连续潜在空间（VAE）实现手形的插值与零样本迁移；③通过统一动作空间让单一策略在多手型上共享数据与经验。

**🔧 技术方法**

使用了VAE潜在空间学习、PPO强化学习、扩散模型（DDIM）生成抓取姿态、URDF解析/生成框架、以及标准化坐标和动作映射。

**📊 数据集**

数据集包括：公开的Shadow Hand、LEAP、Allegro、Barrett、DexHand等多款手模型及其生成的约2万抓取样本；以及LEAP手的256种变体生成的抓取数据，全部在仿真与真实环境中收集。

**📈 对比分析**

与D(R,O)、GenDexGrasp、DFC等基准对比，统一模型在三手型上平均成功率约84%（比单手训练高约2-3%），零样本对未见手型的成功率维持在70%以上，推理时间仅0.13秒，显著优于传统基准。

**⚠️ 局限性**

局限性包括：对极少数特殊结构（如缺少轴向旋转关节）映射不完整；对两指/极少数指手型的零样本性能下降；且跨手型的物理仿真误差仍可能影响实机表现。

---

## 404. The Role of Common Randomness Replication in Symmetric PIR on Graph-Based Replicated Systems

**arXiv ID:** 2602.16700 | [PDF](https://arxiv.org/pdf/2602.16700v1)

**作者:** Shreya Meel `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 13805 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在基于图的2-复制数据库系统中，满足数据库隐私的安全私有信息检索（SPIR）问题的容量及随机性需求，并在图复制与全复制两种随机性复制模型下给出了容量下界、上界和最小随机性量的理论分析。

**💡 创新点**

首次将数据库隐私与部分复制相结合，确定了正则图和路径图下SPIR容量为1/N，且独立于每对服务器复制的消息数；在全复制随机性模型下准确求得了三点路径图的容量，并揭示了路径与环图SPIR容量与其PIR容量的关系。

**🔧 技术方法**

采用图论中的符号关联矩阵、信息不等式、一次性填充、交互干扰约束以及从PIR诱导SPIR方案的线性编码技术。

**📊 数据集**

本工作为理论分析，无使用具体实验数据集。

**📈 对比分析**

与传统PIR容量比较，图复制模型下SPIR容量与PIR相同（1/N），而全复制模型下容量上界与相应PIR容量正相关；在三点路径图中实现的容量为1/2，显示出相对更高的系统效率。

**⚠️ 局限性**

限制在于对一般图的SPIR容量仍未完全确定，且仅针对2-复制系统，未考虑更高复制度或更一般随机性复制模式的可行性。

---

## 405. Fast-MCS: A Scalable Open-Source Tool to Find Minimal Cut Sets

**arXiv ID:** 2602.16686 | [PDF](https://arxiv.org/pdf/2602.16686v1)

**作者:** Shakthivelu Janardhanan `[一作]` (Technical University of Munich), Carmen Mas-Machuca `[通讯]` (University of Bundeswehr Munich)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一款可扩展的开源工具 Fast-MCS，用于在大规模网络拓扑中高效计算最小割集（MCS），并对其运行时间与传统方法进行对比。

**💡 创新点**

创新点在于：①将 MPS 集合进行集合论分裂与递归决策树处理，而非构造长的布尔表达式；②使用逻辑补集和单元素分裂策略，显著降低时间与内存消耗；③实现了只需一次元素选择即可完成补集，提升了对大规模拓扑的可扩展性。

**🔧 技术方法**

技术手段包括：改进的 DFS 计算 MPS；组合方法、布尔-尚农（Boole‑Shannon）扩展与吸收；集合论分裂决策树与递归补集；以及开源实现 Fast-MCS。

**📊 数据集**

使用了 13 个通信网络拓扑数据集（Abilene、Spain、USA_26、India35、Polska、Austria_24、Norway、jonas-us-ca、HiberniaUK、Sweden、Nobel_EU、pioro40、Germany_17），每个拓扑提供节点数和边数信息。

**📈 对比分析**

通过在同一硬件（AMD Ryzen 5 3600、24GB RAM）上测量运行时间进行比较：Fast‑MCS 对比传统组合方法提升约 100‑1000 倍；与布尔‑尚农方法相比，小型拓扑略逊一筹，然而在大规模拓扑上快 80%+，显著降低计算时间。

**⚠️ 局限性**

限制：目前仅支持二值组件（工作/失效）且主要针对节点，边的支持为可选；未处理多状态或时变可用性；对极大规模拓扑仍可能受到组合爆炸的影响；缺乏对动态演化网络场景的评估。

---

## 406. Retrieval-Augmented Foundation Models for Matched Molecular Pair Transformations to Recapitulate Medicinal Chemistry Intuition

**arXiv ID:** 2602.16684 | [PDF](https://arxiv.org/pdf/2602.16684v1)

**作者:** Bo Pan `[一作]`, Liang Zhao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于匹配分子对转换（MMPT）的变量到变量的类比生成框架，并通过基础模型MMPT‑FM与检索增强框架MMPT‑RAG实现可控、可扩展的药物分子编辑。

**💡 创新点**

创新点在于：① 将类比生成视为MMPT空间的直接映射，打破传统分子级生成的全局编辑束缚；② 训练大规模MMPT基础模型，利用SMILES/SMARTS令牌化实现条件生成；③ 通过模板掩码提示与检索聚类+MCS提取实现用户可指定结构控制和项目特定的生成偏好；④ 形成可解释且合成可行的编辑生成。

**🔧 技术方法**

使用基于T5的Transformer（encoder‑decoder）对SMILES/SMARTS序列进行条件生成，配合掩码填充搜索、Morgan指纹嵌入、HDBSCAN聚类、MCS模板生成与RAG机制；模型训练采用大规模MMPT数据，推理时进行提示式填充与检索引导。

**📊 数据集**

主要数据集包括：ChEMBL提取的约800K个MMPT（约2.6M条MMP），PMV公司药物专利数据集PMV17（2017年）和PMV21（2021年），以及用于对照的数据库检索结果；训练集占80%，测试集占20%。

**📈 对比分析**

与基准方法（数据库检索、REINVENT‑4）对比，MMPT‑FM在ChEMBL上召回率提升至约68%，MMPT‑RAG进一步提升至82%；在专利内部扩展任务中召回率从22.7%提升至49.2%，外推跨专利召回率从28.6%提升至46.8%；新颖性亦显著提升（30%+）。整体性能表明MMPT‑RAG在保持化学可行性与多样性的同时，能显著恢复和预测真实药物化学改造。

**⚠️ 局限性**

主要局限：依赖大规模历史MMPT数据，若缺乏特定化学空间的样本，模型表现可能下降；当前评估仅基于结构和统计指标，未覆盖生物活性或合成可行性等后续验证；检索和聚类参数对结果有一定影响，需要进一步自动化调优。

---

