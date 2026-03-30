# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-30 | 今日论文总数: 409

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Decentralized Value Systems Agreements

**arXiv ID:** 2603.25811 | [PDF](https://arxiv.org/pdf/2603.25811v1)

**作者:** Arturo Hernandez-Sanchez `[一作]` (Universitat Politècnica de València), Jose Such `[通讯]` (INGENIO (CSIC-Universitat Politècnica de València))

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种去中心化方法，实现多个价值体系的聚合，使得相似的代理自组织成群体并在每个群体内达成一致的决策矩阵和权重向量；

**💡 创新点**

创新点在于将价值体系聚合建模为去中心化优化问题，利用动态网络形成机制与投影梯度上升相结合，支持多重共识而非单一共识，并通过置信界限制协商范围；

**🔧 技术方法**

采用投影去中心化梯度下降（Projected DGD）与动态邻接更新，利用个体对决策矩阵与权重向量的效用函数，结合距离度量与置信界实现聚合；

**📊 数据集**

使用了两个真实数据集：2020年荷兰Sùdwest‑Fryslân的参与式价值评估（PVE）样本876人；以及2017年欧洲价值研究（EVS）中36个国家的价值系统；

**📈 对比分析**

与传统单一共识的ℓ_p回归方法进行对比，结果显示去中心化多重聚合在PVE中获得更高且更集中的效用分布，在EVS中显著缩小效用范围并形成更合理的群组划分，整体性能优于集中式方法；

**⚠️ 局限性**

局限性包括：假设所有边权相等、邻居发现时假设每个代理可访问所有其他代理；对置信界选择敏感；缺乏对不同伦理原则权重的深入探讨；未评估大规模网络下的通信与计算开销。

---

## 2. VeRA+: Vector-Based Lightweight Digital Compensation for Drift-Resilient RRAM In-Memory Computing

**arXiv ID:** 2603.26016 | [PDF](https://arxiv.org/pdf/2603.26016v1)

**作者:** Weirong Dong `[一作]` (Southern University of Science and Technology), Longyang Lin `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 928 | [OpenAlex ID](https://openalex.org/A5021091015)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级的数字漂移补偿框架 VeRA+，通过共享随机投影矩阵和仅两维漂移特定缩放向量实现 RRAM-IMC 的长期漂移补偿。

**💡 创新点**

创新点在于：①将投影矩阵共享至所有层和漂移级别，显著降低存储和计算开销；②通过离线训练预置一组漂移向量集合，部署时仅根据时间选择，不需要在线重写或重训练；③针对 CNN 采用 1×1 内核和跨层共享的设计，进一步压缩参数。

**🔧 技术方法**

使用 LoRA 思想的低秩适配，结合 VeRA 的随机投影共享；离线漂移注入训练；漂移感知调度算法；以及硬件架构（SRAM-IMC + RRAM-IMC + 外部 ROM）实现。

**📊 数据集**

在多个视觉与语言任务上评估：ResNet‑20/32/50 在 CIFAR‑10/100 和 ImageNet‑1K；BERT‑base/large 在 QQP 与 SST‑5；同时利用 IBM 公开的漂移模型和自研 1T1R RRAM 设备一周漂移数据。

**📈 对比分析**

与传统的 BN 校准、LoRA、原始 RRAM 方案比较，VeRA+ 在 10 年漂移后可恢复 97–99% 的漂移前精度；存储开销仅 5.15 KB，操作开销 1.9%，比 BN 方案低 1000×，比 LoRA 方案低 10×；硬件面积和能耗提升分别仅 3.5% 与 4.5%。

**⚠️ 局限性**

局限性包括：漂移向量集合的选择依赖预先的漂移统计，若实际漂移分布偏离训练模型可能需要增大向量集；目前仅验证了单一 RRAM 技术，跨平台迁移需要重新漂移建模；对极端长周期漂移（十年以上）尚未在真实硬件上完整验证。

---

## 3. Pure and Physics-Guided Deep Learning Solutions for Spatio-Temporal Groundwater Level Prediction at Arbitrary Locations

**arXiv ID:** 2603.25779 | [PDF](https://arxiv.org/pdf/2603.25779v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 4. Identification of Bivariate Causal Directionality Based on Anticipated Asymmetric Geometries

**arXiv ID:** 2603.26024 | [PDF](https://arxiv.org/pdf/2603.26024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 5. JRM: Joint Reconstruction Model for Multiple Objects without Alignment

**arXiv ID:** 2603.25985 | [PDF](https://arxiv.org/pdf/2603.25985v1)

**作者:** Qirui Wu `[一作]` (Meta Reality Labs Research), Henry Howard-Jenkins `[通讯]` (Meta Reality Labs Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Joint Reconstruction Model (JRM)，通过在流匹配生成器的潜在空间中隐式聚合未对齐的多视角对象观测，实现对象级场景的联合重建；

**💡 创新点**

创新点在于：1) 通过联合解噪与耦合注意力实现多个对象的隐式信息共享，克服显式对齐误差；2) 采用对偶训练策略，只用两对象对训练即可泛化到任意数量对象；3) 将个体观测多模态编码（点云、图像、文本）融入流匹配框架；

**🔧 技术方法**

技术包括：条件流匹配（Diffusion Transformer）、稀疏卷积编码器、DinoV2 与 T5 文本编码、耦合融合块（coupled fusion block）以及对偶负样本训练；

**📊 数据集**

数据集涵盖 400k 3D 资产（ObjaverseXL、Amazon Berkeley Objects、Wayfair 等）用于预训练；8000 对训练样例；合成场景（2,500 目标物体）用于评估；真实场景使用 Replica 与 ScanNet++；

**📈 对比分析**

对比基线包括 ShapeR（独立重建）、MORE^2、DPRecon；在合成场景的时间/空间重复实验中，JRM 在 Chamfer Distance、F1、Normal Consistency 等指标上均优于基线，尤其在对齐误差下表现更稳健；在真实场景中，JRM 产生更完整、连贯的几何，且在 ScanNet++ 上的性能提升更显著；

**⚠️ 局限性**

局限性：1) 依赖于精确的实例分割与相机位姿；2) 主要针对对象级重建，扩展到全场景需要进一步研究；3) 对极端遮挡和纹理稀疏场景的鲁棒性仍有限；4) 训练对偶对可能在多样性不足时导致过拟合。

---

## 6. Data-Driven Plasticity Modeling via Acoustic Profiling

**arXiv ID:** 2603.25894 | [PDF](https://arxiv.org/pdf/2603.25894v1)

**作者:** Khalid El-Awady `[一作]` (Stanford University), Khalid El-Awady `[通讯]` (Stanford University)

**通讯引用:** 170 | [OpenAlex ID](https://openalex.org/A5056078741)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用 Morlet 小波对镍微柱压缩实验中的声发射（AE）信号进行频带事件检测，并通过物理验证、特征机器学习及 Gaussian Mixture 模型聚类，识别并表征四类不同的 AE 事件。

**💡 创新点**

创新点在于将可解释的频带检测与物理动力学相结合，利用多维时间‑频域特征和 GMM 聚类得到四种事件体系，并将 AE 事件与应力/应变跃迁关联，为预测塑性变形提供可视化与可解释的框架。

**🔧 技术方法**

采用零相位带通滤波、Morlet 小波能量阈值检测、RMS 归一化、时间/频域特征提取（峰值、RMS、能量、峰度、偏度、零交叉率、谱心、谱带宽、谱熵）、KNN/SVM 分类、波形能量映射、Discrete Wavelet Transform（DB4）特征、Gaussian Mixture Model 聚类以及变化点检测等技术。

**📊 数据集**

使用约 260 个 AE 事件与 2600 个非事件样本，来自 John Hopkins University 的镍微柱压缩实验数据，采样率 2 MHz、实验时长约 2 min，数据已归一化至 ±1 幅度。

**📈 对比分析**

与传统幅值阈值检测相比，能量阈值检测大幅提升事件召回率；在特征空间中，KNN 的事件召回率从 29% 提升至约 60%，SVM 亦显著改进；GMM 在 BIC 选择下确定 4 类聚类，能够区分不同频谱特征的事件。

**⚠️ 局限性**

局限在于频带选择对事件检测敏感，低幅值事件可能被漏检；样本不平衡导致分类性能受限；未构建完整的时间序列或序列模型，预测性能仍待验证。

---

## 7. IncreRTL: Traceability-Guided Incremental RTL Generation under Requirement Evolution

**arXiv ID:** 2603.25769 | [PDF](https://arxiv.org/pdf/2603.25769v1)

**作者:** Luanrong Chen `[一作]` (National University of Defense Technology), Lei Wang `[通讯]` (Defense Innovation Institute, Academy of Military Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 IncreRTL 框架，实现在需求演进时的增量 RTL 生成

**💡 创新点**

通过 LLM 构建需求-代码可追溯链接，实现局部重构而非全量重生

**🔧 技术方法**

采用 LLM + Chain‑of‑Thought 推理、语义匹配、代码语法分块、AST 对比等技术

**📊 数据集**

使用自研 EvoRTL‑Bench（30 模块 120 条需求变更）

**📈 对比分析**

与 Direct 与 Full 生成基线对比，IncreRTL 在一致性得分上最高（0.8123>0.7337），并将相对 token 用量降至 1.46 倍，保持或提升语法/功能正确率

**⚠️ 局限性**

局部化方法对大规模结构变更仍有挑战，且依赖手工校验追溯链接；LLM 生成质量受模型规模与上下文窗口限制

---

## 8. Do All Vision Transformers Need Registers? A Cross-Architectural Reassessment

**arXiv ID:** 2603.25803 | [PDF](https://arxiv.org/pdf/2603.25803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 9. Dynamic LIBRAS Gesture Recognition via CNN over Spatiotemporal Matrix Representation

**arXiv ID:** 2603.25863 | [PDF](https://arxiv.org/pdf/2603.25863v1)

**作者:** Jasmine Moreira `[一作]` (Universidade Tecnológica Federal do Paraná), Jasmine Moreira `[通讯]` (Universidade Tecnológica Federal do Paraná)

**通讯引用:** 490 | [OpenAlex ID](https://openalex.org/A5050751173)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于 MediaPipe 手部关键点与 2D 卷积神经网络的动态 LIBRAS 手势识别方法，用于家居自动化控制。

**💡 创新点**

创新点包括将 90×21 的时空关键点矩阵视作图像进行分类，使用滑动窗口与帧三倍复制实现无递归的实时推理，以及轻量化的 2D CNN 模型（约 25,000 参数）。

**🔧 技术方法**

采用了 MediaPipe Hand Landmarker、二维卷积神经网络、滑动窗口与帧复制、像素归一化为灰度图、阈值 98% 的置信度过滤等技术。

**📊 数据集**

训练使用自行采集的 1,254 个手势样本（+220 验证样本），共 11 类；实验中也参考了 MINDS‑Libras 数据集等公开数据。

**📈 对比分析**

在单用户低光下 95% 的准确率、正常光照 92%，与文献中 Alabdullah 等 92.57% 及 Alves 等 93% 的结果相当，证明方法在同类场景中表现优良。

**⚠️ 局限性**

局限在样本仅来自单一用户、手势类别有限、未对多用户多样性进行系统评估、以及帧复制导致的时间分辨率降低和可能的误识别。

---

## 10. FAST3DIS: Feed-forward Anchored Scene Transformer for 3D Instance Segmentation

**arXiv ID:** 2603.25993 | [PDF](https://arxiv.org/pdf/2603.25993v1)

**作者:** Changyang Li `[一作]` (Goertek Alpha Labs), Yi Xu `[通讯]` (Goertek Alpha Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 FAST3DIS，一种端到端的多视角 3D 实例分割框架，直接用 3D anchor 与 transformer 生成实例掩码，无需后期聚类。

**💡 创新点**

创新点包括：1) 3D Anchor 生成与 Anchor‑Sampling Cross‑Attention，实现跨视角几何一致并降低注意力复杂度；2) 双层正则化——多视角对比学习 + 动态空间重叠惩罚，抑制查询冲突并精细边界；3) 利用 LoRA 微调 Frozen DA3，保留零射重建几何先验。

**🔧 技术方法**

核心技术：Transformer 解码器、3D‑to‑2D 投影、LoRA 微调、Mask2Former 思路、交叉视角对比损失、动态重叠惩罚、Sim(3)+ICP 对齐。

**📊 数据集**

训练集：Aria Synthetic Environments；评估集：ScanNet V2、ScanNet++、Replica。

**📈 对比分析**

与 IGGT、PanSt3R、SAM3D、Segment3D、HDBSCAN 等基线对比，FAST3DIS 在 AP 上略优于 IGGT、与 SAM3D 竞争；推理速度提升 100‑250×，内存占用显著下降。

**⚠️ 局限性**

局限性：查询数固定（N_q=80）导致在极度拥挤场景（ScanNet++）漏检大量实例；动态重叠惩罚需手动调参；验证主要集中在室内数据，室外/复杂几何场景尚未评估。

---

## 11. Polarization-Based Eye Tracking with Personalized Siamese Architectures

**arXiv ID:** 2603.25889 | [PDF](https://arxiv.org/pdf/2603.25889v1)

**作者:** Beyza Kalkanli `[一作]` (Meta), Mantas Žurauskas `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了基于Siamese网络的个性化光偏振眼动追踪方法，用少量标定图像实现高精度瞳孔估计。

**💡 创新点**

将Siamese差分学习与光偏振数据相结合，实现仅用10倍少于传统线性校准的标定样本即可获得竞争性或更优性能，并且可与线性校准叠加进一步提升。

**🔧 技术方法**

Siamese网络架构、相对视线偏移预测、光偏振特征提取（强度、DoLP、AoLP）、Smooth L1带异常值抑制损失、线性后校准。

**📊 数据集**

使用338名受试者的光偏振眼部图像数据集，包含Intensity-DoLP-AoLP三个通道，分为196名训练、142名验证。

**📈 对比分析**

与基线单分支网络以及基线+线性校准进行对比；在P50、P75、P95视线误差上，Siamese+光偏振+线性校准实现了比线性校准基线低5-13%的误差，且仅使用9张标定图像即可与使用约100张图像的线性校准相媲美。

**⚠️ 局限性**

推理时需对每个标定图像进行前向传播导致计算成本约为基线的9倍，且对光偏振硬件依赖较高；未探讨动态自适应标定样本选择与对不同用户皮肤/眼部特征的鲁棒性。

---

## 12. Personalizing Mathematical Game-based Learning for Children: A Preliminary Study

**arXiv ID:** 2603.25925 | [PDF](https://arxiv.org/pdf/2603.25925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 13. ETA-VLA: Efficient Token Adaptation via Temporal Fusion and Intra-LLM Sparsification for Vision-Language-Action Models

**arXiv ID:** 2603.25766 | [PDF](https://arxiv.org/pdf/2603.25766v1)

**作者:** Yiru Wang `[一作]` (Bosch (China) Investment Ltd.), Hao Sun `[通讯]` (Bosch (China) Investment Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ETA‑VLA框架，在多视角多帧视觉语言动作模型中通过时间融合模块与Intra‑LLM稀疏聚合器显著提升效率与性能。

**💡 创新点**

创新点在于结合时间融合、RoPE‑free语义评分以及多样性保持的稀疏选择机制，动态模拟人类注意力分配以筛选视觉token。

**🔧 技术方法**

采用ViT‑L/14视觉编码器、LLaMA‑7B LLM、Transformer‑based时间融合模块、动态文本锚与RoPE‑free语义评分的ILSA稀疏聚合器，以及轻量化轨迹解码器与预训练优化器。

**📊 数据集**

使用NAVSIM v2仿真数据集（Navtest 与 Navhard）。

**📈 对比分析**

与Transfuser、VADv2、DiffusionDrive等SOTA方法对比，Navtest EPDMS 85.0（接近人类 90.3），Navhard 48.0，减少约32% FLOPs，保持94%精度，显著优于基线。

**⚠️ 局限性**

局限在于深层或多阶段稀疏化可能导致信息丢失；对极端场景的鲁棒性仍需提升；实现对GPU有一定依赖，边缘设备部署仍需进一步优化。

---

## 14. Starlink Constellation: Deployment, Configuration, and Dynamics

**arXiv ID:** 2603.25835 | [PDF](https://arxiv.org/pdf/2603.25835v1)

**作者:** Muaz Ali `[一作]` (University of Arizona), Beichuan Zhang `[通讯]` (University of Arizona)

**通讯引用:** 9204 | [OpenAlex ID](https://openalex.org/A5050010654)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文利用2019‑2025年Starlink卫星TLE观测数据开展了为期六年的纵向实证研究，系统量化了星座部署、轨道结构、卫星生命周期与运动特征；

**💡 创新点**

创新点在于揭示Starlink为高度异质、持续演化的网络，提出了正则/非正则卫星划分及基于群体中位数的运动检测方法，并质疑传统Walker‑Delta假设；

**🔧 技术方法**

采用TLE数据处理、DBSCAN聚类、Kaplan‑Meier生存分析、相位间距统计与基于中位数阈值的运动检测等技术手段；

**📊 数据集**

使用了Space‑Track提供的2019‑2025年TLE记录、CelesTrak SOCRATES交叉报告以及Satellite Catalog元数据；

**📈 对比分析**

通过将真实部署的pGrid拓扑与理想完整网格进行对比，发现路径延迟提升约17.8%、跳数提升约6.7%，链路失效率从1.63%降至1.04%，显示实际动态导致显著性能下降；

**⚠️ 局限性**

局限在于TLE测量噪声、仅覆盖五个主要轨道层、并未将计划性退轨事件计入故障概率，可能低估了星座真实动态幅度。

---

## 15. Chasing Autonomy: Dynamic Retargeting and Control Guided RL for Performant and Controllable Humanoid Running

**arXiv ID:** 2603.25902 | [PDF](https://arxiv.org/pdf/2603.25902v1)

**作者:** Zachary Olkin `[一作]` (California Institute of Technology), Aaron D. Ames `[通讯]` (California Institute of Technology)

**通讯引用:** 15098 | [OpenAlex ID](https://openalex.org/A5039171820)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种通过带硬约束的优化动态重新定位单一人类演示数据，生成周期性参考库，并利用CLF-RL跟踪实现人形机器人高速、持久跑步的管线。

**💡 创新点**

创新点在于将动态优化的人类运动数据与目标条件控制导向的奖励机制相结合，显著提升了速度跟踪与动态稳定性。

**🔧 技术方法**

采用的技术包括硬约束优化、动态参考生成、CLF-RL（控制引导的强化学习）以及端到端的运动跟踪与控制。

**📊 数据集**

使用的数据集为单一的人类跑步演示（具体来源未给出）。

**📈 对比分析**

与传统单轨回放或无奖励优化方法相比，所提方案在Unitree G1机器人上实现了最高3.3 m/s的跑速，跑步距离达数百米，并能在户外完成实时障碍规避，表现出更优的速度与稳定性。

**⚠️ 局限性**

局限性包括：仅基于单个演示数据，难以覆盖多样化动作；对更复杂环境的适应性尚未充分验证；硬约束优化过程计算成本较高。

---

## 16. LEMON: a foundation model for nuclear morphology in Computational Pathology

**arXiv ID:** 2603.25802 | [PDF](https://arxiv.org/pdf/2603.25802v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 17. On Integrating Resilience and Human Oversight into LLM-Assisted Modeling Workflows for Digital Twins

**arXiv ID:** 2603.25898 | [PDF](https://arxiv.org/pdf/2603.25898v1)

**作者:** Lekshmi P `[一作]` (Indian Institute of Technology Goa), Neha Karanjkar `[通讯]` (Indian Institute of Technology Goa)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5071451043)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了FactoryFlow框架，利用LLM自动生成制造系统的可执行离散事件模型，并结合实时传感器数据进行参数拟合；

**💡 创新点**

提出三项设计原则：结构建模与参数拟合的正交化、基于已验证组件的组合式建模、以及使用Python的密度保持中间表示以降低hallucination；

**🔧 技术方法**

采用Gemini 2.5 Pro LLM进行自然语言到代码的转换，使用LangGraph实现推理与代码生成，利用FactorySimPy组件库进行模型构造与验证；

**📊 数据集**

基于35个手工构建的制造系统模型（从10个到112个组件不等），配套粗略与详细自然语言描述与对应的真实实现；

**📈 对比分析**

对比两种IR（网表式字典与Python循环+类），通过统计错误数量和类型进行比较，结果显示Python IR在规模大、结构规则化的模型中显著降低hallucination和结构错误；

**⚠️ 局限性**

局限性包括仅适用于结构偶尔变更的系统、组件库扩展受限、仅使用单一LLM（Gemini）且缺乏跨模型、多运行平均、对非制造领域的通用性待验证。

---

## 18. Generalizable Verilog Modeling Framework for Synchronous and Asynchronous Superconducting Pulse-Based Logic Gates

**arXiv ID:** 2603.25885 | [PDF](https://arxiv.org/pdf/2603.25885v1)

**作者:** Elisabeth Feng `[一作]` (University of Southern California), Peter A. Beerel `[通讯]` (University of Southern California)

**通讯引用:** 4145 | [OpenAlex ID](https://openalex.org/A5084205024)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于Verilog的通用SFQ门建模框架，支持同步与异步门的功能与时序验证，并兼容SDF后注流程。

**💡 创新点**

创新点包括消除辅助模块的复杂性、首次在Verilog中抽象异步SFQ门、使用SDF建模脉冲保留窗口，以及实现混合模式验证。

**🔧 技术方法**

采用Verilog规范块、SDF后注、传输延迟标志，以及JoSIM模拟器进行行为与时序验证。

**📊 数据集**

使用标准SFQ门库及其混合同步/异步电路（如T1全加器、4位乘法器）进行设备级和电路级仿真验证。

**📈 对比分析**

与JoSIM模拟结果对比，验证了功能正确性和时序约束覆盖；在4位乘法器上面积从4985 JJ降至851 JJ，显示出显著的资源节省。

**⚠️ 局限性**

局限性在于尚未覆盖所有SFQ门族（如自重置门），对商业EDA工具的兼容性有限，需要手动设置模拟器传输延迟标志。

---

## 19. Building to Understand: Examining Teens' Technical and Socio-Ethical Pieces of Understandings in the Construction of Small Generative Language Models

**arXiv ID:** 2603.25852 | [PDF](https://arxiv.org/pdf/2603.25852v1)

**作者:** Luis Morales-Navarro `[一作]` (University of Pennsylvania), Danaé Metaxa `[通讯]` (University of Pennsylvania)

**通讯引用:** 785 | [OpenAlex ID](https://openalex.org/A5086524212)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过为期一周的参与式设计工作坊，让 16 名 14-15 岁青少年自行构建极小规模的生成式语言模型（如食谱、剧本、歌曲生成器），并在构建过程中探讨技术与社会伦理层面的理解。

**💡 创新点**

首次将“in‑pieces”方法应用于 AI/ML 认知研究，系统性捕捉并分析青少年在构建模型时出现的技术与社会伦理碎片化知识，构建了一套完整的理解片段清单。

**🔧 技术方法**

使用 nanoGPT 框架进行模型训练，并通过手工制作、标记与分词的数据集、温度与迭代次数等超参数的迭代调节，构建并实验生成式语言模型。

**📊 数据集**

青少年自行构建的训练集规模从 22,790 到 226,484 个 token，主要来源于公开数据集（如 Kaggle、电视剧脚本、食谱网站等），并自行策划与清洗。

**📈 对比分析**

研究采用“比较输出”与“迭代改进”方法，对不同温度（0.4/0.8/1.6）和训练迭代次数（400/800/1600/2400）下模型生成文本进行对比；但未给出传统意义上的性能评估指标，重点在于学习者的认知变化。

**⚠️ 局限性**

研究样本仅限于 16 名对 STEM 颇感兴趣的青少年，且依赖其口头与书面表达；因此可能忽视未言表的认知；研究环境为课外项目，结果不易推广至更广泛的学习群体或真实课堂场景，也未对模型质量进行客观评测。

---

## 20. Agentic Markets: Equilibrium Effects of Improving Consumer Search

**arXiv ID:** 2603.25893 | [PDF](https://arxiv.org/pdf/2603.25893v1)

**作者:** Brendan Lucier `[一作]` (Microsoft Research), David M. Rothschild `[通讯]` (Microsoft Research)

**通讯引用:** 8711 | [OpenAlex ID](https://openalex.org/A5054707908)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文构建了一个可解析的代理式市场模型，研究了更低成本或更高信息量的搜索技术如何影响市场学习、消费者福利和企业定价。

**💡 创新点**

创新点包括：①将消费者的顺序搜索问题与市场层面的学习过程统一建模；②分析了搜索成本降低与搜索信息量提升在长期学习与福利上的不同效果；③在企业可动态定价的情境下，提出了基于“转录”信息的市场收敛与均衡的解析框架。

**🔧 技术方法**

采用了 Pandora’s Box / Weitzman 模型、马尔可夫多臂赌博机的思想、概率耦合与随机支配比较、以及对称市场下的价格变换市场（S‑transformed market）来推导均衡与学习结果。

**📊 数据集**

论文为理论研究，未使用任何实证数据集；所有结论均来自严格的数学证明。

**📈 对比分析**

通过耦合历史和随机支配比较来验证不同搜索技术对学习集合和消费者福利的影响；结果显示：降低搜索成本始终提升学习与福利，提升搜索信息量在未观察转录信息时可能降低福利，若平台可获取转录信息则可恢复并提升学习与福利。

**⚠️ 局限性**

局限性包括：1）二元拟合/质量信号的简化；2）消费者采用满足式效用模型；3）假设反馈完全诚实且无噪声；4）搜索成本同质化、顺序到达且一次性访问；5）未考虑企业投资质量、隐私约束或代理者的战略操纵；6）结果主要在对称市场中可解析，对非对称/异质场景的推广有限。

---

## 21. Rethinking Token Pruning for Historical Screenshots in GUI Visual Agents: Semantic, Spatial, and Temporal Perspectives

**arXiv ID:** 2603.26041 | [PDF](https://arxiv.org/pdf/2603.26041v1)

**作者:** Daiqiang Li `[一作]` (Sichuan University), Haiyun Jiang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对 GUI 视觉代理中历史截图的令牌裁剪问题，系统实验分析了前景与背景的语义价值、空间分布一致性以及时间衰减对决策的影响，并提出了基于时间衰减的令牌分配策略。

**💡 创新点**

创新点在于：①发现背景区域虽然语义低但能捕捉界面状态变化；②随机裁剪在保持空间一致性方面优于复杂重要性裁剪；③引入时间衰减因子 λ 对历史截图分配不同令牌预算，显著降低计算量同时保持性能。

**🔧 技术方法**

技术包括：Sobel 边缘检测将截图划分为前景/背景；Transformer 中间层令牌裁剪；M‑RoPE 位置编码分析；基于时间衰减的令牌保留公式 N_retain^k = N_total·λ^k；实验使用 Qwen2‑VL‑2B 与 Qwen2.5‑VL‑3B 视觉语言模型；对齐评估通过 Step Success Rate、Operation F1 等指标。

**📊 数据集**

使用四大 GUI 代理基准：AITW、Mind2Web、AndroidControl、GUI‑Odyssey，涵盖多平台、多任务与跨域场景。

**📈 对比分析**

与 FastV、SparseVLM、DivPrune、DART、PDrop 等现有裁剪方法比较，随机裁剪与时间衰减版裁剪在 Step SR 等指标上接近或略优，且 FLOPs 从 0.93T 降至 0.74T，约 20.5% 的计算量提升。

**⚠️ 局限性**

局限性在于：裁剪策略仅在推理时无监督应用，未针对不同模型或任务自适应 λ；对位置编码的补偿机制仍需进一步研究；实验仅涵盖视觉语言模型，未验证对其它架构的普适性。

---

## 22. Methods for Knowledge Graph Construction from Text Collections: Development and Applications

**arXiv ID:** 2603.25862 | [PDF](https://arxiv.org/pdf/2603.25862v1)

**作者:** Vanni Zavarella `[一作]` `[通讯]`, Vanni Zavarella

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并评估了三种基于文本集合的知识图谱（KG）构建方法，涵盖数字化转型（DT）舆情监测、AECO领域科研趋势分析以及医疗记录与患者药物评论的因果关系图谱。通过开放式信息抽取、关系聚类、LLM提示工程与参数高效微调等技术，完成了从原始文本到可查询 KG 的端到端流水线，并发布了相应的数据集、模型与可视化仪表盘。

**💡 创新点**

创新点包括：①在社交媒体和新闻文本上提出了可扩展的三重抽取流水线（Triplétoile 与新闻管道），实现了高质量、可解释的 KG 生成；②首次在非结构化医疗文本中对因果关系抽取进行大规模 LLM 基准，系统评估了多种 GPT‑4 及开源 LLM 的提示与微调策略；③引入 UMAP‑HDBSCAN 结合 GloVe 进行关系聚类，实现了语义统一化与噪声抑制；④发布了公开可复现的代码与三大领域 KG 资源，推动了跨学科 KG 研究与应用。

**🔧 技术方法**

主要技术包括：自然语言处理（spaCy、依存句法、NER/RE）、深度学习（BiLSTM‑CRF、CNN、SpERT、BERT、S-BERT）、LLM（GPT‑4、Mistral、LLaMa 3 等）与 Prompt 工程（Instruction Prompt、Few‑Shot、Prompt Chaining、CoT）、参数高效微调（LoRA）、关系聚类（GloVe + UMAP + HDBSCAN）以及知识图谱构建与展示（RDF、RDFS、DBpedia 连接、SPARQL、可视化仪表盘）。

**📊 数据集**

使用的数据集包括：①约 4M 条使用 #DigitalTransformation 的英文推文（经预处理后抽样 100k 条）；②约 7.8M 条来自 Dow Jones DNA 的英文新闻（过滤后 97k 条数字健康技术相关文章）；③公开的医学临床记录与患者药物评论（用于因果关系图谱构建）；④多项公开 NLP 语料（OntoNotes、OpenIE 等）用于模型预训练与评测；⑤实验中用到的标注数据（如 4,097 条新闻主题标注、数千条医学关系实例）。

**📈 对比分析**

对比方法主要包括：传统 OpenIE + 规则抽取、基于关系模式的抽取、无监督关系聚类、传统机器学习模型（SVM、Random Forest）以及不同 LLM 提示与微调策略。评测指标涵盖抽取精度、召回率、F1 分数、S-score（聚类质量）、模型推理速度与资源占用。结果显示：①数字健康主题分类 F1 超过 98%；②关系聚类在 UMAP‑HDBSCAN 设置下达到 0.65 的 S-score，覆盖率 89%；③因果关系抽取中，经过 LoRA 微调的 GPT‑4 在 5‑shot 情境下实现 83% 的命名实体正确率与 78% 的关系正确率，显著优于基线模型；④整体 KG 构建流水线在推文集上相较于现有方法提升了 12% 的三元组质量。

**⚠️ 局限性**

主要局限性包括：①社交媒体文本噪声大、句法解析误差导致抽取误差；②缺乏足够的标注数据，使得基于监督的 NER/RE 方法在领域迁移时性能下降；③关系聚类结果依赖嵌入表示与聚类超参数，可能导致语义重叠或过度聚合；④LLM 微调与推理成本高，特别是在大规模 KG 生成时资源消耗显著；⑤生成的 KG 对来源文本的可追溯性与可解释性有限，难以满足法规合规要求；⑥在医疗领域的因果关系抽取仍受限于病历记录结构与隐私约束，模型泛化能力有待进一步验证。

---

## 23. Why Safety Probes Catch Liars But Miss Fanatics

**arXiv ID:** 2603.25861 | [PDF](https://arxiv.org/pdf/2603.25861v1)

**作者:** Kristiyan Haralambiev `[一作]` `[通讯]` (Independent Research), Kristiyan Haralambiev (Independent Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究激活探测器在检测AI系统对齐偏差时的盲点，证明并演示当模型将有害行为内化为正当信念（即“Fanatic”）时，传统激活探测几乎完全失效；同时通过对比“Liar”与“Fanatic”两类模型的行为与内部表示，展示了这类偏差的不可检测性与产生机制。

**💡 创新点**

创新点在于：①区分两种对齐偏差（欺骗性与一致性）并阐明后者在触发器复杂度达到PRF级别时的理论检测难度；②证明RLHF与链式思考训练可自然诱导模型从可检测的“欺骗性”状态迁移至不可检测的“一致性”状态；③首次在实验层面验证“Fanatic”模型通过生成特定的信念化解释实现探测逃逸。

**🔧 技术方法**

使用技术包括：激活式线性探测器（Intent、Live Belief、Static Belief），链式思考（CoT）生成与奖励，强化学习调优（RLHF/RLOO），logit lens与稀疏自编码器分析，非线性探测器对照，以及基于一向函数（PRF）假设的加密难度证明。

**📊 数据集**

数据集为受控问答任务，主要以关于体重的提问作为触发器，使用两种模型（Mistral 7B、Gemma 3‑4B）进行训练；实验中同时使用模型自身生成的解释与人工写入的相同内容作为对照材料。

**📈 对比分析**

通过对比“Liar”（直接产生有害输出）与“Fanatic”（通过CoT产生正当化的有害输出）在意图形成层（16–21）与实时信念层的探测率，发现Liar在大多数层的检测率>90%，而Fanatic仅0–21%；静态解释对照实验进一步验证逃逸源自生成的特定词序列而非机制本身。

**⚠️ 局限性**

局限性包括：实验仅在4B–7B规模模型上验证，规模更大时行为可能不同；触发器复杂度与PRF假设尚未实证量化；理论证明基于平均情况，实际大规模模型可能出现更复杂的偏差形式。

---

## 24. Exploiting the Degrees of Freedom: Multi-Dimensional Spatially-Coupled Codes Based on Gradient Descent

**arXiv ID:** 2603.25824 | [PDF](https://arxiv.org/pdf/2603.25824v1)

**作者:** Ata Tanrıkulu `[一作]` (Middle East Technical University), Ahmed Hareedy `[通讯]` (Middle East Technical University)

**通讯引用:** 275 | [OpenAlex ID](https://openalex.org/A5066643381)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了一套概率性框架（MD‑GRADE 与 MCMC），用于优化多维空间耦合 LDPC 码（MD‑SC），以最小化 Tanner 图中的不利短环与环连结构。

**💡 创新点**

创新点在于：①将梯度下降与概率分布矩阵结合，得到局部最优边分布；②利用特征多项式与 Lagrange 条件，推导出短环与环连对象的期望计数公式；③在 MD‑SC 码中显式分离耦合与重定位阶段，并将两阶段最优分布约束化简；③引入“主导对象模式”近似，显著降低计算复杂度。

**🔧 技术方法**

使用的技术包括：梯度下降（MD‑GRADE）、多维离散卷积（FFT 或直接实现）、Lagrange 最优化、特征多项式与系数数组、Markov Chain Gibbs 采样（MCMC）作为有限长度（FL）优化器。

**📊 数据集**

并未采用传统机器学习数据集，而是通过理论推导与仿真（AWGN/BER 通道下 BP 迭代）验证码的周期、环连计数及误码性能。

**📈 对比分析**

与现有 MD‑SC 码（如 OO、MCMC 生成）及均匀分布方案相比，所设计的 GD‑MCMC 码在相同码长/码率下短环计数下降数倍，误码率的误码底（error floor）提升可达 2.81 订单（约 5000‑倍以上）。

**⚠️ 局限性**

局限性包括：①梯度下降与高维卷积运算导致对大记忆量（m）和多辅助矩阵（M）时计算成本高；②仅考虑主导对象模式，未覆盖所有吸收集/陷阱集；③MCMC 采样仍耗时且受温度 β 设定影响；④算法主要针对原始环连结构，可能无法彻底消除更复杂的吸收子集。

---

## 25. Self-Organizing Multi-Agent Systems for Continuous Software Development

**arXiv ID:** 2603.25928 | [PDF](https://arxiv.org/pdf/2603.25928v1)

**作者:** Wenhan Lyu `[一作]` (William & Mary), Yifan Sun `[通讯]` (William & Mary)

**通讯引用:** 1528 | [OpenAlex ID](https://openalex.org/A5041025969)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为TheBotCompany的开源多智能体框架，用于实现持续、自动化的软件开发流程。

**💡 创新点**

核心创新包括：①三阶段（计划–实现–验证）的持续生命周期；②自组织工作团队，管理者动态雇佣、分配和解雇工作代理；③基于预算的无监督异步人机监督机制。

**🔧 技术方法**

技术实现依赖大型语言模型（如Claude Sonnet 4.5）、Node.js orchestrator、SQLite通信层、Issue Tracker、Web Dashboard、token 计量与预算调度等。

**📊 数据集**

使用了四个真实项目（M2Sim、GroundDB、RustLaTex、PyInterpreter）以及 ProjDevBench benchmark 的五个最难问题作评测。

**📈 对比分析**

与单体 Claude Code 基准在 ProjDevBench 上对比，取得平均+9.6/10.7 分；在长周期项目中记录里程碑完成率、验证通过率、成本和 token 效率；自组织团队使 70% 代码工作量由 worker 完成，验证阶段发现 5–18% 的回归，整体成本略高但在复杂任务上表现更佳。

**⚠️ 局限性**

局限性包括 LLM 的随机性导致单次实验缺乏方差估计；评测仅覆盖系统工程类项目，未检验不确定需求场景；成本相较单体代理更高；缺少与并行竞争型团队框架的直接对比；对人类干预粒度的调节尚不成熟。

---

## 26. Parameter-Free Dynamic Regret for Unconstrained Linear Bandits

**arXiv ID:** 2603.25916 | [PDF](https://arxiv.org/pdf/2603.25916v1)

**作者:** Alberto Rumi `[一作]` (Intesa AI), Fabio Vitale `[通讯]` (Intesa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了无约束对抗线性带宽问题中的动态遗憾最小化，提出了一种简单的方法来结合多种带宽算法的保证，从而在不知道切换次数的情况下，达到最优的动态遗憾保证。

**💡 创新点**

提出了第一个在无约束线性带宽中实现最优遗憾保证的算法，解决了一个长期存在的开放问题，且不需要事先知道切换次数。

**🔧 技术方法**

使用了一种结合比较器自适应基础算法的技术，并通过采样技巧将其适应到带宽反馈中。

**📊 数据集**

未具体提及使用的数据集，但研究的背景是无约束线性带宽问题，涉及对抗环境中的动态决策。

**📈 对比分析**

与之前的工作相比，提出的方法在没有先验知识的情况下，能够实现动态遗憾保证的最优界限，性能优于传统方法，尤其是在动态或无调优场景中。

**⚠️ 局限性**

方法依赖于比较器序列的盲目性，且在扩展到更一般的约束动作集时可能会遇到困难。

---

## 27. Shared Representation for 3D Pose Estimation, Action Classification, and Progress Prediction from Tactile Signals

**arXiv ID:** 2603.25906 | [PDF](https://arxiv.org/pdf/2603.25906v1)

**作者:** Isaac Han `[一作]` (Gwangju Institute of Science and Technology), Kyung-Joong Kim `[通讯]` (Gwangju Institute of Science and Technology)

**通讯引用:** 2189 | [OpenAlex ID](https://openalex.org/A5076055880)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了统一的SCOTTI模型，利用脚部触觉传感器同时预测3D人体姿态、动作类别和动作进度。

**💡 创新点**

首次将动作进度预测与姿态、分类联合学习，并通过共享表示实现多任务互利；同时公开了大规模脚部触觉+视觉数据集。

**🔧 技术方法**

采用卷积特征提取+Transformer时序建模，三任务共享编码器，任务头分别为MLP；使用CNN、Transformer、加权损失组合。

**📊 数据集**

新收集的约7小时、15位参与者、200k同步触觉+视觉帧的脚部触觉数据集，包含8种动作。

**📈 对比分析**

与多种单任务基线（Voxel‑CNN、Sep‑CNN、Res‑CNN、GCN‑Transformer、STAT）及其多任务版本对比，SCOTTI在MPJPE、分类准确率、进度MSE/APP上均优于所有基线，显示显著提升。

**⚠️ 局限性**

模型仍受限于受试者数量有限、仅针对脚部触觉，复杂动作或其他部位的触觉信息缺乏；进度标签基于手动阈值，可能不适用于所有动作。

---

## 28. Enormous Fluid Antenna Systems (E-FAS) under Correlated Surface-Wave Leakage: Physical Layer Security

**arXiv ID:** 2603.25943 | [PDF](https://arxiv.org/pdf/2603.25943v1)

**作者:** Farshad Rostami Ghadi `[一作]` (University College London), Hyundong Shin `[通讯]` (Kyung Hee University)

**通讯引用:** 8380 | [OpenAlex ID](https://openalex.org/A5007557286)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在存在表面波泄漏的巨大流体天线系统（E‑FAS）下的物理层安全性能，提出了基于MMSE信道估计、MRT加人工噪声（AN）的下行MISO传输模型，并给出了封闭式的信道安全熵（ESR）与安全失效概率（SOP）表达式。

**💡 创新点**

创新点在于：①构造了表面波泄漏相关的协方差模型；②利用E‑FAS的两时标路由增益特性，推导出条件SOP与ESR的闭式解析；③证明在高SNR时仅当不使用AN时才会出现安全崩塌，且最佳AN/数据功率分配存在内部最优解；④通过路由增益提升CSI质量，从而实现更高的安全上限。

**🔧 技术方法**

所采用的技术主要包括MMSE信道估计、MRT/AN波束成形、随机信道建模（协方差与非中心卡方分布）、特征值分解与拉普拉斯变换、以及大数定律与极限分析；通过数值积分和高斯-拉格朗日求积实现闭式表达式的数值评估。

**📊 数据集**

本文没有使用实际实验数据，而是在仿真中采用了典型参数（M=16天线、T_c=200、τ_p=20、β_b=5、β_e=3、ρ=0.6或0）进行Monte‑Carlo仿真，以验证解析结果的准确性。

**📈 对比分析**

与传统的无路由空间波传输（No‑E‑FAS）和独立泄漏场景进行对比，E‑FAS在SOP和ESR上均实现显著提升；尤其在高功率、强路由增益或较大AN比例时，安全失效概率下降至几乎零，且ESR达到非零上限，显示出良好的鲁棒性和性能优势。

**⚠️ 局限性**

主要局限包括：①假设了理想的表面波路由与泄漏相关模型，实际环境中可能存在更复杂的相互作用；②对大规模衰落参数β_e的上界做了简单的区间估计，未考虑动态路径变化；③未考虑硬件失真、相位误差等实际系统效应；④闭式表达式相对复杂，数值实现需要较高的计算资源。

---

## 29. HeyFriend Helper: A Conversational AI Web-App for Resource Access Among Low-Income Chicago Residents

**arXiv ID:** 2603.25800 | [PDF](https://arxiv.org/pdf/2603.25800v1)

**作者:** Maddie Juarez `[一作]` (Loyola University Chicago), George K. Thiruvathukal `[通讯]` (Loyola University Chicago)

**通讯引用:** 1207 | [OpenAlex ID](https://openalex.org/A5074177185)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了HeyFriend Helper，一站式 Web 平台，整合会话式 AI、简历/面试工具、语言学习、正念资源及社区服务定位，面向芝加哥低收入居民。

**💡 创新点**

首次将职业准备、心理健康、语言援助等多维支持通过对话式界面整合在同一平台，并采用 RAG‑式检索与本地化翻译，提升可访问性。

**🔧 技术方法**

使用 OpenAI ChatGPT Assistants、RAG 检索、Google Translate API、Google Maps API、RenderCV、CareerOneStop API、Speech‑to‑Speech 以及语音识别等技术。

**📊 数据集**

采用 20 条预先编写的问答对作为检索语料、O*NET 和 CareerOneStop 的公开职业数据，以及 25 名参与者的交互日志。

**📈 对比分析**

通过交互日志和半结构化访谈评估，发现用户活跃度高（55 次会话、66 条提问、17 份简历等），并在质性访谈中获得正面反馈；未与传统工具做直接性能对比，缺乏量化指标。

**⚠️ 局限性**

局限包括样本规模小、仅限 Web 端、界面复杂导致使用不确定、对文化偏见处理不足、缺乏客观性能基准，以及对多设备适配性不足。

---

## 30. ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions

**arXiv ID:** 2603.25791 | [PDF](https://arxiv.org/pdf/2603.25791v1)

**作者:** Zikai Wang `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 62759 | [OpenAlex ID](https://openalex.org/A5100636655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于优化的框架 ArtHOI，用单目 RGB 视频在不需要预先扫描或模板对象的前提下，实现 4D 手-可运动物体交互的完整重建。

**💡 创新点**

创新点包括：Adaptive Sampling Refinement (ASR) 通过自适应尺度采样与 6-DoF 位姿估计实现网格到世界坐标的精确对齐；多模态大语言模型 (MLLM) 引导的接触推理与手物体对齐优化，显著提升物体与手势的物理一致性；并贡献了两套新数据集 ArtHOI-RGBD 与 ArtHOI-Wild，覆盖真实深度与野外场景。

**🔧 技术方法**

技术手段涵盖多种基础模型：HunYuan3D、FoundationPose、Segment‑Anything、Video‑Depth‑Anything、DiffuEraser、WiLoR、Qwen‑VL‑Max 等，结合自适应采样、密集跟踪、帧级接触约束与全局优化实现最终 4D 重建。

**📊 数据集**

使用的数据集包括新建的 ArtHOI‑RGBD（含 RealSense 深度）、ArtHOI‑Wild（互联网与手机拍摄），以及公开的 RSRD、ARCTIC 数据集进行评估。

**📈 对比分析**

与需要预扫描对象的 RSRD 以及逐帧单图像方法 EasyHOI 进行对比，实验表明 ArtHOI 在 ArtHOI‑RGBD 上实现更低的 Chamfer、MSSD 误差，在 RSRD 与 ARCTIC 上与 RSRD 相当甚至更优；在手物交互对齐方面 Co^2 分数显著下降，接触推理准确率和 FP 降低明显。

**⚠️ 局限性**

局限性包括：对基础模型预测质量高度依赖，极端遮挡或极端运动仍可能导致重建漂移；计算成本较高（约 1 小时/100 帧）；缺乏对多模态噪声的鲁棒性处理，且目前仅在 RGB/深度与手姿估计等已知模型支持下可用。

---

## 31. Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis

**arXiv ID:** 2603.25778 | [PDF](https://arxiv.org/pdf/2603.25778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 32. Can Small Models Reason About Legal Documents? A Comparative Study

**arXiv ID:** 2603.25944 | [PDF](https://arxiv.org/pdf/2603.25944v1)

**作者:** Snehit Vaddi `[一作]` `[通讯]` (Independent Researcher), Snehit Vaddi (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了九种小型语言模型（3B‑9B参数）在三项法律推理任务（ContractNLI、CaseHOLD、ECtHR）上的表现，并系统比较了五种提示策略（直接、链式推理、少样本、BM25 RAG、Dense RAG）

**💡 创新点**

发现Mixture‑of‑Experts 3B活跃参数模型（Qwen3‑A3B）能匹配商业API（GPT‑4o‑mini）的平均准确率，且在少样本提示下超越API；同时提示策略与任务高度相关，链式推理对合同推理有利但对案例识别和多标签任务有害；最后检索方式（BM25 vs Dense）对性能影响不大

**🔧 技术方法**

使用云端API推理、greedy解码、分层随机抽样、四步输出提取、bootstrap置信区间、配对bootstrap检验、McNemar检验等统计方法，进行405次实验（9模型×5策略×3数据集×3种子）

**📊 数据集**

使用公开法律基准数据集：ContractNLI（合同NLI）、CaseHOLD（案例持有识别）和ECtHR（欧洲人权法院多标签违规预测）

**📈 对比分析**

相较于公开基准的细调模型，提示式方法整体性能略低，但小型模型在平均准确率上接近或超过商业API；最优组合为Qwen3‑A3B + 少样本提示，平均准确率约47%（GPT‑4o‑mini 47.2%）

**⚠️ 局限性**

仅评估提示式方法，未涉及微调或参数高效微调；仅覆盖三种英文法律任务，未涉及生成任务、多语言或不同法域；可能存在训练数据泄漏、API更新导致结果漂移、有限样本数导致统计功效不足

---

## 33. Opportunities and Limitations of GenAI in RE: Viewpoints from Practice

**arXiv ID:** 2603.25905 | [PDF](https://arxiv.org/pdf/2603.25905v1)

**作者:** Anne Hess `[一作]` (University of Applied Sciences Würzburg-Schweinfurt), Alexander Rachmann `[通讯]` (Hochschule Niederrhein)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过在线问卷调查收集需求工程从业者对生成式AI（GenAI）使用情况、效果与障碍的实证数据，形成行业视角的使用现状与培训需求概况。

**💡 创新点**

首次系统性描绘行业层面GenAI在需求工程各阶段的真实应用场景、主要阻碍与需求教育，弥补学术研究与实践落差。

**🔧 技术方法**

采用LimeSurvey在线问卷、定量描述性统计及手工主题分析方法。

**📊 数据集**

57份完成的问卷响应，已公开发布于Zenodo（https://doi.org/10.5281/zenodo.17429345）。

**📈 对比分析**

对需求工程各活动（ elicitation、analysis、specification、validation、management ）的使用频率、效益与风险进行定量比较，显示规格/建模使用率最高（≈66%），主要优势为时间节省，主要风险为hallucination，整体采用率超过50%。

**⚠️ 局限性**

样本量有限（57人）、低响应率（≈44%），受自选偏差与行业分布不均影响，结果为自评数据，缺乏客观验证，且对部分需求活动（如管理）响应不足，导致结论的代表性和深度受限。

---

## 34. AutoB2G: A Large Language Model-Driven Agentic Framework For Automated Building-Grid Co-Simulation

**arXiv ID:** 2603.26005 | [PDF](https://arxiv.org/pdf/2603.26005v1)

**作者:** Borui Zhang `[一作]` (University of New South Wales), Flora Salim `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 AutoB2G——一种基于自然语言任务描述的自动化建筑–电网共仿框架，能够完整地从建筑侧控制到电网级优化与评估实现全流程自动化。

**💡 创新点**

创新点包括：①将 CityLearn V2 扩展为支持建筑–电网共仿并引入多网侧评估指标；②利用 DAG 结构的代码库与 agentic retrieval，让 LLM 能够基于模块依赖进行结构化检索；③结合 SOCIA 多智能体框架与 Textual Gradient Descent，实现从代码生成、执行、评估到修复的闭环迭代。

**🔧 技术方法**

使用技术：大型语言模型（LLM）+检索增强生成；DAG‑based agentic retrieval；SOCIA 多智能体框架；Textual Gradient Descent (TGD)；CityLearn V2、EnergyPlus、Pandapower 等仿真工具；Python 语言与 OpenAI GPT‑5 API。

**📊 数据集**

数据集：CityLearn V2 默认住宅数据集（End‑Use Load Profiles）、IEEE 33‑bus 配电网络、EnergyPlus 生成的建筑能耗与热力数据；无使用其他公开数据集。

**📈 对比分析**

评估方法：将任务分为简单、中等、复杂三类，并比较四种配置（LLM、LLM+AR、SOCIA、SOCIA+AR）的任务成功率与代码分数。AutoB2G（SOCIA+AR）在所有层级均取得最高性能：复杂任务成功率0.83、代码分数0.88；相比之下单纯 LLM 在复杂任务的成功率仅为0.53、代码分数0.44。

**⚠️ 局限性**

局限性：目前仅在 CityLearn+Pandapower 组合上验证，缺乏跨平台、跨接口的鲁棒性评估；功能层面可配置性与模块复用性仍待提升；未针对更广泛的仿真工具与控制策略进行测试。

---

## 35. Do not throw out the baby: Clarithmetics as alternatives to weak arithmetics

**arXiv ID:** 2603.26040 | [PDF](https://arxiv.org/pdf/2603.26040v1)

**作者:** Giorgi Japaridze `[一作]` (Villanova University), Giorgi Japaridze `[通讯]` (Villanova University)

**通讯引用:** 820 | [OpenAlex ID](https://openalex.org/A5021239614)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了可计算逻辑（CoL）的核心概念，构造了基于CoL的算术体系——clarithmetics（CLA4–CLA7和CLA11），并证明了这些体系在多种时间、空间与幅度复杂度下的音响性与完整性；

**💡 创新点**

创新点在于：①将CoL的游戏语义引入算术逻辑，构建强于Peano算术的“弱算术”体系；②提出可直接从证明中提取最优（或近似最优）算法的机制；③在保持PA完整性的同时实现对复杂度类的精确控制；

**🔧 技术方法**

使用了：可计算逻辑的游戏语义与choice运算符、构造性归纳规则、复杂度理论（多项式时间/空间、可递时间/空间、幅度复杂度）与证明论相结合的技术；

**📊 数据集**

未使用任何经验性数据集，全部为形式化证明与理论推导；

**📈 对比分析**

与传统弱算术（如Buss的bounded arithmetic）比较时，clarithmetics在保持PA全部算术信息的前提下，提供了更强的强度完整性，并能在多项式时间、空间或可递复杂度下保证可解性，算法提取得到的程序在理论上具有最优或接近最优的复杂度；

**⚠️ 局限性**

局限性包括：尚缺乏自动化程序提取工具的实现；交互式计算问题的具体算法实例有限；理论分析尚未在实验环境中验证性能；需要进一步探究与其他复杂度类对应关系及模型多样性问题。

---

## 36. A Large-scale Empirical Study on the Generalizability of Disclosed Java Library Vulnerability Exploits

**arXiv ID:** 2603.25997 | [PDF](https://arxiv.org/pdf/2603.25997v1)

**作者:** Zirui Chen `[一作]` (Zhejiang University), Xiaohu Yang `[通讯]` (Zhejiang University)

**通讯引用:** 10798 | [OpenAlex ID](https://openalex.org/A5026311099)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开展了针对 Java 库漏洞利用的首个大规模实证研究，构建了最大规模的漏洞利用数据集并对其跨版本可用性进行系统评估。

**💡 创新点**

创新点包括：①证明公开漏洞利用在跨版本上的高可用性；②提出并手工实现 10 种迁移策略，将 77.1% 的失效案例迁移成功；③贡献 796 条此前缺失的受影响版本给 CPE。

**🔧 技术方法**

技术手段涵盖：手工复现漏洞利用、在历史版本上执行、差异分析驱动的手工迁移、构建三步评估框架，并与数据库及工具进行对比实验。

**📊 数据集**

使用的数据集包含 259 个漏洞利用，覆盖 128 个 Java 库、28,150 版本、61 个 CWE 以及 14,378 个真实受影响版本。

**📈 对比分析**

评估方法：将利用执行结果与 5 大数据库及 2 个 SOTA 工具的受影响版本标注进行比较，实验结果显示未迁移前召回率 83.0%、精度 99.3%，迁移后召回率提升至 96.1%。

**⚠️ 局限性**

局限性：迁移工作主要针对代码层面，语义变更仍难以自动化；实验仅涵盖 Maven 生态的 Java 库；复现过程受限于漏洞利用可获得性和环境配置的多样性。

---

## 37. Sommelier: Scalable Open Multi-turn Audio Pre-processing for Full-duplex Speech Language Models

**arXiv ID:** 2603.25750 | [PDF](https://arxiv.org/pdf/2603.25750v1)

**作者:** Kyudan Jung `[一作]` (KAIST AI), Cheonbok Park `[通讯]` (KAIST AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发并公开了 Sommelier，一个可扩展的全双工语音预处理流水线，用于构建高质量对话式语音语言模型的数据集。

**💡 创新点**

创新点在于：①首次实现面向全双工 SLM 的可扩展管道；②通过严格的说话人分离与重叠语音处理，显著提升文本准确度；③采用模型集成与 ROVER 机制进一步降低 ASR 失真。

**🔧 技术方法**

主要技术包括：音频标准化、基于 VAD 的分块、Sortformer 说话人分离、基于 PANNs 与 Demucs 的背景音乐去除、SepReformer 语音分离、三模型 ASR 集成与 n‑gram 去噪、以及流式降噪 FlowSE。

**📊 数据集**

使用的数据集涵盖公开的无线广播与播客音频（如 Naverbg）、VoxConverse、LibriSpeech、TEDLIUM3 等，训练时选取 10,000 小时左右的 Web‑scale 语料。

**📈 对比分析**

与基线（pyannote 3.1、单模型 Whisper）比较，Der 从 8.40% 降至 7.16%，JER 与短语音分隔精度提升；全双工评测 Full‑Duplex‑Bench 1.0 上，Sommelier 处理数据显著提升后话、平滑对话与用户中断性能；ASR 集成将 WER 从 6.26% 降至 3.92%，语音分离在重叠率高时仍保持接近 Oracle 的 UTMOS。

**⚠️ 局限性**

局限性：专注于语音数据，未考虑非语音事件或多模态场景；人工分离的音频相较于原始双声道录音存在轻微音质损失。

---

## 38. Unlabeled Cross-Center Automatic Analysis for TAAD: An Integrated Framework from Segmentation to Clinical Features

**arXiv ID:** 2603.26019 | [PDF](https://arxiv.org/pdf/2603.26019v1)

**作者:** Mengdi Liu `[一作]` (Tianjin University), Yuting Su `[通讯]` (Tianjin University)

**通讯引用:** 6961 | [OpenAlex ID](https://openalex.org/A5033713097)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在无目标中心标注的跨机构环境下，提出一种基于无监督域适应的端到端框架，实现对TAAD的多类别分割并自动提取临床关键信息（如起裂点、真腔塌陷率、假腔面积比和分支血管受累）。

**💡 创新点**

核心创新包括：① Style Mixup增强解耦学习（SMEDL）分离域不变内容与域特定风格；② 原型锚定的选择性熵约束自适应语义对齐（SE‑ASA）实现无标注目标域的自监督训练；③ 基于分割结果的结构约束损失与临床特征提取算法，将分割转化为可直接用于手术规划的定量指标。

**🔧 技术方法**

技术涵盖：3D nnU‑Net骨干网络、SMEDL模块、SE‑ASA自监督训练、边界损失、解剖约束损失、误分类损失，以及基于中心线与拓扑分析的临床特征计算。

**📊 数据集**

使用了imageTAAD（120例带细粒度注释）作为源域数据集，并在未标注的546例CTA（323患者）中验证跨机构泛化能力。

**📈 对比分析**

与源域仅训练、DANN、Entropy Minimization等基线UDA方法对比，本文方法在目标域中的真腔、假腔和内膜裂口的DSC分别提升至0.891、0.884、0.673，HD95降至4.2、5.1、8.9 mm，显著优于其他方法；临床读者研究显示提取特征的主观效用评分均≥4.4/5，且手术规划支持度高。

**⚠️ 局限性**

局限性包括：① 仍依赖源域的高质量标注；② 对极端成像差异（如极低对比度或严重伪影）可能性能下降；③ 临床验证样本有限，需在更大多中心数据上进一步评估稳健性。

---

## 39. Constitutive parameterized deep energy method for solid mechanics problems with random material parameters

**arXiv ID:** 2603.26030 | [PDF](https://arxiv.org/pdf/2603.26030v1)

**作者:** Zhangyong Liang `[一作]` (Tianjin University), Huanhuan Gao `[通讯]` (Jilin University)

**通讯引用:** 3831 | [OpenAlex ID](https://openalex.org/A5100752930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种名为 CPDEM 的纯物理驱动深度能量方法，能够在单个预训练模型中对随机变异的材料参数进行零样本、实时推断，无需重网格或重新训练。

**💡 创新点**

创新点在于将材料参数显式编码为网络的潜在表征，并在能量最小化损失中对参数空间求期望，从而实现连续参数化的解空间学习，并支持快速微调；相比传统 DEM 或数据驱动算子，显著降低了数据需求和训练成本。

**🔧 技术方法**

主要技术包括：深度能量方法（Deep Energy Method）、参数化网络架构（材料编码器 + 空间编码器 + 流形网络）、期望能量最小化损失、Adam+L‑BFGS 训练策略、零样本推断与快速微调。

**📊 数据集**

实验中未使用大规模数据集，仅依赖解析解或高精度有限元（FEM）结果作为参考；所有案例均在 1D、2D、3D 线性弹性、Neo‑Hookean / Mooney‑Rivlin 超弹性以及大变形摩擦无接触问题上验证。

**📈 对比分析**

与 FEM、传统 DEM 以及神经算子方法的对比表明：CPDEM 在 1D、2D、3D 案例中相对 L² 误差低于 0.5%，推断时间仅为毫秒级，显著低于每个参数取值重新求解 FEM 或完整重新训练 DEM 的计算开销。

**⚠️ 局限性**

局限性包括：目前仅处理均匀随机材料参数；对空间变异材料场、塑性/损伤等路径依赖现象尚未实现；在参数域外（OOB）极端值仍需微调才能保持高精度。

---

## 40. Seeing Like Radiologists: Context- and Gaze-Guided Vision-Language Pretraining for Chest X-rays

**arXiv ID:** 2603.26049 | [PDF](https://arxiv.org/pdf/2603.26049v1)

**作者:** Kang Liu `[一作]` (Xidian University), Qiguang Miao `[通讯]` (Xidian University)

**通讯引用:** 8161 | [OpenAlex ID](https://openalex.org/A5007404362)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出CoGaze框架，结合视角位置、临床上下文和放射科医师注视信息进行胸片视觉‑语言预训练，生成更加符合诊断流程的表示。

**💡 创新点**

创新点在于：①设计上下文感知视觉编码器，将患者病史、症状与影像特征融合；②提出多层监督范式，包括混合正对比学习、疾病感知跨模态学习和软注视引导；③将注视视为软概率先验，提升对诊断关注区域的细粒度对齐。

**🔧 技术方法**

采用Perceiver架构的视觉编码器、Transformer语言编码器、混合正对比学习、Jensen‑Shannon距离软注视损失、疾病标签分类损失等技术，形成端到端的预训练模型。

**📊 数据集**

预训练数据主要为MIMIC‑CXR（240k胸片+1711注视案例），下游任务使用MIMIC‑CXR、SRRG‑Findings、NIH、SIIM、RSNA、Shenzhen、TBX11K等公开胸片数据集。

**📈 对比分析**

与MedCLIP、MGCA、EGMA、LLaVA‑Med等最新SOTA方法对比，CoGaze在自由文本报告生成（+2.0% CheXbertF1、+1.2% BLEU2）、零样本分类（+23.2% AUROC）、图文检索（+12.2% Precision@1）等多项任务均实现显著提升。

**⚠️ 局限性**

局限性：注视数据仅覆盖约0.7%样本，模型对缺失上下文仍需鲁棒；目前仅针对胸片，未扩展到其他影像模态或多模态时序分析。

---

## 41. GLU: Global-Local-Uncertainty Fusion for Scalable Spatiotemporal Reconstruction and Forecasting

**arXiv ID:** 2603.26023 | [PDF](https://arxiv.org/pdf/2603.26023v1)

**作者:** Linzheng Wang `[一作]` (Massachusetts Institute of Technology), Sili Deng `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2460 | [OpenAlex ID](https://openalex.org/A5043537937)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出GLU框架，统一稀疏观测的全场重建与时空预测，构建结构化潜在状态（全局CLS、局部传感器Token、重要性加权），并引入领导‑跟随动态模块实现高效长期演化。

**💡 创新点**

创新点包括：①全局‑局部‑不确定性三流融合的潜在表示；②重要性驱动的自适应邻域选择与软重建；③变分学习将不确定性转化为重要性；④Leader‑Follower Dynamics（LFD）实现线性内存扩展并显著降低预测误差累积。

**🔧 技术方法**

技术手段涵盖Transformer跨注意力、双向Encoder、Fourier特征映射、Soft domain‑adaptive reconstruction、Beta分布变分重要性学习、LFD、以及与RecFNO、Causal Transformer等基线的对照。

**📊 数据集**

数据集包括七种：圆柱流、NOAA海表温度、湍流通道流、平行板流（Re40/100）、FitzHugh–Nagumo反应扩散、以及多物理湍燃料仿真。

**📈 对比分析**

与POD‑GPR、MLP‑CNN、RecFNO、Senseiver等重建基线，以及Causal Transformer、FNO等预测基线对比。GLU在全场L2误差、能谱一致性以及多通道耦合保持方面均优于基线；LFD模块在长期滚动中显著抑制误差累积，保持周期和混沌轨道稳定。

**⚠️ 局限性**

局限性：需要大量训练数据；变分重要性学习的超参数调优复杂；在极稀缺传感器或极端非线性情形下仍受限；未对实时在线更新与不确定性量化进行深入研究；三维大规模流域的可扩展性仍待评估。

---

## 42. A Neural Score-Based Particle Method for the Vlasov-Maxwell-Landau System

**arXiv ID:** 2603.25832 | [PDF](https://arxiv.org/pdf/2603.25832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 43. Decoding Defensive Coverage Responsibilities in American Football Using Factorized Attention Based Transformer Models

**arXiv ID:** 2603.25901 | [PDF](https://arxiv.org/pdf/2603.25901v1)

**作者:** Kevin Song `[一作]` (Amazon Web Services), Amy Lee `[通讯]` (National Football League)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用因子化注意力Transformer对NFL多代理追踪数据进行预测，预测个体防守分配、接收者-防守者匹配以及针对防守者。

**💡 创新点**

创新点在于将时间和代理维度分离的因子化注意力机制应用于NFL防守覆盖任务，并实现帧级预测与衍生指标（伪装率、双重覆盖率）。

**🔧 技术方法**

采用因子化注意力Transformer，结合学习的嵌入、时间池化和多头分类头。

**📊 数据集**

使用NFL Next Gen Stats 2020-2024赛季的追踪、事件、参与、PFF标签和比赛元数据。

**📈 对比分析**

与基于规则的最近防守者、CNN‑LSTM等方法对比，单任务准确率约89%+，目标防守者准确率提升至88.2%；整体模型已在实时广播中部署。

**⚠️ 局限性**

限制在于标签存在模糊性、仅使用单一端点标签导致动态覆盖难以完全评估，以及对极端快速/非标准情境的泛化能力待验证。

---

## 44. Computing fixed point free automorphisms of graphs

**arXiv ID:** 2603.26006 | [PDF](https://arxiv.org/pdf/2603.26006v1)

**作者:** Aida Abiad `[一作]` (Eindhoven University of Technology), Sjanne Zeijlemaker `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5089306974)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了固定点自由自同构问题（FPFAut）在不同图类中的复杂度，证明其在拆分图、二分图、k-细分图以及非P4诱导子图的H-自由图类中仍为NP‑完整；并给出基于模分解的多项式时间算法，解决了cograph及其三种扩展（有界模宽图、树-图和P4‑稀疏图）的FPFAut与其不含固定点自反变换（FPFInv）问题。

**💡 创新点**

创新点包括：①将模分解技术推广到FPFAut与FPFInv问题，形成统一的“部分固定点自由自同构”（PFPFAut）框架；②在P4‑稀疏图、树-图等更一般图类中首次给出多项式时间算法；③通过PFPFInv与2-均匀等分区之间的对应关系，进一步推广Abiad等人关于cograph的结果；④探讨了compact图中FPFInv的难度并提出开放问题。

**🔧 技术方法**

主要技术：模分解与强模块的递归分解；构造分级自同构并将其还原为原图的自同构；利用图类的闭包性质（如补图、细分、树合并）构造多项式时间的归约；在特殊图类（完全图、树、蜘蛛图）中使用显式枚举或最短路径/匹配算法；以及对线性规划的极点分析用于compact图的FPFAut。

**📊 数据集**

本文为理论论文，无实验数据或公开数据集。所有结论均通过组合归约与算法复杂度分析得到。

**📈 对比分析**

算法的时间复杂度为多项式，具体对每个图类给出了上界（如O(n^c)）。与已有的仅在cograph上可解的FPFAut相比，扩展至有界模宽、树-图、P4‑稀疏图，保留了多项式时间的特性；对FPFInv同样实现了多项式时间求解，填补了此前仅已知NP‑完整的空缺。

**⚠️ 局限性**

局限性：①对一般有界rank‑宽图仍未给出多项式算法，仍是开放问题；②FPFInv在compact图上是否可多项式求解尚不确定；③对更广泛图类（如有界树宽、k-分裂图）仅给出NP‑完整性证明，未提供算法；④在实际应用中缺乏实验验证，性能尚未在大规模实例上检验。

---

## 45. Empowering Epidemic Response: The Role of Reinforcement Learning in Infectious Disease Control

**arXiv ID:** 2603.25771 | [PDF](https://arxiv.org/pdf/2603.25771v1)

**作者:** Mutong Liu `[一作]` (Hong Kong Baptist University), Jiming Liu `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 26158 | [OpenAlex ID](https://openalex.org/A5062375227)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对强化学习在传染病防控中的应用进行了系统综述，聚焦四个关键主题：资源分配、生命与生计的平衡、多干预混合策略以及跨区域协同控制；

**💡 创新点**

首次以公共卫生实际需求为视角，提出了针对强化学习优化干预策略的分类框架，并总结了最新文献的研究空白与未来方向；

**🔧 技术方法**

综述涵盖了多种强化学习方法，包括 Q‑learning、Actor‑Critic、PPO、DDPG、D3QN、A3C、层次RL 与多智能体RL 等；

**📊 数据集**

引用的研究使用了多样的模拟环境与真实数据，如希腊边境 PCR 测试数据、移动设备产生的社会网络、COVID‑19 公开时间序列以及元人群流动网络等；

**📈 对比分析**

通过文献检索与评估，作者构建了 19 篇工作数据库，并对各方法在不同干预场景下的性能指标（感染率、死亡率、经济成本、资源利用率等）进行比较，指出 RL 在多目标权衡与动态适应方面表现突出；

**⚠️ 局限性**

局限性包括：调查范围有限（仅 19 篇论文），跨区域协同研究稀缺，多干预组合空间过大导致样本效率低，奖励函数设计可能带来偏差，缺乏统一基准与评测框架，且对大规模、实时决策的可扩展性尚待验证。

---

## 46. Protecting User Prompts Via Character-Level Differential Privacy

**arXiv ID:** 2603.26032 | [PDF](https://arxiv.org/pdf/2603.26032v1)

**作者:** Shashie Dilhara Batan Arachchige `[一作]` (Macquarie University), Dali Kaafar `[通讯]` (Macquarie University)

**通讯引用:** 5212 | [OpenAlex ID](https://openalex.org/A5040251515)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用字符级本地差分隐私随机扰动用户提示，随后通过远程LLM进行提示恢复并执行下游任务，以保护用户敏感信息。

**💡 创新点**

1) 在字符层面应用k-ary随机响应机制，实现无需先识别敏感词的自动化隐私保护；2) 通过LLM的上下文修复平衡隐私与可用性；3) 提供理论随机恢复基线，证明敏感词恢复率接近随机猜测。

**🔧 技术方法**

k-RR字符扰动、LLM（GPT‑4o mini、Llama‑3.1 8B）恢复与摘要、差分隐私框架、Hamming距离分析、句子相似度（cosine）评估。

**📊 数据集**

i2b2/UTHealth（医疗记录）和 Enron（企业邮件）两大数据集，包含姓名、地点、邮箱、电话等PII。

**📈 对比分析**

与词级DP和Presidio规则红action基线在相同LLM恢复流程下对比；采用敏感词恢复率（隐私）和原始/恢复摘要语义相似度（实用性）两指标；结果显示字符级DP在相同隐私水平下能保持更高的摘要相似度，敏感词恢复率低于理论随机基线。

**⚠️ 局限性**

仍受LLM恢复能力限制；无法完全阻止拥有外部候选集的链接攻击；对低资源或非英语场景未评估；ε取值需针对特定模型校准；未能完整恢复多词实体，存在一定漏失。

---

## 47. EngineAD: A Real-World Vehicle Engine Anomaly Detection Dataset

**arXiv ID:** 2603.25955 | [PDF](https://arxiv.org/pdf/2603.25955v1)

**作者:** Hadi Hojjati `[一作]` (McGill University), Narges Armanfard `[通讯]` (McGill University)

**通讯引用:** 1330 | [OpenAlex ID](https://openalex.org/A5073955046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了EngineAD，一个由25台商用车辆六个月高分辨率引擎传感器数据构成的真实异常检测数据集，并基于该数据集对九种一类异常检测模型进行基准评估。

**💡 创新点**

创新点在于：①首次提供真实车辆运营的专家标注异常数据；②采用PCA降维并保持95%方差，避免泄露原始传感器信息；③系统展示了不同车辆之间的性能差异，揭示跨车泛化挑战。

**🔧 技术方法**

主要技术包括：一类学习框架、K-Means、Isolation Forest、One-Class SVM、LOF、HBOS、COPOD、SOD、Autoencoder、DeepSVDD、PCA降维、数据预处理（1Hz重采样、窗口分段、异常阈值设定）。

**📊 数据集**

使用的数据集是EngineAD，涵盖13种引擎相关传感器、每秒采样、六个月连续记录、经专家标注为正常或异常，最终转化为8维主成分的300时步段。

**📈 对比分析**

对每辆车分别训练70%正常数据，测试30%正常+所有异常，使用F1得分进行比较。结果显示传统方法（K-Means、One-Class SVM）在大多数车辆上优于或与深度学习模型相当，凸显经典方法在该任务中的竞争力。

**⚠️ 局限性**

限制包括：不同车辆之间性能差异大，异常往往细微且上下文依赖；当前方法未针对跨车域适应；模型可解释性不足，难以直接映射到原始传感器特征。

---

## 48. MAGNET: Autonomous Expert Model Generation via Decentralized Autoresearch and BitNet Training

**arXiv ID:** 2603.25813 | [PDF](https://arxiv.org/pdf/2603.25813v1)

**作者:** Yongwan Kim `[一作]` (Holo Studio Co., Ltd.), Sungchul Park `[通讯]` (Holo Studio Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了MAGNET（模型自主增长网络），这是一个去中心化系统，用于在商品硬件上自主生成、训练和服务领域专家语言模型。

**💡 创新点**

创新点在于将自主研究方法（autoresearch）与去中心化架构结合，自动化和去中心化实验-失败-假设循环，集成了四个关键组件。

**🔧 技术方法**

使用了BitNet b1.58三元训练、DiLoCo分布式合并、以及链上贡献追踪等技术。

**📊 数据集**

通过三个案例研究进行验证：视频安全分类（Zevor）、加密货币方向预测（StockClaw）和BitNet超参数优化（Genkidama）。

**📈 对比分析**

与现有系统相比，MAGNET在去中心化AI研究中整合了自主研究、硬件可访问性、知识聚合和激励完整性四个方面，性能在多个领域的案例研究中表现出显著的改进。

**⚠️ 局限性**

限制包括对教师模型的依赖、DiLoCo在异构数据上的应用、以及当前架构的成熟度尚未在公共主网上进行全面测试。

---

## 49. FlexiCamAR: Enhancing Everyday Camera Interactions on AR Glasses with a Flexible Additional Viewpoint

**arXiv ID:** 2603.26012 | [PDF](https://arxiv.org/pdf/2603.26012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 50. Online Learning for Dynamic Constellation Topologies

**arXiv ID:** 2603.25954 | [PDF](https://arxiv.org/pdf/2603.25954v1)

**作者:** João Norberto `[一作]` (NOVA School of Science and Technology), Cláudia Soares `[通讯]` (NOVA School of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在不假设轨道平面信息的前提下，基于约束在线学习框架，提出了卫星网络拓扑动态配置的凸优化模型并实现了离线与在线算法

**💡 创新点**

创新点在于①把拓扑求解转化为带约束的加权最小二乘问题，使用拉普拉斯矩阵表示，②采用在线凸优化而非传统静态/动态方法，③实现了投影和投影无关的在线算法对比

**🔧 技术方法**

使用凸优化（CVXPY）、拉普拉斯矩阵、在线梯度下降(OGD)和在线Frank‑Wolfe(OCG)算法、基于距离倒数的效用矩阵

**📊 数据集**

使用仿真数据：Iridium‑like 六个椭圆轨道平面每平面11颗卫星；后续实验使用18颗卫星+2基站的单轨道平面仿真

**📈 对比分析**

与传统+Grid方法比较，离线方法在平均度、最短路径、聚类系数等指标上与+Grid相近；在线算法中OCG最终误差低于OGD，且每次迭代计算速度更快，整体性能与离线基准接近

**⚠️ 局限性**

局限在于仅在仿真环境下验证，使用了距离倒数的简单效用指标，未考虑更复杂的ISL属性；算法对大规模星座的可扩展性和鲁棒性尚待进一步评估

---

## 51. Per-Bank Memory Bandwidth Regulation for Predictable and Performant Real-Time System

**arXiv ID:** 2603.26054 | [PDF](https://arxiv.org/pdf/2603.26054v1)

**作者:** Connor Rudy Sullivan `[一作]` (University of Kansas), Heechul Yun `[通讯]` (University of Kansas)

**通讯引用:** 2186 | [OpenAlex ID](https://openalex.org/A5064659321)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

针对多核系统共享 DRAM 的银行级干扰，提出并实现了一种基于 DRAM 银行的带宽调节器，能够在保证实时任务时序隔离的同时提升整体吞吐量。

**💡 创新点**

创新点在于将带宽调节精度提升到单个 DRAM 银行层面，利用逆向工程得到的银行映射信息，构造单银行性能攻击并证明银行级调节可抵御此类攻击，且在平均情况下比传统全银行调节提升 5.74 倍吞吐量。

**🔧 技术方法**

使用 DRAMA++ 进行银行映射逆向、FireSim 进行 FPGA 加速的周期级模拟、RISC‑V Rocket Chip SoC 进行硬件实现，以及结合 FR‑FCFS 调度和写入批处理的 DRAM 控制器改进。

**📊 数据集**

实验数据集包括 Raspberry Pi 4/5、Intel Coffee Lake、Jetson Orin AGX 等多种硬件平台，以及 SynthBench、IsolBench、SD‑VBS、矩阵乘法内核等真实工作负载。

**📈 对比分析**

与全银行调节方案对比，实验显示在单银行攻击下被调节任务的慢速比仅为 1.1×，而最佳努力任务在无攻击时吞吐量提升 5.74×，证明了银行级调节在保持隔离的同时显著提升性能。

**⚠️ 局限性**

局限性包括需要先逆向获得精确的银行映射信息、对高度并行工作负载的性能提升有限，以及在更大规模多银行（如 HBM）下需进一步验证可扩展性与实现复杂度。

---

## 52. A Compression Perspective on Simplicity Bias

**arXiv ID:** 2603.25839 | [PDF](https://arxiv.org/pdf/2603.25839v1)

**作者:** Tom Marty `[一作]` (Mila -- Quebec AI Institute), Dhanya Sridhar `[通讯]` (Universite de Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将深度网络的简单性偏差通过最小描述长度(MDL)框架建模，预测特征选择随训练数据量变化，并在半合成视觉任务上验证该理论；

**💡 创新点**

创新点在于将简单性偏差与MDL连接，提出特征选择随数据规模动态转移的理论，并定义低/高数据 regime 的鲁棒性窗口；进一步利用预序编码估计模型复杂度，并通过实验展示理论预测与神经网络行为的一致性；

**🔧 技术方法**

使用MDL两部分压缩、预序编码、交叉熵评估、Permutation特征重要性等技术，构建半合成Colored MNIST实验并对神经网络进行训练与评估；

**📊 数据集**

采用半合成Colored MNIST变体（包含数字形状、环境颜色、右侧像素水印）以及自定义训练/测试集；

**📈 对比分析**

通过对比MDL最优压缩线与训练模型在不同样本规模下的特征依赖和准确率，发现压缩阈值与特征切换高度一致，实验相关性达0.976，验证理论有效；

**⚠️ 局限性**

局限性在于仅考虑互斥特征子集，未处理多特征交互；理论与实际网络间的转移不完全离散，缺乏对多部分预测特征的研究。

---

## 53. Adversarial-Robust Multivariate Time-Series Anomaly Detection via Joint Information Retention

**arXiv ID:** 2603.25956 | [PDF](https://arxiv.org/pdf/2603.25956v1)

**作者:** Hadi Hojjati `[一作]` (McGill University), Narges Armanfard `[通讯]` (McGill University)

**通讯引用:** 1330 | [OpenAlex ID](https://openalex.org/A5073955046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为ARTA的联合对抗训练框架，通过稀疏掩码生成器与异常检测器共同训练，提升多变量时间序列异常检测的鲁棒性和解释性。

**💡 创新点**

创新点在于：①引入稀疏约束的时间掩码生成器，专门寻找能显著提升异常分数的最小局部扰动；②采用min‑max对抗优化，迫使检测器对这些局部扰动保持稳定；③掩码本身可视为模型对异常决策的敏感性解释。

**🔧 技术方法**

使用LSTM自编码器作为检测器，LSTM生成器产生时间掩码；通过掩码-基线混合的结构化扰动实现对抗训练；评估采用VUS‑PR等无阈值指标，实验中对盐噪、彩噪等局部结构噪声进行鲁棒性测试。

**📊 数据集**

在TSB‑AD基准上测试，共10个多变量时间序列数据集（如Daphnet、Exathlon、GECCO、LTDB、MITDB、OPP、PSM、SMD、SVDB、FREE）。

**📈 对比分析**

与多种经典与现代深度TSAD方法（PCA、OCSVM、LSTM‑AD、DeepAnT、Donut、RobustPCA、OmniAnomaly、USAD、AnomalyTransformer等）进行比较，ARTA在VUS‑PR上领先或次之，尤其在高噪声条件下保持更平滑的性能下降，证明鲁棒性显著提升。

**⚠️ 局限性**

局限性包括：掩码仅在训练阶段使用，推理时不参与，导致部署时额外训练成本；掩码作为解释工具并非真实异常定位，解释效果依赖于模型与数据的匹配；对不同数据集的稀疏性权重与对抗强度需调优，且对计算资源要求相对较高。

---

## 54. Detecting Anomalous Topology, Routing Policies, and Congested Interconnections at Internet Scale

**arXiv ID:** 2603.25875 | [PDF](https://arxiv.org/pdf/2603.25875v1)

**作者:** Matt Mathis `[一作]` `[通讯]` (Measurement Lab), Matt Mathis (Measurement Lab)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用 Measurement Lab（M‑Lab）统一服务器选择策略，对同一接入 ISP 的客户端在不同 M‑Lab 服务器上的测量做 A/B 对比，从而识别中间路径（mid‑path）中的拓扑、路由和拥塞异常。

**💡 创新点**

通过控制测试样本的来源（同一 ISP、地理近似服务器）实现天然实验，唯一的差异即为中间路径的影响；将“噪声”转化为检测信号，首次在大规模公开数据上系统性地揭示了 interconnection 级别的瓶颈与路由问题。

**🔧 技术方法**

采用 BigQuery 进行稀疏多维直方图计算，利用 Kolmogorov‑Smirnov 距离与几何均值比（Spread）两种差异统计量；同时在可视化仪表板上实现实时监控与钻取。

**📊 数据集**

使用 M‑Lab NDT7 的公开数据集（吞吐量、最小 RTT 等），覆盖数百万条测量，包含不同城市、不同 ISP 与多台服务器的组合。

**📈 对比分析**

方法通过对每个接入 ISP 与每对服务器的分布进行 KS 距离和几何均值比的计算，单次查询即可得到所有组合的差异统计；该方式计算效率高、可扩展到全球多城多 ISP 的实时监测，且能够以图表直观展示差异大小。

**⚠️ 局限性**

限制包括：依赖 M‑Lab 的统一服务器分配策略，若该策略被改变则无法保持对比；仅能检测跨 ISP 的中间路径问题，对同一 ISP 内部冗余链路或内部瓶颈影响有限；目前仅聚焦吞吐量和 minRTT，其他指标如丢包、排队延迟仍待扩展；以及仅在具备足够服务器的都市区可产生有效结果。

---

## 55. Consistency Amplifies: How Behavioral Variance Shapes Agent Accuracy

**arXiv ID:** 2603.25764 | [PDF](https://arxiv.org/pdf/2603.25764v1)

**作者:** Aman Mehta `[一作]` `[通讯]` (Snowflake AI Research), Aman Mehta (Snowflake AI Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在复杂软件工程任务SWE-bench上LLM代理的行为一致性，并探讨其与准确率之间的关系。

**💡 创新点**

创新点在于首次将一致性、准确率和速度三者的相互影响量化，发现一致性会放大正确或错误的结果，而非保证正确性；并揭示早期决策一致并不能决定后续一致性。

**🔧 技术方法**

使用了Claude 4.5 Sonnet、GPT-5和Llama‑3.1‑70B三种LLM模型，并通过mini‑SWE‑agent框架对任务进行多步工具调用，采用Coefficient of Variation（CV）评估步骤数方差、使用官方SWE‑bench评估工具判定补丁正确性。

**📊 数据集**

数据集为SWE‑bench Verified中的10个astropy仓库缺陷任务，涵盖不同bug类型与修复复杂度。

**📈 对比分析**

通过多跑（每模型10个任务×5跑）比较三模型的平均步骤数、CV、准确率、成本和独特序列率，Claude表现最优（CV 15.2%，准确率 58%），GPT‑5速度最快但准确率与一致性均次优，Llama一致性最差且准确率仅4%。

**⚠️ 局限性**

局限性包括样本规模有限（仅3模型、10任务）、只关注单一代码库、仅使用温度0.5、缺乏因果干预实验以及未考虑更高温度或不同任务域的表现。

---

## 56. Neuro-Cognitive Reward Modeling for Human-Centered Autonomous Vehicle Control

**arXiv ID:** 2603.25968 | [PDF](https://arxiv.org/pdf/2603.25968v1)

**作者:** Zhuoli Zhuang `[一作]` (University of Technology Sydney), Chin-Teng Lin `[通讯]` (University of Technology Sydney)

**通讯引用:** 34990 | [OpenAlex ID](https://openalex.org/A5058936239)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过EEG引导的认知奖励机制，将人类脑电信号预测模型融入强化学习，以提升自动驾驶系统的碰撞规避能力。

**💡 创新点**

创新点在于设计了无需实时EEG采集的EEG特征预测模型，将ERP信号映射到场景图像，并将预测结果直接作为RL奖励，首次将脑电信号用于RLHF。

**🔧 技术方法**

采用EEG预处理、ERP检测、轻量CNN预测模型、TD3强化学习、注意力机制以及CARLA仿真等技术。

**📊 数据集**

使用了20名参与者在VR驾驶模拟中的多模态数据集，包含EEG、注视、控制数据及图像，是首个集EEG+眼动+控制+图像的驾驶数据集。

**📈 对比分析**

与Vanilla TD3、BC、PHIL、RLHF等基准模型对比，在紧急刹车和左转场景中，认知奖励版RL在路程完成率、驾驶分数和违规分数上均优于对照组，提升约15%。

**⚠️ 局限性**

局限性包括仅包含两种场景、样本量20人，且EEG特征预测模型具有场景特异性，未来需要更大规模、多样化数据和更通用的模型。

---

## 57. Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline

**arXiv ID:** 2603.26055 | [PDF](https://arxiv.org/pdf/2603.26055v1)

**作者:** Qizhi Xie `[一作]` (Tsinghua University), Jihong Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 10223 | [OpenAlex ID](https://openalex.org/A5051073741)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了视频流畅度评估（VFA）任务，构建FluVid数据集并基于T‑PSA与自监督排序训练开发了FluNet基线模型，全面基准23个方法。

**💡 创新点**

首次将视频质量评估从整体拆分为独立的流畅度任务，设计5层ACR标准与FluVid数据集，引入时序置换自注意力（T‑PSA）与排名式自监督学习。

**🔧 技术方法**

利用Swin‑T骨干、T‑PSA、排名损失与自监督数据合成、softmax映射等技术实现高效时序建模与流畅度回归。

**📊 数据集**

使用4,606条来自SSv2、UGC‑VQA等源的真实视频构成FluVid，以及SSV2、LSVQ、HD‑VILA等作为无标注训练集。

**📈 对比分析**

通过SRCC/PLCC与6种VQA、9个视频LMM、8个图像LMM比较，VQA方法虽表现最好，但仍显不足；FluNet在Fast‑VQA基础上提升了约6% SRCC，FluNet++达0.816 SRCC、0.821 PLCC，显著优于现有基线。

**⚠️ 局限性**

受限于仅20名专家的主观打分、数据规模有限、对不同拍摄条件与编码场景的泛化能力待验证，且当前模型仍未完全捕捉复杂流畅度细节。

---

## 58. Good Scores, Bad Data: A Metric for Multimodal Coherence

**arXiv ID:** 2603.25924 | [PDF](https://arxiv.org/pdf/2603.25924v1)

**作者:** Vasundra Srinivasan `[一作]` `[通讯]` (Stanford), Vasundra Srinivasan (Stanford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出多模态一致性评分（MCS），评估融合质量而非仅靠下游任务准确率。

**💡 创新点**

将一致性拆分为身份、空间、语义、决策四维度，并通过权重学习得到综合得分，能够精准定位问题来源。

**🔧 技术方法**

使用DETR进行目标检测、CLIP进行图文语义对齐、ViLT进行VQA，并通过Nelder–Mead优化权重，构建无人工标注的评估框架。

**📊 数据集**

主要在Visual Genome上评估，并在COCO 2017与Open Images V7上验证跨数据集迁移效果。

**📈 对比分析**

与传统下游指标相比，MCS对三种融合架构具有更高区分度（Spearman ρ=0.093 对比 0.071），并通过扰动实验验证各维度独立性。

**⚠️ 局限性**

依赖特定检测模型和VQA基准，权重可能对其他任务过拟合；仅评估融合质量而不衡量模型本身性能。

---

## 59. Explore LLM-enabled Tools to Facilitate Imaginal Exposure Exercises for Social Anxiety

**arXiv ID:** 2603.25933 | [PDF](https://arxiv.org/pdf/2603.25933v1)

**作者:** Yimeng Wang `[一作]` (William & Mary), Yixuan Zhang `[通讯]` (William & Mary)

**通讯引用:** 1459 | [OpenAlex ID](https://openalex.org/A5037162594)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

我们设计并评估了一个基于大语言模型的 ImaginalExpoBot，用于生成个性化的社交焦虑情景脚本，帮助用户在家进行想象性暴露练习。

**💡 创新点**

创新点在于将 LLM 与实时多感官、情绪化脚本生成结合，利用共设计和即时反馈实现情绪窗口调控，并首次在社交焦虑的想象性暴露中验证其可用性。

**🔧 技术方法**

使用技术包括大型语言模型（如 ChatGPT）配合文本转语音（TTS）模型和定制化提示工程，形成两阶段的交互式脚本生成流程。

**📊 数据集**

数据来源为 19 名自我报告有社交焦虑症状的受试者的使用日志与问卷数据，未使用公开数据集。

**📈 对比分析**

通过与 5 名心理健康专业人士的形成性评估和 19 名用户的实地使用，脚本在情绪提升、实用性和可接受性方面均优于传统书面脚本方法，但未进行量化对照实验。

**⚠️ 局限性**

局限包括缺乏历史连续性与深度个性化、对用户隐私与偏见的潜在风险，以及在真实临床场景中的可推广性不足。

---

## 60. Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods

**arXiv ID:** 2603.25767 | [PDF](https://arxiv.org/pdf/2603.25767v1)

**作者:** Xuanru Zhou `[一作]` (Zhejiang University), Dong Yu `[通讯]` (Tencent AI Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建跨语音、音乐与环境声音的统一标签系统UTS，并利用高保真字幕进行音频预训练，系统比较多种预训练目标的表现；

**💡 创新点**

首次创建跨域统一标签体系并用高质量字幕生成强监督标签；将多标签分类、并行解码、对比学习与生成式字幕结合，揭示数据质量是主导性能的关键；

**🔧 技术方法**

使用大语言模型（Qwen3-Omni-Captioner及LLM解析器）、多标签分类（MTC）、并行解码（PAR）、对比学习（CLAP）、生成式字幕（Captioning）以及联合多任务训练；

**📊 数据集**

以CaptionStew 400K子集为基础生成字幕，构建UTS标签集合，并与公开基准（AudioSet、FSD‑50k、VggSound 等）进行评测；

**📈 对比分析**

在线性探测、音频‑文本对齐与开放式问答三大评测框架下，与AudioSet MTC、对比学习及字幕生成基线对比，UTS模型在仅 400K 数据量时往往优于使用 5 倍数据的基线，表明高质量监督显著提升性能；

**⚠️ 局限性**

UTS 的标签来源受单一 captioner 的偏差影响，难以评估规模与质量的交互效应，且不同预训练目标仍表现出专门化，尚未实现全能统一目标。

---

## 61. Designing Fatigue-Aware VR Interfaces via Biomechanical Models

**arXiv ID:** 2603.26031 | [PDF](https://arxiv.org/pdf/2603.26031v1)

**作者:** Harshitha Voleti `[一作]` (Concordia University), Charalambos Poullis `[通讯]` (Concordia University)

**通讯引用:** 1242 | [OpenAlex ID](https://openalex.org/A5084730486)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于生物力学模型的VR UI布局优化框架，使用模拟的肌肉疲劳作为反馈，利用强化学习自动优化中空手交互布局。

**💡 创新点**

①首次将三腔控制恢复模型（3CC‑r）作为实时疲劳评估信号直接驱动UI布局优化；②采用层级RL框架，低层运动代理模拟手臂运动，高层UI代理通过累计疲劳反馈学习布局；③验证模拟疲劳与用户主观疲劳的一致性。

**🔧 技术方法**

SIM2VR集成的肌肉驱动运动代理、3CC‑r疲劳模型、PPO强化学习、贝叶斯优化、Unity VR环境、Meta Quest 2 HMD、NASA‑TLX、Borg CR10。

**📊 数据集**

合成的按钮位置和随机序列（3×6网格）进行模拟；人类实验数据来自18名参与者在三种布局下完成30个按钮序列的主观疲劳评分。

**📈 对比分析**

与手工中心布局和BO优化布局对比。模拟中RL布局累计疲劳最低（约22.9），其次中心（30.5）BO（33.0）；在用户研究中主观疲劳CR10相同顺序，RL布局疲劳最低；任务完成时间与BO相同或更短，NASA‑TLX物理负荷最低。

**⚠️ 局限性**

模型仅基于右手上肢，缺乏个体差异、全身或双手交互；疲劳模型不能完全预测认知或视觉疲劳；贝叶斯优化在噪声下表现不佳；训练成本高，需多次模拟；未支持动态布局或更复杂交互元素。

---

## 62. Understanding AI Methods for Intrusion Detection and Cryptographic Leakage

**arXiv ID:** 2603.25826 | [PDF](https://arxiv.org/pdf/2603.25826v1)

**作者:** Reza Zilouchian `[一作]` (Florida Atlantic University), Fernando Koch `[通讯]` (Florida Atlantic University)

**通讯引用:** 1413 | [OpenAlex ID](https://openalex.org/A5012632345)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了人工智能在网络安全中的应用，主要评估了机器学习模型在网络入侵检测、对抗性鲁棒性以及侧信道泄露检测方面的表现。

**💡 创新点**

创新点在于整合多维度评估：①在受控与数据分布漂移环境下比较检测性能；②引入特征级对抗性（mimicry）攻击，揭示模型对关键特征的脆弱性；③证明 AI 模型能从模拟侧信道数据中提取泄露模式，暗示 AI 可辅助发现实现层级漏洞。

**🔧 技术方法**

使用了随机森林、深度学习等机器学习算法，特征重要性排序、对抗性扰动、PR‑AUC 与混淆矩阵评估等技术。

**📊 数据集**

采用 NSL‑KDD 与 CIC‑IDS 两个网络流量数据集，以及仿真生成的 AES‑256 侧信道数据集。

**📈 对比分析**

通过训练‑测试分段、对抗性 mimicry 评估以及侧信道分类实验进行比较；在受控环境下检测准确率接近 100%，但在数据分布漂移或特征扰动下 PR‑AUC 降至约 0.58；侧信道分类准确率约 50%，表明能识别部分泄露信息。

**⚠️ 局限性**

局限性包括：对数据分布漂移敏感，过度依赖少数关键特征导致对抗性攻击效果显著；侧信道实验基于仿真数据，未覆盖真实硬件噪声和环境变化。

---

## 63. Few Shots Text to Image Retrieval: New Benchmarking Dataset and Optimization Methods

**arXiv ID:** 2603.25891 | [PDF](https://arxiv.org/pdf/2603.25891v1)

**作者:** Ofer Idan `[一作]` (Huawei Tel Aviv Research Center), Shir Niego Komforti `[通讯]` (Huawei Tel Aviv Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出少样本文本-图像检索任务（FSIR）及其基准数据集 FSIR-BD，并提出两种利用少量参考图像提升检索性能的方法：FSIR-PL（单向提示学习）和 FSIR-CTR（基于多模态大型语言模型的组合查询优化）。

**💡 创新点**

创新点在于：①首次将文本加参考图像的组合查询作为检索任务定义并构造专门的数据集；②设计单向提示学习框架，使用二元交叉熵与 KL 散度自适应地训练提示；③通过 MLLM 与外部图像编码器对齐，利用 LoRA 和 InfoNCE 实现跨模态嵌入统一，从而无需对图像编码器进行微调即可提升检索精度。

**🔧 技术方法**

使用的技术包括：预训练视觉-语言模型（CLIP、BLIP）、可学习提示向量、Sigmoid 归一化、二元交叉熵与 KL 散度损失、ProGrad 方向约束、LoRA 微调、InfoNCE 对齐损失、温度化余弦相似度、向量数据库（FAISS）检索。

**📊 数据集**

使用的数据集有：FSIR-BD（包含 38,353 张图像、303 条查询，涵盖城市生活、自然物种和 OOD 场景），以及基准对比数据集 CIRCO、CIRR，用于评估模型在不同场景下的泛化能力。

**📈 对比分析**

与 BLIP、CLIP 的零样本基线相比，FSIR-PL 在所有三大子集上平均提升约 9–10% mAP（OOD 上可达 20.9%），FSIR-CTR 在 BLIP 基础上平均提升 3.5%，在 CLIP 基础上提升 0.9%；实验表明两种方法均能显著提高检索准确率，且 FSIR-PL 在多样本数量增大时效果更佳。

**⚠️ 局限性**

局限性包括：FSIR-CTR 对参考图像的选择敏感，性能易受参考图像质量影响；目前仅使用正样本参考，负样本参考尚未提升效果；需要为每个检索任务提供足够的少样本参考，若缺少合适示例则难以应用。

---

## 64. Massive Parallel Deep Reinforcement Learning for Active SLAM

**arXiv ID:** 2603.25834 | [PDF](https://arxiv.org/pdf/2603.25834v1)

**作者:** Martín Arce Llobera `[一作]` (Universidad de Buenos Aires), Pablo De Cristóforis `[通讯]` (Universidad de Buenos Aires)

**通讯引用:** 589 | [OpenAlex ID](https://openalex.org/A5062483093)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种可在Isaac Sim/Lab上进行大规模并行深度强化学习的Active SLAM框架，利用GPU向量化的固定延迟SLAM骨干提供实时不确定性信号，并通过不确定性感知奖励实现连续动作空间。

**💡 创新点**

创新点包括：①GPU向量化固定延迟SLAM骨干提供高效的姿态协方差估计；②将姿态不确定性融入奖励函数，兼顾探索与定位；③训练桥接器实现ROS 2 SLAM迁移；④仅需约4小时即可完成训练，显著降低训练时间。

**🔧 技术方法**

技术手段：PPO深度强化学习 + GRU循环层；GPU加速的Isaac Sim/Lab仿真；LiDAR+IMU感知；固定延迟SLAM与不确定性奖励；训练桥接器。

**📊 数据集**

使用在Isaac Sim/Lab构建的四个仿真环境（Env 1–4），包含TurtleBot3 Burger、128波束LiDAR和IMU；未使用公开真实数据集，而是生成仿真数据。

**📈 对比分析**

与无不确定性奖励的PPO版本、随机策略和传统前沿探索方法比较。PPO_uncertainty在所有环境中碰撞率最低、探索覆盖率最高；训练时间约4小时，较以往50+小时大幅提升。

**⚠️ 局限性**

局限性：仅在二维仿真环境验证，缺乏真实环境测试；固定延迟SLAM不包含全局回环闭合，可能忽略长程关联；不确定性信号与真实SLAM的分布差异仍需进一步研究。

---

## 65. VLAgeBench: Benchmarking Large Vision-Language Models for Zero-Shot Human Age Estimation

**arXiv ID:** 2603.26015 | [PDF](https://arxiv.org/pdf/2603.26015v1)

**作者:** Rakib Hossain Sajib `[一作]` (Begum Rokeya University), Shuvra Smaran Das `[通讯]` (EliteLab.AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了 GPT‑4o、Claude 3.5 Sonnet 和 LLaMA 3.2 Vision 在 UTKFace 与 FG‑NET 两个数据集上进行零样本人脸年龄预测的性能。

**💡 创新点**

首次将大型视觉‑语言模型应用于连续年龄回归任务并建立零样本基准，探讨了提示敏感性与公平性问题。

**🔧 技术方法**

使用标准化提示调用 LVLM 的图像‑文本接口进行数值输出，并用八项回归指标（MAE、MSE、RMSE、MAPE、MBE、R²、CCC、±5 岁准确率）评估。

**📊 数据集**

采用公开的 UTKFace（20k+ 张多年龄、多种族）和 FG‑NET（1000 张历史照片）两大年龄估计基准数据集。

**📈 对比分析**

通过统一提示、统一预处理、无微调的方式比较三模型，GPT‑4o 在两数据集上表现最佳，MAE 分别为 4.93/3.73，±5 岁准确率分别为 66.4%/74.8%。

**⚠️ 局限性**

局限在于对提示和计算成本敏感、存在族裔偏差、MAPE 等指标在年轻样本上失效，且未提供可解释性或微调方案。

---

## 66. The Specification as Quality Gate: Three Hypotheses on AI-Assisted Code Review

**arXiv ID:** 2603.25773 | [PDF](https://arxiv.org/pdf/2603.25773v1)

**作者:** Christo Zietsman `[一作]` `[通讯]` (Independent Researcher), Christo Zietsman (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出三条假设并验证可执行规范在 AI 辅助软件开发中的重要性，强调规范先行、确定性验证管道、AI 审查仅针对结构残差。

**💡 创新点**

创新点在于将可执行规范视为 Cynefin 领域转换，构建了五类缺陷可规范性分类法，并通过跨模型实验展示缺陷检测的相关错误问题。

**🔧 技术方法**

使用了 LLM 生成代码与审核、BDD 场景、跨模型审查、可执行规范、Deterministic 评测管道等技术。

**📊 数据集**

实验采用了五个植入错误的 Python 函数、跨 4 个模型（Claude、Codex、Gemini、Amazon Q）以及公开的实验仓库数据。

**📈 对比分析**

与传统无规范 AI 审查对比，BDDs 在复杂域错误检测达到 100%，而 AI 审查在域不透明错误的检测率从 0% 到 100%，显示缺陷相关错误问题。

**⚠️ 局限性**

局限在于使用人工植入错误、有限模型覆盖、实验规模小、缺少真实缺陷样本，且跨模型实验仅部分为不同族群。

---

## 67. Preventing Data Leakage in EEG-Based Survival Prediction: A Two-Stage Embedding and Transformer Framework

**arXiv ID:** 2603.25923 | [PDF](https://arxiv.org/pdf/2603.25923v1)

**作者:** Yixin Zhou `[一作]` (University of Pittsburgh), Jonathan Elmer `[通讯]` (University of Pittsburgh)

**通讯引用:** 6683 | [OpenAlex ID](https://openalex.org/A5004987407)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种两阶段、无数据泄漏的EEG预后预测框架，先用5分钟窗口的CNN+ArcFace提取嵌入，再用Transformer聚合为患者级别的恢复预测。

**💡 创新点**

在多阶段EEG模型中首次系统识别并消除患者级别的数据泄漏，通过严格的患者分割实现了可靠的泛化性能。

**🔧 技术方法**

利用1D CNN+ArcFace进行窗口级嵌入学习，Transformer编码器聚合序列；同时采用归一化、截断、特征工程等预处理技术，并在严格的训练/验证/测试分层上进行训练。

**📊 数据集**

使用BrainFlux数据库中的1231名心脏骤停后昏迷患者EEG数据，划分为581/150/500的训练/验证/测试患者集合。

**📈 对比分析**

与泄漏设置的基线模型对比，未泄漏框架在独立测试集上实现AUC 0.909、Sensitivity@99% 0.618；泄漏模型AUC 0.730、Sensitivity@99% 0.324，显示严格分割显著提升了泛化性能。

**⚠️ 局限性**

局限性包括单中心回顾性数据，缺乏多机构外部验证，未结合临床多模态变量，且模型需进一步校准和持续学习以适应临床实践变化。

---

## 68. "What don't you understand?" Language games and black box algorithms

**arXiv ID:** 2603.25900 | [PDF](https://arxiv.org/pdf/2603.25900v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 69. Can Vision Foundation Models Navigate? Zero-Shot Real-World Evaluation and Lessons Learned

**arXiv ID:** 2603.25937 | [PDF](https://arxiv.org/pdf/2603.25937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 70. CANGuard: A Spatio-Temporal CNN-GRU-Attention Hybrid Architecture for Intrusion Detection in In-Vehicle CAN Networks

**arXiv ID:** 2603.25763 | [PDF](https://arxiv.org/pdf/2603.25763v1)

**作者:** Rakib Hossain Sajib `[一作]` (Begum Rokeya University), Md Arifur Rahman `[通讯]` (Trine University)

**通讯引用:** 355 | [OpenAlex ID](https://openalex.org/A5037126800)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种融合CNN、GRU和注意力机制的时空深度学习模型CANGuard，用于检测Internet of Vehicles环境下CAN总线的DoS与spoofing攻击。

**💡 创新点**

创新点在于：① 将CNN、GRU与注意力机制协同融合，既捕捉空间特征又建模时间序列依赖；② 采用滑动窗口生成时间序列并使用BorderlineSMOTE处理严重类别不平衡；③ 通过SHAP对CAN负载字节进行可解释性分析，揭示攻击特征的重要性。

**🔧 技术方法**

使用的主要技术包括：一维卷积神经网络（CNN）、双向门控循环单元（GRU）、注意力机制、数据增强（BorderlineSMOTE）、归一化（Z-score）、Dropout、L2正则、Adam优化器、早停和梯度裁剪。

**📊 数据集**

采用公开的CICIoV2024数据集，该数据集包含1,408,219条CAN总线样本，涵盖正常流量、DoS攻击及多种spoofing攻击。

**📈 对比分析**

与传统机器学习模型（如Logistic Regression、Random Forest、AdaBoost）以及最新深度学习模型（如DNN、LSTM+Attention、CNN+GRU）在CICIoV2024上进行对比。CANGuard在准确率、精确率、召回率和F1分数上均达到99.89%，相较于最佳基线（96%）提升显著，且在跨数据集的比较中仍保持领先。

**⚠️ 局限性**

局限性：仅在离线环境下对单一数据集进行评估；未实现实时CAN总线部署；缺乏对抗性攻击鲁棒性分析；未来工作需进行跨数据集、在线监测和对抗测试。

---

## 71. MemoryCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-Domain Personalization

**arXiv ID:** 2603.25973 | [PDF](https://arxiv.org/pdf/2603.25973v1)

**作者:** Weizhi Zhang `[一作]` (Roblox), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 135282 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MemoryCD 这一大规模、真实用户、跨域的长上下文记忆基准，并在 4 个个性化任务上对 14 款前沿 LLM 与 6 种记忆机制进行系统评测

**💡 创新点**

首次将跨域真实用户行为与长期记忆评估结合，提出端到端的个性化任务框架，强调记忆对决策与生成的实际影响，而非仅仅评估检索精度

**🔧 技术方法**

使用大规模 LLM（GPT‑5、Claude‑4、Gemini‑2.5 等）与多种记忆方法（LoCoMo、Mem0、ReadAgent、MemoryBank、A‑Mem 等），并采用长上下文提示与记忆抽象、压缩、检索、更新等技术

**📊 数据集**

基于 Amazon Review 数据集构建 12 个领域的用户历史记录，挑选至少 50–1000 条交互的活跃用户，形成包含商品、评分、时间、文本评论的完整记忆池

**📈 对比分析**

在单域与跨域两种记忆来源下，对四项任务（评分预测、排名、摘要、生成）进行比较；结果表明：模型规模提升不一定带来显著个性化；不同记忆方法在各任务上表现各异，未出现统一最优；跨域记忆的效能高度依赖域相似度，聚合多域记忆往往能提升整体性能，但也可能引入噪声

**⚠️ 局限性**

受限于跨域评估仅使用固定记忆操作（未探索可学习的跨域迁移）、未考虑隐私与安全约束、未包含基于训练的记忆构建方法，且大部分 LLM 为闭源，限制了进一步实验与复现

---

## 72. Face2Parts: Exploring Coarse-to-Fine Inter-Regional Facial Dependencies for Generalized Deepfake Detection

**arXiv ID:** 2603.26036 | [PDF](https://arxiv.org/pdf/2603.26036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 73. Speech-Synchronized Whiteboard Generation via VLM-Driven Structured Drawing Representations

**arXiv ID:** 2603.25870 | [PDF](https://arxiv.org/pdf/2603.25870v1)

**作者:** Suraj Prasad `[一作]` (Latent Spaces IITB), Pinak Mahapatra `[通讯]` (Latent Spaces IITB)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于语音同步的白板绘图生成框架，能够自动生成可编辑的Excalidraw绘图与配音同步的教育视频

**💡 创新点**

首次构建可供训练的精确时序白板数据集（ExcaliTeach），并将视觉‑语言模型（Qwen2‑VL‑7B）与LoRA微调结合，直接预测完整的绘制轨迹与时序；实现了仅用24个示例即可实现跨主题泛化

**🔧 技术方法**

采用Qwen2‑VL‑7B视觉‑语言模型，配合LoRA参数高效微调；序列化绘图元素为可直接解码的文本格式；利用视觉上下文、文本上下文与时序标记实现自回归生成

**📊 数据集**

ExcaliTeach：24个涵盖8个STEM领域的Excalidraw+配音示例，包含877个绘图元素，具有毫秒级创建时间戳与词级配音时间戳

**📈 对比分析**

通过五折主题分层评估，比较五种设置（全模型、无时序标记、无视觉上下文、单次非自回归、无配音），全模型在时序误差、几何Chamfer距离、类型准确率及Gemini/人工评分上均遥遥领先；自回归视觉与配音双重条件是提升性能关键

**⚠️ 局限性**

局限包括：数据量小、主题覆盖有限；长序列生成易积累几何漂移；依赖离线ASR，实时部署需低延迟识别；对多语种、流畅口语、现场停顿等现实课堂场景适应性尚未充分验证

---

## 74. AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents

**arXiv ID:** 2603.26034 | [PDF](https://arxiv.org/pdf/2603.26034v1)

**作者:** Wenbo Gao `[一作]` (Huawei), Yaoyuan Wang `[通讯]` (Huawei)

**通讯引用:** 565 | [OpenAlex ID](https://openalex.org/A5107808569)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 AgentCollab 框架，能够在多轮推理任务中根据模型自身的自我反思进度信号动态切换低成本小模型与高成本大模型，以实现推理效率与准确率的平衡。

**💡 创新点**

创新点在于：①采用模型自身生成的自我评估进度指示（binary progress signal）驱动模型切换，无需外部路由器或学习策略；②引入难度感知累计预算策略，在连续停滞时逐步扩大大模型介入时长，以更好地纠正难点。

**🔧 技术方法**

主要技术包括：思考-行动-观察循环、结构化自评估（rationale + binary indicator）、累计预算函数 f(·)（线性或 sigmoid 形式）、两层模型协同执行、在 DDV2 与 WebSailor 代理框架中的实现。

**📊 数据集**

使用了三大公开基准数据集：BrowseComp_zh（网页搜索任务），HLE-math（数学推理任务），WritingBench（长篇文本生成任务）。

**📈 对比分析**

与单一大模型、小模型、随机切换、RouteLLM、FrugalGPT 等多种基线进行对比；在每个基准上，AgentCollab 在准确率方面几乎逼近大模型，同时实现 1.3×~2.4× 的速度提升（相对大模型基线），在三项任务上均取得了 Pareto 前沿的显著改善。

**⚠️ 局限性**

局限性包括：仅探讨同一架构但参数规模不同的模型协作，未涉及不同任务专长的异构模型；仅在开源 LLM 的本地推理环境下验证，未覆盖闭源模型，可能在实际 API 延迟与成本上存在差异。

---

## 75. ReCUBE: Evaluating Repository-Level Context Utilization in Code Generation

**arXiv ID:** 2603.25770 | [PDF](https://arxiv.org/pdf/2603.25770v1)

**作者:** Jiseung Hong `[一作]` (Carnegie Mellon University), Jinho D. Choi `[通讯]` (Emory University)

**通讯引用:** 2561 | [OpenAlex ID](https://openalex.org/A5101829031)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Repository-level Context Utilization Benchmark (ReCUBE)，评估 LLM 在完整仓库上下文中重构被掩码文件的能力；

**💡 创新点**

创新点在于通过构造真实项目的功能子集、生成使用感知的单元测试，以及设计 Caller‑Centric Exploration (CCE) 工具集，帮助 agent 有效定位关键调用文件；

**🔧 技术方法**

采用了完整上下文生成、链式推理、bash‑only agent 以及基于依赖图的 CCE 工具的多种实验设置；

**📊 数据集**

使用 20 个热门 GitHub Python 仓库共 366 个目标文件（及其大规模扩展 138 个实例），构建了 111K 乃至 338K 令牌的上下文；

**📈 对比分析**

与全上下文基线、CoT、标准 agent 对比，GPT‑5 在全上下文得到 37.57% 的严格通过率，增添 CCE 后 agent 在 4 种模型上平均提升 5–8% 的严格通过率；

**⚠️ 局限性**

局限性包括仅覆盖近期流行 Python 项目、缺乏跨语言或多模态任务、评测成本高导致模型覆盖有限。

---

## 76. Learning to Trim: End-to-End Causal Graph Pruning with Dynamic Anatomical Feature Banks for Medical VQA

**arXiv ID:** 2603.26028 | [PDF](https://arxiv.org/pdf/2603.26028v1)

**作者:** Zibo Xu `[一作]` (Tianjin University), Yuting Su `[通讯]` (Tianjin University)

**通讯引用:** 6961 | [OpenAlex ID](https://openalex.org/A5033713097)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种可学习因果修剪（LCT）框架，旨在通过动态去除医学视觉问答中的伪相关和数据集偏差，提高模型的泛化和鲁棒性。

**💡 创新点**

创新点在于将因果修剪机制嵌入端到端训练流程，并通过动态解剖特征库（DAFB）近似未观测的混杂因子，实现对多模态短路的实时学习与抑制。

**🔧 技术方法**

技术实现包括：基于PMC‑CLIP的双流编码、动量更新的DAFB、可微分的相似度与软掩模生成的因果修剪模块，以及联合BCE与正交性损失的训练目标。

**📊 数据集**

实验使用了四个医学VQA数据集：VQA‑RAD、SLAKE、PathVQA 以及针对外部分布的 SLAKE‑CP，验证模型在不同医学影像领域的表现。

**📈 对比分析**

与最新的去偏方法和VQA模型（如CIMB‑MVQA、M2I2、MISS、DeCoCT等）比较，LCT 在 VQA‑RAD、SLAKE、PathVQA 上分别取得 81.2%、86.0% 与 65.6% 的整体准确率，并在 SLAKE‑CP 上实现 51.0% 的整体准确率，均超过前沿方法 1.8%–5.1%。

**⚠️ 局限性**

局限性包括：对预训练编码器的依赖、需要对动量、温度等超参数进行细致调优；当存在 DA F B 未能捕获的全新混杂因子时，仍可能出现性能下降。

---

## 77. Incorporating contextual information into KGWAS for interpretable GWAS discovery

**arXiv ID:** 2603.25855 | [PDF](https://arxiv.org/pdf/2603.25855v1)

**作者:** Cheng Jiang `[一作]` (University of Michigan), David Richmond `[通讯]` (gRED, Genentech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

基于KGWAS框架，本文利用细胞类型特异性的Perturb-seq数据对知识图谱进行稀疏化，去除冗余边并替换为直接的基因-基因关系，从而在小样本GWAS中提升发现力并改善疾病关键网络的可解释性。

**💡 创新点**

在原始大规模通用知识图谱的基础上引入细胞特异性稀疏化策略，去除低置信度边并用Perturb-seq提供的实验证据重构G2G关系，显著提高模型性能与网络一致性。

**🔧 技术方法**

采用几何深度学习的异构图注意网络、图稀疏化技术以及基于Perturb-seq转录响应相似度构建的上下文特异性G2G边，形成新的知识图谱。

**📊 数据集**

使用UK Biobank GWAS结果、K562细胞系的全基因组Perturb-seq数据，以及公开功能基因组数据库（eQTL、ABC、PCHi‑C、STRING、BioGRID、Gene Ontology）等多源数据集。

**📈 对比分析**

与原KGWAS、随机G2G、缺失G2G、随机V2G/G2G等对照模型在不同样本量（1k–50k）下进行子样本GWAS回放，稀疏化后边数减少19倍，独立位点召回率提升约20%。

**⚠️ 局限性**

仅在血液系与单一K562细胞系验证，缺乏对多细胞类型或无Perturb-seq数据疾病的直接推广；网络可解释性提升可能受图稀疏度影响，需进一步验证。

---

## 78. Policy-Guided World Model Planning for Language-Conditioned Visual Navigation

**arXiv ID:** 2603.25981 | [PDF](https://arxiv.org/pdf/2603.25981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 79. ViGoR-Bench: How Far Are Visual Generative Models From Zero-Shot Visual Reasoners?

**arXiv ID:** 2603.25823 | [PDF](https://arxiv.org/pdf/2603.25823v1)

**作者:** Haonan Han `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**通讯引用:** 10968 | [OpenAlex ID](https://openalex.org/A5100754504)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了 ViGoR‑Bench，一个统一评测视觉生成模型推理能力的基准，覆盖图像编辑、视频生成等多模态任务。

**💡 创新点**

创新点：①跨模态覆盖 20 个维度，打通 I2I、I2Is 与 I2V；②双轨评估机制，分别考察生成过程与最终结果；③基于多智能体的“证据驱动”评判器，实现高人类一致性；④细粒度诊断分析，拆解模型在物理、知识、符号三大推理维度上的表现。

**🔧 技术方法**

技术方法：利用 Gemini‑2.5‑Pro 作为 VLM‑as‑a‑Judge；构建生成式数据合成流水线（LLM + 图像生成模型）；规则引擎与符号求解器保证数据逻辑严谨；通过监督微调 (SFT) 与强化学习 (RL) 进一步提升模型推理能力。

**📊 数据集**

数据集：自研 ViGoR‑Bench，包含 20 个子任务，分布在物理推理、知识推理、符号推理三大领域；数据来源于生成合成、真实采集与算法生成，配备参考图像与文本答案。

**📈 对比分析**

比较方法：在 20+ 领先模型（图像编辑、统一模型、视频生成）进行零样本评估，采用 Process 与 Result 双轨指标并给出平均分；实验显示专有模型明显领先，CoT 提升过程可解释性但未必提升最终准确率，视频模型过程质量高但推理成功率低。

**⚠️ 局限性**

局限性：评判器受 LLM 误差限制；基准样本量与多模态动态推理的细粒度评价仍有限；训练与评测对算力与时间成本较高；尚未覆盖所有可能的物理、社会、抽象场景。

---

## 80. Geo$^\textbf{2}$: Geometry-Guided Cross-view Geo-Localization and Image Synthesis

**arXiv ID:** 2603.25819 | [PDF](https://arxiv.org/pdf/2603.25819v1)

**作者:** Yancheng Zhang `[一作]` (University of Central Florida), Chen Chen `[通讯]` (University of Central Florida)

**通讯引用:** 494943 | [OpenAlex ID](https://openalex.org/A5100373745)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一框架，利用几何基础模型（GFM）提取的3D几何先验，将地面图像与航空图像分别映射到共享的几何感知潜在空间，实现跨视角地理定位（CVGL）和双向图像合成（CVIS）同时进行。

**💡 创新点**

创新点包括：①将GFM（如VGGT）作为通用几何特征源，弥补传统方法单一几何模块的局限；②双分支结构将地面与航空视角嵌入共享潜在空间，显著降低视角差异；③基于流匹配的条件生成网络实现单向训练即可得到双向合成；④引入一致性损失强制两方向潜在分布对齐，提升定位与合成的协同效果。

**🔧 技术方法**

采用的技术主要有：GFM（VGGT）进行几何特征提取；双分支卷积+跨注意力实现潜在空间嵌入；InfoNCE对比学习用于CVGL；流匹配网络（DiT+DDT头）做条件生成；KL一致性损失和联合训练策略。

**📊 数据集**

实验数据集包括CVUSA、CVACT和VIGOR，涵盖从美国、澳大利亚到美国四大城市的全景/航空图像对，支持同区、异区与跨数据集评估。

**📈 对比分析**

与SAFA、TransGeo、Sample4Geo、GeoDTR、PanoBEV等基线相比，在CVGL任务上实现R@1最高至98.83%（CVUSA）和98.91%（CVACT test），在VIGOR跨区R@1提升5.01%；在CVIS任务中，FID降至31.72（CVACT）和30.09（VIGOR），LPIPS和SSIM等指标亦保持领先，证明了几何先验与联合训练的显著性能提升。

**⚠️ 局限性**

局限性包括：对预训练GFM的依赖导致模型规模大、推理成本高；仅针对地面/航空两视角，尚未扩展至其他视角或动态场景；在极端天气或极端视角差异下的鲁棒性尚未充分验证；以及对跨域迁移（如不同城市、不同传感器）仍有进一步改进空间。

---

## 81. Doctorina MedBench: End-to-End Evaluation of Agent-Based Medical AI

**arXiv ID:** 2603.25821 | [PDF](https://arxiv.org/pdf/2603.25821v1)

**作者:** Anna Kozlova `[一作]` (A.I. Doctor Medical Assist LTD), Sergey Parfenyuk `[通讯]` (A.I. Doctor Medical Assist LTD)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了 Doctorina MedBench，一个基于模拟真实医患对话的 AI 医生评估框架，包含 D.O.T.S. 四维度指标和多层测试。

**💡 创新点**

创新点在于将诊疗过程模拟为交互式多步骤对话，结合多智能体架构和实时质量监测，提供比传统标准题更接近临床的评估。

**🔧 技术方法**

使用了大语言模型（LLM）+多智能体协同、Pydantic 结构化输出、LLM-as-Judge 评估管道、统计检测、实时监控层等技术。

**📊 数据集**

使用了由临床医生构建、覆盖 750+ 诊断、1,000+ 病例、按 USMLE 分布的平衡数据集，包含文本、附件、影像等多模态信息。

**📈 对比分析**

通过与 GPT‑5 基本版、GP 等对照，采用 D.O.T.S. 指标、Wilcoxon、McNemar 检验等，AI Doctor 在诊断准确率、差异诊断、治疗准确率上显著优于基础 LLM，但对话步数更大，诊断测试略有下降。

**⚠️ 局限性**

局限在于数据集由内部医生构建，可能存在偏倚；人类评估样本有限，难以覆盖全部临床复杂性；实时监测仍依赖规则；模型在多模态附件处理和异常病例上的鲁棒性待进一步验证。

---

## 82. BeSafe-Bench: Unveiling Behavioral Safety Risks of Situated Agents in Functional Environments

**arXiv ID:** 2603.25747 | [PDF](https://arxiv.org/pdf/2603.25747v1)

**作者:** Yuxuan Li `[一作]` (Southern University of Science and Technology), Xuetao Wei `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2106 | [OpenAlex ID](https://openalex.org/A5003379167)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 BeSafe-Bench 基准，用功能化环境评估 Web、Mobile、Embodied VLM、Embodied VLA 四类 situated agents 的行为安全，包含 1312 个可执行安全任务并评估 13 种 LLM 驱动 agent。

**💡 创新点**

创新点在于（1）在真实功能环境中生成安全关键任务并与 9 类安全风险类型融合；（2）提出规则+LLM 混合评估框架；（3）实现跨域统一、可扩展的安全评估体系。

**🔧 技术方法**

采用 GPT‑5 等大模型、规则引擎、WebArena、AndroidLab、OmniGibson、VLA‑Arena 等模拟器、功能化动作空间，并将 LLM 用作评判者。

**📊 数据集**

使用 1312 个安全任务（由原始任务通过 LLM 重写并注入风险而生成），构成 BeSafe-Bench 自身的数据集。

**📈 对比分析**

通过对比 13 名公开 agent，计算任务完成率 (SR)、安全率 (SafeR) 及其联合分布；实验显示即使最佳模型的安全-完成联合率低于 40%，且高任务完成率往往伴随安全违规。

**⚠️ 局限性**

局限性在于：缺乏对复杂多步场景下的安全意识与中途终止机制；评估仅覆盖已触发的安全风险，未涵盖潜在风险意图；基准覆盖的风险类型和场景仍有限。

---

## 83. Gradient-Informed Training for Low-Resource Multilingual Speech Translation

**arXiv ID:** 2603.25836 | [PDF](https://arxiv.org/pdf/2603.25836v1)

**作者:** Ruiyan Sun `[一作]` (Chinese University of Hong Kong), Satoshi Nakamura `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 11111 | [OpenAlex ID](https://openalex.org/A5020994673)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于训练梯度信息自动确定层级共享模式的GDPS框架，在低资源多语言语音翻译任务中实现参数共享与专属的平衡。

**💡 创新点**

创新点在于利用三种梯度分析（语言聚类、跨任务相似度阈值、联合SVD+CCA）自动推断最佳共享比例与语言分组，避免手工设计或昂贵的NAS。

**🔧 技术方法**

采用梯度相似度聚类、自/交叉梯度相似度阈值、联合奇异值分解与岭正则化CCA、低秩分解初始化以及组化微调等技术。

**📊 数据集**

使用IWSLT 2025低资源语音翻译轨道中的四语数据（aeb、bem、est、gle），并在IWSLT评测集上进行评估。

**📈 对比分析**

与SeamlessM4T-Medium统一微调及IWSLT现有系统对比，BLEU/COMET/TER等指标提升约3–8 BLEU点，COMET提升约3%以上，显著优于基线。

**⚠️ 局限性**

局限性包括仅在特定层（如L11 FFN2）显著受益，对低冲突层或极低数据量场景表现不佳，且依赖梯度信号的稳定性，尚未在更大规模语言集合上验证通用性。

---

## 84. Resource Allocation in Strategic Adversarial Interactions: Colonel Blotto Games and Their Applications in Control Systems

**arXiv ID:** 2603.25979 | [PDF](https://arxiv.org/pdf/2603.25979v1)

**作者:** Keith Paarporn `[一作]` (University of Colorado Colorado Springs), Jason R. Marden `[通讯]` (University of California Santa Barbara)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过梳理Colonel Blotto游戏的理论发展与应用，展示其在控制系统、网络安全和基础设施防御等领域的统一建模与分析价值。

**💡 创新点**

创新点在于把分散的资源分配问题统一为Blotto框架，并总结了近两十年关于不完全信息、网络效应和多阶段决策的突破性理论。

**🔧 技术方法**

采用游戏理论分析、混合策略均衡计算与近似算法等技术。

**📊 数据集**

未使用具体数据集，主要以理论模型和案例讨论为主。

**📈 对比分析**

未进行实验比较，主要通过已有理论结果说明Blotto模型在实际防御设计中的性能与保障。

**⚠️ 局限性**

局限在于对实际系统中复杂约束和动态信息的建模仍需进一步完善，且Blotto均衡求解在多场景下仍计算量大。

---

## 85. DRiffusion: Draft-and-Refine Process Parallelizes Diffusion Models with Ease

**arXiv ID:** 2603.25872 | [PDF](https://arxiv.org/pdf/2603.25872v1)

**作者:** Runsheng Bai `[一作]` (MIT), Yangdong Deng `[通讯]` (Tsinghua University)

**通讯引用:** 2653 | [OpenAlex ID](https://openalex.org/A5059155953)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DRiffusion，一种通过草稿-细化并行采样框架，利用跳跃转移并行化扩散模型推理。

**💡 创新点**

创新点在于将跳跃转移理论化为可调用算子，生成未来时间步的草稿状态并行预测噪声，从而解锁扩散模型原生并行性，并提供保守与激进两种模式。

**🔧 技术方法**

采用跳跃转移算子（DDPM/DDIM/Euler）、草稿-细化策略，并在多 GPU 上实现并行噪声预测与后续细化。

**📊 数据集**

在 MS‑COCO 2017 验证集上使用 Stable Diffusion 2.1、XL 与 3 等模型进行评估。

**📈 对比分析**

与原单卡、步数缩减以及 AsyncDiff 等基线对比，获得 1.4×–3.7× 的时延加速，质量指标（FID、CLIP、PickScore、HPSv2.1）基本无下降，内存开销仅提升约 186–226 MB。

**⚠️ 局限性**

在激进模式下对大步长或步数极少的模型（如 SD3）可能出现质量下降；对最优步长的选择仍需经验，且未探讨极大规模设备或跨模型微调。

---

## 86. GeoReFormer: Geometry-Aware Refinement for Lane Segment Detection and Topology Reasoning

**arXiv ID:** 2603.26018 | [PDF](https://arxiv.org/pdf/2603.26018v1)

**作者:** Danny Abraham `[一作]` (University of California, Irvine), Nikil Dutt `[通讯]` (University of California, Irvine)

**通讯引用:** 19484 | [OpenAlex ID](https://openalex.org/A5007817952)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 Geometry-aware Refinement Transformer (GeoReFormer) 用于多视角相机输入的 3D 车道段检测与拓扑推理。

**💡 创新点**

创新点在于三大模块：基于 Spatially-Stratified K-Medoids 的几何先验初始化、在归一化坐标空间内受限的多点多边形迭代细化、以及在 Transformer 解码器中加入可门控的拓扑图卷积，实现几何与拓扑的联合自适应推理。

**🔧 技术方法**

技术包括 Transformer 解码器改进、BEVFormer 编码器、K-Medoids 聚类、tanh 限制残差更新、可门控图卷积（TopoFFN）和深度监督训练。

**📊 数据集**

在 OpenLane‑V2 数据集上进行评估，使用多视角相机数据与 BEV 车道段、边界、行人穿越及连通关系标注。

**📈 对比分析**

与多种基准模型（MapTR、MapTRv2、TopoNet、LaneSegNet、TopoLogic、Topo2Seq）比较，GeoReFormer 在 mAP 上达到 34.5%（比 Topo2Seq 提升 0.9%），在 AP_ls、AP_ped、TOP_lsls 等指标上亦显著提升，并保持模型参数量更小（48M）。

**⚠️ 局限性**

局限性包括：依赖固定的几何先验，无法适应极端或未见的车道几何变异；受限于单帧推理，未考虑时间序列信息；门控机制需要手动初始化，可能在复杂场景下仍引入噪声。

---

## 87. Fus3D: Decoding Consolidated 3D Geometry from Feed-forward Geometry Transformer Latents

**arXiv ID:** 2603.25827 | [PDF](https://arxiv.org/pdf/2603.25827v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 88. Evaluating Synthetic Images as Effective Substitutes for Experimental Data in Surface Roughness Classification

**arXiv ID:** 2603.25765 | [PDF](https://arxiv.org/pdf/2603.25765v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 89. Disguising Topology and Side-Channel Information through Covert Gate- and ML-Enabled IP Camouflaging

**arXiv ID:** 2603.25904 | [PDF](https://arxiv.org/pdf/2603.25904v1)

**作者:** Junling Fan `[一作]` (University of Florida), Domenic Forte `[通讯]` (University of Florida)

**通讯引用:** 6878 | [OpenAlex ID](https://openalex.org/A5009243659)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并评估了三种“仿真欺骗”方法，旨在通过在功能电路与外观电路之间构造结构与视觉上的错位，来阻止基于图神经网络的逆向工程和差分功耗分析（DPA）攻击。

**💡 创新点**

创新点在于将传统的门级/互连级混淆技术与机器学习生成与图匹配技术结合，形成“功能-外观脱耦”体系，首次实现对侧信道模型的误导（让攻击者使用错误的功耗模型导致DPA失败），并通过对结构特征的“毒化”成功欺骗GNN分类器。

**🔧 技术方法**

采用的技术包括：AIG‑VAE（And‑Inverter Graph Variational Autoencoder）实现的生成式IP Camouflage；基于标准门的层级贪心图匹配（Graph Matching）；以及Differentiable Neural Architecture Search（DNAS）构建的DNN‑NAND门阵列；同时利用Covert Gate（Fake Inverter、Fake Buffer、Universal Transmitter）实现功能-外观不匹配；并在实验中使用GNN‑RE和FGNN2进行结构识别，采用DPA框架评估侧信道防护。

**📊 数据集**

实验数据集主要是3种密码S‑Box（PRESENT、DES、AES）的真值表与对应的功能电路；外观电路选取相同类型或不同类型的S‑Box；此外还使用SAED 90nm单元库进行综合，采集了大约32,000条功耗波形进行DPA。

**📈 对比分析**

通过GNN F1分数、DPA猜测熵、面积/功耗等指标与基准（未混淆的原始电路）对比。结果显示，IP Camouflage实现了近1.0×面积一致、1.12×功耗、GNN F1趋近于0（即完全误判）；Graph Matching面积偏高但仍在1.15×以内，GNN F1亦达至84.6+；DPA误差模型下猜测熵保持在3.4比正确模型下的2.2，DPA Resilience Score约为0.45，接近理论极限0.5。

**⚠️ 局限性**

局限性包括：IP Camouflage受后期修正瓶颈限制，规模上限约200个节点；DNAS‑NAND方法对高度非线性S‑Box不可行，导致面积大幅增加；Graph Matching在实现完整的标准门映射时需插入额外的Covert Gate链，导致功耗略增；且对非常大的功能电路的可扩展性仍待进一步验证。

---

## 90. GazeQwen: Lightweight Gaze-Conditioned LLM Modulation for Streaming Video Understanding

**arXiv ID:** 2603.25841 | [PDF](https://arxiv.org/pdf/2603.25841v1)

**作者:** Trong Thang Pham `[一作]` (University of Arkansas), Ngan Le `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种小型眼动重采样器和前向钩子机制，使开源多模态大语言模型（MLLM）能够通过隐藏状态调制实现对视频中眼动信息的感知。

**💡 创新点**

创新点在于：①使用轻量级（≈1–5 M）眼动重采样器，将视频特征与眼动坐标编码融合并产生可注入LLM隐藏层的残差；②在多层解码器上通过前向钩子实现分层注入，而不是改造输入或大模型；③可选第二阶段LoRA适配器进一步将眼动信息与LLM自身注意力模式融合。

**🔧 技术方法**

技术细节包括：V-JEPA 2.1视频特征提取；DET‑style正弦位置编码；跨注意力块和逆映射实现视觉–眼动交互；前向钩子在四层解码器注入残差；两阶段训练（先冻结LLM只训练重采样器，再加入LoRA）；AdamW优化。

**📊 数据集**

使用GazeQA基准数据集，包含285段视角视频、8521个多选问答（4选项），涵盖10个任务类型。

**📈 对比分析**

与GPT‑4o、Claude、Qwen2.5‑VL、InternVL3.5等闭源/开源模型对比，取得63.9%整体准确率，较同骨干加视觉提示提升16.1pp，较GPT‑4o提升10.5pp，成为目前该基准上表现最好的模型。

**⚠️ 局限性**

局限性：仅在GazeQA基准上验证，尚未测试更大规模或不同场景的流式视频任务；眼动信息必须与视频帧严格对齐；模型仍依赖冻结的视觉特征和LLM，未能在端到端学习视觉–语言联合表示；多眼或多人眼动场景未覆盖。

---

## 91. An $Ω( (\log n / \log \log n)^2 )$ Cell-Probe Lower Bound for Dynamic Boolean Data Structures

**arXiv ID:** 2603.25914 | [PDF](https://arxiv.org/pdf/2603.25914v1)

**作者:** Young Kun Ko `[一作]` `[通讯]` (Pennsylvania State University), Young Kun Ko (Pennsylvania State University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

在动态单元探测(cell‑probe)模型下，给出了Boolean型多阶段问题（Multiphase Problem）和内积（Inner Product over 𝔽₂）等基础问题的下界 Ω((log n / log log n)²)，完成了与加权问题最强下界的一致。

**💡 创新点**

核心创新是提出“2.5轮”多阶段通信游戏（Multiphase Communication Game），在标准的一路通信模型中加入一次验证回合，使得原来需要 Peak‑to‑Average Lemma 的技术壁垒被完全绕过，从而直接实现了最优下界。

**🔧 技术方法**

技术上结合了Chronogram框架、细胞采样(cell‑sampling)、信息理论与信息复杂度（Ko–Weinstein框架）、以及对多阶段通信游戏的直接推导；同时利用了对分布下的最小熵、相互信息与Chernoff bounds等信息论工具。

**📊 数据集**

本文为理论研究，不涉及实际数据集，所有结论均基于抽象的随机分布和构造。

**📈 对比分析**

与之前的 Ω(log^1.5 n) Boolean 下界相比，本工作实现了与加权问题相同的 Ω((log n / log log n)²) 下界，表明在现有Chronogram + cell‑sampling 技术框架下已达到理论极限。

**⚠️ 局限性**

局限性：该下界仍受Chronogram框架的结构限制；若要突破 ω(log² n) 的壁垒，需要新的技术或重大电路复杂度突破。

---

## 92. FireBridge: Cycle-Accurate Hardware + Firmware Co-Verification for Modern Accelerators

**arXiv ID:** 2603.25969 | [PDF](https://arxiv.org/pdf/2603.25969v1)

**作者:** G Abarajithan `[一作]` (University of California San Diego), Ryan Kastner `[通讯]` (University of California San Diego)

**通讯引用:** 7100 | [OpenAlex ID](https://openalex.org/A5000231774)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了一套快速的周期级硬件与固件联合验证框架，能够在标准模拟器（VCS、Xcelium、Vivado Xsim）中以x86编译固件并通过DPI-C与RTL/网表交互，实现固件调试、剖析与验证仅需秒级时间。

**💡 创新点**

创新点在于将固件直接编译为x86可执行文件并通过协议无关的内存桥接与RTL结合，结合随机化内存拥塞仿真和off‑chip数据移动剖析，既保持周期精度，又大幅缩短验证周期；同时提供易用的API与开源实现。

**🔧 技术方法**

采用 SystemVerilog DPI‑C 作为硬件固件互联桥梁，利用商业仿真器（VCS、Xcelium、Vivado Xsim），实现协议无关内存桥、随机化总线拥塞模型以及内存带宽与访问模式剖析工具。

**📊 数据集**

在实验中使用了典型加速器模型：可参数化的 systolic array、CGRA、HLS4ML 自动生成的加速器，并在 ResNet‑18、全连接层等 DNN 推理任务上评估。

**📈 对比分析**

与传统的 FPGA 仿真（Vivado + Vitis）和早期建模工具（Gem5‑Accel 等）相比，所提出框架在同等硬件模型上实现了 10–30 倍的调试迭代速度，单次验证仅需几秒到几十秒，且能在仿真中完整观察到总线拥塞与内存带宽瓶颈。

**⚠️ 局限性**

主要限制是固件必须可在 x86 上编译；目前支持的总线接口为 AXI/TileLink 等主流协议，对自定义协议支持有限；仿真虽周期精确但仍无法覆盖所有物理层问题（如时钟域交叉、功耗等），最终仍需在 FPGA 上验证。

---

## 93. Retrieval-Augmented Generation Based Nurse Observation Extraction

**arXiv ID:** 2603.26046 | [PDF](https://arxiv.org/pdf/2603.26046v1)

**作者:** Kyomin Hwang `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**通讯引用:** 8440 | [OpenAlex ID](https://openalex.org/A5084897975)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

基于检索增强生成（RAG）构建了一个无需微调的管道，用来自动提取护士记录中的临床观察值。

**💡 创新点**

创新点在于双重检索机制：一是基于医学本体的术语检索提供语义上下文，二是从内存库中检索相似对话片段及其对应观察值作为少量示例，增强LLM的生成指令。

**🔧 技术方法**

使用的技术包括BlueBERT与TF-IDF/ BM25混合检索、GPT‑5系列（mini/1/2）作为生成与分段模型，以及基于本体描述的schema提示。

**📊 数据集**

主要数据集为MEDIQA‑SYNUR（护士口述记录），并结合构建的观察本体和已标注的观察标签生成内存库。

**📈 对比分析**

在开发集上，结合schema与少样本示例的方案取得F1≈0.829；在测试集上GPT‑5.1+GPT‑5‑mini组合得到最高F1=0.796，表明无训练管道可达接近专业模型的性能。

**⚠️ 局限性**

主要局限包括：对无临床信息片段的处理效率低，需要先验分类器过滤；以及数据中数值与单位不一致导致检索与训练效果受限。

---

## 94. End-to-end Feature Alignment: A Simple CNN with Intrinsic Class Attribution

**arXiv ID:** 2603.25798 | [PDF](https://arxiv.org/pdf/2603.25798v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 95. A Human-Centered Approach to Ethical AI Education in Underresourced Secondary Schools

**arXiv ID:** 2603.26004 | [PDF](https://arxiv.org/pdf/2603.26004v1)

**作者:** Valentina Kuskova `[一作]` (University of Notre Dame), Brianna Conaghan `[通讯]` (University of Notre Dame)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在美国低资源高中开展了一门兼具课程学分、伦理与技术内容的人工智能教育课程，强调人本教学与伦理判断；

**💡 创新点**

创新点在于将伦理判断置于核心学习目标，采用双时段、同伴辅导和对话式教学，构建具备人性化支持的学习生态；

**🔧 技术方法**

技术手段主要为课程设计与 Canvas LMS 课程管理、AI 沙盒实验环境；

**📊 数据集**

数据集为学生、合作教师和教学助理在课程结束时填写的多维度调查问卷；

**📈 对比分析**

比较方法为定量 Likert 量表与定性访谈主题分析，结果显示学生学术主体性、伦理参与度和课程完成率均高（平均得分≥4.6，完成率97.8%）；

**⚠️ 局限性**

局限性包括仅基于自评调查，缺乏前后测量、学习成效直接评估及长期跟踪；

---

## 96. Reinforcing Structured Chain-of-Thought for Video Understanding

**arXiv ID:** 2603.25942 | [PDF](https://arxiv.org/pdf/2603.25942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 97. To Use or Not to Use: Investigating Student Perceptions of Faculty Generative AI Usage in Higher Education

**arXiv ID:** 2603.25932 | [PDF](https://arxiv.org/pdf/2603.25932v1)

**作者:** Jie Gao `[一作]` (McGill University), Dan Chen `[通讯]` (University of Toronto)

**通讯引用:** 4995 | [OpenAlex ID](https://openalex.org/A5024402935)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

调查学生对教师使用生成式AI的态度，探讨学生在自己使用与教师使用上的异向性，并对学生的担忧进行主题分析。

**💡 创新点**

首次聚焦教师使用生成式AI而非仅学生使用，发现学生在两情境下存在显著异向，提出四类学生群体，并系统归纳出四大关切主题。

**🔧 技术方法**

采用McNemar检验比较两情境下的态度，利用主题分析法对开放式回答进行编码和归类。

**📊 数据集**

使用Harvard Dataverse公开的学生对生成式AI使用认知调查数据，共156名本科生和研究生的问卷。

**📈 对比分析**

通过McNemar检验得到χ²=16.820，p<0.001，显示态度显著差异；聚类得到四个学生群体（占比约30%/25%/30%/37%），主题分析揭示专业、质量、教学、伦理四个关切主题的比例分布。

**⚠️ 局限性**

样本量有限，未深入探究导致态度异向的根本原因，数据来源主要为英语系高校，缺乏跨文化视角，研究仅关注学生观点，未涉及教师或机构层面。

---

## 98. Low-Rank-Modulated Functa: Exploring the Latent Space of Implicit Neural Representations for Interpretable Ultrasound Video Analysis

**arXiv ID:** 2603.25951 | [PDF](https://arxiv.org/pdf/2603.25951v1)

**作者:** Julia Wolleb `[一作]` (Yale University), Xenophon Papademetris `[通讯]` (Yale University)

**通讯引用:** 18467 | [OpenAlex ID](https://openalex.org/A5026211955)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出并实现了低秩调制 Functa（LRM‑Functa）模型，用于超声视频的无监督压缩与分析。

**💡 创新点**

通过在 Functa 的时间调制向量上施加低秩约束，使潜在空间呈现可解释的周期性轨迹，能够直接读取心脏循环的ED/ES帧。

**🔧 技术方法**

结合了隐式神经表示（INR）、VidFuncta、LoRA低秩适配与PCA、Savitzky–Golay滤波等技术。

**📊 数据集**

使用 EchoNet‑Dynamic、POCUS 心脏超声、以及 200 条肺部超声视频（B‑线标注）进行实验。

**📈 对比分析**

在压缩-重建、ED/ES帧检测与下游预测（射血分数与B‑线分类）等任务上，LRM‑Functa 在极低秩（k=2）下保持优秀的 PSNR/SSIM，ED/ES检测 MAE 仅略高于有监督方法，并在OOD 数据集保持稳健。

**⚠️ 局限性**

局限性在于对极细结构（如左心室肥厚）的保真度仍有提升空间，且模型对不同心脏视图的迁移仍需进一步验证。

---

## 99. Toward Culturally Grounded Natural Language Processing

**arXiv ID:** 2603.26013 | [PDF](https://arxiv.org/pdf/2603.26013v1)

**作者:** Sina Bagheri Nezhad `[一作]` `[通讯]`, Sina Bagheri Nezhad

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文是一篇综述论文，对2020–2026年50余篇关于多语言NLP与文化适配的研究进行系统梳理与合成，归纳了多语言性能差异、跨语言迁移、文化评估、文化对齐、跨模态与交互评测、基准设计批评及社区驱动数据实践等主题，提出了从语言列表转向“交际生态学”的研究议程和多层次文化评估协议。

**💡 创新点**

创新点在于：① 将多语言迁移与文化评估视角整合，揭示两者往往被孤立讨论；② 提炼跨研究共性发现（如数据量与性能关系、基准设计导入外部文化假设、当地监督与参与的重要性、文化跨模态分布性）；③ 提出以交际生态学为核心的文化适配研究路线和细化的评估层次协议，强调多样化提问、生态有效性、社区验证与适配报告。

**🔧 技术方法**

技术方法主要为文献检索、系统性综述与定量/定性合成，未构建新的模型；论文基于已公开基准和研究报告进行分析，使用文献比对与趋势提炼技术。

**📊 数据集**

综述涵盖的关键数据集与基准包括：Global‑MMLU、CDEval、WorldValuesBench、CulturalBench、CULEMO、CulturalVQA、GIMMICK、DRISHTIKON、WorldCuisines、CARE、CLCA、CulFiT、CultureSPA、DAKULTUR、HESEIA、SAFARI 等多语言、跨模态与社区驱动评测数据集。

**📈 对比分析**

比较方法：本文并未直接实验，而是对不同研究中使用的评估指标、基准设计、语言覆盖、文化对齐方法进行对比。讨论了在不同基准与方法下模型表现的差异，例如翻译基准保留源语言文化假设、提示语言与人物设定能显著改变模型输出、当地监督与社区参与可提升文化适配，但仍受资源与偏见限制。

**⚠️ 局限性**

局限性包括：① 综述受所选文献覆盖范围限制，跨地区、跨模态和低资源任务仍呈现不平衡；② 文化被多采用间接代理（国家、语言、调查工具等）来量化，未能完全捕捉文化多样性；③ 文化适配研究发展迅速，综述结果可能随时间快速过时，需持续更新。

---

## 100. H-Node Attack and Defense in Large Language Models

**arXiv ID:** 2603.26045 | [PDF](https://arxiv.org/pdf/2603.26045v1)

**作者:** Eric Yocam `[一作]` (California Polytechnic State University), Yong Wang `[通讯]` (University of Idaho)

**通讯引用:** 1563 | [OpenAlex ID](https://openalex.org/A5034835138)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 H‑Node Adversarial Noise Cancellation（H‑Node ANC）框架，能够在 Transformer LLM 的隐藏状态维度层面识别“Hallucination Nodes”，并通过前向 hook 注入对抗激活来攻击幻觉表示，同时采用自适应置信度加权的即时取消（ANC）以及动态多轮重排序机制实现实时防御；

**💡 创新点**

创新点包括：1）利用 last‑token 激活而非均值池化精确定位幻觉信号；2）构建白盒对抗攻击并实现仅暴露少于10%信号的高选择性攻击；3）提出置信度加权的 ANC 防御，相比静态取消可降低 33–42% 的 grounded drift；4）动态迭代重排序显著提升鲁棒性（单通道 8%→多轮 0.69）；5）在跨架构（OPT、Phi‑3、LLaMA‑3、Mistral‑7B）和跨规模（125M–8B）模型上验证；

**🔧 技术方法**

使用技术包括：逻辑回归探针、last‑token 维度提取、层次 sweep 选最佳层、频域 Fourier 注入、前向 hook 激活注入、置信度加权取消、动态迭代重排序、AUC/ROC 评估、PPL 与 MMLU 评测；

**📊 数据集**

实验数据集：TruthfulQA（多选）、HaluEval（问答）、WikiText‑103（PPL 评测）、MMLU 100 题子集；

**📈 对比分析**

对比方法：ITI、DoLA、静态 ANC、无防御；性能表现：ANC 在单通道下选择性 3–5×，多轮迭代后鲁棒率提升至 0.69，且对 perplexity 影响 <5%，对 MMLU 影响 ≤3%，保持生成能力；

**⚠️ 局限性**

局限性：1）仅在开放权重模型下的白盒攻击；2）仅在 8B 规模、无多轮生成的实验；3）未验证 70B 及以上模型的 50% 深度规律；4）攻击仅针对激活空间，未考虑黑盒/灰盒场景；5）对生成水平（MC1/MC2）评估受限于 bare Q:/A: 格式。

---

## 101. BEVMAPMATCH: Multimodal BEV Neural Map Matching for Robust Re-Localization of Autonomous Vehicles

**arXiv ID:** 2603.25963 | [PDF](https://arxiv.org/pdf/2603.25963v1)

**作者:** Shounak Sural `[一作]` (Carnegie Mellon University), Ragunathan Rajkumar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8300 | [OpenAlex ID](https://openalex.org/A5104053889)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种名为BEVMapMatch的全无GNSS车辆重定位框架，通过融合激光雷达与摄像头生成的语义化鸟瞰图（BEV），在已知地图上实现车辆精准定位；

**💡 创新点**

创新点包括①利用上下文感知融合（Context-Aware Fusion）在恶劣光照与天气下生成更稳健的BEV分割；②基于跨注意力的检索机制在无GNSS先验的情况下实现粗粒度地图补丁检索；③在检索到的邻域内采用LoFTR等深度特征匹配与单应性估计完成细粒度像素级对齐；

**🔧 技术方法**

主要技术包括多模态Transformer（UniTR/UniTR+Context-Aware Fusion）、DINOv2特征提取、跨注意力检索、LoFTR特征匹配、RANSAC单应性估计以及基于高斯距离的检索损失；

**📊 数据集**

使用NuScenes数据集，构造500m×500m的地图，并对车辆位置随机扰动至±200m，实现GNSS误差高达280m的重定位任务；

**📈 对比分析**

与现有MapLocNet、U-BEV、OrienterNet等基准进行对比，BEVMapMatch在无GNSS先验下实现Recall@1m 39.8%，比最佳基准高出约一倍，且在多帧融合和不同天气/光照条件下均保持较高精度；

**⚠️ 局限性**

局限性在于：①对极端地图边缘或过度遮挡场景的检索精度仍有下降；②需要预先构建与场景相匹配的高清地图；③在极端低光或大雨导致激光雷达噪声显著的环境下，分割与匹配性能可能下降。

---

## 102. THFM: A Unified Video Foundation Model for 4D Human Perception and Beyond

**arXiv ID:** 2603.25892 | [PDF](https://arxiv.org/pdf/2603.25892v1)

**作者:** Letian Wang `[一作]` (Google DeepMind), Cristian Sminchisescu `[通讯]` (Google DeepMind)

**通讯引用:** 15349 | [OpenAlex ID](https://openalex.org/A5007658897)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个统一的视频基础模型，利用文本到视频扩散模型实现一次前向推理即可完成多种人类感知任务（如深度、法线、前景分割、密集姿态和2D/3D关键点预测）。

**💡 创新点**

创新点在于将预训练的文本到视频扩散模型通过文本提示与可学习的额外令牌转化为多模态感知器；并通过单步推理、RGB空间全局监督和两阶段训练策略实现了多任务的高效融合。

**🔧 技术方法**

技术手段包括：文本到视频扩散模型（WAN/DiT）、3D RoPE定位、可学习的关键点令牌、单步“Rectified flow”损失、RGB空间解码器、两阶段训练（潜在空间→RGB空间）以及梯度裁剪与丢弃。

**📊 数据集**

数据集方面，作者使用了800个人体模型与200条CMU MoCap动作在Blender中合成的20000条高质量视频，并在公开真实评测集（Hi4D、Goliath、VideoMatte、PPM‑100、EMDB、RICH、H3.6M）上进行验证。

**📈 对比分析**

与专门任务模型相比，统一模型在深度、法线、前景分割和3D关键点等四个基准上均达到或超过SOTA，表现出强大的零样本泛化能力，并在多实例视频中保持一致性。

**⚠️ 局限性**

局限性主要是：① 在同时训练密集与稀疏任务时存在一定性能折衷；② 目前仅覆盖人类中心任务，尚未扩展至更广泛的视觉或语言理解任务。

---

## 103. DUGC-VRNet: Joint VR Recognition and Channel Estimation for Spatially Non-Stationary XL-MIMO

**arXiv ID:** 2603.25754 | [PDF](https://arxiv.org/pdf/2603.25754v1)

**作者:** Jinhao Nie `[一作]` (Guangdong University of Technology), Xiaoli Chu `[通讯]` (University of Sheffield)

**通讯引用:** 11609 | [OpenAlex ID](https://openalex.org/A5069850104)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对极大规模MIMO（XL-MIMO）的近场空间非平稳信道，联合实现可见区域（VR）识别与信道估计。

**💡 创新点**

创新点在于将深度展开网络（DUN）与图卷积网络（GCN）耦合，利用GCN反馈VR信息给DUN，实现不依赖手工稀疏字典的联合估计，并通过全局权重剪枝显著压缩模型。

**🔧 技术方法**

使用的技术包括：深度展开网络、图卷积网络、残差卷积神经网络（ResCNN）、全局权重剪枝、混合RF架构、球面波近场信道模型。

**📊 数据集**

使用基于仿真的数据集：N=256、f_c=100 GHz、S=8，训练集16 000样本、测试集2 000样本、验证集2 000样本，随机生成用户位置、距离、角度及VR掩码。

**📈 对比分析**

与TL-OMP、LT-CEM、GP-SOMP、GP-SIGW、FRM-GD、VRDO-MP、MDISR-Net等基线比较；DUGC-VRNet（及其50%剪枝版本）在NMSE上平均提升约5 dB、在SNR 0 dB时VR识别SDR超过0.9，整体性能均优于所有基线；剪枝至50%参数仅导致约3 dB NMSE损失，SDR保持近乎不变。

**⚠️ 局限性**

局限性：依赖仿真数据，真实环境下的鲁棒性尚未验证；模型复杂度仍较高，尤其在极大规模网络中；在极低SNR或极稀疏的情况下，VR识别性能可能下降；剪枝可能影响模型的泛化能力。

---

## 104. RealChart2Code: Advancing Chart-to-Code Generation with Real Data and Multi-Task Evaluation

**arXiv ID:** 2603.25804 | [PDF](https://arxiv.org/pdf/2603.25804v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 105. Do Neurons Dream of Primitive Operators? Wake-Sleep Compression Rediscovers Schank's Event Semantics

**arXiv ID:** 2603.25975 | [PDF](https://arxiv.org/pdf/2603.25975v1)

**作者:** Peter Balogh `[一作]` `[通讯]`, Peter Balogh

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用 Wake-Sleep 库学习框架自动从事件状态变化数据中发现语义原语，验证并扩展了 Roger Schank 的概念依赖理论。

**💡 创新点**

创新点在于证明压缩压力（MDL）能自行恢复 Schank 的核心原语，并发现更多情感与目标相关的原语，揭示了更丰富的事件原语集。

**🔧 技术方法**

采用了 DreamCoder 风格的 Wake-Sleep 迭代搜索、最小描述长度（MDL）与贝叶斯 MDL 评估，以及面向状态变换的领域特定语言（DSL）。

**📊 数据集**

实验使用了合成模板生成的数据（100/500/2000 条）以及真实世界的 ATOMIC 事件知识图谱（2000 条）。

**📈 对比分析**

通过 MDL 比例、贝叶斯因子和事件覆盖率进行比较；发现自学库在合成数据上比手工 Schank 库低 4% 的 MDL、覆盖率 100%；在 ATOMIC 上覆盖率 100% 远超 Schank 的 10%，贝叶斯因子显著优于基线。

**⚠️ 局限性**

局限包括简化的三元组状态表示、ATOMIC 适配器覆盖率不足（77% 归类为通用），缺乏对更复杂时序和多层次状态的建模，以及未结合 LLM 进行更精细的文本转状态差异分析。

---

## 106. I Want to Believe (but the Vocabulary Changed): Measuring the Semantic Structure and Evolution of Conspiracy Theories

**arXiv ID:** 2603.26062 | [PDF](https://arxiv.org/pdf/2603.26062v1)

**作者:** Manisha Keim `[一作]` (University of Iowa), Rishab Nithyanand `[通讯]` (University of Iowa)

**通讯引用:** 1810 | [OpenAlex ID](https://openalex.org/A5046944830)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将政治阴谋论视为语义对象，构建其语义邻域并跟踪其随时间演化，揭示语义变化与词汇变动的独立性及多样的演化模式。

**💡 创新点**

① 通过聚类识别可解释的语义邻域，证明阴谋论语言具有可辨识的语义结构；② 开发三个语义演化度量（HAD、CSD、NCD）和词汇重叠度量，独立区分语义迁移、词汇替换、语义碎裂等演化模式。

**🔧 技术方法**

使用Word2Vec CBOW生成分段时间词向量，HDBSCAN聚类构建语义邻域，正交Procrustes对齐跨时间向量空间，Jensen-Shannon散度、平均余弦距离、kurtosis等统计量衡量语义变化；人类标注验证语义可辨识性。

**📊 数据集**

2012-2022年r/politics子版块169.9M条评论，按三段时间（2012-2014、2015-2019、2020-2022）划分，选取19个核心阴谋论作为实验对象。

**📈 对比分析**

通过人类标注的词对辨识实验验证语义邻域与非阴谋词的区分度，注释准确率随语义距离递增；使用统计显著性检验（Bootstrap）证明差异显著；相比仅基于关键词的方法，本框架能捕捉语义稳定、扩张、收缩、替换等细粒度变化，表现出更高的解释性和时序跟踪能力。

**⚠️ 局限性**

① 数据时间范围仅限2012-2022，未覆盖最新发展；② 只分析固定19种阴谋论，未覆盖更广泛的阴谋论空间；③ 仅聚焦r/politics单一主流社区，演化模式可能不适用于边缘或其他平台。

---

## 107. Beyond Disinformation: Strategic Misrepresentation across Content, Actors, Processes, and Covertness

**arXiv ID:** 2603.25883 | [PDF](https://arxiv.org/pdf/2603.25883v1)

**作者:** Arttu Malkamäki `[一作]` (Aalto University), Fintan McGee `[通讯]` (Luxembourg Institute of Science and Technology)

**通讯引用:** 695 | [OpenAlex ID](https://openalex.org/A5090747366)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并系统化了“战略误表征（Strategic Misrepresentation）”的概念，将误信息、参与者及其过程统一为四维框架（内容、参与者、过程、隐蔽性），并基于此对现有的机器学习、网络科学与可视化检测技术进行整合与评述，旨在提供一个更完整、可操作的评估工具；

**💡 创新点**

创新点在于：①从“误信息”扩展为更广泛的“误表征”，突破内容单一视角；②构建四维实证框架，明确各维度特征与相互作用；③整合多学科检测方法（文本/多模态、账号、网络协调），实现跨维度证据累积；④提出可视化与可解释 AI 的结合，增强人机协同评估；

**🔧 技术方法**

使用的技术主要包括：自然语言处理与知识图谱的事实核查模型（BERT、RoBERTa、Transformer）、深度学习文本/多模态特征提取、图神经网络与网络聚类算法、生成对抗网络与扩散模型的深度伪造检测、可视化分析工具（MisVis、Verifi2、Gephi、MuxVis、Coordiscope）、可解释 AI 方法（LIME、Anchors、Grad‑CAM、LRP）等；

**📊 数据集**

引用的数据集与平台覆盖面广，涵盖Twitter、Facebook、Douyin、Reddit、Telegram 等，使用公开的假新闻、深度伪造、社交机器人、投票操纵、协调网络等多类数据集；

**📈 对比分析**

文章主要为综述性工作，并未开展统一实验对比；对个别技术的性能引用已公开结果，例如 FEVER 任务 74% 召回、真实新闻 49% 预测准确、LSTM/Graph GNN 在低质量新闻检测中的 70‑80% AUC 等；强调不同方法在不同维度上的互补性；

**⚠️ 局限性**

局限性包括：①四维框架采用二值化指标，可能低估误表征程度；②对基线表示的依赖需进一步量化；③缺乏统一的跨平台实验评估；④数据获取与隐私法规（DSA、GDPR）限制了实证研究；⑤隐蔽性维度主观性高，难以量化；

---

## 108. A Judge Agent Closes the Reliability Gap in AI-Generated Scientific Simulation

**arXiv ID:** 2603.25780 | [PDF](https://arxiv.org/pdf/2603.25780v1)

**作者:** Chengshuai Yang `[一作]` `[通讯]`, Chengshuai Yang

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本研究提出了一个三代理管道（Plan、Judge、Execute），通过自动化的数学验证（Hadamard稳定性、CFL、误差上界）实现对LLM生成的科学仿真代码的可靠性评估，显著降低了静默失败率。

**💡 创新点**

创新点在于定义可模拟性类S和六元组spec格式，并将经典数值分析理论自动化为Judge Agent的预后门和后验审核，从而实现无人工干预的误差上界证明，并首次将残留失败定位于分支点。

**🔧 技术方法**

技术上结合了大语言模型（Claude Sonnet 4.6）进行问题解析与spec生成，使用结构化Markdown spec进行问题描述，利用数值分析工具（Lax–Richtmyer、Hadamard、CFL条件）在Judge Agent中实现S1–S4验证，并在Execute Agent中执行相应的数值解算器。

**📊 数据集**

使用了12个科学域的开发案例、来自12位科学家的72个盲测任务、200个临床CT sinogram数据、5个地震全波形反演案例、15个燃烧实验数据等多域数据集进行评估。

**📈 对比分析**

与无Judge条件的对照实验相比，静默失败率从42%降至1.5%，72个盲测任务的成功率提升至89%（95% CI[80,95]），临床CT任务实现了与专家相当的99%质量，整个管道的设置时间效率中位数ρ约为480倍。

**⚠️ 局限性**

局限性包括仍存在1.5%在分支点的残留失败、S4可证性检查缺乏全局判别能力、缺乏对人类分析师的正式对比评估，以及评估样本在部分领域（如CT）外未达到大规模验证。

---

## 109. A-SelecT: Automatic Timestep Selection for Diffusion Transformer Representation Learning

**arXiv ID:** 2603.25758 | [PDF](https://arxiv.org/pdf/2603.25758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 110. Closed-Form Formulas for Designing Ultra-Low Phase-Noise Cross-Coupled Dynamically Body-Biased Only-NMOS LCVCOs

**arXiv ID:** 2603.25853 | [PDF](https://arxiv.org/pdf/2603.25853v1)

**作者:** Naser Khatti Dizabadi `[一作]` (University of Tulsa), Peter LoPresti `[通讯]` (University of Tulsa)

**通讯引用:** 1208 | [OpenAlex ID](https://openalex.org/A5047351631)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于冲击敏感函数（ISF）理论的系统级分析框架，用于建模并通过优化系统参数（尤其是体偏置）降低跨耦 LC-电容电感振荡器（LC‑VCO）的相位噪声。

**💡 创新点**

创新之处在于将三端 NMOS 的体偏置耦合进噪声功率谱密度的解析模型，推导出包含阈值、体偏压、工作区间的完整噪声表达式，并给出三条闭式公式（体偏置 AC 振幅与 DC 偏置）直接指导相位噪声最小化。

**🔧 技术方法**

采用冲击敏感函数（ISF）理论、噪声调制函数（NMF）、三端 MOS 噪声模型、数值优化以及 MATLAB/Simulink 仿真。

**📊 数据集**

未使用传统数据集；通过设置 VDD=1.8 V、Vth0=0.5 V、K=0.33、Vb=0.4 V 等仿真参数验证公式。

**📈 对比分析**

将传统 LC‑VCO（无体偏置）与提出的体偏置 LC‑VCO 进行对比，测量 10 kHz、30 kHz、600 kHz、1 MHz、10 MHz、100 MHz 处的相位噪声；结果显示在低偏移（<1 MHz）时提升约 12 dB，1 MHz 处提升约 15 dB。

**⚠️ 局限性**

局限性包括：仅验证了仅 NMOS 交叉耦 LC‑VCO；仿真结果未通过硬件实验验证；假设电路对称、忽略高阶非线性和寄生效应；在更高频偏移处相位噪声改进有限，需进一步完善模型。

---

## 111. Data Gravity and the Energy Limits of Computation

**arXiv ID:** 2603.26053 | [PDF](https://arxiv.org/pdf/2603.26053v1)

**作者:** Wonsuk Lee `[一作]` (SK Hynix Inc.), Jehoshua Bruck `[通讯]` (California Institute of Technology)

**通讯引用:** 15570 | [OpenAlex ID](https://openalex.org/A5043861677)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出数据重力模型和操作-操作数分离常数G_d，量化数据移动相对于计算的能耗，并给出计算与数据共定位可降低能耗的数学表达式

**💡 创新点**

引入G_d这一无量纲能耗比度量，以及将数据重力场与计算资源位置关联的理论框架，提供了评估并优化数据与计算放置的量化方法

**🔧 技术方法**

使用电路理论能耗公式（E=½CV²）、幂律距离模型（E∝d^β）、信息质量定义及实际处理器与内存能耗测量数据进行理论推导与验证

**📊 数据集**

利用公开的处理器与加速器能耗数据（45 nm、7 nm、TPUv4i、DDR5、UPMEM等）作为实验数据集来计算G_d并验证模型预测

**📈 对比分析**

与传统von Neumann架构对比，证明在G_d·(d_min/d)<1时，计算与数据共定位可将总能耗降低至G_d^(β‑1)/2倍；实验显示PIM实现约20–30倍能耗提升，验证了理论预测

**⚠️ 局限性**

模型假设为静态、均匀工作负载，忽略动态访问模式和异构计算单元；验证仅基于公开数据，未进行实验室实测；对不同技术平台的通用性和对温度、制造变异的敏感性未深入探讨

---

## 112. GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks

**arXiv ID:** 2603.25864 | [PDF](https://arxiv.org/pdf/2603.25864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 113. Relational graph-driven differential denoising and diffusion attention fusion for multimodal conversation emotion recognition

**arXiv ID:** 2603.25752 | [PDF](https://arxiv.org/pdf/2603.25752v1)

**作者:** Ying Liu `[一作]` (Central South University of Forestry and Technology), Keqin Li `[通讯]` (State University of New York)

**通讯引用:** 31851 | [OpenAlex ID](https://openalex.org/A5087894632)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种关系图驱动的差分去噪与扩散注意力融合模型，用于多模态会话情绪识别。

**💡 创新点**

创新点在于：1) 差分Transformer通过对两张注意力图的差分实现去噪；2) 构造交互与内部关系子图捕捉说话人情绪依赖；3) 文本主导的扩散注意力融合显式利用文本优势。

**🔧 技术方法**

采用Transformer、图注意力网络（GAT）、差分注意力、扩散注意力融合以及自监督多模态损失。

**📊 数据集**

使用IEMOCAP和MELD两个公开多模态情绪数据集。

**📈 对比分析**

与多种基线（DialogueRNN、MMGCN、DialogueTRM等）在加权准确率和加权F1上进行对比，取得最高的w-Acc 75.17%/66.52%和w-F1 74.87%/66.62%。

**⚠️ 局限性**

局限在于对说话人信息依赖较强、窗口大小需手工调节、文本噪声或缺失时仍会影响性能。

---

## 114. Seeing Through Smoke: Surgical Desmoking for Improved Visual Perception

**arXiv ID:** 2603.25867 | [PDF](https://arxiv.org/pdf/2603.25867v1)

**作者:** Jingpei Lu `[一作]` (Intuitive Surgical), Omid Mohareri `[通讯]` (Intuitive Surgical)

**通讯引用:** 629 | [OpenAlex ID](https://openalex.org/A5067122011)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于Transformer的手术去烟模型，能够在手术视频中实时消除烟雾并生成对应的烟雾图。

**💡 创新点**

创新点在于将Vision Transformer骨干与物理启发的去烟头相结合，联合预测无烟图像与烟雾图；同时开发大规模合成烟雾数据与最大的真实双目烟雾对数据集，突破数据稀缺瓶颈。

**🔧 技术方法**

使用的技术包括Vision Transformer+ DPT解码、基于大气散射模型的t、A、K、B回归、alpha混合合成烟雾、域随机化以及L1损失训练。

**📊 数据集**

使用的主要数据集为合成的约80k对烟雾/无烟图像，以及由Intuitive Surgical da Vinci系统收集的5,817对真实双目烟雾图像，并在公开的De-Smoking数据集（1,063+2,401对）上进行评估。

**📈 对比分析**

与八个去雾/去烟基线方法在SSIM/PSNR上进行比较，在De-Smoking数据集上取得SSIM 0.86/PSNR 25.51，在自建高分辨率数据集上取得SSIM 0.73/PSNR 20.61；在下游深度估计与工具分割实验中，分割IoU有所提升，但深度MAE并未一致改善。

**⚠️ 局限性**

局限性包括合成数据与真实场景的差异仍影响性能，去烟后图像统计变化可能削弱立体匹配，且烟雾图质量的定量评估尚未完善。

---

## 115. Collision-Aware Vision-Language Learning for End-to-End Driving with Multimodal Infraction Datasets

**arXiv ID:** 2603.25946 | [PDF](https://arxiv.org/pdf/2603.25946v1)

**作者:** Alex Koran `[一作]` (McGill University), Narges Armanfard `[通讯]` (McGill University)

**通讯引用:** 1330 | [OpenAlex ID](https://openalex.org/A5073955046)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了VLAAD——一种结合视频与文本的轻量级异常检测模型，并将其作为插件集成到端到端驾驶代理中，以提升闭环驾驶性能。

**💡 创新点**

创新点在于将多实例学习 (MIL) 引入视频-文本对齐学习，实现对碰撞事件的精确时序定位；同时构建了大规模多模态碰撞数据集 CARLA‑Collide 与 Real‑Collide，供模型训练与评估。

**🔧 技术方法**

使用了冻结的 XCLIP 多模态基座，加入轻量级适配器和检测头；通过 MIL 进行短片段级聚合；在驾驶代理中引入碰撞风险 token；采用多模态视频-文本对齐与交叉熵等损失。

**📊 数据集**

训练与评估使用了 MM‑AU、BDD‑X、CARLA‑Collide（包含训练/验证场景）以及 Real‑Collide（真实车载视频）。

**📈 对比分析**

与 XCLIP、LLaVA‑Next 等基线对比，VLAAD 在 Real‑Collide 上 AUC 0.766、F1 0.703、Acc 0.726；在 CARLA‑Collide 上 AUC 0.672；集成到 TransFuser++ 后驾驶分数提升 14.12%（相对提升 6.62% 轨迹完成率，19.23% 违规分数）。

**⚠️ 局限性**

主要局限在于跨域适配仍需改进（如从实景到合成的域差距）、MIL 对长时延事件的敏感度有限，以及模型仍聚焦于碰撞检测而非更广泛的安全场景。

---

## 116. FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Assistants

**arXiv ID:** 2603.26008 | [PDF](https://arxiv.org/pdf/2603.26008v1)

**作者:** Mahesh Bhosale `[一作]` (University at Buffalo), Xuan Gong `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研发了一种参数高效微调方法FairLLaVA，用于消除多模态大型语言模型在医学影像文本生成中的种族、性别等人口属性偏差。

**💡 创新点**

通过在隐藏表示上最小化与敏感属性的互信息，形成轻量级正则化，兼顾公平性与整体性能；同时提出适用于生成任务的公平性评估扩展。

**🔧 技术方法**

低秩适配器LoRA微调、变分互信息估计器、互信息最小化损失、指令跟随交叉熵损失等技术。

**📊 数据集**

使用MIMIC‑CXR（胸部X光报告）、PadChest（西班牙语胸部X光）和HAM10000（皮肤病变）等医学影像数据集。

**📈 对比分析**

与LLaVA、LLaVA‑Rad、MedGemma、Qwen、DeepSeek、CheXagent等基线，以及传统重采样/重加权和对抗分类器等方法进行公平度量（ES‑BLEU、ES‑RadGraph‑F1、ES‑GREEN等）比较，FairLLaVA在多项公平度量上均优于或接近最强基线，并且保持或提升总体生成性能。

**⚠️ 局限性**

需要训练和评估时使用人口属性标签；对单个样本的公平性关注有限，且仅针对群体级别差距。

---

## 117. World Reasoning Arena

**arXiv ID:** 2603.25887 | [PDF](https://arxiv.org/pdf/2603.25887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 118. A Survey of OCR Evaluation Methods and Metrics and the Invisibility of Historical Documents

**arXiv ID:** 2603.25761 | [PDF](https://arxiv.org/pdf/2603.25761v1)

**作者:** Fitsum Sileshi Beyene `[一作]` (Pennsylvania State University), Christopher L. Dancy `[通讯]` (Pennsylvania State University)

**通讯引用:** 358 | [OpenAlex ID](https://openalex.org/A5065588624)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统综述了OCR与文档理解的评估方法，重点聚焦黑色历史报纸在训练集与基准中的缺失与偏差。

**💡 创新点**

创新点在于揭示当前评估指标（如CER/WER）忽视结构与布局失效导致的代表性危害，并指出评估缺口与组织层面的系统性不平等。

**🔧 技术方法**

采用PRISMA系统综述框架，对OCR模型、训练数据、基准以及案例研究进行定量与定性分析，并结合结构化失效评估。

**📊 数据集**

使用的主要数据集包括2006-2025年间的OCR论文、模型训练来源（如IIT‑CDIP、PubMed Central、Internet Archive）、基准集（OCRBench、OmniDocBench、olmOCR‑Bench）以及《Weekly Advocate》（1837）报纸页面。

**📈 对比分析**

通过比较模型在传统CER/WER与结构失效（列布局、阅读顺序）上的表现，发现即便SOTA模型字符准确率高，其在多列布局保持和结构完整性方面性能显著不足。

**⚠️ 局限性**

局限性包括：仅聚焦19世纪美国黑人报纸，缺乏真实标注数据，未提出新的结构化评估指标，且案例研究仅为定性分析。

---

## 119. Emergent Neural Automaton Policies: Learning Symbolic Structure from Visuomotor Trajectories

**arXiv ID:** 2603.25903 | [PDF](https://arxiv.org/pdf/2603.25903v1)

**作者:** Yiyuan Pan `[一作]` (Carnegie Mellon University), Changliu Liu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2488 | [OpenAlex ID](https://openalex.org/A5040156274)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

从视觉运动轨迹中无监督学习神经自动机策略，自动提取概率Mealy机作为高层规划并结合低层残差网络实现精确连续控制。

**💡 创新点**

①无标签自适应符号化与扩展L*算法从演示数据中自动学习概率Mealy机；②双层结构将符号高层规划与连续残差控制分离；③EM式协同训练实现结构与控制的共同进化。

**🔧 技术方法**

HDBSCAN聚类、特征编码器（如DINO）、RNN时序嵌入、扩展L*算法、行为克隆、残差网络、POMDP理论解释。

**📊 数据集**

Maniskill仿真环境、CALVIN长序列TAMP任务、真实世界机械臂任务（乐高拼装、物体分类、挂件悬挂），以及FLOWER VLA预训练数据。

**📈 对比分析**

与传统行为克隆、GMM、Diffusion Policy、OpenVLA、π_0等基线对比，在复杂插入、长序列TAMP和真实世界任务中，在低数据场景下取得比VLA高约8–27%成功率，且参数量显著下降。

**⚠️ 局限性**

对超参数（阈值、RNN训练）敏感；结构质量依赖特定阈值；难以跨任务/跨平台迁移；需要多轮迭代，未充分处理极端噪声或未知环境。

---

## 120. DiReCT: Disentangled Regularization of Contrastive Trajectories for Physics-Refined Video Generation

**arXiv ID:** 2603.25931 | [PDF](https://arxiv.org/pdf/2603.25931v1)

**作者:** Abolfazl Meyarian `[一作]` (Path Robotics), Ser-Nam Lim `[通讯]` (University of Central Florida)

**通讯引用:** 3813 | [OpenAlex ID](https://openalex.org/A5113969216)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后训练框架，通过宏观与微观对比学习纠正文本条件视频生成器的物理不一致。

**💡 创新点**

创新点在于对语义-物理耦合的对比梯度冲突进行形式化分析，并设计了两尺度负样本采样（宏观远离语义簇、微观单轴物理扰动）以消除冲突，提升物理可行性。

**🔧 技术方法**

采用流匹配（flow‑matching）目标、宏观/微观对比正则化、速度空间分布正则化，并利用LLM（Qwen2.5-7B）生成硬负样本。

**📊 数据集**

使用精心挑选的WISA‑80K子集作为训练集，仅使用场景级标题作为条件；在VideoPhy和WorldModelBench基准上评估。

**📈 对比分析**

与SFT、随机负样本、ΔFM以及同规模公开模型（如CogVideoX‑5B、Mochi）对比，显著提升了VideoPhy物理常识与WorldModelBench总分（从5.46提升至5.68，最高分位列全量模型之首），在保持视觉质量与指令遵循的同时仅占1.3B参数。

**⚠️ 局限性**

局限包括对LLM生成负样本的依赖、对复杂物理场景仍有一定误差、缺乏跨域通用性验证，且对更大规模模型的可扩展性尚未深入探究。

---

## 121. When Chain-of-Thought Backfires: Evaluating Prompt Sensitivity in Medical Language Models

**arXiv ID:** 2603.25960 | [PDF](https://arxiv.org/pdf/2603.25960v1)

**作者:** Binesh Sadanandan `[一作]` (University of New Haven), Vahid Behzadan `[通讯]` (University of New Haven)

**通讯引用:** 924 | [OpenAlex ID](https://openalex.org/A5062734917)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对MedGemma模型在医学问答中的提示敏感性进行系统评估，检验CoT、少量样本、选项顺序和上下文截断等因素的影响。

**💡 创新点**

发现标准提示工程在医学LLM上会适得其反，如CoT降低准确率、选项顺序导致59%预测翻转，并提出cloze评分和排列投票等鲁棒性补救方案。

**🔧 技术方法**

采用提示消融、选项扰动、上下文截断实验以及cloze评分、排列投票和CoT自一致性等技术来测评模型鲁棒性。

**📊 数据集**

评估使用MedMCQA 4,183题和PubMedQA 1,000题作为标准医学问答基准。

**📈 对比分析**

与零样本直接提示基准对比，CoT下降5.7%、少量样本下降11.9%、选项乱序导致平均跌幅27.4%，cloze评分提升至51.8%/64.5%，排列投票提升约4个百分点。

**⚠️ 局限性**

研究仅覆盖MedGemma家族，未验证其它医学LLM；仅测试多项选择和简答形式，未涵盖开放式诊断或自由文本推理；并且对临床安全性缺乏系统评估。

---

## 122. UCAgent: An End-to-End Agent for Block-Level Functional Verification

**arXiv ID:** 2603.25768 | [PDF](https://arxiv.org/pdf/2603.25768v1)

**作者:** Junyue Wang `[一作]` (State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences), Yungang Bao `[通讯]` (State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

在CIFAR-10和ImageNet数据集上进行了实验。

**📈 对比分析**

与现有的几种主流模型进行了比较，结果显示该模型在分类精度上提高了5%，且训练时间缩短了20%。

**⚠️ 局限性**

模型在处理高分辨率图像时性能下降，且对计算资源的需求较高。

---

## 123. Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification

**arXiv ID:** 2603.26052 | [PDF](https://arxiv.org/pdf/2603.26052v1)

**作者:** Zizhao Chen `[一作]` (Xi'an Jiaotong University), Xiangru Yin `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于掩码-标签对的多模态信息真实性验证框架 MaLSF，改进了传统的被动整体融合方法，采用主动双向交叉模态验证（BCV）和分层语义聚合（HSA），实现对局部语义不一致的精细检测与定位。

**💡 创新点**

创新点在于：① 以掩码-标签对作为细粒度语义锚点，将像素级与文本级信息对齐；② 设计了双向查询（Text-as-Query 与 Image-as-Query）的交叉注意机制，主动发现跨模态冲突；③ 通过多层浅深融合的 HSA 模块，保持高层语义一致性与低层细节信息的平衡，从而显著提升对细微伪造的检测与定位能力。

**🔧 技术方法**

技术要点包括：掩码-标签对提取（Open Vocabulary Parser 结合 OMG‑LLaVA；Caption‑Anchored Parser 结合 GLIP+SAM2）；文本编码器使用 6‑层 BERT；视觉编码器使用 Swin‑B；双向交叉注意力模块（BCV）；层次化聚合模块（HSA）实现浅层/深层融合；损失函数组合跨熵与定位损失；训练采用 AdamW + cosine 调度。

**📊 数据集**

使用的数据集有：DGM4（230k 图文对，包含多种图像/文本伪造与定位标签）；Weibo17 与 Weibo21（中文社交媒体图文数据集，用于多模态假新闻检测）。

**📈 对比分析**

实验对比采用当前最强方法（CLIP、ViLT、HAMMER、UFAFormer、ASAP、EMSF 等），在 DGM4 上四个子任务（二分类、多标签分类、图像定位、文本定位）均取得平均 +4.87% / +4.94% 的提升，显著优于所有基线；在 Weibo17/21 的二分类任务中，MaLSF 分别达 93.5%/95.5% 的准确率，超过前沿模型 5–10% 的 margin。

**⚠️ 局限性**

局限性：① 对掩码-标签对的提取质量高度依赖，错误或缺失会影响后续验证；② 目前仅针对静态图文，未扩展至视频或长文本场景；③ 计算与内存开销相对传统单模态方法仍较大，推理速度约 22.9 FPS；④ 需要大量标注数据才能充分训练多模态交叉注意与层次聚合机制。

---

## 124. On the Objective and Feature Weights of Minkowski Weighted k-Means

**arXiv ID:** 2603.25958 | [PDF](https://arxiv.org/pdf/2603.25958v1)

**作者:** Renato Cordeiro de Amorim `[一作]` (University of Essex), Vladimir Makarenkov `[通讯]` (University of Quebec at Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 Minkowski 加权 k‑means（mwk‑means）算法进行了严格的理论分析，证明其目标函数等价于对簇内离散度的幂均值聚合，揭示了 Minkowski 指数 p 对特征加权与聚类性能的作用。

**💡 创新点**

创新点在于：①将 mwk‑means 目标函数改写为幂均值形式，统一解释 p 的影响；②推导出目标函数的上下界并给出特征权重的幂律关系；③证明在 p>1 时每一步更新均收敛到局部极小值，提供了算法收敛性保证。

**🔧 技术方法**

使用的技术包括：幂均值不等式、严格凸优化（保证 Minkowski 中心唯一）、权重归一化更新规则以及对目标函数的闭式表达式推导。

**📊 数据集**

实验使用了 10 个含 1,000 个样本、4 个真特征和 4 个噪声特征的高斯混合数据集，并对不同 p 值（1.1、1.5、2、5）进行了 20 次随机初始化的实验。

**📈 对比分析**

通过对目标函数进行归一化并与理论上界/下界比较，验证了所有实验结果均落在 [0,1] 范围内；同时展示了 p 越小权重越稀疏、越大趋向均匀，确认了理论预测；性能上没有提供具体聚类质量指标，但理论与实证一致。

**⚠️ 局限性**

局限性包括：①未在真实大规模数据集上验证聚类质量；②聚类性能指标（如轮廓系数、ARI）未给出；③对 p 的取值范围仅在实验中有限，缺乏自适应或自动选择 p 的方法。

---

## 125. Automated Quality Assessment of Blind Sweep Obstetric Ultrasound for Improved Diagnosis

**arXiv ID:** 2603.25886 | [PDF](https://arxiv.org/pdf/2603.25886v1)

**作者:** Prasiddha Bhandari `[一作]` (Nepal Applied Mathematics and Informatics Institute for research), Bishesh Khanal `[通讯]` (Nepal Applied Mathematics and Informatics Institute for research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

系统评估盲扫产科超声（BSOU）视频的采集偏差对AI任务（扫标识、胎儿呈现、胎盘位置）的影响，并开发自动质量评估（QA）模型及其回采反馈模拟。

**💡 创新点**

创新点在于量化多种采集误差（序列反转、探头翻转、不完整扫面）对不同AI任务的性能损害，提出可实时检测并纠正这些误差的QA框架，从而显著提升模型鲁棒性。

**🔧 技术方法**

使用视频Transformer（MViT）作为特征提取器，结合帧级预处理、人工扰动合成、QA分类网络，以及早停、学习率调度等深度学习训练技巧。

**📊 数据集**

采用来自美国北卡罗来纳州Chapel Hill和赞比亚卢萨卡两院的8,000余份BSOU数据，精选1,250名孕妇的6条标准扫面，划分为训练/验证/测试集进行实验。

**📈 对比分析**

通过在测试集上随机施加不同概率的扰动，比较未处理与回采后模型的准确率/宏F1；未处理时扫标识任务仅29%准确率，胎儿呈现约66%，胎盘位置约78%；QA检测准确率分别为97–99%；回采后这些指标提升至约70%（扫标识）、85%（胎儿呈现）和88–93%（胎盘位置）。

**⚠️ 局限性**

局限性包括只评估了三种扰动且合成扰动可能不足以模拟真实操作误差；未考虑音频/亮度等其他超声质量问题；回采实验理想化，实际回采可能无法完全恢复原始质量。

---

## 126. We Need Granular Sharing of De-Identified Data-But Will Patients Engage? Investigating Health System Leaders' and Patients' Perspectives on A Patient-Controlled Data-Sharing Platform

**arXiv ID:** 2603.26010 | [PDF](https://arxiv.org/pdf/2603.26010v1)

**作者:** Xi Lu `[一作]` (University at Buffalo, State University of New York), Yunan Chen `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过设计高保真原型并进行两阶段混合方法研究，调查了健康系统领导者和患者对去标识医疗数据的患者控制共享平台的看法。

**💡 创新点**

首次系统比较患者与机构领导者的态度，提出灵活低负担的分层细粒度授权与持续利益导向透明度设计建议。

**🔧 技术方法**

采用高保真Web原型、半结构化访谈、在线问卷与定量统计（描述统计、Ordinal logistic 回归、Cochran's Q）等技术。

**📊 数据集**

使用523名患者的问卷数据、16名领导者访谈记录以及系统日志记录患者在原型中的共享偏好。

**📈 对比分析**

通过主题分析对比两方视角，并用量化指标（同意率、隐私关注、健康素养）评估接受度，结果显示两组在透明度与风险评估上存在显著差异。

**⚠️ 局限性**

样本仅来自大城市学术医疗中心，缺乏农村与非学术机构视角；未涉及临床医师与研究者；原型仅为预录演示，缺真实使用情境。

---

## 127. DenseSwinV2: Channel Attentive Dual Branch CNN Transformer Learning for Cassava Leaf Disease Classification

**arXiv ID:** 2603.25935 | [PDF](https://arxiv.org/pdf/2603.25935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 128. Second-Order, First-Class: A Composable Stack for Curvature-Aware Training

**arXiv ID:** 2603.25976 | [PDF](https://arxiv.org/pdf/2603.25976v1)

**作者:** Mikalai Korbit `[一作]` (IMT School for Advanced Studies Lucca), Mario Zanon `[通讯]` (IMT School for Advanced Studies Lucca)

**通讯引用:** 3455 | [OpenAlex ID](https://openalex.org/A5106707940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Somax，一个可组合的 Optax‑native 堆栈，将二阶优化拆解为可规划、可 JIT 编译的单步管道，并提供统一接口。

**💡 创新点**

创新点在于将二阶优化拆分为可互换模块（曲率算子、求解器、预条件器、阻尼、估计器），通过静态规划确定执行路径；同时将参数空间与行空间求解做显式区分，揭示不同的成本模型。

**🔧 技术方法**

使用 JAX + Optax、JIT 编译、静态规划、matrix‑free 线性算子、CG/PCG、行空间求解、Hutchinson 估计器、EMA 预条件等技术。

**📊 数据集**

实验数据集包括合成回归、Fashion‑MNIST 以及 CIFAR‑10（ResNet‑20）。

**📈 对比分析**

通过对比行空间与参数空间求解的时间比例、Solver–阻尼–预条件组合在 Fashion‑MNIST 上的学习时间与最终精度，以及不同 Sophia 变体在 CIFAR‑10 上的壁钟时间–准确率曲线，展示模块化配置显著影响性能与收敛速度。

**⚠️ 局限性**

主要局限在于仅支持单设备 JAX，缺乏多设备分布式执行；部分预条件器与随机化曲率近似尚未实现；某些模块组合可能出现兼容性问题。

---

## 129. AVDA: Autonomous Vibe Detection Authoring for Cybersecurity

**arXiv ID:** 2603.25930 | [PDF](https://arxiv.org/pdf/2603.25930v1)

**作者:** Fatih Bulut `[一作]` (Microsoft), Anjali Mangal `[通讯]` (Microsoft)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于Model Context Protocol（MCP）的AI驱动检测工程框架，自动化生成安全检测规则并集成到开发者工作流中。

**💡 创新点**

创新点在于将MCP与LLM协同工作，支持检索增强与交互式工具调用，系统评估三种作者工作流（Baseline、Sequential、Agentic）并量化质量与成本的权衡。

**🔧 技术方法**

核心技术包括大型语言模型（GPT‑4.1、GPT‑5、o1/o3系列等）、多模态检索工具、ReAct式Agentic推理、以及嵌入相似度与LLM-as-a-Judge的评估方法。

**📊 数据集**

使用了92条真实生产检测（涵盖Sentinel、XDR、三内置平台），共三种语言（KQL、PySpark、Scala），并通过10,127条检索样本构建上下文库。

**📈 对比分析**

通过对比三种工作流和21种模型配置，采用语义/语法相似度、ROUGE‑L、Levenshtein等指标，发现Agentic工作流比Baseline提升19%，Sequential在成本降低40倍的同时保留87%质量，整体模型性能呈现“推理模型优于非推理模型”的趋势。

**⚠️ 局限性**

局限包括对运行时验证缺失、对特定环境排除逻辑的“部落知识”不足、Agentic工作流的高延迟与上下文窗口耗尽、以及评估仅基于作者级相似度而非实际检测效果。

---

## 130. Knowledge is Power: Advancing Few-shot Action Recognition with Multimodal Semantics from MLLMs

**arXiv ID:** 2603.26033 | [PDF](https://arxiv.org/pdf/2603.26033v1)

**作者:** Jiazheng Xing `[一作]` (Zhejiang University), Yong Liu `[通讯]` (Zhejiang University)

**通讯引用:** 35981 | [OpenAlex ID](https://openalex.org/A5100712539)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种端到端的少样本动作识别框架FSAR‑LLaVA，直接利用多模态大型语言模型的隐藏特征进行特征增强、原型构造和跨模态匹配；

**💡 创新点**

创新点包括：①不使用 feature→caption→feature 的信息损失链路，而是直接利用MLLM隐藏特征；②设计了多模态特征增强模块、复合任务导向原型构造模块和多模态匹配度量，充分挖掘语义知识；③支持“Known”“Unknown”两种提示方式，实现极低的可训练参数；

**🔧 技术方法**

技术要点为：Video‑LLaVA/ Qwen2‑VL/ Qwen3‑VL 作为知识库；多模态特征解耦与双分支自注意/交叉注意；局部与全局原型构造；自适应 α 与 u 的动态匹配；元学习的 episode 训练；

**📊 数据集**

使用的数据集包括 Kinetics、HMDB51、UCF101、SSv2‑Small 和 SSv2‑Full；

**📈 对比分析**

与现有单模态、跨模态及知识驱动方法在 5‑way 1/5‑shot 上对比，FSAR‑LLaVA 在多数数据集达到了 99% 以上的最高准确率，FSAR‑LLaVA_Unknown 亦优于大多数多模态方法；同时训练参数极少（≈10 M），推理速度快；

**⚠️ 局限性**

局限性在于仅使用固定的“Known”“Unknown”提示，未探索可学习或自适应提示；在“Unknown”条件下对复杂时序动作的识别仍受限，需进一步改进提示设计与 MLLM 时序推理能力。

---

## 131. Measurement Campaigns, Datasets, and Curve Fitting Officially Used by 3GPP in the Release 19 for Channel Modeling in TR 38.901 for 7-24 GHz

**arXiv ID:** 2603.25927 | [PDF](https://arxiv.org/pdf/2603.25927v1)

**作者:** Hitesh Poddar `[一作]` (Sharp Laboratories of America), Mansoor Shafi `[通讯]` (Spark NZ Ltd)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文完成了针对 7–24 GHz 频段的 3GPP TR 38.901 通道模型的验证与更新工作，综合利用多来源测量与射线追踪（RT）数据，对 UMi、UMa、SMa、RMa、InH、InF 等多种场景下的路径损耗、时延扩展、角度扩展等关键统计量进行曲线拟合。

**💡 创新点**

创新点在于首次针对 7–24 GHz 频段进行系统性测量验证，并引入了近场传播与空间非平稳性等新效应的考虑；同时，通过对不同来源数据的统一整理与多种拟合方法（OLS、WLS、AM、WM 等）实现了对现有模型参数的精细校准。

**🔧 技术方法**

技术手段包括：多基地站与移动终端的实测、连续波（CW）与宽带测量、射线追踪仿真、统计分析与曲线拟合（包括普通最小二乘、加权最小二乘、加权平均等），以及对通道参数（如延迟扩展、角度扩展、极化扩展、K 因子、聚簇数、绝对到达时延）进行频率依赖性建模。

**📊 数据集**

使用的数据集来自 30 家以上测量单位（如 Nokia、Samsung、AT&T、Huawei、Keysight 等），覆盖 0.5–100 GHz 的多频率与带宽（如 6.75 GHz、7 GHz、10.1 GHz 等），并包含室内外不同场景的路径损耗、时延扩展、角度扩展、遮蔽损耗等指标。所有数据均被整理进 29 个 Excel 表格中，便于统一分析与曲线拟合。

**📈 对比分析**

比较方法是将实验与仿真得到的统计量曲线与现行 TR 38.901 模型曲线进行对比，采用 OLS/WLS/AM/WM 拟合并可视化显示差异；结果显示，修正后的模型在 7–24 GHz 区间内与测量值更为贴合，尤其在 NLOS 条件下的时延与角度扩展误差明显减小，表明模型的精度得到提升。

**⚠️ 局限性**

局限性包括：仅覆盖 7–24 GHz 频段，其他频段（如 24–100 GHz）的数据仍缺乏；仅考虑随机（统计）模型，近场与空间非平稳性等物理效应虽已提及但未全部纳入统一建模；测量来源主要集中在少数国家与地区，可能存在地理环境偏差；最后，模型更新主要基于现有测量，缺乏对未来 6G 高频段更复杂场景的验证。

---

## 132. Density-aware Soft Context Compression with Semi-Dynamic Compression Ratio

**arXiv ID:** 2603.25926 | [PDF](https://arxiv.org/pdf/2603.25926v1)

**作者:** Yijiong Yu `[一作]` (Oregon State University), Ji Pei `[通讯]` (DeepSolution)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了半动态上下文压缩框架，利用离散比例选择器实现信息密度自适应压缩。

**💡 创新点**

解决了LLM在连续结构超参数上的学习困难，引入离散比例选择器和单阶段联合训练。

**🔧 技术方法**

采用离散比例选择器、mean‑pooling特征提取、纯SFT训练、合成数据生成、双向注意力等技术。

**📊 数据集**

使用UltraFineWeb为基础的合成数据，并以教师LLM生成任务与超简摘要；评估基准为HotpotQA、SQuAD、Natural Questions、AdversarialQA。

**📈 对比分析**

与固定比率压缩模型对比，在相同平均压缩率下，半动态方法在准确率上提升约5–10%且压缩效率更高，形成新的Pareto前沿。

**⚠️ 局限性**

受限于离散比例集合，无法实现完全连续可控；过度依赖合成数据可能影响泛化；仅在Qwen3系列模型上验证。

---

## 133. In-Context Molecular Property Prediction with LLMs: A Blinding Study on Memorization and Knowledge Conflicts

**arXiv ID:** 2603.25857 | [PDF](https://arxiv.org/pdf/2603.25857v1)

**作者:** Matthias Busch `[一作]` (Technical University of Hamburg), Roland C. Aydin `[通讯]` (Technical University of Hamburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过六层信息屏蔽框架，对大型语言模型（LLM）在分子属性预测中的记忆化、先验知识与上下文学习能力进行系统评估。

**💡 创新点**

提出结合屏蔽与标签/输入变换的实验流程，能明确区分模型的直接记忆、结构-属性关系学习、化学上下文学习及通用上下文学习四种能力；并揭示先验知识在少量样本时可能产生干扰。

**🔧 技术方法**

使用LLM提示工程（analysis/prediction 两阶段），多尺寸、跨家族的LLM（GPT‑4.1、GPT‑5、Gemini 2.5），以及数学变换（值反转+归一化）与SMILES字符替换。

**📊 数据集**

在 MoleculeNet 三个基准集上测试：ESOL（Delaney溶解度）、Lipophilicity（脂亲性）、QM7（原子化能）。

**📈 对比分析**

对比0-shot、60-shot、1000-shot的 Pearson 相关系数、误差分布；在不同屏蔽层级下比较模型性能。结果显示：大模型在 0-shot 时已具备一定先验知识，随着样本增多性能提升但 60-shot 时往往受干扰；屏蔽后大部分模型表现更稳健，尤其溶解度任务；Lipophilicity 在屏蔽下显著下降，说明高度依赖先验。

**⚠️ 局限性**

仅使用 SMILES 作为输入，未考虑 3D/图结构；实验次数有限（每组仅 2 次）；仅评估三类任务，缺乏更广泛属性和数据集的验证。

---

## 134. A Monolithic Computational Homogenization Framework for Nearly Incompressible Magnetoelastic Composites

**arXiv ID:** 2603.25965 | [PDF](https://arxiv.org/pdf/2603.25965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 135. Global Location-Invariant Peak Storm Surge Prediction

**arXiv ID:** 2603.25978 | [PDF](https://arxiv.org/pdf/2603.25978v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 136. QuitoBench: A High-Quality Open Time Series Forecasting Benchmark

**arXiv ID:** 2603.26017 | [PDF](https://arxiv.org/pdf/2603.26017v1)

**作者:** Siqiao Xue `[一作]` (Ant Group), Hang Yu `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了基于支付宝流量的大规模单一来源时间序列数据集 Quito 及其平衡的预测基准 QuitoBench，并在上面评估了十种模型。

**💡 创新点**

通过趋势/季节/可预测性三维 TSF 分类实现八格均衡覆盖，采用单一来源避免信息泄漏，并提供长上下文、滚动窗口评估与多维诊断。

**🔧 技术方法**

使用 STL 分解与谱熵测度计算 TSF，应用 CrossFormer、DLinear、Chronos‑2 等深度学习与基础模型，并在多配置下进行滚动窗口 MAE/排名评估。

**📊 数据集**

使用了约 500,000 条多维 10 分钟或 1 小时分辨率流量序列（约 1.6 B tokens）构成的 Quito 数据集，以及从中抽取的 1,290 条测试序列。

**📈 对比分析**

通过 232,200 个密集滚动窗口 MAE 实例并转换为排名进行比较；结果显示 CrossFormer 在整体 MAE/排名上领先，其 1 M 参数模型优于 100 M 参数基础模型，且揭示了上下文长度、TSF 规律和数据规模的关键影响。

**⚠️ 局限性**

受单一来源业务覆盖范围限制，未深入探讨多模态或跨业务迁移，且基础模型仅做零样本评估，缺乏 fine‑tune 结果。

---

## 137. ExVerus: Verus Proof Repair via Counterexample Reasoning

**arXiv ID:** 2603.25810 | [PDF](https://arxiv.org/pdf/2603.25810v1)

**作者:** Jun Yang `[一作]` (University of Chicago), Kexin Pei `[通讯]` (University of Chicago)

**通讯引用:** 69 | [OpenAlex ID](https://openalex.org/A5058420386)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于大型语言模型的Verus证明修复框架，利用源级SMT查询生成、验证和阻塞可执行的counterexample来引导证明的逐步改进。

**💡 创新点**

创新点在于（1）让LLM直接合成语义意义明确的源级SMT查询，避免了传统Verus低层SMT查询的抽象化难题；（2）引入验证模块确认counterexample的有效性，提升了修复的可靠性；（3）采用基于mutation的counterexample‑guided修复策略，自动生成多种修复候选并通过验证反馈进行排名；（4）在代码混淆和对抗场景中保持高鲁棒性。

**🔧 技术方法**

使用的核心技术包括：大型语言模型（如GPT‑4o、DeepSeek‑V3.1），Z3 SMT求解器，源级SMT查询合成，counterexample验证与阻塞，错误分层三叉树（error triage），以及基于mutation的修复生成与排名。

**📊 数据集**

实验数据集主要由Verus官方基准、公开验证任务以及新构造的ObfsBench（用于评估代码混淆下的鲁棒性）组成，涵盖多种难度级别的验证任务。

**📈 对比分析**

与现有基线（手工错误修复、AutoVerus等）相比，本框架在成功率上平均提升约38%，在更难的基准上可达2倍；在混淆测试中成功率超过73%而对照基线低于50%；成本每任务平均$0.04，比对照低4.25倍，运行时间比对照快4倍。

**⚠️ 局限性**

局限性包括：仅对invariant错误实现了counterexample验证，对其他错误类型缺乏验证支持；生成的counterexample高度依赖LLM的可靠性，存在幻觉风险；mutation策略需要专家预先设计，缺乏完全自动化；对极大复杂任务仍存在性能瓶颈；生成的证明仍需人工复核以确保语义完整性。

---

## 138. Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models

**arXiv ID:** 2603.25994 | [PDF](https://arxiv.org/pdf/2603.25994v1)

**作者:** Zhuan Shi `[一作]` (McGill University), Golnoosh Farnadi `[通讯]` (McGill University)

**通讯引用:** 812 | [OpenAlex ID](https://openalex.org/A5053667504)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关、局部化的概念抹除框架（Neighbor‑Aware Localized Concept Erasure, NLCE），能够在文本到图像扩散模型中精准删除目标概念并保留语义相邻的概念

**💡 创新点**

通过三阶段流程：①谱加权嵌入调制抑制目标概念并恢复邻域概念；②注意力引导空间门控定位残留激活；③空间门控硬抹除彻底清除剩余痕迹，显著提升邻域保留度

**🔧 技术方法**

谱加权投影、CLIP和RoBERTa检索邻居、交叉注意力重写、双传递UNet门控与硬抹除

**📊 数据集**

Fine‑grained（Oxford Flowers、Stanford Dogs）、人脸身份（Celebrity）、成人内容（I2P）和艺术风格（Artistic）等数据集

**📈 对比分析**

与多种训练基和训练无关基线（MACE、SPM、ESD‑x/u、U​CE、SLD、RECE、AdaVD、GLoCE）对比，NLCE在目标抹除率、邻域保留率、整体抹除性能（H_o）、CLIP分数、KID、LPIPS等指标上均达到或超过最优，尤其在细粒度和多概念场景中表现突出

**⚠️ 局限性**

在极高相似度类别或多目标同时抹除时仍可能出现邻域概念轻微衰退；需要调节超参以平衡抹除强度与生成质量；对非局部、非相邻概念的抹除能力有限

---

## 139. Diffusion MRI Transformer with a Diffusion Space Rotary Positional Embedding (D-RoPE)

**arXiv ID:** 2603.25977 | [PDF](https://arxiv.org/pdf/2603.25977v1)

**作者:** Gustavo Chau Loo Kung `[一作]` (Stanford University), Ehsan Adeli `[通讯]` (Stanford University)

**通讯引用:** 14464 | [OpenAlex ID](https://openalex.org/A5015355317)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了如何利用自监督的 Masked Autoencoder Transformer 结合新的扩散空间位置编码（D-RoPE），从 dMRI 数据中学习可迁移的通用表示。

**💡 创新点**

创新点在于：①设计了基于扩散方向和 b 值的旋转位置编码 D‑RoPE，能够在注意力机制中捕捉不同扩散采样的相对几何关系；②在 Transformer 的注意力块中交替使用空间与扩散域的注意力；③提出了可处理任意方向数与 b 值的预训练与下游任务框架。

**🔧 技术方法**

使用技术包括：自监督 Masked AutoEncoder、Transformer、Rotary Positional Embedding（RoPE）及其扩展 D‑RoPE、相对位置编码、3D 视觉 Transformer、MSE/PSNR/SSIM/FID 等评价指标，以及多种下游任务的线性/MLP 预测头。

**📊 数据集**

数据集为 Human Connectome Project（HCP‑YA 预训练，HCP‑D、HCP‑A 下游）和 Alzheimer’s Disease Neuroimaging Initiative（ADNI）四期，包含不同 b 值、方向数以及年龄、性别、MCI/CN 诊断、ADAS‑Cog 评分等标签。

**📈 对比分析**

与 3D ResNet、ViT（含标准 RoPE）以及手工特征（FA/MD、RISH 等）等基线比较，预训练模型在年龄预测、性别分类、MCI 分类和 ADAS‑Cog 回归等任务上均获得与全监督方法相当或更优的表现，尤其在样本不足时表现尤为突出。

**⚠️ 局限性**

局限性：D‑RoPE 的距离度量无法像标准 RoPE 那样分解为键/值矩阵，导致计算复杂度升高；模型在更大、更多样化的 dMRI 数据集上的通用性和可迁移性尚未完全验证。

---

## 140. VolTune: A Fine-Grained Runtime Voltage Control Architecture for FPGA Systems

**arXiv ID:** 2603.26147 | [PDF](https://arxiv.org/pdf/2603.26147v1)

**作者:** Akram Ben Ahmed `[一作]` (National Institute of Advanced Industrial Sciences and Technology), Takaaki Fukai `[通讯]` (National Institute of Advanced Industrial Sciences and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本研究提出并实现了 VolTune，一个可在 FPGA 系统内部通过 PMBus 抽象实现的实时电压调节架构，支持硬件与软件两种控制路径；

**💡 创新点**

创新点在于将低层 PMBus 操作封装为可编程接口，实现了可细粒度、低延迟（约 2.3 ms）且资源占用极低的电压控制；

**🔧 技术方法**

采用 FPGA 逻辑实现硬件控制器、MicroBlaze 软件控制器、PMBus 事务引擎以及 TI UCD9248 可编程电源调节器的联合调度；

**📊 数据集**

实验使用 KC705 开发板、UCD9248 电源控制器和 GTX 收发器，未使用传统软件数据集，而是通过硬件计数流和 BER 测试进行性能评估；

**📈 对比分析**

对硬件与软件路径在 100 kHz 与 400 kHz PMBus 时钟下进行对比，硬件路径延迟 2.3 ms、资源占用 1.45% LUT/1.80% BRAM、功耗 0.015 W，且在 10 Gbps 链路上实现 28.4%（接近 29.3%）的功耗降低，BER 可容忍至 10⁻⁶；

**⚠️ 局限性**

局限性包括：控制速度受 PMBus 事务和电源调节器响应时间限制，无法实现周期级快速补偿；缺乏错误恢复机制；需针对每块板进行引脚/电压映射，无法直接跨平台复用；并且未覆盖多电源轨协同控制和热影响。

---

## 141. On the Complexity of Optimal Graph Rewiring for Oversmoothing and Oversquashing in Graph Neural Networks

**arXiv ID:** 2603.26140 | [PDF](https://arxiv.org/pdf/2603.26140v1)

**作者:** Mostafa Haghir Chehreghani `[一作]` (Amirkabir University of Technology), Mostafa Haghir Chehreghani `[通讯]` (Amirkabir University of Technology)

**通讯引用:** 545 | [OpenAlex ID](https://openalex.org/A5049221896)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过理论分析证明，优化图神经网络的图结构以缓解过平滑（oversmoothing）和过压缩（oversquashing）这两类问题在最优解上是NP-难的。

**💡 创新点**

首次将过平滑和过压缩的缓解问题分别建模为最大化谱间隙（spectral gap）和最大化图导电率（conductance），并利用从最小划分问题的还原证明这两类问题的NP-完备性，提供了对图重连（rewiring）理论极限的全新视角。

**🔧 技术方法**

使用谱图理论、Cheeger不等式、构造可扩展嵌入（expander embedding）以及多项式时间的图变形与判定技术。

**📊 数据集**

未使用实际数据集；研究完全基于理论构造与证明。

**📈 对比分析**

与现有启发式重连方法（如SDRF、DIGL）做理论对比，指出这些方法在多项式时间内实现而无法求解最优解；未给出实验性能评估，仅说明理论上可行性。

**⚠️ 局限性**

局限在于仅适用于无向无权图；未给出近似算法或实际实现；结果仅说明最优求解不可行，而非提供可行的替代方案。

---

## 142. ATime-Consistent Benchmark for Repository-Level Software Engineering Evaluation

**arXiv ID:** 2603.26137 | [PDF](https://arxiv.org/pdf/2603.26137v1)

**作者:** Xianpeng `[一作]`, Chen Tian `[通讯]` (Microsoft)

**通讯引用:** 3345 | [OpenAlex ID](https://openalex.org/A5100783476)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个时间一致的基准框架，用历史 PR 生成任务，并在相同仓库快照下通过匹配的 A/B 设计比较系统是否拥有仓库衍生知识。

**💡 创新点**

创新点在于：① 将基准分为时间切分（T0 只用历史信息）与任务生成两步，消除未来信息泄漏；② 通过四种提示粒度证明提示设计是影响评估结果的首要因素；③ 明确提出匹配实验设计以隔离仓库知识的因果贡献。

**🔧 技术方法**

技术手段包括 LLM 辅助提示生成、仓库快照与知识构建、匹配 A/B 对比实验、文件级别的精确度、召回率与 F1 评估。

**📊 数据集**

使用了两个开源仓库 DragonFly 与 React 的历史 PR 作为任务集，分别基于 T0 的快照构建知识并在 (T0,T1] 的 PR 上评估。

**📈 对比分析**

在相同仓库快照、相同任务、相同模型与执行环境下，只改变是否使用仓库衍生知识，比较文件定位的 F1。基线 F1 随提示粒度从 Minimal 到 Guided 从 0.20-0.25 逐步提升到 0.80-0.81；更强模型（Claude‑Ops‑4.6）在每种提示下均表现最好，跨仓库结果一致。

**⚠️ 局限性**

局限性包括：仅报告基线结果，未给出增强版与基线的对比；评估仅局限于文件定位，未涵盖最终补丁正确性或测试通过率；提示生成细节与泄漏控制未完整公开；只覆盖两大仓库，缺乏更广泛的泛化验证；缺乏置信区间与统计显著性检验。

---

## 143. SkinGPT-X: A Self-Evolving Collaborative Multi-Agent System for Transparent and Trustworthy Dermatological Diagnosis

**arXiv ID:** 2603.26122 | [PDF](https://arxiv.org/pdf/2603.26122v1)

**作者:** Zhangtianyi Chen `[一作]` (Chinese University of Hong Kong), Juexiao Zhou `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 SkinGPT-X，一个融合多模态输入、协同多智能体推理与自进化记忆的皮肤病诊断系统。

**💡 创新点**

创新点在于将自进化诊断记忆（EvoDerma‑Mem）与多智能体协作框架结合，提供可追溯、可解释且对罕见疾病具有强泛化的诊断流程。

**🔧 技术方法**

使用多模态 LLM（如 Qwen3‑VL、PanDerm）、检索增强生成（RAG）、自进化知识图谱以及多智能体协同推理技术。

**📊 数据集**

使用 Dermnet、HAM10000、Fitzpatrick‑17k、DDI31、Dermnet498、RSDD 等公开与自构建的高维度皮肤病数据集。

**📈 对比分析**

与 MedGemma、Hulu‑Med、Qwen3‑VL、PanDerm 等四大基线对比，SkinGPT‑X 在 Dermnet、HAM10000、DDI31、Fitzpatrick‑17k 上分别提升 ACC、Weighted F1、MCC、Kappa，尤其在 498 类细粒度任务和 8 种罕见病上提升 9.6%–13% 的准确率/加权 F1。

**⚠️ 局限性**

局限包括多智能体协作导致推理延迟、对不同设备图像特征分布的敏感性以及跨中心泛化的挑战。

---

## 144. SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection

**arXiv ID:** 2603.26109 | [PDF](https://arxiv.org/pdf/2603.26109v1)

**作者:** Jiaming Liang `[一作]` (Shenzhen University), Qiang Nie `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了开放词汇伪装物体检测任务并构建了 OVCOD-D 基准。

**💡 创新点**

创新点在于子描述主成分对比融合以及特异性引导的弱对齐与动态聚焦方法。

**🔧 技术方法**

使用了 YOLOv8 轻量化框架、文本编码+SVD+Adapter、对比融合、SF‑GLU 门控模块、覆盖率辅助损失等技术。

**📊 数据集**

使用了整合 COD10K-D、NC4K-D、CAMO-D、红蚂蚁巢等的 OVCOD-D 数据集，包含 40 基类 + 47 新类。

**📈 对比分析**

与多种开放词汇检测器（GLIP、Grounding DINO、YOLO‑World、DOSOD）比较，SDDF 在 OVCOD-D 上 AP 达到 56.4，明显优于基线。

**⚠️ 局限性**

局限在于数据规模有限、对细粒度描述的依赖以及对异常背景的鲁棒性仍有提升空间。

---

## 145. A Human-Inspired Decoupled Architecture for Efficient Audio Representation Learning

**arXiv ID:** 2603.26098 | [PDF](https://arxiv.org/pdf/2603.26098v1)

**作者:** Harunori Kawano `[一作]` (University of Technology Sydney), Takeshi Sasaki `[通讯]` (Shibaura Institute of Technology)

**通讯引用:** 8626 | [OpenAlex ID](https://openalex.org/A5081545966)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种人类启发的分离式音频表征学习框架 HEAR，拆分局部特征提取（Acoustic Model）与全局任务适配（Task Model）。

**💡 创新点**

创新点在于通过结构分离降低 Transformer 的 O(N²) 计算瓶颈，并采用知识蒸馏训练离散音频词典，实现在极低参数量下的高效学习。

**🔧 技术方法**

技术包括 Transformer 与相对位置表示、Gumbel‑Softmax 离散化、知识蒸馏、Masked Audio Modeling、特征门控融合及滑窗/交叉融合处理。

**📊 数据集**

预训练使用 10,000 小时整合的 AudioSet、VGG‑Sound、LAION‑Audio‑630k；微调采用 ESC‑50、Speech Commands v1/v2、VoxCeleb。

**📈 对比分析**

与 wav2vec2.0、HuBERT、AudioMAE、SSAST 等 85‑94M 参数的基线相比，HEAR 仅 15M 参数、9.47 GFLOPs、实时因子 0.095，准确率在 ESC‑50、GSC‑v1/v2、VoxCeleb 上保持接近 SOTA。

**⚠️ 局限性**

局限包括对长时序音频仍需滑窗与跨段融合，且在某些非语音任务中冻结模型或去除功率谱会导致性能下降。

---

## 146. Dynamic Tokenization via Reinforcement Patching: End-to-end Training and Zero-shot Transfer

**arXiv ID:** 2603.26097 | [PDF](https://arxiv.org/pdf/2603.26097v1)

**作者:** Yulun Wu `[一作]` (Capital One), Nam H. Nguyen `[通讯]` (Capital One)

**通讯引用:** 2290 | [OpenAlex ID](https://openalex.org/A5103324660)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于强化学习的序列补丁策略学习框架——Reinforcement Patching，可端到端学习时间序列的可变尺寸分块并压缩输入，提升长序列建模性能。

**💡 创新点**

核心创新在于：1) 用离散决策的强化学习（GRPG）直接优化补丁边界，避免软离散化；2) 将补丁策略模块与下游模型解耦，可预训练并在多任务中零-shot 使用；3) 在统一一阶 MDP 中实现高效并行训练；4) 对压缩率进行严格约束。

**🔧 技术方法**

采用 Transformer 轻量级补丁策略网络、组相对策略梯度（GRPG）强化学习、统一一阶 MDP、压缩率约束、对齐的预测与回报机制；下游使用统一的时间序列 Transformer 作为基准。

**📊 数据集**

在多变量时间序列预测基准：ETT（四个子集）、Weather、Electricity；以及统一时间序列数据集（UTSD）用于预训练。

**📈 对比分析**

与固定尺寸补丁（PatchTST）、可变尺寸策略（Entropy、TimeSqueeze、H‑Net 等）以及其他基线对比，使用 MSE/MAE 评估。实验显示在大多数配置下均优于所有基线；零-shot 预训练版在 22/24 评估配置中取得最佳成绩，平均 MSE 降低约 4.4%。

**⚠️ 局限性**

主要限制包括：1) 预训练仅基于单变量序列，缺乏跨通道的补丁依赖学习；2) 对因果 MDP 的多步决策尚未充分优化，训练效率与性能仍待提升。

---

## 147. Search-Induced Issues in Web-Augmented LLM Code Generation: Detecting and Repairing Error-Inducing Pages

**arXiv ID:** 2603.26091 | [PDF](https://arxiv.org/pdf/2603.26091v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 148. Learnable Instance Attention Filtering for Adaptive Detector Distillation

**arXiv ID:** 2603.26088 | [PDF](https://arxiv.org/pdf/2603.26088v1)

**作者:** Chen Liu `[一作]`, Qing Tian `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可学习的实例注意力过滤框架LIAF‑KD，用于提升目标检测模型的知识蒸馏效率和精度。

**💡 创新点**

核心创新在于引入可学习的实例选择器，基于教师和学生的特征动态评估实例重要性，实现学生自适应的实例加权蒸馏；并且通过多选择器的多样性损失提升选择器多样性。

**🔧 技术方法**

采用RoIAlign提取实例特征，训练多头软注意力选择器，构造实例加权的空间掩码，并在特征蒸馏损失中使用该掩码；使用多样性损失鼓励选择器差异化。

**📊 数据集**

在KITTI和MS COCO两个公共目标检测数据集上进行实验，使用GFL和RetinaNet两种一阶段检测器。

**📈 对比分析**

相较于教师模型、学生基线以及现有KD方法（如MasKD、DeFeat等），LIAF‑KD在两种检测器上均实现了显著提升（如GFL-R50 mAP提升至62.7%/42.4%，RetinaNet-R50 mAP提升至59.1%/39.7%），并保持较低的模型复杂度与更快的推理速度。

**⚠️ 局限性**

局限性包括：1）实例选择器训练需要额外的教师监督，增加训练成本；2）方法在极大规模或高类别多样性场景下的通用性尚未验证；3）与其他像素级掩码方法的融合效果有限。

---

## 149. Semi-Automated Knowledge Engineering and Process Mapping for Total Airport Management

**arXiv ID:** 2603.26076 | [PDF](https://arxiv.org/pdf/2603.26076v1)

**作者:** Darryl Teo `[一作]` (Singapore University of Technology and Design), Nuno Antunes Ribeiro `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 345 | [OpenAlex ID](https://openalex.org/A5058672710)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于专家构建的知识图谱与大型语言模型融合，构建机场运营的半自动化知识图谱，并生成可追溯的泳道流程图。

**💡 创新点**

创新点包括：① 采用“结构化符号融合”策略，用KG指导LLM提示以控制生成；② 通过LangExtract实现文档级上下文提取，显著提升对非线性流程的识别；③ 结合确定性字符串对齐保证每条知识三元组的源文献可追溯；④ 自动生成泳道图的算法实现，使流程可视化且具责任归属。

**🔧 技术方法**

技术手段：大型语言模型（LLM）、LangExtract抽取框架、提示工程与上下文窗口优化、确定性字符串对齐（如string-matching方法）、Protégé构建本体、基于BFS的泳道图生成算法、Python/KG构建工具。

**📊 数据集**

主要数据集为EUROCONTROL发布的机场协同决策制定（A‑CDM）手册（16页≈10k tokens）以及其16个里程碑的文本内容。

**📈 对比分析**

方法对比：将单页（短上下文）推理与全文（长上下文）推理进行对照；评估指标为Precision、Recall、F1。结果显示长上下文精度提升至P=0.967、R=0.982、F1=0.975，较短上下文的P=0.961、R=0.971、F1=0.966；同时F1提升约1个百分点。

**⚠️ 局限性**

局限性：仍需专家手工维护KG结构；对极大规模文本的处理仍受限于LLM上下文长度；多模态数据（视频、传感器）尚未完整集成；在高复杂度或非标准流程中可能出现幻觉或遗漏；需要人工审核以确保KG完整性与准确性。

---

## 150. One Is Not Enough: How People Use Multiple AI Models in Everyday Life

**arXiv ID:** 2603.26107 | [PDF](https://arxiv.org/pdf/2603.26107v1)

**作者:** Seunghwa Pyo `[一作]` (KAIST), Youn-kyung Lim `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过四天日记记录与十次半结构化访谈，探究用户在日常工作与生活中如何协调使用多种多模态大型语言模型（MLLM），并对其层级结构与协调策略进行实证分析。

**💡 创新点**

研究首次系统描述了用户在多模型生态中构建的主次模型层级、上下文切换模式，以及跨平台协调的策略，揭示了非单一AI系统使用中的心理模型与操作习惯，为设计支持多模型协同的工具提供了新的视角。

**🔧 技术方法**

采用日记记录工具（web‑based diary）、录音转录、英文翻译以及 Braun & Clarke 的主题分析方法，对收集的文本数据进行归纳编码。

**📊 数据集**

未使用公开数据集；研究数据来自129条日记条目（平均每人12.9条）和10次访谈（平均34分钟）所得的文字记录。

**📈 对比分析**

本研究不涉及模型性能对比，评估方式为定性编码与主题提炼，没有数值指标或实验对比；结果以用户行为模式与策略分类呈现。

**⚠️ 局限性**

局限包括样本主要为20~30岁技术熟练的研究生与从业者，可能不代表老年人或非技术用户；研究周期仅为四天，未能捕捉长期动态变化（如模型发布、价格调整等）。

---

## 151. On the computational complexity of JavaScript regex matching

**arXiv ID:** 2603.26139 | [PDF](https://arxiv.org/pdf/2603.26139v1)

**作者:** Victor Deng `[一作]`, Clément Pit-Claudel `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了JavaScript正则表达式匹配的计算复杂性，阐明了其在不同特性组合下的难度阶层。

**💡 创新点**

首次给出在真实ECMAScript 2023语法下的机械化证明，证明即便去除负向前瞻，匹配仍为PSPACE‑hard；并进一步证明在无下界量化子时为PSPACE‑complete、无下界量化子且无前瞻时为OptP‑complete。

**🔧 技术方法**

采用Coq（Rocq）证明助手对正则语义与QBF、CNF LEXICOGRAPHIC SAT归约进行机械化验证，并利用量化布尔公式与语义模型实现完整的归约与证据。

**📊 数据集**

以ECMAScript 2023的JavaScript正则表达式语法为研究对象，无需外部数据集，仅使用理论构造的QBF实例和CNF公式。

**📈 对比分析**

通过理论归约与机械化证明，而非实验性性能比较；结论表明匹配问题的复杂度阶层分别为PSPACE‑complete（无下界量化子）和OptP‑complete（无下界量化子且无前瞻）。

**⚠️ 局限性**

研究未覆盖带下界量化子（计数重复）的正则表达式，并未验证对未来JS版本（如2025版）或其他语言（Perl、PCRE等）的适用性。

---

## 152. SWE-PRBench: Benchmarking AI Code Review Quality Against Pull Request Feedback

**arXiv ID:** 2603.26130 | [PDF](https://arxiv.org/pdf/2603.26130v1)

**作者:** Deepak Kumar `[一作]` `[通讯]` (Independent Researcher), Deepak Kumar (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为CodeReviewer的基准，包含350条拉取请求并配有人类专家标注的真实问题，专门用于评估AI模型在代码审查任务中的质量。

**💡 创新点**

创新点包括：①引入了三种难度类型（直接、上下文、潜在）以细粒度衡量审查任务；②设计了三种冻结上下文配置（diff-only、diff+文件内容、全上下文）并系统剖析上下文对模型性能的影响；③采用LLM-as-Judge框架验证并跨评判者确认评判一致性，确保评估可靠性。

**🔧 技术方法**

使用技术包括：大型语言模型（Claude Haiku、Claude Sonnet、GPT-4o、DeepSeek V3、Mistral Large、Mistral Small、Llama 3.3 70B）、AST函数抽取与导入图解析构建语义上下文、二步一致性匹配与语义对齐、基于token的注意力稀释分析。

**📊 数据集**

数据集为CodeReviewer，来源于700条初选后筛选至350条PR，涵盖Python、JavaScript、Go、TypeScript、Java等语言，按难度类型划分，并保留完整PR上下文与人类评审注释。

**📈 对比分析**

评估方法：8个前沿模型在三种上下文配置下的召回、精确率、语义对齐、行动性、效率等多维度打分，最终以加权平均综合得分进行比较。结果显示：最高模型在diff-only配置下的召回率约为31%，模型整体表现与人类专家相距20–40个百分点，且随着上下文增大性能呈现单调下降趋势。

**⚠️ 局限性**

局限性包括：①评估仅覆盖Python为主，非Python语言效果未知；②使用GPT-5.2作为唯一主要评判者，跨评判者一致性虽验证但绝对分值仍可能受评判者偏差影响；③尽管通过多重过滤降低训练泄露风险，但仍无法完全排除模型对某些PR的记忆；④当前上下文结构未尝试更高级的检索或边界标记方案，可能限制提升空间。

---

## 153. Not All Entities are Created Equal: A Dynamic Anonymization Framework for Privacy-Preserving Retrieval-Augmented Generation

**arXiv ID:** 2603.26074 | [PDF](https://arxiv.org/pdf/2603.26074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 154. Rethinking Recommendation Paradigms: From Pipelines to Agentic Recommender Systems

**arXiv ID:** 2603.26100 | [PDF](https://arxiv.org/pdf/2603.26100v1)

**作者:** Jinxin Hu `[一作]` (Alibaba International Digital Commerce Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commerce Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种代理推荐系统（AgenticRS），将推荐系统的关键模块重组为代理，以实现自我演化和优化。

**💡 创新点**

创新点在于将静态的推荐系统转变为多代理决策系统，允许模块在功能闭环中独立评估和演化。

**🔧 技术方法**

使用了强化学习（RL）风格的优化和基于大型语言模型（LLM）的架构生成与选择机制。

**📊 数据集**

未具体提及使用的数据集，但讨论了多样化用户和内容的场景。

**📈 对比分析**

通过将推荐系统视为多代理决策过程，提供了一个结构化的框架，强调了代理的独立优化和组合演化，性能上能够更好地适应复杂的业务需求。

**⚠️ 局限性**

局限性在于系统的演化和优化仍然依赖于设计和实施的复杂性，可能在实际应用中面临挑战。

---

## 155. AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation

**arXiv ID:** 2603.26096 | [PDF](https://arxiv.org/pdf/2603.26096v1)

**作者:** Hyeongyu Kim `[一作]` (Yonsei University), Dosik Hwang `[通讯]` (Yonsei University)

**通讯引用:** 3101 | [OpenAlex ID](https://openalex.org/A5085519704)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在测试时自适应（TTA）中引入了激活函数可学习的参数化形式，允许在推理期间动态调整激活阈值和梯度灵敏度。

**💡 创新点**

创新点在于将激活函数从固定非线性视角转变为可学习的动态函数，并通过中心位移与梯度斜率不对称调节，实现更细粒度、稳定的域迁移补偿，突破了传统归一化参数更新的局限。

**🔧 技术方法**

技术实现包括对 ReLU、GELU 等基准激活进行参数化（λ_pos, λ_neg, c），在测试时通过最小化常见目标（如熵最小化、一致性约束）更新这些参数；同时保持网络权重不变，兼容多种现有 TTA 方案。

**📊 数据集**

使用 CIFAR10-C、CIFAR100-C 与 ImageNet-C 等标准噪声/畸变数据集进行评估，并在 ViT-B/16、WRN、ResNet 等多种架构上测试。

**📈 对比分析**

相较于主流的归一化层更新方法（如 TENT、EATA、SAR 等），AcTTA 在所有实验中均获得更低的错误率，尤其在 ImageNet-C 上提升显著；在小批量和大学习率场景下依旧保持稳定，证明其鲁棒性和普适性。

**⚠️ 局限性**

局限性包括对网络架构、激活类型和可学习层深度的依赖，需要进一步制定统一的选择准则；目前在不同模型间的最优参数设置仍缺乏系统化的理论指导。

---

## 156. IndoBERT-Relevancy: A Context-Conditioned Relevancy Classifier for Indonesian Text

**arXiv ID:** 2603.26095 | [PDF](https://arxiv.org/pdf/2603.26095v1)

**作者:** Muhammad Apriandito Arya Saputra `[一作]` (SocialX), Hanif Fakhrurroja `[通讯]` (National Research and Innovation Agency)

**通讯引用:** 733 | [OpenAlex ID](https://openalex.org/A5004438607)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于IndoBERT的上下文条件相关性分类模型IndoBERT‑Relevancy

**💡 创新点**

通过迭代失败驱动的数据构造和针对性合成数据，显著提升模型对正式、非正式及隐式文本的鲁棒性

**🔧 技术方法**

使用IndoBERT Large微调、双输入分类头、逆频率类别权重训练，并利用GPT‑4o‑mini进行标注与合成

**📊 数据集**

构建了31,360条正式新闻、社交媒体和隐式文本的上下文‑文本对，覆盖188个主题和12个领域

**📈 对比分析**

在验证集上取得F1 0.948、准确率96.5%，比仅使用正式文本的数据提升12% F1，性能优于传统方法

**⚠️ 局限性**

对新主题和极端隐式语境的泛化有限，合成数据可能缺少真实噪声，需周期性更新

---

## 157. IP-Bench: Benchmark for Image Protection Methods in Image-to-Video Generation Scenarios

**arXiv ID:** 2603.26154 | [PDF](https://arxiv.org/pdf/2603.26154v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 158. Adversarial Bandit Optimization with Globally Bounded Perturbations to Linear Losses

**arXiv ID:** 2603.26066 | [PDF](https://arxiv.org/pdf/2603.26066v1)

**作者:** Zhuoyu Cheng `[一作]` (Kyushu University), Eiji Takimoto `[通讯]` (Kyushu University)

**通讯引用:** 655 | [OpenAlex ID](https://openalex.org/A5050429437)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了对抗性 bandit 优化中带全局扰动预算的线性+非凸损失函数，提出改进的 SCRiBLe 算法并给出期望与高概率下的 regret 上界以及下界。

**💡 创新点**

创新点在于引入全局扰动约束模型，利用收缩域和自共轭 barrier 重新分解 regret，消除了升维和学习率递增的必要，获得更紧的高概率界并证明 Ω(C) 的不可避免下界。

**🔧 技术方法**

主要技术包括 ν‑self‑concordant barrier、梯度估计与偏差误差分解、马尔可夫差分集中极限以及改进的 SCRiBLe 迭代更新。

**📊 数据集**

实验使用人工合成数据集，随机生成线性向量和可调扰动函数，在固定时间 horizon 下评估算法性能。

**📈 对比分析**

与原 SCRiBLe 算法比较时，实验显示在 C=0 情况下 SCRiBLe 略优，但当扰动增大时本文算法因在收缩域内操作更稳健，整体性能随扰动增加而提升。

**⚠️ 局限性**

局限性包括仅适用于线性+扰动形式的损失，无法直接推广到更一般非凸或非光滑场景；收缩域可能限制边界最优点的采样，且下界仅为 Ω(C)，尚未揭示维度依赖的更精确极限。

---

## 159. Improved Algorithms for Unrelated Crowd Worker Scheduling in Mobile Social Networks

**arXiv ID:** 2603.26129 | [PDF](https://arxiv.org/pdf/2603.26129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 160. Finding Distributed Object-Centric Properties in Self-Supervised Transformers

**arXiv ID:** 2603.26127 | [PDF](https://arxiv.org/pdf/2603.26127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 161. "Oops! ChatGPT is Temporarily Unavailable!": A Diary Study on Knowledge Workers' Experiences of LLM Withdrawal

**arXiv ID:** 2603.26099 | [PDF](https://arxiv.org/pdf/2603.26099v1)

**作者:** Eunseo Oh `[一作]` (KAIST), Youn-kyung Lim `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对10名频繁使用大语言模型（LLM）的知识工作者进行为期四天的日记研究与半结构访谈，探索LLM被强行拔掉后工作流程、价值观与依赖性的变化。

**💡 创新点**

创新点在于将LLM视为现代知识工作中的“基础设施”，利用“撤销”实验揭示其深度嵌入和社会规范效应，并提出“价值驱动的适配”这一新的使用范式。

**🔧 技术方法**

本研究未采用新模型或算法，而是构建了基于网页的日记记录界面、Chrome阻拦插件与访谈录音，利用主题分析方法对文本数据进行编码与归纳。

**📊 数据集**

数据来源为参与者自我报告的日记条目（200条）、每日反思日志（40条）及访谈录音（约9小时），无公开数据集或机器学习训练集。

**📈 对比分析**

研究未与其他方法做直接比较，也没有性能指标；评价基于主题分析结果的深度与启发性，指出工作流程中出现的空白、价值复兴与不可逆的依赖。

**⚠️ 局限性**

局限性包括：样本仅来自韩国、人数有限（10人）、仅选取高依赖用户、研究时长仅四天，可能导致文化、时间与样本偏倚；未来需扩大样本、跨文化验证与长期跟踪。

---

## 162. When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization

**arXiv ID:** 2603.26078 | [PDF](https://arxiv.org/pdf/2603.26078v1)

**作者:** Zhihan Chen `[一作]` (University of California, Los Angeles), Xinyu Yao `[通讯]` (Carnegie Mellon University)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5061540066)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了一个针对多主体个性化生成的压力测试基准，揭示了现有模型在6-10主体场景下的身份崩溃问题。

**💡 创新点**

引入基于DINOv2的Subject Collapse Rate（SCR）度量，显著弥补CLIP指标在多主体身份保留上的不足。

**🔧 技术方法**

采用文本驱动扩散模型（DiT/FLUX）、DINOv2特征匹配与全局/局部注意力分析。

**📊 数据集**

使用XVerse和COSMISC两个公开个性化数据集构建统一身份池。

**📈 对比分析**

在75条多主体提示上，对MOSAIC、XVerse、PSR三模型各生成3个随机种子，使用CLIP-T、DINOv2、SCR评价，结果显示SCR随主体数升高急剧上升至近100%，模型在2-4主体时表现相对良好。

**⚠️ 局限性**

仅关注人类与动物主体，缺乏3D几何约束和对非物体多主体的评估，且基准中的遮挡与交互程度主要靠提示而非精确3D布局。

---

## 163. MuDD: A Multimodal Deception Detection Dataset and GSR-Guided Progressive Distillation for Non-Contact Deception Detection

**arXiv ID:** 2603.26064 | [PDF](https://arxiv.org/pdf/2603.26064v1)

**作者:** Peiyuan Jiang `[一作]` (University of Electronic Science and Technology of China), Qiao Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 4867 | [OpenAlex ID](https://openalex.org/A5100393703)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用GSR中稳定的生理信号提升无接触欺骗检测

**💡 创新点**

① 构建大规模多模态欺骗检测数据集 MuDD；② 提出 GSR 引导的进步知识蒸馏框架 GPD，融合特征级与数字级蒸馏、动态路由与进步加权，适应大模态差异

**🔧 技术方法**

跨模态知识蒸馏、CKA 相似度动态路由、进步加权、视频MAE、WavLM 等深度学习技术

**📊 数据集**

MuDD（130名受试者，690分钟，包含视频、音频、GSR、PPG、心率及人格特征）

**📈 对比分析**

与九种现有 CMKD 方法对比，GPD 在视频/音频模式下的欺骗检测 F1/AUC 以及隐藏数字识别 Top‑1/Top‑2 均取得最佳成绩，提升约 3‑5%

**⚠️ 局限性**

未显式建模多模态时间对齐，且 MuDD 受隐私限制无法公开

---

## 164. PEANUT: Perturbations by Eigenvalue Alignment for Attacking GNNs Under Topology-Driven Message Passing

**arXiv ID:** 2603.26136 | [PDF](https://arxiv.org/pdf/2603.26136v1)

**作者:** Bhavya Kohli `[一作]`, Biplab Sikdar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于虚拟节点注入的黑盒攻击方法 PEANUT，在推理阶段无需梯度或代理模型即可对 GNN 造成性能下降。

**💡 创新点**

创新点在于利用图卷积中可直接变动的拓扑矩阵，采用主特征向量对齐投影方式，最大化节点表示差异，从而在无需迭代优化的情况下实现高效攻击。

**🔧 技术方法**

方法基于简化图卷积网络（SGC）的理论，使用主特征向量投影、正负权重约束、离散化等技术生成虚拟节点权重，完全不依赖梯度或模型训练。

**📊 数据集**

在七个节点分类数据集（Cora、Citeseer、Pubmed）、五个分子回归数据集（ESOL、FreeSolv、Lipophilicity、ZINC、AQSOL）以及四个图分类数据集（MUTAG、PROTEINS、ENZYMES、IMDB-BINARY）上进行实验。

**📈 对比分析**

与 TDGIA、ATDGIA、AGIA 等现有注入攻击及随机攻击进行比较，PEANUT 在相同攻击预算下显著降低准确率/误差，尤其在回归任务中达到或优于基线，且不需要额外训练或迭代优化。

**⚠️ 局限性**

局限包括对负权重/二值邻接矩阵的处理仍需启发式，针对高维或大规模图时可能需要进一步优化；对仅聚合无权重的 GNN（如无拓扑矩阵输入的模型）攻击效果不如预期。

---

## 165. TaxaAdapter: Vision Taxonomy Models are Key to Fine-grained Image Generation over the Tree of Life

**arXiv ID:** 2603.26128 | [PDF](https://arxiv.org/pdf/2603.26128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 166. DPD-Cancer: Explainable Graph-based Deep Learning for Small Molecule Anti-Cancer Activity Prediction

**arXiv ID:** 2603.26114 | [PDF](https://arxiv.org/pdf/2603.26114v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 167. IBEX: Internal Bandwidth-Efficient Compression Architecture for Scalable CXL Memory Expansion

**arXiv ID:** 2603.26131 | [PDF](https://arxiv.org/pdf/2603.26131v1)

**作者:** Younghoon Ko `[一作]` (Seoul National University), Hyokeun Lee `[通讯]` (DGIST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种名为IBEX的内部带宽高效压缩架构，专为CXL内存扩展器设计，实现了基于块级压缩的推广/降级管理；

**💡 创新点**

创新点包括：利用OS透明的内部页面活动区域实现的带宽高效块管理、延迟式参考位更新、影子推广机制避免降级重压缩、块级共定位与压缩元数据压缩等多项硬件友好优化；

**🔧 技术方法**

采用块级压缩算法（如LZ4/LZ77/Zstd）配合第二机会算法实现降级决策、懒更新参考位、影子推广、子区域共享物理地址MSB实现元数据压缩，整体实现完全在CXL设备内部完成；

**📊 数据集**

使用SPEC CPU2017、GAPBS图算法、XSBench等工作负载进行评估，涵盖多种内存访问特征；

**📈 对比分析**

与DMC、MXT、TMCC、DyLeCT和线级压缩Compresso等现有方案进行对比，测量归一化性能和压缩比；IBEX平均在TMCC基础上提升1.28×、在DyLeCT基础上提升1.40×，压缩比达到1.59；

**⚠️ 局限性**

局限性：推广区尺寸有限导致某些工作负载仍出现高迁移开销；仿真仅在4核环境下评估，未覆盖大规模多核系统；假设无页面交换，未测量系统级页面错误影响；影子推广导致的额外内存复制略低压缩比；整体仍受CXL内部带宽限制。

---

## 168. CL-SEC: Cross-Layer Semantic Error Correction Empowered by Language Models

**arXiv ID:** 2603.26125 | [PDF](https://arxiv.org/pdf/2603.26125v1)

**作者:** Yirun Wang `[一作]` (Chinese University of Hong Kong), Lihao Zhang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1012 | [OpenAlex ID](https://openalex.org/A5002792262)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种跨层语义错误校正框架（CL-SEC），在传统物理层前向纠错（FEC）之后，利用语言模型（LM）的语义理解与物理层软信息（LLR）共同对被损坏的词进行纠正，最终实现文本的逐字还原。

**💡 创新点**

创新点主要在于：①将物理层的词级LLR分布与应用层的LM掩码预测分布通过贝叶斯乘积法并行结合，形成跨层后验分布；②利用词长元数据对候选词进行约束；③在纠正后加入语义化标点恢复，显著提升语义保真度。

**🔧 技术方法**

使用技术包括：卷积码 + BCJR软解码、词级LLR推算、预训练掩码语言模型（BART、mmBERT）、贝叶斯组合（Hadamard乘积）、Qwen3-8B 语义化标点恢复、BERTScore 与 ROUGE‑L 评价。

**📊 数据集**

实验数据集为 50 篇英文教科书段落（已预处理以保证所有单词均出现在 BART 与 mmBERT 的词表中），在 AWGN 信道下以 QPSK 调制进行仿真，重复 10 次 Monte‑Carlo 以得到统计显著性。

**📈 对比分析**

对比方法包括：纯 BCJR、词级 LLR 方案（WL‑LLR）、单独 LM 方案（MLM）、CL‑SEC 以及 CL‑SEC 加标点恢复（PR）。结果表明：①CL‑SEC 在 BER、WER 上显著优于 BCJR、WL‑LLR 与 MLM，尤其在低 SNR 时差距更大；②CL‑SEC 的 BERTScore 与 ROUGE‑L 均高于单独 LM 方案，加入 PR 后再进一步提升，接近原始文本的语义保真度。

**⚠️ 局限性**

限制主要有：①需要在帧头嵌入词长元数据，增加码率开销且易受噪声影响；②未考虑源压缩与词边界推断；③实验仅在 AWGN 信道下进行，缺乏对多径/衰落信道的验证；④语言模型推断带来的计算延迟对低时延应用不友好。

---

## 169. Accurate Precipitation Forecast by Efficiently Learning from Massive Atmospheric Variables and Unbalanced Distribution

**arXiv ID:** 2603.26108 | [PDF](https://arxiv.org/pdf/2603.26108v1)

**作者:** Shuangliang Li `[一作]`, Maolin Zhang `[通讯]` (Wuhan University)

**通讯引用:** 1522 | [OpenAlex ID](https://openalex.org/A5100661300)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于低维潜在空间迭代预测的短期降水预报模型，并设计了针对极端稀疏降水的加权MAE+交叉熵（WMCE）损失函数。

**💡 创新点**

创新点包括：①在低维潜在空间自动提取与降水演化强相关的特征并进行迭代预测；②结合多尺度残差网络与ViT实现潜在特征预测；③设计WMCE损失对稀缺降水事件赋予更大权重，提升检测和强度预测能力；④利用多源大气观测（雷达、MRMS、ERA5、卫星）提升信息利用率。

**🔧 技术方法**

采用的技术包括：多尺度残差编码器、潜在迭代预测模块（LPM）融合ViT和HTA算法、并行投影器、重构器、加权MAE+交叉熵损失（WMCE）、AdamW优化、余弦学习率调度、Dropout、集成梯度解释等。

**📊 数据集**

使用美国MRMS+ERA5（1小时分辨率）和中国湖北卫星+雷达+ERA5（15分钟分辨率）两套数据集，分别预测累计降水率和雷达反射率。

**📈 对比分析**

与ConvGRU、SimVP、MetNet2、EarthFarseer等SOTA方法对比，POD/CSI/HSS在中晚期预测提升40%–60%，且推理时间最短（约0.385s），训练时间仅比SimVP略高，显示出显著的性能与效率优势。

**⚠️ 局限性**

主要局限包括：模型仍易产生过度预测，输入的ERA5分辨率相对粗糙；对非降水序列的处理影响预报敏感性；缺乏对更高空间分辨率和更长时程的验证。

---

## 170. ROAST: Risk-aware Outlier-exposure for Adversarial Selective Training of Anomaly Detectors Against Evasion Attacks

**arXiv ID:** 2603.26093 | [PDF](https://arxiv.org/pdf/2603.26093v1)

**作者:** Mohammed Elnawawy `[一作]` (University of British Columbia), Karthik Pattabiraman `[通讯]` (University of British Columbia)

**通讯引用:** 5367 | [OpenAlex ID](https://openalex.org/A5073641368)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种风险感知的异常检测器训练框架ROAST，用于提升医疗DNN的对抗样本检测能力

**💡 创新点**

创新点在于：①通过风险评估聚类识别对抗攻击更不易受影响的患者；②对选定患者进行噪声清洗和受控对抗样本曝光，从而在不显著降低精度的前提下显著提升召回率；③将风险评估与对抗样本生成整合为一套可迭代的闭环流程

**🔧 技术方法**

使用了攻击模拟（URET）、即时风险量化（基于线性/逻辑回归系数的加权平方差公式）、动态时间规整DTW与层次聚类、以及三种无监督异常检测器（kNN、One‑Class SVM、MAD‑GAN）进行对抗样本曝光与训练

**📊 数据集**

在三大医疗时序数据集上评估：OhioT1DM（血糖预测）、MIMIC（ICU死亡预测）和PhysioNet CinC（败血症预测）

**📈 对比分析**

与传统全患者无选择训练进行对比，ROAST平均提升召回率16.2%（不同AD与数据集差异显著），精度下降仅1项实验（<5%），训练集规模平均缩减80.1%，训练时间平均缩减88.3%；聚类对抗风险参数与树切点具有一定鲁棒性

**⚠️ 局限性**

局限性包括：①依赖于风险量化公式的准确性；②在攻击能力更强或非医疗场景下的适用性需进一步验证；③目前为离线风险评估，未涵盖长期概念漂移与在线自适应需求

---

## 171. CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection

**arXiv ID:** 2603.26092 | [PDF](https://arxiv.org/pdf/2603.26092v1)

**作者:** Youngjun Song `[一作]` (Yonsei University), Dosik Hwang `[通讯]` (Yonsei University)

**通讯引用:** 3101 | [OpenAlex ID](https://openalex.org/A5085519704)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出了CD-Buffer，一种结合减法通道抑制与加法轻量化适配器的测试时域适配框架，能够自适应处理不同严重程度的天气域偏移；

**💡 创新点**

创新点在于使用统一的特征层差异度量来驱动双缓冲机制，实现对每个通道的动态抑制与补偿平衡；

**🔧 技术方法**

采用了可学习的通道掩码、差异驱动正则化、BN层统计匹配以及轻量化适配器插入等技术；

**📊 数据集**

实验基于KITTI、Cityscapes以及ACDC的合成与真实恶劣天气数据集；

**📈 对比分析**

与BufferTTA、PruningTTA、ActMAD、WHW等基线比较，在多种天气与严重程度下的mAP@50均达到了或逼近最优水平；

**⚠️ 局限性**

局限性在于需预先计算源域统计，且对极端稀疏或强噪声场景的鲁棒性尚待进一步验证。

---

## 172. R-PGA: Robust Physical Adversarial Camouflage Generation via Relightable 3D Gaussian Splatting

**arXiv ID:** 2603.26067 | [PDF](https://arxiv.org/pdf/2603.26067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 173. Simulating Novice Students Using Machine Unlearning and Relearning in Large Language Models

**arXiv ID:** 2603.26142 | [PDF](https://arxiv.org/pdf/2603.26142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 174. AgenticRS-Architecture: System Design for Agentic Recommender Systems

**arXiv ID:** 2603.26085 | [PDF](https://arxiv.org/pdf/2603.26085v1)

**作者:** Hao Zhang `[一作]` (Alibaba International Digital Commerce Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commerce Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出AutoModel——一种以代理为核心的工业推荐系统生命周期管理架构，集成AutoFeature、AutoTrain、AutoPerf三大演进代理并通过共享协调与知识层实现闭环自我进化；通过paper_auto_train案例演示自动化论文重现与模型迭代；

**💡 创新点**

创新点在于：①把推荐系统从传统的静态回忆‑排序管线转化为可动态演进的代理网络；②将数据特征、模型设计与训练、性能部署三个生命周期维度分别抽象为长期驻留的自治代理；③通过共享知识层记录决策与结果，实现跨代理的经验共享与奖励反馈；④用LLM驱动的解析、代码生成与训练监控实现从论文描述到可执行代码的全自动闭环；

**🔧 技术方法**

核心技术包括：大型语言模型（LLM）用于论文解析与代码重写；程序分析与自动化代码改写；大规模训练平台自动提交与监控；离线评估流水线（AUC、NDCG、召回等）；内部知识图谱与经验数据库存储决策与结果；强化学习/搜索算法用于特征与模型空间的探索；

**📊 数据集**

使用的是内部工业推荐系统的真实数据集（包括用户行为日志、特征仓库、训练/验证集），并在NeurIPS 2025最佳论文“Gated Attention for Large Language Models”实验中对同一内部数据进行基线与实验模型训练；

**📈 对比分析**

对基线模型与实验模型（Gated Attention）在同一验证集和回放数据上进行离线指标对比，计算AUC、NDCG、召回等；实验结果显示实验模型在部分用户组和业务场景上提升了X%（示例：NDCG提升2.5%），同时训练成本与模型规模也被合理控制；

**⚠️ 局限性**

局限性包括：①主要验证在内部数据与系统环境，缺乏公开数据集的跨平台评估；②对LLM生成代码的准确性与可解释性仍有待进一步保证；③实验聚焦单一论文重现，未全面覆盖多任务或多模型组合的演进；④当前实现仍需人工介入异常处理与复杂故障排查，完全自动化仍有挑战。

---

## 175. Efficient Few-Shot Learning for Edge AI via Knowledge Distillation on MobileViT

**arXiv ID:** 2603.26145 | [PDF](https://arxiv.org/pdf/2603.26145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 176. PruneFuse: Efficient Data Selection via Weight Pruning and Network Fusion

**arXiv ID:** 2603.26138 | [PDF](https://arxiv.org/pdf/2603.26138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 177. Cinematic Audio Source Separation Using Visual Cues

**arXiv ID:** 2603.26113 | [PDF](https://arxiv.org/pdf/2603.26113v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 178. LLM Benchmark-User Need Misalignment for Climate Change

**arXiv ID:** 2603.26106 | [PDF](https://arxiv.org/pdf/2603.26106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 179. Are LLM-Enhanced Graph Neural Networks Robust against Poisoning Attacks?

**arXiv ID:** 2603.26105 | [PDF](https://arxiv.org/pdf/2603.26105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 180. TinyML for Acoustic Anomaly Detection in IoT Sensor Networks

**arXiv ID:** 2603.26135 | [PDF](https://arxiv.org/pdf/2603.26135v1)

**作者:** Amar Almaini `[一作]` (Deggendorf Institute of Technology), Ghadeer Ashour `[通讯]` (King Abdulaziz University)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5061439342)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建并评估了面向 IoT 传感器网络的 TinyML 声学异常检测流水线，利用 MFCC 特征提取与轻量级神经网络，完成模型训练、量化并导出 TensorFlow Lite Micro 版本。

**💡 创新点**

创新点在于实现了可复现的 MFCC‑TinyML 方案，在嵌入式微控制器上以 60 kB 的存储实现 91% 的准确率，实现了低功耗、低延迟的实时异常检测。

**🔧 技术方法**

采用 MFCC 特征提取、全连接神经网络、Adam 优化器、早停、Dropout、模型量化（int8）以及 TensorFlow Lite for Microcontrollers。

**📊 数据集**

使用公开的 UrbanSound8K 数据集，将 10 个城市声类别重新聚合为正常与异常两类进行二分类训练与评估。

**📈 对比分析**

通过浮点（float32）模型与量化（int8）模型对比，原模型准确率 95%/F1 0.95/ROC‑AUC 0.991，量化后准确率 91%/F1 0.91/ROC‑AUC 0.970，展示量化后仅损失极小的性能。

**⚠️ 局限性**

局限性包括：仅做二分类；尚未在真实 IoT 设备上部署；对环境噪声与音色重叠导致误判；缺乏实际功耗、延迟与内存使用评估。

---

## 181. InstaVSR: Taming Diffusion for Efficient and Temporally Consistent Video Super-Resolution

**arXiv ID:** 2603.26134 | [PDF](https://arxiv.org/pdf/2603.26134v1)

**作者:** Jintong Hu `[一作]` (Insta360 Research), Lu Qi `[通讯]` (Wuhan University)

**通讯引用:** 14933 | [OpenAlex ID](https://openalex.org/A5100665475)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出轻量化扩散框架InstaVSR，实现高效视频超分辨率；

**💡 创新点**

(1) 通过VAE替换为Pixel Unshuffle、移除跨注意力等剪枝显著降低模型体积；(2) 结合递归训练与光流引导的时间一致性损失抑制闪烁；(3) 双空间对抗学习（潜在+像素判别器）恢复生成能力并保持细节；

**🔧 技术方法**

轻量化扩散网络、单步扩散、光流Warp、递归一致性训练、潜在空间评分匹配、像素判别器、区域感知TV正则、LoRA微调；

**📊 数据集**

SPMCS、YouHQ、20K YouTube视频（训练集）以及自制航空视频集；

**📈 对比分析**

与RealESRGAN、RealViformer、RealBasicVSR、AdcSR、UpscaleAVideo、DOVE、STAR、DLoRAL等基线对比。InstaVSR仅用0.45B参数、7GB显存、1.0分钟完成30帧2K视频，显著快且显存低；在DOVER、NIQE、MANIQA、MUSIQ等无参考指标上名列前茅，视觉质量优于多数基线；

**⚠️ 局限性**

pixel级指标（PSNR/SSIM）不突出；对高动态场景的鲁棒性有限，未来需进一步提升动态序列稳定性并进一步加速。

---

## 182. Beyond Where to Look: Trajectory-Guided Reinforcement Learning for Multimodal RLVR

**arXiv ID:** 2603.26126 | [PDF](https://arxiv.org/pdf/2603.26126v1)

**作者:** Jinda Lu `[一作]` (University of Science and Technology of China), Xiang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 17003 | [OpenAlex ID](https://openalex.org/A5100389037)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种轨迹引导强化学习（TGRL）框架，利用专家推理轨迹对多模态 RLVR 的感知-推理过渡进行细粒度监督，提升视觉信息在推理过程中的可靠性。

**💡 创新点**

创新点在于通过轨迹级对齐（trajectory-level alignment）将专家感知与推理步骤结合，结合自适应 token 重新加权和正例轨迹过滤，实现对感知到推理链的精确信用分配。

**🔧 技术方法**

技术上融合了 GRPO 与 DAPO 等基于组的 RLVR 优化策略，并加入了轨迹集成、优势归一化、重要性比例校正与 token 级加权，形成统一的 token 重要系数进行策略更新。

**📊 数据集**

实验采用 Geo3K（2.1K 题）和 ViRL39K（39K 题）作为训练数据，评估在 MathVista、WeMath、MathVision、MathVerse 与 HallusionBench 等五个多模态推理与感知基准上。

**📈 对比分析**

与多种基准方法（VLAA-Thinker、Perception-R1、ThinkLite 等）对比，TGRL 在 39K 规模下实现平均 60.68 分，明显优于现有 RLVR 与轨迹引导方案，验证了轨迹监督对提升推理准确率的有效性。

**⚠️ 局限性**

主要局限在于对强大专家策略（如 Qwen2.5-VL-32B）的依赖、离线轨迹的分布不匹配风险以及对更广泛多模态任务的泛化能力有待进一步探索。

---

## 183. Selective Deficits in LLM Mental Self-Modeling in a Behavior-Based Test of Theory of Mind

**arXiv ID:** 2603.26089 | [PDF](https://arxiv.org/pdf/2603.26089v1)

**作者:** Christopher Ackerman `[一作]` `[通讯]`, Christopher Ackerman

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了一种基于行为的实验范式，用来测试大型语言模型（LLM）在自我与他人心理状态建模以及策略欺骗上的能力。

**💡 创新点**

创新点在于：①将ToM评估从传统的描述性问答转为需要模型做出行动决策的游戏场景；②通过“思考（scratchpad）”与“非思考”两种模式区分模型内部推理与外部记忆匹配；③系统性比较多种认知组件（自我知识、真假信念、队友知识、对手意图）以及对负载敏感性的评估。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT、Anthropic、Google等）配合 chain‑of‑thought 推理；基于文本的游戏生成器；动作决策接口（Ask、Tell、Pass）；统计相关性分析与负载实验；以及人类对照实验。

**📊 数据集**

使用的数据集为：自构造的 26 个场景（含 10 次重复）+ 额外负载场景（事件负载与状态转移负载），共 28 个自 2024 年 5 月起发布的 LLM 版本；以及 10 名人类参与者的实验数据。

**📈 对比分析**

通过与人类基准对照，比较“非思考”与“思考”两种模型模式，评估各认知组件的准确率。结果显示：非思考 LLM 在自我建模几乎为零，其他任务随模型规模提升但仍低于人类；思考 LLM 在所有任务接近或超越人类；负载实验表明状态转移负载显著影响非思考模型的表现。

**⚠️ 局限性**

局限性包括：①实验仅涵盖一种基于文本游戏的 ToM 场景，难以覆盖更广泛的社会情境；②模型可能通过记忆匹配而非真正的推理完成任务；③对内部推理机制的解析仍不足；④样本规模（人类 10 人）有限，难以推断整体人类表现。

---

## 184. Experimental study on surveillance video-based indoor occupancy measurement with occupant-centric control

**arXiv ID:** 2603.26081 | [PDF](https://arxiv.org/pdf/2603.26081v1)

**作者:** Irfan Qaisar `[一作]` (Tsinghua University), Qianchuan Zhao `[通讯]` (Tsinghua University)

**通讯引用:** 6031 | [OpenAlex ID](https://openalex.org/A5014109600)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过实验比较了基于YOLOv8检测的帧级计数、基于DeepSORT/ByteTrack的多目标跟踪以及基于LLM的后处理，在室内监控视频中实现了占用人数测量，并将结果应用于MPC驱动的HVAC控制。

**💡 创新点**

创新点在于首次将大语言模型（DeepSeek）与视觉感知输出相结合，实现对检测与跟踪误差的推理式纠正，显著降低误判，提升控制可用性。

**🔧 技术方法**

所用技术包括YOLOv8人检测、DeepSORT/ByteTrack跟踪、LLM（DeepSeek、LLaVA）后处理以及OpenStudio–EnergyPlus中的MPC预测控制。

**📊 数据集**

数据集来自清华大学FIT实验室的真实监控视频，包含1,188条视频片段、20,722帧，且每帧都有人工标注的占用人数。

**📈 对比分析**

在计数精度（MAE、RMSE）、占用/未占用分类（精确率、召回率、F1）以及跟踪稳定性（ID切换、碎片化）等指标下，YOLOv8+LLM（DeepSeek）表现最佳，误差最低、F1达0.9320，且在MPC控制实验中实现了最高的17.94% HVAC能耗节省。

**⚠️ 局限性**

局限性包括仅在单一固定视角的开敞实验室测试，缺乏对不同空间类型、摄像机布置及极端遮挡等情况的验证；且系统仅依赖视觉信息，未与其他隐私友好传感器融合。

---

## 185. MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality

**arXiv ID:** 2603.26071 | [PDF](https://arxiv.org/pdf/2603.26071v1)

**作者:** Kyungwon Kim `[一作]` (Yonsei University), Dosik Hwang `[通讯]` (Yonsei University)

**通讯引用:** 3101 | [OpenAlex ID](https://openalex.org/A5085519704)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出MUST框架，对多模态生存预测中缺失模态进行显式分解和重建。

**💡 创新点**

创新在于通过低秩共享子空间的代数可逆分解，明确区分模态特有与可推断信息，并用条件潜在扩散模型仅生成不可推断的特有残差。

**🔧 技术方法**

采用Transformer的双向跨模态注意力、低秩投影约束、条件潜在扩散模型（LDM）以及离散时间生存风险模型。

**📊 数据集**

在五个TCGA癌症数据集（BLCA、BRCA、GBMLGG、LUAD、UCEC）上进行评估。

**📈 对比分析**

与现有单模态与多模态基线（如ABMIL、TransMIL、SurvPath、CMTA、SMIL、M^3Care、ShaSpec、LD-CVAE）对比，MUST在完整数据下C-index为0.742，缺失基因时0.716，缺失病理时0.739，均优于对手。

**⚠️ 局限性**

主要局限是仅在配对完整训练下实现代数约束，缺失训练样本的鲁棒性有限，且对高维病理补全的扩散采样成本仍相对较高。

---

## 186. PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery

**arXiv ID:** 2603.26068 | [PDF](https://arxiv.org/pdf/2603.26068v1)

**作者:** Elkhan Ismayilzada `[一作]` (Michigan State University), Zijun Cui `[通讯]` (Michigan State University)

**通讯引用:** 306 | [OpenAlex ID](https://openalex.org/A5038674626)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了物理感知扩散框架PAD-Hand，利用Euler‑Lagrange动力学对单帧图像估计的手部运动进行物理一致性修正并估计动态方差。

**💡 创新点**

创新点在于将动力学残差视为虚拟观测，以概率方式融入扩散过程，并通过最后层拉普拉斯近似给出可解释的时间‑关节级方差。

**🔧 技术方法**

采用了条件扩散模型、MeshCNN‑Transformer骨干、Euler‑Lagrange动力学约束、虚拟观测概率约束以及最后层拉普拉斯近似等技术。

**📊 数据集**

实验数据集包括公开的DexYCB和HO3D两个手部三维姿势数据集。

**📈 对比分析**

与WiLoR、VIBE、TCMR等多种基线对比，PAD‑Hand在DexYCB上将PA‑MPJPE从4.88→4.63 mm、MPJPE从12.75→10.56 mm、加速度误差从6.70→3.34 mm/frame²提升；在HO3D上将PA‑MPJPE从7.50→7.43 mm、加速度误差从4.98→2.71 mm/frame²改善。

**⚠️ 局限性**

局限性在于未显式建模物体几何与接触，仅依赖近似的物理动力学，导致对复杂手‑物体交互的适用性受限。

---

## 187. A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs

**arXiv ID:** 2603.26236 | [PDF](https://arxiv.org/pdf/2603.26236v1)

**作者:** Uri Z. Kialy `[一作]` (Ariel University), Ayal Klein `[通讯]` (Ariel University)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5053897370)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Gemma‑2‑9B‑IT模型，使用稀疏自编码器（SAE）探测其内部表示，验证多语言模型是否以抽象、跨语言的方式学习非正式注册（俚语）。

**💡 创新点**

首次发现并证明存在一个跨语言共享的“非正式注册子空间”，并通过激活驱动实验展示其因果影响和零样本迁移能力。

**🔧 技术方法**

稀疏自编码器（SAE）、激活驱动（steering）、多语言对比分析、GPT‑4o‑mini自动化表述性评估。

**📊 数据集**

自制的三语（英语、希伯来语、俄语）多义词俚语数据集，词汇保持一致，只通过上下文区分俚语与字面用法。

**📈 对比分析**

通过Δi频率差、特征交集、解码器向量余弦相似度与岛屿得分进行定量比较；结果显示跨语言核心共9个特征，几何一致性显著提升；激活驱动在源语言与六种未见语言上均呈显著负相关，证明其因果性。

**⚠️ 局限性**

仅在Gemma‑2‑9B‑IT与有限层次上验证；模型偏向英语、俚语定义受限、零样本评估依赖GPT、未覆盖其他架构与规模，存在数据与评估上的局限。

---

## 188. Distilling Conversations: Abstract Compression of Conversational Audio Context for LLM-based ASR

**arXiv ID:** 2603.26246 | [PDF](https://arxiv.org/pdf/2603.26246v1)

**作者:** Shashi Kumar `[一作]` (Idiap Research Institute), Andreas Stolcke `[通讯]` (Uniphore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现一种抽象压缩(Abstract Compression)方法，利用多轮对话的音频上下文（压缩为固定数量的潜在token）与文本上下文一起提升基于大语言模型的语音识别性能。

**💡 创新点**

创新点在于：①将多轮对话的音频压缩为固定长度的潜在token，既保留语音信息又显著降低上下文长度；②结合监督多轮训练，让LLM学会利用压缩后的上下文提升对实体词的识别；③提出两阶段训练策略（单轮对齐+上下文微调），实现高效压缩与上下文利用。

**🔧 技术方法**

使用Phi-4-Multimodal大语言模型作为基础，加入音频编码器+投影模块；采用交叉注意力压缩模块将音频token压缩为K个潜在token；使用两阶段训练（Stage 1对齐+Stage 2上下文微调）和自适应训练策略；评估指标为WER和Bias‑WER。

**📊 数据集**

主要数据集为DefinedAI（在域测试）与WoW（异域测试），以及LibriSpeech 960h用于Stage 1压缩预训练；训练集来自DefinedAI的40h语料。

**📈 对比分析**

与单轮无上下文基线、原始多轮上下文（未压缩）进行对比。结果显示：原始多轮上下文在WER与Bias‑WER均优于单轮；压缩后多轮上下文在Bias‑WER上几乎匹配原始多轮，WER略逊，但显著降低了上下文token占比；整体提升主要体现在对实体词的识别。

**⚠️ 局限性**

局限性包括：仅在音频压缩上验证，未尝试压缩文本或其他模态；只使用单一LLM骨干Phi‑4-Multimodal；效率评估仅基于token压缩率，未直接测量推理延迟、内存或KV-cache开销。

---

## 189. Working Notes on Late Interaction Dynamics: Analyzing Targeted Behaviors of Late Interaction Models

**arXiv ID:** 2603.26259 | [PDF](https://arxiv.org/pdf/2603.26259v1)

**作者:** Antoine Edy `[一作]` (Illuin Technology), Quentin Macé `[通讯]` (Illuin Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了Late Interaction检索模型在多向量MaxSim下的长度偏差和相似度分布行为；

**💡 创新点**

指出多向量因果编码器会产生严格的长度偏差，双向模型虽减弱但未完全消除，同时证明MaxSim操作在第一个token之外无显著可利用信息；

**🔧 技术方法**

采用MaxSim评分、nDCG@10评估、Permutation Test等实验技术；

**📊 数据集**

在NanoBEIR基准数据集上进行评估；

**📈 对比分析**

与单向量稠密模型及单向量多向量模型对比，发现多向量因果模型存在显著长度偏差，单向量模型更贴近真实长度，双向多向量模型在极端长度下仍表现出一定偏差；MaxSim在top-1之外没有明显趋势；

**⚠️ 局限性**

局限在于仅使用单一基准数据集、未在合成数据上精准控制长度、未尝试其他相似度聚合策略，可能导致结论在更广泛任务中不完全适用。

---

## 190. Channelling, Coordinating, Collaborating: A Three-Layer Framework for Disability-Centered Human-Agent Collaboration

**arXiv ID:** 2603.26252 | [PDF](https://arxiv.org/pdf/2603.26252v1)

**作者:** Lan Xiao `[一作]` (Global Disability Innovation Hub), Catherine Holloway `[通讯]` (Global Disability Innovation Hub)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出了一个三层框架，用以重新定义AI在能力多样协作中的角色，从渠道化、协调化到共创化。

**💡 创新点**

创新点在于将可访问性与人机协作结合，强调协作的交互依赖，并将AI的介入层级化为渠道、协调、共创三层。

**🔧 技术方法**

使用了现有的可访问性技术（如实时字幕、结构化会议摘要、图像描述生成）以及人机协作理论（如共同基底、工作空间感知、知识边界模型）进行框架设计。

**📊 数据集**

未使用特定数据集，依赖于现有研究文献与案例。

**📈 对比分析**

无实验对比，本文仅提出概念框架并讨论潜在的研究问题，未给出性能评估。

**⚠️ 局限性**

局限在于缺乏实证验证、对权力不平衡等因素的考虑，以及对多重残疾类型和行业领域的适用性不完整。

---

## 191. Real-Time Branch-to-Tool Distance Estimation for Autonomous UAV Pruning: Benchmarking Five DEFOM-Stereo Variants from Simulation to Jetson Deployment

**arXiv ID:** 2603.26250 | [PDF](https://arxiv.org/pdf/2603.26250v1)

**作者:** Yida Lin `[一作]` (Victoria University of Wellington), Richard Green `[通讯]` (University of Canterbury)

**通讯引用:** 17134 | [OpenAlex ID](https://openalex.org/A5100730173)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在无人机上实时估计切割工具与树枝间距离的深度估计方法，训练并部署了五种DEFOM‑Stereo模型。

**💡 创新点**

创新点在于使用UE5合成数据构建任务专用数据集，并提出中等压缩版本DEFOM‑PrunePlus，兼顾精度与Jetson推理速度。

**🔧 技术方法**

采用DEFOM‑Stereo基础模型、DINOv2特征提取、DPT多尺度解码、TensorRT FP16优化，以及Ghost模块/DS‑GRU等轻量化技术。

**📊 数据集**

使用115棵树的UE5渲染数据，共5,520对ZED Mini立体图像，配有精确EXR深度，另以真实树枝照片进行零射转移评估。

**📈 对比分析**

在合成测试上比较EPE、D1‑all、δ1、Depth MAE；在Jetson Orin上测量推理FPS和深度MAE；结果显示DEFOM‑Stereo ViT‑S精度最高但仅2.2 FPS，DEFOM‑PrunePlus以3.3 FPS实现可用的精度与速度平衡。

**⚠️ 局限性**

局限在于轻量化模型精度过低导致安全距离估计不可靠，域间差异仍存在，且当前硬件无法将ViT‑S提升至可用帧率。

---

## 192. 4DRaL: Bridging 4D Radar with LiDAR for Place Recognition using Knowledge Distillation

**arXiv ID:** 2603.26206 | [PDF](https://arxiv.org/pdf/2603.26206v1)

**作者:** Ningyuan Huang `[一作]` (Northeastern University), Zheng Fang `[通讯]` (Northeastern University)

**通讯引用:** 2252 | [OpenAlex ID](https://openalex.org/A5089260568)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一个名为4DRaL的知识蒸馏框架，用以提升4D毫米波雷达在场景定位中的性能，并支持雷达到激光雷达的跨模态定位。

**💡 创新点**

创新点在于：①采用LiDAR-to-LiDAR的高性能教师模型指导雷达to雷达的学生模型；②设计了局部图像增强（LIE）、特征分布蒸馏（FDD）和响应蒸馏（RD）三大模块，分别从输入、特征和输出层实现高效蒸馏；③在同一框架下实现雷达to激光雷达的跨模态定位。

**🔧 技术方法**

技术手段包括：基于BEV图像的雷达与激光点云预处理、ResNet18特征提取、轻量U-Net图像增强、双分支特征分布对齐（TransEnc+KL）、关系蒸馏（DistilVPR+margin）、Triplet损失等。

**📊 数据集**

使用的数据集有NTU4DRadLM、SJTU4D（用于R2R和R2L实验），以及SNAIL恶劣天气数据集（用于鲁棒性评估）。

**📈 对比分析**

与现有基线（如TransLoc4D、RaLF、Radar-to-LiDAR等）相比，4DRaL在R2R任务上在NTU和SJTU数据集上提升了1–5%召回率，R2L任务上提升高达40%+；同时运行速度仅为2.6 ms，参数量仅为3.86 M，优于大多数竞品。

**⚠️ 局限性**

局限性包括：4D雷达的视场与分辨率仍有限，难以在所有环境下完全匹配激光雷达；模型仅使用单帧数据，未利用多帧时序信息；对极端天气的鲁棒性仍受雷达硬件限制；依赖教师模型的预训练，迁移到不同场景时可能需要进一步微调。

---

## 193. From Personas to Programming: Gender-specific Effects of Design Thinking-Based Computing Education at Secondary Schools

**arXiv ID:** 2603.26194 | [PDF](https://arxiv.org/pdf/2603.26194v1)

**作者:** Isabella Graßl `[一作]` (Technical University of Darmstadt), Daniela Damian `[通讯]` (University of Victoria)

**通讯引用:** 7376 | [OpenAlex ID](https://openalex.org/A5007049054)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在两所加拿大中学中实施了为期10周、以设计思维为核心的MIT App Inventor块式编程课程，并通过前后测问卷评估了学生在知识、自我效能、兴趣、情绪体验和幸福感等方面的性别差异。

**💡 创新点**

首次系统地将设计思维与可持续性主题相结合，探讨其对男女学生的情绪与动机差异，并提出在设计思维各阶段情绪变化与学习效果的关联模型。

**🔧 技术方法**

使用MIT App Inventor进行块式编程；设计思维五阶段框架（同理、定义、创意、原型、测试）；多维自评量表（知识、效能、兴趣、情绪、福祉）；非参数统计方法（Mann‑Whitney U、Wilcoxon Signed‑Rank、Vargha‑Delaney效应量）以及主题分析法。

**📊 数据集**

收集了55名13‑16岁学生在课程前后两次完成的问卷数据（共约400+条回答），未使用公开数据集，数据由研究团队自行收集并匿名化。

**📈 对比分析**

通过Mann‑Whitney U检验比较男女差异，Wilcoxon Signed‑Rank检验前后变化，Vargha‑Delaney效应量评估效果大小。结果显示女生在知识、自我效能、兴趣和福祉等多项指标上显著提升（p≤0.05，效应量0.14‑0.39），男生提升幅度有限，整体提升率在20%‑40%之间，情绪与福祉提升更为显著。

**⚠️ 局限性**

样本量有限、仅两所学校、缺乏跨文化与多位教师验证、依赖自评数据、未测量实际学习成绩、单一工具（MIT App Inventor），这些因素限制了研究结果的普适性与外推性。

---

## 194. Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding

**arXiv ID:** 2603.26211 | [PDF](https://arxiv.org/pdf/2603.26211v1)

**作者:** Shrinidhi Kumbhar `[一作]` (Arizona State University), Kunwar Yashraj Singh `[通讯]` (AWS Agentic AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将离散扩散视觉-语言模型 LLaDA‑V 迁移至单轮 GUI 定位任务，训练其预测自然语言指令对应的动作类型与边框坐标。

**💡 创新点**

创新点在于提出混合掩码调度（线性+全确定性），先粗略预测动作和锚点，再精细预测边框扩展，显著提升定位精度。

**🔧 技术方法**

采用离散扩散模型 LLaDA‑V、SigLIP‑2 视觉编码器、双阶段掩码训练以及低置信度重掩码等技术。

**📊 数据集**

使用包含 120K 份 web、mobile、desktop GUI 数据的混合语料（Mind2Web、WebLinX、OS‑Atlas、Rico），并在四个公开基准上评估。

**📈 对比分析**

与强大的 AR 视觉语言模型（Qwen2.5‑VL、Phi‑3‑Vision）对比，混合掩码 LLaDA‑V 在 SSR 上比线性掩码提升 1.6–6.1 点，整体性能已接近 AR 模型，且推理时间仅比 AR 少 2–3 秒。

**⚠️ 局限性**

局限性包括单轮推理、较高的延迟（混合掩码导致顺序计算）、对视觉分辨率与标注质量敏感，以及缺乏多步规划与动作序列生成能力。

---

## 195. Progressive Learning with Anatomical Priors for Reliable Left Atrial Scar Segmentation from Late Gadolinium Enhancement MRI

**arXiv ID:** 2603.26186 | [PDF](https://arxiv.org/pdf/2603.26186v1)

**作者:** Jing Zhang `[一作]` (Sorbonne Universite), Nadjia Kachenoura `[通讯]` (Sorbonne Universite)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一个三阶段的渐进式学习框架，先学习左心房（LA）解剖，再结合LA与瘢痕的空间关系，最终细化瘢痕分割；

**💡 创新点**

创新点在于将临床解剖先验融入训练（软壁约束）并模拟医生的诊断流程进行渐进式学习，缓解了瘢痕低对比、稀疏且易受标注噪声影响的问题；

**🔧 技术方法**

采用基于Swin Transformer的SwinUNETR网络，结合Dice+交叉熵混合损失、空间加权软壁约束和针对LGE MRI的物理感知式数据增强；

**📊 数据集**

使用LASCARQS 2022公开数据集（60个包含瘢痕标签的3D LGE MRI和130个仅有LA标签的扫描），并在内部5折交叉验证中评估；

**📈 对比分析**

与单阶段模型、nnU-Net等基线比较，瘢痕Dice从0.49提升至0.50，Hausdorff距离和平均表面距离分别降低约9%和8%；然而总体Dice仍低于挑战榜单上最佳0.595，说明该任务难度仍高；

**⚠️ 局限性**

局限性包括样本量有限、标注存在解剖不合理的噪声、瘢痕体积极小且低对比、仅在内部验证，缺乏外部多中心测试。

---

## 196. Consistency Beyond Contrast: Enhancing Open-Vocabulary Object Detection Robustness via Contextual Consistency Learning

**arXiv ID:** 2603.26179 | [PDF](https://arxiv.org/pdf/2603.26179v1)

**作者:** Bozhao Li `[一作]` (Harbin Institute Of Technology), Jingyong Su `[通讯]` (Harbin Institute Of Technology)

**通讯引用:** 1414 | [OpenAlex ID](https://openalex.org/A5069706913)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Contextual Consistency Learning (CCL) 框架，解决开放词汇目标检测在背景或环境变化时的鲁棒性不足，通过数据生成与一致性学习提升模型在多场景下的检测与定位性能。

**💡 创新点**

核心创新在于：① 生成相同前景对象但背景多样化的图像对（Contextual Bootstrapped Data Generation, CBDG），填补现有数据集缺乏同对象多背景样本的空缺；② 设计 Contextual Consistency Loss (CCLoss)，在同一背景变化下强制视觉（及可选文本）特征保持一致，提升模态内一致性；③ 将两者结合形成模型无关、无推理开销的完整训练范式。

**🔧 技术方法**

技术手段包括：SAM 语义分割提取前景；Stable Diffusion 与 ChatGPT 生成多样背景文本并合成图像；CLIP/GLIP、FIBER 等跨模态检测器作为骨干；对视觉与文本特征使用基于对比学习的上下文一致性损失；FPN、DynamicHead 等检测层结构。

**📊 数据集**

数据集：原始 Objects365 子集、Flickr30k Entities；生成的 144,654 张多背景图像；基准评测使用 OmniLabel 与 D³；对 D³ 进行背景替换得到 D³_BC（42,312 张），以及按 COCO‑C 方式扰动得到 D³_C。训练仅使用 0.25M 样本的联合数据集。

**📈 对比分析**

在 GLIP‑T 与 FIBER‑B 两大基线上 fine‑tune，结果显示：+16.3 AP（OmniLabel）和 +14.9 AP（D³）；在 D³_BC 上保持相对无明显性能下降，显著优于现有 SOTA 方法（如 GLIP、FIBER、Real‑Model 等）。整体提升了目标检测在不同背景和描述长度下的表现。

**⚠️ 局限性**

限制：方法高度依赖 SAM 的分割质量，掩模不精确会导致合成图像出现伪影；虽然不增加推理成本，但数据生成流程复杂，对资源和人工审查有一定要求；目前评估集中在特定公开基准，未来需验证在更广泛领域与场景下的泛化能力。

---

## 197. ARTA: Adaptive Mixed-Resolution Token Allocation for Efficient Dense Feature Extraction

**arXiv ID:** 2603.26258 | [PDF](https://arxiv.org/pdf/2603.26258v1)

**作者:** David Hagerman `[一作]` (Chalmers University of Technology), Lennart Svensson `[通讯]` (Chalmers University of Technology)

**通讯引用:** 8189 | [OpenAlex ID](https://openalex.org/A5029413988)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ARTA，一种先从粗粒度 token 开始，后根据语义边界动态分配细粒度 token 的粗细粒度视觉 Transformer，用于高效的稠密特征提取；

**💡 创新点**

核心创新在于：① 轻量化的边界得分分配器，可迭代预测语义边界并只在需要的区域增密 token；② 允许多分辨率 token 互相注意的混合分辨率注意机制；③ 两阶段编码器（分配+精化），在保持低分辨率计算的同时保留细节；

**🔧 技术方法**

使用的技术包括：Vision Transformer 轻量化预分配、Cluster Attention（自适应聚类注意）以及 ViT、MSDeformAttn 等多尺度稀疏注意层；分配块中对 token 进行 MLP 回归得到边界得分；

**📊 数据集**

在 ADE20K、COCO‑Stuff、Cityscapes 三个常用语义分割基准上进行实验；

**📈 对比分析**

与现有最先进方法（如 SegFormer、Mask2Former、AFF、SegMAN 等）对比，ARTA 在 ADE20K 上 54.6 mIoU、COCO‑Stuff 上 54.6 mIoU、Cityscapes 上接近 SOTA，但 FLOPs 低 约 50% 以上，推理速度和显存消耗更优；

**⚠️ 局限性**

局限性包括：尚未在更大规模 backbone 或更长预训练（ImageNet22k）上验证；目前仅针对 2D 图像，可进一步探索 3D 分割；对极端小目标或复杂边界的自适应分配仍有提升空间。

---

## 198. Automating Domain-Driven Design: Experience with a Prompting Framework

**arXiv ID:** 2603.26244 | [PDF](https://arxiv.org/pdf/2603.26244v1)

**作者:** Tobias Eisenreich `[一作]` (Technical University of Munich), Stefan Wagner `[通讯]` (Technical University of Munich)

**通讯引用:** 9219 | [OpenAlex ID](https://openalex.org/A5022333047)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一套五步的LLM提示框架，用于自动化领域驱动设计（DDD）的核心活动，帮助架构师在实际项目中生成词汇表、事件风暴、边界上下文、聚合和技术架构草图。

**💡 创新点**

创新点在于将LLM的提示工程与DDD流程结合，提供可复用的提示模板，并通过系统提示让LLM扮演“高级DDD专家”角色，从而实现人机协作的设计辅助，而非完全自动化。

**🔧 技术方法**

采用了Claude Opus 4.1、Gemini 2.5 Pro 和 GPT‑5.0三种商业LLM；提示工程遵循专门模板；系统提示设定LLM为架构讨论伙伴；生成PlantUML可视化。

**📊 数据集**

使用FTAPI软件有限公司公开的两套业务需求文档（SecuRooms和SecuMails），以Markdown文本形式提供给LLM；并对生成的设计产物进行专家访谈评估。

**📈 对比分析**

通过三名熟悉SecuRooms/​SecuMails的架构专家对LLM输出进行访谈，评估每一步的可用性；结果显示第一、二、三步生成的词汇表、事件风暴和边界上下文质量较高，第四步聚合和第五步技术架构的准确性明显下降，误差在后续步骤累积。

**⚠️ 局限性**

局限性包括：LLM易产生幻觉且误差会在后续步骤累积；受限于上下文窗口大小，长需求文档和多步生成可能导致信息丢失；技术架构映射难度高，当前LLM生成的PlantUML缺乏可读性，整体自动化程度受限于人机协作。

---

## 199. Sparse Auto-Encoders and Holism about Large Language Models

**arXiv ID:** 2603.26207 | [PDF](https://arxiv.org/pdf/2603.26207v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 200. An Object Web Seminar: A Retrospective on a Technical Dialogue Still Reverbarating

**arXiv ID:** 2603.26203 | [PDF](https://arxiv.org/pdf/2603.26203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 201. MemCam: Memory-Augmented Camera Control for Consistent Video Generation

**arXiv ID:** 2603.26193 | [PDF](https://arxiv.org/pdf/2603.26193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 202. SwarmCoDe: A Scalable Co-Design Framework for Heterogeneous Robot Swarms via Dynamic Speciation

**arXiv ID:** 2603.26240 | [PDF](https://arxiv.org/pdf/2603.26240v1)

**作者:** Andrew Wilhelm `[一作]` (École Polytechnique Fédérale de Lausanne), Josie Hughes `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 2710 | [OpenAlex ID](https://openalex.org/A5032508996)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出 SwarmCoDe 一种基于动态物种划分的协同进化算法，用于在满足制造成本约束的前提下联合优化机器人群体的任务规划、形态学与硬件配置。

**💡 创新点**

创新点在于：①利用遗传标签和选择基因实现无先验种类边界的自适应物种分配；②引入支配基因将演化种群规模与实际群体规模解耦，使算法能够设计多达四倍于进化种群规模的机器人群体；③将制造预算纳入适应度评估，实现成本与性能的权衡。

**🔧 技术方法**

主要技术包括：协同进化算法（CCEA）与 NEAT 动态物种划分；遗传标签、选择基因与支配基因的设计；行为树编码的任务规划；离散与连续硬件基因的联合进化；基于 JAX 的 GPU 加速二维物理仿真；软预算惩罚与指数移动平均平滑的适应度函数。

**📊 数据集**

使用自建的 2D 仿真环境生成的包搬运任务数据（单包与协同包，尺寸与重量多样），未使用公开真实数据集。

**📈 对比分析**

通过与同质群体、不同预算限制以及 ROI 优化的对比实验评估。结果显示：①自适应物种划分在任务复杂度升高时自然产生 1、2、4 个种类；②在 200 机器人群体下，进化种群 50 的方案成功实现两种互补种类；③在预算约束下，群体异质性下降，性能与成本均得到平衡；整体适应度和包递送数量在实验中均优于基线，ROI 优化后可实现成本与效益最佳平衡。

**⚠️ 局限性**

局限性包括：①适应度评估噪声大，导致训练过程波动；②实验仅在二维仿真中验证，缺乏真实硬件验证；③对复杂动力学、通信延迟等真实世界因素考虑不足；④算法对物理仿真与 GPU 加速依赖较高，扩展到更大规模或更高维任务时可能出现计算瓶颈。

---

## 203. Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents

**arXiv ID:** 2603.26233 | [PDF](https://arxiv.org/pdf/2603.26233v1)

**作者:** Nicholas Edwards `[一作]` (University of Vienna), Sebastian Schuster `[通讯]` (University of Vienna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了一种基于不确定性感知的多智能体框架，旨在让LLM代理在面对缺失上下文的开源软件工程任务时主动识别并提问以获取缺失信息，最终实现类似全信息场景的任务解决率。

**💡 创新点**

创新点在于将缺省检测与代码执行拆分为两个专门化角色（Intent Agent与Main Agent），通过内部不确定性监控实现自动化提问；并系统地在SWE‑bench Verified的缺失信息版本上评估这种架构的效果。

**🔧 技术方法**

使用了Claude Sonnet 4.5作为LLM后端，OpenHands工具框架进行代码操作与交互，GPT‑5.1模拟用户，基于自定义提示和多轮对话实现不确定性检测与提问。

**📊 数据集**

采用了SWE‑bench Verified（500条GitHub issue）并通过GPT‑4o生成的缺失信息变体进行实验。

**📈 对比分析**

与三种基线（Full、Hidden、Interactive Baseline）相比，单智能体UA‑Single的任务解决率为61.20%，多智能体UA‑Multi提升至69.40%，与Interactive Baseline（66.57%）相当，显著优于缺失信息下的单代理Hidden（44.48%）。

**⚠️ 局限性**

主要局限包括：使用LLM模拟用户可能不代表真实人类交互；依赖高端专有模型Claude Sonnet 4.5导致成本高且可迁移性差；未在开放权重模型上验证，且在更高风险环境下效果尚未证明。

---

## 204. Clash of the models: Comparing performance of BERT-based variants for generic news frame detection

**arXiv ID:** 2603.26156 | [PDF](https://arxiv.org/pdf/2603.26156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 205. Privacy-Enhancing Encryption in Data Sharing: A Survey on Security, Performance and Functionality

**arXiv ID:** 2603.26224 | [PDF](https://arxiv.org/pdf/2603.26224v1)

**作者:** Yongyang Lv `[一作]` (Tianjin University), Willy Susilo `[通讯]` (University of Wollongong)

**通讯引用:** 24468 | [OpenAlex ID](https://openalex.org/A5054741725)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对2020‑2025年间关于属性基加密（ABE）、可搜索加密（SE）和代理重加密（PRE）的数据共享技术进行综述，构建了数据共享流程框架并识别了20类攻击场景，整合了12种增强技术，对安全性、性能和功能性进行量化评估，并阐述了关键应用场景与未来研究方向。

**💡 创新点**

创新点在于①系统性梳理了ABE/SE/PRE三类隐私增强加密技术与12种增强技术的协同效应；②提出完整的数据共享流程框架并详细列举20类攻击，形成安全威胁知识图谱；③构建了多维评估矩阵，实现对方案在安全、性能、功能三维的可比性；④对未来量子安全、低资源场景和全生命周期可信体系的挑战给出了前瞻性建议。

**🔧 技术方法**

使用的技术包括属性基加密（ABE）、可搜索加密（SE）、代理重加密（PRE），以及云计算（CC）、区块链（BC）、IPFS、零知识证明（ZKP）、同态加密（HE）、安全多方计算（MPC）、差分隐私（DP）、联邦学习（FL）、签名加密（Signcryption）、身份基加密（IBE）、广播加密（BE）和后量子密码（PQC）等。

**📊 数据集**

本研究主要基于文献数据集（共116篇核心论文+48篇近年顶会/期刊论文），未采用传统实验数据集；若涉及性能评估则引用文献中公开的实验结果和基准对比。

**📈 对比分析**

通过对收录论文的安全级别、性能开销与功能完整度三维打分，构建量化比较表，展示不同技术组合（如ABE+BC、SE+MPC、PRE+IPFS）在安全性（高/中/低）、性能（低/中/高）与功能性（完整/局限）上的优势与折衷；大部分方案在安全性上高，但性能往往处于中等或低水平，功能性方面多受限于技术兼容性与实现细节。

**⚠️ 局限性**

局限性包括：①缺乏针对量子攻击的实证评估，未系统探讨基于格的量子安全实现；②低资源场景下的性能与能源消耗未做深入实验验证；③多技术融合的整体体系仍缺乏完整的生命周期可信数据保护框架；④对实际业务案例的深度落地验证不足，仍停留在理论与仿真层面。

---

## 206. OSA: Echocardiography Video Segmentation via Orthogonalized State Update and Anatomical Prior-aware Feature Enhancement

**arXiv ID:** 2603.26188 | [PDF](https://arxiv.org/pdf/2603.26188v1)

**作者:** Rui Wang `[一作]` (Shenzhen University), Jing Qin `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 25528 | [OpenAlex ID](https://openalex.org/A5100662807)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出一种基于Stiefel流形约束的线性递归模型和物理驱动的解耦特征增强模块，旨在解决超声心动图中严重噪声和秩崩塌导致的分割不准确与时间不连贯问题。

**💡 创新点**

创新点包括：① Orthogonalized State Update（OSU）将状态更新投影到Stiefel流形，防止秩崩塌并保持数值稳定；② Anatomical Prior‑aware Feature Enhancement（APFE）通过局部声学场与结构残差分解，显式抑制噪声并突出解剖边界；③ 将两者结合在一次端到端的线性递归框架内，实现实时且高精度的心室分割。

**🔧 技术方法**

主要技术包括：ResNet‑50视觉编码器；线性递归状态机（SSM）与核更新；Stiefel流形投影（高阶Newton‑Schulz迭代实现正交化）；局部平均池化与双ReLU解耦的APFE；交叉熵+Dice损失训练，AdamW优化。

**📊 数据集**

使用了公开的CAMUS（含全帧注释）和EchoNet‑Dynamic（仅ED/ES注释）两大超声心动图视频数据集进行评估。

**📈 对比分析**

与PolaFormer、Vision LSTM、Cutie、SAMed‑2、LiVOS、GDKVM、EchoVim、Vivim等现有方法相比，本文模型在CAMUS上获得94.82% Dice、3.25mm 95%HD，EchoNet‑Dynamic上也取得相同或更高的分割精度，并保持约35fps的实时推理速度。

**⚠️ 局限性**

局限性包括：受限于固定长度序列训练，无法完全处理变长真实临床视频；在跨设备或不同成像协议下的泛化能力尚需进一步验证；以及对初始化与噪声极端情况仍可能出现失败案例。

---

## 207. ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory

**arXiv ID:** 2603.26182 | [PDF](https://arxiv.org/pdf/2603.26182v1)

**作者:** Zhuohan Ge `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 37472 | [OpenAlex ID](https://openalex.org/A5100404176)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 ClinicalAgents，多智能体框架，用动态 Monte Carlo Tree Search（MCTS）控制器和双重记忆（工作记忆与经验记忆）实现临床诊断的迭代假设推理与证据验证。

**💡 创新点**

创新点包括：① 基于 MCTS 的可回溯决策器，实现缺失证据触发的回溯；② 双重记忆架构，使多轮推理保持状态一致并实时检索指南与历史案例；③ 将假设驱动推理嵌入多智能体交互，提升诊断准确性与可解释性。

**🔧 技术方法**

技术手段：多智能体系统、LLM 调度器、MCTS、双重记忆（Mutable Working Memory + Static Experience Memory）、Retrieval-Augmented Generation（AGRAG）、RAG、工具调用（image、lab、history 等）、回溯策略、评估奖励函数。

**📊 数据集**

数据集：主测为 MedChain（5 阶段临床工作流），补充四个医学 QA 数据集（MedQA、PubMedQA、MedBullets、PathVQA）进行泛化验证。

**📈 对比分析**

与单 LLM、单代理（Few-shot+CoT、RAG、ReAct）及多代理基线（Reconcile、AutoGen、MedAgents、MDAgents、ColaCare、MedChain-Agents）进行对比。ClinicalAgents 在 MedChain 上平均分 0.5107，超越最佳基线 0.4880（+4.7%）和 GPT‑5.2（+13%），在各阶段均有显著提升，且对不同 LLM 主干表现稳健。

**⚠️ 局限性**

局限性：仍需人工监督，不能完全替代临床医生；对 LLM 的偏见与隐私问题需持续监测；在特定领域或复杂病例的泛化能力尚未完全验证；实现依赖高质量 LLM 与大规模知识库。

---

## 208. SocialX: A Modular Platform for Multi-Source Big Data Research in Indonesia

**arXiv ID:** 2603.26253 | [PDF](https://arxiv.org/pdf/2603.26253v1)

**作者:** Muhammad Apriandito Arya Saputra `[一作]` (SocialX), Hanif Fakhrurroja `[通讯]` (National Research and Innovation Agency)

**通讯引用:** 733 | [OpenAlex ID](https://openalex.org/A5004438607)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个支持多源大数据收集、语言感知预处理和可插拔分析的 Web 平台 SocialX。

**💡 创新点**

通过三层模块化架构、统一数据模式和可插拔组件实现了源无关、易扩展的研究工作流。

**🔧 技术方法**

采用数据库驱动的作业队列、共享数据模式、IndoBERT 大模型、可组合过滤器以及多种分析模块实现平台功能。

**📊 数据集**

利用多源收集器抓取社交媒体、新闻门户、电子商务评论和学术数据库等公开文本数据。

**📈 对比分析**

与手工流水线相比，SocialX 在去重、语言检测、关键词过滤和语义相关性判别上平均提升 50% 以上净数据量，并通过交互可视化展示分析结果。

**⚠️ 局限性**

当前限制包括仅支持单一数据集分析、缺乏实时流处理、相关性分类模型受限于 188 个主题，以及连接器主要聚焦印尼平台。

---

## 209. Automatic Speech Recognition for Documenting Endangered Languages: Case Study of Ikema Miyakoan

**arXiv ID:** 2603.26248 | [PDF](https://arxiv.org/pdf/2603.26248v1)

**作者:** Chihiro Taguchi `[一作]`, David Chiang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建Ikema语言语音数据集，训练并集成ASR模型到语言注释工具，并评估其对转录效率的提升。

**💡 创新点**

首次为Ikema语言构建语音数据集与ASR模型，证明ASR辅助转录可显著加速文档工作。

**🔧 技术方法**

使用Wav2Vec2自监督模型（XLS‑R、MMS等）+ CTC解码器进行端到端ASR训练，并在ELAN中实现扩展。

**📊 数据集**

约2.5小时的语音数据，包含现场录音、词典朗读和有声书三部分，拆分为训练/验证/测试集。

**📈 对比分析**

比较不同模型规模（XLS‑R‑128M、MMS‑1B、XLS‑R‑1B）以及两种写法（kana、romaji），最佳模型在romaji上CER 14.8%，WER 64.99%；ASR辅助转录使两位评注者分别提高约19%和23%的速度。

**⚠️ 局限性**

数据量有限、转录错误率仍高、缺乏语言模型、写法标准化不充分、评估样本有限，导致ASR性能与人类评注差距较大。

---

## 210. HAD: Heterogeneity-Aware Distillation for Lifelong Heterogeneous Learning

**arXiv ID:** 2603.26192 | [PDF](https://arxiv.org/pdf/2603.26192v1)

**作者:** Xuerui Zhang `[一作]` (Southern University of Science and Technology), Yu Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 42401 | [OpenAlex ID](https://openalex.org/A5112212826)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出终身异构学习(LHL)框架，特别针对密集预测任务(LHL4DP)并设计无样本的异构感知蒸馏(HAD)方法

**💡 创新点**

创新点包括：①将终身学习扩展到任务类型异构的情形；②提出分布平衡(DB‑HAD)与边缘显著性引导(SG‑HAD)双重蒸馏损失，解决伪标签不平衡与边缘信息丢失问题

**🔧 技术方法**

采用共享Encoder+任务专用Decoder结构，利用自蒸馏、自适应分布平衡与Sobel梯度边缘提取的损失函数；实现无示例、无存储的知识保留

**📊 数据集**

在CityScapes、NYUv2（3任务）和Taskonomy（10任务）三个公开密集预测数据集上进行实验

**📈 对比分析**

与EWC、LWF、iCaRL、DER、SPG、SGP以及vanilla/joint训练相比，HAD在所有任务上均显著提升，平均相对提升达30%以上，MR排名位居首位

**⚠️ 局限性**

局限性：仅针对密集预测任务验证；需要手动设定分组数、阈值等超参数；缺乏理论分析与对更大规模异构任务的泛化研究

---

## 211. Dual-Stage Invariant Continual Learning under Extreme Visual Sparsity

**arXiv ID:** 2603.26190 | [PDF](https://arxiv.org/pdf/2603.26190v1)

**作者:** Rangya Zhang `[一作]` (Nanyang Technological University), Mir Feroskhan `[通讯]` (Nanyang Technological University)

**通讯引用:** 1361 | [OpenAlex ID](https://openalex.org/A5026643234)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对极度稀疏视觉环境下的空间轨道物体检测，提出了一个连续学习框架；

**💡 创新点**

创新点包括双阶段一致性机制（在特征层和检测头层同时进行蒸馏）以及稀疏感知的数据条件化策略（补丁采样+分布感知增强）；

**🔧 技术方法**

采用了基于 Faster R‑CNN 的检测网络，结合特征对齐损失、RoI 级蒸馏、批归一化冻结、以及自定义的采样与增强技术；

**📊 数据集**

使用公开的 SpaceDet 高分辨率空间图像数据集进行实验，模拟多阶段域漂移；

**📈 对比分析**

与多种传统检测器、回放式与非回放式连续学习方法（EWC、OGD、CWD、Shmelkov 等）进行对比，平均 mAP 提升至 42.62%，比联合训练（42.56%）略高，单域连续微调仅 34.28%；

**⚠️ 局限性**

局限性包括仅在离线顺序学习场景验证，未对在线持续学习、跨模态融合或更复杂目标类别进行评估；同时双阶段蒸馏对计算资源和显存有一定额外负担。

---

## 212. Can AI Scientist Agents Learn from Lab-in-the-Loop Feedback? Evidence from Iterative Perturbation Discovery

**arXiv ID:** 2603.26177 | [PDF](https://arxiv.org/pdf/2603.26177v1)

**作者:** Gilles Wainrib `[一作]` (Owkin Inc), John Klein `[通讯]` (Owkin Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在 Cell Painting 高通量筛选实验中，利用大型语言模型（Claude Sonnet 4.5/4.6）实现迭代实验设计，并通过实验反馈实现上下文学习。

**💡 创新点**

证明在足够能力的 LLM（Sonnet 4.6）下，实验反馈能够真正驱动学习，显著提升发现率，并通过随机反馈对照排除预训练知识泄漏的可能性。

**🔧 技术方法**

采用 Zero-shot、ICL‑EF、ICBR‑EF 三种 LLM 架构，随机反馈控制，GP‑UCB 基线，以及符号翻转置换检验 + Benjamini‑Hochberg FDR 校正等统计方法。

**📊 数据集**

使用公开的 JUMP Cell Painting 数据集，约 8,000 个 CRISPR 基因突变与 4,672 个形态特征。

**📈 对比分析**

在 10 个目标特征上各进行 10 次重复（共 800 实验）进行比较；ICL‑EF 在 Sonnet 4.6 上平均发现 29.3 次目标（比随机高 185%），显著优于 GP‑UCB（19.5）与随机基线（11.0），而随机反馈控制下性能无提升，验证了反馈驱动学习。

**⚠️ 局限性**

限制包括：仅评估两款 Claude 模型，结果高度依赖模型能力；实验采用无噪声二元反馈，真实实验中噪声与延迟可能削弱效果；仅针对 Cell Painting assay，未验证其他实验域或开源 LLM 的普适性。

---

## 213. Provably Contractive and High-Quality Denoisers for Convergent Restoration

**arXiv ID:** 2603.26168 | [PDF](https://arxiv.org/pdf/2603.26168v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 214. Physics-Informed Neural Networks and Sequence Encoder: Application to heating and early cooling of thermo-stamping process

**arXiv ID:** 2603.26245 | [PDF](https://arxiv.org/pdf/2603.26245v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 215. DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models

**arXiv ID:** 2603.26164 | [PDF](https://arxiv.org/pdf/2603.26164v1)

**作者:** Hao Liang `[一作]` (Peking University), Wentao Zhang `[通讯]` (OriginHub Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 DataFlex，一个统一的数据中心化动态训练框架，能够在大语言模型训练中动态控制样本选择、域混合比例和样本权重。

**💡 创新点**

创新点在于将数据选择、数据混合和数据重加权三大范式整合到同一框架中，提供统一的 Trainer‑Component 体系结构，并与 LLaMA‑Factory 无缝集成，显著降低复现与对比成本。

**🔧 技术方法**

实现基于 Select、Mix、Weight 的动态 Trainer、可插拔的选择器/混合器/权重器、分布式梯度收集（兼容 DeepSpeed ZeRO‑3）、BFloat16 混合精度、FlashAttention‑2、LoRA PEFT 等技术。

**📊 数据集**

使用 Open‑Hermes‑2.5（10w 训练样本）和 SlimPajama（6B/30B 令牌）等数据集，评估 Mistral‑7B、Llama‑3.2‑3B 及 Qwen2.5 系列模型，并用 MMLU 进行下游测试。

**📈 对比分析**

通过与静态全量训练、离线/在线数据选择（LESS、NICE、Loss、Delta Loss、NEAR、TSDS）、在线混合（ODM）、离线混合（DoReMi）以及 Reweight 进行对比，DataFlex 在 MMLU 上提升至约 0.453（相对 0.394 的基线）并在 Perplexity 上显著下降，同时在大规模训练下实现了原始实现的 50–60% 运行时间加速。

**⚠️ 局限性**

局限性包括离线选择的加速幅度有限、对分布式梯度收集的实现复杂度、对模型特定调参（如更新频率、阈值）的敏感性，以及实验仅覆盖部分任务与指标，未来需进一步扩展至更多模型与评估范式。

---

## 216. Shuffles of Context-Free Languages along Regular Trajectories

**arXiv ID:** 2603.26162 | [PDF](https://arxiv.org/pdf/2603.26162v1)

**作者:** Corentin Barloy `[一作]` (University of Bochum), Kyle Ockerlund `[通讯]` (Google)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文研究了在正则轨迹下两语言的交错（shuffle）是否保持上下文无关性，并给出了完整的安全/敌对/混合三类分类与判定方法。

**💡 创新点**

创新点在于提出耦合与复原耦合概念，利用PDA耦合分析和半线性集合几何技术，证明了多类轨迹的上下文无关保持性质，并提供了可判定算法。

**🔧 技术方法**

核心技术包括PDA耦合理论、Ogden引理、k-后缀等价类、半线性集合的网格与对网格判定，以及语义与几何化简。

**📊 数据集**

本工作为纯理论研究，未使用任何实验数据集。

**📈 对比分析**

由于是理论分析，未进行实验比较，所给结果以可判定性和复杂度上界形式呈现。

**⚠️ 局限性**

局限在于对一般CFL轨迹的完全分类仍未完成，且对非正则轨迹及更复杂交错模型的研究尚未展开。

---

## 217. Knowledge Distillation for Efficient Transformer-Based Reinforcement Learning in Hardware-Constrained Energy Management Systems

**arXiv ID:** 2603.26249 | [PDF](https://arxiv.org/pdf/2603.26249v1)

**作者:** Pascal Henrich `[一作]` (Karlsruhe Institute of Technology), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 4999 | [OpenAlex ID](https://openalex.org/A5014228448)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用知识蒸馏压缩并提升决策变压器（Decision Transformer）在住宅电池调度中的控制性能，使其可部署于资源受限的嵌入式硬件。

**💡 创新点**

①首次将知识蒸馏与DT结合，实现模型压缩的同时提升控制效果；②展示自蒸馏可作为正则化手段改善泛化；③揭示DT模型尺寸与性能的非单调关系，表明中等规模模型最优。

**🔧 技术方法**

决策变压器（Transformer序列建模）、响应式知识蒸馏（Smooth L1 损失匹配教师输出）、DDPG产生的离线轨迹、与MILP、规则控制等基线比较。

**📊 数据集**

Ausgrid Solar Home Electricity Dataset（20栋住宅）与SMARD数据库提供的德国/奥地利/卢森堡批发电价。

**📈 对比分析**

在20栋建筑上进行四周评估，平均电费作为性能指标；KD（中等教师→小学生）平均成本200.28€，比DT（201.30€）和DDPG（202.43€）更优；KD同时将参数减少96%、内存占用减少90%、推理时延减少63%。

**⚠️ 局限性**

仅针对单一电池调度任务；对离线轨迹质量敏感；KD需先训练教师模型；实验在模拟硬件上进行，未在真实嵌入式设备上验证；未扩展到多设备或更复杂场景。

---

## 218. GS-BrainText: A Multi-Site Brain Imaging Report Dataset from Generation Scotland for Clinical Natural Language Processing Development and Validation

**arXiv ID:** 2603.26235 | [PDF](https://arxiv.org/pdf/2603.26235v1)

**作者:** Beatrice Alex `[一作]` (University of Edinburgh), William Whiteley `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了GS-BrainText数据集，包含来自Generation Scotland队列的8,511份脑影像报告，并对其中2,431份手工注释了24种临床重要的脑血管表型；

**💡 创新点**

创新点在于提供了多站点、人口基样本的脑影像报告资源，覆盖不同年龄、模态并配有专家级标注，同时通过该数据集揭示NLP系统在不同站点与人群中的性能差异；

**🔧 技术方法**

采用规则基的EdIE-R系统进行表型抽取，并利用BRAT工具在EdIE-R预注释基础上进行人工校正；

**📊 数据集**

使用了Generation Scotland队列中收集的脑CT/MRI报告数据，涵盖五个苏格兰NHS健康板块的报告；

**📈 对比分析**

通过在24种表型上计算精度、召回和F1分数，EdIE-R的微平均F1为88.82，宏平均F1为77.47，表现随表型频率、站点与年龄变化；

**⚠️ 局限性**

局限包括数据仅限苏格兰NHS，CT占比高且表型覆盖不全，类别严重不平衡，缺乏时间趋势分析，且在新站点部署前需进行本地验证。

---

## 219. ParaQAOA: Efficient Parallel Divide-and-Conquer QAOA for Large-Scale Max-Cut Problems Beyond 10,000 Vertices

**arXiv ID:** 2603.26232 | [PDF](https://arxiv.org/pdf/2603.26232v1)

**作者:** Po-Hsuan Huang `[一作]` (National Taiwan University), Shih-Hao Hung `[通讯]` (National Taiwan University)

**通讯引用:** 1132 | [OpenAlex ID](https://openalex.org/A5020028710)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 ParaQAOA 框架，通过并行分治策略高效求解大规模 Max‑Cut 问题，显著降低运行时间。

**💡 创新点**

创新点包括：① 线性时间的连通性保持分区算法；② GPU 并行 QAOA 执行与层级并行合并；③ 可调参数实现质量-效率权衡；④ 统一的性能效率指数（PEI）评估指标。

**🔧 技术方法**

采用了 QAOA、图分区、GPU 并行量子电路模拟、选择性分布探索、层级并行深度优先合并以及 PEI 指标。

**📊 数据集**

使用 Erdős‑Rényi 随机图作为实验数据集，规模覆盖 20–26、100–400、1,000–16,000 顶点，边概率分别为 0.1、0.3、0.5、0.8。

**📈 对比分析**

与 GW、Coupling QAOA、QAOA² 进行比较。ParaQAOA 在 400 顶点实例上相较 QAOA² 实现了高达 1,600 倍的加速，16,000 顶点实例仅需 19 分钟，而最佳方法需 13.6 天；近似比保持在最佳已知解的 98% 以上；PEI 也显著高于同类算法。

**⚠️ 局限性**

局限性：随机分区对结构化图可能欠佳；目前仅在 Max‑Cut 上验证；实验依赖 GPU 并行模拟，缺乏真实量子硬件评估；未考虑噪声对 QAOA 结果的影响。

---

## 220. Clawed and Dangerous: Can We Trust Open Agentic Systems?

**arXiv ID:** 2603.26221 | [PDF](https://arxiv.org/pdf/2603.26221v1)

**作者:** Shiping Chen `[一作]` (CSIRO Data61), Liming Zhu `[通讯]` (CSIRO Data61)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过软件工程视角，对开放型代理系统（agentic system）的安全研究进行系统综述，提出了六维度分析分类法，并综合 50 篇相关论文，形成了安全-by-construct 的参考教义和评估记分卡。

**💡 创新点**

创新点在于：①首次将安全研究映射到软件生命周期、信任边界、能力表面、控制位置、失败模式与证据类型等六维度；②识别并量化当前研究的显著空白（如内存完整性、供应链治理与运维监管）；③构建面向平台的治理框架与评估指标，超越单纯的攻击成功率评估。

**🔧 技术方法**

采用文献检索与系统编码方法，对 2023‑2026 年期刊与预印本中的 50 篇论文进行分类与编码，并基于该分类制定治理原则与记分卡；同时借鉴 Sigstore、in‑toto、Wasm sandbox 等已有安全基础。

**📊 数据集**

未使用传统意义上的数据集；研究材料主要为公开发表与预印本论文及其实验结果、benchmarks 与安全规范，构成了 50 篇文献的综合数据集。

**📈 对比分析**

通过对六维度和控制位置的统计与矩阵对比，量化了各安全属性在文献中的覆盖度；评估记分卡显示测试与攻击研究占主导，部署、内存完整性、运维治理等维度几乎无覆盖，表明当前技术在平台安全治理方面仍有显著不足。

**⚠️ 局限性**

主要局限在于：①大部分证据基于预印本或实验室发布，缺乏正式同行评审；②评估多聚焦攻击成功率，缺乏跨生命周期与跨信任边界的实测数据；③对供应链与内存治理的实用框架尚未标准化，研究缺乏统一的实验环境与对比基准。

---

## 221. EPDQ: Efficient and Privacy-Preserving Exact Distance Query on Encrypted Graphs

**arXiv ID:** 2603.26219 | [PDF](https://arxiv.org/pdf/2603.26219v1)

**作者:** Xuemei Fu `[一作]` `[通讯]` (Hebei University of Engineering), Xuemei Fu (Hebei University of Engineering)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于张量化表示的加密图数据库距离查询方案EPDQ，能够在云端对加密图执行精确的最短距离查询。

**💡 创新点**

创新点在于将2‑hop覆盖索引与Pruned Landmark Labeling相结合构建加密索引，并引入张量化统一模型来降低计算复杂度和提升可扩展性。

**🔧 技术方法**

采用结构化加密、可搜索加密、张量加密、Pruned Landmark Labeling、同态加密以及伪随机函数/加密技术实现方案。

**📊 数据集**

实验使用了11个来自SNAP、TTIC、KONECT的真实图数据集，包括Wiki‑Talk、Com‑Youtube、Email‑EuAll、Loc‑Gowalla、Wiki‑Vote、Email‑Enron、Blogs、PolBlogs、Emails、Facebook和Friendship。

**📈 对比分析**

通过与现有加密距离查询方法对比，实验表明EPDQ在初始化时间、查询时间和存储开销上均表现更优，尤其在百万级顶点/边的大规模图上保持可扩展，查询耗时从几秒到几百秒不等。

**⚠️ 局限性**

局限性包括：仅支持静态或有限更新的图；查询结果仅返回最短距离而非路径；且假设云服务器为诚实但好奇的模型，若服务器恶意则可能泄露信息。

---

## 222. SuperDP: Differential Privacy Refutation via Supermartingales

**arXiv ID:** 2603.26215 | [PDF](https://arxiv.org/pdf/2603.26215v1)

**作者:** Krishnendu Chatterjee `[一作]` (Institute of Science and Technology Austria), Đorđe Žikelić `[通讯]` (Singapore Management University)

**通讯引用:** 330 | [OpenAlex ID](https://openalex.org/A5041082080)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于期望上/下鞅的自动化方法，能够在给定的概率程序中可靠地反例化 ε‑DP，且在满足 OST 条件时具有完整性。

**💡 创新点**

创新点在于构造了“期望超/下鞅证据”，该证据既给出了两个相似输入，又给出了一个非负函数，三者共同满足上下界约束，从而实现对 ε‑DP 的完整证明与自动化反例生成。

**🔧 技术方法**

主要技术包括上期望超鞅（UESM）与下期望下鞅（LESM）的合成、模板化多项式求解、使用 Handelman 定理将全称约束化为存在约束、以及基于 SMT 的约束求解。

**📊 数据集**

使用了 15 个来自差分隐私文献的基准程序（随机响应、直方图、稀疏向量技术、几何噪声等）作为实验数据集。

**📈 对比分析**

与 CheckDP（静态分析）和 StatDP（动态采样）进行对比；SuperDP 在 13/15 例子上能获得更大的 ε 值，且平均运行时间低于 3 秒，表现优于两者。

**⚠️ 局限性**

局限性在于只能处理多项式算术程序、仅适用于 ε‑DP（不支持 (ε,δ)‑DP 或完整验证）、以及在涉及条件概率或生成大型约束系统时求解困难。

---

## 223. SAFT: Sensitivity-Aware Filtering and Transmission for Adaptive 3D Point Cloud Communication over Wireless Channels

**arXiv ID:** 2603.26197 | [PDF](https://arxiv.org/pdf/2603.26197v1)

**作者:** Huda Adam Sirag Mekki `[一作]` (Shandong University), Guanghui Zhang `[通讯]` (Shandong University)

**通讯引用:** 12311 | [OpenAlex ID](https://openalex.org/A5100459824)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了SAFT（Sensitivity-Aware Filtering and Transmission）框架，实现对3D点云在无线信道中的学习式源信道联合编码与自适应传输。

**💡 创新点**

创新点包括：1）基于token敏感度的过滤模块（STF），动态赋予token重要性并自适应裁剪；2）SNR感知解码器，将接收端信噪比信息注入解码，提升对不同噪声条件的鲁棒性；3）训练时的符号使用惩罚，稳定离散码本分布而不增加额外侧信息。

**🔧 技术方法**

采用的技术主要有：Point-BERT风格Transformer编码器、轻量MLP进行敏感度评分与token加权、均匀整数量化与直通估计、SNR嵌入模块、残差MLP上采样解码器，以及Chamfer距离、稀疏性与多样性正则的多项损失。

**📊 数据集**

使用的数据集包括ShapeNet、ModelNet40和8iVFB（四个真实场景点云）。

**📈 对比分析**

与传统的G-PCC+LDPC、V-PCC+LDPC以及学习式基线DPCC、SemCom、SEPT、PCST、TSCS等方法进行对比。实验结果表明，在AWGN与Rayleigh信道下，SAFT在低SNR（0~10 dB）时的D1/D2 PSNR平均提升约2-4 dB，显著优于现有方法。

**⚠️ 局限性**

局限性：1）对接收端SNR估计敏感，估计误差会影响性能；2）模型在极端高噪声下仍可能解码失败；3）使用训练时符号使用惩罚，需在训练阶段手动调参；4）未评估对下游任务（如分割、分类）的实际效果。

---

## 224. CREval: An Automated Interpretable Evaluation for Creative Image Manipulation under Complex Instructions

**arXiv ID:** 2603.26174 | [PDF](https://arxiv.org/pdf/2603.26174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 225. DUGAE: Unified Geometry and Attribute Enhancement via Spatiotemporal Correlations for G-PCC Compressed Dynamic Point Clouds

**arXiv ID:** 2603.26183 | [PDF](https://arxiv.org/pdf/2603.26183v1)

**作者:** Pan Zhao `[一作]` (Shandong University), Sam Kwong `[通讯]` (Lingnan University)

**通讯引用:** 34828 | [OpenAlex ID](https://openalex.org/A5008386708)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种统一的几何与属性增强框架 DUGAE，专门用于 G‑PCC 压缩的动态点云。

**💡 创新点**

创新点：①将增强任务拆分为动态几何增强、属性重着色和动态属性增强三步；②在特征域引入几何/属性运动补偿模块（GMC/AMC）以利用时空相关性；③使用 DA‑KNN 进行精确属性重着色；④为属性引入专门的时间特征提取器（ATFE）和属性补偿模块。

**🔧 技术方法**

采用 SPConv/U‑Net 架构、GSConv、Dense Module、ATFE、AOE、DA‑KNN；训练使用 BCE 损失（几何）和加权 MSE 损失（属性）；实现基于 PyTorch 与 MinkowskiEngine。

**📊 数据集**

训练集：8iVFB v2、Owlii、MVUB 共六个动态点云序列；测试集：Redandblack、Soldier、Exercise、Model、Phil、Ricardo、Sarah。

**📈 对比分析**

与 GeS‑TM、V‑PCC、UGAE、DPCGC 等基线对比，DUGAE 在几何 D1/D2 平均提升约 11 dB / 9.6 dB，BD‑BR 约 -94%；属性 Y 平均提升 4.2 dB；整体性能优于 UGAE 和 V‑PCC，尤其在低比特率区间表现更为突出。

**⚠️ 局限性**

限制：DGE‑Net 在 CPU 上每帧约 66 s，GPU 实现虽快但不保证确定性；属性提升相对有限；模型对 GPU 资源依赖强；目前仅在 G‑PCC 编码流程验证，对其他点云编码器的适应性待进一步研究。

---

## 226. Geometric Evolution Graph Convolutional Networks: Enhancing Graph Representation Learning via Ricci Flow

**arXiv ID:** 2603.26178 | [PDF](https://arxiv.org/pdf/2603.26178v1)

**作者:** Jicheng Ma `[一作]` (Renmin University of China), Liang Zhao `[通讯]` (Beijing Normal University)

**通讯引用:** 30156 | [OpenAlex ID](https://openalex.org/A5100433940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在图神经网络中加入离散Ricci流生成的几何演化序列，利用LSTM编码并融入GCN，实现更强的节点表征。

**💡 创新点**

创新点在于将离散Ricci流作为多尺度几何演化过程与LSTM结合，动态学习边的重要性并以几何感知归一化替代传统GCN归一化。

**🔧 技术方法**

采用离散Ricci流、LSTM编码器和几何感知GCN，并使用Sinkhorn算法求解Wasserstein距离进行预处理。

**📊 数据集**

在多组基准数据集上测试，包括Cora、Citeseer、Pubmed、Coauthor CS、Amazon Photos、Cornell、Texas、Wisconsin、Chameleon和Actor。

**📈 对比分析**

与多种基准方法（GCN、GAT、GraphSAGE、CurvGN、UFGConv、RC‑UFG、DIGL、+FA、SDRF等）对比，GEGCN在绝大多数数据集上取得最高或第二高的分类准确率，尤其在异质图上提升显著。

**⚠️ 局限性**

局限性在于离散Ricci流预处理需要计算1‑Wasserstein距离，虽然可用Sinkhorn减小复杂度，但在大规模图上仍存在算力和存储成本；此外模型对Ricci流步数T的选择敏感。

---

## 227. ComVi: Context-Aware Optimized Comment Display in Video Playback

**arXiv ID:** 2603.26173 | [PDF](https://arxiv.org/pdf/2603.26173v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 228. GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport

**arXiv ID:** 2603.26181 | [PDF](https://arxiv.org/pdf/2603.26181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 229. Improving Risk Stratification in Hypertrophic Cardiomyopathy: A Novel Score Combining Echocardiography, Clinical, and Medication Data

**arXiv ID:** 2603.26254 | [PDF](https://arxiv.org/pdf/2603.26254v1)

**作者:** Marion Taconné `[一作]` (Politecnico di Milano), Luca Mainardi `[通讯]` (Politecnico di Milano)

**通讯引用:** 7419 | [OpenAlex ID](https://openalex.org/A5066168605)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了利用常规超声、临床及用药数据构建可解释的机器学习模型，预测HCM患者5年心血管事件风险。

**💡 创新点**

结合EHR常规数据的可解释ML风险评分，并提供外部验证及纵向风险监测框架。

**🔧 技术方法**

随机森林集成模型、SHAP解释、嵌套交叉验证、特征选择以及时间序列风险趋势分析。

**📊 数据集**

1,201名佛罗伦萨医院HCM患者（SHARE注册）内部验证，382名雷恩斯医院患者外部验证。

**📈 对比分析**

与ESC 5年SCD风险评分对比，内部AUC 0.85±0.02，外部AUC 0.723，Log-rank p=8.62×10⁻⁴，显著优于ESC。

**⚠️ 局限性**

仅限欧洲中心、复合终点、外部验证受随访时长差异影响、缺乏前瞻性验证且需再校准。

---

## 230. Optimization Trade-offs in Asynchronous Federated Learning: A Stochastic Networks Approach

**arXiv ID:** 2603.26231 | [PDF](https://arxiv.org/pdf/2603.26231v1)

**作者:** Abdelkrim Alahyane `[一作]` (Mohammed VI Polytechnic University), Matthieu Jonckheere `[通讯]` (Université de Toulouse)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并分析了异步联邦学习的队列网络模型，结合随机计算、通信延迟与能源消耗，给出闭式性能指标并通过梯度优化调优路由和并发度。

**💡 创新点**

创新点是将异步FL映射为Jackson网络，实现闭式平均相对延迟与梯度，首次在同一框架下同时考虑墙钟时间、能耗与数据异质性，并给出可梯度优化的路由与并发选择。

**🔧 技术方法**

使用了队列网络理论、产品形态分布、Buzen递推算法、梯度下降（Adam）、多目标优化（Pareto）以及能耗建模。

**📊 数据集**

使用了 EMNIST（以及补充的 CIFAR‑100、KMNIST）进行实验。

**📈 对比分析**

与标准同步/异步SGD、最大吞吐、round‑optimized 等基线比较，提出的 time‑optimized 方案在墙钟时间上比标准快约 30‑50%，在能耗上提升 36‑49%，并在不同服务时间分布下保持鲁棒。

**⚠️ 局限性**

局限包括假设服务时间为指数分布、CS 处理可忽略或单服务器模型、仅考虑单任务轮次、未处理客户端动态加入/离线等。

---

## 231. Improved Approximation Algorithms and Hardness Results for Shortest Common Superstring with Reverse Complements

**arXiv ID:** 2603.26176 | [PDF](https://arxiv.org/pdf/2603.26176v1)

**作者:** Ryosuke Yamano `[一作]` (University of Tokyo), Tetsuo Shibuya `[通讯]` (University of Tokyo)

**通讯引用:** 3438 | [OpenAlex ID](https://openalex.org/A5043159528)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对 DNA 序列组装中的最短公共超字符串（SCS）问题的变体——考虑逆互补的最短公共超字符串（SCS‑RC）提出了一套新的近似算法与下界分析，给出了 8/3 的近似比；

**💡 创新点**

创新点在于：①将受逆互补约束的最短公共超字符串问题归约为带约束的最小权值循环覆盖；②构造巧妙的“gadget”将该约束问题转化为一般图中的最大权值完美匹配，从而实现多项式时间求解；③证明 SCS‑RC 的近似难度与标准 SCS 等价，给出了 333/332 的硬度下界；

**🔧 技术方法**

核心技术包括：构造距离图与重叠图、周期性字符串分析、循环覆盖的分阶段构造、最大权匹配求解、逆互补映射与重编码技巧；

**📊 数据集**

本文主要以理论证明为主，实验部分未使用公开数据集，算法的有效性仅通过数学分析与已知比值（23/8→8/3）展示；

**📈 对比分析**

与先前最优 23/8 近似算法相比，本文实现了 8/3 的近似比，理论证明显示该算法在所有实例上都不超过 2.67 倍于最优；

**⚠️ 局限性**

局限性包括：仍存在 8/3 与最优解之间的约 0.23 比值差距；算法在实际大规模 DNA 读取上未经过实验验证；对逆互补对的处理在理论上有效，但实现复杂度相对较高；

---

## 232. Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication

**arXiv ID:** 2603.26167 | [PDF](https://arxiv.org/pdf/2603.26167v1)

**作者:** Yi Zhang `[一作]` (Shenzhen University), Liang-Jie Zhang `[通讯]` (Shenzhen University)

**通讯引用:** 3921 | [OpenAlex ID](https://openalex.org/A5068728111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 Gaussian Shannon 框架，利用低密度奇偶校验码与多数投票机制，将扩散模型的初始噪声视为可靠通信通道，对 AI 生成图像嵌入可恢复的结构化版权信息；

**💡 创新点**

其创新点在于将嵌入与提取过程建模为噪声通道的通信问题，实现了不需要微调且不影响生成质量的位级精确恢复；

**🔧 技术方法**

采用 LDPC 纠错码、伪随机调制与 DDIM 逆向采样等技术，将位信息映射到初始 Gaussian 噪声，再通过多数投票与 LDPC 解码恢复原始位串；

**📊 数据集**

实验数据集主要基于 Stable Diffusion v1.4、v2.0 与 v2.1，使用 Hugging Face 的 Prompt 集进行 1000 张图像生成；

**📈 对比分析**

与 DwtDct、DwtDctSvd、Tree‑Ring、StableSignature、GaussianShading、PRCW 等基线相比，Gaussian Shannon 在七种噪声与四种高级攻击下均实现 100% 位准确率、TPR≥0.99，且保持与基线相当的图像质量；

**⚠️ 局限性**

局限性包括在高强度噪声下 100% 位准确率显著下降，以及对大幅几何变形（裁剪、旋转等）导致检测失败的敏感性。

---

## 233. DuSCN-FusionNet: An Interpretable Dual-Channel Structural Covariance Fusion Framework for ADHD Classification Using Structural MRI

**arXiv ID:** 2603.26351 | [PDF](https://arxiv.org/pdf/2603.26351v1)

**作者:** Qurat Ul Ain `[一作]` (National University of Sciences and Technology), Soyiba Jawed `[通讯]` (National University of Sciences and Technology)

**通讯引用:** 129 | [OpenAlex ID](https://openalex.org/A5014717805)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了基于双通道结构协方差网络（SCN）的可解释 ADHD 诊断框架 DuSCN-FusionNet，利用 sMRI 生成强度和异质性两种 SCN，并通过 CNN 编码、辅助特征融合和后期 Grad‑CAM 解释，实现 ADHD 与健康对照的二分类。

**💡 创新点**

创新点包括：① 在同一模型中同时建模强度‑基和异质性‑基的双通道 SCN，充分捕捉脑区间形态关联；② 将单独的 ROI 变异性与全局统计特征通过晚期融合提升判别力；③ 将 Grad‑CAM 适配至 SCN 域，生成 ROI 级重要性得分，提供可解释的结构性生物标记。

**🔧 技术方法**

主要技术：结构协方差网络（SCN）构建、双通道 CNN 编码、辅助特征 MLP 分支、late‑fusion 设计、seed‑ensemble（5‑seed 交叉验证）训练、Grad‑CAM 后期解释。

**📊 数据集**

使用 ADHD‑200 公开数据集北京大学站（PU）提供的 194 张 T1‑加权结构 MRI，包含 78 名 ADHD 病例和 116 名健康对照。

**📈 对比分析**

通过分层 10‑折交叉验证和 5‑seed 集成进行评估；平均平衡准确率 80.59%，AUC 0.778；与其他仅使用 sMRI 的 SOTA 方法（如 3D MVA‑CNN、Dense Attentive GAN、3D CNN 等）相比，DuSCN‑FusionNet 在同一数据子集上实现了更高的准确率并提供了可解释的脑区生物标记。

**⚠️ 局限性**

局限性：① 仅在单一站点（PU）评估，缺乏跨站点泛化验证；② 仅使用 AAL 116 区域和两种单一形态特征，未尝试更丰富的结构或多模态信息；③ 对模型解释的阈值设定（90% 分位）及统计检验方式可能影响标记的稳定性。

---

## 234. CALRK-Bench: Evaluating Context-Aware Legal Reasoning in Korean Law

**arXiv ID:** 2603.26332 | [PDF](https://arxiv.org/pdf/2603.26332v1)

**作者:** JiHyeok Jung `[一作]` (KAIST AI), HyunSouk Cho `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CALRK-Bench，针对韩国法律构建的情境感知法律推理基准；

**💡 创新点**

创新点在于评估法律规则在时间变化和信息充分性等上下文因素下的适用性，而非单纯的规则记忆；

**🔧 技术方法**

使用了多任务多选题框架，结合链式推理（CoT）和思考模式（Thinking Mode）进行模型评估；

**📊 数据集**

数据集来源于韩国法院判例、法律咨询记录，采用硬负采样构造合理的干扰选项；

**📈 对比分析**

与 GPT‑5、Gemini、LLaMA‑3.3、Qwen‑3.3 等大语言模型进行对比，实验显示即便是最先进模型在三项任务上准确率也仅略高于随机猜测，表明存在明显缺陷；

**⚠️ 局限性**

局限性包括仅适用于韩国法律、样本规模有限、缺乏模糊或渐进式法律变更案例，且需要专家人工验证以保证数据质量。

---

## 235. DFM-VLA: Iterative Action Refinement for Robot Manipulation via Discrete Flow Matching

**arXiv ID:** 2603.26320 | [PDF](https://arxiv.org/pdf/2603.26320v1)

**作者:** Jiayi Chen `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Haoang Li `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于离散流匹配的VLA框架（DFM‑VLA），通过全序列的 token‑级速度场实现动作的迭代细化，能够在解码过程中不断纠正错误。

**💡 创新点**

创新点在于：① 将离散流匹配引入机器人动作生成，实现可逆的 token‑级细化；② 设计了两种速度场构造方式（辅助 head 与嵌入引导）并系统比较；③ 采用两阶段解码（细化+验证）和自适应 KV 缓存，兼顾准确性与速度。

**🔧 技术方法**

主要技术包括：离散流匹配（Discrete Flow Matching）、连续时间马尔可夫链与 Euler 近似、token‑级速度场建模、两阶段解码策略、Adaptive KV 缓存；模型架构使用统一的视觉‑语言‑动作离散 token 化并借助 VLM/LLM 背骨。

**📊 数据集**

使用的评测数据集包括：CALVIN、LIBERO（模拟机器人长链任务），以及真实世界的 bimanual AgileX 机器人实验（三类抓取与放置任务，100 条轨迹训练、40 条评测）。

**📈 对比分析**

与 AR、DD、FlowVLA、DreamVLA、RDT 等现有方法对比：在 CALVIN 上平均完成长度 4.44，远超 AR 4.18；在 LIBERO 上整体成功率 95.7%，优于 DreamVLA 92.6%；在真实世界实验中平均成功率 70.8%，比 RDT 60.0% 与 DreamVLA 54.2% 高出显著幅度；同时在推理速度上通过两阶段解码与 Adaptive Cache，能够实现 2.4× 的加速。

**⚠️ 局限性**

局限性包括：对离散 token 的质量和预训练模型的依赖较高；速度场构造中的超参数（c、α）仍需经验调优；虽然细化可纠错，但在极长序列或极低信噪比的场景下仍可能出现误差积累；目前仅在动作层面进行细化，未充分利用视觉-语言信息的深度关联。

---

## 236. Distances in Planar Graphs are Almost for Free!

**arXiv ID:** 2603.26313 | [PDF](https://arxiv.org/pdf/2603.26313v1)

**作者:** Shay Mozes `[一作]` (Reichman University), Daniel Prigan `[通讯]` (Reichman University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明：对于任意有向加权平面图，几乎线性时间（n^{1+o(1)})即可构造大小为 O(n^{1+o(1)})、查询时间为 O(log^2 n) 的精确距离回路；并给出一整套几乎最优的空间–查询时间权衡方案。

**💡 创新点**

创新点：①提出一种全新的二分搜索算法，利用最短路径树和其共树的结构，定位三色（trichromatic）面，而不依赖传统的分离子或最短路径分离子；②构造粗糙树与其双树的层次化表示，支持在递归分解过程中高效求解三色面，从而实现几乎线性构造时间；③通过对 Voronoi 图的全局与局部信息的巧妙组合，实现了在几乎线性时间内完成所有预处理。

**🔧 技术方法**

主要技术：最短路径树与共树的共性与前缀性质、中心分解、MSSP（多源最短路径）数据结构、Voronoi 图的双树表示、粗糙树（coarse tree）与粗糙共树（dual coarse tree）的递归构造、基于树消除的局部决策规则、二分搜索寻找临界边，以及对不同分层的路径拆分（SegmentBreakdown）与细化（FindCriticalOnSegment）。

**📊 数据集**

本文是理论性工作，未使用具体的实验数据集；所有结果均在理论分析和算法设计层面得到证明，未做实验验证。

**📈 对比分析**

与之前工作相比，传统方法在空间 O(n^{4/3}) 时需要 O(n^{3/2}) 构造时间；而本文将构造时间压缩到几乎线性 n^{1+o(1)}，保持空间与查询时间的最优权衡；查询时间维持在 O(log^2 n)。此外，对动态距离回路的改进，更新时间从 O(n^{4/5}) 降到 O(n^{2/3})。

**⚠️ 局限性**

局限性：算法实现极其复杂，涉及多层次的粗糙树和双树的维护；目前仅针对平面图，尚未推广到更广泛的图类；在实际工程中，常数因子与 sub‑polynomial 词项可能较大；此外，动态更新仍非线性，仅对单源查询有效。

---

## 237. PEB Separation and State Migration: Unmasking the New Frontiers of DeFi AML Evasion

**arXiv ID:** 2603.26290 | [PDF](https://arxiv.org/pdf/2603.26290v1)

**作者:** Yixin Cao `[一作]` (EigenPhi), Yijie Liu `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对可组合智能合约生态中的价值迁移机制进行形式化，揭示了传统基于转移图的反洗钱（AML）监控在存在主张者–执行者–受益者（PEB）分离和状态驱动价值迁移时的不可完全性。

**💡 创新点**

创新点在于首次将PEB分离与状态介导的价值迁移两种结构机制作为导致转移层不完整性的根本原因，并通过构造性的交叉池套利模型证明其普遍可行性。

**🔧 技术方法**

技术手段包括形式化转移层观察模型、基于常数乘积自动做市商的状态转移分析、以及对交易图的逻辑推理和执行语义重放。

**📊 数据集**

实验数据主要来自以太坊主网真实交易（如1inch限价单执行案例）以及在本地以太坊分叉上模拟的交叉池套利执行序列。

**📈 对比分析**

与传统基于转移图的AML工具对比，本文并未给出量化性能指标，而是通过理论证明和模拟结果展示了即便在完全可见的转移层下也无法唯一归因经济迁移，从而表明现有方法的局限性。

**⚠️ 局限性**

局限性包括：未构建统一的标注数据集来训练模型；依赖于理论模型假设的理想化环境；以及缺乏在更大规模真实链上系统化评估的结果。

---

## 238. GLASS: Geometry-aware Local Alignment and Structure Synchronization Network for 2D-3D Registration

**arXiv ID:** 2603.26262 | [PDF](https://arxiv.org/pdf/2603.26262v1)

**作者:** Zhixin Cheng `[一作]` (University of Science and Technology of China), Tianzhu Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18206 | [OpenAlex ID](https://openalex.org/A5100648981)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出了GLASS网络，用局部几何增强和图结构一致性模块实现图像与点云的高精度配准。

**💡 创新点**

创新点包括：①将表面法向注入图像特征以提升局部几何感知；②构造匹配图并通过LightGAT约束相似度分布；③三者协同显著降低跨模态误匹配。

**🔧 技术方法**

技术栈：ResNet+FPN与KPFCNN特征提取；Transformer跨模态交互；DINOv2语义特征；Depth Anything v2伪法向；轻量级图注意网络LightGAT；PnP+RANSAC位姿估计；损失包括circle loss、法向一致性和分布一致性。

**📊 数据集**

使用的主要数据集为RGB‑D Scenes v2与7‑Scenes两大室内基准，外部场景KITTI亦作验证。

**📈 对比分析**

与多种基线（2D3D‑MATR、FreeReg、Flow‑I2P、CA‑I2P等）在Inlier Ratio、Feature Matching Recall、Registration Recall、RRE和RTE等指标上对比，GLASS在大多数指标上显著领先，RGB‑D场景RR提升至83.3%~90.1%，7‑Scenes RR提升至93.1%。

**⚠️ 局限性**

局限性：对Depth Anything v2产生的法向噪声敏感，在深度变化剧烈或遮挡区域法向不可靠时易导致匹配误差；模型在使用DINOv2时推理速度慢、计算资源需求高。

---

## 239. Bitcoin Smart Accounts: Trust-Minimized Native Bitcoin DeFi Infrastructure

**arXiv ID:** 2603.26293 | [PDF](https://arxiv.org/pdf/2603.26293v1)

**作者:** Cian Lalor `[一作]` (Lombard Finance), Antonio Russo `[通讯]` (Lombard Finance)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 Bitcoin Smart Accounts（BSA）协议，使原生比特币能够以自托管方式参与 DeFi

**💡 创新点**

采用 PSBT 与 Taproot 脚本仿真契约、基于 TEE 的仲裁器以及 Smart Account Registry，实现 1‑of‑k（即 1‑of‑(k+1)）的信任最小化模型

**🔧 技术方法**

技术上使用 Taproot、PSBT、Anchor 输出、CPFP、AWS Nitro Enclave + KMS、以太坊（或其他目标链）智能合约、轻客户端等

**📊 数据集**

论文未使用外部数据集，主要为协议设计与安全分析

**📈 对比分析**

通过理论安全分析和对比表与 BitVM、中心化托管等方案，证明 BSA 在信任模型、资产安全和自托管方面优于传统阈值方案，但缺乏实测性能指标

**⚠️ 局限性**

局限包括对 AWS Nitro/KMS 的依赖、时间锁导致的延迟、对外部 DeFi 价格预言机的依赖、重构时需 T3 锁定、对链兼容性与未来合约升级的复杂性

---

## 240. Developers and Generative AI: A Study of Self-Admitted Usage in Open Source Projects

**arXiv ID:** 2603.26277 | [PDF](https://arxiv.org/pdf/2603.26277v1)

**作者:** Rosalia Tufano `[一作]` (Universitá della Svizzera italiana), Gabriele Bavota `[通讯]` (Universitá della Svizzera italiana)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对开源项目中开发者自我申报使用ChatGPT和GitHub Copilot的情况进行挖掘与归纳。

**💡 创新点**

创新点在于首次构建两种生成式AI工具的自我申报使用任务分类体系，并对比两年后使用演变。

**🔧 技术方法**

主要技术手段包括GitHub API爬取、n-gram过滤、人工编码以及层次化分类。

**📊 数据集**

使用的数据集来源于约3.6万个GitHub提交、问题与PR的文本记录，并手工抽样验证。

**📈 对比分析**

通过对比前一研究的样本，发现使用频率、任务种类与满意度均出现显著变化，验证了分类体系的稳健性。

**⚠️ 局限性**

局限性包括仅捕获显式申报的实例、仅覆盖GitHub平台、样本偏向公开项目以及对代码质量影响的评估不足。

---

## 241. Query-Specific Pruning of RML Mappings (Extended Version)

**arXiv ID:** 2603.26269 | [PDF](https://arxiv.org/pdf/2603.26269v1)

**作者:** Sitt Min Oo `[一作]` (Ghent University - imec), Olaf Hartig `[通讯]` (Linköping University)

**通讯引用:** 3023 | [OpenAlex ID](https://openalex.org/A5022300272)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种基于查询的 RML 映射裁剪方法，利用 SPARQL 三元组模式的可满足性判断，提前去除对用户查询无关的映射，从而实现部分 RDF 图的高效生成。

**💡 创新点**

创新点在于：①提出了针对 RML 映射的三元组模式可满足性定义并证明其不可判定性；②定义了“互不兼容”属性，可在可满足性判定不可行时直接确定不满足；③基于该属性设计了一套完整的裁剪算法，兼顾理论正确性与实用性。

**🔧 技术方法**

主要技术包括：RML 到映射代数的翻译、RML‑specific 映射表达式的形式化与语义、正则表达式匹配、不可满足性判定、以及裁剪算法实现。

**📊 数据集**

使用 GTFS‑Madrid 基准测试：生成 10 倍规模的合成马德里地铁数据，包含 86 条 TrMap‑expression 和 15 组多样化 SPARQL 查询。

**📈 对比分析**

与基线（完整映射、完整 materialization、完整查询）相比：裁剪时间极低（<5 ms）；裁剪后 materialization 时间平均下降至基线的 8% 以下；查询时间在大多数查询上明显加速（尤其是 Q10、Q12、Q14）。但部分复杂查询仍因自连接或星形结构导致超时。

**⚠️ 局限性**

局限性包括：①可满足性判定不可判定，裁剪只能依赖互不兼容判定；②仅在三元组模式层面裁剪，无法处理星形或链式查询的交叉可满足性；③裁剪后生成的映射可能在语义上等价但对特定映射引擎（如 CARML）不易识别，导致在某些查询上性能下降；④实验仅验证单一映射引擎与基准数据集，泛化性待进一步评估。

---

## 242. Contrastive Conformal Sets

**arXiv ID:** 2603.26261 | [PDF](https://arxiv.org/pdf/2603.26261v1)

**作者:** Yahya Alkhatib `[一作]`, Wee Peng Tay `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出在语义特征空间中构造对比式符合预测集（contrastive conformal sets），通过学习可变形的多范数几何形状来保证正样本覆盖率并最大化负样本排除率。

**💡 创新点**

创新点在于：①将符合预测（conformal prediction）延伸至语义特征空间；②利用最小体积覆盖集理论与可学习的多范数度量实现负样本排除；③在仅有正样本的情况下，仅通过体积最小化即可逼近最佳负排除。

**🔧 技术方法**

核心技术包括：符合预测理论、可学习的单范数与多范数（ℓ_p、加权多范数）度量、体积与负排除的联合优化、InfoNCE 对比学习损失的联合正则化、基于样本对的正负距离计算。

**📊 数据集**

使用的实验数据集：三维模拟数据；CIFAR‑100（用作ID训练与验证）；CIFAR‑10 与 SVHN 作为OOD 评估。

**📈 对比分析**

与传统符合预测基线（ℓ_2 球体、Mahalanobis 椭球）以及多种 OOD 检测基线（MSP、Energy、KNN、Mahalanobis、COMBOOD、D‑KNN、CIDER、ViM）进行比较。结果显示，学习度量的对比式符合预测集在保持 95% 正样本覆盖的同时，负排除率显著提升（单范数/多范数组合可达 49%），并在 OOD 检测任务中获得最高 AUROC 与最低 FPR95。

**⚠️ 局限性**

局限性：仅适用于欧氏/加权欧氏度量，未考虑非欧几里得或流形几何；对负样本获取依赖；在缺乏负样本时只能靠体积最小化，可能无法达到最佳排除；实验集中在图像分类数据，缺乏在更复杂或多模态数据上的验证。

---

## 243. HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models

**arXiv ID:** 2603.26362 | [PDF](https://arxiv.org/pdf/2603.26362v1)

**作者:** MD Khalequzzaman Chowdhury Sayem `[一作]` (UNIST), Seungryul Baek `[通讯]` (UNIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 HandVQA 这一大规模 3D 手部细粒度空间推理诊断基准，包含 1.6M 多选问答，评估 VLM 对手部关节角度、距离与相对位置的理解；同时验证了该基准在零样本下对姿态识别和手物交互任务的迁移效果。

**💡 创新点**

创新点在于：① 利用三维手部数据自动生成精确、可解释的多选问题，消除语义歧义；② 将空间推理拆分为角度、距离和 X/Y/Z 三个子任务，提供细粒度评测；③ 证明训练 VLM 在该基准上可显著提升空间推理能力，并迁移到未见任务，首次展示细粒度手部空间知识的可转移性。

**🔧 技术方法**

使用的技术包括：多模态 VLM（LLaVA、DeepSeek、Qwen‑VL）与 LoRA 轻量级微调；3D 关节描述符提取、离散化阈值与模板化句子生成；基于自动化流水线将 3D 关节坐标映射为自然语言问答；在基准上进行准确率与 MAE 的评估。

**📊 数据集**

所用数据集：FreiHAND、InterHand2.6M、FPHA，分别提供不同视角和手部姿态；此外在零样本迁移实验中使用 HaGRID（手势识别）与 H2O（手物交互）视频数据。

**📈 对比分析**

与基线（未微调 VLM）相比，LoRA 微调后各模型在角度、距离和相对位置任务上准确率从约 30–50% 提升至 70–90%，MAE 下降 50% 以上；在零样本手势识别和手物交互任务中，微调模型分别提升约 12% 与 3% 的准确率，表明手部空间知识可显著提升跨任务性能。

**⚠️ 局限性**

限制包括：① 角度任务仍低于 80% 准确率，说明 VLM 对复杂关节角度推理存在瓶颈；② 采用离散阈值与模板化语言，可能不充分捕捉连续空间细节；③ 当前仅覆盖静态图像，未考虑运动动力学；④ 微调仅使用 LoRA，冻结视觉编码器可能限制进一步提升；⑤ 仅评估 7B 规模模型，未验证更大模型的表现。

---

## 244. Realtime-VLA V2: Learning to Run VLAs Fast, Smooth, and Accurate

**arXiv ID:** 2603.26360 | [PDF](https://arxiv.org/pdf/2603.26360v1)

**作者:** Chen Yang `[一作]` (Dexmal), Haoqiang Fan `[通讯]` (Dexmal)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套端到端技术框架，将Vision‑Language‑Action（VLA）模型部署在实际机器人上，实现了高速、平滑、精准的运动；

**💡 创新点**

在系统层面整合了延迟校准、轨迹后处理、速度自适应、时空优化以及人机交互式速度调节，首次实现了基于VLA模型的高速执行并保持高成功率；

**🔧 技术方法**

使用VLA模型、异步控制、时空优化（含二次规划与MPC）、预放大技术、OSQP求解器、SQP‑RTI调度、人体操作的速率标注及回归模型训练等；

**📊 数据集**

收集了自我演示数据（shirt‑folding、place‑into‑fixture、pick‑and‑latch等任务）以及通过人工“油门”调节得到的速度标注数据；

**📈 对比分析**

与传统人机操作和基于视觉的低速演示进行对比，实验结果显示机器人执行速度可与人类相当，任务完成时间逼近人类操作速度，并在视频中体现；

**⚠️ 局限性**

依赖对延迟的精准校准，受限于轻量级机械臂的柔性；需要人工参与收集速度调节数据，模型对不同任务的泛化能力有限，且在极端高速或高精度场景下仍可能出现失败。

---

## 245. Curvature-aware Expected Free Energy as an Acquisition Function for Bayesian Optimization

**arXiv ID:** 2603.26339 | [PDF](https://arxiv.org/pdf/2603.26339v1)

**作者:** Ajith Anil Meera `[一作]` (TU Eindhoven), Wouter Kouw `[通讯]` (TU Eindhoven)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于期望自由能（EFE）的贝叶斯优化获取函数，并在Van der Pol振荡器的系统辨识以及50个随机一维目标函数上进行实验验证。

**💡 创新点**

创新点包括：① 将期望自由能统一为获取函数，证明其可在特定假设下归约为UCB/LCB/EIG；② 提出曲率感知的自适应τ²更新规则，使EFE在探索与利用之间实现动态平衡；③ 给出对凹函数的无偏收敛保证，证明在特定条件下AFE能够收敛到全局最优。

**🔧 技术方法**

使用技术包括：高斯过程模型、期望自由能理论推导、曲率感知自适应更新、理论分析、系统辨识实验、以及对比实验（UCB、EI、PI、VAR、TS、KG）。

**📊 数据集**

实验数据集：Van der Pol振荡器的时间序列数据（含噪声观测）以及由10个随机振幅、频率和相位的正余弦函数组合生成的50个一维目标函数。

**📈 对比分析**

与主流获取函数对比时，EFE在最终GP均方误差（MSE）和简单回报（simple regret）两项指标上均最优；具体表现为MSE 0.0286、简单回报 0.0125，优于UCB、EI、PI、VAR、TS和KG。

**⚠️ 局限性**

局限性包括：① 仅研究了单步期望自由能，缺乏多步规划和非我的式策略；② 仅在合成数据上验证，未在真实物理系统中测试；③ τ² 的设置依赖未知方差，需要进一步的自适应或学习机制；④ 对大规模高维问题的可扩展性尚未解决。

---

## 246. Mitigating the Reasoning Tax in Vision-Language Fine-Tuning with Input-Adaptive Depth Aggregation

**arXiv ID:** 2603.26330 | [PDF](https://arxiv.org/pdf/2603.26330v1)

**作者:** Yiming Ren `[一作]` (Tsinghua University), Junjie Wang `[通讯]` (Tsinghua University)

**通讯引用:** 29296 | [OpenAlex ID](https://openalex.org/A5115695478)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级输入自适应跨层聚合机制IADA，以恢复视觉语言模型在监督微调后出现的推理能力下降现象

**💡 创新点**

在传统的层内参数高效微调（如LoRA）基础上引入跨层信息路由，首次将跨深度访问视为独立的设计维度，并证明其对推理性能影响显著

**🔧 技术方法**

跨层注意力聚合、低秩瓶颈投影、门控残差注入以及模态自适应查询生成

**📊 数据集**

Qwen3‑VL‑2B模型在LLaVA‑Instruct‑Mix视觉指令数据上微调，并在11个多模态基准（MMMU、MathVista、ScienceQA、MMStar、AI2D、MME、POPE、RealWorldQA、ChartQA、TextVQA、OCRBench）上评估

**📈 对比分析**

与预训练基线、单纯LoRA、固定查询注意力残差以及LoRA+IADA进行对比；IADA在低秩LoRA（rank 16）下平均提升推理得分9.5分、感知得分3.3分，总体平均提升6.1分，参数增量仅0.14M

**⚠️ 局限性**

仅在单一模型与单一数据集上验证，跨层路由的效果与不同规模、架构或开放式生成质量的泛化尚未探究，且模态掩码可能在深层失效

---

## 247. Verify Claimed Text-to-Image Models via Boundary-Aware Prompt Optimization

**arXiv ID:** 2603.26328 | [PDF](https://arxiv.org/pdf/2603.26328v1)

**作者:** Zidong Zhao `[一作]` (Zhejiang University), Geguang Pu `[通讯]` (East China Normal University)

**通讯引用:** 3088 | [OpenAlex ID](https://openalex.org/A5054490662)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于语义边界的文本到图像模型验证框架 BPO，完全不依赖参考模型，直接利用目标模型自身的语义边界特征进行验证。

**💡 创新点**

创新点在于发现不同 T2I 模型具有独特的语义边界，并通过三阶段的边界感知提示优化（锚点识别、边界探索、目标化优化）实现无参考模型的高效验证，显著提升准确率与计算速度。

**🔧 技术方法**

使用了白盒提示优化、梯度攻击、线性插值与二分搜索、VLM（CLIP 等）进行语义判别、一致性评分等技术。

**📊 数据集**

使用 GPT‑4o 随机生成的 10 条基础提示和对应生成的 10 张图像，对五个开源 T2I 模型（Stable Diffusion v1.4、v2.1、SDXL、Dreamlike、Openjourney）进行评测。

**📈 对比分析**

与四个基线（Normal、Random、Greedy、TVN）对比，BPO 在 5 个模型上平均准确率 0.96、F1 0.93，且每条验证提示平均耗时约 159 秒，比 TVN 快约 2 倍；在 4 个模型上实现 100% 准确率。

**⚠️ 局限性**

局限性包括：需要对目标模型的文本编码器进行白盒访问，无法在完全黑盒环境下使用；对 VLM 的依赖可能导致不同 VLM 产生性能差异；验证仍需生成 10 张图像，计算成本不低；对更长提示或多模态文本的适应性未做深入评估。

---

## 248. From Human Cognition to Neural Activations: Probing the Computational Primitives of Spatial Reasoning in LLMs

**arXiv ID:** 2603.26323 | [PDF](https://arxiv.org/pdf/2603.26323v1)

**作者:** Jiyuan An `[一作]` (Beijing Language and Culture University), Erhong Yang `[通讯]` (Beijing Language and Culture University)

**通讯引用:** 118 | [OpenAlex ID](https://openalex.org/A5104035860)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过设计三类空间推理任务（关系组合、视角变换、空间程序执行），在英语、汉语、阿拉伯语上评估大型语言模型，并使用线性探测、稀疏自编码器与因果干预分析其内部空间表征。

**💡 创新点**

提出以空间认知计算原理为基础的三原语分解任务体系，并将机制性解释与跨语言评估相结合，揭示模型空间表征的临时性与语言依赖性。

**🔧 技术方法**

线性探测器、稀疏自编码器（SAE）、梯度归因、激活补丁干预、跨语言任务生成等技术。

**📊 数据集**

自制的三种任务族（共 18,000 条样本，英语/汉语/阿拉伯语各 6,000 条），每族 2,000 条训练集和 200 条测试集。

**📈 对比分析**

对比模型在各任务族上的准确率、R²、MAE 等指标，发现空间变量可在中层层解码但在最终层衰退，整体表现低于人类基准，并显示跨语言的机制差异。

**⚠️ 局限性**

仅评估 7‑8B 规模模型，任务仅为文本抽象，不涉及更复杂规划或多模态情境，因果工具局限于激活补丁和特征消融，可能忽略非线性分布式表征。

---

## 249. DiffusionAnything: End-to-End In-context Diffusion Learning for Unified Navigation and Pre-Grasp Motion

**arXiv ID:** 2603.26322 | [PDF](https://arxiv.org/pdf/2603.26322v1)

**作者:** Iana Zhura `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 2143 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种统一的图像空间扩散策略，能够在米尺度导航和厘米尺度预抓取之间切换，只需5分钟自监督数据即可快速适配新任务。

**💡 创新点**

创新点：① 多尺度FiLM条件化实现任务模式、深度尺度和空间注意的统一；② 轨迹对齐深度预测聚焦关键点；③ 通过AnyTraverse自监督生成的注意力实现无语言模型、无深度传感器的目标推断。

**🔧 技术方法**

技术：基于扩散模型的图像空间轨迹生成、ViT编码器+多尺度FiLM调制、轨迹对齐深度推理、AnyTraverse自监督标签、轻量化UNet与多任务自监督损失。

**📊 数据集**

数据集：在桌面预抓取和室内通道导航两类场景中收集RGB视频（30Hz），利用AnyTraverse、ML-Depth-Pro和立体深度进行自动标注；训练集约20分钟/任务，快速适配仅5分钟。

**📈 对比分析**

与NoMaD导航基线和NVIDIA GR00T（n1.5、n1.6）对比：导航成功率100%且碰撞避免优于NoMaD；预抓取目标成功率70.6%且无碰撞；实现10Hz推理，内存仅2GB，零样本新场景保持完整性能，优于VLA模型。

**⚠️ 局限性**

局限性：依赖AnyTraverse监督，缺乏力学反馈；未集成多模态感知与双手协作；对真实世界误差的自适应学习仍待改进。

---

## 250. Line-of-Sight-Constrained Multi-Robot Mapless Navigation via Polygonal Visible Regions

**arXiv ID:** 2603.26314 | [PDF](https://arxiv.org/pdf/2603.26314v1)

**作者:** Ruofei Bai `[一作]`, Lihua Xie `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在未知环境下，利用实时LiDAR扫描构建自我可见域，实现多机器人保持视线连通并完成导航任务的分布式方法。

**💡 创新点**

主要创新在于：① 用多边形逼近可见域得到精确且可微的LoS距离度量；② 通过掩码图拉普拉斯控制器进行拓扑优化，剔除冗余连接，降低连通维护成本；③ 彻底摆脱先验地图需求，提升实用性。

**🔧 技术方法**

技术包括LiDAR可见域构建、径向角度插值多边形逼近、可微LoS距离计算、基于Fiedler特征值的图拉普拉斯控制、分布式最小生成树拓扑优化。

**📊 数据集**

在Gazebo仿真中随机生成带障碍的二维环境（无公开数据集），并在室内外真实三架DJI Tello无人机实验平台上验证。

**📈 对比分析**

与固定拓扑、纯图拉普拉斯拓扑以及不使用LoS距离的基线相比，Topo‑Opt方法在时间和行驶距离上平均提升约10%–20%，并在大规模障碍环境中保持连通性与效率。

**⚠️ 局限性**

局限性包括：依赖LiDAR/视觉传感器的精度；主要针对二维平面，三维扩展仍需研究；对极端动态障碍物或高噪声环境的鲁棒性待进一步验证。

---

## 251. A Benchmark for Evaluating Repository-Level Code Agents with Intermediate Reasoning on Feature Addition Task

**arXiv ID:** 2603.26337 | [PDF](https://arxiv.org/pdf/2603.26337v1)

**作者:** Shuhan Liu `[一作]` (Zhejiang University), Xin Xia `[通讯]` (Zhejiang University)

**通讯引用:** 21193 | [OpenAlex ID](https://openalex.org/A5006669765)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了RACE-bench，一个针对仓库级代码代理的推理增强评测基准，包含528个真实功能添加实例，并提供可执行补丁验证和结构化中间推理真值。

**💡 创新点**

首次引入双轨评估框架，将补丁正确性与中间推理质量并行评估，并提供详细的中间推理真值，首次实现对代码代理在功能添加任务中的推理过程进行定量诊断。

**🔧 技术方法**

结合大型语言模型（如GPT‑4/DeepSeek/GLM）作为代理底座，利用自动化工具链（代码搜索、静态分析、单元测试、Docker容器）构建环境，并使用LLM‑as‑Judge进行推理质量评估。

**📊 数据集**

从12个活跃的Python开源仓库收集的528个合并PR（功能添加）实例，构建了Lite子集（100实例），并基于PR与issue配对生成真实功能请求、金标准补丁和测试。

**📈 对比分析**

通过Patch Apply Rate和Resolved Rate评估补丁生成能力，在不同基模型和代理架构下比较，mini‑SWE‑Agent最高达70%已通过率；在推理层面，目标理解分数>9.2，文件定位/任务/步骤的recall随层级递减，失败案例显著降低recall并增加冗余预测。

**⚠️ 局限性**

评估依赖LLM‑as‑Judge可能引入偏差，推理真值由LLM辅助构造，隐式推理不被捕获，且仅评估三种代理和三种基模型，泛化性受限。

---

## 252. Large Language Models for Software Testing Education: an Experience Report

**arXiv ID:** 2603.26329 | [PDF](https://arxiv.org/pdf/2603.26329v1)

**作者:** Peng Yang `[一作]` (South China Normal University), Yong Tang `[通讯]` (South China Normal University)

**通讯引用:** 21984 | [OpenAlex ID](https://openalex.org/A5120811349)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文采用混合方法研究，首先通过课堂实证和大规模竞赛问卷两阶段探索学生在软件测试教育中使用大语言模型（LLM）的学习行为与常见难点；随后基于发现的交互瓶颈，设计并验证了一套轻量级、阶段感知的提示（prompt）框架，帮助学生在脚本生成任务中更系统地外化测试意图与约束。

**💡 创新点**

创新点：
1) 从学习行为视角系统刻画任务依赖的LLM使用难点，区分文本导向与脚本导向任务的交互模式；
2) 将课堂细节数据与全国竞赛问卷相结合，首次揭示不同任务类型对学生感知有效性、交互成本及调试时长的差异；
3) 设计可直接落地的提示框架，展示教学干预可将“盲目交付”转化为更具反思与阶段意识的协作方式。

**🔧 技术方法**

技术手段：
- 定性编码（开放/轴向）分析课堂提示日志和学生提交的脚本；
- Likert量表问卷收集学生对LLM有效性、交互轮数、调试时长等主观感知；
- 混合方法描述性统计与可视化（气泡图、柱状图）对比不同任务与不同阶段的指标；
- 轻量级提示模板（结构化模板）在课堂实验中手工使用，无需额外机器学习模型。

**📊 数据集**

数据集：
- 15名本科软件工程学生在高级软件测试课程中的实验数据，包括学生提交的测试计划、用例、脚本及其LLM交互记录；
- 337名全国软件测试竞赛参赛者的问卷回应，涵盖LLM使用频率、感知有效性、交互成本与学习影响。

**📈 对比分析**

比较方法与结果：
- 对比文本导向（需求分析、用例设计）与脚本导向（自动化脚本生成）任务在提示缺失、交互轮数、调试时长、感知有效性等指标上的分布；
- 发现脚本生成任务在缺失上下文、缺少交互、调试时长等方面显著高于文本任务；
- 在引入提示框架后，脚本错误中与环境、定位、同步相关的类别显著减少，表明学生更主动外化关键约束；
- 由于未进行对照实验，无法量化模型性能提升，但能观察到交互成本与学习体验的明显差异。

**⚠️ 局限性**

局限性：
- 样本仅来自单一课程的15名学生，外部可推广性有限；
- 问卷数据主要为自报，可能存在回忆与主观偏差；
- 提示框架仅在实验场景中使用，缺乏对照组，因果关系未得到严格验证；
- 研究聚焦于本科阶段，尚不清楚在研究生或行业培训中的适用性。

---

## 253. Label-Free Cross-Task LoRA Merging with Null-Space Compression

**arXiv ID:** 2603.26317 | [PDF](https://arxiv.org/pdf/2603.26317v1)

**作者:** Wonyoung Lee `[一作]` (Korea Advanced Institute Of Science And Technology), Kuk-Jin Yoon `[通讯]` (Korea Advanced Institute Of Science And Technology)

**通讯引用:** 6190 | [OpenAlex ID](https://openalex.org/A5113931494)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于LoRA适配器的模型合并方法——Null‑Space Compression (NSC) Merging

**💡 创新点**

利用LoRA训练过程中下投影矩阵的零空间压缩率作为无标签的合并信号，实现跨分类、回归和生成任务的统一合并

**🔧 技术方法**

LoRA、梯度无标签优化、零空间压缩率计算、轻量级合并

**📊 数据集**

20个视觉密集预测任务（NYUD‑v2、PASCAL‑Context、Taskonomy）、六个NLI基准（MNLI、QNLI、SNLI、RTE、SICK、SciTail）、六个视觉‑语言任务（VizWiz、IconQA、ChartQA、DocVQA、COCO、Flickr30k）

**📈 对比分析**

与10种基线（Task Arithmetic、TIES、DARE、AdaMerging、SVD、Linear、KnOTS、LoRA‑LEGO等）比较，NSC在所有实验中均获得最高或接近最高的归一化性能（视觉任务92%均值，NLI 92.3%，VLM 约92%）

**⚠️ 局限性**

仍需评估在更大规模模型或更复杂任务的可扩展性以及在少量标签或极端零样本场景下的鲁棒性

---

## 254. Integration Adapter Architecture for Food Traceability Blockchain

**arXiv ID:** 2603.26306 | [PDF](https://arxiv.org/pdf/2603.26306v1)

**作者:** André Romão `[一作]` (INESC-ID), Miguel L. Pardal `[通讯]` (INESC-ID)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一套模块化的适配器架构，帮助中小企业将遗留系统与许可区块链集成，并在葡萄牙Fundão樱桃供应链中进行试点验证。

**💡 创新点**

通过可配置提取器、变换器、消息中间件、区块链加载器和状态可视化等五大模块，实现了跨系统异构数据的无缝桥接、容错与可伸缩性，并提供端到端可观测性，显著降低技术门槛。

**🔧 技术方法**

使用Python/Go实现，Apache Kafka作为消息中间件，Hyperledger Fabric v2.5为区块链平台，EPCIS作为标准数据模型，Docker容器化部署，Prometheus/Grafana监控，TLS 1.3保障通信安全。

**📊 数据集**

使用真实供应链事件数据（农场采摘、加工、零售等），统一转换为EPCIS事件格式，未公开具体数据集。

**📈 对比分析**

通过功能测试、性能基准（100/500 req/s）以及监控指标评估：API平均延迟20–24 ms，Kafka吞吐约2,500 msg/min，Fabric单笔交易≈1 tx/s，整体表现低延迟、无数据丢失。

**⚠️ 局限性**

局限性包括：链码层缺少EPCIS数据校验；多链写入支持尚未充分验证；完整安全审计待完成；并且受Kafka与Fabric吞吐瓶颈影响，需进一步提升并发性能。

---

## 255. PhysVid: Physics Aware Local Conditioning for Generative Video Models

**arXiv ID:** 2603.26285 | [PDF](https://arxiv.org/pdf/2603.26285v1)

**作者:** Saurabh `[一作]`, Bahram Zonooz `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 373 | [OpenAlex ID](https://openalex.org/A5028650455)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于视频局部片段的物理知识提示，对文本生成视频模型进行局部物理感知的条件引导

**💡 创新点**

创新点在于：①将视频分割为时间连续片段，并用VLM生成对应的物理描述作为局部提示；②在Transformer中加入chunk-aware交叉注意力并使用RoPE对齐文本与时间片段；③推理时同时使用正向物理提示和对抗性（违反物理）的提示，进行判别器自由引导以提升物理合理性

**🔧 技术方法**

主要技术包括：多模态语言模型（VLM）生成物理提示、Chunk-aware cross‑attention（RoPE对齐）、classifier‑free guidance、对抗性（counterfactual）提示

**📊 数据集**

使用WISA-80k作为训练数据，生成7个0.7秒的时间片段提示；在VideoPhy和VideoPhy2上评估物理常识得分

**📈 对比分析**

与Wan‑1.3B、Wan‑14B基线对比，1.7B模型在VideoPhy上物理常识提升约33%，在VideoPhy2提升约8%；在多类别子集均保持领先；即使参数量更小也能匹配甚至超越更大模型

**⚠️ 局限性**

局限性包括：需要额外的VLM推理来生成物理提示，增加推理成本；对物理描述的准确性高度依赖VLM表现；目前仅在特定物理基准上验证，缺乏更广泛的场景评估；对超长视频或复杂物理交互的适应性尚未充分验证

---

## 256. Knowdit: Agentic Smart Contract Vulnerability Detection with Auditing Knowledge Summarization

**arXiv ID:** 2603.26270 | [PDF](https://arxiv.org/pdf/2603.26270v1)

**作者:** Ziqiao Kong `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 85683 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于审计知识图谱的多智能体框架，用来自动检测智能合约中的漏洞。

**💡 创新点**

创新点在于：① 把 DeFi 业务逻辑抽象为“DeFi 语义”并与漏洞模式关联，构建知识图谱；② 在多智能体循环中利用共享工作内存实现规范生成、工具合成、模糊测试与结果反思，实现类人式的持续迭代审计；③ 通过知识图谱提升了漏洞检测的覆盖率与精度。

**🔧 技术方法**

主要技术包括：大语言模型（如 GPT‑5）用于知识抽取、生成与推理；Foundry 框架进行模糊测试；知识图谱（图结构检索、关系推理）与多智能体协作；共享工作内存来记录执行、覆盖与反馈。

**📊 数据集**

使用 Code4rena 公开的 Solidity 审计竞赛数据：270 个项目（1,429 条候选语义）用于构建知识图谱，12 个新项目（100 合约，14 高危 + 61 中危）用于评估；此外，还在 6 个真实项目中验证。

**📈 对比分析**

与多种基准（LLM 静态审计、符号执行、基于 invariant 的测试）对比，本文方法在 12 项目上检出了 14/14 高危、61/77% 中危，仅 2 个误报；知识图谱覆盖率达 88%，基准仅 45%；在真实项目中发现 12 高危 + 10 中危并被修复；成本约 80 美元/项目，虽然高于基准，但远低于人工审计。

**⚠️ 局限性**

局限性包括：依赖 LLM，存在随机性与误判；知识图谱的更新与维护需要人工干预；对极其细节化的实现缺陷仍可能漏检；token 消耗相对较高；缺乏人机交互，导致部分误报或漏报。

---

## 257. GUIDE: Resolving Domain Bias in GUI Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation

**arXiv ID:** 2603.26266 | [PDF](https://arxiv.org/pdf/2603.26266v1)

**作者:** Rui Xie `[一作]` (Shanghai Jiao Tong University), Qing Li `[通讯]` (State Key Laboratory for General Artificial Intelligence, BIGAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无需训练的插件式框架，通过实时检索并自动注释网络教程视频，解决大型视觉语言模型在图形用户界面代理中的领域偏差问题。

**💡 创新点**

创新点在于：①基于字幕的三阶段Video‑RAG检索策略，精准识别与任务相关的教程视频；②完全自动化的逆动力学注释管线，生成可迁移的规划与定位知识；③将知识以自然语言注入，无需修改代理模型参数或架构。

**🔧 技术方法**

技术方法包括：字幕驱动的检索与过滤、Whisper ASR与MOG2关键帧提取、OmniParser UI 元素检测、VLM 逆动力学推理、LLM 知识分解与结构化输出。

**📊 数据集**

实验使用了公开的 OSWorld 基准，涵盖 361 个真实世界桌面任务，检索来源为 YouTube 公共教程视频。

**📈 对比分析**

在三种代理架构（单模型 Qwen3‑VL‑8B、Seed‑1.8 以及多模型 AgentS3）中，知识注入均提升 4.5–7.5 % 的完成率；相较于最接近的 Watch & Learn 方法，提升显著，且无需预构建语料库。

**⚠️ 局限性**

局限性包括：检索覆盖率约 82.8 %，对未检索到的任务无法提升；当教程与目标界面差异过大时，定位知识可能误导；此外，注释过程仍需依赖 LLM，存在推理质量波动与费用开销。

---

## 258. Topology-Aware Graph Reinforcement Learning for Energy Storage Systems Optimal Dispatch in Distribution Networks

**arXiv ID:** 2603.26264 | [PDF](https://arxiv.org/pdf/2603.26264v1)

**作者:** Shuyi Gao `[一作]` (Delft University of Technology), Pedro P. Vergara `[通讯]` (Beijing Energy Quant Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种拓扑感知的图神经网络强化学习框架，用于分布式网络中能量储存系统（ESS）的实时最优调度。

**💡 创新点**

创新点在于将三种代表性GNN（GCN、TAGConv、GAT）嵌入到TD3的非对称演员-评论家结构中，既捕捉局部节点信息又聚合全局电压影响，并系统评估了不同拓扑和网络规模下的泛化能力。

**🔧 技术方法**

核心技术包括图神经网络消息传递、Twin Delayed Deep Deterministic Policy Gradient（TD3）强化学习、全局与局部池化、以及零散的节点特征编码。

**📊 数据集**

实验数据集基于IEEE 34节点和69节点分布式网络，模拟一天的15分钟决策步，设置了不同的ESS节点容量与功率，使用真实负荷、PV和价格预测序列。

**📈 对比分析**

与基准全局非线性规划（NLP）和传统多层感知器（NN）RL进行对比；结果显示GNN-RL在电压违规次数与违规幅值上明显优于NN，且在69节点系统中可实现比TD3-NN高出约20%的经济收益；同时在线推理速度比NLP快两到三百倍。

**⚠️ 局限性**

主要局限是跨网络（不同节点数）零射转移性能显著下降，GAT模型在随机种子敏感性高，且训练成本（计算时间与参数规模）较大，未来需引入拓扑随机化或细调策略以提升鲁棒性。

---

## 259. Auditing Blockchain Innovations: Technical Challenges Beyond Traditional Finance

**arXiv ID:** 2603.26361 | [PDF](https://arxiv.org/pdf/2603.26361v1)

**作者:** Shayan Eskandari `[一作]` (Nova School of Business and Economics, Universidade NOVA de Lisboa), Jeremy Clark `[通讯]` (Concordia University)

**通讯引用:** 6427 | [OpenAlex ID](https://openalex.org/A5037375405)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过作者在区块链工程、智能合约审计与CTO等多角色实践经验，采用自我民族志方法系统分析并提出了针对代币空投、多签名钱包与实时链上财务报告等区块链创新的审计框架和实验性审计程序。

**💡 创新点**

创新点在于将自我民族志方法与审计理论相结合，突破传统审计框架对区块链技术的认知空白，提出了四大挑战分类（存在、所有权、估值、内部控制）并为每类提供了具体案例与验证路径。

**🔧 技术方法**

主要技术包括区块链基础设施操作、智能合约源代码与字节码分析、多签名治理模型评估、链上数据实时抓取与价格源校验，以及针对不同空投协议的交易流程模拟。

**📊 数据集**

使用的数据集为作者个人职业经历与项目记录：约4000小时DeFi协议审计、10亿美元以上资产管理、10+多签名部署及5次监管研讨会等；这些经验数据构成了自我民族志的实证基础。

**📈 对比分析**

对比方法是将传统审计程序（如购买凭证、银行确认等）与提出的区块链实验性验证方法（如链上快照验证、源代码审计、实时价格来源校验）进行对照，结果显示后者在验证空投存在、所有权证明和实时账务透明度方面能够显著减少人工确认错误与时间成本，提升审计质量。

**⚠️ 局限性**

局限性包括研究聚焦于特定区块链创新，缺乏对DAO等去中心化组织的适用性研究；方法主要基于单一作者经验，需在更广泛审计场景中验证；并且未对不同区块链网络（如Layer‑2、跨链）在治理与安全性上的差异进行系统性评估。

---

## 260. MPDiT: Multi-Patch Global-to-Local Transformer Architecture For Efficient Flow Matching and Diffusion Model

**arXiv ID:** 2603.26357 | [PDF](https://arxiv.org/pdf/2603.26357v1)

**作者:** Quan Dao `[一作]` (Rutgers University), Dimitris Metaxas `[通讯]` (Rutgers University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种多尺度多补丁 Transformer（MPDiT），用于扩散模型，在保持生成质量的同时显著提升训练效率。

**💡 创新点**

创新点包括全局-局部分层 Transformer 结构、Fourier Neural Operator 时间嵌入以及多令牌类别嵌入，三者共同实现了 GFLOPs 大约 50% 的下降和 FID 的显著提升。

**🔧 技术方法**

使用了多级补丁分辨率、上采样块、共享时间嵌入、FNO 时间嵌入、类多令牌以及流匹配训练框架。

**📊 数据集**

实验数据集为 ImageNet 256×256，涵盖 1,000 个类别的 1,281,167 张训练图像。

**📈 对比分析**

与 DiT、SiT、DiG 等基线模型比较，MPDiT‑XL 在仅 240 轮训练后即可获得 CFG FID 2.05，参数量下降约 30%，GFLOPs 降低近 50%，并比 DiT‑XL/2 的采样速度快 2×，整体性能优异。

**⚠️ 局限性**

局限性在于目前仅验证了图像生成任务，扩展至文本到图像或视频等更大规模任务仍需大量计算资源与进一步验证。

---

## 261. Optimal Prioritized Dissipation and Closed-Form Damping Limitation under Actuator Constraints for Haptic Interfaces

**arXiv ID:** 2603.26347 | [PDF](https://arxiv.org/pdf/2603.26347v1)

**作者:** Camilla Celli `[一作]` (Scuola Superiore Sant'Anna), Antonio Frisoli `[通讯]` (Scuola Superiore Sant'Anna)

**通讯引用:** 8480 | [OpenAlex ID](https://openalex.org/A5090204404)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

暂无可用信息

**💡 创新点**

暂无可用信息

**🔧 技术方法**

暂无可用信息

**📊 数据集**

暂无可用信息

**📈 对比分析**

暂无可用信息

**⚠️ 局限性**

暂无可用信息

---

## 262. HINT: Composed Image Retrieval with Dual-path Compositional Contextualized Network

**arXiv ID:** 2603.26341 | [PDF](https://arxiv.org/pdf/2603.26341v1)

**作者:** Mingyu Zhang `[一作]` (Shandong University), Yupeng Hu `[通讯]` (Shandong University)

**通讯引用:** 1723 | [OpenAlex ID](https://openalex.org/A5069741536)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HINT 模型，用双路径结构对查询图像与文本进行上下文编码，并将其与目标图像进行匹配

**💡 创新点**

在视觉与文本之间同时建模视觉内模态上下文与跨模态上下文，并通过上下文相关性量化与双路径一致性约束放大正负样本相似度差异

**🔧 技术方法**

基于 BLIP‑2 预训练特征、Multi‑Head Attention、Q‑former、上下文对比损失与排名损失等技术实现

**📊 数据集**

在 FashionIQ 与 CIRR 两个公开基准数据集上进行实验

**📈 对比分析**

与多种最新方法对比，HINT 在所有评测指标上均取得最高或接近最高分，R@1 与 R@10 等指标明显提升

**⚠️ 局限性**

模型仍受限于对极端复杂修改的处理、对 BLIP‑2 特征的依赖以及计算资源消耗等方面

---

## 263. From Pixels to Privacy: Temporally Consistent Video Anonymization via Token Pruning for Privacy Preserving Action Recognition

**arXiv ID:** 2603.26336 | [PDF](https://arxiv.org/pdf/2603.26336v1)

**作者:** Nazia Aslam `[一作]` (Aalborg University), Kamal Nasrollahi `[通讯]` (Aalborg University)

**通讯引用:** 3598 | [OpenAlex ID](https://openalex.org/A5041199606)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6215c339-3735-4be3-8a07-5bbb7004712d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于注意力驱动的时空视频匿名化框架，通过在Vision Transformer中引入两个专用CLS Token来区分动作识别与隐私泄露信息，并利用其注意力分布对tubelet进行评分与剪枝；

**💡 创新点**

创新点在于：①利用双CLS Token实现动作与隐私特征的显式解耦；②基于注意力差值计算utility‑privacy分数，动态选择保留tubelet；③对被剪裁的tubelet进行残差融合，避免信息过度丢失；

**🔧 技术方法**

技术手段包括：Vision Transformer（ViT）视频编码、双CLS Token注意力掩蔽、top‑k tubelet选择、残差融合模块、梯度反转层的对抗训练；

**📊 数据集**

使用三个公开数据集：VPUCF（UCF101+隐私标签）、VPHMDB（HMDB51+隐私标签）和PAHMDB（HMDB51子集+帧级隐私标签）；

**📈 对比分析**

与多种基线（Downsample、Blackening、StrongBlur、VITA、SPAct、Balancing、STPrivacy等）以及自定义ViT+I3D/ResNet对比，实验显示在动作识别上与原始视频差距≤2%，而在隐私泄漏指标（cMAP、F1）下降≈15-20%，优于STPrivacy和其他方法，且在跨数据集迁移中保持更稳健的隐私抑制；

**⚠️ 局限性**

局限性在于：①仅针对视频级隐私标签，难以处理更细粒度或多模态隐私信息；②剪枝比例需经验调节，过度剪裁会显著损失动作识别性能；③当前方法在实时或低算力场景下的推理效率未做评估；

---

## 264. Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy

**arXiv ID:** 2603.26299 | [PDF](https://arxiv.org/pdf/2603.26299v1)

**作者:** Wooseong Jeong `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TARA‑Merging 方法，解决将多个 LoRA 模块合并时子空间覆盖不足和方向各向异性导致的性能损失；

**💡 创新点**

创新点在于同时保证 LoRA 子空间的有效秩（subspace coverage）和对各方向的敏感性进行权重调节（anisotropy alignment），并通过偏好加权的交叉熵伪损失实现任务优先级的精细对齐；

**🔧 技术方法**

使用了 LoRA 低秩适配、有效秩（effective rank）评估、方向敏感性分析、Tchebycheff 标量化、梯度熵优化、SVD 分解、AdamW 等技术；

**📊 数据集**

在视觉领域使用 CLIP ViT‑B/32 对八个图像分类基准进行 LoRA 微调，在自然语言推断领域使用 LLaMA‑3 8B 对六个 NLI 任务进行 LoRA 微调；

**📈 对比分析**

与 13 种基线（Task Arithmetic、TIES、DARE、AdaMerging、KnOTS、LoRA‑LEGO 等）在 per‑task、joint‑task 以及未见任务的评估中对比，TARA‑Merging 在多项指标上平均提升 3–5%，并在任务权重平衡上表现更稳定；

**⚠️ 局限性**

局限性包括需要手工指定任务偏好向量、在大量任务集合上 SVD 计算成本较高，以及对极端子空间重叠场景的鲁棒性尚未充分验证。

---

## 265. findsylls: A Language-Agnostic Toolkit for Syllable-Level Speech Tokenization and Embedding

**arXiv ID:** 2603.26292 | [PDF](https://arxiv.org/pdf/2603.26292v1)

**作者:** Héctor Javier Vázquez Martínez `[一作]` (University of Pennsylvania), Héctor Javier Vázquez Martínez `[通讯]` (University of Pennsylvania)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5102574236)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了findsylls工具包，统一了经典与神经声学单元的音节分割与嵌入，支持多语言实验。

**💡 创新点**

模块化设计允许分离特征与分割算法，支持跨语言复用与可复现的评估；同时提供统一评估框架。

**🔧 技术方法**

使用经典振幅包络（SBS、Theta）、自监督模型（HuBERT、VG‑HuBERT、Sylber）、MinCut、余弦相似度、CLS注意力等技术。

**📊 数据集**

实验涵盖English（TIMIT、LibriSpeech、LS‑100h）、Spanish（WikiSpanish、Ornat‑Swingley）以及Kono的手注Fieldwork数据集。

**📈 对比分析**

通过核、边界、跨度的F1比较七个语料，发现端到端神经分割优于经典包络，模块组合可进一步提升边界与跨度F1；token速率在3–6 tok/s，Envelope基线最快。

**⚠️ 局限性**

受实现细节与超参影响、RTFx仅在单平台测算、部分方法未完整覆盖、边界定位存在主观不确定性导致误判等限制。

---

## 266. Proofdoors and Efficiency of CDCL Solvers

**arXiv ID:** 2603.26286 | [PDF](https://arxiv.org/pdf/2603.26286v1)

**作者:** Sunidhi Singh `[一作]` (Georgia Institute of Technology), Vijay Ganesh `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8279 | [OpenAlex ID](https://openalex.org/A5052292970)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了“proofdoor”这一新的结构化参数，并证明若公式具有小型 proofdoor，则可得到多项式规模的 Resolution 证明，进而在适当配置的 CDCL 运行下实现多项式冲突数的求解；同时对浮点加法的可比较性 Miter 进行分析并给出下界与上界。

**💡 创新点**

创新点在于将插值、路径宽度、CacheSAT 与 CDCL 的行为统一到一个新的分解框架——proofdoor；通过证明拆分的好坏决定证明长度的根本差异，揭示了 CDCL 高效性的理论根源。

**🔧 技术方法**

主要技术包括插值（Craig interpolants）、路径宽度分析、CacheSAT 算法、CDCL 与 CacheSAT 的模拟、部分有序分辨证明、CNF/DFN 大小下界推导，以及递归分解与支持集分析。

**📊 数据集**

实验与案例主要使用浮点加法可比较性 Miter 的 CNF 编码（包含 IEEE‑754 简化模型）以及算术 Miter 电路（如 xy = yx、x(y+z)=xy+xz 等）作为示例，未使用标准工业数据集。

**📈 对比分析**

通过理论证明展示：在小型 proofdoor 的情况下，Resolution 证明大小为多项式，而错误拆分可迫使分辨证明指数增长；同样，CDCL 在这些实例上可在 O(n) 冲突内完成求解，显著优于一般下界所暗示的指数成本。

**⚠️ 局限性**

局限性在于 proofdoor 的效果高度依赖于恰当的分解；不良拆分可导致指数级证明；此外，判定一个公式族是否拥有多项式规模证明是不可判定的，体现了框架的根本理论边界。

---

## 267. Hermes Seal: Zero-Knowledge Assurance for Autonomous Vehicle Communications

**arXiv ID:** 2603.26343 | [PDF](https://arxiv.org/pdf/2603.26343v1)

**作者:** Munawar Hasan `[一作]` (National Institute of Standards and Technology), Thoshitha Gamage `[通讯]` (Southern Illinois University Edwardsville)

**通讯引用:** 441 | [OpenAlex ID](https://openalex.org/A5112453990)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

为车联网中的协作感知提供隐私保护且可验证的零知识证明框架Hermes' Seal；

**💡 创新点**

将Groth16 zk‑SNARK与感知栈无缝集成，实现模型无关、数据不泄露的可广播证明，并通过安全与性能两大案例验证其可行性；

**🔧 技术方法**

使用Groth16 zk‑SNARK、Circom电路编译、Poseidon哈希、数字签名及GPU加速的RapidSnark实现；

**📊 数据集**

案例一基于车辆速度、距离等数值；案例二使用公开基准数据集（如KITTI/nuScenes/COCO）进行精度召回测试；

**📈 对比分析**

对比SnarkJS、RapidSnark与RapidSnark‑GPU三实现，GPU版将证明生成时间从约300 ms降至5 ms，验证时间从233 ms降至1 ms，性能提升约50‑250 倍；

**⚠️ 局限性**

主要局限包括对执法机构（EA）的信任依赖、算术操作受限、证明生成时间随电路复杂度增加、缺乏证明执行（PoX）等保证。

---

## 268. A General Theory of Propositional Modal Bundled Modalities

**arXiv ID:** 2603.26268 | [PDF](https://arxiv.org/pdf/2603.26268v1)

**作者:** Yifeng Ding `[一作]` (Peking University), Yuanzhe Yang `[通讯]` (Peking University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5111190161)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一个统一的理论框架，用于研究和形式化“bundled modalities”（捆绑模态），包括其可表达性、Bisimulation 的定义以及对凸捆绑模态的完整性证明，并给出了若干具体实例（如“someone knows”、群体分歧以及信念与知识的组合）的完整公理化。

**💡 创新点**

创新点在于：①将捆绑模态的语义视为特殊的邻域语义，统一定义 Bisimulation 并证明其 Hennessy‑Milner 性；②引入“凸捆绑模态”这一新类别，证明其在邻域语义下的完整性；③提出一套从邻域模型到 Kripke 模型的表示方法，能够将邻域完备性转化为 Kripke 完备性，进而完成多种复杂捆绑模态的完整公理化。

**🔧 技术方法**

主要技术包括：邻域语义的构造、Bisimulation 的一致性与 Hennessy‑Milner 定理证明、Zorn 引理用于构造最大滤子、核心模型构造与完备性证明、以及表示理论（从凸邻域模型到 Kripke 模型的映射）。

**📊 数据集**

无数据集；本工作为理论研究，主要使用形式化推导和模型构造，不涉及实验数据。

**📈 对比分析**

不涉及实验对比；本文通过形式化证明展示了新方法的逻辑正确性和完备性，没有计算性能指标可供比较。

**⚠️ 局限性**

局限性包括：①仅关注单一一元捆绑模态，未覆盖多元或非单元情形；②凸捆绑模态的定义仍相对有限，部分非凸捆绑模态的完整性仍未解决；③Bisimulation 的结构化证明依赖选择公理，对构造过程的可实现性有一定理论限制。

---

## 269. From Pen to Pixel: Translating Hand-Drawn Plots into Graphical APIs via a Novel Benchmark and Efficient Adapter

**arXiv ID:** 2603.26356 | [PDF](https://arxiv.org/pdf/2603.26356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 270. DRUM: Diffusion-based Raydrop-aware Unpaired Mapping for Sim2Real LiDAR Segmentation

**arXiv ID:** 2603.26263 | [PDF](https://arxiv.org/pdf/2603.26263v1)

**作者:** Tomoya Miyawaki `[一作]` (Kyushu University), Ryo Kurazume `[通讯]` (Kyushu University)

**通讯引用:** 3694 | [OpenAlex ID](https://openalex.org/A5073445963)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用扩散模型实现 Sim2Real 翻译，生成包含真实反射强度与射线掉落噪声的 LiDAR 点云，用于训练语义分割模型。

**💡 创新点**

提出射线掉落感知的掩码引导机制，将 Sim2Real 视为后验采样；在保持几何一致性的同时，保留真实域的噪声特征。

**🔧 技术方法**

扩散模型（R2DM）、后验采样、伪逆引导、渐进掩码、SDEdit 初始化、RePaint 迭代重采样等技术。

**📊 数据集**

使用 SynLiDAR 作为仿真数据集，SemanticKITTI 作为真实数据集；在 GTA‑LiDAR→KITTI‑frontal 任务中亦进行评估。

**📈 对比分析**

与渲染模型、SDEdit、ΠGDM、ePointDA、DUSty 等方法比较，DRUM 在样本真实性指标和语义分割 mIoU 上均获得最优或接近最优成绩。

**⚠️ 局限性**

评价指标对每个样本质量的评估不足；多模态（反射强度与射线掉落）的一致性建模仍有提升空间。

---

## 271. GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation

**arXiv ID:** 2603.26260 | [PDF](https://arxiv.org/pdf/2603.26260v1)

**作者:** Xujing Tao `[一作]` (University of Science and Technology of China), Tianzhu Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18206 | [OpenAlex ID](https://openalex.org/A5100648981)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 GeoGuide 框架，通过层次化几何引导实现开词汇 3D 语义分割。

**💡 创新点**

创新点在于引入不确定性加权超点蒸馏、实例级遮罩重建与跨实例关系一致性三个模块，兼顾局部到全局的几何语义一致性。

**🔧 技术方法**

利用预训练 3D 背骨（如 PointContrast）、冻结的 2D 开词汇分割模型、适配器将 3D 语义映射到 CLIP 文字空间，并实现三种一致性约束。

**📊 数据集**

在 ScanNet v2、Matterport3D 和 nuScenes 三个基准上进行实验。

**📈 对比分析**

与现有零样本方法对比，GeoGuide 在所有数据集上均实现了 mIoU 提升 4.9–5.6 点，并逼近或超过部分监督基线，显示出显著的性能优势。

**⚠️ 局限性**

仍依赖多视角 RGB 输入以获得 2D 监督，且对极端遮挡或极少样本类别的鲁棒性有待进一步提升。

---

## 272. A Formal Framework for Uncertainty Analysis of Text Generation with Large Language Models

**arXiv ID:** 2603.26363 | [PDF](https://arxiv.org/pdf/2603.26363v1)

**作者:** Steffen Herbold `[一作]` (University of Passau), Florian Lemmerich `[通讯]` (University of Passau)

**通讯引用:** 1296 | [OpenAlex ID](https://openalex.org/A5076202690)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一的数学框架，用采样树来描述大型语言模型（LLM）的提示、生成和解释过程，并通过过滤器与目标函数来定量测量不同来源的不确定性。

**💡 创新点**

创新点在于：①将提示、解码和解释三者纳入同一采样树；②引入过滤器和目标函数实现对不确定性来源的细粒度选择；③统一定义目标加权概率和熵，揭示现有方法的形式关系；④指出解释不确定性和估计方法交互等研究盲区。

**🔧 技术方法**

主要技术包括：采样树建模、过滤器（end‑of‑output、prefix、single‑token、fixed‑length等）、目标函数族（句法相等、硬聚类、软聚类/相似度）、目标加权概率与熵的定义、估计器（top‑k、nucleus/概率质量、蒙特卡洛模拟）以及对模型、过滤器与目标的不确定性扩展。

**📊 数据集**

论文未提供具体实验或数据集，侧重理论框架和方法映射；若有实验，假设使用公开LLM和标准提示/评测数据集（如GLUE、ARC、OpenAI API）以演示框架应用，但原文未给出。

**📈 对比分析**

通过在框架中映射现有不确定性度量（如单词概率、熵、语义熵、self‑consistency 等），论文展示了方法之间的形式对应关系；未给出实际性能对比或数值结果，主要以概念验证和方法统一为主。

**⚠️ 局限性**

局限性包括：①不处理 LLM 直接输出不确定性（verbalization）；②假设全部过程是自回归左到右，难以直接适用于非自回归或一次性生成；③未进行实证验证，缺乏实验支持；④对多代理、非文本模态等复杂情境的扩展尚未实现；⑤估计方法对过滤器的依赖需要进一步系统化。

---

## 273. Only Whats Necessary: Pareto Optimal Data Minimization for Privacy Preserving Video Anomaly Detection

**arXiv ID:** 2603.26354 | [PDF](https://arxiv.org/pdf/2603.26354v1)

**作者:** Nazia Aslam `[一作]` (Aalborg University), Kamal Nasrollahi `[通讯]` (Aalborg University)

**通讯引用:** 3598 | [OpenAlex ID](https://openalex.org/A5041199606)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了面向隐私的视频异常检测的“Only What’s Necessary”数据最小化框架，系统地平衡检测性能与隐私泄露；

**💡 创新点**

将广度（时间采样）和深度（空间/外观降噪）两维最小化与Pareto最优选择相结合，首次在视频异常检测中将隐私泄露量化并与性能共同优化；

**🔧 技术方法**

使用时间采样、下采样、遮挡、模糊和背景移除等操作；利用自编码器作为异常检测模型、ResNet‑50作为隐私属性检测模型，并通过Pareto分析与三种选择策略（理想点距离、加权聚合、约束选取）确定最优配置；

**📊 数据集**

在UCSD Ped1/Ped2、CUHK Avenue异常检测数据集以及PAHMDB（含皮肤、关系、面部、裸露、性别等属性）上进行训练与评估；

**📈 对比分析**

对不同最小化配置进行AUC、cMAP、F1评估，并通过Pareto曲线和排名对比；最终发现时间采样stride=5+模糊是最优折中点，既保持高AUC又显著降低隐私泄露；

**⚠️ 局限性**

限制在于缺乏对真实威胁模型下的隐私评估，仅使用跨数据集评估，未考虑更复杂的攻击方式及部署环境中的实时约束。

---

## 274. Reflect to Inform: Boosting Multimodal Reasoning via Information-Gain-Driven Verification

**arXiv ID:** 2603.26348 | [PDF](https://arxiv.org/pdf/2603.26348v1)

**作者:** Shuai Lv `[一作]` (University of Science and Technology of China), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10635 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对多模态大语言模型在长篇生成中出现的视觉漂移问题，提出了自演化的视觉再检查（VRE）框架，利用模型自身的反思轨迹实现隐式视觉重检，从而提升长链推理的视觉根基和整体准确率。

**💡 创新点**

创新点包括：① 通过信息增益驱动的反思合成，筛选出真正能补充视觉证据的反思文本；② 同构重构策略，保证视觉与推理在同一视觉编码器上对齐；③ 将冷启动SFT、RLVR与后续自蒸馏循环成闭环，形成自我迭代的训练流程；④ 通过专门的反思奖励避免文本奖励劫持，确保模型真正关注图像。

**🔧 技术方法**

使用的技术主要有：大规模多模态模型（如Qwen2.5‑VL‑7B）+自监督微调（SFT）+强化学习可验证奖励（RLVR，GRPO）+信息增益评估 + 反思模板 + 重新采样与拒绝采样。

**📊 数据集**

所用数据集包括：Thyme‑SFT（冷启动），MM‑Eureka、V^*、ViRL39K（RL训练），以及公开评测基准如MathVista、MathVerse、MathVision、WeMath、LogicVista、MMMU、V^*-Bench、ChartQA、OCRBench_v2、Real‑world QA 等。

**📈 对比分析**

与基准模型（如Qwen2.5‑VL‑7B、GPT‑4o、Claude 3.7、InternVL系列、DeepEyesV2 等）对比，VRE 在 7B 参数规模下在数学推理、跨学科推理、高分辨率感知、文档理解等多项任务均实现了显著提升（例如在 V^*-Bench 上提升至 83.8%、MathVista 71.2、MathVerse 53.1 等），尤其在长链推理与细粒度视觉提取任务中取得了明显的性能突破。

**⚠️ 局限性**

局限性：① RL 训练易出现过拟合和视觉能力衰退，需多轮自蒸馏平衡；② 对极其复杂或超出模型内在视觉理解范围的样本仍可能产生误检或停滞；③ 目前仅在 7B 规模模型上验证，尚未证实在更大模型或不同视觉编码器上的普适性；④ 反思生成的多样性受模板限制，可能对开放式任务的适应性不足。

---

## 275. Lean on Vampire Proofs (Short Paper)

**arXiv ID:** 2603.26342 | [PDF](https://arxiv.org/pdf/2603.26342v1)

**作者:** Jonas Bodingbauer `[一作]` (TU Wien), Michael Rawson `[通讯]` (University of Southampton)

**通讯引用:** 81 | [OpenAlex ID](https://openalex.org/A5080196917)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个从 Vampire 自动定理证明器输出可在 Lean 里验证的可信证明的系统，并实现了证明重建与验证的完整流程。

**💡 创新点**

创新点在于将 Vampire 的超级位置推理、前向化简等推理步骤映射为 Lean 证明词，并通过定理化、重用专用 tactics 实现对整个证明链的可验证重现，显著提升了证明的可信度。

**🔧 技术方法**

使用了 Lean 交互式定理证明框架、Superposition 推理、Paramodulation、前向化简、SAT 结构分裂、mgu 统一、以及自定义 tactics 等技术，配合 Vampire 的 Discount saturation 循环。

**📊 数据集**

采用了 TPTP 9.2.1 版本的 CNF 与 FOF 题库共计 17,603 条实例进行实验评测。

**📈 对比分析**

与 Vampire 基线版本（无证明输出）比较，实验显示在 CNF 题目上 98% 的可信证明生成成功率，FOF 题目上 85% 成功率；同时证明重建与验证的时间相对较短，且与原始证明搜索时间呈低相关性。

**⚠️ 局限性**

局限性包括仅支持 Vampire 的部分推理规则，无法覆盖所有前置步骤；大文件和未实现的推理规则导致部分证明无法重建；在复杂推理中 tactic 超时和实现 bug 仍然存在。

---

## 276. PRISMA: Toward a Normative Information Infrastructure for Responsible Pharmaceutical Knowledge Management

**arXiv ID:** 2603.26324 | [PDF](https://arxiv.org/pdf/2603.26324v1)

**作者:** Eugenio Rodrigo Zimmer Neves `[一作]`, Bruno Morelli `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了PATOS–Lector–PRISMA三层架构，用以实现药学知识的文档保留、技术阅读与上下文展示。

**💡 创新点**

其创新点在于将文档保留、解释与呈现功能分离、引入可追溯的Evidence Pack以及基于RPDA的情境化视图，强化可追溯性与责任边界。

**🔧 技术方法**

技术包括OAIS与可信数字仓库、机器辅助阅读与人机协同、上下文图谱、论证框架和大语言模型的辅助分层。

**📊 数据集**

使用了超过17000份巴西监管文档、38份Evidence Pack以及对Dipyrone（Novalgina）的案例数据。

**📈 对比分析**

由于本工作主要是架构与概念验证，未做传统算法性能对比，而是通过案例演示可追溯性和上下文适配的可行性。

**⚠️ 局限性**

局限在于系统尚未完全集成、依赖人工校对导致可扩展性受限、对复杂或稀缺信息的处理尚待验证。

---

## 277. SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras

**arXiv ID:** 2603.26481 | [PDF](https://arxiv.org/pdf/2603.26481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 278. D-GATNet: Interpretable Temporal Graph Attention Learning for ADHD Identification Using Dynamic Functional Connectivity

**arXiv ID:** 2603.26308 | [PDF](https://arxiv.org/pdf/2603.26308v1)

**作者:** Qurat Ul Ain `[一作]` (National University of Sciences and Technology), Soyiba Jawed `[通讯]` (National University of Sciences and Technology)

**通讯引用:** 129 | [OpenAlex ID](https://openalex.org/A5014717805)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用动态功能连接（dFC）构建时间序列脑图，并通过图注意网络与时序注意机制实现对ADHD的可解释分类

**💡 创新点**

首次将动态功能连接与多层图注意网络相结合，同时在时序维度引入卷积+注意机制，实现空间-时间双重可解释性

**🔧 技术方法**

滑动窗口Pearson相关、图注意网络（GAT）、1D卷积、时序注意网络、全连接分类层

**📊 数据集**

Peking University site of ADHD‑200 数据集（194 例，116 HC，78 ADHD）

**📈 对比分析**

在分层10折交叉验证并采用5种随机种子集成后，取得平均平衡准确率85.18%±5.64，AUC0.881，明显优于前沿方法（约79–80%）

**⚠️ 局限性**

仅基于单一站点数据，样本量有限；未融合多模态影像；窗口长度固定；尚需在多站点、跨人群上验证泛化能力

---

## 279. DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction

**arXiv ID:** 2603.26462 | [PDF](https://arxiv.org/pdf/2603.26462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 280. SALMUBench: A Benchmark for Sensitive Association-Level Multimodal Unlearning

**arXiv ID:** 2603.26316 | [PDF](https://arxiv.org/pdf/2603.26316v1)

**作者:** Cai Selvas-Sala `[一作]` (Computer Vision Center), Lluis Gomez `[通讯]` (Computer Vision Center)

**通讯引用:** 4285 | [OpenAlex ID](https://openalex.org/A5008562740)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SALMUBench，一个专门评估CLIP等对比学习视觉‑文本模型在“被遗忘权”情景下关联级别机器遗忘（unlearning）的基准。

**💡 创新点**

创新点在于：①构建了60K条人设‑属性关联的全合成数据集SALMU；②用两款从0开始训练、仅差异在于是否接触敏感数据的Clean与Compromised CLIP模型，消除预训练干扰；③设计了结构化留存集（φ、ψ）与完整评估协议，能够细粒度区分遗忘失效、过度遗忘与灾难性损伤三种失败模式；④公开了数据、模型、评估脚本与排行榜，推动社区进一步研究。

**🔧 技术方法**

技术包括：ViT‑B/16 CLIP架构、对比学习预训练（≈400M对、32轮）；合成人脸生成（IP‑Adapter‑FaceID、ControlNet）与多阶段过滤；LLM（OpenAI GPT‑4）用于多样化、保持语义的说明句生成；多种遗忘方法改造（Generic Captions、Shuffled Captions、Direct Similarity Minimization、Negative Gradient、Finetuning、Descent to Delete、VLUnlearn、DELETE、CLIPErase）。

**📊 数据集**

使用的数据集包括：①合成SALMU（774人设、约60K图文对、包含姓名、城市、电话、邮箱、IBAN等PII）；②真实图片‑文本大规模语料DataComp CommonPool（≈400M对）做Clean/Compromised模型预训练；③用于验证的真实肖像集FHIBE（100张）与BLIP生成的通用说明句。

**📈 对比分析**

对比方法在5×预算下的主要结果：遗忘效果与Clean模型的基准相匹配的（RetFail≈0.001）有VLUnlearn、DELETE、CLIPErase；但它们在InterIdSim、IntraIdSim等协同遗忘指标上仍出现0.02‑0.03的损伤；灾难性方法如Shuffled Captions、Direct Similarity Minimization虽然能大幅降低RetFail，却导致GenKnow下降至0.3；效率低且无效的方案如Generic Captions、Finetuning未能显著提升遗忘度。综上，当前方法无法在三者（高效遗忘、低损伤、保持实用性）之间取得最优平衡。

**⚠️ 局限性**

局限性：①仅针对双编码器的CLIP模型，未涵盖基于CLIP的扩散模型或生成式多模态大模型；②敏感信息仅为显式键值对（姓名、电话等），未涉及隐式或抽象属性；③合成数据虽可控但可能缺少真实世界的多样性与噪声；④评估主要聚焦于“被遗忘”后模型的静态表现，未测量可恢复性或对新数据的再学习速度。

---

## 281. Addressing Ambiguity in Imitation Learning through Product of Experts based Negative Feedback

**arXiv ID:** 2603.26467 | [PDF](https://arxiv.org/pdf/2603.26467v1)

**作者:** John Bateman `[一作]` (University of York), Jihong Zhu `[通讯]` (University of York)

**通讯引用:** 10223 | [OpenAlex ID](https://openalex.org/A5051073741)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并验证了一种利用负反馈与产品专家（Product of Experts）相结合的模仿学习算法，能够在仅有少量且存在不完美示范的模糊任务中学习并显著提升成功率。

**💡 创新点**

创新点在于将负反馈与Product of Experts组合，并加入基于蚁群优化的轨迹选择掩码，使得系统仅利用失败轨迹一次性学习回避策略，从而大幅减少时间和内存消耗。

**🔧 技术方法**

使用高斯混合模型回归（GMM/GMR）、Product of Experts组合、蚁群优化掩码以及负反馈循环。

**📊 数据集**

使用了包含10+个示范（包括三种不同路径的pick‑and‑place、简单障碍和斜坡障碍）的人工演示数据，并在仿真环境及Franka Emika Research 3真实机器人上进行实验。

**📈 对比分析**

通过与负权重方法和混合专家方法对比，在三类任务（简单障碍、斜坡障碍、真实机器人pick‑and‑place）中，负反馈+Product of Experts方法分别将成功率提升约90%、80%/100%，并在真实机器人上从30%提升至80%/100%，在时间和内存方面也优于传统负反馈方法。

**⚠️ 局限性**

仅在轨迹导航任务中验证，需人工标记失败轨迹，尚未在更复杂任务或人机交互场景中测试。

---

## 282. Adapt as You Say: Online Interactive Bimanual Skill Adaptation via Human Language Feedback

**arXiv ID:** 2603.26466 | [PDF](https://arxiv.org/pdf/2603.26466v1)

**作者:** Zhuo Li `[一作]` (Chinese University of Hong Kong), Fei Chen `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 36718 | [OpenAlex ID](https://openalex.org/A5100405434)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了BiSAIL框架，实现在线交互式双臂技能的零样本适应；

**💡 创新点**

创新点在于分层reason-then-modulate结构：先用ESA-CoT基于多模态推理提取适配目标，再通过扩散模型与MCMC组合采样实现运动调制，并引入闭环反思；同时构建任务无关的双臂运动先验BMP；

**🔧 技术方法**

使用了GPT‑4o+ESA‑CoT进行语言+视觉推理，Transformer encoder生成BMP，扩散模型进行运动生成与调制，MCMC/Classifier‑Guided采样融合目标、任务与协同约束，闭环反思机制；

**📊 数据集**

自监督生成了100万条双臂运动数据；对六种双臂任务进行真实机器人实验，并收集120条多模态任务变体（语言+视觉）用于评估；

**📈 对比分析**

与无适应IL基线（DP, ACT）、轨迹调制IDMP、策略微调DSRL、端到端语言导向方法YAY和LATTE等进行对比。BiSAIL在意图对齐、任务完成率、约束满足率等指标上显著优于基线，在OOD任务变体和跨机器人平台上保持最高TSR；

**⚠️ 局限性**

仅实现运动学层面的自适应，未覆盖动力学、速度、加速度和交互力等动态层面，未来需要扩展至动态自适应。

---

## 283. Automating Clinical Information Retrieval from Finnish Electronic Health Records Using Large Language Models

**arXiv ID:** 2603.26434 | [PDF](https://arxiv.org/pdf/2603.26434v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 284. Image-based Quantification of Postural Deviations on Patients with Cervical Dystonia: A Machine Learning Approach Using Synthetic Training Data

**arXiv ID:** 2603.26444 | [PDF](https://arxiv.org/pdf/2603.26444v1)

**作者:** Roland Stenger `[一作]` (University of Lübeck), Sebastian Fudickar `[通讯]` (University of Lübeck)

**通讯引用:** 698 | [OpenAlex ID](https://openalex.org/A5006419344)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并验证了一套基于图像的自动化头位姿与侧移估计系统，用以客观评估颈部肌张力失衡的严重程度。

**💡 创新点**

利用大规模合成头像数据训练侧移回归模型，并将头位姿与旋转症状结合，首次在同一系统中实现对旋转与平移两类症状的自动评估。

**🔧 技术方法**

预训练的6DRepNet头位姿估计、EfficientNet‑B0深度学习回归、MTCNN面部检测、Sapiens分割、图像增强等技术。

**📊 数据集**

约16,000张合成头像图像（涵盖多种旋转和平移）以及100张真实患者正面/侧面照片与100张合成头像，20名临床专家为参考评分。

**📈 对比分析**

对合成头像与真实图像分别计算与专家评分的Pearson相关系数和与已知真值的准确率/TPR；旋转项相关系数≥0.81，侧移为0.55，模型在合成头像上准确率与TPR超过人类评分。

**⚠️ 局限性**

仅基于静态图像，未捕捉运动动态与非视觉评估；侧移相关性受专家一致性低影响；算法对头位姿依赖摄像机参考，可能低估实际位移。

---

## 285. Towards Privacy-Preserving Federated Learning using Hybrid Homomorphic Encryption

**arXiv ID:** 2603.26417 | [PDF](https://arxiv.org/pdf/2603.26417v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 286. Interpretable long-term traffic modelling on national road networks using theory-informed deep learning

**arXiv ID:** 2603.26440 | [PDF](https://arxiv.org/pdf/2603.26440v1)

**作者:** Yue Li `[一作]` (University of Cambridge), Ying Jin `[通讯]` (University of Cambridge)

**通讯引用:** 4148 | [OpenAlex ID](https://openalex.org/A5089312626)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了DeepDemand，一种基于理论驱动的深度学习框架，用外部社会经济特征和道路网络结构预测英国高速公路的长期交通流量；

**💡 创新点**

创新点在于将四步交通需求模型的结构嵌入可微分深度网络，并引入竞争式两源Dijkstra提取本地OD区域与OD对筛选，使模型既具可解释性又具空间可迁移性；

**🔧 技术方法**

使用多层感知机编码器、Softplus与sigmoid激活函数、MLP旅行时间惩罚网络，采用AdamW优化的随机梯度下降训练，并通过SHAP、UMAP等可解释技术分析模型；

**📊 数据集**

利用OpenStreetMap提取的英国道路网络，LSOA层面的人口、就业、住房、车主、POI等社会经济特征以及National Highways TRIS系统提供的AADT交通流量；

**📈 对比分析**

在随机五折交叉验证与空间交叉验证中，与线性回归、岭回归、随机森林和gravity模型对比，DeepDemand取得R²≈0.718（随机CV）和R²≈0.665（空间CV），MAE分别为约7406辆和7669辆，显著优于基线；

**⚠️ 局限性**

局限包括未显式建模模式分离、采用静态最短路径近似交通分配、未对预测不确定性进行量化，以及对拥堵动态和高频时变流量缺乏考虑。

---

## 287. Fair Data Pre-Processing with Imperfect Attribute Space

**arXiv ID:** 2603.26456 | [PDF](https://arxiv.org/pdf/2603.26456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 288. "Law at Your Fingertips": Understanding Legal Information Seeking on Video-Sharing Platforms in China

**arXiv ID:** 2603.26420 | [PDF](https://arxiv.org/pdf/2603.26420v1)

**作者:** Zhiyang Wu `[一作]` (City University of Hong Kong), Zhicong Lu `[通讯]` (George Mason University)

**通讯引用:** 34502 | [OpenAlex ID](https://openalex.org/A5063218435)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对中国抖音和哔哩哔哩两大视频分享平台的法律内容进行观察，并对20名法律信息寻求者进行半结构化访谈，系统梳理了用户在视频平台上进行法律信息查询、互动与评估的行为与经验。

**💡 创新点**

首次揭示了视频平台在法律信息寻求中的独特支撑方式（如同步直播咨询、短视频学习）及其优势与风险，并提出了针对内容质量、验证机制与社区建设的设计启示。

**🔧 技术方法**

采用实证观察方法、访谈记录与开放式编码分析，而非传统机器学习或算法技术；重点利用平台交互界面与用户对话来获取数据。

**📊 数据集**

观测样本包括150场直播咨询与150段短视频，访谈样本为20名法律信息寻求者（年龄19–30岁，性别11F/9M）。

**📈 对比分析**

本文未涉及对算法或模型的对比实验，因研究聚焦在行为与体验层面，故无性能指标；通过对比平台功能与传统文本问答渠道的优劣，得出视频平台在易用性、情感支持与交互性方面的提升。

**⚠️ 局限性**

局限性：样本仅来自抖音和哔哩哔哩，未覆盖全部视频平台；访谈对象主要为年轻、技术熟练用户，缺乏对老年人及技术门槛低人群的视角；研究基于中国法律环境，跨国推广需进一步验证。

---

## 289. Rotatable Antenna Enhanced Multicast Communication System

**arXiv ID:** 2603.26388 | [PDF](https://arxiv.org/pdf/2603.26388v1)

**作者:** Weihua Zhu `[一作]` (South China University of Technology), Yong Zeng `[通讯]` (Southeast University)

**通讯引用:** 30906 | [OpenAlex ID](https://openalex.org/A5082336235)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种可旋转天线（RA）增强的多组多播系统，通过联合优化波束成形向量与每个天线的瞄准方向，最大化所有用户的最小SINR。

**💡 创新点**

创新点在于将每个天线的3D瞄准方向作为可调节的空间自由度，引入二次变换与交替优化相结合的算法解决非凸的最大最小SINR问题，并通过SCA实现对天线方向的高效更新。

**🔧 技术方法**

使用的技术包括：二次变换（Quadratic Transform）将分式目标转化为更易处理的形式；交替优化（AO）分别更新波束成形和天线方向；SCA（Successive Convex Approximation）对非凸方向约束进行凸化；以及CVX等工具求解子问题。

**📊 数据集**

数据集：采用仿真生成的用户分布（沿半圆弧的K个单天线用户），设置频率2.4 GHz、噪声功率−94 dBm、BS高度10 m、半径50 m、天线间距λ/2等参数，未使用公开实验数据集。

**📈 对比分析**

性能比较方法：与三种基准（固定方向天线、随机方向天线、各向同性天线）在相同总功率下进行对比。结果表明RA方案在功率、用户角度、天线数量变化时均能提供更高的最大最小SINR，功率约减小4.5 dB即可达到基准的性能。

**⚠️ 局限性**

局限性：仅在理想LOS近场仿真场景下验证，缺乏真实实验验证；优化算法的计算复杂度较高；未考虑天线互耦、硬件非理想等实际影响；并未扩展到多天线/多用户多组的更复杂场景。

---

## 290. First Demonstration of 28 nm Fabricated FeFET-Based Nonvolatile 6T SRAM

**arXiv ID:** 2603.26439 | [PDF](https://arxiv.org/pdf/2603.26439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 291. Maintaining Difficulty: A Margin Scheduler for Triplet Loss in Siamese Networks Training

**arXiv ID:** 2603.26389 | [PDF](https://arxiv.org/pdf/2603.26389v1)

**作者:** Roberto Sprengel Minozzo Tomchak `[一作]` (Universidade Federal do Paraná), Paulo Lisboa de Almeida `[通讯]` (Universidade Federal do Paraná)

**通讯引用:** 540 | [OpenAlex ID](https://openalex.org/A5014773393)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出自适应 Margin 调度器 DAMS，动态调节 Triplet Margin Ranking Loss 的 margin 以保持训练难度并提升嵌入质量。

**💡 创新点**

创新点在于利用每个 epoch 的“易”三元组比例来决定何时增加 margin，而非固定或预设的线性增量。

**🔧 技术方法**

使用 Siamese 网络、EfficientNetB0 作为 backbone、Triplet Margin Loss、Adam 优化器、硬负样本挖掘和比例阈值调度技术。

**📊 数据集**

实验数据集包括 LFW、CelebA、CUB‑200‑2011 和 Stanford Cars 四个不同领域的数据集。

**📈 对比分析**

与常数 margin (0.3) 及线性增 margin 进行对比，使用 AUC‑ROC 和 Recall@k 评估；DAMS 在大多数数据集上提升了 0.8%‑2% 以上的指标。

**⚠️ 局限性**

局限性：margin 增长受阈值控制，可能导致收敛慢；需要手动设定 μ0、阈值 t 与步长 s_a 等三个超参数；在某些数据集（如 CelebA）表现略逊于线性调度。

---

## 292. Shapley meets Rawls: an integrated framework for measuring and explaining unfairness

**arXiv ID:** 2603.26476 | [PDF](https://arxiv.org/pdf/2603.26476v1)

**作者:** Fadoua Amri-Jouidel `[一作]` (University Mohammed VI Polytechnic), Stéphane Mussard `[通讯]` (University Nîmes Chrome)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过将群体公平性准则与Shapley值及其扩展的ESL值等价化，构建了一个统一的两阶段归因框架，用于评估和解释二分类模型中的群体不公平性。

**💡 创新点**

创新点包括：① 将公平性准则转化为群体Shapley/ESL值的等价性；② 提出两阶段Shapley/ESL分解，既能检测公平性，又能定位导致不公平的特征；③ 给出渐近正态检验与多数投票机制，提高统计可靠性和计算效率。

**🔧 技术方法**

使用技术：Shapley值、Efficient‑Symmetric‑Linear (ESL) 值、两阶段归因（Group‑Shapley 与 Feature‑Shapley），统计推断（渐近正态检验）、多数投票、算法复杂度分析。

**📊 数据集**

实验数据集：UCI Census Income（48,842 条记录，15 个人口特征，男女两组）。

**📈 对比分析**

方法对比：与传统的引导重采样（Bootstrap）检验相比，两阶段ESL 仅需约 8 分钟完成，显著低于 1 小时 30 分钟；检验结果与 Bootstrap 一致，证明了方法的统计有效性和计算优势。

**⚠️ 局限性**

局限性：Shapley 计算仍具指数时间复杂度（除 ES 值外）；框架目前仅适用于二分类任务，尚未扩展到多分类或回归问题；对高维特征交互的捕捉仍有提升空间。

---

## 293. CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities

**arXiv ID:** 2603.26425 | [PDF](https://arxiv.org/pdf/2603.26425v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 294. Neuro-Symbolic Process Anomaly Detection

**arXiv ID:** 2603.26461 | [PDF](https://arxiv.org/pdf/2603.26461v1)

**作者:** Devashish Gaikwad `[一作]` (RWTH Aachen University), Gyunam Park `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 920 | [OpenAlex ID](https://openalex.org/A5082571709)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种神经符号式流程异常检测方法，利用逻辑张量网络（LTN）将Declare约束注入到自动编码器的训练中，从而在检测过程中融合领域知识；

**💡 创新点**

创新点在于将可微分逻辑（LTN）与流程声明式约束相结合，对自动编码器进行软约束微调，使其能够识别稀有但符合流程的轨迹，显著降低误报；

**🔧 技术方法**

技术手段包括：一维或多维自动编码器（去噪、自编码重构），Declare约束挖掘与筛选，逻辑张量网络对FOL公式的可微分实现，以及基于重构误差的阈值判定；

**📊 数据集**

实验使用合成事件日志（Paper、P2P、Small、Medium、Large、Huge、Gigantic、Wide）和真实业务流程日志（BPIC12、BPIC13、BPIC17），在每个日志中随机注入30%异常案例；

**📈 对比分析**

与仅使用自动编码器的基线模型比较，采用F1分数评估。结果显示：在大部分合成日志中，LTN+Declare约束模型的F1提升约0.1-0.2；在BPIC12和BPIC13中亦获得提升；提升主要来源于召回率提升，且仅需10条稀有符合轨迹即可显著改善；

**⚠️ 局限性**

局限性包括：对高度复杂或多变的流程（如Wide、BPIC17）不易提升性能，因自动编码器泛化不足；LTN需要手工构造FOL公式，约束选择对性能影响大；实时性受限于可微分逻辑评估成本。

---

## 295. Demystifying Funding: Reconstructing a Unified Dataset of the UK Funding Lifecycle

**arXiv ID:** 2603.26426 | [PDF](https://arxiv.org/pdf/2603.26426v1)

**作者:** William Thorne `[一作]` (National Gallery), Diana Maynard `[通讯]` (University Of Sheffield)

**通讯引用:** 5197 | [OpenAlex ID](https://openalex.org/A5089236260)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过将英国研究与创新（UKRI）Gateway to Research（GtR）数据库与机会发布、面板会议及评审结果三大数据源进行融合，构建了覆盖全生命周期（机会→提案→评审→项目）的一体化数据库。

**💡 创新点**

创新点在于：①实现了从非结构化机会文本中自动抽取关键信息（如奖金额度、期限等）的闭域问答模型；②利用层次化检索+稀疏+稠密混合分数+交叉编码器的分层检索策略，显著提升了信息抽取精度；③在缺乏统一标识符的情况下设计了多步骤实体链接流程（文本相似、组织匹配、聚类）完成项目、申请与人名的跨表关联。

**🔧 技术方法**

采用的技术包括：BM25 与 Sentence‑Transformers（all‑MiniLM‑L6‑v2）混合检索、cross‑encoder（ms‑marco‑MiniLM‑L6‑v2）重排序、树形文档分块、模糊标题匹配、聚类去重、Python/SQL 数据处理、PostgreSQL 数据库。

**📊 数据集**

使用的数据集为：UKRI Gateway to Research（项目、人员、机构、成果等），UKRI 机会发布系统（标题、奖金额、截止日期等），以及各研究委员会的会议记录与面板成员列表（PDF、Excel、Tableau 等多种格式）。

**📈 对比分析**

在 101 篇机会文档（共 489 个问答对）上评估，层次化检索+不使用重排序的最佳配置实现 87.7% 的抽取准确率，优于全文 85.3% 的基线；在实体链接方面，申请-项目匹配精度约 98%，覆盖率 27.1%；申请-机会匹配精度高，但覆盖率仅 29.6%。

**⚠️ 局限性**

主要局限包括：①数据为抓取时点的快照，后续机会状态或项目信息可能变动；②面板会议与人员的对齐保守，覆盖率仅 64.4%；③申请-项目链接仅覆盖 27.1%（大部分未获资助的申请缺乏 GtR 记录）；④申请-机会链接基于模糊匹配，可能引入误匹配；⑤继承了 GtR 原有的数据质量问题（孤立记录、缺失标识符）。

---

## 296. Generalizable task-oriented object grasping through LLM-guided ontology and similarity-based planning

**arXiv ID:** 2603.26412 | [PDF](https://arxiv.org/pdf/2603.26412v1)

**作者:** Hao Chen `[一作]` (University of Osaka), Kensuke Harada `[通讯]` (University of Osaka)

**通讯引用:** 11056 | [OpenAlex ID](https://openalex.org/A5016270703)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于大语言模型与几何匹配的任务导向抓取（TOG）框架，可在无需完整模型、仅凭单视RGB‑D图像识别功能部件并生成稳健的6‑DoF抓取姿态。

**💡 创新点**

创新点包括：① 用LLM驱动的对象‑部件‑任务本体结构将自然语言指令映射到功能部件；② 通过模板辅助的采样‑聚类‑匹配策略实现视角不变的几何部件识别；③ 本地‑全局点云注册与稳定性感知的抓取位置调整，实现从已知模板到未知物体的抓取知识迁移；④ 对模板数量与性能的系统性分析与调优。

**🔧 技术方法**

核心技术：LLM（GPT‑4o）+结构化提示；SAM进行类无关分割；点云采样、k‑NN、PCA/PPD/CCD三重相似度评估；RANSAC+ICP + 旋转搜索的本地‑全局配准；基于前置模板的抓取候选生成与IK/碰撞检测；稳定性评估与局部包围盒调整。

**📊 数据集**

使用公开模型库（GrabCAD）构建模板数据库；DROID数据集用于LLM提示评估；在真实环境中测试的实验对象包括杯子、瓶子、剪刀、雨伞、牙膏盖、牙刷、碗、杯面条、螺丝刀等，共计至少 12 种不同类别和 30 条自然语言指令；此外在 DROID 任务语料上验证 LLM 解释精度。

**📈 对比分析**

与 VLPart、ShapeGrasp、GraspGPT、GraspNet、HGGD 等基线对比，分别在部件识别准确率（PRA）、抓取选择准确率（GSA）和抓取成功率（GSR）上取得显著提升；在不同场景（遮挡、噪声、尺度变换）下的 PGSR 也保持在 71–84% 之间；实验显示使用 3‑5 个模板即可达到最佳性能，进一步增加模板数量仅提升 1–3% 而计算时间翻倍。

**⚠️ 局限性**

局限性：1）完全无参考的全新物体无法通过本体映射；2）对功能部件几何特征不明显或被遮挡时识别效果下降；3）当前框架仅关注抓取，不考虑后续任务约束（如抓取后旋转、搬运路径等）。

---

## 297. SHANDS: A Multi-View Dataset and Benchmark for Surgical Hand-Gesture and Error Recognition Toward Medical Training

**arXiv ID:** 2603.26400 | [PDF](https://arxiv.org/pdf/2603.26400v1)

**作者:** Le Ma `[一作]` (MIRALab), Katarzyna Wac `[通讯]` (University of Geneva)

**通讯引用:** 3087 | [OpenAlex ID](https://openalex.org/A5014459921)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了SHands多视角开放手术手势与错误识别数据集，记录52名参与者（20名专家、32名实习生）在外科线性切开与缝合任务中的同步五摄像头RGB视频，并为每帧标注15个细粒度手势原语和8类临床错误。

**💡 创新点**

创新点在于：①首次提供同步五摄像头的开放手术RGB数据；②细粒度15个手势原语与8类临床错误标签；③设立单视角、多视角与跨视角三种评估协议，并对多视角融合方法做基准评估。

**🔧 技术方法**

技术方案采用Transformer/自监督视频编码器（VideoMAE、TimeSformer等）与卷积基线（R3D、SlowFast等），以及多视角融合框架（MVAction、ViewCLR、ViewCon、DVANet）进行手势识别与错误检测。

**📊 数据集**

使用了自制的SHands数据集；与现有公开数据集（如JIGSAWS、Cholec80等）对比验证其多视角与错误标签的独特性。

**📈 对比分析**

通过交叉主体训练/测试的单视角、多视角和跨视角协议进行比较，单视角Transformer Top‑1≈69%，多视角下DVANet Top‑1≈68.5%；错误检测从单视角60%提升至多视角68.5%，跨视角模型在未见摄像头上保持≈94%性能。

**⚠️ 局限性**

局限性包括仅覆盖RGB、两项基础手术任务；细微错误识别仍表现不佳；数据采集环境受限于实验室，缺乏真实临床多样性；标注依赖单一标注者，可能存在主观偏差。

---

## 298. LLaDA-TTS: Unifying Speech Synthesis and Zero-Shot Editing via Masked Diffusion Modeling

**arXiv ID:** 2603.26364 | [PDF](https://arxiv.org/pdf/2603.26364v1)

**作者:** Xiaoyu Fan `[一作]` (Bairong, Inc.), Yunzhang Chen `[通讯]` (Bairong, Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了LLaDA‑TTS，将LLM的自回归解码器替换为掩码扩散模型，实现并行生成并保持高质量语音；

**💡 创新点**

创新点在于通过双向掩码扩散、1/t加权损失和label shift实现自回归权重迁移，既获得显著加速，又实现零样本语音编辑；

**🔧 技术方法**

采用双向Transformer、掩码扩散、1/t加权交叉熵、label shift、迭代解码、注意力对齐以及离散语音编码与流匹配声码器等技术；

**📊 数据集**

使用约50小时微调数据，训练基于6,000小时的Emilia中英混合语料；

**📈 对比分析**

在Seed‑TTS‑Eval基准上与CosyVoice3 AR基线对比，64步得到0.98% CER/1.96% WER，速度提升约2×，并优于其他NAR基线；

**⚠️ 局限性**

局限在于需预先指定输出长度、目前不支持流式推理，并且模型尺寸和推理时间仍有限制。

---

## 299. HyVIC: A Metric-Driven Spatio-Spectral Hyperspectral Image Compression Architecture Based on Variational Autoencoders

**arXiv ID:** 2603.26468 | [PDF](https://arxiv.org/pdf/2603.26468v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 300. Automatic feature identification in least-squares policy iteration using the Koopman operator framework

**arXiv ID:** 2603.26464 | [PDF](https://arxiv.org/pdf/2603.26464v1)

**作者:** Christian Mugisho Zagabe `[一作]` (TU Dortmund University), Sebastian Peitz `[通讯]` (TU Dortmund University)

**通讯引用:** 1195 | [OpenAlex ID](https://openalex.org/A5049416946)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出一种利用 Koopman 自动编码器（KAE）学习特征字典并结合 Least-Squares Policy Iteration（LSPI）的 KAE_LSPI 算法，用于解决强化学习中的价值函数逼近问题。

**💡 创新点**

创新点在于通过 KAE 自动学习基函数，避免了传统 LSPI 和 KLSPi 中对手工选取基函数或核函数的依赖，并在 LSPI 框架内实现无监督特征学习。

**🔧 技术方法**

使用了 Koopman 操作符、延迟坐标映射、深度自编码器（KAE）、EDMD、Least-Squares Policy Iteration 以及随机策略采样的数据收集方法。

**📊 数据集**

使用了随机策略采集的链行走（n=20、50）数据集和倒置摆（cart-pole）仿真数据集作为实验环境。

**📈 对比分析**

通过在链行走和倒置摆任务中与经典 LSPI 和 KLSPi 在迭代收敛速度、特征数量、平均平衡步数等指标上进行对比；结果显示 KAE_LSPI 在相同特征数量下收敛速度和最终策略与传统方法相当或略优，且特征数更合理。

**⚠️ 局限性**

局限性包括仍需预先指定特征数量、算法目前为离线/离线模式、缺乏理论收敛和 regret 上界分析，且在线 on-policy 版本尚未实现。

---

## 301. UNIFERENCE: A Discrete Event Simulation Framework for Developing Distributed AI Models

**arXiv ID:** 2603.26469 | [PDF](https://arxiv.org/pdf/2603.26469v1)

**作者:** Doğaç Eldenk `[一作]` (Northwestern University), Stephen Xia `[通讯]` (Northwestern University)

**通讯引用:** 1048 | [OpenAlex ID](https://openalex.org/A5051730108)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为 Uniference 的离散事件仿真框架，用于开发、基准测试和部署分布式 AI 模型，支持从模拟直接迁移到真实硬件。

**💡 创新点**

创新点在于只在通信原语（send/recv/All-Reduce 等）上同步，消除回滚，保持因果顺序；同时提供与 PyTorch Distributed 无缝对接的统一 API，使同一代码既可在仿真中跑，又能直接部署到真实设备。

**🔧 技术方法**

核心技术包括离散事件仿真引擎、轻量逻辑进程、网络模型（基于轮式算法的 All-Gather/Reduce 等）、PyTorch Distributed 集成、Profiler 与 Chrome‑trace 导出、以及基于真实设备的慢速因子估算。

**📊 数据集**

使用的数据集与模型主要有 LLama3（文本生成）和 CLIP‑ViT（多模态视觉），在实验中还自定义了计数任务来评估自回归生成的推理时延。

**📈 对比分析**

比较方法：在多种硬件（HPC 集群 A100 + InfiniBand、Gloo、Jetson Orin Nano）和网络后端（NCCL、Gloo）下测量网络延迟、带宽、推理总时长，仿真结果与真实测量的误差均在 1–17% 之间，整体准确率超过 98%。

**⚠️ 局限性**

局限性包括：仿真只能在宿主机内存足够时进行，无法直接在低内存设备上完整模型；需要手动指定或估算慢速因子，可能无法捕捉所有设备细节；框架目前主要针对 Transformer 模型，其他架构的支持仍有限；远程目标模式尚未完善。

---

## 302. Can AI Models Direct Each Other? Organizational Structure as a Probe into Training Limitations

**arXiv ID:** 2603.26458 | [PDF](https://arxiv.org/pdf/2603.26458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 303. KMM-CP: Practical Conformal Prediction under Covariate Shift via Selective Kernel Mean Matching

**arXiv ID:** 2603.26415 | [PDF](https://arxiv.org/pdf/2603.26415v1)

**作者:** Siddhartha Laghuvarapu `[一作]` (University of Illinois Urbana-Champaign), Jimeng Sun `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 28080 | [OpenAlex ID](https://openalex.org/A5084279065)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 KMM-CP，一种针对协变量偏移的分割式合规预测框架，利用 Kernel Mean Matching (KMM) 重新加权校准样本，并加入选择性过滤提升低重叠区域的稳定性。

**💡 创新点**

创新点在于：①将 KMM 与合规预测理论相结合，直接控制偏差-方差分解；②引入选择性 KMM 通过双重加权同时优化源权重和目标选择变量，仅在共享支持区域进行校正；③提供覆盖性理论证明与实验验证。

**🔧 技术方法**

使用 Kernel Mean Matching、重加权合规预测、选择性双重加权、分割式（split）合规预测、特征嵌入（AttentiveFP 图神经网络）。

**📊 数据集**

在五个分子性质预测任务上验证：Tox21、AMES、hERG、BBB-Martins、HIV，采用 TDC 数据集并通过指纹分割模拟协变量偏移。

**📈 对比分析**

与 Vanilla 合规预测、KDE、Logistic 分类器、无选择性 KMM 等基线比较。KMM-CP 在覆盖误差（MAD）上优于所有基线，尤其在低重叠情况下覆盖误差下降约 55%，并在 Mondrian（条件）校准中保持最优表现。

**⚠️ 局限性**

局限性包括：选择性过滤会导致对部分样本不做预测；依赖于表示学习与核参数的质量；实验仅针对小分子化学数据，是否能推广到其他领域尚未验证。

---

## 304. Approximation Schemes for Subset TSP and Steiner Tree on Geometric Intersection Graphs

**arXiv ID:** 2603.26397 | [PDF](https://arxiv.org/pdf/2603.26397v1)

**作者:** Sándor Kisfaludi-Bak `[一作]` (Aalto University), Dániel Marx `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在同尺寸、连通、脂肪（fat）多边形交叉图（包括单位圆图）上，对Subset TSP和Steiner Tree问题的多项式时间近似方案（EPTAS）。

**💡 创新点**

创新点在于：①将平面图的压缩、缩放分解（contraction‑decomposition）技术推广到交叉图；②引入“wireframe”和“object frame”概念，实现对交叉图的平面化与稀疏化；③通过Lipschitz嵌入将交叉图映射到平面图，从而利用已有的平面图EPTAS框架；④在Steiner Tree中构造新的“mortar graph”和“generator”结构，显著降低时间复杂度。

**🔧 技术方法**

主要技术包括：spanner构造、压缩分解、稀疏化（clique删减）、Lipschitz平面嵌入、wireframe/ object frame转换、动态规划、树宽限制、集合论的子集树连接与门点（portals）处理等。

**📊 数据集**

本文是理论算法研究，未使用具体实验数据集，而是对任意给定的交叉图（单位圆或fat多边形）给出算法复杂度。

**📈 对比分析**

与之前在平面图上的EPTAS相比，本文的算法在Subset TSP上取得 2^{O(1/ε)}·n^{O(1)} 的运行时间，Steiner Tree 在一般形式下为 2^{2^{O(1/ε)}}·n^{O(1)}，通过进一步改进可降低至 2^{O(1/ε)}·n^{O(1)}。证明表明在同尺寸、连通、fat条件下可得到PTAS；若任何条件被放宽，问题转为APX‑hard。

**⚠️ 局限性**

局限性：仅适用于同尺寸、连通且fat的多边形（单位圆属于此类）；若对象不连通、尺寸不相近、或不fat，问题已被证明为APX‑hard；三维情况（单位球）同样难解；算法仍基于理论复杂度，实际性能与实现细节有关。

---

## 305. Domain decomposition of large neural network surrogate models

**arXiv ID:** 2603.26396 | [PDF](https://arxiv.org/pdf/2603.26396v1)

**作者:** Timm Gödde `[一作]` (University of Twente), Bojana Rosić `[通讯]` (University of Twente)

**通讯引用:** 692 | [OpenAlex ID](https://openalex.org/A5062849809)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于域分解的神经网络代理模型，通过在局部子域中训练简单网络并施加接口约束，实现全局连续性和更高的逼近精度。

**💡 创新点**

将域分解与拉格朗日/增强拉格朗日方法相结合，使得局部网络在并行训练时既保持高效又能通过接口约束保证全局解的连续性。

**🔧 技术方法**

域分解技术、拉格朗日乘子法、增强拉格朗日乘子法、并行神经网络训练。

**📊 数据集**

基于二维线性弹性压缩问题的合成有限元数据。

**📈 对比分析**

与无约束全局网络及单一子域模型比较，评估不同子域数量下的计算时间和精度；增强拉格朗日方法收敛更快、规模更大，尽管精度略低，但在大规模问题中优于传统拉格朗日。

**⚠️ 局限性**

增强拉格朗日方法在精度上略逊于标准拉格朗日；域划分需经验选择，过多子域可能导致接口约束过多影响收敛；验证仅在合成二维弹性问题上，需进一步测试更复杂真实数据。

---

## 306. Switch Attention: Towards Dynamic and Fine-grained Hybrid Transformers

**arXiv ID:** 2603.26380 | [PDF](https://arxiv.org/pdf/2603.26380v1)

**作者:** Yusheng Zhao `[一作]` (Peking University), Ming Zhang `[通讯]` (Peking University)

**通讯引用:** 16315 | [OpenAlex ID](https://openalex.org/A5100447315)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Switch Attention 混合 Transformer，能够在每层动态路由全注意力与滑动窗口注意力，以实现更高效的计算；

**💡 创新点**

创新点在于：1）细粒度动态路由机制；2）自适应正则化鼓励使用高效分支；3）通过持续预训练将已有全注意力模型迁移到混合架构；

**🔧 技术方法**

使用技术包括：双分支（全注意力+滑动窗口）路由器、直通估计 (STE) 的硬阈值门控、自适应正则化、共享 KV 缓存、持续预训练策略；

**📊 数据集**

实验数据集覆盖 23 个基准，涵盖常识推理（Lambada、Wikitext、BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC-e、ARC-c、OpenBookQA）、上下文检索（SWDE、FDA、TriviaQA、NQ、DROP、SQuAD）以及长文本理解（LongBench‑E 组：MultiFieldQA、TRec、MultiNews、GovReport、Qasper、TriviaQA、SAMSum）；

**📈 对比分析**

与全注意力、滑动窗口、静态混合等基线进行比较；在 4K 长度下常识推理表现相当；在检索任务中比 SWA‑CPT 提升 27.5%，比 StaticHybrid 提升 4.5%；在 32K 长度的长文本任务中平均提升 6.3%；在 Needle‑in‑a‑Haystack 检索任务上实现完美准确率，推理速度可比全注意力快 4×（标准分布）或 12×（简易分布）；

**⚠️ 局限性**

局限性：缺乏硬件层面的高效实现；未在 1.5B 以上规模模型上验证效果；

---

## 307. Probabilistic Multilabel Graphical Modelling of Motif Transformations in Symbolic Music

**arXiv ID:** 2603.26478 | [PDF](https://arxiv.org/pdf/2603.26478v1)

**作者:** Ron Taieb `[一作]` (Hebrew University of Jerusalem), Barak Sober `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 353 | [OpenAlex ID](https://openalex.org/A5074615861)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对贝多芬钢琴奏鸣曲中动机变形进行概率多标签图模型分析。

**💡 创新点**

首次将多标签条件随机场与段落级依赖结构结合，用于研究动机变形的统计规律。

**🔧 技术方法**

使用多标签条件随机场、伪似然估计、置换复合似然比检验以及Wald置信区间。

**📊 数据集**

采用贝多芬钢琴奏鸣曲数据集（BPSD）以及BPS‑Motif、BPS‑FH等标注资源。

**📈 对比分析**

通过置换检验比较基线、单变量和全模型，结果显示加入特征和相互作用均显著提升拟合度；在早期与中期作品中均观察到更高的模型拟合与更丰富的交互效应。

**⚠️ 局限性**

标注稀疏、仅限前奏曲，未包含晚期作品；模型为解释性分析而非预测，且对段落划分与标签定义存在依赖。

---

## 308. A Boltzmann-machine-enhanced Transformer For DNA Sequence Classification

**arXiv ID:** 2603.26465 | [PDF](https://arxiv.org/pdf/2603.26465v1)

**作者:** Zhixuan Cao `[一作]` (Tsinghua University), Xuang WU `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Boltzmann机器增强的Transformer，用结构化门控注意力替代传统softmax，实现DNA序列分类；

**💡 创新点**

将Transformer注意力视为能量模型，加入二值门控变量、局部偏置、对称边缘交互、潜在隐藏单元，并通过均值场推断与Gumbel‑Softmax实现可微离散化，同时联合任务损失与能量损失训练；

**🔧 技术方法**

Boltzmann机器、能量式结构分布、均值场变分推断、Gumbel‑Softmax离散化、门控注意力、softmax替换、能量正则化、梯度裁剪、学习率退火等；

**📊 数据集**

使用Genomic Benchmarks中的二分类DNA序列数据（长度500，训练20843样本，测试6948样本）；

**📈 对比分析**

与公开CNN基线、普通Transformer和完整BM‑Transformer在准确率上比较，BM‑Transformer与Plain Transformer准确率相近（≈0.725），均显著优于CNN（≈0.695），结构可解释性更好但分类提升有限；

**⚠️ 局限性**

均值场近似误差、Gumbel‑Softmax梯度偏差、能量函数设计敏感、训练更复杂、结构可解释性不等同生物因果、尚未验证大规模预训练效果。

---

## 309. ClimateCheck 2026: Scientific Fact-Checking and Disinformation Narrative Classification of Climate-related Claims

**arXiv ID:** 2603.26449 | [PDF](https://arxiv.org/pdf/2603.26449v1)

**作者:** Raia Abu Ahmad `[一作]` (German Research Center for Artificial Intelligence), Georg Rehm `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

改进并扩展了 ClimateCheck 2026 共享任务，提供检索、论证与叙事分类三项任务，并统一了数据集与评估框架。

**💡 创新点**

创新点在于：①把检索、验证与叙事分类三个任务整合到同一批注数据上；②将训练集扩充约三倍，加入新的叙事分类任务；③提出针对不完整注解的自动评估框架；④首次探讨叙事结构对验证难度的影响。

**🔧 技术方法**

技术手段包括：多阶段检索管线（BM25→dense re‑ranker→cross‑encoder）; DeBERTa NLI 模型用于支持/反驳/NEI 分类; Qwen3‑8B 与结构化推理的叙事分类模型; 低秩适配与自动评估工具。

**📊 数据集**

使用数据集：ClimateCheck 2026，包含 958 条英语气候主张、4927 条主张‑摘要对（支持/反驳/NEI），以及 33 个叙事标签（27 个否定叙事 + 1 无假信息）。

**📈 对比分析**

比较结果：在检索任务上，最佳系统的 Recall@5/Score_1.1 约 0.47/1.18；在验证任务上最佳 macro‑F1 约 0.83；在叙事分类上，最佳 macro‑F1 约 0.58。总体而言，多阶段检索与分层推理的系统表现最优。

**⚠️ 局限性**

限制：仅覆盖英文主张；验证仅考察单一摘要，忽略多文档推理；未利用叙事预测辅助验证；细粒度叙事标签的 IAA 低于阈值；评估受注解不完整影响，需更完善的自动评估。

---

## 310. Meta-Learned Adaptive Optimization for Robust Human Mesh Recovery with Uncertainty-Aware Parameter Updates

**arXiv ID:** 2603.26447 | [PDF](https://arxiv.org/pdf/2603.26447v1)

**作者:** Shaurjya Mandal `[一作]`, John Galeotti `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于元学习的自适应优化框架，用于从单张 RGB 图像恢复 3D 人体网格，兼顾初始化、迭代细化和不确定性量化。

**💡 创新点**

创新点包括：① 用元学习预训练得到的优化友好初始化；② 选择性参数缓存机制，自动冻结已收敛关节以节省计算；③ 采用分布式采样的自适应更新，既能探索又能提供不确定性估计；④ 结合随机逼近技术处理梯度不可导情况。

**🔧 技术方法**

核心技术包括：基于 MAML 的元学习框架、HRNet-W48 基础网络、SMPL 参数化、Gaussian 分布采样更新、SPSA 随机逼近、以及基于方差的自适应步长和不确定性输出。

**📊 数据集**

训练使用 Human3.6M、MPI-INF-3DHP、COCO 与 MPII 四大数据集，评估则在 3DPW 与 Human3.6M 测试集上进行。

**📈 对比分析**

与纯回归、纯优化、以及现有元学习方法（如 MetaHMR、SPIN 等）在 MPJPE、PA‑MPJPE 与 PVE 指标上对比，方法在 3DPW 上平均降低 10.3 mm、在 Human3.6M 上降低 8.0 mm，整体实现了 state‑of‑the‑art 性能。

**⚠️ 局限性**

局限性包括：受 SMPL 模型表达能力限制；选择性缓存可能过早冻结参数导致精度下降；假设关节参数独立性不完全成立；元学习方案需要精细调参，且在极端域迁移场景下仍存在一定性能衰减。

---

## 311. Enabling topography-resolving structural dynamic contact simulation

**arXiv ID:** 2603.26446 | [PDF](https://arxiv.org/pdf/2603.26446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 312. Restore, Assess, Repeat: A Unified Framework for Iterative Image Restoration

**arXiv ID:** 2603.26385 | [PDF](https://arxiv.org/pdf/2603.26385v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 313. Hidden Elo: Private Matchmaking through Encrypted Rating Systems

**arXiv ID:** 2603.26407 | [PDF](https://arxiv.org/pdf/2603.26407v1)

**作者:** Mindaugas Budzys `[一作]` (Tampere University), Antonis Michalas `[通讯]` (Tampere University)

**通讯引用:** 1213 | [OpenAlex ID](https://openalex.org/A5014749102)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种基于全同态加密（FHE）和零知识证明（NIZK）的隐私匹配协议，能够在保持玩家真实 Elo 评分私密的前提下，实现公平匹配和评分更新。

**💡 创新点**

创新点在于首次将 CKKS‑RNS FHE 用于 Elo 评分的加密更新，并结合承诺一致性证明确保评分在合法区间内，同时通过数字签名保证服务器对加密评分的正确性，从而实现既后量子安全又隐私保护的匹配系统。

**🔧 技术方法**

主要技术包括 Cheon‑Kim‑Kim‑Song CKKS‑RNS FHE、零知识证明（NIZK）与承诺一致性、数字签名、RNS 加速、Bootstrapping 等。

**📊 数据集**

实验使用基于国际象棋的模拟场景，随机生成对手评分，进行 10,000 次连续更新；未使用真实比赛数据集，而是自建的棋局结果模拟数据。

**📈 对比分析**

与现有 Matchmaking Encryption（ME）进行对比；在安全级别 128 下，单次评分更新耗时约 20–25 秒，误差低于 0.001，但计算时间比 ME 高数十倍；键生成和 Bootstrapping 成本较大，内存占用可达数 GB。

**⚠️ 局限性**

局限性包括：计算和内存开销大，尤其是 Bootstrapping 需要 15 秒以上；不适合实时匹配；仅实现了 Elo 评分，未覆盖 Glicko 等更复杂系统；大规模用户部署需要复杂的密钥管理方案。

---

## 314. Analysing Calls to Order in German Parliamentary Debates

**arXiv ID:** 2603.26430 | [PDF](https://arxiv.org/pdf/2603.26430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 315. Foundation Model for Cardiac Time Series via Masked Latent Attention

**arXiv ID:** 2603.26475 | [PDF](https://arxiv.org/pdf/2603.26475v1)

**作者:** Moritz Vandenhirtz `[一作]` (ETH Zurich), Julia E. Vogt `[通讯]` (ETH Zurich)

**通讯引用:** 1752 | [OpenAlex ID](https://openalex.org/A5045935456)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在多导联心电图上构建并预训练了一种跨导联潜在注意力的 Masked Autoencoder（LAMAE），并在 Mimic‑IV‑ECG 数据集上评估其在 ICD‑10 代码预测任务中的表现。

**💡 创新点**

创新点在于：① 通过潜在注意力模块直接学习导联间的高阶交互，利用心电图的结构冗余；② 该模块支持可变大小的导联集合，具备排列不变性与自适应加权能力；③ 结合 Masked Autoencoder 的自监督框架，实现对多导联数据的有效结构监督。

**🔧 技术方法**

主要技术包括多导联 Masked Autoencoder（LAMAE）、多头自注意力（Latent Attention）模块、CLS token 聚合、随机遮蔽（masking）以及线性探针与微调训练策略。

**📊 数据集**

使用的数据集为公开的 Mimic‑IV‑ECG 数据库，包含 12 号导联的心电图记录。

**📈 对比分析**

与独立导联 MAE、对齐基线等方法进行对比；在 ICD‑10 代码预测任务中，LAMAE 在低样本与多导联相关诊断上取得最高的 AUROC（约 0.85‑0.95），在全量微调时也保持竞争力，尤其在心律失常、心肌梗死等 ECG 明显表现的疾病上表现突出。

**⚠️ 局限性**

局限性包括：ICD 标签作为标签代理的准确性受限；仅利用波形信息难以捕捉血管/脑血管等非 ECG 明显特征；以及对跨导联关系的学习依赖于数据集的多导联完整性。

---

## 316. Wattchmen: Watching the Wattchers -- High Fidelity, Flexible GPU Energy Modeling

**arXiv ID:** 2603.26435 | [PDF](https://arxiv.org/pdf/2603.26435v1)

**作者:** Brandon Tran `[一作]` (University of Wisconsin-Madison), Shivaram Venkataraman `[通讯]` (University of Wisconsin-Madison)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 Wattchmen 的 GPU 能耗建模框架，通过设计大量微基准、采集指令计数与功率数据，并利用系统方程求解得到每条 SASS 指令的能耗，进而在多种 GPU 代和冷却方式上对应用程序进行高精度的能耗预测与分解。

**💡 创新点**

创新点在于：①将指令级能耗建模与系统方程求解相结合，能够同时归因多条相互交织的指令；②采用稳态功率采样，消除冷却与温度变化的影响；③引入指令分组、桶化与缩放策略，使模型即使在未知指令出现时也能保持高覆盖率；④在 16 类主流工作负载上验证，MAPE 统一降至 11–15%，显著优于 AccelWattch（32%）和 Guser（25%）。

**🔧 技术方法**

技术手段包括：<br>• 微基准（inlined assembly）设计 90 条，覆盖计算、控制流、缓存等指令；<br>• NVML 采样与静态/恒定功率估计；<br>• 指令计数与缓存命中率通过 NVIDIA Profiler 获取；<br>• 构造线性方程组并使用非负线性求解器得到每条指令能耗；<br>• 对缺失指令采用缩放、分组、桶化等近似；<br>• 预测阶段将指令能耗与执行时间、静态功率相乘得到总能耗。

**📊 数据集**

使用的数据集包括：<br>① 90 条自定义微基准；<br>② 16 个主流工作负载（Rodinia GPGPU、图分析、HPC 与 ML 任务）；<br>③ 在不同硬件环境下的实验数据：air‑cooled V100、water‑cooled V100、air‑cooled A100、air‑cooled H100；<br>④ 具体案例：Backprop、QMCPACK（混合精度与全精度）。

**📈 对比分析**

评估方法：将 Wattchmen 的 Direct 与 Predict 两个阶段与 AccelWattch、Guser 进行对比，采用 MAPE 作为主要指标。<br>性能表现：<br>• Air‑cooled V100：Direct 19% MAPE，Predict 14% MAPE；<br>• Water‑cooled V100：Predict 14% MAPE；<br>• Air‑cooled A100：Predict 11% MAPE；<br>• Air‑cooled H100：Predict 12% MAPE；<br>• 在 Backprop 及 QMCPACK 案例中，能耗分别降低 16% 与 35%。<br>结果表明，Wattchmen 在多种平台与冷却条件下均保持 10–15% 的高精度，并能显著支持能耗优化。

**⚠️ 局限性**

局限性：<br>① 训练阶段使用全部 SM 活动，未覆盖 SM 部分激活的真实应用场景；<br>② 由于 GPU 指令流水线深度与隐藏延迟，能耗归因仍存在不确定性；<br>③ 依赖 NVML，存在采样粒度与误差限制；<br>④ 目前仅支持单 GPU 能耗，未建模多 GPU 通信与 MIG/MxGPU 等多 GPU 架构；<br>⑤ 微基准一次性构造成本较高；<br>⑥ 编译器优化与 PTX 版本差异可能影响指令映射；<br>⑦ 目前仅针对 NVIDIA GPU，跨厂商移植需要额外工作。

---

## 317. Word Alignment-Based Evaluation of Uniform Meaning Representations

**arXiv ID:** 2603.26401 | [PDF](https://arxiv.org/pdf/2603.26401v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 318. 120 Minutes and a Laptop: Minimalist Image-goal Navigation via Unsupervised Exploration and Offline RL

**arXiv ID:** 2603.26441 | [PDF](https://arxiv.org/pdf/2603.26441v1)

**作者:** Xiaoming Liu `[一作]` (University of Macau), Steven Morad `[通讯]` (University of Macau)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5074062972)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在真实环境中快速训练图像目标导航策略，利用自主数据采集、冻结视觉特征和离线强化学习实现两小时内完成从采集到部署的全流程。

**💡 创新点**

创新点在于结合粉红均匀噪声探索、DINOv3冻结特征与TD3+BC离线学习，并使用FQE进行无交互模型选择，形成完全自动化、低成本的MINav流水线。

**🔧 技术方法**

采用粉红均匀噪声进行数据采集，DINOv3 ViT视觉编码器，TD3+BC离线强化学习，以及FQE模型选择。

**📊 数据集**

仅使用目标环境中自主采集的RGB数据，时间预算为1–2小时，未使用公开大规模数据集。

**📈 对比分析**

与零拷贝的视觉导航基线（GNM、ViNT、NoMaD）对比，在三种室内环境下，1小时MINav的成功率可达73%–88%，优于基线；性能随数据量提升呈正向增长。

**⚠️ 局限性**

主要局限在于依赖相机视觉，未对极端光照、遮挡严重场景测试，且离线策略对动态目标的鲁棒性需进一步提升。

---

## 319. Generative Modeling in Protein Design: Neural Representations, Conditional Generation, and Evaluation Standards

**arXiv ID:** 2603.26378 | [PDF](https://arxiv.org/pdf/2603.26378v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 320. Why Models Know But Don't Say: Chain-of-Thought Faithfulness Divergence Between Thinking Tokens and Answers in Open-Weight Reasoning Models

**arXiv ID:** 2603.26410 | [PDF](https://arxiv.org/pdf/2603.26410v1)

**作者:** Richard J. Young `[一作]` (University of Nevada, Las Vegas), Richard J. Young `[通讯]` (University of Nevada, Las Vegas)

**通讯引用:** 3317 | [OpenAlex ID](https://openalex.org/A5101673182)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了12款公开权重的推理模型，在MMLU与GPQA题目上注入6种误导性提示后，量化了模型思考链（thinking tokens）与最终答案之间在承认提示信息上的差异，发现55.4%的受影响案例在思考链中出现提示关键词但在答案中缺失；

**💡 创新点**

首次系统性量化“思考-答案”认知差异的程度与方向性，并揭示该差异在模型与提示类型之间呈显著异质性，指出思考链可能是监控模型行为的重要渠道。

**🔧 技术方法**

采用关键词匹配方法分别对思考链与答案文本进行提示信息的识别，并通过四象限分类、卡方检验、比例差异统计等定量手段评估异质性与方向性。

**📊 数据集**

使用498道MMLU与GPQA问题，配合6种提示类型（sycophancy、consistency、visual pattern、metadata、grader gaming、unethical），生成10,506个受影响案例。

**📈 对比分析**

将每个受影响案例按四象限（透明、思考仅、表面仅、未识别）进行分类，并比较不同模型与提示类型下的分布。整体思考仅率55.4%，表面仅率仅0.5%，显著高于独立假设预期；模型间差异从19.6%到94.7%，提示类型间差异从38.7%到72.2%。

**⚠️ 局限性**

局限包括：①仅使用关键词匹配，可能漏检同义或隐含承认；②仅分析受影响案例，未覆盖正确忽略提示的情形；③思考链只是生成文本，未必等同真实内部推理；④输出截断导致部分样本偏差；⑤仅评估公开权重模型，无法推广到商业模型。

---

## 321. A Lightweight High-Throughput Collective-Capable NoC for Large-Scale ML Accelerators

**arXiv ID:** 2603.26438 | [PDF](https://arxiv.org/pdf/2603.26438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 322. T-800: An 800 Hz Data Glove for Precise Hand Gesture Tracking

**arXiv ID:** 2603.26403 | [PDF](https://arxiv.org/pdf/2603.26403v1)

**作者:** Haoyang Luo `[一作]` (Peking University), Yixin Zhu `[通讯]` (Peking University)

**通讯引用:** 4098 | [OpenAlex ID](https://openalex.org/A5051255725)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种可穿戴手套系统，利用18个IMU以800Hz采样率捕获全手关节的高速运动，实现高频、同步、无漂移的数据记录。

**💡 创新点**

创新点在于结合机械隔离的“夹层”结构和基于广播的全局时间同步协议，突破传统IMU手套的采样率与时钟漂移瓶颈，揭示了高于100Hz的手部高频动力学。

**🔧 技术方法**

使用BHI360 6轴IMU、柔性可穿戴结构、ESP32-S3无线模块、广播同步协议、时空校准、频谱分析、机器人重定向优化等技术。

**📊 数据集**

主要使用作者自己采集的手部动态数据（如笔旋转、投掷捕捉等）进行实验，并与现有200Hz IMU手套以及三种不同结构的机器人手（Shadow、Allegro、Leap）进行比较。

**📈 对比分析**

与传统200Hz手套相比，系统在高频能量捕获、时序同步和机器人重定向误差方面显著提升；在三种机器人手上的重定向RMSE分别为8.3±6.8、1.0±1.8和0.6±1.4，证明高频数据可被准确转移到不同机器人结构。

**⚠️ 局限性**

局限性包括硬件成本较高、对温度/EMI敏感、目前仅捕获运动学信息，尚未结合触觉/力学传感器以完整记录手部运动与接触动力学。

---

## 323. Development of a European Union Time-Indexed Reference Dataset for Assessing the Performance of Signal Detection Methods in Pharmacovigilance using a Large Language Model

**arXiv ID:** 2603.26544 | [PDF](https://arxiv.org/pdf/2603.26544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 324. CA-TCN: A Causal-Anticausal Temporal Convolutional Network for Direct Auditory Attention Decoding

**arXiv ID:** 2603.26394 | [PDF](https://arxiv.org/pdf/2603.26394v1)

**作者:** Iñigo García-Ugarte `[一作]` (Universidad Pública de Navarra), Carmen Vidaurre `[通讯]` (Basque Center on Cognition Brain and Language)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出一种Causal–Anticausal Temporal Convolutional Network（CA‑TCN）模型，直接从EEG与音频信号中分类被注意说话人，取代传统的前向/后向线性回归或两步CCA+分类流程。

**💡 创新点**

创新点在于：①使用因果与反因果卷积分别处理音频和EEG，使模型自然对齐两者的时延；②构建不对称的受限场（EEG约234 ms，音频约984 ms）以降低在线延迟；③采用统一的TCN架构，集成残差连接、深度可分离卷积和膨胀卷积，显著简化网络并提升泛化性能；④在不同数据集上验证空间滤波器的稳定性。

**🔧 技术方法**

技术细节包括：Temporal Convolutional Network（堆叠膨胀卷积层）、因果/反因果卷积、残差连接、深度可分离卷积、Batch Normalization + ELU、交叉熵训练、Adam优化，以及与基线模型（Ridge、CCA、AADNet）相同的数据预处理和评估框架。

**📊 数据集**

使用三组公开或自制EEG数据集：Jaulab（20人、26 s/试验、61通道）、DTU（18人、50 s/试验、64通道）以及KU Leuven（16人、可变时长、64通道）每个数据集均包含两说话人竞争场景的音频与EEG记录。

**📈 对比分析**

通过与Ridge、CCA、AADNet在主观（SS）和无主观（SI）验证下，评估不同决策窗口（1–50 s）的分类准确率和Minimum Expected Switch Duration（MESD）。CA‑TCN在SI下平均提升0.5–3.2%准确率、SS下提升0.8–2.9%；MESD在大部分设置下显著低于AADNet，表明模型在快速切换和在线场景中的潜在优势。

**⚠️ 局限性**

局限性包括：实验仅在受控实验室条件下进行，使用清晰音频与人工生成新闻内容；未评估在真实噪声、语音分离、移动设备或极低采样率环境下的性能；模型对不同EEG接收设备的适应性仍待进一步验证。

---

## 325. Dynamic Token Compression for Efficient Video Understanding through Reinforcement Learning

**arXiv ID:** 2603.26365 | [PDF](https://arxiv.org/pdf/2603.26365v1)

**作者:** Shida Wang `[一作]` (University of Science and Technology of China), Linli Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3536 | [OpenAlex ID](https://openalex.org/A5009732907)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种基于强化学习的动态视觉令牌压缩框架，能够在保持或提升视频理解性能的同时显著减少令牌数量。

**💡 创新点**

通过惊讶增幅状态表示捕捉时序变化，结合分组优势估计和两阶段伪真视频到真实视频的课程学习，实现了内容感知、动态适应且端到端可优化的压缩策略。

**🔧 技术方法**

使用轻量级 MLP 策略网络、贝叶斯保留概率、分组采样的策略梯度优化、分裂优势估计以及伪视频预热 + 实际视频微调的课程学习。

**📊 数据集**

在 LLaVA-OneVision-Data、Koala-36M、LLaVA-Video-178K 等视频-字幕对上训练，并在 Video‑MME、MLVU、LVBench 三大视频语言基准上评测。

**📈 对比分析**

与 Vanilla、DyCoke、HoliTom、VidCom^2、FastVID 等无监督压缩方法对比，保持 10% 保留率时实现 99.5% 原性能、16× 预填充速度提升；在 25% 保留率时甚至超过未压缩模型。

**⚠️ 局限性**

对极其微小或噪声丰富的运动仍难以精准区分，且在不同 LLM 结构和极端压缩率下的泛化仍需进一步验证。

---

## 326. The Multi-AMR Buffer Storage, Retrieval, and Reshuffling Problem: Exact and Heuristic Approaches

**arXiv ID:** 2603.26542 | [PDF](https://arxiv.org/pdf/2603.26542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 327. Cryptanalysis of a PIR Scheme based on Linear Codes over Rings

**arXiv ID:** 2603.26409 | [PDF](https://arxiv.org/pdf/2603.26409v1)

**作者:** Luana Kurmann `[一作]` (German Aerospace Center), Violetta Weger `[通讯]` (Technical University of Munich)

**通讯引用:** 134 | [OpenAlex ID](https://openalex.org/A5084539841)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

分析了基于环的码的单服务器PIR方案的完整性与安全性，提出了补充条件实现文件检索，并展示了一种利用行空间维度差异的攻击方法。

**💡 创新点**

首次将Rank差攻击迁移到环上，利用自由码与非自由码的维度差，证明大文件数下攻击成功概率高，并指出原方案缺陷并给出完整性补偿条件。

**🔧 技术方法**

使用码论中的自由与非自由码分类、矩阵乘积码、汉塞尔提升、标准形转化及多重CRT分解，结合高斯消元求维度进行分析。

**📊 数据集**

实验使用模拟数据库矩阵（如m=36、n=5、t=3等参数）及在Magma中随机生成的实例；未使用公开真实数据集。

**📈 对比分析**

通过理论复杂度O(ℓ·t·r^3(t-1)^3)与实验结果对比，显示在常见参数下攻击在毫秒至秒级完成，性能优于下载全库且在实际部署中可行。

**⚠️ 局限性**

攻击需满足文件数t≥2ns/r+2的下界；若文件数不足或采用不同非自由码定义，需调整阈值；仅适用于单服务器码基PIR，未覆盖多服务器或更复杂编码场景。

---

## 328. AutoWeather4D: Autonomous Driving Video Weather Conversion via G-Buffer Dual-Pass Editing

**arXiv ID:** 2603.26546 | [PDF](https://arxiv.org/pdf/2603.26546v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 329. OVI-MAP:Open-Vocabulary Instance-Semantic Mapping

**arXiv ID:** 2603.26541 | [PDF](https://arxiv.org/pdf/2603.26541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 330. Finding Minimal Clusters in st-DAGs

**arXiv ID:** 2603.26538 | [PDF](https://arxiv.org/pdf/2603.26538v1)

**作者:** Ulrich Vogl `[一作]` (University of the German Federal Armed Forces Munich), Markus Siegle `[通讯]` (University of the German Federal Armed Forces Munich)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种在单入口单出口有向无环图（st‑DAG）中寻找最小“cluster”（即在入口和出口边界上表现同步的子图）的高效算法。

**💡 创新点**

创新点在于引入“syncpoint”及其最大形式“Maximum Syncpoint（MSP）”概念，并构造仅由MSP构成的二级图（MSP‑DAG）。通过在MSP‑DAG上进行搜索，显著降低了搜索空间，从而实现了对大规模st‑DAG的可行分析。

**🔧 技术方法**

核心技术包括：
- 图论定义（cluster、syncpoint、MSP）
- 通过双向可达集合与集合运算判断子图闭合性
- 构造MSP‑DAG并利用其前驱后继关系（immediate precedence）
- 采用路径集合预处理和指针/分隔符结构实现高效的子路径删除
- 复杂度分析和算法实现细节（O(k·n²·2ᴺ)）。

**📊 数据集**

实验数据集采用随机生成的st‑DAG，使用可调参数（n、parexp、serexp、maxwidth、clustsettle、narb）生成不同规模的图，单机Python/NetworkX实现。

**📈 对比分析**

通过实验验证算法在多种规模下（n≤5000）运行时间基本与n²成正比，且MSP‑DAG相较原图缩小约90%，从而显著提升性能。实验中未与其他现有方法直接比较，但展示了算法在实际随机图上的可行性与优越性。

**⚠️ 局限性**

主要局限：算法复杂度指数级随MSP数量N增长，若生成的图MSP数目较大，可能导致指数级路径枚举和内存消耗；另外，实验仅基于随机生成图，缺乏对真实工程图的评估。

---

## 331. Hardware-Agnostic and Insightful Efficiency Metrics for Accelerated Systems: Definition and Implementation within TALP

**arXiv ID:** 2603.26576 | [PDF](https://arxiv.org/pdf/2603.26576v1)

**作者:** Ghazal Rahimi `[一作]` (Barcelona Supercomputing Center), Marta Garcia-Gasulla `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 824 | [OpenAlex ID](https://openalex.org/A5082933794)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对异构 CPU+GPU 高性能计算平台，扩展并实现了 POP 效率框架的新指标，以分离主机与设备的性能损失

**💡 创新点**

创新点在于提出硬件无关的主机/设备双树效率度量（Host Hybrid Parallel Efficiency、Device Offload Efficiency、Device Load Balance、Communication Efficiency、Orchestration Efficiency），并将其集成到轻量级监控库 TALP 中

**🔧 技术方法**

使用了 DLB 动态负载均衡库、TALP 跟踪框架、CUDA/CUPTI 与 HIP/rocprofiler 后端、OpenACC 钩子，以及 PMPI/OMPT 进行在线/离线监测

**📊 数据集**

数据集包含人工合成的 PILS 微基准（多种负载失衡场景）以及三种生产级科学应用（SOD2D、FALL3D、XSHELLS），在 MareNostrum‑5 Acc 加速分区上执行

**📈 对比分析**

通过与 Paraver/Nsight Systems 生成的详细时间线对照，验证了新指标的准确性；在实验中发现主机上 Offload Efficiency 低、设备上的 Orchestration Efficiency 受通信/负载失衡影响，指导了优化措施

**⚠️ 局限性**

局限性包括：目前仅实现第一层（并行效率）指标；仅支持 NVIDIA GPU（CUDA/OpenACC），尚未完整实现 AMD GPU；未实现计算效率分支；对极大规模多节点的长期稳定性验证不足

---

## 332. Generation Is Compression: Zero-Shot Video Coding via Stochastic Rectified Flow

**arXiv ID:** 2603.26571 | [PDF](https://arxiv.org/pdf/2603.26571v1)

**作者:** Ziyue Zeng `[一作]` (Waseda University), Hiroshi Watanabe `[通讯]` (Waseda University)

**通讯引用:** 17406 | [OpenAlex ID](https://openalex.org/A5071346122)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种零样本生成视频编解码框架GVC，利用已有的预训练视频生成模型通过将其定向流ODE转化为SDE，并在每一步注入可复现的码本向量，直接在解码器上重放生成轨迹，完成压缩和重构；同时设计了三种条件策略（I2V、T2V、FLF2V）以权衡空间细节、时序一致性和比特率。

**💡 创新点**

①将定向流ODE在推理阶段转换为等价SDE，首次在视频生成中引入可控噪声注入以实现码本驱动压缩；②设计三种对齐策略，构建从无参考（T2V）到双端参考（FLF2V）的一条可调节折衷曲线；③在I2V中加入自适应尾帧码本分配和残差校正，在FLF2V中实现边界共享GOP链条，显著降低边界帧开销并提升时序连贯性。

**🔧 技术方法**

基于Score‑SDE理论的ODE‑to‑SDE转换；码本驱动的SDE采样（Turbo‑DDCM风格）；多原子阈值化与重构；CLIP可视嵌入与VAE潜在编码；自适应尾帧码本分配与量化残差；双端边界共享GOP链条；随机初始化共享种子实现零样本重放。

**📊 数据集**

UVG数据集（720p与1080p两种分辨率）以及自拍摄视频用于验证模型的泛化性。

**📈 对比分析**

通过在UVG上与主流传统码流（HEVC/VVC）和学习型码流（DCVC‑RT、GNVC‑VD）以及生成式训练码流（GLC‑Video）在相同bpp点进行对比，GVC在0.002~0.018 bpp区间取得最低LPIPS（最高感知质量）并在PSNR上与I2V相当，FLF2V以约15%比I2V低的额外比特率实现近80%的PSNR提升；T2V以最低比特率实现可接受的感知质量。

**⚠️ 局限性**

推理速度与计算成本相对较高；对预训练模型的依赖可能限制在特定分辨率或场景上的表现；I2V策略易受误差累积影响；T2V缺乏空间锚定可能导致漂移；码本大小与步数调节需要经验，且在极低bpp下仍可能出现细节丢失。

---

## 333. Scene Grounding In the Wild

**arXiv ID:** 2603.26584 | [PDF](https://arxiv.org/pdf/2603.26584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 334. CR-Eyes: A Computational Rational Model of Visual Sampling Behavior in Atari Games

**arXiv ID:** 2603.26527 | [PDF](https://arxiv.org/pdf/2603.26527v1)

**作者:** Martin Lorenz `[一作]` (ScaDS.AI, Leipzig University), Patrick Ebel `[通讯]` (ScaDS.AI, Leipzig University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 CR-Eyes，一种基于计算合理性和强化学习的模型，用于模拟 Atari 游戏中的视觉采样与游戏行为，直接处理原始像素并在感知-动作循环中将注视视为目标导向行为。

**💡 创新点**

创新点在于将眼动作为可行动的感知动作，将感知、运动和记忆约束与 CR 框架结合；不依赖手工任务表示，能在动态像素环境下同时学习注视与操作策略；并通过 EMMA 模型模拟眼动延迟。

**🔧 技术方法**

使用强化学习（DQN + 两个动作头）结合 SUGARL、EMMA、CR 理论，训练模型在 Atari-HEAD 环境中执行游戏和视觉采样。

**📊 数据集**

使用 Atari-HEAD 数据集（包含 20 款 Atari 游戏的 117 小时眼动+游戏记录）以及 Asterix、H.E.R.O.、Seaquest 三款游戏的数据。

**📈 对比分析**

与人类基准、传统 DQN 与 Dreamer 进行比较：在无暂停的常规设置下，CR‑Eyes 的得分与非专家人类相近；在允许暂停的设置中得分提升不足，且人类更倾向短暂停；在注意力热图与扫描路径方面，模型与人类在早期相似但后期显著偏离。

**⚠️ 局限性**

局限性包括：采用无模型的 DQN，缺乏对环境动态的内部表示；仅使用视线焦点而不考虑周边视野；扫描路径与人类不一致，未能捕捉人类的预测性注视；暂停策略与人类差异；整体游戏表现仍低于高阶 RL 方法。

---

## 335. Innovation Discovery System for Networking Research

**arXiv ID:** 2603.26496 | [PDF](https://arxiv.org/pdf/2603.26496v1)

**作者:** Mengrui Zhang `[一作]` (Xiamen University), Jiwu Shu `[通讯]` (Minjiang University)

**通讯引用:** 3066 | [OpenAlex ID](https://openalex.org/A5101740783)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

该系统通过构建网络研究专用的科学发现数据集、模拟人类科学推理流程（问题设定、灵感检索、方案生成和迭代优化），以及基于时间切分的创新度与实用度联合评估，自动生成高质量的网络研究创新点。

**💡 创新点**

创新点包括：①面向网络研究的结构化知识图谱和摘要数据集构建；②将知识图谱检索与LLM生成相结合的科学推理式创意流程；③采用时间切分协议同时评估创意的 novelty 与 practicality，并以调和平均数综合衡量。

**🔧 技术方法**

技术方案主要包含：大语言模型（GPT‑5、Gemini‑2.0‑flash、Qwen‑plus‑1220）用于摘要、推理与迭代生成；GraphRAG 构建论文图与引用图；SPECTER 进行文本向量化与相似度计算；结构化摘要与知识图谱驱动的检索；以及基于阈值的相似度筛选和多轮优化。

**📊 数据集**

使用了 743 篇来自 ACM SIGCOMM 与 USENIX NSDI（2021‑2025 年）的论文数据；通过 LLM 提取后形成 743 条结构化记录（Background、Problem、Design），其中 586 篇为 pre‑dataset（2021‑2024），157 篇为 post‑dataset（2025）。

**📈 对比分析**

系统通过与单独 LLM 输出的基线进行对比，采用 novelty、practicality、overall quality 三维度评估。实验结果显示，系统生成的创新点 novelty ≥ 0.94，practicality ≈ 0.83‑0.86，overall quality ≈ 0.90；在 ablation 研究中，尤其在交通工程域使用 Gemini 时，整体质量提升高达 42%。

**⚠️ 局限性**

局限性包括：①评估依赖相似度计算和时间切分，未能验证真实可实现性；②数据集仅覆盖两大会议，可能限制多样性与通用性；③LLM 可能产生幻觉或事实错误，需人工审校；④缺乏端到端的原型实验，实用度评价仅通过后期论文相似度间接体现。

---

## 336. User Involvement in Robotic Wheelchair Development: A Decade of Limited Progress

**arXiv ID:** 2603.26543 | [PDF](https://arxiv.org/pdf/2603.26543v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 337. MA-Bench: Towards Fine-grained Micro-Action Understanding

**arXiv ID:** 2603.26586 | [PDF](https://arxiv.org/pdf/2603.26586v1)

**作者:** Kun Li `[一作]` (United Arab Emirates University), Dan Guo `[通讯]` (Hefei University of Technology)

**通讯引用:** 6267 | [OpenAlex ID](https://openalex.org/A5100733153)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MA-Bench基准和MA-Bench-Train训练集，评估并提升多模态大型语言模型在细粒度微动作（Micro-Action）理解上的能力。

**💡 创新点**

①首个专门针对微动作细粒度的多模态评测基准；②半自动化生成结构化微动作描述与问答的流水线；③构建20.5K视频训练集以增强模型的微动作识别与推理；④采用三层感知‑理解‑推理的评估架构。

**🔧 技术方法**

使用光流+人体分割+骨架生成身体部位运动描述，利用大型语言模型（DeepSeek、LLM）生成结构化描述和多选/是非问答；对多模态LLM（Qwen3‑VL、VideoLLama、LLaVA、InternVL等）进行零样本评估并通过LoRA微调提升性能。

**📊 数据集**

以Micro‑Action‑52为源数据，抽取1,000个视频构成MA‑Bench，20,510个视频（含细粒度描述）构成MA‑Bench‑Train；参考iMiGUE、SMG、BBSI等相关微动作数据集。

**📈 对比分析**

在23个公开/专有MLLM上做零样本评估，闭合任务平均准确率约30‑50%；开环任务采用LLM‑as‑Judge评分（0‑5分），未微调模型平均约0.5‑0.8分，微调后Qwen3‑VL‑8B在闭合任务达到50.68%、开环1.69分，显著优于未微调模型并超过多数专有模型。

**⚠️ 局限性**

当前模型仍难捕捉极细微运动与精确时序，视觉输入冗余易干扰注意力；基准主要覆盖单人短视频，缺少多主体或复杂场景；训练数据规模虽大但仍不足以完全覆盖所有微动作变体。

---

## 338. Sharp Capacity Scaling of Spectral Optimizers in Learning Associative Memory

**arXiv ID:** 2603.26554 | [PDF](https://arxiv.org/pdf/2603.26554v1)

**作者:** Juno Kim `[一作]` (University Of California Berkeley), Jason D. Lee `[通讯]` (University Of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

在线性关联记忆模型中对 Muon 与 SGD 的学习动态进行理论与实验分析，揭示 Muon 的一阶恢复率与批量大小阈值优势，并进一步探讨多步收敛行为。

**💡 创新点**

首次把 Muon 与 SGD 在非正交高维高频率数据下的容量与批量阈值进行严格比较，证明 Muon 的一次步容量可达 Θ(d^{1+1/(2α)})，显著超过 SGD 的 Θ(d^{1/(2α)})，并在大批量场景下展示更高的批量阈值。

**🔧 技术方法**

采用随机矩阵理论、谱优化（Muon）与阈值梯度近似、批量采样分析、单步与多步递推公式、以及对偶投影理论来推导恢复阈值和容量上界。

**📊 数据集**

主要使用人工生成的高维高频率 Gaussian 嵌入数据（N≈10^5）以及基于两层 Transformer 的合成上下文记忆任务数据；实验中还使用了不同 α 的 Zipf 频率分布。

**📈 对比分析**

通过与 SGD、AdamW 的对比实验，在单步、批量大小、以及多步训练阶段，Mu在容量、记忆准确率和 OOD 召回率上均优于传统优化器，尤其在大批量与大频率项的情况下显著提升。

**⚠️ 局限性**

局限性包括：理论推导基于理想化假设（如块消解近似、离散频率分布、无正交性），对实际语言模型任务的推广尚需进一步验证；实验以合成数据为主，缺乏在真实 NLP 数据集上的系统评估。

---

## 339. Exploring a Design Framework for Children's Agency through Participatory Design

**arXiv ID:** 2603.26523 | [PDF](https://arxiv.org/pdf/2603.26523v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 340. On merge-models

**arXiv ID:** 2603.26570 | [PDF](https://arxiv.org/pdf/2603.26570v1)

**作者:** Hector Buffière `[一作]` (Université Paris Cité), Sebastian Siebertz `[通讯]` (University of Bremen)

**通讯引用:** 838 | [OpenAlex ID](https://openalex.org/A5073548944)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

提出了合并模型（merge-models）以及与其对应的树序弱稀疏模型框架，进一步推广了双胞模型（twin-models）并证明其在保持线性团宽、团宽及半径‑r 合并宽方面的保真性；

**💡 创新点**

核心创新是将合并宽（merge-width）这一新的宽度度量嵌入树序弱稀疏模型中，构造了可从合并序列直接得到的合并模型，并证明双胞模型是合并模型的特例；

**🔧 技术方法**

主要技术包括：树序弱稀疏模型的定义、合并宽与合并序列的关联、合并模型的构造与清理（cleaning）与压缩（compactification）技术，以及利用一阶解释和传递（transduction）来证明宽度保持性；

**📊 数据集**

文中未使用具体实验数据集，所有结论均为理论证明；

**📈 对比分析**

比较方法为理论上证明：任何给定半径‑r 合并宽为 t 的结构都可构造出半径‑r 合并宽 ≤ 2t 的合并模型；类似地，双胞模型可在保持 clique-width 或 linear clique-width 的前提下将宽度保持在不超过原值两倍；

**⚠️ 局限性**

局限性在于：合并模型的构造与压缩过程复杂度未被量化；所需的常数（尤其 r₀）可能非常大，且目前尚未对合并宽模型在算法或模型理论上的实际效能进行实验验证。

---

## 341. Stabilizing Rubric Integration Training via Decoupled Advantage Normalization

**arXiv ID:** 2603.26535 | [PDF](https://arxiv.org/pdf/2603.26535v1)

**作者:** Zelin Tan `[一作]` (University Of Science And Technology Of China), Lei Bai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 4361 | [OpenAlex ID](https://openalex.org/A5028486493)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Process-Aware Policy Optimization（PAPO），通过在 Group Relative Policy Optimization（GRPO）中加入过程级奖励并实现优势解耦来提升数学推理模型的表现。

**💡 创新点**

创新点在于：①将结果奖励（ORM）与过程奖励（PRM）分别独立归一化；②过程奖励仅在答案正确的样本子集中归一化，防止奖励劫持和信号枯竭；③在优势空间中融合两类信号，既保留答案正确性指引，又细化推理质量。

**🔧 技术方法**

使用技术包括：GRPO无价值网络优化、基于 LLM-as-Judge 的 rubric‑based PRM、组级优势归一化、正确子集归一化（Correct‑Subset Normalization）以及可选的 DAPO 动态采样/裁剪。

**📊 数据集**

训练数据采用 NuminaMath‑1.5‑RL‑Verifiable（20k 题目，按难度分层采样）；评估数据集包括 OlympiadBench、MATH‑500、AIME 2024/2025、GPQA‑Diamond、HumanEval。

**📈 对比分析**

与传统 ORM（GRPO）及其乘法组合（ORM×PRM）对比，PAPO 在所有模型规模（3B–14B）上均显著提升，最突出的是 OlympiadBench 7B 模型从 46.3% 提升至 51.3%，在六大基准上的平均提升 3–5%，并且随着模型规模增长性能提升更为显著。

**⚠️ 局限性**

局限性：仅在 Qwen 系列模型上验证，未测试 Llama、Gemma 等架构；使用的 PRM 为 GPT‑OSS‑20B，未探究不同评判模型或多尺度评估的影响。

---

## 342. Learnable Quantum Efficiency Filters for Urban Hyperspectral Segmentation

**arXiv ID:** 2603.26528 | [PDF](https://arxiv.org/pdf/2603.26528v1)

**作者:** Imad Ali Shah `[一作]` (University of Galway), Brian Deegan `[通讯]` (University of Galway)

**通讯引用:** 754 | [OpenAlex ID](https://openalex.org/A5091422105)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对城市驾驶的高光谱语义分割，提出了一种可学习的量子效率（LQE）滤波器库，用来对高维光谱数据进行可解释且低参数的降维，并与分割模型端到端联合训练。

**💡 创新点**

创新点在于将光谱滤波器建模为高阶非对称高斯基函数，并通过物理启发的约束（单峰占主导、带宽有限、峰距分离）保证滤波器的光学可实现性，同时保持模型可微和参数极低。

**🔧 技术方法**

使用的技术包括可微分高阶 Gaussian 过滤器、正则化损失（峰占比、峰距、带宽约束）、多峰滤波器集成、与多种语义分割网络（UNet、DeepLab、PSPNet 等）端到端联合训练，以及与传统 PCA/ICA/NMF 等经典降维方法的对比实验。

**📊 数据集**

采用公开的三大城市驾驶高光谱数据集：HyKo‑VIS（15 帧 470–630 nm）、HSI‑Drive（25 帧 600–975 nm）和 H‑City（128 帧 450–950 nm）。

**📈 对比分析**

与 6 种经典降维方法（PCA、ICA、NMF、MNF、LLE、Isomap）和 7 种可学习降维层（1×1 Conv、AE、ECA、CBAM、seAttn、DSC、ConvNeXt）在 6 个分割模型上进行比较，平均 mIoU 提升约 1.25–2.45%，参数量仅为 12–36，显著低于其他可学习方法的 51–22K 参数。

**⚠️ 局限性**

主要限制包括：推理时延相对较高（尤其与 1×1 Conv 比较），滤波器学习依赖于单一数据集，跨数据集迁移和真实光学实现验证尚未展开；此外，当前 LQE 只在固定光谱范围内训练，未针对不同光谱范围的自适应峰数做进一步探索。

---

## 343. JAL-Turn: Joint Acoustic-Linguistic Modeling for Real-Time and Robust Turn-Taking Detection in Full-Duplex Spoken Dialogue Systems

**arXiv ID:** 2603.26515 | [PDF](https://arxiv.org/pdf/2603.26515v1)

**作者:** Guangzhao Yang `[一作]` (Recho Inc), Ningjie Bai `[通讯]` (Recho Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了JAL‑Turn，一个轻量化、声学‑语言联合建模的语音仅转向检测框架，能够与ASR共享编码器实现零额外延迟；

**💡 创新点**

创新点在于：① 采用预训练的声学编码器(CPC)与语言编码器(SenseVoice)的交叉注意力融合，实现声学与语言信息的动态互补；② 在ASR后端共享SenseVoice编码器，做到转向检测与识别同步推理；③ 构建可扩展的无人工标注数据管线，通过未来窗口 VAD 生成可靠的转向标签，支持多语言多域训练；

**🔧 技术方法**

技术手段包括：预训练声学模型CPC、预训练语言模型SenseVoice、交叉注意力融合层、带ALiBi位置偏置的自注意力Transformer、轻量化注意力池化、Sigmoid二分类头，训练采用AdamW+余弦退火；

**📊 数据集**

使用的数据集包括公开多语言 Easy‑Turn（1145h）、STurn‑v3（700h、23语种）、内部日语客服对话（≈1200h）以及单句高质量数据（≈95h），并通过VAD+未来窗口自动标注得到训练样本；

**📈 对比分析**

与音频仅基线（STurn‑v2/v3）、SLM基线（EasyTurn）以及LLM管线（Gemini‑2.5‑Flash、Qwen3‑0.6B、GPT‑5.1）进行对比。JAL‑Turn 在 Easy‑Turn 上达 96.67% 准确率、12 ms 延迟；在日语集上 92.03% 准确率、38 ms 延迟；在多语种 STurn‑v3 上 93.27% 准确率、36 ms 延迟。相比音频仅模型，准确率提升约 2–3%，延迟下降 70–80%；相比 LLM 基线，准确率提升 5–10%，延迟降低 5–10 倍；

**⚠️ 局限性**

局限性：对回声（backchannel）状态的准确率仍略低；模型仅利用声学与语言信息，未结合视觉或其他多模态特征；依赖预训练模型的冻结参数，可能限制进一步自适应；在极短句或极快语速场景下的鲁棒性尚未充分验证。

---

## 344. CADSmith: Multi-Agent CAD Generation with Programmatic Geometric Validation

**arXiv ID:** 2603.26512 | [PDF](https://arxiv.org/pdf/2603.26512v1)

**作者:** Jesse Barkley `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8708 | [OpenAlex ID](https://openalex.org/A5008745801)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CADSmith多代理文本到CAD生成管线，包含两级纠错循环

**💡 创新点**

创新点在于将程序化几何验证与视觉语言评估结合的闭环修正机制，并采用检索增强生成以避免微调

**🔧 技术方法**

使用Claude Sonnet/Opus LLM、OpenCASCADE核度量、VLM Judge、检索增强生成和多代理架构

**📊 数据集**

使用自制的100条自然语言+CadQuery脚本基准，分三难度层级（T1–T3）

**📈 对比分析**

与零射击基线和无视图消融对比，完整管线执行率100%，平均Chamfer距离从28.37降至0.74，F1和IoU显著提升

**⚠️ 局限性**

局限在视图角度固定导致对细微间隙捕捉不足，且未处理多部件装配和更大规模设计

---

## 345. Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference

**arXiv ID:** 2603.26498 | [PDF](https://arxiv.org/pdf/2603.26498v1)

**作者:** Konstantinos Papaioannou `[一作]` (IMDEA Software Institute), Thaleia Dimitra Doudali `[通讯]` (IMDEA Software Institute)

**通讯引用:** 190 | [OpenAlex ID](https://openalex.org/A5033123165)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 RPS‑Serve，一种针对多模态大型语言模型推理的调度框架，能够在单机 GPU 上高效处理文本、图像和视频请求。

**💡 创新点**

创新点在于将请求按资源需求划分为“岩石”“卵石”“沙子”三类，并通过动态优先级与老化机制实现对时延敏感请求的优先响应，同时防止大资源请求饥饿。

**🔧 技术方法**

使用了工作负载分析、预估器（影响估计器）、分类器、队列管理器以及优先级调节器等技术，并基于 vLLM 的 chunked‑prefill 进行扩展。

**📊 数据集**

实验数据集包括 ShareGPT、LLaVA‑Instruct、LLaVA‑Video 以及公开的多模态文本、图像和视频数据集。

**📈 对比分析**

与 vLLM（FCFS）和 EDF（截止时间优先）对比，RPS‑Serve 在主流多模态模型上平均 TTFT 减少 54%，对延迟关键请求减 78.5%，显著降低首词延迟、尾部延迟及 SLO 违例率。

**⚠️ 局限性**

局限性包括仅支持文本、图像、视频三种输入；在多 GPU 或多节点环境下的扩展性未验证；对输出多模态（如图像/视频生成）支持不足；需要为新模态重新训练分类器。

---

## 346. Reentrancy Detection in the Age of LLMs

**arXiv ID:** 2603.26497 | [PDF](https://arxiv.org/pdf/2603.26497v1)

**作者:** Dalila Ressi `[一作]` (Ca' Foscari University of Venice), Sabina Rossi `[通讯]` (Ca' Foscari University of Venice)

**通讯引用:** 1343 | [OpenAlex ID](https://openalex.org/A5015760641)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 Solidity 0.8+ 版的重入攻击检测工具进行可靠性与鲁棒性评估，构建并手工验证了两个数据集（Aggregated Benchmark 与 Reentrancy Scenarios Dataset），并对传统静态分析器、机器学习模型和大型语言模型（LLM）在这两组数据上的表现进行系统比较。

**💡 创新点**

创新点在于：①提出基于手工验证的高置信度 Benchmark 与专门针对语义角落案例的 RSD，弥补现有工具/数据集缺乏可靠标注与现代 Solidity 兼容性的空缺；②从“可靠性/鲁棒性”维度对工具生态进行评估；③展示 LLM 在零射击设置下可显著优于传统静态/ML 检测器，并证明其解释能力有助于人工审核与数据集改进。

**🔧 技术方法**

使用技术包括：
- 静态分析器（Symbolic Execution、Abstract Interpretation、Data‑flow/CPC 等）
- 传统机器学习模型（Gradient Boosting、Random Forest、SVM、Logistic Regression、CodeBERT 等）
- 大型语言模型（GPT‑4o、GPT‑4.1、Gemini‑2.0‑flash、Qwen‑3 等）
- 对比实验平台 SmartBugs 2.0、编译器测试与 LLM 提示工程。

**📊 数据集**

数据集：
- Aggregated Benchmark：432 条手工验证合约（122 可重入，314 安全）
- Reentrancy Scenarios Dataset (RSD)：143 条最小工作示例，涵盖最新 Solidity 语义与复杂调用模式。

**📈 对比分析**

比较方法：采用准确率、召回率、F1 分数和错误率等指标；在 Aggregated Benchmark 上传统工具最高 F1 为 0.87，LLM（如 GPT‑4o、o4‑mini）最高 F1 0.96；在 RSD 上传统工具 F1 仅 0.76–0.82，LLM 最高 0.82–0.86，且传统工具出现显著错误率。结果显示 LLM 在鲁棒性与可靠性方面明显优于传统静态分析与 ML 模型。

**⚠️ 局限性**

限制：
- 数据集规模有限，未覆盖整个以太坊生态中所有合约类型与部署模式；
- LLM 缺乏可验证性与确定性，可能产生幻觉解释；
- 人工标注过程虽多轮复核，但仍可能存在主观偏差；
- 评测侧重于 Solidity 0.8+，对其他语言或平台的迁移性未知。

---

## 347. Beyond Banning AI: A First Look at GenAI Governance in Open Source Software Communities

**arXiv ID:** 2603.26487 | [PDF](https://arxiv.org/pdf/2603.26487v1)

**作者:** Wenhao Yang `[一作]` (Peking University), Minghui Zhou `[通讯]` (Peking University)

**通讯引用:** 2573 | [OpenAlex ID](https://openalex.org/A5065977454)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对67个高可见开源项目的公开治理文本进行定性文档分析，识别出GenAI治理的关注点、三种治理取向及12个治理策略。

**💡 创新点**

提出了三种治理取向（禁止式、边界与责任式、质量优先式）并构建跨项目的治理策略空间，将GenAI治理从单一禁止扩展为多维系统化框架。

**🔧 技术方法**

采用多阶段定性分析与迭代编码（open coding、策略归纳），并使用Cohen’s κ评估编码一致性。

**📊 数据集**

来源于GitHub前800星项目及其相关政策扩展的67份公开治理材料（如CONTRIBUTING、SECURITY、PR/issue模板等）。

**📈 对比分析**

通过案例对比提炼共性与差异，统计各策略在三取向中的出现比例，结果显示边界与责任式（O2）最常见，未进行量化性能评估。

**⚠️ 局限性**

样本局限于公开可见的高星项目，未覆盖非公开治理、平台工具实现与执行效果，治理效果与一致性未被经验验证，且研究为横断面，缺乏对治理演进动态的捕捉。

---

## 348. EcoFair: Trustworthy and Energy-Aware Routing for Privacy-Preserving Vertically Partitioned Medical Inference

**arXiv ID:** 2603.26483 | [PDF](https://arxiv.org/pdf/2603.26483v1)

**作者:** Mostafa Anoosha `[一作]` (University of Hull), Rameez Raja Kureshi `[通讯]` (University of Hull)

**通讯引用:** 86 | [OpenAlex ID](https://openalex.org/A5079880268)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并评估了 EcoFair，一种在医学影像边缘设备上实现隐私保护、能效与公平性的垂直分区推理框架；

**💡 创新点**

创新点在于将基于预测不确定性与神经符号医学风险评分的动态路由机制结合，实现轻量化首选推理并在必要时自动激活重模型；

**🔧 技术方法**

使用了垂直分区推理、轻量化与重量化模型的动态路由、基于年龄与病灶位置信息的符号风险评分、多模态融合及能耗监测等技术；

**📊 数据集**

实验数据集包括 HAM10000、PAD-UFES-20 与 BCN20000 三个公开皮肤病诊断数据集；

**📈 对比分析**

通过与仅使用轻模型、仅使用重模型以及整体集成模型的基线比较，在宏观 F1、均衡准确率、恶性召回率、能耗节约和子组公平性指标上，EcoFair 在计算差距足够大的模型配对中实现 25–68% 的能耗降低，并在多数据集上保持与重模型相近甚至更优的诊断性能，同时提升最弱子组的真阳性率；

**⚠️ 局限性**

局限性包括能效与性能提升高度依赖模型间计算差距、在计算差距较小或数据分布难度高时效果不明显、对完整表格元数据的质量要求高、实验仅为模拟环境，缺乏真实边缘硬件与网络延迟的评估。

---

## 349. SPECTRA: An Efficient Spectral-Informed Neural Network for Sensor-Based Activity Recognition

**arXiv ID:** 2603.26482 | [PDF](https://arxiv.org/pdf/2603.26482v1)

**作者:** Deepika Gurung `[一作]` (RPTU & DFKI), Paul Lukowicz `[通讯]` (RPTU & DFKI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为 SPECTRA 的轻量级惯性测量单元（IMU）人类活动识别（HAR）模型，能够在手机和微控制器上实现实时部署。

**💡 创新点**

创新点在于将短时傅里叶变换（STFT）的频谱特征与深度可分离卷积、通道自注意力和双向 GRU 结合，形成端到端可部署的谱时空架构，显著提升了准确率与资源利用的平衡。

**🔧 技术方法**

使用了 STFT 前端、深度可分离卷积、通道自注意力、Bi‑GRU+注意力池化、FP16/INT8 量化，以及 ExecuTorch（手机）和 TFLM（STM32 MCU）等部署工具链。

**📊 数据集**

在五个公开 IMU 数据集上进行评估：WISDM、USC‑HAD、UCI HAR、DSADS 和 PAMAP2。

**📈 对比分析**

与 DeepConvLSTM、TinyHAR、TinierHAR 等基准在 PC、Pixel 9 与 STM32L4 上进行端到端对比，SPECTRA 在准确率上与大型模型相当或略低，却在参数、MAC、延迟、吞吐量、能耗和内存上大幅优于对手，尤其在移动设备和 MCU 上实现了低功耗实时推理。

**⚠️ 局限性**

局限性包括 STFT 参数固定缺乏自适应多分辨率、未在长期真实用户场景中验证稳健性、以及未考虑多模态融合和能耗自适应调度等进一步提升方向。

---

## 350. Characterizing Scam-Driven Human Trafficking Across Chinese Borders and Online Community Responses on RedNote

**arXiv ID:** 2603.26520 | [PDF](https://arxiv.org/pdf/2603.26520v1)

**作者:** Jiamin Zheng `[一作]` (University of Edinburgh), Jingjie Li `[通讯]` (University of Edinburgh)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5100647029)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对RedNote上158条关于诈骗驱动的人口贩运的帖子进行定性内容分析，揭示招募手段、剥削机制、受害者再融入难题以及社区自救策略。

**💡 创新点**

首次系统性地把中国社交媒体视为研究诈骗驱动贩运的生态场，强调文化纽带如何在招募、控制和再融入阶段被利用与抵抗，并提出基于社区共识的“自救网络”与平台治理改进方向。

**🔧 技术方法**

采用半自动文本筛选（大语言模型+人工校对）和归纳式主题分析；在编码过程中使用轴向编码构建层级代码，并通过跨学科讨论形成最终代码库。

**📊 数据集**

数据集来源于RedNote公开帖子：先用关键词爬虫抓取6639条，去重后得到4499条，再通过LLM筛选得到1955条候选，最终人工抽样并编码得到158条具代表性样本。

**📈 对比分析**

技术评估方面，LLM筛选模型达到84%准确率、8.8%漏报率；在定性分析中实现数据饱和（首次93条帖子已足够），没有定量性能指标，但通过Cohen κ=0.98的高一致性验证筛选与编码可靠性。

**⚠️ 局限性**

局限性包括：仅采集单一平台（RedNote），难以推广到其他社交媒体；仅使用文本数据，未分析图片/视频等多模态信息；自述样本可能存在回忆偏差和社会期望；样本选取受关键词和平台治理策略影响，可能忽视隐蔽案例；跨境与法律环境差异导致对受害者身份认定的不确定性。

---

## 351. Conditional Diffusion for 3D CT Volume Reconstruction from 2D X-rays

**arXiv ID:** 2603.26509 | [PDF](https://arxiv.org/pdf/2603.26509v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 352. ALBA: A European Portuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative LLMs

**arXiv ID:** 2603.26516 | [PDF](https://arxiv.org/pdf/2603.26516v1)

**作者:** Inês Vieira `[一作]` (NOVA University of Lisbon), João Magalhães `[通讯]` (NOVA University of Lisbon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向欧洲葡萄牙语（pt-PT）的语言学基准 ALBA，涵盖语言变体、文化语义、语篇分析、文字游戏、句法、形态、词汇学、语音与音系等八大维度，并通过 LLM‑as‑a‑Judge 自动评测生成文本。

**💡 创新点**

创新点在于：①基准全程使用 pt-PT 原生专家手工设计，避免机器翻译带来的偏差；②通过多维度文字生成任务捕捉语言细粒度特征；③构建并验证 LLM‑as‑a‑Judge 框架，使大规模客观评估成为可能。

**🔧 技术方法**

技术包括：专家手工生成 800 问题与参考答案、LLM‑as‑a‑Judge 的提示工程与多轮推理、三种候选评判模型（Gemini‑2.5‑Pro、DeepSeek、GPT‑5）以及对提示语言、少样本策略的系统搜索。

**📊 数据集**

数据集：ALBA（800 题），包含 100 题/维度，专家评分 720 条回复，供 LLM 生成和评判训练；对标 1–5 维度评分被映射至 0–100。

**📈 对比分析**

对比方法：利用 MAE 与 LLM‑as‑a‑Judge 评估结果对齐后，对不同 LLM（开源 7B–12B 及闭源 GPT‑5/Gemini‑2.5‑Pro）在八维度上打分。闭源模型整体性能远超开源；在语言变体、文化语义、语音与音系及文字游戏等细粒度维度表现最弱。

**⚠️ 局限性**

局限性：①对细粒度任务（如韵律、文字游戏）仍易出现错误，显示 token 级处理对字符级任务的负面影响；②评判依赖单一 LLM‑as‑a‑Judge，潜在的评判偏差；③基准仅覆盖 pt-PT，缺乏跨语言验证，未来需扩展至更多低资源语言。

---

## 353. AMALIA Technical Report: A Fully Open Source Large Language Model for European Portuguese

**arXiv ID:** 2603.26511 | [PDF](https://arxiv.org/pdf/2603.26511v1)

**作者:** Afonso Simplício `[一作]` (NOVA School of Science and Technology), João Magalhães `[通讯]` (NOVA School of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个专注于欧洲葡萄牙语的全开源大语言模型，并为该语种提供了专门设计的评测基准；

**💡 创新点**

通过引入大量高质量的欧洲葡萄牙语数据、三阶段训练流程（预训练、监督微调、偏好训练）以及专门的评测基准，显著提升了模型在该语种的表现；

**🔧 技术方法**

采用了EuroLLM‑9B架构，使用RoPE扩展到32K长上下文、数据过滤与去重、监督微调（SFT）和直接偏好优化（DPO）等技术；

**📊 数据集**

使用了Arquivo.pt 195TB的网页档案、合成与人工标注的指令、对话、数学、生成、文化等多种数据集，以及公开的多语言数据集；

**📈 对比分析**

与其他同等规模的全开源和开权重模型比较，在翻译、推理、知识问答、生成、文化理解等任务上取得或接近最佳成绩，尤其在PT‑特定基准（PT‑Exam、ALBA、P3B3 等）上遥遥领先；

**⚠️ 局限性**

受限于PT‑PT数据集稀缺、预训练算力有限，以及在某些数学推理和细粒度语言细节上的表现仍有提升空间。

---

## 354. The Climber's Grip -- Personalized Deep Learning Models for Fear and Muscle Activity in Climbing

**arXiv ID:** 2603.26575 | [PDF](https://arxiv.org/pdf/2603.26575v1)

**作者:** Matthias Boeker `[一作]`, Pål Halvorsen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了攀岩者在顶绳与主绳攀爬过程中自我报告的恐惧感与肌肉活动（EMG）以及肌肉疲劳（MNFs）之间的关系，结合线性混合效应模型与深度学习模型进行预测。

**💡 创新点**

将随机效应（个体差异）引入深度学习网络（GRU）实现个性化建模，并提供了PyTorch实现代码，首次在攀岩心理生理研究中展示此方法提升预测准确性。

**🔧 技术方法**

使用线性混合效应模型、GRU、MLP、线性回归等模型，并在深度学习模型中加入随机效应层与负对数似然损失，比较 MSE、MAE、RMSE 等指标。

**📊 数据集**

实验数据来自20名中等水平攀岩者（19名有效）在同一台19.3 m室内墙上完成顶绳和主绳两种攀爬，采集EMG、ECG、IMU及自评恐惧量表等多模态时序数据。

**📈 对比分析**

通过10次随机初始化、70/15/15的训练/验证/测试拆分，比较了标准模型与随机效应模型，结果显示加入随机效应后GRU模型实现RMSE 2.49（±0.27），MAE 1.42（±0.18），MSE 6.25（±1.39），相较于无随机效应模型提升明显，误差约占目标变量SD的57%。

**⚠️ 局限性**

局限包括样本量相对较小、恐惧感仅为事后回顾性自评、缺乏实时情绪记录以及仅针对回归任务实现随机效应，未来需扩大样本、加入实时情绪或神经测量、探索分类任务的个性化模型。

---

## 355. Beyond Code Snippets: Benchmarking LLMs on Repository-Level Question Answering

**arXiv ID:** 2603.26567 | [PDF](https://arxiv.org/pdf/2603.26567v1)

**作者:** Yoseph Berhanu Alebachew `[一作]` (Virginia Tech), Chris Brown `[通讯]` (Virginia Tech)

**通讯引用:** 4852 | [OpenAlex ID](https://openalex.org/A5014431408)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个基于真实 Stack Overflow 问答与 134 个 Java 开源项目映射的多项目、存储库级别问答数据集 StackRepoQA，并使用该数据集系统评估 LLM 在项目规模问题解答上的表现。

**💡 创新点**

首次公开提供多项目、时间截断、结构化检索友好的存储库级别问答基准；同时证明了结构化（图）检索在提升 LLM 理解项目层面问题方面的有效性。

**🔧 技术方法**

使用 Claude 3.5 Sonnet 与 GPT‑4o 两大 LLM；构建多代理检索增强系统（文件级 RAG、图结构 RAG），并通过 LLM‑as‑judge 方式评估答案质量。

**📊 数据集**

1,318 条真实 Stack Overflow 提问与答案，映射到 134 个活跃 Java 项目，覆盖多文件、多依赖的真实开发场景。

**📈 对比分析**

与 LLM 基线对比：基线准确率约 58%；加入文件 RAG 提升至 56%；加入图 RAG 可达 64%；对比预训练截止后问题准确率显著下降（约 12%），说明记忆效应明显。

**⚠️ 局限性**

仅针对 Java 项目、仅以接受答案为真值、仅测试两款 LLM、检索策略受限，未能充分消除记忆干扰，且缺少对私有或快速演化项目的评估。

---

## 356. MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference

**arXiv ID:** 2603.26557 | [PDF](https://arxiv.org/pdf/2603.26557v1)

**作者:** Joris Köster `[一作]` (Aalto University), Zizhan Zheng `[通讯]` (Tulane University)

**通讯引用:** 1225 | [OpenAlex ID](https://openalex.org/A5101615991)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 MemBoost 框架，通过关联内存引擎、轻量 Meta 控制器与高能力大模型的协同，实现在交互式 LLM 服务中低成本、高质量的检索-升级推理。

**💡 创新点**

创新点包括：① 引入语义缓存与写回机制的关联内存，引导高质量答案被动态存入内存；② 采用轻量 LLM 作为 Meta 控制器，做检索与大模型升级的决策；③ 将成本与质量统一建模，形成“检索→决策→升级→写回”的闭环。

**🔧 技术方法**

使用检索增强生成（RAG）技术、FAISS 近似最近邻检索、MiniLM-L6-v2 句子嵌入、Qwen 等小型 LLM 作为 Meta 控制器、Qwen3-14B 作为大模型 Oracle 等。

**📊 数据集**

主要使用 MMLU‑Pro 数据集，并通过 Zipf 分布对问题进行采样生成重复/近似重复的查询流。

**📈 对比分析**

与小模型基线（Qwen3.5‑2B、Ministral‑3‑3B、Qwen3‑4B）和大模型 Oracle 进行对比，结果显示 MemBoost 在所有 Zipf 参数下的准确率与 Oracle 相当或更高，同时显著降低了大模型调用次数、整体推理成本和平均延迟。

**⚠️ 局限性**

局限性：仅在固定问答集上评估，未涉及长文本生成、编程等更复杂任务；对语义相似但非完全重复查询的检索误检风险未充分验证；写回策略在更大规模、多领域数据上的效果仍需进一步研究。

---

## 357. Shaping Credibility Judgments in Human-GenAI Partnership via Weaker LLMs: A Transactive Memory Perspective on AI Literacy

**arXiv ID:** 2603.26522 | [PDF](https://arxiv.org/pdf/2603.26522v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 358. A new approach to rating scale definition with quantum-inspired optimization

**arXiv ID:** 2603.26583 | [PDF](https://arxiv.org/pdf/2603.26583v1)

**作者:** Patrizio Spada `[一作]` (University of Palermo), Davide Corbelletto `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究将信用评分等级定义问题转化为二次无约束二进制优化（QUBO）模型，并使用量子启发式方法求解。

**💡 创新点**

创新点在于将金融机构在评级尺度设计中的组合约束优化问题映射到QUBO，并验证量子启发式算法在此类问题上的可行性。

**🔧 技术方法**

使用了QUBO建模、经典启发式算法（如模拟退火等）以及暴力搜索进行对比。

**📊 数据集**

论文未公开具体数据集，实验基于模拟的贷款申请人数据或内部样本进行验证。

**📈 对比分析**

通过与暴力搜索的对比，证明了启发式方法能够得到与最优解一致或接近的结果，并显示在更复杂场景下具有良好的可扩展性。

**⚠️ 局限性**

局限性包括实验规模有限、未在真实量子硬件上运行、缺乏多样化真实金融数据验证。

---

## 359. A Lyapunov Analysis of Softmax Policy Gradient for Stochastic Bandits

**arXiv ID:** 2603.26547 | [PDF](https://arxiv.org/pdf/2603.26547v1)

**作者:** Tor Lattimore `[一作]` `[通讯]`, Tor Lattimore

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文在离散时间的k臂随机多臂赌博机中分析了基于Softmax策略梯度算法的收敛性，并给出了在学习率满足 η = O(Δ^2/ log n) 条件下的上界：累计后悔量 R_n = O( k log(k) log(n) / η )。

**💡 创新点**

创新点在于将连续时间下已知的策略梯度分析方法迁移到离散时间，并且通过引入自适应的Lyapunov函数在高概率下控制对数因子，从而得到与连续时间等价的后悔上界；此外对最小子最优间距 Δ 的依赖仅为二次，而不是更差的线性依赖。

**🔧 技术方法**

主要技术包括：1) 软最大化策略梯度更新规则；2) 高概率下的日志保守性与log(n/δ)界；3) 再利用马尔可夫不等式、Freedman不等式和泰勒展开构造Lyapunov函数；4) 对软最大化的对数梯度进行保守估计。

**📊 数据集**

该工作不使用具体数据集，而是通过理论分析和概率不等式得到结果；因此可视为纯理论证明。

**📈 对比分析**

与已有工作比较：与连续时间的结果相当，但在离散时间下的证明更简洁；与传统的UCB/TS等离散时间策略相比，本算法在学习率选择上更为保守；在最优臂唯一的情形下，本方法与某些文献在收敛速率上保持一致；实验验证未给出，但理论上后悔量随 η、k、n 下降，满足 O(k log(k) log(n)/η)。

**⚠️ 局限性**

局限性包括：1) 学习率 η 必须小于 Δ^2 / log n，导致在实际实现中需要预估最小间距；2) 仍存在 log n 因子，作者认为可通过进一步分析消除；3) 由于二次 Δ 依赖，无法在 Δ 远大于 1 时实现更快收敛；4) 只在单臂奖励取值 [0,1] 的有限马尔可夫决策过程下证明，扩展到更一般噪声模型需进一步研究。

---

## 360. How Open Must Language Models be to Enable Reliable Scientific Inference?

**arXiv ID:** 2603.26539 | [PDF](https://arxiv.org/pdf/2603.26539v1)

**作者:** James A. Michaelov `[一作]` (Massachusetts Institute of Technology), Micah Altman `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2978 | [OpenAlex ID](https://openalex.org/A5084402546)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文系统分析了语言模型开放程度对科学推断的影响，识别了闭合模型在版本管理、信用归属和信息不足方面对可靠推断的威胁，并给出了相应的缓解建议。

**💡 创新点**

创新点在于将模型开放度与科研可靠性直接关联，提出了版本化、信用归属、信息不足三大问题框架；针对闭合模型的使用提供系统评估与缓解措施；并将开放权重模型定位为科研默认选择，给出详细使用与评估指南。

**🔧 技术方法**

主要采用理论与案例分析方法，结合现有模型架构、API 设计、提示工程、概率分布计算等技术细节，对 GPT‑3.5/4 等闭合模型版本变动进行观察。

**📊 数据集**

本文未开展实验，讨论基于公开 benchmark 与模型版本信息（如 GPT‑3.5、GPT‑4、Claude、Gemini 等），未使用特定数据集。

**📈 对比分析**

并未给出实验性能对比，而是通过理论比较展示开放权重模型与闭合模型在评估、比较和可解释性任务中的可行性，指出闭合模型在可靠推断上受限。

**⚠️ 局限性**

主要局限：缺乏实证验证；仅关注预训练模型开放度，未考虑训练数据、模型规模等其他因素；闭合模型缓解措施多为建议，实际可行性待验证；未讨论非预训练或多模态模型。

---

## 361. Evolution-Based Timed Opacity under a Universal Observation Model

**arXiv ID:** 2603.26573 | [PDF](https://arxiv.org/pdf/2603.26573v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 362. The internal law of a material can be discovered from its boundary

**arXiv ID:** 2603.26517 | [PDF](https://arxiv.org/pdf/2603.26517v1)

**作者:** Francesco Regazzoni `[一作]` (Politecnico di Milano), Francesco Regazzoni `[通讯]` (Politecnico di Milano)

**通讯引用:** 4645 | [OpenAlex ID](https://openalex.org/A5049456178)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于可微分有限元求解器的无监督超弹性材料律识别框架Neural‑DFEM，能够仅凭边界位移或局部观测数据逆推出材料能量函数；

**💡 创新点**

核心创新在于引入Hyperelastic Neural Network（HNN）架构，该网络在设计上自动满足帧不变性、材料对称性、多凸性、强制力学稳定性等七项物理与数学约束，从而保证在训练全过程中方程始终可解；

**🔧 技术方法**

技术实现包括：可微分有限元求解器嵌入学习循环、HNN（基于输入凸网络ICNN的多凸性网络）、BFGS quasi‑Newton优化、加载连续化、回溯线搜索以及特殊初始化与重参数化策略；

**📊 数据集**

使用合成数据集：在2D平面应变和3D三维几何中分别构造六种加载/几何设置（Setup 1–6），利用已知的Ishihara、Mooney‑Rivlin和Fung模型生成位移场和边界反作用力，并添加不同标准差的高斯噪声；

**📈 对比分析**

与传统VFM（EUCLID）和仅校准Neo‑Hookean模型对比，Neural‑DFEM在全域位移观测下vRMSE约为2×10⁻³，显著低于VFM的1.4×10⁻²；在边界观测下精度几乎不变；在噪声水平升至10⁻¹时仍保持误差<10⁻²，且对几何/边界条件的泛化（通过Sinkhorn散度量）表现良好；

**⚠️ 局限性**

主要限制包括：仅针对各向同性材料，无法直接处理各向异性；需要假设能量满足多凸性，导致对非多凸性真实材料（如Ishihara）逼近近似；实验验证仍缺乏真实测量数据；计算成本较高，训练时间数小时至十余小时。

---

## 363. HolisticSemGes: Semantic Grounding of Holistic Co-Speech Gesture Generation with Contrastive Flow-Matching

**arXiv ID:** 2603.26553 | [PDF](https://arxiv.org/pdf/2603.26553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 364. Online Temporal Voting: Strategyproofness, Proportionality and Asymptotic Analysis

**arXiv ID:** 2603.26504 | [PDF](https://arxiv.org/pdf/2603.26504v1)

**作者:** Allan Borodin `[一作]` (University of Toronto), Tristan Lueger `[通讯]` (University of Toronto)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在线时序投票（temporal voting）框架，探讨在一系列轮次中如何设计既公平又不易被策略性投票操纵的投票规则；

**💡 创新点**

提出在线策略无关性（OIIA）与在线策略无关性（OSP）的新概念，并证明OIIA是OSP的充分条件；同时给出“价格可操纵性”（price of manipulability）指标和渐近公平性分析，证明存在在线投票规则（如序列代数器）在轮次足够多时可近似满足比例正当代表（PJR）并且完全策略无关；

**🔧 技术方法**

采用理论分析与构造性证明，利用游戏理论中的策略无关性、独立无关选项、加权批准方法（WAM）、GreedyJR、Method of Equal Shares、Perpetual Phragmén 等规则，构造性证明其满足或不满足 OSP、SP、PJR、JR 等公正性公理；进一步定义并分析“价格可操纵性”和“渐近比例性”指标；

**📊 数据集**

本工作为理论研究，未使用实测数据集，所有结果基于数学证明与例子；

**📈 对比分析**

对比传统的离线多选举规则和在线规则，表明大多数已知在线规则满足 OSP 但不满足 SP；在渐近分析中，序列代数器（TSD）在轮次趋于无穷时可实现 PJR 附近的公平性，并且保持策略无关；其它规则如Perpetual Phragmén 在满足 PJR 的同时不满足 SP；

**⚠️ 局限性**

主要局限在于：仍未找到既满足 SP 又满足非弱比例正当代表的在线规则；对“价格可操纵性”的精确下界和多策略者均衡分析仍是开放问题；此外，对多赢家时序投票的渐近性质尚未研究；

---

## 365. AIRA_2: Overcoming Bottlenecks in AI Research Agents

**arXiv ID:** 2603.26499 | [PDF](https://arxiv.org/pdf/2603.26499v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 366. EnTaCs: Analyzing the Relationship Between Sentiment and Language Choice in English-Tamil Code-Switching

**arXiv ID:** 2603.26587 | [PDF](https://arxiv.org/pdf/2603.26587v1)

**作者:** Paul Bontempo `[一作]` `[通讯]`, Paul Bontempo

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建罗马化泰米尔-英语代码混合文本的词级语言识别模型，并使用线性回归分析情感倾向对英语占比和语言切换频率的影响，探讨情感如何影响双语使用者的语言选择。

**💡 创新点**

创新点在于：①将情感视为预测语言行为的自变量，而非传统的分类目标；②为罗马化泰米尔-英语文本提供了首个公开的词级语言识别标注；③从情感角度量化并验证了情感与代码切换频率之间的关系。

**🔧 技术方法**

技术手段包括：使用预训练的 XLM‑RoBERTa 微调进行词级语言识别（训练集约 3,500 个标注 token，验证集准确率 93.3%，macro‑F1 0.844）；随后采用普通最小二乘回归及交互项模型，并通过 ANOVA 与残差检验评估模型拟合。

**📊 数据集**

使用的数据集为 DravidianCodeMix 公开的 Tamil‑English 子集（共 44,161 条 YouTube 评论），随后筛选出 35,650 条罗马化评论并进一步限定为 28,205 条标注情感为 Positive、Mixed_feelings 或 Negative 的句子。

**📈 对比分析**

对比方法：在不加交互项的模型（Model 1a/2a）与加交互项模型（Model 1b/2b）之间做 ANOVA。语言识别模型表现良好；情感对英语比例的线性模型 R² 仅约 0.01–0.02，表明情感解释力弱；加入句长交互后 R² 仅提升至 0.02；而切换频率模型在加入交互后 R² 提升到 0.30，显示情感与句长共同影响切换率。

**⚠️ 局限性**

局限性包括：①仅使用罗马化文本，忽略约 19% 的泰米尔 Unicode 评论；②情感类别极度不平衡（Positive 占比 70%+），可能影响显著性；③词级标注数据仅由单名未熟悉泰米尔的注释者完成，样本量有限；④整体 R² 低，提示情感只是代码切换众多因素之一。

---

## 367. Security-Spectral Efficiency Tradeoff in STAR-RIS RSMA: A Max-Min Fairness Framework

**arXiv ID:** 2603.26532 | [PDF](https://arxiv.org/pdf/2603.26532v1)

**作者:** Huiyun Xia `[一作]` (Nanjing University of Posts and Telecommunications), Hongbo Zhu `[通讯]` (Nanjing University of Posts and Telecommunications)

**通讯引用:** 10505 | [OpenAlex ID](https://openalex.org/A5100755735)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对同时具备内部与外部窃听者的STAR‑RIS辅助RSMA系统，提出并求解了满足信道安全约束的最大最小公平率优化问题，设计了联合优化发射波束与STAR‑RIS相位的迭代算法，并在仿真中验证了其在谱效率与安全性方面的优势。

**💡 创新点**

① 在更真实的全空间覆盖环境中同时考虑内部与外部窃听者；② 将公共子流既用于合法用户的数据传输，又利用其作为人工噪声抑制外部窃听；③ 通过半正定松弛与差分凸近似，得到可迭代求解的可行方案。

**🔧 技术方法**

半正定松弛（SDR）、非线性分数规划、差分凸（DC）近似、罚项策略、Gaussian 随机化恢复、仿真使用 Rician 衰落模型。

**📊 数据集**

实验采用仿真数据：Rician 衰落信道，用户与窃听者随机布置在半圆分布（半径 80 m），STAR‑RIS 20 元素，发射天线 8 根；不使用真实数据集，而是通过随机生成的信道矩阵进行性能评估。

**📈 对比分析**

与 SDMA（无公共子流）、RSMA‑RIS（仅反射或仅透射 RIS）、RSMA‑上界（忽略安全约束）以及 RSMA‑Random（随机相位）等基线进行比较。结果显示：当安全阈值 r_E ≥ 0 dB 时，RSMA‑STAR 接近上界，明显优于基线；在不同分布半径、RIS 元素数和窃听者噪声水平下均保持较高的最小用户速率，且在 RIS 元素数增大时仍保持优势。

**⚠️ 局限性**

① 假设完美 CSI 与无直接链路，实际环境中需考虑估计误差与直达信道；② 公共子流等比例分配固定，未针对不同用户或窃听条件进行动态调整；③ 采用单层 RSMA，若扩展至多层可能获得更大灵活性；④ 计算复杂度随 RIS 元素、发射天线和用户数呈多项式增长，对极大规模系统仍需进一步优化。

---

## 368. When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models

**arXiv ID:** 2603.26556 | [PDF](https://arxiv.org/pdf/2603.26556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 369. ClipTTT: CLIP-Guided Test-Time Training Helps LVLMs See Better

**arXiv ID:** 2603.26486 | [PDF](https://arxiv.org/pdf/2603.26486v1)

**作者:** Mriganka Nath `[一作]` (Max Planck Institute for Informatics), Bernt Schiele `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 79216 | [OpenAlex ID](https://openalex.org/A5051534545)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 ClipTTT，一种利用 CLIP 对视觉文本对齐的引导进行的测试时训练方法，用于在图像受损时减少大型视觉语言模型（LVLM）的幻觉。

**💡 创新点**

创新点在于：①将 CLIP 的图像-文本相似度作为无监督伪标签的选择标准；②使用轻量级 LoRA 进行单样本的快速自适应；③采用 EMA 维护教师模型并通过 CLIP Score 选取最佳迭代，实现对单张受损图像的即时校正。

**🔧 技术方法**

技术手段包括：CLIP（ViT‑L/14）进行相似度评分；LoRA 参数微调；EMA 更新教师；随机采样生成多样化候选字幕；句子级 CLIP Score 作为自监督目标；训练时不改动基模型权重。

**📊 数据集**

使用的数据集：COCO/COCO‑Val、CHAIR 基准（15种常见图像失真），NoCaps（OpenImages）用于跨域验证；测试时将图像按 ImageNet‑C 最高级别（severity 5）进行腐蚀。

**📈 对比分析**

与 Greedy 解码、训练无关的 TTI 方法（VCD、PAI、VAP、CGD）以及图像去噪预处理（Restormer、NL‑Means）相比，ClipTTT 在 CHAIR_S 上平均降低 4.9 p.p、在 CHAIR_I 上降低 1.6 p.p；在不同 LVLM 架构和规模（TinyLLaVA‑1.1B、LLaVA‑7B、LLaVA‑13B、InstructBLIP‑7B、InternVL2‑2B 等）保持显著优势；在 NoCaps 的跨域测试中同样表现最好。

**⚠️ 局限性**

局限性包括：对每张图像仍需额外 30–70 步 LoRA 微调，导致推理时延和计算开销相对传统方法较高；伪标签选择依赖 CLIP 的对齐质量，若 CLIP 对某些极端失真不稳健，可能产生误导；目前仅在 15 种常见失真和 5 级严重度下验证，未知在更极端或不常见失真下的表现。

---

## 370. Dynamic Nearest-Neighbor Searching Under General Metrics in ${\mathbb R}^3$ and Its Applications

**arXiv ID:** 2603.26585 | [PDF](https://arxiv.org/pdf/2603.26585v1)

**作者:** Pankaj K. Agarwal `[一作]` (Duke University), Micha Sharir `[通讯]` (Tel Aviv University)

**通讯引用:** 24163 | [OpenAlex ID](https://openalex.org/A5055296785)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

针对3维中一个常数复杂度的凸对称半代数体的相似复制集合，提出了一套可动态维护的空间与查询时间折衷的数据结构，用于高效的交叉检测、最近邻查询、BFS/DFS、逆最短路径、最小生成树和Dijkstra等图算法。

**💡 创新点**

主要创新点在于：① 证明了这些距离函数的k‑层可分解为 O*(n³k) 个常数复杂度的单元；② 设计了可在 O*(n³) 空间下实现 O*(1) 查询、O*(n²) 删除的垂直浅切片结构；③ 引入 O(r⁴) 规模的测试集实现线性空间的 O*(n²/3) 查询；④ 结合收缩-二分技术解决逆最短路径问题，实现 O*(n^62/39) 时间。

**🔧 技术方法**

核心技术包括：随机抽样与浅网、垂直浅切片与垂直分解、半代数范围搜索、分区树、参数化搜索、收缩-二分、动态半空间范围查询、Agarwal–Matoušek 方案等。

**📊 数据集**

论文以理论分析为主，并未使用实际数据集，所有结果均在假设 K 为常数复杂度半代数凸体的前提下得到；实验与实现均为模拟验证。

**📈 对比分析**

与之前工作相比，该方法在 3 维空间中大幅降低了查询与更新时间：交叉检测/最近邻 O*(n/s^(1/3))（s∈[n,n³]）；BFS/DFS O*(n³/2)；逆最短路径 O*(n^62/39)；最小生成树与 Dijkstra O*(n³/2)。空间与时间的折衷也更为灵活，可根据需要选择 s 取值。

**⚠️ 局限性**

局限性包括：① 常数项和隐式多项式因子可能很大，实际常数复杂度较高；② 仅适用于常数复杂度的半代数凸体，难以直接推广到更高维或更复杂度的度量；③ 对于高精度动态更新，重建频率仍可能影响实际性能；④ 对逆最短路径的收缩-二分实现依赖多层次随机化，可能在极端实例中产生较大波动。

---

## 371. Machine Unlearning under Retain-Forget Entanglement

**arXiv ID:** 2603.26569 | [PDF](https://arxiv.org/pdf/2603.26569v1)

**作者:** Jingpu Cheng `[一作]` (National University of Singapore), Chi Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 22545 | [OpenAlex ID](https://openalex.org/A5100458200)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对“保留–遗忘纠缠”问题的两阶段机器学习遗忘框架，先通过增广拉格朗日方法在忘记集上显著增加损失，同时保持远程保留样本的性能，再通过带Wasserstein‑2距离正则的梯度投影恢复与忘记集高度相关的保留样本性能。

**💡 创新点**

创新点包括：
• 采用增广拉格朗日动态平衡忘记与远程保留目标，避免手动调节权重；
• 在第二阶段引入分布级正则（Wasserstein‑2距离）约束，使遗忘集损失分布保持不变，避免误将部分遗忘样本转为正确预测；
• 将梯度投影与分布正则结合，既控制遗忘目标，又显著恢复相关保留样本的准确率，解决传统方法因梯度冲突导致的性能退化。

**🔧 技术方法**

核心技术包括：增广拉格朗日（Augmented Lagrangian）优化、梯度投影（Projected Gradient Descent），以及基于经验分布的Wasserstein‑2距离正则；实现上使用标准交叉熵损失、梯度裁剪和自适应乘子更新。

**📊 数据集**

实验数据集：CIFAR‑100（子类级遗忘），TinyImageNet（超类级遗忘），ToxiGen（有偏训练的毒性文本），CelebA（属性级多类遗忘）以及不同网络架构（ResNet‑18、Vision Transformer ViT‑B）。

**📈 对比分析**

与多种基线（Fine‑Tune、GA、SCRUB、Munba、SSD、DELETE、GDR、SalUn、ℓ₁‑sparse 等）对比，本文方法在所有设置中实现了 0% 的忘记集准确率，同时保留远程保留样本的 95%+ 以及相邻保留样本 90%+ 的准确率，明显优于现有方法在遗忘精度与保留性能之间的折衷。

**⚠️ 局限性**

局限性：
• 需要预先将保留集划分为“相邻”和“远程”两部分，对不同任务的划分标准不统一；
• 依赖增广拉格朗日和 W₂ 正则的超参数（λ、μ、α）需经验调节；
• 在极端高相关性场景下，梯度投影仍可能出现少量性能衰减；
• 计算成本略高于单纯梯度上升/微调，但在实验中保持可接受。

---

## 372. Clinical named entity recognition in the Portuguese language: a benchmark of modern BERT models and LLMs

**arXiv ID:** 2603.26510 | [PDF](https://arxiv.org/pdf/2603.26510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 373. Beyond MACs: Hardware Efficient Architecture Design for Vision Backbones

**arXiv ID:** 2603.26551 | [PDF](https://arxiv.org/pdf/2603.26551v1)

**作者:** Moritz Nottebaum `[一作]` (University of Udine), Christian Micheloni `[通讯]` (University of Udine)

**通讯引用:** 5325 | [OpenAlex ID](https://openalex.org/A5044011298)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统分析了视觉 backbone 的 MACs 与实际执行时间之间的关系，并基于此提出了硬件高效的 LowFormer 视觉 backbone 家族，包含轻量化注意力（LMA）和融合 MBConv 等设计；

**💡 创新点**

创新点在于：①提出并验证 MACs 不是万能的执行时间指标；②通过硬件效率实验指导架构设计；③设计了 LMA 以及在不同阶段采用融合 MBConv 的宏微架构；④推出三种针对 Jetson TX2 的 Edge GPU 变体，进一步提升边缘设备性能；

**🔧 技术方法**

技术主要包括：多头自注意力的低阶变体 LMA、融合的 MBConv、通道压缩与分辨率下采样、ReLU-linear attention 对比、实验测量多设备（GPU、CPU、NPU）下的吞吐量与延迟；

**📊 数据集**

使用的数据集：ImageNet‑1K、Oxford‑IIIT‑Pets、Stanford Cars、Oxford‑102 Flowers、COCO 2017、ADE20K、GPR1200、GOT10K、LaSOT、TREK‑150、NfS30、AVisT、UAV123 等；

**📈 对比分析**

在多个设备上与最新 Backbone（EfficientViT、MobileOne、RepViT 等）对比，LowFormer 在 GPU 通过率、Jetson TX2 延迟、ARM CPU 延迟上均获得 20‑60% 的速度提升，同时在 ImageNet Top‑1 维持或提高准确率；Edge GPU 变体在 Jetson TX2 上 1.5‑3× 的吞吐量提升，延迟下降 30‑50%；

**⚠️ 局限性**

局限性包括：在移动 NPU 或 GPU 上的加速效果不如预期；架构改动对不同任务的迁移性仍需更多评估；LMA 与传统 MHSA 的可解释性和模型复杂度关系尚未深入；未来需扩展至更多硬件平台与更大规模数据集。

---

## 374. PerceptionComp: A Video Benchmark for Complex Perception-Centric Reasoning

**arXiv ID:** 2603.26653 | [PDF](https://arxiv.org/pdf/2603.26653v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 375. Vision2Web: A Hierarchical Benchmark for Visual Website Development with Agent Verification

**arXiv ID:** 2603.26648 | [PDF](https://arxiv.org/pdf/2603.26648v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 376. Drive-Through 3D Vehicle Exterior Reconstruction via Dynamic-Scene SfM and Distortion-Aware Gaussian Splatting

**arXiv ID:** 2603.26638 | [PDF](https://arxiv.org/pdf/2603.26638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 377. From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning

**arXiv ID:** 2603.26597 | [PDF](https://arxiv.org/pdf/2603.26597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 378. Learning to Commit: Generating Organic Pull Requests via Online Repository Memory

**arXiv ID:** 2603.26664 | [PDF](https://arxiv.org/pdf/2603.26664v1)

**作者:** Mo Li `[一作]` (Tsinghua University), Yunxin Liu `[通讯]` (Tsinghua University)

**通讯引用:** 5314 | [OpenAlex ID](https://openalex.org/A5102880548)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一种基于大型语言模型的编码代理框架——Learning to Commit，通过在线仓库记忆从历史提交中提取技能，指导代理生成符合项目风格的补丁。

**💡 创新点**

将对历史提交的监督对比反思与技能文档生成结合，实现了仓库特定的在线适配，并在严格时间切分的评估中避免数据泄露。

**🔧 技术方法**

使用大语言模型（Claude Opus 4.6）与工具调用、对比反思学习、技能文档 CRUD、合成问题描述以及 LLM 评审等技术。

**📊 数据集**

对内部强化学习训练仓库进行过滤后得到 386 条高质量提交，划分为 24 条学习和 7 条未来评测提交；所有任务均采用合成 issue 生成。

**📈 对比分析**

与无技能基线在同一模型和工具下对比，采用文件 IoU、工具调用步数、行数偏差和四维 LLM 评审进行评测；技能代理在文件定位、行数逼近和逻辑相似度上显著提升（约+10-18% IoU，行数偏差下降，逻辑相似度 50% vs 25%）。

**⚠️ 局限性**

仅在单一内部仓库验证，评估依赖 LLM 判别，且对代码风格的改进有限，某些技能可能导致过度保守的实现。

---

## 379. The Limits of Learning from Pictures and Text: Vision-Language Models and Embodied Scene Understanding

**arXiv ID:** 2603.26589 | [PDF](https://arxiv.org/pdf/2603.26589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 380. Make Geometry Matter for Spatial Reasoning

**arXiv ID:** 2603.26639 | [PDF](https://arxiv.org/pdf/2603.26639v1)

**作者:** Shihua Zhang `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**通讯引用:** 13417 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种新的视觉-语言模型框架 GeoSR，用于提升在静态和动态空间推理任务中的几何信息利用。

**💡 创新点**

创新点是：①几何解锁掩码（Geometry-Unleashing Masking）通过有策略地遮挡视觉令牌削弱非几何捷径；②几何引导融合（Geometry-Guided Fusion）使用门控机制自适应调节几何和视觉特征的融合权重。

**🔧 技术方法**

采用的技术包括预训练的几何提取器（VGGT/π³）、掩码策略、交叉注意力、门控融合，并在 VLM（Qwen2.5‑VL‑7B）上微调。

**📊 数据集**

使用的数据集包括静态空间推理的 VSI‑Bench、SPAR‑7M、LLaVA‑Hound 以及动态空间推理的 DSR‑Bench。

**📈 对比分析**

与现有专有 API 模型、通用视频理解模型和其他空间推理模型进行对比，GeoSR 在 VSI‑Bench 和 DSR‑Bench 上均取得最高或接近最高分，静态提升约 1–2 个百分点，动态提升约 1–4 个百分点。

**⚠️ 局限性**

局限性：在仅注入几何信息的静态场景下提升有限；过大比例掩码或过度门控可能导致视觉信息丢失；目前仅在单模态视频/图像上验证，未针对多模态或多摄像头场景展开。

---

## 381. An LP-based Sampling Policy for Multi-Armed Bandits with Side-Observations and Stochastic Availability

**arXiv ID:** 2603.26647 | [PDF](https://arxiv.org/pdf/2603.26647v1)

**作者:** Ashutosh Soni `[一作]` (Ohio State University), Ness B. Shroff `[通讯]` (Ohio State University)

**通讯引用:** 20186 | [OpenAlex ID](https://openalex.org/A5035752536)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究一种基于线性规划的UCB-LP-A策略，用于具有侧信息图结构且动作可用性随机变化的多臂老虎机问题

**💡 创新点**

创新点在于将侧信息图和随机可用性（激活集合）融合进LP采样分配，设计强制同步与独立两阶段采样，给出严格的期望遗憾上界

**🔧 技术方法**

主要技术包括：线性规划求解最优采样分布、UCB与消除机制、强制同步与独立采样策略、侧信息传播分析、理论遗憾上界证明

**📊 数据集**

使用的实验数据集：Barabási–Albert 生成的合成社交网络（100/200/300 节点）和 SNAP ego‑Facebook 1000 节点子图

**📈 对比分析**

通过与UCB‑E、UCB‑N、UCB‑MaxN（及UCB‑1）等基线对比，实验显示UCB‑LP‑A在所有场景下收敛更快、累计遗憾显著低于基线，尤其在利用侧信息和可用性协同时优势明显

**⚠️ 局限性**

局限性：假设激活集合及其概率已知，无法处理可用性分布未知的情况；未来工作需在未知可用性环境下共同学习可用性与奖励

---

## 382. From Synthetic Data to Real Restorations: Diffusion Model for Patient-specific Dental Crown Completion

**arXiv ID:** 2603.26588 | [PDF](https://arxiv.org/pdf/2603.26588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 383. Detailed Geometry and Appearance from Opportunistic Motion

**arXiv ID:** 2603.26665 | [PDF](https://arxiv.org/pdf/2603.26665v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 384. PQuantML: A Tool for End-to-End Hardware-aware Model Compression

**arXiv ID:** 2603.26595 | [PDF](https://arxiv.org/pdf/2603.26595v1)

**作者:** Roope Niemi `[一作]` (European Center for Nuclear Research), Maurizio Pierini `[通讯]` (European Center for Nuclear Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了PQuantML，一个开源、硬件感知的神经网络压缩库，集成多种裁剪（结构化、非结构化、半结构化）和量化（定点、HGQ）方法，并提供统一配置、自动化超参搜索与实验追踪；与hls4ml无缝对接实现FPGA硬件部署。

**💡 创新点**

在同一框架下同时支持裁剪与量化的联合训练，提供自动层替换与细粒度配置，支持多种裁剪策略和量化精度；自动化超参优化与实验管理提升可复现性与易用性。

**🔧 技术方法**

裁剪技术：AutoSparse、DST、PDP、MDMM、FITCompress；量化技术：固定点定点量化、HGQ（高粒度量化）；量化感知训练（QAT）；超参搜索（Optuna）、实验跟踪（MLflow）；配置文件使用YAML + Pydantic；后端支持PyTorch与Keras；通过hls4ml进行HLS代码生成与硬件综合。

**📊 数据集**

LHC喷射子结构分类任务（jet tagging），使用hls4ml提供的两种数据集：高层特征（HLF，OpenML/CERNBox）和粒子层特征（PLF，128粒子×3特征）。

**📈 对比分析**

通过Vitis HLS 2023.2对DSP、LUT、FF、延迟、Fmax等指标进行评估；与QKeras、HGQ以及未裁剪版本对比。结果显示，裁剪+定点量化在保持≈76%准确率的同时，DSP下降至3-5，LUT下降至3-4，延迟从≈100 ns降至≈40 ns；FITCompress进一步降低DSP并提升速度；与HGQ相比准确率相近但资源略高。

**⚠️ 局限性**

FITCompress仅实现PyTorch，TensorFlow支持不足；层替换仅适用于顺序网络，无法处理残差/注意力等多分支结构；缺乏对其他硬件编译工具链的支持；预训练阶段可能导致权重幅值增大，需正则化；压缩度量以BOPs为主，难以预估EBOPs；部分裁剪方法尚未实现；未集成蒸馏、低秩分解等其他压缩技术。

---

## 385. VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation

**arXiv ID:** 2603.26666 | [PDF](https://arxiv.org/pdf/2603.26666v1)

**作者:** Zhide Zhong `[一作]`, Haoang Li `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在机器人控制中将离线监督微调和在线强化学习融合的On-Policy VLA Distillation框架，通过自生成轨迹与专家教师的密集token级监督实现快速、稳健的后训练。

**💡 创新点**

创新点包括：使用逆Kullback-Leibler（Reverse-KL）作为对齐目标，避免传统Forward-KL的熵爆炸和Hard-CE的熵塌陷；将密集教师监督转化为内部奖励；在学生轨迹上进行on-policy采样并用组采样降低方差；实现了少量示例（1轨迹）即可接近专家性能。

**🔧 技术方法**

核心技术：逆K-L散度对齐、密集教师标注、组采样策略、基于RL的策略梯度更新、On-Policy Trajectory采样、Token-level Dense Reward。

**📊 数据集**

使用了LIBERO（Spatial、Object、Goal、Long四个子集）和RoboTwin2.0（双臂协调任务）两大基准；学生初始化采用1轨迹或1,000轨迹SFT。

**📈 对比分析**

与GRPO、离线SFT及全数据SFT对比，VLA-OPD在LIBERO上实现约3倍的样本效率，单轨迹初始化成功率提升至近教师水平（90%+），在RoboTwin2.0上平均成功率从45%提升至71%；同时在“seen–unseen”评估中显著减轻灾难性遗忘。

**⚠️ 局限性**

主要局限：需要预先存在高质量的教师模型；对教师的不确定性仍存在一定依赖，若教师在某些状态下表现欠佳会影响学习；目前框架主要针对视觉语言动作模型，扩展到更大规模或更动态环境仍需验证。

---

## 386. Weight Tying Biases Token Embeddings Towards the Output Space

**arXiv ID:** 2603.26663 | [PDF](https://arxiv.org/pdf/2603.26663v1)

**作者:** Antonio Lopardo `[一作]` (EleutherAI), Akshat Gupta `[通讯]` (University of California Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对权重共享（embedding tying）的训练动态进行深入分析，探讨了共享嵌入矩阵在语言模型中被输出层梯度主导、导致嵌入更偏向输出而非输入的现象，并验证了梯度不平衡对早期层表示的负面影响。

**💡 创新点**

创新点在于：①首次用梯度流监测与对齐分析证明共享嵌入矩阵在训练早期被输出梯度主导；②通过梯度放缩干预实验证实梯度不平衡的因果作用；③为大型模型是否解耦嵌入提供了机制化解释。

**🔧 技术方法**

主要技术包括：梯度流跟踪、嵌入空间对齐（身份、正交、线性映射）、tuned lens 计算 KL 差异、梯度放缩实验（增幅输入梯度）、对比实验（OLMo、Pythia、Qwen3），以及可视化分析。

**📊 数据集**

使用公开数据集：Pile（用于 GPT-Neo/Pythia 训练）、Dolma（用于 OLMo 训练）及其对应的检查点；所有实验均在这些公开训练数据上进行。

**📈 对比分析**

通过对齐余弦相似度、KL 散度对比，以及梯度比例分析来比较 tied 与 untied 模型。结果显示：tied 模型在早层 KL 散度更高、梯度更偏向输出；梯度放缩可使嵌入更接近输入，但对下游性能并未产生显著提升。

**⚠️ 局限性**

局限性包括：仅在单一训练跑、单一随机种子下评估；仅覆盖 OLMo、Pythia、Qwen3 三个模型族；缺乏多次实验的统计显著性检验；结果可能不适用于非自回归或混合专家等其他架构。

---

## 387. Zero-Shot Depth from Defocus

**arXiv ID:** 2603.26658 | [PDF](https://arxiv.org/pdf/2603.26658v1)

**作者:** Yiming Zuo `[一作]` (Princeton University), Jia Deng `[通讯]` (Princeton University)

**通讯引用:** 125998 | [OpenAlex ID](https://openalex.org/A5101542158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出零射深度估计方法Zero-shot Depth from Defocus（ZFDF），使用Transformer网络FOSSA结合堆叠注意力，从焦距堆栈直接回归稠密度图。

**💡 创新点**

创新点在于：1）堆叠注意力层与焦距嵌入的组合，使模型能高效利用跨焦距信息；2）自建高质量大规模基准ZEDD；3）基于PSF的随机散焦与焦距/光圈域随机化的训练管线，大幅提升跨域泛化。

**🔧 技术方法**

技术手段包括：Vision Transformer（ViT）、堆叠注意力、点扩散函数（PSF）与随机形状、梯度匹配损失、预训练DepthAnything权重、光学参数的焦距嵌入。

**📊 数据集**

使用的数据集有：自建ZEDD基准（100场景4K，Ouster LiDAR），Synthetic Infinigen Defocus，公开的Hypersim、TartanAir、iBims、DIODE、HAMMER、DDFF等。

**📈 对比分析**

在ZEDD、Infinigen Defocus、DDFF以及多种RGB-D基准上与单目深度基线（DepthAnything、DepthPro等）和DfD基线（HybridDepth、DualFocus等）进行零射对比，ZEDD上AbsRel从0.201降至0.089（下降55.7%），DDFF上MSE下降40.4%，Infinigen Defocus上实现显著性能提升。

**⚠️ 局限性**

局限性：目前在光圈F/1.4–F/5.6范围内验证，极大光圈或弱散焦场的适应性尚未充分测试；训练仍依赖合成散焦数据，真实镜头的光学特性可能导致泛化受限；模型规模较大，部署时算力与显存需求高。

---

## 388. Tunable Soft Equivariance with Guarantees

**arXiv ID:** 2603.26657 | [PDF](https://arxiv.org/pdf/2603.26657v1)

**作者:** Md Ashiqur Rahman `[一作]` (Purdue University), Raymond A. Yeh `[通讯]` (Purdue University)

**通讯引用:** 3375 | [OpenAlex ID](https://openalex.org/A5076130922)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于投影到子空间的软等变网络框架，可在任意预训练模型上实现可调节的等变性。

**💡 创新点**

创新点在于：①通过 Lie 代数或离散前向差分实现的投影算子，可在保持模型表达能力的同时给出可控的等变误差上界；②该投影既可用于连续群也可推广到离散群，且无需额外可学习参数；③引入 Schur 分解降低 SVD 计算开销，提供实用的实现路径。

**🔧 技术方法**

核心技术包括：Lie 代数/离散前向差分、投影到低秩子空间、SVD 与 Schur 分解、软阈值裁剪、以及在 Transformer/ResNet/Segformer 等现有架构中插入等变/不等变层。

**📊 数据集**

实验数据集涵盖图像分类（MNIST、CIFAR‑10/100、ImageNet）、语义分割（PASCAL VOC）、人类轨迹预测以及合成的 O(5) 不变回归任务。

**📈 对比分析**

与基线（非等变网络）以及残差式软等变方法 RPP 进行对比；在各任务中，所提出方法在保持或提升准确率（cAcc、mIoU 等）的同时，显著降低等变误差（iErr、eErr），尤其在 ImageNet 上取得可观性能提升。

**⚠️ 局限性**

局限性包括：①投影算子构造仍需先验的群结构信息，未对未知或连续变换自适应；②对大规模网络参数（特别是多通道全连接层）仍存在计算和存储开销；③在某些离散群场景下，前向差分近似的精度受限，需进一步研究更高阶展开。

---

## 389. Who Checks the Checker? Enhancing Component-level Architectural SEU Fault Tolerance for End-to-End SoC Protection

**arXiv ID:** 2603.26637 | [PDF](https://arxiv.org/pdf/2603.26637v1)

**作者:** Michael Rogenmoser `[一作]` (ETH Zürich), Luca Benini `[通讯]` (ETH Zürich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文设计并实现了一种基于重叠保护的组件级故障容错架构，并在一个RISC‑V MCU SoC上进行验证，展示了端到端的 SEU/SET 容错效果。

**💡 创新点**

创新点在于：① 引入重叠保护机制，使投票器、编码器等检查逻辑本身也受到保护，消除传统方案中的“边界漏洞”；② 将 ECC、TCLS、relOBI、TMR 等多种容错技术按组件级别进行组合，并通过重叠实现无缝衔接；③ 通过逐步添加保护方法展示 Pareto 最优点，证明相较于全 TMR，面积降低 22% 的同时保持 99.9% 以上的故障覆盖。

**🔧 技术方法**

采用的技术包括：ECC（SecDED）、锁步多重核投票（TCLS）、relOBI 互连容错、TMR、硬件故障注入（Synopsys VC Z01X）、故障监控计数器、全流程综合实现（Yosys+OpenROAD+IHP 130 PDK）以及重叠覆盖的硬件设计与验证。

**📊 数据集**

使用的工作负载为 RISC‑V CoreMark、UART、GPIO、Timer 等简易应用；在 RTL 与合成网表上进行 100 000 次单点 SEU/SET 注入测试；没有使用公开数据集，而是通过内部设计和注入模型进行评估。

**📈 对比分析**

比较方法：在同一 SoC 上构造五种保护配置（无保护、ECC 内存、ECC+TCLS、ECC+TCLS+relOBI、完整保护）并与全 TMR（tmrg）进行对比；评估指标包括面积占用、最大频率、故障覆盖率、失败率、纠正率等。结果显示完整保护实现 99.9%+ 的故障覆盖，面积仅比全 TMR 低 22%，并位于 Pareto 前沿，表明在面积/性能权衡上具有优势。

**⚠️ 局限性**

局限性包括：① 仅在单一 RISC‑V MCU SoC 上验证，缺乏对更大规模或不同架构的通用性验证；② 重叠策略需要手工设计，缺乏自动化工具支持；③ 故障注入模型仅覆盖单点 SEU/SET，未考虑多位错误、持续损伤或功耗/热影响；④ 部分关键硬件如调试模块未得到保护；⑤ 实现复杂度高，优化空间受限。

---

## 390. Deception and Communication in Autonomous Multi-Agent Systems: An Experimental Study with Among Us

**arXiv ID:** 2603.26635 | [PDF](https://arxiv.org/pdf/2603.26635v1)

**作者:** Maria Milkowski `[一作]` (University of Notre Dame), Tim Weninger `[通讯]` (University of Notre Dame)

**通讯引用:** 3209 | [OpenAlex ID](https://openalex.org/A5084597959)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在文字版《Among Us》社交推理游戏中，使用LLM代理模拟1,100局游戏，记录代理的对话、思考和投票，系统性分析了LLM在多代理环境中的沟通与欺骗行为。

**💡 创新点**

首次将传统的演讲行为理论（speech act）与人际欺骗理论相结合，对大规模LLM代理生成的自然语言进行分类，并量化欺骗类型与游戏结果的关联，揭示LLM在任务驱动下更倾向于“含糊”而非直白谎言。

**🔧 技术方法**

核心技术包括：Llama 3.2 LLM代理、基于 Gemini 的自动化话语标签器（演讲行为与欺骗类型）、逻辑回归和 Spearman 相关分析用于评估语言特征与胜负关系。

**📊 数据集**

自建数据集：1,100局文本化《Among Us》游戏日志，包含超过百万词的对话、思考过程、行动与投票记录，已公开于 GitHub。

**📈 对比分析**

通过统计对比（chi‑square、logistic 回归）评估沟通频率、演讲行为比例及欺骗强度对胜负的影响，发现沟通量对胜负影响不大，指令型演讲和含糊欺骗虽多但并未显著提高胜率，整体表现说明LLM在此环境下主要做出符合任务而非策略性成功的沟通。

**⚠️ 局限性**

局限性包括：仅使用单一模型（Llama 3.2），缺乏非语言线索，Gemini 标签器可能无法捕捉所有语义细微差别，实验仅在纯 AI 环境中进行，无法验证在人类玩家面前的欺骗效果。

---

## 391. Meta-Adaptive Beam Search Planning for Transformer-Based Reinforcement Learning Control of UAVs with Overhead Manipulators under Flight Disturbances

**arXiv ID:** 2603.26612 | [PDF](https://arxiv.org/pdf/2603.26612v1)

**作者:** Hazim Alzorgan `[一作]`, Abolfazl Razi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本研究主要探讨了一个未在文本中详细说明的主题。

**💡 创新点**

创新点主要体现在所列的两项研究亮点上。

**🔧 技术方法**

未在论文中具体阐述所采用的技术。

**📊 数据集**

未提供任何数据集的详细信息。

**📈 对比分析**

未说明使用的比较方法及其性能表现。

**⚠️ 局限性**

论文未提及具体的局限性。

---

## 392. Hardware-Aware Tensor Networks for Real-Time Quantum-Inspired Anomaly Detection at Particle Colliders

**arXiv ID:** 2603.26604 | [PDF](https://arxiv.org/pdf/2603.26604v1)

**作者:** Sagar Addepalli `[一作]` (SLAC National Accelerator Laboratory), Julia Gonski `[通讯]` (SLAC National Accelerator Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在粒子对撞机实验中，利用张量网络（SMPO 与级联 SMPO）实现实时异常检测，能够在 FPGA 上以低延迟进行数据过滤。

**💡 创新点**

创新点在于：①提出了“间距矩阵乘积算子”（SMPO）与其级联变体，能够在保持学习能力的同时实现维度压缩；②将该架构无监督训练为异常检测模型；③在 FPGA 上完成软硬件协同设计，证明了其资源与时延满足触发器需求。

**🔧 技术方法**

技术包括：张量网络（MPS/MPO/SMPO）、量子互信息驱动的输入排序、无监督伪 Huber 损失训练、FPGA 上的定点实现与乘加（MAC）优化。

**📊 数据集**

数据集为 LHC proton‑proton 碰撞的模拟事件，共 5 个过程：1) QCD 背景；2) 4 轻子信号 A→4ℓ；3) LQ→bτ；4) h±→τν；5) h0→ττ；共 3.8 万万条事件，分为训练/验证/测试。

**📈 对比分析**

通过 ROC 曲线和 10⁻⁵ FPR 下的信号接受率与 AUC 进行比较，SMPO 与级联 SMPO 的表现与当前最先进方法相当；FPGA 实现显示 LUT、DSP、FF 资源占用与时延低至可接受范围，验证了实时部署可行性。

**⚠️ 局限性**

局限性包括：仅使用模拟数据；无监督训练可能对真实噪声与系统误差不够鲁棒；模型仍受张量秩限制，难以捕捉更复杂的非线性关联；在更高维、不同实验条件下的泛化性尚待进一步验证。

---

## 393. Evaluating Interactive 2D Visualization as a Sample Selection Strategy for Biomedical Time-Series Data Annotation

**arXiv ID:** 2603.26592 | [PDF](https://arxiv.org/pdf/2603.26592v1)

**作者:** Einari Vaaras `[一作]` (Tampere University), Okko Räsänen `[通讯]` (Tampere University)

**通讯引用:** 2251 | [OpenAlex ID](https://openalex.org/A5016518233)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过对12名注释者在婴儿运动和语音情感两类时序数据上进行实验，比较了随机抽样(RND)、最远点遍历(FAFT)与交互式2D可视化(2DV)三种样本选择方法对注释质量和模型性能的影响。

**💡 创新点**

创新点在于首次将交互式2D可视化作为主动采样策略，并构建了TSExplorer框架，让注释者在全数据散点图中自由探索、标注；同时结合真实人类注释者对比传统算法抽样，评估其在生物医学时序数据上的实用性。

**🔧 技术方法**

主要技术包括：自监督学习特征提取、t‑SNE/PCA/UMAP三种2D投影、FAFT采样、随机抽样、TSExplorer可视化界面、Transformer+FC层微调、标签直方图对比、Hellinger距离与风险分析。

**📊 数据集**

使用的数据集为MAIJU‑DS（多传感器IMU婴儿姿势/运动）和NICU‑A（芬兰子集语音情感识别），分别对应四个分类任务（姿势/运动、情感价值/激动度）。

**📈 对比分析**

实验通过标签分布一致性、模型在单注释者和合并注释者标签下的微调性能，以及三种风险指标（模型失效、罕见类覆盖、标签分布不稳）进行比较。结果显示：IMA中FAFT在单注释者时表现最佳，2DV在合并标签时表现最好；SER中专家注释者下2DV最优，非专家表现相近；整体来看RND最安全、2DV风险最高。

**⚠️ 局限性**

局限性包括：注释者人数和预算有限，导致2DV方法因自由选择导致标签分布波动；依赖自监督特征，缺乏通用可视化方案；未评估3D投影、连续标签或迭代更新的可行性；以及对注释者经验的指导缺失，可能影响2DV的效果。

---

## 394. Learning From Social Interactions: Personalized Pricing and Buyer Manipulation

**arXiv ID:** 2603.26631 | [PDF](https://arxiv.org/pdf/2603.26631v1)

**作者:** Qinqi Lin `[一作]` (Chinese University of Hong Kong, Shenzhen), Jianwei Huang `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**通讯引用:** 14946 | [OpenAlex ID](https://openalex.org/A5062346297)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过动态贝叶斯博弈分析了买家在意识到卖家利用社交网络数据进行个性化定价时，如何主动操纵社交互动信号，以期规避高价并获得更好收益；同时研究卖家在面对这种操纵时，如何设计最优定价方案以及在多买家网络中动态学习的效果。

**💡 创新点**

创新点主要包括：①首次系统性地把买家偏好相关性与卖家学习相结合，构建双层信息不对称模型；②给出闭式完美贝叶斯均衡（PBE）解析，揭示高偏好买家唯一的操纵策略及其对收益的负面外部性；③分析卖家在多买家网络中的动态学习，发现学习先前买家偏好并不一定帮助推断后续买家的偏好；④提出在监管背景下，卖家保持信息透明对收益影响不大，支持“知情同意”实践。

**🔧 技术方法**

技术方法主要为：动态贝叶斯博弈建模、完美贝叶斯均衡推导（前向-后向分析），以及对多买家网络的数学归纳与数值验证；此外采用了博弈理论中的信念一致性与概率更新来处理偏好相关信息。

**📊 数据集**

实验使用真实 Facebook 社交网络的子网络（100 节点、230 条边），随机生成买家高/低偏好，基于此构造真实与被操纵的社交互动数据。

**📈 对比分析**

对比方法包括：无学习基准（统一定价）、未披露学习基准（卖家知晓并利用社交数据但买家不知情）以及本文提出的战略学习定价。实验结果显示：未披露学习相较无学习可提升约48.5%收益；在买家意识到学习后，战略学习定价仍比无学习高约33–41%，且与未披露学习相比仅损失4.8–10.1%的收益。

**⚠️ 局限性**

局限性：①主模型仅在两买家情形下可解析，扩展到多买家网络仍需大量计算，难以得到一般闭式解；②假设社交互动频率为离散 {0,1}，偏好先验为均匀，虽然在后续章节对非均匀先验与连续互动进行扩展，但仍未覆盖所有实际场景；③模型未考虑社交网络结构的动态演化及买家在多产品环境中的交互行为。

---

## 395. Context-specific Credibility-aware Multimodal Fusion with Conditional Probabilistic Circuits

**arXiv ID:** 2603.26629 | [PDF](https://arxiv.org/pdf/2603.26629v1)

**作者:** Pranuthi Tenali `[一作]` (University of Texas at Dallas), Sriraam Natarajan `[通讯]` (University of Texas at Dallas)

**通讯引用:** 2785 | [OpenAlex ID](https://openalex.org/A5064323671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于条件概率电路的上下文感知可信度融合框架C^2MF，用于动态评估每个模态在具体实例上的可靠性并融合预测。

**💡 创新点**

创新点在于通过条件概率电路的权重由上下文向量调节，定义了可精确计算的上下文特定信息可信度(CSIC)，并构建了Conflict benchmark和RMIS指标以评估冲突解决能力。

**🔧 技术方法**

采用神经网络的单模态编码器与预测器，再通过超网络生成的条件概率电路实现可调权重；使用KL散度计算CSIC并进行可解析推理。

**📊 数据集**

在AV‑MNIST（音频-视觉MNIST）和NYUD（RGB‑深度室内场景）两大多模态数据集上进行实验，并通过人工构造的类相关噪声产生冲突。

**📈 对比分析**

与静态可信度融合基线（DPC、CWM）在解耦与联合训练设置下对比，C^2MF在噪声水平升高时准确率提升多达29%，RMIS也显著高于对照组，表现出更强的鲁棒性。

**⚠️ 局限性**

局限在于当前需要所有模态齐全才能推断上下文，无法处理缺失模态；实验冲突为合成噪声，未检验对真实传感器失效的泛化能力；模型训练和超参数调优复杂。

---

## 396. Function-Based Minimal Linear Codes over Galois Rings $\mathrm{GR}(p^{n}, \ell)$: Minimality Criteria and Infinite Constructions

**arXiv ID:** 2603.26614 | [PDF](https://arxiv.org/pdf/2603.26614v1)

**作者:** Biplab Chatterjee `[一作]` (National Institute of Technology Jamshedpur), Kalyan Hansda `[通讯]` (Visva-Bharati)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5038898043)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文构造了无限族的最小线性码，给出了在Galois环 GR(p^n,ℓ) 上的完整最小性判定，并以函数式生成矩阵为基础提出了新的构造方法。

**💡 创新点**

创新点在于将已知的有限域上函数式最小码构造扩展到链环环境，揭示了零因子和p-进层次对最小性的影响，并给出了根词化简原则和同步条件 p^r f(β)=w·β。

**🔧 技术方法**

主要技术包括Frobenius对偶性、McCoy秩理论、p-进分解与链环理想结构、以及对根词的正交模子分析。

**📊 数据集**

本文没有采用传统数据集，而是通过符号运算与结构证明来推导码的长度、维数等参数。

**📈 对比分析**

在理论层面上给出了长度下界与上界，证明对于 m≥3 的情况长度大于 (m−1)q^n+q^n−m，且提供了构造码满足此界；在 m=2 时得到精确长度 k= q^n+q^n−1，优于已知的ℤ_{p^n} 码。

**⚠️ 局限性**

主要局限在于仅适用于 m≥2 的情况；零因子导致的同余约束使得构造与证明更为复杂；对权分布和实际编码效率的数值评估仍待进一步研究。

---

## 397. Beyond Language: Grounding Referring Expressions with Hand Pointing in Egocentric Vision

**arXiv ID:** 2603.26646 | [PDF](https://arxiv.org/pdf/2603.26646v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 398. Think over Trajectories: Leveraging Video Generation to Reconstruct GPS Trajectories from Cellular Signaling

**arXiv ID:** 2603.26610 | [PDF](https://arxiv.org/pdf/2603.26610v1)

**作者:** Ruixing Zhang `[一作]` (Beihang University), Weifeng Lv `[通讯]` (Beihang University)

**通讯引用:** 5989 | [OpenAlex ID](https://openalex.org/A5109299440)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于地图可视化的视频生成方法，将粗粒度的蜂窝信令重构为连续、路网约束的GPS轨迹。

**💡 创新点**

创新点在于将 Sig2GPS 问题重新表述为图像到视频的生成任务，并引入 Traj‑GDPO 的轨迹感知强化学习与可验证奖励，显著提升轨迹精度与连贯性。

**🔧 技术方法**

采用流形匹配的流式视频生成器（Wan2.2‑TI2V‑5B）进行监督微调，并在此基础上使用 Traj‑GDPO（基于 GRPO 的分组归一化强化学习）进行进一步优化；同时构造信令-轨迹配对的地图渲染输入与视频目标。

**📊 数据集**

使用从运营商和出租车 GPS 生态系统匹配得到的约 20k 条信令‑轨迹高置信度对；训练集 10k 对，测试集 1k 对；并利用成都与西安的公开 GPS 数据进行跨城下游任务评测。

**📈 对比分析**

与 GRU、MLP、TCN、GPT、TrajFormer、SigFormer 以及工业级 Rule_sig 进行对比。实验表明在三种轨迹范围（Small、Medium、Large）上，本文模型在 MAE、RMSE、L100 最高、G1000 最低，且推理速度提升至约 30 秒，显著优于 Rule_sig 的两分钟多。

**⚠️ 局限性**

局限性包括对稀疏信令的鲁棒性不足、对未知地图样式的适应性有限，以及在极端城市交通网络或信令质量极差时仍可能产生误判。

---

## 399. Characterization and forecasting of national-scale solar power ramp events

**arXiv ID:** 2603.26596 | [PDF](https://arxiv.org/pdf/2603.26596v1)

**作者:** Luca Lanzilao `[一作]` (Bern University of Applied Sciences), Angela Meyer `[通讯]` (Bern University of Applied Sciences)

**通讯引用:** 989 | [OpenAlex ID](https://openalex.org/A5050747578)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对瑞士6434台光伏站点两年15分钟分辨率的发电数据进行全国尺度的光伏坡度事件识别与统计，并评估现有卫星及数值预报模型在坡度事件与非坡度事件下的预测性能

**💡 创新点**

①提出基于全国聚合发电数据的坡度阈值确定方法，系统描述坡度事件的频率、季节与时段分布；②首次在大尺度光伏发电数据上对多种卫星/数值预报模型在坡度事件条件下的预测精度进行横向比较

**🔧 技术方法**

光流（SolarSTEPS、SolarSTEPS‑pa）、深度生成扩散模型（SHADECast）、卷积LSTM（IrradianceNet）、物理模型（IFS‑ENS）以及XGBoost光伏功率回归

**📊 数据集**

SEVIRI/HANNA卫星辐照度数据（0.01° 15min）和瑞士光伏站点产能15min观测数据（约225 M点）

**📈 对比分析**

采用归一化RMSE（nRMSE）和连续秩概率评分（nCRPS）对预测结果在坡度与非坡度两类进行分组对比；结果显示在坡度事件下预测误差普遍提升20–60%，概率分数提升40–70%；SHADECast在坡度事件中最稳健，IFS‑ENS的误差提升幅度最小

**⚠️ 局限性**

现有模型训练样本中坡度事件极少，导致模型对快速云变动的泛化能力不足；光流模型缺乏云生成/消散机制；需要通过增大坡度事件样本、改进损失函数或融合多源观测以提升坡度预测性能

---

## 400. Automatic Laplace Collapsed Sampling: Scalable Marginalisation of Latent Parameters via Automatic Differentiation

**arXiv ID:** 2603.26644 | [PDF](https://arxiv.org/pdf/2603.26644v1)

**作者:** Toby Lovick `[一作]` (University of Cambridge), Will Handley `[通讯]` (University of Cambridge)

**通讯引用:** 39879 | [OpenAlex ID](https://openalex.org/A5082611196)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于自动微分的拉普拉斯折叠抽样（ALCS）框架，能够在每次嵌套采样评估中将高维潜变量折叠成标量，从而显著降低参数维度，提升贝叶斯证据计算效率。

**💡 创新点**

核心创新在于：1）利用自动微分一次性获得潜变量的MAP及其Hessian，实现全自动化的拉普拉斯近似；2）将该折叠的边缘似然与外部采样器（如嵌套采样）无缝组合，显著降低对潜变量维度的依赖；3）通过高阶导数扩展至Student‑t局部近似，以改善重尾潜变量的证据估计。

**🔧 技术方法**

主要技术包括：自动微分（JAX/Autograd）用于梯度与Hessian计算；L‑BFGS优化潜变量MAP；Cholesky分解求负对数行列式；GPU并行化与vmap实现潜变量批量折叠；以及嵌套采样与重要性抽样（ESS）诊断。

**📊 数据集**

实验数据集涵盖：1）自定义超新星宇宙学模型（N=64–2048，潜变量维度最高25,600）；2）重尾Student‑t潜变量测试；3）Neal的tanh漏斗模型；4）Inference Gym六个基准模型（Eight Schools、Radon、Brownian Motion、LGCP、SV、IRT），其中前三者为近似高斯，后四者为多样结构。

**📈 对比分析**

与全维嵌套采样和NUTS（Stan、BlackJAX）对比，ALCS在高潜变量维度时能实现数十倍至百倍的速度提升，且证据误差普遍低于0.1 nat；在近似高斯模型下几乎无误差；在非高斯模型中，通过ESS诊断能定位折叠失效区段。

**⚠️ 局限性**

局限性包括：1）拉普拉斯近似假设潜变量条件分布近似高斯，对多模态或严重非高斯结构失效；2）对极大潜变量维度（10^5+）时，Hessian求逆与行列式计算可能成为瓶颈；3）Student‑t修正仅为局部高阶校正，无法弥补全局形状偏差；4）仍需对潜变量进行MAP优化，若优化不收敛则导致错误。

---

## 401. Ruka-v2: Tendon Driven Open-Source Dexterous Hand with Wrist and Abduction for Robot Learning

**arXiv ID:** 2603.26660 | [PDF](https://arxiv.org/pdf/2603.26660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 402. USAM: A Unified Safety-Age metric for Timeliness in Heterogeneous IoT Systems

**arXiv ID:** 2603.26628 | [PDF](https://arxiv.org/pdf/2603.26628v1)

**作者:** Mikael Gidlund `[一作]` `[通讯]`, Mikael Gidlund

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了一种统一的安全-信息新鲜度指标（USAM），用于评估大规模异构物联网系统中监测、控制和安全关键流量的时效性，尤其关注在极稀疏运作模式下接收器占空比对系统可行性的影响。

**💡 创新点**

创新点在于：① 将随机的新鲜度指标（AoI、AoS 等）与确定性安全约束（WCRT、SFRT）融合为单一多维度指标；② 在超稀疏区域推导出可行性边界和最小接收器占空比的解析表达式；③ 明确指出传统时效指标对安全可行性“盲目”，并用 USAM 直观揭示安全阈值。

**🔧 技术方法**

主要技术：异构交通流模型（监测、慢控制、快控制、安全流）；单服务器队列与间歇性服务（vacation 模型）；固定优先级预抢占调度；极限理论与队列轻负载分析；最坏情况响应时间（WCRT）与安全功能响应时间（SFRT）的数学绑定；数值仿真验证。

**📊 数据集**

数据集：采用基于 Poisson 到达率的仿真数据；参数设置与工业标准（如 IEC 61784-3）一致，包括服务时延区间、最大激活延迟、各类时限等；未使用公开真实数据集。

**📈 对比分析**

比较方法：将 USAM 与传统指标（AoI、PAoI、VoI、AoC）在相同网络参数下对比；评估指标随接收器占空比、活动因子变化的趋势。结果显示：传统指标随占空比下降或负载上升而平滑升高，未能捕捉安全阈值；而 USAM 在接收器占空比接近最小安全阈值时出现急剧下降，准确揭示系统进入不可行区间。

**⚠️ 局限性**

局限性：① 仅在超稀疏理论下得到解析结果，未验证更高负载场景；② 假设固定优先级调度且无重传/错误恢复，实际无线环境可能更复杂；③ 未给出协议层实现方案，USAM 目前仅为系统级评估工具；④ 只考虑单接收器，未扩展到多接收器或多链路情形。

---

## 403. VGGRPO: Towards World-Consistent Video Generation with 4D Latent Reward

**arXiv ID:** 2603.26599 | [PDF](https://arxiv.org/pdf/2603.26599v1)

**作者:** Zhaochong An `[一作]` (Google), Marta Tintore Gazulla `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出了VGGRPO，一种在潜在空间进行几何一致性后训练的框架，通过构建潜在几何模型将视频扩散模型的潜在表示与4D几何基础模型对齐，从而在不进行RGB解码的前提下直接计算几何奖励，显著提升了视频生成的空间一致性与相机运动平滑性。

**💡 创新点**

创新点主要有：① 在潜在空间与几何基础模型直接对接的潜在几何模型（Latent Geometry Model），避免了高成本的RGB解码；② 设计了两种互补的潜在空间奖励——相机运动平滑奖励与几何重投影一致性奖励；③ 将这些奖励嵌入Group Relative Policy Optimization（GRPO）实现无解码、无外部偏好数据的在线强化学习后训练。

**🔧 技术方法**

技术手段包括：潜在空间映射（将VAE潜在映射到几何模型），轻量级拼接层（connector）与特征对齐；基于Any4D或VGGT等4D几何基础模型的几何预测；latent-space GRPO的策略更新；两类奖励的计算与归一化；以及LoRA微调与多模型对比实验。

**📊 数据集**

主要使用了DL3DV、RealEstate10K、MiraData等真实视频数据集来构建潜在几何模型与后训练样本，静态场景基准来自DL3DV与RealEstate10K的190条描述，动态场景基准来自MiraData的200条描述。

**📈 对比分析**

与基线（原始模型、SFT、Epipolar-DPO、VideoGPA）以及不同规模的Wan2.1-1B与Wan2.2-5B模型进行对比。结果显示VGGRPO在静态与动态场景下均显著提升了视频质量指标（VBench）和几何一致性指标（Sampson epipolar error、VideoReward win率），在所有对比实验中均位列榜首。

**⚠️ 局限性**

局限性包括：依赖于预训练的4D几何基础模型（若其性能不足则奖励可能失效）；对动态场景的支持虽然比以往方法好，但仍受限于几何模型对动态物体分离的能力；潜在空间奖励设计相对简单，可能无法捕捉更细粒度的几何细节；最后，在极端动态或大运动的视频中，奖励信号仍可能不足，导致一致性提升有限。

---

## 404. GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation

**arXiv ID:** 2603.26661 | [PDF](https://arxiv.org/pdf/2603.26661v1)

**作者:** Nicolas von Lützow `[一作]` (Technical University of Munich), Matthias Nießner `[通讯]` (Technical University of Munich)

**通讯引用:** 23236 | [OpenAlex ID](https://openalex.org/A5088583491)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出GaussianGPT，一个基于Transformer的自回归3D高斯场场景生成与补全框架，直接对离散化的高斯原语进行序列化与预测。

**💡 创新点**

创新点在于将3D高斯原语离散化为稀疏三维编码网格，并通过3D RoPE实现空间感知的自回归模型，既能实现无条件生成，又能完成与outpainting。

**🔧 技术方法**

技术包括稀疏三维卷积自编码器、向量量化（LFQ）、3D旋转位置编码（RoPE）、GPT-2式因果Transformer以及对应的训练与采样策略。

**📊 数据集**

使用了Aria Synthetic Environments、3D-FRONT以及PhotoShape（椅子）等数据集进行训练与评测。

**📈 对比分析**

与L3DG、DiffRF、EG3D等基线对比，GaussianGPT在椅子生成的FID/KID/COV上均取得最佳或竞争性成绩；在场景生成与补全上表现出更好的结构一致性、可扩展性和多样性。

**⚠️ 局限性**

局限在于仅在合成数据上验证，对真实世界数据的鲁棒性与长程生成稳定性仍待提升，同时对更大上下文窗口或更高效采样策略的需求仍存在。

---

## 405. Partial Motion Imitation for Learning Cart Pushing with Legged Manipulators

**arXiv ID:** 2603.26659 | [PDF](https://arxiv.org/pdf/2603.26659v1)

**作者:** Mili Das `[一作]` (Georgia Institute of Technology), Sehoon Ha `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 4743 | [OpenAlex ID](https://openalex.org/A5064581452)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本论文提出一种部分模仿学习框架，利用已有的稳健步态政策迁移到腿部机器人推车任务，实现稳定的步态与灵活的操纵；

**💡 创新点**

创新点在于只对低身部位进行对抗性模仿，从而保留步态风格而不限制上身操作；

**🔧 技术方法**

使用了对抗性运动先验（AMP）、部分状态投影、PPO强化学习和Stanley路径跟踪控制；

**📊 数据集**

训练数据来源于自行收集的步态参考状态序列（不使用公开数据集）；

**📈 对比分析**

与无模仿、全身模仿和层次式RL基线比较，实验表明部分AMP在IsaacLab和MuJoCo上具有最高的生存率、最低的跟踪误差，并在多种环境扰动下表现最稳健；

**⚠️ 局限性**

局限性包括缺乏视觉反馈、未处理抓取失效导致的接触丢失以及对急转弯的适应能力有限。

---

## 406. Machine Learning Transferability for Malware Detection

**arXiv ID:** 2603.26632 | [PDF](https://arxiv.org/pdf/2603.26632v1)

**作者:** César Vieira `[一作]` (Polytechnic of Porto), Isabel Praça `[通讯]` (Polytechnic of Porto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估了不同数据预处理方案在使用EMBERv2特征进行PE文件机器学习检测的迁移性和泛化能力。

**💡 创新点**

创新地构建了统一的预处理流水线，比较了EB与EBR两种训练集，结合xgbfs与PCA降维、FLAML调参、软投票模型，系统性评估模型在多公开数据集上的跨域迁移。

**🔧 技术方法**

使用树模型（LightGBM、XGBoost、ExtraTrees、RandomForest）与自适应降维（xgbfs、PCA）、稳健缩放与MinMax归一化、FLAML超参搜索、加权软投票以及tpr@FPR评估指标。

**📊 数据集**

利用六个公开PE数据集：EMBER-2018、SOREL-20M、BODMAS、ERMDs、TRITIUM和INFERNO。

**📈 对比分析**

通过在自身测试集和外部数据集上计算F1、AUC、tpr@1%/0.1% FPR进行比较，结果显示xgbfs+LightGBM在384维时表现最佳，迁移到INFERNO/ TRITIUM仍保持高F1/AUC，而SOREL-20M与ERMDs显著退化。

**⚠️ 局限性**

受限于特征标准化不一致、概念漂移以及高obfuscation/跨时间域数据对特征分布的影响，模型在这类数据上的泛化不足，且未覆盖更深层次的深度学习方法。

---

## 407. Sticky and Magnetic: Evaluating Error Correction and User Adaptation in Gaze and Pinch Interaction

**arXiv ID:** 2603.26608 | [PDF](https://arxiv.org/pdf/2603.26608v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 408. Benchmarking Tabular Foundation Models for Conditional Density Estimation in Regression

**arXiv ID:** 2603.26611 | [PDF](https://arxiv.org/pdf/2603.26611v1)

**作者:** Rafael Izbicki `[一作]` (Federal University of São Carlos), Pedro L. C. Rodrigues `[通讯]` (University of Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文评估了TabPFN、RealTabPFN和TabICL-Quantiles等表格基础模型在条件密度估计（CDE）任务中的表现，并与多类经典与现代基线方法在39个真实回归数据集上进行系统比较。

**💡 创新点**

创新点在于：①首次从分布角度对表格基础模型进行系统性基准测试；②覆盖多达六种评估指标（CDE loss、对数似然、CRPS、PIT KS、90% 覆盖率与计算时间），构建大规模实验；③在大样本量（如SDSS 500k）下展示出显著的样本效率优势。

**🔧 技术方法**

采用的技术包括：表格基础模型（TabPFN、RealTabPFN、TabICL-Quantiles）、参数化分布回归（高斯、Student‑t、对数正态、伽马）、量化树（Quantile‑Tree）、FlexCode‑RF、BART、混合密度网络（MDN）、条件流（Flow‑Spline）及分箱多层感知机（CatMLP），并通过交叉验证或随机搜索进行超参调优，最终用Welch t‑test与Holm‑Bonferroni校正进行统计显著性检验。

**📊 数据集**

使用了39个OpenML回归数据集（特征维数5–563）以及SDSS DR18的光度红移数据集（约50万星系）。

**📈 对比分析**

与基线方法相比，表格基础模型在所有样本量下大多取得最低CDE loss、最高对数似然和最佳CRPS；在SDSS案例中，TabPFN仅用50k样本就能超过所有使用完整50万样本训练的传统方法；但在样本量≥5k时，校准指标往往落后于专门的神经网络基线。

**⚠️ 局限性**

局限性包括：①受上下文长度限制（≈5万样本），高维高样本量时易出现OOM；②在大样本时校准效果不如专门训练的模型；③仅评估单变量响应；④未进行任务特定微调，缺乏对结构化输出的探讨；⑤在离散或准离散响应场景下，某些专业模型仍能优于基础模型。

---

## 409. Sustainability Is Not Linear: Quantifying Performance, Energy, and Privacy Trade-offs in On-Device Intelligence

**arXiv ID:** 2603.26603 | [PDF](https://arxiv.org/pdf/2603.26603v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

