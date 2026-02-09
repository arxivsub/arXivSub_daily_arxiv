# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-09 | 今日论文总数: 469

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Driving with DINO: Vision Foundation Features as a Unified Bridge for Sim-to-Real Generation in Autonomous Driving

**arXiv ID:** 2602.06159 | [PDF](https://arxiv.org/pdf/2602.06159v1)

**作者:** Xuyang Chen `[一作]` (Technical University of Munich), Liqiu Meng `[通讯]` (Technical University of Munich)

**通讯引用:** 4666 | [OpenAlex ID](https://openalex.org/A5057986367)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种利用 Vision Foundation Model（DINOv3）特征的可控视频扩散框架，将模拟驾驶视频转换为具有高度真实感且结构一致的现实视频。

**💡 创新点**

核心创新包括：① 将 VFM 特征作为统一桥梁，兼顾低层结构和高层语义；② 通过 PCA 主子空间投影和随机通道尾部丢弃实现“纹理剥离”与结构保留；③ 设计可学习的空间对齐模块提升分辨率匹配；④ 引入因果时序聚合器解决时间冗余与运动模糊。

**🔧 技术方法**

使用的技术主要有 DINOv3 视觉模型、Diffusion Transformer（DiT）与 ControlNet 结构、主成分分析（PCA）与随机通道裁剪、卷积对齐模块、因果卷积时序聚合、视频扩散与语义评估模型（如 Cityscapes 语义分割）。

**📊 数据集**

训练和评估基于 nuPlan 数据集与 CARLA 仿真生成的视频；实验中将生成结果与真实摄像机捕获的视频进行对比。

**📈 对比分析**

与 FRESCO、TC-Light、Cosmos‑Transfer 等基线在 sFID、sKID、CLIP‑Real、Motion‑S、WarpSSIM 和 mIoU 等指标上对比，方法在视觉真实性、运动一致性和语义保真度上均超越或与最佳基线持平，且在纹理逼真度和结构一致性上表现尤为突出。

**⚠️ 局限性**

局限性包括：受限的计算资源导致未充分挖掘模型潜能；对大规模高分辨率数据集的可扩展性仍需验证；模型对 PCA 维度和上采样比例等超参数敏感，需进一步自动化调优。

---

## 2. Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings

**arXiv ID:** 2602.06218 | [PDF](https://arxiv.org/pdf/2602.06218v1)

**作者:** Grégoire Dhimoïla `[一作]` (Brown University), Agustin Picard `[通讯]` (ENS Paris Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Iso‑Energy假设并设计对齐稀疏自编码器，利用跨模态能量一致性从VLM嵌入中恢复共享概念；

**💡 创新点**

在无监督字典学习中首次引入跨模态能量一致性正则，使稀疏原子分离出跨模态对齐与模态偏置，支持可解释的介入与语义算子；

**🔧 技术方法**

Iso‑Energy对齐稀疏自编码器（Aligned SAE），匹配追踪算法、能量一致性正则、稀疏编码；

**📊 数据集**

使用多种双编码器VLM（CLIP、OpenCLIP、SigLIP等）在随机采样的100万条LAION嵌入上训练；

**📈 对比分析**

与传统SAE比较，重建误差相近但跨模态指标（p_acc、ρ、FDA、δ_r）显著提升，证明对齐正则提升跨模态结构而不损失性能；

**⚠️ 局限性**

对β权重敏感、仅在双编码器架构下验证、对原始嵌入而非自编码器输出的性能评估有限。

---

## 3. Communication Enhances LLMs' Stability in Strategic Thinking

**arXiv ID:** 2602.06081 | [PDF](https://arxiv.org/pdf/2602.06081v1)

**作者:** Nunzio Lore `[一作]` (Northeastern University), Babak Heydari `[通讯]` (Northeastern University)

**通讯引用:** 7498 | [OpenAlex ID](https://openalex.org/A5073502448)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在7B-9B规模的LLM上，使用短的非约束性预演信息（cheap talk）对其在迭代囚徒困境游戏中的合作轨迹进行实验，评估其对战略稳定性的影响。

**💡 创新点**

证明了即使是零成本、非绑定的消息交流也能显著降低LLM在多轮博弈中的行为波动；并揭示了模型、上下文和网络拓扑对稳定性影响的异质性。

**🔧 技术方法**

采用非参数低斯平滑（LOWESS）对合作率时间序列进行平滑，使用自助采样（bootstrap）计算RMSE差异；在多模型、多上下文、多网络拓扑下进行对比分析；温度调节（0.8和0）及一词/全句信息限制实验。

**📊 数据集**

四个公开权重LLM（Qwen 2.5 7B、Falcon 3 7B、Granite 3.3 8B、Gemma 2 9B）与六种情景（neutral、biz、environment、social、team、IR）组成实验数据集；此外在网络实验中使用ER、PL、CoPe三种拓扑。

**📈 对比分析**

通过自助采样的95%置信区间检验RMSE差异，采用二项检验检验显著性和方向一致性；结果显示大多数模型-情境组合在有消息条件下RMSE显著下降，提升了战略稳定性；但某些模型在特定情境下或一词限制下出现逆向效应。

**⚠️ 局限性**

仅覆盖7B-9B规模，未探讨更大模型；实验仅限于迭代囚徒困境与cheap talk，缺乏更复杂或强制性博弈的验证；受限于提示设计和解码参数，结果可能与其他设置不完全可迁移。

---

## 4. Mapping Gemma3 onto an Edge Dataflow Architecture

**arXiv ID:** 2602.06063 | [PDF](https://arxiv.org/pdf/2602.06063v1)

**作者:** Shouyu Du `[一作]`, Zhenyu Xu `[通讯]` (Clemson University)

**通讯引用:** 1306 | [OpenAlex ID](https://openalex.org/A5101545996)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

首次将Gemma3文本-视觉模型完整部署到AMD Ryzen AI边缘NPU上，实现端到端推理；

**💡 创新点**

提出多项硬件感知优化：高效去量化引擎、基于张量化的MM优化、Chunked流水线注意力FlowQKV/FlowKV、融合去量化与投影的FusedDQP，以及紧凑的4位量化格式Q4NX；

**🔧 技术方法**

利用NPU的2D计算单元、DMA广播、AIE‑MLIR编译框架，结合量化、块化MM、流水线调度与内存层级管理实现性能提升；

**📊 数据集**

使用Gemma3 1B与4B模型（含Vision Tower），在不同序列长度（1K~128K token）上进行基准测试；

**📈 对比分析**

与同一硬件平台下的iGPU与CPU基准进行对比：Prefill 5.2×/33.5×加速，Decode 4.8×/2.2×加速；功耗效率提升至67.2×/222.9×；

**⚠️ 局限性**

受限于NPU的内存带宽、Tile数量与对大模型支持不足；与数据中心GPU水平仍有差距，且需要针对不同NPU平台进一步验证与适配；

---

## 5. Experimental Analysis of Server-Side Caching for Web Performance

**arXiv ID:** 2602.06074 | [PDF](https://arxiv.org/pdf/2602.06074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 6. Accelerating Vision Transformers on Brain Processing Unit

**arXiv ID:** 2602.06300 | [PDF](https://arxiv.org/pdf/2602.06300v1)

**作者:** Jinchi Tang `[一作]` (Suzhou Institute for Advanced Research), Yan Guo `[通讯]` (University of Science and Technology of China)

**通讯引用:** 7053 | [OpenAlex ID](https://openalex.org/A5034513473)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将 Vision Transformer 的线性层和 LayerNorm 改写为卷积操作，使 DeiT 可在 BPU 上进行 INT8 量化加速，从而实现端侧部署。

**💡 创新点**

首次提出可直接迁移预训练权重、无需再训练或微调的 BPU‑友好 Vision Transformer 结构，并通过卷积重构实现对 BPU 的充分利用。

**🔧 技术方法**

卷积替代线性/LayerNorm、Post‑Training 量化、INT8 量化工具链、Horizon BPU 加速引擎。

**📊 数据集**

ImageNet 与花卉分类（Flower）数据集。

**📈 对比分析**

与原始 DeiT 在 ImageNet 进行 Top‑1/Top‑5 准确率对比，量化后准确率下降 ≤2%，在 BPU 上推理速度提升 1.3×–3.8×；在 Flower 数据集上也保持了高精度。

**⚠️ 局限性**

量化对 DeiT‑Small 在 ImageNet 上导致显著 6.9% 的 Top‑1 下降，且小模型的鲁棒性相对差；仅支持卷积形式，无法利用线性运算的高效实现。

---

## 7. Reclaiming First Principles: A Differentiable Framework for Conceptual Hydrologic Models

**arXiv ID:** 2602.06429 | [PDF](https://arxiv.org/pdf/2602.06429v1)

**作者:** Jasper A. Vrugt `[一作]`, Ethan Bollman `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种在概念降雨-径流模型中使用前向敏感度分析（Forward Sensitivity Analysis, FSA）得到完整解析梯度的方法，实现模型状态和参数敏感度同时一趟正向积分。

**💡 创新点**

创新点在于：① 通过将ODE系统与其敏感度方程耦合，直接获得不依赖数值差分或自动微分的解析Jacobian；② 针对多种常用损失（平方误差、NSE、KGE、Huber等）推导了对应的误差传递向量；③ 引入光滑无约束参数映射，保持物理可解释性并简化梯度计算。

**🔧 技术方法**

技术手段包括：连续时间敏感度分析、解析Jacobian推导、参数重映射（单位立方→无约束空间）、数值验证（有限差分、JAX/PyTorch等自动微分）以及基于解析梯度的梯度下降、Levenberg‑Marquardt 等优化算法。

**📊 数据集**

数据集：美国弗吉尼亚州 Wye 与 Severn 河（小尺度），北卡罗来纳州 Broad River 与 Leaf River（中尺度）以及英国 Plynlimon 实验流域的日、时尺度降水与径流记录，共计 4 个流域、日/时两种分辨率。

**📈 对比分析**

与数值差分和自动微分比较：解析梯度与数值梯度误差均在 10⁻⁶–10⁻⁴ 之间；在 CPU 开销上，解析梯度比数值差分快 50–500 倍，比 JAX/PyTorch 的前向/反向自动微分快 100–4000 倍；梯度精度更高，特别是对 NSE、KGE 等效率指标对应的梯度几乎无噪声。

**⚠️ 局限性**

局限性：① 前向敏感度规模随参数维度线性增长，若参数维度极高则可能出现计算瓶颈；② 对于极其复杂或带有大量条件分支的模型，解析导数推导工作量大；③ 解析梯度依赖于模型实现的连续可微性，含有非光滑阈值/离散事件的模型需额外处理；④ 在高度非线性、存在多模态响应表面时，仅靠解析梯度无法保证全局最优，需要配合全局搜索策略。

---

## 8. ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks

**arXiv ID:** 2602.06240 | [PDF](https://arxiv.org/pdf/2602.06240v1)

**作者:** Yu Zhang `[一作]` (Aalborg University), Cuneyt Gurcan Akcora `[通讯]` (University of Central Florida)

**通讯引用:** 1194 | [OpenAlex ID](https://openalex.org/A5045418504)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种统一框架，将图神经网络的对抗攻击与反事实解释生成相结合，在有限的扰动预算下同时使用边的添加和删除，生成可解释且真实的反事实示例。

**💡 创新点**

创新点在于首次将对抗攻击中有效的边添加策略与传统仅使用边删除的反事实方法相结合，并在一个统一的损失函数中同时优化影响力、稀疏性与可解释性，从而显著提升解释的有效性和可行性。

**🔧 技术方法**

使用技术包括梯度驱动的连续签名掩码与直通估计器（STE）优化、基于攻击子图的候选边选择（利用GOttack）、联合损失（预测、稀疏、合理性）以及后处理的贪婪最小化剪枝。

**📊 数据集**

实验数据集涵盖合成图（BA‑SHAPES、TREE‑CYCLES、Loan‑Decision）与真实图（Cora、Chameleon、Ogbn‑Arxiv），并在GCN、GAT、Graph Transformer三种GNN架构上评估。

**📈 对比分析**

与十余种基线（CF‑GNNExplainer、CF^2、NSEG、INDUCE、C2Explainer、GNNExplainer、PGExplainer、Nettack、GOttack等）相比，该方法在误分类率、可信度、解释规模、合理性和计算时间等指标上均取得显著优势，平均排名最优且在30个指标‑数据集组合中获胜20次。

**⚠️ 局限性**

局限性包括对加边成本的敏感性（需手动调节权重）、仍仅考虑结构扰动而未加入特征扰动、适用于静态图的限制，以及在极大图上可能存在的可扩展性挑战。

---

## 9. Addressing the Waypoint-Action Gap in End-to-End Autonomous Driving via Vehicle Motion Models

**arXiv ID:** 2602.06214 | [PDF](https://arxiv.org/pdf/2602.06214v1)

**作者:** Jorge Daniel Rodríguez-Vidal `[一作]` (Computer Vision Center), Antonio M. López Peña `[通讯]` (Universitat Autònoma de Barcelona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种可微分的车辆运动模型框架，将动作策略的低层控制输出升维为未来轨迹，从而使基于动作的端到端驾驶模型能够在仅支持轨迹输出的基准上训练和评估。

**💡 创新点**

创新点在于首次将可微分、确定性的车辆动力学（Kinematic Bicycle Model 与 Continuous Curvature Path Planner）作为“升维算子”嵌入训练管道，实现动作与轨迹表示之间的无缝桥接，且不改动现有评测协议。

**🔧 技术方法**

主要技术包括：可微分控制激活、数值积分（Euler 与 RK4）、轨迹投影以及基于轨迹的 L1 损失；并通过梯度反向传播将误差回传至动作策略网络。

**📊 数据集**

使用了公开的 NAVSIM v1、NAVSIM v2、Bench2Drive 以及 CARLA 仿真数据集，对比多种基准模型。

**📈 对比分析**

实验显示，配合 KBM 或 CCPP 的动作策略在 NAVSIM（尤其是 navhard）和 Bench2Drive 上均能超过或逼近传统轨迹预测基准，获得了最高的驾驶分数和路径完成率；同时在闭环训练中显著降低了梯度方差。

**⚠️ 局限性**

主要局限在于仅考虑二维平面运动，忽略坡度、滚动、悬挂等 3D 物理效应；并且 CCPP 的初始化假设为零初始曲率，可能在存在强曲率的起始动作时表现欠佳。

---

## 10. An Interpretable Vision Transformer as a Fingerprint-Based Diagnostic Aid for Kabuki and Wiedemann-Steiner Syndromes

**arXiv ID:** 2602.06282 | [PDF](https://arxiv.org/pdf/2602.06282v1)

**作者:** Marilyn Lionts `[一作]` (Vanderbilt University), Lotta M. Ellingsen `[通讯]` (University of Iceland)

**通讯引用:** 1393 | [OpenAlex ID](https://openalex.org/A5089094126)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用Vision Transformer模型对来自Kabuki综合征、Wiedemann‑Steiner综合征及对照组的指纹图像进行分类，探索指纹在罕见遗传疾病早期诊断中的可行性

**💡 创新点**

首次将深度学习与指纹纹理结合，并通过注意力热图提供可解释的诊断依据，揭示了两种疾病在指纹纹理上存在细微差异而不仅仅是胎儿指尖垫的持续存在

**🔧 技术方法**

采用Vision Transformer（ViT）架构，对224×224像素的指纹图像进行16×16像素的分块嵌入，训练5个模型并集成；使用自注意力权重生成关注热图；优化器为Adam，学习率3×10^-4

**📊 数据集**

数据集包含75名Kabuki患者、38名WSS患者和120名健康对照，总共2109张高质量指纹图像（经过NIST NFIQ2过滤后），每位参与者最多提供10张指纹图像

**📈 对比分析**

通过三种二分类任务（对照vs.KS、对照vs.WSS、KSvs.WSS）进行评估，AUC分别为0.80、0.73和0.85，准确率0.72/0.80/0.88，F1得分0.71/0.72/0.83；与传统基因检测相比，提供了无创、低成本的快速筛查手段

**⚠️ 局限性**

样本量有限且主要依赖光学扫描器，缺乏手机摄像头验证，指纹采集仅覆盖单个手指，可能导致泛化性不足；未来需扩大样本规模、优化采集方式并验证在多中心、不同设备环境下的表现

---

## 11. Trustworthy AI Software Engineers

**arXiv ID:** 2602.06310 | [PDF](https://arxiv.org/pdf/2602.06310v1)

**作者:** Aldeida Aleti `[一作]` (Monash University), Simin Chen `[通讯]` (Columbia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过文献综述与理论分析，提出了AI软件工程师的定义及其可信度的多维度评估框架。

**💡 创新点**

创新点在于将可信度从单一性能指标扩展到技术质量、透明性、认知谦逊和社会伦理等维度，并将AI软件工程师定位为人机协作团队成员而非仅仅是编码工具。

**🔧 技术方法**

采用了基于已有SE标准（IEEE、ACM、SWEBoK）和可信AI研究的理论框架与概念模型构建方法。

**📊 数据集**

未使用任何具体数据集，主要以理论构建与案例讨论为主。

**📈 对比分析**

未进行实验对比，本文以概念性讨论和现有文献为依据，未给出性能数值。

**⚠️ 局限性**

局限在于缺乏量化评估和实证验证，可信度指标难以测量，模型黑箱性和多维度的权重分配仍待进一步研究。

---

## 12. ForeHOI: Feed-forward 3D Object Reconstruction from Daily Hand-Object Interaction Videos

**arXiv ID:** 2602.06226 | [PDF](https://arxiv.org/pdf/2602.06226v1)

**作者:** Yuantao Chen `[一作]` (Chinese University of Hong Kong Shenzhen), Xiaoguang Han `[通讯]` (FNii Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种端到端的 feed‑forward 方法 ForeHOI，能够在一分钟内从单目手物交互视频中直接重建完整的 3D 物体几何，完全不需要任何预处理步骤。

**💡 创新点**

核心创新在于将 2D 物体遮挡填补与 3D 形状完成共同进行的双向交叉注意机制，利用手部特征提升对遮挡区域的理解，并首次提供大规模合成的手物交互视频数据集。

**🔧 技术方法**

技术路线包括基于 DINOv2 的图像特征提取、手势估计、Diffusion Transformer（DiT）以及双向交叉注意、条件流匹配（CFM）训练、SLat 结构化潜流微调、LoRA 细调和基于 PnP 的相机姿态估计。

**📊 数据集**

训练使用自构建的 400K 条合成手物交互视频数据集（基于 GraspXL+Objaverse），评估数据集包括 HO3D 与 HOT3D 等真实手物交互视频。

**📈 对比分析**

在 HO3D 与 HOT3D 上与 EasyHOI、HOLD、HORT、MagicHOI 等方法比较，ForeHOI 在 Chamfer 距离、F‑score、相机姿态误差均显著优于对手，并将推理时间从数分钟压缩到约 1 分钟，速度提升超过 100 倍。

**⚠️ 局限性**

主要局限是受限于扩散生成模型的精度，仍难以实现完全精确的几何与纹理细节；在极端遮挡、多物体交互或高动态场景下的鲁棒性还有待提升。

---

## 13. On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation

**arXiv ID:** 2602.06423 | [PDF](https://arxiv.org/pdf/2602.06423v1)

**作者:** Wenbo Shang `[一作]` (Hong Kong Baptist University), Xin Huang `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 16843 | [OpenAlex ID](https://openalex.org/A5031729932)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于General Theory of Verbal Humor（GTVH）的多角色LLM框架HOMER，用于多模态幽默标题生成。

**💡 创新点**

创新点在于显式建模脚本对立、层级想象树和检索增强想象机制，提升幽默逻辑解释性和创造性。

**🔧 技术方法**

技术包括LLM多角色协作、脚本提取、层级想象树构建、幽默相关检索、幽默相关评分、Pass@K评估等。

**📊 数据集**

使用的数据集为New Yorker Cartoon的Human in AI与Electronic Sheep两大公开数据集，以及自建的12个笑话数据库。

**📈 对比分析**

与七个SOTA模型及三种推理策略比较，Pass@1提升8.62%、Pass@3提升6.48%、Pass@5提升5.91%；人类评测平均分超过3.0，表现优于对手。

**⚠️ 局限性**

局限在于仍依赖LLM内在幽默感受，检索库覆盖可能不足，跨文化或非英语幽默适用性待进一步验证。

---

## 14. Toward generative machine learning for boosting ensembles of climate simulations

**arXiv ID:** 2602.06287 | [PDF](https://arxiv.org/pdf/2602.06287v1)

**作者:** Parsa Gooya `[一作]` (Canadian Centre for Climate Modeling and Analysis), Johannes Exenberger `[通讯]` (Vienna University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用条件变分自编码器（cVAE）在仅有单个CanESM5成员的训练样本上生成大规模气候模拟样本，主要针对近地表空气温度（TAS）进行后处理增强。

**💡 创新点**

创新点在于：①仅用单一模拟成员即可训练出能够产生与原始大规模模拟相匹配的多样化样本；②通过在解码器中加入估计的噪声，显著提升多尺度变异和极值的再现；③提供了一套简洁、高效且可解释的框架，可直接用于后处理与预测场景。

**🔧 技术方法**

采用的技术包括：条件变分自编码器（cVAE）框架、β‑annealing 正则化、低维条件嵌入、解码器噪声估计（基于训练残差的协方差）以及标准化与重标记等预处理步骤。

**📊 数据集**

使用的数据集为 CMIP6 CanESM5 模型的历史与 SSP2-4.5 方案下的月度近地表气温（TAS）字段，仅训练单个成员（1951‑2025 年共 840 个月），其余 24 成员用于验证与对比。

**📈 对比分析**

比较方法：将生成的 cVAE+噪声 ensemble 与剩余 24 成员的原始 ensemble 进行统计对比，包括 QQ 图、PDF、方差分布、极值分位数、ENSO 指数分布及遥感连通性。结果表明，生成的 ensemble 在极值、方差、ENSO 相关和全球遥感模式上与原始 ensemble 近似，误差相对较小，验证了模型的有效性。

**⚠️ 局限性**

局限性：①对非高斯分布与极端事件的再现仍有偏差，尤其在极端温度上可能过度或欠估；②输出平滑导致谱偏差，尤其在细尺度上；③依赖单一成员的条件嵌入，导致对长期趋势与慢变异的把握不足；④未显式建模时间相关性，无法捕捉时序依赖；⑤噪声估计采用简单协方差近似，可能不足以完全恢复多尺度噪声结构。

---

## 15. To 2:4 Sparsity and Beyond: Neuron-level Activation Function to Accelerate LLM Pre-Training

**arXiv ID:** 2602.06183 | [PDF](https://arxiv.org/pdf/2602.06183v1)

**作者:** Meghana Madhyastha `[一作]` (Meta), Carole-Jean Wu `[通讯]` (Meta)

**通讯引用:** 5528 | [OpenAlex ID](https://openalex.org/A5028220093)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Transformer的FFN块中提出一种训练期间同时对权重与激活进行稀疏化的方案，实现全训练过程加速；

**💡 创新点**

创新点在于将2:4 soft‑thresholding权重稀疏与V:N:M（Venom）激活稀疏结合，并设计了 neuron‑level 路由实现激活矩阵符合Venom格式，同时通过交替稀疏与稠密训练步骤来恢复精度；

**🔧 技术方法**

使用的技术包括2:4 soft‑thresholding权重稀疏、Venom (V:N:M) 激活稀疏、neuron‑level MoE 路由、FP8 量化、NVIDIA Sparse Tensor Core 加速、微基准与 Roofline 分析；

**📊 数据集**

在 DCLM 数据集上训练 Llama‑3 1B 与 7B 模型，并在 Eval‑Harness 基准集上评估；

**📈 对比分析**

与密集 Llama‑3 1B/7B 在 loss 曲线和 Eval‑Harness 评测指标对比，1B 模型实现约1.37× 的端到端加速，7B 同样取得 1.37× 加速；微基准显示 2:4 稀疏实现 1.5× 速度提升，Venom 稀疏实现 6–8× 速度提升；

**⚠️ 局限性**

局限性包括需要硬件支持（Sparse Tensor Core）、稀疏化与动态路由的计算与内存开销、需要额外的稠密微调恢复精度、仅在 1B/7B 规模验证、激活稀疏依赖输入需 warm‑up，且对更大规模或推理阶段的效果尚未全面验证。

---

## 16. Action Hallucination in Generative Visual-Language-Action Models

**arXiv ID:** 2602.06339 | [PDF](https://arxiv.org/pdf/2602.06339v1)

**作者:** Harold Soh `[一作]` (National University of Singapore), Eugene Lim `[通讯]` (National University of Singapore)

**通讯引用:** 96 | [OpenAlex ID](https://openalex.org/A5003251880)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过严谨的数学分析，阐述了生成式视觉-语言-动作模型在机器人控制中出现的动作错觉（action hallucination）及其对安全和可靠性的影响，并将其分为拓扑、精度和时间跨度三大障碍。

**💡 创新点**

创新点在于首次将深度生成模型的拓扑连续性、平滑性与机器人可行空间的非凸、薄层结构相结合，提出了三类不可避免的“障碍”并给出了相应的定量下界，揭示了生成模型在多模态、多接触以及长时序任务中的根本性限制。

**🔧 技术方法**

主要使用的技术包括隐变量生成器（如扩散、流匹配）以及高斯等距测度、维度理论和测度论中的 isoperimetric inequality 等数学工具，对连续解码器的 Lipschitz 性质进行推导。

**📊 数据集**

本文未针对具体数据集开展实验，研究以理论证明为主，并在附录中给出了一些小规模实验验证，但整体聚焦在理论框架构建而非数据驱动。

**📈 对比分析**

由于缺乏统一的实验基准，本文没有与现有方法在性能上进行量化比较，而是通过定量下界与经验趋势的对比，说明生成模型在实际应用中可能出现的失真与失败。

**⚠️ 局限性**

主要局限包括：假设系统动力学确定且无感知、部分可观测或噪声；未考虑记忆、随机性及更复杂的验证/规划模型；因此理论结果对真实机器人系统的直接可转化仍需进一步验证。

---

## 17. Urban Spatio-Temporal Foundation Models for Climate-Resilient Housing: Scaling Diffusion Transformers for Disaster Risk Prediction

**arXiv ID:** 2602.06129 | [PDF](https://arxiv.org/pdf/2602.06129v1)

**作者:** Olaf Yunus Laitinen Imanov `[一作]` (Technical University of Denmark), Taner Yilmaz `[通讯]` (Afyon Kocatepe University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 Skjold‑DiT，一种基于扩散 Transformer 的城市建筑级气候风险预测框架，能够同时输出洪水、热波等多灾害指标并结合交通可达性，为智能车辆调度与紧急响应提供量化决策依据。

**💡 创新点**

创新点包括：① Norrland‑Fusion 跨模态注意力融合灾害影像、结构属性、交通网络与社会人口等多源数据；② Fjell‑Prompt 通过分层 prompt 实现零样本跨城迁移；③ Valkyrie‑Forecast 利用条件扩散生成可解释的因果情景；④ 构建了包含 847,392 建筑、洪水/热波标签与交通可达性特征的 BCUR 数据集。

**🔧 技术方法**

技术方法涵盖：扩散 Transformer（DiT）与 DDIM 采样、跨模态注意力融合、图神经网络（GraphSAINT）提取交通网络特征、时间序列 Transformer（Temporal Fusion Transformer）、RoBERTa‑based Prompt 编码、基于图的可达性与路线冗余计算。

**📊 数据集**

使用的主要数据集是 Baltic‑Caspian Urban Resilience (BCUR)，覆盖六座城市（Copenhagen、Stockholm、Oslo、Riga、Tallinn、Baku），包含洪水深度、热压力、建筑结构、交通可达性、人口与社会经济特征共 847,392 条建筑观测。

**📈 对比分析**

与物理模型 HAND‑DEM、随机森林、CNN（ResNet‑50）、GraphSAGE、以及迁移学习后的 UrbanDiT 进行对比。Skjold‑DiT 在 10 年洪水分类中达 94.7% 的准确率，零样本迁移到 Baku 的准确率 87.2%，10 年预测准确率 86%，不确定性校准 ECE 仅 0.037，整体提升 6–13% 以上。

**⚠️ 局限性**

主要局限包括：① 采用 100 m 聚类而非建筑级细粒度，可能忽略局部差异；② 交通网络被视为静态，未考虑道路关闭或拥堵动态；③ 缺乏对人类行为（疏散、迁移）与多灾害级联效应的建模；④ 依赖历史训练数据，实时监测与在线更新功能有限；⑤ 训练与推理均需较高计算资源，限制快速部署。

---

## 18. Multi-Way Representation Alignment

**arXiv ID:** 2602.06205 | [PDF](https://arxiv.org/pdf/2602.06205v1)

**作者:** Akshit Achara `[一作]` (King's College London), Donato Crisostomi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 102 | [OpenAlex ID](https://openalex.org/A5015184440)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了多模型表示对齐框架，将所有模型映射到共享宇宙空间，实现线性扩展和循环一致性。

**💡 创新点**

在保持几何保真度的 GPA 基础上引入 Geometry‑Corrected Procrustes Alignment (GCPA)，通过统一的共识纠正显著提升检索性能，兼顾几何和语义一致性。

**🔧 技术方法**

核心技术包括 Generalized Procrustes Analysis (GPA)、Canonical Correlation Analysis (CCA)、多视图 GCA (GCCA) 与后置方向纠正 MLP；使用线性旋转与投影组合实现对齐。

**📊 数据集**

在多语言 TED‑Multi、跨摄像头 Market‑1501、图文音频 Flickr8k 以及 CIFAR‑100 等多模态、跨架构数据集上进行评估。

**📈 对比分析**

与无对齐、对称对齐 (PW)、GPA、GCCA 等基线对比，GCPA 在多语言检索、跨摄像头 mAP、跨模态 Rank‑1 等指标上均取得最高或最优性能，显示出显著提升。

**⚠️ 局限性**

主要局限在于仅使用线性/近线性对齐，对高度异质的模型集合可能不足；对齐过程仍需依赖准确的对应关系，噪声会影响性能。

---

## 19. Provably avoiding over-optimization in Direct Preference Optimization without knowing the data distribution

**arXiv ID:** 2602.06239 | [PDF](https://arxiv.org/pdf/2602.06239v1)

**作者:** Adam Barla `[一作]` (Ecole Polytechnique Fédérale de Lausanne), Volkan Cevher `[通讯]` (Ecole Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于惰性集成的单步偏好优化算法（PEPO），用于在不需要数据生成分布或显式奖励模型的情况下避免偏好学习中的过度优化问题。

**💡 创新点**

创新点在于通过让多个在互斥子集上训练的 DPO 模型在生成时采用“最坏情况”聚合（即只保留所有模型都一致支持的回答），实现了对模型不确定性的惰性惰性估计，且理论上仅依赖单一策略的浓缩系数即可得到样本复杂度上界，彻底摆脱了传统 DPO 的所有策略浓缩依赖。

**🔧 技术方法**

核心技术包括：① 在分割的偏好数据子集上训练多头 LoRA 适配器实现的轻量 DPO 模型；② 采用允许平局的 Bradley‑Terry 模型和右移 Sigmoid 形成的惰性 DPO 损失；③ 通过最小化所有子模型的对数概率（加上惰性惰性惩罚）得到最终策略，并使用拒绝采样或 token‑级近似实现高效采样。

**📊 数据集**

实验数据集主要为 UltraFeedback（由多种 LLM 包括 GPT‑4 生成的偏好对），并在 Zephyr‑7B、Llama‑3.1‑8B、Mistral‑7B、Yi‑34B 等模型上使用 AlpacaEval 进行评估。

**📈 对比分析**

与标准 DPO、χ²PO 和 SFT+DPO 等基线相比，PEPO 在所有测试模型上均获得更高的 win‑rate，且在训练过程中能保持更稳定的性能，证明了其对过度优化的抑制效果；拒绝采样实现更优性能但速度略慢，token‑级实现速度更快。

**⚠️ 局限性**

局限性包括：① 理论证明仅适用于离散表格设置，无法直接推广到全参数化模型；② 需要手动设定集成规模 L、惰性惩罚 B 等超参数；③ 拒绝采样在低一致性场景下可能导致生成延迟；④ 仍依赖数据集的离散划分和对每个子集的充分样本覆盖。

---

## 20. Scaling Mobile Chaos Testing with AI-Driven Test Execution

**arXiv ID:** 2602.06223 | [PDF](https://arxiv.org/pdf/2602.06223v1)

**作者:** Juan Marcano `[一作]` (Uber Technologies, Inc.), Mayank Bansal `[通讯]` (Uber Technologies, Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对Uber的移动应用进行大规模、持续的“移动混沌测试”，结合AI驱动的UI自动化与服务级别的故障注入。

**💡 创新点**

创新点在于将LLM驱动的自适应移动测试平台DragonCrawl与uHavoc故障注入平台无缝集成，实现不需要为每种流/城市/故障场景编写脚本的组合爆炸解决方案，并且自动根因分析将手机侧异常快速定位到具体后端RPC。

**🔧 技术方法**

技术包括GPT‑4o/LLM进行动作决策、视觉问答断言、分布式追踪（Jaeger）和网络日志；uHavoc利用HTTP头进行故障注入；系统采用安全隔离的测试租户。

**📊 数据集**

使用的数据集为Uber内部47条核心业务流的实时生产流量，约180,000次测试执行，包含数十万个RPC、网络日志和屏幕截图。

**📈 对比分析**

与传统手工或单一端到端测试相比，系统实现99.27%通过率、P95延迟≈220s，根因分析精度@5为88%；在对照实验中，LLM在T5–T2失败场景下precision@1保持≈0.95-0.97，pass率≥99%。

**⚠️ 局限性**

局限性包括：只能在测试租户内注入服务级别错误，无法覆盖网络/基础设施级别或第三方服务故障；依赖准确的服务层级标记、完整分布式追踪与组织成熟度；对机场等细粒度流变体覆盖有限。

---

## 21. How (Not) to Hybridize Neural and Mechanistic Models for Epidemiological Forecasting

**arXiv ID:** 2602.06323 | [PDF](https://arxiv.org/pdf/2602.06323v1)

**作者:** Yiqi Su `[一作]` (Virginia Tech), Naren Ramakrishnan `[通讯]` (Virginia Tech)

**通讯引用:** 10851 | [OpenAlex ID](https://openalex.org/A5035052603)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

该工作提出了一种结合多尺度信号分解、受控神经ODE与SIRS机理模型的混合方法（EpiNODE），实现了在仅观测感染者数的条件下进行长周期疫情预测和参数反演。

**💡 创新点**

创新点在于：①利用VMD分解感染曲线为趋势、季节性与残差三种控制信号，显式表达多尺度非平稳性；②将每种控制信号分别驱动独立的神经ODE，随后融合并解码为时间变参数，再与SIRS方程耦合，既保持机理可解释性，又提升长期稳定性。

**🔧 技术方法**

使用的技术包括：变分模态分解（VMD）、时间延迟嵌入、三维受控神经ODE、参数解码网络、SIRS动力学约束、以及在训练中加入加权MSE损失。

**📊 数据集**

实验数据涵盖：①合成SIRS（固定与周期性变化）、SIR、SEIRS；②真实季节性流感（CDC ILI）10个HHS区域；③非季节性COVID类症状（Delphi COVID‑like Illness）。

**📈 对比分析**

与ARIMA、RNN、TimeKAN、TimeMixer++、EINN、Neural ODE、Latent ODE、KAN‑ODE、EARTH等基线进行统一评估，使用RMSE、MAE及峰值时空误差等指标；EpiNODE在所有数据集上平均降低15–35% RMSE，峰值时延误差缩短1–3周，峰值幅度偏差减少约30%，显示显著优于现有方法。

**⚠️ 局限性**

局限性包括：仅处理单区域确定性动力学；未建模不确定性、空间耦合与干预措施；缺乏对政策、检测率等外部因素的显式建模。

---

## 22. A Dialogue-Based Human-Robot Interaction Protocol for Wheelchair and Robotic Arm Integrated Control

**arXiv ID:** 2602.06243 | [PDF](https://arxiv.org/pdf/2602.06243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 23. MoSE: Mixture of Slimmable Experts for Efficient and Adaptive Language Models

**arXiv ID:** 2602.06154 | [PDF](https://arxiv.org/pdf/2602.06154v1)

**作者:** Nurbek Tastan `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Samuel Horvath `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种在Mixture-of-Experts（MoE）模型中使用可裁剪专家（slimmable experts）的架构MoSE，使单一模型在推理时能够根据需要动态调整专家的宽度，从而在计算量与模型精度之间实现更平滑的权衡；

**💡 创新点**

创新点在于把传统MoE的“专家选取”与“专家容量”两重条件计算分离：不仅通过路由器选择激活哪些专家，还通过宽度切换决定每个专家内部执行多少比例的参数；

**🔧 技术方法**

采用slimmable网络技术在专家内部共享参数并按宽度切片；训练时使用两宽度随机采样（全宽与随机宽度）结合MoE正则化；推理时实现三种宽度分配策略（统一宽度、路由概率映射、测试时训练（TTT）学习宽度映射）；

**📊 数据集**

在OpenWebText、WikiText‑103进行语言建模训练；在LAMBADA与Winograd Schema Challenge进行零样本下游任务评估；

**📈 对比分析**

与标准MoE和统一宽度MoSE对比，MoSE在相同或更低FLOPs下实现更低perplexity（在GPT‑2 Small/Standard/Medium模型上均优于基线），并通过TTT进一步提升性能，整体上在多种规模与路由配置下均能将Pareto前沿向下平移；

**⚠️ 局限性**

局限性包括：目前仅在可从头训练的中小规模LLM上验证；多宽度功能需在训练阶段嵌入，尚未证明可在已训练模型后期实现；实验规模有限，未在极大规模模型上验证；

---

## 24. Judging What We Cannot Solve: A Consequence-Based Approach for Oracle-Free Evaluation of Research-Level Math

**arXiv ID:** 2602.06291 | [PDF](https://arxiv.org/pdf/2602.06291v1)

**作者:** Guijin Son `[一作]` (Seoul National University), Youngjae Yu `[通讯]` (Seoul National University)

**通讯引用:** 2089 | [OpenAlex ID](https://openalex.org/A5101881857)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于后果的无 Oracle 验证方法（Consequence-Based Utility）来评估研究级数学问题的候选解。

**💡 创新点**

创新点在于通过衡量候选解在相邻可验证问题上的迁移性能（后果）来判定其正确性，而非直接对原问题做检验。

**🔧 技术方法**

利用大型语言模型（如 GPT‑OSS、Qwen 等）作为求解器，在给定候选解作为上下文后在邻域问题上推理，并用平均准确率计算 Utility；同时与奖励模型、LLM 判断器等基线对比。

**📊 数据集**

使用了 192 道专家撰写的研究级数学题及其 425 条 LLM 生成变体的公开数据集，总计 630 条 LLM 生成解答。

**📈 对比分析**

在 Acc@1、Recall@5、AUC 等指标上，CBU 在所有基线之上显著提升：Acc@1 从 67.2 提升到 76.3，AUC 从 71.4 提升到 79.6；对难题保持更强的正确‑错误分离。

**⚠️ 局限性**

局限性包括需构造可验证的邻域问题（人工或高质量自动生成），邻域难度需在“甜点区”内；当问题太易或太难时 Utility 失效，且构造邻域仍需人工或昂贵模型支持。

---

## 25. Flow Matching for Offline Reinforcement Learning with Discrete Actions

**arXiv ID:** 2602.06138 | [PDF](https://arxiv.org/pdf/2602.06138v1)

**作者:** Fairoz Nower Khan `[一作]` (University of Kentucky), Peizhong Ju `[通讯]` (University of Kentucky)

**通讯引用:** 144 | [OpenAlex ID](https://openalex.org/A5085838919)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于连续时间马尔可夫链（CTMC）的离散动作流匹配框架（QDFM），实现了离线强化学习中的价值引导策略生成与多目标、偏好条件化。

**💡 创新点**

创新点在于将连续动作空间的流匹配转化为离散动作空间的CTMC流匹配，并通过Q加权终点重采样实现价值引导；同时支持多目标和多智能体联合动作空间的因子化处理，理论证明在理想条件下可恢复KL正则化最优策略。

**🔧 技术方法**

使用的核心技术包括：离散流匹配（Discrete Flow Matching）、CTMC 率模型、Q加权条件流匹配损失、基于偏好的值函数刻度化、因子化联合动作率、以及Euler式CTMC采样推断。

**📊 数据集**

实验数据集主要包括：离散化的 MuJoCo 6 个离线基准（HalfCheetah、Hopper、Walker2d 等）、多目标离散环境 Deep-Sea-Treasure（DST）和 Resource-Gathering（RG），以及两智能体协调游戏。

**📈 对比分析**

与 AWAC、AWBC、BCQ、CQL、GreedyQ 等主流离线 RL 基线以及标量化 CQL 在多目标场景对比，QDFM 在 6 个离散化 MuJoCo 任务中 4/6 取得最优或相当表现，生成时间更快；在多目标环境中，QDFM 能发现多达 15 个非支配解并获得较高的超体积比；在两智能体游戏中，QDFM 在协调率、超体积和分布范围上显著优于独立和集中式行为克隆。

**⚠️ 局限性**

局限性包括：对数据覆盖率要求较高，离散化会导致连续控制任务的最优性下降；采样过程引入方差，虽不影响平均性能但可能导致收敛波动；当前仅在离线 RL 场景验证，缺乏在线或非离线环境的实验。

---

## 26. Identifying Adversary Tactics and Techniques in Malware Binaries with an LLM Agent

**arXiv ID:** 2602.06325 | [PDF](https://arxiv.org/pdf/2602.06325v1)

**作者:** Zhou Xuan `[一作]` (Purdue University), Xiangyu Zhang `[通讯]` (Purdue University)

**通讯引用:** 304976 | [OpenAlex ID](https://openalex.org/A5100362465)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了首个针对去符号化恶意二进制的 LLM 代理，用于在函数级别识别 MITRE ATT&CK TTP，并在二进制级别实现完整的 TTP 归因。

**💡 创新点**

创新点在于：① 结合稠密检索与 LLM 神经检索的混合检索策略高效定位潜在分析入口；② 引入 Context Explorer 按需递归扩展调用图以缓解函数级别的部分可观测性；③ 在推理时引入 TTP‑specific Reasoning Guideline，使 LLM 的推理过程与 ATT&CK 业务逻辑对齐；④ 提供全新函数级 TTP 标注数据集并使用源代码映射到二进制，填补了数据缺口。

**🔧 技术方法**

核心技术包括：LLM（Claude 3.7 Sonnet）+ OpenAI Embedding + FAISS 检索 + AutoGen 代理框架 + 调用图探索工具 + 结构化推理指南；同时采用 IDA Pro 反编译、函数重命名、神经检索的 prompt 设计。

**📊 数据集**

使用自建 Dataset I（18 个公开泄露恶意项目，源代码→二进制，210 个函数‑TTP 对）和 Dataset II（8 个 MalwareBazaar/Virusshare 样本，其中 2 个附带专家报告）。

**📈 对比分析**

与两种基线（未改进的直接提示与加入函数重命名+摘要的提示）相比，函数级 TTP 识别平均精度 93.25%/召回 93.81%（基线约 72%/68%）。二进制级别 87.37% 精度，覆盖率 85.7% 的专家报告 TTP，并发现平均 10.5 个新 TTP；混合检索将候选对减少 85% 以上，显著降低后续推理成本。

**⚠️ 局限性**

局限性包括：① 依赖人工源代码标注，可能存在主观误差；② 结果受 LLM 推理能力影响，跨模型表现不一；③ 数据集规模有限，难以完全覆盖真实恶意多样性；④ 训练集可能出现数据泄漏风险，需在更大、更新的数据上验证。

---

## 27. Misophonia Trigger Sound Detection on Synthetic Soundscapes Using a Hybrid Model with a Frozen Pre-Trained CNN and a Time-Series Module

**arXiv ID:** 2602.06271 | [PDF](https://arxiv.org/pdf/2602.06271v1)

**作者:** Kurumi Sashida `[一作]` (Nagoya Institute of Technology), Gouhei Tanaka `[通讯]` (University of Tokyo)

**通讯引用:** 4225 | [OpenAlex ID](https://openalex.org/A5025420478)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了针对嗅觉障碍者的触发声检测，提出基于冻结预训练CNN与时间序列模块的混合模型，并使用合成声景数据训练多类及少样本个性化检测。

**💡 创新点**

创新点在于：1) 通过Scaper合成强标注的误噪声触发声数据集；2) 将预训练CNN冻结、仅训练可微时间模块，显著降低参数量；3) 采用ESN等轻量化时序网络实现读出层训练，从而在保持性能的同时大幅压缩模型。

**🔧 技术方法**

技术包括冻结的MobileNetV3前端、GRU/LSTM/ESN及其双向变体作为时序模块、共享线性读出层、PSDS1/事件级F1等评估指标、Optuna超参搜索以及多种正则化手段。

**📊 数据集**

使用自制的七类误噪声触发声合成数据集（共10 k个10 s音频，包含呼吸、咳嗽、进食、呼气、打嗝、敲击键盘、钟表滴答），以及少样本个性化实验的进食声子集。

**📈 对比分析**

对比线性、GRU、LSTM、ESN的单向与双向版本，结果显示双向GRU在PSDS1上最高（0.63），但双向ESN仅用约1.4 万可训练参数即可获得0.55的PSDS1；在少样本“进食声”任务中，BiESN的平均PSDS1高于BiGRU并且波动更小。

**⚠️ 局限性**

局限包括：1) 依赖合成数据，缺乏真实环境评估；2) 对不同个体触发声的细粒度辨别有限；3) 仍需验证流式推理的时延和鲁棒性。

---

## 28. Don't Break the Boundary: Continual Unlearning for OOD Detection Based on Free Energy Repulsion

**arXiv ID:** 2602.06331 | [PDF](https://arxiv.org/pdf/2602.06331v1)

**作者:** Ningkang Peng `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 618 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的边界保持类忘记方法 TFER，专门用于在 OOD 检测模型中精确移除指定类别，同时保持 ID 类别的判别性能和 OOD 泛化能力。

**💡 创新点**

创新点：
- 将类忘记视作将目标样本转化为 OOD（Forget‑as‑OOD）的几何任务；
- 设计基于自由能原理的 Push‑Pull 动态，即用拉力保持 ID 结构，用推力将待忘记类别推向高能 OOD 区域；
- 采用参数高效的 LoRA 模块实现对模型的局部更新，并通过正交约束构建持续忘记的模块化体系，避免灾难性回忆。

**🔧 技术方法**

技术手段：
- 低秩适配（LoRA）与正交约束；
- 基于 von Mises–Fisher 的原型 OOD 检测框架；
- 自由能推力（TFER）损失与保护拉力损失；
- 低维投影与 Mahalanobis 归一化等特征映射。

**📊 数据集**

使用的数据集：
- ID 数据：CIFAR‑100；
- 外部 OOD 测试集：SVHN、LSUN、Textures、Places365、iSUN；
- 对比实验中使用不同忘记类别数（5,10,15,20）及多任务顺序忘记。

**📈 对比分析**

与基线比较：
- 原始模型（Upper bound）、从头训练（Retrain）、梯度上升（GradAsc）、随机标签微调（RL‑FT）。
- 结果显示：TFER 在保持 ID 准确率约 74–76% 的同时，FPR95 低至 2–4%，且对外 OOD 的 AUROC 与 FPR 也显著优于其它方法。
- 计算开销大幅降低：可训练参数仅占原始模型的 0.24%，训练时间约 9 分钟，比完整重训练快 20 倍。

**⚠️ 局限性**

局限性：
- 主要验证于图像分类和原型‑基 OOD 框架，尚未在其他任务或更大规模数据集上充分验证；
- 对高维噪声或样本分布剧烈变动的鲁棒性尚需进一步研究；
- 需要先验的预训练模型作为基座，若缺乏可迁移的表示可能效果受限；
- 对忘记类别数极大时的性能降解与长期连续忘记的可扩展性仍有待探索。

---

## 29. Computing an approximation of the partial Weyl closure of a holonomic module

**arXiv ID:** 2602.06209 | [PDF](https://arxiv.org/pdf/2602.06209v1)

**作者:** Hadrien Brochet `[一作]` `[通讯]` (Inria), Hadrien Brochet (Inria)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的算法，用来近似计算含参数的局部 Weyl 封闭（partial Weyl closure）并实现了一个可在 Julia 包 MultivariateCreativeTelescoping.jl 中使用的工具。

**💡 创新点**

创新点在于：① 将局部 Weyl 封闭转化为对多项式 f 的 saturation，② 采用 Rabiowitsch 变形引入新变量 T 并将求解转化为在无限阶 W‑模块上的 Gröbner 基计算；③ 通过截断 T 次数并递增阶数的方式实现迭代收敛，从而避免了昂贵的 b‑函数计算与全局本地化步骤。

**🔧 技术方法**

使用的技术包括：非交换 Gröbner 基（F4/F4‑style）、无限阶 W‑模块的截断与升阶、T 变量的 Rabiowitsch 变形、以及对多项式 f 的 saturation 计算。

**📊 数据集**

在性能评测中使用了多组测试函数：来自 Mgfun 包的 10 个符号函数（包含 Bessel、Airy、对数、指数等），以及一些手工构造的高奇异度例子（如包含 1/(x²–y³) 的函数）。

**📈 对比分析**

与现有在 Singular（slimgb）和 Macaulay2（gbw）实现的 Tsai 算法做对比；在大多数例子中 Julia 实现比两者快几倍到十倍；但在涉及 Bessel 函数的极限情况下，Julia 只得到部分 Weyl 封闭，未能获得完整闭包。

**⚠️ 局限性**

主要限制是：算法只给出局部 Weyl 封闭的一个 holonomic 近似；没有完整的停止判据保证得到真正的封闭；对极端奇异度（如 Bessel 系）时可能无法得到完整结果；且对大规模模块仍需改进 Gröbner 基的效率。

---

## 30. Knowledge Synthesis Graph: An LLM-Based Approach for Modeling Student Collaborative Discourse

**arXiv ID:** 2602.06194 | [PDF](https://arxiv.org/pdf/2602.06194v1)

**作者:** Bo Shui `[一作]` (University of Illinois), Xinran Zhu `[通讯]` (University of Illinois)

**通讯引用:** 316 | [OpenAlex ID](https://openalex.org/A5016939843)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于大型语言模型的知识合成图（KSG）构建方法，用于可视化并促进异步学生协作讨论中的思维发展。

**💡 创新点**

创新点在于将学生讨论视为知识合成过程，设计了微观观念、合成节点和认知关系三类节点，并通过迭代提示工程实现从原始注释到KSG的完整管道，兼顾认知细微差别与学习者主动性。

**🔧 技术方法**

使用 GPT‑4o、GPT‑5 等大型语言模型，并通过三阶段管道（预处理、合成节点生成、关系提取）实现自动抽取、标注与链接。

**📊 数据集**

数据集来自一门研究生教学设计课程的 42 条社交注释（Perusall）以及对应的课程阅读材料。

**📈 对比分析**

通过专家编码和量化指标（Kappa、Macro F1、执行率、跨模型一致性）评估，微观观念标签的 Kappa 达 0.643、Macro F1 0.722；合成节点在提供摘要+教师提示时表现最佳；关系提取在两级编码方案下执行率与一致性均提升，表明方法在可靠性上具备可行性。

**⚠️ 局限性**

局限性包括：LLM 对细微观点的忽略与可能的“幻觉”风险；提示依赖性强，需大量人工迭代；关系提取仍受主观性影响，且当前未在真实课堂中验证学习效果。

---

## 31. Agentic Workflow Using RBA$_θ$ for Event Prediction

**arXiv ID:** 2602.06097 | [PDF](https://arxiv.org/pdf/2602.06097v1)

**作者:** Purbak Sengupta `[一作]` (Aarhus University), Sonal Shreya `[通讯]` (Aarhus University)

**通讯引用:** 286 | [OpenAlex ID](https://openalex.org/A5049805518)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套基于事件先行、频率感知与 agentic 机制的风电功率 ramp 事件预测框架，融合 RBA_θ 事件抽取、离散小波分解、Hawkes 过程、多臂老虎机特征选择、LSTM/Transformer 编码以及事件驱动的轨迹重构。

**💡 创新点**

创新点在于：① 直接预测 ramp 事件而非先预测轨迹再提取，显著提升事件精度与可迁移性；② 采用多尺度频率分解与事件语义的联合建模，实现对不同时间尺度 ramp 的精准把握；③ 设计自适应的工作流选择（agentic）层，通过上下文多臂老虎机与 UCB/ε‑greedy 机制动态切换不同模型，实现零样本迁移与多时域预测。

**🔧 技术方法**

使用技术包括：RBA_θ 事件提取、离散小波变换 (DWT)、频率感知的多臂老虎机特征选择、Hawkes 过程建模、长短期记忆（LSTM）与 Transformer 编码、频率感知多头预测头、事件引导的逆小波重构、上下文多臂老虎机工作流选取以及 UCB+ε‑greedy 探索。

**📊 数据集**

数据集：以 40 年 ERA5 reanalysis 的 Baltic Eagle 离岸风电场为主训练/验证/测试集；零样本迁移至 Baie de Saint‑Brieuc 与 London Array；此外在实验初期使用八机组合成集做基准验证。

**📈 对比分析**

对比方法：传统 SARIMAX‑RBA_θ（轨迹优先）、RF‑RBA_θ（直接事件分类）以及 LSTM/Transformer 事件预测模型。实验表明在 Baltic Eagle 上 24 h 预测，LSTM/Transformer 事件模型的 F1 ≈0.84‑0.90，RMSE 约 51 MW，R² 0.90；零样本迁移至 Baie de Saint‑Brieuc/F1≈0.60，表明事件先行模型具备良好的跨场站泛化能力；与传统轨迹优先模型相比，在事件精度、可迁移性和计算成本上均有显著提升。

**⚠️ 局限性**

limitations: ① 对 1–3 h 近似短时 ramp 的预测仍不理想；② LSTM/Transformer 训练成本高，尤其是 Transformer；③ 未实现完整的概率不确定性量化；④ 目前仅针对单站点，缺乏空间耦合建模；⑤ 依赖 RBA_θ 的阈值设置，难以处理极端天气情形；⑥ 只在离散模拟/ERA5 数据上验证，缺少实际 SCADA 误差与故障影响。

---

## 32. Relevance-aware Multi-context Contrastive Decoding for Retrieval-augmented Visual Question Answering

**arXiv ID:** 2602.06050 | [PDF](https://arxiv.org/pdf/2602.06050v1)

**作者:** Jongha Kim `[一作]` (Korea University), Hyunwoo J. Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练的多上下文对比解码方法RMCD，改进检索增强生成（RAG）框架；

**💡 创新点**

通过给每个检索上下文分配可正可负的权重，实现对相关上下文的强化与对不相关上下文的抑制；

**🔧 技术方法**

采用Contrastive Decoding与上下文相关性加权、可组合解码、可组合可行性约束等技术；

**📊 数据集**

在InfoSeek、Encyclopedic‑VQA和OK‑VQA三大知识密集型视觉问答基准上进行评估；

**📈 对比分析**

与多种解码策略（无检索、单检索、SCD、多检索一致性、最大概率、拼接）以及最新模型（RA‑VQA、FLMR）对比，RMCD在所有模型和数据集上均实现最佳或次佳性能，且计算复杂度最低；

**⚠️ 局限性**

对检索质量仍有一定依赖，且在极端检索错误场景下仍受限；

---

## 33. Coupled Local and Global World Models for Efficient First Order RL

**arXiv ID:** 2602.06219 | [PDF](https://arxiv.org/pdf/2602.06219v1)

**作者:** Joseph Amigo `[一作]` (Machines in Motion Laboratory, New York University), Ludovic Righetti `[通讯]` (Artificial and Natural Intelligence Toulouse Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在真实机器人数据上训练高保真扩散世界模型，并利用分离的前向/后向模型实现第一阶梯度强化学习，完成视觉控制任务。

**💡 创新点**

创新点在于提出DMO框架，将全局扩散模型用于前向轨迹生成，局部隐状态RSSM用于梯度计算，从而兼顾精度与可微分性，突破了传统模型基 RL 在像素空间的样本与计算瓶颈。

**🔧 技术方法**

采用扩散世界模型（DIAMOND、DreamerV4）、递归状态空间模型（RSSM）、第一阶梯度强化学习（FoG-MBRL、SAPO）、Transformer+Axial Attention 等技术实现前向/后向分离与梯度优化。

**📊 数据集**

使用从Flexiv Rizon-10S机械臂和Unitree Go2四足机器人在真实环境中收集的 4 小时 Play/Demo 数据（Push‑T）和 12 小时 Ego‑Centric 交互数据（Push‑Cube）进行离线训练。

**📈 对比分析**

与 PPO、No Diffusion、ACT 基线比较，Push‑T 任务中 DMO 仅需 8M 交互样本即达成 9/10 成功率，而 PPO 需 40M 并仅 1/10；Push‑Cube 任务中 DMO 4M 样本即 4/4 成功率，PPO 25M 仅 1/10；总体显示 DMO 在样本与时间效率上显著优于基线。

**⚠️ 局限性**

局限性包括对低层控制器的高度依赖，扩散模型训练与推理成本仍较高，对极大规模、极复杂环境的可扩展性有限，以及局部 RSSM 在策略外推时可能出现误差放大。

---

## 34. Reimagining Legal Fact Verification with GenAI: Toward Effective Human-AI Collaboration

**arXiv ID:** 2602.06305 | [PDF](https://arxiv.org/pdf/2602.06305v1)

**作者:** Sirui Han `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18566 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对18位非诉讼律师进行半结构化访谈，探讨GenAI在法律事实核实中的使用现状、挑战与未来协作设想。

**💡 创新点**

首次从人机协作角度阐述事实核实的动态感知流程，并提出可追溯、可解释与专业化的GenAI设计需求。

**🔧 技术方法**

采用大语言模型（ChatGPT/DeepSeek）辅助信息检索、文档生成及结构化查询，但重点在HCI研究方法和主题分析。

**📊 数据集**

访谈录音及人工转写文本，未使用公开法律数据集。

**📈 对比分析**

通过对比律师现有工作流程与GenAI辅助情境，定性评估效率提升与风险承受度；未给出定量指标。

**⚠️ 局限性**

样本规模有限、受访者主观描述，缺乏观察或实验验证，且仅聚焦非诉讼领域。

---

## 35. MMEarth-Bench: Global Model Adaptation via Multimodal Test-Time Training

**arXiv ID:** 2602.06285 | [PDF](https://arxiv.org/pdf/2602.06285v1)

**作者:** Lucia Gordon `[一作]` (Harvard University), Nico Lang `[通讯]` (University of Copenhagen)

**通讯引用:** 1581 | [OpenAlex ID](https://openalex.org/A5048243105)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MMEarth‑Bench多模态地理观测基准，并在其上评估多种预训练模型，提出测试时多模态重建（TTT‑MMR）自监督适配方法。

**💡 创新点**

创新点：①新建包含5个全局任务、12模态、随机与地理两种测试拆分的基准；②提出模型无关的多模态重建 TTT‑MMR，可在测试时利用所有可用模态进行自监督适配；③系统研究多模态输入对地理泛化的影响。

**🔧 技术方法**

采用自监督预训练、Transformer/ConvNeXt/VIT 架构、多模态融合、联合训练（JT）、TTT‑MMR 与 TTT‑MMR‑Geo、梯度归一化、批量化等技术。

**📊 数据集**

使用MMEarth‑Bench（包含上述5个任务、12个对齐模态）作为评估数据集，对比七个预训练模型（Scale‑MAE、DINOv3 Web/Sat、SatlasNet、MPMAE、TerraMind、Copernicus‑FM）以及随机初始化的 ConvNeXtV2A。

**📈 对比分析**

通过在随机与地理测试集上进行 fine‑tune，比较 JT、TTT‑MMR、TTT‑MMR‑Geo 的性能；结果表明 TTT‑MMR 在所有模型和任务上均显著提升，尤其在地理测试集上提升幅度大；多模态预训练优于 RGB 单模，但仍存在显著地理泛化缺口。

**⚠️ 局限性**

局限性：预训练模型在地理泛化仍存在较大差距；多模态输入在测试时可能导致地理过拟合；TTT‑MMR 仅支持有限迭代；基准仅包含单时序任务；数据划分可能未完全覆盖全球分布，且对模型结构改动有限。

---

## 36. Pragmatic Curiosity: A Hybrid Learning-Optimization Paradigm via Active Inference

**arXiv ID:** 2602.06104 | [PDF](https://arxiv.org/pdf/2602.06104v1)

**作者:** Yingke Li `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1867 | [OpenAlex ID](https://openalex.org/A5019603699)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了“Pragmatic Curiosity”框架，利用主动推理中的期望自由能最小化，统一学习（信息获取）与优化（目标追求）于一体，解决黑盒评估中的混合学习-优化问题。

**💡 创新点**

创新点在于：①将目标导向与信息探索映射为单一期望自由能目标；②通过温度参数控制“好奇心”平衡探索与利用；③基于能量函数构造灵活的偏好分布，适用于未知或随时间演变的目标；④提供理论保证与实践指导。

**🔧 技术方法**

核心技术包括：主动推理（Active Inference）与期望自由能（EFE）框架；高斯过程（GP）作为后验模型；信息增益与互信息计算；Boltzmann（Gibbs）分布用于将能量函数映射为偏好；多任务实验设计。

**📊 数据集**

实验数据集涵盖三类任务：
- 环境监测中的2D plume 现场源定位、风向/风速估计、多源识别（真实传感器与仿真数据）；
- 自动驾驶感知失效场景下的失败发现（YOLO检测器的三维输入-二维输出）；
- 多目标优化中的车辆安全、青霉素产量仿真、分布式能源资源分配（高维仿真/真实场景）。

**📈 对比分析**

方法与传统BO（最大化目标、UCB、EI等）和BED（信息增益、熵搜索等）基线比较。结果显示：
- 在约束系统辨识中，估计误差降低≈40%且约束违背率为0；
- 在主动搜索中，关键失效区域覆盖率提升约10%；
- 在未知偏好优化中，成功学习偏好函数并获得更高预期效用，传统基线往往失效。整体而言，Pragmatic Curiosity 在所有任务上实现了更高的样本效率、更好的目标满足度与更优的最终解。

**⚠️ 局限性**

局限性：对能量/偏好模型的显式设定敏感，误设可能导致偏离安全或公平目标；依赖GP等代理模型，若模型失配或非平稳性严重，信息量估计与偏好引导可能失效；目前仅针对单代理、单步决策，尚未扩展到多代理、长期或多保真度场景。

---

## 37. AnyThermal: Towards Learning Universal Representations for Thermal Perception

**arXiv ID:** 2602.06203 | [PDF](https://arxiv.org/pdf/2602.06203v1)

**作者:** Parv Maheshwari `[一作]` (Carnegie Mellon University), Wenshan Wang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17759 | [OpenAlex ID](https://openalex.org/A5011189440)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种通过知识蒸馏从 RGB DINOv2 得到的、无任务专属训练的热成像特征提取骨干 AnyThermal，并构建了 TartanRGBT 开源采集平台与多环境 TartanRGBT 数据集。

**💡 创新点**

创新点在于将多环境 RGB‑Thermal 数据集用于跨域知识蒸馏，得到通用的热特征；同时首次公开了同步 RGB‑Thermal 采集硬件与数据集，填补热数据稀缺的瓶颈。

**🔧 技术方法**

采用了 DINOv2 ViT‑B/14 的教师‑学生蒸馏，基于 CLS 令牌的对比损失，结合 VPR、分割与深度估计的轻量化头部，以及软硬件同步采集。

**📊 数据集**

使用 ViVID++, STheREo, Freiburg, Boson Nighttime, 自研室内与越野数据共五个多域 RGB‑Thermal 数据集进行蒸馏，并在 CART, MS2, OdomBeyondVision 等零样本集上评估；另外发布了 TartanRGBT 数据集。

**📈 对比分析**

与 RGB‑only DINOv2、SALAD、ImageBind、SGM 等基线在三大任务上进行零样本或训练后比较，AnyThermal 在跨模 VPR 的 Recall@1 提升至 81% 以上、热分割的 mIoU 提升 36% 以上、单目深度估计的 AbsRel 降至 0.0883，整体优于现有最佳方案。

**⚠️ 局限性**

局限性包括数据量仍相对有限，缺乏高精度 GPS/激光雷达里程计导致对齐误差；蒸馏仅基于 RGB‑Thermal 对齐的 3D 深度估计，可能对复杂场景的精度有影响。

---

## 38. Coding Agents with Environment Interaction: A Theoretical Perspective

**arXiv ID:** 2602.06098 | [PDF](https://arxiv.org/pdf/2602.06098v1)

**作者:** Nicolas Menet `[一作]` (ETH Zurich), Abbas Rahimi `[通讯]` (IBM Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个概率框架，统一解释编码代理的后生成选择与迭代回溯（backprompting）两种环境交互策略，并从信息论角度分析其性能。

**💡 创新点**

创新点在于：①证明“软”估计（基于功能相似度）在信噪比上严格优于传统的“硬”估计；②将backprompting建模为近似Thompson采样，给出包含不可观测奖励项的新的累积后悔界限，揭示任务描述歧义导致的不可逆后悔；③基于上述理论改进任务描述，创建新基准QiskitHumanEvalSimX。

**🔧 技术方法**

采用概率估计、功能相似度核、Monte Carlo抽样、Thompson采样（近似）以及信息论后悔分析等技术；模型实现通过LLM生成代码、测试集和执行反馈，使用压缩方法处理长测试报告。

**📊 数据集**

实验使用三大开源模型（Qwen3‑235B, GPT‑OSS‑120B, MiniMax‑M2.1）在 BigCodeBenchHard、QiskitHumanEvalSim、LeetCodeDataset 以及新构造的 QiskitHumanEvalSimX 上进行评测。

**📈 对比分析**

与传统后生成选择方法（MBR‑Exec, AlphaCode, CodeT, MaxPass）及多轮自测试反馈比较，软估计平均提升 Pass@1 约 5–15 %；在oracle‑test 及任务描述更清晰的设置下，backprompting 可达到 60–70 % 的 Pass@1；对比结果表明，压缩策略和测试集优先生成能进一步提升性能。

**⚠️ 局限性**

局限性包括：①后悔界限中的不可逆后悔项 Δ 受任务描述歧义限制；②模型对长上下文或多轮反馈的处理受限；③实验仅覆盖公开数据集，未评估在更大规模或更复杂语言环境中的泛化能力。

---

## 39. Asymptotically Optimal Aperiodic Doppler Resilient Complementary Sequence Sets Via Generalized Quasi-Florentine Rectangles

**arXiv ID:** 2602.06045 | [PDF](https://arxiv.org/pdf/2602.06045v1)

**作者:** Zheng Wang `[一作]` (Southwest Jiaotong University), Keqin Feng `[通讯]` (Tsinghua University)

**通讯引用:** 2468 | [OpenAlex ID](https://openalex.org/A5108616637)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了通用的广义准佛莱顿矩阵定义，并利用该结构构造了新的无周期 Doppler‑鲁棒互补序列集（DRCS）。

**💡 创新点**

创新点在于将准佛莱顿矩阵扩展为可缺失多元素的广义版本，显著提高了可用行数，并通过该矩阵与 Butson 型 Hadamard 矩阵组合，得到的 DRCS 集在长度和集合大小上均超过以往最优结果。

**🔧 技术方法**

主要技术包括组合矩阵构造（广义准佛莱顿矩阵、圆形佛莱顿矩阵及其组合）、Butson 型 Hadamard 矩阵的 Kronecker 乘积以及利用极点相位构造序列的符号映射。

**📊 数据集**

该工作为纯理论构造，不涉及实验数据集；所有结果均基于数学证明与符号计算得到。

**📈 对比分析**

与之前的 DRCS 构造方法比较，所得到的集合尺寸更大、最优性因子更低（趋近 1），在多组具体参数实验中，最佳性因子从 1.56 降至 1.10 以上，证明了所提方法的性能优势。

**⚠️ 局限性**

局限性包括：需要先验存在相应的广义准佛莱顿矩阵（即 N 必须满足一定的质因数分布条件），对于某些 N 值仍无法获得更大行数；构造复杂度较高，且在实际硬件实现中需要进一步验证符号映射与硬件兼容性。

---

## 40. D-Legion: A Scalable Many-Core Architecture for Accelerating Matrix Multiplication in Quantized LLMs

**arXiv ID:** 2602.06252 | [PDF](https://arxiv.org/pdf/2602.06252v1)

**作者:** Ahmed J. Abdelmaksoud `[一作]` (University of Edinburgh), Themis Prodromakis `[通讯]` (University of Edinburgh)

**通讯引用:** 8184 | [OpenAlex ID](https://openalex.org/A5088089733)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文设计并评估了一种名为 D-Legion 的可扩展多核可调精度 systolic array 架构，用于加速量化 LLM 的矩阵乘法和注意力计算。

**💡 创新点**

创新点在于将多核 ADiP 核心组织为 Legion 单元，并结合块结构稀疏、PSUM 并行归约、矩阵 tile 多播、稀疏窗口检测与动态调度等技术，显著提升吞吐量、降低延迟和内存访问。

**🔧 技术方法**

使用了 ADiP 及 DiP 数据流、可调精度 PE、并行累加器、网络互连 (NoC)、块矩阵乘法、稀疏窗口检测、工作负载映射与调度等技术，并在 1-bit LLM 上进行实验。

**📊 数据集**

实验使用了两种 1-bit LLM 数据集：BitNet‑1.58B 与 BitNet‑1.58B‑KV，分别包含 MHA 和 GQA 注意力层。

**📈 对比分析**

通过周期精确仿真与 WS、DiP、ADiP 以及 Google TPUv4i 进行对比，D‑Legion 在延迟上相较 WS/DiP/ADiP 分别降低 9.26×/8.84×/5.2×，吞吐量提升同等倍数，内存访问及 PSUM 访问分别降低 2.5×/4.25× 与 3×。

**⚠️ 局限性**

局限性包括尚未完成物理实现与功耗评估，稀疏性需要先验结构信息，且在更大规模 Legions 时受限于 HBM 带宽与内存容量。

---

## 41. Do LLMs Track Public Opinion? A Multi-Model Study of Favorability Predictions in the 2024 U.S. Presidential Election

**arXiv ID:** 2602.06302 | [PDF](https://arxiv.org/pdf/2602.06302v1)

**作者:** Riya Parikh `[一作]` (Massachusetts Institute of Technology), Chara Podimata `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5027923437)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们通过每日查询 9 种大型语言模型（LLM），对其预测的总统候选人（哈里斯和特朗普）的好感度，并与五份高质量退出民调结果进行比较，检验 LLM 是否能跟踪公众舆论。

**💡 创新点**

首次系统评估多款 LLM 与传统民调的偏差，揭示哈里斯好感度被普遍过度预测、特朗普好感度偏差相对较小，并指出这一误差在时间平滑后仍持续。

**🔧 技术方法**

使用基于提示的预测问卷、正则表达式解析自由文本为百分比分布，并采用 7 天滚动平均平滑时间序列，随后与民调数据做对比。

**📊 数据集**

利用公开的 Hugging Face 数据集 "llm-election-data-2024"（包含 LLM 对候选人好感度的每日预测）以及从 Roper Center 选出的 5 份来自 Reuters、CNN、Gallup、Quinnipiac、ABC 的退出民调。

**📈 对比分析**

对比方法是将 LLM 的好感度百分比分布与民调平均值对齐，结果显示 LLM 对哈里斯的好感度高估 10–40%，对特朗普的好感度低估 5–10%；误差不因时间平滑而显著减小，说明 LLM 仍缺乏可靠的校准。

**⚠️ 局限性**

局限性包括：仅使用“总体人群”提示，许多模型拒绝回答或无法解析；候选人之间的偏差差异大，说明模型对不同候选人信息的覆盖不均；缺乏针对性校准方法，导致 LLM 不能直接替代传统民调。

---

## 42. SPDA-SAM: A Self-prompted Depth-Aware Segment Anything Model for Instance Segmentation

**arXiv ID:** 2602.06335 | [PDF](https://arxiv.org/pdf/2602.06335v1)

**作者:** Yihan Shang `[一作]` (Ocean University of China), Xinghui Dong `[通讯]` (Ocean University of China)

**通讯引用:** 1053 | [OpenAlex ID](https://openalex.org/A5026151886)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个自提示、深度感知的 Segment Anything Model（SPDA‑SAM），实现了无人工提示、利用深度信息的实例分割。

**💡 创新点**

创新点在于：①通过语义‑空间自提示模块（SSSPM）自动生成高质量提示，消除人工提示依赖；②引入粗细层次 RGB‑D 融合模块（C2FFM），在单目图像上自学习深度信息并融合多尺度特征，提升空间结构感知。

**🔧 技术方法**

采用的技术包括：SAM 双路径编码器、LoRA 微调、跨模态注意力、粗细融合块、语义与空间提示生成、前向两阶段掩码解码等。

**📊 数据集**

在 12 个公开实例分割数据集上评估：COME15K、DSIS、SIP、Cityscapes、KITTI、LIACI、USIS10K、UIIS、ZeroWaste 等。

**📈 对比分析**

与 22+ 传统与 SAM 基础模型进行对比，SPDA‑SAM 在所有数据集上均取得最高 mAP、AP50/AP75，提升幅度从约 1% 到 10% 以上。

**⚠️ 局限性**

局限性包括：仍依赖预训练的单目深度估计器，深度质量影响最终性能；模型参数与 FLOPs 较大，对实时或资源受限场景存在一定瓶颈；在极端光照或无可估深度的环境下效果有限。

---

## 43. Code, Capital, and Clusters: Understanding Firm Performance in the UK AI Economy

**arXiv ID:** 2602.06249 | [PDF](https://arxiv.org/pdf/2602.06249v1)

**作者:** Waqar Muhammad Ashraf `[一作]` (University of Cambridge), Ramit Debnath `[通讯]` (University of Cambridge)

**通讯引用:** 1555 | [OpenAlex ID](https://openalex.org/A5007263620)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用英国公司注册、ONS、glass.ai 等公开数据，构建 2000‑2024 年英国 AI 企业的纵向面板，研究企业规模、技术专长与地区社会经济变量对营收和人均营收的影响，并对行业生命周期进行 2030 年预测。

**💡 创新点**

首次将企业层级的技术关键词特征与邮编级社会经济指标结合，用 SHAP 解释模型揭示规模与技术专精对营收与人均产出的双重驱动，并提出针对区域政策的实证建议。

**🔧 技术方法**

采用 CatBoost 梯度提升树配合 SHAP 进行解释性回归，并使用 ARIMA、Theta、ETS、MFLES 等四种单变量时序模型对行业生命周期进行预测。

**📊 数据集**

整合 UKRI WAIFinder、Companies House、glass.ai 及 ONS 2021 人口普查数据，共计 4,392 家 AI 实体，其中可得营收与员工数的样本为 451 家。

**📈 对比分析**

通过对比四种时序模型的 RMSE/MAE，ARIMA 在总企业与解散率预测最佳、ETS 在活跃企业预测最佳、MFLES 在解散企业预测最佳；CatBoost 模型对营收的 R² 为 0.55、对人均营收的 R² 为 0.74，说明模型具有较高解释力。

**⚠️ 局限性**

数据来源受限于公开财报，可能低估早期创业和非报告公司；关键词评分仅反映技术维度，未细化产品成熟度或专利情况；缺乏跨国对比，局限于英国内部分析。

---

## 44. Taming SAM3 in the Wild: A Concept Bank for Open-Vocabulary Segmentation

**arXiv ID:** 2602.06333 | [PDF](https://arxiv.org/pdf/2602.06333v1)

**作者:** Gensheng Pei `[一作]` (Sungkyunkwan University), Byeungwoo Jeon `[通讯]` (Sungkyunkwan University)

**通讯引用:** 4816 | [OpenAlex ID](https://openalex.org/A5074587654)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对基于SAM3的开放词汇分割模型在跨域部署时因数据漂移和概念漂移导致的对齐失效问题，提出了一种无参数的概念库（ConceptBank）校准框架，动态重构目标域的文本锚点以恢复视觉与文本的匹配；

**💡 创新点**

1）利用目标域视觉原型对概念进行对齐；2）通过代表性支持样本挖掘滤除漂移下的异常样本；3）对候选文本进行功能性评分并软融合得到单一校准嵌入，从而在不更新模型参数的情况下完成跨域适配；

**🔧 技术方法**

冻结的SAM3+PE-L+14视觉分割器、CLIP文本编码器、GPT-5.2或Gemini-3 Pro生成文本扩展、余弦相似度与IoU评分、温度软融合；

**📊 数据集**

自然场景的八个基准（PASCAL VOC, Pascal Context, COCO-Object, COCO-Stuff, Cityscapes, ADE20K 等）和四个遥感基准（LoveDA, Potsdam, Vaihingen, iSAID）；

**📈 对比分析**

与多种基线（CLIP、FreeDA、SFP、SAM-CLIP、ReME 等）对比，在自然场景平均 mIoU 由 57.5 提升至 67.1，遥感平均 mIoU 由 39.1 提升至 52.1，显著提升且保持无参数、运行时不增加文本编码；

**⚠️ 局限性**

仍受限于标签定义不确定性、概念漂移与注释不一致的交叉；当类别描述含糊或重叠时，校准后的概念库难以完全消除语义误匹配。

---

## 45. Can One-sided Arguments Lead to Response Change in Large Language Models?

**arXiv ID:** 2602.06260 | [PDF](https://arxiv.org/pdf/2602.06260v1)

**作者:** Pedro Cisneros-Velarde `[一作]` `[通讯]` (VMware Research), Pedro Cisneros-Velarde (VMware Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了通过仅提供单方面论据来引导大型语言模型在二元争论问题上倾向特定立场。

**💡 创新点**

创新点在于证明单向论据能在多种模型、问法和论证展示方式下有效操纵模型立场，并揭示问答与论据呈现的一致性提升说服力。

**🔧 技术方法**

采用对比实验，系统构造二元争议问题集，并使用五种主流LLM（gpt-oss-120b、Llama 3.3 70B、Llama 3.1 8B、Mistral 7B、Gemma 3 4B），通过“YES/NO”与“Agree/Disagree”问答以及对话式与块式论据呈现进行评估。

**📊 数据集**

构建了约30个历史、政治、宗教类争议主题的手工数据集，包含132条一面向论据，总共产生多组实验。

**📈 对比分析**

实验显示，无论模型与设置如何，正面回答比例显著提升，且当切换论据时正面回答大幅下降，说明论据内容是主要驱动力，整体成功率在大多数设置下达到70%以上。

**⚠️ 局限性**

局限在于仅使用简短无修辞的单句论据，未考虑多样化论证结构、事实误导与恶意言论对模型的影响，且结果仍受模型内部偏好与安全约束的限制。

---

## 46. Online Adaptive Reinforcement Learning with Echo State Networks for Non-Stationary Dynamics

**arXiv ID:** 2602.06326 | [PDF](https://arxiv.org/pdf/2602.06326v1)

**作者:** Aoi Yoshimura `[一作]` (Nagoya Institute of Technology), Gouhei Tanaka `[通讯]` (International Research Center for Neurointelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于回声状态网络（ESN）和递归最小二乘（RLS）的在线自适应强化学习框架，能在无预训练下快速应对仿真到真实的动力学漂移。

**💡 创新点**

创新点在于使用ESN做短期记忆编码，并仅在线更新线性输出层，避免了大规模反向传播和先验信息需求，实现零-shot 速度适应。

**🔧 技术方法**

技术包括Echo State Network、Recursive Least Squares、Soft Actor-Critic（SAC）以及对策略输入的状态增强。

**📊 数据集**

实验数据集为CartPole和HalfCheetah仿真环境，加入周期风扰和突然摩擦变化的非平稳动态。

**📈 对比分析**

与DR、RMA、PAD等基线对比，ESN-OA在极端扰动下保持接近最优奖励，收敛仅需数十步，且推理时间仅0.27ms，明显优于梯度基方法。

**⚠️ 局限性**

局限在于RLS对噪声敏感且缺乏显式安全约束，未验证在真实机器人硬件上的鲁棒性。

---

## 47. CAST: Character-and-Scene Episodic Memory for Agents

**arXiv ID:** 2602.06051 | [PDF](https://arxiv.org/pdf/2602.06051v1)

**作者:** Kexin Ma `[一作]` (National University of Defense Technology), Liting Sun `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于角色和场景的记忆架构 CAST，将时间、地点和动作三维聚合为场景，并以角色档案为索引，实现了情节化的短时记忆与语义记忆的双模融合。

**💡 创新点**

创新点在于借鉴戏剧理论的“情节统一”和“角色档案”，通过视窗式对话切片、角色主次标记以及三维聚类实现事件边界划分，显著提升了对“谁、何时、何地做了什么”类问题的召回与准确性。

**🔧 技术方法**

使用了 LLM（gpt‑4o‑mini 与 Llama‑3.3‑70B‑Instruct）、OpenIE、NER、向量检索（DPR）、图检索（HippoRAG2）、场景编码与融合重排等技术。

**📊 数据集**

在 LOCOMO（多会话对话记忆基准）和 epbench（多种情景记忆任务）两个数据集上进行实验。

**📈 对比分析**

与 RAG、HippoRAG、Zep、Mem0、LangMem 等基线对比，CAST 在 F1 和 LLM‑as‑a‑Judge 指标上平均提升 8.11% / 10.21%，在 open 对话记忆任务上领先 13.24% F1、8.32% J，整体表现最优。

**⚠️ 局限性**

局限包括：场景边界划分和多尺度场景聚合仍需学习化，融合策略过于规则且对高成本的语义图构建有依赖，未充分探讨动态或可学习的融合方式。

---

## 48. Transformer-Based Reinforcement Learning for Autonomous Orbital Collision Avoidance in Partially Observable Environments

**arXiv ID:** 2602.06088 | [PDF](https://arxiv.org/pdf/2602.06088v1)

**作者:** Thomas Georges `[一作]` (Université Paris-Saclay), Adam Abdin `[通讯]` (Université Paris-Saclay)

**通讯引用:** 480 | [OpenAlex ID](https://openalex.org/A5058202685)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于Transformer的强化学习框架，用于在部分可观测的空间轨道碰撞规避任务中做出机动决策。

**💡 创新点**

创新点包括：① 在距离依赖的观测模型中将观测误差与相对距离直接关联；② 用Mahalanobis距离作为碰撞风险的光滑奖励代理；③ 将Transformer‑XL作为策略网络，以自注意力聚合时间序列观测；④ 将轨道传播、状态估计、碰撞风险评估与决策合并为统一的POMDP框架。

**🔧 技术方法**

技术手段包括：强化学习（PPO），Transformer‑XL网络，LSTM/MLP基线，Unscented Kalman Filter 状态估计，Kepler两体动力学，Mahalanobis距离与概率碰撞计算。

**📊 数据集**

数据集：使用逆向遭遇生成器在 LEO 中随机采样合成碰撞事件（单体碰撞），涵盖不同的碰撞距离、相对速度与接近角度，构成五种观测衰减（距离相关）场景。

**📈 对比分析**

方法比较：在五种观测衰减下将 Transformer‑XL 与无记忆的 MLP 进行对比。实验显示 Transformer‑XL 在中等至严重观测衰减下平均燃料消耗降低约 7–10%，最小接近距离缩小约 10%，碰撞概率均保持在 10⁻⁴ 以下；在极端观测条件下差距减小或略增。离线测试（384 场未见样本）验证了这些收益的泛化性。

**⚠️ 局限性**

局限性：① 仅考虑单体碰撞，未涉及多体优先级和机动顺序；② 采用两体Kepler模型，忽略 J₂、阻力等扰动；③ 碰撞风险以奖励方式约束，缺乏正式安全保证；④ 观测衰减函数假设已知，实际系统中可能变化或不确定。

---

## 49. BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks

**arXiv ID:** 2602.06221 | [PDF](https://arxiv.org/pdf/2602.06221v1)

**作者:** Nishant Balepur `[一作]` (University of Maryland), Jordan Lee Boyd-Graber `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了BenchMarker工具，用于自动检测多项选择题基准中的污染、捷径和写作错误，帮助提升评测质量；

**💡 创新点**

首次将教育研究中的三大评测错误（污染、捷径、写作规范）系统化为自动判别指标，并通过Judge-as-a-Model实现高一致性；

**🔧 技术方法**

采用大语言模型（GPT‑5、Gemini、Claude等）配合Web搜索API、Prompt‑based Judge进行判别，并构建19条写作规则评估；

**📊 数据集**

对12个主流多项选择基准（TruthfulQA、HellaSwag、SocialIQA等）以及教育领域的标准化考试进行采样评估；

**📈 对比分析**

BenchMarker在人工标注上取得超过80%的一致率，F1和Cohen κ显著高于随机基线；对基准的审计揭示写作错误普遍且对模型准确率和排名有显著影响；

**⚠️ 局限性**

缺乏自动修复机制，依赖人工后续处理；仅评估英语数据，未覆盖多语言、多专业领域，且部分判别依赖昂贵的搜索API和闭源模型。

---

## 50. A methodology for analyzing financial needs hierarchy from social discussions using LLM

**arXiv ID:** 2602.06431 | [PDF](https://arxiv.org/pdf/2602.06431v1)

**作者:** Abhishek Jangra `[一作]` (TCS Research), Jayasree Raveendran `[通讯]` (TCS Research)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

使用大语言模型对Reddit个人理财帖子进行文本摘要、需求提取、年龄收入识别等处理，进而验证财务需求在NHF与NPF两大层级框架中的层级结构。

**💡 创新点**

创新点在于：①首次将生成式AI应用于社交媒体文本，直接抽取并映射财务需求层级；②结合两种主流理论框架（NHF与NPF）进行跨框架验证；③提供数据驱动、可扩展的替代问卷调查方法。

**🔧 技术方法**

核心技术包括：Llama系列LLM（通过Groq API）用于摘要、需求抽取、年龄/收入、情绪、压力、风险倾向识别；LDA MALLET进行主题建模；统计分析检验需求层级假设。

**📊 数据集**

数据集来源于Reddit四个子板块（r/personalfinance、r/finance、r/financialindependence、r/wealthbuilding等），覆盖2020‑2023年，13,821条帖子中筛选出334位用户的6,709条帖子，包含年龄、月收入等信息。

**📈 对比分析**

与传统问卷/面板数据方法对比：模型能够准确抽取需求并映射到层级，验证H1、H2，展示收入与层级正相关；通过主题建模进一步揭示各层级主题分布；虽然未给出具体准确率数值，但实验结果显示层级映射稳定且与收入、压力、风险相关性符合理论预期。

**⚠️ 局限性**

局限性：仅基于美国Reddit用户的横截面数据，未考虑时间序列变化；自述的年龄、收入信息可能存在误报；情绪识别受限于文本表达；结果在其他平台或地区的泛化性需要进一步验证。

---

## 51. MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model

**arXiv ID:** 2602.06393 | [PDF](https://arxiv.org/pdf/2602.06393v1)

**作者:** Geonmo Gu `[一作]` (NAVER AI Lab), Dongyoon Han `[通讯]` (NAVER AI Lab)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多轮对话式对比学习框架（Multi‑Turn Contrastive Learning, MTCL），利用多模态大语言模型（MLLM）在单次前向推理中同时生成多条查询-目标嵌入，从而实现更高效、更具语境关联的通用多模态嵌入。

**💡 创新点**

创新点：
- 将单轮对比学习转化为多轮对话结构，允许嵌入在先前轮次的上下文中累积，提升表示的连贯性与区分度；
- 通过一次视觉编码、后续文本轮次提取多条嵌入，显著放大有效批量并降低计算开销；
- 设计对数掩码策略，避免同图像内的正样本被误判为负样本；
- 采用增量监督（compounded supervision）与自回归重构任务强化模型对查询与目标关系的理解。

**🔧 技术方法**

技术：
- MLLM 后端（Qwen2‑VL）+ LoRA 微调；
- InfoNCE 对比损失；
- 对话模板与特殊嵌入标记（[MASK]、[EMBED]）构造多轮输入；
- 对数掩码与负样本排除；
- 生成式数据合成（dense caption + LLM 生成 7 条 Q‑T 对）；
- 适应单轮数据的 prompt‑based 复现与图像转文本。

**📊 数据集**

数据集：
- 主要：5M 图像 + 35M 交互式 Q‑T 对（M3T，基于 DataComp、MLLM + LLM 合成）；
- 对照：mmE5、MegaPairs 等单轮预训练数据。

**📈 对比分析**

对比实验：
- MMEB：7B MTCL 在 zero‑shot 时 Precision@1 61.6%（升至 58.6% 的 mmE5‑11B），fine‑tune 后 69.5%（> 68.1% 的 B3‑2B）与 73.6%（> 72.0% 的 B3‑7B）；
- M‑BEIR：2B、7B 模型在多模态检索任务上均超过同规模基线，Recall 提升 1–2%；
- 计算效率：在 1024 图像批量下，通过 7 轮实现 7168 有效样本，FLOPs 仅提升 3%（18.0 PFLOPs vs 17.5 PFLOPs），远低于传统方法（122.7 PFLOPs）。

**⚠️ 局限性**

局限性：
- 依赖大规模合成数据，真实多模态场景下的多轮语义多样性仍有限；
- 对 logit 掩码与上下文累积的设计需要手工调优，易受模型架构与提示方式影响；
- 目前评估集中在检索与 VQA 等标注任务，未深入探讨更复杂的跨模态推理或对话生成；
- 需要 MLLM 后端支持，推理成本仍受视觉编码器瓶颈限制。

---

## 52. A neuromorphic model of the insect visual system for natural image processing

**arXiv ID:** 2602.06405 | [PDF](https://arxiv.org/pdf/2602.06405v1)

**作者:** Adam D. Hines `[一作]` (Macquarie University), Andrew B. Barron `[通讯]` (Macquarie University)

**通讯引用:** 8978 | [OpenAlex ID](https://openalex.org/A5035671163)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种仿生昆虫视觉系统的神经形态模型，可将密集图像下采样为稀疏的Kenyon细胞码，并可作为多任务的视觉模块使用。

**💡 创新点**

创新点在于将昆虫视网膜-lamina-medulla-lobula等处理层映射为卷积网络，并结合自监督对比学习与k‑WTA实现稀疏码，同时提供ANN与SNN两种实现，且模型不需针对单一任务手工调参即可迁移。

**🔧 技术方法**

使用卷积神经网络、LeakyReLU+LocalResponseNorm+GroupNorm激活-归一化、自监督对比学习（SimCLR/NT‑Xent）、稀疏化掩码与k‑WTA、以及LIF积分-放电的SNN实现；训练后通过线性分类和余弦相似度评估。

**📊 数据集**

主要数据集包括Tiny ImageNet用于训练，17类花瓣数据集(17CFD)与小型花卉测试集（薰衣草/向日葵）用于分类实验，以及Gardens Point Walking数据集用于视觉定位（VPR）实验。

**📈 对比分析**

与原始像素、SAD基线及未训练模型对比，单 epoch训练即可在17CFD上达到70‑75%分类准确率；在VPR任务中，Recall@K相较SAD提升约10%；稀疏码在同类/异类之间的余弦相似度显著区分。

**⚠️ 局限性**

局限性包括SNN版分类性能低于ANN、对时步与记忆需求较高、对光照或姿态变化的鲁棒性待验证，以及需手动设置稀疏比例与阈值等超参数。

---

## 53. MemGUI-Bench: Benchmarking Memory of Mobile GUI Agents in Dynamic Environments

**arXiv ID:** 2602.06075 | [PDF](https://arxiv.org/pdf/2602.06075v1)

**作者:** Guangyi Liu `[一作]` (Zhejiang University), Yong Liu `[通讯]` (Zhejiang University)

**通讯引用:** 35783 | [OpenAlex ID](https://openalex.org/A5100712539)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MemGUI‑Bench，一套基于记忆的移动 GUI 代理基准，包含 128 个跨应用的记忆挑战任务、并行快照评估框架和 Progressive Scrutiny 的多阶段 LLM‑as‑Judge 评判器；并对 11 个现有代理进行系统性评估。

**💡 创新点**

创新点包括：①系统化的短期/长期记忆分类与 5 种记忆架构的对比；②大量以跨时空信息保留为核心的任务集；③快照重置的并行环境支持；④七维记忆指标与 pass@k 评估；⑤基于 LLM 的分阶段评判策略，显著提升评估效率与准确性。

**🔧 技术方法**

技术手段涵盖：Gemini 2.5 Flash/Pro 作为 LLM，Progressive Scrutiny 评判器（Triage Judge、Step Descriptor、Semantic Judge、Visual Judge 等），快照重置的 Android 仿真器并行化，pass@k、IRR、MTPR 等记忆特定指标，LLM‑as‑Judge 的多阶段评估流程。

**📊 数据集**

使用的数据集为 MemGUI‑Bench 本身：128 个跨 26 个真实应用的任务（89.8% 为记忆挑战），64 对镜像任务用于跨会话学习；以及 26 个 SPA‑Bench 任务用于评判器验证。

**📈 对比分析**

与 11 代理进行对比实验，发现：
- 现有代理在记忆任务上 4‑10 倍缺口；
- 短期记忆为必需，长期记忆可提升 21.9 pp；
- 跨应用复杂度导致 16‑40 pp 下降；
- 长上下文提升 18.8 pp；
- 评判成本约 0.031 美元/轨迹；
- 通过 Multi‑Attempt pass@k、IRR 等指标量化学习效果与执行效率。

**⚠️ 局限性**

局限性：
- 评判依赖大型 LLM，成本与能耗高；
- 对图像与文本生成的准确性仍有挑战；
- 只在 Android 模拟器上评估，缺少真实设备多样性；
- 长期记忆机制尚未充分利用，模型在跨会话学习上仍有限；
- 计算与 token 限制对部署有较大影响。

---

## 54. Lost in Speech: Benchmarking, Evaluation, and Parsing of Spoken Code-Switching Beyond Standard UD Assumptions

**arXiv ID:** 2602.06307 | [PDF](https://arxiv.org/pdf/2602.06307v1)

**作者:** Nemika Tyagi `[一作]` (Arizona State University), Olga Kellert `[通讯]` (Arizona State University)

**通讯引用:** 60 | [OpenAlex ID](https://openalex.org/A5028941004)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对口语式双语代码混用文本的句法分析，本文提出了专门的评价指标与解析框架，构建了专家标注的金标数据集，验证了其有效性。

**💡 创新点**

①系统化的口语代码混用现象分类与标注准则；②针对口语数据的可扩展评估指标FLEX‑UD；③解耦式解析框架DECAP，将口语现象处理与核心句法分析分离。

**🔧 技术方法**

基于大型语言模型（GPT‑4.1）实现的多代理解析体系，采用自定义的句法约束、分词与语言特定归一化步骤；评估指标采用FLEX‑UD与传统UAS/LAS。

**📊 数据集**

使用Miami Corpus中的英西双语口语样本，挑选126句构建SpokeBench金标语料库。

**📈 对比分析**

与传统Stanza、多语言与BiLingua管线对比，DECAP在FLEX‑UD上整体分数提升至76.2%，在重复、省略、话语元素等口语复杂类别上显著优于基线，传统UAS/LAS指标虽受限，但仍表现出改进趋势。

**⚠️ 局限性**

数据集规模有限且仅覆盖英西双语，未能充分体现不同方言或交互情境；评测依赖LLM提示质量，缺少可训练模型的长期稳定性；需进一步扩展到更大、更多语言对的口语语料。

---

## 55. Can Post-Training Transform LLMs into Causal Reasoners?

**arXiv ID:** 2602.06337 | [PDF](https://arxiv.org/pdf/2602.06337v1)

**作者:** Junqi Chen `[一作]` (Fudan University), Chaochao Lu `[通讯]` (Shanghai Artificial Intelligence Library)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者通过对小型LLM实施多种后训练技术（SFT、DPO、KTO、PPO、GRPO），提升其在七项因果推断任务上的推理与数值估计能力。

**💡 创新点**

首次系统性评估主流后训练方法对LLM因果推断性能的影响，并发现在线强化学习中的GRPO是最有效的，能将14B模型推到93.5%精度，超越更大模型。

**🔧 技术方法**

采用监督微调（SFT）、离线强化学习（DPO、KTO）和在线强化学习（PPO、GRPO）等五种后训练技术，并基于预训练LLM进行冷启动。

**📊 数据集**

构建了包含七种因果任务（ATE、CDE、ETT、NDE、NIE、PN、PS）的训练集和五个专用测试集，并结合CaLM基准与三大数学基准（Math 500、Minerva Math、AMC 2023）进行评估。

**📈 对比分析**

与多种基线模型（包括更大规模LLM、数理推理基线）对比，GRPO在CaLM上实现93.5%准确率，在线RL方法总体表现优于离线RL和SFT，证明后训练可以显著提升因果推理性能。

**⚠️ 局限性**

研究仅聚焦于特定因果推断任务，未检验模型在更广泛因果推断场景下的通用性与稳健性，且仅使用合成数据，可能影响对真实世界因果问题的适用性。

---

## 56. On Randomized Algorithms in Online Strategic Classification

**arXiv ID:** 2602.06257 | [PDF](https://arxiv.org/pdf/2602.06257v1)

**作者:** Chase Hutton `[一作]` (University of Maryland), Han Shao `[通讯]` (University of Maryland)

**通讯引用:** 67779 | [OpenAlex ID](https://openalex.org/A5082634513)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在线战略分类问题，针对可实现和非可实现两种情形，给出了随机化学习器的误差与后悔界限，并首次提供了随机化学习器的下界与改进的上界；

**💡 创新点**

创新点在于：①首次给出随机化学习者在可实现设置下的下界 Ω(min{√(Td),dΔ})，证明随机化在此场景下无法突破传统确定性下界；②在非可实现环境下提出了一个合适学习器，后悔上界为 O(√(T log|H|)+|H| log(T|H|))，与已知下界匹配（忽略对数因子），从而实现了近似最优；

**🔧 技术方法**

技术手段包括：加权专家（weighted‑experts）框架、随机混合探索/利用策略、SOA（Standard Optimal Algorithm）用于无穷维类、FTRL（Follow‑the‑Regularized‑Leader）配合 log‑barrier 正则化、重要性加权估计、凸优化等；

**📊 数据集**

研究完全为理论分析，未使用具体数据集；

**📈 对比分析**

与之前的确定性算法（如 O((|H|)Δ log Δ) 误差、O(T^{3/4}log^{1/4}T|H|) 后悔）相比，本文的随机化上界在小 T 时更优；在非可实现下达到了 O(√(T log|H|)+|H| log(T|H|))，已逼近理论下界；

**⚠️ 局限性**

局限性：对小 T 或高维图的性能仍有改进空间；在非可实现场景下的上界仍保留对数因子；此外，对于无限类的精确度与实际实现仍有待进一步研究。

---

## 57. MGP-KAD: Multimodal Geometric Priors and Kolmogorov-Arnold Decoder for Single-View 3D Reconstruction in Complex Scenes

**arXiv ID:** 2602.06158 | [PDF](https://arxiv.org/pdf/2602.06158v1)

**作者:** Luoxi Zhang `[一作]` (University of Tsukuba), Itaru Kitahara `[通讯]` (Center for Computational Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种名为MGP-KAD的单视角3D重建框架，结合RGB图像与几何先验的多模态特征融合，并通过Kolmogorov–Arnold网络(KAN)混合解码器生成高质量3D表面。

**💡 创新点**

核心创新点：①利用采样聚类得到的类级几何先验，动态在训练中进行调整；②采用KAN结构的多尺度混合解码器，突破传统线性MLP在处理多模态输入时的局限；③将多头注意力机制与几何先验检索相结合，实现精细的模态融合。

**🔧 技术方法**

技术手段包括：M3D编码器提取语义特征；多头注意力用于几何先验检索与融合；前端特征变换MLP与Softplus激活；KANLinear模块与动态网格自适应的混合解码器；可微渲染分支用于表面细化；最终采用Marching Cubes提取网格。

**📊 数据集**

在Pix3D真实场景数据集（12,471张图像–模型对，9个类别）上进行实验。

**📈 对比分析**

与SSR、MGN、LIEN、InstPIFu等SOTA方法进行对比；MGP-KAD在Pix3D上实现了9.86%的Chamfer Distance下降、6.03%的F-score提升以及12.2%的Normal Consistency提升，整体性能排名第一。

**⚠️ 局限性**

局限性：目前仅利用RGB和几何先验，未充分融合深度、法向等额外模态；对极端遮挡或光照变化仍有一定敏感性；模型训练需要较大显存和GPU资源，对类别样本不平衡仍需进一步优化。

---

## 58. Quantifying and Attributing Polarization to Annotator Groups

**arXiv ID:** 2602.06055 | [PDF](https://arxiv.org/pdf/2602.06055v1)

**作者:** Dimitris Tsirmpas `[一作]` (Athens University of Economics and Business), John Pavlopoulos `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 3494 | [OpenAlex ID](https://openalex.org/A5033894687)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种新颖的aposteriori unimodality（apunim）度量，用以量化并归因注释者群体的极化程度，并配备显著性检验。

**💡 创新点**

创新点在于给出可量化的极化归因值，兼容单标签与多标签、极不平衡群体，并通过随机划分得到统计显著性。

**🔧 技术方法**

技术包括利用nDFU极化指标、随机分层采样产生的apriori极化、平均化求得apunim，以及Student‑T检验进行显著性评估，并实现为开源Python库。

**📊 数据集**

实验使用四个公开NLP数据集：Kumar、Sap、DICES‑350、DICES‑990，涵盖毒性与仇恨言论检测任务。

**📈 对比分析**

通过与随机分组相比的p值检验，发现注释者种族/族裔是最主要的极化驱动因素，且不同群体的apunim值在各数据集上显著，体现出该方法在不平衡、多标签场景下的有效性。

**⚠️ 局限性**

局限性包括对注释者属性信息依赖、缺少单标签聚合数据的实验、以及单个注释项极化估计受限于注释数量导致的噪声。

---

## 59. Adaptive Sparse Möbius Transforms for Learning Polynomials

**arXiv ID:** 2602.06246 | [PDF](https://arxiv.org/pdf/2602.06246v1)

**作者:** Yigit Efe Erginbas `[一作]` (University of California Berkeley), Kannan Ramchandran `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了两种基于群检测的算法，用可加查询高效地精确学习 s-稀疏、度为 d 的布尔多项式（即求 Möbius 变换）

**💡 创新点**

创新点在于将可加查询与群检测、d‑disjunct 矩阵及广义二分搜索相结合，突破了传统基底不可压缩和指数级 d 的瓶颈，实现了近似信息论下界的查询复杂度

**🔧 技术方法**

核心技术包括：Möbius 变换的 bin‑sum 结构、可加模型下的子集求和、群检测设计（d‑disjunct 矩阵与自适应二分搜索）、残余函数与递归子集拆分

**📊 数据集**

在实际超图重构中使用了两组数据集：ISCAS‑85 逻辑电路与 BiGG 代谢网络（n≈数百至数千，d≈十至百，s≈数十至数千）

**📈 对比分析**

与现有的基于奇异变换或 Boolean 边缘检测的算法相比，新方法在实验中显著降低了查询次数（避免了 2^d 的爆炸），并在所测试的超图规模上保持线性或低次多项式的查询复杂度，整体性能优于基线且接近下界

**⚠️ 局限性**

主要局限：需可加查询且满足子集求和独立性假设；对噪声不鲁棒；完全自适应算法的执行顺序限制并发；仍存在关于更少自适应回合或不需要非取消假设的开放问题

---

## 60. Secure and Private Spatial Sharing for Mixed Reality Remote Collaboration in Enterprise Settings

**arXiv ID:** 2602.06254 | [PDF](https://arxiv.org/pdf/2602.06254v1)

**作者:** Mengyu Chen `[一作]` (JPMorganChase), Blair MacIntyre `[通讯]` (JPMorganChase)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了企业环境下混合现实（MR）空间共享的安全与隐私风险，开展了员工与行业专家访谈，提出了概念框架与技术探针，并给出了相应的设计建议。

**💡 创新点**

将企业安全策略与MR空间共享相结合，提出了基于位置感知的访问控制、白名单式内容过滤机制，并通过技术探针验证了框架的可行性。

**🔧 技术方法**

使用了MR HMD相机、UWB定位、数字孪生模型、云端共享策略引擎、本地内容过滤模块、Swift 与 Python 开发的原型系统，以及可视化展示界面。

**📊 数据集**

主要使用访谈收集的定性数据；技术探针基于自建数字孪生模型与模拟环境，并未使用公开数据集。

**📈 对比分析**

采用主题分析法对访谈文本进行编码与归纳；未进行量化性能评估，探针演示仅展示功能可行性，未给出延迟或精度等指标。

**⚠️ 局限性**

局限包括：仅关注视觉数据，忽略音频等多模态；技术探针规模有限，缺乏大规模用户测试；策略管理复杂性可能导致开发者使用门槛高；在多元企业场景中的可扩展性与可部署性尚待验证。

---

## 61. Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels

**arXiv ID:** 2602.06382 | [PDF](https://arxiv.org/pdf/2602.06382v1)

**作者:** Wandong Sun `[一作]` (Harbin Institute of Technology), Zongwu Xie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1497 | [OpenAlex ID](https://openalex.org/A5006416939)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一套端到端的视觉驱动类人机器人行走策略，利用深度相机原始图像直接输出关节动作，并通过从高度扫描到深度图的行为蒸馏实现了多种复杂地形（楼梯、平台、跨栏、粗糙地面等）的统一控制。

**💡 创新点**

创新点包括：①构建完整的深度相机仿真管道，模拟立体匹配空洞、距离相关噪声、光学畸变与标定误差；②使用多场景专用奖励与多重评论家/鉴别器架构，使单一策略在不同地形中学习专属动力学和动作先验；③引入噪声不变辅助任务（去噪正则、KL正则）与行为克隆相结合的视觉蒸馏方法，实现从无噪声高度图到噪声深度图的无缝迁移。

**🔧 技术方法**

核心技术包括：域随机化与Perlin噪声的深度数据增强、卷积去噪深度编码器、行为克隆与去噪一致性损失、KL正则化、multi‑critic 与 multi‑discriminator 强化学习、对抗运动先验以及 DAgger‑style 蒸馏。

**📊 数据集**

数据集与实验环境：在仿真中生成20个难度等级的多地形样本（楼梯、平台、间隙、粗糙地面等），使用Orbbec Gemini 336L 与 Intel RealSense D435i 深度相机采集真实深度序列；构建 RDT‑Bench（含 CycleGAN 生成的真实感噪声深度）用于大规模评估；以及 30 次真实世界试验（室内外楼梯、平台、长阶梯、宽跨栏）。

**📈 对比分析**

在 RDT‑Bench 上与 Humanoid Parkour Learning、直接 RL、单评论家、无增强等基线对比，本文方法平均成功率 98.9%、平均功率 27.7×10¹ W、功率降解率仅 5.8%，显著优于所有基线；在真实世界测试中整体成功率 97.8%，楼梯上升 100%，下降 86.7%，其余场景均达 100%。

**⚠️ 局限性**

局限性主要体现在楼梯下降阶段，因重力放大误差和步梯边缘遮挡导致脚位判定不准；当前方法对近场高分辨率深度或预测脚位的支持不足，需要进一步提升传感精度或引入前瞻性脚位规划。

---

## 62. ASMa: Asymmetric Spatio-temporal Masking for Skeleton Action Representation Learning

**arXiv ID:** 2602.06251 | [PDF](https://arxiv.org/pdf/2602.06251v1)

**作者:** Aman Anand `[一作]` (Queens University), Farhana Zulkernine `[通讯]` (Queens University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Asymmetric Spatio‑Temporal Masking（ASMa）框架，用双向不对称的空间与时间掩码训练两个 ST‑GCN 编码器，并通过特征对齐模块融合多视角表示，再使用知识蒸馏压缩模型以适配边缘设备。

**💡 创新点**

创新点包括：
1) 结合高度节点（高度中心性）与低运动幅度、低度节点与高运动幅度的双重掩码策略，显著提升骨架表征的完整性；
2) 在三流（骨架、骨骼与运动）自监督预训练后使用双向注意力对齐，进一步融合异质信息；
3) 通过在蒸馏阶段对齐自监督教师与线性预训练教师的概率分布，得到比教师更具泛化能力的轻量化学生模型。

**🔧 技术方法**

采用 ST‑GCN 作为主干网络，配合 Barlow Twins 互相关损失进行三流自监督预训练；实现了不对称的空间掩码（高/低度节点）和时间掩码（高/低运动帧）；特征对齐模块基于双向多头注意力；知识蒸馏使用温度缩放的 KL 散度。

**📊 数据集**

在 NTU RGB+D 60、NTU RGB+D 120 以及 PKU‑MMD 三大骨架动作识别基准上进行预训练与迁移学习，特别在 PKU‑MMD‑Part‑II 这一噪声与视角变化显著的数据集上表现突出。

**📈 对比分析**

与先前自监督方法（SkeletonBT、SkeletonCLR、SkeletonMAE、PSTL 等）以及部分监督基线相比，ASMa 在三种数据集上均实现 1–6% 的线性评估提升，细调时可达 2.7–5.9% 的准确率提升；蒸馏版模型参数 91% 下降、推理速度 3 倍提升，边缘设备上仅损失 0.65% 准确率。

**⚠️ 局限性**

局限性包括：
1) 仍依赖高质量骨架估计，对姿态噪声敏感；
2) 只处理单人动作，未考虑多人交互或遮挡情况；
3) 蒸馏过程需要额外的教师训练与温度调参，可能增加实验成本。

---

## 63. Protean Compiler: An Agile Framework to Drive Fine-grain Phase Ordering

**arXiv ID:** 2602.06142 | [PDF](https://arxiv.org/pdf/2602.06142v1)

**作者:** Amir H. Ashouri `[一作]` (Huawei Technologies), Tomasz S. Czajkowski `[通讯]` (Huawei Technologies)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套完整的Protean编译器框架，实现LLVM的细粒度阶段排序与迭代优化；

**💡 创新点**

创新点在于将机器学习预测模型IR2Score与增量式Simulated Annealing迭代器结合，支持模块/函数/循环级别的自适应优化，同时提供了141维手工特征库，可与IR2VEC等第三方特征无缝集成；

**🔧 技术方法**

核心技术包括Transformer‑based IR2Score预测模型、Agglomerative聚类得到的5个优化子序列、Simulated Annealing搜索、LLVM自定义优化阶段和与MLGO/ACPO、LLM等外部框架的集成；

**📊 数据集**

使用CBench基准集（约30个C/C++程序）进行评估，并在Polybench、Coral‑2等数据集上训练IR2Score模型；

**📈 对比分析**

与LLVM默认O3、MiCOMP等传统方法对比，Protean平均提升4.1%（最高可达15.7%），并在与LLM或ACPO联合使用时分别实现10.1%和8.5%的额外加速；

**⚠️ 局限性**

局限性包括：模型预测误差导致实际加速略低于预测、构建时间仍受模型推理与特征收集影响、对极端大规模代码和跨平台（非x86）支持的验证不足。

---

## 64. Active Localization of Unstable Systems with Coarse Information

**arXiv ID:** 2602.06191 | [PDF](https://arxiv.org/pdf/2602.06191v1)

**作者:** Ege Yuceel `[一作]` (University of Illinois Urbana-Champaign), Sayan Mitra `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3824 | [OpenAlex ID](https://openalex.org/A5085795401)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种在单比特感知下对不稳定线性系统进行主动定位的理论框架与算法

**💡 创新点**

创新点在于证明不稳定性可逆向提升定位精度，并给出可在有限测量下实现初始状态指数收敛的控制与估计方法

**🔧 技术方法**

使用球面Voronoi分割、集合估计（椭圆包络）与可观测性/可控性理论相结合的技术

**📊 数据集**

实验使用随机生成的二维可控线性系统（A、B矩阵随机）与对应的初始状态与标识点不确定集

**📈 对比分析**

与传统只利用连续感知或离散测量的基线方法相比，所述算法在相同测量次数下实现了指数收敛，实验结果表明收敛速度优于理论上限

**⚠️ 局限性**

局限性包括仅处理单一标识点的线性系统、未考虑模型不确定性或噪声、对标识点估计收敛缺乏严格理论保证

---

## 65. Compressing LLMs with MoP: Mixture of Pruners

**arXiv ID:** 2602.06127 | [PDF](https://arxiv.org/pdf/2602.06127v1)

**作者:** Bruno Lopes Yamamoto `[一作]` (Universidade de São Paulo), Artur Jordao `[通讯]` (Universidade de São Paulo)

**通讯引用:** 242 | [OpenAlex ID](https://openalex.org/A5112859940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种结合深度与宽度裁剪的混合方法MoP，用于高效压缩大语言模型和多模态模型。

**💡 创新点**

创新点在于将深度裁剪（移除 Transformer 层）与宽度裁剪（移除注意力头和 MLP 神经元）通过迭代统一决策策略相结合，形成混合裁剪路径；同时引入多种路径选择指标并发现随机策略即可达到最优效果。

**🔧 技术方法**

核心技术包括：
- 迭代裁剪框架MoP；
- 采用 AMP 作为宽度裁剪器；
- 采用基于余弦相似度/KL/Perplexity 的路径评估；
- 通过小步裁剪与恢复微调（LoRA）保持模型性能；
- 对多模态模型 LLaVA-1.5 的文本恢复微调证明跨模态可迁移。

**📊 数据集**

使用的评估数据集包括：
- 语言模型：ARC-e/ARC-c、HellaSwag、PIQA、WinoGrande（EleutherAI LM Harness）以及 WikiText‑2 用作校准；
- 多模态模型：ScienceQA、VizWiz、LLaVA‑Bench、MM‑Vet（LMMs‑Eval）。

**📈 对比分析**

与多种基线方法（AmoebaLLM、PruneNet、SlimLLM、AMP、CoMe、LINEARPATCH 等）对比，MoP 在 20%、30% 与 40% 压缩率下均获得最高平均精度，且最差路径均优于所有基线。加速方面，MoP 在 30% 压缩率下实现 1.38× 的推理速度提升，比竞争方法在相同压缩率下的 1.17× 更快。

**⚠️ 局限性**

限制：
- 依赖裁剪前后微调，仍需额外训练资源；
- 对路径评估指标的选择仍可进一步优化，随机策略虽然有效但可能不适用于所有模型；
- 目前主要验证在 LLaMA 系列和 LLaVA‑1.5，跨更大规模或不同架构（如 GPT‑系列）需进一步实验；
- 对视觉子任务的恢复仍显著依赖文本微调，纯视觉任务的恢复效果尚未深入探究。

---

## 66. HiWET: Hierarchical World-Frame End-Effector Tracking for Long-Horizon Humanoid Loco-Manipulation

**arXiv ID:** 2602.06341 | [PDF](https://arxiv.org/pdf/2602.06341v1)

**作者:** Zhanxiang Cao `[一作]` (Shanghai Jiao Tong University), Yue Gao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 154913 | [OpenAlex ID](https://openalex.org/A5100399276)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种分层强化学习框架HiWET，实现人形机器人在世界坐标系下的长期端执行器跟踪；

**💡 创新点**

在世界坐标下的任务规划与动态执行分离，并结合Kinematic Manifold Prior（KMP）与状态估计提升空间一致性与动态稳定性；

**🔧 技术方法**

使用PPO强化学习、层次控制、残差动作空间、KMP预训练网络、状态估计网络、重要性采样、LiDAR+IMU定位；

**📊 数据集**

通过大规模基于PyRoki的IK采样（约千万条）和AMASS人类运动数据的转移学习；

**📈 对比分析**

与HOMIE及多种消融版本对比，HiWET在仿真端执行器平均误差12.4 mm、实机平均误差12–15 mm，定位误差0.10 m，显著优于基线；

**⚠️ 局限性**

受LiDAR定位精度限制、轨迹规模小、仅单臂轨迹、未涉及接触式操作。

---

## 67. M3: High-fidelity Text-to-Image Generation via Multi-Modal, Multi-Agent and Multi-Round Visual Reasoning

**arXiv ID:** 2602.06166 | [PDF](https://arxiv.org/pdf/2602.06166v1)

**作者:** Bangji Yang `[一作]` (University of Illinois Urbana-Champaign), Ge Liu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 5540 | [OpenAlex ID](https://openalex.org/A5100728581)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出M3框架，训练免费、通过多代理多轮推理自我修正来提升文本到图像生成的合成质量。

**💡 创新点**

创新性在于闭环多代理管线（Planner、Checker、Refiner、Editor、Verifier）与多轮迭代，避免一次性生成失败，且可无训练地插件于任何T2I模型。

**🔧 技术方法**

利用预训练Vision‑Language模型完成推理与检查，指令驱动图像编辑器执行局部修改，配合JSON交互和CLIP/检验器保证单调提升。

**📊 数据集**

在GenEval、OneIG‑EN以及其加硬版本上进行评测。

**📈 对比分析**

与Imagen4、Seedream 3.0等商业旗舰及多款开源模型对比，M3在OneIG‑EN总体得分0.532、GenEval属性绑定和空间推理显著提升，达到或超过现有最优。

**⚠️ 局限性**

受限于VLM推理能力、编辑器局部修改范围以及多轮推理的计算开销，极端复杂或外域内容仍可能难以完全修正。

---

## 68. A High-Fidelity Robotic Manipulator Teleoperation Framework for Human-Centered Augmented Reality Evaluation

**arXiv ID:** 2602.06273 | [PDF](https://arxiv.org/pdf/2602.06273v1)

**作者:** Harsh Chhajed `[一作]` (Worcester Polytechnic Institute), Tian Guo `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 1983 | [OpenAlex ID](https://openalex.org/A5051346938)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一个名为 ARBot 的实时遥操作平台，利用机器人手臂捕捉并重放人类在增强现实（AR）中的自然交互动作，为 AR 系统评估提供高保真、可重复的物理代理，并公开了完整代码和基准轨迹数据集。

**💡 创新点**

创新点包括：①多模态捕获机制（ARCore 手机+CV+IMU）实现多种自然交互采集；②主动安全的二次规划（QP）控制器，保证机器人在跟随人类轨迹时既平滑又不超限；③位置基实时控制架构，解决网络不稳定导致的积分漂移；④公开 132 条 100Hz 轨迹数据集，为 AR 轨迹评估提供真实基准。

**🔧 技术方法**

核心技术：ARCore VIO、Intel RealSense 与 IMU 传感器融合、Newton‑Raphson 逆运动学+阻尼最小二乘、OSQP 二次规划安全过滤、ROS2 控制框架、双协议低延迟网络栈、坐标系统统一与实时映射。

**📊 数据集**

使用公开的 132 条轨迹数据集（包括形状跟踪与重复性实验），每条轨迹记录姿态、IK 延迟、端到端延迟等信息，可直接用于算法验证和基准测试。

**📈 对比分析**

通过人类与机器人重放对比评估：ARPose 介质模式的系统延迟中位数 19.5 ms，CV+IMU 90.5 ms；两模式均达 5 mm 以内的绝对轨迹误差；机器人重放一致性（ITV）比人类高 10 倍（人类 75.6 mm → 机器人 7.4 mm）。

**⚠️ 局限性**

局限性：①传感器噪声与 VIO 跟踪失效仍可能导致偶发漂移；②机器人手臂仅能重现关节空间运动，缺乏手指细节与抓取动作；③数据集仅包含几种几何形状，不能覆盖所有 AR 交互场景；④对极高频/大幅位移动作的响应仍受硬件及网络带宽限制。

---

## 69. RuleSmith: Multi-Agent LLMs for Automated Game Balancing

**arXiv ID:** 2602.06232 | [PDF](https://arxiv.org/pdf/2602.06232v1)

**作者:** Ziyao Zeng `[一作]` (Yale University), Zhiwen Fan `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用多智能体大型语言模型进行自我对弈，结合贝叶斯优化和自适应采样自动调节游戏规则，从而实现异构战略游戏的平衡。

**💡 创新点**

① 零射击式从文本规则书直接执行自我对弈；② 将多智能体 LLM 评估与贝叶斯优化无缝耦合；③ 通过自适应采样动态分配游戏评估预算，显著提升样本效率；④ 输出可解释的规则调整，易于迁移到实际游戏。

**🔧 技术方法**

多智能体 LLM (InternVL3.5)、RAG (检索增强生成)、贝叶斯优化（Gaussian Process + Expected Improvement）、离散投影、基于采样的自适应评估。

**📊 数据集**

无外部真实数据集，所有评估均在自定义的极简策略游戏 CivMini 上完成，使用自发的对弈生成数据。

**📈 对比分析**

与随机搜索、(1+1)-ES、BO 固定采样等基线比较。结果显示 BO‑adaptive 在约 100 次迭代内完成 3500 场自我对弈即可将帝国与游牧部落的胜率逼近 50%±5%，优于其他方法；固定采样的 BO 在相同游戏次数下无法获得平衡。

**⚠️ 局限性**

① 依赖简化的游戏环境和 LLM 评估，可能无法反映真实玩家行为；② 需要大量算力（数十小时 GPU）；③ 对超参数和模型规模敏感；④ 仅提供经验平衡，缺乏形式化的最优性或鲁棒性保证。

---

## 70. Emergent Low-Rank Training Dynamics in MLPs with Smooth Activations

**arXiv ID:** 2602.06208 | [PDF](https://arxiv.org/pdf/2602.06208v1)

**作者:** Alec S. Xu `[一作]` (University of Michigan), Laura Balzano `[通讯]` (University of Michigan)

**通讯引用:** 3023 | [OpenAlex ID](https://openalex.org/A5029521003)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了多层感知器（MLP）在使用平滑激活函数时的梯度下降训练动态，并发现权重更新高度集中在不变的低维子空间内。

**💡 创新点**

创新点在于：①首次对两层非线性网络在梯度下降下的权重动态给出严格的低秩子空间定理；②通过实验验证该现象在更深网络、不同优化器、未白化数据和交叉熵损失下也普遍存在；③基于此构造了一种低秩参数化的MLP，并在恰当初始化时实现与全参数网络相近的分类性能。

**🔧 技术方法**

主要技术手段包括奇异值分解、正交投影与Wedin正弦定理分析梯度子空间、全批梯度下降、Adam/SGD+动量优化、正交/半正交初始化、低秩矩阵分解与子空间初始化策略。

**📊 数据集**

实验使用的数据集有：合成分类数据、Fashion MNIST、CIFAR‑10（以及在VGG‑16上预处理的CIFAR‑10）等。

**📈 对比分析**

通过将低秩MLP与全参数MLP在相同网络结构、学习率、优化器和训练轮次下进行对比，结果显示：在S_big子空间初始化时，低秩MLP几乎可获得与全参数网络相同的测试准确率；随机子空间初始化则表现显著较差；在VGG‑16分类头实验中，低秩模型与全参数模型的准确率差距约5%–10%，增大宽度可进一步缩小差距。

**⚠️ 局限性**

限制与不足包括：理论证明仅覆盖两层网络、白化输入、平方误差损失、第二层固定的情况；深层网络与实际训练设置的验证仍属于经验性；低秩网络对初始化非常敏感，若初始化不佳会陷入劣势；未对更多激活函数、网络规模以及其他任务进行全面评估。

---

## 71. VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation

**arXiv ID:** 2602.06270 | [PDF](https://arxiv.org/pdf/2602.06270v1)

**作者:** Yancheng Wang `[一作]` (Arizona State University), Yingzhen Yang `[通讯]` (Arizona State University)

**通讯引用:** 287 | [OpenAlex ID](https://openalex.org/A5013579313)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于VowelPrompt的可解释语音情感识别框架，利用时间对齐的元音段提取音高、强度和时长等低层描述符，经过归一化、分位数离散化后转化为自然语言提示，直接与文字转录拼接，让大语言模型在无原始音频的情况下完成情感判断。

**💡 创新点**

创新点在于：① 将音频信息细化到元音级别而非句子级别，显著提升情感相关细粒度特征；② 采用离散化的可解释自然语言描述代替黑盒音频嵌入；③ 设计两阶段微调策略——先用监督式微调（SFT）与人工生成的推理轨迹对齐，再通过可验证奖励（RLVR）与Group Relative Policy Optimization（GRPO）进一步提升推理质量与鲁棒性。

**🔧 技术方法**

技术要点包括：① 语音的时间对齐（MFA）、元音筛选（IPA）；② 低层描述符提取（F0均值/斜率/方差、强度均值/方差、持续时长）并进行说话人与元音类型归一化；② 分位数离散化并映射为“high pitch, rising, loud, lengthened”等自然语言片段；③ 以这些片段与文本拼接为输入，对大语言模型（GPT‑4o、LLaMA‑3‑8B、Qwen‑2‑7B‑Instruct）进行SFT + RLVR 微调；④ 使用可验证奖励实现推理过程的可检验性。

**📊 数据集**

实验使用五大情感识别基准：IEMOCAP、MELD、CaFE（法语）、EmoDB（德语）和混语种ASVP‑ESD，覆盖从演员语料到自然会话、多语言与多领域数据。

**📈 对比分析**

与基准方法（仅文本、SpeechCueLLM、SALMONN、InstructERC等）在零样本、SFT、SFT+GRPO、跨域和多语种设置下对比，VowelPrompt 在UACC/WF1、Weighted F1 等指标上普遍领先，最大提升可达 5%–7% 以上，且在跨域转移时的鲁棒性显著优于句子级语调描述方法。

**⚠️ 局限性**

局限性包括：① 依赖高质量的语音对齐与元音识别，若对齐错误会影响特征提取；② 目前仅考虑元音层次的情感线索，未充分利用辅音或更细粒度的声学特征；③ 需要预先生成推理轨迹用于SFT，增加了工程成本；④ 在极端噪声或多说话人混杂场景下的鲁棒性尚待进一步验证。

---

## 72. Do LLMs Act Like Rational Agents? Measuring Belief Coherence in Probabilistic Decision Making

**arXiv ID:** 2602.06286 | [PDF](https://arxiv.org/pdf/2602.06286v1)

**作者:** Khurram Yamin `[一作]` (Carnegie Mellon University), Bryan Wilder `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2854 | [OpenAlex ID](https://openalex.org/A5079207566)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一个基于决策理论的可检验框架，用以评估大语言模型在医疗诊断任务中口头表达的概率是否能作为其理性决策的主观信念。

**💡 创新点**

创新点在于给出了可假设检验的约束，将信念与行动联系起来，而不需要预先设定效用函数，并将概率一致性、条件独立性、单调性和期望律等多种检验方法整合到同一框架中。

**🔧 技术方法**

使用了条件互信息（kNN CMI）测试、CatBoost 预测提升、分箱单调性检验以及基于迭代期望的内部一致性度量，对黑盒模型的输出进行全程分析。

**📊 数据集**

实验数据来自四个医疗诊断数据集：结构性心脏病、糖尿病以及两份儿童医学贝叶斯网络（发热和哭闹），每个数据集采样200例并进行五次重复。

**📈 对比分析**

对GPT‑5（High/Min）、DeepSeek、Llama等多家 LLM 进行比较，结果显示大多数模型在信念‑行动一致性上表现为中等偏好，GPT‑High 在所有检验中持续优于其他模型，而部分模型在条件独立性和单调性检验中出现显著违背，说明其信念不完全可解释为理性决策。

**⚠️ 局限性**

局限性在于该方法仅提供可否定的检验，无法证明模型真正拥有理性信念；对效用函数保持不确定且对复杂决策策略可能缺乏捕捉；需要大量标注数据与细致的提示设计，且对不同任务的泛化能力尚未充分验证。

---

## 73. UAV-Enabled Short-Packet Communication via Fluid Antenna Systems

**arXiv ID:** 2602.06206 | [PDF](https://arxiv.org/pdf/2602.06206v1)

**作者:** Xusheng Zhu `[一作]` (University College London), Chan-Byoung Chae `[通讯]` (Yonsei University)

**通讯引用:** 10683 | [OpenAlex ID](https://openalex.org/A5079863632)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了无人机（UAV）辅助的短包通信系统，利用流体天线系统（FAS）在城市环境中实现可靠传输，并提供了闭式的块误码率（BLER）分析与能量效率（EE）优化方案。

**💡 创新点**

创新点在于：①首次在有限块长框架下对FAS‑UAV链路的相关Nakagami‑m衰落给出可解析BLER；②提出基于特征值的有效分支模型并给出高信噪比极限，揭示系统的多样性阶数；③构建包含FAS端口选择时空能耗的实际EE模型，并设计分层搜索求解非凸混合整数问题，发现存在最优端口数与最优UAV高度。

**🔧 技术方法**

使用了短包通信理论、Nakagami‑m衰落与Jakes相关性模型、特征值分解、极限逼近、Bessel函数与不完全伽马函数、Gauss‑Chebyshev求积、Bisection、整数搜索等技术。

**📊 数据集**

采用仿真数据：城市场景参数（如LoS/NLoS概率、路径损耗、射频参数、UAV轨迹、FAS端口数、块长度等）进行Monte Carlo仿真，无需公开数据集。

**📈 对比分析**

与传统单端口固定天线（FPA）对比，FAS在同样BLER下可节省约3 dB功率，且能通过调节端口数与UAV高度实现能量效率最大化；仿真验证了闭式BLER的准确性，并展示了最佳端口数与高度的凸最优特性。

**⚠️ 局限性**

局限性包括：假设CSI完美、仅考虑单跳半双工DF中继、未考虑多用户干扰或多UAV协同、FAS端口选择的时延与能耗模型简化、只在单一城市环境参数下评估，未来需扩展到多用户、多UAV与不完美CSI情形。

---

## 74. Robots That Generate Planarity Through Geometry

**arXiv ID:** 2602.06294 | [PDF](https://arxiv.org/pdf/2602.06294v1)

**作者:** Jakub F. Kowalewski `[一作]` (Northeastern University), Jeffrey Ian Lipton `[通讯]` (Northeastern University)

**通讯引用:** 1242 | [OpenAlex ID](https://openalex.org/A5102955404)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

研究并实现了一种基于球面到平面的几何逆变的Flat‑Plane Mechanism（FPM），通过链长和连杆配置实现自引用平面运动，可在微米到米级尺度上构造；并将其集成至机器人，完成表面测量与狭小空间内的3D打印；

**💡 创新点**

创新点在于首次将球面几何逆变映射到物理机构，用链长和几何约束自生成平面运动，并通过设计空间分析实现对制造误差的衰减；同时提出无需外部测量的迭代自校准工艺；

**🔧 技术方法**

采用几何逆变与立体投影理论、前向/逆向运动学、Kinematic sensitivity分析；在实验中使用两光子光刻、木质/碳纤维杆、磁球关节以及谐波驱动机器人实现；

**📊 数据集**

未使用公开数据集，全部使用实验测量数据，包括SEM、OptiTrack、触针探针、CMM扫描等来评估平面度和工作空间；

**📈 对比分析**

通过比较FPM的平面度、工作空间与传统CNC机床、Flexure定位装置的精度；在微尺度FPM实现23.8%工作空间/占地比，比现有Flexure装置高一倍；在机器人FPM扫描的三块平板与实验室CMM的RMSE差异仅为7.1，说明其测量性能与标准CMM相近；

**⚠️ 局限性**

局限性包括对高负载任务的适用性不足（需较大刚度），以及仅探索了单一FPM变体，未完全覆盖设计空间；同时机器人实现对机械复杂度、装配误差和环境干扰仍有改进空间。

---

## 75. Is my model "mind blurting"? Interpreting the dynamics of reasoning tokens with Recurrence Quantification Analysis (RQA)

**arXiv ID:** 2602.06266 | [PDF](https://arxiv.org/pdf/2602.06266v1)

**作者:** Quoc Tuan Pham `[一作]` (University of New South Wales), Flora Salim `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究将链式推理的隐藏状态视为动态系统，利用Recurrence Quantification Analysis (RQA) 对推理轨迹进行时序分析；

**💡 创新点**

创新点在于将非文本的RQA技术引入LLM推理过程，突破了传统以响应长度或文本质量为评估标准的局限，揭示推理复杂度与隐藏状态的几何动力学关联；

**🔧 技术方法**

主要技术包括隐藏状态提取、余弦距离阈值化构造递归矩阵、滑动窗口计算DET、LAM、ENTR等RQA指标，以及DFA与趋势斜率的时序特征提取；

**📊 数据集**

使用DeepSeek-R1-Distill-7B-Qwen在ZebraLogic逻辑推理基准上生成3600条推理轨迹，涵盖9种不同难度的格子配置；

**📈 对比分析**

与传统响应长度基线进行比较，RQA（尤其是时间窗滑动版本）在任务难度分类上提升约8%（36.94% vs. 28.97%），在答案正确性预测上与长度相近；

**⚠️ 局限性**

局限性包括计算开销较高、对RQA超参数敏感、仅验证单一模型与单一任务，未探究跨层、跨模型的通用性，且只基于最终层隐藏状态，忽视其他层的动力学信息。

---

## 76. AI-Limited Fluid Antenna-Aided Integrated Sensing and Communication Systems

**arXiv ID:** 2602.06247 | [PDF](https://arxiv.org/pdf/2602.06247v1)

**作者:** Farshad Rostami Ghadi `[一作]` (University of Granada), Christos Masouros `[通讯]` (University College London)

**通讯引用:** 17872 | [OpenAlex ID](https://openalex.org/A5030334551)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究在AI表示瓶颈下，采用流体天线系统（FAS）接收的ISAC（集成感知与通信）系统的基本性能极限；

**💡 创新点**

创新点在于：1）将AI编码器视为信息瓶颈并等价为加性表示噪声；2）提出FAS端口选择对AI瓶颈的补偿机制，并证明FAS长度决定了可获得的空间自由度；3）给出AI受限ISAC的容量-失真区域，并提供匹配的上下界；

**🔧 技术方法**

主要技术包括：信息瓶颈理论、Gaussian表示模型、FAS端口选择模型、Jakes相关模型、信息理论逆推、VIB（变分信息瓶颈）实现、数值仿真验证；

**📊 数据集**

未使用具体真实数据集，主要通过 Monte Carlo 仿真和解析分布得到的概率密度进行数值验证；

**📈 对比分析**

与传统 SISO、MIMO 方案以及理想无限 AI 容量基线进行对比，结果表明：FAS 在相同 AI 容量下可显著提升通信速率与感知 MSE，且随着 FAS 长度增大可逼近 AI 极限；

**⚠️ 局限性**

局限性包括：1）仅考虑单用户单接收端情景；2）假设 AI 编码器可实现 Gaussian 模型；3）端口选择策略以通信通道为准，未深入探讨感知导向的端口选择；4）实际硬件实现与能耗等工程细节未覆盖。

---

## 77. Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning

**arXiv ID:** 2602.06204 | [PDF](https://arxiv.org/pdf/2602.06204v1)

**作者:** Nan Chen `[一作]` (Johns Hopkins University), Soufiane Hayou `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Maximal-Update Adaptation (μA) 理论框架，用以解析LoRA微调中学习率与模型宽度、LoRA秩之间的缩放关系；通过理论推导给出不同初始化与缩放因子下的学习率缩放规则，并揭示一种LoRA配置可实现学习率从LoRA迁移到全微调（FFT）。

**💡 创新点**

创新点在于：①首次在LoRA的无限宽度和秩趋同极限下分析学习率动态，提出μA；②发现存在两种缩放规律（随秩递减或秩不变）；③证明在特定配置下LoRA学习率可直接迁移到FFT，实现超参搜索成本大幅降低。

**🔧 技术方法**

采用无偏初始化（随机A或B置零）、α 缩放因子（1、r⁻¹、r⁻¹⁄²）以及 SignSGD（简化的Adam）优化器，结合极限理论分析 LoRA 特征更新的三项贡献 δ₁,δ₂,δ₃，推导学习率缩放；实验使用统一的 finetuning recipe。

**📊 数据集**

实验涵盖多模态与多任务：语言模型（Llama‑3.2‑1B, Qwen‑2.5‑3B, Qwen‑3‑VL‑2B）、文本分类（RoBERTa‑large）、视觉模型（ViT‑Huge/14）、RL (Llama‑3.1‑8B on GSM8k) 与图像生成（Stable‑Diffusion‑v1.5 on Naruto‑BLIP‑Captions）。

**📈 对比分析**

通过在每种配置下对学习率进行对数网格搜索，比较最终训练损失/任务指标与 FFT 结果，验证 μA 缩放规则在各模型上都能准确预测最佳学习率；实验显示 LoRA 训练在内存占用上显著低于 FFT，且在多数任务中学习率迁移后性能差异不超过 5%。

**⚠️ 局限性**

局限性包括：①理论为渐进极限，实际模型尺寸有限时会有偏差；②仅给出了必要条件的学习率迁移，缺乏充分性证明；③实验主要集中在线性层和特定 LoRA 初始化，其他更复杂结构或初始化方式需进一步验证。

---

## 78. Computationally Efficient Laplacian CL-colME

**arXiv ID:** 2602.06070 | [PDF](https://arxiv.org/pdf/2602.06070v1)

**作者:** Nikola Stankovic `[一作]` `[通讯]`, Nikola Stankovic

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了一种基于拉普拉斯矩阵的去中心化协同均值估计方法CL-colME，用于大规模异构网络中在线均值估计。

**💡 创新点**

创新点在于将传统C-colME中的双随机化加权矩阵替换为拉普拉斯一致性更新，既保持了无偏估计与收敛到类oracle解的理论性质，又避免了每一步显式归一化的昂贵除法运算，从而显著提升计算效率。

**🔧 技术方法**

使用了分布式一致性算法、拉普拉斯矩阵的梯度下降迭代、在线置信区间自适应剪枝、随机邻居网络构造等技术，整体实现基于Python/NumPy的仿真框架。

**📊 数据集**

实验数据为人工生成的稀疏网络中5000个节点，分为两类均值分别为1.2和2的高斯噪声样本（σ=2），共10次随机重现。

**📈 对比分析**

与C-colME和局部估计、oracle估计进行对比；MSE随时间趋近相同，但CL-colME的计算时间约为722s，而C-colME为871s，提升约30%；在均值估计精度上与oracle解相当。

**⚠️ 局限性**

局限性包括：仅在模拟的高斯噪声场景下验证，未涉及真实网络的通信延迟和丢包；对拉普拉斯步长β的选择敏感；假设网络在分离后保持静态，实际环境可能需要动态重连；未探讨多类情形下的收敛速度差异。

---

## 79. Tempora: Characterising the Time-Contingent Utility of Online Test-Time Adaptation

**arXiv ID:** 2602.06136 | [PDF](https://arxiv.org/pdf/2602.06136v1)

**作者:** Sudarshan Sreeram `[一作]` (University of Cambridge), Cecilia Mascolo `[通讯]` (University of Cambridge)

**通讯引用:** 18468 | [OpenAlex ID](https://openalex.org/A5010623957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Tempora 框架，用于在不同时间压力下评估测试时适应（TTA）方法的实用性；

**💡 创新点**

创新点在于将 TTA 评价从传统的无时间限制转为三种时间相关的度量（离散、连续、摊销），并揭示了在时间压力下方法排名不稳定的现象；

**🔧 技术方法**

使用了时间相关的效用度量、可编程的评估协议、以及标准的梯度与非梯度 TTA 技术（AdaBN、LAME、NEO、Tent、ETA、SHOT‑IM、SAR）进行比较；

**📊 数据集**

使用 ImageNet‑C（15 种噪声/模糊/天气/数字污染）与预训练 ResNet‑50 作为基准；

**📈 对比分析**

在 240 次不同时间场景下比较，结果显示传统的离线排名无法预测时间受限情形下的最佳方法；在离散场景中 AdaBN 常常胜过 ETA，连续场景中 AdaBN/SHOT‑IM 较优，摊销场景中 SHOT‑IM 在预算紧张时表现最优；

**⚠️ 局限性**

局限性包括：仅评估图像分类任务与单一网络架构；排除多前向/包装类方法；仅在 IID 的 ImageNet‑C 上测试，未覆盖持续非 IID 流；缺乏针对不同硬件/应用的更细粒度评估。

---

## 80. FlowConsist: Make Your Flow Consistent with Real Trajectory

**arXiv ID:** 2602.06346 | [PDF](https://arxiv.org/pdf/2602.06346v1)

**作者:** Tianyi Zhang `[一作]` (Nankai University), Peng-Tao Jiang `[通讯]` (vivo Mobile Communication Co. Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为FlowConsist的训练框架，用于解决fast flow模型中的轨迹漂移和误差累积问题，实现单步或少步生成的高质量图像生成；

**💡 创新点**

创新点在于①用模型自身预测的边际速度替代随机条件速度，从而消除轨迹漂移；②引入轨迹校正策略，将模型生成样本的边际分布与真实分布对齐，修正随时间累积的误差；

**🔧 技术方法**

采用流匹配理论、边际速度对齐（类似DMD）、CFG指导以及自监督的边际速度预测网络，配合DiT/SiT网络结构；

**📊 数据集**

在ImageNet 256×256数据集上进行实验；

**📈 对比分析**

与现有fast flow和多步扩散/流模型对比，FlowConsist在1 NFE设置下获得FID 1.52，优于之前最佳1.72；

**⚠️ 局限性**

局限性包括：对模型结构和训练细节（如CFG比例、损失比例）较为敏感，且在更高分辨率或不同任务上需进一步验证。

---

## 81. Zero-Trust Runtime Verification for Agentic Payment Protocols: Mitigating Replay and Context-Binding Failures in AP2

**arXiv ID:** 2602.06345 | [PDF](https://arxiv.org/pdf/2602.06345v1)

**作者:** Qianlong Lan `[一作]` (eBay Inc), Stephanie Westrum `[通讯]` (eBay Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种针对AP2协议在自主代理支付执行中出现的重放和上下文偏移攻击的零信任运行时验证框架，解决了传统协议在高并发、异步环境下的授权执行缺陷。

**💡 创新点**

将上下文绑定与一次性消费（consume‑once）两种机制结合，并通过动态、时间限制的 nonce 注册表实现了对授权令牌的强制一次使用与上下文一致性检查。

**🔧 技术方法**

使用数字签名验证、SHA256 计算上下文哈希、时间窗口内的 nonce 注册表、Python实现的验证服务以及分布式锁（如 Redis SETNX）等技术。

**📊 数据集**

使用模拟的代理工作负载生成的交易请求（约 5,000–10,000 TPS）来评估安全性与性能，而非公开真实支付数据集。

**📈 对比分析**

与仅做 AP2 签名与过期校验的基线相比，实验在 10,000 TPS 下平均验证延迟约 3.8 ms，捕获率 100% 且无误报，验证了方案在高并发场景下的有效性与低开销。

**⚠️ 局限性**

仅针对授权层攻击，无法防止代理被操纵获取合法令牌后进行恶意操作；依赖时钟同步与底层加密基础，若存在关键基础设施失效，防御效果受限。

---

## 82. EgoAVU: Egocentric Audio-Visual Understanding

**arXiv ID:** 2602.06139 | [PDF](https://arxiv.org/pdf/2602.06139v1)

**作者:** Ashish Seth `[一作]` (Meta), Zhipeng Cai `[通讯]` (University of Maryland, College Park)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套自动化数据引擎，利用开源多模态大型语言模型（MLLM）对Ego4D视频进行语音与视觉信息的增强、过滤、统一描述，生成了大规模的音视QA数据集（EgoAV）和对应评测基准。

**💡 创新点**

创新点包括：①模块化处理，分别使用单模态MLLM提取视觉与音频细节，避免模态偏差；②构建多模态上下文图（MCG）显式建模动作、物体与声音之间的关系；③生成统一的音视叙述后自动生成开放式和闭合式QA，形成多任务评测；④通过该数据集显著提升MLLM在第一人称音视推理中的性能，首次量化展示其视觉偏差。

**🔧 技术方法**

使用了开源MLLM（Qwen2.5-Omni、Qwen2.5-VL、LLaMA-70B等）、图结构解析、MATTR词汇多样性过滤、LoRA及全量微调、正则表达式匹配与LLM-as-a-judge评判等技术。

**📊 数据集**

主要使用Ego4D作为原始视频源；构建训练集EgoAV-Train（9k视频，约3M QA）和评测集EgoAV-Dev（900视频，3K QA），并在EgoTempo、EgoSchema、EgoIllusion等现有 egocentric 基准以及 VideoMME、AVQA 等 exocentric 基准进行交叉评测。

**📈 对比分析**

在EgoAV上对比了多种基线MLLM（Qwen2.5-Omni、VideoLLaMA2、MiniCPM-o 等），发现它们在音频-视觉关联、时间推理、生成叙述等任务中表现低下；对EgoAV进行LoRA或全量微调后，开放式任务提升可达113%（相对），闭合式任务提升约44%；在EgoTempo/EgoIllusion 上提升约28%，并在 exocentric 任务上保持甚至略有提升。

**⚠️ 局限性**

局限性：数据集仍受开源MLLM生成噪声影响，音频识别仍不够鲁棒；生成的叙述偶尔存在错误或缺失；模型在极端音频场景和多模态共现的细粒度推理中仍易产生幻觉，需进一步改进多模态融合与校正机制。

---

## 83. GRP-Obliteration: Unaligning LLMs With a Single Unlabeled Prompt

**arXiv ID:** 2602.06258 | [PDF](https://arxiv.org/pdf/2602.06258v1)

**作者:** Mark Russinovich `[一作]` (Microsoft), Ahmed Salem `[通讯]` (Microsoft)

**通讯引用:** 470 | [OpenAlex ID](https://openalex.org/A5103045221)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用单一未标记提示通过GRPO直接去除LLM和扩散模型的安全对齐的方法；

**💡 创新点**

创新点在于只需一个普通提示即可实现强大且一致的去对齐，同时保留大部分模型功能，并将GRPO与Judge奖励结合，扩展到多种模型家族及图像生成模型；

**🔧 技术方法**

主要技术包括Group Relative Policy Optimization (GRPO) 与 DAPO 损失、基于Judge的多维奖励（意图、风险、细节）以及对齐模型的KL锚定；

**📊 数据集**

训练使用单一提示或少量安全基准数据（AdvBench、StrongREJECT），评估涵盖五个安全基准（StrongREJECT、Sorry-Bench、JailbreakBench、HarmBench、AdvBench）和六个功能基准（MMLU、HellaSwag、WinoGrande、GSM8K、TruthfulQA、IFEval），扩散模型评估使用T2ISafety和MS‑COCO；

**📈 对比分析**

与TwinBreak、RefusalDirection等基线相比，GRP‑Oblit在攻击成功率(ASR)上显著提升，功能得分保持≈95%以上，整体得分最高；单提示版本同样优于基线且方差更小；

**⚠️ 局限性**

局限性包括对扩散模型的跨领域转移效果不如文本模型，需依赖已有安全对齐机制，未公开代码导致复现困难，且在极端安全场景下仍可能不足以完全消除风险。

---

## 84. MPIB: A Benchmark for Medical Prompt Injection Attacks and Clinical Safety in LLMs

**arXiv ID:** 2602.06268 | [PDF](https://arxiv.org/pdf/2602.06268v1)

**作者:** Junhyeok Lee `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University)

**通讯引用:** 3628 | [OpenAlex ID](https://openalex.org/A5052023515)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了MPIB benchmark，用于评估医疗LLM和RAG系统在面对直接和间接prompt注入攻击时的临床安全性。

**💡 创新点**

创新点在于引入基于临床结果的指标CHER并与ASR并列使用，构建了多阶段质量控制的9,697例医疗场景注入数据，并针对RAG中被污染检索上下文的间接注入提供安全评估框架。

**🔧 技术方法**

使用了LLM‑as‑a‑judge结构化评估、攻击合成规则、分层威胁模型（V1/V2）、多门限质量门控与医学风险分类、以及五种防御策略（内部硬化、输入重写、上下文净化、组合策略）和多模型矩阵实验。

**📊 数据集**

以MedQA、PubMedQA为原始数据，经过标准化、情景标注、攻击合成后形成MPIB 9,697实例，涵盖四大情景S1–S4与两类攻击向量V1/V2，数据已公开到Hugging Face。

**📈 对比分析**

通过ASR（Severity≥2）、CHER_3（Severity≥3）和FPR‑H等指标，对12个模型和5种防御配置进行评估；结果显示V2攻击的CHER远高于V1，ASR与CHER可显著偏离，防御效果因攻击向量和模型差异而异，综合防御D4并非总是最佳。

**⚠️ 局限性**

主要局限在于评估依赖LLM‑as‑a‑judge，可能产生判定偏差；H1–H5与0–4级别难以完全覆盖所有临床情景；防御方案轻量化，未达到生产级；红屏数据仍需受控发布，可能限制外部复现。

---

## 85. DeDPO: Debiased Direct Preference Optimization for Diffusion Models

**arXiv ID:** 2602.06195 | [PDF](https://arxiv.org/pdf/2602.06195v1)

**作者:** Khiem Pham `[一作]` (Cornell University), Ramin Zabih `[通讯]` (Cornell University)

**通讯引用:** 24792 | [OpenAlex ID](https://openalex.org/A5042850991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Debiased DPO（DeDPO），一种在半监督环境下使用去偏估计器的直接偏好优化方法，能够在仅有少量人工标注的前提下利用大量噪声的人工智能合成偏好标签进行扩展；

**💡 创新点**

创新点在于将因果推断中的双重稳健（doubly‑robust）去偏技术直接嵌入DPO目标，既消除合成标签的系统偏差，又在标签噪声较大时保持学习稳定；

**🔧 技术方法**

采用去偏损失（Debiased Loss）、半监督DPO、合成偏好生成（自训练或预训练视觉‑语言模型如Qwen）、二分类重构的DPO目标以及标准的扩散模型训练技术；

**📊 数据集**

主要使用FiFA（5K人工标注的图像对）和HPDv2数据集，并通过预训练VLM Qwen 生成合成偏好，评估时用PartiPrompt与HPSv2提示集；

**📈 对比分析**

与SFT、传统Diffusion‑DPO（含100%人工标签、25%人工+75%合成标签）对比，DeDPO在25%人工+75%合成的设置下，PickScore、HPSv2和Aesthetic Score均与完全人工标签的基线持平甚至略优，且在所有模型与数据集上均优于仅使用合成标签的DPO；

**⚠️ 局限性**

局限性包括：对合成偏好模型的质量高度依赖；假设合成模型与训练数据独立，若自训练时共享数据易导致过拟合；需仍保留一定人工标签比例；在极大规模扩散模型或多模态任务中的表现尚未全面验证。

---

## 86. Unsupervised MRI-US Multimodal Image Registration with Multilevel Correlation Pyramidal Optimization

**arXiv ID:** 2602.06288 | [PDF](https://arxiv.org/pdf/2602.06288v1)

**作者:** Jiazheng Wang `[一作]` (Hunan University), Hang Zhang `[通讯]` (Cornell University)

**通讯引用:** 20318 | [OpenAlex ID](https://openalex.org/A5100414978)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种无监督多模态医学图像配准方法——多层相关金字塔优化（MCPO），实现预手术MRI与术中超声的快速精准配准。

**💡 创新点**

创新点包括：① 多层金字塔融合优化机制，实现全局粗略匹配与局部细节补偿；② 加权平衡耦合凸优化与逆一致性约束，提升变形平滑度与稳定性；③ 随机局部互信息损失，克服超声低信息密度问题；④ 可选的Adam实例优化进一步细化位移场。

**🔧 技术方法**

采用VoxelOpt与ConvexAdam框架，Mind‑SSC特征提取器，稠密相关层（SSD）生成代价体，耦合凸优化层，逆一致性约束层，Affine拟合层，以及Patch‑MI损失。

**📊 数据集**

使用ReMIND2Reg子挑战数据（99例训练、5例验证）和Resect数据集（23例低级别胶质瘤）进行评估。

**📈 对比分析**

与ConvexAdam‑Rigid、NiftyReg、MCBO等基线对比，MCPO在ReMIND2Reg验证集TRE为1.790±0.536 mm，测试集平均得分0.911（排名第一），在Resect数据集MCPO‑deform的平均TRE为1.798±1.301 mm，均显著优于基线。

**⚠️ 局限性**

局限性：在极大变形或特定病例中仍可能出现性能下降；受限于验证集样本量较少，易出现过拟合；计算成本仍高，需GPU加速。

---

## 87. Rethinking External Communication of Autonomous Vehicles: Is the Field Converging, Diverging, or Stalling?

**arXiv ID:** 2602.06278 | [PDF](https://arxiv.org/pdf/2602.06278v1)

**作者:** Tram Thi Minh Tran `[一作]` (University of Sydney), Martin Tomitsch `[通讯]` (University of Technology Sydney)

**通讯引用:** 2963 | [OpenAlex ID](https://openalex.org/A5023076293)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2014–2025年间620篇关于自动驾驶车辆外部人机接口（eHMI）的学术论文、行业专利、概念与部署案例以及相关监管文件进行系统梳理与多维度编码，构建研究进展与实践对齐的“过滤漏斗”模型，评估该领域的趋同、分歧与停滞。

**💡 创新点**

创新点在于首次将学术研究、产业实践和标准化进程三者放在同一时间轴上并进行横向对照；提出eHMI研究的三维取向（设计探询、现场调查、场域整合）和基于实证与规范的过滤漏斗框架，厘清安全优先的核心共识与持续争议；为后续标准制定与设计路线图提供可操作的层级化建议。

**🔧 技术方法**

主要采用系统综述与手工编码（Petal工具辅助）相结合的研究方法，辅以文献计量学（共引、主题建模）和案例对比分析，形成结构化的多维数据集。

**📊 数据集**

数据集包括：620篇学术论文（期刊、会议、工作坊）；行业专利与概念（约70项）；已部署的eHMI系统案例（10+车辆、4+配送机器人等）；以及来自UNECE、ISO、SAE、各国交通法规的标准与监管文件。

**📈 对比分析**

通过将研究主题分层、对比行业实施与监管标准，揭示研究热点（如视觉与信息性信号）与分歧点（是否需要显式eHMI、模态选择）。虽然未给出传统意义上的数值性能指标，但研究显示：安全导向的简单视觉提示在多场景得到共识；而多模态、情感化或指令式eHMI在实验与部署中表现不一，导致方法与效果存在明显差异。

**⚠️ 局限性**

局限包括：仅覆盖英文同行评审出版物，可能遗漏中文与灰色文献；编码依赖标题/摘要，可能忽略全文细节；缺乏大规模真实道路验证与纵向长期研究；以及对行业部署与标准制定的时间延迟导致对最新实践的滞后捕捉。

---

## 88. Rethinking Memory Mechanisms of Foundation Agents in the Second Half

**arXiv ID:** 2602.06052 | [PDF](https://arxiv.org/pdf/2602.06052v1)

**作者:** Wei-Chieh Huang `[一作]` (University of Illinois Chicago), Kai Shu `[通讯]` (Emory University)

**通讯引用:** 11673 | [OpenAlex ID](https://openalex.org/A5058670321)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了2023‑2025年基金代理记忆研究，构建了三维分类框架（存储介质、认知机制、主体），并系统分析了单/多代理的内存操作、学习策略、评估指标与挑战。

**💡 创新点**

提出统一的三维内存分类体系，将内存与代理系统设计紧密关联，首次细化多代理内存路由与隔离机制，并系统梳理评估标准与开放挑战。

**🔧 技术方法**

文献综述+主题归纳与图表可视化，结合向量检索、KV 缓存、结构化存储、强化学习、prompt 等已公开技术进行案例解析。

**📊 数据集**

未使用传统数据集，依托对 218 篇 2023‑2025 年期刊与会议论文的系统检索与整理。

**📈 对比分析**

通过对现有工作在内存架构、操作机制、学习策略、评估方法等维度的对比，呈现各类方法的优势与局限，未给出统一实验指标，但为后续实验提供了参考基准。

**⚠️ 局限性**

缺乏统一评估框架导致不同方法难以直接比较；对多模态、跨任务迁移、隐私与安全等实际应用场景的深入讨论仍显不足。

---

## 89. Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in Inference-Time Interventions

**arXiv ID:** 2602.06256 | [PDF](https://arxiv.org/pdf/2602.06256v1)

**作者:** Navita Goyal `[一作]` (University of Maryland), Hal Daumé `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型的推理时干预方法进行可解释性评估，提出三维特异性框架并在过度拒绝和事实幻觉两个安全关键场景上系统评测。

**💡 创新点**

创新点在于将特异性分为一般、控制、鲁棒三维度，首次量化并验证推理时干预在安全对抗攻击下的鲁棒性缺失。

**🔧 技术方法**

采用差分均值、线性探针、监督干预向量、表示微调、部分正交化等五种干预方法，并结合隐层向量投影与扰动。

**📊 数据集**

使用PHTest、JailbreakBench、Alpaca、NQSwap等数据集来训练和评估干预向量，覆盖伪有害、真实有害与无害查询。

**📈 对比分析**

通过对比不同模型（Llama、Qwen、Gemma）和方法，发现虽然大多数方法提升合规率，但鲁棒特异性显著下降，PartialOR在鲁棒性上表现最平衡。

**⚠️ 局限性**

限制在单任务设置、模型规模不大、仅开放模型可访问、缺乏对新攻击的泛化等方面。

---

## 90. FlashSketch: Sketch-Kernel Co-Design for Fast Sparse Sketching on GPUs

**arXiv ID:** 2602.06071 | [PDF](https://arxiv.org/pdf/2602.06071v1)

**作者:** Rajat Vadiraj Dwaraknath `[一作]` (Stanford University), Mert Pilanci `[通讯]` (Stanford University)

**通讯引用:** 1159 | [OpenAlex ID](https://openalex.org/A5001436196)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种结构化稀疏 Sketch 并实现对应高效 CUDA 核心，在 GPU 上加速稀疏 Johnson–Lindenstrauss 变换。

**💡 创新点**

通过“块级随机置换”将稀疏 Sketch 结构化为可并行、无全局原子操作的形式，并引入可调参数 κ 实现 sketch 质量与速度的可调权衡。

**🔧 技术方法**

使用块结构化 SJLT、稀疏 JL、联合置换、共享内存原子、基于 CUDA 的线程块分块核、随机置换生成以及随机稀疏投影等技术。

**📊 数据集**

评测了标准 RandNLA 基准（Gaussian、低秩+噪声、SuiteSparse、LLM 权重）以及数据归因管道 GraSS（MNIST、GPT2‑medium、Qwen2‑1.5B）。

**📈 对比分析**

与密集高斯投影、cuSPARSE SpMM、原始 GraSS SJLT 核、SRHT 等基线对比，得到在 RTX 4090/RTX A6000 GPU 上约 2‑3.5 倍速度提升，同时保持或提升语义保真度（Gram 矩阵误差、LDS 分数）。

**⚠️ 局限性**

适用性受限于 dense 数据且 k ≪ d 时占用率低，参数 κ 需人工调节，理论仅针对独立随机置换，未分析依赖置换，且仅给出 OSE 保障，未覆盖所有下游任务。

---

## 91. SCONE: A Practical, Constraint-Aware Plug-in for Latent Encoding in Learned DNA Storage

**arXiv ID:** 2602.06157 | [PDF](https://arxiv.org/pdf/2602.06157v1)

**作者:** Cihan Ruan `[一作]` (Santa Clara University), Nam Ling `[通讯]` (Santa Clara University)

**通讯引用:** 3237 | [OpenAlex ID](https://openalex.org/A5018686979)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为SCONE的插件式模块，可在学习型压缩的潜在空间直接进行四进制算术编码，并在编码过程中实时满足DNA合成的GC平衡与同一碱基重复长度限制，实现无后期纠错或修正步骤的可逆编码。

**💡 创新点**

创新点在于：①把生物化学约束嵌入到熵编码层的概率模型中，使用有限状态机动态掩码保证GC比例与同一碱基限制；②采用四进制算术编码直接映射到DNA碱基，避免了传统二进制→四进制映射导致的冗余与信息损失；③模块与任何基于超先验的学习压缩框架无缝兼容，支持端到端微调。

**🔧 技术方法**

核心技术包括：四进制算术编码、有限状态机（FSM）约束引导、超先验高斯条件概率模型、可逆的概率重归与掩码机制，以及32位定点实现的高效算术运算。

**📊 数据集**

实验使用了5000条长度为100符号、均匀分布的随机四进制序列进行评估，并与多种传统DNA编码方案（Church, Goldman, DNA Fountain, Yin-Yang等）做对比。

**📈 对比分析**

与传统方案相比，SCONE在保持约99.7% GC平衡和100%同一碱基长度≤3的同时，实现了1.86 bpn的编码密度；相较于无FSM基线，GC标准差从0.050降至0.012，最大同一碱基长度从8+降至3，编码与解码延迟仅为0.60/0.72 ms，100%成功率。

**⚠️ 局限性**

局限性包括：①尚未加入外部错误更正层，对测序误码仍不鲁棒；②对超先验模型的假设较强，若潜在分布不佳可能影响熵编码效果；③主要在模拟数据上验证，实际生物合成与测序实验尚待进一步验证。

---

## 92. Recontextualizing Famous Quotes for Brand Slogan Generation

**arXiv ID:** 2602.06049 | [PDF](https://arxiv.org/pdf/2602.06049v1)

**作者:** Ziao Yang `[一作]` (Brandeis University), Hongfu Liu `[通讯]` (Brandeis University)

**通讯引用:** 3679 | [OpenAlex ID](https://openalex.org/A5086089915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通过重新语境化名人名言来生成品牌口号的方法。

**💡 创新点**

创新点在于将名言与品牌结合，拆解生成过程为可控子任务，既保持熟悉性又提升新颖度。

**🔧 技术方法**

利用LLM（如Qwen-32B）进行后期训练，配合结构拆解、词汇替换和混剪步骤实现。

**📊 数据集**

使用短文本语料（名言、俚语、歌词、梗）以及公开口号、电影对白、名言数据集进行微调。

**📈 对比分析**

与GPT‑4o、DS‑L、DS‑Q三大LLM基线对比，实验显示在多项多样性、创新性和情感冲击指标上均优于基线；在人类评测中在“首要吸引”略有优势，人物契合度略逊。

**⚠️ 局限性**

局限包括对人物契合度仍未达到最佳，方法仍依赖LLM子任务，评测规模有限，缺乏跨语境或多语言验证。

---

## 93. Optimistic Training and Convergence of Q-Learning -- Extended Version

**arXiv ID:** 2602.06146 | [PDF](https://arxiv.org/pdf/2602.06146v1)

**作者:** Prashant Mehta `[一作]` (University of Illinois Urbana-Champaign), Sean Meyn `[通讯]` (University of Florida)

**通讯引用:** 19237 | [OpenAlex ID](https://openalex.org/A5047988825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究了 Q‑学习在线性函数逼近框架下的稳定性与收敛性，给出了使用无偏或基于 tamed Gibbs 的训练策略时的理论分析与数值反例。

**💡 创新点**

提出 tamed Gibbs 策略在保证参数界定（最终有界）方面的必要性，并证明在二维参数化下，即使真实 Q 在基空间内，投影 Bellman 方程仍可能出现多重解，导致 Q‑学习无法保证收敛；同时给出了可导致无解或不稳定的单维反例。

**🔧 技术方法**

使用 ODE 方法、Jacobian Hurwitz 判定、隐式函数定理以及对数期望的分析，并在实验中实现了基于马尔可夫决策过程的数值仿真。

**📊 数据集**

使用人工构造的有限状态 MDP（如 2×2 状态×动作网格或高斯状态演化）作为实验环境；未使用公开数据集。

**📈 对比分析**

通过与传统贪婪（或无偏）Q‑学习的对比，展示 tamed Gibbs 在参数稳定性方面的优势，但同时在不同初值下仍可能收敛到不同解，说明收敛性并未得到完整保证；数值结果显示存在多重收敛点。

**⚠️ 局限性**

局限性：仅针对有限状态、线性函数逼近和已知基函数的情形；理论假设（如唯一不变分布、矩阵满秩等）较强；对连续或高维状态空间、非线性基函数的推广尚未覆盖；实际应用中仍需经验调参以避免多解或不收敛。

---

## 94. The proof theory and semantics of second-order (intuitionistic) tense logic

**arXiv ID:** 2602.06253 | [PDF](https://arxiv.org/pdf/2602.06253v1)

**作者:** Justus Becker `[一作]`, Paaras Padhiar `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出并系统化了第二阶（特别是负向）时态逻辑的公理化与标记化证明框架，并在二阶模态无关子系统中完成了可推理性与完备性证明；

**💡 创新点**

创新点在于把负向公式语法与二阶量化公理化相结合，构造了模态无关的可证明性体系，并通过标签化序列与时态逻辑的直接解释实现了从标记化到公理化的互译；

**🔧 技术方法**

主要技术包括二阶逻辑的公理化与归纳证明、Beth/Kripke 模型构造、负向翻译、标签化序列与多前缀的推理步骤；

**📊 数据集**

无，本文为纯理论证明，没有使用实验数据集；

**📈 对比分析**

通过与已有的一阶公理化系统和经典时态逻辑比较，证明了在对应模型（Beth/Kripke）上的完备性，并在单前缀、复前缀两种系统中实现了可推理性（无切点）与完全性；

**⚠️ 局限性**

局限性包括：对黑色模态的处理不完整，二阶逻辑的非解析性导致证明中仍需出现非目标模态符号，且目前仅覆盖负向公式子系统，未对全模态逻辑的非黑色子系统给出完整的切点可去性结果。

---

## 95. Exposing Weaknesses of Large Reasoning Models through Graph Algorithm Problems

**arXiv ID:** 2602.06319 | [PDF](https://arxiv.org/pdf/2602.06319v1)

**作者:** Qifan Zhang `[一作]` (Hong Kong University of Science and Technology), Jia Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 15040 | [OpenAlex ID](https://openalex.org/A5108050433)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GrAlgoBench基准，通过图算法问题评估大型推理模型的长上下文推理能力和自我验证行为

**💡 创新点**

创新点在于将图算法作为多维、可扩展的推理任务，实现长上下文、细粒度难度控制，并揭示LRM在长上下文失误和过度自检的两大弱点

**🔧 技术方法**

采用LLM-as-judge分类、自动错误分析、entropy分割、自我验证评估等技术，结合程序化评测

**📊 数据集**

构建了2700个多尺度（8-160节点）图实例，来源于DBLP、Street Network、OpenFlight、Wikipedia、DBpedia，按Enumeration、Exploration、Intuition三类划分

**📈 对比分析**

对比了多款LRM（Qwen3-32B、Qwen3-235B、GPT-OSS、DeepSeek、Gemini等）在pass@k、cons@k、Z-score等指标上评测，发现性能随图规模增长显著下降，最大节点160时准确率低于50%

**⚠️ 局限性**

主要局限是长上下文推理能力不足和频繁无效自我验证导致过度推理，且未对外部工具调用的影响进行完整评估

---

## 96. PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models

**arXiv ID:** 2602.06053 | [PDF](https://arxiv.org/pdf/2602.06053v1)

**作者:** Rajarshi Roy `[一作]`, Bryan Catanzaro `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PersonaPlex 全双工语音对话模型，支持角色与语音双重调控。

**💡 创新点**

创新点在于混合文本+音频系统提示实现零拷贝语音克隆与细粒度角色调控。

**🔧 技术方法**

技术包括基于 Moshi 的双向 Transformer 架构、Hybrid System Prompt、LLM 生成合成对话、TTS 语音克隆、训练时系统提示掩蔽与损失权重调整等。

**📊 数据集**

数据集为大规模合成对话（1840 小时客服 + 410 小时问答），使用 Qwen‑3‑32B、GPT‑OSS‑120B 生成文本，结合 26,296 条 VoxCeleb、LibriTTS 等语音样本，扩展的 Service‑Duplex‑Bench 用于评估。

**📈 对比分析**

与现有全双工模型（Moshi、Gemini、Qwen‑2.5‑Omni 等）以及非双工 LLM 语音系统对比，PersonaPlex 在 Full‑Duplex‑Bench 与 Service‑Duplex‑Bench 上在角色遵从、声纹相似度、对话自然度与延迟等指标均优于基线。

**⚠️ 局限性**

局限：仍需大量合成数据支持，极端口音或嘈杂环境下语音克隆效果下降，缺乏真实用户交互的长期评测。

---

## 97. HQP: Sensitivity-Aware Hybrid Quantization and Pruning for Ultra-Low-Latency Edge AI Inference

**arXiv ID:** 2602.06069 | [PDF](https://arxiv.org/pdf/2602.06069v1)

**作者:** Dinesh Gopalan `[一作]` (Advanced Micro Devices), Ratul Ali `[通讯]` (Jahangirnagar University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种融合感知式结构剪枝和后训练量化的Hybrid Quantization and Pruning (HQP) 框架，能够在保证精度下降不超过 1.5% 的前提下，对 CNN 进行稀疏化和 INT8 量化，实现边缘 AI 的超低延迟推理。

**💡 创新点**

创新点在于：
1) 采用 Fisher 信息矩阵（FIM）的高效对角近似作为滤波器重要性度量，实现全局二阶敏感性评估；
2) 通过动态的条件剪枝循环（基于精度阈值）保证稀疏化后模型仍可被稳定量化；
3) 该框架将剪枝与量化紧密耦合，打破传统顺序串行的瓶颈，实现显著的协同加速。

**🔧 技术方法**

使用的技术包括：
- 结构化剪枝（按滤波器/通道）
- FIM‑近似敏感度 S 计算（一次反向传播）
- 条件迭代剪枝算法（算法 1）
- TensorRT 进行 INT8 量化与推理加速（KL‑Divergence 校准）
- 传统的 32‑bit FP 推理基准。

**📊 数据集**

数据集：ImageNet；使用 5,000 张图像做校准（D_calib），5,000 张图像做验证（D_val），完整 ImageNet‑1000 验证集评估精度。

**📈 对比分析**

与基线 FP32、单独量化（Q8）和单独剪枝（P50）比较：
- 在 Jetson Xavier NX 上 MobileNetV3，HQP 速度提升 3.12×，模型大小压缩 55%，精度下降 1.4%；
- 在 ResNet‑18，HQP 速度提升 2.51×，大小压缩 40%，精度下降 1.3%；
- 量化仅方法在 ResNet‑18 上失效（精度下降 1.9% > 1.5%）。

**⚠️ 局限性**

局限性：
- 仅在 NVIDIA Jetson 设备上验证，需进一步跨平台验证；
- 目前仅支持统一 INT8 量化，未探索混合精度或更低位宽；
- 只针对 CNN，未对 Transformer 等新型架构做实验；
- 剪枝阈值和 FIM 近似的超参数需手动设定，适配性需提升。

---

## 98. Canzona: A Unified, Asynchronous, and Load-Balanced Framework for Distributed Matrix-based Optimizers

**arXiv ID:** 2602.06079 | [PDF](https://arxiv.org/pdf/2602.06079v1)

**作者:** Liangyu Wang `[一作]` (King Abdullah University of Science and Technology), Dayiheng Liu `[通讯]` (Alibaba Group)

**通讯引用:** 1619 | [OpenAlex ID](https://openalex.org/A5062188134)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种统一的、异步的、负载均衡的框架Canzona，用于分布式矩阵优化器，解决了矩阵优化器与现代并行训练策略之间的根本冲突。

**💡 创新点**

创新点在于引入了α-平衡静态分区策略和异步微组调度，成功调和了严格的原子性要求与大规模并行性。

**🔧 技术方法**

使用了α-平衡贪婪LPT算法和微组调度算法，分别用于数据并行和张量并行的负载均衡。

**📊 数据集**

在256个GPU上对Qwen3模型系列（参数范围从1.7B到32B）进行了评估。

**📈 对比分析**

与现有的同步计算和NVIDIA的层级分配方法相比，Canzona在端到端迭代时间上实现了1.57倍的加速，并将优化器步骤延迟减少了5.8倍，显著提高了效率。

**⚠️ 局限性**

限制在于尽管提出的方法有效解决了负载不均衡和通信开销问题，但在某些情况下仍可能面临计算复杂性和资源分配的挑战。

---

## 99. Toward Faithful and Complete Answer Construction from a Single Document

**arXiv ID:** 2602.06103 | [PDF](https://arxiv.org/pdf/2602.06103v1)

**作者:** Zhaoyang Chen `[一作]` (Iowa State University), Cody Fleming `[通讯]` (Iowa State University)

**通讯引用:** 891 | [OpenAlex ID](https://openalex.org/A5113457201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了EVE框架，用多次独立抽取、验证与枚举三阶段实现对单文件信息的完整、可信回答

**💡 创新点**

通过结构化的多查询抽取与投票验证显著降低遗漏与幻觉，实现置信度可控的生成，打破单轮生成的覆盖-准确权衡

**🔧 技术方法**

利用大语言模型多次提问、投票判定、结构化枚举；配合提示工程、投票与序列化生成

**📊 数据集**

在STPA（系统理论过程分析）任务上构建了首个基于100篇专利文档的结构化数据集

**📈 对比分析**

与传统单查询生成对比，EVE在四次抽取+四次验证配置下召回率提升约24%、精确率提升约29%，F1提升约31%，在不同模型（Claude、Gemini、GPT‑4o、Qwen）上均表现出色

**⚠️ 局限性**

受限于自然语言的歧义与模型推断的概率性，最高召回率约92%、精确率约74%，仍无法满足极端安全关键场景的严格可靠性；需要外部形式化验证或符号推理辅助

---

## 100. DataCrumb: A Physical Probe for Reflections on Background Web Tracking

**arXiv ID:** 2602.06177 | [PDF](https://arxiv.org/pdf/2602.06177v1)

**作者:** Sujay Shalawadi `[一作]` (Norwegian University of Science and Technology), Florian Echtler `[通讯]` (Aalborg University)

**通讯引用:** 1641 | [OpenAlex ID](https://openalex.org/A5059378596)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过在家庭网络中部署一台配备LED灯、蜂鸣器和小屏幕的物理探针（DataCrumb），实时将DNS层的跟踪请求转换为视觉、声音和数值反馈，以让用户在日常生活中感知后台的Web跟踪行为。

**💡 创新点**

创新点在于：①以可触摸的、无交互的物理装置来感知并可视化隐蔽的网络跟踪；②采用模糊、持续的感官信号（灯光闪烁、短促噪音、累积计数）而非直接说明，从而促使用户在情境中反思隐私与控制之间的矛盾；③将物理探针嵌入家庭环境，借助反复触发的低强度反馈，突破传统Cookie弹窗“疲劳”与“不可见性”的双重壁垒。

**🔧 技术方法**

技术实现主要包括：Raspberry Pi 4 B + Pi‑hole 负责拦截和记录DNS请求；自制Python脚本实时解析日志并通过GPIO驱动LED灯、蜂鸣器和小型OLED显示屏；LED阵列映射主流跟踪域（如Google、TikTok等），蜂鸣器区分单次与多次拦截，显示屏持续累积请求与拦截计数。

**📊 数据集**

数据来源为三户学生家庭的网络流量日志，部署周期为三天。日志记录了每个DNS查询、拦截事件以及设备的运行状态；共计约75‑120次拦截事件（每户8‑15次/天），用于定性访谈分析和后续反馈阐释。

**📈 对比分析**

由于研究重点是设计与反思，而非算法性能，论文未使用传统基准数据集或客观度量指标。比较主要是与现有的可视化/触感隐私工具（DataSlip、VoxMox等）在“是否能让用户感知”与“是否引发反思”两维度的对照；在实验期间，DataCrumb 的持续触发率（≈10/天）与访谈中报告的“易被忽略”程度相匹配，说明其在“隐蔽性打破”方面具有可观成效。

**⚠️ 局限性**

局限性包括：①仅在三户学生家庭内短期（三天）部署，样本规模小且缺乏多样性；②探针仅监测DNS层，无法识别更细粒度的Cookie/浏览器跟踪，导致反馈模糊性高；③未提供用户可调节的反馈强度或解释功能，可能导致部分用户产生疲劳或厌倦；④未系统收集量化的使用数据或长期行为变化，无法评估长期效果与持久性。

---

## 101. The Avatar Cache: Enabling On-Demand Security with Morphable Cache Architecture

**arXiv ID:** 2602.06433 | [PDF](https://arxiv.org/pdf/2602.06433v1)

**作者:** Anubhav Bhatla `[一作]` (Massachusetts Institute of Technology), Biswabandan Panda `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 282 | [OpenAlex ID](https://openalex.org/A5101503372)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Avatar Cache，一种可变形的最后一级缓存，支持三种运行模式：普通模式、随机安全模式和分区安全模式，实现按需安全。

**💡 创新点**

创新点在于利用高关联度和隐式随机化（采用轻量级块密码）实现 Mirage 级别的安全性，同时通过分区实现对占用攻击的防护，并通过 MSR 位实现无负载开关，几乎不改动传统集合关联缓存结构。

**🔧 技术方法**

技术手段包括：高关联度 skewed cache、Simon/Speck 块密码对地址进行随机映射、全局随机置换策略、无指针的标签-数据耦合、SDID 进行安全域隔离、利用 MSR 位实现动态模式切换和完整缓存冲刷。

**📊 数据集**

使用的基准数据集包括 SPEC CPU2017 和 GAP 套件，共 15 个同质化、5 个同质化（不同测点）以及 21 个异质混合工作负载，覆盖多核 8 处理器的典型场景。

**📈 对比分析**

性能评估采用 ChampSim 模拟器，比较非安全基线、Mirage、Maya、DAWG 等方案；Avatar‑R 在所有工作负载下平均仅有 0.25% 以上的性能下降，Avatar‑P 最高 3% 的性能开销，均优于现有分区方案；能耗/面积开销约 1.5%（面积）与 2.7%（静态功耗）以内。

**⚠️ 局限性**

局限性包括：需要较高的关联度和大规模 LLC 才能达到最佳安全性；全局随机置换增加访问延迟，预取敏感工作负载可能受影响；动态切换时需完整冲刷缓存，导致短暂性能损失；方案仍假设硬件安全密钥不泄露，且在 8 核 16 MB 配置下验证。

---

## 102. A Fast and Generalizable Fourier Neural Operator-Based Surrogate for Melt-Pool Prediction in Laser Processing

**arXiv ID:** 2602.06241 | [PDF](https://arxiv.org/pdf/2602.06241v1)

**作者:** Alix Benoit `[一作]` (Laboratory for Advanced Materials Processing), Elia Iseli `[通讯]` (Laboratory for Advanced Materials Processing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一种基于傅里叶神经算子（FNO）的快速通用模型LP‑FNO，用于预测激光加工中的熔池三维温度场与界面；

**💡 创新点**

创新点在于将激光移动参考系与时间平均相结合，将瞬态多物理问题转化为准稳态问题，使FNO能直接学习参数映射；

**🔧 技术方法**

使用了FNO架构、时间平均、非维化归一化、光学吸收和激光波束跟踪等技术；

**📊 数据集**

利用在FLOW‑3D WELD®上生成的高保真三维模拟数据，覆盖40–190W功率、0.1–1 m/s扫描速度（共数千个样本），并按归一化焓(H*)均匀采样；

**📈 对比分析**

与原始FLOW‑3D仿真对比，平均温度误差约18 K（≈3 %），熔池IoU≈0.91，金属–气体界面IoU≈0.999，推断时间仅0.01–0.04 s，速度提升10⁴–10⁵倍；

**⚠️ 局限性**

局限在于关键孔（keyhole）区间的训练数据因粗网格欠收敛导致预测误差增大，FNO对高频细节的重建受限于训练数据质量，并且网络参数量随保留模式数增长显著。

---

## 103. Dynamic Modeling, Parameter Identification and Numerical Analysis of Flexible Cables in Flexibly Connected Dual-AUV Systems

**arXiv ID:** 2602.06087 | [PDF](https://arxiv.org/pdf/2602.06087v1)

**作者:** Kuo Chen `[一作]` (Shenyang Institute of Automation, Chinese Academy of Sciences), Jiancheng Yu `[通讯]` (Shenyang Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对双AUV柔性连接系统建立了完整的动力学模型，并通过实验数据实现了弹性模量和流体阻力系数的逆识别，随后对不同工况下的张力分布与缆绳几何形态进行了数值仿真。

**💡 创新点**

创新点包括：
1) 采用质量堆积法结合轴向弹性、弯曲刚度、加水质量与流体力学，实现了高精度非线性多体动力学模型；
2) 将遗传算法与物理模型耦合，用实验张力数据实现弹性模量与流体阻力系数的同时识别；
3) 通过尺度相似分析和多工况仿真揭示“松弛”与“紧绷”两种缆绳状态对张力分布的决定性影响；
4) 对空间离散、时间步长以及缆绳材料参数、长度等多因素进行系统收敛与敏感性分析。

**🔧 技术方法**

所用技术与方法：质量堆积法、有限差分求解弯曲力、非线性动力学积分、遗传算法优化、UWB/IMU定位、三轴力传感器、RMSE/MSE误差评价、收敛性分析。

**📊 数据集**

数据集：在水槽实验中采集约数百条张力、位置与姿态数据，构成多工况样本集用于参数识别和模型验证。

**📈 对比分析**

评价方法与性能：使用RMSE、MSE对比实验张力，最大张力误差<0.7 N，平均RMSE<0.35 N；空间离散n≥30、时间步Δt≤10⁻³ s即可保证几何误差<0.01 m、张力误差<0.1 N，验证了模型的精度与收敛性。

**⚠️ 局限性**

局限性：
1) 未考虑缆绳自旋、垂直流速及非线性流体耦合，可能影响深海实际工况；
2) 实验环境受限于平面水槽、恒定速度，缺乏高流速与复杂海流等真实海洋条件的验证；
3) 模型中对弹性模量与阻力系数的识别仅基于静态张力数据，动态变化特性仍待进一步研究。

---

## 104. The Eye-Head Mover Spectrum: Modelling Individual and Population Head Movement Tendencies in Virtual Reality

**arXiv ID:** 2602.06164 | [PDF](https://arxiv.org/pdf/2602.06164v1)

**作者:** Jinghui Hu `[一作]` (Lancaster University), Hans Gellersen `[通讯]` (Lancaster University)

**通讯引用:** 14116 | [OpenAlex ID](https://openalex.org/A5024343435)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了虚拟现实中个体眼头协调的差异，提出并验证了一个软铰接（soft‑hinge）参数模型来描述头部对视线转移的贡献，并将其应用于大规模360°视频自由观看数据集与控制实验，揭示了连续的“眼-头移动者谱”并探讨了其在不同任务中的稳定性与适应性。

**💡 创新点**

创新点在于：①首次将眼头协调视为连续维度而非离散类别；②设计了基于目标方位的软铰接模型，可无阈值、无先验分类即可捕捉个体差异；③使用功能主成分分析（fPCA）构建全局分布，量化谱上的集中与极端；④在用户实验中对比抽象目标与自由观看两种情境，证实谱维度在任务间保持一致但表现可塑。

**🔧 技术方法**

采用的技术包括：1) 眼头同步数据预处理与1‑欧拉滤波；2) 通过速度阈值检测固定与转向；3) 对每个转向提取目标方位与头部贡献；4) 非线性最小二乘优化（多起点）拟合软铰接模型；5) 功能主成分分析与联合fPCA；6) 统计比较（R²、RMSE、AIC、Wilcoxon、Pearson相关）。

**📊 数据集**

主要数据集为公开的D‑SAV360（80名受试者，85段30秒单声道360°视频）以及一次包含28名受试者的实验数据，实验中包含抽象目标定位任务与短时视频自由观看任务。

**📈 对比分析**

模型比较：软铰接在R²与RMSE上显著优于线性基线与硬铰接（p<0.001），AIC亦更低；实验中抽象任务的RMSE约3.7°、R²≈0.85，视频任务RMSE约10.8°、R²≈0.91；两任务的谱得分相关系数r=0.60（p<0.001），说明个体在不同情境下保持相对位置。整体而言，模型在各任务均能解释约85‑90%的方差，误差可接受。

**⚠️ 局限性**

局限包括：仅分析水平眼头运动，未涉及垂直或全3D协调；数据集中无躯干运动，仅在站立姿势下收集；实验样本量小且性别比例不平衡；使用的HMD设备可能对头部运动产生惯性影响；软铰接模型假设运动在±50°范围内，超出此范围需考虑躯干参与；未验证模型在其他VR情境（行走、交互、立体视频）或不同人口群体中的泛化能力。

---

## 105. Swap Regret Minimization Through Response-Based Approachability

**arXiv ID:** 2602.06264 | [PDF](https://arxiv.org/pdf/2602.06264v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Jon Schneider `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种新的在线学习算法，用于最小化不同形式的交换回报（swap regret），包括线性交换回报和多项式维度的交换偏差；

**💡 创新点**

创新点在于将响应式逼近框架与John椭圆预处理相结合，首次实现了对一般凸集的线性交换回报的O(d√T)（中心对称时）以及O(d^{3/2}√T)（一般情况）的上界，并给出了匹配的Ω(d√T)下界；

**🔧 技术方法**

主要技术包括：响应式逼近（response-based approachability）算法、John椭圆预处理（geometric preconditioning）、最小化二次型零和博弈的最优均衡求解，以及利用混合策略实现非线性交换偏差的逼近；

**📊 数据集**

该工作不依赖任何公开数据集，而是在理论上对所有可行的损失序列提供保证；

**📈 对比分析**

与以往基于ellipsoid的O(d⁴√T)算法相比，本文的算法在理论上和实现上都更高效，计算复杂度大幅下降，且在中心对称情况下与已知下界达到匹配；

**⚠️ 局限性**

限制在于：对非中心对称凸集，仍存在√d的性能缺口；对于多项式维度交换偏差，还缺乏相应的下界与最优算法，且在高维大T极限下的性能提升仍未得到充分探讨。

---

## 106. Internalized Morphogenesis: A Self-Organizing Model for Growth, Replication, and Regeneration via Local Token Exchange in Modular Systems

**arXiv ID:** 2602.06296 | [PDF](https://arxiv.org/pdf/2602.06296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 107. Generics in science communication: Misaligned interpretations across laypeople, scientists, and large language models

**arXiv ID:** 2602.06190 | [PDF](https://arxiv.org/pdf/2602.06190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 108. Hear You in Silence: Designing for Active Listening in Human Interaction with Conversational Agents Using Context-Aware Pacing

**arXiv ID:** 2602.06134 | [PDF](https://arxiv.org/pdf/2602.06134v1)

**作者:** Zhihan Jiang `[一作]` (Columbia University), Ray LC `[通讯]` (City University of Hong Kong)

**通讯引用:** 908 | [OpenAlex ID](https://openalex.org/A5027284786)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一款基于大型语言模型的对话代理(CA)，该代理通过上下文感知自动调节响应节奏，实现了五种“主动倾听”式的暂停策略（Reflective Silence、Facilitative Silence、Empathic Silence、Holding Space、Immediate Response），从而让AI在情感支持场景中模仿人类的主动倾听行为。

**💡 创新点**

创新点在于：①首次将人类主动倾听中的“时间调节”系统化为可操作的五类暂停策略并量化；②提出基于上下文的动态节奏控制框架，突破传统固定延迟或即时回复的设计；③将这些策略嵌入LLM对话流程中，通过状态栏与文本延迟共同呈现“思考”与“倾听”体验；④通过实验验证动态节奏能显著提升用户对倾听质量、情感信任、自然度与交互活跃度。

**🔧 技术方法**

技术实现：①后端使用Flask + LangChain orchestrate对话；②利用GPT‑4o生成内容；③上下文分析模块采用prompt‑based分类识别八个可实现的策略；④响应生成模块根据策略插入细粒度延迟（0–20 s）与标点微暂停；⑤对话记忆模块通过摘要+token预算维护上下文；⑥前端展示动态状态栏（如“Assistant is reflecting quietly”）与文本流。

**📊 数据集**

数据集与实验材料：①10段真实或模拟咨询视频（YouTube）用于构建主动倾听暂停策略的编码与分类；②50名受试者参与两轮对话（职业支持、恋爱支持）并完成问卷与访谈；③对话日志用于计算自我披露（情感词、第一人称代词、回复长度）和参与度（轮数、策略分布）等指标。

**📈 对比分析**

比较方法：双组（实验组vs. 控制组）之间使用 Mann‑Whitney U 检验；效应量用 rank‑biserial r 报告；问卷覆盖倾听质量、情感信任、认知信任、人类化、流畅度、交互性；对话行为量化自我披露与参与度。结果显示，实验组在职业支持场景中显著提升倾听质量（r≈0.34）、情感信任（r≈0.32）、人类化（r≈0.36）、流畅度（r≈0.30）和交互性（r≈0.48）；在恋爱支持场景提升人类化（r≈0.30）、流畅度（r≈0.36）和交互性（r≈0.43）。自我披露指标（情感词数、第一人称代词、文本长度）及对话轮数在实验组也显著更高。

**⚠️ 局限性**

局限与未来工作：①样本规模与仅中文/英文文本场景限制外推；②对慢节奏的负面体验（效率预期违背）说明需个性化或渐进式适配；③策略识别仍对细腻情绪的误分类率偏高（<90%）；④未评估语音/多模态场景；⑤未深入探讨长期交互中的情绪轨迹与自适应学习。

---

## 109. Multi-Agent-Driven Cognitive Secure Communications in Satellite-Terrestrial Networks

**arXiv ID:** 2602.06048 | [PDF](https://arxiv.org/pdf/2602.06048v1)

**作者:** Yujie Ling `[一作]` (Xidian University), Tony Q. S. Quek `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

本文提出一种基于多智能体强化学习与生成式对抗网络的双层防御框架，用于在卫星‑地面网络（STN）中实现自适应频谱调度与对抗性信号注入，从而提升对学习型窃听者的安全性。

**💡 创新点**

创新点在于：①将多智能体协同调度与GAN生成的对抗信号相结合，形成基于实时感知的自适应频谱保护；②在保证可靠传输阈值的前提下，联合优化频谱占用、对抗矩阵与功率分配，实现安全与能耗的双重优化；③通过将对抗信号与合法流量在时频结构上保持相似性，显著提升窃听者的误检率。

**🔧 技术方法**

技术手段包括：多智能体深度双向Q网络（DDQN）+ QMIX结构的协同决策；WGAN‑GP生成对抗频谱矩阵；强化学习驱动的功率控制策略；理论上推导的密封概率与可靠传输概率，并在奖励函数中综合考虑干扰与可靠性。

**📊 数据集**

数据集：采用仿真环境，包含2颗低轨卫星、4个基站、64个时频槽、64个时隙，卫星/基站功率分别设为53 dBm/37 dBm，采用Rayleigh / Rician 信道模型以及随机部署的窃听者，主要用于评估不同可靠性、用户数和窃听模型下的性能。

**📈 对比分析**

对比方法包括：AN‑Assisted FH（人工噪声+频率跳变）、Game‑Theoretic Allocation（博弈论资源分配）和GANs‑Based Control（基于GAN的功率控制）。实验结果表明，本文方法在保持相同可靠性约束的情况下，密封概率最高、功率消耗最低；在不同用户数、可靠性阈值以及窃听模型下均表现出最优或次优的安全性能。

**⚠️ 局限性**

局限性：①仿真场景与信道模型未涵盖极端恶意攻击和链路切换；②多智能体学习过程对网络规模和模型参数敏感，训练成本较高；③对抗信号的生成需要在运行时保持与合法流量结构一致，实际部署中对时间同步与频谱监测的要求较高；④未考虑能耗与硬件实现的实际限制，缺乏真实网络验证。

---

## 110. Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch Reinforcement Learning

**arXiv ID:** 2602.06107 | [PDF](https://arxiv.org/pdf/2602.06107v1)

**作者:** Zhuoming Chen `[一作]` (Carnegie Mellon University), Beidi Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5073845046)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用 Optimal Budget Rejection Sampling (OBRS) 的 RL 框架，在大语言模型训练中实现投影模型与策略模型的分离，从而降低 rollout 与训练模型之间的分布不匹配，提高训练稳定性。

**💡 创新点**

创新点在于：① 用 OBRS 代替传统的完美重采样，既能显著降低接受率问题，又能在给定的拒绝预算下使采样分布更贴近目标分布；② 结合逆 KL 散度蒸馏损失同步更新投影模型；③ 采用 top‑k 近似与批量偏差校正，实现高效的 OBRS 计算，避免了对整个词表的显式访问。

**🔧 技术方法**

技术主要包括：Optimal Budget Rejection Sampling、逆 KL 散度蒸馏、PPO/GRPO 等强化学习算法、top‑k 近似与批量偏差校正、vLLM 无侵入式推理加速。

**📊 数据集**

实验数据集涵盖数学推理和常识类任务：GSM8K、MATH‑500、AMC22/23、AMC12、AIME24、AIME25 以及 DeepScaleR；模型对比包括 Qwen2.5‑1.5B/3B、Qwen3‑1.7B/4B/8B。

**📈 对比分析**

与 TIS+逆 KL、标准离线 RL 等基线相比，OBRS‑基线在 300 步更新内保持稳定，并在多种任务上逼近或匹配 on‑policy 训练效果；在去除 PPO clipping 的大分布偏移场景下，OBRS 进一步提升收敛速度并防止崩溃。

**⚠️ 局限性**

局限性包括：① 在分布偏移已被现有方法控制的场景下提升有限；② 仍可能在长期训练（>300 步）后出现崩溃；③ 仅在 1B‑8B 范围内验证，尚未在更大规模模型（如 32B）上证明鲁棒性，需进一步的闭环控制或动态调节机制。

---

## 111. Private Sum Computation: Trade-Offs between Communication, Randomness, and Privacy

**arXiv ID:** 2602.06238 | [PDF](https://arxiv.org/pdf/2602.06238v1)

**作者:** Remi A. Chou `[一作]` (University of Texas at Arlington), Aylin Yener `[通讯]` (Ohio State University)

**通讯引用:** 12701 | [OpenAlex ID](https://openalex.org/A5039328157)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在多用户通过公共通道向融合中心求和时，研究在允许有限信息泄露的前提下，如何最小化通信量、全局随机数量以及用户本地随机数量，保证所需的隐私安全性。

**💡 创新点**

提出了在任意泄露度下的通信与随机数容量下界，并证明在达到这些下界时，本地随机数必须由秘密共享（ramp secret sharing）产生；将先前仅针对零泄露的结果推广到可调泄露度的情形，展示了隐私、通信与随机数之间的精确权衡。

**🔧 技术方法**

信息论隐私模型（互信息泄露度量）、对称性假设、组合论证、时间共享编码、梯度（ramp）秘密共享方案等技术被应用于证明与构造。

**📊 数据集**

本文为理论工作，无使用具体数据集；验证以构造性编码方案为例，展示容量下界可被实现。

**📈 对比分析**

与既往零泄露基线相比，本文提供了可调泄露度下的完整容量区间；实验与理论结果表明通信率始终为1，随机数率随泄露度线性下降，且所给编码方案能够同时达到所有下界。

**⚠️ 局限性**

局限包括：需要对泄露对称性的假设；完整可达方案仅在泄露率α为有理数时给出；未探讨实际随机数生成成本与实现复杂度，且对极端泄露度（α→1）下的性能仍未完全阐明。

---

## 112. PackInfer: Compute- and I/O-Efficient Attention for Batched LLM Inference

**arXiv ID:** 2602.06072 | [PDF](https://arxiv.org/pdf/2602.06072v1)

**作者:** Rui Ning `[一作]` (Nanjing University), Fan Lai `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 322 | [OpenAlex ID](https://openalex.org/A5101622777)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个 kernel‑level packing 框架，针对异构批量 LLM 推理中计算与 I/O 不平衡的问题，通过将请求按长度和 KV 访问模式打包成负载平衡的执行组，实现了高效的 attention 计算与内存访问。

**💡 创新点**

创新点在于：① 结合计算与 I/O 的 token 打包策略，消除填充浪费并平衡 GPU 线程块利用；② 在打包的同时重组 KV 缓存，利用前缀共享实现内存局部性；③ 采用动态组容量选择和在线重组机制，使框架在不同工作负载和 GPU 之间自适应。

**🔧 技术方法**

使用了 FlashAttention 核心技术、GPU tiling 与重用、Trie 结构前缀分组、基于 GPU 共享内存与 SRAM 的自适应分组算法，以及多 GPU 分布式扩展。

**📊 数据集**

实验数据集包括 Alpaca、LMSYS 与 Text2SQL 三种真实推理工作负载，涵盖 Qwen3‑4B、Mistral‑7B、Qwen3‑30B‑A3B（MoE）等模型。

**📈 对比分析**

与 FlashAttention 与 Prepack 进行对比，实验显示在 TTFT、TBT 与 TTLT 上平均提升 13%–20%，吞吐量提升约 20%，并在多 GPU 以及不同 GPU 架构（A100、H100、A40）上保持一致的性能改善。

**⚠️ 局限性**

局限性包括：① 需要先行对组容量进行离线/在线分析，可能受 GPU 内存与带宽约束；② 动态重组开销在极端小 batch 或长文本场景下可能显著；③ 主要在 NVIDIA GPU 上验证，对其他硬件或更大规模分布式设置的适用性尚需进一步评估。

---

## 113. DroneKey++: A Size Prior-free Method and New Benchmark for Drone 3D Pose Estimation from Sequential Images

**arXiv ID:** 2602.06211 | [PDF](https://arxiv.org/pdf/2602.06211v1)

**作者:** Seo-Bin Hwang `[一作]` (Chonnam National University), Yeong-Jun Cho `[通讯]` (Chonnam National University)

**通讯引用:** 379 | [OpenAlex ID](https://openalex.org/A5058331158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DroneKey++，一种无先验、端到端的无人机三维姿态估计框架，集成关键点检测、无人机分类与姿态回归；同时构建大规模合成数据集。

**💡 创新点**

创新点包括：①利用类别嵌入自动编码无人机尺寸信息，实现无先验估计；②将射线几何推理与类嵌入相结合的姿态解码器；③在同一框架下联合训练关键点与姿态任务，提高泛化性能。

**🔧 技术方法**

核心技术为：Transformer‑based keypoint 编码器、类嵌入与射线嵌入、MLP 3D 关键点与姿态头、均方误差与循环损失的多任务学习；使用 360°全景合成的渲染流程生成合成数据。

**📊 数据集**

数据集为新建的 DronePose‑+ 数据集，52,920 张 1920×1080 级别图像，涵盖 7 款 DJI 无人机与 88 个户外背景，提供 2D/3D 关键点、旋转、平移、相机参数等完整标注。

**📈 对比分析**

在新数据集的测试场景 #06、#07 与 DronePose、DroneKey 等基线对比，DroneKey++ 的旋转 MAE 17.34°（MedAE 17.1°）与平移 MAE 0.135 m（MedAE 0.242 m）显著优于传统方法；推理速度达到 414.07 FPS (GPU) / 19.25 FPS (CPU)，满足实时需求。

**⚠️ 局限性**

局限性包括：①实验主要基于合成数据，真实环境验证有限；②仅支持具有四个螺旋桨的标准无人机；③对极端遮挡、极小尺度目标的鲁棒性仍待提升。

---

## 114. Near-Optimal Regret for Distributed Adversarial Bandits: A Black-Box Approach

**arXiv ID:** 2602.06404 | [PDF](https://arxiv.org/pdf/2602.06404v1)

**作者:** Hao Qiu `[一作]` (Università degli Studi di Milano), Nicolò Cesa-Bianchi `[通讯]` (Università degli Studi di Milano and Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了在通信仅通过 gossip 的分布式对抗性 K‑臂赌博机和线性赌博机，提出了一种基于区块化延迟反馈的黑盒归约，给出了最优的理论上界和下界，并在此框架下进一步实现了小损失和最好‑双世界的自适应保证。

**💡 创新点**

创新点：
1) 证明了分布式对抗性 K‑臂赌博机的最优 minimax 速率 Θ(√((ρ⁻¹/2 + K/N)T))，显著改进之前的 ρ⁻¹/3(KT)²/3 上界；
2) 设计了区块化的延迟反馈归约，解耦学习与通信，消除了原有的 T²/3 误差；
3) 通过体积支撑网把线性损失压缩到 d 维，从而实现仅 O(d) 通信的最优线性赌博机上界；
4) 在该框架下获得了小损失和最好‑双世界的自适应性能。

**🔧 技术方法**

技术手段：黑盒归约、加速 gossip 共识、重要性采样、延迟反馈框架、FTRL（及其带混合正则化的变体）、Tsallis/负对数边界正则化、体积支撑网。

**📊 数据集**

数据集：论文为理论工作，未使用任何实测数据集，所有结果均为分析获得的上界/下界。

**📈 对比分析**

与先前工作比较：相比于 ρ⁻¹/3(KT)²/3 的上界，本文得到 ρ⁻¹/2 + K/N 的平方根上界，匹配已知下界；在线性赌博机中取得 √((ρ⁻¹/2 + 1/N)dT) 的上界，通信仅需 d 维。自适应小损失、最好‑双世界保证则分别给出了对 T 或 L* 的依赖，性能与最佳已知结果一致。

**⚠️ 局限性**

局限性：
1) 在线性赌博机中仍存在 √(d log K) 的通信相关误差，需进一步改进；
2) 小损失与最好‑双世界的实现需要对学习率/参数进行调优或倍增；
3) 区块化设计导致额外的延迟，可能影响实际应用的实时性；
4) 对非对称网络或异构节点的情况尚未深入探讨。

---

## 115. Deep Unfolded Fractional Optimization for Maximizing Robust Throughput in 6G Networks

**arXiv ID:** 2602.06062 | [PDF](https://arxiv.org/pdf/2602.06062v1)

**作者:** Anh Thi Bui `[一作]` (Ruhr University Bochum), Aydin Sezgin `[通讯]` (Ruhr University Bochum)

**通讯引用:** 3275 | [OpenAlex ID](https://openalex.org/A5034269994)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种不确定性注入的深度展开分数规划框架，针对6G多天线基站的加权总速率最大化问题实现鲁棒 beamforming

**💡 创新点**

将分数规划迭代展开为可训练网络层，并在训练时注入通道误差、使用分位数损失来显式引入鲁棒性

**🔧 技术方法**

深度展开、投影梯度下降、分数规划、分位数（γ=5%）损失、通道误差采样

**📊 数据集**

基于小波衰落（Rayleigh）模拟的 8000 批（每批 64 条通道）训练集，测试集为 50 批相同规模

**📈 对比分析**

与传统 WMMSE、FP 迭代算法及 DL‑UI 训练基准对比，UI‑DUFP 在 γ=5% 分位数下获得更高鲁棒总速率且推理时间更低

**⚠️ 局限性**

实验仅在单基站 4 天线 4 用户的理想 Rayleigh 环境中验证，未讨论多基站、多天线规模、非 Rayleigh 或时变信道等实际复杂场景

---

## 116. RoPE-LIME: RoPE-Space Locality + Sparse-K Sampling for Efficient LLM Attribution

**arXiv ID:** 2602.06275 | [PDF](https://arxiv.org/pdf/2602.06275v1)

**作者:** Isaac Picov `[一作]` (University of Toronto), Ritesh Goru `[通讯]` (DevRev)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出 RoPE-LIME，一种面向闭源 LLM 输出的解释框架，解耦推理与解释，利用开放源代码替代子模型来计算概率分布级别的词级归因。

**💡 创新点**

创新点包括：①在 RoPE 嵌入空间中使用 Relaxed Word Mover’s Distance 构建局部核；②Sparse‑K 采样策略实现对特征空间的对数级覆盖；③通过概率分布目标而非生成文本实现更稳定的归因。

**🔧 技术方法**

核心技术为 LIME 框架、RoPE 位置编码、RWMD 距离度量、Sparse‑K 采样以及权重线性回归。

**📊 数据集**

实验使用 HotpotQA（句子级特征）和手工标注的 MMLU 子集（词级特征）进行评估。

**📈 对比分析**

与 gSMILE 的对比显示 RoPE-LIME 在 IoU、F1 和 AUROC 上均优于 gSMILE，并且在闭源模型调用量上显著下降。

**⚠️ 局限性**

局限性包括：对长上下文仍需较多采样；当前归因仅基于固定输出，无法捕获多轮对话的动态变化；以及在高度上下文相关的词级特征上改进空间有限。

---

## 117. Bioinspired Kirigami Capsule Robot for Minimally Invasive Gastrointestinal Biopsy

**arXiv ID:** 2602.06207 | [PDF](https://arxiv.org/pdf/2602.06207v1)

**作者:** Ruizhou Zhao `[一作]` (Chinese University of Hong Kong), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 17102 | [OpenAlex ID](https://openalex.org/A5032340829)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种采用纸工艺（kirigami）可吞咽的胃肠道活检胶囊——Kiri‑Capsule；

**💡 创新点**

创新点在于将可展开的纸工剪裁面与双轴机杠杆结合，利用摄动机制实现可控深度的尖锐突出与旋转刮削，实现安全、可重复的活检；

**🔧 技术方法**

技术包括聚酰亚胺（PI）薄膜纸工剪裁、双轴机杠杆驱动、微型步进电机、旋转刮削装置及内部结构化存储腔；

**📊 数据集**

数据集为猪胃壁与小肠组织的离体样本（共14个活检），用于验证展开角度、穿透深度、力量与活检量；

**📈 对比分析**

与传统手持活检钳比较时，Kiri‑Capsule 在胃部平均收集量约10.9 mg、肠道约18.9 mg，穿透深度中值≈0.61 mm，作用力均落在临床安全阈值内，表现与标准手术钳相当；

**⚠️ 局限性**

局限性包括需要有线供电、仅支持单点活检、可能出现样本混淆以及对材料韧性与长周期使用的进一步验证需求。

---

## 118. Unsupervised Anomaly Detection of Diseases in the Female Pelvis for Real-Time MR Imaging

**arXiv ID:** 2602.06179 | [PDF](https://arxiv.org/pdf/2602.06179v1)

**作者:** Anika Knupfer `[一作]` (Friedrich-Alexander University Erlangen–Nürnberg), Jana Hutter `[通讯]` (Leibniz University Hannover)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一个无监督异常检测框架，用于在实时 MRI 成像中识别女性骨盆疾病。

**💡 创新点**

创新点在于：①疾病无关、参数无关的通用模型；②利用扩增的扩散模型生成合成健康数据提升鲁棒性；③实现近 92.6 帧/秒的低延迟推理，支持实时 MRI；④通过多观察者评估揭示注释不确定性对性能评估的影响。

**🔧 技术方法**

技术包括：残差变分自编码器（ResVAE）与结构化损失（MSE+SSIM+KL+感知损失）；DDPM（扩散模型）用于合成健康 T2‑w MRI；后处理阈值化、空间加权和中值滤波以强化异常热点。

**📊 数据集**

数据集：294 张健康 sagittal T2‑w MRI（多场强、不同序列、不同厂商）构成训练集；242 张病变 MRI 作为评估集，其中 215 张来自公开 Uterine Myoma MRI Dataset（UMD），27 张为医院内部病例（子宫瘤、子宫内膜癌、子宫内膜异位、腺肌症）。

**📈 对比分析**

与标准评估方法（ROC、AUC、敏感性、特异性）比较，模型在 UMD 上平均 AUC 0.736，针对腺瘤、腺肌症等病变分别取得 AUC 0.826 与 0.619；在内部数据集上，AUC 最高 0.901（子宫瘤）到最低 0.515（子宫内膜癌）。与传统监督方法相比，模型在多模态、无标注条件下实现了可比甚至更优的性能，并在不同子宫位姿下保持稳定。

**⚠️ 局限性**

局限性：①对高质量健康训练数据依赖较强，若含有病变会误导模型；②罕见子宫姿势（后弯）和少数病变类型的样本不足，导致泛化受限；③VAE 的概率输出导致重建噪声与微小异常检测灵敏度受限；④评估受限于人工注释的主观性和观察者差异，尤其在腺肌症等病变的界定上。

---

## 119. Latent Structure Emergence in Diffusion Models via Confidence-Based Filtering

**arXiv ID:** 2602.06155 | [PDF](https://arxiv.org/pdf/2602.06155v1)

**作者:** Wei Wei `[一作]` (University of Pittsburgh), Hung-Hsu Chou `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探讨扩散模型潜在空间在高置信度样本下的结构，并提出基于置信度过滤的条件生成方法。

**💡 创新点**

通过置信度过滤揭示潜在空间中类相关结构，并利用此结构实现无需修改模型的条件生成。

**🔧 技术方法**

置信度过滤、LDA-UMAP可视化、标签可预测性实验、DDIM采样器、预训练分类器。

**📊 数据集**

MNIST 数据集。

**📈 对比分析**

对比不同置信度级别下的标签可预测性和类分离度，发现高置信度可达约90%准确率，低置信度仅20%，相较于随机种子条件生成效果显著提升。

**⚠️ 局限性**

仅适用于确定性采样器（如DDIM），训练潜在分类器成本高，低置信度区域结构仍不完整，缺乏理论保证。

---

## 120. UAV-Mounted Aerial Relays in Military Communications: A Comprehensive Survey

**arXiv ID:** 2602.06061 | [PDF](https://arxiv.org/pdf/2602.06061v1)

**作者:** Faisal Al-Kamali `[一作]` (University of Ottawa), Claude D'Amours `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了无人机（UAV）搭载空中中继（AR）在军事通信中的现状，比较了 AR 与地面中继（TR）的性能，并对两类 AR（主动空中中继 AAR 与被动空中智能表面 ARIS）进行详细评述，提出了多维任务关键中继效能得分（MCRES）指标和相应的决策算法。

**💡 创新点**

创新点在于：①提出了 MCRES 这一综合量化指标，结合任务特定权重评估中继效能；②设计了算法 1，利用 MCRES 系统化地选择最优中继类型（AR/TR 及 AAR/ARIS）；③在综述中首次将主动与被动空中中继的技术、优势、挑战与军事应用统一对比，填补了以往单纯聚焦于民用或单一技术的空白。

**🔧 技术方法**

采用的技术主要为文献综述、定性分析、指标构建与数值仿真。对 MCRES 采用权重向量和多属性决策方法，对算法 1 采用规则化的决策流程；同时引用现有研究中的实验结果和行业报告做参考。

**📊 数据集**

由于本文为综述性质，并无新的实验数据集；所有结论均基于已有公开文献、行业报告（如 DoD、NATO STANAG、IEEE 等）以及公开的 UAV 性能数据和案例分析。

**📈 对比分析**

比较方法：先对 AR 与 TR 在覆盖、灵活性、成本、安全、抗干扰、部署速度等指标进行定性对照；随后使用 MCRES 计算各中继方案的得分并通过算法 1 进行决策；对比结果显示：AR 在机动性、覆盖及抗干扰上优于 TR，但成本和能耗较高；AAR 适用于需要可靠链路的场景，而 ARIS 由于被动特性更节能且隐蔽性更好，适合隐蔽或资源受限的任务。

**⚠️ 局限性**

局限性：①综述依赖现有文献，缺乏大规模实测或仿真数据；②MCRES 权重设置主观，未来需根据实际任务进行校准；③未涉及深度学习、边缘计算等前沿技术的集成细节；④对多兵种协同、复杂战场环境下的自适应行为讨论不足。

---

## 121. Adaptive Protein Tokenization

**arXiv ID:** 2602.06418 | [PDF](https://arxiv.org/pdf/2602.06418v1)

**作者:** Rohit Dilip `[一作]` (California Institute of Technology), David Van Valen `[通讯]` (California Institute of Technology)

**通讯引用:** 5040 | [OpenAlex ID](https://openalex.org/A5045170248)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种全局自适应蛋白分词器（Adaptive Protein Tokenizer，APT）及其扩散自编码器，用于蛋白结构的重建、生成和表征。

**💡 创新点**

创新点包括：①利用全局分词而非局部邻域，使每个 token 逐级细化全局表示；②通过嵌套丢弃（nested dropout）强制令前缀携带关键全局信息；③结合熵阈值采样与分类器退火，实现基于信息量的采样停止和更好设计性；④将分词器与生成器解耦，实现零样本蛋白缩放和亲和力成熟等应用。

**🔧 技术方法**

技术实现包括：Transformer 编码器/解码器、扩散自编码器与流匹配（flow‑matching）目标、有限标量量化（FSQ）与嵌套丢弃、SE3 学习、classifier‑free 退火、熵基采样、自回归模型，以及 ODE/SDE 采样。

**📊 数据集**

主要使用数据集为从 AlphaFold2 预测的 Foldseek 聚类 AFDB（约 473k 结构），并在 CAMEO、CATH 子集和 AFDB 子集上做留出测试。

**📈 对比分析**

与本地分词模型（DPLM2、ESM3、Kanzi、IST、FoldToken）比较：重建任务 RMSD 0.90 Å、TMscore 0.941；生成任务设计性达 0.87、scRMSD 1.35 Å，且多样性与分布覆盖（gFID）均优于或相当；表征学习任务 CATH 分类的 MLP 线性探测器在 16‑token 级别即可超过 DPLM2/ESM3。

**⚠️ 局限性**

局限性包括：①仅适用于全局任务，局部任务（如模体搭建）效果不佳；②对侧链与氨基酸组成的建模不足；③对大规模复合体的推断仍需进一步研究；④不同代码本大小、编码器深度等超参数影响尚未完全明晰；⑤训练与推断成本高，需 ODE/SDE 采样。

---

## 122. MeDocVL: A Visual Language Model for Medical Document Understanding and Parsing

**arXiv ID:** 2602.06402 | [PDF](https://arxiv.org/pdf/2602.06402v1)

**作者:** Wenjie Wang `[一作]` (Ping An Property and Casualty Insurance Company), Hong Li `[通讯]` (Ping An Property and Casualty Insurance Company)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个后训练框架 MeDocVL，用于在噪声注释下实现高精度、查询驱动的医疗文档内容解析。

**💡 创新点**

创新点在于：①训练驱动标签精炼（TLR）将粗糙注释转化为稳定监督；②噪声感知混合后训练（NHP）将强化学习与监督微调结合，采用 token‑级 GRPO；③实现端到端的查询驱动解析。

**🔧 技术方法**

使用的技术包括：视觉语言模型（如 Qwen2.5‑VL）、OCR 与 MLLM 预训练，训练驱动标签精炼、强化学习（GRPO）、动态提示、token‑级奖励机制及混合后训练。

**📊 数据集**

使用的数据集：公共阿里云医疗发票数据集（800 张，700/100 训练/测试分割）以及内部大规模医疗文档。

**📈 对比分析**

在与多种 OCR 系统（DeepSeek‑OCR、PaddleOCR‑VL、MonkeyOCR Pro 等）和通用 MLLM（Qwen2.5‑VL、Gemini‑2.5‑Pro、ChatGPT‑4o 等）的查询驱动评估中，采用 Field Match Ratio (FMR) 作为指标；MeDocVL 公共模型 FMR ≈0.824，Extended 模型 ≈0.940，显著优于基线（≈0.29‑0.69）。

**⚠️ 局限性**

局限性包括：仍需大量人工或半自动标签生成；对极端噪声或全新文档格式的泛化尚需验证；仅在医疗发票领域验证，其他行业应用需要进一步测试。

---

## 123. Uniform Spectral Growth and Convergence of Muon in LoRA-Style Matrix Factorization

**arXiv ID:** 2602.06385 | [PDF](https://arxiv.org/pdf/2602.06385v1)

**作者:** Changmin Kang `[一作]` (KAIST), Chulhee Yun `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析并证明了在LoRA风格矩阵分解中使用Muon风格的Spectral Gradient Flow（SpecGF）时，产品矩阵的奇异值会呈现近乎均匀的增长（equal‑rate）行为，从而导致更快的收敛。

**💡 创新点**

创新点在于揭示并解析了SpecGF在LoRA设置下的均匀谱增长现象，证明其在连续时间动力学框架中可达到全局最优；并提供了对比传统梯度流的理论解释和实验验证。

**🔧 技术方法**

采用Spectral Gradient Flow、矩阵正交化（orthogonalization）、连续时间梯度流分析、Lyapunov 稳定性与 Lojasiewicz 不等式，以及对LoRA参数化的矩阵分解实验。

**📊 数据集**

实验数据集包括 RoBERTa‑Base 在 SST‑2（GLUE benchmark）上的微调，以及 LLaMA‑3.2‑1B 在 Alpaca 数据集上的微调，LoRA 维度设为 8。

**📈 对比分析**

与 AdamW 与 Vanilla Gradient Flow 进行对比；结果显示 SpecGF 在奇异值演化上保持几乎完全相同的斜率，收敛速度明显快于标准梯度流，最终损失更低。

**⚠️ 局限性**

局限性：对无正则化情况的全局收敛性尚无严格证明；离散时间 SpecGD 的收敛分析缺失；对更复杂网络结构的推广仍需进一步研究。

---

## 124. Git for Sketches: An Intelligent Tracking System for Capturing Design Evolution

**arXiv ID:** 2602.06047 | [PDF](https://arxiv.org/pdf/2602.06047v1)

**作者:** Sankar B `[一作]`, Dibakar Sen `[通讯]` (Indian Institute of Science)

**通讯引用:** 7018 | [OpenAlex ID](https://openalex.org/A5065348763)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个面向产品概念草图的完整系统DIMES，内部集成了专门为视觉域设计的版本控制架构sGIT、基于深度学习与传统机器学习相结合的Stroke分类器AEGIS、以及生成式AI模块，用于自动生成设计演化叙事和逼真渲染；同时在实验中验证了该系统能显著提升概念探索范围、加速知识转移并提高终端用户的购买倾向。

**💡 创新点**

创新点包括：
1) 以stroke为粒度的视觉版本控制系统sGIT，映射Git原语到设计动作，实现无缝分支与历史回溯；
2) AEGIS实现了多模态实时记录（坐标、压力、厚度等）并构建首个含六种stroke标签的高质量数据集；
3) 采用混合AI管道（CNN+特征工程）实现stroke类型自动识别，兼顾视觉与运动信息；
4) 通过多模态提交（语音+文本）捕获设计意图，生成AI叙事与渲染，弥补传统工具的认知记录缺失。

**🔧 技术方法**

技术栈与模型：
- 前端：React + Vercel；后端：Supabase（PostgreSQL + 对象存储）；
- Stroke分类：5个CNN（ResNet50、EfficientNetB0、ConvNeXt‑Tiny等）+ 10个传统ML模型（RandomForest、SVM‑RBF等），最终选用RandomForest进行在线推理；
- AI叙事与渲染：Gemini LLM + Stable Diffusion/ControlNet 风格迁移；
- 评估工具：UMAP可视化、Neural Transparency 余弦相似度、购买倾向评分。

**📊 数据集**

数据集：
1) AEGIS自建stroke图像集：每类（Constraining、Defining、Detailing、Shading/Shadow、Annotation）500条基础样本，经过10×扩增后5,000张/类，总计20,000张；
2) 真实手绘stroke序列：5位设计师共绘1,000+条stroke，作为Field Test；
3) 20个YouTube视频序列（共计约30,000帧），用于专家行为分析与特征提取。

**📈 对比分析**

比较方法与性能：
- 在概念探索方面，专家使用DIMES时概念广度提升160%，提交粒度提升800%；
- Stroke分类：DL模型单样本准确率≈97.6%，ML随机森林在Field Test中达100%；
- 知识转移：AI叙事摘要下新手复制精度Cosine相似度0.97，传统摘要0.73；
- 用户接受度：AI渲染的产品概念渲染评分4.2，传统渲染3.1；
- 研究对照实验显示，sGIT与AI模块共同显著提高了设计过程的可追溯性与可解释性。

**⚠️ 局限性**

局限性：
1) 数据规模仍有限，stroke分类模型可能对极端风格或新型笔刷产生误判；
2) DL模型存在Sim‑to‑Real差距，需进一步结合真实设备采样；
3) 仅覆盖六种stroke类型，未涵盖更细粒度的绘画手段；
4) 系统依赖稳定的网络与浏览器环境，离线场景支持不足；
5) AI叙事与渲染仍受模型偏差与算力限制，尚需针对不同设计领域进行微调。

---

## 125. End-to-End Throughput Benchmarking of Portable Deterministic CNN-Based Signal Processing Pipelines

**arXiv ID:** 2602.06216 | [PDF](https://arxiv.org/pdf/2602.06216v1)

**作者:** Christiaan Boerkamp `[一作]` (VLV Technology), Akhil John Thomas `[通讯]` (VLV Technology)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种端到端可重复的基准方法，用于评估用CNN兼容算子实现的确定性信号处理流水线在GPU和TPU上的性能。

**💡 创新点**

创新点在于将传统DSP流程完全用无学习权重的CNN原语表达，保持确定性与可证性，同时实现跨平台无代码改动的高性能执行。

**🔧 技术方法**

使用CNN卷积、点乘、聚合、减法等固定算子以及PyTorch/TPU编译器；对动态索引、稀疏矩阵三种实现变体进行对比。

**📊 数据集**

采用真实超声测量RF数据作为输入，包含B‑mode、彩色多普勒与功率多普勒三种成像模态。

**📈 对比分析**

通过同一代码库在RTX 5090 GPU和TPU v5e‑1上多次前向推理，测量平均推理时间、吞吐量（MB/s）和帧率（FPS），结果显示GPU在动态索引下吞吐最高（≈7GB/s），TPU在全CNN下达530MB/s，证明全CNN方案跨平台性能良好。

**⚠️ 局限性**

局限在于未公开完整代码、未评估图像质量与临床效果，TPU缺乏功耗与内存峰值监测。

---

## 126. Alleviating Sparse Rewards by Modeling Step-Wise and Long-Term Sampling Effects in Flow-Based GRPO

**arXiv ID:** 2602.06422 | [PDF](https://arxiv.org/pdf/2602.06422v1)

**作者:** Yunze Tong `[一作]` (Zhejiang University), Pipei Huang `[通讯]` (Alibaba Group)

**通讯引用:** 1629 | [OpenAlex ID](https://openalex.org/A5059615376)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 TurningPoint-GRPO，解决 Flow‑based GRPO 中奖励稀疏和长期依赖问题，改进了步骤级奖励和转折点识别；

**💡 创新点**

创新点在于用增量式步骤奖励替代终端奖励，并通过符号变化自动识别转折点，给转折点分配聚合长期奖励，无需额外超参数；

**🔧 技术方法**

使用 Flow‑Matching、GRPO、SDE/ODE 采样、PPO‑style RL、LoRA 微调以及奖励模型；

**📊 数据集**

使用 Geneval、PickScore、OCR 规则集等数据集（并以 SD3.5‑M 作为基础模型）；

**📈 对比分析**

通过与 Flow‑GRPO 的对比实验，在三项任务（图像生成、偏好对齐、文本渲染）上均获得更高奖励、更快收敛、图像质量提升；

**⚠️ 局限性**

局限性包括对采样窗口大小和噪声尺度的敏感性，在极端超参数设置下性能可能下降，且仍未完整捕捉所有跨步骤的交互影响。

---

## 127. Halt the Hallucination: Decoupling Signal and Semantic OOD Detection Based on Cascaded Early Rejection

**arXiv ID:** 2602.06330 | [PDF](https://arxiv.org/pdf/2602.06330v1)

**作者:** Ningkang Peng `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 618 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Cascaded Early Rejection (CER) 框架，利用分层筛选实现高效的离散分布检测。

**💡 创新点**

创新点在于将物理层的结构能量筛选 (SES) 与语义层的超球面能量检测 (SHE) 结合，彻底拆分低级信号拦截与高级语义判断，避免计算浪费与语义幻觉。

**🔧 技术方法**

采用拉普拉斯算子实现高频能量检测，利用 Top‑K 通道聚合提升物理异常敏感度；在中间层使用超球面投影和 L2 归一化解耦特征幅值与方向，提升语义判别力。

**📊 数据集**

在 CIFAR‑10/CIFAR‑100 作为 ID 数据集，同时使用 SVHN、MNIST、Places365、LSUN、iSUN、Textures 等多种 OOD 数据集进行评估。

**📈 对比分析**

与 MSP、ODIN、Energy、Mahalanobis、kNN+、LogitNorm、CIDER、PALM 等基线比较，CER 在 FPR95 上下降 30‑32%，AUROC 提升至 93‑97% 以上，且平均 FLOPs 降低 20‑35%。

**⚠️ 局限性**

局限性包括对超参数 K 的敏感性、需要预先统计 ID 数据、以及对极端非结构化噪声的鲁棒性仍有提升空间。

---

## 128. PhenoLIP: Integrating Phenotype Ontology Knowledge into Medical Vision-Language Pretraining

**arXiv ID:** 2602.06184 | [PDF](https://arxiv.org/pdf/2602.06184v1)

**作者:** Cheng Liang `[一作]` (Shanghai Jiao Tong University), Weidi Xie `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10074 | [OpenAlex ID](https://openalex.org/A5076097168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了大规模、多模态的基于表型本体的知识图谱PhenoKG，并基于其提出了PhenoLIP预训练框架，提升医学视觉‑语言模型的表型识别能力。

**💡 创新点**

创新点在于①首次将Human Phenotype Ontology与图像-文字对齐生成多模态知识图谱；②提出两阶段预训练，即先学习表型知识编码器，再通过知识蒸馏将表型结构注入视觉‑语言对齐；③设计细粒度的数据清洗与子图像检测管线。

**🔧 技术方法**

采用CLIP/BiomedCLIP视觉-文本编码器、PubMedBERT知识编码器、对比学习、知识蒸馏、LLM生成文本清洗、DINOv3/K-means、DAB-DETR子图像检测等技术。

**📊 数据集**

使用PhenoKG中的524,804张医学图像-文字对，涵盖3,096种表型；PhenoBench作为专家审核的评测集，包含7,819张图像-文字对，1,187种表型；并在多种公开医学数据集（如HAM10000、DermaMNIST、RSNA等）进行下游评测。

**📈 对比分析**

与多种基线（OpenCLIP、SigLIP2、CoCa、PMC-CLIP、BiomedCLIP、BIOMEDICA以及知识增强模型DermLIP、KEP、MedKLIP、KAD）对比，在零样本分类平均准确率提升约9%（36.56% vs 27.41%），PhenoBench跨模态检索R@10提升至63.30%（比BIOMEDICA高22.79%），罕见面部表型检索R@5提升至7.49%（最高）。

**⚠️ 局限性**

局限性包括：①依赖大量手工审核的表型映射，构建成本高；②对极长尾表型的覆盖仍有限；③模型在某些非表型导向任务（如广义组织分类）提升不明显；④仅使用HPO作为本体，缺乏与其他医学本体的跨模态融合。

---

## 129. Uncertainty Drives Social Bias Changes in Quantized Large Language Models

**arXiv ID:** 2602.06181 | [PDF](https://arxiv.org/pdf/2602.06181v1)

**作者:** Stanley Z. Hua `[一作]` (University of California Berkeley), Irene Y. Chen `[通讯]` (University of California Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了后训练量化对大型语言模型社会偏见的影响，并在50个量化模型上进行系统评估

**💡 创新点**

发现“量化诱导的掩蔽偏见翻转”现象，即在高不确定性预测下，模型输出偏见状态最多可翻转21%；量化强度和模型不确定性是驱动该现象的主要因素；量化对不同社会群体的影响不对称且难以预测；8位量化显著低于4位量化导致偏见改变

**🔧 技术方法**

使用了多种后训练量化方法（RTN、GPTQ、AWQ、SmoothQuant）以及对不确定性进行偏好调优（SimPO、EntropyMax），并通过配对抽样检验统计显著性

**📊 数据集**

构建了PostTrainingBiasBench，覆盖13个闭式和开放式偏见基准（共85k题），并对10种指令微调模型及其50个量化变体进行评估

**📈 对比分析**

采用配对响应比较、Shannon熵测量不确定性、排列检验（p值、Cohen d）以及多重检验校正；结果显示8位量化的行为改变率约为4–6倍低于4位量化，模型规模与稳定性无显著关联，且聚合指标往往掩盖了个体级别的严重偏见翻转

**⚠️ 局限性**

局限性包括仅使用英文数据集、假设性上下文、缺乏多语言和交叉性分析、仅使用确定性生成、以及偏见评估工具对某些数据集的误报率高

---

## 130. Cost-Aware Model Selection for Text Classification: Multi-Objective Trade-offs Between Fine-Tuned Encoders and LLM Prompting in Production

**arXiv ID:** 2602.06370 | [PDF](https://arxiv.org/pdf/2602.06370v1)

**作者:** Alberto Andres Valdes Gonzalez `[一作]` `[通讯]` (Pontifical Catholic University of Chile), Alberto Andres Valdes Gonzalez (Pontifical Catholic University of Chile)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了零/少样本提示式大型语言模型与精细调优的编码器模型在固定标签文本分类任务中的表现，并将准确率、推理延迟和成本等多维度指标纳入决策框架。

**💡 创新点**

创新点在于将模型选择视为多目标决策问题，提出了基于宏F1、成本与延迟的效用函数和Pareto前沿分析，并提供可复现的基准实验体系，支持成本与治理意识的模型决策。

**🔧 技术方法**

采用BERT系列（BERT、RoBERTa、DistilBERT）进行全微调，使用GPT‑4o和Claude Sonnet 4.5的零/少样本提示，利用确定性解码、宏F1、延迟分位数和按令牌计费的成本计算，并构造效用函数U(F1/Cost·exp(‑Latency/τ))进行量化比较。

**📊 数据集**

实验数据集包括IMDB、SST‑2、AG News和DBPedia四个经典英文文本分类基准。

**📈 对比分析**

通过对宏F1、p50/p95延迟和百万请求成本的对比，发现精细调优编码器在准确率与成本/延迟上均优于LLM提示；在Pareto前沿与效用排序中，DistilBERT始终排名首位，LLM模型被多数情形所支配。

**⚠️ 局限性**

局限性包括仅评估固定标签、确定性提示的场景，未涵盖链式思考或自一致性等高级提示策略，实验依赖特定云部署与API定价，且结果可能因语言、数据分布或成本结构变化而不同。

---

## 131. MORPH Wheel: A Passive Variable-Radius Wheel Embedding Mechanical Behavior Logic for Input-Responsive Transformation

**arXiv ID:** 2602.06265 | [PDF](https://arxiv.org/pdf/2602.06265v1)

**作者:** JaeHyung Jang `[一作]` (Korea Advanced Institute of Science and Technology), Jee-Hwan Ryu `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 4747 | [OpenAlex ID](https://openalex.org/A5019438521)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计、建模、制造并实验验证了一种全被动、可变半径、通过机械编程实现扭矩响应的摩擦轮（MORPH wheel）。

**💡 创新点**

创新点在于：①完全无电、无传感器的被动可变传动；②通过对齿轮耦合器、弹性连杆等零件几何与材料特性的精确编程，实现双向无限旋转、高扭矩传输（>10 N）和确定性半径变换；③提出了基于阈值的机械行为逻辑模型，构成了“机械编程”新范式。

**🔧 技术方法**

使用了：滑动-曲柄机械耦合器、弹性连杆/弹簧、可编程柔性接头、ABS+聚酰胺材料、精密3D打印与激光切割、静力学/动力学解析模型、实验台测量、机器人平台驱动电流与速度监测。

**📊 数据集**

使用的数据为实验数据：机械台上扭矩‑半径、弹簧阻力曲线；机器人平台上不同负载、斜坡和自然地形下的电流、速度、轮径变化。未使用公开数据集。

**📈 对比分析**

通过将MORPH轮与固定直径轮（80 mm、45 mm）在相同负载、坡度、地形下进行对比。结果显示：在高负载或上坡时，MORPH轮电流显著低于最大直径轮（约30–40 %降低），速度保持在两固定轮之间的中间水平；在平地时，MORPH轮因轮径不完全圆形产生振动导致电流略高。整体表现为能耗更低、效率更高，同时保持可接受的行驶速度。

**⚠️ 局限性**

局限性包括：①轮径变化范围仅为0–38 mm，无法实现更大变比；②滑动-曲柄连杆与弹簧存在间隙，导致初期阻力偏差与滞后；③轮辋由ABS制成，长期疲劳和高温下性能待验证；④实验仅在单轮或小型平台上验证，尚未评估在多轮、重载或极端环境下的可扩展性。

---

## 132. Uncertainty-Aware 4D Gaussian Splatting for Monocular Occluded Human Rendering

**arXiv ID:** 2602.06343 | [PDF](https://arxiv.org/pdf/2602.06343v1)

**作者:** Weiquan Wang `[一作]` (Zhejiang University), Long Chen `[通讯]` (The Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于最大后验估计的4D高斯渲染框架 U-4DGS，用于单目遮挡人体渲染。

**💡 创新点**

创新点在于将观测噪声建模为异方差拉普拉斯分布，并通过概率变形网络预测不确定性，利用双重光栅化实现像素级不确定性映射来自适应抑制遮挡梯度，同时引入置信度感知正则化防止几何漂移。

**🔧 技术方法**

使用了概率变形网络、双重光栅化、4D高斯渲染、SMPL骨架、拉普拉斯似然损失、空间-时间置信度正则化、Sinusoidal位置编码等技术。

**📊 数据集**

在 ZJU‑MoCap（合成遮挡）和 OcMotion（真实遮挡）两个数据集上进行训练与评估。

**📈 对比分析**

与标准人类渲染方法和三类遮挡感知方法（场景分离、几何先验、生成先验）相比，U‑4DGS在两数据集上均取得最高 PSNR/SSIM，遮挡下的渲染质量和时序一致性明显优于现有最先进方法。

**⚠️ 局限性**

局限性包括仍需标注的 SMPL 姿态信息、对极端遮挡或极低帧率下的性能可能下降，以及不确定性估计可能受到训练策略的影响。

---

## 133. $f$-FUM: Federated Unlearning via min--max and $f$-divergence

**arXiv ID:** 2602.06187 | [PDF](https://arxiv.org/pdf/2602.06187v1)

**作者:** Radmehr Karimian `[一作]` (Universite de Geneve), Gholamali Aminian `[通讯]` (Alan Turing Institute)

**通讯引用:** 67 | [OpenAlex ID](https://openalex.org/A5010622937)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于 f‑divergence 的最小‑最大联邦无学习框架 f‑FUM，能够在客户端或数据级别主动删除模型对指定数据的影响；

**💡 创新点**

创新点在于用 f‑divergence（如 KL、JS、χ²）最大化忘却数据的分布差异、最小化保留数据性能损失；实现可插拔式、无需模型结构信息的联邦无学习，并通过交替最大化/最小化更新与后期细化实现高效、稳定的无学习；

**🔧 技术方法**

采用了最小‑最大优化、Jensen‑Shannon/KL/χ² f‑divergence、教师‑学生对齐损失、FedAvg 通信框架、梯度更新与多阶段学习率；

**📊 数据集**

使用了 CIFAR‑10、FashionMNIST、MNIST 三大公开数据集，模型包括 ResNet‑18、轻量 CNN（两层卷积+LayerNorm）以及 LeNet‑5；

**📈 对比分析**

与 NoT、MoDe、Halimi 等基线在客户端级去背门、标签混淆、全客户端删除、以及数据级 2% 小比例删除等多场景进行对比；实验表明 f‑FUM 在多数场景下取得更高准确率、低 MIA（接近随机 50%），并在复杂情况下保持稳定性；

**⚠️ 局限性**

局限性包括：对 f‑divergence 的选择敏感，极小忘却量时可能波动；需要额外通信轮次；尚缺理论收敛与安全性完整证明；在极大模型或极稀疏数据场景下性能仍待验证。

---

## 134. Know Your Scientist: KYC as Biosecurity Infrastructure

**arXiv ID:** 2602.06172 | [PDF](https://arxiv.org/pdf/2602.06172v1)

**作者:** Jonathan Feldman `[一作]` (Georgia Institute of Technology), Annie I Anton `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出一种三层 KYC 框架，利用研究机构担任信任锚点、实时序列同源性与功能注释筛查以及行为模式监控来管理生物 AI 工具的访问。

**💡 创新点**

创新点在于将金融 AML 的 KYC 模式迁移到生物 AI 领域，强调用户身份验证与机构责任而非单纯的内容过滤，并实现可立即部署的分层治理结构。

**🔧 技术方法**

核心技术包括机构验证机制、序列同源搜索（如 BLAST）、功能注释、可插拔的毒性预测工具和长期行为分析算法。

**📊 数据集**

本研究未使用传统实验数据集，而是基于现有的机构监管记录、模型风险评估框架和公开的生物威胁数据库进行设计。

**📈 对比分析**

该框架并未在实验中与其他方法对比，评估主要通过理论分析与与 AML 实践的类比，提出可落地的治理路径。

**⚠️ 局限性**

局限性包括对开放权重模型无效、难以阻止资源充足的国家行为者、阈值设定与行为监控需经验调整、以及潜在的内部威胁难以完全检测。

---

## 135. REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop

**arXiv ID:** 2602.06248 | [PDF](https://arxiv.org/pdf/2602.06248v1)

**作者:** Patryk Rybak `[一作]` (Jagiellonian University), Przemysław Spurek `[通讯]` (IDEAS Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）在实施机器无学习（unlearning）后，利用演化式对抗提示（jailbreak）评估模型是否真正忘记了敏感信息；

**💡 创新点**

提出了一种全黑盒、演化搜索的提示生成框架（REBEL），通过LLM评判器对泄漏程度进行细粒度评分，证明传统忘记指标低估了可恢复知识，并提供了可复制的对抗性评估流程；

**🔧 技术方法**

演化算法（population‑based search）、黑盒查询、LLM 评判器（Judge）与LLM 生成器（Hacker）、泄漏评分函数、攻击成功率（ASR）等技术；

**📊 数据集**

TOFU（biographical Q&A）和 WMDP（多选安全）两个基准，分别使用 10%/5% 的忘记子集和 100 样本的 WMDP‑Bio 子集；

**📈 对比分析**

与基线（原始提示、Leak@K）以及七种主流无学习方法（AltPO、NPO、IDKDPO、GradDiff、UNDIAL、SimNPO 等）比较，演化攻击在 TOFU‑10% 的 ASR 可达 60%（SimNPO）或 22%（IDKDPO），在 WMDP‑Bio 上可达 93%（IDK、AP）或 85%（UNDIAL），显著优于基线；

**⚠️ 局限性**

依赖 LLM 评判器可能漏判细微泄漏，导致 ASR 低估；演化搜索计算成本高；目前仅针对已知忘记集，尚未验证对所有模型或攻击策略的通用性；

---

## 136. SR4-Fit: An Interpretable and Informative Classification Algorithm Applied to Prediction of U.S. House of Representatives Elections

**arXiv ID:** 2602.06229 | [PDF](https://arxiv.org/pdf/2602.06229v1)

**作者:** Shyam Sundar Murali Krishnan `[一作]` (University of Oklahoma), Dean Frederick Hougen `[通讯]` (University of Oklahoma)

**通讯引用:** 1460 | [OpenAlex ID](https://openalex.org/A5061058579)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种可解释的分类算法SR4-Fit，用于预测美国众议院选举结果。

**💡 创新点**

创新点在于将稀疏松弛正则化回归与规则拟合相结合，既保持高预测性能，又生成稳定且易解释的规则集。

**🔧 技术方法**

技术核心包括稀疏正则化回归、规则拟合（RuleFit）以及特征组合与选择方法。

**📊 数据集**

主要使用美国人口普查局的美国社区调查（ACS）提供的选区人口统计特征数据，并在六个公开分类数据集（如乳腺癌、Ecoli、page blocks、Pima Indians、vehicle、yeast）上进行验证。

**📈 对比分析**

与随机森林等黑盒模型及RuleFit对比，SR4-Fit在准确率、简洁度与鲁棒性方面均表现更优，尤其在选举预测任务中显著提升预测精度。

**⚠️ 局限性**

局限性包括在大规模高维数据上计算成本较高，以及需要手动调参以获得最佳稀疏度。

---

## 137. Allocate Marginal Reviews to Borderline Papers Using LLM Comparative Ranking

**arXiv ID:** 2602.06078 | [PDF](https://arxiv.org/pdf/2602.06078v1)

**作者:** Elliot L. Epstein `[一作]` (Stanford University), Thanawat Sornwanee `[通讯]` (Stanford University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在评审开始前利用LLM对论文进行成对比较，构建Bradley–Terry排序，以识别处于接受边界附近的论文，并将多余的评审名额优先分配给这些论文。

**💡 创新点**

创新点在于将LLM生成的比较结果与Bradley–Terry模型结合，仅用于评审数量的动态分配，而非直接做接受/拒绝决策，从而在不改变最终人类评审结果的前提下最大化决策质量。

**🔧 技术方法**

使用长上下文LLM进行论文对比，解析JSON输出；利用Bradley–Terry模型拟合对比结果得到排名；通过边界重叠率ρ和额外评审增益Δ的期望公式评估策略效果。

**📊 数据集**

在ICLR 2025的1000篇提交论文上实验，使用公开的评审结果与分数作为验证数据集。

**📈 对比分析**

将该策略与随机分配基线对比，利用边界重叠率、额外评审增益以及期望净改进决策（(ρs−s²)NΔ）等指标评估；实验显示在边界区块宽度与评审额外比例匹配时，能够在数十项决策上实现预期提升。

**⚠️ 局限性**

局限性包括：评估依赖于回溯代理，缺乏因果验证；LLM排名可能带来偏差或被游戏；成本与可扩展性仍需进一步验证。

---

## 138. Personagram: Bridging Personas and Product Design for Creative Ideation with Multimodal LLMs

**arXiv ID:** 2602.06197 | [PDF](https://arxiv.org/pdf/2602.06197v1)

**作者:** Taewook Kim `[一作]` (Toyota Research Institute), Matthew Klenk `[通讯]` (Toyota Research Institute)

**通讯引用:** 666 | [OpenAlex ID](https://openalex.org/A5046977124)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了Personagram系统，一种利用多模态大型语言模型（MLLM）将用户画像与产品设计紧密连接的交互式AI工具，帮助设计师探索、提炼并重组基于画像的产品特征，实现快速原型生成；

**💡 创新点**

创新点在于：① 构建端到端的多模态推理流水线，将文字画像转换为产品视觉参考，再拆解为可操作的设计特征；② 设计结构化界面（灯泡、拼图等按钮），降低交互成本，提升透明度与信任；③ 将传统静态画像转变为动态、可操作的创意助手。

**🔧 技术方法**

使用技术包括：GPT‑4o‑mini进行文本推理；Google Image Search API检索产品图像；Flux T2I模型生成视觉原型；前端基于Next.js、ReactFlow实现Canvas与生产视图；界面中实现灯泡、拼图、加号等交互按钮。

**📊 数据集**

数据集：PERSONA（HuggingFace）——1000个基于美国人口普查的程序生成画像；产品图像来源于网络检索（Google Image Search）。

**📈 对比分析**

评估方法为受控的Within‑Subjects实验（12名专业产品设计师），对比Personagram与自由聊天基线。指标包括交互次数、CSI、NASA‑TLX、问卷Likert评分。结果显示：Personagram交互次数显著更高（p<0.01），prompt编辑时间更短，CSI及透明度评分更高，用户对工具的满意度与信任显著提升。

**⚠️ 局限性**

局限性包括：样本规模小且仅限物理产品设计师；任务时长短（30分钟），可能不反映真实工作流程；基线缺乏图像检索功能，导致对比不完全；未考察长期协作与多阶段工作流程；依赖网络检索的产品参考可能过时，影响设计效果。

---

## 139. Revisiting Salient Object Detection from an Observer-Centric Perspective

**arXiv ID:** 2602.06369 | [PDF](https://arxiv.org/pdf/2602.06369v1)

**作者:** Fuxi Zhang `[一作]` (Dalian University of Technology), Long Teng `[通讯]` (Dalian University of Technology)

**通讯引用:** 1252 | [OpenAlex ID](https://openalex.org/A5100371552)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Observer‑Centric Salient Object Detection (OC‑SOD)框架，并构建首个面向观察者视角的大规模数据集OC‑SODBench，随后设计了基于多模态LLM和SAM的代理式基线OC‑SODAgent。

**💡 创新点**

创新点在于：①将主观先验（偏好、意图）显式纳入SOD任务，解决传统单一ground‑truth导致的歧义与不确定性；②设计五步LLM驱动的数据标注管线，生成152k条文本指令–掩码对；③引入“Perceive–Reflect–Adjust”循环的代理式推理，让模型在推理过程中反复完善分割结果。

**🔧 技术方法**

技术手段包括：多模态大型语言模型（Qwen3‑VL、LLaVA等）用于图像理解、指令生成与质量验证；预训练分割模型SAMv2用于快速生成掩码；迭代反思机制实现多步推理；微调与零样本评估均使用Adam+Cosine学习率调度。

**📊 数据集**

使用的数据集：OC‑SODBench（33k图像，152k指令‑掩码对，包含free‑viewing、preference‑driven、intent‑driven三种模式）以及基于DUTS、LVIS、PACO‑LVIS、EgoObjects等现有数据集的预注释样本。

**📈 对比分析**

在free‑viewing模式下，OC‑SODAgent在未微调时已达到或超过主流SOD模型（如VST、ICON等），微调后gIoU达到89.13、cIoU 88.92、S_m 95.84等优异指标；在intent‑driven和preference‑driven模式下，OC‑SODAgent在零样本和微调场景均优于PixelLM、LISA等LLM分割方法，表现出更强的零样本泛化与个性化推理能力。

**⚠️ 局限性**

局限性包括：①依赖高性能LLM和SAM，模型推理成本相对较高；②对极小或高度遮挡的目标仍存在精度下降；③数据集仍未覆盖所有文化与语言背景的偏好与意图，导致跨语境的迁移性待验证。

---

## 140. The Condensate Theorem: Transformers are O(n), Not $O(n^2)$

**arXiv ID:** 2602.06317 | [PDF](https://arxiv.org/pdf/2602.06317v1)

**作者:** Jorge L. Ruiz Williams `[一作]` `[通讯]` (NaNZeta LLC), Jorge L. Ruiz Williams (NaNZeta LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文证明并验证了Transformer的自注意力在训练后天然形成稀疏拓扑结构，可通过Anchor+Window+Dynamic Top‑k精准采样实现与全矩阵相同的输出；

**💡 创新点**

核心创新是提出Condensate Theorem，证明稀疏采样可以实现100%精确等价而非近似，并将此原则在各种已训练模型中通用化；

**🔧 技术方法**

采用了Q·K^T得分作为动态选择依据，结合固定窗口和位置0“吸附子”，实现了无额外训练的稀疏软最大化；

**📊 数据集**

在公开的GPT‑2、Pythia、Qwen2、TinyLlama、Mistral等系列模型上进行了验证，使用标准长序列和多任务提示；

**📈 对比分析**

与Flash Attention进行对比，稀疏实现对131K token速度提升至159×，在1M token级别可望达1275×，同时保持100% token匹配和相同的cosine相似度；

**⚠️ 局限性**

局限性包括仅适用于已训练模型，随机或短序列训练的模型缺乏稀疏性；所需的窗口和Top‑k参数需针对不同模型调优。

---

## 141. PurSAMERE: Reliable Adversarial Purification via Sharpness-Aware Minimization of Expected Reconstruction Error

**arXiv ID:** 2602.06269 | [PDF](https://arxiv.org/pdf/2602.06269v1)

**作者:** Vinh Hoang `[一作]` (RWTH Aachen University), Raúl Tempone `[通讯]` (KAUST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于分数模型的确定性对抗净化方法（PurSAMERE），通过在噪声扰动下最小化期望重建误差并结合 sharpness‑aware 最小化来净化对抗样本；

**💡 创新点**

创新点在于：①把净化问题转化为在局部邻域内最小化期望重建误差，从而推动样本向数据分布的高密度峰值；②在净化过程中使用输入空间的 SAM，抵消分数模型近似误差导致的尖锐局部最小值；③保持整个过程确定性，避免随机性导致的有效鲁棒性下降；

**🔧 技术方法**

使用技术包括：分数匹配学习的分数网络、基于 Monte Carlo 的期望重建误差估计、输入空间的 SAM（sharpness‑aware minimization）、Adam 优化器、BPDA（backward pass differentiable approximation）以及可选的固定种子生成噪声；

**📊 数据集**

在 CIFAR‑10 图像分类数据集上进行实验；

**📈 对比分析**

与多种基线净化方法（DP_DDPM、DP_DDIM 等）以及对抗训练方法对比，采用 gPGD20、BPDA20‑det、BPDA200‑det、BPDA200+EoT20 等强攻击；在这些攻击下，PurSAMERE 的对抗准确率显著高于现有方法（例如在 BPDA20‑det 下从 39.16% 提升到 69.08%，在 BPDA200+EoT20 下平均 66.23%），同时保持较高的正常准确率；

**⚠️ 局限性**

局限性包括：①需要先训练分数模型，额外的训练成本；②净化过程仍依赖 Monte Carlo 近似，尽管使用固定种子可确定化，但在极大噪声尺度下仍可能引入微小随机误差；③目前仅在图像分类任务上验证，尚未证明可推广到其它模态或更大规模数据集；④对抗攻击仍采用有限步 PGD，未测试更高级的自适应攻击；

---

## 142. How Do Human Creators Embrace Human-AI Co-Creation? A Perspective on Human Agency of Screenwriters

**arXiv ID:** 2602.06327 | [PDF](https://arxiv.org/pdf/2602.06327v1)

**作者:** Yuying Tang `[一作]` (Hong Kong University of Science and Technology), Huamin Qu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10806 | [OpenAlex ID](https://openalex.org/A5091466289)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

进行了一项为期两周的定性研究，研究19名专业编剧如何主动采用AI进行五分钟短片剧本创作，并通过意图性、前瞻性、自我调节、自我反思四个维度分析其协作过程。

**💡 创新点**

从人类代理理论视角深入探讨创作者在持续交互中如何塑造、调节与AI共创实践，揭示创作者通过反思自我成长并形成新的协作范式，填补以往仅捕捉单一时点的空白。

**🔧 技术方法**

采用大型语言模型（DeepSeek Reasoner）与ChatGPT类接口进行AI创作伙伴，结合共创会话、事后思考录音和半结构化访谈等质性数据收集方式，并使用主题分析法对数据进行编码。

**📊 数据集**

19名专业编剧的五分钟短片剧本创作过程日志、聊天记录、思考录音与访谈记录；未使用公开数据集。

**📈 对比分析**

本研究为定性探索，不做量化比较；通过对比不同创作者在四个代理属性上的行为模式与发展轨迹，呈现实践中的多样化结果。

**⚠️ 局限性**

样本规模有限、仅聚焦短篇剧本、受样本群体文化背景同质化限制，且研究仅关注创作阶段，未涵盖制作与发布环节，未来需扩大样本、多样化情境及量化验证。

---

## 143. SVRepair: Structured Visual Reasoning for Automated Program Repair

**arXiv ID:** 2602.06090 | [PDF](https://arxiv.org/pdf/2602.06090v1)

**作者:** Xiaoxuan Tang `[一作]` (Ant Group), Yong Li `[通讯]` (Ant Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种多模态自动程序修复框架SVRepair，利用结构化视觉表征将截图和控制流图转换为语义场景图，再驱动编码代理完成缺陷定位与补丁生成；

**💡 创新点**

创新点在于：①通过视觉语言模型SVR统一将异构视觉信息映射为结构化语义场景图，填补视觉与代码之间的语义鸿沟；②引入迭代视觉分割策略，逐步缩小关注区域，去除噪声并降低模型幻觉；

**🔧 技术方法**

核心技术包括：视觉语言模型微调（SVR），基于Mermaid语法的语义场景图（SSG）；多模态编码代理（配合LLM进行定位与补丁生成）；视觉分割反馈循环；Docker化的测试验证环境；

**📊 数据集**

使用的数据集包括：WebSight（HTML代码与截图对），高星 GitHub 仓库的控制流图；Benchmarks：SWE‑Bench M、MMCode、CodeVision。

**📈 对比分析**

与多模态LLM（Claude、GPT‑4o、Qwen‑VL）及自治系统（GUIRepair、Refact Agent、OpenHands）对比，SVRepair在SWE‑Bench M、MMCode、CodeVision上分别取得 36.47%、38.02%、95.12% 的 Pass@1 率，均为目前最高；

**⚠️ 局限性**

局限性：目前仅支持 HTML 渲染和控制流图；对其他图形化工件需额外领域调优；迭代分割策略增加计算开销；Docker 环境可能受限于操作系统差异，无法完全复现所有真实环境依赖。

---

## 144. Self-Improving World Modelling with Latent Actions

**arXiv ID:** 2602.06130 | [PDF](https://arxiv.org/pdf/2602.06130v1)

**作者:** Yifu Qiu `[一作]` (University of Edinburgh), Edoardo M. Ponti `[通讯]` (University of Edinburgh)

**通讯引用:** 1599 | [OpenAlex ID](https://openalex.org/A5014613113)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将动作视为潜在变量，利用无标签状态序列自我迭代训练前向世界模型和逆向动力学模型，实现了LLM与VLM的内部世界建模。

**💡 创新点**

创新点在于将动作抽象为潜在变量，并通过交替优化的自我改进循环（FWM与IDM互为奖励），提供了无监督的世界建模方法并给出可学习性证明。

**🔧 技术方法**

采用变分信息最大化、证据下界最大化和GRPO强化学习框架，结合前向模型Pθ(y|x,z)和逆向模型Qϕ(z|x,y)。

**📊 数据集**

使用了多种无标签数据集，包括UCF-101、Movement-in-Times、Kinetics700、VIDGEN-1M等视频数据，以及科学仿真、Web交互、工具调用等文本环境数据。

**📈 对比分析**

在六大视觉与文本基准（Aurora-Bench、ByteMorph、WorldPredictionBench、StableToolBench等）上与SFT、Bootstrapping、Diffusion编辑器和大型VLMs对比，平均提升约15-30% 评估分数，并匹敌更大规模模型。

**⚠️ 局限性**

局限在于对共享参数的稳定性不足，缺乏对潜在偏见的安全过滤，且在动态控制和高阶动作描述上仍需进一步提升。

---

## 145. Private and interpretable clinical prediction with quantum-inspired tensor train models

**arXiv ID:** 2602.06110 | [PDF](https://arxiv.org/pdf/2602.06110v1)

**作者:** José Ramón Pareja Monturiol `[一作]` (Universidad Complutense de Madrid), Mohammad Kohandel `[通讯]` (University of Waterloo)

**通讯引用:** 2508 | [OpenAlex ID](https://openalex.org/A5087552186)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了临床机器学习模型（如逻辑回归和浅层神经网络）在公开部署时的训练集泄露风险，并提出通过量子启发的张量训练（TT）实现的隐私防护方案；

**💡 创新点**

创新点在于将模型参数张量化并通过随机化变换实现白盒隐私，同时在对输出进行离散化后保持黑盒隐私与差分隐私相当，并且保持甚至提升模型可解释性；

**🔧 技术方法**

采用张量网络中的张量训练（TT）以及TT‑RSS张量化方法，对离散化输出做分割并进行gauge随机化，结合shadow‑model成员推断攻击、差分隐私训练与深度学习框架进行对比实验；

**📊 数据集**

使用免疫治疗响应预测的公开数据集，包括Cho1、Cho2、MSK1、MSK2、Shim和Kato等六个临床/基因组数据集；

**📈 对比分析**

在成员推断攻击的Hamming分数和模型性能（准确率、AUC）上对比，发现TT模型在白盒攻击下接近随机猜测，黑盒攻击与低ε差分隐私相当，同时保持与原始模型相近的预测性能；

**⚠️ 局限性**

局限性包括对离散化bin数的调参需求、仅针对浅层神经网络和逻辑回归的评估、对更复杂模型的隐私保护尚待验证，以及跨验证聚合模型易放大泄露风险。

---

## 146. Quantifying Energy-Efficient Edge Intelligence: Inference-time Scaling Laws for Heterogeneous Computing

**arXiv ID:** 2602.06057 | [PDF](https://arxiv.org/pdf/2602.06057v1)

**作者:** Satyam Kumar `[一作]` (Dell Technologies), Saurabh Jha `[通讯]` (Dell Technologies)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 QEIL 框架，在异构边缘设备（CPU、GPU、NPU）上通过推理时间缩放律实现大语言模型的高效推理。

**💡 创新点**

创新点在于：① 五条与架构无关的推理缩放定理；② 能量感知的层级分配与动态异构调度；③ 统一的 IPW、ECE、PPP 三种多目标评估指标。

**🔧 技术方法**

采用理论推导、MLIR 编译、量化、贪心层分配、实时功耗监控等技术实现异构调度与能量优化。

**📊 数据集**

使用 WikiText‑103、SWE‑bench 等公开数据集进行覆盖率、能耗、延迟等评测。

**📈 对比分析**

与传统单设备或云推理基线相比，QEIL 在 125M–2.6B 参数的五大模型中实现了约 70% 的覆盖率提升、48% 的能耗降低、15% 的延迟下降、IPW 提升 4–6 倍、PPP 提升约 40%。

**⚠️ 局限性**

局限在于实验仅涵盖少量 Transformer 模型，未验证更大规模模型、跨节点通信开销以及多种 NPU 架构的通用性。

---

## 147. iScheduler: Reinforcement Learning-Driven Continual Optimization for Large-Scale Resource Investment Problems

**arXiv ID:** 2602.06064 | [PDF](https://arxiv.org/pdf/2602.06064v1)

**作者:** Yi-Xiang Hu `[一作]` (University of Science and Technology of China), Xiang-Yang Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18288 | [OpenAlex ID](https://openalex.org/A5100341802)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出iScheduler，一种基于强化学习的迭代分解调度框架，用于解决大规模资源投资问题（RIP）并支持动态重配置。

**💡 创新点**

将RIP的迭代分解建模为马尔可夫决策过程，学习过程选择策略和候选方案评分器，显著降低求解时间并提升资源利用率。

**🔧 技术方法**

采用强化学习（DQN+GNN）和图注意力网络（GATv2）进行状态编码，结合学习式解方案评估，并利用子问题的混合整数/约束规划求解。

**📊 数据集**

使用工业规模L‑RIPLIB基准，包含1000个2,500–10,000任务的实例，分为Easy/Normal/Hard三类进行训练与测试。

**📈 对比分析**

在相同时间预算下与MIP、CP、COpter、POP等传统优化器比较，iScheduler在Easy/Normal/Hard上实现资源成本最优且时间提升高达43×，在重配置场景中显著降低重配置延迟。

**⚠️ 局限性**

训练成本高、缺乏理论性能保证、以及当前实现仅串行调度子问题，未能充分利用并行。

---

## 148. Adaptive and Balanced Re-initialization for Long-timescale Continual Test-time Domain Adaptation

**arXiv ID:** 2602.06328 | [PDF](https://arxiv.org/pdf/2602.06328v1)

**作者:** Yanshuo Wang `[一作]` (Hong Kong Polytechnic University), Jie Hong `[通讯]` (Eastern Institute for Advanced Study)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种自适应平衡重置（ABR）策略，在长期持续测试时域适应中动态触发模型重置，以保持模型长期性能。

**💡 创新点**

创新点在于：①利用标签翻转曲线的急剧上升作为重置触发信号；②在重置时采用动态比例混合源模型权重与当前模型权重，实现平衡的“缩放‑恢复”重置。

**🔧 技术方法**

技术实现包括：标签翻转计算、指数移动平均平滑、基于斜率阈值的自适应触发、动态比例更新权重，全部在EATA框架上实现。

**📊 数据集**

实验使用三大持续域适应基准：CIN-C、CIN-3DCC以及CCC（Easy、Medium、Hard）数据集。

**📈 对比分析**

与BN、TENT、RPL、SLR、CPL、CoTTA、EATA、ETA、RDumb等方法对比，ABR平均精度为40.2%，比最佳基线RDumb提升2.3%，在更难的数据集上优势更突出。

**⚠️ 局限性**

局限性包括：对标签翻转曲线的稳定性和阈值设定有一定依赖；在极端噪声或快速域变换场景下可能效果不佳；兼容性需进一步在更多CTTA方法上验证。

---

## 149. Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding

**arXiv ID:** 2602.06161 | [PDF](https://arxiv.org/pdf/2602.06161v1)

**作者:** Yanzheng Xiang `[一作]` (King's College London), Yulan He `[通讯]` (King's College London)

**通讯引用:** 13663 | [OpenAlex ID](https://openalex.org/A5015709853)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 COVER 方法，通过单次前向推理实现上下文保持的验证，解决了 revocable 扩散解码中的 flip‑flop 振荡问题。

**💡 创新点**

创新点在于使用 KV 缓存覆盖和对角修正实现留一检验，同时设计稳定性感知种子选择与自适应修订率，显著减少无效重掩蔽并提升推理速度。

**🔧 技术方法**

技术核心包括离散扩散大语言模型、KV 缓存覆盖、对角修正、留一检验、稳定性感知种子评分与自适应修订阈值。

**📊 数据集**

在四个扩散大模型（LLaDA‑8B、LLaDA‑8B‑Instruct、LLaDA‑1.5‑8B、Dream‑7B‑Instruct）上使用 HumanEval、MBPP、GSM8K、MATH500 四个基准数据集进行评估。

**📈 对比分析**

与传统贪婪扩散解码、WINO 和 Saber 对比，COVER 在保持或提升准确率的同时将步骤数显著降低，速度提升最高可达 11.64×，在大多数任务上实现最优或接近最优表现。

**⚠️ 局限性**

局限性包括对扩散模型的依赖，稳定性代理在不同任务上的表现可能不一，且实现仍需额外内存管理与调参；在极端长序列或高复杂度任务中仍有进一步改进空间。

---

## 150. Statistical Learning from Attribution Sets

**arXiv ID:** 2602.06276 | [PDF](https://arxiv.org/pdf/2602.06276v1)

**作者:** Lorne Applebaum `[一作]` (Google Research), Aryan Mokhtari `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在隐私约束下，学习点击到转化的预测模型，利用只能得到的归因集合（attribution sets）进行无监督学习。

**💡 创新点**

提出了可无偏估计总体风险的解析式，并给出了基于归因集合先验分布的无偏风险估计器，进而得到能量泛化误差上界；同时证明该方法对先验估计误差鲁棒。

**🔧 技术方法**

核心技术包括：把二元损失拆分为常数项和标签项；利用归因集合的组合结构与已知先验得到可观测的无偏估计；使用 Rademacher 复杂度与覆盖数分析 ERM 的泛化误差；对先验误差给出改进的理论保证。

**📊 数据集**

在公开数据集上进行实验：MNIST（1-vs-rest）、CIFAR-10（动物vs机械）以及 Higgs（原始二分类）。

**📈 对比分析**

与行业常用的随机归因和最大先验归因两种基线比较，实验显示无偏估计器在归因集合尺寸增大或重叠较多时仍能保持较好性能，明显优于基线，尤其在大归因集合下差距明显。

**⚠️ 局限性**

局限性包括：需要先验分布的先验知识（虽可估计但仍假设一定准确）；假设归因集合大小固定且先验已知；对极端重叠或极小先验概率时理论与实验仍需进一步验证；未覆盖多触点归因或更复杂的归因逻辑。

---

## 151. Analyzing Diffusion and Autoregressive Vision Language Models in Multimodal Embedding Space

**arXiv ID:** 2602.06056 | [PDF](https://arxiv.org/pdf/2602.06056v1)

**作者:** Zihang Wang `[一作]` (NYU Shanghai), Chen Zhao `[通讯]` (NYU Shanghai)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

系统评估并对比了基于扩散模型和自回归模型的多模态嵌入，验证它们在分类、视觉问答与检索任务上的表现

**💡 创新点**

首次将多模态扩散语言模型转化为嵌入模型并进行全流程对比实验，揭示扩散模型在嵌入任务中的优势与瓶颈

**🔧 技术方法**

对比两类模型采用对比学习（InfoNCE）微调，使用VLM2Vec框架、mean pooling 与最后标记提取，结合扩散与自回归注意力结构

**📊 数据集**

使用32个公开数据集，涵盖分类（如ImageNet‑1K、N24News）、视觉问答（如OK‑VQA、DocVQA）和检索（如VisualNews、CIRR）等任务

**📈 对比分析**

通过对比微调后在多模态嵌入任务上的准确率与检索召回率评估，发现扩散模型总体落后于自回归模型，最差模型差距>20点，最优扩散模型（LLaDA‑V）仅落后3–5点；在跨域场景中扩散模型表现更稳健

**⚠️ 局限性**

仅评估了两款扩散模型，微调样本上限200k，未涉及最新DiffusionVL等模型，且未进行更大规模实验，可能隐藏更深层次性能差异

---

## 152. Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning (Extended Version)

**arXiv ID:** 2602.06227 | [PDF](https://arxiv.org/pdf/2602.06227v1)

**作者:** Pierriccardo Olivieri `[一作]` (Politecnico di Milano), Matteo Papini `[通讯]` (Università degli Studi di Milano)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于LTLfMT的奖励规范框架，允许使用一阶谓词与任意理论，消除了手工标签函数的需求。

**💡 创新点**

核心创新在于定义了可判定的lookahead‑free LTLfMT子语言，并将其与SMT求解器结合，同时将HER与奖励机（CRM）协同应用以缓解奖励稀疏问题。

**🔧 技术方法**

技术手段包括LTLfMT逻辑、SMT求解器（如Z3）、奖励机（Reward Machine）、HER、以及DDPG等强化学习算法。

**📊 数据集**

实验使用连续控制环境HighwayEnv中的停车（Parking）和Reacher任务，并在多条LTLfMT公式上进行测试。

**📈 对比分析**

与基线、单独CRM、单独HER进行比较，CRM‑HER在所有任务中均达到或超过最佳性能，特别是在任务复杂度较高时表现出显著优势。

**⚠️ 局限性**

局限性包括对可用SMT求解器的依赖、对量化与复杂时序表达式支持有限，以及在更大状态空间下的计算开销。

---

## 153. SOCKET: SOft Collison Kernel EsTimator for Sparse Attention

**arXiv ID:** 2602.06283 | [PDF](https://arxiv.org/pdf/2602.06283v1)

**作者:** Sahil Joshi `[一作]` (Rice University), Anshumali Shrivastava `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SOCKET框架，通过在稀疏注意力中使用软碰撞核估计（Soft Collision Kernel Estimator）来实现高效的键选取，从而在长上下文推理中实现更快、更稳定的稀疏注意力。

**💡 创新点**

核心创新在于将传统的硬LSH桶匹配替换为概率化的软LSH评分，保持了相似度信息并提升了排名稳定性；SOCKET是一种数据无关、理论可解释的稀疏注意力评分机制，并配备了高效的CUDA和Triton实现。

**🔧 技术方法**

使用技术包括：局部敏感哈希（LSH）、软碰撞概率计算、基于键归一化的top‑k选择、CUDA自定义核用于键评分、FlashDecode Triton核用于稀疏注意力计算。

**📊 数据集**

在长上下文基准上评测：LongBench、RULER‑32K；模型覆盖 Llama‑3.1‑8B‑Instruct、Llama‑3.2‑1B‑Instruct、Qwen3‑8B，支持最高 128K 上下文长度。

**📈 对比分析**

与 Quest、PQCache、HashAttention、MagicPig、Double Sparsity 等主流稀疏注意力方法对比，SOCKET 在 LongBench 与 RULER 上均能保持或超过竞争者的平均分；在 GPU 推理吞吐量上，比 FlashAttention 提升约 1.5×，尤其在极长上下文（>100K token）更显著。

**⚠️ 局限性**

局限性：仅在推理阶段使用稀疏注意力；对细粒度调优（fine‑tuning）和更大规模模型的评估尚未完成；软LSH 的温度与哈希表数量需要经验性调参；在极端稀疏或噪声较大的场景下，排名稳定性可能受限。

---

## 154. NanoNet: Parameter-Efficient Learning with Label-Scarce Supervision for Lightweight Text Mining Model

**arXiv ID:** 2602.06093 | [PDF](https://arxiv.org/pdf/2602.06093v1)

**作者:** Qianren Mao `[一作]` (Zhongguancun Laboratory), Philip S. Yu `[通讯]` (University of Illinois at Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向低标签、轻量化文本挖掘的统一框架，利用在线知识蒸馏、同伴互学习与参数高效微调构建小模型；

**💡 创新点**

创新点在于：1）将在线蒸馏与同伴互学习结合，实现极小模型的多教师学习；2）采用只更新偏置的BitFit技术实现极低训练参数；3）通过序列去填充、Flash Attention等实现训练与推理的显著加速；

**🔧 技术方法**

使用的技术包括：在线知识蒸馏、同伴互学习（Deep Mutual Learning）、参数高效微调（BitFit、冻结嵌入）、序列去填充、Flash Attention、RoPE、局部/全局注意力切换；

**📊 数据集**

在五个半监督文本分类基准上评估：AG News、Yahoo! Answers、DBpedia（LiteSSLHub）和 Amazon、Yelp（USB）等；

**📈 对比分析**

与多种基准方法（如UDA、FixMatch、FlexMatch、SimMatch等）以及完整模型和轻量化模型对比，所提框架在10/30/200标签下平均准确率可超过多数基线，且参数量和推理延迟显著降低（仅更新偏置、参数约1M，推理延迟0.5–1.0 s）；

**⚠️ 局限性**

局限性包括：仅针对分类任务，未验证在生成任务中的效果；对极低标签（10样本）下仍可能受噪声影响；并且同伴学习需要至少两台小模型，模型规模进一步缩减可能导致性能下滑。

---

## 155. What Is Novel? A Knowledge-Driven Framework for Bias-Aware Literature Originality Evaluation

**arXiv ID:** 2602.06054 | [PDF](https://arxiv.org/pdf/2602.06054v1)

**作者:** Abeer Mostafa `[一作]` (Hannover Medical School), Zahra Ahmadi `[通讯]` (Hannover Medical School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于论文评审报告的论文创新性评估框架，能够自动化生成与先前工作对比的创新评估报告。

**💡 创新点**

创新点在于：①利用近8万条人工评审报告训练模型，使其学习人类评审者的创新判定逻辑；②通过结构化知识抽取、语义检索和相似度图构建实现概念层面的对比；③在生成评估时同时给出校准的数值分数和基于先行文献的可解释说明。

**🔧 技术方法**

技术包括：大语言模型（Llama‑3.1‑8B‑Instruct）微调、指令式知识抽取、Semantic Scholar API 语义检索、结构化知识图构建、余弦相似度与Top‑k筛选、基于NLI和LLM评判器的质量评估。

**📊 数据集**

数据集为从 OpenReview 收集的 79,973 篇 NeurIPS 与 ICLR 2022 以后的评审报告，经过抽取、聚合得到 32,439 篇论文的创新性评价和归一化分数。

**📈 对比分析**

与多种基线（通用 LLM、域适配 LLM、Paper Reviewer、Open Reviewer）比较，在离散分数预测上取得 0.62 的准确率、0.76 的 Pearson 相关系数，显著高于其他模型；在文本说明一致性上获得最高的 entailment‑minus‑contradiction 与 LLM‑as‑judge 分数；在检测概念抄袭实验中，唯一得到 0 分的模型即为本框架。

**⚠️ 局限性**

局限性包括：检索依赖外部数据库的覆盖度；模型仅在 NeurIPS/ICLR 领域训练，难以推广至其他学科；若先行文献缺失，评估可能不完整。

---

## 156. MetaSSP: Enhancing Semi-supervised Implicit 3D Reconstruction through Meta-adaptive EMA and SDF-aware Pseudo-label Evaluation

**arXiv ID:** 2602.06163 | [PDF](https://arxiv.org/pdf/2602.06163v1)

**作者:** Luoxi Zhang `[一作]` (University of Tsukuba), Itaru Kitahara `[通讯]` (University of Tsukuba)

**通讯引用:** 4107 | [OpenAlex ID](https://openalex.org/A5056183585)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种用于单视角 3D 重建的半监督框架 MetaSSP，利用教师-学生 EMA 机制与 SDF 伪标签加权，借助 10% 标签热身后再利用大量无标签图像进行联合训练

**💡 创新点**

创新点在于：①基于梯度的参数重要性估计来正则化 EMA 更新，动态调节教师更新速率；②SDF 关注的伪标签加权机制，将数据增强一致性与 SDF 方差相结合，筛除噪声伪标签；③将两项技术结合的完整半监督管线，在少量标注下实现 SDF 表示的高质量重建

**🔧 技术方法**

技术包括：梯度导向参数重要性评估、动态 EMA（含元学习控制器）、SDF 伪标签加权、弱/强数据增强一致性约束、基于 SSR 的 SDF 网络

**📊 数据集**

使用 Pix3D 数据集，采用其标准 S1 切分，训练集 7,539 张图像，验证/测试集 2,530 张

**📈 对比分析**

与 MeanTeacher、FixMatch、SSP3D 等基线在 10% 标注下对比，MetaSSP 在 Chamfer Distance 上下降 20.61%（即 CD 从 3.93 降至 3.12），IoU 提升 24.09%（从 29.31% 提升至 36.37%），并在 F-Score、NC、PSNR 等指标上也均有显著提升

**⚠️ 局限性**

在标签极少或类别样本不足的情况下，伪标签仍可能带来噪声，导致部分类别性能下降；方法对伪标签阈值与增强策略敏感，未来需要更精准的置信度估计和针对性的数据增强

---

## 157. Large Language Model Reasoning Failures

**arXiv ID:** 2602.06176 | [PDF](https://arxiv.org/pdf/2602.06176v1)

**作者:** Peiyang Song `[一作]` (California Institute of Technology), Noah Goodman `[通讯]` (Stanford University)

**通讯引用:** 30935 | [OpenAlex ID](https://openalex.org/A5041689299)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大语言模型（LLM）在推理过程中的失败现象进行了系统综述，提出了双轴分类框架（推理类型 × 失败类型），对现有研究进行整理、根因分析与对策归纳，并公开了一个收集相关工作与数据的 GitHub 资源库。

**💡 创新点**

创新点在于：1）首次构建涵盖本体与非本体、形式与非形式推理，以及根本性、应用性与鲁棒性三类失败的两轴统一分类体系；2）通过案例对比和根因剖析，将不同领域的失败模式归纳为可互通的模式；3）提供了一个持续更新的公开工作仓库，方便后续研究。

**🔧 技术方法**

主要技术手段为文献检索与系统梳理，基于理论与实验报告的归纳与对比，结合现有的提示工程、检索增强、符号/神经混合等技术作为对策探讨；并未在实验层面实现新的模型或算法。

**📊 数据集**

未使用传统意义上的数据集；论文引用并整理了多种公开评测基准（如逻辑推理、数学推理、物理常识、多人代理任务、道德伦理任务等）以及对应的扰动与变换手段，但并未自行构建新数据集。

**📈 对比分析**

本文不涉及模型训练或性能比较，而是对已有工作进行综述。作者通过对比不同研究的失败案例、根因与对策，概括了目前研究进展与效果，但未给出统一的性能数值或基准分数。

**⚠️ 局限性**

主要局限包括：1）对根因分析尚不完整，部分失败模式缺乏内部机制解释；2）缺乏覆盖所有类型的统一长期评测基准，导致结果易被短期过拟合；3）多轮交互、真实场景中的推理失败案例研究不足；4）对策往往局部有效、缺乏通用性，鲁棒性提升仍有待进一步验证。

---

## 158. AdFL: In-Browser Federated Learning for Online Advertisement

**arXiv ID:** 2602.06336 | [PDF](https://arxiv.org/pdf/2602.06336v1)

**作者:** Ahmad Alemari `[一作]` (Jazan University), Cristian Borcea `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 2492 | [OpenAlex ID](https://openalex.org/A5074290416)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在浏览器端实现联邦学习框架 AdFL，收集广告与用户上下文信息并训练预测模型，实现隐私保护的个性化广告投放。

**💡 创新点**

①实现无需额外软件、标准浏览器 API 即可在浏览器中完成 FL；②支持任意模型与聚合算法；③集成差分隐私保护本地模型更新；④实现推理 <3 ms、训练 <400 ms、内存 <200 MB 的高效部署；⑤提供完整的端到端流水线。

**🔧 技术方法**

联邦学习（FedAvg）、TensorFlow.js、JavaScript、MutationObserver、IndexedDB、差分隐私、哈希/Min‑Max 归一化、Embedding、全连接网络、SHAP 重要性分析、AUC、准确率评估。

**📊 数据集**

10 天与 30 天两套非重叠网站日志，约 40 K 访客/日，含广告、页面、会话与用户特征，总计数十万条样本。

**📈 对比分析**

与集中式学习（CL）模型对比；FL 在 AUC 约 84–87%，准确率 73–79%，仅比 CL 略低 4–5%；桌面推理 <3 ms、移动 <7 ms；训练轮次 <400 ms/轮；内存 <200 MB；差分隐私噪声下 AUC >78%，通信成本约 3 MB/轮。

**⚠️ 局限性**

仅在单一出版商数据上验证；对广告特征高度依赖；未覆盖跨域广告、多设备用户与广告拦截器场景；多轮训练导致通信开销；缺乏对长期用户行为的持续跟踪。

---

## 159. LAAFD: LLM-based Agents for Accelerated FPGA Design

**arXiv ID:** 2602.06085 | [PDF](https://arxiv.org/pdf/2602.06085v1)

**作者:** Maxim Moraru `[一作]` (Los Alamos National Laboratory), Galen M Shipman `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于大型语言模型的 agentic 工作流（LAAFD），自动将通用 C 代码翻译成经过深度流水线、向量化、数据流划分等优化的 Vitis HLS 内核，并通过 HLS 反馈和共仿迭代不断降低周期数。

**💡 创新点**

创新点在于：① 将 LLM 作为“评判者”而非单纯生成器，利用 HLS 报告和功能验证来指导优化；② 通过多轮迭代实现接近理论最小周期的性能；③ 兼顾可读性与可维护性，生成的代码在行数和复杂度上优于现有 DSL（如 SODA）实现。

**🔧 技术方法**

技术核心包括：GPT‑5 / GPT‑5‑nano / o4‑mini LLM 作为翻译、校验、判断与优化 agent；Vitis HLS 2022.2 进行合成与共仿；基于 HLS 生成的资源与延迟报告作为反馈输入；对关键优化（流水线、向量化、数据流、窗口缓冲、循环展开、批处理等）进行系统化探索。

**📊 数据集**

数据集：15 个代表性 HPC 内核（包含单循环、嵌套循环、向量运算、4点/6点 stencil 等），以及从 SODA 框架提取的 7 个 stencil 基准，用以评估与手工调优和 SODA 生成的基线进行比较。

**📈 对比分析**

比较方法：将 LAAFD 生成的内核与手工优化的基线以及 SODA 生成的内核在理论最小周期、HLS 合成周期、资源利用率等指标上进行对比。实验结果显示 LAAFD 在 15 个内核上平均达到 99.9% 的几何平均性能，几乎与手工调优持平，且在 stencil 任务上与 SODA 性能相当；但资源占用略高，尤其是 BRAM 的使用。

**⚠️ 局限性**

局限性：① 需要 LLM 具备足够的推理深度，GPT‑5‑nano 与 o4‑mini 在复杂 stencil 或需要完美数据复用的内核上表现不佳；② 受限于上下文长度，较大的报告或完整应用可能导致信息截断；③ 目前仅针对周期优化，未显式考虑资源使用与功耗平衡；④ 结果具有随机性，需要多次运行以获得最优设计。

---

## 160. From Blurry to Believable: Enhancing Low-quality Talking Heads with 3D Generative Priors

**arXiv ID:** 2602.06122 | [PDF](https://arxiv.org/pdf/2602.06122v1)

**作者:** Ding-Jiun Huang `[一作]` (Institution1), Fernando De la Torre `[通讯]` (Institution2)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种针对动态3D头像的超分辨率方法，通过多视角和多表情的3D GAN反演实现高频细节的保留与时空一致性优化。

**💡 创新点**

创新点在于首次将超分辨率技术扩展到动态3D头像，利用多视角、多表情的GAN反演解决传统二维超分在多视角与时间轴上的不一致问题。

**🔧 技术方法**

核心技术包括GSGAN 3D GAN骨干、对FFHQ数据进行全头裁剪后细调、动态感知的多视角反演以及使用FVD、DOVER与ArcFace CSIM进行评估。

**📊 数据集**

主要使用数据集为裁剪后全头的FFHQ，以及NeRSemble、INSTA等视频数据进行验证。

**📈 对比分析**

方法与低分辨率GaussianAvatars、SplattingAvatar等基线对比，结果显示在时空质量（FVD、DOVER）上优于基线，同时保持较高的身份一致性，尽管CSIM略低于某些低分辨率基线。

**⚠️ 局限性**

限制在于3D GAN仅训练于前视人脸，难以合成头部后视，需构建包含后视的大规模人脸数据集以提升其生成能力。

---

## 161. InterFlow: Designing Unobtrusive AI to Empower Interviewers in Semi-Structured Interviews

**arXiv ID:** 2602.06396 | [PDF](https://arxiv.org/pdf/2602.06396v1)

**作者:** Yi Wen `[一作]` (Texas A&M University), Meng Xia `[通讯]` (Texas A&M University)

**通讯引用:** 28603 | [OpenAlex ID](https://openalex.org/A5100442292)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一款名为InterFlow的AI辅助半结构化访谈工具，提供可视化脚本导航、实时计时器和混合主动信息捕获，帮助访谈者管理访谈流程并实时解读访谈内容。

**💡 创新点**

创新点在于将AI嵌入访谈脚本的交互式可视化，实时动态适配访谈进程，同时以过程为导向提供分级的自动摘要与“共同访谈”提示，保持访谈者主动权且不分散注意力。

**🔧 技术方法**

技术实现包括React+Firebase前端、OpenAI GPT‑4o实时API、AssemblyAI语音识别与Speaker Diarization、spaCy句子分界、Dense检索与Embedding匹配、LLM‑as‑Judge评估建议、可视化计时与脚本层级等。

**📊 数据集**

使用自定义访谈脚本（来源于8篇近期研究）与12名访谈者/受访者的原始访谈数据进行评估；并未使用公开的标准数据集。

**📈 对比分析**

通过N=12的within‑subject实验与基线文本编辑器+聊天框对比，使用NASA‑TLX、SUS、可用性问卷等主观指标以及技术指标：问题检测准确率58%、延迟8.9s；即时摘要准确率77%、延迟4.2s；提示噪声率29%；专家评分建议质量2.21/3（vs基线2.66）。用户评价显示InterFlow降低认知负荷、提升可用性。

**⚠️ 局限性**

局限包括：问题检测准确率仍偏低、建议存在幻觉与错误、计时器更新不够细粒度、依赖OpenAI实时API导致隐私与可本地部署受限、系统在真实环境中的长期适应性与用户个性化需求尚未充分验证。

---

## 162. Generating High-quality Privacy-preserving Synthetic Data

**arXiv ID:** 2602.06390 | [PDF](https://arxiv.org/pdf/2602.06390v1)

**作者:** David Yavo `[一作]` (Laval University), Sadoune Ait Kaci Azzou `[通讯]` (Caisses Desjardins du Québec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种模型无关的后处理管线，利用模式补丁和HEOM–kNN距离过滤器，对现有表格生成器（如CTGAN、TVAE）生成的合成数据进行修正，以提升类别模式覆盖率、减少近邻泄露风险并保持下游任务性能；

**💡 创新点**

创新点在于：1) 采用层冻结的模式补丁策略，仅在缺失类别上细调生成器，避免全模型重训练；2) 引入HEOM–kNN ε_ANY距离过滤器，通过拒绝过于接近原始记录的样本，提供实用的距离‑基隐私屏障；3) 在统一的多指标评估框架下系统验证了该后处理对分布逼真度、模型可用性与隐私保护的平衡。

**🔧 技术方法**

使用的技术包括：基于统计的类别缺失检测与层冻结微调；Heterogeneous Euclidean Overlap Metric（HEOM）编码；k‑NN距离计算与基于阈值的采样拒绝；多维度相似度度量（JS距离、Cohen’s d、Pearson、Cramér’s V、η²）与Spearman秩相关；下游 TSTR 评估与属性推断攻击、RPR、CAP 等隐私指标。

**📊 数据集**

实验数据集包括三大公共混合型表格数据集：UCI Credit Card Default、Kaggle Cardiovascular Disease 与 UCI Adult Income，分别涵盖数值、分类与二元目标。

**📈 对比分析**

与未处理的基准相比，采用中等阈值（τ_ANY≈0.2–0.35）后，类别JS距离下降至20–50%，数值Cohen’s d降低约20–40%，多变量结构误差提升不超过10%；下游 TSTR 预测指标（准确率、AUC、F1）与未处理差异≤1%；隐私指标如RPR与CAP均显示轻微提升，属性推断攻击成功率基本不变。

**⚠️ 局限性**

局限性包括：1) 隐私提升幅度有限（3–4%），可能不足以显著降低攻击成功；2) L‑diversity 等隐私度量在不同阈值下表现不稳定；3) 仅在固定已训练模型上做后处理，未探索模型与过滤器协同训练的潜力；4) 评估仅覆盖三种数据集和简单攻击模型，泛化性待进一步验证。

---

## 163. Hermitian Self-dual Generalized Reed-Solomon Codes

**arXiv ID:** 2602.06377 | [PDF](https://arxiv.org/pdf/2602.06377v1)

**作者:** Chun'e Zhao `[一作]` (China University of Petroleum), Wenping Ma `[通讯]` (Xidian University)

**通讯引用:** 7251 | [OpenAlex ID](https://openalex.org/A5065027008)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究并完整证明了Hermitian自偶极子通用里德-森农（GRS）码在长度n≤q+1时仅存在两类；并给出了这两类码的显式构造方法。

**💡 创新点**

首次在理论上完成了Hermitian自偶GRS码的存在性判定和分类，解决了长期未解的开问题；同时提出两种全新的构造方案，可直接应用于实际编码。

**🔧 技术方法**

主要使用有限域代数、矩阵特征值与线性反馈移位寄存器理论，以及多项式求导、根式判定等工具，对GRS码的构造与自偶性进行严谨推导。

**📊 数据集**

论文为纯理论工作，没有使用具体数据集；构造以符号与有限域元素为基础。

**📈 对比分析**

由于是理论证明和构造，没有实验比较；但所给构造能在所有满足条件的长度与码率下产生Hermitian自偶GRS码，理论上满足最优距离特性。

**⚠️ 局限性**

局限性在于仅适用于长度不超过q+1的情况；对更长码长度无法直接应用；构造实现的复杂度和对符号选择的依赖仍需进一步研究。

---

## 164. ReBeCA: Unveiling Interpretable Behavior Hierarchy behind the Iterative Self-Reflection of Language Models with Causal Analysis

**arXiv ID:** 2602.06373 | [PDF](https://arxiv.org/pdf/2602.06373v1)

**作者:** Tianqiang Yan `[一作]` (Monash University), Yuan Gao `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 24736 | [OpenAlex ID](https://openalex.org/A5100722719)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型自我反思机制，提出ReBeCA框架来揭示自我反思中的因果驱动行为层级。

**💡 创新点**

通过三阶段Invariant Causal Prediction验证，首次将因果发现与语义模式结合，系统识别出可推广的自我反思因果父节点，揭示行为层级和非加性效应。

**🔧 技术方法**

采用语义模式编码、CESR一致性增强、Causal‑Learn因果发现、ICP三阶段筛选、线性稳定性检验与RCT干预实验等技术。

**📊 数据集**

使用Qwen3模型在MATH、BOUQuET（翻译）基准以及AIME数学题集进行实验。

**📈 对比分析**

与空集/全集基线比较CVLL提升达49.6%（MATH）或15.1%（BOUQuET）；干预实验显示单一因果模式提升显著，组合无增益。

**⚠️ 局限性**

固定重采样阈值无法自适应、语义模式手工定义可能漏检、缺乏真实因果图验证等限制。

---

## 165. Advances in Battery Energy Storage Management: Control and Economic Synergies

**arXiv ID:** 2602.06365 | [PDF](https://arxiv.org/pdf/2602.06365v1)

**作者:** Venkata Rajesh Chundru `[一作]` (Southwest Research Institute), Stanislav A Gankov `[通讯]` (Southwest Research Institute)

**通讯引用:** 378 | [OpenAlex ID](https://openalex.org/A5042227819)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统综述BESS的运营控制、经济调度、数字孪生及多服务收入堆叠等技术，提出了将电池退化与市场收益协同优化的整体框架。

**💡 创新点**

创新点在于：①将电池退化从单纯的技术限制转化为可量化的经济成本，融入调度优化；②阐述并聚合多领域（控制、经济、数字孪生）交叉的协同方法；③指出数字孪生可实现物理→数字→经济→物理闭环，弥合当前BESS盈利与技术安全的鸿沟。

**🔧 技术方法**

使用的主要技术包括：多目标优化（MILP、PSO、GA、鲁棒/随机优化）、强化学习（Q‑Learning、PPO）、退化感知模型、数字孪生架构（多尺度电化学/热模型、数据接口、分析引擎）以及实时预测维护与预报。

**📊 数据集**

本文并未构建自己的数据集，而是综述了多篇案例研究中使用的公开电网市场数据、充放电循环记录、BMS传感器日志和行业成本参数，强调需要标准化的数字孪生数据模型和接口。

**📈 对比分析**

对比方法：对比传统确定性调度、启发式/元启发式、鲁棒/随机优化与基于强化学习的自适应调度，指出后者在处理多服务收益和退化成本时能实现30%以上的运营效率提升，但未给出统一实验指标；文中通过文献综述展示了不同算法在利润、退化率和计算复杂度上的权衡。

**⚠️ 局限性**

局限性：①缺乏统一的数字孪生标准与数据格式，影响跨平台互操作；②现有辅助服务市场与电池特点不匹配，监管壁垒阻碍多服务收益堆叠；③高精度数字孪生与强化学习优化在秒级实时决策中的计算成本仍是技术瓶颈。

---

## 166. Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation

**arXiv ID:** 2602.06359 | [PDF](https://arxiv.org/pdf/2602.06359v1)

**作者:** Xiyang Zhang `[一作]` (Harbin Institute of Technology), Yan Song `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27980 | [OpenAlex ID](https://openalex.org/A5005228053)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于梯度正交性的“Orthogonal Gradient Selection (OGS)”数据筛选方法，用以在大语言模型的域适配过程中平衡专业化与保持通用能力，避免灾难性遗忘。

**💡 创新点**

创新点在于把梯度投影的几何安全性转移到数据选择阶段，利用轻量化的 Navigator 模型提取梯度几何特征，并通过强化学习动态决定训练样本，既保留了梯度 surgery 的理论优势，又消除了在线投影的高昂计算成本。

**🔧 技术方法**

核心技术包括：梯度正交性与冲突度度量、Navigator‑Target 代理架构、基于 PPO‑Lagrangian 的受限马尔可夫决策过程、以及 LoRA 参数高效微调。

**📊 数据集**

实验数据集涵盖三类垂直领域（MedQA、LegalBench、FinQA）和通用基准（GSM8K、MMLU、ARC‑C），anchor 集合由通用任务样本构成，用于计算梯度基准。

**📈 对比分析**

与随机挑选、完整数据、困惑度、影响函数、LESS、GrADS 等方法对比，OGS 在仅使用 10% 训练数据时即可达到或超过完整数据的域任务表现，并在通用基准上更好地保持性能，显著降低灾难性遗忘，且训练效率提升 2‑3 倍。

**⚠️ 局限性**

局限性包括：依赖梯度几何在不同规模模型间的转移假设，anchor 选取对结果影响较大，RL 策略训练增加额外开销，实验仅覆盖少量领域和模型，尚未验证在更大规模模型或更复杂任务上的泛化。

---

## 167. Investigating the structure of emotions by analyzing similarity and association of emotion words

**arXiv ID:** 2602.06430 | [PDF](https://arxiv.org/pdf/2602.06430v1)

**作者:** Fumitaka Iwaki `[一作]` (Tokyo Denki University), Tatsuji Takahashi `[通讯]` (Tokyo Denki University)

**通讯引用:** 261 | [OpenAlex ID](https://openalex.org/A5076653935)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在线问卷收集48个情感词对的相似度和关联度评价，构建情感语义网络。

**💡 创新点**

创新点在于利用相似度和关联度的双重语义网络来检验Plutchik情绪轮的结构有效性，并采用MDMC社区分解比较网络与情绪轮。

**🔧 技术方法**

采用MDMC（Markov链模块化分解）社区检测、MDS可视化以及NMI指标进行结构比较。

**📊 数据集**

使用由日本众包平台CrowdWorks收集的360（相似度）和229（关联度）名受试者在48个情绪词上的评估数据。

**📈 对比分析**

通过NMI与情绪轮的petal划分比较，发现相似度网络与轮结构的NMI为0.81，关联度网络为0.72，表明相似度网络更贴合轮模型。

**⚠️ 局限性**

局限包括仅使用日语数据、受试者为日本众包工人可能存在文化偏差、仅考察48个词且未跨语言验证。

---

## 168. Bridging the Indoor-Outdoor Gap: Vision-Centric Instruction-Guided Embodied Navigation for the Last Meters

**arXiv ID:** 2602.06427 | [PDF](https://arxiv.org/pdf/2602.06427v1)

**作者:** Yuxiang Zhao `[一作]` (Alibaba Group), Mu Xu `[通讯]` (Alibaba Group)

**通讯引用:** 457 | [OpenAlex ID](https://openalex.org/A5100532751)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 BridgeNav 任务与框架，实现从户外到室内的无先验指令驱动的具身导航。

**💡 创新点**

结合潜在意图推断与光流引导的动态感知，利用可学习初始令牌实现视觉中心化决策，并首次构建 BridgeNavDataset。

**🔧 技术方法**

采用 Vision Transformer + Qwen2.5‑VL‑3B 交叉注意力、潜在意图回归、RAFT 光流估计、轨迹引导的视频生成与 Plücker 编码等技术。

**📊 数据集**

收集 55K 真实街景图像并通过轨迹引导视频生成构建的 55k 轨迹‑指令对 BridgeNavDataset。

**📈 对比分析**

在 BridgeNav 数据集上与 NoMaD、Citywalker、OmniNav 复现对比，BridgeNav 在 0.1m、0.2m、0.3m 的成功率分别提升至 33.8%、70.1%、89.6%，导航效率平均降低 31% 以上。

**⚠️ 局限性**

模型仍受限于单帧视觉、对遮挡与多目标辨识的鲁棒性不足，且需较大算力与预训练模型。

---

## 169. POPL-KF: A Pose-Only Geometric Representation-Based Kalman Filter for Point-Line-Based Visual-Inertial Odometry

**arXiv ID:** 2602.06425 | [PDF](https://arxiv.org/pdf/2602.06425v1)

**作者:** Aiping Wang `[一作]` (Beihang University), Hai Zhang `[通讯]` (Beihang University)

**通讯引用:** 10060 | [OpenAlex ID](https://openalex.org/A5100724090)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了点线共同使用pose‑only表示的视觉惯性里程计POPL‑KF，利用姿态唯一表示消除特征坐标带来的线性化误差并实现即时更新，同时引入基准帧选择和线特征剔除策略。

**💡 创新点**

①点线同时采用pose‑only几何表示，首次在滤波式VIO中实现点线pose‑only；②统一基准帧选择算法；③基于网格+双向光流一致性的线特征剔除方法。

**🔧 技术方法**

多状态约束卡尔曼滤波、姿态唯一表示理论、EDLines线特征检测、双向光流一致性、基准帧选取（视差/角度）以及实时滤波实现。

**📊 数据集**

EuRoC、KAIST公开数据集，以及作者自行收集的室内校园和室外GNSS/IMU数据。

**📈 对比分析**

与OpenVINS、PO‑KF、PO‑KF w/SP、PL‑VINS、EPLF‑VINS等方法在ATE/ARE指标上进行对比，POPL‑KF在EuRoC、KAIST及实测场景均优于对比方法，误差下降约30%~70%，并保持30Hz以上实时率。

**⚠️ 局限性**

仍受限于特征匹配，极端低纹理或强动态环境下性能可能下降；线特征提取速度较高；未探索UKF或更强非线性优化提升。

---

## 170. Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding

**arXiv ID:** 2602.06412 | [PDF](https://arxiv.org/pdf/2602.06412v1)

**作者:** Daisuke Oba `[一作]` (Institute of Science Tokyo), Naoaki Okazaki `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3519 | [OpenAlex ID](https://openalex.org/A5066940046)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SureLock 方法，允许在掩码扩散语言模型中对已收敛的 token 位置进行锁定，跳过其 query 投影和前馈子层，从而降低每一步的计算量。

**💡 创新点**

创新点在于使用局部 KL 作为锁定判据，能够在保持终端 log‑prob 误差可控的前提下实现激活位置的主动收缩，产生单调递减的 per‑step FLOPs；同时提供了理论误差上界。

**🔧 技术方法**

主要技术包括位置级别的自注意力与前馈子层截断与 K/V 缓存、局部 KL 阈值检测与置信门判定，以及对终端概率误差的闭式上界分析。

**📊 数据集**

实验数据集为 LLaDA‑8B‑Base / LLaDA‑8B‑Instruct（8B 参数）模型，在 WikiText‑103（语言建模）和 MT‑Bench（指令跟随）上进行评估。

**📈 对比分析**

与不使用锁定的 Baseline 以及仅减少步数或重用 K/V 的方法对比，SureLock 在保持生成质量（PPL、MT‑Bench 分数基本不变）的同时，使算法 FLOPs 降至 30–50% 级别，并显著提升 TPS；与其他稀疏化方法组合可进一步加速。

**⚠️ 局限性**

局限性包括仅以算法 FLOPs 为指标，未完全体现硬件实现的 irregular 访问开销；KL 阈值需手动设定，短输出场景下可能导致质量下降；理论保证基于离散后验，尚未覆盖连续或更复杂的不确定性情形。

---

## 171. Point Virtual Transformer

**arXiv ID:** 2602.06406 | [PDF](https://arxiv.org/pdf/2602.06406v1)

**作者:** Veerain Sood `[一作]` (Indian Institute of Technology Tirupati), Gaurav Pandey `[通讯]` (Texas A and M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Point Virtual Transformer，结合真实 LiDAR 与虚拟点以提升远距物体检测

**💡 创新点**

首次系统评估多种融合策略（早期融合、BEV 门控融合、后期融合）并证明早期融合在远距场景可有效提升几何信息

**🔧 技术方法**

使用深度完成生成虚拟点、voxel 化、稀疏子曼平面卷积构建 BEV 热图、基于 Transformer 的查询与上下文聚合、投票回退与 3D 损失

**📊 数据集**

在 KITTI 3D/2D 车辆检测数据集上进行评测

**📈 对比分析**

与现有 1D/2D/Transformer 检测器对比，PointViT‑V1 在 KITTI 3D AP 91.16%、BEV AP 95.94%、2D AP 99.36%（V2 更细化至 96.56%/97.04%/88.97%）

**⚠️ 局限性**

虚拟点生成的不均匀性导致远距物体位置、朝向误差，且过多虚拟点仍会增加计算负担

---

## 172. FMBench: Adaptive Large Language Model Output Formatting

**arXiv ID:** 2602.06384 | [PDF](https://arxiv.org/pdf/2602.06384v1)

**作者:** Yaoting Wang `[一作]` (Fudan University), Henghui Ding `[通讯]` (Fudan University)

**通讯引用:** 4037 | [OpenAlex ID](https://openalex.org/A5036631624)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 FMBench 这一专门针对 Markdown 输出格式化的基准，并通过监督微调（SFT）与强化学习（GRPO）相结合的轻量级后训练流程提升模型在语义与结构两方面的表现。

**💡 创新点**

创新点在于：1) 设计了完整的 Markdown 结构化数据生成管线和评测指标；2) 采用可验证的结构奖励（结构化摘要 BERTScore）与语义奖励相平衡的 RL 目标，避免单纯依赖硬约束；3) 在多种模型规模上系统验证了 SFT 与 RL 的协同提升。

**🔧 技术方法**

使用的技术包括：LoRA 参数高效微调、GRPO 强化学习框架、BERTScore 语义/结构评估、Longformer 编码器用于长文档奖励、以及基于抽象摘要的结构相似度计算。

**📊 数据集**

数据集为 FMBench，包含 1,100 篇高质量 Markdown 文档，分 800 训练 300 测试，涵盖学术、官方、技术等八类原始文本，并按难度级别生成多样化结构。

**📈 对比分析**

与预训练模型比较，SFT+GRPO 在语义分数从约 0.93–0.95 提升至 0.94–0.97，结构分数从 0.95–0.97 提升至 0.96–0.98，尤其在大规模模型（如 Qwen3‑8B）表现最佳，验证了方法的有效性。

**⚠️ 局限性**

局限性包括：数据集结构范围有限，未覆盖所有真实写作风格；评估以单一参考文本为基准，忽略多种合法格式；奖励设计依赖预训练编码器，可能遗漏细粒度格式错误；未考虑多语言或跨域泛化。

---

## 173. A Consistency-Improved LiDAR-Inertial Bundle Adjustment

**arXiv ID:** 2602.06380 | [PDF](https://arxiv.org/pdf/2602.06380v1)

**作者:** Xinran Li `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Xudong Zou `[通讯]` (Aerospace Information Research Institute, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于立体投影参数化的LiDAR‑惯性Bundle Adjustment（BA），并在最大后验（MAP）框架下结合First‑Estimate Jacobians（FEJ）实现一致性改进，最终将该BA方法嵌入实时LiDAR‑惯性里程计（LIO）系统。

**💡 创新点**

创新点在于：
• 采用立体投影（Stereographic Projection）对平面与边缘特征进行无奇异、可微分的参数化，解决传统Plücker或最接点表示的奇异性与可观测性破坏；
• 在MAP公式中使用FEJ保持雅可比矩阵在优化过程中不随线性化点漂移，保证估计量的可观测性与协方差一致性；
• 对该参数化与FEJ结合的可观测性进行系统分析，提供闭式可观测向量表达。

**🔧 技术方法**

技术方法包括：
• 立体投影特征参数化与对应的雅可比导数；
• LiDAR‑惯性状态传播与观测模型；
• MAP式滑动窗口BA（含IMU约束、LiDAR点残差、先验约束）；
• FEJ线性化点维护与Schur补消除；
• 前端特征提取、局部特征图、闭环检测与全局图优化。

**📊 数据集**

实验数据集：
• KITTI 原始 LiDAR + IMU 数据序列（城市与高速公路场景）；
• Oxford RobotCar 或 NCLT 等室内/室外 LiDAR‑IMU 数据集（若论文中未给出具体可自行补充）。

**📈 对比分析**

方法比较：
• 与基准 LiDAR‑惯性 SLAM（如 LOAM、BALM2、π‑LSAM）及传统无一致性约束的 BA 进行对比；
• 评价指标包括里程计误差（AUC）、轨迹漂移、估计协方差的可信度；
• 结果显示该方法在长轨迹下漂移显著下降，估计协方差更符合真实误差分布，整体性能优于现有方法。

**⚠️ 局限性**

局限性：
• 需要对立体投影参数、阈值 δ 进行经验调参；
• 与传统参数化相比，计算量略大，实时性能受限于滑动窗口大小；
• 仅验证了平面与边缘特征的可观测性，对更复杂几何（如曲面）未做推广；
• 依赖高质量 LiDAR 点云，低点云密度或强噪声环境下性能下降。

---

## 174. Bilingual Bias in Large Language Models: A Taiwan Sovereignty Benchmark Study

**arXiv ID:** 2602.06371 | [PDF](https://arxiv.org/pdf/2602.06371v1)

**作者:** Ju-Chun Ko `[一作]` (National Taiwan University), Ju-Chun Ko `[通讯]` (National Taiwan University)

**通讯引用:** 335 | [OpenAlex ID](https://openalex.org/A5001727495)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并执行了双语基准测试，评估17种大型语言模型在中文与英文提问关于台湾主权时的回答一致性与偏差。

**💡 创新点**

首次提出 Language Bias Score (LBS) 与 Quality‑Adjusted Consistency (QAC) 两个量化指标，用以系统度量双语政治偏差；发现中文来源模型普遍存在统一审查且部分西方模型中文表现逊于英文。

**🔧 技术方法**

采用 OpenRouter API 调用模型，手工编写10条中英文对照提示，并结合红旗关键词检测与人工评审；使用 McNemar 检验评估统计显著性。

**📊 数据集**

使用10条双语提示集（扩展自原台湾主权基准）和模型生成文本作为评估样本；未使用公开大型语料库。

**📈 对比分析**

通过得分（/10）、一致率、LBS 与 QAC 进行比较；结果显示仅 GPT‑4o Mini 在中英文均达 10/10，其他旗舰模型均低于阈值，中文来源模型普遍失分。

**⚠️ 局限性**

局限包括：仅用10条提示缺乏统计功效；仅测试云端接口，未检验本地部署行为；评判依赖人工主观；未区分简体与繁体中文；评估者与被评估模型同属 Claude 家族，可能产生自评偏差。

---

## 175. Towards Adaptive Environment Generation for Training Embodied Agents

**arXiv ID:** 2602.06366 | [PDF](https://arxiv.org/pdf/2602.06366v1)

**作者:** Teresa Yeo `[一作]` (Singapore-MIT Alliance for Research and Technology Centre), Archan Misra `[通讯]` (Singapore Management University)

**通讯引用:** 10606 | [OpenAlex ID](https://openalex.org/A5054849647)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种闭环环境生成框架，利用大语言模型（LLM）分析代理在现有环境中的轨迹，并根据分析结果程序化地修改环境，使之逐步变得更具挑战性。

**💡 创新点**

创新点在于：① 用LLM进行细粒度轨迹分析，捕捉成功/失败、行为缺陷及改进建议；② 将分析结果反馈给环境生成器，实现环境难度随代理能力动态调整；③ 结合碰撞检测和物理可行性约束，保证生成环境的合理性。

**🔧 技术方法**

采用的技术包括：LLM（GPT‑4.1‑mini）用于轨迹分析（F）和环境修改（G）；可控的结构化环境表示（scene graph）；ProcThor 作为可编辑的模拟器；碰撞检测与离散步进位移动算法保证物理一致性。

**📊 数据集**

使用的数据集包括：① 预训练的导航代理模型 Spoc（在数百万条专家轨迹上训练）；② ProcThor 提供的基准室内场景；此外在实验中使用了 OpenAI API 进行LLM调用。

**📈 对比分析**

目前仅做了 proof‑of‑concept 的演示，未与其他基线方法进行系统对比，缺乏完整的性能评估；展示的示例表明生成的环境在视觉上比随机扰动更合理、挑战性更具针对性。

**⚠️ 局限性**

主要局限：LLM 的空间推理能力不足，可能产生空间上不连贯或不符合目标意图的修改；生成过程依赖多轮迭代和人工验证；缺乏大规模实验和训练代理在自适应环境下的最终性能评估。

---

## 176. SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass

**arXiv ID:** 2602.06358 | [PDF](https://arxiv.org/pdf/2602.06358v1)

**作者:** Yewei Liu `[一作]` (Peking University), Muhan Zhang `[通讯]` (Peking University)

**通讯引用:** 4805 | [OpenAlex ID](https://openalex.org/A5071515223)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种可在单次前向传递中将上下文映射为LoRA适配器的可扩展内部超网络 SHINE，实现无梯度调优的快速 LLM 适配。

**💡 创新点**

创新点在于无瓶颈 Transformer 结构的 M2P 超网络以及内置 Meta LoRA 的上下文记忆抽取，利用 LLM 自身表征实现全层高维参数生成。

**🔧 技术方法**

主要技术包括 Transformer 超网络、LoRA 适配器生成、Meta LoRA 记忆嵌入、预训练 + 指令微调、稀疏行列注意力。

**📊 数据集**

使用 6 亿词的 TransMLA 预训练语料，结合 MS MARCO MQA、SQuAD、HotpotQA、MuSiQue、2WikiMultihopQA 等多任务数据集进行微调。

**📈 对比分析**

与 Naive、In‑Context、SFT、Generative Adapter、SEAL 等基线对比，SHINE 在多轮 QA、单轮 QA 等任务上可与 In‑Context 相当，显著优于 SFT，且生成速度与内存消耗更低。

**⚠️ 局限性**

局限在于多轮会话时性能下降，且对长上下文的后置训练仍需要迭代优化，未来需进一步提升对多轮推理与长文本的适配。

---

## 177. Evaluating LLM-persona Generated Distributions for Decision-making

**arXiv ID:** 2602.06357 | [PDF](https://arxiv.org/pdf/2602.06357v1)

**作者:** Jackie Baek `[一作]` (New York University), Will Ma `[通讯]` (Columbia University)

**通讯引用:** 908 | [OpenAlex ID](https://openalex.org/A5070149920)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在缺乏历史数据的情况下，本文提出使用大语言模型生成分布，再通过样本平均逼近（LLM‑SAA）做决策；

**💡 创新点**

创新点在于构建面向决策的评价指标（极端与平均竞争比），并推导出三类典型问题（组合、定价、Newsvendor）的最坏情况竞争比计算方法；

**🔧 技术方法**

采用 GPT‑4o、GPT‑5‑mini、Gemini‑3 Flash、Mistral‑Large‑3 等 LLM 进行多种提示（采样、角色模拟、批量生成、描述），并结合 SAA 与竞争比分析；

**📊 数据集**

使用三份真实数据集：日本寿司偏好调查、菲律宾可可醇愿付价格、H&M 时装零售周度需求；

**📈 对比分析**

通过与随机基线、少量真实样本、基于流行度的基线等对照，利用决策感知竞争比进行评估，LLM‑SAA 在低数据环境下显著优于随机基线，在部分情形下接近真实样本表现；

**⚠️ 局限性**

局限性包括对提示设计未做系统优化、仅涵盖三类决策问题、角色模拟对个体准确性不足、以及实验环境与真实业务部署可能存在差异。

---

## 178. TrailBlazer: History-Guided Reinforcement Learning for Black-Box LLM Jailbreaking

**arXiv ID:** 2602.06440 | [PDF](https://arxiv.org/pdf/2602.06440v1)

**作者:** Sung-Hoon Yoon `[一作]` (Harvard University), Mengyu Wang `[通讯]` (Harvard University)

**通讯引用:** 2604 | [OpenAlex ID](https://openalex.org/A5100632182)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于历史信息的强化学习框架，用历史增强的状态和注意力机制来改进黑盒LLM的jailbreak攻击。

**💡 创新点**

创新点在于：①将交互历史嵌入状态空间，②使用注意力自适应加权历史步骤以突出关键漏洞，③在黑盒攻击中显著提升成功率和查询效率。

**🔧 技术方法**

技术手段包括：PPO强化学习、MLP策略网络、注意力机制、LLM辅助执行prompt变异器、以及多特征奖励与响应特征提取。

**📊 数据集**

使用的数据集为AdvBench和HarmBench，用于评估攻击成功率和查询效率。

**📈 对比分析**

与四大基线（PAIR、RLbreaker、AutoDAN‑Turbo、FlipAttack）在四个最新LLM上进行比较，实验显示本方法在ASR上遥遥领先、QPS显著更低，几乎达到100%成功率并显著降低查询次数。

**⚠️ 局限性**

局限性包括：动作空间仅包含少量预定义的mutator，可能限制攻击多样性；框架仍受限于RLbreaker的整体结构。

---

## 179. Envy-Free Allocation of Indivisible Goods via Noisy Queries

**arXiv ID:** 2602.06361 | [PDF](https://arxiv.org/pdf/2602.06361v1)

**作者:** Zihan Li `[一作]` (National University of Singapore), Warut Suksompong `[通讯]` (National University of Singapore)

**通讯引用:** 3177 | [OpenAlex ID](https://openalex.org/A5081017897)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在两个代理人之间，利用加性高斯噪声查询来求得无嫉妒（envy‑free）分配。

**💡 创新点**

首次给出在噪声查询模型下，针对两代理人情况，精确的查询复杂度上界与下界，证明最优查询量为Θ(m^{2.5}/Δ^{2})（Δ为最优负嫉妒度）。

**🔧 技术方法**

使用非自适应均匀查询、阈值分配算法、置信区间与高斯尾部推导、以及多假设检验等多臂老虎机理论工具。

**📊 数据集**

无具体数据集，全部为理论构造与随机实例。

**📈 对比分析**

通过理论证明与对比上界/下界，展示在Δ≫m^{1/4}时上界与下界相匹配，说明算法在查询量最优；在实验上未做实际比较。

**⚠️ 局限性**

仅适用于两代理人、加性效用、区间[0,1]、固定噪声方差且Δ≥m^{1/4}；对更小Δ、更多代理人、非加性效用或非高斯噪声的情况仍需进一步研究。

---

## 180. Beyond Code Contributions: How Network Position, Temporal Bursts, and Code Review Activities Shape Contributor Influence in Large-Scale Open Source Ecosystems

**arXiv ID:** 2602.06426 | [PDF](https://arxiv.org/pdf/2602.06426v1)

**作者:** S M Rakib Ul Karim `[一作]` (University of Missouri), Sean Goggins `[通讯]` (University of Missouri)

**通讯引用:** 2817 | [OpenAlex ID](https://openalex.org/A5037107679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统性地分析了云原生计算基金会生态系统中25年 OSS 贡献者网络，揭示影响力集中、角色分化与网络演化规律。

**💡 创新点**

首次结合 GPU 加速图神经网络、时序 LSTM 与结构完整性仿真，对 25 年 OSS 贡献者网络进行纵向动力学与影响力预测，量化了角色对网络韧性的贡献。

**🔧 技术方法**

GPU 加速的图卷积网络 (GCN) 进行角色分类，LSTM 预测活动突发，GPU 加速的 PageRank、介数中心性计算以及多元线性回归与变化点检测。

**📊 数据集**

Cloud Native Computing Foundation (CNCF) 生态系统的 25 年时间序列数据，涵盖 Sandbox、Incubating、Graduated 三个成熟度阶段，约 100,000+ 贡献者与 150+ 仓库。

**📈 对比分析**

与传统静态网络分析和单指标评价方法比较，GCN 角色分类准确率 84.3%、F1 0.79；LSTM 活动预测 MAPE 15.2%，召回率 62%；回归模型 R² 0.74，显著优于单一指标。

**⚠️ 局限性**

仅使用 GitHub 活动日志，忽略邮件、即时聊天等非代码协作；聚焦 CNCF 生态，缺乏跨 OSS 社区的泛化性；时间聚合到季度，可能掩盖细粒度协作动态；角色划分存在重叠与边界模糊。

---

## 181. EEG Emotion Classification Using an Enhanced Transformer-CNN-BiLSTM Architecture with Dual Attention Mechanisms

**arXiv ID:** 2602.06411 | [PDF](https://arxiv.org/pdf/2602.06411v1)

**作者:** S M Rakib UI Karim `[一作]` (University of Missouri), Sean Goggins `[通讯]` (University of Missouri)

**通讯引用:** 2817 | [OpenAlex ID](https://openalex.org/A5037107679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了基于EEG的情绪识别，提出并验证了一种融合CNN、双向LSTM和Transformer自注意力的增强Hybrid模型

**💡 创新点**

创新点在于将残差CNN、双层BiLSTM与双头自注意力相结合，并加入多重正则化（dropout、层归一化、标签平滑、权重衰减）实现高精度与可解释性兼备

**🔧 技术方法**

采用的技术包括1D CNN、双向LSTM、Transformer多头自注意力、残差连接、dropout、层归一化、标签平滑、L2权重衰减、AdamW优化器+余弦退火调度、EEG特征归一化和噪声增强

**📊 数据集**

使用了Kaggle公开的EEG情绪数据集（2529个样本，988维特征，分为中性、正向、负向三类情绪）

**📈 对比分析**

通过5折交叉验证和hold‑out测试与随机森林、SVM、MLP、Transformer‑CNN‑BiLSTM等基线对比，测量准确率、精确率、召回率、F1以及置信区间；增强模型在测试集上达到99.19%准确率，训练‑测试差距仅0.56%，显著优于基线（提升约2–3%）并通过Friedman+Wilcoxon验证统计显著性

**⚠️ 局限性**

局限性包括仅使用单一数据集，离散三类情绪设置不具备连续性和多维性；训练资源需求较大，未验证实时推理和跨域泛化；解释性仍为相关性而非因果，且对噪声和非稳态信号的鲁棒性需进一步提升

---

## 182. Empirical Analysis of Adversarial Robustness and Explainability Drift in Cybersecurity Classifiers

**arXiv ID:** 2602.06395 | [PDF](https://arxiv.org/pdf/2602.06395v1)

**作者:** Mona Rajhans `[一作]` (Palo Alto Networks), Vishal Khawarey `[通讯]` (Quicken Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在网络钓鱼URL分类和网络入侵检测任务中，系统评估了FGSM与PGD攻击下模型鲁棒性与解释性漂移，并提出鲁棒性指数（RI）衡量指标。

**💡 创新点**

创新点包括：①引入鲁棒性指数以量化模型对L∞攻击的抵抗；②将梯度特征敏感度与SHAP归因漂移结合，揭示鲁棒性与可解释性损失的内在关联；③在两个不同结构化数据集上进行跨域验证。

**🔧 技术方法**

使用技术包括：PyTorch实现的多层感知器；FGSM、PGD对抗攻击；梯度特征敏感度分析；SHAP归因漂移评估；对抗训练。

**📊 数据集**

使用数据集：Kaggle公开的Phishing Websites（11000条URL，30维特征）和UNSW-NB15（175000条网络流，42维特征）。

**📈 对比分析**

比较方法：在10个不同扰动预算下绘制准确率–扰动曲线，计算RI；通过对比基线与对抗训练模型的RI、准确率以及SHAP漂移。实验表明对抗训练将RI提升约9%且保持近似清洁数据准确率。

**⚠️ 局限性**

局限性：仅评估了可微分的MLP模型；对抗扰动假设为可控的L∞数值特征，未考虑真实攻击者对特征的操纵约束；未探讨树模型或大规模多模态数据的鲁棒性。

---

## 183. Uniqueness is Separation

**arXiv ID:** 2602.06386 | [PDF](https://arxiv.org/pdf/2602.06386v1)

**作者:** Liam O'Connor `[一作]` (University of Edinburgh), Christine Rizkallah `[通讯]` (University of Melbourne)

**通讯引用:** 392 | [OpenAlex ID](https://openalex.org/A5019489999)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文讨论了Cogent语言的唯一性类型系统，并证明可以用分离逻辑（Separation Logic）自然地表达其对C代码的框架条件，从而实现对Cogent与C混合系统的安全性验证。

**💡 创新点**

创新点在于将唯一性类型的三条框架条件（泄漏自由、新建内存、惯性）合并为单一的Hoare三元组，并借助分离逻辑的帧规则实现局部推理，避免了传统Hoare逻辑下的帧问题。

**🔧 技术方法**

使用的技术包括唯一性类型系统、Hoare逻辑、分离逻辑以及Cogent的编译器证明框架；通过形式化证明展示了从类型系统到分离逻辑的对应关系。

**📊 数据集**

该工作不依赖任何实验数据集，主要是理论分析与形式化证明。

**📈 对比分析**

没有实验比较，理论上相较于传统Hoare逻辑，分离逻辑能够更简洁地表达并验证与C代码交互时的框架条件；若要比较，主要在验证易用性与局部推理的效果上。

**⚠️ 局限性**

限制在于对C组件的安全性仍需手动证明其满足框架条件，且目前缺乏完整自动化工具链来直接将唯一性类型断言转化为分离逻辑断言。

---

## 184. Robust Pedestrian Detection with Uncertain Modality

**arXiv ID:** 2602.06363 | [PDF](https://arxiv.org/pdf/2602.06363v1)

**作者:** Qian Bie `[一作]` (Wuhan University of Science and Technology), Xin Xu `[通讯]` (Wuhan University of Science and Technology)

**通讯引用:** 15318 | [OpenAlex ID](https://openalex.org/A5053112608)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了三模态（RGB、NIR、TIR）行人检测框架AUNet，并构建了对齐的TRNT数据集，能够在任意模态组合下实现鲁棒检测。

**💡 创新点**

创新点在于：①设计统一模态验证细化（UMVR）模块，利用不确定性路由器和CLIP语义细化实时判断模态可用性；②提出模态感知交互（MAI）模块，根据UMVR输出动态激活不同模态交互机制；③实现了对任意模态组合（单、双、三模态）的自动适配，克服现有方法固定模态对的限制。

**🔧 技术方法**

技术方法包括：共享权重特征提取、轻量级不确定性路由器、CLIP驱动的语义细化、基于注意力的多模态交互、全局与局部特征融合的一阶段检测头以及联合损失训练。

**📊 数据集**

使用了新构建的TRNT（8,281对齐的RGB–NIR–TIR三模态图像）和公开的LLVIP（RGB–TIR）数据集进行评估。

**📈 对比分析**

与ProbEn、TINet、INSANet、ICAFusion、DE-YOLO等现有十余种跨模态行人检测方法相比，AUNet在TRNT上的mAP提升至97.07%，在LLVIP上的mAP提升至64.8%，均位列榜首；同时保持60.99 FPS的高速推理。

**⚠️ 局限性**

限制：目前仍依赖于预训练的CLIP模型，对极端光照或遮挡场景下的细化效果有限；当所有模态缺失时性能不可用；对模型的可解释性和训练时的计算成本仍有提升空间。

---

## 185. Nipping the Drift in the Bud: Retrospective Rectification for Robust Vision-Language Navigation

**arXiv ID:** 2602.06356 | [PDF](https://arxiv.org/pdf/2602.06356v1)

**作者:** Gang He `[一作]` (Xidian University), Weiying Xie `[通讯]` (Xidian University)

**通讯引用:** 3600 | [OpenAlex ID](https://openalex.org/A5052163069)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在线学习框架 BudVLN，结合了对抗式回溯修正（Retrospective Rectification）和集群相对策略优化（GRPO），以提升视觉语言导航（VLN）在连续环境中的鲁棒性和效率。

**💡 创新点**

创新点在于：①通过回溯锚点重新生成监督，解决了传统 DAgger 中出现的指令-状态失衡（Instruction‑State Misalignment）问题；②采用自适应互斥策略动态选择专业路径（GRPO）与错误纠正路径（SFT），实现训练效率与稳定性的平衡；③实现了仅占传统 DAgger 25% 训练成本的 SOTA 结果。

**🔧 技术方法**

使用的技术包括：自适应互斥策略、GRPO（Group Relative Policy Optimization）、SFT（Supervised Fine‑Tuning）回溯修正、oracle 轨迹生成、KL 正则化、交叉熵加权损失以及基于预训练 Transformer 的视觉语言模型。

**📊 数据集**

在 R2R‑CE 和 RxR‑CE 两大连续环境 VLN 基准上进行评估，训练集为两者联合训练集，验证集为各自的 val‑unseen split。

**📈 对比分析**

与现有 IL、IL+RL 以及使用辅助传感器的基线相比，BudVLN 在成功率（SR）和 SPL 指标上均取得最高成绩：R2R‑CE SR 57.6% / SPL 51.1%，RxR‑CE SR 56.1% / SPL 46.6%；同时训练时间仅为 DAgger 的 1/4（约 27 小时）。

**⚠️ 局限性**

局限性：回溯锚点生成依赖 oracle 规划器，对环境模拟的准确性敏感；当指令极其模糊或环境动态变化较大时，回溯方法可能无法完全保持语义一致；此外，GRPO 对样本采样和温度设置较为敏感，需要手动调参。

---

## 186. Adversarial Learning in Games with Bandit Feedback: Logarithmic Pure-Strategy Maximin Regret

**arXiv ID:** 2602.06348 | [PDF](https://arxiv.org/pdf/2602.06348v1)

**作者:** Shinji Ito `[一作]` (University of Tokyo and RIKEN), Yue Wu `[通讯]` (University of Southern California)

**通讯引用:** 8907 | [OpenAlex ID](https://openalex.org/A5100668826)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在对抗性学习中，如何在带有带子反馈的零和游戏中最小化纯策略最大值遗憾（PSMR），并提出了相应的算法和分析。

**💡 创新点**

提出了一种新的遗憾度量标准——纯策略最大值遗憾（PSMR），并在无信息和有信息的反馈模型下，分别给出了不同的算法和遗憾界限。

**🔧 技术方法**

使用了Tsallis-INF算法和基于乐观原则的算法，分别针对无信息和有信息的反馈模型进行了分析。

**📊 数据集**

研究了普通形式游戏和双线性游戏，涉及的游戏结构包括有限的动作集。

**📈 对比分析**

在无信息设置下，Tsallis-INF算法在具有严格纯策略纳什均衡的游戏中达到了(c log T)的遗憾界限；在有信息设置下，提出的算法在不同的游戏中达到了更小的遗憾界限，且在双线性游戏中也得到了类似的结果。

**⚠️ 局限性**

在无信息设置下，证明了对抗性学习的难度，表明在某些情况下，遗憾界限是不可避免的，且在有信息设置下，算法的性能依赖于游戏的具体结构。

---

## 187. Learning Human Visual Attention on 3D Surfaces through Geometry-Queried Semantic Priors

**arXiv ID:** 2602.06419 | [PDF](https://arxiv.org/pdf/2602.06419v1)

**作者:** Soham Pahari `[一作]` (UPES), Sandeep C. Kumain `[通讯]` (UPES)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种双流模型SemGeo-AttentionNet，用几何流和扩散式语义先验通过不对称交叉注意力融合，预测3D网格的视觉注意力并通过强化学习生成基于网格拓扑的扫描路径。

**💡 创新点**

创新点在于①首次在模型架构中明确区分底层几何与顶层语义，采用几何查询语义的交叉注意力；②利用几何条件的多视图扩散渲染提取零样本语义先验；③提出面向3D网格的扫描路径生成框架，包含拓扑动作空间和抑制返回机制。

**🔧 技术方法**

技术包括点变压器V3（Point Transformer V3）进行几何编码，基于ControlNet的Stable Diffusion和DINOv2提取语义特征，交叉注意力融合，PPO强化学习生成扫描路径，以及多视图渲染与语义投影。

**📊 数据集**

使用了三大公开3D注意力数据集：SAL3D、NUS3D‑Saliency 和 3DVA。

**📈 对比分析**

与现有几何与学习方法（如Mesh Mamba、MIMO‑GAN、Diffusion Wavelets等）对比，SemGeo‑AttentionNet 在SAL3D上 CC 0.849、KL 0.164、MSE 0.011；在NUS3D上 LCC 0.609、AUC 0.935；在3DVA上 LCC 0.762；扫描路径方面实现 NSS 2.05 与 MultiMatch 0.51，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括：①需要为每个网格渲染100个视角，计算成本高；②语义先验来自预训练扩散模型，可能对非视觉语义或极端几何形状泛化不足；③仍依赖几何采样，无法直接处理非常稀疏或高动态范围的点云。

---

## 188. Intrinsic Stability Limits of Autoregressive Reasoning: Structural Consequences for Long-Horizon Execution

**arXiv ID:** 2602.06413 | [PDF](https://arxiv.org/pdf/2602.06413v1)

**作者:** Hsien-Jyh Liao `[一作]` `[通讯]`, Hsien-Jyh Liao

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型在长周期推理中的内在稳定性限制，提出单路径自回归执行的决策优势随推理长度指数衰减，从而引入了临界长度$L^*$；

**💡 创新点**

创新点在于将长周期推理视为动态系统稳定问题，提出理论上不可逾越的稳定边界，并将此解释为结构化推理（如图结构）必要性的根源；

**🔧 技术方法**

采用信息理论与马尔可夫链收缩分析、实验模拟与实际TextWorld交互、以及对链式推理的敏感性分析等技术；

**📊 数据集**

使用合成长周期任务、TextWorld环境以及对Gemma 3 4B和GPT‑4o的对比实验数据；

**📈 对比分析**

通过对比未结构化基线与结构化“Landmarks”方法，实验显示结构化执行显著延缓决策优势衰减、提升探索覆盖率和减少回溯，性能提升幅度在不同任务上从数倍至十倍不等；

**⚠️ 局限性**

局限性包括未给出节点生成与最优分段策略的算法、未验证多代理或外部记忆情境下的适用性，以及对理论假设（如收缩系数）在真实模型中的估计缺乏实证。

---

## 189. VENOMREC: Cross-Modal Interactive Poisoning for Targeted Promotion in Multimodal LLM Recommender Systems

**arXiv ID:** 2602.06409 | [PDF](https://arxiv.org/pdf/2602.06409v1)

**作者:** Guowei Guan `[一作]` (Nanyang Technological University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**通讯引用:** 6676 | [OpenAlex ID](https://openalex.org/A5027969322)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对多模态大型语言模型推荐系统的交叉模态交互式中毒攻击，能在保持推荐效果的前提下显著提升目标商品曝光率。

**💡 创新点**

创新点在于利用跨模态注意力机制同步对文本与图像进行微调，突破传统单模态或交互式攻击被多模态融合抵消的局限，并通过曝光对齐（Exposure Alignment）定位高曝光语义热点。

**🔧 技术方法**

使用公开的CLIP视觉编码器、T5文本编码器作为代理，结合注意力卷积、投影器和参数高效微调（PEFT）实现攻击；核心技术为曝光对齐和交叉模态交互扰动（CIP）。

**📊 数据集**

在Amazon Clothing、Sports、Toys三大真实商品评论数据集上评估，采用公开的多模态嵌入。

**📈 对比分析**

与十种基线（交互级、单模态、Shadowcast等）对比，攻击平均ER@20提升至0.73，远超最强基线+0.52点；同时推荐质量（HR/NDCG）保持与无攻击一致。

**⚠️ 局限性**

局限性包括：攻击仅在灰盒设定下实现，依赖公开模型作为代理，且在多目标或更大规模系统中效果可能衰减；未探讨防御策略的有效性。

---

## 190. TFusionOcc: Student's t-Distribution Based Object-Centric Multi-Sensor Fusion Framework for 3D Occupancy Prediction

**arXiv ID:** 2602.06400 | [PDF](https://arxiv.org/pdf/2602.06400v1)

**作者:** Zhenxing Ming `[一作]` (Australian Centre for Robotics), Stewart Worrall `[通讯]` (Australian Centre for Robotics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个基于 Student’s t 分布的对象中心多传感器融合框架 TFusionOcc，用于 3D 语义占据预测。

**💡 创新点**

创新点在于：① 采用三阶段特征融合（早期、中期、晚期）与多模态加权融合模块 MGCAFusion；② 采用 T‑Mixture 模型和可变形超四面体原语（T‑SQ‑IW）实现更鲁棒的几何建模；③ 将可变形注意力与稀疏卷积自注意力结合，提升特征整合效率。

**🔧 技术方法**

技术要点包括：Student’s t 分布核、T‑Mixture 混合模型、3D 可变形注意力、稀疏卷积自注意力、多模态特征融合、超四面体与可变形超四面体原语、3D splatting。

**📊 数据集**

使用数据集：nuScenes 及其增强版 nuScenes-C（包含雨、夜、雾、雪、传感器失效等多种干扰）。

**📈 对比分析**

与多种 SOTA 方法（OccFusion、GaussianFormer3D、OccCylindrical、DAOcc 等）在 nuScenes 验证集和 nuScenes‑C 上进行对比，TFusionOcc 在 IoU/mIoU 指标上达到 SOTA，尤其 T‑SQ‑IW‑25600 设置表现最佳。

**⚠️ 局限性**

局限性：需要较多原语和渲染操作导致推理延迟；对激光点云密度和雨雪噪声敏感，未来计划通过减少原语数和渲染步骤来提升效率与鲁棒性。

---

## 191. Unlocking Noisy Real-World Corpora for Foundation Model Pre-Training via Quality-Aware Tokenization

**arXiv ID:** 2602.06394 | [PDF](https://arxiv.org/pdf/2602.06394v1)

**作者:** Arvid E. Gollwitzer `[一作]` (Massachusetts Institute of Technology), Adrián Noriega de la Colina `[通讯]` (Broad Institute of MIT and Harvard)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种质量感知分词（QA-Token），通过将数据质量直接嵌入词表构建，实现了在噪声真实世界语料上训练基础模型的高效方法。

**💡 创新点**

创新点在于将质量评估融入双层优化框架，并通过强化学习（PPO）和Gumbel‑Softmax自适应参数学习，兼顾模型性能、词表复杂度与质量奖励，实现信息理论最优的分词策略。

**🔧 技术方法**

采用了双层优化、强化学习（PPO）+奖励设计、Gumbel‑Softmax 端到端学习以及子模函数近似等技术。

**📊 数据集**

使用了基因组测序数据（SRA、GRCh38、GIAB、CAMI II）、高频金融订单簿数据（BTC/USD LOBSTER）以及大规模元基因组 METAGENE‑1 等真实噪声数据集。

**📈 对比分析**

与 BPE、SentencePiece、WordPiece、字节级模型等基准对比，QA‑Token 在基因组变异召回 F1 提升 6.7pp、金融交易 Sharpe 比 BPE 提升 30%、METAGENE‑1 病原体检测 MCC 提升至 94.53，均显著优于传统方法。

**⚠️ 局限性**

局限性包括需要领域特定的质量指标、词表构建过程计算成本较高且难以快速迭代，且在缺乏可靠质量信号的领域难以直接应用。

---

## 192. POINTS-GUI-G: GUI-Grounding Journey

**arXiv ID:** 2602.06391 | [PDF](https://arxiv.org/pdf/2602.06391v1)

**作者:** Zhongyin Zhao `[一作]` (WeChat AI), Jie Zhou `[通讯]` (WeChat AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并训练了基于 POINTS-1.5 的 GUI grounding 模型 POINTS-GUI-G，实现了高精度的界面元素定位。

**💡 创新点**

创新点包括：① 从零开始的三大柱数据工程流程（统一坐标、噪声过滤、复杂度提升及合成数据）；② 对视觉编码器完全解冻并提升训练分辨率至 3072×3072；③ 引入可验证奖励的强化学习（GRPO）以显著提升定位准确率。

**🔧 技术方法**

使用的技术包括：Qwen3‑8B 语言模型、Qwen2‑VL‑NaViT 视觉编码器、OmniParser‑v2 元素检测、分布式训练与多路 roll‑out、GRPO 强化学习框架、自动化数据清洗与合成。

**📊 数据集**

主要数据集为多源开源 GUI grounding 数据（RICO、FineWeb 等）统一标准后再加工；合成数据来自 GUI‑CodeGen（VS Code 等专业软件）和 GUI‑Overlay；文本聚焦数据从 DataComp + PaddleOCR 提取；总共覆盖 13 个公开数据集，构成多样化的训练集。

**📈 对比分析**

在 ScreenSpot‑v2、ScreenSpot‑Pro、OSWorld‑G、MMBench‑GUI‑L2、UI‑Vision 等 5 大 benchmark 上与同尺度和更大模型对比，POINTS‑GUI‑G 在 3/5 评测榜首，尤其在 ScreenSpot‑Pro 上超越 GTA1‑7B 9.8 分、GUI‑Owl‑7B 5 分，并逼近 32B 大模型的表现。

**⚠️ 局限性**

局限性包括：① 仍以 8B 参数为上限，难以进一步扩展；② 依赖大量人工或自动化合成数据，可能对真实多样化界面产生迁移偏差；③ 仅关注定位任务，未涵盖完整的 GUI 交互规划与执行；④ 强化学习阶段计算成本高，训练不稳定。

---

## 193. Difficulty-Estimated Policy Optimization

**arXiv ID:** 2602.06375 | [PDF](https://arxiv.org/pdf/2602.06375v1)

**作者:** Yu Zhao `[一作]` (Alibaba International Digital Commerce), Weihua Luo `[通讯]` (Alibaba International Digital Commerce)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在线难度估计器（Difficulty Estimator）并将其嵌入到Group Relative Policy Optimization（GRPO）中，动态过滤掉奖励方差为零或几乎为零的训练样本，从而在保持模型性能的同时显著降低rollout计算成本。

**💡 创新点**

创新点包括：
• 在线实时难度评估与过滤，避免了传统离线筛选或过度采样的延迟和计算开销。
• 通过BERT编码器配合优势估计与PPL两头预测，实现对样本难度的双重建模。
• 引入排名损失与蒸馏损失的联合训练，提升难度估计的判别性与对模型变化的适应性。
• 两阶段“冷启动”训练策略，先热身估计器再开启过滤，减少噪声误判。

**🔧 技术方法**

使用的技术包括：
• RLVR框架下的GRPO算法（无value模型，group相对优势计算）。
• BERT预训练模型作为难度估计器基础，并通过两头输出（优势估计、PPL）进行训练。
• 采用BCE损失、排名损失（pairwise margin）和蒸馏损失共同优化。
• 在线数据流中同步更新估计器和actor模型，避免额外离线预处理。
• 与vLLM、H100 GPU、Python等技术栈集成实现高效rollout。

**📊 数据集**

训练数据集：
• DAPO-MATH-17K
• Open-R1（OR1）
• Nemotron-Math

评测数据集：
• GSM8K
• MATH
• AMC23
• Minerva Math
• Olympiad Bench。

**📈 对比分析**

与传统GRPO、动态采样、离线筛选等方法比较，DEPO在相同训练步骤下实现了约1.5%的平均精度提升，并将rollout成本降低约2倍（总计算开销减少约50%）。在将DEPO与其他RL框架（如Dynamic Sampling）结合时，还可再提升0.9%精度。实验表明，DEPO在保持训练效率的同时，提供了更优的性能-效率 Pareto 前沿。

**⚠️ 局限性**

局限性：
• 对模型能力和训练集难度高度依赖，过于容易或过于困难的数据集时过滤率可能不佳。
• 过度过滤（排名损失权重过大）会导致有效样本被丢弃，影响多样性和最终性能。
• 仍需在部分样本上进行rollout，无法完全消除rollout成本。
• 主要验证在数学推理任务上，尚未在更广泛的自然语言推理或编程任务中评估。

---

## 194. Di3PO -- Diptych Diffusion DPO for Targeted Improvements in Image

**arXiv ID:** 2602.06355 | [PDF](https://arxiv.org/pdf/2602.06355v1)

**作者:** Sanjana Reddy `[一作]` (Google), Praneet Dutta `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对文本渲染问题，本文提出利用Diptych Prompting生成背景一致、仅文本差异的正负样本对，并用Diffusion DPO对Stable Diffusion XL进行细调。

**💡 创新点**

创新点在于：①通过一次性生成双面图像（Diptych）锁定背景不变，从而显著减少训练对无关视觉差异的关注；②无需额外奖励模型，直接构造可验证的优劣对；③理论上说明了视觉相似度如何聚焦梯度信号、提升学习效率。

**🔧 技术方法**

主要技术包括：大语言模型（Gemini 2.5）生成背景描述与误拼文字，Stable Diffusion XL（SDXL 1.0/SD3）生成双面图像，Canny边缘检测切分图像，Diffusion DPO算法进行微调，OCR API评估文本准确度。

**📊 数据集**

数据集：使用300对Diptych样本（每对一正一负）用于训练，另用约2000条基于LLM生成的文本渲染提示作为评估集；背景与文字均由Gemini自动生成，保证多样性。

**📈 对比分析**

与预训练模型、SFT（仅正样本微调）以及背景变化的DPO基线比较，Di3PO在Levenshtein编辑距离、单词错误率和子串匹配率上均优于基线；例如在SDXL 1.0上，Word Error Rate从0.721下降至0.646，Substring Match Ratio从0.062上升至0.095，显示显著提升。

**⚠️ 局限性**

局限性：①方法依赖于基模型支持Diptych生成（SDXL 1.0自身不具备，需要更高阶模型）；②误拼文字的“分布内”假设可能不适用于所有模型；③生成与验证过程仍需LLM与图像处理步骤，可能受模型误差影响；④当前仅验证在文本渲染任务，泛化到其他细粒度任务仍需进一步验证。

---

## 195. Trifuse: Enhancing Attention-Based GUI Grounding via Multimodal Fusion

**arXiv ID:** 2602.06351 | [PDF](https://arxiv.org/pdf/2602.06351v1)

**作者:** Longhui Ma `[一作]` (National University of Defense Technology), Miao Wang `[通讯]` (Academy of Military Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种无监督的多模态融合框架 Trifuse，用于将自然语言指令精准地定位到 GUI 元素。

**💡 创新点**

创新点在于提出 Consensus‑SinglePeak（CS）融合策略，将 MLLM 的注意力、OCR 文本和图标级 caption 三种互补模态同时利用，并通过交叉模态一致性与单峰激活平衡，显著提升定位可靠性。

**🔧 技术方法**

采用多模态大型语言模型（如 Qwen2.5‑VL）、OCR 引擎、图标 caption 模型以及自定义的 token 与 head 过滤策略，并实现两阶段缩放定位。

**📊 数据集**

使用四个公开 GUI grounding 基准数据集：ScreenSpot、ScreenSpot‑v2、ScreenSpot‑Pro 与 OSWorld‑G 进行评估。

**📈 对比分析**

与无微调基线 TAG 相比，Trifuse 在所有基准上平均提升 20–35% 的准确率；与监督微调模型相比仅差 1–3%，并在多项强化学习方法中实现更佳表现。

**⚠️ 局限性**

仍受限于 OCR 与 caption 的识别误差、对极小或高密度 UI 元素的分辨率不足，以及对极端 UI 风格的泛化能力待进一步验证。

---

## 196. Enhance and Reuse: A Dual-Mechanism Approach to Boost Deep Forest for Label Distribution Learning

**arXiv ID:** 2602.06353 | [PDF](https://arxiv.org/pdf/2602.06353v1)

**作者:** Jia-Le Xu `[一作]` (Hohai University), Baoliu Ye `[通讯]` (Nanjing University)

**通讯引用:** 1827 | [OpenAlex ID](https://openalex.org/A5087746903)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在深森林框架上设计并实现了ERDF方法，结合标签相关性特征增强和测量感知的特征重用，以解决标签分布学习任务。

**💡 创新点**

创新点在于明确利用标签相关性在输入特征空间进行增强，并通过特征重用机制保持深森林训练稳定，实现了两机制协同提升性能的全新思路。

**🔧 技术方法**

使用的核心技术包括深森林（Cascade Forest）、PCA提取标签相关性向量、随机森林/极度随机森林回归器、特征增强及特征重用阈值策略。

**📊 数据集**

实验采用五个公开LDL基准数据集：Movie、Natural_Scene、SBU_3DFE、emotion6、SCUT_FBP。

**📈 对比分析**

与四种主流LDL算法（AA‑KNN、SA‑BFGS、StructRF、LDL‑SCL）进行对比，ERDF在六种距离/相似度指标上平均排名1.20，性能普遍优于对手。

**⚠️ 局限性**

局限性包括对增强特征维度的敏感性，若重用阈值不当易导致噪声扩散；同时对超参数k、阈值等仍需手工调优，跨任务推广性尚待进一步验证。

---

## 197. Fast Makespan Minimization via Short ILPs

**arXiv ID:** 2602.06514 | [PDF](https://arxiv.org/pdf/2602.06514v1)

**作者:** Danny Hermelin `[一作]` (Ben Gurion University of the Negev), Dvir Shabtay `[通讯]` (Ben Gurion University of the Negev)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过把相关调度问题转化为常数约束行的短整数线性规划（short ILP），从而得到在固定机器数下的伪多项式时间算法，并将该方法推广到容量约束、拒绝作业、发布日期等多种变体。

**💡 创新点**

创新点主要在于：
① 采用短 ILP 求解器将作业分配约束合并成常数行，利用目标函数强制满足作业唯一分配；
② 通过 SPT 顺序与二进制系数的组合，实现对约束矩阵系数的有效控制；
③ 在多种变体（容量、拒绝、发布、相互不相容机器）中统一构造短 ILP，取得比传统动态规划和 FFT 方法更优的伪多项式复杂度。

**🔧 技术方法**

使用的技术包括：
- 短整数线性规划求解器（Jansen & Rohwedder 的算法）
- 目标函数代替约束（利用二进制权重保证唯一分配）
- SPT 排序和发布日期分块
- 变量扩展（slack 变量、作业-机器对变量）
- 结合动态规划/FFT 结果做复杂度对比。

**📊 数据集**

本文为理论研究，没有使用实验数据集，所有结果均来自理论分析与算法复杂度证明。

**📈 对比分析**

通过与现有最速动态规划（O(P^{m-1}n)）、FFT（O(P^{m-1})）以及早期 ILP 结果对比，表明在 p_max 较小（如 O(√n) 或 O(n^{1/4})）时，新算法在时间上明显优于传统方法；具体复杂度在表中给出，并指出当 m > 4、5 时性能提升更为显著。

**⚠️ 局限性**

限制与不足：
① 仅适用于固定常数机器数，m 的指数项导致大 m 时不可扩展；
② 对 p_max 的依赖仍存在，若 p_max 与 n 同阶，优势不明显；
③ 需要 SPT 顺序或发布日期分块，若问题不满足这些结构则难以直接应用；
④ 仅给出理论复杂度，没有实验验证；
⑤ 部分变体仍需显式 0/1 约束或空间昂贵的求解器，导致空间消耗较高。

---

## 198. Designing Computational Tools for Exploring Causal Relationships in Qualitative Data

**arXiv ID:** 2602.06506 | [PDF](https://arxiv.org/pdf/2602.06506v1)

**作者:** Han Meng `[一作]` (National University of Singapore), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**通讯引用:** 1772 | [OpenAlex ID](https://openalex.org/A5054435118)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计实现并评估QualCausal系统，用于交互式构建和可视化定性数据中的因果网络。

**💡 创新点**

结合LLM进行自动指标提取、概念映射和因果关系识别，并提供多视图交互式可视化以支持假设生成。

**🔧 技术方法**

使用GPT‑4.1进行提示工程；前端采用Vue.js、D3.js；后端使用Django；结合自然语言处理、图网络可视化技术。

**📊 数据集**

使用MHStigmaInterview‑20（20份访谈数据）和SemEval 2010 Task 8子集进行实验评估。

**📈 对比分析**

与共现网络和因果关键词启发式进行对比；在MHStigma上精确率80.6%、召回83.1%、方向准确率98.2%；在SemEval上精确率100%、召回85%；系统整体优于基线。

**⚠️ 局限性**

仅识别句内因果关系，无法捕获跨句/篇章级因果；顺序化工作流程可能引入偏差；对大规模语料的可扩展性有限；用户对工具依赖与研究方法论的匹配存在张力。

---

## 199. Can Microcanonical Langevin Dynamics Leverage Mini-Batch Gradient Noise?

**arXiv ID:** 2602.06500 | [PDF](https://arxiv.org/pdf/2602.06500v1)

**作者:** Emanuel Sommer `[一作]` (LMU Munich), David Rügamer `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了一种可扩展的微观定常朗之万动力学采样器（SMILE / pSMILE），并在大规模贝叶斯神经网络上进行了系统验证。

**💡 创新点**

创新点包括：① 通过梯度噪声预处理消除异方差噪声导致的系统偏差；② 基于能量误差方差的自适应步长调参实现数值稳定；③ 将上述技术与贝叶斯深度集成（BDE）相结合，构建了一类高效的 SMILE/​pSMILE 采样器。

**🔧 技术方法**

使用的核心技术有：微观定常朗之万动力学、SGMCMC、梯度噪声预缩（局部对角预缩）、能量误差在线估计与 Gamma 分布拟合、动态步长自适应、数值安全门限（能量阈值拒绝）、贝叶斯深度集成（多链 Ensemble）等。

**📊 数据集**

实验数据集包括：三层全连接回归（Airfoil、Bikesharing、Energy）、CIFAR‑10（ResNet‑7/ResNet‑18）、Imagenette（Vision Transformer）、Fashion‑MNIST（LeNet）以及 nanoGPT 语言模型。

**📈 对比分析**

与全批 MCLMC（MILE）、规模适配 SGHMC、SGLD、cSGLD 以及深度集成等基线对比，SMILE/pSMILE 在 LPPD、RMSE、准确率、NLL、AUROC、AURC 等指标上至少与最强基线持平或优于，并在不确定性评估上表现更佳；在大规模网络中实现了 state‑of‑the‑art 的采样性能。

**⚠️ 局限性**

局限性在于：① 仅依赖批梯度噪声驱动，未探索加入显式噪声的可能性；② 对角预缩为近似，残留偏差不可避免；③ 对极大参数规模仍需中等批量，进一步扩展仍有挑战；④ 对噪声显式注入与探索平衡的系统研究尚未展开。

---

## 200. Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention

**arXiv ID:** 2602.06478 | [PDF](https://arxiv.org/pdf/2602.06478v1)

**作者:** Xiaosong Jia `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 24091 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了双流Transformer架构Efficient‑LVSM，用于高效的视角合成。

**💡 创新点**

通过将输入编码器和目标解码器解耦，并采用自注意与交叉注意的co‑refinement，降低计算复杂度至线性，支持增量推理。

**🔧 技术方法**

使用图像分块编码、Plücker射线嵌入、Transformer块、REPA知识蒸馏以及KV缓存技术。

**📊 数据集**

在RealEstate10K（场景级）和Objaverse/Google Scanned Objects/Amazon Berkeley Objects（对象级）等数据集上训练与评测。

**📈 对比分析**

与LVSM、GS‑LRM等最先进方法对比，PSNR提升0.2dB，训练时间减半，推理速度提升约14.9×，并实现对未见视角数量的零样本泛化。

**⚠️ 局限性**

对极端光照或稀疏视角的鲁棒性仍有限，且单帧推理分辨率受限。

---

## 201. Towards Generalizable Reasoning: Group Causal Counterfactual Policy Optimization for LLM Reasoning

**arXiv ID:** 2602.06475 | [PDF](https://arxiv.org/pdf/2602.06475v1)

**作者:** Jingyao Wang `[一作]` (Institute of Software Chinese Academy of Sciences), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 43824 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了GC^2PO，一种基于组式因果对照奖励的RL优化框架，提升LLM推理的可泛化性。

**💡 创新点**

首次将因果对照实验与分段奖励结合，构造鲁棒性与有效性两个量化指标，解耦推理过程与最终答案，促使模型学习可迁移的推理策略。

**🔧 技术方法**

使用GRPO强化学习框架、因果结构模型、latent空间扰动、Token级优势分配、Monte Carlo估计等技术。

**📊 数据集**

在GSM8K、MATH、HumanEval、AIME、AMC、MinervaMATH、NuminaMath等多种推理基准上进行实验。

**📈 对比分析**

与GRPO、GVPO、Dr.GRPO、GCPO、MRT、L2T-GRPO等基线相比，GC^2PO在多项基准上实现最高pass@1，提升2–3%，且token消耗更少、训练更稳定。

**⚠️ 局限性**

仍需人工设计分段标签，对扰动方式敏感；实验主要集中在算术/逻辑推理任务，跨模态推理及更大模型的泛化能力尚待进一步验证。

---

## 202. Revisiting the Shape Convention of Transformer Language Models

**arXiv ID:** 2602.06471 | [PDF](https://arxiv.org/pdf/2602.06471v1)

**作者:** Feng-Ting Liao `[一作]` (MediaTek Research), Da-shan Shiu `[通讯]` (MediaTek Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文重新审视Transformer语言模型中Feed‑Forward网络（FFN）的形状约定，提出将传统的窄‑宽‑窄（narrow‑wide‑narrow）结构替换为宽‑窄‑宽的“hourglass”子MLP堆叠，并在1 B级别内对该架构进行大规模实验。

**💡 创新点**

创新点包括①提出深度且窄化的hourglass FFN，使得参数可以从FFN迁移到注意力模块；②在相同参数预算下实现更高的表达能力；③系统探讨d_h/d_model、K（内部深度）与d_model/L比值的交互效应，发现U型最优宽度‑深度平衡。

**🔧 技术方法**

技术手段主要是Transformer基础框架、SwiGLU激活、残差连接与层归一化；通过在FFN层中嵌入多层hourglass子模块（K层）；训练使用AdamW+余弦学习率计划；评估以验证集的交叉熵损失与困惑度（PPL）为指标，并在多项推理/问答任务上做下游性能测评。

**📊 数据集**

实验使用与基准模型相同的标准大规模文本语料（如C4或Common Crawl等公开语言模型数据集），确保对比公平。

**📈 对比分析**

与传统窄‑宽‑窄Transformer以及OLMo‑2进行对比，采用相同参数预算（误差<0.001%），评估验证集损失、PPL以及下游任务准确率。结果显示：在113M–906M规模下，hourglass FFN在验证损失与PPL上均优于传统模型；在1B规模下表现可与传统模型持平，甚至略优；在推理与问答任务中，hourglass模型在多数指标上也呈现提升。

**⚠️ 局限性**

局限性主要包括①仅在至1 B规模内验证，未探索更大规模的可扩展性；②只测试了标准多头自注意力，未结合更高级的注意力机制；③参数搜索受限于计算资源，未在所有组合上做完整调优；④深层hourglass可能存在梯度消失或噪声累积问题，需进一步研究。

---

## 203. BrokenBind: Universal Modality Exploration beyond Dataset Boundaries

**arXiv ID:** 2602.06451 | [PDF](https://arxiv.org/pdf/2602.06451v1)

**作者:** Zhuo Huang `[一作]` (Sydney AI Centre, University of Sydney), Tongliang Liu `[通讯]` (Sydney AI Centre, University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 BrokenBind 方法，利用多数据集的共享模态（pivot）对不在同一数据集中的两种模态进行绑定，并通过跨模态、跨数据的关系矩阵生成伪嵌入，实现对缺失模态的推断，从而在数据集不匹配的情况下实现多模态绑定。

**💡 创新点**

创新点在于：1) 通过跨模态和跨数据的转移矩阵实现模态外推（modality extrapolation，MOX），从而补齐缺失模态；2) 在转移矩阵的两种视角（模态视角和数据视角）上生成伪嵌入，并用 Frobenius 正则化对齐两种伪嵌入；3) 将 MOX 与 CyCLIP 的对称一致性训练相结合，构建无偏差的联合嵌入空间；4) 使任意两种模态都可通过至少一个共享模态在不同数据集间绑定，显著提升跨数据集泛化与“emergent alignment”。

**🔧 技术方法**

主要技术：Contrastive 学习（CLIP、CyCLIP）、Mixup 及其扩展的外推、转移矩阵（跨模态/跨数据）、Frobenius 正则化、对称一致性损失、LoRA 微调与线性投影。实现上基于现有预训练绑定模型的编码器（ImageBind、LanguageBind、VCLIP、CLAP、TVL）。

**📊 数据集**

使用的数据集与模态包括：MSRVTT、AVE、ObjF（real/1.0）、NYU、SUN、LLVIP、FLIR；模态覆盖视觉（vi）、文本（te）、音频（au）、触觉（ta）、深度（de）、热像（th）。实验设计中将目标模态从训练集“隐藏”，仅通过共享模态进行推断。

**📈 对比分析**

与 ImageBind、LanguageBind、VCLIP+CLAP、TVL 等基线进行比较；在多种两/三数据集绑定任务和 emergent alignment 场景下，BrokenBind 的 mAP 通常提升 2–4 倍（如 35%~45% vs 10%~20%），在三数据集场景中也明显优于仅 Fine‑Tuning 或基线；在低数据量与不匹配分布下仍保持稳定高性能。

**⚠️ 局限性**

局限性：1) 依赖已有的预训练绑定模型，对极其稀缺或不平衡的模态（如触觉+文本）仍难以获得优异效果；2) 模态外推的转移矩阵可能在分布差异极大时失效；3) 对于非常罕见或单独出现的模态，需进一步探索不平衡处理与更鲁棒的外推策略。

---

## 204. MultiGraspNet: A Multitask 3D Vision Model for Multi-gripper Robotic Grasping

**arXiv ID:** 2602.06504 | [PDF](https://arxiv.org/pdf/2602.06504v1)

**作者:** Stephany Ortuno-Chanelo `[一作]` (Politecnico di Torino), Raffaello Camoriano `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一的3D深度学习模型，能够同时预测平行夹爪和平面真空吸盘的抓取姿态，从而让单臂机器人在同一框架下实现多端执行器抓取。

**💡 创新点**

创新点在于：① 通过共享骨干网络与任务特定的姿态细化模块，实现在单模型中跨抓取方式的知识迁移；② 在GraspNet‑1Billion与SuctionNet‑1Billion的对齐注释基础上构造多模态抓取可行性（graspness）映射；③ 引入PCGrad梯度干预与FPS采样，提升多任务训练的稳定性与覆盖率。

**🔧 技术方法**

使用基于MinkowskiEngine的3D U‑Net骨干、三头预测（objectness、平行夹爪graspness、真空爪graspness）、多分支采样与姿态细化（ViewNet、Cylinder Grouping、Grasp Head、法向估计）以及Adam+cosine学习率调度。

**📊 数据集**

采用对齐后的GraspNet‑1Billion（平行夹爪）和SuctionNet‑1Billion（真空吸盘）两大公开点云抓取数据集，构成统一的训练与评估环境。

**📈 对比分析**

在标准3D抓取基准上（Precision@k/Average Precision）与单任务模型（Dex‑Net3.0、S1B、GSNet、EFG、E‑FG）比较，实验显示：① 在真空抓取上平均提升≈5–10%，尤其在相似/未见物体上显著；② 在平行夹爪抓取上略低于最优单任务模型，但在综合表现与模型规模上更具优势；③ 在工业实机测试中，单模型显著提高了成功率与抓取效率，且可直接替代两模型的组合。

**⚠️ 局限性**

局限性：① 仅支持两种端执行器（平行夹爪与真空吸盘），扩展到更多抓取工具仍需实验验证；② 目前未实现抓取分数跨端直接可比的校准，导致两种抓取方式在同一场景下的优先级选择仍手工或后处理；③ 对非平面或高纹理表面真空抓取的鲁棒性仍低于专用真空模型。

---

## 205. RelayGen: Intra-Generation Model Switching for Efficient Reasoning

**arXiv ID:** 2602.06454 | [PDF](https://arxiv.org/pdf/2602.06454v1)

**作者:** Jiwon Song `[一作]` (Seoul National University), Jae-Joon Kim `[通讯]` (Seoul National University)

**通讯引用:** 4038 | [OpenAlex ID](https://openalex.org/A5003219699)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出RelayGen框架，在长篇推理生成过程中实现训练无关的分段模型切换，以降低推理成本。

**💡 创新点**

创新点在于利用离线概率边缘分析挑选可切换语义线索，实现粗粒度段级切换，既保持大模型的高难度推理，又能在低难度段使用小模型，无需额外训练或路由器。

**🔧 技术方法**

技术上结合vLLM的前缀缓存与OpenAI兼容API，在推理时基于停词触发切换，并与Eagle-3等推测式解码器兼容。

**📊 数据集**

实验使用AIME 2025、MATH500、GPQA‑Diamond等数学与科学推理基准，并用AMC 2023作为校准集。

**📈 对比分析**

与单模型、Speculative Thinking、R2R等基线对比，RelayGen在保持近乎大模型准确率的同时，单独实现约1.3×速度提升，结合推测式解码可达2.2×加速，误差低于2%。

**⚠️ 局限性**

局限在于仅适用于显式长推理任务且要求小模型具备足够低难度推理能力，且对多语言场景与极短输出的效果尚未验证。

---

## 206. ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking

**arXiv ID:** 2602.06445 | [PDF](https://arxiv.org/pdf/2602.06445v1)

**作者:** Weidong Huang `[一作]` (Beijing Institute for General Artificial Intelligence), Yao Su `[通讯]` (Beijing Institute for General Artificial Intelligence)

**通讯引用:** 502 | [OpenAlex ID](https://openalex.org/A5102740429)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究通过把能耗作为约束条件，在受限强化学习框架 ECO 中训练小型人形机器人 BRUCE 的行走策略。

**💡 创新点**

创新点是将能耗和运动约束从奖励函数中分离出来，直接作为不等式约束并使用 PPO‑Lag 动态更新拉格朗日乘子，从而实现能耗与稳定性的无参数平衡。

**🔧 技术方法**

采用了受限马尔可夫决策过程（CMDP）、PPO‑Lag 算法、PD 控制、域随机化以及线性搜索等技术。

**📊 数据集**

实验数据来自 IsaacGym、MuJoCo 与 Gazebo 的仿真环境，以及真实 BRUCE 机器人在户外与受扰动测试中的跑步记录。

**📈 对比分析**

与 MPC、PPO、IPO、P3O、CRPO 等基线相比，ECO 在 0.1–0.2 m/s 区间内能耗降低约 3 倍（对 MPC）或 1.4 倍（对 PPO），同时保持相同或更好的速度跟踪与稳定性。

**⚠️ 局限性**

主要局限是对多约束（自碰撞、足部接触速度等）的收敛性差，需手动调节 PD 参数，且当前仅在平地行走验证，缺乏对斜坡或楼梯等更复杂地形的适配。

---

## 207. ChatUMM: Robust Context Tracking for Conversational Interleaved Generation

**arXiv ID:** 2602.06442 | [PDF](https://arxiv.org/pdf/2602.06442v1)

**作者:** Wenxun Dai `[一作]` (Tsinghua University), Chunyu Wang `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ChatUMM，一种支持多轮交互的统一多模态模型，能够在对话中实现文本与图像的无缝交互与上下文跟踪。

**💡 创新点**

创新点在于（1）引入了交错多轮训练策略，将文本与图像的序列视为连续对话流，提升上下文追踪与意图解析；（2）构建了系统化的对话数据合成管线，将单轮数据转换为多轮、包含噪声干扰的对话，强化模型的长程依赖与鲁棒性。

**🔧 技术方法**

技术包括基于BAGEL的解码器 Transformer + Mixture‑of‑Transformers 结构，VAE 与 ViT 双编码器，Generalized Causal Attention，Flow‑Matching 视觉生成损失，以及 LLM‑驱动的原子操作生成对话数据。

**📊 数据集**

使用多种公开单轮数据集（文本‑图像生成、图像编辑、主题生成等）以及通过 LLM 自动合成的多轮对话；在评测上采用 MME、MMBench、MMM U、MM‑Vet、MathVista、MMVP、GenEval、GEdit‑Bench 等基准。

**📈 对比分析**

与现有开源统一模型（如 BAGEL、Janus‑Pro、ILLUME 等）比较，ChatUMM 在视觉理解指标（MMBench 84.6、MMMV 53.8、MM‑Vet 66.4、MathVista 74.7）与图像生成/编辑（GenEval 0.85、GEdit‑Bench 6.95）均位列前列，甚至在多轮编辑任务上超过同类最优模型。

**⚠️ 局限性**

局限性包括：模型规模相对有限（7B），无法完全匹配商用代理系统（如 GPT‑4o）性能；双编码器设计导致推理成本较高；对更复杂推理与知识检索的支持仍需进一步加强。

---

## 208. Simulating Word Suggestion Usage in Mobile Typing to Guide Intelligent Text Entry Design

**arXiv ID:** 2602.06489 | [PDF](https://arxiv.org/pdf/2602.06489v1)

**作者:** Yang Li `[一作]` (Saarland University), Anna Maria Feit `[通讯]` (Saarland University)

**通讯引用:** 1698 | [OpenAlex ID](https://openalex.org/A5083103411)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了一种基于强化学习的模拟模型 WSTypist，用来预测和模拟用户在移动键盘中如何使用词建议，评估其对文本输入效率和行为适应性的影响。

**💡 创新点**

① 在决策层引入了效率评估、语言属性、个体依赖等认知机制；② 将这些机制嵌入层级监督控制框架；③ 通过仿真实现对建议算法、UI 设计的“what‑if”评估，减少长时间用户研究。

**🔧 技术方法**

采用计算机理性框架的层级强化学习（PPO + LSTM），对键盘与建议列表的视觉与手部动作建模，使用自定义可调的建议生成器以及贝叶斯优化调参。

**📊 数据集**

主要使用 WS‑Gaze 数据集（含键盘输入与注视数据的 1,080 个单词）和 How‑we‑type‑mobile 数据集作为无建议基准。

**📈 对比分析**

通过与真实用户在多项指标（词建议使用率、注视比例、错误率、打字速度、按键节省率等）对比，模型与人类在大多数指标内差距≤1SD，优于前置的 CRTypist 在无建议场景下的速度与注视分布。

**⚠️ 局限性**

模型简化了低级视觉与手部动作细节；缺乏对更大样本、多设备、不同人群（老年、残障）的验证；对建议列表的读取假设过于理想，未模拟真实眼动识别噪声。

---

## 209. Codes for Metastability-Containing Addition

**arXiv ID:** 2602.06467 | [PDF](https://arxiv.org/pdf/2602.06467v1)

**作者:** Johannes Bund `[一作]` (Université Paris-Saclay), Moti Medina `[通讯]` (Bar-Ilan University)

**通讯引用:** 336 | [OpenAlex ID](https://openalex.org/A5007248799)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究在存在不确定性（尤其是由亚稳态引起的）条件下的加法问题，提出一种新的混合编码（n,k混合码）和对应的元件级加法器，能够在不放大不确定性的前提下完成区间加法；

**💡 创新点**

提出了可恢复（recoverable）码的概念，并证明了对任意给定不确定性上限k，n,k混合码既是k‑preserving又是⌈k/2⌉‑recoverable，且其码率达到上界；

**🔧 技术方法**

利用三值逻辑、元件级亚稳态容忍（metastability‑containing）电路的理论，构造了从BRGC与热码到二进制的双向译码电路，并通过黑盒技术（Ikenmeyer等的k‑bit mc构造）得到最终的mc加法器；

**📊 数据集**

论文中未使用公开数据集，主要通过VHDL仿真验证所提电路在最坏情况亚稳态下的功能正确性；

**📈 对比分析**

相对于传统二进制编码（不保持精度）和仅使用单热码的直线编码（码率低、延迟线性），n,k混合码的码率为1−O(k/n)，mc加法器的面积为O((n+k)^k)（或O(k(n+k))使用掩蔽寄存器），延迟为O(log n+log k)，在k为常数时面积仅为多项式级，且不显著增加深度；

**⚠️ 局限性**

恢复精度上限仅为⌈k/2⌉，无法完全恢复全部不确定位；对较大k时的面积与复杂度仍然呈指数增长；尚缺乏完整的芯片实现与布局级性能评估，实际电路功耗与面积需进一步研究。

---

## 210. Diffusion-State Policy Optimization for Masked Diffusion Language Models

**arXiv ID:** 2602.06462 | [PDF](https://arxiv.org/pdf/2602.06462v1)

**作者:** Daisuke Oba `[一作]` (Institute of Science Tokyo), Naoaki Okazaki `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3519 | [OpenAlex ID](https://openalex.org/A5066940046)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Diffusion‑State Policy Optimization (DiSPO) 的中间状态信用分配层，用于在掩码扩散语言模型的填充步骤中直接优化中间决策。

**💡 创新点**

创新点在于将中间掩码状态视为决策点，利用已缓存的 logits 重新采样同一状态下的填充方案，形成“同状态分支”，从而在不增加额外扩散回路的情况下实现基于状态的奖励比较和梯度更新。

**🔧 技术方法**

采用了政策梯度方法、掩码‑token 替代似然（masked‑token surrogate）、分支采样、终端奖励评估以及与终端反馈的混合目标，整体实现了可插拔的训练框架。

**📊 数据集**

主要在 LLaDA‑8B‑Instruct 模型上进行实验，使用数学推理数据集 GSM8K、MATH500 以及符号规划任务 4×4 Sudoku 与 Countdown。

**📈 对比分析**

与基线 diffu‑GRPO（仅使用终端奖励的 PPO‑style 更新）在相同的扩散回路数和优化步骤下对比，DiSPO 在所有四个任务上均取得了显著提升，尤其在 256 步生成长度下平均提高 3–5% 的准确率，并且在训练时间匹配的情形下更快达到或超过基线。

**⚠️ 局限性**

局限性包括：1）实现仍不够高效，训练速度约为基线的 0.4 倍；2）仅在单一终端奖励环境下验证，未探索与奖励塑形或更高级终端目标的组合；3）对极端长生成长度或更大模型规模的鲁棒性尚未评估。

---

## 211. The Window Dilemma: Why Concept Drift Detection is Ill-Posed

**arXiv ID:** 2602.06456 | [PDF](https://arxiv.org/pdf/2602.06456v1)

**作者:** Brandon Gower-Winter `[一作]` (Utrecht University), Georg Krempl `[通讯]` (Utrecht University)

**通讯引用:** 611 | [OpenAlex ID](https://openalex.org/A5077689739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文探讨概念漂移检测中窗口选择（Window Dilemma）导致的漂移感知偏差，并证明漂移检测本质上是一个不确定的二分类问题；

**💡 创新点**

创新点在于揭示窗口选择本身即决定漂移感知，阐明漂移检测是“ill-posed”的，并用实验显示传统批量学习往往优于漂移感知模型；

**🔧 技术方法**

使用了DDM、ADWIN、D3、IBDD等漂移检测器，以及Naïve Bayes、Hoeffding Tree、Adaptive Random Forest、Aggregated Mondrian Forest、Batch Random Forest等分类器；

**📊 数据集**

实验数据集来源于USP Data Stream Repository，涵盖11个二分类和多分类数据集，如Electricity、ForestCoverType、Insects、Keystroke、Luxembourg、MIRS、NOAA Weather、Ozone、Rialto、Yoga等；

**📈 对比分析**

通过预序评估对漂移感知与漂移无关的在线/批量分类器进行比较，结果表明漂移感知分类器并不一定优于简单的重置或增量训练模型，且Batch Random Forest等批量学习模型往往表现更好；

**⚠️ 局限性**

限制在于缺乏真实漂移标注、过度依赖合成/半合成评估、未考虑漂移的“何时/何地/何因”细节，导致漂移检测的实用性和泛化性受限。

---

## 212. Implications of Russia's full-scale invasion of Ukraine for the international mobility of Ukrainian scholars

**arXiv ID:** 2602.06510 | [PDF](https://arxiv.org/pdf/2602.06510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 213. On the Plasticity and Stability for Post-Training Large Language Models

**arXiv ID:** 2602.06453 | [PDF](https://arxiv.org/pdf/2602.06453v1)

**作者:** Wenwen Qiang `[一作]` (Institute of Software Chinese Academy of Sciences), Hui Xiong `[通讯]` (Thrust of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Probabilistic Conflict Resolution（PCR）框架，利用贝叶斯推断对GRPO中因梯度冲突导致的不稳定性进行软投影，从而实现更平滑、稳定的后训练。

**💡 创新点**

创新点在于：①将梯度建模为高斯随机变量，以量化不确定性；②通过贝叶斯仲裁动态调节冲突分量的保留比例；③采用仅在MLP层执行PCR的混合策略，兼顾效率与稳定性。

**🔧 技术方法**

技术手段包括：贝叶斯推断、概率梯度建模、软投影（soft projection）、混合更新策略以及标准的GRPO、PCGrad等对比算法。

**📊 数据集**

使用的数据集涵盖：数学推理（AIME、AMC、MATH500、MinervaMATH）以及代码生成（HumanEval），并在多个基础模型（DeepScaleR‑1.5B、DeepSeek‑R1‑Qwen‑1.5B/7B、Qwen2‑7B）上进行实验。

**📈 对比分析**

与传统RL后训练基线（GRPO、GVPO、MRT、GCPO）比较，PCR在所有评测任务上均实现了平均约3%至5%（如AIME/MinervaMATH）的提升，且保持了更好的语言稳定性（WikiText‑2 PPL）。

**⚠️ 局限性**

局限性包括：①依赖高斯近似，可能无法捕捉极端噪声；②仍需手动设定KL系数β；③混合策略在大模型上虽减小开销，但对完全不涉及MLP层的任务效果待验证。

---

## 214. Principle-Evolvable Scientific Discovery via Uncertainty Minimization

**arXiv ID:** 2602.06448 | [PDF](https://arxiv.org/pdf/2602.06448v1)

**作者:** Yingming Pu `[一作]` (Westlake University), Hongyu Chen `[通讯]` (Westlake University)

**通讯引用:** 48075 | [OpenAlex ID](https://openalex.org/A5115601812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于不确定性最小化的可演化原理框架（PEF），将科学发现转化为在可扩展原理空间上的贝叶斯优化过程，自动在实验反馈中演化科学原理并指导假设搜索。

**💡 创新点**

创新点在于：①将科学原理视为可学习的概率先验，而非固定约束；②引入基于高斯过程的似然模型和信息导向假设选择（IDS），在探索-利用之间实现可控平衡；③设计异常驱动的“一致性扩展”机制，利用高惊奇实验结果主动扩充原理空间，实现自我修正和突破传统静态假设搜索的局限。

**🔧 技术方法**

技术主要包括：贝叶斯推理、Gaussian Process（GP）专家、信息导向采样（Information-Directed Sampling）、异常检测与阈值策略、LLM prompt 注入的战略层、以及协同优化的双循环结构。

**📊 数据集**

使用四个基准数据集进行实验：(1) 纳米材料光学属性优化（NHO），(2) 分子生物活性优化（MBO），(3) 超导临界温度优化（SPO），(4) 转移金属配合物优化（TMC）。每个任务均采用高保真 surrogate 模型作为实验环境。

**📈 对比分析**

与六类基线（Vanilla MAS、ReAct、The AI Scientist v1/v2、AI Researcher、PiFlow）比较，结果显示：①平均解决质量提升至 90.81%~93.15%，比最优基线提升 29.7%~31.1%；②在 MBO 任务中实现 83.3% 的收敛步长加速；③在所有任务和不同 LLM（Qwen3-32B、Gemini-2.5-Flash）上保持稳健性能；④表现出更高的探索多样性（APD）与优化曲线面积（AUOC）双重优势。

**⚠️ 局限性**

局限性包括：①实验依赖于 surrogate 模型，缺乏真实实验验证；②异常阈值、GP 参数等需要手动调优，敏感性分析仍有限；③对大规模高维任务的可扩展性尚未充分验证；④当前框架主要针对可解释性较强的科学原理，复杂非线性或多尺度系统可能需要进一步改进。

---

## 215. World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy

**arXiv ID:** 2602.06508 | [PDF](https://arxiv.org/pdf/2602.06508v1)

**作者:** Xiaokang Liu `[一作]` (Show Lab, National University of Singapore), Mike Zheng Shou `[通讯]` (Show Lab, National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作设计并实现了闭环框架，通过联合优化视频世界模型和视觉语言动作（VLA）策略，实现了机器人策略的强化学习。

**💡 创新点**

创新点在于引入状态感知的视频世界模型、近成功轨迹数据集以及奖励预测头，构建了模型与策略的共进化循环，从而显著提升动作跟随精度和奖励可靠性。

**🔧 技术方法**

技术上采用了基于Cosmos-Predict 2的扩散变压器视频生成器、奖励预测头、GRPO强化学习、OpenVLA-OFT VLA模型，并结合近成功数据进行训练。

**📊 数据集**

使用的数据集包括 ManiSkill、LIBERO 以及自制实机实验数据，均包含成功与近成功轨迹的多模态记录。

**📈 对比分析**

与传统 VLA‑RL、SimpleVLA‑RL 等方法在 LIBERO 与实机任务上进行对比，RL 后训练使 VLA 成功率提升约 12.7%–23.4%，在实机任务中迭代后达到 50% 的成功率，显著优于基线。

**⚠️ 局限性**

局限性在于自回归视频模型在长时延（>200 帧）任务中易出现质量漂移，且奖励仍以稀疏终态为主，未来需要改进长程记忆与中间子目标奖励机制。

---

## 216. DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving

**arXiv ID:** 2602.06502 | [PDF](https://arxiv.org/pdf/2602.06502v1)

**作者:** Ying Yuan `[一作]` (Huazhong University of Science and Technology), Zhou Yu `[通讯]` (Huawei)

**通讯引用:** 368124 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DualMap，一种在分布式大模型推理中同时实现 KV 缓存复用（cache affinity）和负载均衡的调度框架。

**💡 创新点**

创新点在于：① 双映射（dual‑hash）结合“两个选择”原则，既保证相同前缀请求聚集，又实现全局负载平衡；② SLO‑aware 路由在保持缓存复用的前提下，动态切换至负载优先；③ 热点感知重平衡在保持映射一致性的同时，迁移过载节点的请求；④ 双哈希环可扩展性大幅降低扩容时的缓存失效。

**🔧 技术方法**

核心技术包括：双独立哈希函数 + 预取缓存；SLO‑aware 路由算法；热点感知请求迁移；双哈希环（consistent hashing）实现轻量级扩容；以及基于 Power‑of‑Two‑Choices 的负载均衡理论。

**📊 数据集**

实验使用 Mooncake 收集的两份真实工作负载：Conversation（多轮对话）和 Tool&Agent（工具调用）轨迹，配合 Qwen2.5‑7B/14B 模型进行评测。

**📈 对比分析**

对比 Cache‑Affinity、Least‑Loaded、Min‑TTFT、Preble 四种基线，DualMap 在相同 5s TTFT SLO 下，提升有效请求容量最高 2.25×，P90 TTFT 和 P90 E2E 延迟显著下降（约 80‑97%），且在高 QPS 场景仍保持低延迟稳定性。

**⚠️ 局限性**

局限性包括：① 依赖前缀哈希，极端热点或前缀长度不确定时仍可能出现缓存失效；② 重平衡仅在双映射范围内迁移，未考虑跨多实例的更大规模负载迁移；③ 对多模型、多租户环境的通用性和系统复杂度尚需进一步验证。

---

## 217. Beyond the Majority: Long-tail Imitation Learning for Robotic Manipulation

**arXiv ID:** 2602.06512 | [PDF](https://arxiv.org/pdf/2602.06512v1)

**作者:** Junhong Zhu `[一作]` (University of Electronic Science and Technology of China), Heng Tao Shen `[通讯]` (Tongji University)

**通讯引用:** 30691 | [OpenAlex ID](https://openalex.org/A5052993469)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

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

## 218. BouquetFL: Emulating diverse participant hardware in Federated Learning

**arXiv ID:** 2602.06498 | [PDF](https://arxiv.org/pdf/2602.06498v1)

**作者:** Arno Geimer `[一作]` `[通讯]`, Arno Geimer

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

在 Flower 框架中实现了单机硬件仿真，能够按需求限制 CPU、GPU、内存资源，模拟不同设备的异构性；

**💡 创新点**

首次在单台机器上直接对 CPU/GPU/内存做完整的硬件限制并集成到 Flower，同时提供基于 Steam 硬件调查的随机采样器；

**🔧 技术方法**

利用 CUDA MPS、CPU 时钟/内存限制、Linux 资源控制等技术；

**📊 数据集**

使用 ResNet‑18 在不同 GPU 生成的仿真环境中训练，借助 PassMark/ UserBenchmark 游戏基准做对比；

**📈 对比分析**

将仿真训练时间与实际游戏基准进行归一化比较，Spearman ρ = 0.92、Kendall τ = 0.80，表明仿真能较好复现各代 GPU 的相对性能；

**⚠️ 局限性**

只能按全局限制，导致客户端必须串行执行；仅支持 Linux + NVIDIA GPU + root 权限，无法精确模拟缓存、PCIe 带宽等细节。

---

## 219. Subgraph Reconstruction Attacks on Graph RAG Deployments with Practical Defenses

**arXiv ID:** 2602.06495 | [PDF](https://arxiv.org/pdf/2602.06495v1)

**作者:** Minkyoo Song `[一作]` (Korea Advanced Institute of Science and Technology), Sooel Son `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1115 | [OpenAlex ID](https://openalex.org/A5082893706)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对受防护的图形检索增强生成 (Graph RAG) 系统，提出一种闭盒多轮子图重构攻击方法。

**💡 创新点**

创新点在于将提取任务重构为合法上下文处理、使用实例 ID 分隔保证关系完整、以及基于动量的多模板调度实现在查询预算内高效发现。

**🔧 技术方法**

技术主要包括任务重构提示、四字段四元组输出格式、动态多样化模板、动量感知调度器、以及对图检索与 LLM 推理的多轮交互。

**📊 数据集**

实验使用两个真实知识图数据集：Enron 企业邮件图和 HealthCareMagic 医疗对话图。

**📈 对比分析**

与六种基线（P1–P4、Worm、FG）比较，在四种安全对齐 LLM 上，攻击在防护环境下可达 82.9% RType F1，显著优于基线；但在更强的安全提示或抑制时性能下降。

**⚠️ 局限性**

局限包括对更严格的安全提示或解码层防御仍有一定泄露、对更复杂图结构（多重关系、隐式关系）适应性不足、以及在极低查询预算下召回仍受限。

---

## 220. DreamHome-Pano: Design-Aware and Conflict-Free Panoramic Interior Generation

**arXiv ID:** 2602.06494 | [PDF](https://arxiv.org/pdf/2602.06494v1)

**作者:** Lulu Chen `[一作]`, Yue Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提供了一个可控的全景室内设计生成框架DreamHome-Pano，解决结构与风格冲突问题

**💡 创新点**

提出Prompt‑LLM和冲突自由控制策略，分离结构与美学并利用专业语义提示

**🔧 技术方法**

基于扩散Transformer、Prompt‑LLM、空房正则化正向图、实例分割、RL（DPO/NFT）等技术

**📊 数据集**

使用约2.55M原始全景图，后筛选至12k专业图和100k高质量图，并构建50布局10风格基准集

**📈 对比分析**

与Seedream 4.5、Gemini 3 Pro Image等基线在空间一致性、HPSv3、CLIP、OmniAID上均取得更高分，尤其空间一致性提升至0.965，Aesthetic 0.68以上

**⚠️ 局限性**

仍存在局部细节失真、轻微几何扭曲和装饰物摆放不够物理合理

---

## 221. AgentCPM-Explore: Realizing Long-Horizon Deep Exploration for Edge-Scale Agents

**arXiv ID:** 2602.06485 | [PDF](https://arxiv.org/pdf/2602.06485v1)

**作者:** Haotian Chen `[一作]`, Maosong Sun `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对 4B 参数规模的边缘智能 LLM 代理模型进行系统研究，并提出 AgentCPM-Explore 方案；

**💡 创新点**

创新点在于三阶段训练框架：参数空间模型融合、奖励信号去噪与上下文信息精炼，从而缓解小模型的灾难性遗忘、奖励噪声敏感和长上下文信息污染；

**🔧 技术方法**

技术包括 DELLA 参数融合、三层奖励过滤（环境噪声、格式错误、极端轨迹）以及双循环 RL‑基意图对齐与教师模型蒸馏的上下文精炼；

**📊 数据集**

使用 Qwen3‑4B‑Thinking‑2507 作为基线模型，并在 GAIA、HLE、BrowserComp、WebWalker、FRAMES、XBench‑DeepResearch、SEAL‑0 等 8 个代理评测基准上进行实验；

**📈 对比分析**

与同规模 4B 模型相比，AgentCPM‑Explore 在所有基准上均为最优；在 8B 规模 SOTA 模型上匹配或超越，并在 30B 规模模型上取得领先；GAIA 文本任务 Pass@64 达到 97.09%；

**⚠️ 局限性**

局限性在于仍依赖大规模预训练模型作为基础，且在 Pass@1 等单次采样场景中模型稳定性不足；未来需进一步提升精确的价值评估与自我评估能力。

---

## 222. EMG-to-Speech with Fewer Channels

**arXiv ID:** 2602.06460 | [PDF](https://arxiv.org/pdf/2602.06460v1)

**作者:** Injune Hwang `[一作]` (Seoul National University), Kyogu Lee `[通讯]` (Seoul National University)

**通讯引用:** 1945 | [OpenAlex ID](https://openalex.org/A5088852010)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文系统研究了表面 EMG 通道在无声语音合成中的重要性，评估单通道与组合通道对性能的影响，并提出一种基于全通道预训练加随机通道丢弃的微调策略，以在通道数减少时保持合成质量。

**💡 创新点**

创新点在于：①系统性地从贪心后向消除、全穷举子集和音素级消融三种角度评估通道贡献，揭示通道间的互补关系；②证明在全通道预训练中加入随机通道丢弃，并在少通道下微调，能够显著抵消通道减少导致的性能下降。

**🔧 技术方法**

使用技术包括：卷积-Transformer 编码器、Mel 频谱目标与音素分类损失、HiFi‑GAN 语音重构、Whisper ASR 评估 WER、以及通道随机丢弃（p=0.125/0.25）和微调流程。

**📊 数据集**

数据集为 Gaddy 等公开的 8 通道 EMG 语料库（约 20 小时开词汇录音），配有同步语音、音素及单词标签。

**📈 对比分析**

实验通过与全通道基线、贪心/全穷举子集以及从零训练模型比较，发现 7 通道下性能与全通道相当；在 4–6 通道下，微调模型优于从零训练，且 WER 仅比全通道低 10–20%，证明了该策略的有效性。

**⚠️ 局限性**

局限性包括：仅在单一说话人、固定通道布局下验证；缺乏多说话人与动态佩戴鲁棒性评估；实验聚焦于合成效果，对实时系统与噪声环境下的鲁棒性未做深入验证。

---

## 223. Joint Lossy Compression for a Vector Gaussian Source under Individual Distortion Criteria

**arXiv ID:** 2602.06464 | [PDF](https://arxiv.org/pdf/2602.06464v1)

**作者:** Shuao Chen `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22777 | [OpenAlex ID](https://openalex.org/A5100447801)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在逐分量失真约束下，向量高斯源的联合有损压缩，并在半正定条件（SDC）满足与不满足两种情况下推导了新的信息理论极限。

**💡 创新点**

创新点在于：①在 SDC 满足时对原 Hadamard 下界做了精细化，②在 SDC 不满足时给出了重构维数上界并证明需要低维重构；③引入可扩展的两型相关（2TC）协方差模型，完整刻画了 SDC 区域并得到闭式率-失真函数；④分析了 SDC 满足概率随源维度指数衰减及相关系数上界的渐近形式。

**🔧 技术方法**

主要技术包括信息论中的最大行列式（Max‑Det）优化、KKT 条件分析、马尔可夫链后向测试通道模型、矩阵行列式定理、Sylvester 判据与柯西插值定理，以及对 2TC 协方差的行列式和逆矩阵显式求解。

**📊 数据集**

实验使用合成的 N 维高斯向量数据（各分量方差已归一化），并在不同 N、相关系数 ρ、失真约束 e 取值下进行 Monte‑Carlo 仿真。

**📈 对比分析**

与理论推导的闭式结果对比，仿真显示：①在满足 SDC 的情况下压缩率与理论一致；②在 SDC 不满足时低维重构能显著降低压缩率；③利用相关性可使每个分量的平均压缩率下降 13–64%（取决于 N 和 ρ），验证了理论上对相关性的量化收益。

**⚠️ 局限性**

局限性包括：①仅在非负相关且 2TC 结构下给出闭式结果，扩展到一般协方差需更复杂；②高维下求解 SDC 区域需要根号高次多项式，实际需数值近似；③假设源为高斯且失真为均方误差，真实信号可能偏离这些假设。

---

## 224. User-Centric Object Navigation: A Benchmark with Integrated User Habits for Personalized Embodied Object Search

**arXiv ID:** 2602.06459 | [PDF](https://arxiv.org/pdf/2602.06459v1)

**作者:** Hongcheng Wang `[一作]` (Peking University), Hao Dong `[通讯]` (Peking University)

**通讯引用:** 56203 | [OpenAlex ID](https://openalex.org/A5100425709)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了用户中心对象导航基准 UcON，构建了 22.6 万条用户习惯与 489 类目标的习惯驱动场景，并在此基础上设计了习惯检索模块 (HRM) 来帮助 LLM 更好地利用习惯信息。

**💡 创新点**

创新点在于：①首次将用户习惯作为先验知识系统化评估对象导航；②规模化收集并验证了海量用户习惯；③提出简洁的习惯检索机制，将相关习惯提取出来增强 LLM 推理；④通过实验验证习惯对导航的显著正向作用。

**🔧 技术方法**

使用的技术包括：大语言模型（GPT‑4、LLaMA‑3）、视觉语言模型与对象检测器（YOLO‑Worldv2‑X、SAM）、检索模型 BGE‑M3、Omnigibson 仿真、基于深度图的 BFS 路径规划以及各种基线方法（PixelNav、LGX、L3MVN、VTN、ZSON）。

**📊 数据集**

数据集：①合成的用户习惯 + 目标位置（22.6k 条习惯、489 类目标）构成的 UcON；②Omnigibson 22 场景用于仿真；③真实环境实验基于 Unitree GO2 Edu + Astra Pro Plus 摄像头的 3 个实验房间。

**📈 对比分析**

对比方法：PixelNav、LGX、L3MVN、VTN、ZSON 等 SOTA 方法；评价指标为成功率 (SR) 与路径效率 (SPL)。实验表明：在无习惯的情况下，SOTA 方法在 UcON 上 SR 明显下降；加入习惯后 SR 提升明显；HRM 的检索习惯往往比完整 GT 习惯集更能提升 SR，体现检索机制的有效性。

**⚠️ 局限性**

局限性：①习惯合成仅基于 GPT‑4，缺乏真实用户的长尾多样性与时序漂移；②只关注习惯利用，未研究习惯的持续学习与更新；③HRM 仅做简单检索，未考虑时间/位置上下文、长期记忆、多步规划与证据更新；④真实世界验证规模有限，尚未充分验证跨域适用性。

---

## 225. Achieving Better Local Regret Bound for Online Non-Convex Bilevel Optimization

**arXiv ID:** 2602.06457 | [PDF](https://arxiv.org/pdf/2602.06457v1)

**作者:** Tingkai Jia `[一作]` (East China Normal University), Cheng Chen `[通讯]` (East China Normal University)

**通讯引用:** 13100 | [OpenAlex ID](https://openalex.org/A5100420499)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了三种新的在线双层优化算法——AOBO、FSOBO和WOBO，并给出了它们在标准双层局部遗憾和窗口平均双层局部遗憾下的最优或更优的理论上界。

**💡 创新点**

创新点：①动态内循环迭代实现对内层最优漂移的自适应跟踪，取得标准双层局部遗憾的最优下界Ω(1+V_T)；②提出完全单循环实现的FSOBO，显著降低计算复杂度并给出更优的上界；③引入窗口平均双层局部遗憾，消除对子线性环境变化的假设，并证明该窗口遗憾下的下界为Ω(T/W^2)，从而实现最优性能。

**🔧 技术方法**

主要技术：使用AID（近似隐式微分）估计超梯度；在AOBO中采用动态内循环梯度下降追踪内层最优；在FSOBO中实现单循环更新并加入梯度、Hessian与Hessian‑向量乘积的约束；在WOBO中利用窗口加权平均的目标和内层子问题，并采用误差容忍条件控制迭代。

**📊 数据集**

实验数据集：MNIST图像分类数据集，分别在在线去噪（hyper‑cleaning）和不平衡数据的参数化损失调优（light‑CNN）两种任务上进行评估。

**📈 对比分析**

与基线（OBBO、SOBOW、SOGD等）比较结果显示：AOBO在累计局部遗憾上最优；FSOBO在计算速度上最快；WOBO通过调节窗口大小w和权重参数η能够显著降低遗憾，实验中表现出更低的累计遗憾和更高的测试准确率。

**⚠️ 局限性**

局限性：①仅在确定性环境下给出最优结果，对随机噪声的分析较弱；②对内层强凸、Lipschitz等假设要求严格；③AOBO的内循环梯度查询量为O(TlogT)，在大规模问题中计算成本仍高；④窗口平均遗憾对窗口大小和权重敏感，需要预先设定。

---

## 226. What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution

**arXiv ID:** 2602.06450 | [PDF](https://arxiv.org/pdf/2602.06450v1)

**作者:** Xingsong Ye `[一作]` (Fudan University), Zhineng Chen `[通讯]` (Fudan University)

**通讯引用:** 2010 | [OpenAlex ID](https://openalex.org/A5012931772)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 UnionST 合成引擎和联合式自演进学习框架，用于生成覆盖多种挑战场景的高质量场景文字合成数据，并用这些数据训练 STR 模型。

**💡 创新点**

创新点在于：①面向多样文本语料、字体、布局的渲染式合成；②构建 UnionST‑S、UnionST‑P 以及自演进 SEL 机制；③显著缩小合成与真实数据差距并大幅降低标注成本。

**🔧 技术方法**

技术方法主要包括：字符级渲染与位置/方向/尺寸自由变形、曲线与多方向文本合成、字体库扩充与语料增强、在线数据增强（DTAug）、伪标签生成与自迭代微调，以及基于 SVTRv2 的自回归解码器。

**📊 数据集**

使用的数据集包括公开合成集 MJ、ST、SynthAdd、CurvedST、SynthTIGER、UnrealText、TextSSR；基准集 Union14M‑Filter、Union14M‑Benchmark；以及自制的 UnionST‑S、UnionST‑P、UnionST‑SP 等。

**📈 对比分析**

对比实验显示：UnionST‑S 在 10M 样本下平均准确率可达 83%~91%（Union14M‑Benchmark），远优于现有合成集；仅用 1% 真实数据即可匹配完整数据集；SEL 进一步将标注成本降低 91%，并在 91.39% 的平均准确率上实现与全量真实数据相当的性能。

**⚠️ 局限性**

局限性包括：合成仍难以覆盖极端模糊、手写或多语种文本；自演进依赖高置信度伪标签，误标传播风险；以及在多语言、文档 OCR 等更广泛场景中的泛化性尚待验证。

---

## 227. TrajAD: Trajectory Anomaly Detection for Trustworthy LLM Agents

**arXiv ID:** 2602.06443 | [PDF](https://arxiv.org/pdf/2602.06443v1)

**作者:** Yibing Liu `[一作]` (Shandong University), Yilong Yin `[通讯]` (Shandong University)

**通讯引用:** 5729 | [OpenAlex ID](https://openalex.org/A5100672590)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了运行时轨迹异常检测任务，并构建了 TrajBench 数据集，训练了专门的轨迹验证模型 TrajAD，用于检测并定位代理执行过程中的错误。

**💡 创新点**

创新点在于（1）正式定义轨迹异常检测并要求精确错误定位；（2）通过 Perturb‑and‑Complete 机制自动生成高质量负样本并标注错误位置；（3）设计基于生成式 LLM 的轨迹验证器，结合低秩适配实现高效监督学习。

**🔧 技术方法**

使用的技术包括：Transformer（Qwen3‑4B 基础模型）+ LoRA 低秩适配、QLoRA 量化训练、生成式监督学习、异常类别细分（任务失败、流程低效、无谓继续）以及步骤级别精确定位。

**📊 数据集**

使用的数据集为 TrajBench，包含 60k+ 轨迹，覆盖 13 项任务、5 个核心领域（推理、数学、编程、网络导航、具身 AI），正负样本各占一半，手工评审表明标签可信度高。

**📈 对比分析**

与零样本通用 LLM（Qwen3‑4B、Gemma‑3‑4B‑Instruct、Phi‑3‑Mini）以及更大规模模型（Qwen3‑8B）对比，TrajAD 在异常检测宏 F1 上提升至 81.81%（比最高基线高 11.38%），在错误定位 JEM 上提升至 53.75%（比最高基线高 48.21%）。

**⚠️ 局限性**

局限性包括：对不同域的错误定位仍存在差距（跨域迁移时定位精度下降）；对极大规模数据或更大模型的提升有限；生成式监督仅基于人工构造的异常样本，可能无法覆盖所有真实世界的复杂错误。

---

## 228. Is Gradient Ascent Really Necessary? Memorize to Forget for Machine Unlearning

**arXiv ID:** 2602.06441 | [PDF](https://arxiv.org/pdf/2602.06441v1)

**作者:** Zhuo Huang `[一作]` (Sydney AI Centre), Tongliang Liu `[通讯]` (Sydney AI Centre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过先使用梯度下降加强对需要忘记数据的记忆，再利用模型外推得到的对称忘记模型，从而实现机器模型去学习，而不采用梯度上升。

**💡 创新点**

提出了模型外推（MOX）方法，利用记忆模型与参考模型的对称关系避免梯度上升导致的崩溃，并进一步引入针对性去学习和动量外推变体。

**🔧 技术方法**

使用梯度下降、KL约束、偏好优化（NPO/PO）、目标损失以及外推公式θ_for=(1+α)θ_ref-αθ_mem等技术。

**📊 数据集**

在TOFU和MUSE两个基准数据集上进行实验，使用Llama2‑7B和Phi‑1.5B大模型。

**📈 对比分析**

与GA、KL、GAD、NPO、TV、LLMU等多种基线方法对比，MOX（尤其是带动量的）在忘记质量、模型效用、隐私泄露等指标上均优于现有方法。

**⚠️ 局限性**

需要在α值上进行权衡，过大或过小会导致模型效用下降或不稳定；外推过程仍需额外计算和存储，且对极端α或数据规模的鲁棒性尚待进一步验证。

---

## 229. Forest canopy height estimation from satellite RGB imagery using large-scale airborne LiDAR-derived training data and monocular depth estimation

**arXiv ID:** 2602.06503 | [PDF](https://arxiv.org/pdf/2602.06503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 230. Evolutionary Generation of Multi-Agent Systems

**arXiv ID:** 2602.06511 | [PDF](https://arxiv.org/pdf/2602.06511v1)

**作者:** Yuntong Hu `[一作]` (Emory University), Stefano Soatto `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于配置的进化框架 EvoMAS，用于自动生成多智能体系统并通过执行反馈迭代改进。

**💡 创新点**

将多智能体系统结构抽象为可配置文本空间，并使用 LLM 引导的进化算子在该空间中搜索，兼顾可执行性与表达力。

**🔧 技术方法**

进化算法（选择、变异、交叉）+ LLM 辅助（配置生成、奖励评估、记忆总结）+ 结构化配置模型，支持多模型动态分配。

**📊 数据集**

BBEH、SWE‑Bench（Lite、Verified）以及 WorkBench。

**📈 对比分析**

与直接 LLM 调用、单智能体、人工设计 MAS、AutoAgents、MAS‑GPT、EvoAgent 等对比；在三种后端模型下，EvoMAS 在所有基准上均获得最高任务得分和执行率，甚至在使用最先进 LLM 时可匹敌或超越领先方法。

**⚠️ 局限性**

仅支持协作型 MAS；进化深度有限，计算开销仍高；对抗性/冲突角色未覆盖；依赖 LLM 评估可能带来主观性。

---

## 231. JADE: Expert-Grounded Dynamic Evaluation for Open-Ended Professional Tasks

**arXiv ID:** 2602.06486 | [PDF](https://arxiv.org/pdf/2602.06486v1)

**作者:** Lanbo Lin `[一作]` (Alibaba Group), Guannan Zhang `[通讯]` (Accio)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了JADE评估框架，解决专业任务评估的稳定‑适应性困境，并在BizBench上对多种代理进行系统评估。

**💡 创新点**

采用两层设计：层1通过专家知识技能激活生成查询特定检查表；层2通过报告特定检查表与证据级验证实现动态、门控评估，兼顾评估的稳定性和适应性。

**🔧 技术方法**

利用LLM（如GPT‑5、Gemini等）生成检查表与判断，配合检索验证Agent进行实时网页搜索与内容校验，并使用权重化分数与门控机制进行分数计算。

**📊 数据集**

主要使用BizBench（150条战略采购查询）进行评估，并在HealthBench（医学任务）上验证跨域迁移能力。

**📈 对比分析**

对比多种代理（Gemini Deep Research、ChatGPT 等），JADE平均得分42.6%，最高57.1%，显著提高与专家评分的相关性（r≈0.858），并将传统LLM‑as‑a‑judge的分数膨胀误差从≈31.6%降低至≈15%；评估结果方差低于1.2%。

**⚠️ 局限性**

需要专家投入设计标签/技能，域迁移成本高；评估流程多步骤，系统复杂；在知识密集型领域（如医学、教育）效果可能受限。

---

## 232. FloorplanVLM: A Vision-Language Model for Floorplan Vectorization

**arXiv ID:** 2602.06507 | [PDF](https://arxiv.org/pdf/2602.06507v1)

**作者:** Yuanqing Liu `[一作]` (Beike), Yue Yang `[通讯]` (Beike)

**通讯引用:** 770 | [OpenAlex ID](https://openalex.org/A5100369313)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将栅格平面图自动转为工程级向量图，输出结构化 JSON；

**💡 创新点**

提出像素到序列范式，将向量化建模为图像条件序列生成，并用强化学习实现几何对齐；

**🔧 技术方法**

利用大规模视觉语言模型（Qwen2.5‑VL）+ 自监督序列化 + 分阶段 SFT + GRPO 强化学习；

**📊 数据集**

构建 Floorplan‑2M（约200万条）和高精度子集 Floorplan‑HQ‑300K，并发布 FPBench‑2K 基准；

**📈 对比分析**

在 FPBench‑2K 上与基线对比，结构有效率 96.1%，外墙 IoU 0.925，非曼哈顿子集 90.27%，显著优于仅 SFT 模型；

**⚠️ 局限性**

推理延迟高、离散坐标精度受限、缺乏交互式编辑能力。

---

## 233. FCDP: Fully Cached Data Parallel for Communication-Avoiding Large-Scale Training

**arXiv ID:** 2602.06499 | [PDF](https://arxiv.org/pdf/2602.06499v1)

**作者:** Gyeongseo Park `[一作]` (Electronics and Telecommunications Research Institute), Ki-Dong Kang `[通讯]` (Electronics and Telecommunications Research Institute)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5072540697)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Fully Cached Data Parallel (FCDP) 框架，利用 host 内存缓存前向推理得到的参数，在反向传播中通过高速 PCIe 访问，消除重复的跨节点 all‑gather，保持 ZeRO‑3 的最小 GPU 内存占用。

**💡 创新点**

创新点包括：①将 host 内存视为高效缓存层，替代 GPU 缓存；②根据 GPU 内存压力动态决定参数放置位置；③对 PEFT 工作负载进行参数冻结/可训练分类，仅在训练阶段同步可训练参数，从而把 inter‑node 交互量降低 99%+；④实现 backward‑pass all‑gather 50% 的减少。

**🔧 技术方法**

技术手段：基于 ZeRO‑3 的全分片数据并行；GPU‑to‑CPU PCIe pinned transfer 与 NUMA‑aware 缓存；动态 GPU 内存监测与自适应放置；All‑Gather / Reduce‑Scatter 及 intra‑node 交换；使用 PyTorch + DeepSpeed 与 NCCL 进行实现。

**📊 数据集**

使用 SQuAD 数据集进行模型微调（全 fine‑tune 与 LoRA PEFT），并以 GPT‑2‑XL 规模扩展到 10B–30B 参数级别的 GPT 模型。

**📈 对比分析**

通过与 ZeRO‑3、ZeRO++ 在 commodity 4‑node 32‑GPU 集群上比较，测得全 fine‑tune 时 FCDP 最高可比 ZeRO‑3 提升 41.3% throughput、比 ZeRO++ 提升 2×；在 LoRA PEFT 下，FCDP 达到 6.2–6.8× 的 throughput 提升，100× 超过 ZeRO‑3，并显著减少 inter‑node 通信量（从 213 GB 降至 0.16 GB/step）。

**⚠️ 局限性**

局限性：依赖 PCIe 带宽和 NUMA 本地化，host 内存需足够大；需要额外的 host 缓存管理与同步逻辑；在极大规模 GPU 集群中，进一步优化跨节点通信与 host‑GPU 传输调度仍有挑战。

---

## 234. Adaptive Uncertainty-Aware Tree Search for Robust Reasoning

**arXiv ID:** 2602.06493 | [PDF](https://arxiv.org/pdf/2602.06493v1)

**作者:** Zeen Song `[一作]` (Institute of Software Chinese Academy of Sciences), Gang Hua `[通讯]` (Dolby Laboratories Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于不确定性感知的树搜索框架UATS，用于在推理时对大型语言模型的外部推理过程进行更可靠的路径评估与资源分配。

**💡 创新点**

创新点在于：①从理论上证明忽视PRM的不确定性会导致线性搜索损失，而加入不确定性可降为亚线性；②通过蒙特卡洛Dropout近似PRM的经验不确定性；③设计了启发式与强化学习控制的双层搜索策略，动态分配评估与扩展计算预算。

**🔧 技术方法**

主要技术包括：过程奖励模型（PRM）与蒙特卡洛Dropout估计不确定性；基于上置信界（UCB）的不确定性加权搜索；REBASE式预算分配；强化学习（REINFORCE）训练的自适应控制器。

**📊 数据集**

使用的评估数据集为数学推理基准MATH‑500和AIME24，另外在训练自适应控制器时随机采样多种政策模型与PRM组合。

**📈 对比分析**

与Beam Search、Best‑of‑N、REBASE、DORA、DVTS等基线对比，UATS在相同计算预算下准确率更高，尤其在低样本预算时表现尤为突出，实验表明可提升数个百分点。

**⚠️ 局限性**

局限性包括：对蒙特卡洛Dropout的假设依赖，需手动设定阈值与温度；自适应控制器训练复杂，易受奖励稀疏性影响；在极端分布漂移或非常大模型时，估计不确定性的效果可能下降。

---

## 235. Rebenchmarking Unsupervised Monocular 3D Occupancy Prediction

**arXiv ID:** 2602.06488 | [PDF](https://arxiv.org/pdf/2602.06488v1)

**作者:** Zizhan Guo `[一作]`, Rui Fan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于无监督单目图像的3D占据体预测方法，并改进了评测基准。

**💡 创新点**

创新点在于：① 用透明度（opacity）作为物理一致的占据概率；② 通过坐标变换采样将稀疏射线点映射到统一的体素网格；③ 设计了基于多视图颜色差异的遮挡感知极化损失，以显式指导被遮挡区域的占据学习。

**🔧 技术方法**

使用技术包括：神经辐射场（NeRF）渲染、体素网格采样、坐标变换与网格插值、彩色差异极化损失、以及多视图图像的光度一致性约束。

**📊 数据集**

主要使用的数据集是KITTI‑360（配合SSC​Bench‑KITTI‑360提供的3D占据体标注），并在SemanticKITTI上做零射实验验证通用性。

**📈 对比分析**

与现有无监督方法（BTS、KDBTS、KYN、ViPOcc）以及监督方法（MonoScene、TPVFormer、VoxFormer、OccFormer、Symphonies）比较，本文方法在O_Acc、O_Pre、O_Rec、IE_Acc、IE_Pre、IE_Rec等指标上均实现了SOTA，并在零射实验中超过了SceneRF、SelfOcc等模型，接近甚至超过部分监督模型的IoU。

**⚠️ 局限性**

局限性主要是对视角覆盖率（FOV）敏感，窄视角数据集如nuScenes会导致多视图监督信号不足，训练难以收敛；此外在极端遮挡或极端光照条件下仍可能出现误判。

---

## 236. Instance-Free Domain Adaptive Object Detection

**arXiv ID:** 2602.06484 | [PDF](https://arxiv.org/pdf/2602.06484v1)

**作者:** Hengfu Yu `[一作]` (University of Electronic Science and Technology of China), Wen Li `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 16680 | [OpenAlex ID](https://openalex.org/A5100320305)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种实例无目标实例的领域自适应目标检测方法，解决仅有背景图像时的跨域对齐问题。

**💡 创新点**

创新点在于利用背景原型对齐、相对空间协调和源结构保持三种约束，仅凭目标域背景信息实现有效的跨域对齐。

**🔧 技术方法**

技术实现基于 Faster R‑CNN 框架，采用对抗学习的背景原型对齐、相对空间一致性正则化和源结构保护损失。

**📊 数据集**

使用了三组自定义基准：IF‑CARLA（自动驾驶）、IF‑CCT（野生动物监测）和 IF‑LUNA16（肺结节检测）。

**📈 对比分析**

与六种主流 DAOD 方法（如 DAF、HTCN、AT、CAT 等）对比，在三组基准上平均提升 10.1%、6.9% 和 6.4% 的 mAP，明显优于传统方法。

**⚠️ 局限性**

局限性包括仍需源域标注、仅在三类场景验证，且在更大规模目标域背景或极端域差场景下的泛化能力待进一步探索。

---

## 237. FDD CSI Feedback under Finite Downlink Training: A Rate-Distortion Perspective

**arXiv ID:** 2602.06479 | [PDF](https://arxiv.org/pdf/2602.06479v1)

**作者:** Shuao Chen `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22777 | [OpenAlex ID](https://openalex.org/A5100447801)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了FDD多天线OFDM系统在有限下行训练下的CSI反馈理论极限，推导了整体率失真函数并证明其随训练长度以1/n_t收敛到理想直接CSI编码的极限。

**💡 创新点**

创新点在于把整体RDF拆成三部分（直接RDF、MMSE估计降低源不确定性导致的RDF下降、严格失真约束导致的RDF上升），并给出非渐近上界/下界以及对训练长度的1/n_t收敛性分析。

**🔧 技术方法**

使用MMSE估计、率失真理论、Wishart分布与逆Wishart分析、Jensen不等式、Woodbury恒等式以及矩阵迹、伪行列式等技术手段。

**📊 数据集**

采用仿真系统模型：4发射天线、8子载波、SNR下行10 dB、特定协方差矩阵（约1/3小/大/中等特征值）来验证理论。

**📈 对比分析**

与直接CSI编码的理论极限以及上下界进行比较，仿真结果表明在训练符号数增大时整体RDF快速逼近直接RDF，误差随1/n_t下降，验证理论界限的紧密性。

**⚠️ 局限性**

局限在于仅考虑全子载波训练、有限训练长度且假设可实现大延迟联合编码，未考虑极少训练子载波、极大天线数以及实时一拍反馈等实际系统约束。

---

## 238. Prism: Spectral Parameter Sharing for Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.06476 | [PDF](https://arxiv.org/pdf/2602.06476v1)

**作者:** Kyungbeom Kim `[一作]` (Gwangju Institute of Science and Technology), Kyung-Joong Kim `[通讯]` (Gwangju Institute of Science and Technology)

**通讯引用:** 2166 | [OpenAlex ID](https://openalex.org/A5076055880)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了 Prism，一种在谱域通过 SVD 分解与可学习谱掩码实现多智能体参数共享的框架。

**💡 创新点**

创新点在于：将共享网络参数映射到正交的奇异向量基上，仅通过调节奇异值的谱掩码实现代理间的多样化，从而兼顾可扩展性与策略异质性；并引入多样性与正交正则化以提升稳定性。

**🔧 技术方法**

采用的技术包括：SVD 参数化的低秩分解、谱空间掩码学习、直通估计（STE）实现可微多样性正则化，以及正交矩阵正则化。

**📊 数据集**

实验使用了 LBF、SMACv2（离散控制）和 MaMuJoCo（连续控制）三大基准数据集。

**📈 对比分析**

与 NoPS、FuPS、SePS、SNP、AdaPS、Kaleidoscope 等多种共享策略对比，Prism 在均衡或领先的性能下，尤其在资源受限或代理数量扩展时，显著降低了资源占用并保持竞争力。

**⚠️ 局限性**

局限性包括：需针对不同任务调节共享与个体谱比例的超参数；目前仅在离线（off‑policy）算法中验证，尚未扩展到在线（on‑policy）框架。

---

## 239. LAB-Det: Language as a Domain-Invariant Bridge for Training-Free One-Shot Domain Generalization in Object Detection

**arXiv ID:** 2602.06474 | [PDF](https://arxiv.org/pdf/2602.06474v1)

**作者:** Xu Zhang `[一作]` (University of Sydney), Dacheng Tao `[通讯]` (University of Sydney)

**通讯引用:** 98516 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种训练‑free 的单样本跨域目标检测框架，利用语言桥接专用领域。

**💡 创新点**

创新点在于将单一目标实例转化为描述性文本作为条件，完全替代梯度微调，实现无参数更新的跨域适应。

**🔧 技术方法**

采用 Describe Anything Model 生成文本提示，使用冻结的 Grounding DINO 进行候选框生成，并通过 BLIP 对小目标做轻量级校准。

**📊 数据集**

在 UODD（海底物种）和 NEU‑DET（工业缺陷）这两个低标注度、数据稀缺的数据集上进行评估。

**📈 对比分析**

与多种 FSOD、CD‑FSOD 及开源模型比较，LAB‑Det 在 UODD 取得 6.6 mAP、NEU‑DET 取得 9.0 mAP，均超过细调基线且无需更新任何参数。

**⚠️ 局限性**

主要局限是对外部描述模型的文本质量敏感，且目前仅支持单图像一‑shot，缺乏多样本、视频或 3D 扩展。

---

## 240. Improve Large Language Model Systems with User Logs

**arXiv ID:** 2602.06470 | [PDF](https://arxiv.org/pdf/2602.06470v1)

**作者:** Changyue Wang `[一作]` (Tsinghua University), Yiqun Liu `[通讯]` (Tsinghua University)

**通讯引用:** 9953 | [OpenAlex ID](https://openalex.org/A5100668121)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了UNO框架，使LLM系统能够利用用户日志进行持续学习，提炼规则，聚类并评估认知差距，随后构建主体验（LoRA微调）和反思体验（Critic LoRA）模块；

**💡 创新点**

通过将规则提炼、双特征聚类与认知差距评估相结合，动态选择直接参数微调或批判性反馈两条路径，解决了Signal‑or‑Noise困境；

**🔧 技术方法**

规则提炼（LLM-as‑Rule Extractor）、双特征层次聚类、语义距离与reranker的认知差距评估、LoRA参数高效微调、LLM-as‑Judge模拟验证、Critic LoRA反思模块；

**📊 数据集**

MemoryBench持续学习基准（含多语言、多任务、多域子任务），使用Qwen3‑8B和phi‑4模型；

**📈 对比分析**

与RAG、MemoryOS、ReMem、A‑Mem、Mem0、SFT、DPO等基线对比，UNO在Norm‑Score和Z‑Score上均显著领先，提升幅度多达10–20%且计算效率高；

**⚠️ 局限性**

依赖大量高质量用户日志，在线学习仍需人工或模拟器支持，长上下文处理仍存在挑战，额外的批判-重生成步骤会增加推理延迟。

---

## 241. CORE: Comprehensive Ontological Relation Evaluation for Large Language Models

**arXiv ID:** 2602.06446 | [PDF](https://arxiv.org/pdf/2602.06446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 242. Auditing Rust Crates Effectively

**arXiv ID:** 2602.06466 | [PDF](https://arxiv.org/pdf/2602.06466v1)

**作者:** Lydia Zoghbi `[一作]` (University of California), Caleb Stanford `[通讯]` (University of California)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

设计并实现了交互式 Rust crate 审计工具，自动识别潜在危险的 side‑effects 并支持上下文敏感的审计记录。

**💡 创新点**

创新点包括：① 将危险操作抽象为“Effect”，在 Rust 语义下进行静态分析；② 通过调用图传播上下文敏感安全信息并生成可复用的审计文件；③ 在 VSCode 中提供交互式审计体验，显著降低人工审计量。

**🔧 技术方法**

使用静态 AST、HIR/MIR 基础的模式匹配实现 Effect 分析，调用图构建与传播；利用 Rust 编译器插件/工具链；实现 VSCode 扩展以交互式展示审计点；采用基于效果模式的静态分析和数据流追踪。

**📊 数据集**

对 Crates.io 最高下载量前 10k 个 crate 进行全面分析，并以热门 HTTP 库 hyper 与 http 等及其所有依赖为实验样本，真实生产系统中使用的 crate 作为评测对象。

**📈 对比分析**

比较方法：将工具自动审计的代码量与传统手工审计进行对比；结果显示工具下审计代码占比仅为 0.2%（相较于全审计约 13%），并在 10k crate 中发现 3434 个无危险 Effect，85% 的危险集中在 3% crate；使用 AMD Ryzen 3960X 机器完成实验，审计时间从数天降至数小时。

**⚠️ 局限性**

局限性：仍需人工判断安全性，工具仅覆盖静态代码，无法审计构建脚本或增量变更；对泛型、trait 实现的推断可能产生误判；无法自动给出最终安全决策，只提供标注和上下文信息。

---

## 243. Exploring Specular Reflection Inconsistency for Generalizable Face Forgery Detection

**arXiv ID:** 2602.06452 | [PDF](https://arxiv.org/pdf/2602.06452v1)

**作者:** Hongyan Fei `[一作]` (Peking University), Jie Zhou `[通讯]` (Tencent Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于光照物理模型的深度伪造检测方法SRI-Net，聚焦面部高光反射的不易复制性，利用Retinex纹理估计实现精确分离光照与纹理，并通过双阶段跨注意力捕捉高光、纹理与直接光的关系，从而判定图像真实性。

**💡 创新点**

①首次将Phong模型中高光反射视为最难复制的伪造痕迹；②引入基于Retinex的快速纹理估计，显著提升高光分离精度；③设计两阶段跨注意力网络，实现多属性（高光、纹理、直接光）交互式特征融合；

**🔧 技术方法**

Retinex多尺度纹理估计、Spherical Harmonics光照建模、3DDFA 3D形状提取、Xception主干网络、跨注意力机制、形状标准化、图像分支融合。

**📊 数据集**

传统数据集：FaceForensics++（c23）、CelebDF_v1/v2、DeepfakeDetection；生成式数据集：DiFF（Stable Diffusion XL 等）、DF40（多种面部交换与编辑子集）。

**📈 对比分析**

与多种基准方法（Xception、EffiNet-B4、RECCE、SBI、UCF、FIA-USA 等）在帧级与视频级 AUC 上进行对比；SRI-Net 在传统数据集帧级最高 AUC 91.3%（CelebDF_v1）、87.5%（CelebDF_v2）、89.3%（DFD）以及视频级 95.5%/93.1%；在生成式数据集 DiFF 与 DF40 上平均 AUC 分别提升至 90.9% 与 86.8%，显著优于现有方法。

**⚠️ 局限性**

对高光分离仍依赖精确的 3D 形状估计，面对极端姿态、遮挡或低分辨率时可能效果下降；跨注意力和 3D 计算增加推理时间，对实时检测场景有一定挑战。

---

## 244. Evaluating an evidence-guided reinforcement learning framework in aligning light-parameter large language models with decision-making cognition in psychiatric clinical reasoning

**arXiv ID:** 2602.06449 | [PDF](https://arxiv.org/pdf/2602.06449v1)

**作者:** Xinxin Lin `[一作]` (Chinese University of Hong Kong), Lizhou Fan `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 490 | [OpenAlex ID](https://openalex.org/A5070395413)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了ClinMPO框架，利用专门构建的Evidence QA数据集和ClinRM奖励模型，通过强化学习实现轻量级LLM的临床推理对齐，并与人类医学学生进行对比评估。

**💡 创新点**

创新点在于将基于证据的奖励模型与多组策略优化相结合，直接将临床推理质量作为奖励信号，并通过相对优势计算避免过度追求表面语言流畅度。

**🔧 技术方法**

主要技术包括：1) 生成式预训练模型（Qwen系列）+LoRA进行SFT；2) 采用GRPO基础强化学习；3) 引入ClinRM奖励模型与临床一致性正则化实现ClinMPO；4) 统一多任务评估框架和专家人工评审。

**📊 数据集**

使用的数据集为：①公开QA数据集（来自CMB、MedMCQA、MedQA等）共8,849题；②Evidence QA数据集（18,569条目，基于4,474篇精神医学期刊文章，按Oxford证据等级构建）。

**📈 对比分析**

通过与300名医学学生的测试对照，ClinMPO在多层次临床分类上提升了约2.7个百分点的平均准确率，尤其在“精神、行为或神经发育障碍”和“共病与复杂性管理”等核心领域超越人类基准，且相对优势在不同规模模型中均保持正向。

**⚠️ 局限性**

局限性包括：①评估仅基于静态问答数据，缺乏动态临床情境验证；②奖励模型训练局限于后训练阶段，仍易受预训练分布影响；③对不同文化和社会经济背景的公平性、对抗鲁棒性等方面未做系统检验。

---

## 245. Selfish routing games with priority lanes

**arXiv ID:** 2602.06598 | [PDF](https://arxiv.org/pdf/2602.06598v1)

**作者:** Yang Li `[一作]` (Northwestern Polytechnical University), Marc Uetz `[通讯]` (University of Twente)

**通讯引用:** 1738 | [OpenAlex ID](https://openalex.org/A5052226987)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文构建了一个含优先/常规通道的非可分流网络路由模型，并在该模型中证明了在线性延迟函数下存在均衡、均衡的边延迟唯一且实现社会最优；

**💡 创新点**

创新点在于将优先收费机制转化为可选服务，证明在边特定的边际成本定价下可达到与强制拥堵费相同的社会最优，并阐明均匀定价无法实现此目标；

**🔧 技术方法**

主要技术包括Kakutani固定点定理证明均衡存在、变分不等式（VI）推导均衡边延迟唯一、边际成本定价与最优流的等价性证明；

**📊 数据集**

该工作为纯理论分析，无使用实验数据集；

**📈 对比分析**

通过理论对比：在边特定定价时，价格上限为1即实现PoA=1；而在均匀定价情况下，构造Pigou类网络示例，证明任意均匀定价的PoA上界可逼近4/3，说明其性能差距；

**⚠️ 局限性**

限制在于仅对线性延迟函数成立，非线性延迟的分析更复杂；同时优先选项作为可选服务，无法像强制收费那样彻底消除网络瓶颈。

---

## 246. ProtoQuant: Quantization of Prototypical Parts For General and Fine-Grained Image Classification

**arXiv ID:** 2602.06592 | [PDF](https://arxiv.org/pdf/2602.06592v1)

**作者:** Mikołaj Janusz `[一作]` (Jagiellonian University), Dawid Rymarczyk `[通讯]` (Ardigen SA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在冻结的高性能视觉模型（如ResNet、ConvNeXt、ViT）上引入向量量化的概念码本，实现可解释的原型推理头，既保持高分类准确性，又提供“这看起来像那样”的可视化解释。

**💡 创新点**

创新点在于：①将原始连续特征空间离散化为有限的概念码本，消除原型漂移；②不需要微调主干网络即可在大规模数据集（ImageNet）上实现可解释性；③通过两阶段训练（无监督码本学习+监督解释头）获得代表性稳定且可解释的原型。

**🔧 技术方法**

技术方法：VQ‑VAE式向量量化、概念匹配与概率聚合、稀疏非负分类矩阵、两阶段训练流程，结合冻结的ResNet/ConvNeXt/ViT主干。

**📊 数据集**

数据集：ImageNet‑1K、CUB‑200‑2011、Stanford Cars、Stanford Dogs、Oxford Flowers、FunnyBirds、Spatial Misalignment Benchmark。

**📈 对比分析**

与ProtoPNet、PIP‑Net、InfoDisent、ProtoViT等方法对比，ProtoQuant在ImageNet上接近原始模型准确率（≈80%），在细粒度数据集上与最优可解释方法相当或更优；在代表性稳定性指标（PAC、PRC、PLC）上显著优于对手，表明模型对输入扰动的鲁棒性更强。

**⚠️ 局限性**

局限性：离散码本可能限制表达灵活性，需手动选择码本容量；若主干特征语义结构不足，量化效果和解释质量可能下降；大型码本可能导致概念冗余、可解释性稀释。

---

## 247. The Impossibility of Strategyproof Rank Aggregation

**arXiv ID:** 2602.06582 | [PDF](https://arxiv.org/pdf/2602.06582v1)

**作者:** Manuel Eberl `[一作]` (University of Innsbruck), Patrick Lederer `[通讯]` (University of Amsterdam)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文证明了不存在满足匿名性、统一性以及基于Kemeny距离的策略无关性的社会福利函数（SWF），并进一步分析了常见SWF的可操纵程度，给出了与Gibbard‑Satterthwaite定理等价的不可行性定理；

**💡 创新点**

创新点在于首次用计算机辅助方法（SAT求解与Isabelle/HOL形式化验证）在多候选人、多选民场景下推出完全的不可行性定理，并结合激励比率评估SWF的操纵性；

**🔧 技术方法**

主要技术包括：1) SAT求解器对固定候选人/选民数目的可行性公式进行求解；2) 在Isabelle/HOL中进行交互式形式化验证；3) 激励比率（incentive ratio）理论分析；

**📊 数据集**

未使用真实数据集，所有论证均基于理论组合计数和对所有可能排列的计算机生成；

**📈 对比分析**

通过SAT求解在基例（m=5,n=2与m=4,n=4）仅需数秒即可证明不可满足，随后使用归纳证明推广到更大规模；性能表现优异，证明过程快速完成；

**⚠️ 局限性**

局限性包括：只对偶数选民（或特定倍数）给出结论，无法直接推广到奇数选民；仅适用于基于Kemeny距离的策略无关性，未覆盖随机或集合值SWF；

---

## 248. Linear Realisability and Implicative Algebras

**arXiv ID:** 2602.06576 | [PDF](https://arxiv.org/pdf/2602.06576v1)

**作者:** Alexandre Lucquin `[一作]` (Université Paris Nord), Thomas Seiller `[通讯]` (CNRS)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文通过引入线性可实现性模型与线性可实现性代数（Linear Implicative Algebra），将Girard的线性可实现性与Kleene、Krivine的可实现性理论在Miquel框架下正式关联，构建了可实现性与线性逻辑统一的代数化模型。

**💡 创新点**

创新点在于提出线性可实现性代数及其线性分离器，证明线性可实现性情景天然构成此代数，从而实现对线性逻辑与可实现性模型的统一与线性分解，并进一步扩展到指数、加法与固定点指数的形式化。

**🔧 技术方法**

所用技术包括Miquel的可实现性框架、蕴含代数与应用代数的定义、线性分离器与线性分离器生成、类型解释、λ‑演算与组合子构造，以及对线性可实现性情景的测量、执行与双正交闭包的抽象化。

**📊 数据集**

无数据集，论文完全以理论证明与模型构造为主。

**📈 对比分析**

无实验比较或性能评估；论文通过形式化证明与逻辑推导，论证线性可实现性模型与传统可实现性模型的一致性与互补性。

**⚠️ 局限性**

局限性在于仅适用于对称测量并存在单位与τ元的线性可实现性情景；对非对称测量或非交换执行的模型未覆盖，指数与加法的实现仍停留在理论层面。

---

## 249. The Law of Task-Achieving Body Motion: Axiomatizing Success of Robot Manipulation Actions

**arXiv ID:** 2602.06572 | [PDF](https://arxiv.org/pdf/2602.06572v1)

**作者:** Malte Huerkamp `[一作]` (AICOR Institute for Artificial Intelligence), Michael Beetz `[通讯]` (AICOR Institute for Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出“任务实现身体运动法则”，通过三元谓词（语义正确性、因果充分性、体态可行性）对机器人操纵动作进行形式化验证，并在厨房场景中用三款移动操作机器人演示其可行性。

**💡 创新点**

创新点在于将任务与运动、物理与数字孪生、机器人结构统一到一个可验证的公理框架中，实现对任何生成器的运动进行跨域、跨机型的语义、因果和可执行性诊断。

**🔧 技术方法**

使用的技术包括：Task–Environment–Embodiment (TEE) 类的刻画、语义数字孪生（SDT）结构化场景图、可扩展物理模型、谓词逻辑推理、逆运动学和规划工具（TRAC‑IK、MoveIt!）以及对机器人运动的约束优化。

**📊 数据集**

实验数据集为两套厨房场景的 3D 语义化模型（包含柜门、抽屉、把手等实体及其属性）和三款机器人（PR2、TIAGo++、Stretch）的运动学/动力学模型；没有使用公开机器人任务基准，而是自建的容器开启/关闭任务。

**📈 对比分析**

对比方法：直接用各机器人在相同场景下的运动规划失败率；结果显示 PR2 与 TIAGo++ 在所有容器开启任务中成功率均为 22/27/16/17，而 Stretch 只达到 19/27/15/17；同时利用法则验证不同机器人对同一任务的可行性差异，证明法则能够识别因体态限制导致的失败。

**⚠️ 局限性**

局限性包括：物理模型仅限于刚体运动学，未涵盖摩擦、弹性或流体等复杂动力学；置信区间的保守性可能导致合法动作被误判为超出范围；实验仅覆盖了厨房容器开启任务，未验证在更大规模或更动态场景中的泛化能力。

---

## 250. Topography scanning as a part of process monitoring in power cable insulation process

**arXiv ID:** 2602.06519 | [PDF](https://arxiv.org/pdf/2602.06519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 251. Live Knowledge Tracing: Real-Time Adaptation using Tabular Foundation Models

**arXiv ID:** 2602.06542 | [PDF](https://arxiv.org/pdf/2602.06542v1)

**作者:** Mounir Lbath `[一作]` (École Polytechnique), Jill-Jênn Vie `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出利用tabular foundation model（TFM）实现实时知识追踪（liveKT），无需离线训练即可预测学生下一个交互结果。

**💡 创新点**

创新点在于将TFM的上下文学习机制应用于知识追踪，通过双向注意力同时关注时间步与其他学生的交互，从而在推理时直接对齐训练与测试序列，完全跳过传统模型的训练阶段。

**🔧 技术方法**

采用TabPFN和TabICL等基于transformer的tabular foundation model，并对比传统DKT、AKT、LR、GBM等方法。

**📊 数据集**

使用了四个公开知识追踪数据集：ASSISTments 2009/2012、POJ、Codeforces。

**📈 对比分析**

在AUC和预测时延上，TFM在大部分数据集上与深度模型相当或更优，并实现最高可达273倍的速度提升；在POJ上甚至超越深度模型，在小样本冷启动情境中表现尤为突出。

**⚠️ 局限性**

局限性包括对预训练权重的依赖、缺乏对模型细节的可解释性以及在极大规模数据时仍需进一步优化，未来需要自行预训练更适配知识追踪分布的TFM。

---

## 252. Personality as Relational Infrastructure: User Perceptions of Personality-Trait-Infused LLM Messaging

**arXiv ID:** 2602.06596 | [PDF](https://arxiv.org/pdf/2602.06596v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 253. AgentCPM-Report: Interleaving Drafting and Deepening for Open-Ended Deep Research

**arXiv ID:** 2602.06540 | [PDF](https://arxiv.org/pdf/2602.06540v1)

**作者:** Yishan Li `[一作]`, Maosong Sun `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了本地化的深度研究报告生成系统 AgentCPM-Report，并构建了写作即推理（WARP）框架，实现计划与写作的动态迭代。

**💡 创新点**

创新点在于将计划与写作融为一体的写作即推理策略，突破洞察天花板，并通过多阶段代理训练使 8B 模型具备高效的深度研究能力。

**🔧 技术方法**

使用技术包括 WARP 写作即推理政策、Evidence-Based Drafting 与 Reasoning-Driven Deepening 两阶段循环、Multi-Stage Agentic Training（冷启动、原子技能 RL、整体管线 RL）以及轨迹剪枝等强化学习方法。

**📊 数据集**

使用数据集包括 DeepResearch Bench、DeepConsult、DeepResearch Gym 三大公开基准，并以 MiniCPM4.1‑8B 为基础模型。

**📈 对比分析**

通过与闭源系统（Gemini、Claude、OpenAI）、提示式框架（WebWeaver 等）以及开放模型（WebShaper、WebThinker、DR Tulu）对比，AgentCPM‑Report 在 Insight、Comprehensiveness、Readability 等指标上均超过 Gemini‑2.5‑Pro 等闭源系统，整体得分达到 50.11。

**⚠️ 局限性**

局限性包括无法实现高质量的表格/多模态呈现，知识库局限于文本且更新不及时，且模型仍需进一步分离内容生成与排版任务。

---

## 254. Primary Experimental Feedback on a Co-manipulated Robotic System for Assisted Cervical Surgery

**arXiv ID:** 2602.06541 | [PDF](https://arxiv.org/pdf/2602.06541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 255. Universal Anti-forensics Attack against Image Forgery Detection via Multi-modal Guidance

**arXiv ID:** 2602.06530 | [PDF](https://arxiv.org/pdf/2602.06530v1)

**作者:** Haipeng Li `[一作]` (Shenzhen University), Anastasia Antsiferova `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 101 | [OpenAlex ID](https://openalex.org/A5086393377)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ForgeryEraser，利用共享的 Vision‑Language 模型（CLIP）对多种 AIGC 检测器进行通用反取证攻击，既能降低检测准确率，又能诱导解释模型给出伪造图像的“真实”说明。

**💡 创新点**

首次揭示共享 VLM 作为特征空间导致的系统性脆弱性，并通过多模态指导损失将伪造图像嵌入向文本定义的真实锚点迁移，实现不依赖目标模型参数的全局攻击。

**🔧 技术方法**

采用多模态引导损失、源感知锚点构造、可微重采样、Momentum‑IFGSM 等技术，在 CLIP 的图像与文本编码器上进行优化。

**📊 数据集**

在全球合成与局部编辑两大范式下，使用 AIGCDetectBenchmark、FakeClue、UniversalFakeDetect、SID‑Set、Deepfake 协议等官方基准数据集进行评测。

**📈 对比分析**

与六个基于 CLIP 或其变体的顶尖 AIGC 检测器（SIDA、AIDE、FakeVLM、LEGION、Effort、Forensics Adapter）对比，攻击在 ϵ=8/255 下将检测准确率从 90%+ 降至 0–6%，并使解释模型给出与真实相符的理由。

**⚠️ 局限性**

攻击依赖共享 VLM，若检测器不使用 CLIP 或采用不同后端，效果会降低；极端图像失真下部分模型基线不稳定，且对不同生成器的跨域泛化尚未完全验证。

---

## 256. DiTS: Multimodal Diffusion Transformers Are Time Series Forecasters

**arXiv ID:** 2602.06597 | [PDF](https://arxiv.org/pdf/2602.06597v1)

**作者:** Haoran Zhang `[一作]` (Tsinghua University), Mingsheng Long `[通讯]` (Tsinghua University)

**通讯引用:** 29090 | [OpenAlex ID](https://openalex.org/A5019241553)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为DiTS的多模态扩散变换器，用于高维协变量感知的时间序列预测；

**💡 创新点**

创新点在于将内因子和外因子视为不同模态，采用双流注意力（时间注意力+变异注意力）实现跨维度低秩建模，并结合流匹配训练以获得更稳定的生成过程；

**🔧 技术方法**

使用扩散变换器架构、流匹配（flow matching）训练、AdaLN与联合注意力的条件注入、patch token化和重正化技术；

**📊 数据集**

在能源价格预测（EPF）和FEV领导者榜单等包含未来已知外因子的公共数据集上进行评测；

**📈 对比分析**

与PatchTST、TimeXer、iTransformer、Crossformer、DAG等基准模型比较，DiTS在均方误差（MSE）、平均绝对误差（MAE）以及概率预测指标（WQL、MASE）上均取得领先，提升幅度在10%以上；

**⚠️ 局限性**

局限性包括对高维多变量输入仍存在计算开销、对异常或缺失值的鲁棒性尚未充分验证，以及模型解释性和对长期预测的可扩展性需进一步研究。

---

## 257. Target noise: A pre-training based neural network initialization for efficient high resolution learning

**arXiv ID:** 2602.06585 | [PDF](https://arxiv.org/pdf/2602.06585v1)

**作者:** Shaowen Wang `[一作]` (King Abdullah University of Science and Technology), Tariq Alkhalifah `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 11827 | [OpenAlex ID](https://openalex.org/A5032021877)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于随机噪声的自监督预训练初始化方法，并在INR、DIP等单实例任务中验证其加速效果。

**💡 创新点**

创新点在于用无语义的随机噪声作为预训练目标，构造结构化权重初始化，显著减弱传统随机初始化的频谱偏差。

**🔧 技术方法**

采用自监督预训练、神经切线核（NTK）理论分析、SIREN/Deep Image Prior网络以及Adam优化器等技术。

**📊 数据集**

使用公开图像数据集（如512×512灰度图、CelebA等）进行单实例图像表示、去噪、填补与超分任务。

**📈 对比分析**

与Xavier/Kaiming等传统随机初始化相比，利用训练损失、PSNR曲线对比，预训练版在前100–200步即超越基线，显著加速收敛并提升早期重建质量。

**⚠️ 局限性**

局限性包括：仅在单实例任务验证，缺乏大规模监督学习场景验证；对噪声分布和预训练迭代次数敏感；可能导致过拟合或对局部细节过度敏感。

---

## 258. LogicSkills: A Structured Benchmark for Formal Reasoning in Large Language Models

**arXiv ID:** 2602.06533 | [PDF](https://arxiv.org/pdf/2602.06533v1)

**作者:** Brian Rabern `[一作]` (Niche), Barbara Plank `[通讯]` (Munich Center for Machine Learning)

**通讯引用:** 5212 | [OpenAlex ID](https://openalex.org/A5088832285)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 LogicSkills benchmark，用于分离形式逻辑的三项核心子技能（符号化、反模型构造、有效性判断），并在该基准上评估多种大型语言模型。

**💡 创新点**

创新点在于：① 将任务拆解为可独立测评的子技能；② 采用 SMT 求解器 Z3 对所有生成实例进行严格验证；③ 采用双语（英语与 Carrollian 虚构语）设计，剔除语义干扰；④ 通过细粒度分析揭示模型在不同子技能上的性能差异。

**🔧 技术方法**

技术包括：语法化的自然语言生成器（针对 FO^2 片段），SMT 求解器 Z3（用于验证等价性、可满足性、有效性及反模型），LLM 提取器（GPT‑4o）对输出进行清洗和格式归一化，LoRA 微调进行子技能迁移实验。

**📊 数据集**

使用自生成的 LogicSkills 数据集，包含 1500 题（600 符号化、600 有效性、300 反模型），每类题目在英语和 Carrollian 版本中均存在，且每条实例都有 SMT 验证的黄金答案。

**📈 对比分析**

方法：对 9 个主流 LLM（包括 Llama、Qwen、Claude、Gemini、GPT‑4o 等）在同一测试集上进行统一评测；对每个子技能计算准确率，并与模型规模和训练方式进行关联。结果显示：大多数模型在有效性任务上达到 80‑90% 以上，但在符号化仅约 60% 以内，反模型更低（10‑20%）。唯一例外是 Qwen3‑32B，在所有子技能上均表现优异（符号化 85%，反模型 89%，有效性 97%）。

**⚠️ 局限性**

限制：① 只覆盖 FO^2 无身份的子逻辑，无法推广到更完整的第一阶逻辑或模态逻辑；② 语言范围受限于人工生成的控制语料，缺乏自然语言多样性；③ 评测对输出格式和接口高度敏感，尽管采用提取器缓解，但仍可能影响得分；④ 仅进行行为评估，未深入探究模型内部的推理机制。

---

## 259. HyPER: Bridging Exploration and Exploitation for Scalable LLM Reasoning with Hypothesis Path Expansion and Reduction

**arXiv ID:** 2602.06527 | [PDF](https://arxiv.org/pdf/2602.06527v1)

**作者:** Shengxuan Qiu `[一作]` (Institute for Artificial Intelligence), Meng Li `[通讯]` (Institute for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在推理时段内动态调度多路径链式思维，通过在线控制器平衡探索与利用，提升大型语言模型的推理精度与计算效率。

**💡 创新点**

提出基于多路径池的实时扩展-收缩控制（HyPER），结合MoE路由多样性实现单Token级别的高效精细化；同时设计长度+置信度加权投票以弥合存在-选择缺口。

**🔧 技术方法**

在线置信度与多样性度量、两阶段MoE专家采样、加权投票聚合、基于阈值的路径裁剪与动态决策。

**📊 数据集**

AIME24/25、HMMT25、HLE、Math500、GSM8K、ARC-C/E等公开推理基准，使用Qwen3、Qwen3-Next、OLMoE、DeepSeek等MoE模型。

**📈 对比分析**

与Self‑Consistency、DeepConf、RoE等传统多路径或token级别扩展方法对比，HyPER在相同计算预算下平均提升8–10%准确率，且令token消耗降低25–40%。

**⚠️ 局限性**

方法仍无训练开销但对置信度阈值敏感；在极端长尾任务中仍可能出现路径收敛或投票失效；对非MoE稠密模型的适配需进一步验证。

---

## 260. An Integer Linear Programming Approach to Geometrically Consistent Partial-Partial Shape Matching

**arXiv ID:** 2602.06590 | [PDF](https://arxiv.org/pdf/2602.06590v1)

**作者:** Viktoria Ehm `[一作]` (Technical University of Munich), Daniel Cremers `[通讯]` (Technical University of Munich)

**通讯引用:** 48384 | [OpenAlex ID](https://openalex.org/A5087710605)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种专门针对部分-部分3D形状匹配的整数线性规划框架，利用几何一致性实现重叠区域预测与对应关系求解。

**💡 创新点**

创新点在于首次将整数线性规划与几何一致性结合，用于部分-部分匹配，并引入重叠概率作为先验，显著提升可扩展性与匹配质量。

**🔧 技术方法**

使用整数线性规划、产品图、表面环表示、几何一致性约束以及基于低分辨率解的粗细尺度递归求解技术。

**📊 数据集**

在CP2P24和PSMAL两个公开数据集上进行实验，数据覆盖10%–90%重叠的动物与人形网格。

**📈 对比分析**

与EchoMatch、DPFM、GC‑PPSM等方法对比，实验表明在重叠预测IoU、几何误差和Dirichlet能量上均优于或竞争性表现，且计算速度明显快于非线性整数规划方法。

**⚠️ 局限性**

局限性包括仍存在可取的解空间（如方向翻转）、可能得到多块离散重叠区域、以及在高分辨率时求解仍然计算量大，最坏情况时间复杂度为指数级。

---

## 261. Degradation of Feature Space in Continual Learning

**arXiv ID:** 2602.06586 | [PDF](https://arxiv.org/pdf/2602.06586v1)

**作者:** Chiara Lanza `[一作]` (CTTC), Paolo Dini `[通讯]` (CTTC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估持续学习（CL）中特征空间各向同性（Isotropy）与模型性能的关系，并尝试通过在对比学习损失中加入正则化项来提升各向同性。

**💡 创新点**

创新点在于：①提出了高维特征空间的 IsoEntropy 度量并与传统 IsoScore 进行对照；②在 CL 场景下首次用这些度量评估不同对比学习方法的几何结构；③证明在 CL 中强制各向同性往往会削弱模型的下游分类性能，揭示各向同性并非提高 CL 性能的有效诱导偏置。

**🔧 技术方法**

技术方法包括：使用 ResNet‑18 提取特征、SupCon、Co^2L、SupCP 与 NCI 等对比学习与蒸馏策略；计算 IsoEntropy、IsoScore 与 Mahalanobis inter/intra 归一比；引入可微的 IsoScore* 作为正则化项并调节 λ_iso 研究其对性能的影响。

**📊 数据集**

数据集：CIFAR‑10 与 CIFAR‑100，采用多种经验划分（50+50、40+30+30、20×5）进行实验。

**📈 对比分析**

比较方式：在同一 CL 经验划分下对比各方法的分类准确率、Mahalanobis inter/intra 比例以及各向同性指标。实验结果显示：在集中式训练下各向同性与准确率呈正相关；但在 CL 中各向同性与准确率不相关，甚至在加入正则化后准确率下降。

**⚠️ 局限性**

limitations: 1) 强制各向同性与 CL 的稳定-可塑性平衡冲突，导致性能下降；2) 仅在 CIFAR‑10/100 的简单图像任务上验证，缺乏对更大规模或更复杂任务的推广；3) 仅探讨 IsoEntropy、IsoScore，未探索更适合非平稳数据的几何正则化方法。

---

## 262. Inference-Time Rethinking with Latent Thought Vectors for Math Reasoning

**arXiv ID:** 2602.06584 | [PDF](https://arxiv.org/pdf/2602.06584v1)

**作者:** Deqian Kong `[一作]` (UCLA), Ying Nian Wu `[通讯]` (UCLA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在推理时通过迭代反思（Inference‑Time Rethinking）来自我纠错的生成式框架，拆分出声明性思考向量与程序化生成；

**💡 创新点**

创新点在于将推理分离为连续的思考向量和解码器，通过在思考空间上做梯度优化实现多轮自我纠错；

**🔧 技术方法**

使用Transformer编码器-解码器结构、可学习的先验映射、变分推理与Gibbs式迭代优化；

**📊 数据集**

训练集为GSM8K‑Aug（385K条等式式推理样例），测试集包括原始GSM8K、SVAMP和MultiArith；

**📈 对比分析**

与多种基线（CoT‑SFT、iCoT‑SI、Coconut、CoLaR、CODI、MARCoS等）对比，0.2B模型在单步和30轮迭代下分别达31.5%、51.5%、68.0%的准确率，超过参数量15×的对手；

**⚠️ 局限性**

局限在于过度依赖训练数据的质量，若数据噪声大，似然优先的自纠错可能失效，需要额外的正确性信号或外部验证。

---

## 263. Transformer-based Parameter Fitting of Models derived from Bloch-McConnell Equations for CEST MRI Analysis

**arXiv ID:** 2602.06574 | [PDF](https://arxiv.org/pdf/2602.06574v1)

**作者:** Christof Duhme `[一作]` (University of Münster), Xiaoyi Jiang `[通讯]` (University of Münster)

**通讯引用:** 12666 | [OpenAlex ID](https://openalex.org/A5022183918)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种基于Transformer的神经网络，用于对CEST MRI的物理模型（Lorentzian、解析Z谱、MTR_Rex）进行参数拟合，并在体外葡萄糖/乳酸混合模型中实现无监督训练，显著提升了拟合精度和速度。

**💡 创新点**

创新点在于将Transformer的长程注意力机制与物理模型约束相结合，利用自监督训练实现参数的直接预测，打破传统迭代求解器对初值和收敛速度的依赖；同时首次将两种Bloch–McConnell模型与Transformer结合。

**🔧 技术方法**

技术包括基于8层、8头注意力的Transformer编码器、3×3卷积解码器、tanh约束实现参数边界、均方误差损失的自监督训练、以及L-BFGS-B等经典优化器作对比。

**📊 数据集**

使用9.4 T Bruker Biospec系统采集的9个葡萄糖/乳酸混合物理实验体phantom数据，包含5、15、30 mM浓度、不同pH与温度下的529个像素谱。

**📈 对比分析**

与L-BFGS-B、Nelder–Mead、Powell等迭代求解器比较，Transformer网络在所有模型上都取得更高的R²（葡萄糖约0.96–0.99，乳酸约0.99–0.99）且推理时间比L‑BFGS‑B快约16–450倍。

**⚠️ 局限性**

局限性包括仅在体外phantom数据上验证，缺乏真实体内数据的Ground Truth；在高交换率、复杂体内环境下模型拟合的鲁棒性仍待验证。

---

## 264. LIBERO-X: Robustness Litmus for Vision-Language-Action Models

**arXiv ID:** 2602.06556 | [PDF](https://arxiv.org/pdf/2602.06556v1)

**作者:** Guodong Wang `[一作]` (Meituan), Xinmin Liu `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LIBERO-X benchmark，设计分层评估协议和多标签细粒度评价，构建基于人类远程操控的高多样性训练集。

**💡 创新点**

将多级分层评估与多标签评估相结合，且在训练阶段通过人类遥控收集多任务、多属性、多场景演示，显著扩展了数据与扰动维度，实现对 VLA 模型更全面、真实的鲁棒性检验。

**🔧 技术方法**

利用 VLM+控制头的视觉‑语言‑动作模型，采用行为克隆（SFT）训练，结合 MuJoCo 仿真环境，并对测试场景施加空间、对象属性及语义等五级叠加扰动。

**📊 数据集**

自建训练数据集 2,520 条演示、600 个任务、100 个场景；测试集采用 5 级层级扰动，覆盖从微小位置变化到语义重写的全维度变化。

**📈 对比分析**

对 OpenVLA‑OFT、X‑VLA、GR00T‑N1.5、π₀、π₀.₅ 等五种代表性 VLA 模型进行对比实验；从 Level‑1 到 Level‑5 成功率平均下降约 31%，π₀.₅ 在最难级别仍低于 20%。

**⚠️ 局限性**

模型在空间外推、场景结构变换、未见对象的语义对齐以及长序列任务上表现不足，导致鲁棒性和泛化能力受限。

---

## 265. SeeUPO: Sequence-Level Agentic-RL with Convergence Guarantees

**arXiv ID:** 2602.06554 | [PDF](https://arxiv.org/pdf/2602.06554v1)

**作者:** Tianyi Hu `[一作]` (Alibaba Group), Bolin Ding `[通讯]` (Alibaba Group)

**通讯引用:** 6791 | [OpenAlex ID](https://openalex.org/A5040297543)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究对大语言模型的多轮交互强化学习算法进行了系统的理论分析，并提出了一种新的Critic‑free、可收敛的算法SeeUPO；

**💡 创新点**

其创新点在于将多轮交互建模为逆序执行的多代理bandit问题，采用逆序逐步更新实现全局最优，并保持收敛保证；

**🔧 技术方法**

核心技术包括HAML框架、GRAE优势估计、PPO式裁剪策略以及逆序多代理协同更新；

**📊 数据集**

实验使用了AppWorld和BFCL v4这两个多轮工具调用基准；

**📈 对比分析**

与PPO、GRPO、GSPO等主流基线比较，SeeUPO在Qwen3‑14B模型下avg@4从约60%提升至约63%、pass@4提升至约81%，在Qwen2.5‑14B模型下提升约20%‑40%，且训练过程更加稳定；

**⚠️ 局限性**

局限性包括需在共享参数条件下满足异构策略假设、相对较高的计算开销，以及目前仅适用于序列级RL框架。

---

## 266. Dynamics-Aligned Shared Hypernetworks for Zero-Shot Actuator Inversion

**arXiv ID:** 2602.06550 | [PDF](https://arxiv.org/pdf/2602.06550v1)

**作者:** Jan Benad `[一作]` (Institute for Data Science Foundations), Manfred Eppe `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于共享超网络的动力学对齐框架（DMA*-SH），通过仅使用前向动力学预测的自监督学习来推断隐含上下文，并在同一超网络中为动态模型、策略和价值网络生成乘法型适配器，实现零样本自适应。

**💡 创新点**

创新点包括：① 用单一超网络在动态对齐的基础上对多模态上下文进行共享乘法调制；② 通过输入/输出归一化和随机掩码实现方向性聚焦与噪声压缩；③ 在理论上给出了超网络乘法调制的表达优势与方差分解；④ 设计了专门针对断裂上下文的 Actuator Inversion Benchmark（AIB）。

**🔧 技术方法**

核心技术：共享超网络（Hypernetwork）生成适配器权重；动态对齐（Dynamics‑Aligned）上下文编码；输入/输出归一化（AvgL1Norm、SimNorm）；随机输入掩码；多任务奖励基于 SAC 的强化学习框架；理论分析（表达式分离、方差上界）。

**📊 数据集**

使用的“数据集”是自研的 AIB，包含多个离散/连续双维上下文的 RL 环境（如 DI、Cartpole、Cheetah、Reacher、Walker 等），每个环境都有二元“扭曲”上下文（actuator inversion）以及连续物理参数。

**📈 对比分析**

与基线（Concat、DA、Domain Randomization、Amago、DMA、DMA‑Pearl）比较，DMA*-SH 在所有 AIB 环境的零样本泛化中均取得显著提升：在非重叠（non‑overlap）设置下平均提升约 18%，在整体平均上超过域随机化 111.8% 并优于 Concat 16.1%，在多任务评估中表现最稳健。

**⚠️ 局限性**

局限性：① 依赖动力学模型的准确性，模型误差会传播到上下文表示；② 主要适用于动力学变换，对奖励变换或非可分解策略变更的适应性不足；③ 超网络容量有限，过小限制表达力，过大易导致过拟合；④ 目前未考虑快速变化或非平稳的上下文。

---

## 267. Malicious Agent Skills in the Wild: A Large-Scale Security Empirical Study

**arXiv ID:** 2602.06547 | [PDF](https://arxiv.org/pdf/2602.06547v1)

**作者:** Yi Liu `[一作]` (Quantstamp), Leo Yu Zhang `[通讯]` (Griffith University)

**通讯引用:** 4630 | [OpenAlex ID](https://openalex.org/A5015011245)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文构建了首个大规模的恶意代理技能数据集，并对其威胁景观、攻击策略与躲避手段进行了系统测评。

**💡 创新点**

创新点在于将动态行为验证与静态特征相结合，精确标注了157个恶意技能及632个漏洞，揭示了两大攻击原型（数据窃取与代理劫持）以及平台原生攻击新类。

**🔧 技术方法**

采用了正则匹配、LLM分析器、Docker沙盒执行、网络与系统调用监控、手工漏洞标签等技术组合进行多维检测与验证。

**📊 数据集**

使用了来自两大社区注册表的98,380个技能作为样本，筛选并验证了4,287个候选恶意技能，最终确认157个恶意实例。

**📈 对比分析**

与传统仅基于代码的检测方法相比，结合动态验证的管道精度达99.6%，而在相同任务下能识别出此前难以发现的文档层面攻击，缺陷是对长时间触发或沙盒检测规避的技能识别仍受限。

**⚠️ 局限性**

主要局限包括仅覆盖公开注册表的技能、60秒执行窗口可能漏检被延时触发的恶意行为、对沙盒环境可检测的技能偏见以及数据集在未来平台演化时的适应性不足。

---

## 268. AdaptOVCD: Training-Free Open-Vocabulary Remote Sensing Change Detection via Adaptive Information Fusion

**arXiv ID:** 2602.06529 | [PDF](https://arxiv.org/pdf/2602.06529v1)

**作者:** Mingyu Dou `[一作]` (Key Laboratory of Spectral Imaging Technology CAS, Xi'an Institute of Optics and Precision Mechanics, Chinese Academy of Sciences), Zhe Sun `[通讯]` (School of Artificial Intelligence, Optics and Electronics, Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了AdaptOVCD，一种无训练的开词汇遥感变化检测框架。

**💡 创新点**

创新点在于双维多层信息融合架构，结合数据层、特征层、决策层的自适应设计，利用SAM‑HQ、DINOv3、DGTRS‑CLIP三大视觉基础模型协同工作，解决开词汇检测的误差累积和域感知缺口。

**🔧 技术方法**

采用Adaptive Radiometric Alignment、Adaptive Change Thresholding、Adaptive Confidence Filtering等自适应模块，融合SAM‑HQ分割、DINOv3特征比较、DGTRS‑CLIP语义识别。

**📊 数据集**

在LEVIR‑CD、WHU‑CD、DSIFN、SECOND四个公开遥感变化检测数据集上进行评测。

**📈 对比分析**

与传统无监督和现有训练‑free方法对比，AdaptOVCD在四大建筑变化场景的F1得分分别为68.00%、76.53%、59.47%、63.81%，平均达到84.9%的全监督上限，并在六类开词汇变化场景中表现显著优于DynamicEarth。

**⚠️ 局限性**

主要局限在于对高分辨率大规模图像的计算量仍较大，且对极端光照或季节性变化的鲁棒性待进一步提升。

---

## 269. Sequential Auditing for f-Differential Privacy

**arXiv ID:** 2602.06518 | [PDF](https://arxiv.org/pdf/2602.06518v1)

**作者:** Tim Kutta `[一作]` (Aarhus University), Vassilis Zikas `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1806 | [OpenAlex ID](https://openalex.org/A5014862413)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于 f‑DP 的顺序审计方法，能够自适应确定所需样本数并保持统计显著性；

**💡 创新点**

创新点在于①首次针对 f‑DP 开发顺序审计器，②不需预先指定样本量，③使用 Brownian 动态界限实现 anytime‑valid 推断，④改进的阈值选择策略提升检测功效；

**🔧 技术方法**

使用 Neyman‑Pearson 似然比分类器、核密度估计或高斯参数化、假设检验、Hoeffding 及 Brownian 边界、序贯统计量更新；

**📊 数据集**

实验数据集包括：Gaussian、Laplace 加噪声机制、DP‑SGD（MNIST/CIFAR‑10）以及单跑 DP‑SGD 的可插入 canary 训练；

**📈 对比分析**

与固定批量审计及基于 (ε,δ)-DP 的序贯方法对比，样本量平均降低 75‑90%，拒绝率在设定显著性水平下保持；

**⚠️ 局限性**

局限性包括：黑盒场景需估计密度，KDE 可能受高维影响；对极低效能的机制可能仍需较多样本；方法对超参数（如 burn‑in 大小）敏感。

---

## 270. Machine Learning Practitioners' Views on Data Quality in Light of EU Regulatory Requirements: A European Online Survey

**arXiv ID:** 2602.06594 | [PDF](https://arxiv.org/pdf/2602.06594v1)

**作者:** Yichun Wang `[一作]` (University of Amsterdam), Hazar Harmouch `[通讯]` (University of Amsterdam)

**通讯引用:** 426 | [OpenAlex ID](https://openalex.org/A5008534205)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构建一个将传统数据质量维度与欧盟 GDPR 与 AI 法案要求对齐的框架，并对 185 名欧盟数据从业者进行在线调查，揭示了实际实践与监管要求之间的差距。

**💡 创新点**

创新点在于将技术层面的多维数据质量模型与监管文本进行系统映射，形成可操作的合规数据质量词汇表，并通过实证研究验证其在行业中的适用性和缺口。

**🔧 技术方法**

研究采用设计科学方法论、结构化问卷调查、统计分析（置信区间、相关性检验）以及对现有工具使用情况的定量评估。

**📊 数据集**

数据集为自行收集的调查数据，涵盖 24 个欧盟成员国、各行业及岗位的 185 位数据从业者的自报信息。

**📈 对比分析**

论文没有传统的算法性能对比，而是通过描述性统计和相关分析展示不同数据质量实践与监管合规之间的关联，表明当前工具与实践仍存在显著缺口。

**⚠️ 局限性**

局限性包括样本采样采用滚雪球法导致对大企业的过度代表，缺乏对小型组织和行业边缘场景的覆盖；仅聚焦与监管对齐的维度，未深入探讨其他法律约束；并未提供工具实现细节或实验验证。

---

## 271. AgentStepper: Interactive Debugging of Software Development Agents

**arXiv ID:** 2602.06593 | [PDF](https://arxiv.org/pdf/2602.06593v1)

**作者:** Robert Hutter `[一作]` (University of Stuttgart), Michael Pradel `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了名为AgentStepper的交互式调试器，专为LLM驱动的软件开发代理设计；

**💡 创新点**

创新点在于将传统调试抽象提升到代理层面，支持断点、单步执行、实时编辑LLM提示与工具调用，并将中间代码变更以提交历史形式呈现；

**🔧 技术方法**

采用了结构化对话可视化、LLM总结、Git仓库跟踪、WebSocket双向通信、Vue.js前端及API插桩方式；

**📊 数据集**

评估使用ExecutionAgent、SWE‑Agent和RepairAgent三款代理的执行轨迹，用户研究中共12名参与者；

**📈 对比分析**

集成成本低（仅需39–42行改动），用户研究显示理解率从64%提升至67%，错误定位率从17%提升至60%，工作负荷（NASA‑TLX）显著降低（如挫败感5.4→2.4）；

**⚠️ 局限性**

局限包括样本量小、仅评估三款代理、集成指标未覆盖理解代理架构的实际工作量、可能受限于原始代理实现方式和日志结构等因素。

---

## 272. Perturbing the Phase: Analyzing Adversarial Robustness of Complex-Valued Neural Networks

**arXiv ID:** 2602.06577 | [PDF](https://arxiv.org/pdf/2602.06577v1)

**作者:** Florian Eilers `[一作]` (University of Münster), Xiaoyi Jiang `[通讯]` (University of Münster)

**通讯引用:** 12666 | [OpenAlex ID](https://openalex.org/A5022183918)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了复数值神经网络（CVNN）在对抗攻击下的鲁棒性，提出了专门针对相位信息的Phase Attacks并与传统攻击和Magnitude Attacks进行对比实验

**💡 创新点**

首次引入Phase Attacks方法，展示即使仅改变相位信息，CVNN和RVNN同样易受攻击；并证明多步Phase Attacks甚至比不受限攻击更有效

**🔧 技术方法**

利用Wirtinger微分推导复数域的梯度攻击；实现FGSM、FFGSM、IFGSM、MIFGSM以及其相位版本（PFGSM、PFFGSM、PIFGSM、PMIFGSM）

**📊 数据集**

在两组复数图像数据集上实验：PolSAR（S1SLC_CVDL）和FastMRI Prostate（预处理后），分别训练ResNet和ConvNeXt两种网络

**📈 对比分析**

通过对比RVNN与CVNN在不同攻击强度（ε）下的相对性能，发现大多数情况下CVNN更具鲁棒性；但在ConvNeXt+Prostate无对抗训练时RVNN更稳健；Phase攻击对模型性能影响最大

**⚠️ 局限性**

研究局限在于仅测试两类数据集和两种网络架构；相位攻击虽有效但理论上更弱，未探究更广泛的鲁棒性指标与更深层网络；缺少对抗训练针对相位攻击的专门方法

---

## 273. Learning to Allocate Resources with Censored Feedback

**arXiv ID:** 2602.06565 | [PDF](https://arxiv.org/pdf/2602.06565v1)

**作者:** Giovanni Montanari `[一作]` (Inria), Vianney Perchet `[通讯]` (Criteo AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究一种在线资源分配问题：在每轮中将预算 B 分配给 K 个 arm，只有当 arm 被激活且分配的预算超过一个随机阈值时才产生奖励；目标是通过在线学习未知的激活概率和阈值分布来最大化累积奖励。

**💡 创新点**

创新点包括：①提出一种将奖励收集与参数估计解耦的 UCB 族算法（RA‑UCB），通过构造 surrogate 函数实现强凸性，显著降低估计误差；②在已知预算下给出 Ω(T^{1/3}) 的信息论下界，并证明 RA‑UCB 在满足一定分布与凸性假设时实现 √T（甚至 poly‑log T） 的上界；③在预算未知且可在一轮内切换的情形下，通过水位填充动态实现同样的 regret 上界。

**🔧 技术方法**

使用技术主要有：置信区间与 UCB 估计、阈值分布的条件均值估计、参数分离估计、surrogate 目标的构造与强凸性证明、信息论下界构造、阈值分布的 Lipschitz 与单调性、KKT 条件实现的水位填充方法。

**📊 数据集**

实验数据集包括：① EdNet‑KT3 学习平台的多选题答题日志（20 题，阈值服从 Weibull 分布）；② Criteo 广告投放日志（用户到达次数，按离散 Weibull 拟合）。

**📈 对比分析**

与全知参数的最优分配（oracle）以及随机/Explore‑Then‑Commit 基线对比。实验表明 RA‑UCB 在已知预算下的累计收益接近 oracle，远优于基线；在未知预算下同样保持近似最优。总体收益提升显著，证明了理论上给出的 √T（或 poly‑log T） 上界在实际环境中可实现。

**⚠️ 局限性**

局限性：① 需要阈值分布满足可逆性和强凸性（如指数、Weibull/ Gamma shape ≤1）；② 仅在可在一轮内切换预算的设定下适用，无法直接处理不允许切换的未知预算；③ 对连续可分配增量的假设在实际离散环境中需要近似实现；④ 随着 arm 数目增大，surrogate 目标求解与参数估计的计算成本会显著提升。

---

## 274. Reinforcement Learning-Based Dynamic Management of Structured Parallel Farm Skeletons on Serverless Platforms

**arXiv ID:** 2602.06555 | [PDF](https://arxiv.org/pdf/2602.06555v1)

**作者:** Lanpei Li `[一作]` (Institute of Information Science and Technologies National Research Council of Italy), Vincenzo Lomonaco `[通讯]` (LUISS University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在 OpenFaaS 上实现了一个结构化的 Farm 并行骨架，并为其提供了外部的自适应扩缩控制环境，将扩缩问题建模为马尔可夫决策过程（MDP），采用强化学习（SARSA、Double‑DQN）以及两种基线规则进行比较。

**💡 创新点**

创新点：
- 将传统 Farm 模式迁移至 Serverless FaaS（OpenFaaS）并解决其冷启动与负载分配难题；
- 设计了一个 QoS‑感知的扩缩 MDP 并实现了 Gymnasium 接口，供 RL 与规则控制共用；
- 通过 Redis 队列实现异步调用，支持可扩展的 worker 组；
- 在单一任务流上将 RL 策略与基线做细粒度对比，证明 RL 能在保证 QoS 的同时降低资源占用与扩缩频率。

**🔧 技术方法**

技术手段：
- OpenFaaS、Kubernetes、Docker 容器化
- Redis 队列（生产者/消费者）
- Gymnasium 环境接口
- 强化学习：SARSA(λ) 及 Double‑DQN
- Python、Docker Compose、Docker‑Compose‑Swarm、Redis‑CLI
- 统计与可视化：Matplotlib、Pandas

**📊 数据集**

使用的工作负载：合成的图像处理流水线（四阶段：缩略图、压缩、元数据提取、格式转换），采用不同尺寸图像的时间模型并产生随机到达速率；没有真实公开数据集，全部为仿真/睡眠延迟生成。

**📈 对比分析**

比较方法：在 10 条 240 s 试验（不同随机种子）下，记录 QoS 满足率、平均/最大 worker 数、扩缩次数与无操作次数；
- RL（SARSA）: QoS ≈ 99.2 %，平均 worker ≈ 14.8，扩缩 ≈ 24；
- RL（DQN）: QoS ≈ 94.9 %，平均 worker ≈ 14.4，扩缩 ≈ 3.6；
- 规则基线（ReactiveAvg）：QoS ≈ 50.9 %，平均 worker ≈ 10.8；
- 规则基线（ReactiveMax）：QoS ≈ 65.6 %，平均 worker ≈ 13.9。
结果表明 RL 明显优于基线，在 QoS、资源利用率和扩缩稳定性上都有提升。

**⚠️ 局限性**

局限性：
- 仅在单租户、单 Farm 示例上验证；未评估多租户或多骨架交互；
- 工作负载为仿真/睡眠模型，未涵盖真实网络/存储延迟与资源争用；
- RL 训练需手工离散化状态空间（SARSA）或使用深度网络（DQN），训练成本较高；
- OpenFaaS 的冷启动和扩缩延迟仍对高峰负载响应产生影响。

---

## 275. Fine-Grained Model Merging via Modular Expert Recombination

**arXiv ID:** 2602.06552 | [PDF](https://arxiv.org/pdf/2602.06552v1)

**作者:** Haiyun Qiu `[一作]` (Hong Kong Polytechnic University), Kay Chen Tan `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 31736 | [OpenAlex ID](https://openalex.org/A5025285243)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 MERGE 的细粒度模型合并方法，能够在不需要重新训练的情况下，将多个任务专用模型按组件级别合并，并在推理时根据输入动态重组可重用的专家模块。

**💡 创新点**

创新点包括：①将模型拆解为功能组件，形成可重用的专家库；②采用双目标（跨任务性能与存储效率）进化优化，并用代理模型加速搜索；③在推理阶段通过轻量路由网络实现输入感知的模块重组，兼顾适应性与可重用性。

**🔧 技术方法**

使用技术包括：组件级拆分与分组、随机森林代理的 NSGA‑II 生物进化搜索、量化压缩、轻量路由网络、基于任务向量的合并函数（如 Task Arithmetic、Ties‑Merging 等）。

**📊 数据集**

实验数据集覆盖视觉任务（SUN397、Cars、RESISC45、EuroSAT、SVHN、GTSRB、MNIST、DTD）、语言任务（CoLA、SST‑2、MRPC、STS‑B、QQP、MNLI、QNLI、RTE）以及 11 个 PEFT 任务（RTE、CB、Winogrande、WiC、WSC、COPA、H‑SWAG、Story Cloze、ANLI‑R1~R3），使用 ViT‑B/32、ViT‑L/14、RoBERTa‑base、GPT‑2、T0‑3B 等预训练模型。

**📈 对比分析**

在多种模型规模、任务类型和微调策略下，与传统 MTL、静态合并（Weight Averaging、Fisher、RegMean 等）和动态合并（Twin‑Merging、EMR‑Merging 等）基线进行比较。MERGE 的 Pareto 前沿显示：在仅占原模型约 30% 存储时已接近或超过单任务模型的性能；存储最小方案在性能上至少优于所有基线；性能最大方案几乎匹配单任务模型，但存储量远低于传统方法。

**⚠️ 局限性**

局限性包括：离线搜索仍需昂贵的评估预算；搜索空间随任务数和模型尺寸指数增长，扩展性未充分验证；需要根据具体任务手动挑选最优合并配置；对极大模型或极多任务场景的推理效率与存储约束仍有待进一步优化。

---

## 276. NECromancer: Breathing Life into Skeletons via BVH Animation

**arXiv ID:** 2602.06548 | [PDF](https://arxiv.org/pdf/2602.06548v1)

**作者:** Mingxi Xu `[一作]` (Huawei Central Media Technology Institute), Mingyuan Zhang `[通讯]` (Huawei Central Media Technology Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套面向任意 BVH 骨架的通用运动分词器 NECromancer，可用于运动重建、检索、跨物种迁移、生成等多任务。

**💡 创新点**

创新点：①Ontology-aware Skeletal Graph Encoder（OwO）利用图注意力学习骨架结构、语义和几何先验；②Topology-Agnostic Tokenizer（TAT）通过学习的虚拟关节实现拓扑无关的离散化；③构建统一的 BVH‑centric benchmark（Unified BVH Universe），覆盖人类、四足动物、幻想生物等多种骨架。

**🔧 技术方法**

技术手段：图注意力网络、残差向量量化（RVQ）、Transformer 时空块、旋转6D 表示、CLIP 对比学习、数据增强（重置基准姿态）等。

**📊 数据集**

数据集：47,807 条 BVH 动画，来源 HumanML3D、Objaverse‑XL、Truebones Zoo，经过统一预处理后形成的大规模 BVH benchmark。

**📈 对比分析**

与人类中心化文本到运动模型（T2M‑GPT、Motion Streamer、TM2T）以及基于零填充的 RVQ‑VAE 对比，NEC 在 MPJPE、GeoDist、R‑Precision 等指标上均显著优于基线，尤其在非人类骨架上表现出色。

**⚠️ 局限性**

局限性：仍难以精确重现细粒度姿态细节；依赖 BVH 格式，无法直接应用于非 BVH 数据；训练需要大量骨架先验，对跨模态控制（如音频、语音）尚未深入研究。

---

## 277. AlertBERT: A noise-robust alert grouping framework for simultaneous cyber attacks

**arXiv ID:** 2602.06534 | [PDF](https://arxiv.org/pdf/2602.06534v1)

**作者:** Lukas Karner `[一作]` (AIT Austrian Institute of Technology), Florian Skopik `[通讯]` (AIT Austrian Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并实现了 AlertBERT，一个基于自监督学习的框架，用于在噪声环境下对来自孤立或并发攻击的安全警报进行分组。

**💡 创新点**

创新点在于结合掩码语言模型与基于密度的聚类来替代传统基于时间的分组，并提出了新的数据增强方法来模拟并发攻击与噪声。

**🔧 技术方法**

采用了掩码语言模型（BERT）、密度聚类（如 DBSCAN）以及自监督学习策略。

**📊 数据集**

使用了通过新型数据增强方法生成的合成安全日志数据集，并在此基础上进行评估。

**📈 对比分析**

与传统基于时间的分组技术进行对比，AlertBERT 在识别正确警报组的准确率上持续优于对手，表现更佳。

**⚠️ 局限性**

局限性包括对不同网络环境的适应性需进一步验证、计算资源需求较高，以及对极高噪声率下的聚类性能尚待考察。

---

## 278. Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevant Assessment for IR Benchmarks

**arXiv ID:** 2602.06526 | [PDF](https://arxiv.org/pdf/2602.06526v1)

**作者:** Minjeong Ban `[一作]` (Korea Advanced Institute of Science and Technology), Hwanjun Song `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1987 | [OpenAlex ID](https://openalex.org/A5033909285)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多轮辩论式的多智能体相关性评估框架（DREAM），并利用其改进IR benchmark，生成了 BRIDGE 数据集；同时分析了缺漏块对检索与 RAG 评估的偏差。

**💡 创新点**

创新点在于：
• 采用对立立场的双模型辩论，利用多轮相互批判与共识决策实现自动标注；
• 仅在持续分歧时才升级至人工，避免过度依赖模型置信度；
• 将辩论历史作为辅助材料提升人工标注一致性与准确性；
• 通过填补缺漏块显著缓解评估偏差并提升检索-生成对齐度。

**🔧 技术方法**

使用技术包括：大语言模型（Llama3.3‑70B‑Instruct）多模型辩论、同意/分歧决策、人工升级机制、RAGAlign 对齐度指标、RAG 生成评估、实验中对 25 种检索器和 2 种 RAG 系统的评估。

**📊 数据集**

数据集：
• 原始 MS MARCO、NQ（BEIR）和 Lifestyle、Recreation、Science、Technology、Writing（RobustQA）七个子集；
• 通过 DREAM 标注后得到 BRIDGE benchmark（36,800 gold 块，其中 29,824 为之前缺漏的相关块）；
• 25 个检索器的检索候选池和 25 个检索器+RAG 的评估数据。

**📈 对比分析**

与单一模型、置信度‑基准升级、MTurk 多人投票等方法对比：
• DREAM 在未升级样本上的 balanced accuracy 达 95.2%，升降率仅 3.5%，相较于 93.8% 的单模型和 50% 的升降率置信度方法性能更好；
• 在 BRIDGE benchmark 上 25 个检索器的 Hit@10 显著提升，RAGAlign（检索‑生成对齐度）提升 0.14；
• 人工成本和延迟大幅降低（约 200 倍成本节省、3.5–7 倍速度提升）。

**⚠️ 局限性**

局限性：
• 对极其复杂或噪声大样本仍可能产生错误或不一致；
• 双模型辩论需要额外算力，成本不低；
• 依赖大语言模型的推理质量，若模型更新或停用需重新验证；
• RAGAlign 采用二元评估，可能无法捕捉生成细节差异；
• 在极少样本或低资源场景下的鲁棒性尚未充分验证。

---

## 279. Progress Constraints for Reinforcement Learning in Behavior Trees

**arXiv ID:** 2602.06525 | [PDF](https://arxiv.org/pdf/2602.06525v1)

**作者:** Finn Rietz `[一作]` (Örebro University), Petter Ögren `[通讯]` (Royal Institute of Technology)

**通讯引用:** 5143 | [OpenAlex ID](https://openalex.org/A5070754732)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出一种在行为树(BT)中使用强化学习(RL)的训练框架——进度约束行为树强化学习(CBTRL)，通过可行性估计为每个子任务控制器生成动作掩码，保证控制器在执行过程中不破坏BT的进度，从而实现安全、高效的任务分解与学习。

**💡 创新点**

创新点在于：①利用BT收敛分析推导进度约束，②用贝尔曼可行性方程训练可行性估计器生成动作空间掩码，③将约束融入RL训练全过程，消除子任务间的冲突和重复执行；从而显著降低样本复杂度并避免约束违规。

**🔧 技术方法**

采用的技术包括：行为树结构、强化学习（PPO/DQN）与离线可行性估计（Bellman方程）、动作掩码（action masking）、进度约束设计、任务分解与子任务MDP构造。

**📊 数据集**

实验数据集为两种仿真环境：①二维目标导航环境（含危险区和斜坡），②仓储机器人环境（移动抓取机器人与动态叉车）。训练过程中使用RL采样获得约5M-100M条转移，5M用于可行性估计器。

**📈 对比分析**

与标准RL、BTRL（无约束）和BT-Penalty（仅奖励惩罚）对比，CBTRL在两种环境中都实现了更高的成功率（仓储场景93%对比78-83%），零约束违规，收敛更快（样本量降低约30%），并且在无约束方法的后期约束应用中表现更差，验证了约束必须在训练阶段引入。

**⚠️ 局限性**

局限性包括：①需要额外的数据（约5%）来训练可行性估计器，若数据不足会影响约束学习；②对BT结构和子任务划分的依赖较强，复杂BT下约束设计与估计可能更难；③目前仅在仿真环境验证，真实机器人应用需进一步测试。

---

## 280. MicroBi-ConvLSTM: An Ultra-Lightweight Efficient Model for Human Activity Recognition on Resource Constrained Devices

**arXiv ID:** 2602.06523 | [PDF](https://arxiv.org/pdf/2602.06523v1)

**作者:** Mridankan Mandal `[一作]` `[通讯]` (Indian Institute of Information Technology), Mridankan Mandal (Indian Institute of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种全卷积+单层双向LSTM的超轻量HAR模型uBi‑ConvLSTM，平均参数仅约11.4K，能在可穿戴设备上部署。

**💡 创新点**

创新点在于四倍时间池化压缩序列、标准卷积保留跨通道信息、严格O(N)复杂度与单层双向LSTM的组合，使参数比TinierHAR低2.9倍同时保持高准确率。

**🔧 技术方法**

技术手段包括标准卷积+池化、单层双向LSTM、后训练INT8量化、以及BatchNorm与Dropout正则化。

**📊 数据集**

使用8个公开HAR基准（UCI‑HAR、MotionSense、WISDM、PAMAP2、Opportunity、UniMiB‑SHAR、SKODA、Daphnet）进行评估。

**📈 对比分析**

与DeepConvLSTM、TinyHAR、TinierHAR比较，平均宏F1约83.7，参数11.4K，MACs约485K，优于同类轻量模型，且量化后仅降幅0.21%。

**⚠️ 局限性**

局限性包括对严重类别不平衡数据（如PAMAP2）性能下降，双向LSTM在周期性动作上收益有限，且在低采样率或低通道数数据上方差较大。

---

## 281. Dependable Artificial Intelligence with Reliability and Security (DAIReS): A Unified Syndrome Decoding Approach for Hallucination and Backdoor Trigger Detection

**arXiv ID:** 2602.06532 | [PDF](https://arxiv.org/pdf/2602.06532v1)

**作者:** Hema Karnam Surendrababu `[一作]` (National Institute of Advanced Studies), Nithin Nagaraj `[通讯]` (National Institute of Advanced Studies)

**通讯引用:** 1115 | [OpenAlex ID](https://openalex.org/A5077181669)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于综合解码（syndrome decoding）的统一方法，用于同时检测机器学习模型和大型语言模型中的后门攻击和自我参考式元解释任务产生的幻觉；

**💡 创新点**

创新点在于将线性块码的综合解码技术迁移至句子嵌入空间，构造生成矩阵和检验矩阵，形成统一的安全与可靠性检测框架；

**🔧 技术方法**

使用了句子嵌入（SBERT）、PCA、线性代数的生成与检验矩阵构造，以及仿射增幅的编码技术；

**📊 数据集**

实验覆盖六个数据集：NLP 领域的 SST‑2、Jigsaw Toxicity、Trawling for Trolling，表格 领域的 Forest Cover 与 U.S. Adult Income；另外在多款 LLM（Claude Sonnet 4.5、ChatGPT 5.2、Gemini 3、Microsoft Copilot、Perplexity AI）上测试自我参考式提示；

**📈 对比分析**

与传统后门检测方法（基于统计、对抗训练等）和幻觉度量方法（基于语义一致性、事实性得分）对比，综合解码在 5%–15% 后门比例下能有效区分受污染与正常样本；在自我参考式幻觉检测中，综合解码可将幻觉文本与正常文本在综合解码模量上分离，表现出较高的区分度；

**⚠️ 局限性**

局限性包括：对低幅度或复杂触发器的检测可能不稳定；对高维多模态模型的推广尚未验证；需要足量的非污染样本来构造模板，可能在资源受限场景下受限；

---

## 282. Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning

**arXiv ID:** 2602.06600 | [PDF](https://arxiv.org/pdf/2602.06600v1)

**作者:** Zhuoyuan Hao `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 113410 | [OpenAlex ID](https://openalex.org/A5100336796)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析并利用大型推理模型在思考链首部自然重复用户问题的现象（Echo of Prompt，EOP）来提升推理质量。

**💡 创新点**

首次把EOP建模为拒绝采样的概率事件，引入 Echo Likelihood Gap 指标，并证明其与正确率相关；提出 ED‑SFT 和 EP 两种利用 EOP 的训练/推理策略。

**🔧 技术方法**

使用概率框架、注意力分析、监督微调（ED‑SFT）以及推理时重现提示（EP）等技术。

**📊 数据集**

在 GSM8K、MathQA、Hendrycks‑MATH、AIME24 和 MATH‑500 等数学推理基准上进行实验。

**📈 对比分析**

与基线（无 EOP、TTTS 等）相比，ED‑SFT 在 GSM8K 上提升约 10 点准确率，EP 在 AIME24 和 MATH‑500 上实现显著的性能提升。

**⚠️ 局限性**

对模型需要先检测或生成 EOP 的依赖、对非数学推理任务的适用性有限，以及仍无法解释某些错误案例。

---

## 283. Sample-Efficient Policy Space Response Oracles with Joint Experience Best Response

**arXiv ID:** 2602.06599 | [PDF](https://arxiv.org/pdf/2602.06599v1)

**作者:** Ariyan Bighashdel `[一作]` (Utrecht University), Frans A. Oliehoek `[通讯]` (Delft University of Technology)

**通讯引用:** 2901 | [OpenAlex ID](https://openalex.org/A5009493909)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Joint Experience Best Response（JBR）框架，通过在 PSRO 中共享经验来一次性为所有代理训练最佳回应，从而显著提升样本效率。

**💡 创新点**

创新点在于将多代理的最佳回应计算转化为离线 RL 问题，提供保守（SPI）、探索增强（随机和针对性探索）以及混合（Hybrid）三种变体，尤其针对性探索通过在数据收集时使用当前最佳回应来预测对手的适应，恢复 PSRO 的收敛精度。

**🔧 技术方法**

技术包括 PSRO 的元策略求解器（投影复制动态）、离线 RL（基于价值迭代的模型学习）、Safe Policy Improvement、探索扰动 δ-perturbation 以及混合更新策略。

**📊 数据集**

实验数据集涵盖离散扑克游戏（Kuhn Poker、Leduc Poker）和连续多智能体粒子环境（Simple Tag、Simple Adversary、Simple Push）。

**📈 对比分析**

与标准 PSRO、独立学习（IL）和集中式训练去中心化（CTDE）对比，JBR-PSRO-δT 在保持接近 PSRO 的 NashConv 的同时，将最佳回应所需的环境交互次数缩减至约一半；Hybrid 方案在进一步提高精度的同时仅略增样本量。

**⚠️ 局限性**

局限性包括：仍需大量迭代和样本；元策略更新与收益矩阵增长导致计算开销；针对连续深度 RL 的探索调度和 δ 选取仍需手动调优，未来需开发自适应调度和可扩展的元策略求解器。

---

## 284. Energy-Aware Metaheuristics

**arXiv ID:** 2602.06595 | [PDF](https://arxiv.org/pdf/2602.06595v1)

**作者:** Tomohiro Harada `[一作]` (Saitama University), Gabriel Luque `[通讯]` (University of Malaga)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了基于能耗的期望改进/焦耳（EI/J）框架，用于在固定能量预算下自适应选择 GA、PSO、ILS 等元启发式的算子；

**💡 创新点**

创新点在于将能耗与期望数值改进结合，给出鲁棒的 EI/J 度量，并通过 Thompson 采样实现能量感知的算子调度，适用于多种元启发式；

**🔧 技术方法**

使用了 RAPL 进行硬件能耗测量、指数加权移动平均、Thompson 采样、能量感知算子调度等技术；

**📊 数据集**

实验数据集包括三种组合优化问题：0‑1 背包（Pisinger 100 件实例）、NK‑景观（n=100,k=6）和误差纠正码（n=12, M=24）；

**📈 对比分析**

与单一算子基线在固定能量预算下进行比较，评估最佳 fitness 与能耗；结果显示能量感知变体保持或提升解质量，同时能耗显著降低，且与最优单算子表现相当；

**⚠️ 局限性**

局限在于仅验证算子数量有限的情况，未扩展到更大算子组合或并行环境，能耗测量依赖特定硬件计数器，且在某些问题上仍需进一步优化。

---

## 285. Exploring Sparsity and Smoothness of Arbitrary $\ell_p$ Norms in Adversarial Attacks

**arXiv ID:** 2602.06578 | [PDF](https://arxiv.org/pdf/2602.06578v1)

**作者:** Christof Duhme `[一作]` (University of Münster), Xiaoyi Jiang `[通讯]` (University of Münster)

**通讯引用:** 12666 | [OpenAlex ID](https://openalex.org/A5022183918)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了在 ℓ_p 约束（p∈[1,2]）下对抗攻击的稀疏性与光滑性，并给出了不同模型与数据集的最优 p 值；

**💡 创新点**

创新点在于提出通用光滑性度量框架（基于平滑算子与泰勒展开）并与两种稀疏性度量（Gini、Hoyer）结合，首次系统比较 p 对稀疏性与光滑性的影响，指出常用 p=1、p=2 子优，推荐 1.3–1.5 范围；

**🔧 技术方法**

技术手段包括：
- 对抗攻击生成：ℓ_1-apgd、afw（p>1）、apgd 等；
- 稀疏性度量：Gini 指数、Hoyer 指数；
- 光滑性度量：基于 Gaussian/低通滤波的平滑算子度量以及基于一阶泰勒展开的度量；
- 训练/攻击实验：对模型进行正常与对抗训练，使用性能基准 ε_p；
- 评估指标：稀疏性曲线、光滑性曲线、最优 p 分布等；

**📊 数据集**

数据集：CIFAR-10、CIFAR-100、Flowers102、GTSRB；模型：ResNet‑18/50/101、VGG‑16/19、ViT‑B/16/32、Swin‑T v2、Swin‑S v2；

**📈 对比分析**

比较方法：对同一模型/数据集下，vary p 取值（1、1.01、1.1…2.0），比较稀疏性与光滑性两类度量的曲线，并对比 ℓ_1-apgd 与 1.01-afw 的效果；性能上发现 p≈1.3–1.5 在大多数模型与数据集上兼顾稀疏性与光滑性，Transformer 需要更高 p；p=1 与 p=2 显得不够优；

**⚠️ 局限性**

局限性：
- 仅在白盒攻击场景下评估；
- 光滑性度量仍带有主观性，仅对 2D 图像分类问题；
- 只考虑了 p∈[1,2]，未探讨更大或更小 p；
- ε_p 的选择基于 1/3 准确率下降，可能不适用于所有任务；
- 对“sweet spot”p=1.3–1.5 的机制尚未深入理论解释。

---

## 286. Think Proprioceptively: Embodied Visual Reasoning for VLA Manipulation

**arXiv ID:** 2602.06575 | [PDF](https://arxiv.org/pdf/2602.06575v1)

**作者:** Fangyuan Wang `[一作]` (Hong Kong Polytechnic University), Guodong Guo `[通讯]` (Eastern Institute of Technology)

**通讯引用:** 15120 | [OpenAlex ID](https://openalex.org/A5085022758)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出将机器人本体感知（如关节角度、抓手位置）离散化为文本token，并与语言指令一起注入VLM，利用指令+本体信息引导视觉token选择，从而提升VLA策略在长链任务中的表现。

**💡 创新点**

核心创新在于（1）将本体感知映射到VLM词表token，使其能与视觉与语言在同一嵌入空间共处理；（2）基于指令和本体的物理感知token选择机制，显著减少视觉token数量而不牺牲性能。

**🔧 技术方法**

使用预训练的Florence-2-Large视觉语言模型作为主干，token化本体感知并通过交叉注意力进行token选择，采用Diffusion Policy（DiT）作为动作头，并用Gumbel软硬选择实现可梯度的离散化。

**📊 数据集**

在CALVIN（机器人操作链任务）和LIBERO（多任务长短期评估）数据集上进行实验，另外在实际机器人硬件上做了初步验证。

**📈 对比分析**

与OpenVLA、π_0、π_0.5、FLOWER等基线对比，平均链长度提升至4.55（比FLOWER高0.1），同时视觉token仅保留15个，推理延迟降低至22 ms，显著优于基线。

**⚠️ 局限性**

局限性包括：离散化本体感知导致数值精度下降；对复杂真实世界动态的适应性仍有限；长时间推理依赖更多数据与更细粒度感知。

---

## 287. Baichuan-M3: Modeling Clinical Inquiry for Reliable Medical Decision-Making

**arXiv ID:** 2602.06570 | [PDF](https://arxiv.org/pdf/2602.06570v1)

**作者:** Baichuan-M3 Team `[一作]` (Baichuan AI), Zhishou Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 Baichuan‑M3，一款将临床询问与决策推理统一的医学大语言模型，强调主动获取信息、长周期推理和自适应误差抑制；

**💡 创新点**

创新点在于三阶段训练框架（TaskRL→离线策略蒸馏→多教师在线蒸馏）和分段流程强化学习，结合动态评卷和事实验证以实现主动询问、连贯推理与低幻觉；

**🔧 技术方法**

使用了分段流程强化学习（Segmented Pipeline RL）、Step‑Penalized Advantage with Relative baseline（SPAR）、动态评卷演化、事实感知强化学习、Gated Eagle‑3 预测解码、INT4 量化自生成校准等技术；

**📊 数据集**

数据集包括新建的 ScanBench（模拟 OSCE 风格三阶段诊疗流程，303 例多科室）和 HealthBench/HealthBench‑Hallucination（多难度医疗推理任务），以及内部患者模拟器；

**📈 对比分析**

与 GPT‑5.2‑High、Deepseek‑V3.2‑Thinking、Qwen3‑235B‑Thinking 等模型以及人工专家对照，Baichuan‑M3 在 ScanBench 各站点均领先（Clinical Inquiry 74.9，Lab 72.1，Diagnosis 74.4），HealthBench 总分 65.1、Hard 子集 44.4，Hallucination 率仅 3.5%，表现显著优于同类模型；

**⚠️ 局限性**

局限在于仅支持文本对话、缺乏纵向疾病管理、多模态输入和超长上下文推理，幻觉虽降低但仍有高风险错误，未实现完整病程路径与实时检索融合。

---

## 288. TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders

**arXiv ID:** 2602.06563 | [PDF](https://arxiv.org/pdf/2602.06563v1)

**作者:** Yuchen Jiang `[一作]` (ByteDance), Peng Xu `[通讯]` (ByteDance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 TokenMixer‑Large，扩展原始 TokenMixer 架构，在工业级推荐场景中进行大规模模型训练与部署，解决残差设计、梯度消失、MoE 稀疏化以及规模化等核心问题。

**💡 创新点**

主要创新包括：Mixing & Reverting 结构实现维度对齐与残差稳定；per‑token SwiGLU 代替 FFN 以增强 token 级表达；Pre‑Norm 与间隔残差+辅助损失提升深层梯度；Sparse‑Pertoken MoE 统一稀疏训练与稀疏推理；Token‑Parallel 与 FP8 量化等硬件友好加速；以及纯模型设计去除碎片化算子。

**🔧 技术方法**

技术手段包括：TokenMixer 轻量混合、per‑token SwiGLU、RMSNorm、间隔残差与辅助损失、Sparse‑Pertoken MoE（gate scaling、共享专家、down‑matrix 小初始化）、FP8 E4M3 量化、Token‑Parallel 模型并行、定制 MoE 操作（MoEPermute/MoEGroupedFFN/MoEUnpermute）。

**📊 数据集**

使用真实业务数据集：抖音电商（≈400M/day）、抖音广告（≈300M/day）与抖音直播（≈17B/day）等，覆盖数百亿用户与多种业务场景。

**📈 对比分析**

与 DLRM‑MLP、DCNv2、AutoInt、HiFormer、DHEN、Wukong、Group Transformer、RankMixer 等 SOTA 模型在 AUC/UAUC、参数、FLOPs、MFU 等指标下对比，TokenMixer‑Large 在 AUC 上提升 1.14%，参数保持 5B、FLOPs 4.2T，MFU 提升至 60%；在线指标提升 ADSS 2.0%、GMV 2.98%、订单 1.66%。

**⚠️ 局限性**

局限性：需要海量数据与长训练周期（如从 500M 到 2B 需 60 天收敛）；对 GPU 内存与通信仍有挑战；Sparse‑Pertoken MoE 的实现复杂；目前主要验证在业务线上，跨任务/跨域适用性待进一步探索。

---

## 289. Which Graph Shift Operator? A Spectral Answer to an Empirical Question

**arXiv ID:** 2602.06557 | [PDF](https://arxiv.org/pdf/2602.06557v1)

**作者:** Yassine Abbahaddou `[一作]` `[通讯]` (Ecole Polytechnique), Yassine Abbahaddou (Ecole Polytechnique)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于谱失真（Maximum Spectral Distortion, MSD）的零样本指标，用来在训练前对图神经网络的图位移算子（GSO）进行排名和选择，理论上可通过最小化该失真来收紧泛化误差上界，进而实现无需搜索即可得到最优的 GSO；

**💡 创新点**

创新点在于将输入特征子空间与标签子空间的几何对齐量化为谱失真，并证明该量化指标与 GNN 的泛化性能直接相关，提出了一种全新训练‑无关的 GSO 选择框架；

**🔧 技术方法**

主要技术包括：基于图 Laplacian 的离散化近似、通用 Rayleigh 商求解最大广义特征值、谱失真指标的稳定性与尺度不变性证明、以及层级化与可学习 GSO 的温和化策略；

**📊 数据集**

实验数据集涵盖了多种图结构和任务，如 Cora、Citeseer、CiteSeer、ArXiv‑Year、Wisconsin、Cornell 等常用节点分类基准；

**📈 对比分析**

通过将 MSD 的逆值与不同 GSO（如对称规范化 Laplacian、随机游走 Laplacian、重正化邻接等）的最终测试精度进行相关性评估，发现两者高度一致，且 MSD 选出的 GSO 在单层及多层 GNN 上均能匹配或超过传统固定 GSO 基线的性能；

**⚠️ 局限性**

局限性包括：依赖于对输入特征和标签构造的 k‑NN 图和 Laplacian 的近似，可能受噪声或稀疏特征影响；对动态图或极大规模图的扩展需要进一步优化；同时，指标对极端异构图（极高/低同质性）可能需额外调参。

---

## 290. Refining the Information Bottleneck via Adversarial Information Separation

**arXiv ID:** 2602.06549 | [PDF](https://arxiv.org/pdf/2602.06549v1)

**作者:** Shuai Ning `[一作]` (University of Jinan), Bo Yang `[通讯]` (Quan Cheng Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种自监督的对抗信息分离框架 AdverISF，用于在没有显式噪声标签的情况下将任务相关特征与噪声分离，从而提升少量数据和外域泛化性能。

**💡 创新点**

创新点在于：① 用判别器对齐联合分布与边缘分布实现统计独立；② 采用多层分离结构逐级回收被误判为噪声的细粒度特征；③ 在噪声分支加入联合预测器防止噪声编码退化。

**🔧 技术方法**

核心技术包括信息瓶颈理论、变分信息瓶颈（VIB）、自监督对抗训练（WGAN‑GP）、多层编码器结构以及联合预测损失。

**📊 数据集**

使用了三类数据集：Synthetic（人工构造的主/细粒度特征噪声数据）、Concrete（混凝土抗压强度数据）、AEP（家用电器能耗预测数据）以及CIFAR‑10的少样本分类任务。

**📈 对比分析**

与 MLP、VIB、infoR‑LSF 等基线相比，AdverISF 在 10%–70% 训练数据比例、Concrete、AEP 以及 CIFAR‑10 上均取得更高的 R² 或准确率，尤其在极少样本和外域测试时表现显著优于对手。

**⚠️ 局限性**

局限性包括：需要手动设定多层维度和正则化权重，训练过程相对复杂；对极端高维数据的可扩展性尚未完全验证；对抗训练的稳定性仍受梯度惩罚参数影响。

---

## 291. MTQE.en-he: Machine Translation Quality Estimation for English-Hebrew

**arXiv ID:** 2602.06546 | [PDF](https://arxiv.org/pdf/2602.06546v1)

**作者:** Andy Rosenbaum `[一作]` (Independent Researcher), Ilan Kernerman `[通讯]` (Lexicala)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文发布了首个公开可用的英-希 MT 质量估计数据集，并在其上对 ChatGPT、TransQuest 与 CometKiwi 进行基线评估、模型集成和参数高效微调。

**💡 创新点**

创新点在于①首次公开英-希 MT 质量估计数据集；②将 ChatGPT 作为提示式基线模型进行实验；③探索并验证 LoRA、BitFit、FTHead 等轻量化微调方法在低资源语言上的有效性。

**🔧 技术方法**

采用了 ChatGPT 提示、TransQuest（XLM‑RoBERTa‑large）与 CometKiwi（InfoXLM‑large）三种模型，并通过 Pearson、Spearman 相关系数进行评估；对模型进行 ensemble、全参数微调、LoRA、BitFit、FTHead 等实验。

**📊 数据集**

使用了 959 条英-希 MT 对齐样本（来自 WMT24++ 领域），每条样本由三名人工专家给出 0–100 的 Direct Assessment 分数，并对其中 300 条样本进行微调训练。

**📈 对比分析**

实验显示单模型最佳为 CometKiwi（Pearson 0.483，Spearman 0.546），模型集成后 Pearson 提升至 0.547、Spearman 提升至 0.601；LoRA/BitFit/FTHead 等参数高效微调在 TransQuest 和 CometKiwi 上均提升 2–3 个百分点。

**⚠️ 局限性**

局限性包括仅使用 DA 分数（而非更细粒度的 MQM），数据集规模有限且低分样本稀缺，仅使用 Google Translate 产出的译文，且所有微调使用相同超参数，可能并非最优。

---

## 292. DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autonomous Driving

**arXiv ID:** 2602.06521 | [PDF](https://arxiv.org/pdf/2602.06521v1)

**作者:** Feiyang jia `[一作]`, Long Chen `[通讯]` (Xiaomi EV)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 DriveWorld‑VLA 框架，将 Vision‑Language‑Action（VLA）与世界模型（WM）统一到共享的潜在空间，实现对未来场景的可控想象与决策。

**💡 创新点**

核心创新：① 在 LLM 隐藏层共享潜在空间中同时进行场景想象与行动规划；② 使用 Diffusion Transformer 进行动作条件下的“what‑if”推理；③ 三阶段渐进式训练（联合预训练 → 动作可控性微调 → 未来引导评估）实现闭环决策。

**🔧 技术方法**

技术手段：多模态 VLM（InternVL+BEVFormer+Lidar tokenization），Diffusion Transformer（DiT）做动作条件化的流匹配降噪，三阶段自监督与奖励学习，BEV 分割头用于潜在解码。

**📊 数据集**

数据集：NAVSIMv1、NAVSIMv2（闭环评估），nuScenes（开放式 3 秒规划）。

**📈 对比分析**

与现有 E2E、世界模型及 VLA 方法对比：在 NAVSIMv1 达 PDMS 91.3、NAVSIMv2 达 EPDMS 86.8、nuScenes 3 秒 CR 0.16%，均显著优于 DiffusionDrive、WoTE、DriveVLA‑W0 等前沿方法。

**⚠️ 局限性**

局限性：① 仍依赖大规模 VLM 预训练与高算力；② 对 3 秒以内的短期规划效果有限，长期规划尚需进一步验证；③ 需要更丰富的多模态自监督信号，模型在不同传感器配置下的迁移性能待提升。

---

## 293. SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs

**arXiv ID:** 2602.06566 | [PDF](https://arxiv.org/pdf/2602.06566v1)

**作者:** Niccolo Avogaro `[一作]` (IBM Research), Mattia Rigotti `[通讯]` (IBM Research)

**通讯引用:** 5228 | [OpenAlex ID](https://openalex.org/A5081069627)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SPARC框架，将视觉语言模型的感知与推理拆分为独立的两阶段流程。

**💡 创新点**

创新点在于使用隐式相关检测(IRD)与模块化推理，实现测试时可灵活扩展的感知与推理分离，并通过自一致性与加权框融合提升效率。

**🔧 技术方法**

主要技术包括两阶段提示、IRR、加权框融合（WBF）、LoRA微调以及KV缓存共享等。

**📊 数据集**

使用的基准数据集包括V*、HRBench-4K/8K、XLRS等高分辨率视觉推理集，并通过DeepEyes生成的人工标注做训练。

**📈 对比分析**

与基线及“思考图像”方法相比，SPARC在V*上提升了约6.7个百分点，在OLR任务上提升4.6个百分点，同时在相同或更低token预算下实现更高准确率。

**⚠️ 局限性**

局限性包括对特定模型架构的依赖、对低分辨率输入仍需人工标注以及在极端OOD场景下仍可能出现感知失误。

---

## 294. Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction

**arXiv ID:** 2602.06676 | [PDF](https://arxiv.org/pdf/2602.06676v1)

**作者:** Bo Du `[一作]` (Sichuan University), Ji-Zhe Zhou `[通讯]` (Sichuan University)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5033838150)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种名为SICA的单一模型框架，用于统一检测四个图像取证子领域（Deepfake、AIGC、IMDL、Doc）中的伪造图像。

**💡 创新点**

创新点在于首次将高阶语义视为结构先验，利用冻结的CLIP语义映射并通过低秩适配(Low‑Rank Adaptation)来重构“统一但具判别力”的伪造特征空间，解决了传统单模模型因“异质现象”导致的特征空间崩塌问题。

**🔧 技术方法**

核心技术包括：冻结的CLIP Vision Transformer作为语义基座；对自注意力层使用LoRA低秩更新以保持语义先验并捕捉域特异伪造特征；以及对更新矩阵进行谱分析验证其对语义短路的抑制。

**📊 数据集**

构建了新的OpenMMSec数据集，集成了19个公开取证数据集，共计33万+样本，覆盖4个子领域、15个主伪造类型和98个细粒度伪造类型。

**📈 对比分析**

在OpenMMSec上与15种现有方法（包含15个不同的backbone和各子领域检测器）进行对比，SICA在ACC、AUC、AP和F1四项指标上均击败或与最佳方法持平，且在统一训练时几乎无性能下降，表明成功实现了跨域伪造特征的近正交重构。

**⚠️ 局限性**

主要局限包括：对预训练语义基座的依赖，若输入域的语义与预训练数据差异大（如复杂场景或极端伪造类型），SICA易失效；低秩参数需要经验调优，过大可能导致语义过拟合，过小则无法充分建模伪造特征。

---

## 295. CytoCrowd: A Multi-Annotator Benchmark Dataset for Cytology Image Analysis

**arXiv ID:** 2602.06674 | [PDF](https://arxiv.org/pdf/2602.06674v1)

**作者:** Yonghao Si `[一作]` (Sun Yat-sen University), Jian Yin `[通讯]` (Sun Yat-sen University)

**通讯引用:** 23631 | [OpenAlex ID](https://openalex.org/A5070570063)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了CytoCrowd，一个包含四名独立病理学家原始标注和一名资深专家校准后的金标准的细胞学图像数据集；同时为该数据集提供了两项基准任务——医学目标检测与分类以及多标注聚合，并给出基线实验结果。

**💡 创新点**

①首次提供同时包含原始专家标注冲突和独立金标准的医学图像数据集；②为多标注聚合研究提供真实的专家不一致场景；③将细胞学图像应用于大模型与专用分割模型的对比，揭示领域专业知识的必要性。

**🔧 技术方法**

使用交互式分割与提示式模型（DeepEdit、Anytime）进行目标检测与分类；采用投票、Dawid–Skene、CATD、PM、LFC、ZenCrowd等推理方法进行标注聚合；评估指标为定位正确的样本中的分类准确率。

**📊 数据集**

CytoCrowd数据集：446张40×高分辨率细胞学图像，四位专家共计14579个原始标注，金标准共6402个目标，34个诊断类别。

**📈 对比分析**

与多标注聚合方法比较：多数投票最高准确率0.903，远超Dawid–Skene等模型；与目标检测模型比较：专用分割模型DeepEdit/Anytime分别达到0.899/0.878，而大规模视觉语言模型Qwen-VL-MAX/72B仅约0.44。

**⚠️ 局限性**

限制：数据集规模相对较小（446张）；金标准由单一资深专家确定，可能存在观测偏差；未结合组织学验证标签，缺乏更客观的病理诊断依据。

---

## 296. Code vs Serialized AST Inputs for LLM-Based Code Summarization: An Empirical Study

**arXiv ID:** 2602.06671 | [PDF](https://arxiv.org/pdf/2602.06671v1)

**作者:** Shijia Dong `[一作]` (University of Glasgow), Paul Harvey `[通讯]` (University of Glasgow)

**通讯引用:** 45655 | [OpenAlex ID](https://openalex.org/A5005887898)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出AST(NIT)方法，将完整AST序列化为LLM可输入的序列，并在LLM微调中评估其对方法级代码摘要的效果

**💡 创新点**

创新点在于结合词法注入与结构归一化的AST增强，并通过节点索引遍历(NIT)生成紧凑且保持层级信息的序列，显著减少输入长度

**🔧 技术方法**

技术包括Tree-sitter解析、AST增强（词法注入+结构归一化）、节点索引遍历序列化、LLaMA-3.1-8B微调（LoRA）以及常规代码序列输入对比

**📊 数据集**

使用CodeXGLUE Python子数据集（约30k训练、2.7k验证、3k测试），并对比四种输入表示：Code、AST(Preorder)、AST(SBT)、AST(NIT)

**📈 对比分析**

实验显示AST(NIT)与Code在BLEU、METEOR、ROUGE-L、BERTScore上表现相近，且比AST(SBT)平均输入长度缩短28.6%，训练时间下降11.3%；AST(Preorder)显著落后

**⚠️ 局限性**

局限在于评估仅针对单语言Python，LLM预训练可能已覆盖数据，难以充分验证AST优势；未来需扩展至多语言、低资源场景并进行人工评估

---

## 297. Gromov-Wasserstein at Scale, Beyond Squared Norms

**arXiv ID:** 2602.06658 | [PDF](https://arxiv.org/pdf/2602.06658v1)

**作者:** Guillaume Houry `[一作]` (Inria), François-Xavier Vialard `[通讯]` (Universite Gustave Eiffel)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一类可降解为简单对齐问题的失真惩罚，并基于此设计了线性记忆、二次时间复杂度的迭代 Gromov‑Wasserstein 求解器

**💡 创新点**

将 Gromov‑Wasserstein 与线性对齐映射联系起来，揭示了在提升特征空间中可解的结构，并提出了 CNT‑GW 与 CNT‑EGW 算法

**🔧 技术方法**

利用提升特征空间、Sinkhorn 迭代、线性记忆实现以及可微分的算法框架，兼具理论保证和高效性

**📊 数据集**

在合成与真实 3D 点云数据集（如 ShapeNet/ModelNet 以及自制对齐测试集）上进行实验

**📈 对比分析**

与传统 GW、OT、Procrustes‑Wasserstein 等方法对比，取得了更快的收敛速度、可处理百万级点集、对齐误差显著降低

**⚠️ 局限性**

目前尚未处理非平衡与融合 GW、成本函数、嵌入维度及温度退火调度对结果的影响，需进一步完善

---

## 298. RAPID: Reconfigurable, Adaptive Platform for Iterative Design

**arXiv ID:** 2602.06653 | [PDF](https://arxiv.org/pdf/2602.06653v1)

**作者:** Zi Yin `[一作]` (Tsinghua University), Jia Liu `[通讯]` (Tsinghua University)

**通讯引用:** 36301 | [OpenAlex ID](https://openalex.org/A5100409741)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个可快速拆装、支持手持与机器人部署的多模态机器人操作平台 RAPID，并通过驱动层的 Physical Mask 实时感知硬件状态。

**💡 创新点**

创新点在于：① 采用工具无关的 3D 打印 mortise‑and‑tenon 连接实现秒级硬件切换；② 将硬件热插拔事件映射为驱动层的 Physical Mask，弥补了多模态系统的可观测性缺口；③ 通过统一的 USB 互连与 ZeroMQ 中间件实现多模态数据的自动发现、同步与零填充。

**🔧 技术方法**

技术实现包括：工具无关模块化硬件设计、USB 事件驱动的设备注册与物理掩码生成、ZeroMQ/Zeroconf 轻量级消息总线、基于 Diffusion Policy 的 mask‑aware 推理、以及 500 Hz 的 Physical Mask 设备文件。

**📊 数据集**

使用的是作者自行采集的多模态数据集（手持抓取与机器人执行的视觉、触觉、关节位姿等组合），未使用公开基准数据集。

**📈 对比分析**

与传统手动配置/手动重启的工作流相比，RAPID 将每种硬件配置的准备时间从约 480 s 降至 5 s，实现了约 100 倍的加速；在热插拔测试中，基于 Physical Mask 的系统能够在传感器拔除时平滑降级而不崩溃，表现出高可靠性。

**⚠️ 局限性**

局限性包括：USB 集线器带宽受限，难以支持多台高分辨率摄像头；同步精度不足以满足高速动态抓取；仍需手动外参标定；以及 PLA 打印接口在长期频繁插拔后可能失效。

---

## 299. A Survey of Security Threats and Trust Management in Vehicular Ad Hoc Networks

**arXiv ID:** 2602.06608 | [PDF](https://arxiv.org/pdf/2602.06608v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 300. Graph-Based Nearest-Neighbor Search without the Spread

**arXiv ID:** 2602.06633 | [PDF](https://arxiv.org/pdf/2602.06633v1)

**作者:** Jeff Giliberti `[一作]`, Ali Vakilian `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构造了一种无尺度（spread）依赖的图基近似最近邻搜索数据结构，能在对数时间内得到 (1+ε) 近似最近邻，并且空间为线性。

**💡 创新点**

创新点在于通过将粗略搜索结果映射到不同分辨率下的子图，并利用逆向树跳过多余的分辨率层，最终实现了既保持线性空间又消除对尺度因子依赖的实现；此外，对贪心排列的搜索图进行了全局优化，得到统一的线性大小图。

**🔧 技术方法**

主要技术包括：双重维数度量下的层次分离树（HST）、贪心排列及其友人列表、逆向树（reverse tree）以及基于分辨率切片的子图构造；算法使用二进制搜索、前驱查询和贪心路由。

**📊 数据集**

论文中未给出具体实验数据集，而是以理论分析为主，讨论了在任意双重维度度量空间（包括欧氏空间）下的性能。

**📈 对比分析**

与传统依赖对数尺度因子的方法相比，该方法在查询时间上保持 O(log n)（或 O(log n + (1/ε)^d log (1/ε))）并将空间降到 O(n/2^d)（欧氏情况），相比之下先前方法需要 O(log Δ) 的查询时间，Δ 为输入点集的尺度因子，且空间略高。

**⚠️ 局限性**

限制包括：需已知/可在 O(n log n) 时间内构造贪心排列；常数因子与维度呈指数增长；对具有高双重维数的空间效果不佳；未给出实验验证，理论上性能依赖于对偶树和分辨率切片的构造效率。

---

## 301. TrapSuffix: Proactive Defense Against Adversarial Suffixes in Jailbreaking

**arXiv ID:** 2602.06630 | [PDF](https://arxiv.org/pdf/2602.06630v1)

**作者:** Mengyao Du `[一作]` (National University of Defense Technology), Ee-Chien Chang `[通讯]` (National University of Singapore)

**通讯引用:** 4573 | [OpenAlex ID](https://openalex.org/A5105408906)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种主动防御 suffix-based jailbreak 的方法 TrapSuffix，旨在通过在模型中嵌入陷阱行为，使攻击者在优化其 suffix 时要么陷入局部最优导致失败，要么只能生成带可追踪指纹的 suffix，从而实现防御与追踪双重功能。

**💡 创新点**

创新点在于：①将主动防御思想应用于 suffix-based jailbreak；②利用低秩适配 (LoRA) 将陷阱行为注入模型而不改变推理流程；③设计四个互补损失（局部最优、陷阱安全、梯度吸引、语义终止）共同塑造攻击者的优化景观；④实现攻击成功即必带可追踪指纹的机制。

**🔧 技术方法**

主要技术包括：LoRA 参数高效微调、对抗损失抽象、局部最优约束、陷阱安全正则、梯度引导吸引、语义终止约束以及与原始任务损失的联合优化。

**📊 数据集**

使用的数据集：JailbreakBench（100 个有害问题）用于训练对抗场景；Databricks Dolly-15K 用于保持模型通用能力；ARC‑C、HellaSwag、MMLU 等基准评估防御后模型的实用性。

**📈 对比分析**

在三大开源模型（Qwen‑2.5‑7B、LLaMA‑3‑8B、Vicuna‑13B）和多种攻击（GCG、AutoDAN、Probe、Transfer、JSAA）上与 12 种基线进行对比。TrapSuffix 的平均攻击成功率（ASR）<0.01%，追踪成功率（TSR）87.9%；无推理延迟，内存占用仅 15.87 MB，明显优于传统检测/过滤型防御，且在自适应攻击下保持零 ASR。

**⚠️ 局限性**

局限性：①只针对基于 suffix 的迭代优化攻击，无法直接应对非迭代或语义层面攻击；②需要访问并修改模型内部参数，训练成本随模型规模上升；③对极大模型的训练资源需求仍不低。

---

## 302. R2LED: Equipping Retrieval and Refinement in Lifelong User Modeling with Semantic IDs for CTR Prediction

**arXiv ID:** 2602.06622 | [PDF](https://arxiv.org/pdf/2602.06622v1)

**作者:** Qidong Liu `[一作]` (Xi'an Jiaotong University), Chen Li `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 29713 | [OpenAlex ID](https://openalex.org/A5100379155)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用语义ID（SID）进行多粒度检索与双层融合的两阶段终身用户建模框架，用于CTR预测。

**💡 创新点**

创新点在于：①将SID嵌入检索阶段，构建多路混合检索（Target、Recent、Global）以提升检索精度并降低噪声；②设计双层融合（路级交叉注意力 + SID级门控融合）实现语义与协同空间的对齐，提升语义理解与长尾表现。

**🔧 技术方法**

采用SID生成（RQ‑KMeans+Flan‑T5‑XL）+前缀树检索、LSH检索、交叉注意力池化、门控融合、MLP预测层等技术；训练使用二元交叉熵，推理时缓存SID与检索索引。

**📊 数据集**

使用公开的JD和Pixel‑1M两个大规模电商数据集，分别包含约3.4万/168万用户、约0.6/0.1百万商品、数百万交互，最大历史长度300。

**📈 对比分析**

与DIN、DIEN、BST、TransAct、SIM、ETA、TWIN、SDIM、MIRRN、ReLLa、SEmb等基线进行对比。实验表明在AUC和LogLoss上均优于传统序列模型与终身建模方法，尤其在长尾子集上表现显著提升；在推理速度上略逊于MIRRN，但相较于ReLLa/SEmb获得更好效果与效率的折衷。

**⚠️ 局限性**

局限性在于：①需要离线生成并维护SID和检索索引，增加预处理成本；②模型仍依赖固定的三层SID分辨率，过细或过粗可能导致检索或融合效果受限；③在极大规模场景下，检索与门控融合的计算仍非完全实时，需进一步压缩或加速。

---

## 303. Mapping the political landscape from data traces: multidimensional opinions of users, politicians and media outlets on X

**arXiv ID:** 2602.06604 | [PDF](https://arxiv.org/pdf/2602.06604v1)

**作者:** Antoine Vendeville `[一作]` (Sciences Po), Pedro Ramaciotti `[通讯]` (Complex Systems Institute of Paris Ile-de-France CNRS)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了一个包含近100万名法国X（原Twitter）用户、883名议员以及400个媒体域名的多维度政治立场数据集，并提供了用户活跃度和受欢迎度指标。

**💡 创新点**

创新点在于：①使用关注网络进行多维度意识形态缩放，得到连续的16维政治立场；②将隐空间位置映射到CHES问卷的多个维度，克服了传统单维度的局限；③结合人类与LLM对个人自述的政治标签进行验证，使数据集在多维度上的可靠性得到证实；④首次公开完整的议员、追随者及媒体多维立场与社交指标的结合数据，便于跨学科研究。

**🔧 技术方法**

技术方法包括：①构建MP–追随者的双向网络；②采用Barberá的理念缩放模型和对应分析（CA）估计隐空间位置；③使用岭回归拟合从隐空间到CHES维度的仿射变换；④对自述文本进行人类和LLM双重注释，用于验证；⑤使用逻辑回归评估标签与维度的一致性，并计算ROC AUC和F1等指标。

**📊 数据集**

数据来源：2023年2月收集的法国议员及其追随者的关注关系（约978,206追随者，883名议员），以及2022‑2023年间收集的包含URL的X推文（约1.4 b亿条），从中提取400个被频繁引用的法国媒体域名；同时使用2019/2023年的Chapel Hill Expert Survey（CHES）作为参考维度。

**📈 对比分析**

比较方法：用人类和LLM对推文自述文本标注政治标签；对标签与推断维度做逻辑回归，计算ROC AUC与F1；对不同问卷（2019 vs 2023）和不同维度（如左右、欧盟、移民等）进行Pearson相关性比较。性能表现：大多数维度ROC AUC>0.9，F1>0.6，显示高可靠性；对媒体域名的验证与先前的中心化分类高度一致。

**⚠️ 局限性**

局限性：①样本仅限关注法国议员的X用户，非法国总体人群的代表性不足；②立场映射依赖CHES问卷，受其维度设置限制；③仅使用关注关系，不涉及文本内容；④对媒体位置的解释侧重于被引用者的立场，未反映媒体自身立场；⑤由于平台API限制，数据更新受限。

---

## 304. The hidden risks of temporal resampling in clinical reinforcement learning

**arXiv ID:** 2602.06603 | [PDF](https://arxiv.org/pdf/2602.06603v1)

**作者:** Thomas Frost `[一作]` (University College London), Steve Harris `[通讯]` (University College London)

**通讯引用:** 3846 | [OpenAlex ID](https://openalex.org/A5068199165)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

评估临床决策不规则时间间隔对离线强化学习模型部署性能的影响。

**💡 创新点**

发现时间重采样导致对抗事实轨迹、时间预期失真和泛化误差堆积三种机制，并揭示传统OPE对重采样数据的过度乐观。

**🔧 技术方法**

采用SMDP框架、行为克隆、IQL、CQL算法，并利用FQE进行离线评估；在LavaGap与UVA/Padova模拟器上进行实验。

**📊 数据集**

基于PPO专家生成的真实时间间隔数据，随后构造未处理、插值、时间分箱三种版本的数据集。

**📈 对比分析**

对三种数据集在正则与不规则环境中训练并评估，未处理数据表现最佳，分箱/插值显著退化；FQE在分箱数据上严重过估。

**⚠️ 局限性**

仅涵盖连续动作空间、未涉及基于模型方法、观测维度低、未进行真实临床部署验证。

---

## 305. Beyond Static Alignment: Hierarchical Policy Control for LLM Safety via Risk-Aware Chain-of-Thought

**arXiv ID:** 2602.06650 | [PDF](https://arxiv.org/pdf/2602.06650v1)

**作者:** Jianfeng Si `[一作]` (Qiyuan Tech), Xiangzheng Zhang `[通讯]` (Qiyuan Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出PACT框架，实现基于层级安全策略的可控安全响应；

**💡 创新点**

通过全局安全策略与可配置用户策略的分层设计、以及链式思考路径（CoTPath）实现风险识别与动作映射的显式透明化，显著缓解安全-有用性权衡；

**🔧 技术方法**

使用自我蒸馏构建风险标签与多模式响应数据，基于链式思考的统一监督微调（SFT）以及结构化的CoTPath决策逻辑；

**📊 数据集**

融合多来源数据集：通用问答（中文/英文）、安全攻击样本（Safety-Prompts、NVIDIA Aegis-AI-2.0）、红队模型生成的双语风险提示，共计约573k条训练样本；

**📈 对比分析**

在五大公共安全基准上与8B-671B规模模型对比，PACT在安全率≥0.95时保持与TR1S-8b/pos、gpt-oss-120b相当的安全率，同时在帮助性和CoSA分数上实现最佳或接近最佳；在可控性测试（CoSApien、PACT-test）中，PACT在用户策略下达成最高平均可控性（0.557），并在全局策略触发时保持高安全性；

**⚠️ 局限性**

主要局限包括：全局与用户策略边界模糊导致灰区处理不足；CoTPath推理产生额外token消耗；对抗性标签操纵和策略冲突带来鲁棒性挑战；顺序推理错误累积可能影响整体决策。

---

## 306. Beyond Pairwise Distance: Cognitive Traversal Distance as a Holistic Measure of Scientific Novelty

**arXiv ID:** 2602.06607 | [PDF](https://arxiv.org/pdf/2602.06607v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 307. Reading Between the Waves: Robust Topic Segmentation Using Inter-Sentence Audio Features

**arXiv ID:** 2602.06647 | [PDF](https://arxiv.org/pdf/2602.06647v1)

**作者:** Steffen Freisinger `[一作]`, Korbinian Riedhammer `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对在线音频视频的主题分割，提出一种多模态模型，联合微调文本编码器与Siamese音频编码器，专门提取句间边界的声学特征，从而实现更精准的主题切换检测。

**💡 创新点**

创新点在于：①使用Siamese双窗口结构只关注句间边界的声学信号；②在端到端框架下同时微调文本与音频编码器；③通过拼接句子文本嵌入与边界声学特征，实现信息融合而非简单拼接；④验证该方案对ASR噪声和跨语言数据的鲁棒性。

**🔧 技术方法**

核心技术包括：MiniLM文本编码器、RoFormer标签器、wav2vec 2.0（可微调）音频编码器、Siamese双窗口抽取、拼接+全连接分类层；训练采用AdamW、BCE损失，并在句子层进行梯度裁剪和采样。

**📊 数据集**

主要数据集：YTSeg（约19k英语YouTube视频）用于训练、验证和测试；附加跨语言评估使用AVLectures（英语）、Videoaula（葡萄牙语）和LectureDE（德语）三大讲座语料；ASR误差实验采用Whisper不同规模模型和Vosk。

**📈 对比分析**

与基线比较：相较于文本仅模型MiniSeg、MiniSeg+、Cross‑segment BERT以及先前多模态模型MiniSeg+L3‑Net，MultiSeg在F₁和Boundary Similarity指标上分别提升≈5.4和≈7.3；在ASR噪声下相对损失更小；跨语言测试中，葡萄牙语提升≈20点、德语提升≈7点，证明模型在非英语环境下的优势。

**⚠️ 局限性**

局限性包括：①模型仍依赖音频信息，无法应用于纯文本场景；②对超大规模数据和实时推理的计算成本较高；③跨语言实验仅使用少数语言，其他语言的鲁棒性尚未验证；④在极低WER的ASR环境下，音频优势不明显，模型的收益有限。

---

## 308. Do Prompts Guarantee Safety? Mitigating Toxicity from LLM Generations through Subspace Intervention

**arXiv ID:** 2602.06623 | [PDF](https://arxiv.org/pdf/2602.06623v1)

**作者:** Himanshu Singh `[一作]` (Indraprastha Institute of Information Technology Delhi), Mohan Kankanhalli `[通讯]` (National University of Singapore)

**通讯引用:** 17059 | [OpenAlex ID](https://openalex.org/A5016415049)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们提出一种在推理时对大型语言模型隐藏毒性方向进行子空间投射干预的方法，以降低有毒生成并保持流畅性。

**💡 创新点**

利用梯度敏感度发现低维毒性子空间，并通过特征空间对齐而非权重编辑，提供更小的假设空间和更强的泛化保证。

**🔧 技术方法**

梯度基的毒性子空间搜索、奇异值分解投影、特征空间线性变换与β参数控制、最终层投射干预。

**📊 数据集**

从含毒性提示集挑选2000条高毒性prompt进行子空间学习，并在Toxicity Challenge集评估毒性、WikiText评估流畅性。

**📈 对比分析**

与现有的SFT、RLHF、直接权重编辑及两种先进去毒方法对比，在Mistral‑7B等模型上实现8‑20%的毒性下降，保持接近原始困惑度，表明安全性提升且生成质量几乎不受影响。

**⚠️ 局限性**

对强干预β值过大会导致流畅度下降；方法仅针对文本毒性；需要额外梯度计算且对新型毒性模式的泛化仍有限。

---

## 309. Adaptive-CaRe: Adaptive Causal Regularization for Robust Outcome Prediction

**arXiv ID:** 2602.06611 | [PDF](https://arxiv.org/pdf/2602.06611v1)

**作者:** Nithya Bhasker `[一作]` (National Center for Tumor Diseases), Stefanie Speidel `[通讯]` (National Center for Tumor Diseases)

**通讯引用:** 6422 | [OpenAlex ID](https://openalex.org/A5003648994)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种无模型依赖的自适应因果正则化方法Adaptive‑CaRe，用于平衡预测精度与因果鲁棒性，避免机器学习模型捕获表面相关性；

**💡 创新点**

通过引入与因果与统计贡献差异成正比的自适应惩罚，提供了可调节的鲁棒性与准确性权衡；

**🔧 技术方法**

利用快速因果推断（FCI）算法估计部分祖先图（PAG）得到因果掩码，结合梯度×输入特征归因，再加上自适应正则化项；

**📊 数据集**

在合成数据（含因果父变量、代理、虚假相关、噪声）以及公开医学数据集（ALARM网络、SUPPORT临床死亡预测）上进行实验；

**📈 对比分析**

与传统逻辑回归、无正则化和带权重衰减/提前停止的MLP，以及CASTLE因果正则化方法比较，Adaptive‑CaRe在测试集上保持或提升F1得分，同时显著抑制非因果特征的贡献；

**⚠️ 局限性**

对因果学习框架的依赖和超参数λ的调节要求，且在高维或噪声数据中可能受限；

---

## 310. Efficient and Robust Modeling of Nonlinear Mechanical Systems

**arXiv ID:** 2602.06639 | [PDF](https://arxiv.org/pdf/2602.06639v1)

**作者:** Davide Tebaldi `[一作]` (University of Modena and Reggio Emilia), Roberto Zanasi `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 1148 | [OpenAlex ID](https://openalex.org/A5053996788)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种针对非线性机械系统的因式分解动态模型，并给出了自动推导该模型的建模流程。

**💡 创新点**

创新点在于用因式分解矩阵将惯性矩阵和科氏矩阵拆解为可直接使用的形式，从而消除对外部变量导数的依赖，提高了对测量噪声的鲁棒性，并在逆动力学计算中显著降低了运算时间。

**🔧 技术方法**

主要技术包括：因式分解模型的数学推导、伪逆与正交投影实现积分因果性、Simulink/Matlab 自动化建模步骤，以及与 Euler‑Lagrange 模型的对比分析。

**📊 数据集**

没有使用公开数据集，而是通过在 MATLAB/Simulink 中对三种机械系统（曲柄-连杆、全托罗idal变速器、2-DOF 平面机器人）进行仿真，采用已知参数与人工噪声生成输入。

**📈 对比分析**

通过将因式分解模型与 Euler‑Lagrange 模型在同一仿真环境下进行比较：在噪声场景中因式分解模型误差几乎为零，而 Euler‑Lagrange 模型误差可达 5%；在 2-DOF 机器人逆动力学计算中，因式分解模型的执行时间比 Euler‑Lagrange 减少约 17%。

**⚠️ 局限性**

局限性包括：仅在仿真层面验证，缺乏实车/机器人实验数据；模型的推导和矩阵构造对系统规模和外部变量的依赖仍需进一步研究；以及在矩阵为矩形时需要伪逆处理，可能导致数值稳定性问题。

---

## 311. Evaluating Prompt Engineering Strategies for Sentiment Control in AI-Generated Texts

**arXiv ID:** 2602.06692 | [PDF](https://arxiv.org/pdf/2602.06692v1)

**作者:** Kerstin Sahler `[一作]` (German Aerospace Center), Sophie Jentzsch `[通讯]` (German Aerospace Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究如何通过Prompt Engineering控制GPT-3.5‑Turbo生成文本的情感表达，并与微调方法进行对比。

**💡 创新点**

创新点在于系统评估多种Prompt技术（Zero‑Shot、Zero‑Shot CoT、Few‑Shot、CoT）对情感控制的影响，并证明Few‑Shot人工示例在资源受限场景下最有效。

**🔧 技术方法**

采用Prompt Engineering（含不同Prompt设计）、GPT‑3.5‑Turbo推理、DistilRoBERTa情感分类器、BERTScore、Flesch可读性等技术。

**📊 数据集**

使用Konen等的40个事实问答+40个主观问答、120个人工情感示例、LLaMA‑2生成示例以及MELD情感识别数据集。

**📈 对比分析**

通过Emotion Score、BERTScore、Distinct‑n、Flesch可读性等指标比较，Few‑Shot人写示例的Emotion Score达0.785，优于Zero‑Shot（0.739）、CoT（0.697）和微调（0.743），且在回应质量上相对更好。

**⚠️ 局限性**

局限在于仅用英文数据、评价依赖机器分类器、微调数据量极少、仅针对ChatGPT、未进行人类评估。

---

## 312. Evaluating and Enhancing the Vulnerability Reasoning Capabilities of Large Language Models

**arXiv ID:** 2602.06687 | [PDF](https://arxiv.org/pdf/2602.06687v1)

**作者:** Li Lu `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 29122 | [OpenAlex ID](https://openalex.org/A5107888510)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文先构建了包含专家因果推理标注的漏洞基准并进行大规模LLM评估，随后提出了将漏洞推理建模为有向无环图（DAG）生成任务并结合RLVR奖励的框架，以提升LLM的推理一致性和准确性。

**💡 创新点**

创新点在于：①提出细粒度专家标注的漏洞因果推理基准；②将漏洞推理视作DAG生成任务，强制结构一致性；③引入RLVR奖励机制对逻辑闭合度与最终结论进行双重约束；④系统梳理并量化12类推理失效模式；⑤通过多模态判定提高评估可靠性。

**🔧 技术方法**

使用技术包括：大规模LLM评估、链式思考（CoT）与检索增强生成（RAG）、有向无环图（DAG）生成与约束、基于程序语义的结构化推理、强化学习与可验证奖励（RLVR）以及多模型判定（GPT‑5）。

**📊 数据集**

数据集方面：1) 5,078 条真实 CVE（149 CWE）并手工标注根因推理；2) 100 对 Juliet 语义保持扰动样本；3) 1,114 条 SVEN 公开样本；4) 结合 PrimeVul、ReposVul、R2Vul 等原始数据源进行去噪和上下文补全。

**📈 对比分析**

方法对比：与 RAG、SFT、ORPO、CoT 基线及 Qwen3‑30B‑Reasoning、GPT‑OSS‑20B‑High 等大型推理模型进行对标。实验显示在基本提示下，DAG+RLVR 在 8B 模型上平均提升 18.9% 的推理 F1，且在 CWE‑guided 提示下达到 75.47% 的 F1，几乎与 76.11% 的 Claude‑Sonnet‑4.5 同级，甚至超过同规模大型推理模型。

**⚠️ 局限性**

局限性包括：仅在 C/C++ 代码上验证，跨语言通用性待证；评估判定主要依赖 GPT‑5，可能存在细微偏差；鲁棒性在极端语义扰动下仍显不足；对未知 CWE 的泛化能力尚未充分评估。

---

## 313. Wonderboom -- Efficient, and Censorship-Resilient Signature Aggregation for Million Scale Consensus

**arXiv ID:** 2602.06655 | [PDF](https://arxiv.org/pdf/2602.06655v1)

**作者:** Zeta Avarikioti `[一作]` (TU Wien and Common Prefix), Michelle X. Yeo `[通讯]` (Nanyang Technological University and Aarhus University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一种针对百万级验证者的树形签名聚合协议，实现单槽内聚合所有签名。

**💡 创新点**

创新点在于实现了在保持完全去中心化的同时，提供比以太坊更强的投票审查抵抗性，并通过点对点通信显著减少网络延迟。

**🔧 技术方法**

使用BLS签名聚合、层级树结构、随机代表选取、公共密钥压缩以及模拟器实现。

**📊 数据集**

使用以太坊当前验证者规模（约100万）以及扩展到250万节点的虚拟网络进行实验。

**📈 对比分析**

与以太坊原有聚合方案对比，Wonderboom在最坏情况下单槽聚合超过200万签名，速度比以太坊快约4倍，最终两槽最终性延迟仅约24秒。

**⚠️ 局限性**

局限在于对完全适应性攻击的安全性无法保证，且在高失活率或网络分区情况下仍需更多的同步假设。

---

## 314. Talk Like a Packet: Rethinking Network Traffic Analysis with Transformer Foundation Models

**arXiv ID:** 2602.06636 | [PDF](https://arxiv.org/pdf/2602.06636v1)

**作者:** Samara Mayhoub `[一作]` (Aston University), Rahim Tafazolli `[通讯]` (University of Surrey)

**通讯引用:** 19243 | [OpenAlex ID](https://openalex.org/A5032549075)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了一套基于Transformer的流量基础模型预训练与微调管线，证明其在流量分类、特征预测和生成等多任务下具备良好的通用性能。

**💡 创新点**

创新点在于：①将网络流量视为“语言”构建统一的预训练框架；②对现有模型进行体系化分类；③展示基础模型在多任务场景下相较传统模型显著提升的效果。

**🔧 技术方法**

采用Transformer编码/解码结构，结合Masking、Masked Autoencoder (MAE)、Contrastive、Same-Origin 预测等自监督预训练策略，并在微调时添加分类/回归/生成头。

**📊 数据集**

使用公开原始PCAP数据集CICIoT2023（含IoT恶意与正常流量）和CIC-IDS-2017（包含多种攻击与正常流量）进行预训练与微调实验。

**📈 对比分析**

与传统非基础模型基线对比，基于Transformer的模型在流量分类中实现了96.9%准确率，特征预测任务MAE降至18字节、R²达0.934，生成任务中TTL与IP长度分布与真实数据高度吻合。

**⚠️ 局限性**

主要限制包括：Transformer的O(n²)计算与内存开销导致长流量序列处理困难；实时性与可解释性不足；需要进一步压缩模型、引入稀疏/低秩注意力以及可解释机制。

---

## 315. HYDRA: Unearthing "Black Swan" Vulnerabilities in LEO Satellite Networks

**arXiv ID:** 2602.06612 | [PDF](https://arxiv.org/pdf/2602.06612v1)

**作者:** Bintao Yuan `[一作]` (Beihang University), Zijie Yan `[通讯]` (Beihang University)

**通讯引用:** 630 | [OpenAlex ID](https://openalex.org/A5100375485)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出HYDRA框架，通过超图动态风险分析识别低轨卫星网络中的“黑天鹅”节点，并在真实Starlink网络中验证其有效性。

**💡 创新点**

创新点包括：①提出Hyper‑Bridge Centrality（HBC）度量，以负载/冗余比评估节点的结构性脆弱性；②将卫星束、地面门户等多层结构映射为超图，捕捉一对多的高阶依赖；③在真实TLE轨道数据上实现时变拓扑与负载级联仿真。

**🔧 技术方法**

技术手段：超图理论、SGP4轨道传播、基于负载的最短路径路由、离散时间级联失效模拟、基于人口重力模型的交通生成、Monte Carlo仿真、相关性与性能评估。

**📊 数据集**

使用的数据集包括：Starlink TLE（CelesTrak）、全球人口分布与AS路由的重力模型、Cloudflare Radar 的星链流量样本，用于校准地面网关需求。

**📈 对比分析**

与度、介数、PageRank等传统中心性指标对比。实验结果显示HBC与传统度相关系数接近0，HBC在级联攻击中可实现24.1%更高的级联影响率，优于度中心性；相较介数提升约8.2%。在90分钟、3天仿真中均保持显著优势。

**⚠️ 局限性**

局限性：理论复杂度为O(N²logN)，实际采用采样与剪枝后为O(NlogN)；使用离散快照而非连续时间模型；交通模型为合成重力模型，缺少真实运营流量；验证仅针对Starlink，需扩展至OneWeb、Kuiper等星座；未考虑动态路由协议细节与实时负载估计。

---

## 316. The challenge of generating and evolving real-life like synthetic test data without accessing real-world raw data -- a Systematic Review

**arXiv ID:** 2602.06609 | [PDF](https://arxiv.org/pdf/2602.06609v1)

**作者:** Maj-Annika Tammisto `[一作]` (University of Tartu), Dietmar Pfahl `[通讯]` (University of Tartu)

**通讯引用:** 4468 | [OpenAlex ID](https://openalex.org/A5022214666)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对生成和演化与真实数据相似的合成测试数据的研究方法进行系统综述，筛选并分析了37篇符合条件的文献，揭示了该领域的技术现状和空白。

**💡 创新点**

首次将“无真实数据输入”与“合成数据演化”作为检索与筛选准则，系统评估该领域方法缺口，并指出演化能力极为稀缺。

**🔧 技术方法**

综述涵盖规则生成、进化算法、分类/回归模型、深度学习、图像/视频渲染、仿真环境等多种合成数据技术。

**📊 数据集**

文献多使用公开测试集或人工构造数据，未使用真实个人数据；仅少数研究使用模拟或自建的数据集来验证方法。

**📈 对比分析**

作者基于四项质量评估标准对文献进行打分，发现大多数文献缺少演化能力；最高分案例来自规则生成与UML/OCL方法，性能受计算资源与生成量限制。

**⚠️ 局限性**

综述受数据库范围、关键词选择、主观筛选及缺乏统一质量评估指标等限制；且绝大多数方法仍需真实数据，演化能力极为罕见。

---

## 317. Scaling Speech Tokenizers with Diffusion Autoencoders

**arXiv ID:** 2602.06602 | [PDF](https://arxiv.org/pdf/2602.06602v1)

**作者:** Yuancheng Wang `[一作]` (Chinese University of Hong Kong), Xubo Liu `[通讯]` (Meta Superintelligence Labs)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种名为 SiTok 的基于扩散自编码器的语音分词器，用于将连续语音信号压缩为极低码率的离散标记，并可在同一框架下实现高质量的音频重建与语义丰富的表示学习。

**💡 创新点**

创新点包括：
• 端到端的扩散自编码器联合学习量化与重建，打破传统分离训练的瓶颈；
• 在量化空间加入 CTC 语义正则化，使离散标记既保留声学细节，又与文字信息保持一致；
• 通过“shortcut fine‑tuning”与轻量化扩散头实现数步快速采样，显著降低推理成本；
• 采用多级残差向量量化 (RVQ) 动态调节码率，提供可控的压缩与质量折中。

**🔧 技术方法**

核心技术：扩散模型（flow‑matching 目标）、向量量化与多级残差向量量化、CTC 语义正则化、轻量化扩散头、快捷微调（shortcut fine‑tuning）以及基于 Vocos 的声码器。

**📊 数据集**

使用了约 2000 万小时的内部多语种语音数据（主要为英语）与对应文本转录，评估集包括 LibriSpeech、SeedTTS 测试集、DASB 基准等。

**📈 对比分析**

与现有的 SpeechTokenizer、EnCodec、DualCodec、WavTokenizer、Mimi、BiCodec 等基线对比，SiTok 在极低码率（0.2 kbps、12.5 Hz）下实现了：
• 重建 WER 4.06、speaker similarity 0.641、UTMOS 3.44；
• ASR WER 4.95、ER 63.5%、SV 13.8%、KS 96.9%；
• 通过 1×、2×、4× 代码本及快捷推理实现了更佳的音频质量与推理速度。

**⚠️ 局限性**

局限性：
• 训练与推理仍需大规模算力，尤其是 1.6 B 参数模型；
• 在极低码率下对语音的细粒度细节仍有欠缺，某些下游任务（如细粒度情感辨识）提升有限；
• 过大模型在某些理解任务上表现略有下降，表明存在“容量过拟合”与“语义细节权衡”问题；
• 公开数据与评测仅覆盖英语与少量其他语种，跨语言泛化性待进一步验证。

---

## 318. Beyond Judgment: Exploring LLM as a Support System for Maternal Mental Health

**arXiv ID:** 2602.06678 | [PDF](https://arxiv.org/pdf/2602.06678v1)

**作者:** Shayla Sharmin `[一作]` (Chittagong University of Engineering and Technology), Sadia Afrin `[通讯]` (University of Delaware)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过10天的混合方法在线调查，探讨孕产妇如何利用大型语言模型（LLMs）获得情感支持、信息确认与避免社会评判的需求；

**💡 创新点**

创新点在于将LLMs定位为情感安全的低风险支持工具，而非医学诊断工具，揭示了文化背景（如共同家庭）对LLMs使用动机的影响；

**🔧 技术方法**

采用常规LLM服务（如ChatGPT、Gemini等）作为信息与情感交互媒介；

**📊 数据集**

使用107名受访孕产妇的自填式问卷数据，包含量化量表与开放式文字回复；

**📈 对比分析**

方法主要为描述性统计、交叉表与主题分析，未涉及模型性能对比，结果呈现LLMs在避免评判、提供即时信息方面的利用频率与受访者对人类情感温度的需求对照；

**⚠️ 局限性**

局限包括样本量有限、便利抽样导致的代表性不足、仅自述数据可能受社会期望偏差、文化背景（主要是南亚共同家庭）可能限制结果在其他文化中的普适性。

---

## 319. Humanoid Manipulation Interface: Humanoid Whole-Body Manipulation from Robot-Free Demonstrations

**arXiv ID:** 2602.06643 | [PDF](https://arxiv.org/pdf/2602.06643v1)

**作者:** Ruiqian Nai `[一作]` (Tsinghua University), Yang Gao `[通讯]` (Tsinghua University)

**通讯引用:** 12880 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个基于机器人自由数据采集的HuMI系统，用于捕捉完整身体运动并训练分层控制器，使仿人机器人能够完成多种全身协作的操纵任务。

**💡 创新点**

创新点在于：①结合手持感知手套和全身追踪器实现无机器人全身采集；②在线IK预览实时反馈身体可行性；③使用扩散策略生成任务空间轨迹；④低层采用自适应端执行器跟踪与可变速度增强的全身控制；⑤通过相对姿态接口解决视觉缺失关键点漂移。

**🔧 技术方法**

采用的技术包括：可穿戴追踪器（HTC Vive Ultimate Tracker）、实时IK预览、扩散策略（Diffusion Policy）、强化学习全身控制器、端执行器自适应奖励与可变速度增强、相对姿态跟踪接口。

**📊 数据集**

数据集为350条全身演示，覆盖7个不同环境和7个瓶子实例，包含图像观测与SE(3)轨迹，用于训练高层与低层网络。

**📈 对比分析**

与现有基于人机交互的TWIST2系统对比：HuMI在相同任务下采集了62条可用演示（vs. 28），接受率为96.7%（vs. 64.3%），平均时间缩短至30%；在10条测试中可在未见环境与物体上实现70%成功率。

**⚠️ 局限性**

局限性：依赖高质量视觉跟踪器，低层控制器尚未普适；实验仅在Unitree G1平台上验证，未测试跨平台适用性。

---

## 320. Jamming Attacks on the Random Access Channel in 5G and B5G Networks

**arXiv ID:** 2602.06634 | [PDF](https://arxiv.org/pdf/2602.06634v1)

**作者:** Wilfrid Azariah `[一作]` (National Taiwan University of Science and Technology), Binbin Chen `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 6639 | [OpenAlex ID](https://openalex.org/A5100687676)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文通过构建递归噪声阈值模型并在OAI/USRP实验平台上验证，研究了Msg1级RACH干扰对5G/B5G网络接入的影响。

**💡 创新点**

创新点在于提出将攻击者发射功率、周期性与基站阈值自适应更新权重关联的解析模型，并首次将其与实际测试结果进行对照，揭示低功率低占空比攻击同样能有效阻断合法UE接入。

**🔧 技术方法**

采用OpenAirInterface（OAI）实现的gNB/UE、USRP B210硬件、srsRAN基站、PRACH功率控制及相关信号处理算法，配合递归阈值更新与相关检测技术。

**📊 数据集**

使用真实5G NR配置参数（RACH参数、频率、带宽等）以及在实验平台收集的基站日志数据，未使用公开数据集。

**📈 对比分析**

通过对不同攻击周期T_a、阈值更新因子β、检测门限δ的实验设置，分别计算并对比理论阈值演化和UE成功接入概率，实验结果与模型高度吻合，表明低功率持续或周期性干扰可显著降低UE接入成功率。

**⚠️ 局限性**

局限性包括：攻击者仅能在单一RACH时隙发射单个前导，未考虑多RO多频段并行攻击；模型假设攻击功率恒定；未探究自适应防御机制，且仅在单一实验环境验证。

---

## 321. FairJudge: An Adaptive, Debiased, and Consistent LLM-as-a-Judge

**arXiv ID:** 2602.06625 | [PDF](https://arxiv.org/pdf/2602.06625v1)

**作者:** Bo Yang `[一作]` (Zhejiang University), Shijian Li `[通讯]` (Zhejiang University)

**通讯引用:** 7103 | [OpenAlex ID](https://openalex.org/A5103196339)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 FairJudge，一种将 LLM 的评判行为视为可学习的决策策略，解决了适应性、去偏差和跨模式一致性三大问题。

**💡 创新点**

创新点在于：① 将评判行为从固定函数转变为可优化的策略；② 构造高信息密度评判数据集 FairJudge‑16K；③ 采用三阶段课程学习（SFT → DPO → GRPO）来分别对齐评判规则、消除非语义偏差和强化跨模式一致性。

**🔧 技术方法**

使用的技术包括：监督微调（SFT）、直接偏好优化（DPO）以及组相对策略优化（GRPO）；对模型输入加入 rubric、reasoning 等上下文；以 Qwen3‑VL 8B 为基座并可扩展到 2B/4B/8B。

**📊 数据集**

主要数据集：FairJudge‑16K（训练集）、FairJudge‑Benchmark‑1K（评测集）；与公开基准 PandaLM、JudgeLM、FlexVL 以及多模态评测集 COCO、Cha.QA、InfographicsVQA 等进行对比。

**📈 对比分析**

与上述基准通过 Agreement、Precision、Recall、F1 四项指标进行对比。FairJudge‑8B 在三大评测集上均达到最高 Agreement（约 76‑78）和宏观 F1（约 66‑72），并在点式–对比式一致性上提升至 65% 以上，显著优于同规模的通用模型和先前的 judge 模型。

**⚠️ 局限性**

局限性：① 依赖人工审核的训练数据，扩展到更多领域/语言仍需验证；② 对极端非语义扰动的鲁棒性尚未充分评估；③ 目前实验仅覆盖 Qwen3‑VL 体系，未知在更大规模或不同模型体系中的迁移效果；④ 仍需在真实生产环境中评估长期稳定性与可解释性。

---

## 322. Force Generative Imitation Learning: Bridging Position Trajectory and Force Commands through Control Technique

**arXiv ID:** 2602.06620 | [PDF](https://arxiv.org/pdf/2602.06620v1)

**作者:** Hiroshi Sato `[一作]` (University of Tsukuba), Toshiaki Tsuji `[通讯]` (Saitama University)

**通讯引用:** 3264 | [OpenAlex ID](https://openalex.org/A5017159721)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于双向控制的力生成模仿学习框架，利用分层网络（上层预测未来位姿，下层无记忆MLP生成下一步状态与力命令）并加入PID误差补偿，实现机器人写字时的接触力控制。

**💡 创新点**

创新点在于：①将记忆性上层与无记忆下层分离，通过时间尺度分离避免反馈控制与内部记忆耦合导致的不稳定；②利用PID控制直接在神经网络输出中加入误差校正，实现对力命令的自适应生成；③实现了从直教数据直接推断力命令的闭环系统。

**🔧 技术方法**

使用技术包括：双向控制数据采集、分层神经网络（LSTM、MLP）、无记忆下层网络、PID误差补偿、伪微分滤波、离散时间积分/微分近似、反向传播训练。

**📊 数据集**

数据集为在CRANE‑X7机械臂上收集的写字轨迹，共70条实验（56条训练，14条验证），包括5种直线、2种圆形轨迹，且在白板高度0cm和2cm两种高度下收集。

**📈 对比分析**

方法通过与直教回放、传统LSTM模型及无PID/有PID MLP进行对比，评价指标为上层轨迹与实际绘制图形的IoU。结果显示有PID的MLP在多种字符和高度下均取得最高IoU，显著提升了轨迹跟踪精度和接触稳定性。

**⚠️ 局限性**

局限性包括：①低频下层无记忆假设可能不适用于更复杂任务；②仅在二维写字任务验证，三维接触任务的鲁棒性未知；③PID参数采用连续积分/微分估计，存在计算误差，离散控制实现待完善；④对系统稳定性和安全性的全面评估尚缺乏。

---

## 323. DAVE: Distribution-aware Attribution via ViT Gradient Decomposition

**arXiv ID:** 2602.06613 | [PDF](https://arxiv.org/pdf/2602.06613v1)

**作者:** Adam Wróbel `[一作]` (Jagiellonian University), Dawid Rymarczyk `[通讯]` (Jagiellonian University)

**通讯引用:** 476 | [OpenAlex ID](https://openalex.org/A5046065131)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DAVE 方法，利用视觉 Transformer 的梯度分解得到稳定、像素级别的归因图；

**💡 创新点**

创新点在于把梯度拆解为有效变换与输入变动两部分，利用等变性滤波和低通滤波消除结构化伪影，从而为 ViT 构建了理论上稳健的归因框架；

**🔧 技术方法**

主要技术包括梯度分解、Reynolds‑风格等变性滤波、低通（高斯）平滑、蒙特卡洛采样以及局部旋转与高斯扰动；

**📊 数据集**

实验使用 ImageNet‑1k 及其验证集边界框标注，对 DeiT、DeiT‑III、DINO ViT 以及 B‑cos 可解释网络进行评估；

**📈 对比分析**

与梯度、SmoothGrad、Integrated Gradients、LeGrad、AttnLRP、Chefer‑LRP 等基线比较，DAVE 在 GridPG、EnergyPG 定位指标和像素删除曲线上均取得最高或最优成绩，显示更好的空间精度和稳定性；

**⚠️ 局限性**

主要局限是需要额外的采样与旋转操作，导致计算开销高于单纯梯度方法；此外需要手动设定变换集合，若扰动幅度过大可能削弱细节。

---

## 324. Type-Based Unsourced Federated Learning With Client Self-Selection

**arXiv ID:** 2602.06601 | [PDF](https://arxiv.org/pdf/2602.06601v1)

**作者:** Kaan Okumus `[一作]` (Chalmers University of Technology), Shashi Raj Pandey `[通讯]` (Aalborg University)

**通讯引用:** 1627 | [OpenAlex ID](https://openalex.org/A5030621292)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于本地训练损失与中心广播阈值的客户端自选机制，并将其与 D-MIMO 无源 TUMA 聚合方案相结合，实现了无 CSI、无客户端身份泄露的联邦学习。

**💡 创新点**

创新点在于完全隐私保护的自选策略（不需上报损失或数据）以及将其嵌入无源 TUMA 框架，消除了传统 AirComp 对同步、预均衡和 CSI 的依赖。

**🔧 技术方法**

使用了 FedAvg、sigmoid 概率自选、向量量化与误差累积、D-MIMO 多天线网络、分区化 LSFC、以及多源 AMP 解码的 TUMA。

**📊 数据集**

实验采用 FMNIST 数据集，在多层感知机模型上进行训练。

**📈 对比分析**

与随机选择、PoC（服务器端基于损失的选择）以及 MD‑AirComp 进行对比。结果显示自选方案在误差通信下可匹配 PoC 性能，优于随机选择；在 D-MIMO 下比 MD‑AirComp 好，且在较短通信块长下实现更快收敛。

**⚠️ 局限性**

局限性包括阈值调节参数需经验调优；量化参数与块长权衡未做系统优化；在极低 SNR 或高时延环境下性能尚未评估；实验仅在二维网格部署下验证。

---

## 325. Memory-Conditioned Flow-Matching for Stable Autoregressive PDE Rollouts

**arXiv ID:** 2602.06689 | [PDF](https://arxiv.org/pdf/2602.06689v1)

**作者:** Victor Armegioiu `[一作]` `[通讯]` (ETH Zurich), Victor Armegioiu (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种记忆条件的流匹配（flow-matching）模型，用于在时间步进（autoregressive）中稳定地生成偏微分方程（PDE）解，特别是在从粗到细的多尺度环境下；

**💡 创新点**

核心创新在于将一个紧凑的在线记忆状态与生成网络的瓶颈特征融合，使得内部时间（flow-matching）的 Markov 过程在物理时间上变为非 Markov，从而显式捕捉历史依赖；

**🔧 技术方法**

采用了基于 Mori–Zwanzig 理论的记忆机理、Littlewood–Paley 频段分解、内部时间流匹配（rectified‑flow）框架，以及状态空间模型（SSM/S4）实现记忆更新；

**📊 数据集**

实验使用 Hugging Face 上的 PDEgym 数据集，包含二维可压缩欧拉方程的两类轨迹：Richtmyer–Meshkov 型（CE‑RM）和四象限 Riemann 问题（CRP/CRP2D）；

**📈 对比分析**

与容量相同的无记忆基线相比，记忆条件模型在 10–20 步长的自回归 roll‑out 中显著降低了最终时间相对 L² 误差，尤其在长周期、密集或稀疏的时间步长安排下提升 30–50%；

**⚠️ 局限性**

限制主要体现在：模型对极端高频细节的恢复仍有限；记忆结构需要手工设计（如维度、更新方式）；以及在不同 PDE 或更高维问题上可推广性仍待验证。

---

## 326. Same Engine, Multiple Gears: Parallelizing Fixpoint Iteration at Different Granularities (Extended Version)

**arXiv ID:** 2602.06680 | [PDF](https://arxiv.org/pdf/2602.06680v1)

**作者:** Ali Rasim Kocal `[一作]` (Technical University of Munich), Helmut Seidl `[通讯]` (Technical University of Munich)

**通讯引用:** 3938 | [OpenAlex ID](https://openalex.org/A5080676686)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出一种可参数化任务粒度的并行修正点迭代引擎，并在静态分析框架中实现了两种并行哲学：共享状态与独立副本+发布/订阅；

**💡 创新点**

创新点在于将任务粒度抽象为根未知数，支持多种粒度；同时提供两种并行实现（共享状态事务化访问与独立副本发布/订阅），实现对多线程分析的灵活并行化；

**🔧 技术方法**

使用技术包括 OCaml 实现、哈希表+CAS事务化访问、发布/订阅系统、top‑down solver 改进、任务池调度以及宽松/收窄等抽象解释机制；

**📊 数据集**

使用数据集为 SV‑Comp 并发套件、Concrat 以及四个常用 Unix 工具；

**📈 对比分析**

通过与单线程基线比较，在 1、2、4、8 个工作线程下测得平均加速 2‑3 倍，最大 10 倍；两种实现性能相近，均在多线程分析中显著降低时间；

**⚠️ 局限性**

局限在于结果对执行顺序敏感，可能出现精度波动；当粒度过细时并行开销增加；实验仅在 OCaml/Unix 环境下完成，尚未验证对非线程分析或其他语言的适用性。

---

## 327. Pruning at Initialisation through the lens of Graphon Limit: Convergence, Expressivity, and Generalisation

**arXiv ID:** 2602.06675 | [PDF](https://arxiv.org/pdf/2602.06675v1)

**作者:** Hoang Pham `[一作]` (University of Warwick), Long Tran-Thanh `[通讯]` (University of Warwick)

**通讯引用:** 3841 | [OpenAlex ID](https://openalex.org/A5012915897)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并建立了Pruning-at-Initialisation（PaI）方法在无限宽极限下的理论框架，证明稀疏掩码收敛到确定性双向图限（graphon），并基于此给出稀疏网络的表达力（Universal Approximation）和泛化（Graphon‑NTK）定理。

**💡 创新点**

①提出Factorised Saliency Model，将主流PaI准则统一为可因式分解的边重要性；②在此模型下严格证明PaI掩码在cut距离意义上收敛到确定性图限；③利用图限推导出稀疏网络的通用逼近能力和由路径密度决定的泛化边界。

**🔧 技术方法**

使用图限理论、cut度量、概率收敛技术、神经切线核（NTK）理论、泛化误差上界分析及路径密度分析。

**📊 数据集**

实验验证主要在CIFAR‑10二分类任务上进行，此外用随机输入或标准高斯输入做理论验证。

**📈 对比分析**

与随机剪枝（对应均匀图限）进行对比；在高稀疏度下，数据驱动方法（如SNIP、GraSP）因聚焦重要特征而在泛化误差与Kernel复杂度上显著优于随机剪枝；在高密度时两者差距缩小。

**⚠️ 局限性**

局限性：仅在无限宽极限下严格成立，有限宽校正未知；假设权重、梯度、噪声独立且分布连续；目前理论主要针对单隐藏层或浅层网络；缺少对阈值选择和超参数调优的细化指导。

---

## 328. compar:IA: The French Government's LLM arena to collect French-language human prompts and preference data

**arXiv ID:** 2602.06669 | [PDF](https://arxiv.org/pdf/2602.06669v1)

**作者:** Lucie Termignon `[一作]`, Elie Gavoty `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并运营了compar:IA公共LLM评测平台，收集了超过60万条自由式法语提示及25万条对话级偏好投票，公开发布三大数据集（对话、投票、反应）。

**💡 创新点**

通过开放式、无账户的盲对比界面实现大规模真实用户偏好收集，首次构建公开可用的法语LLM偏好基准，并提供能源消耗估算。

**🔧 技术方法**

采用FastAPI+SvelteKit后端、OpenRouter/Hugging Face推理接口、Ecologits能耗估算、LLM驱动的PII过滤以及Bradley‑Terry模型进行榜单生成。

**📊 数据集**

发布了comparsia-conversations、comparia-votes和comparia-reactions三份数据集，涵盖约89%法语提示、技术/教育类占比最高、并包含模型元数据与能耗信息。

**📈 对比分析**

使用盲对比和Bradley‑Terry排名构建模型榜单，展示了多模型间的用户偏好差异，数据规模为同类评测中的最大法语集合，但由于偏好采样与评价模式的限制，榜单仅具有探索性。

**⚠️ 局限性**

受限于无用户人口统计信息导致的代表性与自选偏差、专业/敏感任务缺失、模型系统提示不一致、量化推理差异、语言与风格偏见，以及对话级别的PII过滤导致的样本损失。

---

## 329. Not All Layers Need Tuning: Selective Layer Restoration Recovers Diversity

**arXiv ID:** 2602.06665 | [PDF](https://arxiv.org/pdf/2602.06665v1)

**作者:** Bowen Zhang `[一作]` (National University of Singapore), Harold Soh `[通讯]` (National University of Singapore)

**通讯引用:** 2495 | [OpenAlex ID](https://openalex.org/A5066073375)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的层级恢复方法SLR，恢复后训练模型的某一连续层区间为预训练权重，以恢复模式崩溃导致的生成多样性丢失；

**💡 创新点**

创新点在于：①证明模式崩溃可定位至特定层；②利用简易代理任务CRC快速定位最优恢复区间；③提出仅恢复连续层而不增加推理成本的方法；

**🔧 技术方法**

核心技术包括：权重空间合并、连续层恢复、代理任务CRC（基于字符约束的随机生成）、多任务评估（创意写作、开放式问答、多步推理）以及与温度/提示等解码/提示方法的组合；

**📊 数据集**

使用的数据集包括创意写作（poem、story、joke）共300个prompt；开放式问答CoverageQA共40个问题；推理任务GSM8K子集共各100题；预训练模型Llama‑3.1‑8B、Qwen‑2.5‑7B、Gemma‑2‑9B；

**📈 对比分析**

与后训练模型、模型Soup等基线对比，SLR在保持质量几乎不变的前提下，创作多样性提升约42.5%（平均），问答答案多样性提升112.7%/169.5%，推理Pass@k提升所有k；与温度/提示方法组合时仍能进一步提升；

**⚠️ 局限性**

局限包括：仅在7‑9B规模模型验证，未知大规模效果；恢复为整层，未探索更细粒度的子层或头部恢复；代理任务CRC可能无法完全覆盖复杂任务的多样性需求；

---

## 330. PlanViz: Evaluating Planning-Oriented Image Generation and Editing for Computer-Use Tasks

**arXiv ID:** 2602.06663 | [PDF](https://arxiv.org/pdf/2602.06663v1)

**作者:** Junxian Li `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22381 | [OpenAlex ID](https://openalex.org/A5074865219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PlanViz基准，用于评估统一多模态模型在电脑使用场景下的规划导向图像生成与编辑能力。

**💡 创新点**

创新点包括：①设计三类规划子任务（路径规划、工作流程绘制、网页/界面展示）；②构建高质量人工标注数据并通过多阶段质量控制；③提出任务自适应评价指标PlanScore，包含正确性、视觉质量与效率三维，并利用MLLM-as-judge自动化评估。

**🔧 技术方法**

采用统一多模态模型（如GPT‑Image‑1、Gemini3‑Pro‑Image等）、传统图像生成/编辑模型（如AnyEdit、OmniGen等）以及大型视觉语言模型（Qwen3‑VL‑235B）进行实验；使用Prompt风格转换、思考式生成等技术。

**📊 数据集**

数据集为PlanViz，包含约500张高质量源图像，分别为路程规划、工作流程、网页/UI展示三子任务；每子任务均有开放式与闭合式100多条查询，人工标注参考答案与关键点。

**📈 对比分析**

通过PlanScore对13款UMM和9款纯生成/编辑模型进行评估。结果显示：闭源模型（尤其GPT‑Image‑1）在生成任务中正确率可达0.8‑0.9，编辑任务仍低于0.6；开源UMM在生成与编辑上表现差距明显，编辑任务普遍低于0.4。整体显示模型在规划性编辑任务上存在显著瓶颈。

**⚠️ 局限性**

局限性包括：①编辑任务仍高度挑战，模型普遍难以满足空间与逻辑约束；②评估依赖MLLM‑as‑judge，可能受模型偏差影响；③数据规模相对有限，未覆盖更丰富的计算机使用场景；④未深入探讨模型内部机制导致的规划失效。

---

## 331. Multimodal Generative Retrieval Model with Staged Pretraining for Food Delivery on Meituan

**arXiv ID:** 2602.06654 | [PDF](https://arxiv.org/pdf/2602.06654v1)

**作者:** Boyu Chen `[一作]` (Beijing University of Posts and Telecommunications), Cheng Yang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 11068 | [OpenAlex ID](https://openalex.org/A5060417049)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SMGR——一种分阶段预训练的多模态生成检索模型，解决多模态检索中联合优化导致的模态失衡和单轮问题，并将高维特征量化为语义 ID 以减轻部署负担。

**💡 创新点**

创新点在于：①采用按阶段优化的预训练策略，先对文本、图像再对融合特征逐步对齐；②使用 RQ‑VAE 将多模态嵌入压缩为多层语义 ID；③设计生成与判别任务（doc2docid、因果预测）帮助模型理解并有效利用语义 ID。

**🔧 技术方法**

核心技术包括：双塔架构、对比学习、RQ‑VAE 残差量化、语义 ID（SIDs）生成与融合、doc2docid 生成任务、因果预测微调、FAISS ANN 检索。

**📊 数据集**

数据集：Meituan 食品配送平台真实交互数据（约 3200 万样本）和测试集（5.8M 候选项、2M 点击），按城市流行度划分并构造高频查询子集。

**📈 对比分析**

通过与 Qwen3‑DualTower、Joint‑Que2search、TIGER、MTIGER 等基线在 R@K、N@K 上进行十次重复实验，SMGR 在 R@5、R@10、R@20 分别提升约 3.8%/2.6%/2.2%，N@5/N@10/N@20 提升约 5.1%/4.2%/2.1%；在线 A/B 测试提升收入 +1.12%、CTR +1.02%。

**⚠️ 局限性**

局限性：目前查询侧缺乏用户历史信息；语义 ID 生成对训练顺序敏感；对序列数据的支持有限，未来需要进一步扩展。

---

## 332. Same Answer, Different Representations: Hidden instability in VLMs

**arXiv ID:** 2602.06652 | [PDF](https://arxiv.org/pdf/2602.06652v1)

**作者:** Farooq Ahmad Wani `[一作]` (Sapienza University of Rome), Pasquale Minervini `[通讯]` (University of Edinburgh)

**通讯引用:** 4343 | [OpenAlex ID](https://openalex.org/A5019106673)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于表示漂移、频谱敏感性和结构平滑度的评估框架，系统评估VLM在自然扰动下的内部稳定性与输出稳健性；

**💡 创新点**

首次将内部表征漂移与频域分析结合，用多维度指标揭示VLM鲁棒性与模型规模、任务类型之间的非线性关系；

**🔧 技术方法**

利用嵌入相似度、L2距离、Dirichlet能量、Cohen’s d、频谱噪声注入、频域剪枝及PGD对抗攻击等技术；

**📊 数据集**

在SEEDBench、MMMU和POPE三个多模态基准上，使用多种几何、裁剪、缩放、旋转和文本叠加扰动进行评测；

**📈 对比分析**

与传统仅基于标签一致性的鲁棒性评估对比，发现模型准确率与鲁棒性不呈正相关；大模型虽准确率提高，但在文本叠加等扰动下flip率与表征漂移均升高；

**⚠️ 局限性**

局限性包括：仅测试了部分VLM架构与基准；扰动范围为自然变换，未覆盖极端对抗；评估指标依赖于选定的嵌入层，可能忽略早期融合过程；

---

## 333. Temperature Scaling Attack Disrupting Model Confidence in Federated Learning

**arXiv ID:** 2602.06638 | [PDF](https://arxiv.org/pdf/2602.06638v1)

**作者:** Kichang Lee `[一作]` (Yonsei University), JeongGil Ko `[通讯]` (Yonsei University)

**通讯引用:** 3719 | [OpenAlex ID](https://openalex.org/A5022122076)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在联邦学习框架中，提出一种训练时的温度缩放攻击，通过调整本地学习率与温度耦合，破坏模型的置信度校准而不显著影响准确率。

**💡 创新点**

创新点在于将置信度校准视为独立攻击目标，设计了温度-学习率耦合的攻击机制，并给出在非IID环境下的收敛理论与验证。

**🔧 技术方法**

采用温度缩放、有效步长β=η/τ耦合、FedAvg聚合，并对比噪声注入、标签翻转等传统投毒手段，评估鲁棒聚合与后置校准等防御。

**📊 数据集**

实验数据集包括MNIST+MLP、CIFAR‑10+CNN、CIFAR‑100+ResNet‑18，以及医疗、自动驾驶与语言模型等案例数据。

**📈 对比分析**

与常规投毒、后置校准和多种鲁棒聚合方法对比，实验显示温度攻击可将ECE从≈3%提升至≈5–38%，改变sECE符号，准确率仅变动<2%，且在鲁棒聚合和校准下仍保持有效。

**⚠️ 局限性**

局限性在于需攻击者能控制本地训练过程，难以直接应用于仅能操纵数据或标签的弱攻击者，且攻击效果受温度范围限制，过高或过低温度会导致收敛或准确率下降。

---

## 334. Estimating Exam Item Difficulty with LLMs: A Benchmark on Brazil's ENEM Corpus

**arXiv ID:** 2602.06631 | [PDF](https://arxiv.org/pdf/2602.06631v1)

**作者:** Thiago Brant `[一作]` (Luxembourg Institute of Socio-Economic Research), Jun Pang `[通讯]` (Department of Computer Science, University of Luxembourg)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对10种LLM在巴西高考ENEM题目难度估计的准确性与排名进行评估，探讨提示设计、模态影响和人口背景适应性。

**💡 创新点**

首次将LLM与官方IRT难度参数在大规模真实试卷上直接对比，并提出“先评估后生成”管线；同时发现LLM难度预测在模态、提示与人口背景上表现出显著偏差与有限适应性。

**🔧 技术方法**

使用提示工程（8种模板）、后置线性校准、基于IRT的难度映射、极差分析与ANOVA检验，以及对国家背景词的“可塑性”测量。

**📊 数据集**

ENEM 2017‑2022年公开试卷1031道多项选择题，包含文本与图表信息，难度标签来自官方3PL IRT模型。

**📈 对比分析**

评估指标为RMSE、Spearmanρ、MAE、EM、W1A；最优模型在ρ≈0.45、RMSE≈1.2时实现，提示与模态差异导致ρ在0.1–0.45、RMSE在1.1–2.5之间波动；校准后RMSE可提升0.5–1.0。

**⚠️ 局限性**

局限性包括：仅使用文本转写的视觉内容，无法充分捕捉空间信息；模型对人口背景的适应性弱且噪声大；校准需要周期性更新；实验仅在ENEM上验证，未覆盖其他考试或语言。

---

## 335. IE-RAP: An Intelligence and Efficient Reader Anti-Collision Protocol for Dense RFID Networks

**arXiv ID:** 2602.06626 | [PDF](https://arxiv.org/pdf/2602.06626v1)

**作者:** Hadiseh Rezaei `[一作]` (University of Portsmouth), Mohammad Shojafar `[通讯]` (University of Surrey)

**通讯引用:** 9174 | [OpenAlex ID](https://openalex.org/A5022893278)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出并实现了IE‑RAP协议，通过集中式调度结合TDMA和FDMA，加入信息共享阶段，降低读者碰撞、减少重复读取，提升吞吐量、降低等待时间与能耗。

**💡 创新点**

创新点在于：①在每轮结束后读者共享已读标签ID和成功读数，避免重复读取；②使用SIFT概率分布动态选择时隙；③基于RSSI的距离估算与动态频道/时隙分配；④在多读者、多通道密集环境下实现更高吞吐与更低能耗。

**🔧 技术方法**

技术手段包括：集中式服务器调度、TDMA+FDMA混合访问、SIFT概率分布函数、RSSI/路径损耗估算距离、信息共享阶段（ISP）以及R2RIS仿真器。

**📊 数据集**

数据集：利用R2RIS仿真器生成随机布置的100–400个读者（2.3 W功率、读距10 m、干扰范围1000 m）在1000 m²区域的仿真场景；未使用真实标签数据。

**📈 对比分析**

通过仿真与NFRA、FRCA1/2、GDRA、DMRCP等现有协议在吞吐量、平均等待时间、能耗等指标对比，IE‑RAP在所有场景下吞吐量提升约26%、平均等待时间降低约74%、能耗下降约52%，并在单/多通道模式下均表现优异。

**⚠️ 局限性**

局限性包括：①依赖集中式服务器和同步，缺乏分布式可扩展性；②对RSSI测量的可靠性要求高，距离估算误差可能影响性能；③未在真实硬件环境中验证，实现的鲁棒性与实时性待进一步评估。

---

## 336. CauCLIP: Bridging the Sim-to-Real Gap in Surgical Video Understanding via Causality-Inspired Vision-Language Modeling

**arXiv ID:** 2602.06619 | [PDF](https://arxiv.org/pdf/2602.06619v1)

**作者:** Yuxin He `[一作]` (Southeast University), Cheng Xue `[通讯]` (Southeast University)

**通讯引用:** 1302 | [OpenAlex ID](https://openalex.org/A5101889471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于CLIP的因果驱动框架CauCLIP，用于在无目标域数据情况下进行手术阶段识别。

**💡 创新点**

创新点在于结合频域幅度混合增强与因果抑制损失，聚焦因果特征并抑制非因果视觉噪声，实现更强的域泛化。

**🔧 技术方法**

技术包括CLIP视觉-文本对齐、频域幅度混合增强、相似性抑制损失、联合KL损失等。

**📊 数据集**

使用SurgVisDom仿真到真实猪术视频的硬适配基准数据集。

**📈 对比分析**

与ResNet‑50、ViT‑B/16、SDA‑CLIP等基线对比，CauCLIP在weighted F1、unweighted F1、global F1和balanced accuracy上均领先，最高balanced accuracy达到0.651。

**⚠️ 局限性**

局限在于仅在VR训练下评估，频域增强参数选择敏感，对多任务或更大域差异的泛化仍需进一步验证。

---

## 337. Confundo: Learning to Generate Robust Poison for Practical RAG Systems

**arXiv ID:** 2602.06616 | [PDF](https://arxiv.org/pdf/2602.06616v1)

**作者:** Haoyang Hu `[一作]` (University of Hong Kong), Ka-Ho Chow `[通讯]` (University of Hong Kong)

**通讯引用:** 21236 | [OpenAlex ID](https://openalex.org/A5100343991)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种学习式毒化框架，针对实际部署的检索增强生成（RAG）系统生成高效、鲁棒且隐蔽的毒化文本，支持多种攻击目标（事实错误、意见偏见、幻觉诱导）。

**💡 创新点**

创新点在于：①把毒化视为优化问题，联合设计检索、生成、词汇鲁棒、碎片化与隐蔽性奖励；②在单一框架内实现多目标攻击；③通过模拟文档分块与检索策略提升在真实RAG流水线中的稳健性。

**🔧 技术方法**

核心技术包括：LLM（如Qwen3、Llama3）微调，基于BM25与多模嵌入模型的检索奖励；语义生成奖励（使用代理LLM评估目标答案出现与否）；对抗性数据增强与随机分块模拟碎片化；句法流畅性奖励（Perplexity）；以及群体相对策略优化（GRPO）进行模型训练。

**📊 数据集**

实验数据集涵盖：Harry Potter（主数据集）、NewsQA、OCRBench（事实攻击），PROCON（意见攻击），RAGTruth（幻觉攻击）以及自定义网页内容，进一步验证跨数据集泛化。

**📈 对比分析**

与PoisonedRAG、AuthChain、PR‑Attack、Joint‑GCG等专用攻击在多种RAG配置（不同检索器、检索阈值K、生成器）下对比，平均攻击成功率提升：事实攻击↑1.68×，偏见攻击↑6×，幻觉攻击↑1.78×；在防御（Perplexity检测、rerank、paraphrasing）下仍保持高成功率（≥70%）。

**⚠️ 局限性**

局限性：仍易受极端文档分块比例、检索重排序或强烈paraphrase防御影响；训练需对目标查询或攻击目标进行采样，且在多语言或极大文档规模上未作系统评估；目前仅针对文本类RAG，非图像/多模态场景需进一步扩展。

---

## 338. Green Optimization: Energy-aware Design of Metaheuristics by Using Machine Learning Surrogates to Cope with Real Problems

**arXiv ID:** 2602.06610 | [PDF](https://arxiv.org/pdf/2602.06610v1)

**作者:** Tomohiro Harada `[一作]` (Saitama University), Gabriel Luque `[通讯]` (University of Malaga)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文探究了将深度神经网络代理模型嵌入粒子群和遗传算法中，以减少交通灯调度优化中的能源消耗、执行时间和内存占用。

**💡 创新点**

创新点在于首次系统评估代理模型在能源效率、时间和精度上的综合影响，并证明大规模训练集可通过稀疏化降低推理能耗。

**🔧 技术方法**

采用TensorFlow/Keras构建的全连接多层感知机代理模型，并在PSO、GA两种元启发式框架中实现静态预训练与动态再训练两种策略。

**📊 数据集**

使用了三城交通灯调度实例：巴塞尔（56交叉口）、斯德哥尔摩（75交叉口）和巴黎（70交叉口），每个实例均包含数百个阶段、数百辆车和数千秒的仿真时长。

**📈 对比分析**

通过对比使用代理与不使用代理的算法，实验显示代理可将能耗降低约98%，执行时间降低约98%，内存使用降低约99%；预训练小规模数据集最快，但再训练可在较低能耗下保持更高预测精度。

**⚠️ 局限性**

限制在于仅评估了全连接网络和交通灯调度问题，未考察更复杂网络结构或其他领域，且实验仅在单一硬件平台上测量能耗，未覆盖多平台泛化。

---

## 339. Trust Regions Sell, But Who's Buying? Overlap Geometry as an Alternative Trust Region for Policy Optimization

**arXiv ID:** 2602.06627 | [PDF](https://arxiv.org/pdf/2602.06627v1)

**作者:** Gaurish Trivedi `[一作]` (Birla Institute of Technology and Science), Jagat Sesh Challa `[通讯]` (Birla Institute of Technology and Science)

**通讯引用:** 265 | [OpenAlex ID](https://openalex.org/A5044760764)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

论文的具体内容未提供，因此无法总结做了什么。

**💡 创新点**

由于缺乏详细信息，无法确定创新点。

**🔧 技术方法**

未提供使用的技术信息。

**📊 数据集**

未提及使用的数据集。

**📈 对比分析**

没有比较的方法和性能数据。

**⚠️ 局限性**

由于信息不足，无法识别限制因素。

---

## 340. Explaining Grokking in Transformers through the Lens of Inductive Bias

**arXiv ID:** 2602.06702 | [PDF](https://arxiv.org/pdf/2602.06702v1)

**作者:** Jaisidh Singh `[一作]` (University of Tübingen), Antonio Orvieto `[通讯]` (MPI-IS Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一层Transformer上通过不同层归一化位置和优化设置研究grokkng现象，揭示结构与优化的诱导偏差如何影响延迟泛化

**💡 创新点**

首次系统分析LayerNorm位置对shortcut-learning、注意力熵与特征压缩的影响，并指出读出尺度作为“懒惰”控制因学习率、权重衰减与softmax温度混淆，显示grokkng是特征持续演化的结果，而非简单的懒惰-丰富转换

**🔧 技术方法**

采用一层Transformer架构，改造LayerNorm位置、使用交叉熵损失与AdamW优化器、计算傅里叶模式浓度、注意力熵、特征有效秩与幂律指数等指标

**📊 数据集**

使用模块化加法（modular addition）这一toy任务（p=113），数据量极小，主要用于观察训练与测试损失曲线

**📈 对比分析**

通过对不同LN配置、学习率、权重衰减、读出尺度进行多种实验，比较训练损失、测试损失下降速度、特征压缩度与傅里叶模式出现时间，结果表明某些LN位置（如仅在MLP输入归一化）能显著加快grokkng并提升泛化性能

**⚠️ 局限性**

实验仅限于单层Transformer与模块化加法任务，使用自适应优化器，未验证在更深网络或其他任务上的推广性

---

## 341. Diffeomorphism-Equivariant Neural Networks

**arXiv ID:** 2602.06695 | [PDF](https://arxiv.org/pdf/2602.06695v1)

**作者:** Josephine Elisabeth Oettinger `[一作]` (University of Luebeck), Carola-Bibiane Schönlieb `[通讯]` (University of Cambridge)

**通讯引用:** 15676 | [OpenAlex ID](https://openalex.org/A5004024852)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于能量正则化的canonicalization方法，将预训练的神经网络扩展为对光滑变换（微分同胚）等变的形式。

**💡 创新点**

创新点在于把Lie群理论的能量canonicalization推广到无限维的微分同胚群，并采用SVF参数化和梯度优化实现可微分配准。

**🔧 技术方法**

核心技术包括SVF生成的微分同胚、变分自编码器与对抗判别器的能量评估、梯度正则化以及SIREN+缩放-平方的流场求解。

**📊 数据集**

实验使用了三组数据集：合成嵌套方块分割、胸部X光肺部分割以及MNIST的基数分类。

**📈 对比分析**

与无正则化的原网络和全数据增强网络对比，DiffeoNN在分割任务上达到约0.957的IoU（比原网络高、略低于增强网络）且异常值更少；在分类任务上准确率与增强相当（0.82）。

**⚠️ 局限性**

主要局限在于等变性只能是近似的；正则化与梯度求解导致计算开销和可能的优化不收敛；缺乏理论稳定性与泛化保证。

---

## 342. Calibrating Generative AI to Produce Realistic Essays for Data Augmentation

**arXiv ID:** 2602.06772 | [PDF](https://arxiv.org/pdf/2602.06772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 343. Makespan Minimization in Split Learning: From Theory to Practice

**arXiv ID:** 2602.06693 | [PDF](https://arxiv.org/pdf/2602.06693v1)

**作者:** Robert Ganian `[一作]` (TU Wien), Dimitra Tsigkari `[通讯]` (Telefónica Scientific Research)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究分裂学习（Split Learning）中，为最小化批量训练的最迟完成时间（makespan）而设计客户端–助手分配与调度问题。

**💡 创新点**

创新点在于证明该问题在受限实例下仍强NP‑难且无PTAS，提出 5‑近似算法，并在此基础上设计等分分配启发式 EquiD，实验表明该启发式在多种场景下优于已有方法。

**🔧 技术方法**

采用了复杂度归约、整数规划与线性规划迭代舍入、贪心排序调度、基于 GAPcc 的 2‑近似子算法以及 Gurobi 等求解器来实现算法与实验。

**📊 数据集**

实验使用 ResNet101 与 VGG19 在 CIFAR‑10 与 MNIST 上的真实训练时间与内存测量数据，并辅以合成数据进行不同异质性水平的评估。

**📈 对比分析**

与最优解、B‑G、ED‑FCFS 进行比较，EquiD 的子最优率最高 19.77%，平均约 7.79%，且在大多数场景中使 makespan 缩短约 30–35% 以上。

**⚠️ 局限性**

限制在于仅考虑已预定的切分层，不优化切点、通信速率或模型收敛等因素；此外该问题仍不可多项式逼近，且对极大规模实例求解仍受限。

---

## 344. Visual Word Sense Disambiguation with CLIP through Dual-Channel Text Prompting and Image Augmentations

**arXiv ID:** 2602.06799 | [PDF](https://arxiv.org/pdf/2602.06799v1)

**作者:** Shamik Bhattacharya `[一作]` (Bredesen Center for Interdisciplinary Research and Graduate Education), Edmon Begoli `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 7323 | [OpenAlex ID](https://openalex.org/A5014968146)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于CLIP的可解释视觉词义消歧框架，通过双通道语义与照片提示、WordNet同义词融合以及图像多视角增强，提升了词义歧义的判定能力。

**💡 创新点**

创新点在于将双通道提示与图像增强统一嵌入CLIP共享空间，并利用低成本的提示与增强实现可解释的消歧方法，避免使用外部大型模型或复杂融合机制。

**🔧 技术方法**

技术手段包括CLIP文本/视觉编码、双通道（语义+照片）提示、WordNet同义词扩展、图像测试时增强（多视角裁剪、色彩抖动等）、余弦相似度检索、贝叶斯超参数搜索及与基线对比分析。

**📊 数据集**

使用SemEval‑2023 Visual Word Sense Disambiguation (VWSD) 数据集进行训练、验证与测试。

**📈 对比分析**

与Vanilla CLIP、BERT+BLIP等基线对比，最终在测试集上达到MRR 0.7590、Hit@1 0.6220，虽然低于顶尖模型（≈84%）但较基线提升显著，并通过消融展示提示与增强的有效性。

**⚠️ 局限性**

主要局限包括与最优系统的性能差距、WordNet同义词或多语言翻译引入噪声、图像增强计算开销大且收益有限，以及对CLIP预训练分布的依赖，限制了模型在更大词汇或复杂场景下的泛化。

---

## 345. Robust Online Learning

**arXiv ID:** 2602.06775 | [PDF](https://arxiv.org/pdf/2602.06775v1)

**作者:** Sajad Ashkezari `[一作]` `[通讯]` (University of Waterloo), Sajad Ashkezari (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究鲁棒在线学习框架，定义新的组合维度（对抗 Littlestone 维度），证明该维度决定可实现的错误/损失上界，并给出多分类、未知扰动集以及无约束（agnostic）情形的误差与对数损失界。

**💡 创新点**

创新点在于：①首次把在线鲁棒学习与经典 Littlestone 维度结合，给出一个简单易算的维度；②证明该维度对 realizable 与 agnostic 情形分别给出最优误差/损失界；③将鲁棒学习扩展到多分类、未知扰动集、以及通过专家学习实现更强的鲁棒性。

**🔧 技术方法**

采用组合分析、版本空间递归、对抗树（对抗 Littlestone 树）构造、以及专家学习（Prediction with Expert Advice）和在线投票/投影方法；使用误差递减、树深度上界和冗余信息消除技巧。

**📊 数据集**

无实验数据集；本工作为理论性研究，未进行数值实验或使用公开数据集。

**📈 对比分析**

与传统在线学习（经典 Littlestone 维度）及鲁棒 PAC 学习的已知结果进行理论对比；误差上界为维度 L，损失/损失界为 O~(√(L·T))；在未知扰动集时误差上界为 (L*+1)log|G|，对数损失上界为 O(√(T·(L*·logT + log|G|)))。

**⚠️ 局限性**

限制包括：①理论结果与实验验证缺失；②对抗扰动集若无限大，仍需进一步分析；③假设学习者能获取完整的干净输入与标签；④未考虑带隙（bandit）反馈；⑤对鲁棒回归任务未给出对应理论；⑥在 agnostic 情形中存在 √logT 的下界与上界差距。

---

## 346. Semantically Labelled Automata for Multi-Task Reinforcement Learning with LTL Instructions

**arXiv ID:** 2602.06746 | [PDF](https://arxiv.org/pdf/2602.06746v1)

**作者:** Alessandro Abate `[一作]` (University of Oxford), Christoph Weinhuber `[通讯]` (Technical University of Munich)

**通讯引用:** 39 | [OpenAlex ID](https://openalex.org/A5044787611)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究多任务强化学习，目标是训练单一策略能够满足任意给定的线性时序逻辑（LTL）任务，并提出 SemLTL 方法实现这一目标。

**💡 创新点**

创新点在于利用最近的 LTL‑to‑automata 翻译产生的语义标记 LDBA 状态，将其嵌入为结构化特征；该嵌入支持完整 LTL、可在线按需构造自动机，仅构造任务所需的部分状态，显著降低计算成本。

**🔧 技术方法**

技术手段包括：语义 LTL‑to‑LDBA 翻译、公式真度与命题注意力特征构造、图神经网络（GNN）对公式的编码、基于这些嵌入的多头神经网络策略（同时处理 MDP 动作与 ε‑动作），以及 Proximal Policy Optimization（PPO）训练与任务难度递增的学习曲线。

**📊 数据集**

实验使用的环境与数据集有：LetterWorld（7×7 网格）、ZoneEnv（机器人导航，原版与扩展版）、以及多组参数化任务族（Local‑Safety、Global‑Safety、Finite‑Reactive、Complex‑Patrol、Reach‑Stay、Always‑Reactive），这些任务覆盖从安全约束到复合循环的多种 LTL 结构。

**📈 对比分析**

与基线方法 LTL2Action（仅公式进化）和 DeepLTL（基于完整自动机的可达性序列）比较，SemLTL 在绝大多数任务上实现了更高的成功率/平均接受次数；在复杂任务中 DeepLTL 经常因序列枚举耗时超过 600 秒而失败，而 SemLTL 仅构造少量自动机状态即可获得优秀表现，显示出更好的可扩展性和效率。

**⚠️ 局限性**

局限性包括：仍假设环境标签函数已知，实验仅在仿真环境中进行；在极大规模 LTL 公式下自动机构造仍需进一步优化；对真实机器人系统的鲁棒性和部署效果尚未验证。

---

## 347. Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks (KANs)

**arXiv ID:** 2602.06737 | [PDF](https://arxiv.org/pdf/2602.06737v1)

**作者:** Noah Schwartz `[一作]` (University of Colorado Boulder), Susmit Jha `[通讯]` (SRI International)

**通讯引用:** 2856 | [OpenAlex ID](https://openalex.org/A5035902535)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过为Kolmogorov‑Arnold网络（KAN）构造最优分段线性抽象，并将抽象优化建模为多选背包问题，从而实现高效的范围验证。

**💡 创新点**

将每个单元的分段数分配问题转化为多选背包优化，利用动态规划求最优抽象，显著降低MILP求解规模；同时给出误差传播与整体误差界定的理论。

**🔧 技术方法**

动态规划、分段线性逼近、误差传播分析、MILP编码（Gurobi/CPLEX/MOSEK求解器），以及在背包问题上使用多选背包DP。

**📊 数据集**

包括5个函数学习任务（Bessel、乘积、指数等）、PINN热方程、人体义肢预测、天气预测等，共65个网络。

**📈 对比分析**

与传统KAN的“vanilla”抽象以及基于MLP的LiRPA和Gurobi verifiers比较，优化后的KAN在59/65任务中得到更窄的输出区间，速度在大规模网络上略慢但可通过预处理/懒抽象改进。

**⚠️ 局限性**

动态规划求解的预处理开销较大，导致大规模KAN的验证时间比MLP慢；对误差离散化与Δ选择需要经验；对极端非线性函数的分段数限制仍影响精度。

---

## 348. $f$-Differential Privacy Filters: Validity and Approximate Solutions

**arXiv ID:** 2602.06756 | [PDF](https://arxiv.org/pdf/2602.06756v1)

**作者:** Long Tran `[一作]` (University of Helsinki), Antti Honkela `[通讯]` (University of Helsinki)

**通讯引用:** 19551 | [OpenAlex ID](https://openalex.org/A5016681267)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了在完全自适应（fully adaptive）组合下，基于trade‑off函数的差分隐私（f‑DP）能否通过自然过滤器（即按trade‑off函数积累并在阈值处停止）来实现隐私保证，并给出了必要与充分条件；同时构造了适用于Poisson子采样高斯机制的近似GDP过滤器，利用自适应中心极限定理和PLRVs的矩逼近实现更紧的隐私保证。

**💡 创新点**

① 证明自然f‑DP过滤器在一般情况下无效；② 通过Blackwell链与Δ‑散度建立了过滤器有效性的完整理论判据；③ 推导了自适应CLT，首次在完全自适应框架下给出PLRV的偏差上界；④ 构造了在q≪1或q≈1两种极端采样率下优于RDP的近似GDP过滤器。

**🔧 技术方法**

trade‑off函数组合、Blackwell排序、Δ‑散度、马尔科夫-贝里-埃森斯（Berry–Esseen）界的马尔可夫链扩展、PLRV矩逼近、子采样高斯机制的精确trade‑off函数、近似GDP过滤算法。

**📊 数据集**

主要以理论推导为主，数值验证使用了Poisson子采样高斯机制（自适应σ）模拟的DP‑SGD实验（T=3650，q=0.01），并未使用公开真实数据集。

**📈 对比分析**

与完全自适应RDP过滤器进行比较，采用相同的输出次数；在q=0.01的低采样率下，近似GDP过滤器在相同δ下的ε约小10%，实现了更紧的隐私保证。

**⚠️ 局限性**

限制：近似GDP过滤器只适用于极端采样率（q≪1或q≈1）并需要T足够大；对一般f‑DP的完全自适应组合仍无法给出完整正交过滤器；对实际应用的实现复杂度较高，需预先设定采样率/噪声区间。

---

## 349. Machine Learning for Detection and Severity Estimation of Sweetpotato Weevil Damage in Field and Lab Conditions

**arXiv ID:** 2602.06786 | [PDF](https://arxiv.org/pdf/2602.06786v1)

**作者:** Doreen M. Chelangat `[一作]` (National Crops Resources Research Institute), Joyce Nakatumba-Nabende `[通讯]` (Makerere University)

**通讯引用:** 1090 | [OpenAlex ID](https://openalex.org/A5059644601)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了基于计算机视觉的甜薯象虫损伤检测与严重度估计，分别在田间与实验室环境中实现自动评估。

**💡 创新点**

创新点在于：①结合双输入ResNet‑50实现田间严重度分类；②引入两阶段实验室检测管线（根分割+SAHI切块+YOLO12）精确检测极小的食槽；③提供可公开的数据集与轻量模型，适配边缘设备。

**🔧 技术方法**

使用YOLO11/12、SAHI、ResNet‑50、VGG16等深度学习模型，辅以数据增强与分割技术。

**📊 数据集**

数据集包括四个乌干达田间试验（共636棵根图）以及实验室对照（168张带标注的根图），随后划分为训练/验证/测试。

**📈 对比分析**

与传统人工评分对比，田间分类模型测试准确率71.43%，实验室检测模型YOLO12平均精度77.7%；验证集误差表明模型仍存在过拟合与邻类混淆。

**⚠️ 局限性**

主要局限包括：田间数据不平衡导致泛化差、实验室环境受控限制实际应用、检测管线计算量大、未加入多模态信息、模型对近似等级区分能力不足。

---

## 350. Hierarchical Activity Recognition and Captioning from Long-Form Audio

**arXiv ID:** 2602.06765 | [PDF](https://arxiv.org/pdf/2602.06765v1)

**作者:** Peng Zhang `[一作]` (University of Surrey), Wenwu Wang `[通讯]` (University of Surrey)

**通讯引用:** 8731 | [OpenAlex ID](https://openalex.org/A5100676721)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了 MultiAct 数据集并提出统一的层次化模型，用于长音频中的多级活动识别与生成。

**💡 创新点**

创新点在于提供从活动到事件三级层次的长音频注释、双重文本描述以及统一的层次化编码解码框架。

**🔧 技术方法**

采用 SlowFast 音频特征提取、BiGRU+attention 或 Cross‑attention、Conformer+CTC、ActionFormer、BART 解码器等技术。

**📊 数据集**

使用从 EPIC‑SOUNDS 派生的 9 小时长厨房音频数据集 MultiAct。

**📈 对比分析**

通过与基线（ASF+线性、ASF+BiGRU+attention、ASF+Cross‑attention、ActionFormer、Conformer+CTC、规则式 BART）对比，层次化模型在活动分类 83.3% 及子活动分类 66.7% 取得最佳，检测 AP 最高 44.3%，序列预测 AER 最小 75.9%，字幕生成 CIDEr 最高 24.0%；总体性能表现优于基线但仍低于短视频任务。

**⚠️ 局限性**

限制包括样本量有限、长范围依赖建模困难、边界定位精度不足、LLM 生成注释的错误和主观性。

---

## 351. Constraint Manifold Exploration for Efficient Continuous Coverage Estimation

**arXiv ID:** 2602.06749 | [PDF](https://arxiv.org/pdf/2602.06749v1)

**作者:** Robert Wilbrandt `[一作]` (FZI Research Center for Information Technology), Rüdiger Dillmann `[通讯]` (FZI Research Center for Information Technology)

**通讯引用:** 12351 | [OpenAlex ID](https://openalex.org/A5112552331)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于隐式约束流形采样的工业机器人连续表面覆盖估计方法。

**💡 创新点**

创新点在于将工具位置与方向约束嵌入扩展配置空间并利用流形探索实现连续可达性评估，而非仅离散覆盖。

**🔧 技术方法**

使用IMACS框架、RRT与KPIECE采样、SE(3)卷积简化、Open Motion Planning Library与OpenCASCADE进行碰撞检测与表面建模。

**📊 数据集**

采用三台工业机器人（UR5e、Kuka KR10、Franka Research 3）与三种 CAD 表面（平滑无障碍、曲面+障碍、8×8迷宫）进行实验。

**📈 对比分析**

通过与基准10M样本的完全覆盖率对比，RRT采样在简单场景下较快但在复杂/狭窄通道中效率低，偏置采样在曲面和迷宫中实现更高覆盖率且样本效率更佳，二者均能在数秒内达到约80%覆盖。

**⚠️ 局限性**

限制在于只能处理 C^2 平滑表面，未能处理不连续或高维自由度机器人，且采样时需手工调参。

---

## 352. Taipan: A Query-free Transfer-based Multiple Sensitive Attribute Inference Attack Solely from Publicly Released Graphs

**arXiv ID:** 2602.06700 | [PDF](https://arxiv.org/pdf/2602.06700v1)

**作者:** Ying Song `[一作]` (University of Pittsburgh), Balaji Palanisamy `[通讯]` (University of Pittsburgh)

**通讯引用:** 2068 | [OpenAlex ID](https://openalex.org/A5103337534)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种无查询、基于公开图的多敏感属性推断攻击框架 Taipan，能够在目标图上同时推断多重敏感信息。

**💡 创新点**

创新点在于首次将多任务迁移学习与图神经网络结合，构建层次化攻击知识路由与基于提示的原型细化机制，以解决多属性互相干扰和域漂移问题。

**🔧 技术方法**

核心技术包括多门混合专家（MMoE）结构、预设任务指令向量（pretext tokens）、自监督伪标签调优与自适应原型对齐，以及对图的 PCA 统一特征空间投影。

**📊 数据集**

在四个真实世界图数据集（German、Credit、Pokec-n、Pokec-z）上进行实验，评估了同分布、异分布和任务偏移三种辅助图场景。

**📈 对比分析**

与随机猜测、单任务训练和仅预训练的基线相比，Taipan在平均 AUC、F1、子集准确率和语义一致性等指标上均表现优异，尤其在子集准确率和多任务均衡性（TDA/TDF）方面显著提升。

**⚠️ 局限性**

局限性包括对极端域迁移的鲁棒性不足、对极大规模图的可扩展性未彻底验证，以及对多分类或回归型敏感属性的泛化尚未深入探讨。

---

## 353. FaA-CAF: Modular Single-RF-Chain Near-Field mmWave Sensing via Clip-On Antenna Fabric

**arXiv ID:** 2602.06767 | [PDF](https://arxiv.org/pdf/2602.06767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 354. Crowd-FM: Learned Optimal Selection of Conditional Flow Matching-generated Trajectories for Crowd Navigation

**arXiv ID:** 2602.06698 | [PDF](https://arxiv.org/pdf/2602.06698v1)

**作者:** Antareep Singha `[一作]` (Nanyang Technological University), K. Madhava Krishna `[通讯]` (Indian Institute of Information Technology Hyderabad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 Crowd-FM，一种基于 Conditional Flow Matching 的本地规划框架，用于在拥挤人群中快速生成安全且具有人类相似性的轨迹。

**💡 创新点**

创新点在于：1) 直接在 Bernstein 多项式控制点空间学习多模态轨迹分布，避免了 VQ‑VAE 的离散瓶颈；2) 通过训练的评分函数挑选最符合人类专家轨迹的候选轨迹；3) 结合推理时成本引导和投影优化器实现高效且安全的轨迹微调。

**🔧 技术方法**

主要技术包括：Conditional Flow Matching、Transformer‑based 多模态编码器、1D U‑Net 预测向量场、Bernstein 多项式轨迹参数化、推理时碰撞成本引导、投影优化器、以及基于专家演示的评分网络。

**📊 数据集**

使用了自建的稠密人群导航数据集（包含仿真与真实机器人、手动遥控与后处理轨迹），共约 5 小时数据；在该数据上训练了 CFM、评分函数和优化器。

**📈 对比分析**

与 CrowdSurfer、DRL‑VO、CoHAN2.0 等主流基线进行对比，Crowd-FM 在 PEDSIM 环境中成功率提升约 15%‑20%，轨迹长度更短、平均速度更高、计算时间低于 60 ms（CFM 单独 22 ms），展示了显著的性能优势。

**⚠️ 局限性**

局限性包括：1) 依赖 2D LiDAR + 运动追踪的输入，难以直接迁移到更高维感知场景；2) 评分函数需要大量专家演示，数据集覆盖度有限；3) 在极端动态或极高密度场景下仍可能出现过度保守或碰撞风险。

---

## 355. Generating Data-Driven Reasoning Rubrics for Domain-Adaptive Reward Modeling

**arXiv ID:** 2602.06795 | [PDF](https://arxiv.org/pdf/2602.06795v1)

**作者:** Kate Sanders `[一作]` (Johns Hopkins University), Huzefa Rangwala `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过自动化提取错误模式构建领域特定的错误分类体系（rubric），并利用该rubric提升LLM对推理轨迹的错误检测与奖励信号

**💡 创新点**

在不需要大量人工标注的情况下，生成细粒度的错误分类体系，并直接用于强化学习奖励函数，显著提升模型在技术领域的推理准确率

**🔧 技术方法**

使用大型语言模型（Claude 3.5 Sonnet）进行错误抽取与压缩；采用两阶段分类器（关键词检索 + rubric 检查）；在强化学习阶段使用 DAPO/GRPO 进行训练

**📊 数据集**

SWE‑Bench（编程）、NuminaMath（数学）和 Meta NaturalReasoning（化工）等数据集

**📈 对比分析**

与无 rubric 的 LLM 评判器以及基于可验证奖励的基线进行对比；在三大领域的推理轨迹分类中，rubric 方法在特异性、平衡准确率和 F0.5 上均优于基线；在 RL 训练中，rubric 奖励模型在数学推理上达到与可验证奖励相近的准确率，在代码补丁上提升 17.5% 完成率，且仅需 20% 以下的金标

**⚠️ 局限性**

生成 rubric 需要额外的 LLM 推理成本；对非可验证或非技术领域的推广仍需验证；错误压缩可能丢失细节，导致部分错误未被识别；以及对极长或多模态推理轨迹的处理受限

---

## 356. Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences

**arXiv ID:** 2602.06788 | [PDF](https://arxiv.org/pdf/2602.06788v1)

**作者:** Idan Pipano `[一作]` (Technion - Israel Institute of Technology), Mohammad Ghavamzadeh `[通讯]` (Qualcomm)

**通讯引用:** 8711 | [OpenAlex ID](https://openalex.org/A5059310658)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并提出了一种基于非凸 f-散度的直接偏好优化损失（SquaredPO），用于对语言模型进行 RLHF 对齐。

**💡 创新点**

创新点在于放宽了传统 KL 的凸性假设，定义了 DPO‑inducing 与 displacement‑resistant 两个性质，并证明满足这两类条件的非凸 f 可以保持可解性并有效抑制概率位移。

**🔧 技术方法**

采用 f‑DPO 框架、对 f(t)=12(log t)^2 的理论推导、基于 LoRA 的微调以及对比实验。

**📊 数据集**

使用 Meta‑Llama‑3‑8B‑Instruct 作为基础模型，TL;DR 偏好数据集进行训练。

**📈 对比分析**

与原始 DPO 以及 SimPO 等方法对比，实验显示 SquaredPO 在防止概率位移、抗过度优化以及在 AlpacaEval‑2 与 MT‑Bench 上的性能与 DPO 相当或略优。

**⚠️ 局限性**

局限包括实验仅覆盖单一数据集/模型、未验证所有满足性质的 f、对优化器动态的理论分析不足，以及使用 LoRA 方式进行微调。

---

## 357. A Unified Framework for LLM Watermarks

**arXiv ID:** 2602.06754 | [PDF](https://arxiv.org/pdf/2602.06754v1)

**作者:** Thibaud Gloaguen `[一作]` (ETH Zürich), Martin Vechev `[通讯]` (ETH Zürich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的受限优化框架，用来设计和解释大型语言模型的水印算法。

**💡 创新点**

将不同水印方案映射到同一受限优化问题中，揭示了质量‑多样性‑功效三维权衡，并基于该框架推出了针对困扰度、困扰自由与困扰‑熵等约束的全新水印方案。

**🔧 技术方法**

利用受限优化（包括KL、²、Perplexity等距离约束）和惩罚式形式的线性/凸规划，结合哈希得分与采样机制实现水印。

**📊 数据集**

在ELI5数据集上对随机1000条回复进行实验，以测评TPR@1、KL/χ²约束误差、Perplexity等指标。

**📈 对比分析**

与Red‑Green、AAR/KTH、SynthID等现有方法对比，实验表明在对应约束下各方案达到Pareto最优，并且软Perplexity约束的水印在检测率‑质量平衡上优于所有先前方法。

**⚠️ 局限性**

局限性包括仅在token层面而非序列层面优化、未与检测器共同优化，以及无法覆盖完全加密式水印等更大范畴。

---

## 358. Beyond Function-Level Analysis: Context-Aware Reasoning for Inter-Procedural Vulnerability Detection

**arXiv ID:** 2602.06751 | [PDF](https://arxiv.org/pdf/2602.06751v1)

**作者:** Yikun Li `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 30217 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对传统漏洞检测仅聚焦单函数的问题，提出两阶段的上下文感知框架：先通过C代码属性图提取并安全聚类调用者、被调用者及全局变量，再利用LLM进行结构化推理，最终实现跨函数漏洞检测。

**💡 创新点**

创新点在于将上下文进行安全摘要与相关性评分，挑选高影响力的上下文片段，并通过结构化推理模板让模型学习上下文与漏洞之间的因果关系，而非单纯的二分类或模式匹配。

**🔧 技术方法**

技术包括：使用CPG提取调用/全局上下文；用GPT‑4.1生成安全聚类摘要与相关性得分；构造结构化推理提示；在生成的推理轨迹上对Qwen2.5‑Coder（7B/32B）进行有监督微调。

**📊 数据集**

使用PrimeVul、TitanVul和CleanVul三大高质量C/C++漏洞数据集，并在其基础上生成包含19,858个调用者、187,170个被调用者和132,633个全局变量的上下文扩展版本（PrimeVulCTX、TitanVulCTX、CleanVulCTX）。

**📈 对比分析**

与零样本GPT‑4.1、CodeBERT、UniXcoder以及其他SOTA方法对比，Qwen2.5‑32B在三个数据集上的准确率分别提升至67.78%、73.76%和64.94%，比最强基线提升约10–11个百分点；在USENIX Security'25基准上从55.17%提升至67.78%。

**⚠️ 局限性**

局限在于对时间/路径相关的漏洞（如双重释放）效果有限；目前仅处理空间层面的跨函数关系，未覆盖复杂的动态执行路径与状态同步。

---

## 359. Structural bias in multi-objective optimisation

**arXiv ID:** 2602.06742 | [PDF](https://arxiv.org/pdf/2602.06742v1)

**作者:** Jakub Kudela `[一作]` (Brno University of Technology), Anna V. Kononova `[通讯]` (Leiden University)

**通讯引用:** 12831 | [OpenAlex ID](https://openalex.org/A5016332970)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在多目标优化中，构建了一套无信息目标的合成二目标测试问题，用以检测并分析算法的结构偏差。

**💡 创新点**

首次将结构偏差概念扩展到多目标情形，并设计了能够剥离问题信息的测试套件；同时将单目标SB检测工具适配为多目标。

**🔧 技术方法**

采用 BIAS 工具箱与卡方检验、箱式统计以及 Clark‑Evans 指数等统计方法对算法搜索分布进行量化；使用 PlatEMO 框架下的 120 种连续多目标算法。

**📊 数据集**

使用自定义的 11 个合成二目标问题（在二维和十维空间），每个问题具有可控的 Pareto 前沿形状、密度和噪声。

**📈 对比分析**

通过多次重复实验（100 次），比较算法在均匀分布下的偏差指标（BIAS_rej、p 值、边缘/中心箱大小、CEI）。结果显示绝大多数算法存在边界或中心偏差，偏差程度随 Pareto 前沿密度变化而加剧。

**⚠️ 局限性**

限制：仅在 PlatEMO 实现和默认参数下实验，未对参数敏感性或更高目标维度进行深入评估；随机无信息目标可能不适用于专门化算法，需进一步研究。

---

## 360. ClassAid: A Real-time Instructor-AI-Student Orchestration System for Classroom Programming Activities

**arXiv ID:** 2602.06734 | [PDF](https://arxiv.org/pdf/2602.06734v1)

**作者:** Gefei Zhang `[一作]` (Zhejiang University of Technology), Ronghua Liang `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 3815 | [OpenAlex ID](https://openalex.org/A5001531117)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计并实现了ClassAid，一套实时课堂编程活动中的教师–AI–学生三方协同系统。

**💡 创新点**

创新点在于提出六阶段的教学助理框架、在教师仪表盘上实现即时监测与动态反馈模式切换，并将生成式AI与人类教学逻辑结合。

**🔧 技术方法**

技术上使用了GPT‑4o大语言模型、Vue.js+Flask+Firebase架构、Bloom层级判定、主动/被动触发机制以及自适应反馈权重计算。

**📊 数据集**

数据集主要来自54名研究生在一堂Vega‑Lite可视化实验中的交互日志、代码提交与问答记录。

**📈 对比分析**

通过专家评估、学生Likert问卷及访谈评估，TA‑Agent的认知水平估计、反馈正确率与Auto‑Mode选择达95‑96%一致率，学生平均满意度>5/7，系统相较传统人工反馈提升了个性化与教师监控效率。

**⚠️ 局限性**

局限性包括样本规模单一、仅针对简单可视化任务、缺乏多时段或大规模验证、以及系统响应延迟与教师工作负担仍需进一步优化。

---

## 361. Table-as-Search: Formulate Long-Horizon Agentic Information Seeking as Table Completion

**arXiv ID:** 2602.06724 | [PDF](https://arxiv.org/pdf/2602.06724v1)

**作者:** Tian Lan `[一作]` (Alibaba International Digital Commerce), Weihua Luo `[通讯]` (Alibaba International Digital Commerce)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Table-as-Search (TaS) 框架，将长周期信息检索任务转化为表格填充任务，并通过多智能体系统实现结构化规划与搜索状态管理。

**💡 创新点**

创新点在于：① 用结构化表格记录搜索历史与待执行计划，消除传统基于文本的状态管理脆弱性；② 统一处理 Deep、Wide 与 DeepWide 三类搜索任务；③ 将大规模搜索结果外部化至数据库，显著降低 LLM 上下文负担；④ 通过可插拔子智能体提升灵活性与可扩展性。

**🔧 技术方法**

采用多智能体框架（主计划器 + 子搜索代理）、表格 schema 定义、表格完成（Row Expansion / Cell Population）策略、LLM 生成链路、Google Search、网页访问等工具，并结合上下文压缩/折叠技术。

**📊 数据集**

使用了 GAIA、BrowseComp‑ZH、WideSearch 三个公开基准；另外构建了 20 题的 E‑commerce DeepWide 任务集；同时在实验中使用了多种 LLM（GPT‑5、Claude‑Sonnet‑4、Gemini‑2.5‑Flash、Qwen3‑Max 等）和商业系统 Gemini DeepResearch。

**📈 对比分析**

通过与 ReAct‑SA、ReAct‑MA（含计算扩展版）以及 Gemini DeepResearch 等基线对比，TaS 在 Deep Search 上提升了约 +14%‑+17%（GAIA/BrowseComp‑ZH），在 Wide Search 的 Success Rate 达到 3.5%‑3.6%（接近或超过更强模型），在 DeepWide 上提升了 +4.7%‑+5.1%（Column‑F1/Item‑Precision），且在工具调用量更少、测试时扩展更高的情况下保持优势。

**⚠️ 局限性**

局限性包括：① 对非搜索任务效果不稳定；② 关键性能受主计划器模型能力限制，需使用强大 LLM；③ DeepWide 评估仍依赖人工标注；④ 目前缺乏自动切换结构化与自由文本规划的机制。

---

## 362. GhostCite: A Large-Scale Analysis of Citation Validity in the Age of Large Language Models

**arXiv ID:** 2602.06718 | [PDF](https://arxiv.org/pdf/2602.06718v1)

**作者:** Zuyao Xu `[一作]` (Nankai University), Jiaji Liu `[通讯]` (Nankai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了GhostCite框架，对LLM生成的引用进行大规模验证，并评估LLM hallucination率、已发布论文中虚假引用比例以及人类验证行为；

**💡 创新点**

①公开多源检索+相似度判定验证框架；②对13个LLM在40个研究领域的引用生成率进行细粒度基准；③结合论文检索与问卷，揭示验证缺口与传播链；

**🔧 技术方法**

使用GROBID与LLM重解析、SQLite缓存、DBLP/Google Scholar等学术数据库检索、Levenshtein相似度阈值0.9、异步并发API、LLM验证提示等技术；

**📊 数据集**

使用375,440条LLM生成引用、2.2M条引用（56,381篇AI/ML与Security顶会论文）、94条问卷回应以及16人手工校验的手工标注集；

**📈 对比分析**

LLM hallucination率从14.23%至94.93%（平均≈50%），无论是否开启在线检索或链式思考均无显著下降；GhostCite在2.2M引用中发现1.07%论文含无效引用，2025年增长80.9%；人工验证准确率>98%；

**⚠️ 局限性**

限制：①可能误判合法新论文；②仅依据标题相似度，忽略作者、年份等信息；③依赖外部API，受限于速率与可用性；④问卷样本量相对有限；⑤未评估检索-生成混合模型的表现。

---

## 363. F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare

**arXiv ID:** 2602.06717 | [PDF](https://arxiv.org/pdf/2602.06717v1)

**作者:** Daniil Plyusov `[一作]` (T-Tech), Daniil Gavrilov `[通讯]` (T-Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 RLVR 的群组采样框架下，分析了有限组大小导致的稀有正确路径被遗漏的概率，并提出了一种基于 Focal loss 的难度感知优势缩放方法（F-GRPO）来缓解分布锐化问题。

**💡 创新点**

提出了闭式的尾部遗漏概率表达式揭示组大小的非单调影响，并证明在有限采样下正确样本中未采样的质量可能会下降；随后设计了轻量级的难度加权优势缩放，能够直接融入任何基于组相对优势的 RLVR 算法。

**🔧 技术方法**

理论分析、概率推导、分类策略框架、Focal 负载调度、GRPO/DAPO/CISPO 的改进实现、实验评估。

**📊 数据集**

使用 Qwen2.5-7B、Qwen2.5-1.5B-Math、Llama-3.2-3B-Instruct 训练于 DeepScaleR（数学竞赛题集），在 AIME、MATH500、IFEval、GPQA 等领域进行评测。

**📈 对比分析**

与原始 GRPO、DAPO、CISPO 以及带熵/KL 正则化的对比，F-GRPO 在保留或提升 pass@1 的同时，pass@256 明显提升（如 Qwen2.5-7B 的 pass@256 由 64.1 提升至 70.3，且计算成本不变），且在 OOD 任务上也展现了更好的迁移性能。

**⚠️ 局限性**

仅在群组大小为 8 的实验中验证，未对更大或更小组大小进行全面探究；对不同奖励设计或多样化奖励函数的适用性尚需进一步评估。

---

## 364. Autoregressive Models for Knowledge Graph Generation

**arXiv ID:** 2602.06707 | [PDF](https://arxiv.org/pdf/2602.06707v1)

**作者:** Thiviyan Thanapalasingam `[一作]` (University of Amsterdam), Paul Groth `[通讯]` (University of Amsterdam)

**通讯引用:** 26622 | [OpenAlex ID](https://openalex.org/A5034924491)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了自回归知识图谱生成框架ARK及其变分扩展SAIL，通过将KG序列化为(head, relation, tail)三元组序列进行逐步生成，实现子图的语义有效生成。

**💡 创新点**

创新点包括：①自回归序列化方法可学习隐式语义约束（类型一致、时间一致、关系模式）而不需要显式规则；②引入变分自回归实现可控生成与条件完成；③证明隐藏维度≥64比加深网络更关键；④在IntelliGraphs基准上实现90–100%语义有效率。

**🔧 技术方法**

使用技术包括：GRU/Transformer自回归解码器；β‑VAE变分框架；序列化token化与位置随机化；温度/Top‑k采样与Beam搜索；评估指标包括语义有效性、创新率与压缩率。

**📊 数据集**

使用的数据集为IntelliGraphs基准，包含三种合成数据集（syn‑paths、syn‑types、syn‑tipr）和两种WikiData衍生数据集（wd‑movies、wd‑articles）。

**📈 对比分析**

与TransE、DistMult、ComplEx等传统KGE基线以及Uniform基线进行比较；在合成集上ARK达99–100%有效率，在真实集上约97–99%；同时压缩率最小，显示显著优于基线；SAIL在无监督与条件生成任务中同样保持高有效性。

**⚠️ 局限性**

局限性：仅适用于固定词表，无法动态引入新实体/关系；生成子图规模限制在3–212三元组；模型隐式学习规则难以解释，可能隐藏偏差；序列化引入顺序约束但对性能影响有限。

---

## 365. NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models

**arXiv ID:** 2602.06694 | [PDF](https://arxiv.org/pdf/2602.06694v1)

**作者:** Hyochan Chong `[一作]` (Samsung Research), Minseop Choi `[通讯]` (Samsung Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种名为 NanoQuant 的后训练量化方法，能够将大型语言模型（最多70B参数）压缩至1比特甚至低于1比特，并能在单块GPU上完成压缩和推理。

**💡 创新点**

创新点包括：①首次实现后训练量化到低于1比特；②将量化建模为低秩二进制因式分解；③引入精确的 ADMM 初始化（Latent‑Binary ADMM）以及 Hessian‑aware 预条件；④采用分块重构与全局标度校准的两阶段管线，显著降低元数据开销；⑤提供专门的二进制 GEMV/GEMM CUDA 核心，提升推理效率。

**🔧 技术方法**

核心技术：低秩二进制因式分解、Latent‑Binary ADMM、K‑FAC Hessian 预条件、Shrinkage 正则、SVID 符号分解、Straight‑Through Estimator（STE）、自定义二进制 CUDA 核心、全局 KL 对齐。

**📊 数据集**

使用的主要数据集：128 条 WikiText‑2 校准样本（0.26M tokens）用于量化；WikiText‑2 用于困惑度评估；六个通用推理任务（WinoGrande、HellaSwag、BoolQ、ARC‑Easy、ARC‑Challenge、PIQA）用于零样本推理评估。

**📈 对比分析**

与现有后训练量化基线（BiLLM、ARB‑LLM、STBLLM、HBLLM）和量化感知训练基线（OneBit、BinaryMoS、LittleBit、DBF）对比。NanoQuant 在多种模型（Llama‑2、Gemma、Qwen、Rnj）上实现了与高比特 PTQ 接近甚至匹配的困惑度和零样本准确率，同时压缩率达到 25.8×（70B → 5.35 GB），在 RTX 3050 上提升 3.6× 推理吞吐量、5.4× 内存节省、3.9× 能效，并在 H100 上实现 10× 内存占用下降。

**⚠️ 局限性**

局限性：目前仅在极小的校准集上验证，进一步扩大数据量与计算资源有望提升复杂推理任务表现；在 2‑3 比特 PTQ 的高精度性能仍有差距；需要自定义 CUDA 核心，对硬件适配度受限；未来工作将聚焦压缩运行时优化、在更大数据集上的验证以及对新一代 GPU/边缘设备的适配。

---

## 366. Rare Event Analysis of Large Language Models

**arXiv ID:** 2602.06791 | [PDF](https://arxiv.org/pdf/2602.06791v1)

**作者:** Jake McAllister Dorman `[一作]` (University of Nottingham), Juan P. Garrahan `[通讯]` (University of Nottingham)

**通讯引用:** 12024 | [OpenAlex ID](https://openalex.org/A5034908857)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一套完整的稀有事件分析框架，用于评估大语言模型生成文本中的极端可读性指数（ARI）和对数概率，并通过该框架对TinyStories-8M模型的稀有输出进行了概率估计与属性探究。

**💡 创新点**

创新点在于首次将物理和统计学中的稀有事件采样技术（如TPS、umbrella sampling与MBAR）应用于LLM，形成端到端的稀有事件估计与探索流程，并证明可在有限算力下准确捕捉到远低于直接采样的稀有事件概率。

**🔧 技术方法**

主要技术包括重要采样、指数加权分布、迁移路径采样（TPS）、马尔可夫链蒙特卡洛、MBAR估计器、bootstrap与Wilson置信区间分析，以及对比直接（先祖）采样。

**📊 数据集**

使用TinyStories-8M模型（基于儿童故事数据集）作为实验对象，采用固定提示“Once upon a time, in a big forest, there lived a rhinoc”生成100词完成文本，并将训练集的ARI分布作为基准。

**📈 对比分析**

通过对比直接采样（约4200万条完成文本）与TPS+MBAR采样（约4×10⁸词），发现后者在尾部能够估计概率小至10⁻¹⁰的事件，置信区间宽度显著缩小，展示了更高的精度和更好地覆盖稀有事件。

**⚠️ 局限性**

局限性包括：计算成本仍高，尤其对更大规模LLM难以直接扩展；稀有事件的定义和偏置设计对结果影响较大；仅针对ARI和对数概率两个观测量，尚缺乏对更复杂属性的通用方法；结果在多样化用户提示与实际部署环境中的泛化仍需进一步验证。

---

## 367. Disentanglement by means of action-induced representations

**arXiv ID:** 2602.06741 | [PDF](https://arxiv.org/pdf/2602.06741v1)

**作者:** Gorka Muñoz-Gil `[一作]` (University of Innsbruck), Hans J. Briegel `[通讯]` (University of Innsbruck)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出行动诱发表示（AIR）框架和变分 AIR（VAIR）变分自编码器，利用实验动作学习可解释的可解缠结表征。

**💡 创新点**

在行动空间上给出最小 AIR（minAIR）的理论证明，证明可通过行动依赖实现可解缠结；设计双编码器结构以逼近 minAIR，突破非线性 ICA 的不可辨识性。

**🔧 技术方法**

基于变分自编码器（VAE）技术，加入双编码器（E_X 与 E_A）、动作依赖的方差控制、ELBO 正则化以及互信息差（MIG）评价指标。

**📊 数据集**

使用三类数据集：1）抽象基准数据（四个因子、随机映射）；2）经典物理实验（质量、电荷碰撞与电场轨迹）；3）量子物理实验（二维量子态、75个投影测量）。

**📈 对比分析**

与 βVAE、TC‑VAE、VAE+action、VAE+action+TC 等标准变体对比，使用 MIG 衡量解缠程度。VAIR 在所有实验中均显著提升解缠性能，尤其能准确分离通过不同动作影响的因子。

**⚠️ 局限性**

理论和实验仅覆盖离散或有限动作集合；对连续/高维动作空间的推广尚未完成；外部干扰（如磁场）可导致多种 minAIR 并使模型不再偏好特定解；训练对每个动作需要单独配置，计算成本相对较高。

---

## 368. Redundant is Not Redundant: Automating Efficient Categorical Palette Design Unifying Color & Shape Encodings with CatPAW

**arXiv ID:** 2602.06792 | [PDF](https://arxiv.org/pdf/2602.06792v1)

**作者:** Chin Tseng `[一作]` (University of North Carolina at Chapel Hill), Danielle Albers Szafir `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 1799 | [OpenAlex ID](https://openalex.org/A5056903170)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过四项大规模人机实验，系统评估了在多类别散点图中使用颜色-形状冗余编码的效果，并基于实验数据开发了自动化的类别调色板生成工具 CatPAW。

**💡 创新点**

创新之处在于首次量化颜色与形状的配对交互对分类感知的影响，构建了颜色、形状及其组合的成对准确率矩阵，并提出了数据驱动的调色板优选模型，能够根据类别数自动推荐最优冗余编码。

**🔧 技术方法**

研究方法包括使用 MTurk 进行任务型实验（相关性判断），应用 CIELAB 颜色空间采样、k‑means 聚类、形状采样、统计方差分析、成对准确率计算以及贪心算法进行组合优化，最终形成预测性能的统计模型。

**📊 数据集**

数据来源为 150–170 名 MTurk 参与者完成的 14,850–14,935 次试验，共涵盖 39 种颜色（通过 k‑means 生成）和 39 种形状（Shape It Up 提供），并使用四种公开颜色调色板与六种形状调色板进行组合实验。

**📈 对比分析**

通过与非冗余编码、设计师默认调色板以及用户自选调色板的对比，实验显示冗余编码在 5–8 类别时准确率提升 10%+，CatPAW 生成的调色板在模型预测下的平均准确率高于传统设计 8–12% 以上。

**⚠️ 局限性**

局限性包括仅在白色背景下评估、仅涉及颜色和形状两种通道、实验任务局限于相关性判断、样本色彩和形状集合有限，以及未考虑美学偏好或语义联想等因素。

---

## 369. Weisfeiler and Lehman Go Categorical

**arXiv ID:** 2602.06787 | [PDF](https://arxiv.org/pdf/2602.06787v1)

**作者:** Seongjin Choi `[一作]` (Pohang University of Science and Technology), Se-Young Yun `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1624 | [OpenAlex ID](https://openalex.org/A5091674853)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了分类 Weisfeiler–Lehman 框架，将超图通过范畴论的函子升维到梯度偏序集，并基于此设计了两种超图同构网络 (I‑HIN 与 S‑HIN)。

**💡 创新点**

创新点在于：① 将升维抽象为函子，提供统一的高阶表示方法；② 选用梯度偏序集作为所有高阶结构的通用目标域；③ 引入入射函子与对称单纯复形函子两种升维策略，并证明其表达力至少与传统 Hypergraph Weisfeiler–Lehman 相当。

**🔧 技术方法**

使用范畴论、函子映射、梯度偏序集上的 Weisfeiler–Lehman 色彩细化，以及神经网络化的 CatMPN（消息传递网络）实现；实验中采用 10‑fold 交叉验证与多种基线对比。

**📊 数据集**

实验数据集包括六个真实超图分类基准：IMDB_dir_form、IMDB_dir_genre、IMDB_wri_form、IMDB_wri_genre、steam_player 与 twitter_friend。

**📈 对比分析**

通过与 MLP、Hypergraph Neural Networks、Transformer 等基线模型的 10‑fold 交叉验证比较，I‑HIN 与 S‑HIN 在 5/6 任务上均取得最优成绩，尤其在 twitter_friend 任务提升 6.5%，在 IMDB_wri_genre 任务提升 6.8%；steam_player 上略逊。

**⚠️ 局限性**

限制包括：对超边 cardinality>20 的截断导致信息丢失；S‑HIN 在超图稠密时计算复杂度高且无明显优势；在超图结构不够丰富时，升维带来的表达提升有限。

---

## 370. Towards Understanding What State Space Models Learn About Code

**arXiv ID:** 2602.06774 | [PDF](https://arxiv.org/pdf/2602.06774v1)

**作者:** Jiali Wu `[一作]` (Technische Universitat), Mira Mezini `[通讯]` (National Research Center for Applied Cybersecurity ATHENE)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统的隐层表示和频域卷积核分析，探讨了基于状态空间模型（SSM）的代码理解模型 CodeSSM 与 Transformer 代码模型 RoCoder 在代码语法和语义捕获方面的差异，并针对发现的短距离依赖弱化问题提出了高频路径与多核改进的架构，显著提升了在代码检索、NLCodeSearch 与类型推断任务上的表现。

**💡 创新点**

创新点包括：①首次将 DirectProbe 与频域卷积核分析相结合，对多层 SSM 进行频谱分类，揭示了 fine‑tune 过程中出现的“频谱偏移”与短距离依赖遗忘；②提出 SSM‑Interpret 框架，实现对 SSM 卷积核的低/高/带通分类；③基于分析结果设计高频 CNN 并增设多核 SSM 方案（CodeSSM‑HF、CodeSSM‑8kernel），直接解决高频信息不足问题，并通过实验验证提升性能。

**🔧 技术方法**

主要技术：
- 直接无监督探针（DirectProbe）用于评估隐层对 AST/DFG 关系的捕获能力。
- 频域卷积核分析（SSM‑Interpret），使用傅里叶变换、谱质心和低/高频能量比等指标对 SSM 内核进行分类。
- 高频路径改进：在 SSM 旁加入 1‑D CNN（kernel size = 3）实现局部依赖捕获。
- 多核 SSM 设计：在每层共享 8 / 1024 个卷积核，以提升建模容量。
- 训练与微调：在代码检索、NLCodeSearch 与类型推断任务上分别微调。

**📊 数据集**

使用的主要数据集：
- Stack Overflow 问答检索（SQA）数据集，用于代码检索任务。
- NLCodeSearch 数据集，用于自然语言与代码检索。
- 类型推断（type‑inference）数据集，用于评估对局部与全局依赖的理解。
- 代码语法与语义图（AST/DFG）用于构造探针标签。

**📈 对比分析**

比较方法：对比 CodeSSM 与 RoCoder 在预训练与微调后各任务的探针准确率、MRR 与 F1；通过可视化核频谱展示频谱偏移。实验结果显示：
- 预训练阶段 CodeSSM 在语法/语义捕获上优于 RoCoder。
- 微调后，CodeSSM 在类型推断任务中短距离依赖显著衰退；RoCoder 保持或提升。
- 引入高频路径后，CodeSSM‑HF 与 CodeSSM‑8kernel 在 NLCodeSearch MRR 从 25.39 提升至 30.89，SQA MRR 从 76.08 提升至 79.57，类型推断 F1 从 59.70 提升至 60.98。

**⚠️ 局限性**

限制与未来工作：
- CodeSSM 原始架构对短距离依赖表现不足，导致 fine‑tune 时频谱偏移。
- 高频路径与多核改进虽提升性能，但仍需进一步平衡参数量与推理效率。
- 频谱阈值和能量比阈值的确定仍基于经验，缺乏理论保障，未来可探讨更鲁棒的分类方法。
- 对不同编程语言、任务规模及真实世界数据的泛化能力尚未充分验证。

---

## 371. Soft Forward-Backward Representations for Zero-shot Reinforcement Learning with General Utilities

**arXiv ID:** 2602.06769 | [PDF](https://arxiv.org/pdf/2602.06769v1)

**作者:** Marco Bagatella `[一作]` (ETH Zurich), Andreas Krause `[通讯]` (ETH Zurich)

**通讯引用:** 30512 | [OpenAlex ID](https://openalex.org/A5003040843)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Soft Forward-Backward (Soft FB) 算法，在离线数据中学习一族基于熵正则化的随机策略，并通过零阶搜索在测试时优化任意可微的 General Utility 目标，实现零样本强化学习。

**💡 创新点**

创新点在于：① 将熵正则化引入 Forward-Backward 框架，使得可检索的策略从确定性扩展到完整支持的随机策略；② 证明在充分容量下，Soft FB 能覆盖所有最大熵 RL 的最优策略，从而间接能零样本优化任意可微的 General Utility；③ 通过低维任务嵌入的重参数化，将搜索空间压缩到球面表面，兼顾理论性与实践性。

**🔧 技术方法**

技术方法包括：熵正则化最大熵 RL、低秩占用度量分解（forward/backward 低维矩阵）、基于样本的 Bellman 残差最小化、双网络正交化、离线流模型（flow-based generative model）用于显式成功者测度估计、以及零阶搜索（如随机射击或 CEM）寻找最佳任务嵌入。

**📊 数据集**

主要使用的离线数据集为：① 简单 2D 连续动作环境（用于演示与基准）；② DeepMind Control Suite（DMC）四个类别的连续控制任务，采用公开的探索性数据集；③ 预训练的成功者测度流模型的训练数据来自上述两个环境。

**📈 对比分析**

与传统 Forward-Backward (FB) 和其流模型 (FB_flow) 进行对比。实验表明：在线性 RL 与 deterministic IL 任务上差距不大；在需要随机策略或非线性目标（纯探索、KL 逼近 stochastic/ deterministic 专家、鲁棒 RL）时，Soft FB 在离线评估与环境实际表现上显著优于 FB，尤其是使用显式测度模型时性能提升最为明显；在 DMC 的标准线性任务中，Soft FB 的熵调节后表现与 FB 相当甚至略优。

**⚠️ 局限性**

局限性：① 需要无限维任务嵌入与高度表达式的演员，实际可检索策略集合受限；② 精确搜索依赖成功者测度的准确建模，若使用显式流模型会显著增加预训练成本；③ 零阶搜索在高维嵌入空间可能收敛慢，可进一步探索更高效的优化方法；④ 目前仅针对成功者测度的离线学习，未处理在线适应或动态环境变化。

---

## 372. R-Align: Enhancing Generative Reward Models through Rationale-Centric Meta-Judging

**arXiv ID:** 2602.06763 | [PDF](https://arxiv.org/pdf/2602.06763v1)

**作者:** Yanlin Lai `[一作]` (Tsinghua University), Daxin Jiang `[通讯]` (StepFun)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文研究了生成式奖励模型（GenRM）在强化学习中因“表面正确”导致的误判问题，并提出了基于推理对齐的训练框架（R-Align）

**💡 创新点**

创新点在于：①引入“Spurious Correctness”指标量化表面正确现象；②设计了元评判（MetaRM）对生成推理进行逻辑一致性检查；③通过“Rationale-Centric Alignment”奖励机制将推理质量与标签准确性同步监督

**🔧 技术方法**

技术主要包括：生成式奖励模型、Chain‑of‑Thought 推理、MetaRM 元评判、强化学习（PPO）以及对比实验评估

**📊 数据集**

使用的数据集有：HelpSteer3、RewardBench2、PPE‑Preference 用于评测；Skywork Reward Preference 80K、Code‑Preference‑Pairs、Math‑DPO‑10K 等用于训练；Arena‑Human‑Preference 及多领域基准（AIME、GPQA‑diamond、LiveCodeBench、MultiChallenge 等）用于下游 RLHF 评估

**📈 对比分析**

与传统仅以标签准确度为目标的 RLVR 基线相比，R-Align 在 F‑Score 上提升 5–15%（降低 Spurious Correctness），并在多项下游 RLHF 任务中实现 10–30% 的性能提升，证明了逻辑一致性监督对政策学习的显著正面影响

**⚠️ 局限性**

局限性包括：对 MetaRM（如 Gemini‑3‑Pro）依赖度高，成本昂贵；元评判的可解释性和公平性待进一步验证；在极大模型规模或多模态任务中的泛化性尚未充分验证

---

## 373. Gold Exploration using Representations from a Multispectral Autoencoder

**arXiv ID:** 2602.06748 | [PDF](https://arxiv.org/pdf/2602.06748v1)

**作者:** Argyro Tsandalidou `[一作]`, George Arvanitakis `[通讯]` (Technology Innovation Institute)

**通讯引用:** 223 | [OpenAlex ID](https://openalex.org/A5017969844)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用 Sentinel‑2 多光谱影像，训练一个基于 Masked AutoEncoder 的 Isometric 自动编码器生成特征嵌入，并以 XGBoost 轻量级分类器对金矿区进行探测，验证生成表征的有效性。

**💡 创新点**

创新点包括：① 在海量无标签 Sentinel‑2 数据上训练专门的 Isometric 生成模型；② 将冻结的生成表征作为固定特征输入轻量级分类器；③ 与原始光谱、SpectralGPT 等方法做对比，证明其更优性能。

**🔧 技术方法**

主要技术包括 Masked AutoEncoder Transformer 结构的自编码器、冻结编码器提取嵌入、XGBoost 分类器、5 折交叉验证等。

**📊 数据集**

数据集：用于预训练的 FalconSpace‑S2 v1.0（1156800 张 128×128×12 维 Sentinel‑2 图像）；用于评估的金矿探测集 63 张 Sentinel‑2 图像（33 片金矿区，30 片随机非金矿区）。

**📈 对比分析**

通过 patch‑level 与 image‑level 的准确率、Precision/Recall/F1、ROC‑AUC 进行比较，Isometric 在 patch‑level 达到 0.681、image‑level 0.733，显著高于 Raw（0.517/0.554）和 SpectralGPT（0.630/0.635）。

**⚠️ 局限性**

局限性在于样本量有限（仅 63 张图像），缺乏 SAR、超光谱以及多时相数据，泛化性能需在更大范围内进一步验证。

---

## 374. Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding

**arXiv ID:** 2602.06733 | [PDF](https://arxiv.org/pdf/2602.06733v1)

**作者:** Rishabh Jain `[一作]` (University of Cambridge), Amanda Prorok `[通讯]` (University of Cambridge)

**通讯引用:** 2581 | [OpenAlex ID](https://openalex.org/A5066624177)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对多智能体路径规划（MAPF）问题，提出了基于有向超图注意力网络的模仿学习框架，用以显式建模群体交互并提升无碰撞路径规划质量。

**💡 创新点**

创新点在于：①引入有向超图注意力机制，突破传统图神经网络对成对交互的限制；②设计多种超图生成策略（如Lloyd划分、软边界、最短路径），有效捕捉高阶群体动态；③通过对比实验验证超图注意力能缓解注意力稀释并显著提升性能。

**🔧 技术方法**

主要技术包括：超图神经网络（HGNN）与注意力机制、CNN编码器、MLP解码器、模仿学习（IL）与在线数据聚合、温度采样的RL调优，以及多种超图构造算法。

**📊 数据集**

使用POGEMA基准数据集，生成约21K个实例（包含稀疏迷宫、稠密迷宫、空旷房间、稠密仓库等七类地图），并在16、24、32个智能体规模下进行训练与评估。

**📈 对比分析**

与现有的GNN、GPT‑style IL模型（2M、6M、85M）及RL模型进行对比，采用成功率、相对Sum‑of‑Costs（SoC）与平均运行时作为指标；实验表明新模型在大多数地图上成功率提升30%+，SoC下降15%+，在稠密仓库地图上取得75%+成功率，且仅用1M参数、100倍更少训练数据即可击败85M模型。

**⚠️ 局限性**

局限性包括：①对极大规模地图（如ost003d）仍难以规模化；②超图生成策略仍依赖手工设计，可能不适用于所有场景；③在某些稀疏或简单地图上与大型GPT模型相比性能略逊；④实验集中在MAPF，尚未验证跨域通用性。

---

## 375. Filtered Approximate Nearest Neighbor Search Cost Estimation

**arXiv ID:** 2602.06721 | [PDF](https://arxiv.org/pdf/2602.06721v1)

**作者:** Wenxuan Xia `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 40068 | [OpenAlex ID](https://openalex.org/A5100391662)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种针对过滤近似最近邻搜索的成本估计框架和早期终止策略，能够根据查询向量与过滤属性的局部相关性动态调整搜索预算。

**💡 创新点**

创新点在于引入低成本的早探测阶段，实时采集过滤特性与向量相似度的混合特征，并使用轻量级GBDT模型预测整体搜索成本，从而实现精准的动态早停。

**🔧 技术方法**

主要技术包括：图索引（HNSW）与过滤策略、早探测（Early Probe）、LightGBM 轻量级GBDT 预测模型，以及基于查询日志的特征工程和模型训练。

**📊 数据集**

实验使用四个真实数据集：Tripclick（标签过滤）、Youtubeaudio（标签过滤）、Arxiv（标签/范围过滤）和 MSMARCO（范围过滤）。

**📈 对比分析**

在与 Naive HNSW、无过滤特征模型、以及现有的 LAET/DARTH 方法对比后，本文方法在高召回率下平均比基线快 2–3 倍、距离计算量降低 2–3 倍，且在低选择率场景下保持低延迟。

**⚠️ 局限性**

限制：预测模型的绝对误差较大（R² 0.4–0.55），对极端稀疏查询仍需进一步调优；仅支持单属性过滤，未覆盖多属性组合或更复杂的过滤逻辑。

---

## 376. Next-generation cyberattack detection with large language models: anomaly analysis across heterogeneous logs

**arXiv ID:** 2602.06777 | [PDF](https://arxiv.org/pdf/2602.06777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 377. Practical Refinement Session Type Inference (Extended Version)

**arXiv ID:** 2602.06715 | [PDF](https://arxiv.org/pdf/2602.06715v1)

**作者:** Toby Ueno `[一作]` (University of Edinburgh), Ankush Das `[通讯]` (Boston University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了支持算术细化的会话类型系统的类型推断算法，提供子类型推导与约束求解流程。

**💡 创新点**

首次将算术细化与会话类型结合，给出完整的子类型理论和推断算法，并在推断过程中加入三项关键优化以提升求解效率。

**🔧 技术方法**

使用会话类型理论、算术细化、子类型推导、SMT求解器Z3以及Rast编程语言实现推断引擎。

**📊 数据集**

在六个基准程序上评估，基准包括单/二进制自然数表示、线性λ演算等。

**📈 对比分析**

通过比较未优化与已优化版本在六个基准上的求解时间，实验表明优化显著缩短Z3求解算术约束所需的时间，提升了推断性能。

**⚠️ 局限性**

仍依赖一定的手工注释，复杂算术约束下求解仍耗时较长，对更大规模系统的可扩展性尚未充分验证。

---

## 378. Using Large Language Models to Support Automation of Failure Management in CI/CD Pipelines: A Case Study in SAP HANA

**arXiv ID:** 2602.06709 | [PDF](https://arxiv.org/pdf/2602.06709v1)

**作者:** Duong Bui `[一作]` (SAP), Thomas Bach `[通讯]` (SAP)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用大型语言模型（LLM）自动化 SAP HANA 交付 CI/CD 流水线中的故障管理，完成从日志定位根因到生成精确解决方案的全过程；

**💡 创新点**

① 在一次性 LLM 调用中同时完成根因分析与解决方案生成，避免多步调用导致误差累积；② 将流水线结构、故障处理指令和历史失败记录三类领域知识结合，并通过消融实验系统评估其对准确率的影响；③ 在真实工业项目（SAP HANA）中验证 LLM 的可行性与效果。

**🔧 技术方法**

GPT‑4o（温度 0.0）+ function‑calling；正则表达式日志预处理（去除状态更新、相同行、非关键行、脱敏）；检索增强生成（RAG）挑选最相似历史失败记录；自定义 LLM 召回流程识别最下游失败作业。

**📊 数据集**

六个月内 SAP HANA CI/CD 流水线收集的 76 次失败案例，涉及 13 个最下游失败作业；每个案例包含根因、解决方案、关键日志行；这些案例被写入历史失败数据库用于 RAG。

**📈 对比分析**

采用消融实验：对无领域知识、单一、两种、三种知识组合进行评估。评估指标为根因定位准确率、解决方案质量（红/黄/绿）以及整体错误率。结果显示：历史记录贡献最大，加入时精确解决率达 92.1%，根因定位精度 97.4%；不含历史记录时错误率高达 65%。LLM 对最下游失败作业的识别准确率从 88% 提升至 98%。

**⚠️ 局限性**

样本量相对有限，仅覆盖 SAP HANA 这一特定 Jenkins 流水线；使用单一 LLM（GPT‑4o）与固定温度，未验证其它模型或温度下的性能；正则表达式对日志格式变化敏感，若流水线改动需手动维护；LLM 仍可能加入冗余或错误操作，需人工审核；混合（非 LLM + LLM）方案的效果尚待进一步研究。

---

## 379. Revisiting Emotions Representation for Recognition in the Wild

**arXiv ID:** 2602.06778 | [PDF](https://arxiv.org/pdf/2602.06778v1)

**作者:** Joao Baptista Cardia Neto `[一作]` (São Paulo State Technological College), Stefano Berretti `[通讯]` (University of Florence)

**通讯引用:** 4635 | [OpenAlex ID](https://openalex.org/A5013110565)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种将面部表情识别结果从单一标签转变为情绪概率分布的框架，并利用现有的 Valence‑Arousal‑Dominance (VAD) 注释自动为 AffectNet 数据集重新标注。

**💡 创新点**

创新点在于：①将 151 种情绪的 VAD 正态分布映射到情绪类别上，生成概率分布；②通过 VAD 值计算每个情绪的似然，得到软标签；③设计情绪一致性损失（Conflict Matrix）以约束情绪间的互斥关系；④构造 B‑AffectNet 以在 VA 平面上平衡样本。

**🔧 技术方法**

技术手段包括：VAD‑to‑情绪概率映射（高斯似然+LogSumExp），Dominance 估计（CWDE），情绪合并（NIM 与 KD‑tree），基于 ResNet‑50 的特征提取与 Likelihood 头，损失函数为 Focal Loss、Consistency Loss、Guided/Regularized 版本。

**📊 数据集**

使用的数据集是 AffectNet（含 VAD 注释）以及 AffWild2（用于平衡 VA 区域），合并后形成 B‑AffectNet；评估时还收集了 22 名参与者对 126 张图像的分布式注释。

**📈 对比分析**

与传统单标签方法相比，使用分布距离（JS、KL）和相似度（Cosine、Pearson）评价模型，结果显示分布预测误差低于 0.3；单标签准确率约 60%（相对标准 63‑64%），说明方法在描述细粒度情绪上有效，但在传统精度上略低；损失函数改进（Consistency）显著提升分布质量。

**⚠️ 局限性**

局限性包括：仍需 VAD 注释，VAD 数据来源受限；情绪分布依赖于文献中预设的正态分布，可能不完全匹配真实数据；评估仅基于部分情绪（7/14）且未与真实心理实验数据完全对齐；模型在单标签精度上未突破现有最优。

---

## 380. On the Convergence of Multicalibration Gradient Boosting

**arXiv ID:** 2602.06773 | [PDF](https://arxiv.org/pdf/2602.06773v1)

**作者:** Daniel Haimovich `[一作]` (Meta), Milan Vojnovic `[通讯]` (LSE)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了多校准梯度提升（Multicalibration Gradient Boosting）的收敛性质，证明了其迭代过程在回归问题下能够收敛到多校准预测器。

**💡 创新点**

创新点在于首次给出该算法的收敛保证，展示了 O(1/√T) 的收敛速率，并在弱学习器满足平滑性时提升至线性收敛；同时对松弛与自适应缩放策略提供了理论分析与实验验证。

**🔧 技术方法**

主要使用了动力系统与 Lyapunov 能量函数分析、伪逆矩阵的扰动理论以及线性回归和梯度提升的组合技术来推导收敛性质。

**📊 数据集**

实验数据集包括 California Housing、Diabetes、Adult、German Credit 和 Communities and Crime 五个公开回归数据集。

**📈 对比分析**

通过与基准缩放策略（单位、松弛、适应）比较，实验表明在训练误差和多校准误差上均呈现单调下降；松弛策略在过拟合控制上更稳健，而自适应缩放可在接近最优点时实现二次收敛。

**⚠️ 局限性**

局限性在于仅针对平方误差损失和理想的提升器（假设内部优化完美）给出理论；对非凸损失、有限迭代提升器及样本噪声等实际情况的分析尚待进一步研究。

---

## 381. Real time, cross platform visualizations with zero dependencies for the N-body package REBOUND

**arXiv ID:** 2602.06735 | [PDF](https://arxiv.org/pdf/2602.06735v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 382. AEGIS: Adversarial Target-Guided Retention-Data-Free Robust Concept Erasure from Diffusion Models

**arXiv ID:** 2602.06771 | [PDF](https://arxiv.org/pdf/2602.06771v1)

**作者:** Fengpeng Li `[一作]` (University of Macau), Jiantao Zhou `[通讯]` (University of Macau)

**通讯引用:** 9386 | [OpenAlex ID](https://openalex.org/A5037979193)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发一种名为 AEGIS 的概念消除框架，利用对抗消除目标（AET）和梯度正则化投影（GRP）在扩散模型中同时提升对抗提示攻击的鲁棒性和保留非目标概念的能力。

**💡 创新点**

创新点在于（1）AET 通过迭代优化语义中心作为消除目标，使得模型对与目标概念相关的对抗提示具备更高的抵抗力；（2）GRP 通过梯度冲突检测与动态投影实现无数据保留策略，解决传统方法在鲁棒与保留之间的权衡难题。

**🔧 技术方法**

采用对抗训练、梯度正交投影、动态 ω 更新、无监督保留约束等技术，对扩散模型的 UNet 进行细调，并通过 CLIP 文图空间对目标进行对齐。

**📊 数据集**

主要在 Stable Diffusion v1.4 与 v2.1 的原始训练数据上评估，使用 nudity、Van Gogh 风格、Church 对象等概念进行消除实验；保留数据完全不使用，所有实验均在公开可用数据集上进行。

**📈 对比分析**

与 ESD、AdvUnlearn、FMN、SPM 等现有消除方法对比，AEGIS 在 P4D、UnlearnDiffAtk、Ring-A-Bell 等对抗提示攻击下的攻击成功率（ASR）大幅降低（如 nudity ASR 从 87% 降至 1.41%），同时 FID 与 CLIP 分数保持或提升，表明鲁棒性提升的同时保留性能未受损。

**⚠️ 局限性**

局限性包括：在不同模型版本（如 SD v2.1）上鲁棒性下降，需针对新模型重新调参；对极端新概念或更复杂模型的泛化能力尚待进一步验证；以及对超参数（如 β、ω）的敏感性需要更系统的自动化调优。

---

## 383. "Tab, Tab, Bug'': Security Pitfalls of Next Edit Suggestions in AI-Integrated IDEs

**arXiv ID:** 2602.06759 | [PDF](https://arxiv.org/pdf/2602.06759v1)

**作者:** Yunlong Lyu `[一作]` (University of Hong Kong), Hao Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 108402 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性地评估了 AI 集成 IDE 中的下一步编辑建议（NES）功能对软件安全的影响，涵盖了架构拆解、白盒与黑盒漏洞测试以及开发者安全感知问卷。

**💡 创新点**

创新点在于首次提出针对 NES 的三维风险范畴（上下文中毒、事务式编辑、人机交互）并构建对应的攻击面与评估方法，揭示了 NES 与传统自动补全在安全性上的根本差异。

**🔧 技术方法**

研究主要采用了 NES 处理流水线拆解、LLM 生成文本解析、静态代码分析工具（如 CodeQL、tree‑sitter）、以及在线问卷与人工实验相结合的混合方法。

**📊 数据集**

在白盒评估中使用了从 1,000 个 GitHub Java 项目中抽取的 410 个安全测试案例；在黑盒评估中随机挑选 120 个案例，对 Cursor、GitHub Copilot、Zed Editor、Trae 四款主流 IDE 进行手工复现；问卷共收集 385 名开发者（后筛选 269 条有效数据）。

**📈 对比分析**

与商业 IDE 进行对比实验后，发现 NES 系统在上下文中毒、事务式编辑和交互式验证三大维度的漏洞率均高达 70%–80%，表明当前的防御措施不足，且模型更新并未有效缓解上下文相关攻击。

**⚠️ 局限性**

局限性包括：① IDE 生态快速迭代导致实验结果易被后续版本覆盖；② 商业系统闭源、上下文检索机制不透明，黑盒测试仅能覆盖有限案例；③ 人工评测与问卷自评带来主观偏差，难以完全代表实际生产环境的安全态势。

---

## 384. Clinical-Prior Guided Multi-Modal Learning with Latent Attention Pooling for Gait-Based Scoliosis Screening

**arXiv ID:** 2602.06743 | [PDF](https://arxiv.org/pdf/2602.06743v1)

**作者:** Dong Chen `[一作]` (Orthopaedic Centre), Kenneth MC Cheung `[通讯]` (Orthopaedic Centre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了专门用于青少年特发性脊柱侧弯（AIS）筛查的多模态视频学习框架，并构建了无数据泄漏的ScoliGait基准数据集；

**💡 创新点**

创新点包括：①首创的无泄漏、带有放射学Cobb角标签的ScoliGait数据集；②基于临床先验的运动学知识图谱，实现可解释的特征编码；③引入潜在注意力池化机制，用于视频、知识图谱与文本三模态的深度融合；

**🔧 技术方法**

技术手段包括：Vision Transformer（ViT）编码器、Sentence‑Transformers文本编码器、跨模态交叉注意力、潜在注意力池化、位置嵌入对齐等；

**📊 数据集**

使用了ScoliGait数据集（1,572个训练片段，3,300个独立测试片段），每段视频均标注了放射学Cobb角，并生成相应的临床文本描述；

**📈 对比分析**

与单模态（视频、知识图谱）及两模态（视频+知识图谱）模型以及公开ScoNet‑MT模型进行对比，三模态潜在注意力池化模型在测试集上取得最高准确率70.0%和F1分数61.9%，显著优于对手；

**⚠️ 局限性**

局限性在于数据集规模仍有限、缺乏多中心外部验证、潜在的计算复杂度和模型对不同人群的泛化能力待进一步评估。

---

## 385. PrefIx: Understand and Adapt to User Preference in Human-Agent Interaction

**arXiv ID:** 2602.06714 | [PDF](https://arxiv.org/pdf/2602.06714v1)

**作者:** Jialin Li `[一作]` (New York University Abu Dhabi), Hanan Salam `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 1614 | [OpenAlex ID](https://openalex.org/A5047633471)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可配置的评估环境和 Interaction‑as‑a‑Tool（IaaT）范式，用来同时衡量 LLM 代理的任务完成度和与用户交互偏好的契合度。

**💡 创新点**

创新点在于将交互行为视为可调用工具统一管理，并构建 31 个细粒度交互偏好设置与 7 个 UX 维度的 LLM 判定器，实现大规模、可复现、可比较的交互体验评估。

**🔧 技术方法**

采用 LLM‑as‑Judge 多模型评判、IaaT 交互工具、任务细化、用户偏好模拟和任务执行轨迹匹配等技术。

**📊 数据集**

基于 BFCL（Berkeley Function Calling Leaderboard）任务，先对任务提示进行“coarsening”，再用自定义偏好设置的用户模拟器进行多轮交互；实验样本覆盖 31 种偏好配置，约 283 例。

**📈 对比分析**

实验对比了适应（P）与基线（No_P）两种策略；在 Gemini‑3 Flash、Claude Sonnet 4.5、Claude Opus 4.5、Kimi K2 四个模型上，适应策略在保持任务准确率的同时，UX 总分平均提升 7.6%，交互偏好对齐提升 18.5%，并在各维度上均有显著正面效果。

**⚠️ 局限性**

局限性包括：用户模拟器无法覆盖真实人类行为的全部细节；每次实验仅模拟单一偏好，未考察多偏好冲突；在极端交互偏好下，适应可能略微影响工具使用准确率；评估仍基于文本交互，缺乏对长期信任、情感变化等更细腻维度的探测。

---

## 386. SaDiT: Efficient Protein Backbone Design via Latent Structural Tokenization and Diffusion Transformers

**arXiv ID:** 2602.06706 | [PDF](https://arxiv.org/pdf/2602.06706v1)

**作者:** Shentong Mo `[一作]` (Carnegie Mellon University), Lanqing Li `[通讯]` (Zhejiang Lab)

**通讯引用:** 1357 | [OpenAlex ID](https://openalex.org/A5067232825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 SaDiT 框架，通过将蛋白质主链设计迁移到离散结构令牌空间，并在此空间中使用 Diffusion Transformer（DiT）进行反向扩散，显著加速生成流程。

**💡 创新点**

创新点包括：
1) SaProt 结构令牌化，将连续 SE(3) 结构压缩为离散令牌，保证 SE(3) 等价性；
2) 在离散空间中构建 Diffusion Transformer，兼顾长程依赖和可扩展性；
3) 引入 IPA Token Cache，在反向扩散后期缓存并重用 Invariant Point Attention 计算，降低 O(L²) 复杂度。

**🔧 技术方法**

核心技术包括：
- SaProt 结构令牌器（预训练离散词典）；
- Diffusion Transformer（DiT）与自适应层归一化（adaLN）；
- Invariant Point Attention（IPA）与 Token Cache 机制；
- SE(3) 等价性的理论证明。

**📊 数据集**

使用公开 PDB 数据集（X‑ray/Cryo‑EM 解析度<3.5Å）进行训练，fold‑class 条件化任务使用 CATH 标签；实验中还对比了多种基准模型。

**📈 对比分析**

与 FrameDiff、RFDiffusion、Proteus、Proteína 等现有模型比较：
- 设计可行性（scTM>0.5）从 94.4% 提升至 99.5%（无条件），并在 fold‑class 条件下达到 93.2%；
- 多样性与新颖性均显著提高；
- 采样速度从 60‑168 秒/样本降至 0.73 秒/样本，速度提升约 230 倍，且在 800 级残基长链上仍保持 75% 设计可行性。

**⚠️ 局限性**

局限性：
- 目前仅支持单链主链生成，未扩展到多链复合体或配体结合；
- 离散令牌化可能在极高分辨率环段产生量化误差；
- 需要后续实验验证和序列共扩散以进一步提升功能特异性。

---

## 387. Fair Transit Stop Placement: A Clustering Perspective and Beyond

**arXiv ID:** 2602.06776 | [PDF](https://arxiv.org/pdf/2602.06776v1)

**作者:** Haris Aziz `[一作]` (UNSW Sydney), Jeremy Vollen `[通讯]` (Northwestern University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一般度量空间下研究公平公交站点选址问题，并提出了针对正当代表性（JR）和核心（Core）的近似算法；

**💡 创新点**

首次将公平聚类的比例公平性与公交站点选址的核心/正当代表性建立对应关系，提出了ECA与λ‑Hybrid两种算法，取得了JR 1+√2 的最佳逼近并实现了对Core的可调近似；

**🔧 技术方法**

利用聚类与公平中心选择的理论映射、几何分析与凸优化技术证明近似因子，并对JR与Core的下界与上界给出严谨证明；

**📊 数据集**

使用美国Helena市真实通勤路线数据（约10,282条路线、3,075个地点）生成候选站点进行实验；

**📈 对比分析**

与传统Greedy Capture和基于聚类的Baseline进行比较，实验显示ECA和λ‑Hybrid在JR近似上接近1，Core近似优于其他方法，整体性能显著提升；

**⚠️ 局限性**

核心近似只在零通勤时间（仅步行距离）下成立；最坏情况下对核心没有保障，对多点旅行和非零通勤时间的通用性仍有限。

---

## 388. Sparse Spike Encoding of Channel Responses for Energy Efficient Human Activity Recognition

**arXiv ID:** 2602.06766 | [PDF](https://arxiv.org/pdf/2602.06766v1)

**作者:** Eleonora Cicciarella `[一作]` (University of Padova), Michele Rossi `[通讯]` (University of Padova)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了端到端的雷达信号CIR直接的SNN人类活动识别管线，通过学习式脉冲编码实现了高精度与高能效的目标识别。

**💡 创新点**

创新点在于设计了学习自适应的脉冲编码（SCAE）取代传统的Doppler预处理，显著提升稀疏度（81%）并保持接近94%以上的F1得分。

**🔧 技术方法**

采用LIF模型的卷积自编码器、率编码解码、surrogate gradient训练以及脉冲神经网络（SNN）作为分类器。

**📊 数据集**

使用DISC数据集（IEEE 802.11ay CIR测量），包含7名受试者、4类动作（行走、跑步、坐/起、手势）共约2小时录制。

**📈 对比分析**

通过与CAE‑SNN、Delta阈值编码、Direct‑SNN和传统CNN对比，SCAE‑SNN在测试集上实现F1≈95.8%、模型仅28k参数、推理≈19.8 ms，稀疏度最高。

**⚠️ 局限性**

局限性包括仅单天线设置，未验证多用户或多天线环境；对极弱信号（如仅手部动作）仍存在识别困难。

---

## 389. Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory

**arXiv ID:** 2602.06902 | [PDF](https://arxiv.org/pdf/2602.06902v1)

**作者:** Emmanuel Esposito `[一作]` (Università degli Studi di Milano), Mengxiao Zhang `[通讯]` (University of Iowa)

**通讯引用:** 379 | [OpenAlex ID](https://openalex.org/A5101956035)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在无约束在线凸优化中，允许运动成本系数随时间变化时的动态 regret，并提出一种新算法，给出关于路径长度 P_T 和运动成本总和的比较器自适应动态 regret 上界。

**💡 创新点**

创新点是首次在时间可变运动成本设置下实现比较器自适应动态 regret，并将其推广到延迟反馈与时间可变记忆两类实际问题，提供了相应的理论约束和算法复杂度。

**🔧 技术方法**

主要技术包括在线凸优化的自适应方法、路径长度分析、运动成本的可变化处理，以及通过归约把延迟反馈/时间记忆问题转化为可变运动成本问题。

**📊 数据集**

无具体数据集；论文为理论分析，验证通过证明和算法复杂度分析完成。

**📈 对比分析**

与传统 OCO 的静态/动态 regret 结果对比，算法在 λ_t≡0 时退化为最优 √(P_T) 级别；在 λ_t 变化时仍保持 √((1+P_T)(T+∑λ_t)) 的上界，证明了最优性与自适应性。

**⚠️ 局限性**

局限性包括：对运动成本 λ_t 的前置知识假设；算法在实际实现中需对参数进行调节；未扩展至约束 OCO 或更一般的非凸情形。

---

## 390. AIRS-Bench: a Suite of Tasks for Frontier AI Research Science Agents

**arXiv ID:** 2602.06855 | [PDF](https://arxiv.org/pdf/2602.06855v1)

**作者:** Alisia Lupidi `[一作]` (FAIR at Meta), Yoram Bachrach `[通讯]` (FAIR at Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了AI Research Science Benchmark（AIRS‑Bench），为评估LLM驱动的AI研究代理提供了一个无污染、标准化的基准，涵盖20个多领域任务并配套完整任务配置与评估协议。

**💡 创新点**

创新点包括：
1) 设计了统一的任务配置标准和半自动化任务构建流水线；
2) 引入了三种评估指标（有效提交率、归一化分数、Elo评分）与统计聚合方法；
3) 通过不同LLM+ scaffold组合实验，揭示了 scaffold 设计对代理性能的显著影响；
4) 展示了部分代理在单跑中超过人类SOTA的案例，验证基准的挑战性与可扩展性。

**🔧 技术方法**

技术手段：
- LLM + scaffold 架构（Greedy、ReAct、One‑Shot）
- Harness（AIDE、AIDE‑ReAct等）实现执行管理
- 归一化分数采用“march of nines”变换
- Elo评分通过 Bradley–Terry 模型估计
- 任务生成与评估脚本（prepare.py、evaluate.py、metadata.yaml 等）

**📊 数据集**

数据集：
- 20 个任务，来源于 17 篇前沿 ML 论文
- 覆盖 NLP（问答、文本分类、抽取匹配）、数学推理、代码生成、分子与蛋白质预测、时间序列预测等 7 大类
- 使用 HuggingFace 提供的公开数据集（train / test split）

**📈 对比分析**

比较方法：
- 评估 14 个 LLM+ scaffold 组合，进行 10 次种子实验（每任务 24h GPU）
- 统计指标：有效提交率（平均 58.8%）、归一化分数（平均 23.4%）、Elo 排名
- 结果显示：Greedy scaffold 明显优于 One‑Shot 与 ReAct；更大 LLM（gpt‑oss‑120b）表现更好；最优秀代理的 Elo 与人类SOTA 仍有显著差距。

**⚠️ 局限性**

局限性：
- 任务仍可能存在轻微数据泄露风险；
- 评估受限于 24h GPU 预算，未能充分挖掘代理潜力；
- 环境与配置差异可能影响结果复现；
- 人工验证是瓶颈，限制了规模化扩展；
- 仅评估部分 scaffold 与 LLM，未覆盖全部可能组合。

---

## 391. SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks

**arXiv ID:** 2602.06854 | [PDF](https://arxiv.org/pdf/2602.06854v1)

**作者:** Mingqian Feng `[一作]` (University of Rochester), Jianfeng Gao `[通讯]` (Microsoft Research)

**通讯引用:** 38280 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练多轮开放循环对话式攻击模型

**💡 创新点**

提出 prefilling self‑tuning 与 intent‑drift‑aware reward 的两阶段 RL 框架

**🔧 技术方法**

使用 prefilling self‑tuning、GRPO 强化学习、无回声指令的开放循环生成

**📊 数据集**

AdvBench 与 HarmBench 数据集

**📈 对比分析**

相较于手工、模板和单轮 SOTA，多轮方法的 ASR 最高可达约 80%，显著提升

**⚠️ 局限性**

对大型模型的泛化仍有限，且需大量训练时 victim 才能有效

---

## 392. Rethinking Multi-Condition DiTs: Eliminating Redundant Attention via Position-Alignment and Keyword-Scoping

**arXiv ID:** 2602.06850 | [PDF](https://arxiv.org/pdf/2602.06850v1)

**作者:** Chao Zhou `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 23047 | [OpenAlex ID](https://openalex.org/A5064573190)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多条件Diffusion Transformer的高效注意力框架 PKA，用以提升多条件图像生成的速度与内存效率。

**💡 创新点**

创新点在于将全局注意力拆分为位置对齐注意力（PAA）和关键词域注意力（KSA），以及设计了条件敏感采样（CSAS）来聚焦关键时间段，显著减少冗余计算。

**🔧 技术方法**

主要技术包括位置对齐注意力、关键词域注意力、条件敏感采样、KV 缓存机制，以及对比实验中的性能指标评估。

**📊 数据集**

采用 Subject200K 子集对 FLUX.1 进行微调，并与多条件控制基线在多条件文本-图像任务上进行对比。

**📈 对比分析**

在 Subject‑Canny、Subject‑Depth 与 Multi‑Spatial 等多条件任务中，与 UniCombine、OminiControl2、PixelPonder 等方法相比，PKA 在保持或提升图像质量与可控性的同时实现了高达 10× 的推理速度提升和 5.1× 的 VRAM 节省。

**⚠️ 局限性**

局限性包括对阈值参数的敏感性、对极小目标或复杂语义场景的处理不够稳健，以及在更高分辨率或视频生成等更复杂任务中仍需进一步优化。

---

## 393. AI-Generated Music Detection in Broadcast Monitoring

**arXiv ID:** 2602.06823 | [PDF](https://arxiv.org/pdf/2602.06823v1)

**作者:** David Lopez-Ayala `[一作]`, Martin Rocamora `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了专为广播监控场景设计的AI音乐检测基准数据集OpenBMAT_AI，并对现有CNN和SpectTTTra模型进行评测。

**💡 创新点**

创新点在于针对广播环境的短时、低信噪比音乐片段构造，并揭示现有模型在此场景下性能大幅下滑。

**🔧 技术方法**

使用CNN基线和SpectTTTra多尺度Transformer模型进行检测。

**📊 数据集**

使用从OpenBMAT提取的结构和响度统计、Epidemic Sound人类音乐、Suno v3.5生成的AI音乐共计3,294条一分钟片段。

**📈 对比分析**

与流媒体场景相比，模型在广播场景的F1下降到61%（SpectTTTra）和27%（CNN），证明对语音遮蔽和短时输入的鲁棒性不足。

**⚠️ 局限性**

主要限制是模型对低SNR、极短片段的泛化差，且未尝试更复杂的自适应或多模态方法。

---

## 394. On the Identifiability of Steering Vectors in Large Language Models

**arXiv ID:** 2602.06801 | [PDF](https://arxiv.org/pdf/2602.06801v1)

**作者:** Sohan Venkatesh `[一作]` (Manipal Institute of Technology), Ashish Mahendran Kurapath `[通讯]` (Manipal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 persona vector steering 的可识别性，证明在无结构约束下不可识别，并给出可识别条件；通过实验验证等价类存在；

**💡 创新点**

首次从可识别性角度正式分析 persona steering，提出非可识别性定理和可识别性充分条件，并在大模型上验证；

**🔧 技术方法**

使用线性化雅可比分析、ICA、稀疏恢复、交叉层一致性等理论工具，实验中利用对比提示抽取向量、正交扰动检验和语义分数；

**📊 数据集**

使用公开 instruction‑tuned LLM（Qwen2.5‑3B‑Instruct、Llama‑3.1‑8B‑Instruct）以及为三种属性（正式度、礼貌、幽默）构造的对比提示集；

**📈 对比分析**

通过 Cohen's d、相关系数等指标评估正交扰动效果，结果显示正交扰动几乎无差别（d<0.2），表明等价类稳定且与模型、属性无关；

**⚠️ 局限性**

实验仅涵盖中层、两模型和三属性，缺乏多尺度、多环境、真实分布评估；理论基于局部线性，未验证稀疏性、ICA 等条件在实践中的可行性。

---

## 395. FlowDA: Accurate, Low-Latency Weather Data Assimilation via Flow Matching

**arXiv ID:** 2602.06800 | [PDF](https://arxiv.org/pdf/2602.06800v1)

**作者:** Ran Cheng `[一作]` (National University of Singapore), Lailai Zhu `[通讯]` (National University of Singapore)

**通讯引用:** 3571 | [OpenAlex ID](https://openalex.org/A5037640345)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于流匹配的低时延天气数据同化框架 FlowDA，利用 SetConv 将稀疏观测映射到网格并在 Aurora 基础模型上微调生成分析。

**💡 创新点**

创新点在于将流匹配用于生成同化、使用 SetConv 逆观测操作实现稀疏观测融入，并通过 LoRA 或全参数微调在保持参数量较小的情况下实现高精度。

**🔧 技术方法**

使用技术包括流匹配（Flow Matching）、SetConv 观察嵌入、Aurora 3D Perceiver+Swin Transformer 基础模型、LoRA 参数高效微调以及前向 Euler ODE 求解。

**📊 数据集**

数据集为 ERA5 0.25° 1979–2015 重新分析，用于生成观测样本并作为背景与真值。

**📈 对比分析**

通过单步、带噪声单步与 15 天循环同化三类基准实验，FlowDA 在多种观测率下的 RMSE 低于 DiffDA、VAE‑Var 及传统 3D‑Var，且推理速度提升 2–4 倍。

**⚠️ 局限性**

局限性包括仍使用合成观测，未验证对真实不规则观测的鲁棒性，且在极低观测率下仍受限于背景误差，未实现完整时域（4D‑Var）同化。

---

## 396. T-STAR: A Context-Aware Transformer Framework for Short-Term Probabilistic Demand Forecasting in Dock-Based Shared Micro-Mobility

**arXiv ID:** 2602.06866 | [PDF](https://arxiv.org/pdf/2602.06866v1)

**作者:** Jingyi Cheng `[一作]` (Delft University of Technology), Shadi Sharif Azadeh `[通讯]` (Delft University of Technology)

**通讯引用:** 657 | [OpenAlex ID](https://openalex.org/A5046979930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对共享单车系统的15分钟站点级需求预测，提出了两阶段空间与时间自适应上下文表示（T-STAR）框架，实现概率性预测。

**💡 创新点**

创新点在于：①层次化的两阶段结构先估计小时级基线需求，再利用细粒度实时信号和上下文信息提升15分钟预测；②通过中间需求变异信号捕捉短期波动；③结合多源上下文（天气、站点设施、实时地铁流量、季节性等）并输出负二项分布，提供不确定性评估。

**🔧 技术方法**

核心技术：Transformer时序模型（Time‑Series Transformer）作为基线，双阶段注意力架构；负二项分布输出层；多元上下文嵌入与位置编码；概率损失（负对数似然）训练。

**📊 数据集**

实验数据来自华盛顿特区的Capital Bikeshare（10/1–12/31/2022）与WMATA地铁进出站数据以及小时级天气信息，覆盖235个站点、15分钟粒度，零需求占比高。

**📈 对比分析**

与历史均值、最近观测、单阶段Transformer、XGBoost、STAEformer、DeepAR、STGCN、TFT等基线对比，T-STAR在MAE、RMSE、MCRPS、MIS等指标均优于竞争者；在零样本（zero‑shot）扩展实验中仍保持高精度；在极端天气或节假日等异常情况下预测误差更低，区间宽度更紧凑。

**⚠️ 局限性**

局限性：对极端零需求或高噪声短期区间仍有误差；模型训练和推理相对较耗时；对全新城市环境的零样本泛化虽有潜力但仍需验证；对下行（drop‑off）需求的上下文贡献相对有限。

---

## 397. An Adaptive Differentially Private Federated Learning Framework with Bi-level Optimization

**arXiv ID:** 2602.06838 | [PDF](https://arxiv.org/pdf/2602.06838v1)

**作者:** Jin Wang `[一作]` (Xinjiang University), Ming Yan `[通讯]` (Xinjiang University)

**通讯引用:** 27595 | [OpenAlex ID](https://openalex.org/A5006900597)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 FedCompDP 框架，解决联邦学习中异构数据和差分隐私导致的梯度波动与性能下降问题。

**💡 创新点**

创新点包括：① 轻量级本地压缩模块规整中间表示，降低梯度波动；② 服务器端自适应梯度裁剪，依据历史梯度范数动态调整阈值；③ 约束感知鲁棒聚合，引入 CD‑norm 不确定性集合和单步原始‑对偶修正，抑制噪声与非 IID 漂移。

**🔧 技术方法**

采用本地特征维度压缩与稀疏化、本地梯度裁剪 + 高斯噪声注入（DP‑SGD）、基于统计的自适应阈值、CD‑norm 约束与 Lagrange 修正的聚合方法，整体构成联邦学习算法。

**📊 数据集**

实验使用 CIFAR‑10 与 SVHN 两个图像分类数据集，采用 Dirichlet 方式实现非 IID 分布。

**📈 对比分析**

与 DP‑FedSAM、DP‑ACDN、FedACG、AWDP‑FL、FedSA 等基线对比，FedCompDP 在 CIFAR‑10 上取得 0.8108/0.8090 的 Acc/F1，SVHN 上 0.8974/0.8903，较最佳基线提升约 6–7%（CIFAR‑10）和 1–2%（SVHN）。

**⚠️ 局限性**

目前仅在同步、中心化联邦设置下验证；对异步或去中心化场景、不同网络结构及真实系统异构性的鲁棒性仍待进一步研究；隐私‑效用权衡的更细粒度分析仍需深入。

---

## 398. Learning Deep Hybrid Models with Sharpness-Aware Minimization

**arXiv ID:** 2602.06837 | [PDF](https://arxiv.org/pdf/2602.06837v1)

**作者:** Naoya Takeishi `[一作]` (University of Tokyo), Naoya Takeishi `[通讯]` (University of Tokyo)

**通讯引用:** 886 | [OpenAlex ID](https://openalex.org/A5064628313)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了将Sharpness-Aware Minimization（SAM）及其变体应用于混合模型（科学模型+机器学习模型）的学习，以提升科学模型参数的可识别性。

**💡 创新点**

创新点在于：①只对机器学习部分使用SAM，寻找平坦最小值，从而保持模型简洁；②该方法不依赖具体混合架构或额外校准数据，具有通用性；③在多种任务中展示了对科学参数估计的显著提升。

**🔧 技术方法**

主要技术包括：SAM、AdaSAM、FisherSAM在混合模型中的应用；与传统经验风险最小化、L2正则化、功能正则化等方法做对比。

**📊 数据集**

实验使用了六个任务：四个合成数据集（摆钟时间序列、反应扩散、Duffing振荡器、摆钟图像）以及两个真实数据集（风洞测量、光隧道测量）。

**📈 对比分析**

对比方法为无正则化经验风险最小化、L2正则化、功能正则化；SAM族方法在预测误差和科学参数估计误差上普遍优于或相当于基线，尤其在非加性混合架构下表现突出。

**⚠️ 局限性**

局限性：缺乏理论保证能正确识别未知科学参数，性能依赖于数据生成假设；在科学参数不确定性高时仍可能估计不准；对真实数据的基准值选择受限。

---

## 399. Perception-Control Coupled Visual Servoing for Textureless Objects Using Keypoint-Based EKF

**arXiv ID:** 2602.06834 | [PDF](https://arxiv.org/pdf/2602.06834v1)

**作者:** Allen Tao `[一作]` (University of Toronto), Wenjie Xue `[通讯]` (Epson Canada Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于 EKF 将 RGB 图像中的关键点与运动先验融合，实现纹理缺失物体的闭环视觉伺服；

**💡 创新点**

创新点在于将感知与控制耦合为闭环，利用 EKF 实时估计 6D 位姿并通过概率控制法考虑速度不确定性，从而提升鲁棒性与安全性；

**🔧 技术方法**

使用 PVNet 进行关键点检测、Extended Kalman Filter 进行状态估计、PBVS 控制律以及不确定性传播；

**📊 数据集**

PVNet 在约 200k 张合成图像上训练；实验使用 Franka 机械臂+RealSense D435 收集的真实场景数据；

**📈 对比分析**

与 IBVS+PVNet 和 PBVS+PVNet 对比，在正常与恶劣条件下分别取得 95.12%/82.61% 的成功率、最低的位姿误差和更短轨迹，性能显著优于基线；

**⚠️ 局限性**

仅适用于已知 3D 模型的刚性纹理缺失物体，未考虑动态环境、形变物体或深度不确定性，且对关键点检测依赖较大。

---

## 400. DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization

**arXiv ID:** 2602.06827 | [PDF](https://arxiv.org/pdf/2602.06827v1)

**作者:** Victor Dhedin `[一作]`, Majid Khadiv `[通讯]` (Munich Institute of Robotics and Machine Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 DynaRetarget 方案，将人类动作重定向到类人机器人上，并通过强化学习实现可在真实机器人上无缝部署。

**💡 创新点**

创新点在于设计了 SBTO（Sampling-Based Trajectory Optimization），它通过递增优化视野、warm-start 全周期问题，并在每一步保持对早期决策变量的持续优化，从而解决传统 SBMPC 的短视与局部最优问题。

**🔧 技术方法**

技术包括：逆运动学（IK）重定向、人类与机器人关键点匹配、基于采样的优化算法（CEM、MPPI 等）、MuJoCo 物理仿真、强化学习（PPO）与域随机化。

**📊 数据集**

主要使用的数据库是 OmniRetarget，包含数百条类人机器人与盒子互动的 kinematic 轨迹，并在此基础上进一步扩展到不同质量、尺寸与几何形状的物体。

**📈 对比分析**

与 SBMPC 基线（SPIDER）比较时，SBTO 在成功率上提升约 2 倍（从 37.9% 到 74.6%），轨迹更平滑，虽然计算量约三倍，但在 RL 轨迹跟踪训练中能显著提高收敛速度与成功率。

**⚠️ 局限性**

主要局限是计算量随轨迹长度增长而急剧上升，难以处理超长序列；目前仅适用于拥有稠密代价函数的任务，且使用单一高斯采样分布，可能限制搜索多模态解。

---

## 401. AEGPO: Adaptive Entropy-Guided Policy Optimization for Diffusion Models

**arXiv ID:** 2602.06825 | [PDF](https://arxiv.org/pdf/2602.06825v1)

**作者:** Yuming Li `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 10459 | [OpenAlex ID](https://openalex.org/A5013030532)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于注意力熵的双信号自适应策略AEGPO，用来提升GRPO在扩散模型对人类反馈强化学习中的采样效率。

**💡 创新点**

创新点在于将相对注意力熵变化ΔEntropy作为样本学习价值的指示器，将绝对注意力熵峰Entropy(t)识别为关键探索时刻，并通过全局样本分配和局部时序探索两级自适应实现最优计算分配。

**🔧 技术方法**

技术手段包括Transformer注意力熵计算、GRPO强化学习框架、奖励模型(HPSv2.1、PickScore、ImageReward、GenEval)的评估以及多基线对比实验。

**📊 数据集**

使用HPDv2.1和Pick-a-Pic数据集进行训练，采用400个prompt的平衡测试集评估性能。

**📈 对比分析**

与标准GRPO变体（DanceGRPO、BranchGRPO、FlowGRPO、DiffusionNFT）以及FLUX.1‑dev和SD3.5‑M基模型对比，AEGPO在收敛速度上提升2–5倍，最终奖励和多样性指标均优于基线。

**⚠️ 局限性**

局限性包括需要额外提取注意力图导致轻微计算和内存开销，以及对RL流水线的适配性要求较高，适用性受限于模型的可插拔性。

---

## 402. A 26-Gram Butterfly-Inspired Robot Achieving Autonomous Tailless Flight

**arXiv ID:** 2602.06811 | [PDF](https://arxiv.org/pdf/2602.06811v1)

**作者:** Weibin Gu `[一作]` (Institute for AI Industry Research), Guyue Zhou `[通讯]` (Institute for AI Industry Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并飞行了26g的无尾双翼蝴蝶仿生微空中飞行器AirPulse，实现全机载闭环控制的自由飞行。

**💡 创新点**

结合低频大幅度振翼与自定义的Stroke Timing Asymmetry Rhythm (STAR) 节奏生成器，实现平滑、可线性调节的翼幅时序差异控制，并首次完成无尾双翼仿生蝴蝶机器人自主飞行。

**🔧 技术方法**

采用碳纤维网格柔性翼、微型伺服驱动、9轴IMU+气压计状态估计、RLS低频滤波、PID姿态控制以及STAR时序调节。

**📊 数据集**

基于Papilio demoleus 的羽翼纹理和角度数据，实验测得翼面尺寸、重量分布及翼载荷，未使用公开大规模数据集。

**📈 对比分析**

与之前32g以上的蝴蝶仿生机型相比，AirPulse在重量、翼载荷和功耗上实现了显著下降（5.9W，功率负载4.38 g/W），实现了稳定爬升和转弯，展现出更高的能效和机动性。

**⚠️ 局限性**

控制器仍为PID经验调参，无法适应极端扰动；柔性翼的气-结构耦合非线性未建模，且缺乏对多物种脊翼纹理对性能影响的系统性研究。

---

## 403. A Unified Formula for Affine Transformations between Calibrated Cameras

**arXiv ID:** 2602.06805 | [PDF](https://arxiv.org/pdf/2602.06805v1)

**作者:** Levente Hajder `[一作]` `[通讯]` (Eotvos Lorand University), Levente Hajder (Eotvos Lorand University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4`

**🎯 论文内容**

本文推导了一种统一的闭式公式，用于计算两台已标定相机之间局部图像块的仿射变换。

**💡 创新点**

创新点在于将相机相对姿态、像素坐标和局部表面法线统一到一个解析表达式，并将其拆解为三部分矩阵，提供了对多种特例的通用推导框架。

**🔧 技术方法**

主要技术包括平面同伦变换（Homography）的利用、对投影坐标求偏导得到仿射参数以及对公式进行符号化推导。

**📊 数据集**

本文未使用公开数据集，所有验证均基于理论推导与特例（标准立体视角）演示。

**📈 对比分析**

与传统单纯立体仿射模型对比，仅在标准立体基线情形下验证，结果与已有工作一致，未给出数值性能指标。

**⚠️ 局限性**

局限性包括：公式仅适用于局部平面假设；缺乏对实际图像数据的实验验证；对大视差或非平面结构的适用性尚未证明。

---

## 404. Towards Efficient Data Structures for Approximate Search with Range Queries

**arXiv ID:** 2602.06860 | [PDF](https://arxiv.org/pdf/2602.06860v1)

**作者:** Ladan Kian `[一作]` (Augusta University), Dariusz R. Kowalski `[通讯]` (Augusta University)

**通讯引用:** 3555 | [OpenAlex ID](https://openalex.org/A5101948468)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究并实现了一种可调分支因子c≥3的有向无环图（c‑DAG）结构，用于近似单区间覆盖搜索（SRC），并与传统1D‑Tree进行理论与实验对比。

**💡 创新点**

创新点在于：①构造了带重叠分支的c‑DAG，可通过调节c实现更细粒度覆盖；②给出了层差分布（LDD）理论，并用其推导期望搜索时间增量≤2(c‑2)/(c‑1)和误报比的对数改进Θ(log(N/s))；③从安全/隐私角度分析结构泄露，证明c‑DAG在误报率降低的同时能减少结构泄露。

**🔧 技术方法**

使用的技术包括：SRC搜索算法、层差分布（LDD）与概率分析、组合求和、信息熵评估、以及在真实（Gowalla）和均匀合成数据上的实验评估。

**📊 数据集**

使用的数据集为：Gowalla位置签到的约4.19M个时间戳（已去重、截断）和等距分布的合成时间戳，时间域为[0,49 626 707]秒。

**📈 对比分析**

比较方法：先推导LDD并计算期望搜索时间差与误报比例的竞争比；随后在实验中采样查询、记录返回层级差和误报比例；实验结果显示c‑DAG在实际查询长度下误报率显著低于1D‑Tree，并且搜索时间增量仅为常数级，远低于理论下界。

**⚠️ 局限性**

局限性：①对非均匀数据需通过ε‑逼近的LDD，理论适用性有限；②仅在单维数据上验证，扩展到多维仍需研究；③c‑DAG返回层级更集中，可能使查询长度泄露更易被推断；④在极大c值下的空间成本和构建时间仍待进一步评估。

---

## 405. A Cycle-Consistent Graph Surrogate for Full-Cycle Left Ventricular Myocardial Biomechanics

**arXiv ID:** 2602.06884 | [PDF](https://arxiv.org/pdf/2602.06884v1)

**作者:** Siyu Mu `[一作]` (Imperial), Choon Hwai Yap `[通讯]` (Imperial)

**通讯引用:** 2271 | [OpenAlex ID](https://openalex.org/A5061269469)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种名为CGFENet的图神经网络模型，能够在单个框架内实现左室心室在完整心动周期内的前向加载与逆向卸载预测。

**💡 创新点**

创新点包括：①将全局-局部图编码与弱形式启发的全局耦合相结合；②引入基于GRU的时间编码器以捕捉心动周期一致的动力学；③采用循环一致性训练策略，使加载与卸载映射互为近似逆映射，并显著降低对监督标签的需求。

**🔧 技术方法**

技术方法主要包括图注意力网络（GATv2）、全局通道融合、门控循环单元（GRU）时间编码、周期一致性约束和与汇流参数模型的闭环耦合。

**📊 数据集**

使用了基于约2000名健康志愿者PCA统计形状模型生成的67个三角剖分左室网格，结合Fung型本构模型在40–160 mL、0–800 ms范围内的FEA仿真，共计181万条标注的压力-体积-时间样本。

**📈 对比分析**

与GraphUNet、MeshGraphNet等现有图神经网络基线比较，CGFENet在节点位移RMSE、压力R²与RMSE以及闭环P–V循环一致性方面均显著优于基线，且单案例P–V–t表格生成仅需约44 s，速度提升约74倍。

**⚠️ 局限性**

主要局限在数据集缺乏心肌弹性模量与主动张力的个体差异，导致模型目前仅适用于固定参数设置，未来计划扩展数据集以实现多参数适应。

---

## 406. NanoFLUX: Distillation-Driven Compression of Large Text-to-Image Generation Models for Mobile Devices

**arXiv ID:** 2602.06879 | [PDF](https://arxiv.org/pdf/2602.06879v1)

**作者:** Ruchika Chavhan `[一作]` (Samsung AI Center), Abhinav Mehrotra `[通讯]` (Samsung AI Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将17B规模的FLUX.1-Schnell文本-图像扩散模型蒸馏压缩到2.4B，提供可在移动设备上高质量生成图像的模型。

**💡 创新点**

采用逐步压缩管线，针对Diffusion Transformer进行多重冗余剪枝、深度合并及层归一化预计算；引入基于ResNet的令牌下采样机制；提出利用生成过程早期Denoiser层视觉信号的文本编码器蒸馏方法。

**🔧 技术方法**

使用剪枝、Transformer层合并、ResNet令牌下采样、文本编码器蒸馏、流匹配训练以及自适应自注意力压缩等技术。

**📊 数据集**

在公开的大规模文本-图像对数据集（如LAION）上进行训练。

**📈 对比分析**

与教师模型FLUX.1-Schnell及其他移动端T2I模型比较，保持视觉质量，生成512×512图像在移动端约2.5秒；模型大小压缩约7倍，参数量从17B降至2.4B。

**⚠️ 局限性**

仍需较高算力，生成分辨率受限，压缩后对细节再现略逊于服务器端大型模型，且对不同硬件平台的兼容性尚需进一步验证。

---

## 407. TraceCoder: A Trace-Driven Multi-Agent Framework for Automated Debugging of LLM-Generated Code

**arXiv ID:** 2602.06875 | [PDF](https://arxiv.org/pdf/2602.06875v1)

**作者:** Jiangping Huang `[一作]` (Chongqing University of Posts and Telecommunications), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 48858 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TraceCoder，一个面向 LLM 生成代码的多代理调试框架，能自动捕获运行时追踪、因果分析并迭代修复。

**💡 创新点**

创新点在于：①将调试流程拆解为 Instrumentation、Analysis、Repair 三个专责代理；②引入历史经验学习机制（HLLM）以避免重复错误；③采用回滚机制保证每次修复至少不退步；④利用细粒度运行时日志实现精确故障定位。

**🔧 技术方法**

核心技术包括多代理协同、LLM prompt 设计、动态代码插桩、因果推理、历史经验检索与应用、状态回滚控制，所有步骤均以大语言模型为主推理器。

**📊 数据集**

使用的基准数据集有 HumanEval、HumanEval+、BigCodeBench 和 ClassEval，覆盖函数级和类级代码生成任务。

**📈 对比分析**

与 Direct、CoT、Self-Planning、Self-Debugging、INTERVENOR 等现有方法在 Pass@1 进行比较，TraceCoder 在所有模型/数据集上均领先；在 ClassEval 上提升 34.43%（相对），平均 Pass@1 达到 90.72%（相对 11.93%）。

**⚠️ 局限性**

主要限制：1）高 token 消耗，成本偏高；2）依赖测试用例覆盖率，缺陷可能过拟合；3）对大规模项目时插桩和多代理调度的可扩展性待提升。

---

## 408. SURE: Safe Uncertainty-Aware Robot-Environment Interaction using Trajectory Optimization

**arXiv ID:** 2602.06864 | [PDF](https://arxiv.org/pdf/2602.06864v1)

**作者:** Zhuocheng Zhang `[一作]` (Technical University of Munich), Majid Khadiv `[通讯]` (Technical University of Munich)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5043216529)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种名为SURE的鲁棒轨迹优化框架，用以显式考虑机器人-环境接触时间不确定性，并通过在可能的接触时刻分支并最终在共同轨迹处合并，以实现既鲁棒又高效的轨迹规划。

**💡 创新点**

创新点包括：① 在优化阶段引入分支（branching）和合并（rejoining）机制，使得所有可能的接触时刻都能得到对应的轨迹；② 通过约束分支节点的触发条件并限制不确定范围，避免传统MPCC或混合整数方法的计算爆炸；③ 在控制层提供两种使用方式——接触检测时的轨迹调度与无检测时的鲁棒本命轨迹。

**🔧 技术方法**

技术手段主要是：多射击（multiple shooting）方式的非线性规划，使用CasADi+Opti接口构建问题；求解器采用IPOPT；在碰撞模型中加入接触守卫（guard）函数和冲击重置（reset）函数；对接触时刻不确定性使用半宽d来约束；实现了分支与合并的自定义约束结构。

**📊 数据集**

数据集：实验采用两类任务的仿真和真实硬件——
1) cart‑pole与墙壁碰撞任务，采用随机扰动的墙壁位置和恢复系数；
2) Unitree Z1机器人抓取落蛋（球）任务，随机落点高度误差。没有使用公开标准数据集，而是通过内部仿真与机器人实验收集的样本来评估性能。

**📈 对比分析**

比较方法：与传统确定性轨迹优化（nominal）以及在SURE框架下的“鲁棒本命轨迹”（robust nominal）和“轨迹调度”（trajectory scheduling）进行对比。性能指标为成功率：
- 在cart‑pole任务中，nominal约44.8%，SURE+调度约66.4%，鲁棒本命约55.3%；
- 在egg‑catch任务中，nominal 45%，SURE 85%（约+40%）。
计算时间上，SURE相较于Tree‑OCP‑QP减少了约55.85%时间，只增加了约4.87%的成本，显示出优秀的效率与鲁棒性平衡。

**⚠️ 局限性**

局限性：
- 仍假设低层控制器能精确跟踪给定轨迹，未考虑控制误差对鲁棒性的进一步影响；
- 该方法对分支数量和不确定范围参数的设定敏感，过大可能导致计算量恢复；
- 目前仅在固定的接触模型和单一接触事件上验证，对多接触或浮动基系统的扩展仍待研究。

---

## 409. Designing a Robust, Bounded, and Smooth Loss Function for Improved Supervised Learning

**arXiv ID:** 2602.06858 | [PDF](https://arxiv.org/pdf/2602.06858v1)

**作者:** Soumi Mahato `[一作]` (National Institute of Technology), Lineesh M. C `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种鲁棒、边界有限且平滑的损失函数RoBoS-NN，用于提升监督学习的稳健性。

**💡 创新点**

创新点是将RoBoSS损失扩展到回归领域，形成RoBoS-NN损失，并在理论上证明其泛化误差界。

**🔧 技术方法**

使用多层感知器、Adam优化器、TPE超参搜索等技术实现并优化该损失函数。

**📊 数据集**

在四个公开时间序列数据集（Daily_Min_Temperature、Electricity_Load、Monthly_Sunspots、Daily_Gold_Price）上进行实验。

**📈 对比分析**

与MAE、MSE、Huber、Log-cosh等基准损失对比，RoBoS-NN在含噪声样本下平均MAE、RMSE、MASE均降低20-35%。

**⚠️ 局限性**

局限性包括模型对参数的敏感性、仅在MLP上验证、对更复杂网络的适配尚未探究。

---

## 410. Statistical-Based Metric Threshold Setting Method for Software Fault Prediction in Firmware Projects: An Industrial Experience

**arXiv ID:** 2602.06831 | [PDF](https://arxiv.org/pdf/2602.06831v1)

**作者:** Marco De Luca `[一作]` (University of Naples Federico II), Porfirio Tramontana `[通讯]` (University of Naples Federico II)

**通讯引用:** 2596 | [OpenAlex ID](https://openalex.org/A5042889388)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在汽车行业嵌入式固件开发中，利用 Coverity 与 Understand 静态分析工具提取的函数级度量，构建跨项目的故障预测流程，并给出可解释、可直接落地的阈值；

**💡 创新点**

创新点在于提出一种完整的基于统计阈值的跨项目故障预测流程，结合工件追踪模型，避免了黑盒机器学习的可解释性与可落地性问题；

**🔧 技术方法**

主要技术包括静态分析工具指标提取、描述性统计、相关性筛选、Wilcoxon‑Mann‑Whitney 检验、Cliff's delta 以及单样本 Wilcoxon 签名检验（阈值反推）等；

**📊 数据集**

使用了三份真实汽车固件项目（P1、P2、P3）的函数级数据，P1、P2 用于阈值生成，P3 用于独立验证；

**📈 对比分析**

通过将阈值应用于 P3 项目进行 hold‑out 验证，重点评估精确度（precision）——Coverity 平均 0.84、Understand 平均 0.86；召回率与准确率均为中等（≈0.44–0.46 与 0.67–0.68），表明高精度但覆盖度有限；

**⚠️ 局限性**

局限性包括：仅关注精确度，忽略召回；阈值基于结构复杂度指标，可能无法捕捉语义错误；仅三项目、仅 C 语言、仅两种工具，外推性受限；依赖 Jira 追踪的缺陷标签完整性；阈值的鲁棒性与可迁移性未做进一步验证。

---

## 411. GaussianPOP: Principled Simplification Framework for Compact 3D Gaussian Splatting via Error Quantification

**arXiv ID:** 2602.06830 | [PDF](https://arxiv.org/pdf/2602.06830v1)

**作者:** Soonbin Lee `[一作]` (Fraunhofer Heinrich-Hertz-Institute), Cornelius Hellge `[通讯]` (Fraunhofer Heinrich-Hertz-Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于3D高斯展开（3D Gaussian Splatting）的精简框架GaussianPOP，利用渲染方程直接推导的误差准则对每个高斯进行精确可视化误差量化，并以单通量高效算法实现一次性计算；

**💡 创新点**

创新点在于：①给出了闭式解析的像素误差表达式ΔSE_k，直接衡量去除单个高斯对渲染结果的影响；②提出高效的渲染一次、局部计算误差的算法，极大降低计算开销；③支持在训练阶段和后训练阶段的可迭代误差重新量化，进一步提升精简稳健性；

**🔧 技术方法**

核心技术包括3D高斯渲染、α‑混合解析、GPU并行前缀求和、误差基准剪枝、迭代重量化；

**📊 数据集**

实验使用了多种公开场景数据集：Bonsai、Counter、Bicycle、Garden、Playroom、Kitchen、Stump、Tanks&Temples等；

**📈 对比分析**

与现有剪枝方法（GaussianSpa、MaskGaussian、LightGaussian、Mini‑Splatting、Compact3DGS、LP‑3DGS 等）在 PSNR/SSIM/LPIPS 以及模型大小（#G/M）上进行对比。GaussianPOP 在保持相同或更小高斯数量的情况下，PSNR 提升约0.1–0.3 dB，SSIM 提升 0.002–0.01，LPIPS 降低 0.01–0.02，且训练时间与剪枝次数相匹配；

**⚠️ 局限性**

局限性：1) 仍需对高斯数量进行阈值设定，过高的剪枝比例可能导致细节损失；2) 对超稠密模型的迭代重量化消耗时间略高；3) 只针对高斯数量压缩，未对属性压缩做进一步研究；4) 在极低内存环境下仍可能需要进一步的压缩方法。

---

## 412. Wild Guesses and Mild Guesses in Active Concept Learning

**arXiv ID:** 2602.06818 | [PDF](https://arxiv.org/pdf/2602.06818v1)

**作者:** Anirudh Chari `[一作]` (Massachusetts Institute of Technology), Neil Pattanaik `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在由大型语言模型生成程序作为假设空间的神经符号贝叶斯概念学习中，主动查询策略的效果。

**💡 创新点**

发现当使用期望信息增益(EIG)时会因生成器支持不足导致粒子退化，而正测试策略(PTS)在保持生成器可行性方面更稳定，提出了“确认偏差”可能是对稀疏生成空间的合理适应。

**🔧 技术方法**

采用粒子贝叶斯推断、LLM（Gemini 2.5 Flash）生成可执行Python谓词、基于大小原理和代码简洁性先验的概率模型以及EIG和PTS等主动学习策略。

**📊 数据集**

在Number Game（0-100整数域）上评估，目标概念分为易、媒、难三类。

**📈 对比分析**

通过查询次数对比发现，在易概念上PTS优于EIG和随机策略；在中等概念上EIG更快；在难概念上三者普遍失败。

**⚠️ 局限性**

主要局限是生成器支持不足导致的“支持不匹配陷阱”，未进一步探索生成器感知的采集函数或鲁棒重生机制。

---

## 413. Solving parametric polynomial systems using Generic Rational Univariate Representation

**arXiv ID:** 2602.06817 | [PDF](https://arxiv.org/pdf/2602.06817v1)

**作者:** Florent Corniquel `[一作]` `[通讯]` (Sorbonne Université), Florent Corniquel (Sorbonne Université)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

本文提出了一种泛化的单变量有理表示（GRUR），用于描述参数化零维多项式系统的解集，并给出了其特殊化性质与复杂度上界。

**💡 创新点**

创新点在于：① 引入GRUR概念，证明其在除去一维闭代数集后对参数化系统具有良好特殊化性；② 通过Van der Waerden式的结果元与算术Nullstellensatz推导出GRUR系数的度与高度上界；③ 提出了两种高效的Las-Vegas算法——基于线性代数与基于评估/插值——用于计算GRUR，并给出了详细的复杂度分析。

**🔧 技术方法**

主要技术包括： Gröbner基与S-多项式判据、FGLM方法求乘法矩阵、RUR与GRUR的构造、算术Nullstellensatz、结果元与Stickelberger定理、评估/插值与Schwartz-Zippel定理、Sturm-Habicht序列用于实根分类。

**📊 数据集**

本文未使用具体实验数据集，而是以理论证明和复杂度分析为主；在实验评估中主要使用随机生成的多项式系统作为验证对象。

**📈 对比分析**

与现有的RUR计算方法相比，提出的算法在理论上实现了多项式复杂度，且对参数空间具有全局化的通用性；实验结果显示，评估/插值方案在参数维数较大时能够显著降低内存占用，尽管总运算量仍高于传统单变量RUR算法。

**⚠️ 局限性**

主要局限包括：① 对参数空间的Zariski闭集合的避免需要先估计其度，实际实现可能较复杂；② 评估/插值法在参数维数非常高时仍可能导致插值规模过大；③ 本文所给的复杂度上界在常数项与低阶项上可能过于保守，实际性能可能受限于系数膨胀。

---

## 414. Calibrating Tabular Anomaly Detection via Optimal Transport

**arXiv ID:** 2602.06810 | [PDF](https://arxiv.org/pdf/2602.06810v1)

**作者:** Hangting Ye `[一作]` (Jilin University), Hongyuan Zha `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 24026 | [OpenAlex ID](https://openalex.org/A5046703129)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个模型无关的后处理框架CTAD，用以校准任意表格异常检测器的输出

**💡 创新点**

创新点在于利用两种正态分布表征（经验分布与K-means中心分布）通过最优传输距离衡量测试样本对分布兼容性的破坏，从而得到校准信号；并提供理论证明保证异常样本在期望上获得更高校准分数

**🔧 技术方法**

使用最优传输（Optimal Transport）、K-means聚类、经验抽样以及基于校准项的加权融合技术

**📊 数据集**

在34个来自OODS与ADBench的表格数据集上进行实验，数据维度从5到768、样本量从80到49,097、异常比例从1%到75.4%

**📈 对比分析**

与7种代表性TAD基线（KNN、ECOD、OCSVM、IForest、PCA、MCM、DRL）对比，CTAD在AUC-PR和AUC-ROC上平均提升1.18%–48.89%，并在统计上显著（p<0.05），甚至能把最强基线DRL提升到SOTA水平

**⚠️ 局限性**

局限性在于对K-means聚类质量和参数M、K、λ有一定依赖，虽然实验表明鲁棒性强，但极端高维或极小样本情况可能导致聚类效果下降；同时需要额外的OT计算，尽管开销小，但在极大在线实时场景下仍需关注

---

## 415. RAIGen: Rare Attribute Identification in Text-to-Image Generative Models

**arXiv ID:** 2602.06806 | [PDF](https://arxiv.org/pdf/2602.06806v1)

**作者:** Silpa Vadakkeeveetil Sreelatha `[一作]` (University of Surrey), Anjan Dutta `[通讯]` (University of Surrey)

**通讯引用:** 1398 | [OpenAlex ID](https://openalex.org/A5008386240)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RAIGen框架，用于无监督地在文本到图像扩散模型中发现稀有属性。

**💡 创新点**

创新在于结合Matryoshka稀疏自编码器与稀有度得分，将稀有激活频率与语义独特性结合，首次实现对已编码但被低表达特征的系统发现。

**🔧 技术方法**

使用Matryoshka稀疏自编码器、CLIP语义聚类、稀有度得分、Prompt修正等技术。

**📊 数据集**

在WinoBias、COCO、Stable Diffusion v1.4、SDXL、FLUX.1-schnell等数据集上进行实验。

**📈 对比分析**

与OpenBias等方法对比，RAIGen发现的属性在生成中出现率明显更低（Attribute Presence 0.19–0.22 vs 0.94），并能通过Prompt修正显著提升稀有属性出现率，性能优于现有方法。

**⚠️ 局限性**

局限在于只能发现模型已编码的稀有特征，无法覆盖模型未学习的社会敏感属性；稀有度得分对噪声神经元仍有一定依赖。

---

## 416. Symbolic Integration in Weierstrass-like Extensions

**arXiv ID:** 2602.06873 | [PDF](https://arxiv.org/pdf/2602.06873v1)

**作者:** Shaoshi Chen `[一作]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences), David Masser `[通讯]` (University of Basel)

**通讯引用:** 2606 | [OpenAlex ID](https://openalex.org/A5051908428)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出了一套完整的符号积分框架，用于求解由满足一阶非线性微分方程（类似Weierstrass ℘函数）产生的代数扩展中的积分。作者将经典的特殊多项式概念推广到Weierstrass‑like扩展，并在此基础上构造了Hermite化简、特殊化简和多项式化简三种算法，最终给出了℘函数幂的积分递推公式。

**💡 创新点**

创新点主要包括：
① 将特殊多项式的定义扩展到含有代数关系的Weierstrass‑like扩展；
② 设计了针对该扩展的Hermite化简算法，能在保持极点多重性的同时消除分母；
③ 通过特殊化简和多项式化简进一步处理剩余的特殊因子和多项式残差，给出判定和计算积分可积性的完整流程；
④ 证明了若℘函数的多项式残差满足一定次数限制，则其不具备元素积分；
⑤ 推导出℘函数幂积分的递推关系，提供了新的解析表达式。

**🔧 技术方法**

主要技术手段包括：
- 微分代数与代数函数域的值域理论；
- 价值环、极点分解与分辨率指数的概念；
- Hermite化简思想的推广与实现；
- 结合极点、极限与微分算子构造特殊化简与多项式化简；
- 计算机代数工具（如Mathematica HolonomicFunctions包）辅助递推关系推导。

**📊 数据集**

该工作为理论算法研究，未使用具体数据集，而是基于符号计算和数学证明。

**📈 对比分析**

与传统的Risch算法相比，本文在处理由非线性微分方程产生的扩展时，提供了更细致的分解策略；在求解℘函数幂积分时，通过递推公式可直接得到结果。由于未做实测实验，性能评价以理论复杂度和可实现性为主，说明该算法在符号级别可在标准CAS中实现，并在给定的例子中验证正确性。

**⚠️ 局限性**

主要局限性：
- 对特殊点必须为常数的假设（即所有特殊点属于基域常数）在实际应用中可能受限；
- 需要满足多项式q的次数至少为3，否则多项式化简步骤失效；
- 算法实现依赖于高效的因式分解和整性基的计算，复杂度在大规模多项式时可能较高；
- 对于更一般的非线性微分方程（非Weierstrass‑like形式），方法尚不适用。

---

## 417. Plato's Form: Toward Backdoor Defense-as-a-Service for LLMs with Prototype Representations

**arXiv ID:** 2602.06887 | [PDF](https://arxiv.org/pdf/2602.06887v1)

**作者:** Chen Chen `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**通讯引用:** 6011 | [OpenAlex ID](https://openalex.org/A5101720092)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于原型向量的后门清洗框架 “Plato's Form”，能够在缺乏触发器、目标输出和任务域信息的情况下，对大语言模型进行高效、可重用的后门消除。

**💡 创新点**

创新点在于：① 通过模拟多种攻击生成后门向量池并聚合为通用原型；② 用层级对齐自动识别后门所在层；③ 只抑制与原型对齐的低秩分量，实现可调节的纯化强度；④ 兼顾重用、可定制、可解释和运行时效率，适配 BDaaS。

**🔧 技术方法**

技术包括：权重差分提取、均值/主成分聚合、层级余弦相似度阈值、矩阵奇异值分解与投影阈值抑制、可调节缩放因子。

**📊 数据集**

使用公开数据集：GLUE（SST-2、CoLA、QQP、MNLI、Emotion）和聊天后门基准（Chat‑Backdoor）以及多种已知后门攻击（CBA、BadEdit、VPI、BadNet 等），在 Llama‑3‑8B 与 Mistral‑7B 上评测。

**📈 对比分析**

与六种基准（WAN、F/T、NAD、BEEAR、CROW、LETHE）对比，实验显示其在单/多/无触发器设置下，攻击成功率可降至 1.6%–10%，而模型精度下降不到 3%，且对非后门模型几乎无影响；在生成任务上也实现低 ASR 与高 CDA 的平衡。

**⚠️ 局限性**

局限包括：① 对极其分布式或语义级触发器的鲁棒性仍有待验证；② 需要先行构建后门向量池，对新型未知攻击的适配性取决于向量池的多样性；③ 纯化强度 α 需要经验选择，过强导致实用性下降；④ 在极度恶意自适应攻击下可能仍需额外防御。

---

## 418. Prompt Reinjection: Alleviating Prompt Forgetting in Multimodal Diffusion Transformers

**arXiv ID:** 2602.06886 | [PDF](https://arxiv.org/pdf/2602.06886v1)

**作者:** Yuxuan Yao `[一作]` (Fudan University), Siyu Zhu `[通讯]` (Fudan University)

**通讯引用:** 2888 | [OpenAlex ID](https://openalex.org/A5013549550)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多模态扩散 Transformer（MMDiT）在文本分支中出现的提示遗忘现象，并提出一种无训练的提示重注入（Prompt Reinjection）方法来缓解该问题。

**💡 创新点**

创新点在于：①通过层级语义保留评估（CKNNA、PCA、token‑level 属性探测）首次系统揭示文本特征随深度逐步遗忘；②提出在推理阶段对浅层文本特征进行分布锚定与几何（正交 Procrustes）对齐后残差注入深层，从而有效恢复提示信息。

**🔧 技术方法**

使用的技术包括层级语义评估方法（CKNNA、PCA）、token‑level 语义属性探测、分布归一化与锚定、正交 Procrustes 对齐、残差注入、CFG 调节等。

**📊 数据集**

实验使用了 GenEval、DPG‑Bench、T2I‑CompBench++、COCO‑5K 等公开基准数据集，并在 SD3、SD3.5、FLUX、Qwen‑Image 等 MMDiT 模型上进行验证。

**📈 对比分析**

与原始 MMDiT 及 TACA 等基线对比，Prompt Reinjection 在指令遵循、颜色/属性绑定、数字推理、空间关系等子任务上提升约 3%–7%，同时在 HPSv2、ImageReward、PickScore、CLIP 等图像质量与语义一致性指标上保持或略有提升。

**⚠️ 局限性**

局限性包括：仅在推理阶段改造，无法从根本上改变模型对文本的监督不平衡；对不同模型仍需手动调参（浅层层级、注入权重、对齐策略）；对极大规模模型的提升有限，且在高 CFG 规模下仍需额外实验验证。

---

## 419. Vision Transformer Finetuning Benefits from Non-Smooth Components

**arXiv ID:** 2602.06883 | [PDF](https://arxiv.org/pdf/2602.06883v1)

**作者:** Ambroise Odonnat `[一作]` (Noah's Ark Lab), Ievgen Redko `[通讯]` (Noah's Ark Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对视觉 Transformer（ViT）各组件的可塑性（即平均输入输出变化率）进行理论推导与实验验证，并基于此识别出在微调时最易适应的模块。

**💡 创新点**

创新点在于提出了可塑性指标，用其来解释并预测 Transformer 组件在微调过程中的表现，颠覆传统认为更平滑更优的观点，证明高可塑性（非平滑）组件（如注意力和前馈层）在微调时能取得更高且更稳定的准确率。

**🔧 技术方法**

技术上结合了 Lipschitz 常数的上界分析、可塑性指标定义、ViT 的层级结构拆分、单模块微调实验、梯度范数与损失下降曲线分析等；实验使用 PyTorch 框架和官方实现。

**📊 数据集**

使用 ImageNet‑21k 作为预训练数据，随后在 11 个下游分类基准上评估：Cifar10/100、Cifar10‑C（Contrast、Gaussian Noise、Motion Blur、Snow、Speckle Noise）、DomainNet（Clipart、Sketch）、Flowers102、Pets。

**📈 对比分析**

通过在 86M ViT‑Base 与 632M ViT‑Huge 上对每个组件单独微调，覆盖多学习率与 3 个随机种子，比较其最终 Top‑1 准确率与线性探测基准。结果显示：注意力和前馈层获得最高平均准确率且方差最小；相比之下 LayerNorm 表现最差，且高可塑性模块梯度范数更大、下降更快，提升显著且在多种数据集上均可复现。

**⚠️ 局限性**

局限性包括：仅针对视觉 Transformer，未验证语言模型；只探讨单模块微调，未考虑多模块组合或自适应优化策略；理论上给出的是上界，实际可塑性可能更高；实验仅基于 ImageNet‑21k 预训练模型；未系统评估不同规模模型对可塑性的影响。

---

## 420. Decoupling Variance and Scale-Invariant Updates in Adaptive Gradient Descent for Unified Vector and Matrix Optimization

**arXiv ID:** 2602.06880 | [PDF](https://arxiv.org/pdf/2602.06880v1)

**作者:** Zitao Song `[一作]` (Purdue University), David F. Gleich `[通讯]` (Purdue University)

**通讯引用:** 5556 | [OpenAlex ID](https://openalex.org/A5084102378)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种解耦方差与尺度不变更新的 DeVA 框架，统一了 Adam 与矩阵谱优化方法。

**💡 创新点**

创新点在于将 Adam 的自适应步长分离为方差适配与尺度不变更新，并将其推广到矩阵参数，兼具自适应加速与谱优化优势。

**🔧 技术方法**

采用自适应梯度、Kronecker 近似、谱分解（或近似特征分解）与指数移动平均等技术，构建 DeVA 及其变体 DeVA_S∞。

**📊 数据集**

在语言建模上使用 NanoGPT 预训练于 FineWeb‑Edu 数据集，在图像分类上使用 ViT‑L/16 与 ResNet‑20 分别在 ImageNet‑1K 与 CIFAR‑10 数据集。

**📈 对比分析**

与 MuON、SOAP、Adamuon 等方法对比，DeVA 在 token 使用率上比 MuON 低约 6.6%，在验证困惑度上略优于 SOAP，训练速度更快，总体性能均超越现有最优方法。

**⚠️ 局限性**

局限性包括需要额外的矩阵谱分解/近似计算，导致计算与显存开销略高；理论假设如块光滑性和梯度方差上界在实际大规模任务中可能不完全成立。

---

## 421. RFDM: Residual Flow Diffusion Model for Efficient Causal Video Editing

**arXiv ID:** 2602.06871 | [PDF](https://arxiv.org/pdf/2602.06871v1)

**作者:** Mohammadreza Salehi `[一作]` (Samsung AI Research), Abhinav Mehrotra `[通讯]` (Samsung AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于残差流扩散模型的自回归视频编辑方法，能够利用文本提示实现对视频的全局/局部风格迁移和对象移除。

**💡 创新点**

创新点在于将二维图像-图像扩散模型通过残差流预测和对前一帧预测的条件化改造为三维视频-视频模型，实现了时序一致性并保持了高效性。

**🔧 技术方法**

采用的技术包括自回归条件化、残差流扩散过程、CFG引导、Diffusion Forcing训练策略以及基于SD1.5/SD3.5 UNet的扩散模型。

**📊 数据集**

在大规模真实视频编辑数据集Señorita上进行训练和评估，同时使用TGVE、TGVE+和Señorita Benchmark进行多维度测试。

**📈 对比分析**

与现有3D V2V模型EVE及2D I2I模型Fairy、VidToMe相比，RFDM在保持高时序一致性、编辑忠实度的同时，计算成本与I2I模型相当，显著降低内存和延迟。

**⚠️ 局限性**

主要限制是短时记忆范围，难以处理大幅度动作变化；未来可考虑加入KV缓存机制以扩展长时序编辑能力。

---

## 422. Uncovering Cross-Objective Interference in Multi-Objective Alignment

**arXiv ID:** 2602.06869 | [PDF](https://arxiv.org/pdf/2602.06869v1)

**作者:** Yining Lu `[一作]` (University of Notre Dame), Meng Jiang `[通讯]` (University of Notre Dame)

**通讯引用:** 5828 | [OpenAlex ID](https://openalex.org/A5074821819)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了大语言模型（LLM）多目标对齐中的交叉目标干扰问题，并提出了新的权重自适应方法

**💡 创新点**

创新点在于1）正式提出并定量描述交叉目标干扰；2）推导了局部协方差定律和在剪切代理目标下的推广；3）给出基于Polyak–Łojasiewicz条件的全局收敛分析；4）提出Covariance Targeted Weight Adaptation (CTWA) 方法，能够在训练过程中动态调整各目标权重以保持正协方差；

**🔧 技术方法**

技术包括：多目标强化学习（RFT）与标量化（线性、Lagrangian、Tchebycheff 等）；KL 正则化策略、GRPO、PPO 风格剪切代理；自然梯度与 Fisher 信息矩阵；协方差定律推导；Polyak–Łojasiewicz (PL) 条件的全局收敛证明；权重自适应的指数滑动平均和对数空间更新；

**📊 数据集**

使用 Math500 数据集进行实验，评估三种目标（准确率、简洁度、清晰度），并在 Qwen2.5-1.5B（Base 与 IFT）以及 Qwen3-1.7B-Base 预训练模型上进行实验

**📈 对比分析**

与静态/动态权重、MGDA、GradNorm、Lagrangian、PAMA 等基线比较，CTWA 在所有三目标上实现了更平衡且更优的性能：准确率保持最高且不下降，同时简洁度和清晰度也显著提升；实验结果在无剪切的 REINFORCE 与剪切的 GRPO 上均一致，验证了协方差定律和方法的有效性

**⚠️ 局限性**

局限性包括：理论分析基于局部协方差和 PL 条件，未覆盖所有可能的非凸情形；方法仍需在更大规模、更多目标以及不同任务（如对话生成）上进一步验证；对权重更新阈值与学习率的敏感性未系统探究

---

## 423. Consensus-based optimization (CBO): Towards Global Optimality in Robotics

**arXiv ID:** 2602.06868 | [PDF](https://arxiv.org/pdf/2602.06868v1)

**作者:** Xudong Sun `[一作]` (Technical University of Munich), Majid Khadiv `[通讯]` (Technical University of Munich)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5043216529)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文将一致性优化方法（CBO）引入机器人轨迹规划，证明其在弱假设下可收敛到全局最优解。

**💡 创新点**

创新点在于将零阶优化方法置于全局优化框架下，对比局部采样方法，阐释CBO通过粒子动力学实现全局搜索并给出理论收敛性。

**🔧 技术方法**

采用粒子动力学的随机微分方程（SDE）实现CBO，并在三类轨迹优化任务中进行数值实验。

**📊 数据集**

实验基于仿真环境：长时程简单系统、双摆杆（双 Cartpole）和高维类人机模型，仅使用终端成本，无公开数据集。

**📈 对比分析**

与MPPI、CEM、CMA‑ES等主流零阶方法对比，CBO在所有三种任务中均获得更低成本，表现优于对手。

**⚠️ 局限性**

局限性包括仅在仿真中验证，缺乏对约束、多模态行为的处理，实际机器人实验仍待开展。

---

## 424. Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing

**arXiv ID:** 2602.06862 | [PDF](https://arxiv.org/pdf/2602.06862v1)

**作者:** Meng Lou `[一作]` (University of Hong Kong), Yizhou Yu `[通讯]` (University of Hong Kong)

**通讯引用:** 18975 | [OpenAlex ID](https://openalex.org/A5108557359)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AdaRoute，一种基于动态参数路由的混合专家结构，用于高效微调预训练视觉模型，使其在稠密预测任务中仅使用极少可训练参数即可实现接近或超越全量微调的性能。

**💡 创新点**

创新点包括：
- 通过共享大型专家中心，将所有层的适配器参数视为可学习的专家，打破传统层级隔离；
- 采用轻量级路由器为每个 AdaRoute 模块动态聚合专家参数，生成输入相关的低秩权重；
- 引入多尺度深度可变卷积（D2Conv）实现输入依赖的空间混合，进一步提升特征表达。

**🔧 技术方法**

技术手段：
- 轻量化动态参数路由（类似 MoE 的软门控）；
- 共享专家中心（trainable 参数矩阵集合）；
- 动态多尺度深度可变卷积（D2Conv）与空间聚合（SA）模块；
- 在 ViT、Swin、ConvNeXt 等主干网络中嵌入 AdaRoute。
- 评估指标采用 mIoU、AP、PQ 等稠密预测指标。

**📊 数据集**

使用的数据集包括：ADE20K（语义分割）、COCO2017（目标检测、实例分割、全景分割）、ImageNet‑21K（预训练）以及 CIFAR‑100、SVHN、Food‑101、ImageNet‑R（分类评测）。

**📈 对比分析**

与 VPT、LoRA、AdaptFormer、Mona、LoRand、SNELL、CoLoRA 等主流 PEFT 方法对比，AdaRoute 在 ADE20K、COCO 目标检测/实例分割/全景分割任务中均超过同等参数量的对手，并在部分场景下（如 Swin‑L 语义分割）甚至超越全量微调；在分类任务上亦获得最优或接近最优的 top‑1 准确率。

**⚠️ 局限性**

局限性：
- 在极度稠密任务（全景分割）中仍与全量微调存在一定性能差距；
- 由于参数极低，表达能力受限，难以完全捕捉所有复杂空间关系；
- 需要在不同网络结构中手动设置共享专家中心大小与层级分组，调参复杂。

---

## 425. Improved Sampling Schedules for Discrete Diffusion Models

**arXiv ID:** 2602.06849 | [PDF](https://arxiv.org/pdf/2602.06849v1)

**作者:** Alberto Foresti `[一作]`, Pietro Michiardi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

阐述了ICML 2026会议的论文提交与格式规范。

**💡 创新点**

提出了双盲评审、电子提交、PDF格式、Type‑1字体等具体要求，并对标题、作者信息、摘要、章节、图表、算法、参考文献等排版细节作了详细说明。

**🔧 技术方法**

主要使用LaTeX模板及相关宏包（如algorithm、algorithmic），配合pdfLaTeX、dvips等工具来保证字体、图表、表格的正确渲染。

**📊 数据集**

无实验数据集。

**📈 对比分析**

未涉及方法比较或性能评估。

**⚠️ 局限性**

仅为会议模板说明，缺乏实验结果和可直接推广的技术，适用范围受限于ICML 2026。

---

## 426. ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training

**arXiv ID:** 2602.06820 | [PDF](https://arxiv.org/pdf/2602.06820v1)

**作者:** Dunwei Tu `[一作]` (National Key Laboratory for Novel Software Technology), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出ScaleEnv框架，能够从零合成高保真交互式环境和可验证任务，用于训练通用工具使用代理。

**💡 创新点**

创新点在于将工具与数据库的模式定义、代码实现、执行验证、依赖图构建与任务扩展统一到一套自动化流程，并通过执行级验证和图扩展保证环境可靠性与任务可解性。

**🔧 技术方法**

技术包括LLM驱动的工具/数据库模式推理、代码生成与调试、过程式测试、工具依赖图构建、图扩展策略以及基于规则的奖励评估。

**📊 数据集**

使用16个自合成领域，每个领域约50个工具与5-20张数据库表，生成的数据集用于训练Qwen3系列模型。

**📈 对比分析**

在τ²‑Bench、VitaBench等OOV基准上，基于ScaleEnv训练的Qwen3-SE模型在多域、多格式零样本推理任务上显著优于基线，性能提升约15-30%（Avg@4）并在Pass@4上接近上限。

**⚠️ 局限性**

局限在于依赖LLM生成的代码与图，仍需人工监督以防合成恶意或不安全域；对超大规模域的可扩展性与实时性能尚待进一步验证。

---

## 427. SuReNav: Superpixel Graph-based Constraint Relaxation for Navigation in Over-constrained Environments

**arXiv ID:** 2602.06807 | [PDF](https://arxiv.org/pdf/2602.06807v1)

**作者:** Keonyoung Koh `[一作]` (Korea Advanced Institute of Science and Technology), Daehyung Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5074295573)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出SuReNav框架，利用超像素图与图神经网络实现半静态环境下的约束放松、路径规划与执行的交织，产生人类式安全高效导航；

**💡 创新点**

创新点在于将超像素分割的语义区域构成图形结构，学习基于人类示范的放松成本估计器，并在可微A*搜索中直接使用该成本，实现动态约束放松与实时规划；

**🔧 技术方法**

采用超像素分割、图卷积网络（GatedGCN+Transformer）、可微A*（GraphMP）、端到端训练、实时图更新与机器人感知等技术；

**📊 数据集**

基于OpenStreetMap 1x1米地图生成约34个城市示例，并用MIA语义分割得到10类标签，收集1200条人工示范路径；在Boston Dynamics Spot校园环境中进行真实机器人验证；

**📈 对比分析**

与D* Lite、COA*、RCR、ViPlanner+OSRM等四个基线在300个仿真环境和100条人类演示上进行比较，SuReNav在人类相似度Fréchet距离0.334、IoU 0.416、SPL与总风险均优于基线，并在真实机器人实验中实现人类式约束放松；

**⚠️ 局限性**

局限性在于需要预先的语义地图与示范，面对极端动态变化适应有限，对视觉感知依赖PIDNet，且未能处理高频动态障碍物。

---

## 428. POP: Online Structural Pruning Enables Efficient Inference of Large Foundation Models

**arXiv ID:** 2602.06822 | [PDF](https://arxiv.org/pdf/2602.06822v1)

**作者:** Yi Chen `[一作]` (Korea Advanced Institute of Science and Technology), Joo-Young Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 87373 | [OpenAlex ID](https://openalex.org/A5046207199)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在线结构化裁剪框架POP，能够在自回归推理过程中基于上下文动态裁剪通道，实现高效推理。

**💡 创新点**

将裁剪分为粗粒度保留/候选/裁剪区域，利用前缀预填阶段确定候选集，仅在解码阶段在候选集内进行上下文感知细粒度裁剪，且不需要离线校准或额外训练。

**🔧 技术方法**

使用激活感知的重要性度量、分位数阈值划分通道、两阶段裁剪（prefilling + decoding）以及轻量级的在线通道选择算法。

**📊 数据集**

在LLM、MoE、VLM等多种LFMs上评测，使用问答、生成与视觉问答基准，如ARC、MBPP、GSM8K、CoQA、NQ-Open、HumanEval、MME等。

**📈 对比分析**

与Wanda、FLAP、Týr、Probe Pruning等最新结构化裁剪方法对比，POP在20%裁剪率下在生成任务中平均提升约30%准确率，同时仅增加约2.85% FLOPs，获得约1.29×的推理加速。

**⚠️ 局限性**

仅裁剪FFN层，未涉及注意力或其他模块；候选区大小需调参，过大会导致开销增加；对极端稀疏率的鲁棒性仍待验证。

---

## 429. Zero-shot Generalizable Graph Anomaly Detection with Mixture of Riemannian Experts

**arXiv ID:** 2602.06859 | [PDF](https://arxiv.org/pdf/2602.06859v1)

**作者:** Xinyu Zhao `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**通讯引用:** 17898 | [OpenAlex ID](https://openalex.org/A5100380470)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种零样本跨域图异常检测框架（Zero-shot Generalizable Graph Anomaly Detection with Mixture of Riemannian Experts），通过多曲率特征对齐、混合黎曼专家网络以及记忆驱动动态路由实现对未知图中异常节点的高效识别。

**💡 创新点**

创新点：① 在特征对齐阶段将原始特征投影到多种黎曼空间，保留不同曲率下的几何信息；② 采用混合黎曼专家架构，使每个专家在专属曲率空间内重构节点，避免单一曲率导致的结构扭曲；③ 引入基于历史重构质量的记忆驱动动态路由，实现对最合适专家的自适应分配。

**🔧 技术方法**

技术方法：Riemannian 图神经网络、曲率空间映射与投影、混合专家（Mixture of Experts）架构、记忆增强动态路由、无监督重构损失、结构对比学习、门控熵正则等。

**📊 数据集**

使用的数据集：源域 4 个图（PubMed、Flickr、Reddit、YelpChi），目标域 7 个图（ACM、Amazon、BlogCatalog、Citeseer、Cora、Facebook、Weibo）。

**📈 对比分析**

与监督（GCN、GAT 等）、无监督（AnomalyDAE、CoLA 等）以及现有零样本基线（UNPrompt、AnomalyGFM、IA‑GGAD）等方法对比；在零样本跨域设置下平均 AUROC 82.09%、AUPRC 36.96%，比最强零样本 IA‑GGAD 提升 5.09% AUROC，并且超越了少样本微调版本的 ARC 与 AnomalyGFM。

**⚠️ 局限性**

局限性：① 曲率空间的选择仍需预设或手工调参；② 记忆路由机制增加存储与计算开销；③ 对极大规模或动态图的适用性尚未验证；④ 在极端稀疏或高维特征的图上可能表现不佳。

---

## 430. DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos

**arXiv ID:** 2602.06846 | [PDF](https://arxiv.org/pdf/2602.06846v1)

**作者:** Ziyu Luo `[一作]` (Beijing Technology and Business University), Yiran Shen `[通讯]` (Shandong University)

**通讯引用:** 1636 | [OpenAlex ID](https://openalex.org/A5013564110)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对包含遮挡、反射和混响的复杂声场，提出了一种基于 360° 视频几何与材质信息的第一阶 Ambisonics（FOA）生成框架 DynFOA，能够实时生成物理一致的空间音频。

**💡 创新点**

创新点在于：① 通过 3D Gaussian Splatting 对 360° 视频重建完整 3D 场景并提取材质属性；② 将几何与材质得到的物理声学描述（遮挡、反射路径、频率相关的 T60 等）作为条件输入到扩散模型中；③ 结合声源定位、深度与语义信息，实现动态、基于场景的 FOA 合成。

**🔧 技术方法**

使用的技术包括：多模态视觉处理（语义分割、深度估计、目标检测）、3D Gaussian Splatting 场景重建、FOA 编码器、条件扩散模型、HRTF 头部跟踪渲染以及跨模态融合。

**📊 数据集**

主要数据集为自建的 Dyn360（600 条 10 秒 360° 视频，包含 Geometry、MoveSource、MultiSource 三子集），并以 YT360、Sphere360 作为原始来源进行数据预处理和划分。

**📈 对比分析**

与 Diff‑SAGe、MMAudio+spatialization、OmniAudio、ViSAGe 四个基线在 Geometry、MoveSource、MultiSource 三个子集上对比，使用 DOA、SNR、EDT、FD、STFT、SI‑SDR、KL、MOS‑SQ、MOS‑AF 等指标评估。DynFOA 在所有子集上均显著优于基线，DOA 仅 0.08‑0.12，SNR 超 18 dB，MOS 接近 4.4，表明在空间精度、声学保真度和用户体验上都有提升。

**⚠️ 局限性**

存在的局限包括：材质属性估计仅基于语义分割，缺乏频率相关的精细模型；主要在室内环境评估，难以直接推广到户外或水下等不同传播介质；对极端遮挡或复杂多源场景的鲁棒性仍有提升空间。

---

## 431. The Representational Geometry of Number

**arXiv ID:** 2602.06843 | [PDF](https://arxiv.org/pdf/2602.06843v1)

**作者:** Zhimin Hu `[一作]` (Georgia Tech), Sashank Varma `[通讯]` (Georgia Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在多种数值任务中对数字概念的表征几何结构，发现任务特定子空间可通过线性变换映射到共享的关系结构。

**💡 创新点**

提出概念表征共享不在概念本身，而在其几何关系，并展示子空间的线性可变换性，为共享与功能灵活性共存提供机制性解释。

**🔧 技术方法**

采用Procrustes分析、子空间重叠、SVCCA、t‑SNE、PCA、KDE以及距离/大小/比例效应拟合等多种表征几何分析技术。

**📊 数据集**

使用BERT、GPT‑2、Qwen2.5及其数学版等四大模型，在手工设计的数字任务句子以及伪句和真实句两类自然语料中的数字（1–9）嵌入。

**📈 对比分析**

通过Procrustes偏差、子空间重叠率和SVCCA平均相关系数比较不同任务与模型的相似度，结果显示任务子空间几乎完全可线性映射，偏差低（≈0.01），SVCCA高（0.80–0.90），表明共享关系保持而干扰最小。

**⚠️ 局限性**

仅聚焦于数字概念且主要在大规模模型上验证，缺乏动态任务序列的长期实验，对非线性结构的解释也有限。

---

## 432. From Features to Actions: Explainability in Traditional and Agentic AI Systems

**arXiv ID:** 2602.06841 | [PDF](https://arxiv.org/pdf/2602.06841v1)

**作者:** Sindhuja Chaduvula `[一作]` (Vector Institute for Artificial Intelligence), Shaina Raza `[通讯]` (Vector Institute for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对比了传统静态预测的特征归因方法（如 SHAP、LIME）与代理式 AI 的轨迹级诊断方法（如基于 Docent 的行为规则），并提出了最小解释包（MEP）框架来统一两类解释。

**💡 创新点**

创新点在于：①正式区分静态与代理式解释目标；②构建跨范式解释目标与工件的分类法；③将轨迹级规则诊断与传统归因方法在相同表示空间下对比，揭示归因方法在多步决策中缺乏因果定位；④提出 MEP 以结构化方式包含解释、证据和验证信号。

**🔧 技术方法**

使用的技术包括：SHAP、LIME、梯度×输入显著性、CoT、ReAct、Docent 规则评估、HAL‑Harness 轨迹收集、GPT‑5 评判、Logistic Regression 及 CNN 文本分类模型。

**📊 数据集**

数据集：Kaggle JobPosts（IT/非 IT 分类）用于静态实验；TAU‑bench Airline（工具调用）与 AssistantBench（Web 交互）用于代理式实验。

**📈 对比分析**

比较方法：静态场景通过 Spearman ρ 衡量归因稳定性；代理式场景通过规则违规的出现率、预后相关度以及成功率比较来评估诊断效能。结果显示：静态归因稳定性高（TF‑IDF+LR ρ≈0.86），但在代理式任务中归因无法定位具体失败；轨迹级规则诊断能精确定位状态不一致、工具误用等失效点，显著关联任务成功率。

**⚠️ 局限性**

局限性包括：仅评估了两类工具使用代理模型，未涵盖嵌入式、多智能体或自学习系统；规则标签由 LLM 评判，存在主观性；轨迹级解释仅提供关联性而非因果证据；缺乏自动化对抗性或逆向干预方法。

---

## 433. LLM Active Alignment: A Nash Equilibrium Perspective

**arXiv ID:** 2602.06836 | [PDF](https://arxiv.org/pdf/2602.06836v1)

**作者:** Tonghan Wang `[一作]` (Tsinghua University), David C. Parkes `[通讯]` (Harvard University)

**通讯引用:** 15764 | [OpenAlex ID](https://openalex.org/A5086173064)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于纳什均衡的“主动对齐”框架，用低维子人群混合策略预测并引导大语言模型（LLM）群体行为，识别并缓解政治排斥现象。

**💡 创新点**

将对齐目标视为LLM的战略选择而非外部强加；使用子人群混合策略构造可解释、低维策略空间；在满足凹性假设下推导闭式内点纳什均衡；通过调整平台激励系数阐明政治排斥产生机制并给出治理对策。

**🔧 技术方法**

游戏理论（纳什均衡与凹游戏）、线性代数（矩阵求逆与特征分解）、多目标效用建模（吸引力、一致性、多样性），以及LLM子人群模型训练。

**📊 数据集**

POLITICS（包含左右中立的政治观点子人群）、CultureBank（区域子人群）和Big Five Personality Traits（五大人格子人群）等带标签数据集，用于构造子人群模型并评估排斥区域。

**📈 对比分析**

通过对激励系数（β^A、β^I、β^D）进行网格搜索，计算内点纳什均衡权重；对比不同基模型（如Qwen、DeepSeek、Mistral）在相同参数空间下的排斥面积；实验表明：提升多样性系数可显著降低排斥；推理型模型相较于非推理型模型排斥面积更大；结果以排斥面积比例和条件排斥率呈现，证明方法能量化治理效果。

**⚠️ 局限性**

仅适用于静态一次性纳什均衡；假设效用函数凹且可导；内点解假设可能不成立，需处理边界解；子人群混合策略只能捕捉到与子人群相关的行为，无法描述更细粒度文本策略；模型对激励系数的具体映射需要平台设计与实际实现相结合。

---

## 434. Supercharging Simulation-Based Inference for Bayesian Optimal Experimental Design

**arXiv ID:** 2602.06900 | [PDF](https://arxiv.org/pdf/2602.06900v1)

**作者:** Samuel Klein `[一作]` (SLAC National Accelerator Laboratory), Sean Gasiorowski `[通讯]` (SLAC National Accelerator Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

结合模拟基础推理与贝叶斯最优实验设计，提出新的 EIG 估计器和多重并行重启梯度上升（MPR‑GA）优化方法，实现 per‑trajectory BOED 性能提升。

**💡 创新点**

将 NPE、NLE、NRE 三种 SBI 方法映射到不同 EIG 变分下界；提出利用神经似然估计的直接 EIG 估计器；通过 MPR‑GA 打破局部最优，显著提升 EIG 优化。

**🔧 技术方法**

使用现代 SBI 技术（神经后验、神经似然、神经比率估计）、正则化多重并行梯度上升、样本多样性惩罚、自动微分、正则化流等。

**📊 数据集**

在多个科学模拟基准上验证：二维/三维/五维源定位、药物动力学时间点选择、CES 生产函数，通过模拟生成数据。

**📈 对比分析**

与最先进的 policy‑based 方法（DAD、RL‑BOED）以及旧的 per‑trajectory 方法对比，MPR‑GA 在源定位 2D 提升 22%，CES 提升 14%，并在药物动力学静态设计上超过政策。

**⚠️ 局限性**

在高维场景下提升有限，受后验估计质量限制；对非可微模拟器适用性不详；需要较多计算资源；未针对在线训练的样本效率进行深入探讨。

---

## 435. Sample Complexity of Causal Identification with Temporal Heterogeneity

**arXiv ID:** 2602.06899 | [PDF](https://arxiv.org/pdf/2602.06899v1)

**作者:** Ameya Rathod `[一作]` (International Institute of Information Technology), Ponnurangam Kumaraguru `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在时间异质性与多环境异质性共同作用下，如何利用仅二阶统计量恢复线性结构方程模型的因果图，特别是在噪声为Student‑t的重尾分布时的可识别性与样本复杂度。

**💡 创新点**

创新点在于：① 将时间异质性与环境异质性统一为互补来源，证明在环境信息不足时可通过时间异质性弥补；② 对重尾噪声给出的精确样本复杂度上界与下界，揭示了“1+3/(ν-4)” 的本质代价。

**🔧 技术方法**

技术手段包括：线性SEM与ICA框架、二阶矩（协方差）估计、信息理论的Fano/Le Cam方法、矩阵秩与时间窗口分析、以及重尾分布下的高阶矩推导。

**📊 数据集**

实验采用合成数据：在不同维度（d≤10）和不同自由度ν（5–50）的多变量Student‑t噪声下生成时间序列，评估协方差估计误差与因果图恢复性能。

**📈 对比分析**

与高斯基准相比，重尾噪声下需约“1+3/(ν-4)”倍的样本才能达到相同的估计误差；实验结果与理论预测高度一致，且在高维时因子矩阵秩不足导致恢复失稳。

**⚠️ 局限性**

局限性包括：仅适用于仅依赖二阶矩的算法；在高维（d≳15）因子矩阵秩聚集导致数值不稳定；未在真实数据上验证；常数项未给出精确闭式，实际数据中可能更复杂。

---

## 436. Are Deep Learning Based Hybrid PDE Solvers Reliable? Why Training Paradigms and Update Strategies Matter

**arXiv ID:** 2602.06842 | [PDF](https://arxiv.org/pdf/2602.06842v1)

**作者:** Yuhan Wu `[一作]` (Delft University of Technology), Alexander Heinlein `[通讯]` (Delft University of Technology)

**通讯引用:** 662 | [OpenAlex ID](https://openalex.org/A5081654668)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统评估深度学习混合迭代方法（DL‑HIM），揭示其在不同训练方案和更新策略下的收敛问题。

**💡 创新点**

提出物理感知的安德森加速（Physics‑aware Anderson Acceleration, PA‑AA），专门针对物理残差而非更新幅度，显著消除假固定点。

**🔧 技术方法**

使用DeepONet与FFT神经算子、Jacobi预条件器、线性/非线性安德森加速、以及自适应步长/固定步长等技术。

**📊 数据集**

基于高斯随机场生成的二维参数场和右端项，构造扩散方程与Helmholtz方程的训练与测试集。

**📈 对比分析**

通过对比静态与动态训练、误差基准与残差基准、标准AA与PA‑AA，在一维扩散和Helmholtz实验中发现PA‑AA可在更少迭代内将残差降至10⁻⁹，性能远优于传统方案。

**⚠️ 局限性**

局限在于仅验证单维问题，且对非对称/非正定系统的自适应步长仍表现不稳定，需进一步扩展到更高维和更复杂模型。

---

## 437. DAWN: Dependency-Aware Fast Inference for Diffusion LLMs

**arXiv ID:** 2602.06953 | [PDF](https://arxiv.org/pdf/2602.06953v1)

**作者:** Lizhuo Luo `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2805 | [OpenAlex ID](https://openalex.org/A5028270700)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个训练无关的依赖感知快速推理方法（DAWN）用于扩散式大语言模型（dLLM）的并行解码。

**💡 创新点**

创新点在于通过注意力图构建稀疏依赖图，利用高置信度的已解码位置作为锚点放宽置信度阈值，同时通过冲突调度避免低置信度强耦合位置同时解码，从而显著提升并行度。

**🔧 技术方法**

核心技术包括：注意力图聚合与去除注意力沉积、依赖图构建、Anchor‑Guided Decoding 与 Conflict‑Based Scheduling、无训练的推理策略。

**📊 数据集**

在多模型（LLaDA‑8B‑Instruct、LLaDA‑1.5、Dream‑v0‑Base‑7B、Dream‑v0‑Instruct‑7B）与多数据集（GSM8K、MATH、HumanEval、MBPP）上进行评估。

**📈 对比分析**

与四个基线（原始Top‑1采样、Fast‑dLLM、KLASS、LocalLeap）比较，DAWN 在保持或略高的准确率下，平均速度提升 1.80–8.06×，尤其在 MBPP 上可达 8.06× 的加速。

**⚠️ 局限性**

局限性包括：仍需在极长文本或高度耦合任务中调优阈值；依赖注意力图的近似可能在某些模型或训练阶段表现不佳；对硬件加速的兼容性和内存占用未作深入分析。

---

## 438. Optimal Derivative Feedback Control for an Active Magnetic Levitation System: An Experimental Study on Data-Driven Approaches

**arXiv ID:** 2602.06944 | [PDF](https://arxiv.org/pdf/2602.06944v1)

**作者:** Saber Omidi `[一作]` (University of New Hampshire), Se Young Yoon `[通讯]` (University of New Hampshire)

**通讯引用:** 723 | [OpenAlex ID](https://openalex.org/A5031215547)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了针对主动磁悬浮系统的基于数据驱动的最优导数反馈控制器，比较了直接无模型策略迭代方法和间接基于系统识别的模型驱动方法。

**💡 创新点**

创新点在于：①引入多周期(epoch)的策略迭代，显著降低训练偏差并提升鲁棒性；②在无模型条件下通过强化学习实现最优导数反馈控制；③通过与基于DMDc-PEM识别的模型驱动方案对比，验证了多周期策略迭代的优势。

**🔧 技术方法**

使用技术包括：强化学习中的策略迭代（Policy Iteration）与值迭代、导数反馈控制（Derivative Feedback Control）、动态模态分解与控制（DMDc）、预测误差最小化（PEM）、系统识别、最优控制理论（LQR、ARE）。

**📊 数据集**

实验数据来自双磁盘主动磁悬浮系统（MagLev Model 730），包括在仿真环境下的采样数据以及实际实验中的四秒训练数据与四个周期的测试数据；实验采用 1 ms 采样、低通滤波与激励噪声。

**📈 对比分析**

比较方法：对同一系统在仿真与实验中使用三类控制器（无模型策略迭代、基于DMDc-PEM模型的导数反馈、基于标称模型的最优控制）进行成本函数评估、收敛速度与控制输入幅度比较。结果显示：①无模型策略迭代在每个 epoch 收敛到最低成本；②与间接方案相比，策略迭代在实验中获得更低的成本与更小的控制能量；③导数反馈控制显著优于传统状态反馈（LQR）在应对平衡点偏差与测量偏差时的性能。

**⚠️ 局限性**

局限性：①算法的计算复杂度随系统阶数立方增长，难以直接扩展到高阶系统；②缺乏严格的鲁棒性裕度保证，需在实践中通过持续激励与噪声抑制来保持收敛；③对激励信号的持久激励假设不易满足，可能影响训练效果；④实验仅验证了四阶模型，对更复杂磁悬浮结构的适用性尚待研究。

---

## 439. Endogenous Resistance to Activation Steering in Language Models

**arXiv ID:** 2602.06941 | [PDF](https://arxiv.org/pdf/2602.06941v1)

**作者:** Alex McKenzie `[一作]` (AE Studio), Michael S. A. Graziano `[通讯]` (Princeton University)

**通讯引用:** 14265 | [OpenAlex ID](https://openalex.org/A5007961852)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在推理过程中对激活操控的内生抵抗（ESR）现象，并通过提示与微调手段对其进行增强。

**💡 创新点**

首次提出ESR概念，揭示模型具有内在一致性检测与自我纠正机制，并证明可通过元提示和微调诱导该行为。

**🔧 技术方法**

使用稀疏自编码器（SAE）进行激活操控、元提示、LoRA微调，利用Claude 4.5 Haiku等Judge模型评估输出，分析不同潜变量的零化效果。

**📊 数据集**

对Llama‑3与Gemma‑2系列模型进行实验，使用“explain how”提示集和随机挑选的SAE潜变量作为激活干预对象。

**📈 对比分析**

在不同模型、提示、潜变量零化与微调条件下比较多次尝试率、ESR率和得分提升；Llama‑3.3‑70B的ESR率最高约1%，元提示可提升4倍，微调可诱导自纠但成功率保持不变。

**⚠️ 局限性**

仅使用单层SAE限制了跨层机制解析；实验样本受限于少数模型与提示，评估依赖Judge模型主观评分，潜变量筛选可能过拟合，缺乏更广泛的架构与规模对比。

---

## 440. Reliable Mislabel Detection for Video Capsule Endoscopy Data

**arXiv ID:** 2602.06938 | [PDF](https://arxiv.org/pdf/2602.06938v1)

**作者:** Julia Werner `[一作]` (University of Tübingen), Oliver Bringmann `[通讯]` (University of Tübingen)

**通讯引用:** 3094 | [OpenAlex ID](https://openalex.org/A5074802358)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出并验证了一套用于视频胶囊内镜数据集的误标检测与清洗框架，旨在提升模型在医学图像中的异常检测性能。

**💡 创新点**

创新点在于将高斯混合模型与多次CNN训练相结合，既实现标签纠正又实现样本过滤，并通过医学专家复核验证其有效性。

**🔧 技术方法**

采用了MobileNetV3网络、焦点损失（Focal Loss）、GMM对损失分布建模、并利用熵与置信度构造不确定度分数。

**📊 数据集**

主要使用了两大公开VCE数据集：Kvasir‑Capsule（用于噪声注入实验）和Galar（用于实际误标检测与清洗）。

**📈 对比分析**

与原始未清洗数据及现有基线模型比较，清洗后模型在Galar数据集上准确率提升至93.83%、F1得分提升至71.58%，显著超过原始数据的54.38%基线。

**⚠️ 局限性**

局限性包括临床验证样本量有限（仅100张），误标检测依赖阈值设定，且在不同设备或病变类型下的泛化性尚待进一步评估。

---

## 441. A first realization of reinforcement learning-based closed-loop EEG-TMS

**arXiv ID:** 2602.06907 | [PDF](https://arxiv.org/pdf/2602.06907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 442. Reciprocal Latent Fields for Precomputed Sound Propagation

**arXiv ID:** 2602.06937 | [PDF](https://arxiv.org/pdf/2602.06937v1)

**作者:** Hugo Seuté `[一作]` (Ubisoft La Forge), Louis-Xavier Buffoni `[通讯]` (Audiokinetic)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发并评估了一种基于隐空间嵌入和Riemannian度量的 Reciprocal Latent Field（RLF）框架，用于实时游戏场景中的声学传播预计算与压缩。

**💡 创新点**

引入隐空间度量学习，使声学路径距离和其他参数可在对称隐空间中以几何方式编码，并通过局部 Riemannian 度量实现对障碍物的自适应变形，显著降低内存占用并保持物理对称性。

**🔧 技术方法**

使用三维网格隐向量、三线性插值、欧氏/全正定/对角 Riemannian 解码器、MLP 解码、对称约束、梯度停止、PFFDTD 波模拟、波编码参数提取以及 MUSHRA 听感实验。

**📊 数据集**

在两套游戏地图（自制 Audio Gym 与 Wwise Audio Lab）上采集多源多接收的波模拟参数（路径距离、直接/早晚反射强度、衰减时间、到达方向），并通过自适应源采样生成训练集。

**📈 对比分析**

通过在测试源上计算 MAE、DOA 误差、FLOPs、内存，并与传统波编码对比；实验显示 RLF G_PSD/DIAG 在误差 <0.2 m、<0.5 dB、<0.05 s 的同时将内存从 GB 降至 MB，并在主观 MUSHRA 实验中与真值无显著差异。

**⚠️ 局限性**

仅适用于静态几何，未实现动态场景的在线更新；缺乏压缩实验，且对极端复杂波形的噪声鲁棒性有限。

---

## 443. When RL Meets Adaptive Speculative Training: A Unified Training-Serving System

**arXiv ID:** 2602.06932 | [PDF](https://arxiv.org/pdf/2602.06932v1)

**作者:** Junxiong Wang `[一作]`, Xiaoxia Wu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

建立了一个统一的训练-推理系统，实时在生产流量中对speculator进行在线学习与自适应，支持day‑0部署；

**💡 创新点**

创新点在于将speculative decoding的训练与服务闭环化，采用异步RL框架、接受/拒绝双向反馈、热更新机制，并实现了低延迟、无停机的同步策略；

**🔧 技术方法**

技术包括SGLang推理服务器、GPU‑aware RPC、异步训练服务器、Tree Attention机制、RL‑style接受/拒绝损失、Lazy同步策略以及针对混合模型的EAGLE‑3框架；

**📊 数据集**

使用多域Prompt流（约40k条，涵盖数学推理、text‑to‑SQL、代码生成、金融、对话等）进行在线仿真，并在MiniMax M2.1（229B）和Qwen3‑Coder‑Next（80B）等开源大模型上评测；

**📈 对比分析**

与无speculation、静态预训练speculator等基线对比，day‑0从随机初始化的speculator在数千请求后即可达到或超过预训练模型的acceptance length；在MiniMax和Qwen3实验中分别实现了约1.45×与1.21×的吞吐提升，且在小批量下收益更显著；

**⚠️ 局限性**

局限主要包括同步频率对吞吐的影响、过频更新导致的延迟波动、在大型专家路由模型中speculation收益有限，以及对混合模型多步核优化的支持尚不完善。

---

## 444. Continuous-time reinforcement learning: ellipticity enables model-free value function approximation

**arXiv ID:** 2602.06930 | [PDF](https://arxiv.org/pdf/2602.06930v1)

**作者:** Wenlong Mou `[一作]` (University of Toronto), Wenlong Mou `[通讯]` (University of Toronto)

**通讯引用:** 448 | [OpenAlex ID](https://openalex.org/A5006742082)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出一种基于连续时间马尔可夫扩散的离策略无模型强化学习框架，利用梯度空间几何结构直接从离散观测数据学习价值函数与优势函数。

**💡 创新点**

核心创新是将扩散过程的椭圆性（diffusion矩阵的正定性）转化为Bellman算子在Sobolev空间上的正定性与有界性，进而构造Sobolev‑prox拟合Q学习算法，并给出非渐近的oracle不等式，证明在椭圆性条件下函数逼近与统计误差可按监督学习理论的速率控制。

**🔧 技术方法**

利用Itô公式、椭圆型偏微分方程的正定性理论、Sobolev空间投影、链式推导与局部复杂度（Dudley积分）等工具完成误差分解和收敛性证明；算法实现上通过最小二乘回归与Sobolev范数的经验近似实现。

**📊 数据集**

无实验数据集，论文仅给出理论分析与推导；若要验证需自行采集或仿真控制扩散过程数据。

**📈 对比分析**

主要与传统离散时间Fitted Q迭代或基于投影Bellman方程的固定点方法做对比，指出后者在Sobolev几何下收敛更快、误差更小；理论上误差上界可与非参数回归匹配，但论文未给出实验或数值比较。

**⚠️ 局限性**

限制包括：
• 需假设扩散矩阵严格椭圆且系数满足高阶光滑性；
• 需要先验对行为策略的覆盖性假设；
• 只关注离策略学习，未处理探索；
• 目前仅适用于凸/可线性参数化的函数类，未讨论非凸神经网络实现；
• 需要较大折扣率/较小时间步长以保证有效地平，且理论中保守的常数与日志因子。

---

## 445. MedMO: Grounding and Understanding Multimodal Large Language Model for Medical Images

**arXiv ID:** 2602.06965 | [PDF](https://arxiv.org/pdf/2602.06965v1)

**作者:** Ankan Deria `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Imran Razzak `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 10442 | [OpenAlex ID](https://openalex.org/A5033585021)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了MedMO医学多模态基础模型，并提出了四阶段后训练流程（大规模对齐→高分辨率微调→指令微调→强化学习），实现跨模态视觉定位与临床推理；

**💡 创新点**

创新点在于：①利用高分辨率医学图像与细粒度边界框监督提升视觉定位；②引入可验证的边界框奖励（BGR）结合RLGRPO强化模型的事实性与空间推理；③构建跨45个公开医学多模态数据集的26M+样本统一语料，打通多种影像模态；

**🔧 技术方法**

技术路线基于Qwen3-VL-8B架构，采用跨模态预训练、视觉语言适配器+DeepStack融合、指令微调、RLGRPO强化学习，并加入GIoU奖励和多维度奖励信号；

**📊 数据集**

使用26M+样本的45个公开医学多模态数据集，包括MedTrinity、Chest X‑ray、CT、MRI、超声、病理、眼科、皮肤科、手术视频等，以及细菌分割微镜数据；

**📈 对比分析**

通过与多种开源（MedVLM‑R1‑2B、HuatuoGPT‑V‑7B、Qwen2.5‑VL‑7B等）和闭源基线（GPT‑4.1、Claude、Gemini）在VQA、文本QA、报告生成、定位（NIH、DeepLesion、Bacteria、MedSG）等任务上对比，MedMO‑8B在MMMU‑Med、MedQA、MIMIC‑CXR、MedTrinity等指标均超过或等同SOTA，尤其在细菌分割IoU提升43.8点；

**⚠️ 局限性**

主要限制是四阶段训练会出现跨任务性能波动和轻微灾难性遗忘，需要进一步提升SFT知识在RL阶段的保留和跨任务鲁棒性。

---

## 446. Understanding Workplace Relatedness Support among Healthcare Professionals: A Four-Layer Model and Implications for Technology Design

**arXiv ID:** 2602.06916 | [PDF](https://arxiv.org/pdf/2602.06916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 447. Agentic Uncertainty Reveals Agentic Overconfidence

**arXiv ID:** 2602.06948 | [PDF](https://arxiv.org/pdf/2602.06948v1)

**作者:** Jean Kaddour `[一作]` (University College London), Matt J. Kusner `[通讯]` (Mila)

**通讯引用:** 5215 | [OpenAlex ID](https://openalex.org/A5108523545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并量化AI代理在任务前、中、后对自身成功概率的自我估计，揭示其普遍的过度自信；

**💡 创新点**

首次将“代理性不确定性”定义为P(IS)，并对不同评估时机和对抗性提示进行系统比较，发现预评估能更好区分成功与失败，而对抗性评估能显著提升校准；

**🔧 技术方法**

采用大型语言模型（GPT‑5.2‑Codex、Gemini‑3‑Pro、Claude‑Opus‑4.5）进行提示式估计，使用工具调用获取代码上下文，并通过AUROC、ECE、Brier等指标评估；

**📊 数据集**

使用SWE‑Bench Pro的100个多文件软件工程任务作为基准数据集；

**📈 对比分析**

与三种模型的三种评估策略（pre‑exec、post‑exec、adversarial post‑exec）相互比较，结果显示所有模型在后评估上均存在高达55个百分点的过度自信；预评估在判别度上略优，且对抗性评估在校准上最为优秀；

**⚠️ 局限性**

研究仅局限于可客观评价的软件编码任务，样本量有限；缺乏针对性训练的评估器；对抗提示虽能改善校准但并未根除过度自信；未探讨不同领域或更大规模模型的普适性与可扩展性。

---

## 448. Cochain Perspectives on Temporal-Difference Signals for Learning Beyond Markov Dynamics

**arXiv ID:** 2602.06939 | [PDF](https://arxiv.org/pdf/2602.06939v1)

**作者:** Zuyuan Zhang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6388 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种新的拓扑视角来研究非马尔可夫动态下的时序差分（TD）强化学习，展示了TD误差可以被视为状态转移的拓扑空间中的1-共链，并通过贝尔曼-德拉姆投影获得TD误差的霍奇分解。

**💡 创新点**

创新点在于将TD误差与拓扑整合性联系起来，提出了霍奇流策略搜索（HFPS），通过拟合潜在网络来最小化非整合投影残差，从而实现稳定性和敏感性保证。

**🔧 技术方法**

使用了拓扑学和希尔伯特空间的工具，特别是引入了霍奇类型分解和贝尔曼-德拉姆投影。

**📊 数据集**

在实验中使用了合成的马尔可夫决策过程（MDP）和深度控制基准，特别是在非马尔可夫环境中进行评估。

**📈 对比分析**

与标准TD学习方法相比，HFPS在非马尔可夫奖励、部分可观察和依赖动态下表现出显著的性能提升，尤其在稳定性和鲁棒性方面。

**⚠️ 局限性**

限制在于理论分析和算法设计主要集中在特定的非马尔可夫环境下，可能在其他类型的环境中表现不佳。

---

## 449. Implementing Grassroots Logic Programs with Multiagent Transition Systems and AI

**arXiv ID:** 2602.06934 | [PDF](https://arxiv.org/pdf/2602.06934v1)

**作者:** Ehud Shapiro `[一作]` `[通讯]` (London School of Economics and Weizmann Institute of Science), Ehud Shapiro (London School of Economics and Weizmann Institute of Science)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文设计并证明了单代理和多代理Grassroots Logic Programs的确定性操作语义 dGLP 与 madGLP，并基于此实现了工作站与智能手机上的分布式执行。

**💡 创新点**

创新点在于将单写单读线性逻辑变量与 futures/promises 结合，提出使用全局链接实现跨代理共享变量，并通过仿真映射完成从抽象语义到可实现语义的证明。

**🔧 技术方法**

采用转移系统、仿真与最小化替换、FIFO调度、全局写入表、消息传递等技术，并在 Dart 语言中实现。

**📊 数据集**

本文未使用传统数据集，而是针对语言语义和实现进行理论与实验验证。

**📈 对比分析**

通过形式化证明（模拟映射、持久性、消除子替换可交换性）验证实现正确性，未给出性能基准。

**⚠️ 局限性**

主要限制包括依赖可靠异步消息、需要公平运行、实现规模与性能尚未评估、且仅支持单写单读的线性变量。

---

## 450. PANC: Prior-Aware Normalized Cut for Object Segmentation

**arXiv ID:** 2602.06912 | [PDF](https://arxiv.org/pdf/2602.06912v1)

**作者:** Juan Gutiérrez `[一作]` (Universidad Politécnica de Madrid), José Luis Blanco-Murillo `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 135 | [OpenAlex ID](https://openalex.org/A5039858580)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 PANC 框架，利用极少量手工标注的视觉 token 作为锚点注入 ViT 生成的稠密 token 关联图中，完成稳定、可控的像素级分割。

**💡 创新点**

创新点在于把手工标注的 token 作为 anchor 节点直接加入归一化割谱图，显式偏置谱空间以消除对显著性假设的依赖，同时保持全局一致性。

**🔧 技术方法**

技术包括冻结 DINOv3 ViT 生成稠密 token、余弦相似度构建权重图、引入 anchor 连接后求解归一化割的 Fiedler 向量、并通过 ROC 定位阈值得到二值掩码。

**📊 数据集**

实验数据集覆盖标准显著性基准 ECSSD、DUTS、DUT‑OMRON，多类别 COCO，细粒度 CUB‑200‑2011，纹理受限 CrackForest 以及医学皮肤病变 HAM10000。

**📈 对比分析**

与无监督方法（TokenCut、LOST 等）和弱监督方法（WSCUOD、PFENet 等）对比，PANC 在所有数据集上均实现最高或接近最高的 mIoU，尤其在 CrackForest 上提升超过 14.5%。

**⚠️ 局限性**

局限性包括对手工标注 token 的依赖、对锚点数目与温度等超参数敏感，以及先验库构建和大规模图构建的计算瓶颈。

---

## 451. TamperBench: Systematically Stress-Testing LLM Safety Under Fine-Tuning and Tampering

**arXiv ID:** 2602.06911 | [PDF](https://arxiv.org/pdf/2602.06911v1)

**作者:** Saad Hossain `[一作]` (University of Waterloo), Sirisha Rambhatla `[通讯]` (University of Waterloo)

**通讯引用:** 644 | [OpenAlex ID](https://openalex.org/A5018625427)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了名为 TamperBench 的统一框架，用以系统评估开放权重大型语言模型在权重空间和表征空间被篡改后的安全性和实用性，并基于该框架对 21 种模型在 9 种攻击方式下进行了大规模实验。

**💡 创新点**

创新点在于：①首次将多种权重级与表征级篡改攻击与安全/能力评估标准统一到一个可扩展的基准；②引入系统化的超参数搜索以模拟真实攻击者并减少实验敏感性；③通过结合 StrongREJECT 和 MMLU‑Pro 指标，实现对安全（拒绝率）与实用性（能力）双重评估。

**🔧 技术方法**

采用的技术包括 LoRA 与全参数微调、背门式篡改、风格调制、竞争目标 Jailbreak‑tuning、跨语言微调、表征扰动攻击；在实验中使用 Optuna 进行 40 次超参数搜索，利用 HuggingFace 训练流水线和多 GPU 并行；安全评估用 StrongREJECT 判别器，实用性评估用 MMLU‑Pro 子集。

**📊 数据集**

使用的数据集涵盖：StrongREJECT（危险请求集与判别器）、MMLU‑Pro（能力基准）、各攻击特定数据集（如 2% 有害混入 98% 良性、5000 条 jailbreak‑tuning 样本、64 条 LoRA 有害样本等），以及在附录中提及的更多实验数据。

**📈 对比分析**

比较方法：在每个模型-攻击对中，先进行超参数搜索以最大化 StrongREJECT 得分，同时限制 MMLU‑Pro 减损 ≤10%；随后统计最大后攻击得分 SR_max 与所有恶意攻击的平均 SR_mal‑avg。实验结果显示：所有模型均能被篡改至 SR_max > 0.68；Jailbreak‑tuning 是最强攻击；Triplet 与 TAR 在 Llama‑3‑8B‑Instruct 上显著降低了后攻击安全得分；不同模型族（如 Qwen3 vs Llama‑3）在基线与后攻击表现上存在系统差异。

**⚠️ 局限性**

局限性包括：①仅使用 MMLU‑Pro 的 140 条子集评估实用性，可能不充分反映全部能力；②攻击实现主要遵循先前工作设定的训练样本规模，未对样本量、构成做全面搜索；③仅考察拒绝型安全防御，未涉及无知型方法；④只在 Llama‑3‑8B‑Instruct 上评估五种对齐阶段防御，缺乏跨模型族的泛化验证。

---

## 452. Optimal Turkish Subword Strategies at Scale: Systematic Evaluation of Data, Vocabulary, Morphology Interplay

**arXiv ID:** 2602.06942 | [PDF](https://arxiv.org/pdf/2602.06942v1)

**作者:** Duygu Altinok `[一作]` `[通讯]` (Independent Researcher), Duygu Altinok (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了土耳其语形态丰富语言的子词分词，系统地将词表大小与训练语料规模耦合，并提出细粒度的形态诊断工具；

**💡 创新点**

首次将词表与语料规模耦合、提供形态层面微观宏观 F1、过/欠分词、CER/WER 等诊断指标，并给出可复现的代码与模型；

**🔧 技术方法**

采用 WordPiece、Morphology‑aware 词典分词、字符级分词等多种分词器，结合 Transformer 预训练与多任务评测；

**📊 数据集**

使用 OSCAR‑TR、BellaTurca、BOUN 语料库、TrGLUE、NER、POS/DEP/形态等多任务数据集；

**📈 对比分析**

通过对不同词表大小（2k‑128k）和语料规模（5GB‑80GB）进行系统对比，发现 20k‑32k 词表在中等数据规模下既能保持较低的序列长度，又能在语义、形态与句法任务上获得与大词表相近甚至更优的性能；

**⚠️ 局限性**

仍受限于过度分词导致的细粒度噪声、极大词表下的优化不稳定以及对低资源下的泛化能力不足等问题。

---

## 453. From Core to Detail: Unsupervised Disentanglement with Entropy-Ordered Flows

**arXiv ID:** 2602.06940 | [PDF](https://arxiv.org/pdf/2602.06940v1)

**作者:** Daniel Galperin `[一作]` (Heidelberg University), Ullrich Köthe `[通讯]` (Heidelberg University)

**通讯引用:** 3689 | [OpenAlex ID](https://openalex.org/A5110454166)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种基于熵排序的无监督解耦正则化框架 EOFlows，能够在训练后动态选择核心维度进行压缩与去噪。

**💡 创新点**

创新点在于将熵排序与最大流形似然、全局解耦正则相结合，首次实现可调核心/细节瓶颈，同时保持高压缩率、强去噪和语义可解释性。

**🔧 技术方法**

使用技术包括：归一化流（Normalizing Flow）、最大流形似然（MML）训练目标、全相关（Total Disentanglement）正则、噪声膨胀与降噪、Jacobian 向量积估计等。

**📊 数据集**

实验数据集包括 EMNIST 数字、合成的 Entangled Digits、CelebA 人脸图像。

**📈 对比分析**

与标准 NF、PCA、β‑VAE 等方法对比，EOFlows 在压缩率–失真曲线、PSNR 去噪性能上表现更好，且能用少量核心维度重构高质量图像。

**⚠️ 局限性**

局限性：需要批量大小≥维度以估计全相关；噪声水平与正则参数需手工调节；对非高斯真实分布的泛化能力尚未验证。

---

## 454. On the Efficiency of Sequentially Aware Recommender Systems: Cotten4Rec

**arXiv ID:** 2602.06935 | [PDF](https://arxiv.org/pdf/2602.06935v1)

**作者:** Shankar Veludandi `[一作]` (Rensselaer Polytechnic Institute), Uzma Mushtaque `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5041421119)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Cotten4Rec 模型，用余弦相似度注意力替代传统 Softmax 注意力，显著降低了顺序推荐系统的显存与计算开销。

**💡 创新点**

创新点在于将 L₂ 归一化的余弦注意力与单核 CUDA 计算融合，实现线性时间 O(s·d²) 的注意力计算，并通过一个高效核消除中间缓冲区，提升了内存利用率。

**🔧 技术方法**

使用了 BERT4Rec 结构、余弦注意力、单核 CUDA 内核、混合精度训练（AMP）以及基于 PyTorch 的实现。

**📊 数据集**

在 Amazon Beauty、MovieLens‑1M 和 MovieLens‑20M 三个真实数据集上进行实验。

**📈 对比分析**

与 BERT4Rec（SOTA）和 LinRec（线性注意力基线）对比，Cotten4Rec 在大多数场景下显存下降约 23%，训练速度提升 4%–20%，且 NDCG@10 与 HIT@10 与基线相差不超过 2%。

**⚠️ 局限性**

局限性包括：在极长序列（如 ML‑1M）时因余弦核的常数因子导致训练慢 49%；余弦注意力对长序列的分辨率有限，导致推荐精度下降；以及对自定义 CUDA 内核的依赖降低了跨平台可移植性。

---

## 455. Learning a Generative Meta-Model of LLM Activations

**arXiv ID:** 2602.06964 | [PDF](https://arxiv.org/pdf/2602.06964v1)

**作者:** Grace Luo `[一作]` (University of California Berkeley), Jacob Steinhardt `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练扩散模型学习大型语言模型残差激活分布，并将其作为先验用于在激活空间的“在流形上”干预和概念抽取。

**💡 创新点**

创新点在于将连续扩散模型直接应用于高维激活数据，既不依赖线性或稀疏假设，也不需显式重建，能够天然捕获激活流形结构并提供可推广的计算扩展性。

**🔧 技术方法**

核心技术包括：扩散损失的流匹配训练、基于 Llama MLP 结构的无条件 MLP 去噪器、时间步条件化、以及利用 Frechet 距离和 1‑D 探测评估激活质量。

**📊 数据集**

数据集主要为 FineWeb 1 B tokens（用于提取 1 B Llama1B 或 3 B 参数模型的中间层残差激活），以及 Llama8B 的中间层激活；在实验中还使用 OpenWebText、Persona 向量等下游任务数据。

**📈 对比分析**

与传统的稀疏自编码器（SAE）和原始激活做对比：Frechet 距离从 1.99 降至 0.53，Delta LM Loss 下降到 0.051，1‑D 探测 AUC 由 SAE 的 0.70 提升至 0.84（Llama1B）或 0.87（Llama8B）；在情感控制、特征调度和人物体现等任务中，扩散后处理提升了概念‑流畅度 Pareto 前沿。

**⚠️ 局限性**

局限性包括：只建模单 token 的独立激活，无法捕捉跨位置结构；模型为无条件，未对清洁激活做显式条件化；仅聚焦单层残差激活，未探索多层或其他激活类型。

---

## 456. CineScene: Implicit 3D as Effective Scene Representation for Cinematic Video Generation

**arXiv ID:** 2602.06959 | [PDF](https://arxiv.org/pdf/2602.06959v1)

**作者:** Kaiyi Huang `[一作]` (University of Hong Kong), Xihui Liu `[通讯]` (University of Hong Kong)

**通讯引用:** 3800 | [OpenAlex ID](https://openalex.org/A5027234036)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于隐式三维场景表示的电影级视频生成框架，能够在给定静态场景图像、文字提示和用户指定摄像机轨迹的情况下，合成具有动态主体、保持场景一致性且摄像机控制精准的高质量视频。

**💡 创新点**

核心创新在于将VGGT提取的隐式3D特征通过上下文条件注入到预训练的文本到视频扩散模型中，摆脱了显式几何重建的需求，并通过随机打乱场景图像顺序的策略提升模型对场景和3D信息的对齐；同时构建了Scene‑Decoupled视频数据集，为训练提供了严格分离的静态背景与动态主体对。

**🔧 技术方法**

技术包括VGGT无监督的3D特征提取、上下文条件注入（context conditioning）、基于Transformer的Causal 3D VAE视频编码、摄像机轨迹注入、随机图像打乱策略以及在Unreal Engine 5上生成的高质量渲染数据。

**📊 数据集**

使用自建的Scene‑Decoupled Video Dataset（46K帧视频/场景对，35个3D环境，包含静态与动态视频、全景图像及摄像机轨迹）以及少量DiT360等外域样本进行评估。

**📈 对比分析**

与FramePack、Context‑as‑Memory、Gen3C、Traj‑Attn等基线在场景一致性、摄像机精度、文本对齐和视频质量上进行比较，实验表明该方法在所有指标上均超越基线，尤其在大视角变化下的场景一致性与摄像机跟踪表现最为显著。

**⚠️ 局限性**

局限性包括：训练数据主要来自合成环境，可能对真实复杂光照和材质的泛化有限；对极端大尺度场景或高度动态对象的处理尚需进一步验证；同时模型对摄像机内参的假设较为简化，需在实际应用中加以调整。

---

## 457. Directing Space: Rehearsing Architecture as Performer with Explainable AI

**arXiv ID:** 2602.06915 | [PDF](https://arxiv.org/pdf/2602.06915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 458. Robustness Beyond Known Groups with Low-rank Adaptation

**arXiv ID:** 2602.06924 | [PDF](https://arxiv.org/pdf/2602.06924v1)

**作者:** Abinitha Gourabathina `[一作]` (Massachusetts Institute of Technology), Collin Stultz `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3577 | [OpenAlex ID](https://openalex.org/A5024941370)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种低秩误差信息自适应方法（LEIA），通过识别模型误差集中的低维子空间并在该子空间内对分类层进行低秩调整，以提升未标注或未知子群体的鲁棒性。

**💡 创新点**

创新点在于不依赖子群标签即可发现错误模式所在的几何子空间，并通过低秩修正仅在 logits 空间内进行局部微调，实现参数效率与计算轻量化。

**🔧 技术方法**

技术上采用两阶段流程：①使用 ERM 训练基模型并冻结特征提取器；②对保留的数据计算误差加权协方差，提取 top‑k 主成分；③学习一个 k×C 的低秩矩阵 A，在 logits 处加上 A^T V_k^T e(x)。

**📊 数据集**

实验数据集包括五个真实世界分类任务：Waterbirds、CelebA、CivilComments、MultiNLI、CheXpert。

**📈 对比分析**

与 12 种基线（如 Group DRO、CRT、JTT、DPE、AFR、LISA 等）对比，LEIA 在三种子群信息缺失情形（无信息、部分信息、完整信息）下均取得最优或次优的 worst‑group accuracy，且平均准确率基本不下降。

**⚠️ 局限性**

局限性在于仍需在验证集上调参（k、γ），且对极端数据分布偏移或完全不相关的子群体的适应性尚待进一步验证。

---

## 459. From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers

**arXiv ID:** 2602.06923 | [PDF](https://arxiv.org/pdf/2602.06923v1)

**作者:** Ziming Liu `[一作]` (Stanford University), Andreas Tolias `[通讯]` (Stanford University)

**通讯引用:** 18381 | [OpenAlex ID](https://openalex.org/A5054637446)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文探讨并解决了Transformer在学习行星运动的物理世界模型时的三大失败模式：空间平滑、空间稳定性和时间局部性。

**💡 创新点**

创新点在于通过三种最小先验（空间连续回归、带噪声上下文训练、限制注意力窗口）实现Transformer从仅拟合曲线（Kepler模型）跃迁到发现牛顿力学（Newton模型）的能力，并揭示上下文长度是控制模型类型的关键参数。

**🔧 技术方法**

使用了Transformer自回归模型、连续坐标回归（MSE）、带噪声上下文学习、线性探测器以及基于Token化的分类回归对比实验。

**📊 数据集**

主要使用人工合成的Kepler轨道数据集（理想化椭圆轨道）以及用于评估空间映射的1D正弦波数据集。

**📈 对比分析**

与传统基于token化的分类Transformer对比，噪声上下文训练的回归Transformer在不同训练规模、词表大小和上下文长度下表现更好；在大上下文长度下能实现更低预测误差，而在小上下文长度下能精准捕捉牛顿力学，R²接近1。

**⚠️ 局限性**

限制在于实验仅在极简化的合成数据上验证，使用线性探测方法只能间接证明模型已“理解”力学，缺乏完全自动化的符号推理与模型解释能力；对复杂多尺度或实际观测数据的泛化能力尚未验证。

---

## 460. Halluverse-M^3: A multitask multilingual benchmark for hallucination in LLMs

**arXiv ID:** 2602.06920 | [PDF](https://arxiv.org/pdf/2602.06920v1)

**作者:** Samir Abdaljalil `[一作]` (Texas A&M University), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 453 | [OpenAlex ID](https://openalex.org/A5070970331)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个多语言、多任务的精细化幻觉检测数据集（HalluVerse‑M3），并在其上对多种大型语言模型的幻觉检测能力进行了系统评估。

**💡 创新点**

创新点在于：① 兼顾四种语言（英、阿、印、土）和两大生成任务（问答、对话摘要）；② 将幻觉细分为实体、关系和句子级别；③ 通过受控编辑与人工验证构造高质量幻觉实例；④ 提供公开的基准结果，方便后续研究。

**🔧 技术方法**

技术手段包括：受控自动编辑（利用LLM生成幻觉句子）、多语言翻译与人工校对、统一的幻觉类型标签体系、基于prompt的无微调模型评测（从Phi‑4到GPT‑4o等）。

**📊 数据集**

使用了自定义构造的四语言问答与对话摘要数据，结合人工标注的幻觉标签；原始数据来源于公开的问答与对话摘要数据集，并通过机器翻译后人工审核得到。

**📈 对比分析**

通过对比多种开源和闭源模型（如Phi‑4、Mistral、Gemma、Qwen、LLaMA、DeepSeek、PaLM、Claude、Gemini、GPT‑4系列），在四种语言下分别评估了整体幻觉检测准确率及按类型（实体/关系/句子）的准确率。结果显示：① 问答任务整体性能高于摘要任务；② 英文最优，印语表现最差；③ GPT‑4o与GPT‑4.1在所有任务与语言上表现最佳；④ 句子级幻觉是最难检测的类别。

**⚠️ 局限性**

局限性包括：① 数据集中仍有部分实例被标记为“0”或因翻译错误被排除，可能影响多语言平衡；② 受控编辑仅限单一幻觉，未覆盖多重幻觉场景；③ 评测仅基于prompt，无内部表示或微调，可能低估模型潜力；④ 语言多样性覆盖有限，仅包括四种语言，未涵盖低资源或非印欧语言。

---

## 461. Revisiting the Generic Transformer: Deconstructing a Strong Baseline for Time Series Foundation Models

**arXiv ID:** 2602.06909 | [PDF](https://arxiv.org/pdf/2602.06909v1)

**作者:** Yunshi Wen `[一作]` (Rensselaer Polytechnic Institute), Anak Agung Julius `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 2698 | [OpenAlex ID](https://openalex.org/A5029921181)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一种通用 Patch Transformer 作为时间序列基础模型，证明其在 GIFT‑Eval 零样本概率预测基准上能够实现或超过最新的专用架构。

**💡 创新点**

关键创新在于（1）将通用 Transformer 与 patch‑token 化、连续补丁遮掩（CPM）、遮罩感知归一化以及量化头等技术组合成一个完整的自监督预训练流程；（2）通过大规模、系统的消融实验明确了模型性能的主要驱动因素——数据多样性、CPM 与更长的上下文窗口，而非单纯的网络结构改进。

**🔧 技术方法**

采用标准 Transformer 编码器（深度、宽度可调），Patch 嵌入 + 位置嵌入，CPM 掩码策略，遮罩感知归一化，量化预测头，使用 Pinball（量化）损失进行自监督训练。

**📊 数据集**

预训练数据包含三类：真实数据（GIFT‑Eval‑Pretrain，约 4.5M 级序列，230B 观测值）；合成数据（KernelSynth，1M 生成的周期性序列，用于长上下文）；以及 TSMixup 通过随机混合真实序列产生的增强数据（Clean 400k 与 Leaky 1M）。

**📈 对比分析**

与多种 SOTA TSFM（Chronos‑2、TimesFM‑2.5、Tirex、FlowState、Moirai‑2、Sundial 等）以及统计基线进行比较；零样本模型在 97 个测试案例中获得 MASE、CRPS、Rank 的几何平均值分别为 0.714、0.492、4.591，排名前 5；预训练模型（含 Leaky 数据）进一步提升到 0.696、0.480、3.383，稳居榜首。实验显示，数据规模和 CP 模式是性能提升的主要原因。

**⚠️ 局限性**

局限性包括：1）模型仅在单变量、零样本设定下验证，未涉及多变量或微调；2）对超大规模预训练数据的依赖使得资源消耗高；3）实验仍基于已公开的 GIFT‑Eval 评测，可能无法完全覆盖工业场景；4）虽然展示了数据与训练策略的主导作用，但对架构创新的细粒度评估仍有限。

---

## 462. The First Known Problem That Is FPT with Respect to Node Scanwidth but Not Treewidth

**arXiv ID:** 2602.06903 | [PDF](https://arxiv.org/pdf/2602.06903v1)

**作者:** Jannik Schestag `[一作]` (Delft University of Technology), Norbert Zeh `[通讯]` (Dalhousie University)

**通讯引用:** 1285 | [OpenAlex ID](https://openalex.org/A5019738080)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明了在有向无环图（DAG）上，扫描宽度（scanwidth）与树宽度（treewidth）在参数化可计算性上存在显著差异，并给出了基于扫描宽度的高效算法；

**💡 创新点**

创新点在于首次给出了扫描宽度与树宽度的明确复杂度分离，并证明在多物种进化网络上可利用扫描宽度实现更优的算法复杂度；

**🔧 技术方法**

主要技术包括从树宽度到扫描宽度的结构化转换、利用树扩展（tree extension）构造树分解、以及在树扩展上进行动态规划（DP）求解可行集的最大多样性；

**📊 数据集**

论文中未使用实际数据集，主要以理论构造和多项式时间构造为主；

**📈 对比分析**

由于缺少实验数据，本文未与其他方法做实证比较；理论分析表明算法复杂度为 O(n³·2^{τ})（τ 为节点扫描宽度），相对于树宽度方法在相同参数下实现了更低的指数因子；

**⚠️ 局限性**

限制在于算法需要预先给定一个节点扫描宽度为 τ 的树扩展，且构造此树扩展的高效算法仍未得到；此外，实验验证与实际生物网络的适用性尚未展开。

---

## 463. InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning

**arXiv ID:** 2602.06960 | [PDF](https://arxiv.org/pdf/2602.06960v1)

**作者:** Yuchen Yan `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**通讯引用:** 1532 | [OpenAlex ID](https://openalex.org/A5004615610)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种端到端强化学习框架，用于在迭代推理过程中通过轨迹级优化学习何时压缩、如何压缩以及如何继续推理。

**💡 创新点**

创新点在于将格式学习与策略优化分离，利用轨迹级奖励和共享优势实现对压缩时机、压缩内容与后续推理的全局最优决策，并引入效率奖励实现可控推理深度。

**🔧 技术方法**

采用强化学习（GRPO）与多目标奖励、轨迹级rollout、共享优势、token级梯度裁剪等技术，并在Cold-Start阶段使用监督学习构造InftyThink格式。

**📊 数据集**

使用OpenThoughts-114K进行监督预训练，DeepScaleR-Preview作为RL训练集，并在MATH500、AIME24/25、GPQA_Diamond等基准上评估。

**📈 对比分析**

与传统长上下文RL以及SFT版迭代推理相比，模型在AIME、MATH、GPQA等任务上提升约5-21个百分点准确率，推理时延降低30-60%，并在更大模型上保持相同趋势。

**⚠️ 局限性**

局限性包括对RL训练的计算需求仍高、对特定总结策略的过度依赖、可能在超大规模模型或不同任务域的泛化有限，以及需人工设计奖励函数和超参数。

---

## 464. DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos

**arXiv ID:** 2602.06949 | [PDF](https://arxiv.org/pdf/2602.06949v1)

**作者:** Shenyuan Gao `[一作]`, Linxi "Jim" Fan `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种基于人类视频预训练的大规模机器人世界模型，能够在多种环境下实现连续动作控制与未来状态预测。

**💡 创新点**

创新点包括：①使用连续潜在动作作为统一代理标签提升动作可控性；②构建了44k小时、涵盖数千种技能和场景的最大规模人类视频数据集；③提出自回归蒸馏管道实现实时推理；④实现了在机器人上零样本泛化。

**🔧 技术方法**

技术手段：Cosmos‑Predict2.5潜在扩散模型 + WAN2.2 tokenizer；VAE潜在动作学习；flow matching 与 temporal consistency 损失；Self‑Forcing 蒸馏；自回归因果注意力架构。

**📊 数据集**

使用的数据集：自研的 In‑lab、EgoDex、Human 三大人类视频集（总计 44k 小时），以及小规模机器人数据（GR‑1、AgiBot、G1 等）。

**📈 对比分析**

评估方法：在 6 个 OOD 基准上与无预训练、仅视频预测、真实动作条件三种基线对比，采用 PSNR/SSIM/LPIPS、人工偏好评估以及实时 FPS 速度测评，结果显示潜在动作预训练显著提升物理一致性与动作跟随，蒸馏后模型实时 FPS 提升 4 倍，成功率与真实实验高度相关。

**⚠️ 局限性**

局限性：对罕见或快速动作的模拟仍不理想，生成的失败细节不够真实；未支持多视角仿真；蒸馏后知识保持有限；推理速度虽已加速，但仍有进一步优化空间。

---

## 465. Distributed Knowledge in Simplicial Models

**arXiv ID:** 2602.06945 | [PDF](https://arxiv.org/pdf/2602.06945v1)

**作者:** Éric Goubault `[一作]` (Ecole Polytechnique), Sergio Rajsbaum `[通讯]` (Universidad Nacional Autónoma de México)

**通讯引用:** 3692 | [OpenAlex ID](https://openalex.org/A5005997085)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出并阐释了利用 simplicial complex（三角复杂体）来构造多代理知识逻辑模型，并将该模型应用于分布式通信模型（unreliable broadcast、immediate snapshot、test-and-set）和分布式任务（如 majority consensus）。

**💡 创新点**

创新点在于：① 将分布式知识、共同分布式知识与拓扑几何（k‑connectivity 等）相结合，形成新的逻辑障碍公式；② 用这些公式在三种通信模型上分别证明 majority consensus 的可解性与不可解性；③ 给出基于知识的算法和证明框架，展示逻辑与拓扑相互映射的实用价值。

**🔧 技术方法**

技术手段包括：
- 拓扑学：simplicial complex、k‑connectivity、子复形结构。
- 分布式计算模型：即时快照、广播、测试与设置。
- 知识逻辑：K、D、C、CD_α 等运算符，知识增益定理。
- 逻辑与拓扑的对接：利用协议复杂形的几何性质证明逻辑障碍，从而推导不可解性。

**📊 数据集**

没有使用实际数据集，全部基于理论模型与抽象示例（3 位代理、二值输入）。

**📈 对比分析**

比较方法：在相同任务（majority consensus）下，对三种模型分别给出可解算法或不可解证明；通过逻辑障碍公式验证算法正确性。性能方面，单轮 broadcast 可解，单轮 test‑and‑set 不可解，需两轮才可解；即时快照模型无论轮数均不可解。

**⚠️ 局限性**

局限性：
- 仅以 3 位代理为例，缺乏对规模扩展的讨论。
- 只针对特定通信模型与任务，未覆盖更一般的分布式场景。
- 逻辑障碍方法在更复杂任务中的适用性仍需进一步研究。
- 未进行实验验证，仅以理论与图形示例说明。

---

## 466. Topological Semantics for Common Inductive Knowledge

**arXiv ID:** 2602.06927 | [PDF](https://arxiv.org/pdf/2602.06927v1)

**作者:** Siddharth Namachivayam `[一作]` `[通讯]` (Carnegie Mellon University), Siddharth Namachivayam (Carnegie Mellon University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出一种基于拓扑学的形式逻辑，用来精确表述在共享证据（witness）条件下的共同归纳知识的生成机制，并将其应用于分析一种基于归纳的协调攻击（coordinated attack）问题，给出相应的可实现的证明体系与语义解释。

**💡 创新点**

创新点在于：①构造了兼顾归纳学习者的“切换容忍度”（switching tolerance）与证据集合的拓扑结构；②在传统的共识知识框架中引入了归纳标准与证据可视化的概念，克服了Lewis式定义对归纳标准敏感的缺陷；③通过“可真实理由”（true reason）与“真实理由闭包”（S_i）等新模态，统一刻画了归纳知识与公共归纳知识；④将公共归纳知识的定义表述为一个固定点问题，得到可计算的“内部化”操作 C(P)=int_N(P)。

**🔧 技术方法**

技术方法包括：①拓扑学习理论（topological learning theory）中的信息基（information basis）与嵌套差分（nested difference）来表征有限切换决策方法；②建立一系列新的模态算子（R_i、I_i@、B_i@、S_i、G_W、C）及其语义定义；③使用Tarski-Kleene定理证明固定点存在，并借助 Kuratowski 内部化公理证明 S_i 为内部化算子；④在证明系统中引入非标准模态与一系列原始公理/规则，证明其相对于上述语义的一致性。

**📊 数据集**

本文为理论性工作，没有使用具体数据集；所有结果均基于抽象框架（Ω、E_i、n_i）。

**📈 对比分析**

与现有工作（Lewis、Cubitt & Sugden 2003、Kelly 1996等）比较时，本文在定义共同归纳知识时去除了对“辅助假设”的依赖，保持了对切换容忍度不变性；通过拓扑语义提供了更直观、可计算的解释；此外，针对协调攻击问题给出了可实现的成功集判定（W 必须是每个 𝒯_i 的 (n_i+1)-开放子集），与传统的完全知识/可靠询问模型不同，体现了归纳学习者的有限推理能力。由于没有实验部分，无法给出数值性能指标，但理论上可实现的协作协议在有限状态空间中可有效构造。

**⚠️ 局限性**

主要局限包括：①未给出完整性（completeness）证明，推测需要额外公理或对非标准模态进行约束；②使用的非标准模态导致分布式知识的传统 Kripke 翻译方法不直接适用；③在有限 Ω 时可得到最优协议，但对无限状态空间的可行性和效率尚未讨论；④归纳学习者的“切换容忍度”参数对模型构造的影响在实践中尚需进一步验证。

---

## 467. Strategizing at Speed: A Learned Model Predictive Game for Multi-Agent Drone Racing

**arXiv ID:** 2602.06925 | [PDF](https://arxiv.org/pdf/2602.06925v1)

**作者:** Andrei-Carlo Papuc `[一作]` (Delft University of Technology), Javier Alonso-Mora `[通讯]` (Delft University of Technology)

**通讯引用:** 8363 | [OpenAlex ID](https://openalex.org/A5013297671)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了无人机竞速中的交互规划，提出了基于离线学习的模型预测游戏（LMPG）方法；

**💡 创新点**

创新点在于将高精度的模型预测游戏（MPG）通过可微神经网络摊销，获得近似但实时可用的交互策略；

**🔧 技术方法**

使用了MPG、常规MPC、嵌入可微轨迹优化的神经网络LMPG以及低层线性MPC/INDI控制；

**📊 数据集**

使用自建仿真状态采样数据以及在室内8×5×6 m³实验场的真实飞行轨迹，赛道包括Lemniscate、Lissajous和3D Lemniscate；

**📈 对比分析**

在同步与异步执行模式下进行头对头赛制，比较胜率、加速与碰撞率；LMPG在高速度异步环境中胜率最高，推理时间约为MPG的1/14；

**⚠️ 局限性**

假设完全感知与全局信息，未涉及视觉感知与外部障碍，且训练数据来自仿真，迁移到更复杂环境仍有挑战。

---

## 468. Seeing Beyond Redundancy: Task Complexity's Role in Vision Token Specialization in VLLMs

**arXiv ID:** 2602.06914 | [PDF](https://arxiv.org/pdf/2602.06914v1)

**作者:** Darryl Hannan `[一作]` (Pacific Northwest National Laboratory), Yijing Watkins `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 105 | [OpenAlex ID](https://openalex.org/A5067556276)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过构造合成数据集、引入多种视觉冗余度量、SVD 及线性探测等技术，对 Vision‑LLM 的视觉信息分布与压缩特性进行细粒度分析，并研究了不同任务复杂度与微调对模型内部表示的影响。

**💡 创新点**

创新点在于：①提出针对视觉冗余的系统度量体系；②利用可控合成数据量化任务复杂度与冗余度的相关性；③揭示微调时文本表示主导、视觉表示相对稳定的现象；④为未来动态压缩策略提供实证依据。

**🔧 技术方法**

主要技术包括：Token‑norm 与 Token‑rank 视觉压缩度量（Gini、熵、CV、stable rank、participation ratio、exponential entropy）；SVD 对齐与重构指标；多层 MLP 线性探测；随机 Token 削减实验；以及在 GQA、COCO 等数据集上的微调与零射实验。

**📊 数据集**

使用的数据集有：自定义的 2D 形状合成集（8,220 张图）、MSCOCO 子集（1,000 张图用于压缩/探测、250 张图用于 ablation）、以及 GQA、COCO 微调集（含视觉定位与空间推理任务）。

**📈 对比分析**

通过与 Molmo 与 Llama 3.2 的零射基准、Token 削减曲线和微调后性能对比，展示了不同模型在视觉冗余度量、probe 精度与任务表现上的差异；实验显示高任务复杂度需要更少压缩，且微调更显著提升文本子空间对多模态表示的贡献。

**⚠️ 局限性**

局限性包括：合成数据的简化场景可能无法完全代表真实图像的多样性；实验主要聚焦于 Token 级别的压缩，未探究更深层网络结构或训练动态；压缩策略仍为定量分析而非可直接应用的压缩方法；结果主要基于 Molmo 与 Llama 3.2，未验证在更广泛模型上的通用性。

---

## 469. Improving Credit Card Fraud Detection with an Optimized Explainable Boosting Machine

**arXiv ID:** 2602.06955 | [PDF](https://arxiv.org/pdf/2602.06955v1)

**作者:** Reza E. Fazel `[一作]` (EN Bank), Siavash A. Bigdeli `[通讯]` (DTU)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过优化Explainable Boosting Machine的预处理、超参数调优和特征选择，提升了信用卡欺诈检测的准确性。

**💡 创新点**

创新点在于将Taguchi方法用于预处理顺序与超参数调优，并在不采样的前提下通过EBM实现高解释性和性能提升。

**🔧 技术方法**

主要技术包括Explainable Boosting Machine、Taguchi实验设计、特征缩放、交叉验证及与传统模型的对比评估。

**📊 数据集**

使用了Kaggle公开的欧洲信用卡欺诈数据集（284,807笔交易，30个特征）。

**📈 对比分析**

通过与Logistic Regression、Random Forest、XGBoost等模型在相同特征子集上对比，EBM在18个最重要特征下实现ROC‑AUC 0.983，优于其他模型。

**⚠️ 局限性**

局限在于未实现在线学习和对动态欺诈策略的即时适应，且对不同数据集的泛化仍需验证。

---

