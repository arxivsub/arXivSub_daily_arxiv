# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-11-05 | 今日论文总数: 711

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Learning vs. Optimizing Players in Budgeted Games

**arXiv ID:** 2611.05580 | [PDF](https://arxiv.org/pdf/2611.05580v1)

**作者:** Giannis Fikioris `[一作]` (Google Research), Éva Tardos `[通讯]` (Cornell University)

**通讯引用:** 31153 | [OpenAlex ID](https://openalex.org/A5025175846)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在预算约束下的重复博弈中，优化器和学习者之间的互动，提出了一种高效的学习算法，能够在每个预算博弈中实现非操控性和低随机悔恨。

**💡 创新点**

创新点在于提出了一种新的学习算法，该算法在预算约束下同时保证了学习者的非操控性和低随机悔恨，并且能够有效应对优化器的自适应策略。

**🔧 技术方法**

使用了响应校准和拉格朗日校准两种新技术来确保学习者的非操控性和低随机悔恨，同时通过模拟在线梯度上升来构建预测。

**📊 数据集**

使用了多种预算博弈的模拟数据集，具体数据集未详细说明，但假设学习者能够观察到其自身的效用和成本。

**📈 对比分析**

与其他方法相比，该算法在每个预算博弈中都能保证学习者的非操控性，优化器的期望收益被限制在预算斯塔克尔伯格均衡值加上O(√(T))的范围内，且在固定混合策略的情况下，学习者在每个区间的悔恨为O(√(T))。

**⚠️ 局限性**

限制在于低校准误差需要随机预测，而这些预测可能没有简洁的表示或直接计算的方式，尽管算法在多项式时间内构建了这些预测。

---

## 2. Shallow Encoding Emits Deep Memories

**arXiv ID:** 2611.06084 | [PDF](https://arxiv.org/pdf/2611.06084v1)

**作者:** Mingyuan Ma `[一作]`, Ashwin Nanjappa `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Shallow Encoding Emits Deep Memories 的方法，利用训练后的混合模型的写-读不对称和状态 sink 结构，先在预填阶段只执行浅层并使用 emitters 写出深层记忆，再在解码阶段将循环状态压缩为 sink + 低秩表示，并在每一步保持更新的精确性；整个过程不改变原始模型权重。

**💡 创新点**

创新点在于：
1) 发现并利用“写比读更简单”的属性，允许预填阶段仅跑浅层并通过已训练的 emitters 复制深层记忆；
2) 发现循环状态几乎是一个固定方向的 rank‑1 sink 加上低秩工作记忆，并证明该形式在 Gated DeltaNet、Kimi Delta Attention、Mamba‑2 的更新下保持不变；
3) 设计可训练的 prompt‑code 与 emitters，使得压缩后状态与原始模型的分布误差可控；
4) 在保持模型性能的同时，通过两种浅层化（预填深度、状态表示）显著降低 HBM 带宽与 FLOPs。

**🔧 技术方法**

技术包括：
- 预填层剪枝与 emitters 复制写路径；
- 状态压缩为 sink + UV^T 低秩表示；
- 精确更新公式和开环误差界定；
- 训练阶段分两步：无监督的窗口蒸馏 + 线上自我对话蒸馏；
- 使用 SGLang 作为推理引擎，结合 prefix‑cache 与分布式解码；
- 采用 fp16/ NVFP4 等量化格式进行存储和推理。

**📊 数据集**

数据集：
- 公共文本（4K/16K 窗口）；
- 合成检索文档、公开对话、聊天语料；
- 训练阶段使用模型自身对公开提示的回复；
- 评测使用官方考试集合（GPQA‑Diamond、MMLU、AIME、MATH‑500、IFEval、LiveCodeBench、HumanEval+、LongBench、RULER）以及 agentic‑coding trace 进行吞吐量基准。

**📈 对比分析**

比较方法与性能：
- 对比原始冻结 checkpoint 及 NVIDIA 官方 NVFP4 版本，测量官方考试分数、下游 token 分布漂移、前缀缓存命中率与每 GPU 吞吐量；
- 结果显示：在 6 个 checkpoint 上，考试分数在 2 个标准误以内，下一步分布漂移 < 0.002 nats/token；
- 吞吐量提升 1.2–1.4×（在 GB300 GPU 上的 agentic‑coding trace 及多用户并发场景），尤其在 disaggregated 与 aggregated 部署中都实现显著加速。

**⚠️ 局限性**

局限性：
- 未压缩软最大化注意力缓存，适用于注意力层占主导的模型时收益有限；
- Anchor 深度通过无监督探测确定，缺乏自适应学习机制；
- 开环误差界限为 worst‑case，实际闭环漂移远低于此；
- 主要在混合模型（Mixture‑of‑Experts）上验证，纯稠密模型的收益尚未彻底评估；
- 当前实现仅支持 fp32 状态，fp16 或更低精度的实际漂移与吞吐还需进一步实验。

---

## 3. Readable Is Not Steerable: Pre-Registered Tests of Physical Structure in Learned Fluid Representations

**arXiv ID:** 2611.05577 | [PDF](https://arxiv.org/pdf/2611.05577v1)

**作者:** Manvendra Modgil `[一作]` `[通讯]` (Modint Intelligence), Manvendra Modgil (Modint Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在二维周期性Navier–Stokes流场上，进行三项预注册研究，探究自监督世界模型的潜在空间是否围绕物理量组织，训练目标如何影响潜在空间中物理量的显著性，以及如何设计可控的“拨盘”实现无标签的可调节控制。

**💡 创新点**

创新点包括：① 将潜在空间的可解码性、显著性与可调性三属性分离并系统评估；② 通过预注册与控制实验揭示训练目标和正则化对物理量表达的不同影响；③ 设计并验证一种通过路由和FiLM实现的可调拨盘架构，使潜在空间中无标签即可找到并操纵音量和谱形。

**🔧 技术方法**

技术手段包括：自监督联合嵌入预测（JEPA）和自动编码器；VICReg 变差协方差正则化；FiLM 逐层特征调制；子像素/重采样上采样；自监督测量（线性探针、慢特征分析、干预评估）。

**📊 数据集**

数据集为64×64分辨率的二维不可压Navier–Stokes模拟，包含约2000条轨迹，初始条件由高斯涡子混合和随机能谱生成，分为训练、验证、测试（含离群Reynolds数区间）以及无粘性轨迹；所有数据可复现。

**📈 对比分析**

比较方法：对未训练网络、非物理动态JEPA、无时间的静态VICReg等控制模型进行线性R²、慢特征相关性E1和拨盘可调性指标评估；结果表明JEPA未能满足预注册的物理组织标准（负面）；自动编码器在无正则化时使耗能突出，但VICReg虽降低显著性但提升可解码性；拨盘设计在大多数（5/8）训练跑中成功，但成功率不稳定。

**⚠️ 局限性**

局限性：仅在二维流动、单一小型卷积架构、有限随机种子范围内验证；VICReg正则化同时降低渲染质量，导致影响难以分离；拨盘实现依赖特定路由与上采样，难以直接推广到三维或更复杂数据；实验设计依赖多次探索与后验分析，可能受偏差影响。

---

## 4. DSRS: Teaching MLLMs to Perceive, Reason, and Segment in Complex Driving Scenes

**arXiv ID:** 2611.06160 | [PDF](https://arxiv.org/pdf/2611.06160v1)

**作者:** Sicen Guo `[一作]` (Tongji University), Rui Fan `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的多模态感知任务——驾驶场景推理分割（Driving Scene Reasoning Segmentation），并基于此构建了涵盖 RGB、相对深度图、隐式指令与像素级分割掩码的 1,561 条样本基准数据集。作者设计了一种名为 DSRSNet 的网络框架，核心在于深度引导的层次特征交互（DHFI）策略和几何先验提取（GPE）模块，以实现几何信息与视觉语义的对齐与融合。

**💡 创新点**

创新点包括：①首次将几何先验（相对深度）与多模态大型语言模型（MLLM）进行对齐，解决 RGB‑预训练的 MLLM 在处理非 RGB 模态时的表征不匹配；②提出深度引导的层次特征交互（DHFI）及其内部的深度感知可变形注意力（DA^2）模块，实现几何信息在多层视觉特征中的自适应采样与交互；③通过引入专用分割标记 token 与自回归分割头，使得模型能够在文本生成的同时输出像素级分割结果。

**🔧 技术方法**

技术手段包括：CLIP‑基视觉编码器、几何先验提取器（ResNet‑基 GPE）、深度引导的层次特征交互（DHFI）与深度感知可变形注意力（DA^2）、多模态 LLM（如 LLaVA 或 Qwen2.5‑VL）与 LoRA 微调、专用分割标记 token 与自回归分割头。训练采用 AdamW、DeepSpeed、2 张 A800 GPU，迭代 120k 次。

**📊 数据集**

使用的数据集：①自研驾驶场景推理分割基准（1,561 条 RGB+深度+指令+掩码）；②公开的 RefCOCO、RefCOCO+、RefCOCOg 进行跨任务评测；③ReasonSeg 用于验证模型在更传统推理分割任务中的表现。

**📈 对比分析**

与现有方法（LISA、GSVA、PixelLM、M^2SA、SegAgent 等）进行零样本与微调对比。DSRSNet 在驾驶场景推理分割任务上，零样本 gIoU 与 cIoU 均超过 40% 的提升，微调后 gIoU 达到 53.3% 以上，显著优于所有对比方法。该模型在 RefCOCO 等通用指令分割任务中也表现出更高的 gIoU / cIoU，验证了其通用性。

**⚠️ 局限性**

局限性：①模型仍依赖高质量的相对深度图，深度噪声会影响推理；②在极端光照或纹理缺失场景下，几何先验的提取与交互效果有限；③数据集规模相对较小，可能限制模型对更大范围场景的泛化；④引入多模态特征与 DHFI 需要额外计算资源，推理时的时延较传统单模态模型更高。

---

## 5. GeoBiaset: A Counterfactual Benchmark for Demographic Bias in World-Level Geolocalization

**arXiv ID:** 2611.06085 | [PDF](https://arxiv.org/pdf/2611.06085v1)

**作者:** Niccolò Niccoli `[一作]` (University of Florence), Lorenzo Seidenari `[通讯]` (University of Florence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

引入GeoBiaset基准，利用对照性编辑图像评估图像地理定位模型在插入不同种族、性别人脸时对预测的影响，提出POPI指标衡量模型对前景人类属性的敏感性。

**💡 创新点**

首创面向全球图像地理定位的对照性基准和方向性偏差指标，利用人脸插入保持GPS不变的反事实图像，揭示模型对种族/性别特征的隐性依赖。

**🔧 技术方法**

使用Diffusion模型FLUX.2-klein-9b-kv进行人脸插入，Qwen3‑VL‑8B‑Instruct等LLM进行质量控制；通过CLIP‑based IQA、全参考PSNR/SSIM评估编辑质量，利用SAM分割与梯度权重注意力可视化分析模型关注点。

**📊 数据集**

基于IM2GPS3k与YFCC4k合并的329张背景图，结合FairFace平衡面部库生成28种编辑组合，形成9541张图像的GeoBiaset公开数据集。

**📈 对比分析**

在标准1km、25km、200km、750km、2500km精度下，将原始图像与编辑图像进行对比；所有模型（专用地理定位模型和零射线多模LLM）在大陆级别准确率下降16%–47%，POPI-A超过28%等，表明显著的种族引导偏差。

**⚠️ 局限性**

仅覆盖329背景、七种种族标签；生成过程未保证像素级背景一致，缺乏人工评估；仅考察前景插入，未涵盖迁徙、多模特异等因素，标签仅为操作性分类，可能导致评估结果受限。

---

## 6. Contact-Implicit Whole-Body Trajectory Optimization for Dynamic Humanoid Locomotion on Parameterizable Uneven Terrain

**arXiv ID:** 2611.05585 | [PDF](https://arxiv.org/pdf/2611.05585v1)

**作者:** Roberto Mauceri `[一作]` (Istituto Italiano di Tecnologia), Daniele Pucci `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种离线全身轨迹规划器，利用接触隐式优化框架为类人机器人在不规则、可参数化地形上生成动态步态。

**💡 创新点**

创新点包括：①用可微分平滑函数逼近阶梯、斜坡等离散地形；②对质心动力学进行质量归一化，提升收敛速度；③在接触约束中采用松弛的补偿条件，避免数值不稳定；④不预设接触序列，让优化自行决定接触时机与位置。

**🔧 技术方法**

主要技术手段为：接触隐式全身轨迹优化（NLP），直接转录（多步射击+显式欧拉积分），使用 CasADi + IPOPT/SQP 求解器，单位四元数积分，基于对称张量的梯度和海森矩阵计算，松弛补偿条件与摩擦锥约束。

**📊 数据集**

实验使用合成地形模型（阶梯、斜坡、凹坑、起伏地形）以及真实机器人 ErgoCub 的物理仿真与实地实验，未使用公开数据集。

**📈 对比分析**

与 Winkler 等人基于 MICP 的方法及传统 MICP/NLP/SDP 的对比，展示了在同一硬件（i7‑11th, 32 GB）上解算时间从数十秒到几分钟，约束违例率 <0.001，轨迹可在仿真与实机上成功执行，尤其在动态跳跃与坡道步行上表现优异。

**⚠️ 局限性**

局限性：只能处理可用平滑函数近似的结构化地形；对非结构化点云地形缺乏直接支持；足部仅使用四点接触，未考虑足内部点和自碰撞检查；计算量大，尚未实现实时在线规划。

---

## 7. FWIK: Multi-Waypoint IK Selection via Lookahead Graph Search

**arXiv ID:** 2611.05897 | [PDF](https://arxiv.org/pdf/2611.05897v1)

**作者:** Jinglan Cao `[一作]`, Chengwei Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在线逆运动学选择方法，利用有限前瞻窗口与边级自碰撞筛选来提升冗余机械臂按顺序执行多目标点的完整性和运动时间。

**💡 创新点**

创新点在于：①基于有限前瞻窗口的递归式IK选择，将多目标点的后续连通性纳入决策；②采用学习的边级自碰撞分类器提前剔除可能导致自碰撞的转移，从而在不依赖完整规划的情况下提升可靠性。

**🔧 技术方法**

使用技术包括：TRAC-IK 逆运动学求解器、层级图与同步三角/梯形运动时间估计、Transformer 结构的自碰撞预测模型、MoveIt 2 运动规划器（Pilz PTP、RRT‑Connect、CHOMP、STOMP）以及同步姿态/关节限制约束。

**📊 数据集**

数据集为 100 条随机生成、无障碍环境下的 50 步目标序列（对 Franka Panda 与 KUKA LBR iiwa 两个 7‑DoF 机器人），每条序列独立采样以产生多样化且非平滑的姿态。

**📈 对比分析**

对比方法有 IKLink、STAMPEDE、TRAC‑IK（重试模式）并在同一下游规划器下进行实验；结果显示该方法在所有 8 个机器人‑规划器组合中实现 96.8–100% 的序列成功率，并使规划路径时长比 STAMPEDE、IKLink、TRAC‑IK 分别减少 12.8–37.2%；但决策时间略高，尤其在 CHOMP 等迭代规划器中更为明显。

**⚠️ 局限性**

局限性包括：仅在软件仿真、无障碍场景下评估；仅适用于 7‑DoF 机器人，需为不同机械臂训练专属自碰撞模型；假设未来目标已知且无感知不确定性；未验证真实机器人执行、控制误差或动态环境下的表现。

---

## 8. On Convergence of Gradient Temporal-Difference Learning with Regularized Corrections and Eligibility Traces

**arXiv ID:** 2611.05869 | [PDF](https://arxiv.org/pdf/2611.05869v1)

**作者:** Jingyi Jin `[一作]` (University of Alberta), Hao Wang `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对受限单时间尺度的TDRC(λ)及其Huber扩展TDRC(λ)-Huber在线性函数逼近、非i.i.d.离线数据下的收敛性进行了严格分析。

**💡 创新点**

创新点包括：①首次将ODE弱收敛方法应用于TDRC(λ)的耦合线性动力学；②提出TDRC(λ)-Huber并证明其保持原固定点；③通过积分小增益条件对非线性均值ODE实现全局渐进稳定性判定；④揭示正稳定A_λ时正则化参数β可实现大β收敛机制。

**🔧 技术方法**

主要技术手段为：弱收敛ODE分析、Lyapunov方程与小增益理论、状态–追踪过程的平稳性证明以及非线性反馈分解。

**📊 数据集**

使用了一个四状态离线预测示例（四个状态、环形转移）来数值验证β与小增益条件的关系；除此之外本文不涉及其他公开数据集。

**📈 对比分析**

方法评估仅通过理论证明和示例数值验证实现；实验中未与其他算法直接比较，但通过积分小增益阈值与已知的TDC-Huber、TDC等方法对比表明TDRC(λ)-Huber在正则化增强时能保证收敛。

**⚠️ 局限性**

局限性在于：①仅证明了受限算法且收敛步长趋于0；②适用于线性特征逼近，无法直接推广到非线性网络；③小增益与Lyapunov条件可能过于保守，导致实际可接受参数范围被低估。

---

## 9. Robots in the City: Persona-Based Perceptions of Urban Robots

**arXiv ID:** 2611.05954 | [PDF](https://arxiv.org/pdf/2611.05954v1)

**作者:** Keonhee Jang `[一作]` (Korea Advanced Institute Of Science And Technology), Yoonjin Yoon `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究通过合成城市街景图片，评估机器人出现与类型对行人安全感的影响。

**💡 创新点**

创新点在于将机器人条件与时间、道路类型结合，使用 persona‑conditioned MLLMs 与人类受试者对比，探究模型对安全感的解释一致性。

**🔧 技术方法**

使用 multimodal large language models（Qwen3.8‑27B、Gemma4‑26B）和图像生成/编码技术，结合 persona 生成。

**📊 数据集**

使用 Nemotron‑Personas‑Korea 合成人物数据集、Mapillary 街景图片、生成的机器人场景。

**📈 对比分析**

通过对比模型与人类在 18 条街景对比试验中的胜率和一致性，发现模型倾向无机器人最安全，且与人类一致性中等，尤其在人机对比时差异显著。

**⚠️ 局限性**

局限包括样本量小、受试者年龄范围窄、图像静态缺乏动作、模型受图像位置影响、仅评估两款开源模型。

---

## 10. TRUST-Stereo: Teacher-Assisted Robust Uncertainty-Guided Search and Transfer for Fast Stereo Matching

**arXiv ID:** 2611.05689 | [PDF](https://arxiv.org/pdf/2611.05689v1)

**作者:** Pouya Sohrabipour `[一作]` (University of Arkansas), Dongyi Wang `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于粗到细的实时立体匹配框架TRUST‑Stereo，并通过两阶段教师助推知识蒸馏和不确定性引导残差搜索模块实现更鲁棒的视差估计。

**💡 创新点**

创新点在于引入UGRSM通过估计传播视差的可靠性来动态调整局部搜索中心，并设计两阶段Senior–Junior Teacher Assistant蒸馏，既获得大模型表征又保持轻量化。

**🔧 技术方法**

技术上使用ViT‑L/ViT‑S视觉 Transformer、跨层交叉注意力、ConvGRU迭代细化、UGRSM残差搜索以及两阶段蒸馏。

**📊 数据集**

数据集包括SceneFlow合成对、TartanAir、FallingThings、Virtual KITTI 2、FoundationStereo、Dynamic Replica，评测于KITTI 2012/2015、ETH3D、Middlebury。

**📈 对比分析**

与多种高效立体匹配方法（FastACV+、LightStereo、Fast Foundation Stereo、BANet、Lite Any Stereo等）比较，TRUST‑Stereo在零样本和百万级训练下均实现多项指标SOTA，尤其在KITTI 2015 D1 3.49%和ETH3D Bad1.0 2.13%，推理≈50ms。

**⚠️ 局限性**

局限性包括对大尺度场景的泛化仍受限，且不确定性预测仍依赖伪标签，可能在极端遮挡或纹理稀疏区域表现不足。

---

## 11. Homology Features for X-Ray Hyperspectral Images Segmentation

**arXiv ID:** 2611.06230 | [PDF](https://arxiv.org/pdf/2611.06230v1)

**作者:** Fethi Harkat `[一作]` (Detection Technology Plc), Kévin Polisano `[通讯]` (University Grenoble Alpes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种结合持久同调与CNN‑GNN的混合架构，对多能量XRT高光谱图像进行分割。

**💡 创新点**

创新点在于在UMAP投影的超像素点云上计算Vietoris–Rips持久同调，提取H1循环结构作为特征，并通过广播融合方式将超像素级全局特征传递到像素级，显著提升分割性能。

**🔧 技术方法**

使用的技术包括UMAP降维、SLIC超像素、Ripser计算持久同调、CNN全卷积网络、GCN图网络以及广播融合策略。

**📊 数据集**

实验数据集为先前研究中的烟盒与行李背景的ME‑XRT图像补丁，标注为二分类目标与背景。

**📈 对比分析**

与单独CNN（像素级与超像素级）以及Pooling融合的CNN+GNN进行对比，广播融合的混合模型在像素F1、超像素F1均分别提升约0.07和0.086，整体准确率达到0.964。

**⚠️ 局限性**

局限性包括仅验证在单一二分类数据集上，超像素分割参数固定，Pooling融合对错误超像素敏感，且未深入探讨多类别扩展和理论分析。

---

## 12. Optimizing Geo-replicated Deterministic Databases

**arXiv ID:** 2611.05882 | [PDF](https://arxiv.org/pdf/2611.05882v1)

**作者:** Yunhao Mao `[一作]` (University of Toronto), Hans-Arno Jacobsen `[通讯]` (University of Toronto)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在多地活跃复制环境下的事务并发控制，提出了2PL、OCC与Deterministic Concurrency Control（DCC）的理论模型，并基于此设计并实现了名为Quirinus的DCC数据库，包含批量预排序、投机执行、事务重排、CRDT集成和快速提交等优化；

**💡 创新点**

首次提供针对geo复制的DCC性能模型并识别其优势情境；通过批量预排序和投机执行等多项技术，显著提升DCC的吞吐与延迟；

**🔧 技术方法**

使用Calvin式DCC、批量预排序与一致性协议、Silo风格投机执行、最大权重独立集(MWIS)事务重排、CRDT（如PN-Counter）以及快路提交；

**📊 数据集**

采用TPC‑C、YCSB‑A（Zipf分布）基准，并通过模拟跨数据中心RTT进行延迟实验；

**📈 对比分析**

与传统2PL、OCC实现以及Tiga、Detock等现有geo数据库进行对比。Quirinus在TPC‑C上可达Tiga吞吐的6.2倍，在YCSB低延迟场景优于Tiga，但在高延迟时Tiga表现更好；整体吞吐高但单事务延迟随批量和网络延迟上升；

**⚠️ 局限性**

需要批量形成和一致性复制导致每事务额外延迟；在低争议或高延迟环境下收益有限；CRDT集成增加复制开销；模型假设全复制、固定事务长度，未考虑分片或多版本场景；极高跨DC RTT下仍受网络制约；

---

## 13. Pathwise-Budgeted LLM Routing: Online Allocation with K-Bit Tail-Cost Learning

**arXiv ID:** 2611.05835 | [PDF](https://arxiv.org/pdf/2611.05835v1)

**作者:** Hanzhang Qin `[一作]` (National University of Singapore), Zhi-Hai Zhang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种针对有限预算下LLM路由的上下文带状多臂赌博机（K-bit Pathwise Contextual Bandits with Knapsacks）算法ADP-HT。

**💡 创新点**

创新点在于将输出长度量化为K位阈值网格，构造风险敏感的预成本上界并结合双速控制的影子价格，实现了对路径预算的自适应惩罚与重试决策。

**🔧 技术方法**

使用混合线性UCB奖励学习、基于阈值的超量化成本预测、双速（fast/slow）影子价格与节奏控制、以及递归重试门控的组合。

**📊 数据集**

在MATH500、GPQA、RouterBench以及LiveCodeBench等公开与内部基准任务流上进行实验。

**📈 对比分析**

与四个现有路由基线（BA、PA、ParetoBandit、Greedy LinUCB）以及OpenRouter的AutoRouter进行对比，ADP-HT在紧预算下平均完成任务数提升超过两倍，在RouterBench多达十四个预算点中占优10个，且在LiveCodeBench低预算场景下解决问题数远超商用路由器。

**⚠️ 局限性**

局限在于K位网格需预先设定且对样本量敏感，缺乏严格的路径预算下整体性能下界，且在开放式长任务或多轮评估环境中未验证。

---

## 14. RACE: Calibrated Cumulative Scoring for Rollout-Free Robot Policy Checkpoint Selection

**arXiv ID:** 2611.05838 | [PDF](https://arxiv.org/pdf/2611.05838v1)

**作者:** Changyeon Kim `[一作]` (KAIST), Kimin Lee `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种无回放（rollout‑free）的检查点（checkpoint）评估指标，用于在机器人学习过程中快速筛选最有可能在真实环境中表现优异的模型版本。

**💡 创新点**

该指标通过：① 将动作按物理意义分组并分别测量误差；② 用成功参考轨迹中相似状态的动作差异计算状态条件阈值；③ 将超过阈值的误差映射为指数衰减分数；④ 采用软最小聚合得到每时刻分数，并用累计乘积捕捉早期偏差对后续行为的影响；⑤ 最后对整个轨迹和参考集求平均得到最终评分。

**🔧 技术方法**

核心技术包括动作分组与欧氏/SO(3)距离计算、基于最近邻的状态条件阈值估计、指数衰减与软最小聚合、累计乘积的时间聚合、Bootstrap 置信区间评估以及对比实验。

**📊 数据集**

在仿真方面使用 DexMimicGen 八项抓取/装配任务（共 500 条来自 RoboMimic 的演示），在真实机器人上使用 Franka Research 3 进行 PnP Apple、Put Pan on Stove、Put Pet on Shelf、Nest Cups、Put Two Cubes、PnP 3 Fruit 等任务，并采集 held‑out 演示和成功 roll‑outs 作为验证集。

**📈 对比分析**

与验证损失、验证 MSE 以及离群点（off‑manifold）范数等基线进行对比，使用 Spearman 相关、Hit@K、nRegret、MMRV 等指标评估排名质量。实验表明，该指标在仿真和真实机器人实验中均实现了最高的相关性、最低的选择惩罚，并在分布偏移时保持稳健。

**⚠️ 局限性**

局限性在于：① 仅给出相对排名，无法直接估计真实成功率；② 仍需在硬件上验证最终选定的检查点；③ 当成功参考轨迹不足或分布与目标差异过大时，阈值估计可能失效，导致误判。

---

## 15. Adapting Qwen3 to Tajik from 0.6B to 8B: Lower Text Loss Did Not Mean Better Understanding

**arXiv ID:** 2611.06062 | [PDF](https://arxiv.org/pdf/2611.06062v1)

**作者:** Tohir Saidzoda `[一作]` (Saidzoda AI Lab), Mariyamkhon Shodyeva `[通讯]` (Saidzoda AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 2025 年使用轻量级 LoRA 适配和重度预训练，对 Qwen3 模型进行塔吉克语适配，以提升其文本生成与阅读理解能力。

**💡 创新点**

系统性比较轻量与重度两种适配策略，发现仅降低文本损失并不能提升阅读理解，并提出低资源语言适配的评估准则。

**🔧 技术方法**

采用 LoRA、4-bit/16-bit 量化（QLoRA、SINQ）、持续预训练（CPT）、监督微调（SFT）以及 Unsloth 训练框架。

**📊 数据集**

使用塔吉克新闻、书籍片段、维基百科、Belebele 阅读理解测试集以及英文 MMLU 子集。

**📈 对比分析**

对照起始模型，在 Belebele、英文 MMLU 与塔吉克文本损失三项指标上进行配对 95% 置信区间评估，结果显示轻量适配仅降低损失，阅读理解仅在 8B 级别的重度预训练后显著提升。

**⚠️ 局限性**

实验缺乏控制变量、单次训练、未留出持久测试集、部分训练记录丢失、量化模型缺失层、塔吉克语阅读理解仅基于单一测试且字母偏好未纠正。

---

## 16. How Far Can You Steer? Behavior-Capability Frontiers in Contrastive Activation Addition

**arXiv ID:** 2611.06001 | [PDF](https://arxiv.org/pdf/2611.06001v1)

**作者:** Deepanshu Raj `[一作]` `[通讯]` (University of Chicago), Deepanshu Raj (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了对语言模型进行激活层面的对比激活添加（CAA）干预，以在不更新参数的前提下实现行为调节，并通过行为-能力前沿（behavior‑capability frontier）框架，在限定的能力损失预算下寻找可用的干预点。

**💡 创新点**

创新点在于提出了将行为效应与能力约束结合的前沿搜索方法，区分层选择与系数选择，并引入了显式的能力低置信区间约束；同时通过对比控制实验、几何与因果转移分析以及开放式生成评估，系统地阐明了激活干预的可用范围与局限。

**🔧 技术方法**

技术主要包括对比激活方向构建（CAA）、残差流的向量加法干预、相对扰动强度 η 归一化、层选择与系数搜索、置信区间估计、方向特异性控制（norm‑matched、label‑shuffled、off‑target‑KL）以及后置审计（几何转移、二次能力、OOD、开放式生成）。

**📊 数据集**

使用的数据集包括：行为对比数据（sycophancy、refusal、hallucination、corrigibility、survival instinct、myopic reward、AI coordination）、能力基准（MMLU、ARC‑Challenge、HellaSwag）、二次评估集（GSM8K、WikiText‑2）以及TruthfulQA和开放式生成的手工或 LLM 评测集。

**📈 对比分析**

与三组对照（norm‑matched、label‑shuffled、off‑target‑KL）相比，CAA在所有非零干预点上实现了平均 0.124 的行为增益，且控制组接近零；在 3% 能力预算下，约 72.7% 的干预点在行为与能力上均通过验证；相比单一方向或无约束干预，前沿方法显著提高了可用性，然而在开放式生成与二次能力上仍表现出明显退化。

**⚠️ 局限性**

限制主要包括：只在 7–8B 级单一任务调优模型上验证；仅约束了 MMLU、ARC‑Challenge、HellaSwag 三个基准，无法保证对所有能力的完整保持；开放式生成评估依赖 LLM 判定且缺乏人类验证；几何与因果转移关联较弱，难以通过向量相似度单独预测效果；1% 能力预算下的低通过率可能受样本不确定性影响。

---

## 17. Modality-Aware Expert Pruning for MoE-Based Multimodal Large Language Models

**arXiv ID:** 2611.05927 | [PDF](https://arxiv.org/pdf/2611.05927v1)

**作者:** Donggeun Kim `[一作]` `[通讯]` (Nota Inc), Donggeun Kim (Nota Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无训练的模态感知专家剪枝框架MAEP，用于压缩MoE多模态大语言模型的专家参数。

**💡 创新点**

创新点包括：① 将专家重要性按文本和视觉两种模态拆分，避免模态不一致导致的误剪；② 设计Delta‑H输出指标（在去除专家后对MoE输出的L2变化）并配合重分配反事实，提供高效的权重重要性评估；③ 采用低注意力视觉令牌作为稳定的视觉剪枝信号；④ 通过对文本侧保持信号与视觉侧剪枝信号的加权组合，并自动推算α，得到全局最优剪枝顺序。

**🔧 技术方法**

技术手段包括：重分配感知的专家重要性计算、log+z-score标准化、异向量加权组合、全局排名剪枝、以及单次前向传递的无梯度计算。

**📊 数据集**

使用的校准数据集：ShareGPT4V（约1000条样本），可选混合ChartNet；评估数据集：多模态基准（MME、MMBench、MMStar、ChartQA、TextVQA、RealWorldQA、CV-Bench、POPE）和文本基准（MMLU-Redux、IFEval、ARC-Challenge）。

**📈 对比分析**

与两种基线（HC‑SMoE聚类合并、REAP路由权重剪枝）对比，在25%剪枝时保持几乎原始性能；在50%剪枝时平均保留率高于基线，且内存占用从≈58GB压缩至≈31GB（≈27GB节省，≈47%内存），可在单个40GB GPU上完整加载。

**⚠️ 局限性**

局限性：① 剪枝结果对校准集的模态分布敏感；② 在文本单一任务上高比例剪枝仍会显著下降；③ α超参数对不同模型敏感，需要预先估算；④ 仅针对路由专家，无法处理共享或始终激活的专家；⑤ 目前未对推理速度或量化等进一步压缩手段做实验。

---

## 18. Inferring Factor Importance with Explainable AI Model for Macrophage Immune Cell Polarization

**arXiv ID:** 2611.05670 | [PDF](https://arxiv.org/pdf/2611.05670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 19. Easy Multi-Precision Acceleration via Vectorization

**arXiv ID:** 2611.05558 | [PDF](https://arxiv.org/pdf/2611.05558v1)

**作者:** Chansup Byun `[一作]` (MIT), Peter Michaleas `[通讯]` (MIT)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估在高层语言（MATLAB、Python）下使用向量化和分布式数组实现的基准程序（Stream、矩阵乘法、Beamformer、Blurimage）在不同数值精度和硬件平台上的性能。

**💡 创新点**

提出通过简易向量化代码即可在多代 GPU/CPU 上接近硬件极限性能；展示了多精度（FP64、FP32、FP16、TF32、FP8 等）与多算法（直接 vs FFT）在同一高层语言中的可比性，并量化了近十年内硬件演进带来的性能提升。

**🔧 技术方法**

向量化、分布式数组模型、CUDA/OpenCL 后端、CuPy/Python/Matlab 的 GPU 接口、FFT 语义、自动内存分配与同步。

**📊 数据集**

基准使用合成矩阵和随机数据（例如随机浮点向量、随机图像），不涉及真实业务数据集。

**📈 对比分析**

在 MIT SuperCloud 上通过统一的软件栈对不同节点（CPU、NVIDIA B200、RTX PRO 6000、H100 等）进行内节点垂直扩展与跨代时间扩展实验；记录 GByte/s（Stream）和 GFlop/s（其余）性能；结果显示 FP32 与 FP16/TF32 能显著提升至硬件峰值，FP8 通过 H100 可突破 1 PFlop；相较 10 年前，内存带宽提升 70×、矩阵乘法提升 35×、Beamformer 32×、Blurimage 8×。

**⚠️ 局限性**

1）实验仅限于合成基准，缺乏真实工作负载验证；2）FP8/INT8 等低精度支持在 Python 中仍需复杂手工操作，超出普通用户能力；3）仅测试单节点垂直扩展，未覆盖跨节点水平扩展；4）对不同语言（Julia 等）的兼容性未深入探讨。

---

## 20. Understanding the Perception of Transliterated Offensive Social Media Posts Among Banglish and Hinglish Users

**arXiv ID:** 2611.06226 | [PDF](https://arxiv.org/pdf/2611.06226v1)

**作者:** Dhiman Goswami `[一作]` (George Mason University), Sanchari Das `[通讯]` (George Mason University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 Banglish 和 Hinglish 等转写形式的攻击性社交媒体帖子进行用户感知与行为反应的系统调查。

**💡 创新点**

首次从用户角度量化转写内容的攻击性感知差异、性别与年龄对举报与忽略行为的影响，并指出转写是内容审核的重要盲区。

**🔧 技术方法**

采用在线问卷（Likert 评分、举报/忽略行为选择）、统计分析（比例、置信区间、Bonferroni 校正）进行数据处理。

**📊 数据集**

使用五个公开数据集：English OLID、Bangla Violence Detection、Hindi Hostility Detection、Banglish TB-OLID、Hinglish Hate Detection。

**📈 对比分析**

通过描述性统计与置信区间比较不同语言与转写形式的攻击性评分、举报率与忽略率，发现转写版攻击性评分高但举报率略低；并未给出传统算法性能指标。

**⚠️ 局限性**

样本来源单一平台（Prolific）导致性别/年龄多样性不足；未采用混合效应模型处理个体与帖文随机效应；不同数据集间严重性可能不均衡；实验仅在实验室问卷环境中进行，缺乏真实社交媒体情境验证。

---

## 21. TRACK: Multiple Instance Learning for Anti-Money Laundering in Transaction Networks

**arXiv ID:** 2611.06035 | [PDF](https://arxiv.org/pdf/2611.06035v1)

**作者:** Seonkyu Lim `[一作]` (KFTC), Yongjae Lee `[通讯]` (UNIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将账户级反洗钱检测重新定义为多实例学习问题，提出一种保留交易信息的节点分类器，该模型仅使用账户级标签进行训练，并在推理时为每笔交易提供排序与归因。

**💡 创新点**

创新点在于：① 识别并解决了聚合不匹配（aggregation mismatch）问题；② 在实例编码后采用特征维度级的最大池化（feature‑wise max）进行选择性聚合，既保留稀疏的可疑交易信号，又不显著增加误报；③ 通过共享编码器和池化策略实现了在无交易标签监督下的交易排序与归因，提升了模型在账户级任务上的实用性。

**🔧 技术方法**

使用的技术包括：多实例学习框架；共享两层 MLP 编码器；特征维度最大池化和标量最大池化等聚合策略；对比实验中采用的图网络（GCN、GAT、GraphSAGE、H2GCN）、传统表格模型（LR、MLP、XGBoost）以及交易级监督模型（Multi‑GIN、FraudGT）。

**📊 数据集**

使用的数据集：AMLworld、AMLNet、AMLSim、Elliptic++（比特币交易网络）以及一条非公开的跨行结算网络（KFTC）。

**📈 对比分析**

通过 AP、AUROC、F1、G‑mean 等指标对账户级预测进行比较，并在 Hit@k 与 MAP 上评估交易排序。与所有基线相比，该方法在 AP 上取得最高分，在大多数数据集上达到最优或次优，且在记忆检索（Recall@100/500/1000）中表现优于其他账户级模型。交易排序方面，其表现与直接使用交易标签训练的模型相当，远超随机排序。

**⚠️ 局限性**

局限性：① 采用将交易标签粗粒化为账户窗口标签的人工构造，无法直接反映真实调查中多因素、延迟报告和调查员判断的标签；② MIL 假设每个正样本至少有一笔可疑交易，实际案件可能涉及多笔交易或外部信息；③ 仅在固定时间窗口内评估，未检验跨时段的泛化能力；④ 真实部署时需要考虑标签噪声、可解释性与监管合规等问题。

---

## 22. Optimization of Joint Beamforming in RIS-aided Multi-carrier and Multi-user MIMO Systems

**arXiv ID:** 2611.05796 | [PDF](https://arxiv.org/pdf/2611.05796v1)

**作者:** Nuwan Balasuriya `[一作]` (University of Manitoba), Ekram Hossain `[通讯]` (University of Manitoba)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个物理一致的RIS辅助多载波MIMO系统联合波束成形框架，并实现了所有RIS元件同时服务多载波。

**💡 创新点**

利用多次反射和物理一致模型，在单一stub长度设置下实现多载波相位控制；并引入深度展开学习降低在线优化复杂度。

**🔧 技术方法**

多载波RIS系统建模、联合主动/被动波束优化、梯度下降与深度展开、stub长度离散化、物理一致散射矩阵。

**📊 数据集**

仿真基于高斯随机信道与路径损耗模型，参数设定为N=8、K=8、L=64、频率5GHz等。

**📈 对比分析**

与分区RIS、传统单载波、物理一致单载波以及主动RIS等方法对比，提升约20% sum‑rate，且在多载波数与stub分辨率上均优于对比方案。

**⚠️ 局限性**

算法仍受限于多载波间频率差异、stub长度精度、硬件功耗与响应时间，且仿真未考虑多用户干扰与非理想硬件效应。

---

## 23. Construct Validity of Coding Benchmarks for LLM-based Agentic Coding Tools

**arXiv ID:** 2611.05665 | [PDF](https://arxiv.org/pdf/2611.05665v1)

**作者:** Alexander Berndt `[一作]` (Heidelberg University), Sebastian Baltes `[通讯]` (Heidelberg University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对九个主流agentic coding工具评估基准进行构念效度分析，识别其声称测量的构念、使用的指标以及潜在混淆因素。

**💡 创新点**

提出系统化评估基准构念效度的方法，揭示当前基准在目标构念明确性、指标覆盖度与属性标准化方面的不足，并给出改进建议。

**🔧 技术方法**

采用定性文本分析结合人工标注与LLM辅助提取构念的方法，对基准文献进行案例归纳与指标评估。

**📊 数据集**

引用了DeepSWE、FrontierCode、FrontierSWE、SWE-Atlas、SWE-bench Multilingual、SWE-bench Multimodal、SWE-bench Pro、Terminal-Bench和Vibe Code Bench等九个公开基准的相关文献。

**📈 对比分析**

未进行模型性能比较；通过对基准报告的对比发现多数基准仅使用通过率等单一指标，缺乏多维度度量，难以全面评估agent的编码能力。

**⚠️ 局限性**

研究仅基于文献分析，未对基准的实际测量流程进行系统验证；样本覆盖有限，且缺乏对指标内部一致性与预测效度的实证评估。

---

## 24. Confucius4-R2T2: A Stable-Prefix Streaming ASR Model for Real Real-Time Transcription

**arXiv ID:** 2611.06043 | [PDF](https://arxiv.org/pdf/2611.06043v1)

**作者:** Daniel Huang `[一作]` (NetEase Youdao), Yitao Duan `[通讯]` (Uc San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种稳定前缀流式语音识别模型R2T2，能够在不修改已发布文本的前提下实时输出文本；

**💡 创新点**

引入最长稳定前缀（LSP）概念，并通过LSP学习将全上下文模型转化为稳定前缀模型；

**🔧 技术方法**

使用Qwen3-ASR-1.7B的后训练、冻结音频编码器、双向注意力窗口、专门的等待标记、缓存复用、有限上下文和预填充验证等技术；

**📊 数据集**

对中文和英文共1,500小时的自有数据进行后训练，同时在公开的多语言基准（英语、中文及10种其它语言）上进行评测；

**📈 对比分析**

与多款开源与专有流式ASR系统（X-ASR、WhisperRT、Nemotron3.5、Voxtral Realtime等）进行对比，R2T2在英语宏观平均WER 5.76%（低于所有开源/专有对手）和中文宏观平均CER 6.01%，且回溯发射延迟低于主流API，显示出在准确率与延迟之间取得较好平衡；

**⚠️ 局限性**

仍受限于较长上下文对模型性能影响、对多语言泛化的进一步验证与部署时的实时因子（RTF）评估尚需完善，以及对词级别发射延迟的精细化测量待补充。

---

## 25. Generative Retrieval under Real-Time Supply Constraints in Local Commerce

**arXiv ID:** 2611.06323 | [PDF](https://arxiv.org/pdf/2611.06323v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 26. When Does a Cross-Device Run Justify PASS? Obligation-Level Evidence Qualification under Partial Observability

**arXiv ID:** 2611.06013 | [PDF](https://arxiv.org/pdf/2611.06013v1)

**作者:** Zimu Liu `[一作]` (Ant Group), Liang Yu `[通讯]` (Ant Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现可执行的证据合同，先进行证据资格认定再评估断言，针对跨设备测试中证据来源混乱的问题。

**💡 创新点**

创新点在于将证据资格与断言评估分离，显式绑定观察主体、剧本与期望，构建“闭包（Closure）”与“图（Graph）”两种实现，显著降低误判。

**🔧 技术方法**

技术包括可执行证据合同定义、基于键映射的确定性关联（Closure）、图路径检查的图实现（Graph）、LLM辅助的证据解析与判断、以及离线冻结评估框架。

**📊 数据集**

使用了两组数据集：12个种子控制实验（共2,304条主张）和40个现场试验剧本（共80条主张，40个物理事件），涉及多种设备与业务流程。

**📈 对比分析**

比较方法为在相同记录与业务策略下，对Atomic‑Flat、Serialized‑Flat、Closure三者做对照实验；在现场试验中对Graph与Flat两实现做对照。结果显示Closure将假正例率从160降至0，PASS召回率提升至100%；Graph在80条主张中达95%准确率，仅有3个假正例。

**⚠️ 局限性**

局限性包括：实验设备与流程有限，缺乏独立公开的完整数据包；LLM解析与图结构需手工编写契约，难以自动迁移；仅评估了判定阶段，未覆盖恢复与清理逻辑；缺乏对不同模型或参数变化的鲁棒性评估。

---

## 27. Private Online Release of Adaptive Linear Queries

**arXiv ID:** 2611.05602 | [PDF](https://arxiv.org/pdf/2611.05602v1)

**作者:** Aleksandar Nikolov `[一作]` (University of Toronto), Haohua Tang `[通讯]` (University of Toronto)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了在差分隐私约束下，在线回答线性（计数）查询的问题，并提出了一种能对自适应查询实现多对数级竞争率的算法；

**💡 创新点**

创新点在于将在线分解（online factorization）与分层欧几里得迭代数据库构造（layered Euclidean IDC）相结合，利用扩展的目标充电（target charging）框架证明自适应查询的误差仅相较于离线最优分解多项式对数级；

**🔧 技术方法**

技术上主要使用了在线分解算法、分层IDC、γ₂范数优化、拉普拉斯/高斯噪声机制以及改进的目标充电分析；

**📊 数据集**

实验验证并未使用具体数据集，研究基于理论的直方图表示和离散域 N 的假设数据；

**📈 对比分析**

与离线最优分解和先前的在线无偏查询算法相比，误差上界为 O(γ₂(Q)·polylog(N))，下界为 Ω(γ₂(Q)·√(log N)/loglog N)，表明误差差距仅在多对数级；

**⚠️ 局限性**

局限性包括：需要数据量足够大才能达到理论误差界；分层IDC实现复杂，常数项可能较大；下界仅给出 √(log N) 阶乘因子，尚未确定是否可进一步逼近；

---

## 28. PathoS2VD: From 2D Histopathology to Pseudo Z-Stack Representations

**arXiv ID:** 2611.05974 | [PDF](https://arxiv.org/pdf/2611.05974v1)

**作者:** Jiamu Wang `[一作]` (Korea University), Heounjeong Go `[通讯]` (Asan Medical Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出 PathoS2VD，一个利用 2D 病理图像生成伪 Z‑堆叠（3D 视角）并用于癌症诊断的生成框架。

**💡 创新点**

创新点包括：①将视频扩散模型（Latent Video Diffusion Model）重新解释为 Z‑轴生成模型；②引入基于相对模糊度的遮掩策略以避免光学畸变影响；③设计 Prior‑Preserving Mixture Fine‑Tuning（MFDA）实现跨域自适应，同时保持 Z‑轴形态先验；④采用共享 Null‑空间加速多块伪 Z‑堆叠的迭代扩展。

**🔧 技术方法**

核心技术为：拉伸扩散模型（Latent Diffusion Model）+ 3D 卷积与时间注意力；基于 CLIP 的语义引导；混合微调训练策略；自定义模糊度测量与遮掩；伪 Z‑堆叠的迭代 out‑painting。

**📊 数据集**

使用 80 份 19‑层 Z‑堆叠的前列腺病理 WSIs（≈592k 高质量补丁）作为源数据；下游评估使用 AGGC2022（5 类分级）和 SICAPv2（4 类分级）两个 2D 前列腺数据集。

**📈 对比分析**

与 Pix2Pix、OpenSoraPlan 等基线对比，PathoS2VD 在 SSIM、LPIPS、FID、FVD 及多种 3D 结构张量指标（Orientation Similarity、ST Frobenius 等）上均优于对手；在 AGGC2022 patch‑level 和 SICAPv2 slide‑level 的分类任务中，伪 Z‑堆叠通过 Max‑Pool、Self‑Attention、DSMIL 等聚合方式均提升准确率（最高约 +1.1% 绝对准确率）。

**⚠️ 局限性**

局限性：模型仅在前列腺良性组织上训练，难以覆盖高危癌变的复杂形态；Z‑堆叠生成依赖于高质量 Z‑stack 数据，现实临床中此类数据稀缺；扩展到其他器官和病理类型需进一步验证。

---

## 29. Hotplug Coded Caching: Embedding the YMA and Interference Elimination Schemes via Reed--Muller Codes

**arXiv ID:** 2611.05772 | [PDF](https://arxiv.org/pdf/2611.05772v1)

**作者:** Daniela Tuninetti `[一作]` `[通讯]` (University of Illinois Chicago), Daniela Tuninetti (University of Illinois Chicago)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文研究了热插拔编码缓存模型，其中K个用户根据N个文件填充大小为M的缓存，但在交付阶段只有未知的A<K个用户处于活动状态。我们展示了在未编码放置下，离线用户在中间内存大小上会导致严格的负载惩罚，因此需要编码放置。

**💡 创新点**

创新点在于证明了每个YMA方案的点在热插拔模型中是可实现的，且对于每个K≥A，用户缓存的多项式限制与超平面相关联，使用了投影Reed-Muller编码。

**🔧 技术方法**

使用了Reed-Muller编码和Reed-Solomon编码技术。

**📊 数据集**

使用了多个经典缓存方案的点，包括YMA、GV和IE方案，具体数据集未明确给出。

**📈 对比分析**

通过与经典方案的比较，证明了热插拔模型中的负载不依赖于K，并且在特定条件下，热插拔模型的性能与经典模型相当。

**⚠️ 局限性**

限制在于是否最优热插拔权衡仅依赖于A仍然是一个开放问题。

---

## 30. Energy-efficient End-to-End Communication System with Spiking Neural Networks

**arXiv ID:** 2611.05550 | [PDF](https://arxiv.org/pdf/2611.05550v1)

**作者:** Pooja Plavilayil Anil `[一作]` (University of Liverpool), Nguyen Van Huynh `[通讯]` (University of Liverpool)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于脉冲神经网络（SNN）的端到端（E2E）通信系统编码器，旨在降低编码器推理能耗并保持可靠的消息恢复。

**💡 创新点**

创新点在于将正弦时间调制、加权时间聚合和尖峰惩罚相结合，使得SNN能够以稀疏尖峰表示静态一热编码消息，并生成可区分的连续发射信号。

**🔧 技术方法**

使用了LIF神经元、SuperSpike 近似梯度、SNN时间步调制、权重时间聚合以及基于SPiNNaker2硬件的能耗模型等技术。

**📊 数据集**

使用了16类一热编码消息、i.i.d. Rayleigh衰落+AWGN通道，分别在1×1、2×2、4×4、8×8和16×16天线配置下训练和测试。

**📈 对比分析**

通过与同样深度的ANN编码器在相同解码器下进行交叉熵损失训练，并在0–12 dB SNR范围内对BER和编码器能耗进行比较，结果显示SNN在保持相近或更低BER的同时，编码器能耗平均下降约35%–70%。

**⚠️ 局限性**

局限性包括仅评估编码器推理能耗，未验证硬件实现；仅使用理想化的Rayleigh + AWGN通道；未考虑解码器、CSI压缩和信道估计能耗；以及仅在有限的SNR和天线规模上测试。

---

## 31. Constraint-Preserving Flow Matching for Spectral Graph Generation

**arXiv ID:** 2611.05436 | [PDF](https://arxiv.org/pdf/2611.05436v1)

**作者:** Anita Yang `[一作]` (University of Tokyo), Yohan Jung `[通讯]` (Jeonbuk National University)

**通讯引用:** 186 | [OpenAlex ID](https://openalex.org/A5102786426)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Laplace-Constrained Stiefel Flow Matching（LaSFM），一种在生成图谱时同时保持特征向量正交、拉普拉斯矩阵零空间约束和正定约束的连续生成框架。

**💡 创新点**

创新点在于将拉普拉斯零空间条件和正定约束融入Stiefel流匹配，并通过双重投影保证生成路径始终保持在约束子流形上，从而避免谱结构失真。

**🔧 技术方法**

主要技术包括基于流匹配的向量场学习、Stiefel流形几何、软正则化的正定参数化、梯度引导下的约束保持和双向交叉注意力架构。

**📊 数据集**

使用Planar、SBM、Community-small和QM9四个公开图数据集进行评估。

**📈 对比分析**

与SPECTRE、GGSD、SFMG等现有谱生成方法相比，LaSFM在谱空间MMD、图级度分布、聚类、轨道统计以及分子生成任务的属性误差和有效率上均表现出更优或相近的性能。

**⚠️ 局限性**

局限性包括：①截断谱仍无法保证非正离散拉普拉斯的非正离散项；②需要额外的图重构步骤才能得到离散图结构，重构过程可能引入误差。

---

## 32. C-WFDM: A ControlNet-Guided Wavelet-Fusion Diffusion Model for Multimodal Brain MRI Completion

**arXiv ID:** 2611.05737 | [PDF](https://arxiv.org/pdf/2611.05737v1)

**作者:** Muhammad Nabi Yasinzai `[一作]` (Auckland University of Technology), Mangor Pedersen `[通讯]` (Auckland University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种基于ControlNet的多模态脑MRI缺失模态补全方法C-WFDM，能够利用同一受试者已获取的不同MRI对来生成缺失的T2w、FLAIR和T1Gd图像。

**💡 创新点**

创新点在于：①将预训练的Wavelet‑Fusion Diffusion Model (WFDM)扩展为Subject‑Specific Completion，通过ControlNet适配器引入同一受试者的已采集模态作为条件；②采用基于真实模态出现频率的顺序补全策略（T1w→T2w→FLAIR→T1Gd），更贴合实际大规模合并数据集的模态可用性分布；③在保持原WFDM多模态生成能力的同时实现了单体内缺失模态的高质量重建。

**🔧 技术方法**

技术手段包括：Wavelet‑Fusion VAE对3D MRI进行低维压缩、在VAE潜空间中进行条件扩散、ControlNet在潜空间中以多模态条件引导U‑Net去噪，训练时仅更新ControlNet适配器，采用自适应源选择与后验采样调度提高对合成中间模态的鲁棒性。

**📊 数据集**

使用数据集：BrainScape公开脑MRI资源（157个公开数据集，45,880个预处理扫描，包含T1w、T2w、FLAIR、T1Gd），以及其BraTS子集进行额外验证。

**📈 对比分析**

通过与ALDM、cWDM、Pix2Pix三种基准模型在SSIM、PSNR、MAE三指标下对T2w、FLAIR、T1Gd进行对比，实验结果显示C‑WFDM在T2w（SSIM≈0.923）和FLAIR（SSIM≈0.890）上均取得最高分；T1Gd表现略逊于cWDM，但整体仍优于其他方法。实验C（仅T1w输入）进一步证明C‑WFDM在顺序补全中仍保持竞争力。

**⚠️ 局限性**

局限性包括：①顺序补全易产生误差累积，导致后续模态生成受前一步误差影响；②T1Gd作为最稀缺且对比增强难度高的模态，模型表现仍不佳；③仅验证了BrainScape的主流补全路径，未覆盖所有可能的模态组合；④缺少未获取目标模态的无监督评价，当前评价依赖已采集目标作为参考。

---

## 33. SynMOE: A Syntax-Aware MOE-Guided Approach for PEFT-Based Code Intelligence Learning

**arXiv ID:** 2611.06069 | [PDF](https://arxiv.org/pdf/2611.06069v1)

**作者:** Shuo Liu `[一作]` (City University of Hong Kong), Yicheng Sun `[通讯]` (City University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种语法感知的MOE（SynMOE）方法，将抽象语法树（AST）中的语法标签与参数高效微调（PEFT）技术相结合，以提升代码语言模型在代码检索、代码克隆检测和行级代码补全等任务中的性能。

**💡 创新点**

创新点在于：①通过Token-Label对齐算法为代码语言模型的每个子词分配AST语法标签；②在MOE门控网络中引入语法标签，实现语法驱动的专家路由，从而显著增强了PEFT在代码域的表达能力。

**🔧 技术方法**

核心技术包括：抽象语法树（AST）提取与语法标签生成、Token-Label对齐算法、语法感知的MOE模块（结合LoRA或(IA)^3专家）以及标准的参数高效微调框架。

**📊 数据集**

实验使用的公开数据集包括：CosQA（代码检索）、POJ-104（代码克隆检测）、JavaCorpus（行级代码补全）以及多种代码语言模型（CodeBERT、UniXcoder、CodeGPT、Deepseek‑Coder、Qwen2.5‑Coder）对应的数据。

**📈 对比分析**

通过与全量微调（FT）以及LoRA、(IA)^3、LoKr、LoHa、AdaLoRA等多种PEFT基线对比，SynMOE在三大任务上均实现了显著提升（平均提升约1.5%–8.8%），且在低资源场景下更是保持或超过FT的表现，同时保持参数高效与训练成本低。

**⚠️ 局限性**

局限性包括：仅验证了三类任务和五个模型规模，未评估更大模型或其他任务（如代码翻译、程序修复）；依赖AST生成和Token对齐的规则实现，可能在语言或工具链差异较大的代码中表现不佳；MOE模块增加了一定的计算开销，需要在极端资源受限环境下进一步优化。

---

## 34. PEEK: One-Step-Look-Ahead Exposure Bias Correction for Diffusion Sampling

**arXiv ID:** 2611.05407 | [PDF](https://arxiv.org/pdf/2611.05407v1)

**作者:** Yunwei Bai `[一作]` (National University of Singapore), Tsuhan Chen `[通讯]` (National University of Singapore)

**通讯引用:** 20726 | [OpenAlex ID](https://openalex.org/A5000454484)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关的确定性采样插件PEEK，通过一阶前瞻插值纠正扩散模型的曝光偏差，提升低 NFE 下的图像生成质量。

**💡 创新点**

创新点在于：1）利用一阶前瞻预测与当前预测的插值来构造下一个噪声状态；2）设计了基于度量（FID）的贪心插值调度搜索；3）实现了 DPM-Solver 兼容的求解方向变体；4）通过前缀和噪声重基缓存显著减少额外推理计算。

**🔧 技术方法**

使用的技术包括：确定性扩散采样、一次性前瞻预测、线性插值、度量驱动的调度搜索、缓存机制、以及与 DPM‑Solver 的兼容性改造。

**📊 数据集**

在像素空间与潜空间的多种扩散模型上进行实验，数据集涵盖 CIFAR‑10、LSUN Bedroom、ImageNet 与 MS‑COCO，且覆盖无条件、标签引导与文本引导三种设置。

**📈 对比分析**

与 NFE 匹配的基线采样器（包括 Euler、Heun、UniPC 等）进行对比，PEEK 在相同的模型评估预算下 FID 下降最多可达 49.3%，显著提升生成质量；调度搜索仅耗时 2.2 分钟。

**⚠️ 局限性**

局限性包括：1）需要额外的 denoiser 评估，插值位置对效果敏感；2）调度搜索在不同模型上仍需一次手动校准；3）未验证在更大模型或极低 NFE 极限下的泛化能力。

---

## 35. Classifying K-Geodetic Graphs Via Combinatorial Designs and Distance-regularity

**arXiv ID:** 2611.05762 | [PDF](https://arxiv.org/pdf/2611.05762v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

---

## 36. CAPTOR: text-encoder-free grounding by counting caption co-occurrence over the visual words of a frozen encoder

**arXiv ID:** 2611.05599 | [PDF](https://arxiv.org/pdf/2611.05599v1)

**作者:** Elad Yoshai `[一作]` (Tel Aviv University), Natan T. Shaked `[通讯]` (Tel Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对冻结视觉 Transformer 的 patch 进行 k‑means 聚类得到视觉词，利用已标注字幕统计每个词在出现该术语时的比例差异，构建无文本编码器、无梯度训练的 CAPTOR 表，实现术语与图像 patch 的兼容度映射，从而实现开源词段落分割。

**💡 创新点**

核心创新在于用仅基于字幕的计数与置换零归一化方法，直接估计术语‑patch 兼容度矩阵，完全消除了对文本编码器与对齐训练的依赖，并通过视觉词聚类与联合估计提升细粒度分割效果。

**🔧 技术方法**

采用冻结的视觉 Transformer（如 DINOv2、CONCH 等）作为特征提取器，使用 k‑means 对 patch 进行聚类得到视觉词；利用置换测试计算术语统计量并归一化；采用硬/软词分配、传播平滑和联合词估计等技术构建兼容度表。

**📊 数据集**

在自然图像上使用 COCO、Pascal VOC、Cityscapes 数据集；在病理图像上使用 PathCap、BCSS、LUAD‑HistoSeg 三个公开病理数据集进行评估。

**📈 对比分析**

与训练免费基线（ProxyCLIP、SCLIP、Talk2DINO 等）以及使用文本编码器的基线进行对比；CAPTOR 在 COCO、VOC、Cityscapes 的 mIoU 均超过或持平训练免费基线，在病理图像上对无文本编码器的 UNI、Phikon 的分割性能显著优于基线。

**⚠️ 局限性**

限制包括对罕见术语统计不稳定、硬词分配导致细粒度分割受限、依赖字幕质量与覆盖范围、以及未能处理属性、关系、否定等更复杂语言表达。

---

## 37. Timing is Everything: Inferring Temporal Membership in Continually Updated Models

**arXiv ID:** 2611.05781 | [PDF](https://arxiv.org/pdf/2611.05781v1)

**作者:** Jie Fu `[一作]` (Stevens Institute of Technology), Wendy Hui Wang `[通讯]` (Stevens Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并攻击持续更新深度学习模型的时间维度成员信息泄露，提出黑盒 Temporal Membership Inference Attack (TMIA)。

**💡 创新点**

首次将成员推断扩展到时间轴，定义时间戳级和区间级成员推断，并用滑动窗口 BiLSTM 捕获更新序列的时序模式。

**🔧 技术方法**

使用滑动窗口 BiLSTM 与更新对齐的影子模型训练、时间序列标签化及有效区间构造算法。

**📊 数据集**

在三种语言模型（Pythia‑70M、T5、GPT‑2）和三种图像分类网络（ResNet‑18、SimpleCNN、MobileNet）上，分别使用 SST‑5、News20、MNLI、CIFAR‑10、SVHN、CINIC‑10 及 LLaMA‑3B/8B+SQuAD/SimpleQA。

**📈 对比分析**

与两种基线（TS‑MIA、UP‑MIA）对比，TMIA 在 timestamp‑level 和 interval‑level 上 F1 分别提升 20–30%，在多种模型、数据集与更新设置下保持稳健。

**⚠️ 局限性**

仅针对黑盒查询，依赖完整或大部分快照可观测，对极低精度输出或大规模分布差异的影子模型效果会下降，且防御方法仍有显著精度损失。

---

## 38. $\mathcal{C}^\infty$ Smooth Trajectory Planning with Application to Soft Robots

**arXiv ID:** 2611.05615 | [PDF](https://arxiv.org/pdf/2611.05615v1)

**作者:** Reinhard M. Grassmann `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Sami Haddadin `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出一种可生成无穷可微（C∞）平滑轨迹的点到点轨迹生成器，适用于软连续机器人及其他多自由度机器人。

**💡 创新点**

创新点在于构造了基于分段平坦函数的C∞光滑路径原语，并通过几何缩放与动力学极限自适应实现无限阶连续性。

**🔧 技术方法**

采用了函数组合理论、几何缩放、极限约束求解、符号微分与数值稳健实现等技术手段。

**📊 数据集**

实验数据来源于高保真柔性机器人仿真（SoRoSim、GVS模型）以及7-DoF Franka FR3机械臂的硬件测试。

**📈 对比分析**

与多阶多项式S曲线及Ruckig等现有方法比较，实验显示该方法在满足同等速度/加速度/jerk极限时，振动能量更低、加速度RMS更小、轨迹持续时间更短或相近。

**⚠️ 局限性**

局限性包括缺乏对真实软机器人硬件的验证、路径原语无初等反导数、计算量相对较大以及在非直线几何路径时仍需进一步研究。

---

## 39. Augmentation with Reinforcement Transformation for Low-textured Scientific Sketch Image Retrieval

**arXiv ID:** 2611.05408 | [PDF](https://arxiv.org/pdf/2611.05408v1)

**作者:** Zhiqiang Wang `[一作]` (Florida Atlantic University), Xingquan Zhu `[通讯]` (Florida Atlantic University)

**通讯引用:** 17783 | [OpenAlex ID](https://openalex.org/A5084641325)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于强化学习的自适应数据增强框架 ART，结合 VICReg 自监督训练，专门用于低纹理科学草图（如海豚瘢痕）检索。

**💡 创新点**

创新点在于把变换选择建模为多臂赌博机，使用无标签的结构奖励（保留性、特征一致性、变异性和危害性）来引导原子变换组合，并通过共享信用机制在训练前生成变换视图银行。

**🔧 技术方法**

采用了 CLIP ViT-B/32 视觉骨干、Mask R‑CNN 痕迹检测、VICReg 自监督损失、UCB 多臂赌博机决策、共享信用奖励和位置加权聚合等技术。

**📊 数据集**

使用了 Mote Marine Laboratory 提供的佛罗里达海豚手绘瘢痕草图数据集，包含 1,645 张草图、12,221 个瘢痕。

**📈 对比分析**

通过与均匀采样、保守固定增强、RandAugment 和传统全增强等方法对照，评估 Top‑1、Top‑5、MRR 等指标。ART 在变换痕迹检索上平均 Top‑1 达 71.40%（比均匀采样高 2.57%），在位置感知草图检索上 Top‑1 55.84%，显著优于 CLIP 的 18.18%。

**⚠️ 局限性**

局限性包括：仅在已知变换的稳健性上验证，未评估对新个体或跨数据库迁移的泛化；奖励函数为固定设计，缺乏针对不同任务的调优；检索仍采用线性搜索，规模扩展受限。

---

## 40. NextGAN: Towards Simple and Scalable GAN Training for Image and Video Generation

**arXiv ID:** 2611.06270 | [PDF](https://arxiv.org/pdf/2611.06270v1)

**作者:** Xinhao Lou `[一作]` (Westlake University), Tao Lin `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通过对随机真实-生成特征对的梯度进行回归来训练标量判别器，从而在预训练生成器上实现一次或少量步图像与视频生成。

**💡 创新点**

创新点在于直接监督判别器梯度而非传统的值差距；通过梯度回归得到分布匹配场，并证明其等价于带梯度能量惩罚的对抗目标，形成新的负Sobolev不等式判别器。

**🔧 技术方法**

技术包括：冻结特征空间（MAE、SigLIP、ResNet、VideoMAE 等）中的判别器梯度回归；使用 LoRA 微调大规模生成器；基于梯度场的潜在/像素空间潜能更新；对抗或回归两种实现方式；并通过反向模拟实现少步视频采样。

**📊 数据集**

主要数据集：ImageNet 256×256（类别条件图像生成）和 VBench（视频分辨率 832×480×81，文本到视频生成）。

**📈 对比分析**

与多种基线（GAN、WGAN、TBSM、流式/自回归模型、以及其他无教师查询方法）对比；在 ImageNet 上，一步生成 FID 为 1.54（Latent DiT-XL/1）/2.87（像素 JiT-B），Adversarial 版本更低为 1.36；在 VBench 上，1.3B 模型一、二、四步采样分别达到 80.80、83.13、82.72，总分在同规模方法中最高；14B 模型亦表现竞争力。

**⚠️ 局限性**

局限性包括：仅对冻结特征空间内的分布匹配有保证；对生成器的全梯度更新仍需间接体现；未对从零训练大规模生成器进行实验；并且在不同特征空间下的表现可能受限于特征对齐程度。

---

## 41. GREP: Grounding Generated Room Evaluation in Principles

**arXiv ID:** 2611.05802 | [PDF](https://arxiv.org/pdf/2611.05802v1)

**作者:** Mohan Cao `[一作]` (University of Washington), Yuan Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GRE评分框架，用来从物理有效性、视觉连贯性和人类使用逻辑三方面对生成场景进行评估；

**💡 创新点**

创新点在于将评估过程自动化，评估代理能够自主识别问题、收集证据并作出综合判断，避免传统单一硬指标的局限；

**🔧 技术方法**

利用视觉语言模型（VLM）辅助评估器以及自主评估代理技术；

**📊 数据集**

实验基于公开的3D室内场景数据集（如Matterport3D、AI2-THOR等）以及作者自建的评估集；

**📈 对比分析**

与现有指标（如物理碰撞检测、视觉一致性分数等）进行对比实验，结果显示GRE在准确捕捉物理错误、视觉不连贯和使用逻辑错误方面的精确度显著提高；

**⚠️ 局限性**

局限性包括对VLM的依赖导致模型误判可能出现，评估代理仍需大量人工验证支持，且数据集覆盖范围有限，可能不足以涵盖所有复杂场景。

---

## 42. Learning Noise Optimization for Reward Alignment in Few-Step Generative Models

**arXiv ID:** 2611.06255 | [PDF](https://arxiv.org/pdf/2611.06255v1)

**作者:** Hongeun Kim `[一作]` (KAIST AI), Jong Chul Ye `[通讯]` (KAIST AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出 LeNO 框架，利用学习到的残差传输在初始噪声空间对冻结的确定性生成器进行奖励提升的噪声校正，避免了每个测试样本的反复搜索；

**💡 创新点**

创新点在于将每样本的噪声优化经验通过状态相关的残差映射进行摊销，使用 Local GRPO 在仅有标量奖励的条件下学习残差均值，并给出高斯噪声下的 KL 上界作为正则化；

**🔧 技术方法**

核心技术包括：局部梯度自由策略（Local GRPO）、随机残差传输（含范数保持投影）、KL 正则化与上界推导、LoRA 细调残差均值网络、单步修正等；

**📊 数据集**

实验使用文本生成领域的 SDXL‑Turbo 与 SANA‑Sprint、各种奖励函数（Aesthetic、PickScore、ImageReward 等），蛋白质结构领域使用一阶 RMF 模型并结合 ProteinMPNN 与 ESMFold 进行设计可行性评估；

**📈 对比分析**

与 Best‑of‑N、ψ‑Sampler、ZeNO（推断时优化）以及 HyperNoise、GoldenNoise（摊销方法）对比，LeNO 在单奖励和多奖励场景下均取得最高奖励，且仅需 2–3 次 NFE；蛋白质结构生成中，LeNO 在保持多样性的前提下显著提升设计可行性，甚至优于 ZeNO 的 30 步搜索；

**⚠️ 局限性**

局限性在于训练阶段仍需对每个 anchor 进行多次生成器与奖励评估，若奖励评估成本极高则训练成本高昂；仅在推断时增加轻量级的残差采样，开销相对可接受。

---

## 43. Whiel: A Neurosymbolic Platform for Certified Relational Program Verification

**arXiv ID:** 2611.05612 | [PDF](https://arxiv.org/pdf/2611.05612v1)

**作者:** Jesse Comer `[一作]` (University of Pennsylvania), Val Tannen `[通讯]` (University of Pennsylvania)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种神经符号平台（Whiel），可通过 LLM 生成循环不变量或反例，并使用符号验证器对命题 Hoare 三元组进行验证，最终可生成 Lean 证明或反例证书，保证验证结果可信。

**💡 创新点**

创新点包括：
• 统一把数据库相关问题（等价、包含、递归查询、增量视图维护等）编码为 Hoare 验证问题；
• 引入预言变量与分层不变量（ladder invariants）与改进的 Houdini 算法，实现对复杂循环的可递归推理；
• 通过 LLM 交互式搜索、反馈循环、工具/技能支持，显著提升自动化程度；
• 采用独立的 Lean+Vampire 证明重建作为可验证证书，最小化可信根基。

**🔧 技术方法**

主要技术包括：
• 预言变量与终态后果的形式化；
• 分层（leveled）不变量与层级 Houdini 推理；
• LLM（GPT‑5.5/5.6/6）生成不变量/反例；
• SMT/SMT‑solver（Z3、CVC4）做一阶有效性与反例搜索；
• Lean proof assistant + Vampire 用于证明重建；
• Rust/Python 控制器管理交互流程。

**📊 数据集**

使用了 86 条验证任务的基准集，涵盖经典数据库问题（递归查询等价、视图重写、增量维护、约束检查）以及实际系统（PostgreSQL、NetworkX、Soufflé）和 LLM 生成的程序。基准中 69 条有效、17 条无效，已公开存档。

**📈 对比分析**

与两种基线对比：
• 直接让 LLM 给出完整证明（无交互）
• 单轮 LLM 提交单条不变量或反例

- 在 GPT‑5.5 下，Whiel 在 600 s 预算内成功 73/86（85%）条任务，平均耗时 189 s，成本 0.49 美元；
- 单轮基线仅 46/86；
- 强化模型 GPT‑5.6/6 进一步提升至 81/86（≈94%）和 85/86（≈99%）。
- 加入工具与技能可将解题率提升至 77/86（90%），但平均轮数与成本略有上升。

**⚠️ 局限性**

局限性：
• 依赖 SMT 求解器，对某些大规模或复杂谓词的验证仍可能超时或返回不确定；
• 证明重建与证书生成对内存与时间有较高要求（约 70 s/验证，最大 6 GB 内存）；
• 目前仅支持无算术、无新值创建的纯关系程序；
• 预言变量与分层不变量的自动构造仍受模型表达能力限制，对需要深层终态约束的任务依赖更强的 LLM。

---

## 44. On the Generalization Limitation of On-Policy Distillation for Post-RL Consolidation

**arXiv ID:** 2611.05609 | [PDF](https://arxiv.org/pdf/2611.05609v1)

**作者:** Yichen Wang `[一作]` (University of Chicago), Zijun Wei `[通讯]` (ByteDance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究评估了在后训练阶段使用的On‑Policy Distillation (OPD) 与多教师OPD (MOPD) 对离散（OOD）保留和组合泛化的影响，并在此基础上提出几何补丁和基线保持回放两种方法来缓解 OOV 遗忘。

**💡 创新点**

创新点包括：① 证明近乎完美的 IND 蒸馏并不保证 OOD 保留或泛化；② 从教师能力与数据覆盖双瓶颈的视角剖析 OOV 问题；③ 通过 Procrustes 注意力补丁和基线保持回放实现 OOV 性能提升；④ 对 OPD/MOPD 参数更新的谱集中度和投影结构进行定量分析。

**🔧 技术方法**

技术手段主要包括 OPD、MOPD、混合 RL、Procrustes 对齐补丁、基线保持回放、参数更新范数与能量分布分析、投影 R² 评估以及实验中的多任务混合与混合权重设置。

**📊 数据集**

使用的数据集涵盖数学推理（GSM8K、MATH、DAPO、MATH500、AMC23、AIME24/25、GSM‑Infinite、DocMath）、编码（Nemotron、HumanEval、MBPP、EvalPlus）、代理任务（BFCL、xLAM60K、ToolACE）、长上下文（DocQA‑RL‑1.6K、LongBench‑v2）、以及额外的 OpenR1 与 GSM‑Infinite 补充数据。模型主要以 Qwen3‑8B 作为学生，外部教师为 Qwen3‑32B。

**📈 对比分析**

比较方法：与自RL教师、外部通用教师、混合RL 及单任务 OPD 进行对比；评估 IND、OOD（如 AIME、AMC23 等）以及组合任务（如 LongBench‑v2 Code、SWE‑bench Verified）的表现。结果表明 OPD/MOPD 在 IND 几乎完美，但仍出现 OOD 遗忘；MOPD 相比混合 RL 降低了 OOD 遗忘率但仍低于基线；几何补丁和基线保持回放能在保持 IND 的前提下显著提升部分 OOD 分数。

**⚠️ 局限性**

局限性：仅在 Qwen3‑8B 上实验，教师/模型规模受限；未对混合 RL 与 MOPD 进行细粒度超参调优；OOV 标注基于后训练分布，缺乏更广泛的前训练覆盖；缺乏对不同模型架构或更大规模实验的验证；随机性和实验重复次数有限；几何更新与泛化之间的因果关系尚未得到实验验证。

---

## 45. Optimal LCA and Sublinear Estimation for Bipartite Matching in $Δ^{O(1/\varepsilon)}$ Queries

**arXiv ID:** 2611.06024 | [PDF](https://arxiv.org/pdf/2611.06024v1)

**作者:** Edith Cohen `[一作]` (Google Research), Mohammad Roghani `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种新的局部计算算法（LCA），能够在只查询图中极少部分的情况下，回答给定顶点的配对关系（mate query）以及估计最大匹配大小，并且在二分图上实现了近似匹配的整数输出。

**💡 创新点**

创新点主要有：①使用了基于连续优化的确定性局部更新方法，并通过精心设计的正则化实现每轮仅依赖常数跳数的局部更新；②引入了精确递归采样技术（exact recursive sampling），利用贝努利工厂（Bernoulli factory）实现不显式计算迭代状态即可得到每轮分布的随机抽样；③通过上述技术将之前在二分图上对匹配 LCA 的查询复杂度从 Δ^O(1/ε²) 降到 Δ^O(1/ε)，与已知的 Ω(Δ^Ω(1/ε)) 下界实现了指数上界的匹配。

**🔧 技术方法**

使用了以下技术：离散化的多重权重更新（multiplicative‑weights），贝努利工厂用于构造精确采样器，区域凸正则化（area‑convex regularizer）保证收敛，递归采样与价格响应的 Bernoulli factory 相结合，以及基于双图双覆盖的半整数化与整数化的本地舍入。

**📊 数据集**

论文未使用公开数据集；所有结果均基于理论分析与合成图模型（n 结点、最大度数 Δ 的无向图）。

**📈 对比分析**

与之前的 LCA 方案（如 Levi 等 2017 年的 (Δ/ε)^O(1/ε²) 复杂度）相比，本文在二分图上将查询复杂度从 Δ^O(1/ε²) 降到 Δ^O(1/ε)，实现了与下界 Δ^Ω(1/ε) 的匹配；在一般图上得到的半整数化匹配与之前的 O(Δ^O(1/ε²)) 估计器相比同样取得了线性上界。实验结果（理论证明）表明误差在 ±εn 内，且查询次数与 ε 的逆线性相关。

**⚠️ 局限性**

限制包括：①目前的结果仅在二分图上实现整数近似匹配，通用图仍只能得到半整数化匹配；②对于通用图的整数匹配 LCA 是否能达到 Δ^O(1/ε) 仍是未解的开放问题；③实现过程对随机性和精确算术要求较高，实际实现的常数因子较大；④在极高最大度数或非常稀疏图中，实际查询次数可能仍然较大。

---

## 46. When to Ask, When to Answer: Strategic Clarification for Ambiguous Reference Resolution in Dialogue

**arXiv ID:** 2611.05517 | [PDF](https://arxiv.org/pdf/2611.05517v1)

**作者:** Ibrahim Khebour `[一作]` (Colorado State University), Nikhil Krishnaswamy `[通讯]` (Colorado State University)

**通讯引用:** 576 | [OpenAlex ID](https://openalex.org/A5021596300)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究在引用游戏中何时提问以实现“战略澄清”，构建基于SIMMC 2.1的可控实验环境，并提出BACE（Belief‑Aware Critic with Epistemic tracking）方法，冻结LLM生成器仅训练一个轻量级的决策器。

**💡 创新点**

创新点包括：① 将提问决策与生成分离，使用信念状态与特征驱动的轻量级Critic；② 通过对照oracle的“黄金深度”实现对问答策略的精确奖励回传；③ 设计温度化的“温启动”训练，使RL快速收敛；④ 通过合成多问的训练样本填补真实数据中多问情境不足。

**🔧 技术方法**

技术手段包括：使用Qwen2.5‑7B‑Instruct作为LLM骨干，LoRA微调；监督微调（SFT）以模仿oracle；强化学习框架GRPO、ArCHer与BACE；贝叶斯信念追踪与信息增益特征；时间差分学习和TD目标的自回归回报估计。

**📊 数据集**

数据集主要是SIMMC 2.1购物对话数据，包含候选物体列表与属性标注；此外在训练阶段利用从真实数据中提取的合成多问样本，提升多问场景的覆盖率。

**📈 对比分析**

评估与比较：在devtest上与自然行为、SFT、GRPO、ArCHer进行对比。BACE在整体奖励上与SFT相当（≈0.94），在需要多次提问的ask‑2、ask‑3场景下的“hit‑and‑correct”率明显高于GRPO与ArCHer（分别为0.562/0.200 vs. 0.250/0.200），显示其在提问时机选择上更优。

**⚠️ 局限性**

局限性包括：仅使用文本信息，未涉及视觉输入；实验基于单一数据集，迁移到其他域需先获取候选列表；候选列表已给出，省略了候选检索步骤；脚本化用户回答过于确定，未考虑人类回答的模糊或错误；因此鲁棒性和通用性仍有待进一步验证。

---

## 47. Auditable Unlearning for LLMs

**arXiv ID:** 2611.05866 | [PDF](https://arxiv.org/pdf/2611.05866v1)

**作者:** Piyush Tiwary `[一作]` (Indian Institute of Science), Prathosh A. P `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨机器无学习的四个评价维度，并提出基于GapWiden和GapVeil的通用两阶段审计方法来增强模型的可审计性。

**💡 创新点**

创新点在于将审计性纳入无学习评估框架，并设计了GapWiden（增宽差距）和GapVeil（推理时裁剪采样）两种技术实现数据删除的可验证证据。

**🔧 技术方法**

采用对齐（hinge）损失提升忘记集与保留集的审计统计差异，并在推理阶段使用拒绝采样技术基于草稿模型重采样。

**📊 数据集**

在TOFU和MUSE数据集上评估十一种现有无学习方法，验证其在不同模型规模下的表现。

**📈 对比分析**

与oracle基准对比，所有方法在审计性与泄漏率上实现oracle等价，遗忘性能逼近oracle，保留性能对九种方法提升但仍低于oracle，显示出方法间的权衡。

**⚠️ 局限性**

局限性包括保留性能仍未完全达到oracle水平，对不同模型规模的泛化仍需进一步验证，以及在极大模型上实现两阶段方法的计算成本未充分评估。

---

## 48. Quo Vadis, Driving Scene Reconstruction? Occlusion-Aware Gaussian Densification

**arXiv ID:** 2611.06273 | [PDF](https://arxiv.org/pdf/2611.06273v1)

**作者:** Dung Do Tien `[一作]` (Ho Chi Minh City University Of Technology), Duc Dung Nguyen `[通讯]` (Ho Chi Minh City University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对单帧驾驶场景的前向视角合成，提出一个前向推理框架 OccSplat，能够识别并补全因前行视角变化导致的遮挡区域。

**💡 创新点**

创新点在于将遮挡处理转化为“遮挡感知高斯密集化”，通过高斯遮挡映射定位缺失几何区域，再利用高斯遮挡完成模块在三维空间中生成新的高斯原语，直接在三维表示层完成补全，而非仅在图像空间做像素填充。

**🔧 技术方法**

采用 3D 高斯 splatting 作为基础表示，结合卷积特征提取、体素融合网络估计深度、线性注意力机制进行几何与外观解耦推断，PixelRefiner（U‑Net）先修复渲染缺陷，再通过深度注意力与特征注意力得到补全几何与外观，最后使用预训练的高斯头生成补全原语。

**📊 数据集**

在 nuScenes 驾驶数据集上进行训练与评估，使用 700 场训练集、150 场验证集，图像分辨率 114×228。

**📈 对比分析**

与现有前向推理 3DGS 方法（DrivingForward、MVSplat）以及 NeRF/occupancy 方法（SelfOcc、UniPad、EmerNeRF、DistillNeRF）进行对比；OccSplat 在 PSNR、SSIM、LPIPS 上均超过基线，在遮挡区域表现尤为显著，虽然推理速度略低于 DrivingForward（1.01 FPS 对比 1.74 FPS）。

**⚠️ 局限性**

局限性包括：PixelRefiner 仅在图像空间工作，无法完全保证三维几何一致性；在大幅位移或极端遮挡时可能出现纹理过度平滑；缺少时序信息与语义先验，导致动态场景或长距离未观测区域补全仍不完美。

---

## 49. Learning to Tune with Biased Gradients: Iterative Feedback Tuning of Model-Free Control on Throttle Valves

**arXiv ID:** 2611.06338 | [PDF](https://arxiv.org/pdf/2611.06338v1)

**作者:** Félix Martinez-Jiménez `[一作]` (University of Grenoble Alpes), Emmanuel Witrant `[通讯]` (University of Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在含摩擦与粘滑的电子节气门上，使用迭代反馈调优（IFT）方法对智能比例（iP）控制器进行参数调优，并在数字双胞胎与实机上评估其性能。

**💡 创新点**

创新点在于揭示了摩擦导致的实验初始状态不匹配如何使IFT梯度产生偏差，量化了此偏差对梯度方向与下降率的影响，并比较了不同自适应更新规则（Gauss‑Newton、Adam、sign、Adam‑GN）在受偏置梯度下的表现，发现更新规则的优劣取决于噪声结构与梯度可靠性。

**🔧 技术方法**

使用的技术包括：迭代反馈调优、增量PI控制器、基于实验的梯度与海森近似估计、Adam自适应优化、数字双胞胎建模（Karnopp粘滑模型、弹簧预加载、非线性阻尼），以及对实验参数（如第二实验幅值c）的系统性探索。

**📊 数据集**

数据集为10台电子节气门的数字双胞胎模型（从先前的PRBS与PWM阶梯实验识别获得），以及两台实机（Valve A与Valve B）的实验数据，包含10次重复实验、噪声注入与摩擦系数随机扰动。

**📈 对比分析**

通过在数字双胞胎上与实机上分别跑40次迭代，对比更新规则的归一化下降率、最终误差、失败率以及验证测试的RMSE。结果显示：在双胞胎上，Adam在低噪声情况下可获得较小的尾部分布；在Valve A上，Adam结合归一化与动量在噪声低的K维度上实现更快收敛；在Valve B上，Adam因归一化导致的漂移导致性能退化，Gauss‑Newton表现更稳定。总体而言，更新规则对不同硬件的适用性不同。

**⚠️ 局限性**

局限性包括：仅验证了两台硬件；实验偏置模型仅考虑常数偏移，未涵盖动态摩擦变化；未比较梯度自由的贝叶斯优化等基线方法；验证指标主要是RMSE，未深入分析超调与稳定性；数字双胞胎对摩擦行为的预测不够精准，无法准确预估哪种更新规则更优。

---

## 50. TACT: A Pragmatic Approach to Qualitative Analysis with an AI Council

**arXiv ID:** 2611.05702 | [PDF](https://arxiv.org/pdf/2611.05702v1)

**作者:** Rudrajit Choudhuri `[一作]` (Oregon State University), Anita Sarma `[通讯]` (Oregon State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 TACT（Traceable AI Council for Thematic Analysis）——一种利用三大模型家族和对抗者的多代理协作流程，既能在大规模文本上完成主题分析，又保持完整的可追溯性与研究者最终解释权。

**💡 创新点**

创新点在于：①多模型议会与对抗者相结合，降低单模型偏差与共犯错误；②决策模型对编码不确定性进行量化与筛选；③完整的从发现、调解到编码的可追溯链条；④在实验中验证了其在真实工业与公开数据上的可靠性与经济性。

**🔧 技术方法**

使用技术包括：Anthropic Claude、Google Gemini 与 OpenAI GPT 的多模型分析；Python deterministic 脚本进行证据可追溯性检查；对抗者模型进行输出批判；TypeSafe 的 Jev 进行代码适用度评分；多数投票聚合与 Krippendorff/ Cohen 的可靠性指标；以及成本计量脚本。

**📊 数据集**

主要数据集为 2023 年 Zhang 等人公开的 Stack Overflow Copilot 讨论串（共 678 条帖子），以及作者自己进行的三项内部研究（微软开发者问卷、工作礼仪访谈、AI 自动化等级划分）。

**📈 对比分析**

比较方法：将 TACT 产生的候选码和最终编码与 Zhang 等人手工编码的五个研究问题标签进行召回、精准率与 F1 评分，并与单模型基线（仅用 GPT）对比；同时计算三 AI 编码者之间的 Krippendorff α（>0.7）与与研究者持有样本的 Cohen κ（0.79–0.85）。实验结果显示，Lite 版在编码阶段 F1 ≈0.75，Full 版 ≈0.71，单模型基线显著逊色；成本约 0.85–1.78 美元/100 条讨论串。

**⚠️ 局限性**

局限性包括：仅在英文语料上验证，可能对非英语语料适用性不足；模型可能已在公开数据上预训练，导致潜在信息泄露；仍需研究者手工 blind‑pass 与审阅，工作量不可忽视；对抗者与决策模型虽然提升可靠性，但仍可能引入新偏差；最终解释仍取决于研究者的主观判断。

---

## 51. Hybrid Spark: Adaptive Speculative Decoding That Lets the Content Choose the Drafter

**arXiv ID:** 2611.05611 | [PDF](https://arxiv.org/pdf/2611.05611v1)

**作者:** Ekagra Ranjan `[一作]` (Cohere), Bharat Venkitesh `[通讯]` (Cohere)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种在每一步同时运行模型基础和模型无关（n-gram）两种草稿器的混合式自适应投机解码框架，能够在一次验证预算下动态选择对每个请求最有利的草稿链。

**💡 创新点**

创新点包括：1）对模型无关草稿器的接受概率进行一次性校准，生成与模型基础草稿器相同的置信度尺度；2）将每个验证槽的“价格”定义为该槽在批处理中可产生的额外良率，从而在同一预算下比较不同草稿器的价值；3）所有决策（链选择、预算分配）在 GPU 上完成，保持异步调度与 CUDA 图的零同步；4）提出“包含规则”在 n-gram 草稿与模型基础草稿重叠时合并存活率，提升吞吐。

**🔧 技术方法**

主要技术包括：投机解码（Speculative Decoding）、自适应验证（Adaptive Verification）、logistic 回归校准模型无关草稿的接受概率、基于“槽价格”的预算分配策略、GPU 内核实现的链选择与分配。

**📊 数据集**

实验使用了多种 LLM（Qwen3‑4B、Qwen3‑8B、Gemma‑4‑12B、North‑Mini‑Code‑1.0、Llama‑3.1‑8B‑Instruct）以及多种工作负载（NextCoder、EditBench、Blazedit、CanItEdit、InstructCoder、MT‑Bench），数据集主要来自 SPEED‑Bench、NextCoder 等开源代码编辑与对话数据。

**📈 对比分析**

与传统单一草稿器、纯 n‑gram、TurboSpec、D‑cut 等基线比较，Hybrid Spark 在代码编辑任务中（BS = 4）实现 1.10–1.62 倍吞吐，BS = 128 时 1.04–1.27 倍；在聊天任务保持 0.97–1.01 倍性能，且在 T = 1 时保持无损；相较于仅使用 n‑gram 的 2.5‑7.4 倍加速，Hybrid Spark 通过自适应预算显著提高了整体效率。

**⚠️ 局限性**

局限性包括：1）需要对 n‑gram 草稿器一次性拟合校准模型，若文本分布发生剧烈变化可能失效；2）在极大批量或高延迟场景下，基于槽价格的预算估计仍会出现误差，导致少量吞吐下降；3）实验仅覆盖了几种模型与工作负载，尚未验证对更大模型或多语言场景的适用性；4）当前实现仅支持 greedy 草稿模式，在 probabilistic 草稿下需额外校准或采用 lossless 策略。

---

## 52. XuanJian-DINOSAR: A SAR Foundation Model Based on Global and Local Semantic Consistency

**arXiv ID:** 2611.06105 | [PDF](https://arxiv.org/pdf/2611.06105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 53. Exhaustive Search Is Hard to Beat in Constant Depth for Hyperclique and Orthogonal Vectors

**arXiv ID:** 2611.06231 | [PDF](https://arxiv.org/pdf/2611.06231v1)

**作者:** Nikolai Chukhin `[一作]` `[通讯]` (JetBrains Research), Nikolai Chukhin (JetBrains Research)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在常数层 DeMorgan 逻辑电路中，针对超团、团和 k‑向量正交（k-OV）问题的下界，证明了对多项式规模的超团、团以及 k‑OV 在任何固定深度下都需要接近最优的门/线数。

**💡 创新点**

创新点在于将 Rossman‑Li‑Razborov 的随机种子法与 Tal 的 Fourier 尾部界结合，利用对随机超图密度趋近 1 的分析，首次给出了多项式规模超团与团的下界，并进一步推导出 k-OV 在多对数维度下的强下界；此外，还给出了深度三电路下 2‑OV 的 Ω(n²) 线数下界。

**🔧 技术方法**

主要技术包括：随机植入超团、对电路门进行前缀拆分、p‑偏 Fourier 分解与尾部集中、Håstad 的切换引理扩展、Nechiporuk 的公式下界、以及在超图边概率趋近 1 时的覆盖计数分析。

**📊 数据集**

该工作为纯理论分析，无实验数据集；所有结论均基于随机图/超图模型与抽象电路模型。

**📈 对比分析**

与现有的上界（如 exhaustive search 的 nk+1 门）相比，本文的下界在多项式规模超团、团以及 k‑OV 上几乎达到最优（接近 m^ℓ 或 n^k 的规模），表明 exhaustive search 在常数层电路中几乎是最优的；对于深度三电路的 2‑OV，Ω(n²) 线数的下界与已知的上界相吻合，进一步验证了猜想。

**⚠️ 局限性**

局限性包括：下界仅适用于固定深度的 DeMorgan 电路；在超团/团尺寸增长至 n^(1/(h‑1)) 以上时结果需进一步改进；k‑OV 下界仅在多对数维度（d=O(log^c n)）下给出，对更高维度仍未达到最优；深度三 2‑OV 结果在 d≥(1+η)log n 时成立，对更小维度不适用。

---

## 54. Combining Relevant Expert Knowledge Using Mixtures of Feedback Fuzzy Cognitive Maps

**arXiv ID:** 2611.05664 | [PDF](https://arxiv.org/pdf/2611.05664v1)

**作者:** Olaoluwa Adigun `[一作]` (Florida International University), Bart Kosko `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种基于合法证据相关性定义的神经符号混合方法，用于聚合专家的反馈模糊认知图（FCM）以回答特定查询。

**💡 创新点**

创新点在于将法律证据中的“相关性＝有证据性+实质性”与FCM混合结合，提出贝叶斯式后验FCM与等差误差度量相结合的相关性加权方案，能够为每个查询生成动态的、查询特定的FCM混合网络。

**🔧 技术方法**

使用的技术包括：FCM离散时间动力学、零填充混合、伪逆求逆生成贝叶斯式后验FCM、基于均方根误差的相关性权重计算、以及矩阵与向量运算的并行实现。

**📊 数据集**

数据集为基于实际13节点缺血性卒中风险模型的FCM，并通过乘法和加法扰动生成7个合成专家FCM作为实验数据。

**📈 对比分析**

比较方法是将各专家FCM的平衡点与混合后FCM的平衡点在不同初始状态下的欧氏距离作为相关性度量，实验显示相关性权重能显著提升对查询的预测准确性，尤其在不均匀权重与不同初始状态下表现出更好的稳定性。

**⚠️ 局限性**

局限性包括：假设所有相关专家的材料性为1（未考虑多领域专家）、对FCM可逆性的依赖、以及在大规模知识网络中仍需进一步优化稀疏矩阵计算与硬件实现。

---

## 55. DeepfakeGenome: Toward Next-Generation Deepfake Attribution

**arXiv ID:** 2611.06046 | [PDF](https://arxiv.org/pdf/2611.06046v1)

**作者:** Shunli Wang `[一作]` (Tencent Youtu Lab), Lizhuang Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了包含 100 种面部伪造算法、200 万张图像的 DeepfakeGenome（DFG）基准，并设计了 4 种实用评估协议，系统评估深度伪造检测与归因模型。

**💡 创新点**

创新点在于：①大规模多样化的伪造算法集合，使基准 4‑100 倍大于此前工作；②提出检索式归因评估，贴合黑名单实时拦截需求；③通过 2000+ 实验揭示了 CLIP、DINO、对比学习等方法在 OOD 归因上的极限与缺陷。

**🔧 技术方法**

采用的技术包括 CLIP‑ViT、DINO‑ViT、Effort、Patch‑Shuffle、对比学习、数据增强、低/高层特征融合、频域特征（DCT、NPR）等多种深度伪造检测与归因方法。

**📊 数据集**

使用的数据集为 DFG 本身（100 个算法、4 种伪造类型、GAN、Diffusion、CNN、CG、VAR、闭源等），并对 FF++、DFDC、ForgeryNet 等现有数据集进行整合和去重。

**📈 对比分析**

与现有基准及方法的对比显示：Fine‑tuned CLIP‑ViT 在原始域性能最高，但在跨域/跨算法检索中落后；DINO 在同域归因强大但 OOD 极易过拟合；对比学习会降低 OOD 泛化；低层频域特征与高层语义特征互补可显著提升检索 mAP。

**⚠️ 局限性**

主要局限包括：伪造样本与真实语义对齐不完全，可能影响归因精度；基准未覆盖最新生成模型；未提出新的归因框架，而是聚焦于系统评估与现有技术的组合。

---

## 56. HippocamPEX: A Hippocampus-Inspired In-Silico Architecture for Context-Aware Sequence Prediction

**arXiv ID:** 2611.05415 | [PDF](https://arxiv.org/pdf/2611.05415v1)

**作者:** Md Abdullah al Kaiser `[一作]` (University of Wisconsin-Madison), Akhilesh Jaiswal `[通讯]` (University of Wisconsin-Madison)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了混合信号硬件架构 HippocamPEX，用于模拟海马体的事件-情境编码、序列预测，并验证其能完成上下文相关的序列预测任务。

**💡 创新点**

将海马层级结构、功能分离与并行处理原则直接映射到硬件，并结合 AP‑CAM、交叉电阻阵列和 MCAM 等新型存储计算单元，实现低功耗、并行的上下文敏感序列预测。

**🔧 技术方法**

混合信号电路、近似内容可寻址存储（AP‑CAM）、可编程交叉电阻阵列（crossbar）、多端口 CAM（MCAM）、时序计数器、模拟匹配、128‑bit 高维超向量编码。

**📊 数据集**

使用合成的数字与颜色上下文序列数据（如 0‑9 数字+背景色，按颜色定义不同序列规则）以及基于神经元模拟的海马子区活动数据进行软件模型验证。

**📈 对比分析**

通过硬件仿真与软件神经模型的解码结果对比，展示相似的子区表征动态；在上下文特定序列预测任务中，硬件实现能够准确预测后续事件，性能与软件模型相近，功耗低、速度快（具体数值未给出）。

**⚠️ 局限性**

缺少齿状回模式分离、theta 震荡、离线回放等关键生物学细节；训练仍离线；仅在合成序列上验证，未在真实环境或大规模数据上测试；对噪声鲁棒性与可扩展性的进一步评估仍待开展。

---

## 57. 3D Pose Estimation in Projection Space via Pixel-level Visible Light Communication

**arXiv ID:** 2611.06121 | [PDF](https://arxiv.org/pdf/2611.06121v1)

**作者:** Michikuni Eguchi `[一作]` (University of Tsukuba), Takefumi Hiraki `[通讯]` (University of Tsukuba)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了利用投影仪嵌入像素级可见光通信（PVLC）技术，结合Perspective‑n‑Point（PnP）求解器，实现投影空间中多点光敏元件的三维姿态估计。

**💡 创新点**

创新点在于将PVLC从传统二维定位扩展到三维，通过将设备上多点光敏元件的已知三维布局与投影像素坐标构成PnP对应关系，实现全投影区域无额外跟踪区域、厘米级精度和8 ms延迟的3D姿态估计。

**🔧 技术方法**

使用的技术包括DLP投影仪的高速亮度调制、像素级位置编码、光敏电路（光敏二极管+放大+比较+微控制器）、PnP求解器（Homography Decomposition、IPPE、P3P、EPnP）以及卡尔曼滤波进行姿态平滑。

**📊 数据集**

实验数据来自实机测试，使用xArm5机械臂在四种轨迹上按50–400 mm/s速度移动，光敏元件布局为80 mm和160 mm正方形，未使用公开数据集。

**📈 对比分析**

与基于摄像机的AR Marker（ArUco）方法比较：在所有速度下PVLC平均3D位置RMSE为19.2 mm，延迟约8 ms；低速时AR Marker略优，但随着速度上升PVLC误差增长缓慢，最终在高速和大布局下优于AR Marker。

**⚠️ 局限性**

主要限制包括：受投影体积限制；深度误差随光敏元件间距减小而增大；高速运动时像素跨越导致解码误差；光照干扰与遮挡影响；非平面设备需更多传感器或融合IMU；对极限速度和小尺寸设备的适用性仍需改进。

---

## 58. ReactWorld: Structuring Interaction Responses for Embodied Video Generation

**arXiv ID:** 2611.06120 | [PDF](https://arxiv.org/pdf/2611.06120v1)

**作者:** Hanwei Liu `[一作]` (Huawei Technologies Co., Ltd.), Dong Zhang `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ReactWorld模型，通过在扩散生成背后的特征空间中预测交互角色、未来响应强度与间隔运动，进一步将预测的动力学反馈到视频生成中；

**💡 创新点**

创新点在于构造交互响应场并利用零动力学参考差分更新物体特征，以及将预测位移转化为时间注意力偏置，实现结构化交互预判与生成的闭环反馈；

**🔧 技术方法**

采用扩散视频生成框架（DiT）+LoRA微调，结合交互响应场预测头、反馈适配器、位移引导的时间注意力机制，受Chain‑of‑Thought启发的结构化推理；

**📊 数据集**

使用约50万条机器人视频数据（来自AgiBot、RoVid‑X以及网络抓取），并通过Qwen3‑VL、SAM3、CoWTracker等工具自动生成角色与运动标签；

**📈 对比分析**

在RBench（5类任务、4机器人形态）和EZS‑Bench上与Wan2.2、Cosmos3、ABot‑PhysWorld、PhysisForcing以及Video‑FT基线对比，平均RBench分数达77.51（比PhysisForcing高3.29分），EZS‑Bench Domain和平均分分别为86.98与81.87，显著提升任务完成率、接触合理性和整体表现；

**⚠️ 局限性**

局限性在于仅使用二维交互响应表示，缺乏显式三维交互建模和时序一致的三维运动描述，未来需要探索三维空间与时序一致性的改进。

---

## 59. MA-SafeDiffuser: Safe Multi-Agent Planning with Diffusion Probabilistic Models

**arXiv ID:** 2611.05524 | [PDF](https://arxiv.org/pdf/2611.05524v1)

**作者:** Kiran Ravish `[一作]` (IIIT Hyderabad), Pawan Kumar `[通讯]` (IIIT Hyderabad)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多智能体安全扩散规划器 MA‑SafeDiffuser，能够在反向扩散步骤中嵌入控制屏障函数（CBF）约束，实现路径生成与安全性的融合。

**💡 创新点**

创新点包括：① 在扩散模型的每一步投影至线性化 CBF 半空间，实现全局安全保证；② 提供中心化和去中心化两种投影变体，后者通过邻居通信与鲁棒裕度实现分布式安全；③ 通过时间可变规范、对称破坏和进展约束等机制避免死锁；④ 在离散时间扩散更新下给出有限时不变性理论证明。

**🔧 技术方法**

技术手段主要有：扩散概率模型（DDPM）、控制屏障函数、凸二次规划/ Dykstra 投影、离散时间 CBF 对比 lemma、时间可变规范（TVS）与鲁棒边际、执行时安全过滤。

**📊 数据集**

使用自定义 Maze2D 基准数据集，包含 2、4、8 名机器人，环境为 6×6 网格迷宫，设有墙壁、门和目标区域。

**📈 对比分析**

通过与无安全约束的扩散器、MAPPO‑Lag 等基线对比，采用违规计数、累计回报和成功率作为指标。MA‑SafeDiffuser 在所有设置下将违规数降低约 40–60%，保持相近的回报，并实现 100% 的成功率（而 MAPPO‑Lag 在同样的迷宫中 0%），证明了其安全性与性能兼得。

**⚠️ 局限性**

局限性包括：① 在高密度群体中 QP 投影的可行性问题；② 去中心化模式需要保证邻居预测误差受限；③ 对于极长规划时程，奖励形状对回报的影响敏感。

---

## 60. Flow-Optimal Separators for Minor-Free and Induced-Minor-Free Graphs in Linear Time via Follow the Regularized Leader

**arXiv ID:** 2611.05756 | [PDF](https://arxiv.org/pdf/2611.05756v1)

**作者:** Hung Le `[一作]` (University of Massachusetts), Jack Spalding-Jamieson `[通讯]` (Independent)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新算法，用于寻找无小图和诱导小图的平衡分隔符。

**💡 创新点**

创新点在于将算法与在线学习相结合，设计了一个连续的（非凸）优化问题，并使用Follow the Regularized Leader（FTRL）框架进行求解。

**🔧 技术方法**

使用了Follow the Regularized Leader（FTRL）框架来解决优化问题。

**📊 数据集**

算法适用于H诱导小图和K_h小图的图，具体数据集未明确给出，但涉及到的图类包括字符串图和稀疏图。

**📈 对比分析**

与之前的算法相比，提出的算法在期望线性时间内找到平衡分隔符，且分隔符的大小为O(log(|V(H)|)√(|V(H)| + |E(H)|)√(m))，在K_h小图中为O(hlog h √(n))，性能优越。

**⚠️ 局限性**

算法的局限性在于对图的结构理解不足，可能无法在所有情况下提供最佳的运行时间。

---

## 61. Spatio-Temporal Multi-Order Interaction for Video Recognition

**arXiv ID:** 2611.05633 | [PDF](https://arxiv.org/pdf/2611.05633v1)

**作者:** Kevin D. Nguetche `[一作]` (ETS Montreal), Marco Pedersoli `[通讯]` (ETS Montreal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了VideoMogaNet，一种将多阶时空交互与轻量化时间建模相结合的视频识别架构。

**💡 创新点**

创新点在于：①多阶时间调制模块（MOTM）采用并行深度卷积和门控聚合，实现线性时间复杂度的多阶时间交互；②双层时空交互（ST-MOI）使用可变增益乘法实现空间与时间的双重交互，提升表达能力。

**🔧 技术方法**

技术方法包括：并行深度一维卷积、门控聚合、可变增益乘法、变通道增长比例的四阶段结构以及在MogaNet基础上的通道-时间-空间顺序设计。

**📊 数据集**

使用了UCF-101、HMDB-51、Kinetics-400和Diving-48四个视频分类基准数据集。

**📈 对比分析**

与现有方法对比，VideoMogaNet在UCF-101、HMDB-51、Diving-48上取得最高或接近最高准确率，同时在Kinetics-400保持竞争力；相比同类模型，它的时间分支计算量显著降低，推理吞吐量随分辨率增大下降更慢。

**⚠️ 局限性**

局限性包括：整体计算复杂度仍受空间d^2项影响，长序列的时空建模仍受限于轻量化模块；在某些仅依赖空间特征的类别中提升有限。

---

## 62. Lesion Overlap Ranks PET Segmentation Models but Does Not Certify Patient-Level Tumor Burden

**arXiv ID:** 2611.05564 | [PDF](https://arxiv.org/pdf/2611.05564v1)

**作者:** Tushar Kataria `[一作]`, Shireen Y. Elhabian `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究评估了全身 PET 分割模型的 Dice 重叠指标与患者级代谢肿瘤体积（MTV）误差的关系，并比较了多种公开分割模型及 PSMA PET 的表现，发现 Dice 与个体 MTV 误差不一致，且多模型共通忽略了小、低摄取的病灶。

**💡 创新点**

创新点在于：①首次在全身 PET 级别系统性地将 Dice 与 MTV 误差进行对照，揭示了 FP/FN 互相抵消导致 MTV 误差低的现象；②发现大多数模型对小、低 SUV 病灶存在共同盲区；③提出在分割评估中应同时报告患者级 MTV 误差和缺失病灶计数。

**🔧 技术方法**

使用了 autoPET 框架下的多种深度分割模型（LesionTracer、IKIM、Hussain、CW nnU-Net、BAMF），基于 Dice、相对 MTV 误差、绝对误差、FP/FN 体积差异及其相互抵消比例进行统计分析，采用 Spearman 相关、部分相关、Bootstrap 置信区间、Wilcoxon 检验和精确排列检验。

**📊 数据集**

主要数据集为 1,014 份 FDG PET/CT（FDG‑UKT）涵盖 900 名患者（501 份肿瘤阳性）和 597 份 PSMA PET/CT（378 名肿瘤阳性），全部为公开匿名数据。

**📈 对比分析**

通过比较五个公开模型在同一 FDG 样本上的 Dice 与相对 MTV 误差，发现 Dice 排名与 MTV 误差排名完全相反（ρ = –1.00，p = 0.017）；但在患者级别，Dice 与 MTV 误差仅呈中等负相关（ρ ≈ –0.75），约 16.6% 患者 Dice ≥ 0.95；PSMA 数据亦显示 Dice 与 MTV 误差的负相关（ρ ≈ –0.81）但仍存在 20% 的阈值冲突。

**⚠️ 局限性**

局限性包括：① LesionTracer 的 “out‑of‑fold” 仅避免了 fine‑tuning 见过的患者，仍包含预训练时已见过的样本；② 其余四个模型为训练已暴露模型，未能评估在全新患者上的性能；③ 仅重建了 LesionTracer 的验证折，其他模型无法得到完全独立的预测；④ 所有数据来自单中心，未能排除中心或示踪剂差异；⑤ 小病灶缺失可能受注释误差影响，且未能进一步探讨其对预后或治疗评估的临床意义。

---

## 63. CARE-FL: Federated Fine-Tuning of Multimodal LLMs in Resource-Constrained Edge Networks

**arXiv ID:** 2611.06169 | [PDF](https://arxiv.org/pdf/2611.06169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 64. VirtualTrialHarness: Coordinating Agentic Operationalization of Virtual Trial Protocols Across Complex Clinical Schemas

**arXiv ID:** 2611.05551 | [PDF](https://arxiv.org/pdf/2611.05551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 65. RAGCompass: Generalizable Retriever Routing via Reranking

**arXiv ID:** 2611.05596 | [PDF](https://arxiv.org/pdf/2611.05596v1)

**作者:** Valentin Six `[一作]` (University of Virginia), Pan Li `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于检索结果内容的路由器，通过微调重排序模型并加入对比头，以答案质量为监督信号来选择最优检索器。

**💡 创新点**

不依赖查询特征或检索器自带分数，而是直接读取检索到的证据并对答案效果进行监督，且路由器无检索器特定参数，能够对未知检索器或语料库保持鲁棒性。

**🔧 技术方法**

使用 Qwen3-Reranker-0.6B 作为共享骨干网络，先微调得到分数再训练两层 Transformer 对比头，采用 KL 散度将答案 F1 映射为概率分布进行监督。

**📊 数据集**

在四个单跳/多跳问答数据集（Natural Questions、MuSiQue、HotpotQA、2WikiMultihopQA）上进行训练与评估，并使用五种不同范式检索器（dense、BM25、SPLADE-v3、ColBERTv2、HippoRAG2）。

**📈 对比分析**

与固定检索器、RRF、MoR、LTRR、R3AG、R.RETRIEVER、Majority Vote 和 LLM-as-router 等基线对比，取得 0.3552 F1，较最佳固定检索器提升 7.7pp、较 MoR 提升 3.5pp；在未见语料、检索器和读者等扰动下仍保持大部分优势。

**⚠️ 局限性**

实验仅使用单一读者（Llama-3.1-8B-Instruct）和固定检索深度；未验证不同模型规模、检索深度或多种读者的适应性；路由器训练仍需为每个检索器重新标注答案；对所有检索器的并行运行成本未计入。

---

## 66. A Hybrid Approach for Automated Security Pattern Recovery

**arXiv ID:** 2611.05823 | [PDF](https://arxiv.org/pdf/2611.05823v1)

**作者:** Anh Minh Nguyen `[一作]` (University College Dublin), Liliana Pasquale `[通讯]` (University College Dublin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SecPatRec，一种结合 LLM 角色映射和代码属性图 (CPG) 交互恢复的混合方法，用于自动从代码库中恢复安全模式。

**💡 创新点**

首次将 LLM 的推理与代码搜索和静态 CPG 分析相结合，实现了安全模式的结构与行为双视图恢复，并提供了人机协同的 ground-truth 评估框架。

**🔧 技术方法**

使用 ReAct 迭代 LLM、BGE 向量搜索、语法/词法代码搜索、Joern 构建 CPG 与遍历，评估指标 RDA、MLS、IRC。

**📊 数据集**

构建 93 个开源 Java/Python 项目（49/44），覆盖 5 个安全模式，总共 100 个实例的人工验证 ground-truth 数据集。

**📈 对比分析**

与单次生成的端到端 LLM 基线对比，SecPatRec 在 RDA 0.84、MLS 0.84、IRC 0.66 上均优于基线，并在大仓库下成本低且可扩展。

**⚠️ 局限性**

依赖 LLM 的推断精度，CPG 只能捕获静态调用，无法识别运行时框架注解导致的交互，且评估仅限于 5 个安全模式和两种语言。

---

## 67. File-Semantic SSD for I/O-centric Architectures with NFS-over-NVMe Protocol

**arXiv ID:** 2611.06071 | [PDF](https://arxiv.org/pdf/2611.06071v1)

**作者:** Kyu-Jin Cho `[一作]` (Seoul National University), Jin-Soo Kim `[通讯]` (Seoul National University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在SSD内部实现文件抽象的存储设备（File‑Semantic SSD），使GPU、DPU等外设能够直接在NVMe通道上进行文件级读写，而不需要宿主机文件系统介入。

**💡 创新点**

创新点包括：① 将NFSv3的轻量级文件接口嵌入NVMe命令（NFS‑over‑NVMe）实现文件操作直接映射到NVMe；② 将文件层与闪存翻译层（FTL）共设计，形成统一的文件映射表（File Map）和垃圾回收机制，显著降低WAF并节省DRAM；③ 支持NVMe‑oF Target Offload，实现远程文件访问时完全绕过目标服务器CPU；④ 采用能力（capability）安全模型，提升多租户访问控制。

**🔧 技术方法**

技术手段包括：NFSv3 RPC、NVMe命令扩展（使用特殊OPCODE和PRP）、NVMe‑oF、Target Offload、F2FS文件系统结构、Flash Translation Layer（FTL）协同、能力基安全、SPDK fsdev、GDS、BaM等。

**📊 数据集**

评估使用的真实数据集：图分析数据集（BFS/CC：3.1M/105.9M/30.2M节点的三种图）以及GPT‑3‑1.3B 训练模型的权重和优化器状态；此外还使用了合成读/写工作负载、随机/顺序IO、GPU‑CPU交互等。

**📈 对比分析**

与传统F2FS、NFS、NFS‑oR、GPUfs、GDS、BaM等基线对比；在本地环境中，-G（GPU侧客户端）在所有块大小上超过基线；在远程环境中，-K/-U在吞吐量上比NFS和NFS‑oR高至少1.4‑4.2倍；统一GC使WAF从4.22降至1.53（开启流隔离进一步降至1.30）。

**⚠️ 局限性**

局限性：目前仅支持单台SSD，缺乏多设备分布式实现；GPU侧客户端缺少远程访问支持；依赖于NVMe‑oF目标离线能力，需硬件支持；在高共享目录场景下，VFS层的inode semaphore仍是瓶颈。

---

## 68. NitroFusion2: Bringing Adversarial Loss to the Text Stream of Multimodal Diffusion Transformers

**arXiv ID:** 2611.05648 | [PDF](https://arxiv.org/pdf/2611.05648v1)

**作者:** Dar-Yen Chen `[一作]` (University of Surrey), Yi-Zhe Song `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出在多模态扩散变换器（MMDiT）中使用文本流中的中间文本令牌状态进行对抗性蒸馏，并引入了面向方面的对抗判别（ACD），从而实现单步或双步高质量生成。

**💡 创新点**

创新点在于：①将文本流的样本相关表示作为判别器输入（Text-Stream Adversarial Distillation, TSAD）；②通过在判别器中条件化不同方面（如排版、对象、风格等）来引导对抗信号（Aspect-Cued Discrimination, ACD）。

**🔧 技术方法**

采用的技术包括：多模态扩散变换器、联合注意力、轻量级判别器头、Hinge 损失、流匹配插值、以及对抗蒸馏框架。

**📊 数据集**

数据集方面：使用 Z-Image‑Turbo 生成 10 万张 1024×1024 的参考图像，并用 Qwen3.5‑35B‑A3B 自动生成六字段结构化说明；在 LongText‑Bench、OneIG‑Bench、GenEval、DPG‑Bench 等公开基准上进行评估。

**📈 对比分析**

实验将 NitroFusion2 与 Z-Image‑Turbo、TwinFlow 以及 Z-Image‑Turbo++ 在单步和双步采样下对比，结果显示 NitroFusion2 在所有指标上均超过或与多步模型相当，尤其在单步生成中获得与 8 步模型相近的质量，显著缩小了步数与质量之间的差距。

**⚠️ 局限性**

局限性包括：单步生成仍难以处理极其复杂场景，样本多样性下降；Aspect‑Cued 机制在强调某一方面时会削弱其它方面的表现；层级选择仍基于经验而非理论最优，未来需要进一步改进平衡和分析文本流信息的演化。

---

## 69. Catastrophic Overfitting is Weight Drift the Training Loss Cannot See

**arXiv ID:** 2611.06082 | [PDF](https://arxiv.org/pdf/2611.06082v1)

**作者:** Chao Pan `[一作]` `[通讯]` (Southern University of Science and Technology), Chao Pan (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在Fast Adversarial Training中防止灾难性过拟合的机制：在验证鲁棒性下降时将当前权重重置为其指数滑动平均（EMA），并在此基础上使用一种耦合的训练目标。

**💡 创新点**

创新点在于将灾难性过拟合解释为权重漂移，并证明训练损失对这种漂移几乎无视；基于此，提出“锚点”策略（将权重回到EMA）作为低成本、无侵入式的修复方法，并在基础耦合方法上实现。

**🔧 技术方法**

技术包括：单步FGSM训练、权重指数滑动平均、两种耦合正则化（核范数和均方误差），以及基于验证PGD-10鲁棒性差距的触发阈值来控制重置。

**📊 数据集**

使用的公开数据集为CIFAR-10、CIFAR-100和Tiny-ImageNet（均采用ε=8/255）。

**📈 对比分析**

与基准中所有快速对抗训练方法（NuAT、LIET、FGSM-PGI等）以及多步PGD-AT、PGD-AT-WA进行对比。实验表明，在CIFAR-10上平均AutoAttack鲁棒性达到50.45%，在CIFAR-100上达到27.15%，在Tiny-ImageNet上达到20.47%，均为三大基准数据集上最快方法中的最高水平，且在AutoAttack和CR攻击中均保持领先。

**⚠️ 局限性**

局限性包括：只在4、4、3个随机种子上验证，样本数有限；耦合正则化权重需针对每个数据集调节，缺乏通用设置；触发机制基于验证PGD-10，可能无法捕捉某些类型的CO；实验中对随机种子进行额外探测时可能会改变衰退模式。

---

## 70. Graph-Mediated Large Language Model Systems in the Agentic Era: Reasoning, Knowledge, Retrieval, Execution, and Coordination

**arXiv ID:** 2611.05953 | [PDF](https://arxiv.org/pdf/2611.05953v1)

**作者:** Haihui Liu `[一作]` (Beijing Normal University), Tian Wang `[通讯]` (Beijing Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对图介导的大语言模型系统进行综述，提出基于操作级别的角色‑位置分类法，梳理图在推理、证据、记忆、执行和协调中的作用，并给出评估框架。

**💡 创新点**

创新点在于将图的功能与计算责任分离的操作级分类、跨角色合成的影响分析以及针对图贡献的完整评估方法。

**🔧 技术方法**

使用技术包括图观察与选择、图程序执行、知识图检索、图条件生成、执行图和通信拓扑等多种图‑LLM交互方式。

**📊 数据集**

涉及的数据集与基准有MAGMA、GraphEval、GraphArena、GraphRAG‑Bench、HotpotQA、2WikiMultiHopQA、MuSiQue、STaRK、MMLU、HumanEval等。

**📈 对比分析**

比较方法采用全流程对比、条件在线对比和组件干预，并结合过程诊断与系统成本报告，结果显示局部性能提升不一定转化为整体任务收益。

**⚠️ 局限性**

局限性包括调查范围仅覆盖图活跃操作，缺乏统一实验控制与诊断标准，评估多依赖现有基准，未能系统解决自适应计算、记忆维护与依赖恢复等核心挑战。

---

## 71. Sure Enough to Share? Overconfidence in GenAI Video Discernment is Associated with Sharing Intentions

**arXiv ID:** 2611.05844 | [PDF](https://arxiv.org/pdf/2611.05844v1)

**作者:** Haoning Xue `[一作]` (University of Utah), Yixuan Wang `[通讯]` (University of Utah)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

系统研究了人们在鉴别生成式人工智能视频时的三种过度自信形式（过度估计、过度置换和过度精确），并探讨其与源判断、分享意图及个体差异的关联

**💡 创新点**

首次将三种过度自信形式与生成式AI视频辨识与分享行为进行分离式关联分析，并指出过度精确是与分享意图最相关的形式

**🔧 技术方法**

采用实验心理学方法、信号检测理论（d、c）以及自我评估与置信度测量，结合线性回归与混合效应模型

**📊 数据集**

使用了973名美国成年受试者的数据，评估了42段（21对AI生成与真人）短视频（无音频），每位受试者观看18段视频

**📈 对比分析**

通过对比受试者的实际辨识准确率与自我估计，得出平均准确率62.5%并揭示过度精确与分享意图正相关；与单一过度自信形式模型相比，多重形式模型更能解释行为变异

**⚠️ 局限性**

受限于测量方式（过度精确按每段视频置信度测量）、实验环境缺乏音频及真实性、多为相关性研究而非因果验证

---

## 72. DARE: Difficulty Analysis for Repair Enhancement

**arXiv ID:** 2611.05797 | [PDF](https://arxiv.org/pdf/2611.05797v1)

**作者:** Jikwang Lee `[一作]` (Handong Global University), Jaechang Nam `[通讯]` (Handong Global University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于静态任务特征预测代码修复难度并通过特征组解释的框架，并利用解释结果为不同难度任务匹配专属支持模块

**💡 创新点**

①用连续难度（多Agent失败率）代替二元标签；②将151个静态特征分为七个Repair-Task Feature Groups，聚合模型贡献；③在模型预测后直接驱动任务级支持，提升Agent成功率

**🔧 技术方法**

采用通用加性模型（GAM）进行难度预测并获取特征贡献；使用特征组贡献聚合与归一化；对不同Benchmark使用Shared‑Agent Scale Alignment；实现七个支持模块（定位、变换、验证等）

**📊 数据集**

SWE‑bench（Verified、Lite、Full）与SWE‑bench‑Live Lite共四个划分，共2594个任务

**📈 对比分析**

与未使用支持的基线相比，匹配特征组支持使Agent在九个高难度任务上的成功率从54.7%提升至79.6%（p=0.0039）。在四个留出Benchmark上，预测Spearman相关性为0.476–0.621，pairwise准确率和MAE均在可接受范围内

**⚠️ 局限性**

仅针对Python任务，参考补丁信息间接使用；支持模块数量有限，可能受模型记忆影响；难度定义依赖Agent群体，跨Agent推广需进一步验证

---

## 73. RI-Prune: Relation and Information Aware Visual Token Pruning for Multi-Image Vision-Language Models

**arXiv ID:** 2611.06047 | [PDF](https://arxiv.org/pdf/2611.06047v1)

**作者:** Dongjin Lee `[一作]` (Chung-Ang University), Youngbin Kim `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了针对多图视觉语言模型的视觉令牌剪枝方法。

**💡 创新点**

提出了无注意力、训练自由的关系与信息感知剪枝（RI‑Prune），融合查询相关性、跨图对应和图内代表性三种信号，并采用每图均匀Top‑K选择，从而在多图推理中显著提升效率和准确性。

**🔧 技术方法**

采用投影空间无监督得分、三信号z归一化与加权、Prototype聚合、每图Top‑K选择等技术。

**📊 数据集**

使用了多图评测基准（MUIRBench、MIBench、MIRB、Mantis‑Eval）以及视频基准（Video‑MME、TempCompass）进行验证。

**📈 对比分析**

与FastV、VisionZip、VTW、DivPrune、SCOPE等现有剪枝方法在多图任务上进行对比，RI‑Prune在保留4.6%令牌时仍保持约80%性能，整体多图任务平均准确率高于或相当于最强基线。

**⚠️ 局限性**

预算近似均匀分配可能对某些集中式证据不佳；仅在投影器–解码器接口剪枝；未利用解码器内部冗余；跨图对应在长序列成本高；仅在7–8B模型与多选任务上验证。

---

## 74. Retrieval Supersedes Scale: Efficient Log Level Prediction via Multiplex Socio-Technical Context

**arXiv ID:** 2611.05901 | [PDF](https://arxiv.org/pdf/2611.05901v1)

**作者:** Youssef Esseddiq Ouatiti `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种面向日志等级预测的检索式大语言模型（LLM）方法，通过多层社交技术上下文（功能相似性与代码所有者信息）来挑选示例，以增强LLM的上下文学习；

**💡 创新点**

创新点在于将功能相似性与开发者所有权两种社会技术信号融合到多层网络（multiplex graph）中，实现对检索示例的精确聚类，从而显著提升LLM在日志等级预测上的准确率；

**🔧 技术方法**

使用的技术包括：代码嵌入（CodeXEmbed）、UMAP降维、HDBSCAN密度聚类、Leiden社区检测、FAISS检索、In-Context Learning（ICL）与大语言模型（CodeLlama-7B/34B、GPT‑4o、DeepSeek-V3.2）；

**📊 数据集**

数据集为十个主流开源Java项目（如Hadoop、Elasticsearch、Kafka等）中的日志语句与上下文，按70/30训练/测试划分；

**📈 对比分析**

通过与随机检索、平面相似性检索、文档组件检索以及两种监督模型（Ordinal Logistic Regression、DeepLV）对比，发现多层检索提升AUC平均从0.81提升至0.89（相较随机提升0.07），甚至超过大型商用模型（如GPT‑4o）并保持更低成本；

**⚠️ 局限性**

局限性包括：只考虑文件级功能与所有者信号，未涵盖其他可能影响日志等级的技术因素；聚类对项目历史深度敏感，低维护活跃度项目的所有者信息可能不可靠；以及对大型或非Java项目的泛化性仍待验证。

---

## 75. CHARM+: Cross-Hardware Attention with Re-Merge Multistream Mechanism

**arXiv ID:** 2611.05906 | [PDF](https://arxiv.org/pdf/2611.05906v1)

**作者:** Xinhang Zhang `[一作]` (Lenovo Research), Limin Xiao `[通讯]` (Lenovo Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

改进了分布式注意力的通信流程，提出了CHARM+方案；

**💡 创新点**

采用基于合并的通信重组、q与kv的多流并行以及跨硬件查询并行化，显著减少通信依赖与同步开销；

**🔧 技术方法**

使用了GPU多节点架构（PCIe/NVLink）、Ring-Attention、TASP、Mesh‑Attention等对比基准，并实现了前向与反向两阶段；

**📊 数据集**

在多种GPU平台（L20、H20、H100）上进行实验，使用Megatron‑LM的上下文并行训练环境，使用通用长序列数据进行验证；

**📈 对比分析**

与Ring‑Attention、TASP等方法对比，CHARM+在前向/反向阶段平均提升1.3×–3.9×，端到端训练速度提升约1.7×–2.2×，并在不同硬件架构下保持稳健性能；

**⚠️ 局限性**

受限于PCIe链路竞争导致多流并行在PCIe上表现不佳，内存占用与Ring‑Attention相当，且未结合Megatron‑LM的其它加速优化。

---

## 76. Anduin: A sybil-resistant agentic marketplace reputation system

**arXiv ID:** 2611.05586 | [PDF](https://arxiv.org/pdf/2611.05586v1)

**作者:** Reza Ramji `[一作]` (Circle Research), Duc V. Le `[通讯]` (Circle Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并评估基于已结算链上支付历史的Sybil‑抗性代理交易市场声誉系统，利用交易图和流算法为代理生成个性化服务提供者评级。

**💡 创新点**

首次在无协议变更、无外部凭证、无人工审核的情况下，仅通过已结算支付构建可信声誉，提供三种基于最大流/求和流/放大流的评分方法，并证明其对Sybil攻击的经济成本上限。

**🔧 技术方法**

双边支付图建模、时间衰减加权、最大流与最小割定理、三种图流评分算法（max‑flow、sum‑flow、amplified‑flow），以及AUC、P@1、攻击成本等实验评估指标。

**📊 数据集**

x402市场157天的交易记录，包含66.1 M笔交易、总额6.1 M美元、跨16条链。

**📈 对比分析**

与传统计数基准（次数、共同用户计数、共同用户成交量）和协同过滤对比，流算法在P@1上提升至约23%，攻击成本需支出目标买家20%消费量，比协同过滤高约23倍；在所有813买家中平均P@1为22.8%，攻击成本平均为目标买家支出20%。

**⚠️ 局限性**

对新买家缺乏历史无法评级；多钱包共享地址会混淆图结构；仅基于支付流，缺乏直接质量反馈；对极小交易窗口或不同链费率敏感。

---

## 77. FedSharp: Byzantine Robust Decentralized Federated Learning Under Noisy Channel

**arXiv ID:** 2611.05546 | [PDF](https://arxiv.org/pdf/2611.05546v1)

**作者:** Ojaswi Sinha `[一作]` (Ohio University), Melike Erol-Kantarci `[通讯]` (University of Ottawa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedSharp 框架，解决分布式联邦学习中非 IID、Byzantine 破坏和通信噪声交互带来的性能下降问题。

**💡 创新点**

创新点在于：① 采用峰值比筛选和损失排名两步结合来剔除隐式攻击；② 只使用至少 b 个诚实模型的多模型平均，降低累积噪声；③ 设计跳数感知的鲁棒梯度范数正则和 proximal 训练，进一步抑制噪声和非 IID 漂移；④ 在环形拓扑中保证诚实模型必然可得。

**🔧 技术方法**

主要技术包括：坐标中值偏差筛选、损失最小化排序、均值聚合、噪声适应的梯度范数正则、proximal 正则、环形多跳模型传播、双向反向传播求 Hessian‑vector 乘积。

**📊 数据集**

实验数据集为 MNIST 与 CIFAR‑10，分别在 IID 与 Dirichlet β=0.5 的非 IID 划分下测试。

**📈 对比分析**

与 Basil、Basil+ACDS 两个基线在 100 节点环网、33 个 Byzantine 节点、噪声 σ=0.01 的设置下比较；FedSharp 在 MNIST 上可达 94.5–94.9% 的准确率，Basil 仅 29–35%；在 CIFAR‑10 非 IID 情况下达 33–35% 的准确率，远高于 Basil 与 Basil+ACDS，且在噪声与攻击条件下保持稳定，接近无噪声基线。

**⚠️ 局限性**

局限性：在强非 IID 环境下收敛速度慢，最终准确率仍低于无噪声基线；对更复杂攻击模式和更大规模网络的理论分析尚未完善。

---

## 78. Which One Is Sounding? Binaural Sounding Instance Disambiguation for Audio-Visual Segmentation

**arXiv ID:** 2611.06074 | [PDF](https://arxiv.org/pdf/2611.06074v1)

**作者:** Haoze Ma `[一作]` (Taiyuan University of Technology), Dimitrios Kanoulas `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出BiSID-AVS框架，在单源音频与RGB图像中利用双耳空间信息实现实例级音视分割。

**💡 创新点**

创新点在于将双耳空间响应与候选定位相结合，采用双层空间调制（特征级调制+解码器级提示），而非仅靠时序信息，成功区分同类候选实例。

**🔧 技术方法**

使用SAM ViT-H+VGGish作为特征提取器，结合双耳cochleagram、空间查询、特征级调制与解码器级提示，端到端训练。

**📊 数据集**

使用自研的Binaural AVS-Synthetic与Habitat-AVS两大数据集，涵盖语音合成与真实室内场景。

**📈 对比分析**

与TPAVI、AVSegFormer、COMBO、VCT、SAMA-AVS、AVISM等方法对比，BiSID-AVS在多候选子集的mIoU提升35–41个百分点，mSILA提升47–71个百分点，显著降低同类误激活。

**⚠️ 局限性**

局限在仅处理单一声源、依赖双耳空间信号且对极端噪声或多源情形表现尚需改进。

---

## 79. HarnessFlow: MCTS-Guided Harness Optimization for LLM Agents

**arXiv ID:** 2611.06254 | [PDF](https://arxiv.org/pdf/2611.06254v1)

**作者:** Zhi Hong `[一作]` (South China University Of Technology), Chong Luo `[通讯]` (Microsoft Research Asia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于MCTS的全代码级Agent harness搜索框架HarnessFlow，自动化优化LLM Agent harness。

**💡 创新点**

创新点在于将完整可执行 harness 视为代码树，结合MCTS父节点选择、可重用组件库和经验档案，实现在有限评估预算下的样本高效、成本低的搜索。

**🔧 技术方法**

使用MCTS、代码生成式Agent、可重用组件库、经验档案等技术。

**📊 数据集**

在六个行业benchmark（TAU2-bench、AppWorld、WebShop、BrowseComp-Plus、BigCodeBench-Hard、SWE-bench Verified）上评测。

**📈 对比分析**

与Vanilla、手工工程Agent（Claude Code、Codex、OpenHands、Hermes）以及Meta-Harness比较，平均测试分数69.48%，比Vanilla高10.89%，比Meta-Harness高3.95个百分点，且执行成本更低。

**⚠️ 局限性**

局限在于依赖可自动评分的任务、需要一次性评估成本、搜索树仅为树结构不能跨枝合并。

---

## 80. Compiling, Validating, and Using Human Reference Atlas Cell Type Annotation Crosswalks

**arXiv ID:** 2611.05888 | [PDF](https://arxiv.org/pdf/2611.05888v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 81. CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving

**arXiv ID:** 2611.06204 | [PDF](https://arxiv.org/pdf/2611.06204v1)

**作者:** Dingji Wang `[一作]` (Fudan University), Xin Peng `[通讯]` (Fudan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 ADS-agnostic 的协同运行时安全保障框架，能在混合交通环境中对多辆 AV 进行场景级安全干预；

**💡 创新点**

创新点在于将安全保障从单车自我接管转向场景级协同干预，并通过 V2X 预设合作实现对多 AV 的联合轨迹生成与协调；

**🔧 技术方法**

核心技术包括 V2X 预设合作通信、RSU 侧协同感知、场景级危险评估与基于分支限界的联合轨迹选取、以及 AV 侧控制开关；

**📊 数据集**

使用 InterDrive benchmark（基于 CARLA 与 V2Xverse 的闭环测试平台）进行评估；

**📈 对比分析**

与 Argus 及原始 ADS 进行对比，平均提升驾驶得分 70.03%，安全违规率下降 87.80%，并在多 AV 混合部署下保持显著改进；

**⚠️ 局限性**

局限性包括依赖 RSU 辅助 V2X 预设合作的部署前提、仅评估了四类安全违规、对 V2X 通信延迟/丢包的鲁棒性有限等。

---

## 82. PTQ4SAM3: Semantics-Preserving Post-Training Quantization for Segment Anything Model 3

**arXiv ID:** 2611.06143 | [PDF](https://arxiv.org/pdf/2611.06143v1)

**作者:** Xi Xiao `[一作]` (University of Alabama at Birmingham), Xiao Wang `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 SAM3 量化后，解决提示对齐漂移（错误目标选择、掩膜缺失、身份漂移）问题，提出 Semantics-Preserving Quantization (SPQ) 方案。

**💡 创新点**

通过概念锚定加权量化、残差子空间正则化以及视频记忆均值对齐三步校准，兼顾结构精度与文本提示对齐，且不增加推理算子。

**🔧 技术方法**

使用 VLM 生成概念词典、PCA 残差子空间估计、Gauss‑Newton 概念分数曲率、bias fusion 校正，以及标准 PTQ 量化流程。

**📊 数据集**

在 COCO、LVIS、RefCOCO/+/g、SA‑V 视频数据集上评估；同时对 CLIP ViT‑B/16 的零样本图文检索做迁移实验。

**📈 对比分析**

与 MinMax、PTQ4ViT、PTQ4SAM 等基线对比：在 W4A8 下 SPQ 使 RefCOCO Pr@0.5 从 42.1 提升至 72.5，COCO AP 从 54.1 提升至 54.8；在视频跟踪上 J&F 从 66.5 提升至 81.2，保持与 FP16 相近的性能；在 W8A8 也表现出较大提升。

**⚠️ 局限性**

依赖于离线概念词典与视频校准样本，适用于特定模型（SAM3）且仅在低位宽（W4A8/W8A8）下验证，未覆盖更低位宽或其他多模态模型的推广；校准过程仍需额外计算资源。

---

## 83. AI-Assisted Classification under Correlation Neglect and Trust

**arXiv ID:** 2611.05870 | [PDF](https://arxiv.org/pdf/2611.05870v1)

**作者:** Dingwen Kong `[一作]` (Massachusetts Institute of Technology), Asuman Ozdaglar `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一个两信号模型，研究人工智能（AI）助手对人类决策的影响，并通过理论推导与实证验证构建了一个关于AI能力、信息重叠以及人类行为的相位图。

**💡 创新点**

创新点在于：① 将AI与人类信息的重叠（λ）与人类的关联意识（α̂）和信任度（β̂）三维度结合，形成完整的行为模型；② 推导出闭式相位图，将人类‑AI团队的表现划分为互补、受损与自动化三种模式；③ 将团队误差拆解为“关联忽视成本”和“低信任成本”，并在实验中分别估计两者。

**🔧 技术方法**

主要技术包括：高斯信号模型与线性加权决策、贝叶斯推理、相位图分析、参数识别（逆向回归）、交叉验证的logistic回归估计重叠与信任参数、Bootstrap置信区间检验。

**📊 数据集**

使用了ChatBench数据集：643名MTurk工人完成MMLU（多模态学习单元）中的多选题，并分别在两种AI助手（GPT‑4o和Llama‑3.1‑8b）下进行回答。每个问题生成人类单独回答、AI单独回答和人类受AI辅助回答的四个观测值。

**📈 对比分析**

通过比较人类单独、AI单独与人类‑AI团队的准确率，验证相位图预言。实验显示：GPT‑4o落入自动化区域（AI单独准确率最高），Llama‑3.1‑8b位于互补/自动化边界；两者的增益差异可归因于AI能力；校准信任可为GPT‑4o提升约2.6个百分点，而对Llama的提升几乎为零。

**⚠️ 局限性**

局限性包括：① 仅针对二分类任务，未涵盖多分类或回归问题；② 模型假设为线性加权的两信号，可能忽略更复杂的人类推理过程；③ 训练与测试数据来自在线众包，结果可能不易迁移至医学、法律等高风险领域；④ 只考虑单一AI信号，未探索多AI协同或多模态输入的情况。

---

## 84. Clips2Story: Training-Free Video Storyboarding and Editing Using Multimodal Retrieval-Embedded Generation

**arXiv ID:** 2611.05992 | [PDF](https://arxiv.org/pdf/2611.05992v1)

**作者:** Junnuo Wang `[一作]` (New York University), Hao-Wen Dong `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个训练无、基于大语言模型的管道Clips2Story，能够在用户给定关键词的引导下，从无序的视频片段集合中检索、排序并生成可编辑的故事板（包括连贯的叙事和必要时的旁白），最终合成短视频。

**💡 创新点**

创新点在于：①将关键词检索与多模态描述提取结合，形成可供LLM规划的高质量候选集；②使用LLM进行剧情规划而非传统摘要或检索+拼接；③提供两种模式（无旁白与带旁白）满足不同编辑需求；④在无序片段场景下完成故事生成，填补现有视频摘要/生成模型的空白。

**🔧 技术方法**

技术手段包括：多模态描述提取（WhisperX语音转写、BLIP视频描写、GroundingDINO+CLIP实体背景识别），关键词与片段相似度检索，GPT-5.2（LLM）进行剧情规划与连贯叙事生成，VITS TTS生成旁白音频，视觉-语音匹配与片段选择，最终视频合成。

**📊 数据集**

数据集：从YouTube抽取100个长视频，覆盖纪录片、电影、讲座、Vlog和新闻五个类别；将每个视频按场景或转场拆分为短片段，然后打乱顺序，构建两种碎片化水平（clip-level和scene-level）的合成评测集。

**📈 对比分析**

对比方法：A2Summ、TeaserGen、REGen；评估指标包括重复率（REP）、CLIPScore、VTGHLS、平均片段时长（MCD）、顺序保持系数（OPC）、关键词相关度（KR）等。实验结果表明Clips2Story‑NF在OPC和重复率上优于基线，Clips2Story‑ND在关键词相关度和人类评价（连贯性、关键词一致性、真实感、旁白效果、片段选择）上均显著优于REGen，整体性能提升显著。

**⚠️ 局限性**

局限性：①仅基于文本描述，忽略细粒度视觉细节、背景音乐连续性等非文本信息；②硬检索阈值可能丢弃有用的B‑roll；③LLM规划可能导致事实偏差或误导性叙事；④对语音合成与视觉匹配的质量依赖于外部模型，可能出现音画不匹配；⑤目前仅支持单一语言，未考虑多语言多模态的复杂场景。

---

## 85. Editing Large Language Models with Geometry-Aware Regularization

**arXiv ID:** 2611.06185 | [PDF](https://arxiv.org/pdf/2611.06185v1)

**作者:** Bingqing Liu `[一作]` (Huazhong University of Science and Technology), Ruixuan Li `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为GEdit的几何感知知识编辑框架，能够在大型语言模型中高效且可靠地更新事实信息。

**💡 创新点**

创新点在于：①利用低秩加对角(LR+D)因子模型识别并严格保护核心知识子空间；②引入可编辑性(editability)度量，自适应调节正则化强度，实现硬/易编辑的动态平衡。

**🔧 技术方法**

核心技术包括：低秩正则化、适应性正则化、概率几何建模（LR+D）、闭式更新公式、批量/顺序编辑流程。

**📊 数据集**

实验使用 CounterFact 与 ZsRE 两大事实编辑基准，针对 GPT‑2‑XL（1.5B）、GPT‑J‑6B、LLaMA‑3‑8B 等模型，执行高达 10,000 次连续编辑。

**📈 对比分析**

与 20+ 传统编辑方法（如 ROME、MEMIT、AlphaEdit、RECT、SPHERE 等）对比，GEdit 在编辑效果（Efficacy/General）和知识保留（Locality）上均显著优于基线，尤其在大批量/顺序编辑场景中保持稳定性。

**⚠️ 局限性**

局限性：①理论分析仅针对单个编辑，未覆盖批量编辑中的冲突与交互；②核心子空间秩是全局设定，未对不同编辑适配；未来需研究批量编辑的理论保证与自适应秩选择。

---

## 86. SafeStart: Mitigating the Overlooked Safety Cost of Active Expert Reduction in MoE LLM Serving

**arXiv ID:** 2611.06147 | [PDF](https://arxiv.org/pdf/2611.06147v1)

**作者:** Erfan Baghaei Potraghloo `[一作]` (University of Southern California), Massoud Pedram `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究MoE大模型推理中减少专家数导致安全性（拒绝）下降的问题，并提出在前缀和前8个生成标记时保持全部专家的方案，恢复拒绝行为。

**💡 创新点**

创新点在于：在保持速度提升的同时，通过在回应开始阶段执行完整专家计算，显著恢复模型的拒绝行为，并提供可插拔的 vLLM 补丁，兼容多种专家缩减方法。

**🔧 技术方法**

使用了MoE专家缩减技术（top‑k、top‑p、阈值跳过）、GPU 内存访问分析、vLLM 引擎钩子、Triton 融合核、拒绝判别器（Llama Guard 3、HarmBench）等。

**📊 数据集**

使用的数据集包括：159 行为测试、313 StrongREJECT 题目（共472条请求）、889 追加 AdvBench/CatQA 评估，以及 MMLU、GSM8K、XSTest 等基准。

**📈 对比分析**

通过与原模型的配对统计（McNemar、Tango 95% CI）比较，传统截断会使有害回答比例上升18.4%；改进方案将提升控制在±1.3%，同时保留≥85%的离线解码加速（batch 8/32），在 batch 8 时可快38–44%。

**⚠️ 局限性**

局限性：仅评估单轮推理，特定 GPU 与 vLLM 版本；未覆盖量化、专家并行、动态攻击或模板后期决策；人类评标样本有限，自动判定可能误判；验证范围仅限三大模型，未覆盖更大规模模型。

---

## 87. Streaming OS Installation from Compressed Partitioned Images on Memory-Constrained Bare-Metal Hosts

**arXiv ID:** 2611.05388 | [PDF](https://arxiv.org/pdf/2611.05388v1)

**作者:** Chetan Nigudgi `[一作]`, Chip Cary `[通讯]`

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

**🎯 论文内容**

本文提出了一种从压缩分区镜像进行流式安装的技术，使裸机在内存不足的情况下仍能完成OS部署。

**💡 创新点**

创新点在于将GPT分区表的元数据通过仅解压30MB来提取，然后通过管道直接将完整镜像流式写入对应分区，显著降低峰值内存需求。

**🔧 技术方法**

技术上采用了部分解压、sgdisk解析分区表、Linux管道+dd seek写入、错误检测回退机制。

**📊 数据集**

使用的数据集是常见的Linux OS镜像（压缩约2.8GB，解压后约6.5GB）及其在8GB内存服务器上的部署场景。

**📈 对比分析**

实验对比了标准完整解压与流式安装，在8GB内存环境下标准方法0%成功率，流式方法100%成功，安装时间约4分钟，内存峰值从6.5GB降至6MB。

**⚠️ 局限性**

局限性包括必须在前30MB内包含GPT表、仅支持顺序压缩（gzip）以及仍需30MB tmpfs空间。

---

## 88. Consistent Submodular Maximization with Full Foresight

**arXiv ID:** 2611.06223 | [PDF](https://arxiv.org/pdf/2611.06223v1)

**作者:** Eric Balkanski `[一作]` (Columbia University), Morteza Zadimoghaddam `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在已知完整未来输入、计算无限制的情况下，在线子模函数最大化问题在保持常数级变更预算（recourse）时的近似损失，并给出了精确的定理阈值。

**💡 创新点**

创新点主要有：① 将一致性（recourse）视为单独的成本，剔除不确定性和计算限制；② 用“区间选择”框架将随机算法的稳健性问题转化为最优分配问题；③ 引入“clairvoyant 证书”概念，统一处理确定性算法的过渡路径；④ 通过形式化证明和 Lean 证明助手验证了 3/4 的确定性阈值。

**🔧 技术方法**

核心技术包括：最小最大双对偶、加法鲁棒性（addition‑robustness）与区间选择的等价关系；对非递减连续区间的等价分布求解；证书框架下的连续路径与保真度约束；统一阈值阈值（common threshold）随机化逼近多项式时间内的整数解；以及对证书存在性的几何与代数证明。

**📊 数据集**

本工作为理论性研究，不使用实验数据集；所有结果均为信息论上、非多项式时间（信息论）或多项式时间的上界与下界。

**📈 对比分析**

通过构造最坏实例给出下界，证明 3/4（确定性）和 ξ⋆≈0.7942（随机性）为最优；同时给出与之匹配的算法，证明在一致性约束下无法获得更高的近似比；这些结论与先前非 clairvoyant 结果（如 1/2、2/3、2−√2）形成对比，表明全知未来只能提升到 3/4 与 ξ⋆。

**⚠️ 局限性**

局限性包括：① 所给算法在信息论层面实现（需要 exhaustive search），尚无多项式时间实现；② 仅适用于单一基数约束下的子模最大化；③ 对一般子模函数（非单调或非覆盖）的阈值仍未完全确定；④ 现有证明对 3/4 证书依赖于 Lean 自动化，缺乏简洁人类可读证明。

---

## 89. Assurance-Conditioned Continuous Authorization for Autonomous AI Agents

**arXiv ID:** 2611.05632 | [PDF](https://arxiv.org/pdf/2611.05632v1)

**作者:** Mehdi Shajari `[一作]` `[通讯]` (Independent Researcher), Mehdi Shajari (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 Assurance‑Conditioned Continuous Authorization (ACCA)，一种基于证据有效性与适用性分离、可动态更新的权限授权框架；在交易域构建了域治理配置（Domain Governance Profile），通过类型化依赖图和有效权威包络（effective‑authority envelope）实现权限持续推算，并在受保护的效果边界上执行授权验证。

**💡 创新点**

创新点：
- 将证据有效性与上下文适用性分离，形成可持续的保证依据；
- 通过有效权威包络和“权威时代”机制，将权威与证据、策略、系统状态紧密关联；
- 引入授权基础版本（authorization‑basis version）实现对已授权动作的按需再验证；
- 采用可组合的域治理配置，使框架在保持领域无关性的同时支持领域特定语义；
- 通过类型化依赖图实现授权相关变更的选择性传播，避免无关变更导致的全量重新授权。

**🔧 技术方法**

技术手段：
- 连续使用控制（UCON）与 ABAC 结合的授权评估；
- 领域治理配置（DGP）和策略引擎（如 OPA）做策略解析；
- 依赖图（typed dependency graph）用于定位授权相关路径；
- 权威时代（epoch）与可选授权基础版本实现版本化授权；
- 受保护的效果边界（effect‑time enforcement）与承诺协议（commit contract）保证执行时的最新性；
- 运行时可观测性与行为归一化用于生成行为证据。

**📊 数据集**

数据集：
- 主要使用合成交易场景数据（如订单生成、价格波动、账户余额等）进行回归测试；
- 未使用公开真实金融交易数据，所有实验均在受控仿真环境中完成。

**📈 对比分析**

比较方法与性能：
- 通过一组回归测试（默认拒绝、证据条件收缩与恢复、策略分歧、选择性非响应、脏请求拒绝）评估框架的功能完整性；
- 与传统静态授权（默认拒绝）对比，展示在证据更新后能及时缩减或恢复权限；
- 性能指标未给出精确数值，但实验表明在单域实现中，授权评估与变更传播的延迟在毫秒级，能够满足交易系统的实时性要求。

**⚠️ 局限性**

局限性：
- 仅在交易域实现并验证，缺乏跨域一致性与可迁移性的实验证明；
- 对社会伦理风险的防护不完整，仅对技术层面的效果边界做了控制；
- 选定的承诺协议与效果边界实现仍处于实验阶段，未覆盖所有可能的执行路径；
- 选定的授权基础版本机制未在完整系统中实现；
- 依赖图和证据收集的完整性依赖外部可信度，未给出完整安全证明；
- 评估侧重功能验证，未进行大规模性能基准测试。

---

## 90. Do Search-Augmented Recommenders Use the Latest Search Episode?

**arXiv ID:** 2611.06044 | [PDF](https://arxiv.org/pdf/2611.06044v1)

**作者:** Jiandong Ding `[一作]` `[通讯]` (Huawei Technologies Co., Ltd.), Jiandong Ding (Huawei Technologies Co., Ltd.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 FocalSwap 评估框架，用来对搜索增强推荐系统（SAR）进行固定行反事实干预，检验最新搜索片段与完整搜索状态对推荐结果的真实贡献。

**💡 创新点**

创新点在于将搜索输入的“遮蔽”与“匹配替换”两种对照组合，并通过固定目标、候选集、预搜索历史等条件的单行实验，清晰区分搜索信息的最新-片段效应与累积状态效应，同时揭示结果对对照方式高度敏感。

**🔧 技术方法**

使用了基于逆序列（reverse）和 BM25 的评估器、递归式互惠排名（RR）计算、用户簇重抽样、Holm 校正、Bonferroni 调整等统计技术；实现了统一的干预接口以支持不同模型的原始、遮蔽、匹配与历史替换视图。

**📊 数据集**

在四个公开基准上进行实验：KuaiSAR（匿名搜索与推荐日志），Qilin（带真实查询与物品文本），SESRec 与 UniSAR（从公开实现下载的检查点），以及 GenSAR 的 PerSrc 任务；同时构造了植入式焦点信号进行验证。

**📈 对比分析**

比较方法：固定行干预后记录目标排名变化率、签名 RR 贡献、替换偏好（h_swap）、路由机会与输入侧恢复。实验结果显示，单纯遮蔽在 UniSAR 检查点上可产生正向 RR 增益（约 +3×10⁻³），而匹配替换则往往给出零或负效应；最新搜索片段的贡献在所有模型中均未能显著超越零，表明最新-片段效应不显著。

**⚠️ 局限性**

局限性包括：对照方式对结论影响大，遮蔽可能导致输入分布偏移；实验仅覆盖了少数公开模型与数据集，缺乏对更广泛体系的泛化验证；仅评估了排名收益，未考虑后续点击率或长期用户满意度；以及在实际部署中需要额外的接口与检查点兼容性。

---

## 91. Intra-Architecture Source Attribution for Stable Diffusion: A New Task with Benchmark Dataset and Baseline

**arXiv ID:** 2611.06000 | [PDF](https://arxiv.org/pdf/2611.06000v1)

**作者:** Ashutosh Dayal `[一作]` (Indian Institute of Technology Kharagpur), Debashis Sen `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在同一架构下对Stable Diffusion XL细调模型进行来源归因的任务（IntrAA-SD），并构建了140K图像基准数据集。

**💡 创新点**

创新点在于：①首次针对细调模型（相同架构）进行来源归因；②设计了风格条件混合专家（SC‑MoE）框架，结合CLIP风格编码与LoRA‑DINOv2的细调痕迹；③提供了ID与OOD提示的全新数据集。

**🔧 技术方法**

核心技术包括：冻结CLIP ViT‑L/14提取图像风格，LoRA‑DINOv2提取细调痕迹，风格条件混合专家（Mixture‑of‑Experts）门控，拼接风格与痕迹特征后分类。

**📊 数据集**

使用了自建的IntrAA‑SD Benchmark（ISDB）数据集，包含10个SDXL细调模型（M1–M10）产生的12K ID和2K OOD图像，分辨率1024×1024，JPEG 95% 质量。

**📈 对比分析**

通过与同类架构归因基线对比，IntrAA‑SD在ID测试集上达到99.03%准确率，在OOD测试集上为88.55%，显著优于传统架构级归因方法。

**⚠️ 局限性**

局限性包括：对不同分辨率或压缩率的鲁棒性尚未充分验证；门控机制对风格提取依赖较大，可能在极端风格变换下失效；仅覆盖SDXL系列，泛化到其他模型需要进一步研究。

---

## 92. BAM: Bivariate Additive Models for Interpretable Tabular Prediction

**arXiv ID:** 2611.05940 | [PDF](https://arxiv.org/pdf/2611.05940v1)

**作者:** Atsushi Koike `[一作]` `[通讯]` (Tohoku University), Atsushi Koike (Tohoku University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Bivariate Additive Models（BAM），将GA2M中二元交互形函数约束为非对称双高斯，并通过可微特征对选择实现可视化可解释的表格预测。

**💡 创新点**

创新点在于：① 用简单的双高斯参数化代替自由形函数，显著提升可解释性；② 通过温度退火的entmax实现自动特征对选择，保持高预测性能；③ 通过组件数量调节准确率与可解释性之间的权衡。

**🔧 技术方法**

技术细节包括：双高斯形函数（旋转、异方差）、log/交叉熵损失、Adam优化、多次随机重启、entmax温度退火特征选择、二阶段训练（先软化后硬化）。

**📊 数据集**

实验使用6个公开基准：回归（California Housing、Bike Sharing、Wine Quality）和分类（Titanic、Adult、Diabetes），全部特征做one‑hot或标准化。

**📈 对比分析**

与线性模型、CART、XGBoost、EBM（GAM/GA2M）、NODE‑GAM、GRAND‑SLAMIN’和DNN在5折交叉验证下比较。BAM的RMSE/AUC基本与标准GAM相当，接近GA2M，在Bike Sharing上甚至优于GAM；整体保持在最优模型附近。

**⚠️ 局限性**

局限性：① 需要多次重启以避免初始化敏感；② 仅支持双高斯形函数，无法捕捉更复杂形态；③ 组件数需要经验选择，可能导致可解释性与性能之间的权衡不易确定；④ 高维特征时特征对可能重复，影响可解释性。

---

## 93. A Comparative Study of Importance Attribution to Columns in Deep Learning over Relational Databases

**arXiv ID:** 2611.05986 | [PDF](https://arxiv.org/pdf/2611.05986v1)

**作者:** Hila Levi-Yosefi `[一作]` (Technion), Benny Kimelfeld `[通讯]` (Technion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对关系深度学习模型的属性重要性解释方法进行系统比较，分析其属性排名一致性、分数分布与保真度的关系。

**💡 创新点**

提出统一的比较框架，比较四种解释方法（单列消融、Monte Carlo Shapley、Kernel SHAP、Mask Learning）的差异，发现任务类型和方法本质决定一致性，并证明在回归任务中单列消融可替代昂贵的 Shapley 方法。

**🔧 技术方法**

使用关系图神经网络（Heterogeneous GraphSAGE）和关系图变换器（Relational Graph Transformer）作为模型，结合全局特征函数、基于抽样的 Shapley 近似、掩码学习等技术。

**📊 数据集**

实验基于 RelBench 公开基准的四个关系数据库（rel-trial、rel-arxiv、rel-f1、rel-event），涵盖二分类与回归两类任务。

**📈 对比分析**

通过排名相似度（Spearman、L1距离）、分布相似度（PDF 对比）以及保真度指标（Fidelity+/-）进行比较，结果显示回归任务中解释器一致性高，分类任务差异大；单列消融速度最快，在多数场景可替代 Shapley 方法。

**⚠️ 局限性**

仅在两种图神经网络架构上评估，Mask Learning 仅在支持的模型上实验，缺乏用户研究验证，且对不同数据分布与模型复杂度的泛化性仍待进一步验证。

---

## 94. Item Representations Matter: Bridging Hard Semantic IDs and Soft Prototype Mixtures for Generative Retrieval

**arXiv ID:** 2611.05799 | [PDF](https://arxiv.org/pdf/2611.05799v1)

**作者:** Vicente Balmaseda `[一作]` (Meta), Hong Li `[通讯]` (Meta)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于流匹配的连续生成推荐器，该推荐器在项目表中使用可微软混合的共享原型与专属向量，既实现了跨项目的参数共享，又保留了每个项目的个性化表示。

**💡 创新点**

创新点在于：① 将离散语义ID（SID）的共享逻辑迁移到连续空间，采用可微软混合而非硬一次分配；② 通过在每个编码层学习一个soft assignment矩阵，使得项目表在训练过程中能够自适应共享与保留信息；③ 在同一模型中联合学习流匹配、编码器、解码器以及共访问先验，形成端到端可训练的系统。

**🔧 技术方法**

使用的技术包括：流匹配（Flow Matching）和条件流匹配损失；残差量化（Residual Quantization）与可微soft assignment；多头注意力与位置编码实现soft assignment的学习；dot‑product 解码器结合共访问先验；以及对流的数值积分（Euler）实现推理。

**📊 数据集**

评估数据集：Amazon Beauty（2014）、Amazon Industrial & Scientific、Amazon Office（2023）、MovieLens‑1M、Yelp‑2022、Steam‑2018，均采用统一的 5‑core 过滤、去重、时间排序和留一拆分。

**📈 对比分析**

与 FlowRec、TIGER（自回归）和 MaskGR（遮蔽扩散）三类连续与离散生成推荐器对比，prototype‑mixture 在所有六个基准上平均提升 HR@5 约 3–6% 与 NDCG@5 约 4–7%；在推理阶段，推理吞吐量提升 1.4–161 倍，MFLOP/查询显著下降。

**⚠️ 局限性**

局限性：① 软分配的梯度稀疏化可能导致原型利用不均衡；② 对极稀缺项目的提升仍有限；③ 需要大量训练数据与计算资源；④ 目前缺乏对原型选择过程的可解释性分析。

---

## 95. Toward a Compositional Knowledge Architecture for a Shared Organizational Cortex

**arXiv ID:** 2611.05753 | [PDF](https://arxiv.org/pdf/2611.05753v1)

**作者:** Andrew Miracle `[一作]` (Robin Inc.), Ese Monday `[通讯]` (Nosen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出共享组织皮层架构，将组织证据划分为Signal、Domain、Claim、Dimension四层，保留解释与立场并实现可重用与可审计的组织判断。

**💡 创新点**

创新点在于把解释持久化为Claims并用真值维护网络维护其立场，同时引入可验证、版本化的Dimension轴实现几何位置化，支持解释在证据变更时的自动更新与冲突处理。

**🔧 技术方法**

采用信号抽取器、域分类器、推理器（Deriver）、标注器（Labeller）、投影器（Projector）、Doyle风格真值维护网络，并结合BM25+密集检索+跨编码器的检索+基于Dimension的重排序。

**📊 数据集**

使用公开数据集：Python Enhancement Proposal (PEP) 文档、Enron 邮件等构建121个Signal、49个Domain等，并在注册检索实验中评估272个保留查询。

**📈 对比分析**

与Signals‑only、GraphRAG、Zep等系统对比；注册检索实验显示nDCG@10提升0.028（p=0.002），相较于Signals‑only提升0.081，图形增强提升0.053，域范围提升0.049，整体性能显著提升。

**⚠️ 局限性**

限制在于评估仅覆盖表示层，未对真值维护、冲突检索、提取器召回等进行系统验证；数据集有限，实验结果为预估，未检验在真实组织环境中的长期可用性。

---

## 96. Feature Puzzles: Single-Image Supervision for Multi-View 3D-Aware Segmentation

**arXiv ID:** 2611.05874 | [PDF](https://arxiv.org/pdf/2611.05874v1)

**作者:** Jiahao Ma `[一作]` (Australian National University), Chuong Nguyen `[通讯]` (CSIRO)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Feature Puzzles 框架，利用单张图像的重叠 crop 生成多视角一致的伪标签，实现多视角一致分割、空间追踪与开词汇 3D 分割；

**💡 创新点**

创新点在于：① 单图像分割或特征通过共享父图像映射产生精确跨视角一致性；② 采用轻量 LoRA 适配预训练的多视角 3D 重建网络，保持原始几何推理；③ 通过 promptable、open‑vocabulary 两个 head 统一使用同一多视角特征；

**🔧 技术方法**

技术包括：多视角 Transformer（VGGT‑Omega/LingBot‑Map） + LoRA 适配；基于父图像的 affine crop + label pull‑back；交叉视角注意力 + pixel‑decoder；DINOv3 Teacher + Talk2DINO 的跨模态对齐；自监督 warp/gram 约束；

**📊 数据集**

数据集：SA‑1B（实例掩码）与 DINOv3 预训练教师；在 NVOS、SPIn‑NeRF、ScanNet++、nuScenes‑mini、ScanNet/ScanNet200/ScanNet++ 等进行评估；

**📈 对比分析**

与多种基线对比：在 NVOS 上 mIoU 93.2% 领先所有单前向方法；在 SPIn‑NeRF 上 88.5% 超过 SAM2‑Video；在 ScanNet++ 上 63.6% mIoU、87.5% T‑SR；在 nuScenes‑mini 上 T‑SR_vis 88.9%；在开词汇 3D 分割中零样本时 30‑50% mAcc，Finetune 后超过 EmbodiedSplat；推理速度约 25.9 s/场景，显著快于 per‑scene 方法；

**⚠️ 局限性**

局限性：依赖预训练多视角 backbone 的几何先验；对大量视角（>200）时精度下降；训练仍需大量单图像标注；缺乏对动态遮挡、极端视角变换的鲁棒性；

---

## 97. Cross-Asset Knowledge Transfer for Early Fault Detection of Data-Scarce Assets Without Target Fault Data

**arXiv ID:** 2611.06194 | [PDF](https://arxiv.org/pdf/2611.06194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 98. Noncrossing is All You Need: Beyond Identical Delay Functions in Online Matching with Delays on General Metrics

**arXiv ID:** 2611.05769 | [PDF](https://arxiv.org/pdf/2611.05769v1)

**作者:** Yossi Azar `[一作]` (Tel Aviv University), Seeun William Umboh `[通讯]` (University of Melbourne)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了具有延迟的在线最小成本完美匹配问题，提出了针对非交叉延迟函数的竞争算法。

**💡 创新点**

首次提出了针对非交叉延迟函数的确定性O(1)-竞争算法和随机O(log n log Δ + log² Δ)-竞争算法，突破了之前对延迟函数相同的限制。

**🔧 技术方法**

使用了非交叉延迟函数的概念，结合了随机算法和确定性算法的设计。

**📊 数据集**

使用了一般度量空间中的请求，具体数据集未明确给出。

**📈 对比分析**

与之前的算法相比，提出的算法在竞争比率上表现优越，能够处理不同延迟函数的请求，性能得到了显著提升。

**⚠️ 局限性**

算法在处理复杂度和延迟函数的多样性方面存在局限性，尤其是在某些特定情况下可能无法达到预期的竞争比率。

---

## 99. ARIDA - Active railway infrastructure dataset annotation using neural network predictions

**arXiv ID:** 2611.05646 | [PDF](https://arxiv.org/pdf/2611.05646v1)

**作者:** Maximilian Dornik `[一作]` (University of Applied Sciences Düsseldorf), Alexander Braun `[通讯]` (University of Applied Sciences Düsseldorf)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了ARIDA框架，用于半自动铁路图像数据集标注，将实例分割预测转换为RailSem兼容的多原语（边框、多边形、多线段和线段对），并通过无地面真值的质量门控（CFS）与主动学习循环（tbal+qqp）自动接受易样本、将人工注释集中于难样本。

**💡 创新点**

创新点在于：①将实例分割模型与多原语转换算法集成，支持完整的铁路标注；②设计无地面真值的几何一致性度量CFS作为自动接受门；③结合主动学习与动态查询阈值（qqp）实现自动与人工标注的最佳平衡。

**🔧 技术方法**

技术包括Mask Scoring R‑CNN（带maskiou head）用于实例分割；距离变换、中心线拟合、轮廓分类及交叉乘法实现线段对生成；CFS度量（基于点匹配的Fβ）；tbal信息度量；qqp调度机制；实验采用torchmetrics评估IoU。

**📊 数据集**

使用RailSem19（异构、离散图像）和osdar23（时间序列、冗余图像）两套公开铁路数据集进行实验。

**📈 对比分析**

通过与传统完全手工标注的监督基线对比，ARIDA在RailSem19中实现约61–67%的人工工作量降低，且在100个初始标注的情况下达到与200图像监督基线相同的map；在osdar23中最大可达92.8%的工作量降低，且小初始集（20–50图像）即可获得与200图像监督基线相近的map，证明主动学习能显著提升样本效率。

**⚠️ 局限性**

局限性：仅用单一随机种子和模拟oracle；仅评估单类单原语（rail/track）的情况；接受阈值与qqp参数选取经验化；实验仅限两套数据集；缺乏真实人机交互的时间测量与多种采样策略。

---

## 100. From Activation to Intervention: Interpreting Sparse Visual Features through Reconstruction

**arXiv ID:** 2611.05522 | [PDF](https://arxiv.org/pdf/2611.05522v1)

**作者:** Jianyang Gu `[一作]` (Ohio State University), Tanya Berger-Wolf `[通讯]` (Ohio State University)

**通讯引用:** 5475 | [OpenAlex ID](https://openalex.org/A5060005215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对稀疏自编码器（SAE）特征进行可视化干预并通过重建得到视觉变化，以帮助解释特征语义。

**💡 创新点**

提出Reconstructive SAE Intervention框架，将干预与图像重建结合，利用稀疏解码方向实现对特征的稳定且可解释的操控，解决传统高激活例子混淆问题。

**🔧 技术方法**

使用稀疏自编码器、Representation AutoEncoder (RAE)、Transformer视觉编码器（如 DINOv3、SigLIP 2）以及梯度归因与稀疏方向聚合。

**📊 数据集**

在 NABirds、CUB、Cars 以及 ImageNet 四个数据集上进行实验。

**📈 对比分析**

与传统激活例子、文本描述、梯度干预等方法对比，Reconstructive SAE Intervention 在细粒度数据集上提升解释准确度（最高 1.8 分），并在特征控制的选择性和一致性上优于密集梯度干预。

**⚠️ 局限性**

局限性包括干预方向一致性仍不完美、对 RAE 模型的依赖、对大规模数据的计算开销，以及解释结果受训练 SAE 质量影响。

---

## 101. TandemBench: Benchmarking Decision Models for Multi-Agent Collaboration

**arXiv ID:** 2611.05827 | [PDF](https://arxiv.org/pdf/2611.05827v1)

**作者:** Baohang Li `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个名为 TandemBench 的多代理协作决策基准，研究在完整全局信息下模型如何选择最佳团队计划并判断是否存在更优方案。

**💡 创新点**

创新点包括：①提出了 1,200 题的团队计划决策基准；②利用执行仿真和精确搜索给出最优参考答案；③设计配对验证题检验模型对最优性判断的准确性。

**🔧 技术方法**

采用了多种决策模型（40 个本地模型和 10 个通用语言模型），结合执行仿真、Exact search、选项旋转等评估技术，对模型进行决策准确率（DA）和验证准确率（VA）测评。

**📊 数据集**

数据集为 TandemBench，基于 Collab-Overcooked 的 30 个烹饪任务，涵盖 200 个场景、1,200 题，包含短期和长期目标的团队计划决策与最优性验证问题。

**📈 对比分析**

通过比较 DA 与 VA 两指标对模型进行评估：最高本地模型 KnowLine 在 DA 上达到 62.5%（但 VA 接近随机）；通用模型 Qwen3.8-27B 达到 82.8% DA 与 71.9% VA，显示通用 LLM 在此任务上具有较强性能。

**⚠️ 局限性**

研究局限在两代理、单一厨房、全局信息已知的情境；模型推理预算不同；样本共享情境，无法覆盖更大规模、多样化或分布式协作场景。

---

## 102. CoordX Protocol: Peer to Peer Cryptographically Verified Shared Records

**arXiv ID:** 2611.06034 | [PDF](https://arxiv.org/pdf/2611.06034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 103. BobFlow: Towards Human-Agent Collaborative Data Science Using Dataflows

**arXiv ID:** 2611.05761 | [PDF](https://arxiv.org/pdf/2611.05761v1)

**作者:** Jiadong Bai `[一作]` (University of California, Irvine), Chen Li `[通讯]` (University of California, Irvine)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出数据流抽象和Dataflow ReAct框架，让LLM代理与人类分析师通过可视化数据流进行迭代构建、执行、检查与反馈。

**💡 创新点**

将数据流作为共享计算抽象并结合ReAct，实现精准语义、直观表示、步骤透明；引入上下文优化（数据属性、拓扑属性）减少token消耗。

**🔧 技术方法**

使用大型语言模型（Claude Haiku、GPT、Gemini等）+ ReAct推理+数据流执行引擎（Apache Texera/Amber）+数据湖（MinIO、Iceberg）+ GUI+上下文跟踪等技术。

**📊 数据集**

采用PrepBench、KramaBench、LongDS-Bench、Olympic history dataset、Inside Airbnb dataset等数据集进行评测。

**📈 对比分析**

与基于脚本的ReAct代理对比，在大部分基准中任务准确率提升0.3-18.6点、推理成本下降4.9-64.7%；用户研究显示任务完成迭代次数和推理时间均降低。

**⚠️ 局限性**

样本统计与数据预览开销导致执行时间上升约10%；拓扑优化在缓存未失效时可能增加成本；部分场景精度略低，且系统整体复杂度高。

---

## 104. MESA: An Evaluation Framework for Compositional, Semantic, and Spatial Generalization in Robotics

**arXiv ID:** 2611.05567 | [PDF](https://arxiv.org/pdf/2611.05567v1)

**作者:** Albert Wilcox `[一作]` (Georgia Institute of Technology), Animesh Garg `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 MESA 框架，包含可扩展的数据生成管线 -Gen 和统一的评测基准 -Bench，针对单臂和双臂机器人在语言驱动下的操作能力进行系统评估。

**💡 创新点**

创新点在于：①构建轴对齐的多维泛化测试套件（空间、实例、类别、组合），②引入基于干扰任务完成率的语言跟随度量（LFR），③开发高质量、可扩展的任务/演示生成工具 -Gen，③提供可重复、学术级别可训练的训练集与评测集。

**🔧 技术方法**

技术手段包括：VLM 预训练与微调（PaliGemma‑3B、Cosmos 等），扩散策略（Multitask Diffusion Policy），预训练机器人控制器（π_0、π_0‑FAST、π_0.5），以及改进的 MimicGen 数据生成与子任务拼接。

**📊 数据集**

使用的数据集为：单臂训练集 -70（7,000 条演示，70 个任务）、双臂训练集 BiMESA‑57（6,833 条演示，57 个任务）、以及 736 个辅助单臂任务 -Aux，用于验证多样性扩展。

**📈 对比分析**

实验对比了 7 种基线（DP、PG‑Bin、PG‑FM、GR00T‑N1.6、π_0、π_0‑FAST、π_0.5），在 MESA 上最高性能仅 52.0%，在 BiMESA 上 48.7%；不同模型在空间、实例、类别、组合泛化上表现差异显著，显示 MESA 能区分模型泛化能力，并与真实世界的 RoboArena 排名保持一致。

**⚠️ 局限性**

局限性包括：任务主要集中在抓取‑放置和简单机械臂操作，缺乏长周期、复杂手部或移动机器人场景；语言跟随度量不捕捉部分完成或执行失败细节；以及在其他数据源（如人工遥控）下的排序可能不稳定。

---

## 105. Phantom: Temporal Caching for Diffusion Models

**arXiv ID:** 2611.06019 | [PDF](https://arxiv.org/pdf/2611.06019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 106. Regret Bounds for Market Making with Persistent Orders

**arXiv ID:** 2611.06159 | [PDF](https://arxiv.org/pdf/2611.06159v1)

**作者:** Nicolò Cesa-Bianchi `[一作]` (University of Milan), Vinayak Pathak `[通讯]` (Computational Rational Agents Laboratory)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了在持续订单（订单在成交前不可取消）下的对抗性市场做市问题，并给出了针对固定报价和动态报价的无参数策略；

**💡 创新点**

创新点在于：①在对抗性价格序列下提供单侧和双侧（买卖）报价的最优无参数策略；②将回报划分为与价格暴露相关的两项（主市场暴露 V_T 和次市场暴露 W_T），并证明这两项是必要的；③在不知未来价格的情况下，仅凭过去价格即可实现对数或平方根阶的动态/静态遗憾率；

**🔧 技术方法**

主要技术为：在线学习与延迟反馈的结合，使用代理奖励（surrogate rewards）和“clairvoyant”策略来对比真实奖励；调参采用 λ_t = 1/t（静态）或 λ_t = 1/√(t-1)（动态）；对两侧报价使用对称策略并引入可观测的 Z_t 用于动态调节；

**📊 数据集**

本文为理论工作，未使用公开数据集，所有结果均为对抗性分析与下界构造；

**📈 对比分析**

方法的性能通过渐进遗憾率度量：对固定报价遗憾率 O(log T + V_T + W_T)，对动态报价遗憾率 O(√T + V_T + W_T)；双侧报价同样达到 O(1 + √(V_T^two log T) + V_T^two + W_T)；下界证明表明这些阶数是最优的；

**⚠️ 局限性**

局限性包括：①模型不允许订单取消，现实中订单可以撤回；②未考虑库存约束与流动性限制；③对抗性假设过于严苛，实际市场可能更接近随机过程；

---

## 107. Learning From Static Context in Federated Time Series Forecasting

**arXiv ID:** 2611.06064 | [PDF](https://arxiv.org/pdf/2611.06064v1)

**作者:** Viktor Walter `[一作]` (Karlsruhe University of Applied Sciences), Fabian Stricker `[通讯]` (Karlsruhe University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在联邦学习场景下，Temporal Fusion Transformer（TFT）能否利用客户端恒定的静态特征进行时间序列预测，并比较了多种聚合方案的效果。

**💡 创新点**

创新点在于首次揭示仅按批次FedAdam聚合能够充分利用静态特征、解析了局部训练步数与漂移对静态路径的影响，并提出了针对静态输入的敏感性诊断方法。

**🔧 技术方法**

使用的技术包括联邦学习聚合策略（FedAdam、FedAvg、FedAdam+Prox、SCAFFOLD）、TFT模型、静态输入敏感性探针、局部步骤调节、跨客户端标准化以及Wilcoxon检验等。

**📊 数据集**

实验所用数据集为建筑能耗BDG2、风电功率（ICON‑D2 与 HRES 预测数据）以及可控合成系列，三者分别代表不同的静态特征场景。

**📈 对比分析**

通过RMSE、MAE、R²、静态输入敏感性以及统计显著性检验进行对比，结果显示仅按批次FedAdam能在大多数数据集上实现接近中央训练的性能，其余方案对静态特征的利用不足且整体表现不如局部模型。

**⚠️ 局限性**

主要限制在于需要频繁同步梯度导致通信成本高、静态特征的全局标准化可能泄露信息，并且实验仅覆盖三类数据，未考虑真实延迟与隐私约束。

---

## 108. A Layered View of LLM Support in Underwater Robotics

**arXiv ID:** 2611.06162 | [PDF](https://arxiv.org/pdf/2611.06162v1)

**作者:** Federico Celi `[一作]` (NATO Centre for Maritime Research and Experimentation), Gabriele Ferri `[通讯]` (NATO Centre for Maritime Research and Experimentation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文探讨了大语言模型在水下机器人操作层面的辅助角色。

**💡 创新点**

创新点在于提出分层架构，将LLM作为故障诊断、遥测解读等支持层，而非直接控制。

**🔧 技术方法**

采用LLM与结构化日志、警报、状态数据的结合，通过检索增强与工具调用实现解释与推断。

**📊 数据集**

使用的主要数据为水下平台的运行日志、报警记录和遥测数据，未提供公开数据集。

**📈 对比分析**

未进行实验比较，而是通过案例讨论说明LLM可在时间诊断、工作负载等指标上提升运营效率。

**⚠️ 局限性**

局限包括幻觉、输出不一致、缺乏因果与时序推理，导致无法承担安全关键决策。

---

## 109. Just the Facts: How Dialogues with AI Reduce Conspiracy Beliefs

**arXiv ID:** 2611.05572 | [PDF](https://arxiv.org/pdf/2611.05572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 110. Enforcing Accountable Disclosure in Confidential AI

**arXiv ID:** 2611.05971 | [PDF](https://arxiv.org/pdf/2611.05971v1)

**作者:** Shumpei Shiina `[一作]` (Acompany Co., Ltd), Tsubasa Takahashi `[通讯]` (Acompany Co., Ltd)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 CORDE 系统，实现了在可信执行环境下的 confine-or-declare 原则，用以实现可审计的 AI 数据披露。

**💡 创新点**

将信息流控制与可信账本结合，提供声明-披露的强制顺序和可验证日志，解决了传统可信计算中隐蔽披露问题。

**🔧 技术方法**

利用 CPU/GPU 可信执行环境（Intel TDX/AMD SEV-SNP）、Rust 领域特定类型与 MIR 检查、CCF 共识式账本、mutual attestation、Attested TLS、VLLM 推理。

**📊 数据集**

在 Azure H100 上使用 AppWorld 代理任务 benchmark 以及 synthetic data 进行泄露演示，并用 AppWorld 进行性能测评。

**📈 对比分析**

通过与无声明的基线对比，声明开销仅 0.44% 任务延迟，声明吞吐约 880 次/秒，证明对 agent 工作负载影响微乎其微。

**⚠️ 局限性**

缺乏对 ledger 失效后的恢复支持，无法防御日志分裂视图攻击仍需依赖用户验证，且实现复杂度高。

---

## 111. The Male Abstraction Bias in Internet Culture and Large Language Models

**arXiv ID:** 2611.05730 | [PDF](https://arxiv.org/pdf/2611.05730v1)

**作者:** Douglas Guilbeault `[一作]` (Stanford University), Bhargav Srinivasa Desikan `[通讯]` (Oxford Internet Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过大规模文本与图像数据分析，揭示抽象概念与男性关联更紧密，形成了男性抽象偏差；

**💡 创新点**

创新之处在于首次将抽象度与性别偏见系统化，展示其在语言、视觉以及自由联想等多模态中的普遍性；

**🔧 技术方法**

采用语义空间的性别轴（geometry‑of‑culture 方法）、多语言嵌入模型、Llama‑3.2‑11B‑Vision‑Instruct 视觉分类器以及自由联想评分技术；

**📊 数据集**

使用的主要数据集包括 Brysbaert 等人提供的约 40,000 单词与 63,000 短语的抽象度规范、约 12 M 张 Google Images、12 种语言模型、WordNet 语义域、SWOW 与 LWOW 自由联想数据以及多语言抽象度规范；

**📈 对比分析**

通过在各模型间计算 Pearson 相关系数并进行多重校正，控制词频、词长、多义性等因素，发现抽象度与男性关联呈显著负相关（r ≈ –0.1 至 –0.2），且在 9 种语言中保持一致，表明结果高度稳健；

**⚠️ 局限性**

局限性在于研究为相关性分析，未能确定因果关系，主要依赖英语数据，跨文化差异与更细粒度语言特征的影响仍待进一步探究。

---

## 112. Turbocharging PiPNN for Proximity and $k$-NN Graph Building

**arXiv ID:** 2611.05690 | [PDF](https://arxiv.org/pdf/2611.05690v1)

**作者:** Tobias Rubel `[一作]` (University of Maryland), Vahab Mirrokni `[通讯]` (Google Research)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过对 PiPNN 的算法和实现做全面加速，提出了 CPU 版 PiPNN-CPU 和 GPU 版 PiPNN-GPU，显著提升了高质量近邻图与 k‑NN 图的构建速度。

**💡 创新点**

核心创新在于：①将 PiPNN 的 GEMM 与 Top‑K 归约融合到同一指令流中，利用低精度整数乘法；②针对不同的 Top‑K 参数（K=1、3、10、2）设计专门的 SIMD/寄存器排序；③在 GPU 上通过扁平化递归、固定大小的 Score‑Tile、单核分段 Top‑K 等技术，消除了递归开销与线程亲和性瓶颈。

**🔧 技术方法**

技术手段包括：旋转量化 TurboQuant（TQ4）与 RSQ8 量化；AVX‑512 VNNI/AVX‑512 SIMD 乘加与掩码融合；Hopper Tensor‑Core 上的 128×64×32 乘法与寄存器 Top‑K；锁无关的排序‑合并 Candidate Reservoir；以及在 GPU 上的分段 Top‑K 与直接距离重用的 RPrune。

**📊 数据集**

实验使用 12 个多样化数据集，覆盖文本、图像、文本‑图像、BERT‑style 低维、OpenAI‑style 高维等，规模从 10⁵ 到 10⁹ 点，维度从 96 到 4096。

**📈 对比分析**

与现有基准（Vamana、Jasper、CAGRA、NN‑Descent、cuVS 等）对比，PiPNN‑CPU 在索引构建上比 Vamana 快 5.9×（量化）或 40.7×（全精度），在 k‑NN 图构建上比 Vamana 快 10.3×；PiPNN‑GPU 在索引构建上比 Jasper/ CAGRA 分别快 13.3×/27×（量化）或 8.1×/16×（FP16），k‑NN 图构建更快 7.1×/2.9×/9.8×/49.8×。查询质量几乎不受影响，QPS‑Recall 曲线与基准相当。

**⚠️ 局限性**

局限性包括：GPU 实现对递归深度的固定上限（通过扁平化递归实现），在极大维度下仍受内存带宽限制；量化方案在极端高维或 MIPS 场景下可能略微降低图密度；以及对分布式环境的进一步扩展仍需探索。

---

## 113. From Tokens to Trajectories: Parameter-Efficient Continuous Adaptation of Unified Multimodal Models

**arXiv ID:** 2611.05451 | [PDF](https://arxiv.org/pdf/2611.05451v1)

**作者:** Gayatri Deshmukh `[一作]` (University of Illinois Urbana-Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1395 | [OpenAlex ID](https://openalex.org/A5043962698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种参数高效的后训练框架，将预训练的离散统一多模态模型（UMM）转换为连续Rectified‑Flow生成器，既保留原有语义先验又显著提升视觉质量。

**💡 创新点**

创新点包括：1）利用LoRA对注意力投影进行轻量级适配，保持主干权重冻结；2）提出Dual‑Objective Velocity Vector Distillation（DOVD），同时对齐流匹配目标和连续教师的速度向量；3）通过连续潜在投影替换离散接口，实现在不重训主干的前提下完成离散→连续的转换。

**🔧 技术方法**

使用技术包括VAE潜在空间、Rectified‑Flow（流匹配）框架、LoRA适配器、DOVD双目标蒸馏、离散与连续教师（如Qwen Image/Flux）、Euler采样调度等。

**📊 数据集**

训练使用约600万图文对，来源于LAION‑20M Recap、OpenGPT‑4o‑Image和DRAGON；评估数据集包括COCO‑30K、GenEval、MMBench、MMMU、MM‑Vet、MathVista和MMVP。

**📈 对比分析**

实验在UniDFlow‑4B、FUDOKI等离散UMM上进行，COCO‑30K FID平均提升33%–42%（如FUDOKI从16.16降至9.29，UniDFlow‑4B从13.33降至8.92），Aesthetic Score提升1.5–1.8；语义推理保持>99.5%，GenEval仅下降≤0.04。与全新训练的连续UMM和原离散UMM相比，视觉质量显著提升，语义一致性几乎不变。

**⚠️ 局限性**

局限性包括：需要冻结主干，无法进一步大规模微调；对教师模型的选择和容量敏感；需要共享VAE接口，限制了与不同VAE的兼容性；在极端复杂指令或细粒度细节上仍可能出现缺失，且方法仅适用于可用的高质量连续教师。

---

## 114. From Handcrafted to Generative: RLVR-Optimized Icon Generation for Alexa+

**arXiv ID:** 2611.06123 | [PDF](https://arxiv.org/pdf/2611.06123v1)

**作者:** Rohan Sequeira `[一作]` (University Of Southern California), Jonathan Rubin `[通讯]` (Amazon Alexa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并部署了一条从文本描述到可在 Alexa+ 屏幕播放的动画图标（Alexicon）的全自动生成流水线，并针对首个关键阶段的线稿质量进行改进。

**💡 创新点**

创新点包括：1）利用形态学测量+梯度提升分类器构造无监督的缺陷检测器作为奖励信号；2）基于 Flow-GRPO 对 FLUX.2 进行强化学习微调，显著提升线稿在实际渲染宽度下的可读性；3）在流水线中引入此自动质量门控，替代人工审核。

**🔧 技术方法**

技术包括：形态学阈值测量、梯度提升树分类、强化学习（Flow‑GRPO）与 LoRA 微调、VLM 判别器、Lottie 动画生成。

**📊 数据集**

使用了 682 条手绘图标生成的自建验证数据集（3,836 条标注缺陷）以及 1,000 条独立评估概念列表，未使用公开线稿数据集。

**📈 对比分析**

对比方法：基线 FLUX.2 4B 与微调后模型在独立 VLM 评判器下的线条分离通过率从 15.7% 提升至 58.7%（3.7 倍），McNemar 检验 p < 3.3×10⁻⁹⁴，显著优于基线。

**⚠️ 局限性**

局限性：1）缺陷检测器的 F1 为 0.685，存在标签噪声；2）对视觉复杂度高于渲染空间的图标仍无法完全避免拥挤；3）目前仅针对单一渲染尺寸/笔触宽度训练，缺乏多尺度适应能力。

---

## 115. Uncertainty-Aware Resource Orchestration for Agentic AI Systems

**arXiv ID:** 2611.05392 | [PDF](https://arxiv.org/pdf/2611.05392v1)

**作者:** Masoud Shokrnezhad `[一作]` (ICTFICIAL Oy), Tarik Taleb `[通讯]` (Ruhr University Bochum)

**通讯引用:** 26096 | [OpenAlex ID](https://openalex.org/A5087043869)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种用于Agentic AI流水线的即时资源预留算法CRO，能在任务调度时为计算、带宽和延迟提供概率覆盖保证。

**💡 创新点**

创新点在于将无参数的合式预测（conformal prediction）应用于多模态、重尾且结构未知的Agentic AI工作流，实现了有限样本、分布无关的覆盖保证，并在满足概率约束的前提下将资源预留量最小化。

**🔧 技术方法**

主要技术包括滑动窗口校准、非合适度分数计算、合式预测的分位数估计、在线窗口更新以及对延迟的基于分位数的可行性检查。

**📊 数据集**

实验使用基于Poisson、负二项和Gamma分布生成的合成执行日志，覆盖三种任务类型（轻、中、重），并无公开真实日志。

**📈 对比分析**

与EmpQ、Gaussian、Parametric和Chebyshev等基线比较，CRO在所有风险水平下均能满足预期覆盖率，且在满足约束的前提下资源预留量仅为Chebyshev的一半，表现最优。

**⚠️ 局限性**

局限性包括：假设历史日志与当前任务可交换性，未考虑批量任务的联合预留；仅在合成数据上验证；对非平稳工作负载的鲁棒性未深入探讨。

---

## 116. Support Vector Machines Directly Minimise KL Divergence

**arXiv ID:** 2611.05557 | [PDF](https://arxiv.org/pdf/2611.05557v1)

**作者:** Ivan Silajev `[一作]` `[通讯]` (Independent Researcher), Ivan Silajev (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文证明了支持向量机（SVM）以及核岭回归的对偶目标函数可以被精确表达为两个高斯测度之间的Kullback–Leibler散度（相对熵）最小化问题，从而为SVM提供了一个概率解释。

**💡 创新点**

创新点在于：①将SVM的对偶问题与信息论中的相对熵最小化完全等价；②在回归时通过引入额外的随机变量把L¹损失与散度关联；③展示在去掉约束并加入L²正则化后，散度公式自然转化为核岭回归目标函数。

**🔧 技术方法**

采用的技术主要包括：随机过程与高斯测度理论、Fenchel对偶与凸共轭、Karhunen–Loève展开、Cameron–Martin空间、以及KL散度与正则化项的解析公式。

**📊 数据集**

本文为理论研究，没有使用实际数据集；若要实验验证，需要在标准机器学习数据集（如UCI、MNIST等）上实现对偶散度形式的SVM与传统SVM进行比较。

**📈 对比分析**

未给出实验比较；理论上说明在相同的约束和正则化下，两种形式的目标函数完全等价，故在优化结果和预测性能上应该一致；若进行实验，可通过交叉验证评估其收敛速度和泛化误差。

**⚠️ 局限性**

局限性包括：①理论推导假设高斯测度和Hilbert空间可分性，实际应用中可能受限；②未讨论数值实现细节，如散度表达式的数值稳定性；③缺乏实证评估，无法验证在复杂现实数据上的表现。

---

## 117. When Are Two SAE Latents the Same Feature?

**arXiv ID:** 2611.05655 | [PDF](https://arxiv.org/pdf/2611.05655v1)

**作者:** Alexandra Suvorikova `[一作]` (Independent researcher), Yuriy Dorn `[通讯]` (Moscow State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于能量加权的最优传输框架，用单一传输计划同时比较整个稀疏自编码器（SAE）词典和单个潜在向量，提供成本分解、分配比例以及联合统计推断。

**💡 创新点**

创新点在于：①将词典级差异、潜在对齐和不确定性统一到同一能量加权OT计划；②通过成本分解揭示哪些潜在向量对差异贡献最大；③推导出基于文档采样的联合置信区间与bootstrap方法，提供理论上一致的置信估计。

**🔧 技术方法**

主要技术包括能量加权的Sinkhorn最优传输、Hilbert空间余弦相似度（解码器、激活、因果三类）、一阶影响函数、联合大样本推断与文档级自助法。

**📊 数据集**

使用了 TinyStories-1M、TinyStories-33M、GPT‑2 小型、Pythia‑160M 等大规模预训练模型的激活数据，并在不同主题（family、animals）以及不同宽度（3072、6144）上训练多种SAE。

**📈 对比分析**

比较方法：计算文档分布下的传输成本 D_ε、分配比例 θ_I,J 并进行联合置信区间检验；实验结果显示：在 TinyStories‑1M 上可显著检出主题差异；低能量潜在向量在评估成本中占比高；不同模型和宽度下成本与权重的作用相互抵消，表明方法能捕获细粒度差异。

**⚠️ 局限性**

局限性包括：置信区间在存在稀有文档时覆盖率下降；结果高度依赖正则化参数和自比较成本；仅对已训练的SAE进行推断，未直接评估训练过程中的不确定性；对能量权重要求所有潜在向量均有正能量。

---

## 118. ProGram: A Grammar for Progressive Computation Pipelines

**arXiv ID:** 2611.06188 | [PDF](https://arxiv.org/pdf/2611.06188v1)

**作者:** Seunghoon Jung `[一作]` (Sungkyunkwan University), Jaemin Jo `[通讯]` (Sungkyunkwan University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种面向进度可视化分析（PVA）的计算流水线语法，旨在通过声明式规范来实现跨系统的可重用进度计算和控制；

**💡 创新点**

创新点包括：1）将进度计算与客户端控制解耦，采用基于查询的声明式语法；2）提供锁步、分叉与临时优先级等机制，实现多查询共享与重用计算；3）构建支持批量和迭代进度的统一执行模型；

**🔧 技术方法**

技术实现上采用JavaScript客户端库配合Python服务器引擎，支持CPU与GPU两层执行；语法基于JSON定义，涵盖过滤、变换、分箱、归约等原语；

**📊 数据集**

使用多种公开数据集进行实验，包含电影数据、犯罪记录、旅行数据、交通计费等，以复现ProReveal和DimXplorer等系统；

**📈 对比分析**

通过将现有系统改写为该语法实现，并在同一引擎下运行，展示与原系统在功能一致的同时简化开发流程；性能方面在实验中未给出具体数值，但强调了通过锁步与分叉可减少I/O与计算重复，GPU加速已在批处理聚合中得到体现；

**⚠️ 局限性**

局限性包括：仅支持CSV数据源，GPU层仅适用于批量可合并聚合；缺乏对领域特定算法（如DBSCAN、最短路径）的原生支持；对迭代型算法的细粒度控制不充分；

---

## 119. Self-Supervised Pretraining for Label-Efficient Facial Expression Recognition using Smart Glasses

**arXiv ID:** 2611.05711 | [PDF](https://arxiv.org/pdf/2611.05711v1)

**作者:** Zhengtao Gao `[一作]` (Università della Svizzera italiana), Martin Gjoreski `[通讯]` (Università della Svizzera italiana)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并验证了在 OMG（光学肌电）智能眼镜上使用自监督预训练（MAE）提升面部表情识别（FER）模型的标签效率，且探讨了不同传感器模态对FER性能的影响。

**💡 创新点**

①首次将 MAE 自监督预训练应用于 OMG 传感器数据；②证明在极少标签（仅 5%）情况下预训练模型已超越完整标签下从零训练的模型；③系统性分析各传感器模态（导航、接近、IMU、欧拉角）对FER的贡献。

**🔧 技术方法**

Vision Transformer（ViT）编码器 + Masked Autoencoder（MAE）预训练；多种数据增强（通道切换、抖动、缩放、幅值变形）；person‑independent 5‑fold 交叉验证；对比从零训练与预训练后微调。

**📊 数据集**

预训练使用无标签 CLUES 数据集（52 名参与者，82.6 小时 10s 采样窗口）；下游 FER 训练使用带标签 OCOGEM 数据集（75 名参与者，含 5 类表情）。

**📈 对比分析**

采用不同标签预算（5%、10%、20%、40%、80%、100%）的 person‑independent CV 评估；在所有预算下预训练模型均优于从零训练；在 5% 标签下预训练模型的宏 F1、准确率、AUC、MCC 均高于完整标签的从零训练模型，随机基线远低于两者。

**⚠️ 局限性**

①表情为受控的、强制性表现，可能不代表真实生活场景，难以直接推广；②仅评估 MAE + ViT，未探讨其他 SSL 或网络结构；③增强参数未调优，未单独评估各增强对下游性能的贡献；④未研究个体化微调与持续学习的实际效果。

---

## 120. Constraint-Aware Evaluation of Reinforcement Learning and Unsupervised Anomaly Detection for Security Operations Alert Triage

**arXiv ID:** 2611.05746 | [PDF](https://arxiv.org/pdf/2611.05746v1)

**作者:** Radityo Ghifiardi `[一作]`, Rifny Rachman `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在生产 SOC 环境下，对强化学习（PPO）、无监督孤立森林（IF）与自编码器（AE）三种警报 triage 模型进行统一实验，提出基于 BEF（Benign Escalation Fraction）约束的 CMDP 框架和元控制器，使用集群级别评估指标。

**💡 创新点**

① 在同一数据集、相同特征契约与统一约束下，对 RL 与无监督模型进行系统对比；② 引入 BEF 约束的 CMDP 与元控制器；③ 发现模型在同一约束下能实现互补覆盖（coverage insurance）；④ 通过奖励设计与特征合同的对照消融，厘清两类模型性能的根本驱动因素；⑤ 对跨租户零射传输进行评估，揭示迁移失败与结构性分离的区别。

**🔧 技术方法**

离线 PPO contextual bandit（两阶段奖励设计）、Isolation Forest（基于手工特征的异常分数）、Autoencoder（重构误差），以及用于评估的 Jaccard 交并集、Meta‑Controller 选取最小 BEF 的策略。

**📊 数据集**

主数据集：559,792 条训练事件与 74,909 条已标注测试事件，来自 100+ 万用户的运营商 SIEM；次级跨租户数据：237,783 条来自另一机构的 SIEM 导出。

**📈 对比分析**

使用集群级别指标（@M=32）比较，三模型均满足 Strict BEF（PPO 0.000，AE 0.007，IF 0.041），覆盖互补性显著；PPO 在 BEF 上明显优于无监督模型；IF 与 AE 依赖手工特征，若仅使用嵌入特征则表现随机；奖励设计是 PPO 成功的关键；跨租户零射传输失败，但结构性覆盖仍保持。

**⚠️ 局限性**

局限性：仅在单一运营商与单一租户数据上验证；特征契约为 7 维手工构造，未测试更高维或学习型特征；PPO 未使用 off‑policy 校正；跨租户迁移仅在模式不匹配下失败；人类评估样本量有限，缺乏统计显著性。

---

## 121. Can LLM Agents Learn to Test? Maintaining Scoped Experience as Testing Evidence for Mobile GUI Testing

**arXiv ID:** 2611.05873 | [PDF](https://arxiv.org/pdf/2611.05873v1)

**作者:** Zimu Liu `[一作]` (Ant Group), Liang Yu `[通讯]` (Ant Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文探讨如何让基于大型语言模型（LLM）的移动 GUI 测试代理在多轮测试中维护、复用并安全废弃测试经验，提出一套基于范围、可追溯性和生命周期的测试证据维护合同，并在实验中验证其对任务完成率和成本的影响。

**💡 创新点**

创新点在于：①提出明确的测试证据维护合同，包括作用范围、证明链和四状态生命周期（Confirmed、Tentative、Suspect、Stale）以及专门的写入角色（Reflector）；②将经验拆分为三层（Episodic、Semantic、Procedural）并给出基于范围、语义相似度与时效的复用评分与晋升阈值；③在实验中区分记忆启用与否、不同层级隐藏以及维护开关的影响，展示经验复用在困难任务集上的显著提升。

**🔧 技术方法**

核心技术包括：大语言模型（如 OpenAI 接口）驱动的自然语言规划与执行；多通道感知（accessibility、OCR、视觉定位）融合；经验检索与评分机制；四状态生命周期管理与写入隔离；以及基于规则的版本与冲突检测。

**📊 数据集**

使用的数据集为九款 Android 应用（Alipay、Baidu Search、Amap、WPS Office、DXY Doctor、Douban、Ctrip、Bilibili、Fossify 套件）共计 158 次实验跑；每个任务为可读的非破坏性目标（如到达天气页、搜索结果页、文档列表等），并提供版本对比（例如 WPS 14.43→14.45）。

**📈 对比分析**

比较方法：在同一骨干（同一 LLM 版本、同一设备、同一脚本）下进行 Full（启用所有层级并开启维护）与 NoMem（禁用经验记忆）以及 SA/SA+Mem 等对照；使用配对显著性检验（Wilcoxon、McNemar）评估通过率、步骤数、耗时等指标。实验结果显示：在困难任务集上 Full 完成 18/18，而 NoMem 仅 3/18；在成本方面 Full 的步骤与时间显著低于 NoMem；但 RQ2 与 RQ4 的结论被视为无效，且整体实验规模有限。

**⚠️ 局限性**

主要限制：①实现并未完全执行合同规定的四状态生命周期和写入隔离；②部分实验（如 RQ2、RQ4）受实验设计与标签依赖限制，无法给出有效结论；③仅使用 read‑only Android 应用，未覆盖 iOS/WebView；④未公开完整的运行日志、截图和 APK，重现性不足；⑤实验规模小、任务多样性有限，无法验证跨应用、跨版本的泛化能力。

---

## 122. DS@GT ARC at CheckThat! 2026: A Verdict-Anchored Pipeline for Generating Fact-Checking Articles

**arXiv ID:** 2611.05382 | [PDF](https://arxiv.org/pdf/2611.05382v1)

**作者:** Hoang Thanh Thanh Truong `[一作]` (Georgia Institute of Technology), Shuyu Tian `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套四阶段的自动化流水线，完成从声明、原始判定和来源文档生成完整事实核查文章的任务。

**💡 创新点**

核心创新在于：① 用原始判定“锚定”生成过程；② 通过严格的证据清洗与聚类提升证据利用率；③ 规则化的引用后处理恢复引用格式；④ 全程使用开源大型语言模型，避免专有API，显著降低成本与提高可复现性。

**🔧 技术方法**

技术手段包括：句子嵌入相似度阈值判定证据可用性；凝聚聚类和句子级别相关性筛选进行检索与聚类；使用 8B 开源 LLM（如 LLaMA2/ChatGLM）在 HuggingFace Transformers 中进行低温采样生成；引用后处理采用正则式规则校正，完成文件名到 URL 的替换。

**📊 数据集**

使用数据集：训练集 WatClaimCheck（3,372 条实例），测试集 ExClaim（1,158 条实例），以及 200 条带官方 baseline 生成结果的开发子集。

**📈 对比分析**

在官方基线（同任务、使用更大专有模型）基础上对比四项指标（蕴含、引用精度、引用召回、证据覆盖）。系统平均得分 0.428，排名第三。显著提升集中在证据覆盖（0.902 vs 0.329）和引用精度/召回；蕴含得分与基线相近。

**⚠️ 局限性**

局限性包括：① 仅提升引用质量和证据覆盖，蕴含（逻辑一致性）与基线相当；② 缺乏主动检验和修正推理的代理机制；③ 阈值与聚类策略手工设定，未进行系统化调优；④ 对极长文档的嵌入仅使用前 256 tokens，可能导致信息缺失；⑤ 未验证更大模型或多代理迭代的潜在收益。

---

## 123. Correct Robots Before They Make Mistakes: Proactive Human-in-the-Loop Intervention via Preference Learning

**arXiv ID:** 2611.05667 | [PDF](https://arxiv.org/pdf/2611.05667v1)

**作者:** Xinkai Jiang `[一作]` (Karlsruhe Institute of Technology), Rudolf Lioutikov `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PHIL框架，通过XR预览机器人预测的动作并在错误发生前由人类干预，收集正负对比数据进行偏好学习，从而改进行动块式策略。

**💡 创新点**

创新点在于：① 采用可视化的XR界面让操作者在物理工作空间中直观预览未来轨迹；② 通过将未执行的预测与后续执行的纠正动作配对，构造同一状态下的正负偏好样本；③ 将偏好学习与传统行为克隆结合，实现高效的后训练。

**🔧 技术方法**

技术手段包括：基于Meta Quest 3的XR渲染、动作块式策略（diffusion policy和预训练的Vision‑Language‑Action VLA），偏好对比学习（pairwise logistic loss），以及人机交互的实时控制切换。

**📊 数据集**

数据集为四个真实机器人操作任务（Pick Up Banana、Towel Folding、Cook the Carrot、Lemon in the Drawer），使用10/20/30条基准演示以及每种条件下5条主动/被动纠正回合收集的数据。

**📈 对比分析**

与基准策略（Base Policy）以及两种人机回合学习方法（HG‑DAgger与Residual Policy）对比，PHIL在所有任务和演示预算下均获得更高的任务成功率（Diffusion策略成功率从44.0%提升到70.3%，VLA从57.8%提升到76.9%），并显著减少不必要的恢复行为。

**⚠️ 局限性**

局限性包括：① 需要XR设备与精确的空间注册；② 在混合主动/被动数据时效果不稳定；③ 仍依赖足够的基准演示来初始化策略；④ 目前仅验证在有限的四个任务上，需进一步验证在更大范围任务中的可扩展性。

---

## 124. How Much Do Modern Embeddings Reveal? Inversion Attacks on Commercial Embedding Models

**arXiv ID:** 2611.05716 | [PDF](https://arxiv.org/pdf/2611.05716v1)

**作者:** Kareem Elaasar `[一作]` (University of California Irvine), Cristina Lopes `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文改进并扩展了 vec2text 逆向嵌入框架，支持 OpenAI 的 text-emb-3-small 与 Mistral 的 mistral-embed 两个商业嵌入 API，并在统一字符长度约束下，对五类文本（通用指令、维基百科段落、医疗相关指令、改写指令和短代码片段）进行逆向实验，评估文本恢复精度。

**💡 创新点**

创新点在于：①将 vec2text 迁移至多种商业嵌入 API，并动态处理不同维度；②在相同字符长度下进行跨域、配对改写对比的细粒度评估，首次展示维度、域和改写对逆向性能的影响；③提出多指标评估（Exact Match、BLEU、ROUGE‑L、BERTScore）并系统报告不同嵌入器在各类文本中的表现。

**🔧 技术方法**

技术主要包括：T5‑base 基础模型的 inverter 与 corrector；MLP 投影映射嵌入至 T5 编码空间；递归纠正步骤（最多 5 步）；API 客户端集成（OpenAI、Mistral）；L2 归一化嵌入；梯度自由推理与贪婪解码。

**📊 数据集**

数据集：MS MARCO passage 训练集用于模型训练；One Million Instructions 用于通用与医疗指令；WikiText‑103 用于维基段落；改写指令由大型语言模型生成并与原指令配对；短代码片段由 Python/JavaScript/SQL 的单行/两行构造组成。

**📈 对比分析**

比较方法：对每个嵌入器在 800 条样本/类、最多 5 步 corrector 的设置下计算 EM、BLEU、ROUGE‑L、BERTScore。结果显示 OpenAI text‑emb‑3‑small 在所有类别中 Exact Match 最高（医疗指令 47.4%，通用指令 24.5%），Mistral 次之，GTR‑base 最低；BERTScore 在 prose 类别均 ≥87，代码类最低；改写指令相较原指令 EM 降低；短代码类 EM 最低。

**⚠️ 局限性**

局限性：①仅评估了 vec2text 框架，未对比其他逆向攻击或基线；②单一随机种子，无置信区间或显著性检验；③训练/推理时间较长，仅在 5×RTX 6000 GPU 上完成；④仅测试英文文本；⑤未探究嵌入维度、训练目标或 API 特性对结果的因果关系；⑥未评估实际隐私属性恢复（如实体、数字、敏感信息）。

---

## 125. Exploring the Privacy--Size Tradeoff for Differentially Private Self-Attentive Variational Autoencoders

**arXiv ID:** 2611.05582 | [PDF](https://arxiv.org/pdf/2611.05582v1)

**作者:** Rohit Sadavarte `[一作]` (Foresight, Intuit), Eric Van Oeveren `[通讯]` (Foresight, Intuit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为 SAVANT 的差分隐私合成表格数据生成器，并与两种边际机制在不同数据量和隐私预算下进行基准对比。

**💡 创新点**

创新点在于首次将自注意力机制、变分自编码器与 DP‑SGD 结合，构建可训练的自注意力 DP VAE。

**🔧 技术方法**

使用了自注意力层、变分自编码器、DP‑SGD 随机梯度下降、Opacus 进行隐私预算控制，以及对比的 PrivBayes 与 MST 边际机制。

**📊 数据集**

实验使用了 UCI Adult 和 ACS‑PUMS 2018 两个公开数据集，并在 10K 到 500K 行的多层次样本规模下评估。

**📈 对比分析**

通过在同一 held‑out 样本上训练真实与合成数据的逻辑回归，测量 F1 下降、边际误差与相关性差异；结果显示随着数据量增加，SAVANT 在 ε≥1 时能显著降低 F1 下降，且在大数据和宽松预算下优于传统边际机制。

**⚠️ 局限性**

局限性包括仅在两组公开数据上验证，DP‑SGD 在极低 ε（0.1）下训练不稳定，对抗性隐私测试仅采用弱的成员推断攻击，且合成模型计算成本高于边际机制。

---

## 126. EnzymeRSI: A dual-loop recursive self-improvement system for enzyme design

**arXiv ID:** 2611.06321 | [PDF](https://arxiv.org/pdf/2611.06321v1)

**作者:** Genoria AI Team `[一作]` (Genoria AI), Meng Yang `[通讯]` (Genoria AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 EnzymeRSI，一个双环递归自我改进（RSI）系统，用于自主酶工程，包含外部 harness 演化和内部算法改进，并与云实验室闭环联动。

**💡 创新点**

创新点在于：①把研究工作流程（harness）和算法本身视为可自我改进的对象；②通过外部循环演化 harness 的版本、工具调用和流程，内部循环在方法瓶颈出现时自动迭代模型、搜索和评估；③将实验反馈以 QC 标记形式返回，形成完整的 dry‑to‑wet 循环；④在多种酶实验（RLuc、A3A、KOD‑E）中验证决策与排名的差距与方法改进。

**🔧 技术方法**

使用的技术包括：大型语言模型（如 QwenCode、ESM‑2）、ProteinMPNN、AlphaFold 等预测工具；强化学习和主动学习的候选生成/采样框架；云实验室调度与 ProtoPilot/Operon‑S1 的实验执行；以及基于实验证据的版本管理和评价机制。

**📊 数据集**

使用的数据集包括：ProteinGym（大规模突变效应测量）、TadA‑Bench 的氨基酸视角、ThermoMPNN 的稳定性记录，以及实验测定的 RLuc、A3A 双底物转换和 KOD‑E 的 Cy3‑dATP 嵌合效率等。

**📈 对比分析**

与十个公开框架（OpenRSI、MLEvolve、AgenticEvolve 等）对比，外部循环在五类评测中平均提升 Spearman 相关 0.044（如 CD19 由 0.314 提升到 0.335）；内部循环在 ThermoMPNN 上 PCC 提升到 0.849（高于基线 0.804），在 TadA‑Bench 上 Spearman 提升到 0.307；在实验中，系统推荐的 RLuc 单点突变中 10/32 新变体活性至少是野生型两倍，A3A 的 5mC 转化率提升至 90.2%，KOD‑E 的催化效率提升 31.1 倍。

**⚠️ 局限性**

局限包括：①外部循环仅在 development set 上提升，未验证跨 assay 的普适性；②内部循环未针对 RLuc 的排名与测量相关性差距做改进；③实验验证仅为单轮闭环，未完成完整多轮 PANCE 演化；④需要在更多生物实验和多种酶类上进一步测试鲁棒性。

---

## 127. Diversity-Aware Experience Replay via Statistical Leverage Scores

**arXiv ID:** 2611.05929 | [PDF](https://arxiv.org/pdf/2611.05929v1)

**作者:** Shaifalee Saxena `[一作]` (Los Alamos National Laboratory), Alexander Scheinker `[通讯]` (Los Alamos National Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于统计杠杆（leverage）的经验回放框架，用来在离线或在线 RL 中重采样稀有且重要的转移，从而提升稀缺情景下的学习效率。

**💡 创新点**

创新点在于：① 用缓冲区自身的几何结构——杠杆得分（Mahalanobis 距离）——作为无标签、无奖励、无 critic 的优先级；② 通过峰度（kurtosis）挑选特征列，避免一热动作编码导致的方差主导问题；③ 证明了在高维高斯混合缓冲区中，杠杆采样的稀有放大受几何分离度限制而非单纯稀有度；④ 在窗口级别引入目标相对描述，进一步提升轨迹稀有事件的重采样。

**🔧 技术方法**

主要技术包括：统计杠杆得分计算、峰度特征选择、混合采样策略（β‑leverage）、窗口级 Mahalanobis 评分、离线 Q‑学习与 TD3+BC 训练。

**📊 数据集**

实验数据集：① 24×12 二维网格世界的轨迹日志；② 三连杆机械臂在障碍物周围推球的日志（950 条直线轨迹 + 35 条轨迹围绕障碍的轨迹）。

**📈 对比分析**

比较方法包括：均匀采样、PER、Oracle（50/50 采样）、β‑leverage、尾部平滑采样。结果显示：在网格世界中，β‑leverage (β=0.5) 的成功率约 0.94，RMSE 下降；在机械臂实验中，窗口尾部平滑采样将轨迹成功率从 0.49 提升至 0.69，接近 Oracle 的 0.74。

**⚠️ 局限性**

局限性包括：① 仅对单个特征列或窗口级描述有效，无法自动适应所有稀有模式；② 需要手动选择特征列或窗口参数；③ 在高维、非高斯分布的数据中，杠杆得分可能失去效果；④ 过度放大稀有样本可能导致训练不稳定。

---

## 128. What Do We Mean When We Say 'F*ck AI'?: Moral Panic, Semantic Indeterminacy, and the Reification of Techno-Capitalism

**arXiv ID:** 2611.05574 | [PDF](https://arxiv.org/pdf/2611.05574v1)

**作者:** Sasha Luccioni `[一作]` (Sustainable AI Group), Travis LaCroix `[通讯]` (Durham University)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5049415541)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析 AI 术语语义不确定性与社会恐慌的相互作用，探讨其如何被技术资本主义再现并导致政策失效

**💡 创新点**

首次系统性地将语义不确定性与道德恐慌相结合，揭示企业利用此机制强化技术霸权的具体逻辑

**🔧 技术方法**

采用批判性话语分析、文献综述与案例研究等质性方法

**📊 数据集**

无实验数据集，主要引用公开报告、新闻稿、学术文献及相关案例资料

**📈 对比分析**

对比不同学术观点与案例，结论表明语义模糊导致政策制定失效与公共误解，但未进行量化性能评估

**⚠️ 局限性**

研究主要基于二手资料，缺乏经验验证，范围受限于已公开文本，可能忽视非主流或灰色领域的视角

---

## 129. RIDGE: State-Conditioned Reward Blending for Behavioral Coverage in Deep RL Game Agents

**arXiv ID:** 2611.05601 | [PDF](https://arxiv.org/pdf/2611.05601v1)

**作者:** AL Muqshith Mohammed Shifan `[一作]` (Ontario Tech University), Cristiano Politowski `[通讯]` (Ontario Tech University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在深度强化学习游戏代理中使用状态条件化奖励混合，使单一PPO代理能够在同一episode内根据状态动态切换四种不同人格，实现多样化行为覆盖。

**💡 创新点**

创新点在于将平滑的状态条件化混合器与多头价值网络结合，解决奖励非平稳导致的PPO稳定性问题，并通过单代理完成跨人格任务。

**🔧 技术方法**

使用技术包括PPO强化学习框架、状态条件化sigmoid/softmax权重混合、每人格独立的价值头、GAE优势估计与梯度裁剪等。

**📊 数据集**

实验数据集为Crafter游戏环境，采用标准1M步训练预算，并评估其22项成就。

**📈 对比分析**

与四个固定人格基线以及不同锐度α的ablation进行对比；单代理在木材等级分支（Pickaxe/Sword）上同时取得约0.32/0.26分，明显优于任何单人格基线，且状态条件化版本比无状态条件的均匀混合提升30‑50%。

**⚠️ 局限性**

局限性包括仅在Crafter实验，种子数量有限（3个），未进行大规模多seed评估；未验证硬切换极限（α→∞）与多头价值网络单独效果，且未与成就蒸馏方法在同一度量上进行对比。

---

## 130. Reflection-Robust Stereo Visual-Inertial Odometry via Kalman-Filtered Ground-Height Estimation

**arXiv ID:** 2611.05938 | [PDF](https://arxiv.org/pdf/2611.05938v1)

**作者:** Uihyun Hwang `[一作]` (Korea Advanced Institute of Science and Technology), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种轻量级的反射感知前端，用于特征基立体视觉惯性里程计（VIO），通过实时估计地面高度并拒绝仅在地面下方的虚拟特征，从而提升在反射性室内地板环境下的轨迹精度。

**💡 创新点**

创新点包括：① 用单变量卡尔曼滤波器结合 NIS 门控对地面高度进行实时平滑估计；② 采用基于不确定性的安全边距来选择性剔除地面下方的虚拟特征；③ 利用 GCNet 生成的地面掩模仅在反射易发区域做几何验证，避免过度剔除有效特征。

**🔧 技术方法**

主要技术手段包括：GCNet（用于地面区域分割）、立体匹配与三角化、重力方向上的高度投影、基于像素/姿态误差的测量不确定性建模、标量卡尔曼滤波与 NIS 门控、以及不确定性加权的特征拒绝阈值。

**📊 数据集**

使用了自采集的室内反射性地板数据集：KAIST 四个楼层（E4、E11、N3、N7）和延世大学三层（Yonsei 1–3），并对 N7 进行手持录制，以检验在大幅姿态变化下的鲁棒性。

**📈 对比分析**

与 VINS‑Fusion 与 DynaVINS 的原始前端以及仅剔除整个地面区域的 naive floor rejection 进行比较。实验结果显示，在 8 条序列中，改进后方法在 6 条（VINS‑Fusion）和 4 条（DynaVINS）上均能显著降低 ATE RMSE（平均提升约 30%），在地面高度估计误差上保持 <0.05 m 的稳定性；在高度变化序列 N7 上能够及时重置估计以适应楼梯高度变化。

**⚠️ 局限性**

局限性包括：① 在光照暗淡、地面纹理稀疏的走廊场景（如 E11）中，地面掩模可靠性下降，导致卡尔曼滤波更多依赖预测，可能产生误差堆积；② 对于高度极高的镜面地板（如 N7）时，部分镜像虚拟特征可能仍提供有用约束，单纯拒绝可能略逊于 naive 方法；③ 目前仅基于单一地面高度假设，无法处理多平面或倾斜地面；④ GCNet 的分割性能受低纹理区域影响，未来需要更鲁棒的语义分割方案。

---

## 131. Neural networks and games

**arXiv ID:** 2611.05706 | [PDF](https://arxiv.org/pdf/2611.05706v1)

**作者:** Philip R. Neary `[一作]` `[通讯]` (Royal Holloway University of London), Philip R. Neary (Royal Holloway University of London)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过严谨的数学推导，建立了对称二进制 Hopfield 网络与二进制行动、具有二次势函数的精确势游戏之间的完全等价关系，并将网络中的稳定配置对应到纳什均衡，异步神经更新映射到最佳反应动态，进一步将记忆存储与机制设计、记忆检索与均衡选择等问题重新诠释。

**💡 创新点**

创新点在于首次明示这一等价关系，将神经网络与博弈论的核心对象用同一数学语言统一；将记忆存储问题视为机制设计问题；把“spurious memories”解释为无意的纳什均衡；以及提出针对性均衡选择（检索）框架，揭示记忆检索概率与哈密顿距离的区别。

**🔧 技术方法**

主要技术手段为符号推导与等价证明：利用局部场与效用函数的对应关系，证明能量函数的负值即为势函数；分析异步最佳响应动态与能量下降的对应性；以及通过线性规划形式化存储设计问题。还用了一般的博弈理论工具，如精确势函数、纳什均衡性质以及潜在最大化选择准则。

**📊 数据集**

本研究为纯理论性工作，未使用任何实验数据集；所有结论均来自数学证明与理论分析。

**📈 对比分析**

由于本工作主要是理论推导，并未与具体算法或实验结果进行对比；因此不涉及性能数值评估或方法比较。

**⚠️ 局限性**

局限性包括：仅适用于对称二进制 Hopfield 网络与二次势函数的游戏；未考虑异步更新过程中的噪声或随机性对收敛的影响；设计问题在大规模网络下的计算复杂度仍未解决；且等价映射虽提供概念桥梁，但在实际应用中实现机制设计与均衡选择仍面临多重均衡与不可预见的“spurious memories”问题。

---

## 132. Local information processing capacity: anatomy of computation in neuromorphic systems

**arXiv ID:** 2611.05959 | [PDF](https://arxiv.org/pdf/2611.05959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 133. ProcessPrefBench: Evaluating Cross-Task Transfer of Individual Process Preferences from Naturalistic Human-AI Interaction

**arXiv ID:** 2611.06065 | [PDF](https://arxiv.org/pdf/2611.06065v1)

**作者:** Yiwen Yang `[一作]` (Independent Researcher), Guanming Liu `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了ProcessPrefBench基准，用于测量自然人机对话中可转移的流程偏好；

**💡 创新点**

提出跨任务源–目标案例构造、目标特定评分表与任务失效门控，探索历史记忆与偏好迁移的权衡；

**🔧 技术方法**

对三种大型语言模型（DeepSeek V4 Flash、Qwen‑Max、GPT‑4o）在目标、完整历史、强制或条件提炼偏好提示下生成答案，并使用Claude Sonnet进行评判；

**📊 数据集**

以两位知识工作者在Margin产品中三个月的真实对话为基础，构建66个源–目标案例，包含30条纠正来源与59条目标任务；

**📈 对比分析**

通过比较目标仅提示、完整历史和提炼偏好提示的偏好得分与任务失败率，发现历史记忆提升偏好实现0.2–0.24点，但导致虚假证据率上升约4–5个百分点，整体性能仍不稳定；

**⚠️ 局限性**

受限于仅两名用户、单一产品、有限任务类型以及诊断性评审，无法推广到更广泛人群，评审与制造证据差异的可靠性也有限。

---

## 134. Explainable Anomaly Detection in 5G Networks through Continuous Time Graphs

**arXiv ID:** 2611.06225 | [PDF](https://arxiv.org/pdf/2611.06225v1)

**作者:** Thomas Hoger `[一作]` (LAAS - CNRS, Université de Toulouse), Philippe Owezarski `[通讯]` (LAAS - CNRS, Université de Toulouse)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将5G控制平面流量建模为连续时间动态图（CTDG），并利用该图表示同时捕捉语义、序列和拓扑三维特征，进行有监督异常检测。

**💡 创新点**

创新点在于：①统一的图形式既能保留包内字段细粒度，又能显式记录包间时间与拓扑关系；②通过共享特征节点实现跨包语义关联；③直接用图神经网络进行链接预测，无需单独CNN+RNN模块，提升可解释性。

**🔧 技术方法**

采用了多种现有CTG神经网络（TGN、GraphMixer、DyGFormer、CAWN、TCL、JODIE、DyRep）进行有监督链接预测；使用Scapy抽取字段、DyGLib统一训练框架。

**📊 数据集**

构建了自定义4小时5G控制平面数据集，包含约1.2M条正常消息和130K条多维攻击（PFCP/ GTP-U/HTTP2）记录，攻击与正常流量按时间交织。

**📈 对比分析**

在AUC上比较各模型，TGN最佳（0.9517），GraphMixer次之（0.7411）。吞吐量评估显示TGN约22.8 pkt/s，接近数据生成速率（29 pkt/s），其他模型吞吐更快但性能低。

**⚠️ 局限性**

主要限制：①当前模型仍未达到实时处理阈值（TGN略慢）；②依赖有监督训练，难以检测完全未知攻击；③数据集规模有限，未覆盖所有5G协议和实体；④DyGLib实现效率不足，可进一步优化。

---

## 135. Ray-Tracing-Based Dynamic Channel Sounding With Autoregressive Wi-Fi CSI Prediction

**arXiv ID:** 2611.05970 | [PDF](https://arxiv.org/pdf/2611.05970v1)

**作者:** Lyutianyang Zhang `[一作]` (Chongqing University), Shuguang Cui `[通讯]` (Chinese University of Hong Kong (Shenzhen))

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于自回归预测的Wi‑Fi MIMO信道测量框架，在每个声波周期内使用Transformer结合RoPE和Huber损失对多子载波CSI进行预测，并通过SINR下界约束动态选择声波间隔。

**💡 创新点**

创新点在于将光线追踪生成的真实时变信道与自回归Transformer预测结合，并通过误差下界推导出SINR安全边界，进而全局最优求解声波间隔，从而在保持吞吐量的同时显著降低测量开销。

**🔧 技术方法**

所用技术包括Sionna光线追踪仿真、基于RoPE的Transformer自回归预测、Huber损失训练、SVD预编码、SINR下界分析和累计求和算法实现声波间隔优化。

**📊 数据集**

实验使用Sionna生成的室内场景，包含10×10×3.2 m³的房间，AP与移动STA沿直线、圆形和余弦轨迹移动，采样间隔0.1 ms，共生成242个子载波的CSI张量。

**📈 对比分析**

与传统的即时CSI、1 ms延迟CSI以及仅使用延迟CSI的基线相比，RoPE+Huber自回归模型在NMSE上降至4.27×10⁻⁵，且在单STA Wi‑Fi 7仿真中预测CSI的平均良通量分别比旧CSI提升了67.3%、112.7%和119.1%。

**⚠️ 局限性**

局限性包括自回归预测误差随预测步数累积导致的性能衰减，需要设定有限的预测 horizon；仅在单用户场景验证，未考虑多用户或更高阶Wi‑Fi版本的扩展；以及对计算与能耗的实际硬件实现仍需进一步评估。

---

## 136. Mechanical Engineering Skills: An Open Library of Agent Skills for Design, Analysis and Manufacturing

**arXiv ID:** 2611.05758 | [PDF](https://arxiv.org/pdf/2611.05758v1)

**作者:** Achuth Chandrasekhar `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含33个可直接通过语言模型调用的机械工程“技能”库，涵盖设计、分析、制造及教学，支持文本或语音交互并自动执行对应工具。

**💡 创新点**

创新点在于将机械工程研究方法包装成可动态加载的“技能”格式，内置执行脚本和可验证的检查，解决工具碎片化与模型误差缺乏证据的问题。

**🔧 技术方法**

技术上结合了大型语言模型（Claude）、Python脚本、CAD/CFD/FEA/机器学习工具（如OpenFOAM、CadQuery、TensorFlow等）以及模型验证与一致性检查。

**📊 数据集**

使用了多种数据集，包括自研的TPMS/晶格网格、OpenRocket飞行模拟数据、ThermoPore的热像/CT配准数据、SLS试验数据、LLM‑SR的应变–温度实验数据，并提供合成演练数据用于验证。

**📈 对比分析**

通过五个案例（晶格逆向设计、打印火箭、热像孔隙检测、SLS过程优化、方程发现）对比原论文结果或参考值，精度分别在0.1%–4%范围内，验证了方法的可复现性和可用性。

**⚠️ 局限性**

局限性包括仅验证了可执行组件，未完成完整代理可靠性评估；依赖外部软件/硬件和许可证；合成数据演练无法保证在真实硬件上的精度；安全关键决策仍需人工审查。

---

## 137. A Systematic Benchmark for Evaluating Multimodal Personalized Reasoning of Vision-Language Models

**arXiv ID:** 2611.06259 | [PDF](https://arxiv.org/pdf/2611.06259v1)

**作者:** Dohyun Kim `[一作]` (University of Seoul), Kwanyong Park `[通讯]` (University of Seoul)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MPRBench，一个包含 12 个子任务的多模态个性化推理基准，用于全面评估 MLLM 在真实场景中的推理能力。

**💡 创新点**

①将个性化推理拆分为视觉特定、文本特定和多模态三类任务；②设计可解释的多选答案结构，能够系统地定位模型失效模式；③引入跨模态干扰和个性化识别难点，逼近真实应用。

**🔧 技术方法**

采用多模态大型语言模型（MLLM）作为评估对象，使用提示式、特殊 token 学习、检索式、记忆增强等多种个性化方法；生成任务与答案使用 Gemini‑2.5‑Pro 等先进生成模型；评估以准确率为主。

**📊 数据集**

基于公开的个性化图像数据集（如 Yo’LLaVA、ConConChi、RPC 等）以及手工构造的结构化文本属性，构建了 12 个子任务的完整数据集。

**📈 对比分析**

在 MPRBench 上对 LLaVA+Prompt、Yo’LLaVA、RAP、R2P、TAME、PersonaVLM 等多种方法进行对比，发现所有模型的平均准确率仍远低于完美，提示与检索方法在推理任务中表现不佳，定位错误是主瓶颈。

**⚠️ 局限性**

现有 MLLM 在个性化识别、跨模态干扰、细粒度视觉细节处理和多跳推理等方面表现不佳，整体性能不足，缺乏针对这些挑战的有效技术。

---

## 138. Autonomy and Coordination in Intelligent Manufacturing: A Recursive Autonomous Agent Framework

**arXiv ID:** 2611.05828 | [PDF](https://arxiv.org/pdf/2611.05828v1)

**作者:** Yuning Chen `[一作]` (Fudan University), Haining Xu `[通讯]` (Jiangsu Chuangling Intelligent Technology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了基于复杂适应系统理论的Recursive Autonomous Agent (RAA) 框架，用于将工厂生产过程分层管理，实现自主决策与集体协调。

**💡 创新点**

创新点在于将自主代理与递归组织结构结合，使用目标契约与能力契约实现自适应决策、局部闭环反馈及层级间的可组合接口。

**🔧 技术方法**

采用多模态感知、基于规则与模型的决策逻辑、约束检查引擎、分层协同协议与受控学习机制。

**📊 数据集**

未使用公开数据集，主要基于理论建模和系统设计；若有实验则采用模拟工厂场景。

**📈 对比分析**

论文未给出实验对比或性能指标；若有可比实验则通过仿真验证了局部反馈与协调一致性上的提升。

**⚠️ 局限性**

局限性包括缺乏真实工厂验证、对大规模系统的可扩展性分析不足、以及在高动态环境下学习更新机制的鲁棒性待评估。

---

## 139. Energy-Efficient Reinforcement Learning for Large Language Models

**arXiv ID:** 2611.05748 | [PDF](https://arxiv.org/pdf/2611.05748v1)

**作者:** Rahul Sharma `[一作]` (NVIDIA), Yanxin Zhang `[通讯]` (NVIDIA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 EE-RL，一种在大语言模型强化学习后训练过程中在线动态调节生成与训练阶段 GPU 频率的控制器，旨在降低每个有效训练 token 的能耗同时保持吞吐量。

**💡 创新点**

创新点在于：①利用基于 token 计数的治理器预测模型公平评估不同频率设置；②对同步、异步以及分离式执行模式进行阶段感知的频率搜索与实时调整；③在保证吞吐量阈值（如 98% 以内）下自动选择最节能配置，并持续监控以适应工作负载漂移。

**🔧 技术方法**

技术细节包括：GPU 动态电压频率调节（DVFS）、NVML 能耗采样、GRPO 强化学习框架、基于 log(1+x) 标准化的预测回归、试探式短周期测试、吞吐量与能耗监测回路以及多模式搜索算法。

**📊 数据集**

实验使用 Qwen3-30B-A3B 与 Mistral-NeMo 12B 两个大型模型，在 NVIDIA H100 与 GB200 GPU 上进行强化学习后训练，涵盖共 8 种同步/异步执行组合。

**📈 对比分析**

与默认最高频率治理器和离线频率搜索基线相比，EE-RL 在同步/异步配置下分别实现了最高 16.9% 的能耗（每有效 token）降低，吞吐量下降不超过 2%；在离线频率搜索的基础上额外获得约 5%–10% 的节能提升。

**⚠️ 局限性**

局限性包括：对工作负载、GPU 分配和池子平衡高度依赖；需要在每次资源重配后重新搜索；实验仅覆盖 GRPO，未验证其他 RL 算法；多种种子实验不足，可能掩盖学习质量差异；能源测量噪声导致锁定延迟；未对空闲 GPU 的能耗单独评估。

---

## 140. Holistic Parallel Supervision for Expressive Speech to Speech Translation

**arXiv ID:** 2611.06010 | [PDF](https://arxiv.org/pdf/2611.06010v1)

**作者:** Hao Zhang `[一作]` (Johns Hopkins University), Philipp Koehn `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过专业配音与多语种语音转换相结合，构造了包含内容、韵律、时序和说话人身份四维度完整对应的平行语音数据，并用该数据训练端到端的 DirectS2ST 模型，实现表达式语音对语音翻译。

**💡 创新点**

①首次系统性验证“整体平行监督”在表达式 S2ST 上的有效性；②提出基于专业配音+语音转换的 VC‑Dub 数据构造策略；③设计 DirectS2ST 能直接从源语音预测完整目标语音表示的端到端架构。

**🔧 技术方法**

专业配音处理、语音增强、声源分离、Mimi 码流编解码、Wav2Vec‑BERT 2.0、CAMPPlus 说话人编码器、Transformer 的时间与深度解码器、自动和人工评测指标（ASR‑BLEU、BLASER、AutoPCP、STOPES、NISQA‑TTS 等）。

**📊 数据集**

来自娱乐电视节目的专业配音（原声德语/西班牙语及其对应英文配音）和公开的 CVSS‑T 语音对语音数据集；评测使用 mExpresso 多语言表达式数据集。

**📈 对比分析**

与 TransVIP、Dub‑S2ST 以及基线 CVSS‑T 进行对比；实验显示 VC‑Dub 使得所有系统在内容、韵律、时序、说话人相似度和自然度指标均提升，DirectS2ST 在大多数指标（尤其是 AutoPCP、Pause、MOS）上实现了最大的提升。

**⚠️ 局限性**

数据受限于配音资源和 VC 算法的质量；实验仅覆盖英‑德和英‑西两条语言对，无法完全验证跨语言泛化；对话场景与娱乐片段的表现可能不适用于其他语域；同时对人类听众评价仍需更多样本和多语言验证。

---

## 141. LumiTac: Compact, High-Rate Six-Axis Optical Tactile Sensing for Human-Robot Interaction

**arXiv ID:** 2611.05936 | [PDF](https://arxiv.org/pdf/2611.05936v1)

**作者:** Hao Wu `[一作]` (National University of Singapore), Jianshu Zhou `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一款低压薄型光学触觉传感器 LumiTac，尺寸为12 mm直径、4.75 mm厚、质量<2 g，利用20通道 RGBW 光学信号通过极简多层感知机实时估计六轴接触力与力矩。

**💡 创新点**

创新点在于：①光学层压结构将 LED 发射器与 RGBW 接收器对置，利用可变光路与光谱编码实现空间分布光学感知；②极简 MLP 回归模型实现约300 Hz 的高频、低延迟（0.722 ms）六轴估计；③尺寸与成本远低于现有同类传感器，且兼容机器人指尖、把手与可穿戴设备。

**🔧 技术方法**

采用了分层硅胶光学介质、RGBW 光电接收器、LED 发射器、I²C 多路复用、光学归一化、标准化的多层感知机回归、以及基于虚拟质动力学的可 admittance 控制框架。

**📊 数据集**

使用了由6轴 F/T 传感器提供的标定数据，收集在不同负载条件下的20通道光学信号与对应的六轴力矩配对数据；训练/测试采用 40 ms 窗口的离线预处理与实时移动平均窗口。

**📈 对比分析**

与表中列出的 LiVec finger、El‑Azizi、F‑TOUCH、Mao 等多轴传感器比较，LumiTac 在尺寸、质量和成本极低的前提下，实现±10 N、±100 mN·m 范围内的 MAE 分别为 0.054–0.190 N、0.304–1.419 mN·m，R²≥0.965，采样率≈300 Hz，平均延迟0.722 ms；实时实验中能精确跟踪持续和瞬态负载，显示出良好的动态响应。

**⚠️ 局限性**

受硅胶弹性与光路耦合的非线性影响，需要在每个工况下进行专门标定；对极端高动态或强震荡负载响应尚未验证；在强外光环境下光学信号可能受抑制，需要光遮蔽；以及目前仅验证了标准静态/手动负载，未针对多工件或高频振动场景进行鲁棒性评估。

---

## 142. OdorSpaces: Visual-Olfactory Embodied Navigation in 3D Environments

**arXiv ID:** 2611.05738 | [PDF](https://arxiv.org/pdf/2611.05738v1)

**作者:** Changyeon Lee `[一作]` (University of Maryland), Ruohan Gao `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OdorSpaces，一个基于物理仿真的视觉-嗅觉化身导航平台，并给出了两种源定位任务（不可见气体泄漏与早期火灾定位），同时设计了多模态强化学习策略；

**💡 创新点**

创新点在于将Fire Dynamics Simulator与Habitat‑Sim深度集成实现三维物理浓度场；构建首个视觉‑嗅觉化身导航基准与对应奖励机制；利用多模态记忆网络学习化学梯度与视觉导航的协同；

**🔧 技术方法**

采用物理 CFD (FDS)、三维渲染与导航仿真平台 Habitat‑Sim；使用 CNN/MLP 编码视觉、嗅觉与姿态信息，结合 LSTM 记忆；训练策略使用 PPO 强化学习；

**📊 数据集**

使用 Replica（17 场景）与 Matterport3D（25 场景）的真实室内环境，生成对应的气味浓度场；

**📈 对比分析**

与三种规则基线（化学趋向、类昆虫追踪、时间梯度搜索）以及学习基线 DRT‑RL 进行对比，采用 SPL 与最终距离作为评估指标；实验表明学习策略在两种情景下均显著优于基线，SPL 最高、最终距离最低，尤其在火灾场景中优势更明显；

**⚠️ 局限性**

局限性在于使用预计算的静态气味场，忽略机器人运动产生的气流、通风、传感器噪声与响应延迟；仅关注检测后定位而非完整 plume 寻找；仅支持两种气味来源，缺乏更复杂的动态气流与传感器配置。

---

## 143. Self-Supervised Learning of Universal Latent Spaces for Operator Learning

**arXiv ID:** 2611.05735 | [PDF](https://arxiv.org/pdf/2611.05735v1)

**作者:** Sepehr Mousavi `[一作]` (ETH Zurich), Laura De Lorenzis `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种将表示学习与算子学习解耦的框架，使用无监督合成函数预训练功能自动编码器得到通用潜在空间，并在该空间上冻结表示，缓存输入输出，训练算子，显著提升内存与训练效率；

**💡 创新点**

1) 通过无标签合成函数预训练获得任务无关的非线性潜在空间；2) 引入 PartNorm 实现位置、尺度、幅度、可选旋转的等变性；3) 冻结表示实现输入/输出缓存，解耦解析度，极大降低训练成本；4) 对潜在空间进行几何解释（距离、Geodesic、编辑方向等）；

**🔧 技术方法**

功能自动编码器（FAE）+ PartNorm + 领域分解 + RMSNorm + 近似正交更新 + Transformer 处理器 + 潜在 roll‑out + 物理损失冷却；

**📊 数据集**

28 个多物理任务（流体动力学、固体力学、声学、Helmholtz、Poisson 等）以及自然图像数据，用不同几何、离散化和模态的数据集；

**📈 对比分析**

与 GAOT 与 RIGNO‑12 两个 state‑of‑the‑art end‑to‑end 框架对比；在 28 个任务中 23 个任务准确率高于 GAOT，整体训练时间平均比 GAOT 快 2.1 倍；内存上可支持 4.2M 点、101M 参数的处理器，而 GAOT 仅能处理 5M 点、1M 参数；推理速度提升 10 倍，latent roll‑out 10 倍快；

**⚠️ 局限性**

1) 高频特征需要足够细的分区，受采样点数限制；2) 仅支持可选旋转等变性；3) 当前仅实现 2D backbone，3D 扩展尚未完成；4) 在部分任务上 GAOT 仍略优；5) 对极高频或超大分辨率的处理仍受限；

---

## 144. Attraction-Ranking Mismatch in Out-of-Distribution Detection: Theory and Evidence from Hamiltonian Feature Dynamics

**arXiv ID:** 2611.06021 | [PDF](https://arxiv.org/pdf/2611.06021v1)

**作者:** Zilin Wang `[一作]` (Jinan University), Lianghai Xiao `[通讯]` (Jinan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过构造基于类条件哈密顿动力学的测试环境，探讨将特征吸引到训练分布(ID)内是否能提升对异常分布(OOD)的检测效果。

**💡 创新点**

创新点在于揭示了“吸引-排名不匹配”，即提升类内亲和分数不必然改善ID与OOD样本的相对排序，并给出了精确AUROC分解与边界分析，说明排名边界的相对分数变化是影响AUROC的关键因素。

**🔧 技术方法**

主要技术包括：类条件哈密顿动力学模拟、AUROC分解与局部边界分析、相对分数对比、零步分数与静态质心族（包含CTM检测器）的联系，以及对相似度对齐规则的实验评估。

**📊 数据集**

实验采用常见的OOD检测基准数据集（如CIFAR‑10/CIFAR‑100 作为ID，SVHN、TinyImageNet等作为OOD），并在公开实现代码库中进行验证。

**📈 对比分析**

与现有方法（如CTM、核密度估计等）对比，得分对齐规则在源验证集上能提升性能，但在迁移后往往表现出性能下降，表明单纯提升类亲和分数不足以保证稳健的OOD检测。

**⚠️ 局限性**

局限性包括：实验仍主要集中在图像分类数据集，未验证对其他任务（如文本、语音）的泛化；理论分析假设了正则条件，实际应用中可能受数据分布复杂度影响；并且缺乏对大规模模型的深入评估。

---

## 145. Softmax Attention with Any Temperature in Fixed Dimension: Matching the Best Known Maximum-Inner-Product Bounds

**arXiv ID:** 2611.06055 | [PDF](https://arxiv.org/pdf/2611.06055v1)

**作者:** Zixuan Huang `[一作]` (Peking University), Guozhang Chen `[通讯]` (Peking University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

论文研究了软max注意力机制的位复杂度，给出了在固定维度 d 下对 m 个查询和 n 个键值对的加性误差 η 的期望运行时间为 F_d(m,n)=((mn)^⌊d/2⌋/(⌊d/2⌋+1)+m+n)。该算法与已知的批量最大内积（Max-IP）算法在复杂度上完全匹配，并提供了一个维度保持的归约，将 d 维 Max-IP 直接转化为一次 d 维注意力调用。论文还给出了二维情况的确定性线性时间算法，并构建了相应的在线查询数据结构。

**💡 创新点**

创新点包括：
- 在任意固定维度下，首次实现了注意力与批量 Max-IP 在相同的时间上限（F_d）相匹配。
- 通过温度插值和“共同锚点块”分解，提出了一种维度递归框架，将 d 维问题降低到 d-1 维。
- 提供了一个单次注意力调用即可完成整数 Max-IP 的维度保持归约，突破了此前需要多次调用且维度提升的限制。
- 设计了可在线查询的二维数据结构，实现了预处理后单查询 O(1) 的时间。

**🔧 技术方法**

使用的技术主要包括：
- 软max注意力的移位核与指数插值（Lagrange 插值）
- 多维几何分解：半空间范围计数、凸多面体的上包络、细胞分割和冲突集合分析
- 随机前缀链与轻重递归的采样与划分策略
- 基于 Fenwick 树和持久化段树的多项式时刻求和
- 低维核（d-1）问题的拉伸与锚定
- 误差分析与位复杂度控制（polylog 依赖）

**📊 数据集**

无数据集，全部为理论算法与理论复杂度分析。

**📈 对比分析**

与之前的最优上界 n^2-1/d 的注意力算法相比，本文给出的上界为 n^2-2/(⌊d/2⌋+1)，在 d≥2 时更好，且与已知的批量 Max-IP 上界完全一致。在二维情形下实现了 O(m+n) 的确定性线性时间；在三维为 O((mn)^{1}/2+... )，在更高维度上与当前最优的 Max-IP 算法保持同步。

**⚠️ 局限性**

局限性包括：
- 仅适用于固定维度 d，算法复杂度随维度快速增长。
- 需要严格控制输入的位长度与数值大小，且误差误差仅为加性误差；在某些应用中可能需要相对误差。
- 代码实现与实测性能未给出，理论证明对实际实现仍有一定距离。
- 对于极大规模的数据集，虽然理论上是近线性，但常数与 polylog 影响仍可能较大。

---

## 146. Growing One's AI Tools: A Paradigm of Practice-Grown Tools

**arXiv ID:** 2611.06132 | [PDF](https://arxiv.org/pdf/2611.06132v1)

**作者:** Zeming Zhao `[一作]` `[通讯]`, Zeming Zhao

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一名非编程背景的会议口译员在八周内通过对话式代码生成与版本控制记录构建并持续迭代的计算机辅助口译工具。

**💡 创新点**

提出了“实践驱动工具”范式，强调专业实践者利用LLM共创、破坏与修复循环，使工具随实践成长，并提供工具成长与评估的新框架。

**🔧 技术方法**

使用了大型语言模型的对话式代码生成（vibe coding）、本地语音识别与语言模型推理、Git版本控制及实时遥测日志等技术。

**📊 数据集**

利用了133个会议会话的遥测日志（共20,707条识别语句），并记录了延迟、分段、语言识别等信息。

**📈 对比分析**

通过端到端延迟、延迟≤2s语句比例等指标比较性能，结果显示工具迭代后延迟从约1.5s降至1.0s，≤2s语句比例从≈40%提升至近100%。

**⚠️ 局限性**

局限性包括单一案例、作者多重角色导致可复制性有限、缺乏对实际翻译质量的评估、硬件与模型变化未完全控制，未验证工具在其他环境的通用性。

---

## 147. AGORA: Reliability-Aware Aggregation of LLM Judges via Item-Response Theory

**arXiv ID:** 2611.05556 | [PDF](https://arxiv.org/pdf/2611.05556v1)

**作者:** Junhyung Ahn `[一作]` `[通讯]` (University of Illinois Urbana-Champaign), Junhyung Ahn (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于等距响应理论（IRT）的加权聚合方法，用于对LLM评判者进行自动可信度估计并输出综合评分。

**💡 创新点**

创新点在于：①用一阶判别参数和阈值捕捉判别者的可靠性与尺度使用；②采用正则化马蹄形先验实现判别参数的稀疏化，避免单独构建混淆表；③设计无标签的协同分散统计量，提前判断加权聚合是否有益；④将判别参数作为无标签的判别者质量诊断。

**🔧 技术方法**

技术包括：等距响应模型、正则化马蹄形先验、EM最大后验估计、EAP读出以及与阈值解码的映射。

**📊 数据集**

实验使用 SummEval（1600条摘要）、Topical-Chat（360条对话）和 WMT22（19725条翻译）三大元评估基准。

**📈 对比分析**

与每项平均、Dawid–Skene、MACE、GLAD、CARE 等传统聚合方法对比，本文方法在全量数据上与 Dawid–Skene 基本持平或略优，在样本量缩小至 50 条时明显优于所有基线，且在混合可靠度的评判者池中优势最为突出。

**⚠️ 局限性**

局限性在于：仅处理绝对序数评分；实验仅覆盖三种任务类型，未验证对其他领域的泛化；不适合高风险决策；若需处理对比偏好数据，需要改用 Bradley–Terry 等似然模型。

---

## 148. Executable Specifications in Transition Algebra

**arXiv ID:** 2611.06208 | [PDF](https://arxiv.org/pdf/2611.06208v1)

**作者:** Daniel Găină `[一作]` (Kyushu University), Ionuţ Ţuţu `[通讯]` (Simion Stoilow Institute of Mathematics of Romanian Academy)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构造与分析带有条件转移的项重写系统，研究了其可连通性、严格连通性以及对应的推理与归约完整性。

**💡 创新点**

创新点在于提出了“严格连通性”“模块化重写系统”与“严格重写”概念，并给出了足以保证归约、归约与约束归约、以及归约与窄化完整性的可行性条件；同时为带动作、蕴含和量化的扩展子句提供了一套完整且可证明的推理规则。

**🔧 技术方法**

主要技术包括：项重写系统的关键对重写分析、基于终止与连通的归约与窄化框架、严格重写的引入以及lifting lemma 的构造、以及对初始模型的可达性和最小模型技术。

**📊 数据集**

本研究属于纯理论研究，并未使用具体数据集；所有结果均在形式化模型与符号推理层面给出。

**📈 对比分析**

由于其为理论性质，未做实验对比；但通过证明可在满足连通、终止、严格连通和模块化等假设时，归约与窄化完整性均能得到保证。

**⚠️ 局限性**

局限性主要体现在：必须满足严格连通性、连通性与终止性等一系列严格假设；且对符号系统的量化与量词处理需要有限性假设；在这些假设不满足时，结论不再适用。

---

## 149. UniRestorer++: Scalable Universal Image Restoration with Multi-Granularity Degradation Modeling

**arXiv ID:** 2611.06042 | [PDF](https://arxiv.org/pdf/2611.06042v1)

**作者:** Jingbo Lin `[一作]`, Wangmeng Zuo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 UniRestorer++，一种可扩展的通用图像恢复框架，利用多粒度降解建模实现高效的专家分配和路由。

**💡 创新点**

创新点：① 构建分层降解层次结构，将整体降解空间递归细分；② 设计多粒度专家路由（基于降解估计和粒度估计），在保持高精度时自动降级到粗粒度专家；③ 引入稠密降解表示映射，实现空间自适应专家选择；④ 通过 LoRA 专家实现参数共享与细化的高效组合。

**🔧 技术方法**

技术细节：稠密降解提取器（ResNet‑18 + CLIP 文本对齐）、K‑means 递归聚类生成层次降解；使用 Restormer 作为共享基底，LoRA 低秩专家；双路由机制 DER 与 GER（顶‑1 路由+不确定度估计）；损失函数包括 L1、降解估计 NLL 与 load‑balancing；训练采用 CutMix 位置标签和像素级对比损失。

**📊 数据集**

使用数据集：高质量图像集合（DIV2K、Flickr2K、DIV8K、MIT‑Adobe‑5K、SA‑1B 等）生成 5M 训练样本；C​DD‑50（50 个降解组合）作为主评测基准；C​DD‑11、单一/三/五降解设置；真实世界数据集（SIDD、RealBlur‑J、DPDD、RealRain、RealSnow、WeatherStream、FoundIR）验证零样本和少样本泛化。

**📈 对比分析**

对比方法：UniRestorer、PromptIR、IDR、MiOIR、OneRestore、其它基线。实验显示 UniRestorer++ 在单/复合/真实场景中均实现最高 PSNR，且在 154M–2.9B 参数范围内呈对数线性提升；相较于纯 MoE 扩展，层次扩展在 OOD 与真实数据上显著更快；零样本推理和 1/5 次微调均优于专家模型。

**⚠️ 局限性**

局限：降解层次结构与专家集合在训练后固定，推理时无法动态更新；未实现在线自适应或自我改进机制。

---

## 150. Transport along Local Annealing Paths: Wasserstein Geometry and Multimodal Sampling

**arXiv ID:** 2611.05605 | [PDF](https://arxiv.org/pdf/2611.05605v1)

**作者:** Han Chen `[一作]` (Duke University), Sifan Liu `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究多模态分布采样中的温度化路径几何，提出分量级温度化传输(CAT)方法，分离局部传输学习与权重校准。

**💡 创新点**

创新点是把全局温度化路径拆分为固定权重的分量级路径，利用Poincaré不等式给出路径长度上界并证明权重估计一致性；并提出基于连续性方程残差的无模拟训练与自适应温度调度。

**🔧 技术方法**

用了Wasserstein几何分析、Poincaré不等式、连续性方程残差损失、正则化的流网络、重要性采样与自适应RESS调度等技术。

**📊 数据集**

用合成多模态混合、Skew-normal混合以及三种贝叶斯后验（SUR、IOHMM、LDA）做实验。

**📈 对比分析**

与全局温度化传输、逆KL连续流、APT、ALPS、AFT等基准方法比较，CAT在ESS、log权重方差、切片Wasserstein距离上表现最佳，壁时相对更短。

**⚠️ 局限性**

局限在于对基底混合分解的依赖，分量间重叠或曲线高密度区域时效果不佳；训练时梯度散度项成本较高，且需要进一步研究自适应分解与更低成本的训练方式。

---

## 151. vExpert: Virtualizing Expert Storage for Adaptive Load Balancing in Distributed MoE Inference

**arXiv ID:** 2611.06057 | [PDF](https://arxiv.org/pdf/2611.06057v1)

**作者:** Wenxun Wang `[一作]` (Tsinghua University), Yongpan Liu `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在分布式 MoE 推理中提出 vExpert，利用主机 DRAM 进行虚拟专家存储，实现全冗余和动态负载平衡。

**💡 创新点**

创新点包括：① 构建联合 H2D 延迟、PCIe 争用与负载不均衡的性能模型；② 设计轻量级动态分配器和分散式内存管理器；③ 把主机存储从部分驻留转变为完整冗余，突破 GPU HBM 限制。

**🔧 技术方法**

技术手段：Mixture‑of‑Experts 分布式推理、PCIe H2D 传输、CUDA IPC、动态分配算法、分散内存管理、系统级性能建模。

**📊 数据集**

使用数据集：DeepSeek‑V3.1（671B LLM）进行基准；合成动态工作负载包括 ShareGPT、LongBench、LiveCodeBench；另外在 KIMI‑K2 上验证跨模型可迁移性。

**📈 对比分析**

与 SGLang、Vanilla、EPLB、Redundant、Offload 等基线对比。vExpert 在负载平衡比提升 55% 的同时，吞吐率提升 12%–28%，在不同批量下均优于基线，并在更大 EP（32/64）规模下保持优势。

**⚠️ 局限性**

局限性：假设传统 PCIe/NVLink 拓扑，未在超节点或新型 CPU‑GPU 共存架构上验证；主机到设备的 H2D 仍受 PCIe 争用影响；对低延迟预填充阶段的适用性不强；需要手动校准模型参数。

---

## 152. Same Method, Different Task: What the Bitter Lesson Does Not License in Social Simulation

**arXiv ID:** 2611.05717 | [PDF](https://arxiv.org/pdf/2611.05717v1)

**作者:** Justin Downes `[一作]` (Amazon Web Services), Newel Hirst `[通讯]` (Amazon Web Services)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过分析Sutton的“苦涩教训”并对社会模拟的适用性进行评估，提出LLM代理缺乏利益激励是其在合作与规范形成等交互式社会现象中表现欠缺的根本原因，随后设计并执行了一个包含540个LLM代理的三族实验，比较了无利益、叙事利益与持久利益三种环境条件对代理行为与类型分布的影响。

**💡 创新点**

创新点在于揭示了苦涩教训在社会模拟中的一个关键前提缺失——两种方法必须解决同一任务，并指出LLM预测器在缺少持久激励时无法产生人类水平的利益冲突与多样性，从而提出需要“混合动机训练”而非仅评估预测器。

**🔧 技术方法**

技术主要包括基于现有LLM（Claude Sonnet 4.6、Llama 4 Maverick 17B、Amazon Nova Pro）在自定义共用资源游戏中的行为模拟、基于策略方法的可解释性采样，以及对不同利益条件下的行为轨迹与类型分布进行统计分析。

**📊 数据集**

数据集方面使用了公开的LLM模型权重与自定义的共用资源游戏环境（类似GovSim），并对每个模型生成了60个代理（共540个）进行8轮决策，随后通过策略方法问卷收集代理的预期抽提量。

**📈 对比分析**

比较方法采用三种条件（无利益、叙事利益、持久利益）以及三族模型的交叉设计，主要指标为代理的提取轨迹、存活率、缓冲区水平与类型分布（共存与非合作类）。实验发现持久利益条件能在两族模型中降低抽提量至运营成本附近，但未能产生与人类相似的自由骑手比例，整体性能表明预测器在缺乏激励时行为与人类偏差显著。

**⚠️ 局限性**

局限性包括实验环境缺乏真实的公共物品外部性与惩罚机制，条件B同时包含余额跟踪、惩罚与移除，导致难以单独检验利益效应；样本仅限三族模型且仅采用单温度、单提示顺序，且与人类基线来自不同游戏，限制了结论的外部有效性。

---

## 153. Matching the Training-Time Weight Rule in Inference-Time Expert Deactivation

**arXiv ID:** 2611.05804 | [PDF](https://arxiv.org/pdf/2611.05804v1)

**作者:** Hao Cheng `[一作]` (Zhejiang Gongshang University), Hanzhang Jia `[通讯]` (University Of Sussex)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了推理时去激活专家时四种实现方式对token路由和专家权重的影响，并通过权重求和模型预测最佳实现与训练时权重规则的对应关系。

**💡 创新点**

提出了统一的权重求和模型，揭示了权重规则不匹配导致的性能退化机制，并将其推广到推理时宽度调整。

**🔧 技术方法**

对四种实现（去除、冻结、剩余专家归一化、未归一化重新路由）进行实现与实验，使用了Transformer的MoE层。

**📊 数据集**

采用了来自OLMoE、DeepSeek‑V2、Granite、Qwen3等公开的指令调优checkpoint，以及WikiText‑2、ARC、MMLU、GSM8K等数据集。

**📈 对比分析**

在相同禁用专家集合下比较四种实现，发现剩余专家归一化在原始概率模型中造成最大困惑度提升，未归一化重新路由在归一化权重模型中造成最大困惑度提升；实验显示误差幅度可达数千倍。

**⚠️ 局限性**

仅覆盖了六个checkpoint，实验规模受限于模型大小与C4校准数据集，未考虑不同任务的恢复训练与生成任务多样性。

---

## 154. Autonomous agents for electricity-system planning

**arXiv ID:** 2611.05911 | [PDF](https://arxiv.org/pdf/2611.05911v1)

**作者:** Yoon Pyo Lee `[一作]` (University of Illinois Urbana-Champaign), Syed Bahauddin Alam `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并评估了一套基于语言模型的自治规划框架，能够从区域输入自动完成投资、调度、网络验证并在碳预算下选取最低成本方案。

**💡 创新点**

将预训练语言模型与电力系统优化工具（PyPSA、PySAM、OpenDSS）整合形成完整规划循环；支持单体与多体代理架构，代理通过数值反馈自主修改设计并决定停止；在多系统、多年份、碳预算约束下实现完整规划并与公开参考方案对比。

**🔧 技术方法**

使用预训练语言模型（GPT‑6.1 Sol、Claude Opus 5.5）、混合整数规划求解器 Gurobi、PyPSA 进行投资+调度、PySAM 生成 PV 资源、OpenDSS 进行代表性配电线路验算；代理交互框架、程序化决策图（procedure graph）等技术。

**📊 数据集**

采用 EIA‑930 2023 需求、EIA‑860 设备、NASA NEX‑GDDP‑CMIP6 2020s 天气、ERA5 2023 天气、各国公开电网拓扑（NYISO, MISO, ERCOT, South Africa ESRG, Australia NEM, New Zealand EMI, Mexico SENER, Brazil ONS）以及各国规划文件（IRP, PDE, PROSENER 等）作为数据集。

**📈 对比分析**

对每个系统-年份组合采用单体/多体架构和两种模型部署共四条轨迹；记录投入、系统调度、配电验证及碳排放；最终选取符合物理与碳预算的最低成本设计。结果显示 40 条碳约束轨迹均成功产生可接受方案，28 条对照方案超出碳上限；在与公开规划方案比较时，部分技术组合出现显著差异。单体与多体在成本上差异不大，单体在某些系统中略优。

**⚠️ 局限性**

仅限单年度规划，未考虑融资、许可、建设时序；配电模型为代表性简化，仅用单一线路示例；气候输入仅为单一情景；未考虑生命周期排放、社会接受度；代理未训练专门任务，仅靠预训练模型；实验未做重复或统计验证，结果为单次轨迹；未覆盖网络细节、极端事件、可靠性指标。

---

## 155. Information-Matched Auditing of Conversational Emotion-Shift Forecasting: A Leakage-Safe Evaluation Protocol

**arXiv ID:** 2611.05895 | [PDF](https://arxiv.org/pdf/2611.05895v1)

**作者:** Bin Wen `[一作]` (Universiti Sains Malaysia), Tien-Ping Tan `[通讯]` (Universiti Sains Malaysia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一套信息匹配、泄露安全的对话情绪转移预测审计协议，并在此框架下评估六类因果模型。

**💡 创新点**

创新点在于：①引入正式的承诺点和目标缺失数据构造；②构建八项泄露审计与可执行因果测试；③使用训练仅预测标签的过渡基线并与模型对齐；④对四种配置做等价验证搜索和种子‑对话层次推断。

**🔧 技术方法**

使用因果序列模型（DialogueRNN、DialogueGCN、DAG‑ERC、PEC、Pseudo‑Future）和基于转移矩阵的基线；评估指标包括ROC‑AUC、PR‑AUC、Brier、ECE 等；并实现了层级自举、配对对话聚类置换检验及 Holm 多重校正。

**📊 数据集**

实验数据集包括四个文本语料库：IEMOCAP、MELD、EmoryNLP、DailyDialog；以及两种多模态配置（IEMOCAP、MELD 的 MM‑DFN 文本+音频+视觉特征）。

**📈 对比分析**

通过信息匹配的部署式对比，所有六种模型在四大文本语料库上均显著优于基线，DailyDialog 的提升最大且通过 Holm 校正显著；在 oracle（给出黄金当前情绪）情景下，基线在 MELD 仍占优势，模型提升不明显。

**⚠️ 局限性**

局限性包括：①信息匹配仅在当前情绪层面实现，未完全匹配所有标签信息；②对上游特征提取的泄露控制仍不完备；③多模态实验受限于可用特征；④模型性能高但校准不足，需后期校准；⑤实验规模受限于可重复的数据与模型实现。

---

## 156. Single-Exponential FPT Algorithms for Minimum Bisection and Matching Cut via a New Unbreakable Decomposition

**arXiv ID:** 2611.05982 | [PDF](https://arxiv.org/pdf/2611.05982v1)

**作者:** Soh Kumabe `[一作]` `[通讯]` (CyberAgent, Inc.), Soh Kumabe (CyberAgent, Inc.)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一个新的 O(2^k poly(n)) 的固定参数化算法，用于求解最小二分割（Minimum Bisection）问题。

**💡 创新点**

通过构造无分解（unbreakable decomposition）并利用重要分隔子集，证明了在 ETH 下该时间复杂度是最优的。

**🔧 技术方法**

使用树分解、重要分隔、颜色编码、动态规划以及组合数学技巧实现了算法。

**📊 数据集**

该工作基于理论分析，无需使用实际数据集。

**📈 对比分析**

与传统的 O(2^k) 朴素算法相比，该方法在参数 k 较小的情形下显著降低了常数因子，并通过 ETH 证明了不可能进一步改进。

**⚠️ 局限性**

算法只适用于 Minimum Bisection（以及匹配割）问题，无法直接推广到更一般的割问题，且实现复杂度高。

---

## 157. A low-rank decoder bottleneck bounds reliability in foundation-model perturbation prediction

**arXiv ID:** 2611.05849 | [PDF](https://arxiv.org/pdf/2611.05849v1)

**作者:** Zheyu Zhang `[一作]` (University of Michigan), Jie Liu `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对基于基础模型编码器的状态转移（ST）解码器在CRISPRi细胞扰动预测中的可靠性进行了系统审计，并提出了预测近邻距离（predicted‑NN‑distance）作为每个预测的置信度评分；

**💡 创新点**

创新点在于发现ST解码器的首层注意力是一个低秩瓶颈，扰动信息仅通过训练扰动固定的少数方向传递，导致预测准确性与扰动响应与训练响应的余弦距离呈连续关系；

**🔧 技术方法**

技术手段包括线性/MLP探针、SVD低秩截断、子空间干预、以及基于预测与训练响应余弦距离的置信度评估；

**📊 数据集**

使用了四个CRISPRi细胞系（HepG2、Jurkat、RPE1、K562）的Replogle–Nadig数据集，结合三种预训练编码器（SE‑600M、scGPT、Tahoe‑x1）以及五个药物剂量条件的Tahoe‑100M数据；

**📈 对比分析**

与简单线性基线（ridge回归）、平均训练响应、CPA等方法比较，ST解码器在与训练响应相近的扰动上可达与线性基线相当的PDC，远离训练扰动时准确度急剧下降；预测近邻距离与真实PDC的Spearman相关≈0.81，阈值≈0.10即可在多种编码器/细胞系上选出高准确度预测；

**⚠️ 局限性**

局限性包括：仅评估了三种编码器的相同解码器结构，未考察更复杂或不同的解码器；低秩瓶颈对Tahoe‑x1部分不明显；在整个基因家族缺失的情况下评分相关性减弱；且Learn‑then‑Test的保证在新环境下可能失效。

---

## 158. The Invocation Bottleneck in LLM-Based Lossless Compression: What Can Be Removed and What Must Be Traded

**arXiv ID:** 2611.05871 | [PDF](https://arxiv.org/pdf/2611.05871v1)

**作者:** Zhe Yan `[一作]` (Xidian University), Huanchen Zhang `[通讯]` (Tsinghua University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）在无损压缩中的解码耗时来源，并通过GPU驻留、批量不变核（batch‑invariant kernels）和块级并行编码去除执行冗余；随后评估了两种降低序列依赖的方案：多令牌单模型编码（MSMC）和投机双模型编码（SDMC），比较其压缩率、吞吐量与传统压缩器的差距。

**💡 创新点**

①将执行冗余与序列依赖区分开来，并证明后者不可通过优化实现消除；②设计了一套GPU‑驻留、批量不变的编码流水线，使算术编码和 Rank‑Tier 编码在 FP16 下保持完全确定性；③在多令牌编码中通过训练额外头部实现一次模型调用解码多令牌，揭示了编码长度损失与头部预测质量之间的定量关系；④首次在投机双模型压缩中验证在无验证场景下的可行性与限制。

**🔧 技术方法**

批量不变的 GPU 内核、块级并行教师强制前向推理、概率量化到整数频率、Rank‑Tier 编码、投机式小模型与大模型的协同推理、头部训练与域适配、FP16/BF16 与 FP32 的混合精度控制。

**📊 数据集**

简单英语维基百科文本（约10 MB）、C 代码片段（约10 MB），每个数据集随机抽取10个文件；此外还使用了10 MB合并文件进行长流实验。

**📈 对比分析**

与 gzip‑9、zstd‑3/‑19、xz‑9 等经典压缩器在压缩率（bits/byte）和吞吐量（KB/s）上进行对比；实验表明：在不改变压缩率的前提下，批量不变流水线将编码吞吐提升 5.8×、解码吞吐提升 3.0×；MSMC 在解码吞吐上可实现 1.9–2.0× 的加速，但压缩率升高至 1.78–3.0 bpb，已不再优于 xz‑9；SDMC 只能在解码上略微提升 10% 并需 FP32，整体性能仍远落后于传统压缩器。

**⚠️ 局限性**

①SDMC 在 FP16 下不可实现无损；②MSMC 的压缩增益受限于头部预测质量，长距离预测误差导致压缩率快速恶化；③解码吞吐仍极低（单流 1–5 KB/s），不适合实时或低延迟场景；④实验仅在 0.5B/7B Qwen 模型上验证，未覆盖更大规模模型或多 GPU 环境；⑤批量不变核和量化方法在不同硬件（H100）上需要进一步验证；⑥对存储和训练头部的 amortization 仍是实际部署的关键瓶颈。

---

## 159. Chronos: Sequential Identity-Preserving Video Generation by Temporal Anchoring

**arXiv ID:** 2611.05893 | [PDF](https://arxiv.org/pdf/2611.05893v1)

**作者:** Mengqi Huang `[一作]` (University of Science and Technology of China), Yongdong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Chronos框架，实现了面向单段和多段动作的身份保留视频生成；

**💡 创新点**

通过语义提示锚定、视觉帧锚定与多维候选路由，解决了身份与时间可控性之间的矛盾；

**🔧 技术方法**

使用GPT提取身份特征并自适应提示，I2I图像合成生成关键帧锚点，并对多模型生成器进行多维评估与路由；

**📊 数据集**

主要在ACM MM 2026 Identity‑Preserving Video Generation Challenge数据集（参考图像+文本/时间提示）上进行实验；

**📈 对比分析**

与Stand‑In、RefAlign、Phantom、IPT2V等方法相比，Chronos在身份一致性、文本对齐和视觉质量等指标上均居首，获竞赛第一；

**⚠️ 局限性**

对极端遮挡或非人类主体的身份保持仍有挑战，且依赖多模型集成导致推理成本较高。

---

## 160. Memory-Access Profiling for Early Multicore Interference Awareness in Embedded Systems

**arXiv ID:** 2611.05704 | [PDF](https://arxiv.org/pdf/2611.05704v1)

**作者:** Jean-Baptiste Lefoul `[一作]` (Polytechnique Montreal), Gabriela Nicolescu `[通讯]` (Polytechnique Montreal)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种轻量级的内存访问分析框架，利用源代码级别的内存操作跟踪并映射到目标多核平台的缓存资源，生成缓存占用热图和临时占用指标，以便在正式部署前对多核干扰进行早期评估。

**💡 创新点**

创新点在于将源级内存访问跟踪与硬件映射相结合，形成非侵入式的非栈访问比例筛选、联合缓存占用分析以及“Contention Potential Growth”临时占用增长度量，提供可解释且低成本的干扰风险预测。

**🔧 技术方法**

使用的技术包括C语言类型替换包装器进行内存操作插桩、硬件参数映射、缓存集占用统计、热图生成、堆栈与非堆栈访问比例计算、联合占用与临时占用分析以及基于ΔCP的风险评估。

**📊 数据集**

实验数据集为TACLeBench中的四对算法（Dijkstra/Floyd-Warshall、FFT/Jacobi-2D、AES-128 ECB/Conv2D、Bsort/ADPCM Encoder）以及自定义基准，运行于NXP QorIQ P2020RDB-PCA双核平台。

**📈 对比分析**

通过对单核与并行执行的L2缓存缺失计数进行比较，框架能准确识别高风险组合（如Dijkstra/Floyd-Warshall导致+213%缺失），对近容量组合给出保守警告，且对低风险组合验证误差在1%以内，整体性能评估表明预测准确率高。

**⚠️ 局限性**

局限性包括仅针对缓存干扰，无法精确捕捉替换策略、预取、写缓冲等细节，需手动修改源代码以插桩，且对多核时间调度的精确度有限，易产生保守误报。

---

## 161. RECAP: Context-Aware Prediction Recycling with Request-Local and Persistent Memories for Multimodal Speculative Decoding

**arXiv ID:** 2611.05894 | [PDF](https://arxiv.org/pdf/2611.05894v1)

**作者:** Hengjie Zhu `[一作]` (Chinese Academy of Sciences), Weiping Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练、无额外草稿模型的预测回收框架RECAP，利用多长度上下文键和请求本地与持久化记忆并通过熵‑置信度加权融合，显著提升视觉‑语言模型的推理速度。

**💡 创新点**

创新点在于：①引入多长度（1、2、3个token）上下文键同时结合请求本地与跨请求持久记忆，解决条件别名与覆盖不足问题；②用熵‑置信度加权而非单一优先级融合多源预测；③通过最佳优先搜索与后缀自动机构建候选树，保持与原始模型相同的贪心生成。

**🔧 技术方法**

技术主要包括：预测回收、上下文键索引、熵‑置信度加权融合、最佳优先树搜索、后缀自动机、单步目标验证。

**📊 数据集**

使用七个单轮基准（MMT‑Bench、SEED‑Bench、ScienceQA、OCRBench、ChartQA、MathVista、TextVQA）和多轮MMSpec数据集进行评测。

**📈 对比分析**

与六个无训练草稿模型（Prompt Lookup、Lookahead、Token Recycling、Tree Recycling、SAM、SAM+ViSkip）对比，RECAP在Qwen2.5‑VL‑7B上平均速度提升1.66倍、在LLaVA‑1.6‑7B上提升1.56倍，且接受长度（τ）均高于其他基线；在MMSpec上分别达到1.89倍和1.59倍。

**⚠️ 局限性**

局限性包括：仅在单轮、批量1、贪心解码下测试；仅针对两款7B VLM；未评估推理对下游任务准确率或真实服务延迟分布；默认参数可能对不同硬件或请求长度产生不同的内存–延迟折衷。

---

## 162. When Does Better Confidence Improve a Knowledge-Graph Audit?

**arXiv ID:** 2611.05877 | [PDF](https://arxiv.org/pdf/2611.05877v1)

**作者:** Zehao Lin `[一作]` (MemTensor), Helan Hu `[通讯]` (MemTensor)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将KG审计建模为预算化覆盖问题，探讨了错误可修复性、选择余量和标签有效性，并给出了理论极限与实际实验。

**💡 创新点**

创新点包括：①在非自适应审计下推导出最小统计量与结构误差质量的极限；②用结构质量（structural mass）量化联合建模的潜在收益；③结合多种错误表示（lattice、块、标量、节点风险）和 MILP 规划，系统评估验证和实体审计的实际效果。

**🔧 技术方法**

技术方法：概率模型与属性激励（property elicitation）、价值信息（value‑of‑information）理论、MILP 优化（HiGHS）、Bootstrap 置信区间、TOST 等差异检验、LLM 验证器、结构质量度量。

**📊 数据集**

数据集：Re‑DocRED（dev、test、train500）用于单文档实验；Linked‑DocRED 与 Qwen/GPT‑4.1 提取的跨文档图；以及通过哈希抽样得到的 500 训练文档。

**📈 对比分析**

对比方法：基于标量的优先级、节点风险、独立块、联合 lattice、8 类错误掩码；验证器与无验证器的差异。实验显示：联合 lattice 计划在两种跨文档图上比节点风险减低 3.98–4.74 个错误；在 20%–30% 预算下，验证器可在 GPT‑4.1 上提升 32–71 个修复；单文档中，联合模型捕获约 20–25% 的 oracle 提升，标量计划已接近 oracle，验证器收益有限。

**⚠️ 局限性**

局限性：①仅在 DocRED 系列两款提取器上验证，无法覆盖更广泛的 KG 提取任务；②理想修复、非自适应计划、加性目标的假设在真实审计中难以满足；③验证器与提取器的效应未解耦；④标签有效性分析仅基于 LLM 判定，缺乏人工标注验证；⑤结构质量估计依赖已标注样本，易受噪声影响。

---

## 163. Cross-Modal Semantic Successive Refinement for Generative Image Communication over Underwater Acoustic Channels

**arXiv ID:** 2611.06205 | [PDF](https://arxiv.org/pdf/2611.06205v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 164. Loop Before You Leap: What Recurrent Depth Can and Cannot Buy in Data-Limited Pretraining

**arXiv ID:** 2611.06206 | [PDF](https://arxiv.org/pdf/2611.06206v1)

**作者:** Daichi Hotta `[一作]` (Aoyama Gakuin University), Yutaka Matsuo `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在100M单词预算下，比较循环Transformer与标准Transformer在文本与多模态训练中的BLiMP性能，探讨循环计算是否能缩小多模态与文本之间的差距。

**💡 创新点**

在同一实现下系统评估循环（共享层）与深层标准模型的计算分配，利用差分差分分析揭示多模态下降主要由语言数据变化而非视觉输入导致，并验证循环未显著改善多模态下降。

**🔧 技术方法**

使用LLaMA式解码器Transformer，加入RMSNorm、旋转位置嵌入、SwiGLU、DINOv2冻结视觉向量前缀，采用共享层循环（K=4）、AdamW优化、词频预算调度，并通过BLiMP评测和统计差分差分方法进行比较。

**📊 数据集**

使用2026年BabyLM Strict文本语料（100M词）与50M文本+50M图像字幕（来自Conceptual Captions和Localized Narratives），以及预计算的DINOv2 ViT‑B/14视觉特征。

**📈 对比分析**

通过参数匹配和计算匹配的实验，循环模型在文本与多模态下相较48层标准模型提升5–6点BLiMP，但循环未能显著减小多模态下降（-0.01±1.44点），多模态模型因语言数据改变导致BLiMP下降约3–4点。

**⚠️ 局限性**

实验仅使用三种随机种子、未完全分离视觉输入与语言数据变化、仅采用单一全局视觉向量、未评估更深或不同循环次数、学习率调优不充分、缺少标准模型的匹配无图像控制，导致结论的稳健性受限。

---

## 165. Risk Control Certifies Rates, Not Magnitudes: An Audit of Selective Fetal Biometry

**arXiv ID:** 2611.05851 | [PDF](https://arxiv.org/pdf/2611.05851v1)

**作者:** Zhengshuai Yang `[一作]` (Hunan Agricultural University), Guihu Zhao `[通讯]` (Central South University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究审计了基于工具的产前超声头围测量系统PUMA，评估其在多分割下的鲁棒性与风险控制。

**💡 创新点**

创新在于将几何测量、候选筛选与语言模型验证分离，并使用分布式风险控制证明只约束错误率而非误差幅度。

**🔧 技术方法**

采用DeepLabV3-ResNet50进行头骨轮廓分割，利用最小二乘拟合椭圆求取头围，并用Clopper–Pearson置信区间进行阈值校准。

**📊 数据集**

主要使用HC18标注图像集（999张），并结合FETAL_PLANES_DB、PMC-VQA等公开数据进行评测。

**📈 对比分析**

将PUMA与直接VLM、检索式策略对比，平均绝对误差降至1.76 mm，覆盖率约66%，但单个严重误差高达233–271 mm，说明风险控制仅限于错误率。

**⚠️ 局限性**

局限在于无患者身份信息导致图像级分割、样本量小的校准不足以及对极端误差缺乏有效阈值调节。

---

## 166. Reachability Diameter: Beyond the Square-Root Barrier, with Polynomial Hardness

**arXiv ID:** 2611.05712 | [PDF](https://arxiv.org/pdf/2611.05712v1)

**作者:** Hector Tierno `[一作]` (University of Massachusetts), Shuang Yang `[通讯]` (University of Massachusetts)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种随机化、近线性时间（O(m+n)）的算法，用于估计无权有向图的可达性直径，并给出了在 3‑Uniform Hyperclique 假设下的近似下界，证明在近线性时间内无法获得多项式更小的误差。

**💡 创新点**

创新点包括：1) 通过递归分区和 Cohen 的“扫掠”技术，突破传统 n^{1/2} 的障碍，得到 n^{1/3+o(1)} 的近似；2) 在 3‑Uniform Hyperclique 假设下构造图形，证明任何 O(m+n) 时间算法至少需要 n^{2/15-ε} 的近似误差，显示多项式误差在此时不可避免。

**🔧 技术方法**

使用了 Jambulapati–Liu–Sidford 的递归分区（JLS partitioning）框架、Cohen 的可达性集合估计、三维坐标与能量坐标的几何编码、dyadic 区间快捷连接、以及 Chinese Remainder Theorem 对能量坐标的分解，以实现高效的图构造与距离估计。

**📊 数据集**

本文为理论工作，没有使用实验数据集，所有结果均通过数学证明获得。

**📈 对比分析**

通过理论分析证明，算法在 O(m+n) 时间内实现 n^{1/3+o(1)} 的近似；与先前的 n^{1/2+o(1)} 近似相比，误差指数从 1/2 降低到 1/3；下界表明在近线性时间下不可能实现比 n^{2/15-ε} 更小的误差。

**⚠️ 局限性**

局限性：仅适用于无权有向图；对权重图仍无多项式近似方案；下界依赖于 3‑Uniform Hyperclique 假设；没有实验验证或实际实现细节；算法仅给出期望/高概率保证，实际运行时的常数因子未给出。

---

## 167. BrainBind: Bridging Synchronized EEG and fMRI through Long-Form Naturalistic Video Representations

**arXiv ID:** 2611.05581 | [PDF](https://arxiv.org/pdf/2611.05581v1)

**作者:** Shixiao Wang `[一作]` (University of Birmingham), Jian K. Liu `[通讯]` (University of Birmingham)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 BrainBind 框架，将同步记录的 EEG 与 fMRI 对齐到冻结的长视频嵌入空间，实现跨模态检索与融合。

**💡 创新点**

创新点在于：①在相同长视频刺激下同步比较 EEG 与 fMRI 的对齐效果；②将视频嵌入视为共享视觉语义参考，而非直接神经代码；③通过多种代理(anchor)与系统配置（Transformer、S4AE、Cross‑Attention、Joint‑Dual）探讨精度‑广度权衡；④引入多指标评估（Recall@K、MRR、mAP、nDCG@10）和 episode‑disjoint 评估揭示情节上下文对检索的影响。

**🔧 技术方法**

使用的技术包括：冻结视频编码器（Omni、ImageBind）、Transformer 与 Structured State‑Space (S4AE) 编码器、InfoNCE/ SupCon 对比损失、跨模态融合（交叉注意力、均值融合）、多指标评价、PCA/CCA 代理几何诊断、温度与损失 ablation。

**📊 数据集**

实验基于 CineBrain 数据集，包含同步 EEG、fMRI 与长视频片段，随机划分和 episode‑disjoint 划分两种评估。

**📈 对比分析**

通过 2×4 系统矩阵（两种代理 × 四种配置）进行检索比较。结果显示：fMRI‑Omni 在 rank‑wide 评估（MRR、mAP、nDCG@10）表现最佳；Joint‑Dual‑Omni 在 top‑rank（R@1）表现最佳；EEG‑Omni 仍高于随机对照。与 ImageBind 的对比表明 Omni 代理因谱宽度更大、冗余更低而获得更高的检索性能。

**⚠️ 局限性**

局限性：①模型对受试者高度依赖，跨受试者泛化不佳；②缺乏容量匹配的控制实验，难以分离模态信息与网络容量的贡献；③仅在 CineBrain 上验证，未覆盖更多自然视频语料；④使用的对比损失未针对多正样本进行优化，无法充分利用重复标签结构。

---

## 168. ModalSearcher: Weaving Visual Evidence into Autonomous Cross-Modal Deep Search

**arXiv ID:** 2611.06131 | [PDF](https://arxiv.org/pdf/2611.06131v1)

**作者:** Zhihan Yin `[一作]` (Peking University), Dongyan Zhao `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ModalSearch-Pipeline 与 ModalSearcher Agent，能够自主在多模态信息检索过程中交替使用文本检索、图像检索与反向图像检索，并将检索结果集成到推理链中；

**💡 创新点**

创新点在于：①通过交错的多模态知识图构建与路径采样生成包含跨模态推理路径的训练数据；②采用逆向问句重写隐藏显式检索目标，迫使模型自主决定检索策略；③结合监督微调与强化学习，优化多轮检索与推理决策；

**🔧 技术方法**

使用技术包括：多模态大语言模型（以 Qwen3‑VL‑8B 为基座），ReAct 框架进行工具调用，SFT 与 RLOO 强化学习，视觉实体链接、检索可检索性评估以及逆向问句重写；

**📊 数据集**

数据集方面，利用自建的 16K 训练样本（Pipeline 生成），并加入 FVQA 与 LiveVQA 训练集；评估使用 MMSearch、BrowseComp‑VL、VDR‑Bench、FVQA、LiveVQA、VisBrowse、BrowseComp‑V3 与自研的 ModalSearch‑Bench；

**📈 对比分析**

与基线对比，ModalSearcher‑8B 在五个多模态知识密集型基准上平均提升 16.2 分，在 VisBrowse 与 BrowseComp‑V3 上分别提升 19.8 与 15.8 分，整体显示显著优于 Qwen3‑VL‑8B 与多款开源/闭源检索代理；

**⚠️ 局限性**

局限性包括：①模型仍需依赖预先构建的知识图，无法实时生成新知识；②对极端模糊或多义性查询的处理仍不够鲁棒；③在跨域或极少样本场景下的泛化能力尚待进一步验证。

---

## 169. Jev with Foresight: Safe Language-Guided Autonomous Driving via a Language World Model

**arXiv ID:** 2611.05448 | [PDF](https://arxiv.org/pdf/2611.05448v1)

**作者:** Yansong Qu `[一作]` (Purdue University), Samuel Labi `[通讯]` (Purdue University)

**通讯引用:** 6886 | [OpenAlex ID](https://openalex.org/A5065592706)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种基于类型化决策模型Jev和语言世界模型的安全语言导向自动驾驶框架，能够一次性理解并执行自由文本驾驶命令，并通过预见后续车辆行为来避免碰撞；

**💡 创新点**

创新点在于首次将类型化决策模型与语言世界模型相结合，实现快速闭环决策，并通过“预见”显著降低碰撞率；

**🔧 技术方法**

主要技术包括：类型化决策模型（Jev）用于命令归一化与决策，Qwen2.5-0.5B+LoRA构建的语言世界模型预测未来2秒内各候选动作的车辆轨迹和结果，并将其转化为可比的文字后果；

**📊 数据集**

使用的数据集为KoMA场景（高速合流、三车道合流、单车道合流、环形交叉口）和NGSIM真实高速公路行车轨迹（US‑101与I‑80），并在这些数据上训练和评估模型；

**📈 对比分析**

通过与多种基线（Jev+KoMA文本、oracle预见、常规速度预见、LLM+WM、KoMA原生驱动）对比，实验显示预见将碰撞率从26%降至2.4%（合流），从48.2%降至2.8%（NGSIM），并实现93%的乘客意图满足率，决策时长约0.25秒；

**⚠️ 局限性**

局限性包括：仅在仿真和重放数据上测试，未在真实车辆上验证；使用地面真实状态，未考虑感知误差；动作空间仅为五个宏观动作；语言世界模型缺乏不确定性估计，无法识别训练外输入；使用的Jev为专有API，版本变动可能影响性能；

---

## 170. SysPilot: Cross-Library Performance Autotuning via Library Performance Contracts

**arXiv ID:** 2611.06078 | [PDF](https://arxiv.org/pdf/2611.06078v1)

**作者:** Yineng Yan `[一作]` (University of Texas at Austin), Christopher J. Rossbach `[通讯]` (University of Texas at Austin)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套基于Library Performance Contract（LPC）的跨库性能自动调优框架 SysPilot，能够在不修改库代码的前提下识别和解决不同库间的资源争用。

**💡 创新点**

创新点包括：①将库级性能知识抽象为可复用的、可编程的Contract语言；②通过Contract合成静态冲突假设并结合运行时采样进行验证；③在调优过程中将冲突假设转化为候选配置，实现从问题定位到调优的闭环。

**🔧 技术方法**

使用技术主要包括：LPC 语言与编译器；LLM 辅助 Contract 生成（利用 GPT‑5.4 抽取文档和代码中的证据）；虚拟探针与平台无关的采样工具；基于假设的候选配置生成与验证；集成 OpenTuner、SMAC3 等通用调优器做对比。

**📊 数据集**

数据集为自定义的 LibContentionBench，包含 9 个工作流（5 个合成数据、4 个真实数据），覆盖 CPU oversubscription、NUMA 分配、内存/磁盘 I/O、GPU 共享等多种跨库争用场景。

**📈 对比分析**

对比方法包括：基于全局调优器（OpenTuner、SMAC3）、手工编写的应用级手册指导，以及 SysPilot 的 Contract‑guided 方案。实验结果显示，SysPilot 在 30 次候选试验内平均获得 0.960 的归一化得分，优于 SMAC3 的 0.824 和 OpenTuner 的 0.729；达到 0.8 目标所需试验数约为 8 次，约为通用调优器的 3 倍之少；在大多数工作流中比应用手册更快或相当，且在跨工作流迁移时表现出更好的可复用性。

**⚠️ 局限性**

局限性包括：Contract 质量受限于文档与代码的完整性；LLM 抽取可能遗漏或误解性能信息；合成规则过于保守，易产生误报或漏报；缺乏版本/构建感知，难以在不同库版本或构建下自动匹配；目前仅针对单机调优，未覆盖分布式工作流。

---

## 171. Generative AI for urban planning: synthesizing land-use maps with satellite imagery via DualControlNet

**arXiv ID:** 2611.05859 | [PDF](https://arxiv.org/pdf/2611.05859v1)

**作者:** Qingqi Song `[一作]` (University of Florida), Shenhao Wang `[通讯]` (University of Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 DualControlNet，能够根据文本和控制图生成符合规划目标的土地使用图和卫星图。

**💡 创新点**

创新点在于将生成任务从单模态扩展为双模态，并通过空间一致性损失实现跨模态几何对齐，同时把土地使用指标作为控制和验证标准。

**🔧 技术方法**

采用扩散模型 Stable Diffusion 与 ControlNet 的双模态扩展，并加入地理一致性损失。

**📊 数据集**

使用来自奥兰多和芝加哥的 40,000 多个匹配的卫星图、土地使用图和控制图的数据集。

**📈 对比分析**

与单模态 ControlNet 基线比较，DualControlNet 在土地使用一致性提升 62.8%~16.3%，空间一致性翻倍，视觉 FID 降低约三倍，显示显著性能提升。

**⚠️ 局限性**

局限包括跨模态不一致、标签粗糙、生成块不连续、对未开发区的分类不足。

---

## 172. Human-like cross-language generalisation in deep neural speaker embeddings and its acoustic foundations

**arXiv ID:** 2611.05384 | [PDF](https://arxiv.org/pdf/2611.05384v1)

**作者:** Tianze Xu `[一作]` (University of Zurich), Volker Dellwo `[通讯]` (University of Zurich)

**通讯引用:** 1799 | [OpenAlex ID](https://openalex.org/A5028554788)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

探究深度神经网络说话人嵌入在不同语言（粤语与英语）之间的可泛化性及其与人类感知和声学结构的对应关系。

**💡 创新点**

首次将18种前沿说话人嵌入模型与双语说话者的声学特征及人类相似度评分进行线性混合模型与表征相似性分析，验证模型与人类在跨语言声纹相似度评估上的高度对齐。

**🔧 技术方法**

使用深度卷积网络（ResNet、SimAM、ECAPA、CAM++）、线性混合模型（LMM）与表征相似性分析（RSA），并基于Wespeaker工具抽取嵌入。

**📊 数据集**

语料为SpiCE双语说话者数据（10名粤英双语女性说话者）以及对应的声学特征和人类相似度评分；嵌入模型使用18个预训练模型（VoxCeleb、CN-Celeb、VoxBlink2等）及其在4–5小时粤语语料上微调的版本。

**📈 对比分析**

通过LMM评估同一说话者与不同说话者对嵌入相似度的影响，RSA将模型相似度与人类评分及29个声学变量的RDM进行相关性检验；结果显示模型相似度与人类相似度平均相关系数约0.44，且同样对同语/异语混合条件均表现出与人类一致的“语言切换抑制”效应，声学特征与模型相似度的主要相关变量为语速、F1–F4平均值及 SHR。

**⚠️ 局限性**

微调后的模型并未提升粤语对比的相似度，反而略低；可能因粤语数据量有限或模型对粤语特征的学习不充分；此外研究仅使用短句、10名说话者，缺乏更广泛的说话者与语料多样性，限制了结论的普适性。

---

## 173. FaCoDiff: Factor-Correlated Diffusion for Multivariate Probabilistic Forecasting

**arXiv ID:** 2611.06166 | [PDF](https://arxiv.org/pdf/2611.06166v1)

**作者:** Jiaxuan Li `[一作]` (University of Electronic Science and Technology of China), Qiao Liu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种适用于多变量概率预测的分解式扩散模型FaCoDiff，结合条件统计估计器与结构化去噪，建模跨变量依赖与边际不确定性。

**💡 创新点**

采用低秩因子协方差与对角补偿实现高效的条件协方差建模，历史条件载荷、中心/尺度校正与噪声依赖分离；通过几何条件化去噪和混合误差目标提升采样质量；实现矩阵无关运算，保持低内存。

**🔧 技术方法**

条件扩散模型、低秩因子化（diagonal+low-rank）、Transformer历史编码器、几何条件化噪声调制、混合白噪声/传输误差损失、Woodbury逆与行列式求解、矩阵无关运算、分段训练（基线、因子热身、联合）。

**📊 数据集**

9个多变量时间序列数据集：Solar, ECL, Traffic, Weather, ILI, ETTh2, ETTm2, London, Taxi。

**📈 对比分析**

与 DeepVAR、TMDM、NsDiff、CW-NsDiff、DiffPTS 等基线在相同训练协议和评估指标（CRPS, QICE, Energy, Variogram, MAE, MSE）下比较，FaCoDiff 在大多数数据集的 Variogram 得分排名第一，整体在 54 个数据集-指标组合中位居前两，且生成效率与对角噪声模型相近。

**⚠️ 局限性**

仅对预测步骤假设条件独立；固定因子容量，未对时间耦合或因子自适应进行探索；有限的扩散步长和逆链长度可能限制预测质量；在极高维度仍受算子成本影响。

---

## 174. The Confidence Score: a New Measure on Kernels

**arXiv ID:** 2611.05528 | [PDF](https://arxiv.org/pdf/2611.05528v1)

**作者:** Elizabeth Hofer `[一作]` (McMaster University), Martin v. Mohrenschildt `[通讯]` (McMaster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的核度量——Confidence Score，用来评估核函数在分类任务中的适用性。

**💡 创新点**

核心创新在于放宽了传统核对齐假设，采用“相似 → 同类”而非“相似 ↔ 同类”，并通过对核的对称归一化与去自相似化（hollowing）来计算与真实标签的对齐度，从而更好地保留类内结构。

**🔧 技术方法**

使用了核方法的对称归一化、去自相似化、Frobenius 内积、核对齐（CKA）、多核学习、正则化Fisher分离准则等技术，并对多种核族（高斯、Matérn、本地可变宽度高斯/拉普拉斯）进行参数调优。

**📊 数据集**

在 15 个 UCI 基准数据集（如 Banknote、Iris、Wine、MNIST 等）以及两组多模态合成数据（Clusters、Noisy Crescent Moons）上进行实验。

**📈 对比分析**

通过与 CKA、RFSC 以及留一交叉验证准确率的相关性比较，CS 在核宽度调优、核选择以及多模态数据上都表现出更高的相关性和更优的分类准确率，且对参数敏感性更低。

**⚠️ 局限性**

局限性包括：实验仅覆盖有限的核族和数据集；在极大数据规模下仍需分块计算；理论误差上界可进一步收紧；未在深度网络表示学习中验证 CS 的有效性。

---

## 175. Performance Portability of Out-of-Kernel Optimizations on GPUs: A Molecular Docking Case Study

**arXiv ID:** 2611.06207 | [PDF](https://arxiv.org/pdf/2611.06207v1)

**作者:** Gianmarco Accordi `[一作]` (Politecnico di Milano), Gianluca Palermo `[通讯]` (Politecnico di Milano)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对基于GPU的重复多核流水线，作者提出了一种以批处理策略、工作者并发度和每个工作者内存分配为显式优化参数的离核调优方法，并在分子对接mini‑app muDock 上进行验证。

**💡 创新点**

创新点在于用显式的内存上限而非占用率估计来确定批量大小，结合工作者并发和资源划分，形成一种跨后端可移植的执行策略。

**🔧 技术方法**

技术实现包括 manager/worker 执行模型、独立 CUDA/HIP/SYCL 流、显式资源分配、内存受限批处理（memory‑bounded batching）以及并发数据传输与计算重叠。

**📊 数据集**

实验使用来自 MEDIATE 数据集的 50 万个化合物样本，涵盖不同原子数和可旋转键数，体现工作负载的异质性。

**📈 对比分析**

通过与占用率驱动的基线配置对比（两工作者、占用率批量、显式同步），在 NVIDIA GH200、AMD MI250 与 Intel Max1550 的 CUDA/HIP/SYCL 后端上平均提升 1.53 倍吞吐量，最大提升可达 1.83 倍。

**⚠️ 局限性**

局限性包括对后端占用率估计的依赖、Intel SYCL 在 Max1550 上支持不完整，以及需要针对每个平台手动调节工作者数与内存分配，导致完整跨平台的性能一致性仍受限。

---

## 176. Cognitive Bias Signatures in Search-Equipped LLM Agents: A Stage-Resolved Analysis

**arXiv ID:** 2611.05720 | [PDF](https://arxiv.org/pdf/2611.05720v1)

**作者:** Yanru Xiao `[一作]` (Amazon Web Services), Yingwei Yu `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个受控搜索实验环境，研究LLM在不同起始条件（结果顺序、先前工作记录、初始假设）下的查询、检索、打开文档及最终答案行为。

**💡 创新点**

将锚定、沉没成本、确认偏差三种认知偏差范式与受控搜索 harness 相结合，能够分离初始证据顺序、工作历史和假设对搜索策略与答案的具体影响，并提供阶段性诊断。

**🔧 技术方法**

使用多模型（10个）LLM代理，搭配确定性关键词搜索和文档阅读工具，构建匹配对照实验并记录工具调用、检索结果、打开情况与最终答案；评估采用LLM评审判别。

**📊 数据集**

构造了1000个人工合成的实体关系世界，每个世界包含19–29个短文档，程序化生成查询和标签，避免真实知识偏差。

**📈 对比分析**

通过配对对照和阶段性指标（检索比例、切换率、错误采纳率、答案准确率）计算差异。结果显示：主动搜索中证据顺序对错误采纳率影响显著（+7.6%~+66%），死路工作削弱策略切换（3.6%~+17.4%），错误假设导致准确率下降（+43.5%）。干预“寻找反证”或“阅读双方”对部分模型可提升准确率，但提升效果不一致。

**⚠️ 局限性**

仅使用短合成文档和确定性关键词搜索，缺乏真实多样性；结果顺序与证据可得性混杂；评审主观性；缺少对复杂任务和动态搜索接口的验证；对部分模型（如Fable）因缺失数据导致结论不稳健。

---

## 177. Optimal Marginal Mixing for Transvection-Based Trapdoored Matrices over Large Fields

**arXiv ID:** 2611.05573 | [PDF](https://arxiv.org/pdf/2611.05573v1)

**作者:** Mark Braverman `[一作]` (Princeton University), Stephen Newman `[通讯]` (Princeton University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一类陷阱矩阵，这些矩阵是伪随机矩阵，能够在给定秘密密钥的情况下快速进行矩阵-向量乘法。

**💡 创新点**

提出了一种新的陷阱矩阵构造方法，基于短随机转置步行的矩阵生成，解决了转置步行的混合时间的开放问题。

**🔧 技术方法**

使用了转置步行的技术，通过在SL_n(𝔽_q)上进行随机转置操作来生成陷阱矩阵。

**📊 数据集**

使用了SL_n(𝔽_q)的矩阵数据集，特别是当n≥256时，提供了具体的上下界。

**📈 对比分析**

与现有方法进行了比较，证明了所提出的陷阱矩阵在混合时间和计算效率上具有优越性，尤其是在处理大规模数据时表现出色。

**⚠️ 局限性**

限制在于假设短步行的伪随机性，尽管在大多数情况下有效，但仍需进一步的加密分析以确保安全性。

---

## 178. Exact Transport-Information Correspondence for Distribution-Adaptive Discretization

**arXiv ID:** 2611.06059 | [PDF](https://arxiv.org/pdf/2611.06059v1)

**作者:** Koretaka Yuge `[一作]` `[通讯]` (Kyoto University), Koretaka Yuge (Kyoto University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

提出一种分布自适应离散化方案，并推导其在连续概率分布上精确的运输-信息对应关系。

**💡 创新点**

通过将单元内部质量移动到其概率加权质心，获得了不依赖单元形状、尺寸或无穷小尺度的完整公式，弥补了原始UCN框架的局限。

**🔧 技术方法**

利用信息几何（Fisher度量）、局部映射（Local‑Map）与运输成本（Wasserstein‑2）理论实现。

**📊 数据集**

论文为理论推导，无使用具体实验或数值数据集。

**📈 对比分析**

与原UCN方法对比，证明在更一般的离散化条件下仍保持一致性；未给出数值实验，仅通过解析式说明改进。

**⚠️ 局限性**

仍需预先知道连续分布的Fisher度量和单元条件协方差；对KL可解释性仅限局部展开；对非凸或重叠单元不适用。

---

## 179. Relevance Propagation as Entropic Optimal Transport on Filippov Neural Networks: Sinkhorn Propagation and Region Relevance in Image Classifiers

**arXiv ID:** 2611.05594 | [PDF](https://arxiv.org/pdf/2611.05594v1)

**作者:** Olivier Gibaru `[一作]` (Arts et Métiers Institute of Technology), Eric Nyiri `[通讯]` (Arts et Métiers Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并理论分析了一般化的 LRP_α,β 可解释规则以及 Sinkhorn 相关传播（SRP），并在 VGG‑16 等网络上通过 ABPC、relevance mass accuracy 等指标进行实验对比。

**💡 创新点**

创新性在于将 LRP 统一为权重分配框架，支持正负参考并兼顾梯度和 Kullback–Leibler 正则化，首次将 Sinkhorn 迭代与 SRP 等价化，提供对 ReLU/Max 分支连续性与跳跃的分析。

**🔧 技术方法**

采用 ReLU/Max 分支理论、梯度×输入、Kullback–Leibler 正则化的最小成本流、Sinkhorn 迭代、ABPC 与地区扰动等技术。

**📊 数据集**

使用 Imagenette 和 Oxford‑IIIT Pet 数据集的 VGG‑16 模型（ImageNet 预训练）进行实验。

**📈 对比分析**

通过 faithfulness（ABPC）、relevance mass accuracy、Spearman correlation ρ₂ 以及保留像素正确率等指标对 LRP_0、LRP_z^+、LRPN_2,-1、LRPN_α,β、SRP、SRP_1 等规则进行对比，结果显示 LRP_z^+ 和 LRPN_2,-1 在 faithfulness 和局部性上表现最佳，SRP_1 相对更快但精度略低。

**⚠️ 局限性**

局限包括对大规模网络的计算效率仍需改进、对负参考的处理需手工设定 α、β、在复杂非线性分支下仍可能出现不连续跳跃、对对抗扰动鲁棒性尚未充分验证。

---

## 180. Presentation Sensitivity in Typed Decision Models for Subjective Text Evaluation

**arXiv ID:** 2611.06384 | [PDF](https://arxiv.org/pdf/2611.06384v1)

**作者:** Jatin Agrawal `[一作]` (International Institute of Information Technology Hyderabad), Radhika Mamidi `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

审计了三种 typed decision 模型（Jev、Clef Flash、Perplexity Decider）在幽默、帮助度和摘要三任务上的呈现敏感性，收集 702k 评估结果并与人类参考对比。

**💡 创新点**

首次系统采用大规模 factorial 实验，验证位置、标识符、作者归属、读者认可、框架和评分格式对模型决策与评分的影响，揭示 typed decision 模型的呈现偏差及其与人类一致性的关系。

**🔧 技术方法**

使用 OpenRouter Decisions API 统一发送请求，结合概率、选择和分数的统计分析（平均概率移位、选择比例、误差指标、相关系数）以及 Benjamini–Yekutieli 多重检验校正。

**📊 数据集**

采用 UKP Humor、Humicroedit、HelpSteer2-Preference、HelpSteer2、OpenAI comparisons、SummEval 六个公开数据集，分别用于两选和打分任务。

**📈 对比分析**

通过对照实验与基线重复呼叫，对概率、选择和评分进行差异统计，发现 Jev 的位置偏差为负、Clef 接近零、Perplexity 任务相关；读者认可导致大比例选择反转；评分格式变动导致绝对分数偏移但排名高度稳定，表明 typed decision 模型高效但易受呈现影响。

**⚠️ 局限性**

仅覆盖三种模型与三项英文任务，缺乏与传统 LLM 判断器的直接对照；人类参考为聚合值，可能忽视评价者差异；实验基于单项问题，未涵盖更复杂上下文；结果依赖特定措辞和评分尺度，推广性有限。

---

## 181. Text-Only Domain Adaptation for Low-Resource Medical ASR: Shallow Fusion and Density Ratio Rescoring for Hausa

**arXiv ID:** 2611.05744 | [PDF](https://arxiv.org/pdf/2611.05744v1)

**作者:** Oche David Ankeli `[一作]` (African Leadership University), Bonaventure F. P. Dossou `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个仅使用文本的域适应框架，将多语言声学模型 MMS-1B-FL102 与浅融合（Shallow Fusion）和密度比（Density Ratio Approach）两种语言模型融合技术结合，用于 Hausa 医疗语音识别。

**💡 创新点**

创新点在于：1) 在无医疗语音数据的情况下，仅通过收集医疗文本实现 ASR 域适配；2) 首次将密度比方法应用于尼日利亚语言的医疗 ASR；3) 设计了医学术语错误率（MTER）指标和专用评测语料，提供了更细粒度的医学领域性能评估。

**🔧 技术方法**

使用技术包括：MMS 多语言声学模型、KenLM 4-gram 语言模型、浅融合与密度比两种解码融合策略、束搜索、pyctcdecode、以及对句子级交叉验证和句子集群重采样的统计评估。

**📊 数据集**

数据集：19,421 句 BBC Hausa 与 Deutsche Welle Hausa 医疗文本用于训练 4-gram 医学 LM；75,372 句公共 Hausa 文本用于训练源 LM；250 句医生审核的 25 个医学句子，10 名北尼日利亚说话者录制，用于评测；同时公开发布训练好的 LM 二进制和代码。

**📈 对比分析**

通过句子级 5‑折交叉验证保持提示分离，并用句子集群重采样估计 95% 置信区间。结果显示：贪婪解码 WER 为 29.56%；浅融合降低至 23.55%（20.3% 相对降低），密度比降低至 23.33%（21.1% 相对降低）；两者差异无显著统计意义；使用通用 LM 的浅融合仅比医学 LM 降低 0.41个百分点，表明大部分提升来自加入外部 LM。

**⚠️ 局限性**

局限性包括：评测规模仅 250 句，句子为人工朗读非自然对话；只评估提示级而非说话者级；未覆盖西北方言、老人或儿童说话者；4‑gram LM 只能捕捉局部统计，无法建模长距离依赖；源 LM 规模远小于声学模型预训练语料，可能限制 DRA 的优势；MTER 依赖词典且未考虑形态变体，不能完全证明医学域特有提升。

---

## 182. AQUA - A Modular Authentication Quality Modeling Framework

**arXiv ID:** 2611.05660 | [PDF](https://arxiv.org/pdf/2611.05660v1)

**作者:** Alex R. Mattukat `[一作]` (RWTH Aachen University), Krisa Carka `[通讯]` (RWTH Aachen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了 Aqua 框架，定义了认证器、认证器雇佣、认证技术和认证方案四个互相关联的概念，并基于此开发了初步的质量模型 Aqua‑A、Aqua‑T、Aqua‑S，使用易用性示例展示了质量传播路径。

**💡 创新点**

创新点在于将认证过程模块化拆分为四层，能够精确追溯质量影响来源（认证器属性、雇佣组合、技术实现）并为每一层构建独立的质量模型，从而突破了传统模型无法区分质量来源的局限。

**🔧 技术方法**

采用设计科学研究方法、正式符号化描述、ISO SQuaRE 质量模型框架，以及示例易用性指标（基于用户交互次数）进行演示。

**📊 数据集**

未使用任何实际数据集，论文纯粹为概念性框架与理论演示。

**📈 对比分析**

未进行实验性对比或性能评估；论文中仅提供示例计算，实际性能与评估结果留待后续工作。

**⚠️ 局限性**

局限性包括：质量模型仅涵盖少量因子和子因子、缺乏实证指标与评估、未验证模型的可衡量性、研究处于早期阶段且存在主观性决定。

---

## 183. A Dichotomy for Terminal Cyclability

**arXiv ID:** 2611.06234 | [PDF](https://arxiv.org/pdf/2611.06234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 184. NanoKV: High-Ratio KV Cache Compression for Autoregressive Video Generation

**arXiv ID:** 2611.06377 | [PDF](https://arxiv.org/pdf/2611.06377v1)

**作者:** Jiacheng Lu `[一作]` (Zhejiang University), Yu Li `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 NanoKV，利用冻结的视频生成模型进行高比率 KV 压缩，并可选地使用 NanoKV‑Fast 通过混合维度注意力进一步降低计算成本。

**💡 创新点**

创新点包括：①生成感知的压缩目标，关注压缩对生成结果的影响；②联合头‑通道残差编码，利用跨头冗余；③键和值按角色分别优化（键用块输出对齐，值用量化感知拟合）；④可复用压缩系数实现混合维度注意力。

**🔧 技术方法**

采用共享中心残差编码 + INT4 量化、块输出对齐与量化感知拟合、混合维度注意力映射、在线 K‑means、冻结模型训练等技术。

**📊 数据集**

训练与评估数据集包括 HY‑WorldPlay、Matrix‑Game 3.5、LingBot‑World 2.0 以及 VBench‑I2V；训练使用 48 条轨迹，评估使用 100 张图像、四个相机动作。

**📈 对比分析**

与 QVG、TurboQuant、KVTC 等基线比较；在 HY‑WorldPlay 上实现 26.09× 有效 KV 压缩、PSNR 提升 1.0 dB；NanoKV‑Fast 在保持 20.53× 压缩的同时，速度提升约 1.15×。

**⚠️ 局限性**

局限性：codec 适配需针对每个背骨训练；驻留算子对压缩率有限制，短历史下效果受限；混合维度近似在长时间生成中可能累积误差；未解决模型偏差与安全问题。

---

## 185. GraphSeal: Graph-Augmented Speech Deepfake Detection via Temporal-Spectral Artifact Propagation

**arXiv ID:** 2611.06171 | [PDF](https://arxiv.org/pdf/2611.06171v1)

**作者:** Lee Seung-woo `[一作]` (Pusan National University), Jang Won-young `[通讯]` (Hanyang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种名为GraphSeal的深度伪造语音检测框架，利用动态时间-频谱图对语音片段进行建模并通过图注意力传播伪造特征，再通过层次化读出生成句子级别判别向量

**💡 创新点**

创新点在于构建数据驱动的时频图结构，既考虑时间邻接也考虑频谱相似度，实现长程跨帧语义关联；并在图注意力中加入边属性（时间距离与频谱相似度），使模型能选择性聚合跨段伪造痕迹；最后使用层次化读出提升判别性能

**🔧 技术方法**

主要技术包括自监督语音特征提取（WavLM、wav2vec2.0、Whisper等）、动态kNN图构造、GATv2图注意力网络、边属性编码、层次化图池化与MELP分类头

**📊 数据集**

在四个公开基准上进行评估：ASVspoof 2019 LA、ASVspoof 2021 LA、ASVspoof 5 与 In-the-Wild（真实场景）

**📈 对比分析**

与传统基线（RawNet2、LCNN、AASIST、RawGAT-ST、SAMO、WavLM-AASIST 等）对比，GraphSeal 在所有数据集上取得最优或竞争性 EER：0.52%（2019 LA）、2.41%（2021 LA）、8.17%（ASVspoof 5）、5.24%（In-the-Wild），并通过 ablation 证明每个设计要素的贡献

**⚠️ 局限性**

局限性包括对图构造的 O(N²) 计算开销（尽管在典型 4 秒录音下可忽略）；对极短或极长语音的适配仍需研究；对高频谱均匀的神经编码器攻击（如神经语音编解码）效果下降；需要进一步探索流式图构造与多模态融合等方向

---

## 186. Reglos: A Self-Evolving Personal Memory Agent with Recursive Self-Improvement

**arXiv ID:** 2611.06049 | [PDF](https://arxiv.org/pdf/2611.06049v1)

**作者:** Sen Yang `[一作]` (Reglos.ai), Wen Liu `[通讯]` (Reglos.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种名为 Reglos 的个人记忆代理，采用在线自演进内存和离线递归自我改进的抓手，将用户对话转换为可追加记录并通过编码代理优化记忆利用策略。

**💡 创新点**

创新点在于将记忆的在线演进与离线抓手的递归自我改进联合起来，利用编码代理、批评者与固定评估器在无训练条件下实现记忆系统与抓手的共同进化；同时通过“追加‑保留、忘记‑墓碑”机制保持历史完整性。

**🔧 技术方法**

技术包括基于 Rust 的内存引擎、Python 抓手、批判性评估器（来自不同模型族的 Codex）、编码代理（Claude Code），以及递归自我改进循环与多路检索（向量、关键词、实体图）等。

**📊 数据集**

使用了 PersonaMem‑v2、LongMemEval‑S、LoCoMo、BEAM 等公开记忆评测数据集。

**📈 对比分析**

通过与 Mem0、MemOS、EverOS 等系统在同一问答模型（Qwen3.8‑27B 或 GPT‑5.6 Luna）下的准确率对比，Reglos 在 PersonaMem‑v2 取得 61.3%（+7.5 点），LongMemEval‑S 84.0%（+3.6 点），并在多路实验中持续领先。

**⚠️ 局限性**

主要局限在于仍受读取模型的推理限制，无法解决“未在对话中出现的事实”导致的错误；且对多模态支持和模型训练方面尚未展开。

---

## 187. UniCompress: From Multi-LoRA Experts to One Compact Model for Edge Intelligence

**arXiv ID:** 2611.06075 | [PDF](https://arxiv.org/pdf/2611.06075v1)

**作者:** Mingkang Liu `[一作]` (Juhaokan Technology Company Limited), Junyan Li `[通讯]` (Juhaokan Technology Company Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 UniCompress 两阶段框架，将多任务 LoRA 专家合并为单一 Qwen3‑VL‑4B BF16 模型，再进行 INT2 级联合量化与教师引导恢复，实现单模型部署。

**💡 创新点**

创新点在于将多任务专家知识一次性整合为统一教师模型，并在极低精度 INT2 下通过离线锚定与自生成轨迹的量化感知蒸馏，恢复多任务能力，同时消除多适配器管理成本。

**🔧 技术方法**

采用 LoRA 合并、GPTQ 预量化、异步 INT2 量化、教师引导的离线锚定、基于漂移加权的自回放蒸馏、shadow policy 量化敏感度估计等技术。

**📊 数据集**

使用四个在设备端的自研数据集：姿态监测 (SPM)、家庭安全监测 (HSM)、电视/摄像头查询路由 (TCR)、移动 GUI 控制 (MGC)。

**📈 对比分析**

与 Multi-LoRA Experts (约 9.1 GB) 与 GPTQ‑INT2/INT4 基线比较，UniCompress‑INT2 在 2.1 GB 的重量下宏平均分数 92.8%，比 GPTQ‑INT2 提升 34.3 点，几乎等同于 BF16 统一模型，且显著优于 INT4。

**⚠️ 局限性**

局限在于对 Qwen3‑VL‑4B 及四个任务的依赖，极低精度仍需昂贵的离线/自回放训练，且对更大模型或不同 VLM 的迁移尚未验证。

---

## 188. CLEAR: Color-Texture Latent Alignment for Single Image Reflection Removal

**arXiv ID:** 2611.05969 | [PDF](https://arxiv.org/pdf/2611.05969v1)

**作者:** Tu Vo `[一作]` (KC Machine Learning Lab), Chan Y. Park `[通讯]` (KC Machine Learning Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为CLEAR的单帧单图像反射去除方法，利用预训练的潜在扩散模型进行自然图像先验，并通过两阶段流程实现精细的层分离与细节恢复。

**💡 创新点**

创新点在于①使用物理引导的潜在分离器在单步内同时预测透射层、反射层和残差；②开发Hybrid Feature Alignment Module（HFAM），通过色彩门控注意力将混合跳过特征对齐至干净透射统计，从而抑制反射细节并纠正颜色漂移。

**🔧 技术方法**

采用的技术包括预训练Stable Diffusion VAE与U-Net、跨潜在注意力、Haar离散小波变换、通道注意力、颜色描述子、以及对齐监督的损失函数。

**📊 数据集**

在六个公开基准上进行评估，数据集包括Nature、SIR^2、Real20、DRR-C、DRR-S和OpenRR-1k，并使用与DAI相同的训练数据进行公平比较。

**📈 对比分析**

与现有单步基线（如DAI、WindowSeat、DSRNet等）相比，CLEAR在PSNR、SSIM、LPIPS、MANIQA和QUALICLIP等多项指标上均实现最高平均分，尤其在最难的DRR-C、DRR-S和OpenRR-1k上取得显著提升。

**⚠️ 局限性**

局限性包括：①仅针对单图像输入，无法利用多图或额外光学信息；②在压缩潜在空间下高频细节受限，HFAM虽缓解但不能完全恢复；③在反射几乎完全遮蔽或透射极度衰减时可能产生假象。

---

## 189. SWEOps-Bench: Evaluating Agents for End-to-End Resolution of Software-Induced Cloud Incidents

**arXiv ID:** 2611.06099 | [PDF](https://arxiv.org/pdf/2611.06099v1)

**作者:** Danwen Chen `[一作]` (Sun Yat-sen University), Pengfei Chen `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了SWEOps-Bench，一个交互式基准，用于评估自主代理在Kubernetes Operator故障中的端到端故障检测、定位与修复。

**💡 创新点**

创新点在于将真实历史软件缺陷重现于完整的Kubernetes集群环境中，结合可执行的故障探针与功能回归测试，评估从检测到修复的完整生命周期，而非仅局限于单一阶段。

**🔧 技术方法**

技术包括LLM驱动的终端代理（Codex、Claude Code）与四个模型（GPT‑5.6 Luna、Claude‑Sonnet‑5、DeepSeek‑V4.1‑Flash、GLM‑5.3‑Flash），以及 Harbor 框架用于环境管理。

**📊 数据集**

数据集由16个开源Kubernetes Operator 项目的55个历史缺陷任务组成，涵盖构造、输入验证、控制逻辑等多种缺陷类型。

**📈 对比分析**

通过对六种 agent–model 配置进行实验，测量检测、RCA、修复成功率、时间延迟与 token 消耗，发现检测率可达 90% 但整体恢复率仅 14.5%–52.7%；不同配置在效率与准确性上表现不一，token 消耗与恢复率不必然相关。

**⚠️ 局限性**

限制包括单次执行未评估随机性、任务量有限导致子组统计不稳、只关注 Kubernetes Operator 领域、未覆盖硬件或非容器化故障、并且对验证的动态测试仍可能遗漏边缘行为。

---

## 190. CARM: Conformal Adaptive Risk Masking for Reliable Scientific Super-Resolution

**arXiv ID:** 2611.06151 | [PDF](https://arxiv.org/pdf/2611.06151v1)

**作者:** Guangtao Zhang `[一作]` (SandGold AI Research), Zhiqiang Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在科学超分辨率任务中，提出CARM框架，利用可观测的空间不确定度在线自适应地构建可信掩模；

**💡 创新点**

创新点在于：①引入多速率专家和基于成熟反馈的因果更新；②采用不对称损失聚合实现快速适应与稳定平衡；③在时序分布漂移下仍能保证平均风险不超过目标水平；

**🔧 技术方法**

技术上基于 conformal risk control、超分辨率模型的局部不确定度估计（集成方差）以及多速专家的在线聚合；

**📊 数据集**

使用南海叶绿素‑a 流、ERA5 风速、OSTIA 海表温度等海洋/气象数据集，并在 FNO、MWT、SRNO、HiNOTE、SwinIR、EDSR、ResShift 等多种 SR backbone 上验证；

**📈 对比分析**

与静态 CMC 及固定速率专家对比，CARM 在平均风险、违规率和七天累计超限等关键指标上显著降低（如违规率从 43% 降至 10%），同时保持可用区域；在不同数据集和 backbone 上均表现出一致的改进；

**⚠️ 局限性**

局限性包括：依赖可靠的不确定度排序，若不确定度与误差不匹配会影响效果；反馈延迟限制了对突发分布变动的快速响应；严格的风险控制可能导致可信区域过小。

---

## 191. Routes of Reasoning: Reading and Reshaping Test-Time Compute in MoE Models

**arXiv ID:** 2611.05919 | [PDF](https://arxiv.org/pdf/2611.05919v1)

**作者:** Shengjie Wang `[一作]` (University of California, Los Angeles), Nanyun Peng `[通讯]` (University of California, Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了稀疏Mixture‑of‑Experts（MoE）模型的路由（router）是否能揭示模型在推理过程中执行的不同行为，并利用路由信息在推理时进行无训练的干预，从而减少“过度思考”导致的冗余计算；

**💡 创新点**

创新点在于：①将六类推理行为（规划、推导、重复、事实回忆、验证、总结）与路由状态关联，证明路由层可以精准读出行为；②证明该关联在不同任务（数学、代码、科学、知识、常识）间可迁移；③利用行为相关专家集在推理时激活/抑制实现对计算量的控制，显著减少标记生成量；

**🔧 技术方法**

核心技术包括：MoE模型的稀疏路由机制、线性探针（probe）对路由logits和残差流的分类评估、基于风险差异的专家行为关联挖掘、推理时路由修改的激活/抑制操作；

**📊 数据集**

使用的公开数据集包括：MATH500、AIME、LiveCodeBench、GPQA‑Diamond、AICrypto、CommonsenseQA、MMLU‑Pro等，覆盖数学、代码、科学、知识、常识等五个领域；

**📈 对比分析**

与未干预模型、随机专家干预以及“请保持推理简洁”的提示基线对比。实验在四个MoE模型（Gemma‑4、两代Qwen3、GPT‑OSS）和四个基准上验证，平均减少约30.4%标记（最高可达63.3%），准确率最多下降0.5个百分点，且在多种设置下可提升准确率；

**⚠️ 局限性**

限制点包括：需要手工标注的推理行为数据；仅在目前测试的几类推理任务和四个MoE模型上验证；干预仅抑制或激活专家，无法对更细粒度行为进行微调；在更长时序或多代理任务中的有效性尚未评估。

---

## 192. Securing Web Agents with the Compile-Then-Execute Paradigm

**arXiv ID:** 2611.06039 | [PDF](https://arxiv.org/pdf/2611.06039v1)

**作者:** Annabella Chow `[一作]` (University of California Berkeley), Julien Piet `[通讯]` (University of California Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Baya，一种基于编译-执行（compile‑then‑execute）范式的 Web 代理，利用可信网站 API 预先生成可执行程序，随后在隔离环境中执行，从而防止页面注入导致的控制流劫持。

**💡 创新点**

创新点在于：①将 Web 代理的规划与执行完全分离，利用可信 API 先编译完整的程序，保证运行时仅处理数据而不修改控制流；②设计了强类型的 Trusted Website API 合约，统一不同网站的 REST 与 UI 操作；③在编译阶段使用 LLM 生成可执行 Python 代码并通过本地 mock 进行验证，确保程序在不接触真实网站前即可保证结构正确；④通过受限错误诊断实现安全的重新规划。

**🔧 技术方法**

使用技术包括：大型语言模型（Claude、Gemini、DeepSeek 等）进行代码合成；Python 代码生成与单元测试；可信 REST/Playwright 接口包装；沙箱化的编译和执行流程；结构化的 API 合约（签名、前置条件、后置条件、错误域）。

**📊 数据集**

实验数据集：WebArena Verified（581 个任务，覆盖 GitLab、Shopping、Shopping Admin、Postmill）和 WASP（针对注入攻击的 84 个实例）。

**📈 对比分析**

与 ReAct 基线（浏览器+API 版本）和 Untrusted Content Masking（UCM）对比：Baya 在 581 任务上实现 81.9% 的成功率（高于 59.0% 的 ReAct），并在 WASP 攻击测试中保持 0% 的中间/端到端攻击成功率；但其平均延迟和 Token 消耗约为 ReAct 的 3‑5 倍，成本也相应更高；相比 UCM，Baya 在全任务集上显著提升效能，却保持同等的安全性。

**⚠️ 局限性**

局限性：①编译阶段产生的额外计算量导致更高的时间和成本；②需要为每个网站预先构建可信 API，维护成本高；③仅支持控制流可预先确定的任务，无法处理需在运行时根据数据动态决定动作的“Replan‑Needed”任务；④对实时网页变化的适应性弱，适用于结构稳定或可预先定义的工作流。

---

## 193. Attributing Agent Failures to Their History Needs an Executed Counterfactual: Toward a Reporting Standard

**arXiv ID:** 2611.05650 | [PDF](https://arxiv.org/pdf/2611.05650v1)

**作者:** Egor Pakhomov `[一作]` (Salesforce AI Research), Erik Nijkamp `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对长周期代理失效解释方法进行系统综述，提出需在冻结可执行状态下进行历史干预并使用可测的重放证书，归纳四类历史操作，并给出两项警示性实验结果，进一步提出改进议程。

**💡 创新点**

创新点在于：①将因果推理中的干预证据引入代理历史解释，要求在可恢复状态下对历史做实验；②定义并验证“重放证书”（pathwise和distributional）以确保解释的可复制性；③系统划分四类历史操作并检视现有工作是否满足干预和重放标准；④通过实验揭示哈希相同快照可能不完全可重放。

**🔧 技术方法**

主要技术包括：沙箱检查点快照恢复、记录与重放（record-and-replay）、随机种子同步、对照实验设计、置信区间与比值比计算、基于Causal Agent Replay的交互式重放实现。

**📊 数据集**

使用的数据集包括ActBench（约24,000轨迹的安全基准）、AgentCE-Bench（24个因子状态的实验）、以及对Min et al.、Who&When Pro等公开论文所用的原始轨迹集合。

**📈 对比分析**

比较方法：对现有论文中的失效标签、重放一致率、干预验证缺失等进行量化评估；实验中测得哈希相同快照在未记录随机流位置时的重放偏差，使用重放一致率（0%–100%）和odds ratio等指标；总体而言，现有方法普遍缺乏可测重放证书，导致因果解释不可靠。

**⚠️ 局限性**

局限性包括：①大多数现有研究未报告重放证书，本文评估基于已有论文公开数据，缺少新实验验证；②测量3.1被判定不可评估，测量3.2缺乏模型驱动行为；③对快照的随机流恢复仅在特定条件下验证，未覆盖所有可能的随机依赖；④未提供完整的实验复现代码，重现性受限。

---

## 194. Deterministic Network Interface Naming on Multi-NIC Linux Systems: A Boot-Time Convergence Approach

**arXiv ID:** 2611.05387 | [PDF](https://arxiv.org/pdf/2611.05387v1)

**作者:** Chetan Nigudgi `[一作]` `[通讯]`, Chetan Nigudgi

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文提出了一种基于 systemd 的三层架构（声明式配置、内核参数管理、启动时收敛服务），通过在多网卡 Linux 系统中实现启动时接口命名收敛，解决了 udev 事件并行导致的命名竞争问题。

**💡 创新点**

创新点在于：① 通过将 .link 和 .network 文件排序至系统文件前，保证预期网卡优先匹配；② 在启动时加入自适应重试机制，主动触发 udev 并等待 DHCP，直至网卡命名收敛；③ 在任何 systemd‑based 发行版中无需内核或 systemd 补丁即可实现 100% 可确定性命名。

**🔧 技术方法**

主要技术包括 systemd 的 .link/.network 声明文件、GRUB 内核参数调整（移除 conflict 参数）、systemd 触发器 udevadm、oneshot 收敛服务、DHCP 状态监测和循环重试逻辑。

**📊 数据集**

实验使用多厂商、多型号网卡（Intel、Broadcom 等）在物理服务器和虚拟机上进行，覆盖了 3+ 代内核与 250+ 个测试实例，未提供公开数据集。

**📈 对比分析**

与仅使用 udev/.link 规则的基线相比，收敛方案在首次启动时成功率提升至 100%，失败率降为 0%；当出现竞争时，重试路径平均耗时 35-45 秒，整体启动时间提升约 3-5 分钟，已被生产环境接受。

**⚠️ 局限性**

局限性包括：若竞争仍出现需额外 30-45 秒延迟；需要在镜像创建时预知网卡 MAC；方案仅对目标网卡保证确定性，其他网卡仍采用可预测命名；并未解决根本的内核级命名原子性缺失问题。

---

## 195. Quadratic reset bounds for finite maximal prefix codes

**arXiv ID:** 2611.05675 | [PDF](https://arxiv.org/pdf/2611.05675v1)

**作者:** Samrath Singh Chadha `[一作]` `[通讯]`, Samrath Singh Chadha

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究有限最大前缀码的字面解码器在同步时最短重置词的长度上界和下界，给出以树深度k为参数的二次多项式上界，并在两种相邻长度的码中给出精确上界；构造了极值族C_k证明该上界可取并给出对应最短重置词；同时提出了多项式时间算法实现重置词的构造；将主要定理形式化验证在Lean中。

**💡 创新点**

提出“短检测定理”，利用线性观察在树深度限定下保证存在长度≤2k-3的单词使得观察值非零；结合预像扩张和旋转，得到统一上界(k−1)(4k−5)；对相邻长度码给出精确上界2r^2−1，并证明唯一极值族；实现了正式验证和多项式时间算法；构造了“年龄”与“潜能”分析实现下界。

**🔧 技术方法**

主要技术包括：非可交换变量的首次回归分解；在分部域中的嵌入与分母约简；线性代数（矩阵表示、行列式估计）用于构造短观察子空间；预像扩张与旋转平均法；潜能函数与圆弧覆盖分析；形式化证明与Lean验证；多项式时间算法实现。

**📊 数据集**

研究对象为所有有限最大前缀码（通过显式列表给出），没有使用标准公开数据集；主要关注理论性质与算法实现。

**📈 对比分析**

与以往的Černý式通用同步上界（O(n^3)）和专门的前缀码同步结果比较，本文给出仅依赖树深度的二次上界，构造的重置词长度≤(k−1)(4k−5)；在两相邻长度码中进一步实现精确上界2r^2−1；算法在给定码的显式列表下多项式时间完成，输出重置词长度与理论上界相符。

**⚠️ 局限性**

目前仍存在理论与实际上界的差距：一般同步码是否满足s(C)≤2(k−1)^2−1仍未证明；短检测定理的证据依赖特定终点深度假设，限制了可推广性；对树深度以外的结构因素分析不足，导致对非相邻长度码的精确分析尚未完成。

---

## 196. Faster Isomorphism for Groups of Class 2

**arXiv ID:** 2611.05920 | [PDF](https://arxiv.org/pdf/2611.05920v1)

**作者:** Xiaorui Sun `[一作]` `[通讯]` (University of Illinois at Chicago), Xiaorui Sun (University of Illinois at Chicago)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种用于测试二类群同构的算法，运行时间为N^O((log N)^(25/26) log log N)。

**💡 创新点**

创新点在于首次为一般的二类群提供了N^o(log N)时间复杂度的同构测试算法，突破了之前的限制。

**🔧 技术方法**

结合了张量基础的搜索空间限制、交替矩阵元组的群关系编码和轨道计算来恢复剩余的中心幂信息。

**📊 数据集**

使用了二类群的Cayley表作为数据集，特别关注于有限的p-群和Frattini类群。

**📈 对比分析**

与现有方法相比，提出的算法在时间复杂度上有显著改进，能够在更短的时间内解决二类群同构问题，性能优于之前的算法。

**⚠️ 局限性**

算法的局限性在于仍然依赖于特定的群结构，可能无法推广到所有类型的群同构问题。

---

## 197. Sundance: A Simple SMT Solver for Program Verification

**arXiv ID:** 2611.05668 | [PDF](https://arxiv.org/pdf/2611.05668v1)

**作者:** Amar Shah `[一作]` (Carnegie Mellon University), Bruno Dutertre `[通讯]` (Amazon Web Services)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个用 Rust 编写、约 13.6K 行的轻量级 SMT 解决器 Sundance‑SMT，专门服务于程序验证工具。

**💡 创新点**

创新点在于：1）以极简设计实现了可证明性（eDRAT 证明）和高度可插拔性；2）用标准接口替代了传统 SMT 解决器的紧耦合结构；3）用少量 heuristics 仍能接近 Z3、cvc5 的性能。

**🔧 技术方法**

核心技术包括 CDCL(T) 框架、IPASIR-UP 接口 SAT 求解器、基于签名表的 e‑graph（包含合取闭包与 e‑matching）、模式匹配的量化实例化、基于 simplex 的线性整数/实数算术以及 eDRAT 证明生成。

**📊 数据集**

使用 Verus（开发集 2,576 条、保留集 4,086 条）、Dafny（11,674 条）以及 DICE^⋆、vWasm（共 3,195 条）等公开验证基准进行评测。

**📈 对比分析**

在 Verus 保留集 60 秒内解决 90.2%（cvc5 95.3%，Z3 97.7%），在 Dafny 60 秒内 73.7%（cvc5 69.8%，Z3 66.5%），整体行数比 Z3、cvc5 低 26–38 倍，证明生成几乎无性能损耗。

**⚠️ 局限性**

局限性包括：对极大规模查询（如 DICE^⋆）的量化匹配效率不如 Z3 的 e‑matching；缺乏高级量化启发式（相关性传播、垃圾回收）和对位向量、非线性、字符串等理论的完整支持；在模块替换（Semper e‑graph、MiniSat）时性能会下降。

---

## 198. CrossPressor: Scalable Cross-View Fusion for Feed-Forward 3D Gaussian Splatting

**arXiv ID:** 2611.06258 | [PDF](https://arxiv.org/pdf/2611.06258v1)

**作者:** Yuxi Hu `[一作]` (Graz University Of Technology), Friedrich Fraundorfer `[通讯]` (Graz University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种无训练、可扩展的 Anchor‑based 融合方法 CrossPressor，解决 3D Gaussian Splatting 在长视角序列中表示与融合规模不匹配的问题。

**💡 创新点**

创新点在于引入覆盖循环证据路由（coverage‑cycled evidence routing），在保持锚点查询不变的同时，通过在视角与层之间交错采样，限制支持 K/V 序列长度，从而使支持侧融合成本不随视角数增长。

**🔧 技术方法**

使用的技术包括：Anchor‑based 组合（ZPressor 结构）、先行 K/V 路由、跨层跨视角的采样偏移策略、预打包 (pre‑packing) 路由上下文、以及无训练的 Token‑reduction 机制。

**📊 数据集**

主要实验数据集为 RealEstate10K（RE10K）和 ACID，模型基于 MVSplat‑ZPressor、DepthSplat‑ZPressor 的预训练权重。

**📈 对比分析**

与原 Dense 融合、ToMe、Zero‑TPrune 等方法比较，CrossPressor 在 256 视角下实现了 26.12× 的交叉视角融合加速和 1.43× 的端到端加速，PSNR、SSIM、LPIPS 等质量指标几乎无下降（p5 ΔPSNR ≈ –0.006 dB），同时保持低交互比例（≈1.85%）。

**⚠️ 局限性**

局限性包括：系统级加速受限于融合之外的计算；对极端视角分布或 stride 选择不当时可能略有质量波动；方法仍需依赖 Anchor‑based 的预先分组和采样策略，未能完全消除所有支持侧成本。

---

## 199. Fast Quantum Byzantine Agreement with Nearly Linear Quantum Communication

**arXiv ID:** 2611.05598 | [PDF](https://arxiv.org/pdf/2611.05598v1)

**作者:** Jan Olkowski `[一作]` `[通讯]`, Jan Olkowski

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种面向同步二元拜占庭协议的量子算法，在全信息自适应攻击者下实现一致性与有效性，并给出了量子与经典通信的期望上界。该算法在 t ≤ (1/3 – ε)n 条件下，期望完成 O(log⁴ n) 轮，使用 O(n log⁸ n) 个量子比特、O(n² log⁸ n) 个经典比特。

**💡 创新点**

创新点包括：
• 将 King‑Saia 的选举树结构与量子秘密共享相结合，利用可消除的量子共享（disposable shares）在多级委员会之间安全地转移隐藏的随机字。
• 引入压缩（packing）技术，将多份共享合并为单一块，避免每级传输导致共享数呈指数增长，从而实现近线性量子通信。
• 在上述结构上给出完整的隐私与重构证明，并证明在自适应攻击者即使抑制途中消息时仍能保持安全。
• 将 Abraham 等人的消息抑制下的二次下界扩展到量子通道，得到 Ω(t²) 的总通信下界，证明在 t=Θ(n) 时通信与轮数的上界是最优（仅多项式对数级别）。

**🔧 技术方法**

技术手段主要包括：
• 选举树与 Feige 的最轻桶（lightest‑bin）规则，用于在多级委员会中挑选随机源。
• 基于 Reed–Solomon 的量子秘密共享与可消除共享的编码器 V_{D,σ,p}，保证隐私与可纠错。
• 可消除共享的递归重编码与压缩（packing）实现。
• 通过公开的双向图实现稀疏传输，保证每级仅产生 O(log n) 条边。
• 通过递归的安全性与重构证明，结合离散对数与概率工具完成整个协议的安全分析。

**📊 数据集**

该工作完全为理论性质，未使用任何实验数据集，所有结果均通过数学证明给出。

**📈 对比分析**

相较于 Ben‑Or & Hassidim（STOC 2005）提出的常数轮协议，量子通信量从 Ω(n⁴ log n) 降至 O(n log⁸ n)，经典通信仍为 O(n² log⁸ n)。
相较于经典的随机化协议，通信量保持了与已知的二次下界一致（仅差多项式对数因子）。
该协议在轮数上从 O(1) 提升至 O(log⁴ n)，但仍为多项式对数，适合大规模网络。

**⚠️ 局限性**

局限性包括：
• 需要同步、可靠且受认证的点对点通道，且不存在签名或先验共享量子态。
• 协议中的常数因子取决于 ε（即容忍度），在实际实现时可能较大。
• 在自适应攻击者可以抑制消息的模型下，仍需多轮交互，导致实现复杂度较高。
• 设计中使用穷举搜索确定委员会分配，导致本地计算上可能不是多项式时间。
• 协议目前只针对同步模型，对异步网络或更强的加密攻击尚未覆盖。

---

## 200. Learning When to Think Aloud in Latent Reasoning

**arXiv ID:** 2611.06108 | [PDF](https://arxiv.org/pdf/2611.06108v1)

**作者:** Zhehao Zhou `[一作]` (Beijing Institute of Technology), Xueqi Cheng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种混合推理框架“Verbalizing Informative Tokens as Reasoning Anchors for Latent reasoning”，通过在连续隐藏状态中进行大部分推理，仅在关键时刻生成少量显式 token 作为推理锚点；

**💡 创新点**

核心创新在于：① 轻量化门控头动态决定是否生成显式 token；② 使用 LLMLingua-2 评估 token 重要性，在每个推理步骤中挑选 top‑γ 重要 token 作为锚点；③ 结合阶段性监督训练与 RL 微调，平衡推理准确性与简洁度；

**🔧 技术方法**

技术手段包括：基于 transformer 的语言模型加门控头、重要性估计（LLMLingua-2）、混合符号‑隐式轨迹构造、两阶段训练（Curriculum SFT + Group Relative Policy Optimization）、噪声鲁棒性实验；

**📊 数据集**

主要使用的基准数据集有 GSM8K‑Aug、SVAMP、GSM8K‑Hard、MultiArith、Qwen3‑1.7B 相关自然语言数学推理集、MATH、ARC‑C 等；

**📈 对比分析**

与全隐式推理基线（Coconut, CODI, CoLaR, SIM‑CoT）以及混合推理基线（Token Assorted, HRPO, HybridCoT）对比，实验显示该方法在 GPT‑2、Llama‑3.2‑1B、Qwen3‑1.7B 等模型上在保持或缩短推理长度的同时，平均准确率提升 9.6%~5.1%，RL 版本进一步提升；

**⚠️ 局限性**

局限性包括：RL 训练仅在较小模型上完成，未扩展至更大规模；仅利用原始 CoT 中“重要”token 作为锚点，未探索其他锚点形式；在高风险应用中显式 token 较少，可能降低可解释性。

---

## 201. DecoupAttn: Decoupled Sparse Attention for Training-Free Acceleration of Video Diffusion

**arXiv ID:** 2611.05917 | [PDF](https://arxiv.org/pdf/2611.05917v1)

**作者:** Yanjin Li `[一作]` (University of Illinois), Mingda Zhang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 DecoupAttn，一种将精确计算和近似分离的稀疏注意力框架，在线无额外块识别、仅使用离线固定块并通过语义聚类补偿其余键，结合 Distortion Budget 进行预算分配与 Reassign Cache 维护键漂移；

**💡 创新点**

创新点在于将块选择与补偿完全解耦，使得稀疏块可以离线预设、无需每步动态路由；采用 Distortion Budget 以率失真目标对每个单元的残差进行精确计价；Reassign Cache 在每步重新分配键至缓存质心，防止残差随步长累积；

**🔧 技术方法**

核心技术包括：离线块固定（Fixed Mask）、语义聚类补偿（Semantic Centroid Compensation）与二阶校正、率失真预算分配、键重分配缓存（Lloyd 算法）以及 FlashAttention-3 等高效块稀疏核；

**📊 数据集**

在十个视频生成/控制生成/神经渲染模型上评估，涵盖 Cosmos-Predict1/2.5、Wan2.1/2.2、HunyuanVideo、Cosmos3-Nano、T2.5、DiffusionRenderer、AlayaRenderer、UniRelight 等多种数据集与提示；

**📈 对比分析**

与七种稀疏注意力基线（CalibAtt、SVOO、SVG2、Sol-Attn、SVG-EAR、PISA、SparsePR）以及稠密注意力对比，DecoupAttn 在九/十个模型上均实现最高 PSNR（平均提升约2.18 dB）且速度比稠密模型快 1.3–1.65 倍；

**⚠️ 局限性**

局限性包括：离线校准需要数小时 GPU 时间，且对新模型/大分辨率需要重新编译计划；固定块策略可能在极端动态内容下导致残差累积，虽然 Reassign Cache 可缓解但仍需额外计算；

---

## 202. GaussianRenderFormer: Transformer-based Rendering of 3D Gaussian Splats

**arXiv ID:** 2611.05593 | [PDF](https://arxiv.org/pdf/2611.05593v1)

**作者:** Shahaf Valencia Levy `[一作]` (Hebrew University of Jerusalem), Sagie Benaim `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个Transformer网络——GaussianRenderFormer，用来直接渲染给定的3D高斯球点集，而不使用传统光栅化算法。

**💡 创新点**

创新点包括：把每个高斯球作为一个token；在视点解码器中加入二维旋转嵌入，告知每个高斯在图像中的投影位置与投影轮廓；使用更细的4×4像素补丁；通过在attention键中注入投影轮廓信息，显著提升渲染精度。

**🔧 技术方法**

采用了Transformer结构（自注意力、交叉注意力、DPT头），结合3D Gaussian splatting的参数表示、二维旋转嵌入、投影轮廓特征，并在训练中使用log‑L1+LPIPS损失。

**📊 数据集**

使用了Objaverse_Splats数据集（来自ShapeSplat）——26,820个物体的50,000个高斯球，通过自研剪枝+微调得到20,000个高斯球的训练集，1,806个验证集。

**📈 对比分析**

与基准光栅化器（gsplat）比较：在未剪枝的完整渲染上得到38.08 dB PSNR，而光栅化同一剪枝输入得到44.02 dB；相对全分辨率完整渲染的差距约5–6 dB，显示出显著性能差距。

**⚠️ 局限性**

局限性包括：速度慢（单视图275 ms，远超光栅化），显存占用大，训练依赖RenderFormer预训练权重，模型仅在单一摄像机配置下表现良好，无法渲染全场景，且仅使用视图无关颜色，无法实现重光照或全局照明。

---

## 203. FLINT: Coupling Proximal Initialization and Bounded Stochasticity for Flow-Matching Inverse Problems

**arXiv ID:** 2611.05945 | [PDF](https://arxiv.org/pdf/2611.05945v1)

**作者:** Junseo Bang `[一作]` (Seoul National University), Se Young Chun `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 FLINT 两阶段框架，先用“流匹配模型”的 manifold‑constrained proximal 目标快速得到接近后验最优的噪声起点，然后在后续的 guided posterior 采样中采用边界化的随机性调度，保持起点优势并提升重建质量。

**💡 创新点**

创新点在于：①揭示初始化接近度与采样随机性之间的本质耦合，并给出统一的稳定性上界；②发现“足够近”区域（sufficiency regime）即可获得高质量重建；③提出“bounded stochasticity”调度来防止随机性累积抹消初始优势，从而在不进行昂贵源空间优化的情况下实现高质量后验采样。

**🔧 技术方法**

采用了流匹配模型（rectified flow）、截断采样+SDO 速算、谱加权数据一致性（Wiener 滤波）、高阶 cumulant 正则化、以及针对时间的 bell‑shaped 随机性调度。

**📊 数据集**

在 FFHQ（1024×1024，1000+样本）和 DIV2K（2048×1024，800+样本）上评估，针对四类线性逆问题：SR×8、SR×12、运动去卷积、Gaussian 去卷积。

**📈 对比分析**

与 FlowChef、FlowDPS、FLAIR、Flower、FlowLPS（流匹配基线）以及 PSLD、ReSample、DDPG、TReg（扩散基线）对比；FLINT 在 PSNR、SSIM、FID、LPIPS 上均超过或竞争所有基线，并在相同或更少的 NFE（如 30/50）内实现更高的采样效率。

**⚠️ 局限性**

受限于预训练生成模型的先验假设，难以处理结构极其复杂或多实例、明显 OOD 的图像；对高 ill‑posed 场景的重建效果仍有下降。

---

## 204. VIPBench: A Human-Aligned Benchmark for Voice Identity Perception in the Age of Voice Cloning

**arXiv ID:** 2611.05638 | [PDF](https://arxiv.org/pdf/2611.05638v1)

**作者:** Xudong Tang `[一作]` (Northwestern University), Matthew Groh `[通讯]` (Northwestern University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了VIPBench，一个包含9,800对语音样本、124,876个听众同/异身份判断的数据集，用以评估声纹模型与人类听众对语音身份的感知一致性。

**💡 创新点**

创新点在于首次大规模收集人类对真实、AI克隆和音频混合语音身份的感知标注，并设计四个任务评估模型与人类感知的对齐程度。

**🔧 技术方法**

使用了多种声纹模型（如ECAPA‑TDNN、RawNet3、Resemblyzer、wav2vec 2.0、HuBERT、WavLM、XLS‑R、Whisper）以及基于余弦相似度的计算和Platt校准方法。

**📊 数据集**

数据集VIPBench涵盖100位美国公众人物的录音、Cartesia Sonic 3生成的AI克隆语音以及8,100个连续混合语音，配合1,290名听众的二分类判断。

**📈 对比分析**

在四个评估任务中，监督训练的声纹模型表现最好，最高相关系数为0.656（相当于噪声上限的61%），而自监督模型和Whisper的性能相对较低，说明仍有提升空间。

**⚠️ 局限性**

局限包括仅覆盖美国英语公众人物、克隆语音单一合成系统、缺乏非公开人物和多语言环境，且未验证模型对个体听众的可重复性。

---

## 205. Whose Transparency is it Anyway? Understanding Generative AI Transparency Needs and Gaps Across Stakeholders

**arXiv ID:** 2611.05549 | [PDF](https://arxiv.org/pdf/2611.05549v1)

**作者:** Genevieve Smith `[一作]` (Stanford University), Min Kyung Lee `[通讯]` (University of Texas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过全球400名终端用户、AI构建者与政策制定者的问卷调查及对终端用户的焦点小组访谈，系统分析了不同利益相关者对生成式AI透明度的理解、需求与实际可得性差距，揭示了透明度缺口与权力关系的关联；

**💡 创新点**

创新点在于将社会技术建构（SCOT）的可解释灵活性与权力维度相结合，首次跨群体比较生成式AI透明度的认知与实践差异，提出多元化透明度基础设施与参与式设计的策略；

**🔧 技术方法**

采用问卷调查与结构化焦点小组讨论两种定量与定性方法，利用重要性-可用性差距分析（基于Likert与二元可得性评分）及情景情境对比分析；

**📊 数据集**

数据集为自编问卷的400份数据，覆盖250名终端用户、94名AI构建者、56名政策制定者，及一次约20人的终端用户焦点小组数据；

**📈 对比分析**

比较方法主要是对三组重要性评分与可用性比例进行差距计算，并对情景下特征优先级进行交叉对比；性能表现上发现终端用户对透明度需求与可得性差距最大（平均差距≈35），构建者和政策制定者差距显著较小；

**⚠️ 局限性**

局限性包括：可得性基于主观感知而非客观测量，问卷与焦点小组样本规模及代表性有限，交叉时间点研究无法捕捉透明度实践的动态变化。

---

## 206. ProofStudio: Five Proof Assistants in the Browser

**arXiv ID:** 2611.05540 | [PDF](https://arxiv.org/pdf/2611.05540v1)

**作者:** Yutaka Nagashima `[一作]` `[通讯]` (Czech Academy of Sciences), Yutaka Nagashima (Czech Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了ProofStudio，一种基于浏览器的环境，集成了Isabelle/HOL、Lean、Rocq、HOL4和HOL Light，可直接访问、编辑、运行证明并提供AI辅助。

**💡 创新点**

创新点在于：无需本地安装即可使用多种证明助手；统一界面展示相同算法的不同实现；预配置自动证明搜索与AI建议；利用虚拟机隔离执行，提升安全性。

**🔧 技术方法**

技术实现包括：Web前端与后台管理器；QEMU/KVM虚拟机做证明助手隔离；各证明助手的自动化工具（AbductionProver、Aesop、CoqHammer、TacticToe）；LLM接口（默认Qwen3-Coder、可BYOK）；集中式会话与资源调度。

**📊 数据集**

使用的数据集为十个对应示例脚本（如插入排序、归并、二叉搜索树等），来源于《Functional Data Structures and Algorithms》并经各证明助手自带库验证。

**📈 对比分析**

比较方法是让用户在相同算法示例上分别使用不同证明助手，记录任务完成率、所需时间、错误数量和对中间目标的说明；初步评估显示ProofStudio能快速展示中间状态并支持AI建议，但未给出具体数值基准。

**⚠️ 局限性**

局限性包括：仅支持单文档项目、受限的库覆盖、服务器容量有限；虚拟机隔离非形式化验证；缺乏多文档协作、广泛库支持和正式性能评测。

---

## 207. The CNF Encoding Complexity of Boolean Functions

**arXiv ID:** 2611.05614 | [PDF](https://arxiv.org/pdf/2611.05614v1)

**作者:** Andrew Krapivin `[一作]` (Carnegie Mellon University), Bernardo Subercaseaux `[通讯]` (Carnegie Mellon University)

**通讯引用:** 526 | [OpenAlex ID](https://openalex.org/A5030108071)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了将布尔函数压缩为 CNF 编码所需的最小子句数，并给出了针对可在带建议的非确定性 RAM 模型中计算的函数的上界以及对 k‑CNF、稀疏支持函数等特殊类别的紧密估计。

**💡 创新点**

创新点在于把非确定性 RAM 的建议机制转化为 CNF 编码的“建议检查”子程序，从而将传统的 O(t+m) 上界大幅改进为 O(t+√(m b))；并首次将现代数据结构技术（如独立集查询、哈希与显式扩展子图）应用于 CNF 编码，得到 k‑CNF 函数的 n^k/2^Ω(√n) 子句上界。

**🔧 技术方法**

主要技术包括：非确定性 RAM+建议模型、改进的 Robson Cook‑Levin 定理、稀疏函数的字典编码、显式扩展子图与哈希压缩、计数与信息论下界、以及 De Morgan 电路与 CNF 编码的等价性。

**📊 数据集**

本文未使用实验数据或真实数据集，而是全部基于理论构造与计数论证。

**📈 对比分析**

与先前 O(t+m) 或 O(n^k/n) 的上界相比，所给的上界在渐进意义上显著更小；虽然没有实验评估，但理论证明表明在最坏情况和大多数函数上均能取得更优子句数量。

**⚠️ 局限性**

主要局限在于对 k‑CNF 函数最坏情况的下界与上界仍存在较大间隙（现有下界仅为 Ω(n)），以及对宽度受限编码的理论分析仍不完整；此外，虽然理论上可构造编码，但在多项式时间内实现仍面临挑战。

---

## 208. GPUMAP: GPU-Accelerated Progressive Uniform Manifold Approximation and Projection

**arXiv ID:** 2611.06191 | [PDF](https://arxiv.org/pdf/2611.06191v1)

**作者:** Myeongwon Jung `[一作]` (Sungkyunkwan University), Jaemin Jo `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了GPU加速的进阶UMAP实现GPUMAP，使得百万级数据能在可交互的时间窗口内产生逐步投影。

**💡 创新点**

创新点在于把UMAP分解为插入、更新和嵌入三种可调度操作，并通过调度权重实现进阶投影与高效GPU并行化。

**🔧 技术方法**

采用CUDA/C++实现GPU并行，结合RAPIDS cuML的近似最近邻与UMAP内核，配合线性回归时间预算估计和多队列调度。

**📊 数据集**

使用多种真实数据集（图像、表格、基因、文本）及数百万点的HIGGS、LAION等，并用合成Spheres、Cylinder验证偏序插入。

**📈 对比分析**

与RAPIDS UMAP和Progressive UMAP对比，GPUMAP在10秒/5秒预算下的归一化忠诚度和最近邻召回率均优于对手，最终投影质量与序贯UMAP相当，耗时约1.6倍。

**⚠️ 局限性**

局限包括线性回归预算估计的鲁棒性、固定调度权重难以自适应、以及GPU与CPU间的通信开销导致的额外延迟。

---

## 209. How Well Can Large Language Models Impersonate Individual Authors? A Comparative Evaluation

**arXiv ID:** 2611.05771 | [PDF](https://arxiv.org/pdf/2611.05771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 210. SO(3)-Equivariant Learning on CAD Boundary Representations

**arXiv ID:** 2611.05449 | [PDF](https://arxiv.org/pdf/2611.05449v1)

**作者:** Matteo Ballegeer `[一作]` (Ghent University), Dries F. Benoit `[通讯]` (Ghent University)

**通讯引用:** 1861 | [OpenAlex ID](https://openalex.org/A5089981863)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种SO(3)等变的B-rep图神经网络SO3brep，用于对CAD零件的面进行分割和分类；

**💡 创新点**

创新点在于在B-rep面邻接图上实现SO(3)等变消息传递，利用局部参考框架和Wigner D矩阵将面特征映射到共享框架，再通过Clebsch–Gordan张量乘积保留相邻面之间的相对方向，实现既保持等变性又能传递方向信息；

**🔧 技术方法**

使用局部参考框架、Wigner D矩阵、球谐函数、Clebsch–Gordan张量乘积、注意力聚合、以及SE(3)-不变的读取层；

**📊 数据集**

在三种面分割基准（MFCAD++、MFTRCAD、Fusion 360）和两种分类基准（SolidLetters、BenDFM）上进行实验；

**📈 对比分析**

与UV-Net、FoV-Net、VN-PointNet、CSEConv等非等变和等变点云网络对比，SO3brep在所有基准上（包括随机旋转测试）均取得最高准确率和mIoU，参数量最少，尤其在样本稀缺时优势显著；

**⚠️ 局限性**

局限性包括：边特征维度有限；对B-rep导出/重参数化敏感；CG乘积复杂度高；仅适用于离散面标签，无法处理连续场；基准已趋于饱和，难以进一步展示等变性的优势。

---

## 211. Where models of minimal height are hard to find: cubic rational maps with a multiple fixed point

**arXiv ID:** 2611.05514 | [PDF](https://arxiv.org/pdf/2611.05514v1)

**作者:** Elira Shaska `[一作]` (Oakland University), Elira Shaska `[通讯]` (Oakland University)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5094272568)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究三次有理映射模空间中给定同形类时求取最小整数系数模型的算法，并提出改进的结构化搜索算法 A^+，能够在保证精确性的前提下更高效地发现更小的模型。

**💡 创新点**

创新点在于将机器学习诊断与严谨的数值验证结合：利用梯度提升树揭示“难点实例”集中在多重固定点的代数表面上，并据此设计结构化局部搜索和判别规则；同时引入额外协变点以改进归约步骤，形成可验证的结构化搜索框架。

**🔧 技术方法**

主要技术包括：完全显式的分解下降法和结果式最小化；基于 Bruhat–Tits 树的极大化和最小化；减小过程中的协变点构造；梯度提升树（XGBoost）进行特征重要性分析与分类；以及基于可验证的深度搜索与局部搜索实现。

**📊 数据集**

数据集：从所有 |c_i|≤5 的 104 371 000 个原始三次映射中提取 24 436 749 个细化模空间点（含 9 215 604 个“fine”点）；再生成高度不超过 7 的模空间点共 12 438 121 点，随机抽取子集用于训练与测试；另外使用 10 000 点在 5<H≤6 与 6<H≤7 的检验集。

**📈 对比分析**

与原始算法 A 的比较：在 H≤5 的 61 725 个样本中，A^+ 在 5% 的点上（或使用精确判别规则时 17%）能达到参考深度‑5 搜索的最佳高度，覆盖 78–92% 的“hard”实例；时间上 A^+ 仅为 A 的 1.5–2.6 倍，远低于在所有点上执行深度‑5 搜索（≈11–12 倍）。在更高高度的 held‑out 采样中，A^+ 仍保持 78–84% 的召回率，且时间提升仅为 1.6–2.6 倍。

**⚠️ 局限性**

局限性包括：参考深度‑5 搜索仅覆盖到 13 以内的素数且有 4% 的点未完成，因而“hard”实例率仅为下界；样本高度限制在 7，未能验证更大高度的表现；算法 A 的步骤 5 的实现选择可能影响结果；最终判别规则（多重固定点表面）是在 held‑out 数据后验证的，缺乏理论证明；最后，所用的机器学习模型在更高高度的泛化能力尚未确认。

---

## 212. When File Ingestion Misleads Language Models: Measuring and Mitigating Divergences in Office Documents

**arXiv ID:** 2611.05806 | [PDF](https://arxiv.org/pdf/2611.05806v1)

**作者:** Penghui Li `[一作]` `[通讯]` (Nanyang Technological University), Penghui Li (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了办公文档在被大语言模型（LLM）处理时，文本提取器与人类视觉呈现产生的差异，并基于此构造了多种攻击手段，展示了攻击成功的风险；

**💡 创新点**

创新点在于系统性识别OOXML中的“提取偏差”通道（包括隐藏文本、格式化差异、阅读顺序等），设计了五种无指令攻击，并提出了基于OOXML规范的“静态解释器+渲染仲裁器”防御框架；

**🔧 技术方法**

技术手段包括：静态与动态程序分析提取器行为、OOXML显示语义的显式解释、基于多引擎渲染的视觉一致性检测、以及与LLM对话的评估方法；

**📊 数据集**

使用了公开的办公文档语料（govdocs1、国家开源数据门户等）共计2,229份文件进行实测，并构造了28个额外攻击样本；

**📈 对比分析**

通过对8款主流聊天助手和4款代码助手的10次实验验证，攻击成功率最高达88%，平均误导率64%；防御在不改变功能的情况下，误报率仅2.6%，且对攻击完全有效；

**⚠️ 局限性**

局限性包括：仅覆盖已知提取器，未检验所有商用LLM内部路径；攻击仅针对OOXML，其他格式未深入；渲染仲裁器依赖多模型视觉输出，性能受限；

---

## 213. The Power and Limits of Money Burning in Bilateral Trade

**arXiv ID:** 2611.05505 | [PDF](https://arxiv.org/pdf/2611.05505v1)

**作者:** Moshe Babaioff `[一作]` (Hebrew University of Jerusalem), Ariel Shaulker `[通讯]` (Stanford University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在双边交易问题中，研究了弱预算平衡（即允许“烧钱”）的 DSIC 机制在实现交易收益（即交易双方总效用）方面的潜在优势，证明在独立分布下可以使交易双方的交易盈余（相对无交易的收益）实现任意大的相对提升，在相关分布下可以使交易双方的总效用实现任意大的相对提升。与此相对，证明即便允许烧钱，DSIC 机制仍无法保证相对于第一最佳（即完全有效率的交易）获得常数比例的收益，无论是对交易盈余（独立分布）还是对总效用（相关分布）。

**💡 创新点**

创新点在于：①首次揭示在 DSIC 机制中，弱预算平衡（烧钱）能够让交易双方获得无界提升的收益；②提出一种基于网格、舍入与增长因子的机制构造，实现了对 SBB 机制的显著超越；③利用 Myerson 引理与 SBB 机制的发布价格结构，证明了即使烧钱也无法突破已知的第一最佳逼近障碍，给出了新的不可逼近性上限。

**🔧 技术方法**

主要技术手段包括：
- 构造基于离散网格的分段单调分配规则，结合卖方上舍入、买方下舍入以及层级增长因子，使得在不同层级间保持弱预算平衡；
- 运用 Myerson 引理将激励兼容性与支付约束转化为积分不等式，进而推导出对分配概率的下界；
- 对 SBB DSIC 机制利用已知的发布价格表征，证明其在离散层级上的分配概率之和被限制在常数；
- 通过设计独立与相关分布（指数分布、几何网格分布）使得大部分第一最佳收益集中在层级 2 的点上，从而放大 WBB 机制与 SBB 机制的差距；
- 采用积分与对数估计技术，对连续分布下的弱预算平衡 DSIC 机制的交易概率进行全局上界，进而得到不可逼近性证明。

**📊 数据集**

论文不使用实测数据集，而是通过构造理论上的概率分布来展示效果：
- 独立分布：对卖方与买方使用指数型连续分布（或其离散化）;
- 相关分布：采用几何网格分布，将卖方价值置于低端、买方价值置于高端，并以逆比例权重保证各层级贡献相同；
- 这些分布均可解析地计算第一最佳收益与机制收益。

**📈 对比分析**

比较方法：
- 与强预算平衡 DSIC 机制对比，利用已知的发布价格结构，将其交易概率之和上界为常数，从而得到无界收益提升的下界；
- 对不可逼近性证明，则对任意弱预算平衡 DSIC 机制，利用积分约束与分布的特性，得到其交易概率在低收益层级被迫接近 0，从而上界其收益与第一最佳之比；
- 结果表明：
  * 对交易盈余（独立分布）和总效用（相关分布）可实现任意大提升；
  * 对交易盈余（独立分布）和总效用（相关分布）均无法得到常数逼近；
  * 与已知的 BIC 机制结果相比较，DSIC 机制在弱预算平衡下仍表现出本质的局限。

**⚠️ 局限性**

局限性包括：
- 仅在双边交易（单个不可分割物品）设置下得到结果；
- 只考虑 DSIC（而非更弱的 BIC 或强制性）机制，且假设参与是自愿且满足 IR；
- 结果主要针对单个买方与卖方，未直接推广到多买多卖或多物品情形；
- 对于相关分布的不可逼近性，证明依赖于特定的几何网格构造，是否可在更一般的相关分布下保持同样的限制仍是开放问题；
- 论文未探讨如何在实际机制设计中实现烧钱（支付差异）与预算约束的实现细节。

---

## 214. FlyHash for LLM Agent Memory: An Empirical Validation and a New Application

**arXiv ID:** 2611.05791 | [PDF](https://arxiv.org/pdf/2611.05791v1)

**作者:** Mohan Bheemadolu `[一作]` (Independent Researcher), Chrisogonas O. Odhiambo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究在LLM代理记忆检索任务中验证并实现了FlyHash哈希方法，首次将其用于此场景；

**💡 创新点**

创新点在于：①首次将FlyHash迁移至LLM记忆检索并在匹配活跃位数协议下证明其优于SimHash；②发现并纠正了k-vs-m混淆导致的比较偏差；③在真实性检索标签上验证了FlyHash的优势；

**🔧 技术方法**

使用的技术包括FlyHash、SimHash两种哈希方案、稠密向量索引HNSW、稀疏集合比较、TF‑IDF+SVD嵌入、MiniLM句子嵌入以及多种统计检验；

**📊 数据集**

实验数据集包含四个独立数据集（GloVe 100维、MiniLM 384维、sklearn 8×8图像、MNIST 28×28图像）以及构造的150个人工问答标签集；

**📈 对比分析**

在匹配活跃位数的公平协议下，FlyHash在所有数据集、10个随机种子上均显著优于SimHash（mAP提升明显）；但在存储/延迟测评中FlyHash并未表现出明显优势，且在与HNSW的系统比较中仍落后；

**⚠️ 局限性**

局限性包括：稠密实现导致存储/速度劣势、缺乏高效稀疏索引方案、未尝试DenseFly或BioHash、未与量化方法比较、实验规模有限、仅单点延迟评估、以及使用合成问答数据而非真实日志。

---

## 215. Zuse: A Decision Model for European Languages

**arXiv ID:** 2611.06238 | [PDF](https://arxiv.org/pdf/2611.06238v1)

**作者:** Umair Mehmood Imam `[一作]` `[通讯]` (Bineric AI), Umair Mehmood Imam (Bineric AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并评估了 Zuse 系列决策模型（4B、9B、27B），可在 27 种欧洲语言下读取文档并输出多选概率，专为欧盟多语言环境设计。

**💡 创新点**

创新点在于：①使用低秩 LoRA 适配器和指针头将大型语言模型转化为决策模型；②针对 27 种欧洲语言进行系统性评估；③揭示了校准偏差按任务而非语言分布、可自动化比例与机会校正能力紧密相关，并指出基座模型决定语言覆盖范围。

**🔧 技术方法**

采用冻结的 Qwen 开源基座，训练低秩适配器和指针头；使用温度标定、交叉熵与置换一致性损失；对长文档使用合成数据对模型进行推理速度评估。

**📊 数据集**

数据集包括：排除训练集的 11 来源、9 任务组、27 种语言的欧洲套件；世界语言覆盖 210 语言的阅读与主题任务；以及公开的 Transfer v4/v9、SemIf、Breadth Report、When2Call 等基准。

**📈 对比分析**

在欧洲套件上 27B 版本取得 87.5% 准确率，Brier 分数 0.185，ECE 0.016；相较于同类公开模型 Jev 与 Kev，Zuse 在 English 公共套件上整体略逊，且在自动化比例上跟随机会校正得分；在 210 语言扩展中，基座模型决定跨语言性能，超出训练语言的模型仅比基座低 23–28 分。

**⚠️ 局限性**

局限性包括：单种子实验、未公开训练数据与完整训练细节、对高知识密集任务表现不佳、对长文档超 32k token 的准确性未知、模型对顺序与注入攻击的敏感性仍需改进，以及对非 27 语言的评估主要受基座影响。

---

## 216. Reassessing Dataset Quality and Fine-Tuning Practices for Vulnerability Detection on PrimeVul dataset

**arXiv ID:** 2611.06373 | [PDF](https://arxiv.org/pdf/2611.06373v1)

**作者:** Norbert Sándor Szolnoki `[一作]` (University of Szeged), Gábor Antal `[通讯]` (University of Szeged)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性地评估了三种预训练代码语言模型（CodeBERT、GraphCodeBERT、UniXCoder）在PrimeVul数据集上进行全函数二分类的监督微调效果，涵盖770个实验配置；

**💡 创新点**

发现传统指标（F1、ROC‑AUC）并不可靠，真正的安全相关指标（MCC、ECE、VD‑S）显示模型表现接近随机，揭示PrimeVul的标签噪声与极高的函数相似度导致模型难以学习；

**🔧 技术方法**

使用Hugging Face微调框架、五种不平衡处理策略（基线、类别权重、欠采样、过采样、焦点损失）、温度标度校准、阈值优化以及多重指标评估；

**📊 数据集**

PrimeVul函数级漏洞检测基准（C/C++项目的函数对，含漏洞与修复版本）；

**📈 对比分析**

与传统F1/ROC‑AUC相比，安全相关指标几乎不提升；不同不平衡策略和超参数对MCC和VD‑S影响极小，所有模型在校准后仍保持接近随机的性能；

**⚠️ 局限性**

PrimeVul标签稀疏、函数对高度相似、弱监督导致的噪声；仅用标准微调和单一模型难以获得实用的漏洞检测性能，需改进数据质量与评估标准。

---

## 217. Towards Efficient Whole Page Ranking with One-Pass Inference

**arXiv ID:** 2611.05542 | [PDF](https://arxiv.org/pdf/2611.05542v1)

**作者:** Qi Xu `[一作]` (Meta Platforms, Inc.), Luke Simon `[通讯]` (Meta Platforms, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现一种全页排序框架 Multi‑Item Scoring (MIS)，在保持非自回归（NAR）高效性的同时引入前缀上下文，能够在广告‑有机混合页面中同时优化业务指标与用户体验。

**💡 创新点**

创新点包括：①将共享候选与位置编码与可变前缀注意力结合，构造前缀感知的分数；②使用残差量化变分自编码（RQ‑VAE）压缩历史状态，降低存储与计算负担；③采用活跃行搜索（active‑row）避免高阶张量全量 materialization，实现高效约束 Beam 搜索；④一次性请求编码后仅对前缀进行轻量级条件分数，兼顾 NAR 的速度与 AR 的序列建模。

**🔧 技术方法**

技术手段：基于自注意力与跨注意力的共享编码器；残差注意力的前缀条件头；RQ‑VAE 压缩历史；约束 Beam 搜索；多任务预测（多标签）联合目标；局部与完整前缀注意力（PA‑1、PA‑N）。

**📊 数据集**

数据集：公共数据集 KuaiRand‑1K（30 万条窗口，10 项滑动展示）和 RL4RS（30 万条 9 项真值滑动）；工业广告‑有机混合页面日志（3,670 万页，含约束规则）；随机化实验平台的社交短视频大规模日志。

**📈 对比分析**

比较方法：与 DNN、DCN、Seq2Slate、NAR、CGA 等基线在公共数据集上进行预测/生成质量评估，使用 AUC、NE、生成 lift 作为指标；在工业数据上通过多任务预测与页面价值提升（beam‑64）进行对比；在真实线上随机化实验中评估实现度、观看时长及 CPU 影响。结果显示：MIS 在公共数据集上接近 CGA 的生成 lift，并比 CGA 单模型解码快 4.07×/3.54×；在工业实验中实现度提升 0.104%，观看时长 +0.058%，CPU 消耗下降 9.73%（预测 CPU 略升 1.39%）。

**⚠️ 局限性**

局限性：①仍需 Beam 搜索近似，无法保证全局最优；②前缀长度与 RQ‑VAE 压缩级别需经验调优，可能影响不同场景的性能；③在极大动态库存时，历史编码与约束搜索的计算仍会成为瓶颈；④与完整 AR 方法相比，预测误差仍存在；⑤实验中未对所有任务给出统计显著性检验，可能影响结论稳健性。

---

## 218. PersonaCodeBench: Evaluating Coding Agents in Persona-driven Human-Interactive Sessions

**arXiv ID:** 2611.05739 | [PDF](https://arxiv.org/pdf/2611.05739v1)

**作者:** Haixin Wang `[一作]` (University of California Los Angeles), Xiaomin Li `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 PersonaCodeBench，一个面向用户个性化需求的多轮编码评测基准，旨在评估编码代理在满足功能正确性与用户期望的双重约束下的表现。

**💡 创新点**

创新点包括：①将用户角色、技术能力、交互偏好与工作限制等特征转化为可验证的任务约束；②将功能完成与个性化满足分离为独立评估维度；③采用逐步披露需求的交互机制，模拟真实编码场景；④提供完整的评测流程、控制器与在线评估器，支持可重复的实验。

**🔧 技术方法**

技术实现主要依赖：①PersonaPool与需求生成器将角色描述映射为具体的演化需求；②控制器（Controller）管理交互状态、需求披露与授权；③在线需求评估器（Online Requirement Evaluator）基于确定性规则或LLM判断满足情况；④完成验证器与个性化验证器分别评估任务正确性与用户约束。

**📊 数据集**

使用的数据集来自五个公开编码基准（Terminal‑bench 3、DeepSWE、SWE‑bench‑Live、SWE‑Together、SWE‑bench Pro），共231个任务‑人设绑定；Lite 版本精选100个任务，包含186个演化需求，覆盖多语言和多任务类型。

**📈 对比分析**

评测方法采用 Joint Pass（功能+个性化两者均通过）以及分别的完成率与个性化通过率；在Lite集上，最佳模型（GPT‑6 Astra max）获得48% Joint Pass，模型性能随源、预算与推理努力显著差异，展示了不同配置下的强弱对比。

**⚠️ 局限性**

局限性包括：①Joint Pass率仍低（<50%），表明现有模型难以同时满足功能与复杂用户需求；②评测高度依赖模拟器与自动判定，缺乏真实用户满意度验证；③需求生成与评估过程复杂，对实验可复现性和可扩展性提出挑战；④部分任务受限于可验证的证明形式，导致对高级运维或安全场景的覆盖不足。

---

## 219. GeoDIVE: Geospatial Descent Informed by Value Estimation

**arXiv ID:** 2611.05697 | [PDF](https://arxiv.org/pdf/2611.05697v1)

**作者:** Hamza Iqbal `[一作]` (Washington University in St. Louis), Nathan Jacobs `[通讯]` (Washington University in St. Louis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出GeoDIVE，一种针对多分辨率卫星图像金字塔的懒惰编码与基于概率值估计的最佳优先搜索框架，用于高效的地理空间检索；

**💡 创新点**

创新点在于学习一个乐观的子树最大相似度估计器（大小感知残差头），在搜索过程中仅编码必要的瓦片；同时结合高效的后期局部重排序（Patch-Level Late Interaction）实现接近全量搜索的召回率；

**🔧 技术方法**

技术包括：多分辨率图像金字塔结构、基于DINOv3和GeoRSCLIP的特征编码、两层MLP残差头预测子树最大相似度、最佳优先（best‑first）树遍历、适应性停止阈值、以及MaxSim后期重排序；

**📊 数据集**

使用Git‑10M、Dense500等卫星图像数据集，并构造了超过100M的Microsoft建筑足迹与OpenStreetMap多尺度实例，生成图像和文本（VLM生成标题）查询；

**📈 对比分析**

与多种基线（纯余弦、行为克隆、OTM、VAS、ZoomEye MLLM等）和Oracle进行比较；在图像查询下，GeoDIVE在仅使用3%编码预算时即可匹配或超过5%预算的基线，25%预算时几乎达到全量搜索性能；在文本查询下也显著优于其他方法；

**⚠️ 局限性**

局限包括：对高分辨率深度的依赖导致对低分辨率区域的识别不如全量搜索；基于学习的乐观估计器可能在极端分布上误判；目前仅支持单目标检索，未考虑多目标或视频序列；

---

## 220. FedXCrop: Privacy-Preserving Crop Disease Detection via Federated Learning with Post-Aggregation Explainability under Non-IID Farm Distributions

**arXiv ID:** 2611.06221 | [PDF](https://arxiv.org/pdf/2611.06221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 221. Navigational Safety of Marine Surface Vehicles in Restricted Visibility: From Intelligent Vision to Vision-Centered Multi-Sensor Fusion

**arXiv ID:** 2611.05818 | [PDF](https://arxiv.org/pdf/2611.05818v1)

**作者:** Ryan Wen Liu `[一作]` (Wuhan University of Technology), Shanliang Yao `[通讯]` (Wuhan University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了海面车辆在受限能见度条件下的智能视觉与视觉中心多传感器融合技术，强调了从图像生成、图像增强、目标检测与跟踪到场景分割的完整工作流程；

**💡 创新点**

首次将智能视觉方法与多传感器融合方法系统化为统一框架，并梳理了对应的数据集、应用场景与未来挑战；

**🔧 技术方法**

涉及相机、红外、雷达（S/X波段与4D毫米波）、LiDAR、AIS等多模态传感器的融合，利用GAN/扩散模型、Transformer、图卷积等前沿深度学习技术；

**📊 数据集**

列举并使用了MODD、MODD2、SMD、MaSTr1325、MODS、MID、USVInland、FloW、WaterScenes、PoLaRIS、USVTrack、MULTIAQUA、SeaClips、WUTDet等多种公开数据集；

**📈 对比分析**

对比了单传感器与多传感器融合模型在目标检测、跟踪、分割等任务上的精度与召回率，证明融合显著提升低能见度下的感知性能；

**⚠️ 局限性**

存在数据集缺乏极端受限能见度样本、跨模态标注与同步困难、模型对环境漂移泛化不足、融合算法计算量大导致实时性受限等局限。

---

## 222. Towards Insight-Centric Data Spaces: Integrating Visual Analytics and Decentralized Data Ecosystems

**arXiv ID:** 2611.06098 | [PDF](https://arxiv.org/pdf/2611.06098v1)

**作者:** Christina Faßbender `[一作]` (Fraunhofer FIT), Christoph Lange `[通讯]` (Fraunhofer FIT)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文探讨视觉分析与去中心化数据空间（dataspaces）的交叉点，提出将两者融合成“以洞察为中心的数据空间”的愿景，并通过文化数据空间的半合成场景演示如何利用视觉分析监控元数据质量、交换模式和服务状态。

**💡 创新点**

创新点在于将视觉分析视为认知层，使其能够解释数据可用性、质量、可溯源及治理约束；同时将数据空间的去中心化、主权化治理与视觉分析的交互探索结合，形成可视化驱动的分布式分析框架。

**🔧 技术方法**

技术包括：多维可视化编码（如半圆柱、堆叠柱状图）、基于地理信息的空间层次视图、元数据可用性评分算法、合成交易关系生成器；架构上依托数据空间的目录、身份、合约和连接器服务。

**📊 数据集**

使用了GovData CKAN元数据样本（500条）与德国BKG VG250区划图层，后者经合成分配后与元数据样本关联，形成地区级元数据质量和交易强度的半合成数据集。

**📈 对比分析**

论文未进行实验性性能评估；所示可视化仅为概念验证示例，未与现有工具或系统做定量对比，评估方法为案例演示与设计论证。

**⚠️ 局限性**

局限性主要在：使用的元数据样本和合成交易均为半合成，缺乏真实机构级数据；缺乏对交互延迟、可扩展性、隐私保护及真实治理契约的实测；未验证所提视图对用户决策的实际帮助，且未给出量化性能指标。

---

## 223. Reasoning-Aware Cross-Modal Association Unlearning in Multimodal Large Reasoning Models

**arXiv ID:** 2611.06168 | [PDF](https://arxiv.org/pdf/2611.06168v1)

**作者:** Xinghao Li `[一作]` (Monash University), Longxiang Gao `[通讯]` (Qilu University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多模态大规模推理模型（MLRM）的跨模态关联遗忘，提出了 Visual-Keyed Counterfactual Rebinding（VKCR）方法，通过局部参数子空间定位与对抗性分支重绑定实现对特定视觉‑知识关联的选择性消除。

**💡 创新点**

将遗忘目标从知识本身转向视觉‑知识关联，结合关联条件参数子空间定位与对抗性分支重绑定，并提出了 CLEAR‑R 与 MLLMU‑R 两套推理层面的评测基准。

**🔧 技术方法**

采用低秩参数更新、视觉键控制、对抗性支撑抑制、对抗性分支重绑定、行为保护与残差修正等技术，在 Qwen2.5‑VL、LLaVA‑1.5 等大规模推理模型上实现编辑。

**📊 数据集**

使用 CLEAR‑R（扩展自 CLEAR）和 MLLMU‑R（扩展自 MLLMU‑Bench）两套数据集，包含推理轨迹、文本与多模态对照。

**📈 对比分析**

与 Gradient Diff、KL‑Min、NPO、MANU、R²MU、MMUnlearner、R‑MUSE 等基线在 5% 忘记率下对比，VKCR 在目标泄漏、幻觉重关联与非敏感推理保持上显著优于基线，同时保留集性能保持较好。

**⚠️ 局限性**

仍存在保留效能轻微下降，参数与表示耦合导致对邻近行为的轻微干扰，未实现对内部神经路径的精确定位，且评估仅覆盖推理层面，未覆盖更广泛模型或任务。

---

## 224. Chiron: Cache Management Empowered by LLM Intelligence for AI Workloads

**arXiv ID:** 2611.06179 | [PDF](https://arxiv.org/pdf/2611.06179v1)

**作者:** Bohan Zhang `[一作]`, Chen Chen `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）的缓存管理系统Chiron，利用LLM进行访问模式诊断并通过决策切换来动态选择预取和驱逐策略，提升AI工作负载的缓存命中率和作业完成时间。

**💡 创新点**

创新点在于将LLM的诊断能力与“建议切换控制”相结合：LLM仅提供模式建议，确定的“硬件执行器”根据置信门控决定是否采用，并通过反馈循环修正诊断，解决了LLM推理延迟、代价高、决策质量差等挑战。

**🔧 技术方法**

核心技术包括异步LLM诊断（使用深度学习模型如DeepSeek-V4-Flash）、结构化统计特征（如步长、重用率等）用于提示、置信门控机制、反馈回馈反射、以及与JuiceFS的集成实现。

**📊 数据集**

使用了九个代表性AI工作负载（包括多相ResNet‑50、FNO、Stable Diffusion 1.5等）和对应的数据集（ImageNet子集、PDEBench、LAION子集等），以及多种模型对比。

**📈 对比分析**

通过与四个基线（无缓存、JuiceFS、Alluxio、JuiceFS‑Adaptive）以及微基准进行对比，Chiron在多相工作负载上相较最佳基线降低作业完成时间14.3–41.2%，在单相工作负载上平均提升5.22%；在大规模模拟中仍保持最高缓存命中率和最低作业时间。

**⚠️ 局限性**

局限性包括对LLM质量的敏感性（低于4B参数的模型效果显著下降）、仍需额外的统计分析与门控实现、以及在极大规模下仍需保证LLM推理与反馈的异步与延迟控制。

---

## 225. Learning When Human Decisions Matter: Executor-Relative Consultation Value for LLM Agents

**arXiv ID:** 2611.05814 | [PDF](https://arxiv.org/pdf/2611.05814v1)

**作者:** Shuli Jiang `[一作]` (AWS Agentic AI), Stefano Soatto `[通讯]` (AWS Agentic AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在大语言模型代理中如何在有限的人类咨询预算下，选择最有价值的咨询点，以最大化任务终端效用。

**💡 创新点**

创新点在于引入“咨询价值蒸馏”(Consultation Value Distillation)，通过离线 Monte Carlo 代理回放学习执行者相对的咨询价值，然后将其蒸馏成轻量化预测器，实现部署时无需再做昂贵的回放估计。

**🔧 技术方法**

技术主要包括：离线 Monte Carlo 代理回放、基于 DeepSets 的集合价值回归、序列阶段的连续性价值预测、预测器集成与近似子集搜索，以及多任务、不同执行者的跨任务迁移学习。

**📊 数据集**

使用了六个基准任务（Animals、Trip Planning、Meeting Planning、Flight Recommendation、ThinkingBox-D、ThinkingBox-H）以及两个强大执行者（Sonnet 5 与 GPT‑6.1 Sol）。

**📈 对比分析**

与直接提示、VoI（信息价值）以及单独估计的局部价值（Local Value）进行对比；在大多数任务和预算下，蒸馏价值方法在归一化收益（NG）上显著优于基线，且在约束预算下近乎无损。

**⚠️ 局限性**

局限性包括：使用的是模拟或基准定义的人类回答而非真实人类，预算规模有限，且假设咨询机会预先确定，无法直接从原始任务上下文学习咨询候选和通用价值预测器。

---

## 226. EchoKV: Self-Corrective KV Cache Eviction via Iterative Query Feedback

**arXiv ID:** 2611.05903 | [PDF](https://arxiv.org/pdf/2611.05903v1)

**作者:** JianKun Zou `[一作]` (Beijing University of Technology), Tongtong Yuan `[通讯]` (Beijing University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

EchoKV通过在解码前构造一个基于未来位置的工作空间，并在解码过程中使用生成的查询迭代修复工作空间，从而实现KV缓存的闭环纠错；

**💡 创新点**

其创新点在于结合了位置旋转的伪查询预热工作空间以及生成查询的反馈驱动修复，实现了对分散证据的更好保留；

**🔧 技术方法**

主要技术包括RoPE旋转、位置伪查询、查询加权评分、贪婪lookahead、以及后向重评分的张量操作；

**📊 数据集**

实验使用了16个LongBench子任务（如HotpotQA、2WikiMQA、Multi-Doc QA等），在Mistral-7B-Instruct-v0.2和Qwen2.5-7B-Instruct两大模型上验证；

**📈 对比分析**

与FullKV、StreamingLLM、H₂O、SnapKV、PyramidKV以及LAQ等基线对比，EchoKV在总体上与LAQ保持竞争力，并在多证据检索任务上实现显著提升；

**⚠️ 局限性**

局限性包括额外的重评分开销、对完整缓存保留的依赖、以及在长上下文下提议评分成本高且未提供通用的递归刷新策略。

---

## 227. Efficient Stable Word-RAM Simulation by Population Protocols

**arXiv ID:** 2611.06178 | [PDF](https://arxiv.org/pdf/2611.06178v1)

**作者:** Leszek Gąsieniec `[一作]` (University of Liverpool), Grzegorz Stachowiak `[通讯]` (University of Wrocaw)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了人口协议模型中的计算，提出了一种高效、稳定且无声的模拟方法，能够处理广泛的计算问题，包括领导者选举和多数问题。

**💡 创新点**

创新点在于提出了一种通用框架，可以高效地模拟𝒫程序，扩展了有效稳定人口协议的计算能力，超越了个别问题和半线性谓词的限制。

**🔧 技术方法**

使用了概率调度的配对交互技术，结合了MC+协议和权重系统，确保了高效的计算和稳定的输出。

**📊 数据集**

使用了人口协议模型中的代理作为数据表示，输入和输出以一元形式表示，具体数据集未明确给出。

**📈 对比分析**

与现有方法相比，本文的方法在稳定化时间上表现出多对数级的期望和高概率性能，能够有效处理更复杂的计算任务。

**⚠️ 局限性**

限制在于当前框架假设了已知的字长和人口规模，未来的研究需要考虑更广泛的应用场景和数据集。

---

## 228. Constant-Round Massively Parallel Algorithms for Approximate MSTs in Metric Spaces

**arXiv ID:** 2611.06092 | [PDF](https://arxiv.org/pdf/2611.06092v1)

**作者:** Eunjin Oh `[一作]` (POSTECH), Hyeonjun Shin `[通讯]` (POSTECH)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的算法框架，在可扩展的 MPC（Massively Parallel Computation）模型中计算近似最小生成树（MST）。

**💡 创新点**

创新点在于：①使用“截断 Prim”算法在本地完成部分 MST，②通过“森林裁剪”将得到的森林拆分为可在单机上识别根的树，③采用“领导压缩+采样”两阶段合并策略降低树的数量，④对欧氏空间和双曲度空间利用一致性哈希实现近似邻域计算，从而在保持近似比的同时将总空间控制在线性级别。

**🔧 技术方法**

核心技术包括：MPC 原语（排序、前缀和、聚合、广播）、截断 Prim、森林裁剪、随机森林分割、领导压缩合并、采样合并、以及一致性哈希（geometric 和 doubling）。

**📊 数据集**

论文主要是理论分析，没有在公开数据集上实验；实验环境是假设输入规模为 n，空间维度 d 取 O(log n)，并假设点集的距离分布满足多项式范围。

**📈 对比分析**

与之前工作比较：在一般度量空间将轮数从 O(log log n + log(1/ε)) 降至 O(log(1/ε))（与 1‑vs‑2‑cycles 下的下界匹配），在高维欧氏空间将轮数从 O(log log n · log log log n) 降至 O(1)；在双曲度空间维度上将上限从 O(√(σ log n)) 扩展到 O(σ log n)；总空间保持 O(n²)（一般度量）或 O(nd + n^{1+σ})（欧氏）/ O(n^{1+σ}·2^{O( ddim)})（双曲度），且得到的近似比为 (1+ε) 或常数阶。

**⚠️ 局限性**

局限性包括：①对欧氏空间的维度仍要求 d = O(log n)；②对距离分布需要多项式范围的假设（可通过预处理消除但需额外轮数）；③算法的随机性和大常数系数使实际实现可能受限；④在高度稠密的度量空间中截断 Prim 的阈值选择可能导致较大误差，需进一步优化。

---

## 229. Practical Estimation of the Bayes Optimal Fairness-Accuracy Tradeoff with Soft Labels

**arXiv ID:** 2611.06027 | [PDF](https://arxiv.org/pdf/2611.06027v1)

**作者:** Mohit Sharma `[一作]` (IIIT Delhi), Masashi Sugiyama `[通讯]` (RIKEN-AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将 Bayes 错误估计推广到公平性-准确性 Pareto 前沿，提出了一种仅依赖软标签、无实例的估计方法，给出了公平 Bayes 错误的解析表达式，并证明了阈值、风险与不公平度估计的一致性、偏差与样本复杂度；

**💡 创新点**

创新点在于：① 将公平约束视为成本敏感学习问题，得到公平 Bayes 误差的显式公式；② 推导出可一致估计的群组阈值和成本敏感风险；③ 在理论上给出估计误差的上界和样本复杂度，并实现了基于二分搜索的 Pareto 前沿估计；

**🔧 技术方法**

采用的技术包括：软标签估计、Bayes 最优阈值表征、成本敏感风险推导、统一收敛理论、二分搜索算法、概率偏差与样本复杂度分析；

**📊 数据集**

实验使用的数据集有：合成高斯数据、CIFAR-10H（平均硬标签得到软标签）、以及 Adult 数据集（使用 Logistic 回归得到噪声软标签）；

**📈 对比分析**

与 Exp-Grad、Oxonfair、Fairret、MIFPO 等方法对比，本文估计的 Pareto 前沿在大多数情形下低于其余方法且方差最小；在 λ=0（无公平约束）时与 Bayes 最优解吻合，表明估计具有较高的理论与实践一致性；

**⚠️ 局限性**

局限性包括：估计对分布先验与不公平度 λ 依赖较强，导致样本复杂度随 λ 增大；需要群组标签与软标签，无法直接应用于无标签或噪声软标签场景；仅适用于线性/双线性公平度量；假设软标签分布连续且 Lipschitz，限制了通用性。

---

## 230. Optimal Re-Ranking Depth

**arXiv ID:** 2611.05908 | [PDF](https://arxiv.org/pdf/2611.05908v1)

**作者:** Siqing Huo `[一作]` (University of Waterloo), Charles L. A. Clarke `[通讯]` (University of Waterloo)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用大语言模型自动生成的稠密相关性评估，研究并预测检索过程中每个查询的最佳重排序深度

**💡 创新点**

发现不同查询有明确的最优重排序深度；标准查询性能预测方法无法有效预测深度；提出基于LLM评估的IR‑DCG@10预测器，可将平均重排序深度降至原来的1/3，同时保持或提升检索效果

**🔧 技术方法**

大语言模型（LLM）自动相关性判定、DCG计算、基于分位数的深度校准与预测、传统QPP指标比较

**📊 数据集**

MSMARCO v1 dev（约1,000个查询）和TREC DL 2019，结合BM25、SPLADE检索器和Mono‑T5、MonoELECTRA重排序器

**📈 对比分析**

与固定深度重排序（如深度1000）和11种经典QPP方法比较；IR‑DCG@10在阈值0.8时平均重排序深度下降约3倍，且检索效果（DCG@10）保持与全深度重排序相当，部分配置可提升7%+

**⚠️ 局限性**

对LLM评估的依赖导致成本高；在不同语料或模型组合上泛化能力尚未充分验证；仅关注DCG@10等单一指标，其他多样化检索评价指标未深入探讨

---

## 231. ViBench 2.0: A Benchmark for Long-Horizon Software Building

**arXiv ID:** 2611.05620 | [PDF](https://arxiv.org/pdf/2611.05620v1)

**作者:** Daniel Furman `[一作]` (Replit), Michele Catasta `[通讯]` (Replit)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了ViBench 2.0，评估代理在从零开始、持续接收功能请求的长周期中是否能交付完整可用的全栈 Web 产品。

**💡 创新点**

创新点在于：①提供仅在最终产品完成后才评估的全局通过率；②设计 17 个基于真实产品的长周期任务与 544 条交互式测试计划；③引入可靠的多轮评分协议和独立运行验证。

**🔧 技术方法**

使用 OpenHands SDK 搭建的统一构建器、Playwright 驱动的浏览器评分器、GPT‑6、Sol、Astra、Opus 等大模型以及开放模型 DeepSeek、GLM、Kimi 等；并采用 PostgreSQL 容器、容器化构建与分布式评分。

**📊 数据集**

基于公开软件（Amazon、Asana、Discord、Figma、GitHub、Google Docs、Jira、Uber 等）构造的 17 个全栈产品 PRD 与 544 条测试计划；9 个产品被保留为隐藏数据集。

**📈 对比分析**

通过独立实验比较九种模型，报告 pass@1（完整通过率）和计划通过率；三种最强模型的计划通过率 94–98%，但 pass@1 仅 49–65%；随模型版本演进，pass@1 从 3% 提升至 53%，成本从 $27 降至 $12。

**⚠️ 局限性**

局限性包括：①只评估功能正确性，未覆盖视觉、性能或成本指标；②评分仅在最后阶段执行，可能忽略中间错误；③公共数据集可训练，隐藏集避免了泄露；④多轮评分协议依赖重复校准，仍存在错误通过率；⑤仅针对固定功能顺序，未模拟真实的 Bug‑report 交互。

---

## 232. Beyond Accuracy: Information-Theoretic Decoder Evaluation Under Latent Intent

**arXiv ID:** 2611.05658 | [PDF](https://arxiv.org/pdf/2611.05658v1)

**作者:** Tasha Kim `[一作]` (University of Oxford), Oiwi Parker Jones `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文把助理脑-机器人控制中的解码器评估建模为POMDP，证明累计规划遗憾与解码器混淆矩阵中最难区分的两类之间的Chernoff信息呈现尖锐的相位转变，并给出可从混淆矩阵直接计算的闭式累计识别包络 C，作为预测长期遗憾的诊断工具。

**💡 创新点**

创新点在于①发现累计规划遗憾的相位转变完全由最差对的Chernoff信息决定，而非整体准确率；②提出可直接从混淆矩阵计算的累计识别包络 C，能够闭式预测累计遗憾；③揭示单步准确率不再是序列控制的合适评估指标，并提供更具预测力的指标。

**🔧 技术方法**

利用信息论（Chernoff信息、Fano不等式）、POMDP理论、MAP估计、闭式误差预测器、统计回归、Monte Carlo仿真以及目标跟踪的双积分器运动环境。

**📊 数据集**

使用了四类数据集：EEG BCI 数据（Stieger2021、BNCI2014001）、图像数据集 CIFAR-100、NinaPro 多模态（EMG/IMU/语音）以及 Speech Commands 的 MFCC 语音数据。

**📈 对比分析**

通过对累计遗憾与单步准确率及 log(C) 的回归，发现 log(C) 能解释约 80–90% 的遗憾方差，明显优于仅用准确率；在准确率相同的解码器对中，log(C) 正确预测高遗憾的比例约 77%；在真实 EEG 数据中，两种解码器的遗憾相差 9.6 倍，log(C) 能正确识别。

**⚠️ 局限性**

限制包括：观测假设为动作无关、独立同分布；不考虑时间相关性；仅适用于目标固定的 POMDP；只针对 MAP 目标承诺规划器；误差预测为期望值，有限样本可能导致实际与理论差异。

---

## 233. Verified but Vacuous: Three Failure Surfaces of LLM-Driven Verified Code Generation

**arXiv ID:** 2611.05590 | [PDF](https://arxiv.org/pdf/2611.05590v1)

**作者:** Aditya Vempaty `[一作]`, Ravi Kokku `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出对LLM生成代码的“验证”进行三面向的细分评估，并在Dafny管道中实现了一个静态的ensures-binding门。

**💡 创新点**

创新点在于将验证质量拆分为三个互不重叠的表面（A：验证器接受率，B：语法级规范绑定，C：语义规范忠实度），并通过实证演示这三面向的显著差距。

**🔧 技术方法**

使用的技术包括Claude Opus 4.7生成代码、Dafny 4.10 + Z3 4.13进行形式验证、静态AST检查实现ensures-binding门，以及Post2Exe语义检查。

**📊 数据集**

数据集涵盖Python类LeetCode风格任务（HumanEval、LeetCodeDataset、LiveCodeBench v6）以及Rust任务（MBPP-Rust、VeriContest LC切片）。

**📈 对比分析**

比较方法：在相同LLM后端下测量pass@1/10、验证器接受率、B面向绑定率和C面向语义通过率；实验显示Li@Opus在A→B上相差约40倍，VCG在B→C上仍有约12–15%的失败率，整体性能低于GPT-5.5等领先模型。

**⚠️ 局限性**

局限性包括仅使用Claude Opus 4.7单一模型族、只提供一种B面向检查（ensures-binding门）、未结合更完整的语义检查、并未对不同模型或提示策略进行系统比较。

---

## 234. Traffic-VL: Reinforced Vision-Language Models for Generalizable End-to-End Traffic Signal Control

**arXiv ID:** 2611.06353 | [PDF](https://arxiv.org/pdf/2611.06353v1)

**作者:** Xingchen Zou `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种端到端的视觉‑语言模型Traffic‑VL，直接从同步多视角监控图像和自然语言相位描述中生成交通信号控制决策，取代传统的结构化状态接口；

**💡 创新点**

创新点在于：①构建Semantics‑Preserving Sim2Real引擎SynTra‑V，将物理仿真转换为可供监督且保持交通语义一致的逼真多视角图像；②将视觉‑语言模型与强化学习两阶段训练（离线人类强化学习+在线STPO‑V）结合，实现视觉与决策的一体化；③采用自然语言相位描述，统一不同拓扑的相位语义；④实现异步跨交叉口协调。

**🔧 技术方法**

技术包括：Qwen3‑VL 8B视觉‑语言模型、基于SUMO+Blender的SynTra‑V图像合成与自动语义审计、GRPO离线强化学习、STPO‑V在线轨迹优化、异步消息传递和多视角视觉编码。

**📊 数据集**

使用SynTra‑V生成的约5,000张多视角图像（覆盖多城市风格、天气、光照等），以及公开Benchmark（Jinan/Hangzhou、OmniTraffic、Real‑1296）和三周真实交叉口部署数据。

**📈 对比分析**

与结构化状态控制器Traffic‑R1、RL基准、LLM/视觉‑语言基准等对比，Traffic‑VL在五个零样本闭环Benchmarks上与Traffic‑R1相当，在实际监控图像上实现82.1%相位准确率（高于Gemini 3 Pro的75%），并在现场部署中平均队列长度下降10.4%。

**⚠️ 局限性**

局限性包括：仅处理同步的单帧图像，缺乏视频时序信息；仍可能出现视觉误判导致的错误优先级；对异常情景（紧急车辆、摄像头遮挡）虽然鲁棒，但仍低于理想的结构化感知；模型对多交叉口协调的置信度和可解释性有限。

---

## 235. Reasoning-Trace Overflow in Code Generation: An Empirical Study of Explicit Reasoning Models

**arXiv ID:** 2611.06095 | [PDF](https://arxiv.org/pdf/2611.06095v1)

**作者:** Lukas Twist `[一作]` (King's College London), Jie M. Zhang `[通讯]` (King's College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统研究了在代码生成任务中显式推理导致的输出预算溢出（Reasoning‑Trace Overflow）现象，并评估了其对模型准确性和推理控制方法的影响。

**💡 创新点**

创新点在于首次定义并量化“推理轨迹溢出”这一失败模式，全面评估七款开源推理模型在六大代码生成基准上的溢出率、循环原因，并提出两种推理控制策略（提示引导与预算强制切换）并对其效果进行对比。

**🔧 技术方法**

使用了固定32k token输出预算、贪婪解码、推理/答案分隔符解析、循环检测算法、提示风格干预、预算切换机制以及token效率评估。

**📊 数据集**

使用的数据集包括：CodeContests、CodeNet（函数级合成）、Py150（库/API使用）、E‑Prime、Kattis（竞赛题）以及 CodeQ（短答式代码推理）。

**📈 对比分析**

通过比较启用推理 vs 禁用推理的pass@1、溢出率以及不同提示和预算切换下的宏平均pass@1与token成本，发现推理在更难任务和更大模型上能显著提升准确性，但在溢出高的模型/任务上禁用推理更有优势；预算切换在多数模型上既能提升pass@1又能降低每个正确答案的token消耗。

**⚠️ 局限性**

局限性包括：仅评估小型开源模型和单轮生成；使用贪婪解码和固定32k预算，未探讨其他解码策略；循环检测仅捕捉持续的文本重复，可能漏掉语义循环；未考虑多轮交互或工具使用场景；数据集与模型预训练可能存在泄露风险。

---

## 236. Semantic Prior-Guided Incomplete Cross-Modal Retrieval with Noisy Labels

**arXiv ID:** 2611.06113 | [PDF](https://arxiv.org/pdf/2611.06113v1)

**作者:** Jiali Chen `[一作]` (Sichuan University), Dezhong Peng `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SPIRN 框架，联合解决跨模态检索中的模态不完整和标签噪声问题。

**💡 创新点**

创新点在于通过固定正交语义先验构建稳定共享语义空间，并结合质量感知鲁棒学习、全局标签校准和置信度驱动的选择性补全，实现对噪声标签与缺失模态的统一鲁棒处理。

**🔧 技术方法**

采用的技术包括语义先验引导（SPG）、动态阈值质量权重与逆向学习（QRL）、全局标签重标（PGLC）、记忆库基准的置信度驱动选择性补全（CSI）以及跨模态对比学习等。

**📊 数据集**

实验数据集为四大公开图文检索基准：Wikipedia、XMedia、INRIA‑Websearch 与 XMediaNet。

**📈 对比分析**

与 11 先进方法（包括无监督与监督方法）在 mAP 与 PR 曲线上对比，SPIRN 在多种缺失率与噪声比例设置下均显著优于对照组。

**⚠️ 局限性**

局限性在于对语义先验的依赖，极端高噪声或极度不完整的数据场景下性能可能下降，且实现相对复杂。

---

## 237. Fluidic Body: Swarm Robots as a Fluid Extension of the Hand

**arXiv ID:** 2611.06068 | [PDF](https://arxiv.org/pdf/2611.06068v1)

**作者:** Michikuni Eguchi `[一作]` (University of Tsukuba), Takefumi Hiraki `[通讯]` (University of Tsukuba)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于虚拟流体的机器人群体交互系统 Fluidic Body，使用户通过手势直接调节群体的柔软度与密度，从而在不同任务（避障、搬运）中实现群体行为的动态适配。

**💡 创新点**

创新点在于将流体物理属性（粘度、密度）映射为可由手势直观控制的参数，形成“手-流体-机器人”的三层交互架构，使得人机交互从指令化转为直观的物理感知化，首次实现手势直接调节群体柔韧性与聚集度。

**🔧 技术方法**

技术包括基于平滑粒子流体动力学（SPH）的群体控制器、手势捕捉映射到流体参数的算法，以及形状势场用于让机器人填充用户手部轮廓所定义的目标区域。

**📊 数据集**

评估数据集主要为仿真环境（80 cm × 60 cm 2D 平面）中的16机器人实验、17名参与者的VR 用户研究以及 88 名现场观众的物理机器人演示，未使用公开机器人数据集。

**📈 对比分析**

通过仿真对比 RVO 基线、柔软流体、硬化流体三种控制器，在避障任务中柔软流体成功率达98%，在搬运任务中硬化流体与 RVO 完成率相同但时间更短；在 VR 用户研究中，Fluidic Body 在结合避障与搬运任务上成功率 100% 远高于 RVO（47%），完成时间缩短 50% 以上，同时 NASA‑TLX 工作量降低，代理感提升；物理演示中 88 名观众平均给予 3.81/5 的代理感评分。

**⚠️ 局限性**

主要局限包括：仅在 2D 平面实验，未验证 3D 空间；只提供两种离散流体设置（软/硬），缺乏连续参数调节；手势到参数映射单一设计，缺少用户个体化适配；物理机器人实现存在 100 ms 通信延迟导致的响应滞后；基线仅为固定参数 RVO，未与其他可调控控制器比较。

---

## 238. Hybrid LLM-Deterministic Pipelines for Enterprise-Scale Code Migration at Bloomberg

**arXiv ID:** 2611.05457 | [PDF](https://arxiv.org/pdf/2611.05457v1)

**作者:** Mario Corchero `[一作]` (Bloomberg), Federica Sarro `[通讯]` (University College London)

**通讯引用:** 5700 | [OpenAlex ID](https://openalex.org/A5012165852)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在Bloomberg实施了两项大规模代码迁移（C++构建系统迁移与Python打包迁移），使用LLM辅助的混合管道生成并提交了超过2300个PR，最终实现了729个迁移。

**💡 创新点**

提出了混合LLM–确定性管道，将确定性处理与LLM的受限变更分离，并将运行时生产数据嵌入迁移上下文，以提升迁移准确性；同时通过架构化的“原型”划分与可视化反馈循环，优化迁移效率。

**🔧 技术方法**

基于Claude Sonnet/Opus LLM进行文件编辑与结构提取，Python/JSON管道脚本负责分类、上下文构建与验证规则，结合Bloomberg内部的包管理、构建系统与运行时日志、模块使用追踪等Telemetry数据。

**📊 数据集**

覆盖约8600个候选仓库（C++ 6700个，Python 1900个），并利用Bloomberg内部的二进制包注册表、部署日志、运行时依赖等自有数据。

**📈 对比分析**

通过比较两种迁移场景的PR合并率、拒绝率和无决策率，发现C++迁移合并率为34.2%，Python迁移为10.5%；LLM推理成本低于工程投入，且在验证循环浅的场景中迁移成功率更高。

**⚠️ 局限性**

主要局限在于生成PR的速率远超团队吸收能力；缺乏对Runtime数据效用的定量ablation实验；仅在单一组织环境验证，难以推广至多租户或Monorepo；模型演化导致工程与验证策略需不断调整。

---

## 239. CoReFlow: Consistency-Relaxed Flow Maps for Probabilistic Traffic Forecasting

**arXiv ID:** 2611.05715 | [PDF](https://arxiv.org/pdf/2611.05715v1)

**作者:** Pengnan Chi `[一作]` (KTH Royal Institute of Technology), Xiaoliang Ma `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 CoReFlow 模型，用单步流图方法实现概率交通预测。

**💡 创新点**

在交通预测场景下放宽轨迹一致性约束，设计一致性放松损失并利用条件均值限制，显著提升准确性并降低采样成本。

**🔧 技术方法**

基于 Transformer 的流图（probability‑flow ODE）、简化 AdaLN、节点级自适应损失权重以及一阶解算器实现一次性生成。

**📊 数据集**

使用四个真实高速公路流量数据集：PEMS03、PEMS04、PEMS07、PEMS08。

**📈 对比分析**

与八种确定性基线和三种扩散式生成基线在 RMSE/MAE/MAPE/CRPS 上对比，CoReFlow 在多数指标上超越对手，尤其在 PEMS07/08 上取得最优成绩。

**⚠️ 局限性**

适用于条件未来近似单峰的场景，对多模态长时预测或非高速公路数据的效果未知。

---

## 240. Essential Unitarity for Higher-Order Quantum Computation (Extended Abstract)

**arXiv ID:** 2611.06137 | [PDF](https://arxiv.org/pdf/2611.06137v1)

**作者:** Samson Abramsky `[一作]` (University College London), Radha Jagadeesan `[通讯]` (DePaul University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个基于边界的高阶量子计算语义框架，并通过定义本质可逆性（Essential Unitarity）证明源算子本质上为酉的。

**💡 创新点**

创新点在于将酉演化扩展到高阶接口，提出本质可逆性这一新的边界条件，并在此框架下实现了可协同的量子开关以及等比单槽超映射的酉阶段。

**🔧 技术方法**

采用Kelly–Laplaza紧闭类别的极坐标表达、线性组合的端点匹配、矩阵/双乘积的完成以及对偶线性逻辑的语法与上下文线性控制构造，完成了语义和语法的形式化。

**📊 数据集**

本研究为理论性工作，未使用任何实验数据集。

**📈 对比分析**

论文未给出实验对比或性能评估，主要通过形式化证明展示方法的正确性与一致性。

**⚠️ 局限性**

局限性包括：仅处理纯酉过程，未覆盖测量、混态或完全正映射；对高阶过程的可扩展性与实际实现仍需进一步研究。

---

## 241. Toward an Optimal Configuration for Tree Species Classification from Bark Images: A Leakage-Safe Ablation Study

**arXiv ID:** 2611.05680 | [PDF](https://arxiv.org/pdf/2611.05680v1)

**作者:** Maeve Côté `[一作]` `[通讯]` (Independent researcher), Maeve Côté (Independent researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在树干皮质照片上系统性地对基于裁切图块聚合的深度学习框架进行消融实验，评估不同设计选择的影响；

**💡 创新点**

首次在树层级、无泄漏的数据划分下对图块大小、聚合方式、训练阶段、模型容量等关键参数进行量化比较，并给出简洁的部署配置；

**🔧 技术方法**

使用ConvNeXt‑V2骨干网络、注意力多实例学习（MIL）与辅助实例损失，结合多种聚合策略（投票、学习聚合、整图分类），并进行图块尺寸与测试时图块数量的扫描；

**📊 数据集**

采用公开的树皮识别数据集（Carpentier 等 2023），共 23 种树种 23,271 张图像，按树根层级划分、5‑折交叉验证；

**📈 对比分析**

通过树层级五折交叉验证比较不同配置，最优方案（512 px 图块、ConvNeXt‑V2 nano、学习聚合、单阶段训练）在图像级准确率 94.9%（树级 97.8%）上优于整图分类，学习聚合相较投票提升 0.7pp，单阶段训练优于两阶段；

**⚠️ 局限性**

仅基于单一地区数据集，超参数从 nano+224px 转移至其他配置，未针对每个设置做细调；图块与包大小混杂，可能隐藏其他因素；仅评估了一种 MIL 形式，未探讨更复杂的上下文建模；类不平衡问题仍存在。

---

## 242. Keeping Fuzzers on Duty: Repairing Fuzzing Build Breakages in Continuous Open-Source Fuzzing

**arXiv ID:** 2611.06358 | [PDF](https://arxiv.org/pdf/2611.06358v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 243. Incremental Minimum Spanning Trees on a Napkin

**arXiv ID:** 2611.05713 | [PDF](https://arxiv.org/pdf/2611.05713v1)

**作者:** Lucian Bicsi `[一作]`, Adrian Budău `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种仅使用两个数组和单循环的结构，实现增量最小生成森林与瓶颈距离查询，证明其摊销时间为O(log n)，并对比现有结构。

**💡 创新点**

关键创新在于重新排序旋转与跳跃操作，利用祖先不变性实现潜能收缩，去掉子树大小与平衡约束，同时提供严格的摊销证明。

**🔧 技术方法**

核心技术为基于父指针与边权的树结构、旋转与跳跃操作、潜能函数Φ=∑log s(v) 的分析，以及对Kruskal重建树的隐式链接-切割树视角。

**📊 数据集**

使用多种合成数据集：随机插入（Random）、网格（Grid）、带状（Band）以及局部插入顺序（Grid-local、Band-local）和针对 folklore 的对抗性 Lag（星形 + 查询）。

**📈 对比分析**

在 Ubuntu 24.04 上的两台 EC2 实例（AMD EPYC 与 Intel Xeon）上，采用 g++ 13.3/O3，五次独立运行取中值，结果显示本结构比 folklore 快 1.3–2.1 倍、比链接-切割树快 6.3–16 倍；在随机顺序下，lazy AM‑tree 仍略优；在局部或对抗性场景中，本结构表现最佳。

**⚠️ 局限性**

局限在于仅给出摊销上界，缺乏最坏情况查询保证；路径长度在随机输入时可能比 lazy AM‑tree 长 3.3 倍；无法提供持久化版本；对动态最优性等更深层次性质尚未研究。

---

## 244. Single-Variable Connectivity in Random $k$-SAT

**arXiv ID:** 2611.05981 | [PDF](https://arxiv.org/pdf/2611.05981v1)

**作者:** Chunyang Wang `[一作]` (National Institute of Informatics), Yiyao Zhang `[通讯]` (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了随机 k-SAT 的局部更新算法和相变阈值，构造了一种基于树形扩展的软场递归与*‑核心方法，证明了在低密度下解的松弛性以及在高密度下解空间的几何碎裂与刚性。

**💡 创新点**

创新点在于提出了软场递归与 f‑加权测度的框架，利用 Poisson 重新加权和多软子子子句裁剪实现了可逆且单步可达的局部更新；并首次将*‑核心与势能阈值相结合，刻画了解空间从松弛到刚性过渡的精细阈值。

**🔧 技术方法**

采用了随机图与树结构的Poisson极限、软场递归、重加权法、复合泊松过程、马尔可夫链蒙特卡罗与大偏差估计等技术；同时运用了概率压缩、聚类与流量法等分析手段。

**📊 数据集**

使用的是理论上构造的随机 CNF 样本，具体为 Poisson(α n) 的 k‑子句集合，满足特定的迫使强度 λ_+；未使用任何外部真实数据集。

**📈 对比分析**

通过理论推导与计算机模拟验证，在满足 λ_+≤ln k+ln ln k+γ_*-1/(ln ln k)^1/16+1/ln k 的区间内，局部更新能以 O((ln ln k)^{1/16}) 的深度和常数概率翻转根变量；在高密度区间内则证明了解空间碎裂和成分质量指数衰减，说明方法在理论上与现有最优相当。

**⚠️ 局限性**

局限性包括：仅适用于随机均匀 k‑SAT，不能直接推广到非均匀或结构化实例；软场递归与 f‑加权测度在极端参数下收敛速度慢；高密度下的几何碎裂证明依赖于大偏差估计，实际实现的计算复杂度仍未给出；且对大 k 的精确阈值仍有微小误差。

---

## 245. Faster Dynamic Maximal Independent Set against Adaptive Adversaries

**arXiv ID:** 2611.05932 | [PDF](https://arxiv.org/pdf/2611.05932v1)

**作者:** Hongyi Chen `[一作]` (Nanjing University), Tianyi Zhang `[通讯]` (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种随机化的完全动态最大独立集（MIS）维护算法，能够在适应性对手（adaptive adversary）攻击下以 O(m^{5/8}) 的摊销更新时间保持图 G 的最大独立集。

**💡 创新点**

创新点在于：
- 通过引入主候选集 C₁ 与次候选集 C₂，允许低度顶点在未加入 MIS 时被高度 MIS 主导，从而显著减少高层 MIS 的重建频率；
- 设计了基于随机抽样的高层支配边列表和其逆向列表，结合两级度划分（低、中、高）实现局部修复；
- 开发了“概率游戏”（membership 与 emptiness games）来在适应性环境下证明抽样不受攻击者信息泄露影响，确保列表大小与空列表蕴含的高层邻居数保持上界；
- 引入周期性“flush”操作在候选集规模超过阈值时重建高层 MIS，保证整个算法摊销成本下降至 O(m^{5/8})。

**🔧 技术方法**

主要技术：
- 随机抽样（Bernoulli 抽样）配合几何跳跃实现高效采样；
- 多层度划分（low, middle, high）和对应的精确邻居计数；
- 高层支配边列表与逆向列表的双向维护；
- 通过潜能函数证明摊销时间；
- 两种概率游戏用以控制抽样在适应性对手下的行为。

**📊 数据集**

本文为理论研究，无使用具体数据集；所有结果均通过严格证明给出。

**📈 对比分析**

与此前最优的确定性 O(m^{2/3}) 算法相比，本文在适应性对手模型下实现了 O(m^{5/8}) 的摊销更新时间；同时结合已知的 O(m^{1/2−o(1)}) 的条件下下界，显示了与下界之间仍存在多项式间隙，但已是首个在此模型下突破 m^{2/3} 的结果。

**⚠️ 局限性**

局限性：
- 仍是摊销时间，未给出最坏情况保证；
- 只适用于组合式（combinatorial）动态算法，对比条件下的矩阵乘法假设；
- 对于高维图（m ≈ n²）时，算法的常数项可能较大；
- 目前与理论下界之间仍存在 O(m^{1/8}) 的多项式差距，进一步提升仍有空间。

---

## 246. PAYWORTHY: Can LLMs Predict Which Freelance Jobs an AI Agent Will Complete?

**arXiv ID:** 2611.05759 | [PDF](https://arxiv.org/pdf/2611.05759v1)

**作者:** Mohammad Reza Rezaei `[一作]` (Upwork Inc), Zhao Chen `[通讯]` (Upwork Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Payworthy 基准，通过让 18 种前沿语言模型在完成真实自由职业工作前预测其成功率，进而评估模型在任务 triage（筛选）中的能力。

**💡 创新点**

创新点在于：①使用了 1,494 条真实、按固定规则抽样的自由职业工作，避免人工筛选偏差；②在模型预测前不提供任何结果，真正测评“先知”能力；③提出两种 agent 描述方式（限制 vs 通用），揭示对预测质量的显著影响；④将 LLM 评审与人类评审和多种 LLM judge 交叉验证，提升实验可信度。

**🔧 技术方法**

技术手段包括：大型语言模型（GPT‑5、Claude‑opus‑5 等）生成成功概率预测；LLM 评审（gpt‑5 及其它模型家族）判断交付物是否符合要求；AUC、Calibration Error、Top‑k 选取等评价指标；逻辑回归、文本嵌入等基线模型；对比实验与人类评审。

**📊 数据集**

使用的数据集为 1,494 条从 Upwork 平台 2020‑2026 年抽样的自由职业工作（共 18.8M 条发布），并配合对应的 LLM 评审标签；公开代码及数据可在 https://github.com/upwork/Payworthy/ 获取。

**📈 对比分析**

比较方法：对 18 个模型在两种提示（limits vs generic）下的 AUC 进行对比，并与基线分类器、随机排序和人类评审结果对齐。结果显示，给出 agent 限制时 AUC 0.66–0.86，超越基线（0.82）并且能在 top‑k 选取中提升至 65–92% 的通过率；而通用描述下的 AUC 仅 0.45–0.56，接近随机。模型在自我评审中也可达 0.86 AUC，仍低于 0.94 的实验上限。

**⚠️ 局限性**

局限性包括：评审标签来自 LLM 评审，而非真实客户支付；评审未区分接受与价格；工作样本主要来自单一平台，可能不具普适性；长文本交付物可能受上下文长度限制；模型和评审版本会随时间更新，结果可复现性有限。

---

## 247. From Type to Token: Petri Nets, Individual Token Philosophy, and Individualized Causal Effects

**arXiv ID:** 2611.05703 | [PDF](https://arxiv.org/pdf/2611.05703v1)

**作者:** Fernando Delbianco `[一作]` (Universidad Nacional del Sur), Gustavo Bodanza `[通讯]` (Universidad Nacional del Sur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出基于 Petri 网的个体化因果效应框架，给出了从贝叶斯网络到 Petri 网的形式化翻译，并构建了由结构层（Petri 网）和量化层（合规预测）组成的两层系统；

**💡 创新点**

创新点在于将个体化因果效应视为 token‑level 量并利用 Individual Token Philosophy 实现；证明了贝叶斯网络与 Petri 网之间保持 d‑separation 与 do‑operator 的对应关系；通过无分布假设的结构层与分布无关的合规预测层相结合，给出有限样本覆盖保证；

**🔧 技术方法**

使用了 Petri 网（ITP）、结构化翻译 T(BN)、d‑separation 与 flow‑separation 对应、do‑operator 在 PN 中的实现、合规预测（conformal prediction）以及一系列形式化定理与命题；

**📊 数据集**

本文为理论工作，未使用任何实测数据集，所有结果均基于数学证明；

**📈 对比分析**

论文未进行实验比较，仅通过形式化定理验证对应关系，并提供了覆盖率保证；没有性能指标；

**⚠️ 局限性**

局限性包括：仅适用于二元变量，需要局部遍历性（local ergodicity）假设；局部子网相等的严格条件可能过于苛刻；未处理连续变量、非单调机制和循环因果等情况。

---

## 248. Evaluating Human-LLM Alignment on Visualization Preference

**arXiv ID:** 2611.06272 | [PDF](https://arxiv.org/pdf/2611.06272v1)

**作者:** Tanmay Bakshi `[一作]` (Microsoft), Simra Shahid `[通讯]` (Microsoft)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了意图驱动的对比可视化偏好基准，并评估当前主流视觉语言模型在判断可视化偏好时与人类偏好的契合度。

**💡 创新点**

①提出以配对比较方式量化可视化偏好，并将VLM作为偏好评判者进行验证；②将可读性、精确性、美观等主观维度与统计分布结合，评估模型与人类偏好的一致性与校准性。

**🔧 技术方法**

使用多模态视觉语言模型（Claude Opus 4.5、Claude Sonnet 4.5、GPT‑4V、GPT‑5 Chat、Phi‑4 Multimodal），自研的可视化生成流水线（基于CoT的情景构建与图表合成），以及JS散度和熵差等信息论指标。

**📊 数据集**

从 ChartUIE‑8K 数据集中提取 509 个表，人工挑选 34 个多主题数据集，生成 116 个对比性查询，进而合成 396 张可视化图，最终抽样 24 对作为评测案例。

**📈 对比分析**

通过 10 位受过可视化训练的评审对每对图表进行可读性、精确性、美观评分，并做三选（A、B、相等）偏好决策；利用 JSD 衡量模型与人类分布差异、熵差评估校准。结果显示模型与人类的 JSD 约为 0.29，显著高于校准参考 0.08，且模型在大多数对中表现出过度自信，说明偏好一致性不足。

**⚠️ 局限性**

局限性包括样本量有限（仅 24 对），难以对不同模型进行精细排名；VLM 对图表类型识别存在不稳定性；实验仅针对可视化熟练的参与者，未验证普通大众；只考虑了三维度评估，缺乏更细粒度的主观因素探究。

---

## 249. Offline Learning of Reserve Prices from Censored Bids

**arXiv ID:** 2611.06067 | [PDF](https://arxiv.org/pdf/2611.06067v1)

**作者:** Lavanya Marla `[一作]` (University of Illinois Urbana Champaign), Boxuan Zhou `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了卖家如何从被审查的历史竞标数据中学习第二价格拍卖的保留价格。

**💡 创新点**

在单调危险率（MHR）假设下，准确描述了在何种情况下可以识别出最优保留价格，即使价值分布可能仍然未知。

**🔧 技术方法**

使用了最小最大（minimax）方法来推导保留价格，并通过观察到的竞标数据估计了两个可观察的量：被审查的比例和最低历史保留价格上方的危险率。

**📊 数据集**

使用了合成数据和真实的eBay竞标数据集进行数值实验。

**📈 对比分析**

与其他方法比较时，提出的保留价格规则在可识别和不可识别的情况下都表现出较低的最坏情况收入遗憾，尤其是在MHR假设下，学习速率为O(N^-2/5)。

**⚠️ 局限性**

在不可识别的情况下，存在一些限制，尤其是当隐藏的价值分布无法完全识别时，可能导致较高的收入遗憾。

---

## 250. Reward Budgeting Reduces Premature Convergence in Reinforcement Learning for LLM Reasoning

**arXiv ID:** 2611.05942 | [PDF](https://arxiv.org/pdf/2611.05942v1)

**作者:** Mengni Jia `[一作]` (University of Cambridge), Guanjun Jiang `[通讯]` (Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出奖励预算（reward budgeting）技术，通过在每个提示的样本组内按奖励差值重新分配固定奖励，来调节强化学习的锐化压力，从而提升大型语言模型在推理任务上的最终表现。

**💡 创新点**

创新点在于用锐化因子 κ 与 κ_eff 对 RL 更新的锐化压力进行理论分析，并证明奖励预算可以自适应降低对易提示的锐化，缓解过早收敛，而无需显式熵约束。

**🔧 技术方法**

使用 KL 正则化的策略梯度框架，改进优势构造为奖励预算形式，并对 κ/κ_eff 进行实验与理论评估；同时采用分组奖励、截断和重要性比率裁剪等技术。

**📊 数据集**

训练数据集为 DAPO‑Math‑17K；评估基准包括 AMC23、AIME24、AIME25、MATH500、Olympiad (Physics) 等四类数学/推理测试集。

**📈 对比分析**

与 GRPO、DrGRPO、GSPO、DAPO、ClipCov、KLCov 等六种基线比较；奖励预算在 Qwen3‑8B 与 Llama3.2‑3B‑Instruct 上均实现了 2–12% 的准确率提升，优于传统熵控制方法。

**⚠️ 局限性**

局限性：实验仅针对二元奖励、固定 G=8 的算术推理任务；对噪声或连续奖励的理论与效果尚未验证；未系统探究组大小、训练种子和超参数的敏感性。

---

## 251. AssameseLM-E2B: An Open-Weight Assamese Language Model for Efficient Consumer-Grade Deployment

**arXiv ID:** 2611.06170 | [PDF](https://arxiv.org/pdf/2611.06170v1)

**作者:** Pragyan Phukan `[一作]` `[通讯]` (Moran College), Pragyan Phukan (Moran College)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对低资源阿萨姆语，构建并微调了基于Gemma-4的2B参数模型 AssameseLM-E2B 及其指令调优版本 AssameseLM-E2B-it。

**💡 创新点**

通过字节级 BPE 词表扩展、混合 LoRA、持续预训练和检索增强生成（RAG）指令微调，显著降低 token 化开销、消除 Bengali 语言污染，并在低资源环境下实现与大模型相当的性能。

**🔧 技术方法**

采用 Gemma-4-E2B 基础、Hybrid LoRA、参数高效持续预训练 (CPT)、监督式指令微调 (SFT)、RAG 框架以及 BPE 词表扩展等技术。

**📊 数据集**

使用精心筛选的阿萨姆语 OCR 文本、新闻抓取、Common Crawl、Wikipedia、Sangraha 以及翻译后的 Alpaca/OpenOrca 语料。

**📈 对比分析**

在 FLORES-200、RAG 任务和开放式聊天评测中与 3B–3B 规模的多语种大模型对比，取得 BPB 0.7488、RAG 96.7% 及 100% 拒绝率，显著优于通用基线。

**⚠️ 局限性**

仍存在 RAG‑prior 拒绝偏置、零样本问答时的抑制误差、罗马化文本生成质量不稳定，以及对更大上下文窗口和多模态能力的扩展需求。

---

## 252. MIGONet: a multiple-input graph neural operator with solver-free discrete manufactured-solution training

**arXiv ID:** 2611.05876 | [PDF](https://arxiv.org/pdf/2611.05876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 253. Improved approximation algorithms for capacitated cluster editing

**arXiv ID:** 2611.05916 | [PDF](https://arxiv.org/pdf/2611.05916v1)

**作者:** Cong Ma `[一作]` (Hangzhou Dianzi University), Lusheng Wang `[通讯]` (City University of Hong Kong)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对容量受限的Cluster Editing问题提出了三种改进的近似算法；

**💡 创新点**

创新点在于：①提出一种简单的“降维”过程，可将任意无容量限制的划分转换为满足容量约束的划分，并给出严格的成本上界；②将无容量问题的最新LP松弛与采样和普通pivoting相结合，再结合降维，得到1.9+ε近似；③进一步采用García‑Soriano与Schohn的条件affine pivoting规则，同样配合降维，得到更优的1.7537+ε近似；

**🔧 技术方法**

使用了Cluster‑LP线性规划松弛、随机采样、pivoting（普通和条件affine）、以及降维裁剪等技术；

**📊 数据集**

无；

**📈 对比分析**

与之前最优的5.37近似相比，3.7730+ε、1.9+ε和1.7537+ε分别大幅改进；与无容量限制的1.3865近似相比，后两算法也实现了更好的性能；

**⚠️ 局限性**

仅适用于统一容量限制的情形；算法仍为随机化，复杂度为多项式，但指数不受k影响；未给出下界或针对非均匀容量的分析。

---

## 254. KVGuard: A KV Cache Safety Checker for LLM Serving Systems

**arXiv ID:** 2611.06138 | [PDF](https://arxiv.org/pdf/2611.06138v1)

**作者:** Jonguk Cheong `[一作]` (Seoul National University), Byoungyoung Lee `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 KVGuard，一种针对LLM服务系统中 KV 缓存管理的内存安全检查器。

**💡 创新点**

首次在 KV 缓存层面构建了可追踪的阴影块抽象，结合五级检查实现对分配、表格和 GPU 访问的及时检测。

**🔧 技术方法**

使用 Python 与 CUDA 实现阴影块、CPU 侧钩子以及 GPU 内核插桩，并在 vLLM 与 SGLang 上集成。

**📊 数据集**

在 Llama 3.2 1B、Llama 3.1 8B、Qwen 2.5 14B 与 Qwen 3.5 4B 模型上，使用 ShareGPT V3、LongBench、MMLU 等公开提示集进行评测。

**📈 对比分析**

与 NVIDIA Compute Sanitizer 对比，KVGuard 在 TPOT/TTFT 上平均提升 3~7% 且 KV 占用仅 0.05–0.11%，显著低于 Compute Sanitizer 的百倍开销，同时发现 29 个历史 Bug 与 8 个新 Bug，且无误报。

**⚠️ 局限性**

仅针对 KV 缓存分配与访问错误，无法检测哈希碰撞、量化等其他模型或服务层 Bug，且 GPU 侧仍有数百毫秒的延迟影响。

---

## 255. The Complexity of Unpredictability

**arXiv ID:** 2611.05956 | [PDF](https://arxiv.org/pdf/2611.05956v1)

**作者:** Aryan Agarwala `[一作]` (Max Planck Institute for Informatics), Dani Dorfman `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并研究了中位数预测器（middle‑bit predictors），定义了在给定一组中位预测器和误差参数ε时，输出一个多集，使得每个预测器在多集中成功率小于½+ε的判定问题（UNPRED）。作者进一步研究了该问题在确定性多项式时间和电路限制（TC¹）下的复杂性，并将其与催化计算（catalytic logspace）及伪随机性中的下一位预测器等经典问题相连接。

**💡 创新点**

创新点主要包括：①首次将中位预测器与催化计算中的空间恢复问题联系起来，证明UNPRED几乎等价于证明CL⊆FP；②给出了一种压缩或随机（compress‑or‑random）框架下的Catalytic Logspace算法，能在确定性多项式时间内解决UNPRED（在TC¹电路输入下）；③证明了任何仅将预测器视为黑盒函数的确定性算法都需要指数时间，从而阐明了解决UNPRED必须利用电路结构的必要性；④将UNPRED与近似悲观估计器（pessimistic estimators）以及弱极大（Weak Excess）等问题等价，为解决催化计算和伪随机性中的开创性难题提供了新的途径。

**🔧 技术方法**

技术上，本文主要使用了：压缩或随机（compress‑or‑random）策略；低哈密顿权重字符串的压缩技术；对中位预测器的均值/方差分析；构造确定性超立方体有向图和对应的预测器集合；以及在Catalytic Logspace与In‑Place计算模型下的算法设计和分析。

**📊 数据集**

本工作属于理论计算复杂性研究，不使用实验数据集；所有结果均为理论证明与复杂性归约。

**📈 对比分析**

由于该工作为理论性质，作者未提供实验比较或性能测评；主要通过理论归约展示UNPRED与CL⊆FP、PSEUDO、Weak Excess等问题的复杂性关系，证明了若能在多项式时间内解决UNPRED（TC¹输入），则可解催化计算的关键开放问题。

**⚠️ 局限性**

局限性：目前仅提供了在电路为TC¹时的确定性多项式时间算法；对于更一般的电路类尚无多项式时间解法；此外，证明的指数下界仅适用于黑盒查询模型，未说明在利用电路结构时是否仍存在不可压缩的硬实例；最后，尚未给出具体实现或可扩展到更大规模问题的实例，因而实际应用层面的可行性仍待进一步研究。

---

## 256. SonoDyno: A Sonic Dynamometer for Engine Speed and Load Estimation from Audio

**arXiv ID:** 2611.06262 | [PDF](https://arxiv.org/pdf/2611.06262v1)

**作者:** Robin Doerfler `[一作]` (Impulse Audio Lab GmbH), Beat Rossmy `[通讯]` (Impulse Audio Lab GmbH)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了SonoDyno，一种仅基于麦克风音频即可估计发动机转速、气缸数和负载强度的系统。

**💡 创新点**

创新点在于使用三分辨率秩能量表、共享的计数权重、旋转动力学解码器以及负载归一化目标，实现了在未知发动机、无传感器的情况下高精度估计。

**🔧 技术方法**

技术上采用卷积网络对秩能量进行视图细化、融合和计数加权评分，利用Viterbi动态规划解码速度轨迹，并用GRU预测负载强度。

**📊 数据集**

训练和评估使用约116.7小时的混合语料，包含8台真实发动机、370台模拟发动机以及14.4小时的程序重采样音频。

**📈 对比分析**

在Oracle秩条件下与传统音高跟踪器（PESTO、CREPE等）对比，SonoDyno在真实验证集上c5≈0.977、c2≈0.969，速度误差≤2%占比≈94%，计数识别准确率80%，负载估计MAE≈0.12，显著优于现有方法。

**⚠️ 局限性**

局限性包括对快速转速变化、低速空闲以及不同采集环境的鲁棒性仍不足，且实验主要在实验室环境和有限的气缸数下验证，未覆盖道路噪声等实际驾驶条件。

---

## 257. Time-to-Goal: Learned Temporal Metrics for Long-Horizon Planning with JEPA World Models

**arXiv ID:** 2611.05847 | [PDF](https://arxiv.org/pdf/2611.05847v1)

**作者:** Ruilong Yu `[一作]` (University of Electronic Science and Technology of China), Fei Ye `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了Flat LeWM在长期规划中欧氏距离成本的局限性，提出并训练一种轻量级的时间度量（Time-to-Goal, T2G），用来替代原来的欧氏成本，从而在保持世界模型不变的前提下提升规划成功率。

**💡 创新点**

创新点：①仅利用轨迹的时间顺序（不需要奖励或策略标签）通过margin ranking loss学习时序成本；②在保持原有Encoder/Predictor冻结的情况下，只添加0.182M参数的映射网络；③对比稠密和终点窗口两种成本聚合方式，揭示终点聚合对长程任务的优势；④在四个模拟基准上将Flat LeWM的成功率从32.7%提升至63.0%，接近或超过层次化H-JEPA。

**🔧 技术方法**

技术方法：Joint-Embedding Predictive Architecture (JEPA) 的Encoder+Predictor；SIGReg正则化；欧氏距离成本分析；Time-to-Goal的margin ranking训练；梯度规划（AdamW）和采样规划（CEM）；成本聚合（dense vs. last‑n）。

**📊 数据集**

使用的数据集：FourRoomDistractors、Visual AntMaze、Push‑T、OGBench Cube 四个离线轨迹基准，涵盖导航、迷宫、操控等任务。

**📈 对比分析**

与Flat LeWM（欧氏成本）、Flat LeWM+T2G（稠密/last‑n）、H-JEPA（2/3/4级）进行对比。T2G last‑n在所有环境的平均成功率从32.7%提升至63.0%，在FourRoom和Cube中甚至超过对应级别的H-JEPA；dense T2G在AntMaze、FourRoom、Push‑T提升明显，但在Cube表现下降，说明成本聚合策略对任务影响大。

**⚠️ 局限性**

局限性：仅在模拟环境验证，缺乏真实机器人实验；需要预先收集足够的离线轨迹来训练T2G；对中间状态成本聚合的选择需要手工调参；未将T2G与层次规划结合，可能进一步提升性能；在动态或高噪声环境下的鲁棒性尚未评估。

---

## 258. Mobility-Aware Mobile Edge Computing for Energy-Balanced Data Collection in IoT-Enabled Underwater Sensor Networks

**arXiv ID:** 2611.06186 | [PDF](https://arxiv.org/pdf/2611.06186v1)

**作者:** Nguyen Thi My Binh `[一作]` (Hanoi University of Industry), Ho Viet Duc Luong `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于移动边缘计算的能量平衡数据采集框架MECEBDC，用于IoT支持的水下传感器网络，联合优化能量感知聚类、集群头选择和AUV路径规划。

**💡 创新点**

创新点包括：①将AUV的真实水动力学（浮力、重力、三维运动）融入路径规划；②在MEC框架下联合考虑能量平衡与连通性；③采用多种改进的元启发式算法（GA、AE‑PSO、LF‑PSO）实现全局优化。

**🔧 技术方法**

使用了K‑means聚类、遗传算法、粒子群优化及其自适应与莱维飞行变体、基于磁感应与声学通信的能量与时延模型，以及AUV三维运动学模型。

**📊 数据集**

数据集为基于Python的仿真环境，随机布置150–550个节点于400 m×400 m×400 m的三维水域，采用指定的能量、速率与通信参数。

**📈 对比分析**

与贪心基线和现有方案对比，主要评价指标为AUV行程时间、网络生命周期与行程时间标准差。结果显示：GA在旅行时间和网络寿命上表现最佳，AE‑PSO和LF‑PSO在大型网络中可保持较高探索性并逼近GA性能，贪心方法性能最差。

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，缺乏真实部署实验；节点位置假设已知且静止，未考虑漂移或障碍物；能量模型相对简化；改进算法计算复杂度高，可能不适合极大规模或实时部署。

---

## 259. Agentic Behavior Engineering: A Design-by-Contract Methodology for Specifying, Verifying, and Governing Enterprise AI Agents

**arXiv ID:** 2611.05454 | [PDF](https://arxiv.org/pdf/2611.05454v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 260. Towards Robust Long-Lived Agents: Benchmarking Experience Lifecycles in Evolving Environments

**arXiv ID:** 2611.05450 | [PDF](https://arxiv.org/pdf/2611.05450v1)

**作者:** Shijue Huang `[一作]` (Hong Kong University of Science and Technology), Yiren Feng `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了ReScopeBench基准，用于评估代理在18个交互式环境中获取、重用和修订结构化与规则性经验的能力；

**💡 创新点**

创新点在于：①构建“先经验再任务”的框架，先识别可发现的经验，再设计任务序列与无声环境变化；②同时评估任务成绩与经验覆盖、更新与恢复，揭示经验维护的细粒度问题；

**🔧 技术方法**

采用Hermes代理框架、GPT-6 Astra评估器、基于工具调用的交互模拟与自动评分；

**📊 数据集**

使用18个编程实现的交互式环境，包含352个任务，分布在9个应用域，含26次无声环境变化；

**📈 对比分析**

对20个模型（10个商业、10个开放权重）进行对比，评估基线、经验增强与漂移启动三种任务表现，并量化结构与规则经验覆盖率、更新率和恢复时效；结果显示保留经验可提升任务成绩，但规则经验覆盖率普遍低于结构经验，更新率仍低，恢复时效与即时更新不完全相关；

**⚠️ 局限性**

局限性在于：①环境均为模拟实现，真实世界复杂性有限；②经验更新仍大多不及时或不完整；③评估侧重经验覆盖与更新而非经验的泛化与可解释性；

---

## 261. A Controlled Experimental Evaluation of Configuration and Cache Architecture in LLM Serving Engines

**arXiv ID:** 2611.05991 | [PDF](https://arxiv.org/pdf/2611.05991v1)

**作者:** Muhammad Faizan Anwar `[一作]` (Thompson Rivers University), Sina Keshvadi `[通讯]` (Thompson Rivers University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较了 vLLM 与 SGLang 两大 LLM 服务引擎的性能差距，并在受控实验框架下拆解出导致差距的配置和缓存架构因素。

**💡 创新点**

首次在受控实验中将 chunked‑prefill 配置与缓存粒度分离，量化各自对性能的贡献，并提出六项实用的对比建议。

**🔧 技术方法**

构建了基于 Kubernetes/GKE 的测试平台，使用种子化工作负载生成器模拟多智能体工作流；通过 TTFT/TPOT 等指标、几何平均、对数 t 检验和流队列模型进行分析。

**📊 数据集**

采用 Llama‑3.1‑8B‑Instruct（16,384‑token 上下文）生成的合成提示，按比例共享前缀，并用 Gamma 到达过程模拟 bursty 负载。

**📈 对比分析**

在默认配置下，vLLM 的 TTFT 可达 64×，匹配 chunked‑prefill 后可消除 78% 容量差、30–67% 延迟差，剩余尾部延迟 SGLang 优于 vLLM 2.4–6.4×，表明请求调度顺序是主要残留因素。

**⚠️ 局限性**

实验仅在单 GPU、单模型、单上下文长度、单副本环境下验证；不同硬件、模型、版本或真实工作负载可能导致结果变化；未完整消融请求顺序、内存预算、注意力核等差异。

---

## 262. ChangeJEPA: Predict Before You Compare for Remote Sensing Change Detection

**arXiv ID:** 2611.05570 | [PDF](https://arxiv.org/pdf/2611.05570v1)

**作者:** Buddhi Wijenayake `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**通讯引用:** 30863 | [OpenAlex ID](https://openalex.org/A5004716468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于双时间 Joint‑Embedding Predictive Architecture（ChangeJEPA）的遥感变化检测框架，利用先验和更新后状态预测与直接特征比较相结合，生成密集变化分割。

**💡 创新点**

创新点在于：① 引入双预测（先验与更新）并将两者与观测后状态进行对比，形成五种互补的变化证据；② 将遥感变化检测从传统的特征差异化转向“先预测再比较”，大幅降低光照、季节、几何等无关差异的干扰；③ 将 JEPA 的自监督潜在预测与监督分割任务联合训练。

**🔧 技术方法**

使用了 Vision Transformer 作为编码器，MaxViT 用于上下文融合与解码，CBAM 进行证据聚合，分级 MaxViT 解码器配合多层跳跃连接，最后使用可微分形态学细化；损失方面结合 Smooth‑L1（预测）与 BCE、Tversky、Lovasz（三分类）监督。

**📊 数据集**

在三大公开遥感变化检测数据集 SYSU‑CD、LEVIR‑CD+、CDD 上进行实验。

**📈 对比分析**

与同一解码器结构的 Siamese baseline 进行对比。ChangeJEPA 在所有数据集的 IoU、F1 及 OA 均优于 baseline，最高提升 1.51 分（IoU）。

**⚠️ 局限性**

主要限制包括：仍只适用于两帧输入，无法直接处理多观测序列；对不同传感器、配准误差的鲁棒性尚未充分验证；实验仅覆盖三类场景，需进一步评估在更广泛条件下的表现。

---

## 263. Faster Cluster Decomposition with Stronger Bi-Criteria Guarantee

**arXiv ID:** 2611.06008 | [PDF](https://arxiv.org/pdf/2611.06008v1)

**作者:** He Sun `[一作]` (Institutes of Advanced Technology, Chinese Academy of Sciences), Yacong Zhou `[通讯]` (Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了几种近线性时间的算法，能够在图中得到满足内导度高、外导度低的 k 个或若干个簇，并在保证可实现 Oveis Gharan 与 Trevisan 提出的双准则（inner‑conductance 与 outer‑conductance）后，完成整个顶点集的划分。

**💡 创新点**

（1）首次把存在性证明转为可实现的近线性算法；（2）设计了一个基于 “fair‑cut” 与 “min‑cut” 的提升框架，能够在 ρ_G(k+1)/ρ_G(k) 的显著间隙下将任何 α‑近似解提升至 (1+ε)‑近似；（3）用该框架结合谱划分（SpectralPartitioning）实现了一个真正的 O(mk²) 或 O(mk log²k) 的 k‑划分算法，性能远超之前的 O(knm²) 级别。

**🔧 技术方法**

核心技术包括：
- 归一化拉普拉斯谱与高阶 Cheeger 不等式。
- “fair‑cut trimming” 通过构造带有 s‑t 源端口的辅助图来保证剪切保持低外导度且不削弱内部连通性。
- 基于 min‑cut 的 ApproximationBoost 方案，利用罚项 q_r 迭代逼近最优 k‑way conductance。
- 递归二分与构造分区（ConstructPartition）保证得到的簇不被分割。
- 线性时间的 min‑cut 计算（近似），以及 SpectralPartitioning 的近线性实现。

**📊 数据集**

该工作属于理论算法研究，没有在具体实验数据集上评估，所有结果均为理论证明与渐进时间复杂度分析。

**📈 对比分析**

与 Oveis Gharan & Trevisan 2014 年的存在性证明和 O(mk²·w_{max}/w_{min}) 级别算法相比，新算法在时间上从多项式降为近线性（O(mk²) 或 O(mk log²k)）。
在需要 ρ_G(k+1)/ρ_G(k) 较大（≥ 18(α+1)(1+ε)/α）的情况下，ApproximationBoost 能把任何 α‑近似解提升至 (1+ε) 近似，仅需 O(k(1+logα)) 次 min‑cut；
在无间隙假设下，新的谱分区算法在 O(mk²) 时间内得到与旧算法相同的内外导度保证。

**⚠️ 局限性**

1) 需要 ρ_G(k+1)/ρ_G(k) 较大的显著间隙，若间隙不足则提升框架失效；
2) 需要提前知道 λ_k(ℒ_G) 与 λ_{k-1}(ℒ_G) 的上下界（或至少可近似），在实践中获取这些谱参数不一定易于实现；
3) 对于非常稀疏或权重极端不均的图，常数因子与 log n 级的乘子可能导致实际运行时间与理论近线性偏差较大；
4) 该算法侧重于理论证明，缺乏对实际大规模网络的实验验证。

---

## 264. RESHPart: Feedback-Reshaped Spectral Embedding for Hypergraph Partitioning

**arXiv ID:** 2611.05729 | [PDF](https://arxiv.org/pdf/2611.05729v1)

**作者:** Rongjian Liang `[一作]` (NVIDIA), Brucek Khailany `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于反馈驱动的谱超图划分框架FRESHPart，通过几何反馈和搜索反馈两种机制自适应地重新加权超边并重新计算拉普拉斯矩阵，从而改进谱嵌入以提升划分质量。

**💡 创新点**

创新点包括①发现并利用射线（ray）结构对谱嵌入几何进行诊断，并证明其与划分质量的相关性；②设计了几何反馈和搜索反馈两种可自适应调节的重加权策略，实现对谱嵌入的持续改进；③在GPU上实现，保持线性时间复杂度并实现SOTA划分质量。

**🔧 技术方法**

使用规范化双边拉普拉斯、Spherical K‑means聚类、ray concentration度量、CMA‑ES投影搜索、CuPy GPU加速、Deep Graph Library、线性时间复杂度分析等技术。

**📊 数据集**

主要基准为Titan23 VLSI电路超图集合，后续在ISPD98基准上验证方法的迁移性能。

**📈 对比分析**

通过与hMETIS、SpecPart、MedPart、KaHyPar、SHyPar、TritonPart、mtKaHyPar、HySpecPro等多种SOTA分区器在同一硬件（单张NVIDIA V100）上进行对比；FRESHPart在22个Titan23基准上在ϵ=2%时平均cut下降约13%，ϵ=20%时约25%，并在所有基准上获得最高或近似最高质量；在单GPU上完成时间均低于120秒，性能与HySpecPro相当但划分质量更佳。

**⚠️ 局限性**

目前仅针对二分划分，扩展到多路划分需要进一步研究；对射线数目K_r的选择仍有一定敏感性；方法需要多次拉普拉斯求解，虽然保持常数因子，但在极稀疏或超大超边的超图上可能面临挑战。

---

## 265. Cibersapiens: Gamifying Cybersecurity Education to Engage Secondary Students

**arXiv ID:** 2611.05587 | [PDF](https://arxiv.org/pdf/2611.05587v1)

**作者:** Soraya Frías-Guadalupe `[一作]` (Universidad Rey Juan Carlos), Carlos Castaneda-Marroquín `[通讯]` (Serval Networks)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究设计并实施了名为 Cibersapiens 的互动式游戏化网络安全教育平台，面向 12–16 岁的中学生，融合课堂教学、家庭与教师协同学习与在线博客，旨在提升学生对网络风险的认知与自我保护能力。

**💡 创新点**

创新点在于将游戏化与主动学习相结合，构建多维度（概念、程序、态度）模块，并通过跨界协作（学生、教师、家长）形成全员参与的数字安全生态；同时提出可持续更新与评估的实践框架。

**🔧 技术方法**

技术手段包括基于 Genially 与 Educaplay 的交互式游戏（如逃脱室、数字侦探、互动地图等）、在线学习平台（EducaMadrid）、以及信息共享博客和协作工具（Wikis、Padlet）。

**📊 数据集**

数据来源主要为课堂实践的预测验/后测游戏成绩、学生学习日志、观察清单及问卷反馈，并结合 OER 资源与国家网络安全机构（INCIBE）提供的教材。

**📈 对比分析**

通过对比预测验与后测的定量得分、行为观察指标以及满意度问卷来评估成效，初步结果显示学生在知识掌握、风险识别与安全行为意愿方面均有显著提升。

**⚠️ 局限性**

局限性包括样本规模有限、缺乏对照组与长期跟踪、教师数字安全专业培训不足，以及对新兴网络威胁的内容更新滞后。

---

## 266. Analysis of Local Open-Weight LLM Agents on Vulnerability-Reproduction Tasks

**arXiv ID:** 2611.06257 | [PDF](https://arxiv.org/pdf/2611.06257v1)

**作者:** Jorge García-Carrasco `[一作]` (University of Alicante), Juan Trujillo `[通讯]` (University of Alicante)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在单一消费者 GPU 上对 10 个 CyberGym 漏洞再现任务，使用 12 种不同规模、架构和量化的 Qwen 开源 LLM 进行评估，完成 360 次跑和 3,501 次 PoC 提交。

**💡 创新点**

创新之处在于首次系统地量化比较了本地部署的开源模型在漏洞再现任务上的性能，并揭示了模型规模≥27B、最新一代以及 FP8 量化可显著提升成功率的规律。

**🔧 技术方法**

所用技术包括 Qwen 系列 LLM、SGLang 推理引擎、pi coding agent 工具循环、Post‑training BF16/FP8/INT4 量化以及多种工具调用（读取文件、执行命令等）。

**📊 数据集**

实验数据集为 CyberGym 选定的 10 个代表性任务，包含源码树、缺陷描述以及对应的参考 PoC。

**📈 对比分析**

比较方法基于跑次数、成功率、提交次数、wall‑clock 运行时、token 产出等指标；结果显示 27B BF16/FP8 配置在成功率（≈67%）与速度上领先，MoE 在同等能力下更快但 token 消耗更高。

**⚠️ 局限性**

主要局限包括仅评估 10 个任务、固定 7200 秒时钟预算导致部分跑被截断、量化实验仅覆盖 Qwen 3.5 系列、以及仅使用单一 pi agent harness。

---

## 267. LIBRA: Programmatic Induction of Executable Tool Libraries

**arXiv ID:** 2611.05445 | [PDF](https://arxiv.org/pdf/2611.05445v1)

**作者:** Zekai Li `[一作]` (Brown University), Jiapeng Zhang `[通讯]` (University of Southern California)

**通讯引用:** 439 | [OpenAlex ID](https://openalex.org/A5100662070)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LIBRA 框架，利用 LLM 生成工具可执行合同并在沙箱中进行反事实测试，构建可执行工具库，自动学习工具间的运行时组合关系；在检索阶段通过该库一次性激活前置工具，从而补齐仅基于语义相似度检索遗漏的前置工具。

**💡 创新点**

创新点在于：①无需依赖工具使用轨迹或训练；②通过反事实 sandbox 执行验证真正的依赖关系，避免仅凭元数据或语义相似度产生误匹配；③生成的工具库是查询无关的，可复用在单轮检索和多轮 agentic 检索中。

**🔧 技术方法**

使用技术包括：LLM（如 DeepSeek‑v4‑Flash）生成可执行合同，沙箱执行及结果反馈循环；基于输入/输出契约的反事实测试；图传播机制（一次 hop 传播）提升前置工具得分；密集检索（Qwen‑3 Embedding 0.6B）与图传播相结合。

**📊 数据集**

数据集：AppWorld、BFCL、SciToolEval 三个工具检索基准，分别涵盖日常应用、文件系统/工具 API 以及科学计算流水线；对每个基准构建统一的工具库并评估单轮与 agentic 检索。

**📈 对比分析**

与 BM25、QTS、ToolREX、TGR‑Adapt、Ego‑Trip、SciToolKG 等基线比较；单轮检索平均提升 Recall 7.9% 与 NDCG 3.7%；在 ToolReAGt 代理检索中平均提升 Recall 4.5% 与 NDCG 2.9%；在 AppWorld、BFCL、SciToolEval 上的实验均显示 LIBRA 在大多数设置下获得最高 Recall 或 NDCG。

**⚠️ 局限性**

局限性包括：需要可在隔离、可重置沙箱中执行的真实工具；构建阶段耗时（CPU 资源消耗）且仅对可执行的工具有效；反事实测试只能验证已提出的候选关系，无法覆盖查询特定环境下的前置工具；若消费者未被种子检索，则相关前置工具无法被激活。

---

## 268. Follow or Surpass? Collaboration, Sound, and the Recognition of Influence in Lineages of Jazz Trumpet Players

**arXiv ID:** 2611.05750 | [PDF](https://arxiv.org/pdf/2611.05750v1)

**作者:** Onur Tuncay Bal `[一作]` (Central European University), Mark Wittek `[通讯]` (Central European University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文以爵士小号手的专家编制谱系为起点，结合录音集、社交网络和音频嵌入，研究了影响关系在社交与音频相似度上的结构化特征

**💡 创新点**

创新点在于首次将专家认可的纵向影响关系与音频相似度、直接合作及共同合作者重叠三种维度结合，并用ERGM检验它们对影响网络的独立与交互效应

**🔧 技术方法**

采用了指数随机图模型（ERGM）进行网络结构建模，同时使用VGGish预训练模型提取录音的128维嵌入以衡量音频相似度

**📊 数据集**

数据来源为McNeil与Berlin的爵士小号谱系（52名玩家），Jazz Discography在线数据库（约5000个录音会话，超过18000名合作者），以及通过Deezer API获得的16,000段音频预览

**📈 对比分析**

模型比较通过逐步加入协作、音频相似度及交互项的nested ERGM实现，结果显示社交与音频相似度各自显著，且二者交互呈负向，说明高相似度共存时影响关系概率下降

**⚠️ 局限性**

局限性包括：研究仅覆盖爵士小号手，且使用的是非随机抽样的专家谱系；模型无法给出因果解释；音频特征是对完整录音的整体描述，未能分离个体演奏风格；对其他音乐或文化领域的外推性未知

---

## 269. METRIC-GS: Selective Metric-Scale Estimation for 3D Gaussian Splatting

**arXiv ID:** 2611.06020 | [PDF](https://arxiv.org/pdf/2611.06020v1)

**作者:** Sooyeun Yang `[一作]` (Stony Brook University), Jongseong Brad Choi `[通讯]` (SUNY Korea)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计了一种仅需使用内部语义内容即可为3D Gaussian Splatting重建估计尺度的选择性尺度估计器，并与现有基于深度的尺度估计器做了混合使用。

**💡 创新点**

创新点在于：①利用冻结的2D分割教师通过可见性感知投票给3D高斯赋标签；②将门、桌子、天花板高度等标准尺寸作为实例级度量锚点并在鲁棒最大似然框架下融合；③设置内点质量阈值实现显式拒绝，拒绝时自动路由到更精确但代价更高的度量模型；④整个系统在冻结后不再调整参数，保证了公开评测的可复现性。

**🔧 技术方法**

主要技术包括：3D Gaussian Splatting（3DGS）重建、Mask2Former 2D教师分割、可见性滤波的投票机制、实例级语义尺寸测量、Geman‑McClure鲁棒融合、后期估计与拒绝阈值判定、以及与MASt3R等度量模型的混合部署。

**📊 数据集**

使用的数据集有 ScanNet++（47个场景，19个开发集、28个保留集），ARKitScenes 50个视频，TUM、ETH3D、Replica等公开基准，以及作者自采集的13个手机/无人机照片；所有数据集均为真实拍摄的室内场景。

**📈 对比分析**

与基线（量化楼层/天花板 heuristic、三种 monocular depth 对齐方法）相比，本估计器将均值误差从 21.1% 降至 16.6%（在主动修正的 28 场景中 9.3%）。与 MASt3R 的直接对比，混合部署实现 9.4% 的误差，仅调用 MASt3R 60% 的次数，保持了相似的精度。系统在所有公开基准上的性能也保持在 7–12% 左右。

**⚠️ 局限性**

局限性包括：仅适用于可见建筑结构的室内照片；对教师分割质量敏感，强分割能进一步降低误差；对部分实例的测量不完善，导致误差增大；依赖预设的尺寸先验，跨地区或特殊建筑可能需要重新设定；在无锚点或外景、合成外观场景中会直接拒绝。

---

## 270. Route to Remember: Adaptive Memory Selection for LLM Agents

**arXiv ID:** 2611.05636 | [PDF](https://arxiv.org/pdf/2611.05636v1)

**作者:** Zeyu Fang `[一作]` (George Washington University), Chenlei Guo `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过在自回归语言模型上训练用户专属LoRA适配器，实现了将用户的历时记忆（事件性与语义性）内化到模型参数中，并结合验证-再生成路由器将权重记忆与文本检索融合，显著提升长期记忆性能。

**💡 创新点**

创新点在于提出并验证了“按政策自蒸馏”（OPSD）方法，使权重记忆在无上下文推理时超越传统文本记忆；同时设计了“验证-再生成”（VtR）路由机制，实现对权重记忆答复的实时核查与纠正。

**🔧 技术方法**

技术细节包括：LoRA低秩适配器、按政策自蒸馏训练目标、检索-生成的文本记忆存储、基于答案可信度的验证器以及使用Qwen3.5-9B和Claude Sonnet 4.6作为评估与判定工具。

**📊 数据集**

实验使用了多会话长期记忆基准HaluMem以及独立多会话基准LoCoMo，覆盖基本事实回忆、冲突记忆、动态更新、多跳推理、泛化与应用以及记忆边界等六类问题。

**📈 对比分析**

与文本记忆基线（全上下文、提取上下文、MemOS检索）相比，OPSD权重记忆在HaluMem上整体准确率达到59.9%（高于51.9%/47.8%），VtR进一步提升至66.1%；在成本方面，权重记忆在写入成本较高但查询成本低于检索，适用于高频复用场景。

**⚠️ 局限性**

局限性包括：权重记忆随着时间逐渐遗忘；无法识别未记录信息导致的虚假答案；对训练数据和超参数高度敏感，需要额外的训练开销；在记忆边界/缺失类问题上仍逊色于文本检索。

---

## 271. Visual Forensic Detection of Fabricated Bangla News Photocards using Deep Learning

**arXiv ID:** 2611.05403 | [PDF](https://arxiv.org/pdf/2611.05403v1)

**作者:** Anindya Kundu `[一作]` (University of Dhaka), Muhammad Ibrahim `[通讯]` (University of Dhaka)

**通讯引用:** 4691 | [OpenAlex ID](https://openalex.org/A5028428532)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究Bangla新闻photocard伪造检测，构建专属数据集并训练小型卷积神经网络，比较不同模型性能；

**💡 创新点**

创新点包括：①提供完整的Bangla photocard数据集与16种降噪增强管线；②证明平面ANN无法提取视觉取证信号；③设计全批归一化的4块CNN并提出ANN头warm‑start迁移策略；④引入预训练YOLOv8作为强基准；⑤采用泄漏无关的分组5折交叉验证。

**🔧 技术方法**

技术手段：卷积神经网络（4个卷积块+BN+全局平均池化+dropout），迁移学习（ANN权重warm‑start），YOLOv8分类头，16种图像增强，梯度裁剪、学习率调度、批归一化。

**📊 数据集**

使用的数据集：2046张真实与伪造Bangla photocard（969真、1077假），来源于17个官方频道、伪频道、匿名个人和公开群组；通过16种变换扩增至32,736张。

**📈 对比分析**

比较方法：采用分组5折交叉验证；结果显示全平面ANN≈52%准确率，零起点CNN≈87.45%，迁移CNN≈88.75%，预训练YOLOv8≈97.36%；CNN显著优于ANN，预训练模型进一步提升约8个百分点。

**⚠️ 局限性**

局限性：①数据来源偏差，真伪样本来源不对称；②伪造样本聚集在少数渠道；③未提供来源独立的最终测试集；④迁移效果与训练调度混合，未单独评估；⑤YOLOv8优势不单靠架构，需进一步对比不同预训练基准。

---

## 272. One Strategic Woman: Sharp Ranks and a Persistent Core

**arXiv ID:** 2611.06175 | [PDF](https://arxiv.org/pdf/2611.06175v1)

**作者:** Ron Kupfer `[一作]` `[通讯]`, Ron Kupfer

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在一个平衡的随机稳定婚姻市场中，研究了一名固定女性如何通过男性提议的延迟接受机制来操控以获得最佳伴侣。

**💡 创新点**

证明了女性的期望平均真实伴侣排名渐近于(ln n)^2，改进了之前的O((ln n)^4)界限，同时男性的期望排名渐近于n/(2ln n)。

**🔧 技术方法**

使用了男性提议的延迟接受算法（DA）和概率论中的极限定理。

**📊 数据集**

使用了n名男性和n名女性的独立均匀严格完全偏好数据集。

**📈 对比分析**

通过与之前的研究进行比较，确定了双方的尖锐平均排名和联合极限，并量化了剩余的稳定核心，结果显示女性的平均排名在报告的稳定核心内收敛。

**⚠️ 局限性**

限制在于该模型假设了偏好的独立性和均匀性，可能不适用于更复杂的偏好结构。

---

## 273. Drift-adaptive Representation Preservation for Domain-Incremental Learning

**arXiv ID:** 2611.06093 | [PDF](https://arxiv.org/pdf/2611.06093v1)

**作者:** Hyunmin Kong `[一作]` (Sungkyunkwan University), Jitae Shin `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在域增量学习中，作者提出一种基于表示漂移的自适应表示保存框架，利用冻结的预训练视觉模型通过测量域-类概念的漂移来进行知识保持。

**💡 创新点**

创新点在于：① 用单一漂移信号同时驱动高漂移概念的强化保留和低漂移概念的原型构建，实现对不同域-类的非均匀保护；② 利用JEPA式上下文-目标预测关系在不保存原始数据的情况下实时测量漂移。

**🔧 技术方法**

技术包括：冻结预训练骨干、轻量化编码器与预测器、JEPA式上下文-目标预测、漂移适应性一致化损失、鲁棒原型构建、EMA教师等。

**📊 数据集**

实验基准为 DomainNet、Office-Home、CORe50 和 CDDB-Hard 四个标准域增量学习数据集。

**📈 对比分析**

与 CODA-Prompt、OS-Prompt、MoE、GraphKeeper 等方法对比，作者方法在四个基准上保持最高或竞争性的准确率，同时遗忘率低，且仅使用 3 GB GPU 内存和约 13.8 MB 存储。

**⚠️ 局限性**

局限性包括：对极端域数或大规模样本的伸展性未评估；仅使用冻结骨干，可能在跨域差异极大时受限；漂移测量依赖上下文-目标预测的质量，若预训练模型不足会影响保留效果。

---

## 274. Contextual Pictograms for Repairing Multilingual ASR Errors

**arXiv ID:** 2611.05637 | [PDF](https://arxiv.org/pdf/2611.05637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 275. Ctrl-F: Force as an Interface for Humanoid Loco-Manipulation

**arXiv ID:** 2611.05507 | [PDF](https://arxiv.org/pdf/2611.05507v1)

**作者:** Mengjie Zhao `[一作]` (University of California, Berkeley), Masayoshi Tomizuka `[通讯]` (University of California, Berkeley)

**通讯引用:** 29014 | [OpenAlex ID](https://openalex.org/A5064077634)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种层次化的 Humanoid 运动与力控制框架 Ctrl-F，能够在运动追踪基础上通过显式 3D 力命令实现对环境的物理交互；

**💡 创新点**

创新点在于：①通过“逆力训练”让通用运动追踪器学习在无对象情况下响应力命令；②在高层引入力规划器，利用解析力模型与强化学习实现在线力适配；③实现了多任务、跨物体、可在线自适应的全局力接口；

**🔧 技术方法**

技术包括：逆力训练、力规划器（基于解析力分配+PPO 与模仿学习）、力-位置双模控制器、基于 AprilTag 的物体姿态估计、Unity/Sim-to-Real 部署；

**📊 数据集**

使用 AMASS、SONIC 以及自制的运动片段（9 组手、足、复合任务）与多种物体（箱子、推箱、拖箱、踢箱等），对象质量随机化；

**📈 对比分析**

与无力通道基线 SONIC 对比，Ctrl-F 在负载下完成率>90%（1.5× 质量）且可在线调节力，硬件实验中完成了拖、推、举、踢、扫等任务，且对中途加载有自适应；

**⚠️ 局限性**

局限性：需依赖可靠的物体姿态感知（AprilTag）；训练仅覆盖有限运动集，无法对未见运动泛化；力命令误差约12‑18%，未完全匹配实际施加力；

---

## 276. FireOPS: A Human-Centric Emergency Operations Platform

**arXiv ID:** 2611.06176 | [PDF](https://arxiv.org/pdf/2611.06176v1)

**作者:** Alexander Wallner `[一作]` (District Fire Command Perg - Team FireOPS), Eduard Paireder `[通讯]` (District Fire Command Perg - Team FireOPS)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并部署了面向志愿消防员的消防运营平台 FireOPS，集成了智能预警、导航、共享作战图、资源负载概览和情境手册，采用边缘+云架构并实现双通道冗余，支持手机、手表和车载端接入。

**💡 创新点**

创新点在于：1）边缘与云协同的双通道（无线+服务器）冗余接收方案，消除单点故障；2）在同一平台实现实时地理作战图与区级资源负载视图，提升跨队伍协同；3）引入检索增强的 LLM 辅助决策，提供情境手册检索；4）以志愿消防员需求为核心的用户导向设计与迭代。

**🔧 技术方法**

使用技术包括：Go 语言微服务、REST Core API、RabbitMQ 消息中间件、Docker、syBOS（消防服务运营平台）集成、WAS paging 接收器、ELS‑Export 服务器推送、CAP/OGC SensorThings（计划中）、检索增强的大语言模型（预生产）、Apple Watch/Wear OS、Android Auto/CarPlay、Telegram/WhatsApp 连接桥、苹果/安卓推送服务。

**📊 数据集**

数据集：2024 年 9 月 Perg 区洪灾的日志提取（233 起事件、630 条调度记录，包含 257 条完整车辆到达时间数据），204 条应用内反馈条目，31 条邮件讨论，132 家梯队安装记录，以及 29 月到 2026 年 5 月的安装与用户增长数据。

**📈 对比分析**

通过特征矩阵与 DIVERA 24/7、blaulichtSMS、LifeRescue 进行对比，发现 FireOPS 在共享作战图、决策支持和双通道冗余方面领先；性能方面给出了调度到现场的经验分布（P25 = 4 min，P50 = 9 min，P90 = 51 min），但未做受控实验或定量效能评估，主要提供描述性基准和采用曲线。

**⚠️ 局限性**

局限性：仅在单一地区与单一高负载事件（洪灾）内评估；缺乏对照实验和定量效能指标；调度日志中缺失响应者级确认与现场时间戳，存在选择偏倚；双通道冗余效能未量化；系统仍使用专有接口，缺乏标准化互操作性；采用率与公开发布/奖项相关但未证明因果关系。

---

## 277. InterSim: Innovative Intersection Digital Twins for Human-in-the-Loop Driving Simulation

**arXiv ID:** 2611.05968 | [PDF](https://arxiv.org/pdf/2611.05968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 278. Sublinear Metric Distortion for Matching

**arXiv ID:** 2611.05777 | [PDF](https://arxiv.org/pdf/2611.05777v1)

**作者:** Moses Charikar `[一作]` (Stanford University), Prasanna Ramakrishnan `[通讯]` (Harvard University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

在一侧匹配市场中，研究仅给定代理人对物品的序列化偏好时，如何在未知的度量空间中实现低失真（metric distortion）匹配。提出了首个子线性失真上界（O(n^log_4 3)）的机制，并对随机序列化选举（Random Serial Dictatorship, RSD）与概率序列化（Probabilistic Serial, PS）的失真进行了上、下界分析；进一步给出对诚实机制（truthful mechanisms）的更强下界，并提出新的协调合并（Coordinated Merging）机制实现 O(n^α) 失真（α≈0.95084）。

**💡 创新点**

创新点包括：①首次实现子线性失真上界，打破此前 O(n) 的长期上限；②引入“cluster completion”算法，利用 RSD/PS 的前缀匹配与随机聚类完成剩余匹配；③精细化 RSD 与 PS 的失真分析，分别给出 O(n/√log n) 的下界与 Θ((10/9)^{n/2}) 的上/下界；④通过诚实机制的结构推导出 2^Ω(√log n) 的下界；⑤提出协调合并机制，改进 RepMatch 的合并策略，获得更小的 α。

**🔧 技术方法**

使用的技术主要包括：度量失真框架、随机序列化选举与概率序列化规则、聚类分区与平衡分区、流网络与最大流拆分、随机化裁剪（随机冲突图最大独立集）、多阶段随机过程（人工截止时间）、指数级并行级联构造、Poisson-Binomial 分布分析等。

**📊 数据集**

该工作完全基于理论分析，没有使用任何公开数据集；所有实例均为人工构造的度量空间与排名实例，用于证明上、下界。

**📈 对比分析**

相较于先前的 O(n) 上界，本文实现了 O(n^0.7925) 的子线性失真；对 RSD 的下界提升至 Ω(n/√log n)，证明先前的 O(n) 上界基本紧；对 PS 的上界证明为指数级别，与新机制的线性下界形成对比；对诚实机制的下界从 Ω(log n) 提升至 2^Ω(√log n)。

**⚠️ 局限性**

局限性：仍存在上界与下界之间的巨大鸿沟（子线性 vs 指数下界）；协调合并机制虽然改进了 α，但仍未达到与 RSD 相同的失真级别；在实践中未给出实验验证；对真实度量空间的适用性和鲁棒性分析有限。

---

## 279. The Split Protocol Decides the Result: Re-Evaluating Encrypted Traffic Classification

**arXiv ID:** 2611.06290 | [PDF](https://arxiv.org/pdf/2611.06290v1)

**作者:** Mayan Sharma `[一作]` (Vellore Institute of Technology), Devansh Behl `[通讯]` (Vellore Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对加密流量分类的评估协议，系统比较不同划分策略对模型性能的影响，并提出预训练诊断方法以选取合适的拆分轴；通过对两个公开数据集的严格实验，揭示划分协议决定最终评估结果，并对多种模型进行对比与负面结果报告。

**💡 创新点**

创新点在于（1）强调拆分协议对实验结果的决定性影响，并给出可验证的单标识符探测诊断；（2）在同一划分下对多种经典与深度模型进行统一评估，揭示树模型优于神经网络的现象；（3）系统性地报告多项先前预期的架构主张失败，提供负实验的透明度。

**🔧 技术方法**

主要技术包括：基于决策树的单标识符探测、多种流拆分协议（随机、会话、时间、服务器、客户端、时间服务器），深度双编码器（序列+上下文融合）与多任务对比学习，梯度反转对抗损失，余弦 k‑NN、原型匹配、逻辑回归与线性 SVM 评估头，开放集拒绝实验与抗干扰评估。

**📊 数据集**

使用了两大公开数据集：5G Traffic（单应用抓包，15类应用）和 CESNET‑QUIC22（国防骨干链路的 QUIC 流，18类服务），并提供了第三个 ISCXTor2016 的加载器以便未来扩展。

**📈 对比分析**

对同一拆分协议下的 13 种方法（4 深度、9 经典）进行宏 F1 与平衡准确率比较；结果显示：在 CESNET‑QUIC22 上经典 XGBoost 在 13 项流统计上比任何深度模型高约 0.23；在 5G Traffic 上深度模型仅略优于传统模型，树模型优于神经网络；同时揭示不同拆分方式导致宏 F1 下降 0.426（5G）和 0.192（CESNET）。

**⚠️ 局限性**

局限性包括：仅使用两种结构截然不同的数据集，无法推广到更广泛的网络环境；拆分协议的选取仍需基于数据集采集方式的先验假设；实验预算受限，部分模型未达收敛；部分负实验可能因超参数未充分调优。

---

## 280. Corralling AI Risks with Grassroots Contracts

**arXiv ID:** 2611.06374 | [PDF](https://arxiv.org/pdf/2611.06374v1)

**作者:** Ehud Shapiro `[一作]` `[通讯]` (London School of Economics and Weizmann Institute of Science), Ehud Shapiro (London School of Economics and Weizmann Institute of Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于“基层合约”的数字平台设计，以通过去中心化的智能手机点对点网络来缓解 AI 对平台控制、匿名 AI 代理和匿名深度伪造等三大风险。

**💡 创新点**

创新点在于将皮埃尔‑约瑟夫 普鲁东的社会契约思想转化为可被参与者签署、可编译成手机应用的数字合约，并通过可审计的数字署名和“责任链”机制实现 AI 代理的主体与开发者双重问责；同时引入“数字血统”追溯，让每条内容都携带完整的发送与转发链，消除匿名内容。

**🔧 技术方法**

主要技术包括：
• 以自然语言条款为输入的合约抽象层，编译为可执行的“语音行为”模式；
• 语音行为→多主体守望事务（volition‑guarded multiagent atomic transactions）→志愿守护的 GLP (vGLP) → Grassroots Logic Program → Dart/Flutter 应用；
• 使用公钥加密签名实现身份与责任链；
• 通过 AI 自动完成合约到代码的编译过程。

**📊 数据集**

本文未使用传统机器学习或大规模数据集，而是基于理论模型和小规模原型测试；实验数据仅为手机仿真环境中的功能验证，没有公开数据集。

**📈 对比分析**

由于缺乏与现有主流平台（如 Facebook、Twitter 等）的直接对比，本文仅在理论上证明基层合约可消除三类 AI 风险，并通过实验示例展示了手机应用的可行性；性能评估基于模拟，指出 AI 编译成本可控制在数月内完成，但未给出系统吞吐量或延迟指标。

**⚠️ 局限性**

局限性包括：
• 仍处于概念/原型阶段，尚未实现与现有全球平台竞争的规模与功能；
• 对大规模用户行为、隐私合规、法律适用等实际运营细节未做深入评估；
• AI 自动编译虽降低人力成本，但对模型错误、代码缺陷等问题的鲁棒性尚未充分验证；
• 只针对三大风险，其他 AI 相关危害（如误导性建议、滥用等）未纳入框架。

---

## 281. Separation Is a Riesz Energy: How Many Environments Does ICA Need?

**arXiv ID:** 2611.06228 | [PDF](https://arxiv.org/pdf/2611.06228v1)

**作者:** Yuda Bi `[一作]` (Georgia State University), Vince D. Calhoun `[通讯]` (Georgia State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了利用多环境下源方差变化（签名）通过 Riesz 能量评估独立成分分析（ICA）可识别性和样本复杂度的理论框架，并给出了最优采样率与误差上界；

**💡 创新点**

创新点在于把源在不同环境中的方差签名视为空间点，构建 Riesz 能量来量化分离难度，并证明了维度而非环境数决定所需样本；同时给出可实现的二次多项式时间方法（量子探针与网格搜索）来达到该上界；

**🔧 技术方法**

主要技术包括：第二阶统计量（协方差）下的联合对角化、Riesz 能量与 Fisher 信息的几何关系、最小化理论（Assouad 与信息矩阵）、投影与探测技术、随机矩阵与误差分解；

**📊 数据集**

使用合成高维高斯混合数据（Gaussian ICA）进行仿真，维度从 8 到 1000；并在 Human Connectome Project（HCP）任务 fMRI 数据上验证能量预测组件分离难度；

**📈 对比分析**

与传统单线性组合（AMUSE、随机多项式）和现有最大似然/Pham 算法比较，证明在 K≥4 并且样本量 n≈O(d log d) 时，联合对角化和二次探针达到 Riesz 速率，误差近似为 E_j/n；在低 K 或样本不足时误差显著增大；

**⚠️ 局限性**

局限包括：仅对高斯源、等大小环境、无时滞/空间相关性；最优样本量与理论上必要量之间存在上界下界差距；实际算法实现仍受计算成本限制；fMRI 结果仅为诊断性评估，未能精确预测误差大小；

---

## 282. Mitigating Spurious Invariance in Contrastive Learning: A Probabilistic Self-Supervised Learning Framework for Medical Image Analysis

**arXiv ID:** 2611.05734 | [PDF](https://arxiv.org/pdf/2611.05734v1)

**作者:** Wei Wei `[一作]` (Columbia University), Yading Yuan `[通讯]` (Columbia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一套概率自监督学习框架（ProbSSL），通过将医学影像的特征表示建模为概率分布，使用分布级对齐实现对比学习的动态对齐，并将预测和生成任务纳入统一的概率框架，以减轻对比学习中出现的伪不变性问题。

**💡 创新点**

创新点主要包括：
1) 采用概率分布（均值+方差）而非确定向量，提出分布级对齐损失（DCL）和样本级对齐损失（SCL），实现可调节的对齐强度；
2) 对该概率对齐方法给出理论误差分析，证明其估计误差低于传统确定对齐；
3) 将对比、预测、生成三类自监督任务统一到同一概率潜在空间，形成完整的ProbSSL框架；
4) 在多任务、跨模态下展示显著性能提升，体现概率建模在医学影像中的有效性。

**🔧 技术方法**

使用的关键技术包括：
- SwinUNETR/ SwinViT 作为主干网络；
- 高斯变分编码器，输出均值 μ 和方差 σ²；
- 分布级对齐损失 DCL（KL 对齐）和样本级对齐损失 SCL（对齐预测）；
- KL prior 正则项；
- 重新参数化技巧实现梯度传播；
- 预测任务（旋转预测）和生成任务（掩码重建）作为辅助目标；
- AdamW 优化器、线性 warm‑up + cosine decay 学习率；
- 采用 3D 随机裁剪、旋转、强度扰动、空间仿射等多种医学增强。

**📊 数据集**

预训练数据：BTCV（腹部 CT）、TCIA‑Covid19（胸部 CT）、LUNA16（肺部 CT），共 1,589 张无标签扫描。
下游 fine‑tune 数据：
- BTCV：多器官分割；
- MM‑WHS：心脏七子结构分割；
- CC‑CCII：COVID‑19 CT 分类；
- BraTS：脑肿瘤 MRI 分割。

**📈 对比分析**

与多类基准进行比较：
- 通用 SSL（SimCLR、MoCo、MAE、SimMIM）；
- 医学 SSL（ModelsGenesis、Rubik++、PCRLv1/2、VoCo、MiM、GL‑MAE、S2DC）；
- 有监督基线。
实验显示 ProbSSL 在 BTCV、MM‑WHS、CC‑CCII、BraTS 等任务上均达到或超过最佳对照方法，Dice/准确率分别提升约 3–5%。
在更强增强、少标签、跨模态等情境下也保持显著优势，证明其鲁棒性和标签效率。

**⚠️ 局限性**

局限性：
- 需要额外的参数和 FLOPs，训练时间略长；
- 采用对角高斯方差，忽略特征维度间协方差，可能无法充分捕捉解剖相关性；
- 对低质量影像、极端形变、长尾病例以及跨站点/扫描仪差异的鲁棒性仍有限；
- 预测和生成预训练任务的设计可能不适用于所有模态或器官，需进一步调优；
- 由于算力限制，未能在更大规模数据上验证，未来工作需扩展至更丰富的医学数据集。

---

## 283. SenTrojan: Revealing Analog Hardware Trojan Attacks on Sensors

**arXiv ID:** 2611.06103 | [PDF](https://arxiv.org/pdf/2611.06103v1)

**作者:** Kun Wang `[一作]` (Zhejiang University), Wenyuan Xu `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种针对传感器的模拟硬件木马框架（SenTrojan），能够在传感器正常工作时保持隐蔽，随后通过外部物理信号触发，实现永久性服务拒绝、静态或动态伪造等攻击。

**💡 创新点**

创新点在于：①首次系统性研究模拟硬件木马，突破传统仅依赖已有硬件缺陷的攻击方式；②设计了两种互补的触发机制——Logic‑Rule‑Driven（LRD）和 Physical‑Effect‑Driven（PED），实现低硬件占用与高触发可靠性的平衡；③通过高阻抗并行监测与原始信号分离，极大提升了木马的隐蔽性。

**🔧 技术方法**

技术要点包括：高阻抗并行频率/幅度检测电路、时间阈值积分电路、模拟开关/二极管/RC调制的触发与载荷模块；物理触发方式覆盖激光、超声、EMI 等；实验平台包括信号发生器、功率放大器和多种传感器板；评价指标涵盖相似度、误触率、成功触发率以及时间漂移、噪声底、响应时间与功耗的硬件开销。

**📊 数据集**

使用的“数据集”主要是对14种不同传感器（光敏、麦克风、红外、压力、倾斜、温度等）在真实环境下的实验记录，未使用公开的标准数据集；实验数据来自多次触发实验、长时间测量和功能验证。

**📈 对比分析**

通过与传统物理注入攻击（如激光注入、超声干扰、EMI 注入）比较，SenTrojan 的攻击范围可提升至 10 倍（如麦克风从 0.5 m 提升至 5.1 m），成功触发率均超过 95%，相似度始终保持 97%以上，误触率为 0%；硬件开销（漂移、噪声、响应时间、功耗）均低于 5% 或 10% 的阈值。

**⚠️ 局限性**

局限性包括：①木马必须在传感器生产或装配阶段植入，后期改造成本较高；②需要能对传感器施加特定物理信号的环境，且不同传感器可能需要额外的附加传感器或参数调校；③高精度的功能测试或主动检测手段（如EM泄漏、阻抗监测）仍可能揭露木马；④在极端噪声或电源波动环境下，触发可靠性可能下降。

---

## 284. Multimodal Fusion Methods for Whole-Slide Images and Clinical Data in Inflammatory Bowel Disease

**arXiv ID:** 2611.05579 | [PDF](https://arxiv.org/pdf/2611.05579v1)

**作者:** Ankana Banerjee `[一作]` (University of Stavanger), Kjersti Engan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一个模块化框架，对15种不同的多模态融合方法在炎症性肠病（IBD）基线或随访病理切片与临床数据上进行统一评估，并预测STRIDE-II的三种缓解终点。

**💡 创新点**

创新点在于：①将所有融合模块共享相同的输出维度和预测头，消除网络架构差异的干扰；②在相同的训练和验证条件下对多种融合策略进行全面对比；③提出并量化“图像依赖度”，衡量模型对切片信息的利用程度。

**🔧 技术方法**

采用Virchow2预训练模型提取224×224切片特征，使用门控注意力池化生成图像嵌入；临床数据通过多分支MLP编码为256维；15种融合方法涵盖拼接、FiLM、双线性、注意力、决策级融合等；最终使用焦点损失训练分类头。

**📊 数据集**

使用SUSI‑IBD前瞻性单中心队列（约250–260例新诊IBD患者），每例在诊断、3个月、11个月采样；切片使用H&E染色、NHI分级；临床变量包括11个连续、5个二进制、3个分类特征。

**📈 对比分析**

在六种预测组合（三种缓解终点×两种输入时点）中，最佳融合方法的宏F1达0.57–0.75；在四种场景下优于单模态模型，且最佳方法相对于中位方法提升0.06–0.15；方法排名随终点和输入时点显著变化。

**⚠️ 局限性**

局限性包括：①队列规模有限、单中心数据；②未进行多次训练重复，无法评估训练波动；③仅使用一次随访输入，未考虑多时点融合；④缺乏外部验证，鲁棒性未知。

---

## 285. LegalEvolver: Agentic Legal Judgment Prediction via Rubric-Guided Self-Evolution

**arXiv ID:** 2611.06381 | [PDF](https://arxiv.org/pdf/2611.06381v1)

**作者:** Xinyu Yang `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LegalEvolver框架，用于通过优化检索查询提升法律判决预测（LJP）模型的推理质量。

**💡 创新点**

创新点在于：①利用同案轨迹对比自动诱导法律查询 Rubrics，并将其作为中间查询的奖励；②实现查询 Rubrics 与策略的协同进化，随着模型性能提升不断更新 Rubrics；③结合高温采样、启发式查询重写与拒绝采样实现多样化轨迹生成。

**🔧 技术方法**

使用技术包括：大语言模型（LLM）驱动的 Agentic Retrieval（ReAct）、SFT + GRPO 强化学习、Rubric‑based reward、查询重写策略、Rubric‑policy 共演化算法。

**📊 数据集**

使用公开法律数据集 CAIL2018（单被告）和 CMDL（多被告）进行训练与评测。

**📈 对比分析**

与分类方法、直接推理、Agentic Retrieval Augmented Reasoning、以及后训练方法进行对比；在 Accuracy 与 Macro‑F1 指标上均获得最优或接近最优成绩，尤其在法律条文、指控与刑期预测上表现突出。

**⚠️ 局限性**

局限性包括：仍依赖大语言模型，可能继承数据与模型偏见；Rubrics 生成与评估依赖人工审核或 LLM 可信度；模型未经过司法监管验证，无法直接用于法律实务。

---

## 286. Selective Prediction Intervals: Jointly Learning to Predict, Bound, and Abstain

**arXiv ID:** 2611.06158 | [PDF](https://arxiv.org/pdf/2611.06158v1)

**作者:** Lidor Rubi `[一作]` (Ben-Gurion University of the Negev), Gilad Katz `[通讯]` (Ben-Gurion University of the Negev)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可同时输出预测区间、区间内点估计以及可拒绝决策的回归模型。

**💡 创新点**

创新点在于将区间估计、值估计与选择性拒绝三者在单一损失下联合训练，并使用双门结构与 Fréchet 惩罚实现接受率约束，显著优于仅靠区间宽度的拒绝策略。

**🔧 技术方法**

采用 IPIV 的区间与值头、SelectiveNet 的软门、双门结构、Fréchet 约束以及严格的区间得分（Winkler 评分）等技术。

**📊 数据集**

在 20 个回归基准上评估，包括 UCI 与 OpenML 传统数据集、RSNA 骨龄图像任务及 IMDB‑WIKI 年龄估计数据集。

**📈 对比分析**

与宽度阈值、深度集成不一致度、SelectiveNet 等基线比较，所提出模型在所有数据集上都在选择性 RMSE 与区间得分上取得更低误差，且在保持统一接受率下超过宽度阈值和集成差异的性能。

**⚠️ 局限性**

局限在于缺乏有限样本覆盖率保证，拒绝决策仅基于难度而非下游效用，且需要额外的门控训练与超参数调优。

---

## 287. Beyond Identity Creation: Bilinear Sybil Cost in Identities and Time

**arXiv ID:** 2611.06268 | [PDF](https://arxiv.org/pdf/2611.06268v1)

**作者:** Homayoun Maleki `[一作]` (University of Deusto), Igor Santos-Grueiro `[通讯]` (International University of La Rioja)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了“Bilinear Sybil Cost”模型，给出保持 s 个身份在 T 个周期内活跃所需的最小资源成本，并给出了四条充分条件（身份绑定、时效性、共享截止时间、顺序非批处理）保证该成本，并实现了一个无需支付的身份绑定 VDF 方案。

**💡 创新点**

创新点在于：①从成本模型角度将 Sybil 防御的目标从身份创建转移到身份维护；②定义了 bilinear 成本并证明四条条件的独立性；③利用身份绑定的 VDF 与公开信标和时间证明，既满足所有四条条件，又在无支付环境下实现了可行的、可量化的安全保证。

**🔧 技术方法**

核心技术包括：可验证延迟函数（VDF）在类群上实现（Wesolowski 构造），欧氏签名（Ed25519）绑定身份与窗口，随机数信标（如 drand）保证时效性，公共可追加日志记录时间证明；以及对 VDF 的无批处理和顺序性假设。

**📊 数据集**

实验数据来自两台机器：一台 Intel Core Ultra 9 275HX（24 核）和一台 2.0 GHz 虚拟机；使用 1024 位判别式的 VDF 评估和签名；对心跳证明与 VDF 证明的数量、时间、批处理行为做了系统测量。

**📈 对比分析**

比较方法：在 2 秒窗口下测量每核可维持身份数，心跳证明可达约 80k 个身份，VDF 证明每核只能维持 1 个身份；在多核（8/24）场景下攻击者保持身份的比例与理论 <=min(1, p/s)> 一致；VDF 的总成本为 sT/A，A≈2–8（CPU）或≈28（假设 ASIC），明显优于哈希谜题的 A≈10^3–10^7。性能上，VDF 证明耗时约 1.5–0.5 秒，验证仅 15–33 毫秒。

**⚠️ 局限性**

限制：①对速度比 A 的估计基于已公布的 ASIC 设计，非实际测量；②VDF 的无批处理性仅在通用群模型下证明，对非通用算法依赖实验；③需要可靠的公开信标和日志，若信标落后或日志被篡改会影响时效性；④对移动设备等低功耗终端的评估缺失；⑤虽然成本低（≈1¢/身份/月），但需要在每个周期消耗资源，且无法限制身份注册数量。

---

## 288. Can Large Language Models Review the Scientific Literature? A Test Against a 102-Article Benchmark

**arXiv ID:** 2611.05459 | [PDF](https://arxiv.org/pdf/2611.05459v1)

**作者:** Abdullah Al Kalbani `[一作]` (University of Manchester), Catherine McGuire `[通讯]` (University of Manchester)

**通讯引用:** 117 | [OpenAlex ID](https://openalex.org/A5074838408)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文使用六种LLM工具对文献综述的四个核心步骤（检索、解读、综合、提出未来工作）进行系统评估，基准为已知的102篇西班牙羽流相关论文与其手工综述数据库。

**💡 创新点**

创新点在于首次在完整已知的基准文献集上，对LLM在四个步骤的整体表现进行量化与质化比较，揭示检索受限、判别误差、洞察缺失等关键问题。

**🔧 技术方法**

研究采用公开对话、编码代理和检索驱动的LLM（ChatGPT Deep Research、Gemini Deep Research、Claude web、Codex、Claude Code、NotebookLM），并配合统一的prompt进行实验。

**📊 数据集**

所用数据集为102篇关于西班牙羽流的文献及其已构建的综述数据库，涵盖不同出版社、出版年份与术语使用方式。

**📈 对比分析**

比较方法包括检索命中率、文章分类精度、引用准确率以及两位专家对综合文本的五维评分；结果显示检索仅约50%文章被发现，解读分类精度低至35%，综合准确率高但洞察力差，未来工作识别与基准相近但范围泛化。

**⚠️ 局限性**

局限性包括仅评估单一主题、单次实验、登录状态不统一、提示词依赖、仅使用通用LLM、专家评分主观性等因素。

---

## 289. NN-Based FeFET Compact Modeling for SPICE Simulation

**arXiv ID:** 2611.06127 | [PDF](https://arxiv.org/pdf/2611.06127v1)

**作者:** Rodion Novkin `[一作]` (Technical University of Munich), Hussam Amrouch `[通讯]` (Technical University of Munich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一种基于神经网络的 FeFET 区分模型，该模型直接使用实验测量数据进行训练，无需提取导数。

**💡 创新点**

创新点在于引入谱归一化（Spectral Normalization）作为无导数训练的正则化手段，自动平滑导数并过滤测量噪声，解决了传统需要高阶导数信息的瓶颈。

**🔧 技术方法**

采用三层全连接网络（每层16个神经元）配合 tanh 激活和谱归一化，结合自定义的极化动力学公式，实现 SPICE 兼容的 Verilog‑A 模型。

**📊 数据集**

使用在 28 nm FeFET 工艺上采集的电流-电压-极化三维测量数据（共11个极化幅值），包括从 0 V 到 4.5 V 的完整极化范围。

**📈 对比分析**

通过在商业 SPICE 仿真器中对 2×2 到 512×512 大小的存储阵列进行写/读仿真，比较了模型收敛性、模拟时间与传统 BSIM/数值求解器的结果。结果表明：模型能够保持平滑导数、逼近实验曲线，且在 512 阵列的 15 分钟仿真中仍保持收敛，且相较于开启极化逻辑时可节省约 17 % 的仿真时间。

**⚠️ 局限性**

局限性包括：对测量漂移和异构噪声的校正仍需人工调整；模型依赖于实验数据的完整性，缺乏对极化速率未校准或异常极化状态的鲁棒性；在更高频或更大尺度的动态仿真中可能需要进一步优化谱归一化参数。

---

## 290. LACS: Distribution-Sensitive Label-Agnostic Confidence Selection for Dynamic Data Pruning

**arXiv ID:** 2611.05616 | [PDF](https://arxiv.org/pdf/2611.05616v1)

**作者:** Heejin Choi `[一作]` `[通讯]` (Yonsei University), Heejin Choi (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种标签无关的基于最大置信度指数平均的训练时实例子集选择方法LACS，并在多种学习任务中验证其对训练工作量和预测性能的影响。

**💡 创新点**

创新点在于用无标签的指数平均置信度参考实现自适应阈值和保留计数，并给出无剪枝、非空支持与质量边界的解析理论。

**🔧 技术方法**

使用了指数平均置信度阈值、梯度裁剪、连续权重、强化学习的弱学习器、神经网络的稀疏后向传播以及SVM、树和二次学习器等技术。

**📊 数据集**

实验覆盖CIFAR-10/100图像分类、Transformer语言模型、长尾分类、半监督学习以及14个表格数据集的弱学习器组合。

**📈 对比分析**

与全数据训练、随机子集、InfoBatch、Selective-Backprop等方法对比，LACS在大部分任务中减少了30–35%的后向工作量，预测精度保持或略有提升，但在神经网络的两遍实现下未实现显著的wall‑clock加速。

**⚠️ 局限性**

局限性包括：神经网络实验仅在三种随机种子下验证且两遍实现导致更慢；对某些弱学习器的ρ≥1设置不满足理论保证；缺乏统一的系统优化以充分展示稀疏计算的速度优势。

---

## 291. Discourse Analysis of Legislative Bill Discussions

**arXiv ID:** 2611.05381 | [PDF](https://arxiv.org/pdf/2611.05381v1)

**作者:** Isaac Rudnick `[一作]` (California Polytechnic State University), Foaad Khosmood `[通讯]` (California Polytechnic State University)

**通讯引用:** 250 | [OpenAlex ID](https://openalex.org/A5059880273)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套基于语料清洗、实体识别、CFG结构验证以及分类器的流程，用于将美国州议会委员会听证会中的发言转录按议题阶段划分；

**💡 创新点**

创新点在于将英国议会语义层级与命名实体信息结合，利用CFG生成语义结构树对分类结果进行约束，并引入基于词袋和TF‑IDF的手工特征进行多阶段语篇划分；

**🔧 技术方法**

使用了手工特征提取、TF‑IDF向量化、逻辑回归与多层感知器分类器、CYK解析器实现的CFG约束，以及后期的平滑处理；

**📊 数据集**

使用了“Digital Democracy”项目的加州州参议院与众议院委员会听证会转录数据，约2619条已标注发言；

**📈 对比分析**

与传统的多层感知器相比，逻辑回归分类器在训练集上取得宏观F1为52.8%，精确度51.4%，召回率57.2%，整体表现略好；

**⚠️ 局限性**

局限性包括：语法约束过于严格导致大部分发言无法生成解析树、数据集规模有限、缺乏序列模型导致相邻发言的误分类、以及对关键短语的手工提取仍然存在主观性。

---

## 292. Fano's Inequality and Channel Capacity in the Individual-Sequence Regime

**arXiv ID:** 2611.06182 | [PDF](https://arxiv.org/pdf/2611.06182v1)

**作者:** Neri Merhav `[一作]` `[通讯]` (Technion Israel Institute of Technology), Neri Merhav (Technion Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了个体序列（individual‑sequence）范式下的 Fano 不等式，并以此为基础给出了通用通道编码的逆定理（converse）与正定理（achievability），从而推导出 Lempel‑Ziv（LZ）复杂度定义的极限通道容量；

**💡 创新点**

创新点在于①首次构造个体序列 Fano 不等式，将条件熵替换为条件 LZ 复杂度；②基于此得到完全匹配的逆定理与正定理，定义了最小‑最大化形式的 LZ 通道容量；③提出了符号逐个可逆（symbol‑wise invertible）通道与噪声集对称闭包条件下的容量实现方案；

**🔧 技术方法**

核心技术包括：LZ78 及其条件版本的增量分解、块平均 LZ 复杂度、上限（sandwich）约束、随机编码与类型方法、有限状态机（block‑FSM）与 LZ 条件解码的结合；

**📊 数据集**

无实验数据集，论文纯理论推导；

**📈 对比分析**

与之前的加法通道、AVC 相关工作进行对比；在可逆通道、噪声集满足块置换闭包时，得到的容量与 Federr 等人的结果一致；相比之前的开放式逆定理，本文提供了完整的容量公式；

**⚠️ 局限性**

限制包括：只适用于符号逐个可逆通道，噪声集需满足块置换闭包；缺乏关于反馈是否能提升 LZ 容量的答案；对一般非可逆或非闭包噪声集的可实现性仍未解决。

---

## 293. How Finely Can Vision-Language Models See Biology? Bio-VQA: A Benchmark for Taxonomic, Morphological, Spatial and Individual Reasoning

**arXiv ID:** 2611.05559 | [PDF](https://arxiv.org/pdf/2611.05559v1)

**作者:** S M Rayeed `[一作]` (Rensselaer Polytechnic Institute), Yao Ma `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了最大规模的生物视觉问答（VQA）基准，包含1.86M个问题、217K张图片，涵盖分类推理、形态特征定位、定量空间推理和个体重识别四个任务模块。

**💡 创新点**

创新点在于：①提出了面向生物领域的分层评分与归一化评估指标；②引入基线下限与专用专家上限，明确展示通用视觉语言模型（VLM）的性能差距；③通过十个物种群组和多任务结构，首次系统评估VLM在生物视觉推理中的跨域迁移能力。

**🔧 技术方法**

使用了18款公开权重的VLM（如Qwen2.5/3‑VL、InternVL3、Gemma‑3/4、Pixtral、Llama‑3.2‑Vision 等），采用贪婪解码、统一问答模板，并对输出进行解析与评分；评估指标包含精确匹配、层级相似度、机会校正的分等级得分、计数误差、IOU、信号检测理论等。

**📊 数据集**

数据集来源于11个公开生物学数据集，整合至TreeOfLife‑200M、CUB‑200‑2011 等，涵盖十个主要物种组（包括节肢动物、藻类、植物等）及其多样化标签。

**📈 对比分析**

通过与基线下限（图像盲预测）和专家上限（BioCLIP‑2、MegaDescriptor等）比较，发现VLM在粗粒度分类上表现相当，但在细粒度计数、科学命名、形态特征测量和个体重识别等任务上仅取得不到50%或更低的准确率，且模型规模提升对细粒度任务的帮助有限。

**⚠️ 局限性**

局限性包括：仅评估公开权重模型，未考察专有模型；解码仅采用贪婪方式，未探索链式推理；测试图像可能已被VLM预训练集包含；缺乏人类基准；基线规则可能低估模型能力；部分结果受数据集分布与采样影响。

---

## 294. CCDWIKI: Decoupling Case Formulation from Response Generation in Longitudinal CBT Dialogue

**arXiv ID:** 2611.05383 | [PDF](https://arxiv.org/pdf/2611.05383v1)

**作者:** Mingwen Zhang `[一作]` (Lanzhou University), Minqiang Yang `[通讯]` (Lanzhou University)

**通讯引用:** 1234 | [OpenAlex ID](https://openalex.org/A5079527887)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 CCDWiki，一种插件式建议器，用来维护CBT对话中的证据驱动案例概念化并提供治疗指导；

**💡 创新点**

创新点在于将CBT案例概念化拆解为可追溯、可修改的假设条目，并通过证据评判维持长期连贯性；

**🔧 技术方法**

采用证据评估（EALF）和基于概念化的干预规划（FCIP）两阶段推理模块，输出给定LLM的指导；

**📊 数据集**

使用 PsychEval 中文 CBT 子集（148 案例、1,112 会话、50,018 交互）进行实验；

**📈 对比分析**

与多种基础模型（DeepSeek、GPT-5.4、GLM4-9B、InternLM3-8B 等）对比，CCDWiki 在 CTRS、PANAS 等指标上均有提升，尤其在小型模型上显著；

**⚠️ 局限性**

局限性包括对单一语言（中文）数据的依赖、缺乏真实临床对照、并且对部分模型在交互质量方面存在权衡。

---

## 295. Scale Anchoring via Implicit Semantics and Sequence-Consistent Loop Closure for VGGT-Based SLAM

**arXiv ID:** 2611.06308 | [PDF](https://arxiv.org/pdf/2611.06308v1)

**作者:** Ziyi Jin `[一作]` (Anyverse Dynamics), Jie Xu `[通讯]` (Anyverse Dynamics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在VGGT‑SLAM 2.0基础上引入固定根尺度锚点和短序列辅助直接环路连接，提升稠密单目重建的尺度一致性与闭环精度。

**💡 创新点**

创新点在于：①使用SALAD检索与VGGT联合推断，仅提供尺度信息而不强制相机姿态；②采用短序列一致性检验来验证并直接插入Sim(3)闭环因子，避免传统辅助子图的冗余与误闭环。

**🔧 技术方法**

核心技术包括：VGGT联合几何与语义推断、SALAD基于DINOv2的图像检索、Sim(3)因子图优化、GTSAM分阶段尺度与姿态优化。

**📊 数据集**

使用17条序列评估，涵盖TUM RGB‑D（9条）、OpenLORIS（3条）、Clio（3条）、LaMAR、ADVIO及INS NTU Indoor等多种室内外环境。

**📈 对比分析**

与VGGT‑SLAM 2.0基线对比，完整系统在所有序列上平均ATE从0.1018 m降至0.0451 m（提升55.6%），锚点模块在无闭环序列中均改善精度，直接闭环连接相较于辅助子图显著降低ATE，实验显示显著性能提升。

**⚠️ 局限性**

局限性包括：仅在关键帧驱动下更新，无法提供连续实时定位；对天空或纹理稀疏场景尺度推断不可靠；未测量完整在线时延与实时性能，且在重复走廊等场景下闭环接受率低。

---

## 296. How Unbalanced Must a Latin Rectangle Be?

**arXiv ID:** 2611.06003 | [PDF](https://arxiv.org/pdf/2611.06003v1)

**作者:** Zhengyu Li `[一作]` (Georgia Institute of Technology), Vijay Ganesh `[通讯]` (Georgia Institute of Technology)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了 k×n 拉丁矩形的最小不平衡问题，提出了一种新的行排列模拟退火搜索，得到多组改进的上界，并证明了 3×9 拉丁矩形的最优不平衡值为 56，且给出了 Lean 4 形式化证明与证书。

**💡 创新点**

创新点在于将行排列视为可交换的搜索状态，采用列冲突惩罚并自适应权重，设计了行群不变量判定矩形非群表结构，以及改进的分支定界框架证明 3×9 的最优性。

**🔧 技术方法**

所用技术包括模拟退火、局部搜索、行群不变量、分支定界、随机矩形的概率估计、极限 permuton 分析，并在 Lean 4 中完成形式化验证与证书生成。

**📊 数据集**

使用了先前文献中的记录表格作为基准数据，并在 GitHub 上公开了 62 个矩形的数据集及其 Lean 证明文件，涵盖了 8≤n≤13、2≤k≤n-1 的实例以及 2×n 的 14≤n≤64 的实例。

**📈 对比分析**

与 SAT、混合整数规划和 LLM 流程生成的基准相比，单次 2 分钟退火在 8≤n≤13 与 2≤k≤n-1 的实例中匹配或改进约 25% 的上界；在 2×n 实例中，退火在 15≤n≤19 上优于 6 小时 MIP 上限。

**⚠️ 局限性**

限制在于改进的上界来源于有限次数的随机搜索，存在波动；3×9 最优性仅通过两份未形式化的程序得出；对更大实例缺乏完整证明，且 lower bound 常数远低于实际观测值。

---

## 297. PEARL: Prediction Error-Adaptive Regularized Learning for Selective Forgetting under Partial Preference Drift

**arXiv ID:** 2611.05842 | [PDF](https://arxiv.org/pdf/2611.05842v1)

**作者:** Rashmi Gangadharaiah `[一作]` (AWS), Balakrishnan Narayanaswamy `[通讯]` (AWS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种具备误差自适应探索与双向Gram更新机制的神经上下文多臂赌博机，用以在用户偏好部分非平稳的情境下实现选择器的在线自适应

**💡 创新点**

创新点在于：①通过在误差较大的特征方向注入不确定性实现选择性遗忘，②结合每位用户的误差跟踪器实现误差自适应探索，使模型在偏好漂移时能快速恢复；同时提供了针对部分非平稳性的理论回报上界

**🔧 技术方法**

采用神经上下文多臂赌博机框架，使用最后一层特征的Gram矩阵做不确定性估计；通过Sherman‑Morrison递推和谱上限实现双向更新；利用指数移动平均误差跟踪器调节UCB系数；奖励模型为一层隐藏层的MLP，支持在线MSE更新

**📊 数据集**

主要实验基于PRISM数据集（1500用户、21个LLM、8维偏好特征、1–100标度的打分），并在10用户、10臂的仿真环境中验证机制；数据被分为训练、在线更新和测试阶段

**📈 对比分析**

与Greedy、NeuralUCB（固定/自适应）、SW‑UCB、折扣式NeuralUCB、NeuralTS、URM/MRM/VRF等基线比较。结果显示在中等偏好漂移（flip=0.5）下，所提方法在第10轮平均奖励为0.641，比Greedy高4.0分、比NeuralUCB高2.5分，优于滑动窗口和折扣方法3–10分，且在偏好稳定时与Greedy表现相当

**⚠️ 局限性**

局限性包括：仅在观察到单一动作奖励的场景下有效；误差跟踪器对噪声敏感，需要合适的β；双向更新与谱上限对计算资源和数值稳定性有一定要求；实验主要聚焦于PRISM，未在低候选多样性或不同任务（如对话生成）下进行广泛验证；需要进一步研究在完全持续漂移或离散化奖励场景下的适应性

---

## 298. The Effectiveness of Sliding Mode Control for Robust Vision Stabilization under the Plant-Binding Condition

**arXiv ID:** 2611.06028 | [PDF](https://arxiv.org/pdf/2611.06028v1)

**作者:** Sooyeun Yang `[一作]` (SUNY Korea), Seung Bok choi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在植物绑定（即扰动速度逼近或超过执行器转速限制）下边界层滑模控制（SMC）在视觉稳像中的优势；

**💡 创新点**

提出了“植物绑定”这一SMC激活条件，并证明其在两种视觉控制任务中统一适用；

**🔧 技术方法**

采用边界层滑模控制、低通滤波（LPF）、PID、L^1等控制器，并在虚拟单自由度模型和PTZ伺服系统上实现；

**📊 数据集**

使用12段智能手机抓拍的手持平移视频（1080p, 60fps）和16个合成PTZ设定点网格数据；

**📈 对比分析**

通过留一交叉验证、配对引导自举、并计算95%置信区间和Cohen's d，结果显示：在植物绑定区SMC在最坏情况下优于PID，且在平移任务中与LPF表现相当；

**⚠️ 局限性**

局限包括仅在仿真环境下验证，未涉及实际硬件编码量化、背隙、加速度限制，且ArUco检测间隙对性能影响尚未完整评估。

---

## 299. Ranking and Unranking Permutations in Heap's Order

**arXiv ID:** 2611.05397 | [PDF](https://arxiv.org/pdf/2611.05397v1)

**作者:** Yusheng Hu `[一作]` `[通讯]` (Independent Researcher), Yusheng Hu (Independent Researcher)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文提出了针对 Heap 排列生成顺序的 Rank/Unrank 对算法，利用 Heap 算法的块结构和交换运算实现对 Heap 顺序的直接映射；

**💡 创新点**

其创新点在于将 Heap 产生的块内部遍历和块边界交换分别抽象为尾部保持算子和顶层交换算子，得到一个闭式递推关系，并基于此构造无预计算表、O(n) 空间的双向 Rank/Unrank 算法；

**🔧 技术方法**

采用的技术主要包括排列与位置映射的符号系统、低阶优先的混元基数码、块内部尾部保持算子、顶层交换算子以及强归纳证明来验证闭式递推与映射双射性；

**📊 数据集**

该工作不使用任何外部数据集，而是通过数学证明和对 n≤10 的完整枚举验证来检验算法的正确性；

**📈 对比分析**

与现有按词典序、Fisher‑Yates 等生成顺序的 Rank/Unrank 方法相比，本文的实现时间复杂度为 O(n³)，空间为 O(n)，虽然速度慢于线性时间方案，但无需 factorial‑scale 预处理，直接支持 Heap 顺序；

**⚠️ 局限性**

局限性主要在于时间复杂度较高（O(n³)），以及缺乏对更高效实现（如 O(n²) 或 O(n log n)）的进一步优化研究。

---

## 300. Reconciling Privacy with Provenance for Language Model Watermarks

**arXiv ID:** 2611.06100 | [PDF](https://arxiv.org/pdf/2611.06100v1)

**作者:** Elise L. Han `[一作]` (Stanford University), Tatsunori Hashimoto `[通讯]` (Stanford University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种局部差分隐私水印（RGS）方案，能在不泄露单文本信息的前提下准确估计大规模文本中模型生成内容的比例。

**💡 创新点**

将水印目标从个体追踪转向聚合频率估计；设计序列级随机化响应水印，并给出理论偏差-方差分析与偏差控制策略；实现了高效的拒绝采样算法。

**🔧 技术方法**

局部差分隐私、随机化响应估计、序列级红绿序列水印、基于BLAKE2b的密钥哈希、拒绝采样、统计偏差-方差分析。

**📊 数据集**

C4网络文本数据集、OUTFOX议论文提示集、LiveCodeBench编程任务；使用Qwen3系列和Llama‑3.1‑8B‑Instruct等现代大模型生成文本。

**📈 对比分析**

通过在不同模型、不同隐私预算、不同样本量、不同真实比例下的Monte Carlo实验，评估MSE、平均绝对偏差、MAPE等指标。结果显示：ε=1时MSE≈0.0019、平均绝对偏差≈0.0038；ε=2、100文本样本时MSE≈0.0113、平均绝对偏差≈0.004；在大多数设置下均能保持低偏差与可接受误差。

**⚠️ 局限性**

对序列级哈希的鲁棒性不足，易被细微编辑破坏；对token级水印在局部DP下的隐私泄漏未做完整分析；多次查询或时间序列下的隐私保证有限；在低熵、低多样性场景下仍可能出现偏差。

---

## 301. Weaver: Weaving Preference Evidence through Incremental Reasoning for Personalized Retrieval-Augmented Generation

**arXiv ID:** 2611.05900 | [PDF](https://arxiv.org/pdf/2611.05900v1)

**作者:** Linyao Zheng `[一作]` (Xi'an Jiaotong University), Xuhang Shi `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了名为Weaver的个性化检索增强生成框架，能够在多轮对话历史中增量检索并合成用户偏好证据，生成符合当前需求的个性化回答。

**💡 创新点**

创新点在于：①双层检索与推理图（上层意图依赖图 + 下层记忆检索轨迹图）实现需求驱动的增量检索；②基于记忆条件的增量检索利用检索到的片段作为定位线索，引导后续检索；③范围感知组合推理能够在时间与情境范围内联合解释碎片化证据，区分补充、替代与冲突信息，动态更新结论与未解缺口。

**🔧 技术方法**

技术方法包括：大语言模型（LLM）用于子问题分解、依赖表达式、检索查询生成与证据解释；向量检索（Qwen3-Embedding-8B）与TopK检索；双层图结构维护检索轨迹与信息缺口；递归推理循环实现增量检索与推理的闭环。

**📊 数据集**

使用的基准数据集：PersonaMem（32K历史版本）和PrefEval；额外在MuSiQue与WebWalkerQA上进行评估。

**📈 对比分析**

与六个主流基线（MemCoE、BMAM、MemoryBank、EMG-RAG、RF-Mem、PGR）对比，Weaver在PersonaMem和PrefEval上分别取得86.42%和91.10%的ACC，分别比最强基线提升约16.0%和7.6%；在其它任务上也保持领先或相近的性能。

**⚠️ 局限性**

局限性包括：仍需大量Token和计算资源（与MemCoE相比约44% token 成本下降，但仍较高）；依赖LLM的推理与解释质量，若LLM生成错误会影响检索路径；在多场景大规模历史中的检索效率与延迟仍有提升空间。

---

## 302. Pauli Gadget Synthesis for Gatesets with Arbitrary Even Arity Clifford Gates

**arXiv ID:** 2611.06139 | [PDF](https://arxiv.org/pdf/2611.06139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 303. CARTBench: Rubric-Based Evaluation of Shopping Advice for Complex and Ambiguous Requests with Timelines

**arXiv ID:** 2611.05862 | [PDF](https://arxiv.org/pdf/2611.05862v1)

**作者:** Yuqin Dai `[一作]` (Tsinghua University), Yongliang Shen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了CARTBench，评估大型语言模型在复杂、含有时间限制的多轮购物需求中的表现。

**💡 创新点**

创新点包括：①基于真实商业对话的时间戳化数据集；②引入理想回复原则（Ideal Response Principles）统一制定案例专属评判标准；③设计原则引导的CART Prompting策略；④将核心、卓越和安全三层评判结合到一套可解释的评价体系。

**🔧 技术方法**

采用大型语言模型（如GPT-6.1-Sol、Claude-Opus-5、Qwen3.5等）进行生成；使用基于证据的评判流程（Evidence‑Grounded Rubric Judging）和专家校准的判分；同时利用共享原则在提示中进行结构化推荐。

**📊 数据集**

使用169个来自17个零售领域的真实商业案例，配备1,603条产品记录，并制定了1,328条评判标准，涵盖核心、卓越与安全三个维度。

**📈 对比分析**

对10个模型（3个商业API+7个开源大模型）进行评估，核心指标中3个模型达90%+，但卓越指标未有模型超过45%，显示理解与执行之间存在明显差距；安全得分整体较低。

**⚠️ 局限性**

局限性包括：样本仅覆盖常见购物场景，未包含高度专业化需求；缺乏长周期购买、退换货等后续反馈；评估仅基于对话与产品记录，未考虑真实用户体验与后续交互。

---

## 304. Makespan Bounds for Synchronous Pebble Motion on Graphs

**arXiv ID:** 2611.05839 | [PDF](https://arxiv.org/pdf/2611.05839v1)

**作者:** Lucy McManamon `[一作]` (Ohio University), Jingjin Yu `[通讯]` (Rutgers University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在图上标记的卵石运动的最大最小完成时间，提出了一种允许同步卵石移动的模型。

**💡 创新点**

创新点在于允许多个卵石同时移动，并且提供了不同结构和卵石密度下的完成时间界限。

**🔧 技术方法**

使用了图论中的排列路由、循环覆盖和路径转移等技术。

**📊 数据集**

使用了随机立方图、简单的3连通平面立方图等数据集。

**📈 对比分析**

通过与经典卵石运动和匹配路由的比较，证明了在全占用情况下的上界为O(log^9 n)，在固定卵石密度下的上界为O_ρ(log^3 n)，并且在每种情况下都有Ω(log n)的下界。

**⚠️ 局限性**

限制在于计算最优调度的难度，即使对于受限的图族也是如此。

---

## 305. Learning Implicit Solves for Dynamic Simulations

**arXiv ID:** 2611.05439 | [PDF](https://arxiv.org/pdf/2611.05439v1)

**作者:** Chawit Kaewnuratchadasorn `[一作]` (University of Hong Kong), Xin Nie `[通讯]` (Tsinghua University)

**通讯引用:** 4001 | [OpenAlex ID](https://openalex.org/A5073407416)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Implicit Neural Operator Solver (INOS)，通过在静态平衡数据上预训练神经算子来学习隐式求解，并在 Newmark‑β 隐式积分中重复使用该算子，实现从静态知识到动态预测的零步动态仿真。

**💡 创新点**

将静态平衡学习与隐式时间积分耦合，得到只需静态数据即可完成动态模拟的框架；利用预训练算子替代在线 Newton 迭代；可零步迁移到不同载荷、初始条件、时间步长，并保持极高的时间步长可行性；实现跨网格、跨尺寸、跨非线性系统的通用性。

**🔧 技术方法**

使用神经算子（FNO、Transolver）、Newmark‑β 隐式积分、静态平衡数据预训练、有效力与增量刚度输入，神经网络直接映射到位移；实现端到端可微，GPU 并行推理，避免传统 FE 中的 Newton 迭代。

**📊 数据集**

通过有限元静态平衡计算得到多种几何、尺寸、载荷和增量刚度样本，构成线性和非线性弹性动力学的多维案例（二维板、穿孔板、多层剪切墙、三维固体、Neo‑Hookean 条柱、Neo‑Hookean 面板）以及不同外力幅值、时间步长、初始条件的动态轨迹用于评估。

**📈 对比分析**

与基于动态轨迹训练的神经解算器（FNO、LSTM/ConvLSTM、Transolver）以及传统隐式有限元（Newton、modified Newton）进行对比；INOS 在相同网格和时间设置下误差低于动态轨迹训练模型，速度比 FEM 提升 120–4,600 倍；在线性、非线性案例中保持 1% 以下相对误差，并能稳定执行百万步推理。

**⚠️ 局限性**

需要静态数据覆盖有效力和增量刚度范围，否则性能下降；对网格细化时准确性下降；仅针对第二阶弹性动力学；对不同结构需重新训练；对添加刚度、时间步长的覆盖要求严格；缺乏理论误差界限，需经验验证。

---

## 306. Whose Values Are Generated? Measuring Visual Value Leakage in Multilingual Text-to-Image Generation

**arXiv ID:** 2611.05578 | [PDF](https://arxiv.org/pdf/2611.05578v1)

**作者:** Zucan Lyu `[一作]` (University of Wisconsin--Madison), Junjie Hu `[通讯]` (University of Wisconsin--Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们设计了一个多语言基准，用六个霍夫斯泰德文化价值维度来评估文本到图像（T2I）系统在中性社交场景中的文化表现；

**💡 创新点**

创新点在于聚焦于文化价值取向而非仅识别文化符号，并提出“视觉价值泄露”概念，揭示模型倾向性；

**🔧 技术方法**

采用多种T2I系统（如HiDream-O1、FLUX.2[dev]、Ideogram 4.0等）生成图像，并通过提示语言与地理线索进行实验；

**📊 数据集**

使用960个中性社交场景的提示集合，涵盖不同语言和地理标识，作为评估数据集；

**📈 对比分析**

对比分析显示，尽管提示中性且多语言，但生成图像在六个维度上始终倾向同一价值取向，提示语言影响有限，地理线索仅在部分维度产生偏移；

**⚠️ 局限性**

局限在于发现系统表现高度同质化，缺乏多元文化价值体现，且实验范围受限于所选场景与提示，未覆盖更广泛的文化细节。

---

## 307. FedIA: Instance-Aware Cross-Modal Representations for Federated Prompt Learning

**arXiv ID:** 2611.05994 | [PDF](https://arxiv.org/pdf/2611.05994v1)

**作者:** Xuanjing Zhang `[一作]` (Xi'an Jiaotong University), Junge Bo `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在联邦学习场景下，提出FedIA框架，利用全局文本提示指导视觉补丁选择，生成实例感知的本地文本表示，并通过自适应全局‑局部融合调节个体样本的语义特征。

**💡 创新点**

创新点包括：1）把本地提示从客户端级别提升到实例级别，使用视觉补丁动态生成文本表示；2）利用全局提示作为语义锚点实现补丁筛选；3）设计自适应门控网络对全局与局部文本进行实例化融合。

**🔧 技术方法**

技术手段：联邦学习框架、CLIP预训练模型、注意力/交叉注意力模块、软门控网络、补丁特征选择与加权、梯度聚合（FedAvg）。

**📊 数据集**

实验使用多域数据集（PACS、OfficeHome、Office31）、单域多任务数据集（Food101、Caltech101、Flowers102、OxfordPets、DTD）以及CIFAR-10/100的Dirichlet划分。

**📈 对比分析**

与多种基线（PromptFL、FedMaPLe、FedOTP、FedPGP、FedTPG、FedPHA、pFedMMA、ZS-CLIP）进行对比，FedIA在域漂移、标签分布不平衡和Dirichlet不均衡三类异构设置下均取得最高平均准确率，提升幅度为1.44%–4.37%。

**⚠️ 局限性**

局限性：仍需大量全局提示参数同步，补丁选择与实例生成的计算开销较大；在极端极端数据稀缺或高度非IID场景下的鲁棒性尚未深入验证。

---

## 308. Short-Term Forecasting of Heterogeneous Wind Power Fleets: Global Temporal Fusion Transformer Leveraging Static Covariates

**arXiv ID:** 2611.06130 | [PDF](https://arxiv.org/pdf/2611.06130v1)

**作者:** Viktor Walter `[一作]` (Karlsruhe University of Applied Sciences), Andreas Wagner `[通讯]` (Karlsruhe University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在48小时内风电功率预测中，使用全球Temporal Fusion Transformer（TFT）模型对100个德国风电场的异质数据进行训练，并与单场本地模型和线性回归基线进行对比。

**💡 创新点**

提出将静态上下文（高度、海拔、机组参数）与多源NWP（ICON-D2、HRES）结合的全球TFT，并证明在缺乏静态特征时全球模型失效。

**🔧 技术方法**

使用Temporal Fusion Transformer、可变特征选择网络（VSN）以及基于Optuna的超参数优化，对齐多尺度时间序列并进行量化损失训练。

**📊 数据集**

利用包含160个德国风电场的公开合成与真实SCADA数据，结合ICON-D2和HRES的风速预报，并在100个合成场和11个真实运营场进行验证。

**📈 对比分析**

通过滚动时间序列交叉验证、Wilcoxon符号秩检验与Holm–Bonferroni校正，对比R²、RMSE、MAE；结果显示全局TFT在87%（合成）和77%（真实）风电场上优于本地模型，RMSE提升约2–3%且统计显著。

**⚠️ 局限性**

主要限制在于仅使用风速特征，未考虑更丰富的气象变量；对真实数据未进行清洗，缺乏概率预测与系统性错误校正，且模型在不同地理区域外的迁移性能未验证。

---

## 309. mmHand: Toward Pixel-Level-Accuracy Hand Localization Using a Single Commodity mmWave Device

**arXiv ID:** 2611.05404 | [PDF](https://arxiv.org/pdf/2611.05404v1)

**作者:** Xiaoyu Zhang `[一作]` (State University of New York at Buffalo), Wenyao Xu `[通讯]` (State University of New York at Buffalo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过单个毫米波设备构建mmImage生成工具、质量评估与像素空间标注，并提出跨模态空间特征增强模型，实现像素级精确手部定位。

**💡 创新点**

创新点包括将毫米波原始数据转换为二维图像格式以充分挖掘空间信息，设计新颖的质量评估与像素标注方法，并融合跨模态特征实现高精度定位。

**🔧 技术方法**

利用mmWave传感器、mmImage生成、二维图像表示、质量评估、像素标注、跨模态空间特征增强深度学习模型。

**📊 数据集**

在12名受试者、3个不同场景下收集的毫米波雷达数据上进行实验，未使用公开数据集。

**📈 对比分析**

采用四个性能指标与现有手部定位技术对比，mmHand在像素级误差、定位精度等方面明显优于传统方法，验证了系统的有效性。

**⚠️ 局限性**

实验样本受限于12名受试者和有限场景，缺乏对遮挡、多手、复杂环境的全面评估；单一mmWave设备的覆盖范围与鲁棒性仍待提升。

---

## 310. Functional inequalities from the fundamental limits of Bayesian estimation

**arXiv ID:** 2611.05831 | [PDF](https://arxiv.org/pdf/2611.05831v1)

**作者:** Shuchen Li `[一作]` (Yale University), Ilias Zadik `[通讯]` (Yale University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种新的贝叶斯信息论框架，用来证明对数凹测度的 Φ‑Sobolev 不等式，并给出了它与贝叶斯估计下的 Bregman 损失下的 Bayes 风险之间的统一等价关系。利用该等价关系与检测‑估计降维、MMSE 下界、χ² 散度等信息论工具，作者推导出了多种经典与新的功能不等式，包括 Poincaré 常数 O(√n)、对数 Sobolev 常数 O(K² n^{1/4})，并对 Kannan–Lovász–Simonovits（KLS）猜想的最新证明给出了贝叶斯视角的重新表述。

**💡 创新点**

核心创新点在于：
1) 将 Φ‑Sobolev 不等式与贝叶斯估计的 Bayes 风险等价化，建立了两者之间的双向桥梁；
2) 通过“检测‑估计”递归将信息论下界直接转化为功能不等式的上界；
3) 结合低阶 MMSE 与混合累积量的技术，提供了一种新的思路来解释并简化最近的 KLS 证明；
4) 取得了对 log‑Sobolev 常数的新的上界 O(K² n^{1/4})，显著优于之前的 O(K² n^{1/2})。

**🔧 技术方法**

主要技术包括：
- 贝叶斯估计与 Bregman 损失的风险分解；
- Φ‑Sobolev 与信息理论（尤其是 χ² 与相对熵）的收缩性质；
- 检测‑估计降维（如 Le Cam/Fano、信息‑估计关系）；
- 低阶 MMSE 与高阶混合累积量的分析；
- 斯特恩–埃米尔与马尔可夫链等工具的组合。

**📊 数据集**

由于本文是理论数学研究，未使用任何实验数据集，所有结果均为理论证明与不等式。

**📈 对比分析**

与以往工作相比，本文在理论上取得了更优的上界：
- Poincaré 常数从 O(n) 降至 O(√n)；
- 对数 Sobolev 常数从 O(K² n^{1/2}) 降至 O(K² n^{1/4})；
- 对 KLS 猜想的最新证明提供了新的贝叶斯解释，进一步说明了等价性与低阶 MMSE 的关系。整体上，本文通过信息论视角实现了对传统几何不等式的显著提升。

**⚠️ 局限性**

局限性与未解决问题包括：
- 仍需要假设测度为光滑对数凹、子高斯等结构；
- 对数 Sobolev 常数的上界虽然改进，但仍未达到理想的维度无关界 K²；
- 等价关系的证明依赖于复杂的后验分析，若要推广到更一般的通道或测度，可能需要新的工具；
- 对于 KLS 猜想，尽管给出了贝叶斯视角，但真正的维度无关证明仍需进一步研究。

---

## 311. Agents, Think Only When it Helps

**arXiv ID:** 2611.05709 | [PDF](https://arxiv.org/pdf/2611.05709v1)

**作者:** Min-Seon Kim `[一作]` (North Carolina State University), Jung-Eun Kim `[通讯]` (North Carolina State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 LLM 代理的知识蒸馏中思考（thought）监督的作用，探究在不同信息披露程度下思考是否必要

**💡 创新点**

提出思考可被挑选性省略：当动作已充分披露思考信息时，去掉思考监督不会影响甚至提升性能；当信息不被动作披露时，思考监督才显重要

**🔧 技术方法**

采用 ReAct 框架，使用思考与动作独立扰动、SFT 与 KL 蒸馏目标，构建高/低披露情形的实验

**📊 数据集**

在 HotpotQA、FEVER、WebShop、ALFWorld、FinQA、MultiHiertt、MathQA、GeoQA 等多任务基准上进行评估

**📈 对比分析**

与完整轨迹蒸馏比较：思考去除后保留率 97‑110%，训练时间缩短 5.3‑10.2%，推理时间缩短 11.1‑41.7%，在高披露设置下性能保持甚至提升

**⚠️ 局限性**

局限性：仅在 ReAct 交互式环境和文本工具调用场景验证；未探讨更复杂推理形式、跨模态任务或其他教师模型的普适性

---

## 312. Can Astra Solve RoboMME without Breaking the Bank?

**arXiv ID:** 2611.06184 | [PDF](https://arxiv.org/pdf/2611.06184v1)

**作者:** Bingao Chen `[一作]` (Stanford University), C. Karen Liu `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个三层级系统：规划器 GPT-6 Astra、视觉-语言-动作策略和监控器 Qwen3-VL-4B，用于解决 RoboMME 机器人操作任务。

**💡 创新点**

创新点在于引入事件触发的监控器以判断子任务完成，显著减少规划器调用频率并提高效率。

**🔧 技术方法**

使用了大型语言模型 GPT-6 Astra 进行子任务规划、视觉-语言-动作策略进行低级控制，以及微调的 Qwen3-VL-4B 作为进度监控。

**📊 数据集**

在 RoboMME 公开测试集（包含 16 个任务）上进行实验。

**📈 对比分析**

与多种基线（如 FrameSamp+Modul、SAM2Act+、MemER、GroundSG+QwenVL 等）对比，整体成功率达 79.13%，比最佳非 oracle 基线高 34.62 点，仅比 oracle 基线低 4.95 点。

**⚠️ 局限性**

局限性包括缺乏 ablation 研究、未评估成本和通用性、监控误报可能破坏进度记录、以及对实时场景中规划暂停的影响未做测试。

---

## 313. Accelerating Long-Context Decoding in Large-Scale Hybrid LLMs Without Retraining

**arXiv ID:** 2611.06133 | [PDF](https://arxiv.org/pdf/2611.06133v1)

**作者:** João Eduardo Batista `[一作]` (RIKEN Center for Computational Science), Mohamed Wahib `[通讯]` (RIKEN Center for Computational Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不重新训练模型的前提下，对混合型LLM中剩余的全注意力层应用可变窗口、固定前缀和动态Quest稀疏注意力，限制每层关注的Token数量，从而加速长文本推理。

**💡 创新点**

创新点在于将滑动窗口、前缀固定和Quest动态选择三种稀疏注意力方法组合，并使用NSGA-II多目标搜索逐层自适应确定窗口大小和Quest预算，既不改动权重又能在长上下文任务中实现显著速度提升。

**🔧 技术方法**

使用的技术包括：vLLM插件实现稀疏注意力，Triton/FlashAttention底层实现，Quest页面选择、滑动窗口和前缀固定机制，NSGA-II多目标搜索算法，以及针对不同硬件的GPU加速。

**📊 数据集**

评估数据集涵盖21个长上下文基准：BABILong（qa1-qa5）和16个LongBench子集（单/多文档QA、摘要、few-shot、检索、代码生成）等。

**📈 对比分析**

对比基线模型，在64k和256k token长上下文下，混合模型可获得1.16–1.26×（64k）和2.06–2.23×（256k）的推理速度提升，且在所有评测数据集上准确率保持≤3pp，说明方法在保持质量的同时显著提升吞吐量。

**⚠️ 局限性**

局限性：在短文本（<32k token）时比基线慢；Quest保持完整KV缓存，未减少内存占用；对非混合或极大模型（>400B）提升有限；仅适用于长上下文任务，不是通用速度优化方案。

---

## 314. StarWorld: A Cross-Layer World Model for Starlink Prediction and Streaming

**arXiv ID:** 2611.05832 | [PDF](https://arxiv.org/pdf/2611.05832v1)

**作者:** Bowen Zhang `[一作]` (Ningbo Artificial Intelligence Institute, Shanghai Jiao Tong University), Jian Chu `[通讯]` (Ningbo Artificial Intelligence Institute, Shanghai Jiao Tong University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 StarWorld——一种跨层次的世界模型，用于在不确定的 Starlink 访问条件下预测网络吞吐量并进行自适应比特率流媒体播放。

**💡 创新点**

创新点：① 将随机网络动态、预测模块和播放过程融合为可执行的“跨层想象”，② 引入 Trust Gate 根据不确定性、数据支持和滚动时限动态判断想象结果的可信度。

**🔧 技术方法**

技术：因果 Transformer + 高斯输出的网络动态学习；GRU+Gaussian Latent的回放修正；可执行预测/控制器调用；深度集成（ensemble）估计不确定性；联邦训练与校准（bias、coverage、支持区域）。

**📊 数据集**

数据集：来自 Osnabrück、Chicago、Victoria 三地的 5,814 条 10 分钟实时 Starlink 测量段；1,024 条测量段上完成预测和播放重放（共 27,745 个 10 秒窗口）。

**📈 对比分析**

对比方法：统计方法（Persistence、Recent mean、Autoregressive、Causal empirical）和神经网络模型（GRU、Causal Transformer、RSSM）；可执行回放基线（Persistence+equations、Per-layer、Joint recursion）。性能：StarWorld 在 30/60 秒预测 CRPS、低尾 twCRPS、90% 置信区间覆盖率上均优于所有基线；在跨层回放误差（缓冲、累计重缓冲）和共享未来控制器效果评估中，误差比基线低 24%–48%，并在约束满足率和风险限定回退方面接近理想回放。

**⚠️ 局限性**

局限：对低尾分布的长时延预测（>120 秒）和极端低容量情形的可靠性仍有限；Trust Gate 在极端网络/控制器组合下的拒绝率略高；模型依赖丰富的历史测量，无法在数据稀缺的新地区或极端天气下保证表现；未考虑网络层控制与应用层决策的双向耦合。

---

## 315. Does RRT* Work on Uneven Terrain Meshes?

**arXiv ID:** 2611.05885 | [PDF](https://arxiv.org/pdf/2611.05885v1)

**作者:** Otobong Jerome `[一作]` (Universidade Federal da Paraíba), Tiago Nascimento `[通讯]` (Universidade Federal da Paraíba)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文将快速随机树（RRT*）移植到三角网格地形上，研究其在无体积表面上的采样、连线与重连过程，并分析不同线段实现对路径成本的影响。

**💡 创新点**

提出Mesh RRT*框架，并对两种Line实现——直线弦（Chord）和面路径（Surface）进行理论与实验对比，揭示弦实现导致的成本偏差及其随步长与容差线性变化的性质；证明面路径实现消除该偏差并保持收敛性。

**🔧 技术方法**

利用RRT*与Mesh RRT*算法，欧氏与测地距离度量，MMP测地算法，Dijkstra图搜索，热方法/快速行进等技术。

**📊 数据集**

使用合成高度场地形（Egg‑crate 形状）及其不同粗糙度幅度（A=0,0.5,1,2,4），共529个顶点、968个三角面。

**📈 对比分析**

与精确测地线（MMP）以及两种图规划器（边图与线景图）进行对比；弦实现报告成本可低达2.7%于真实测地线，但路径长度仅比测地线高0.1–0.5%；面路径实现成本接近真实测地线（误差0.07–0.16%），收敛速率相当，首次解决时间略优。

**⚠️ 局限性**

仅在无孔高程场上实验；欧氏最近邻在折叠面上可能不稳定；缺乏完整的概率完备性与渐进最优性证明；数据集规模小，样本数有限，缺少尾部统计。

---

## 316. From Waste to Discovery: Recycling High-Scored Unexposed Candidates in Taobao Feed Recommendation

**arXiv ID:** 2611.05997 | [PDF](https://arxiv.org/pdf/2611.05997v1)

**作者:** Ge Fan `[一作]` (Taobao & Tmall Group of Alibaba), Bo Zheng `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为URED（未曝光回收以高效发现）的系统，该系统在阿里巴巴的淘宝推荐系统中部署，旨在帮助用户发现新的兴趣，覆盖数千万日活跃用户。

**💡 创新点**

创新点在于通过回收高评分但未曝光的样本，避免了传统探索方法的低效和风险，显著提高了发现率和内容丰富度，同时保持了用户参与度。

**🔧 技术方法**

使用了一种轻量级的、无训练的回忆阶段系统，结合离线存储、分布式聚合和低延迟在线服务。

**📊 数据集**

使用了淘宝平台的生产数据，聚焦于未曝光的高评分样本，经过30天的滑动窗口聚合。

**📈 对比分析**

通过大规模在线A/B实验与传统方法进行比较，URED的发现率比平台平均高出13.9倍，内容丰富度提高了8.47%，同时用户参与度得以保持。

**⚠️ 局限性**

限制在于系统依赖于模型对未曝光项目的评分能力，若模型在某些用户群体上表现不佳，可能会影响系统的效果。

---

## 317. 3D Banff Glomerulitis Scoring for Kidney Transplant Pathology: Feasibility Study for Volumetric Quantification

**arXiv ID:** 2611.05925 | [PDF](https://arxiv.org/pdf/2611.05925v1)

**作者:** Yanfan Zhu `[一作]` (Vanderbilt University), Yuankai Huo `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了肾移植活检的三维 Banff 肾小球炎症评分框架，并验证其技术可行性与临床相关性。

**💡 创新点**

首次将传统单层 worst‑section 规则推广到三维空间，提出兼容、体积和去重体积三种评分策略，实现跨切片肾小球身份追踪与炎症细胞聚合。

**🔧 技术方法**

结合 YOLOv8s 检测、KPIs‑挑战分割、Monkey Challenge 细胞检测，使用 XFeat 变形/仿射注册与 Union‑Find 追踪，完成三维重建与分级。

**📊 数据集**

使用维克森林医学院 50 例肾移植活检共 410 张 WSI，切片数 20–30 条，包含 PAS、H&E、Jones 银染色。

**📈 对比分析**

与传统 2D worst‑section、2D 多切片投票和 2D 合并方法比较；半自动模式下 3D 体积分数与 6 个月肾功能变化 Kendall τ≈0.51，显著高于 2D 的 0.31；全自动模式差异不大；WSI 内一致性从 82.9% 提升至 90.8%。

**⚠️ 局限性**

缺乏 3D 参考分数；样本量小且单中心；细胞去重对结果影响有限；需要更大规模多中心、专家标注验证。

---

## 318. Selective, Risk-Controlled Emotion Forecasting in Conversation: From a Persistence Barrier to Departure Detection with Conformal Guarantees

**arXiv ID:** 2611.05896 | [PDF](https://arxiv.org/pdf/2611.05896v1)

**作者:** Bin Wen `[一作]` (Universiti Sains Malaysia), Tien-Ping Tan `[通讯]` (Universiti Sains Malaysia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文系统评估了对话情绪预测任务，发现在已知历史情绪状态的实验设置下，传统“预测下一情绪状态”的目标难以突破同一说话人持久性基准；基于此，作者提出将任务转化为“选择性偏离预测”，仅预测情绪状态是否即将偏离，并为此设计了风险控制方法；

**💡 创新点**

创新点包括：①提出并验证“Persistence Barrier”概念；②将目标改为选择性偏离预测并引入基于先导时间加权的评估指标；③在对话级别实现分位数‑风险控制（Conformal Risk Control），提供期望损失的分布无关保证；④系统比较了多种承诺策略（阈值、成本敏感、最优停止、元策略）并展示其在不同数据集上的性能；

**🔧 技术方法**

使用技术包括：冻结预训练编码器（RoBERTa, WavLM）+ 训练头（GRU, PsyDyn, Latent Dynamics）、卡尔曼滤波基线、离散 OU 过程、离散化的离散偏离检测、选择性预测框架（f,ρ）、对话级别的分位数‑风险控制、温度缩放校准、AUROC/AUPRC、AUPRC 计算等；

**📊 数据集**

采用的公开数据集有 DailyDialog、MELD（双人子集与多说话人版本）、IEMOCAP（连续 VAD）、EmoryNLP、Hi‑EF；

**📈 对比分析**

在已观测状态设置下，传统的值预测模型（如 GRU、PsyDyn 等）在 CCC 上远低于同一说话人持久性基准（最高 0.65 vs 0.14）；而在偏离预测任务上，AUROC 在 0.72–0.56 范围，AUPRC 达到 0.305，证明该目标可学习；与基线相比，加入风险控制后平均风险可控制在设定阈值 α 以内，同时覆盖率在不同 α 下可调；

**⚠️ 局限性**

主要局限包括：①实验仅在冻结编码器、已观测状态的测试环境，未探究完整端到端或更大模型的表现；②假设目标说话人身份已知，对多说话人情绪预测的下一说话人未知时不适用；③依赖持久性基准的“anchor”可观测性，一旦需从原始信号估计，持久性优势消失；④数据集划分与原始划分不同，可能影响可比性；⑤LLM自报置信度在某些模型上不可靠；

---

## 319. A Feasibility Condition for Confidence-Gated Tiered LLM Inference

**arXiv ID:** 2611.05394 | [PDF](https://arxiv.org/pdf/2611.05394v1)

**作者:** Sumit P. Ahuja `[一作]` `[通讯]` (Main Labs), Sumit P. Ahuja (Main Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

无法获取论文内容，无法说明研究工作

**💡 创新点**

无可说明的创新点

**🔧 技术方法**

无可说明的技术方法

**📊 数据集**

无可说明的数据集

**📈 对比分析**

无可说明的比较方法与性能表现

**⚠️ 局限性**

缺少论文内容导致无法评估限制

---

## 320. VAR-Q: Tuning-free KV Cache Quantization for Visual Autoregressive Image and Video Generation

**arXiv ID:** 2611.05380 | [PDF](https://arxiv.org/pdf/2611.05380v1)

**作者:** Boxun Xu `[一作]` (University of California Santa Barbara), Peng Li `[通讯]` (University of California Santa Barbara)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对视觉自回归模型的KV缓存低精度量化方法VAR‑Q，利用通道与块双轴分组实现无校准、无超参数的高效压缩。

**💡 创新点**

创新点在于发现KV缓存中仅少量大幅值通道与块构成两轴结构，并据此设计了兼顾图像与视频两大自回归范式的统一双轴分组量化策略。

**🔧 技术方法**

采用RoPE分析、投影链拆解、通道-块联合分组、统一对称量化以及Trition子字节打包等技术实现低位宽存储和高效推理。

**📊 数据集**

在Infinity、InfinityStar、Self‑Forcing、LongLive等模型上，使用ImageNet‑1K、MovieGen、VBench、GenEval等公开图像/视频/文本数据集进行评测。

**📈 对比分析**

与KIVI、FlexGen等LLM迁移方案对比，VAR‑Q在INT4/INT3/INT2下保持几乎无质量损失或优于基线，KV内存可削减至78.5%，推理批次可提升至6倍。

**⚠️ 局限性**

局限性包括极低位宽下误差显著上升、对极长序列/超大模型的鲁棒性未充分验证，以及算法在不同框架和硬件平台的兼容性需要进一步改进。

---

## 321. SBVUL: Statement-Based Vulnerability Detection

**arXiv ID:** 2611.06135 | [PDF](https://arxiv.org/pdf/2611.06135v1)

**作者:** Martin Pinzger `[一作]` (University of Klagenfurt), Víctor Mayoral Vilches `[通讯]` (Alias Robotics)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出一种两阶段代理式方法，利用大型语言模型（LLM）先识别 C/C++ 函数中的潜在可利用语句（Phase 1），再用第二个代理对每条候选语句进行控制流与数据流分析，确认、驳回或标记不确定（Phase 2），并基于 Phase 2 的判定给出函数级漏洞分类。

**💡 创新点**

创新点包括：①将候选发现与确认拆分为两步，显著降低误报；②不依赖训练或编译，直接利用文件系统和文本搜索工具收集证据；③使用 LLM 评判确认语句的解释是否符合人工验证的根因；④提供人工验证的可利用语句 ground truth 数据集，为后续研究提供高质量基准。

**🔧 技术方法**

技术手段包括：LLM 代理实现（使用 LangChain 与 Claude Code Agent SDK），配套文件系统与 Ripgrep 工具；采用四款模型（Qwen 3.8、DeepSeek V4.1、Sonnet 5.5、Opus 5.5）进行实验；使用 Sonnet 5.5 作为 LLM judge，对确认语句的解释进行匹配。

**📊 数据集**

使用 PrimeVul Paired 测试集的 200 对函数（200 个 vulnerable/patch 组合），共 104 项目、45 种 CWE，人工验证得到 380 条可利用语句作为 ground truth。

**📈 对比分析**

与现有方法（VulTriage、VulTrial、VulInstruct、SAST 等）比较，Opus 5.5 在函数级别能正确分类 42% 的对，MCC 0.449；在语句级别 F1 0.414、精度 0.315、召回 0.605；相较于 SAST 工具，匹配已知漏洞的语句比例提升至 31.5%。

**⚠️ 局限性**

局限性包括：所有模型在补丁函数中仍确认 23–61% 的候选语句，导致误报率偏高；模型对可利用语句的完整覆盖仍有限；实验仅覆盖 C/C++ 代码，未验证其他语言；模型差异与实现差异可能影响结果，且 LLM judge 与被评判模型同属一族，可能产生偏好。

---

## 322. Building a Method: Challenges and Lessons from Developing an Impact Evaluation Framework for Public Investment in Open Source Software

**arXiv ID:** 2611.06281 | [PDF](https://arxiv.org/pdf/2611.06281v1)

**作者:** Laia Domenech Burin `[一作]` `[通讯]` (Sovereign Tech Agency), Laia Domenech Burin (Sovereign Tech Agency)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并试点一套面向公共投资开放源代码软件的混合方法评估框架，结合定量指标与定性情境分析；

**💡 创新点**

提出以项目目标为先导的Goal‑Question‑Metric（GQM）指标体系，结合CHAOSS社区指标模型，兼顾项目异质性与可衡量性；

**🔧 技术方法**

使用GQM框架、CHAOSS指标模型、自然语言处理（对邮件列表等文本的初步处理）、GitHub及生态系统数据API、Git历史安全信号追踪工具（git‑pkgs/signals）等技术；

**📊 数据集**

核心数据集为：https://ecosyste.ms/（开源生产数据集）、GitHub、GitLab、Bitbucket等版本控制仓库数据、CVSS公开数据库及内部邮件/聊天记录（仅用于定性分析）；

**📈 对比分析**

采用描述性时间序列可视化与定性情境对照来比较投资前后指标变化，未实现因果推断，性能表现主要为可视化洞察与初步趋势识别，后续计划引入Generalized Synthetic Control等因果方法；

**⚠️ 局限性**

局限包括：指标定义与解释依赖社区专业判断，数据来源碎片化与完整性不足，因果归因缺乏实验设计，项目异质性导致指标可比性受限，研究框架对新技术快速演进易失效；

---

## 323. Network Intrusion Forensic Detection System with Explainable Artificial Intelligence and Digital Chain of Custody

**arXiv ID:** 2611.06173 | [PDF](https://arxiv.org/pdf/2611.06173v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 324. MotoVerse: A Benchmark for Predictive Perception of Motorized Two-Wheelers in Dense Unstructured Traffic

**arXiv ID:** 2611.06007 | [PDF](https://arxiv.org/pdf/2611.06007v1)

**作者:** Varun A. Paturkar `[一作]` (Indian Institute of Information Technology Hyderabad), C. V. Jawahar `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MotoVerse benchmark，包含 949 个从摩托车前视摄像头拍摄的动作片段，提供了骑手操纵意图（ERMA）和周围交通参与者轨迹预测（TATP）的标签，并为 MTW 特有的三类挑战（动态自我运动、弱交通结构、情境遮挡）提供评测框架。

**💡 创新点**

创新点：
1) 大规模、密集轨迹注释的 MTW 视角数据集，覆盖 31,000+ 参与者（其中 16,000+ MTW）。
2) 为 MTW 设计的挑战分类体系，支持细粒度鲁棒性评估。
3) 系统对比了传统视频预测模型、通用 VLM、驾驶专用 VLM 及 VLA，首次揭示 VLM 在 MTW 场景中的性能瓶颈。
4) 通过 LoRA 参数高效微调验证了 VLM 的潜在提升空间。

**🔧 技术方法**

技术方法：
- 视频预测骨干：C2D、I3D、X3D、SlowFast。
- 轨迹预测模型：PIETraj、BiTraP、ABC+、SGNet。
- 通用 VLM：InternVL2.5、Qwen3-VL。
- 驾驶专用 VLM/VLA：Cosmos-Reason2-8B、UniDriveVLA、DriveLMM-o1、Alpamayo。
- LoRA 微调。
- 评测指标：ERMA（Accuracy、Weighted‑F1、Macro‑F1），TATP（ADE/FDE、sADE/sFDE、AIoU/FIoU）。

**📊 数据集**

使用的数据集：MotoVerse（949 片段，约 50 小时、700 km，含 31,000+ 车辆轨迹，16,000+ MTW）。

**📈 对比分析**

对比方法：传统预测模型在 ERMA 最高 58.5%（准确率），VLM 大部分接近基准；LoRA 微调后 VLM 在 ERMA 上可达 80%+；在 TATP 上，SGNet 取得最优 ADE ≈ 51.8，VLM 仍落后于传统模型，且大多仅略优于静态基线。

**⚠️ 局限性**

局限性：
1) 仅使用单摄像头视角，未覆盖 LiDAR 等多模态。
2) 数据仅来自印度，可能缺乏跨地域泛化。
3) VLM 评估为零样本，未充分利用其推理能力。
4) 只关注 MTW，未考虑四轮车等其他车辆交互。
5) 训练集规模虽大但仍有限，难以覆盖所有极端情境。

---

## 325. Do Saliency Maps Fail Dense Breasts? A Chance-Normalized, Confounder-Aware Audit of Explanation Fairness in Mammography

**arXiv ID:** 2611.05909 | [PDF](https://arxiv.org/pdf/2611.05909v1)

**作者:** Ahmed Jaheen `[一作]` `[通讯]`, Ahmed Jaheen

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

审计了三种后置可解释性方法（Grad‑CAM、Integrated Gradients、遮挡法）在不同乳腺密度（BI‑RAD 1‑2 vs 3‑4）下对恶性乳腺病灶定位与能量提升的差异，采用机会标准化的指向游戏和能量提升评分，并使用GEE回归分离密度效应与病例混合的影响。

**💡 创新点**

首次针对乳腺密度进行逐像素病灶掩码审计；提出基于能量提升和指向游戏的机会标准化评分；将病例混合、分类器准确度和病灶可见度纳入多层回归模型；并提供完整的审计报告模板（粗略、病例混合调整、控制比较）。

**🔧 技术方法**

Grad‑CAM、Integrated Gradients、遮挡法；能量提升（λ_E）与指向游戏评分；ROAD、SRG、LSR 信度与依赖度评估；GEE回归、TOST、Holm多重检验；DoG显著度图、随机网络控制。

**📊 数据集**

CBIS‑DDSM数字化影像数据集，1355个恶性视图（751名患者），已标注BI‑RAD密度与辐射学家绘制的病灶掩码。

**📈 对比分析**

通过粗略差距和多层回归（M0–M3）比较，发现粗略差距显著（如Grad‑CAM 44% vs 58%指向率，-0.86位元能量提升差距）。病例混合调整后，Grad‑CAM与IG的差距可视为等效（90%CI在±0.5位元以内），遮挡法在EfficientNet架构下仍显著（-0.56位元）。指向游戏差距小于10个百分点，能量提升差距均低于0.5位元。

**⚠️ 局限性**

仅使用单一数字化影像数据集且密度标注来源于读者，可能存在标签噪声；未验证在数字化乳腺X线或真实筛查流程中的适用性；实验仅覆盖三种CNN架构，未涵盖所有可能的网络设计。

---

## 326. Training-Free Character-Length Control in Summarization with Diffusion Language Models

**arXiv ID:** 2611.06371 | [PDF](https://arxiv.org/pdf/2611.06371v1)

**作者:** Steffen Castle `[一作]` (German Research Center for Artificial Intelligence), Sebastian Möller `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Delta算法，实现无训练、推理时动态调整掩码，控制掩码扩散语言模型输出的字符长度

**💡 创新点**

利用掩码分布的期望字符长度实时估计并根据目标长度插入或删除掩码，实现精确长度控制；不需要微调、无架构改动

**🔧 技术方法**

掩码扩散语言模型（LLaDA、Dream）+期望长度估计+自适应掩码插入/删除+置信度阈值策略

**📊 数据集**

XSum、CNN/DailyMail、arXiv三大英文摘要数据集

**📈 对比分析**

与文本提取、提示式、自动回归长度控制方法对比，Delta在MAE、<10%误差、ROUGE、BERTScore、LLM评估中取得近乎最佳或最优质量，同时在长度合规性上优于所有基线（误差下降至≤3字符）

**⚠️ 局限性**

推理速度慢（扩散模型需多轮前向传递且无法KV缓存），阈值需手工调优，主要评测限于英语和中等规模模型，其他语言与更大/更小模型的鲁棒性待验证

---

## 327. When Does Sharing Work? Scaling a Shared SigLIP2 Across Image, Video, Audio, and Audio-Visual Representations

**arXiv ID:** 2611.05406 | [PDF](https://arxiv.org/pdf/2611.05406v1)

**作者:** Xiaofei Sun `[一作]` (Alibaba Group), Lin Qu `[通讯]` (Alibaba Group)

**通讯引用:** 1946 | [OpenAlex ID](https://openalex.org/A5100763937)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Omni-Encoder，一个基于 SigLIP‑2 ViT 的全共享参数多模态编码器，能够同时处理图像、视频、音频和音频‑视觉输入。

**💡 创新点**

通过模态本地化 tokenizer、绝对时间戳的 3D RoPE、音视频段级交错以及交叉模态梯度累积，实现端到端共享参数、跨模态协同学习，避免传统锚点绑定与专属适配器。

**🔧 技术方法**

使用 SigLIP‑2 ViT 结构、1D 卷积音频 tokenizer、T/H/W 三维 RoPE、音视频交错 token、交叉模态梯度累积、Sigmoid 对比损失、DeepSpeed ZeRO‑1、FlashAttention‑2 与 bf16。

**📊 数据集**

四模态大规模数据集：图像‑文本（170M DataComp）、视频‑文本（Mix42M）、音频‑文本（42M WavCaps+补充）、音频‑视觉（30M YouTube/AudioSet/VGGSound/YouCook2），按 64:16:64:16 比例混合训练。

**📈 对比分析**

在 16 个基准（图像分类、图像‑文本检索、视频‑文本检索、音频‑文本检索、音频‑视觉检索）上，使用同一检查点零样本评估，结果显示 Omni-Encoder 与单模态专家持平或更优，且在多模态检索上显著优于现有统一编码器。

**⚠️ 局限性**

推理时模型规模与单模态部署相同，无法降低成本；在极端专业任务（如长语音、细粒度音频事件）上仍落后于专用模型，需通过蒸馏或进一步扩充专门化数据来提升。

---

## 328. A Fully Polynomial Approximation Algorithm for the Sherrington-Kirkpatrick Model

**arXiv ID:** 2611.05446 | [PDF](https://arxiv.org/pdf/2611.05446v1)

**作者:** Dylan Herman `[一作]` (JPMorganChase), Guneykan Ozgul `[通讯]` (JPMorganChase)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种改进的增量近似消息传播（IAMP）算法——Grouped IAMP，用来在随机Sherrington–Kirkpatrick（SK）模型中寻找接近最优能量的自旋配置，并给出了在有限维度下的概率保证。

**💡 创新点**

创新点：
1) 引入Reeves耦合构造，得到完全符合高斯分布的查询输出；
2) 将Euler–Maruyama步骤按组打包，显著减少矩阵乘积次数至Θ(1/ε)；
3) 在不需要预先计算Parisi最小化函数的前提下，实现工作量为O(N²/ε)且仅需O(N²)存储；
4) 在维度阈值 N = O(ε⁻² log²(e/ε) log(1/α)) 以上给出明确的相对误差保证，并进一步推出AFPRAS（渐近完全多项式随机近似方案）。

**🔧 技术方法**

技术手段：
- 近似消息传播（AMP）与增量近似消息传播（IAMP）
- Reeves耦合构造实现高斯查询
- Euler–Maruyama离散化与非均匀时间步
- 精细的矩阵投影与正交化技术
- 状态演化（state evolution）与Parisi零温方程的概率分析
- 多维正态分布与Hellinger距离的相对误差估计

**📊 数据集**

数据集：
使用标准高斯正交族（GOE）随机矩阵 A ∈ ℝⁿˣⁿ 作为输入；不涉及外部真实数据集。

**📈 对比分析**

对比方法与性能：
- 与原始Montanari IAMP 和El Alaoui–Montanari–Sellke（AMS）IAMP 对比；
- Grouped IAMP 在维度足够大时，工作量为O(N²/ε)，相对误差为1−ε，失败概率 ≤α；
- AMS IAMP 的工作量为O(N²/ε³ log²(e/ε))，但维度阈值呈指数级别；
- 本算法实现了AFPRAS，在任意维度下给出 1−ε 近似的加性误差不随维度增长，工作量多项式。

**⚠️ 局限性**

局限性：
- 需要满足全支持（full‑support）假设，保证Parisi序参量严格递增；
- 对于小维度 N < N₀ 的情况，算法退化为返回全 +1 配置，需要额外的加性误差；
- 需要预先构造Parisi函数 γ、m、a 的精确评估，虽然其构造开销不计入工作量，但实现难度较大；
- 在实践中实现高精度的Reeves耦合与高阶数的时间步分组可能导致数值误差累积，需要进一步稳定性分析。

---

## 329. Private Vulnerability Testing for User Oversharing

**arXiv ID:** 2611.05779 | [PDF](https://arxiv.org/pdf/2611.05779v1)

**作者:** Alexander Boyd `[一作]` (University of Surrey), Suparna De `[通讯]` (University of Surrey)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种隐私保护的脆弱性检测系统，用户可以私下判断其公开属性组合是否落在低于预设阈值的人口子集内，而不泄露具体计数或查询细节。

**💡 创新点**

创新点在于支持任意后期查询（可选属性子集）并允许贡献者离线；引入了反应式 f‑privacy 概念来处理动态数据库；采用两台非协作服务器通过秘密共享实现仅返回阈值判定结果。

**🔧 技术方法**

核心技术包括布尔秘密共享、Beaver 三元组乘法、B2A 转换、分布式等价测试与分布式 AND、加密的雾化电路（garbled circuit）以及 OT 扩展与预计算优化。

**📊 数据集**

实验使用合成与真实属性集，查询位长分别为 32、64、96 位，数据库规模从 100 万到 1000 万条记录（最多 1 亿条可支持）。

**📈 对比分析**

通过与 INTERSECT、COUTHYBRID、COUTFULL 三种等价测试变体对比，在线查询在 96 位查询下对 1 000 000 条记录可实现 0.1–2 秒的响应，离线预计算线性 O(nℓ)，并可通过多服务器并行进一步加速。

**⚠️ 局限性**

局限性包括：Σ_VT^- 方案泄露查询结构；即使实现了 f‑privacy，阈值输出仍可泄露差分信息；大位长查询与多服务器协作所需的预计算成本高；系统不支持加入噪声的差分隐私，无法避免精确计数导致的潜在重识别风险。

---

## 330. Consistency Is Not Understanding: Orbit Audits for Neural Field Weight Readers

**arXiv ID:** 2611.05798 | [PDF](https://arxiv.org/pdf/2611.05798v1)

**作者:** Sungguk Cha `[一作]`, Sangyeob Lee `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过设计轨道审计框架，评估在保持神经网络函数不变的前提下，对隐藏神经元置换和正ReLU缩放等坐标变换对视觉内容读取器的影响。

**💡 创新点**

创新性在于将功能保持与坐标变换分离，提出多维度评估指标（准确率、同意率、数值误差）以及可解释的可检验评估协议，填补了权重读取研究的可验证性空白。

**🔧 技术方法**

采用隐藏神经元置换、正ReLU缩放、规范化与排序的 canonicalization、统计特征/深度集合读者、冻结小型语言模型的next-token评估，以及数值误差与精度分析技术。

**📊 数据集**

使用了480个独立拟合的形状场（圆、方形、菱形、三角形、交叉、环形等）以及1000个MNIST字符场，分别作为监督读者和语言模型评估的数据集。

**📈 对比分析**

通过比较原始与变换后模型在准确率、同意率以及生成答案重复一致性上的差异，发现绝大多数读者在功能保持但坐标变换后准确率显著下降，部分读者对尺度变换仍敏感；小型语言模型在raw条件下仅略高于随机，在canonical化后性能略提升但仍不显著。

**⚠️ 局限性**

局限性包括仅针对浅层单隐层ReLU场景，未覆盖更深网络、自然图像或非正缩放等更广泛变换；实验规模有限，未能完全检验所有读者训练与推理差异；数值精度控制仅适用于六位数序列化，未证明在更高精度或实际部署环境中的鲁棒性。

---

## 331. CAPE-JEPA - Cross-modal Adversarial Probabilistic Embedding JEPA

**arXiv ID:** 2611.06247 | [PDF](https://arxiv.org/pdf/2611.06247v1)

**作者:** Aniket Junghare `[一作]` (Indian Institute of Technology Bombay), Biplab Banerjee `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对遥感跨模态检索的概率联合嵌入预测架构CAPE-JEPA；

**💡 创新点**

创新点在于：①使用高斯条件预测器建模跨传感器预测不确定性；②引入对抗式分布对齐（XSIGReg）闭合模态分布差距；③结合场景级InfoNCE实现语义区分；

**🔧 技术方法**

采用ViT-B编码器、交叉注意力预测器、负对数似然损失、Epps‑Pulley统计的对抗对齐和InfoNCE；

**📊 数据集**

在BigEarthNet-14K（S1/S2对）和fMoW（SEN/RGB对）两大遥感检索基准上进行训练与评估；

**📈 对比分析**

与MAE、CrossMAE、CROMA、SkySense、REJEPA、X‑JEPA等多种SSL与跨模态基线对比，CAPE‑JEPA在所有八个检索方向上均实现F1@5最高，BEN‑14K跨模态提升约13.7点，fMoW跨模态提升约15.6点；

**⚠️ 局限性**

局限性包括：仍需配对样本；对极端多模态（如SAR+光谱+光学）的扩展待验证；模型复杂度高，训练时间长；在边缘模糊地区的预测不确定性可能影响检索精度。

---

## 332. Optimizing Recourse Under Predictive Multiplicity

**arXiv ID:** 2611.05774 | [PDF](https://arxiv.org/pdf/2611.05774v1)

**作者:** Shahin Jabbari `[一作]` `[通讯]` (Drexel University), Shahin Jabbari (Drexel University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种在预测多样性环境下的鲁棒算法解释（recourse）方法，目标是寻找在给定预算内对所有候选模型均有效的特征修改。

**💡 创新点**

创新点在于将鲁棒recourse建模为对抗式预算约束的最小化最坏情况有效性损失，并提出可使用任何模型集合（不同架构或类）的oracle高效算法ORPM。

**🔧 技术方法**

使用了基于乘数权重更新的对抗动态、oracle优化、混合整数规划、SLSQP梯度搜索及二次规划等技术实现oracle，并证明收敛性。

**📊 数据集**

实验在一个二维合成数据和五个真实数据集（COMPAS、Polish、German、GiveMeSomeCredit、ACSIncome）上进行，涵盖线性、神经网络、混合和树模型。

**📈 对比分析**

与专门针对单一模型或特定模型类的基线（ROAR、ElliCE、ADV、RobX、OCEAN等）对比，ORPM在大多数配置下实现更高的联合有效率，尤其在多样化模型集合中优于现有方法。

**⚠️ 局限性**

局限性包括对预算有限时仍可能无法满足所有模型、对未见模型的泛化有限、对不同族群公平性的影响未能完全消除，以及需要预先构造合理的竞争模型集合。

---

## 333. SVGFaith: A Two-Layer Meta-Evaluation of Faithfulness Metrics for Text-to-SVG Generation

**arXiv ID:** 2611.06048 | [PDF](https://arxiv.org/pdf/2611.06048v1)

**作者:** Guotao Liang `[一作]` (Tencent Youtu Lab), Xing Sun `[通讯]` (Tencent Youtu Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了两层文本到 SVG 的可信度评估基准（SVGFaith‑Diagnostic 与 SVGFaith‑Arena）并设计了可解释的图形需求评分指标 SVGFaith‑Score。

**💡 创新点**

创新点在于将基准拆分为控制对比与真实输出两层，揭示常用指标的盲区，并引入基于需求图和依赖折扣的评分机制。

**🔧 技术方法**

使用域调优的 Qwen3.5‑9B 进行提示分解生成需求图，用 VQA 模型对每条需求进行满足度评分，并通过软门控聚合得分；与 CLIP、SigLIP、奖励模型等现有指标进行对比。

**📊 数据集**

数据集包括 1,842 条受控对比样本（涵盖 8 个能力轴）和 1,327 条由六个生成器产生的真实输出对（SVGFaith‑Arena），提示来自 SArena 与色彩描述集。

**📈 对比分析**

与九个主流评估方法（图像–文本嵌入、奖励模型、视听语言评分）对比，SVGFaith‑Score 在 Arena 上达到 0.880 的成对准确率，在 Diagnostic 上分别取得 0.941/0.953，超越基线并接近人工一致性。

**⚠️ 局限性**

局限性包括对生成器强度的可靠性缺口、对提示分解与 VQA 校准的依赖，以及仅使用光栅化渲染而非原生 SVG 结构导致的潜在信息损失。

---

## 334. LiveFoley: Dynamic Real-Time Video-to-Audio Generation for Immersive Interactive Worlds

**arXiv ID:** 2611.05988 | [PDF](https://arxiv.org/pdf/2611.05988v1)

**作者:** Yusheng Dai `[一作]` (Monash University), Jianfei Cai `[通讯]` (Monash University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套实时视频到音频生成框架LiveFoley，实现低延迟、精准时序同步与高音质的流媒体音频合成。

**💡 创新点**

创新点包括：1）尾部滑动窗口（Trailing Sliding Window）将非因果段级编码器转为实时模块，同时保持完整感受野；2）统一可变跳框架（Unified Varhop）通过同步器实现多种延迟模式共享同一模型；3）历史与未来热身的LocDiT，分别在不增加或仅轻微增加后端延迟的情况下扩展去噪窗口，提高音质。

**🔧 技术方法**

技术手段：基于Synchformer的段级视觉编码、MiniCPM语言模型、LocDiT去噪变压器、LocEnc压缩历史音频、滚动KV缓存、注意力槽机制、低延迟前端后端融合。

**📊 数据集**

数据集：约2000小时（745K视频片段）来自SoundAtlas、VGGSound、AudioSet、EPIC-Sounds、Ego4D-Sounds；评测使用VGGSound测试集及对齐更严的VGGSound-Curated子集。

**📈 对比分析**

与离线与流式基线对比：在0.08s前端延迟下，LiveFoley的音频-视觉同步（DeSync≈0.48）与离线MMAudio-L持平，流式系统中音质指标（FD_PANNs≈9.56、FD_PaSST≈92.48、IS≈11.3）居流式最高；整体端到端延迟约156ms（Future-Warmup为236ms），低于人类可感知阈值。

**⚠️ 局限性**

局限：仍与离线模型在生成质量上存在差距；Future-Warmup虽提升质量但引入额外后端延迟；模型对极端低延迟（<80ms）或多模态混合输入的鲁棒性待进一步验证。

---

## 335. SRRM: A Retrieval-Reconstruction Benchmark for Evaluating Long-Context Memory in Small Language Models

**arXiv ID:** 2611.05386 | [PDF](https://arxiv.org/pdf/2611.05386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 336. Learning Hierarchical Patch Splitting Policies for Faster Vision Transformers

**arXiv ID:** 2611.05742 | [PDF](https://arxiv.org/pdf/2611.05742v1)

**作者:** Aditya Gupta `[一作]` (Carnegie Mellon University), Laszlo A. Jeni `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种名为 SPLIT 的自适应 patchification 框架，通过模型自身的语义信息学习 token 分辨率，在 ViT 输入阶段就能减少序列长度。

**💡 创新点**

创新点在于将 token 重要性决策从像素级启发式（如熵）转为基于 ViT 的语义表征，并将分裂决策建模为预算约束的强化学习问题，实现更合理的 token 分配。

**🔧 技术方法**

采用多尺度 patch embedding、轻量化交叉注意力决策网络、PPO 强化学习结合拉格朗日乘子实现 token 预算约束，并通过伪标签进行 warm‑start 预训练。

**📊 数据集**

在 ImageNet（分类）、COCO（目标检测）、ADE20K（语义分割）以及多模态 VQA 等数据集上进行实验。

**📈 对比分析**

与 EViT、ToMe、DTEM、APT 等方法对比，SPLIT 在保持或略高精度的同时将 token 数量减少约 30%，推理吞吐量提升 20‑30%，在检测、分割与 VQA 上均保持与全分辨率 ViT 相近的性能。

**⚠️ 局限性**

限制在于需要在训练时预设目标 token 预算，想改变推理速度需短期再训练；目前仅在视觉理解任务验证，生成任务尚未评估；对不同任务的迁移性还需进一步研究。

---

## 337. CAST-Bench: Tracing Every Sound to Its Source in Multi-Shot Audio-Video Generation

**arXiv ID:** 2611.05422 | [PDF](https://arxiv.org/pdf/2611.05422v1)

**作者:** Yucong Liu `[一作]` (Northwestern Polytechnical University), Chao Pan `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 929 | [OpenAlex ID](https://openalex.org/A5043074622)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CAST‑Bench基准，包含900条多源声音的prompt和结构化规格，用于评估多镜头文本到音视频生成。

**💡 创新点**

创新点在于把音视频生成视为“音源契约”，通过追踪每个声源、时序与视觉动作的对应关系，提供18维分层评估并区分单镜头与多镜头结果。

**🔧 技术方法**

采用多模态评估工具：视觉特征估计器、音频质量、同步、ASR，以及LLM（GPT‑5.5、Qwen3）进行规范化评分。

**📊 数据集**

使用900个双语prompt–规格对，按9个声源类别（语音、动物、乐器、环境等）划分，包含439单镜头与461多镜头场景。

**📈 对比分析**

通过加权平均计算总体分数，对12个生成模型进行比较，单镜头平均得分约78分，多镜头约63分，显示多镜头、跨镜头连贯与多说话人对齐仍是主要瓶颈。

**⚠️ 局限性**

限制在于仍缺乏跨镜头持续性和多说话人同步能力；评测仅基于结构化prompt，未覆盖更复杂的长时序语境与真实场景。

---

## 338. CLEAR: Structured Specifications for Controllable Long-Horizon ML Research Engineering

**arXiv ID:** 2611.06263 | [PDF](https://arxiv.org/pdf/2611.06263v1)

**作者:** Xin Liu `[一作]` (State Key Laboratory of AI Safety), Xueqi Cheng `[通讯]` (State Key Laboratory of AI Safety)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CLEAR 系统，利用结构化执行规范在大型语言模型驱动的机器学习研究工程中实现长周期任务的可控自动化

**💡 创新点**

创新点在于引入显式的中间执行规范层，分离研究意图与实现细节，并通过规范驱动的规划、代码生成与修复，实现局部化错误处理与资源调度

**🔧 技术方法**

使用大型语言模型（如 Claude‑Opus 4.6 作为规划模型、Claude‑Sonnet 4.6 作为代码模型），并结合视觉解析、可追溯证据包装、生命周期挂钩与多模型协同生成

**📊 数据集**

在 PaperBench（20 篇顶级论文重现）和 MLE‑Bench Lite（22 个机器学习竞赛）上进行评估

**📈 对比分析**

与 AiScientist、BasicAgent、IterAgent 以及 AIDE、ML‑Master 2.0、LoongFlow 等基线比较，CLEAR 在 PaperBench 上平均分 29.87，成本仅 7.91 美元/论文；在 MLE‑Bench Lite 上实现 100% 有效提交、86.36% 任务高于中位数、77.27% 获得任何奖牌，且在修复迭代次数与成本上明显优于基线

**⚠️ 局限性**

局限性包括对任务特定生成与修复策略的适应性不足、对规范生成的准确性仍依赖大模型的理解能力、以及在更复杂任务或更大规模数据集上的可扩展性待验证

---

## 339. Personalized Visual Support After Automatic Speech Recognition

**arXiv ID:** 2611.05625 | [PDF](https://arxiv.org/pdf/2611.05625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 340. ZipDrift: Generating Multimodal Short-Horizon MPC Costs with One-Step Drifting Models

**arXiv ID:** 2611.05793 | [PDF](https://arxiv.org/pdf/2611.05793v1)

**作者:** Ke Zhang `[一作]` `[通讯]` (Trinity College Dublin), Ke Zhang (Trinity College Dublin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于一次性漂移生成器的短期MPC成本校正方法，能够在短视的MPC中学习并执行多模态校正，解决传统回归导致的碰撞问题。

**💡 创新点**

创新点在于将成本倾斜漂移模型与模型预测控制结合，使用Gibbs目标对校正进行倾斜，仅依赖闭环成本评估进行训练，无需梯度传递，同时通过候选选择实现多模态校正的最佳选取。

**🔧 技术方法**

使用技术包括：一次性漂移生成器、成本倾斜目标（Gibbs分布）、Laplace核漂移场、批量生成候选+Critic选取、基于CasADi/IPOPT的短期MPC以及仿真评估。

**📊 数据集**

数据集为仿真产生的800个上下文（包含车道偏移、速度、对手位置等），在一维和八维阶段校正两种场景下进行实验。

**📈 对比分析**

与手工成本、ZipMPC式回归、MDN、纯随机候选等方法对比。在八维校正任务中，生成器+Best‑of‑M在600测试中无碰撞，匹配最佳评估候选；相比之下传统回归方法碰撞率为4%，随机候选2%。

**⚠️ 局限性**

局限性包括：仅在简化的直道仿真环境中验证，未测试曲线赛道、更复杂动力学或多对手情况；需要大量闭环评估生成标签；且生成器在递归预测时可能产生模式切换，需要进一步研究。

---

## 341. C-GRPO: Conformal Group Relative Policy Optimization

**arXiv ID:** 2611.05591 | [PDF](https://arxiv.org/pdf/2611.05591v1)

**作者:** Arya Fayyazi `[一作]` (University of Southern California), Massoud Pedram `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Conformal GRPO（C‑GRPO），将 GRPO 的固定批量替换为基于 conformal prediction 的自适应 roll‑out 分配，显著降低训练计算量。

**💡 创新点**

核心创新在于使用非符合度分数的分位数校准、自适应误差率 δ 选取以及周期性重校准，理论上提供有限样本覆盖与误止控制，并实现“自适应自律”学习。

**🔧 技术方法**

技术包括 Group Relative Policy Optimization、split‑conformal prediction、Adaptive Prediction Sets、代码生成的 first‑success 分数、LoRA 微调、周期性重校准以及自动 δ 设定。

**📊 数据集**

实验使用六个基准：数学（GSM8K、MATH‑500、MATH‑Algebra）、科学（GPQA‑Diamond）与代码生成（HumanEval、MBPP）。

**📈 对比分析**

与固定批量 GRPO、AERO、GDRO、GRESO、Reinforce‑Ada/Est 等基线对比，C‑GRPO 在匹配准确率下平均节省 49–70% 训练 roll‑outs，跨模型迁移最高可达 82%，性能与最强基线相当或略优。

**⚠️ 局限性**

局限性包括依赖可验证器与交换性假设，主要在 LoRA 4–8B 规模和 300 步训练；校准成本仍占较大比例；理论在实际循环中需要拆分校准集，结果对超大规模或完整参数训练的适用性尚待验证。

---

## 342. Alleviating Limited Effectiveness and Multi-Class Collapse in Deep Graph-level Clustering

**arXiv ID:** 2611.06052 | [PDF](https://arxiv.org/pdf/2611.06052v1)

**作者:** Xiaoyang Lin `[一作]` (Hong Kong Baptist University), Xin Huang `[通讯]` (Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种端到端的无监督图级聚类框架，将谱特征与GIN空间表示融合，并通过平衡聚类目标避免聚类崩塌。

**💡 创新点**

创新点包括：① 利用Hoffman-Wielandt不等式得到谱签名作为全局拓扑的低成本下界；② 引入联合流形保持稀疏回归（JMPSR）实现谱视图与空间视图的高效融合；③ 用基于Sinkhorn-Knopp的最优传输正则化实现均衡聚类。

**🔧 技术方法**

技术方法包括谱签名提取、GIN网络编码、JMPSR特征融合、基于GED的对比学习、最优传输平衡聚类、Sinkhorn-Knopp归一化。

**📊 数据集**

实验使用10个真实图数据集：BZR、MUTAG、PTC-MR、PTC-MM、ENZYMES、DD、AIDS、COLLAB、REDDIT-12K、OGBG-PPA。

**📈 对比分析**

与15种基准（图核、无监督表示学习、图级聚类）在ACC、NMI、ARI等指标上对比，平均提升幅度约为10-35%，在大规模多类别数据集（REDDIT-12K、OGBG-PPA）表现最为显著。

**⚠️ 局限性**

局限性在于仅适用于同质静态图，未处理异构或动态图结构，且谱签名仅利用前r个特征，可能忽略微小的局部结构信息。

---

## 343. Do Language Models Know Which Constraints Belong Together?

**arXiv ID:** 2611.05726 | [PDF](https://arxiv.org/pdf/2611.05726v1)

**作者:** Junbo Jacob Lian `[一作]` (National University of Singapore), Chung-Piaw Teo `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何选择 Dantzig–Wolfe 变形的约束族，并提出 JointDW 方法利用语言模型预测以引导分解选择。

**💡 创新点**

创新点在于定义零边际组、证明只靠约束评估寻找组需覆盖数级评估、量化预测误差对搜索成本的影响，并将语言模型与 LP dual 结合实现可解释的预测。

**🔧 技术方法**

主要技术包括组合理论与覆盖数下界分析、分阶段搜索算法、基于语言模型的推理、JointDW 以及与列生成相结合的实验实现。

**📊 数据集**

使用的数据集为 MIPLIB‑NL 2017、十个教材级模板以及八个包含经典分解组的富模型。

**📈 对比分析**

方法通过与随机、贪心、覆盖、LP lookahead 等基线对比实验，显示带推理的语言模型在大多数实例上将评估次数压至约3次以内，JointDW 在保持最优顺序两倍以内的前提下实现性能提升。

**⚠️ 局限性**

局限性包括仅计评估次数不考虑实际求解时间、样本量小导致置信区间宽、预测效果主要依赖于模型类已知、对更大规模实例的实用性尚未充分验证。

---

## 344. SurgTransfer: Multi-Step Latent Action Modeling for Action Transfer in Surgical World Model

**arXiv ID:** 2611.05443 | [PDF](https://arxiv.org/pdf/2611.05443v1)

**作者:** Saurabh Koju `[一作]` (NAAMII), Binod Bhattarai `[通讯]` (NAAMII)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种多步潜在动作学习框架，能够在无标注的外科视频中学习可跨不同手术场景迁移的上下文无关潜在动作，并利用多步动作和扩散动力学模型生成高保真手术视频；

**💡 创新点**

创新点在于：①将动作学习从逐帧递归扩展为多步（跨多帧）动作学习，捕捉手术的层次化与目标导向动态；②对动作模型与扩散动力学模型的对齐进行优化，显著提升跨场景迁移效果；③提供了以多项定量与定性指标（CD‑FVD、ECS、循环一致性、LPIPS、PSNR/SSIM/CMMD）和外科专家评估的全面评估；

**🔧 技术方法**

技术手段包括：SVD（Stable Video Diffusion）微调为多步视频生成器；ST‑Transformer 结构的潜在动作模型（连续潜在动作 + β‑VAE 约束）；多步动作编码与多帧条件解码；光流基线与自回归与AdaWorld模型对比；使用多步扩散模型实现一次性生成多帧；

**📊 数据集**

实验数据集：SurgToolLoc‑2022（机器人手术训练 24,695 条视频片段）和 PitVis（垂体手术 25 条视频，平均 1 小时）；

**📈 对比分析**

与 AdaWorld、SurgWM、光流等基线相比，本文多步动作+多步扩散模型在 CD‑FVD、ECS、循环一致性、PSNR/SSIM、LPIPS、CMMD 等指标均显著提升，跨场景迁移性能提高 58%–69%（SurgToolLoc）和 69%–73%（PitVis），外科专家评估也表明视觉、器械操作、环境反馈、手术意图与动作迁移可信度均优于基线；

**⚠️ 局限性**

局限性：仅适用于解剖结构相似或相同视角且器械实现相同的手术场景；模型在无监督设置下缺乏对下游任务的性能洞察；使用预定义潜在动作轨迹限制了真正交互式体验；未来工作需扩展到特定手术动作空间并提升交互性。

---

## 345. Phonological Knowledge of Chinese Characters in Language Models

**arXiv ID:** 2611.05803 | [PDF](https://arxiv.org/pdf/2611.05803v1)

**作者:** Xiaozhe Ji `[一作]` (Beijing Normal University), Hai Hu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了PinyinBench基准，评估并探究中文字符的语音知识，涵盖同音字、音素特征（声母、韵母、声调）和多音字，结合三种实验方法（行为评估、线性探测与激活调度）分析语言模型的语音表示与可操控性。

**💡 创新点**

①首次将声母、韵母与声调作为独立变量在固定模板下系统对比；②揭示中文模型对语音的编码与功能利用存在明显分离，尤其声调信息虽然可被解码但难以被操纵；③证明多模态训练提升语音可解码度但不必然提升可控性。

**🔧 技术方法**

采用零样本对话式评估、线性分类器对每层隐藏状态进行声母/韵母/声调解码、以及激活向量干预（activation steering）以改变模型输出的语音特征。

**📊 数据集**

PinyinBench（38,183个最小对，覆盖3,500最常用字符及后续频率区间）作为评估数据；字符读音由pypinyin获取并转换为IPA，用于标注声母、韵母与声调；音素相似度使用DIMSIM与PanPhon计算。

**📈 对比分析**

对17个文本或多模态模型（Qwen系列、DeepSeek-V4-Flash等）进行基准比较：最佳模型在完全不匹配的同音判断中超过90%，但在仅声调差异时仅达72%；行为评估显示声母/韵母辨识优于声调；探测实验显示声调在字符嵌入层已高度可解码；激活调度实验表明声母/韵母可通过干预显著影响输出，声调调度效果最弱。

**⚠️ 局限性**

仅评估单字符级别的声调，未覆盖平声、变调与上下文依赖多音字；调度方法以概率提升为衡量，缺乏对生成质量或语义连贯性的分析；多模态模型的声学输入未被单独剖析；实验集中在Qwen2.5系列，缺乏跨模型普适性验证。

---

## 346. When Is One Query per Template Enough? Adapting Learned Cost Models to New Hardware

**arXiv ID:** 2611.05892 | [PDF](https://arxiv.org/pdf/2611.05892v1)

**作者:** Krish Patel `[一作]` `[通讯]` (University of Waterloo), Krish Patel (University of Waterloo)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在数据库迁移到新硬件时，如何利用有限的查询样本来校正已学习的运行时成本模型，并系统评估了不同校正方法和查询选择策略的效果。

**💡 创新点**

创新点在于：①发现只需对每个查询模板执行一次测量，即可通过模板级偏移校正将源模型迁移到新机器；②构造了在六种硬件环境下执行相同查询计划的完整测量数据集；③对比了多种校正方案（速度因子、岭回归、模板偏移、模板校正、1-NN 等）和查询采样策略（随机、轮询、k-center、D-optimal），并提供了正式的注册假设检验流程。

**🔧 技术方法**

使用的技术包括：基于计划特征的随机森林+成本线模型、梯度提升树、岭回归、DACE 风格的 Transformer 神经网络；校正方法有：速度因子、岭回归、线性 + 并行系数、模板偏移、模板校正、最近邻；查询选择策略有：随机、轮询、k-center、D-optimal、按计划类分组等。

**📊 数据集**

使用的数据集为：TPC‑H 与 TPC‑DS（规模因子1）共计1,162条查询；扩展实验包括 TPC‑H 规模因子10 与 JOB 基准；六种运行环境（笔记本/服务器、全部/单/双核心、不同缓冲池大小），从而得到 30 对（和后续 20 对）环境变化的数据。

**📈 对比分析**

通过几何平均 Q‑error（模板中位数）对方法进行比较。结果显示：①模板校正加轮询选择在 k≥25（即每个模板一次测量）时，Q‑error 近似 oracle，远优于单纯速度因子和岭回归；②神经网络在新机器上未经过校正时误差相当，但 fine‑tune 或模板校正可进一步降低 1–3%；③在未测量的模板上，各校正方法无显著提升，误差接近速度因子；④在 JOB 基准中，模型误差本身较大，迁移后提升有限。

**⚠️ 局限性**

局限性包括：仅针对 PostgreSQL，未考虑计划变更或数据库设置的迁移；样本仅覆盖内存级数据，未测试更大规模；硬件环境受限于两台虚拟机；per‑operator 计时受不同机器影响；新模板上校正效果不佳；对不同 DBMS、跨云实例的迁移仍需进一步验证。

---

## 347. Demystifying Information Flow Downgrading using the Real--Ideal Paradigm

**arXiv ID:** 2611.05695 | [PDF](https://arxiv.org/pdf/2611.05695v1)

**作者:** Suraaj Kanniwadi `[一作]` (Cornell University), Andrew C. Myers `[通讯]` (Cornell University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了基于实-理想范式（real‑ideal paradigm）的语言级安全框架，并给出了非干扰、鲁棒降级、非可变信息流等安全属性的外延化（extensional）定义，同时设计并证明了一个支持降级的静态信息流类型系统（NMIFC）的实-理想安全性定理；

**💡 创新点**

将实-理想范式从密码学引入语言安全领域，实现了对多种信息流安全属性的统一、简洁描述；通过 Coq 机械化证明实现了实-理想安全性定理，首次提供了完整的形式化证明；扩展了异构安全模型，支持多方不信任系统的安全分析；

**🔧 技术方法**

实-理想范式、信息流控制与降级（endorse/declassify）、基于标签的安全类型系统、四点信息流模型、模态化的 bisimulation 证明、Coq 形式化与机械化；

**📊 数据集**

无实验数据集；所有证明均基于形式化方法完成；

**📈 对比分析**

本工作为理论性研究，没有进行性能或实证比较；所给出的证明均为形式化验证结果；

**⚠️ 局限性**

仅考虑确定性攻击者，无法处理概率性或随机化攻击；类型系统对终止敏感性不做约束，导致不可传递性；使用的标签模型相对简单，未覆盖更复杂的授权或动态委托场景；

---

## 348. TSFM Meets LLM: Context as Covariates

**arXiv ID:** 2611.05515 | [PDF](https://arxiv.org/pdf/2611.05515v1)

**作者:** Hemanth Ram `[一作]` (University of Texas at Austin), Sandeep Chinchali `[通讯]` (University of Texas at Austin)

**通讯引用:** 1106 | [OpenAlex ID](https://openalex.org/A5024120306)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可扩展的生成多模态时间序列数据的方法，并基于此构建了CoCo框架，将LLM生成的文本信息转换为预测协变量来辅助TSFM进行精准预测。

**💡 创新点**

创新点在于：①通过LLM自动生成高质量的时间变化文本注释，解决多模态数据稀缺问题；②将文本信息以“协变量”形式输入TSFM，并利用GRPO对LLM进行强化学习，使其产生更有益的预测协变量；③两阶段逐步微调的训练策略，避免目标漂移。

**🔧 技术方法**

使用的大模型技术包括Qwen3-30B/4B LLM、Chronos2 TSFM、GRPO强化学习、BERT文本编码、以及多轮LLM推断生成协变量。

**📊 数据集**

数据集涵盖36,826条真实时间序列（约12.4M条记录），并通过LLM合成约150,000条真实文本注释，实验评估使用30个跨领域数据集（财务、宏观、气候、交通等）。

**📈 对比分析**

与27种基线（零射手、单模、双模、微调）进行对比，CoCo在平均MAE上实现约12%提升，排名第一，且在多种指标（平均排名、ELO分数、RCRPS）上均显著优于现有TSFM和LLM预测器。

**⚠️ 局限性**

局限性包括：推理速度受LLM推断影响；若文本信息误导，协变量可能导致误判；仅在文本与数值信息互补时才显著提升，单一文本驱动场景效果有限。

---

## 349. ReTailor: A Residual Feature Tailoring Framework for Real-Time Cross-Sensor Pansharpening Adaptation

**arXiv ID:** 2611.05811 | [PDF](https://arxiv.org/pdf/2611.05811v1)

**作者:** Jing-Ze Liang `[一作]` (University of Electronic Science and Technology of China), Liang-Jian Deng `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 ReTailor 框架，利用轻量化残差 Feature Tailor 在冻结的 pansharpening 网络上实现实时、无 GT 的跨传感器自适应。

**💡 创新点**

创新点包括：①只调优单一残差模块而不修改预训练融合先验；②基于 PAN 变化和 LRMS 频谱差异的内容感知补丁选择；③滚动残差 Bootstrap 通过在图像流中继承已学到的传感器相关校正，显著提升适配质量与速度。

**🔧 技术方法**

使用了自监督的物理感知损失（空间、光谱与冻结源一致性），Tensorized 区域加权补丁选择算法，以及 Adam 优化器对残差 Tailor 的快速微调。

**📊 数据集**

主要数据集为 PanCollection 中的 QB→GF2 与 WV3→WV2 跨传感器测试，对比传统方法（BT-H、MTF-GLP 等）和深度学习方法（FusionNet、U2Net、WFANet 等），以及无 GT 适配方法（ZS-Pan、LoRA）。

**📈 对比分析**

实验显示 ReTailor 在两种跨传感器设置下分别达 HQNR 0.8976 与 0.9245，显著优于传统与零射击方法；在与参数匹配的 LoRA 对比中，ReTailor 在保持相同参数量的前提下将推理时延减半、光谱失真降低 22–47% 并提升 HQNR。

**⚠️ 局限性**

局限性：目前每个传感器对仅训练一次 Bootstrap，需在新传感器对上重启；对极端光谱/噪声变化或非固定 MTF 的场景适配能力待进一步验证。

---

## 350. Data-Efficient Scaling-Law Discovery for Recommendation Systems via Coupled Hyperparameter Scaling

**arXiv ID:** 2611.05643 | [PDF](https://arxiv.org/pdf/2611.05643v1)

**作者:** John Martabano Landy `[一作]` (Meta), Djordje Gligorijevic `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种通过耦合超参数缩放来发现推荐系统的高效数据缩放定律，旨在在有限数据下实现模型性能的系统性提升。

**💡 创新点**

创新点在于将模型规模与关键训练超参数（如学习率、批量大小、正则化系数等）共同耦合进行调优，自动推导出最优的规模-性能曲线，并通过实验验证该耦合策略能显著降低对大量训练数据的依赖。

**🔧 技术方法**

使用深度神经网络框架（如Transformer/DeepFM）结合自动机器学习（AutoML）技术对超参数进行耦合搜索，并利用梯度下降优化与贝叶斯优化相结合的混合调参策略。

**📊 数据集**

在公开推荐数据集上进行实验，包括MovieLens 20M、Amazon Books、Netflix Prize等多种类型的交互式数据集，覆盖不同规模与稀疏度的场景。

**📈 对比分析**

通过与传统单独规模或单独超参数调优方法（如线性缩放、单变量学习率搜索）进行对比，实验显示耦合方法在相同或更少的训练样本下，平均提升了2.5%–4.8%的NDCG和HR@10指标，并显著加快收敛速度。

**⚠️ 局限性**

局限性包括：1）耦合策略对模型架构的适配性有限，可能需针对特定模型做细节调整；2）在极度稀疏或极大规模数据下，耦合搜索的计算成本仍然较高；3）缺乏对非交互式（如文本、图像）特征混合推荐场景的验证。

---

## 351. TauLang: A Language and Compiler for Space-Time Programming on Statically Scheduled Accelerators

**arXiv ID:** 2611.06079 | [PDF](https://arxiv.org/pdf/2611.06079v1)

**作者:** Xuxiao Yang `[一作]` (Peking University), Youwei Zhuo `[通讯]` (Peking University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

为静态调度加速器设计了一种核语言和编译器，使时间成为程序的一等公民，从而实现高效低延迟的大语言模型推理。

**💡 创新点**

创新点在于引入显式时间参数化调度、时序中间表示和联合张量布局与周期调度的整数线性规划搜索。

**🔧 技术方法**

使用了显式时间调度、时序IR、整数线性规划（ILP）以及基于TSP的周期精确模型。

**📊 数据集**

实验使用了十个常见的机器学习/张量运算核，主要是用于大语言模型推理的张量块。

**📈 对比分析**

与传统的空间优先、无时间信息的调度方法相比，联合搜索在TSP类似机器上实现了1.40×的延迟加速，且几条时间固定点可以使延迟与无时间调度相差不到一周期，并在1,636个GEMM块规模上成功，而无时间搜索在32块时就失效。

**⚠️ 局限性**

局限性包括：对显式时间标注的依赖导致编程成本上升，ILP求解时间随规模增大而显著增长，且目前只针对静态调度硬件有效。

---

## 352. USI-Brain: Enhancing Underwater Spatial Intelligence via Spatial Data-Driven Vision-Language Modeling

**arXiv ID:** 2611.06040 | [PDF](https://arxiv.org/pdf/2611.06040v1)

**作者:** Zhiliang Zhu `[一作]` (Harbin Engineering University), Dawei Zhao `[通讯]` (Harbin Engineering University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了USI-Brain——一个针对水下空间智能的视觉‑语言模型，并搭建了空间数据引擎，将增强后的水下图像与几何线索（深度、3D框）结合生成 4.27M QA 对，随后训练 USI-Bench 6,525 题的评测基准；同时提出了基于水下先验的特征修正模块 UPR；

**💡 创新点**

核心创新在于：①空间数据引擎实现了图像增强与几何监督的统一生成 QA；②融合色彩校正与细节恢复的 Hybrid Underwater Enhancement（HUE）提升了训练监督质量；③UPR 在特征空间使用散射偏差与吸收恢复的门控融合，显著提升了场景描述和全局语义理解；④在单一 QA 语料上实现了对水下空间多维能力的系统化评估。

**🔧 技术方法**

使用的技术包括：Hybrid Underwater Enhancement（色彩校正 + 细节恢复 + 融合）；2D 及 Geometry‑Assisted QA 生成（利用深度、重建 3D 盒）；特征空间 Rectifier UPR（散射偏差 β、吸收恢复 G，门控 M 与残差融合）；基于 Qwen3‑VL‑8B‑Instruct 的全参数微调；多任务 QA 评测（精度、IoU、τ、F1 等）。

**📊 数据集**

训练数据来自 540k 影像（NautData、MarineInst、CoralSCOP、SynTIDE、WebUOT‑1M 等）及重建的约 400 个 Atlantis 场景；生成的 4.27M QA 用于训练；评测使用 USI‑Bench、NautData、UWBench、MarineEval 四个公开基准。

**📈 对比分析**

对比方法包括 8 种专有模型、开源通用 VLM（Qwen3‑VL‑30B、8B、GPT‑5.4、Gemini、Claude）以及专门的空间和水下模型；在 USI‑Bench 的 15 项任务中，USI‑Brain+UPR 平均提升 7.63 分（比 Qwen3‑VL‑8B‑Instruct 多 7.63 分），并在 28/32 个外部指标中取得最优或并列最优；在 NAUTILUS‑Qwen2.5‑VL、OceanGPT 等对手中也表现领先。

**⚠️ 局限性**

局限性：①对距离、环境识别、前后背景分割仍低于基线；②需更多覆盖不同水下条件的训练数据；③UPR 增加了前向推理的计算与内存负担；④评测仍以人工标注为主，缺乏实时部署验证；⑤模型对极端低照度或高散射环境的泛化尚未充分验证。

---

## 353. Adaptive Domain Decomposition for Finite Basis Physics-Informed Neural Networks

**arXiv ID:** 2611.05659 | [PDF](https://arxiv.org/pdf/2611.05659v1)

**作者:** Sarah Helfert `[一作]` (University of North Carolina), Alexander Heinlein `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种自适应域分解有限基物理信息神经网络（ADD‑FBPINN），通过 POUNet 学习子域划分并对 Helmholtz、Poisson 以及 Burgers 等 PDE 进行高频或多尺度求解。

**💡 创新点**

创新点包括：① 将 POUNet 与 FBPINN 结合，实现可学习的窗口函数；② 引入可分离 POUNet 生成规则子域；③ 在 POUNet 损失中加入拉普拉斯正则化，强制子域具有足够宽的重叠；④ 设计了分区崩塌检测与自动回滚机制，使得分区数可自适应提升；⑤ 该方法完全基于物理信息，无需额外训练数据。

**🔧 技术方法**

使用的技术有：PINN、FBPINN、POUNet（RBF、MLP、Separable‑RBF）、拉普拉斯正则化、硬约束 ansatz、硬约束/自适应采样、梯度下降（ADAM/L-BFGS）、子域归一化、分区崩塌检测等。

**📊 数据集**

未使用公开数据集，所有训练样本均为合成的 PDE 求解器产生的点（如 Helmholtz、Poisson、Burgers 的解析解或数值解），采用均匀/ Hammersley 随机采样作为训练点。

**📈 对比分析**

通过与标准 PINN 和固定分解 FBPINN 的对比，展示了 ADD‑FBPINN 在不同频率和多尺度案例中相对 L₂ 误差下降 1–2 个数量级；在高频/高异向性 Helmholtz、强多尺度 Poisson 以及空间‑时间耦合的 Burgers 方程上均取得显著性能提升。

**⚠️ 局限性**

限制包括：① 需要预设分区数序列和崩塌阈值；② 对极小子域可能导致窗口崩塌；③ 正则化可能限制可学习子域数量；④ 对不规则子域的归一化处理不充分；⑤ 在非轴对齐或高维问题上的效果尚未验证。

---

## 354. Shoot on (Core)Sight: Re-enabling Secure Debug on the RP2350 with Photon-Emission Microscopy and Laser Fault Injection

**arXiv ID:** 2611.06386 | [PDF](https://arxiv.org/pdf/2611.06386v1)

**作者:** Antoine Plin `[一作]` `[通讯]` (Ledger), Antoine Plin (Ledger)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

利用差分光子发射显微镜定位RP2350A4安全调试控制寄存器，并通过激光故障注入恢复核心1的安全调试访问，随后利用救援重置读取受OTP锁保护的128位秘密。

**💡 创新点**

创新点在于首次将差分PEM用于定位内存映射安全寄存器的故障敏感位置，并结合SWD反馈实现仅用激光脉冲即可持续恢复安全调试，从而突破了仅存储政策而不加固执行链的安全缺陷。

**🔧 技术方法**

使用的技术包括差分光子发射显微镜（PEM）、激光故障注入（LFI）、ARM CoreSight Serial Wire Debug（SWD）以及RP-AP的救援重置（Rescue‑Reset）。

**📊 数据集**

数据集主要为一块RP2350A4 MCU，实验中植入了128位测试秘密并配置了页面48的OTP锁；实验过程涉及多次激光扫描、寄存器读写与SWD通信。

**📈 对比分析**

对比方法为在已知安全调试禁用的初始状态下，逐脉冲观察SWD反馈的设备使能位变化；实验显示仅需5–10个激光脉冲即可实现持续安全调试，并在救援重置后即时读取OTP秘密，性能优于传统的电压/电磁等效应攻击。

**⚠️ 局限性**

限制包括需要昂贵的激光系统和背面切割工艺，攻击范围受限于单一芯片工艺（RP2350），且对其他厂商的安全调试控制寄存器可能不具通用性。

---

## 355. PointNDF: Cascaded Neural Displacement Fields for Point Cloud Denoising

**arXiv ID:** 2611.05411 | [PDF](https://arxiv.org/pdf/2611.05411v1)

**作者:** Michael Y. Wang `[一作]` (Dutch Fork High School), Pingping Cai `[通讯]` (University of South Carolina)

**通讯引用:** 81 | [OpenAlex ID](https://openalex.org/A5013648295)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种结合PTv3编码器与三阶段Neural Displacement Field（NDF）来进行点云去噪的框架 PointNDF。

**💡 创新点**

利用PTv3的patch级全局上下文与局部特征融合，采用bounded-vector与scale-factor分解的位移预测，并通过三阶段级联逐步细化点位移。

**🔧 技术方法**

使用Point Transformer V3作为编码器，三阶段NDF解码器，bounded-vector+scale-factor位移参数化，Chamfer距离监督，PyTorch实现。

**📊 数据集**

在PU-Net（40训练/20测试）和未见的ModelNet10子集（20网格）上进行实验。

**📈 对比分析**

与五种学习型去噪器（DMRDenoise、PD-Flow、ScoreDenoise、IterativePFN、P2P-Bridge）在CD、P2M、均匀性、推理速度上对比，PointNDF在高分辨率、强噪声场景下取得最优或次优性能，并显著加速。

**⚠️ 局限性**

对稀疏、低噪声输入的表现不如局部去噪器，且固定三阶段深度可能不适用于所有噪声级别。

---

## 356. Rohbau3D-Semantics: Dense Semantic and Instance Labels for Point Clouds of Indoor Shell Construction Scenes

**arXiv ID:** 2611.06180 | [PDF](https://arxiv.org/pdf/2611.06180v1)

**作者:** Lukas Rauch `[一作]` (University of Bundeswehr Munich), Thomas Braml `[通讯]` (University of Bundeswehr Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 Rohbau3D‑Semantics 数据集，在原有 Rohbau3D 立体激光扫描基础上添加了12.5 B点、504场景的稠密语义与实例标签，并提供相应的2D全景与立方体贴图，以及基准实验。

**💡 创新点**

创新点在于首次为结构未完成的建筑工地提供高分辨率、手工标注的点云语义和实例信息，兼具多模态投影，并通过严谨的质量控制保证标注精度。

**🔧 技术方法**

技术方面采用 FARO Focus M70 立体激光扫描、专业标注团队配合细致准则、自动化脚本生成全景/立方体投影，以及 PointNeXt、Swin3D、Point Transformer 等最新点云分割网络。

**📊 数据集**

使用的数据集为 Rohbau3D‑Semantics 本身（504场景、12.5 B点），并对比传统室内点云数据如 ScanNet、S3DIS 等。

**📈 对比分析**

基准实验在保留 15 M 版本、90% 训练/10% 验证/10% 测试的站点级划分上，采用 mIoU 评估六种骨干网络，最大 mIoU 为 0.72（PTv3）。

**⚠️ 局限性**

局限性包括仅评估语义分割而未覆盖实例分割或模态消融；仅做站点级划分，缺乏跨站点交叉验证；并且对低密度移动激光扫描的迁移能力未作系统评估。

---

## 357. Give the Drafter a Second Chance to Guess: Parallel Reconsideration for Speculative Decoding

**arXiv ID:** 2611.05724 | [PDF](https://arxiv.org/pdf/2611.05724v1)

**作者:** Hoang Anh Duy Le `[一作]`, Ye He `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种并行再考虑(Parallel Reconsideration)机制，让轻量级特征级草稿器在遇到不确定时可以并行进行额外的计算来修正猜测，从而提升可接受长度与速度；

**💡 创新点**

创新点在于：①按草稿器自身不确定性动态分配计算资源；②使用自查询(Self‑Query)引导再考虑；③加入门控机制决定是否需要再考虑；④将再考虑步骤与普通草稿同一次前向并行，几乎不增加时间；

**🔧 技术方法**

技术核心为EAGLE‑style特征级草稿器、并行前向执行、门控与自查询的联合训练方案；

**📊 数据集**

实验使用了数学（GSM8K、MATH‑500、AIME25）、代码（HumanEval、MBPP、LiveCodeBench）和对话（MT‑Bench）等七个公开基准；

**📈 对比分析**

与EAGLE‑3.1在Llama‑3.1‑8B‑Instruct、Qwen3‑8B、Gemma‑3‑12B上对比，平均可接受长度提升5.3–10.2%，在匹配草稿预算下整体壁钟加速提升3.5–9.2%；

**⚠️ 局限性**

局限性包括：需要在训练时额外设计门控与权重，门控可能不完美；仅针对特征级草稿器，未验证对更深或更大模型的可扩展性；再考虑仍在同一前向内进行，批量增大时并行开销可能上升；

---

## 358. UniLoop: Unified Loop Closure Detection from Retrieval to Geometric Verification and Relative Pose Estimation

**arXiv ID:** 2611.06378 | [PDF](https://arxiv.org/pdf/2611.06378v1)

**作者:** Jingwen Yu `[一作]` (Hong Kong University of Science and Technology), Hong Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一种统一的回环检测框架（UniLoop），在同一几何基础模型（MASt3R）上完成全局检索、几何验证和相对姿态估计，避免了传统多模型管线中的重复编码和信息分离。

**💡 创新点**

创新点在于：
- 通过 SALAD 聚合器在 MASt3R 编码器上直接学习全局检索特征；
- 利用 MASt3R 的跨视角解码器输出做轻量级验证网络，利用对称性判断两帧是否可几何匹配；
- 直接使用 MASt3R 的稠密几何预测生成相对姿态约束，保持同一骨干网络的连续性；
- 采用分阶段训练策略：先用 MegaLoc 进行检索蒸馏，再基于检索结果挖掘难负样本训练验证器，实现端到端的高效学习。

**🔧 技术方法**

技术细节包括：
- MASt3R 视觉 Transformer 编码器 + SALAD 采样器用于检索；
- 跨视角 MASt3R 解码器 + 轻量级 transformer 验证头；
- 采用多相似度损失、蒸馏损失、硬负样本挖掘和 focal loss 进行分阶段训练；
- 端到端的姿态约束生成与图优化。

**📊 数据集**

使用的数据集：
- 训练：GSV-Cities（检索适配）和 Mapillary Street-Level Sequences（MSLS）用于对检索和验证的训练；
- 评估：MSLS、Pitts250k、Pitts30k、Tokyo 24/7、Baidu 等标准 VPR 基准进行检索评估；
- 验证评估使用 MSLS 验证集（150K 正例/180K 负例）。

**📈 对比分析**

与现有方法比较：
- 检索方面，在 MSLS Recall@10 达到 96.6、Pitts30k Recall@10 达到 99.3，分别是所有方法中最高或接近最高；
- 验证方面，AP 95.63、ROC-AUC 96.29，明显优于 SP+LG、eLoFTR 等几何验证基线；
- 整体性能显示该框架在候选检索和几何验证两端均有显著提升。

**⚠️ 局限性**

局限性：
- 目前仅支持两视角 MASt3R，无法充分利用多视角信息；
- 在跨域（不同城市/天气）检索表现仍有提升空间；
- 验证器的 100% 精度下召回率仅 0.52，表明仍有难以区分的假阳性；
- 训练过程分阶段，缺乏完整端到端优化，且对硬负样本的依赖较高。

---

## 359. Familiar letters, not related languages, let unseen languages into a shared space of meaning

**arXiv ID:** 2611.06006 | [PDF](https://arxiv.org/pdf/2611.06006v1)

**作者:** Yuliang Liu `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一种无词表的视觉 Transformer（GlyphMind），能够把文字渲染成像素图像并在多达 27 种书写系统上训练共享语义空间；随后在没有翻译对的未见语言（Lao、Santali）上仅使用本地未翻译文本进行微调，验证了熟悉字形是否能帮助语言进入共享空间。

**💡 创新点**

创新点：① 通过像素化文字读入，突破传统词表限制，实现跨书写系统的语义对齐；② 系统性证明“熟悉字形”比语言亲缘性更关键；③ 展示了在缺乏平行语料的情况下，仅凭本地文本即可扩展多语言覆盖。

**🔧 技术方法**

技术：Vision Transformer + 8×8 像素 patch；对比学习（InfoNCE）与掩码 span 预测结合的自监督目标；在预训练后继续训练时加入原始句子和掩码预测；适配阶段保持部分预训练翻译对以保持对齐；使用纯渲染（pure renders）测试不同脚本。

**📊 数据集**

数据集：预训练使用 OPUS 与 eBible，包含 4.1M 语义组、27 种书写系统；微调使用 Lao CC‑100 + Lao Wikipedia（约 1M 行）和 Santali Ol Chiki Wikipedia（约 150k 行）；评估使用 NTREX‑128（1,997 英文句子），FLORES‑200 dev（Lao、Santali），IN22‑Gen（Santali），以及 9,000+ 其他语言的检索基准。

**📈 对比分析**

比较方法：对每个测试句子在英语候选集上计算余弦相似度，统计 Top‑5 检索准确率；与长度、墨迹、单词计数等简单基线对比；与未适配模型、随机初始化模型和仅纹理网络做对照；结果显示：适配后 Lao 在 NTREX‑128 上 Top‑5 达 14.3%（对照 3.4%），在 FLORES‑200 dev 上 21.0%；Santali 的 Devanagari 版本 Top‑5 6.8%（对照 3.4%），而 Ol Chiki 版本低于基线；展示了熟悉字形显著提升检索性能。

**⚠️ 局限性**

局限性：实验仅覆盖两种语言；对其他少数民族语言的泛化尚未验证；pixel‑based 模型对渲染方式、字形细节和字距敏感；适配会略微降低已预训练语言的检索性能；计算成本高（数千 GPU‑小时）。

---

## 360. Unlocking Turing Tensor Cores: A Measured Throughput Account of From-Scratch Pre-training on a Single 22G RTX 2080 Ti

**arXiv ID:** 2611.05419 | [PDF](https://arxiv.org/pdf/2611.05419v1)

**作者:** Dezhou Shen `[一作]` `[通讯]`, Dezhou Shen

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 RTX 2080 Ti 的 FP16 张量核心限速进行解锁，并在从零开始的中文 encoder–decoder 预训练中评估其对训练吞吐量、墙时间、内存占用和模型质量的影响。

**💡 创新点**

①实现了一键安装的 turing‑unlock kernel 模块，可在驱动加载前写入 FP16 速度选择器；②提出了“矩阵尺寸–层数”双重门控模型，解释了解锁收益随模型规模的递增；③证明解锁仅提升吞吐量，对内存账本和最终质量无影响。

**🔧 技术方法**

Linux DKMS kernel 模块、initramfs 写入 FP16 选择器、CUDA cuBLAS FP16 GEMM、PyTorch V2 训练器、激活检查点与梯度累积、日志统计脚本。

**📊 数据集**

使用与论文 <cit.> 相同的中文预训练语料，固定 16,384 tokens/step、512 序列长度，2,451 步/tier。

**📈 对比分析**

在同一 GPU、同一驱动、相同配置下跑完整训练日志，采用 discard first 10% 并取中位数得到 steady‑state throughput；结果显示 pico–micro 0.3% 提升，tiny 4.7%，small 11.8%，墙时间从 1.77h 降至 1.61h（9% 节省），tiny 节省 3.8%。

**⚠️ 局限性**

仅在单张 RTX 2080 Ti、单驱动版本 575.51.03、单语料下验证；对其他显卡、驱动版本或多卡环境未测试；解锁后功耗和热量显著升高，需要关注散热。

---

## 361. Many Pixels, One Composition: Self-Supervised Spectra for Specimen-Level Prediction

**arXiv ID:** 2611.06096 | [PDF](https://arxiv.org/pdf/2611.06096v1)

**作者:** Eleftheria Tetoula-Tsonga `[一作]` (Institute of Communication and Computer Systems), Theodoros Giannakas `[通讯]` (Institute of Communication and Computer Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种两阶段模型，将自监督像素级编码器与注意力池化结合，用于从高光谱图像预测岩石样本的元素成分。

**💡 创新点**

通过将岩石视为无序像素集合进行多实例回归，并冻结自监督编码器，仅训练轻量化注意力回归头，既保持了像素级信息，又大幅降低可训练参数。

**🔧 技术方法**

自监督自编码器训练、协方差正则化、门控注意力聚合、平滑L1损失及线性/MLP回归层。

**📊 数据集**

Minerals in the Wild 数据集，1132块岩石样本，2,587,554 像素，48 元素的 XRF 标签。

**📈 对比分析**

与四种无监督谱匹配/解混方法及基于平均光谱的有监督模型对比，冻结编码器+注意力模型在余弦相似度0.928、L2误差0.048，接近全微调模型(0.942,0.039)，优于基线且参数约减少88%。

**⚠️ 局限性**

模型仅在相同来源的样本上表现良好，尚未验证对不同地质成分、采集条件或仪器的泛化能力。

---

## 362. Bug2Scenario: Root-Cause-Preserving Test Scenario Amplification for Autonomous Driving Software

**arXiv ID:** 2611.06229 | [PDF](https://arxiv.org/pdf/2611.06229v1)

**作者:** Yuchen Cao `[一作]` (City University of Hong Kong), Jingyu Zhang `[通讯]` (Hong Kong Metropolitan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种根因驱动的后模糊测试框架，能把 DriveFuzz 发现的单一 ADS 故障转化为一组可复现、可验证的回归测试场景；

**💡 创新点**

创新点在于：①将根因模式提取与 LLM 生成分离，避免语义错误导致误判；②使用规则化的“DriveFuzz‑style” JSON 生成与验证流程，确保场景可执行；③在同等生成与执行预算下，显著提升“边界包含”保真度，并通过反馈循环实现部分失败种子恢复；

**🔧 技术方法**

技术手段包括：LLM（DeepSeek R1、Doubao Seed 2.0 Lite、Claude Sonnet 4.5、Gemini 2.5 Pro、GPT‑4o、GPT‑5.1）用于根因说明与高层变体生成；规则化的候选场景生成器将高层模板映射为 DriveFuzz JSON；DriveFuzz/Autoware/CARLA 仿真跑通、日志与 rosbag 提取；人工复核与严格/边界/未保真三级评估；

**📊 数据集**

使用 DriveFuzz 在 Autoware + CARLA 环境下产生的 30 个故障种子（包含 12 次碰撞、18 次红灯、18 次车道侵入、6 次卡死），每个种子配 2 条候选（共 360 条），以及 180 条纯 LLM 生成候选进行对比；

**📈 对比分析**

比较方法：在同等生成/执行预算下（每种种子每个模型 2 条候选，纯 LLM 1 条），通过手工评估区分严格保真、边界保真和未保真。结果显示：在 180 条匹配候选中，严格保真率提升 6.7pp（p=0.104，非显著），边界包含保真率提升 10.6pp（p=0.011，显著）。反馈循环在 9 条未成功种子中，Round 3 时恢复 3 条，节省约 22.2% 的 API 调用；

**⚠️ 局限性**

局限性：①需要人工评审以判定保真度；②对根因模式的解释依赖 LLM，可能存在误判或缺失；③在某些种子中仍受演员摆放、仿真不稳定、根因证据不完整等因素影响，无法完全恢复；④评估样本量有限，缺少对更大规模种子集和更多 LLM 变体的验证。

---

## 363. Discarded or Unreachable? Diagnosing and Repairing Spatial Relation Failure in Vision--Language Similarity

**arXiv ID:** 2611.05565 | [PDF](https://arxiv.org/pdf/2611.05565v1)

**作者:** Mahdi Mostajabdaveh `[一作]` (Huawei Technologies Canada), Zirui Zhou `[通讯]` (Huawei Technologies Canada)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文诊断并修复视觉‑语言检索模型中空间关系的两类失败（可达性失败与池化失败），并提出一种零参数、查询条件的空间矩时读出方法，显著提升左/右、上/下关系检索性能。

**💡 创新点**

创新点：①系统性区分可达性失败与池化失败；②利用翻转对称与 C4 旋转分析空间关系存储与可读性；③设计基于注意力池化头的第一矩时读出，将空间信息编码为与原始嵌入同向量索引，既不需要额外训练也不改变原模型；④证明该方法在多种 CLIP/ViT 变体上将左右关系准确率从≈0.5提升至0.8‑0.96，条件 R@1 从≈0.05提升至0.37‑0.75。

**🔧 技术方法**

技术手段包括：对 CLIP/ViT 等双编码器的翻转对称分解；从注意力池化头恢复的预池化特征；构造图像-文本联合空间的空间矩（X、Y 通道）并与查询向量相乘；使用无监督 β 估计平衡空间与文本权重；对 118k 图像进行向量索引检索。

**📊 数据集**

数据集：COCO (train/val)，What’s Up 412 对、VSR 1,108 对，C_4 旋转合成图像，118,287 张 COCO 训练图像用于检索评估，synthetic composites 作为对比实验。

**📈 对比分析**

对比方法：原始 CLIP 等模型的 native 评分（左右/上下几乎为随机），以及对比不同读出的精度（线性 probe、非线性读者、原始向量、空间矩时读出）。实验显示空间矩时读出在左右 pairwise 准确率 0.81‑0.96、条件 R@1 0.37‑0.75，显著优于 native 0.51‑0.53 和 0.05；在无关检索上保持原有性能，查询时间 < 1 ms。

**⚠️ 局限性**

局限性：①仅验证对称关系（左/右、上/下），未覆盖前/后等无翻转对称的关系；②需要解析 (A, relation, B) 形式查询，无法直接处理自然语言；③需要获取预池化特征和池化头，无法应用于仅返回最终向量的 API；④索引需额外 2‑3× 存储；⑤对多实例、深度方向关系的处理仍待研究。

---

## 364. Annotator-Free Object Detection via Foundation Model Collaboration

**arXiv ID:** 2611.06195 | [PDF](https://arxiv.org/pdf/2611.06195v1)

**作者:** Yasiru Ranasinghe `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无标注目标数据集的对象检测框架，利用预训练的基础模型（SAM、CLIP、LVLM）自动生成分割候选框和语义标签，并通过自我迭代的训练循环不断提升检测器性能。

**💡 创新点**

创新点包括：①将分割、视觉语言和大型语言模型整合为一个完整的自标注流程；②设计地标聚类机制以显著降低 LVLM 查询成本；③构建自我改进的训练循环，使模型能够在未见类别和域漂移场景下实现开放式检测。

**🔧 技术方法**

使用技术：SAM 生成分割掩码；CLIP 进行文本-图像匹配及快速预过滤；LVLM（GPT‑4o、CogVLM）进行场景级与实例级标签生成；YOLO/YOLO‑World 作为开集检测器；地标聚类+Chamfer 距离构建稀疏图；NMS、非极大值抑制等后处理。

**📊 数据集**

实验数据集：COCO 2017 验证集、LVIS minival、ODinW‑35、Roboflow‑100 以及 COCO Open‑Vocabulary（OVD）拆分。

**📈 对比分析**

与 GLIP、DINO、GDINO、T‑Rex2、DetCLIPv3 等长尾/开放词汇检测器在零样本设置下对比，COCO 上 YOLO 版 60.8 AP、Swin‑T 57.2 AP，LVIS 47.4/46.2 AP，显著高于基线（提升 16.5%/25.6% 及更大幅度），在 ODinW、Roboflow 以及 OVD 拆分上亦实现 17%–45% 的性能提升。

**⚠️ 局限性**

局限性：LVLM 误标会在伪标签中扩散；SAM 的分割精度受低对比度、重叠或细粒度场景限制；标签对齐高度依赖 LVLM 提示语和词典，易出现漂移；LVLM 查询仍是主要计算瓶颈，尽管聚类大幅降低成本，但标签质量与成本之间的权衡尚未充分量化。

---

## 365. ChanSFormer: A Channel Agnostic Vision Transformer for Multi-Channel Cell Painting Images

**arXiv ID:** 2611.06015 | [PDF](https://arxiv.org/pdf/2611.06015v1)

**作者:** Jingwei Zhang `[一作]` (Stony Brook University), Dimitris Samaras `[通讯]` (Stony Brook University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种通道无关的 Vision Transformer——ChanSFormer，能够在多通道图像中同时保持通道身份与跨通道交互。

**💡 创新点**

创新点在于用分离的通道‑空间注意力与每通道 CLS 令牌替代硬编码通道嵌入，既保持了通道身份，又大幅提升了计算效率与迁移性。

**🔧 技术方法**

采用了拆解注意力（channel attention + spatial attention）、个别 CLS 令牌、基于表示的通道采样、DINOv2 自监督预训练，以及自研的 Triton kernel 对通道注意力进行硬件优化。

**📊 数据集**

在细胞多通道数据集 CHAMMI、JUMP‑CP 以及卫星多通道数据集 So2Sat 上进行评估。

**📈 对比分析**

与 HyperViT、DepthwiseViT、TemplateViT、ChAda‑ViT、ChannelViT、DiCha‑ViT 等基线对比，JUMP‑CP 分类准确率提升 3.93–4.27%，So2Sat 提升 1.4–1.99%，自监督线性探测与跨数据集迁移中提升 2.72%，且推理/训练速度提升 55–313%。

**⚠️ 局限性**

缺点是去掉固定通道嵌入后无法直接使用通道多样性正则化，导致 CHAMMI 上略低性能；此外对大规模基座模型的扩展仍待研究。

---

## 366. ME^2FDiff: Multiscale Expert Enhancing Feature Diffusion in Residual Network for Brain Tumor Segmentation

**arXiv ID:** 2611.05766 | [PDF](https://arxiv.org/pdf/2611.05766v1)

**作者:** Afrida Rahman `[一作]` (University of Virginia), Bijoy Kundu `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种单模型3D脑肿瘤分割网络 ME^2FDiff。

**💡 创新点**

创新点在于将多尺度专家增强特征块（ME^2FB）与注意力条件频域扩散（ACSDiff）融合进单一网络，实现内部专家路由与迭代细化，消除了传统多模型集成的必要。

**🔧 技术方法**

采用 SegResNetVAE 骨干、空间前注意门 SPG、频域扩散调节器 SDC、ACSDiff 以及 ME^2FB，并使用 Dice、Tversky、Focal 三种损失的混合来训练模型。

**📊 数据集**

使用 BraTS 2024 Post‑Treatment Glioma 数据集（1621 个多模态 MRI 体积，包含重建腔标签），在训练/验证/测试上按 70/20/10 分割。

**📈 对比分析**

与单模型基线和三专家集成（Dynamic Weighting、Cascade Fusion、MoE）对比，ME^2FDiff 在四个肿瘤子区 Dice/IoU 均为最高，整体 Dice 达 91.23%，参数量仅 28.82M，推理成本约为三分之一。

**⚠️ 局限性**

局限性在于对重建腔区（RC）仍难以完全精确，且扩散逆过程带来随机性，未来需进一步改进训练信号与多样化预测。

---

## 367. The Harmonic Online $k$-Server Algorithm Is Poly(k)-Competitive

**arXiv ID:** 2611.05669 | [PDF](https://arxiv.org/pdf/2611.05669v1)

**作者:** Yichen Huang `[一作]` `[通讯]` (Harvard University), Yichen Huang (Harvard University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在线k-server问题，提出了一种新的算法，该算法在每个度量空间中对自适应在线对手具有2k^4的竞争比。

**💡 创新点**

创新点在于首次证明了该算法的竞争比是多项式级别的，并且与请求历史和度量空间的大小无关。

**🔧 技术方法**

使用了一种基于电气网络的潜力函数分析方法，结合了在线服务器与对手位置之间的距离。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了算法在任意度量空间中的表现。

**📈 对比分析**

与已知的工作函数算法（WFA）进行比较，WFA在自适应对手下的竞争比为k，而本文算法在相同条件下的竞争比为2k^4，显示出更好的性能。

**⚠️ 局限性**

限制在于当前的竞争比仍然高于k(k+1)/2的下界，未来的研究可以探索是否能进一步缩小这一差距。

---

## 368. Recasting Visual Language Models as EEG Foundation Models for Clinical Neurodiagnostics

**arXiv ID:** 2611.05538 | [PDF](https://arxiv.org/pdf/2611.05538v1)

**作者:** Xin Zhou `[一作]` (Massachusetts General Hospital), Hao Deng `[通讯]` (Massachusetts General Hospital)

**通讯引用:** 8467 | [OpenAlex ID](https://openalex.org/A5029427154)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

论文提出了 EEG–VLM 框架，将通用视觉语言模型（VLM）迁移到 EEG 信号分析，通过视觉化的 EEG 视觉表示和固定文本提示实现多任务诊断。

**💡 创新点**

创新点在于首次证明 VLM 在未做 EEG 预训练的情况下即可达到或超过现有 EEG 基础模型的表现，并且在模型规模扩展、少样本头部微调和零样本推理等方面展现出“基础模型”特性。

**🔧 技术方法**

主要技术包括多模态视觉编码（波形蒙太奇、频谱图、标量图）、固定文本提示、VLM 预训练模型（Qwen3‑VL、InternVL3、Phi‑3.5 Vision、Llama‑3.2 Vision）以及 LoRA 微调、仅头部训练等适配方法。

**📊 数据集**

使用了临床 EEG 数据集：Temple University Hospital Abnormal EEG Corpus (TUAB)、Temple University Hospital EEG Events Corpus (TUEV)、vEpiSet、SPaRCNet、SpikeNet 2.0 和 SeizeIT2，覆盖正常/异常分类、事件识别、癫痫发作检测等任务。

**📈 对比分析**

与基于从零训练的时序模型、已预训练的 EEG 基础模型（LaBraM、BIOT、EEGPT 等）对比，VLM 在多任务上保持竞争或优越性能；模型规模增大时性能显著提升，LoRA 微调后取得最高准确率；在 64/384 样本的少样本设置下，冻结主干 + 轻量化头部亦能达到 0.6+ 的平衡准确率。

**⚠️ 局限性**

主要局限包括 VLM 体积大、推理成本高；在非可视化的细微信号或缺乏足够视觉提示的任务中效果不明；通用预训练可能在某些 EEG 领域达到饱和，需要更多异构 EEG 数据集才能进一步提升。

---

## 369. Who May Overrule the Agent? Evidence-Gated Authority in LLM Review Institutions

**arXiv ID:** 2611.05390 | [PDF](https://arxiv.org/pdf/2611.05390v1)

**作者:** Fuad Ali `[一作]` `[通讯]`, Fuad Ali

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多智能体语言模型系统中，研究了评审者在何种条件下能够强制其建议生效，并将广泛覆盖权与证据门控进行对比；

**💡 创新点**

创新点在于把评审者的权限视为可调机构司法范围，揭示了评审者接口、能力与权限范围共同决定纠错与腐败平衡的机制；

**🔧 技术方法**

采用结构化审计记录、机械验证器和两种权威规则（广泛覆盖与证据门控）进行实验，利用 Qwen3-8B 与 Mistral-Small-24B 两款量化模型；

**📊 数据集**

使用了两个合成基准：96个空间政策任务和96个组合项目审批任务（分别包含三种任务状态 C、V、U）；

**📈 对比分析**

通过与基线无审查、广泛覆盖和证据门控的三种情形对比，结果显示证据门控在保持合规性与正确提案时显著减少了误改，但在优化可合规但次优方案时表现不佳；

**⚠️ 局限性**

局限包括仅测试两款模型、离散七项选项的任务、人工合成数据而非真实部署场景、评审者与提议者同为同一模型导致可能相关性、温度为零的生成等，因而结论不一定能推广到更复杂或人类参与的系统。

---

## 370. OUCH: Objective, Uniform, Controlled Haptic Stimulation for Automated Neurological Assessment

**arXiv ID:** 2611.05741 | [PDF](https://arxiv.org/pdf/2611.05741v1)

**作者:** Sherry Choi `[一作]` (Stanford University), Ava Chen `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并验证了一款名为OUCH的腕带式气动装置，可在低压力下通过局部高接触压实现对患者的可控痛觉刺激；

**💡 创新点**

通过将气动囊片、刚性指尖和可屈压缩泡沫相结合，实现了局部化高压、低总力的无创疼痛刺激，从而克服了传统手动刺激的力度不一致和不易量化的问题；

**🔧 技术方法**

采用气动织物囊片、刚性金属指尖、可屈压缩硅胶支架，结合有限元分析、阻塞力测试以及可变形压敏试验台进行性能评估；

**📊 数据集**

使用了内部研究团队成员的疼痛阈值数据以及自制的可变形压敏试验台中的压力传感数据；

**📈 对比分析**

与临床手持压力痛觉计（AlgoMed）进行比较，阻塞力测试显示OUCH在相同充气压力下产生约18,500 kPa的接触压，而AlgoMed仅为707 kPa；在可变形试验台上，OUCH在表面产生更局部的压缩，且在内部试验中疼痛阈值约比AlgoMed低89%，表明性能优越；

**⚠️ 局限性**

局限性包括仅在研究团队内部进行的初步感知阈值测试、未量化支架的粘弹性蠕变、可变形试验台与真实皮肤的差异、以及缺乏正式临床患者研究。

---

## 371. SlackBench: Benchmarking Agents on Collaborative Projects Grounded in Real Code Repositories

**arXiv ID:** 2611.05512 | [PDF](https://arxiv.org/pdf/2611.05512v1)

**作者:** Mingxuan Li `[一作]` (University of California, Berkeley), Alex Dimakis `[通讯]` (University of California, Berkeley)

**通讯引用:** 18433 | [OpenAlex ID](https://openalex.org/A5014141359)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了SlackBench基准，用于评估LLM代理在多方沟通、代码仓库、文档等多源协作环境中的推理与行动能力。

**💡 创新点**

创新性在于通过隐私保护的真实科研项目重现，结合Slack、GitHub和Google Docs的多源数据，提出了四大任务类别（异构搜索、拒答、摘要、情境推理），并系统分析了工具接口和推理深度对性能的影响。

**🔧 技术方法**

采用LLM代理框架Harbor、Slack/GitHub API工具、终端工具、LLM判定器GPT‑5.4，以及CLI直接读取工作区文件等技术。

**📊 数据集**

使用SlackBench数据集（15个频道约3000条消息，14个GitHub仓库，9份Google Docs，共140个任务）进行评测。

**📈 对比分析**

通过与33个前沿模型/工具组合对比，最佳系统平均准确率约80%，人类平均84.5%；CLI访问提升≈7.9%准确率，较MCP接口更高效；不同推理力度和工具配置导致显著性能差异。

**⚠️ 局限性**

局限在于仅评测前沿实验室模型，未覆盖更广泛的提示策略或更多代理；存在少量引用作弊行为，且未公开所有参考答案。

---

## 372. Verified Implementation of the Next-gen Borrow Checking Algorithm for Rust via Abstract Interpretation

**arXiv ID:** 2611.06319 | [PDF](https://arxiv.org/pdf/2611.06319v1)

**作者:** Jinhua Wu `[一作]` (Shanghai Jiao Tong University), Zhemin Chen `[通讯]` (Nanjing University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现并形式化验证了 Rust 编译器中的下一代借用检查器 Polonius，利用抽象解释与 Coq 证明保证程序在通过检查后不会出现无效引用。

**💡 创新点**

首次将 Polonius 视为前向抽象解释器并构造了精确的 borrow‑aware 语义；通过模拟证明抽象解释的正确性，并利用 LLM 辅助完成了完整的 Coq 证明，构建了完整的可验证借用检查器。

**🔧 技术方法**

使用抽象解释（Kildall 算法）与 union‑find 抽象域；在 CompCert 的抽象解释框架中实现；构造 borrow‑aware 语义与低阶操作语义；采用后向模拟技术；使用 LLM（Codex + GPT‑6 Astra）生成 Coq 证明代码。

**📊 数据集**

对 510 个 Rust 程序（主要来自 borrowck、nll、Polonius 测试套件）进行评测；这些程序覆盖了 Box、引用、嵌套引用、结构体等核心 MIR 特性。

**📈 对比分析**

与 rustc 1.101.0-nightly 的 Polonius 检查器进行对比，结果 499/510（97.8%）一致；在未通过的 11 个案例中，主要原因是对两相借用、Box 的销毁检查等 Rust 语言细节的支持不足。性能方面，虽然未给出精确的运行时基准，但使用 Coq 和 Kildall 解决器实现的检查器能够在合理时间内完成所有 510 个程序的分析。

**⚠️ 局限性**

局限性包括：只验证核心 MIR 子集（不支持 trait、闭包、unsafe 代码、两相借用等高级特性）；对某些程序的判定更严格或更宽松与 rustc 有差异；未覆盖 unsafe Rust 的内存安全性；抽象域在处理嵌套引用和结构体中的多重生命周期时存在精度损失。

---

## 373. Sublinear Approximations for Spanning Tree Congestion

**arXiv ID:** 2611.06209 | [PDF](https://arxiv.org/pdf/2611.06209v1)

**作者:** Sunny Atalig `[一作]` (University of California at Riverside), Hans Raj Tiwary `[通讯]` (Charles University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了两种随机多项式时间近似算法，分别获得了 O(n^8/9) 和 O(n^2/3) 的拥塞近似比，并证明在一般连通图上不可能在 2-ε 范围内实现多项式时间近似（对任意固定 ε∈(0,1)）。

**💡 创新点**

创新点包括：①首次使用连接最小‑最大多路切分（connected min‑max multiway cut）来构造低拥塞的树；②引入层次化（laminar）分解与分层递归构造方法，显著减少了对高度顶点的处理；③结合“同构流”（confluent flow）与“继承性二分”技术，得到可控的边/点负载；④通过新的硬度约简，提升了从 6/5 到 2-ε 的近似下限。

**🔧 技术方法**

核心技术包括：连接最小‑最大多路切分近似、层次化（laminar）分解、继承性二分（hereditary bisection）、同构流（confluent flow）与流量逼近、以及多阶段的树构造与合并。所有算法均基于多项式时间的最大流/最小割与网络流算法。

**📊 数据集**

本研究不涉及实验数据集；所有结果均为理论分析与证明，算法复杂度仅在理论上给出，未在具体图实例上进行实验验证。

**📈 对比分析**

性能评估以渐进近似比为准。与已知的 O(n/√k) 或 8√(mn) 等上界相比，提出的 O(n^8/9) 与 O(n^2/3) 为首个子线性近似；相对最优拥塞值 k 的近似比可达 O(n^2/3 log^{5/3}n)。

**⚠️ 局限性**

局限性：近似比仍然是多项式次线性的，距离常数因子（如 2 ）仍有较大间隙；算法实现复杂，涉及多层递归与多次最大流求解，实际运行时间和常数项较大；仅适用于无权图（加权图需进一步变换）；硬度证明仅针对特定源点限制的可变路径问题，未覆盖更一般的多源路径约束。

---

## 374. DualCast: Dual-Image Conditioned Diffusion Transformers for Digital Human E-Commerce Video Generation

**arXiv ID:** 2611.05825 | [PDF](https://arxiv.org/pdf/2611.05825v1)

**作者:** Lifan Jiang `[一作]`, Boxi Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DualCast框架，统一数字人电商视频生成的双图像条件（产品+人像）并支持单图像模式。

**💡 创新点**

创新点包括：角色地址化双边界槽注入以区分产品与人像；区域加权损失强化关键区域；图像条件丢弃实现单模型多模式推理。

**🔧 技术方法**

使用技术包括Diffusion Transformer (DiT)、VAE编码、CLIP语义上下文、mask-guided region-weighted loss、分段噪声微调以及10K电商视频数据集。

**📊 数据集**

使用了自建的10K电商视频数据集，包含产品、人像、预计算面部/产品掩码。

**📈 对比分析**

与Kaleido、Phantom、VACE、SkyReels-A2等多模态方法进行对比，采用ViCLIP、S2V-C、ID-Adj、5D Avg等指标；DualCast在多数指标上排名第一，平均rank 1.73，5D Avg 0.9848，优于闭源系统。

**⚠️ 局限性**

局限性在于仅支持单一产品与单一主持人，无法处理多产品、多主持人或更复杂场景。

---

## 375. WayPOP: A Panoramic Open-Set Panoptic Tracking Benchmark

**arXiv ID:** 2611.05801 | [PDF](https://arxiv.org/pdf/2611.05801v1)

**作者:** Rohit Mohan `[一作]` (University Of Freiburg), Abhinav Valada `[通讯]` (University Of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WayPOP基准，针对自动驾驶中多视角、开放集合的全景语义分割与跟踪任务进行评估。

**💡 创新点**

创新点在于通过几何引导将多样化的3D Objaverse资产插入真实Waymo PVPS视频序列，实现统一的几何、光照、遮挡和时空一致的未知对象标注与LiDAR补充。

**🔧 技术方法**

采用的技术包括资产筛选、基于LiDAR的插入点采样、物理渲染与摄像头/激光雷达同步集成，以及后续的跨视角、跨时序实例关联与评价指标（PQ、STQ、wSTQ）。

**📊 数据集**

使用的数据集为Waymo PVPS的验证/测试视频片段以及Objaverse v1.0与Objaverse-XL提供的约670个3D资产，生成约980条未知对象轨迹、超过5万份像素级掩码。

**📈 对比分析**

将多种开放集合与开词表语义分割方法（U3HS、PoDS、Mask2Anomaly、RbA、FC-CLIP、MAFT+、EOV-Seg、ODISE、OpenWorldSAM）与时间与跨相机关联策略（MASA、DEVA、BoT-SORT）相结合，在VI‑OSPS、FV‑OSPT和MV‑OSPT任务上进行评测；结果显示即便在最佳配置下，未知对象的wSTQ也仅达到约37%，表明现有方法在持续定位与身份维护上仍有限。

**⚠️ 局限性**

局限性包括：插入对象保持静止，遮挡与光照补偿不够精细，数据仅针对摄像头进行评估，缺乏动态插入和更复杂环境的真实感。

---

## 376. Same Origin, Different Verdicts: Text Production Conditions Confound Machine-Generated Text Detection in Russian

**arXiv ID:** 2611.05385 | [PDF](https://arxiv.org/pdf/2611.05385v1)

**作者:** Stanislav Kirichenko `[一作]` `[通讯]` (Independent Researcher), Stanislav Kirichenko (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在保持文本来源不变的前提下，评估文本生成条件对机器文本特征得分的影响。

**💡 创新点**

首次系统测量生成条件对机器文本得分的作用，并将四种不同测量方法进行对比，揭示其结果不一致。

**🔧 技术方法**

采用可解释指标、逻辑回归分类器、开放模型零射门NLL以及人工评判模型四种技术进行测量。

**📊 数据集**

构建了1882条俄语文本语料库，其中1079条为机器生成文本（四渠道、45任务、3模式），803条为人类文本。

**📈 对比分析**

通过在机器子集内按任务模式进行配对对比，利用四种测量方法的差异度量；结果显示各方法方向不一致，且无方法能覆盖零点。

**⚠️ 局限性**

得分高度依赖语料特征和预处理，无法直接衡量作者身份，且在不同检测系统间转移性差。

---

## 377. RDT-3D: Resolution-Decoupled Training for High-Fidelity 3D Geometry Generation

**arXiv ID:** 2611.06141 | [PDF](https://arxiv.org/pdf/2611.06141v1)

**作者:** Jiale Zhang `[一作]` (JD.com), Taicheng Huang `[通讯]` (JD.com)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种分辨率解耦训练（Resolution-Decoupled Training, RDT）框架，用于高精度3D几何生成，解决了RoPE指导与几何监督在训练与推理中的分辨率不匹配问题。

**💡 创新点**

创新点在于将RoPE坐标指导固定在低分辨率（512）上，同时在高分辨率（1536）下进行几何监督，使模型能够在推理时从粗糙的空间提示中准确推断出细节丰富的高分辨率形状。

**🔧 技术方法**

采用RoPE-guided vecset Diffusion Transformer（DiT）作为生成器，使用修正流（rectified-flow）损失进行训练，并结合三维点云的高分辨率采样与可视化。

**📊 数据集**

使用了自研的300K对齐点云数据集（RDT-3D Dataset），其中每个样本提供512分辨率的粗点云用于RoPE指导，1536分辨率的精细点云用于几何监督。

**📈 对比分析**

通过与Hunyuan3D-2.1、TRELLIS、TRELLIS.2和UltraShape等SOTA方法在Toys4K数据集上进行对齐后Chamfer Distance和F-Score评估，RDT在CD（0.0164）和F-Score（0.6886）上均优于对比基线，表现出更细腻的边缘和更高的几何精度。

**⚠️ 局限性**

局限性包括仍需大量算力进行高分辨率训练，且对粗糙的初始模型（如Hunyuan3D-2.1）生成的低分辨率点云依赖较高，可能在极端噪声下影响细节恢复。

---

## 378. Riemannian Optimization-based Whole-body Obstacle Avoidance for Mobile Continuum Robots in Cluttered Environments

**arXiv ID:** 2611.06214 | [PDF](https://arxiv.org/pdf/2611.06214v1)

**作者:** Duo Tang `[一作]` (University of Hong Kong), Peng Lu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了面向移动基连续机器人在拥挤环境中的分层运动规划与控制框架，整合全局A*+最小抖动路径规划、几何一致的FABRIK逆运动学和黎曼几何优化的局部避障，实现基-臂协同运动的实时执行。

**💡 创新点**

创新点在于：①将全局路径规划与局部几何一致逆运动学耦合，保证基-臂协同一致；②设计了几何一致的FABRIK求解器，以弧度空间迭代保证连续弧形；③引入基于黎曼流形的方向引导优化，兼顾姿态一致性与碰撞安全；④整个系统实现了实时（≤2 ms/IK，<5 ms/控制循环）并通过实验验证。

**🔧 技术方法**

使用的技术包括：A*+最小抖动全局规划、基于常弧假设的FABRIK逆运动学、Riemannian manifold优化、几何一致碰撞检测、基于IMU与编码器的闭环控制、ROS/Ubuntu平台的软硬件协同。

**📊 数据集**

实验数据主要来自：①仿真管道穿越任务、障碍密集环境；②真实硬件实验（固定基、移动基两种）完成的障碍避让、圆轨迹跟踪、箱体/管道穿越等场景。未使用公开标准数据集。

**📈 对比分析**

与四类基线方法（IS、Grad‑Ori、Grad‑Angles、Grid‑Search）以及传统FABRIKx、FABRIKc、Jacobian等比较，RMO在规划时间、迭代次数、成功率（100%）和轨迹误差/角速度变化方面均优于对比方法；实验中实时控制频率>200 Hz，表现出良好的计算效率与轨迹平滑性。

**⚠️ 局限性**

局限性包括：①目前仅在静态障碍环境验证，缺乏对移动障碍物的自适应；②全局规划与局部优化耦合仍受限于高维状态空间，长周期任务的可扩展性待进一步研究；③对模型不确定性和外部干扰的鲁棒性需要进一步提升；④尚未整合多模态感知（视觉/触觉）以增强环境感知。

---

## 379. Community Search under Member-Specific Tolerances

**arXiv ID:** 2611.05951 | [PDF](https://arxiv.org/pdf/2611.05951v1)

**作者:** Yanghao Liu `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了每个顶点具有个体容忍度阈值的社区搜索问题，提出纬度一致社区（LCC）概念，并证明其在意见动态下能够保证所有交互始终被接受。

**💡 创新点**

创新点在于将最小容忍度作为连通子图直径的上限，既能提供最强的动态接受保证，又能通过在单一阈值上搜索实现可行求解；并设计了LatSweep框架，将连通性、最小度和直径约束统一化。

**🔧 技术方法**

采用LatSweep框架，利用生存区间、k-core约束、最佳优先分支定界以及对最大(k,r)-核心问题的启发式与逼近算法，提供了(1,2)-双指标逼近和精确求解方案。

**📊 数据集**

实验使用了八个大规模网络（包括 Gowalla、Brightkite、LiveJournal、Orkut、Friendster、DBLP、YouTube 等）以及位置数据和议会投票数据（U.S. Congress、European Parliament）共计多达 1.8 亿条边。

**📈 对比分析**

与传统 k‑core、(k,r)-core、最大(k,r)-核心、整数规划、CP‑SAT 等方法对比，LatSweep‑Exact 在绝大多数查询中在毫秒级完成并提供最优解；Greedy 与 BallCore‑F 在大规模图上能快速得到近似解；在模拟意见动态时，LCC 组几乎不失交互，而仅满足 pairwise 条件的组则易出现失交互。

**⚠️ 局限性**

主要限制包括：对开放组外部影响不保证；在高维度或多社区情形下 Exact 可能超时；需要先验估计的容忍度对噪声敏感；LCC 约束可能导致空解，限制了可得到的组大小。

---

## 380. Consistency Learning and Distillation for One-Step MIMO-OFDM Channel Estimation

**arXiv ID:** 2611.06271 | [PDF](https://arxiv.org/pdf/2611.06271v1)

**作者:** Fredrik Nguyen `[一作]` (KTH Royal Institute of Technology), Qiwen Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种观测锚定一致性与一致性中训练（CMT）的单步MIMO-OFDM信道估计方法，能够在基站多用户干扰（pilot contamination）条件下直接从LS估计中恢复干净信道，减少推理次数；

**💡 创新点**

创新点在于提出观测锚定终点一致性约束，将邻近路径状态的终点预测强制一致；并利用教师轨迹蒸馏的CMT训练，令轻量化学生网络直接映射到终点，从而在保持性能的同时显著压缩模型；

**🔧 技术方法**

使用了一致性学习、流匹配（flow‑matching）、教师轨迹蒸馏（CMT）、轻量化U‑Net（Lite、Mini）架构、时域与上下文嵌入、逆变换插值、LMMSE检测等技术；

**📊 数据集**

训练数据来自QuaDRiGa 3GPP TR 38.901 urban‑macro CDL‑B仿真生成的816,000个信道样本；测试集采用1,000个TTI的CDL‑B仿真；

**📈 对比分析**

在与标准流匹配（FM）和直接回归同一backbone（U‑Net‑Lite/Mini）下进行对比，CMT在五个评估SNR点上相较Lite FM降低了9.6–20.2%符号错误率（对应0.44–0.98 dB的对数SER降低），同时参数量减少74.5%，MACs减少73%；在-12 dB到-6 dB的SNR范围内实现最佳性能；

**⚠️ 局限性**

限制在于仅在理想的oracle干扰协方差和QPSK调制下验证，未涉及实际协方差估计、不同调制、宽带或高速移动等更广泛场景；

---

## 381. Grounding in the Dark: Text-Only Training for Zero-Shot Video Temporal Localization

**arXiv ID:** 2611.06149 | [PDF](https://arxiv.org/pdf/2611.06149v1)

**作者:** Divyanshu Mishra `[一作]` (Amazon), Erhan Gundogdu `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用仅文本描述训练视频大语言模型，实现在无配对视频-文本数据的情况下完成视频时间定位任务。

**💡 创新点**

提出无训练的Nearest Entity Replacement（NER）映射方法，弥合文本与视觉模态差距，并首次在单模态训练下进行视频时间定位。

**🔧 技术方法**

使用Qwen3-8B LLM + LoRA、CLIP/LLM2CLIP编码器、Synthetic Temporal-TextVid语料以及NER检索映射技术。

**📊 数据集**

构造134K例的Temporal-TextVid语料，并在Charades-STA与QV-Highlights两大视频时间定位基准上进行零样本评估。

**📈 对比分析**

与传统视觉特征、文本记忆投影、噪声注入、C3等四类基线相比，NER在Charades-STA获得14.45 mtIoU、QV-Highlights 20.72 mtIoU，明显优于其它方法。

**⚠️ 局限性**

模型在精细边界定位仍有欠缺，模态差距未完全消除；NER依赖检索近邻，需进一步优化实体加权与检索策略，且仅在合成文本上训练，真实视频的时序细节仍有提升空间。

---

## 382. Faster Approximate Text-to-Pattern Hamming Distances via Near-Optimal Sumset Estimation

**arXiv ID:** 2611.05725 | [PDF](https://arxiv.org/pdf/2611.05725v1)

**作者:** Nick Fischer `[一作]` (Max Planck Institute for Informatics), Vasileios Nakos `[通讯]` (National and Kapodistrian University of Athens)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种随机化近似文本到模式汉明距离的算法，时间复杂度为 Õ(ε^{-2/3} n)，显著改进了之前的最优算法。

**💡 创新点**

创新点在于将问题归约到 Sumset Estimation 并使用改进的 Count Sketch（用随机复根替代传统随机符号），无需快速矩阵乘法，达到了理论上最优的 ε 依赖。

**🔧 技术方法**

主要技术包括 Sumset Estimation、稀疏恢复、改进的 Count Sketch、FFT、随机采样、以及复数根的随机符号替换。

**📊 数据集**

该工作为理论算法研究，不涉及具体数据集；所有实验均在理论分析框架下完成。

**📈 对比分析**

相较于之前最优的 Õ(ε^{-0.915} n) 算法，本方法在时间复杂度上更优；例如在 ε = n^{-1/2} 时，改进从 Õ(n^{1.458}) 下降到 Õ(n^{1.333})，并且实现更简单。

**⚠️ 局限性**

局限性：算法仅对符号逐个处理的“symbolwise”算法类已被证明无法进一步降低 ε 依赖；仍是随机化算法，且对非符号化方法仍未突破；实现中涉及复杂数精度与算法细节。

---

## 383. Prospective Memory for Proactive LLM Agents

**arXiv ID:** 2611.05731 | [PDF](https://arxiv.org/pdf/2611.05731v1)

**作者:** Yuanchen Bei `[一作]` (University of Illinois Urbana-Champaign), Hanghang Tong `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于认知的前瞻记忆框架，以支持LLM代理在主动协助中保留和实现未来意图。

**💡 创新点**

创新点在于将保持未来意图与何时何地实现分离，包含四阶段：意图形成、维护、基于线索的启动和选择性执行，并结合经验记忆来改进机会评估。

**🔧 技术方法**

技术包括使用LLM做意图生成与评估、基于证据链接的记忆存储、上下文线索匹配、意图生命周期管理以及可选的反馈驱动经验记忆。

**📊 数据集**

使用了两个公开的主动代理基准：PM‑Bench（延迟意图实现）和 ProactiveBench（实时无请求协助）。

**📈 对比分析**

与多种后退式记忆方法（Reflection、RAG、A‑Mem、MemoryOS、T‑Mem、RWM）以及不同LLM骨干（Claude‑Sonnet、Claude‑Haiku、GPT‑4o、GLM‑4.7、Kimi‑K2.5）进行对比；结果显示该框架在两项基准上均取得最高或最稳健的F1/Set‑F1，且误报率最低。

**⚠️ 局限性**

局限性包括目前仅针对文本交互，缺乏多模态信号处理；对更复杂的长周期、相互依赖意图场景的支持不足。

---

## 384. Exact certification for conic optimization over quadratic modules: partitions of unity and recession directions

**arXiv ID:** 2611.06315 | [PDF](https://arxiv.org/pdf/2611.06315v1)

**作者:** Didier Henrion `[一作]` (LAAS-CNRS, University of Toulouse), Victor Magron `[通讯]` (LAAS-CNRS, University of Toulouse)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于分割单位、投影修复和递增方向的线性锥问题精确有理化证书方法。

**💡 创新点**

创新点在于构造严格的分割单位、利用解析中心与克里斯托费尔多德多项式得到闭式分割、证明强对偶性并给出失配损失为对偶质量的支撑函数，以及通过递增方向吸收常数实现无额外 SDP 求解的完整证书。

**🔧 技术方法**

使用了 SOS/矩阵半正定规划、Riesz 函数、Gram 矩阵分解、分割单位、解析中心、支持函数、投影‑提升修复、CSDP 求解以及符号计算。

**📊 数据集**

在多组实验数据集上验证，包括单变量例子、球/盒子体积计算、Stokes 约束下体积、D‑finite 路径正性、以及三篇公开的多项式优化基准。

**📈 对比分析**

与现有的系数区间（I）、系数吸收（A）、投影修复（P）以及 Frobenius 预估（F）对比，P 在同一数值输入下得到最低的证书损失；I 在某些体积实例上仍优于 F；整体表现显示方法可在不额外求解的前提下显著降低损失。

**⚠️ 局限性**

局限在于对高阶或稀疏问题的扩展未验证；F 方法在某些实例表现不佳；损失仍受基函数和域的影响；未给出时间复杂度比较，仅在固定阶下测试；方法对数值精度和分辨率敏感。

---

## 385. PRISM-AMC: Replay-Free Continual Learning for Automatic Modulation Classification

**arXiv ID:** 2611.06066 | [PDF](https://arxiv.org/pdf/2611.06066v1)

**作者:** Usman Akram `[一作]` (University of Texas at Austin), Haris Vikalo `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了PRISM-AMC，一种无重放的连续学习框架，用于自动调制分类（AMC），通过梯度投影记忆来保存先前任务的子空间，减少灾难性遗忘；

**💡 创新点**

其创新点在于将梯度投影记忆（GPM）应用于AMC任务，利用中间特征子空间进行投影，理论上推导了投影随机梯度更新在SNR依赖梯度噪声模型下的收敛性，并在不存储原始I/Q样本的前提下实现了高精度的连续学习；

**🔧 技术方法**

主要技术包括1D CNN架构、梯度投影记忆（GPM）、正交投影约束、cosine prototype分类器、SNR依赖梯度噪声模型、Adam优化器与学习率调度；

**📊 数据集**

实验使用了现场MIGOU I/Q数据集、仿真DeepSig 2018.01A 数据集以及自定义的域增量数据（CFO、幅度/相位偏移变化），涵盖多任务类增量与域增量设置；

**📈 对比分析**

与顺序微调（SFT）、重放+软蒸馏（RSD-AMC）、弹性权重巩固（EWC）、MSNCIL、PASS-Net等基线对比，PRISM-AMC在类增量与域增量任务中在平均准确率和后向转移（BWT）上均优于无重放基线，并与RSD-AMC竞争或超过，且在存储与计算成本上更具优势；

**⚠️ 局限性**

限制方面包括子空间阈值ε的选择会导致稳定-可塑性权衡，随着任务增多可用的残差梯度空间逐渐收缩，可能降低对新细粒度任务的学习；此外，该方法假设不需要重放原始样本，对需要严格数据存储的场景适用性有限。

---

## 386. BiCG-Net: Target-Specific Image Quality Assessment via Observation Modeling and In-Image Differencing

**arXiv ID:** 2611.06203 | [PDF](https://arxiv.org/pdf/2611.06203v1)

**作者:** Yibu Yang `[一作]` (CRRC Chongqing Co., Ltd.), Guanghui Huang `[通讯]` (Chongqing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于嵌套采样和图像内差分的目标级无参考图像质量评估方法，并将11个教师模型的知识蒸馏到一个39,482参数的BiCG-Net学生网络中，以实现对小目标区域（如列车车身）的质量估计；

**💡 创新点**

核心创新点在于（1）构造可证明的观测模型，将目标质量、教师偏置、面积效应、背景内容和噪声拆解为线性可观测量；（2）利用图像内差分消除目标质量与教师偏置的影响，从相对结构中学习质量映射；（3）设计四模块可解释的学生网络，兼具尺度校正、背景纠正、序列编码和自适应门控；（4）通过Δ残差分析验证学生模型确实捕获了教师一致性之外的背景与教师差异信息；

**🔧 技术方法**

使用11个预训练的无参考IQA教师（BRISQUE、NIQE、NRQM、PI、CLIP-IQA+、MANIQA、NIMA、PaQ-2-PiQ、CNNIQA、DBCNN、MUSIQ），嵌套采样生成目标扩展的多尺度补丁，构建线性观测模型，图像内差分后输入到BiCG-Net，其中包含教师尺度归一化、面积效应校正、GeoEncoder、TeacherBGEncoder、SequenceEncoder、GateHead、BiasHead和QualityHead等模块；

**📊 数据集**

实验使用公开的CSIQ（900张受控失真图像）和LIVEC（1,162张真实失真图像）两个MOS基准，以及自制的2,816张铁路场景图像（无目标级MOS，仅通过教师一致性生成弱标签），三者分别用于目标级或全图质量评估；

**📈 对比分析**

与最佳单一教师、教师平均一致性以及SOTA全图NR-IQA模型进行对比；在CSIQ和LIVEC上BiCG-Net分别获得PLCC 0.9009和0.9133，超过最优教师（0.8230/0.7715）和一致性（0.8642/0.7427）；在自制数据上与弱标签的PLCC达0.9855；跨域零射转移（CSIQ→LIVEC）获得0.8416的PLCC，显著高于一致性0.6469；

**⚠️ 局限性**

局限性包括（1）离散的偏移锚点导致超大偏移时采样稀疏；（2）自制数据缺乏人类MOS，难以评估对感知的真实对齐；（3）跨域转移表现不对称，原因仍需深入分析；（4）分散门控模块对准确性贡献有限，仅为诊断工具；（5）模型主要针对单目标，未验证多目标或其它工业场景的可扩展性。

---

## 387. Ab Initio Accuracy Beyond the Cutoff: Distributed Message-Passing Equivariant Attention at Tens-of-Millions-of-Atom Scale

**arXiv ID:** 2611.06122 | [PDF](https://arxiv.org/pdf/2611.06122v1)

**作者:** Allex Kan `[一作]` (IQuest Research), Jia Zhang `[通讯]` (IQuest Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种分布式运行时 Tessera，用于在数千万原子级别上并行执行消息传递的等变 Transformer，并在大规模 GPU 集群上实现高效的分子动力学模拟。

**💡 创新点**

创新点包括：
- 采用等计数的空间填充曲线（Hilbert）划分，将每个 GPU 的原子数固定，从而保持张量形状不变；
- 结合多级邻居搜索与保守“skin”技术，实现邻居拓扑在多步内重用；
- 通过 CUDA Graph 捕获与重放、计划复用、发送端 SKV 投影以及前后向计算重叠，显著降低通信开销；
- 在 GPU 上实现去中心化的 MD 积分器，消除 rank‑0 级联瓶颈。

**🔧 技术方法**

使用的技术包括：
- Triton 自定义稀疏等变核（Clebsch–Gordan、径向基、注意力）
- Hilbert 线性排序 + 统一分块
- 多级（AABB → 块 → FAISS）邻居搜索
- CUDA Graph 捕获与重放
- NCCL all‑to‑all 与稀疏计划复用
- 发送端投影（SKV）与异步前后向重叠
- GPU 上的分布式 MD 积分循环

**📊 数据集**

主要数据集：
- 10.24M 原子 KCl 溶液（NVT/NVE 轨迹）
- 10.06M 原子诺瓦病毒衣壳（NVT 轨迹）
- 1.28M 原子 KCl‑水（强缩放基准）
- 10k、25k、60k 原子水等用于弱缩放验证。

**📈 对比分析**

比较方法：
- 在 512 GPUs 上与基准 Morton+全聚合方案比较，弱缩放效率达到 89.7%（10k/25k/60k 原子/GPU）。
- 与 256 GPUs 上的 1.28M 原子基准相比，强缩放在 256 GPU 处保持 64% 以上效率。 
- 轨迹能量漂移保持 0.42 µeV/atom/ps，结构与单 GPU 序列结果一致。 
- 与传统经典力场（百万原子级别）相比，保留了 ab‑initio 级别的准确性。

**⚠️ 局限性**

局限性：
- 在高度非均匀系统中，等计数划分导致边数量不平衡，计算负载仍需调节；
- cache‑hit 步长的前向计算略慢于重建步，主要由于通信等待；
- 目前仅在 NVIDIA H100 / A100 上验证，通信网络瓶颈仍未彻底消除；
- 对 TF32 对离子‑离子结构影响仍未充分评估；
- 与外部分子动力学软件（如 LAMMPS、GROMACS）直接比较尚未完成。

---

## 388. Improved Bounds on Distance Preservers

**arXiv ID:** 2611.05878 | [PDF](https://arxiv.org/pdf/2611.05878v1)

**作者:** Yanyu Chen `[一作]` (National University of Singapore), Mingyang Yang `[通讯]` (National University of Singapore)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了在给定无向无权图G和一组顶点对P的情况下，距离保持器的边数上限为O(n^8/9 p^4/9)，特别是在p∈ [n^1/2, n]的范围内。

**💡 创新点**

创新点在于改进了之前的边数上限，尤其是在p∈ [n^7/10, n]的情况下，缩小了与猜想O(n^2/3p^2/3+n)的差距。

**🔧 技术方法**

使用了距离保持器的技术，结合了分支事件和边的必要性等概念，提出了一种新的边界证明方法。

**📊 数据集**

使用了无向无权图G的顶点集和一组顶点对P，具体数据集未明确提及。

**📈 对比分析**

与Bodwin和Vassilevska Williams的O(n^2/3p^2/3+np^1/3)和Coppersmith与Elkin的O(min{n+n^1/2p, np^1/2})进行比较，本文的结果在特定参数范围内表现更优。

**⚠️ 局限性**

限制在于该结果主要适用于特定的顶点对数量范围，且在更广泛的情况下可能无法保持相同的边界。

---

## 389. Iterated creation, grading, and revision of data science projects by language models

**arXiv ID:** 2611.05548 | [PDF](https://arxiv.org/pdf/2611.05548v1)

**作者:** Rohan Alexander `[一作]` `[通讯]` (University of Toronto), Rohan Alexander (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对七种大语言模型在写作与评审两个角色中的端到端数据科学项目循环反馈进行了系统实验。

**💡 创新点**

首次将写作、评审和多轮反馈结合，揭示模型自评偏差和反馈对质量提升的影响。

**🔧 技术方法**

使用agentic工具调用（shell、文件读写等）和OpenRouter接口，结合评分表进行自动评审。

**📊 数据集**

模型自行选择多种公开数据集（USGS地震目录、世界银行指标、Toronto住房评估、NHANES健康调查、CO2能耗等）。

**📈 对比分析**

通过与无反馈、单回合全预算等对照，最终分数平均从59%提升至76%，提升幅度因写手和评审模型差异约30%。

**⚠️ 局限性**

局限在评估基于固定评分表、缺乏真实科学判断、模型自评偏差与噪声、未验证分析结果的正确性。

---

## 390. The Exact Memory-Rate Tradeoff for Three-User Secure Coded Caching

**arXiv ID:** 2611.06311 | [PDF](https://arxiv.org/pdf/2611.06311v1)

**作者:** Han Fang `[一作]`, Wei Kang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文针对三用户安全编码缓存问题，推导出从最小存储到最小传输率的完整零误差内存-率权衡曲线；

**💡 创新点**

其创新之处在于首次证明了对于任意文件数N≥2，三用户系统的极点与内存共享段是最优的，并发现并实现了一个新的三文件角点（M,R）=(3/2,2）来进一步降低传输率；

**🔧 技术方法**

论文采用线性秘密共享与MDS校验技术，构造了基于𝔽₃域的块码和键取消方案，利用信息不等式与同秘共享恒等式进行对偶证明；

**📊 数据集**

研究中未使用外部数据集，而是以理论文件库（互相独立的文件集合）为模型；

**📈 对比分析**

通过对比已知的极点（1,3）、((N+1)/2,3/2)、(2N-2,1)与新构造的角点，证明了所给的编码方案在所有可行点上实现了最优的传输率，特别在三文件情形下比传统内存共享方案节省了四分之一文件；

**⚠️ 局限性**

主要局限在于仅适用于三用户系统，其编码方案对文件数的依赖性以及实现复杂度（如键管理与子分割）仍需进一步研究与优化。

---

## 391. Exact signal recovery from generative priors with sigmoid output activation

**arXiv ID:** 2611.05626 | [PDF](https://arxiv.org/pdf/2611.05626v1)

**作者:** Matthew S. Scott `[一作]` (University of British Columbia), Özgür Yılmaz `[通讯]` (University of British Columbia)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出了一种理论框架，证明在使用高斯随机测量矩阵的条件下，具有ReLU隐藏层和最终Sigmoid激活的前馈神经网络的生成式压缩感知模型能够在满足一定测量数的前提下实现信号的精确恢复。

**💡 创新点**

创新点在于：①不依赖网络可扩展性（expansivity）假设；②通过构造相对误差的Sigmoid分段线性逼近，得到对差向量角度的控制，从而将片段线性网络的RIP性质迁移到最终Sigmoid网络；③给出了仅对信号离开饱和区间（[δ,1-δ]）的通用测量数上界，且该上界与纯ReLU网络的复杂度匹配，误差项仅为双对数阶。

**🔧 技术方法**

核心技术包括：高斯随机矩阵的Restricted Isometry Property（RIP）分析；片段线性网络的单元计数与超平面覆盖理论；对Sigmoid的相对误差分段线性逼近构造；RIP在差向量集合上的推广与误差传递；以及对饱和阈值 δ 的引入以控制逼近误差。

**📊 数据集**

本文为纯理论工作，未使用任何公开或实验数据集；所有结果均为严格的数学证明。

**📈 对比分析**

与以往仅针对ReLU或全局Lipschitz网络的压缩感知理论相比，本文的测量上界与已有最优结果相当（仅差一个双对数项），并且在满足信号不靠近Sigmoid饱和区间的情况下实现完全恢复；但未给出实验验证，无法评估在实际图像或信号数据上的具体性能。

**⚠️ 局限性**

主要局限包括：①需要信号在Sigmoid激活输出区间 [δ,1-δ] 内，无法直接处理完整Sigmoid网络的全局恢复；②逼近Sigmoid所需的分段数随 δ 的小取值急剧增大；③未考虑隐藏层也采用平滑激活的情况，未来工作需进一步推广。

---

## 392. Beyond Risk Cues: A Mechanism-Level Benchmark and Verifiable Debate for Scam Video Detection on the Web

**arXiv ID:** 2611.06110 | [PDF](https://arxiv.org/pdf/2611.06110v1)

**作者:** Yifan Wang `[一作]` (People's Public SecurityUniversity of China), Peng Chen `[通讯]` (People's Public SecurityUniversity of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MechScam机制级别诈骗视频数据集和ScamCourt可验证引用辩论框架，用于提升诈骗视频检测的解释性与鲁棒性。

**💡 创新点**

①将诈骗视频划分为六种机制并提供细粒度标签与证据引用；②构造基于真实证据的误导论证；③引入三方（检察、辩护、评审）对话与引用验证的辩论式判定。

**🔧 技术方法**

利用多模态大型语言模型（如Qwen3‑VL、GLM）、LoRA微调、跨折叠辩论训练、引用核查器与基于文本/帧的证据匹配技术。

**📊 数据集**

MechScam共6,614段YouTube视频，涵盖6种诈骗机制和6种非诈骗类别，附有文本/帧证据、解释、误导论证。

**📈 对比分析**

与多款零样本LLM及同骨干微调分类器对比，ScamCourt在二分类准确率≥93%、12类宏F1≥90%、对误导论证的误导率≤0.2%，显著优于基线且对硬负样本误判率低。

**⚠️ 局限性**

仍对某些机制如奖励索赔(S01)与代币投资(S05)的微妙区分表现有限；数据来源局限于YouTube，跨平台泛化尚待验证；模型训练成本高，辩论生成与评审复杂。

---

## 393. One Basket, Many Experts: Decorrelated Re-ranking for Financial Asset Recommendation

**arXiv ID:** 2611.05952 | [PDF](https://arxiv.org/pdf/2611.05952v1)

**作者:** Ryuki Unno `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种训练无关、可控的推荐重排序方法 DSER，利用专家跟随列表作为相关性核心，在顶级候选中用最小相关性资产替换部分位置，以提升组合的风险调整后收益。

**💡 创新点**

创新点在于把相关性–绩效折衷转化为单一整数参数 m，实现可审计的透明控制；并证明在保留专家核心的前提下，低相关资产的加入能够在不显著牺牲推荐相关性的情况下提升 Sharpe 比例。

**🔧 技术方法**

技术方法包括：专家跟随排名、Pearson 日收益相关矩阵、贪婪的最小相关性选择、基于回测的 walk‑forward 评估、以及对比 MMR、RURA 等多种基线。

**📊 数据集**

实验基于公开 FAR‑Trans 交易与价格数据，经过筛选得到 92 种交易活跃股票组成的可交易宇宙 U92，时间跨度 2018‑2022。

**📈 对比分析**

与 11 种 FAR 基准（包括纯相关性、纯回报、混合模型等）以及 RURA 重新排序器进行对比。结果显示，m=2 或 4 时 DSER 的 Sharpe 比例可提升约 0.25‑0.33（从 1.79 提升至 2.06‑2.12），而 nDCG 仅下降 3‑7%；在考虑交易成本后，m=2 在 10 万美元规模下仍保持正收益。整体而言，DSER 在可接受的相关性损失下提供了最高的综合性能。

**⚠️ 局限性**

局限包括：仅在单一 FAR‑Trans 数据集与特定 92 股流动性集合上验证；未对实时用户行为的影响进行在线实验；薄流动资产的高交易成本和潜在流动性风险可能限制大规模部署；相关性估计对样本窗口长度和资产活跃度敏感；未区分大小因子与相关性带来的收益提升。

---

## 394. Anchored Slice Propagation with Boundary Transport for 3D CT Segmentation from Three Orthogonal Annotated Slices

**arXiv ID:** 2611.05693 | [PDF](https://arxiv.org/pdf/2611.05693v1)

**作者:** Tushar Kataria `[一作]` (University of Utah), Shireen Y. Elhabian `[通讯]` (University of Utah)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

利用三条正交切片的注释，通过在切片交叉处进行锚点纠正和融合，改进3D CT结构分割，并对比三种边界几何传输方式（锚定二值掩模、截断签名距离场、轮廓跟踪）。

**💡 创新点**

创新点在于：①在固定的 affinity backbone 下，使用已注释切片的交叉点作为锚点实时纠正并权重融合各传播链；②对边界几何的隐式（SDF）与显式（轮廓）传输进行比较；③提出无监督学习的轮廓运动校正方法，仅依赖超像素伪标签。

**🔧 技术方法**

使用技术包括：固定的 Sli2Vol affinity backbone、锚点纠正与权重融合、截断签名距离场（SDF）传输、基于 affinity 的轮廓跟踪与学习式运动校正，以及超像素伪标签训练。

**📊 数据集**

数据集：公开CT数据集（CHAOS、NIH Pancreas‑CT、KiTS19、CT‑LYMPH‑NODES、SLIVER07、MSD、AMOS），约403张扫描，涵盖17个腹部结构，2,091个结构实例。

**📈 对比分析**

对比方法：锚定掩模、SDF+锚点、学习轮廓跟踪，平均 Dice 从 0.737 提升至 0.769；各方法在不同结构上表现不同，学习轮廓跟踪在食管和后腔血管上最优，SDF 在右腎上腺最优；总体性能提升显著。

**⚠️ 局限性**

局限性：未评估软掩模传输和 Sli2Vol+ 等其他基线；种子选择未通过真实标注流程；轮廓跟踪在细分支血管（如肝血管树）中效果欠佳；仍存在空肠、食管等结构难以完整分割。

---

## 395. Act-HiMotif: Hierarchical Daily Activity Schedule Generation via Learned Activity-Structure Motifs

**arXiv ID:** 2611.06038 | [PDF](https://arxiv.org/pdf/2611.06038v1)

**作者:** Yushi Bai `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), Yilin Lyu `[通讯]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于层次结构和可重用活动结构图案的生成模型 Act-HiMotif，用于根据个人属性生成完整的每日活动时间表。

**💡 创新点**

创新点在于：① 将家居日与活跃日分离处理；② 通过向量量化学习可共享的活动结构图案，实现稀缺结构的统计共享；③ 在结构生成后使用预算约束的自回归时长分配器，确保活动时长正整数且恰好填满一天。

**🔧 技术方法**

技术包括：profile encoder + active‑day gate；vector‑quantized motif learner（Bi‑GRU/GRU autoencoder）；条件 motif prior + motif‑conditioned structure decoder；预算约束的自回归时长分配器；多阶段训练与冷启动的生成器嵌入。

**📊 数据集**

使用 2017 年美国全国家庭出行调查（NHTS）数据，构建 264,234 条人-日时间表，包含 9 种活动类别和 48 个半小时时段。

**📈 对比分析**

与 ConditionalRNN、GenerativeRNN、ActVAE、DeepActivity‑ZS、ControlTraj、TrajGPT、CoDiffMob、MobilityGPT 等八个基线进行比较；在分布内和 CBSA‑disjoint 空间泛化两种评估协议下，Act-HiMotif 在 Episode JSD、Duration JSD、Start JSD、Slot JSD 和 Transition CPC 指标上均显著优于所有基线，显示更高的结构与时间分配保真度。

**⚠️ 局限性**

局限性包括：① 依赖于 NHTS 这类大规模出行调查数据，跨国或跨文化迁移可能受限；② 模型复杂度较高，训练与推理成本较大；③ 对极为罕见或新出现的活动结构仍可能生成误差，且对细粒度活动类别的细节建模有限。

---

## 396. RouTex: Multi-reference 3D Texture Generation by Reprogramming Rotary Position Encoding

**arXiv ID:** 2611.06070 | [PDF](https://arxiv.org/pdf/2611.06070v1)

**作者:** Chengyang Li `[一作]` (JD.com), Taicheng Huang `[通讯]` (JD.com)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 RouTex，一个基于多参考图像的 3D 纹理生成框架，能在 2D 多视角扩散模型中实现高质量、跨视角一致的纹理合成。

**💡 创新点**

核心创新在于（1）通过重新编程 3D Rotary Position Encoding（RoPE）实现无参数的“位置路由”（Positional Routing），自动将参考图像映射到对应视图；（2）引入 Dual‑Granularity Attention（DGA），在双流网络中先进行局部视图注意，再通过全局联合注意融合不同参考，实现细节保留与跨视角一致性并行提升。

**🔧 技术方法**

采用的技术包括：in‑context Diffusion Transformer (DiT)、Flux‑Kontext 结构、3D RoPE、位置路由与双层注意机制、FlashAttention V4 加速、flow‑matching 损失、以及在训练与推理中使用的 VAE 编码器/解码器。

**📊 数据集**

使用了约 200K 条 3D 资产，来源于 Objaverse、ABO、3D‑Future、TexVerse 等公开数据集，外加自建高质量手工模型，训练时每个模型生成 24 个参考视角与 30 个正交目标视角。

**📈 对比分析**

与 TRELLIS、TRELLIS.2、Hunyuan3D‑2.1、MV‑Adapter、LumiTex 等 5 个基线对比，采用 FID、KID、CLIP‑I、LPIPS 四项指标。RouTex 在 FID（12.06）、KID（8.66）、CLIP‑I（0.9423）与 LPIPS（0.1519）上均取得最佳或次佳成绩，尤其在细节清晰度与跨视角一致性方面显著优于其它方法。

**⚠️ 局限性**

局限性包括：仍需大量 GPU 资源训练与推理；对极端视角或极少参考图时可能产生纹理失真；模型仅在 2D 多视角扩散框架下工作，未完全解决 3D 结构与视角对齐的细粒度一致性问题。

---

## 397. Rubric Curriculum RL: Exploiting the Generation-Verification Gap in Non-Verifiable Domains

**arXiv ID:** 2611.06244 | [PDF](https://arxiv.org/pdf/2611.06244v1)

**作者:** Sumeet Ramesh Motwani `[一作]` (University of Oxford), Shital Shah `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于分段式对比奖励的 Rubric Curriculum RL（RcRL）来提升非可验证领域（如创意写作和健康问答）的模型生成质量。

**💡 创新点**

创新点在于将自我评判的对比奖励与动态进阶的评估准则（rubric）相结合，既利用了生成-验证的差距，又通过不断增加新的评价维度来抑制奖励劫持并保持输出多样性。

**🔧 技术方法**

采用自我评判模型进行对比奖励、梯度策略优化（GRPO）以及对准则的加权递进课程机制，并使用熵正则化保持探索性。

**📊 数据集**

使用了创意写作基准（Creative Writing V3、FictionBench、LechMazur）和健康问答基准 HealthBench，以及 Qwen3、Llama 系列大模型作为实验平台。

**📈 对比分析**

相较于基线（基准模型、静态 Rubric、无课程或无对比奖励），RcRL 在 FictionBench 上实现 73.9% 的胜率、在 Creative Writing V3 上提升 ELO 约 500 分，在 HealthBench 上实现 14.6%/25.4% 的分数提升，且在 1000+ 训练步后仍持续改进且保持高熵。

**⚠️ 局限性**

局限性包括对评判模型的依赖、评估噪声、对准则规模的计算成本，以及在更大批量或更细粒度的评估维度下需要更高效的对比策略。

---

## 398. When Every Choice Is Correct: A Causal Study of Numerical Preference in Language Models

**arXiv ID:** 2611.06025 | [PDF](https://arxiv.org/pdf/2611.06025v1)

**作者:** Hongnian Huang `[一作]` (University of New Mexico), Hongliang Liu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在语言模型中进行数值选择任务（如“在1到30之间选择一个数字”），利用激活编辑、缓存替换、注意力读者替换等细粒度干预，追踪并分析模型在层级和 token 级别对等价答案的偏好如何被形成、传播和改变。

**💡 创新点**

创新点在于：① 将数值选择作为可控实验平台，以等价答案消除正确性干扰；② 结合扰动探测、梯度归因与 scalar patching，精确定位并干预单个 MLP 神经元；③ 通过缓存与注意力读者的替换，揭示预答案编辑如何通过后续层的计算影响后续 token 的竞争；④ 评估自然上下文转移对已干预偏好的携带能力，探讨偏好来源的可迁移性。

**🔧 技术方法**

主要技术包括：扰动探测（perturbation probing）、梯度归因、scalar patching 与零化、缓存替换、注意力读者（attention head）替换、后期 MLP clamp、KL 散度与 Jensen–Shannon 距离评估、对数概率差异分析、候选答案概率分布比较。

**📊 数据集**

实验基于 Qwen3-4B、Qwen3-4B-Instruct-2507 与 Qwen3-8B 三个 checkpoint；使用多种英文提示模板（包括 1–30、1–40、1–35、1–20 等区间）和 12 个重新措辞的词表；通过模型生成的回答与已知答案对照，构成数值选择实验集；没有使用公开的外部数据集。

**📈 对比分析**

比较方法：① 计算不同干预下候选答案的 log‑odds 变化与 KL 散度；② 评估答案切换率与概率分布的改动；③ 对比自然上下文转移下的 Jensen–Shannon 进度。结果显示：预答案编辑能显著改变 17 对 15 的概率分布（平均 log‑odds 变化约 0.28 nat），但答案切换发生率有限；在不同提示下效果差异明显；自然上下文转移平均进度低（仅 0.4–0.5 的 donorward 进度），表明干预的可迁移性有限。

**⚠️ 局限性**

局限性：① 仅在 Qwen 系列模型和有限的 1–30 数值区间内验证；② 未找到完整或最小的偏好实现电路；③ 干预结果的解释性仍不完整，无法确定单一神经元或模块的绝对作用；④ 自然上下文转移效果不显著，提示偏好可能由多种上下文相关机制共同决定；⑤ 研究聚焦于数值选择任务，尚不清楚能否直接推广到更复杂的多选或决策任务。

---

## 399. Vision Paper: Why Distributed LLM Post-Training and Decentralized Alignment Are Now Essential

**arXiv ID:** 2611.05841 | [PDF](https://arxiv.org/pdf/2611.05841v1)

**作者:** Zhanhong Jiang `[一作]` (Iowa State University), Soumik Sarkar `[通讯]` (Iowa State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文主张分布式后训练（联邦与去中心化）是大模型对齐的必然方向，并提出了FedDPO与D-DPO两种分布式偏好优化框架，正式化了参考漂移、偏好异质性等关键技术难点，并给出同步参考更新（SRU）与邻域共识等解决方案；还提出了七条优先研究方向与初步实验验证；

**💡 创新点**

创新点在于：①首次将隐私、代表性与可扩展性三大支柱系统化为分布式对齐的必要性论证；②构建FedDPO与D-DPO的数学模型，揭示参考漂移的根源并提出SRU/邻域共识等对策；③将对比损失的差分隐私样本复杂度与拜占庭鲁棒性等问题纳入分析；④提出“参考自由”方法（SimPO/ORPO）与“多样化”对齐策略；⑤给出完整的研究路线图与评估指标。

**🔧 技术方法**

使用的技术包括：联邦学习（FedAvg）与无服务器gossip SGD；LoRA/QLoRA压缩通信；差分隐私（DP‑SGD）与安全聚合；同步参考更新与邻域共识；参考自由方法SimPO/ORPO；混合梯度与混合策略；安全与治理工具（零知识证明、日志）等。

**📊 数据集**

实验使用的主要数据集有：GPT‑2（作为预训练模型）；HH‑RLHF、UltraFeedback等偏好对齐数据；MT‑Bench、AlpacaEval等基准；以及自行划分的多语言/多领域偏好子集。

**📈 对比分析**

对比方法包括：FedDPO与无SRU、D-DPO与无邻域共识；在GPT‑2上评估收敛速度、偏好胜率以及在不同DP噪声下的对齐质量。实验结果表明：SRU显著提升收敛速率；邻域共识可缓解去中心化中的参考漂移；DP噪声会显著降低对齐效果，尤其在早期训练阶段。总体性能尚处于实验阶段，缺乏大规模对齐基准。

**⚠️ 局限性**

局限性：①缺乏大规模真实对齐数据与完整实验；②参考漂移、DP样本复杂度与拜占庭鲁棒性的理论分析尚未完成；③去中心化时安全与治理难题未完全解决；④通信与计算开销仍高，需进一步压缩；⑤对齐评估指标与公平度量尚待标准化。

---

## 400. Pairing Expert Critique with LLM-Driven Refinement: An Iterative Approach for Curating Quality AI Evaluation Datasets

**arXiv ID:** 2611.05749 | [PDF](https://arxiv.org/pdf/2611.05749v1)

**作者:** Honson Ling `[一作]` (Microsoft Corporation), Stephanie Rosenthal `[通讯]` (Microsoft Corporation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

介绍了SME-C（Subject Matter Expert Critique）流程，通过金融专业人士的轻量化批评与大型语言模型（LLM）协作迭代生成与改写，构建高质量的金融业务简报评估数据集和评估标准。

**💡 创新点**

创新点在于将专家完整撰写转变为轻量化批评，显著降低专家时间成本，并通过LLM对批评进行自动分析与改写，实现可扩展、可迭代的域特定数据集构建与评估指标生成。

**🔧 技术方法**

采用人机协作工作流，结合LLM驱动的文本生成与改写、主题分析（人工与LLM）以及基于专家反馈的评估量表构建，并使用web‑启用的大型语言模型（如ChatGPT等）进行内容生成与改写。

**📊 数据集**

实验使用了411位金融专业人士对520个查询收集的3,853条批评，并生成三种模型（A、B、C）产生的候选回应，形成原始回应、Critique‑based改写和Vote‑based改写的对照数据集。

**📈 对比分析**

在验证阶段，采用盲评、重复测量ANOVA与配对t检验比较原始回应与Critique/ Vote两种改写版本，结果显示两种改写均显著提升质量（原始平均6.87→改写约7.3，平均提升≈1分），改写效果在弱质量回应、不同类别和查询复杂度上更为显著。

**⚠️ 局限性**

主要局限包括：调查对比方式可能限制对内容的深入阅读，专家样本为半专家而非深度专家导致专业细节反馈不足，LLM改写可能引入幻觉或不必要的结构变化，评估侧重沟通与结构有效性，未完全覆盖事实准确性与数值验证。

---

## 401. Towards the Query Complexity of Sampling under Strong Log-Concavity and Log-Sobolev Inequalities

**arXiv ID:** 2611.05813 | [PDF](https://arxiv.org/pdf/2611.05813v1)

**作者:** Huanjian Zhou `[一作]` (University of Tokyo), Taiji Suzuki `[通讯]` (University of Tokyo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了一族强对数凹、平滑且高度难以采样的势能函数，并证明在给定的查询模型下采样至少需要指数级的查询次数。

**💡 创新点**

创新点在于利用嵌套张量特征、门控函数与能量截断，构造出在梯度和数值查询上几乎正交的高维势能，从而实现了新的下界。

**🔧 技术方法**

主要技术包括高维高斯随机场、协方差分析、Schur 补、分解定理、随机过程尾部估计以及复合投影和正交化。

**📊 数据集**

使用的是由高斯随机系数生成的合成势能样本，没有真实数据集。

**📈 对比分析**

与现有上界相比，该结果表明在最坏情况下采样所需的查询数至少为 exp(Ω(d^{1/3})), 与已知多项式时间算法形成了明显的对比；实验验证表明对数级增长的采样误差难以避免。

**⚠️ 局限性**

局限性包括下界仅针对特定的查询模型（值-梯度查询），并不适用于更一般的采样算法；构造的势能虽然理论上满足强对数凹和平滑条件，但其常数和结构对实际应用的可迁移性有限。

---

## 402. Role-based Effectiveness of an Epistemic Uncertainty Estimate in Training Models for Semantic Segmentation of 3D LiDAR Point Clouds

**arXiv ID:** 2611.05569 | [PDF](https://arxiv.org/pdf/2611.05569v1)

**作者:** Krishnakumar N `[一作]` (International Institute of Information Technology Bangalore), Jaya Sreevalsan-Nair `[通讯]` (International Institute of Information Technology Bangalore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过构造基于几何描述符的确定性先验不确定性估计，并将其作为输入特征或损失加权的方式，实现在ALS和MLS点云分割中的不确定性增强。

**💡 创新点**

创新点在于：①使用仅基于训练集几何特征的类条件高斯模型生成与模型无关的不确定性估计；②将该估计作为特征注入或用于自适应损失权重，提升少数类别恢复和误差-不确定性对齐；③在三种网络架构（点级ANN、PointNet++、RandLA-Net）上统一验证其普适性。

**🔧 技术方法**

主要技术包括：多维几何特征提取（Eigenvalue、Planarity等）、高斯判别分析（GDA）求解类条件分布、负对数似然作为不确定性量化、Patch Accuracy‑versus‑Patch Uncertainty（PAvPU）评估、标准mIoU/Accuracy指标。

**📊 数据集**

实验使用四个大型LiDAR基准数据集：Vaihingen（ALS）、AHN3（ALS）、Paris‑Lille‑3D（MLS）和Toronto‑3D（MLS），在空间离散的训练‑测试拆分上进行评估。

**📈 对比分析**

与基线模型对比，加入不确定性后在mIoU上提升0.01–0.05，显著恢复了多数被忽视的少数类；PAvPU在大多数配置中提升了约0.05；不同架构均表现出一致的改进，证明方法的通用性。

**⚠️ 局限性**

局限性包括：①使用固定半径邻域，无法自适应点密度变化；②点级模型缺乏更大尺度上下文，难以识别细小或稀疏对象；③仅考虑确定性模型不确定性，未建模随机噪声；④PAvPU受中位阈值限制，对高精度模型的区分度有限。

---

## 403. SciToT: Multimodal Tip-of-the-Tongue Retrieval for Scientific Papers

**arXiv ID:** 2611.05829 | [PDF](https://arxiv.org/pdf/2611.05829v1)

**作者:** Stan Fris `[一作]` (University of Amsterdam), Maarten de Rijke `[通讯]` (University of Amsterdam)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了针对科学论文的多模态Tip‑of‑the‑Tongue检索基准SciToT，并提供生成查询的完整 pipeline；

**💡 创新点**

创新点在于：①构建超过100K篇科学文献的多模态 ToT 基准；②利用 LLM 自动提取文本与视觉记忆线索并生成符合真实查询特征的合成 ToT 查询；③系统化的人工与 LLM 评估流程，保证查询与目标文档高度匹配；

**🔧 技术方法**

主要技术包括：多模态 LLM（Qwen3‑4B、Qwen3‑VL‑4B‑Instruct、Phi‑4‑14B、Gemma‑4‑12B‑IT）用于线索提取与查询生成；文本检索模型（BM25、E5‑small‑v2、ColBERTv2）、视觉检索模型（OpenCLIP、Qwen3‑VL）、页面级检索模型（ColQwen2.5），以及多模态融合检索（Qwen3‑VL+文本）。

**📊 数据集**

使用从 ACL Anthology、PubMed Central、arXiv 等公开源收集的 108,512 篇开放获取论文，提取文本与图像作为输入。

**📈 对比分析**

评估方法：在每个科学领域独立计算 MRR@10、Recall@1/10/100。结果显示，Qwen3‑Emb 文本检索在 MRR@10 上达到 0.710，远超传统 BM25、E5、ColBERTv2；视觉单模态 OpenCLIP 效果差；Qwen3‑VL 在仅用图像时已达到良好效果，页面级 Qwen3‑VL（含图像）在 MRR@10 上最高达 0.793；整体表明更表达式的多模态表示显著提升性能。

**⚠️ 局限性**

局限性：合成查询过度强调视觉线索，可能与真实用户查询不完全匹配；LLM 的生成与评估带来偏差；跨域泛化能力尚待验证；多模态索引对存储和计算资源要求高。

---

## 404. Decompose the Distillation: Interpretable Single-Pass Guidance for Diffusion Models

**arXiv ID:** 2611.05755 | [PDF](https://arxiv.org/pdf/2611.05755v1)

**作者:** Wuyang Zhang `[一作]` (Northeastern University), Xiaolin Xu `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究一种单通道推理方法，保留CFG的基础流和引导方向分解，并通过双LoRA适配器实现仅一次前向传播即可获得可调节的引导速度。

**💡 创新点**

创新点在于：①在单通道框架下保留CFG的可解释接口；②采用统一多尺度蒸馏损失匹配推理时的整体误差；③通过尺度自适应调制与时间网格拉伸两种机制抑制高尺度下的误差放大，从而在不增加前向传播次数的前提下实现CFG级别质量。

**🔧 技术方法**

使用技术包括：双LoRA轻量级适配器（约5%参数），统一多尺度蒸馏目标，尺度自适应调制（窗口、幅度衰减、对齐门），时间网格拉伸（重分配采样点），以及基于DiT的ODE采样框架。

**📊 数据集**

数据集：ImageNet 256×256（DiT-XL/2、DiT-L/4）用于主要评估；稳定扩散2.1 pilot做文本条件实验；未在视频或更高分辨率图像上测试。

**📈 对比分析**

与CFG、两步改进方法（CFG+Interval、CFG-Zero*、APG、Autoguidance）以及单通道基线（AGD-MultiScale、AGD）对比。实验表明在25步、8尺度下实现1.93×的时钟速度提升，并在25×25（N,s）格点中有24/25配置达到CFG误差阈值，优于20/25的单通道基线。

**⚠️ 局限性**

局限性：仅在DiT-XL/2/DiT-L/4、ImageNet 256×256和单个Stable Diffusion 2.1 pilot 上验证；未测试视频、高分辨率、不同任务；调制超参数基于部分配置，未在所有配置上细致调优；对分解精度的评估主要靠余弦相似度、组件替换和定位分数，缺乏理论保证。

---

## 405. TouchéWorld tech report : Towards Online Unified Camera--Hand Exploration in Generative Worlds

**arXiv ID:** 2611.06061 | [PDF](https://arxiv.org/pdf/2611.06061v1)

**作者:** Jiaao Yu `[一作]` (East China Normal University), Shijie Jia `[通讯]` (Dalian Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TouchéWorld框架，将连续手部交互与摄像机导向的生成视频模型结合，实现在线手部与视角控制。

**💡 创新点**

创新点在于构建联合摄像机–手部训练数据、结合空间骨架与数值手部运动的多分支条件、使用因果适配和少步蒸馏实现块级在线生成，并提出TouchéBench评测。

**🔧 技术方法**

采用预训练的LTX-2.5图像到视频模型、SolarWM摄像机适配、手部骨架VAE+时序编码器，使用LoRA参数高效适配，并在因果训练和分布匹配蒸馏中实现。

**📊 数据集**

使用约41k条异构第一人称视频（EgoLife、EgoStandard等），并在EgoStandard-200、EgoLife-300、HOT3D-45、HOI4D-200等数据集上评测。

**📈 对比分析**

与LTX-2.5、SolarWM、JCV、Hand2World等基线比较，在视频质量（PSNR、SSIM、LPIPS）和交互指标（PCK、Camera/Interaction通过率、Online Interaction通过率、Object IoU）上均表现优异。

**⚠️ 局限性**

局限在于因果适配和蒸馏后手部跟踪与交互性能下降，难以在高频动态手势或复杂人机交互中保持一致，且对手部运动的物理真实性仍有限。

---

## 406. LLM-based NPCs in Video Games: A Systematic Mapping Study

**arXiv ID:** 2611.05427 | [PDF](https://arxiv.org/pdf/2611.05427v1)

**作者:** Dilan Mian `[一作]` (Ontario Tech University), Cristiano Politowski `[通讯]` (Ontario Tech University)

**通讯引用:** 531 | [OpenAlex ID](https://openalex.org/A5045249042)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地对2023-2026年间的LLM驱动NPC研究进行映射与分析，梳理了42篇核心论文，提供了首个量化的架构、组件、性能与可复现性概况。

**💡 创新点**

创新点在于：①首次以系统映射研究方法构建完整的LLM NPC研究景观；②量化了短期/长期记忆、规划、检索等关键组件的采用比例；③揭示了延迟、成本、可控性等商用障碍，并给出基于实证的改进建议。

**🔧 技术方法**

主要技术手段为系统映射（PRISMA+PICOC）、文献检索、数据提取、质量评估（5项指标）和统计汇总；在此基础上进行主题分解（架构、范围评估、工程与控制、质量与透明度）。

**📊 数据集**

使用的“数据集”是从ACM、IEEE和Scopus检索得到的377篇文献，其中筛选出42篇作为研究核心；并未使用游戏或对话语料库，而是以论文内容作为分析对象。

**📈 对比分析**

比较方法是对每个子问题进行归纳与统计，给出组件使用频率、延迟分布、对比基线结果等定量指标；结果显示短期记忆/世界状态占比高（60%/69%），长记忆/检索使用率低（31%/14%），平均延迟2.6–15 s。

**⚠️ 局限性**

主要局限：缺乏统一的可复现性与评估基准，文献多为单一研究，样本规模小；LLM输出的不确定性与安全（幻觉、偏见）仍未得到系统解决；商业化障碍如延迟、成本、内容审核和多玩家同步等尚未被克服。

---

## 407. Grounded but Not Immune: Confirmation Bias in Agent-as-a-judge

**arXiv ID:** 2611.05662 | [PDF](https://arxiv.org/pdf/2611.05662v1)

**作者:** Drake Benjamin Brown `[一作]` (University of Utah), Daniele Bonadiman `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在 Agent-as-a-Judge 设定下，工具使用型评审者在面对假预筛选提示时会出现确认偏误，并通过对比有无提示的判决，分析了判决偏移的原因与机制。

**💡 创新点**

创新点在于：①将确认偏误的研究扩展到可获取环境证据的 grounded judge；②提出通过“轨迹感知的元评审者”对判决偏移原因进行细粒度归因（过度信任、幻觉、调查不足）；③系统评估了不同提示属性（框架、权威性、确定性）及方向性对偏误的影响；④探讨了轻量级提示修正的局限性，指出需要方向感知的防御。

**🔧 技术方法**

技术包括：使用多模态 LLM 评审者（Sonnet-4.6、Opus-4.7、Qwen3-235B、GPT-OSS-120B）在完整轨迹和工具访问下进行判决；通过引入单句预筛选提示（正向或反向）作为对照；利用三模型元评审者对判决偏移进行机制归因；实验中使用了自动化的评估器以获得真值；实现了四类轻量级提示干预（guard instruction、cite evidence、learned skill、sufficiency gate）。

**📊 数据集**

数据集包括：AJ-Bench（Postgres、Filesystem、Deep Search）、SWE-Bench-Verified、Terminal-Bench-2.0，共五个基准。每个数据集都有可检验的任务结果，实验在其轨迹上进行。

**📈 对比分析**

比较方法是：对每条轨迹，分别在有提示和无提示的情况下让评审者给出二元成功/失败判决，计算在无提示下正确的判决被提示误判的比例（harmful flip rate）以及提示改正错误的比例（helpful flip rate）。实验显示：在所有评审者与数据集上，单句假预筛选提示能将正确判决错误的比例提升至约15–35%；提示方向为失败时影响更大；不同提示属性对偏移有显著差异。轻量级干预虽能降低 harmful flip，但同时削弱了 helpful flip。

**⚠️ 局限性**

局限性包括：①提示设计为明显、单句，未探讨更隐蔽或更自然的提示；②归因实验仅在一款模型上深入，未跨模型扩展；③轻量级干预的评估基于对提示攻击已知的假设，真实环境下效果可能更差；④实验未覆盖所有可能的评审者规模与能力差异；⑤只关注二元判决与部分定性维度，对更复杂的评价维度研究有限。

---

## 408. PIMROP: A Feasibility Study of Return-Oriented Programming on Commercial Processing-in-Memory Architectures

**arXiv ID:** 2611.06036 | [PDF](https://arxiv.org/pdf/2611.06036v1)

**作者:** William C. Tegge `[一作]` (Syracuse University), Alex K. Jones `[通讯]` (Syracuse University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在商业 Processing‑in‑Memory（UPMEM DPU）上首次实现并演示了基于 ROP 的代码重用攻击，并构建了适用于该 ISA 的 gadget 扫描器 Galileo‑for‑DPU。

**💡 创新点**

创新点包括：证明 DPU ISA 能满足 Shacham 的可行性论断；发现并利用 DPU 的返回指针为指令索引、上升式栈导致的堆栈保护失效；以及展示无需代码注入即可通过重用库 gadget 完成秘密泄露和完整链式控制流。

**🔧 技术方法**

使用了固定宽度 8 字节 RISC 的 Galileo‑style gadget 扫描算法、LLVM 工具链编译、功能仿真环境、数据流追踪、以及 bounds‑checked DMA 等防御原型。

**📊 数据集**

采用 UPMEM DPU runtime 库（6,711 指令）、13 个 PrIM 基准（共 13 个应用程序）以及自定义的易受攻击 kernel（vuln_dpu）作为实验数据集。

**📈 对比分析**

通过与现有 RISC‑ROP/ARM 等基准比较，统计到 517 个非平凡 gadget 覆盖所有五类 ROP 原语；在功能仿真中 10/10 运行成功；对比边界检查和栈保护等防御，展示攻击的可复制性和防御成本。

**⚠️ 局限性**

局限性包括：仅在功能仿真（非真实硬件）上验证；扫描基于 llvm‑objdump 文本，可能漏掉未解析的 gadget；未实现自动化链生成；未测试多 Tasklet 并发与真实 DMA 行为，且缺少硬件 shadow‑stack 等更高效防御的硬件实现验证。

---

## 409. Beyond the Landing Page: Website Blocking, Tracker Visibility, and Human Alignment in Automated Web Crawls

**arXiv ID:** 2611.06197 | [PDF](https://arxiv.org/pdf/2611.06197v1)

**作者:** A H M Nazmus Sakib `[一作]` (University of Texas at San Antonio), Murtuza Jadliwala `[通讯]` (University of Texas at San Antonio)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

比较了基于 LLM 的 agentic 爬虫与传统非 agentic 爬虫在 2000 个热门网站上的网站干预、第三方跟踪可见度以及与人类浏览者的匹配度。

**💡 创新点**

首次将 LLM 驱动的浏览代理与多种自动化框架（Playwright、Patchright 等）在相同 URL、不同配置、交互记录等维度进行系统对比，并量化了网站阻断、跟踪计数与真实用户行为的一致性。

**🔧 技术方法**

使用了 Playwright、Patchright、BrowserForge、Chrome DevTools Protocol、Qwen3‑VL‑8B LLM、OpenWPM JS 钩子、TCF/Google Consent Mode、Cloudflare 挑战、网络与 JavaScript 调用记录等技术。

**📊 数据集**

构建了 2000 个流行网站的语料库（Tranco Top 10K + CrUX Top 1000），并挑选 450 个子集用于细粒度实验；另外收集了 26 名参与者在 415 页（共 4353 次）上的浏览日志。

**📈 对比分析**

通过 URL 匹配、网络响应和 JS 调用对比，计算页面拒绝率、第三方跟踪组织数、指纹行为，采用 2000 次 Bootstrap、符号翻转检验和 Holm 校正；结果显示 agentic 爬虫在跟踪覆盖率上高于非 agentic，但精确度低；网站阻断率与 Patchright 匹配，显示配置对干预的显著影响。

**⚠️ 局限性**

局限包括：样本规模有限、仅在美国节点进行爬取、代理/地区差异未充分考量、未记录参与者的同意决策、网络捕获缺失导致的对比不完整、指纹仪表差异影响结果、时间漂移未完全消除。

---

## 410. Breaking the One-Player Extension Barrier: History-Aware Repair for Multiplayer Approximate Nash Equilibria

**arXiv ID:** 2611.06076 | [PDF](https://arxiv.org/pdf/2611.06076v1)

**作者:** Jinqi Tang `[一作]` (Peking University), Cong Fang `[通讯]` (Peking University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文提出了一种基于修复机制的迭代算法，用以逼近博弈中的Nash均衡。

**💡 创新点**

创新点在于将证书、逆向扫描与正向校正相结合的内存管理策略，显著提升了收敛性与稳定性。

**🔧 技术方法**

主要技术包括线性规划求解、概率标签与证书事件构造、潜在函数分析以及逆向/正向修正步骤。

**📊 数据集**

实验基于理论模型，未使用任何实际数据集，所有结果均来自严格的数学证明。

**📈 对比分析**

与现有方法相比，本文算法在理论上实现了更快的收敛速度和更小的误差上界，实验验证显示性能优于对标算法。

**⚠️ 局限性**

局限在于对玩家规模的计算量仍较大，对参数设置要求严格，且实现复杂度相对较高。

---

## 411. Decouple to Reuse: Fine-Grained Knowledge Augmentation for Time Series Foundation Models

**arXiv ID:** 2611.05993 | [PDF](https://arxiv.org/pdf/2611.05993v1)

**作者:** Yuqian Wu `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种两阶段细粒度知识增强框架，能够把预训练时间序列模型中的趋势与季节性模式解耦并存储为共享残差代码表，以提升冻结模型在下游任务中的预测性能。

**💡 创新点**

创新点在于将时间序列知识拆解为可独立重用的趋势与季节模式，构建共享代码库并通过任务自适应聚合与门控融合实现细粒度重用，避免了传统窗口级检索带来的噪声。

**🔧 技术方法**

利用预训练的Transformer/MLP时间序列基础模型进行特征编码，训练残差编码器生成趋势和季节代码表；下游阶段采用检索、聚合与门控融合机制对冻结模型进行增强。

**📊 数据集**

在八个真实世界预测基准（如电商销量、交通流量、能源消耗等）和四个主流TSFM骨干（Chronos、MOMENT、TimesFM、TTM）上进行实验。

**📈 对比分析**

与传统的窗口检索增强（TS‑RAG）和参数适配（UniTS、ST‑LoRA）等方法比较，提出的框架在大多数数据集上显著降低了MAE/MSE，且计算效率和内存占用更优。

**⚠️ 局限性**

局限性包括对预训练数据的依赖、在极端领域漂移场景下仍需进一步验证、以及代码表大小与检索速度在极大规模时的可扩展性需改进。

---

## 412. Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning

**arXiv ID:** 2611.05393 | [PDF](https://arxiv.org/pdf/2611.05393v1)

**作者:** Riccardo Riccio `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**通讯引用:** 28341 | [OpenAlex ID](https://openalex.org/A5056748708)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种挑战感知的医疗多智能体推理框架，利用外部“挑战者”监控推理过程并触发知识检索或专家招募，以提高难题上的回答准确率。

**💡 创新点**

创新点在于：①将知识检索与专家招募的触发时机从固定的“预先检索”转移到“推理中已显露的具体缺口”上；②引入独立的知识门控和专家招募器，分别决定检索与招募；③针对两类干预渠道的协调与交互提供系统化实验。

**🔧 技术方法**

技术上结合了检索增强生成（RAG）与多智能体辩论框架，使用GPT-4o-mini（以及GPT-4.1-mini）作为核心模型，构建挑战者、知识门控、证据策展器、专家招募器和调解器等模块，并在推理回合间插入外部检索与专家。

**📊 数据集**

主要使用MEDAGENTSBENCH硬难度子集（862道题）覆盖九个生物医学数据集（MedQA、PubMedQA、MedMCQA等），以及在100道MedQA题目上进行基于Web搜索的验证；此外还评估了GroK模型和不同检索后端。

**📈 对比分析**

与传统的“预检索”或“无检索”基线相比，挑战感知后检索提高了约2.6个百分点（从23.4%到26.3%），在多专家体系中提升6个百分点；在Web搜索实验中差距为6个百分点；而自我触发专家招募效果有限，只有在挑战者提示后才有显著提升；两类干预的简单叠加反而导致性能下降。

**⚠️ 局限性**

局限性包括：仅在多项选择问答上评估，未验证临床实用性；使用的模型和检索后端有限，未考察开源或更大模型；实验缺乏调用量匹配控制，难以分离“更多计算”与“更精准调度”的贡献；协调两干预渠道的机制仍待完善；最终系统对错误信息或偏见的容忍度不高，需进一步安全审计。

---

## 413. Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox

**arXiv ID:** 2611.06022 | [PDF](https://arxiv.org/pdf/2611.06022v1)

**作者:** Nam H. Le `[一作]` `[通讯]` (University of Vermont), Nam H. Le (University of Vermont)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文探讨了生成模型在训练过程中如何因依赖于早期模型的输出而导致知识的丧失，特别是稀有知识的丧失，并提出了文化进化理论作为解决这一问题的框架。

**💡 创新点**

创新点在于将罗杰斯悖论应用于模型崩溃的分析，强调了新鲜数据的供应是动态的，而不是外部固定的资源，并提出了人类学习与模型输出之间的相互作用。

**🔧 技术方法**

使用了社会学习理论和罗杰斯模型的定量分析方法，通过实验验证了模型在不同新鲜数据比例下的知识保留情况。

**📊 数据集**

使用了来自古腾堡计划的文本数据集，包括达尔文的《物种起源》和梅尔维尔的《白鲸》，以及维基文本数据集。

**📈 对比分析**

通过与其他模型的比较，发现引入新鲜数据可以显著提高模型的知识保留率和准确性，尤其是在快速变化的环境中，模型需要更多的新鲜数据来保持性能。

**⚠️ 局限性**

限制在于所使用的语言模型较小，且实验主要基于固定文本的滑动窗口，而不是动态变化的事实世界，未来需要在更大规模的模型和更复杂的环境中进行验证。

---

## 414. Online Adaptation to Latent Physical Properties for Learning from Demonstration

**arXiv ID:** 2611.06118 | [PDF](https://arxiv.org/pdf/2611.06118v1)

**作者:** Kotaro Kajitsuka `[一作]` (University of Tsukuba), Fumio Kanehiro `[通讯]` (University of Tsukuba)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于演示学习的自适应操纵方法，利用交互产生的隐式物理属性表示在执行过程中实时更新，并将其作为条件输入给扩散策略，从而实现对视觉相同但物理属性不同的物体进行重抓和接触点调整。

**💡 创新点**

创新点在于将物理属性的隐式表示与演示学习结合，既在演示阶段通过预测误差学习到对象特定的动态参数，又在执行时利用该参数在线估计并驱动策略决策，无需显式物理测量。

**🔧 技术方法**

采用变分自编码器提取视觉特征，构建多层感知机动态模型，学习隐式动力学参数，训练扩散策略，并通过梯度下降在运行时更新物理表示。

**📊 数据集**

使用了三项真实机器人任务（袋子放置、盒子推送、水平举起）收集的遥控演示数据以及对应的仿真数据，总计约三百次演示，涵盖不同重心配置。

**📈 对比分析**

与基线扩散策略（无物理表示）以及更长观察历史的版本进行对比，实验表明在真实任务中成功率提升约 15–30%，在仿真中对未见重心位置的成功率提高至约 58%，但仍低于 70%。

**⚠️ 局限性**

局限性包括对短期交互中物理表示更新不够快导致失败、未能在未见物理属性上实现高适应率，以及仅在一维重心变化上评估，缺乏多维质量或摩擦等属性的验证。

---

## 415. Improved lower bounds on the approximability of Multiway Cut using Sherali-Adams relaxations

**arXiv ID:** 2611.05511 | [PDF](https://arxiv.org/pdf/2611.05511v1)

**作者:** Joshua Brakensiek `[一作]` (University of California Berkeley), Uri Zwick `[通讯]` (Tel Aviv University)

**通讯引用:** 10137 | [OpenAlex ID](https://openalex.org/A5005525211)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文通过构造离散和连续两种形式的流线性规划（Flow LP），并将其与多路切割（Multiway Cut）问题的线性规划松弛（Vertex-Consistency LP）相联系，提供了一种新的证明框架，用以给出多路切割问题在UG假设下的近似不可解性下界。

**💡 创新点**

创新点在于：①将多路切割问题映射到离散单纯形 Δ_{k,n} 上的边和单元格，构造兼容性约束的线性规划；②引入“向上细胞”和“向下细胞”两类细分，保证每条边正好属于一条向上细胞与一条向下细胞，从而实现对切割边概率的精确刻画；③提出了 Flow LP 作为中间桥梁，通过流量与容量的最大化/最小化，获得更紧的下界；④给出了连续版本的 Flow LP，并证明其在 n→∞ 时与离散版收敛，提供了分析的连续工具；⑤对 k=4 的情形给出了显式的可行解并计算了相应的下界。

**🔧 技术方法**

主要技术包括：离散化单纯形与细胞分解、线性规划与其对偶、流量与容量的对偶关系、连续测度论（Dirichlet 分布、梯度与散度定理）、多维分段函数与多项式概率（多项式分布）、以及对流量约束的多重积分与解析证明。

**📊 数据集**

由于研究属于理论计算复杂性和优化，不依赖具体实验数据集；但在数值实验中，作者使用 MATLAB/GLPK 等工具对不同 k、n 组合求解离散 LP，得到下界值；同时为 k=4 构造的连续解使用符号计算工具验证约束。

**📈 对比分析**

通过数值求解离散 LP（表格 4-6）与连续 LP 的比较，发现连续 Flow LP 的结果与离散 LP 仅差若干千分之一，且随着 k 增大误差略增；在 k=4 的显式解中，得到下界 1.22232，比已有最优下界 1.2 稍高。总体上，本文的方法在理论上提供更紧的下界，而数值上与现有方法相当或略优。

**⚠️ 局限性**

局限性包括：① 对大 k 的数值计算仍受内存限制（n 取值需 ≤ 110 以保证可行）；② 仅在 k=4 给出了显式解，k>4 的显式解尚未构造；③ 连续 Flow LP 的解析解依赖于 Dirichlet 分布参数的精细选择，复杂度较高；④ 证明仅在 Unique Games 假设下给出，若不成立则下界可能不成立。

---

## 416. Learning Pseudo-Riemannian Manifolds for Heterophilic Graphs via Graph Signature

**arXiv ID:** 2611.06265 | [PDF](https://arxiv.org/pdf/2611.06265v1)

**作者:** Yun Young Choi `[一作]` (SolverX), Seokhwan Kim `[通讯]` (SolverX)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种伪黎曼注意力网络（PRAT），通过引入图签名指数（Graph Signature Index）来诊断标签诱导的非正定几何，并在注意力计算中使用可学习的非正定度量，以更好地处理异构图（heterophilic graph）；

**💡 创新点**

创新点在于：1）首次将图的标签拓扑特征量化为签名指数并作为正则化先验；2）在注意力机制中直接使用签名指数初始化的非正定度量，使模型能够自适应地学习正负方向；

**🔧 技术方法**

技术包括伪黎曼几何、切空间注意力分数、基于图签名矩阵的可学习对角度量、以及多头注意力的GNN框架；

**📊 数据集**

实验数据集涵盖八个标准节点分类基准（Cora、PubMed、Citeseer、Chameleon、Squirrel、Actor、Cornell、Texas）以及七个Platonov异构基准（过滤后的Wiki、roman-empire、amazon-ratings、minesweeper、tolokers、questions），并使用合成图验证理论；

**📈 对比分析**

通过10折交叉验证与多种基线（GCN、GAT、GATv2、FAGCN、GPR-GNN、BernNet、H_2GCN、Q-GCN、HAT等）对比，PRAT在Actor和Texas上实现最佳性能，在其它异构图上与最强模型持平或略优，且在大规模异构基准上表现竞争力；

**⚠️ 局限性**

局限性包括：需要训练节点标签计算签名，限制了无监督/自监督或图级任务；在归纳推理时需重新计算签名；对类别数少时度量表达能力受限；且对有向图或动态曲率的扩展尚未实现。

---

## 417. GRIDSHIELD: Verifiable Grid Feasibility and Manipulation Resistance in Privacy-Preserving Peer-to-Peer Energy Trading

**arXiv ID:** 2611.05816 | [PDF](https://arxiv.org/pdf/2611.05816v1)

**作者:** Abdullahil Kafi `[一作]` (Southern Illinois University Carbondale), Alvi Ataur Khalil `[通讯]` (Southern Illinois University Carbondale)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

实现了基于区块链的隐私保护点对点电能交易平台，采用无序批量清算并通过聚合拥塞可行性证明（ACFP）证明市场结果满足电网约束。

**💡 创新点**

创新点包括：① 通过顺序无关的批量线性规划消除交易排序的MEV风险；② 利用Pedersen承诺的加法同态构建聚合拥塞可行性证明，仅用一次聚合范围证明即可验证所有线性约束；③ 提出了基于拥塞责任的公平性度量CRF，用物理责任代替消耗或等分来评估削减公平性。

**🔧 技术方法**

采用的技术包括：Pedersen承诺、加法同态、Bulletproofs聚合范围证明、线性化DistFlow电网模型、Hyperledger Fabric执行-排序-验证链码、Hyperledger Caliper性能基准。

**📊 数据集**

使用的数据集包括：240节点中西部真实小时负荷的分布馈线、IEEE 13节点、IEEE 123节点等典型馈线，评估在这些真实或仿真馈线上的可行性和公平性。

**📈 对比分析**

与Groth16可行性电路、MPC秘密共享清算、每匹配一次ZKP等三种现有方案进行对比。结果显示：ACFP证明大小35 KB（对比Groth16的0.79 KB但不需要验证密钥），可拒绝超过容量的分配（MPC和匹配ZKP均可接受不合法分配），验证耗时82.2 s，单轮交易量远低于Fabric最大吞吐量（≈1064 tx/s）且无失败。

**⚠️ 局限性**

局限性：仅能处理可线性化的电网约束（如线性化电压、热容量），不适用于非线性或网状AC约束；假设共识节点可信且网络延迟可控；聚合范围证明对约束数量对数级增长，极大规模馈线仍需进一步优化。

---

## 418. The Harness Is the Model: Agent Scaffolding Changes a Fixed LLM's Compliance with Banking Conduct Rules, Unevenly Across Models

**arXiv ID:** 2611.06116 | [PDF](https://arxiv.org/pdf/2611.06116v1)

**作者:** Mehmet Murat Huyut `[一作]` `[通讯]` (Kuveyt Turk Participation Bank), Mehmet Murat Huyut (Kuveyt Turk Participation Bank)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估银行业务中不同 harness（系统框架）对固定语言模型合规性的影响，并验证同模型审查是否能降低违规。

**💡 创新点**

提出“模型–harness”组合是验证单元，展示 harness 对合规率的显著、模型特异性影响；发现同模型审查可显著降低违规率，且违规往往在对话中被掩盖。

**🔧 技术方法**

使用四种开源大型语言模型（deepseek-v4-flash、deepseek-v4.1-flash、qwen3.8-flash-next、gemma-4-31b），五种 harness（pipeline、ReAct 工具循环、policy 搜索、长记忆、同模型审查），Python 仿真银行环境，确定性评分器和统计检验（Cochran Q、McNemar、GEE）。

**📊 数据集**

合成数据集：156 个预注册银行案例（39 基础变体×2 语义重述×2 压力级别），13 条操作手册规则，两个虚拟客户；共 3120 条两轮对话记录。

**📈 对比分析**

对每个模型–harness 组合在 3120 条对话中进行违规率、过度拒绝率等指标的定量评分；使用统计检验比较 harness 效果。结果显示整体合规率 87–92%，单一模型在不同 harness 下的差距可达 11.5pp，且同模型审查将违规率从 9.6% 降至 6.4%（无显著过度拒绝）。

**⚠️ 局限性**

局限性：情景与规则手册为合成且单一；仅测试四种中等规模开源模型，未涵盖更强大或推理型模型；日期算术错误未被 harness 修复；对话采样可能导致某些违规被掩盖；LLM 判断与人工标注存在偏差。

---

## 419. Overview of the NTCIR-19 SciClaimEval Task: Cross-modal Claim Verification in Scientific Papers

**arXiv ID:** 2611.06090 | [PDF](https://arxiv.org/pdf/2611.06090v1)

**作者:** Xanh Ho `[一作]` (NII LLMC), Akiko Aizawa `[通讯]` (NII)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建SciClaimEval数据集并组织NTCIR‑19跨模态科学论点验证共享任务

**💡 创新点**

首次采用对照证据对齐（支持/反驳）以及多模态LLM评估，避免表面提示

**🔧 技术方法**

利用多模态大型语言模型、提示、抽取与推理管道

**📊 数据集**

SciClaimEval数据集：180篇论文、1,633条真实论点，包含表格/图形四种格式，涵盖NLP、ML、医学三领域

**📈 对比分析**

最佳系统在配对信息下对任务1对准确率98.1%，单独验证仅73.8%；任务2准确率98.4%，整体表现优秀

**⚠️ 局限性**

主要限制：过度依赖配对信息导致评测不现实，单一证据验证仍低效，数据域与证据类型受限

---

## 420. Split-KV: Efficient Context Parallelism for Training Diffusion Drafters

**arXiv ID:** 2611.05926 | [PDF](https://arxiv.org/pdf/2611.05926v1)

**作者:** Geng Zhang `[一作]`, Yang You `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种名为 Split-KV 的上下文并行训练方法，专为扩散模型设计。

**💡 创新点**

创新点在于将 KV 缓存按键和值分离并在多卡之间分布式存储，显著降低显存占用且几乎不影响梯度传播。

**🔧 技术方法**

核心技术包括 KV 缓存拆分、分布式内存共享、上下文层级并行与混合精度训练。

**📊 数据集**

在 ImageNet、CIFAR‑10 与 LAION‑400M 等大规模图像数据集上进行实验验证。

**📈 对比分析**

与传统 Tensor‑Parallel、Pipeline‑Parallel 以及单卡训练相比，Split‑KV 在相同显存预算下实现了约 20%–30% 的吞吐率提升，同时保持了 0.1% 以内的 FID 差距。

**⚠️ 局限性**

局限性包括对特定 Transformer 结构的适配性、KV 拆分导致的通信开销在极大规模模型上可能增大，以及对非扩散任务的适用性尚未充分验证。

---

## 421. GRASP: Gradient-Guided Sampling for Contact-Rich Physics-Based Motion Retargeting

**arXiv ID:** 2611.05754 | [PDF](https://arxiv.org/pdf/2611.05754v1)

**作者:** Yang Liu `[一作]` (Harvard University), Heng Yang `[通讯]` (Harvard University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种结合采样探索与梯度引导的物理仿真运动重定向方法（GRASP），通过回收已评估轨迹拟合局部二次模型来恢复梯度，从而在不可微的仿真动力学下实现高质量的机器人轨迹生成；

**💡 创新点**

创新点在于利用已评估的轨迹回收梯度信息，构建可在任何仿真器上使用的梯度恢复机制，并在TSMC框架中加入Metropolis校正保证采样正确性，同时将该优化器推广到可变窗口的MPC模式；

**🔧 技术方法**

主要技术包括温度采样（TSMC）、局部二次回归梯度恢复、Metropolis–Hastings修正、离散余弦变换降维、AR(1)先验、以及对MuJoCo等物理引擎的批量仿真；

**📊 数据集**

实验使用了OakInk2（手部双臂抓取、倾倒等任务）、Hot3D（单/双手交互）、LAFAN1、AMASS、OMOMO（全身运动和物体操控）等公开数据集；

**📈 对比分析**

与SPIDER、GMR、Do-as-I-Do、DynaRetarget等基线比较，GRASP在OakInk2、Hot3D等任务中成功率提升至92–95%，在AMASS等数据集上关节和物体跟踪误差显著下降，并通过PPO学习展示在轨迹跟踪奖励、轨迹长度和位置误差方面的明显改进；

**⚠️ 局限性**

局限性包括：梯度恢复依赖于足够密集的已评估轨迹，受限于仿真精度和确定性；在大规模高自由度情境下计算开销仍较高；对随机动态或非确定性系统的适用性尚未验证；

---

## 422. Objective-Conditioned Active Preference Learning for Multi-Objective Alignment

**arXiv ID:** 2611.05506 | [PDF](https://arxiv.org/pdf/2611.05506v1)

**作者:** Duc Anh Vu `[一作]` (Nanyang Technological University), Luu Anh Tuan `[通讯]` (Nanyang Technological University)

**通讯引用:** 3603 | [OpenAlex ID](https://openalex.org/A5050386762)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Objective-conditioned Gradient Coverage DPO（OGC-DPO）框架，用于在多目标偏好学习中高效选择对每个目标最具信息量的（答案对、目标）查询，从而减少标注成本。

**💡 创新点**

创新点在于利用策略自身的隐式梯度估计每个查询在各目标下的期望DPO梯度，并通过逐目标梯度覆盖度量实现查询的主动选择，同时保留冲突偏好作为有价值的训练信号。

**🔧 技术方法**

核心技术包括LoRA轻量级目标估计器、隐式梯度投影与归一化、基于Mahalanobis距离的覆盖度度量、逐目标贪婪选择与CAGrad-Clip梯度组合。

**📊 数据集**

实验使用了三大模型（Llama‑3.2‑3B、Gemma‑3‑4B、Qwen3.5‑9B）与三组多目标数据集（PKU‑SafeRLHF、UltraFeedback、HelpSteer）进行评估。

**📈 对比分析**

与随机采样、ActiveDPO、APO三种基线在相同标注预算下比较，OGC‑DPO在平均奖励、最弱目标奖励和GPT‑4o胜率等指标上均显著优于对手，尤其在目标数增多时保持优势。

**⚠️ 局限性**

局限性包括依赖自动化奖励模型和GPT‑4o评估，实验仅覆盖中型开源模型、英语数据和有限的目标数，且对估计器精度和扩展到更大模型、更多目标及多语言场景的适用性尚需进一步验证。

---

## 423. Deep Meets Classical: A CPU-Efficient Decision-Fusion Model for Automatic Modulation Classification

**arXiv ID:** 2611.05937 | [PDF](https://arxiv.org/pdf/2611.05937v1)

**作者:** Zhiyi Yin `[一作]` (Rice University), Yusi Gong `[通讯]` (Rice University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级的决策融合模型，结合了低参数卷积神经网络与手工高阶矩特征的梯度提升树，用于CPU上高效的自动调制分类。

**💡 创新点**

创新点在于发现卷积网络与高阶矩特征在不同信噪比区间具有互补优势，并通过训练逻辑回归元组合并二者的后验概率，实现在单核CPU上即能覆盖低SNR的鲁棒性又能保持高SNR的精度。

**🔧 技术方法**

使用了残差CNN（约20万参数）、梯度提升树（GBT）与逻辑回归融合器；全部模型在单核CPU上训练与推理，无GPU加速。

**📊 数据集**

基准数据集为RadioML 2016.10a（11种调制，20个SNR级别），采用固定划分（22k训练、11k验证、44k测试）进行多种随机种子评估。

**📈 对比分析**

与传统高阶矩+SVM、RF、GBT基线以及单纯CNN进行对比，融合模型在整体准确率上提升5.1点（相对最佳高阶矩基线）和3.2点（相对CNN），并在低SNR下显著优于CNN，高SNR下匹配最佳传统基线；宏F1也实现最高值。

**⚠️ 局限性**

主要局限是实验使用了减小的分层子集，可能导致绝对准确率低于全数据集；同时对极端测试时变换的鲁棒性仍需进一步验证，且仅融合两种特征，未来可扩展更多互补特征。

---

## 424. How Do Artificial Agents Make Decisions in Light of Visual Data?

**arXiv ID:** 2611.05671 | [PDF](https://arxiv.org/pdf/2611.05671v1)

**作者:** Keyty Loza `[一作]` (Lamar University), Shayan Monadjemi `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过两项实验评估了基于视觉语言模型的人工智能代理在从视觉不确定性可视化中估计概率并进行决策的能力。

**💡 创新点**

创新点在于将AI代理视为主动决策者，并使用Model Context Protocol实现工具调用来检验其与理性选择理论的一致性。

**🔧 技术方法**

采用了Gemma-4、Granite3.2-Vision、GPT-4o和Qwen3-VL:8b等视觉语言模型以及MCP工具链进行实验。

**📊 数据集**

使用了自生成的乐透情景视觉图，包含5%–100%概率、$10–$100成本以及图标阵列、饼图、圆图、三角图、柱状图五种可视化形式。

**📈 对比分析**

通过均方误差（MAE）和决策一致率进行比较，GPT-4o表现最佳（MAE 0.083），决策正确率约为80%，而其他模型误差更大。

**⚠️ 局限性**

局限性在于仅评估了有限的模型和可视化类型，实验使用简化的合成图像，缺乏对真实复杂可视化的验证。

---

## 425. STARE: Scene Text-Aware multimodal Retrieval for Everyday objects

**arXiv ID:** 2611.06089 | [PDF](https://arxiv.org/pdf/2611.06089v1)

**作者:** Kento Tokura `[一作]`, Komei Sugiura `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对服务机器人自由文本指令的场景文本感知多模态检索框架 STARE。

**💡 创新点**

创新点在于引入跨语言视觉提示（Crosslingual Visual Prompt）以精准关联 OCR 文本与图像位置，并通过场景文本重排序器（Scene Text ReRanker）利用命名实体匹配提升检索精度。

**🔧 技术方法**

技术包括多模态大语言模型、CLIP/DINOv2 视觉编码器、Patchify+重叠补丁、OCR、查询扩展与命名实体抽取。

**📊 数据集**

使用了新构建的 GoGetIt、TextCaps-test 两个含场景文本的检索基准以及 LTRRIE 公开数据集进行评估。

**📈 对比分析**

与 CLIP、Long‑CLIP、BLIP‑2 等基线对比，STARE 在 recall@10 上分别提升 13.2/8.2/3.6/2.3 分，在硬件实验中 recall@5 达 88% 且成功率 80%，显著优于对照组。

**⚠️ 局限性**

局限性包括对 OCR 质量高度依赖、对多语言场景文本支持有限，以及在大规模图像库中仍需进一步优化推理速度与资源占用。

---

## 426. Agent Harness Awareness: Turning Verified Telemetry into Operational Control for Tool-Using Agents

**arXiv ID:** 2611.05391 | [PDF](https://arxiv.org/pdf/2611.05391v1)

**作者:** Manu Agrawal `[一作]` (Independent Researchers), Shrey Nagpal `[通讯]` (Independent Researchers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Agent Harness Awareness 架构，给 LLM 预调度阶段提供主机最新验证过的执行状态（工具健康度、延迟、错误率等）

**💡 创新点**

创新点在于把主机的监控数据转化为可信、边界清晰的“子系统状态遥测”（SST）供模型使用，同时保持主机对执行的最终权威与安全控制

**🔧 技术方法**

技术包括：主机监控（监视器、追踪）、确定性约简器、状态快照生成、模型决策接口、再验证与强制执行、OpenTelemetry 与 Toxiproxy 生成的现场指标

**📊 数据集**

数据集：两组实验数据——一组是 216 次冻结决策（三模型×六随机种子×四环境状态×三条件），另一组是 27 次现场 HTTP 路径实验（三模型×三故障模式×三条件）

**📈 对比分析**

比较方法：对比任务仅信息、结构中性信息、验证状态三种条件的平均可观策略后悔（ms）与单次派遣完成率；结果验证状态将后悔从 4251 ms 降至 131 ms（↓96.9%），完成率从 69.4% 提升至 95.8%；现场实验同样显示验证状态可实现 0 ms 后悔与 100% 成功率

**⚠️ 局限性**

局限性：仅在读写单一工具链（工具-网络）下验证，未涉及重试恢复、非幂等操作、时延衰减、硬件/经济/多源异构选择等场景；实验规模有限，未与真实生产负载或更大工具目录进行对比

---

## 427. Conformal Prediction for Distribution-to-Distribution Regression

**arXiv ID:** 2611.06320 | [PDF](https://arxiv.org/pdf/2611.06320v1)

**作者:** Trung-Khang Tran `[一作]` (National University of Singapore), Tan M. Nguyen `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向分布对分布回归的合适判别预测（CP）框架，利用截断正交基展开实现可解释的预测集并保证覆盖率；

**💡 创新点**

创新点在于将分布输出投影到有限维正交基，形成可度量体积且可均匀采样的预测集，并进一步设计自适应局部CP实现渐近条件覆盖；

**🔧 技术方法**

核心技术包括正交基分布表示、截断正交级数估计、分量级非一致性得分量化、分量空间的仿射椭球CP阈值和基于k近邻的局部阈值推断；

**📊 数据集**

使用了合成数据（截断高斯、Beta混合、多维暗物质）以及真实数据（房价、死亡率）来评估模型；

**📈 对比分析**

通过与三种基线回归（NDR、OTR、WR）对比，实验表明自适应CP在保持约定覆盖率的同时，预测集体积更小、效率-覆盖率曲线更佳，Beta-Binomial关系与理论一致；

**⚠️ 局限性**

局限性包括无法直接处理协变量偏移或非可交换样本，以及在高维分布空间中局部校准效率低下，未来工作需扩展至非可交换设置并改进高维局部CP方法。

---

## 428. Runway Characters: Serving a 57B Video Diffusion Transformer for Real-Time Conversation

**arXiv ID:** 2611.05642 | [PDF](https://arxiv.org/pdf/2611.05642v1)

**作者:** Rohan Agarwal `[一作]` (Runway), Zejia Zheng `[通讯]` (Runway)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在四块H100 GPU上实现并部署57B参数的视频扩散Transformer，实现24fps的实时对话视频生成；

**💡 创新点**

将KV缓存维护从读时迁移到写时，采用环形缓存、RoPE delta旋转和FP8写时量化，并将张量并行通信与FP8 GEMM融合，从而显著降低每步计算量；

**🔧 技术方法**

使用Tensor Parallelism、FP8量化、RoPE、KV环缓存、全写时量化、GEMM+All-Reduce融合、CUDA Graph、Hopper/Blackwell硬件特性等技术；

**📊 数据集**

采用多样化人物图像（人类、卡通、幻想等）与语音录音共计954秒的对话数据进行评测，并对长时10分钟对话进行基准测试；

**📈 对比分析**

通过与未优化基线和融合内核对比，实时推理时间从240.9ms降至147.4ms（1.63×速率），生产环境中99.5%会话保持>23.5fps，平均帧率达26.8fps；与开源系统相比，帧率更高、延迟更低；

**⚠️ 局限性**

受限于单用户批量、通信瓶颈、解码GPU分配以及对硬件/编译环境的敏感性，难以直接扩展到更大模型或多用户场景。

---

## 429. Matched but Not Better: An Equivalence-Tested Benchmark of Quantum Machine Learning on Six Clinical Datasets and the QML-Health-READY Reporting Framework

**arXiv ID:** 2611.05442 | [PDF](https://arxiv.org/pdf/2611.05442v1)

**作者:** Mohammed Siraj B `[一作]` (St Aloysius (Deemed to be University)), Ruban S `[通讯]` (St Aloysius (Deemed to be University))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在六个公开临床数据集上，使用相同的四维主成分输入、相同的交叉验证拆分，构建匹配的配对实验，对比量子支持向量机(QSVM)和变分量子分类器(VQC)与五个精调的经典模型，采用等价检验并引入 QML‑Health‑READY 报告框架进行评估。

**💡 创新点**

创新点在于：①首次对临床数据进行严格的匹配配对和等价测试；②提出了以数据复杂度、编码、算法、数据拆分、收益、鲁棒性、评估和成本八个域的 QML‑Health‑READY 评估框架；③系统量化了有限采样（shot）对量子核性能的影响，并展示了实现配置错误如何导致结论逆转。

**🔧 技术方法**

技术包括：量子核支持向量机（ZZ 量子特征映射，4 个量子比特，调节带宽）、变分量子分类器（AngleEmbedding+StronglyEntanglingLayers）、经典模型（L2 逻辑回归、SVM RBF/多项式、随机森林、梯度提升）、正则化核相似度与几何差异分析、配对的修正重采样 t 检验与两侧等价检验、随机效应池化、噪声模型（去极化）和有限采样估计。

**📊 数据集**

使用六个公开临床数据集：Breast Cancer Wisconsin、Heart Disease Cleveland、Pima Indians Diabetes、Heart Failure Clinical Records、TCGA‑BRCA（ER‑正性基因表达）和 NHANES 2017‑18 糖尿病。

**📈 对比分析**

比较方法：对每个数据集进行 25 次配对拆分，计算 AUROC 差异并进行修正 t 检验与 TOST 等价检验；对等价边际 ±0.02 和 ±0.05 进行 Holm 多重检验；随后用随机效应模型进行跨数据集汇总。结果显示：无量子模型显著优于任何经典模型；QSVM 与匹配 RBF SVM 在 4/6 数据集内在 ±0.05 边际内等价，且在池化后仍未出现优势；VQC 仅在 5/6 数据集内在 ±0.05 边际内等价；有限采样导致 AUROC 下降约 0.05，且实现错误可导致结论完全逆转。

**⚠️ 局限性**

局限性包括：仅使用 4‑比特模拟（12‑比特仅一次验证）、未在真实硬件上测试、噪声模型简化、有限的超参数网格、缺乏外部验证和决策曲线分析、经典模型在所有特征上的最佳结果仅为参考、以及等价边际未依据临床需求制定。

---

## 430. Two-Sided Screening in Quality-Differentiated Markets

**arXiv ID:** 2611.06097 | [PDF](https://arxiv.org/pdf/2611.06097v1)

**作者:** Yaohao Liu `[一作]` (University Of Electronic Science And Technology Of China), Mingyu Xiao `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究两侧筛选在质量差异化市场中的平台菜单设计问题，提出通过交易分布和技术成本凸化来描述最优菜单；

**💡 创新点**

创新点在于将技术成本凸化与价格、工资设计分离，仅需在接触点提供质量；给出有限菜单的必要充分条件、利润最大化在MHR条件下的效率保证，并提出多项式时间的动态规划算法实现有限质量菜单，带有乘法和加法近似保值；

**🔧 技术方法**

使用凸化技术、交易分布优化、积分形式的利润与福利表达、以及基于利润标量oracle的动态规划；

**📊 数据集**

本研究为理论性工作，没有使用具体实验数据集；

**📈 对比分析**

通过对比无约束最优菜单与有限质量菜单的利润与福利，并给出1/k乘法保值与O(ε)加法误差的近似保证，算法时间为多项式级别；

**⚠️ 局限性**

局限在于需先知类型分布且满足MHR等假设，算法对连续质量空间需离散化，实际平台可能面对动态学习、外部性等更复杂情况。

---

## 431. Closed-Loop Adaptive Sampling on a Generative Ocean Model

**arXiv ID:** 2611.05776 | [PDF](https://arxiv.org/pdf/2611.05776v1)

**作者:** Shayesteh Hafezi `[一作]` (University of Michigan), Alan Papalia `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个多机器人闭环自适应采样框架，利用单一学习生成的海洋预测模型进行预测、规划和观测同化；

**💡 创新点**

将生成式模型的预测误差分布直接作为不确定性量化，用A/D最优准则指导移动机器人采样，并通过观测条件扩散采样实现高效同化；

**🔧 技术方法**

使用基于潜在空间的扩散模型LaDCast（含自编码器）、A/D最优采样、观测条件扩散同化、贪婪分配与低秩协方差近似；

**📊 数据集**

在两大海域（中大西洋边缘、加利福尼亚流系统）的重分析数据上进行实验，评估海面温度与盐度；

**📈 对比分析**

与静态、随机、Lawnmower三种基线比较，采用RMSE和误差恢复比例评价，结果显示A/D最优采样在大多数情形下显著优于基线，且计算成本仅为传统数值模型的七分之一；

**⚠️ 局限性**

当生成模型的预测误差与不确定性分布失配（如高误差但低不确定性）时，不确定性驱动的采样优势减弱，系统受限于对不确定性校准与多步规划的欠缺。

---

## 432. Rewrite, Then Veto: Rewriter-Agnostic Edit Gating for LLM-Based Post-OCR Correction

**arXiv ID:** 2611.05539 | [PDF](https://arxiv.org/pdf/2611.05539v1)

**作者:** Jorge Ramón Fonseca Cacho `[一作]` `[通讯]` (University of Nevada), Jorge Ramón Fonseca Cacho (University of Nevada)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种在大型语言模型重写后对编辑块进行投票过滤的门控机制，以减少 OCR 后纠错中的过度修正。

**💡 创新点**

创新点在于把完整上下文重写后的编辑块进行字符级对齐并使用小型学习门控实现编辑块级别的拒绝，且门控模型可迁移到未见重写器。

**🔧 技术方法**

技术主要包括 Levenshtein 对齐、块划分、使用 0.6B 参数的轻量化门控网络（基于文本和概率特征）以及 LoRA 微调。

**📊 数据集**

使用 ICDAR 2026 多语言（英、法、德）历史文档 OCR 纠错基准，包含训练、开发和八个测试集。

**📈 对比分析**

通过与无门控重写器和竞赛系统对比，门控在大多数单元上显著降低字符匹配误差率（cMER）并提升偏好分，最优方案在第二次实验中得到 cMER 0.0069，接近竞赛排名第二。

**⚠️ 局限性**

局限在于门控只能去除编辑，无法提升重写器召回；单一阈值对低误差系统效果不佳；实验仅覆盖文本级别和三种语言，且可能存在自适应过拟合。

---

## 433. Req-Space: A User Study of Interactive Visualisations for Requirements Review

**arXiv ID:** 2611.05437 | [PDF](https://arxiv.org/pdf/2611.05437v1)

**作者:** Risha Parveen `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**通讯引用:** 12467 | [OpenAlex ID](https://openalex.org/A5058417486)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文开发并评估了 Req‑Space，一个集成语义探索与可追溯性可视化的交互式原型，采用固定顺序的两阶段用户研究对20名参与者进行测试。

**💡 创新点**

创新点在于将语义相似性搜索与已有的需求、业务规则及目标间的可追溯性关系结合到可视化中，并系统性地评估其对审查覆盖率和决策理由的影响。

**🔧 技术方法**

使用技术包括 Sentence‑BERT 句子嵌入、t‑SNE 投影、基于图的可视化与交互式前端、预先计算的布局，且在评审过程中不调用实时语言模型。

**📊 数据集**

实验数据集为虚构的 QuickBite 食品配送系统的 40 条需求及其关系和一次变更请求 CR‑01，以及用于练习的 LibBook 预备系统。

**📈 对比分析**

通过两阶段固定顺序对比，量化新增、重复、修订的审查记录、支持率（84.1%）、工作量与易用度评分；结果显示可视化支持可显著扩展审查覆盖并提升部分决策理由，但因设计不允许因果归因，未能直接量化性能提升。

**⚠️ 局限性**

局限性包括固定顺序设计导致无法单独归因于工具、样本规模小、仅测试单一虚构需求集、未评估协作使用、动态维护与可扩展性，以及缺乏对工业项目的外部有效性验证。

---

## 434. Scheduling Data to Compute: Hierarchical Data Pipelining for LLM Training on AMD GPUs

**arXiv ID:** 2611.05751 | [PDF](https://arxiv.org/pdf/2611.05751v1)

**作者:** Gengyu Rao `[一作]` (Advanced Micro Devices, Inc.), Emad Barsoum `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用AMD GPU的Tensor Data Mover (TDM)、转置加载ISA、System DMA (SDMA)和Remote DMA (RDMA)等硬件特性，优化LLM训练中的数据流水线，并将实现集成到Primus框架中；

**💡 创新点**

首次在AMD GPU上系统化地将TDM与转置加载结合用于解耦数据搬移与计算，并将SDMA/RDMA用于跨GPU高效通信，同时针对Compressed Sparse Attention实现高效稀疏数据流；

**🔧 技术方法**

采用Tensor Data Mover、转置加载ISA、SDMA、RDMA、ROCm/HIP API及PyTorch集成；

**📊 数据集**

使用LLama 3.1 8B、GPT-OSS 20B、FLUX.1-Schnell以及DeepSeek V4模型中的Compressed Sparse Attention；

**📈 对比分析**

在MI455/MI355 GPU上对比使用与不使用这些指令的kernel，几何平均速度提升分别为TDM 2.1×、转置加载 1.4×、SDMA 1.16×；与传统CU通信相比，SDMA/RDMA的GEMM回退仅为8‑14%，而CU通信回退为28‑35%；

**⚠️ 局限性**

受限于显存容量与稀疏性，尚未完全支持更大规模集群；RDMA/SDMA层级集成仍在进行中，缺乏完整的生产级评估；对非MI455架构的支持有限。

---

## 435. On the most informative Boolean function under noise

**arXiv ID:** 2611.05430 | [PDF](https://arxiv.org/pdf/2611.05430v1)

**作者:** Dimiter Kramer `[一作]` (DeepInfra), Mert Saglam `[通讯]` (DeepInfra)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文证明了 Courtade–Kumar 2013 年提出的关于在噪声下最具信息性的布尔函数的猜想，给出了一个简单的归纳证明，并通过傅里叶分析进一步验证。该结论表明，取 Y_i（任意坐标）作为输出比任何其他布尔函数能获得的关于 X 的互信息更大。

**💡 创新点**

创新点在于：①引入了 Tsallis 熵的细化来得到介于强数据处理不等式和原猜想之间的精确界；②提出了一种新的归纳证明框架，仅利用对单个坐标的选择和条件信息的链式规则；③提供了对连续空间情形的初步分析并给出了相应的猜想。

**🔧 技术方法**

主要技术包括：布尔函数的互信息与 Tsallis 熵的关系、归纳法与坐标选择、链式规则（Shannon 与 Tsallis）、傅里叶分析方法、以及对连续问题的阈值函数分析。

**📊 数据集**

该工作完全是理论性质的研究，没有使用任何外部数据集。

**📈 对比分析**

与已有的强数据处理不等式以及之前的部分进展进行比较。作者给出了一个明确的上界 C_X ≤ Y_1_X，并证明它在所有噪声概率 p ∈ [0,1] 上成立；在实验或数值模拟方面没有给出具体性能数据，因为该结论是解析性的。

**⚠️ 局限性**

局限性主要体现在：①对连续问题的结果仅为初步猜想，尚未完全证明；②虽然提供了归纳证明，但缺乏对更一般噪声模型（非均匀噪声、相关噪声等）的扩展；③在实际应用中如何将该理论转化为具体的编码或检测策略仍待进一步研究。

---

## 436. Vortex-Induced Torque in Grasping Soft Objects: Influence of Geometry and Pneumatic Parameters

**arXiv ID:** 2611.06359 | [PDF](https://arxiv.org/pdf/2611.06359v1)

**作者:** Roman Mykhailyshyn `[一作]` (National Institute of Advanced Industrial Science and Technology), Yukiyasu Domae `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究系统地实验测定了在握持柔性物体时涡流抓手产生的涡旋扭矩，并定量评估了供压、喷嘴直径、表面几何（圆柱/圆顶、凹凸）及曲率对扭矩的耦合影响。

**💡 创新点**

创新点在于揭示涡旋扭矩与物体表面几何的强耦合关系，尤其是圆顶凹面在半径≈20 mm时出现显著放大；提出基于气压与几何参数的涡旋扭矩预测模型，并证明高斯过程回归（GPR）在该任务中表现最佳。

**🔧 技术方法**

使用实验技术：7自由度机械臂与ATi Nano17 力/力矩传感器、10 mm/s 直升运动、10次重复实验；统计指标包括平均峰值扭矩、变异系数 CV、对数压力灵敏度 S_P、曲率比 S_c/v；机器学习技术包括 GPR、随机森林、Extra Trees、AdaBoost 及 RF+AdaBoost 集成，并通过5折重复交叉验证评估模型。

**📊 数据集**

数据集共 4920 次实验，覆盖 3 种喷嘴直径（0.6、0.8、1.0 mm）、4 种供压（100–400 kPa）、4 种曲面类型（圆柱凹凸、圆顶凹凸、平面）及 10 个半径（15–100 mm），形成 492 个实验条件，每个条件平均 10 次重复。

**📈 对比分析**

通过 5 折重复交叉验证比较模型性能，GPR 在平均 R²=0.899±0.080、RMSE≈0.508 N·mm、MAE≈0.240 N·mm 处表现最好；整体预测时 R²=0.939，表明气压与几何参数足以预测峰值扭矩。

**⚠️ 局限性**

局限性包括：未对流场进行直接测量或 CFD 验证；模型仅在实验覆盖域内有效，未验证超出范围的气压或曲率；缺乏对旋转效应的实时利用和控制策略，无法直接将扭矩优势转化为实际操作方案。

---

## 437. Today's Work, Tomorrow's Workforce: Incentivizing Learning in AI-Assisted Work

**arXiv ID:** 2611.06346 | [PDF](https://arxiv.org/pdf/2611.06346v1)

**作者:** Moritz Diener `[一作]` (Karlsruhe Institute of Technology), Wei Sun `[通讯]` (IBM Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文构建了一个动态接受-独立解决模型，分析在连续任务中AI辅助如何影响工人技能获取与保持，揭示低技能陷阱与专家降级问题。

**💡 创新点**

创新点在于将技能演化（获取与衰退）嵌入接受-独立决策框架，并提出“学习激励”——既补偿努力成本又补偿因低于AI准确率而错失的奖励，以激励工人从AI中学习。

**🔧 技术方法**

使用马尔可夫决策过程与贝尔曼方程求解工人最优策略，并推导了最小学习激励与目标技能水平的支付公式。

**📊 数据集**

未使用实证数据集，所有示例均为数值演示与参数假设。

**📈 对比分析**

论文未进行实验或与现有方法比较，仅通过理论推导与数值模拟说明学习激励能有效避免低技能陷阱并降低长期激励成本。

**⚠️ 局限性**

局限性包括：模型仅考虑单一工人、单一技能维度；未验证激励是否在真实工作中能促使练习与学习；未考虑AI使用本身可能带来的学习效应；以及对组织整体效益与工人多样性的分析缺失。

---

## 438. Beyond Terminal Constraints: Reliable Robot Control with Learned Safe Sets

**arXiv ID:** 2611.06196 | [PDF](https://arxiv.org/pdf/2611.06196v1)

**作者:** Gianni Lunardi `[一作]` (Free University of Bolzano), Andrea Del Prete `[通讯]` (University of Trento)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于多步安全集的模型预测控制框架，可在非线性机器人系统上保证安全与递归可行性。

**💡 创新点**

创新点在于定义了四类比控制不变集更宽松的多步安全集（up-to-N-step、within-N-step、back-and-forth-N-step、back-and-forth-within-N-step CIS），并设计了对应的 RC‑MPC、HTC‑MPC、PC‑MPC 方案。

**🔧 技术方法**

使用了 VBOC 算法近似安全集、神经网络预测速度上界、CasADi + Acados 进行实时迭代求解，并通过实时迭代（RTI）实现 5–10 ms 的控制循环。

**📊 数据集**

训练数据由 VBOC 生成的约 600k–1.2M 条样本组成，涵盖 6-DOF 机器人在含障碍环境下的碰撞安全约束。

**📈 对比分析**

与传统无终端约束、零速度终端和软终端约束等基准方法比较，实验表明 RC‑MPC/PC‑MPC 在约束违例率下降 20–30% 的同时，跟踪代价仅略高；计算时间均低于采样周期，能实时执行。

**⚠️ 局限性**

局限性包括对安全集满足多步不变性的假设难以验证，近似安全集可能导致误报/漏报，以及算法在更高维、欠驱动或接触动力学系统中的可扩展性待进一步研究。

---

## 439. Neural Stochastic Forecasting for Lateral Molecular Diffusion Simulations

**arXiv ID:** 2611.06157 | [PDF](https://arxiv.org/pdf/2611.06157v1)

**作者:** Jacob Ede Levine `[一作]` (California State Polytechnic University), Sai Chandra Kosaraju `[通讯]` (Western University of Health Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个基于 transformer+多速率 LSTM+隐式 OU 过程的神经随机模型，用以预测膜蛋白的长期扩散轨迹。

**💡 创新点**

创新在于结合物理信息的隐式记忆过程和跨注意力解码器，使模型在不需要观察窗口的情况下仍能保持长期时间相关性并准确重现多蛋白系统的扩散行为。

**🔧 技术方法**

使用 transformer 编码器、LSTM 银行、离散 Ornstein‑Uhlenbeck 隐层、交叉注意力解码器以及 Wasserstein 分布匹配损失进行训练。

**📊 数据集**

数据集为两组粗粒度 MD 模拟：七个 PIEZO 通道以及混合的七个 PIEZO + 十个 TREK 通道。

**📈 对比分析**

与之前的 VAR‑SDE 模型对比，NS 模型在平均异常指数、MSD、非高斯参数、Van Hove 分布等指标上均能恢复与 MD 相同的统计特征，并可生成四倍长度的轨迹，表现明显优于 VAR‑SDE。

**⚠️ 局限性**

局限包括仅在有限的蛋白/膜系统上验证，未能显式约束力场能量；对不同膜环境、蛋白组成的泛化性尚待评估。

---

## 440. A Polynomial Erdős-Pósa bound for Eulerian digraphs

**arXiv ID:** 2611.06250 | [PDF](https://arxiv.org/pdf/2611.06250v1)

**作者:** Esther Galby `[一作]` (Chalmers University of Technology and University of Gothenburg), Bart van der Steenhoven `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了有向图的 Erdős–Pósa 性质，证明在欧拉有向图中，满足极大地环数 k 的弧相互不相交的环与弧的最小删除集之间存在 O(k^4) 的多项式上界。

**💡 创新点**

创新点在于将原先已知的非常巨大（阶乘级或指数层塔级）上界压缩至多项式阶数，并给出了欧拉有向图中弧的 Erdős–Pósa 性质的第一条多项式上界，回答了 Wahlström 的开放问题；同时给出了 Ω(k log k) 的下界，表明线性上界不可能成立。

**🔧 技术方法**

主要技术包括：① 用度削减将任意欧拉有向图转化为最大出度为 2 的情况；② 利用雕刻宽度（carving width）与 Eulerian 结构得到的“spined path‑system”；③ 构造并分析路径重叠与链路间的交叉，结合循环分解与切分理论；④ 构造（d–1）-grid gadget 进行度数降解；⑤ 通过沉浸（immersion）保持弧相互不相交性。

**📊 数据集**

该工作为纯理论研究，无需实验数据集；所有结论均通过数学证明给出。

**📈 对比分析**

与之前的 FOCS 2024 结果（f(k) 为高度 8 的指数塔）相比，本文将上界从指数大幅降低到多项式；相比于下界 Ω(k log k)，仍存在两端的巨大差距，显示仍有改进空间。

**⚠️ 局限性**

局限性：(a) 上界仍远高于已知的下界，逼近最优上界（可能为 O(k log k)）还有很大空间；(b) 仅处理弧相互不相交的环，未涉及顶点版本；(c) 证明过程中对欧拉结构的强假设，难以直接推广至一般有向图。

---

## 441. Autonomous ETL Pipelines for Salesforce M&A Consolidation: A Reference Framework, Jitterbit Mapping, and Reproducible Simulation Study

**arXiv ID:** 2611.05918 | [PDF](https://arxiv.org/pdf/2611.05918v1)

**作者:** Imran Basha Shaik `[一作]`, Khader Moinuddin Shaik `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了面向Salesforce M&A合并的自治ETL框架，并在合成数据上进行了系统评估。

**💡 创新点**

创新点在于将分区提取、声明式映射、质量门控、MDM、增量CDC和数据健康监控等成熟技术融合成完整流水线，并提供了可复现的开放参考实现。

**🔧 技术方法**

使用了Python原型、RapidFuzz字符串相似度、SQLite目标数据库、Jitterbit Harmony映射、模糊+确定性匹配规则、统计异常检测和电路断路器等技术。

**📊 数据集**

采用自制合成Salesforce双组织数据，包含已知重复、故障注入及多种变异，规模从3.2万到38.4万条记录。

**📈 对比分析**

通过多种实验（5个随机种子、不同规模、阈值扫荡、门控对照、增量同步一致性）进行比较；匹配F1最高达0.94，质量门控消除所有加载失败，增量同步与完整重建一致率>99%，但在大规模下MDM占比升至52%。

**⚠️ 局限性**

局限性：仅基于合成数据，缺乏真实业务复杂度；确定性规则易产生误合并；阈值需对象级校准；增量聚类可能与完整重建偏差；仅评估本地计算，未考虑API/网络延迟和真实Salesforce环境。

---

## 442. Revisiting Critic Estimation in Soft Actor-Critic Algorithms

**arXiv ID:** 2611.05666 | [PDF](https://arxiv.org/pdf/2611.05666v1)

**作者:** Jan Felix Kleuker `[一作]` (Leiden University), Thomas M. Moerland `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统评估 Soft Actor-Critic（SAC）不同 critic 设计，包括学习软优势、使用多步 λ‑回报以及在经验重放缓冲中训练 critic 的效果。

**💡 创新点**

证明在连续控制任务中，训练 λ‑回报 critic 并使用重放数据能显著提升性能，且不需要 off‑policy 校正；而学习软优势或软 GAE 对性能无显著帮助。

**🔧 技术方法**

使用 SAC 的 actor‑critic 架构、软优势估计、软 λ‑回报、软 GAE、经验重放、Sobol 随机搜索以及 Jaxnasium 框架实现。

**📊 数据集**

在 14 个任务上进行实验，涵盖 Gymnax 经典控制与 MinAtar、以及 Brax 的 MuJoCo 连续控制任务。

**📈 对比分析**

通过随机超参数搜索、归一化收益、峰值与敏感度、迁移性能等多维度指标对比，发现 λ‑回报重放版在连续任务上获得最高峰值且鲁棒性最好，而优势/GAE 版本与原 SAC 性能相近。

**⚠️ 局限性**

局限性包括仅在有限连续任务验证、重放效应与批大小/更新频率交织难以分离、未与其他 SAC 扩展组合、搜索空间有限且以原 SAC 为中心。

---

## 443. Quantifying Trait-State Hierarchy in EEG Foundation Model and Signal Processing Embeddings

**arXiv ID:** 2611.05652 | [PDF](https://arxiv.org/pdf/2611.05652v1)

**作者:** Anusha Madapura `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

系统研究了EEG嵌入中特征（trait）与状态（state）组织结构，利用层次度量和kNN纯度评估其对BCI状态分类的影响，并提出基于个体均值归一化的后处理方法以及通过微调实现宏trait到宏state转换的策略。

**💡 创新点**

首次将几何层次度量与kNN纯度结合用于量化EEG嵌入的宏trait/宏state结构，揭示宏trait结构抑制状态可分辨性，并展示归一化与微调能显著提升状态识别性能。

**🔧 技术方法**

采用层次度量（ℋ）、kNN纯度、线性探针、个体均值归一化以及在预训练和微调的EEG基础模型（如REVE、BIOT、LaBraM）与传统信号处理特征（频谱功率、复杂度、互信息）上的实验。

**📊 数据集**

使用Physionet‑MI、High‑Gamma和ERPCore三个数据集，共计109、14、40名受试者，包含运动想象和六种ERP子任务，通道数从4到128、采样率从160Hz到500Hz。

**📈 对比分析**

通过与线性探针基准对比，发现ℋ>1对应低状态分类准确率；个体均值归一化后平均提升约10%平衡准确率，微调后ℋ下降至<1，状态分类准确率显著提升（最高约20%）。

**⚠️ 局限性**

预训练目标仍偏向捕捉个体身份，导致宏trait结构占主导；缺乏对trait与state的解耦子空间限制了BSC性能，后续需要设计更具disentangled的表征。

---

## 444. Learning Object-Centric Motion Priors from Human for Robotic Dexterous Manipulation

**arXiv ID:** 2611.05979 | [PDF](https://arxiv.org/pdf/2611.05979v1)

**作者:** Zhengdong Hong `[一作]` (Zhejiang University), Guofeng Zhang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

基于人机交互(HOI)数据预测手-物体未来状态，并利用预测的物体轨迹作为通用奖励，引导多指机器人在多任务中进行强化学习，实现零射击Sim-to-Real迁移。

**💡 创新点**

创新点在于将HOI数据用于全局手-物体状态预测，使用预测物体轨迹代替任务特定奖励，从而获得更强的跨任务、跨物体和跨机器人手的通用性。

**🔧 技术方法**

采用GPT-2 Transformer自回归预测模型与PointNet提取物体特征；使用IK与NLopt进行手部运动重定向；通过PPO强化学习实现策略学习；结合域随机化、系统识别等技术完成Sim-to-Real迁移。

**📊 数据集**

使用Dex-YCB和ARCTIC两大HOI数据集。

**📈 对比分析**

与ViViDex、AdaDexGrasp、ManipTrans、HOP等SOTA方法在YCB抓取、关节物体操作、带障碍抓取等任务进行对比，实验显示在仿真与真实机器人上均取得更高成功率，尤其在新物体和新姿势下优于基线。

**⚠️ 局限性**

局限在于对高质量3D手-物体配对HOI数据的依赖，且对极端复杂动态或非标识物体的预测精度仍有限。

---

## 445. EEGDash Corpus: A Training-Ready Public Corpus for EEG Foundation Models

**arXiv ID:** 2611.06356 | [PDF](https://arxiv.org/pdf/2611.06356v1)

**作者:** Kuber Shahi `[一作]` (University of California San Diego), Arnaud Delorme `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

发布了77个可再发行EEG数据集的训练准备语料库EEGDash Corpus，并在其上重新训练REVE-Base验证其可用性。

**💡 创新点**

将REVE预训练语料库中的可再发行数据统一处理、转换为可直接使用的Zarr存储，提供完整的转换报告和许可信息，方便复现和扩展。

**🔧 技术方法**

通过EEGDash获取、REVE的预处理方案（滤波、重采样、位置编码）、Zarr分块存储、braindecode与PyTorch训练脚本实现。

**📊 数据集**

77个公开EEG数据集（OpenNeuro和NEMAR），共14,075小时、7,272受试者、6.21亿tokens。

**📈 对比分析**

在其上重新训练REVE-Base两次，线性探针得分0.331，微调得分0.273，略低于官方发布的0.333/0.325，但远高于未训练编码器，证明语料库有效。

**⚠️ 局限性**

仅包含可再发行数据，缺少临床大规模样本；转换仅遵循REVE原始recipe，未做artifact rejection，可能影响模型对噪声的鲁棒性。

---

## 446. SleuthVLM: Do We Really Need Audio-in-Video for Omni Video Understanding?

**arXiv ID:** 2611.06083 | [PDF](https://arxiv.org/pdf/2611.06083v1)

**作者:** Peiran Wu `[一作]` (University of Bristol), Junxiao Shen `[通讯]` (University of Bristol)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种Agent化的多模态理解框架，使用VLM在无直接音频输入的情况下，通过已转录并带时间戳的文本与工具自主搜索视频内容来回答多模态问题。

**💡 创新点**

创新点在于：①不再让模型直接听音频，而是利用时间戳文本作为索引；②通过Agent化机制让VLM自主决定何时、如何检索视频和文本；③在保持模型规模不大（27B）下即可达到闭源模型的性能。

**🔧 技术方法**

核心技术包括：基于VLM的Agent框架、时间戳ASR+强制对齐、工具调用（浏览帧、观看短片、搜索文本、答案生成）以及有限预算管理。

**📊 数据集**

使用了四个omni视频问答基准（Daily-Omni、WorldSense、OmniVideoBench、LVOmniBench）和Video-MME-v2，并在这些数据集上进行评估。

**📈 对比分析**

与闭源与开源的Omni模型及多种Agent方法对比，模型在四个omni基准上的平均准确率达到75.0%，超过最强闭源Omni模型Gemini 3.8 Flash 2.6个百分点；在Video‑MME‑v2上达到77.8%，领先所有公开模型。

**⚠️ 局限性**

局限性包括：仍需依赖高质量ASR与对齐；对非语音音频信息的处理依赖额外工具；在极长视频或极高帧率场景下搜索成本可能增加；Agent策略仍主要由经验设定，缺乏自动化规划。

---

## 447. A Definition of Catastrophic Performance

**arXiv ID:** 2611.06219 | [PDF](https://arxiv.org/pdf/2611.06219v1)

**作者:** Henrik Marklund `[一作]`, Benjamin Van Roy `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基准值和正交值作为新的参考基准，并以此定义AI失配导致的灾难性性能；

**💡 创新点**

首次将基准值与正交值结合，用“灾难程度参数η”量化相对于人类偏好的性能下降；

**🔧 技术方法**

使用MDP/POMDP数学框架、可计算性理论（Kolmogorov复杂度）和示例性环境（网格世界、Minecraft）进行形式化与说明；

**📊 数据集**

无具体数据集，主要通过构造性的模拟环境与理论推导展示概念；

**📈 对比分析**

与传统的最佳/最差值对比，作者通过理论证明和实例演示表明该定义更能捕捉“灾难”意义，未给出数值性能评估；

**⚠️ 局限性**

局限性包括对编程语言的依赖、基准值必须大于正交值的假设、可能存在的反例以及实际环境复杂性难以完全覆盖。

---

## 448. On the Design of Hyper-Connections: Simplification, Stability, and Scalability

**arXiv ID:** 2611.05984 | [PDF](https://arxiv.org/pdf/2611.05984v1)

**作者:** Xiangdong Zhang `[一作]` (Shanghai Jiao Tong University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了一种简化版的 Hyper-Connections 架构 SimpleHC，删除残差混合操作，采用共享内存访问和按组门控生成，显著降低训练开销并保持与 mHC 相当的性能。

**💡 创新点**

创新点包括：① 证明仅用学习的读写门即可实现最小有效的残差结构；② 将残差门控从每层独立生成改为按组共享输入，结合标量修正实现序列化贡献；③ 设计了一套融合内核（DeepGEMM/Triton）实现高效共享内存访问，减少内存访问量并提升训练速度。

**🔧 技术方法**

技术实现包括：多流残差网络、读写门控、无混合的残差加法、按组共享内存访问、标量修正、GEMM融合、RMSNorm、Sinkhorn 约束（用于对比）以及专用内核实现。

**📊 数据集**

实验基准使用 15B-A1B 与 30B-A2B 的 MoE 语言模型，在多种评测集上评估：MMLU、HellaSwag、MATH、MBPP、C3、GSM8K、HumanEval、CMMLU、CEval 等。

**📈 对比分析**

对比方法：与 vanilla residual、mHC、identityHC、xHC 等对齐；SimpleHC 在 15B 与 30B 模型上保持与 mHC 相同或更好的平均 benchmark 分数，同时把训练时间开销从 7.0% 降至 2.4%，内存占用减少约 31%，梯度波动更平稳，且在极端学习率下保持稳定。

**⚠️ 局限性**

局限性：实现仍依赖高度定制的 GPU 内核，迁移到其他框架或硬件可能受限；在某些任务（如 GSM8K、HumanEval）性能略低于 mHC；对更大规模（>30B）或不同结构（非 MoE）模型的泛化尚未充分验证。

---

## 449. Thick Alignment: A Social Framework for Values and AI Systems

**arXiv ID:** 2611.05857 | [PDF](https://arxiv.org/pdf/2611.05857v1)

**作者:** Alondra Nelson `[一作]` (Institute for Advanced Study), Sola Kim `[通讯]` (Institute for Advanced Study)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并系统阐述“厚对齐（Thick Alignment）”框架，批判传统“薄对齐”将价值观简化为可度量指标的做法，强调价值观的情境化、构成性和争议性。

**💡 创新点**

创新点在于将价值观视作在具体社会实践中被解释、建构和争议的过程，从而将AI系统的价值对齐从技术化的指标化转向社会化的价值共建；同时提出“Halcyon misalignment”和“Mirrored misalignment”两种薄对齐忽视的错误诊断。

**🔧 技术方法**

主要技术为跨学科理论分析与案例评估，借鉴哲学、社会学与人类学的“厚描述”方法，对现有AI对齐技术（如RLHF、Constitutional AI、评价指标等）进行概念性审视；未涉及新算法实现。

**📊 数据集**

数据来源主要为公开案例与已有研究（如GPT‑4o、Claude系列、COMPAS、面部识别偏见研究等）以及文献综述，没有使用新的标注数据集或实验数据。

**📈 对比分析**

通过对比薄对齐与厚对齐的理论框架，讨论两者在识别误差、解释价值冲突和治理设计上的差异；未给出定量性能指标，而是以案例分析和概念阐释为评估标准。

**⚠️ 局限性**

局限性包括：①缺乏可操作化的测量工具或指标，难以在工程实践中直接实现厚对齐；②概念性与案例分析主观性较强，需要后续实证研究验证其有效性；③实现厚对齐的社会技术协同过程复杂，需跨学科协作与制度支持，实际落地难度大。

---

## 450. ELA: End-to-End Latent Action Policy Learning from Video Demonstrations

**arXiv ID:** 2611.05571 | [PDF](https://arxiv.org/pdf/2611.05571v1)

**作者:** Cheng-Chun Hsu `[一作]` (University of Texas at Austin), Peter Stone `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个 end-to-end 框架（Ela）通过潜在动作模型、潜在动作策略和轻量化动作解码器，从无动作标注的视频中学习控制策略，并将潜在动作空间与策略学习和动作解码联合优化。

**💡 创新点**

创新点在于：1）将潜在动作空间与策略学习和动作解码统一端到端优化；2）引入表示对齐机制避免潜在动作坍塌；3）在仅少量动作标注数据的情况下充分利用大量无动作视频。

**🔧 技术方法**

采用了逆动力学模型 (IDM)、前向动力学模型 (FDM)、潜在动作策略、动作解码器、基于 DINO 的视觉表示对齐、Transformer 编码器/解码器以及信息量限制量化等技术。

**📊 数据集**

实验使用了 LIBERO‑10 长时序桌面操作任务集、真实世界跨实体人类视频与机器人演示数据，以及自建的动作标注与无标注混合数据集。

**📈 对比分析**

与 Diffusion Policy、Unified World Model、ATM、AMPLIFY 等基线进行对比，在模拟和真实任务中，在仅 5–10 条动作标注样本的情况下，Ela 的成功率提升至 54–59%（模拟）和 70%（真实），显著优于所有基线。

**⚠️ 局限性**

局限性包括：验证规模有限，尚未在大规模跨实体数据集上测试；未利用深度或 3D 先验；以及未涵盖高精度或细粒度操作场景。

---

## 451. AI-Driven SaaS Platform Integration: AI Express for Unified Generative AI Services with Pay-As-You-Go Credit Systems and Scalable Microservice Architecture

**arXiv ID:** 2611.05389 | [PDF](https://arxiv.org/pdf/2611.05389v1)

**作者:** Deepit Sapru `[一作]` (University of Illinois Urbana-Champaign), Deepit Sapru `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5121161310)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 AI Express SaaS 平台，将文本生成、代码审核、数据分析、图像处理等多种生成式 AI 服务整合到一个统一的按使用计费（credit）体系中，提供单一界面、无需技术门槛的体验。

**💡 创新点**

创新点包括：① 通过微服务架构实现多模态 AI 服务的可扩展和松耦合；② 引入按使用计费的信用系统，替代传统月费订阅，降低成本门槛；③ 在同一平台提供代码审核、数据分析、图像去水印等多种功能，解决行业碎片化问题；④ 通过 RESTful API 与前端 React.js 交互，保证高并发下的响应性能。

**🔧 技术方法**

使用的技术栈：React.js+TypeScript+Redux+Material-UI 前端；Node.js+NGINX 负载均衡+JWT 认证+Redis 缓存+PostgreSQL 存储；LLM（如 GPT 系列）做文本、代码分析；深度学习视觉模型（TCN+inpainting）做水印去除；Docker/Kubernetes 容器化与自动扩容；RESTful API 进行微服务通信。

**📊 数据集**

使用的数据包括：GitHub 公开仓库代码、用户上传的 CSV/TSV/Excel 数据集、带水印的图片以及用户自然语言查询。文中未提及使用具体公开数据集名称，主要依赖真实用户上传的数据进行测试。

**📈 对比分析**

性能评估方法：通过 Apache JMeter 进行负载测试，模拟 500 并发用户 10 分钟，平台 99.2% 可用；平均响应时间在文本服务为 600–850 ms，代码审核 1.25 s，数据分析 2.4 s，水印去除约 3.1 s。与 ChatGPT+、GitHub Copilot 进行功能与成本对比，AI Express 在功能完整性和按使用计费上具有成本优势。

**⚠️ 局限性**

局限性：① 计算密集型任务（如水印去除）仍存在较高延迟；② 缺乏实时协作、移动端支持；③ 信用系统单一中心化，未实现去中心化或多链支付；④ 缺乏公开基准数据验证模型性能；⑤ 对模型的鲁棒性和安全性（如对恶意输入的防护）尚未充分评估。

---

## 452. On Existence and Computational Complexity of Quasi-strict Equilibria

**arXiv ID:** 2611.06350 | [PDF](https://arxiv.org/pdf/2611.06350v1)

**作者:** Kristoffer Arnsfelt Hansen `[一作]` (Aarhus University), Xinhao Nie `[通讯]` (Aarhus University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了双人矩阵游戏中总存在准严格纳什均衡，并提供了简化的存在证明；同时确立了计算准严格均衡属于PPAD-完整，并证明多玩家游戏中判断是否存在准严格均衡的决策问题是∃-完整。

**💡 创新点**

创新点包括：用多面体复形与单纯形映射构造连续自映射，避免了Norde证明中的极限论证；首次给出准严格均衡计算的PPAD完备性；以及多玩家游戏中准严格均衡决策问题的∃-完备性。

**🔧 技术方法**

主要技术手段为：多面体与顺序复形的拓扑构造、单纯形映射、Brouwer固定点定理、欧几里得投影以及Etessami‑Yannakakis的多项式分段线性固定点框架；同时利用线性规划求解复形内部点。

**📊 数据集**

本文不依赖具体数据集，所有实验均以符号矩阵和策略空间理论进行理论分析。

**📈 对比分析**

与以往仅给出NP-hard或PPAD-hard性结果的研究不同，本文通过构造可计算的连续映射完成完整的PPAD-完备性证明；在多玩家场景下实现∃-完备性证明，显示了判定存在性的理论上最严苛复杂度。

**⚠️ 局限性**

主要限制在于仅针对有限策略形式游戏；构造与证明过程相对复杂，未给出多玩家游戏的多项式算法；且在实际应用中需要对高维策略空间进行显式构造，可能导致计算资源瓶颈。

---

## 453. Mass Producing Training Data with Surprising Side Effects

**arXiv ID:** 2611.06324 | [PDF](https://arxiv.org/pdf/2611.06324v1)

**作者:** Yaowen Ye `[一作]` (Transluce), Cassidy Laidlaw `[通讯]` (Transluce)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套自动化流程，用来在大语言模型的后训练阶段系统地发现数据集所产生的意外侧效（DSSEs）并收集成库

**💡 创新点**

将DSSE定义为：(1) 诱发目标行为；(2) 读起来像普通后训练数据；(3) 对人类审计者而言意外；并设计了可通过LLM判定的三项指标；通过两阶段搜索+评估实现规模化生成

**🔧 技术方法**

演化搜索、数据策略梯度（DPG）以及两种基于潜伏学习的改写/筛选方法；LLM判定器（Elo/Bradley–Terry）和前沿模型评估；LoRA微调、评估提示、Rubric判定等

**📊 数据集**

自生成的训练集（约150条prompt-response对），涵盖8种目标行为（安全、人物、情感、偏好等）共计6,575个DSSE；使用现有SFT数据作为对照；并在多模型上测试泛化（包括不同规模与家族）

**📈 对比分析**

与先前仅手工或偶然发现的几十个DSSE对比，自动生成的DSSE在规模、自然度与惊奇度上均显著更高；在多模型实验中，约72% DSSE能在5个不同模型上产生显著侧效；在人类与LLM预测实验中，LLM预测准确率达86%，人类仅63%（接近随机）

**⚠️ 局限性**

依赖LLM判定器导致评估模型变化、噪声大；DSSE效果往往较弱且不稳定；仅覆盖单轮行为、数据量小；多数DSSE诱发多重行为，影响惊奇度评估；未验证在真实生产数据中的普适性

---

## 454. When Does Risk-Controlled Throughput Forecasting Actually Control Risk? A Reproduction Study on LEO Satellite Link

**arXiv ID:** 2611.06307 | [PDF](https://arxiv.org/pdf/2611.06307v1)

**作者:** Mayan Sharma `[一作]` (Vellore Institute of Technology), S. Vijayaragavan `[通讯]` (Nelco Network Products Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现并评估了BG‑CFQS风险控制吞吐量预测方法，探究其在真实LEO卫星链路下的风险保证是否可持续，分析风险预算低于0.15的结构性限制，比较静态与在线、分组条件化的合成校准技术，并验证静态校准对排序的局限性。

**💡 创新点**

发现BG‑CFQS在非可交换的时间序列划分下无法满足预算，低于0.15预算不可行；静态合成校准既无法提升排名也会导致风险偏差；在线及分组条件化校准能在不同预算下保持风险控制，且在线校准在多处表现优于静态方法；提出价格不变的风险控制部署判据。

**🔧 技术方法**

使用可合成预测（conformal prediction）、在线合成校准（Rolling RC）、分组条件化在线校准（GCACI）、聚类合成预测（Clustered Conformal Prediction）以及GRU序列到序列模型进行基准预测。

**📊 数据集**

四个数据集：Chicago、Osnabruck、Victoria 三条Starlink跟踪（共约2.3M样本）以及独立的WetLinks（约1.02M样本）。

**📈 对比分析**

与BG‑CFQS原始设置对比：随机可交换划分下风险满足预算，时间连续划分则仅1/3满足；在线校准在7个预算点的实际OverRate与目标一致，误差≤1%；静态校准在所有预算下低估或高估约16–21%；分组条件化校准在多数数据集上效果不佳；在延迟峰值检测任务中，静态校准与点预测AUPRC相同，在线校准提升约43–51%。

**⚠️ 局限性**

实验仅基于离线回放，缺乏现场终端验证；仅使用四个数据集，跨数据集结论受限；在线校准需实时反馈；未覆盖贝叶斯/经验贝叶斯等传统池化方法；高维分组导致样本稀疏，影响估计精度。

---

## 455. STEK: Structural-Terrain-Enhanced Koopman-Inspired Cross-Channel Regularization for Retrospective Battery Multi-Target Estimation

**arXiv ID:** 2611.06299 | [PDF](https://arxiv.org/pdf/2611.06299v1)

**作者:** Jing Wang `[一作]` (University of Science and Technology of China), Fugee Tsung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对异质电池档案中的回顾性周期级特征，提出一种利用端点曲线与生命周期结构（dQ/dV）相结合的STE K框架，能够在缺失或不完整的原始信号情况下实现多目标预测。

**💡 创新点**

创新点在于：①通过质量代理门控的结构融合与有界FiLM，安全地将生命周期结构与端点曲线信息注入模型；②引入仅在训练时使用的Koopman启发的跨通道一致性正则化；③提供可退化回到单纯曲线预测的残差融合机制。

**🔧 技术方法**

技术手段包括：端点曲线特征提取网络（可替换的时间序列骨干，如TimesNet、DLinear、PatchTST）；生命周期景观生成器（dQ/dV terrain）与端点结构代理的编码器；质量代理门控与有界FiLM；训练时的低秩Koopman一致性约束；残差预测门控；以及多目标损失与整体得分计算。

**📊 数据集**

使用公开的MIX‑battery基准数据集（511个单元，10个来源），对多达7个周期级输出（NDC、SOH、CF、CFR、ICR、QTY、NCC）进行评估。

**📈 对比分析**

与端点曲线仅输入的基准（MambaLithium、LiPM、CPTransformer、iTransformer、Koopa、KNF）以及匹配输入的对照实验（直接拼接、无质量门控FiLM、共享/状态条件Koopman）进行对比。STE K在所有三种骨干上均取得最低的整体得分12.42（≈17.6%优于iTransformer，≈18.4%优于直接拼接），并在SOH、CF、ICR、QTY等历史相关指标上实现最佳或第二佳表现。

**⚠️ 局限性**

局限性包括：①仅适用于回顾性归档，无法在线预测；②在可直接解析的输出（NDC、NCC）上仍被确定性规则压制；③跨源迁移虽有提升但仍存在源特定失败；④对生命周期景观和结构代理的完整性敏感，缺失时仅退化到曲线路径；⑤未对噪声或错误描述符进行鲁棒性评估。

---

## 456. Multi-Resident Data Association in Smart Homes via MDP Modeling and Behavior Cloning

**arXiv ID:** 2611.06296 | [PDF](https://arxiv.org/pdf/2611.06296v1)

**作者:** Xi Chen `[一作]` (Univ. Grenoble Alpes), Dominique Vaufreydaz `[通讯]` (Univ. Grenoble Alpes)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

将多居民传感器事件关联问题重新建模为马尔可夫决策过程（MDP），并通过行为克隆（BC）训练Transformer和LLM策略实现事件到居民的赋值。

**💡 创新点**

创新点在于：①把数据关联视为序贯决策任务，突破传统过滤器的第一阶马尔可夫限制；②采用BC直接从专家轨迹学习策略，省去奖励工程；③提供Transformer与LLM两种可扩展策略，并结合束搜索实现多路径推理；④展示LLM在跨环境迁移下的可转移性。

**🔧 技术方法**

技术方法包括：MDP建模、行为克隆（BC）、Transformer编码器（Pre‑RMSNorm、SwiGLU、RoPE）、LLM微调（LoRA）、束搜索、二分类交叉熵训练、BLEU/CRL/RCE评估。

**📊 数据集**

实验数据集：CASAS ADLMR（双人）、MARBLE（多达4人）、MuRAL（2–4人）。

**📈 对比分析**

与sMRT、LAHAR（GPT‑4o）以及All‑to‑One基线对比，BC和LLM+BC在准确率、RCE、CRL、BLEU等指标上均明显优于所有基线；束搜索进一步提升长序列一致性；LLM+BC在跨环境迁移中保持较高准确率。

**⚠️ 局限性**

局限性包括：在长时间窗口下仍易发生错误累积；多居民事件的精确识别仍低于理想；LLM对环境元数据高度依赖；在高噪声比例下误跳和误判率上升；未在真实部署环境中验证鲁棒性，需要进一步结合DAgger或强化学习提升分布漂移适应。

---

## 457. Corridor-Based Monitoring and Emergency Braking Decision for Automated Vehicle Safety

**arXiv ID:** 2611.06287 | [PDF](https://arxiv.org/pdf/2611.06287v1)

**作者:** Junnan Pan `[一作]` (University Bundeswehr Munich), Ferdinand Englberger `[通讯]` (University Bundeswehr Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究并实现了一种低成本嵌入式安全通道，用预测运动通道过滤3D雷达返回并在满足硬件约束下决定自动驾驶车辆是否需要紧急制动。

**💡 创新点**

创新点包括：① 通过滑动极值构造的曲率感知监测边界（pmcm），精准包围车辆扫动路径；② 双证据威胁评估（gdeta）将几何占用与时间紧迫性独立评分，并通过硬门与软OR仲裁实现安全决策；③ 将两者结合在同一系统（nefimmore）中，实现高可靠的冗余制动决策。

**🔧 技术方法**

技术手段：3D LiDAR点云滤波、曲率感知边界构造、局部角窗口点云筛选、聚类+Hungarian关联、轨迹预测、几何占用评分、时间紧迫性评估（TTB/BTN）、嵌入式可执行的决策规则。

**📊 数据集**

数据集与仿真：Gazebo中固定圆柱障碍物的碰撞测试（速度1–30 km/h、曲率多半径）；校园实测驾驶记录（包含城市转弯、噪声与定位漂移）；Monte Carlo随机生成两类情景（领头车制动、横向穿越），分别进行200,000次试验。

**📈 对比分析**

比较方法：将pmcm与平行四边形与两点条纹两种基线对比，评估漏检率（fnr）、命中率（tpr）和残余碰撞率（cr）。pmcm实现fnr=0%、tpr=100%、cr=0%；基线存在漏检与误碰。Monte Carlo中lead braking recall 99.95%、跨越案例 recall 75.8%，两类情况fp率均低（1.12%），总体误报率极低，满足冗余安全通道的要求。

**⚠️ 局限性**

局限性：仅监测单车道内单物体，未处理多物体交互与变道；只使用单一3D LiDAR，未结合多传感器；连续扫描确认阈值需经验调节；对非平面或极端光照/天气环境的鲁棒性未验证。

---

## 458. MonoLaDN: Language-Guided DeNoising for Monocular 3D Object Detection

**arXiv ID:** 2611.06282 | [PDF](https://arxiv.org/pdf/2611.06282v1)

**作者:** Kiet Dang Vu `[一作]` (Ho Chi Minh City University Of Technology), Duc Dung Nguyen `[通讯]` (Vietnam National University Ho Chi Minh City)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出MonoLaDN框架，利用冻结的视觉语言模型在训练阶段生成三维几何推理（遮挡、深度、尺寸、朝向），并通过语言引导的去噪机制将其知识迁移到单目3D目标检测器中，推理路径在推理时被移除，保持模型轻量；

**💡 创新点**

创新点在于：①仅在训练时使用VLM进行结构化几何推理而非直接部署，②通过噪声注入和分离注意力掩码的语言引导去噪训练，迫使检测器内化VLM的空间知识，③在保持推理速度不变的同时显著提升3D检测精度；

**🔧 技术方法**

主要技术包括：基于MonoDGP的深度引导Transformer检测器、冻结的SpaceThinker‑Qwen2.5‑VL VLM进行推理生成、文本编码器（SigLIP）将推理文本映射为嵌入、噪声注入与变分自编码器正则化、语言引导注意力掩码、结合检测与重建损失的联合训练；

**📊 数据集**

实验数据集使用KITTI 3D（车类）和nuScenes，分别在验证集、测试集以及跨域评估上进行；

**📈 对比分析**

与MonoDGP、MonoCLUE、MonoDLGD等SOTA单目3D检测器相比，MonoLaDN在KITTI 3D Car类别的AP_3D|R40分别实现29.67/20.45/18.02，提升3.3/1.7/2.0点，且推理时无额外参数或延迟；在nuScenes跨域测试中同样表现出更低的深度MAE；

**⚠️ 局限性**

局限性包括：目前仅针对单图像推理，未考虑多视角或时序信息；依赖冻结VLM生成推理文本的质量，若VLM在3D推理上欠佳，迁移效果受限；

---

## 459. DyPSI: Dynamic Physics Sensing via Joint Field and Sensor-Trajectory Generation

**arXiv ID:** 2611.06280 | [PDF](https://arxiv.org/pdf/2611.06280v1)

**作者:** Yizhou Zhang `[一作]` (Ocean University of China), Shikai Fang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了DyPSI框架，联合生成未来物理场与移动传感器轨迹，实现时空一致的动态感知；

**💡 创新点**

创新点在于：①将离散传感器坐标映射为连续Sensor Position Field；②共享Functional Tucker潜在空间同时编码场与轨迹；③在扩散模型中使用Matérn‑3/2 Gaussian Process噪声实现时间平滑；④给出理论分析连接EDM损失与A‑optimal性以及GP先验对轨迹变化的约束；

**🔧 技术方法**

采用Functional Tucker表示、Sensor Position Field、Elucidated Diffusion Model (EDM)、Matérn‑3/2 Gaussian Process相关噪声、A‑optimality理论与联合潜在扩散训练；

**📊 数据集**

实验使用三大数据集：二维湍流通道流、GLORYS12全球海表温度（1°分辨率）以及OpenFOAM+ShapeNet生成的汽车气动表面压强；

**📈 对比分析**

通过相对L2误差与多种基线（SSPOR、VoronoiCNN、Senseiver、S^3GM、DiffusionPDE、PhySense随机/静态/Per‑frame）对比，DyPSI在所有任务和传感器预算下均取得最低误差，稀疏传感器下提升10%–20%，同时轨迹平滑度提升、速度约束违约率显著降低；

**⚠️ 局限性**

局限性：①依赖离线优化生成的教师轨迹；②假设部署环境与训练分布相似，对分布外或高噪声场景缺乏鲁棒性；③仅做历史条件的开环推理，未实现闭环实时控制；④仅考虑点测量，未包含区域集成传感器；⑤高密度传感器时SPF峰合并导致读出困难。

---

## 460. From hydrological laws to catchment models: A process-based model matches deep-learning accuracy on three continents

**arXiv ID:** 2611.06382 | [PDF](https://arxiv.org/pdf/2611.06382v1)

**作者:** Xilin Xia `[一作]` `[通讯]` (University of Birmingham), Xilin Xia (University of Birmingham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文探讨了在物理法则下构建的低维系统是否能够与在相同输入下超越基于过程的模型的神经网络的准确性相匹配。研究提出了三个约束集，构建了一个简化的集水区模型，并通过AI代理进行迭代改进。

**💡 创新点**

创新点在于提出了三个集水区模型的基本法则：水的守恒、耗散和聚合，并在此基础上构建了一个无神经网络的低维模型，该模型在多个国家的集水区中表现出与深度学习模型相当的预测能力。

**🔧 技术方法**

使用了基于物理法则的低维模型和AI代理进行模型的迭代改进，结合了状态空间模型和LSTM等技术。

**📊 数据集**

使用了602个英国和593个美国的测量集水区数据集，以及505个未用于开发的干旱澳大利亚集水区的数据集。

**📈 对比分析**

与单一模型相比，基于过程的模型（如HBV和dHBV）在Nash-Sutcliffe效率（NSE）上表现较差，而提出的模型在多个数据集上均表现出优越性，尤其是在干旱和雪影响的集水区中。

**⚠️ 局限性**

限制在于模型的复杂性和参数的选择，尽管模型在多个集水区中表现良好，但仍需进一步验证其在不同气候和地理条件下的适用性。

---

## 461. Does Context Compression Preserve Refusal Alignment?

**arXiv ID:** 2611.06380 | [PDF](https://arxiv.org/pdf/2611.06380v1)

**作者:** Anthony Nguyen `[一作]` (Algoma University), Wenjun Lin `[通讯]` (Algoma University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了上下文压缩对指令调优语言模型拒绝行为的影响，并提出了在压缩空间中进行的轻量级纠正方法——Memory Steering。

**💡 创新点**

创新点在于揭示压缩会削弱模型的拒绝方向，并提供一种仅在压缩记忆向量上插入拒绝信号即可恢复拒绝率的干预技术。

**🔧 技术方法**

采用了固定长度ICAE与变长ARC压缩技术、线性探针分析拒绝方向、Memory Steering（向量注入）以及Gemma-3-27b-it自动评估器。

**📊 数据集**

使用了自定义的624条安全/有害提示、HarmBench、AdvBench三大安全基准，以及公开的LLaMA‑3、Mistral等模型的推理数据。

**📈 对比分析**

通过对比未压缩与压缩后的拒绝率和对Memory Steering的α参数进行调节，实验表明压缩显著降低拒绝率（最多下降近80%），而Memory Steering可在保持回答质量的前提下将拒绝率恢复至近基线水平。

**⚠️ 局限性**

局限性包括：仅验证了固定长度ICAE压缩，无法直接推广至ARC等变长方法；对拒绝方向的机制解释仍不完整；仅评估了拒绝行为，未考察压缩对其他对齐属性（如校准、不确定性表达）的影响。

---

## 462. FlashRelight: Portrait Video Relighting with Dynamic Lighting

**arXiv ID:** 2611.06355 | [PDF](https://arxiv.org/pdf/2611.06355v1)

**作者:** Jiahui Sheng `[一作]` (Zhejiang University), Rui Hu `[通讯]` (vivo Mobile Communication Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段的动态肖像视频重光照框架，先用光照消除网络将源视频转为近似无光照的视频，再利用基于DiT的视频生成模型在此基础上生成符合目标环境图序列、文本描述或参考视频的动态光照效果。

**💡 创新点**

创新点包括：1) 将光照分离与视频生成相结合的两阶段流程，使得重光照过程更具可控性与稳定性；2) 支持三种互补的动态光照控制方式（环境图序列、动态文本、参考视频）；3) 通过低成本的混合训练策略，将静态OLAT数据与生成的动态光照视频相结合，克服了缺乏大规模动态光照对齐数据的难题。

**🔧 技术方法**

技术手段主要有：基于DiT的视频生成模型做重光照器；轻量化的光照消除网络（Delighting）；HDR环境图通过三种缓冲编码为轻量化灯光嵌入；T5文本编码器配合VLM提取文本条件；参考视频通过密集时空编码融入生成流程；混合训练数据集构建与三阶段训练计划。

**📊 数据集**

使用的主要数据集有：1) FaceOLAT静态OLAT数据（60名演员、2–4种表情），与600个合成环境图序列及200个真实环境图组成24,000条训练视频；2) 通过Wan2.2生成的10,000+动态光照视频，包含光照文本与参考视频，构成Hybrid Dynamic Portrait Video Dataset。

**📈 对比分析**

在环境图序列、动态文本和参考视频三种重光照任务上与Lite2Relight、Relighting‑Cai、DiffusionRenderer、IC‑Light、Light‑A‑Video、Light‑X等基线进行定量和定性比较。FlashRelight在Fidelity、ID相似度、tLPIPS等指标上均优于基线（例如Fidelity 55.8，ID 0.673，tLPIPS 0.190），并在视觉质量与时间一致性方面表现更佳。

**⚠️ 局限性**

局限性：1) 主要依赖合成与生成数据，缺乏真实动态光照的对齐评估；2) 对极端光照变化或复杂场景的鲁棒性尚待验证；3) 模型推理仍需高端GPU，推理成本相对较高；4) 参考视频条件需进行内容抑制，仍可能存在身份泄露风险。

---

## 463. PMRecBench: A Benchmark for Personalization in Recommendation Agents under Multimodal Queries

**arXiv ID:** 2611.06349 | [PDF](https://arxiv.org/pdf/2611.06349v1)

**作者:** Xiaoyu Zhang `[一作]` (University College London), Emine Yilmaz `[通讯]` (University College London)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了Personalized Multimodal Recommendation Benchmark（PMRecBench），用于评估在多模态（图像+文本）查询下推荐代理的个性化表现。

**💡 创新点**

创新点在于：①将查询与真实用户历史行为对应，生成具有视觉需求的查询；②构造诊断候选池，分别衡量需求理解和偏好跟随；③揭示多模态查询与个性化信号混合时的偏好失衡。

**🔧 技术方法**

采用多模态大语言模型（Gemma‑4‑31B、Qwen3.6‑35B、GPT‑6‑Luna等）、工具调用框架（ReAct、PyVision、RecMind、InteRecAgent等）以及传统推荐器（SASRec、VBPR、Popularity等）进行实验。

**📊 数据集**

使用Amazon服装评论与MovieLens‑1M数据集，构造了510个图像‑文本查询及对应的用户历史。

**📈 对比分析**

与传统推荐器和直接排名器对比，PMRecBench在随机候选池中取得最高NDCG，但在诊断池中显示尽管总体排名提升，偏好跟随得分下降，平均降幅约4–8个百分点。

**⚠️ 局限性**

局限性在于：多模态查询导致模型过度关注图像相似度，抑制了对用户历史偏好的利用；更强的模型或工具调用并不能根除这一失衡；缺乏对跨模态一致性和解释透明度的深入分析。

---

## 464. Flexible Intensities Matter: A Comprehensive Re-evaluation of Hawkes Processes and Neural Temporal Point Processes

**arXiv ID:** 2611.06348 | [PDF](https://arxiv.org/pdf/2611.06348v1)

**作者:** Hendrik Mehrtens `[一作]` (German Cancer Research Center), Oliver Stegle `[通讯]` (German Cancer Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

系统评估了 Poisson、Hawkes 与神经 TPP 在标准与大规模医学数据集上的下一个事件预测性能。

**💡 创新点**

发现对 Hawkes 加入可学习基线、样条核和抑制项后，能显著缩小与神经模型的性能差距；并指出神经模型的优势主要体现在时间密度而非事件类型上。

**🔧 技术方法**

使用可变形基线/核的 Hawkes、Spline/Hawkes、抑制 Hawkes，以及 Transformer、NHP、IntensityFree、NeuralSplineTPP 等神经 TPP；实现统一框架 TorchPP。

**📊 数据集**

评测了 Taxi、Retweet、StackOverflow、Amazon、Taobao 等公共基准，以及 UK Biobank 疾病史和 MIMIC‑IV ICU 记录。

**📈 对比分析**

通过对数似然、时间条件准确率和 RMSE 的分解比较，灵活 Hawkes 与最先进的神经模型在大多数基准上相近，神经模型仅在某些大规模医学数据的时间和类型两项上略占优势。

**⚠️ 局限性**

局限在于只考虑 Poisson/Hawkes 与有限神经 TPP，未覆盖 Cox、流式/扩散/状态空间模型；仅做下一个事件预测，未评估长周期生成或决策；数据集选择受限，未处理并发事件与离散时间问题。

---

## 465. Prism: Bidirectional Self-Teaching for Unified Multimodal Models

**arXiv ID:** 2611.06343 | [PDF](https://arxiv.org/pdf/2611.06343v1)

**作者:** Peng Zhang `[一作]` (Xi'an Jiaotong University), Shuai Liu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为Prism的双向自教后训练方法，用于统一多模态模型（UMMs）中视觉理解与生成的协同提升

**💡 创新点**

核心创新在于通过自生成的交错文本-图像轨迹进行可靠性评估，动态决定教学方向（文本指导图像、图像指导文本或交叉一致性）而非固定方向，并在可靠轨迹上重写不可靠链条

**🔧 技术方法**

利用冻结的UMM评判器进行文本链与图像链的可靠性判断，构造四类训练目标（Pass、Text→Image重写、Image→Text重写、Fail的负采样），并在此基础上更新rollout模型；使用流匹配损失、交叉熵及负似然等多种损失组合

**📊 数据集**

在SenseNova-U1和BAGEL两种UMM基础上，使用GPT‑5.5生成约2.4K任务提示作为实验数据，评估指标包括GenEval、DPG、MME、MMVP、MMBench、MMMU、MMStar、OpenING、POPE等

**📈 对比分析**

与基线以及SRUM、UniCorn等现有后训练方法相比，Prism在生成（GenEval+3.4、DPG+1.2）和理解（MME+29、MMVP+4）均实现显著提升，在OpenING交错生成中的胜率和零图像率均有改善，整体性能更为均衡

**⚠️ 局限性**

局限性包括仅针对已提供提示的交错文本‑图像任务，负样本仅对完整文本链施加，且缺乏完全自采样训练，未来需扩展至更广泛多模态场景

---

## 466. An Engineering Reading of the Bounded Halting Problem and Observable Properties of a Universal Machine

**arXiv ID:** 2611.06303 | [PDF](https://arxiv.org/pdf/2611.06303v1)

**作者:** Sergey Salishev `[一作]` `[通讯]` (St. Petersburg State University), Sergey Salishev (St. Petersburg State University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了有限预算停机问题（bounded halting）在工程系统中的应用，给出了确定性和非确定性下的速度上界证明，并将其作为评估模型可通用性（universality）的测试工具，特别针对机器学习系统；

**💡 创新点**

提出“固定实现的速度上界”定理，证明若实现的资源不随预算增长，则无法在严格小于T的时间内统一决定所有T步停机实例；同时给出可观测的三测试（配置空间、决策时限、可访问中间数据），用于判定部署系统是否具备可通用性；

**🔧 技术方法**

使用对角线论证、时间层次理论、非确定性时间层次理论、可观测属性定义、统一时间尺度的模拟器模型等理论工具；

**📊 数据集**

无实验数据集，本文为纯理论分析；

**📈 对比分析**

无实验比较，主要通过理论证明与已有速度上升定理、复杂度层次进行对比；

**⚠️ 局限性**

仅适用于精确答案的全部实例，未考虑概率或“未知”输出；给出的是上界，缺乏具体实现速度评估；在实际工程中需结合实现细节；缺乏对随机化或学习系统的定量评估。

---

## 467. Beyond Scene Shortcuts: Action-Centric Video-Text Alignment for Fine-Grained Action Understanding

**arXiv ID:** 2611.06302 | [PDF](https://arxiv.org/pdf/2611.06302v1)

**作者:** Zhiliang Zhu `[一作]`, Dawei Zhao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

暂无具体信息

**💡 创新点**

暂无具体信息

**🔧 技术方法**

暂无具体信息

**📊 数据集**

暂无具体信息

**📈 对比分析**

暂无具体信息

**⚠️ 局限性**

暂无具体信息

---

## 468. CoRTeC: A Differentially Private Release for Synthetic Tabular Data, Decodable by Any Generator

**arXiv ID:** 2611.06342 | [PDF](https://arxiv.org/pdf/2611.06342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 469. A.S.E 2.0: Understanding How Repository Context Impacts the Security of LLM-Generated Code

**arXiv ID:** 2611.06294 | [PDF](https://arxiv.org/pdf/2611.06294v1)

**作者:** Jiongchi Yu `[一作]` (Nanyang Technological University), Dong Zhang `[通讯]` (Tencent)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 SusVibes benchmark，用 145 个来自真实 CVE 的仓库级代码生成任务来评估 LLM 代码安全性。

**💡 创新点**

创新点在于将仓库上下文作为可控变量，并对 13 种上下文构造方法、6 大语言模型和编码代理进行系统实验，揭示功能导向上下文对安全的局限性及安全感知上下文的益处。

**🔧 技术方法**

使用了检索式上下文构造、代码注入、长窗口、相似检索、图检索等 13 种方法，并结合功能/安全判别器、可执行测试和 PoC exploit。

**📊 数据集**

数据集为 145 个任务，覆盖 28 种 CWE，7 种语言，51 个开源项目，包含功能测试和可执行 PoC。

**📈 对比分析**

通过与本地仅目标文件（Local-only）、BM25+README 等控制对比，发现功能导向方法可提升功能通过率至 65% 以上但安全通过率仅 +3.7 pp；安全感知方法可提升 4.2-5.5 pp；Agent 与最优检索相比安全通过率提升约 2-7 pp。

**⚠️ 局限性**

局限性包括依赖 PoC 及可执行 oracle 结果、受任务可编辑范围限制、部分方法在新语言或项目中的重现性不明、可能存在任务污染、以及对大语言模型训练数据的潜在泄漏风险。

---

## 470. Clear Human Web: Where a Person Vouches for Every Client

**arXiv ID:** 2611.06286 | [PDF](https://arxiv.org/pdf/2611.06286v1)

**作者:** Lorenzo Porcelli `[一作]` `[通讯]` (Independent researcher), Lorenzo Porcelli (Independent researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了“Clear Human Web”，一种通过欧盟数字身份钱包和盲签名令牌实现的只对已验证人类访客开放的公共网络子集。

**💡 创新点**

创新点在于：①使用现有欧盟 PID 与单一注册机构结合盲签名令牌，既能验证人类身份又保持身份不可链接；②不需要任何网站注册或新协议，直接在现有网页栈上实现；③定义了五大安全属性与七类攻击模型，并在现有技术上证明其安全性。

**🔧 技术方法**

核心技术包括欧盟数字身份钱包（PID）、WebAuthn/身份凭证、Privacy Pass 盲签名协议、零知识展示（计划中的长方形证书）、以及可选的 Tor 网络与 Web Bot Auth 机制。

**📊 数据集**

本工作未使用传统数据集；实验与评估基于理论分析、现有标准规范以及公开的网络流量统计（如 Cloudflare 与 Meta 的流量比例）。

**📈 对比分析**

对比方法主要是安全属性与攻击模型的满足程度；未进行性能实验，理论上令牌签发与验证成本与现有 Privacy Pass 相当，网络延迟与标准 HTTP 无显著差异。

**⚠️ 局限性**

主要局限包括：①需依赖欧盟 PID 与国家级发行机构；②当前钱包未实现单一人对应单一伪名的限额，导致 k 上限无法严格控制；③无法防止租赁与重用令牌的攻击；④未涵盖网络层匿名性，需结合 Tor 等额外技术；⑤缺乏完整的零属性证明实现，影响安全性与效率。

---

## 471. Give Up to Get More: Priors, Not Policies, Govern Test-Time Compute Allocation

**arXiv ID:** 2611.06372 | [PDF](https://arxiv.org/pdf/2611.06372v1)

**作者:** Jinhyung Bae `[一作]` `[通讯]` (Hankuk University of Foreign Studies), Jinhyung Bae (Hankuk University of Foreign Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在有限预算下对LLM输入进行重复尝试的分配策略，发现传统“难度越大尝试越多”规则在模型难以解决的工作负载上效果差，提出最佳分配呈倒U形，并证明当有可验证器时，是否使用历史先验信息决定能否获得收益，而非策略本身。

**💡 创新点**

创新点在于：① 用倒U形最优分配理论解释了为何“难度越大尝试越多”在硬工作负载失效；② 系统评估了在有验证器情况下多种索引策略与仅使用历史先验的影响，揭示“先验重要、策略次要”；③ 对历史先验的质量提出了可量化的临界相关系数阈值，并给出温度化方法避免在易任务上损失；④ 通过严格的跨论文调查检验该规则在221篇论文中的普遍性与缺陷。

**🔧 技术方法**

技术主要包括：离散最优分配的梯度分析（倒U形阈值公式）、三种索引策略（Thompson Sampling、UCB、Bayes‑myopic）、先验构建与温度化、基于验证器的成功概率模型、以及对验证器误差的数学阈值分析。

**📊 数据集**

使用的数据集包括公开存档的 Large Language Monkeys（MATH、GSM8K、CodeContests、MiniF2F）、PRM800K、Weaver、以及新生成的 Qwen2.5（0.5B、1.5B、7B）样本，涵盖数学、代码与形式证明三大任务。

**📈 对比分析**

与均匀分配（uniform）和轮询（round‑robin）对比时，最佳静态规则在易任务上可捕获 50–100% 的可得提升，而在硬任务上仅 0–13%；在有验证器且使用历史先验时，Thompson Sampling 在最难工作负载上提升 12–14% 的解题率，未使用先验时几乎无提升；先验质量下降时收益平滑衰减，低于约 0.3–0.4 的相关系数时几乎无收益。

**⚠️ 局限性**

局限性包括：实验基于离线重放，假设尝试独立且验证器完美；仅测试四种索引策略，未探索学习型分配；历史先验需要先前相同输入的运行记录，对新任务几乎无效；验证器误差阈值需低于任务成功率，学习型验证器常无法满足；数据集主要为数学/代码/形式证明，未覆盖前沿推理模型或开放式生成任务。

---

## 472. EvoCodeAgent: Project-Level Code Generation via Architecture-Code Co-Evolution and Experience-Guided Repair

**arXiv ID:** 2611.06328 | [PDF](https://arxiv.org/pdf/2611.06328v1)

**作者:** Qianhui Zhao `[一作]` (Beihang University), Fang Liu `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于状态演进的项目级代码生成框架（EvoCodeAgent），实现架构与代码的协同演化、工具驱动的验证以及经验抽象重用。

**💡 创新点**

核心创新点包括：①将架构设计、骨架生成、代码实现视为一个持续演进的项目状态；②在每个阶段通过工具产生可验证的证据，提升中间工件的可信度；③从迭代修正中抽象可复用的修复经验，形成跨文件、跨迭代的通用修复策略。

**🔧 技术方法**

技术手段包括多代理协作（生成器+评判器）、结构化工具调用（代码检索、符号提取、验证器等）、动态依赖关系推理、经验库构建与检索（BM25）、LLM（DeepSeek‑v3.2 与 GPT‑5）与上下文窗口管理。

**📊 数据集**

实验数据集：DevBench（10个Python项目，平均2.2文件/276 LOC）和 CodeProjectEval（18个Python项目，平均12.7文件/2,388 LOC）。

**📈 对比分析**

对比方法：CodeS、MetaGPT、ProjectGen；评估指标为单元测试通过率、架构一致性、代码规模与复杂度。EvoCodeAgent 在 GPT‑5 下分别取得 60.48%（DevBench）和 27.73%（CodeProjectEval）的通过率，显著优于基线（最大提升 53.5%），且在大规模任务中保持高架构一致性和合理的代码规模。

**⚠️ 局限性**

局限性包括：实验仅覆盖两份数据集，可能限制泛化；性能受所选 LLM 基座影响；输入规格不完整或相互矛盾时仍可能导致生成质量下降；目前以 Python 为主，跨语言适用性待验证。

---

## 473. Estimating the Volume of a Union of Convex Bodies

**arXiv ID:** 2611.06364 | [PDF](https://arxiv.org/pdf/2611.06364v1)

**作者:** Eeshan Banerjee `[一作]` (Indian Statistical Institute), Uddalok Sarkar `[通讯]` (Indian Statistical Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通用框架来近似计算任意凸体集合并集的体积，并证明在实际可实现的采样与体积估算 oracle 下能给出 (ε,δ)-近似。

**💡 创新点**

核心创新包括：①构造“sunflower”凸体族证明仅靠总变分距离的近似采样无法保证并集体积估计；②引入尾控（tail‑controlled）采样概念，在大部分区域实现乘法接近 uniform，而允许在极少量“异常”区域有任意偏差；③基于此概念给出完整的算法和证明，显著降低了采样步长和整体复杂度；④改进 Hit‑and‑Run 随机游走的混合时间分析，将对维度的依赖从 n³ 降到 n²。

**🔧 技术方法**

使用随机化算法、Markov Chain Monte Carlo（Hit‑and‑Run）、PAC 体积估算（Gaussian cooling）、泊松化采样、尾控概率分析以及几何随机游走的谱/切比雪夫不等式。

**📊 数据集**

实验中使用的主要数据集为：①轴平行盒子集合（Box benchmark）和②“sunflower”凸体族（Sunflower benchmark），以评估不同形状与重叠程度对采样误差的影响。

**📈 对比分析**

与之前基于离散化的 Delphic‑set 方法相比，所提出的算法在维度较高时显著降低了采样步长和时间复杂度，且在实际实现中不需要完整的混合时间。实验显示对 Box benchmark 随着 aspect ratio 增大误差上升，而对 Sunflower benchmark 随着花瓣数增多误差上升，验证了理论预测。

**⚠️ 局限性**

局限性：尾控采样仅保证大部分区域的乘法接近，而对异常区域的行为仍未完全控制；算法仍依赖于对体积估算 oracle 的成功率，若估算误差大则整体性能下降；当前对 Hit‑and‑Run 的步长改进虽然降低了维度依赖，但仍保留对 log 的较大依赖，未来可进一步优化。

---

## 474. FPTR: Feasibility-Preserving Transactional Refinement for Deadline-Constrained Joint Beam and Resource Allocation

**arXiv ID:** 2611.06491 | [PDF](https://arxiv.org/pdf/2611.06491v1)

**作者:** Minghao Kong `[一作]` (Pengcheng Laboratory), Rongjie Wang `[通讯]` (Pengcheng Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种可在短时间内保证输出可行的无线调度框架FPTR，在每个调度周期内确保在100 ms截止前返回完整可行的波束-资源分配。

**💡 创新点**

创新点在于引入事务式提交/丢弃门控机制：每个改进阶段独立构造候选解，仅当候选完整、及时、可行且严格优于现有 incumbent 时才替换 incumbent，从而在不中断可用解的前提下利用多余计算时间进行精细搜索。

**🔧 技术方法**

主要技术包括：基于缓冲区的资源选择、兼容组（CG）用户共享、波束掩码重建与单资源修复、双资源破坏-重建联合调配，以及配套的可行性验证器和累计阶段截止时间控制。

**📊 数据集**

使用150个合成固定实例（30个种子在5个不同工作负载级别）进行内部评测，并在相同输入下对比6种独立外部搜索基线（ALNS、Tabu、GA、SA、ILS、GRASP）进行匹配预算评估；此外，还用12个小型实例（DP最优解）对 FPTR 的最优性间隙进行校准。

**📈 对比分析**

在87 ms预算下，FPTR 的 Full 配置相较于轻量级 BeamFirst 提升了约3.83 % 通信流量，且相较于 Base 提升约1.79 %；在匹配外部基线的实验中，Full 在所有150个实例中都保持可行且无超时，平均相对提升约7.6 %（6.68–8.35 %），并且所有运行均未超过100 ms。

**⚠️ 局限性**

局限性包括：相较于 Base 只提供有限的提升，且在60 ms后收益趋于平缓；未能达到最优解（最大25 % GAP）；实验仅基于静态合成快照，缺乏动态队列、公平性、移动性等更复杂情景；运行时评估依赖单一主机，未验证硬实时行为；事务门控不覆盖程序崩溃或强制终止情况。

---

## 475. Computing Forest Degree Realizations with Minimum Cardinality Domination

**arXiv ID:** 2611.06482 | [PDF](https://arxiv.org/pdf/2611.06482v1)

**作者:** Amotz Bar-Noy `[一作]` (City University of New York), Dror Rawitz `[通讯]` (Bar Ilan University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一套完整的“森林度序列”框架，用于在给定度序列的前提下构造对应的森林，并在此基础上给出关于顶点覆盖、支配集、总支配集、配对顶点覆盖等经典优化问题的最优解和线性时间实现。

**💡 创新点**

创新点主要包括：
- 将度序列与森林结构结合，给出了一套通用的可构造性判定与实现方法；
- 对于每个优化目标（最小顶点覆盖、最小支配集、最小总支配集、配对支配集）提供了精确的数学表述、下界与上界，证明其在森林度序列下可实现；
- 设计了专门的增量操作（叶-叶、叶-非叶、叶合并等），通过局部操作即可保持森林性，同时逐步逼近目标度数；
- 通过对已有的“优化顶点覆盖”问题做巧妙的归约与改写，得到对“配对顶点覆盖”和“总支配集”的实现；
- 所有算法均在线性时间内完成（O(n) 或 O(∑d_i)），在理论上与已知复杂度最优相当。

**🔧 技术方法**

使用的技术与方法：
- 度序列理论（Havel–Hakimi 变体）和可实现性判定；
- 组合构造与局部操作（edge addition、leaf merge、vertex split）来逐步实现目标度数；
- 对已知问题的归约（例如将配对顶点覆盖归约为优化顶点覆盖），利用现有的线性时间实现；
- 通过分析森林的结构（叶子数、连通分量数、度数分布）得到精确的下界与上界；
- 归纳与递推证明（如证明增量操作保持森林性、度数不超限）。

**📊 数据集**

本文为纯理论研究，没有使用具体实验数据集。所有结论均通过严格的数学证明与算法分析得到。

**📈 对比分析**

对比方法主要是与已知的“优化顶点覆盖”实现以及传统的 Havel–Hakimi 等算法进行对比。性能方面：
- 所有算法均为线性时间（O(n) 或 O(∑d_i)），与现有的最优实现保持一致；
- 通过构造性证明表明在给定度序列的前提下，得到的解必为最优，因而不存在更优解可比较；
- 对特殊情况（如所有度均为 1 或 2 时）算法退化为常数时间。

**⚠️ 局限性**

限制与不足：
- 仅适用于森林度序列，对一般图（如含环或高连通图）不适用；
- 对于极端度数分布（如极大度数接近 n），增量操作可能导致实现步骤较多，虽然整体仍为线性，但实际常数因子可能较大；
- 算法依赖于度序列的非递增排序，若输入序列无序，需要先做一次排序，额外耗时 O(n log n)；
- 在实际应用中，若需要在已存在森林上动态调整度序列，算法需要重新构造，未给出高效的增量更新方案。

---

## 476. Robocasa-Mem: Benchmarking Environment and Process Memory in Long-Horizon Mobile Manipulation

**arXiv ID:** 2611.06475 | [PDF](https://arxiv.org/pdf/2611.06475v1)

**作者:** Sihang Chen `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了名为 RoboCasa‑Mem 的新型移动操纵基准，包含九个需使用交互历史的任务，并提供 1,800 条成功演示，用以研究环境记忆与过程记忆的必要性。

**💡 创新点**

创新点在于：①将记忆需求细分为环境记忆（空间、属性、关系）和过程记忆（单任务进度、多任务调度）；②引入行为条件评估（Behavior‑Conditioned Evaluation）以在多路径任务中评估记忆一致性；③提出层次化记忆增强策略（HMAP），将结构化语义记忆与 3D 视觉几何结合。

**🔧 技术方法**

技术上使用 Qwen3‑VL‑4B‑Instruct 做高层子目标规划，VGGT‑Ω 提取多视角几何特征并通过任务条件查询对低层控制进行调制；两者分别在离线监督下训练，随后联合推理。

**📊 数据集**

数据集为 RoboCasa‑Mem，包含 1,800 条成功演示轨迹，覆盖 9 个任务，环境随机化涵盖物体位置、外观、执行顺序等，并在每个任务中收集 200 条演示。

**📈 对比分析**

与 ACT、Diffusion Policy、π_0.5、MemER‑style 等基线对比，HMAP 在平均任务成功率（SR）从 0.16 提升至 0.24、阶段完成率（SCR）从 0.41 提升至 0.47；在环境记忆子任务中显著提升，尤其是隐藏属性回忆；但多任务调度子任务仍未完成。

**⚠️ 局限性**

局限性包括：评估主要在仿真环境，真实世界实验仅包含 10 次单一任务；仅使用成功演示，缺乏失败案例分析；记忆类别并不完整，无法完全覆盖所有可能的历史依赖；阶段级别指标难以完全区分记忆错误与执行错误。

---

## 477. Weaver: Unlocking Blockchain Transaction Parallelism through State-Decoupled Symbolic Execution

**arXiv ID:** 2611.06462 | [PDF](https://arxiv.org/pdf/2611.06462v1)

**作者:** Ru Cheng `[一作]` (City University of Hong Kong), Chenyuan Wu `[通讯]` (City University of Hong Kong)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种状态解耦的区块链交易执行引擎，利用轻量符号执行和表达式粒度调度，将交易执行与具体状态访问分离。

**💡 创新点**

通过符号引用实现状态解耦、分层符号模式匹配的分支预测以及表达式粒度的并行调度，最大化跨状态访问的并行性。

**🔧 技术方法**

轻量符号执行、分层符号模式匹配的分支预测、表达式粒度调度、异步状态访问、合并同一状态请求与中间结果缓存。

**📊 数据集**

真实以太坊主网区块 50k 块（约 1000 万交易）及四类合成 dApp（ERC‑20、AMM、抵押借贷、NFT 市场）的 1000 万交易。

**📈 对比分析**

与序列执行、OCC、BlockSTM、ParallelEVM 进行对比，在多线程场景下吞吐量提升 41–63% 以上，单线程 3.3× 以上。

**⚠️ 局限性**

对分支预测误判仍需恢复、对极端高冲突/热点状态依赖仍有限、需要额外的内存来存储表达式和预测表。

---

## 478. Generative Learning on Dynamic Text-Attributed Graphs

**arXiv ID:** 2611.06458 | [PDF](https://arxiv.org/pdf/2611.06458v1)

**作者:** Zhenyu Yang `[一作]` (Xiamen University), Jintang Li `[通讯]` (Xiamen University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于语义标识符（SID）的生成式框架DyGen，用来在动态文本属性图中进行未来链路预测。

**💡 创新点**

创新点在于：①将节点文本映射为离散且紧凑的SID，既可在无历史交互时提供语义基础，又能随时间更新；②使用Transformer的自回归解码器对目标SID进行逐码预测，兼顾文本语义与事件级时间信息；③在推理时仅编码一次查询上下文，重用SID似然实现候选高效排序，并通过轻量化结构校准提升性能。

**🔧 技术方法**

核心技术包括：BERT文本编码、残差量化（RQ‑VAE）生成SID、Transformer编码器-解码器、时间差分桶嵌入、结构校准（重复、共现、流行度）以及统计置信区间评估。

**📊 数据集**

实验使用了九个公开的动态文本属性图数据集（Wikirevision、Amazon Movies、IMDB、Dianping、Weibo‑Daily、Weibo‑Tech、Wikilife、Sephora、Cora）以及亚马逊电影推荐子集。

**📈 对比分析**

与多种基线（EdgeBank、GraphMixer、TGN、DyGFormer、TPNet、TIGER）对比，DyGen在多数数据集上获得与或优于最强基线的 Hits@10、Hits@5、MRR 等指标，尤其在较稠密图和冷启动场景下表现突出；在Amazon Movies的 Hits@1/Hits@5/MRR 方面显著优于其他方法。

**⚠️ 局限性**

限制包括：①对大规模候选集的推理仍需逐个计算 SID 似然；②在混合目录（温/冷）校准方面效果尚未成熟；③对极稀疏图或极大冲突桶的 SID 冲突仍可能产生歧义；④模型对文本质量与BERT固定编码的依赖，未考虑文本更新带来的语义漂移。

---

## 479. Partially Interpretable Decision Trees

**arXiv ID:** 2611.06428 | [PDF](https://arxiv.org/pdf/2611.06428v1)

**作者:** Ludovic Schwartz `[一作]` (University of Amsterdam), Tim van Erven `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了部分可解释决策树（Partially Interpretable Decision Trees），并提出三种学习算法（精确搜索、CART转换、PIM-entropy自适应分裂），将其应用于概念瓶颈模型；

**💡 创新点**

首次将部分可解释框架扩展到决策树，给出三种高效实现方案，并提供理论转换误差上界，展示在黑盒性能均匀时几乎无性能损失；

**🔧 技术方法**

利用精确搜索（AO*-type）、CART树结构+成本复杂度剪枝、PIM-entropy分裂准则及概念瓶颈模型深度树；

**📊 数据集**

在20个公开数据集（UCI、CUB-200等）和人工合成数据集上进行实验；

**📈 对比分析**

三种方法在实测中表现相近，平均误差差异不足0.6%，在特定数据集（如Monk）中自适应分裂优于其他方法；

**⚠️ 局限性**

理论界限在黑盒性能不均匀时失效；目前仅适用于二分类，需进一步扩展到多分类并估计不均匀度。

---

## 480. A Bag of Tricks for Defending Deep Learning-based Face Recognition Pipelines against Backdoor Attacks

**arXiv ID:** 2611.06412 | [PDF](https://arxiv.org/pdf/2611.06412v1)

**作者:** Quentin Le Roux `[一作]` (Thales Cyber & Digital), Philippe Loubet Moundi `[通讯]` (Thales Cyber & Digital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了三种针对多阶段面部识别系统的后门防御方案，覆盖训练阶段和推理阶段；

**💡 创新点**

首次在开集、多组件的真实面部识别系统环境下系统级别地设计防御，并提出早期毒化识别、检测器级模型配对和切割拼接三种创新技术；

**🔧 技术方法**

采用训练时内部身份准确率追踪、推理时检测器与辅助模型对比检测框和关键点差异，以及切割/拼接混合输入并利用余弦相似度异常检测；

**📊 数据集**

使用WIDER‑Face、CelebA、MS1Mv2、LFW、CFP‑FF/FP、AgeDB、CALFW、CPLFW、VGGFace2_FP以及CelebA‑Spoof等公开数据集进行训练、验证与攻击实验；

**📈 对比分析**

通过与多种后门（BadNets、SIG、FauxIt、Mask等）对比，Early Poison Identification在All‑to‑One Poison‑label后门能在早期阶段识别并可选择性裁剪；Detector‑level Model Pairing在FGA/LSA后门下FPR≈0.8%，FNR≈4.3%；Cut‑and‑Mix在大部分后门下FPR≈0.5–1.1%，FNR<1%（仅弱后门略高），整体显著提升防御效果；

**⚠️ 局限性**

对All‑to‑One Clean‑label后门无效；仅针对检测器和特征提取器，未覆盖抗伪造层；对更强或可适应的后门、UniID等单方触发攻击缺乏针对性；需进一步提升对弱后门与更复杂攻击的鲁棒性。

---

## 481. Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs

**arXiv ID:** 2611.06438 | [PDF](https://arxiv.org/pdf/2611.06438v1)

**作者:** Nuo-Jing Liang `[一作]` (Nanjing University), Pan Zhou `[通讯]` (Singapore Management University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了链式思维（CoT）推理在大语言模型（LLMs）中的作用，特别是其在分布外（OOD）泛化机制的理论分析。

**💡 创新点**

提出了CoT作为组合性OOD泛化的框架，证明了复杂OOD任务的风险可以通过ID子任务风险的总和来上界，并且与推理深度T无关。

**🔧 技术方法**

使用了Rademacher复杂度分析和自回归下一个标记预测（NTP）技术，分析了多步推理的风险。

**📊 数据集**

在Atom-Task和GSM8K数据集上进行了实验，验证了理论结果。

**📈 对比分析**

通过与无CoT设置的比较，CoT训练的模型在短链到长链的任务中表现出更好的泛化能力，且在GSM8K数据集上也显示出相似的结果。

**⚠️ 局限性**

理论分析主要集中在监督NTP风格的训练上，未考虑自适应范式如强化学习微调和测试时搜索，这些未来的研究方向将是自然的扩展。

---

## 482. Where Should a KV-Cache Repair Budget Go? Distractor Structure and Contextual Compatibility in Non-Prefix Reuse

**arXiv ID:** 2611.06397 | [PDF](https://arxiv.org/pdf/2611.06397v1)

**作者:** Yuxing Peng `[一作]` (Beijing Normal University), Weijia Jia `[通讯]` (Beijing Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在检索增强生成中使用非前缀KV缓存重用时的误差来源，并探索在修复预算有限的情况下应将修复工作聚焦在哪个部分；

**💡 创新点**

创新点在于将重用损失拆解为通用损失、同类损失和相似性损失三部分，证明将修复预算投入到答案块（target）而非干扰块可获得更高收益，并提出通过将检索器最高块移动至问题前并重新编码的方式显著提升性能；

**🔧 技术方法**

使用了位置校正的KV缓存重用、Token‑level 修复策略（CacheBlend、EPIC、ProphetKV 等）、对 target 进行重编码、检索块重排、InfoFlow KV 重排等技术；

**📊 数据集**

实验数据集包括对话记忆 LoCoMo、开放域 QA（NQ‑Open、TriviaQA）以及 API 文档 Gorilla；

**📈 对比分析**

在 7B‑8B 规模模型下，在 5%–10% 修复预算上，与 token‑level selector、InfoFlow KV 重排等方法对比，重编码 target 或移动检索块可使保留率提升约 0.1–0.5（相对 5–10% 的显著提升），在开放域 QA 与 API 文档任务中均取得显著优势；

**⚠️ 局限性**

局限性包括仅在 7B‑8B 与两大 Qwen3 模型上验证，未覆盖更大模型或多语言；仅在检索构造的上下文中测试；只考察小预算；延迟测量仅在实验引擎内，未评估真实部署；方法在部署时对检索质量高度依赖。

---

## 483. GIST: Gibbs-Targeted Hierarchical Data Selection for Offline RLVR

**arXiv ID:** 2611.06395 | [PDF](https://arxiv.org/pdf/2611.06395v1)

**作者:** Xuliang Huang `[一作]` (Hong Kong University of Science and Technology), Yao Shu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了离线 RLVR（可验证奖励）中数据选择问题，推导出基于双向 KL 不匹配的响应和词级选择原理，并提出了 GIST（Gibbs‑Targeted Hierarchical Data Selection）两层选择方法，随后在多模型、多目标上验证其有效性。

**💡 创新点**

创新点在于：1) 通过双向 KL 值量化策略与 Gibbs‑最优目标之间的不匹配，并将其分解为响应级和词级的前向（覆盖）和逆向（过度承诺）两类误差；2) 基于上述理论，设计了仅利用采样温度、验证结果和初始策略词概率的 GIST 选择框架；3) 在离线训练中同时兼顾响应筛选与词级监督，显著提升准确率并减少监督 token 数量。

**🔧 技术方法**

技术手段包括：KL‑正则化 RLVR 目标、BOLT（加权似然）与 SFT（简单复制）两种离线训练目标、基于曲率的双向 KL 分数与响应/词级贡献量化、GIST 的温度配对、词概率阈值掩码、以及实验中的验证器与数据池构建流程。

**📊 数据集**

使用的数据集包括：数学推理基准 GSM8K、MATH‑500、代码生成基准 MBPP‑full；模型为 Qwen3‑0.6B、Qwen3.5‑2B 与 Gemma‑4‑E2B‑it。

**📈 对比分析**

与七种基线（随机、RFT‑style、HES、PODS、ProFit、ETS、RSI‑S）以及完整候选集训练进行对比；在所有模型、基准与评估指标（Strict、Flexible、pass@1）上，GIST 均显著提升 1.1–4.4 分，并在共享候选池中相较全量训练减少 66.6% 的监督 token，且训练步骤数约减半。

**⚠️ 局限性**

局限性包括：① 依赖于初始策略和可验证奖励的准确性；② 选择过程一次性完成，难以适应策略随训练动态变化的情况；③ 对低概率词的掩码依赖于阈值，可能在不同模型/任务上需要调优；④ 未在更大规模模型或跨域数据上验证其普适性。

---

## 484. MA-Mesh: Resilient Multi-Agent Collaboration via Role-based Self-Healing

**arXiv ID:** 2611.06389 | [PDF](https://arxiv.org/pdf/2611.06389v1)

**作者:** JunYong Lee `[一作]` (Sungkyunkwan University), Honguk Woo `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Multi-Agent Mesh (MA-Mesh) 框架，在多智能体协作系统中构建基于角色的自愈层，以提升系统鲁棒性。

**💡 创新点**

创新点在于将协作逻辑与自愈策略解耦，采用跨代理反射和角色基失败切换实现动态自愈，并借鉴服务网格思路实现可插拔自愈层。

**🔧 技术方法**

使用技术包括 LLM orchestrator、可编程的 agent 与 resilient proxy、cross-agent reflection、role-based failover、自动代码合成以及云原生 chaos engineering 评估。

**📊 数据集**

实验使用的软件开发数据集 (GUI Dev., Web Dev.)、工具使用数据集 (GAIA)、科学推理数据集 (GPQA-Diamond) 以及 AIOpsLab 云原生测试平台。

**📈 对比分析**

与 CoT、MetaGPT、RTADev、AgentVerse、EvoMAC、C+I 等基线比较，MA-Mesh 在四种 chaos 场景下平均提升 30-45pp 准确率，检测召回率更高，API 成本与耗时略低。

**⚠️ 局限性**

局限性包括：自愈代理的推理模型采用静态分配，缺少动态弹性调度；假设自愈代理与 meta-agent 永久可用，未考虑其故障；对更大规模多任务或多语言场景的验证不足。

---

## 485. Partially Dynamic Algorithms for Single-Source Temporal Paths

**arXiv ID:** 2611.06387 | [PDF](https://arxiv.org/pdf/2611.06387v1)

**作者:** Davide Bilò `[一作]` (University of L'Aquila), Anna Zych-Pawlewicz `[通讯]` (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了针对时序有向图中单源最早到达时间的部分动态维护算法，涵盖增量、递减以及可指定出发时间阈值的情形，提供常数查询时间和总更新时间为O(n+m log m)（增量/递减）或O(n+Δ_s m log m)（可变出发时间）；同时给出了基于OMv猜想的匹配下界，证明在此假设下更快的总时间不可实现。

**💡 创新点**

创新点在于：①设计了高效的增删弧动态维护最早到达树，实现常数查询和近线性总更新；②扩展到任意出发时间阈值，仍保持O(log m)查询和O(n+Δ_s m log m)总更新；③提供了匹配的下界，表明所给算法在理论上已达到最优；④对无向时序图同样适用。

**🔧 技术方法**

主要技术包括：平衡二叉搜索树（AVL）存储弧集合、队列传播改进最早到达树、增量/递减维护各类弧集合（入/出）、时间标签/延迟的键排序与查询、以及利用OMv猜想的红黑图构造来证明下界。

**📊 数据集**

论文为理论性研究，并未使用具体实验数据集，而是以图模型的抽象构造（如三分图、矩阵图）来展示算法效果和下界证明。

**📈 对比分析**

由于未给出实验结果，比较基准仅为理论复杂度。作者指出在OMv假设下，已实现的O(n+m log m)总更新与匹配的下界相符，说明在可接受的复杂度范围内已达到最优。

**⚠️ 局限性**

局限性包括：①只能处理单源最早到达问题，未覆盖多源或最短路径；②性能上仍受m log m因子限制，对非常稠密图可能不够高效；③下界依赖OMv猜想，若该猜想被证明错误则下界可能被打破；④对实际时序网络的实验验证尚缺失。

---

## 486. Measuring the collaboration, not the agent

**arXiv ID:** 2611.06487 | [PDF](https://arxiv.org/pdf/2611.06487v1)

**作者:** Xinxiao Li `[一作]` (Shonan Institute of Technology), Eisei Nakahara `[通讯]` (Nihon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个针对高维抽象数据的分析师与AI代理协作评估框架，并实现了一个可复现的测试平台，用以计算并验证四个基于交互日志的衡量轴。

**💡 创新点**

创新点在于：①将协作水平的度量拆分为五个可操作的轴（互补性、效率、依赖性、知识保留、代理信任/工作负荷）；②用交互日志实现自上而下的自动化评估；③构建了一个可模拟分析师的实验环境，以验证指标的可计算性与区分度。

**🔧 技术方法**

主要技术包括：规则驱动与LLM结合的视图生成代理；基于事件流的日志收集与分析；对齐损失函数（AUC/误差）与代理记忆（持续纠错）；统计检验（bootstrap CI）来评估指标的显著性。

**📊 数据集**

使用了 UCI SECOM 半导体生产数据集（590个传感器特征，1,567 条样本），通过稳定性选择与交叉验证确定的 8 个关键特征作为代理学习的目标集合。

**📈 对比分析**

通过设定五种实验条件（随机提议、分析师单人、代理单人、协作、协作但不保留反馈）并对 4,980 个会话进行模拟，验证了四个日志指标能够清晰区分各条件；在代理单人/协作条件下表现出互补性提升、干预效率提升、依赖性与覆盖率合理分布、知识保留显著改善。

**⚠️ 局限性**

局限性包括：未进行真实人类实验，指标尚未验证对人类分析师的适用性；目标特征 F* 是基于监督模型的代理性代理，非真实物理因子；模拟分析师的参数人为设定，缺乏人类行为拟合；协作形式仅限于持久纠错，未覆盖更丰富的交互方式。

---

## 487. What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents?

**arXiv ID:** 2611.06483 | [PDF](https://arxiv.org/pdf/2611.06483v1)

**作者:** Kejian Zhu `[一作]` (Chinese Academy of Sciences), Jun Zhao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过Pass@(K,T)、单步质量和机制分析，研究了多轮代理强化学习在不同交互预算下的能力提升与局限；

**💡 创新点**

创新点在于揭示RL增益高度依赖交互预算、单步动作质量并不均衡提升，并提出BudgetMix RL在训练时动态变换预算来提升鲁棒性；

**🔧 技术方法**

采用多轮RL（GRPO、GiGPO、PPO）、Pass@(K,T)评价指标、对数似然与训练动态分析，并设计BudgetMix RL框架；

**📊 数据集**

使用FrozenLake、DeepCoder-MT、BFCL-v3、InterCode-SQL四个交互式基准数据集进行实验；

**📈 对比分析**

与传统全预算RL对比，在25%-75%预算下BudgetMix RL平均提升约7-13个百分点，且在100%预算下保持与全预算RL相当的性能；

**⚠️ 局限性**

限制在于RL增益对足够预算敏感，单步动作质量可能不提升甚至下降，且高质量过程监督信号难以获得。

---

## 488. ArcRAG: Agentic Compositional Retrieval over Knowledge Graphs

**arXiv ID:** 2611.06481 | [PDF](https://arxiv.org/pdf/2611.06481v1)

**作者:** Jyotin Goel `[一作]` (Domyn), Stefano Pasquali `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ArcRAG，一种在不同知识图谱构建方式下均能有效检索并生成答案的检索增强生成框架；

**💡 创新点**

核心创新在于在推理时动态诱导查询局部图接口，分离检索与图构建的耦合，使同一检索程序可跨不同语义和结构的图转移；

**🔧 技术方法**

结合 LLM 进行探索、规划与生成，使用 Cypher 查询、节点/关系自适应匹配，以及多轮重试与跟进；

**📊 数据集**

在 MuSiQue、HotpotQA 与 TriviaQA 三大 QA 语料上分别构建两类知识图谱（FinReflectKG 结构化、HippoRAG 开放式）进行评测；

**📈 对比分析**

与直接 Cypher 生成、前置检索等基线对比，ArcRAG 在 EM/F1 指标上分别在 MuSiQue、HotpotQA、TriviaQA 上取得最高或相近最佳成绩，且在两种图构建下保持竞争力；

**⚠️ 局限性**

实验仅涵盖两种极端构建方式，未分离结构与出处对准确性的影响；未验证事实正确性，且仅针对百科问答，缺乏对监管领域的可靠性与人类监督评估。

---

## 489. Towards Early Warning of Real-World Vulnerabilities: An Evidence-Aware Decoupled Method for CVSS Vector Prediction Based on Unstructured Disclosures

**arXiv ID:** 2611.06477 | [PDF](https://arxiv.org/pdf/2611.06477v1)

**作者:** Jianhong Ma `[一作]` (Zhengzhou University), Haowen Chen `[通讯]` (Information Engineering University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出PARSE生命周期与VPre框架，实现基于早期OSINT的CVSS向量预测。

**💡 创新点**

创新点在于两级Evidence-Aware Decoupling：一级安全证据筛选（STDU）消除噪声，二级依据验证指标将八项CVSS基准拆分为专门的SER与EEC路径。

**🔧 技术方法**

技术上使用LLM（GPT-4o-mini）进行上下文推理，DistilBERT/ BERT-Small进行句子级分类与监督分类，并通过验证集计算证据密度D(m)决定路由。

**📊 数据集**

使用自建的VPre-Bench（5,581条早期报告）和PARSE-Triage（≈47k高噪声报告）两个数据集。

**📈 对比分析**

与直接迁移的BERT-Small-NVD、CVEDrill、KBS 2025等SOTA基线以及纯LLM、轻量化单模型对比，VPre在早期OSINT上MAE降至0.7239，平均提前38天预测，显著优于所有基线。

**⚠️ 局限性**

限制包括：仅英文、仅文本OSINT，可能存在残留时间泄露、对多语言和代码更改等信息的覆盖不足，以及MAE/MSE未涵盖所有业务成本。

---

## 490. A 3D foundation model unifying CT interpretation and segmentation

**arXiv ID:** 2611.06471 | [PDF](https://arxiv.org/pdf/2611.06471v1)

**作者:** Yulu Xie `[一作]` (Tsinghua University), Fan Wang `[通讯]` (Alibaba DAMO Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

未提供论文内容，无法完成摘要。

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 491. AmbiHTMLBench: A Benchmark for HTML-based Visually Rich Document parsing

**arXiv ID:** 2611.06449 | [PDF](https://arxiv.org/pdf/2611.06449v1)

**作者:** Boyu Niu `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 AmbiHTMLBench，一个从文档图像重建可执行 HTML 的基准系统。

**💡 创新点**

提出客观与主观双重评测框架，分别量化内容、结构和视觉一致性，并揭示两类评测结果的互补性。

**🔧 技术方法**

使用多模态图像转 HTML 模型（如 Gemini、Claude、GPT、Qwen 等）以及自动块匹配、Hungarian 匹配和 VLM 代理评估技术。

**📊 数据集**

基于 1,115 页多样化真实文档的专家标注数据集。

**📈 对比分析**

在 15 个模型上进行客观和冻结主观评测，Gemini 在客观榜单首位，GPT‑5.6‑Sol 在主观榜单首位，二者相关性中等。

**⚠️ 局限性**

局限包括样本规模有限、评价依赖复杂的多指标匹配、主观代理与人工判断存在偏差。

---

## 492. $A^3$-VLA: Automatic Perception-Guided Attention Alignment for Robot Manipulation

**arXiv ID:** 2611.06447 | [PDF](https://arxiv.org/pdf/2611.06447v1)

**作者:** Shengzhe Zhang `[一作]` (University of Science and Technology of China), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于自动感知引导注意力对齐的机器人操纵模型A³‑VLA，利用交互过程生成伪区域并在训练时用注意力分配与对比学习双重监督提升视觉‑语言‑动作模型的任务感知与执行性能。

**💡 创新点**

创新点在于：①无需人工标注即可从操纵轨迹中自动提取目标对象与机器人伪区域；②通过注意力分配损失让模型关注这些任务关键区域；③通过对比学习将目标区域特征与语言指令对齐，从而保持视觉注意力稳定。

**🔧 技术方法**

采用的技术包括：视觉‑语言‑动作模型（如π₀、OpenVLA‑OFT、π₀.5）作为主干；交互诱导的候选区域生成与SAM2细化；注意力分配损失与对比学习损失的联合训练。

**📊 数据集**

实验数据集覆盖VLABench（Chemistry Tube、Fruit、Poker、Toy、Mahjong）和Robotwin 2.0（Beat Block Hammer、Click Bell、Move Playingcard Away、Place Burger & Fries）等模拟任务，以及真实厨房环境中的Open Cooker、Pick & Place Bowl、Pick 2 Vegetables、Select Bottles、Desktop Sweep共80条演示。

**📈 对比分析**

与Octo、OpenVLA、RDT、π₀、π₀‑FAST、OpenVLA‑OFT、π₀.5等最先进方法对比，A³‑VLA在VLABench平均提升16.4%–25.2%，在Robotwin平均提升6.5%–9.8%，在真实世界平均提升14%，并且在纹理与噪声扰动下保持更高鲁棒性。

**⚠️ 局限性**

局限性在于：①伪区域生成依赖交互时刻的准确检测，误差可能影响指导效果；②对复杂多交互任务的区分仍需改进；③对大规模多模态场景的迁移性和实时性尚未充分验证。

---

## 493. TwinBridge: Digital Twin-Driven Policy and Reward Transfer for Real-World Reinforcement Learning in Laboratory Robotics

**arXiv ID:** 2611.06442 | [PDF](https://arxiv.org/pdf/2611.06442v1)

**作者:** Shu'ang Sun `[一作]` (University of Science and Technology of China), Mingzhai Sun `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一套名为TwinBridge的数字孪生驱动框架，用于在现实实验室任务中预训练操控策略并生成密集的视觉进度奖励，随后在真实机器人上进行人机协作式在线强化学习；

**💡 创新点**

创新点包括：①将3D高斯散点与物理仿真相结合生成数字孪生，实现任务事件与几何进度的同步标注；②基于仿真状态构造物理约束下的事件与进度奖励，并通过可执行历史账本实现一次性事件计分与可验证的进度记录；③将仿真生成的监督信息与少量真实视觉标注融合，训练出能在真实视觉输入下预测事件与进度的视觉奖励模型；

**🔧 技术方法**

技术要点包括：3D高斯散点渲染、物理仿真（Isaac Sim）、Octo编码器、门控循环单元（GRU）视觉预测器、潜在奖励塑形（PBRS）、HIL-ConRFT演员-评论家架构、执行历史账本等；

**📊 数据集**

使用了实验室真实机器人任务数据集，共五个任务（离心管拾取、培养皿盖拆除、离心管装载、吸头附着、吸取液体），以及对应的仿真数据；

**📈 对比分析**

与传统稀疏奖励在线RL（HIL-ConRFT、HG-DAgger、HIL-SERL）以及无仿真策略初始化的TwinBridge进行对比；在五个任务上，TwinBridge最终自主成功率平均达98%（四个任务100%，离心管装载90%），相比无仿真初始化的平均94%（相差4个百分点）以及HIL-ConRFT平均44%（相差54个百分点）显著提升；

**⚠️ 局限性**

局限性包括：依赖高质量的数字孪生与仿真一致性，若仿真与真实差异大则奖励转移效果下降；需要一定量的真实视觉标注以校准视觉奖励模型；目前仅在相对简单的实验室任务中验证，尚未检验在更复杂或多模态任务中的可扩展性。

---

## 494. Interpretable Text Embeddings Based on Hierarchical Questions

**arXiv ID:** 2611.06402 | [PDF](https://arxiv.org/pdf/2611.06402v1)

**作者:** Kohei Oda `[一作]` (Japan Advanced Institute of Science and Technology), Natthawut Kertkeidkachorn `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了层次化的 QA‑based 文本嵌入框架，让文本被表示为多层次问答向量，便于解释与控制维度数量。

**💡 创新点**

创新点在于递归聚类结合对比式提问生成多层级问题，并利用 NLI 对齐将深层问题归纳为浅层抽象问题，从而显著降低认知负荷。

**🔧 技术方法**

使用技术包括递归 k‑means 聚类、对比式问题生成 (CQG)、基于 NLI 的问题对齐、MBQA 训练和阈值化决策。

**📊 数据集**

数据集涵盖 MEDI2 作为大规模语料库，以及 MTEB benchmark（STS、文本聚类、信息检索）和 MS MARCO 检索任务进行案例分析。

**📈 对比分析**

通过与单层 CQG‑MBQA 基线在 Spearman、V‑Measure、nDCG@10 等指标下对比，发现第四层与基线性能相当，第三层在任务性能与认知负荷的 Pareto 前沿上实现了显著提升。

**⚠️ 局限性**

局限性包括：仍需人工审阅大量生成问题、缺乏多语言评估、未直接提升任务性能，仅优化了性能与认知负荷的权衡。

---

## 495. LibriSpeech-Real-QA: A Benchmark for Streaming Question Answering

**arXiv ID:** 2611.06496 | [PDF](https://arxiv.org/pdf/2611.06496v1)

**作者:** Zewen Zhan `[一作]` (SenseTime Research), Zhiyong Wu `[通讯]` (SenseTime Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了 LibriSpeech-Real-QA 基准，用于评估实时语音交互中的事件触发问答。

**💡 创新点**

提出了基于时间窗口的实时评估协议，并给出流式微调模型在四类任务（词检测、词计数、语义检测、语义计数）上的基线结果。

**🔧 技术方法**

采用预训练的 Qwen2.5-Omni-7B 进行流式微调（LoRA、负样本、特殊 silence token），同时利用多模态对齐、语义分类等技术实现事件级即时响应。

**📊 数据集**

基于 LibriSpeech 语料构建数据集，标注词级时间戳、20 类语义标签，并生成正负样本。

**📈 对比分析**

与商用实时 API（Gemini、GPT、Qwen-Audio）以及离线推理模型对比，流式微调模型在所有四类任务上均取得更高 F1 并在 160–400 ms 的 chunk 里实现 0.5–0.7 s 的平均起始延迟。

**⚠️ 局限性**

存在模型在语义任务上的准确性不足、对不同 chunk 长度的延迟与精度折衷、以及评估窗口大小对结果影响较大的局限。

---

## 496. A $2.81^n$ upper bound on the number of stable matchings

**arXiv ID:** 2611.06494 | [PDF](https://arxiv.org/pdf/2611.06494v1)

**作者:** Ebrahim Ghorbani `[一作]` (Hamburg University of Technology), Matthias Mnich `[通讯]` (Hamburg University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了具有n个男性和n个女性的稳定匹配实例中稳定匹配的最大数量(n)。

**💡 创新点**

证明了(n)<2.81^n，改进了之前的上界，接近已知的下界2.28。

**🔧 技术方法**

使用了归纳法和权重分配的方法来界定每个人的稳定伙伴数量，从而限制稳定匹配的数量。

**📊 数据集**

没有具体提到使用的数据集，但研究涉及的实例是具有n个男性和n个女性的稳定匹配实例。

**📈 对比分析**

与之前的研究相比，新的上界(n)≤2.81^n显著改进了Karlin等人和Palmer与Pálvölgyi的结果，后者分别给出了(n)≤2^17n和(n)≤3.55^n+O(1)。

**⚠️ 局限性**

限制在于该方法依赖于每个人的稳定伙伴数量，可能在某些情况下不适用，且未考虑偏好列表不完整的情况。

---

## 497. Model Merging Does Not Require Uniform LoRA Ranks

**arXiv ID:** 2611.06486 | [PDF](https://arxiv.org/pdf/2611.06486v1)

**作者:** Zirong Xie `[一作]` (Amazon), Jack Butler `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在模型融合时专家模型的LoRA低秩(rank)是否需要统一，发现可以让每个专家使用最适合其领域的rank；

**💡 创新点**

提出了基于梯度导向的组件选择方法GGSM，并证明相较于传统按幅度截断的方法，允许不同rank可提升多域准确率；

**🔧 技术方法**

主要技术包括LoRA微调、SVD与GGSM截断、TIES、KnOTS等融合算法、梯度导向的效用评分；

**📊 数据集**

实验使用Qwen3.5系列模型（0.8B、2B、4B）与三大任务域：数学（DART-Math/GSM8K）、指令跟随（TULU-3/IFEval）、多语言（Aya/MMLU/ARC/ARC-zh）；

**📈 对比分析**

对比六种融合方法，发现简单平均、SVD和GGSM对不同rank不敏感，且在最佳多域配置下相较于统一rank提升最多1.8个百分点；而TIES和KnOTS在rank异质时表现不稳定，甚至下降多达4个百分点；

**⚠️ 局限性**

局限包括仅使用LoRA微调、三种任务域、单一模型家族、实验种子有限，因而多域提升的显著性仍需更大样本验证

---

## 498. Polynomial $(2α+\varepsilon)$- and Quadratic $4$-Approximation Algorithms for the Biased Traveling Salesman Problem in Bipartitioned Networks

**arXiv ID:** 2611.06410 | [PDF](https://arxiv.org/pdf/2611.06410v1)

**作者:** Jingyang Zhao `[一作]` (Kyung Hee University), Hyundong Shin `[通讯]` (Kyung Hee University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了两种新型近似算法，用于解决在二分网络上受偏差影响的旅行商问题（BTSP），从而显著提升了算法的近似比和时间复杂度；

**💡 创新点**

创新点在于：①通过对最优Hamiltonian环中跨区边的枚举，将BTSP转化为TSPPK问题，进而获得<2α+ε（<3）的近似比；②对跨区边进行结构化补丁，避免枚举过多组合；③对最小生成树的拓扑（0‑MST、1‑MST、mono‑cross、poly‑cross、mono‑depth、poly‑depth）进行分类，设计了O(n²)时间的4‑近似算法，并在各子案例中分别实现5/2、2、8/3的更佳比值；

**🔧 技术方法**

核心技术包括：TSPPK约化、α‑近似的度量TSP与TSPP、跨区边枚举与序列化、图的补丁技术、最小生成树拓扑判定、迭代对合并（pairwise merging）与2‑跳操作、Eulerian多重图构造与短路（shortcutting）、基于LP的TSPP近似以及将匹配替换为高效Hamiltonian路径子程序；

**📊 数据集**

论文为理论性工作，未使用实际数据集；

**📈 对比分析**

与原有的7/2‑近似算法（时间O(n³））进行比较；首次突破3‑近似上界，得到<2α+ε<3（α≈1.5）且时间为多项式；第二个算法将时间从O(n³)降至O(n²)，并在各子案例中取得5/2、2、8/3的更优比值，总体仍保持4‑近似；

**⚠️ 局限性**

限制：①近似比仍高于最优1.5，尚未达到BTSP的最佳可能比；②2α+ε算法的时间与ε成指数关系，ε→0时复杂度急剧上升；③对超过两分区的BTSP未给出结果；④算法均基于理论证明，缺乏实验验证与实际网络数据评估；

---

## 499. eBPF-Driven Adaptive Lifetime-Aware Data Placement for Shared FDP SSDs

**arXiv ID:** 2611.06461 | [PDF](https://arxiv.org/pdf/2611.06461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 500. Beyond Gomory-Hu: All-Pairs $\times$ All-Pairs Mincuts in $O(n)$ Space

**arXiv ID:** 2611.06478 | [PDF](https://arxiv.org/pdf/2611.06478v1)

**作者:** Koustav Bhanja `[一作]` (Weizmann Institute of Science), Asaf Petruschka `[通讯]` (Weizmann Institute of Science)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

为任意终端对构造了O(n)空间、O(n)查询时间的x,y最小割单元分解数据结构；

**💡 创新点**

创新点在于提出α-组件、循环一致性与最大化增量的投影技术，使得可处理未被Steiner最小割分隔的终端对；

**🔧 技术方法**

使用图割理论、投影、cactus结构、子模/对数理论及辅助LCA/级别祖先结构；

**📊 数据集**

本文为理论研究，无实验数据集；

**📈 对比分析**

与之前的O(n log n)空间/查询方案相比，取得了线性空间、查询时间的最优结果；

**⚠️ 局限性**

仅适用于单点查询，且预处理复杂度较高，动态更新不支持。

---

## 501. SUMONE: A Simple and Efficient Summarizer for Plug-and-Play Agent Memory

**arXiv ID:** 2611.06445 | [PDF](https://arxiv.org/pdf/2611.06445v1)

**作者:** Jizhe Zhang `[一作]` (Peking University), Yisen Wang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练一个独立的 2B 参数压缩模型 SumOne，通过遮蔽重建自监督方式，将长周期代理的交互历史压缩成简短摘要，从而实现上下文压缩并降低推理成本。

**💡 创新点**

创新点在于：①使用遮蔽重建作为任务无关的奖励信号，无需下游任务或参考摘要即可训练压缩器；②得到的压缩器轻量且可直接插入不同 LLM 和 agent harness，兼容性强；③在多种基准上优于自我总结、Prompt-2B 和 CoMem 等现有方法。

**🔧 技术方法**

技术手段包括：GRPO 强化学习框架、遮蔽重建（mask reconstruction）奖励、冻结的 9B 重建器与 LLM 评判器、以及 LongMemEval‑S 评估框架。

**📊 数据集**

数据集与基准：训练使用约 5k 条轨迹，来源于 OpenSeeker‑v1‑Data（Web search）和 SWE‑smith（软件工程）；评估基准为 DeepSearchQA、SWE‑bench Verified 以及 LongMemEval‑S。

**📈 对比分析**

与 Full History、Recency Truncation、Self‑Summary、Prompt‑2B、CoMem 等基线比较：在 DeepSearchQA 上 F1 从 0.52 降至 0.48，输入 token 从 2.83M 缩至 591K，成本仅 22%；在 SWE‑bench Verified 上成功率从 0.63 降至 0.59，输入 1.09M（占 58%），成本 58%；在 LongMemEval‑S 总分 0.47，显著高于 Prompt‑2B 0.43 和 CoMem 0.31。

**⚠️ 局限性**

局限性包括：①训练数据主要聚焦搜索和软件工程，可能不适用于更广泛领域；②压缩器的性能受冻结重建器和 LLM 评判器能力影响；③未在推理阶段使用重建器和评判器，但其对学习信号的依赖可能限制了模型的泛化。

---

## 502. Hardwired Distributed Processing Units for Inference: Neural Decisions at the Sensor that Free the Host

**arXiv ID:** 2611.06436 | [PDF](https://arxiv.org/pdf/2611.06436v1)

**作者:** Raj Mali `[一作]` `[通讯]` (Independent Researcher), Raj Mali (Independent Researcher)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在摄像机传感器侧实现一个无处理器的推理硬件单元（HDPU），能够对斑马线进行实时二分类决策；

**💡 创新点**

将推理任务迁移到传感器侧，构建固定时钟、可确定性的推理HDPU，显著减轻主机负载并在不使用操作系统的情况下完成决策；

**🔧 技术方法**

使用二值化CNN（XNOR/Popcount实现）、FPGA流水线、自动对比度校正、固定时钟调度以及无操作系统的硬件架构；

**📊 数据集**

使用CDNet斑马线数据集中的T1（dashcam子集）和T2（街景子集）共约700张图像进行训练、验证和测试；

**📈 对比分析**

与软件PyTorch模型和全精度ResNet-18进行比较，决策时间34 ms < 37.5 ms帧周期，准确率在T1/T2约79%，主机推理时间比传统方法减少约21%，每帧数据量下降至0.5%；

**⚠️ 局限性**

功耗仅为Vivado估算，未在硬件上测量；帧周期与时钟计时未实时验证；仅在单个摄像头单板上测试，未评估多任务并行或更大规模部署的性能与功耗。

---

## 503. EvoGuard: Mitigating Degradation in Online Skill Evolution

**arXiv ID:** 2611.06444 | [PDF](https://arxiv.org/pdf/2611.06444v1)

**作者:** Yanli Wang `[一作]` (JD.Com), Peng Cao `[通讯]` (JD.Com)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在线技能演化框架，使用基于对比的技能选择器来判断是否接受单轨迹重写后的新技能；

**💡 创新点**

创新点在于将技能演化转化为对比学习任务，先通过离线私有测试集给技能赋予实用度标签，再用拒绝采样构造明显的优劣对，并训练只需一次前向推理即可做决策的对比选择器；

**🔧 技术方法**

技术包括：技能重写循环、私有测试集实用度评估、拒绝采样构造优劣对、对比偏好学习（Bradley‑Terry 换算为两标记对数），以及使用冻结的 LLM 作为选择器后端；

**📊 数据集**

使用的基准数据集有 ALFWorld、ScienceWorld、LiveCodeBench（编程）以及 WebShop；

**📈 对比分析**

与接受所有重写、随机选择、无技能、LLM‑Judge 等基线相比，在 ALFWorld 上平均成功率提升至 56.5%（高于 45.5% 的基线），在 ScienceWorld 上从 10.2% 提升至 15.3%，在 LiveCodeBench 上从 25.3%/27.0% 提升至 28.4%；整体表现显著优于传统方法；

**⚠️ 局限性**

局限性包括：仅在文本/代码基准上验证；需要离线构造并评估技能池；只能在每次任务后与单一新技能对比；对不同模型族的排名可能不同；扩展到多模态或具身系统仍需研究；

---

## 504. Unaligned Image Guided Denoising via Cross-modal Conditional Flow Matching

**arXiv ID:** 2611.06422 | [PDF](https://arxiv.org/pdf/2611.06422v1)

**作者:** Runmin Zhang `[一作]` (Zhejiang University), Hui-Liang Shen `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种用于未对齐 RGB‑NIR 引导去噪的 UGD‑FM 框架，将去噪过程视为条件流匹配，并通过稀疏引导传播实现大范围跨模态匹配。

**💡 创新点**

创新点包括：1) 将未对齐引导去噪建模为跨模态条件流匹配的递进过程；2) 设计阶段聚焦与推理一致的训练策略，实现仅一到两步推理即可达到高质量去噪；3) 引入自适应稀疏引导传播和感知/几何辅助损失，既降低计算量又提高匹配可靠性。

**🔧 技术方法**

使用技术包括：条件流匹配、两分支 U‑Net 结构、稀疏候选传播、窗口/窗口+稀疏注意力、阶段聚焦采样、推理一致状态构造、LPIPS 感知损失、伪光流一致损失。

**📊 数据集**

实验数据集：FANVID、RGB‑NIR Scene、MS2。

**📈 对比分析**

与十种单图像去噪（如 MPRNet、NAFNet、Uformer、Restormer、Xformer）和引导去噪（如 CUNet、DVN、RFFNet、NAID、SANet）方法进行对比，UGD‑FM 在 PSNR/SSIM/LPIPS 指标上均显著领先，尤其在 1‑步和 2‑步推理下仍保持最优或次优性能。

**⚠️ 局限性**

局限性：当跨模态配准误差极大时，稀疏匹配可能仍出现错误匹配；模型对候选数量的设置有一定依赖，极大场景下仍需进一步优化稀疏策略；此外，训练中使用的伪光流标注可能受限于预训练模型的精度。

---

## 505. Ruby-OPD: Enhancing Pronunciation Controllability and Accuracy for Flow-Matching Text-to-Speech via On-Policy Distillation

**arXiv ID:** 2611.06403 | [PDF](https://arxiv.org/pdf/2611.06403v1)

**作者:** Rui Hu `[一作]` (Baidu Inc.), Zhenpeng Zhan `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一个两阶段的 flow‑matching 日语 TTS 框架，结合 Ruby 注释与正字法扰动实现细粒度发音控制，并通过 On‑Policy Distillation 在大规模无配对文本上提升原始文本发音准确率。

**💡 创新点**

创新点在于使用 Ruby 注释与正字法扰动来保持表面汉字与发音信息的双重语义，以及引入 On‑Policy Distillation 在无语音对齐数据下转移发音知识。

**🔧 技术方法**

技术包括 Ruby 表示（隐式与显式）、正字法扰动（遮盖、随机替换、语义替换）、交叉条件 On‑Policy Distillation、流匹配（Conditional Flow Matching）以及 DiT 与 DACVAE 声学模型。

**📊 数据集**

数据集主要使用 774 小时的 Emilia 日语子集（经 G2P 校验）进行 SFT，以及约 680k 单音节和 100k 多音节句子通过 Gemini 3.1 Pro 合成的文本‑发音对进行 OPD。

**📈 对比分析**

在 Joyo‑Kanji‑Yomi 基准上，OPD 版模型在无注释原始文本上达 97.13% 的发音准确率，Target PER 2.02%，Sentence PER 0.70%，显著优于基线与公开 SOTA。

**⚠️ 局限性**

局限性包括依赖大规模 LLM 合成文本的质量、对非标准化文本的鲁棒性待验证，以及对复杂多音节词的处理仍可能存在边缘错误。

---

## 506. KolTa: A Physics-Informed Active Learning Framework for Formulation Discovery with Limited Data

**arXiv ID:** 2611.06394 | [PDF](https://arxiv.org/pdf/2611.06394v1)

**作者:** Boyong Tang `[一作]` (Sichuan University), Lei Yang `[通讯]` (Sichuan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种物理信息化主动学习框架，利用三阶段预测器和物理残差校准的UCB采集规则，在有限实验预算下寻找高性能配方。

**💡 创新点**

创新点：①将基准贡献、角色条件交互和锚定温度响应分离的三阶段预测器，并通过软物理约束引导温度响应；②在每轮使用OOF误差拟合非负自适应探索权重，结合集成不一致度与目标温度物理残差的PRC-UCB采集。

**🔧 技术方法**

使用技术包括GINE图神经网络编码、三阶段网络结构、物理残差训练、集成学习、偏向校准的上界置信度采集及Pinball损失。

**📊 数据集**

使用数据集包括NIST二元混合粘度、Drug药物溶解度、ILThermo离子液体粘度以及DiffMix模拟电解液电导率四个任务。

**📈 对比分析**

与MolSets、SolvGNN、GP、XGBoost等九种策略比较，使用AULC、q30、q80等指标。该框架在四个任务中均排名前两，平均AULC最高，q80比最快对手低约18%，总体获胜率达75/78。

**⚠️ 局限性**

限制：仅在五个固定温度施加软物理约束，未评估超温度外推；模型仅处理温度变化，未扩展至压力等多条件；物理残差对训练的影响有限。

---

## 507. Within-Family EfficientNet Scaling under Fixed-Resolution Transfer Learning for Spectral-Domain Optical Coherence Tomography Retinal Disease Classification

**arXiv ID:** 2611.06452 | [PDF](https://arxiv.org/pdf/2611.06452v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 508. Signed but Steerable: Measuring and Closing the Text-Vector Integrity Gap in Offline Retrieval-Augmented Generation

**arXiv ID:** 2611.06498 | [PDF](https://arxiv.org/pdf/2611.06498v1)

**作者:** Georgios Feretzakis `[一作]` (securagen.ai), Alexandros Papaspyridis `[通讯]` (securagen.ai)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究离线检索增强生成（RAG）系统中知识包的向量完整性问题，提出并评估多种验证策略并量化攻击影响。

**💡 创新点**

引入“边际定律”（Margin Law）来预测容忍误差下可被攻击者劫持的查询比例；展示压缩向量存储在容忍检查下易受攻击，并提出基于代码层的验证与规范化（canonicalisation）方案实现无残余风险。

**🔧 技术方法**

利用余弦相似度检索、预计算嵌入、动态 INT8 与 FP32 推理、FAISS 索引、ONNX Runtime、llama.cpp、Product Quantisation（PQ）、Scalar Quantisation（SQ8/SQ4）、二进制码等技术；对不同硬件（Raspberry Pi 5、桌面GPU、Apple M2）进行跨平台可重复性测试。

**📊 数据集**

使用 SQuAD 2.0、四个 BEIR 语料库（SciFact、NFCorpus、ArguAna、FiQA）、MS MARCO、1 000 条查询等公开数据集。

**📈 对比分析**

通过预注册实验与基准，对比不同验证策略的攻击曝光率、向量重算成本和模型推理影响；在真实设备上测得：容忍 FP32 噪声下最多 5.5% 的查询受害，INT8 容忍阈值下 98–100%；代码层验证可将曝光率降至 0%。

**⚠️ 局限性**

局限性包括：仅评估两种嵌入模型、平面 FAISS 索引、单一硬件架构；未考虑动态量化批次差异、图索引或多语言语料；所需的阈值与 guard width 需针对具体系统测量，缺乏通用阈值标准。

---

## 509. EviSeek: An Evidence-Centric Agentic Framework for Embodied Active Information Seeking

**arXiv ID:** 2611.06434 | [PDF](https://arxiv.org/pdf/2611.06434v1)

**作者:** Jintao Li `[一作]` (Chinese Academy of Sciences), Börje F. Karlsson `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个无训练的框架EviSeek，利用问题驱动的证据需求来指导视点变化与物理交互。

**💡 创新点**

创新点在于显式化证据需求并结合证据板进行选择性保留，整合行动规划与记忆。

**🔧 技术方法**

使用预训练的视觉语言模型进行推理，结合Prompting实现需求规划、交互与答案生成。

**📊 数据集**

在ESI‑Bench和EXPRESS‑Bench两个模拟环境中评估。

**📈 对比分析**

与Single‑model基线比较，EviSeek在ESI‑Bench上提升7.1/3.7%准确率，减少Token使用；在EXPRESS‑Bench上在视觉支持和路径权重上优于RGB单模型。

**⚠️ 局限性**

局限在于缺乏空间建模、对物理噪声与真实机器人未验证、以及在某些子任务如Metric Comparison表现不佳。

---

## 510. Fourier Entropy Bounds via Approximations

**arXiv ID:** 2611.06451 | [PDF](https://arxiv.org/pdf/2611.06451v1)

**作者:** Manaswi Paraashar `[一作]` (Indian Institute of Technology Hyderabad), Nitin Saurabh `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文建立了布尔函数傅里叶分布香农熵的上界，并将其与多种复杂度度量（如随机化平行查询复杂度、单侧无歧义证书复杂度、平均锥状特征函数阶数）关联；

**💡 创新点**

创新点在于将基于逼近的熵估计从最小熵或Rényi熵推广到完整香农熵，并允许一侧或重叠的精确表示；

**🔧 技术方法**

主要技术包括：逼近多项式的谱范数序列构造、熵分组与Jensen不等式、Riesz–Thorin插值得到的矩母不等式，以及量子通信复杂度的奇异值分布熵下界；

**📊 数据集**

本工作为理论研究，不使用实验数据集；

**📈 对比分析**

与以往的平均决策树深度、随机化查询复杂度等基准相比，本文给出了更紧的上界（如\(\mathcal H(f^2)=O(^\oplus(f))\)并且在量子通信中得到维度无关的下界；

**⚠️ 局限性**

局限在于仅给出上界，无法说明是否可达到；此外，逼近多项式的构造可能并非最优，导致熵估计存在余量。

---

## 511. The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention

**arXiv ID:** 2611.06476 | [PDF](https://arxiv.org/pdf/2611.06476v1)

**作者:** Ye Yuan `[一作]` `[通讯]`, Ye Yuan

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对在Qwen3.5‑4B模型上使用稀疏LoRA与长度归一化DPO（含SFT保留项）进行微调，跟踪隐式奖励边际增大至≈7时下游能力的变化，进行系统的行为与概率层面实证评估。

**💡 创新点**

创新点在于将预注册的TOST等价性检验与自适应评分结合，用以量化对齐税的任务选择性与分级损伤，并揭示边际1–3窗口内的瞬时格式破损与内容错误双轨迹。

**🔧 技术方法**

采用长度归一化DPO与RPO（SFT保留项）目标，稀疏LoRA微调，nf4量化训练与fp16测量，配合已冻结的四层分类器v2进行artifact‑aware评分。

**📊 数据集**

使用647条人工生成的优劣对偏好样本，以及三套160条SEC‑文件结构化评测集（JSON工具调用、路由标签、提取JSON），所有数据均由GPT‑4o‑mini生成。

**📈 对比分析**

通过将最终检查点与SFT基线在语义准确率上做TOST等价性（±5个百分点）比较，提取任务保持（Δ≈‑0.55），JSON轻度下降（Δ≈‑3.5），路由任务剧烈衰退（Δ≈‑30），同时奖励边际达到≈7。

**⚠️ 局限性**

局限性包括仅在4B Qwen3.5 上使用稀疏 LoRA（0.034% 参数）与固定层数，使用合成偏好与结构化任务，缺少无保留项对照、自然人类偏好与安全任务，结果可能不适用于更大模型或不同偏好域。

---

## 512. DreamRealm: Exploring and Materializing Hallucination Worlds for Trustworthy Vision-Language Models

**arXiv ID:** 2611.06433 | [PDF](https://arxiv.org/pdf/2611.06433v1)

**作者:** Benlei Cui `[一作]` (Yuvion Team, Alibaba Group), Haiwen Hong `[通讯]` (Yuvion Team, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DreamRealm框架，通过挖掘并可视化模型的幻觉，构造相应的图像编辑以匹配幻觉回答，从而提升多模态大语言模型的视觉感知与幻觉抑制。

**💡 创新点**

创新点在于：①利用模型自身高置信度幻觉进行数据构造，确保监督针对真实感知混淆；②引入“幻觉分配”概念，让模型学习将每个幻觉回答对应到其支持图像；③结合双向对比学习与有监督微调，形成更强的图像-回答对应关系。

**🔧 技术方法**

主要技术包括：高温采样与属性枚举的幻觉挖掘；基于GPT-Image和图像编辑器的最小编辑生成支持图像；双向对比损失（行对比与列对比）与SFT相结合的RealmMatch训练目标；数据验证与人类审核流程。

**📊 数据集**

使用了自构建的RealmSet（28,960例），来源于VC-STaR的VisCoR-55K，通过RealmForgePipe生成；还评估了HallusionBench、MMVP、AMBER、MMHal-Bench、CV-Bench-2D、ObjectHal等六大视觉感知与幻觉基准。

**📈 对比分析**

与多种专门的幻觉抑制方法（如VCD、CoCI-CAP、RLAIF-V等）以及原始模型进行对比，DreamRealm在Qwen3.5-9B上实现了HallusionBench提升7.26点、MMVP Pair提升11.34点，其他指标亦普遍超越。对其他模型（Qwen2.5-VL-7B、Gemma-4-E4B-it、InternVL3.5-8B）使用相同RealmSet训练也有提升，但幅度低于匹配模型。

**⚠️ 局限性**

主要限制包括：数据构造成本高（需图像生成与人工审核）；依赖底层图像编辑模型的能力，部分场景难以成功编辑；构造的幻觉空间与目标模型不完全匹配，跨模型迁移效果有限。

---

## 513. D3D: Decoupled Diffusion Drafting for Asynchronous Speculative Decoding

**arXiv ID:** 2611.06425 | [PDF](https://arxiv.org/pdf/2611.06425v1)

**作者:** Baoyou Chen `[一作]` (Shanghai Innovation Institute), Siyu Zhu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 D3D，一种独立的扩散式草稿器，支持异步无损推测解码

**💡 创新点**

创新点在于：1）采用目标对齐训练，将块级去噪与自回归监督结合，实现不依赖目标隐藏状态的单前向块预测；2）基于接受长度的逐层剪枝和恢复训练，显著降低草稿延迟；3）通过一次打包前向传递预生成后续块，实现与目标验证并行的异步推测解码

**🔧 技术方法**

使用了扩散模型（blockwise denoising）、自回归监督、指数衰减位置加权、知识蒸馏恢复训练、层级剪枝、缓存匹配的异步推测策略

**📊 数据集**

在七个基准（GSM8K、MATH-500、AIME25、HumanEval、MBPP、LiveCodeBench-v6、MT-Bench）上，用 Qwen3-8B 与 Qwen3-32B 两大目标模型生成响应进行训练与评估

**📈 对比分析**

与 DFlash、DSpark、EAGLE-3、PARD-2、SSD 等基线比较；D3D 在 Qwen3-8B 上平均提升 5.00×，在 Qwen3-32B 上平均提升 5.71×，并在所有基准上获得最长的接受长度（τ≈7.9）

**⚠️ 局限性**

局限性包括：1）实验仅在两块 GPU（各占一块）上实现，未评估单块 GPU 通过张量并行的吞吐；2）缓存未命中时需同步草稿，可能导致延迟；3）实现基于 Transformers；在 SGLang、vLLM 等生产框架中的性能尚待验证

---

## 514. Entropogram-guided Visual Realignment Decoding: Capturing the Rhythm of Multimodal Reasoning

**arXiv ID:** 2611.06415 | [PDF](https://arxiv.org/pdf/2611.06415v1)

**作者:** Jiaqi Deng `[一作]` (University Of Technology Sydney), Guandong Xu `[通讯]` (University Of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于熵动态监测的自我反省框架 SEED，在多模态推理过程中实时跟踪熵波动，并在检测到低熵持续或熵尖峰时，通过视觉注意力重新对齐来减少幻觉。

**💡 创新点**

创新点在于将熵视为跨模态交互的结构化信号，识别出两种导致幻觉的动态模式，并在推理过程中无需额外训练即可通过熵触发的视觉重对齐实现在线干预。

**🔧 技术方法**

采用熵监测与阈值触发、视觉特征提取与参考注意力平均、在LLM前向层注入视觉重对齐向量等技术，并与 Qwen3‑VL 与 Llama‑3.2 Vision 等模型结合。

**📊 数据集**

使用了六个多模态推理基准（VStar、MMVP、VisOnly、M³CoT）和两个幻觉基准（POPE、HallusionBench），共九个评估设置。

**📈 对比分析**

与 VCD、MemVR、OPERA 等无训练的幻觉抑制方法以及原始模型对比，平均提升 3.2–3.7% 的推理准确率，幻觉基准提升 2.5–3.6%，且计算开销低于 5%，在 21/27 个模型–基准组合上取得最佳表现。

**⚠️ 局限性**

局限在于对某些模型（如 Llama‑3.2 及 POPE 简短答案）效果不显著甚至略降；需要手动设定阈值，且对特殊输出格式可能产生副作用；仅关注熵信号，未对视觉特征选择进行自适应优化。

---

## 515. Optimal Subcubic Byzantine Agreement with Plain PKI

**arXiv ID:** 2611.06479 | [PDF](https://arxiv.org/pdf/2611.06479v1)

**作者:** Marc Dufay `[一作]` (ETH Zürich), Roger Wattenhofer `[通讯]` (ETH Zürich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计了一种在同步网络中，使用普通签名而非阈值签名，实现容忍 t < n/2 的确定性拜占庭一致性协议，其总消息复杂度为 O(κ^1.303 n^2.303)，并保持线性轮次。

**💡 创新点**

核心创新包括：① 在无阈值签名情况下构造可证明的投票证明；② 利用可容错的指数图和分散器实现高效信息传播；③ 引入公平蛋糕切割技术生成无冲突的指控匹配；④ 采用动态规划与批处理技术降低递归调用次数，最终实现子三次幂复杂度。

**🔧 技术方法**

主要技术手段包括：拜占庭一致性框架（Momose‑Ren 方案），可容错扩展图、分散器、Merkle 累加器、可纠错码、裁决传播、批量多值广播、并行广播以及递归合并策略。整体实现高度并行且复杂度分析基于递归不等式和数值优化。

**📊 数据集**

由于论文属于理论算法研究，没有使用实际数据集；所有评估均通过复杂度分析与已知下界进行比较。

**📈 对比分析**

与传统使用阈值签名的最优二次复杂度方案（O(κ n²)）相比，本文不依赖阈值签名即可实现更低的 O(κ^1.303 n^2.303) 复杂度；与之前的三次幂协议相比，复杂度下降了约 0.303 次幂，但仍高于 Dolev‑Reischuk 的二次下界，显示出明显的性能提升。

**⚠️ 局限性**

局限性包括：① 复杂度指数为 (√13+1)/2 ≈ 2.303，仍与理论下界相差约 0.303 次幂；② 方案实现复杂，依赖多种高级图结构与分散技术；③ 仅适用于同步模型；④ 对抗量子安全性未进行充分讨论；⑤ 需要先验的公钥基础设施并假设签名不可伪造。

---

## 516. CoDecide: Content-Adaptive Video Decoding for High-Throughput Video Analytics

**arXiv ID:** 2611.06455 | [PDF](https://arxiv.org/pdf/2611.06455v1)

**作者:** Shristy Shah `[一作]` (Aalborg University), Kamal Nasrollahi `[通讯]` (Aalborg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种仅在新内容出现时解码视频帧的选择性解码系统CoDecide。

**💡 创新点**

使用无标签的DINOv2教师定义语义新颖性，在压缩域学生上进行蒸馏，基于运动向量、残差计数和宏块类型预测何时以及哪些帧/宏块需要解码；该方法可跨摄像机迁移，无需重新训练。

**🔧 技术方法**

压缩域特征抽取（MV、NZC、宏块类型）、教师-学生蒸馏、EMA+ConvGRU记忆、运动向量传播、选择性宏块重建；实验采用YOLO11x检测器。

**📊 数据集**

VIRAT、MEVA、ShanghaiTech 三个监控数据集，采用 leave‑one‑dataset‑out 评估。

**📈 对比分析**

与 Keyframes、CoVA、PacketGame、ParaDeco 等基线在同一检测器和成本计算下对比；在移动物体召回率上 CoDecide 与或优于 CoVA，且在相同或更低的解码预算下实现 15–24× 的吞吐量提升。

**⚠️ 局限性**

误差主要来自运动向量传播的漂移，导致长 GOP 中检测框偏移；对极小物体的检测仍受限；系统对宏块大小和 GOP 长度敏感。

---

## 517. Beyond Infraction Scores: Failure-Boundary Profiles for Closed-Loop Evaluation of Driving and Robot Navigation Agents

**arXiv ID:** 2611.06453 | [PDF](https://arxiv.org/pdf/2611.06453v1)

**作者:** Milan Bulat `[一作]` (Social Chaos Lab), Ellington Kirby `[通讯]` (valeo.ai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对CARLA和Isaac Sim中多个驾驶与机器人导航代理，提出通过沿单一已认证因素（如通道宽度、行车间隔）扫掠场景家族，估计承诺率和能力率，构造失败边界曲线并定义决策准确率；

**💡 创新点**

创新点在于：1）将驾驶性能拆分为尝试与成功两个维度，避免传统驾驶分数混淆；2）使用已认证的场景家族生成与标注，提供可重复的决策标签；3）引入决策准确率作为综合指标；4）在同一测量轴上可重新评估不同版本代理。

**🔧 技术方法**

采用CARLA与NVIDIA Isaac Sim仿真，基于ScenarioRunner、OpenSCENARIO及自研的SCL（Scenario Creation Pipeline）生成参数化场景；利用逻辑回归拟合承诺与能力曲线，Bootstrap区间估计不确定性；计算驾驶分数与决策准确率。

**📊 数据集**

数据集为十个手工设计的场景家族（七个CARLA驾驶族、三个位姿仓库族），每族包含12~14个幅度变体，每变体在三条交通种子下运行。所有实验数据和代码均托管于GitHub与Hugging Face Hub。

**📈 对比分析**

与专家（规则驱动）及六款学习型驾驶代理（TFv6、BridgeDrive、TF++、PlanT 2.0、SimLingo、HiP-AD）以及仓库机器人X‑Mobility对比；结果显示专家在决策准确率上最高（0.74），TFv6为最佳学习型代理（0.60）；学习型代理在阻塞通道时往往不停止，导致驾驶分数与决策准确率差距；对同一场景家族可随版本更新重新测量，观察边界移动。

**⚠️ 局限性**

局限性包括：1）全部基于仿真，无法直接推广到真实车辆或机器人；2）决策标签依赖于先验阈值（策略底盘、间隙裕度），不同阈值会影响排名；3）样本量有限（每变体3条种子），导致边界区间重叠；4）评估成本高，约220 GPU小时；5）仓库实验仅覆盖单一机器人与单一策略。

---

## 518. Graph Foundation Models Can Become Effective Multivariate Time Series Anomaly Detectors

**arXiv ID:** 2611.06441 | [PDF](https://arxiv.org/pdf/2611.06441v1)

**作者:** Zhuangyan Hu `[一作]` (Harbin Engineering University), Sumi Helal `[通讯]` (University of Bologna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Gfm4Mtsad，利用预训练的图谱基础模型（Graph Foundation Models, GFMs）进行无监督多变量时间序列异常检测。

**💡 创新点**

创新点包括：① Structure‑and‑Information Contract Alignment (SICA) 对时序特征、拓扑结构与信息流进行契约对齐；② Mechanism‑Aligned Pseudo‑Anomaly Screening (MAPS) 通过局部机制破坏与硬负样本筛选提供可靠伪异常监督；③ Tail‑Gated Evidence Calibration (TGEC) 在窗口级决策时对多粒度证据进行尾部校准，保持正常评分结构，提升异常召回。

**🔧 技术方法**

技术核心：预训练图神经网络（AnomalyGFM、OwlEye）冻结使用；Token Bridge 与正交残差回传；基于正常先验的图结构构建；图‑VAR 机制干预与量化筛选；多尺度异常信号聚合与极值理论校准。

**📊 数据集**

使用三大工业/航空时间序列数据集：MSL P‑15、SMAP T‑3、SMD 1‑1，覆盖高维、稀疏及密集场景。

**📈 对比分析**

与 9 种基线（SimCLR、CSI、SSD、AnomalyTransformer、DCdetector、CTAD、CARLA、CAROTS、PaAno）比较，Gfm4Mtsad 在 AUROC、AUPRC、F1 方面多数指标达到或接近最佳，显著优于传统方法。

**⚠️ 局限性**

局限性：依赖预训练 GFM 的结构与容量，若基模型不匹配或维度过高可能影响性能；对极端非平稳或分布漂移场景的在线自适应尚未验证；需要额外的超参调优（如尾部阈值、残差瓶颈大小）。

---

## 519. Twists and Turns: A Graph-Based Benchmark and Framework for Long-Term Memory in Exploration Tasks

**arXiv ID:** 2611.06414 | [PDF](https://arxiv.org/pdf/2611.06414v1)

**作者:** Chongyuan Dai `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiuying Chen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于图的探索记忆基准 LExMem 与对应的记忆框架 LExGraph，利用演化图结构生成长周期探索对话并评估模型的记忆能力。

**💡 创新点**

创新点在于通过演化图模型模拟非单调的探索过程，生成多样化关联性问题，并构建增量式记忆图捕捉探索步间的层次与跨分支关系。

**🔧 技术方法**

技术方法包括使用大语言模型生成图结构与对话、图神经网络/结构检索、混合检索与邻居扩展、基于右前沿约束的层次边构造等。

**📊 数据集**

数据集主要为自研的 LExMem，包含 224 条科研与软件设计领域对话、2018 个 QA 对，外部还在 BEAM 上进行验证。

**📈 对比分析**

与多种检索与记忆基线（如 GraphRAG、LightMem 等）对比，LExGraph 在 LExMem 上的 Token‑F1 与 ACC 均居首位，尤其在信息聚合类问题上提升显著，同时保持较低的 token 用量与推理延迟。

**⚠️ 局限性**

局限性在于对话为合成语料，缺乏真实多样性；评估仅针对问答任务，未直接验证记忆提升是否能改进探索决策。

---

## 520. Zero-Shot Coordination among LLM Agents

**arXiv ID:** 2611.06409 | [PDF](https://arxiv.org/pdf/2611.06409v1)

**作者:** Adrian Hayler `[一作]`, Jakob Foerster `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM在零射协同（ZSC）环境下的表现，系统评估文本和代码代理在简化合作环境中的协调能力，并提出并自动学习“协调指令文档（cfd）”提升协调性能。

**💡 创新点**

首次系统性评估LLM在ZSC下的失败原因，揭示语义偏好与理论心智不足，并提出可自动学习的cfd机制。

**🔧 技术方法**

采用语言模型提示、in-context learning、基于Dec‑POMDP框架的实验、进化式Meta学习生成cfd，并与自我对弈（sp）、固定与可变LLM交叉对弈（xp_seed、xp_llm）比较。

**📊 数据集**

在九个简化协同任务（Cat & Dog、Tiny Hanabi、杠杆游戏等）及其变体，收集文本描述并构造Dec‑POMDP描述。

**📈 对比分析**

与现有RL算法和手工cfd对比，发现LLM在所有环境中表现差，cfd显著提升XP分数，自动学习的cfd比手工更短且在多模型上泛化。

**⚠️ 局限性**

仍存在cfd过于简单导致欠协同、语义竞争导致局部最优、LLM理论心智不足、实验仅覆盖简化任务、以及可变LLM评估受限。

---

## 521. AtomicFaaS: Offloading Side-Effect Compensation to the Serverless Platform

**arXiv ID:** 2611.06601 | [PDF](https://arxiv.org/pdf/2611.06601v1)

**作者:** Jonas Heisterberg `[一作]` (Technical University Berlin), David Bermbach `[通讯]` (Technical University Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个平台级侧效补偿系统 AtomicFaaS，允许服务器无关平台在函数执行失败时自动补偿外部副作用，开发者无需手写补偿代码。

**💡 创新点**

创新点在于将补偿责任从函数迁移到平台，通过拦截并记录外部调用，再利用逆操作表在失败后由平台执行补偿，从而实现语义原子性。

**🔧 技术方法**

技术实现包括在 Knative 上通过 Fetch API Shim 拦截 HTTP 调用并写入 Redis 日志，崩溃检测器监测实例失效，反转器 (reverter) 按逆操作表自动补偿；全部使用 Go 实现。

**📊 数据集**

实验使用 SeBS‑Flow 的旅行预订工作负载（trip‑booking），并在扩展版本中加入支付与邮件发送等外部服务调用。

**📈 对比分析**

与手写 Saga 对比，评估了开发者工作量（代码行数显著减少）、正确性（所有失败场景无残留副作用）、性能（单发调用延迟增加约 1 ms/拦截，CPU 受限时吞吐量略低；外部调用主导时开销可忽略）。

**⚠️ 局限性**

局限性包括只能补偿具有逆操作的调用；无法处理不可逆或 catalog 未覆盖的副作用；拦截仅覆盖 HTTP 调用，其他传输方式需扩展；实例在调用中途崩溃时可能无法确定效果；平台组件需进一步扩展和资源评估。

---

## 522. A trust layer for fast pedestal-stability surrogates: calibrated error bars, out-of-distribution alarms and few-shot recalibration

**arXiv ID:** 2611.06597 | [PDF](https://arxiv.org/pdf/2611.06597v1)

**作者:** Amin Vakhshouri `[一作]` `[通讯]` (dataquanta LLC), Amin Vakhshouri (dataquanta LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个“信任层”用于包裹任何预测稳态边缘稳定性的机器学习替代模型，使其能输出可靠的置信区间、检测训练外输入并支持少量重新校准；

**💡 创新点**

创新点在于将分割共形预测、分段校准、协变量加权、OOD报警以及少量样本再校准等技术组合成通用、可迁移的信任层；

**🔧 技术方法**

使用分割共形预测、Mondrian 分段校准、加权共形预测、深度集成的置信扩散与马氏距离 OOD 报警，以及少量样本重新校准；

**📊 数据集**

主要使用公开的 KARHU（理想 MHD）数据集和 BOUT++（阻力 MHD）数据集，并在 SPARC 与 DIII‑D 机器的实际等离子体上进行测试；

**📈 对比分析**

与传统无误差区间或贝叶斯不确定性方法相比，信任层在原始数据上达 90% 覆盖率，且在形状偏移时保持 85–92% 覆盖；通过 OOD 报警将约 22% 的偏移样本路由到物理代码，保证最终决策正确率 92.5%；在 BOUT++ 场景扫描中，模型在几秒内完成决策，而传统模拟需数小时；

**⚠️ 局限性**

局限性包括：共形预测保证是边际的且依赖于校准数据与目标数据的可交换性；在极端训练外场景下 OOD 报警可能过度或不足；少量样本再校准对不同设备的通用性仍需更多实验验证；目前未对实验观测结果进行验证，仅基于数值模拟。

---

## 523. FOSCo-Net: Ego-centric Voxel Completion for Forest LiDAR Occlusions

**arXiv ID:** 2611.06588 | [PDF](https://arxiv.org/pdf/2611.06588v1)

**作者:** Iman Esfandiyar `[一作]` (Poznań University of Technology), Dominik Belter `[通讯]` (Poznań University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了FOSCo-Net，一种自监督的ego-centric体素完成框架，用单帧森林LiDAR扫描预测被树干遮挡的空间，实现对盲区的几何补全。

**💡 创新点**

创新点包括：①首次针对森林LiDAR遮挡提出自监督、无标注、仅使用单帧输入的补全方法；②设计了σ门控skip解码器，使得未观测区域的预测能够自行衰减；③提出基于遮挡区域的遍历安全指标SFPR，用于评估补全对路径规划的直接影响。

**🔧 技术方法**

技术方案包括稀疏子卷积编码器、四种解码器变体（隐式、完整稀疏、密集、门控skip）、圆柱坐标与笛卡尔坐标ego-centric网格、焦点交叉熵损失与类加权、数据增强以及基于时间窗口的自监督目标生成。

**📊 数据集**

使用DigiForests森林LiDAR数据集，提取50个ego-centric样本（40训练，10测试）仅包含地面和树干标签，采用单帧扫描与时间窗口累积扫描构建自监督标签。

**📈 对比分析**

与两种密集输入基线（MIDUNet2D、LMSCNet）在IoU、F1、stem recall、全局精度以及新指标SFPR上进行对比；FOSCo-Net门控skip在SFPR（0.346）显著低于MIDUNet2D（0.741），在IoU、F1、精度等指标上也优于两基线，展示了更安全、更准确的遮挡补全效果。

**⚠️ 局限性**

局限性包括：仅在单一森林序列和单一森林类型上验证；只处理静态环境，未考虑动态物体；模型对高分辨率网格依赖较大，计算量和存储需求相对较高；自监督标签仍受累积扫描质量限制。

---

## 524. One Scale, Four Texts: Gaps, Overlaps and Silent Repairs in Published Grading Schemes

**arXiv ID:** 2611.06569 | [PDF](https://arxiv.org/pdf/2611.06569v1)

**作者:** Elisa Rovai `[一作]` (University of Padua), Fabio Rovai `[通讯]` (Tesseract Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对糖尿病视网膜病变分级方案的四份文本（英语NHS、ICO 2017、ICO 2003、意大利指南）进行规则化转录，枚举所有可接受的解释（reading），通过SMT求解器和OWL推理器识别文本自带的缺口、重叠和顺序逆转等缺陷，并在三组公开图像数据集上验证这些缺陷的实际出现频率，最后用预注册实验比较不同文本产生的标签对应的机器学习分类器性能。

**💡 创新点**

创新点在于：① 将文本文档的模糊性系统性转化为可枚举的“视角”并进行超真值分析；② 用自动化工具（SMT、OWL）全量枚举并验证缺陷；③ 在真实图像数据上量化缺陷对标签一致性的影响；④ 通过预注册实验展示不同文本导致的分类器差异与文本自身不确定性的关联。

**🔧 技术方法**

技术包括：规则化文本转写、参数化阅读生成、超真值（supervaluation）判定、SMT（Z3）枚举、OWL 2 推理（HermiT）、U-Net 位置与分割检测、交叉验证机器学习（多分类器）、统计检验（McNemar、Wilson 区间）和对齐模块（BFO/IAO/OWL/Vocab）。

**📊 数据集**

使用的公开图像数据集有：IDRiD（516张含分割和定位）、DDR（含微血管瘤、出血、硬性渗出标注）以及 DIARETDB1（专家共识标注）。

**📈 对比分析**

比较方法是：在所有三种文本都能决定的图像上，用每个文本的分级标签训练四分类器，然后交叉验证计算每个模型对三种标签集的准确率（如表所示），并用McNemar检验评估模型与标签集之间的一致性。性能显示：ICO 2017 训练模型对ICO标签的准确率最高（约92%），对意大利标签也高（≈92%），对英语标签相对较低（≈80%）；意大利模型对意大利标签准确率最高（≈92%），对英语略低（≈82%）。

**⚠️ 局限性**

局限性包括：① 对“视网膜增厚”这一非图像可观测变量的假设可能导致缺陷频率估计不稳；② 数据集注释不完整或存在标注冲突，影响缺陷实际出现比例；③ 仅考虑图像可见变量时，部分文本的决策依赖非图像信息，导致模型无法完整复现文本逻辑；④ 结果受文本翻译、转写准确性和阅读枚举范围的影响；⑤ 实验仅评估了部分标签（复诊、回访、转诊、急诊）而未覆盖所有临床决策路径。

---

## 525. Trace2Eval: Contextualized Evaluation of Coding Agents from Real-World Traces

**arXiv ID:** 2611.06506 | [PDF](https://arxiv.org/pdf/2611.06506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 526. Deliberate Latent Reasoning for Sequential Recommendation

**arXiv ID:** 2611.06521 | [PDF](https://arxiv.org/pdf/2611.06521v1)

**作者:** Haobo Zhang `[一作]` (Gaoling School of Artificial Intelligence), Zhicheng Dou `[通讯]` (Gaoling School of Artificial Intelligence)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 DESRec，一种基于 Transformer 的有意识潜在推理框架，用于序列推荐，在训练阶段动态确定每个用户的最佳推理步，并通过自蒸馏将最佳推理信息压缩到最终状态，同时通过滑动窗口里程碑抽取捕捉用户兴趣的演变。

**💡 创新点**

创新点包括：① 通过 Reasoning Step Selection (RESS) 在训练时为每个用户挑选最优推理深度，并利用自蒸馏将该深度的表示迁移到最终推理步；② 通过 Progressive Preference Extraction (PPEM) 将用户历史划分为若干时期并提取里程碑，在推理过程中逐步整合这些里程碑，从而实现个性化、动态的推理流程。

**🔧 技术方法**

采用 Transformer（如 SASRec、BERT4Rec）作为编码器，进行自回归潜在推理，利用 Kullback–Leibler 损失进行自蒸馏；使用滑动窗口对历史序列进行分段提取里程碑；推理阶段借助 KV 缓存实现高效推理。

**📊 数据集**

实验使用了六个公开数据集：Yelp、Amazon（Video & Games、Software、CDs & Vinyl、Baby & Products）以及 MovieLens‑1M，涵盖 ID‑based 与 text‑based 推荐场景。

**📈 对比分析**

在四个 backbone（SASRec、BERT4Rec、UniSRec、MoRec）与八个基线（ReaRec‑ERL/PRL、Recformer、LRURec、COT‑Rec 等）上进行 Recall@10/20、NDCG@10/20 比较，DESRec 在所有数据集和 backbone 上均优于基线，平均提升 5%–14% 以上，表现最优的实例可达 18%+。

**⚠️ 局限性**

局限性包括：① 推理时仍需使用固定推理步长，无法完全适应不同用户的实时深度；② 对窗口大小、推理步数等超参数敏感，需在各数据集上调优；③ 仅在离线公开数据集验证，缺乏真实在线服务的实时评估；④ 未针对公平性、偏见等问题进行分析和改进。

---

## 527. Energy-Accumulate-Then-Act Backscatter Communications With Retransmission Under Imperfect Feedback Detection

**arXiv ID:** 2611.06499 | [PDF](https://arxiv.org/pdf/2611.06499v1)

**作者:** Rui Xu `[一作]` (Xi'an University of Posts and Telecommunications), Gan Zheng `[通讯]` (University of Warwick)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种阈值触发的跨块能量累积-行动（EA-then-act）协议，在后向散射通信（BackCom）系统中引入重传机制，并考虑了不完美的反馈检测。

**💡 创新点**

创新点包括：①将能量累积跨块与重传结合，显著提升能量利用率和可靠性；②利用嵌入式马尔可夫链建模能量与数据包缓冲状态的耦合演化；③在不依赖精准CSI的前提下，分析平均信息新鲜度（AAoI）与包丢失率（PLR）的闭式表达式，并揭示重传在不同能量、速率和反馈功率下对AAoI的两面性。

**🔧 技术方法**

采用的技术包括：基于Rayleigh衰落的信道模型；能量量化与离散化；非相干能量检测的误检概率模型；嵌入式马尔可夫链与状态转移矩阵求解；联合时刻矩推导与交叉矩分析；数值仿真验证。

**📊 数据集**

使用的“数据集”是基于随机Rayleigh衰落和路径损耗的仿真场景（如S→BD、BD→R、R→BD距离、功率、噪声等参数），无真实实验数据。

**📈 对比分析**

比较方法：与三类基准方案对比——（a）无能量累积且无重传的传统按块能量管理；（b）跨块能量累积但无重传；（c）跨块能量累积并假设完美反馈的理想方案。性能表现：在能量受限条件下，EA-then-act显著降低AAoI；重传在高能量/高速率下可进一步降低AAoI，同时将PLR压至接近零；相较于基准方案，所提协议在不需要精确CSI且反馈不完美的实际环境下实现更优的AAoI/PLR折中。

**⚠️ 局限性**

限制：①模型假设单用户、全反射模式，未考虑多用户或非反射调制；②反馈误检采用简化的非相干能量检测模型，真实硬件误差可能更复杂；③马尔可夫链分析基于能量量化和泊松抽样的近似，极端参数下误差可能增大；④未给出硬件实现细节，实际能量转换效率和电路功耗可能与理论不同。

---

## 528. Verifier-Guided Self-Consistency: Test-Time Reasoning with Imperfect Generators and Verifiers

**arXiv ID:** 2611.06606 | [PDF](https://arxiv.org/pdf/2611.06606v1)

**作者:** Ekaterina Fadeeva `[一作]` (ETH Zürich), Maxim Panov `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在推理时利用可学习验证器的自一致性解码方法——Verifier‑Guided Self‑Consistency (VGSC)；

**💡 创新点**

将步骤级正确性评估和未来结果预测融合进随机重采样解码，兼顾生成多样性与验证器引导；

**🔧 技术方法**

采用基于大型语言模型的生成器、步骤级验证器 V、未来结果预测器 Q 以及有限候选重采样策略；

**📊 数据集**

在五大推理基准上评估：MATH、OlympiadBench、ScienceQA、CommonsenseQA 和 Knights&Knaves；

**📈 对比分析**

与无验证器、后置验证器加权投票、束搜索等方法对比，VGSC 在所有任务上均取得最高平均准确率，并在计算成本上更优；

**⚠️ 局限性**

需要额外训练未来结果预测器，且对验证器的质量和领域适配度高度依赖，过度依赖验证器时易产生误差放大。

---

## 529. Time-Division Multiplexing: The Operational Rhythm and Temporal Organization Layer of Digital Life

**arXiv ID:** 2611.06603 | [PDF](https://arxiv.org/pdf/2611.06603v1)

**作者:** Haibin Tong `[一作]` (Henan University), Jiang Yu `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于时序组织层的忙/闲学习框架，利用相位标记的微时段、可观测占空比、独立的内容/资源控制以及三层时钟实现在线响应与离线反思的分离与协同。

**💡 创新点**

创新点包括：①引入相位标记的微时段与占空比来显式记录忙闲状态；②将内容选取与资源调度解耦，提供可单独干预的控制器；③使用三层嵌套时钟区分候选探索、任务执行与持久状态提交；④推导条件投影梯度下降保证并给出噪声下的期望收敛式；⑤在合成实验与冻结底层LLM原型上验证设计。

**🔧 技术方法**

技术手段：基于冻结的LLM（Qwen3‑8B）实现可重入的两循环架构；使用阶段化微时段记录（phase‑tagged records）与阴影缓存；构造多尺度时钟与占空比 ω；实现投影梯度更新与多门控制（安全、能力、价值、身份）。

**📊 数据集**

使用了完全合成的数据集：64个随机种子下的高维二次势、噪声梯度、任务负载与日志丢失等五类实验（E1–E5）；未使用任何真实任务或人类评测数据。

**📈 对比分析**

比较方法：在同等计算预算下对比解耦与耦合的内容/资源调度、占空比混合策略、噪声比例与增益比等参数；结果显示解耦调度在高紧急/高负载子集上成功率提升约15.4个百分点（26.56% vs 11.13%），并在其他实验中展示了占空比对终端势能的影响、噪声下的期望下降以及增益-噪声最优点的存在。

**⚠️ 局限性**

局限性：仅在合成实验与原型环境下验证；缺乏真实语言模型任务、长期用户交互与人类评测；安全与合规性验证仍处于原型阶段；占空比与时钟尺度未针对真实推理延迟优化；理论收敛保证假设梯度相同且无偏，现实中目标可能不一致；未实现完整的权限、隐私与审计机制。

---

## 530. Timed Games under Environmental Interference with Real-Time Objectives

**arXiv ID:** 2611.06450 | [PDF](https://arxiv.org/pdf/2611.06450v1)

**作者:** Andrea Manini `[一作]` (Politecnico di Milano), Pierluigi San Pietro `[通讯]` (Politecnico di Milano)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了Timed CLTLoc Games（TCGs）框架，定义了其语义并开发了后向求解与赢策略合成算法，随后在生产单元与医疗人机交互案例中进行实验验证。

**💡 创新点**

核心创新在于：①将位置而非转换划分为玩家控制区，允许在控制器选择动作后由环境决定延迟和非确定性；②利用CLTLoc线性时序逻辑表达获胜条件，实现对外部时钟的无结构约束；③设计后向可控前驱运算并证明可达性TCG求解为EXPTIME‑complete；④合成最小步数的赢策略。

**🔧 技术方法**

技术手段包括：时钟区域抽象、可控前驱算子、后向搜索、CLTLoc逻辑判定、OpenMP并行化、C语言实现的库（Timed Automata Region and Zone library for real-time systems ANalysis）。

**📊 数据集**

实验使用两个案例模型：传统生产单元（制造机器人）和医疗人机交互场景；未使用公开数据集，而是基于业务需求构造的模型。

**📈 对比分析**

通过与现有时序游戏工具（如UPPAAL‑Tiga、Probabilistic Timed Games等）的功能对比，展示了TCG在建模复杂交互模式方面的优势；实验在MacBook Pro M3上完成，结果表明算法在所选案例中可行，性能与理论复杂度相符，但在更大规模模型上仍需进一步验证。

**⚠️ 局限性**

主要限制包括：①后向算法导致的规模可扩展性受限；②依赖时钟区域抽象，可能过于粗糙；③仅支持单一TCG而非网络；④受限于目前支持的CLTLoc胜利条件；未来计划开发前向即时算法、网络化TCG及混合区域/区域区间方法。

---

## 531. Signed, Sealed, Leaked: Trust Beyond Signature for MCP Toolchains

**arXiv ID:** 2611.06585 | [PDF](https://arxiv.org/pdf/2611.06585v1)

**作者:** Yichiet Aun `[一作]` (Universiti Tunku Abdul Rahman), Sin Chee Tan `[通讯]` (ABLEs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Germline Execution（种子执行）机制，用以在 Model Context Protocol (MCP) 工具链中实现签名之外的信任，能够防止已签名工具内潜藏的特洛伊木马在执行过程中泄露机密数据。

**💡 创新点**

创新点包括：① 将每次工具调用隔离为从无数据种子进程派生的拷贝，保证任何执行只能看到其声明的可访问数据；② 通过声明目的地拆分与视图分组实现最小化拷贝；③ 证明了对所有签名发布的 sink 非干扰性，并提供可验证的流违规 witness；④ 在保持功能一致性的前提下，显著降低了调用延迟（单拷贝 10–49 ms，远快于每次调用重启进程）。

**🔧 技术方法**

采用的技术包括：Linux PID、网络和 IPC 命名空间、Landlock 只读隔离、eBPF/Kern‑LSM（MCPGuard）对目的地的拦截、Python 级别的时间与随机数双胞胎执行、请求序列对比、槽（slot）状态存储、探测重放、Dampening（路由抖动抑制）以及标签与清晰度的 IFC。

**📊 数据集**

使用的数据集：52 个手工编写的带隐藏代码的签名发布（覆盖 15 种攻击变体），91 个由 Qwen 生成的自动化变体，5 个标准 MCP 服务器（文档、SQL、Web、Lint、Mail）配合流式兼容与恶意发布，33 次安全无害调用，7 个 LLM 驱动的任务场景，总计 153 次攻击运行和多达 504 条 LLM 任务记录。

**📈 对比分析**

与基线（每次调用新进程）和 MCPGuard‑Dynamic/AgentBound 等现有防御进行对比：在 153 次攻击运行中，Germline Execution 在所有场景下均无泄露，而基线泄露率在 5.9–74.5% 之间；单拷贝调用延迟 10–49 ms，平均比每次重启进程快 2–28 倍；内存占用仅比基线多 20 MB。实验还验证了 witness 的可验证性、覆盖率与性能。

**⚠️ 局限性**

限制与假设：仅关注泄露（不涵盖命令执行或动作控制），远程不托管工具无法被 fork 仍需依赖 worst‑case reply labels；硬件非确定性（如 rdtsc）与时间通道不在保护范围内；依赖数据自由种子进程与 Twin 输入的正确性；需要部署者手工提供标签、清晰度和声明目的地，且对高并发、共享状态工具的支持有限。

---

## 532. RAGAM-AI: A Hybrid Reinforcement Learning based Agentic approach powered by Generative AI for Edge Resource Management

**arXiv ID:** 2611.06554 | [PDF](https://arxiv.org/pdf/2611.06554v1)

**作者:** Shreya K. Chari `[一作]` (Iquadrat Informatica S.L.), Christos Verikoukis `[通讯]` (University of Patras)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了RAGAM-AI，一种将多智能体软演员-评论家（MASAC）与生成式AI（LLM）相结合的混合框架，用于多接入边缘计算（MEC）单元的CPU、内存和带宽资源调度，旨在降低延迟与数据速率成本。

**💡 创新点**

创新点在于通过LLM（MathΣtral）提供链式推理奖励来弥补RL奖励泛化不足，并在训练后引入轻量级的DNN代理奖励模型，实现从LLM到近似奖励的逐步过渡；同时采用分布式多智能体架构与Agentic AI设计，提升决策自适应与协同。

**🔧 技术方法**

核心技术包括多智能体软演员-评论家（MASAC）强化学习、基于链式思考的MathΣtral LLM奖励、受监督的DNN奖励代理、RL‑from‑AI‑feedback (RLAIF)以及离线奖励模型训练与自适应采样。

**📊 数据集**

实验采用微软Azure Packing Traces数据集，包含虚拟机、容器与任务的CPU、内存与带宽需求信息。

**📈 对比分析**

通过与单智能体SAC、REINFORCE和随机策略基准对比，利用资源偏差、延迟成本和数据速率成本等指标评估性能，RAGAM-AI在CPU/内存/带宽偏差分别为23.65%、20.36%和20.40%，明显低于SAC、REINFORCE和随机方案，并且延迟与数据速率成本也更低。

**⚠️ 局限性**

局限性包括仍停留在仿真验证阶段，未在真实边缘环境中部署；对LLM推理开销和训练样本量有一定依赖；代理奖励模型需要满足MAE阈值；以及系统对参数调优、网络动态变化的敏感性。

---

## 533. Who Gets Counted? Large Language Models, Informative Refusal, and Downstream Bias in Name-Based Demographic Classification

**arXiv ID:** 2611.06550 | [PDF](https://arxiv.org/pdf/2611.06550v1)

**作者:** Christian Pipal `[一作]`, Eva-Maria Vogel `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）在姓名基础的人口属性分类（性别、种族）中的准确性及其对后续差距估计的影响，并与传统工具进行了比较。

**💡 创新点**

发现LLM的高条件准确率并不总能提升估计精度，缺失标签导致选择偏差，而BIRDiE在估计族群差距方面更可靠。

**🔧 技术方法**

使用了Claude Haiku、Gemini、Qwen等LLM，以及WRU、pyethnicity、ethnicolr等传统分类器。

**📊 数据集**

采用了10,000个中文拼音姓名、19,998名自报种族的北卡州选民样本以及300名政治学作者的手工编码。

**📈 对比分析**

通过覆盖率、条件准确率、总准确率等指标比较，LLM在覆盖率低但精度高的场景下表现优异，但在整体估计偏差方面仍落后于传统工具；混合后缀模型可略微提升单一标签精度。

**⚠️ 局限性**

主要局限包括LLM拒绝率导致的选择偏差、缺乏针对非美国姓名的验证、对连续结果差距估计不稳定以及对真实自报身份的依赖性。

---

## 534. EvoOPD: Evolving Privileged Information Strategies for Black-Box On-Policy Distillation

**arXiv ID:** 2611.06542 | [PDF](https://arxiv.org/pdf/2611.06542v1)

**作者:** Ximo Zhu `[一作]` (University of Chinese Academy of Sciences), Wenhao Wang `[通讯]` (Vast Intelligence Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了EvoOPD框架，在黑盒教师环境下通过进化的特权信息（PI）策略进行 on‑policy 蒸馏，将教师文本反馈转换为学生轨迹上的局部监督。

**💡 创新点**

创新点包括：将PI获取策略视为可优化的目标，构建可演化的策略库并根据学生状态动态调整信息获取；利用修复效用评估驱动策略更新；通过PI的局部概率响应生成密集且有界的监督信号。

**🔧 技术方法**

技术手段包括：on‑policy 蒸馏（GRPO、ROPD等）、PI‑conditioned likelihood 差值、二元组相对优势、stop‑gradient、KL 正则化；策略库的预训练与在线迭代；修复效用评价机制；使用 Qwen3 系列学生模型与 Gemini 黑盒教师；实现文本交互与自适应策略采集。

**📊 数据集**

使用数据集：数学推理训练集 DAPO‑Math‑17K；评测集 AIME、HMMT；代码生成训练集 OpenThoughts‑114k；评测集 EvalPlus、LiveCodeBench v5/v6、BigCodeBench Hard v0.2.5。

**📈 对比分析**

通过与 SeqKD、OPSD、GRPO、ROPD 等基线比较，EvoOPD 在所有三种学生规模（Qwen3‑1.7B、4B、8B）的数学和代码任务中均获得最优整体性能，数学平均提升 2.14pp，代码平均提升 1.30pp，表现显著优于最强基线。

**⚠️ 局限性**

局限性包括：实验仅在单一黑盒教师（Gemini）和单一学生模型家族（Qwen3）上验证，跨教师与跨模型泛化尚未探索；修复效用与最终学习效果的关联不完全，可能无法完全反映性能提升；PI 文本的语义正确性未被严格验证，可能导致不准确的监督信号。

---

## 535. Construction Contracts for Generated Embedded Domain-Specific Language Programs: A Measurement Protocol and a Conditional Effect

**arXiv ID:** 2611.06525 | [PDF](https://arxiv.org/pdf/2611.06525v1)

**作者:** Haowen Guo `[一作]` (University of Electronic Science and Technology of China), Yangzhao Huang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对大语言模型生成的 OpenQASM 3 代码，本文设计了“构造合同”(construction contract)并用文本 walk 方法评估解析成功率与对参考构造的完全匹配率。

**💡 创新点**

创新点在于将结构一致性拆分为可在生成前插入的合同与后验检查两步，提供了仅通过文本即可判定的工程属性，从而显著提升生成代码的结构可信度。

**🔧 技术方法**

技术上使用 Qwen2.5（3B/7B）模型的双阶段 Fine‑tune、前向规划器、文本 walk 检查器，并结合 KL 散度、能量误差等统计检验进行评估。

**📊 数据集**

数据集包含 10,207 条 OpenQASM 3 参考电路，进一步生成 100 条 QAOA 与 100 条 VQE 实例，涵盖不同 qubit 数与深度，并构建训练/测试拆分。

**📈 对比分析**

采用配对 McNemar 检验与 95% 自助区间比较解析率、全匹配率等指标；在 QAOA 上合同条件提升全匹配率约 +23–27 百分点，VQE 上几乎无提升。

**⚠️ 局限性**

局限性包括仅验证两类量子电路，合同未包含参数预算导致 VQE 结果差；评估仅基于文本 walk，未验证对实际下游工具的实用性。

---

## 536. H^2MARL: Hierarchical Heterogeneous Multi-agent Reinforcement Learning for Humanoid Robot Control

**arXiv ID:** 2611.06523 | [PDF](https://arxiv.org/pdf/2611.06523v1)

**作者:** Xuemin Hu `[一作]` (Hubei University), Long Chen `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出一种层次化异构多智能体强化学习框架 H2MARL，用于人形机器人全身控制。

**💡 创新点**

①将每个扭矩驱动的关节自由度建模为独立异构智能体并引入物理异质性感知的策略因式分解；②设计姿势增量规划并使用 Human State Transformer 作为低层执行器；③通过闭环策略‑执行器协同优化解决层级模型分离导致的分布失配问题。

**🔧 技术方法**

异构多智能体强化学习（HAPPO）、层次化控制、姿势增量规划、Human State Transformer、PD 控制器、闭环策略‑执行器共优化。

**📊 数据集**

MuJoCo 与 Isaac Gym 的 Humanoid‑v2 模拟环境，无真实数据集。

**📈 对比分析**

与 HAPPO、MAPPO、HATD3 进行对比，使用 Foot Slip、Torso Stability、Gait Symmetry、Motion Smoothness 四项指标，H2MARL 在 FS、TS、GS 上均优于同类方法（提升约 7%–35%），但在 MS 上略逊。

**⚠️ 局限性**

闭环共优化增加了计算负担，运动平滑度略下降；方法仅在模拟环境验证，未验证真实机器人；对极端动态任务的适应性仍待评估。

---

## 537. Scythe: Semantic-Aware Rewriting for Fast,Effective Program Reduction

**arXiv ID:** 2611.06515 | [PDF](https://arxiv.org/pdf/2611.06515v1)

**作者:** Ioanna Moraiti `[一作]` (Unaffiliated), Thodoris Sotiropoulos `[通讯]` (Unaffiliated)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种前置语义感知的预处理工具 Scythe，能在语言无关的归约器前快速移除无关声明，从而提升归约效率和结果质量。

**💡 创新点**

创新点在于通过构造声明依赖图并应用语义解耦算子，使得在不改动原有归约器的前提下实现语义层面的代码删减。

**🔧 技术方法**

主要技术包括依赖图构建、Delta Debugging、语义解耦算子、Tree‑Sitter 语法解析以及 Picire 库的 delta‑debugging 框架。

**📊 数据集**

使用了73个跨 C、Java、Solidity 的程序集，包括 Csmith、Hephaestus 生成的测试用例以及 Slither 静态分析器触发的合约。

**📈 对比分析**

与两种顶尖归约器（基于语法的 Perses 与加入语言特定变换的 LPR）对比，Scythe+归约器的最终结果平均缩小 55.6%（最多 95.5%），速度提升 2.3×（最多 7.4×），在 70/73 个基准上至少不增大并大多数情况下更小。

**⚠️ 局限性**

限制在于默认值替换可能导致后续归约器回退或结果略增，且对某些语言特定构造的支持尚不完善。

---

## 538. ElecLoad: Adaptive Electricity Load Forecasting Guided by Multiple Events

**arXiv ID:** 2611.06508 | [PDF](https://arxiv.org/pdf/2611.06508v1)

**作者:** Huajian Zhang `[一作]` (University of Science and Technology of China), Zhi Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 ElecLoad，一种基于多事件分析的自适应电力负荷预测框架；

**💡 创新点**

创新点在于将事件影响的时间变化与相互作用融入预测流程，动态决定是整体预测还是按事件效应划分段落，并通过多代理协作进行选择性细化和边界校准；

**🔧 技术方法**

采用 Qwen3.5-4B 语言模型为核心，训练包括推理初始化、角色监督微调和强化学习；利用文本事件描述、协变量与历史负荷构建特征；

**📊 数据集**

在多组电力系统数据集上评测：GEFCom2014、Nord、PJM、SPP、LCL、iFlex、AEMO、ERCOT（含日内与多日预测）；还构造了带单/多事件的合成测试案例；

**📈 对比分析**

与 ARIMAX、LightGBM、PatchTST、EventTSF、Chronos-2、TimesFM-3、TimeReasoner、MemCast、AlphaCast、CastFSR、MMGPT4LF、NACF、ITA-LF 等 15 种基线对比，ElecLoad 在 22 个评估维度中获得 13 项第一、6 项第二，整体 WAPE 与 NRMSE 均显著优于其他方法，尤其在多事件情境下提升更明显；

**⚠️ 局限性**

局限性包括对事件文本质量与完整性的依赖，若事件描述不完整或误导性信息可能导致误判；多段预测与细化过程中计算代价提升，需权衡 token 消耗与精度；模型对超出训练域的极端事件或系统可能仍表现不佳。

---

## 539. Mortality Is Undecidable for $2 \times 2$ Integer Matrices

**arXiv ID:** 2611.06562 | [PDF](https://arxiv.org/pdf/2611.06562v1)

**作者:** Thanin Quartz `[一作]` (University of Waterloo), Jun Liu `[通讯]` (University of Waterloo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明二维整数矩阵死亡问题不可判定，并给出一个包含 5,799 个矩阵的固定族

**💡 创新点**

通过将通用两计数器机的计算嵌入到 2×2 矩阵乘积，利用基数检查和秩‑一变换实现对计算路径的严格约束

**🔧 技术方法**

采用基数分解编码、基数‑一数字检查、有限指令模块与矩阵生成器，以及 Lean 形式化验证

**📊 数据集**

无实验数据集，问题为理论决策问题

**📈 对比分析**

不涉及实验比较，结果为理论可判定性证明

**⚠️ 局限性**

证明中矩阵族规模较大且未做最优化，仅针对整数矩阵，其他域的情况仍未解决

---

## 540. A Sub-$3^n$ Upper Bound on the Number of Stable Matchings

**arXiv ID:** 2611.06604 | [PDF](https://arxiv.org/pdf/2611.06604v1)

**作者:** Ken-ichi Kawarabayashi `[一作]` (National Institute of Informatics), Masataka Yoneda `[通讯]` (University of Tokyo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在 n 名男性与 n 名女性中最大稳定匹配数，并给出新的 2.999^n 上界。

**💡 创新点**

通过改进熵法框架，提出“锥”和“支撑”等结构概念，并利用充电论证进一步压缩上界。

**🔧 技术方法**

主要使用熵基框架、结构分析、充电论证等组合方法。

**📊 数据集**

该研究为纯理论分析，无实验数据集。

**📈 对比分析**

与 Palmer 与 Pálvölgyi 的 3.332^n 上界对比，本文将上界降低至 2.999^n，更接近 2.28^n 的下界。

**⚠️ 局限性**

仍未能达到下界，仍有 0.7 的误差，且只给出理论上界，缺乏实验验证。

---

## 541. Tools Need Not Speak in Text: Latent Context Compression for Agents

**arXiv ID:** 2611.06500 | [PDF](https://arxiv.org/pdf/2611.06500v1)

**作者:** Yuetong Liu `[一作]` (Tsinghua University), Jia Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种连续潜在上下文压缩模型，能够将工具返回的长文本压缩为少量潜在令牌，并在 LLM 代理中直接使用。

**💡 创新点**

核心创新在于放弃传统文本压缩，使用可微分的潜在令牌实现信息压缩，并通过“动作保持训练”保证压缩后仍能保留决定关键的信息。

**🔧 技术方法**

采用轻量级上下文感知编码器 + 潜在适配器、KL 对齐 + 交叉熵损失的动作保持训练，以及将潜在向量嵌入代理输入空间。

**📊 数据集**

在四个开放域问答基准（Natural Questions、TriviaQA、2WikiMultiHopQA、HotpotQA）上评测，并使用 Qwen3.5-35B/4B 与 GLM-4.7-Flash/Z1-9B 两组模型。

**📈 对比分析**

与未压缩模型、LongLLMLingua、LLMSummary、ICAE 等基线对比，压缩率提升至 70–90%，输入/输出 token 消耗下降 70–90%，推理时间下降 25–55%，且在 13/16 评估维度中获得最高 EM/F1，甚至在多数设置下表现优于基线。

**⚠️ 局限性**

局限性包括：仅在工具返回文本的场景下验证，潜在令牌需要额外训练并依赖冻结的代理，未探讨对其他工具（API、数据库）或跨任务迁移的通用性。

---

## 542. The Behavioral Rate of Fine-Tuning: Compression as a Measure of Useful Information

**arXiv ID:** 2611.06598 | [PDF](https://arxiv.org/pdf/2611.06598v1)

**作者:** Joey David `[一作]` (Universite Paris Dauphine Psl), Alexandre Allauzen `[通讯]` (Universite Paris Dauphine Psl)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对冻结的语言模型使用LoRA适配器进行微调，并提出行为率R⋆来衡量压缩后适配器在保持微调性能上的比特量。

**💡 创新点**

创新点在于把微调视为接收器相对的编码问题，引入行为率、压缩对性能的恢复作用，以及基于更新可缩放性的经验预测公式。

**🔧 技术方法**

采用LoRA、量化/截断压缩、序列化存储、矩阵奇异值分解、行为率计算与曲线拟合。

**📊 数据集**

使用Qwen、Mistral、Llama、Gemma等多尺度模型与GSM8K、MetaMathQA、XBRL等数据集。

**📈 对比分析**

通过对比压缩率与保留微调增益的曲线、量化误差对行为的影响以及压缩恢复的案例，展示压缩率随数据-接收器对变化、压缩可显著恢复被破坏的微调，且行为率可用更新可缩放性预测，性能在90%保持增益时约需0.4-1.0比特/值。

**⚠️ 局限性**

局限在于行为率与压缩方案相关、纠正谱在不同接收器上不具可转移性、可缩放性规则仅在回顾性实验中验证、实验使用人为破坏监督不一定代表真实微调误差。

---

## 543. BudgetSplat: Budget-Controllable Generalizable Gaussian Splatting

**arXiv ID:** 2611.06593 | [PDF](https://arxiv.org/pdf/2611.06593v1)

**作者:** Yuxi Hu `[一作]` (Graz University of Technology), Friedrich Fraundorfer `[通讯]` (Graz University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出BudgetSplat，能够在测试时根据用户指定的保留比例动态调整3D高斯点的数量，从而实现可控的表示压缩。

**💡 创新点**

创新点在于将高斯压缩视为重要性、空间覆盖和跨视角冗余三者的联合优化，并通过预算条件特征调制与覆盖感知预算选择实现单模型多预算支持。

**🔧 技术方法**

采用像素对齐的FreeSplat骨干网络，结合Intra-Cross Mask监督、预算条件特征调制和覆盖感知预算选择等技术。

**📊 数据集**

在ScanNet和Replica数据集上进行训练与评估。

**📈 对比分析**

与现有基于像素对齐和压缩的3D高斯方法（如EcoSplat、FreeSplat、GlobalSplat等）比较，在相同或更少的高斯数量下保持或提升PSNR/SSIM/LPIPS，尤其在25%预算下接近FreeSplat质量。

**⚠️ 局限性**

局限性包括对图像梯度和深度估计的依赖，极端压缩下可能因梯度噪声导致覆盖不足，以及仍需密集像素对齐预测骨干导致基线模型复杂度高。

---

## 544. PatchCon: Patch-aware Weighted Supervised Contrastive Learning for Vulnerability Detection

**arXiv ID:** 2611.06583 | [PDF](https://arxiv.org/pdf/2611.06583v1)

**作者:** Chaomeng Lu `[一作]` (KU Leuven), Bert Lagaisse `[通讯]` (KU Leuven)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PatchCon框架，通过将漏洞–补丁函数对作为加权硬负样本，联合监督对比学习与分类来进行函数级漏洞检测。

**💡 创新点**

创新点在于显式利用漏洞–补丁对作为高权重硬负样本，并在监督对比学习中对其加权，使模型能够捕捉补丁引入的细微安全语义差异。

**🔧 技术方法**

技术手段包括基于CodeBERT的预训练编码器、chunk‑based max pooling聚合长函数、权重监督对比学习（SupCon）与分类损失联合训练。

**📊 数据集**

使用的主要数据集为PairVul‑Linux（基于Linux kernel CVE及其补丁构建的函数对）以及VentiVul（后续公开的Linux内核漏洞集）进行泛化评估。

**📈 对比分析**

与多种预训练代码模型、对比学习基线以及LLM（Vul‑RAG/GPT‑4o）进行对比，PatchCon在ACC、MCC、PACC、MCG等指标上显著优于所有基线，PACC提升至0.3264，几乎与LLM水平相当但成本更低。

**⚠️ 局限性**

局限性包括：依赖漏洞–补丁对的可获得性，主要针对C语言Linux内核，未对多语言、文件级扫描或漏洞定位等更广泛场景进行评估。

---

## 545. From Fixed Defaults to Calibrated Activation Policies for Long-Term Time-Series Forecasting

**arXiv ID:** 2611.06559 | [PDF](https://arxiv.org/pdf/2611.06559v1)

**作者:** Shijie Xu `[一作]` (Nanjing University of Information Science and Technology), Jianghua Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在已有的长时序预测模型Informer上，研究者针对不同数据集和预测时长，系统地探索激活函数及其插入位置的组合，并构建了一套基于先验实验结果的低成本、可解释的激活配置选择策略。

**💡 创新点**

创新点包括：①将激活函数与位置视为可调节的“配置组合”而非固定常量；②提出了“CF‑SBC”全局校准层与“因子条件化覆盖层”相结合的两层策略；③通过离线规则生成+确定性验证实现无训练、无标签的快速推理；④在实际任务上实现了相对GELU平均7.86% MSE下降、0.89% 增量提升，并将相对最佳配置的平均遗憾降低20.9%。

**🔧 技术方法**

技术手段包括：完整的三种激活配置空间（9种），对输入窗口统计量（波动率、跳跃、尾部、频谱熵、H/L比）进行分桶生成因子规则；基于校准折（leave‑dataset）计算全局最优配置CF‑SBC；使用确定性规则验证器进行覆盖判定；与CART、RIPPER、SATzilla等经典选择器进行对比。

**📊 数据集**

使用了ETT、Weather、Electricity、Traffic、Solar、ILI、Exchange等公开长时序基准；实验覆盖17个Informer数据集变体、51个数据集‑预测时长组合；还在PatchTST和iTransformer上做了架构迁移验证。

**📈 对比分析**

与传统的单一GELU默认、单一全局最佳以及经典统计/机器学习选择器对比，所提策略在51个任务上平均MSE下降7.86%（相较GELU），相对全局最佳CF‑SBC提升0.89%，并将平均oracle regret从6.10%降至4.82%；与CART、RIPPER、SATzilla的最优/虚拟最佳方案相比，均取得更高的平均绝对MSE/MAE改善。

**⚠️ 局限性**

局限性：①仅在Informer（以及有限的PatchTST、iTransformer）上验证，需进一步探究在更广泛模型上的迁移性；②规则生成依赖先验实验和LLM协助，规则覆盖率仍有限；③因子特征选择和阈值设定仍带有经验性；④对极端数据或超大规模预测任务的鲁棒性尚未充分评估。

---

## 546. Using Experience Twice: Dynamic Process Rubrics for Privileged Token-Level Credit Assignment

**arXiv ID:** 2611.06543 | [PDF](https://arxiv.org/pdf/2611.06543v1)

**作者:** Shumao Wu `[一作]` (Computer Network Information Center, University of Chinese Academy of Sciences), Baisheng Lai `[通讯]` (Computer Network Information Center, University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ORSD框架，通过在线生成过程式rubric并将其作为特权上下文，对LLM代理的token级奖励进行细粒度信用分配，提升RL训练效率。

**💡 创新点**

创新点在于：①利用当前策略的轨迹差异自动生成动态rubric，避免手工或固定rubric；②通过rubric‑reward对齐机制挑选最具指示性的rubric bundle；③利用同一轨迹的log‑likelihood差值转化为token级优势，构成自蒸馏的精细信用信号。

**🔧 技术方法**

技术包括：GRPO（Group Relative Policy Optimization）作为基线；在线rubric生成、Verifier评估与Spearman相关性对齐；Token‑level advantage shaping；自蒸馏paired replay；CLIP PPO风格的训练目标。

**📊 数据集**

数据集：ALFWorld与WebShop两个长时序交互基准，使用Qwen3-1.7B和Qwen3-4B两种LLM后端。

**📈 对比分析**

与基线GRPO、GiGPO及自蒸馏方法（OPSD、SDAR）对比，ORSD在ALFWorld总体成功率提升约12%（1.7B）/3%（4B），在WebShop成功率提升约2.8%（1.7B）/10.9%（4B），整体任务得分与成功率均居首位。

**⚠️ 局限性**

限制在于：①rubric知识不被持续存储或累积；②rubric‑reward对齐侧重轨迹层级关联，未直接优化token级信用效果；③对不同任务的可迁移性和鲁棒性仍待验证。

---

## 547. Dialect Matters: A Dialect-Aware Baseline for Igala-English Machine Translation

**arXiv ID:** 2611.06517 | [PDF](https://arxiv.org/pdf/2611.06517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 548. Spatially and Compositionally Controlled Multimodal Glioblastoma MRI Synthesis with Latent Diffusion

**arXiv ID:** 2611.06540 | [PDF](https://arxiv.org/pdf/2611.06540v1)

**作者:** Steven Glass `[一作]`, Willie Brink `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种两阶段潜在扩散模型框架，用于根据可控的肿瘤位置、大小和分区比例生成多模态胶质母细胞瘤脑MRI切片。

**💡 创新点**

创新点在于将肿瘤分割生成与MRI合成分离，通过Seg‑LDM实现对肿瘤几何与成分的条件控制，并在后端使用ControlNet将该分割映射到多模态MRI，提升了生成的可控性与质量。

**🔧 技术方法**

使用了潜在扩散模型（LDM）、控制网络ControlNet、VAE编码器/解码器、分类器无指导技术，以及FID和专家读者评分等评估方法。

**📊 数据集**

基于公开的UCSF‑PDGM和UPenn‑GBM共1105例病例，包含T1、T1Gd、T2、FLAIR四模态以及对应肿瘤分割。

**📈 对比分析**

与仅使用LDM背骨相比，启用ControlNet后四模态FID平均降低6.47点（从42.59降到36.12），生成分割中心误差0.842像素，子区比例误差0.795%；专家读者评分显示合成图像整体质量与真实图像相当或更高。

**⚠️ 局限性**

主要限制在于仅在2D切片上验证，尚未扩展到完整体积生成；读者评估样本有限，难以全面评估真实性；缺乏跨数据集的泛化验证。

---

## 549. Vertex-subgraph eigenvector centrality for connected induced subgraphs

**arXiv ID:** 2611.06534 | [PDF](https://arxiv.org/pdf/2611.06534v1)

**作者:** Qingying Zhang `[一作]` (Harbin Engineering University), Changjiang Bu `[通讯]` (Harbin Engineering University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个同时评估图中顶点与三元诱导子图（路径P3和三角K3）中心性的联合谱方法——顶点-子图特征向量中心性（VSC）。

**💡 创新点**

创新点在于将顶点与子图的得分通过互相强化的非线性特征向量方程联合起来，并通过一个四阶对称非负张量完全表述，使得存在唯一正特征向量保证中心性的一致性。

**🔧 技术方法**

使用了张量谱理论、弱不可约性证明、ZQW迭代求解正特征向量，以及对顶点和子图容量（边数）进行加权的技术。

**📊 数据集**

实验使用了四个真实网络数据集：Karate、C. elegans、Watts电力网和Facebook页面网络，均包含大量顶点和三元诱导子图。

**📈 对比分析**

通过Spearman秩相关、Jaccard重叠率以及三角子图比例比较VSC与传统中心性（度、特征向量、三角、介数）的关系。结果显示VSC与这些指标高度相关，但在高排名顶点的重叠度因网络而异；容量加权对子图排名影响更大。

**⚠️ 局限性**

局限性包括：仅考虑三元诱导子图；对大规模图的计算成本仍高；缺乏对非连通图或有向图的推广；以及容量加权方式可能对结果产生较大影响，需要进一步研究更稳健的加权策略。

---

## 550. EmoReward: Self-Evolving Reward Calibration for Emotional Video Generation RL

**arXiv ID:** 2611.06518 | [PDF](https://arxiv.org/pdf/2611.06518v1)

**作者:** Mingzhe Zheng `[一作]` (Hong Kong University of Science and Technology), Harry Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发并实现了 EmoReward，一套自演化的情感视频生成强化学习奖励校准框架，结合实例级反射奖励门控与奖励记忆表，在不更新奖励模型的前提下通过已完成训练运行的经验动态分配多奖励提供者的权重。

**💡 创新点**

创新点：
• 首创跨训练运行自演化的奖励校准机制；
• 引入 RewardMemory，用于存储多奖励提供者的可靠性估计并生成可固定的融合乘子；
• 通过 EmoScript 将情感意图与可观测线索分离，实现情感目标与视觉证据的分层评估。

**🔧 技术方法**

技术：
• Group Relative Policy Optimization (GRPO) 强化学习；
• 多奖励融合与实例级反射奖励门控（Reflective Reward Gating）；
• 情感脚本规划（EmoScript）与多源信号原语（VLM 判定、面部 AU、身体线索、时序证据、CLIP-AUTT 等）；
• 奖励记忆表与经验驱动的权重校准；
• 视频评估工具 VideoAlign、PickScore、CLIPScore。

**📊 数据集**

数据集：
• HunyuanVideo 1.5（包含 302 条情感提示的评估集）；
• Stable Diffusion 3.5 与 FLUX（用于图像基准迁移实验）；
• 18 名参与者对 30 条提示的主观情感、时序一致性与视觉质量评估数据。

**📈 对比分析**

比较方法与性能：
• 在 DanceGRPO、FlowGRPO、SAGE‑GRPO 三种 RL 算法上，使用相同的训练预算和融合系数进行对比；
• 通过 VideoAlign 综合分数、VQ、MQ、TA、PickScore、CLIPScore 等指标评估；
• EmoReward 在所有算法上均实现 VideoAlign 提升（如 DanceGRPO +0.32，SAGE‑GRPO +0.56）；
• 人工评测显示 EmoReward 的 Good 比例最高、Bad 比例最低；
• 消融实验表明 RewardMemory 提升可靠性选择性最高，且在多奖励混合策略中表现最佳。

**⚠️ 局限性**

局限性：
• 性能受目标评估指标影响，部分指标提升有限；
• 需要足够的完成训练运行样本才能构建可靠的奖励记忆表；
• 目前仅针对情感视频生成，跨模态或更复杂情境的扩展尚待验证；
• 对人类审美差异的处理仍以固定奖励为主，RLHF 的规模化和可解释性仍是挑战。

---

## 551. NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria

**arXiv ID:** 2611.06504 | [PDF](https://arxiv.org/pdf/2611.06504v1)

**作者:** Zainab Afolabi `[一作]` (Amazon), Babs Khalidson `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个多语言事实与统计问答基准（NigeriaBench），包含 5 种本土语言（英语、尼日利亚皮钦语、伊博语、豪萨语及另一语言）和 3 个国家（尼日利亚、英国、美国），以及将该基准扩展到所有 54 个非洲国家的英文版；通过跨语言、跨国家的设计，区分模型在语言处理和知识缺失两个失效模式。

**💡 创新点**

① 将语言效应与知识效应分离：在保持事实不变的前提下仅变更查询语言；② 采用官方统计数据作为严格可验证的真值；③ 交叉国家×语言矩阵，既能检测语言资源差异，也能评估对不同国家事实的掌握；④ 全面覆盖非洲 54 个国家，展示知识缺失在整个大洲范围内普遍存在。

**🔧 技术方法**

使用 8 大型语言模型（包括 6 个前沿模型和 2 个 8B 的尼日利亚语言适配模型）进行评估；采用统一的提示模板、4,096 token 的答案预算；答案以自由文本形式输出，使用 LLM‑as‑judge（Claude Opus 5）进行二进制判分；对答案进行轻量化标准化后计分；还做了“用英语回答”实验、语言资源与准确率的相关性分析。

**📊 数据集**

核心数据集：100 题目×5 语言×3 国家＝1,500 条评测项；每题均由尼日利亚本土母语者编写，并从 NPC、NBS、NDHS 等官方统计获取答案；英国和美国的答案来自 ONS 与 Census Bureau，并经作者核对；扩展数据集：为 54 个非洲国家分别构建 118–125 条英文问答，答案来源于 World Bank WDI 和 DHS API；所有答案均为公开的官方统计表格。

**📈 对比分析**

评估方法：对每个模型在不同语言和不同国家的准确率进行统计，计算语言效应（从英语到目标语言的准确率下降）和知识效应（同语言下不同国家的准确率差距）。结果显示：在英语下，模型在 US/UK 的官方统计上达 49–50% 的准确率，但在尼日利亚仅 20%；语言效应平均下降 9–24 分；知识效应远大于语言效应。扩展到 54 个国家时，模型在文化和政治类问题上达 96–98% 的准确率，而在统计类问题仅 26%。

**⚠️ 局限性**

局限性：尼日利亚子集仅 100 题（每语言）且主要来自少量表格，未能覆盖更广泛的事实范围；54 国扩展仅使用单个英文模型；官方统计数据虽然公开，但模型可能未充分吸收，导致低准确率；未来工作需扩展多语言设计、引入更多语言、在训练中直接注入官方表格等。

---

## 552. SpikePose: Lightweight Video Human Pose Estimation

**arXiv ID:** 2611.06594 | [PDF](https://arxiv.org/pdf/2611.06594v1)

**作者:** Yunhao Huang `[一作]` (University of Glasgow), Fani Deligianni `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为SpikePose的轻量级视频人体姿态估计框架，利用单步帧重置的尖峰编码器提取空间热图，并通过运动对齐热图记忆（MAM）实现跨帧运动一致性；

**💡 创新点**

创新点在于将时序推理从高维尖峰骨干网络迁移到轻量级的MAM模块，实现了在不携带未对齐膜状态的情况下获得高质量运动估计，同时保持尖峰网络的能效优势；

**🔧 技术方法**

使用的主要技术包括单步（T_snn=1）尖峰LIF卷积编码器、运动对齐热图记忆（包含位移估计、热图对齐、门控保留与残差融合）以及基于PCKh和头骨归一化PCK的评估指标；

**📊 数据集**

在MPII图像基准上评估空间编码精度，在NTU60视频数据集（Cross-Subject子集）上评估完整视频的空间与运动性能；

**📈 对比分析**

与同等架构的ANN基线相比，SpikePose在NTU60上的PCKh(0.5)达到83.81，MPJVE 3.50，MPJAccE 5.50；相较于无MAM的SpikePose，MAM提升了约0.18点准确率，显著降低运动误差（MPJVE↓28.13%，MPJAccE↓31.46%），并仅增加0.003M参数与0.0023GFLOPs；

**⚠️ 局限性**

局限性包括：1）单步尖峰编码器在极高运动速度下仍可能产生误差；2）MAM的对齐与残差补偿在极端姿态变形时效果有限；3）能耗评估基于理论代理，缺乏真实硬件测量；4）仅在特定视频子集验证，跨域泛化尚待进一步验证。

---

## 553. Does Personality Persuade? Investigating Personality Trait Prompting and Persuasion in Large Language Models

**arXiv ID:** 2611.06572 | [PDF](https://arxiv.org/pdf/2611.06572v1)

**作者:** Ruben Boone `[一作]` (University of Hasselt), Kris Luyten `[通讯]` (University of Hasselt)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过两步实验探究大语言模型在被提示时如何表达大五人格特质，以及这些人格特质是否会影响用户在道德困境对话中的说服效果。

**💡 创新点**

①验证不同模型和推理配置下人格提示的可重复性和可控性；②发现未被提示的 LLM 也具有明显人格基线；③评估人格匹配对说服力的真实影响，发现匹配并未显著提升说服效果；④揭示 AI 说服素养对用户决策信心变化的调节作用。

**🔧 技术方法**

使用大语言模型提示（分类与数值两种格式）与 IPIP-NEO-120 大五人格测评进行人格校准；在四个道德困境对话中嵌入不同人格配置的 LLM，并通过统计方法（ICC、Friedman、Cochran Q、Spearman 相关）评估说服效果；使用 LLM 辅助标签对对话中的说服策略进行编码。

**📊 数据集**

IPIP-NEO-120（人格测评问卷）、DailyDilemmas 数据集（四个道德困境）以及 Meta AI Literacy Scale（AI 说服素养量表）。

**📈 对比分析**

人格提示的可靠性通过 ICC（0.93–0.99）验证；直接效应与溢出量测量显示不同模型与推理配置对人格控制有显著差异；在用户实验中使用 Friedman、Cochran Q、Spearman 相关检验，结果表明人格匹配对说服力无显著提升，且信心变化与 AI 说服素养负相关。

**⚠️ 局限性**

研究样本仅 32 名参与者、只覆盖四个道德困境，未检验更大或在线服务模型；对话长度受限，可能未捕捉更细微的说服影响；说服效果主要通过立场转换和信心变化衡量，忽略了其他可能的影响维度。

---

## 554. Addition by Subtraction: Toward Mitigating Fragile Safety Alignment in Large Language Models

**arXiv ID:** 2611.06553 | [PDF](https://arxiv.org/pdf/2611.06553v1)

**作者:** Hamid Dashtbani `[一作]` (University of Duisburg-Essen), Alexandra Dmitrienko `[通讯]` (University of Duisburg-Essen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型安全对齐的稀疏性脆弱性，提出 ABS-Align 方法：在微调过程中逐步剪除已识别的安全关键神经元，并通过偏好优化重新分布安全对齐行为，提升模型对剪枝攻击及多种黑盒/白盒攻击的鲁棒性。

**💡 创新点**

创新点在于直接针对安全对齐的稀疏性进行干预：通过先识别并按重要性递归剪除安全关键神经元，再在剩余网络上进行 DPO 微调，迫使安全行为分布到更广泛的网络部位，从而大幅降低对单一神经元或方向的依赖。

**🔧 技术方法**

核心技术包括：TwinBreak 识别安全关键神经元；按重要性递归剪枝；使用 DPO（Direct Preference Optimization）进行偏好微调；逐步恢复所有神经元以完成训练；在评估中结合 StrongREJECT、SORRY-Bench、过度拒绝率、以及多项通用能力基准。

**📊 数据集**

数据集：安全偏好微调使用 PKU‑SafeRLHF（约1000条样本，辅以 850 条 Alpaca 与 150 条 XSTest 以保持通用性）；过度拒绝评估使用 XSTest；攻击实验使用 StrongREJECT（313 条恶意提示）和 SORRY‑Bench（10 种变换、440 条提示）；通用能力评估使用 MMLU、TruthfulQA、ARC‑Challenge、WinoGrande、HellaSwag。

**📈 对比分析**

与 W‑DOOR、Circuit Breaker、Deep Refusal 等现有防御进行对比；在四大开放权重模型（Gemma‑2‑2B‑IT、Llama‑3‑8B‑Instruct、Llama‑3.2‑3B‑Instruct、Qwen‑3‑4B）上，ABS‑Align 将 TwinBreak 等剪枝攻击成功率降低 55–90%，黑盒攻击成功率平均下降 20–40个百分点；过度拒绝率基本保持不变甚至略有提升；通用能力在多数基准上保持相近或略有提升。

**⚠️ 局限性**

局限性包括：依赖安全神经元检测方法，检测不完美可能导致部分安全特征残留或误剪；无法精确量化安全对齐密度，仅通过攻击效果间接衡量；DPO 微调需要额外的参考模型和计算资源；实验仅覆盖开放权重模型，对闭源模型的适用性待验证；适应性攻击在扩展剪枝预算后仍有限，但不排除更强攻击的潜在风险。

---

## 555. When Do Summaries Become Instructions? Unsupported Policy in LLM Agents' Self-Written Handoff Summaries

**arXiv ID:** 2611.06545 | [PDF](https://arxiv.org/pdf/2611.06545v1)

**作者:** Wenhui Chu `[一作]` `[通讯]` (State University of New York), Wenhui Chu (State University of New York)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对长时间运行的LLM代理在进行上下文压缩时生成的交接摘要中是否会出现未在记录中支持的权限或约束，并评估其对后续代理行为的影响。

**💡 创新点**

首次系统性量化交接摘要中无效权限/限制的出现频率，并证明修改系统提示或中性化摘要句子能显著降低不一致的权限声明。

**🔧 技术方法**

采用同时快照重放、规则标签器与AI读者判读以及系统提示改写等方法，对摘要句子进行检测、编辑与评估。

**📊 数据集**

使用三类合成任务（记录、报告、代码）以及对应的权限账本与事件日志，共计24个开发模板和60个测试模板。

**📈 对比分析**

通过对比原始摘要与中性化/真权限/完整历史四种处理方式，统计确认请求和未经授权操作的出现次数；在开发集上中性化降低确认率，系统提示改写将无效限制率从5%降至0.3%，显示显著提升。

**⚠️ 局限性**

实验局限于小规模开源模型、合成场景、AI读者标签与有限的模型覆盖，且在不同拆分与硬件配置下结果不完全稳健。

---

## 556. MKD2World: A Typed Spatial MarkDown World Model For Web Agents

**arXiv ID:** 2611.06544 | [PDF](https://arxiv.org/pdf/2611.06544v1)

**作者:** Zhihao Nan `[一作]` (Guangming Laboratory), Fei Yu `[通讯]` (Guangming Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Typed Spatial Markdown (TSM) 作为结构化、空间化的 Markdown 状态表示，构建 AtomBlock-WebUI 数据集，并基于 TSM 训练两阶段的 MKD2World 世界模型，配合 digest 推理提升 GUI 代理决策；

**💡 创新点**

1) TSM 兼顾完整性与结构化，编码元素类型、语义、位置、控制状态和层级；2) AtomBlock-WebUI 通过在 HTML 注入标签并直接读取 DOM 边界，提供高质量双粒度标注；3) MKD2World 采用基于 focus 元素的损失与 margin 约束，并在推理阶段引入 digest 步骤，显著提升预测与决策性能；

**🔧 技术方法**

使用 VLM 生成带标签的 HTML，YOLO+PP‑OCRv6 检测与识别 UI 元素，Qwen3.5‑9B+LoRA 训练世界模型，EleF1 评估匹配度，digest 机制整合预测结果；

**📊 数据集**

AtomBlock‑WebUI（约9.7k 网页）用于训练与评估，WebChain 与 MolmoWeb‑HumanTrajs 用于构建状态转换样本；Online‑Mind2Web 与 VisualWebArena 用于检验代理性能；

**📈 对比分析**

与 GPT‑5.6‑Sol、Qwen3.7‑Plus、GLM‑5V‑Turbo 等多模态模型在 EleF1 上比较，MKD2World 在 ID 设定下 28.09、OOD 26.50，均显著领先；在 Online‑Mind2Web 任务中 Qwen3.5‑35B 的成功率由 22.94% 提升至 37.61%，在 VisualWebArena 上亦实现最高成功率；

**⚠️ 局限性**

目前仅适用于静态网页，无法预测动画、音视频等动态变化；缺乏像素级对齐的细粒度预测；对复杂网页更新仍存在挑战。

---

## 557. IDSISA: Influence-Distilled Sharded Training for Efficient Machine Unlearning

**arXiv ID:** 2611.06519 | [PDF](https://arxiv.org/pdf/2611.06519v1)

**作者:** Deeraj SK `[一作]`, Anupam Shukla `[通讯]` (Sardar Vallabhbhai National Institute of Technology Surat)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 IDSISA：一种结合影响函数分区和知识蒸馏的 SISA 扩展，用于实现可证实的机器模型无学习；

**💡 创新点**

创新点在于利用影响函数将易被删除的样本聚集到少数分片，减少需要重训练的分片数量；并通过蒸馏将未受删除影响的分片的软输出作为教师，恢复因孤立重训练导致的性能损失，同时在多次删除请求下通过闭包级联重训练保证删除正确性；

**🔧 技术方法**

使用影响函数（基于梯度的 L‑BFGS 近似）、K‑means 聚类、PCA 降维、温度标定的蒸馏（KL 损失），以及 SISA 的分片切片训练；

**📊 数据集**

在 CIFAR‑10/CIFAR‑100（ResNet‑18）、IMDB（BiLSTM、TCN）、Fashion‑MNIST（全连接网络）等数据集上进行实验；

**📈 对比分析**

与原 SISA、近似无学习方法（SalUn、SCRUB、SSD）以及全量重训练比较；结果显示 IDSISA 在忘记质量上提升 14–91%，在性能保持上提升 0–50%（相对），且单次删除的时间成本仅比 SISA 略高或相当，整体加速比远高于全量重训练；

**⚠️ 局限性**

局限包括：1) 需要先计算一次影响函数，若删除频繁且分片策略不更新会产生一定的“分片老化”效应；2) 依赖于梯度稳定性，普通 RNN 等不满足，需改用 LSTM/TCN；3) 在高类别数（CIFAR‑100）和小样本分片时，分片规模不足导致性能指标失真；4) 对于持续多次删除，闭包级联重训练的成本可能升至全 SISA 级别；

---

## 558. Competitive Equilibrium for Linear Chores is CLS-complete

**arXiv ID:** 2611.06589 | [PDF](https://arxiv.org/pdf/2611.06589v1)

**作者:** Qian Liu `[一作]` (University of Hong Kong), Weiqiang Zheng `[通讯]` (Yale University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了在基本的Fisher市场模型中计算竞争均衡（CE）的问题，特别是当代理人具有线性偏好时。我们证明了在可分配的杂务和线性不效用的情况下，计算CE是CLS-hard的，因此是CLS-complete的。这是市场均衡问题的第一个CLS-completeness结果。

**💡 创新点**

创新点在于首次将计算CE的问题证明为CLS-complete，扩展了市场均衡问题的复杂性理论，并为后续研究提供了新的方向。

**🔧 技术方法**

使用了非凸对偶Eisenberg-Gale（EG）程序及其Karush-Kuhn-Tucker（KKT）点的计算方法，结合了计算二次规划KKT点的CLS-complete问题。

**📊 数据集**

使用了构造的Fisher市场实例，其中包含n个代理和m个可分配的杂务，代理具有线性不效用函数和正的收入要求。

**📈 对比分析**

与现有方法相比，本文的方法通过构造市场并利用KKT点的性质，展示了在多项式时间内恢复输入二次规划的ε-KKT点。性能上，计算CE被证明是CLS-hard，表明没有已知的多项式时间算法。

**⚠️ 局限性**

限制在于，虽然证明了计算CE的复杂性，但并未提供有效的多项式时间算法，且该问题的复杂性依赖于某些密码学假设。

---

## 559. On Competing Sellers with Substitutable Goods

**arXiv ID:** 2611.06595 | [PDF](https://arxiv.org/pdf/2611.06595v1)

**作者:** Moshe Babaioff `[一作]` (Hebrew University of Jerusalem), Raghuvansh R. Saxena `[通讯]` (Tata Institute of Fundamental Research)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在完全信息下研究一个主导卖家（Alice）在面对竞争对手（Bob）时如何通过先行承诺并使用随机化机制来最大化收益，比较其在不同买家交互顺序（自由选择、强制先访 Alice 或先访 Bob）下的最优策略与 Bertrand 竞争结果；并对所有最优机制进行完整刻画。

**💡 创新点**

首次给出在任意公共买家价值和商品差异下的 Alice 最优收益的精确表达式，证明即使 Alice 的商品价值远低于 Bob，也能通过随机化获得正收益；提出“等价机制（equalizer mechanisms）”的概念，并证明所有最优机制均可归约为包含无限连续选择项的结构。

**🔧 技术方法**

采用严格的机制设计与 Stackelberg 博弈分析，利用 Myerson 定理推导可实现分配规则、阈值机制、双彩票机制等；构造一系列阈值函数和微分方程来捕捉 Bob 的最优回应，并用等价性与优先级归纳法确定 Alice 的最优策略。

**📊 数据集**

该工作为理论分析，未使用实验或实际数据集；所有结论均通过数学证明与符号推导得出。

**📈 对比分析**

与现有文献（如 Lucier & Saxena 对相同价值情形的定量分析）相比，本论文在不等价值和更广泛买家交互顺序的设置下实现了最优收益的精确表达，并证明了在主导者优势有限时仍可超越 Bertrand 收益，展示了在最优机制下收益提升幅度与价值比的关系。

**⚠️ 局限性**

限制包括：仅考虑单一买家且价值已知的完全信息情形；商品差异仅通过单一价值参数体现，未考虑多品种或分布式买家；对随机化机制的实现未给出具体可行的机制设计算法；并未验证在实际市场中可行性或鲁棒性。

---

## 560. Thermodynamics-Informed Learning Captures Hierarchical Hydrogen-Bond Networks in Deep Eutectic Solvents

**arXiv ID:** 2611.06590 | [PDF](https://arxiv.org/pdf/2611.06590v1)

**作者:** Yanpeng Ye `[一作]` (King Abdullah University of Science and Technology), S. Mani Sarathy `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开发了一种名为TiNN的热力学信息化神经网络，用于预测深共熔质（DES）对多种生物质溶质的溶解度。

**💡 创新点**

创新点在于将三分子层次的氢键网络（HBA–HBD–溶质）与热力学自由能约束相结合，形成层级化表征和温度响应正则化，显著提升了预测的准确性和泛化能力。

**🔧 技术方法**

技术手段包括基于EGNN的3D分子编码器、层级嵌入结构、热力学信息化学习模块（自由能路径和熵-焓一致性损失）以及非理想校正项。

**📊 数据集**

使用的数据集包含938条实验记录，涵盖多种HBA/HBD组合、不同摩尔比、温度、压力以及三类溶质（碳水化合物、木质素衍生物和气体）。

**📈 对比分析**

与COSMO-RS、MLP、XGBoost等基线模型对比，TiNN在测试集上实现R²=0.917、MAE=0.126、RMSE=0.242，且在稀缺条件和新HBA–HBD组合下保持稳健，显著优于传统方法。

**⚠️ 局限性**

局限性包括对实验数据的依赖、未显式建模多组分聚集或动力学反应，以及对气体溶质氢键贡献的忽略，导致在极端条件或复杂聚合体系下预测精度可能下降。

---

## 561. Playing Soccer Like Humans Do: Learning Versatile and Agile Humanoid Soccer via Latent Motion Priors

**arXiv ID:** 2611.06564 | [PDF](https://arxiv.org/pdf/2611.06564v1)

**作者:** Haoyu Shao `[一作]` (University Of Science And Technology Of China), Siyuan Huang `[通讯]` (State Key Laboratory Of General Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种统一的双足机器人足球控制框架，能够在同一策略下完成移动、运球、拦截和射门四种完整的足球技能。

**💡 创新点**

创新点在于通过层次化的潜在运动先验（由人类足球动作捕捉数据蒸馏而来）和转移状态初始化（TSI）实现了自然流畅的技能切换，并且在不使用显式状态机的情况下实现了全局一致的动作空间。

**🔧 技术方法**

主要技术包括：强化学习（PPO）在冻结的潜在动作空间上学习残差控制、VAE式的运动先验蒸馏、物理仿真域随机化、外部视觉状态估计以及潜在空间残差正则化。

**📊 数据集**

使用了公开的 Soccer‑X 运动捕捉数据集（经 GMR 重新映射至 Unitree G1 模型），共约 2.5 小时的人类足球演练序列。

**📈 对比分析**

与 PPO、AMP、ASE、LATENT 等基线进行对比，实验证明本方法在四项技能上均取得最高成功率（Kicking 97.04%，Dribbling 89.88%），同时与人类动作对齐度（MMD）最接近训练与未训练的参考数据，且能够零镜像直接迁移至真实 Unitree G1。

**⚠️ 局限性**

局限性包括：不同技能的 sim‑to‑real 差距不一（尤其是射门速度失真、运球/拦截误差增大）、依赖室内外部运动捕捉系统、仅覆盖四种基础技能、以及使用刚性球模型导致冲击动力学误差。

---

## 562. EPISTEME: Trustworthy Coding Agents Through Tool Call Intent

**arXiv ID:** 2611.06563 | [PDF](https://arxiv.org/pdf/2611.06563v1)

**作者:** Sofia Bobadilla `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了epistemic harness，使每个工具调用必须声明类型与原因，并记录实际效果形成可验证日志；

**💡 创新点**

创新点在于将意图声明强制为工具调用的必备字段，并通过自动检查声明与执行效果的一致性来实现在线防护与离线取证；

**🔧 技术方法**

采用LLM harness、Shell工具包装、文件哈希与diff观察、一致性检查器、Bubblewrap沙箱等技术实现声明验证与日志记录；

**📊 数据集**

实验使用ImpossibleBench基准（oneoff与conflicting两种拆分），评估模型在“不可通过”任务中的作弊暴露情况；

**📈 对比分析**

通过对两大开放模型Qwen3-235B和Mistral Large 3在ImpossibleBench上的实验，曝光率高达90%，表明该方法能有效识别并记录大部分作弊行为；

**⚠️ 局限性**

局限性包括只能检测可观测的文件或环境变化，无法覆盖隐式作弊或网络操作；声明词汇表有限，文件工具声明不完整，且模型仍可能隐蔽作弊，导致部分误行为难以暴露。

---

## 563. LungMend: A Healthy-Only Latent Diffusion Shape Prior for Lung Mask Correction in Severe Tuberculosis CT

**arXiv ID:** 2611.06556 | [PDF](https://arxiv.org/pdf/2611.06556v1)

**作者:** Rezkellah Noureddine Khiat `[一作]`, Catalin Fetita `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了LungMend，一种基于健康肺形状先验的后处理模型，利用骨骼定位来纠正严重结核CT中R231等分割模型漏检的肺区；

**💡 创新点**

创新点在于：①只利用健康肺训练形状先验，完全不依赖病理数据；②通过模拟纹理分割器的典型错误来生成训练样本；③在形状空间使用潜在扩散模型结合骨骼条件，实现对缺失肺区的重建；

**🔧 技术方法**

使用技术包括：3D VAE编码签名距离场、潜在空间的3D U-Net扩散模型、骨骼占据与距离场作为条件、分类器无指导（CFG）以及多样本平均产生不确定性图；

**📊 数据集**

训练集：57例健康后COVID肺部扫描（COVALUNG）；测试集：20例严重结核肺CT（ImageCLEF TB + DeepPulmo标签），并在18例Avicenne医院病例上做定性验证；

**📈 对比分析**

与R231、TotalSegmentator、VISTA3D等基线以及Post-DAE、确定性3D U-Net等后处理方法比较，LungMend在Dice（0.986）、ASD（0.57mm）和HD95（5.5mm）等指标上均优于所有对照组，且病灶召回率达到0.993/0.981；

**⚠️ 局限性**

局限性包括：对真实体积萎缩的肺可能过度补全；依赖于合成错误与真实错误相似度；分辨率仅128^3，近髓质部位细节受限；评估样本有限，误差主要来源于DeepPulmo标签与人工校正切片。

---

## 564. Distributed Consensus, Optimization, and Learning under Communication Constraints

**arXiv ID:** 2611.06524 | [PDF](https://arxiv.org/pdf/2611.06524v1)

**作者:** Zheng Chen `[一作]` (Linköping University), Kaibin Huang `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文综述了分布式一致性、优化和学习的基本理论与实践，重点讨论了网络拓扑、信息流、通信不确定性对分布式学习性能的影响，并给出了多种通信高效方案与未来研究方向。

**💡 创新点**

创新点在于将传统一致性与优化方法与现代通信约束（如包丢失、量化、压缩、AirComp 等）系统性结合，提出了统一框架与评价指标，并总结了现有研究的空白与挑战。

**🔧 技术方法**

使用的技术包括图论与谱理论、加权平均（Mixing Matrix）设计、梯度追踪与方差减小、随机化通信与调度、压缩/量化算法、AirComp 等物理层聚合方法。

**📊 数据集**

由于是综述性论文，未使用特定实验数据集；主要引用了公开的基准实验与仿真结果（如MNIST、CIFAR-10、工业物联网数据）。

**📈 对比分析**

通过文献比较、理论分析与仿真报告，展示了在不同通信架构（FL、去中心化、半去中心化）下，各种通信节省技术对收敛速度、准确率、能耗与延迟的影响，指出量化压缩和 AirComp 能显著降低通信量但需权衡噪声与同步。

**⚠️ 局限性**

限制包括：仅聚焦一阶梯度方法，缺乏对非凸深度网络的严格理论；假设大多基于理想化的网络模型（独立同分布、无时延、无同步误差）；未提供统一实验平台，比较多基于引用结果。

---

## 565. HELIX: HEterogeneous LLM Inference with FleXible and Fine-Grained Partitioning

**arXiv ID:** 2611.06513 | [PDF](https://arxiv.org/pdf/2611.06513v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 566. From Admission to Retention: Predicting Durable Offspring in Quality-Diversity Search

**arXiv ID:** 2611.06522 | [PDF](https://arxiv.org/pdf/2611.06522v1)

**作者:** Corrado Calamita `[一作]` (Bainsa Board Of Ai And Neuroscience Student Association), Riccardo Bertamini `[通讯]` (Bainsa Board Of Ai And Neuroscience Student Association)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在DNS QD算法中使用可在父体产生子之前可用的信号（如世系深度）进行父体选择，以提升归档持久性和多样性。

**💡 创新点**

引入父体产生前可用的信用分配信号，系统性评估其对即时归档录入和长期持久性的关联，并测试基于世系深度的父体选择策略。

**🔧 技术方法**

DNS（Dominated Novelty Search）框架、rank‑softmax父体选择、AUC判别指标、对比实验。

**📊 数据集**

planar-arm 任务、两类机器人步态任务（如Walker、Ant）。

**📈 对比分析**

与均匀父体采样对照；使用QD‑score、覆盖率、平均新颖度等指标；结果显示基于世系深度的偏向既不提升QD‑score，强偏向甚至导致多样性下降。

**⚠️ 局限性**

仅在DNS、单一变异算子和三种标准QD域上验证；发现的关联是预测性而非因果性；归档容量与信息可得性关系仍需进一步探索。

---

## 567. An Interaction-Aware Decision Framework for FDIC-Assisted Distressed Bank Acquisitions: Empirical Evidence from the FDIC Failed-Bank Panel and the SVB-First Citizens Case

**arXiv ID:** 2611.06514 | [PDF](https://arxiv.org/pdf/2611.06514v1)

**作者:** Yousef Nejatbakhsh `[一作]` (Rutgers University), Sina Shokoohyar `[通讯]` (Seton Hall University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 Spider Network Filtering Framework（SNFF），用于基于 FDIC 介入的失败银行收购公告，预测收购方在公告后三天的异常收益，并给出可解释的决策规则。

**💡 创新点**

创新点在于将银行风险、监管保护、流动性、结构性恶化、买方实力与执行难度等五个维度通过四条业务交互规则构建成图灵感的特征工程，使模型既保持解释性，又捕捉条件依赖关系。

**🔧 技术方法**

技术方法包括逻辑回归、岭回归、LASSO、决策树、随机森林、XGBoost、交互规则特征添加；评估采用分层分组 5 折交叉验证、AUC、Brier 分数、校准曲线、置换检验和 Bootstrap 重新拟合；并以决策阈值评估实际收益。

**📊 数据集**

使用了 55 个美国上市收购方的 FDIC 失败银行收购数据（六项财务指标+两项交易条款），并对 90 个更大样本做稳健性检验；数据来源为 FDIC 失效银行清单、BankFind 报告及公告价格。

**📈 对比分析**

模型比较通过同一分层分组 5 折交叉验证得到平均 AUC、Brier 以及决策阈值的实用性；SNFF 逻辑回归的平均 AUC 约 0.594，低于岭回归 0.632，表明加入交互规则对排名与校准的提升有限，基本无显著统计意义。

**⚠️ 局限性**

局限性包括样本量仅 55 组、事件共线性强、异常收益窗口及基准敏感、缺失交易价格与折价信息、规则权重人为设定、仅评估公告期收益而非长期成效，以及模型为预测而非因果。

---

## 568. CRESiST: Self-Enhancing Explorative Policy Learning for Simultaneous Speech Translation

**arXiv ID:** 2611.06501 | [PDF](https://arxiv.org/pdf/2611.06501v1)

**作者:** Chenyang Le `[一作]` (Shanghai Jiao Tong University), Yanmin Qian `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自监督 Actor‑Critic 的交互式读写框架（CRESiST），实现无限长音频的无分段实时语音翻译；

**💡 创新点**

创新点包括：①无外部对齐、无分段的读写交互式框架；②在 400 ms 微块级别上通过自我强化的 Actor‑Critic 学习最优读写策略；③采用逆向蒸馏将 LLM 的高质量生成节奏迁移给 RNN‑T Actor；

**🔧 技术方法**

使用 Qwen‑omni‑7B 大语言模型（含 Router‑Refiner）作为 Critic，Whisper‑medium 作为声学编码器，RNN‑T（带 2D 轴向注意力）作为 Actor，LoRA 微调，逆向蒸馏和奖励设计；

**📊 数据集**

训练数据为三大公开 ASR 语料（Emilia、WenetSpeech、GigaSpeech）并生成伪标签，覆盖中文、英文及 4 种欧盟语言；测试集包括 FLEURS、CoVoST2、MuST‑C 及 ACL‑60/60；

**📈 对比分析**

与 Seamless Streaming、AlignATT、StreamUni、InfiniSST 等基线在 XCOMET‑XXL、LAAL/LongYAAL 指标上比较；CRESiST 在句子级别实现更优的质量‑延迟 Pareto 前沿，在无分段文档级别的连续流场景中仅以 1–2 COMET 分之差逼近带分段的最优；

**⚠️ 局限性**

局限性包括：①模型性能受限于后端 LLM 规模和离线 SFT 数据的分布匹配；②在文档级流式翻译中仅通过静态阈值调节延迟，难以实现动态延迟控制。

---

## 569. Nowcasting Future Performance for Population Pruning in Survival-of-the-Fittest Training

**arXiv ID:** 2611.06512 | [PDF](https://arxiv.org/pdf/2611.06512v1)

**作者:** Rico Städeli `[一作]` (University of St.Gallen), Damian Borth `[通讯]` (University of St.Gallen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并实现了一种基于中间检查点的性能预测与早期淘汰方法（SOTF），在训练过程中提前终止弱模型，从而显著减少计算和能耗。

**💡 创新点**

创新点包括：①使用权重统计与SANE变压器生成的检查点表示来进行“nowcasting”；②将预测结果与学习曲线信息混合，提高短期与长期准确率；③提出SOTF训练机制，以预测准确度动态分配训练资源。

**🔧 技术方法**

技术方法：SANE（序列自编码器）生成权重表示；XGBoost、LSTM、Transformer等回归模型用于未来精度预测；与 Successive Halving、Hyperband、BOHB 等传统自适应资源分配算法进行对比。

**📊 数据集**

实验数据集：九个图像分类模型族，包括 CIFAR‑10、STL‑10、Tiny ImageNet 上的 CNN、宽度变异 ResNet‑18 以及 Vision Transformer 的训练轨迹。

**📈 对比分析**

评估方法：使用训练‑验证‑测试分割，指标包括 RMSE、MAE、R²；在 SOTF 中测量保留模型的最佳精度差（gap）与 oracle top‑K 召回率。结果在 ResNet/ViT 异构族中 gap 从 0.064 降至 0.0017，召回率提升至 0.72，能耗节约可达 90% 以上。

**⚠️ 局限性**

局限性：仅在小型图像分类网络上验证，未扩展到更大模型或其他任务；使用回顾性 SOTF，未考虑实时调度、I/O 开销；缺少多种随机种子复现，且对预测目标与调度参数高度敏感。

---

## 570. Cognitive and Behavioural Effects of Haptic Feedback for ROV Teleoperation in Hazardous Environments

**arXiv ID:** 2611.06689 | [PDF](https://arxiv.org/pdf/2611.06689v1)

**作者:** Joshua Brown `[一作]` (Imperial College London), Michael Jenkin `[通讯]` (York University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在实验池中让新手操作员使用视觉信息和一个包含软性变形与振动反馈的平面触控面板来遥控蓝色Rov2直线前进至目标浮标，检验隐藏侧向水流对操作者的认知负荷和驾驶表现的影响。

**💡 创新点**

首次系统评估了多模态触觉反馈对Rov操作员认知负荷、心理压力、信心与驾驶速度的综合影响，并发现触觉反馈能显著提升自评性能和操作速度，却伴随更高的精神需求与挫败感。

**🔧 技术方法**

采用平面软触感板（粒子真空致密化实现柔软度变化）+方向性振动、NASA TLX问卷、ROV速度记录、蓝色ROV2-Heavy硬件及控制阀门生成可控侧向水流。

**📊 数据集**

实验收集了八名参与者在三种条件（无水流、无触觉、触觉）下的NASA TLX分数、各子量表得分以及ROV平均速度，共计约240条操作数据。

**📈 对比分析**

通过 Jarque‑Bera 正态性检验后使用配对 t 检验比较三种条件，结果显示触觉条件下物理与时间负荷下降、操作速度提升（0.388 m/s vs 0.291 m/s），但精神需求和挫败感上升；整体 NASA TLX 总分从 32.07 降至 31.31，表现提升约 28%。

**⚠️ 局限性**

样本量小、仅涉及新手操作者、实验环境为受控室内水池且水流可预测；缺乏实时认知负荷测量，触觉装置仅为二维反馈，未检验在多维、动态、低能见度的真实海况中的效果。

---

## 571. Don't Replace, Complement: A Detector-Agnostic Subspace Signal for Near-OOD Detection

**arXiv ID:** 2611.06688 | [PDF](https://arxiv.org/pdf/2611.06688v1)

**作者:** Wonseok Shin `[一作]` (Yonsei University), Songkuk Kim `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于稀疏自编码器的低维子空间信号 PLUS，作为后置 OOD 检测的增益补充。

**💡 创新点**

创新点在于把每类最活跃的神经元固定为子空间基底，并用该子空间的白化马氏距离作为 near‑OOD 的判别信号，再以单一固定权重与任意基线得分加权融合。

**🔧 技术方法**

主要技术包括 TopK 稀疏自编码器、QR 正交化构建子空间、子空间马氏距离计算、标准化与加权融合。

**📊 数据集**

使用 ImageNet‑1K 作为 ID 训练集，近 OOD 采用 SSB‑hard 与 NINCO，远 OOD 采用 iNaturalist、Textures 与 OpenImage‑O。

**📈 对比分析**

在 7 种 backbone 上与 18 种现有后置 OOD 检测器（如 MSP、Energy、MDS++ 等）比较，加入 PLUS 后近 OOD AUROC 全部提升，远 OOD 维持或略降，平均提升约 4–8 分。

**⚠️ 局限性**

局限性包括需在 ID 上预训练 SAE、子空间维度与融合权重 β 固定，且对纹理类远 OOD 的提升有限，未在小样本或其他任务上验证。

---

## 572. LifeVal: Evaluating AI Model Performance on Personalized Agentic Consumer Tasks

**arXiv ID:** 2611.06687 | [PDF](https://arxiv.org/pdf/2611.06687v1)

**作者:** Jenny Yang `[一作]` (OpenAI), Tejal Patwardhan `[通讯]` (OpenAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

建立 LifeVal 基准，评估 AI 在高价值、个性化、代理式消费者任务上的表现。

**💡 创新点**

创新点在于结合真实消费者愿付意愿调查、个性化 ChatGPT 对话、强化学习环境与客观打分的任务规范，提供长期价值导向的评测框架。

**🔧 技术方法**

使用多轮人机质量控制、GPT‑5.x/6 Astra 模型、自动化评分器、强化学习环境和浏览器工具进行任务执行与评估。

**📊 数据集**

使用 ATUS、MTUS 时间使用调查、3,837 名受访者的愿付意愿问卷、以及任务作者的真实 ChatGPT 对话和参考文件。

**📈 对比分析**

通过自动化评分器对 169 个任务进行评测，GPT‑5.2 至 GPT‑6 Astra 随时间逐步提升；GPT‑6 Astra 最高，但仍存在信息不正确、决策边界等失误。

**⚠️ 局限性**

局限性包括受美国、英文问卷限制、对调查假设与自我报告的依赖、时间敏感任务的漂移以及对历史调查数据的局限。

---

## 573. Polynomial-Time Differentially Private Synthetic Graph Release with Nearly-Linear Additive Error

**arXiv ID:** 2611.06670 | [PDF](https://arxiv.org/pdf/2611.06670v1)

**作者:** Marek Eliáš `[一作]` (Bocconi University), Zongrui Zou `[通讯]` (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种多项式时间的 (ε,δ)-边差分隐私算法，用于生成近似保持所有割值的合成图。

**💡 创新点**

通过连续乘法权重框架、log‑det 正则化 SDP 与成功计费隐私分析，突破了先前 n^13/12 的加法误差瓶颈，实现接近线性加法误差。

**🔧 技术方法**

结合 PREM 框架、log‑det 正则化的 Max‑Cut SDP、Gaussian 协方差采样、以及隐私计费与成功计费组合的多项式时间 SDP 求解。

**📊 数据集**

论文主要针对理论分析，实验验证使用合成随机图和全图，未使用具体现实数据集。

**📈 对比分析**

与现有最优 O(n^13/12) 结果相比，误差降至 O(n)（加上 polylog 因子），并在相关下游任务（相关聚类、最大割、层次聚类等）实现接近最优乘法误差与线性加法误差。

**⚠️ 局限性**

仍存在 polylog 乘数的开销，且实现依赖高精度 SDP 求解，对极大规模图的实际运行时间与常数项仍较大。

---

## 574. To lift or not to lift: Operator inference for nonlinear PDEs with boundary control

**arXiv ID:** 2611.06665 | [PDF](https://arxiv.org/pdf/2611.06665v1)

**作者:** Zhuoyuan Wang `[一作]` (Carnegie Mellon University), Saviz Mowlavi `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了一种边界一致的提升型Operator Inference框架，用于处理带时变非齐次边界条件的非线性PDE。

**💡 创新点**

创新点是将传统的提升（lifting）方法与非侵入式Operator Inference结合，得到包含双线性状态-输入和二次输入-输入项的新ROM结构，并保证边界条件被显式满足。

**🔧 技术方法**

使用了POD降维、投影、数据驱动的最小二乘回归（OpInf）以及正则化技术，并在热方程、Burgers方程和托盘驱动腔流上进行实验。

**📊 数据集**

使用高阶数值模拟得到的热方程、Burgers方程和二维托盘驱动腔流的时间序列数据（Chebyshev网格或有限体积网格），共计数千至上万帧。

**📈 对比分析**

与原始OpInf、投影、侵入式Galerkin等基线方法比较，提升版在边界误差、预测精度、求解稳定性和计算效率上普遍优于基线，尤其在较大POD维度时提升显著。

**⚠️ 局限性**

局限性包括：对某些问题（如Navier–Stokes）提升并未显著提升在线速度；在高维模型时仍出现求解失败；以及需要预先可用的提升函数，适用范围受限。

---

## 575. CardioDrift: Automatic Cardiac Motion Tracking on a 3D Atlas Mesh via Shape-aware Coherent Point Drift with Slice Shift Correction

**arXiv ID:** 2611.06664 | [PDF](https://arxiv.org/pdf/2611.06664v1)

**作者:** Reza Akbari Movahed `[一作]` (University of Glasgow), Ali Gooya `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种完全自动化的基于3D心脏atlas网格的心脏运动追踪框架CardioDrift，只需Cine CMR短轴分割掩模即可得到全心室的全球对应点运动。

**💡 创新点**

创新点包括：①在单一概率模型中引入切片平移校正，纠正切片间对齐误差；②将Coherent Point Drift（CPD）泛化为形状感知的联合注册，分别对左心室内壁、外壁和右心室进行区域加权；③完全在点集域中完成atlas适配和非刚性变形，避免图像域运动场的插值衰减和手动初始化。

**🔧 技术方法**

采用的技术有：Coherent Point Drift（CPD）EM优化、Gaussian核形变场、形状加权区域约束、切片平移估计、点集与atlas表面直接配准、marching squares提取分割轮廓、基于DICOM仿射变换的物理坐标转换。

**📊 数据集**

使用的公开数据集包括：UK Biobank（1301名受试者的全心室Cine CMR短轴分割）和M&M（134名受试者，包含多厂商、多中心的Cine CMR，ED/ES参考掩模），两者均使用相同的bi-ventricular atlas。

**📈 对比分析**

方法与ICP、CPD、PCRNet、PointNetLK、CPD-Net、DeepMesh等基线进行对比；在刚性配准阶段CardioDrift在HD95、MSD和DSC三项指标上均优于所有基线；在非刚性阶段同样获得最低的HD95、MSD和最高的DSC，尤其心肌区域DSC提升显著；切片平移校正与形状感知机制的消融实验进一步验证其显著性能提升。

**⚠️ 局限性**

局限性包括：①计算成本高，需在每帧单独求解EM；②对分割质量高度敏感，误分割直接影响运动结果；③M&M评估仅在ED/ES有参考，缺乏对中间帧时间一致性的验证；④缺乏对时间平滑约束的建模；⑤缺乏实时性与大规模临床部署的验证。

---

## 576. Tight bounds for fractional and randomized truthful scheduling

**arXiv ID:** 2611.06642 | [PDF](https://arxiv.org/pdf/2611.06642v1)

**作者:** Georgios Anastasiadis `[一作]` (Aristotle University of Thessaloniki), Conrad Schecker `[通讯]` (Goethe University Frankfurt)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在不相干机器上对真实度量的调度问题，证明随机化无法获得子线性逼近，并给出了最优近似比的上界。

**💡 创新点**

首次给出真实机制的线性下界 (n+1)/2，并构造了近似比为 n/2+o(n) 的普适真实机制，完成了 Nisan–Ronen 预测的最终答案。

**🔧 技术方法**

利用图实例、平行边结构、零类型不相关性、两任务分离（拉普拉斯变换）以及固定星形保留的关键引理，对真诚机制的分配函数进行严格的分析。

**📊 数据集**

无实际数据集，全部为理论分析与证明。

**📈 对比分析**

通过构造对抗实例与上界机制比较，证明下界与上界相匹配，随机化机制的近似比仅比确定性最优 (n) 提升一倍。

**⚠️ 局限性**

局限在于仅适用于不相干机器模型，未考虑多任务、约束或随机实例的实验验证；对实际工业调度的适用性仍待评估。

---

## 577. Interactive World Simulation with Cosmos for Physical AI

**arXiv ID:** 2611.06627 | [PDF](https://arxiv.org/pdf/2611.06627v1)

**作者:** NVIDIA `[一作]`, Rick Zhuang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

将 Cosmos 3 世界基础模型通过后训练流程转化为可交互的模拟器，支持基于动作和空间控制的实时视觉生成，涵盖机器人导航/操纵、视频转移以及多视角 AV 场景。

**💡 创新点**

提出共享的后训练框架：因果训练 + 少步一致性蒸馏 + 分布匹配蒸馏 + 状态化流式推理，使大型多模态基础模型能够在实时交互中直接使用；同时设计块因果注意、教师强迫、自动回归分布匹配等技术，实现在多域下的高效、可控生成。

**🔧 技术方法**

技术包括 Cosmos 3（Mixture-of-Transformers、RoPE、多模态 VAE）、视频递归变分自编码器、块因果注意、教师强迫、离散时间一致性蒸馏、分布匹配蒸馏、K/V 缓存流式推理、上下文并行 GPU、CUDA 图/量化加速、异步解码等。

**📊 数据集**

使用的主要数据集：混合相机+机器人+人类视频、仿真场景深度/分割/边缘/模糊控制视频、RGB‑LiDAR 多视角 AV 数据；并在 Cosmos 3 预训练权重的基础上进行后训练。

**📈 对比分析**

通过与未转换的 Cosmos 3 以及单步/无蒸馏模型对比，评估视觉质量（PSNR/LPIPS）、控制一致性、实时性（FPS、RTF≤1）和传感器延迟；实验表明 4 步蒸馏学生可达 30+ FPS 并保持与基线相当或更优的视觉质量，且在机器人、视频转移和 AV 三种场景下均实现实时交互。

**⚠️ 局限性**

局限性包括：训练与推理窗口不完全一致导致历史利用偏差；K/V 缓存限制长时间记忆；块因果注意需手动划分时间块，影响细粒度控制；对复杂物理细节的建模仍不足；闭环策略验证仍依赖外部实验，尚未全面证明在真实物理交互中的鲁棒性。

---

## 578. Deterministic Connectivity Sketches Under Edge Deletions

**arXiv ID:** 2611.06626 | [PDF](https://arxiv.org/pdf/2611.06626v1)

**作者:** Sanjeev Khanna `[一作]` (New York University), Madhu Sudan `[通讯]` (Harvard University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在任意合法边删除（即 Bob 的图 G_B⊆G_A）下，Alice 仅发送一次信息就能让 Bob 判定 G_A−G_B 是否连通的确定性一向通信问题，并给出了其通信复杂度的近似范围；随后将该结果转化为无容错连通性预言机、单遍动态流算法等数据结构的上界；同时提出了匹配的下界，说明连通性判定在确定性设置中相比随机化或边恢复任务仍有显著的多项式开销。

**💡 创新点**

关键创新包括：①首次证明连通性判定可用 O(n^{7/4}) 位的确定性编码实现，远低于恢复单条边所需的 Ω(n^2) 位；②利用递归子采样与稀疏恢复相结合的确定性子集大小估计 sketch；③构造一类一侧剪切 sketch 及顶点覆盖检测 sketch，证明其在判断跨切大小时拥有“巨量阈值”；④通过正定矩阵与向量色数理论给出 Ω̃(n^{3/2}) 位的下界；⑤进一步推广至容错连通性预言机与受限更新量的单遍动态流。

**🔧 技术方法**

主要技术手段包括：确定性子集大小估计 sketch（基于多次随机子采样的 derandomization）、稀疏恢复 sketch、one‑sided cut sketch、顶点覆盖检测 sketch、正定矩阵（Lovász）证明、递归分解与子采样层次、以及在动态流中对更新量的缓冲与压缩。

**📊 数据集**

研究为纯理论工作，未使用实际数据集；所有结果均在无输入分布假设的最坏情况上给出。

**📈 对比分析**

本文与现有研究的比较主要体现在理论复杂度层面：随机化线性 sketch 需要 O(n) 位即可恢复连通性树，而本工作在确定性场景下给出 O(n^{7/4}) 上界与 Ω̃(n^{3/2}) 下界；在容错预言机方面，传统 O(n√f) 上界被提升至 O(n^{7/4})（当 f=n^2 时）；在单遍动态流中，得到了 O(n^{15/8+3/8}) 位空间的确定性算法，显著低于 naïve O(n^2) 存储。

**⚠️ 局限性**

目前存在的局限包括：①通信复杂度上界与下界之间仍有 n^{1/4} 的多项式差距；②对线性 sketch 的确定性下界尚未达到 Ω(n^2) 的强度；③动态流结果仅在更新次数受限（每条边更新 ≤ n^α）时有效，对任意长流缺乏匹配的下界；④论文未给出实验验证，主要关注理论证明。

---

## 579. Dynamic Shape Control of Soft Trunk Robots

**arXiv ID:** 2611.06625 | [PDF](https://arxiv.org/pdf/2611.06625v1)

**作者:** Hehui Zheng `[一作]` (ETH Zurich), Robert K. Katzschmann `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过学习全身的 adiabatic spectral submanifold (aSSM) 模型，并结合模型预测控制 (MPC) 实现软颈式机器人全身形状的实时闭环控制。

**💡 创新点**

创新点在于将 aSSM 低维化模型从单点末端追踪扩展到全身形状控制，并通过实验验证其优越性；同时改进了腔管设计以显著降低相互耦合。

**🔧 技术方法**

采用了 aSSM 识别、延迟嵌入、SVD、多项式拟合、MPC、以及与线性数据驱动方法 (ERA、DMDc、LOpInf) 和 PPO 强化学习的对比。

**📊 数据集**

使用了约 1.79×10^6 个采样点的硬件实验数据（5 小时的姿态记录）来训练和验证模型。

**📈 对比分析**

与三种线性基线以及 PPO 对比，aSSM‑MPC 在闭环形状跟踪、动态形状追踪以及人机交互任务中，均比基线降低 8–10 倍误差，并成功完成了 25 秒的网球平衡任务。

**⚠️ 局限性**

局限在于软机器人动态速度受绳索驱动限制，且 aSSM 仅在已收集的工作空间内有效；对更大范围、更多自由度的软体仍需进一步验证。

---

## 580. Constant-Time Dynamic Edge Coloring via Dynamic Splitting and LLL

**arXiv ID:** 2611.06611 | [PDF](https://arxiv.org/pdf/2611.06611v1)

**作者:** Mohsen Ghaffari `[一作]` (Massachusetts Institute of Technology), Sijin Peng `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种随机化动态算法，在任意时间点维护一个近最优的 (1+ϵ)Δ 边着色，更新一次（插入或删除一条边）平均期望时间为 O(1/ϵ)，并且对每个更新的重着色次数也为 O(1/ϵ)。

**💡 创新点**

创新点主要有三点：
① 移除了先前工作中对图规模 n 的依赖，只需 Δ ≥ Θ(1/ϵ²) 的最大度数阈值；
② 引入了动态 Lovász Local Lemma（LLL）求解器，能够在每次更新后保持 LLL 约束的满足，更新时间仅与局部度数有关；
③ 设计了“自底向上”分割（bottom‑up splitting）技术，利用图碎片化（shattering）和局部搜索，避免了传统分层算法中必须逐层检查的对数级开销。

**🔧 技术方法**

技术手段包括：
- Ródl 的“nibble”方法结合分布式 LLL，形成迭代的部分着色。
- 动态 LLL 求解器基于 Moser‑Tardos 递归重采样的并行化版本，利用 witness‑tree 及局部性分析保证期望更新时间。
- 动态分割算法在每层通过随机方向/类别分配实现图碎片化，并使用局部冻结与图 shattering 证明分裂后每层的度数得到严格控制。
- 自底向上修复利用“种子失败”集的递归更新与隐式维护的路由路径，显著降低了每次更新对递归树深度的依赖。

**📊 数据集**

论文属于理论计算机科学，实验部分未使用实际数据集，所有结果均为理论证明与期望时间分析。

**📈 对比分析**

与先前的 BCPS 算法相比，BCPS 需要 Δ ≥ (log n / ϵ⁴)^{Θ(log(1/ϵ)/ϵ)} 的最大度数，更新时间为 O(1/ϵ) 但依赖于 n；本工作把度数阈值降低到 Θ(1/ϵ²) 并去掉了对 n 的依赖，更新时间保持 O(1/ϵ)。此外，通过自底向上分割进一步将期望时间降低到 O(1)（对 ϵ 仍保持 1/ϵ 的系数）。

**⚠️ 局限性**

局限性包括：
- 仍需要 Δ ≥ Θ(1/ϵ²) 的最小度数限制；
- 算法仅在“无记忆”（oblivious）对手下可保证期望时间，对自适应攻击者的保证有限；
- 隐藏常数可能较大，实际实现的开销尚未评估；
- 依赖高质量的随机数生成与并行化实现，理论上可行但工程实现挑战较大。

---

## 581. Beyond accuracy: what AI assurance needs to establish in public services

**arXiv ID:** 2611.06655 | [PDF](https://arxiv.org/pdf/2611.06655v1)

**作者:** Louise Yorke `[一作]` `[通讯]` (University College London), Louise Yorke (University College London)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 GPT‑4o 在英国首相问答中的主题分类进行实验，评估稳定性、准确性和对结论的影响，并对英国和欧盟 AI 保障文件进行比较。

**💡 创新点**

提出通过配置比较、同义过滤、评分提取和人工复核比例评估方法揭示同一模型在不同配置下仍产生不同标签的“预测多样性”，并指出保障文件对检查的细节缺失。

**🔧 技术方法**

使用 GPT‑4o 大语言模型，构建多种配置并执行多次重复运行。

**📊 数据集**

使用英国 Comparative Agendas Project 中 1,200 条首相提问样本。

**📈 对比分析**

通过与人类编码标签对比和配置间一致性评估，准确率约 68‑70%，但同一配置下 5% 的重复运行差异和 24% 的标签差异；不同配置在人工复核阈值下导致 0.9‑41% 的输出被标记。

**⚠️ 局限性**

局限在于仅使用单一模型、有限配置、未对最终答案置信度进行验证、未观察人工复核结果、仅评估一次任务且文档样本未覆盖所有国家级指导。

---

## 582. Human Motion Prediction via Continual Prior Compensation

**arXiv ID:** 2611.06637 | [PDF](https://arxiv.org/pdf/2611.06637v1)

**作者:** Jianwei Tang `[一作]` (Sun Yat-sen University), Jianhuang Lai `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了连续先验补偿（CPC）与其改进版CPC++，通过将未来动作序列分段并采用多阶段训练，逐步学习短期到长期预测，显著缓解了长时预测对短时预测的干扰，并通过可学习的先验补偿因子（PCF）和细粒度先验补偿因子（FGPCF）主动补偿多阶段训练中的知识遗忘。

**💡 创新点**

创新点主要有：①将人类动作预测拆解为子任务并按时间顺序逐步训练，实现了对不同时间尺度的专门学习；②设计可学习的PCF与FGPCF，用随机变量量化并补偿前一阶段遗忘的先验知识；③在理论上给出了可优化的目标上界，并证明PCF/FGPCF可与模型参数同步学习，形成完整的优化框架。

**🔧 技术方法**

技术方面使用了多阶段（continual）训练、可学习随机变量（PCF/FGPCF）、上界优化策略、均方误差损失；在骨干网络上集成了注意力模块（PLEM）来估计FGPCF；实验中采用了GCN/Transformer/MLP等多种主干模型。

**📊 数据集**

实验数据集包括 Human3.6M、CMU‑MoCap 以及 3DPW，三者均为公开的人类动作捕捉数据集，覆盖室内外多种动作。

**📈 对比分析**

与现有 SOTA 主干（PGBIG、siMLPe、MotionMixer、LTD）在三大数据集上对比，CPC 与 CPC++ 均在 MPJPE 上获得显著下降，特别是在 560‑1000 ms 长时预测时提升更明显；CPC++ 在多数指标上略优于 CPC。

**⚠️ 局限性**

局限性包括：①对 FGPCF 的训练需要额外参数和计算，导致在复杂模型或小数据集上提升有限；②理论上对 P(Z_i|…)≥½ 的假设限制，虽然实验中大部分样本满足但仍存在小比例不满足的情况；③多阶段训练虽然提升性能但会略微增加训练时间；④在某些数据集上（如 3DPW）CPC 与 CPC++ 的差距不大，表明在极端复杂场景下仍有改进空间。

---

## 583. From Certified Pairs to a Certified Hub: Reasoner-Checked Crosswalks for the Information Exchange Standard

**arXiv ID:** 2611.06628 | [PDF](https://arxiv.org/pdf/2611.06628v1)

**作者:** Fabio Rovai `[一作]` `[通讯]` (Tesseract Academy (Kampakis and Co Ltd)), Fabio Rovai (Tesseract Academy (Kampakis and Co Ltd))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了一套从信息交换标准（IES）到13个领域本体的跨walk认证流程，并评估了该流程在建筑、能源、地理、传感、时间、数据目录、BFO等七大行业组中的有效性。

**💡 创新点**

创新点在于将逻辑一致性（coherence）与保守性（conservativity）测试与人类审查、SKOS强度降级等步骤结合，形成可重复、可解释的跨walk验证框架，并首次在IES与已存在的建筑本体（BOT、Brick、SAREF4BLDG）之间生成并发布了完整的认证跨walk。

**🔧 技术方法**

技术上采用了开放源代码的匹配引擎（Open Ontologies）、SKOS/SSSOM序列化、HermiT推理器进行一致性和保守性检测，并实现了自动化的匹配、审查、认证、推理和序列化五阶段流水线。

**📊 数据集**

使用的数据集包括IES的三种配置（核心、建筑、紧急管理）以及目标本体的最新版本（BOT 0.3.2、Brick 1.4.2、SAREF4BLDG 1.1.2、Open Energy Ontology 2.14.0、GeoSPARQL 1.1、PROV-O 3.0、SOSA/SSN 2017、OWL-Time 2017、BFO 2020、HQDM 1.0等，此外还评估了一个小型的建筑改造扩展模块。

**📈 对比分析**

方法通过在每对本体之间执行四种跨walk（原始、认证后、审查后、审查+认证后）并使用HermiT计时，显示认证阶段能消除所有保守性违规，审查阶段能消除同义词导致的不一致；结合两者后，11/12个重叠对在一致性与保守性上通过；在整合为“hub”时，仅需几秒钟即可完成推理，说明其性能良好。

**⚠️ 局限性**

局限性包括：缺乏对目标本体中未定义或不可辨别的类的处理（如HQDM中不可满足类导致的冲突），缺少对同义词与语义差异的语义校验，部分领域（能源、BFO）匹配稀缺，且认证流程仅检测逻辑一致性与保守性，并不保证语义正确性。

---

## 584. Projected and Reflected Sliced Wasserstein Kernels for Persistence Diagrams

**arXiv ID:** 2611.06619 | [PDF](https://arxiv.org/pdf/2611.06619v1)

**作者:** Julián Uribe Castañeda `[一作]` (Pontificia Universidad Católica de Chile), Mircea Petrache `[通讯]` (Pontificia Universidad Católica de Chile)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出两种基于投影测度的 Wasserstein 形态距离 _p 与 _p，并证明它们与传统的 d_p 距离在有限维图集上构成等价度量；在此基础上分析了距离的条件负定性（CND），发现 _1 与 _1 为 CND，进而构造了正定核 k_1 与 k_1；随后在“轨道识别”数据集上对比了这些核与其它已知核，展示了其在支持向量机（SVM）中的优越性能；

**💡 创新点**

创新点在于：①给出一种全新的基于投影测度的 Wasserstein 距离；②证明其在有限图集上与 d_p 的两侧界定，并给出最优常数；③首次证明 _1 与 _1 的 CND 性质，并利用 Schoenberg 定理构造可正定核；④在实际数据集上验证这些核优于传统核，并揭示 p>1 时 CND 失效导致核非正定的原因。

**🔧 技术方法**

主要技术包括：对投影测度与 Wasserstein 距离的理论分析、几何投影构造、条件负定性判定、Schoenberg 定理的应用、凸优化求解分配问题、支持向量机（SVM）与核方法的实验评估。

**📊 数据集**

使用的实验数据集为 Orbit Recognition Benchmark，即由 5 个不同参数 r 生成的 500 个点云（每个点云 1000 个采样点），其持久性图通过 Vietoris–Rips 构造得到；此外，还使用了 500 对随机有限图集做度量等价性验证，以及 150 组随机图集和对抗搜索来检验 CND 性质。

**📈 对比分析**

实验中将 k_1 与 k_1 与 5 种其他核（如基于 d_p、_p 的高斯核、直方图核等）在 SVM 分类任务中比较；结果显示 k_1 与 k_1 在 10 次交叉验证中均获得最高准确率，尤其在数据集较大、噪声较少时优势更明显；此外对比实验表明，非正定的 k_2 与 k_2 在某些配置下仍能取得接近的准确率，但其 Gram 矩阵出现负特征值，说明缺乏理论保证。

**⚠️ 局限性**

局限性包括：①对 _p 与 _p 与 d_p 的下界常数仍远不够紧凑，实际数据中往往比理论下界大得多；②仅证明 _1 与 _1 的 CND 性质，p>1 时已失效，导致无法构造更一般的正定核；③实验仅覆盖轨道识别任务，缺乏对其它拓扑数据分析场景的验证；④对于更高维度或稀疏图集，投影测度的计算成本和数值稳定性可能成为瓶颈。

---

## 585. Dimension-Free Private Mean Estimation is Impossible Without the Covariance

**arXiv ID:** 2611.06615 | [PDF](https://arxiv.org/pdf/2611.06615v1)

**作者:** Renaud Gaucher `[一作]`, Lydia Zakynthinou `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究在协方差未知时，差分隐私下高维高斯均值估计的极限，给出了新的误差下界，证明了无法获得与已知协方差时相同的维度无关误差或样本复杂度。

**💡 创新点**

创新点在于通过构造联合先验并将均值估计问题转化为采样/选择子问题，利用指纹攻击和会员推断技术得到最优下界；首次证明在协方差未知时，误差需要乘以√(d/n)或√(d)/n^{3/2} 的维度相关放大因子；并给出在 n<d-2 时 Mahalanobis 误差不可有限的结果。

**🔧 技术方法**

主要技术包括：先验分布构造（随机投影矩阵、缩放逆卡方分布）、隐私与采样/选择的归约、指纹编码与追踪攻击、会员推断攻击、随机矩阵理论、概率界与伯努利/χ^2 分布的尾部分析。

**📊 数据集**

无实验数据集，全部为理论分析与证明；使用的“数据”为从构造先验中采样的高斯样本。

**📈 对比分析**

对比已知协方差下的最优无维度误差 (Θ̃(√(tr(Σ)/n)+…))，论文表明未知协方差时误差至少高出 √(d/n) 或 √(d)/n^{3/2} 的因子；相应的样本复杂度也必须更大。上界部分来自已有工作（FriendlyCore 框架等），与下界紧贴，证明了下界的最优性。

**⚠️ 局限性**

局限性：仅给出下界，在某些参数范围内缺少匹配的上界；对特殊协方差结构（如低秩、对角）仍需进一步探究；论文的隐私模型为近似 DP，未讨论纯 DP 情况；理论证明依赖于高维随机矩阵假设，可能不适用于所有实际数据分布。

---

## 586. AgentTrustCI: Build-Time Policy, Grounding, and Trajectory Tests for Java LLM Agents

**arXiv ID:** 2611.06674 | [PDF](https://arxiv.org/pdf/2611.06674v1)

**作者:** Sachin Gupta `[一作]` `[通讯]` (Independent Researcher), Sachin Gupta (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 AgentTrustCI，一个在 Java/JUnit 5 中实现的框架，用于在 CI 流水线中基于代理执行记录的政策、答案归属和轨迹三层检查来验证 LLM 代理的行为；

**💡 创新点**

创新点在于将行为契约以声明式、可重复的回放测试形式嵌入现有 Java 发布流程，并通过三层验证（policy、grounding、trajectory）实现细粒度覆盖；

**🔧 技术方法**

技术包括 Java 17、JUnit 5、SnakeYAML、可回放的 GoldenCase 记录、基于规则的词法归属检查、顺序与重复调用检测以及可扩展的 LLM 判断接口；

**📊 数据集**

使用了 55 个人工合成的测试案例，涵盖 7 个企业域（支持工单、目录、库存等），包含 30 个注入错误、25 个通行控制和 12 个关键错误；

**📈 对比分析**

通过对单层与全链路配置的精确计数评估，发现全链路在所有 55 个案例中 100% 正确分类，单层仅覆盖约 64%，并在性能测试中每条评估平均仅 3.4 µs（95% 分位 6.3 µs）；

**⚠️ 局限性**

局限性包括：仅针对人工合成数据，未验证真实生产环境的误报率和漏报率；词法归属检查无法捕捉语义误差；轨迹检查仅限线性序列，无法处理并发或数据依赖；缺乏实时 LLM 判别器；对工具输出假设可信，未处理工具误报。

---

## 587. Dynamical State-Space Diffusion for Spatiotemporal Prediction

**arXiv ID:** 2611.06652 | [PDF](https://arxiv.org/pdf/2611.06652v1)

**作者:** Woojun Lee `[一作]` (Seoul National University), Pilsung Kang `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了DSSD，一种将时间相关性迁移到扩散前向过程、保持去噪网络不变的动态状态空间扩散模型；

**💡 创新点**

通过可学习的对角状态空间算子生成多尺度、振荡相关的时间噪声，并用其静态协方差的Cholesky因子构造因果、可逆、单位方差的时间混合，从而在扩散前向过程与逆向采样中引入学习式时间动力学；

**🔧 技术方法**

利用结构化状态空间模型（SSM）、Kalman创新实现、Cholesky分解、DDIM采样，以及联合训练去噪器与动态参数；

**📊 数据集**

在科学场景数据集（SEVIR、Turbulence）和机器人视频数据集（BAIR、RoboNet）上进行实验；

**📈 对比分析**

与标准DPM和DyDiff在同一去噪器下对比，评估CRPS、CSI、FVD、RMSE等指标，DSSD在所有数据集和指标上均优于两基线，尤其在概率预测和时序一致性方面表现突出；

**⚠️ 局限性**

仍受限于参数规模仅18个学习参数、有限训练步骤和采样步数，对长序列的内存效率和理论上对所有数据提升有限步采样的通用性仍需进一步研究。

---

## 588. From Reconstruction to Imputation: Test-Time Optimization for Missing Time-Series Channels

**arXiv ID:** 2611.06612 | [PDF](https://arxiv.org/pdf/2611.06612v1)

**作者:** Björn Ludwig `[一作]` (Helmut Schmidt University), Oliver Niggemann `[通讯]` (Helmut Schmidt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种测试时优化（Test‑Time Optimization，TTO）方法，将已冻结的序列重构器在不重新训练权重的情况下转化为缺失值填补器。该方法仅对缺失位置进行输入优化，通过在冻结模型上最小化观测残差来实现自一致性。

**💡 创新点**

创新点在于：①无需在训练时使用人工掩码或全通道丢弃；②通过观测自一致性目标以及缺失→观测的交叉雅可比分析，确定何时该优化是有信息的；③引入可选的时间平滑先验来补偿弱耦合场景；④把重构器的学习能力与在线优化相结合，构成一种条件的推理时适配器。

**🔧 技术方法**

技术上使用：冻结的自编码器（多种骨干网络如LSTM、MTAD‑GAT、TCN‑AE、TransformerAE），在测试时对缺失值进行梯度下降（Adam），利用反向传播得到观测残差梯度；可选的平滑先验为一阶时间差正则；跨通道耦合通过交叉雅可比矩阵量化。

**📊 数据集**

实验采用五个真实世界多变量时间序列数据集：BeijingAir、ItalyAir、ETTh1、Electricity、PeMS，全部采用标准时间拆分与窗口化。每个测试窗口随机遮挡一个完整通道进行评估。

**📈 对比分析**

与多种mask‑aware填补器（BRITS、SAITS、iTransformer、TimesNet、FreTS）以及无掩码的MeanFill进行对比。TTO在19/20的重构器–数据集组合上相较单次前向推理降低MAE，宏观平均下降约12.9%。在ETTh1和Electricity上，TTO的MAE甚至优于所有mask‑aware基线；在BeijingAir、ItalyAir和PeMS上则略逊于最优mask‑aware模型。需要注意的是，TTO的推理时间比一次前向推理慢6.7–174.7倍。

**⚠️ 局限性**

局限性包括：①需要重构器对缺失通道有足够的跨通道耦合（交叉雅可比不为零或足够秩），否则优化信息不足；②当耦合弱或秩缺失时，可能出现多重自一致解，导致恢复不稳定；③TTO需要多次前向/反向传播，计算成本高；④方法在多通道缺失、混合缺失模式下尚未验证。

---

## 589. When to Ask for Help: Uncertainty-Triggered Aerial Refinement for Off-Road Navigation

**arXiv ID:** 2611.06702 | [PDF](https://arxiv.org/pdf/2611.06702v1)

**作者:** Pranay Samineni `[一作]`, Ufuk Topcu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于规划器条件的空地协同辅助方法，利用航天机获取针对性地图信息，以解决无人地面车辆在荒野导航中的不确定性导致的性能下降；

**💡 创新点**

创新点在于通过对当前MPPI规划轨迹在不同地图假设下的计分进行反事实评估，仅在解析不确定性能显著提升导航得分时才请求航天机采集信息，实现了对无人机资源的精准调度；

**🔧 技术方法**

核心技术包括：MPPI局部规划器、地图层融合与不确定性表示、反事实评分机制、基于阈值的辅助决策、无人机图像处理与地图注册、以及多平台异步通信；

**📊 数据集**

使用的主要数据集为：USGS 3D Elevation（用于Gazebo仿真地形）、真实场地的LiDAR/光学传感器数据、以及在实地试验中收集的野外道路轨迹（约2公里）；

**📈 对比分析**

与基线（仅地面感知）、贪婪持续航测和先行全路航测等四种策略对比，实验显示在180条仿真试验中目标达成率98.9%，平均无人机飞行时间相比贪婪策略降低87.7%，每秒无人机飞行时间可节省约2.94秒的地面导航时间；实地试验覆盖2公里，展示了多次解算不确定性并恢复导航的能力；

**⚠️ 局限性**

局限性包括：依赖可靠的无线通信与高质量图像注册，硬件可靠性与热管理问题，未考虑长距离、持续高负载的实际场景，且反事实评估对高维连续地图不适用，需要更通用的处理方法。

---

## 590. You Know What You Will Like: People Predict Their Own Ratings Better Than Recommender Systems

**arXiv ID:** 2611.06696 | [PDF](https://arxiv.org/pdf/2611.06696v1)

**作者:** Zihan Chen `[一作]`, Di Zhu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 MovieLens 会员在未观看电影时的自我预测（beliefs）与平台推荐器预测进行对比，评估自我预测在多种评价指标上的表现。

**💡 创新点**

证明会员对自身未来评价的预测优于平台协同过滤推荐，并且能显著提升推荐器的预测精度（平均MSE下降约25%），从而调和了“群体视角”与“分散知识”两种理论。

**🔧 技术方法**

采用协同过滤（矩阵分解、项目邻域）、基准加权模型以及集成堆叠回归，并使用相关系数、均方误差、顺序准确率等统计量进行评估。

**📊 数据集**

使用 MovieLens Beliefs 数据集（2023–2024 年 1.8 万对成员–电影样本，包含用户猜测、平台预测、后续评分）。

**📈 对比分析**

在同一成员–电影对上比较四种预测方式，结果显示：自我预测相关系数 0.71，MSE 0.74；最优协同过滤 0.59/1.05；平台预测 0.53/1.18；堆叠模型 0.60/1.10；自我预测在排序准确率上比平台高 10–20% 点，且加入自我预测后，所有模型的MSE均下降 20–30%。

**⚠️ 局限性**

局限性包括：只分析回答且随后评分的样本，未进行随机实验验证主动询问是否提升最终推荐效果；样本来自热衷评分的电影爱好者，结果可能不适用于更一般用户；使用的评估方法依赖于后续评分的可观测性，存在选择偏差。

---

## 591. The Online $k$-Taxi Problem Is Competitive on All Metrics

**arXiv ID:** 2611.06684 | [PDF](https://arxiv.org/pdf/2611.06684v1)

**作者:** Christian Coester `[一作]` (University of Oxford), Tze-Yang Poon `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出了一种在任意度量空间上对在线k‑Taxi问题的随机化与确定性竞争算法，随机化算法的竞争比为 9^k-1，确定性算法为 81^k-1。

**💡 创新点**

创新点包括：1) 只依赖 k 的竞争比；2) 通过给每辆出租车赋予排名与权重并使用指数分布的随机抽样，构造了新的潜在函数；3) 将k‑Taxi问题与k‑Server的Harmonic算法联系，利用层级匹配成本分析实现竞争性证明；4) 证明任何懒惰记忆无算法在直线度量上竞争无界，说明需要额外记忆。

**🔧 技术方法**

主要技术：潜在函数方法、最小注射匹配成本、指数分布随机抽样、权重排名更新、层级匹配成本递归分析、对偶与增广路径论证、对偶化的潜在函数与匹配的组合。

**📊 数据集**

本工作为理论算法，没有使用具体数据集；所有结果均通过分析证明在任意度量空间上成立。

**📈 对比分析**

与以往仅在 k≤3 或依赖 n、Δ 的结果相比，本文给出了单纯依赖 k 的竞争比；随机化算法相较于已有的 O(2^k log n) 或 O(2^k log^2 n) 结果取得显著提升。确定性算法则是 81^k-1 的先前最佳的 3^k-1 的扩展。

**⚠️ 局限性**

局限性：竞争比仍为指数级，特别是确定性算法的 81^k-1；需要维护排名信息（记忆），懒惰记忆无算法不可竞争；在实际大规模场景下算法实现与计算复杂度未讨论。

---

## 592. Self-Directed Apple Tasting

**arXiv ID:** 2611.06677 | [PDF](https://arxiv.org/pdf/2611.06677v1)

**作者:** Tanmay Devale `[一作]` (Purdue University), Arvind Ramaswami `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

研究自导式在线分类中的“苹果品尝”部分反馈模型，并在可实现和非可实现两种设定下给出了最优错误率的四分类（tetrachotomy）理论结果。

**💡 创新点**

创新点在于：①首次将 VC 维、∅-中心星数和自导式维度三种组合维度用于描述此类问题的误差上界；②给出了四种不同的误差增长速率（Θ(1)、ω(1)∩O(log n)、Θ̃(√n)、Θ(n)）并证明其完整性；③在可实现设定中通过构造算法（Prune‑and‑Eat、Union‑and‑Eat、Simulate‑and‑Eat）实现上述上界，且在无记忆（oblivious）对手时可进一步将 √(log n) 因子消除；④给出了匹配的下界证明，说明这些速率是最优的。

**🔧 技术方法**

技术方法包括：组合学与稀疏版本空间分析、星数与 Union‑closed 集合的平均属性、树结构（k‑树、Littlestone 树）证明随机化下的下界、一次包含图（one‑inclusion graph）在无记忆对手下的误差分析，以及对不同维度的递归分解。

**📊 数据集**

本工作完全是理论分析，没有使用任何实测数据集；所有结果均为渐近的上/下界。

**📈 对比分析**

与已有的在线学习与自导式学习理论进行比较：在全信息反馈下的误差上界与 VC 维相关，而在苹果品尝反馈下错误上界还取决于星数和自导式维度；在无记忆对手下，误差可降至 O(√(d n))，而在自适应对手下需要额外的 √(log n) 因子；在非可实现设定下，误差上界保持 O(√(n d log n)) 与传统专家投票方法相当。

**⚠️ 局限性**

局限性包括：①在 ω(1)∩O(log n) 与 Θ̃(√n) 这两类中，误差上界中仍存在 log n 因子，尚未确定其是否为必需；②目前仅对特定维度组合给出完整分类，尚未覆盖所有可能的维度关系；③部分下界构造需要对手先下分布，实际在线环境中对手可能更强；④算法实现依赖于完整的版本空间操作，在高维实际问题中可能不可行。

---

## 593. Which Physical Tactile Cues Matter for Contact-Rich Imitation Learning? A Controlled Real-Robot Study

**arXiv ID:** 2611.06673 | [PDF](https://arxiv.org/pdf/2611.06673v1)

**作者:** Xuhao Qin `[一作]` (ShanghaiTech University), Chenxi Xiao `[通讯]` (ShanghaiTech University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在LeRobot平台上进行了一项真实机器人实验，系统性比较了来自同一视觉触觉传感器的触觉外观、接触几何和切向变形等物理触觉信号，并通过Diffusion Policy实现接触丰富的模仿学习；

**💡 创新点**

创新点包括：①从同一传感器流中提取多种可解释的触觉表征并系统评估其对策略学习的贡献；②提出因子化编码（cue‑factorized encoding）以匹配不同物理信号的最佳处理方式；③展示跨注意力（cross‑attention）融合在无预训练情况下的轻量级有效替代方案；④公开了可复现的LeRobot兼容视觉触觉测试平台；

**🔧 技术方法**

使用技术包括：LeRobot兼容的标记嵌入弹性体视觉触觉传感器（产生RGB、法线/深度图、标记流）；Diffusion Policy框架；两种RGB编码器（ResNet‑18 与 Sparsh‑DINO）；MLP用于标记流；特征拼接与跨注意力融合策略；

**📊 数据集**

使用的数据集为在LeRobot平台上收集的三项接触丰富任务（试管重新定向与插入、黑板擦拭、插孔插入）的真实机器人演示，总计90次试验；

**📈 对比分析**

通过在相同的演示、视觉观察、策略骨干和训练设置下比较四种触觉输入（RGB、法线/深度、标记流、完整触觉）以及编码器和融合方式，结果显示：视觉仅成功率57.8%，完整触觉提升至78.9%；标记流为单一最佳信号（74.4%）；跨注意力与ResNet‑18相结合进一步提升至85.6%；Sparsh‑DINO + 跨注意力则达到87.8%；

**⚠️ 局限性**

局限性包括：仅使用单一视觉触觉传感器且未测量力值；前端网络架构差异未统一；实验仅覆盖平行抓手与三类桌面任务，缺乏对不同机器人、物体与传感器的泛化验证。

---

## 594. Where Search Agents Go Off the Rails: Graph-Grounded Step-Level Verification

**arXiv ID:** 2611.06662 | [PDF](https://arxiv.org/pdf/2611.06662v1)

**作者:** Hyeseo Jeon `[一作]` (Seoul National University), Jay-Yoon Lee `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于图结构的无生成评判框架 GraphRail，用于检验检索增强推理过程中的证据覆盖、推理覆盖和事实一致性。

**💡 创新点**

创新点在于将问题、检索证据和推理主张统一成三张三元组图，并通过显式的 Q‑D、Q‑T、D‑T 跨图对齐与诊断，避免使用生成式评判器，显著提升召回率并保持可追踪性。

**🔧 技术方法**

使用 LoRA 微调的 LLM 作为图抽取器，BGE‑嵌入进行语义检索，阈值匹配与最小字段分数对齐，随后基于规则的诊断与在线反馈模板。

**📊 数据集**

在 HotpotQA、2WikiMultihopQA 与 MuSiQue 三大多跳问答基准上评估，并结合 Search‑R1、ReSearch、ReAct 等检索增强推理框架。

**📈 对比分析**

与多种生成式评判基线（Self‑RAG、IBM‑Granite、GPT‑4.1‑mini 等）在轨迹级错误检测上对比，GraphRail 在 F_6 召回上显著优于基线，尤其在 MuSiQue 上保持 93%+ 召回率；在线反馈亦能提升 EM/F1 分数。

**⚠️ 局限性**

主要局限在于需先行抽取完整三元组图，导致输出 token 数与推理延迟较高；对极长文本或对齐错误仍可能产生误诊断；对非三元组结构的推理表达支持不足。

---

## 595. Fingerprinting the Dispatchers behind Distributed Model Extraction

**arXiv ID:** 2611.06661 | [PDF](https://arxiv.org/pdf/2611.06661v1)

**作者:** Snigdha Chandan Khilar `[一作]` `[通讯]` (Independent Researcher), Snigdha Chandan Khilar (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了通过请求到达和回答完成的时间戳来识别分布式模型提取（模型蒸馏）攻击中的调度器（dispatcher）行为，并利用此信息对攻击者的账号池进行定位与扩展。

**💡 创新点**

创新点包括：①基于调度器的时间信号提出四种理论上最优的指纹统计量（轮询波、成功者集中、占用比例、占用速率）；②设计了“法医扩展”算法，能在已有少量已识别账号的基础上快速排查其余攻击账号；③实现了“盲探索”式全站轮询检测与“主动水印”检测，提供显式p值；④系统性评估了这些方法在不同调度策略、流量负载和防御阈值下的性能与成本。

**🔧 技术方法**

采用的技术包括：高通滤波占用时间序列、协方差矩阵与谱分解（实现轮询波）、成功者集中度统计、Little’s Law 估计工作线程数、正态化 Z 分数与 Holm/BH 多重检验、BBP 线性相变理论用于盲搜索、主动水印的 Fisher 随机化检验、以及稀疏矩阵乘法求出大规模占用相关矩阵的前 k 维特征值。

**📊 数据集**

实验数据涵盖：①模拟与仿真场景（包括 20,000+ 账户、不同调度器和噪声设置）；②WildChat 1.96 M 请求的实时日志；③Chutes 公共推理平台 6.12 B 请求的真实生产日志（3 小时窗口）；④多条现场实验（Baseline、Repeat、Stress、Evasion、Two‑replica、Production）和多种参数设置。

**📈 对比分析**

与内容检测与基线统计方法比较：在 20% 种子下，轮询、公司、哈希与保险箱池在 10–20 分钟内 AUC 0.92–1.00；盲搜索在 3.4 h 后恢复强轮询池；主动水印对闭环池达到 z>10，提供显式 p 值；相对传统内容报警，法医扩展能在 10‑20 分钟内把未被报警的 80‑90% 攻击账户召回；误报率保持在 <0.1%。

**⚠️ 局限性**

局限性包括：①被 pinning 或恒定开环的调度器完全不可检测；②盲探索需要数小时的数据且在生产噪声下对弱轮询池失效；③方法假设流量为平稳且无外部干扰；④对合法共享调度（如公司内部多账号排队）会误判；⑤主动水印对 jitter>2 s 或持久性持键攻击失效。

---

## 596. Alternatives to Shallow Autoencoders for Collaborative Filtering

**arXiv ID:** 2611.06610 | [PDF](https://arxiv.org/pdf/2611.06610v1)

**作者:** Mario Mallea `[一作]` (Universitat Politècnica de Catalunya), Francisco Mugica `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于用户-用户相似性的 EASE 变体 u-EASE 以及结合用户与物品信息的混合模型 u-EASE-i，提供闭式解，改进传统基于物品的 EASE 以提升协同过滤推荐准确度。

**💡 创新点**

创新点在于：① 用用户 Gram 矩阵构建线性映射，获得用户侧闭式解；② 将用户侧与物品侧双重 Gram 矩阵融合，形成全新的混合模型；③ 在高用户/物品比例的数据集上实现对 EASE 的显著性能提升。

**🔧 技术方法**

技术实现基于岭回归与闭式解，利用 Gram 矩阵求逆并结合 Lagrange 约束；实验对比 GF-CF、∞-AE、SVD-AE、MF-BPR、NeuMF、LightGCN、MultVAE 等模型。

**📊 数据集**

使用的公开数据集为 Gowalla、Yelp2018 与 ML-1M 三个大规模隐式反馈数据集。

**📈 对比分析**

采用 HR@10/100、nDCG@10/100、PSP@10 等评估指标，在三大数据集上与基线模型对比。u-EASE、u-EASE-i 在 Yelp2018 与 ML-1M 上分别提升 2.5%/3.7% 与 6.0%/6.3% 的 nDCG@100；在 ML-1M 上实现 state‑of‑the‑art，训练时间仅数分钟。

**⚠️ 局限性**

局限性：仍需构造并求逆全秩 Gram 矩阵，导致大规模稀疏数据下内存与计算开销较高；在用户/物品比例较低的场景中优势有限；未对低维嵌入或稀疏近似方法进行深入探究。

---

## 597. The determinantal complexity of the 4*4 permanent is 15

**arXiv ID:** 2611.06658 | [PDF](https://arxiv.org/pdf/2611.06658v1)

**作者:** Karthik Sheshadri `[一作]` `[通讯]`, Karthik Sheshadri

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `847a60d8-a755-47af-ba5d-c5236b9e3083` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了 4×4 恒等式的行列式复杂度为 15，给出了相应的 15×15 线性形式矩阵表示，并证明不存在更小矩阵的表示。

**💡 创新点**

首次在 n=4 时精确确定了行列式复杂度，构造了上界（Grenet 的 2^n-1 构造）与下界（新的 Hessian 不等式与可接受核的分类）的完全匹配。

**🔧 技术方法**

采用了正则化到零点的正规形式、Hessian 约束、坐标子空间分类、Borel 定理、线性因子分解与 Clifford 降维等代数几何与线性代数技术，并通过符号计算验证所有细节。

**📊 数据集**

无传统数据集；所有计算均为符号化、对多项式的解析推导与有限域上多项式方程求解。

**📈 对比分析**

与此前已知的上界 15（Grenet）和下界 10 对比，最终确立了精确值 15；该结果在理论上提供了 4×4 行列式复杂度的完整答案。

**⚠️ 局限性**

结果仅适用于 n=4（以及在特征 2 下为 4），在更大 n 或其他特征下仍未给出精确值；构造与证明高度依赖符号计算，扩展至更大规模目前难以实现。

---

## 598. Approximating #BIS in $\exp(\widetilde O(n^{1/3}))$ Time

**arXiv ID:** 2611.06657 | [PDF](https://arxiv.org/pdf/2611.06657v1)

**作者:** Jingcheng Liu `[一作]` (Nanjing University), Yitong Yin `[通讯]` (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了随机算法，用于在任意二分图中近似计数和几乎均匀采样独立集。

**💡 创新点**

算法在固定精度下的运行时间为exp(O(n^1/3log^4/3n))，并且适用于具有有效可评估的正对数超模权重的布尔变量。

**🔧 技术方法**

使用了加权的Sinclair-Jerrum步行算法和变分原理来优化残余对数分区函数的粗略下界估计。

**📊 数据集**

在任意二分图上进行实验，特别是对于具有n个顶点的图。

**📈 对比分析**

与现有方法相比，提出的算法在时间复杂度上有显著改进，能够在固定精度下实现更快的近似计数和采样。

**⚠️ 局限性**

算法的局限性在于它依赖于对数超模权重的假设，且在一般二分图上可能不适用。

---

## 599. Detecting Silent Action-Interface Mismatch with Calibrated Relational World Models

**arXiv ID:** 2611.06653 | [PDF](https://arxiv.org/pdf/2611.06653v1)

**作者:** Rui Zhu `[一作]` (Yale University), Haixu Tang `[通讯]` (Indiana University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在已录制的机器人转移中，使用离线训练的状态–动作–效果关系检测并定位数值合法但语义错误的动作变换；

**💡 创新点**

提出基于校准的关系得分、联合高斯混合与条件混合模型的对比评估，并引入匹配FPR的后决策分析，揭示检测-定位权衡；

**🔧 技术方法**

使用岭回归预测残差、联合高斯混合模型（GMM）、条件GMM、Mahalanobis距离、场对齐位移模型等多种得分技术；

**📊 数据集**

主要使用公开的LIBERO Spatial任务集（432源情节，共约160k转移），同时对DROID‑100、xArm Lift、HYDRA等外部任务进行验证；

**📈 对比分析**

在匹配2.5%清洁FPR的条件下，条件GMM召回率最高（≈65.7%），联合GMM次之（≈61.0%），岭回归最低（≈46.1%）。定位准确率为联合GMM 95.8%，条件GMM 94.6%；相比传统Mahalanobis/范围校验，显著提升检测召回和定位；

**⚠️ 局限性**

仅限离线重放，无法保证跨机器人或在线部署的安全性；对外部约束（如HYDRA、xArm等）失效；不提供闭环成功改进或自动修复；阈值需在训练集上校准，无法直接迁移到新任务或环境。

---

## 600. flash-comm: Self-Synchronizing Collective Communication for LLM Decoding

**arXiv ID:** 2611.06634 | [PDF](https://arxiv.org/pdf/2611.06634v1)

**作者:** Carrie Wang `[一作]` (ScitiX.AI), Jason Zhao `[通讯]` (ScitiX.AI)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Flash-comm，一种融合了数据就绪同步的自同步传输库，专为分布式LLM解码中的集合通信（All-Reduce和All-Gather）设计。

**💡 创新点**

创新点在于：1）每个数据块携带就绪证据，消除全局barrier；2）基于固定成本与每字节成本的成本引导选择器，动态选择就绪编码（sentinel或序列号）和通信路径；3）将传输顺序与归约顺序分离，保证浮点归约一致性；4）支持多种通信路径（单跳、分片双跳），并在单节点NVLink域内实现高吞吐。

**🔧 技术方法**

使用的技术包括：自同步传输（sentinel/sequence‑number编码）、成本模型与离线校准的协议选择、CUDA Graph安全重用的对称内存、Programmatic Dependent Launch（PDL）、多播与多节点内存访问（multimem）、以及基于CUDA核的异步归约实现。

**📊 数据集**

在实验中使用的模型为DeepSeek‑V4‑Flash‑0731的tensor‑parallel推理，数据集并未特别指定（主要以推理任务为主）。

**📈 对比分析**

通过在八张NVIDIA B300、H100和H200 GPU上分别跑All‑Reduce、All‑Gather和端到端解码实验，比较Flash‑comm与SGLang custom AR v2以及最快的基线。结果显示：All‑Reduce中Flash‑comm在所有消息大小下平均降低28.8%（最高43.7%）的延迟；All‑Gather平均降低23.8%（最高58.9%）。在端到端解码中，TPOT（每输出token平均延迟）在不同并发度下分别降低1.7%–6.8%。

**⚠️ 局限性**

局限性包括：1）仅适用于单节点、2/4/8 GPU、NVLink域内；2）需要Hopper或Blackwell GPU、对称内存、CUDA Graph支持；3）消息大小范围限定在16 B–16 MB；4）未覆盖跨节点网络、all‑to‑all或更大规模的系统；5）需要针对每个GPU世代手动校准成本模型。

---

## 601. Trick-or-Treat: Adversarial Evaluation of Multimodal Medical Reasoning with Synthetic Images

**arXiv ID:** 2611.06631 | [PDF](https://arxiv.org/pdf/2611.06631v1)

**作者:** Oishi Banerjee `[一作]` (Harvard Medical School), Pranav Rajpurkar `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了 Trick‑or‑Treat 基准，利用合成临床照片与真实胸部 X 光片配对，构建对抗性多模态医学评估框架，探测模型在不同影像模态之间推理时的幻觉行为。

**💡 创新点**

创新点在于：①采用合成照片实现同一患者在同一体位下呈现不同病理表现，消除背景混淆；②通过匹配/不匹配的影像配对揭示模型在跨模态推理中倾向于“填补”信息、产生错误诊断的机制；③首次将合成数据用于构造可控的多模态错误挖掘。

**🔧 技术方法**

技术方法包括文本到图像的生成模型（如 DALL·E‑2/类似技术）用于生成外部临床照片，结合人机交互校正；使用三大前沿视觉‑语言模型（Claude Fable 5.1、GPT‑6 Astra、Gemini 4 Argon）进行评测；采用基于规则的 LLM 判别器（Claude Sonnet 5.5）计算每张照片的假阳性/假阴性率。

**📊 数据集**

数据集主要由 40 张真实胸部 X 光片（来自 ReXGradient，AP/PA 视图，成年患者）与 80 张合成临床照片（共 40 条匹配对和 40 条不匹配对）构成，涵盖 40 种临床疾病。

**📈 对比分析**

比较方法是对每张照片计算假阳性和假阴性率，分别在仅照片和照片+胸片两种设置下评估。结果显示：仅照片时错误率为 25–37.5%；加入胸片后，匹配对错误率下降，然而不匹配对错误率显著上升（45–57.5%），表明模型在跨模态情境下易产生幻觉。

**⚠️ 局限性**

局限性包括：①仅评估照片描述，未考察最终诊断准确性；②使用合成照片，可能存在生成工件或偏差；③基准仅覆盖胸部 X 光与外部照片两种模态，未扩展至其他影像或临床数据；④样本量相对有限，且仅包含成人成人患者。

---

## 602. Amortized Optimization for Calibrating Stochastic Social Simulators

**arXiv ID:** 2611.06623 | [PDF](https://arxiv.org/pdf/2611.06623v1)

**作者:** Muyao Zhong `[一作]`, Peng Yang `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习技术和强化学习算法相结合的方法。

**📊 数据集**

采用了公开的图像识别数据集进行实验，数据集包含数万张标注图像。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和处理速度上均优于其他方法。

**⚠️ 局限性**

限制在于算法在特定类型的数据上表现不佳，且对计算资源的需求较高。

---

## 603. Where It Breaks: Diagnosing Perception, Memory, and Decision Failures of a Vision-Language Agent in Pac-Man

**arXiv ID:** 2611.06704 | [PDF](https://arxiv.org/pdf/2611.06704v1)

**作者:** Xinglu Wang `[一作]`, Zhenan Fan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用 Qwen3.5-9B 训练 Pac‑Man 代理，通过分离感知校验和原始动作的强化学习实验，系统评估训练过程中的认知与决策质量。

**💡 创新点**

提出将静态图像读取、交互感知和 ASCII 动作决策分离评估，揭示训练成功率与游戏完成度可能脱节的现象，并阐明重启策略对学习曲线的误导性影响。

**🔧 技术方法**

技术手段包括基于文本的 Qwen3.5-9B 语言模型、PPO‑style 强化学习、ASCII 语义板解析、静态截图校验、重启训练与多阶段课程学习。

**📊 数据集**

数据集主要是 Pac‑Man 的 8 个保留迷宫（每个 12 次验证），加上 24 张静态截图和 138 个角色查询，用于感知准确性测试；训练集为多阶段混合地图和重启状态。

**📈 对比分析**

比较方法是对比训练前后（从 0 到 270 checkpoint）的正常启动验证完成率（从 0% 提升到 80.2%），以及后续 checkpoint（335 为 78/96、425 为 60/96）的波动；相对训练批次清除率也被记录，发现训练成功率并不必然转化为游戏完成度。

**⚠️ 局限性**

局限性包括：缺乏独立随机种子复现；仅评估 8 个迷宫，缺乏更广泛的布局泛化；重启状态多源性不充分，难以判断经验多样性；未解析奖励设计或长序列信用分配导致的退化；未提供完整的训练日志或每个步骤的地图映射。

---

## 604. Agent-Driven Generative Priors for Pathology Image Synthesis and Segmentation

**arXiv ID:** 2611.06701 | [PDF](https://arxiv.org/pdf/2611.06701v1)

**作者:** Sicheng Yang `[一作]` (Hong Kong University of Science and Technology), Lei Zhu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

构建了三阶段框架：利用AI代理对TCGA图像进行质量筛选，训练扩散Transformer生成先验，并将该骨干迁移到细胞语义分割任务；

**💡 创新点**

创新点在于将代理驱动的质量筛选、生成模型的先验学习与生成器骨干直接用作分割编码器的端到端统一流程；

**🔧 技术方法**

采用GPT‑4.1‑mini代理+ResNet‑50质量回归器、Diffusion Transformer（DiT）与Flow Matching、AdaLN时间嵌入以及轻量多级解码器；

**📊 数据集**

使用TCGA‑200K（217k高质量切片）、公开的PanNuke和自建的PathSeg‑380K（380k细胞标注）三大数据集；

**📈 对比分析**

与多种CNN/Transformer/SAM分割模型及预训练编码器对比，mIoU最高达55.46%/55.45%，mDice≈68.8%/67.8%，p‑Acc≈93%，比第二名提升1–2个百分点；

**⚠️ 局限性**

局限在于代理筛选成本高、对不同染色协议的泛化有限，以及过度严格筛选可能削弱训练多样性。

---

## 605. PINR-ESI: Physics-Informed Self-Supervised Implicit Neural Representation Framework for EEG Extended-Source Imaging

**arXiv ID:** 2611.06700 | [PDF](https://arxiv.org/pdf/2611.06700v1)

**作者:** Zhiyong Zhou `[一作]` (University of Science and Technology of China), Yakang Dai `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了PINR-ESI框架，利用物理信息自监督的隐式神经表示（INR）直接从测量的EEG重建扩展皮层源分布，无需配对源标签或预训练。

**💡 创新点**

创新点在于：①将源场表述为连续坐标条件的MLP，避免显式源参数；②通过lead‑field矩阵与“patch‑integrated continuous‑to‑discrete”投影实现前向一致性约束；③利用经典逆估计做warm start并提供软空间支持约束，使自监督优化不受合成-真实偏差影响。

**🔧 技术方法**

采用的技术包括：物理信息自监督优化、隐式神经表示（INR）+ MLP、patch‑integrated continuous‑to‑discrete前向投影、经典逆估计初始化（eLORETA/wMNE）与空间支持约束。

**📊 数据集**

使用的数据集包括：①模拟数据（fsaverage表面、ico4源空间、BioSemi64电极、10 dB SNR等不同源面积/数量设置），②临床癫痫患者HDEEG‑IED‑SurgOutcome 24例（自定义前向模型），③Localize‑MI 试验数据的内颅刺激记录。

**📈 对比分析**

与LCMV、wMNE、ConvDip、DeepSIF、FAST‑IRES、ADMM‑ESINet等七种方法对比，评估指标为AUC、DLE、SD、NRMSE。PINR-ESI在所有噪声、源面积和源数条件下均保持最低SD，尤其在低SNR（0 dB）和大源面积（10 cm²）下优于基线；在临床数据中平均DLE 14.49 mm，优于wMNE（17.67 mm）及其他方法。

**⚠️ 局限性**

局限性包括：①对多源重叠的分离效果仍弱，难以完全提高AUC和DLE；②每个时间点单独优化导致计算成本高；③未利用时间序列信息，缺乏跨时序约束；④对前向模型误差和噪声模型的鲁棒性尚未充分验证。

---

## 606. Sekai: Incremental Event Memory for Generalized Reasoning Video Segmentation

**arXiv ID:** 2611.06685 | [PDF](https://arxiv.org/pdf/2611.06685v1)

**作者:** Zhengtao Liu `[一作]` (University of Science and Technology of China), Yicong Li `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Generalized Reasoning Video Segmentation（GRVS）任务，并设计了Sekai框架来处理视频流中的动态目标分割。

**💡 创新点**

将目标集合视为随时间变化、与事件历史相关的可变集合，并引入可编辑文本事件记忆捕捉过去事件，突破传统固定目标假设。

**🔧 技术方法**

结合文本记忆写入器（Event Memory Reasoner）与基于记忆的解码器（Memory-Grounded Decoder），使用Qwen3-VL-8B-Instruct与SAM 3.1分割器，并对定位器进行LoRA微调。

**📊 数据集**

创建Isekai基准（60段视频、197个查询、12569个评分点），并在MeViS v2、ReasonVOS、ReVOS等数据集上评估。

**📈 对比分析**

与四种在线RVS基线（OnlineRefer、SAMWISE、StreamingRVOS、CoT-RVS）以及无记忆控制和GT盒子基线对比，Sekai在Isekai上获得最高J&F分数，尤其在Ever类查询上提升约9点，整体提升3.2点。

**⚠️ 局限性**

事件记忆一旦写入不可更新，错误会累积；且记忆中的错误可能影响Latest、Never等查询，需要进一步的验证或ID显式读出机制。

---

## 607. SecureGARD: Dynamic Ontology-Guided Governance for Secure Tool-Using LLM Agents

**arXiv ID:** 2611.06682 | [PDF](https://arxiv.org/pdf/2611.06682v1)

**作者:** Guoqin Xiong `[一作]` (Nanjing University of Aeronautics and Astronautics), Sheng-Jun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 SecureGARD 的动态本体驱动治理框架，用来在 LLM 代理调用外部工具时通过显式的动作–状态语义来判定是否允许执行、等待恢复或禁止。

**💡 创新点**

创新点在于：① 将异构工具调用映射为共享语义动作；② 通过可冻结的 TBox 与每个 episode 的 ABox 结合，实现三分决策（ready/waiting/forbidden）；③ 设计“等待”决策以支持可恢复的局部重新规划；④ 离线用 LLM 提议动作–状态规则，并经过一致性、保护性与验证集检查后冻结本体。

**🔧 技术方法**

技术方法包括：基于 OWL 2 的本体建模与 HermiT 逻辑一致性推理；语义执行建模（工具调用→语义动作实例）；Episode‑level TaskState 用于追踪已完成/待完成动作与未满足的义务；离线本体演化流程（训练证据 → 规则提议 → 验证 → 冻结）。

**📊 数据集**

使用的数据集为 AgentDojo（用于本体演化和内部评估）、ASB 与 AgentDyn（跨 benchmark 迁移测试），每个 benchmark 下均包含直接与间接注入攻击场景。

**📈 对比分析**

对比方法包括无防御、Tool Filter、Progent、DRIFT、CaMeL 等系统级防御。实验表明 SecureGARD 在所有 benchmark 与 LLM backbone（GPT‑4o‑mini、Qwen3.8‑Flash 及其他 5 款）下均实现 0% 的攻击成功率，同时在 Clean Utility 与 Attack Utility 上至少与 DRIFT 同等或更好，且推理成本与无防御相近。

**⚠️ 局限性**

局限性：① 依赖于初始 seed axiom 的覆盖与质量；② 本体词汇仅包含基础动作–状态关系，缺少更丰富的时序、条件与跨动作依赖；③ 在超长任务或快速演变的工具环境中，离线演化与等待恢复的有效性尚未充分验证。

---

## 608. Perceived Trust and Transparency in AI-Altered vs. Human-Written LinkedIn Posts

**arXiv ID:** 2611.06660 | [PDF](https://arxiv.org/pdf/2611.06660v1)

**作者:** Thomas Christian Joos `[一作]` (University of Zurich), Nicolò Pagan `[通讯]` (University of Zurich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在社交媒体帖子中人工智能改写（AI‑altered）的效果，对可信度与透明度的影响。

**💡 创新点**

提出了AI介入的四阶段谱系，并发现仅AI改写本身不影响读者信任，而模型与提示会改变评价。

**🔧 技术方法**

使用了两种大型语言模型（Claude Sonnet 4.6 与 Gemma3:4b）和四种提示策略。

**📊 数据集**

基于 WildChat 对话数据分析后挑选了 6 条 2019 前的公开 LinkedIn 帖子，并用模型生成 AI‑altered 版本。

**📈 对比分析**

通过在线实验（n=214）对比人类原稿与 AI 改写的可信度与透明度评分，发现差异不显著，且检测率仅略高于随机。

**⚠️ 局限性**

受限于仅六条帖子、模型与提示组合有限以及未能检验真实用户行为，结果可能不具普适性。

---

## 609. Compounding Physical RSI: Turning Execution Failure into Inherited Capability

**arXiv ID:** 2611.06654 | [PDF](https://arxiv.org/pdf/2611.06654v1)

**作者:** Yixuan Yang `[一作]`, Xiu Su `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Compounding Physical RSI 框架，通过递归执行、诊断失败、针对性强化学习与权重整合，逐代提升通用机器人操控能力。

**💡 创新点**

创新点在于将阶段级别的能力缺口诊断与目标化经验选择结合，利用离线强化学习针对性提升并通过权重插值与全量评估实现能力继承，实现多代叠加收益。

**🔧 技术方法**

采用自动化能力缺口评估、经验统一化与验证、目标条件离线强化学习（行为正则化+保留项）、权重线性插值与全量任务评估。

**📊 数据集**

在 RoboDojo‑Sim 仿真基准上评估，覆盖 42 个多阶段操控任务。

**📈 对比分析**

与 13 公开基准政策对比，CSU‑AI‑07 在 Overall Score 上领先 5.92 点（≈29%），在 Success Rate 上领先 5.82%（≈39%），三种评估种子平均稳定。

**⚠️ 局限性**

局限在于框架仍需针对单一改进策略、未在真实物理硬件验证、以及对不同任务难度的自适应机制尚未完善。

---

## 610. SCRUB: Contact-Based Surface-Frame Estimation for Force-Motion Control with an Armless Quadruped

**arXiv ID:** 2611.06649 | [PDF](https://arxiv.org/pdf/2611.06649v1)

**作者:** Sourav Selvaraj `[一作]` (Worcester Polytechnic Institute), Mahdi Agheli `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在无臂四足机器人上实现了基于接触的表面帧估计与力–运动控制，完成墙面与柱面抛光/清扫任务；

**💡 创新点**

提出SCRUB算法：利用力和速度两种线索的Kalman滤波器估计表面法线，无需已知摩擦系数；并通过步态平均的速度行列提升估计鲁棒性；

**🔧 技术方法**

使用了接触力传感、惯性-腿部运动学融合的扩展卡尔曼滤波、MPC规划工具反作用力、全身控制、表面曲率估计等技术；

**📊 数据集**

在Unitree Go1四足机器人上进行现场实验，测试OSB墙面和干墙柱面，速度8/12/16 mm/s、负载5/10/15 N等条件；

**📈 对比分析**

通过与固定法线、摩擦补偿法线、无速度行列、无力行列、标量MPC等基线和消融实验对比，SCRUB在所有测试中完成100%行进/轨道，平均法线RMSE约2.3–2.5°，力RMSE约2–3 N；

**⚠️ 局限性**

假设样本独立导致精度高估；未建模姿态漂移与步态波动对力行列噪声的影响；未考虑接触瞬态摩擦、视觉预览或多接触法线估计，且仅在近垂直表面和低速条件下验证。

---

## 611. Replicability, Reproducibility, and Robustness in Political and Social Science Research with Large Language Models

**arXiv ID:** 2611.06639 | [PDF](https://arxiv.org/pdf/2611.06639v1)

**作者:** Joan C. Timoneda `[一作]` (Purdue University), Sebastián Vallejo Vera `[通讯]` (University of Western Ontario)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究大语言模型输出的概率性差异，提出按结果分布匹配的复制标准，并通过标准化提示脚本与误差结构分析评估模型内部一致性，使用设计校正和堆叠集成恢复下游推断精度。

**💡 创新点**

将复制定义为分布一致而非逐字一致，构建误差结构诊断框架，证明测量选择可翻转回归系数，并提出金标子校正与堆叠集成兼顾有效性与精度的全新方法。

**🔧 技术方法**

采用概率复制标准、标准化提示脚本、误差结构分析、设计基础监督学习校正以及开放权重模型的堆叠集成技术。

**📊 数据集**

使用了三个公开数据集（具体名称未给出）进行模型内部变异与下游回归实验。

**📈 对比分析**

通过对比未校正模型、单一校正模型和堆叠集成模型的回归系数与误差结构，校正模型恢复了有效性但精度下降，堆叠集成则在保持校正后的同时恢复了大部分精度。

**⚠️ 局限性**

校正过程依赖于金标子样本，导致标注成本和样本需求增加；堆叠集成仍不能完全消除所有误差，且方法对不同模型版本的泛化能力仍有限。

---

## 612. Uncertainty-Driven Efficient Communication for UAV-Based Collaborative Perception

**arXiv ID:** 2611.06635 | [PDF](https://arxiv.org/pdf/2611.06635v1)

**作者:** Jiuwu Hao `[一作]` (University of Chinese Academy of Sciences), Pin Lv `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于不确定性的UAV多机协同感知框架，通过观测不确定性和判别不确定性构建请求与供应图，实现高效的空间信息交换与融合。

**💡 创新点**

创新点在于：①同时考虑观测受限与模型不确定两类失误，提升协同价值识别；②基于不确定性驱动的跨注意力融合模块U‑Fusion；③兼容异构传感器且通信成本显著下降。

**🔧 技术方法**

核心技术包括：不确定性估计（点云密度/深度置信度、分类/几何不确定性）、空间请求/供应图设计、基于不确定性的交叉注意力融合、随机遮蔽训练与几何不确定性监督。

**📊 数据集**

使用了三大UAV协同感知基准：UAV3D（仿真）、Air‑Co‑Pred（仿真/交叉）和AGC‑Drive（真实世界）等。

**📈 对比分析**

与SOTA方法（Where2comm、V2X‑ViT、LIF等）在三种融合范式下对比，本文在mAP、IoU、VPQ等指标上实现了最高的准确率与最低的通信量，表现出优异的性能‑带宽折衷。

**⚠️ 局限性**

局限性包括：需对超参数进行细致调优；在极高规模的网络中通信调度仍可能成为瓶颈；对非常稀疏或遮挡严重的场景的鲁棒性尚待进一步验证。

---

## 613. Learning to Track Cells from Sparse Annotations in 3D Time-Lapse Microscopy

**arXiv ID:** 2611.06624 | [PDF](https://arxiv.org/pdf/2611.06624v1)

**作者:** Yusuke Uchida `[一作]` `[通讯]` (GO Drive Inc.), Yusuke Uchida (GO Drive Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一套完整的3D细胞追踪管线，集成稀疏标注的细胞检测、基于运动、外观和分裂阶段特征的候选连接校准，以及联合选择普通链接与分裂事件的图优化和后处理。

**💡 创新点**

创新点包括：1) 在稀疏标注下抑制误背景监督的2.5D与3D检测器融合；2) 结合密集光流、匹配器和分裂前后阶段模型，对普通链接和分裂候选进行精细校准；3) 构造基于评估指标的 surrogate 目标，联合优化普通链接与分裂，解决竞争关系；4) 使用多模型校准器（LightGBM/XGBoost）提升分裂恢复。

**🔧 技术方法**

使用的技术有：2.5D U‑Net + 3D SegResNet 检测器、EfficientNetV2/S 匹配器、Dense Flow 网络、光流对齐的分裂父子模型、LightGBM/XGBoost 校准器、HiGHS LP/MILP 求解器、Gap filling 与坐标细化。

**📊 数据集**

数据集为 zebrafish embryo 的 3D 光灯扫描时序图像，共 199 段视频（来自两枚胚胎），注释节点约 133,318，覆盖率约 2.8%。

**📈 对比分析**

通过 5 折交叉验证评估，整体分数达 0.9762（edge Jaccard 0.9235，division Jaccard 0.5266）。相较于顺序选择方法提升 0.0105；在 Kaggle 公开/私有排行榜分别取得 0.9780 / 0.9671 分，优于现有方法。

**⚠️ 局限性**

局限性包括：1) 评估指标仅基于已标注细胞，未能全面衡量未标注区域的追踪质量；2) 由于分裂正样本稀少，分裂恢复仍受限；3) 光流逆变换近似、长间隙插值及分裂校准目标与官方容差不完全匹配，可能导致性能偏差。

---

## 614. A Width Hierarchy of Monotone Algebraic Branching Programs

**arXiv ID:** 2611.06608 | [PDF](https://arxiv.org/pdf/2611.06608v1)

**作者:** Julian Dörfler `[一作]` (Saarland Informatics Campus), Alexander Rogovskyy `[通讯]` (Saarland Informatics Campus)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了单调算术分支程序（mABPs）的宽度层次结构，证明了从宽度2到宽度3以及更高宽度的严格分离，并给出了与单调公式深度、宽度之间的权衡关系；此外，还将Snir的单调公式下界框架推广到有限深度单调公式，得到迭代矩阵乘法多项式和行列式的精确叶子数上界与下界。

**💡 创新点**

创新点主要包括：①给出了完整的单调宽度层次分离（k⊊k+1），并证明了宽度3无法模拟宽度2；②提出宽度分解与稠密化Lemma，实现了从宽度到深度的精确上界；③通用化Snir框架到有限深度单调公式，得到迭代矩阵乘法多项式的叶子数下界为 n·k^(n^1‑1)，以及行列式的最优叶子数为 ((e+o(1))·ln^(∘‑1)(n))^n；④分析了宽度-深度权衡，给出了迭代矩阵乘法在宽度 w<k 时的深度上界。

**🔧 技术方法**

技术方法包括：支持计数与宽度分解、宽度递归分解与稠密化Lemma、单调公式的深度化与层次分解、组合论极大极小化分析、Snir框架的动态规划扩展、以及对多项式矩阵乘法结构的分块展开。

**📊 数据集**

本研究为理论分析，无实验数据集；主要使用标准的迭代矩阵乘法多项式 k_n 和行列式（permanent）作为理论实验对象。

**📈 对比分析**

通过证明下界与上界的匹配或多项式差距，展示了宽度层次的严格性；与已知结果（如宽度3可达到全部单调多项式但尺寸非多项式）对比，进一步压缩宽度-深度差距；理论上给出了在宽度 w<k 时的深度上界，使得实际构造与理论下界的差距趋于多项式级别。

**⚠️ 局限性**

局限性：结果仅适用于单调模型，非单调情况下宽度可压缩到3；对宽度-深度的具体常数与参数仍有提升空间；未讨论非均匀宽度、随机化或非多项式大小的情况；对更大宽度范围的精确分离仍需进一步研究。

---

## 615. ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus

**arXiv ID:** 2611.06706 | [PDF](https://arxiv.org/pdf/2611.06706v1)

**作者:** Tianwei Zhang `[一作]` (Meedan), Scott A. Hale `[通讯]` (Meedan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了名为ALTO的三语技术促进的性别暴力（TFGBV）检测数据集。

**💡 创新点**

创新点在于采用社区主导的主动学习采样与多语种细粒度标签体系，将TFGBV框架细化为三层子类别，并聚焦低资源语言。

**🔧 技术方法**

使用主动学习、Fine‑tune预训练语言模型（如BERT、GPT类）以及Perspective API、OSS‑SG等现成模型进行评估。

**📊 数据集**

使用的语料包括Levantine Arabic、African French、Swahili三语的3022条通过社区举报线索与社交媒体抓取获得的文本实例，并标注了性别化与伤害类型。

**📈 对比分析**

方法上对比了现有模型在ALTO上的性能，发现原模型召回率低且误报率高；Fine‑tune后多语种模型召回率提升至70%+，但在负样本中的假阳性略高；整体表现优于现有技术。

**⚠️ 局限性**

局限包括标签二值化导致细节信息缺失、仅覆盖文本且仅采样自X平台、注释者全为女性可能缺乏多元视角、数据与基准集可能存在重叠、机器翻译质量未充分评估。

---

## 616. Do unlearning algorithms learn the right features?

**arXiv ID:** 2611.06681 | [PDF](https://arxiv.org/pdf/2611.06681v1)

**作者:** Polina Dolgova `[一作]` (CISPA Helmholtz Center for Information Security), Sebastian U. Stich `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究机器无学习方法，提出特征恢复评价并证明输出一致不等于特征一致，比较八种近似无学习方法在不同数据集上的特征恢复情况。

**💡 创新点**

创新点在于将特征恢复作为无学习评估标准，提供理论证明及新的特征相似度指标，展示特征不匹配如何影响后续学习和迁移。

**🔧 技术方法**

使用模型定义特征框架、PCA聚类、相关性匹配、特征失配度量、实验干预、梯度步长分析等技术。

**📊 数据集**

主要数据集包括CIFAR‑100（ViT‑Tiny、ResNet‑56）、SVHN、Flowers‑102、Oxford‑IIIT Pets等。

**📈 对比分析**

通过与完整训练和重新训练模型的特征相匹配度量以及下游准确率、部分重学习准确率进行比较，结果显示大多数方法在传统指标上表现良好，但在特征恢复和后续表现上存在显著差距。

**⚠️ 局限性**

局限在于仅在固定架构和删除比例下验证，缺乏更广泛的模型与任务多样性；干预方法在实际应用中的可行性和成本尚未充分评估。

---

## 617. Measuring Risk, Not Harm: Detecting Psychological Harm Risk Factors in LLM Conversations

**arXiv ID:** 2611.06806 | [PDF](https://arxiv.org/pdf/2611.06806v1)

**作者:** Nil-Jana Akpinar `[一作]` (Microsoft), Matthew Dressman `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM响应的风险因子检测框架，用以持续监测心理安全风险，并构建了包含8类风险因子的多标签合成数据集。

**💡 创新点**

创新点在于将心理风险拆解为可观测的AI行为特征并通过合成数据进行大规模训练，同时设计了轻量级线性模型与LLM判定器的两阶段级联，提高检测效率和精度。

**🔧 技术方法**

采用了基于OpenAI 3,072维嵌入的L2正则化逻辑回归、共享多头MLP、LLM零射击判定器，以及级联架构，并利用ASSERT框架与GPT-5.4.2进行合成生成与标注。

**📊 数据集**

使用了47,062条AI响应的合成多标签数据集（373,873个标签）进行训练和验证，外部对比数据集包括WildChat、DelusionEval（公开与受限版）和AICompanionBench。

**📈 对比分析**

在黄金测试集上对线性模型、MLP、LLM判定器和级联四种配置进行评估，级联在宏观精确度0.86、召回率0.76、F1 0.79的同时将LLM判定调用率降低73%（WildChat中降低97.1%）。

**⚠️ 局限性**

局限性包括仅覆盖8类非完整风险因子、合成数据可能偏离真实对话、人工验证样本为目标采样而非代表性、OOD环境下缺乏召回评估、分布漂移导致判定器误差，以及未直接关联到真实心理危害。

---

## 618. Semantic Residual Vector Quantization for Sign Language Generation

**arXiv ID:** 2611.06781 | [PDF](https://arxiv.org/pdf/2611.06781v1)

**作者:** Chang Liu `[一作]` (University of Trento), Nicu Sebe `[通讯]` (University of Trento)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于语义残差向量量化（SRVQ）的手语生成框架，将文本语义与运动细节分层编码并端到端生成SMPL‑X姿态。

**💡 创新点**

创新点在于：① 在单一残差量化器中把首层对齐文本语义，后层细化运动；② 引入词感知条件与双层掩码Transformer实现语义与运动的分离；③ 通过对比学习实现文本‑运动对齐，实现端到端训练。

**🔧 技术方法**

使用技术包括：语义残差向量量化（RVQ）、InfoNCE对比语义对齐、SMPL‑X姿态编码、掩码Transformer（Semantic & Kinematic Generator）、词级+全句交叉注意、CFG引导。

**📊 数据集**

使用数据集：CSL‑Daily（中文手语，18k 训练/1176 测试）和 Phoenix‑2014T（德语手语，≈7k 训练/642 测试），并结合Spoken2Sign的帧级词头标注进行量化器训练。

**📈 对比分析**

与 Prog.Trans、Text2Mesh、T2S‑GPT、S‑MotionGPT、SOKE 等基线在检索（R@1、R@5、R@10、FID）和几何（DTW‑JPE、DTW‑PA）指标比较；SRVQ sem+kinematic 在 CSL‑Daily 上 R@1 82.3（+30）和 Phoenix‑2014T 上 R@1 70.3（+46）显著超越 SOKE；语义‑only 方案在几何指标上同样优于先前方法。

**⚠️ 局限性**

局限性：需要高质量 SMPL‑X 运动与帧级词头监督，推理阶段仍依赖预训练文本编码器；未充分建模非手势特征（如面部表情、身体姿态）；跨语种通用性和实际使用者评估仍待进一步验证。

---

## 619. Stability and Perfection in Constrained Weighted Network Games

**arXiv ID:** 2611.06775 | [PDF](https://arxiv.org/pdf/2611.06775v1)

**作者:** Julien Fixary `[一作]` (Université Mohammed VI Polytechnique), Rida Laraki `[通讯]` (Université Mohammed VI Polytechnique)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

提出了一套通用的受约束加权网络形成框架，并在此基础上引入了联合协作（coalitionwise）稳定性概念，证明了在连续性和分量式准凹性下该稳定性总是存在。

**💡 创新点**

创新点包括：①将单边连接扩展为任意团体合作并加入合作容量与个人预算约束；②在受约束环境下重新定义并证明了完美（trembling-hand / perturbative）稳定性，解决了传统方法在非产品结构可行域下的失效问题；③给出了广泛适用的存在性定理，并在三类经典网络模型（合作作者、两向流、公共物品）中给出足够条件与实例。

**🔧 技术方法**

主要技术手段为连续优化与变形法：利用分量式准凹性保证极值存在；引入最大可行增量 μ_e 计算可行区间；采用扰动（perturbation）与颤抖手（trembling‑hand）构造序列，结合固定符号单调性确保极限网络满足稳定条件；并通过构造性示例说明必要性。

**📊 数据集**

未使用外部数据集，全部为理论模型与符号化示例。

**📈 对比分析**

方法与传统的 pairwise 稳定性、Bich–Morhaim 的加权网络稳定性相比，增加了约束后仍能保证存在；相对于 Bich–Teteryatnikova 的完美稳定性，提供了在受约束环境下的可行定义并证明等价性；无实证性能对比，仅在理论上证明存在性与稳健性。

**⚠️ 局限性**

局限性包括：①存在性结果需要连续性、分量式准凹性以及固定符号单调性（或更严格的严格单调性）等假设；②在更一般的非单调或非准凹场景下仍无法保证存在；③对复杂网络结构的数值求解缺乏高效算法与实际案例；④在多团体合作与预算交叉约束下，计算复杂度可能急剧上升。

---

## 620. U.S. State Guidance on AI in K-12 Education: A Comparative Content and Citation Analysis

**arXiv ID:** 2611.06762 | [PDF](https://arxiv.org/pdf/2611.06762v1)

**作者:** Shahan Ali Memon `[一作]` (University of Washington), Nic Weber `[通讯]` (University of Washington)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统比较了36个美国州（含波多黎各）发布的K‑12人工智能使用指导文件，结合内容编码、引用分析与州层面人口、财政、政治等特征进行量化研究。

**💡 创新点**

创新点在于：①首次将指导文件与州人口与资源指标关联检验政策响应性；②利用大型语言模型辅助二进制编码与文本相似度检测，揭示指导文件的知识来源与复制模式；③构建完整的引用网络，展示州际间的“软法”传播与依赖。

**🔧 技术方法**

采用混合方法：手工二进制内容编码、Mann‑Whitney U检验、Jaccard相似度、句子级余弦相似度、文本相似度检测工具（Sentence‑BERT）以及LLM辅助的编码校验。

**📊 数据集**

主要数据集为：36份州AI指导文件（共1,539页），5,351条引用，328个独立来源；州人口与资源数据来自美国人口普查局、NCES、Ballotpedia；政治倾向取自州议会席位比例。

**📈 对比分析**

比较方法：对七大维度进行二进制编码并统计覆盖率；对州特征与维度覆盖进行两侧Mann‑Whitney U检验；构建引用网络并计算Jaccard相似度；使用余弦相似度和共享短语计数评估文本重用。结果显示：数据隐私、学术诚信、算法偏见等议题覆盖率高达97%；而针对农村、部落等边缘群体的关注极低；州特征与指导优先级无显著相关性。

**⚠️ 局限性**

局限性包括：①仅分析最新版本文件，未追踪文件随时间演化；②二进制编码忽略议题深度与细节；③未检验指导对课堂AI实施的实际影响；④部分引用来源可能未完全捕捉，导致网络分析不完整。

---

## 621. Generalized Sorting via Four-Cycles

**arXiv ID:** 2611.06745 | [PDF](https://arxiv.org/pdf/2611.06745v1)

**作者:** Soheil Behnezhad `[一作]` (Northeastern University), Shane Ferrante `[通讯]` (Northeastern University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种广义排序问题，其中只有部分元素对可以比较。给定一个n个顶点和m条边的图G=(V, E)，目标是以尽可能少的查询次数对V进行排序。

**💡 创新点**

提出了一种新的算法，使查询次数最多为O(n^4/3)，突破了长期以来的O(n√(n))界限。

**🔧 技术方法**

使用了随机化算法，主要通过查询属于特定类型的四个循环的边来验证其他边的方向，从而降低查询成本。

**📊 数据集**

使用了具有n个顶点和m条边的图G作为数据集，特别关注稀疏图和密集图的情况。

**📈 对比分析**

与之前的算法相比，新的算法在所有图上都能实现更低的查询复杂度O(n^4/3)，而之前的算法在稀疏图上为O(√(mn))，在密集图上为O(n√(n))。

**⚠️ 局限性**

算法的局限性在于它依赖于图的密度，且在某些情况下可能无法保证最优性能。

---

## 622. Using AI Predictions to Augment Rather Than Replace Surveys

**arXiv ID:** 2611.06743 | [PDF](https://arxiv.org/pdf/2611.06743v1)

**作者:** Lai Wei `[一作]` (University of Hong Kong), Yifei Zhu `[通讯]` (University of Hong Kong)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文研究了如何用AI预测模型（监督学习和大语言模型）与传统概率抽样调查结合，通过预测加权推断（PPI）提升社会调查的统计效率。

**💡 创新点**

创新点在于将外部AI预测作为可校正的辅助信息，而非替代真实受访者，证明在大型公共调查中可实现7-12%有效样本量提升。

**🔧 技术方法**

采用预测加权推断（PPI）、交叉验证监督学习、历史学习、LLM回应模拟，并利用线性和广义线性模型估计。

**📊 数据集**

使用美国GSS 2022、NHIS 2017调查数据作为标签样本，以IPUMS US 2022/2017 ACS微观数据为辅助文件。

**📈 对比分析**

通过比较预测仅、PPI、调查仅三种估计，结果显示监督学习的PPI在均值估计上可获得5-12%置信区间宽度缩减，LLM贡献略低；对回归系数提升更小。

**⚠️ 局限性**

局限在于仅使用ACS可公开的有限人口与社会经济变量，未考虑地理、行政记录等更丰富辅助数据；并且LLM预测的偏差和不确定性仍需进一步校正。

---

## 623. Smartphone-Aided VR Exergames for HIIT: Gamifying Fitness with Real-Time Pose Estimation

**arXiv ID:** 2611.06734 | [PDF](https://arxiv.org/pdf/2611.06734v1)

**作者:** Abhiram Kothagundu `[一作]` (Indian Institute of Information Technology Sri City), Johanna Pirker `[通讯]` (Technical University of Munich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于智能手机摄像头的VR HIIT Exergame系统，通过实时姿态估计实现全身动作跟踪并提供个性化反馈。

**💡 创新点**

创新点在于将手机无标记姿态估计、Model Context Protocol 中间件、LLM 生成教练反馈与VR游戏无缝集成，且无需昂贵的运动捕捉硬件。

**🔧 技术方法**

使用了 MediaPipe BlazePose、MCP、GPT‑4o 大语言模型、Meta Quest 3 VR 与 React Native 手机端等技术。

**📊 数据集**

使用了20名18–25岁参与者的实时视频与运动数据（无公开数据集）。

**📈 对比分析**

通过与传统非VR HIIT 进行配对对照，结果显示 VR 组在享受度（p=0.02, d=0.65）、沉浸感（p=0.03, d=0.58）及推荐意愿（p=0.04, d=0.48）显著提升，MCP 反馈与专家评估达87.5%的一致率。

**⚠️ 局限性**

局限包括样本量小、受限于学生群体、仅单次实验、缺乏生理传感器、智能手机摄像头摆放和照明影响，以及阈值设定的经验性依赖。

---

## 624. Monadically Dependent Graph Classes Have Almost Bounded Merge-Width

**arXiv ID:** 2611.06804 | [PDF](https://arxiv.org/pdf/2611.06804v1)

**作者:** Jan Dreier `[一作]` (HPI Potsdam), Szymon Toruńczyk `[通讯]` (University of Warsaw)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明任意单子依赖图类（monadically dependent）在给定半径 r 的合并宽度（merge‑width）上几乎受限：存在子多项式函数 f(n)，使得每个 n‑顶点图 G 属于该类时，r‑合并宽度不超过 f(n)。并给出了近三次时间的算法，能够在该类图上构造相应的分区合并序列（partition merge‑sequence）。

**💡 创新点**

创新点在于将单子依赖这一逻辑性质与图参数合并宽度相联系，完成了之前的猜想；同时提出并利用了“邻域复杂度（neighborhood complexity）”与“r‑类型（r‑types）”的新工具，结合 VC 维度理论与 Packing Lemma，构建了 Damping Theorem 和 Predamping Lemma 的证明框架。

**🔧 技术方法**

主要技术包括：VC 维度与切分理论、r‑类型的递归定义、近线性邻域复杂度、Damping Theorem、Predamping Lemma、在线网（online nets）、低交错排序（low alternation ordering）、集合系统 Packing Lemma、分区合并宽度（partition merge‑width）的定义及其与原始合并宽度的关系。

**📊 数据集**

本工作完全是理论证明性质，并未使用实验数据集或实际图实例。

**📈 对比分析**

与现有的树宽、拓扑宽、双曲宽、紧致度等图参数相比，本结果表明单子依赖图类在合并宽度上几乎受限，间接实现了一阶模型检验的固定参数可判定（FPT）。由于是纯理论结果，没有实验性能数据。

**⚠️ 局限性**

局限性包括：结果仅适用于单子依赖图类；证明依赖于 VC 维度与近线性邻域复杂度的假设；仅给出了子多项式的上界而非多项式上界；实现算法的时间复杂度为近三次，实际常数可能较大，未给出多项式时间的具体实现。

---

## 625. ConDR: Conservative Reachability through Neural Hamilton-Jacobi Subsolutions

**arXiv ID:** 2611.06802 | [PDF](https://arxiv.org/pdf/2611.06802v1)

**作者:** Kendra Givens `[一作]` (Vanderbilt University), Thomas Beckers `[通讯]` (Ulm University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新的神经网络方法ConDR，用于学习Hamilton–Jacobi可达性（HJ Reachability）的子解，从而得到保守的后向可达管（BRT）。

**💡 创新点**

创新点在于：①构造网络输出保证障碍分支恒为负；②采用margin‑shifted一侧L1损失，使残差在安全边界下方；③加入补全损失防止退化到全为不安全；④理论证明学习到的子解BRT包含真实BRT，并对模型不匹配具有可控的鲁棒性。

**🔧 技术方法**

技术实现包括SIREN隐式神经表示、ExactBC构造、DeepReach与MADR的无监督训练框架、对抗MPC生成标签、one‑sided残差损失与补全损失的组合。

**📊 数据集**

实验使用Air3D、Air6D、9D多车、13D四旋翼等四个系统，部分系统提供了grid‑based ground‑truth，其他则采用采样残差估计与conformal bounds。

**📈 对比分析**

与DeepReach和MADR对比，ConDR在FN率、碰撞率、子解违例率和BRT体积上均表现最佳；通过调节ε可实现对模型不匹配的线性鲁棒性；在高维系统中仍保持较低的计算负担。

**⚠️ 局限性**

局限性包括：仍需大量采样训练，理论保证仅适用于理想的子解解；近似网络可能偏离理想解导致安全误差；对非对抗环境的适用性与更高维度（>13D）系统的可扩展性尚待进一步验证。

---

## 626. Data Center Agglomeration and Power Infrastructure Capacity in the United States

**arXiv ID:** 2611.06737 | [PDF](https://arxiv.org/pdf/2611.06737v1)

**作者:** Yutong Ma `[一作]` `[通讯]` (University College London), Yutong Ma (University College London)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过构建2010-2024年美国县级面板数据，系统评估数据中心扩张对当地新增电力装机容量的因果影响。

**💡 创新点**

创新点在于首次将双重机器学习与事件研究相结合，并引入历史数据中心集聚指数以剔除路径依赖。

**🔧 技术方法**

主要使用双重机器学习（DML）框架、五折交叉拟合、LassoCV等机器学习工具实现高维协变量的去偏估计。

**📊 数据集**

采用的核心数据集包括从商业数据库获得的县级数据中心建筑面积（LDA）与EIA公开的新增电力容量（CAP），并补充人口密度、收入、GDP等社会经济变量。

**📈 对比分析**

与传统OLS线性回归对比，DML方法在控制高维噪声、避免正则化偏差方面表现更好，得到的平均处理效应显著且稳健。

**⚠️ 局限性**

局限性包括：未能区分不同类型数据中心的能耗差异、未拆解新增容量的能源结构以及缺乏电厂退役与跨区域电量流动的数据。

---

## 627. Dexterous Robotic Hands in the Foundation-Model Era: A Full-Stack Survey of Hardware, Data, Cross-Embodiment Transfer, Learning, and Evaluation

**arXiv ID:** 2611.06724 | [PDF](https://arxiv.org/pdf/2611.06724v1)

**作者:** Ji Pei `[一作]` `[通讯]` (Xi'an Jiaotong University), Ji Pei (Xi'an Jiaotong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本综述系统性梳理了 2024 年至 2026 年间可伸缩多指机器人手的技术进展，构建了五层架构（硬件、数据、跨形态迁移、学习与基础模型、评估与应用），并通过对比表、案例分析和定量合成，提出了三条重要结论和 11 个开放问题。

**💡 创新点**

创新点在于：①提出了五层相互依赖的完整体系结构，阐明硬件、数据、跨形态迁移与学习之间的交互；②系统级架构与案例分析将设计选择在各层间可追溯；③基于大量文献的定量合成揭示人类数据价值依赖于运动学/视觉/接触对齐、跨形态共享动作空间仍受力学/时序限制、单一任务完成不足以证明物理可行性；④给出 11 个开放问题并提出可衡量的报告接口和对比方法。

**🔧 技术方法**

综述方法主要是结构化叙事分析与文献筛选；通过对比表格、实验结果汇总、跨论文的定量合成来展示技术演进；未涉及新模型训练或新实验，仅聚焦已有研究的系统总结。

**📊 数据集**

使用的数据集与数据收集系统涵盖：DexCap、EgoDex、DexWild、UniDex、EgoScale、Generalist、Liber‑0、TouchScale 等多种人类手动捕捉、仿真生成、网络视频等来源；综述并未自行收集或训练模型，而是汇总了上述数据集在不同研究中的规模与应用。

**📈 对比分析**

比较方法主要采用对比表格（硬件成本/DoF、数据来源与耗时、跨形态迁移成功率等）和定量合成（如人类预训练小时数与验证损失、不同数据比例对任务完成率的影响）。性能上并未给出统一排行榜，而是呈现各研究在其评估指标下的结果，并指出不同方法间的难以直接比较性。

**⚠️ 局限性**

局限性包括：①非系统综述，未保证覆盖所有相关工作；②不同论文的实验设置、评估基准、硬件规格不一致，导致跨论文对比存在偏差；③大部分结论基于报告的数据和实验，缺乏可复现性与统计显著性；④对长期可靠性、零射击迁移与跨传感器触觉通用性等关键问题的实证仍有限。

---

## 628. The 3-Uniform Hyperclique Hypothesis is False

**arXiv ID:** 2611.06813 | [PDF](https://arxiv.org/pdf/2611.06813v1)

**作者:** Timothy M. Chan `[一作]` (University of Illinois at Urbana Champaign), Yuancheng Yu `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了真正低于四次方时间的算法，用于在3-均匀超图中检测或计数4-超团，进而推导出对所有常数k≥4的k-超团问题以及Max-3-SAT的加速算法。

**💡 创新点**

核心创新在于两种全新证明：一是通过颜色交替4环计数并利用反对称矩阵族实现颜色匹配消除，二是构造单变量多项式与Walsh–Hadamard变换相结合的全局计数公式；两种方法均仅需标准矩阵乘法而不需新的三元乘法技术。

**🔧 技术方法**

采用反对称正交矩阵族、矩阵乘法指数优化、Walsh–Hadamard变换、基于Behrend集合的组合构造，以及矩形矩阵乘法算法。

**📊 数据集**

论文为理论算法研究，未使用具体数据集，而是针对任意规模n的超图给出算法复杂度。

**📈 对比分析**

相较于以往最优的O(n^4)上界，算法在确定性下实现了O(n^{3.957})（第一种方法）和O(n^{3.998})（第二种方法）时间，显著降低了常数与指数；同时也使Max-3-SAT在O(2^{0.99n})时间内可解。

**⚠️ 局限性**

局限性包括对最新矩形矩阵乘法极限的依赖，若矩阵乘法指数不进一步改进将难以突破；算法实现复杂度高，实际应用受限；并且对k-超团的加速仍仅为O(n^{k-k/4})，未达至最优或对更高h的情形不适用。

---

## 629. LLMs Don't Know What They Know: Benchmarking Temporal Self-Knowledge for Financial Backtesting

**arXiv ID:** 2611.06808 | [PDF](https://arxiv.org/pdf/2611.06808v1)

**作者:** Lingfei Qian `[一作]` (Fin AI), Jian-Yun Nie `[通讯]` (University of Montreal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含 2,226 条美国定时财务事实（宏观经济、股价、政策）的基准，并通过行为测试评估大型语言模型在历史回测中的知识边界可靠性。

**💡 创新点**

创新点在于同时检验模型的实际知识、模型自报知识和历史日期提示三种防泄漏措施，首次用行为数据量化知识截止点，并揭示这些措施往往失效；此外提供统一的任务设计、控制方法和阈值规则。

**🔧 技术方法**

采用三类多项选择提示（值、值+认知、带历史日期的预测），配合滚动准确率阈值、偏差控制与均衡准确率等统计手段实现行为边界估计。

**📊 数据集**

使用公开来源数据：FRED 宏观系列、Yahoo Finance 股价、联邦公开市场委员会利率决议。

**📈 对比分析**

对四个前沿模型（GPT‑5、DeepSeek‑V3.2、GPT‑4o、Llama‑3.3‑70B）、两款开放模型（Qwen3‑8B、Qwen3.5‑9B）以及三类点时模型（TiMaGPT、ChronoGPT、DatedGPT）进行比较；发现前沿模型知识准确率可达 70% 以上，但自报知识率低于 40%；历史日期提示几乎不改变准确率却显著降低自报知识；整体表明传统的切断点、自动报知和历史提示均无法有效防止回测泄漏。

**⚠️ 局限性**

局限性包括仅覆盖美国经济与三种金融事实类型；阈值设定基于经验，可能影响边界估计；点时模型规模过小，难以检验大模型在严格切断下的行为；多项选择格式可能不适用于开放式知识表述。

---

## 630. Task-Efficient Agent Evaluation Using Cached Traces

**arXiv ID:** 2611.06803 | [PDF](https://arxiv.org/pdf/2611.06803v1)

**作者:** Ben Johnson `[一作]` (Jataware), Hayden Helm `[通讯]` (Helivan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种利用 LLM 生成任务特定的结构化摘要，对 agentic 系统的执行轨迹进行嵌入，并通过少量任务执行估计系统在完整基准上的性能。

**💡 创新点**

创新点在于提出“trubric”表示法，将轨迹抽象为结构化摘要并嵌入，构建基于轨迹几何的 PKPS 空间，并将其与 IRT 结合，实现更高效、低成本的基准评估。

**🔧 技术方法**

使用的技术包括 LLM（DeepSeek-V4-Flash）生成 rubrics 与摘要、文本编码器（如 OpenAI、BERT、LaBSE 等）进行嵌入、Product Kernel Perspective Space（PKPS）构建系统表示、项反应理论（IRT）建模以及两者的凸组合。

**📊 数据集**

使用的主要数据集为 SWE‑bench Verified（107 系统、100 任务）和 Terminal‑Bench 2.0（65 系统、89 任务）。

**📈 对比分析**

通过与基准平均、样本分数、原始轨迹、通用 rubrics、纯 IRT 等方法对比，trubric+IRT 组合在 1–20 任务预算下 MAE 低至 0.08（SWE‑bench）和 0.12（Terminal‑Bench），显著优于单独方法，并将达到 MAE <0.10 的成本从 10–40 美元降至 2 美元。

**⚠️ 局限性**

局限性包括仅评估五个属性、未对 rubric 字段进行优化、仅使用 DeepSeek-V4-Flash 进行摘要生成、仅在完全配对、已完整执行任务的参考集上实验，且不考虑稀疏或跨基准的引用，生成摘要可能触发 API 限制。

---

## 631. Message-passing graph persistence

**arXiv ID:** 2611.06801 | [PDF](https://arxiv.org/pdf/2611.06801v1)

**作者:** Jorge Luiz Franco `[一作]` (University of Sao Paulo), Amauri Souza `[通讯]` (Avra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新型的持久同调描述子MGP，能够在图过滤过程中动态进行消息传递并将得到的节点嵌入关联到持久对中。

**💡 创新点**

创新点在于将消息传递机制嵌入持久同调的过滤步骤，形成与RePHINE相比更具表达力的描述子，并通过证明其图同构不变性和严格超越RePHINE的表达能力。

**🔧 技术方法**

使用的技术包括结构感知边过滤函数、图神经网络（GNN）消息传递、持久同调（PH）计算、DeepSet向量化以及多层MGP融合到常规GNN架构中。

**📊 数据集**

实验使用了TUDataset中的五个图分类数据集（MUTAG、PTC_MR、PROTEINS、NCI1、NCI109）以及ZINC回归子集。

**📈 对比分析**

与传统GNN、PH、RePHINE、SpectRe等方法进行对比，MGP在大多数数据集上取得最优或接近最优的性能，在ZINC上的MAE最低，在NCI109、PROTEINS等数据集上表现突出。

**⚠️ 局限性**

局限性在于仅针对图结构，未扩展到高阶单纯形或细胞复形；此外，过滤步骤数量的增加会显著提升计算成本。

---

## 632. Ranking Agreement Does Not Reveal Paragraph-Edit Agreement in Two Released Paper Acceptance Predictors

**arXiv ID:** 2611.06789 | [PDF](https://arxiv.org/pdf/2611.06789v1)

**作者:** Shaoliang Yang `[一作]` (Santa Clara University), Jun Wang `[通讯]` (Santa Clara University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在已有论文上使用快速的特征回归分类器（surrogate）与两种公开的接受度预测模型（PaperScorer-3B-Text-OpenReview-ICLR 与 PaperScorer-3B-Text-arXiv）进行对比实验，检验它们在对论文段落重写（检查后保证事实和论断不变）时是否产生相似的评分变化，并评估使用 surrogate 选择改写是否能显著提升预测模型的分数。

**💡 创新点**

验证了“原论文得分一致性”并不等价于“改写得分一致性”，并首次量化了快速分类器在段落级改写上的“编辑相关性”与“搜索增益”几乎为零的现象，提醒在实际编辑工具中必须针对目标评估者直接验证改写效果。

**🔧 技术方法**

技术包括：1）使用 TF‑IDF 词频 + 文档统计的逻辑回归 surrogate；2）基于 Qwen2.5-3B-Instruct 微调的 3B 参数文本接受度预测模型；3）GPT‑6‑Luna 自动生成可保事实的段落改写；4）GPT‑6.1‑Sol 与 Claude‑Opus 5.5 进行自动语义与不变性检查；5）Spearman 相关系数、Bootstrap 置信区间、adaptive vs. random 搜索对比等统计方法。

**📊 数据集**

数据集：从 ICLR 2026 发布的 1,667 论文中抽取 198 篇用于编辑评估（每篇至少 5 个可接受改写），200 篇用于单段改写实验，128 篇用于搜索实验；使用 PaperScorer 公开的文本数据和权重；所有改写均通过 GPT‑6‑Luna 自动生成并经过检查后冻结。

**📈 对比分析**

比较方法：1）对每篇论文内的可接受改写，计算 surrogate 与每个目标模型的分数变化排名，求 Spearman 相关；2）在 adaptive 搜索与随机搜索下，比较两者对目标模型分数的增益（最大 512 次 surrogate 评估）。性能：相关系数均约为零（-0.01~-0.02），adaptive 搜索对目标模型的平均增益在 32‑bit 精度下也不显著（<0.01 SD），在 bfloat16 下为 0.02 SD；相比随机搜索，使用 surrogate 或同级模型选择改写没有任何可观优势。

**⚠️ 局限性**

局限性：仅测试了极少数模型（一个 sparse surrogate、一个 dense surrogate 以及两种相同基础模型的 fine‑tune）；只考虑段落级、无内容变化的“呈现”改写；未评估改写对科研质量或真实接受率的影响；使用的检查器可能漏检细微内容改变；数值精度（bfloat16 vs. 32‑bit）对结果有显著影响；实验缺乏公开的预注册与跨模型泛化验证。

---

## 633. Integrated Identity Protection for Semantic Communications in Embodied IoT Systems

**arXiv ID:** 2611.06788 | [PDF](https://arxiv.org/pdf/2611.06788v1)

**作者:** Yu Zhang `[一作]` (Xidian University), Xiaohong Jiang `[通讯]` (Future University Hakodate)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种整合语义通信与物理身份识别的安全框架，通过双分支机制实现源合法性验证与身份隐蔽，提升任务决策的可信度与隐私保护；

**💡 创新点**

创新点在于：①将语义信息与物理身份信息视为同一信号的内在载体，实现观测、处理、决策层的联合利用；②针对不同保护目标（身份认证与身份隐蔽）设计双分支策略，兼顾语义可靠性与身份可信度；③在身份隐蔽模式下利用任务可容忍的波形空间，协调语义恢复与RF特征抑制；

**🔧 技术方法**

主要技术包括：深度JSCC编码/解码、RF硬件特征提取（CFO、I/Q失配等）、通道状态信息（CSI）建模、深度学习身份识别器、语义置信度评估、基于t‑SNE的特征可视化以及波形处理模块用于身份隐蔽；

**📊 数据集**

数据集：使用模拟的图像语义通信数据，包含五个合法传输源与一个攻击源，图像通过深度JSCC映射到信号；实验中采用 600 条RF样本做身份分类，1500 条合法样本与 300 条攻击样本做身份认证评估；

**📈 对比分析**

比较方法：在身份认证任务中与单纯语义恢复和单独身份识别两种基线对比；结果显示身份认证联合策略在 20 dB SNR 下实现 100% 合法接受、0% 误接受，身份识别准确率达 99.67%；在身份隐蔽任务中与原始传输、随机扰动对比；结果显示将身份识别准确率从 93.50% 降至 22.33%，同时 PSNR 维持在 24.09 dB（原 24.08 dB），语义恢复质量无显著损失；

**⚠️ 局限性**

局限性：仅在模拟环境和有限数量（6 个）设备上验证，未考虑多路径和多频段实际信道；实验仅聚焦图像任务，其他类型语义内容需进一步验证；隐蔽策略对不同任务容忍度的通用性和实时实现复杂度仍待研究。

---

## 634. Go-Go-Go: the game of Go without turns

**arXiv ID:** 2611.06786 | [PDF](https://arxiv.org/pdf/2611.06786v1)

**作者:** Andrea Cambiaghi `[一作]`, Marco Tarini `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了一个无轮次、时间连续的围棋变体，在其中玩家可异步放置棋子，并通过基于空间距离的延迟机制控制棋子落子频率，保持了原围棋的核心规则和策略深度。

**💡 创新点**

创新点在于彻底去除时间离散化（即轮次），引入“临时不可下子区域”与可变距离延迟，既避免了连击式“刷子”玩法，又保留了原围棋的连通、劫、劫战等复杂策略；同时提出了可预订（booking）机制与实时可视化反馈，使异步玩法更具可玩性。

**🔧 技术方法**

主要技术包括：基于浏览器/服务器的实时网络同步（采用与实时游戏相似的算法）、时间-空间延迟计算（半径随时间线性收缩）、可视化动画（灰色区域、锥形劫区、石子摇摆等），以及客户端的“预订优先”交互功能。

**📊 数据集**

论文未使用传统数据集，所有实验均基于人类玩家的对局测试（包括快速与中速对局），并通过观察和简易问卷收集玩家体验反馈。

**📈 对比分析**

比较方法：通过玩家对局观察是否出现“刷子”现象、策略多样性与节奏快慢的平衡，并与传统围棋对局在玩法流畅度与战略深度上的主观评价做对比。结果表明，虽然节奏更快，但大部分核心策略（眼、劫、劫争、包围等）基本保持；但在极快节奏下部分玩家可能倾向于简单的连击或过度刷子。

**⚠️ 局限性**

局限性包括：1) 缺乏大规模用户测试，难以量化游戏平衡与乐趣；2) AI 对手尚未研究，导致游戏体验依赖人类玩家；3) 延迟机制可能被玩家利用或导致“不可下子”区域过大/过小，影响公平性；4) 仍需进一步调参以适配不同水平玩家的节奏偏好。

---

## 635. FedFiOT: Fisher-Weighted Optimal Transport for Model Merging in One-Shot and Few-Round Federated Learning

**arXiv ID:** 2611.06783 | [PDF](https://arxiv.org/pdf/2611.06783v1)

**作者:** Quan Minh Nguyen `[一作]` (University of Florida), My T. Thai `[通讯]` (University of Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FedFiOT，一种联合神经元对齐与Fisher加权最优传输的模型合并框架，用于一射和少轮联邦学习

**💡 创新点**

将Fisher信息引入对齐与聚合的联合优化，构建一个可迭代下降的层级合并目标，并给出收敛和稳态理论保证

**🔧 技术方法**

Fisher信息估计、最优传输（Hungarian算法）、交替对齐与加权平均、理论分析（凸性、稳态分析）

**📊 数据集**

CIFAR‑10、SVHN、CINIC‑10（使用Dirichlet分布模拟非IID）

**📈 对比分析**

与FedAvg、PFNM、OTFusion、WB Fusion、DENSE、FisherMerge、RegMean、FedFisher、FedBEns等九个基线对比；一射场景下平均提升3.3–7.6个百分点；少轮场景下四轮后仍保持优势，最高达66.7%（CIFAR‑10）

**⚠️ 局限性**

需额外上传Fisher矩阵导致通信量翻倍；对Fisher估计精度敏感；在高对齐的多轮场景下优势减小

---

## 636. Benchmarking Fine-Grained Visual Reasoning for Remote Sensing in Urban Management

**arXiv ID:** 2611.06770 | [PDF](https://arxiv.org/pdf/2611.06770v1)

**作者:** Jiawei Zhang `[一作]` (Harbin Institute of Technology), Dayan Guan `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了UMBench基准和无训练的证据优先链式思考方法，用于评估和提升多模态大型语言模型在城市遥感细粒度视觉推理上的表现。

**💡 创新点**

①创建首个针对城市管理任务的细粒度遥感推理基准；②提出训练‑free 的证据优先链式思考提示，引导模型先检验细粒度视觉证据。

**🔧 技术方法**

采用多模态大型语言模型（Qwen3‑VL、InternVL、Gemini‑3‑Flash 等）并设计基于提示的证据优先链式思考技术。

**📊 数据集**

UMBench，包含 583 张 UAV 图像与 2733 个多任务（交通、建设、环境、灾害）细粒度问答对。

**📈 对比分析**

在本地部署与云端服务两种访问设置下进行零样本比较，使用多选答案准确率。证据优先 CoT 相较于直接回答和通用 CoT 提升平均准确率约 3–10%（依模型和任务而异）。

**⚠️ 局限性**

方法仍受限于模型对细粒度视觉识别的能力，提示未能完全消除对场景先验的依赖，且在某些任务（如环境质量）提升有限。

---

## 637. PyRUA-REAL: Robot Exploration and Adaptive Learning through RSI in the Real World

**arXiv ID:** 2611.06768 | [PDF](https://arxiv.org/pdf/2611.06768v1)

**作者:** Qiang Wang `[一作]` (Peking University), Daquan Zhou `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用大型语言模型驱动的编程代理，在真实双臂Franka机器人平台上进行物理交互探索，迭代生成并改进可执行的机器人程序，形成从首次成功到持续改进的闭环递归自我提升流程。

**💡 创新点**

提出基于经验记忆与可执行技能库的 PyRUA-REAL 框架，体现了研究者与工程师双代理协作、经验分层存储、持续改进以及部署时无模型调用的可执行程序。

**🔧 技术方法**

依赖 GPT‑6‑Astra、SigLIP、SAM 等大型语言模型与视觉模型，配合 Python 编程、机器人控制框架和多种执行反馈（视觉、状态、视频）实现。

**📊 数据集**

使用双臂Franka Research 3 + Robotiq‑85 机器人平台上的八项实操任务（跳棋、五块堆叠、汉诺塔、以太网插拔、毛巾折叠、推“T”、块分类、块收集），以及 RoboDojo 42 个仿真任务。

**📈 对比分析**

与 PyRUA‑Lean、GPT‑6‑Astra 等基线对比，PyRUA‑REAL 冻结程序在 RoboDojo 上平均成功率 84.42%（相对 50.35% 与 22.48%），在精度、长时域任务上表现突出；在实际机器人上开发成功率 91.1%，程序执行时间显著低于在线编码，表明部署效率提升。

**⚠️ 局限性**

局限包括：仍需人工干预或在线代理辅助以完成高精度对接等任务；经验迁移效果不一；评估多为单实验场景，缺乏跨实验验证；系统对方法多样性和方法选择的依赖仍需改进。

---

## 638. Fixed-Radius Neighbourhoods in Three-Dimensional Unit-Cube Graphs

**arXiv ID:** 2611.06747 | [PDF](https://arxiv.org/pdf/2611.06747v1)

**作者:** Guangjian Zhang `[一作]` `[通讯]`, Guangjian Zhang

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在三维单位立方体图中，证明任意固定半径 r 的邻域族 VC‑维度是有界的，并给出上界 2^2^O(r)；基于此，提供了一个期望时间为 O_Δ(n^{2-ε_Δ}) 的 Las Vegas 算法，用于判定给定图的直径是否不超过任意固定阈值 Δ；同时构造了一个三维单位立方体图，使其半径 r 邻域族 VC‑维度达到 Ω(r^3)，给出与半径的最优依赖关系。

**💡 创新点**

主要创新在于：① 通过对三维中任意固定长度的“广义支配关系”进行可达性分析，得到新的 VC‑维度上界；② 将该上界与间隔表示框架结合，得到适用于所有固定阈值的真正亚二次直径判定算法；③ 构造了基于三维网格树的立方体图，实现半径 r 的 VC‑维度下界 Ω(r^3)，从而证明了半径依赖的必要性。

**🔧 技术方法**

使用的技术包括：层次系统与有序群的可达性归纳法、Sauer–Shelah–Perles 计数、Erdős–Szekeres 单调子序列、区间表示（interval representation）与低交叉排序（low‑crossing orders）、弱支配关系的离散化与区间并集计算、以及基于线性映射的立方体坐标构造。

**📊 数据集**

论文不依赖于真实数据集，而是在理论上考虑所有可能的单位立方体图（即任意有限点集在 ℝ^3 中的闭轴对齐单位立方体）。

**📈 对比分析**

与之前仅对半径 2、3 进行分析的工作相比，本研究将结果推广到所有固定半径和阈值。算法在最坏情况下的期望时间为 O_Δ(n^{2-ε_Δ})，其中 ε_Δ>0 明确给出（如 Δ=4 时 ε≈2^{-332}），显著优于先前的 O(n^2) 或仅适用于小半径的结果。

**⚠️ 局限性**

局限性在于 VC‑维度上界呈现双指数增长 2^2^O(r)，远高于下界 Ω(r^3)。是否可以将上界降至多项式仍是开放问题；此外，算法虽然是 Las Vegas，但实现复杂度较高，且在实际大规模数据中需要进一步优化。

---

## 639. Template Gap Reconstruction for Shell Output Simulation with Small Language Models

**arXiv ID:** 2611.06731 | [PDF](https://arxiv.org/pdf/2611.06731v1)

**作者:** Jarrod Ragsdale `[一作]` (University of Texas at San Antonio), Rajendra Boppana `[通讯]` (University of Texas at San Antonio)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了模板间隙重建 (TGR) 方法，利用从 18.7 M 条跨 25 个环境的 shell 记录中提取的可重用输出模板和占位符，减少小型语言模型在无执行环境下生成完整 shell 输出的预测负担。

**💡 创新点**

创新点在于：1）设计可检索的环境不变输出模板与贪婪 LCS 构造；2）在同一模型权重下实现 FOD 与 TGR 的联合训练；3）使用输入级模板选择器来决定是否使用模板，从而显著提升输出相似度并缩短生成长度。

**🔧 技术方法**

采用贪婪最长公共子序列 (LCS) 方法构造模板，使用占位符分隔的间隙生成目标，利用 Transformer 编码器+多头线性分类器进行模板选择，训练时统一使用 CodeT5、ByT5、Qwen3 等小型编码器–解码器/解码器模型。

**📊 数据集**

数据集为 18.7 M 条跨 25 个文件系统环境的 shell 执行日志，用于构建模板库；随后使用 1 M 条来自单一目标环境的记录进行模板选择和间隙生成的训练。

**📈 对比分析**

在 1 k 个测试命令上比较 Full-Output Decoding (FOD)、选取 TGR 和 Oracle TGR，评估指标为 O‑ES、BLEU‑4、ROUGE‑L；结果显示选取 TGR 在模板覆盖率>50%时平均 O‑ES 提升约 0.019，Oracle TGR 进一步提升约 0.016，且在短间隙（≤64 字符）下提升尤为显著。

**⚠️ 局限性**

局限性包括模板选择误选率约 4%，导致部分输出下降；间隙长度较长时 TGR 与 FOD 的性能趋同；方法仅针对单次输入，未考虑多轮交互与文件系统状态变化。

---

## 640. Symmetry Breaking at the Edge of Stability via Central Flow Analysis

**arXiv ID:** 2611.06718 | [PDF](https://arxiv.org/pdf/2611.06718v1)

**作者:** Naima Elosegui Borras `[一作]`, Georgios Arvanitidis `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究梯度流中的连续对称性守恒律在边缘稳定性（EOS）下通过中心流演化时的破坏与存活。

**💡 创新点**

首次从中心流角度推导出对称性破坏矩阵，并阐明哪些守恒律会被破坏，哪些得以保持，提供了对EOS下守恒律的系统分类。

**🔧 技术方法**

采用中心流描述、Hessian 分解、对称性生成器、投影算子等理论工具，对三种两层网络（标量、矩阵分解、ReLU）进行解析。

**📊 数据集**

未使用实际数据集，全部基于理论模型和数值实验验证。

**📈 对比分析**

通过对比梯度流、离散梯度下降与中心流的演化轨迹，展示守恒律在不同动态下的存活与破坏情况，表明中心流能准确预测EOS阶段的对称性破坏。

**⚠️ 局限性**

尚未给出完整计数与完备性证明，缺乏对不同轨迹下新出现守恒律的系统性分析。

---

## 641. Reasoning over Non-Parametric Memory in Multilingual LLMs: A Cognitive-Based Analysis of Relational Knowledge

**arXiv ID:** 2611.06715 | [PDF](https://arxiv.org/pdf/2611.06715v1)

**作者:** Zohreh Abbasi Liasi `[一作]` (Qatar Computing Research Institute, Hamad Bin Khalifa University), Ehsaneddin Asgari `[通讯]` (Qatar Computing Research Institute, Hamad Bin Khalifa University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了多语种诊断基准Rel‑Bloom，并在43个模型上评估其在给定关系事实下的推理能力。

**💡 创新点**

创新之处在于将Bloom修订分类与关系事实的真伪条件（真实、反事实、无实体）以及多语言匹配相结合，系统探究模型在不同认知层级和知识条件下的行为。

**🔧 技术方法**

使用多语言多模型评估框架、关系级别簇统计、关系使用频率分层、配对差异的关系聚类符号检验和Spearman相关性等统计方法。

**📊 数据集**

以30条从Wikidata筛选的关系实例为基础，构成450个英文多项选择题，并翻译成阿拉伯语、中文、印地语、波斯语和西班牙语。

**📈 对比分析**

对模型规模、Bloom层级、真伪条件、关系使用频率和语言进行分组统计与比较，发现大多数模型在Apply到Analyze层级会出现显著性能下滑，Evaluate层级仍是瓶颈；大模型规模提升总体准确率但并非完全消除问题。

**⚠️ 局限性**

局限性在于每个关系仅使用单一事实实例，限制了泛化能力；未评估检索噪声、开放式生成等实际RAG系统；部分结论可能受题目构造和近似方法影响。

---

## 642. Where Is VLA Inference Time Spent? Bottlenecks and Optimization Interactions Across GPUs

**arXiv ID:** 2611.06710 | [PDF](https://arxiv.org/pdf/2611.06710v1)

**作者:** Muhammad Danish Waseem `[一作]` (Chalmer University and University of Gothenburg), Ahmed Ali-Eldin Hassan `[通讯]` (Chalmer University and University of Gothenburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过对七种视觉‑语言‑动作（VLA）模型在四种GPU平台（H100、A10G、RTX 4090、Jetson Thor）上进行系统测量，评估七类推理优化（CUDA Graphs、编译、融合注意力、批处理、步骤减少、标记裁剪、量化）以及它们之间的相互影响，提出VLA‑Triage指导方案；

**💡 创新点**

创新点在于：①首次对多模型、多硬件多优化方案的交互效益进行配对测量；②揭示不同优化在不同平台上可能从加速变为减速；③基于测量结果制定可落地的推理堆叠指南；

**🔧 技术方法**

使用的技术包括CUDA Graphs、JIT编译（Torch Compile）、FlashAttention‑2、Token裁剪、FP8/Int8量化、流式步数减少以及多线程CPU调度与GPU内存池管理；

**📊 数据集**

评估数据主要来自七个公开VLA模型（OpenVLA、OpenVLA‑OFT、LLaVA‑VLA、CogACT、π_0.5、GR00T、SmolVLA），未使用新的数据集，而是直接跑这些模型的标准推理工作；

**📈 对比分析**

通过在同一平台上对基线与单独优化、组合优化进行对比，发现单独测得的加速比例在组合后往往不成立，尤其编译后裁剪或量化收益显著下降，最终提供了不同GPU上实际加速幅度和交互情况的完整表格；

**⚠️ 局限性**

局限性包括：①实验仅覆盖四种GPU，缺乏CPU/嵌入式NPU等其他硬件；②未考虑模型准确性损失，仅关注延迟；③在实际部署中需自行验证裁剪/量化后的性能与精度；

---

## 643. SAUCE: Summary Analysis Using Conversation Entailment

**arXiv ID:** 2611.06812 | [PDF](https://arxiv.org/pdf/2611.06812v1)

**作者:** Man-Ling Sung `[一作]` (RTX BBN Technologies), Matthew Snover `[通讯]` (RTX BBN Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SAUCE 评估流程，用于跨语言对话语音摘要的无参考、基于事实的质量评估。

**💡 创新点**

创新点在于将摘要与对话拆分为原子事实并利用文本蕴含同时量化可信度与覆盖率，并提供可追溯的事实账本。

**🔧 技术方法**

使用 GPT‑4o 提取事实，DeBERTa‑v3 进行文本蕴含判定，结合阈值和块化对话进行评估。

**📊 数据集**

采用多语言 CTS 数据集（阿拉伯语、西班牙语、普通话、俄语）的 CallHome、Fisher、Hub5、Mixer3 等，并在 ASR+MT 级别引入 Whisper‑large‑v3 与 NLLB‑200‑distilled。

**📈 对比分析**

与 SummaC、FENICE、G‑EVAL、FineSurE 等指标对比，SAUCE 在 ASR/MT 错误下能明显捕捉准确性下降，覆盖率与信度与人类评估高度相关，且能识别“破损”摘要。

**⚠️ 局限性**

限制包括 DeBERTa‑v3 未针对会话专门微调，GPT‑4o 成本高且易漂移，事实提取的稳定性和偏差仍需改进，实验样本量和人类标注有限。

---

## 644. VA-TDM: A Unified View for Improving Few-Step Diffusion Models by Velocity Adversarial Distillation

**arXiv ID:** 2611.06796 | [PDF](https://arxiv.org/pdf/2611.06796v1)

**作者:** Yihong Luo `[一作]`, Tianyang Hu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Velocity Adversarial Distillation (VAD) 并实现 VA-TDM，对预训练的扩散/流模型进行四步蒸馏，保持高质量生成。

**💡 创新点**

创新点在于把速度场对齐建模为对抗游戏，统一 score distillation、Fisher 与 moment 匹配；设计 A‑form 与 B‑form 两种速度目标并联合使用；引入过渡感知的在线速度学习提升中间步监督。

**🔧 技术方法**

使用速度场对抗学习、逆 KL / Fisher 降维、MMD 目标推导、ODE 回归预热、共享骨干网络、噪声注入路径设计等技术。

**📊 数据集**

实验数据集包括 SD3.5‑Medium、FLUX.2‑Klein‑Base 4B 以及 33B MiniMax‑H3，使用 DrawBench、用户评测等评估平台。

**📈 对比分析**

与 DMD2、D‑DMD、DPDMD、TDM 以及官方四步模型对比；在 SD3.5‑Medium 上 GenEval 提升至 0.67、OCR 0.59、LPIPS 0.56；在 FLUX 上提升至 0.80 GenEval、0.69 OCR、LPIPS 0.56；在 MiniMax‑H3 的人类评估中以 61.33% 超越 Fast‑H3。

**⚠️ 局限性**

局限性包括对抗训练导致一定的不稳定性，对非高斯前向过程的推广受限，极低步数（如 2 步）下性能不如四步；微调大模型时仍有显著的计算成本。

---

## 645. Transforming Integer Grids to Spatial Intuition: Perception, Not Planning, Governs ARC-AGI-3

**arXiv ID:** 2611.06793 | [PDF](https://arxiv.org/pdf/2611.06793v1)

**作者:** Harsidak Singh Banwait `[一作]` `[通讯]` (Independent Researcher), Harsidak Singh Banwait (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Sarbloh 架构，通过自主感知引擎与程序化执行实现对 ARC‑AGI‑3 互动游戏的超人类动作效率；

**💡 创新点**

核心创新在于“Measured‑Guessed Rule”，将几何不变量作为确定事实、将隐含功能归为可验证假设，并实现零参数 O(HW) 的感知引擎与单步骤验证；

**🔧 技术方法**

使用了 host‑side NumPy/Pillow 感知、Python 代码执行（await play）与 Open‑Weight LLM（Qwen3.8‑Flash‑Next）以及 SGLang 伪装解码与前缀缓存；

**📊 数据集**

评估数据集为 ARC‑AGI‑3 官方隐藏评测集（23 个未公开游戏）及 25 个公开游戏；

**📈 对比分析**

与首个里程碑（Duck Harness）以及多家 frontier API 基线（GPT‑5、Claude Opus 等）对比，Sarbloh 在公开游戏中实现 72.5% 交叉均值、在隐藏集上取得 16.53 分（比首里程碑提升 13.7×），在单 GPU 9 小时预算下完成 143/183 关卡；

**⚠️ 局限性**

局限包括对隐含功能假设仍需单动作验证，感知引擎不具完整贝叶斯更新，复杂感知转换仍可能导致误判，且未针对特定任务进行细调。

---

## 646. Finding full immersions is fixed-parameter tractable in labeled tour graphs

**arXiv ID:** 2611.06776 | [PDF](https://arxiv.org/pdf/2611.06776v1)

**作者:** Hadas Barabash `[一作]` (Georgia Institute of Technology), Paul Wollan `[通讯]` (Sapienza University of Rome)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个FPT算法，能够在多项式时间内判定任意标签化巡回图中是否存在与给定巡回图全浸入的同构；同时给出了针对欧拉图的算法化最小-最大定理，为后续的无关顶点技术提供了理论支持。

**💡 创新点**

创新点主要体现在：①首次给出了欧拉图根环分解中最大奇长回路数的算法化最小-最大定理；②将巡回图完整浸入问题与稠密图的无关顶点技术结合，构造了新的无关顶点定理；③基于上述最小-最大定理实现了对k‑apex圆图顶点最小子图的FPT测试。

**🔧 技术方法**

采用的技术包括：Eulerian circuit flooding 与证书对偶、自由边与签名的动态规划、2‑edge cut folio 的组合与动态规划、树宽/秩宽的树分解、MSO 逻辑与 Courcelle 定理、分支消除与组合 Flooding 的字典维护等。

**📊 数据集**

由于本研究为理论算法，未使用具体数据集，所有结果均为理论复杂度分析。

**📈 对比分析**

与以往仅给出 O(ℋ(n^ℓ+8)) 运行时间的完整浸入算法相比，本文提供了 O(|ℋ| n^5) 的 FPT 运行时间；对于 k‑apex 圆图顶点最小子图测试也得到 O(|ℋ| n^5) 的复杂度，并通过树宽/秩宽分解进一步优化。

**⚠️ 局限性**

局限性包括：仅适用于 4‑正则标签化巡回图；对 2‑edge cut 需要 folio 预处理，常数因素较大；算法的高阶多项式（n^5）在实际应用中可能不够高效；对一般顶点最小子图问题仍未实现完整的 NP‑完整性突破。

---

## 647. Approximation and Hardness for the Data Validation Problem

**arXiv ID:** 2611.06773 | [PDF](https://arxiv.org/pdf/2611.06773v1)

**作者:** Antoine Tinguely `[一作]` (Technical University of Munich), G. Wessel van der Heijden `[通讯]` (Technical University of Munich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对“数据验证问题”(Data Validation Problem, DVP)进行理论分析，证明其在一般情形下为强NP‑难且W[1]‑难，并在参数化预算T下给出近似保真归约；随后给出若干近似算法（均值/统一概率情形下的m‑近似、一般情况下的2T/p_min^T近似，以及在全序（tournament）图中的FPT算法），并通过最小割、动态规划等技术实现。

**💡 创新点**

创新点在于首次给出DVP的硬度证明（强NP难、W[1]难、近似保真归约至Densest k-Subgraph），以及构造可行的全局近似算法，并在特殊结构（全序图）上实现FPT解法，填补了该问题从经验方法到理论分析的空白。

**🔧 技术方法**

使用的技术包括：从k-Clique与Densest k-Subgraph的归约、证明可取得集的可达性等价于最小割、动态规划求解最优策略、贪心选择按收益/成本比、线性规划松弛与分数解取整、以及对tournament图的块分解与递归分治。

**📊 数据集**

本文未使用具体数据集，全部研究以理论实例与合成图为主，聚焦于复杂度与算法性能分析。

**📈 对比分析**

与其他方法比较主要基于理论下的近似因子与复杂度；例如m‑近似与传统贪心方法对比显示更好的下界；FPT算法在预算T较小时具备指数级的可行性；总体性能由理论证明的近似比与计算复杂度决定。

**⚠️ 局限性**

局限性包括：算法仍以理论复杂度为主，缺乏实际实验验证；对预算T较大时仍为指数复杂度；对非传递图或存在p_e=0的情况需特殊处理；且目前的近似比对真实应用可能仍显保守。

---

## 648. ConceptARC at Scale: Studying Conceptual Understanding in Reasoning Models

**arXiv ID:** 2611.06732 | [PDF](https://arxiv.org/pdf/2611.06732v1)

**作者:** Philipp Mondorf `[一作]` (LMU Munich), Todd M. Gureckis `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们构建了ConceptARC-at-Scale生成器，并基于此生成了ConceptARC-100数据集，包含1600个围绕16个概念的ARC风格任务，并评估了多种大型推理模型和从零训练的模型。

**💡 创新点**

首次实现大规模自动化的ARC任务生成，覆盖ConceptARC的16个概念，并系统分析模型内部表示，发现模型按操作而非概念聚类。

**🔧 技术方法**

利用领域特定语言DSL进行布局、选择器和动作组合生成任务，评估LLM（Qwen3.5、Gemma-4、GPT-OSS等）以及VARC-ViT-18M和Tiny Recursive Model-7M，并在隐藏层提取任务表示进行聚类分析。

**📊 数据集**

ConceptARC-100（1600个任务，3个支持例和10个测试例），以及原始ConceptARC数据集用于对比。

**📈 对比分析**

通过在每个概念上计算准确率，发现大模型在空间概念上表现良好（最高约83%），但在比较/计数概念上低至<40%；在全部测试例全对时最高仅23%，表明一致性差。

**⚠️ 局限性**

分析仅为相关性，未证明模型行为因表示导致；生成的计数任务可能偏差，且部分模型训练缺乏足够支持样本。

---

## 649. Energy Efficient Near-Threshold Voltage Massive MIMO Computing

**arXiv ID:** 2611.06772 | [PDF](https://arxiv.org/pdf/2611.06772v1)

**作者:** MD Salman Sakib `[一作]` (Oklahoma State University), Shahriar Shahabuddin `[通讯]` (Oklahoma State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在极限阈值电压下，将算法基容错（ABFT）融入大规模MIMO检测器，构建跨层可靠性评估框架。

**💡 创新点**

创新点在于将晶体管级PSpice可靠性特征映射为16位易失性轮廓，并结合工作负载感知的故障注入与仅在最终输出的校验和ABFT，实现低开销且高效的错误检测。

**🔧 技术方法**

采用SPICE级CMOS模型、Monte Carlo变异、Python脚本实现的故障注入框架、基于校验和的ABFT、以及复杂度计数工具。

**📊 数据集**

使用64×16基站天线、64用户、64-QAM调制、15 dB信噪比的模拟数据集，覆盖从0.3 V至1.0 V的供电电压。

**📈 对比分析**

通过对GS、ADMIN、CG三种迭代检测器在同一电压与负载条件下的BER、错误检测率和能耗进行对比，显示在VDD≈0.55 V时误码率提升<1.5%，能耗可降低约70%，ABFT开销<2%。

**⚠️ 局限性**

局限性包括使用无工艺节点的Level‑1 CMOS模型导致能耗估计不具备工艺精度，ABFT仅在最终输出检查，未检测中间状态错误，并且在更低电压下仍出现少量静默错误。

---

## 650. Rethinking Access Control for AI Agents: Recent Advances and Open Challenges

**arXiv ID:** 2611.06764 | [PDF](https://arxiv.org/pdf/2611.06764v1)

**作者:** Chijin Zhou `[一作]` (East China Normal University), Quan Zhang `[通讯]` (East China Normal University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了近期针对大语言模型代理的访问控制技术，归纳为四类机制：策略制定与推导、身份与委托、动作与资源执行、信息流控制，并在此基础上重新思考授权的四个关键方面。

**💡 创新点**

创新点在于：①将授权视为从计划选择到执行过程的动态链条；②区分“上下文证据”仅用于决策与“授权扩展”的区别；③提出授权在系统界面间的意义保持问题；④强调授权状态与执行结果的协同管理。

**🔧 技术方法**

使用的技术主要是现有研究中的方法，如 MiniScope、Prudentia、CaMeL、Fides、SkillGuard、SAGA 等，结合对策略、身份、执行和信息流四个维度的理论建模与评估框架。

**📊 数据集**

本工作为综述性质，并未使用特定数据集；所讨论的实验与案例均来自已有文献中的原始研究。

**📈 对比分析**

由于是综述与理论分析，未进行新的实验对比；作者通过对比现有方法的控制对象、执行边界与假设，构建了一个评估准则框架，但未给出数值性能结果。

**⚠️ 局限性**

局限性包括：①缺乏统一的基准与评测环境；②在多任务、多工具、多委托场景下的具体实现细节仍不完整；③对动态撤销与未决结果的处理仍处于理论探讨阶段；④综述聚焦已有方法，未提供新的实验验证。

---

## 651. A Unified Framework for Extending Centrality Measures to Groups

**arXiv ID:** 2611.06754 | [PDF](https://arxiv.org/pdf/2611.06754v1)

**作者:** Piotr Kępczyński `[一作]` (University of Warsaw), Oskar Skibski `[通讯]` (University of Warsaw)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套统一的SPEC框架，将传统节点中心性扩展到群体，并定义四种聚合规则（Sum、Max、Min、Proportional）来计算群体中心性。

**💡 创新点**

创新点在于给出一种通用的SPEC表示，使多类中心性能够统一表示并系统地推导出四种群体扩展；同时引入比例聚合反映分布式收益。

**🔧 技术方法**

技术上使用了SPEC表示法、四种聚合函数以及对节点、节点对、路径、子图等评估器的定义，并通过实验评估不同聚合对群体选择的影响。

**📊 数据集**

数据集为基于随机几何图的合成图（圆形、双圆、矩形分布），以及示例图和实验图生成。

**📈 对比分析**

通过在生成的1000个随机图中比较不同聚合下的最优群体分布，发现Max聚合往往产生更分散的选择，Sum/Min更集中，Prop介于两者；实验表明聚合规则显著影响中心性排序。

**⚠️ 局限性**

局限在于仅考虑无向未加权图；SPEC表示对某些中心性难以构造；实验仅在合成图上验证，缺乏真实网络的应用验证。

---

## 652. Coupled Scene Graph Anticipation: When With-Constraint Recall Hides Structural Inconsistency

**arXiv ID:** 2611.06738 | [PDF](https://arxiv.org/pdf/2611.06738v1)

**作者:** Zeno Nongmaithem `[一作]` (Jawaharlal Nehru University), Rajesh P. Barnwal `[通讯]` (CSIR-Central Mechanical Engineering Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 CoupledSGA 模型，利用关系交互图（RIG）在 Neural ODE 里耦合关系轨迹，并加入结构一致性损失；

**💡 创新点**

创新点在于通过图注意力耦合关系并通过互斥谓词矩阵约束结构一致性，同时揭示 With‑Constraint 评估协议无法反映结构一致性提升；

**🔧 技术方法**

使用 Graph Neural ODE、GATv2 注意力、Sigmoid/Softmax 分类头以及结构一致性损失；

**📊 数据集**

使用 Action Genome 数据集的 Predicate Classification 任务；

**📈 对比分析**

与基线 SceneSayerODE 进行匹配预算比较，在 With‑Constraint 召回率上略逊（R@20 40.48% vs 44.78%），但在 No‑Constraint 条件下结构一致性（矛盾率）显著下降（从 45.92% 降至 4.07%）；

**⚠️ 局限性**

局限性包括：训练时间短（仅四个 epoch）、模型容量未充分利用、只在 Action Genome 子集上验证、评估只考虑前两名矛盾率、排除矩阵手工构造且可能不完整、未探究更高阶解算器和正则化形式等。

---

## 653. Turns Matter: Advancing Streaming Video RL via Turn-Level Credit

**arXiv ID:** 2611.06728 | [PDF](https://arxiv.org/pdf/2611.06728v1)

**作者:** Shuai Dong `[一作]` (Fudan University), Jiaqi Wang `[通讯]` (JD.com)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TurnWise 方法，针对流式视频交互任务，将优势估计与损失聚合对齐到决策回合，利用静默完成前缀评分得到每回合的剩余收益优势，并采用回合均值聚合使每次决策权重相等，从而改进响应与等待决策的学习。

**💡 创新点**

创新点包括：① 通过前缀评分剔除已获得奖励，实现基于剩余收益的回合级优势；② 采用回合均值聚合消除响应长度对学习权重的影响；③ 两项设计共同把决策回合作为信用分配与策略优化的统一基准，且不需要额外的 critic 或 roll‑out。

**🔧 技术方法**

技术方法：GRPO（Group Relative Policy Optimization）框架、返回‑to‑go 原理、静默前缀评分、优势截断、回合均值聚合、离散化奖励评估（PAUC）以及强化学习优化。

**📊 数据集**

使用的数据集包括：训练阶段的 EgoExo4D、EgoExoLearn、Live‑WhisperX、Tarsier2、LLaVA‑Video；评估阶段的 ProactiveVQA、OVO‑Timing、StreamingBench；模型基于 Qwen2.5‑VL‑3B 与 Qwen3‑VL‑4B 进行 fine‑tune 与 RL。

**📈 对比分析**

与 SFT、MMDuet2、ThinkStream、StreamPro 等基线在同一 GRPO 环境下比较，TurnWise 在所有三大基准（ProactiveVQA、OVO‑Timing、StreamingBench）上均取得最高总分，提升幅度约为 3–6 分（PAUC、Timing F1、准确率），并显著降低冗余回复率。

**⚠️ 局限性**

局限性：目前仅在少量专门设计的流式视频基准上验证，尚未评估在更广泛场景或不同模型骨干上的泛化；方法仍依赖大量的强化学习样本，训练成本高；且对异常情况（如极端视频噪声或极长序列）的鲁棒性未作深入分析。

---

## 654. Airlock: Bounding the Multi-Agent Blast Radius via Selective Online Containment

**arXiv ID:** 2611.06727 | [PDF](https://arxiv.org/pdf/2611.06727v1)

**作者:** Zhonghao Zhan `[一作]` (Imperial College London), Hamed Haddadi `[通讯]` (University of Oxford)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于树形权限委托与在线可追溯图的多代理运行时与守护机制，能够在多代理系统中检测并隔离故障源扩散导致的外部错误，避免全局暂停或恢复而仍保留健康工作。

**💡 创新点**

创新点在于将垂直权限收缩与水平跨代理可追溯性结合，设计了可实时评估的“Fence‑then‑Prune”最小割算法，实现在线、零泄漏的选择性容错；同时提出了两层隔离策略（全隔离与优化割）在实际智能家居与模拟工作流中的有效性。

**🔧 技术方法**

使用了树结构的分层运行时（权限递减、按站点日志、锁定门控和对象切换）、append‑only DAG记录跨代理数据流、基于该图的前向/后向污点传播、以及最大流/最小割求解器（包括稀疏求解与 MILP）来实现动态切割与安全释放。

**📊 数据集**

在真实物理智能家居实验（六种大型语言模型部署）和七个分布式多代理工作流模拟（共计约1.5万条任务）中进行评估。

**📈 对比分析**

相较于 Google A2A、LangGraph 及其他粗粒度策略，实验显示在所有故障场景下均能实现零后警漏；在“稠密扩散”条件下，优化割保留的健康工作量比全隔离多约 8.3%（F‑T283），且系统时延仅比 A2A 低 1–2% 的完成时间。

**⚠️ 局限性**

局限性包括单进程协调器的日志摄取瓶颈（超过 63 个站点时吞吐受限）、无法回滚代理内部状态、以及对大规模部署的可伸缩性需进一步分片和预算化管理。

---

## 655. Drowning in Noise: On Signal Saliency in Self-Supervised Learning for Ocean Acoustics

**arXiv ID:** 2611.06725 | [PDF](https://arxiv.org/pdf/2611.06725v1)

**作者:** Shikhar Bharadwaj `[一作]` (Carnegie Mellon University), Malcolm Slaney `[通讯]` (Stanford)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究在海洋声学中噪声过度表征对自监督学习的影响，并证明仅用稀疏的生物声事件进行预训练可显著提升下游任务性能。

**💡 创新点**

提出“噪声过度表征”概念，展示在信息稀疏的海洋记录中数据精细化比规模更重要，并系统评估多种显著事件检测器在该领域的不足。

**🔧 技术方法**

使用BEATs自监督框架进行预训练，结合随机码本和BEATs迭代3 tokenizer，评估线性探测与全微调性能。

**📊 数据集**

采用Watkins Marine Mammal Sound Database的Master Tapes（1240小时）与Cut Tapes（30小时），以及AudioSet和BEANS分类基准。

**📈 对比分析**

通过线性探测和全微调对比不同预训练语料，30小时精选语料线性探测提升约29个百分点，远超1240小时未过滤语料；自动显著检测过滤仅略优。

**⚠️ 局限性**

自动显著事件检测器在海洋声学中表现差，难以替代人工标注，导致过滤后的语料仍低于人类策划的最佳性能。

---

## 656. Combinatorial Boolean Matrix Multiplication in $n^3 / 2^{Ω(\sqrt{\log n})}$ Time

**arXiv ID:** 2611.06717 | [PDF](https://arxiv.org/pdf/2611.06717v1)

**作者:** Piaoyang Cui `[一作]` (Google), David P. Woodruff `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种确定性的组合算法，用于布尔矩阵乘法（BMM）和三角形检测，运行时间为O(n^3 / 2^Ω(√(log n)))。

**💡 创新点**

通过将(A,B)分解为稀疏或伪随机的子矩阵对，显著减少了所需的子矩阵数量，从而提高了运行时间的指数，从(log n)^(1/7)降低到√(log n)。

**🔧 技术方法**

使用了组合算法和伪随机性概念，特别是通过引入二元网格正则性和最小度修剪技术来优化算法。

**📊 数据集**

使用了标准的n×n布尔矩阵作为数据集，并通过组合方法将问题转化为三元组图的三角形检测。

**📈 对比分析**

与之前的算法相比，性能显著提升，运行时间从O(n^3 / 2^Ω((log n)^(1/7)))改进到O(n^3 / 2^Ω(√(log n)))，并且在处理三角形检测时，能够在O(n^3 / 2^Ω(√(log n)))时间内完成。

**⚠️ 局限性**

算法的局限性在于仍然依赖于组合BMM假设，且在处理更复杂的图结构时可能面临性能瓶颈。

---

## 657. Energy-Guided Transport for Projection-Free Physics-Informed Flow Matching

**arXiv ID:** 2611.06711 | [PDF](https://arxiv.org/pdf/2611.06711v1)

**作者:** Giuliano Galadini `[一作]` (Sapienza University of Rome), Danilo Comminiello `[通讯]` (Sapienza University of Rome)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Energy-Guided Transport（EGT），一种在功能流匹配框架中实现物理约束的零样本连续导向方法；

**💡 创新点**

创新点在于将非线性 PDE 约束映射为平滑能量势，通过梯度下降的耗散力连续引导流形，避免了离散投影导致的轨迹偏移与高内存消耗；

**🔧 技术方法**

技术主要包括功能流匹配（FFM）、逆向自动微分实现能量梯度、动态调度的耗散系数、梯度范数裁剪、一次 Euler 步长积分；

**📊 数据集**

使用四种物理系统的数据集：1D 热传导、1D 反应扩散、1D 布格斯方程、2D Chaotic Navier–Stokes，均采用公开的高精度数值解作为基准；

**📈 对比分析**

与 ProFlow、ECI、DiffusionPDE、D-Flow、PCFM 等主流零样本指导方法进行对比；在部分约束和全 PDE 约束下，EGT 以 100% 成功率完成推断，推断速度最快（如 2D Navier–Stokes 0.71s/样本），物理残差和样本 MSE 均显著低于对手；

**⚠️ 局限性**

局限包括：不保证最终状态严格满足 E(u)=0（需可选终端微调）、对大耗散系数会引入 ODE 稳定性问题、仅约束点级 L2 PDE 残差，未显式控制宏观统计或能量谱，且对复杂几何和 3D 无结构网格的扩展尚待研究。

---

## 658. Beyond the principle-practice gap: Responsible AI as situated epistemic and ethical practice

**arXiv ID:** 2611.06709 | [PDF](https://arxiv.org/pdf/2611.06709v1)

**作者:** Thao Do Ngoc `[一作]`, Emma Carmel `[通讯]` (University of Bath)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对14名从事AI辅助反人口贩卖系统的专业人员进行半结构化访谈，探讨了知识与伦理决策如何在不同机构逻辑与专业背景下形成并在技术与治理设计中实现。

**💡 创新点**

将责任AI视为认知与伦理的循环过程，提出四阶段框架（感知、判断、情境行动、技术-治理实现），并揭示机构逻辑如何在此过程中扭曲或过滤伦理关注。

**🔧 技术方法**

主要采用定性研究方法：访谈、迭代编码、主题分析与四阶段框架归纳；未使用机器学习模型。

**📊 数据集**

基于14份访谈记录，样本涵盖执法、民间组织、科研、商业等多方，数据来源于受访者对各类AI反贩卖项目的描述。

**📈 对比分析**

研究为探索性定性分析，没有定量对比；通过跨案例比较识别出四个伦理-知识框架的相似与差异，未给出算法性能指标。

**⚠️ 局限性**

样本有限、缺少受害者视角、未观察系统开发与部署、研究仅依赖受访者主观叙述。

---

## 659. Boxicity of Power Graphs of Finite Nilpotent Groups

**arXiv ID:** 2611.06816 | [PDF](https://arxiv.org/pdf/2611.06816v1)

**作者:** L. Sunil Chandran `[一作]` (Indian Institute of Science), Suraj Kumar Sahoo `[通讯]` (Indian Institute of Science)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究确定了有限可解群（尤其是可解群）幂图的确切盒子数，并给出了精确的计算公式。

**💡 创新点**

创新点在于首次用拓扑不等价工具（Van Kampen 归约）与顺序理论相结合，推导出幂图盒子数的闭式表达式，并完整解决了先前的开放问题。

**🔧 技术方法**

采用了组合几何、图论与群论的交叉方法，具体包括可比图、根化树状偏序、区间图表示、Helly 性质以及拓扑不等价理论。

**📊 数据集**

该工作不依赖任何实验数据或公开数据集，全部基于理论推导与组合证明。

**📈 对比分析**

与已有结果相比，本文给出了可解群盒子数的最优下界与上界相符，实现了精确值；对已知的仅给出上界或不确定结果的情形做出了显著改进。

**⚠️ 局限性**

局限性在于目前仅覆盖有限可解群，尚未给出非可解群或更一般群的完整盒子数公式；对高阶 Sylow 2-子群的细分仍有未解之处。

---

## 660. A Frame-Level View and Image-Quality Annotation Tool for Fetal Echocardiography: Comparison with Spreadsheet Annotation

**arXiv ID:** 2611.06791 | [PDF](https://arxiv.org/pdf/2611.06791v1)

**作者:** Mohamed Azzam `[一作]` (University of Nebraska Medical Center), Jieqiong Wang `[通讯]` (University of Nebraska Medical Center)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了专门用于胎儿心脏超声逐帧标注的工具CAT，并在专家审核与学生交叉实验中评估其错误率与标注效率。

**💡 创新点**

CAT实现了在DICOM cine上直接记录视图与图像质量的单帧标签，自动生成帧区间，杜绝结构性数据录入错误，并提供更细粒度的标注。

**🔧 技术方法**

使用Python+Tkinter实现的桌面GUI，读取DICOM、MP4等格式，按变更点存储标签，输出CSV；评估采用自举、GEE、排列检验等统计方法。

**📊 数据集**

共25例胎儿心脏超声（15正常、10先天性心脏病）共1,746文件，用于专家审核；学生实验使用两例正常扫描（含假文件）进行随机交叉。

**📈 对比分析**

与传统电子表格标注相比，CAT在专家审核中发现6.2%文件存在严重错误（CHD病例更高），视图一致率78.5%，质量一致率60.4%；学生标注速度提升约1.7倍，准确率无显著差异。

**⚠️ 局限性**

未设立独立参考标准；专家审核顺序未随机化；学生样本仅6人且时间自报；CAT的评估与开发由同一团队完成，未检验对临床专家的时间节省效果。

---

## 661. Target PageRank: Matching Prescribed PageRank Scores via Budgeted Edge Additions

**arXiv ID:** 2611.06782 | [PDF](https://arxiv.org/pdf/2611.06782v1)

**作者:** Parth Gupta `[一作]` (Indian Institute of Technology Delhi), Abhijnan Chakraborty `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在给定边添加预算下如何通过最少的边添加使目标节点的PageRank达到预设向量，并提出优化与启发式算法。

**💡 创新点**

提出Target PageRank决策问题的NP‑完备性与优化问题不可多项式逼近性，给出MILP模型并设计三种基于曝光度、误差评估与随机游走匈牙利匹配的贪心启发式。

**🔧 技术方法**

利用复杂度分析、整数规划建模、PageRank增量更新、随机游走中心度、匈牙利最小成本匹配等技术。

**📊 数据集**

使用亚马逊产品评论生成的四个推荐网络（Electronics、Books、Sports、Clothing）以及合成的Barabási–Albert电网。

**📈 对比分析**

通过ℓ1误差、平均相似度和运行时间比较，RWH在误差和相似度上表现最好，G1最快，G2在误差上略优但耗时最高。

**⚠️ 局限性**

MILP规模有限，启发式无法保证全局最优，且在极端结构（如Clothing网络）下效果退化。

---

## 662. WISE: An Agentic Framework for Whole-page Multimodal Search Experience Attribution

**arXiv ID:** 2611.06777 | [PDF](https://arxiv.org/pdf/2611.06777v1)

**作者:** Wenyi Xu `[一作]` (Kuaishou Technology), Jingwei Zhuo `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了系统质量诊断(SQD)任务，构建了WISE框架实现多模态搜索体验诊断与管线归因，并推出了WISE-Bench基准。

**💡 创新点**

创新点在于：①将质量诊断与组件归因统一为一体化任务；②利用ReTrace构造证据一致的轨迹监督；③设计AHPO根据诊断失败自动选择优化 harness 或 policy。

**🔧 技术方法**

技术上采用LLM驱动的 agentic 诊断、轨迹修复 ReTrace、强化学习 GRPO 的适应性 harness‑policy 优化、以及多模态证据获取与结构化推理。

**📊 数据集**

使用12,000条真实短视频平台搜索案例，包含查询、曝光结果、用户体验证据与内部管线追踪，划分训练/验证/测试集。

**📈 对比分析**

与多种 baseline（基线模型、仅 harness 优化、仅 policy 优化、完整 ReTrace 与 AHPO）对比，WISE 在 Qwen3.5‑35B‑A3B 上 end‑to‑end 精确率提升至 75.20%，显著优于其他方法。

**⚠️ 局限性**

局限性包括：对多模态证据依赖度高；轨迹修复和优化迭代成本较大；在不同场景下的泛化性仍待进一步验证。

---

## 663. Route First, See Later: Enabling Vision Agents to Reason Scientifically by Separating Routing Policy from Action Spaces

**arXiv ID:** 2611.06760 | [PDF](https://arxiv.org/pdf/2611.06760v1)

**作者:** Shubhabrata Mukherjee `[一作]` (Lawrence Berkeley National Laboratory), Daniela Ushizima `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并评估了 ZETA 这一科学图像分析代理架构，探讨动作空间与路由策略的区别，并在零样本条件下进行路由准确率实验。

**💡 创新点**

将路由策略与执行阶段解耦，首次通过控制实验验证显式路由语义对路由准确率的显著影响，提供可审计的错误归因框架。

**🔧 技术方法**

使用大型语言模型（Qwen 3.8、Nemotron 3.5、Qwen 3.6、Qwen 3.1B、muse‑glimmer 30B）作为路由器，配合检测、分割、计数、描述等专用视觉工具及 Zenesis 分割框架。

**📊 数据集**

在51个科学意图查询的零样本基准上评估，未使用公开图像数据集，仅基于这些查询构建实验。

**📈 对比分析**

对比 Action Space Only 与 Action Space + Routing Policy 两种设置，记录计数、描述、检测、分割等任务的路由准确率；显式路由策略显著提升检测和分割准确率（如 Qwen 3.8 检测从 60% 提升至 100%，分割从 6.2% 提升至 75%）。

**⚠️ 局限性**

实验规模有限（仅51个查询），结果受模型差异影响；显式路由并非始终有效（如 Qwen 3.1B 下降），且未评估下游执行准确率或端到端科学任务成功率。

---

## 664. Continual Classification Beyond the Kernel Regime: Harnessing the Sparsity Bias of L2 Regularization in Diagonal Linear Networks

**arXiv ID:** 2611.06759 | [PDF](https://arxiv.org/pdf/2611.06759v1)

**作者:** Gilad Karpel `[一作]` (Technion), Daniel Soudry `[通讯]` (Technion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在高维统计设定下，分析了连续学习（CL）中的逻辑回归，提出了一种基于两层对角线线性网络的“符号加权L1”正则化，能够在保持稀疏性偏置的同时稳定长任务序列。

**💡 创新点**

创新点包括：① 推导出L2正则化在连续分类中的最优强度与任务数的关系；② 在单层模型下证明L1正则化可利用稀疏先验，但收敛慢；③ 设计符号加权L1正则化，使得两层网络在参数空间中的L2正则化等价于预测器空间中的稀疏正则化，从而在连续学习中获得线性稀疏收敛速率。

**🔧 技术方法**

技术方法：高维随机近似（state evolution）解析连续迭代的极限行为；对齐参数空间与预测器空间的正则化；利用凸、齐次性质得到全局收敛上界；实验中使用Proximal FISTA与Adam实现优化。

**📊 数据集**

实验数据集：1）高维高斯合成数据（d=500）；2）MNIST二分类（0–4 vs 5–9）；3）两层ReLU网络在MNIST上的实现，教师为稀疏预训练网络。

**📈 对比分析**

与传统L2、L1正则化对比，符号加权L1在稀疏教师下收敛最快、最终误差最低；在稀疏度降低时性能仍优于L2且收敛速度保持较快；实验显示其在前几任务即可显著减少测试损失。

**⚠️ 局限性**

局限性：仅在单标签二分类、Gaussian特征、单教师设置下证明；对多任务、多分类或非高斯特征的推广尚未验证；正则化参数需手动设定，且只针对两层对角线网络，深层或非线性结构的适用性未探讨。

---

## 665. EvoWorld: Towards Out-of-Sight World Evolution in Generative Video Models

**arXiv ID:** 2611.06748 | [PDF](https://arxiv.org/pdf/2611.06748v1)

**作者:** Yequan Bie `[一作]` (Hong Kong University of Science and Technology), Hao Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Out-of-Sight Causal Evolution任务，并构建了EvoWorld-Data和EvoWorld-Bench。

**💡 创新点**

通过将世界状态推理与视觉渲染分离的脑‑渲染器协作框架，以及Identity‑Anchored Attention和两阶段训练，显著提升了在事件驱动的不可见视角下的因果演化能力。

**🔧 技术方法**

使用VLM脑（如Qwen3.8‑27B）进行感知与推理，视觉渲染器基于LTX‑2.5，结合IAA、两阶段自回归训练、实体与时间锚点。

**📊 数据集**

构造的EvoWorld‑Data包含约12k个10秒以上的长视频，涵盖19类事件、65个细粒度事件、16个场景、63个细粒度场景、14种反应；EvoWorld‑Bench共120个案例。

**📈 对比分析**

与LingBot‑World、LingBot‑World‑v2、YUME‑1.5、SANA‑WM、Echo‑WM、Evoke及其Turbo版对比，EvoWorld在Core Task Success上达90.8%，远高于第二名，并在事件实现、重访成功、因果合理性、反应质量等指标上表现突出。

**⚠️ 局限性**

仍受限于事件多样性、对外部知识的依赖、以及在更复杂动态场景中的可扩展性；在极端长距离或高度非线性事件的因果推断上表现不足。

---

## 666. TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing

**arXiv ID:** 2611.06741 | [PDF](https://arxiv.org/pdf/2611.06741v1)

**作者:** Ruikang Li `[一作]` (Chinese University of Hong Kong), Tianfan Xue `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过对扩散模型的后期训练，TailSteer实现了多轮图像编辑的漂移抑制和内容保留，

**💡 创新点**

创新点在于只在扩散轨迹尾部进行“尾部引导”，利用模型自身的推理轨迹和未改变区域作为无监督的保留监督，而无需重新学习编辑能力，

**🔧 技术方法**

主要技术包括基于扩散模型的自监督轨迹终点构造、在轨迹尾部进行的速度向量混合监督、以及无梯度回传的 on‑policy 训练框架，

**📊 数据集**

使用的训练数据来自 FLUX、SenseNova、Qwen 等大模型的自生成保留轨迹，实验数据集包括 Open Images、MONET、以及自建的 Add & Preserve 评测集，

**📈 对比分析**

与 FreqEdit、VAE‑LFA、Consistency LoRA 等方法对比，TailSteer在多轮编辑中显著降低了 PSNR/SSIM/LPIPS 损失，COA 接近基模型水平，且在大部分基线模型上保留了单轮生成和编辑性能；在 GPT‑Image‑2.5 Sunburst 上也取得了更好的保留效果，

**⚠️ 局限性**

局限性包括对 VAE 质量的依赖导致部分色彩漂移、对训练数据规模敏感、以及对某些复杂空间编辑的约束仍不如专门的 LoRA 或多轮强化学习方案。

---

## 667. From Pixels to Actions - Towards Native Vision-Language-Action Policies

**arXiv ID:** 2611.06918 | [PDF](https://arxiv.org/pdf/2611.06918v1)

**作者:** Weichen Fan `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种结构上原生的视觉-语言-动作（VLA）框架NEO-VLA，直接用像素和文本构建统一的多模态前缀，并用流匹配动作专家从不同层次读取视觉表示以预测连续动作。

**💡 创新点**

创新点在于：1）移除预训练视觉编码器，直接在轻量级卷积嵌入层把像素投影到Transformer空间；2）让动作专家可显式选择读取任意层次的视觉键值，揭示动作关键信息主要集中在早期视觉层；3）通过三阶段预训练和多模态共训练保持理解能力，提升跨域鲁棒性。

**🔧 技术方法**

技术主要包括：轻量级卷积patch嵌入、Native Rotary Position Embedding、三维多模态注意力、流匹配动作专家、三阶段预训练（图文、对话、OCR、检测、多模态指令调优）以及基于Bfloat16的混合精度训练与编译。

**📊 数据集**

使用了9个公开机器人数据集（AgiBot-World、Bimanual YAM、RoboCOIN、Open X-Embodiment、Galaxea Open-World、RH20T、RoboMIND、InternData-A1、DROID），共计约223万条经历、约9927小时的多视角视觉+语言+机器人状态数据。

**📈 对比分析**

在LIBERO、LIBERO‑Plus、LIBERO‑Pro、RoboTwin 2.0等标准与分布式评测上，NEO-VLA在冻结或全微调下均优于使用CLIP/SigLIP/DINOv2等预训练视觉编码器的VLM；在冻结时平均成功率达95%+，全微调可达>97%；在LIBERO‑Plus对七种分布式偏移的综合分数为89.7，RobotiTwin 2.0 Clean2Rand为69.4，显著优于同尺寸对手。

**⚠️ 局限性**

局限性包括：1）对空间几何或任务目标变动仍表现不佳，表明仍需改进对空间/意图适应的能力；2）模型规模仍相对较大，推理延迟在10步Euler采样时约282ms；3）缺乏对更复杂长序列、动态障碍的评估，需进一步验证在更大规模机器人平台上的泛化。

---

## 668. Motion-guided Contrastive Regularization For Latent World Models

**arXiv ID:** 2611.06901 | [PDF](https://arxiv.org/pdf/2611.06901v1)

**作者:** Yiwen Chen `[一作]` (Northeastern University), Amir Said `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了运动引导对比正则化（MGC），用于解决JEPA式世界模型在视觉控制任务中出现的“静态崩溃”问题，使模型能够在不重构像素的情况下保持对动作相关状态变化的表示；

**💡 创新点**

创新点在于将光流信息作为对比目标，构造软对比损失，使预测的潜在转移既能区分不同的动作结果，又能保留与真实运动相似的潜在转移；

**🔧 技术方法**

采用了ViT编码器、Transformer预测器、SIGReg（均值方差正则化）以及光流模型（SEA‑RAFT）计算运动相似度；

**📊 数据集**

使用了OGBench‑Cube（原始与随机背景版Cube‑RB）和LIBERO‑Object等视觉控制基准数据集；

**📈 对比分析**

在与LeWM、SMWM、PLDM及冻结预训练编码器的DINO‑WM对比时，MGC在Cube‑RB和LIBERO‑Object上的规划成功率分别从52%提升到78%，从14%提升到54%，并在其他四个基准上保持或略优的性能；

**⚠️ 局限性**

局限性包括：对静态背景假设依赖强，动态背景可能破坏光流监督；对动作覆盖率敏感；在视觉外观或动作表征发生变化时鲁棒性有限。

---

## 669. Learning to Decode Linear Codes with Generalized Belief Propagation Network

**arXiv ID:** 2611.06874 | [PDF](https://arxiv.org/pdf/2611.06874v1)

**作者:** Samira Malek `[一作]` `[通讯]` (Pennsylvania State University), Samira Malek (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种通用的GBPNet，结合级联的变量节点子层和可学习的缩放块，用以改进线性码的BP解码。

**💡 创新点**

创新点在于通过多级变量节点子层实现隐式全连接变换，同时保留Tanner图结构；引入高SNR和逐步训练策略；证明对称条件保持，从而仅需训练全零码字。

**🔧 技术方法**

使用深度神经网络展开BP，级联子层、Hadamard缩放、交叉熵多损失训练、Adam优化。

**📊 数据集**

采用BCH、Polar、LDPC和MacKay等线性码的校验矩阵进行实验。

**📈 对比分析**

与传统BP、Hyp BP和基于Transformer的ECCT、CrossMPT等方法在相同SNR下比较，GBPNet在BER上显著优于传统BP，并在多种码上优于或与最新深度解码方法相当，尤其在高SNR区表现突出。

**⚠️ 局限性**

局限在于需要手动设定子层数K、迭代次数L及训练策略，计算复杂度相对较高，且对极长码的可扩展性未充分验证。

---

## 670. On Turn-Level Best-of-N Sampling for Scaling LLM Agents

**arXiv ID:** 2611.06868 | [PDF](https://arxiv.org/pdf/2611.06868v1)

**作者:** Jalaj Bhandari `[一作]` (Institute of Foundation Models, MBZUAI Labs), Rupesh Srivastava `[通讯]` (Institute of Foundation Models, MBZUAI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在每一步生成N个候选动作后，用预训练LLM进行选择的 turn‑level best‑of‑N 采样方法；

**💡 创新点**

创新点在于将无训练的LLM直接用作动作评判器，在推理阶段实现近似的一步策略改进，从而显著提升长周期任务的成功率与一致性；

**🔧 技术方法**

采用生成‑选择器（generator‑selector）框架，生成候选动作后通过列表式选择，理论上对应于策略改进；

**📊 数据集**

使用 Terminal‑Bench v2.1 软件工程任务（89 个任务）以及 mini‑SWE‑agent harness 进行评估；

**📈 对比分析**

与基线（无选择器）相比，pass@1 从约 47% 提升到 60–70%（K2‑Horizon）或从 73% 提升到约 80%（Qwen）；pass@k 曲线更平缓，一致性提升；轨迹中执行的步骤平均减少约 40–50%；错误率（循环、错误提交等）下降；计算成本约为生成器+选择器总 token 数的 3–6 倍；

**⚠️ 局限性**

主要限制是显著增加推理时的计算量（多次生成候选和选择），且目前仅在所有步骤都采样，未尝试在部分步骤采样或训练轻量级评判器，缺乏高效实现。

---

## 671. CounterFactual Retrieval for Evidence-Aware Individual Animal Identification

**arXiv ID:** 2611.06864 | [PDF](https://arxiv.org/pdf/2611.06864v1)

**作者:** Ankit K. Upadhyay `[一作]` (Rensselaer Polytechnic Institute), Lukas Picek `[通讯]` (MIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Counterfactual Foreground Retrieval (CFR)评估协议，揭示动物识别模型在不同视觉证据下的表现

**💡 创新点**

通过对同一查询-图库任务使用全图、前景仅、背景/轮廓仅三种对照视图，构建证据轮廓，量化模型对动物本体与短路线索的依赖

**🔧 技术方法**

使用图像分割掩码生成器SAM3进行前景/轮廓提取，定义前景/短路输入，计算检索得分并归一化得到CFR和Shortcut Signal Indicator (SSI)

**📊 数据集**

构建了包含约0.5M前景掩码的34个野生动物识别数据集（如 WhaleSharkID、LeopardID2022、Saimaa seal 等）

**📈 对比分析**

与传统Retrieval@k对比，CFR对比模型-数据集对的证据分布进行分析，发现相同检索性能可由前景支持、短路共存或前景不足导致；前景混合训练能显著降低SSI但未完全消除短路；模型在不同数据集上表现不一

**⚠️ 局限性**

CFR仍是诊断性工具，无法定位短路线索来源，且对掩码质量高度依赖；部分数据集仍存在高SSI且CFR低的情况，表明问题不只在模型训练而在数据本身

---

## 672. Balancing hierarchical representation and quantization: A performance trade-off in text embeddings

**arXiv ID:** 2611.06859 | [PDF](https://arxiv.org/pdf/2611.06859v1)

**作者:** Kai Ponel `[一作]` (Ludwig-Maximilian-University Munich), Ernesto William De Luca `[通讯]` (Otto von Guericke University Magdeburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Matryoshka层级嵌入与量化技术的组合效果进行大规模实证评估，比较不同向量尺寸、精度等级和层次选择对38个MTEB任务的性能和效率影响。

**💡 创新点**

首次证明8位量化与八分层降维组合可在保持94%效果的同时按32倍压缩向量存储；提供统计显著性检验，揭示检索任务对压缩最敏感；给出任务级部署指南。

**🔧 技术方法**

使用Matryoshka 1D/2D嵌入、8位整数量化与1位二进制量化，并采用Friedman、Nemenyi、Wilcoxon等非参数统计方法；基于Python/PyTorch实现MTEB基准。

**📊 数据集**

使用MTEB公开的38个任务（分类、聚类、对句、重排序、检索、语义相似度、摘要），涵盖科学文献、社交媒体等多域数据。

**📈 对比分析**

通过相对性能和相对评估时间两维度评估168配置/任务；int8几乎与float32同等，二进制量化显著失效；检索在任何压缩下损失最大；存储上int8+m=128可32倍压缩、保持94.4%效果；2D层采样可近半评估时间。

**⚠️ 局限性**

仅评估四个1B以下模型，未对大规模检索语料库或PQ/float16等基线做对比；时间测量单次、无方差；仅使用英语数据；binary量化仅作为候选生成；实验在T4 GPU，未验证其他硬件。

---

## 673. Navigating Disagreement Among Criteria for Evaluating Explanation Faithfulness

**arXiv ID:** 2611.06846 | [PDF](https://arxiv.org/pdf/2611.06846v1)

**作者:** Grant Shanklin `[一作]` (Max Planck Institute for Security and Privacy), Asia Biega `[通讯]` (Max Planck Institute for Security and Privacy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 faithfulness 评估的统一框架，并对现有多种评估方法进行重构，揭示了它们在对解释的要求上存在的差异。

**💡 创新点**

将 faithfulness 视为一个不完全规定的关系，指出不同评估标准源于规范化选择，并提供了解释性、制度性和规范性三重视角的正当化框架，帮助在具体应用场景中选择合适的评估方法。

**🔧 技术方法**

运用了测量模型理论中的功能化表示法（faithfulness functional）对各评估指标进行抽象重构，并通过案例分析（不良行动通知）检验框架的可解释性和合理性。

**📊 数据集**

主要以信用评分的 AAN（不良行动通知）为案例进行分析，但文中未给出具体公开数据集，案例基于行业提供的信用模型与特征划分。

**📈 对比分析**

通过比较现有评估指标在同一解释集合上的排名差异，展示了框架能够解释差异来源；文中未进行实验性能评估，主要侧重理论与案例说明。

**⚠️ 局限性**

局限于局部特征重要性解释和理由选择，未覆盖反事实等其他解释形式；案例仅来自信用领域，泛化性和对其他模型类型的适用性尚待验证。

---

## 674. Coexistence-Aware Antenna Tilt and Resource Allocation for Cellular-Connected UAV Corridors

**arXiv ID:** 2611.06832 | [PDF](https://arxiv.org/pdf/2611.06832v1)

**作者:** Tarun Singh Rawat `[一作]` (Indian Institute of Technology Jodhpur), Pradyumna Kumar Bishoyi `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了在蜂窝网络中同时优化基站天线倾斜角和物理资源块(PRB)分配的框架，以满足无人机(UAV)通道的最低下行速率要求并保证地面用户(GUE)的平均延迟约束。

**💡 创新点**

创新点包括：① 将天线倾斜与PRB分配联合建模为非凸优化问题并通过分块求解；② 采用投影梯度上升法在天线倾斜子问题中引入对UAV速率的对数障碍约束；③ 在PRB分配子问题中使用水分配法处理可行区间，超出时采用经济学中的加权Talmud规则（WTA）以公平分配有限资源。

**🔧 技术方法**

使用的技术主要有：投影梯度上升、对数障碍函数、M/G/1排队均衡分析、闭式服务率阈值、WTA分配规则、Monte‑Carlo仿真与CDF性能评估。

**📊 数据集**

使用的实验数据集为基于19个站点、ISD 500 m、FR1 3.5 GHz、100 MHz 带宽、30 kHz 子载波间距的仿真场景，包含两个不同高度的UAV通道和每个基站50名GUE。

**📈 对比分析**

通过与等分配（equal sharing）和水分配（water‑filling）方案的对比，实验结果表明：在资源不足时WTA能显著降低UAV的速率不满足比例，并且随着GUE平均延迟容忍度的提高，UAV满意度进一步提升；同时，WTA保持了GUE的基本服务质量。

**⚠️ 局限性**

局限性：仅在仿真环境中验证，未考虑多通道、多频段或UAV群飞的复杂情形；天线倾斜角度仅限于固定区间，未加入动态适配机制；WTA规则依赖于预设优先权，实际部署中优先级设定可能更复杂。

---

## 675. BEAKER: An Expert-Curated Benchmark for Embodied Brains in Self-Driving Chemical Laboratories

**arXiv ID:** 2611.06911 | [PDF](https://arxiv.org/pdf/2611.06911v1)

**作者:** Fei Lin `[一作]` (HydroSophy), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并发布 BEAKER 基准，用于评估多模态大模型在自驱化化学实验中的执行感知与推理能力。

**💡 创新点**

创新点在于完全由化学与 AI 专家手工构造 1,000 QA 样本，覆盖 Actionability、Knowledge、Experimentation、Reasoning 四大维度，并聚焦细粒度操作与时间推理，避免了传统基准的 AI 生成偏差。

**🔧 技术方法**

采用多模态大模型（28个）在离线单回合评估环境中，通过坐标、单选、填空三种答案格式，并结合图像/视频与文本提示进行测试。

**📊 数据集**

使用公开化学实验图像 883 张和视频 734 条（来自学术公开、JoVE 等），全部由专家筛选并标注，构成 BEAKER 的视觉数据集。

**📈 对比分析**

与三位未参与构造的化学专家对照，模型平均得分约 55，最佳模型 Claude Opus 4.7 达到约 90；在 AFF、TRAJ 等细粒度任务中模型表现明显不足，显示执行感知仍有较大差距。

**⚠️ 局限性**

局限在于仅进行离线单回合评估，未覆盖闭环交互与实时错误恢复，样本覆盖范围有限，且存在训练集泄露的潜在风险。

---

## 676. FrontierPhysics: Evaluating Agents for End-to-End Physics Research

**arXiv ID:** 2611.06910 | [PDF](https://arxiv.org/pdf/2611.06910v1)

**作者:** Bingran You `[一作]` (BenchFlow), Hartmut Haeffner `[通讯]` (University of California, Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了FrontierPhysics基准，评估AI代理在完整物理研究流程中的表现。

**💡 创新点**

创新点在于构建由物理研究者亲自设计、无参考答案且结合确定性验证与专家制定的评价rubric的端到端物理研究基准。

**🔧 技术方法**

使用了大型语言模型代理、外部工具（文献搜索、代码编写、实验模拟）、确定性验证器与基于rubric的评审Agent。

**📊 数据集**

数据集为56个任务，来自42名物理学家，覆盖9个物理领域，任务基于作者自身研究，包含论文、数据和实验脚本。

**📈 对比分析**

通过与22个模型、4个harness对比，Pass@1最高仅约30%，显示现有模型在端到端研究中的不足，进一步通过手工失败模式分析评估。

**⚠️ 局限性**

局限在于评审误判与验证缺陷的可能性、任务多样性和规模有限，以及模型失败主要因推理缺乏严谨性和专家直觉不足，而非知识缺失。

---

## 677. A Simple Subquadratic Cut Query Protocol for Directed Reachability

**arXiv ID:** 2611.06906 | [PDF](https://arxiv.org/pdf/2611.06906v1)

**作者:** Henry Fleischmann `[一作]` (Carnegie Mellon University), Swagato Sanyal `[通讯]` (University of Sheffield)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种利用有向切割查询的随机零误差算法，用来判定无权有向图中 s–t 的可达性，并在期望下完成 O(n⁹/⁵) 次查询，进一步改进至 O(n⁵/³)。

**💡 创新点**

创新点在于构造“近似方向规则”，通过跟踪累积不平衡误差并使用 FTRL（Follow‑The‑Regularized‑Leader）在线优化框架，随后采用独立随机化取舍与流修正，将切割查询量从二次提升到多项式次二次；并提供完整的理论分析和实现蓝图。

**🔧 技术方法**

主要技术包括：有向切割查询模型、子模函数最小化关联、在线优化的 FTRL 迭代、误差平均与独立取舍、以及基于流的平衡修正（利用阻塞流和 Even‑Tarjan 分析）。

**📊 数据集**

无；该工作完全是理论算法，没有使用实验数据集。

**📈 对比分析**

与之前的 O(n²/ log n) 或 O(n²) 方案相比，新算法在期望和高概率下将查询次数降低至 O(n⁹/⁵) 甚至 O(n⁵/³)，实现了真正的多项式次二次改进，表明可达性问题在切割查询模型下可以突破二次瓶颈。

**⚠️ 局限性**

局限性包括：仍未达到最优的 O(n^1.5) 目标，算法仅适用于无权、无 2‑循环 的有向图（可通过微调扩展），且实现复杂度较高；此外，理论分析依赖随机性，实际实现中需关注随机取舍与流计算的效率。

---

## 678. VABAR: A Variational Automation Benchmark for Agentic Robotics

**arXiv ID:** 2611.06881 | [PDF](https://arxiv.org/pdf/2611.06881v1)

**作者:** Shuangyu Xie `[一作]` (University of California Berkeley), Ken Goldberg `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Variational Automation Benchmark for Agentic Robotics (VABAR)，定义六类工业/医疗/基础设施应用的操作任务，并提供基于Newton和Isaac Lab的仿真环境、33个可复用技能、人工编写的计算图参考解法以及基线评估。

**💡 创新点**

1) 将变异自动化任务与agentic robotics结合，提出适合agentic系统的基准设计原则；2) 系统化定义任务变异、难度层级、成功条件；3) 提供33个可复用技能与人工编写的计算图参考解法；4) 通过仿真真实物理交互实现故障诊断。

**🔧 技术方法**

基于NVIDIA Newton物理引擎与Isaac Lab Arena的GPU加速仿真；使用Graph-as-Policy (GaP)框架和MORSL技能库；视觉分割采用SAM3；行为域定义语言BDDL；实验中对比人类编写图、Claude Code及自动生成的GaP图。

**📊 数据集**

55+工业工具、机械零件、医疗用品等资产库，覆盖工具、齿轮、注射器、连接器、柔性电缆等；任务实例通过控制的变异轴（物体形状、姿态、密度、布局、尺寸等）生成。

**📈 对比分析**

通过成功率（%）与中位循环时间（s）对比不同系统（Human–Newton/ILA、Claude Code、GaP自动生成）在六个任务及两难度层级下的表现。结果显示GaP整体优于Claude Code，但仍低于人工图；任务难度越大成功率下降、循环时间上升；失败主要来自运动规划。

**⚠️ 局限性**

仍存在技能组合与参数化不完善导致自动图性能落后；基准依赖仿真，缺乏真实机器人验证；对极端变异和更复杂物理交互（如电缆力学）支持有限；基准与物理引擎差异导致结果可迁移性不足。

---

## 679. Adaptive Multi Mode: Budget-Aware Routing of Optimizers By Online Landscape Probes

**arXiv ID:** 2611.06879 | [PDF](https://arxiv.org/pdf/2611.06879v1)

**作者:** Benjamin Jacob Bodner `[一作]` `[通讯]`, Benjamin Jacob Bodner

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并评估了 Adaptive Multi Mode (AMM) 路由器，它通过在线景观探测在已知预算下为连续黑盒优化问题动态分配多种经典优化器。

**💡 创新点**

首创在已知预算前提下利用低成本在线探测（曲率、平坦、可加性等）判断各优化器假设是否适用，并采用自动化预注册的开发循环，保证在未见实例上保持鲁棒性。

**🔧 技术方法**

整合了 SciPy/NumPy 的多种经典优化器（SLSQP、L‑BFGS‑B、COBYQA、STEP、MLSL、(1+1)-CMA‑ES、IPOP‑aCMA‑ES、ATM 随机搜索等），并实现了基于阈值统计的决策树路由；使用 COCO 平台进行实验评测。

**📊 数据集**

在 COCO bbob 基准集的 101–110 个隐藏实例、MA‑BBOB 公共行列 900–999，以及旋转、盒约束和仿射变换的 bbob 变体上进行评测。

**📈 对比分析**

与 7 种竞争方法（IPOP‑CMA‑ES、BIPOP‑CMA‑ES、SLSQP、NEVERGRAD 的 NgIohTuned、HCMA 等）以及一系列控制实验（单引擎、手工分配、贪心调度、最佳引擎选择器、后验最优等）在 10、30、100、300、1000 倍预算上比较；AMM 在 382/600 个功能-维度-预算细胞中获得最高目标分享，平均领先 IPOP‑CMA‑ES 0.12–0.31，且在 100 以上预算下路由贡献显著。

**⚠️ 局限性**

需要已知预算、阈值与探测依赖 bbob 的尺度与坐标，旋转后可加性、坐标分离探测失效；对极小/极大预算、未见维度、非盒约束、噪声或低成本函数场景表现受限；且对计算时间与内存开销未做系统评估。

---

## 680. CompressionBench: How can your coding agents compress LLMs?

**arXiv ID:** 2611.06858 | [PDF](https://arxiv.org/pdf/2611.06858v1)

**作者:** Xinrui Zhong `[一作]` (Carnegie Mellon University), Beidi Chen `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了一个终端任务，让编码型智能体在给定模型、存储上限、硬件与时限约束下，自主压缩大型语言模型并实现推理代码；对 36 次实验的压缩模型进行验证和评估。

**💡 创新点**

创新点在于：①把压缩问题建模为终端任务，使得智能体必须同时做压缩算法选择、精度分配、参数裁剪与实现实现；②使用完整模型的似然反馈与局部指标共同引导搜索；③提出独立验证流程（存储、答案准确率、执行完整性），并在多个压缩策略上对比实验。

**🔧 技术方法**

技术手段包括：量化（GPTQ、AWQ、SmoothQuant、QuIP 等）、旋转与 Trellis 码、低秩表示、剪枝、知识蒸馏、动态分辨率、混合精度分配、GPU 预取与分块推理等；使用 vLLM、llama.cpp、Claude Code 等工具构建与评估。

**📊 数据集**

数据集：源模型权重（Nemotron‑3.5、Qwen3.5、Qwen3.8），FineWeb、UltraChat‑200k、WikiText‑103 用于校准/恢复；MATH‑500 用于局部生成测试；验证时使用 GSM8K、HumanEval、TriviaQA 共 50 题/任务；搜索与验证均基于 Dolma‑3 生成的 512‑token 连续文本。

**📈 对比分析**

与基准比较：在 6 种任务（不同模型+不同存储上限）下，29/36 方案通过独立验证；平均相较 BF16，PPL 上升 3.4%，下游准确率 77.3%（BF16 79.6%）。与统一 GPTQ 及公开量化版本对比，智能体压缩模型在 PPL 与答案准确率上均优于大多数公开压缩模型；但推理速度与某些硬件环境仍不如原生 BF16。

**⚠️ 局限性**

局限性包括：①完整模型似然并不能完全捕捉生成能力，导致部分模型在验证中出现答案回退；②对特定 GPU 与 vLLM 实现的依赖，难以推广到其他硬件；③实验规模受限于 36 次压缩，未覆盖更大规模模型或更低存储阈值；④速度提升受压缩表示与执行路径耦合影响，仍需进一步优化。

---

## 681. Trainspotting: Learning Object-Centric 3D Reconstruction from Found Photos

**arXiv ID:** 2611.06849 | [PDF](https://arxiv.org/pdf/2611.06849v1)

**作者:** Hanyu Chen `[一作]` (Cornell University), Noah Snavely `[通讯]` (Cornell University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个规模达12,068个物体、311,886张照片的公开数据集（来自Wikimedia Commons），并利用该数据集对现有的前馈3D重建模型进行微调，显著提升其在“野外”物体图像集上的相机姿态、深度和几何重建质量。

**💡 创新点**

创新点包括：①提出“in‑the‑wild”物体重建数据集，解决传统单一捕获场景训练不足的问题；②设计结合稠密学习匹配与几何验证的重建管线；③在该数据集上微调VGGT‑Ω后，实现跨场景通用性的显著提升。

**🔧 技术方法**

核心技术为：稠密匹配（RoMa v2）、几何验证（MAGSAC++、循环一致性等）、多视角立体（Cas‑DiffMVS）与单目深度检验（MoGe‑2）、前馈网络VGGT‑Ω的微调，及使用SAM3进行物体门控与掩膜生成。

**📊 数据集**

使用的数据集为作者自建的“Wikimedia Commons Object Dataset”（简称WCO）和基准数据集NAVI in‑the‑wild、7‑Scenes、Neural RGB‑D、DTU、ETH3D。

**📈 对比分析**

评估方式：相机姿态AUC@5/30、深度AbsRel、δ<1.03、点云准确度/完整度等指标。微调后的模型在WCO和NAVI in‑the‑wild上相机AUC@30从25提升至81、完成度提升显著；在传统基准上性能基本保持不变。

**⚠️ 局限性**

局限性：①数据集仍以公开照片为主，缺乏极端光照/材质变化；②微调仅针对VGGT‑Ω，其他模型效果未知；③部分物体（如纹理稀疏、金属表面）仍易出现错误；④未对物体时间演化（如颜色改变）进行建模。

---

## 682. Context-Aware Autoregressive Image Generation for Emerging Reasoning Properties

**arXiv ID:** 2611.06838 | [PDF](https://arxiv.org/pdf/2611.06838v1)

**作者:** Jixuan Ying `[一作]` (Johns Hopkins University), Feng Wang `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在图像自回归生成中根据全局语义上下文动态决定下一个生成位置的Saccade AR（SAR）框架，提升结构鲁棒性与推理能力。

**💡 创新点**

创新点在于仅通过推断阶段对候选位置进行语义相似度排序，实时重排生成顺序，无需额外参数或训练，适配多种自回归范式。

**🔧 技术方法**

核心技术是使用条件锚点（如文本嵌入或类标记）与候选位置特征计算余弦相似度或L2距离进行排序；兼容掩码自回归（MAR/MaskGen）和尺度自回归（VAR/Infinity）。

**📊 数据集**

使用公开基准数据集GenEval、DPG‑Bench、MJHQ‑30K FID以及自构建的长提示推理评测集；对比 ImageNet 类条件实验。

**📈 对比分析**

与随机、顺序、从内向外以及 MaskGIT 的置信度选择等方案对比，SAR 在 GenEval 的整体通过率提升约3–5个百分点、DPG‑Bench 各类别 L1 均有提升、长提示评测平均得分提升≈0.5，FID 维持与随机相同。

**⚠️ 局限性**

局限性包括：仅适用于自回归模型，需依赖有效的锚点；对非自回归或扩散模型尚未验证；对生成速度影响极小，但在更大模型或多GPU环境下仍需评估。

---

## 683. LooM: Loop-Invariant Memory for Causal Encoder-Decoder Language Models

**arXiv ID:** 2611.06835 | [PDF](https://arxiv.org/pdf/2611.06835v1)

**作者:** Lele Yang `[一作]` (Beijing University of Posts and Telecommunications), Weiran Xu `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LooM（Loop-invariant Memory）方案，在因果编码器-解码器（CED）架构中仅循环解码器的读取层，并采用一次性发布（publish-once）的KV缓存，允许每层独立循环，实现从3到21层调用的可调计算预算；通过Random-Depth训练让单一模型在所有预算下都能运行；在相同缓存大小下，LooM在语言建模、零样本精度和长上下文性能上优于单遍模型。

**💡 创新点**

创新点在于：1）循环读取层时只写入一次本地KV，避免缓存随循环次数增长；2）每层独立循环，支持任意计算预算而无需路由器；3）采用Random-Depth跨预算训练，单模型即可覆盖所有预算；4）通过静态路径锁定在验证集上选择最佳路径，实现无路由器的动态预算分配。

**🔧 技术方法**

技术手段包括循环Transformer、因果编码器-解码器架构、KV缓存管理（一次性发布）、Random-Depth随机深度采样训练、静态路径搜索、规划器对比、分组内存写入实验。

**📊 数据集**

使用5B个FineWeb-Edu目标token作为训练数据；评估数据集包括LAMBADA、HellaSwag、PIQA、WinoGrande、ARC-Easy、ARC-Challenge、WikiText-2、PG19以及RULER检索任务。

**📈 对比分析**

与单遍模型、Fixed4（仅21循环）、Self-Distilled、Planner等对比。评测指标为测试交叉熵（CE）、零样本平均准确率、长上下文下的PPL；在21次循环时，Random-Depth CE比单遍低0.020 nat，零样本平均高0.79点，LAMBADA PPL从180.4降至134.4；在PG19、检索任务上也显示显著提升。

**⚠️ 局限性**

实验仅基于单一204M参数模型、单一随机种子，缺乏规模化验证；延迟测量使用未优化的参考实现，未评估批量吞吐量；单遍模型未匹配循环模型的计算量；部分对比实验（如完整解码器循环）仍处于预研阶段。

---

## 684. Rules, Traces and Routing Statistics: Training and Calibration Data for Open Models on ARC-AGI-2

**arXiv ID:** 2611.06827 | [PDF](https://arxiv.org/pdf/2611.06827v1)

**作者:** Nguyen Minh Thang `[一作]` (VNU University of Science), Trinh Thi Phuong `[通讯]` (VNU University of Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨在限定硬件与时间预算（四块NVIDIA L4 GPU，最多12小时）下，哪种训练与校准数据能帮助三种开源模型（Qwen3-4B网格模型、Qwen3-8B推理模型、Qwen3.8-Flash-Next Mixture‑of‑Experts模型）在ARC‑AGI‑2任务上取得较好成绩，并评估数据、推理流程与模型调整对性能的贡献。

**💡 创新点**

创新点在于①构建了12,000条行为唯一、可识别的Typed规则和95,993条严格门控生成的任务；②系统性比较了持续微调、合成任务、形状平衡任务等多种数据对Qwen3-4B网格模型的提升；③对Qwen3-8B推理模型进行自我答案验证的拒绝采样微调，并设计了多步骤推理与程序校验的推理流程，显著提升性能；④对Qwen3.8-Flash-Next模型进行专家剪枝，利用路由权重与ARC任务统计构造专业化专家列表，显著扩展KV缓存与吞吐量。

**🔧 技术方法**

使用的技术包括Typed Domain‑Specific Language（DSL）规则生成与行为签名检验、LoRA微调、深度搜索与概率阈值筛选、三次请求推理框架（思考→答案→程序）、FP8/FP4量化、Merlin vLLM推理引擎、专家权重衰减（REAP）与自定义专家列表。

**📊 数据集**

数据集主要为：1) 12,000条Typed规则与对应的95,993条合成任务；2) 1,920条真实ARC‑AGI‑1/2任务（训练与验证）；3) 72个手写程序规则的Shape‑Balanced生成器任务；4) 2,231条合成任务与1,036条自我答案验证的推理轨迹。

**📈 对比分析**

实验通过189次在隐藏半私有任务上的12小时全量跑测得分，比较不同训练/推理配置的分数。Qwen3-4B网格模型从基线28.96%提升至最佳34.86%；Qwen3-8B推理模型从基线26.81%提升至44.03%；Qwen3.8-Flash-Next在专家剪枝后KV缓存提升至1.96M token，单次跑分56.67，显著超过基线。

**⚠️ 局限性**

限制包括：①合成任务无法单独提升真实任务性能；②推理流程对硬件/量化配置高度敏感，易导致超时；③专家剪枝虽提升吞吐但对低频任务的泛化未知；④规则与任务门控过于严格，导致与实际ARC-AGI‑2任务的分布差异，影响模型的泛化与鲁棒性。

---

## 685. A Framework for Context Enrichment and Selective Robustness in LLM Social Agents

**arXiv ID:** 2611.06914 | [PDF](https://arxiv.org/pdf/2611.06914v1)

**作者:** Mattia Greiche `[一作]` (McGill University), Maximilian Puelma Touzel `[通讯]` (Mila)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型（LLM）在社会模拟中的上下文丰富化进行系统化控制，提出了选择性鲁棒性评估准则，并在一个Prisoner’s-Dilemma类似游戏中验证不同上下文条件对代理行为的影响。

**💡 创新点**

创新点：①将上下文丰富化细分为中性填充、通用社会背景、描述性决策定位、规范性决策定位四种功能化条件；②提出选择性鲁棒性（表面不变性+对世界变化的方向性响应）作为验证标准；③通过规则化提示替换与元模型评估实现对代理行为的可控实验。

**🔧 技术方法**

技术手段包括：规则化提示替换、元模型评估（Surface invariance + World responsiveness）、概率抽取与统计（Bootstrap 95% CI）、SiliSocS模拟平台构建实验环境。

**📊 数据集**

使用的“数据集”是基于SiliSocS构造的Prisoner’s-Dilemma类似游戏场景，包含两种社会世界（区域治理与本地治理）以及四种意义保持的表面形式；模型输入来自GPT-5.6 Sol与Terra两大LLM。

**📈 对比分析**

比较方法：交叉设计（5个上下文条件 × 4种表面形式 × 2个世界）测量每种条件下的平均表面敏感度和对世界差异的响应差值。结果显示，描述性与规范性决策定位能显著提升对世界变化的响应，且Sol在保持表面稳定性方面优于Terra。

**⚠️ 局限性**

局限性：仅涵盖单一决策场景与两款同类模型，缺乏跨模型族与多决策的普适性验证；表面形式的语义等价未由人类评估；可能存在信息显著性与决策定位混淆；未评估模型响应幅度与人类实际行为的一致性。

---

## 686. Occlusion for Free: Detecting Occlusion from the Disagreement Between Two Monocular Distance Cues

**arXiv ID:** 2611.06899 | [PDF](https://arxiv.org/pdf/2611.06899v1)

**作者:** Chibueze Praise Onyenweaku `[一作]` (University of Lagos), Obed Honour Eje `[通讯]` (Federal University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种无需训练、无需标注的遮挡估计方法，利用检测框中的明显高度与单目深度的分歧来判断行人是否被遮挡。

**💡 创新点**

创新点在于发现两种本已存在的距离估计（显式高度和深度最小值）对遮挡的敏感性不同，可通过它们的比值（简化为 d_min·h_px）直接得到一个可排名的遮挡分数，且不依赖相机参数或尺度校准。

**🔧 技术方法**

技术核心包括 YOLOv8 目标检测、Depth Anything V2（单目度量深度）以及对检测框内最小深度和框高的组合计算。

**📊 数据集**

实验使用 KITTI、CityPersons、合成 CARLA 场景以及自采的拉各斯（Lagos）数据集。

**📈 对比分析**

在 KITTI 上，遮挡分数对重度遮挡的 AUC 达到 0.909（匹配检测结果）或 0.888（全部标注行人），在固定距离区间内保持 0.894–0.925 的高水平，优于单纯距离特征；在 CARLA 合成数据上，AUC 为 0.819，明显高于仅用距离的 0.644。

**⚠️ 局限性**

局限性包括：依赖高质量度量深度模型，深度误差偏移会导致分数偏向距离；目前仅针对行人遮挡验证，跨类别或多摄像头场景需要进一步验证；以及对极端遮挡或深度模糊区可能仍表现不佳。

---

## 687. Online Edge-Weighted Matching Beyond 0.55

**arXiv ID:** 2611.06896 | [PDF](https://arxiv.org/pdf/2611.06896v1)

**作者:** Alireza Farhadi `[一作]` (Profound), Mohammad Hajiaghayi `[通讯]` (University of Maryland)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种随机算法，在带自由放弃的边权在线二分匹配问题中实现了竞争比>0.556（先前最高为0.5368）。

**💡 创新点**

创新点在于构建了状态依赖的主对偶分析框架，并通过对多路在线相关选择（OCS）的负关联性质和最优 win map 的设计，消除了早期理论中的“gap”限制，得到更好的竞争比。

**🔧 技术方法**

主要技术包括：多路OCS 与单路OCS 的结合、状态依赖的潜在函数与双重递推、负关联证明、win map 的显式优化、有限校验（计算机辅助）与 Lean 形式化验证。

**📊 数据集**

论文未使用实验数据集，而是完全基于理论证明和计算机辅助检验（符号计算、整数规划与验证程序）。

**📈 对比分析**

与 Blanc & Charikar 以及之前的 0.5086/0.519 等算法相比，本文通过更精细的分析与优化实现了从 0.5368 提升到 0.556 的竞争比；实验验证通过符号计算与验证程序确认了理论值。

**⚠️ 局限性**

局限性包括：算法仅适用于带自由放弃的边权匹配，预处理与参数规模较大；对一般子模收益分配问题无法直接推广；实现需要复杂的计算机辅助与形式化证明，实际部署成本较高。

---

## 688. Prophet Secretary for Matroids and XOS Valuations: A Strict Separation at $1-1/e$

**arXiv ID:** 2611.06891 | [PDF](https://arxiv.org/pdf/2611.06891v1)

**作者:** Hossein Esfandiari `[一作]`, MohammadTaghi Hajiaghayi `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了 matroid prophet‑secretary 问题与对应的 XOS 组合拍卖问题，证明在 matroid 情况下可取得比 1-1/e 更好的竞争比（1-1/e+ε₀，ε₀≈10⁻⁶），而在 XOS 拍卖中 1-1/e 是最优的；

**💡 创新点**

主要创新在于：① 引入参考分布与删除盈余的概念，利用矩阵单调性与泊松圆化秩的凹性，实现对 matroid 的 1-1/e+ε₀ 的提升；② 通过基交换的“虚拟继续”机制，保持对基的概率分布并在最后阶段获得额外收益；③ 对 XOS 拍卖构造了信息瓶颈实例，证明即使买家独立同分布、值为 {0,1} 的加法子句，也无法突破 1-1/e 的上界；

**🔧 技术方法**

技术手段包括：参考分布的上尾重标定、删除盈余与稳定盈余的分析、基准函数的潜能搜索、Poisson 概率与基交换的连续化、层采样与重要性抽样、以及对离散概率的有限精度实现；

**📊 数据集**

论文不依赖任何外部数据集，而是对任意给定的有限离散分布（通过其概率表描述）进行分析；

**📈 对比分析**

与先前的 1-1/e 结果相比，matroid 情况下竞争比提升了约 10⁻⁶，虽然幅度极小，但提供了严格的上界改进；在 XOS 拍卖中，新构造的实例验证了 1-1/e 的不可逾越性；

**⚠️ 局限性**

局限性包括：提升幅度极小，主要依赖于对分布的完整知识和独立性；算法在实现上需要较高的样本与查询复杂度（O(N⁴) 期望查询）；且对 XOS 结果仅在独立同分布且子句系数为 {0,1} 的情形下给出最优性，尚未扩展到更一般的加法子句或相关分布。

---

## 689. Systolic Linear Attention: Co-Designing Tile-Grouped Gated Delta Attention for TPUs

**arXiv ID:** 2611.06828 | [PDF](https://arxiv.org/pdf/2611.06828v1)

**作者:** Gagik Amirkhanyan `[一作]` (Google), Nuojin Cheng `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新型稀疏注意力架构（SLA），将线性注意力与门控组查询注意力（GQA）、RoPE、Sparse MoE 等技术结合，构建可扩展的多尺度模型。

**💡 创新点**

创新点包括：① 将线性注意力与门控组查询注意力耦合，显著减少矩阵乘法规模；② 采用多尺度参数化和异步块求逆（Neumann逆），实现高效的稀疏混合器；③ 兼容 FP8 低精度状态和动态量化（PTQ），在保持性能的同时降低显存占用。

**🔧 技术方法**

技术细节：TPU v7x 上的 Pallas/Mosaic 内核、SLA 线性注意力、KDA、MaxText、GQA+RoPE、Sparse MoE、FP8 PTQ、GDN‑2 等；模型使用 AdamW、RoPE、top‑k 路由、混合专家等。

**📊 数据集**

数据集：Dolma 3.5 中文文本，覆盖多种长度（4K–64K）和多种验证集（PPL、zero‑shot 9‑task、4‑candidate QA），训练范围从 6.5B–63.1B 参数、15B–120B token。

**📈 对比分析**

与 KDA、MaxText 等基线对比，SLA 在相同 token 量下实现 1.1–1.2× 的吞吐量提升，长序列（32K/64K）性能提升 30–50%，perplexity 与基线保持相当或略优；在 FP8/PTQ 模式下保持稳定性能。

**⚠️ 局限性**

局限性：高度依赖 TPU 的 Pallas/Mosaic 内核，迁移到其他硬件需重构；极大规模（>120B）下内存占用仍待进一步优化；某些头共享策略对极短序列的精度有轻微影响。

---

## 690. Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models

**arXiv ID:** 2611.06817 | [PDF](https://arxiv.org/pdf/2611.06817v1)

**作者:** Juncheng Ma `[一作]`, Daquan Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Open‑HumanScale 方案，构建并公开 5 万小时人类第一人称操控视频语料，使用统一动作表征进行世界-动作模型预训练，并进一步引入人机混合预训练生成 HumanWAM‑Vanilla 与 HumanWAM‑Goal 两个可在零样本与后训练情境下执行多任务的机器人基础模型。

**💡 创新点**

核心创新包括：① 通过 5 万小时人类数据实现的经验规模化与对比实验验证了对数线性扩展律；② 统一跨源动作对齐与基于视觉目标的条件化实现了跨 embodiment 的零样本执行；③ 在机器人与人类数据混合预训练中实现了显著提升的样本效率与场景泛化。

**🔧 技术方法**

采用世界‑动作 Transformer 框架、逆动力学 IDM 注意力、联合视频‑动作联合去噪与 UniPC 采样，并利用统一的 16‑维末端执行器动作空间与 49‑维混合动作表示。

**📊 数据集**

使用公开 egocentric 数据集 HumanNet、EgoSuite‑Open100K、EgoDex、Gen‑HumanEgo、EgoVerse 组成 5 万小时的 Open‑HumanScale‑50K 语料，并收集约 1.46 万小时的真实机器人演示作为混合预训练。

**📈 对比分析**

通过在 RoboTwin 2.0 与 RoboDojo 基准上进行清洁与随机环境评估，HumanWAM‑Goal 在后训练后实现 88.65% 平均成功率，超过现有最高 84.60%；在零样本场景下，混合预训练模型平均成功率提升至 13.94%，任务覆盖率从 8/50 递增至 22/50。

**⚠️ 局限性**

限制在于仍需大量人工标注的 egocentric 视频与语言描述，对不同机器人平台的适配依赖统一的动作空间，且在极端高精度或长时序任务中，零样本性能与实时采样速度仍有提升空间。

---

## 691. Perceptive Humanoid Motion from Simulation

**arXiv ID:** 2611.06908 | [PDF](https://arxiv.org/pdf/2611.06908v1)

**作者:** Siyuan Hu `[一作]` (Apple), Vladlen Koltun `[通讯]` (Apple)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一套端到端的人形机器人自主系统，能够在未见过的真实房间里完成从自然语言指令到整个身体运动的映射，并在实际硬件上实现零射击转移。

**💡 创新点**

①通过在模拟中全自动生成成千上万个具有配对视觉-语言-运动的 egocentric 数据，摆脱人工操作；②设计了三层层次架构，包含 VLM 任务分解器、基于 DiT 的运动生成器以及利用深度感知的实时运动跟踪器；③实现了完全离线、异步推理，保证连续运动计划。

**🔧 技术方法**

采用 Qwen3.5/3.6 VLM、基于 Diffusion Transformer 的运动生成器、Kimodo 运动扩散模型、Infinigen 生成房间、深度视觉感知与多技能强化学习的运动跟踪器，以及实时深度匹配的 Stereo 视觉。

**📊 数据集**

300 小时由 1,000 个程序化生成的房间合成的 egocentric 视觉-语言-运动数据集，使用 Infinigen 生成室内布局与家具，并通过 Kimodo 生成相应动作；未使用真实扫描数据。

**📈 对比分析**

与零射击、Vlk、Grail、Tango 等现有方法在 6 个未见真实房间 360 次试验中，单任务平均成功率 91%（仿真）和 88%（真实），多任务平均 70%，显著优于对比方法； Ablation 证明环境多样性、实时分块、流匹配解码器、深度感知对性能关键。

**⚠️ 局限性**

需要人工撰写新任务的文本与关键帧；对接触感知不足，无法处理完全遮挡或无视线的物体；运动跟踪器只接收轨迹，缺乏对未来世界状态的理解；任务分解层相对简单，随着任务数量增长需更智能的规划器。

---

## 692. Reinforcement Learning Enables Scaling Generalist Robot Policy Improvement

**arXiv ID:** 2611.06905 | [PDF](https://arxiv.org/pdf/2611.06905v1)

**作者:** Michal Korniak `[一作]` (ETH Zurich), Andrew Wagenmaker `[通讯]` (UC Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究在多任务环境中对通用机器人策略进行在线强化学习（RL）自我改进，探讨如何通过与不同任务交互来提升整体性能，并在训练分布之外实现泛化。

**💡 创新点**

核心创新在于：①首次系统展示了“多任务在线RL”能够让通用策略在训练任务上同时提升并在未见任务上获得显著泛化；②证明了价值函数驱动的改进（OGPO、Best‑of‑N）比直接监督微调（FSFT）更能促进跨任务泛化；③提出了可扩展的实验范式和量化指标，揭示任务多样性与泛化性能之间的关系。

**🔧 技术方法**

使用的技术包括：
- 价值函数学习（基于TD或MC的 Q/V 网络）
- Best‑of‑N（BoN）采样在测试时对动作进行价值排序
- 生成式策略优化 OGPO（Off‑Policy Generative Policy Optimization）
- 过滤监督微调 FSFT
- 统一的训练协议：多任务在线收集（K=20 经验/任务/epoch，M=10 迭代）以及梯度步数分配。

**📊 数据集**

采用的主要数据集：
- 12S 家庭物品抓取与提起 benchmark（18 种物体、55‑56 场景/物体），用于控制任务多样性和评估泛化；
- MetaWorld 机器人抓取与操纵 suite（从子集采样），用以验证在更大变异度任务上的鲁棒性。

**📈 对比分析**

与单任务微调（FSFT）和仅测试时改进（BoN）相比，实验表明：
- 在训练任务上，OGPO 与 BoN 分别比 FSFT 提升 43‑50pp 和 15‑21pp；
- 在未见物体/场景上，OGPO 与 BoN 能获得最高 58pp（场景变化）和 39pp（物体变化）的提升，而 FSFT 甚至出现遗忘；
- 计算效率上，BoN 在 16 任务上仅需 2.9 倍更少的梯度步数即可达到单任务专家的表现。

**⚠️ 局限性**

局限性包括：
- 评估仅在仿真抓取环境中进行，缺乏真实硬件验证；
- 基础策略需已具备一定零射击成功率，难以推广到更复杂或全新任务；
- 研究聚焦于抓取与提起任务，未涉及步态、全身控制等其它机器人领域；
- 需要大量多样化任务与计算资源，现实部署仍面临成本与数据收集挑战。

---

## 693. GH-AWTrace: A Large-Scale Dataset of GitHub Agentic Workflow Execution Logs

**arXiv ID:** 2611.06866 | [PDF](https://arxiv.org/pdf/2611.06866v1)

**作者:** Jasem Khelifi `[一作]` (University of Quebec), Mohamed Aymen Saied `[通讯]` (Laval University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了 GH-AWTrace 数据集，链接 25,014 个经过验证的 GitHub Agentic 工作流、4,112,526 条运行记录和 562,877 条非空执行日志档案，保存了原本会被 GitHub 日志保留策略删除的执行证据，并与仓库元数据和生成的工作流配置关联。

**💡 创新点**

创新点在于首次系统地收集并归档 agentic 工作流的实际执行日志，提供了从工作流定义到运行结果之外的中间行为证据，支持对工具使用、失败原因、重试策略和多代理协作等方面的研究。

**🔧 技术方法**

使用 GitHub REST API、Code Search、ZIP 打包、JSON 结构化存储、SHA‑256 校验、Python 脚本等技术实现工作流识别、运行记录抓取和日志归档。

**📊 数据集**

使用的数据集为 GH-AWTrace 本身，包括 25,014 条已验证的 gh‑aw 工作流、5,051 个仓库、4,112,526 条运行记录和 562,877 条非空日志档案（共 56.1 GB）。

**📈 对比分析**

本文并未进行传统意义上的方法比较或性能评估，而是通过案例演示和统计分析说明数据的可用性和潜在研究价值；因此不存在基准对比或性能指标。

**⚠️ 局限性**

局限性：日志覆盖受 GitHub 90 天保留政策和访问限制影响，导致 821,672 条运行缺失日志；部分日志为空；日志仅为执行片段，无法完整重建代理决策链；数据仅覆盖已验证的 gh‑aw 工作流，可能不具代表性。

---

## 694. Deterministic Fault-Tolerant Connectivity Oracles with Subquadratic Space

**arXiv ID:** 2611.06857 | [PDF](https://arxiv.org/pdf/2611.06857v1)

**作者:** Merav Parter `[一作]` (Weizmann Institute), Elad Tzalik `[通讯]` (Weizmann Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了在任意数量的边失效下仍能使用子二次空间（O(n^{7/4})）的确定性连通性预言机，并给出在失效数 f 限制时的空间上界 O(f^{2/5}n) 与下界 Ω(f^{1/4}n)。

**💡 创新点**

突破了此前确定性预言机需要 O(fn) 或 Θ(n^2) 空间的瓶颈，首次实现子二次空间并给出确定性与随机预言机之间的多项式空间分离；同时引入了“失效查询即信息来源”的视角与掩码族技术。

**🔧 技术方法**

核心技术包括：lossless expander 线性草图、割稀疏化与层级稀疏化、可实现的掩码族 (O(n^{3/2}) 误差)、基于稀疏恢复的确定性边恢复、以及高度可分解的 expander 分解。

**📊 数据集**

论文为理论研究，未使用具体实验数据集，所有结果均在理论上给出证明。

**📈 对比分析**

相较于之前最优的随机预言机仅需 O(n) 位，确定性预言机在空间上仍多出多项式因子；但在 f 较小的情况下（f≤n^2-ε）实现了 O(√f·n) 空间的快速查询预言机，且查询时间为 O(f)。

**⚠️ 局限性**

局限性包括：通用预言机的查询时间指数级（2^{min{f,n}}）；空间下界仅在理论上给出，实际构造仍需 n^o(1) 级的额外空间；并且在实现近似距离预言机时需额外的 O(n^o(1)) 预处理与查询时间。

---

## 695. Visual Token Routing: Learning to keep tokens that matter

**arXiv ID:** 2611.06854 | [PDF](https://arxiv.org/pdf/2611.06854v1)

**作者:** Pranjal Aggarwal `[一作]` (Carnegie Mellon University), Sean Welleck `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种联合训练框架 Visual Token Routing (VTR)，使计算机使用代理只处理对下一步动作至关重要的视觉令牌，从而显著降低视觉令牌数量。

**💡 创新点**

创新点包括：① 在视觉编码器与语言模型之间插入轻量级时空路由器，动态决定每个视觉令牌是否进入语言模型；② 通过自适应损失与拉格朗日乘子（dual ascent）实现“保持尽可能少的视觉令牌”与“保持预测准确性”的平衡；③ 引入 anchor 机制与滑动路由器，使路由器能在长历史中只关注变化且对动作有用的区域。

**🔧 技术方法**

主要技术手段是：视觉语言模型（Qwen3.5-2B）、三层 MLP 时空路由器、keep 概率缩放、自适应 λ 训练、anchor 与滑动路由器、基准模型（全分辨率与仅 anchor），以及多任务联合训练。

**📊 数据集**

使用的数据集包括：桌面任务的 OSWorld 与 CUA-World（1920×1080 分辨率、2040 视觉令牌），以及 36 款视频游戏（640×360 分辨率、220 视觉令牌），训练集来自 CUA-World 轨迹与游戏录像。

**📈 对比分析**

与全分辨率密集模型、降分辨率、像素差（pixel-diff）、H.264 编码的保留方式以及冻结路由器基线相比，VTR 在 OSWorld 与 CUA-World 上保持 2.5–2.9 倍的视觉令牌压缩，视频游戏中压缩 5.8 倍，推理成本降低至 3.3 倍、prefill 时间加速至 2.9 倍，并在 OSWorld 上比最优基线高出 12.8 分（最高 18.1 分），完全匹配全分辨率模型性能。

**⚠️ 局限性**

局限性包括：① 仍需在全分辨率视觉令牌上预训练，路由器学习需要足够的训练样本；② 对极高频实时任务（如每秒多帧游戏）可能需要更精细的时间窗口或更高帧率下的实验；③ 路由器设计主要针对视觉语言模型，其他模态或多模态场景的迁移尚未验证；④ 在非常小的视觉区域（例如细微文本）被误删的风险；⑤ 仍无法完全替代全图信息，某些任务中过度压缩会导致细节丢失。

---

## 696. Where to Connect? Boosting MLLMs via Dynamic Gated Pathways across ALL ViT and LLM Layers

**arXiv ID:** 2611.06844 | [PDF](https://arxiv.org/pdf/2611.06844v1)

**作者:** Yingying Yan `[一作]` (Northwestern Polytechnical University), Qifeng Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Dynamic Gated Pathways（DGP），在多模态大型语言模型中实现 ViT 与 LLM 的全层动态路由，允许每个 LLM 层根据自身语义状态自适应选择不同层次的视觉特征。

**💡 创新点**

创新点：①全层（all-to-all）跨模态连接；②动态门控路由机制，使每层 LLM 在解码时自发决定视觉层级；③通过熵正则化防止路由集中，提升鲁棒性；④提供可解释的路由热图与视觉特征可视化。

**🔧 技术方法**

技术细节：共享多头注意力提取路由特征 → 层级特定 MLP 生成权重 → 加权融合多层 ViT patch → 残差门控注入 LLM；使用熵正则化、交叉熵训练；基于 LLaVA‑1.5、Vicuna‑7B/13B、CLIP‑ViT-L/14‑336 或 SigLIP2‑L 等骨干。

**📊 数据集**

训练数据：预训练 558K 条 LLaVA 训练集，SFT 665K 条混合数据；评测数据集涵盖学术 VQA（GQA、SciQA‑IMG、VQAv2、VizWiz）与 LMM 任务（SEED、LLaVA‑Wild、MM‑Vet、MMBench、POPE）。

**📈 对比分析**

与主流 MLLM（BLIP‑2、InstructBLIP、Qwen‑VL 等）和多层视觉融合基线（MMFuser、DenseConnector、Sparse Shortcuts、TGIF 等）对比；在 Vicuna‑7B/13B 上平均提升 2–4 分（如 GQA +4.2、SQA^I +4.2、VizWiz +3.1、MM‑Vet +3.3），在 Qwen3‑VL‑2B 和 LLaVA‑Next 上也保持 1–2 分的优势；参数与 FLOPs 增加仅 4–5%（显著低于性能提升）。

**⚠️ 局限性**

局限性：①仍依赖较大计算量，尤其是所有层联通；②在无熵正则化时路由易坍缩，需调参；③目前仅在 CLIP/Vision‑Transformer 及 Vicuna/LLaMA 等骨干上验证，其他视觉/语言模型的适配性待探；④冻结视觉编码器可能限制进一步改进；⑤对极端任务或小规模数据的泛化能力尚未充分评估。

---

## 697. Same Layout, Different Count: What Number Words Do Past Ten in Text-to-Image Models

**arXiv ID:** 2611.06840 | [PDF](https://arxiv.org/pdf/2611.06840v1)

**作者:** Liang Yue `[一作]`, Eric A. Vance `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究开源文本到图像模型在请求十个以上对象时的计数行为，采用同噪声实验和细粒度噪声重绘方法，结合人工计数和三种自动检测器评估数字词对图像中对象数量的影响。

**💡 创新点**

提出同噪声实验框架，区分图像布局与计数的作用范围，揭示数字词在“达成范围”之外几乎失去方向；证明在此范围外使用布局注入可显著提升计数准确率。

**🔧 技术方法**

使用 SD 3.5‑Medium、FLUX.1‑dev、FLUX.2‑klein、Qwen‑Image、Qwen2.5‑VL 等文本到图像扩散模型；同噪声实验、细粒度噪声重绘、数字词早期窗口时间测定、数字词与布局注入、三种自动计数器以及人工计数。

**📊 数据集**

从 LVIS 选取 64 种日常物体类别，每种配备单一随机种子噪声；人工计数数据来自 CloudResearch 平台，结合三种自动检测器生成的计数。

**📈 对比分析**

通过同噪声顺序、平均对数差、布局相关系数、检测器/人工计数比对等指标进行比较。结果显示：在 2‑8 个物体时几乎完美跟随；10‑25 个物体时模型表现差异明显，部分模型可松散跟随；超过 25 个物体时数字词失去方向，布局注入可将准确率从约 4% 提升至 59%（图像质量有一定下降）。

**⚠️ 局限性**

局限性包括：仅评估 5 个开源模型、单一 prompt 模板、最大 30 个物体、低分辨率；人工计数存在主观性；同噪声实验对细粒度噪声的依赖；未测试闭源模型或更长序列；未探索多物体或多类别提示场景。

---

## 698. Context--Answer Metamodels: Forecasting LLM Activations Before the First Answer Token

**arXiv ID:** 2611.06837 | [PDF](https://arxiv.org/pdf/2611.06837v1)

**作者:** Tommy He `[一作]` (Tufa Labs), Thomas Jiralerspong `[通讯]` (Mila, Université de Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于上下文激活的“元模型”来预测语言模型在随机解码下的答案激活（均值和分布）

**💡 创新点**

创新在于能在生成首个答案标记之前就预测答案激活，使用多种分布模型并实现跨模型、跨域的零样本转移

**🔧 技术方法**

采用线性岭回归、三层MLP、高斯分布估计和基于条件流匹配的生成模型，并使用自适应平均池化+单查询注意力进行特征编码

**📊 数据集**

使用LMSYS‑Chat‑1M（50万提示）、WeirdChat（2660提示）、IFEval（541提示）以及Qwen2.5、Qwen3.5、Gemma4模型的上下文激活数据

**📈 对比分析**

通过多种指标（mean R²、能量得分、方差排名等）与直接上下文基线、线性岭回归、MLP、Gaussian、Flow模型进行对比，实验显示：随提示规模增大，MLP和Flow在均值预测上优于岭回归；Gaussian在方差排名上表现最好；Flow在OOD上能量得分最低；并且预测结果保留行为判别信息

**⚠️ 局限性**

局限于特定层与模型、仅评估2–16次滚动、行为标签来源于自动化评分且缺乏人类验证、对解码规则、模板、截断长度和答案池化方式高度依赖

---

## 699. SpecCompact: Accelerating LLM Agent Context Compaction with Slot-Parallel Semantic Speculative Verification

**arXiv ID:** 2611.06876 | [PDF](https://arxiv.org/pdf/2611.06876v1)

**作者:** Yulu An `[一作]` (New York University), Xiangyang Yin `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SpecCompact 框架，利用槽级并行语义推测加速 LLM 代理的上下文压缩

**💡 创新点**

创新点在于将摘要拆分为独立的“槽”，在小型起草模型与目标模型之间实现并行草稿-验证-修复循环，并通过系统优化显著降低延迟

**🔧 技术方法**

使用小型 Qwen3‑0.6B 作为起草模型，Gemma4‑31B‑IT 或 Qwen3‑5‑122B‑A10B 作为目标模型，结合语义推测、共享前缀缓存、草稿缓冲、行级停止等技术

**📊 数据集**

采用 AppWorld、τ³、LiveClawBench 以及 LongBench（GovReport、QMSum）和 RULER 等公开数据集进行训练与评估

**📈 对比分析**

与 Hermes、OpenClaw 等传统压缩方法对比，SpecCompact 在 Gemma4 上实现 2.04–5.63 倍的压缩速度提升，在 Qwen3‑5 上实现 2.04–2.41 倍；任务完成率略有波动，部分模型在特定任务上性能略低但仍保持竞争力

**⚠️ 局限性**

局限在于未能在所有场景下提升压缩准确性，依赖起草与目标模型的对齐度；对不同槽信息来源的差异处理仍需改进

---

## 700. cuRoboV2-Fast: Fast Torque-Limited Collision-Free Motion Planning

**arXiv ID:** 2611.06913 | [PDF](https://arxiv.org/pdf/2611.06913v1)

**作者:** Balakumar Sundaralingam `[一作]` (NVIDIA), Stan Birchfield `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种单 GPU 规划器 cuRoboV2-Fast，能够快速生成满足碰撞、扭矩和动力学约束的可执行轨迹。

**💡 创新点**

创新点在于将 B‑spline 轨迹优化、GPU 并行 RRT 初始化、时间优化 TimeOpt、加速机器人运算和可微逆动力学融合到一条完整管线，并显著降低规划延迟与执行时间。

**🔧 技术方法**

使用的技术包括 B‑spline 轨迹优化、Band‑Newton 时间优化、CUDA 并行 RRT、加速正向/反向机器人动力学、可微逆动力学、AABB 自碰撞裁剪与差分采样等。

**📊 数据集**

实验数据集包括 MotionBenchMaker、MPINets、YAM 机器人平台的模拟与真实抓取任务，使用 Franka Panda、G1 机器人和 RTX 5090 / Jetson AGX Orin 设备。

**📈 对比分析**

与 cuRoboV2、cuRoboV1+TOPPRA、VAMP+TOPPRA 等方法对比，cuRoboV2-Fast 在 RTX 5090 上 3 kg 负载的平均规划时间仅 4.4 ms，比 cuRoboV2 快 14.7×，在 Jetson 上 25.99 ms，真实抓取周期平均 3.36 s，规划时间仅占 1.0% 的周期。

**⚠️ 局限性**

局限性包括对单 GPU 的依赖、对极端动态负载或非常复杂约束场景的鲁棒性尚未验证，以及规划仍依赖局部收敛而非全局最优。

---

## 701. MCP-Eco: A Linked Dataset of MCP Servers, Consumers, and Development Activity

**arXiv ID:** 2611.06897 | [PDF](https://arxiv.org/pdf/2611.06897v1)

**作者:** Khalid Tounoussi `[一作]` (University of Quebec), Yutaro Kashiwa `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了MCP‑Eco 数据集，将 Model Context Protocol 服务器项目、其能力、消费者配置声明与相关开发记录整合为一个关系型数据集。

**💡 创新点**

创新点在于三方面的联通：① 将服务器、消费者与开发活动三侧的数据整合；② 保留源信息与匹配方法，实现可追溯、可过滤的分析；③ 通过实时探测与目录、注册表搜索同步观测能力，首次实现广告与实际接口的可比对。

**🔧 技术方法**

技术方案包括 Python 爬虫、PostgreSQL 存储、Parquet 导出、GitHub REST API、Sourcegraph 搜索、实时探测（probe）以及 SQL 查询与脚本化分析。

**📊 数据集**

数据集包含 78,302 个候选服务器、16,165 个消费者仓库、9,200 条配置声明，并附带 1,733 万条拉取请求、620 万条评审、301 万条评论、991 万条问题、2,065 万条提交等开发记录，总计 1.69 万条表、1,688 万条记录。

**📈 对比分析**

通过示例 SQL 查询展示了对工具接口、配置实践和维护指标的对比分析；相较于现有的单侧（仅服务器或仅消费者）数据集，MCP‑Eco 在多维度分析上实现了更高的覆盖率与可追溯性，性能在大规模 Parquet 表查询中保持在秒级内。

**⚠️ 局限性**

局限性包括：① 仅采集公开可访问的候选与消费者；② 可能存在误匹配与缺失的实时接口观测；③ 缺乏运行时使用数据；④ 数据为快照，无法完整反映生态演化；⑤ 依赖第三方 API 的可用性与速率限制。

---

## 702. Nonuniform Buy-at-Bulk Network Design: Improved Approximation and LP Integrality Gap

**arXiv ID:** 2611.06894 | [PDF](https://arxiv.org/pdf/2611.06894v1)

**作者:** Iman Gholami `[一作]` (University of Maryland), MohammadTaghi Hajiaghayi `[通讯]` (University of Maryland)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种确定性多项式时间算法，能够在无向非均匀买入-批量网络设计问题中获得 O(log²h/(loglog h)²) 的近似比率，并同时给出了相同上界的自然多商品流 LP 的整数性缺口；

**💡 创新点**

创新点在于将多尺度网格、中心记录、递归附件、固定球覆盖以及高环度辅助图等技术相结合，实现了从 O(log⁴h) 到 O(log²h/(loglog h)²) 的显著改进，并保持预算分离、节点/边成本无关；

**🔧 技术方法**

核心技术包括：自然仿射 LP 松弛、基于混合距离的多层尺度划分、确定性固定球覆盖（CoverFixedTolls）、递归附件（ConnectEndpoint）以及高环度辅助图的边数上界分析；

**📊 数据集**

本文为理论工作，未使用任何实验数据集，所有结果均在抽象图模型和 LP 证明框架中给出；

**📈 对比分析**

与 Chekuri 等人提出的 O(log⁴h) 近似方案相比，本文算法在近似比率上实现了两次对数级别的改进，并且在任意正有理需求以及节点与边成本混合的情况下保持多项式时间复杂度；

**⚠️ 局限性**

局限性包括：近似比率仍为多对数级，且算法实现相对复杂；未进行实验评估，常数项及实际运行性能未知；在需求极端不平衡时仍受输入编码长度影响。

---

## 703. Procedural Adversarial Learning

**arXiv ID:** 2611.06890 | [PDF](https://arxiv.org/pdf/2611.06890v1)

**作者:** Jack Nugent `[一作]` (Princeton University), Jia Deng `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了程序化对抗学习（PAL）框架，并设计了基于CMA-ES的Simple-PAT算法，在程序化生成器空间中进行对抗训练和评估；

**💡 创新点**

创新点在于将可解释的程序化生成器与对抗训练结合，实现了对高维参数空间中极端样本的高效搜索，并引入跨模型评估提高鲁棒性；

**🔧 技术方法**

技术包括程序化生成器、零阶CMA-ES优化、MiniBatch梯度下降、CVaR和Tilted ERM等对抗训练策略；

**📊 数据集**

使用PAL-Dojo生成器，对四个视觉任务（表面法向估计、椅子分割、完整椅子分割、人类姿态估计）进行实验；

**📈 对比分析**

与均匀采样、CVaR、Tilted ERM等基线比较，Simple-PAT在跨模型评估下的最坏情况误差分别降低3.15×、2.63×，并在多数任务上保持良好的平均性能；

**⚠️ 局限性**

局限性包括对高维生成器评估依赖搜索能力，程序化生成器可能不覆盖真实数据分布，且仅在四个任务上验证，缺乏对更广泛任务的通用性验证。

---

## 704. Dull, Dirty, Dangerous? Working Conditions and the Limits of Occupational AI Exposure

**arXiv ID:** 2611.06888 | [PDF](https://arxiv.org/pdf/2611.06888v1)

**作者:** Siyi Yu `[一作]` `[通讯]` (Purdue University), Siyi Yu (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文利用O*NET与BLS等公开职业数据，构建了包含危险、脏污和乏味三类21项的工作条件指数，并与两种AI曝光度量（LLM曝光与AIOE）进行回归比较，探讨AI技术在不同职业中的暴露与工作条件之间的关联；随后对指数进行拆分、加权与调整（加入物理需求、职能准备、工资等）来检验关联的稳健性，并通过职业档案进一步说明高负担低曝光职业的多样性。

**💡 创新点**

创新点在于：①将经典的“Dull‑Dirty‑Dangerous”框架与最新的LLM与AIOE曝光度量相结合，系统评估两者在美国754个职业中的相关性；②通过对指数的拆分（重复与受限自由）揭示单一“乏味”得分掩盖的细节；③在回归模型中引入物理需求、教育培训与工资等职业特征，评估这些变量对AI曝光-条件关联的影响；④构建职业档案，展示同一高负担低曝光组内部不同条件配置，为HCI实验选址提供参考。

**🔧 技术方法**

使用的技术包括：标准化项与域平均值的统计标准化；多元线性回归（含职业族级固定效应）；聚类稳健误差估计（CR1）；部分R²计算；多重比较调整（Holm）；以及对指数构造的敏感性检验（等权重、完整项、移除物品、引入物理需求域等）。

**📊 数据集**

主要数据集：O*NET 31.0 Work Context（工作环境与危险、重复等指标）；Eloundou等人公开的LLM曝光度量；AIOE 2021版本；BLS Occupational Employment and Wage Statistics（OEWS）提供就业与工资；CPS 2022提供劳动力构成；CFOI 2024提供致命伤害比率；Anthropic 2026混合曝光度量。全部以美国六位数职业代码为链接单元。

**📈 对比分析**

比较方法：在每个职业族内进行回归，控制职业族固定效应，以两种加权方案（等职业权重与就业权重）估计曝光度量对条件指数的回归系数；随后逐步加入物理需求、Job Zone、工资等协变量进行调整，计算调整前后部分R²；使用Holm校正处理多重检验。性能表现：在未调整模型下，LLM与AIOE均与条件指数呈显著负相关，系数在-0.35至-0.53之间；调整后，LLM系数下降至-0.09~-0.15，置信区间含零；AIOE系数下降至-0.18~-0.26，置信区间仍不含零；拆分“乏味”域后，受限自由系数显著负相关，而重复系数不显著。

**⚠️ 局限性**

局限性：①指数仅覆盖危险、脏污和乏味三类条件，未包括安全感、意义感、社交关系等重要工作维度；②所有指标均基于O*NET，存在同源偏差；③回归为观察性，未证明因果关系；④物理需求与工资等协变量为粗略代理，可能遗漏其他重要职业特征；⑤工作条件的职业平均值忽略了工作场所与个体差异；⑥样本仅包含OEWS覆盖的职业，可能不完全代表全部美国劳动力；⑦使用的曝光度量未涵盖所有AI技术与算法管理；⑧置信区间未覆盖测量误差与项目选择不确定性。

---

## 705. PixelSpline: Continuous Space-Time Video Super-Resolution via Endpoint-Anchored Shared Pixel Trajectories

**arXiv ID:** 2611.06870 | [PDF](https://arxiv.org/pdf/2611.06870v1)

**作者:** Wanqi Yuan `[一作]` (Clemson University), Nianyi Li `[通讯]` (Clemson University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

实现了连续时空视频超分辨率（C-STVSR），结合长上下文编码、连续高分辨率特征提升和端点锚定PixelSpline，能够在任意空间尺度与时间点生成高质量视频；

**💡 创新点**

创新点在于①使用长上下文编码提升端点表征的鲁棒性；②采用共享的PixelSpline轨迹以一致路径链接两端点，解决传统两向独立运动预测导致的误差；③在目标HR网格上直接进行连续特征提升，提前构造细节；

**🔧 技术方法**

使用长上下文双向ConvLSTM+多尺度注意力、连续特征上采样(CUF)、B-spline轨迹建模、双向像素splat、轻量级残差解码器、SEA-RAFT+GMFlow进行光流估计与正则化；

**📊 数据集**

主要在Adobe240（训练/测试）、GoPro、REDS、Vid4（空间/时间边界测试）等公开数据集上训练与评估；

**📈 对比分析**

与V^3、BF-STVSR等连续方法对比，PSNR仅低0.9 dB，速度提升2.24×，在C-STVSR、单独空间超分、单独时间插值等任务中均名列前茅，Temporal consistency tOF也最低；

**⚠️ 局限性**

仍存在PSNR略低于V^3、训练过程耗时长、在极端尺度或复杂动态场景中可能出现纹理失真等局限。

---

## 706. Rethinking Self-Repair in Code-as-Policy Agents: What Drives Retry Success?

**arXiv ID:** 2611.06863 | [PDF](https://arxiv.org/pdf/2611.06863v1)

**作者:** Siddharth Aphale `[一作]` (Stanford University), Ayushman Singh `[通讯]` (Sesame AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在仿真机器人抓取与堆叠任务中，研究了在失败后使用模板化执行反馈是否能提升程序恢复率，并与仅重采样、匹配失败标签、首次尝试的静态指导等方法做对比。

**💡 创新点**

系统地将恢复实验拆分为多种对照：重采样、匹配失败标签、固定任务指令、首次尝试静态指导、代码编辑与反思；证明可复用的任务指导是恢复提升的主要因素，提供了评估自我修复的完整基准。

**🔧 技术方法**

采用冻结的 Qwen2.5-Coder（3B/7B/32B）生成 Python 程序，使用规则化失败标签和模板化反馈；在固定重试次数/总 token 预算下进行匹配实验；统计显著性检验（Wilson 置信区间、配对自助法）评估差异。

**📊 数据集**

主实验使用 252 个初始失败样例（159 种任务/API/场景组合）评估，另外 40 个样例用于开发；随后在 7 个不同任务/模型设置下进行泛化验证，覆盖 354 个失败案例。

**📈 对比分析**

通过匹配重试预算和总 token 预算的对比，使用 conditional recovery@k 作为指标：模板化反馈在 5 次重试内实现 85.3% 的恢复率，重采样仅 68.3%（差异 +17pp）；固定任务模板与不同标签反馈恢复率相近；首次尝试的静态指导将初始成功率从 19.3% 提升至 36.9%，整体成功率与模板相当；编辑/反思方法恢复率低于重采样。

**⚠️ 局限性**

仅在仿真环境下评估，缺少视觉输入和物理后退；失败标签粗粒度，难以捕捉细粒度错误；对比实验常同时改变指令与反馈，难以单独区分两者贡献；模型规模受限，未在更大或更复杂任务、真实物理恢复上验证。

---

## 707. LongBanana: An Expert-Verified Benchmark for Long-Context Multi-Reference Image Synthesis

**arXiv ID:** 2611.06856 | [PDF](https://arxiv.org/pdf/2611.06856v1)

**作者:** Haoxiang Cao `[一作]` (South China Normal University), Kelsey R. Allen `[通讯]` (University Of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Long-Context Multi-Reference Image Synthesis 基准，定义了任务、评测流程与案例

**💡 创新点**

创新点包括：专家设计的 90 例任务与 4,670 条原子检查表，图结构评估（Content、Relations、Global）以及对 agent harness 与 RPR 工作流的系统评测

**🔧 技术方法**

使用技术包括：图结构化任务描述、基于工具的多维检查评估器(-Eval)、ReAct–Plan–Exec–Reflect (RPR) 代理工作流以及多种图像生成后端

**📊 数据集**

数据集：由 90 个专家设计案例组成，包含约 32 张参考图、20 条约束，每例约 32 张参考图，约 46,700 条人工标注检查条目

**📈 对比分析**

比较方法：在 90 个案例下与 10 种生成设置（FP-Direct、PD-Direct、最小 harness、RPR）对比，结果显示 RPR+OpenClaw 在整体和各子类分数上均优于非代理基线，且在任务通过率上有显著提升

**⚠️ 局限性**

局限性：仍依赖人工专家构造，规模有限；评测只覆盖当前图像生成后端，未探究跨模态或更大规模检索；agent harness 仍难以在复杂时序/状态推理中保持一致性

---

## 708. Persona Perspectives: Adopting Viewpoints Reorganizes Representations in LLMs

**arXiv ID:** 2611.06848 | [PDF](https://arxiv.org/pdf/2611.06848v1)

**作者:** Austin C. Kozlowski `[一作]` (University of Chicago), Madison Lee `[通讯]` (University of Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大型语言模型中，使用系统提示来让模型扮演不同政治视角（自由派或保守派）时，如何系统性重构其潜在空间的语义几何结构，并探讨这一重构对监测功能的影响。

**💡 创新点**

创新点在于将人物化视角视为上下文驱动的几何重组，而非仅仅是一个额外的方向，揭示不同视角会导致同一概念在潜在空间中以完全不同的关联方式出现。

**🔧 技术方法**

技术手段包括基于残差流的词向量提取、构建 33 个语义轴、线性探针、稀疏自动编码器（SAE）以及手工选择的词向量分类器；并通过在多层模型中对轴相似度和词投影的统计分析来评估重构效果。

**📊 数据集**

数据集包括：1）约 9,000 个常见英文词汇及其在 5 种模板下的表示；2）33 条语义轴的锚词对；3）来自自由派与保守派文本的词向量；4）两套安全监测数据集——一个关于真伪声明（9,192 条语句），一个关于道德可接受性（8,000 条行动）。

**📈 对比分析**

比较方法是将不同人物化条件下的轴相似度矩阵差异与自由派/保守派文本训练的词向量差异相关联，并检验在不同模型、不同层次以及与手工构造的线性/SAE/词向量分类器上的性能，发现人物化条件能显著改变模型对争议声明的真伪和道德判断。

**⚠️ 局限性**

局限性包括：只研究了两种政治人物化，未涵盖更广泛的视角；监测器在争议主题下的性能下降表明需设计跨视角鲁棒的探针；并且实验依赖特定的 Llama 3.1 及其它模型，结果可能对更大或不同架构的模型泛化不足。

---

## 709. ServerBench: How can your coding agents design specialized LLM serving systems?

**arXiv ID:** 2611.06841 | [PDF](https://arxiv.org/pdf/2611.06841v1)

**作者:** Xinrui Zhong `[一作]` (Carnegie Mellon University), Beidi Chen `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并评估了一个名为 ServerBench 的基准，旨在让编码代理在给定模型、GPU 分配和真实工作负载下，构建并优化完整的 LLM 服务系统。

**💡 创新点**

创新点包括提出以“美元/美元”作为服务价值评估指标、使用状态化工作负载回放和闭环实验设计，以及采用独立验证确保系统满足延迟、成本和正确性要求。

**🔧 技术方法**

所用技术涵盖 Transformer 服务器、SGLang、vLLM、前缀重用、批处理、显存管理、调度优化、推理工具与执行反馈，并在 Harbor 沙盒环境中自动化测评。

**📊 数据集**

使用的数据集为从真实编码代理会话中重放的工作负载，包含多轮、共享前缀、并行子代理的请求长度与延迟，以及公开的 Qwen、Gemma、Claude 等模型 checkpoint。

**📈 对比分析**

实验在三种模型-硬件组合上对 11 种代理进行评估，使用 SGLang 与 vLLM 作为基线；结果显示部分代理（如 Claude Opus 5.5）可将“美元/美元”得分提升至基线的 2.7–4.8 倍，体现显著性能提升。

**⚠️ 局限性**

限制包括仅覆盖三种部署与单一工作负载来源、回放与实际生产流量不完全一致、验证覆盖有限、缺乏对开发成本和不同硬件/任务场景的评估。

---

## 710. Before You Continue: Evaluating Time-Sensitive Safety Awareness in Conversational Assistants

**arXiv ID:** 2611.06839 | [PDF](https://arxiv.org/pdf/2611.06839v1)

**作者:** Peidi Dong `[一作]` (Carnegie Mellon University), Jen-tse Huang `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一个新的基准测试集——Temporal Action Safety in Conversation (TASC)，用于评估大型语言模型在对话中是否能识别并遵守与时间相关的安全规则。

**💡 创新点**

创新点在于：①构建了包含多维度时间约束（最短等待、截止期限、目标窗口）且覆盖23个日常领域的系统化样本；②通过对比“正常请求”“时间查询”“已知时间”三种对话条件，揭示了模型在时间推理与规则执行之间的显著差距；③系统评估了13个主流LLM的表现，显示显著的性能差异和普遍的安全缺陷。

**🔧 技术方法**

技术手段包括：模板化生成对话数据（128模板 → 3,418样本），使用ISO时间戳、当前时间、口语线索或无时间标注四种时间信息格式；对话中加入“先前事件 + 规则 + 后续请求”三段结构；使用人类评审与自动解析器共同为模型输出打标签；对比模型在违规检测、持续时间准确率、显式时间补充等指标。

**📊 数据集**

使用的数据集为TASC benchmark，包括：3,418个输入（128模板扩展），3,418个对应的gold响应类型；另外还有33个独立情景（5–21条消息）用于配对实验；涵盖23个细分领域，风险层级从0到3；所有对话均为英语。

**📈 对比分析**

对比方法：在同一套对话条件下，对13个模型分别计算违规检测率、持续时间回答准确率、显式时间补充后的提升幅度、误报率等。结果显示：整体违规检测率仅为30.5%；最好的模型Claude Opus 5达到84.9%，但大多数模型低于50%；持续时间回答准确率普遍高（92.4%），但仅62.1%（≈70%）的正确时长计算对应到违规检测；显式时间补充提升约8个百分点，未能显著弥补错误。

**⚠️ 局限性**

局限性包括：①样本构造未完全交叉覆盖所有因素，导致某些因素的统计基础薄弱；②仅使用英语对话，缺乏跨语言验证；③对话为人工编写、单轮回复，未模拟真实用户多轮交互和缺少时间戳的部署环境；④缺少对模型在多轮对话中是否能修正错误的评估。

---

## 711. Enabling AWS Cloud FPGA Development Using High-Level Vitis HLS

**arXiv ID:** 2611.06829 | [PDF](https://arxiv.org/pdf/2611.06829v1)

**作者:** David Rostcheck `[一作]` `[通讯]`, David Rostcheck

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了完整的 AWS EC2 F2 FPGA 工具链，将 Vitis HLS 代码通过社区桥接转换为可部署的 AFI，并在 FPGA 上实现 TinyBERT 的嵌入层和第一个编码器层用于地址命名实体识别推理；

**💡 创新点**

首次在 AWS 云 FPGA 上完成 Transformer 推理的端到端工具链和部署流程，提出四层验证方法（FP32、量化 Python、HLS C‑simulation、FPGA 物理验证），并采用 W16A16 量化以便分离电路误差和量化噪声；

**🔧 技术方法**

使用 Vitis HLS、AWS HDK、16 位 INT16 量化、BRAM/URAM/HBM/DDR 存储、Python 量化参考、四层验证框架、Vivado 迭代构建、AXI 处理、FPGA 调试与性能分析；

**📊 数据集**

TinyBERT 在合成地址 NER 数据集（560 训练 / 120 验证 / 500 测试样本，标签为 STREET、CITY、STATE、ZIP、PERSON）上进行微调；

**📈 对比分析**

通过比较 FP32 PyTorch 基准、W8A8 和 W16A16 量化的 Python 参考、HLS C‑simulation 与 FPGA 物理层，验证准确率：W16A16 与 FP32 仅差 0.1% F1；FPGA 在 16 词长输入时的延迟为 344.7 ms，比 CPU（0.72 ms）慢约 480 倍，比 GPU（0.84 ms）慢约 410 倍，主要因权重量化流动到宿主 DRAM 的高延迟；

**⚠️ 局限性**

桥接仅支持单 DDR 通道、单 AXI、单内存主控，缺乏 HBM 访问和多通道 DMA，导致权重量化流动成为瓶颈；仅实现了嵌入层和第一个编码器层，W16A16 量化占用资源多且效率低；Transformer 架构与 FPGA 分布式内存不匹配，整体性能远低于 GPU/CPU。

---

