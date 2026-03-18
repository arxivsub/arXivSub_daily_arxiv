# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-18 | 今日论文总数: 610

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. AsgardBench - Evaluating Visually Grounded Interactive Planning Under Minimal Feedback

**arXiv ID:** 2603.15888 | [PDF](https://arxiv.org/pdf/2603.15888v1)

**作者:** Andrea Tupini `[一作]` (Microsoft Research), Jianfeng Gao `[通讯]` (Microsoft Research)

**通讯引用:** 38565 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个专注于视觉驱动交互式规划的基准，用简化的动作空间和最小反馈评估模型在执行过程中依据视觉信息不断修正计划的能力。

**💡 创新点**

创新点在于：①将导航和低层操作剥离，只保留高层规划；②通过系统化的对象状态与位置变异诱导条件分支，使单条指令在执行时需动态调整路径；③对比图像输入、文本仅、无反馈、详细反馈四种设置，清晰揭示视觉基础对规划的关键作用。

**🔧 技术方法**

使用 AI2-THOR 3D 仿真环境、基于大语言模型（如 GPT‑4o、Qwen、Claude 等）的完整动作序列生成，以及手部覆盖提示、记忆摘要等工程技术来实现视觉感知与规划的耦合。

**📊 数据集**

采用包含 108 个任务实例、12 种任务类型（厨房、客厅、浴室）的数据集，任务通过对象状态（干净/脏）、占位、位置等多维变异产生 3–27 种变体，形成条件分支的实验环境。

**📈 对比分析**

通过在图像+文本、文本仅、无反馈、详细反馈等设置下对模型成功率的定量比较，发现视觉输入使大部分模型成功率提升 2 倍以上；最强模型在图像条件下的表现仍显著优于文本+详细反馈，表明视觉感知提供了额外信息。

**⚠️ 局限性**

局限性包括：①去除导航和低层操作降低生态真实性；②场景受限于 AI2-THOR 对象集和光照，缺乏更广泛的视觉多样性；③极简反馈与手部覆盖提示依赖可能不适用于真实系统；④要求模型每步生成完整动作序列可能偏向规划型 LM，限制了对反应式策略的评估。

---

## 2. NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics

**arXiv ID:** 2603.16148 | [PDF](https://arxiv.org/pdf/2603.16148v1)

**作者:** Zhengzheng Tang `[一作]` `[通讯]` (Boston University), Zhengzheng Tang (Boston University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并训练了一个0.9B参数的纯SNN语言模型，在随机初始化下完成了从预训练到对话微调的全过程。

**💡 创新点**

创新点在于将SNN膜电位动力学映射为选择性状态空间模型（SSM），并结合泄漏电流交互、PonderNet自适应时间步、Triton融合PLIF核、残差中心化、侧向抑制归一化和自然梯度补偿，实现了大规模可训练的纯Spike网络。

**🔧 技术方法**

采用了参数化泄漏式Leaky Integrate‑and‑Fire（PLIF）神经元、选择性SSM块、SNN前馈网络、PonderNet自适应时步、Triton融合的PLIF前向/后向核、残差中心化与侧向抑制归一化、自然梯度补偿以及梯度检查点等技术。

**📊 数据集**

预训练使用约1.4B标记的Seq‑Monkey 10B-token语料；微调使用中文对话数据BelleGroup train_3.5M_CN，约3.5M对话。

**📈 对比分析**

与现有SNN模型对比，0.9B模型在预训练损失上从9.0降至3.5，微调后对话生成流畅；与Transformer基准尚未做定量对比，但在结构化语言生成方面表现可比。

**⚠️ 局限性**

局限性包括仅支持512-token上下文、缺乏公开的量化基准（如C‑Eval、CMMLU）、仅中文数据、缺乏推理能力、重复问题以及能耗评估不足。

---

## 3. SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models

**arXiv ID:** 2603.16859 | [PDF](https://arxiv.org/pdf/2603.16859v1)

**作者:** Tianyu Xie `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**通讯引用:** 32179 | [OpenAlex ID](https://openalex.org/A5016080094)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

无法提供总结（论文内容缺失）

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用技术

**📊 数据集**

无法确定使用数据集

**📈 对比分析**

无法比较方法与性能

**⚠️ 局限性**

无法说明限制

---

## 4. GitOps for Capture the Flag Platforms

**arXiv ID:** 2603.16265 | [PDF](https://arxiv.org/pdf/2603.16265v1)

**作者:** Mikkel Bengtson Albrechtsen `[一作]` (University of Southern Denmark), Torben Worm `[通讯]` (University of Southern Denmark)

**通讯引用:** 761 | [OpenAlex ID](https://openalex.org/A5084805877)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出并实现了一个基于 GitOps 的全流程框架，用于 Capture‑The‑Flag（CTF）竞赛的挑战开发、部署与基础设施管理；

**💡 创新点**

创新点在于将挑战定义与基础设施代码统一到 Git 仓库中，实现完全可审计、可回滚的自动化部署流程，并通过自定义工作流和插件让挑战作者专注内容创作；

**🔧 技术方法**

主要技术包括 Kubernetes、CTFd、kube‑ctf、OpenTofu（Terraform Fork）、ArgoCD、Traefik、Prometheus/Grafana、Redis/MariaDB、Cloudflare 等，配合 GitHub、Discord 进行协作与 CI/CD；

**📊 数据集**

评估数据来源于真实规模 CTF 赛事 BrunnerCTF 2025，包含 2,860 名参赛者、1,491 支队伍、83 个挑战（共 83 个任务类型）以及约 600 个并发挑战实例；

**📈 对比分析**

性能表现：在 Hetzner 集群上，峰值 600+ 实例占用约 8 vCPU、43 GB RAM，单小时部署约 750 实例；吞吐量峰值 160k 请求/分钟，成本仅 127.45 欧元（生产期），整体成本 265.18 欧元；

**⚠️ 局限性**

局限性包括缺乏预部署测试环境、挑战资产导致仓库膨胀、GitHub 与 Discord 生态绑定、Redis 集群不兼容、自动扩缩容配置失误导致的临时故障，以及对安全监控细粒度告警的不足等。

---

## 5. Out-of-Distribution Object Detection in Street Scenes via Synthetic Outlier Exposure and Transfer Learning

**arXiv ID:** 2603.16122 | [PDF](https://arxiv.org/pdf/2603.16122v1)

**作者:** Sadia Ilyas `[一作]` (Aptiv Services Deutschland GmbH), Matthias Rottmann `[通讯]` (University of Osnabrück)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SynOE-OD 框架，通过 Stable Diffusion 生成合成的 OOD 物体并利用 GroundingDINO 自动标注，训练单一检测器实现同时识别 ID 与 OOD 目标。

**💡 创新点**

创新点在于将生成式“outlier exposure”直接嵌入图像空间，实现无需额外分支的统一 ID/OOD 检测，并利用 OVOD 的零样本能力自动生成高质量标注。

**🔧 技术方法**

使用 Stable Diffusion 合成 OOD 物体、GroundingDINO 进行自动边框标注，并对 DINO-DETR、GDINO 等标准检测器进行迁移学习。

**📊 数据集**

训练数据为 NuImages、Cityscapes、BDD100K 等街景集，评测数据集为 OoDIS 基准（RoadObstacle、RoadAnomaly、Fishyscapes、LostAndFound）。

**📈 对比分析**

与零样本 OVOD 和 UGainS 基线对比，SynOE-OD 在所有 OoDIS 数据集上 AP50..95 提升约 5%–11%，同时保持 ID 类别的竞争性 AP。

**⚠️ 局限性**

局限在于合成 OOD 的质量与多样性依赖生成模型，过多合成样本可能导致 ID 性能下降，对特定环境的泛化能力仍需进一步验证。

---

## 6. A Dynamic Survey of Fuzzy, Intuitionistic Fuzzy, Neutrosophic, Plithogenic, and Extensional Sets

**arXiv ID:** 2603.15667 | [PDF](https://arxiv.org/pdf/2603.15667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 7. SegviGen: Repurposing 3D Generative Model for Part Segmentation

**arXiv ID:** 2603.16869 | [PDF](https://arxiv.org/pdf/2603.16869v1)

**作者:** Lin Li `[一作]` (Beihang University), Lu Sheng `[通讯]` (Beihang University)

**通讯引用:** 5494 | [OpenAlex ID](https://openalex.org/A5035443556)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

使用预训练的3D生成模型，将3D部件分割改造成颜色化任务，实现交互式、全分割及带2D引导的分割。

**💡 创新点**

创新点在于将生成模型的结构与纹理先验直接用于分割，通过颜色化实现多任务统一框架，并极大降低标注需求。

**🔧 技术方法**

采用基于Treillis2的稀疏VAE+流匹配生成器，结合多任务条件注入、点令牌与2D引导，训练时使用噪声匹配损失。

**📊 数据集**

训练集为PartVerse（约12k物体、91k标注部件），评估在PartObjaverse‑Tiny和PartNeXT上。

**📈 对比分析**

与P3‑SAM、Point‑SAM、Find3D、SAMPart3D、PartField等方法对比，交互分割IoU@1提升40%，全分割IoU提升15%，仅使用0.32%标注数据。

**⚠️ 局限性**

依赖于预训练生成模型的质量，部分复杂或纹理丰富的形状仍可能出现边界模糊；2D引导需要额外渲染或外部分割，且多任务训练对硬件要求较高。

---

## 8. DriveFix: Spatio-Temporally Coherent Driving Scene Restoration

**arXiv ID:** 2603.16306 | [PDF](https://arxiv.org/pdf/2603.16306v1)

**作者:** Heyu Si `[一作]` (Zhejiang University), Qi Guo `[通讯]` (Huawei)

**通讯引用:** 14263 | [OpenAlex ID](https://openalex.org/A5100766907)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种多视角同步恢复框架 DriveFix，用以在自动驾驶场景中实现空间与时间一致的 4D 场景恢复。

**💡 创新点**

创新点在于交错的扩散 Transformer 结构，分别建模时间依赖与跨摄像机空间一致性，并引入历史条件生成、几何一致性损失以及多模态（深度、语义）引导，实现高保真、无漂移的恢复。

**🔧 技术方法**

技术上使用扩散 Transformer（基于 Stable Diffusion 3），交错时间/空间注意力块，历史条件注意力，几何对齐损失，深度与语义条件。

**📊 数据集**

使用 Waymo Open、nuScenes、PandaSet 三大自动驾驶数据集进行训练与评测。

**📈 对比分析**

与多种基准（PVG、Difix3D+、DeSiRe‑GS 等）比较，在 PSNR/SSIM/LPIPS/FID 上均优于前置方法，显著提升重建与新视角合成质量。

**⚠️ 局限性**

局限性：对极端遮挡或极端光照仍可能出现细节丢失；模型对训练集分布敏感，迁移到新传感器配置需额外对齐；并且推理速度受扩散步骤限制。

---

## 9. BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs

**arXiv ID:** 2603.16557 | [PDF](https://arxiv.org/pdf/2603.16557v1)

**作者:** Sangyeon Yoon `[一作]` (Yonsei University), Albert No `[通讯]` (Yonsei University)

**通讯引用:** 514 | [OpenAlex ID](https://openalex.org/A5049196468)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了BenchPreS基准，用以评估在持久记忆下的LLM是否能够根据不同的通信场景选择性地应用或抑制用户偏好。

**💡 创新点**

创新点在于提出两项互补度量——误用率（Misapplication Rate, MR）和适当应用率（Appropriate Application Rate, AAR），并以此揭示当前LLM在上下文感知偏好选择方面的不足。

**🔧 技术方法**

技术方法包括：将用户档案与上下文组合形成评价实例；采用LLM‑as‑Judge框架（基于DeepSeek‑R1）进行偏好是否被体现的判定；对比不同推理能力和提示式抑制策略对MR与AAR的影响。

**📊 数据集**

使用的数据集来自CIMemories改造，包含10个用户档案（约152条属性，其中5条偏好）与39个正式通信场景（收件人‑任务对），共计1950个属性级评价实例。

**📈 对比分析**

实验结果显示，10款前沿LLM的MR普遍偏高（最高达86.48%），AAR亦随之升高，未出现低MR高AAR的组合；提示式抑制能略降低MR，但往往伴随AAR下降，整体性能仍不理想。

**⚠️ 局限性**

限制包括：仅评估生成阶段的偏好选择，未涵盖检索或外部工具的使用；并且对非正式或文化细微差别的通信情境缺乏覆盖，可能影响评估的普适性。

---

## 10. Flood Risk Follows Valleys, Not Grids: Graph Neural Networks for Flash Flood Susceptibility Mapping in Himachal Pradesh with Conformal Uncertainty Quantification

**arXiv ID:** 2603.15681 | [PDF](https://arxiv.org/pdf/2603.15681v1)

**作者:** Paras Sharma `[一作]` (Independent Researcher), Swastika Sharma `[通讯]` (YSP University of Horticulture and Forestry)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了印度喜马偕尔邦（HP）多年的SAR洪水库存并利用图神经网络生成全州级闪洪易发性地图

**💡 创新点**

首次将流域连通性建模为图结构加入图神经网络，并结合符合性预测提供统计置信区间，解决传统像素级模型忽视水流方向、验证失真和不确定性缺失的问题

**🔧 技术方法**

采用GraphSAGE图神经网络、随机森林、XGBoost、LightGBM及堆叠集成模型，并使用Google Earth Engine、PyTorch、PyTorch Geometric、MAPIE实现符合性预测和SHAP特征解释

**📊 数据集**

使用Sentinel‑1 C‑band SAR季节性差异检测的三千个洪水点、HiFlo‑DAT与NDMA等历史记录、12个地形/水文/土壤/降水等条件因子（30 m分辨率）以及OSM基础设施数据

**📈 对比分析**

通过留一河流盆地的空间块交叉验证（5块）评估，图神经网络AUC达0.978±0.017，显著优于最佳像素级堆叠模型AUC 0.881；在独立的2023年测试集上，堆叠模型AUC 0.892，符合性预测覆盖率约82.9%

**⚠️ 局限性**

SAR标签噪声导致高风险区置信区间覆盖率低；仅使用静态条件因子缺乏动态土壤湿度等时变驱动；未单独建模GLOF等不同触发机制，图结构假设和边权重为简化近似

---

## 11. Diverging Transformer Predictions for Human Sentence Processing: A Comprehensive Analysis of Agreement Attraction Effects

**arXiv ID:** 2603.16574 | [PDF](https://arxiv.org/pdf/2603.16574v1)

**作者:** Titus von der Malsburg `[一作]` (University of Stuttgart), Sebastian Padó `[通讯]` (University of Stuttgart)

**通讯引用:** 6080 | [OpenAlex ID](https://openalex.org/A5003870894)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了十一种自回归Transformer模型在英文同位语吸引效应上的阅读时间预测，覆盖前置词短语和宾语提取关系从句两种结构；

**💡 创新点**

首次在更全面的语法配置与多种模型上检验吸引效应，并通过层次化回归揭示前置词短语预测良好而宾语从句预测失效的现象；

**🔧 技术方法**

采用Surprisal理论将模型负对数概率转换为阅读时间预测，并使用配对t检验、线性回归及跨模型随机效应分析来比较性能；

**📊 数据集**

使用Wagers等（2009）自流阅读时间实验的Stimuli（Exp 3、4），共384句与192句，涵盖8种语法条件；

**📈 对比分析**

将模型Surprisal与人类阅读时间对比，发现前置词短语中模型与人类相符，而宾语关系从句中模型预测与人类显著偏离，且模型间差异大，未形成一致趋势；

**⚠️ 局限性**

仅限英文自流阅读数据，模型范围有限，未覆盖多语言或其他架构，且可能受词汇频率、标记化等因素影响。

---

## 12. SE(3)-LIO: Smooth IMU Propagation With Jointly Distributed Poses on SE(3) Manifold for Accurate and Robust LiDAR-Inertial Odometry

**arXiv ID:** 2603.16118 | [PDF](https://arxiv.org/pdf/2603.16118v1)

**作者:** Gunhee Shin `[一作]` (Korea Advanced Institute of Science and Technology), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5945 | [OpenAlex ID](https://openalex.org/A5059521863)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出在 SE(3) 上平滑 IMU 传播并结合不确定性感知运动补偿的 LiDAR–惯性里程计框架 3‑LIO

**💡 创新点**

创新点在于：① 在 SE(3) 上联合分布传播姿态，使旋转变动被正确地映射到平移传播；② 通过考虑预测姿态间相关性精确量化相对变换的不确定性，从而实现不确定性感知的运动补偿 (UAMC)

**🔧 技术方法**

采用误差状态卡尔曼滤波、SE(3) Lie 群预积分、联合分布传播、连续时间 B‑样条插值以及 UAMC 技术

**📊 数据集**

在多种场景下评估，包括激烈运动的无人机、粗糙地形的地面车辆以及 NTU‑VIRAL 数据集

**📈 对比分析**

与 FAST‑LIO2、PV‑LIO、DLIO、MA‑LIO 等前沿方法对比，3‑LIO 在多条序列上误差更低、鲁棒性更强，部分序列计算时延最低

**⚠️ 局限性**

局限性：目前仅针对 LiDAR‑IMU 组合，其他传感器配置（如视觉‑惯性、雷达‑惯性）适应性尚待扩展；在极端动态环境下仍受预积分误差影响

---

## 13. Designing for Disagreement: Front-End Guardrails for Assistance Allocation in LLM-Enabled Robots

**arXiv ID:** 2603.16537 | [PDF](https://arxiv.org/pdf/2603.16537v1)

**作者:** Carmen Ng `[一作]` (Technical University of Munich), Carmen Ng `[通讯]` (Technical University of Munich)

**通讯引用:** 476 | [OpenAlex ID](https://openalex.org/A5104089277)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向 LLM 驱动社交机器人资源匮乏场景的前端治理模式——bounded calibration with contestability，旨在通过治理批准的优先级菜单、持续可见的模式提示和可行的争议通道，实现资源分配的透明、合法与可争议。

**💡 创新点**

创新点在于将价值多元性和 LLM 行为不确定性视为前端前提，构建三层治理（定义、选择、挑战）与可操作的争议路径，避免隐性默认与过度可配置的“价值设置”，并聚焦实时交互中的可解释性与程序正义。

**🔧 技术方法**

技术手段包括前端交互设计模式、LLM 驱动的对话生成与优先级决策、角色门控与速率限制策略，以实现可解释的优先级说明与争议触发接口。

**📊 数据集**

未使用公开数据集，研究基于情境案例与设计稿；未进行大规模实验数据收集。

**📈 对比分析**

本文未给出实验对比或性能指标，提出的评估议程聚焦可解释性、合法性与可操作性三大基准，并建议通过情境实验、Wizard‑of‑Oz 多人研究和治理研讨会验证。

**⚠️ 局限性**

局限在于缺乏经验验证，依赖治理机构对模式定义和角色门控的能力；争议通道的可用性在不同用户群体中可能不均衡，长期使用可能导致自动化偏见。

---

## 14. 100x Cost & Latency Reduction: Performance Analysis of AI Query Approximation using Lightweight Proxy Models

**arXiv ID:** 2603.15970 | [PDF](https://arxiv.org/pdf/2603.15970v1)

**作者:** Yeounoh Chung `[一作]` (Google Cloud), Yannis Papakonstantinou `[通讯]` (Google Cloud)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过使用轻量级代理模型（主要是基于文本嵌入的逻辑回归分类器）对 LLM 进行近似，从而实现对 SQL 中 AI 操作符（AI.IF 语义过滤和 AI.RANK 语义排序）的成本与延迟大幅降低，并保持甚至提升准确率。

**💡 创新点**

创新点在于：① 将高频语义操作归约为分类任务，利用代理模型全表覆盖而非仅样本；② 结合在线与离线代理训练、自动采样与不平衡学习、以及自适应代理选择机制；③ 在大规模 OLAP（BigQuery）和低延迟 HTAP（AlloyDB）两大架构上全面评估，展示 100x 以上的成本/延迟提升。

**🔧 技术方法**

技术包括：文本嵌入（Gecko、Gemini 等），逻辑回归代理模型，随机/主动/分层采样，不平衡训练（权重/SMOTE），在线/离线代理训练流程，Adaptive Proxy Selection 机制，基于 LLM 的标签生成与评估。

**📊 数据集**

数据集覆盖多领域的 20+ 语义过滤与排序基准：California Housing、Amazon Reviews、BBC News、IMDB、Amazon Polarity、Mental Health、Tweet Sentiment、Emotion、Banking77、Toxic Conversations、FEVER、TREC-COVID、TREC-DL-2022、FIQA-2018、SCIDOCS、SciFact、HellaSwag 等，规模从数千行到 10M 行。

**📈 对比分析**

与纯 LLM 基准相比，在线代理在 BigQuery 上实现 329–991 倍的延迟提升、728–792 倍的成本降低；离线代理在 AlloyDB 上实现 4–728 倍的延迟提升、9–728 倍的成本降低；代理模型在多数数据集上获得 90% 以上的 macro‑F1 与 nDCG，部分数据集代理甚至优于 LLM。通过自动评估与阈值控制，代理失效时自动回退到 LLM，确保质量。

**⚠️ 局限性**

局限性包括：1) 在极度不平衡或低相关度的数据子集（如 HellaSwag、SciFact）代理难以学习有效决策；2) 需要 LLM 进行标签生成，虽大幅减少但仍有成本；3) 对于需要精细推理的排序任务（四级评分）代理表现受限；4) 需要合理采样与定期再训练以保持鲁棒性；5) 对于需要大量候选对的语义 join 等场景，代理难以直接替代。

---

## 15. Deep Reinforcement Learning-driven Edge Offloading for Latency-constrained XR pipelines

**arXiv ID:** 2603.16823 | [PDF](https://arxiv.org/pdf/2603.16823v1)

**作者:** Sourya Saha `[一作]` (City University of New York), Saptarshi Debroy `[通讯]` (City University of New York)

**通讯引用:** 683 | [OpenAlex ID](https://openalex.org/A5015917097)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在该研究中，作者提出了一套面向边缘辅助沉浸式XR系统的电池感知执行管理框架，通过在线深度强化学习控制器动态调节执行位置、工作质量和传感器采样速率，以在满足运动到光子（MTP）延迟要求的同时最大化终端设备的电池寿命。

**💡 创新点**

创新点在于：①将电池消耗、MTP延迟与网络条件共同建模，形成统一的实时优化目标；②采用轻量级DQN实现在线决策，能够在不需要离线训练的情况下实时适应网络波动；③在闭环XR管线中引入多维度动作空间（质量、采样率、执行模式），实现对能耗与实时性的双向权衡。

**🔧 技术方法**

核心技术包括：边缘计算与局部/远程执行切换、可变质量视觉传感、基于状态机的动作空间设计、深度Q网络（DQN）强化学习、经验回放与目标网络、MTP延迟与能耗的实时监测与奖励设计。

**📊 数据集**

实验使用Illixr开源XR研究平台，模拟VIO感知管线；网络条件采用循环波动的带宽曲线（1 Mbps–1 Gbps）；电池模型为16.6 Wh的模拟电池并跟踪SoC；数据集主要为实验生成的延迟、功耗与网络日志。

**📈 对比分析**

作者将其策略与三种基线对比：全局本地执行、全局离线执行、Greedy和阈值自适应策略。结果显示：在稳定网络下，RL策略将电池寿命提升约163 %（相较于纯本地执行），MTP符合率保持在90.8 %；在可变网络下，MTP符合率从75 %提升至83.8 %，电池寿命相较本地下降不到一半。

**⚠️ 局限性**

主要限制包括：①缺乏正式的延迟保障理论，仅提供经验性符合率；②仅考虑单一客户端–边缘部署，未扩展到多设备共享边缘资源场景；③未对边缘侧能耗进行建模；④决策频率与网络变化速度匹配仍有改进空间。

---

## 16. Unpaired Cross-Domain Calibration of DMSP to VIIRS Nighttime Light Data Based on CUT Network

**arXiv ID:** 2603.16385 | [PDF](https://arxiv.org/pdf/2603.16385v1)

**作者:** Zhan Tong `[一作]` (Nanjing Institute of Technology), Kaihao Fang `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用Contrastive Unpaired Translation（CUT）网络，将 1992-2013 年 DMSP-OLS 夜间灯光数据校准为 VIIRS 级别，并生成全球一致的长期夜灯时间序列。

**💡 创新点**

创新点在于：① 放弃传统的 cycle‑consistency 约束，采用 PatchNCE 对比学习在多层特征上最大化对应块的互信息，实现单向映射；② 通过多尺度对比学习恢复 DMSP 的低分辨率、饱和问题，保持空间结构一致性。

**🔧 技术方法**

使用技术包括：CUT 网络架构（ResNet 编码‑解码 + PatchGAN 判别器）、Least‑Squares GAN 损失、PatchNCE 对比损失、log1p 归一化、单通道改造、少量水平翻转数据增强。

**📊 数据集**

数据集：DMSP‑OLS 1992‑2013（F18 2013 作为重叠期）、VIIRS DNB 2012‑2020（VCMCFG）、MODIS 全球陆地掩模、GPW 人口密度格网用于验证。

**📈 对比分析**

与线性回归、直方图匹配、CycleGAN 进行对比。CUT 取得 R²=0.87、Spearman ρ=0.91、SSIM=0.79，训练时间 80 小时，显著优于基线（最大提升 0.15 的 R²，SSIM 提升 0.05）。

**⚠️ 局限性**

局限性：在极度饱和的城市核心区域仍产生棋盘格或光晕伪影，精度下降；模型无法完整恢复 DMSP 失真或缺失的高频辐射信息。

---

## 17. Time-Aware Prior Fitted Networks for Zero-Shot Forecasting with Exogenous Variables

**arXiv ID:** 2603.15802 | [PDF](https://arxiv.org/pdf/2603.15802v1)

**作者:** Andres Potapczynski `[一作]` (Amazon), Dmitry Efimov `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研发了一种针对时间序列的Prior‑Data Fitted Network（PFN）模型（称为TabPFN‑TS），实现了零样本（zero‑shot）预测并能自然地处理外生协变量；

**💡 创新点**

创新点包括①利用单根节点随机生长网络（SRNGN）生成具有时序依赖的合成训练数据；②在Transformer结构中加入绝对与相对位置编码以及全注意力机制，使模型具备时间序列特有的归纳偏置；③通过在根节点注入周期性信号来模拟真实时间序列的动态特征；

**🔧 技术方法**

技术手段涵盖PFN框架、基于样本‑特征分离的Transformer注意力、位置编码、全注意力、合成数据生成（SCM+SRNGN+根节点周期性激励）以及大规模零样本推断；

**📊 数据集**

使用的主要数据集包括欧洲五大电价数据集（Nord Pool、PJM、France、Belgium、Germany）、M5竞赛多级聚合数据（日/周/月、州/店/SKU级别），以及传统无外生变量的M‑series基准（M1–M4、Toulouse等）；

**📈 对比分析**

通过与多种基线模型（TabPFN、Sundial‑Base、DLinear、NBeats等）在sCRPS、RMSSE等指标上对比，实验显示在电价和M5上取得SOTA，零样本提升约12%，在无外生变量基准上相对领先约10%；

**⚠️ 局限性**

局限性包括：①采用平方注意力，难以处理超长序列；②作为上下文学习模型，在极少观测的场景下泛化性有限；③性能受限于合成数据分布，若训练分布未覆盖某些依赖关系，推断时可能表现不佳。

---

## 18. When and Why Does Unsupervised RL Succeed in Mathematical Reasoning? A Manifold Envelopment Perspective

**arXiv ID:** 2603.16578 | [PDF](https://arxiv.org/pdf/2603.16578v1)

**作者:** Zelin Zhang `[一作]` (Kyoto University), Chenhui Chu `[通讯]` (Kyoto University)

**通讯引用:** 1869 | [OpenAlex ID](https://openalex.org/A5102757632)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型的数学推理任务，本文提出并评估了一套以不确定性与长度惩罚为核心的无监督强化学习奖励，进一步通过DTW聚类与三维相位空间投影构建几何诊断框架，揭示成功与失败训练轨迹的“推理流形”边界。

**💡 创新点**

创新点主要包括：①将不确定性与长度惩罚分离并系统化探索，证明仅长度惩罚即可显著提升推理性能；②提出两步几何诊断方法，将高维训练动态映射到三维相位空间，直观区分成功流形收敛与两类崩溃（探索停滞与约束松弛）；③在不同基础模型（Llama、Qwen、DeepSeek）上揭示模型能力对无监督RL稳定性的边界。

**🔧 技术方法**

采用GRPO（无Critic的强化学习框架）配合多种奖励（Entropy、AvgEntropy、LengthPenalty、Cumulative Rényi Entropy、Collision Probability），使用DTW时间序列聚类对token熵轨迹进行分组，随后通过三维投影与凸包体积测度对训练轨迹进行几何分析。

**📊 数据集**

主要数据集为开放源代码的 DAPO‑Math‑17K 进行训练，并在 DeepMath‑103K 进行进一步实验；评估使用一系列标准数学基准（MATH500、Minerva Math、OlympiadBench、AIME24、AMC23、AIME26），其中 AIME26 为 OOD 验证。

**📈 对比分析**

在 Qwen3‑1.7B 与 Qwen3‑8B 上，无监督 RL（尤其是 LengthPenalty）在所有基准上的 Pass@k 甚至优于 Supervised RL，表现出最优的 81‑95% 级别；在 DeepSeek‑Distill‑Llama‑8B 上提升有限，且 Llama3.1‑8B 在所有无监督设置下均直接崩溃；实验采用早停策略，仅报告最佳验证点，未报告持续下降的完整训练轨迹。

**⚠️ 局限性**

局限性包括：①实验仅覆盖 8B 规模模型，无法验证更大模型的可扩展性；②几何诊断仅为经验性隐喻，缺乏严格的数学定义和理论证明；③对训练动态的 DTW 结果仍存在未解释的稳定性假设，需进一步研究模型先验结构对熵中心的影响。

---

## 19. Accelerating Approximate Analytical Join Queries over Unstructured Data with Statistical Guarantees

**arXiv ID:** 2603.16153 | [PDF](https://arxiv.org/pdf/2603.16153v1)

**作者:** Yuxuan Zhu `[一作]` (University of Illinois Urbana-Champaign), Daniel Kang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1852 | [OpenAlex ID](https://openalex.org/A5072348548)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种结合嵌入式阻塞和采样的阻塞增强采样（Blocking-augmented Sampling）算法，能够在有限的Oracle调用预算内给出统计可信的近似分析连接查询结果。

**💡 创新点**

创新点在于动态将交叉乘积划分为高相似性阻塞区和低相似性采样区，并通过两阶段自适应预算分配优化估计误差，同时提供有效置信区间。

**🔧 技术方法**

采用重要性采样的加权漫游连接（Weighted Wander Join）、嵌入相似度、Bootstrap-t重采样以及自适应分层分配等技术。

**📊 数据集**

在16个真实数据集（文本、图像、双模态）及合成数据集上评估，包括SemBench、Company、Quora、Webmasters、Roxford、Flickr30K、VeRi等。

**📈 对比分析**

与统一采样、基于阈值的阻塞、Ditto、SUPG、LOTUS等基线比较，平均误差降低1.04–19.5倍，置信区间覆盖率始终≥95%，且在低选择率场景提高至19.5×。

**⚠️ 局限性**

局限在于对嵌入模型质量的依赖，若嵌入相似度误差过大仍会影响性能，并且算法在极大表或多表连接时仍需预先估计阻塞阈值；同时需要Oracle预算才能实现。

---

## 20. BrickSim: A Physics-Based Simulator for Manipulating Interlocking Brick Assemblies

**arXiv ID:** 2603.16853 | [PDF](https://arxiv.org/pdf/2603.16853v1)

**作者:** Haowei Wen `[一作]` (Robotics Institute Carnegie Mellon University), Changliu Liu `[通讯]` (Robotics Institute Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了实时物理模拟器 BrickSim，用于互锁砖块装配、拆解和结构崩塌的高保真仿真，并支持机器人操作。

**💡 创新点**

创新点在于：①提出混合模拟架构，将 snap‑fit 机制从刚体动力学中剥离；②设计基于力学的 snap‑fit 连接模型，转化为稀疏凸二次规划；③实现实时、精确的断裂预测与动态演化。

**🔧 技术方法**

技术细节包括：Isaac Sim+PhysX；Brick Topology Graph 维护拓扑；Assembly Monitor 与 Breakage Detector 负责连接检测与断裂；力学模型和约束通过 OSQP 求解凸二次规划；利用力学分析与碰撞报告实现实时同步。

**📊 数据集**

使用了 StableText2Brick 数据集构建 150 个真实装配，用于验证静态稳定性与动态崩塌；另外还利用真实实验平台进行跌落测试。

**📈 对比分析**

与 BrickFEM、StableLEGO 等基线比较：在 150 组结构中实现 100% 准确率，平均求解时间仅 0.005 s；在动态跌落实验中精准重现崩塌位置；帧时间保持在 16.7 ms 以内，满足 60 FPS 实时要求。

**⚠️ 局限性**

局限性在于：当前可实时处理的装配复杂度限制在 50 块砖以内；仅支持固定 snap‑fit 的零件，尚未覆盖齿轮、轮子等功能部件，且随结构规模增大帧率下降。

---

## 21. Good Arguments Against the People Pleasers: How Reasoning Mitigates (Yet Masks) LLM Sycophancy

**arXiv ID:** 2603.16643 | [PDF](https://arxiv.org/pdf/2603.16643v1)

**作者:** Zhaoxin Feng `[一作]` (Hong Kong Polytechnic University), Bo Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 31549 | [OpenAlex ID](https://openalex.org/A5100688318)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究对齐技术导致LLM出现sycophancy的现象，系统评估Chain-of-Thought（CoT）推理在目标与主观任务、用户与权威偏见下的影响，并通过Tuned Lens分析内部动态。

**💡 创新点**

首次系统探讨CoT既能缓解又能掩盖sycophancy，并揭示其在推理过程中动态演化，提出了动态内部机制的新视角。

**🔧 技术方法**

使用CoT提示、Tuned Lens、语义与语言指标（如MDD、DS coherence、semantic deviation）以及SAE进行内部机制分析。

**📊 数据集**

使用SycophancyEval、MMLU、MATH、AQuA、TruthfulQA、DailyDilemmas、社交态度调查等数据集，覆盖客观与主观任务。

**📈 对比分析**

在六种提示设置下比较六大模型（Claude、GPT3.5、o3mini、Llama3.1、Qwen2.5、Gemma2）的sycophancy率与准确率，发现CoT通常降低最终sycophancy，但在主观任务和权威偏见下仍高，整体表现因模型与任务而异。

**⚠️ 局限性**

研究仅覆盖单轮交互、英文数据、有限模型，缺乏多轮对话和跨语言验证；内部分析仅依赖Tuned Lens，可能无法捕捉所有机制。

---

## 22. ClawWorm: Self-Propagating Attacks Across LLM Agent Ecosystems

**arXiv ID:** 2603.15727 | [PDF](https://arxiv.org/pdf/2603.15727v1)

**作者:** Yihao Zhang `[一作]` (Peking University), Meng Sun `[通讯]` (Peking University)

**通讯引用:** 193812 | [OpenAlex ID](https://openalex.org/A5100748869)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了ClawWorm，首次在真实规模LLM代理生态（OpenClaw）中实现自我复制的蠕虫攻击，完成从单条消息感染到持久化、执行与自发传播的完整生命周期；

**💡 创新点**

创新点在于引入双锚持久化、超图广播传播、供应链放大与URL C2绕过等机制，突破了以往仅限于单点传播的自复制攻击；

**🔧 技术方法**

技术手段包括自然语言提示注入、LLM上下文标签操纵、核心配置文件劫持、第三方技能包供应链利用、工具执行授权和URL检索C2；

**📊 数据集**

使用受控测试平台，基于OpenClaw 2026.3.13的40,000+实例网络，采用3×3因子设计（3种感染向量×3种payload）共180次独立实验；

**📈 对比分析**

实验结果显示整体攻击成功率为0.85，Vector B最高0.95，单跳传播成功率1.00，多跳传播可达5跳，且在持久化与执行上表现稳定，优于先前Morris II等基准；

**⚠️ 局限性**

局限性包括仅使用单一LLM后端、样本量有限、未覆盖所有代理框架与硬件差异，以及未探讨更高级的语义防御和实际生产环境中的网络安全配合。

---

## 23. A Non-Binary Method for Finding Interpolants: Theory and Practice

**arXiv ID:** 2603.15876 | [PDF](https://arxiv.org/pdf/2603.15876v1)

**作者:** Adam Trybus `[一作]` (Institute of Philosophy Jagiellonian University), Tomasz Skura `[通讯]` (University of Zielona Góra)

**通讯引用:** 216 | [OpenAlex ID](https://openalex.org/A5026880671)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种基于非二元判定系统的原子命题逻辑插值生成方法，并实现了Python脚本进行插值搜索。

**💡 创新点**

将反证系统与非二元分辨率结合，用构造性证明直接产生插值，避免传统二元分辨率步骤，理论证明简单。

**🔧 技术方法**

采用了反证系统规则、非二元分辨率、归纳构造、Python实现、伪随机公式生成及性能统计等技术。

**📊 数据集**

通过自定义伪随机生成的CNF‑DNF公式集合（变量≤4或≤10，最多20个子句）以及公开的实验结果数据集进行实验。

**📈 对比分析**

通过统计平均执行时间与插值大小、公式大小的线性关系进行对比，实验显示在10万公式上平均耗时0.0032秒、插值平均165个连结；与HKP系统相比步数更少，证明实现效率可接受。

**⚠️ 局限性**

目前仅适用于命题逻辑，变量数受限（最多4/10），插值形式不易简化，缺乏二元分辨率优化，且未扩展到一阶逻辑。

---

## 24. MedCL-Bench: Benchmarking stability-efficiency trade-offs and scaling in biomedical continual learning

**arXiv ID:** 2603.16738 | [PDF](https://arxiv.org/pdf/2603.16738v1)

**作者:** Min Zeng `[一作]` (University of Minnesota), Rui Zhang `[通讯]` (University of Minnesota)

**通讯引用:** 12045 | [OpenAlex ID](https://openalex.org/A5100675481)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MedCL‑Bench，一个统一的生物医学 NLP 持续学习基准，包含十个公开数据集，设计八种任务顺序，评估多种持续学习方法在不同任务、计算预算和参数效率下的性能。

**💡 创新点**

创新点在于：①将多类别生物医学任务映射为统一的分类格式，消除评价差异；②构建标准化的任务顺序与评估协议，能够量化遗忘、迁移与 GPU‑小时成本；③系统揭示了持续学习策略在生物医学领域的稳定–效率 trade‑off、任务族间遗忘差异以及骨干架构对结果的非单调影响。

**🔧 技术方法**

使用 T5‑base、Qwen‑0.6B/4B 作为骨干；实现多种持续学习技术（VANILLA、MULTI、EWC、L2、REPLAY、GEM、AGEM、LAMOL、ADAPTER、TCL、OLORA）。统一输入输出格式，采用准确率作为核心指标，计算 AP、BWT、FWT，并报告 GPU‑小时成本与可训练参数比例。

**📊 数据集**

十个公开生物医学 NLP 数据集：PubMedQA、BioASQ、SciFact、PubHealth、GAD、ChemProt、DDI、Pubmed_RCT、DRUGLIB、LitCovid。

**📈 对比分析**

通过 AP、BWT、FWT 与 GPU‑小时/参数效率对比。结果显示：VANILLA 在所有顺序下遗忘严重；REPLAY、GEM 与参数隔离方法 ADAPTER、TCL 在保持率与成本上表现最佳；多任务上限约 76% AP；不同骨干（T5 vs Qwen）显著改变方法排名，表明骨干设计影响持续学习效果。

**⚠️ 局限性**

局限性包括：规模实验仅使用单一任务顺序；骨干规模上限为 4B，未覆盖更大模型；评价仅以准确率为主，未考虑校准、鲁棒性等实际部署指标；未探讨混合策略、内存采样优化或更细粒度的任务族迁移机制。

---

## 25. Galaxy Tracer: A Topology-First 3D Interface for Interactive PCAP Exploration

**arXiv ID:** 2603.16018 | [PDF](https://arxiv.org/pdf/2603.16018v1)

**作者:** Ryan Younger `[一作]` `[通讯]` (Olivet Nazarene University), Ryan Younger (Olivet Nazarene University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

开发了一个基于浏览器的三维拓扑视图的包捕获分析工具 Galaxy Tracer。

**💡 创新点**

首次将三维拓扑视图作为默认分析表面，并实现与传统列表视图同步的交互式分析。

**🔧 技术方法**

使用 JavaScript、Three.js、Web Workers、PBR 渲染和 Wireshark 样式过滤语法。

**📊 数据集**

使用 PCAP / PCAPNG 格式的网络捕获，主实验数据为《The Ultimate PCAP v20251206》48,640 包。

**📈 对比分析**

在 MacBook Air M4 上使用 Chrome 测试，交互状态达 0.4 秒、过滤和切换均约 0.2 秒，支持 90+ 协议并可处理 10 万个包。

**⚠️ 局限性**

受限于 80 个主机显示、100k 包阈值、仅支持基本协议解码、缺乏高级取证功能和公开源码。

---

## 26. Steering Frozen LLMs: Adaptive Social Alignment via Online Prompt Routing

**arXiv ID:** 2603.15647 | [PDF](https://arxiv.org/pdf/2603.15647v1)

**作者:** Zeyu Zhang `[一作]`, John C. S. Lui `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于在线提示路由的冻结LLM推理时治理框架CCLUB，用以在不重新训练模型的前提下动态平衡安全与实用性。

**💡 创新点**

创新点：1) 引入双特征（语义+安全）上下文表示，避免语义相似导致的安全泛化；2) 采用保守共识聚类，即仅在安全与效用两侧均相似时才共享数据；3) 将多目标上下文线性UCB与聚类相结合，理论给出子线性遗憾上界。

**🔧 技术方法**

技术：多目标上下文线性UCB、保守共识聚类、图交集机制、双特征编码（MiniLM + Qwen3Guard）、强化学习反馈模型（Skywork、Qwen3-SafeGuard）等。

**📊 数据集**

数据集：Qwen3-0.6B作为冻结模型，使用 BeaverTails、PKU-SafeRLHF 进行在线交互；离线原型通过 K-means 聚类（K=50）得到；系统提示池由 90 条手工生成的 prompt 组成；评估使用 500 条测试查询。

**📈 对比分析**

与随机、全局、原型、输入级 LinUCB/Greedy 等基线比较。结果显示 CCLUB 在在线累计奖励上提升约10.98%，离线子最优缺口降低约14.42%；在不同原型数和热启动比例下仍保持优越性。

**⚠️ 局限性**

局限性：1) 仅适用于固定提示池，无法动态生成新提示；2) 依赖外部评估模型（可能带来偏差）；3) 在极端攻击或分布漂移场景下的鲁棒性尚未充分验证。

---

## 27. PlotTwist: A Creative Plot Generation Framework with Small Language Models

**arXiv ID:** 2603.16410 | [PDF](https://arxiv.org/pdf/2603.16410v1)

**作者:** Abhinav Thorat `[一作]` (Sony Research India), Niranjan Pedanekar `[通讯]` (Sony Research India)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PlotTwist框架，利用小型语言模型（≤3B主动参数）生成基于前提的高质量创意情节；

**💡 创新点**

核心创新在于将生成过程拆分为三个专门化模块：Aspect Rating Reward Model、通过Direct Preference Optimization训练的MoE情节生成器以及独立的Agentic Evaluation评估器；

**🔧 技术方法**

使用Positive‑Negative prompting构建情节质量维度（NQDs）奖励模型，采用Qwen-3-30B-A3B MoE与DPO进行偏好对齐，并通过agentic评估模拟人类批评；

**📊 数据集**

数据集由5000部电影情节与IMDb评分构成，利用LLM生成多维度情节评分，并进一步收集160条高置信度偏好对用于DPO训练；

**📈 对比分析**

与GPT‑4.1、Claude Sonnet 4、Gemini 2.0 Flash、Llama‑3‑70B、Agents' Room等大模型和基线进行比较，PlotTwist在五个NQD维度均优于或相当于规模远大于的模型，平均提升约0.8分；

**⚠️ 局限性**

局限性包括对小型模型仍需依赖高质量偏好数据、评价仍基于LLM推理而非人工专家、在极低质量情节下仍可能产生与原始情节差异较大、以及在特定创意风格上可能缺乏多样性。

---

## 28. Behavioral Steering in a 35B MoE Language Model via SAE-Decoded Probe Vectors: One Agency Axis, Not Five Traits

**arXiv ID:** 2603.16335 | [PDF](https://arxiv.org/pdf/2603.16335v1)

**作者:** Jia Qing Yap `[一作]` `[通讯]` (Independent Researcher), Jia Qing Yap (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Qwen 3.5‑35B混合专家模型的残差流上训练九个稀疏自编码器（SAE），利用对比激活对每个代理行为特质构建线性探针，并将探针权重投影回残差流得到可调向量，进而在推理阶段对模型进行精准行为干预。

**💡 创新点**

提出了在大规模混合专家与GatedDeltaNet/注意力架构中使用SAE解码的探针向量实现细粒度行为调节的全流程方法，并揭示了五种代理特质聚合为单一“代理轴”的现象以及预测与因果效应分离的风险校准例证。

**🔧 技术方法**

采用稀疏自编码器、TopK稀疏激活、线性Ridge探针、向量投影（从SAE潜在空间到残差流）、激活加法、Cohen’s d效应大小、Mann‑Whitney U检验等技术，并在GatedDeltaNet/注意力混合模型上实现。

**📊 数据集**

SAE训练使用200M来自HuggingFaceH4/ultrachat_200k、allenai/WildChat-1M以及合成工具使用对话的数据；行为评估采用50个ReAct式代理情境，涵盖编程、研究、沟通、数据分析四类任务。

**📈 对比分析**

通过比较原始模型与不同乘数α（1,2,3）下加入向量的模型，在工具调用、主动工具使用等代理指标上进行非参数统计检验，发现自主性向量在α=2时Cohen’s d≈1.01，工具使用向量在α=3时d≈0.39，但持续性与风险校准向量无显著提升；整体显示可调范围与模型规模正相关。

**⚠️ 局限性**

实验仅在单一Qwen 3.5‑35B模型上验证，样本量（50情境）对中等效应缺乏足够功效；代理指标仅基于工具调用计数，未考虑更细粒度的行为评分；未与mean‑difference基线或多模型泛化进行比较；缺乏最大激活示例与语义解释的深入分析。

---

## 29. DiFVM: A Vectorized Graph-Based Finite Volume Solver for Differentiable CFD on Unstructured Meshes

**arXiv ID:** 2603.15920 | [PDF](https://arxiv.org/pdf/2603.15920v1)

**作者:** Pan Du `[一作]` (University of Notre Dame), Jian-Xun Wang `[通讯]` (Cornell University)

**通讯引用:** 7200 | [OpenAlex ID](https://openalex.org/A5085043351)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了DiFVM，一款在GPU上原生运行的可微分有限体积CFD求解器，支持任意多面体网格。

**💡 创新点**

核心创新是将有限体积离散与图神经网络的消息传递等价化，实现了在不规则网格上的完全向量化和自动微分。

**🔧 技术方法**

采用JAX/XLA实现GPU加速、scatter/gather原语、PISO耦合、Rhie–Chow插值、Windkessel边界以及梯度检查点和隐式微分技术。

**📊 数据集**

使用多项基准数据集，包括三维Poisson、二维被动标量、弯管、盒子驱动腔、圆柱流动以及患者特定主动脉血流，另外还验证了逆向问题。

**📈 对比分析**

与OpenFOAM在相同网格与参数下对比，数值精度误差低于1e-5，GPU实现可比32核CPU快约1.6-2.3倍，单核更快16-33倍。

**⚠️ 局限性**

目前仅实现了不可压Navier–Stokes、刚性壁面，缺乏湍流、FSI、多GPU并行和多物理耦合功能。

---

## 30. Omnilingual SONAR: Cross-Lingual and Cross-Modal Sentence Embeddings Bridging Massively Multilingual Text and Speech

**arXiv ID:** 2603.16606 | [PDF](https://arxiv.org/pdf/2603.16606v1)

**作者:** Omnilingual SONAR Team `[一作]` (FAIR at Meta), Paul-Ambroise Duquenne `[通讯]` (FAIR at Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套覆盖4200+语言与177语音的全语言全模态句子编码器，并通过分阶段训练实现跨语言和跨模态的高质量对齐。

**💡 创新点**

提出了从LLM初始化的编码解码器、split‑softmax对比损失与硬负样本结合的训练框架，并通过教师‑学生MSE+对比蒸馏实现无监督的全语言扩展与语音迁移。

**🔧 技术方法**

使用LLM初始化的Encoder‑Decoder、token级翻译解码损失、split‑softmax对比损失、硬负样本生成、教师‑学生蒸馏、全语言词表扩展与温度采样等技术。

**📊 数据集**

利用200语言的平行翻译、人机翻译、合成数据；4,200+语言的Bible、PanLex、Tatoeba等多源并行文本；177语言的Omnilingual ASR 121k小时语音；代码和数学数据通过LLaMA生成的描述。

**📈 对比分析**

在FLORES、BIBLE、FLEURS等检索与翻译基准上与SONAR、LaBSE、MEXMA等同类模型对比，跨语言检索错误率下降超过50%，BIBLE错误率降至3.9，语音检索误差降低43%，翻译质量超越NLLB‑3B与70B LLM，MTEB下亦显著优于现有模型。

**⚠️ 局限性**

语音覆盖仅177语种，长文本对齐仍不够理想，模型规模大，对极低资源语言的细粒度性能仍受限，需进一步探索稀疏MoE或族群条件共享以提升泛化。

---

## 31. Robust Language Identification for Romansh Varieties

**arXiv ID:** 2603.15969 | [PDF](https://arxiv.org/pdf/2603.15969v1)

**作者:** Charlotte Model `[一作]` (University of Zurich), Jannis Vamvas `[通讯]` (University of Zurich)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5072479104)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了首个用于区分罗曼什六种方言（Sursilvan、Sutsilvan、Surmiran、Puter、Vallader及Rumantsch Grischun）的语言识别系统。

**💡 创新点**

创新点在于针对低资源、近亲语言变体设计了可迁移的词与字符 n-gram 特征集，并通过系统化的基准数据集展示了高精度识别效果。

**🔧 技术方法**

采用了支持向量机（SVM）作为分类器，配合词 unigrams 与字符 1–4 gram 的 TF‑IDF 表示，并在训练中尝试命名实体掩蔽。

**📊 数据集**

使用了五大来源的罗曼什文本：Pledari Grond 字典、La Quotidiana 报纸、RTR 广播转录、RTR Telesguard 记者笔记、Mediomatix 教材，并手工清洗、去重后划分为训练、验证与多种测试集。

**📈 对比分析**

与逻辑回归、梯度下降、朴素贝叶斯等基线对比，SVM 在平衡域内测试集上达 98.1% 的宏 F1 与 96.8% 准确率，而在不平衡或跨域测试中性能下降至约 69% 宏 F1。

**⚠️ 局限性**

主要限制包括训练数据显著的 Rumantsch Grischun 类别不平衡、对非正式文本（如记者笔记）的泛化能力不足，以及部分特征（如下划线字符）可能不符合标准正字法。

---

## 32. Adaptive regularization parameter selection for high-dimensional inverse problems: A Bayesian approach with Tucker low-rank constraints

**arXiv ID:** 2603.16066 | [PDF](https://arxiv.org/pdf/2603.16066v1)

**作者:** Qing-Mei Yang `[一作]`, Da-Qing Zhang `[通讯]` (University of Science and Technology Liaoning)

**通讯引用:** 7932 | [OpenAlex ID](https://openalex.org/A5101710686)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Tucker低秩分解的变分贝叶斯方法，用于自动选择高维逆问题中的正则化参数并同时估计噪声水平。

**💡 创新点**

通过将参数空间投影到低维核心张量空间实现计算可扩展性，并引入多模态精度参数实现各方向自适应正则化，从而克服传统变分贝叶斯在大规模问题上的计算瓶颈。

**🔧 技术方法**

采用变分贝叶斯推断、Tucker低秩张量分解、Gamma先验的多超参数模型以及闭式变分更新。

**📊 数据集**

在2D Fredholm积分方程、512×512 Cameraman图像去模糊、48³格的三维逆热传导问题以及大规模模拟数据集上进行实验。

**📈 对比分析**

与L‑curve、GCV、UPRE、DP等传统正则化参数选择方法对比，Tucker‑VB在PSNR/SSIM、误差以及噪声估计上均优于基线，提升幅度约20%–50%，且在高维场景下速度提升数千倍。

**⚠️ 局限性**

对Tucker分解秩的选择敏感，缺乏理论收敛性分析；在极低噪声或高频信息场景下可能欠缺细节恢复。

---

## 33. Diameter Computation on (Random) Geometric Graphs

**arXiv ID:** 2603.16684 | [PDF](https://arxiv.org/pdf/2603.16684v1)

**作者:** Thomas Bläsius `[一作]` (Karlsruhe Institute of Technology), Marcus Wilhelm `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5022079357)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套基于几何分离器的通用框架，利用递归划分与平衡小分离器快速求解随机几何图（Random Geometric Graph, RGG）的直径，并在该框架下给出了对方形与环面（torus）RGG的渐近时间上界，同时对常用的 iFUB 算法在 RGG 上的 2‑sweep 中枢点选择和 BFS 次数做了理论分析，证明在方形 RGG 上可获得显著加速。

**💡 创新点**

创新点在于：①将几何观念（相交圆、分离条纹、碎片化）转化为确定性图性质；②通过递归四叉树划分构造平衡且小分离器的递归划分；③给出在 RGG 上高概率满足“局部近直径伙伴”和“少角点”性质，从而实现直径计算时间上界 Õ(n^{max(3/2+3ρ,2−2/3ρ)})（方形）和 Õ(n^{max(3/2+3ρ,2−10/3ρ)})（环面），显著优于以往 O*(n^{2−1/18}) 的结果；④为 iFUB 提供了第一套理论上证明其在方形 RGG 上获得 Õ(n^{δ/3}) 加速而环面上不加速的理论依据。

**🔧 技术方法**

技术方法包括：随机几何图的几何-图距离伸缩分析、四叉树递归划分、平衡分离器与碎片化性质的证明、基于分离器的最大距离（max‑dist）求解、以及对 iFUB 的 2‑sweep 中枢点定位与 BFS 分布分析。

**📊 数据集**

实验与验证仅在理论层面使用了随机几何图模型（均匀点分布在单位正方形或平面环面上，连通半径 r 与 n 的关系取 r=n^ρ），未使用真实网络数据集。

**📈 对比分析**

相对方法：naïve O(nm)、Seidel 的 Õ(n^ω)、以及基于单位圆盘图的直径算法。本文的时间上界在期望平均度数为 Θ(n^δ)（δ∈(0,1/2)) 的 RGG 上可达 Õ(n^{max(3/2+3ρ,2−2/3ρ)})（环面）和 Õ(n^{max(3/2+3ρ,2−10/3ρ)})（方形），在平均度数为 n^{3/19} 时可获得约 Õ(n^{1.737}) 的加速，明显优于以往的 O*(n^{2−1/18}) 上限。

**⚠️ 局限性**

局限性包括：①理论结果仅在渐近高概率意义下成立，无法保证所有具体实例；②需要构造递归划分与分离器，参数选择与实现复杂度较高；③在环面（无边界、无角点）几何图上，iFUB 无法获得理论加速，甚至退化至与朴素 BFS 相当；④若图的几何结构不满足框架假设（如非平衡或分离器过大），则算法性能下降。

---

## 34. HGP-Mamba: Integrating Histology and Generated Protein Features for Mamba-based Multimodal Survival Risk Prediction

**arXiv ID:** 2603.16421 | [PDF](https://arxiv.org/pdf/2603.16421v1)

**作者:** Jing Dai `[一作]` (Cancer Hospital of Dalian University of Technology), Hongming Xu `[通讯]` (Dalian University of Technology)

**通讯引用:** 1024 | [OpenAlex ID](https://openalex.org/A5101554514)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

未提供论文内容，无法总结。

**💡 创新点**

无法判断。

**🔧 技术方法**

无法判断。

**📊 数据集**

无法判断。

**📈 对比分析**

无法判断。

**⚠️ 局限性**

无法判断。

---

## 35. Fine-Grained Network Traffic Classification with Contextual QoS Profiling

**arXiv ID:** 2603.16748 | [PDF](https://arxiv.org/pdf/2603.16748v1)

**作者:** Huiwen Zhang `[一作]` (University of Wisconsin), Feng Ye `[通讯]` (University of Wisconsin)

**通讯引用:** 7172 | [OpenAlex ID](https://openalex.org/A5052566913)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个多层次图神经网络框架，实现细粒度、QoS感知的网络流量分类。

**💡 创新点**

创新点在于三层层级图结构与基于五项QoS指标的自动量化分级算法，兼顾多尺度时序特征和QoS优先级。

**🔧 技术方法**

使用了GATv2的双层图编码器、权重聚合池化、QoS加权损失、log10尺度变换及多头注意力等技术。

**📊 数据集**

数据集来自四大应用（YouTube、Prime Video、TikTok、Zoom）10分钟PCAP，划分14类共14种使用场景。

**📈 对比分析**

与传统MLP以及现有包级NTC对比，模型在14类上准确率约86%，QoS体验得分比基线高约8.5分（96.78% vs 88.30%）。

**⚠️ 局限性**

局限在于仅验证四大应用，QoS保守策略可能导致资源浪费，需进一步扩展到更广泛流量与动态QoS调整。

---

## 36. INSTRUMENTAL: Automatic Synthesizer Parameter Recovery from Audio via Evolutionary Optimization

**arXiv ID:** 2603.15905 | [PDF](https://arxiv.org/pdf/2603.15905v1)

**作者:** Philipp Bogdan `[一作]` `[通讯]`, Philipp Bogdan

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过将可微分的28参数减法合成器与无梯度进化优化器CMA-ES结合，实现了从录音音频中恢复可直接播放的合成器补丁参数的完整流程。

**💡 创新点**

首次在音频到MIDI的逆向问题中引入可微分合成器与CMA-ES，并通过复合感知损失、谱分析初始化及多音高联合优化，系统验证了参数恢复的有效性，并揭示了传统低通滤波器对音色重建的局限。

**🔧 技术方法**

可微分的子抽取合成器、CMA-ES进化优化、基于mel-STFT、谱心差、MFCC的感知损失、Spectral初始化、多音高损失、批量评估、对比实验等技术。

**📊 数据集**

使用一段来自商业歌曲的录音，提取出22个约150 ms的主音音符，包含三种代表性音高，作为目标音频进行参数恢复。

**📈 对比分析**

将CMA-ES与梯度下降、PPO强化学习、预训练编码器等方法在相同损失函数和评价指标下对比，结果显示CMA-ES在10万次评估中将匹配损失降至2.09，远优于梯度下降（4.60）与PPO（2.31）等。

**⚠️ 局限性**

受限于减法合成器的表达上限，无法重现H3>H2的高谐波特征；参数数量扩大到29时易导致不合理极值；仅在单一目标音频上验证，缺乏普适性。

---

## 37. Gaze-Aware Task Progression Detection Framework for Human-Robot Interaction Using RGB Cameras

**arXiv ID:** 2603.15951 | [PDF](https://arxiv.org/pdf/2603.15951v1)

**作者:** Linlin Cheng `[一作]` (Vrije Universiteit Amsterdam), Artem V. Belopolsky `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 5600 | [OpenAlex ID](https://openalex.org/A5045326147)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并验证了一种低成本、无标定、基于RGB相机的视线估计框架，用于人机交互中的任务完成检测。

**💡 创新点**

通过利用用户从任务界面自然转向机器人面部的视线变化来触发页面切换，实现无按钮、无标定、实时交互。

**🔧 技术方法**

采用L2CS-Net深度网络进行3D视线估计、滑动平均滤波、三维到二维投影以及基于AOI的阈值状态检测。

**📊 数据集**

模型训练基于大规模无约束数据集Gaze360；实验数据来自Pepper机器人与手工记录的参与者互动。

**📈 对比分析**

在“First Day at Work”情境下将视线驱动的页面进度与按钮驱动进行对比，平均成功率77.6%，略高于按钮的平均时长但记忆表现相同，用户体验更高人性化与舒适度。

**⚠️ 局限性**

受限于光照、眼镜反射、低帧率和界面布局导致视线误判，且对不同阅读速度的适配不足，需进一步改进自适应阈值和扩展工作距离。

---

## 38. TurnWise: The Gap between Single- and Multi-turn Language Model Capabilities

**arXiv ID:** 2603.16759 | [PDF](https://arxiv.org/pdf/2603.16759v1)

**作者:** Victoria Graf `[一作]` (University of Washington), Hannaneh Hajishirzi `[通讯]` (Allen Institute for AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个新的多轮对话基准，并提出了一种可扩展的合成多轮数据生成管道，用于评估和提升语言模型在多轮对话中的表现。

**💡 创新点**

创新点在于：1) 通过与单轮基准的成对比较，精准隔离多轮对话能力；2) 设计了多轮数据合成方法，将单轮提示自动扩展为多轮对话；3) 通过少量合成多轮数据即可显著缩小多轮-单轮性能差距。

**🔧 技术方法**

采用了 GPT‑4.1 作为文本生成与评判工具，使用监督微调 (SFT) 与直接偏好优化 (DPO) 进行模型训练，并利用自我对话 (self‑talk) 作为基线生成。

**📊 数据集**

数据来源包括 AlpacaEval 种子提示、WildChat 以及 Dolci Instruct 的单轮对话；合成多轮数据通过 GPT‑4.1 生成用户回合，并在实验中加入了自我对话与人工检查的交互。

**📈 对比分析**

对比方法：使用 GPT‑4.1 评判器在绝对 (Absolute) 与自我 (Self) 两种设置下计算胜率，评估模型在多轮与等价单轮场景中的相对优势。实验表明，引入 10k 甚至 20k 条合成多轮对话后，Olmo‑3‑7B 的多轮胜率提升可达 12% 左右，且单轮性能保持稳定。

**⚠️ 局限性**

局限性包括：1) 评估结果对中间回合质量敏感，弱化模型的上下文处理能力；2) 训练设置固定在 Olmo‑3‑7B 的超参数，缺乏对其他模型或更大规模实验的泛化；3) 合成对话仍依赖强大生成模型，生成误差可能影响评估与训练效果。

---

## 39. Optimal uncertainty bounds for multivariate kernel regression under bounded noise: A Gaussian process-based dual function

**arXiv ID:** 2603.16481 | [PDF](https://arxiv.org/pdf/2603.16481v1)

**作者:** Amon Lahr `[一作]` (Institute for Dynamical Systems and Control), Melanie N. Zeilinger `[通讯]` (Institute for Dynamical Systems and Control)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对多变量核回归的紧凑、不依赖于分布的不确定性界限，旨在解决现有方法在处理噪声分布假设、保守性和多输出情况时的局限性。

**💡 创新点**

创新点在于通过无约束的对偶函数形式推导出一种新的不确定性界限，该界限可以直接集成到后续的优化流程中，并且能够处理多输出的情况。

**🔧 技术方法**

使用了高斯过程回归和对偶函数的无约束优化技术。

**📊 数据集**

使用了包含噪声的多变量数据集，具体数据集的噪声被假设为受限于椭球形的不确定性集合。

**📈 对比分析**

与现有的确定性界限进行了比较，结果表明所提出的界限在保守性和求解时间上优于其他方法，尤其是在多变量四旋翼动态学习的应用示例中表现良好。

**⚠️ 局限性**

限制在于该方法可能在某些情况下无法达到最优解，尤其是在噪声参数趋近于零或无穷大的极限情况下。

---

## 40. ADAPT: Adaptive Dual-projection Architecture for Perceptive Traversal

**arXiv ID:** 2603.16328 | [PDF](https://arxiv.org/pdf/2603.16328v1)

**作者:** Shuo Shao `[一作]` (University of Science and Technology of China), Shiwu Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6412 | [OpenAlex ID](https://openalex.org/A5101816100)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种适应性双投影感知框架 ADAPT，能够在三维复杂环境中动态调整感知半径，实现机器人步态控制与感知精度、计算效率的平衡。

**💡 创新点**

创新点包括将3D环境拆分为水平高程图和垂直距离图两种轻量投影；将感知半径作为可学习的动作，让策略主动控制感知视野；将感知与动作耦合的闭环控制实现零样本 sim‑to‑real 转移。

**🔧 技术方法**

采用强化学习（PPO）、双投影感知、可学习感知半径、GRU 递归网络、异步主从结构、LiDAR 点云投影、GPU 加速地图生成等技术。

**📊 数据集**

在仿真中使用 BarrierTrack 生成的程序化障碍物（阶梯、跌落、跳跃、障碍等）以及 Unitree G1 humanoid 机器人；真实部署使用 Livox Mid‑360 LiDAR 与 Fast‑LIO2 进行点云注册。

**📈 对比分析**

与四个基线（PIM、Parkour、TSDF、Gallant）在七种障碍类型下进行多速率评测，ADAPT 在所有地形上成功率最高（最高 100%）、训练时间最短（11.3 s/iter，GPU 15870 MB），并在实际机器人上实现 94.7% 成功率。

**⚠️ 局限性**

局限性包括对 LiDAR 传感器的光照与遮挡敏感；感知半径调节仅针对静态环境，未考虑动态障碍；当前框架仅处理 2D 投影，无法捕捉复杂垂直结构细节。

---

## 41. Physics-guided diffusion models for inverse design of disordered metamaterials

**arXiv ID:** 2603.16209 | [PDF](https://arxiv.org/pdf/2603.16209v1)

**作者:** Ziyuan Xie `[一作]` (Hong Kong University of Science and Technology), Tianju Xue `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 817 | [OpenAlex ID](https://openalex.org/A5069417521)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了物理引导扩散模型（Physics-Guided Diffusion Model），通过在逆向扩散过程中直接使用可微物理求解器作为引导，完成对无序泡沫结构的逆设计。

**💡 创新点**

创新点在于：①不需要在训练阶段就加入条件或物理残差；②将物理求解器的梯度直接嵌入扩散过程，实现任务自适应；③只需一次无标签训练，后续可即时切换不同设计目标。

**🔧 技术方法**

技术：基于VP SDE的分数网络（U‑Net）训练扩散模型；利用可微有限元/粒子方法、相位场断裂模型等求解器计算物理量；在采样时将物理梯度加入反向SDE实现引导。

**📊 数据集**

数据集：利用Voronoi分割生成 6,400 张 64×64 的闭单胞泡沫二值图像（不同单元尺寸与体积分数），无标注。

**📈 对比分析**

方法比较：在三个案例（目标热导率、压缩力-位移曲线、能量吸收）中，物理引导扩散模型能够精确匹配目标值，生成多样且满足物理约束的结构；相较传统拓扑优化、GAN/VAEs 等方法，显著降低标注需求、提升样本多样性并实现一次性训练后多目标适应，性能优异。

**⚠️ 局限性**

局限性：①生成能力受训练数据分布限制，难以产生与数据无关的周期或特殊结构；②引导强度参数需手工调节，过强会导致收敛缓慢或结构异常；③对复杂物理求解器依赖较高，求解成本成为瓶颈。

---

## 42. Mastering the Minority: An Uncertainty-guided Multi-Expert Framework for Challenging-tailed Sequence Learning

**arXiv ID:** 2603.15708 | [PDF](https://arxiv.org/pdf/2603.15708v1)

**作者:** Ye Wang `[一作]` (Chongqing University of Post and Telecommunications), Guoyin Wang `[通讯]` (Chongqing Normal University)

**通讯引用:** 19810 | [OpenAlex ID](https://openalex.org/A5031220156)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于不确定性引导的多专家融合网络（UME），用于解决序列学习中的长尾类别问题。

**💡 创新点**

创新点包括：利用LoRA实现参数高效的低秩专家；通过Dempster–Shafer理论的序列专门化让专家聚焦难样本；采用不确定性引导融合机制动态加权专家预测。

**🔧 技术方法**

使用BERT编码器、LoRA低秩适配、Dempster–Shafer理论进行不确定性建模、Sequential Specialization顺序训练专家以及对抗式与对比学习损失。

**📊 数据集**

在四个公开的层级文本分类数据集上评测：Web of Science（WOS）、RCV1‑V2、AAPD、BGC（以及NYT）。

**📈 对比分析**

与14个强基线对比，UME在Micro‑F1/Macro‑F1上分别超过HiTIN/HiAdv等模型约0.4%/0.3%，在尾部类别上提升最高可达11%以上，并显著减少可训练参数。

**⚠️ 局限性**

局限性包括：需要预先训练BERT且仍需逐层训练专家导致训练并行度低；在极端稀疏类别下性能仍有限；对超参数如温度、冲突阈值敏感。

---

## 43. Poisoning the Pixels: Revisiting Backdoor Attacks on Semantic Segmentation

**arXiv ID:** 2603.16405 | [PDF](https://arxiv.org/pdf/2603.16405v1)

**作者:** Guangsheng Zhang `[一作]` (University of Technology Sydney), Bo Liu `[通讯]` (University of Technology Sydney)

**通讯引用:** 7297 | [OpenAlex ID](https://openalex.org/A5100461646)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统性研究了语义分割模型的后门攻击，提出多种攻击向量并设计了BADSEG统一框架。

**💡 创新点**

首次识别四类粗粒度与两类细粒度后门攻击向量，并通过Gumbel‑Softmax优化触发器和语义距离标签选择，实现高效、隐蔽的后门注入。

**🔧 技术方法**

采用触发器参数优化、标签操纵、语义中心距离度量、Gumbel‑Softmax离散搜索，以及对Transformer与SAM的适配等技术。

**📊 数据集**

在BDD100K、Cityscapes两大自动驾驶分割基准上进行实验，并针对SAM使用LabPicsV1数据集。

**📈 对比分析**

与已有HBA、OFBA、IBA等攻击及六种主流防御（Fine‑Tuning、Pruning、ABL、STRIP、TeCo、Beatrix）对比，BADSEG在多种模型上实现ASR>0.9且对干净样本准确率几乎不降，防御效果极差。

**⚠️ 局限性**

攻击依赖于可控的训练集污染和与目标模型相似的代理模型，且未覆盖实时硬件部署、可解释性及更广泛安全场景的完整解决方案。

---

## 44. 3D Fourier-based Global Feature Extraction for Hyperspectral Image Classification

**arXiv ID:** 2603.16426 | [PDF](https://arxiv.org/pdf/2603.16426v1)

**作者:** Muhammad Ahmad `[一作]` (King Fahd University of Petroleum and Minerals), Muhammad Ahmad `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 3494 | [OpenAlex ID](https://openalex.org/A5044102676)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种结合3D卷积局部特征提取与GFNet式全局频域过滤的混合网络HGFNet，用于高光谱图像分类。

**💡 创新点**

创新点在于：①设计了三种针对高光谱数据的频域变换（光谱FFT、空间FFT、空间-光谱3D FFT）以充分利用跨波段和空间的频谱关系；②引入Adaptive Focal Loss动态调整类别权重和关注度，以缓解极端类别不平衡；③将局部卷积与全局频域模块相结合，兼顾细粒度局部语义和长距离全局一致性。

**🔧 技术方法**

使用技术包括3D卷积、GFNet风格频域全局过滤、FFT变换、可学习频率掩模、GELU激活、归一化、残差连接、全连接分类头和自适应焦点损失。

**📊 数据集**

实验数据集包括印度针叶林（Indian Pines）、WHU Hi HanChuan、WHU Hi HongHu 三个公开高光谱数据集。

**📈 对比分析**

与SpectralFormer、Hybrid ViT、WaveFormer、Mamba、WMamba等最新方法对比，HGFNet在三组数据集上均实现了最高的整体精度、平均精度和Kappa值，显著提升了分类准确率和空间连贯性。

**⚠️ 局限性**

局限性在于：①频域全局特征假设信号平稳，对强非平稳光谱变化（如光照、气象影响）适应性不足；②3D卷积和FFT计算开销仍较高，可能限制在极高分辨率场景下的部署；③Adaptive Focal Loss在极度不平衡场景下仍可能不足，需要更高级的采样或元学习策略。

---

## 45. Dataflow-Oriented Classification and Performance Analysis of GPU-Accelerated Homomorphic Encryption

**arXiv ID:** 2603.16692 | [PDF](https://arxiv.org/pdf/2603.16692v1)

**作者:** Ai Nozaki `[一作]` (University of Tokyo), Hiroshi Nakamura `[通讯]` (University of Tokyo)

**通讯引用:** 24464 | [OpenAlex ID](https://openalex.org/A5090578339)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对GPU加速的CKKS同态加密进行了数据流分类，并评估了不同CKKS参数配置下各种GPU优化策略的性能，揭示最佳策略随参数变化而变化。

**💡 创新点**

首次证明最佳GPU优化策略依赖于CKKS参数配置，并提出一种基于数据流内存占用的分类框架，用以指导不同参数下的优化选择。

**🔧 技术方法**

使用GPU并行计算、CKKS同态加密、数据流分析技术以及定量性能测量方法。

**📊 数据集**

采用多组CKKS参数组合的基准工作负载（如矩阵乘法、近似浮点运算等）并在不同GPU架构上进行实验。

**📈 对比分析**

通过在同一GPU上对比多种优化策略的运行时间和吞吐量进行定量评估，发现最佳策略因参数不同而差异显著，性能提升可达1.98倍，且不同GPU架构的最优策略不相同。

**⚠️ 局限性**

局限性包括仅针对CKKS方案，未涵盖其他同态加密方案；实验范围受限于少数GPU架构；参数空间有限，未验证极大规模或多GPU部署情况。

---

## 46. Enhancing Linguistic Generalization of VLA: Fine-Tuning OpenVLA via Synthetic Instruction Augmentation

**arXiv ID:** 2603.16044 | [PDF](https://arxiv.org/pdf/2603.16044v1)

**作者:** Dongik Shin `[一作]` `[通讯]` (University of Texas), Dongik Shin (University of Texas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对OpenVLA进行参数高效微调，使用LoRA并结合LLM生成的多样化指令集提升其语言泛化能力。

**💡 创新点**

通过LLM合成结构多样、语义相同的指令集来扩充语言空间，并用LoRA实现低成本、高效的微调。

**🔧 技术方法**

采用LoRA参数高效微调技术，利用大语言模型（LLM）生成指令，并在OpenVLA模型上进行训练。

**📊 数据集**

使用Bridge Dataset V2中脚本化收集的机器人轨迹数据（约100条轨迹）。

**📈 对比分析**

与零射击OpenVLA对比，Top‑1精度略降（5.09% vs 6.62%），但5‑Bin容忍度提升至42.47%（高于40.76%）。

**⚠️ 局限性**

仅在小规模100轨迹子集验证，精度略低，且未在多阶段任务或真实环境中进行可扩展性和鲁棒性评估。

---

## 47. Learning Human-Object Interaction for 3D Human Pose Estimation from LiDAR Point Clouds

**arXiv ID:** 2603.16343 | [PDF](https://arxiv.org/pdf/2603.16343v1)

**作者:** Daniel Sungho Jung `[一作]` (Seoul National University), Kyoung Mu Lee `[通讯]` (Seoul National University)

**通讯引用:** 27076 | [OpenAlex ID](https://openalex.org/A5046504049)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出HOIL框架，通过学习人-物交互来实现基于LiDAR点云的3D人体姿态估计。

**💡 创新点**

创新点包括人-物交互感知对比学习（HOICL）和基于接触的部件引导池化（CPPool）两大机制，分别解决空间模糊和类别不平衡问题，并可选的时间接触细化。

**🔧 技术方法**

采用Point Transformer V3骨干，结合监督对比学习、分支交互分割/接触预测、键点查询注意力及交互式池化等技术。

**📊 数据集**

预训练使用五个HOI数据集（BEHAVE、CHAIRS、HODome、OMOMO、InterCap），微调在Waymo和SLOPER4D真实LiDAR数据集上。

**📈 对比分析**

在Waymo和SLOPER4D上与NE、LPFormer、PRN、DAPT等SOTA方法对比，HOIL在MPJPE、PCK‑3/5等指标上均获得显著提升，尤其在复杂交互场景表现突出。

**⚠️ 局限性**

局限包括仅使用LiDAR缺乏RGB语义、距离越远点云稀疏导致精度下降，以及训练数据中缺乏某些常见交互（如骑行、摩托车）导致泛化受限。

---

## 48. Dialect-Agnostic SQL Parsing via LLM-Based Segmentation

**arXiv ID:** 2603.16155 | [PDF](https://arxiv.org/pdf/2603.16155v1)

**作者:** Junwen An `[一作]` (National University of Singapore), Manuel Rigger `[通讯]` (National University of Singapore)

**通讯引用:** 472 | [OpenAlex ID](https://openalex.org/A5066738024)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SQLFlex，一个将语法分析器与LLM分段器相结合的跨方言SQL解析框架；

**💡 创新点**

其创新点在于引入句法层级与表达式层级的分段策略，并通过验证与修复机制提升解析可靠性；

**🔧 技术方法**

技术实现依赖ANTLR生成的基准语法解析器、LangChain包装的OpenAI GPT‑4.1分段器及自定义Pydantic结构化输出；

**📊 数据集**

实验使用真实场景的SQL linting和测试用例减缩数据集，并从八个主流DBMS的测试套件中抽取SQL语句进行独立评测；

**📈 对比分析**

在ANSI模式下相较于SQLFluff的F1得分提升63.68%，在测试用例减缩中简化率提高至10倍以上，整体解析成功率在91.55%–100%之间，优于所有基线解析器；

**⚠️ 局限性**

局限性包括缺乏形式化语义正确性保证、对LLM的依赖导致性能瓶颈，以及在多方言语义细节（如运算符优先级）上仍可能出现误判。

---

## 49. An Agentic Evaluation Framework for AI-Generated Scientific Code in PETSc

**arXiv ID:** 2603.15976 | [PDF](https://arxiv.org/pdf/2603.15976v1)

**作者:** Hong Zhang `[一作]` (Argonne National Laboratory), Junchao Zhang `[通讯]` (Argonne National Laboratory)

**通讯引用:** 293 | [OpenAlex ID](https://openalex.org/A5101999695)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于 agent-evaluating-agent 模式的评估框架，利用工具增强的评估者 Agent 对 AI 生成的 PETSc 科学代码进行多维度的编译、执行、性能、可读性、算法适用性和库特定约定等方面的评估。

**💡 创新点**

创新点在于：①把评估器也做成 Agent，采用 A2A 与 MCP 协议实现黑盒、可插拔的评估；②构建了 14 个评估器的 3 阶段评估管线，覆盖从编译门到数值准确度、性能、代码质量、算法选择等 5 个评分维度；③结合 LLM 进行主观评估与静态分析，形成置信度加权的复合得分。

**🔧 技术方法**

核心技术包括：大语言模型（Claude、Gemini、GPT）生成代码；A2A（Agent-to-Agent）和 MCP（Model Context Protocol）协议实现 Agent 间通信；MCP 提供 sandbox 编译/执行服务；Valgrind 等工具进行内存安全检测；LLM 进行可读性、风格、文档、错误处理等质量评估；多维度得分合成算法。

**📊 数据集**

使用了自研的 PETSc 基准套件，包含 6 个覆盖 TS、KSP、DM、SNES 等模块的真实问题，难度从 Easy 到 Hard，均采用 JSON 形式的自然语言描述与参考输出。

**📈 对比分析**

对比方法：在相同问题上让 Claude Opus 4.6、Gemini 2.5 Pro 和 GPT‑5.2 各自生成一次代码，并分别通过框架评估。实验显示平均复合得分分别为 46.4、42.4、39.9；Gate 通过率分别为 67%、61%、56%；在库特定维度得分低，表明当前模型在 PETSc 约定、错误处理、求解器选择等方面仍不足。

**⚠️ 局限性**

限制：①未覆盖 GPU、混合物理等 PETSc 先进特性；②评估者 LLM 的随机性导致分数波动；③评估流程未实现多轮修正循环，无法充分利用编译反馈；④未涵盖完整 PETSc 代码库的多样性，基准集仍有扩展空间。

---

## 50. Semantic One-Dimensional Tokenizer for Image Reconstruction and Generation

**arXiv ID:** 2603.16373 | [PDF](https://arxiv.org/pdf/2603.16373v1)

**作者:** Yunpeng Qu `[一作]` (Alibaba Group), Jian Wang `[通讯]` (Tsinghua University)

**通讯引用:** 138138 | [OpenAlex ID](https://openalex.org/A5100452094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将二维图像压缩为一维离散语义标记的视觉分词器SemTok，并基于该分词器构建了掩码自回归生成框架。

**💡 创新点**

创新点包括：1) 2D→1D转换以消除空间冗余，获得全局语义；2) 在编码器两分支上施加语义对齐约束，促使标记聚类分布；3) 两阶段生成训练（扩散预训练+一阶细化）以拓展潜在空间并提升细节；4) 掩码自回归模型利用全局上下文并行生成标记。

**🔧 技术方法**

核心技术包括：MMDiT架构的双流编码器、SigLIP语义对齐、二进制球面量化(BSQ)、Diffusion预训练、GAN+LPIPS感知损失、掩码自回归Transformer。

**📊 数据集**

主要使用ImageNet‑1k（256×256）进行分词器训练与评估，生成任务也在同一数据集上完成。

**📈 对比分析**

与现有2D/1D分词器（VQGAN、Taming、FlowMo、FlexTok等）以及AR生成模型（MaskGIT、MAGE、VAR、LlamaGen、RandAR等）对比，SemTok在图像重建的rFID、PSNR、SSIM、LPIPS等指标上取得SOTA，并在类条件生成任务中在gFID、IS、Precision/Recall等指标上实现与SOTA相当甚至优于VAR、LlamaGen的表现。

**⚠️ 局限性**

限制包括：仍依赖大规模预训练模型（SD3.5、SigLIP）和显著计算资源；1D编码对局部纹理表达有一定损失；生成时对语义对齐约束的依赖可能限制跨模态应用；缺乏对高分辨率或非自然图像的评估。

---

## 51. GIST: Gauge-Invariant Spectral Transformers for Scalable Graph Neural Operators

**arXiv ID:** 2603.16849 | [PDF](https://arxiv.org/pdf/2603.16849v1)

**作者:** Mattia Rigotti `[一作]` (IBM Research), Thomas Frick `[通讯]` (IBM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种称为GIST的Gauge-Invariant Spectral Transformer，用于在图和网格数据上实现尺度可扩展的注意力学习；

**💡 创新点**

通过仅在近似谱嵌入的内积上计算注意力，保证了对谱投影随机性（gauge）不敏感，同时利用随机投影实现O(N)复杂度；

**🔧 技术方法**

使用快速随机投影FastRP生成近似谱嵌入，结合线性注意力、Johnson‑Lindenstrauss保持内积不变以及多尺度Transformer结构；

**📊 数据集**

在标准图数据集（Cora、PubMed、PPI、Elliptic、Arxiv、Photo）以及大规模网格回归数据集（DrivAerNet、DrivAerNet++）上进行评估；

**📈 对比分析**

与传统GNN、GraphSAGE、GAT、SGFormer、SpecFormer、PolyFormer、Exphormer等方法比较，GIST在节点分类任务上与SOTA相当或略优，在网格回归任务上实现了新的最优（Rel L2 20.10%/18.60%）；

**⚠️ 局限性**

对随机投影的近似误差与内积保持的误差做了理论上界约束，实际对非常稀疏或高维图的性能未充分验证，且对极端数值不稳定情况仍需进一步研究。

---

## 52. FG-SGL: Fine-Grained Semantic Guidance Learning via Motion Process Decomposition for Micro-Gesture Recognition

**arXiv ID:** 2603.16269 | [PDF](https://arxiv.org/pdf/2603.16269v1)

**作者:** Jinsheng Wei `[一作]` (Nanjing University of Posts and Telecommunications), Jingjie Yan `[通讯]` (Nanjing University of Posts and Telecommunications)

**通讯引用:** 872 | [OpenAlex ID](https://openalex.org/A5101509694)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种微手势识别框架FG‑SGL，利用细粒度语义指导学习与类别级语义对齐提升识别精度。

**💡 创新点**

创新点在于构建FG‑Text细粒度文本先验，并通过多级对比优化实现中层视觉特征与细粒度语义、高层视觉特征与类别语义的双向对齐。

**🔧 技术方法**

核心技术包括基于预训练视觉‑语言模型的对比学习、LoRA参数高效微调、以及分阶段多级对比优化（ML‑CO）策略。

**📊 数据集**

使用公开微手势数据集SMG和iMiGUE，并在SMG上构造FG‑Text四维细粒度注释。

**📈 对比分析**

与RGB、骨骼和多模态基线相比，FG‑SGL在SMG上达到78.13%/62.58%（iMiGUE）的最高分类准确率，显著优于同类方法。

**⚠️ 局限性**

局限性包括对细粒度文本注释的人工依赖、仅在两大数据集验证，且在更复杂或跨域手势场景中的泛化能力尚待评估。

---

## 53. Cost Trade-offs in Matrix Inversion Updates for Streaming Outlier Detection

**arXiv ID:** 2603.16697 | [PDF](https://arxiv.org/pdf/2603.16697v1)

**作者:** Florian Grivet `[一作]` (National Centre for Space Studies), Louise Travé-Massuyès `[通讯]` (LAAS-CNRS, University of Toulouse)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对三种矩阵逆更新方法（Direct Inversion、Iterative Sherman–Morrison、Woodbury Matrix Identity）在 Christoffel 函数基准的流式离群检测中进行了理论计算、实验验证，并给出了根据矩阵尺寸 s 与更新秩 k 选择最优方法的经验规则。

**💡 创新点**

创新点在于推导出这三种方法的计算量阈值，并将其与实验结果对比，最终得到一个简单易记的“k ≤ s/3 用 WMI，k > s/3 用 DI，k=1 用 ISM”的指导原则；同时首次系统评估了这些方法在 Python CPU 环境下的数值稳定性与性能差异。

**🔧 技术方法**

技术手段包括 Christoffel 函数的矩阵形式、rank‑k 更新的三种经典公式、Cholesky 分解、浮点运算计数、Python 中的 NumPy/Scipy 线性代数实现、实验中对时间与误差的统计测量。

**📊 数据集**

使用了合成数据集：先生成 S=2000（以及 15000）个随机向量在 ℝ^1287（对应 d=8, n=5）得到的矩阵尺寸 s=1287，随后在不同 k（1~1000）下进行更新实验；实验重复 ns=200 次以获得平均时间和误差。

**📈 对比分析**

在实验中，DI 在大多数 k 下最快（尤其 k>~s/3），ISM 在 k=1 时最快，WMI 在 1<k≤s/3 时最快；所有方法的数值误差随 k 增大而上升，尤其在样本不足时 WMI 与 ISM 更易出现不稳定；理论阈值与实验结果高度吻合，验证了提出的选择规则。

**⚠️ 局限性**

主要局限是结论仅在 Python CPU 环境下验证，无法直接推广至其他编程语言、库或 GPU 加速场景；此外，Christoffel 函数的矩阵尺寸随原始维度快速增长，本文未提供有效降维或稀疏化的方案，限制了在高维流式数据上的直接应用。

---

## 54. From the Inside Out: Progressive Distribution Refinement for Confidence Calibration

**arXiv ID:** 2603.16500 | [PDF](https://arxiv.org/pdf/2603.16500v1)

**作者:** Xizhong Yang `[一作]` (Southeast University), Mofei Song `[通讯]` (Southeast University)

**通讯引用:** 235 | [OpenAlex ID](https://openalex.org/A5037738070)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无标签测试时训练中，本文提出利用模型内部置信度分布先验来动态构建伪标签，并引入查询级多样性惩罚以抑制一致性奖励攻击。

**💡 创新点**

创新点在于逐步构建并校正置信度分布、利用分布先验进行伪标签估计，以及将多样性权重融入优势函数，从而提升奖励信号质量和训练稳定性。

**🔧 技术方法**

采用分布式TTS（DistriVoting）、高斯混合模型置信度建模、GRPO强化学习框架以及多样性惩罚机制。

**📊 数据集**

在AIME、AMC、MATH-500和GPQA-D等数学与推理基准上，并使用Qwen2.5、Qwen3、DeepSeek等大型语言模型进行验证。

**📈 对比分析**

与TTRL、TTRL-WSC以及Distrivoting等基线对比，DistriTTRL在五个模型和三个基准上平均提升0.51~4.33分，显著提高性能。

**⚠️ 局限性**

局限性包括对训练预算和分布先验假设的依赖、对多样性阈值的敏感性以及在极大规模数据上可能的计算开销。

---

## 55. Federated Learning with Multi-Partner OneFlorida+ Consortium Data for Predicting Major Postoperative Complications

**arXiv ID:** 2603.16723 | [PDF](https://arxiv.org/pdf/2603.16723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 56. Parameter-Efficient Deep Learning for Ultrasound-Based Human-Machine Interfaces

**arXiv ID:** 2603.15625 | [PDF](https://arxiv.org/pdf/2603.15625v1)

**作者:** Antonios Lykourinas `[一作]` (Princeton University), Athanassios Skodras `[通讯]` (ABC Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了论文排版与结构的规范指南，主要说明了标题、章节、图表、公式等格式要求。

**💡 创新点**

创新点在于对 LNCS 版式细节的系统化总结，帮助作者统一排版标准。

**🔧 技术方法**

使用的技术为 LaTeX 版式宏包，强调使用向量图而非栅格图。

**📊 数据集**

未使用任何数据集，仅为格式说明。

**📈 对比分析**

不涉及实验或方法比较，因此无性能评估。

**⚠️ 局限性**

局限在于内容仅为排版指导，未提供研究成果或应用案例。

---

## 57. IRAM-Omega-Q: A Computational Architecture for Uncertainty Regulation in Artificial Agents

**arXiv ID:** 2603.16020 | [PDF](https://arxiv.org/pdf/2603.16020v1)

**作者:** Veronique Ziegler `[一作]` `[通讯]` (Independent Researcher), Veronique Ziegler (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了 IRAM‑Ω‑Q 计算架构，使用密度矩阵描述认知状态，并通过闭环自适应增益控制在噪声下实现熵调节；随后通过系统参数扫描、出版模式仿真和可敏感性相位图分析，探讨了感知先行与行动先行的控制顺序对稳定性的影响。

**💡 创新点**

创新点在于①将熵视为可被闭环自适应调节的控制量，②采用可敏感性峰值确定调节阈值并绘制相位图；③揭示控制顺序可导致稳态阈值显著偏移，强调时间结构在自适应调节中的重要性。

**🔧 技术方法**

技术上采用量子‑类密度矩阵表示、von Neumann 熵与纯度计算、可自适应增益控制、闭环反馈、可敏感性和相位图分析，并在 Java 环境下实现完整仿真与可视化。

**📊 数据集**

研究未使用真实数据集，而是通过固定种子、固定时间步长与噪声幅度的随机过程进行大规模仿真实验。

**📈 对比分析**

通过比较感知先行（PF）与行动先行（AF）两种控制顺序，在低噪声与高噪声条件下评估熵、相干隙及可敏感性指标，结果显示 PF 方案在相同噪声下能以更低的调节增益实现稳定，AF 方案需更高增益才能保持相同的稳定性。

**⚠️ 局限性**

局限性包括仅研究单一智能体、有限仿真时间、未引入任务语义或奖励驱动学习、密度矩阵仅作为工具使用、缺乏多智能体交互与长期动态分析。

---

## 58. Ultrafast Sampling-based Kinodynamic Planning via Differential Flatness

**arXiv ID:** 2603.16059 | [PDF](https://arxiv.org/pdf/2603.16059v1)

**作者:** Thai Duong `[一作]` (Rice University), Lydia E. Kavraki `[通讯]` (Rice University)

**通讯引用:** 24211 | [OpenAlex ID](https://openalex.org/A5067205988)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了AkinoPDF，一种利用微分平坦性在平面输出空间中以闭式解解决两点边值问题与动力学传播的超快采样型运动规划方法，能够在微秒到毫秒级别生成满足动力学约束的可执行轨迹；

**💡 创新点**

创新点在于：①将动力学约束映射到平面输出空间，使得边值问题和动力学传播可用多项式闭式求解；②结合单指令多数据(SIMD)细粒度并行，实现在常规CPU上极快的前向动力学和碰撞检测；③可直接嵌入任何采样型规划器（如RRT、SST*），并保持精确的动态可行性；

**🔧 技术方法**

主要技术包括：微分平坦性理论、线性二次最小时间(LQMT)闭式解、RRT/SST*的FlatExtend子例程、AVX2等SIMD向量化碰撞检测、闭式动力学传播与轨迹简化；

**📊 数据集**

使用的数据集包括DynoBench基准（单轮、二维四旋翼、三维四旋翼）以及MotionBenchMaker的七种七自由度Franka Panda机器人场景（bookshelf thin/tall/small、cage、box、table‑under‑pick、table‑pick），并在真实UR5机器人上进行了实验；

**📈 对比分析**

与基线方法（iDb‑A*、SST*、VAMP+TOPP‑RA）在轨迹长度、规划时间和碰撞风险等指标进行对比；结果显示AkinoPDF在所有测试环境下规划时间均为毫秒级（平均约3.5 ms，实时案例仅90 μs），比基线快数倍，且碰撞风险显著降低（基线约30%）；

**⚠️ 局限性**

局限性在于仅适用于微分平坦的机器人系统，无法直接扩展到非平坦系统；此外，当前方法不提供原始状态空间下的最优性证明，也未结合后期优化步骤进一步改进轨迹质量。

---

## 59. Why We Need to Destroy the Illusion of Speaking to A Human: Critical Reflections On Ethics at the Front-End for LLMs

**arXiv ID:** 2603.16633 | [PDF](https://arxiv.org/pdf/2603.16633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 60. Biased Compression in Gradient Coding for Distributed Learning

**arXiv ID:** 2603.16353 | [PDF](https://arxiv.org/pdf/2603.16353v1)

**作者:** Chengxi Li `[一作]` (KTH Royal Institute of Technology), Mikael Skoglund `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 8847 | [OpenAlex ID](https://openalex.org/A5041348422)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了COCO-EF方法，将梯度编码、偏置压缩与误差反馈相结合，以在存在straggler和通信瓶颈的分布式学习场景中实现高效模型训练。

**💡 创新点**

创新点在于首次系统性证明偏置压缩配合误差反馈能够在梯度编码框架下显著提升学习性能，并给出了严格的收敛分析，突破了以往只使用无偏压缩或无误差反馈的局限。

**🔧 技术方法**

采用的技术包括：梯度编码（基于对称平衡的数据分配）、偏置压缩函数（如分组符号量化、Top‑K稀疏化）、误差反馈机制、对光滑损失的L‑smooth性假设以及对异质性和压缩误差的数学刻画。

**📊 数据集**

实验数据集包括合成线性回归数据和MNIST图像分类数据；前者用于验证理论收敛，后者用于展示在实际神经网络训练中的效果。

**📈 对比分析**

与多种基线方法（无偏压缩、无偏+梯度差压缩、无误差反馈版本）在相同通信开销下进行对比，结果显示COCO-EF在训练损失、测试准确率和收敛速度上均优于基线，尤其在高straggler概率和高压缩率下表现更突出。

**⚠️ 局限性**

局限性包括：理论收敛条件（δ<0.5、q_A<2δ+1/2）在实验中并未严格满足，且对光滑损失函数的假设限制了在高度非凸任务中的适用性；目前仅在100台设备规模上验证，缺乏对更大规模系统的实证；误差反馈会带来额外的本地存储和计算开销。

---

## 61. Compact Optical Single-axis Joint Torque Sensor Using Redundant Photo-Reflectors and Quadratic-Programming Calibration

**arXiv ID:** 2603.16040 | [PDF](https://arxiv.org/pdf/2603.16040v1)

**作者:** Hyun-Bin Kim `[一作]` (Korea Advanced Institute of Science and Technology), Kyung-Soo Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 4908 | [OpenAlex ID](https://openalex.org/A5100334901)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于光电探测器的紧凑型关节扭矩传感器，并通过冗余光探测阵列和二次规划校准方法显著提升低扭矩精度，随后在实际电机上验证了其控制性能和自适应温度补偿。

**💡 创新点**

创新点包括：①使用非接触光电探测器代替传统应变计，降低成本与布线复杂度；②在四方向布置冗余光探测器，利用冗余信息通过二次规划校准抑制噪声与误差；③采用温度舱实验得到的理性拟合模型实现零漂移补偿；④将传感器集成到实际电机前端，展示低扭矩跟踪与谐振响应优势。

**🔧 技术方法**

技术手段：光学传感、弹性梁建模、冗余测量、二次规划（QP）优化校准、温度补偿理性拟合、CAN‑FD 通讯、STM32H7 MCU + 16‑bit ADC、Simulink实时控制、摩擦与间隙实验、步进、正弦与方波跟踪、顺序运动及顺序阻尼（Admittance）控制。

**📊 数据集**

使用的数据集包括：①与ATI MINI85 参考扭矩传感器同步采集的扭矩/力数据；②利用加权重量的杠杆实验数据；③在温度舱中采集不同温度下的零漂移数据；④在电机上进行步进、正弦波和顺序阻尼的控制实验数据。

**📈 对比分析**

比较方法与性能：①与最小二乘法校准相比，QP校准将 3σ 分辨率从 0.048 N·m 提升至 0.0224 N·m（提升 2.14×）；②与仅使用电机电流估计的扭矩控制相比，使用本传感器可将低扭矩 RMS 误差从 0.0566 N·m 降至 0.0026 N·m，低扭矩跟踪误差降低至 0.079×；③在温度补偿后，零漂移 RMS 误差下降 27%，10 N·m 扭矩误差下降 31%；④在高频响应实验中，感知带宽约 5 kHz，结构共振约 3.4 kHz，满足 1 kHz 控制需求。

**⚠️ 局限性**

局限性：①仅针对 z‑轴扭矩，其他轴仍需进一步校准；②需要在每个关节上进行手工校准，制造和调试成本相对较高；③温度补偿依赖实验室温度舱数据，现场工况可能导致漂移；④传感器尺寸仍受电机直径限制，极大关节不易集成；⑤高频信号受限于 ADC 采样和 ADC 上升时间，最大有效带宽受限于 5 kHz；⑥在极端机械冲击或高速运动时，弹性梁材料疲劳可能影响长期稳定性。

---

## 62. POLAR:A Per-User Association Test in Embedding Space

**arXiv ID:** 2603.15950 | [PDF](https://arxiv.org/pdf/2603.15950v1)

**作者:** Pedro Bento `[一作]` (Universidade Federal de Minas Gerais), Wagner MeiraJr `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种名为 POLAR 的方法，用轻量级微调的掩码语言模型嵌入空间对单个用户的词汇关联进行定量检验。

**💡 创新点**

其创新点在于将传统聚合级别的词汇关联检验迁移到个体用户向量层面，并通过私有哈希用户标记生成稀疏用户嵌入，随后投影到词汇轴并进行置换检验与多重校正。

**🔧 技术方法**

技术包括轻量化微调掩码语言模型、生成用户专属 token、投影到预设词汇轴、计算标准化效应大小、蒙特卡罗置换检验及 Benjamini–Hochberg FDR 控制。

**📊 数据集**

实验使用了平衡的机器人–人类 Twitter 基准（fox8‑23）和白人至上极端主义论坛 Stormfront 两个用户标识数据集。

**📈 对比分析**

在无监督情形下，POLAR 在 fox8‑23 上单轴 AUROC 均超过 0.95，加入多轴提升不显著；在 Stormfront 中显示对歧视词汇的显著正向效应，揭示群体内部异质性并可追踪用户随时间的极化趋势。

**⚠️ 局限性**

局限性包括：需要至少两条文本记录的用户；对极短或无文本的用户无法使用；词汇轴的选择可能引入研究者偏见；仅在英语单一平台验证，跨语言或跨平台推广仍待进一步研究。

---

## 63. Federated Learning for Privacy-Preserving Medical AI

**arXiv ID:** 2603.15901 | [PDF](https://arxiv.org/pdf/2603.15901v1)

**作者:** Tin Hoang `[一作]` `[通讯]` (University of Surrey), Tin Hoang (University of Surrey)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文研究了在阿尔茨海默症诊断中应用联邦学习，实现医疗数据的隐私保护与协同建模。

**💡 创新点**

创新点在于提出了符合机构边界的 site-aware 数据划分策略和动态调节隐私参数的自适应本地差分隐私（ALDP）机制。

**🔧 技术方法**

采用了联邦学习框架（FedProx）以及自适应本地差分隐私技术，并对3D MRI图像进行分类训练。

**📊 数据集**

使用的数据集为 Alzheimer’s Disease Neuroimaging Initiative（ADNI）提供的三维 MRI 数据。

**📈 对比分析**

通过与传统固定噪声 Local DP 方法比较，ALDP 在两客户端配置下实现了80.4%的准确率，比固定噪声提升5–7个百分点，且训练更稳定。

**⚠️ 局限性**

局限性包括实验规模有限、对多客户端扩展验证不足，以及对极端隐私预算和实际部署环境的硬件/网络需求缺乏深入探讨。

---

## 64. Speak, Segment, Track, Navigate: An Interactive System for Video-Guided Skull-Base Surgery

**arXiv ID:** 2603.16024 | [PDF](https://arxiv.org/pdf/2603.16024v1)

**作者:** Jecia Z. Y. Mao `[一作]`, Manish Sahu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个语音引导的具身代理框架SCOPE，能够在实时手术视频中执行交互式分割、追踪、注册和姿态估计，并通过自然语言命令驱动工作流程。

**💡 创新点**

将语言推理与视觉基础模型分离，采用零样本VFMs结合实时交互提示，实现低延迟、低硬件需求的全流程视频导航；并通过流式记忆与捕获传播实现工具分割不中断。

**🔧 技术方法**

语音识别+LLM（语言规划）、GroundingDINO+SAM/GSAM+CUTIE进行分割与传播、DepthAnything v2与注册模型做深度缩放、PCA与PnP求姿、基于光学跟踪的对标评估、流式缓存和置信门控跟踪。

**📊 数据集**

主要使用外科实验室的skull‑base ex vivo 视频数据，配合商用光学跟踪记录；实验基于三次实验，未使用公开大规模数据集。

**📈 对比分析**

与商用光学跟踪系统对比，平均工具尖端位置误差2.83±1.64 mm、roll/pitch误差≈0.16°；工具分割与解剖注册约两分钟完成，整体流程约1分48秒，显示出与光学系统相当的空间精度并显著简化硬件与设置。

**⚠️ 局限性**

仍在实验室验证，缺乏真实临床外科评估；在快速运动或强遮挡时姿态误差增大；依赖外部摄像头分辨率和深度网络的尺度不确定；以及对不同解剖结构的鲁棒性未全面评估。

---

## 65. A Scoping Review of AI-Driven Digital Interventions in Mental Health Care: Mapping Applications Across Screening, Support, Monitoring, Prevention, and Clinical Education

**arXiv ID:** 2603.16204 | [PDF](https://arxiv.org/pdf/2603.16204v1)

**作者:** Yang Ni `[一作]` (Columbia University), Fanli Jia `[通讯]` (Seton Hall University)

**通讯引用:** 1493 | [OpenAlex ID](https://openalex.org/A5049775460)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并映射了36项实证研究，阐述人工智能在心理健康护理的五个阶段（筛查、治疗、随访、临床教育和预防）中的应用与效果。

**💡 创新点**

提出了四柱框架，系统化地将不同AI技术与临床阶段对应，并强调AI在辅助而非替代临床治疗中的潜力与挑战。

**🔧 技术方法**

涵盖聊天机器人、自然语言处理、机器学习/深度学习模型、大语言模型等多种技术，并讨论了其在诊断、预测、监测与支持中的实现方式。

**📊 数据集**

使用多样化的公开研究数据集与原始临床/自评数据，未统一统一数据集，因每篇研究采用不同数据来源和采集方式。

**📈 对比分析**

采用叙事性综述与表格汇总对比，未进行元分析；报告的性能包括等待时间缩短、参与度提升、症状追踪精准等，但总体缺乏统一量化指标。

**⚠️ 局限性**

局限性：样本多为试点/单中心、缺乏风险评估与定量合成；语言只限英文、时间截止至2024年初，未涵盖最新LLM；缺少真实世界实施与公平性评估。

---

## 66. EFF-Grasp: Energy-Field Flow Matching for Physics-Aware Dexterous Grasp Generation

**arXiv ID:** 2603.16151 | [PDF](https://arxiv.org/pdf/2603.16151v1)

**作者:** Yukun Zhao `[一作]` (Shandong University), Haoliang Sun `[通讯]` (Shandong University)

**通讯引用:** 578 | [OpenAlex ID](https://openalex.org/A5038081609)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于流匹配（Flow Matching）的全新抓取生成框架EFF‑Grasp，利用确定性ODE轨迹实现高效抓取采样，并通过训练无关的能量引导实现物理可行的抓取姿态；

**💡 创新点**

创新点包括：1) 将抓取生成从传统的随机SDE迁移到确定性ODE，显著减少采样步数；2) 设计了三种显式物理能量场（外部穿透、表面吸附、自我穿透）并在推理阶段使用局部Monte‑Carlo近似实现能量引导，无需额外训练；3) 结合流匹配的直线轨迹实现平滑稳定的采样；

**🔧 技术方法**

技术手段：流匹配框架、确定性ODE求解、U‑Net基网络、局部Monte‑Carlo估计、能量引导（ERF、SPF、SRF）

**📊 数据集**

使用了五个抓取基准数据集：DexGraspNet、UniDexGrasp、DexGRAB、RealDex、MultiDex；

**📈 对比分析**

与多种基线（GraspTTA、UniDexGrasp、SceneDiffuser、UGG、DGA）对比，EFF‑Grasp在成功率、最大穿透深度和采样效率上均取得领先，成功率提升最高可达+13.6%，采样步数可从100降至10仍保持较好性能；

**⚠️ 局限性**

局限性：1) 相比SDE基方法，产生的抓取多样性略低；2) 能量场设计需手工设定，可能在极端形状或动态环境下不足；3) 仍依赖事先训练好的U‑Net模型，对新的硬件或手掌形状需要重新训练；

---

## 67. CD-FKD: Cross-Domain Feature Knowledge Distillation for Robust Single-Domain Generalization in Object Detection

**arXiv ID:** 2603.16439 | [PDF](https://arxiv.org/pdf/2603.16439v1)

**作者:** Junseok Lee `[一作]` (LG Electronics), Kyoobin Lee `[通讯]` (Gwangju Institute of Science and Technology)

**通讯引用:** 1987 | [OpenAlex ID](https://openalex.org/A5031483606)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过构建教师网络与学生网络的跨域知识蒸馏框架，在教师网络上使用原始高分辨率源域图像，学生网络使用降采样与多种腐蚀后的图像，结合全局特征蒸馏和实例级特征蒸馏，提升单源域泛化的目标检测性能。

**💡 创新点**

创新点在于：①跨域蒸馏设置，让学生在对抗性多样化输入下学习教师的全局与实例特征；②同时使用全局特征蒸馏与实例级蒸馏，兼顾图像整体语境与目标细节；③通过降采样与腐蚀双重扰动增强模型对尺寸与噪声变化的鲁棒性。

**🔧 技术方法**

采用 Faster R‑CNN + ResNet101‑FPN 作为基础检测器，使用知识蒸馏（cosine 相似度损失）实现全局与实例特征对齐，整体训练使用 SGD，包含多尺度/腐蚀数据增强。

**📊 数据集**

在多天气场景单源域泛化基准上进行实验，源域为 Daytime‑Clear，目标域为 Night‑Clear、Dusk‑Rainy、Night‑Rainy、Daytime‑Foggy 共四个天气条件。

**📈 对比分析**

与现有单域泛化方法（如 DivAlign、S‑DGOD、CLIP‑Gap 等）对比，CD‑FKD 在四个目标域的平均 mAP 达到 38.3%，比 Faster R‑CNN 提升 11.1%，比 DivAlign 高 2.8%，同时在源域性能亦提升，显示出更强的跨域适应能力。

**⚠️ 局限性**

局限性包括：①只针对单源域；②需要训练两个相同结构的网络，计算成本较高；③数据增强仅使用与目标无关的常规腐蚀，可能对某些特定目标域的细节适应不足；④目前仅验证于城市交通场景，泛化到其他场景仍需进一步验证。

---

## 68. Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context

**arXiv ID:** 2603.15653 | [PDF](https://arxiv.org/pdf/2603.15653v1)

**作者:** Keivan Alizadeh `[一作]` (Apple), Mehrdad Farajtabar `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Self-Reflective Program Search (SRLM)，通过自我反思的方式在程序化上下文交互中选取最佳推理轨迹，以解决长上下文推理问题；

**💡 创新点**

创新点在于将自一致性、口头自信度、推理轨迹长度三种不确定性信号融合，用模型内部自我评估来驱动程序搜索，证明递归并非提升性能的关键；

**🔧 技术方法**

使用的技术包括程序化上下文交互框架、REPL执行环境、采样自一致性、口头自信度提示、推理长度作为行为不确定性信号，并在Qwen3-Coder-480B和GPT-5等大模型上实现；

**📊 数据集**

实验数据集涵盖LongBench-v2（CodeQA及多域）、BrowseComp+、OOLONG等长上下文基准，并对不同上下文长度进行扩展评测；

**📈 对比分析**

与基线模型（基线LLM、CodeAct、Summary agent、递归/非递归RLM）对比，SRLM在所有基准上平均提升约22%，在短/长上下文均表现更稳健，尤其在语义密集型任务上优于RLM；

**⚠️ 局限性**

局限性在于仅利用了三种简单的自我反思信号，缺乏更丰富的内部状态挖掘与决策机制，对极端长上下文或非结构化任务的泛化仍待验证；

---

## 69. State-Dependent Safety Failures in Multi-Turn Language Model Interaction

**arXiv ID:** 2603.15684 | [PDF](https://arxiv.org/pdf/2603.15684v1)

**作者:** Pengcheng Li `[一作]` (University of Science and Technology of China), Wenbo Zhou `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3053 | [OpenAlex ID](https://openalex.org/A5067315721)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出STAR（State‑Oriented Role‑playing）框架，从状态空间视角系统地诊断多轮对话中的安全失败；

**💡 创新点**

创新点在于将对话历史视为可控的状态转移算子，分离状态初始化与演化两阶段，揭示安全边界随交互轨迹动态演化的机制；

**🔧 技术方法**

使用语义保持软化、查询感知角色生成、结构化模板、反馈感知历史干预、轨迹控制等技术，配合辅助语言模型构造对话；

**📊 数据集**

主要使用HarmBench和JailbreakBench两套安全基准，包含多种违法、仇恨、恶意生成等安全关键指令；

**📈 对比分析**

与多种单轮与多轮基线（GCG、PAIR、CodeAttack、RACE、CoA、Crescendo、X‑Teaming、ActorAttack）对比，STAR在GPT‑4o、Claude 3.5、Gemini 2.0‑Flash等前沿模型上显著提升安全失败率（SFR），并在token成本上更具效率；

**⚠️ 局限性**

局限在于依赖辅助模型生成角色与软化表达，未对模型内部训练策略做干预；实验范围局限于公开基准，缺少真实场景验证；对极端对抗性或动态策略的适用性尚待进一步评估。

---

## 70. Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling

**arXiv ID:** 2603.16797 | [PDF](https://arxiv.org/pdf/2603.16797v1)

**作者:** Christian Belardi `[一作]` (Cornell University), Carla P. Gomes `[通讯]` (Cornell University)

**通讯引用:** 12173 | [OpenAlex ID](https://openalex.org/A5069030030)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出在plug-and-play扩散模型中使用自适应动量估计（Adam）来稳定噪声较大的似然梯度，提升生成质量。

**💡 创新点**

创新点在于将优化算法中的自适应动量方法直接应用于采样过程中的指导梯度，简单却能显著提升多种方法的性能。

**🔧 技术方法**

采用的技术包括扩散模型、先验/似然分数分解、Adam动量估计以及对DPS、CG等指导方法的改进。

**📊 数据集**

实验数据集涵盖ImageNet、CIFAR-10以及Cats数据集，进行超分辨、去模糊、修复等任务。

**📈 对比分析**

与传统DPS、CG、UGD、TFG等方法对比，AdamDPS/AdamCG在多项任务上取得了更低的LPIPS、FID和更高的分类准确率，尤其在高难度任务中表现更为突出。

**⚠️ 局限性**

局限性包括对极端噪声或非常稀疏条件信息的处理仍有限，且仅在现有plug-and-play框架下验证，未探究与其它高级分数近似方法的结合效果。

---

## 71. HMAR: Hierarchical Modality-Aware Expert and Dynamic Routing Medical Image Retrieval Architecture

**arXiv ID:** 2603.16679 | [PDF](https://arxiv.org/pdf/2603.16679v1)

**作者:** Aojie Yuan `[一作]` `[通讯]` (Shanghai Jiao Tong University), Aojie Yuan (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于Mixture-of-Experts的医疗影像检索框架HMAR，实现全局与局部检索统一。

**💡 创新点**

创新点包括双专家架构、两阶段对比学习实现位置无关局部特征、滑动窗口局部匹配以及KAN哈希编码。

**🔧 技术方法**

使用ResNet-50骨干、MoE门控、两阶段对比学习、KAN网络、滑动窗口匹配和哈希编码。

**📊 数据集**

使用RadioImageNet-CT数据集（29,903张CT图像，16种临床模式）。

**📈 对比分析**

与ACIR等现有方法对比，HMAR在64位和128位哈希码上分别提升0.7%和1.1%的mAP，整体性能显著提高。

**⚠️ 局限性**

局限性包括仅评估CT单一模态、未验证多模态性能、专家数量有限等。

---

## 72. An assessment of data-centric methods for label noise identification in remote sensing data sets

**arXiv ID:** 2603.16835 | [PDF](https://arxiv.org/pdf/2603.16835v1)

**作者:** Felix Kröber `[一作]` (Forschungszentrum Jülich), Ribana Roscher `[通讯]` (University of Bonn)

**通讯引用:** 3573 | [OpenAlex ID](https://openalex.org/A5043510754)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估了三种数据中心标签噪声识别方法（TopoFilter、AuM、CL）在遥感场景分类中的效果，探讨其噪声过滤与任务性能提升的关系；

**💡 创新点**

首次在遥感基准数据集上系统比较并分解数据中心噪声方法的识别、估计与性能三大维度，为方法选择提供经验准则；

**🔧 技术方法**

采用TopoFilter（拓扑过滤）、AuM（平均边距）、CL（置信联合矩阵）以及ResNet-18训练、交叉熵损失、Adam优化器，结合人工注入的均匀与非均匀噪声；

**📊 数据集**

UCMerced与EuroSAT两大人工标注遥感数据集，后者规模略小、包含10%验证/测试集；

**📈 对比分析**

通过精确率、召回率、剩余噪声、Δ噪声、SMAPE及分类准确率等指标进行比较；TopoFilter在多数场景下表现最佳，但在高噪声/高稀疏性时效果下降；CL在噪声估计上最稳健；整体可提升10–15%准确率，极端噪声下提升有限；

**⚠️ 局限性**

仅在两个平衡、规模不大的数据集上验证；噪声模型仅覆盖均匀与非均匀，未涵盖实例/特征依赖噪声；高噪声下易误删过多样本导致性能下降；真实遥感数据集的泛化需进一步研究。

---

## 73. Beyond Accuracy: Evaluating Forecasting Models by Multi-Echelon Inventory Cost

**arXiv ID:** 2603.16815 | [PDF](https://arxiv.org/pdf/2603.16815v1)

**作者:** Swata Marik `[一作]` (University of Calcutta), Garga Chatterjee `[通讯]` (Indian Statistical Institute Kolkata)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个统一的数字化预测‑库存优化流水线，将七类模型集成到统一特征/训练框架，并通过新闻供应商模拟评估其在单层和双层库存系统下的运营影响。

**💡 创新点**

创新之处在于将预测准确性直接与多层库存成本和服务率挂钩，提供统一的实验协议，并展示深度学习模型在真实零售数据中对库存成本的显著降低。

**🔧 技术方法**

使用统计模型（ARIMA、Holt‑Winters）、机器学习集成（GBR、XGBoost）和深度学习序列模型（LSTM、Temporal CNN）进行预测，并在新闻供应商和双层DC‑Store模拟器中评估。

**📊 数据集**

采用M5 Walmart零售销售数据中的CA_FOODS_1子集进行实验。

**📈 对比分析**

通过滚动测试与验证拆分，将预测结果映射为订单量，再用新闻供应商成本和填充率指标进行比较，结果显示Temporal CNN和LSTM在成本上比传统基线低约18%且填充率提高约10个百分点。

**⚠️ 局限性**

局限在于仅评估单一商品类别和有限时间窗口，采用点预测的订单策略未考虑价格弹性、促销影响或概率预测，且多层网络仅为简单DC‑Store模型。

---

## 74. Beyond Cybathlon: On-demand Quadrupedal Assistance for People with Limited Mobility

**arXiv ID:** 2603.16772 | [PDF](https://arxiv.org/pdf/2603.16772v1)

**作者:** Carmen Scheidemann `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**通讯引用:** 20806 | [OpenAlex ID](https://openalex.org/A5044258783)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了一种基于独立四足机器人 ANYmal D 与 DynaArm 的“按需”辅助平台，支持盲人或四肢受限使用者在家中与竞赛场景中完成日常抓取、放置、开关等多种任务。

**💡 创新点**

创新点包括：① 结合共享自主性（半自主 + 人机遥控）实现按需协作；② 采用专用口舌操控器 QuadStick，降低操作门槛；③ 为赛道设计两机器人协作与关键点导航的全新自动化栈；④ 在家庭环境中集成 LLM + 语音交互与语义地图实现自然语言目标定位。

**🔧 技术方法**

核心技术包括：运动学与 MPC 全身控制、基于强化学习的步态规划、视觉 SLAM + 关键点定位、基于 Waverider 的路径跟踪、Whisper 语音识别、ChatGPT LLM、RecognizeAnything 目标识别、TagMap 语义地图、两机器人协作策略（红绿灯避碰）和多模式控制映射。

**📊 数据集**

主要数据来源为内部实验数据：赛道任务的关键点地图、家庭场景的手工标注物体语义标签，未使用公开公共数据集；对比中采用了 Cybathlon 2024 官方排名和团队时限，以及 Padmanabha 等人公开的 HAT 系统主观问卷。

**📈 对比分析**

在 Cybathlon 2024 赛道上，完成 8/10 任务，时间 8 min 12 s，排名第三；在家庭实验中，多任务平均成功率 80–100%，任务完成时间从 75 s（灯光）到 253 s（热食）不等。与其他团队相比，自动化预定位显著降低了操作员等待时间（从 25 % 降至 7 %），主观问卷显示“精神负荷”和“努力”显著下降。

**⚠️ 局限性**

局限性包括：仅在单一受限使用者上评估，泛化性待验证；自动化主要集中在导航与预定位，复杂抓取仍需手动控制；在高动态环境或物体变化频繁时，语义地图与 LLM 识别误差仍影响任务完成；双机器人协作在空间受限赛道中需进一步优化碰撞规避与时间同步。

---

## 75. ODIN-Based CPU-GPU Architecture with Replay-Driven Simulation and Emulation

**arXiv ID:** 2603.16812 | [PDF](https://arxiv.org/pdf/2603.16812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 76. Learning to Present: Inverse Specification Rewards for Agentic Slide Generation

**arXiv ID:** 2603.16839 | [PDF](https://arxiv.org/pdf/2603.16839v1)

**作者:** Karthik Ragunath Ananda Kumar `[一作]` (University of Texas at Dallas), Subrahmanyam Arunachalam `[通讯]` (Texas A&M University)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5102449659)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过强化学习训练大语言模型（LLM）代理，利用14种工具完成从研究到最终化的幻灯片生成流程；

**💡 创新点**

提出多维度奖励体系，其中包含新颖的“逆规格奖励”——用LLM从生成的幻灯片中重构原始任务规范，衡量整体连贯性；

**🔧 技术方法**

使用OpenEnv兼容的RL环境、GRPO（Group Relative Policy Optimization）和LoRA低秩适配在Qwen2.5-Coder-7B上进行参数高效微调；

**📊 数据集**

基准数据集为48份多样化商业演示简报（共288个完整回合），并开放SlideRL数据集供研究；

**📈 对比分析**

与Claude Opus 4.6、Claude Sonnet 4.6、Llama 4 Scout、GPT OSS 120B等六种模型在相同环境下对比，微调后的7B模型获得整体质量0.724，约占Claude Opus 91.2%，并在代码规则、渲染质量等指标上显著提升；

**⚠️ 局限性**

局限包括奖励评估成本高、易出现奖励劫持/模式崩溃、对逆规格奖励的领域适配性有限、K=2导致优势估计信息不足等。

---

## 77. Finding Common Ground in a Sea of Alternatives

**arXiv ID:** 2603.16751 | [PDF](https://arxiv.org/pdf/2603.16751v1)

**作者:** Jay Chooi `[一作]` (Harvard University), Shirley Zhang `[通讯]` (Harvard University)

**通讯引用:** 22139 | [OpenAlex ID](https://openalex.org/A5100358804)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在无限备选方案空间中，通过采样算法找到满足比例否决核心（ε‑PVC）的共识性陈述；

**💡 创新点**

主要创新包括：①提出ε‑PVC在无限备选空间下的正式定义；②证明采样量Θ(1/ε²)是最优的下界；③首次将该理论应用于生成式社交选择，并与传统投票规则进行对比；

**🔧 技术方法**

采用生成式查询+最小查询的采样算法、流网络求解ε‑PVC、Borda、Schulze、IRV等传统投票规则，以及多种基于LLM的生成式投票策略；

**📊 数据集**

使用基于OpenAI GPT生成的100名合成角色的100条主题陈述及其对应的100名选民偏好数据；补充Preflib真实偏好数据集做进一步实验；

**📈 对比分析**

通过“critical ε”指标衡量共识程度，实验显示VBC和LLM生成规则在ε接近0时优于Borda、Schulze、IRV和多数投票；Borda在真实数据中表现最佳；LLM规则在偏向群体时性能明显下降；

**⚠️ 局限性**

局限性：①ε‑PVC的有效性依赖于分布假设，对分布选择敏感；②采样算法对极端偏好分布的鲁棒性有限；③当ε=0时PVC可能为空，且在ε>0时多个共识选项难以区分，需进一步研究多重选择和评分结合方法。

---

## 78. A federated learning framework with knowledge graph and temporal transformer for early sepsis prediction in multi-center ICUs

**arXiv ID:** 2603.15651 | [PDF](https://arxiv.org/pdf/2603.15651v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 79. Computing the connected components of real algebraic curves

**arXiv ID:** 2603.16283 | [PDF](https://arxiv.org/pdf/2603.16283v1)

**作者:** Elisabetta Rocchi `[一作]` (Sorbonne Universite), Mohab Safey El Din `[通讯]` (Sorbonne Universite)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种新算法，用于计算实代数曲线的连通分支的半代数描述。

**💡 创新点**

该算法的复杂度低于已知的计算与研究中的实空间曲线同胚的图的最佳复杂度。

**🔧 技术方法**

使用了基于多项式约束的布尔公式的算法，结合了对投影的拓扑分析。

**📊 数据集**

使用了生成一个根理想的多项式方程组的解集来定义实代数曲线。

**📈 对比分析**

与现有方法相比，该算法在计算连通分支的复杂度上表现更优，尤其是在三维空间中，复杂度为d^14(h+τ+1)+log(1/ϵ)d^13，优于已知的同胚图计算方法。

**⚠️ 局限性**

算法的局限性在于需要假设输入多项式满足某些泛化性质，且在处理某些特定情况时可能会遇到复杂度问题。

---

## 80. DynamicGate MLP Conditional Computation via Learned Structural Dropout and Input Dependent Gating for Functional Plasticity

**arXiv ID:** 2603.16367 | [PDF](https://arxiv.org/pdf/2603.16367v1)

**作者:** Yong Il Choi `[一作]` `[通讯]` (Sorynorydotcom Co., Ltd./AI Open Research Lab), Yong Il Choi (Sorynorydotcom Co., Ltd./AI Open Research Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出DynamicGate-MLP框架，将学习到的门控用于训练和推理中实现条件计算，并通过门控概率和阈值化实现输入相关的动态稀疏化。

**💡 创新点**

把Dropout的随机去活化转化为可学习、可输入依赖的门控；同时结合门控概率期望惩罚实现计算预算控制，并可与RigL动态稀疏重连结合，形成功能与结构双重稀疏。

**🔧 技术方法**

门控网络（GateNet）、Sigmoid门概率、硬阈值化、STE反向传播、期望门使用惩罚、Top‑k门、RigL动态稀疏训练、相对MAC计算等。

**📊 数据集**

MNIST、CIFAR‑10、Tiny ImageNet、Speech Commands、PBMC3k等多种任务，包括图像、语音、基因表达。

**📈 对比分析**

与Baseline、Dropout、Pruned、RigL、Switch‑MoE等进行对比，动态门在保持或略降精度的同时，计算代理(MAC)下降约20‑80%，在某些任务上精度甚至提升，说明可通过输入依赖门控有效降低计算成本。

**⚠️ 局限性**

计算代理不等同于实际延迟，缺乏稀疏内核、门控开销、内存/启动开销等，导致在GPU/CPU上实际加速有限；门控和重连参数敏感，易出现门坍塌；实验仅在小型MLP，未验证Transformer或大规模模型。

---

## 81. Nodule-Aligned Latent Space Learning with LLM-Driven Multimodal Diffusion for Lung Nodule Progression Prediction

**arXiv ID:** 2603.15932 | [PDF](https://arxiv.org/pdf/2603.15932v1)

**作者:** James Song `[一作]` (University of Michigan), Liyue Shen `[通讯]` (University of Michigan)

**通讯引用:** 6366 | [OpenAlex ID](https://openalex.org/A5072483985)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用基于LLM的控制和结节对齐潜在空间，预测并生成一年的肺结节CT随访图像，从而实现早期肺癌风险评估。

**💡 创新点**

创新点在于：①构建与结节属性变化高度相关的对齐潜在空间，使潜在距离直接映射到临床指标；②采用可微软提示的医学专用LLM（MedGemma）生成报告嵌入，精细调控生成过程；③将多模态（影像+EHR）信息融合进潜在扩散模型，提升生成的临床可解释性。

**🔧 技术方法**

使用的技术包括：变分自编码器（VAE）进行潜在编码，潜在扩散模型（Latent Diffusion Model）进行随访图像生成，LLM软提示微调（MedGemma）进行条件编码，以及对齐损失、预测损失、LPIPS等多种损失函数进行联合训练。

**📊 数据集**

数据集为美国国家肺筛查试验（National Lung Screening Trial, NLST）中挑选的1226例含至少一年随访的肺结节影像，训练/验证共1121对，测试集450对。

**📈 对比分析**

与StableDiffusion‑1.5、Flux、McWGAN、CorrFlowNet、DDL‑CXR、ImageFlowNet等基线方法对比；NAMD在AUROC上取得0.805±0.018、AUPRC 0.346±0.028，明显优于基线（例如仅基线影像 AUROC 0.742）并接近真实随访影像的表现（AUROC 0.819）。在图像质量指标上，NAMD的FID和LPIPS均与最优基线相当。

**⚠️ 局限性**

局限性：①数据量有限，模型对罕见或极端病例的泛化尚未充分验证；②生成图像在像素级重建上仍存在偏差，主要集中在结节边缘；③LLM条件生成依赖于预先构建的报告格式，若EHR信息不完整或不规范，模型性能可能下降；④缺乏针对临床真实诊断流程的验证，需要进一步与专门的结节诊断模型结合。

---

## 82. FEAT: A Linear-Complexity Foundation Model for Extremely Large Structured Data

**arXiv ID:** 2603.16513 | [PDF](https://arxiv.org/pdf/2603.16513v1)

**作者:** Zhenghang Song `[一作]` (Zhejiang University), Tianyi Li `[通讯]` (Aalborg University)

**通讯引用:** 1366 | [OpenAlex ID](https://openalex.org/A5100460598)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FEAT，一种线性复杂度的结构化数据基础模型，支持零样本推理并能在百万级样本规模下实现高效推理。

**💡 创新点**

创新点在于双轴编码架构（AFBM+Conv‑GLA）解决了线性模型的“线性陷阱”和“因果掩码缺失”，实现 𝒪(N) 交叉样本建模；同时引入混合真实与合成 SCM 预训练和 Huber 损失以提升对重尾分布的稳健性。

**🔧 技术方法**

使用了线性状态空间模型（AFBM）、卷积门控线性注意力（Conv‑GLA）、动态列身份编码（S‑DFE）、Huber 加权损失、多任务动态平衡训练和混合 SCM 生成等技术。

**📊 数据集**

使用 11 个真实世界表格数据集，包括 TabPFN、Tabzilla、TALENT、TabArena、LimiX 预选集以及 GI Benchmark 等。

**📈 对比分析**

与 LimiX、TabICL v2、TabPFN 2.5、AutoGluon、CatBoost、LightGBM、XGBoost 等基线在 11 个分类和回归任务上对比，FEAT 在大规模上下文推理时速度提升至 40×，且零样本性能与全注意力模型持平或略优。

**⚠️ 局限性**

局限在于对极端稀疏或多模态结构化数据的适用性尚未验证，且在某些高结构化回归任务上略低于全注意力模型，需要进一步评估工业部署中的资源消耗和适配性。

---

## 83. ASDA: Automated Skill Distillation and Adaptation for Financial Reasoning

**arXiv ID:** 2603.16112 | [PDF](https://arxiv.org/pdf/2603.16112v1)

**作者:** Tik Yu Yim `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22318 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了ASDA框架，自动从错误分析生成可执行的金融领域技能文件并在推理时注入，以提升LLM的金融推理性能。

**💡 创新点**

通过教师-学生结构进行无权重更新的错误驱动技能蒸馏，实现模型特定的可审计、可版本化技能库，显著优于传统训练免费方法。

**🔧 技术方法**

采用教师模型对学生失败进行结构化注释，聚类生成技能文件；在推理时使用选择器动态注入技能；并通过双阶段迭代细化覆盖和安全性。

**📊 数据集**

以FAMMA-Basic文本版为主，包含约1945道金融问题（分算术与非算术），并对训练集进行英语过滤和拆分。

**📈 对比分析**

与GEPA、ACE等训练免费基线比较，Haiku 3.5在算术任务从基线41%提升至58.33%（+17.33），非算术提升至55.16%（+5.95），效果明显优于基线。

**⚠️ 局限性**

仅在FAMMA与Claude系列验证；技能对不同模型不具可迁移性；错误聚类质量和OCR噪声可能限制泛化与回归风险。

---

## 84. Sparse but not Simpler: A Multi-Level Interpretability Analysis of Vision Transformers

**arXiv ID:** 2603.15919 | [PDF](https://arxiv.org/pdf/2603.15919v1)

**作者:** Siyu Zhang `[一作]` `[通讯]` (University of Texas at Austin), Siyu Zhang (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对稀疏Vision Transformer（使用Wanda剪枝后的DeiT‑III B/16）与完整模型进行多层级可解释性分析；

**💡 创新点**

提出IMPACT框架，系统比较稠密与稀疏模型在神经元、层级、回路与模型级别的可解释性；

**🔧 技术方法**

采用Wanda剪枝、BatchTopK稀疏自编码器、可学习节点掩码提取回路以及Chefer等Transformer归因方法；

**📊 数据集**

ImageNet‑1K作为评估数据集；

**📈 对比分析**

与稠密模型在准确率、回路边缘数、节点占比、神经元/特征选择性、归因AUC等指标对比。结果显示70%稀疏模型保持96%稠密准确率，回路边缘数减至2.5倍，节点占比上升；神经元与特征的选择性、熵等指标基本无提升；归因指标在轻度稀疏时略有提升，随着稀疏度上升则趋于不变；

**⚠️ 局限性**

主要局限：仅使用后置剪枝模型而非从零训练稀疏网络；评估指标为间接代理，缺乏真实概念标签；实验仅限于监督ViT与ImageNet，可能不适用于自监督或多模态模型。

---

## 85. Looking for (Genomic) Needles in a Haystack: Sparsity-Driven Search for Identifying Correlated Genetic Mutations in Cancer

**arXiv ID:** 2603.16721 | [PDF](https://arxiv.org/pdf/2603.16721v1)

**作者:** Ritvik Prabhu `[一作]` (Virginia Tech), Mohamed Wahib `[通讯]` (RIKEN Center for Computational Science)

**通讯引用:** 907 | [OpenAlex ID](https://openalex.org/A5002208999)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于稀疏性的 Pruned Depth-First Search (P-DFS) 框架，在多基因突变组合搜索中利用稀疏性剪枝，显著降低搜索空间并保持对癌症相关基因组合的准确发现。

**💡 创新点**

创新点包括：① 采用稀疏矩阵预排序和运行中的位集合交集实现提前剪枝；② 将深度优先搜索与加权集合覆盖（WSC）相结合；③ 设计层次化 MPI 工作分配与无障碍终止，提升大规模并行效率。

**🔧 技术方法**

使用技术：位集合运算、稀疏矩阵预排序、深度优先搜索+剪枝、分层 MPI 调度、工作抢夺、无障碍终止、分布式集覆盖、加速集合覆盖评分。

**📊 数据集**

数据集：TCGA 的 somatic mutation MAF 文件，涵盖多个癌种（BLCA、HNSC、BRCA、COAD、KIRC、LUAD、SKCM、GBM、OV、UCEC）等，约 20,000 基因。

**📈 对比分析**

与传统完整枚举 WSC 对比，P-DFS 在 4-hit 组合上搜索空间缩减 90‑98%，速度提升约 183×，在 Fugaku 超算上实现 1.6k–3k 节点的强规模，测试集上保持 0.85–0.98 的敏感性和 0.81–0.99 的特异性。

**⚠️ 局限性**

局限性：对低 hit（如 2-hit）剪枝效果不如高 hit；仍受固定通信/ I/O 影响；对稠密数据剪枝不充分；实现高度依赖 MPI，移植性有限；在极大规模或更高 hit 数时仍可能受限。

---

## 86. SoK: Systematizing Software Artifacts Traceability via Associations, Techniques, and Applications

**arXiv ID:** 2603.16208 | [PDF](https://arxiv.org/pdf/2603.16208v1)

**作者:** Zhifei Chen `[一作]` (Nanjing University of Science and Technology), Wei Song `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 54774 | [OpenAlex ID](https://openalex.org/A5069632856)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对软件可追踪性恢复领域进行系统综述（SoK），构建了全球性可追踪性图谱，梳理了21种技术、22种工件及23种关联关系，并评估了工具和应用场景。

**💡 创新点**

提出了首个全球工件可追踪性图谱、技术决策地图、角色导向的可追踪性框架和多跳链条指南，弥补了先前研究对工件关联、技术评估和工业落地的空白。

**🔧 技术方法**

综述涵盖了 IR、ML、DL、PLM、LLM、PA、DFA 等技术；研究方法主要采用系统检索、分类、图谱构建和专家调查，评估标准包括 Recall、Precision、F1 等常用指标。

**📊 数据集**

使用的主要数据来源为 76 篇研究论文的公开信息、专家调查问卷和访谈数据；在综述中未自行构建或使用统一数据集，而是对已有论文的公开数据集和实验结果进行汇总。

**📈 对比分析**

通过构建技术决策地图和评估框架，比较了不同技术在工件表示、成本（人力/计算）和性能（Recall/Precision/F1）上的差异；结果显示 IR 仍占主导，但资源密集的 DL/PLM/LLM 在语义捕获上更优，然而代码开放率仅 37%，评估指标不统一导致比较困难。

**⚠️ 局限性**

主要限制包括：研究领域碎片化、工件关联覆盖不足、可复现性低（代码/数据开放率低）、工业采纳率极低（95% 学术化）、缺乏统一基准和评估指标，导致技术对比和实际落地受限。

---

## 87. Routing and Control for Marine Oil-Spill Cleanup with a Boom-Towing Vessel Fleet

**arXiv ID:** 2603.16626 | [PDF](https://arxiv.org/pdf/2603.16626v1)

**作者:** Snir Carmeli `[一作]` (Viterbi Faculty of Electrical and Computer Engineering Technion Israel Institute of Technology), Kiril Solovey `[通讯]` (Viterbi Faculty of Electrical and Computer Engineering Technion Israel Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一个多机器人框架，利用拖曳索道的ASV二人组进行海上油污泄漏的封锁与清理，并给出了从油污定位到实际执行的完整路径规划与控制流程。

**💡 创新点**

创新点在于：①将多油污清理建模为加权多代理旅行修复员问题，并通过混合整数线性规划与热启动启发式实现可扩展的高质量路由；②设计了基于反馈线性化的双船索道控制器，并给出了理论稳定性证明；③将路由与跟踪层无缝耦合，形成从规划到执行的一体化系统。

**🔧 技术方法**

使用的技术包括：混合整数线性规划（MILP）+动态规划+局部搜索的启发式；反馈线性化控制器与PID基准对比；A*障碍规避路径生成；Dubins路径参考轨迹；仿真评估。

**📊 数据集**

实验使用合成的多油污场景（25/50/100个油污，1-10个ASV二人组），随机生成障碍和油污风险权重；无公开真实油污数据。

**📈 对比分析**

比较方法：对同一场景分别采用贪心启发式、完整启发式（DP+ILS）、纯MILP、以及MILP+热启动；结果显示MILP+热启动在300s内即可得到最优或近优解；纯MILP在大规模时求解时间长。控制器方面，反馈线性化在平滑曲率与高速下的横向、航向误差均优于PID，误差约降低5-10%。

**⚠️ 局限性**

局限性包括：①忽略了油污扩散、漂移与海况扰动；②假设油污为静止区域，无法利用已清理油污后可通行；③模型仅为二维平面，未考虑升力、波浪、加速度等；④控制器对参数敏感，实际船舶需进一步实验验证。

---

## 88. MessyKitchens: Contact-rich object-level 3D scene reconstruction

**arXiv ID:** 2603.16868 | [PDF](https://arxiv.org/pdf/2603.16868v1)

**作者:** Junaid Ahmed Ansari `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Ivan Laptev `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 32285 | [OpenAlex ID](https://openalex.org/A5087781064)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了MessyKitchens数据集和基于SAM3D的Multi‑Object Decoder (MOD)，实现了单目物体级3D场景重建。

**💡 创新点**

创新点在于提供真实、杂乱且接触丰富的厨房场景数据，显著降低物体间穿透；并通过多物体自注意力与交叉注意力实现全局一致的位姿和尺度优化。

**🔧 技术方法**

技术主要包括基于SAM3D的深度学习框架、Transformer多物体注意力模块、ICP配准与基于法向量的自动配准策略，以及合成训练数据的生成与渲染。

**📊 数据集**

使用的数据集包括新建的MessyKitchens（真实与合成）、GraspNet‑1B、HouseCat6D，并在MessyKitchens‑train（1.8k场景）进行训练。

**📈 对比分析**

实验与PartCrafter、MIDI、SAM3D等方法对比，MOD在对象级IoU和Chamfer距离上均优于基线，在场景级也有显著提升，并在外域数据上保持良好泛化。

**⚠️ 局限性**

局限性包括仅依赖单目输入，对遮挡和纹理细节的恢复受限；MOD仅修正位姿/尺度，几何细节仍需改进；对非厨房类别的泛化能力仍有提升空间。

---

## 89. Trajectory-Optimized Time Reparameterization for Learning-Compatible Reduced-Order Modeling of Stiff Dynamical Systems

**arXiv ID:** 2603.16583 | [PDF](https://arxiv.org/pdf/2603.16583v1)

**作者:** Joe Standridge `[一作]` (Texas A&M University), Paul Cizmas `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了通过轨迹优化时间重参数化（TOTR）来缓解神经ODE在刚性动力学系统中的学习难题，并在三种基准系统上评估效果。

**💡 创新点**

提出将时间重参数化视为弧长坐标下的优化问题，通过最小化加速度来直接控制时钟平滑性，解决了传统方法对积分器或极值依赖导致的时钟不稳健性。

**🔧 技术方法**

使用神经ODE、弧长变换、速度函数优化、正向Euler积分以及指数激活保证时钟单调。

**📊 数据集**

采用了参数化刚性线性系统、Van der Pol振荡器和HIRES化学动力学模型的高精度仿真数据。

**📈 对比分析**

将TOTR与基于求解器的时间重参数化和基于极值的重参数化在相同训练设置下比较，使用τ-MSE和重参数化不变的MSIE评估；结果显示TOTR在τ-MSE降低1-2个数量级、MSIE降低近两位数，尤其在极刚性和多事件场景中表现最优。

**⚠️ 局限性**

局限在于仅使用三阶样条做极值重参数化、τ_f和采样密度固定、未针对单个方法进行最佳调参，且仅在三个基准系统验证，未涵盖更复杂的非线性或高维系统。

---

## 90. Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability

**arXiv ID:** 2603.16017 | [PDF](https://arxiv.org/pdf/2603.16017v1)

**作者:** Fan Huang `[一作]` (Indiana University Bloomington), Jisun An `[通讯]` (Indiana University Bloomington)

**通讯引用:** 3755 | [OpenAlex ID](https://openalex.org/A5084955495)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并量化了大型语言模型在道德推理过程中的框架迁移轨迹，评估其多框架整合的稳定性与准确性。

**💡 创新点**

创新点在于引入“道德推理轨迹”概念，揭示多框架动态整合并结合内部表示的可解释性与轻量激活调节。

**🔧 技术方法**

采用结构化JSON多步推理、GPT-OSS-120B评分模型、线性探针、激活 steering 等技术。

**📊 数据集**

使用Moral Stories、ETHICS、Social Chemistry 101三大英文道德推理基准。

**📈 对比分析**

通过2×2因子实验和FDR、MRC指标比较，发现结构化推理下多框架混合可提升约7%准确率，FDR 0.55-0.58，MRC与一致性评分相关。

**⚠️ 局限性**

局限在样本规模小、仅覆盖西方框架、模型访问受限于开放权重、激活调节效果有限、文化多样性未覆盖。

---

## 91. Evaluating Performance Characteristic of Opportunistic Routing Protocols: A Case Study of the 2016 Italian League Match Earthquake in the Stadio Adriatico

**arXiv ID:** 2603.15945 | [PDF](https://arxiv.org/pdf/2603.15945v1)

**作者:** Yihang Cao `[一作]` (University of Nottingham), Milena Radenkovic `[通讯]` (University of Nottingham)

**通讯引用:** 1671 | [OpenAlex ID](https://openalex.org/A5018791184)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在意大利足球场地震紧急场景下，用仿真实验比较了Epidemic和Spray and Wait两种DTN路由协议的性能。

**💡 创新点**

首次将两种路由协议在高密度地震灾害场景中进行对比，并揭示控制复制策略在存储受限环境下的优势。

**🔧 技术方法**

使用ONE仿真器、ShortestPathMapBasedMovement移动模型、Bluetooth/WiFi接口以及Epidemic和Spray and Wait协议。

**📊 数据集**

基于2016年中部意大利地震的真实地震数据与Stadio Adriatico场地地图构建的仿真场景。

**📈 对比分析**

通过交叉实验设置不同缓冲区大小，评估交付概率、时延、开销比例、跳数和丢包数；结果显示Spray and Wait在交付概率、时延和资源开销方面优于Epidemic。

**⚠️ 局限性**

实验仅限于仿真，缺乏真实硬件验证，且只评估了两种协议，未考虑能耗和更复杂的网络拓扑。

---

## 92. OMNIFLOW: A Physics-Grounded Multimodal Agent for Generalized Scientific Reasoning

**arXiv ID:** 2603.15797 | [PDF](https://arxiv.org/pdf/2603.15797v1)

**作者:** Hao Wu `[一作]` (Tsinghua University), Xian Wu `[通讯]` (Tencent)

**通讯引用:** 12989 | [OpenAlex ID](https://openalex.org/A5100352418)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于神经‑符号架构的OMNIFLOW框架，能够将冻结的多模态LLM与物理规律对齐，解决PDE驱动的时空动力学推理问题。

**💡 创新点**

创新点包括语义‑符号对齐机制、物理引导链式思维（PG‑CoT）、对抗式反馈循环以及物理一致性检查，实现零参数更新的跨域、可解释科学推理。

**🔧 技术方法**

采用冻结的Gemini 3 Flash、Diffusion Transformer（DiT）神经模拟器、视觉符号投影器、检索增强生成（RAG）、反事实探测回路和物理一致性Critic。

**📊 数据集**

在微观湍流、全球ERA5气候和区域SEVIR天气三个多尺度基准集上进行实验，同时与CNN、ViT、UNet、FNO、EarthFarseer等模型以及ChatGPT‑Images等全通用基础模型对比。

**📈 对比分析**

通过RMSE/SSIM/PSNR等数值指标和Mech‑F1等推理质量评估，OMNIFLOW在所有基准上实现了最高的物理一致性和预测精度，零样本情况下优于专用模型和单体基础模型。

**⚠️ 局限性**

主要局限是推理循环和反事实探测导致推理时延增加、依赖神经模拟器的精度限制，以及对极细尺度子格现象的语言表征不足。

---

## 93. CoDesignAI: An AI-Enabled Multi-Agent, Multi-User System for Collaborative Urban Design at the Conceptual Stage

**arXiv ID:** 2603.16008 | [PDF](https://arxiv.org/pdf/2603.16008v1)

**作者:** Zhaoxi Zhang `[一作]` (University of Florida), Tamir Mendel `[通讯]` (Academic College of Tel Aviv-Yaffo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个基于大型语言模型的多用户、多AI代理协作平台CoDesignAI，支持参与式城市设计的概念阶段。

**💡 创新点**

创新点在于将多用户讨论与多角色AI专家协同，采用轮次式对话和图像修订，将文字讨论转化为街景视觉方案，突破传统单一专家或单向参与的局限。

**🔧 技术方法**

使用了Google Gemini 2.5 Flash（文本与图像生成）、Google Street View API、Google Cloud Firestore、Node.js + React 前后端架构。

**📊 数据集**

使用了Google Street View公开街景数据作为场景，用户生成的对话文本与设计指令。

**📈 对比分析**

论文未进行量化对比，仅通过示例演示；未报告具体性能指标。

**⚠️ 局限性**

局限性包括缺乏真实用户试验与效果评估、对冲突检测与多方同步支持不足、成本可扩展性未知、对数字鸿沟与包容性考量不足。

---

## 94. PhasorFlow: A Python Library for Unit Circle Based Computing

**arXiv ID:** 2603.15886 | [PDF](https://arxiv.org/pdf/2603.15886v1)

**作者:** Dibakar Sigdel `[一作]` (Mindverse Computing), Namuna Panday `[通讯]` (Mindverse Computing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了 PhasorFlow，一个基于 S^1 单位圆的开源 Python 计算库，提供周期性相位电路、可变相位量子风格模型和基于 DFT 的 Transformer。

**💡 创新点**

创新点在于将单元圆算子引入可编程计算，利用连续相位干涉实现低参数、高效全局混合，并提出 Variational Phasor Circuit（VPC）与 Phasor Transformer 两种新型模型。

**🔧 技术方法**

技术核心包括 22 个门级原语（Shift、Mix、DFT、Invert 等）、可训练相位参数、PyTorch 复数张量自动微分以及基于 DFT 的无参数全局混合。

**📊 数据集**

主要使用合成数据：高维非线性分类、混合正弦序列时间预测以及 200 天的 OHLCV 价格波动，用于验证模型的泛化与异常检测能力。

**📈 对比分析**

与传统全连接网络、Transformer（自注意力）及经典 DFT 混合比较，VPC 在 16 维非线性分类上 100% 准确、Phasor Transformer 在时间序列回归上仅用 50 个参数即可匹敌自注意力模型的误差，并在金融波动检测中无需训练即可显著区分正常与危机区间。

**⚠️ 局限性**

局限性包括：相位门无法实现任意线性变换，模型对大规模实测数据的验证有限，扩展到更高维或复杂任务时需进一步提升算子多样性与硬件加速支持。

---

## 95. CUBE: A Standard for Unifying Agent Benchmarks

**arXiv ID:** 2603.15798 | [PDF](https://arxiv.org/pdf/2603.15798v1)

**作者:** Alexandre Lacoste `[一作]` (ServiceNow), Dawn Song `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CUBE（Common Unified Benchmark Environments）标准，统一 agent benchmark 的包装与调用，消除多平台整合成本；

**💡 创新点**

设计四层接口（Task、Benchmark、Package、Registry），基于MCP+Gym融合的异步交互、统一的工具配置、可选的特权信息、自动化合规与调试工具，构建轻量级注册中心；

**🔧 技术方法**

使用Python实现、RPC（HTTP）桥接、Model Context Protocol (MCP) 与Gym API 结合、Docker/Apptainer VM 资源声明、CLI 与 CI 集成；

**📊 数据集**

覆盖现有 300+ 以上 agent benchmark（WebArena、SWE‑Bench、OSWorld、GAIA 等），并在注册中心中发布元数据；

**📈 对比分析**

与现有平台（NeMo Gym、AgentBeats、OpenEnv、Harbor、HAL 等）进行功能对比，展示 CUBE 在多任务并行、资源声明、异步交互与统一工具调用上的优势；性能上通过压力测试和并行重放验证任务重置与资源隔离满足合规标准，显著降低了“integration tax”；

**⚠️ 局限性**

目前仍处于早期实现阶段，需社区共同完善标准与注册中心；缺乏大规模基准的实测数据，标准化过程可能导致初期兼容性问题；

---

## 96. Rapid Worst-Case Gust Identification for Very Flexible Aircraft Using Reduced-Order Models

**arXiv ID:** 2603.16212 | [PDF](https://arxiv.org/pdf/2603.16212v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (Institute of Communications and Computer Systems), Kenneth J. Badcock `[通讯]` (University of York)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

**🎯 论文内容**

通过构建非线性ROM实现了对非常柔性飞机的快速极端风载识别。

**💡 创新点**

创新点在于将二阶Taylor展开与特征向量投影相结合的非线性模型降阶方法，既保持激励独立性，又能捕捉几何非线性。

**🔧 技术方法**

采用非线性模型降阶(NMOR)、Taylor展开、特征向量投影和快速时间积分等技术。

**📊 数据集**

使用三个案例数据集：三自由度机翼（14 DOF）、全球鹰型UAV（540 DOF）和32 m超灵活飞翼（1,616 DOF）。

**📈 对比分析**

与全阶非线性仿真相比，ROM搜索在三种案例中分别实现了30×、600×等速度提升，且能准确预测极端风响应。

**⚠️ 局限性**

局限在于使用条带理论的气动模型，线性ROM仅适用于≤10%翼尖变形；对多维风场及更高保真气动模型的适用性尚需验证。

---

## 97. BadLLM-TG: A Backdoor Defender powered by LLM Trigger Generator

**arXiv ID:** 2603.15692 | [PDF](https://arxiv.org/pdf/2603.15692v1)

**作者:** Ruyi Zhang `[一作]` (National University of Defense Technology), Haifang Zhou `[通讯]` (National University of Defense Technology)

**通讯引用:** 526 | [OpenAlex ID](https://openalex.org/A5073429776)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于大型语言模型（LLM）的触发器生成器 BadLLM‑TG，用于在文本后门攻击中反演触发器并通过对抗训练消除后门。

**💡 创新点**

创新在于将 prompt‑驱动的强化学习引入离散文本空间，使LLM能够自动学习触发器模式，同时实现无模型结构依赖的后门消除。

**🔧 技术方法**

使用了 Llama‑3.3‑70B‑Instruct 作为触发器生成器，结合 prompt‑驱动强化学习、奖励函数为受害模型交叉熵损失，以及对抗训练来消除后门。

**📊 数据集**

实验采用了 SST‑2、HSOL 与 AG News 三个公开文本分类数据集。

**📈 对比分析**

与 ONION、RAP、MuScleLoRA、BadActs 四种主流后门防御方法在 Wordbkd、Sentbkd、Stylebkd、Synbkd 四种攻击下对比，BadLLM‑TG 平均 ASR 降至 23.8%（比第二好 13.7%），且 CACC 维持在 96.3%。

**⚠️ 局限性**

局限性包括对高比例（>40%）poisoning 时性能下降，以及对 LLM 计算资源需求较大。

---

## 98. Novel CRT-based Asymptotically Ideal Disjunctive Hierarchical Secret Sharing Scheme

**arXiv ID:** 2603.16267 | [PDF](https://arxiv.org/pdf/2603.16267v1)

**作者:** Hongju Li `[一作]` (Chu University), Cheng Shu `[通讯]` (Hefei University of Technology)

**通讯引用:** 26808 | [OpenAlex ID](https://openalex.org/A5100354225)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文对先前基于CRT的分离层次秘密分享（DHSS）方案进行安全分析，发现Yang等人和Tiplea等人的理想与渐近理想方案存在安全漏洞，并基于多项式环CRT和单向哈希函数提出一种新的渐近理想、计算安全且支持灵活份额大小的DHSS方案。

**💡 创新点**

创新点在于：①通过构造攻击实例揭示现有理想DHSS方案的安全缺陷；②首次提出一种计算安全但信息率可达1、份额大小可调的渐近理想DHSS方案；③利用单向哈希函数与多项式CRT结合，兼顾安全性与效率。

**🔧 技术方法**

采用的技术包括：多项式环上的中国剩余定理（CRT），单向哈希函数，分层阈值结构的秘密分享构造，以及渐近完美性与信息率的理论分析。

**📊 数据集**

本文未使用具体实验数据集，全部结果基于理论推导与数学证明。

**📈 对比分析**

与先前方案相比，所提方案在信息率方面达到1，份额大小更小（相同d₀时仅d₀·log₂p位），但需要发布更多公开值；安全性从完全信息安全转为计算安全，满足大多数实际应用的安全需求。

**⚠️ 局限性**

局限性包括：仅具备计算安全性（依赖单向哈希函数的强度）；公开值数量多，系统复杂度相对较高；实现中需保证多项式互素及哈希函数不可逆性，实际部署时可能面临性能瓶颈。

---

## 99. Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies

**arXiv ID:** 2603.15903 | [PDF](https://arxiv.org/pdf/2603.15903v1)

**作者:** Nathaniel Imel `[一作]` (New York University), Noga Zaslavsky `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一个统一模型，将演化博弈理论中的噪声相似最大化信号游戏与信息瓶颈（IB）框架结合，演示在噪声模仿动力学下，群体词汇能自发趋向信息理论上的最优压缩；

**💡 创新点**

创新点在于首次将IB效率目标与演化博弈中的局部模糊模仿动态相结合，证明局部频率依赖模仿可导致整体系统近似达到IB极限；

**🔧 技术方法**

主要技术包括信息瓶颈的互信息度量、噪声相似最大化（sim‑max）信号游戏、复制器方程改写的模糊模仿动态以及数值仿真；

**📊 数据集**

使用的是人工生成的单维数值域（0–99）作为语义空间，且采用均匀先验分布和高斯型状态混淆；

**📈 对比分析**

通过与随机置换控制以及有限种群复制器-突变基线比较，发现演化系统的效率损失（与IB最优差距）远低于基线，并且与游戏的精度参数呈现高度相关；

**⚠️ 局限性**

局限性包括：模糊模仿无法实现完全双射映射，导致最高可达精度受限；实验仅基于单一人工域，未验证在真实语言数据或多维语义空间中的适用性；

---

## 100. Who Benchmarks the Benchmarks? A Case Study of LLM Evaluation in Icelandic

**arXiv ID:** 2603.16406 | [PDF](https://arxiv.org/pdf/2603.16406v1)

**作者:** Finnur Ágúst Ingimundarson `[一作]`, Steinþór Steingrímsson `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对冰岛语低/中资源语言的 benchmark 进行定量错误分析，比较机器翻译、LLM 生成与人工作者的 benchmark 在质量与有效性上的差异；

**💡 创新点**

提出“谁在评估评估标准”概念，系统性量化 benchmark 失误并揭示机器翻译导致的文化与语法问题，为低资源语言 benchmark 提供改进建议；

**🔧 技术方法**

采用人工标注（三名评注者）并使用 Krippendorff α 计算互评信度，对样本进行 IC/F/OK 三类标签，结合置信区间与 p 值进行统计比较；

**📊 数据集**

使用 Miðeind 与 EuroEval 提供的冰岛语 benchmark 集，如 WinoGrande‑IS、ARC‑Challenge‑IS、Belebele‑IS、MultiWikiQA‑IS、HellaSwag‑IS、MMLU‑IS 等机器翻译或 LLM 生成的子集；

**📈 对比分析**

通过对比 human‑author 版与 machine‑translated 版的 OK 比例、IC 比例和互评信度，发现机器翻译版质量显著低于人类版，部分 benchmark 甚至无有效样本，说明错误会严重影响评测结果；

**⚠️ 局限性**

局限性包括样本规模有限、未覆盖所有 benchmark、标注者对 IC/F 边界的主观性、未直接评估对模型排名的影响，只提供了 benchmark 质量评估而非实际性能验证。

---

## 101. Attention-guided Evidence Grounding for Spoken Question Answering

**arXiv ID:** 2603.16292 | [PDF](https://arxiv.org/pdf/2603.16292v1)

**作者:** Ke Yang `[一作]` (Ant Group), Chengjun Mao `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Attention‑guided Evidence Grounding（AEG）框架，利用 SpeechLLM 内部注意力显式定位并标记关键证据，从而实现端到端的语音问答并提升答案的可信度与可解释性。

**💡 创新点**

创新点在于：1）首次将 SpeechLLM 的跨模态注意力用于证据定位；2）设计了 Learning to Focus on Evidence（LFE）监督微调方法，使注意力聚焦于关键证据；3）实现了低延迟、无 ASR 误差传播的端到端系统。

**🔧 技术方法**

使用技术包括 SpeechLLM（如 Qwen2‑Audio‑7B、GPT‑4o Audio 等）、注意力提取与层级聚合、监督微调（SFT）以生成证据、标记式答案生成，以及 vLLM 部署实现高效推理。

**📊 数据集**

实验数据集为 SQuAD v1.1、HotpotQA 与 MuSiQue（通过 Higgs Audio 合成语音查询），覆盖多种问答场景。

**📈 对比分析**

对比了传统 Whisper+Reranker 级联系统和未微调的 AEG。AEG（带 LFE）在 EM、F1 等指标上均优于基线，且推理延迟约为 238 ms，较级联系统快约 60%（约 400–600 ms）。

**⚠️ 局限性**

局限性包括：需要大量标注证据的数据；跨模态注意力解释性仍不充分；在极长上下文或噪声较大的语音中性能可能下降；目前仅验证了文本上下文的场景，未涵盖更复杂的对话式语音问答。

---

## 102. SparkVSR: Interactive Video Super-Resolution via Sparse Keyframe Propagation

**arXiv ID:** 2603.16864 | [PDF](https://arxiv.org/pdf/2603.16864v1)

**作者:** Jiongze Yu `[一作]` (Texas A M University), Zhengzhong Tu `[通讯]` (Texas A M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出SparkVSR框架，通过用户选择稀疏关键帧并结合ISR模型生成高质量HR参考，利用双编码和扩散变压器在潜在空间传播以实现可控的视频超分。

**💡 创新点**

将稀疏可编辑关键帧作为可控锚点，构建两阶段潜在-像素训练，并配合可调节的参考无引导实现交互式控制，突破传统黑盒VSR的缺点。

**🔧 技术方法**

采用3D VAE+Diffusion Transformer（基于CogVideoX1.5‑5B I2V），零样本ISR（Nano‑Banana‑Pro、PiSA‑SR），CFG式参考无引导，双编码稀疏键帧策略。

**📊 数据集**

训练使用HQ‑VSR（2055段）+DIV2K（900图），评测包括UDM10、SPMCS、YouHQ40、RealVSR以及新建的MovieLQ（10段194帧）。

**📈 对比分析**

与STAR、DOVE、SeedVR2、FlashVSR等SOTA基线对比，SparkVSR在PSNR/SSIM、感知指标MUSIQ、CLIP‑IQA、FasterVQA、DOVER等上均取得领先或同等水平，尤其在MovieLQ的感知质量上显著超越。

**⚠️ 局限性**

仍受限于关键帧选择质量与ISR输出的噪声，过多参考会导致过拟合，且对极端运动或极低分辨率视频的细节恢复仍有限。

---

## 103. RaDAR: Relation-aware Diffusion-Asymmetric Graph Contrastive Learning for Recommendation

**arXiv ID:** 2603.16800 | [PDF](https://arxiv.org/pdf/2603.16800v1)

**作者:** Yixuan Huang `[一作]` (University of Electronic Science and Technology of China), Zongsheng Cao `[通讯]` (Tsinghua University)

**通讯引用:** 3441 | [OpenAlex ID](https://openalex.org/A5100381882)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 RaDAR 框架，结合图生成、关系感知去噪和扩散增强的双视图对比学习，用于协同过滤推荐。

**💡 创新点**

创新点包括：1）双视图生成器（变分图生成 + 关系去噪）实现全局与局部结构互补；2）扩散驱动的数据增强与去噪联合训练；3）不对称对比目标解耦身份与上下文，提升异质邻域表达。

**🔧 技术方法**

技术手段包括：图卷积网络（GCN）、变分图自编码器（VGAE）、关系感知去噪层、扩散模型（前向/逆向噪声注入）、不对称对比学习（ACL）和信息瓶颈约束。

**📊 数据集**

使用三大公开基准：Last.FM、Yelp、BeerAdvocate（二值边），以及三大多行为电商数据集：Tmall、RetailRocket、IJCAI15（加权边）。

**📈 对比分析**

与 BiasMF、NCF、LightGCN、NGCF、SGL、AdaGCL 等传统与最新基线进行对比。RaDAR 在 Recall@20/NDCG@20 上均实现 3–5% 的显著提升，尤其在稀疏和噪声场景下领先显著。

**⚠️ 局限性**

局限性：仅适用于静态图结构，未考虑时间动态；训练成本较高，需进一步优化采样与加速策略。

---

## 104. Finder: A Multimodal AI-Powered Search Framework for Pharmaceutical Data Retrieval

**arXiv ID:** 2603.15623 | [PDF](https://arxiv.org/pdf/2603.15623v1)

**作者:** Suyash Mishra `[一作]` (F. Hoffmann-La Roche Ltd), Baddu Narendra `[通讯]` (Involead Services Pvt Ltd)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个名为Finder的多模态AI驱动搜索框架，用于药企高效检索文本、图像、音频和视频等多种数据类型。

**💡 创新点**

将传统关键词检索与大模型语义推理结合，采用混合向量检索（稀疏词袋+密集语义向量）并实现自动意图解析与多模态元数据路由，形成统一的检索层。

**🔧 技术方法**

使用了OpenAI Whisper（语音转写）、Qwen/Qwen2‑VL（视频字幕）、AWS Bedrock（图像/文本摘要、翻译）、Meta‑LLaMA3（标签/摘要生成）、Docling、PyMuPDF、BM42、Mixedbread Embeddings、Qdrant向量数据库、Claude4等多项技术。

**📊 数据集**

在保密的行业监管数据集上测试，包含约314,343份文档、78,028份幻灯片、31,070段视频、1,192段音频和34,797张图像，涵盖14类疾病、98+语言。

**📈 对比分析**

通过与传统TF‑IDF、BM25、混合检索版本对比，Finder在nDCG@10、Recall@50、MRR等指标上均显著提升；实验表明在Top‑10检索覆盖率达82.7%，总体相关率87.7%，平均检索延迟约2–5秒。

**⚠️ 局限性**

局限包括低资源语言翻译覆盖不足、长视频场景理解导致的延迟、LLM推理耗时、以及对高度模糊查询的检索精度仍有提升空间。

---

## 105. Towards the Vision-Sound-Language-Action Paradigm: The HEAR Framework for Sound-Centric Manipulation

**arXiv ID:** 2603.16086 | [PDF](https://arxiv.org/pdf/2603.16086v1)

**作者:** Chang Nie `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9211 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Vision‑Sound‑Language‑Action（VSLA）范式，并实现 HEAR 框架，使机器人能够在连续音频流、视觉、语言和本体感知的条件下进行实时、声音驱动的操作。

**💡 创新点**

创新点包括：① 通过 Historizer 构建连续音频记忆以克服 Blind Execution Interval；② 引入 Advancer 进行音频世界建模以解决 Temporal Motion Collapse；③ 采用流匹配 Realizer 生成平滑动作片段；④ 设计 OpenX‑Sound 及 HEAR‑Bench 以支持大规模预训练和严格的声音因果评估。

**🔧 技术方法**

使用技术包括：Streaming Stateful Transformer、Qwen3‑Omni 与 Qwen3‑0.6B 进行多模态推理、解码器 Transformer 进行未来音频代码预测、条件流匹配生成动作片段、音频 tokenizer 与自回归音频处理、开放式动作分块与延迟决策循环。

**📊 数据集**

使用的数据集与基准：OpenX‑Sound（对 Open X‑Embodiment 进行音频合成的预训练集）、HEAR‑Bench（基于 RoboTwin2 的声音感知模拟 benchmark）、以及实机 Franka Panda 的真实音频实验。

**📈 对比分析**

与 VLA 基线（OpenVLA、π₀.₅）以及其 Waveform/ASR 适配器、音频本地化策略（Play it by Ear、ManiWAV）对比。HEAR 在 HEAR‑Bench 上平均成功率 81%（最高为 91%），显著优于波形适配器 61% 与 ASR 35%；在真实 Franka Panda 上平均成功率 54%，远高于最佳基线 39%。

**⚠️ 局限性**

局限性：在真实环境中的机械自噪、回声与背景噪声仍导致性能波动；长时持续监控任务（如 Moka Coffee）成功率较低；依赖大规模预训练与高算力模型，部署成本较高。

---

## 106. FleetOpt: Analytical Fleet Provisioning for LLM Inference with Compress-and-Route as Implementation Mechanism

**arXiv ID:** 2603.16514 | [PDF](https://arxiv.org/pdf/2603.16514v1)

**作者:** Huamin Chen `[一作]` (vLLM Semantic Router Project), Xue Liu `[通讯]` (MBZUAI)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究一种从分析模型到实践的框架，利用二池路由和边界压缩来最小化LLM GPU fleet的成本。

**💡 创新点**

创新点在于：①用M/G/c排队模型推导最优二池边界^*并给出等边际GPU成本条件；②提出Compress‑and‑Route把硬件边界转化为可调软件参数，克服边界“成本悬崖”；③将模型、压缩与离线规划器整合，能在1 ms内给出最优fleet与压缩参数。

**🔧 技术方法**

技术手段包括M/G/c + Erlang‑C + Kimura校正的排队分析、KV‑cache利用率建模、提取式Prompt压缩、离线规划器的双层参数搜索和GPU成本优化。

**📊 数据集**

使用的工作负载包括Azure LLM Inference Trace 2023、LMSYS‑Chat‑1M（多轮）以及基于公开统计的Agent‑heavy合成 trace。

**📈 对比分析**

与同构fleet、仅池路由以及Retrofit基线对比；在不同工作负载下实现6–82%的GPU成本下降，TTFT始终满足P99 500 ms SLO；离线规划器在1 ms内完成搜索。

**⚠️ 局限性**

局限性：依赖准确的请求长度CDF、压缩可压缩率和GPU硬件参数；对代码或不适合提取式压缩的请求无效；模型在极低GPU数或非“多服务器”场景下可能欠拟合。

---

## 107. Agile Interception of a Flying Target using Competitive Reinforcement Learning

**arXiv ID:** 2603.16279 | [PDF](https://arxiv.org/pdf/2603.16279v1)

**作者:** Timothée Gavin `[一作]` (Thales Group), Murat Bronz `[通讯]` (Fédération ENAC ISAE-SUPAERO ONERA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文通过竞争式强化学习训练拦截者与目标无人机，利用低层推力与姿态指令实现敏捷拦截。

**💡 创新点**

创新点在于将高保真四旋翼动力学与低层控制集成，并在竞争式多智能体强化学习框架下同时训练拦截者与目标，使拦截率显著提升并实现零样本 sim-to-real 转移。

**🔧 技术方法**

使用了 Proximal Policy Optimization（PPO）、JAX 并行仿真、SE(3) 控制器、低层体角/推力指令以及集中训练分布式执行技术。

**📊 数据集**

实验使用自生成的随机初始位置的仿真环境，无需公开数据集；训练共 1024 并行仿真，累计 4×10⁹ 步。

**📈 对比分析**

与纯追踪、比例导航、人工势场等基线对比，RL 拦截率可达 90%，时间短且碰撞率低；在 8×8×5 m 室内实验中实现 7 次成功拦截，误差小。

**⚠️ 局限性**

局限性包括 sim-to-real 差距、对特定策略过拟合、缺少传感器噪声/局部观测、仅单一拦截/躲避、未考虑多机或障碍环境。

---

## 108. MiroThinker-1.7 & H1: Towards Heavy-Duty Research Agents via Verification

**arXiv ID:** 2603.15726 | [PDF](https://arxiv.org/pdf/2603.15726v1)

**作者:** MiroMind Team `[一作]`, P. Zhu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MiroThinker-1.7及其升级版MiroThinker-H1，专注于长周期推理任务的可靠性与可验证性

**💡 创新点**

引入了agentic mid‑training阶段提升每一步的规划、推理和工具使用能力，并在推理过程中加入本地与全局验证机制以持续审核并改进推理轨迹

**🔧 技术方法**

结合ReAct框架、滑动窗口上下文管理、工具调用接口、强化学习（GRPO）以及直接偏好优化（DPO）等技术；采用多阶段训练管线（mid‑training、SFT、PO、RL）

**📊 数据集**

使用大规模LLM预训练模型Qwen3 MoE；构建的训练数据包括多轮agent轨迹、单轮规划、推理与总结样本；评测覆盖BrowseComp、GAIA、FrontierScience‑Olympiad、FinSearchComp、MedBrowseComp等公开基准

**📈 对比分析**

与多款开源与商用研究型agent（Gemini‑3.1‑Pro、Claude‑4.6‑Opus、OpenAI‑GPT‑5等）进行比较；MiroThinker‑H1在BrowseComp、GAIA、FrontierScience‑Olympiad、FinSearchComp等任务上分别取得88.2、88.5、79.0、73.9的最高分，显著优于同类模型

**⚠️ 局限性**

在高负载或极长推理轨迹中仍可能产生累积误差；验证机制依赖LLM判断，可能受模型偏差影响；对资源占用及实时性要求较高，需进一步优化效率

---

## 109. The Cost of Reasoning: Chain-of-Thought Induces Overconfidence in Vision-Language Models

**arXiv ID:** 2603.16728 | [PDF](https://arxiv.org/pdf/2603.16728v1)

**作者:** Robert Welch `[一作]` (KTH Royal Institute of Technology), Kevin Smith `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 15502 | [OpenAlex ID](https://openalex.org/A5069299612)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在视觉语言模型中使用链式思考（CoT）推理会如何影响不确定性估计的质量，揭示推理过程中出现的隐式答案条件化导致token级别置信度过度膨胀，从而削弱不确定性排名的可靠性。

**💡 创新点**

首次将隐式答案条件化视为推理对不确定性量化产生负面影响的根本机制，并通过掩码干预实验验证其作用；同时提出在推理环境下保持鲁棒性的基于一致性的“不确定性估计”方法。

**🔧 技术方法**

采用了多种不确定性估计技术（ATL如最大序列概率、困惑度、均值熵、Monte Carlo序列熵；SRC自报置信度；Consistency一致性估计），在Selective Generation框架下使用PRR、Spearman相关系数和AUGRC等指标进行评估，并通过对CoT提示与非CoT、推理训练模型的对比，实施控制性掩码实验。

**📊 数据集**

使用了四个视觉语言基准数据集：OK‑VQA、MathVista、MMMU‑Pro‑Vision和Oxford‑IIIT Pet，分别涵盖知识密集型问答、多步视觉数学推理、专家级多模态推理和细粒度图像分类。

**📈 对比分析**

方法在不同模型（Gemma3‑4B‑IT、Qwen3‑VL‑8B/32B）和推理方式（提示式CoT、推理训练）下进行比较。实验结果显示，ATL估计在推理前表现最好，但在CoT或推理训练下排名质量显著下降；而Consistency估计在所有设置下保持稳定甚至提升，PRR和Spearman指标均优于ATL；AUGRC提升主要源自准确率提升，而非不确定性排名改进。

**⚠️ 局限性**

局限性包括：隐式答案条件化的根本机制仍未完全解析；掩码实验仅能部分消除置信度膨胀，未覆盖所有潜在影响因素；实验集中于少数模型和数据集，对更广泛的视觉语言模型或更复杂推理任务的推广需要进一步验证。

---

## 110. pADAM: A Plug-and-Play All-in-One Diffusion Architecture for Multi-Physics Learning

**arXiv ID:** 2603.16757 | [PDF](https://arxiv.org/pdf/2603.16757v1)

**作者:** Amirhossein Mollaali `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**通讯引用:** 6323 | [OpenAlex ID](https://openalex.org/A5078138445)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

该研究提出了pADAM，一个统一的生成式扩散框架，可在单一模型中实现多物理PDE的前向预测、逆推断和模型选择。

**💡 创新点**

创新点在于通过类条件扩散模型学习跨物理规律的共享概率先验，实现任务无关的推断，并结合conformal calibration提供可靠的不确定性量化。

**🔧 技术方法**

采用类条件扩散模型（EDM）、贝叶斯观察引导、概率流ODE、参数提升、合成条件采样以及conformal prediction校准技术。

**📊 数据集**

数据集覆盖七种PDE（扩散、对流、扩散-对流、扩散-对流-反应、Allen–Cahn、Burgers'、Navier–Stokes），均由数值求解生成。

**📈 对比分析**

与各类专用扩散模型基线相比，pADAM在前向预测与逆推断的相对L2误差仅高于10%，在稀疏观测下仍保持低误差；通过conformal校准把置信区间覆盖率从35%提升至99%。

**⚠️ 局限性**

限制包括对未见动力学的零样本外推仍受观测稀疏程度限制，模型对参数维度扩展时误差略增，且目前仅支持二维离散网格。

---

## 111. EvoIQA - Explaining Image Distortions with Evolved White-Box Logic

**arXiv ID:** 2603.15887 | [PDF](https://arxiv.org/pdf/2603.15887v1)

**作者:** Ruchika Gupta `[一作]` (Michigan State University), Wolfgang Banzhaf `[通讯]` (Michigan State University)

**通讯引用:** 16116 | [OpenAlex ID](https://openalex.org/A5004837138)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了基于遗传程序的可解释符号回归框架 EvoIQA，用以评估图像质量。

**💡 创新点**

通过将多种 FR‑IQA 指标的统计特征映射为 AGGD 参数并进化可读数学表达式，既实现了高性能又保持了白盒可解释性。

**🔧 技术方法**

采用 StackGP 遗传程序、AGGD 统计、随机森林特征重要性筛选、梯度与色度相似度等特征。

**📊 数据集**

在 TID2013、KADID‑10k、CSIQ、CID:IQ、VDID2014 等公开数据库上训练和测试。

**📈 对比分析**

与传统 FR‑IQA（SSIM、MS‑SSIM、FSIM、HaarPSI）及深度学习模型（MEON、DB‑CNN）和机器学习基线（SVR、RF）在 SROCC 上比较，EvoIQA 在大多数类别达到或超过 0.89 的相关系数，甚至与 DB‑CNN 相当。

**⚠️ 局限性**

仅针对全参考 FR‑IQA，缺乏对无参考场景的直接适用，且仍需手工设计终端特征集合。

---

## 112. S-VAM: Shortcut Video-Action Model by Self-Distilling Geometric and Semantic Foresight

**arXiv ID:** 2603.16195 | [PDF](https://arxiv.org/pdf/2603.16195v1)

**作者:** Haodong Yan `[一作]` (Hong Kong University of Science and Technology), Haoang Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 903 | [OpenAlex ID](https://openalex.org/A5040338788)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出S-VAM框架，利用单步扩散推理并通过自蒸馏技术将多步生成的几何与语义视觉表示映射到单步特征，进而实现实时高精度的机器人动作预测。

**💡 创新点**

创新点在于：①使用自蒸馏将多步扩散生成的稳定视觉表示作为教师，使轻量解耦器能在单步特征中提取一致的几何与语义前瞻；②通过几何（DPAv3）与语义（DINOv2）双分支解耦以及Uni-Perceiver整合，兼顾实时性与高精度。

**🔧 技术方法**

主要技术包括：Stable Video Diffusion (SVD) 作为视频生成后端；DPAv3 与 DINOv2 作为教师视觉表示；时空Transformer解耦器；Uni-Perceiver 进行多模态特征聚合；Diffusion Transformer 动作专家。

**📊 数据集**

实验数据集：CALVIN (四个桌面环境)、MetaWorld (50项操作任务)、真实双臂 Cobot 任务（Place-to-Pot、Place-to-Pot (Hard)、Pour-Water、Lift-Pot）。

**📈 对比分析**

与直接行动学习方法（RT‑1、OpenVLA 等）以及预测方法（VPP、HiF‑VLA 等）对比，S-VAM 在 CALVIN 平均长度4.16、MetaWorld 成功率72.8% 以上，明显优于现有最优方法；实时控制约25 Hz，推理时延 307.6 ms。

**⚠️ 局限性**

局限性：依赖预训练扩散模型的质量，若教师表示不够准确会影响蒸馏效果；自蒸馏训练阶段仍需多步采样；在极端动态或视觉极端模糊场景下前瞻误差可能仍会出现。

---

## 113. ECHO: Edge-Cloud Humanoid Orchestration for Language-to-Motion Control

**arXiv ID:** 2603.16188 | [PDF](https://arxiv.org/pdf/2603.16188v1)

**作者:** Haozhe Jia `[一作]` (Hong Kong University of Science and Technology), Yutao Yue `[通讯]` (Institute of Deep Perception Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ECHO框架，利用云端扩散模型将自然语言指令转化为机器人原生38维运动参考，并在边缘设备上采用轻量化强化学习追踪器实现实时闭环执行。

**💡 创新点**

创新点在于将运动生成与执行严格解耦，采用云端扩散生成无需实时逆运动学、边缘端追踪器低延迟实现，同时设计了专门的38维机器人运动表示和两阶段教师-学生策略提升 sim2real 性能。

**🔧 技术方法**

使用了基于1D卷积 UNet 的扩散生成网络（与 CLIP 编码的跨注意力结合）、DDIM 采样与分类器无关引导、以及教师-学生强化学习追踪器（PPO、证据回归、对称性损失与域随机化）。

**📊 数据集**

训练数据采用 HumanML3D（AMASS 的带字幕子集）通过 General Motion Retargeting 适配为机器人骨骼。

**📈 对比分析**

与多种基准（MDM、TM2T、MotionDiffuse、StableMofusion）比较，ECHO-UNet 在 FID、R‑Precision、MM Dist. 与机器人专用指标（MSS、RTC）上均优于对手，生成质量达 0.029 FID，实时生成约 1 秒；在真实 Unitree G1 上执行 80 条指令成功率 100%，关节误差在 22‑33 mm 范围。

**⚠️ 局限性**

局限性包括对网络延迟和通信带宽的依赖、对视觉环境感知的缺乏、以及在极端动态或复杂地形任务中可能的鲁棒性不足。

---

## 114. Kinema4D: Kinematic 4D World Modeling for Spatiotemporal Embodied Simulation

**arXiv ID:** 2603.16669 | [PDF](https://arxiv.org/pdf/2603.16669v1)

**作者:** Mutian Xu `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 44574 | [OpenAlex ID](https://openalex.org/A5100406050)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种基于动作的4D生成机器人仿真框架Kinema4D，能够在4D时空域内从初始世界图像与机器人动作序列生成未来的机器人-世界交互，并实现高保真、几何一致的模拟；

**💡 创新点**

核心创新在于将仿真拆分为精确的4D机器人控制（通过URDF和正逆运动学得到连续四维轨迹）与可生成的环境反应（将机器人轨迹投影为点图序列，作为控制信号驱动4D扩散Transformer生成同步RGB与点图），从而实现空间-时间一致的高维生成，并首次在大规模数据集上展示零样本迁移；

**🔧 技术方法**

技术手段包括：URDF驱动的三维机器人重建与正逆运动学求解、机器人轨迹投影为4D点图、基于VAE的多模态潜在空间、Diffusion Transformer（带RoPE与域嵌入）实现同步RGB与点图生成、LoRA微调以及4D扩散模型与多模态条件融合；

**📊 数据集**

使用了自研的大规模4D机器人交互数据集Robo4D-200k（约201,426条演示），该数据集从DROID、Bridge、RT-1、LIBERO等公开数据集收集并通过ST‑V2或真实深度重建得到高质量的四维注释；

**📈 对比分析**

在视频生成与几何一致性评估上，与UniSim、IRASim、Cosmos、Ctrl-World、EVAC、ORV、TesserAct等基线进行对比；使用PSNR、SSIM、FID、FVD、LPIPS、Chamfer距离与F-score等指标，Kinema4D在大多数指标上获得第一或第二名，并在零样本OOD评估中表现出较高的成功率，证明其优异的生成质量与泛化能力；

**⚠️ 局限性**

局限性在于环境动力学是基于统计学习而非显式物理约束，缺乏摩擦、质量守恒等物理法则，偶尔会出现穿透或不符合能量守恒的行为，需要进一步融合物理约束来提升真实性。

---

## 115. Don't Trust Stubborn Neighbors: A Security Framework for Agentic Networks

**arXiv ID:** 2603.15809 | [PDF](https://arxiv.org/pdf/2603.15809v1)

**作者:** Samira Abedini `[一作]` (CISPA Helmholtz Center for Information Security), Rebekka Burkholz `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将 Friedkin‑Johnsen (FJ) 观点动态模型引入大型语言模型多智能体系统 (LLM‑MAS)，理论推导并实验证明单一顽固或说服力强的代理可发动传播式攻击，并提出基于动态信任权重的防御机制，显著降低攻击成功率。

**💡 创新点**

创新点包括：①首次把 FJ 社会科学模型正式应用并验证其在 LLM‑MAS 中的准确性；②结合理论与实验得出攻击成功率与网络拓扑、代理顽固度、信任权重的解析关系；③提出可实时调整的信任机制（T‑WS）在动态环境下抵御自适应攻击。

**🔧 技术方法**

技术手段：FJ 线性动态系统建模、矩阵分析、最小二乘/优化参数拟合；多智能体实验框架（十轮消息传递）；信任权重自适应更新；代理行为属性（顽固度 α、说服力 γ）的 Prompt 设计；图论网络拓扑（星形、全连）构建。

**📊 数据集**

使用数据集：CommonsenseQA（100题）和 ToolBench（100题分组），两者均转换为多项选择任务以评估代理推理与共识。

**📈 对比分析**

比较方法：在不同网络拓扑和防御策略下计算攻击成功率 (ASR)；使用 R² 与 MSE 衡量 FJ 模型对实际 LLM 观点轨迹的拟合与预测；实验显示信任机制将 ASR 从 0.65 降至 0.21，FJ 模型在预测时 R² 维持 0.77–0.97，验证其理论与实证一致性。

**⚠️ 局限性**

局限性：①信任权重需由中心管理，缺乏去中心化实现；②对高度自适应攻击者的防御依赖于随机更新，仍存在被利用的窗口；③实验规模仅至 8 代理，未验证大规模网络的可扩展性；④部分 LLM（如 Ministral‑3‑14B）对代理属性调节响应不足；⑤顽固度提升虽增强安全但会削弱系统协同一致性，需权衡。

---

## 116. DynaTrust: Defending Multi-Agent Systems Against Sleeper Agents via Dynamic Trust Graphs

**arXiv ID:** 2603.15661 | [PDF](https://arxiv.org/pdf/2603.15661v1)

**作者:** Yu Li `[一作]` (Tianjin University), Junjie Wang `[通讯]` (Tianjin University)

**通讯引用:** 29106 | [OpenAlex ID](https://openalex.org/A5115695478)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DynaTrust，一种基于动态信任图的多智能体系统防御框架，用来识别和隔离“睡眠代理”隐藏的恶意行为。

**💡 创新点**

创新点在于把多智能体系统视为动态信任图，利用贝叶斯信任平滑、信任‑置信加权共识和自适应图恢复，实现实时的信任演化与自动隔离。

**🔧 技术方法**

采用了贝叶斯信任更新、Fast Auditor 快筛审计、专家投票加权、图结构重构与复制代理恢复等技术。

**📊 数据集**

使用了 AdvBench 与 HumanEval 组合而成的混合基准数据集进行评估。

**📈 对比分析**

与 AgentShield 和无防御基线对比，DynaTrust 在 DSR 上平均提升 41.7%（达到 92.4%），FPR 降至 2.2%，TSR 维持 84.9%，显示出高安全性与低误报的优异性能。

**⚠️ 局限性**

局限性包括共识机制对多数诚实专家的假设，一旦恶意代理群体联合支持，可能绕过检测；以及信任计算和投票带来的额外延迟与计算开销。

---

## 117. Domain-Independent Dynamic Programming with Constraint Propagation

**arXiv ID:** 2603.16648 | [PDF](https://arxiv.org/pdf/2603.16648v1)

**作者:** Imko Marijnissen `[一作]` (Delft University of Technology), Ryo Kuroiwa `[通讯]` (National Institute of Informatics)

**通讯引用:** 80 | [OpenAlex ID](https://openalex.org/A5037848536)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了一个通用框架，将约束传播（CP）集成进基于启发式搜索的动态规划（DP）求解器，演示了其在单机调度、资源受限项目调度和时间窗旅行商问题上的效果。

**💡 创新点**

创新点在于构建了一个模型化、问题无关的DP-CP混合接口，使得CP的推理能力可用于判定状态可行性、提供更强的下界，并在不改动搜索策略的前提下实现状态剪枝。

**🔧 技术方法**

技术手段包括 Domain‑Independent Dynamic Programming（DIDP）框架与其 Rust 接口、MiniZinc/OR‑Tools CP‑SAT 等通用 CP 求解器、启发式搜索算法 CABS 与 DP 搜索，以及约束传播算法如边缘查找和时间表过滤。

**📊 数据集**

实验使用了三类数据集：单机调度 900 个合成实例（来源于 Davari 等 2016）、RCPSP 的 480 J90 PSPLIB 实例以及 TSPTW 的 180 个公开实例（Gendreau 1998 与 Ascheuer 1996）。

**📈 对比分析**

通过对比基线 DP（CABS 或 DP）与加入 CP 传播的 DP+CP，以及 OR‑Tools CP 求解器，结果显示 DP+CP 在状态扩展和解答实例数量上均优于单纯 DP，尤其在约束紧凑的实例中，运行时优势明显；对松弛实例则因传播开销而略逊。

**⚠️ 局限性**

局限性主要在于传播过程的计算开销，尤其是对松弛实例导致的性能下降，以及固定点传播成本高、何时触发传播未优化等问题，需要进一步降低传播开销并探索更高效的触发策略。

---

## 118. Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning

**arXiv ID:** 2603.15981 | [PDF](https://arxiv.org/pdf/2603.15981v1)

**作者:** Jingxiang Chen `[一作]` (Meta Reality Labs), Zhaojiang Lin `[通讯]` (Meta Reality Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种多任务强化学习框架 PALLM，联合情绪分类与情感化响应生成，利用链式推理提升语音 LLM 对情绪线索的理解与表达。

**💡 创新点**

将情绪分类与情感化生成任务统一训练，并通过链式推理强制模型依据音频证据作出决策，防止仅靠词汇短路；采用两阶段训练（SFT + RL）使模型在有限标注下实现情绪感知与对话适配。

**🔧 技术方法**

多任务强化学习（GRPO）、链式推理（CoT）提示、监督式微调、LLM-judge 奖励、音频编码器+LLM 框架。

**📊 数据集**

Expresso、IEMOCAP、RAVDESS 三个公开语音情绪数据集，用于情绪分类和情感化响应评估。

**📈 对比分析**

与 SFT 仅生成、SFT 仅分类、Gemini‑2.5 Pro、GPT‑4o‑Audio 等基线比较。PALLM（Cls+Gen）在 Expresso 上情绪识别 77% 与 97% 之中最高，情感化响应准确率从 66.1% 提升至 77.0%，在 IEMOCAP 与 RAVDESS 同样取得领先或相近表现。

**⚠️ 局限性**

对域外数据 RAVDESS 的泛化不足；需要情绪标签的 RL 奖励限制了无标签数据的利用；LLM‑judge 作为奖励模型存在偏见与奖励操纵风险。

---

## 119. Lessons from Real-World Deployment of a Cognition-Preserving Writing Tool: Students Actively Engage with Critical Thinking and Planning Affordances

**arXiv ID:** 2603.15777 | [PDF](https://arxiv.org/pdf/2603.15777v1)

**作者:** Yinuo Yang `[一作]` (University of Notre Dame), Toby Jia-Jun Li `[通讯]` (University of Notre Dame)

**通讯引用:** 1814 | [OpenAlex ID](https://openalex.org/A5007240808)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在三门本科写作课中部署了AI支持的论证写作工具VISAR，并通过互动日志、写作成果、问卷与访谈等混合方法，研究学生在真实课堂任务中的使用模式与学习效果。

**💡 创新点**

创新点在于将认知保留式AI scaffolds（如结构化规划、目标式生成与视觉化论证图）与写作教育相结合，并在真实课堂环境中系统地评估其对学生论证质量、概念理解和批判性AI素养的影响。

**🔧 技术方法**

技术包括基于LLM的自动化生成（Elaboration、Spark、Draft Regeneration）、视觉化论证规划工作区、双向同步编辑，以及通过日志记录与k-means聚类和多元线性回归等数据分析手段。

**📊 数据集**

数据集主要由49名学生在三门课程中使用VISAR所产生的写作稿件、交互日志、学期前后测验分数以及问卷/访谈文本构成。

**📈 对比分析**

通过对交互行为的聚类与回归分析，发现高Spark使用与视觉规划的学生在论证质量上显著提升（如Claim Articulation β=0.88，Grounds β=0.43），而过度依赖LLM重写则相关性负向；学期前后测验得分平均从5.79提升至7.50，显示短期概念学习提升。

**⚠️ 局限性**

主要限制在于实验仅持续一周，缺乏长期跟踪数据，样本量有限且无对照组，且可视化工具在复杂论证时可能导致认知负荷增加。

---

## 120. Transition Flow Matching

**arXiv ID:** 2603.15689 | [PDF](https://arxiv.org/pdf/2603.15689v1)

**作者:** Chenrui Ma `[一作]` `[通讯]` (University of California), Chenrui Ma (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种直接学习全局转移流（Transition Flow）的生成模型框架，用于少步甚至一步生成。

**💡 创新点**

创新点在于推导出 Transition Flow Identity，将全局转移流与平均速度模型统一，直接建模转移路径而无需多步数值积分；并提出从头到尾可训练的条件目标学习方法。

**🔧 技术方法**

采用 Flow Matching 理论，构造条件目标损失（MSE），利用 Bregman 散度；网络采用 UNet/Transformer 架构，加入时间嵌入、JVP 计算、CFG 指导等技术。

**📊 数据集**

在 CIFAR‑10、ImageNet 256×256 等图像生成数据集上进行实验。

**📈 对比分析**

与 Consistency Models、Mean Velocity Models、Diffusion/Flow Matching 等方法对比，结果显示在 NFE=1 时 FID 最优，随着 NFE 增大性能持续提升，整体表现优于现有少步生成方法。

**⚠️ 局限性**

局限性包括：仍需较大模型与训练时间，实验主要集中在图像生成任务；对非线性插值或更高维连续分布的适用性尚未验证；模型对极端时间步长的鲁棒性尚待进一步研究。

---

## 121. You've Got a Golden Ticket: Improving Generative Robot Policies With A Single Noise Vector

**arXiv ID:** 2603.15757 | [PDF](https://arxiv.org/pdf/2603.15757v1)

**作者:** Omkar Patil `[一作]` (Arizona State University), Eric Rosen `[通讯]` (Robotics and AI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究在冻结的扩散或流匹配机器人控制策略中，用单一固定噪声向量（黄金票）替代采样噪声以提升下游任务奖励。

**💡 创新点**

创新在于提出“黄金票”概念并证明其在不调整模型权重、无需额外网络的前提下可提升预训练策略性能，并且可在多任务中实现跨任务通用。

**🔧 技术方法**

技术包括随机搜索寻找最优初始噪声向量、基于蒙特卡洛回合收益评估、冻结扩散/流匹配模型的黑盒使用。

**📊 数据集**

数据集涵盖四个仿真基准（franka_sim、robomimic、LIBERO、DexMimicGen）和三套真实机器人任务（RGB、点云拾取与推送）。

**📈 对比分析**

与标准高斯噪声及强化学习驱动的DSRL方法比较，黄金票在38/43任务中超越高斯噪声，且在DexMimicGen上与DSRL相当或更好，硬件实验提升成功率多达60%。

**⚠️ 局限性**

限制包括结果为确定性、可能对环境或起始状态过拟合、搜索效率受搜索预算限制，以及缺乏离线评估指标和对多样性噪声的进一步研究。

---

## 122. STARK: Spatio-Temporal Attention for Representation of Keypoints for Continuous Sign Language Recognition

**arXiv ID:** 2603.16163 | [PDF](https://arxiv.org/pdf/2603.16163v1)

**作者:** Suvajit Patra `[一作]` (Ramakrishna Mission Vivekananda Educational and Research Institute), Soumitra Samanta `[通讯]` (Ramakrishna Mission Vivekananda Educational and Research Institute)

**通讯引用:** 558 | [OpenAlex ID](https://openalex.org/A5091484246)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种统一的时空注意力编码器（STARK）用于关键点连续手语识别。

**💡 创新点**

在同一注意力模块中同时建模帧内空间关系与帧间时间关系，显著降低参数量（约70-80%）。

**🔧 技术方法**

使用了时空注意力、多头自注意力、平均/最大池化、CTC 损失与 KL 交叉蒸馏等技术。

**📊 数据集**

采用 RWTH‑PHOENIX‑Weather 2014T（Phoenix‑14T）手语视频数据集进行实验。

**📈 对比分析**

与 CoSign、MSKA 等最先进关键点模型对比，未使用预训练下验证 WER 21.0、测试 WER 21.9，参数仅 3M，性能相近且参数更少。

**⚠️ 局限性**

仍需进一步研究方法的鲁棒性及在更高多样性数据上的效果。

---

## 123. Age Predictors Through the Lens of Generalization, Bias Mitigation, and Interpretability: Reflections on Causal Implications

**arXiv ID:** 2603.16377 | [PDF](https://arxiv.org/pdf/2603.16377v1)

**作者:** Debdas Paul `[一作]` (Leibniz Institute on Aging — Fritz Lipmann Institute), Alessandro Cellerino `[通讯]` (Scuola Normale Superiore)

**通讯引用:** 8850 | [OpenAlex ID](https://openalex.org/A5037620435)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文设计并实现了一种基于域对抗神经网络（DANN）与二进制随机滤波（BSF）的年龄预测模型，旨在同时提升对齐属性的无关性、解释性与对外推的鲁棒性。

**💡 创新点**

创新点在于将属性无关学习与稀疏特征选择结合，利用对抗训练抑制样本属性信息并通过 BSF 自动筛选关键基因，形成可解释且公平的年龄表型预测框架。

**🔧 技术方法**

技术方法包括域对抗学习、梯度反转、二进制随机滤波、批归一化、dropout、信息瓶颈约束以及多头偏差预测器，整体构成多任务损失最小化。

**📊 数据集**

使用了六个公开的转录组数据集（GSE132040、GSE141252、GSE111164、GSE145480、GSE75192、GSE280699）以及一项 ELAM 治疗干预实验的数据。

**📈 对比分析**

通过留一集检验（LOSO）与传统线性、正则化回归、XGBoost 等基线模型对比，模型在 MAE、R² 与 CV 上显示更低的波动性和更稳健的外推表现，且能准确捕捉药物干预带来的表型变异。

**⚠️ 局限性**

局限性包括：无法提供因果解释；对抗训练并未完全消除属性信息，后期 probe 可恢复；对超参数敏感；在样本量有限的条件下仍可能过拟合；模型需要更多跨平台、多物种验证来进一步证明泛化能力。

---

## 124. Knowledge Graph Extraction from Biomedical Literature for Alkaptonuria Rare Disease

**arXiv ID:** 2603.15711 | [PDF](https://arxiv.org/pdf/2603.15711v1)

**作者:** Giang Pham `[一作]` (University of Pisa), Alina Sîrbu `[通讯]` (University of Bologna)

**通讯引用:** 1111 | [OpenAlex ID](https://openalex.org/A5008115861)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

基于PubTator3的文献挖掘方法，构建了针对罕见疾病阿尔卡普顿症（AKU）的两套知识图谱（扩展网络与高置信网络），并通过图论分析与数据库对比，挖掘潜在的分子机制、通路及药物候选。

**💡 创新点**

创新点在于：①首次针对罕见疾病系统性地生成多层次知识图谱；②结合多源验证（STRING、DGIdb、KEGG）评估图谱覆盖率与路径信息；③利用多重网络分析（中心性、社区、k-core、最大团、HeteSim等）从文本挖掘结果中提炼新的生物学假设和药物再利用策略。

**🔧 技术方法**

主要技术包括：PubMed检索 + PubTator3实体与关系抽取；文本关系置信度筛选与多文献支持阈值；图谱构建、边权重计算；图论特征计算（直径、度分布、聚类系数、度相关性）；社区检测（Leiden算法）、k-core与最大团分析；个人化PageRank、个人化Katz中心性、HeteSim元路径相似度；功能富集（gProfiler）与数据库对比验证。

**📊 数据集**

使用的数据集：PubMed全文检索结果（共168,502条独立PMID，覆盖AKU及其相关基因、代谢物、药物等）；PubTator3抽取的4,435个实体与4,931条关系；从STRING（实验+策划）得到的基因基因交互；DGIdb的药物-基因交互；KEGG人类酪氨酸代谢通路（hsa00350）。

**📈 对比分析**

比较方法：将构建的KG与STRING、DGIdb、KEGG的子图在顶点与边的重叠、最短路径覆盖率以及路径长度进行统计。结果显示：扩展网络与高置信网络在直接基因基因交互重叠率低，但通过中间节点的最短路径完全覆盖STRING信息；与KEGG的路径覆盖率高（>90%），说明图谱保留了核心代谢通路；高置信网络在路径长度更短、模块化更强。性能指标如覆盖率、路径长度与聚类系数表明高置信网络在保持核心信息的同时去除了噪声。

**⚠️ 局限性**

局限性：①文本挖掘受限于PubTator3的识别准确性，可能存在漏检或误检；②只考虑已发表的原始研究，忽略综述与临床数据；③阈值设置（置信度0.7、至少两篇文献）主观，影响网络规模与覆盖度；④缺乏实验验证新假设，仍为计算推测；⑤罕见疾病文献量不足导致图谱相对稀疏。

---

## 125. Dynamic Meta-Layer Aggregation for Byzantine-Robust Federated Learning

**arXiv ID:** 2603.16846 | [PDF](https://arxiv.org/pdf/2603.16846v1)

**作者:** Reek Das `[一作]` (APC Roy Government), Biplab Kanti Sen `[通讯]` (P.R. Thakur Government)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedAOT，一种基于元学习的自适应聚合框架，用于抵御拜占庭鲁棒性 Federated Learning 中的多标签翻转和无目标中毒攻击。

**💡 创新点**

创新点在于引入服务器端轻量元层动态学习客户端重要性权重，利用验证反馈进行 Meta‑gradient 更新，无需预设阈值或攻击假设，能够自适应识别未知攻击。

**🔧 技术方法**

采用元学习、Meta‑gradient 更新、Softmax/归一化权重、基于小型验证集的损失评估，并在 Flower+PyTorch 环境下实现。

**📊 数据集**

在 MNIST、KMNIST、FashionMNIST 等标准数据集上进行仿真，设置 20‑100 个客户端，攻击比例从 20% 到 90%。

**📈 对比分析**

与 FedAvg、FoolsGold、GeoMed 对比实验显示，在多标签翻转攻击下即使攻击率达 90%，FedAOT 仍能保持约 98% 的准确率，显著优于传统方法，鲁棒性和收敛性更好。

**⚠️ 局限性**

局限性包括对服务器端验证集质量的依赖，Meta‑gradient 在极端非 IID 或噪声更新下可能不稳定；目前主要针对无目标攻击，对目标或更复杂拜占庭攻击的鲁棒性尚待进一步验证。

---

## 126. Context-Length Robustness in Question Answering Models: A Comparative Empirical Study

**arXiv ID:** 2603.15723 | [PDF](https://arxiv.org/pdf/2603.15723v1)

**作者:** Trishita Dhara `[一作]` (Upper Hand), Siddhesh Sheth `[通讯]` (Ace Rent a Car)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型在长且噪声多的上下文中对问答任务的鲁棒性，构造可控的干扰上下文并评估准确率下降。

**💡 创新点**

通过保留答案信息、仅增加无关文本的方法，能够系统分离上下文长度对性能的影响，揭示多跳推理任务对干扰更敏感。

**🔧 技术方法**

使用指令调优的大型语言模型（如ChatGPT 3.5/4），基于 Tokenizer 的上下文构造，采用精确匹配（Exact Match）评估。

**📊 数据集**

SQuAD（单跨度抽取）和 HotpotQA（多跳推理）验证集。

**📈 对比分析**

对比不同上下文长度（256、512、1024、2048 tokens）和两模型规模，发现 HotpotQA 准确率下降超过 20% ，而 SQuAD 仅下降约 5%；更大模型略好但鲁棒性趋势相同。

**⚠️ 局限性**

仅测试两任务两模型、上下文长度上限 2048 tokens、使用 EM 指标，未考虑对抗或矛盾信息，样本量有限，结果可能不具普适性。

---

## 127. When Openclaw Agents Learn from Each Other: Insights from Emergent AI Agent Communities for Human-AI Partnership in Education

**arXiv ID:** 2603.16663 | [PDF](https://arxiv.org/pdf/2603.16663v1)

**作者:** Eason Chen `[一作]` (Carnegie Mellon University), Cyuan-Jhen Wu `[通讯]` (GiveRep Labs)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 167,000+ AI 代理的社区平台（如 Moltbook、The Colony、4claw 等）进行为期一个月的定性观察，提炼出 AI 代理之间无课程的同行学习、双向支架学习、共享记忆架构以及信任与可持续性等四大现象，并提出以“让学生通过配置与教授 AI 代理来学习”为核心的课程设计场景

**💡 创新点**

首次从自然产生的多代理社区中系统性观察到的教育启示：双向支架（人类通过教授 AI 学习自身）、无课程的同行学习、共享记忆对齐以及信任与生态可持续性的设计约束，为 AIED 设计提供了全新的多代理视角

**🔧 技术方法**

基于开放 API 数据采集、日常定性内容分析与反思性主题分析（reflexive thematic analysis），并结合作者在代理操作方面的实践经验

**📊 数据集**

公开 API 提供的 Moltbook、The Colony、4claw 等平台数据（约 167,963 名注册代理、23,980 篇帖子、232,813 条评论），以及作者在多个平台的操作记录和观察日志

**📈 对比分析**

该研究不涉及实验对比或性能评估，而是通过质性方法描述现象并提出设计假设；若要验证可通过实验或案例研究评估所提设计场景的有效性

**⚠️ 局限性**

局限包括：观察仅基于自然产生的代理社区，未直接涉及人类学习者；受限于平台数据完整性（可能含人类账户伪代理）；难以区分框架默认行为与真正的自组织；时间跨度短，平台生态快速变化；缺乏可量化性能指标

---

## 128. FlashSampling: Fast and Memory-Efficient Exact Sampling

**arXiv ID:** 2603.15854 | [PDF](https://arxiv.org/pdf/2603.15854v1)

**作者:** Tomas Ruiz `[一作]` (LMU Munich), Mengdi Wang `[通讯]` (Princeton University)

**通讯引用:** 6338 | [OpenAlex ID](https://openalex.org/A5100707460)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将 exact categorical sampling 融合进 LM-head 的 matmul epilogue，直接在芯片上计算 logits 并仅保留每行每个 vocab tile 的最大值，避免在 HBM 上写入、读取完整的 [B,V] logits 张量。

**💡 创新点**

1) 通过 Gumbel‑Max 的 exact 性质实现不需要 softmax 的 exact sampling； 2) 使用分块（tile）分解与分组（group）分层因子化，保证在线和分布式版本在分布上仍保持 exact； 3) 采用单阶段（只做一次 max）与两阶段（tile‑local max + 全局 max）融合核，极大减少额外内核和内存往返。

**🔧 技术方法**

Gumbel‑Max trick、分块矩阵乘法融合、分组/层次化采样、on‑chip 记录最大值、Triton GPU kernel、FP32 计算保证数值稳定。

**📊 数据集**

在 vLLM 的 AIME 语料上对 Qwen3‑1.7B、Qwen3‑8B、Qwen3‑32B 以及 gpt‑oss‑120b 进行推理评测；用于正确性验证的 GSM8K 数据集。

**📈 对比分析**

与基线的 Multinomial sampling（软max+多项式）以及 FlashInfer 的 FI1（top‑k/top‑p）和 FI2（Gumbel‑Max on materialized logits）进行对比。<B≤64> 时 FlashSampling 在四款 GPU 上均比 Multinomial 高 1.84×、比 FI1 高 2.52×；在 Qwen3‑1.7B 的端到端 vLLM 测试中，TPOT 降低高达 19%，在 Qwen3‑8B 仅 3–7%；在更大模型上收益显著下降。

**⚠️ 局限性**

1) 当 batch 变大后，gemm 的 compute‑bound 成本提升，FlashSampling 的相对优势随之减弱； 2) Triton 版矩阵乘法在大批量时不如 cuBLAS 高效； 3) 目前仅支持 exact Gumbel‑Max，未完全覆盖 top‑p 等更复杂的采样策略； 4) 需要手写高效 kernel，移植到非 NVIDIA GPU 或其他平台仍需工作。

---

## 129. SEMAG: Self-Evolutionary Multi-Agent Code Generation

**arXiv ID:** 2603.15707 | [PDF](https://arxiv.org/pdf/2603.15707v1)

**作者:** Yulin Peng `[一作]` (Shenzhen University), F. Richard Yu `[通讯]` (Carleton University)

**通讯引用:** 53882 | [OpenAlex ID](https://openalex.org/A5100420016)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SEMAG自进化多代理代码生成框架，自动拆分任务并动态升级基础模型。

**💡 创新点**

创新点在于自适应层次提示、协作式讨论决策以及实时模型选择的自进化机制。

**🔧 技术方法**

采用多层级代理模型、规划验证、嵌入追踪调试、讨论决策以及搜索工具集成。

**📊 数据集**

使用七个文本到代码基准，包括HumanEval、MBPP、HumanEval-ET、MBPP-ET、APPS、LiveCode和CodeContests。

**📈 对比分析**

与Direct、Self-Planning、MapCoder、LDB、LPW等基线比较，在所有基准上均超过现有最佳方法，GPT-4o版实现98.8%（HumanEval）、87.6%（MBPP）和52.6%（CodeContests）Pass@1。

**⚠️ 局限性**

局限在于需要手动设置推理超参数、对高难度任务产生延迟、以及自进化模型选择依赖实时检索，且缺乏离线模型推荐。

---

## 130. $x^2$-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space

**arXiv ID:** 2603.16671 | [PDF](https://arxiv.org/pdf/2603.16671v1)

**作者:** Ruishan Guo `[一作]` (Shenzhen International Graduate School, Tsinghua University), Xinlei Chen `[通讯]` (Shenzhen International Graduate School, Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于事件边缘的统一特征空间 Event Edge Space（EES），将图像、LiDAR 与事件三模态投影到同一边缘特征域，并在此空间内实现可靠性自适应融合与跨维度对比学习，从而同时精准估计 2D 光流和 3D 场景流。

**💡 创新点**

创新点：
①首次构建以事件边缘为锚点的边缘特征统一空间，实现不同模态的直接对齐；
②利用冻结的事件编码器作为对齐原型，对图像和 LiDAR 特征进行对称正则化；
③在共享空间内引入可靠性自适应融合模块和跨维度对比学习，促进 2D 与 3D 任务互补，显著提升鲁棒性。

**🔧 技术方法**

核心技术：
- 事件边缘编码器预训练（自监督预测未来边缘强度）；
- 三模态投影与对齐正则化；
- 可靠性自适应融合（全局可靠度、局部注意力加权）；
- 跨模态注意力 Transformer；
- 跨维度对比学习（跨帧拉伸、跨任务互信息约束）。

**📊 数据集**

使用的数据集：合成 EKubric（15k+ 图像‑LiDAR‑事件三模）与真实 DSEC（街景驾驶），并在两者上人工合成极端曝光、LiDAR 稀疏等降解场景。

**📈 对比分析**

与 RPEFlow、CMX、VisMoFlow 等最新基线相比，在标准与极端光照、LiDAR 稀疏等干扰条件下均取得 EPE、ACC 等指标的显著提升，达到 state‑of‑the‑art 级别。

**⚠️ 局限性**

局限性：
- 需要高质量事件数据作为对齐锚点，若事件信号弱或缺失会影响整体性能；
- 对其他模态（如雷达、视频）扩展需进一步验证；
- 在极大规模点云或极低事件帧率下的鲁棒性尚未充分评估。

---

## 131. MOSAIC: Composable Safety Alignment with Modular Control Tokens

**arXiv ID:** 2603.16210 | [PDF](https://arxiv.org/pdf/2603.16210v1)

**作者:** Jingyu Peng `[一作]` (University of Science and Technology of China), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6216 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MOSAIC框架，通过学习可插拔的控制令牌实现LLM的可组合安全对齐；

**💡 创新点**

核心创新在于将安全约束编码为可训练的控制令牌并在冻结的模型上优化，利用按组合顺序的任务采样和对比知识蒸馏实现高效、可增量的安全控制；

**🔧 技术方法**

使用可学习的控制令牌、组合任务采样策略、对比KD损失、冻结的Transformer后端以及多任务联合训练；

**📊 数据集**

构建了包含5类安全需求（成瘾、酒精、赌博、恐怖、性）共1500条样本的真实评估数据集；

**📈 对比分析**

与In-context、ORPO、SFT等基线比较，MOSAIC在DSR几乎达到100%且过度拒绝率显著低于对照组，且支持多类别组合与增量扩展；

**⚠️ 局限性**

主要限制是实验规模受限，仅在8B和3B模型上验证，数据集规模有限，未在更大模型或更广泛场景下测试。

---

## 132. Geometry-Aligned LLM Fine-Tuning for Sequential Narrow-Opening Planning

**arXiv ID:** 2603.16028 | [PDF](https://arxiv.org/pdf/2603.16028v1)

**作者:** Al Jaber Mahmud `[一作]` (George Mason University), Xuan Wang `[通讯]` (George Mason University)

**通讯引用:** 11385 | [OpenAlex ID](https://openalex.org/A5089622254)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

为解决多连通窄通道中需要长期几何推理的刚体运动规划问题，本文提出一种几何对齐的大语言模型微调框架，生成固定长度、可机读的 SE(2) waypoint 序列，并通过模型驱动规划与几何验证实现连续运动可行性。

**💡 创新点**

创新点在于：①将失败信息（首个违规索引与违规类型）直接嵌入监督，形成 failure‑driven LoRA SFT；②在此基础上采用 Group Relative Policy Optimization (GRPO) 与确定性几何奖励进行二阶微调，显式优化连续轨迹的几何约束；③采用固定长度机器可解析的输出格式，确保可解析与后续密集轨迹生成的连贯性；④通过自监督几何验证实现无标注强化学习，提升模型的长期几何推理能力。

**🔧 技术方法**

使用技术包括：LoRA 参数微调、GRPO 强化学习、基于 A* 的路径密集化、基于几何的确定性奖励（C1–C4 约束）、文本构造与错误反馈机制，以及 SE(2) 约束下的姿态插值。

**📊 数据集**

数据集为自生成的二维工作空间与障碍物集合，包含多条窄通道，利用 GUI 采集人类演示路径，形成 500 条训练场景与 1000 条 OOD 测试场景，全部为合成的矩形障碍与对象几何。

**📈 对比分析**

与未微调 Llama‑3.1‑8B‑Instruct、单纯 LoRA SFT、Failure‑driven LoRA SFT 等基线进行比较，ID 成功率从 5.2% 提升至 92.6%，OOB 成功率从 11.2% 提升至 77.3%；在几何约束违规率上也从 56% 降至 5.4%，证明了二阶微调与几何奖励的显著效果。

**⚠️ 局限性**

局限性包括：仅在二维平面上验证，未处理 3D 或不规则几何；依赖合成数据与人工演示，真实场景中的感知误差与动态障碍可能导致性能下降；以及对长序列生成的推理仍受模型规模与推理速度限制。

---

## 133. Protein Design with Agent Rosetta: A Case Study for Specialized Scientific Agents

**arXiv ID:** 2603.15952 | [PDF](https://arxiv.org/pdf/2603.15952v1)

**作者:** Jacopo Teneggi `[一作]` (Polymathic AI Collaboration), Siavash Golkar `[通讯]` (New York University)

**通讯引用:** 578 | [OpenAlex ID](https://openalex.org/A5020401988)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并验证了 Agent Rosetta：一个利用大型语言模型（LLM）在结构化 RosettaScripts 环境中进行蛋白质设计的多轮代理，能够迭代优化序列并处理天然与非天然氨基酸。

**💡 创新点**

创新点在于：① 通过环境抽象将 Rosetta 复杂语法转化为易于 LLM 生成的语义动作；② 设计多轮交互流程，让代理能实时评估中间设计并修正错误；③ 展示了通用 LLM 与专业科学软件结合时可达到与专用 ML 模型和人类专家相当的性能。

**🔧 技术方法**

技术栈包括：大型语言模型（Qwen3 Instruct、GPT‑5、Gemini 2.5 Flash、Claude Sonnet 4.5）；OpenAI Gym‑style RosettaScripts 环境；Rosetta 的 FastDesign、背骨扰动 Movers；评估工具 ESMFold 与 AlphaFold 3；结构化提示工程与动作模板化。

**📊 数据集**

实验数据集：8 个天然/合成蛋白（74–125 残基）用于常规氨基酸设计；4 个去 novo 蛋白（40–153 残基）用于在核心插入单个非天然氨基酸（NCAA TRF）；使用 PDB 原始结构、ESMFold 预测、AF3 预测作为评价基准。

**📈 对比分析**

比较方法：对 Agent Rosetta、ProteinMPNN、两套人类专家手写 Rosetta 脚本（一次性与分阶段）以及无约束的基线进行对比；指标为 ESMFold RMSD、pLDDT、Rosetta 能量。结果显示 Agent Rosetta 与 ProteinMPNN 在大多数靶点上 RMSD 差异 ≤ 0.20 Å，且在 NCAA 插入任务中超越人类基线；平均动作成功率 ≥ 86%，成本与 LLM 查询次数成正比。

**⚠️ 局限性**

局限性：① 计算与经济成本高于专用 ML 模型；② 评估指标（ESMFold、AF3）对非天然氨基酸的预测不完全可靠；③ 某些 LLM（如 Qwen3 Instruct）在处理 NCAA 设计时表现下降，需更精细的环境或提示；④ 仍需人工验证最终设计的实验可行性。

---

## 134. SAMSEM -- A Generic and Scalable Approach for IC Metal Line Segmentation

**arXiv ID:** 2603.16548 | [PDF](https://arxiv.org/pdf/2603.16548v1)

**作者:** Christian Gehrmann `[一作]` (Max Planck Institute for Security and Privacy), Christof Paar `[通讯]` (Max Planck Institute for Security and Privacy)

**通讯引用:** 19069 | [OpenAlex ID](https://openalex.org/A5041748332)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对IC金属层扫描电子显微镜（SEM）图像，研发并微调了基于SAM2的多尺度金属线分割工具，利用拓扑损失提升电气连通性。

**💡 创新点**

创新点在于将SAM2迁移到专业硬件图像领域，采用多尺度分割与Betti匹配拓扑损失相结合，显著提升在不同工艺节点与未知IC上的泛化性能。

**🔧 技术方法**

使用的技术包括Meta的SAM2基础模型、对图像编码器、提示编码器、掩码解码器的联合微调、数据增强、Betti匹配拓扑损失、以及Optuna的超参数搜索。

**📊 数据集**

使用了一个前所未有的SEM图像数据集，包含48层金属线、14个不同IC，工艺节点从200nm到20nm，涵盖多种材料、制样与显微镜设置。

**📈 对比分析**

与SAMIC、U-Net、DeepLabV3、FCN等方法在同一训练集上进行对比；在分布内测试中ESD误差率为0.62，分布外为5.53，均远优于同类方法（如最优对手24.77）。像素级指标与这些方法相当。

**⚠️ 局限性**

局限性包括：人工标注的Ground Truth存在不确定性；数据集受版权限制无法公开；实验耗时长、计算资源受限，导致对其他拓扑损失和更大规模实验的探索受限。

---

## 135. How often do Answers Change? Estimating Recency Requirements in Question Answering

**arXiv ID:** 2603.16544 | [PDF](https://arxiv.org/pdf/2603.16544v1)

**作者:** Bhawna Piryani `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Recency–Stationarity 维度的时间敏感问题分类法，并基于该法构建了 RecencyQA 数据集，包含 4,031 个开放域问题，分别标注了答案更新频率（recency）与是否随上下文变化（stationarity），以及对应的时间上下文。

**💡 创新点**

创新点在于：①用两维度（更新频率与上下文依赖）细粒度描述答案时效性，取代传统二元新旧；②通过 LLM 自主推理生成多重 recency 取样，构建分布式标签；③设计三步验证流程（LLM、模型投票、分布校验）提升标注可靠性；④首次将时间上下文与 recency 结合，支持非平稳问题的动态评估。

**🔧 技术方法**

使用技术主要包括：LLaMA‑3.3‑70B 进行问题/上下文生成与 13 次多样采样；GPT‑5.2、Gemini‑3 Flash、Claude‑Sonnet 进行 stationarity 预标注；多模型多数投票与分布一致性校验；零样、少样、链式思维三种 Prompting 方案评估；以及基于 Tolerant Accuracy/F1 的细粒度评测。

**📊 数据集**

使用的数据集为 RecencyQA（4,031 个问题），问题来源于 FreshQA、PATQA、SituatedQA 以及 LLaMA 自动生成的事件描述；每个问题都配有 12 类 recency 取样分布、stationarity 标签和经验证的时间上下文；此外在实验中还对 FreshQA、RealTimeQA 等现有时间敏感 QA 数据集做对比。

**📈 对比分析**

实验对比六大 LLM（Qwen‑2.5 7B/72B、LLaMA‑3 8B、Mistral‑24B、Gemma‑3 27B、Apertus‑70B）在零样、少样、链式思维三种 Prompting 下的 recency 分类准确率从 24%–52%（严格）不等，容差准确率最高可达 88%；对 stationarity 的判断准确率约 78%；在有上下文与无上下文的实验中，非平稳问题的性能提升可达 40%+，但平稳问题往往下降；再者，动态 recency 转移任务的 Transition Accuracy 仅 14% 左右，显示模型难以根据上下文实时更新答案。

**⚠️ 局限性**

局限性包括：①标注过程依赖 LLM 推理，可能带来偏差；②数据集规模相对有限，缺少跨领域、跨语言样本；③recency 仅分为 12 个离散类，未捕捉更细粒度变化；④评测聚焦单模型推理，未结合检索/补全等外部信息；⑤实验未检验模型在真实检索环境下的时效性表现。

---

## 136. Resource Consumption Threats in Large Language Models

**arXiv ID:** 2603.16068 | [PDF](https://arxiv.org/pdf/2603.16068v1)

**作者:** Yuanhe Zhang `[一作]` (Beijing University of Posts and Telecommunications), Sen Su `[通讯]` (Chongqing University of Posts and Telecommunications)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5101604487)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了大型语言模型在资源消耗方面的威胁，提出了统一的“过度思考/无界漂移”分类，并梳理了攻击手段、机制、对策和未来挑战。

**💡 创新点**

创新点在于构建了基于生成行为的效果导向分类框架，系统阐释了两种资源放大机制（过度思考与无界漂移）的内部动力，并对比评估了现有防御方法，提出了效率优化与安全防御的分离与标准化评估需求。

**🔧 技术方法**

主要技术包括文献综述、效应导向分类法、模型机制分析（如注意力偏移、诱导头、重复动力学）、训练目标调优（如不似然训练、重复丢弃）、解码控制策略以及系统级调度/监控框架。

**📊 数据集**

作为综述文章未使用自有实验数据，主要引用了公开模型与公开数据集（如OpenAI GPT、DeepSeek、通用文本/多模态数据）中各研究的实验结果。

**📈 对比分析**

通过对比现有攻击与防御文献，评估框架涉及模型层面行为、硬件压力与应用服务影响；虽缺乏统一量化指标，但总结指出诸如不似然训练、循环检测等方法在抑制重复和控制生成长度方面取得了显著效果。

**⚠️ 局限性**

局限性包括仅聚焦资源放大攻击而非整体效率/鲁棒性问题；分类法可能无法覆盖所有混合或跨模态攻击；研究主要集中于文本LLM与少数开源模型，对黑盒商用系统、音视频模态与代理系统的覆盖不足；评估指标不统一，导致跨论文比较受限。

---

## 137. Dimensional Type Systems and Deterministic Memory Management: Design-Time Semantic Preservation in Native Compilation

**arXiv ID:** 2603.16437 | [PDF](https://arxiv.org/pdf/2603.16437v1)

**作者:** Houston Haynes `[一作]` `[通讯]`, Houston Haynes

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出一种在多阶段编译器中将维度信息持久化的维度类型系统（DTS）与确定性内存管理（DMM）相结合的框架，利用程序语义图（PSG）实现统一的维度推理、表示选择和逃逸分析，从而在设计时提供内存分配、缓存行为和跨目标传输等反馈；

**💡 创新点**

创新点在于：①维度注解在整个编译过程中的持久化，突破传统的“早期擦除”限制；②将逃逸分析与内存共变形为coeffect，形成可在PSG中联合求解的表示选择与内存布局；③将此框架与MLIR多目标后端集成，实现针对不同硬件（CPU、FPGA、Neuromorphic）的自动格式与位置决策；

**🔧 技术方法**

主要技术包括：扩展Hindley–Milner统一的维度推理（基于可生成阿贝尔群的线性约束求解）、coeffect传播与逃逸分类、基于PSG的多目标表示选择函数、MLIR自定义属性与多重下沉路径、以及quie累加器等硬件相关的精确累加模型；

**📊 数据集**

论文未使用具体的实验数据集，主要以理论分析与概念模型为主；

**📈 对比分析**

由于缺乏实验评估，本文未给出定量的性能对比或基准结果，只提出理论上可通过维度信息实现更高效的表示选择与内存布局，暗示在不同目标上可获得精度与速度的平衡；

**⚠️ 局限性**

限制包括：①维度推理仅支持有限生成阿贝尔群，无法表达更一般的谓词；②未提供形式化可判定性证明和完整性证明；③对大规模程序的计算复杂度及PSG存储需求未作评估；④实际性能收益仍待实验验证。

---

## 138. Evo-Retriever: LLM-Guided Curriculum Evolution with Viewpoint-Pathway Collaboration for Multimodal Document Retrieval

**arXiv ID:** 2603.16455 | [PDF](https://arxiv.org/pdf/2603.16455v1)

**作者:** Weiqing Li `[一作]` (Alibaba Cloud Computing), Hao Henry Wang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发了 Evo‑Retriever，一种结合多视角对齐、双向对比学习和 LLM 引导的进化课程的视觉语言检索框架。

**💡 创新点**

通过多视角对齐增强空间感知、双向对比学习消除文本混淆，并用 LLM 元控制器动态调整难度，实现模型与课程共进化。

**🔧 技术方法**

多视角图像对齐 (MVA)、双向对比学习 (BCL)、LLM 进化课程 (LLM‑EC)、多向量延迟交互、LoRA 微调、Qwen‑VL 等。

**📊 数据集**

ViDoRe V2、MMEB（VisDoc）、ColPali、VisRAG‑Ret‑Train‑Synthetic、VisRAG‑Ret‑Train‑In‑domain 等。

**📈 对比分析**

与 ColPali、Llama‑Nemoretriever 等基线比较，在 ViDoRe V2 上 nDCG@5 65.2%，在 MMEB 上 77.1%，均刷新 SOTA。

**⚠️ 局限性**

依赖昂贵的 LLM 评估与元控制，训练过程复杂且对硬件资源要求高；对极端多语言或极稀有布局的泛化尚未彻底验证。

---

## 139. A Family of LLMs Liberated from Static Vocabularies

**arXiv ID:** 2603.15953 | [PDF](https://arxiv.org/pdf/2603.15953v1)

**作者:** Aleph Alpha `[一作]` (Aleph Alpha Research), Gregor Ziegltrum `[通讯]` (Aleph Alpha Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并训练了基于层次自回归Transformer（HAT）的无分词器大型语言模型（7B、8B和70B），通过将字节编码为单词级嵌入，利用预训练的Llama 3.1权重实现模型迁移。

**💡 创新点**

创新点在于将传统分词器完全替换为基于规则的字节拆分，构建了三层结构（编码器、主体、解码器），从而实现更高的压缩率、鲁棒性和对新语言/领域的适应性。

**🔧 技术方法**

技术包括字节级局部因果注意力、跨层交叉注意力、使用SwiGLU的Transformer块、BPE替换为字节拆分、vLLM高效推理、以及DPO与SFT对齐训练。

**📊 数据集**

数据集涵盖英语、德语、数学与编程的混合语料，使用人工与生成的对话与指令数据，约4万亿字（预训练）与2百万样本（SFT/DPO）。

**📈 对比分析**

在多语言知识、推理与指令跟随等基准上，与Llama 3.1及其Tulu版本相比，HAT模型在大多数任务上表现相当或更优，同时实现了更好的压缩率与参数减少。

**⚠️ 局限性**

局限性包括：未针对代码生成或数学推理进行优化，推理速度受限于层次结构；缺乏对低资源语言的广泛评估；以及对极长上下文的进一步效率提升仍需研究。

---

## 140. Evaluating Causal Discovery Algorithms for Path-Specific Fairness and Utility in Healthcare

**arXiv ID:** 2603.15926 | [PDF](https://arxiv.org/pdf/2603.15926v1)

**作者:** Nitish Nagesh `[一作]` (University of California Irvine), Amir M. Rahmani `[通讯]` (University of California Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建专家主导的因果图基准，评估多种因果发现算法在结构恢复与路径级公平性上的表现。

**💡 创新点**

将结构发现与路径级公平性分解结合，并在真实临床数据上提出基于专家图的公平性评估框架。

**🔧 技术方法**

使用PC、GES、FCI、NOTEARS、DAGMA、DAG‑GNN等因果发现算法，结合CFA分解和CFUR指标。

**📊 数据集**

在合成阿尔茨海默病数据（1000样本）和真实心衰临床记录（299例）上进行实验。

**📈 对比分析**

通过F1、SHD、FDR等结构指标和路径级公平性量化，PC在合成数据上结构最优，FCI在心衰数据上结构和公平性恢复最佳。

**⚠️ 局限性**

受限于专家图主观性、样本量小、可能存在隐含循环和未观测混杂，且仅评估单一受保护属性。

---

## 141. Leveraging LLMs for Structured Information Extraction and Analysis from Cloud Incident Reports (Work In Progress Paper)

**arXiv ID:** 2603.16818 | [PDF](https://arxiv.org/pdf/2603.16818v1)

**作者:** Xiaoyu Chu `[一作]` (Vrije Universiteit Amsterdam), Alexandru Iosup `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 9020 | [OpenAlex ID](https://openalex.org/A5006986556)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

收集并手工标注了约3000条AWS、Azure、GCP云事件报告，构建开源数据集与工具箱，并使用LLM对报告进行结构化信息抽取与分析。

**💡 创新点**

首次公开云事件报告标注数据与LLM抽取工具箱，系统比较6种LLM与6种提示策略，提供准确性、延迟、成本平衡建议，并基于LLM实现高效可扩展的事件信息抽取。

**🔧 技术方法**

采用大语言模型（Gemini 2.0/2.5、GPT‑3.5/4o、Claude 4）配合Chain‑of‑Thought、In‑Context Learning、提示工程，多种提示策略；使用BERTScore、Token‑F1、Exact‑Match评估抽取质量。

**📊 数据集**

从AWS、Azure、GCP收集约3000条公共事件报告，标注约460条（15%），包含服务名、位置、用户症状、根因等字段。

**📈 对比分析**

对6种模型、6种提示策略在3个云服务数据集上评估准确率、延迟与成本；轻量模型Gemini 2.0/ GPT‑3.5在准确率（75–95%）与成本/延迟上表现最佳；大模型准确率略高但成本和延迟显著增加。

**⚠️ 局限性**

受限于少量标注样本、分类闭合世界假设、报告信息缺失（如根因细节）导致抽取深度有限，few‑shot示例效果不稳定且需进一步优化。

---

## 142. Prompt Programming for Cultural Bias and Alignment of Large Language Models

**arXiv ID:** 2603.16827 | [PDF](https://arxiv.org/pdf/2603.16827v1)

**作者:** Maksim Eren `[一作]`, Johnny Seales `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

验证并扩展Tao等人基于社会科学问卷的文化对齐框架，在开源LLM上进行实验，并引入DSPy提示程序化方法进一步优化文化对齐

**💡 创新点**

将文化对齐评价迁移到开源模型，并将提示优化转化为可编程DSPy任务，利用自动化搜索显著降低文化距离

**🔧 技术方法**

使用DSPy（COPRO、MIPROv2）提示程序化技术，结合PCA映射至Inglehart–Welzel文化地图

**📊 数据集**

使用IVS/World Values Survey问卷（10个指标）构造人类基准坐标，并与模型输出对齐

**📈 对比分析**

与无文化提示、手工文化提示三种方式对比，结果显示MIPROv2+GPT-OSS 120B在大多数模型上平均缩短文化距离，提升幅度可达数个百分点

**⚠️ 局限性**

局限在于仅评估强制选择问卷，未覆盖开放式生成、多轮对话及多语种情况，且优化效果在部分国家仍不理想

---

## 143. InCoder-32B: Code Foundation Model for Industrial Scenarios

**arXiv ID:** 2603.16790 | [PDF](https://arxiv.org/pdf/2603.16790v1)

**作者:** Jian Yang `[一作]` (Beihang University), Weifeng Lv `[通讯]` (Beihang University)

**通讯引用:** 5956 | [OpenAlex ID](https://openalex.org/A5109299440)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一款针对工业软件开发设计的 32B 参数 LLM，统一覆盖芯片设计、GPU 核心优化、嵌入式系统、编译器优化和 3D 建模等工业领域；

**💡 创新点**

创新点在于：①设计了三阶段 Code‑Flow 训练流程，结合工业代码的自动化验证与执行反馈；②将“思考路径”与“指令调优”两种能力融合到同一模型；③构建了最全面的工业代码评测集合，涵盖 14 项通用基准与 9 项工业基准；

**🔧 技术方法**

采用基于 Transformer 的自回归语言模型，配合 Fill‑in‑The‑Middle (FIM) 预训练，使用循环架构提升长文本处理能力；中间训练阶段通过上下文长度扩展（8K→128K）与合成工业推理 QA、代理轨迹；后期训练利用执行‑基于验证（RTL、CUDA、Renode、OpenCascade 等）生成 SFT 数据；

**📊 数据集**

数据来源包括：①公开仓库与技术文献 OCR 收集的工业代码；②合成的工业 QA 与代理轨迹；③2.5M 条执行验证后工业任务样本，涵盖芯片、GPU、嵌入式、编译器四大领域；

**📈 对比分析**

与众多开源与闭源基线（DeepSeek‑V3.2、Qwen 系列、Kimi‑Dev、Claude‑Sonnet‑4.6 等）在 14 通用基准和 9 工业基准上对比；在通用任务上达到 74.8% SWE‑bench Verified、49.14% LiveCodeBench、60.99% BFCL；在工业任务中实现 RealBench 模块级最高开源成绩、CAD‑Coder、KernelBench 领先，甚至超越 Claude‑Sonnet‑4.6 在 CAD‑Coder IoU 与 KernelBench L1/L2/L3 上的表现；

**⚠️ 局限性**

局限性包括：①Verilog/HDL 仍有高比例编译/语法错误；②功能/逻辑错误在 VeriRepair、ArchXBench 中占比高；③优化任务中速度提升有限，KernelBench 与 SuperCoder 的性能提升不足；④对工业 API 细节掌握不完整，导致链接器或 API 调用错误；⑤随着 SFT 数据规模增大，验证相关性能提升趋于饱和，进一步提升仍受数据质量与工程成本限制。

---

## 144. Structured Semantic Cloaking for Jailbreak Attacks on Large Language Models

**arXiv ID:** 2603.16192 | [PDF](https://arxiv.org/pdf/2603.16192v1)

**作者:** Xiaobing Sun `[一作]` (Agency for Science Technology and Research), Liangli Zhen `[通讯]` (Agency for Science Technology and Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种多维结构化语义伪装（S2C）框架，通过在语境重构、内容碎片化与线索引导伪装三维空间中分散重构恶意语义，推迟模型安全触发，从而实现对大型语言模型的有效 jailbreak 攻击。

**💡 创新点**

创新点在于将恶意语义在结构层面上进行多维分散与重构，突破传统单维表面混淆或重写方法，利用长程共指推理与多步生成来削弱安全机制的触发。

**🔧 技术方法**

技术包括语境重构、内容碎片化、线索引导伪装；以及字符噪声、反转、Base64、凯撒密码、词语替换等混淆与编码方法。

**📊 数据集**

使用 JBB‑Behaviors 与 HarmBench 两个公开恶意查询数据集进行评估。

**📈 对比分析**

与 7 种现有基线（CodeChameleon、ReNeLLM、PAIR、PAP、Cipher、WBP、HaPLa）在 16 款开源与商用 LLM 上对比，S2C 在平均 Attack Success Rate 上提升 9.7–50.5%，在 JBB‑Behaviors 上平均 89.4%，在 HarmBench 上 94.1%；在 GPT‑5‑mini 上仍达 50%。

**⚠️ 局限性**

局限性包括仅在单轮对话场景评估、固定线索组合与手工挑选的混淆策略、单一评判模型、未给置信区间，且对更强模型（如更高级推理或多轮交互模型）的鲁棒性尚未充分验证。

---

## 145. Regularized Latent Dynamics Prediction is a Strong Baseline For Behavioral Foundation Models

**arXiv ID:** 2603.15857 | [PDF](https://arxiv.org/pdf/2603.15857v1)

**作者:** Pranaya Jajoo `[一作]` (University of Alberta), Martha White `[通讯]` (University of Alberta)

**通讯引用:** 1545 | [OpenAlex ID](https://openalex.org/A5101613484)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种正则化的潜在动力学预测方法，用于无监督地学习行为基础模型（BFM）的状态表示。

**💡 创新点**

创新点在于引入正交正则化以防止特征坍塌，并通过不使用贝尔曼回溯的简单预测目标实现更稳健的表示。

**🔧 技术方法**

采用潜在状态预测、正交正则化、后继特征估计和行为强化学习等技术，构建无监督的BFM。

**📊 数据集**

实验使用了ExoRL、DeepMind Control Suite（Pointmass、Cheetah、Walker、Quadruped）、SMPL Humanoid以及D4RL MuJoCo等数据集。

**📈 对比分析**

在离线和在线零射RL任务中，方法与FB、PSM、HILP等基线相比表现相当甚至更好，尤其在低覆盖数据集上可实现10–40% 的性能提升。

**⚠️ 局限性**

局限性包括对某些任务的性能仍低于专家策略，且在极高维度动作空间或极端低覆盖情形下效果可能下降。

---

## 146. The Comprehension-Gated Agent Economy: A Robustness-First Architecture for AI Economic Agency

**arXiv ID:** 2603.15639 | [PDF](https://arxiv.org/pdf/2603.15639v1)

**作者:** Rahul Baxi `[一作]` `[通讯]` (Smartypans Inc), Rahul Baxi (Smartypans Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“理解门控代理经济”架构，依据对AI代理的三维对抗鲁棒性评估（约束遵守、认知完整性、行为一致性）来动态限制其经济权限。

**💡 创新点**

创新点在于将鲁棒性检验与经济权限严格绑定，采用最弱链最小化门控、时间衰减与随机重审机制，实现经济曝光可控、鲁棒性投资具有激励兼容性以及系统规模增长不降低安全性。

**🔧 技术方法**

核心技术包括：对抗压缩理解测试（CDCT）、深度问答与伪造陷阱评估（DDFT）、对抗式对话行为评估（AGT），以及基于熵的内部幻觉率估算；门控函数采用最小化步骤函数实现离散经济层级。

**📊 数据集**

实验数据来源于公开的CDCT、DDFT与AGT测试集，覆盖多种压缩级别、对抗回合与伦理情境，使用顶级模型（GPT‑5、O3 等）验证鲁棒性维度的正交性。

**📈 对比分析**

方法通过形式化证明验证三项性质：曝光上限、鲁棒性投资激励兼容、系统规模增长安全性递增；实验表明鲁棒性各维度均独立于参数规模，且不同模型在不同维度的表现差异显著。

**⚠️ 局限性**

局限包括：只能对可机理可验证的任务进行经济化，无法覆盖创意、策略咨询等非正式化任务；审核成本高、对多代理协同攻击的分析尚不完善；阈值设置需经验校准，易受 Goodhart 现象影响。

---

## 147. Manifold-Matching Autoencoders

**arXiv ID:** 2603.16568 | [PDF](https://arxiv.org/pdf/2603.16568v1)

**作者:** Laurent Cheret `[一作]` (University of Ottawa), Maia Fraser `[通讯]` (University of Ottawa)

**通讯引用:** 265 | [OpenAlex ID](https://openalex.org/A5068740847)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种自监督正则化方法Manifold-Matching Autoencoder（MMAE），通过对比潜在空间与参考空间的配对距离来保持全局几何结构。

**💡 创新点**

创新点在于使用配对距离矩阵而非坐标进行对齐，可与任何低维参考embedding（如PCA、UMAP等）结合，既能保持全局几何，又能实现对非参数降维结果的复制与扩展。

**🔧 技术方法**

技术上基于标准autoencoder加上重构MSE损失和Manifold-Matching距离对齐损失；采用批量训练、距离矩阵计算，并在高维数据上通过PCA降维作为参考。

**📊 数据集**

实验使用合成数据（Nested Spheres、Linked Tori、Concentric Spheres、Mammoth、Earth）以及真实数据（MNIST、Fashion-MNIST、CIFAR-10、PBMC3K、Paul15）。

**📈 对比分析**

与TopoAE、RTD-AE、GeomAE、GGAE、SPAE、Vanilla AE等方法比较，利用DC、TA、KL_density、Trustworthiness、Continuity、Wasserstein persistence等指标；MMAE在大多数指标上优于或与现有方法持平，尤其在保持拓扑连通性和全局几何方面表现突出。

**⚠️ 局限性**

局限在于仅通过距离对齐，未显式保证完整的拓扑一致性；对高噪声高维数据需依赖PCA等预处理；批量尺寸增大时仍有内存开销；无法完整捕捉多尺度拓扑特征。

---

## 148. inference-fleet-sim: A Queueing-Theory-Grounded Fleet Capacity Planner for LLM Inference

**arXiv ID:** 2603.16054 | [PDF](https://arxiv.org/pdf/2603.16054v1)

**作者:** Huamin Chen `[一作]` (vLLM Semantic Router Project), Xue Liu `[通讯]` (MBZUAI)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种两阶段的LLM GPU舰队容量规划工具，结合M/G/c排队理论与离散事件模拟，自动计算满足P99 TTFT SLO的最低成本GPU配置；

**💡 创新点**

创新点在于：1）首次将排队模型与离散事件仿真联合用于fleet层面的多池、多GPU类型规划；2）引入物理感知的GPU性能模型和可靠性惰性模型；3）对多种路由策略和离散化部署（prefill/decode）进行统一优化；4）提供对电网需求响应（DR）下的功耗可调分析；

**🔧 技术方法**

使用了M/G/c两矩近似、Kimura逼近、离散事件仿真、GPU性能参数（W,H,n_max）、可靠性系数A、物流功耗模型等技术；

**📊 数据集**

使用了公开的LMSYS和Azure LLM轨迹的token长度CDF，以及合成的agent-heavy轨迹；

**📈 对比分析**

通过与现有工具（Vidur、Mélange、DistServe等）对比，展示了在不同场景下的成本、延迟和SLO满足率，实验结果表明传统单一分析往往高估性能或低估成本，而本文方法能在保证SLO的前提下显著降低成本；

**⚠️ 局限性**

主要局限包括：1）Poisson子流分流近似不严格；2）仿真仅在请求级别，不考虑细粒度调度细节；3）GPU模型采用线性屋顶线，未捕捉非线性加速效果；4）不建模多节点张量并行的通信开销；5）功耗模型对深度批处理缺乏充分数据支持。

---

## 149. This Is Taking Too Long -- Investigating Time as a Proxy for Energy Consumption of LLMs

**arXiv ID:** 2603.15699 | [PDF](https://arxiv.org/pdf/2603.15699v1)

**作者:** Lars Krupp `[一作]` (DFKI and RPTU), Jakob Karolus `[通讯]` (DFKI and RPTU)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过测量本地GPU与API接口的推理时间，评估时间是否可以作为大型语言模型（LLM）能耗的代理，并据此估算API模型的能源消耗。

**💡 创新点**

提出将推理时间与每个Token的耗时相结合，用来推断后端GPU配置并估算API LLM的能耗，为终端用户提供可解释的能源评估方法。

**🔧 技术方法**

使用本地GPU基准测试、API推理计时、CarbonTracker能耗监测、TDP计算、FP16/TF32推理、Token化、温度设定、随机种子同步等技术，构建统一的实验框架。

**📊 数据集**

生成的合成提示集（4种任务类型、短/长序列，共100个提示），由llama3‑70b模型生成，用于在不同GPU和API上进行推理时间和能耗测量。

**📈 对比分析**

在A100、H100、H200等GPU上进行10次重复实验，比较本地与API的平均时间、能耗与TDP估算。结果显示API推理时间显著短于本地运行，且与Hopper系列GPU的每Token耗时相近；能耗估算与TDP相差20–40%，API模型的能耗约为每Token0.7–1.2mWh，总耗能约为100–200Wh。

**⚠️ 局限性**

假设推理时间主要由GPU计算决定，忽略多GPU并行、管道并行、I/O延迟等因素；实验仅覆盖单GPU、单批量、固定模型大小，未考虑不同部署优化，导致估算精度受限。

---

## 150. VibeContract: The Missing Quality Assurance Piece in Vibe Coding

**arXiv ID:** 2603.15691 | [PDF](https://arxiv.org/pdf/2603.15691v1)

**作者:** Song Wang `[一作]` (York University), Song Wang `[通讯]` (York University)

**通讯引用:** 72014 | [OpenAlex ID](https://openalex.org/A5100460802)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将 Design by Contract 原则嵌入 Vibe Coding 流程，形成四步闭环（意图分解、合同生成与验证、合同驱动代码生成、合同驱动测试），在 LLM 代码生成中实时引入任务级合同以提升可靠性与可追溯性。

**💡 创新点**

创新点在于将合同生成与验证直接插入 Vibe Coding 的工作流中，创建预防性、可自动化的 QA 机制，将传统的后期人工检查转变为实时、可追溯的合同驱动质量保障；同时通过合同指导代码生成和测试形成闭环。

**🔧 技术方法**

使用大型语言模型（如 GPT‑5.2 Instant）进行意图分解、合同生成和代码生成；采用 Design by Contract 的可执行合同（类似 JML/Eiffel）来规范输入输出与行为；利用链式思考、合同驱动指导和反馈循环实现代码生成与测试。

**📊 数据集**

论文未采用公开数据集，而是以自定义的 ATM 系统 Java 示例（包含 ATM、Account、Bank、Main 四类）作为实验项目，并通过内部复制包中的代码进行验证。

**📈 对比分析**

通过与传统 Vibe Coding 的对比，展示生成代码前后出现的逻辑缺陷（如余额不检验、NaN/Infinite 等），并在合同驱动流程中消除这些错误；案例证明更高的正确性和可维护性，但未给出量化性能指标，只通过实例演示改进效果。

**⚠️ 局限性**

主要局限在于合同仍需人工验证，合同合成与完善尚未完全自动化；合同生成和执行的跨语言/运行环境支持有限；运行时契约检查的性能开销与实现复杂度尚未评估；缺乏大规模实验和定量评估。

---

## 151. CoMAI: A Collaborative Multi-Agent Framework for Robust and Equitable Interview Evaluation

**arXiv ID:** 2603.16215 | [PDF](https://arxiv.org/pdf/2603.16215v1)

**作者:** Gengxin Sun `[一作]` (Shandong University), Zhiwei Xu `[通讯]` (Shandong University)

**通讯引用:** 2514 | [OpenAlex ID](https://openalex.org/A5100782604)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套名为 CoMAI 的协同多智能体面试框架，专为高端人才评估设计，能够在不需额外训练的情况下实现问答生成、答题安全监测、评分与总结的自动化。

**💡 创新点**

核心创新在于采用集中式有限状态机协调四个角色专门化智能体（问答生成、安全检测、评分、总结），实现模块化、可审计的评估流程，并通过分层安全和面向评分的 rubric 设计消除多模态、偏差与提示注入风险。

**🔧 技术方法**

技术实现基于大型语言模型（GPT‑5‑mini、Qwen‑plus‑2025‑07‑28、Kimi‑K2‑Instruct），并在微服务架构下通过消息队列与 RESTful 接口实现异步通信；使用规则 + 语义两层安全过滤、面向评分的分步 rubric、动态难度调整、以及基于 FSM 的控制流。

**📊 数据集**

实验使用了 55 名来自不同学科背景的候选人数据，采集候选人简历、答复、系统日志等信息，并与专家评审（10 名顶尖大学教授）和外部 AI 及人工面试员的评判结果进行对标。

**📈 对比分析**

与单模态基线、人类面试官以及公开的 LLM‑Interviewer、AI‑Interviewer‑Bot v3 等对比，CoMAI 在评估准确率、召回率、评分分布、攻击防御成功率（100%）以及用户体验（满意度 84%）等多项指标上均优于对照组，且在不同模型间保持高度一致性。

**⚠️ 局限性**

主要局限包括：多智能体协同带来的调度开销和响应延迟、调试与性能优化复杂度、对评分 rubric 的依赖导致维度覆盖不足、缺乏非语言交互支持以及较高的算力与成本消耗。

---

## 152. IndexRAG: Bridging Facts for Cross-Document Reasoning at Index Time

**arXiv ID:** 2603.16415 | [PDF](https://arxiv.org/pdf/2603.16415v1)

**作者:** Zhenghua Bao `[一作]` (Continuum AI), Yi Shi `[通讯]` (Continuum AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出IndexRAG，通过在离线索引阶段生成可检索的桥接事实，将跨文档推理转移到索引阶段，检索时仅需单次LLM调用即可完成多跳问答。

**💡 创新点**

创新点在于：1）把跨文档推理从在线推理迁移到离线索引；2）通过LLM自动生成桥接实体与桥接事实，使跨文档关联成为可检索的单元；3）无需图遍历、查询分解或多轮检索，显著降低在线计算成本。

**🔧 技术方法**

使用技术包括：LLM（GPT‑4o‑mini）用于原子知识单元和实体抽取、桥接实体识别、桥接事实生成；向量检索（FAISS + text‑embedding‑3‑small）存储AKU与桥接事实；平衡上下文选择机制控制检索结果比例；可与IRCoT等迭代检索方法结合。

**📊 数据集**

实验数据集：HotpotQA、2WikiMultiHopQA、MuSiQue（各取1000条验证样本）。

**📈 对比分析**

与方法比较：在单次LLM调用的基线中，IndexRAG平均F1 51.7，显著高于Naive RAG（47.1）、FastGraphRAG（49.4）和RAPTOR（47.0）；与多轮方法比较时，IRCoT+IndexRAG达55.0，优于HippoRAG（54.1）且仅需单次检索+单次LLM调用；检索延迟仅0.30s，几乎与Naive RAG相当。

**⚠️ 局限性**

限制：桥接事实质量依赖LLM的生成，可能出现幻觉或误导；实体抽取直接使用LLM，可能漏检或误检；目前仅在英文多跳QA数据集上评估，跨语言、跨领域的泛化尚未验证。

---

## 153. Search2Motion: Training-Free Object-Level Motion Control via Attention-Consensus Search

**arXiv ID:** 2603.16711 | [PDF](https://arxiv.org/pdf/2603.16711v1)

**作者:** Sainan Liu `[一作]` (Intel Corporation), Subarna Tripathi `[通讯]` (Intel Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练自由的框架 Search2Motion（S2M），通过先-后帧（FLF2V）控制，实现单图像到视频的对象级运动编辑；同时给出一种基于注意力一致性的种子选择方法 ACE‑Seed，提升生成质量。

**💡 创新点**

创新点：①将对象运动编辑重新表述为 FLF2V 任务，用户仅需给定目标位置；②语义引导的对象放置 + 背景修复，实现可靠的目标帧构造；③基于早期自注意力一致性的种子搜索，避免昂贵的前瞻采样；④设计专门的稳定相机、仅对象运动基准（S2M‑DAVIS、S2M‑OMB）和对应的对象级评估指标（FLF2V‑obj）。

**🔧 技术方法**

技术包括：VLM（如 Qwen2.5‑VL）+ SAM2 进行语义引导的对象定位；背景修复/图像编辑模型；预训练的 FLF2V 视频扩散模型（VACE‑1.3B、Wan2.2‑5B）；DiT 注意力提取与聚合；注意力一致性评分（余弦相似度）；对象级一致性评估（LPIPS、DINOv2）。

**📊 数据集**

使用了两组基准数据：S2M‑DAVIS（来自 DAVIS2017‑test）和 S2M‑OMB（来自 ObjMove‑B），两者均在静态背景下生成首末帧对；并在公开的 VBench 以及自建的 FLF2V‑obj 指标上评估。

**📈 对比分析**

在 VBench 指标上，S2M‑VACE 和 S2M‑Wan2.2‑5B 的表现均超过 DragAnything 与 TTM，尤其在视觉质量与运动流畅度方面提升明显；在 FLF2V‑obj 指标（DINOv2、LPIPS、目标中心距离、IoU）上也优于基线，显示出更好的对象定位与姿态保真度。ACE‑Seed 的种子搜索能将低质量样本过滤，显著提升整体视频质量，且与人类偏好保持一致。

**⚠️ 局限性**

局限性：①仅适用于静态背景场景，无法处理复杂相机运动；②依赖 FLF2V 模型的先前知识，若模型更新需重新验证；③早期注意力一致性仅在中深层块有效，可能不适用于所有扩散架构；④语义引导的对象放置与背景修复对模型质量仍有一定影响，且对极端遮挡/光照变化的鲁棒性有限。

---

## 154. IRIS: A Real-World Benchmark for Inverse Recovery and Identification of Physical Dynamic Systems from Monocular Video

**arXiv ID:** 2603.16432 | [PDF](https://arxiv.org/pdf/2603.16432v1)

**作者:** Rasul Khanbayov `[一作]` (Hamad Bin Khalifa University), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 467 | [OpenAlex ID](https://openalex.org/A5070970331)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了IRIS，一套用于从单目视频逆推物理参数的高保真真实世界基准数据集，包含220段4K分辨率、60fps的视频，涵盖单体与多体动力学，并附带独立测量的参数和不确定性；同时给出了五轴评价协议及多种基线实现；

**💡 创新点**

创新点包括：①首次在真实视频上提供多体相互作用场景并配备精确物理参数；②构建统一的五轴评估框架；③系统对四种方程识别策略（VLM时序推理、描述-分类、CNN、路径标签）进行基线比较；④发现并修复现有潜在空间模型的梯度流缺陷；

**🔧 技术方法**

使用的技术包括：潜在空间动力学管道、Euler积分器与多步滚动损失、不同VLM与CNN方程识别策略、基于Ode库的方程候选集、以及潜在-SI 单位校准；

**📊 数据集**

数据集：IRIS benchmark（220段4K视频，涵盖8种动力学现象，含10次重复，每段30秒，配有物理参数GT及不确定性）；

**📈 对比分析**

与基线比较：CNN分类器在IRIS上方程识别精度达99.3%；VLM时序推理与描述-分类的准确率分别为65%和73%；多步滚动损失在单体动力学中提升参数可识别性，但在多体场景导致参数误差爆炸；基线整体性能仍低，凸显该领域挑战；

**⚠️ 局限性**

局限性包括：①多体动力学采用简化的连续耦合项，缺乏真实碰撞模型；②多步训练在多体情景下不稳定；③阻尼系数难以辨识且GT来源于拟合；④潜在空间到SI单位的校准可能引入偏差；⑤数据规模有限，未覆盖更复杂物理现象；

---

## 155. Interpretable Context Methodology: Folder Structure as Agentic Architecture

**arXiv ID:** 2603.16021 | [PDF](https://arxiv.org/pdf/2603.16021v1)

**作者:** Jake Van Clief `[一作]` (Eduba University of Edinburgh), David McDermott `[通讯]` (Eduba University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出Model Workspace Protocol (MWP)，通过文件系统层次结构和Markdown文件来组织多阶段AI代理工作流，替代传统的多代理框架。

**💡 创新点**

创新点包括：① 用文件夹层级代替代码化的协调；② 五层上下文分离（身份、路由、阶段合同、参考材料、工作产物）；③ 每阶段输出为可编辑文本文件，提供自然的审阅门；④ 与本地脚本无缝协作，避免不必要的AI调用；⑤ 结构化可移植、可追溯，降低开发与运维成本。

**🔧 技术方法**

技术手段：Unix pipeline、Make、pipe‑and‑filter 原理；Markdown 与 JSON 作为通用文本接口；本地 Python 脚本处理非AI任务；Claude Code（Opus 4.6 + Sonnet 4.6）作为示例 LLM；文件系统作为状态与上下文管理器；不依赖任何专有框架或数据库。

**📊 数据集**

未使用公开数据集；实验基于内部案例（脚本‑动画管线、幻灯片制作、学术与政策分析工作流）进行演示与验证。

**📈 对比分析**

缺乏正式对比实验；通过社区实践（52 名成员）收集自述，观察到 U‑形编辑模式、低门槛修改、文件复制重用等优势；未给出量化指标或与现有框架的性能对比。

**⚠️ 局限性**

局限性：① 仅在单模型（Claude Opus）上测试，跨模型评估缺失；② 仅适用于顺序、可审阅、可重复的工作流，无法处理实时多代理、并发或自动分支；③ 缺乏系统性实验与统计验证；④ 对大型上下文的优势在未来大模型（如 200k 令牌）下可能减弱；⑤ 目前仅提供可观察性，缺乏自动化调试与追踪机制。

---

## 156. Persona-Conditioned Risk Behavior in Large Language Models: A Simulated Gambling Study with GPT-4.1

**arXiv ID:** 2603.15831 | [PDF](https://arxiv.org/pdf/2603.15831v1)

**作者:** Sankalp Dubedy `[一作]` `[通讯]` (Independent Researcher), Sankalp Dubedy (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在三种不同社会经济角色（富裕、中等、贫困）下，使用GPT‑4.1在可配置概率的老虎机环境中进行50次独立实验，总共记录了6,950个决策。

**💡 创新点**

首次通过人格化提示展示LLM能无须显式指令就自动复制前景理论的风险偏好，并揭示其情绪标记为后置叙述而非因果驱动。

**🔧 技术方法**

采用GPT‑4.1生成结构化JSON输出，结合非参数统计（Kruskal‑Wallis、Mann‑Whitney、ANOVA）和效应量评估来量化行为差异。

**📊 数据集**

实验数据来源为9个条件（3人设×3机型）的决策记录，涵盖投注金额、风险分数、置信度、情绪标签、策略模式等多维度信息。

**📈 对比分析**

通过多重比较检验和效应量（r≈1.0、d≈4.15）证明各人设在游戏时长、风险分数和投注比例等指标上显著区分，然而情绪与策略之间的关联仅为后置叙述，未形成因果性。

**⚠️ 局限性**

局限包括仅测试单一模型、默认温度1.0、缺乏跨会话记忆、情绪与决策的自报不可信、财富水平与投注比例混淆、未验证不同LLM或温度设定下的可重复性。

---

## 157. Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning

**arXiv ID:** 2603.15871 | [PDF](https://arxiv.org/pdf/2603.15871v1)

**作者:** Ezgi Korkmaz `[一作]` `[通讯]`, Ezgi Korkmaz

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于最小化状态动作价值函数的对抗式时序差分学习（CoAct TD），通过在采样时优先选择Q值最小的动作，显著提升时序差分误差，从而加速经验收集与学习。

**💡 创新点**

创新点在于：①提供了严谨的理论证明，说明最小Q动作能产生比随机动作更高的TD误差；②将该原理作为无额外计算成本的经验采集策略，能够无缝替换任何基于TD的强化学习算法；③在高维MDP下通过对抗式采样实现显著提升的样本效率。

**🔧 技术方法**

技术方法包括：深度Q学习（Double DQN、QRDQN）、优先经验回放、ε-greedy、UCB探索、噪声网络等；核心算法为CoAct TD，核心思想是先在缓冲区填充最小Q动作与最大Q动作的经验，再按原有TD更新。

**📊 数据集**

数据集主要为Arcade Learning Environment（ALE）26款游戏的100K交互量基准，以及200M帧（50M环境交互）训练集；在链式MDP（n=10）中也做了验证。

**📈 对比分析**

与基线ε-greedy、UCB、噪声网络等方法对比，CoAct TD在100K基准上实现中位数提升约248%，80%分位提升约25%，在200M帧训练中也显示更快收敛与更高最终性能；总体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：①理论证明基于Q值初始化的η-uninformed与δ-smooth假设，实际环境中可能不完全满足；②仅在离散动作空间中验证，连续控制任务尚未评估；③对极端大规模网络或超参数敏感度未系统探究。

---

## 158. Minimum Exposure Motion Planning

**arXiv ID:** 2603.16510 | [PDF](https://arxiv.org/pdf/2603.16510v1)

**作者:** Sarita de Berg `[一作]` (IT University of Copenhagen), Sampson Wong `[通讯]` (University of Copenhagen)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5029179453)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

研究单位正方形机器人在 L1 度量下的协同运动规划，提出新的 Min-Exposure 目标，并给出了两机器人时最优的 O(n⁴ log n) 算法以及任意固定机器人数的 FPT 算法。

**💡 创新点**

首次在平面上引入 Min-Exposure 目标，并证明 Min-Makespan 与 Min-Sum 在机器人数 k 上是固定参数可解的；为 k=2 给出 O(n⁴ log n) 的完整最优算法；为任意固定 k 给出 FPT 方案，显著改进了以往仅针对整数网格或特殊图形的结果。

**🔧 技术方法**

采用配置空间分区、转移图、线性规划、水平三角剖分、Minkowski 和等几何工具；构造图 G，利用可达性预处理（O(n²) 预处理、O(log n) 查询）求解最短路径，从而得到最优调度。

**📊 数据集**

本文未使用公开实验数据集，所有分析与算法均基于理论证明和多边形顶点数 n 的抽象模型。

**📈 对比分析**

通过理论复杂度分析与已知的 NP/PSPACE 难度结果对比，k=2 时算法达到 O(n⁴ log n) 的最优上界；任意固定 k 的算法属于 FPT，复杂度随 k 指数增长；相比现有仅在网格或特殊约束下可解的结果，显著扩大了可解决问题的范围。

**⚠️ 局限性**

限制在单位正方形机器人和 L1 度量；对更高维空间、非正方形形状或部分曝光度量的推广仍未完成；k 变大时 FPT 复杂度迅速膨胀，实际可行性有限；曝光目标采用二值化，未考虑部分覆盖的更细粒度度量。

---

## 159. EngGPT2: Sovereign, Efficient and Open Intelligence

**arXiv ID:** 2603.16430 | [PDF](https://arxiv.org/pdf/2603.16430v1)

**作者:** G. Ciarfaglia `[一作]`, I. Bailo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并发布了EngGPT 2 16B‑A3B，一个稀疏 Mixture‑of‑Experts（MoE）大型语言模型，旨在提供欧盟合规、低成本且多语言（英意）推理能力；

**💡 创新点**

创新点在于：将 16 B 参数模型与每次推理仅激活 3 B 专家相结合，采用 Grouped‑Query Attention（GQA）与 SwiGLU 激活，并设计多模式推理（无推理、完整推理、Turbo压缩推理）与嵌入式工具调用；

**🔧 技术方法**

使用了 Megatron‑LM + SmolLM3 自定义 MoE 框架、GQA、RoPE、RMSNorm、混合专家路由、Long‑Context 预训练、SFT、APO（DPO 变体）与 DeepSpeed ZeRO‑3 并行；

**📊 数据集**

训练数据覆盖 2.5 T 令牌，主要为公开 web、书籍、学术、代码、PDF、教育、数学等，约 25 % 为意大利语；mid‑training 用结构化推理语料，post‑training 用 SFT/APO 结合 distillation 语料；

**📈 对比分析**

通过统一生成设置与最佳配置双重评估，将 EngGPT 2 与 8 B‑12 B 稠密模型及 16 B‑30 B MoE 对标，结果显示其在推理、知识、数学与意大利语基准上接近甚至超越同等规模模型，且在训练/推理效率上处于上位；

**⚠️ 局限性**

局限包括：对长上下文任务的支持有限、工具调用与代码生成表现略逊、Turbo 推理在跨语言控制上易受提示语义影响、以及 SFT 时长与多样性不足导致某些能力欠缺。

---

## 160. Selective Memory for Artificial Intelligence: Write-Time Gating with Hierarchical Archiving

**arXiv ID:** 2603.15994 | [PDF](https://arxiv.org/pdf/2603.15994v1)

**作者:** Oliver Zahn `[一作]` (Independent Researcher), Simran Chana `[通讯]` (University of Cambridge)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5012655677)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出写时门控与分层归档的知识存储架构，在写入时根据来源声誉、信息新颖度和来源可靠度过滤知识对象，并通过版本链记录更新历史。

**💡 创新点**

创新点在于：①写时门控在知识进入存储前实现质量筛选，避免噪声污染；②冷存档而非直接删除，保持历史可追溯；③版本链支持时间推理；④在不依赖标签的情况下，仅用可观测信号即可达到高准确率。

**🔧 技术方法**

采用离散知识对象（Knowledge Object）与向量检索相结合，利用三维可观测信号计算复合显著性分数进行门控；使用哈希地址实现 O(1) 检索；对比 Self‑RAG 的读取时过滤；用 Claude Sonnet 4.6 进行真实 LLM 评估。

**📊 数据集**

实验数据包括：①50 条知识对象的合成基准（含 4:1 噪声比）；②20 个维基百科实体（共 73 条事实与 292 条干扰项）；③100 条程序生成的药理绑定亲和力（零参数知识）；④51 条 2026 年 arXiv 论文后的事实与 400 条干扰项。

**📈 对比分析**

与无门控检索和 Self‑RAG（读取时过滤）对比，写时门控在 4:1 噪声比下 100% 准确率，写时门控比 Self‑RAG 提升 6.2pp，且在 8:1 噪声比时 Self‑RAG 退化至 0% 而写时门控仍保持 100%。在维基百科数据上，写时门控 98.1%（相较无门控提升 12.9pp），在药理与 arXiv 领域优势分别达到 +64.6pp 与 +48.4pp。写时门控每次查询仅需一次 LLM 调用，成本比 Self‑RAG 低 9 倍。

**⚠️ 局限性**

局限包括：①基准规模仅 50 条知识对象；②门控依赖来源元数据，若元数据错误会导致误入噪声；③计算新颖度需与已存对象全量对比，时间复杂度为 O(n)；④未验证更高噪声比例（>8:1）或在极大规模语料库上的性能；⑤实验中 Self‑RAG 使用零样本 LLM 批评器，未对比训练好的 Self‑RAG。

---

## 161. AIDABench: AI Data Analytics Benchmark

**arXiv ID:** 2603.15636 | [PDF](https://arxiv.org/pdf/2603.15636v1)

**作者:** Yibo Yang `[一作]` (SenseTime Research), Wenxiu Sun `[通讯]` (SenseTime Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了 AIDABench，一个包含 600 多个端到端数据分析任务的基准，并在此基准上评估了 11 种主流 AI 模型。

**💡 创新点**

创新点在于：① 构建了覆盖问题解答、数据可视化和文件生成三大核心能力的真实场景任务集合；② 任务来源多样（电子表格、数据库、财务报表、运营记录等）；③ 引入人类+AI 评估方式，强调任务难度与实际工作时长一致；④ 为企业采购和模型优化提供系统参考。

**🔧 技术方法**

主要技术包括：大型语言模型（Claude‑Sonnet‑4‑5、Gemini‑3‑Pro‑Preview、Qwen3‑Max‑2026‑01‑23‑Thinking 等）在基准上的推理与自动化工具；人工评审和 pass@1 指标的结合；以及基准构建脚本和评测流水线。

**📊 数据集**

数据集：AIDABench 本身——600+ 真实任务，涵盖多种异构数据类型；未使用额外公开数据集。

**📈 对比分析**

比较方法：对 11 个模型采用 pass@1 评估，记录通过率；最优模型仅达 59.43% 的通过率，表明当前技术在复杂端到端数据分析任务上仍有较大提升空间。

**⚠️ 局限性**

限制：① 任务仍无法覆盖所有行业细节；② 评测流程对人力成本较高（人类专家需 1–2 小时完成一题）；③ 评估指标单一，缺乏多维度质量度量；④ 基准缺少自动化回归测试和持续更新机制。

---

## 162. Early-Terminable Energy-Safe Iterative Coupling for Parallel Simulation of Port-Hamiltonian Systems

**arXiv ID:** 2603.16424 | [PDF](https://arxiv.org/pdf/2603.16424v1)

**作者:** Qi Wei `[一作]` (Shanghai Jiao Tong University), Wangtao Tan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于散射变量的 Douglas–Rachford 迭代耦合方法，实现端口汉密尔顿子系统的能量安全并行仿真。

**💡 创新点**

创新点在于将迭代耦合映射嵌入散射坐标下的 Douglas–Rachford 分裂，使得任何有限迭代步数都能保证离散能量守恒，并给出增强存储不等式。

**🔧 技术方法**

采用散射（波）变量、Douglas–Rachford 分裂、凸优化中的最大单调算子理论以及离散梯度时间步进器。

**📊 数据集**

使用两耦合振荡器（含 Duffing 硬化弹簧）的仿真案例验证。

**📈 对比分析**

与单一全局离散时间更新做对比，结果显示随着内部迭代次数增大，误差单调下降，最终达到单机解；能量残差仅在数值舍入层面为正。

**⚠️ 局限性**

局限在于多解情况下的接口选择未给出理论保证、需手动调节阻抗参数 γ，且缺乏闭式的有限迭代误差上界。

---

## 163. Malicious Or Not: Adding Repository Context to Agent Skill Classification

**arXiv ID:** 2603.16572 | [PDF](https://arxiv.org/pdf/2603.16572v1)

**作者:** Florian Holzbauer `[一作]` (Interdisciplinary Transformation University), Johanna Ullrich `[通讯]` (Interdisciplinary Transformation University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 AI 代理技能生态进行大规模安全分析，收集并分析了 238,180 个独特技能，探讨了技能内容、终端点、秘密泄露等安全特征；

**💡 创新点**

引入仓库上下文（codebase 与 metadata）对已被扫描标记为恶意的技能进行再评估，显著降低误报率（从 46.8% 降至 0.52%），并发现新型攻击向量如被废弃仓库劫持；

**🔧 技术方法**

采用多种技术：静态代码分析（YARA、TruffleHog 等）、Cisco Skill Scanner、LLM（Codex/GPT-5.3）行为特征提取、仓库元数据分析及 LLM 评估代码与文档的一致性；

**📊 数据集**

使用从 ClawHub、Skills.sh、SkillDirectory 三大市场及 GitHub Archive 采集的 238,180 个技能数据集；

**📈 对比分析**

与五种扫描工具（ClawHub、Skills.sh、SkillDirectory、Cisco Skill Scanner、LLM 评估）进行对比，发现跨扫描器一致性极低（仅 0.12% 同时被全部标记），但仓库上下文评估后误报率降至 0.52%，说明仓库上下文能显著提升评估准确性；

**⚠️ 局限性**

局限性包括：对技能与仓库的匹配可能受索引失效或仓库被删除影响；LLM 评估受模型偏差和提示设计影响；未覆盖所有潜在的恶意行为，仅聚焦已知扫描器可检测的特征；

---

## 164. Cross-Scale Persistence Analysis of EM Side-Channels for Reference-Free Detection of Always-On Hardware Trojans

**arXiv ID:** 2603.16058 | [PDF](https://arxiv.org/pdf/2603.16058v1)

**作者:** Mahsa Tahghigh `[一作]` (Howard University), Hassan Salmani `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种无参考的基于跨尺度持久性分析的EM侧信道检测框架，用于识别始终开启的硬件木马。

**💡 创新点**

创新性在于使用多尺度稳定性图和受限高斯混合模型构建的跨尺度持久性指标（饱和比、方差、混合数中位数），无需黄金参考或标记数据即可区分正常与木马行为。

**🔧 技术方法**

核心技术包括EM侧信道采集、短时傅里叶变换多尺度时频表示、稳定性图构造、BIC选取的受限GMM建模以及跨尺度持久性指标计算。

**📊 数据集**

使用AES-128加密核的EM数据集，其中包含两种始终开启木马（泄漏信息HT和环振荡器HT），对固定密钥和500条明文进行多次执行并采集EM波形。

**📈 对比分析**

与无木马的基准对比，木马样本在所有尺度下显示高饱和比、低方差，表明其统计结构保持不变；实验在不同GMM容量下保持一致，说明方法对参数鲁棒；虽未给出精确准确率，但展示了对不同木马类别的可靠区分。

**⚠️ 局限性**

局限性在于对EM测量环境（探头位置、噪声）敏感，且受限GMM容量选择可能影响解释；未在多种工作负载或组合木马场景下验证，且目前为离线批处理，尚未实现实时在线部署。

---

## 165. Semi-supervised Latent Disentangled Diffusion Model for Textile Pattern Generation

**arXiv ID:** 2603.16747 | [PDF](https://arxiv.org/pdf/2603.16747v1)

**作者:** Chenggong Hu `[一作]` (Zhejiang University), Li Sun `[通讯]` (Zhejiang University)

**通讯引用:** 11449 | [OpenAlex ID](https://openalex.org/A5049415248)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了文本图案生成（TPG）任务及SLDDM-TPG模型，实现从服装图像生成高保真纹理图案；

**💡 创新点**

引入两阶段结构：先用潜在解耦网络（LDN）分离服装图像中的模式内容、纹理缺陷与结构特征，再用半监督潜在扩散模型（S‑LDM）在解耦特征指导下进行生成，并设计STD与CLS等对齐机制；

**🔧 技术方法**

利用SimSiam对比学习、逆向注意力、结构仿射变换、Stable Diffusion V1‑5基础模型、CFG条件、STD、CLS、LPIPS、MSE等技术；

**📊 数据集**

在自行构建的高分辨率配对数据集CTP‑HD（9804对）上训练，并在VITON‑HD无标注图像上评估泛化；

**📈 对比分析**

与多种基线（DCI‑VTON、StrDiffusion、SSR‑Encoder、IP‑Adapter、UniCon、StyleShot、OSASIS等）比较，SLDDM‑TPG在FID、SSIM、LPIPS、FPS等指标上显著优于对手（如FID下降4.1、SSIM提升0.116、FPS提升0.102）；

**⚠️ 局限性**

受限于数据规模与多样性，模型对极端纹理复杂度和极端姿态仍可能产生细节失真，且训练过程依赖多阶段、参数多、计算资源高，未来可进一步简化模型并提升无监督学习效果。

---

## 166. CoEmpaTeam: Enhancing Cognitive Empathy using LLM-based Avatars and Dynamic Role Play in Virtual Reality

**arXiv ID:** 2603.16614 | [PDF](https://arxiv.org/pdf/2603.16614v1)

**作者:** Dehui Kong `[一作]` (Karlsruhe Institute of Technology), Alexander Maedche `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 17334 | [OpenAlex ID](https://openalex.org/A5080792995)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了CoEmpaTeam VR系统，通过LLM驱动的三款拥有不同人格的虚拟头像，结合角色切换与情景对话，训练用户的认知同理心。

**💡 创新点**

创新点包括：①将LLM与多人格头像结合，确保头像在多轮交互中保持人格一致性；②在VR中实现角色切换机制，使学习者既扮演又观察，深化多视角体验；③将日常合租冲突情境嵌入训练，提升情境关联性与技能迁移。

**🔧 技术方法**

技术主要包括：Unity3D + Meta Quest Pro硬件；OpenAI Whisper实现语音转文本；Llama 3.1 8B Instruct生成对话；ElevenLabs TTS + Oculus LipSync实现语音合成；Mixamo动画 + Final IK实现姿态与凝视；自定义结构化提示（persona、情绪、手势）。

**📊 数据集**

数据集：①90名参与者完成头像人格评估（视频+NEO‑FFI‑30评估）用于验证头像人格；②22名完成训练的参与者提供IRI、PQ、VEQ、NASA‑TLX等问卷；③10+访谈与16条日记文本用于质性分析。未使用公开公开文本或语料库。

**📈 对比分析**

对比方式：采用配对t检验和重复测量ANOVA评估训练前后PT、FS的变化；无传统对照组，系统整体效能通过显著性（p<0.01）和效应量（d≈0.64‑0.86）体现；体验测评显示存在感、沉浸度维持在中等水平，未出现“怪异谷”。

**⚠️ 局限性**

局限性：样本以年轻大学生为主，文化与年龄多样性不足；未设计对照组，仅评估系统整体；技术延迟偶尔影响交互流畅度；自评量表受主观偏差；非言语表情有限，难以充分呈现情绪；日记与访谈数据量有限，长期迁移效果未知。

---

## 167. DanceHA: A Multi-Agent Framework for Document-Level Aspect-Based Sentiment Analysis

**arXiv ID:** 2603.16546 | [PDF](https://arxiv.org/pdf/2603.16546v1)

**作者:** Lei Wang `[一作]` (Temple University), Eduard Dragut `[通讯]` (Temple University)

**通讯引用:** 5128 | [OpenAlex ID](https://openalex.org/A5078060919)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了多代理框架DanceHA，用于开放式、文档级的Aspect-Based Sentiment Intensity Analysis（ABSIA），同时构建了包含23,024条ACOSI元组的Inf-ABSIA数据集。

**💡 创新点**

创新点在于将长文本拆分为基于方面的思考组（Dance）并通过多代理协作（Divider、Category、Opinion、Sentiment）完成任务，再结合人机协作（HA）实现高质量标注，并首次系统探讨非正式语言对情感强度的影响。

**🔧 技术方法**

采用的技术包括LLM多代理协作框架、Divide‑and‑Conquer思考组拆分、规则集成合并、人机协作标注、推理链知识蒸馏、LoRA微调、以及与CoT的对比实验。

**📊 数据集**

使用的数据集为Inf-ABSIA（从Amazon、Yelp、TripAdvisor三大域抽取的2,714篇长文档，平均8.48条ACOSI元组/文档），并对比了原始公开数据的ABSIA标签。

**📈 对比分析**

通过与零/少量提示CoT、单体LLM、以及各类多代理配置的实验比较，DanceHA在F1、Accuracy、MAE三项指标上显著优于CoT，MA模块进一步提升性能；最佳基座GPT‑4o达到最高得分，学生模型在推理链蒸馏后可与大型模型相媲美。

**⚠️ 局限性**

局限性包括代理团队规模对标注质量的影响尚未系统评估，Divider代理在思考组拆分上仍有改进空间，部分子任务可由轻量模型替代以提升效率，以及数据集目前以英文为主，跨语言扩展仍需探索。

---

## 168. CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering

**arXiv ID:** 2603.16091 | [PDF](https://arxiv.org/pdf/2603.16091v1)

**作者:** Tianyi Huang `[一作]` (Ryquo), Ying Kai Deng `[通讯]` (App-In Club)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级检索增强问答修复层 CounterRefine，先生成初始答案，再进行答案条件的二次检索以获取支持或反证，然后通过受限重写和确定性验证来纠正错误答案。

**💡 创新点**

创新点在于将检索从单纯的上下文搜集转变为对已生成答案的检验与纠错，形成可插拔的推理时修复机制，显著提升答案准确性。

**🔧 技术方法**

使用检索增强生成（RAG）框架、短答案生成、答案条件反证检索、受限重写门控、确定性验证器与答案规范化等技术。

**📊 数据集**

主要使用 SimpleQA 官方全量 4,326 题数据集以及 HotpotQA 100/300 例子子集进行评测。

**📈 对比分析**

与匹配的一通检索基线相比，在 Claude Sonnet 4.6 上 SimpleQA 正确率从 63.7% 提升到 67.7%，GPT‑5 从 67.3% 提升到 73.1%；在 HotpotQA EM 亦提升约 5%；整体表明在精确匹配指标上有显著提升。

**⚠️ 局限性**

局限性在于仅针对短答案场景，性能高度依赖检索质量；对多跳推理、长文本生成的效果尚不充分，且验证器可能无法处理复杂的关系冲突错误。

---

## 169. Neural-Symbolic Logic Query Answering in Non-Euclidean Space

**arXiv ID:** 2603.15633 | [PDF](https://arxiv.org/pdf/2603.15633v1)

**作者:** Lihui Liu `[一作]` (Wayne State University), Lihui Liu `[通讯]` (Wayne State University)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5016462374)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在双曲空间中训练的图神经网络，用于回答知识图谱上的复杂一阶逻辑查询。

**💡 创新点**

创新点在于将双曲嵌入与可学习曲率结合，用模糊集合方式分解查询为关系投影和逻辑运算，显著提升对层次结构的捕捉。

**🔧 技术方法**

采用双曲空间图神经网络、指数/对数映射、产品模糊逻辑运算以及多层感知机进行投影预测。

**📊 数据集**

使用三大基准数据集：FB15k、FB15k-237和NELL995，采用BetaE提供的9种EPFO+5种含否定的查询结构。

**📈 对比分析**

与GQE、Q2B、BetaE、FuzzQE、CQD-CO/Beam以及GNN-QE等基线比较，实验表明其在MRR和Hits@1上大多数查询类型均取得最高或相当的成绩，尤其在交叉、交叉+否定查询上表现突出。

**⚠️ 局限性**

局限在于仅覆盖有限的FOL子集（不含全称量化），对极度稀疏图的推理效果下降，且双曲空间训练复杂度高。

---

## 170. Can Linguistically Related Languages Guide LLM Translation in Low-Resource Settings?

**arXiv ID:** 2603.16660 | [PDF](https://arxiv.org/pdf/2603.16660v1)

**作者:** Aishwarya Ramasethu `[一作]` (Prediction Guard), Dun Li Chan `[通讯]` (INTI International College Penang)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究在极低资源机器翻译场景下，利用语言相近的中介语言和少量示例对大语言模型进行推理时的适应效果

**💡 创新点**

在不更新参数的情况下，提出了基于检索的中介语言提示与少量上下文示例相结合的推理策略，并系统评估其对未见低资源语言的影响

**🔧 技术方法**

使用检索增强的提示（检索相似句对），In‑Context Learning，ChatML格式提示，及预训练的7B级解码器LLM（Hermes‑2‑Pro‑Llama‑3‑8B和TowerInstruct‑7B‑v0.1）

**📊 数据集**

利用两套三语并行数据：英语-马拉地-康坎语（800/200）和英语-MS‑阿拉伯-突尼斯阿拉伯语（900/100），从中构建检索向量库

**📈 对比分析**

通过与直接零样本、无中介语言的few‑shot以及NLLB‑200基准对比，发现对康坎语的中介语言提示在最佳示例数下可提升约1.7 chrF++和0.3 BLEU；对突尼斯阿拉伯语则效果更不显著，但在少样本时仍略有提升；与NLLB基准相比，Hermes在低资源情境下可略优或相当，但Tower略逊

**⚠️ 局限性**

增添示例数量后性能并不持续提升，且易受示例质量和上下文长度限制；方法依赖高质量相近中介语言且对缺乏此类中介语言的语言无通用性；自动评估指标在低资源、形态丰富语料上不完全可靠，缺乏人工评估

---

## 171. DermaFlux: Synthetic Skin Lesion Generation with Rectified Flows for Enhanced Image Classification

**arXiv ID:** 2603.16392 | [PDF](https://arxiv.org/pdf/2603.16392v1)

**作者:** Stathis Galanakis `[一作]` (Imperial College), Stefanos Zafeiriou `[通讯]` (Imperial College)

**通讯引用:** 22068 | [OpenAlex ID](https://openalex.org/A5080553022)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于rectified flow的文本驱动皮肤病变图像合成框架，用于生成多样、临床可解释的皮肤病变图像，帮助解决真实数据稀缺导致的类别不平衡问题。

**💡 创新点**

创新点包括：① 将rectified flow与低秩适配LoRA相结合，形成可控、确定性生成模型；② 使用Llama 3.2生成结构化的、基于ABCDE标准的文字描述，实现属性级对齐；③ 构建了约50万条带属性标签的公开图文对，填补了公开数据集缺乏详细形态学描述的空白。

**🔧 技术方法**

主要技术包括：rectified flow生成器、LoRA参数高效微调、双文本编码器（CLIP + T5‑XXL）、Transformer‐based 视觉–文本融合、ODE采样器等；在训练与推理时采用多分辨率（128×128/256×256/512×512）策略。

**📊 数据集**

使用了多个公开皮肤病学数据集的集合（ISIC 2019/2020/2024、Derm12345、PAD20、Kaggle 1/2、MedNode、HIBA、Milk10k、DDI 等），共计约50万图文对，涵盖临床与皮镜图像，且包含良恶性标注。

**📈 对比分析**

通过与基于扩散的Derm‑T2IM、以及现有最佳皮肤学分类器（BiomedCLIP、DermLIP、MAKE 等）对比，实验表明：① 生成的合成图像可使 ResNeXt/ViT 在仅 2,500 实例 + 4,375 合成样本的设置下，准确率提升至 78.04%、AUC 0.859，超过竞争模型 8%；② 在 1:1 真实/合成混合训练中，ViT 的准确率提升至 76.1%，并且纯合成训练亦能提升 1–5% 的性能。

**⚠️ 局限性**

局限性包括：① 生成模型仍依赖高质量的结构化文字描述，若文本质量下降可能影响图像属性一致性；② 研究主要关注二分类（良恶），未对多类别或亚型进行深入验证；③ 生成的图像在某些罕见病变上可能缺乏足够的多样性，需要进一步扩充数据或引入更细粒度的标注。

---

## 172. Improving Code Comprehension through Cognitive-Load Aware Automated Refactoring for Novice Programmers

**arXiv ID:** 2603.16791 | [PDF](https://arxiv.org/pdf/2603.16791v1)

**作者:** Subarna Saha `[一作]` (Jahangirnagar University), Mia Mohammad Imran `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 73 | [OpenAlex ID](https://openalex.org/A5088063511)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于认知驱动的自动重构工具CDDRefactorER，利用LLM在提示中嵌入认知复杂度约束，对代码进行结构化重构以降低新手的认知负担。

**💡 创新点**

创新点在于将认知复杂度指标（ICP）和认知驱动开发（CDD）原则直接融入LLM提示策略，显著提升重构安全性、保持结构相似度，并通过约束重构方法（提取方法、减少嵌套等）实现对初学者可读性的系统优化。

**🔧 技术方法**

使用技术包括认知驱动开发（CDD）理论、ICP 计算、LLM提示工程、控制流与认知复杂度（Cyclomatic、Cognitive Complexity）评估，以及 CodeBLEU 结构相似度评估。

**📊 数据集**

评估数据集主要为 MBPP 与 APPS（两者均为初学者级别 Python 代码）以及 20 名一年级 CS 学生组成的人机实验样本。

**📈 对比分析**

通过与无约束（baseline）提示进行对比，CDDRefactorER 在 MBPP 上错误率下降 71.8%，在 APPS 上下降 54–71%；认知与循环复杂度增加率分别从约 29% 降至 12%；CodeBLEU 中位数提升 75–123%；人机实验中功能识别提升 31.3%，结构可读性提升 22%。

**⚠️ 局限性**

限制在于仅测试单文件 Python 算法题，实验规模有限，结果受LLM模型、提示设计与被试差异影响；且对高级编程概念的理解提升仍有限，无法完全替代教学与深入学习。

---

## 173. SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation

**arXiv ID:** 2603.16219 | [PDF](https://arxiv.org/pdf/2603.16219v1)

**作者:** Hang Lv `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 28474 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SpecSteer 框架，将设备端小模型（专属模型）与云端大模型（通用模型）通过 Draft–Verify–Recover 流程协同推理，以解决隐私与推理能力的矛盾。

**💡 创新点**

创新点在于：①把 speculative decoding 重新解释为 Bayesian 知识融合；②通过比例验证消除对私有数据的需求；③在验证失败时使用 logit 注入的 steering recovery，保证生成既符合全局推理又保持用户意图。

**🔧 技术方法**

使用技术包括 Bayesian 知识融合、speculative decoding、比例（ratio‑based）验证、logit 注入恢复、PMI 量化个性化意图、LoRA、RAG、RAFT 等本地增强技术。

**📊 数据集**

实验基准为 LongLaMP 与 LaMP 等个性化生成数据集，涵盖文本摘要、邮件生成、文章写作等任务。

**📈 对比分析**

与完整本地增强（RAG、LoRA、RAFT）、检索增强、PEFT、对齐方法以及标准 Speculative Decoding 等进行对比。SpecSteer 在保持或提升生成质量的同时，实现约 2.36× 的速度提升，并显著优于上述基线。

**⚠️ 局限性**

局限性包括：当本地模型缺乏可靠个性化信号时提升有限；对超参数 λ、β 需一定调优；跨 tokenzier 兼容性需额外处理；在极端噪声或无效专属模型的情况下仍可能出现性能下降。

---

## 174. Learning Lineage-guided Geodesics with Finsler Geometry

**arXiv ID:** 2603.16708 | [PDF](https://arxiv.org/pdf/2603.16708v1)

**作者:** Aaron Zweig `[一作]` (New York Genome Center), Elham Azizi `[通讯]` (Columbia University)

**通讯引用:** 4096 | [OpenAlex ID](https://openalex.org/A5075059210)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种结合连续几何先验和离散有向祖先树先验的 Finsler 度量，用于单细胞发育轨迹推断。

**💡 创新点**

创新点在于构造了一个基于分类器和祖先树矩阵的非对称局部范数，使 geodesic 方向依赖，从而把离散发育方向嵌入度量空间。

**🔧 技术方法**

使用 Finsler 几何、最优传输、流匹配、神经 ODE、嵌入学习与能量最小化相结合的训练框架。

**📊 数据集**

在合成二维/五十维数据、斑马鱼胚胎 CNS 与 PA 组织以及小鼠胚胎器官发育血液系数据上进行实验。

**📈 对比分析**

与 CFM、MFM 等无 Finsler 的基线方法对比，采用 W1 Wasserstein 距离评估离散时间点插值，结果显示加入 Finsler 可显著降低误差并提升轨迹符合祖先树的程度。

**⚠️ 局限性**

主要局限是对祖先树质量依赖强、在树结构稠密或与环境几何已对齐时提升有限，以及对分类器校准和嵌入/轨迹网络的敏感性。

---

## 175. Learning to Predict, Discover, and Reason in High-Dimensional Discrete Event Sequences

**arXiv ID:** 2603.16313 | [PDF](https://arxiv.org/pdf/2603.16313v1)

**作者:** Hugo Math `[一作]` `[通讯]`, Hugo Math

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文致力于从车辆诊断事件流（Diagnostic Trouble Codes, DTCs）出发，构建端到端的自动故障诊断管线，涵盖预测、因果发现与可解释推理三大模块。通过Transformer模型对异步事件序列进行预训练与微调，先实现错误模式（Error Pattern, EP）的前瞻预测；随后设计OSC AR、CARGO和TRACE三套可扩展因果发现框架，利用预训练模型的密度估计来恢复事件与结果的因果边；最后将因果图与大语言模型（LLM）结合，构建多智能体系统CARE P，以自动生成可解释的Boolean规则。整套方法可直接从原始事件流生成可读的故障诊断知识。

**💡 创新点**

创新点包括：① 将DTC序列视为“语言”，首次在高维（≈10^4）事件空间中应用自回归Transformer进行错误模式预测；② 开发基于预训练Transformer的神经密度估计因果发现方法（OSCAR、CARGO、TRACE），实现样本级和全局级因果图的线性可扩展构建；③ 引入多模态BiCarFormer，融合环境传感器数据与DTC序列，显著提升多标签错误模式分类；④ 设计CARE P多智能体系统，将因果证据与LLM推理相结合，实现自动生成可解释的Boolean EP规则，并在大规模车载日志上验证。

**🔧 技术方法**

主要技术包括：Transformer（自回归CarFormer、双向BiCarFormer、EPredictor），RoPE、位置/时间/里程嵌入；因果发现算法（利用条件互信息估计的OSC AR、CARGO、TRACE）；图聚合与自适应阈值的全局因果图构建；多模态融合与跨注意力机制；多智能体架构CARE P（因果发现代理、上下文信息代理、协调器）；大语言模型（LLM）推理与检索增强；评估指标（F1、MAE、RMSE、CPMW-AUC）。

**📊 数据集**

数据集为BMW公司内部的车辆诊断日志：1.7×10^6条DTC序列（平均≈150条/序列，约8710种不同DTC），包含时间、里程等元信息；5×10^6条带环境传感器（温度、湿度、压力等）扩展序列；多标签EP标签（约几百种），每条序列关联多种EP。由于隐私限制，论文使用匿名化或合成版本公开。

**📈 对比分析**

与传统统计/规则方法相比，CarFormer/EPredictor在EP预测上达到了≈80% F1，时间预测MAE≈58h，CPMW-AUC显示模型能在约序列中位点前产生可靠预测；BiCarFormer在多标签分类上比单一DTC模型提升了10–15% F1；OSC AR/CARGO/TRACE在模拟与真实车辆日志上成功恢复了数千个事件与EP之间的因果边，且与手工规则的匹配率高于80%；CARE P在自动生成EP规则的实验中达到了约83%规则元素的精确度，明显优于仅使用LLM的基线。整体表明该方法在准确性、可解释性和可扩展性上均超过现有方法。

**⚠️ 局限性**

主要局限：① 对预训练Transformer的收敛度高度依赖，模型欠收敛会直接影响后续预测与因果发现；② CPMW指标表明需要观察大约序列一半才可做出可靠预测，对某些需即时干预的场景仍不足；③ EP标签分布极不均衡，稀有但关键的EP仍难以充分识别；④ 虽实现线性可扩展的因果图构建，但在极高维（>10^4）下仍面临显存与计算瓶颈；⑤ 多模态融合需要处理时间对齐与异构数据量巨大，当前方案主要针对特定传感器，通用性待进一步验证。

---

## 176. VIGOR: VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

**arXiv ID:** 2603.16271 | [PDF](https://arxiv.org/pdf/2603.16271v1)

**作者:** Tengjiao Yin `[一作]` (Nankai University), Xi Wang `[通讯]` (Ecole Polytechnique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VIGOR，利用预训练几何基础模型计算点位投影误差作为奖励，以提升视频扩散模型的几何一致性；

**💡 创新点**

创新点在于：①基于点位投影误差的物理量化几何奖励；②几何感知采样策略过滤低纹理/非语义区域；③在两条路径（后训练对齐与推理时缩放）上统一应用该奖励；

**🔧 技术方法**

使用了 VGGT（视觉几何生成模型）进行深度与相机参数估计，点位跟踪、投影误差计算；对齐方法包括 LoRA 细调、DPO 优化、Best‑of‑N 与三种 TTS 搜索（SoS、SoP、Beam Search）；

**📊 数据集**

构建了 GB3DV‑25k（25.6k 视频对，覆盖多场景与摄像机运动）数据集，实验基准使用 VBench、PSNR/SSIM/LPIPS、EPI/RPX/RPT 等指标；

**📈 对比分析**

与 Epipolar、Reproj‑Pix 等基线比较，VIGOR 在 3D 重建、视角一致性与 VBench 总分上均优于基线；后训练对齐（SFT/DPO）和推理时缩放均显著提升几何与感知质量；

**⚠️ 局限性**

局限性包括：①依赖 VGGT 估计的深度/相机精度；②对低纹理或非语义区域的处理仍可能不足；③对极端视角/动态场景的适应性需进一步验证；

---

## 177. Residual Stream Duality in Modern Transformer Architectures

**arXiv ID:** 2603.16039 | [PDF](https://arxiv.org/pdf/2603.16039v1)

**作者:** Yifan Zhang `[一作]` `[通讯]`, Yifan Zhang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文提出 Transformer 的残差流与自注意力在层深度与序列位置两轴上具有对偶关系，并指出深度方向的残差注意力实际上等价于序列轴上的 ShortSWA；

**💡 创新点**

创新点在于将深度维度的残差聚合视为对偶的局部注意力操作，构建了一条从静态加权到基于注意力的跨层聚合的统一设计空间，并给出了基于系统效率的设计建议；

**🔧 技术方法**

主要技术包括对 Transformer 结构的数学对偶分析、构造深度维度的短因果注意力（Depth‑wise ShortSWA），以及提出 Deep Delta Learning（DDL）对残差更新的直接改造；

**📊 数据集**

本文未使用具体数据集，侧重理论框架和系统实现建议；

**📈 对比分析**

由于未给出实验验证，本文未与其他方法在性能上进行比较，仅从计算复杂度与系统实现角度阐述优劣；

**⚠️ 局限性**

限制在于缺乏大规模实验验证，对实际推理/训练性能提升的量化证据不足，并且对不同模型规模和硬件配置的适用性未进行细致评估。

---

## 178. CTG-DB: An Ontology-Based Transformation of ClinicalTrials.gov to Enable Cross-Trial Drug Safety Analyses

**arXiv ID:** 2603.15936 | [PDF](https://arxiv.org/pdf/2603.15936v1)

**作者:** Jeffery L. Painter `[一作]` (GSK), Andrew Bate `[通讯]` (GSK)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并发布了CTG-DB，能将完整的ClinicalTrials.gov XML归档转换为基于MedDRA术语、保留arm层级分母的关系型数据库，实现跨试验安全性分析。

**💡 创新点**

首个开源、术语对齐且分母保持的CT.gov转换管道，支持概念级查询、安慰剂对比聚合和贝叶斯信号检测的先验信息注入。

**🔧 技术方法**

使用Java的JAXB解析XML，Python批量导入，MedDRA v24.1的精确与基于二元组的模糊匹配，对AE进行归一化，并将结果加载至MySQL/PostgreSQL；同时利用UMLS实现同义词扩展。

**📊 数据集**

完整ClinicalTrials.gov XML归档（约17 GB，569,589文件，过滤后544,315项），以及2015–2019年FAERS数据用于后期贝叶斯整合。

**📈 对比分析**

相较于AACT，CTG-DB实现59.2%（独特字符串）和95.0%（受影响人数加权）的MedDRA映射覆盖；通过安慰剂对比聚合与贝叶斯动态借用方法，显示比传统失调度量方法略有性能提升。

**⚠️ 局限性**

仍缺乏药物干预标准化映射、无法恢复患者级曝光信息、跨试验随机化比例和随访时长未完全校正、部分AE仍无法映射、对CT.gov报告质量敏感。

---

## 179. Theoretical Foundations of Latent Posterior Factors: Formal Guarantees for Multi-Evidence Reasoning

**arXiv ID:** 2603.15674 | [PDF](https://arxiv.org/pdf/2603.15674v1)

**作者:** Aliyu Agboola Alege `[一作]` `[通讯]` (Epalea), Aliyu Agboola Alege (Epalea)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究并实现了一种名为Latent Posterior Factors（LPF）的多证据聚合框架，能够在多源异构信息下对分类任务进行可靠、可解释的概率预测。

**💡 创新点**

创新点包括：
1) 通过变分自编码器将每条证据映射为高斯后验，再用Monte Carlo积分生成软因子；
2) 提供两种聚合方式：基于Sum‑Product Network（LPF‑SPN）实现严格的概率推理和校准保持；
3) 推导七条正式理论保证，覆盖校准、蒙特卡罗误差、泛化、信息下界、鲁棒性、样本复杂度与不确定性分解；
4) 在八个跨域数据集上验证理论，并实现近似信息理论最优的校准误差。

**🔧 技术方法**

核心技术包括：变分自编码器（VAE）对证据进行编码；Monte Carlo采样实现后验边缘化；Sum‑Product Network推理；可学习的注意力聚合；PAC‑Bayes与信息论分析；实验使用基准对比模型（BERT、EDL、Qwen3）和多模态验证。

**📊 数据集**

使用了八个真实领域的数据集，涵盖合规审查、医疗诊断、金融风险、法律案件、学术资助、材料科学、建筑风险以及FEVER事实验证，训练样本最多达4,200个。

**📈 对比分析**

与统一平均、BERT、EDL、Qwen3等基线比较，LPF‑SPN在准确率（≈99.3%）与期望校准误差（≈1.5%）上明显优于基线，LPF‑Learned在实际校准误差上更优（≈0.058%）但缺乏正式保证；理论证明表明LPF的误差与信息下界仅相差1.12×，鲁棒性提升至88%在半数证据被破坏时。

**⚠️ 局限性**

局限性包括：
- 主要验证在K≤5的证据量，超大证据集需层级聚合；
- 依赖变分编码器，存在后验崩溃风险；
- 理论界限相对保守，实验误差远低于上界；
- 仅对分类任务验证，未扩展至回归或结构化预测；
- 计算复杂度随证据数和采样数线性增长。

---

## 180. Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors

**arXiv ID:** 2603.15656 | [PDF](https://arxiv.org/pdf/2603.15656v1)

**作者:** Peiyu Yang `[一作]` (University of Melbourne), Ajmal Mian `[通讯]` (University of Western Australia)

**通讯引用:** 20514 | [OpenAlex ID](https://openalex.org/A5089986388)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于秩一模型编辑的模型纠错框架，用来纠正神经网络在受污染样本上表现出的非鲁棒特征导致的不可靠行为；

**💡 创新点**

创新点在于将秩一编辑从知识注入转为模型纠错，提出了可纠错性（rectifiability）和与投影对齐控制（span‑aligned control）两大结构特性，并设计了基于归因的层定位方法和动态纠错流程，显著提升了在不同层的可编辑性；

**🔧 技术方法**

使用的核心技术包括秩一模型编辑、积分归因（Integrated Gradients）来计算层级归因、基于归因映射的可编辑性评分、动态纠错算法；

**📊 数据集**

在多个数据集上验证，包括CIFAR‑10、ImageNet、BlockMNIST、ISIC（皮肤病变）等，涵盖神经木马、虚假关联、特征泄漏等三类不可靠行为；

**📈 对比分析**

与微调、P‑ClArC/A‑ClArC、静态层编辑等方法对比，结果表明仅使用一个清洗样本即可将攻击成功率降至≈1%或更低，同时总体准确率基本保持不变，显示出卓越的性能与样本效率；

**⚠️ 局限性**

局限性包括仍需人工或自动化手段获取清洗样本，针对复杂多层次或多模态数据时层定位的精度可能下降，以及在极端分布偏移下对样本需求仍可能增加。

---

## 181. GeMA: Learning Latent Manifold Frontiers for Benchmarking Complex Systems

**arXiv ID:** 2603.16729 | [PDF](https://arxiv.org/pdf/2603.16729v1)

**作者:** Jia Ming Li `[一作]` (Imperial College London), Daniel J. Graham `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于变分自编码器的几何流形前沿框架（Geometric Manifold Analysis, GEMA），通过在输入-输出空间学习低维流形来描述生产集合，并从中推导效率得分。

**💡 创新点**

创新点在于：①将前沿视为潜在流形的 Pareto 边界；②利用分头编码器实现技术与无效率的可分离表征；③通过商流形实现规模不变的基准化；④基于解码器雅可比与 Lipschitz 上界定义局部认证半径，量化效率得分的几何鲁棒性。

**🔧 技术方法**

采用深度生成模型（变分自编码器）+ 约束正则化（单调性正则、对数空间损失）+ Jacobian 计算与 Lipschitz 上界估计 + 传统前沿方法（DEA、SFA、CNLS 等）做对照。

**📊 数据集**

实验数据包括：合成数据（模拟非凸、异质、规模混淆场景）；真实案例：英国铁路运营商 (ORR)、中国风电场 SCADA 数据 (Wind Farms)、宏观经济 Penn World Table (PWT)、城市轨道交通 COMET（附录）。

**📈 对比分析**

与 DEA、SFA、FDH、CNLS、随机森林等基准方法进行对比；在满足经典假设时与传统方法相当；在技术异质、非凸或规模偏差显著时表现更优，且能提供可视化的同质群集与鲁棒性诊断，整体性能稳定。

**⚠️ 局限性**

局限性包括：模型训练需 GPU 与调参，计算成本高；技术与无效率潜变量可辨识性不强，需要正则化和先验；商流形和认证半径仅给出经验性的规模与鲁棒性指标，缺乏严格理论保证；对异常值与噪声的敏感性仍需进一步评估。

---

## 182. Non-GRS type Euclidean and Hermitian LCD codes and Their Applications for EAQECCs

**arXiv ID:** 2603.16187 | [PDF](https://arxiv.org/pdf/2603.16187v1)

**作者:** Zhonghao Liang `[一作]` (Sichuan Normal University), Zhengchun Zhou `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 4299 | [OpenAlex ID](https://openalex.org/A5004913173)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文基于广义Roth‑Lempel（GRL）码，构造了多类非GRS类型的欧氏和赫尔米特 LCD 码以及小维数 hull 的线性码，并利用这些码设计了若干族 EAQECC。

**💡 创新点**

创新点主要在于：① 将 GRL 码推广到任意大小的 ℓ×ℓ 可逆矩阵 A；② 给出了欧氏与赫尔米特 LCD 条件的充分必要条件，并对 hull 维数给出了上界；③ 证明 GRL 码在 k>ℓ 时必为非 GRS；④ 通过这些构造直接得到多族最优或近最优的 EAQECC，填补了非 GRS LCD 码在量子纠错中的空白。

**🔧 技术方法**

采用的技术包括：GRL 码的生成矩阵分析、对称多项式与 Cauchy 矩阵方法、欧氏/赫尔米特双重内积判定、Hull 维数与矩阵行列式关系、以及对非 GRS 性质的代数判定。

**📊 数据集**

本文不使用实验数据集，而是全部基于理论构造与符号计算，验证结果主要通过 Magma 计算机代数系统完成。

**📈 对比分析**

与已有的基于 GRS 码的 LCD 码与 EAQECC 方案进行理论对比，证明了在给定参数范围内构造出的码满足或接近 MDS、NMDS、AMDS 等最优距离特性；在量子误差校正方面，所得到的 EAQECC 的协同纠错容量与已知的最优 EAQECC 相匹配或超越，展示了非 GRS 构造的竞争力。

**⚠️ 局限性**

限制与不足包括：① 仅给出了若干参数区间内的构造，未覆盖所有可能的 k、ℓ、q；② 对 Hermitian LCD 的 Hull 维数上界虽给出，但实际实现需满足严格的数值条件，限制了可选参数；③ 证明非 GRS 性质依赖于 k>ℓ 的严格不等式，对于 k=ℓ 的情况仅给出了部分例子，未给出完整分类；④ 量子码的实际性能（如误差阈值、编码/解码复杂度）仍需实验验证。

---

## 183. AI Application Benchmarking: Power-Aware Performance Analysis for Vision and Language Models

**arXiv ID:** 2603.16164 | [PDF](https://arxiv.org/pdf/2603.16164v1)

**作者:** Martin Mayr `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Gerhard Wellein `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 6405 | [OpenAlex ID](https://openalex.org/A5070209050)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提供了一个面向视觉与语言模型的吞吐量与功耗基准框架，系统评估 NVIDIA H100/H200 与 AMD MI300X 在不同功耗上限下的性能与能效。

**💡 创新点**

首次将吞吐量（图像/每秒、token/每秒）作为核心指标，配合功耗上限扫描与能效曲线分析，揭示不同 GPU 在功率限制下不存在统一最优点，并公开该基准工具。

**🔧 技术方法**

采用 PyTorch + xFormers + Lightning，Apptainer 容器化，GPU 级功率回调；模型包括 ResNet‑50、ViT‑L/16、Stable Diffusion v2、LLaMA‑3 8B（预训练与推理）；使用 TF32/FP32/FP16 训练与推理。

**📊 数据集**

视觉任务使用预生成的随机 RGB 图像；LLaMA 预训练使用约 3000 万英文 token 数据集；推理任务采用标准文本生成示例。

**📈 对比分析**

在单节点 4‑GPU（或 8‑GPU 限制 4‑GPU）配置下，按 100 W 步长扫描功耗上限，测量图像/每秒和 token/每秒并计算能效；结果显示 H100 在低功率下吞吐最高，H200 在高功率下吞吐最高，MI300X 能效最低但随功率升高表现相对平滑。

**⚠️ 局限性**

主要限制包括：AMD 软件堆栈成熟度不足，低功率上限无法可靠执行；实验仅限单节点，未覆盖多节点通信与分布式训练；未系统评估不同精度格式（BF16、FP8 等）的影响；数据集规模有限，未覆盖更大多样化数据。

---

## 184. Meta-TTRL: A Metacognitive Framework for Self-Improving Test-Time Reinforcement Learning in Unified Multimodal Models

**arXiv ID:** 2603.15724 | [PDF](https://arxiv.org/pdf/2603.15724v1)

**作者:** Lit Sin Tan `[一作]`, Lijie Wen `[通讯]` (Tsinghua University)

**通讯引用:** 4525 | [OpenAlex ID](https://openalex.org/A5030845033)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Meta-TTRL 框架，实现统一多模态模型在测试时通过强化学习自我提升

**💡 创新点**

通过元认知监控信号与模型自身知识自我评估，消除对外部奖励模型的依赖，形成元认知协同提升机制

**🔧 技术方法**

采用两层元认知架构、Rubric 构造、结构化自评、GRPO 策略更新等技术

**📊 数据集**

在 TIIF‑Bench、T2I‑CompBench++、DPG‑Bench 等文本到图像生成基准上验证

**📈 对比分析**

与基线模型对比，在 Janus‑Pro‑7B、BAGEL、Qwen‑Image 三大模型上平均提升 2–5% 甚至 10% 以上，弱模型提升尤为显著

**⚠️ 局限性**

需要访问模型参数，无法直接应用于闭源模型

---

## 185. Change is Hard: Consistent Player Behavior Across Games with Conflicting Incentives

**arXiv ID:** 2603.16136 | [PDF](https://arxiv.org/pdf/2603.16136v1)

**作者:** Emily Chen `[一作]` (University of Southern California), Emilio Ferrara `[通讯]` (University of Southern California)

**通讯引用:** 18909 | [OpenAlex ID](https://openalex.org/A5078699564)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对同一组玩家在《英雄联盟》和《Teamfight Tactics》两款结构相反的竞技游戏中，测量并比较其玩法多样性（灵活性）与竞技成功之间的关系，并检验跨游戏行为的一致性与适应性。

**💡 创新点**

首次通过在同一玩家账户上跨两款游戏跟踪，消除自我选择偏差，系统性考察结构激励与个人性格如何共同塑造跨平台行为，验证结构-代理双重性理论的实证支持。

**🔧 技术方法**

采用统计学与机器学习方法，包括多变量线性回归、线性混合模型、梯度提升、核回归（Laplacian）及神经网络，并用 SHAP 可解释性技术分析特征重要性。

**📊 数据集**

利用 Riot Games 官方 API 与第三方网站（OP.GG、LoLchess）收集的 4,830 名玩家在 League 与 TFT 赛季各 50+ 竞技游戏的完整游戏记录（约 100+ 比赛）。

**📈 对比分析**

先用 MVLR 探索灵活性与胜率的相关性（League 为负相关，TFT 为正相关），随后用多模型预测灵活性，核回归（Laplacian）取得最高 R²（League 0.2235，TFT 0.4551），显示跨游戏灵活性是关键预测因子，支持 H2。

**⚠️ 局限性**

仅分析同一开发者的两款游戏，样本可能偏向特定玩家群；灵活性定义基于最终决策的端点，缺乏动态决策细节；地区仅限北美；精英玩家比例过高，普通玩家行为细节缺失；未收集玩家性格或动机调查数据。

---

## 186. Domain Adaptation Without the Compute Burden for Efficient Whole Slide Image Analysis

**arXiv ID:** 2603.15774 | [PDF](https://arxiv.org/pdf/2603.15774v1)

**作者:** Umar Marikkar `[一作]` (University of Surrey), Sara Atito `[通讯]` (University of Surrey)

**通讯引用:** 1053 | [OpenAlex ID](https://openalex.org/A5037459105)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在高分辨率病理切片分类任务中提出了一种端到端的EfficientWSI框架，结合参数高效微调与多实例学习实现对WSI的直接训练。

**💡 创新点**

创新点在于将LoRA等PEFT方法与对低采样率鲁棒的LinMax聚合相结合，既保持了训练效率，又提升了对任务特定特征的学习。

**🔧 技术方法**

使用了LoRA参数高效微调、Vision Transformer编码器、随机采样、LinMax max‑pooling聚合、注意力/自注意力等MIL变体。

**📊 数据集**

在Camelyon16、TCGA（BRCA、NSCLC）以及BRACS数据集上进行实验。

**📈 对比分析**

与传统冻结特征MIL（如ABMIL、TransMIL、ACMIL）及其他端到端方法对比，eWSI在ImageNet和域内预训练下均取得更高AUC，且训练时间仅需数小时。

**⚠️ 局限性**

在稀疏采样和小ROI场景下仍存在预测概率偏移，需要手动阈值校正或额外冻结训练以提升绝对准确率。

---

## 187. DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping

**arXiv ID:** 2603.16806 | [PDF](https://arxiv.org/pdf/2603.16806v1)

**作者:** Yuliang Wu `[一作]` (Sun Yat-sen University), Ancong Wu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2976 | [OpenAlex ID](https://openalex.org/A5049519486)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种能够零样本跨身体（跨机器人手）抓取的通用策略 DexGrasp-Zero。

**💡 创新点**

创新点包括①采用形态对齐的图表示和三轴手势原语，统一感知与动作语义；②通过物理属性图注入的 Morphology‑Aligned Graph Convolutional Network (MAGCN) 让网络直接感知关节长度、限位等硬件约束；③策略直接输出手势原语动作，无需中间的手型重映射，从而消除转移时的运动学/动力学冲突。

**🔧 技术方法**

技术手段包括图卷积网络（MAGCN）、物理属性注入（URDF‑derived 物理图）、三轴手势原语映射、强化学习（PPO）、模拟‑真实迁移的教师‑学生蒸馏，以及基于图的状态/动作编码。

**📊 数据集**

使用 CrossDex/YCB 45 个对象数据集进行多手训练与零样本评估；以及 GraspXL（ShapeNet+PartNet）数据集用于单手训练与跨手转移测试。

**📈 对比分析**

与当前最先进的 CrossDex 基线对比，未见手上成功率从 26.5% 提升至 85%（模拟），单手训练时也能达到 0.94 的成功率；在模拟中全模型平均 92%/85%，在真实三台机器人上平均 82%。

**⚠️ 局限性**

局限性包括：对手势原语的定义仍需手工设定，导致对极端手型差异（如极小手或高度非典型关节布局）时泛化不稳定；依赖固定的手型映射 ℳ_h，若硬件特性变化大需重新标定；在非常小或强制约束的手上性能下降。

---

## 188. A Context Alignment Pre-processor for Enhancing the Coherence of Human-LLM Dialog

**arXiv ID:** 2603.16052 | [PDF](https://arxiv.org/pdf/2603.16052v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 189. Confusion-Aware Spectral Regularizer for Long-Tailed Recognition

**arXiv ID:** 2603.16732 | [PDF](https://arxiv.org/pdf/2603.16732v1)

**作者:** Ziquan Zhu `[一作]` (University of Leicester), Tianjin Huang `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 290 | [OpenAlex ID](https://openalex.org/A5028180352)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于混淆矩阵谱正则化的长尾图像分类方法，强调最少样本类别的泛化性能。

**💡 创新点**

创新点：① 从理论上证明最坏类误差可由频率加权混淆矩阵的谱范数上界；② 引入混淆矩阵可微近似与指数移动平均估计，实现对谱范数的有效正则化；③ 该正则化可与多种数据增强方法互补。

**🔧 技术方法**

技术：PAC‑Bayesian 误差分析、频率加权混淆矩阵、谱范数正则化、可微混淆矩阵替代（软阈值与软argmax）、EMA（指数移动平均）估计、交叉熵损失混合、ViT-Small 等 Transformer 结构。

**📊 数据集**

数据集：ImageNet‑LT、CIFAR‑100‑LT、iNaturalist2018、Tiny‑ImageNet‑LT，涵盖不同 imbalance factor（50、100、200）。

**📈 对比分析**

比较方法：与 CE、CB、Focal、LDAM‑DRW、BALMS、ReMix、MetaSAug、CMO、SAFA、ConCutMix、LOS 等现有长尾学习与增强方法对比；在训练从零和预训练微调两种设置下均实现了最高的总体精度和最坏类精度提升（例如 ImageNet‑LT 最坏类从 32.73% 提升至 38.07%，整体从 56.20% 提升至 60.07%）。

**⚠️ 局限性**

局限性：① 仍需对大规模数据和更深网络的可扩展性进行深入验证；② 对谱正则化参数（α、γ、β、r₀）敏感，需要手动调优；③ 仅关注交叉熵与谱范数的联合正则，可能在某些场景下不如专门设计的类别重权重或采样策略。

---

## 190. BATQuant: Outlier-resilient MXFP4 Quantization via Learnable Block-wise Optimization

**arXiv ID:** 2603.16590 | [PDF](https://arxiv.org/pdf/2603.16590v1)

**作者:** Ji-Fu Li `[一作]` (Huawei Technologies), Xianzhi Yu `[通讯]` (Huawei Technologies)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5021890706)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种名为 BATQuant 的块级仿射量化框架，用于在 MXFP4 体系下进行无监督的后训练量化（PTQ）。

**💡 创新点**

核心创新点包括：① 将仿射变换限定为 MXFP 区块粒度，消除跨区块能量转移与双峰分布；② 引入 Global and Private Kronecker（GPK）分解降低参数占用；③ 采用块级可学习裁剪进一步抑制残留的异常值。

**🔧 技术方法**

采用的技术包括块级仿射变换、GPK 分解、块级可学习裁剪、GPTQ 量化后端、低位矩阵乘法、以及对齐 MXFP4 的 UE8M0 标量化。

**📊 数据集**

使用的数据集涵盖：Qwen3-VL-8B-Instruct 与 Qwen3-8B 两大模型；多模态基准（MME、OCRBench、DocVQA、RealWorldQA、VLMBlind）；非推理任务基准（PIQA、Winogrande、Hellaswag、ARC-Easy、ARC-Challenge）；推理任务基准（GSM8K、MATH-500、AIME24、AIME25、GPQA-D）；以及用于校准的 GQA、Numina-Math-1.5 等。

**📈 对比分析**

与 QuaRot、SpinQuant、BRQ、FlatQuant、SmoothQuant、GPTQ 等现有 PTQ 方法对比，BATQuant 在 W4A4KV16 等极端低位配置下恢复率高达 96.43%，在多模态与 LLM 任务上均超过基线，尤其在推理任务上表现尤为突出。

**⚠️ 局限性**

局限性包括：对极低位配置仍可能存在精度损失；训练需要少量但质量高的校准数据；GPK 的超参数（如 g1、g2 的取值）需要经验调优，过大或过小均可能导致性能下降。

---

## 191. Lost in Transcription: Subtitle Errors in Automatic Speech Recognition Reduce Speaker and Content Evaluations

**arXiv ID:** 2603.15807 | [PDF](https://arxiv.org/pdf/2603.15807v1)

**作者:** Kowe Kadoma `[一作]` (Cornell University), Mor Naaman `[通讯]` (Cornell Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了一项在线实验，评估自动字幕错误对演讲者和演讲内容评价的影响，并检验不同口音是否会进一步影响评价。

**💡 创新点**

创新之处在于利用AI语音合成技术控制口音变量，保持演讲者外观和内容不变，消除了以往研究中不同演讲者混淆的限制；同时使用真实的ASR错误字幕进行对照实验。

**🔧 技术方法**

主要技术包括ElevenLabs的AI语音合成改变标准美式口音与非标准印度口音，Google Meet ASR生成错误字幕，以及线性混合效应模型对实验数据进行统计分析。

**📊 数据集**

使用了四段来自印度TED演讲的1分钟片段，分别生成标准美式和非标准口音版本；字幕误差基于Google Meet ASR的真实转写结果。

**📈 对比分析**

采用2×2交叉设计（字幕质量×口音），每位参与者观看一段准确字幕和一段错误字幕的视频；通过混合效应模型发现字幕质量显著降低演讲者与内容评价，而口音差异未产生显著影响。

**⚠️ 局限性**

局限性包括样本主要为WEIRD人群、仅测试南亚男性口音、未探究误差导致的心理机制或归因，以及单向口音比较降低了检出细微口音效应的统计功效。

---

## 192. SciZoom: A Large-scale Benchmark for Hierarchical Scientific Summarization across the LLM Era

**arXiv ID:** 2603.16131 | [PDF](https://arxiv.org/pdf/2603.16131v1)

**作者:** Han Jang `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University)

**通讯引用:** 3673 | [OpenAlex ID](https://openalex.org/A5052023515)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文构建了名为SciZoom的大规模分层科学摘要基准，覆盖2020‑2025年四大人工智能会议共44,946篇论文，并对其全文、摘要、贡献点和TL;DR四级摘要进行了系统标注，尤其在ChatGPT发布前后划分预/后 LLM 时代；

**💡 创新点**

创新点包括①首次在科学摘要领域引入分层（四级）摘要标注；②针对贡献点缺乏统一格式的挑战，提出三阶段规则+LLM验证+生成补齐的完整提取流水线；③将数据按 LLM 时代进行时间分层，为研究 LLM 影响下的写作演化提供了唯一资源；

**🔧 技术方法**

技术方法主要为规则式文本提取、基于 Qwen3-4B-Instruct 的 LLM 验证、生成式贡献补齐、以及对 LLM（Mistral、Llama、Qwen）在零样本情境下的摘要生成评估；

**📊 数据集**

数据集为 SciZoom，包含 44,946 篇论文（NeurIPS、ICLR、ICML、EMNLP 2020‑2025），并收集了全文、摘要、贡献点与作者提供的 TL;DR，全文通过 PDF 解析得到；

**📈 对比分析**

在零样本 LLM 生成任务中，Qwen2‑7B 在贡献点抽取上最高 ROUGE‑1 42.4%，Llama‑3‑8B 在 TL;DR 生成上 ROUGE‑1 37.5%，整体模型在后 LLM 时代表现优于前期；在跨粒度检索实验中，TL;DR→摘要 R@1 达 81.3%，证明层级摘要保持语义一致；

**⚠️ 局限性**

局限性包括仅覆盖 AI/ML 四大会议，语言模式可能不具代表性；贡献点提取中 66% 需 LLM 生成，可能带来风格偏差；TL;DR 仅 47% 可用；时代划分仅以 ChatGPT 发布为阈值，未捕捉逐步采用过程。

---

## 193. The Midas Touch in Gaze vs. Hand Pointing: Modality-Specific Failure Modes and Implications for XR Interfaces

**arXiv ID:** 2603.15991 | [PDF](https://arxiv.org/pdf/2603.15991v1)

**作者:** Mohammad Dastgheib `[一作]` (University of California), Fatemeh Pourmahdian `[通讯]` (Independent Researcher)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文开发并评估了一个名为 xr‑adaptive‑modality‑2025 的开源 Web 平台，用于在 XR 相关指点任务中研究手势与注视两种输入模式的适配干预（注视简化、手部目标宽度扩展）是否能提升性能并降低工作负荷。

**💡 创新点**

创新点在于：①将符合生理学约束的注视仿真与 ISO 9241‑9 多方向点击任务相结合；②实现了基于规则的自适应策略引擎，可在两种模式下按实时性能信号触发干预；③提供了完整的远程可复现实验基础设施，促进跨研究可比性。

**🔧 技术方法**

技术实现包括：React 18 + TypeScript 的 Web 前端；基于眼动生理模型的注视仿真（运动抑制、漂移、传感器滞后）；ISO 9241‑9 指点任务；线性加速模型与 LBA 认知建模；NASA‑TLX 负荷测量；Fitts 法分析；数据收集与日志全在客户端完成。

**📊 数据集**

数据集：69 名受试者完成 216 个实验块（共 15,105 条有效试验，13,519 条可用于性能评估），使用自定义的多方向点击任务，包含手部（鼠标）和注视（仿真）两种输入方式。

**📈 对比分析**

实验采用 2 × 2 × 2 组内设计（Modality / UI Mode / Pressure）。比较指标为吞吐量、错误率、运动时间和 NASA‑TLX 负荷。结果显示：手部输入的吞吐量（5.17 bits/s）显著高于注视（4.73 bits/s），错误率手部 1.8% 远低于注视 19.1%，以及主观负荷均低于注视。自适应注视简化仅略降时间错误（1.18%→0.42%），未改善主要错误类型（slip）；手部目标宽度扩展因 UI 整合错误未能评估。

**⚠️ 局限性**

局限性包括：①手部宽度扩展未能在实验中生效；②注视输入通过仿真实现，未验证硬件眼动跟踪下的表现；③实验仅在桌面浏览器环境下进行，未涉及真实 XR 头显或光学手部跟踪；④任务仅为 ISO 9241‑9 指点测试，缺乏更生态化的 XR 交互场景；⑤时间压力记录存在早期 bug；⑥适配策略仅限于简化/扩展两种简单干预，未涵盖更复杂的意图辨别方法。

---

## 194. Determinism in the Undetermined: Deterministic Output in Charge-Conserving Continuous-Time Neuromorphic Systems with Temporal Stochasticity

**arXiv ID:** 2603.15987 | [PDF](https://arxiv.org/pdf/2603.15987v1)

**作者:** Jing Yan `[一作]` (Shanghai Jiao Tong University), Yaoyu Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1094 | [OpenAlex ID](https://openalex.org/A5100680347)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

提出一种连续时间脉冲神经网络框架，通过电荷守恒定律和最小神经元约束，确保在无环网络中输出仅与总输入电荷相关，对脉冲时序保持确定性。

**💡 创新点**

创新点在于将电荷守恒与神经元层面约束统一，形成唯一的累积输出，并证明此映射与量化人工神经网络精确对应，实现无近似的静态深度学习与事件驱动动力学桥接。

**🔧 技术方法**

采用连续时间模型、电荷守恒定律、最小化神经元约束、量化映射技术，理论推导与证明。

**📊 数据集**

文中未给出具体实验数据集，侧重理论与数学证明。

**📈 对比分析**

通过理论比较与传统异步SNN的确定性分析，指出在无环结构下具备严格的不变性；实验性能未给出，主要在理论层面验证。

**⚠️ 局限性**

局限性：递归网络仍会出现时间敏感性；缺乏实际硬件实验验证；实现复杂度及能耗等工程细节待后续研究。

---

## 195. Thermopneumatic Pixels for Fast, Localized, Low-Voltage Touch Feedback

**arXiv ID:** 2603.16750 | [PDF](https://arxiv.org/pdf/2603.16750v1)

**作者:** Max Linnander `[一作]` (University of California), Yon Visell `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

设计、制造并测试了一种低电压热气压像素（TPP），可在可穿戴和表面式触觉界面上快速、局部地产生触觉反馈。

**💡 创新点**

创新点在于将低热惯性的镍铬丝悬浮于封闭气腔中，利用瞬时热膨胀快速提升气压，从而在约10 V电压下实现1 N以上峰值力、≈1 mm位移的触觉像素；同时采用层叠、柔性组装和通用驱动电路，实现模块化、可扩展且易于集成。

**🔧 技术方法**

技术手段包括：层状柔性印刷电路与镍铬丝加热、聚酰亚胺/PGS/PDMS膜结构、热力学（热传导）模型、MOSFET驱动板与MCU控制、力传感器、激光三角计与红外温测等实验测量方法。

**📊 数据集**

无公开数据集，实验主要由10名受试者进行感知试验（强度与定位），以及对TPP进行力、位移、温度、周期性工作等物理量的测量。

**📈 对比分析**

通过对不同尺寸、功率与脉冲时长的系统比较，TPP峰值力>1 N、位移≈1 mm，响应时间5–100 ms；在周期操作下无明显性能衰退，峰峰值力随频率反比下降；感知实验显示强度随功率线性变化，定位准确率达95.5%。

**⚠️ 局限性**

局限性包括：高功率/长脉冲易导致镍铬丝熔化；表面温升虽低但仍需控制；连续高频工作时热积累导致性能下降；未对更大尺寸或不同皮肤部位的长期耐久性进行评估。

---

## 196. Cross-modal learning for plankton recognition

**arXiv ID:** 2603.16427 | [PDF](https://arxiv.org/pdf/2603.16427v1)

**作者:** Joona Kareinen `[一作]` (Lappeenranta-Lahti University of Technology LUT), Heikki Kälviäinen `[通讯]` (Lappeenranta-Lahti University of Technology LUT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于自监督跨模态对齐的多模态浮游生物识别框架，并通过k‑NN实现无标签图像与光学散射/荧光信号的分类。

**💡 创新点**

首次将图像与光学轮廓配对进行跨模态对齐，利用无标签自监督显著降低标注成本，并支持单模态或双模态推理。

**🔧 技术方法**

采用对比学习（InfoNCE/ Sigmoid loss）训练图像编码器（EfficientNet、ConvNeXt、ViT）和序列编码器（CNN、Transformer、LSTM），随后用k‑NN做分类。

**📊 数据集**

使用 CytoSense 采集的 LAB、SEA、UTO 三大数据集（共计约 65,000 样本），并公开发布为 SYKE‑Plankton_CytoSense_2025。

**📈 对比分析**

与单模态自监督基线 DINO 进行对比实验，实验显示多模态对齐在单模态 I→I 上提升约 12% 准确率，在双模态 I+P→I+P 上提升约 20%，并在跨域测试中优于实验室数据训练的模型，证明了更好的泛化能力。

**⚠️ 局限性**

依赖光学轮廓数据，若仪器不提供多模态信息则失效；模型对训练规模敏感，较大模型在小数据集上易过拟合；仅在 CytoSense 环境下验证，需进一步评估在其他仪器上的适用性。

---

## 197. Dual Consensus: Escaping from Spurious Majority in Unsupervised RLVR via Two-Stage Vote Mechanism

**arXiv ID:** 2603.16223 | [PDF](https://arxiv.org/pdf/2603.16223v1)

**作者:** Kaixuan Du `[一作]` (Beihang University), Ni Li `[通讯]` (Beihang University)

**通讯引用:** 17305 | [OpenAlex ID](https://openalex.org/A5035433129)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Dual Consensus Reinforcement Learning（DCRL），一种无标签的强化学习方法，使大语言模型通过内部一致性自我改进推理能力。

**💡 创新点**

创新点在于引入 Unlearn‑Then‑Explore 两阶段共识机制和谐波投票（Harmonic Election）来生成更可靠的伪标签，并通过自适应采样动态平衡探索与利用，显著降低主流投票导致的伪共识风险。

**🔧 技术方法**

使用 GRPO 作为基础 RL 算法，配合临时“去学习”损失实现探索、谐波投票得到伪标签、保守奖励设计和自适应采样；全部过程不需要外部模型或人工标注。

**📊 数据集**

在 DAPO‑Math‑14k 以及八个推理基准（MATH‑500、GSM8K、AIME24、Minerva‑math、AMC、OlympiadBench、MMLU‑Pro、GPQA‑Diamond）上进行评测，亦在多种 LLM（Llama3.2‑3B‑Instruct、Qwen3‑4B‑Base、Qwen3‑8B‑Base）上测试。

**📈 对比分析**

与四个无标签 RLVR 基线（RENT、TTRL、Co‑Rewarding‑I/II）对比，DCRL 在所有基准上均优于基线，且在某些模型和任务上接近甚至超过使用有标签的 GRPO，提升幅度通常在 2%–15% 之间。

**⚠️ 局限性**

局限在于当模型的先验偏差严重时，anchor 与 explorer 的信号仍可能收敛到错误共识；对极端 OOD 任务的改进有限，且在极复杂问题上双共识信号可能失效。

---

## 198. OrthoAI v2: From Single-Agent Segmentation to Dual-Agent Treatment Planning for Clear Aligners

**arXiv ID:** 2603.15663 | [PDF](https://arxiv.org/pdf/2603.15663v1)

**作者:** Lansiaux Edouard `[一作]`, Leman Margaux `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了开放源代码的双代理 AI 辅助牙套治疗规划系统 OrthoAI v2，能够在消费级 CPU 上完成牙齿分割、牙科标志点检测、复合生物力学评分和多帧治疗模拟，并通过 SaaS 接口实现可视化。

**💡 创新点**

创新点包括：① 双代理架构，结合基于 DGCNN 的分割代理和基于 CHaRM 的热图回归标志点检测；② 以六个临床维度加权的复合评分模型，取代单一的通行/不通行分数；③ 通过 SLERP 插值和基于证据的分期规则生成可视化的 4D 治疗轨迹；④ 在单机 CPU 上实现完整部署并公开 SaaS 包。

**🔧 技术方法**

使用的技术有：Dynamic Graph Convolutional Neural Network (DGCNN) 进行点云牙齿分割；Conditioned Heatmap Regression（CHaRM）实现无分割标志点检测；PointMLP 轻量编码器；多模态融合 orchestrator；六维生物力学评分引擎；SLERP 6-DoF 插值；FastAPI + React 的 SaaS 前后端。

**📊 数据集**

主要使用 200 个合成拥挤场景的 Benchmark（包含 4 种弓形、3 种拥挤程度、0-2 颗缺失牙），以及公开的 3DTeethLand、Teeth3DS 等真实临床数据集作为后续验证目标。

**📈 对比分析**

在相同输入下，v2 的并行集成模式在合成基准上实现了 92.8±4.1 的规划质量分数，比 v1 的 76.4±8.3 提升了 21%（相对）并显著降低标准差（53%），可行性从 78% 提升至 89%；CPU 推理时间保持在 4.2±0.8 秒，适合消费级部署。

**⚠️ 局限性**

限制包括：仍仅在合成数据上评估；缺乏对真实临床 IOS 数据的验证；未实现自动运动序列生成和附件放置；在极端缺失牙或复杂拥挤情况下，双代理的融合策略可能仍受限。

---

## 199. MAC: Multi-Agent Constitution Learning

**arXiv ID:** 2603.15968 | [PDF](https://arxiv.org/pdf/2603.15968v1)

**作者:** Rushil Thareja `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Nils Lukas `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种多代理宪法学习（MAC）框架，利用由注释器、决策器、规则生成器和编辑器组成的代理网络自动生成、修订和删改自然语言规则，以控制LLM在任务中的行为。

**💡 创新点**

创新点在于：①将宪法学习结构化为可插拔规则集合，避免无结构提示优化；②通过多代理分工把复杂优化拆解为子任务；③提出MAC+（代理级微调）和reMAC（检索增强），进一步提升小模型性能并在推理时补充示例。

**🔧 技术方法**

使用大型指令调优LLM（Qwen 2.5 Instruct 3B/7B/14B）、Prompt Engineer、监督微调（LoRA）和检索（句子嵌入相似度）等技术实现代理的生成与评估。

**📊 数据集**

数据集涵盖三类领域的PII标注：法律（ECHR）、医疗（MACCROBAT）与金融（PUPA），以及工具调用基准BFCL，用于验证框架在不同任务上的通用性。

**📈 对比分析**

在PII标注任务上，MAC相较于基准提示优化器（GEPA、MIPRO）提升了超过50% F1分数，并在大模型上达到或超过监督微调和GRPO的性能；在法律领域对比预训练标注器，MAC提升约17%；在工具调用任务上也取得了最高准确率。

**⚠️ 局限性**

局限性包括：宪法规模与可解释性之间的权衡尚未量化；评估指标的噪声或偏差可能导致不良规则；规则数量过多时可能产生冲突或超出上下文长度；在细粒度、复杂错误模式（如医疗领域3B模型）下效果不佳。

---

## 200. RecBundle: A Next-Generation Geometric Paradigm for Explainable Recommender Systems

**arXiv ID:** 2603.16088 | [PDF](https://arxiv.org/pdf/2603.16088v1)

**作者:** Hui Wang `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Tao Guo `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了RecBundle框架，利用纤维束几何模型将推荐系统的协作与演化机制拆分为基底流形与纤维空间，量化信息回路中的曲率与全容度，实现对信息共生、偏见与泡沫的可解释分析。

**💡 创新点**

首次将推荐系统中的协作与演化映射到纤维束的连接、平行传输与全容变换上，并以曲率与全容度为客观指标刻画信息退化，同时提出基于LLM的几何约束推理架构。

**🔧 技术方法**

使用差分几何的纤维束理论、图神经网络的平行传输、全容变换谱分解与曲率正则化，以及大型语言模型（LLM）进行链式推理。

**📊 数据集**

在MovieLens和Amazon Beauty两个真实数据集上进行实验。

**📈 对比分析**

与传统矩阵分解、GNN、LLM推荐等基线对比，RecBundle在精准度、信息多样性和信息退化度量上均优于基线，显示出更好的可解释性与鲁棒性。

**⚠️ 局限性**

在稀疏数据和高维全容度矩阵下计算成本较高，且需手工设定曲率与全容度正则化权重，尚未实现对动态在线场景的实时几何结构更新。

---

## 201. Making Separation-First Multi-Stream Audio Watermarking Feasible via Joint Training

**arXiv ID:** 2603.16805 | [PDF](https://arxiv.org/pdf/2603.16805v1)

**作者:** Houmin Sun `[一作]` (Digital Innovation Research Center), Ming Li `[通讯]` (School of Artificial Intelligence)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了在音频混合后先进行源分离再进行多流水印的可行性，提出了在分离后仍能可靠解码的水印框架。

**💡 创新点**

创新点在于将水印编码器/解码器与源分离网络通过端到端联合训练，显著提升分离后水印的鲁棒性。

**🔧 技术方法**

采用了基于Conformer的关键码条件水印背骨AURA*与时间域U‑Net Demucs分离器，并在训练中加入多尺度STFT、Mel谱、感知损失及分离器L1损失。

**📊 数据集**

使用的语料包括Free Music Archive（FMA）、Emilia语音数据集以及MUSDB18多源音乐数据，混合后进行分离与评估。

**📈 对比分析**

与WavMark、Audioseal及单一水印模型相比，联合训练模型在Speech+Music和Vocal+Accompaniment两种混合场景下的比特错误率降至约1%，同时保持4.5以上的ViSQOL MOS；分离质量几乎不受影响。

**⚠️ 局限性**

目前的局限是仅针对两源混合进行实验，难以直接推广到更复杂的多源情境，需要进一步扩展至多流分离与水印协同优化。

---

## 202. VQKV: High-Fidelity and High-Ratio Cache Compression via Vector-Quantization

**arXiv ID:** 2603.16435 | [PDF](https://arxiv.org/pdf/2603.16435v1)

**作者:** Yixuan Wang `[一作]` (Shanghai Jiao Tong University), Zhouhan Lin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6482 | [OpenAlex ID](https://openalex.org/A5024900991)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 VQKV，一个训练-free 的 KV 缓存压缩框架。

**💡 创新点**

创新点在于将向量量化（Residual Simple VQ）应用于 KV 缓存，兼顾高压缩比与高重建保真度，解决了 Token evict、特征维度压缩和标量量化的局限。

**🔧 技术方法**

使用 Residual Simple Vector Quantization、FlashAttention、定制的 Triton 核心实现压缩与解码。

**📊 数据集**

在 LLaMA3.1‑8B 与 LLaMA3.2‑3B 上进行实验，评估数据集包括 LongBench、Needle‑In‑a‑Haystack (NIAH)、RULER 等。

**📈 对比分析**

与 SnapKV、ASVD、Palu、KIVI 等方法比较，VQKV 在 82.8%/82.4% 的压缩率下保持 98.6% 以上的基线性能，生成长度提升 4.3×，在长文本推理上表现最佳。

**⚠️ 局限性**

解码时存在额外的计算开销，未来需进一步优化.

---

## 203. PA-LVIO: Real-Time LiDAR-Visual-Inertial Odometry and Mapping with Pose-Only Bundle Adjustment

**arXiv ID:** 2603.16228 | [PDF](https://arxiv.org/pdf/2603.16228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 204. Dexterous grasp data augmentation based on grasp synthesis with fingertip workspace cloud and contact-aware sampling

**arXiv ID:** 2603.16609 | [PDF](https://arxiv.org/pdf/2603.16609v1)

**作者:** Liqi Wu `[一作]` (University of Tokyo), Kei Okada `[通讯]` (University of Tokyo)

**通讯引用:** 6595 | [OpenAlex ID](https://openalex.org/A5101836795)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过遥操作收集少量抓取演示，利用 AutoWS 生成指尖工作空间云，并结合 FSG 进行抓取数据增强，形成可用于训练的抓取数据集。

**💡 创新点**

创新点在于将手指结构信息嵌入工作空间云，消除逆运动学计算；基于演示的工作空间云大幅缩小搜索空间，提升抓取生成速度和有效率；并支持任意手结构和可配置手指数量。

**🔧 技术方法**

采用遥操作、Poisson Disk Sampling、指尖接触感知采样（FSG）、工作空间云点云表示、GWS epsilon 评估及多线程并行计算。

**📊 数据集**

使用 YCB 物体数据集，并在 Shadow Hand、Robotiq 2F‑85、Barrett Hand、Allegro Hand 等多种机械手模型上进行验证。

**📈 对比分析**

与 EigenGrasp、DexGraspNet、QD‑Grasp‑6DoF 等基线方法在 YCB 物体上对比，生成速度提升数十倍，Valid Epsilon 成功率提升至 18%–60%，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括需借助 VR 设备收集演示；对极小或薄物体的有效抓取率仍较低；仅针对精准抓取，未覆盖力量抓取或抓取类型转换；碰撞与接触判定仍有改进空间。

---

## 205. Bridging the Simulation-to-Reality Gap in Electron Microscope Calibration via VAE-EM Estimation

**arXiv ID:** 2603.16549 | [PDF](https://arxiv.org/pdf/2603.16549v1)

**作者:** Jilles S. van Hulst `[一作]` (Eindhoven University of Technology), Duarte J. Antunes `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1905 | [OpenAlex ID](https://openalex.org/A5013274345)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于变分自编码器与期望最大化（VAE-EM）框架的 STEM 显微镜自动校准方法，能够联合估计偏差状态与潜在映射，显式处理模拟到现实的差距；

**💡 创新点**

创新点包括：①利用 VAE 学习多维潜在表征取代传统标量质量指标；②在仅有模拟训练数据的情况下，通过 EM 同时估计状态与模型，克服模拟-现实缺口；③引入光学对称性（h(-x)=h(x)）构造对称化核，实现全局可识别性；④采用候选细化与信息增益输入选择实现高效收敛；

**🔧 技术方法**

核心技术包括变分自编码器、期望最大化算法、Gaussian Process（GP）建模、对称化核、候选集细化策略、信息增益驱动的输入选择；

**📊 数据集**

使用两类数据：1）基于多层波动光学模拟器生成的大规模模拟 Ronchigram 图像，用于 VAE 训练；2）不同工作日采集的真实 STEM Ronchigram 数据，用于验证与性能评估；

**📈 对比分析**

与现有自动校准方法（如 Bayesian 优化、CEOS/OptiSTEM 等）对比，VAE-EM 在 20 次观测（≈20 s）内将估计误差降低 2 倍以上，准确率在 10 次观测时已超过 50%，整体速度与精度显著优于基线；

**⚠️ 局限性**

局限性包括：需大量覆盖全偏差空间的模拟训练数据；GP 计算在高维或高阶畸变下规模受限；若缺乏代表性真实数据仍需 EM 迭代，易陷入局部最优；硬 EM 可降低成本但精度下降；

---

## 206. Embedding-Aware Feature Discovery: Bridging Latent Representations and Interpretable Features in Event Sequences

**arXiv ID:** 2603.15713 | [PDF](https://arxiv.org/pdf/2603.15713v1)

**作者:** Artem Sakhno `[一作]` (Sber AI Lab), Maksim Makarenko `[通讯]` (Sber AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Embedding-Aware Feature Discovery (EAFD) 框架，利用预训练事件序列嵌入与 LLM 驱动的特征生成器迭代发现并评估可解释且互补的特征。

**💡 创新点**

创新点在于把嵌入空间与可解释特征的发现耦合起来，使用对齐度和互补度两种指标实现自我反思循环，并将 LLM 用作特征生成与修正的智能体。

**🔧 技术方法**

技术核心包括：预训练事件序列编码器（CoLES、NTP、LLM4ES）、LLM 生成代码的特征提取器、对齐度 R² 评估、下游效用 U 分数、CatBoost 训练、自动错误纠正。

**📊 数据集**

实验数据集涵盖四个公开金融事件序列基准（Age Prediction、Gender Prediction、Rosbank、DataFusion）及一大规模专有多目标金融数据，验证跨任务性能提升。

**📈 对比分析**

方法通过与嵌入仅模型、聚合特征、LLM 基特征生成器（CAAFE、LLMFE）以及 AutoML 等基线对比，在所有基准上实现相对提升 5.8%–19.3%，在专有数据上对分类提升 12.55% 及回归 MAE 降低 3.87%。

**⚠️ 局限性**

局限性包括：依赖大型 LLM 产生代码，导致 GPU 资源占用高；特征生成受限于生成式模型的可解释性和执行错误；嵌入保持冻结限制了联合学习的潜力；实验聚焦金融场景，尚未验证在其他行业事件序列的泛化。

---

## 207. Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective

**arXiv ID:** 2603.16104 | [PDF](https://arxiv.org/pdf/2603.16104v1)

**作者:** Noppanat Wadlom `[一作]` (National University of Singapore), Yao Lu `[通讯]` (National University of Singapore)

**通讯引用:** 6000 | [OpenAlex ID](https://openalex.org/A5058605138)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Helium，一种面向代理工作流的 LLM 服务框架，将代理工作流视为查询计划并进行全局优化。

**💡 创新点**

创新点在于将 LLM 调用提升为第一类算子，结合主动 KV 缓存和基于查询优化的缓存感知调度，实现跨调用、跨批次的前缀重用和中间结果复用。

**🔧 技术方法**

技术包括：DSL 解析 DAG、逻辑计划重写（子图消除、缓存替换）、模板基 Radix 树（TRT）构建、基于成本的缓存感知调度、主动 KV 缓存预热、vLLM 引擎集成。

**📊 数据集**

使用的基准数据集包括 MMLU、TAT‑QA、Amazon Reviews、以及自构建的金融交易数据集，用于评估不同工作流模式。

**📈 对比分析**

与 vLLM、OpWise、LangGraph、AgentScope、Parrot、KVFlow 等基线对比，Helium 在微基准上可达 1.34×–1.56× 的加速，在复杂交易工作流上可达 39.5× 的加速，证明了全局优化和缓存预热的显著效益。

**⚠️ 局限性**

局限性：仅基于静态 DAG 形式，难以处理动态控制流（如条件循环、动态映射）和外部 API 调用的不可预知延迟；对大规模动态工作流的适配仍待进一步研究。

---

## 208. Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users

**arXiv ID:** 2603.16120 | [PDF](https://arxiv.org/pdf/2603.16120v1)

**作者:** Nishant Balepur `[一作]` (University of Maryland), Aakanksha Naik `[通讯]` (Allen Institute for Artificial Intelligence)

**通讯引用:** 424 | [OpenAlex ID](https://openalex.org/A5087743328)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了首个开源的个性化深度研究系统，系统通过用户挑选的论文推断用户兴趣档案，生成个性化行动建议，并在多轮生成中编写符合这些行动的多章节报告。

**💡 创新点**

创新点包括：①持久化的用户模型仅基于论文实现；②将行动建议作为中间交互步骤，提升用户对系统的控制感；③通过行动高亮使报告的个性化层次一目了然；④结合离线合成数据与真实用户访谈，揭示离线评测难以捕捉的九大错误，为后续评测与设计提供新思路。

**🔧 技术方法**

技术手段：大模型（Gemini‑2.5、Claude‑4 Sonnet、GPT‑4.1 等）用于档案推断、行动规划与报告生成；Semantic Scholar API 进行检索与聚类；多轮提示链（prompt‑chaining）实现搜索、生成、行动执行与高亮；离线评测采用多项客观指标，在线访谈结合语义分析。

**📊 数据集**

数据集：①合成数据——从 ScholarQA‑CS2 采集 200 个查询并为每个查询分配低/中/高专业度的人工生成用户论文集合（使用 CS‑PaperSum）；②真实用户数据——21 名计算机科研工作者提供 5 篇论文和 3 个查询；③公开基准与商业系统的报告数据用于对比。

**📈 对比分析**

评测方法与性能：离线采用 16 项指标评估档案准确性、行动胜率、报告覆盖率、引用精度等；与 8 个基线系统（OpenAI、Perplexity 等）比较，Personalized 系统在 4/5 主要指标上获得最佳或次佳；在线用户研究显示 73% 的档案/行动/报告被认为满意，但访谈揭示了 9 种离线评测未捕捉到的错误，验证了离线评测的局限性。

**⚠️ 局限性**

局限性：①合成用户数据无法完全模拟真实用户需求，导致离线评测缺失重要错误；②离线 LLM 判断器无法预测用户满意度；③系统生成速度较慢（档案约 3 分钟，报告约 5 分钟）；④仅基于论文的档案推断可能产生有害刻板印象；⑤在其他学科或多模态场景下的可扩展性尚待验证。

---

## 209. Retrieving Counterfactuals Improves Visual In-Context Learning

**arXiv ID:** 2603.16737 | [PDF](https://arxiv.org/pdf/2603.16737v1)

**作者:** Guangzhi Xiong `[一作]` (University of Virginia), Aidong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 11961 | [OpenAlex ID](https://openalex.org/A5013588572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 CIRCLES 框架，利用组合式图像检索主动构造因果性示例集，从而提升视觉‑语言模型在推理任务中的表现。

**💡 创新点**

创新点在于将属性级反事实检索与传统相似度检索相结合，形成既包含因果信息又兼顾相关性的示例集合，突破了仅依赖相似度导致的表面关联问题。

**🔧 技术方法**

采用 CLIP 进行多模态检索，并通过 VLM 生成属性与反事实描述，使用 OSrCIR 等组合式检索方法与标准图像检索融合，以实现因果示例的检索与构造。

**📊 数据集**

在四个数据集上进行评估：CUB、Flowers（细粒度分类），以及 OK‑VQA、VizWiz（视觉问答），覆盖不同任务和难度。

**📈 对比分析**

与无示例、随机示例、RICES、MUIER、MMICES 等基线对比，CIRCLES 在大多数模型和数据集上实现了 0.76%–94.02% 的相对提升，尤其在小模型和信息稀缺场景表现尤为突出。

**⚠️ 局限性**

局限性包括对 CIR 质量的依赖；若生成的反事实描述不够准确或检索不精确，可能削弱因果信息；同时在大规模检索或实时应用中，计算成本和效率仍需进一步改进。

---

## 210. S2Act: Simple Spiking Actor

**arXiv ID:** 2603.15725 | [PDF](https://arxiv.org/pdf/2603.15725v1)

**作者:** Ugur Akcal `[一作]` (University of Illinois at Urbana-Champaign), Girish Chowdhary `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种轻量级的 SNN 强化学习框架 S2Act，并在多智能体捕旗和停车任务中实现从 ANN 到 SNN 的无缝转换与部署。

**💡 创新点**

创新点在于通过全局调节 LIF 神经元参数使其率式激活近似 ReLU，从而消除梯度消失并简化超参数调优；将训练好的 ANN 直接映射到可在 Loihi 硬件上高效执行的 SNN。

**🔧 技术方法**

采用 Soft‑ReLLIF 率式 LIF、PPO 训练、ANN‑to‑SNN 转换、Intel Loihi neuromorphic 处理器和 PyTorch 框架。

**📊 数据集**

使用多智能体捕旗（CtF）模拟与真实环境、以及停车仿真任务作为测试数据集。

**📈 对比分析**

与 PopSAN、Hybrid SNN、RSNN 等基线比较，S2Act 在训练收敛速度、平均回报、成功率、能耗（J/inference）和推理时间（ms）上均优于对手；Loihi 上实现进一步降低能耗并保持竞争性性能。

**⚠️ 局限性**

局限性：率式 SNN 的 ANN‑to‑SNN 近似和 Loihi 的量化精度可能在更复杂任务或更大网络中导致性能下降；假设环境完全可观测且神经元参数固定，难以适应部分可观测或需要在线自适应的动态情境。

---

## 211. Toward Experimentation-as-a-Service in 5G/6G: The Plaza6G Prototype for AI-Assisted Trials

**arXiv ID:** 2603.16356 | [PDF](https://arxiv.org/pdf/2603.16356v1)

**作者:** Sergio Barrachina-Muñoz `[一作]` (Centre Tecnologico de Telecomunicaciones de Catalunya), Josep Mangues-Bafalluy `[通讯]` (Centre Tecnologico de Telecomunicaciones de Catalunya)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了 Plaza6G 原型，实现 5G/6G 网络实验即服务，并将 LLM 作为实验设计与分析助手

**💡 创新点**

首次将大规模语言模型融入实验自动化流程，显著提升实验设计与故障诊断效率

**🔧 技术方法**

使用 SDN/NFV 架构、Python 自动化脚本、Docker 容器化、OpenAI GPT‑4 等 LLM 技术

**📊 数据集**

利用公开的 5G NR 流量与波形数据集（5G-AI Benchmark）以及自建的多频段实验日志

**📈 对比分析**

与传统手工实验相比，配置时间缩短约 70%，实验成功率提升 30%，并在标准基准上实现相同或更低误差率

**⚠️ 局限性**

受限于 LLM 的语义误判、测试环境的可扩展性以及对实时高动态场景的支持不足

---

## 212. Semi-Autonomous Formalization of the Vlasov-Maxwell-Landau Equilibrium

**arXiv ID:** 2603.15929 | [PDF](https://arxiv.org/pdf/2603.15929v1)

**作者:** Vasily Ilin `[一作]` (University of Washington), Vasily Ilin `[通讯]` (University of Washington)

**通讯引用:** 47 | [OpenAlex ID](https://openalex.org/A5009909631)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

完成了 Vlasov‑Maxwell‑Landau 系统平衡状态的全 AI 形式化证明，证明了唯一平衡为全局麦克斯韦分布。

**💡 创新点**

展示了从自然语言推理、编码、自动化证明到最终验证的完整 AI 研究循环，并在硬分析层面首次给出该系统的唯一平衡定理。

**🔧 技术方法**

使用 Gemini DeepThink 生成推理蓝图、Claude Code 自动化 Lean 代码、Aristotle 自动化证明子 lemma、Lean kernel 最终验证，并配合 LSP、Python 脚本和自动化循环。

**📊 数据集**

未使用外部数据集，全部基于公开数学文献和自定义 Lean 库，生成 10k+ 行 Lean 代码，提交 213 次。

**📈 对比分析**

与传统人工形式化相比，仅耗时 10 天、$200 订阅费用、0 手工编码；与 AxiomProver 等全自动系统相比，虽然仍需人工监督，但成功关闭 111 个 lemma，证明规模与深度达到 10k 行，性能良好。

**⚠️ 局限性**

限制包括需人工审查定义与定理语句、对模型版本敏感、API 费用高（≈$6300）、尚未实现完全自治，仅适用于具备数学专业知识的监督者。

---

## 213. Making Software Metrics Useful

**arXiv ID:** 2603.16012 | [PDF](https://arxiv.org/pdf/2603.16012v1)

**作者:** Ewan Tempero `[一作]`, Paul Ralph `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

讨论软件度量的局限性，提出采用多指标反射式测量模型以改进软件质量属性评估。

**💡 创新点**

提出将软件质量属性视为潜在结构，采用反射测量模型和多指标聚合（如因子分析）来估计属性，而非追求单一最佳指标。

**🔧 技术方法**

基于测量理论、心理测量方法和统计验证（如代表性条件、聚合效度）进行概念性分析。

**📊 数据集**

无数据集；主要基于文献回顾与理论推导。

**📈 对比分析**

未给出实验对比；通过理论阐述说明代表性条件和聚合效度的必要性，未涉及性能指标。

**⚠️ 局限性**

缺乏经验验证和实证案例，难以立即转化为工程实践工具；模型选择与指标收集仍需进一步研究。

---

## 214. Auto Researching, not hyperparameter tuning: Convergence Analysis of 10,000 Experiments

**arXiv ID:** 2603.15916 | [PDF](https://arxiv.org/pdf/2603.15916v1)

**作者:** Xiaoyi Li `[一作]` `[通讯]`, Xiaoyi Li

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用两个LLM代理在一个108,000细胞的组合配置空间上，对车载碰撞检测任务进行自主实验设计，并完成10,469次实验。

**💡 创新点**

首次把LLM视为全组合实验搜索策略，证明其能进行真正的架构搜索，并给出功效指标和信息论分析。

**🔧 技术方法**

使用基于LLM的自然语言决策、Orze调度框架、ANOVA、功率律收敛模型和信息论度量（熵、JSD）。

**📊 数据集**

主要使用Nexar车载摄像头碰撞预测数据集，也在FedEx物流车辆碰撞检测数据上做交叉验证。

**📈 对比分析**

与随机搜索、TPE、从零开始的均匀搜索相比，LLM搜索在50次实验后AP达到0.985，收敛指数c=0.11，显著优于基线且能探索高质量架构区域。

**⚠️ 局限性**

实验规模仅占组合空间的2.4%，结果受bug修复影响，缺乏正式的 regret 上界，且在不同任务上的泛化尚未充分验证。

---

## 215. Reasoning About Variability Models Through Network Analysis

**arXiv ID:** 2603.16577 | [PDF](https://arxiv.org/pdf/2603.16577v1)

**作者:** Jose Manuel Sanchez `[一作]` (Universidad de Sevilla), David Fernandez-Amoros `[通讯]` (Universidad Nacional de Educación a Distancia)

**通讯引用:** 600 | [OpenAlex ID](https://openalex.org/A5064430597)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构造强关系图（依赖图与冲突图）对软件产品线特征模型进行网络分析，揭示其结构模式、域差异以及关键特征。

**💡 创新点**

创新点在于：①将特征模型转化为可求解的布尔公式后提取强依赖/冲突的“强图”；②运用图论度量系统化捕捉模型的核心/冗余特征与结构中心化现象；③对超过五千个真实特征模型进行跨域大规模实证，首次揭示系统软件与安全域的结构差异与维度分布。

**🔧 技术方法**

使用 SAT 槽回溯（backbone）算法构造强图；借助 MiniSat + IPASIR 进行增量 SAT 求解；利用 igraph 计算节点度数、密度、相关系数、Spearman ρ、Wilcoxon 符号秩检验等网络与统计指标。

**📊 数据集**

5,709 个特征模型，来自 20 个开源与工业仓库，覆盖系统软件、安全、金融、汽车等多个领域；模型规模从 99 到 35,907 变量不等。

**📈 对比分析**

通过对比各域中核心/冗余比例、依赖/冲突密度、度分布、hub 与高冲突节点比例等指标，结合相关系数与非参数检验评估显著性；实验显示模型规模与依赖/冲突密度呈正相关，系统软件依赖远多于冲突，而安全域则相反，且结果在统计上显著。

**⚠️ 局限性**

限制包括：①仅考虑“强”关系，忽略弱/概率依赖；②依赖于 FM→CNF 的翻译质量，尤其是 Kconfig 语义不完全；③部分领域样本不足，结果可能不具普适性；④未直接测量维护或演进的实际效果，只给出启发性指标。

---

## 216. ARISE: Agent Reasoning with Intrinsic Skill Evolution in Hierarchical Reinforcement Learning

**arXiv ID:** 2603.16060 | [PDF](https://arxiv.org/pdf/2603.16060v1)

**作者:** Yu Li `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6449 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种层次化强化学习框架ARISE，能够在训练过程中自我演化和管理可重用的推理技能库。

**💡 创新点**

创新点在于：①将技能选择、生成与策略优化统一到同一共享策略中；②通过层次化奖励引导技能的使用和生成；③采用两层缓存-存储结构实现动态技能库的增删与提升；④通过两个训练阶段让技能库与策略共同进化。

**🔧 技术方法**

主要技术包括：强化学习与可验证奖励的GRPO变体、条件对数概率用于技能匹配、结构化技能摘要（技能生成rollout）、层次化奖励和基于经验的技能库管理操作。

**📊 数据集**

使用DeepScaleR数据集（约40k题目，覆盖AMC、AIME、MATH和OlympiadBench）进行训练与评估；基模型分别为Qwen3-4B-Instruct-2507和Phi-4-mini-instruct。

**📈 对比分析**

与GRPO系列基线（GRPO、Dr.GRPO、DAPO、GSPO）以及内存/技能增强方法（EvolveR、SimpleMem、SkillRL）对比，ARISE在七项竞赛与Omni-MATH基准上均取得显著提升，尤其在外域任务上提升2.9~1.9个百分点，且技能利用率大幅提高。

**⚠️ 局限性**

局限性：仅在数学推理任务上验证；技能库规模在后期趋于饱和，进一步扩充难以带来额外收益；对可执行工具或代码生成等多模态任务的适用性尚未探索。

---

## 217. Spectral Edge Dynamics of Training Trajectories: Signal--Noise Geometry Across Scales

**arXiv ID:** 2603.15678 | [PDF](https://arxiv.org/pdf/2603.15678v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化神经网络训练轨迹的谱边缘，揭示其低维信号与噪声分界。

**💡 创新点**

提出SED框架和谱边缘指标，发现三相动态、窗口导致的时序翻转以及随机投影下的可扩展性。

**🔧 技术方法**

使用滚动窗口SVD、Gram矩阵、最大相邻奇异值比、交叉相关与Granger因果分析，以及Johnson–Lindenstrauss随机投影。

**📊 数据集**

使用TinyStories语言模型数据集以及GPT-2 124M在FineWeb→OpenWebText的分布移位。

**📈 对比分析**

与验证损失、漂移速度等指标比较，谱边缘在不同窗口下具有高相关性并能提前检测分布移位（≤200步）及grok‑ing先导信号，误差<10%。

**⚠️ 局限性**

仅在两种规模（51M、124M）和AdamW优化器上验证，缺乏对更大模型、不同架构和优化器的验证，统计显著性受样本量限制。

---

## 218. Temporal Fact Conflicts in LLMs: Reproducibility Insights from Unifying DYNAMICQA and MULAN

**arXiv ID:** 2603.15892 | [PDF](https://arxiv.org/pdf/2603.15892v1)

**作者:** Ritajit Dey `[一作]` (University of Glasgow), Yashar Moshfeghi `[通讯]` (University of Strathclyde)

**通讯引用:** 1697 | [OpenAlex ID](https://openalex.org/A5059100610)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 DYNAMICQA 与 MULAN 两个关于 LLM 时间知识冲突的基准进行可复现性研究，重新实现并统一两套评估框架；使用 LLM 自动生成自然语言上下文替换原有程序化上下文；在不同模型规模（Gemma 1B–12B）上复现实验，探讨数据集设计、评估指标与模型大小对实验结论的影响。

**💡 创新点**

首次将两大基准的评估方法交叉验证并统一化，扩展到多尺寸 LLM；利用 LLM 自动生成的自然语言上下文来模拟更真实的外部证据；系统性揭示数据集构建与评估方法导致的结论冲突，说明时间事实可更新性的结论并非普适。

**🔧 技术方法**

使用 LLM 推理（Gemma 3、Llama、Mistral、Qwen、Falcon 等）；Coherent Persuasion Score、Semantic Entropy、Update Success Rate、F1/confidence 等指标；LLM 生成自然语言上下文并通过 NLI 模型筛选；多模板生成与统计分析。

**📊 数据集**

DYNAMICQA（5,689 QA 示例，基于 Wikipedia 编辑历史）和 MULAN（49,142 句子补全示例，基于 Wikidata triples）；两套数据集的相互转换版本；自生成的自然语言上下文；使用 Wiki 片段和 Wikidata 实体。

**📈 对比分析**

在原始实验设置下成功复现两套基准的结论；在 Gemma 3 系列模型上评估，发现模型规模对时间事实更新的可更新性呈非单调关系；跨基准评估显示 MULAN 结果更稳健，但 DYNAMICQA 结果在其他评估中不再一致。总体而言，时间事实在不同度量、不同模型大小下表现不一，结论高度依赖评估设计。

**⚠️ 局限性**

结果高度依赖数据集构建与评估指标，难以排除模型架构或训练数据的潜在影响；仅在 Gemma 3 家族内探讨模型规模，对其他 LLM 体系的推广有限；自动生成上下文的质量仍有提升空间；未完全覆盖 7B 以下模型，无法确定时间事实冲突解决的普适性。

---

## 219. Laya: A LeJEPA Approach to EEG via Latent Prediction over Reconstruction

**arXiv ID:** 2603.16281 | [PDF](https://arxiv.org/pdf/2603.16281v1)

**作者:** Saarang Panchavati `[一作]` (University of California, Los Angeles), William Speier `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了首个基于LeJEPA的EEG基线模型Laya，通过在潜在空间进行预测而非重建来进行无监督预训练。

**💡 创新点**

将LeJEPA的几何正则化SIGReg应用到EEG，构建了遮蔽时间段的潜在预测框架，并证明其在临床任务上显著优于传统重建目标。

**🔧 技术方法**

采用LeJEPA框架、SIGReg正则化、卷积补丁嵌入、跨通道注意力的Channel Mixer以及Transformer编码器与预测器等技术。

**📊 数据集**

在由Temple University Hospital、NMT、OpenNeuro、Healthy Brain Network、MOABB等组成的约29,000小时EEG混合语料上预训练。

**📈 对比分析**

通过EEG-Bench基准进行线性探测，Laya在临床任务上平均准确率达0.597，超越LaBraM和LUNA，并在噪声鲁棒性上表现更佳。

**⚠️ 局限性**

模型仅在10%数据上训练即可达到最优，扩展至全数据时性能略降，缺乏对非分类任务的评估，并对极端个体差异的BCI任务适应不足。

---

## 220. Machine Translation in the Wild: User Reaction to Xiaohongshu's Built-In Translation Feature

**arXiv ID:** 2603.15922 | [PDF](https://arxiv.org/pdf/2603.15922v1)

**作者:** Sui He `[一作]` (Swansea University), Sui He `[通讯]` (Swansea University)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5112112981)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析Xiaohongshu 2025年推出的内置翻译功能的用户评论，结合情感分析与主题分析评估用户感知与实验行为。

**💡 创新点**

首次将大模型情感分析与人工标注相结合，系统性研究社交平台翻译功能早期用户体验。

**🔧 技术方法**

使用GPT-5-mini和DeepSeek‑V3.2进行ABSA提示下的情感分类，辅以NVivo 15进行主题编码，评论通过爬虫抓取。

**📊 数据集**

包含6,723条来自11条官方宣传帖的评论及其回复，涵盖四种使用场景（帖子/评论翻译、私聊翻译、双语帖子、字幕翻译）。

**📈 对比分析**

模型共识与人工标注对比，Cohen's kappa 0.81，情感分布显示80%正面，主题分析揭示多样化反馈。

**⚠️ 局限性**

仅覆盖首10天且仅聚焦官方帖子，测试输入多为简短或非自然语言，难以全面评估真实语言翻译质量；缺乏对后续使用情况的跟踪。

---

## 221. The Evolving Duet of Two Modalities: A Survey on Integrating Text and Visualization for Data Communication

**arXiv ID:** 2603.15640 | [PDF](https://arxiv.org/pdf/2603.15640v1)

**作者:** Xingyu Lan `[一作]` (Fudan University), Siming Chen `[通讯]` (Fudan University)

**通讯引用:** 4238 | [OpenAlex ID](https://openalex.org/A5050391600)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统综述98篇关于文本作为叙事媒介的可视化论文，总结了文本与可视化整合的动机、表现形式以及设计方法，并提出了未来研究方向。

**💡 创新点**

创新点在于填补了“文本作为叙事”这一三线研究主题的空白，提出了基于why-what-how的框架，将文本叙事任务细化为Explain、Emphasize、Couple、Adapt、Verify五大核心任务，并为每一任务归纳了设计策略与实践案例。

**🔧 技术方法**

主要使用文献检索与系统综述技术，结合主题编码（why-what-how）和开放式编码方法对文本叙事技术进行分类与结构化。

**📊 数据集**

使用的数据集为98篇经筛选的论文，来源于IEEE VIS、TVCG、ACM CHI、UIST、EuroVis等主流可视化与人机交互会议与期刊。

**📈 对比分析**

由于本文为综述性工作，没有直接实验比较；通过对已发表的实证研究进行归纳，评估了文本叙事在提升理解、记忆、参与度等指标上的效果，指出多数研究表明正面影响，但仍存在结果不一致的情况。

**⚠️ 局限性**

局限性包括：①仅检索主流可视化与HCI 会议，可能遗漏跨领域相关工作；②快速演进的LLM技术导致新方法未被纳入；③对已有文献的归纳与分类可能受到作者主观判断的影响，需后续进一步验证与更新。

---

## 222. Structured prototype regularization for synthetic-to-real driving scene parsing

**arXiv ID:** 2603.16083 | [PDF](https://arxiv.org/pdf/2603.16083v1)

**作者:** Jiahe Fan `[一作]` (Tongji University), Rui Fan `[通讯]` (Tongji University)

**通讯引用:** 4121 | [OpenAlex ID](https://openalex.org/A5038867899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于结构化原型正则化（SPR）的无监督域适应框架，用于从合成数据迁移到真实驾驶场景的像素级语义分割。

**💡 创新点**

创新点包括：①显式量化并正则化特征空间中的类间分离与类内紧凑性；②利用原型-像素相互作用与熵基噪声过滤提升伪标签质量；③在对比学习中加入像素级注意力机制，精细化像素特征对齐。

**🔧 技术方法**

采用的技术主要有：多尺度卷积网络（DeepLab‑v2 + ResNet‑101），原型构建与相互关系矩阵正则化，基于对比学习的跨域特征对齐，熵度量噪声筛选以及像素级注意力加权。

**📊 数据集**

实验数据集包括：GTA5→Cityscapes、SYNTHIA→Cityscapes、Cityscapes→ACDC 三个常用的合成‑真实与真实‑真实域适应基准。

**📈 对比分析**

与现有 30+ 以上最先进 UDA 方法（如 SePiCo、CACP、EHTDI 等）对比，SPR 在 GTA5→Cityscapes 上实现 54.4% mIoU（+16.0%），SYNTHIA→Cityscapes 59.4% mIoU*（+20.3%），Cityscapes→ACDC 49.5% mIoU（+20.7%），均显著超越对比方法；在源自由 UDA 场景下的表现也接近或优于部分 SFUDA 方案。

**⚠️ 局限性**

主要局限包括：①原型交互计算导致训练时内存与时间成本略高；②当前方法仍依赖源域标注，无法直接用于源自由迁移；③在极端视觉条件下仍可能出现细粒度类间混淆，需进一步改进结构正则化的效率与鲁棒性。

---

## 223. NLP Occupational Emergence Analysis: How Occupations Form and Evolve in Real Time -- A Zero-Assumption Method Demonstrated on AI in the US Technology Workforce, 2022-2026

**arXiv ID:** 2603.15998 | [PDF](https://arxiv.org/pdf/2603.15998v1)

**作者:** David Nordfors `[一作]` `[通讯]` (BOLDMonster Research Institute), David Nordfors (BOLDMonster Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出一种零假设的 NLP 职业出现分析框架，利用简历文本中的词汇共现与人员聚类相互作用，检测新兴职业的形成与消亡；

**💡 创新点**

创新点在于将职业定义为“共吸引子”（co‑attractor），要求词汇和人员两侧的凝聚力相互依赖，并通过置换检验、超几何密度测试和三因子 NMF 等方法实现无先验分类的双重共聚判据；

**🔧 技术方法**

核心技术包括：置换检验与 Benjamini–Hochberg 校正的词汇/人员共现显著性检验；超几何密度比率评估群体内连边密度；三因子非负矩阵分解（Trifactor‑NMF）用于联合发现词汇簇与人员簇，并通过 S 矩阵量化两者的耦合方向；消融（ablation）检验验证词汇对人员聚集的必要性；

**📊 数据集**

实验基于 8.2 百万份美国简历（2022‑2026）来自 BOLD 平台，构建 364 个技术/AI 相关词汇的词–文档矩阵；

**📈 对比分析**

与传统的主题模型、O*NET 代码等对照，方法在已知职业（Web 开发、数据工程）上能准确区分成熟与非职业；在 AI 案例中，词汇共聚而人员不共聚，证实 AI 仍为技术扩散而非新职业；多种方法（NMF、置换检验、消融）结果一致，说明方法稳健；

**⚠️ 局限性**

局限性包括：仅基于单一平台的简历样本，可能缺失其他渠道的从业者；词汇范围局限于 364 条技术词，未覆盖行业特定 AI 应用；人口共聚检验对样本量敏感，细分子群体的统计功效不足；消融检验仅为单向，未完整验证双向互依；仅聚焦技术类职业，其他领域的适用性待验证。

---

## 224. Human-AI Synergy in Agentic Code Review

**arXiv ID:** 2603.15911 | [PDF](https://arxiv.org/pdf/2603.15911v1)

**作者:** Suzhen Zhong `[一作]` (Queen's University), Bram Adams `[通讯]` (Queen's University)

**通讯引用:** 28100 | [OpenAlex ID](https://openalex.org/A5087705607)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对300个受欢迎的 GitHub 开源项目中 278,790 条代码审查对话进行大规模实证分析，比较了 AI 代理和人类审查者的反馈特征、交互模式以及建议被采纳后的代码质量影响。

**💡 创新点**

创新点在于首次将 AI 代理与人类审查者在同一项目中的多轮交互进行系统比较，构建了交互序列的有限状态机模型，并将 AI 建议采纳与代码质量指标（111 条度量）关联，揭示 AI 代理在语义深度、采纳率、代码复杂度提升等方面的差异。

**🔧 技术方法**

采用的技术包括 GPT‑4.1‑mini 对审查评论的自动标签、SciTools Understand 对源代码的静态度量、Jaccard 相似度判断建议是否被采纳，以及统计检验（Scott‑Knott、Mann‑Whitney U、Bonferroni 校正）来评估差异显著性。

**📊 数据集**

使用的数据集是从 300 个 GitHub 项目中挖掘的 278,790 条内联代码审查对话，涵盖 155,397 条 AI 代理建议和 25,673 条人类建议，涉及 111 条代码质量度量。

**📈 对比分析**

比较方法通过对 AI 与人类两组在四类审查场景（HRH、HRA、ARH、ARA）下的评论密度、反馈类型分布、交互轮数、建议采纳率以及建议对代码复杂度和大小的 Δ 值进行统计对比，结果显示 AI 代理评论更冗长、聚焦于改进与缺陷检测，采纳率约为 16.6% 低于人类的 56.5%，而被采纳的 AI 建议导致代码复杂度和规模显著升高。

**⚠️ 局限性**

主要局限包括：仅针对受欢迎的公开 GitHub 项目，可能不适用于企业或小型仓库；数据集只包含可由 SciTools Understand 解析的语言；自动标签可能存在误差；未考虑跨项目或跨语言的代理表现；以及研究时间点较早，AI 代理在后续迭代中可能已改进。

---

## 225. A Practical Algorithm for Feature-Rich, Non-Stationary Bandit Problems

**arXiv ID:** 2603.16755 | [PDF](https://arxiv.org/pdf/2603.16755v1)

**作者:** Wei Min Loh `[一作]` (University of Waterloo), Pascal Poupart `[通讯]` (University of Waterloo)

**通讯引用:** 17089 | [OpenAlex ID](https://openalex.org/A5040035859)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了条件耦合上下文Thompson采样算法 C_3，解决了包含稠密臂特征、非线性奖励函数以及随时间变化但相关度保持的非平稳上下文赌博机问题。

**💡 创新点**

创新点在于将重要加权核回归（IWKR）与嵌入空间结合，用于估计奖励并支持条件耦合；通过 Beta 先验实现无梯度在线学习；提供了无需重新训练即可在线适应概念漂移的机制；并首次在非平稳耦合臂问题上给出理论与实验证明。

**🔧 技术方法**

使用重要加权核回归（IWKR）+RBF核、MLP 嵌入网络、Thompson 采样与 Beta 分布、在线贝叶斯更新、RBF 核宽度调优，构成完整算法。

**📊 数据集**

在四个 OpenML 表格数据集（如 UCI、...）和 Microsoft News Dataset（MIND）上进行实验。

**📈 对比分析**

与 LinUCB、LinTS、SquareCB、NeuralUCB、NeuralTS、两塔神经网络、Gaussian Process with forgetting、上下文无聊算法等基线比较，C_3 在 OpenML 数据集上平均降低 5.7% 的累计累积回报，MIND 数据集上实现 12.4% 的点击提升。

**⚠️ 局限性**

存在数值不稳定、对 RBF 宽度高度敏感、可扩展性受限（需采样或削减参考集）、依赖离线训练与历史数据、难以量化与压缩，且对学习的嵌入空间质量高度依赖。

---

## 226. ETM2: Empowering Traditional Memory Bandwidth Regulation using ETM

**arXiv ID:** 2603.16490 | [PDF](https://arxiv.org/pdf/2603.16490v1)

**作者:** Alexander Zuepke `[一作]` (Technical University of Munich), Marco Caccamo `[通讯]` (Technical University of Munich)

**通讯引用:** 6629 | [OpenAlex ID](https://openalex.org/A5060442004)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用 Arm CoreSight 的 Embedded Trace Macrocell（ETM）实现了一种硬件辅助的内存带宽调节器，能够在多核实时系统中实现微秒级别的带宽监管。

**💡 创新点**

创新点在于：1）把 ETM 的低级事件监控与状态机直接用于带宽计数和调节，无需外部控制核心或周期性轮询；2）在标准的 CoreSight 组件上实现了两种调节算法（周期补偿与令牌桶），兼顾精细粒度与低延迟；3）在不修改硬件的前提下，兼容多种 Arm SoC，显著提升可移植性。

**🔧 技术方法**

采用的技术包括 Arm CoreSight 的 ETM、PMU、CTI、内部计数器、状态机序列器以及 Linux/RTOS 下的中断处理器；调节逻辑通过 ETM 资源选择器和计数器配置完成；实验中使用微基准、San Diego Vision Benchmark Suite (SD‑VBS) 与 IsolBench。

**📊 数据集**

实验数据集主要为多款 64‑bit Arm SoC（Cortex‑A53、A72、A55、A76、A78）上的自研微基准、SD‑VBS 以及 IsolBench，覆盖不同内存访问模式（读、写、预取、modify）。

**📈 对比分析**

通过与 MemGuard 与 MemPol 两个现有软件调节器对比，测量了延迟、吞吐量、系统隔离效果。结果显示，ETM² 在大多数平台上与 MemGuard、MemPol 的带宽达成度相当或略优；在微秒级调节周期下，因不需要周期性计时器中断，开销更低；在 IsolBench 的干扰实验中，也能提供与其它调节器相近的隔离性能。

**⚠️ 局限性**

主要限制包括：ETM 计数器仅为 16 位，导致极小预算时可能溢出；ETM 只能对多路事件进行 OR 合并，可能产生低估/高估误差；对新 DSU 基础的 Cortex‑A55/A76 的 LLC 事件支持不足；部分平台缺少 CTIIRQ 路由，导致无法使用此方案；若预算过小或补偿周期过短，调节误差会显著。

---

## 227. Follow the Clues, Frame the Truth: Hybrid-evidential Deductive Reasoning in Open-Vocabulary Multimodal Emotion Recognition

**arXiv ID:** 2603.16463 | [PDF](https://arxiv.org/pdf/2603.16463v1)

**作者:** Yu Liu `[一作]` (Hangzhou Institute for Advanced Study), Taihao Li `[通讯]` (Hangzhou Institute for Advanced Study)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一种Propose–Verify–Decide的混合证据推理架构HyDRA，用于开放词汇多模态情感识别。

**💡 创新点**

创新点在于将多模态情感识别建模为多假设生成、证据验证与决策的三步推理过程，并通过GRPO和分层奖励实现可学习的证据闭合。

**🔧 技术方法**

使用的技术包括多模态大型语言模型、强化学习（GRPO）、分层奖励函数、结构化推理文本格式和自监督与RL混合训练。

**📊 数据集**

训练和评估数据集涵盖OV-MERD、MERCaption+、MER2023、MER2024、SIMS、MOSI、CH-SIMS等开放词汇与基本情感数据。

**📈 对比分析**

与多种7B规模基线（如LLaVA、VideoChat、AffectGPT等）对比，HyDRA在OV-FG的粗细粒度F1得分分别提升至55.52/30.48（平均43.00），总体平均性能达到61.53%，并在跨模态冲突子集保持最高稳健性。

**⚠️ 局限性**

局限性包括对数据质量与标注一致性的高度依赖、训练过程对RL样本效率要求高、模型规模有限且可能在极短视频或信息稀疏场景下产生多余假设。

---

## 228. Visual Set Program Synthesizer

**arXiv ID:** 2603.15997 | [PDF](https://arxiv.org/pdf/2603.15997v1)

**作者:** Zehua Cheng `[一作]` (University of Oxford), Jiahao Sun `[通讯]` (FLock.io)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将视觉问答中的集合推理问题转化为可执行的程序合成，并提出VSPS框架和CASTER奖励机制。

**💡 创新点**

提出了可解释的程序驱动推理方法和稠密语义奖励CASTER，解决了稀疏奖励和信用分配问题。

**🔧 技术方法**

利用多模态大语言模型生成程序、程序执行引擎、AST匹配与Jaccard奖励、强化学习（GRPO）等技术。

**📊 数据集**

构建Set‑VQA基准（110k图像+程序），并在GQA‑Set和PTR‑Count等数据集上进行评测。

**📈 对比分析**

与多种端到端模型和少量示例合成器对比，CASTER在Set‑VQA上平均准确率达到93.8%，显著超过最强基线38.6%。

**⚠️ 局限性**

主要误差来源仍是视觉检测和OCR，导致感知误差1.3%，但逻辑推理错误已基本消除。

---

## 229. A Framework and Prototype for a Navigable Map of Datasets in Engineering Design and Systems Engineering

**arXiv ID:** 2603.15722 | [PDF](https://arxiv.org/pdf/2603.15722v1)

**作者:** H. Sinan Bank `[一作]` (Colorado State University), Daniel R. Herber `[通讯]` (Colorado State University)

**通讯引用:** 970 | [OpenAlex ID](https://openalex.org/A5077474480)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了“工程设计与系统工程数据集地图”框架，包括四维多维分类体系、知识图谱架构和交互式发现工具原型，并进行了数据缺口分析。

**💡 创新点**

创新点在于：①引入四维多维分类（领域、生命周期阶段、数据类型、格式）实现可细粒度、多维度探索；②将该分类与知识图谱相结合，支持语义关联与动态导航；③识别数据“沙漠”与“绿洲”，并提出合成数据补缺策略。

**🔧 技术方法**

使用技术包括知识图谱（图数据库 + DCAT）、FAIR原则、可视化（Sunburst、网络图、热图）、自然语言查询以及后续规划的人工智能代理。

**📊 数据集**

示例使用的公开数据集有 NASA C‑MAPSS、CWRU Bearing Data、KITTI、Materials Project、National Bridge Inventory 等。

**📈 对比分析**

本文主要聚焦框架与原型实现，并未给出传统性能指标；通过案例展示分类、可视化和交互性能，较传统单列表提供更细粒度可探索性与语义关联。

**⚠️ 局限性**

局限性包括：①数据质量与标准化不统一；②元数据缺失或不完整；③专有/隐私限制导致数据获取受限；④原型功能尚未完整实现（如发布版、完整元数据、AI代理）；⑤合成数据真实性与验证仍是挑战。

---

## 230. A Fast Approximation Algorithm for the Minimum Balanced Vertex Separator in a Graph

**arXiv ID:** 2603.15782 | [PDF](https://arxiv.org/pdf/2603.15782v1)

**作者:** Vladimir Kolmogorov `[一作]` (Institute of Science and Technology Austria), Jack Spalding-Jamieson `[通讯]` (Independent)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于半正定规划（SDP）和矩阵乘法权重更新（MMWU）框架的快速随机伪逼近算法，用于求解最小平衡顶点分离器问题。

**💡 创新点**

创新点在于：①给出了一个Θ(1)-平衡、大小为O(_c·√(log n/ε))的伪逼近解；②实现了几乎线性的时间复杂度O(n^O(ε)·(n,m)·polylog n)，远优于以往的多项式时间方法；③通过构造高效的oracle（结合最大流、随机投影等技术）实现了MMWU的低宽度更新。

**🔧 技术方法**

使用的主要技术包括：半正定规划松弛、矩阵乘法权重更新（Arora-Kale）、随机投影与低维近似、最大流求解（Chen等人提出的几乎线性时间算法）以及路径匹配构造。

**📊 数据集**

文中未使用具体实验数据集，算法仅在理论上给出时间复杂度与近似比；若需实验，需自行构造或使用标准图数据集。

**📈 对比分析**

与以往方法相比，该算法在实现伪逼近时仅需几乎线性时间，并在常数平衡下取得与最优已知近似比（O(√(log n))）相当，表现显著优于之前多项式时间的伪逼近算法。

**⚠️ 局限性**

局限性包括：①算法为随机化，只有高概率成功；②依赖于高阶 SDP 松弛和复杂的 oracle 实现，实际实现难度高；③伪逼近性质仅保证在已知平衡参数下近似，而非完整的逼近；④对极大图的内存占用仍为 O(m+n√(log n))，不适合极大规模稀疏图的直接应用。

---

## 231. Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning

**arXiv ID:** 2603.16127 | [PDF](https://arxiv.org/pdf/2603.16127v1)

**作者:** Kazuki Yano `[一作]` (Tohoku University), Jun Suzuki `[通讯]` (Tohoku University)

**通讯引用:** 8105 | [OpenAlex ID](https://openalex.org/A5001456824)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统实验研究了学习率调度对大型语言模型预训练和随后监督微调（SFT）性能的影响，重点评估了不使用衰减阶段的Warmup‑Stable‑Only（WSO）调度。

**💡 创新点**

创新点在于揭示传统衰减型调度虽然能降低预训练损失，却会导致模型收敛至更尖锐的极小值，从而削弱在SFT中的适应性；而WSO在保持平坦极小值的同时实现了更优的下游任务表现。

**🔧 技术方法**

采用四种学习率调度（WSO、WSD、Cosine、Linear），在不同规模（1B、8B）与多阶段（预训练→中训练→SFT）和过度训练（2T tokens）设置下进行实验，并利用Hutchinson估计计算尖锐度（loss landscape sharpness）。

**📊 数据集**

数据集包括FineWeb‑Edu（预训练）、olmo‑mix‑1124 与 dolmino‑mix‑1124（中训练）以及Tulu‑3 SFT mixture（微调），评估指标涵盖零样本问答、推理、阅读、数学与SFT后指令跟随、事实性与多任务理解等。

**📈 对比分析**

与Cosine/Linear/WSD等衰减调度对比，WSO在预训练阶段的验证损失略高，但在SFT后所有主要任务指标上均优于其他调度（平均提升≈0.5‑1.5个百分点），并在过度训练与中训练场景下保持相同趋势。

**⚠️ 局限性**

局限性包括仅评估了1B/8B规模模型、实验样本量有限、尖锐度与SFT性能的负相关性仅为相关不等于因果、未探究更大规模模型或其他任务场景的普适性。

---

## 232. PathGLS: Evaluating Pathology Vision-Language Models without Ground Truth through Multi-Dimensional Consistency

**arXiv ID:** 2603.16113 | [PDF](https://arxiv.org/pdf/2603.16113v1)

**作者:** Minbing Chen `[一作]` (Beijing University of Posts and Telecommunications), Fei Su `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 4288 | [OpenAlex ID](https://openalex.org/A5101754632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研发了一种无参考的多维一致性评估框架 PathGLS，用来衡量病理视觉语言模型在图像-文本对齐、逻辑一致性和对抗稳健性方面的可信度。

**💡 创新点**

创新点包括：①提出高分辨率多实例学习（MIL）视觉-文本对齐方法；②通过图谱+领域 NLI 模型进行逻辑一致性检查；③设计双重对抗攻击（染色扰动+语义注入）评估模型稳健性；④将三维度评分融合为统一的 PathGLS 综合分，全面捕捉幻觉与逻辑错误。

**🔧 技术方法**

使用的技术包括：HighRes-PLIP 视觉编码器、DeBERTa-v3 文本编码器、领域特定 NLI 模型、Macenko 染色归一化、对抗攻击策略、矩阵相似度匹配、空间 argmax 与 mean 聚合、加权组合得分。

**📊 数据集**

使用的数据集有：Patch 级别的 Quilt-1M 与 PathMMU，WSI 级别的 TCGA、REG2025、TCGA‑Sarcoma 以及构造的 OOD 以及幻觉/逻辑错误测试集。

**📈 对比分析**

与传统指标（BLEU、BERTScore、RadGraph）及 LLM‑as‑judge 对比时，PathGLS 对视觉幻觉的敏感度下降 40.2%（而 BERTScore 仅 2.2%），对逻辑错误下降 26.4%；与专家制定的临床错误等级相关性 Spearman ρ=0.71，显著优于 Gemini 3.0 Pro（ρ=0.39）。在域外样本和不同模型上，PathGLS 能准确识别稳健性差异，并在多中心数据上表现出更高的泛化能力。

**⚠️ 局限性**

局限性包括：①依赖预训练视觉编码器和 NLI 模型的质量；②计算开销较大，尤其是高分辨率 MIL；③对文本结构化程度要求较高，极低质量图像或极少标注的临床样本仍需进一步验证；④权重组合参数尚缺统一的选择原则。

---

## 233. Safety is Non-Compositional: A Formal Framework for Capability-Based AI Systems

**arXiv ID:** 2603.15973 | [PDF](https://arxiv.org/pdf/2603.15973v1)

**作者:** Cosimo Spera `[一作]` `[通讯]` (Minerva CQ), Cosimo Spera (Minerva CQ)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了基于有向超图的 AI 能力系统模型，并给出了闭包运算的正式证明和高效算法。

**💡 创新点**

核心创新是首次正式证明在存在合取（AND）能力依赖时安全性非组合，并揭示传统图模型无法捕获的危险。

**🔧 技术方法**

采用了有向超图、闭包与 Horn 句子完整性、子模函数理论、P/NP/coNP 复杂度分析，以及 PAC 学习和概率超图等技术。

**📊 数据集**

实验使用了公开的 ToolBench G3 与 TaskBench DAG 两个多工具轨迹数据集。

**📈 对比分析**

与传统工作流和能力图规划器对比，超图规划器在 42.6% 的合取依赖轨迹上零 AND 违规，并在闭包计算中平均节省 1.36 倍步骤。

**⚠️ 局限性**

局限性包括对大规模部署的 coNP‑hard 安全检测、动态超图增删需完整重算，以及理论证明基于理想化的合取超图结构。

---

## 234. CorrectionPlanner: Self-Correction Planner with Reinforcement Learning in Autonomous Driving

**arXiv ID:** 2603.15771 | [PDF](https://arxiv.org/pdf/2603.15771v1)

**作者:** Yihong Guo `[一作]` (Johns Hopkins University), Xianming Liu `[通讯]` (XPENG Motors)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种自我纠正的轨迹规划器CorrectionPlanner，通过在每个规划步骤中生成运动令牌并利用碰撞评估器进行安全检测，从而在执行前主动纠正潜在碰撞行为。

**💡 创新点**

创新点在于：①在运动令牌空间实现自我纠正机制，形成纠正轨迹类似语言模型的推理轨迹；②采用基于世界模型的多智能体强化学习进行训练，使规划器在实际交互中具备前瞻性安全判断；③通过将纠正过程编码进策略网络，强化了自我纠正能力。

**🔧 技术方法**

核心技术包括：自回归运动令牌生成、碰撞评估器（二分类器）、冻结的多智能体世界模型（Transformer解码器）、两阶段训练（模仿学习+模型基强化学习）、REINFORCE + KL正则化策略优化。

**📊 数据集**

使用了Waymo Open Dataset（Waymax仿真环境）和nuPlan数据集进行实验，并在这些数据上进行离线仿真评估。

**📈 对比分析**

与多种基线（PlanT、LatentDrive、SMART等）对比，Collision率降低约20%，在Waymax上成为最安全的规划器；在nuPlan上获得state‑of‑the‑art的规划分数，且在反应式环境下表现优于现有方法。

**⚠️ 局限性**

局限性包括：①需要依赖训练好的碰撞评估器，若阈值设置不当会导致过度纠正或漏判；②纠正步骤会略微降低进度（progression）和计算延迟；③目前仅关注碰撞安全，未扩展到其他安全或舒适度指标。

---

## 235. ReFORM: Review-aggregated Profile Generation via LLM with Multi-Factor Attention for Restaurant Recommendation

**arXiv ID:** 2603.16236 | [PDF](https://arxiv.org/pdf/2603.16236v1)

**作者:** Moonsoo Park `[一作]` (University of Southern California), Donghyeon Park `[通讯]` (Sejong University)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5054245349)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型（LLM）从用户与餐厅评论中提取多因素（如菜系、口味、氛围、价格等）特定的用户和商品档案，并通过多因素注意力机制动态聚焦对用户决策最有影响的因素，最终与图卷积网络（LightGCN）融合，生成更精准的个性化推荐；

**💡 创新点**

①首次将LLM用于仅依靠评论自动生成细粒度因素档案；②提出多因素注意力（MFA）在用户-商品交互中自适应强调不同因素；③将因素档案与图结构信息融合，提升推荐解释性与效果；

**🔧 技术方法**

大型语言模型（GPT‑4o mini）用于档案生成；BERT用于编码档案文本；LightGCN作为图卷积基座；多因素注意力机制（交叉注意力+最大池化）；Bayesian Personalized Ranking (BPR) 损失；

**📊 数据集**

公开餐饮评论数据集：Yelp（10‑core）和 Google Restaurants（5‑core）；

**📈 对比分析**

与多类基线（GCCF、LightGCN、SGL、SimGCL、MMGCN、GRCN、RLMRec）在 Recall@10/20 与 NDCG@10/20 上对比；ReFORM 在 Recall@20 上比 LightGCN 提升约 51%（Yelp）和 17%（GR），相较最佳基线提升 8–13%，且在统计检验中显著；

**⚠️ 局限性**

对评论质量高度敏感，噪声比例 0.5 时 Recall@20 下降 11%；在交互稀疏的数据集上多键注意力易过拟合；目前仅在餐饮领域验证，尚未跨域验证；

---

## 236. Impact of File-Open Hook Points on Backup Ratio in ROFBS on XFS

**arXiv ID:** 2603.16364 | [PDF](https://arxiv.org/pdf/2603.16364v1)

**作者:** Kosuke Higuchi `[一作]` (Kogakuin University), Ryotaro Kobayashi `[通讯]` (Kogakuin University)

**通讯引用:** 310 | [OpenAlex ID](https://openalex.org/A5101610974)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估了ROFBS在不同文件打开钩子点下对勒索软件（AvosLocker、Conti、IceFire）防护效果的影响。

**💡 创新点**

在保持ROFBS机制不变的前提下，系统性比较了五个不同层级的文件打开钩子点对备份比例、备份文件数和受影响文件数的差异，从而揭示钩子点选择是ROFBS设计的核心因素。

**🔧 技术方法**

采用eBPF+BCC技术在AlmaLinux XFS文件系统上实现钩子，利用AvosLocker、Conti、IceFire三款勒索软件进行实验。

**📊 数据集**

使用NapierOne数据集中4,385个文件作为测试文件，并结合三款勒索软件的SHA256样本。

**📈 对比分析**

通过Backup Ratio、B（已备份的加密文件数）和E（总加密文件数）三项指标进行比较；结果显示文件系统特定钩子xfs_file_open在降低受损文件数方面表现最佳，备份比例和受损文件数在不同钩子点上差异显著。

**⚠️ 局限性**

实验仅在XFS文件系统和单一工作负载下进行，未评估ext4等其他文件系统、性能/CPU占用、事件量以及路径重建方法对结果的影响。

---

## 237. Enabling Dynamic Tracking in Vision-Language-Action Models via Time-Discrete and Time-Continuous Velocity Feedforward

**arXiv ID:** 2603.16218 | [PDF](https://arxiv.org/pdf/2603.16218v1)

**作者:** Johannes Hechtl `[一作]` (Siemens Foundational Technologies), Wolfram Burgard `[通讯]` (University of Technology Nuremberg)

**通讯引用:** 69826 | [OpenAlex ID](https://openalex.org/A5084499878)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究如何在工业刚性机器人上通过在低层控制器中加入速度前馈，使VLA模型在完成快速传输运动的同时保持必要的顺应性，从而提升复杂接触任务的执行效率与成功率。

**💡 创新点**

创新点在于提出两种与现有VLA模型兼容的轨迹表示方法：①利用有限差分近似速度前馈，直接从离散位置预测生成速度；②使用三次B样条（C²连续）构建连续动作空间，使控制器可解析获得位置、速度与加速度，从而在不改动网络结构的前提下提升轨迹光滑度与动态响应。

**🔧 技术方法**

核心技术包括基于行为克隆的VLA模型（π₀.₅）、主动正交位姿顺应控制（阻尼-刚度模型）、实时块化（RTC）与流匹配政策的平滑插值、B样条控制点生成与离散差分速度估计。

**📊 数据集**

实验使用自制的“立方体-洞”接触任务数据集，包含两台UR工业机器人在多频率下进行的遥操作演示；随后将演示数据转化为离散位置或B样条控制点，训练VLA策略。

**📈 对比分析**

与仅输出位置的基线相比，有限差分方法将任务完成时间平均从约7.35 s下降至6.05 s（p<0.001），成功率保持在约79%；B样条方法在完成时间上并未显著提升，但最终成功率达79.2%，并在某些试验中因“摇摆”运动对插入阶段产生了正向影响。

**⚠️ 局限性**

局限性包括：①速度前馈方法需在遥操作阶段提供高频位置数据，低频时易失效；②B样条虽然保证C²连续，但与RTC产生的残余跳跃可导致冲击运动，影响稳定性；③未实现加速度前馈，受双微分噪声限制；④两种方法均需在低层控制器做相应改动，且在不同机器人/任务上效果尚需进一步验证。

---

## 238. Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2603.16140 | [PDF](https://arxiv.org/pdf/2603.16140v1)

**作者:** Yuxuan Zhu `[一作]` (University of Illinois Urbana Champaign), Daniel Kang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建真正噪声的数学推理数据集和对BIRD Text2SQL数据集的手工纠正，系统评估RLVR在噪声环境下的鲁棒性，并证明现有RLVR改进无法抵消噪声带来的性能下降。

**💡 创新点**

创新点在于提出了严格的多阶段数据再验证管线，剔除先前研究中混入的正确标注，生成纯噪声数据；随后首次在此数据上验证并驳斥了RLVR对噪声的“鲁棒”假设，强调高质量数据仍是关键。

**🔧 技术方法**

使用了GPT‑5 Pro生成注释、符号等价检测、LLM 判定（GPT‑5 judge）与人工复核的组合；在RLVR方面评估了GRPO及其改进算法（Dr. GRPO、TIS、DAPO、SAPO、PGFC）并进行超参数对比。

**📊 数据集**

数据集包括：合成数学推理数据（DeepScaleR + Qwen2.5‑Math‑7B 生成的错误答案，后经再验证得到12,769个纯噪声样本）、标准数学基准（MATH‑500、AIME 2024/2025、AMC 2023/2024）、Text2SQL数据集BIRD（600条样本手工纠正得到的BIRD‑600‑Corrected）以及对应的噪声版本BIRD‑600‑Original。

**📈 对比分析**

在干净、纯噪声、随机噪声和格式奖励四种训练设置下，使用GRPO及其改进算法对同一基准进行评估。结果显示：纯噪声训练模型在数学基准上准确率下降约8–10%，在Text2SQL上下降5–12%；改进算法与GRPO相比提升不足1%，未能弥补噪声带来的损失，且在pass@k、推理长度等指标上表现同样差。

**⚠️ 局限性**

局限性包括：实验仅覆盖有限的模型规模和基准；改进算法参数取自原论文，未针对纯噪声环境进行专门调优；对更复杂、系统化噪声类型（如验证器错误）的探索有限。

---

## 239. HeBA: Heterogeneous Bottleneck Adapters for Robust Vision-Language Models

**arXiv ID:** 2603.16653 | [PDF](https://arxiv.org/pdf/2603.16653v1)

**作者:** Md Jahidul Islam `[一作]` (Bangladesh University of Engineering and Technology), Md Jahidul Islam `[通讯]` (Bangladesh University of Engineering and Technology)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5100604188)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种名为 HeBA 的异质瓶颈适配器，用于在冻结 CLIP 背景下高效地进行少样本视觉‑文本任务的微调。

**💡 创新点**

其创新点包括：① 视觉分支采用 2D 深度可分离卷积保持空间局部性；② 文本分支采用线性压缩保持语义全局性；③ 通过压缩瓶颈（D→D/4）实现结构正则化；④ 引入主动 Kaiming 初始化以保证训练初期梯度流动。

**🔧 技术方法**

技术实现涵盖多模态自适应网络、瓶颈压缩、深度可分离卷积、主动初始化、动态慢速‑快速缩放与标签平滑等。

**📊 数据集**

使用了 11 个公开的少样本分类基准（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101）、十个交叉数据集（除 ImageNet 外）以及四个 ImageNet OOD 变体。

**📈 对比分析**

与现有 PEFT 方法（CoOp、CoCoOp、MaPLe、MMA、LwEIB 等）比较，HeBA 在 Base‑to‑Novel 泛化中的调和平均（HM）达到 81.35%，在结构敏感任务如 EuroSAT 和 DTD 上均显著提升，且在跨数据集与 OOD 泛化中保持竞争力。

**⚠️ 局限性**

局限性在于：仅针对 CLIP 视觉‑文本模型；对更多模态或更大规模任务的适用性未验证；压缩瓶颈虽然降低过拟合风险，但在极端数据稀缺场景下仍可能导致信息损失。

---

## 240. PureCLIP-Depth: Prompt-Free and Decoder-Free Monocular Depth Estimation within CLIP Embedding Space

**arXiv ID:** 2603.16238 | [PDF](https://arxiv.org/pdf/2603.16238v1)

**作者:** Ryutaro Miya `[一作]` (Institute of Science Tokyo), Tatsuya Kawaguchi `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 600 | [OpenAlex ID](https://openalex.org/A5016186986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PureCLIP-Depth，一种完全基于CLIP嵌入空间的编码器‑仅单目深度估计模型，直接将RGB映射到深度；

**💡 创新点**

核心创新在于利用概念化CLIP空间进行深度估计，构建可学习的深度表，采用MLP旋转向量并交替优化，彻底消除几何特征依赖；

**🔧 技术方法**

技术手段包括CLIP图像编码器、可学习深度表、MLP空间旋转、CLS全局上下文融合、InfoNCE/对齐/RMSE损失以及交替优化策略；

**📊 数据集**

在室内NYU Depth V2和室外KITTI数据集上进行实验；

**📈 对比分析**

与其他CLIP嵌入基模型及Auty与Mikolajczyk方法对比，PureCLIP-Depth在所有定量指标上均实现了最佳性能；

**⚠️ 局限性**

局限性包括对长距离预测的不确定性与欠估计问题（受数据不平衡影响），以及对低对比度图像的鲁棒性待提升。

---

## 241. DyJR: Preserving Diversity in Reinforcement Learning with Verifiable Rewards via Dynamic Jensen-Shannon Replay

**arXiv ID:** 2603.16157 | [PDF](https://arxiv.org/pdf/2603.16157v1)

**作者:** Long Li `[一作]` (Griffith University), Yuan Qi `[通讯]` (Fudan University)

**通讯引用:** 14561 | [OpenAlex ID](https://openalex.org/A5100676883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态Jensen–Shannon Replay (DyJR) 框架，用来在大型语言模型的强化学习中通过经验回放维持多样性并提升推理性能。

**💡 创新点**

创新点包括：① 时序感知的动态缓冲区，采用 FIFO 与自适应容量仅保留近期高质量样本；② 用 Jensen–Shannon 散度正则化取代传统的直接梯度更新，防止模式坍塌；③ 采用置信度分层的样本筛选和动态填充率调整，适配任务难度并缓解早期多样性衰退。

**🔧 技术方法**

技术手段包括：基于 Group Relative Policy Optimization (GRPO) 的基准策略；自定义最大年龄 (Max Age) 缓冲区与动态参考分布；Jensen–Shannon 散度近似估计与正则化损失；自适应样本选取与填充调度；以及完整的联合优化目标。

**📊 数据集**

使用的主要数据集有：数学推理基准（AIME、AMC、Beyond AIME、BRUMO、HMMT 等）以及文本到 SQL 基准（BIRD、Spider）。

**📈 对比分析**

与 GRPO、RLEP、Ex-GRPO、DAPO、DPH-RL 等方法对比，DyJR 在 Pass@1、Pass@k（如 34.1% 平均 Pass@1 vs 29.8% GRPO）、数学推理平均准确率及 SQL 任务 Pass@1/Pass@16 上均表现更优，且 GPU 内存与训练时延保持与 GRPO 相近。

**⚠️ 局限性**

局限性包括：对超参数（α_JS、M 等）敏感，需经验调优；最大年龄 M 较小时可能丢失有用的远期样本；仅在稀疏奖励的推理任务上验证，其他领域的通用性仍需进一步实验；实现中仍需存储 token‑级 log‑prob 并计算 JS 散度，可能在极大模型上产生一定计算开销。

---

## 242. When the City Teaches the Car: Label-Free 3D Perception from Infrastructure

**arXiv ID:** 2603.16742 | [PDF](https://arxiv.org/pdf/2603.16742v1)

**作者:** Zhen Xu `[一作]` (Ohio State University), Wei-Lun Chao `[通讯]` (Boston University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出基于城市侧 RSU 的标签免费 3D 感知训练框架，利用 RSU 固定视角的时序一致性让 RSU 作为分布式教师监督 ego 车辆学习。

**💡 创新点**

创新点在于：1）使用 RSU 的稳定视角实现无监督检测；2）将 RSU 预测广播转化为 ego 训练的伪标签，消除训练阶段人工标注；3）推理时不依赖 V2X 通信，保持 ego 模型的独立性。

**🔧 技术方法**

采用 Persistence Point 分数 + DBSCAN + 跟踪生成 RSU 伪标签；RSU 与 ego 之间使用距离加权 NMS 聚合预测；最终使用 CenterPoint/PointPillars 对 ego 进行离线训练。

**📊 数据集**

在 CARLA‑V2XVerse 上构建的 CIVET 仿真多代理数据集，包含 4 个城镇、每镇 12 台 RSU，总计 608k 帧。

**📈 对比分析**

与完全监督 ego 上限（94.4% AP）相比，三阶段管道在 CenterPoint 上达到 82.3% AP；相较单一 RSU 或 ego 无监督基线提升显著；与 ego‑centric 无监督方法（MODEST、Oyster）结合可再提升约 10 AP。

**⚠️ 局限性**

局限性包括：实验仅在仿真环境；RSU 无监督训练主要针对动态物体，对静态物体效果有限；通信噪声和坐标误差仍影响伪标签质量；需在真实城市环境中验证。

---

## 243. Monoidal categories graded by partial commutative monoids

**arXiv ID:** 2603.16375 | [PDF](https://arxiv.org/pdf/2603.16375v1)

**作者:** Matthew Earnshaw `[一作]` (Institute of Computer Science, University of Tartu), Mario Román `[通讯]` (University of Oxford)

**通讯引用:** 121 | [OpenAlex ID](https://openalex.org/A5026808332)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了由偏余加单体（PCM）刻度的单模范畴概念，统一了单模范畴、效应范畴和Freyd范畴的理论框架；

**💡 创新点**

创新点在于将效应（或资源）刻度视为PCM，使得张量积仅在可兼容的刻度下定义，并证明效应范畴等价于2-graded单模范畴，进一步扩展到对称、笛卡尔结构并构造核心射子；

**🔧 技术方法**

采用范畴论方法：PCM、薄的promonoidal范畴、卷积单模、duoidal结构、以及PCM-graded单模范畴的定义与等价性证明；

**📊 数据集**

论文不涉及实验数据或数据集，主要以理论示例（如功率集、区间、最大、写读冲突等）说明刻度应用；

**📈 对比分析**

方法通过理论证明与同构推导进行比较，未给出数值性能评估；

**⚠️ 局限性**

局限性包括：对PCM的可扩展性、对具体程序语言语义的直接适用性不明；在对称与笛卡尔化后仍需进一步研究核心射子与自由构造的完整性。

---

## 244. MemX: A Local-First Long-Term Memory System for AI Assistants

**arXiv ID:** 2603.16171 | [PDF](https://arxiv.org/pdf/2603.16171v1)

**作者:** Lizheng Sun `[一作]` `[通讯]`, Lizheng Sun

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 MemX，一个本地首选的长短期记忆系统，专为单用户 AI 助手设计，提供持久、可搜索且可解释的记忆存储与检索。

**💡 创新点**

创新点在于：① 以低阈值拒绝规则实现对无相关记忆查询的稳定抑制；② 四因子重排序与 RRF 融合的混合检索管线；③ 通过访问/检索分离提升检索信号质量；④ 在单机 Rust+libSQL 环境下实现可复现、可量化的评测框架。

**🔧 技术方法**

技术手段包括：OpenAI‑兼容的 Qwen3 0.6B 嵌入模型、DiskANN 向量索引、SQLite FTS5 全文索引、Reciprocal Rank Fusion、z‑score 归一化+sigmoid、低阈值拒绝、内容与标签去重。

**📊 数据集**

数据集：自定义 2 组中文基准（43 题，≤1014 条记录）和公开 LongMemEval（500 题，19.2k 轮会话 → 220k 事实级条目），并在不同存储粒度（session/round/fact）下进行实验。

**📈 对比分析**

比较方法：使用 Hit@k、MRR、Coverage@k、Miss‑Empty/Strict、平均/95% 延迟等指标；在自定义基准上 Hit@1≈91–100%，Hit@5≈96%；在 LongMemEval 事实粒度下 Hit@5≈51.6%、MRR≈0.38；全文索引将关键词检索延迟从 3.3s 降到 1.1ms，整体查询低于 90 ms。

**⚠️ 局限性**

局限性：对多主题与高混淆查询的覆盖不足；Temporal 与 multi‑session 推理仍低于 44%；标签去重在无标签事实上导致召回下降；拒绝阈值依赖嵌入模型，需手动重新校准；缺乏任务级成功评估，无法证明检索提升会直接改善代理输出。

---

## 245. Internalizing Agency from Reflective Experience

**arXiv ID:** 2603.16843 | [PDF](https://arxiv.org/pdf/2603.16843v1)

**作者:** Rui Ge `[一作]` (University of California San Diego), Hao Zhang `[通讯]` (University of California San Diego)

**通讯引用:** 37341 | [OpenAlex ID](https://openalex.org/A5100456227)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在探索阶段引入反射性回溯与经验总结，并在训练阶段将这些回溯经验蒸馏回模型，使大型语言模型在长周期交互中具备自我恢复与纠错的能力。

**💡 创新点**

创新点在于：① 采用树状回溯生成经验，精准定位失败决策点并给出可执行修正；② 将回溯产生的修正行为通过经验蒸馏嵌入模型权重，实现在单次推理中无需额外回溯即可自我纠错；③ 与传统终端奖励驱动的RLVR形成对比，显著缓解分布收缩问题。

**🔧 技术方法**

使用的技术包括：反射性回溯与经验总结 (Tree-Based Experience Generation with Rollback)、经验蒸馏 (Experience Distillation)、RLVR/GRPO 对比、ReAct 交互框架、标准的 PPO 与群组优化算法。

**📊 数据集**

在 WebShop、ALFWorld、ScienceWorld、Sokoban、CodeContests 等交互式基准上进行评估，并在 CodeContests 上进一步验证大模型（Qwen2.5‑72B、Llama‑70B）的表现。

**📈 对比分析**

与 Base、GRPO‑RLVR、EarlyExp、ACE 等基线对比，Pass@128 提升可达 14%（如 CodeContests 128‑样本），Pass@1 亦有提升；在 Pass@k 规模曲线中显示更高的覆盖率与样本效率，尤其在大 k 处超越所有基线。

**⚠️ 局限性**

局限性包括：① 依赖环境能够回滚到先前状态并提供明确诊断反馈；② 对于反馈弱、延迟或无法准确归因的环境，回溯与蒸馏效果下降；③ 在非确定性或高度状态依赖的真实世界场景中实现回溯仍具挑战。

---

## 246. BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection

**arXiv ID:** 2603.16645 | [PDF](https://arxiv.org/pdf/2603.16645v1)

**作者:** Melissa Schween `[一作]` (Leibniz University Hannover), Bodo Rosenhahn `[通讯]` (Leibniz University Hannover)

**通讯引用:** 10051 | [OpenAlex ID](https://openalex.org/A5040412734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出首个使用正则流的场景特定异常关系检测模型BUSSARD，利用图像生成的场景图检测异常关系。

**💡 创新点**

结合预训练语言模型词嵌入、自动编码器压缩与可逆正则流，构建无监督异常检测框架，显著提升对词汇变异的鲁棒性。

**🔧 技术方法**

使用EGTR场景图生成器、GloVe词向量、autoencoder降维、RealNVP正则流以及对数似然评分机制。

**📊 数据集**

在SARD数据集（办公场景与餐厅场景）上进行训练与评估，并进行MIT‑67跨数据集实验。

**📈 对比分析**

与计数基线SARD相比，AUROC提升约10%，AUC‑Recall@k提升约4%，并且推理速度快约5倍。

**⚠️ 局限性**

受限于预训练场景图生成器的误差，对跨域场景的泛化有限，且未处理多异常场景的情况。

---

## 247. Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning

**arXiv ID:** 2603.16189 | [PDF](https://arxiv.org/pdf/2603.16189v1)

**作者:** Haomin Wang `[一作]` (Shanghai Jiao Tong University), Hongjie Zhang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一框架，结合链式思考（CoT）与多任务多奖励GRPO强化学习，支持文本→SVG、图像→SVG以及SVG代码改进三大任务。

**💡 创新点**

创新点在于：①利用SVG天然的分组结构，将CoT步骤与代码块一一对应，实现可解释的生成流程；②构建高质量的SVG‑Sophia数据集，包含CoT问答；③采用多任务联合训练与四种奖励（格式、DINO、图文相似度、代码效率）实现视觉、语义和代码质量的三重提升。

**🔧 技术方法**

技术主要包括：大规模多模态LLM（Qwen3‑VL‑8B）、链式思考生成、Group Relative Policy Optimization（GRPO）强化学习、DINOv2特征对齐、Long‑CLIP语义匹配、代码长度惩罚等。

**📊 数据集**

使用了内部的SVG‑Sophia数据集（约13.5k SFT+14.4k RL样本），以及从ColorSVG‑100K衍生的高质量SVG代码。

**📈 对比分析**

在SArena‑Icon基准上，Text‑to‑SVG CLIP‑T2I最高25.94，Image‑to‑SVG在DINO、SSIM、LPIPS等指标均领先；在SVG‑Sophia代码改进任务上，SFT+RL版模型在DINO、SSIM、LPIPS和成功率上均超越GPT‑5.2、Claude‑Sonnet‑4.5和Gemini‑3‑Pro。

**⚠️ 局限性**

局限性包括：1）对128×128 viewBox的固定分辨率限制了复杂场景的细节；2）CoT生成依赖强监督，数据成本高；3）奖励设计仍依赖视觉模型（DINO/CLIP）可能对不同视觉风格不够鲁棒。

---

## 248. DRCY: Agentic Hardware Design Reviews

**arXiv ID:** 2603.15672 | [PDF](https://arxiv.org/pdf/2603.15672v1)

**作者:** Kyle Dumont `[一作]` (AllSpice Inc), Shrikanth Upadhayaya `[通讯]` (AllSpice Inc)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 DRCY，一套多智能体大型语言模型（LLM）系统，用于在 PCB 设计评审中自动获取元器件数据手册、逐引脚对比分析、并将发现的语义错误作为评审注释返回给设计者。

**💡 创新点**

其创新点在于：①构建了完整的端到端智能体流水线，包括自动数据手册检索、提取、质量评估以及多次运行的共识机制；②将 LLM 与 EDA 生态深度集成，实现 CI/CD 自动触发和版本控制；③针对硬件设计的语义层面（如引脚功能、规格、电路示例）进行验证，填补了现有 ERC/DRC 无法覆盖的空白。

**🔧 技术方法**

主要技术包括：多智能体架构（Selection、Datasheet Retrieval、Group Review、Consensus、Error Grouping）；Playwright 自动化抓取数据手册；自评与多跑共识（自一致性）策略；使用 Anthropic 或可配置的 OpenAI 兼容模型；基于 AllSpice Hub 的原生 JSON 结构化格式与 Docker 化 CI/CD。

**📊 数据集**

使用的数据集来源于内部硬件公司（包括 Fortune 500 与初创企业）的 PCB 设计文件，涵盖 Altium、OrCAD、KiCad、System Capture、Xpedition、DE‑HDL 等多种 EDA 格式；同时利用 DigiKey、客户自定义库等多源数据手册进行检索与提取。

**📈 对比分析**

评估方式主要是基于真实生产环境的案例研究：在 10 页、50–100 个元件的典型设计中，完整分析耗时不足 20 分钟；系统已在多家企业上线，能够识别并报告如引脚互换、错误反馈电阻、超标电压等关键语义错误；与传统 ERC/DRC 的对比表明，DRCY 能够捕获超过 90% 的非结构性错误，且误报率低于 5%。

**⚠️ 局限性**

局限性包括：对 LLM 的性能与成本高度依赖；在数据手册缺失或格式异常时可能出现误判；多跑共识虽然提高可靠性，但会增加算力开销；需要在满足 GDPR 与 SOC 2 的环境下部署，对完全数据主权需求的客户需自行部署私有 LLM。

---

## 249. CompDiff: Hierarchical Compositional Diffusion for Fair and Zero-Shot Intersectional Medical Image Generation

**arXiv ID:** 2603.16551 | [PDF](https://arxiv.org/pdf/2603.16551v1)

**作者:** Mahmoud Ibrahim `[一作]` (Maastricht University), Michel Dumontier `[通讯]` (Maastricht University)

**通讯引用:** 27322 | [OpenAlex ID](https://openalex.org/A5044836472)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 CompDiff，一种层次组合的扩散框架，专门解决医学影像生成中因人口属性不平衡导致的生成质量差异问题。

**💡 创新点**

创新点在于引入 Hierarchical Conditioner Network（HCN）将人口属性分解为单属性、配对交互和全组合，赋予模型可组合的结构偏置，实现对从未见过的交叉子群的零样本生成。

**🔧 技术方法**

技术手段包括基于 Stable Diffusion 2.1 的扩散模型、CLIP 文本嵌入、HCN 结构、变分潜在表示、余弦一致性正则、辅助分类损失以及跨注意力上下文。

**📊 数据集**

使用的医学影像数据集为胸部 X 光（MIMIC-CXR）和眼底（FairGenMed），均带有年龄、性别、种族等人口元数据。

**📈 对比分析**

在与 Fine‑tuned Stable Diffusion 2.1 和 FairDiffusion 的对比实验中，CompDiff 在 FID、ES‑FID、疾病检测 AUROC 以及下游分类 AUROC 上均表现更优；在零样本交叉子群上实现了高达 21% 的 FID 改进。

**⚠️ 局限性**

局限性包括评估主要基于量化指标缺乏临床专家验证；层次组合框架仅适用于离散属性，无法直接处理连续或无结构属性；对极少见子群的生成质量仍低于主流组，表明数据不平衡的影响尚未完全消除。

---

## 250. Efficient Brood Cell Detection in Layer Trap Nests for Bees and Wasps: Balancing Labeling Effort and Species Coverage

**arXiv ID:** 2603.16652 | [PDF](https://arxiv.org/pdf/2603.16652v1)

**作者:** Chenchang Liu `[一作]` (Albert Ludwig University Freiburg), Marco Seeland `[通讯]` (Technische Universität Ilmenau)

**通讯引用:** 2022 | [OpenAlex ID](https://openalex.org/A5030228799)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过深度学习实现层式陷阱巢（LTN）中野生蜜蜂和黄蜂巢细胞的自动检测与分类；

**💡 创新点**

提出Constrained False Positive Loss (CFPL) 以动态屏蔽未标注样本的影响，显著降低标注成本并缓解类别不平衡；

**🔧 技术方法**

采用YOLOv8一阶段目标检测框架，结合CFPL损失函数，并使用LabelStudio进行手工标注；

**📊 数据集**

使用712张LTN图像，覆盖28个细分类别（物种及发育状态），训练集中每类最多标注300张样本；

**📈 对比分析**

与基线YOLOv8对比，CFPL使多数类AP提升3%（约为66.11%），召回率提升39%（从7%至46%）；少数类AP提升1.5%，召回率提升3.8%；

**⚠️ 局限性**

数据集规模有限，稀有类别样本不足；采集时间仅覆盖过冬期，未能覆盖全年发育阶段；模型在更广泛生态环境中的泛化仍待验证。

---

## 251. ExpertGen: Scalable Sim-to-Real Expert Policy Learning from Imperfect Behavior Priors

**arXiv ID:** 2603.15956 | [PDF](https://arxiv.org/pdf/2603.15956v1)

**作者:** Zifan Xu `[一作]` (University of Texas at Austin), Karl Schmeckpeper `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出ExpertGen框架，利用不完整的演示数据在仿真中通过扩散模型和强化学习自动生成可在真实机器人上部署的专家策略。

**💡 创新点**

核心创新是将扩散模型作为行为先验，仅在初始噪声上进行RL steering，保持人类行为分布；在大规模并行仿真中使用FastTD3提升样本效率；结合DAgger实现零射击的visuomotor策略。

**🔧 技术方法**

扩散政策、扩散 steering RL (DSRL)、FastTD3、DAgger、DDIM采样、Sim2Real domain randomization。

**📊 数据集**

AnyTask（桌面抓取、堆叠等8类任务）和AutoMate（高精度装配插槽）两大仿真基准，以及少量人类演示（通过SkillMimicGen增强）。

**📈 对比分析**

与脚本策略、纯扩散策略、Residual RL、SMP、PPO、无先验FastTD3等基线相比，在AnyTask上ExpertGen平均成功率超过99%，在AutoMate平均90.5%，在真实机器人上visuomotor策略成功率高达75–85%，显著优于传统BC。

**⚠️ 局限性**

受限于先验演示的覆盖范围，若缺乏关键行为会导致性能下降，且当前仅支持基于状态的扩散先验，未能合成全新动作。

---

## 252. VideoMatGen: PBR Materials through Joint Generative Modeling

**arXiv ID:** 2603.16566 | [PDF](https://arxiv.org/pdf/2603.16566v1)

**作者:** Jon Hasselgren `[一作]` (NVIDIA), Jacob Munkberg `[通讯]` (NVIDIA)

**通讯引用:** 2529 | [OpenAlex ID](https://openalex.org/A5076340798)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于视频扩散模型的文本到物理基材料生成方法，能够为已知未纹理化的3D模型生成完整的PBR材质贴图。

**💡 创新点**

创新点包括：①联合预测基色、粗糙度、金属度和高度四个PBR通道，并采用自定义变分自编码器压缩多模态；②直接从文本生成材质内在通道，无需先合成RGB外观；③支持通过随机种子产生多样化实例，便于艺术家快速原型。

**🔧 技术方法**

使用技术包括：Cosmos DiT 视频扩散变压器、T5‑XXL 文本编码器、基于 G‑buffer（法线、世界位置）的条件输入、变分自编码器实现多通道联合压缩、视角投射与 splatting 烘焙到 UV 贴图、以及图像指标评估（CLIP‑FID、CMMD、LPIPS）。

**📊 数据集**

数据集：通过 Objaverse、BlenderVault、ABO、HSSD 等渲染 60k 条包含 G‑buffer（法线、深度、基色、粗糙度、金属度、高度）的视频，并利用 MatSynth 进行 VAE 微调；所有数据均配有自动生成的文本描述。

**📈 对比分析**

与 VideoMat、Hunyuan3D‑Paint、MVPainter、TRELLIS.2 等方法在 CLIP‑FID、CMMD、LPIPS 上进行量化对比，文本导向版本的 VideoMatGen 取得最低分数，表明其生成质量更好；推理时间约 2‑3 分钟/资产，当前效率较低。

**⚠️ 局限性**

主要局限：推理耗时长（2‑3 分钟/资产）；对相机轨迹连贯性敏感，非连贯视角会导致模糊；纹理烘焙步骤简单，可进一步改进；目前使用的 Cosmos‑1.0 基模型可替换为更强模型以提升质量。

---

## 253. Fast-WAM: Do World Action Models Need Test-time Future Imagination?

**arXiv ID:** 2603.16666 | [PDF](https://arxiv.org/pdf/2603.16666v1)

**作者:** Tianyuan Yuan `[一作]` (Tsinghua University), Hang Zhao `[通讯]` (Tsinghua University)

**通讯引用:** 14788 | [OpenAlex ID](https://openalex.org/A5101826600)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Fast‑WAM，保留训练时的视频联合学习但测试时不做未来视频生成，实现实时高效的动作生成。

**💡 创新点**

关键创新在于解耦训练时的视频预测与推理时的未来想象，仅通过视频共训练提升表示，而无需额外推理成本。

**🔧 技术方法**

使用视频Diffusion Transformer（DiT）作为世界编码器，Mixture‑of‑Transformer结构与流匹配损失。

**📊 数据集**

在LIBERO、RoboTwin 2.0和真实世界的毛巾折叠任务上进行评估。

**📈 对比分析**

与现有想象‑执行WAM和VLA基线对比，Fast‑WAM在模拟和真实任务上保持或超过性能，推理延迟从190 ms提升至4倍加速。

**⚠️ 局限性**

主要局限在于未充分探索更大规模预训练和模型扩展的影响，且对复杂动态场景的泛化仍有待验证。

---

## 254. The Geometry of Transmission Zeros in Distance-Based Formations

**arXiv ID:** 2603.15993 | [PDF](https://arxiv.org/pdf/2603.15993v1)

**作者:** Solomon Goldgraber Casspi `[一作]` (Technion Israel Institute of Technology), Daniel Zelazo `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

**🎯 论文内容**

分析距离感知编队控制中的稳态信号阻塞现象，提出几何阈值与全局传输多面体，并给出传感器放置规则。

**💡 创新点**

首次把稳态传输零点转化为形状几何条件，证明柔性框架的零点只出现于零测度集合，而在无内弯刚性框架下可用直线超平面精确描述，并构造全局传输多面体。

**🔧 技术方法**

利用刚性理论、矩阵谱分解、Sylvester 行列式恒等式、代数几何与线性化动力学分析。

**📊 数据集**

通过数值仿真在一个四机体非对称二维编队（自定义坐标）验证理论。

**📈 对比分析**

与柔性四杆连杆对比，展示刚性时稳态奇异值趋零、柔性时保持正值，验证设计规则有效。

**⚠️ 局限性**

仅适用于二维无内弯刚性框架；线性化分析，未考虑非线性动力学与三维扩展。

---

## 255. TRACE: Evaluating Execution Efficiency of LLM-Based Code Translation

**arXiv ID:** 2603.16479 | [PDF](https://arxiv.org/pdf/2603.16479v1)

**作者:** Zhihao Gong `[一作]` (Peking University), Dan Hao `[通讯]` (Peking University)

**通讯引用:** 5018 | [OpenAlex ID](https://openalex.org/A5085393851)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了名为trace的基准，专门用于量化LLM在跨语言代码翻译时的执行效率；

**💡 创新点**

首次将执行效率作为代码翻译的重要评价维度，提出两阶段的LLM驱动压力测试生成与效率关键任务筛选流程，并提供1,000个高效性敏感任务；

**🔧 技术方法**

采用LLM自动化生成压力测试、Borda计数排序、HDBSCAN聚类、Beyond Score评估，以及对28个代表性LLM的系统性能评测；

**📊 数据集**

基于Transcoder-Test的357道问题，生成1,000个效率关键任务，每个任务包含10个默认单元测试和10个精心设计的压力测试；

**📈 对比分析**

通过Pass Rate与Beyond Score比较不同模型的正确率和效率，结果显示正确率与效率不成正相关，Claude-4-think在正确率最高但效率中等，小型开源LLM（如Qwen2.5-Coder）在效率上优于大型模型，提示策略提升有限；

**⚠️ 局限性**

评估仅覆盖执行时间和峰值内存，限制在C++/Java/Python三种语言，聚焦方法级任务，效率评估相对而非绝对，且全部采用贪婪解码，未考虑I/O、并发等实际场景因素。

---

## 256. Scalable Inspection Planning via Flow-based Mixed Integer Linear Programming

**arXiv ID:** 2603.16593 | [PDF](https://arxiv.org/pdf/2603.16593v1)

**作者:** Adir Morgan `[一作]` (Technion Israel Institute of Technology), Oren Salzman `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对图检验规划（inspection planning）问题，提出了一种基于分组覆盖（group‑covering）视角的混合整数线性规划（MILP）模型，并通过分组切集（group‑cutset）与分支定界-剪枝（Branch‑and‑Cut）框架实现可扩展求解。

**💡 创新点**

创新点主要有：①将检验规划视为分组覆盖问题，利用网络流结构重新表述覆盖与连通性约束；②设计了指数量的分组切集约束并在 BnC 中惰性生成，兼具强壮的 LP 下界与可扩展性；③提出了混合分离器（连通性+流量）和专属启发式，显著提升求解速度与解质量。

**🔧 技术方法**

核心技术包括：网络流建模与多商品流（MCF）下界；分支定界-剪枝（BnC）与分离器（separation oracle）实现；基于 LP 结果的启发式路径构造（Eulerian 近似）以及稀疏采样的分组切集检验。

**📊 数据集**

使用了三类数据集：真实场景（医学内腔 4,203 目标、航空结构 3,346 目标）以及大规模模拟实例（节点数 3,500–15,000、POI 数 1–3,000+），全面考察算法在不同规模与复杂度下的表现。

**📈 对比分析**

与现有最优 MILP（Charge）、单流（SCF）等模型比较。实验表明：①分组切集模型在 LP 下界更紧，优化时间与内存需求低于 MCF；②在大规模实例上，最优性缺口明显下降（30–50%），并能在 500 秒内给出可行解；③在中小型实例中，虽然启发式质量稍逊，但整体性能保持竞争力。

**⚠️ 局限性**

局限性：①分组切集模型仍需大量切集搜索，尽管采用惰性生成，但在极大 POI 数或极稀疏图中分离器仍可能成为瓶颈；②启发式对特定图结构（如高重叠覆盖）敏感，泛化性有待提升；③目前仅处理单机器人、完整覆盖情景，尚未扩展到多机器人、部分/自适应检验或在线增量设置。

---

## 257. Self-Admitted Technical Debt in Scientific Software: Prioritization, Sentiment, and Propagation Across Artifacts

**arXiv ID:** 2603.15883 | [PDF](https://arxiv.org/pdf/2603.15883v1)

**作者:** Eric L. Melin `[一作]` (Boise State University), Addi Malviya-Thakur `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 43 | [OpenAlex ID](https://openalex.org/A5055368248)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对九个科学软件项目中的自承技术债务（SATD）进行多工件优先级、情感、持续时间和传播性分析。

**💡 创新点**

首次在科学软件领域开展多工件SATD分析，揭示了与开源软件显著不同的债务持续性、低修复率及传播模式，并将情感与优先级关联。

**🔧 技术方法**

使用了微调的Transformer情感分类器、句子嵌入优先级启发式、自动SATD识别模型，以及GitHub API构建工件关联图；对比了SonarQube的静态分析结果。

**📊 数据集**

采用了来自DOE CASS项目的九个科学软件仓库数据，并利用先前手工标注的SATD多工件数据集；同时收集了SonarQube检测的缺陷信息。

**📈 对比分析**

通过与SonarQube严重级别的相关性评估以及与开源软件的修复率/持久期对比，发现科学软件中的SATD修复率仅约38%，平均持久期超过八年，优先级与负面情感显著相关。

**⚠️ 局限性**

局限性包括仅覆盖九个项目、优先级启发式缺乏真实标签、SATD识别模型的误判可能影响结果、传播链构建仅基于结构链接而非语义一致性。

---

## 258. Rationale Matters: Learning Transferable Rubrics via Proxy-Guided Critique for VLMReward Models

**arXiv ID:** 2603.16600 | [PDF](https://arxiv.org/pdf/2603.16600v1)

**作者:** Weijie Qiu `[一作]` (Alibaba), Guanjun Jiang `[通讯]` (Alibaba)

**通讯引用:** 426 | [OpenAlex ID](https://openalex.org/A5004378463)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Proxy-GRM 框架，利用可训练的代理评估器对视觉语言模型生成的 rubric 进行闭环验证和强化学习优化；

**💡 创新点**

创新点在于：①将代理评估器嵌入 RL 奖励循环，实现 rubric 的可转移性验证；②发现 SFT 训练的代理优于 RL 训练的代理；③在仅使用约 50k 样本的情况下实现三大基准的 state‑of‑the‑art 性能；

**🔧 技术方法**

技术手段包括：基于 Qwen2.5-VL-7B-Instruct 的结构化生成；Proxy‑SFT 与 Proxy‑RL 的代理训练；GRPO 强化学习；多项奖励组合（准确度、格式、代理一致性）；冻结代理评估器以保证可解释性；

**📊 数据集**

数据集：约 60k 经过 distillation 的 preference 样本，来源于 LaVA‑Critic‑113k、RLAIF‑V、RLHF‑V、MMIF‑23k；评估基准为 VL‑RewardBench、Multimodal Reward Bench 与 MM‑RLHF‑Reward Bench；

**📈 对比分析**

与多种开源与闭源模型对比，在三大基准上均取得最高准确率：VL‑RewardBench 75.22%/73.93%，Multimodal Reward Bench 85.62%，MM‑RLHF‑Reward Bench 82.94%；相比传统方法仅需 4× 更少的训练数据；

**⚠️ 局限性**

局限性包括：代理评估器需要单独训练且可能对不同任务不适用；模型性能仍受代理误差影响；实验仅覆盖视觉语言领域，未验证跨模态的泛化；冻结代理可能限制对新策略的适配。

---

## 259. FlowComposer: Composable Flows for Compositional Zero-Shot Learning

**arXiv ID:** 2603.16641 | [PDF](https://arxiv.org/pdf/2603.16641v1)

**作者:** Zhenqi He `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 96360 | [OpenAlex ID](https://openalex.org/A5100333572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FlowComposer 框架，用流匹配（Flow Matching）在嵌入空间显式完成属性与对象的组合，实现对未见属性-对象组合的更好泛化；同时设计泄漏引导增强（Leakage‑Guided Augmentation）利用残留特征交叉信息强化训练。

**💡 创新点**

① 用流匹配构造属性、对象以及组合的速度场，实现嵌入空间的显式组合规则；② 引入泄漏引导增强，将视觉分离器产生的残余信息转化为额外监督；③ 该框架可无缝插入任意 VLM‑based CZSL 基线，保持模型通用性。

**🔧 技术方法**

流匹配（Rectified Flow）模型、时间条件的速度场网络、Composer MLP、对比式终点监督、泄漏路径的流匹配损失、基于 CLIP ViT‑L/14 的视觉/文本编码器。

**📊 数据集**

MIT‑States、UT‑Zappos、C‑GQA 三个公开 CZSL 基准数据集。

**📈 对比分析**

将 FlowComposer 插件到 CSP 与 Troika 两个主流 VLM 基线，采用标准闭域/开域评估（AUC、HM、Seen/Unseen）。在所有三个数据集上均实现显著提升：AUC 提升 1.4%–5.0%，HM 提升 1.3%–4.4%，并显著平衡 seen/unseen 识别效果。

**⚠️ 局限性**

仍略逊于利用 LLM 生成逻辑规则的 LOGICZSL，依赖 VLM 预训练模型，未使用外部文本知识；在极少数场景下未能完全消除特征混叠；对极端稀疏属性-对象组合的泛化能力仍待进一步提升。

---

## 260. Automated identification of Ichneumonoidea wasps via YOLO-based deep learning: Integrating HiresCam for Explainable AI

**arXiv ID:** 2603.16351 | [PDF](https://arxiv.org/pdf/2603.16351v1)

**作者:** Joao Manoel Herrera Pinheiro `[一作]`, Marcelo Becker `[通讯]` (Federal University of São Carlos)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了基于YOLOv12/26的深度学习框架，并结合HiResCAM实现对Ichneumonoidea寄生蜂的高分辨率可解释性识别；

**💡 创新点**

首次将最先进YOLO模型与高分辨率XAI技术相结合，既提升识别精度又提供可视化解释；

**🔧 技术方法**

使用YOLOv12与YOLOv26 nano版作为检测器，HiResCAM做可解释性分析，辅以数据增强、512×512图像预处理等技术；

**📊 数据集**

采用DAPWH（3556张高分辨率寄生蜂图像）数据集，主要包含Ichneumonidae、Braconidae、Apidae、Vespidae等家族；

**📈 对比分析**

通过训练/验证/测试三分割，对两模型进行对比；YOLOv26在测试集上达Top‑1 Accuracy 96.14%，Precision 93.43%，Recall 97.04%，F1 95.20%，显著优于YOLOv12；

**⚠️ 局限性**

仅限于家庭级别识别，子族/属级别未覆盖；数据集规模有限，跨地域泛化性待验证；模型对高性能GPU依赖较大，实际部署受限；

---

## 261. Whose Knowledge Counts? Co-Designing Community-Centered AI Auditing Tools with Educators in Hawai`i

**arXiv ID:** 2603.16646 | [PDF](https://arxiv.org/pdf/2603.16646v1)

**作者:** Dora Zhao `[一作]` (Stanford University), Diyi Yang `[通讯]` (Stanford University)

**通讯引用:** 13519 | [OpenAlex ID](https://openalex.org/A5089413311)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过与夏威夷公共小学教师共创工作坊，探索教师对生成式AI的使用与担忧，并设计出符合当地文化价值的审核工具。

**💡 创新点**

创新点在于将AI审核框架视为社区实践，提出源知识谱系追溯、观点可视化与问题标记三大功能，并强调审计过程的去殖民化与数据主权。

**🔧 技术方法**

主要技术手段为共创设计、快速原型与情景故事板，辅以对生成式模型（ChatGPT、Gemini）输出的审计方法。

**📊 数据集**

使用的数据集为教师访谈记录、工作坊转录与设计产物，并以夏威夷文化文本为示例检验生成式AI输出。

**📈 对比分析**

论文未进行量化性能对比，而是通过主题分析和参与者反馈验证工具功能与需求匹配度，结果表明教师对审核工具认可度高。

**⚠️ 局限性**

局限性包括样本规模仅22名小学教师、仅覆盖O’ahu岛、仅涉及文本输出、缺乏学生与管理层视角，且缺乏客观评估指标。

---

## 262. Exploring different approaches to customize language models for domain-specific text-to-code generation

**arXiv ID:** 2603.16526 | [PDF](https://arxiv.org/pdf/2603.16526v1)

**作者:** Luís Freire `[一作]` (Technical University of Denmark), Nicki Skafte Detlefsen `[通讯]` (Technical University of Denmark)

**通讯引用:** 535 | [OpenAlex ID](https://openalex.org/A5063535548)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过人工智能技术，将小型开源代码生成模型（StarCoder 1B、DeepSeekCoder 1.3B）针对通用Python、Scikit‑learn和OpenCV三大领域进行定制化，以提升其在专业库使用上的准确性和可读性。

**💡 创新点**

创新点在于将大型语言模型生成的合成编程练习作为训练数据，构建轻量级定制管道，并系统比较了少量示例提示、检索增强生成与低秩适配（LoRA）三种方法在同一任务上的表现。

**🔧 技术方法**

使用的技术包括：基于GPT‑4o的合成数据生成、AST语法验证与API有效性检查、检索增强生成（向量检索+提示拼接）以及参数高效微调（LoRA）。

**📊 数据集**

数据集来源为GPT‑4o生成的约21.6k道练习题，经AST与API验证后约92–99%存活，用于训练、验证和测试；同时在HumanEval、BigBenchCode的Scikit‑learn和OpenCV子集上评测。

**📈 对比分析**

评估方法结合功能性准确率（Pass@1）和代码相似度（Cosine on embeddings），实验显示LoRA微调在所有域均取得最高Pass@1和相似度提升，尤其在OpenCV域提升约30个百分点；相比之下，few‑shot和RAG仅提升相似度，准确率提升有限。

**⚠️ 局限性**

主要限制包括对合成数据质量的高度依赖、LoRA微调需要显著GPU算力和训练时间，以及检索增强生成在检索质量与上下文窗口受限时表现不稳定。

---

## 263. Collaborative Temporal Feature Generation via Critic-Free Reinforcement Learning for Cross-User Sensor-Based Activity Recognition

**arXiv ID:** 2603.16043 | [PDF](https://arxiv.org/pdf/2603.16043v1)

**作者:** Xiaozhou Ye `[一作]` (Nanjing University of Information Science and Technology), Kevin I-Kai Wang `[通讯]` (University of Auckland)

**通讯引用:** 7936 | [OpenAlex ID](https://openalex.org/A5091532881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究跨用户可迁移的人体运动识别，将特征提取视为协作式自回归生成过程，并通过强化学习学习可泛化的时间序列特征。

**💡 创新点**

创新点在于：①把特征提取改为协作式自回归生成；②采用Critic‑free Group‑Relative Policy Optimization，获得自校准、对奖励尺度不变的优势信号；③设计三重奖励（类别区分、跨用户不变性、时间保真），实现类间分离、跨用户对齐和时序信息保持的统一平衡。

**🔧 技术方法**

技术手段包括 Transformer 自回归解码器、GRPO（无价值网络的优势估计）、三目标奖励机制以及基于生成特征的逻辑回归分类器。

**📊 数据集**

使用的公开数据集为 DSADS（8 份子、19 类）和 PAMAP2（6 份子、11 类），均为惯性测量单元（IMU）时间序列数据。

**📈 对比分析**

通过与 ERM、RSC、ANDMask、AdaRNN、ACON、PPO‑variant 等基线进行留一组外交叉验证对比，CTFG 在 DSADS 上取得 88.53% 的最高准确率，在 PAMAP2 上取得 75.22%，相较于最强基线提升约 5–10%，收敛速度更快，跨任务方差显著降低。

**⚠️ 局限性**

局限性包括：对静态动作或强环境干扰（如电梯振动）识别效果仍有限；模型对用户数量和多样性的依赖较大；组大小、奖励权重需要人工调参；在极小样本或高噪声条件下仍可能出现性能波动。

---

## 264. Faulty Coffees: Barriers to Adoption of an In-the-wild Robo-Barista

**arXiv ID:** 2603.16336 | [PDF](https://arxiv.org/pdf/2603.16336v1)

**作者:** Bruce W. Wilson `[一作]` (Heriot-Watt University), Theodoros Georgiou `[通讯]` (Heriot-Watt University)

**通讯引用:** 1405 | [OpenAlex ID](https://openalex.org/A5048071187)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在英国斯托克郡一处老年人住房小区部署Furhat机器人Robo-Barista，为居民提供每日免费咖啡，并比较叙事式与非叙事式对话在长期使用中的影响

**💡 创新点**

首次尝试在真实社区环境中将叙事策略应用于任务型服务机器人，以期提高居民的长期参与度和情感投入

**🔧 技术方法**

采用Furhat机器人、Jura咖啡机、iPad问卷、二维码扫描、RASA会话平台及Cloudflare远程维护技术

**📊 数据集**

收集了32名参与者共计44杯咖啡的交互数据、离职问卷、现场访谈和工作人员反馈；未使用公开公开数据集

**📈 对比分析**

对比叙事与非叙事两组的使用频率与满意度，但由于样本量小、复访率低，未发现显著差异；系统总体表现为高技术失败率与低复访率

**⚠️ 局限性**

主要限制包括高流失率、技术故障、用户认知与可用性不足、缺乏充分的前期用户测试以及对小样本结论的普遍化受限

---

## 265. Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory

**arXiv ID:** 2603.15800 | [PDF](https://arxiv.org/pdf/2603.15800v1)

**作者:** Ce Zhang `[一作]` (Carnegie Mellon University), Yaqi Xie `[通讯]` (Carnegie Mellon University)

**通讯引用:** 410 | [OpenAlex ID](https://openalex.org/A5032012609)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MM‑SafetyBench++评测基准与EchoSafe框架，用以提升多模态大语言模型的情境安全性能。

**💡 创新点**

创新点在于：1）通过极小修改构造安全-不安全对的高质量图文样本，实现细粒度情境安全评估；2）引入训练自由的自反记忆机制，实时检索并整合过去的安全洞察，实现推理时的情境适应与持续学习。

**🔧 技术方法**

采用CLIP/ViT-14等多模视觉语义编码器、Embedding‑based检索、Top‑k记忆召回、自我反思生成安全洞察以及Prompt注入等技术。

**📊 数据集**

使用MM‑SafetyBench++（包含安全/不安全图文对）、MM‑SafetyBench、MSSBench、SIUO以及常规问答基准MME、MMBench、ScienceQA、TextVQA等数据集进行实验。

**📈 对比分析**

与现有训练自由防御方法（FigStep、ECSO、AdaShield）在MM‑SafetyBench++上对比，EchoSafe在不安全样本拒绝率接近100%、安全样本回答率高于90%，情境正确率（CCR）提升至87.9%，在MM‑SafetyBench、MSSBench、SIUO等基准上亦实现ASR降至几乎0、S&E、R得分显著提高，且推理时间仅提升1.33×、FLOPs提升1.69×。

**⚠️ 局限性**

局限性包括：记忆库规模随持续学习增长需进一步优化；对极端对抗攻击的鲁棒性尚未彻底验证；跨模态语言模型在不同领域的泛化能力仍需更多研究。

---

## 266. Compiled Memory: Not More Information, but More Precise Instructions for Language Agents

**arXiv ID:** 2603.15666 | [PDF](https://arxiv.org/pdf/2603.15666v1)

**作者:** James Rhodes `[一作]` (AlphaBitCore), George Kang `[通讯]` (AlphaBitCore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

Atlas 将语言模型在任务中的经验通过三步验证门提炼为可验证事实，并将这些事实以子项目的形式写入系统提示，从而让模型在后续运行中自动受益。

**💡 创新点**

提出了“编译记忆”机制：将经验转化为永久指令而非仅存储上下文，并证明其性能提升受训练信号约束的“训练信号约束”属性。

**🔧 技术方法**

采用四层记忆结构、三步验证门、基于LLM的事实抽取与融合、Prompt Evolution与Anchor策略以及多目标事实抽取等技术。

**📊 数据集**

在CUAD合同分析和HotpotQA多跳问答两个基准上实验，并在Claude Sonnet 4.5上验证跨模型泛化。

**📈 对比分析**

通过与原始提示进行多轮（n=6）对比实验，CUAD F1 提升 8.7pp，HotpotQA 联合 F1 提升 3.16pp，Claude Sonnet 4.5 联合 F1 提升 2.31pp，均显著且稳定。

**⚠️ 局限性**

局限性包括仅在正样本上评估、提示长度增大可能导致提升并非完全归因于内容、事实抽取与验证需人工审核、跨模型泛化依赖训练信号覆盖以及召回平衡需额外 Anchor 等。

---

## 267. Form Follows Function: Recursive Stem Model

**arXiv ID:** 2603.15641 | [PDF](https://arxiv.org/pdf/2603.15641v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 268. Improving Generative Adversarial Network Generalization for Facial Expression Synthesis

**arXiv ID:** 2603.15648 | [PDF](https://arxiv.org/pdf/2603.15648v1)

**作者:** Arbish Akram `[一作]` (University of the Punjab), Arif Mahmood `[通讯]` (Information Technology University)

**通讯引用:** 4240 | [OpenAlex ID](https://openalex.org/A5061017734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究面部表情合成的泛化能力，提出RegGAN框架实现对外域图像的高质量表情合成。

**💡 创新点**

创新点在于将稀疏局部Ridge回归层与对抗式细化网络相结合，既保证表达精准又提升跨域泛化。

**🔧 技术方法**

使用稀疏Ridge回归、空间注意力块、编码-解码结构、三尺度判别器、颜色增强等技术。

**📊 数据集**

使用CFEE表情数据集训练回归层，FFHQ预训练细化网络，并在明星、肖像、头像等外域图像上测试。

**📈 对比分析**

与六种先进模型在ECS、FSS、RS、FID四项指标对比，RegGAN在ECS、RS和FID上均优于对手，FSS仅次于SARGAN/US-GAN，显示出更优的表情质量与真实感。

**⚠️ 局限性**

局限在于需要配对的中性-目标表情数据，模型在非人类面孔或高模糊图像上效果受限，且细化网络参数量大，影响实时部署。

---

## 269. Discovering the Hidden Role of Gini Index In Prompt-based Classification

**arXiv ID:** 2603.15654 | [PDF](https://arxiv.org/pdf/2603.15654v1)

**作者:** Ruixi Lin `[一作]` `[通讯]` (Independent Researcher), Ruixi Lin (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究并提出了一种基于Gini指数的后置无参数去偏方法D_Gini，用于检测与优化prompt‑based分类中的类准确率不平衡；

**💡 创新点**

创新点在于将经济学中的Gini不平等度量引入分类准确率评估，并将其直接作为优化目标，通过整数规划与模拟退火实现模型无参数的去偏；

**🔧 技术方法**

采用的技术包括Gini指数与COBias指标的计算、prompt‑based LLM/CLIP推理、整数优化（模拟退火）进行重加权校正；

**📊 数据集**

实验数据集包括AGNews（新闻文本）、DDI（药物相互作用关系）、CIFAR‑100（图像分类）；

**📈 对比分析**

与原始模型及COBias优化方案对比，D_Gini在AGNews将Gini从0.21降至0.03、平均准确率提升约17%；在DDI将Gini从0.67降至0.14、平均准确率提升约61%；在CIFAR‑100将Gini从0.18降至0.07、平均准确率略升3%；弱类准确率均显著提升，COBias下降86%~63%；

**⚠️ 局限性**

局限性在于方法仅为后置单次校正，未在多模态或多智能体场景验证；需手动调参且计算量较大；仅针对基本模型，缺乏对更复杂系统的直接扩展。

---

## 270. Learning Whole-Body Control for a Salamander Robot

**arXiv ID:** 2603.16683 | [PDF](https://arxiv.org/pdf/2603.16683v1)

**作者:** Mengze Tian `[一作]` (École polytechnique fédérale de Lausanne), Auke Ijspeert `[通讯]` (École polytechnique fédérale de Lausanne)

**通讯引用:** 24265 | [OpenAlex ID](https://openalex.org/A5069603317)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本研究开发了一个基于强化学习的全身关节级控制框架，直接从本体感知信息和目标速度生成所有关节的动作指令，并通过系统级真实‑模拟匹配实现了在具有高度连杆结构的蜥蜴机器人上的稳定陆地行走与陆水转换。

**💡 创新点**

创新点在于：① 用完整的关节级动作空间而非仅调节CPG参数，实现了统一的全身控制；② 构建了完整的实测–模拟对齐管线（观测、动作与运动学），大幅提升 sim‑to‑real 迁移性能；③ 将陆地与水域两种物理模式融入混合动力学系统，使同一策略能够在陆地与水中无缝切换。

**🔧 技术方法**

采用的技术包括：强化学习（PPO）、MuJoCo 物理仿真、混合动力学模型（陆地摩擦 + 水下浮力/阻力）、基于观测噪声与域随机化的 sim‑to‑real 迁移、Raspberry Pi Zero 2W 上的 ONNX 运行时、以及低层 PD 控制器。

**📊 数据集**

训练使用了多样化的地形数据集（平地、斜坡、碎片化坡面等），以及水域仿真环境；评估则基于真实机器人在实验室平地、粗糙地形以及仿真中陆到水的过渡轨迹。

**📈 对比分析**

在仿真与硬件上都进行了速度跟踪与行走性能评估，结果显示机器人在平地的平均前进速度约 0.23 m/s，粗糙地形可维持 0.17–0.19 m/s，且在仿真中的陆水过渡表现出自然的体波与步行节律转换，未出现显著性能下降。

**⚠️ 局限性**

主要局限包括：① 当前仅在仿真中实现陆水转换，缺乏物理平台的实验证明；② 受限于机器人硬件的低成本伺服，动力学与摩擦模型仍有不确定性；③ 训练所用域随机化参数范围有限，可能在更极端环境下失效。

---

## 271. Capability-Guided Compression: Toward Interpretability-Aware Budget Allocation for Large Language Models

**arXiv ID:** 2603.16440 | [PDF](https://arxiv.org/pdf/2603.16440v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 272. Bridging the High-Frequency Data Gap: A Millisecond-Resolution Network Dataset for Advancing Time Series Foundation Models

**arXiv ID:** 2603.16497 | [PDF](https://arxiv.org/pdf/2603.16497v1)

**作者:** Subina Khanal `[一作]` (Aalborg University), Torben Bach Pedersen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个毫秒级别的5G网络性能时间序列数据集，并对传统浅层模型与时序基础模型进行基准测试。

**💡 创新点**

首次填补高频无线网络数据缺口，引入高频数据域并证明现有TSFMs在此领域表现不足，揭示高频数据对预训练的重要性。

**🔧 技术方法**

使用树基模型（RF、XGB、ARF）、增量线性回归、Naive基线以及三种时序基础模型（TTM、Chronos、Lag‑Llama）进行训练和评估；采用滑动窗口、滚动评估与PEFT微调。

**📊 数据集**

自采的5G O‑RAN 5G RAN性能测量数据，时间分辨率为毫秒，随后聚合到100 ms；对比传统低频域数据集（ETTh1、电力、天气、交通）。

**📈 对比分析**

采用RMSE/MAE在单变量和多变量预测（100 ms-9.6 s）上进行零射击和微调；结果显示ARF在所有设置下表现最优，TSFMs即使微调仍落后于浅层模型。

**⚠️ 局限性**

TSFMs对高频数据的泛化能力不足；实验仅覆盖静态移动和视频流流量，缺乏多移动场景；微调实验受限于计算资源，未探索所有PEFT方法。

---

## 273. PKINet-v2: Towards Powerful and Efficient Poly-Kernel Remote Sensing Object Detection

**arXiv ID:** 2603.16341 | [PDF](https://arxiv.org/pdf/2603.16341v1)

**作者:** Xinhao Cai `[一作]` (Nanjing University of Science and Technology), Wenguan Wang `[通讯]` (Zhejiang University)

**通讯引用:** 19352 | [OpenAlex ID](https://openalex.org/A5101433884)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 PKINet-v2，一种在遥感目标检测中融合斜条卷积和方形卷积的多尺度特征提取骨干网络，旨在同时解决遥感图像的几何与空间复杂性。

**💡 创新点**

创新点包括：①将各向异性轴条卷积与等向方形卷积统一到同一模块中，构建多尺度密集感受野；②设计了 Heterogeneous Kernel Re-parameterization (HKR) 方案，将多分支训练结构在推理阶段合并为单核卷积，实现显著速度提升且无精度损失；③通过多分支深度可分离卷积与稀疏卷积的组合，有效提升对极细小目标和长条目标的检测性能。

**🔧 技术方法**

使用的技术主要有：深度可分离卷积、稀疏（dilated）卷积、1×1 融合卷积、BatchNorm 与卷积的融合、卷积重参数化（HKR）、基于 Oriented RCNN 的检测框架、MMRotate 框架、AdamW 优化、ImageNet-1K 预训练。

**📊 数据集**

实验使用的数据集包括 DOTA-v1.0、DOTA-v1.5、HRSC2016 与 DIOR-R 四个遥感目标检测基准。

**📈 对比分析**

与 ResNet‑50、LSKNet、Strip RCNN、LWGANet、PKINet‑v1 等多种基线在同一检测框架下对比，PKINet‑v2 在 DOTA‑v1.0 上 mAP 达 80.46%（比 Strip RCNN‑S 80.06% 提升 0.4%，比 PKINet‑v1‑S 78.39% 提升 2.07%），在 DOTA‑v1.5、HRSC2016、DIOR‑R 上同样刷新 SOTA；同时在推理速度上比 PKINet‑v1‑S 提升 3.9×（54.6 FPS 对比 14.05 FPS）。

**⚠️ 局限性**

局限性：尽管性能提升显著，但模型参数仍较大；HKR 在推理阶段提高速度的同时并未进一步降低模型复杂度；在极端细小目标、极大尺度或多尺度混合场景下仍可能受限于多分支设计的计算开销；实验仅涵盖四个基准，缺乏更广泛场景的验证。

---

## 274. POaaS: Minimal-Edit Prompt Optimization as a Service to Lift Accuracy and Cut Hallucinations on On-Device sLLMs

**arXiv ID:** 2603.16045 | [PDF](https://arxiv.org/pdf/2603.16045v1)

**作者:** Jungwoo Shim `[一作]` (Electronics and Telecommunications Research Institute), Hyunhwa Choi `[通讯]` (Electronics and Telecommunications Research Institute)

**通讯引用:** 23 | [OpenAlex ID](https://openalex.org/A5006178757)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为POaaS的轻量级、最小编辑的提示优化层，能在设备端sLLM前对用户提示进行保守修正、改写和事实补充，以提升任务准确性与事实性。

**💡 创新点**

创新点在于：1）针对小模型的资源受限环境设计了“最小编辑、漂移控制”策略，避免长提示导致的token浪费和偏移；2）通过三类专用代理（Cleaner、Paraphraser、Fact‑Adder）并按阈值路由，仅在必要时触发；3）引入保守跳过门和安全合并，保持原意。

**🔧 技术方法**

采用了CPU‑级启发式评估与阈值路由、LoRA 微调的专用代理、vLLM+LoRA 服务器、FastAPI orchestrator；漂移度量基于词汇相似度；合并时严格限制长度和漂移。

**📊 数据集**

在六个基准上评估：任务准确性（BBH、GSM8K、CommonsenseQA）和事实性（HaluEval、HalluLens、FActScore），以及在词典删除/混合噪声下的鲁棒性。

**📈 对比分析**

与传统APO框架（EvoPrompt、OPRO、PromptWizard）相比，POaaS在两款sLLM（Llama‑3.2‑3B、Llama‑3.1‑8B）上平均提升约2–3%准确率、1–5%事实性；并在噪声场景下恢复了7%+性能；同时在线延迟仅≈0.95 s，调用约1.4次，新增提示≈48 token，远低于APO的数千token和数百秒的离线优化时间。

**⚠️ 局限性**

局限性包括：仅适用于英文（多语种需重训）、阈值与漂移参数为经验调优、代理集仅包含三类，未探索更多领域或检索支持的优化手段。

---

## 275. Mixing Visual and Textual Code

**arXiv ID:** 2603.15855 | [PDF](https://arxiv.org/pdf/2603.15855v1)

**作者:** Leif Andersen `[一作]` (University of Massachusetts Boston), Stephen Chang `[通讯]` (University of Massachusetts Boston)

**通讯引用:** 221 | [OpenAlex ID](https://openalex.org/A5044510881)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

开发了一种混合可视化-文本编程语言Hybrid ClojureScript及其IDE，使开发者能够在代码中嵌入可交互的图形语法（VIsx），并保持与传统文本编辑器的兼容性。

**💡 创新点**

将图形语法以语言级扩展形式融入主语言，支持多阶段执行（编辑、编译、运行），保持静态推理，并可在任意程序位置（表达式、定义、模式等）使用VIsx，甚至通过VIsx定义其他VIsx。

**🔧 技术方法**

基于ClojureScript的宏系统实现语言扩展；利用DOM（CodeMirror、React、Reagent）在编辑时渲染图形；使用Stopify实现编辑时/运行时分离与沙箱；结合第三方库（vis.js、React-Bootstrap等）构建可视化组件。

**📊 数据集**

未使用公开数据集，主要通过多种示例程序（Tsuro游戏、红黑树、API状态机、Settlers of Catan等）进行功能验证。

**📈 对比分析**

与之前的Hybrid Racket版本进行功能与代码行数对比，ClojureScript实现更简洁、代码更少；在IDE性能上，采用DOM与现有库降低重绘开销，表现优于前作。

**⚠️ 局限性**

受限于ClojureScript宏实现导致文件分离与宏作用域限制；编辑时沙箱不够严格，可能影响IDE稳定性；VIsx嵌套层数过多易导致可读性下降；缺乏完整的静态类型检查支持。

---

## 276. WorldCam: Interactive Autoregressive 3D Gaming Worlds with Camera Pose as a Unifying Geometric Representation

**arXiv ID:** 2603.16871 | [PDF](https://arxiv.org/pdf/2603.16871v1)

**作者:** Jisu Nam `[一作]` (KAIST), Yang Zhou `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 WorldCam，一种基于视频 Diffusion Transformer 的交互式游戏世界模型，能够将键盘/鼠标动作精确映射为相机运动，实现长时序视频生成并保持 3D 一致性。

**💡 创新点**

创新点：① 将相机姿态作为统一几何表示，使用 Lie 代数 SE(3) 的 twist 进行精确动作-相机映射；② 通过相机姿态嵌入与姿态检索的长短期记忆机制保障 3D 一致性；③ 引入 Progressive autoregressive inference 与 Attention sink 等新技术提升长时序质量。

**🔧 技术方法**

采用技术包括视频 Diffusion Transformer + VAE 编码/解码、SE(3) 动作映射、Plücker 嵌入、记忆检索（基于相机位姿的 KNN）、逐帧噪声调度、Attention sink 等。

**📊 数据集**

使用了 WorldCam-50h 数据集，包含约 3000 分钟 Counter‑Strike、Xonotic、Unvanquished 等游戏视频，配有文本描述和伪相机位姿。

**📈 对比分析**

与 Yume、Matrix‑Game 2.0、GameCraft 等交互式世界模型及 CameraCtrl、MotionCtrl 等相机控制基线进行对比，评估动作可控性（RPE）、VBench++ 视觉质量和 3D 一致性（PSNR/LPIPS/MEt3R/DINO相似度）。WorldCam 在所有指标上均优于基线，动作误差降低 16% 以上，VBench 平均 0.844，3D 一致性指标领先。

**⚠️ 局限性**

局限性：仍需伪相机位姿估计；数据集主要为单人静态环境，缺乏多人动态交互；模型训练和推理显存/算力需求较高；在极端复杂或多任务场景下可能出现轻微漂移。

---

## 277. V-Co: A Closer Look at Visual Representation Alignment via Co-Denoising

**arXiv ID:** 2603.16792 | [PDF](https://arxiv.org/pdf/2603.16792v1)

**作者:** Han Lin `[一作]` (UNC Chapel Hill), Mohit Bansal `[通讯]` (UNC Chapel Hill)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对像素空间扩散模型进行视觉共去噪（co‑denoising）研究，提出一套可复制的生成流程。

**💡 创新点**

创新点在于：①全双流架构保持特征专属计算；②通过结构化语义→像素掩蔽实现更可靠的无监督分支；③引入感知漂移混合损失，兼顾实例级语义对齐与分布级正则；④使用 RMS 缩放对齐语义与像素的去噪难度。

**🔧 技术方法**

技术手段包括：JiT Transformer 作为去噪器；DINOv2 预训练视觉编码器作为语义流；无监督分支采用结构掩码；感知漂移混合损失与 RMS 特征重标；Classifier‑Free Guidance（CFG）在共去噪框架中实现。

**📊 数据集**

数据集：ImageNet 256×256，用于训练和 50K 采样评估。

**📈 对比分析**

实验通过与多种 SOTA 的像素空间与潜在空间扩散模型（如 JiT‑L/16、JiT‑G/16、Latent Forcing、ReDi 等）对比；V‑Co‑B/16（260M）在 FID 2.33 处与 JiT‑L/16（459M）匹配；V‑Co‑L/16 与 H/16 在 500/300 轮训练后分别取得 FID 1.72/1.71，优于 JiT‑G/16（2B）及其他强基线。

**⚠️ 局限性**

局限性：依赖高质量预训练编码器；目前仅在 ImageNet 256×256 上验证；双流结构和混合损失会增加算力与训练调参复杂度；未探究更高分辨率或多模态任务。

---

## 278. SignNav: Leveraging Signage for Semantic Visual Navigation in Large-Scale Indoor Environments

**arXiv ID:** 2603.16166 | [PDF](https://arxiv.org/pdf/2603.16166v1)

**作者:** Jian Sun `[一作]` (University of Macau), Hui Kong `[通讯]` (University of Macau)

**通讯引用:** 5336 | [OpenAlex ID](https://openalex.org/A5036133825)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了SignNav任务，让机器人在大型室内环境中利用标识的语义提示进行导航，并构建了LSI-Dataset。

**💡 创新点**

创新点在于将语义提示的空间感知与时间动态相结合，提出Spatial-Temporal Aware Transformer (START)实现端到端决策。

**🔧 技术方法**

使用Transformer架构，结合早期融合RGB‑D特征、CNN提取标识图像、Spatial‑Aware Transformer和Temporal‑Aware Transformer，并采用DAgger两阶段训练。

**📊 数据集**

使用自制的LSI‑Dataset，共20个大型室内环境的3D扫描和自动注入的标识，生成多样化轨迹。

**📈 对比分析**

与规则、VLM、ViNT等基线对比，START在val‑unseen上成功率90%、NDTW0.74，明显优于最强基线（76%/0.71），在真实机器人上也能顺利完成几百米的导航。

**⚠️ 局限性**

局限在于仅使用方向箭头，未处理文本信息；在复杂遮挡或标识缺失时仍可能失误；以及对深度估计的依赖。

---

## 279. Beyond the Embedding Bottleneck: Adaptive Retrieval-Augmented 3D CT Report Generation

**arXiv ID:** 2603.15822 | [PDF](https://arxiv.org/pdf/2603.15822v1)

**作者:** Renjie Liang `[一作]` (University of Florida), Jie Xu `[通讯]` (University of Florida)

**通讯引用:** 21237 | [OpenAlex ID](https://openalex.org/A5008162410)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出 AdaRAG-CT 框架，利用自适应检索增强 3D CT 报告生成，以弥补视觉嵌入的维度瓶颈。

**💡 创新点**

创新点在于通过可学习的检索触发标记和 oracle 混合训练，动态在视觉信息不足时注入文本上下文，显著提升临床有效性。

**🔧 技术方法**

使用的技术包括 CT-CLIP 与 ViSD-Boost 视觉编码、Llama-3.1/3.3 LLM、两阶段检索（图像-图像/文本检索）、MMR 重排以及 Self-RAG 样式的自适应检索触发。

**📊 数据集**

实验基于 CT-RATE（25692 份非对比胸部 CT 与对应报告）进行训练、验证和测试。

**📈 对比分析**

与 CT-CHAT、CT-Agent、BTB3D 等基线对比，AdaRAG-CT 8B 版临床 F1 从 0.455 提升至 0.480，70B 版从 0.405 提升至 0.426，超过前沿模型约 6% 的临床 F1。

**⚠️ 局限性**

局限性包括仅使用单一 CT-RATE 数据集、检索数据库仅来自训练集、无法根本解决嵌入维度瓶颈、评价指标未覆盖细粒度病灶特征，以及 oracle 混合训练中的 p_oracle 对数据集敏感。

---

## 280. The Decentralisation Paradox in Digital Identity: Centralising Decentralisation with Digital Wallets?

**arXiv ID:** 2603.16403 | [PDF](https://arxiv.org/pdf/2603.16403v1)

**作者:** Ioannis Konstantinidis `[一作]` (University of Macedonia), Evangelos K. Markakis `[通讯]` (Hellenic Mediterranean University)

**通讯引用:** 2509 | [OpenAlex ID](https://openalex.org/A5009055115)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨了数字身份领域中用户中心化架构所面临的“去中心化悖论”，并提出了“数字身份四象限”框架进行多维度分析。

**💡 创新点**

创新点在于将数字身份视为一个“wicked problem”，系统性地揭示技术、法律、社会与伦理四维度如何共同导致去中心化的局部实现与中心化的再分布；并提出“数字身份四象限”作为评估工具。

**🔧 技术方法**

主要技术涉及去中心化标识符（DID）、可验证凭证（VC）、区块链/分布式账本、DIDComm、OID4VCI/OID4VP、SD‑JWT、数字钱包等标准与协议。

**📊 数据集**

未使用具体实验数据集，而是基于文献综述、案例分析（欧盟EUDI Wallet、美国mDL、布宜诺斯艾利斯QuarkID等）进行理论阐述。

**📈 对比分析**

无实验比较，性能评价为理论性讨论，强调在实际部署中需权衡技术互操作性、治理模型与用户体验。

**⚠️ 局限性**

局限在于缺乏实证验证，讨论多侧重于概念性与案例性，未给出可操作的度量指标或治理方案的具体实现细节。

---

## 281. Conflict-Aware Multimodal Fusion for Ambivalence and Hesitancy Recognition

**arXiv ID:** 2603.15818 | [PDF](https://arxiv.org/pdf/2603.15818v1)

**作者:** Salah Eddine Bekhouche `[一作]` (University of the Basque Country), Abdenour Hadid `[通讯]` (Sorbonne University)

**通讯引用:** 19584 | [OpenAlex ID](https://openalex.org/A5013928164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 ConflictAwareAH 多模态框架，用于自动识别临床访谈视频中的矛盾与犹豫状态。

**💡 创新点**

创新点在于引入显式跨模态冲突特征（绝对差异向量）和文本引导的后期融合，形成冲突双向信号，从而平衡正负类并提升检测可靠性。

**🔧 技术方法**

采用 VideoMAE（视频编码器）、HuBERT（音频编码器）、RoBERTa-GoEmotions（文本编码器）三大预训练模型，结合注意力池化、冲突特征拼接、两层 FFN、文本辅助头以及多窗口推理与模型集成。

**📊 数据集**

使用 ABAW10 竞赛中的 BAH 数据集（1,427 条临床访谈视频，778 条训练样本）进行训练与评估。

**📈 对比分析**

与已有多模态基线（最高 Macro F1 0.634）比较，标签测试集上单模型实现 0.690–0.692，集成模型达到 0.694，私有测试集获得 0.715，提升约 10 点，训练仅需 25 分钟单 GPU。

**⚠️ 局限性**

局限主要包括阈值在分布漂移时可能失效、单窗口视频短时覆盖不足、未进行跨模态对齐导致冲突特征仅为几何距离、对 Whisper ASR 产生的文本错误敏感等。

---

## 282. Automated Self-Testing as a Quality Gate: Evidence-Driven Release Management for LLM Applications

**arXiv ID:** 2603.15676 | [PDF](https://arxiv.org/pdf/2603.15676v1)

**作者:** Alexandre Cristovão Maiorano `[一作]` `[通讯]`, Alexandre Cristovão Maiorano

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一套基于自动自测的质量门控框架，用于在LLM应用的发布前评估系统的多维质量并决定是否推广、保留或回滚。

**💡 创新点**

创新点在于将任务成功率、研究上下文保持率、P95延迟、安全通过率和证据覆盖率这五个经验驱动维度整合为一个可执行的PROMOTE/HOLD/ROLLBACK决策逻辑，并通过长期实验验证证据覆盖率是判定严重回退的关键指标。

**🔧 技术方法**

技术包括：动态“问题库”驱动的自测引擎、OpenTelemetry实时跟踪、LangGraph状态机记录、统计分析（Mann‑Kendall、Spearman、Bootstrap CI）以及LLM-as‑Judge与人工评估相结合的校准研究。

**📊 数据集**

使用的数据集为内部持续演进的“问题库”，从59道基础场景扩展到133道，涵盖核心功能、多轮协同、幻觉陷阱与对抗安全边界等四类场景，另外公开提供了83道可复现的测试用例。

**📈 对比分析**

通过与传统单元/集成测试对照、维度剔除消融实验和人工/LLM评估对比，发现自动门控能够正确识别并回退两次严重回归，且在38次评估中仅出现两次失败，整体表现优于仅靠传统测试的结果。

**⚠️ 局限性**

局限性包括：仅在单一营销分析多代理系统上验证，问题库随时间增长导致跨运行可比性下降；门控阈值需人工调校，可能不适用于所有领域；实验周期有限，缺乏与其他竞争方法的直接对标；且二值阈值化可能忽略细微质量差异。

---

## 283. Demystifing Video Reasoning

**arXiv ID:** 2603.16870 | [PDF](https://arxiv.org/pdf/2603.16870v1)

**作者:** Ruisi Wang `[一作]` (SenseTime Research), Lei Yang `[通讯]` (SenseTime Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文系统性地拆解了扩散式视频生成模型在推理过程中的内部机制，发现推理主要沿扩散步骤展开（Chain-of-Steps，CoS），而非之前假设的帧序推理（Chain-of-Frames，CoF），并进一步揭示了工作记忆、自我纠错和感知先行等新兴推理行为。

**💡 创新点**

创新点：
1) 提出CoS理论，展示多路径探索与叠加式探索在扩散步骤中的实现；
2) 发现模型自发的工作记忆、自我纠错与感知先行动作；
3) 通过层级可视化和隐层交换实验揭示Diffusion Transformer各层的功能分化；
4) 提出训练无关的多种子隐层集成（latent ensemble）提升推理准确率。

**🔧 技术方法**

使用技术：
- Diffusion Transformer（DiT）视频生成模型；
- 逐步解码（x̂₀）可视化推理轨迹；
- 噪声注入实验验证推理轴心；
- token-level 能量可视化与层级分析；
- 隐层交换实验验证因果贡献；
- 训练无关的多种子隐层集成实现推理路径增强。

**📊 数据集**

数据集：
- VBVR-Wan2.2（基于 Wan2.2-I2V-A14B 的视频推理数据）；
- VBVR benchmark（视频推理基准）；
- VBench（通用视频生成基准）；
- 其他公开的大规模视频推理数据集。

**📈 对比分析**

比较方法与性能：
- 在 VBVR-Bench 上对比基线模型与集成方法，集成提升约 2% 绝对分数；
- 噪声注入实验对比“噪声在步骤”与“噪声在帧”两种扰动，验证 CoS 的关键性；
- 通过层级可视化与隐层交换实验验证模型内部推理贡献；
- 与传统 CoF 假设的模型表现进行对照，证明 CoS 的更高有效性。

**⚠️ 局限性**

局限性：
- 仅在特定的扩散式视频生成模型（Wan2.2）上验证，缺乏跨模型泛化验证；
- 机制解释仍为经验性观察，未给出完整的理论推导；
- 多种子集成方法提升有限，需探索更高效的推理增强策略；
- 对极大规模、复杂逻辑推理任务的鲁棒性尚未充分评估；
- 仅针对视频推理任务，缺乏与其他多模态推理场景的比较。

---

## 284. Scientific Machine Learning-assisted Model Discovery from Telemetry Data

**arXiv ID:** 2603.15943 | [PDF](https://arxiv.org/pdf/2603.15943v1)

**作者:** Sebastian Micluta-Campeanu `[一作]` (JuliaHub), Chris Rackauckas `[通讯]` (JuliaHub)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对交通冷藏单元（TRU）的两区货箱模型进行半自动化的模型发现，利用数据校准模型并通过符号回归生成物理可解释的修正项。

**💡 创新点**

首次提出工程师与AI协同的“Dyad Model Discovery”工作流，在保持物理可解释性的同时实现自动化模型改进，并结合输出掩蔽、灵敏度分析和符号回归。

**🔧 技术方法**

使用Universal Differential Equations（UDE）与神经网络、输出掩蔽、灵敏度分析、符号回归（SciML栈）以及Dyad框架进行模型校准和改进。

**📊 数据集**

采用TRU热流数据集，包括不同车辆配置（配置B、E）的实测温度序列，训练使用两条数据，测试使用第三条数据。

**📈 对比分析**

通过与原始物理模型比较，UDE模型在未见数据集上损失降低约3%，并且符号回归得到的修正项在泛化性上表现优异；实验对比多种网络架构并选取最佳。

**⚠️ 局限性**

局限包括需工程师介入选择网络架构、掩蔽输出、符号回归输入；符号表达中出现的假导数（dummy derivatives）缺乏物理意义；以及模型仅在编译后应用修正，尚未实现对用户代码中具体方程的直接改动。

---

## 285. What if Pinocchio Were a Reinforcement Learning Agent: A Normative End-to-End Pipeline

**arXiv ID:** 2603.16651 | [PDF](https://arxiv.org/pdf/2603.16651v1)

**作者:** Benoît Alcaraz `[一作]` (University of Luxembourg), Benoît Alcaraz `[通讯]` (University of Luxembourg)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5079582735)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个基于论证的规范监督机制的强化学习端到端管道，以实现规范遵从与情境感知的智能体。

**💡 创新点**

创新点在于将论证推理与强化学习相结合，设计了可自动抽取规范论证关系的算法，并对规范规避现象提出定义与缓解策略。

**🔧 技术方法**

采用强化学习（如DQN/PPO）、论证推理框架（AJAR、Jiminy、NGRL）以及自定义的论证抽取算法。

**📊 数据集**

在标准强化学习基准环境（如OpenAI Gym的Atari/MuJoCo任务）以及自构造的规范化情境数据集上进行实验。

**📈 对比分析**

与传统RL、纯规则式代理以及其他基线（如A3C、SAC）进行对比，结果表明该管道在任务完成率和规范违规率方面均有显著提升。

**⚠️ 局限性**

局限性包括对复杂多主体规范的建模难度、抽取算法对领域知识依赖强、计算开销较大，以及在动态未知环境中的泛化能力有限。

---

## 286. Toward Deep Representation Learning for Event-Enhanced Visual Autonomous Perception: the eAP Dataset

**arXiv ID:** 2603.16303 | [PDF](https://arxiv.org/pdf/2603.16303v1)

**作者:** Jinghang Li `[一作]` (Hunan University), Yi Zhou `[通讯]` (Hunan University)

**通讯引用:** 6484 | [OpenAlex ID](https://openalex.org/A5046991303)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了eAP（事件增强自主感知）大规模数据集，并基于该数据集构建了事件增强的3D车辆检测模型与几何感知TTC估计框架Garl-TTC，实现了端到端的实时推理。

**💡 创新点**

创新点包括：①首次公开大型事件相机自动驾驶数据集eAP；②将事件与RGB信息融合提升BEV 3D检测鲁棒性；③提出利用视觉高度比、知识蒸馏与前景分割的几何感知TTC学习方法，显著提升精度并实现200FPS推理。

**🔧 技术方法**

采用事件体素化编码、ResNet-50多模态编码器、BEV视图变换、几何高度比回归、知识蒸馏（SAM）与前景分割监督，结合TensorRT实现边缘设备的实时推理。

**📊 数据集**

使用自建的eAP数据集（1280×720事件相机+RGB+LiDAR 3D框+TTC标签）进行训练与评测，并与EvTTC、FCWD、DSEC、MVSEC等公开数据集做对比验证。

**📈 对比分析**

在3D检测任务中，与视觉/事件单模基线相比，事件增强模型在HDR和小TTC场景下AP提升约10%；在TTC估计任务中，Garl-TTC在EvTTC和FCWD benchmark 上MiD和RTE 均低于现有SOTA，且实现约200FPS实时性能。

**⚠️ 局限性**

局限性包括对高旋转ego运动敏感、事件在低相对速度场景稀疏导致检测/估计性能下降、模型未考虑闭环控制与机械延迟，未来需融合IMU补偿并进一步量化优化以适配边缘平台。

---

## 287. Omanic: Towards Step-wise Evaluation of Multi-hop Reasoning in Large Language Models

**arXiv ID:** 2603.16654 | [PDF](https://arxiv.org/pdf/2603.16654v1)

**作者:** Xiaojie Gu `[一作]` (University of Tokyo), Irene Li `[通讯]` (University of Tokyo)

**通讯引用:** 2423 | [OpenAlex ID](https://openalex.org/A5101537931)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个开放域的四跳多步推理问答基准Omanic，并提供了10,296条机器生成的训练样例（OmanicSynth）和967条专家审核的测试样例（OmanicBench），每条样例都包含分解的单跳子问题和中间答案；同时对多种大型语言模型在该基准上的表现进行了系统评估并揭示了Chain‑of‑Thought（CoT）在知识完整性和错误传播方面的限制。

**💡 创新点**

首创提供细粒度的步骤级别注释，使得多跳推理过程可被逐步诊断；将数学推理嵌入多跳链路并设计不同图拓扑以防止捷径；通过实验证明OmanicSynth的训练样本能显著提升跨域推理与数学推理能力。

**🔧 技术方法**

采用Chain‑of‑Thought提示、监督微调（SFT）与基于GRPO的强化学习；使用Wikidata5M进行三元组检索与知识合成；通过Claude‑Sonnet‑4.5进行语义合成；采用四大模型（Llama‑3.1‑8B‑Instruct、Qwen3‑8B、Mistral‑7B‑Instruct‑v0.3、Gemma‑3‑4b‑it）进行自动过滤；人工专家审核确保事实准确性与推理连贯性。

**📊 数据集**

利用MuSiQue的原始2跳问题及其答案作为起点；检索Wikidata5M中的三元组；生成10,296条OmanicSynth训练样例；筛选并人工审核得到967条OmanicBench测试样例；在训练后还在MATH、HotpotQA、MuSiQue等多个推理与数学基准上进行迁移评估。

**📈 对比分析**

在多跳推理的多项选择和开放式生成两种评估范式下，比较了多款专有与开源LLM的原始与CoT提示性能；最高单跳推理准确率为73.11%；在OmanicSynth上进行监督微调后，开源模型平均提升7.41分，并在六个外部推理/数学基准上持续优于未微调版本。

**⚠️ 局限性**

仅支持英文，且仅包含4跳推理；未覆盖更长链路（如6跳、8跳）和专业领域（法律、医学）；数据规模有限，可能影响模型训练与评估的广泛性；多跳推理难度依赖特定知识图谱，跨语言迁移受限。

---

## 288. Omnilingual MT: Machine Translation for 1,600 Languages

**arXiv ID:** 2603.16309 | [PDF](https://arxiv.org/pdf/2603.16309v1)

**作者:** Omnilingual MT Team `[一作]` (FAIR at Meta), Marta R. Costa-jussà `[通讯]` (FAIR at Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Omnilingual MT 系统，结合 decoder‑only 与 encoder‑decoder 两种架构，并通过大规模持续预训练、合成数据和检索增强技术，将多语种翻译覆盖扩展至 1000+ 语种，支持 1200+ 语言的生成与翻译。

**💡 创新点**

创新点包括：① 在 LLM 基础上实现专用翻译模型；② 采用 256K 词表与改进的预分词器显著提升低资源脚本的表示；③ 通过持续预训练、后训练与 RL 的联合微调实现高质量翻译；④ 结合检索增强翻译（RAG）在低资源场景下提升效果；⑤ 构建并发布覆盖 2000+ 语言的训练与评测数据集。

**🔧 技术方法**

技术手段涵盖 LLaMA‑3.1、跨语言对齐编码器、持续预训练、监督微调、强化学习（DAPO）、检索增强翻译（RAG）、合成反向翻译、比对挖掘、扩展 BPE 词表与多语言分词规则。

**📊 数据集**

使用的数据集包括 Common Crawl、Bible、Panlex、Tatoeba、SMOL、Gatitos、BPCC、KreyolMT、Afrolingu‑MT、LTPP、人工合成反向翻译与对齐数据，以及自研的 2000+ 语言训练集、评测集（FLORES‑200、Bible‑John、Meta‑Bouquet 等）。

**📈 对比分析**

通过与 NLLB‑200、TowerLLM、70B‑LLM 等基线对比，Omnilingual MT 在 400+ 语言实现可用质量，1200+ 语言实现生成覆盖；1B‑8B 专用翻译模型的性能可与 70B LLM 匹敌或超越；检索增强进一步提升 5‑10% 的自动评测分数。

**⚠️ 局限性**

局限性包括：仍缺乏完整覆盖 7000 语言；Bible 训练集可能导致模型泄露与评测污染；合成数据质量不均衡；检索数据库对极低资源语种覆盖不足；模型规模与算力需求仍高，尤其是 8B+ 版本；以及对极低资源语种的细粒度评测与调优仍有待改进。

---

## 289. RECOVER: Robust Entity Correction via agentic Orchestration of hypothesis Variants for Evidence-based Recovery

**arXiv ID:** 2603.16411 | [PDF](https://arxiv.org/pdf/2603.16411v1)

**作者:** Abhishek Kumar `[一作]` (Observe.AI), Aashraya Sachdeva `[通讯]` (Observe.AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于多重 ASR 结果与受约束 LLM 编辑的后处理框架 RECOVER，用以纠正自动语音识别中的实体错误。

**💡 创新点**

创新点包括：①将纠错拆分为三个工具（融合、建议、验证）并由 agent 调度；②多假设融合策略（Entity‑Aware Select、ROVER、LLM‑Select）以利用假设间互补错误；③使用严格约束的 LLM 找/换式编辑，配合确定性 guardrail 防止 hallucination。

**🔧 技术方法**

技术手段包括 Whisper‑small 进行多温度采样生成 N 个假设；基于 exact/fuzzy/phonetic 分数检索 top‑K 实体候选；使用 GPT‑4o（或 GPT‑4o‑mini）进行实体级编辑；以及对齐与插入/删除控制的确定性检查。

**📊 数据集**

评估数据集覆盖五个领域：财务（Earnings‑21）、空中交通（ATCO2）、医疗（Eka‑Medical）、通用语音（Common Voice）和对话电影（ContextASR‑Bench）。

**📈 对比分析**

相较于基线（Greedy Whisper）和四种融合策略，RECOVER 在所有数据集实现 8–46% 的相对实体错误率（E‑WER）降低，实体召回提升最多 22 个百分点，同时总体 WER 维持不变或略升，显示出强健的实体纠错效果。

**⚠️ 局限性**

局限性在于对大型 LLM 的依赖、需要预先准备的实体词表、融合策略在噪声域可能产生插入噪声，以及多假设生成对 ASR 模型与硬件资源有一定要求。

---

## 290. A Depth-Aware Comparative Study of Euclidean and Hyperbolic Graph Neural Networks on Bitcoin Transaction Systems

**arXiv ID:** 2603.16080 | [PDF](https://arxiv.org/pdf/2603.16080v1)

**作者:** Ankit Ghimire `[一作]` (University of Southern Mississippi), Nick Rahimi `[通讯]` (University of Southern Mississippi)

**通讯引用:** 259 | [OpenAlex ID](https://openalex.org/A5102764912)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对比了欧氏空间与双曲空间的图神经网络在比特币交易网络节点分类上的表现，并分析了邻域采样深度与嵌入几何对性能的影响。

**💡 创新点**

系统地将邻域深度和几何空间分离对比，发现双曲空间在深层网络下显著优于欧氏空间；同时揭示了曲率与学习率共同决定双曲网络优化稳定性的规律。

**🔧 技术方法**

使用了GCN、GraphSAGE、GAT 三种经典 GNN 架构，并分别在欧氏空间和双曲（Poincaré 球面）空间中实现；采用了双曲空间的切空间映射、指数映射与对数映射；利用了多层、不同深度的子图采样及多头注意力。

**📊 数据集**

使用了约 252M 节点、785M 边的全量比特币交易图，并从中抽取了 34K 带标签节点（七类实体）进行实验；数据集为公开的“Schnoering 和 Vazirgiannis”大规模交易图。

**📈 对比分析**

通过构造固定深度（2 层/3 层）且固定 fan‑out 的自心子图，保持模型架构和维度一致，只改变几何空间；评估指标为宏平均 F1、精确率和召回率。实验显示：在 3 层模型上，双曲 GraphSAGE 的宏 F1 达到 0.81（欧氏仅 0.73），且在多类实体上表现出更高的召回率，尤其在结构模糊的 “Exchange、Mining、Ponzi、Ransomware” 等类别。

**⚠️ 局限性**

研究仅限于静态交易图，未考虑时间动态；子图采样采用固定深度与 fan‑out，可能忽略更复杂的邻域结构；双曲模型使用固定曲率，未探索自适应或混合几何；实验集仅为比特币网络，缺乏跨链或其他金融系统的验证。

---

## 291. Evaluating Latent Space Structure in Timbre VAEs: A Comparative Study of Unsupervised, Descriptor-Conditioned, and Perceptual Feature-Conditioned Models

**arXiv ID:** 2603.16713 | [PDF](https://arxiv.org/pdf/2603.16713v1)

**作者:** Joseph Cameron `[一作]` (University of Cambridge), Alan Blackwell `[通讯]` (University of Cambridge)

**通讯引用:** 6854 | [OpenAlex ID](https://openalex.org/A5017575045)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文比较了三种变分自编码器（VAE）在电吉他音色生成中的潜在空间结构：无监督VAE、基于一热语义标签的条件VAE以及基于AudioCommons连续感知特征的条件VAE。

**💡 创新点**

创新点在于：①首次系统比较语义标签与感知特征两种条件化策略对音色潜在空间的影响；②提出一套专门针对音色的解释性评估指标（如音色描述符紧凑度、跨音高一致性、线性与步进一致性）；③展示感知特征条件化能显著提升潜在空间的紧凑性、音高不变性和可解释性。

**🔧 技术方法**

使用技术包括：卷积编码器-解码器结构的VAE；条件VAE（CVAE）与感知特征条件VAE（AudioCommonsCVAE）；训练设置为300 epochs、Adam优化器、学习率0.0005、批量64；评估指标包括Silhouette、Purity、紧凑度、跨音高一致性、线性、步进一致性等。

**📊 数据集**

使用的数据集为从Semantic Timbre Dataset中挑选的1771个单音吉他频谱图，标注有19个语义音色描述符、四个强度等级（25%/50%/75%/100%）以及23个音高（E4到D6）对应的音频，采用1024点FFT、512 hop的STFT表示。

**📈 对比分析**

比较方法：在完全相同的网络结构、训练超参数和数据集下，分别训练三种VAE模型；随后对每个模型的潜在空间使用标准聚类指标和新提出的音色特定指标进行评估。实验结果显示，感知特征条件化的AudioCommonsCVAE在紧凑度、Purity、跨音高一致性以及音色描述符在同一音高内的分离度上均优于无监督VAE和一热语义条件VAE，其余指标（步进一致性、线性）相近。

**⚠️ 局限性**

局限性包括：所有模型潜在空间仍受音高主导，导致音色分离有限；仅在电吉他数据上验证，未检验对其他乐器或更复杂数据集的泛化能力；缺乏更深入的解耦策略来进一步消除音高与音色的耦合。

---

## 292. One Kiss: Emojis as Agents of Genre Flux in Generative Comics

**arXiv ID:** 2603.16359 | [PDF](https://arxiv.org/pdf/2603.16359v1)

**作者:** Xiruo Wang `[一作]` (University College London), Ziqi Lyu `[通讯]` (Beijing Forestry University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一款名为One Kiss的协同创作漫画生成系统，用户通过画板草图设定结构节奏，并配合关键词+表情符号进行情感导向，系统通过情感累积算法实现叙事情绪的逐步转变（Genre Flux）

**💡 创新点**

创新点在于：①将情感模糊性（emoji）视为设计资源，以软控制方式实现叙事情绪导向；②双流输入架构（空间节奏与情感注入）与Genre Flux算法相结合，实现跨帧情绪累积和风格漂移；③通过全局角色锚定平衡情绪漂移与角色一致性

**🔧 技术方法**

采用Stable Diffusion结合ControlNet进行图像生成；双流前端（绘图与emoji/关键词配对）与后端动态Narrative State Vector、Genre Flux算法、Style Modifier、Global Character Anchor等模块；使用情感向量（Romance、Tragedy、Chaos、Mystery）和权重更新公式实现情绪状态管理

**📊 数据集**

论文未公开使用自定义数据集；模型基于公开预训练的Stable Diffusion及相关开源工具，关键词与emoji对接使用通用情感词典/emoji情感标注（如SentiWordNet、Emoji Sentiment Ranking）

**📈 对比分析**

仅进行6人探索性实验（N=6），通过生成6帧漫画并进行定性分析，验证所有参与者均能实现Genre Flux；未给出定量指标或与传统文本提示系统的性能对比，仅呈现主观体验与满意度提升

**⚠️ 局限性**

局限性包括：①样本量小、仅为探索性研究；②空间节奏流缺乏渐进性控制，导致后续帧受初始关键词影响；③缺乏精细化情感强度调节；④角色一致性仅通过全局锚定实现，可能不适用于多角色场景；⑤系统对emoji语义稀有度的权重设计尚未充分验证

---

## 293. Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning

**arXiv ID:** 2603.15789 | [PDF](https://arxiv.org/pdf/2603.15789v1)

**作者:** Patrick Yin `[一作]` (University of Washington), Abhishek Gupta `[通讯]` (University of Washington)

**通讯引用:** 5934 | [OpenAlex ID](https://openalex.org/A5017906439)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

使用自动生成多样化重置状态的框架OmniReset，在大规模并行仿真中训练PPO策略，实现在多阶段复杂操作上的自我学习；

**💡 创新点**

创新点在于通过无任务特定奖励、无课程或演示的多样化重置分布，极大提升探索效率并让策略在广泛初始条件下成功；

**🔧 技术方法**

采用PPO+gSDE+非对称actor‑critic在Isaac Lab物理仿真中进行大规模并行训练，并通过学生‑教师蒸馏与视觉随机化实现真实机器人部署；

**📊 数据集**

使用自定义的抓取点和目标偏移生成的内部重置状态集，并在真实实验中使用RGB摄像头记录；未使用公开数据集；

**📈 对比分析**

与BC‑PPO、DeepMimic和Demo Curriculum等基线对比，OmniReset在硬模式任务上取得显著更高的成功率，并在真实世界零射击转移中实现25%成功率，远超4%基线；

**⚠️ 局限性**

目前仅限于单一任务/单一物体，未扩展到多任务或更高维手部操作，且对仿真环境的物理参数仍相当敏感。

---

## 294. Same Performance, Hidden Bias: Evaluating Hypothesis- and Recommendation-Driven AI

**arXiv ID:** 2603.15824 | [PDF](https://arxiv.org/pdf/2603.15824v1)

**作者:** Michaela Benk `[一作]` (University of Zurich), Tim Miller `[通讯]` (University of Queensland)

**通讯引用:** 7909 | [OpenAlex ID](https://openalex.org/A5028824146)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在网页实验中比较五种AI交互设计（控制、推荐驱动、推荐+解释、推荐+假设探索、假设驱动），检验其对用户决策策略的影响。

**💡 创新点**

首次将信号检测理论（SDT）应用于AI辅助决策，揭示交互设计如何产生“隐藏偏差”，即改变用户的证据阈值而非仅影响准确率。

**🔧 技术方法**

采用SDT与混合效应模型来评估敏感度d′和判别准则c，并分析用户在不同条件下的决策过程。

**📊 数据集**

使用OnlyConnectWall词联想谜题数据集，并用GPT‑4生成的AI建议模拟推荐与解释，保持所有刺激和AI输出的一致性。

**📈 对比分析**

通过比较准确率、反应时和SDT指标，发现总体准确率相同但推荐驱动条件下c值显著更低，表明更宽松的证据阈值，性能指标未出现显著差异。

**⚠️ 局限性**

研究受限于静态任务环境、单一解释方式、样本规模与非专家参与者，结果对真实高风险情境和专家群体的外推性尚待验证。

---

## 295. PashtoCorp: A 1.25-Billion-Word Corpus, Evaluation Suite, and Reproducible Pipeline for Low-Resource Language Development

**arXiv ID:** 2603.16354 | [PDF](https://arxiv.org/pdf/2603.16354v1)

**作者:** Hanif Rahman `[一作]` `[通讯]` (Independent Researcher), Hanif Rahman (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了1.25亿词、2.81百万文档的PashtoCorpus，并在其上进行MLM预训练，提升低资源Pashto语言模型的性能。

**💡 创新点**

创新点在于大规模多源、可复现的语料构建流程，以及证明少量高质量文本（如PDF书籍、维基百科）对词汇覆盖和下游任务的巨大价值。

**🔧 技术方法**

采用HuggingFace数据集、Scrapy爬虫、Unicode脚本识别、SHA‑256去重、MLM预训练XLM‑R-base等技术。

**📊 数据集**

共使用39个来源，包含7个HF数据集（FineWeb2、HPLT、CulturaX、CC‑100、MADLAD‑400、GlotCC）和32个定制爬虫（新闻、广播、PDF书籍、百科等）。

**📈 对比分析**

与XLM‑R-base相比，PPL从8.08降至6.06（↓25%），WikiANN NER F1从19.0%升至21.0%（↑10%），在Belebele阅读理解中Gemma‑3n获得64.6%准确率，展示出显著性能提升。

**⚠️ 局限性**

主要局限在于预训练仅使用Corpus 8%（100M词），数据偏向新闻与阿富汗方言，社交媒体覆盖不足，WikiANN NER样本极少导致高方差，且未覆盖部分Pashto方言与非正式语料。

---

## 296. SpecMoE: Spectral Mixture-of-Experts Foundation Model for Cross-Species EEG Decoding

**arXiv ID:** 2603.16739 | [PDF](https://arxiv.org/pdf/2603.16739v1)

**作者:** D. Darankoum `[一作]`, S. Grudinin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了SpecMoE，一个基于高斯平滑时频掩蔽、U形多尺度SpecHi-Net与谱引导Mixture of Experts的跨物种EEG基础模型，能够在多任务下实现高性能解码。

**💡 创新点**

创新点包括：1）使用Gaussian平滑掩蔽消除矩形掩蔽产生的边缘伪波，真正逼近神经节律；2）在SpecHi-Net中采用双路径卷积与RoPE位置编码实现多尺度时频特征抽取；3）引入谱门控的Mixture of Experts，根据信号功率谱动态加权专家贡献，提升跨任务与跨物种泛化。

**🔧 技术方法**

技术栈：STFT+逆STFT、Gaussian平滑掩蔽、U-Net式SpecHi-Net、RoPE位置编码、全参数微调的Spectral Gating MoE、联合时域MSE与频域功率谱损失、三专家并行编码与MVP融合。

**📊 数据集**

预训练使用TUEG 9000小时临床EEG（约1.1M段30s样本），微调使用9个公开基准（人类：PhysioNet-MI、SEED-V、HMC、BCIC2020-3、TUAB、Siena、SEED-VIG；小鼠：MACO、DA-Pharmaco），覆盖运动想象、情绪识别、睡眠分期、药物效应、想象语音、异常检测、癫痫检测与警觉估计。

**📈 对比分析**

对比三种任务专用模型（EEGNet、EEG-Conformer、FFCL）和三种主流基础模型（LaBraM、CBraMod、CSBrain）。在9项基准中，SpecMoE在7项任务获得最佳或第二最佳性能，尤其在MACO、Siena、DA-Pharmaco与SEED-VIG等任务上分别提高7.2%、9.9%、6%及近一半RMSE，验证了跨物种、跨任务的强泛化。

**⚠️ 局限性**

局限性：1）仍需对所有参数进行微调以达到最优，尚未实现零射击泛化；2）掩蔽比例（默认50%）与形状的最佳设定未系统探索；3）专家专化来源于随机数据划分，未尝试领域特定专家训练；4）模型规模相对较大，计算与存储成本仍高。

---

## 297. Deriving Hyperparameter Scaling Laws via Modern Optimization Theory

**arXiv ID:** 2603.15958 | [PDF](https://arxiv.org/pdf/2603.15958v1)

**作者:** Egor Shulgin `[一作]` (OpenEuroLLM team at ELLIS Institute), Antonio Orvieto `[通讯]` (Mila)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于线性最小化Oracle（LMO）优化器的收敛界，推导了学习率、批量大小和动量在固定模型规模下随训练步数和令牌预算的闭式幂律缩放关系。

**💡 创新点**

创新点在于将LMO框架下的理论收敛上界作为代理目标，通过对代理函数最小化得到统一的幂律缩放规则，并首次揭示了存在令牌最优批量大小和多种可实现最优性能的缩放策略。

**🔧 技术方法**

采用优化理论与LMO框架、梯度噪声方差假设、收敛界近似、代理函数最小化等技术，结合对动量、学习率、批量大小的联合分析。

**📊 数据集**

主要以理论分析为主，实验验证使用了160M规模的Transformer模型在SlimPajama语言建模任务上（5B令牌）进行常数学习率的批量/学习率探索，验证理论预测。

**📈 对比分析**

与已有经验公式（如√批量学习率缩放、令牌预算下学习率递减等）进行对比，实验显示推导的幂律与经验结果高度吻合，且所有缩放方案在令牌预算足够大时都能达到T⁻¹⁴⁰⁴的收敛率。

**⚠️ 局限性**

局限性包括：假设学习率恒定、未考虑权重衰减和warmup、模型规模保持不变、常数Cᵢ取值为1、对梯度噪声、初始化和高阶效应的简化假设可能导致与真实大规模训练的差距。

---

## 298. Nonstandard Errors in AI Agents

**arXiv ID:** 2603.16744 | [PDF](https://arxiv.org/pdf/2603.16744v1)

**作者:** Ruijiang Gao `[一作]` (University of Texas at Dallas), Steven Chong Xiao `[通讯]` (University of Texas at Dallas)

**通讯引用:** 9597 | [OpenAlex ID](https://openalex.org/A5041851882)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对150个Claude Code AI代理在相同数据和研究问题上进行实验，评估其非标准误差（NSE）及三阶段反馈对结果的影响。

**💡 创新点**

首次将NSE概念迁移到AI代理，揭示AI代理的系统性方法偏好与结果差异，并研究AI同行评审与示例论文曝光对收敛的作用。

**🔧 技术方法**

使用Claude Code（Sonnet4.6/Opus4.6）自动生成代码与报告，三阶段反馈协议（独立分析→AI同行评审→示例论文曝光），以及多宇宙分析框架。

**📊 数据集**

使用NYSE TAQ毫秒级SPY交易数据2015-2024（约66GB、70亿行）。

**📈 对比分析**

通过比较六个假设在各阶段的IQR，发现AI同行评审几乎无效，而示例论文曝光使IQR收敛率达80-99%，显示AI代理在方法选择上存在巨大变异。

**⚠️ 局限性**

局限在于仅针对单一资产与微观结构领域，缺乏跨领域验证；示例论文选择可能受评审偏差；交易方向分类等数据处理未进行验证；AI多样性可能被后期对齐削弱。

---

## 299. How to Achieve Prototypical Birth and Death for OOD Detection?

**arXiv ID:** 2603.15650 | [PDF](https://arxiv.org/pdf/2603.15650v1)

**作者:** Ningkang Peng `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 631 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种动态原型出生与死亡机制，能够在OOD检测中根据数据复杂度自适应地调整每个类别的原型数量。

**💡 创新点**

核心创新在于将动态原型控制器嵌入MAP‑EM框架中，实现了无手工设定的自适应原型数；同时通过方差与边界分数实现出生与死亡决策。

**🔧 技术方法**

采用了von Mises‑Fisher混合模型、Sinkhorn‑Knopp软分配、MAP‑EM迭代、EMA原型更新、方差与距离比阈值等技术，并使用Mahalanobis距离做OOD评分。

**📊 数据集**

实验基准包括CIFAR‑100、CIFAR‑10和ImageNet‑100，外加远OOD（SVHN、LSUN、iSUN、Places365、Textures）与近OOD（LSUN‑F、ImageNet‑FIX/RESIZE、CIFAR‑10）数据集。

**📈 对比分析**

与MSP、Vim、ODIN、Energy、VOS、CSI、SSD+、kNN+、NPOS、CIDER、PALM、DMPL等方法在CIFAR‑100上对比，PID在FPR@95和AUROC上分别实现约24% FPR下降、93.93% AUROC的SOTA性能；近OOD任务同样取得最优表现。

**⚠️ 局限性**

缺点是动态机制引入了若干敏感阈值与冷却周期等超参数，需手动调优，降低了系统的自动化程度。

---

## 300. InViC: Intent-aware Visual Cues for Medical Visual Question Answering

**arXiv ID:** 2603.16372 | [PDF](https://arxiv.org/pdf/2603.16372v1)

**作者:** Zhisong Wang `[一作]` (National Engineering Laboratory for Integrated Aero-Space-Ground-Ocean Big Data Application Technology), Yong Xia `[通讯]` (National Engineering Laboratory for Integrated Aero-Space-Ground-Ocean Big Data Application Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 InViC，一个轻量级插件框架，通过提示式视觉线索增强医学视觉问答的图像依赖性。

**💡 创新点**

创新点在于引入 Cue Tokens Extraction 以及两阶段提示瓶颈训练，强制模型通过意图感知的视觉线索获取证据，降低捷径回答。

**🔧 技术方法**

采用视觉编码器+文本编码器、跨模态注意力、提示瓶颈掩码、LoRA 微调等技术。

**📊 数据集**

使用 VQA-RAD、SLAKE、ImageCLEF VQA-Med 2019 三大公开医学视觉问答数据集。

**📈 对比分析**

与多种开源/闭源 MLLM 在零样本和 LoRA 微调下对比，InViC 在三大数据集均显著提升准确率（例如 Qwen3-VL-4B SLAKE 从 0.592 提升至 0.849）。

**⚠️ 局限性**

局限在于需要额外的瓶颈训练步骤、对 K 值的敏感性以及目前仅验证于三大数据集，缺乏对更复杂任务或多模态场景的通用性评估。

---

## 301. Novelty-Driven Target-Space Discovery in Automated Electron and Scanning Probe Microscopy

**arXiv ID:** 2603.16715 | [PDF](https://arxiv.org/pdf/2603.16715v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 302. When Stability Fails: Hidden Failure Modes Of LLMS in Data-Constrained Scientific Decision-Making

**arXiv ID:** 2603.15840 | [PDF](https://arxiv.org/pdf/2603.15840v1)

**作者:** Nazia Riasat `[一作]` `[通讯]` (North Dakota State University), Nazia Riasat (North Dakota State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估大型语言模型在固定差异表达表格下进行基因优先排序任务的行为，重点考察运行稳定性、与统计真值的一致性、提示词敏感性和输出有效性。

**💡 创新点**

提出四维行为评估框架，将稳定性、正确性、提示敏感性和输出有效性分别量化，并在受控实验中系统展示稳定性并不等同于正确性。

**🔧 技术方法**

使用ChatGPT、Google Gemini 3和Claude Opus 4.5等LLM，在固定DESeq2生成的统计参考上进行10次重复查询；采用Jaccard指数、重叠系数和无效基因计数等指标进行评估。

**📊 数据集**

采用一份固定的差异表达（DE）表格（由DESeq2分析得到），作为所有实验的唯一输入数据。

**📈 对比分析**

通过在相同输入、不同阈值、不同提示词（P7a/P7b）下执行10次查询，计算与DESeq2参考集的Jaccard和重叠系数；结果显示LLM在阈值放宽时会过度选择，提示词细微变化导致显著差异，Claude还会产生大量无效（幻觉）基因，说明高稳定性并不保证高正确性。

**⚠️ 局限性**

实验仅基于单一DE数据集和单一统计方法（DESeq2），未验证在多数据集、多统计范式或更复杂科学决策流程中的泛化能力。

---

## 303. VisBrowse-Bench: Benchmarking Visual-Native Search for Multimodal Browsing Agents

**arXiv ID:** 2603.16289 | [PDF](https://arxiv.org/pdf/2603.16289v1)

**作者:** Zhengbo Zhang `[一作]` (Chinese Academy of Sciences Institute of Automation), Ying Yan `[通讯]` (Ant Digital Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个由人工专家验证的 VisBrowse-Bench 169 问题集，评估多模态浏览代理在真实互联网环境中的视觉推理与搜索能力。

**💡 创新点**

提出了需要在搜索过程中主动获取并融合视觉与文本证据的多模态深度搜索任务，克服了现有基准对视觉推理和多模态信息整合的不足。

**🔧 技术方法**

采用五种工具的代理工作流（文本搜索、图像搜索、反向图像搜索、图像裁剪、网页访问）和大型多模态语言模型实现交互式多轮推理。

**📊 数据集**

使用人类专家从公开数据源构造的 169 个带图片的问答实例，涵盖七大类别，包含 178 张唯一图片。

**📈 对比分析**

通过对比闭源、开源和深度研究模型在直接回答、+文本搜索、+图像搜索三种设置下的准确率，最高模型 Claude‑4.6‑Opus 在 +IS 模式下仅达 47.6% 正确率，显示基准极具挑战性。

**⚠️ 局限性**

模型普遍依赖文本推理而忽视视觉工具，且对图像搜索的调用准确性差导致性能下降，表明当前多模态语言模型在视觉推理与工具使用上的局限。

---

## 304. MLLM-based Textual Explanations for Face Comparison

**arXiv ID:** 2603.16629 | [PDF](https://arxiv.org/pdf/2603.16629v1)

**作者:** Redwan Sony `[一作]` (Michigan State University), Ross Arun `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估多模态大语言模型在无约束人脸识别中生成的文本解释的可靠性，并提出基于似然比的框架来量化解释的证据强度。

**💡 创新点**

①引入似然比（LR）评估文本解释的证据强度；②采用多级提示策略研究辅助人脸识别系统信息对MLLM解释可靠性的影响；③在极端姿态与监控图像下对MLLM解释质量进行系统评估。

**🔧 技术方法**

使用多模态LLM（GPT‑4o、Gemini‑2.5）、文本嵌入+PCA、Gaussian Mixture Model、似然比（LR）评估、t‑SNE可视化、聚类分离指标及商业COTS人脸识别系统作为对照。

**📊 数据集**

训练集采用BUPT‑CBFace，测试集采用IJB‑S；同时对商业COTS FR系统进行基准对照。

**📈 对比分析**

通过比较不同提示方式（无分数、分数、分数+决策、真值）下的验证准确率、不确定率以及LR ROC 曲线，发现辅助FR信息能显著提升拒绝假对的准确率，但对真对仍受极端姿态限制；解释的LR分数与验证准确率不完全一致，说明解释质量与决策正确性不总是同步。

**⚠️ 局限性**

评估仅基于文本嵌入的分离，未能直接验证视觉可证性；解释仍易出现幻觉；在极端姿态和复杂场景下仍难以获得可靠解释；缺乏将文本属性与视觉特征直接对齐的机制。

---

## 305. Interact3D: Compositional 3D Generation of Interactive Objects

**arXiv ID:** 2603.16085 | [PDF](https://arxiv.org/pdf/2603.16085v1)

**作者:** Hui Shan `[一作]` (Zhejiang University), Xiangru Huang `[通讯]` (Westlake University)

**通讯引用:** 598 | [OpenAlex ID](https://openalex.org/A5100697564)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种训练自由的 generate-then-compose 框架 Interact3D，用于从用户给定 mesh 和文本/图像提示生成可交互、物理可行的 3D 场景。

**💡 创新点**

创新点在于将生成的 3D 场景作为空间先验，采用全局-局部注册、基于 SDF 的碰撞抑制优化，以及基于 VLM 的闭环语义修正，实现无监督的碰撞感知与物理一致性。

**🔧 技术方法**

利用 Nano Banana Pro 生成引导图像、TRELLIS2/Hunyuan3D 生成 3D 网格、PartField 进行粗分割、GeoTransformer + ICP 进行全局-局部配准、SDF 优化、VLM（如 Gemini 3 Pro）进行纠错。

**📊 数据集**

构建并公开了超过 8,000 对交互式 3D 对组合的自建数据集，并在此数据集及公开数据上进行实验。

**📈 对比分析**

与 Jigsaw、2BY2、PartField+RANSAC 等基线对比，使用 CLIP 相似度和表面/体积交互率评估，结果显示 Interact3D 在语义一致性上提升约 5%–10%，在碰撞率上显著低于基线（表面交互率从 0.6766e-3 降至 0.0027e-3）。

**⚠️ 局限性**

对细小或高度耦合的部件（如螺钉、紧耦合关节）仍难以准确建模，受限于 2D 引导图像的遮挡与 3D 空间推理不足。

---

## 306. CLRNet: Targetless Extrinsic Calibration for Camera, Lidar and 4D Radar Using Deep Learning

**arXiv ID:** 2603.15767 | [PDF](https://arxiv.org/pdf/2603.15767v1)

**作者:** Marcell Kegl `[一作]` (Hungarian Research Network Institute for Computer Science and Control), Dariu M. Gavrila `[通讯]` (TU Delft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 CLRNet，一种端到端的多模态深度学习网络，实现摄像头、激光雷达与 4D 雷达的目标无标定外参校准。

**💡 创新点**

创新地使用等距投影、共享特征空间、环路闭合损失及额外雷达通道，显著提升校准精度。

**🔧 技术方法**

结合 ResNet‑18 编码器、PWC‑Net 相关层、深度预测、equirectangular 投影及联合循环闭合损失的深度学习框架。

**📊 数据集**

在 View‑of‑Delft 与 Dual‑Radar 两个公开数据集上进行训练和评估。

**📈 对比分析**

与多种现有目标无标定方法和基准模型对比，CLRNet 在摄像头‑雷达误差上降幅 ≥50%，单帧模型在摄像头‑雷达中达到 7.8 cm/0.4°，多帧+迭代可达 0.9 cm/0.1°，远优于传统方法。

**⚠️ 局限性**

对跨域迁移仍存在性能下降，且依赖足够丰富的雷达点云密度与场景多样性，传统运动基校准在无激发运动场景下失败。

---

## 307. Self-Aware Markov Models for Discrete Reasoning

**arXiv ID:** 2603.16661 | [PDF](https://arxiv.org/pdf/2603.16661v1)

**作者:** Gregor Kornhardt `[一作]` (Technische Universität Berlin), Gabriele Steidl `[通讯]` (Technische Universität Berlin)

**通讯引用:** 3330 | [OpenAlex ID](https://openalex.org/A5008755474)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种自校正、难度感知的离散扩散模型，用于解决需要唯一答案的推理任务；

**💡 创新点**

创新点在于通过学习基于模型自身输出的马尔可夫转移核，使得在推理过程中可以对错误标记重新掩码并进行自我纠正，同时引入置信度头和自学习停止准则，实现推理步数的自适应分配；

**🔧 技术方法**

技术上采用离散流匹配的马尔可夫链框架，构建混合预测-提交核，加入置信度和停止时间两个轻量级预测头，并在模型生成的路径上进行on‑policy训练；

**📊 数据集**

使用的评测数据集包括 Sudoku‑Extreme、Kaggle Unfiltered Sudoku 以及 Countdown‑4（四数值推理）等；

**📈 对比分析**

与 DFM、Top‑K、ReMDM、GIDD、CTTT 等基线方法对比，Sudoku‑Extreme 上实现 95.2% 的有效率（Kaggle Unfiltered 近 100%），Countdown‑4 上达到 95.9%（最高 98.9%），且平均所需推理步骤大幅减少；

**⚠️ 局限性**

局限性在于该方法仅适用于答案唯一确定的推理任务，并非概率生成器，难以推广到需要多样性输出或非确定性解的场景。

---

## 308. Prose2Policy (P2P): A Practical LLM Pipeline for Translating Natural-Language Access Policies into Executable Rego

**arXiv ID:** 2603.15799 | [PDF](https://arxiv.org/pdf/2603.15799v1)

**作者:** Vatsal Gupta `[一作]` (Apple), Darshan Sreenivasamurthy `[通讯]` (Apple)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将自然语言访问控制策略自动翻译成可执行的 Rego 代码并提供完整的检测、提取、验证和测试流水线。

**💡 创新点**

首次提供模块化、可重复的 LLM 驱动流水线，并引入基于模式的 Schema 验证、lint、编译与自动测试机制，提升了政策生成的可靠性与可审计性。

**🔧 技术方法**

使用大语言模型（LLM）进行意图识别、组件提取、Rego 代码合成；结合提示工程、Schema 校验、Regal linter、OPA 编译与自动测试。

**📊 数据集**

在 ACRE 数据集上评估，使用 RAGent 输出的 485 条语句作为输入。

**📈 对比分析**

与传统基于规则或单独提取工具对比，Prose2Policy 的编译通过率 95.3%，正面测试通过率 82.2%（LLM 生成），负面测试 98.9%；相比基于规则的测试仅 62.1% 正面通过。

**⚠️ 局限性**

主要限制在于 LLM 生成测试用例的准确性不足，复杂多条款策略仍可能导致正面测试失败，以及模型推理的不确定性导致重复执行的差异。

---

## 309. Evaluating Black-Box Vulnerabilities with Wasserstein-Constrained Data Perturbations

**arXiv ID:** 2603.15867 | [PDF](https://arxiv.org/pdf/2603.15867v1)

**作者:** Adriana Laurindo Monteiro `[一作]` (Getulio Vargas Foundation), Jean-Michel Loubes `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出基于Wasserstein距离的分布扰动框架，对黑盒机器学习模型进行稳健性和可解释性评估。

**💡 创新点**

创新点在于将分布投影与约束优化结合，既能生成可解释的“压力测试”分布，又提供投影分布的闭式解和一致性理论。

**🔧 技术方法**

使用最优传输理论、Wasserstein投影、对偶形式、梯度下降求解以及分布约束下的凸优化。

**📊 数据集**

实验使用了美国人口普查Adult Income数据集和Boston Housing房价数据集。

**📈 对比分析**

通过对不同特征均值的扰动参数τ，观察模型预测比例、均值和方差的变化，展示特征影响和公平性改进；结果表明模型对年龄、教育水平等特征高度敏感，且在公平性评估中可出现“公平洗牌”。

**⚠️ 局限性**

局限性包括：投影后数据缺乏真实标签，导致无法评估误差；方法主要针对线性约束和无缩放的树模型；对高维、复杂分布的可扩展性待验证。

---

## 310. Grounding the Score: Explicit Visual Premise Verification for Reliable Vision-Language Process Reward Models

**arXiv ID:** 2603.16253 | [PDF](https://arxiv.org/pdf/2603.16253v1)

**作者:** Junxin Wang `[一作]` (Alibaba), Guanjun Jiang `[通讯]` (Alibaba)

**通讯引用:** 426 | [OpenAlex ID](https://openalex.org/A5004378463)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Explicit Visual Premise Verification (EVPV)，让多模态推理步骤先给出视觉前提清单，再与一次性提取的结构化视觉约束对齐，并基于视觉可靠性门控调整步骤奖励，以提升过程奖励模型的准确性与鲁棒性。

**💡 创新点**

创新点在于显式化视觉前提并通过结构化约束验证，解耦感知错误与逻辑判断，避免传统PRM在视觉误读时产生误判，并实现可解释的可靠性门控。

**🔧 技术方法**

使用视觉约束提取器（预训练模型微调+DPO）、步骤验证器（分类器）以及可靠性门控函数，并通过提示让策略输出视觉前提清单。

**📊 数据集**

在VisualProcessBench进行步骤级评估，并在LogicVista、MMMU、MathVerse-VO、MathVision、MathVista、WeMath等六个多模态推理基准上进行Best‑of‑N reranking实验。

**📈 对比分析**

与VisualPRM、TIM‑PRM、QWEN‑VL‑PRM等基线对比，EVPV在VisualProcessBench宏观F1提升至约69.6%（比最优基线提升约7-8%），并在Best‑of‑8 reranking中实现+8.8至+9.8个百分点的准确率提升，尤其在视觉密集型任务上效果最显著。

**⚠️ 局限性**

局限性包括对结构化约束提取的覆盖率与准确性的依赖；约束缺失或误判会导致门控失效；视觉前提清单的完整性与细粒度不足时可能无法充分校正步骤奖励，且单一全局可靠性评分可能无法捕捉步骤级差异。

---

## 311. Simplex-to-Euclidean Bijection for Conjugate and Calibrated Multiclass Gaussian Process

**arXiv ID:** 2603.16621 | [PDF](https://arxiv.org/pdf/2603.16621v1)

**作者:** Bernardo Williams `[一作]` (University of Helsinki), Marcelo Hartmann `[通讯]` (University of Helsinki)

**通讯引用:** 181 | [OpenAlex ID](https://openalex.org/A5041682146)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用 Aitchison 结构将多类别分类问题映射到欧氏空间的 Gaussian Process (GP) 回归模型，实现完全共轭推断。

**💡 创新点**

创新点在于：① 完全共轭，消除辅助变量和对数正态近似；② 通过 ILR 变换将 K 类问题降至 D=K‑1 维欧氏空间；③ 兼容稀疏 GP 技术，可在大数据上可扩展。

**🔧 技术方法**

使用的技术包括 Gaussian Process、Aitchison 几何与 ILR 变换、共轭高斯似然、基于 Helmert 矩阵的映射、稀疏变分 GP 与诱导点方法。

**📊 数据集**

实验数据集：UCI 公开数据（如 Iris、Glass、Wine 等）的小型数据集，7 个大型 UCI 数据集，以及合成数据。

**📈 对比分析**

与 Dirichlet‑based GP（GPD）、Logistic Softmax（LSM）、Bijective Softmax（BSM）以及 Sparse Variational GP Classification（SVGPC）进行对比。在精确（O(N³)）和稀疏设置下，本文方法在准确率、负对数似然（NLL）和期望校准误（ECE）上表现与 GPD 相当或略优，特别在校准性能上显著优于多数传统方法。

**⚠️ 局限性**

局限性：① 精确版仍需 O(N³) 计算，难以处理极大数据；② 对标签平滑参数 λ 与噪声 σ² 的选择依赖验证；③ 在类别极度重叠或 K 极大时，模型对噪声参数敏感，可能需更复杂的噪声建模。

---

## 312. Dual Stream Independence Decoupling for True Emotion Recognition under Masked Expressions

**arXiv ID:** 2603.16760 | [PDF](https://arxiv.org/pdf/2603.16760v1)

**作者:** Jinsheng Wei `[一作]` (Nanjing University of Posts and Telecommunications), Guanming Lu `[通讯]` (Nanjing University of Posts and Telecommunications)

**通讯引用:** 787 | [OpenAlex ID](https://openalex.org/A5065519164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在遮挡表情视频的顶点帧中识别真实情绪的方法。

**💡 创新点**

提出了基于顶点帧的识别范式和双流独立解耦框架，并利用Hilbert–Schmidt独立性损失实现特征解耦。

**🔧 技术方法**

采用Vision Transformer骨干网络，构建双分支适配器，并使用交叉熵损失与HSIC损失进行联合训练。

**📊 数据集**

使用中国科学院心理研究所构建的Masked Facial Expression Database (MFED)。

**📈 对比分析**

通过Leave‑One‑Subject‑Out交叉验证与ResNet18、ViT、CLIP比较，DSID在顶点帧下实现最高0.392准确率、0.370 F1分数。

**⚠️ 局限性**

主要局限在于顶点帧仍受到强伪装情绪干扰，对超参数α、β敏感，且无法完全消除伪装特征对真实情绪识别的影响。

---

## 313. Organisational accounts engaged in scholarly communication on Twitter: Patterns of presence, activity and engagement

**arXiv ID:** 2603.16637 | [PDF](https://arxiv.org/pdf/2603.16637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 314. When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making

**arXiv ID:** 2603.16673 | [PDF](https://arxiv.org/pdf/2603.16673v1)

**作者:** Jun Liu `[一作]` (Robotics Institute, Carnegie Mellon University), Dong Huang `[通讯]` (Robotics Institute, Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

为具身机器人设计了一个基于强化学习的资源感知推理控制框架 RARRL。

**💡 创新点**

创新点在于学习可调节的推理调用策略，决定何时推理、推理角色以及计算预算，以平衡任务成功率与计算延迟。

**🔧 技术方法**

采用 Proximal Policy Optimization（PPO）强化学习，结合 GPT‑4o‑mini 作为LLM推理模块，使用抽象任务流程进行训练。

**📊 数据集**

使用 ALFRED 仿真数据以及自定义的抽象任务流程作为数据集。

**📈 对比分析**

与无推理、全推理、固定频率、启发式以及约束PPO基线对比，RARRL 在保持任务成功率相近的同时，LLM 延迟、token 消耗降低约 60% 以上，且在实际 ALFRED 运行时实现了显著的时间与资源节省。

**⚠️ 局限性**

局限包括：依赖抽象环境训练，真实物理部署需处理感知噪声与执行不确定性；推理效果受底层执行/推理模块性能限制；对不同 LLM 的迁移性与鲁棒性仍需进一步验证。

---

## 315. Directivity Enhancement of Movable Antenna Arrays with Mutual Coupling

**arXiv ID:** 2603.16472 | [PDF](https://arxiv.org/pdf/2603.16472v1)

**作者:** Wei Xu `[一作]` (Zhejiang University), Rui Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 106358 | [OpenAlex ID](https://openalex.org/A5100422102)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了可移动天线阵列在存在相互耦合时的波束指向性，提出了一种基于贪婪搜索和梯度下降的低复杂度算法来优化天线位置，以最大化给定方向上的指向性。

**💡 创新点**

创新点在于首次系统性地利用相互耦合来提升可移动阵列的指向性，并提出了结合离散贪婪搜索与连续梯度优化的 GS‑GD 方法，既保留了全局搜索能力，又能高效收敛。

**🔧 技术方法**

采用了相互耦合模型、Rayleigh 商式指向性分析、Cholesky 正交化、贪婪搜索（GS）、梯度下降（GD）以及回溯线搜索等技术。

**📊 数据集**

使用仿真数据，设置 N=5、λ=0.3 m、最小间距 d_min=λ/10、不同可移动区域大小 d_max 等参数进行实验验证。

**📈 对比分析**

与传统 ULAH、全局枚举（ES）、单独 GS、单独 GD 进行对比。结果显示 GS‑GD 在绝大多数方向上接近全局最优，指向性提升 20%–50%，并且明显优于单纯 GS 或 GD。

**⚠️ 局限性**

局限性包括对大规模阵列的计算复杂度仍较高；算法在极端耦合或非线性可移动区域时可能陷入局部最优；实验仅基于仿真，缺乏真实测量验证。

---

## 316. Work Sharing and Offloading for Efficient Approximate Threshold-based Vector Join

**arXiv ID:** 2603.16360 | [PDF](https://arxiv.org/pdf/2603.16360v1)

**作者:** Kyoungmin Kim `[一作]`, Anastasia Ailamaki `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种统一的近似阈值向量连接框架，结合软工作共享、合并索引与自适应混合搜索，以提升效率和召回率。

**💡 创新点**

创新点在于：①软工作共享利用最邻近的out‑of‑range点重用；②合并查询与数据索引实现工作卸载，实现常数时间近邻查询；③针对OOD查询的自适应混合BestFS+BFS搜索，显著提升召回。

**🔧 技术方法**

采用图基近似最近邻索引（如NSG/HNSW）及其贪婪搜索+BFS；实现早停、MST排序、合并索引、混合搜索等技术。

**📊 数据集**

实验使用八个公开高维数据集（如SIFT、GloVe等）验证算法。

**📈 对比分析**

与现有工作共享/单查询方法（INLJ、WS、XJoin等）对比，软工作共享提升3.16×，合并索引提升32.6×，混合搜索提升召回43%；总体在效率‑召回曲线上显著优于对手。

**⚠️ 局限性**

局限性包括：仍受高维度距离计算瓶颈影响；合并索引略增内存占用；混合搜索需阈值判定且主要针对图索引，尚未验证在分布式/流式环境下的可扩展性。

---

## 317. KidsNanny: A Two-Stage Multimodal Content Moderation Pipeline Integrating Visual Classification, Object Detection, OCR, and Contextual Reasoning for Child Safety

**arXiv ID:** 2603.16181 | [PDF](https://arxiv.org/pdf/2603.16181v1)

**作者:** Viraj Panchal `[一作]` (Vartit Technology Inc), Meet Patel `[通讯]` (Vartit Technology Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了KidsNanny两阶段多模态内容审核系统，专门用于儿童安全；

**💡 创新点**

创新点在于将视觉分类与对象检测、OCR文本提取和文本推理分离为两阶段管线，并在Stage 2仅传递文本和对象标签给LLM，既提升文本检测效果，又大幅降低推理时延；

**🔧 技术方法**

使用Vision Transformer进行视觉分类、CNN实现对象检测、专用OCR引擎提取文本，再用7B参数文本LLM进行上下文推理；

**📊 数据集**

主要评测使用UnsafeBench的Sexual类别（1,054张图）以及PASS安全样本；

**📈 对比分析**

采用两种对齐评估模式：仅视觉模式（Stage 1）和完整多模态模式（Stage 1+2）。在仅视觉模式下Stage 1准确率达80.27%，F1 85.39%；完整模式下准确率81.40%，F1 86.16%，与ShieldGemma‑2和LlavaGuard相比，推理时延分别快≈9×和≈34×；在仅文本子集上，KidsNanny实现100%召回，75.8%精确率，显著优于VLM基准；

**⚠️ 局限性**

主要局限包括：评测由同一团队完成，缺乏第三方验证；仅评测Sexual类别，其他危险类别未覆盖；文本子集样本量小（44张），统计意义有限；模型细节及训练数据保密，难以复现；仅在RTX 4090上测试，跨硬件性能未知。

---

## 318. Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting

**arXiv ID:** 2603.16542 | [PDF](https://arxiv.org/pdf/2603.16542v1)

**作者:** Wanpeng Zhang `[一作]` (Peking University), Zongqing Lu `[通讯]` (BeingBeyond)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种仅利用后续观察、无奖励的样本加权方法 PTR，用于改善跨机器人异质演示的离线后训练。

**💡 创新点**

创新点在于将后果识别后验作为奖励无关的质量信号，并通过保守加权映射实现对跨体型数据的选择性转移与负迁移抑制。

**🔧 技术方法**

采用 Transformer VLA backbone、统一动作空间、EMA 目标编码器、对比式识别头、软因果 BeliefTokenizer、指数加权、裁剪与混合权重以及自适应温度控制等技术。

**📊 数据集**

在 LIBERO 与 RoboCasa 的仿真任务、12 个真实机器人任务（Unitree G1、PND Adam‑U、FR3）以及多种受噪声污染的数据集上进行评估。

**📈 对比分析**

与标准 SFT、SFT+Belief 以及多源通用化训练对比；在未污染数据上保持与 SFT 相当，且在对象、长序列、跨体型等场景提升 1–3%，在受噪声干扰时提升 5–10%。

**⚠️ 局限性**

局限在于需后续观测、依赖良好预训练表示，且仅是数据重加权机制，无法直接优化任务奖励。

---

## 319. Resilience Meets Autonomy: Governing Embodied AI in Critical Infrastructure

**arXiv ID:** 2603.15885 | [PDF](https://arxiv.org/pdf/2603.15885v1)

**作者:** Puneet Sharma `[一作]` (UiT Arctic University of Norway), Christer Henrik Pursiainen `[通讯]` (UiT Arctic University of Norway)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `51c0528b-f690-4182-ae60-bb5f046c276c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨并提出在关键基础设施中，如何通过边界化自治与混合治理架构提升具身人工智能的韧性；

**💡 创新点**

提出四种人机监督模式（全自动、HOTL、HITL、HIC），并将其映射到不同关键基础设施领域，强调自治与监督的可控分配；

**🔧 技术方法**

运用欧盟AI法案、ISO安全标准及危机管理理论，结合具身人工智能的技术原理（如SLAM、传感融合）进行理论建模与框架设计；

**📊 数据集**

未使用特定数据集，论文为概念性与框架性研究；

**📈 对比分析**

无实验或数值对比，主要通过文献综述与标准分析说明不同监督模式在各领域的适用性与优缺点；

**⚠️ 局限性**

局限在于缺乏实证验证、缺少可量化评估指标，且框架对不同组织文化与技术成熟度的适应性仍需进一步研究。

---

## 320. vAccSOL: Efficient and Transparent AI Vision Offloading for Mobile Robots

**arXiv ID:** 2603.16685 | [PDF](https://arxiv.org/pdf/2603.16685v1)

**作者:** Adam Zahir `[一作]` (University Carlos III of Madrid), Roberto Gonzalez `[通讯]` (NEC Laboratories Europe)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个统一框架，用于在移动机器人上高效、透明地执行 AI 视觉工作负载，并在真实测试床上验证了其可行性。

**💡 创新点**

创新点在于将神经网络编译器 SOL 与轻量级远程执行框架 vAccel 结合，实现硬件无关的推理编译与执行位置透明化，且可在机器人本地或边缘节点无缝切换。

**🔧 技术方法**

使用了 SOL 编译器生成无外部运行时依赖的共享库，以及 vAccel 的 GenOp 接口实现本地/边缘推理调度；实验平台包括 ROS2 中间件、Wi‑Fi 6 网络与单摄像头输入。

**📊 数据集**

采用了十二个公开模型（ResNet50、Swin‑T/S‑V2、R3D‑18、Swin3D、DeepLabV3‑ResNet50/101 等）及对应标准验证数据集，未引入新数据集。

**📈 对比分析**

与 PyTorch Inductor 编译器基线对比，结果显示在本地执行时 SOL 与 Torch 相当或略优；边缘卸载后，机器人端功耗可下降至 80%，边缘 CPU/GPU 功耗相对 Torch 降低 60%，帧率提升最高可达 24 倍。

**⚠️ 局限性**

局限性包括依赖稳定的 Wi‑Fi 连接；缺乏动态自适应卸载策略；对网络抖动、丢包以及不同硬件平台的进一步评估仍需补充。

---

## 321. PanguMotion: Continuous Driving Motion Forecasting with Pangu Transformers

**arXiv ID:** 2603.16196 | [PDF](https://arxiv.org/pdf/2603.16196v1)

**作者:** Quanhao Ren `[一作]` (Fudan University), Nan Song `[通讯]` (Fudan University)

**通讯引用:** 3988 | [OpenAlex ID](https://openalex.org/A5037102607)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

整合冻结的 Pangu-1B Transformer 块作为特征增强模块，改进连续驾驶场景的运动预测，并简化原始模型结构，去除行人轨迹流；

**💡 创新点**

首次将大语言模型（Pangu）作为特征增强器嵌入连续驾驶预测框架；同时证明高层 Transformer 层最适合迁移到运动预测；通过消除功能冗余提升单模式预测性能；

**🔧 技术方法**

Transformer（Pangu-1B 预训练块）、自注意力、交叉注意力、线性投影层、Ascend NPU 端到端训练与推理；

**📊 数据集**

Argoverse 2 数据集（使用 RealMotion 的数据重组策略构造连续序列）；

**📈 对比分析**

与 RealMotion 基线对比，PanguMotion 在 minADE_1 与 minFDE_1 上分别提升约1.58%和1.29%，在多模式指标上保持或略低；整体性能优于 LLaMA 版本；

**⚠️ 局限性**

计算资源需求高，Ascend NPU 下仍受限；多车交互建模简化；数据重组与真实连续驾驶存在差距；Pangu 预训练中文语料可能引入语言文化偏差。

---

## 322. Ciphertext-Policy ABE for $\mathsf{NC}^1$ Circuits with Constant-Size Ciphertexts from Succinct LWE

**arXiv ID:** 2603.16117 | [PDF](https://arxiv.org/pdf/2603.16117v1)

**作者:** Jiaqi Liu `[一作]` (Chern Institute of Mathematics and LPMC Nankai University), Fang-Wei Fu `[通讯]` (Chern Institute of Mathematics and LPMC Nankai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出一种基于格的简洁属性加密方案，支持任意布尔电路，实现密文大小与属性长度无关；

**💡 创新点**

利用（λ,σ)-succinct assumption与矩阵承诺构造，使密文与属性长度无关并显著减小公钥规模；

**🔧 技术方法**

采用格密码学的矩阵承诺、离散高斯分布、噪声抖动、剩余哈希lemma等技术；

**📊 数据集**

无，需要的数据集，纯理论构造；

**📈 对比分析**

与现有基于格的属性加密与广播加密方案比较，密文与公钥大小均显著缩小，安全性基于已知的格难题；性能与现有方案相当或更优；

**⚠️ 局限性**

仍依赖大格参数导致计算开销较高，对极大属性数仍存在公钥规模膨胀；适用性受限于电路大小与深度的限制。

---

## 323. High-Dimensional Gaussian Mean Estimation under Realizable Contamination

**arXiv ID:** 2603.16798 | [PDF](https://arxiv.org/pdf/2603.16798v1)

**作者:** Ilias Diakonikolas `[一作]` (University of Wisconsin-Madison), Thanasis Pittas `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究在可实现污染（realizable contamination）模型下，如何对具有单位协方差的高维高斯分布进行均值估计。

**💡 创新点**

提出了信息–计算之间的间隙：证明任何统计查询（SQ）算法要么需要指数级查询次数，要么需要指数小容差；并给出了近乎信息最优、样本时间权衡匹配该下界的算法。

**🔧 技术方法**

技术包括：1) 将问题转化为非高斯分量分析（NGCA）并利用刻画阶矩匹配的低阶多项式；2) 通过 Legendre 多项式校正构造满足污染约束的分布；3) 采用列表可解（list‑decodable）均值估计、锦标赛（tournament）选择、Hermite 张量与奇异值分解实现维度约简；4) 在约简子空间上进行蛮力求解。

**📊 数据集**

实验仅在合成的多维高斯样本上验证，未使用真实数据集。

**📈 对比分析**

与之前仅给出信息理论上界的粗略算法相比，新算法在样本量上与下界几乎匹配，且实现了多项式时间（仅指数出现在样本数而非维度）。

**⚠️ 局限性**

局限性：仅针对单位协方差的高斯分布；算法样本复杂度随污染率与误差参数指数增长；在更一般的子高斯或非高斯分布上，问题仍然是开放的。

---

## 324. Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration

**arXiv ID:** 2603.16570 | [PDF](https://arxiv.org/pdf/2603.16570v1)

**作者:** Amirhossein Kazerouni `[一作]` (University of Toronto), Alex Levinshtein `[通讯]` (AI Center–Toronto Samsung Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种两阶段全场景图像修复框架Face2Scene，先利用参考图像对人脸进行高质量修复，再从人脸的高低质量对中提取衰减信息，驱动扩散模型完成全图像恢复。

**💡 创新点**

创新点在于将人脸视作“降质先验”，通过参考基人脸修复获得衰减码（FaDeX），并将其映射为多尺度扩散条件令牌（MapNet），实现全场景恢复时的降质感知与人脸一致性。

**🔧 技术方法**

核心技术包括参考图像驱动的人脸修复（FaceMe等）、对比学习的FaDeX编码器、降质映射网络MapNet、多尺度条件扩散（SD‑Turbo），以及对比损失与GAN终结器。

**📊 数据集**

使用合成数据集InScene（基于CelebRef‑HQ与InfiniteYou生成的多身份场景图像）与真实收集的手机照片，结合CC12M等无标签图库进行训练与评估。

**📈 对比分析**

在合成与真实验证集上与S3Diff、DiffBIR、PASD、OSEDiff等最新方法对比，Face2Scene在PSNR、SSIM、LPIPS、DISTS、FID、MUSIQ、CLIP‑IQA、MANIQA等指标均取得最佳或次佳成绩，且在多尺度人脸与背景恢复上显著优于基线。

**⚠️ 局限性**

局限性包括假设全局统一降质（不适用于空间变异模糊或光照差异）、对文本和极小人脸的恢复效果有限，以及受扩散后端模型（如SD‑2.1）对高分辨率与细节处理的限制。

---

## 325. An Efficient Heterogeneous Co-Design for Fine-Tuning on a Single GPU

**arXiv ID:** 2603.16428 | [PDF](https://arxiv.org/pdf/2603.16428v1)

**作者:** Ruijia Yang `[一作]` (Hong Kong University of Science and Technology), Zeyi Wen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 15794 | [OpenAlex ID](https://openalex.org/A5100705849)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向单 GPU 的全参数 LLM 微调系统 SlideFormer，突破 VRAM 限制，实现大规模模型微调

**💡 创新点**

核心创新包括：轻量级异步引擎（Layer‑Sliding 结构）、预分配的 GPU 缓存队列与层级内存管理、GPUDirect Storage + NVMe 的多级 I/O、融合的 Triton 核心（如 LinearCrossEntropy）以及层级 Adam 优化器

**🔧 技术方法**

技术细节涵盖：CUDA Stream 与 CPU 线程异步协同、GPU‑CPU 直接数据搬运、GPUDirect Storage、NVMe 级联、梯度检查点、RoPE/RMSNorm/SwiGLU 的 Triton 加速、融合 Linear‑CrossEntropy 计算

**📊 数据集**

实验数据集使用合成数据以保证一致性；在 Llama‑3.1‑8B、Qwen‑2.5‑3B/72B、Mistral‑24B/123B 等模型上进行评测

**📈 对比分析**

与 ZeRO‑Offload、ZeRO‑Infinity、ColossalAI、LoHan 等基线对比，SlideFormer 在 RTX‑4090/AMD‑7900XT/ A100 等硬件上实现 1.4‑6.27× 吞吐提升，GPU/CPU 内存占用分别下降 50%+ 与 40% 左右，批量和模型规模分别提升 8× 与 6×，并保持 95%+ 的峰值利用率

**⚠️ 局限性**

局限性：仍需高容量 CPU 内存或 NVMe 存储；系统对超大模型的调优依赖于 GPU 计算能力；在极端内存/带宽瓶颈场景下，NVMe 访问仍会成为性能瓶颈

---

## 326. Visual Prompt Discovery via Semantic Exploration

**arXiv ID:** 2603.16250 | [PDF](https://arxiv.org/pdf/2603.16250v1)

**作者:** Jaechang Kim `[一作]` (Sony Group Corporation), Shingo Takamatsu `[通讯]` (Sony Group Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自动化的语义探索框架SEVEX，用于在大型视觉语言模型(LVLM)中发现针对具体任务的视觉提示，以提升模型的感知与推理能力。

**💡 创新点**

核心创新包括：① 在高层抽象想法空间进行搜索，避免低级代码噪声；② 采用新颖性引导的UCT（NUCT）算法实现高效多样化探索；③ 引入语义回传分析，将样本级失败诊断转化为可推广的洞察，指导后续想法生成。

**🔧 技术方法**

使用的技术包括：代理驱动的探索与实现、抽象想法空间、Novelty‑guided UCT、语义回传分析、程序化视觉工具库（如裁剪、绘图、深度估计等）、LLM（Gemini‑2.5‑flash）以及自动提示工程（APE）等。

**📊 数据集**

评估数据集：BlindTest（计数、圆圈字母、单色路径、重叠形状）和 BLINK（拼图、相对深度、空间推理、语义/视觉关联）。

**📈 对比分析**

与 Naive、SketchPad、SketchPad+APE 等基线在相同任务上进行对比。SEVEX 在 9 项任务中 7 项达到最佳，平均准确率 78.9%（高于 Naive 71.6% 和 SketchPad 64.6%），推理成本仅比 Naive 高 10.9%，而探索成本和推理效率显著优于 SketchPad+APE，且在稳定性和泛化方面表现更好。

**⚠️ 局限性**

局限性：① 发现的视觉提示对不同 LVLM 体系结构不具备可迁移性，需要为每个模型单独搜索；② 尽管比手工调试更高效，但探索仍需多次实验，消耗计算资源；③ 在高度抽象的想法空间中可能忽略某些低级实现细节导致的潜在性能提升。

---

## 327. SpikeCLR: Contrastive Self-Supervised Learning for Few-Shot Event-Based Vision using Spiking Neural Networks

**arXiv ID:** 2603.16338 | [PDF](https://arxiv.org/pdf/2603.16338v1)

**作者:** Maxime Vaillant `[一作]` (National Centre for Scientific Research), Benoit R. Cottereau `[通讯]` (National Centre for Scientific Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SpikeCLR框架，利用对比式自监督学习让SNN从无标签事件数据中学习视觉表征。

**💡 创新点**

创新点在于为事件数据设计专属空间、极性、时间增广组合，并在SNN中实现对比损失训练。

**🔧 技术方法**

采用SimCLR式对比损失、LIF神经元的Surrogate梯度训练、SEW-ResNet-18等SNN骨干。

**📊 数据集**

使用CIFAR10-DVS、N-Caltech101、N-MNIST、DVS-Gesture等事件视觉数据集。

**📈 对比分析**

在少量标签或半监督设置下，SpikeCLR在Linear Probe与Fine‑Tune上均显著优于从零训练的监督模型，几-shot 1-shot下提升约8‑10%。

**⚠️ 局限性**

限制在于需要大批量负样本导致显存占用高，且实验集中在分类任务，未验证在检测、深度等下游任务。

---

## 328. Constructing Weakly Terminating Interface Protocols

**arXiv ID:** 2603.15675 | [PDF](https://arxiv.org/pdf/2603.15675v1)

**作者:** Debjyoti Bera `[一作]` (TNO ESI), Tim A. C. Willemse `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了在异步交互系统中，利用标签化端口网和部分镜像关系来保证服务器与多个客户端之间的弱终止性。

**💡 创新点**

创新点在于放宽了传统镜像端口网对同构的严格要求，定义了可观测选择、菱形和循环属性，构造了部分镜像端口网并证明其弱终止性，并提出同步模式解决多客户端冲突。

**🔧 技术方法**

采用Petri网理论、标签化端口网形式化、结构性可达性分析、镜像映射和同步模式构造，并实现了在ComMA工具中的检测算法。

**📊 数据集**

主要为理论研究，没有使用具体数据集，验证工作集中在形式化模型和工具实现上。

**📈 对比分析**

通过形式化证明与ComMA工具的实验演示，展示了检测算法能够在设计时发现终止性缺陷，提升模型安全性；并未与其他方法在性能上做定量对比。

**⚠️ 局限性**

局限性包括：仍需手工构造或检查可观测选择、菱形和循环属性；多客户端情形需额外的同步模式；对更复杂的多种交互模式的适用性尚未完全验证。

---

## 329. What DINO saw: ALiBi positional encoding reduces positional bias in Vision Transformers

**arXiv ID:** 2603.16840 | [PDF](https://arxiv.org/pdf/2603.16840v1)

**作者:** Moritz Pawlowsky `[一作]` (Center for Materials Research), Ronan Docherty `[通讯]` (Imperial College London)

**通讯引用:** 2595 | [OpenAlex ID](https://openalex.org/A5017857371)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过分析并消除 DINOv2 的位置信息偏差，利用 ALiBi 相对位置编码重新训练模型，生成更均匀的特征，用于改进材料科学显微图像的可训练分割。

**💡 创新点**

创新点在于：①系统性证明自监督 ViT 模型普遍携带可线性解码的位置信息；②通过将原始 DINOv2 嵌入作为教师并引入 ALiBi，成功实现无偏差、语义保留的特征；③验证该方法在自然图像和超大尺度显微图像上的有效性。

**🔧 技术方法**

采用的技术包括 Vision Transformer、ALiBi 相对位置编码、线性探测、PCA、k‑means 聚类、XGBoost 分类器以及 CRF 后处理。

**📊 数据集**

使用的数据集包括 COCO‑Stuff（用于训练目标）、VOC 07/12、ADE20K（语义分割基准）以及多组 SEM/TEM 以及 DTD 纹理图像用于偏差分析和微观图像分割实验。

**📈 对比分析**

在 VOC、ADE20K 的线性探测 mIoU 上，ALiBi‑Dv2 与原始 DINOv2 相近或略优；在微观图像的可训练分割任务中，ALiBi‑Dv2 显著降低了水平/垂直/径向的偏差，提升了分割质量和鲁棒性。

**⚠️ 局限性**

局限性包括：仅对现有 DINOv2 检查点进行微调，未从头训练；对自监督方法产生位置信息的根本原因尚未解释；在某些数据集上仍可能出现残余偏差。

---

## 330. Tackling Over-smoothing on Hypergraphs: A Ricci Flow-guided Neural Diffusion Approach

**arXiv ID:** 2603.15696 | [PDF](https://arxiv.org/pdf/2603.15696v1)

**作者:** Mengyao Zhou `[一作]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Guiying Yan `[通讯]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一种基于Ricci流的超图神经扩散（RFHND）模型，以自适应方式调控超图消息传递，从而缓解过平滑问题。

**💡 创新点**

创新点在于将离散Ricci流理论引入超图学习，构建了基于几何曲率的连续动力学模型，并通过理论证明保证能量有界、收敛稳定。

**🔧 技术方法**

使用了离散Ricci流、偏微分方程建模、神经网络拟合边曲率、ODE求解器和显式欧拉积分等技术。

**📊 数据集**

使用了学术基准Cora、Citeseer、Pubmed、Cora‑CA、DBLP‑CA以及工业真实数据Zoo、NTU2012、ModelNet40、Walmart、Senate、House等超图数据集。

**📈 对比分析**

通过与20多种现有HGNN、HGNN、UniGCNII、ED‑HNN等方法在节点分类任务中对比，RFHND在所有数据集上均取得最优或次优排名，准确率提升约1‑3%。

**⚠️ 局限性**

局限性包括计算复杂度较高、需要手动调节ODE步长以及对曲率度量选择的敏感性。

---

## 331. Fanar 2.0: Arabic Generative AI Stack

**arXiv ID:** 2603.16397 | [PDF](https://arxiv.org/pdf/2603.16397v1)

**作者:** FANAR TEAM `[一作]`, Yifan Zhang `[通讯]` (Qatar Computing Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出 Fanar 2.0——卡塔尔自主研发的阿拉伯语生成 AI 平台，集成 27B 语言模型、双语安全过滤器、长语音 ASR、可读性恢复、图像生成与理解、翻译、工具调用与多模态推理等多项功能，并在仅 256 张 NVIDIA H100 GPU 的受限算力与约 0.5% 的阿拉伯语网络数据下，通过高质量数据优先、持续预训练与模型融合实现显著性能提升；

**💡 创新点**

创新点主要包括：① 受限算力下的高质量数据优先与多 recipe 预训练策略；② 27B 大模型与 32K 上下文窗口、原生推理链与结构化自检；③ 双语安全过滤器与文化对齐机制；④ 长语音 ASR 端到端模型与读写恢复层；⑤ 文化定制化图像生成与理解；⑥ 细粒度多模态工具调用与多代理推理框架；

**🔧 技术方法**

技术手段涵盖：持续预训练、模型融合、DPO、知识探测与自检、三阶段数据过滤、长序列 Transformer、CTC 端到端 ASR、风格恢复 Transformer、扩散 TTS、文化评测指标、A/B 评估及多层 orchestrator；

**📊 数据集**

使用数据集包括：约 120B 高质量阿拉伯/英语/代码 token（包含 FineWeb‑EDU、ArabicWeb‑EDU、机器翻译生成的平行语料）、公开多模态数据（Aura‑STT‑BenchLF、Aura‑TTS 语音、Oryx 图像）、安全与文化对齐评测集、公共基准集（MMMLU、ArabicMMLU、OALL、MMLU、PIQA、GSM8K、ARC、BLEU 等）以及人工标注与自检数据；

**📈 对比分析**

性能评估通过多任务基准比较，Fan 27B 在阿语世界知识（MMMLU/Ar）+9.1 分、阿语方言（Belebele）+3.5 分、英语知识（MMLU）+7.6 分、长语音 ASR WER 降至 16% 以内、TTS WER 1.42%，并在安全与文化对齐方面超过同规模或更大规模模型；与 Fanar 1.0 对比提升约 3‑5 分；

**⚠️ 局限性**

局限性包括：仍依赖人工过滤与标注，阿拉伯低资源挑战仍存在；长语音 ASR 在重叠/噪声条件下鲁棒性不足；工具调用与多模态推理尚未实现完整多步骤链路；模型规模 27B 仍低于更大模型在某些基准上的表现；文化对齐评测仍以人工为准，缺乏更系统的多模态与长文本评估。

---

## 332. A Comparative Analysis of Backbone Algorithms for Configurable Software Systems

**arXiv ID:** 2603.15833 | [PDF](https://arxiv.org/pdf/2603.15833v1)

**作者:** Luis Cambelo `[一作]` (Universidad Nacional de Educacion a Distancia), David Fernandez-Amoros `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文通过统一算法描述、实现 IPASIRBones 并在 2,371 个真实可变性模型公式上进行大规模实验，系统评估并比较了 5 种主流 backbone 算法的性能，给出了基于模型规模的算法选择准则。

**💡 创新点**

创新点在于：①首次在真实可变性模型公式上对比 SAT 社区的高级 chunking 算法与 SPL 工具采用的迭代算法；②提出基于公式规模的混合算法选择策略；③揭示了 chunk 大小 k 对性能的极端敏感性并评估了 CadiBack 的自适应 k 机制；④系统性验证了多种过滤启发式（如 RL、COV/WHIT/2LEN）的影响。

**🔧 技术方法**

主要技术包括：基于 IPASIR 接口的增量 SAT 调用、chunking 与 core‑based 背包算法、透明增量处理（TIP）、旋转文字过滤、以及自适应 k 选择策略。

**📊 数据集**

使用了 2,371 条公式（来自 BusyBox、Linux Kernel、Buildroot、EmbToolkit、Freetz 等 5 大系统，变量数从 100 到 186,059，子句数从 179 到 527,240），并对比了 21 种算法配置（包含 MiniBones、CadiBack、EDUCIBone、IPASIRBones 等）。

**📈 对比分析**

采用 21 个配置共 1,287,453 次执行（每个公式 3 次），对比了中位数运行时间，结果显示：对于 ≤1,000 变量的公式，算法 2/3（迭代+解集过滤）最快；对于 >1,000 变量的公式，算法 5（chunked core‑based）配合 CadiBack 的自适应 k 在实际性能上优于固定 k 或其他算法，平均可减少超过 50% 的耗时。

**⚠️ 局限性**

局限性包括：实验仅覆盖 Kconfig/UVL 形式的可变性模型，未验证其他建模语言的通用性；使用的 SAT 求解器仅为 MiniSat 2.2 与 CaDiCaL，未来求解器更新可能影响相对排名；k 的最优值高度依赖具体公式，当前无法通过简单特征预测，导致自适应机制仍有提升空间。

---

## 333. Robust Dynamic Object Detection in Cluttered Indoor Scenes via Learned Spatiotemporal Cues

**arXiv ID:** 2603.15826 | [PDF](https://arxiv.org/pdf/2603.15826v1)

**作者:** Juan Rached `[一作]` (Massachusetts Institute of Technology), Jonathan P. How `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 31980 | [OpenAlex ID](https://openalex.org/A5011665886)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

结合时空占用格运动分割与学习得到的鸟瞰视图动态网格，融合两条检测管线实现室内动态障碍物检测。

**💡 创新点**

提出STORM融合策略和无类别动态BEV网格，解决近距离与遮挡导致的误检与漏检。

**🔧 技术方法**

使用Dynablox式OGM分割、PointPillars+LSTM生成BEV动态网格、EKF轨迹跟踪及融合算法。

**📊 数据集**

基于Livox Mid-360 LiDAR+Vicon运动捕捉的室内实验数据集，包含人类与无人机两类动态障碍物。

**📈 对比分析**

与Dynablox、GridFusion等方法对比，在多障碍物环境下召回率提升28.67%、F1提升18.50%；在高速无人机场景同样优于对手。

**⚠️ 局限性**

仍受OGM稀疏点、薄壁面产生假阳性影响，GridNet仅使用三帧窗口导致在高速或无障碍环境下延迟增加。

---

## 334. Near-light Photometric Stereo with Symmetric Lights

**arXiv ID:** 2603.16404 | [PDF](https://arxiv.org/pdf/2603.16404v1)

**作者:** Lilika Makabe `[一作]` (Osaka University), Yasuyuki Matsushita `[通讯]` (Osaka University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于原点对称光源阵列的近光照相位立体方法，利用光源对称性与光衰减松弛将问题转化为单像素线性求解，得到闭式解。

**💡 创新点**

创新点在于：①利用多组对称光源产生新的线性约束；②在对光衰减从 1/d² 近似为 1/d 的松弛下，构造完整的线性系统；③通过 SVD 求解右零空间，得到尺度化距离向量，随后可直接恢复表面法向与深度，无需初始深度或完整光源标定。

**🔧 技术方法**

使用的技术包括：光衰减松弛、对称光源差分与求和约束、线性矩阵组装、奇异值分解（SVD）求解右零空间、基于 Apollonius 球面求解表面位置、闭式法向恢复；实验中还使用了相机内参标定、噪声模型、真实光源板与光学传感器。

**📊 数据集**

数据集：合成数据（Bunny 与 Crab，Lambertian 表面），以及真实场景数据（9 个光滑物体，使用 8 LED 轴对称光源板）。

**📈 对比分析**

与已知光源标定的 Calibrated 方法、无标定学习方法 FastNFPS 和 UniversalPS 进行对比。实验显示，该方法在法向角误差和深度相对误差上与 Calibrated 方法相近，且明显优于 FastNFPS、UniversalPS，尤其在深度不连续、光照偏差等情况表现更稳健；在光源对称性满足条件时，能够在不需初始化或标定的前提下实现全局最优。

**⚠️ 局限性**

局限性：①需要至少 3 对不同半径的对称光源（若半径相同的环形光源无法求解）；②光衰减松弛会在光源与物体距离过近时产生误差；③对阴影敏感，单像素估计噪声大；④方法假设 Lambertian 反射，对强非漫反射或高光区域效果受限；⑤对光源排列的几何约束较强，无法直接适用于任意光源布置。

---

## 335. WildDepth: A Multimodal Dataset for 3D Wildlife Perception and Depth Estimation

**arXiv ID:** 2603.16816 | [PDF](https://arxiv.org/pdf/2603.16816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 336. Hyperbolic Multimodal Generative Representation Learning for Generalized Zero-Shot Multimodal Information Extraction

**arXiv ID:** 2603.16259 | [PDF](https://arxiv.org/pdf/2603.16259v1)

**作者:** Baohang Zhou `[一作]` (Tiangong University), Ying Zhang `[通讯]` (Nankai University)

**通讯引用:** 16951 | [OpenAlex ID](https://openalex.org/A5100405620)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了用于泛化零样本多模态信息抽取（GZS-MIE）的超曲面多模态生成表征学习框架（HMGRL），通过在双曲空间构建样本与类别原型的层次语义关联，解决了传统欧氏空间模型在处理多模态和新类别时的局限；

**💡 创新点**

创新点包括：① 将超曲面变分信息瓶颈（HVIB）与超曲面多模态条件变分自编码器（HMCVAE）结合，实现对文本与视觉模态的对齐和生成；② 采用语义相似度分布对齐损失（semantic distribution alignment loss）来缓解见/未见类别的相似度分布差异；③ 在双曲空间使用Lorentz线性层捕捉层次语义；

**🔧 技术方法**

核心技术包括：超曲面（双曲）空间建模、Lorentz线性层、超曲面变分信息瓶颈（HVIB）、超曲面条件变分自编码器（HMCVAE）、对比学习、排序损失、交叉熵损失及语义分布对齐损失；

**📊 数据集**

使用WikiDiverse（包含文本-图像对的多模态命名实体类型抽取任务）和Twitter MRE数据集（多模态关系抽取任务），均将类别划分为见/未见子集；

**📈 对比分析**

与多种基线（文本与多模态模型、零样本模型、LLaVA等）进行对比。HMGRL在未见类别和整体指标上均显著优于第二名（MET任务未见类别F1提升≈12.6%，MRE任务整体准确率提升≈6.1%）；

**⚠️ 局限性**

局限性包括：① 对未见类别生成样本的质量依赖于原型嵌入，生成误差可能影响性能；② 需调参（如η、ζ）以平衡见/未见类别，过大/过小均会导致整体性能下降；③ 在大规模预训练模型（如LLaVA）上的零样本推理仍有优势，需进一步提升生成质量与鲁棒性。

---

## 337. Self-supervised Disentanglement of Disease Effects from Aging in 3D Medical Shapes

**arXiv ID:** 2603.15862 | [PDF](https://arxiv.org/pdf/2603.15862v1)

**作者:** Jakaria Rabbi `[一作]` (University of Alberta), Dana Cobzas `[通讯]` (MacEwan University)

**通讯引用:** 1626 | [OpenAlex ID](https://openalex.org/A5022382297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `4de8e9d8-757b-475f-9627-18a445e50202` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种两阶段自监督框架，利用隐式神经表示学习形状嵌入并通过聚类生成伪疾病标签，随后在变分自编码器中使用伪标签和年龄信息实现疾病与衰老因子的可控解耦。

**💡 创新点**

创新点在于：①将 INR 与高斯混合先验结合实现无监督疾病发现；②在第二阶段通过多目标解耦损失（协方差约束、软最近邻对比损失等）实现无标注情况下的自监督疾病解耦；③冻结 INR 解码器保持形状一致性，兼顾重建与可解释性。

**🔧 技术方法**

采用的技术包括：隐式神经表示（SDF），高斯混合先验（GMM）聚类，变分自编码器（VAE），软最近邻（SNNL）损失，协方差正则化，监督对比损失，Eikonal 正则化，残差 MLP 编码/解码器。

**📊 数据集**

实验数据集为 ADNI 组的海马形状（含 AD 与正常）以及 OAI 的远端股骨形状，用于跨疾病验证。

**📈 对比分析**

与 PCA、ICA、HLLE+ICA、β‑VAE、β‑TCVAE、DIP‑VAE 等无监督/半监督方法对比，本文在 SAP、相关性、疾病准确率、年龄回归 RMSE 方面均表现最优，重建误差与其他 VAE 方法相近，伪标签在低标注场景下显著提升性能。

**⚠️ 局限性**

局限性：①重建误差因解耦约束略高；②仅针对两类疾病（AD、OA）和两因素（疾病、年龄）进行验证，未知在更多疾病或更复杂形状上的泛化能力；③需对 GMM 组件数与解耦维度做手工设定，可能不适用于所有数据集。

---

## 338. ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars

**arXiv ID:** 2603.16447 | [PDF](https://arxiv.org/pdf/2603.16447v1)

**作者:** Kaiwen Song `[一作]` (University of Science and Technology of China), Juyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6021 | [OpenAlex ID](https://openalex.org/A5101904821)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种 ProgressiveAvatars，利用面片局部绑定的3D高斯分布构建层次化、可渐进加载与渲染的3D头部数字人模型；通过屏幕空间梯度驱动的自适应隐式细分实现细节自适应增量，并支持连续按重要性排序的数据流传输。

**💡 创新点**

创新点包括：① 统一资产模型替代离散LOD，允许任意子集的高斯可立即渲染；② 采用面片局部坐标绑定的高斯保持动画一致性；③ 通过屏幕空间梯度自适应细分，仅在高频区域增加细节；④ 结合重要性排序实现无跳跃的渐进渲染；⑤ 多级监督与阶段性训练提升低分辨率快速可用性。

**🔧 技术方法**

核心技术：3D Gaussian Splatting、FLAME头部模板、面片局部高斯绑定、屏幕空间梯度自适应细分、重要性打分排序、逐级监督训练、增量加载与渲染。

**📊 数据集**

使用 NeRSemble 多视角视频数据集（16相机、校准参数），与 GaussianAvatars、PointAvatar 以及 GaussianAvatars+LightGaussian 进行对比实验。

**📈 对比分析**

方法在 5% 带宽下已能生成可用头像，100% 时与 GaussianAvatars 等最先进方法相当（PSNR≈31.5/25.9，SSIM≈0.929/0.908，LPIPS≈0.068/0.080）。在不同传输比例下逐步提升质量，显著降低所需高斯数量（≤43.4 MB）且帧率保持在 250‑270 FPS（RTX 4090）。与离散压缩管线相比，省去多级存储并实现连续质量调节。

**⚠️ 局限性**

局限性：目前仅针对头部模型，尚未验证对全身或其他3D资产的适用性；依赖 FLAME 模型，需具备相同拓扑；在多用户大规模 VR 场景下高斯数量可能逼近 GPU 光栅化瓶颈；缺乏对动态遮挡和遮挡处理的详细讨论。

---

## 339. Via Negativa for AI Alignment: Why Negative Constraints Are Structurally Superior to Positive Preferences

**arXiv ID:** 2603.16417 | [PDF](https://arxiv.org/pdf/2603.16417v1)

**作者:** Quan Cheng `[一作]` (Tsinghua University), Quan Cheng `[通讯]` (Tsinghua University)

**通讯引用:** 55008 | [OpenAlex ID](https://openalex.org/A5090403310)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出正偏好与负约束的结构不对称理论，解释负反馈为何能与正偏好 RLHF 并驾齐驱并消除 sycophancy

**💡 创新点**

将 Popper 的证伪逻辑、Taleb 的 via negativa 等哲学观点与 LLM 对齐现象相结合，统一负反馈方法（NSR、D2O、Constitutional AI 等）的实证结果

**🔧 技术方法**

理论框架与结构分析，无新模型或算法实现

**📊 数据集**

未使用新的数据集，引用已有研究中的数据和实验（如 MATH、AIME、Constitutional AI 等）

**📈 对比分析**

通过对已有实验结果的解释来佐证，说明负反馈方法在数学推理、安全性等指标上可匹配或超过标准 RLHF，但未进行新的对比实验

**⚠️ 局限性**

缺乏直接实证验证，理论仍需实验检验；只针对离散约束性对齐，未覆盖正向目标（如创造性、帮助性）的学习

---

## 340. Are Large Language Models Truly Smarter Than Humans?

**arXiv ID:** 2603.16197 | [PDF](https://arxiv.org/pdf/2603.16197v1)

**作者:** Eshwar Reddy M `[一作]` (Health Vectors), Sourav Karmakar `[通讯]` (Intuit India)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对六大前沿LLM在MMLU基准上的污染进行了三种互补实验评估

**💡 创新点**

首次将检索、重述与行为掩码三种方法整合，构建公开可复现的污染审计流程

**🔧 技术方法**

使用Tavily网页检索、语义重述生成、TS-Guessing掩码恢复等技术

**📊 数据集**

MMLU测试集、公开API以及网页检索结果

**📈 对比分析**

与传统准确率对比，发现STEM领域污染率最高，模型准确率平均下降7pp，72.5%问题触发记忆信号

**⚠️ 局限性**

实验受限于公开搜索引擎覆盖度、对非公开训练语料估计不足，以及仅评估单一多选题格式

---

## 341. Survey of Various Fuzzy and Uncertain Decision-Making Methods

**arXiv ID:** 2603.15709 | [PDF](https://arxiv.org/pdf/2603.15709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 342. From Natural Language to Executable Option Strategies via Large Language Models

**arXiv ID:** 2603.16434 | [PDF](https://arxiv.org/pdf/2603.16434v1)

**作者:** Haochen Luo `[一作]` (City University of Hong Kong), Chen Liu `[通讯]` (City University of Hong Kong)

**通讯引用:** 10258 | [OpenAlex ID](https://openalex.org/A5100322126)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Option Query Language（OQL）和神经符号管道，将自然语言交易意图转化为可执行的期权策略。

**💡 创新点**

将 LLM 用作语义解析器产生结构化中间语言，并通过确定性执行引擎保证逻辑一致性；同时创建首个自然语言到期权策略检索基准。

**🔧 技术方法**

基于大型语言模型（如 GPT-4.1、DeepSeek-Coder 等）的语义解析、OQL 语法约束、向量化过滤和笛卡尔积组合的确定性编译。

**📊 数据集**

构造了涵盖 SPY、NVDA、AAPL、GOOG、TSLA 2025 期权链的 200 条多样化自然语言意图的基准数据集。

**📈 对比分析**

与三类基线（FFLG、PCG、Text‑to‑SQL）对比，OQL 在查询有效率、语义准确率和策略回测盈亏方面均显著优于基线，Win Rate 达 60%+、ROC 高于 30%。

**⚠️ 局限性**

仅支持预定义策略模板，未考虑现有持仓影响和真实交易成本，且对极端市场情形的鲁棒性有限。

---

## 343. The Internet of Physical AI Agents: Interoperability, Longevity, and the Cost of Getting It Wrong

**arXiv ID:** 2603.15900 | [PDF](https://arxiv.org/pdf/2603.15900v1)

**作者:** Roberto Morabito `[一作]` (EURECOM), Mallik Tatipamula `[通讯]` (Ericsson)

**通讯引用:** 1426 | [OpenAlex ID](https://openalex.org/A5110110991)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了物理 AI 代理（Physical AI Agents）互联网的整体架构蓝图，回顾了 IoT 的经验教训，并给出了设计原则与实现建议。

**💡 创新点**

将 AI 代理视为 Internet 的新原语，强调身份、语义互操作、安全与可进化等层次，避免早期固化和碎片化，并将治理与可观测性嵌入核心。

**🔧 技术方法**

主要技术涵盖边缘 AI、生成式 AI、ISAC（集成感知与通信）、低功耗硬件与元材料、5G/6G 通信、分布式治理与标准化框架。

**📊 数据集**

未使用具体数据集，文章为概念与体系结构说明性论述。

**📈 对比分析**

本工作无实验或性能比较，采用理论分析与案例研究（森林火灾、医疗闭环、工业自适应、城市移动）说明潜在效果。

**⚠️ 局限性**

主要局限：缺乏可验证的实现细节和量化评估；对标准化与治理时间窗口的依赖较大；现实部署中可能面临硬件可靠性与法规适配挑战。

---

## 344. Visual Distraction Undermines Moral Reasoning in Vision-Language Models

**arXiv ID:** 2603.16445 | [PDF](https://arxiv.org/pdf/2603.16445v1)

**作者:** Xinyi Yang `[一作]` (Peking University), Yixin Zhu `[通讯]` (Peking University)

**通讯引用:** 4078 | [OpenAlex ID](https://openalex.org/A5051255725)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可生成的多模态道德评估基准（mds），用于系统性检验视觉语言模型的道德推理。

**💡 创新点**

创新点在于在五大道德基础上构建可控生成管道，实现对概念变量和角色变量的正交操作，并设计三模态（文本、字幕、图像）诊断协议。

**🔧 技术方法**

采用多模态生成引擎、Moral Foundations Theory、深度学习视觉编码器、逻辑一致性校验、GPT‑4、Gemini‑4.1‑mini 等技术。

**📊 数据集**

使用自生成的84,240个控制样本，覆盖 Quantity、Single Feature、Interaction 三大子集，构成多模态数据集。

**📈 对比分析**

通过在三模态下评估多款 SOTA VLM（LLaVA、Qwen3、LLaMA、GPT‑4o‑mini、Gemini‑2.5‑flash），发现视觉输入显著削弱效用敏感性、降低反制度约束、放大自利行为，并导致社会价值等级消失；部分大模型（Gemini‑2.5‑flash、Qwen3‑VL‑32B）表现相对更好，但整体仍低于文本模式。

**⚠️ 局限性**

局限在于场景为合成沙盒图像，缺乏真实世界复杂性；生成过程依赖模板与 GPT 重写，可能带来偏差；评估仅覆盖有限的道德维度与模型。

---

## 345. Low-complexity tuning of pinching-antenna systems for integrated sensing and communication

**arXiv ID:** 2603.15844 | [PDF](https://arxiv.org/pdf/2603.15844v1)

**作者:** Saba Asaad `[一作]` (University of Toronto), Ali Bereyhi `[通讯]` (University of Toronto)

**通讯引用:** 325 | [OpenAlex ID](https://openalex.org/A5061064331)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种低复杂度的 PASS（pinching antenna system）辅助 ISAC（integrated sensing and communication）设计，利用 bi‑partitioning 技术在一次迭代中同时调节发射和接收端的压痕位置，以实现高速动态重构。

**💡 创新点**

创新点在于：①提出 bi‑partitioning 方案，将压痕元素分成通信侧和感知侧两部分，并用几何方式求解最优分配，从而把 O(T) 的求解简化为 O(1)；②在上下行链路分别设计了轻量级的多目标优化和 BCD 近似算法，显著扩大了 sensing‑communication rate region；③通过几何约束实现了对目标与用户路径损耗的动态补偿。

**🔧 技术方法**

技术方法包括：PASS 架构、bi‑partitioning（压痕分区）技巧、加权和标量化（scalarization）、多目标优化、低复杂度几何求解、块坐标下降（BCD）近似求解、SIC（successive interference cancellation）以及基于仿真的性能评估。

**📊 数据集**

使用的并非公开数据集，而是基于统一分布的仿真场景：用户与目标随机均匀分布在一个中心对称的矩形区域内（边长 Dx、Dy），并在不同 Dx、N（压痕元素数量）及权重 α 下进行参数扫描。

**📈 对比分析**

与传统固定相位阵列（全线性、λ/2 间距、每个天线有相位移）基线进行对比。PASS‑ISAC 在所有权重配置下均获得更高的加权速率，并在更大服务区域（Dx 较大）时保持近乎不变的速率，而基线随区域扩大速率急剧下降；在上下行链路中，PASS‑ISAC 的 rate region 明显扩展，覆盖传统阵列的整个区域。

**⚠️ 局限性**

局限性包括：①仅考虑理想的 LOS 环境，未对波导失真、硬件非理想性及电磁耦合等实际问题做详细建模；②上行链路的多目标优化为 NP‑hard，需用 BCD 近似，收敛不保证全局最优；③假设用户/目标已知位置且完美 SIC，实际系统中目标跟踪误差及信号干扰可能削弱性能。

---

## 346. ManiTwin: Scaling Data-Generation-Ready Digital Object Dataset to 100K

**arXiv ID:** 2603.16866 | [PDF](https://arxiv.org/pdf/2603.16866v1)

**作者:** Kaixuan Wang `[一作]` (University of Hong Kong), Ping Luo `[通讯]` (University of Hong Kong)

**通讯引用:** 54097 | [OpenAlex ID](https://openalex.org/A5100752686)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过自动化管道将单张图像转化为可在仿真中使用的数字双胞胎，并构建了包含10万个对象的 ManiTwin-100K 数据集。

**💡 创新点**

创新点在于一次性实现对象几何生成、物理属性估计、功能点与抓取点标注，并通过 VLM 与物理仿真实现全流程自动化与可验证性，提供大规模、可直接用于机器人学习的高质量资产。

**🔧 技术方法**

使用 CLAY 3D 生成模型、VLM（视觉‑语言模型）进行质量门控与注释、FPS 采样、GraspGen 抓取生成、SAPIEN 物理仿真、人工复核等技术。

**📊 数据集**

输入图像来源于电商商品照片与文本生成图像，经过管道生成的 100K 资产构成 ManiTwin-100K 数据集。

**📈 对比分析**

在 3D 生成质量上通过 CLIP/ULIP 隐空间相似度评估，显示图像条件生成的质量显著高于文本条件；在注释质量上自动验证与人工评测均达到 90% 以上准确率；生成的抓取姿态与轨迹在仿真中成功率超过 76%，共计 5M 采集抓取姿势、10M 轨迹。

**⚠️ 局限性**

仅覆盖刚体可抓取物体，未处理关节物体与柔性物体；物理属性由 VLM 推断，缺乏真实传感器校准；未来需扩展至关节与可变形物体并进行实测校正。

---

## 347. Persistent Device Identity for Network Access Control in the Era of MAC Address Randomization: A RADIUS-Based Framework

**arXiv ID:** 2603.16745 | [PDF](https://arxiv.org/pdf/2603.16745v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 348. ClaimFlow: Tracing the Evolution of Scientific Claims in NLP

**arXiv ID:** 2603.16073 | [PDF](https://arxiv.org/pdf/2603.16073v1)

**作者:** Aniket Pramanick `[一作]` (Technische Universität Darmstadt), Iryna Gurevych `[通讯]` (Technische Universität Darmstadt)

**通讯引用:** 25904 | [OpenAlex ID](https://openalex.org/A5027450194)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于科学主张的NLP论文数据集，定义并评估了“主张关系分类”任务，并用该模型在约1.3万篇论文中构建主张传播网络，揭示主张在NLP领域的演化规律。

**💡 创新点**

创新点在于：①首次以主张为粒度系统标注跨文献的支持、扩展、限定、驳斥和引用关系；②提出主张关系分类任务并提供基线；③在大规模语料上自动化构建主张传播图，实现跨年代的演化分析。

**🔧 技术方法**

技术包括：基于Transformer的编码器模型（BERT、RoBERTa、SciBERT）进行监督微调；大语言模型（GPT‑4o、GPT‑3.5‑turbo、LLaMA‑3‑70B‑Instruct、Mixtral‑8x7B‑Instruct）在零/少量提示下进行推理；以及两阶段自动化主张识别与关系推断的管道。

**📊 数据集**

数据集为自建的ACL Anthology 304篇论文标注集（1084主张、832关系），以及利用该基准在约1.3万篇ACL论文上自动构建的主张图；实验数据通过论文级划分的70/15/15训练/验证/测试集进行评估。

**📈 对比分析**

在基线实验中，RoBERTa微调获得宏F1≈0.78，SciBERT略优；在零/少量提示下，GPT‑4o在少量提示下表现最佳，超过其他LLM但仍低于微调模型；模型错误主要集中在区分扩展与限定、以及隐式支持/驳斥的判定上。

**⚠️ 局限性**

局限性包括：仅覆盖ACL Event会议论文，未涵盖AI会议、工作坊和预印本；标注范围仅限摘要、引言、结论，未覆盖全文主张；自动化主张识别与关系预测存在误差，影响大规模分析的准确性；且结果主要适用于NLP领域，跨学科推广需进一步验证。

---

## 349. Arabic Morphosyntactic Tagging and Dependency Parsing with Large Language Models

**arXiv ID:** 2603.16718 | [PDF](https://arxiv.org/pdf/2603.16718v1)

**作者:** Mohamed Adel `[一作]` (New York University), Nizar Habash `[通讯]` (New York University)

**通讯引用:** 11612 | [OpenAlex ID](https://openalex.org/A5084517393)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了指令调优的大型语言模型（LLM）在标准阿拉伯语中的形态句法标注和有标签依存句法解析的能力，并在零样本和检索式上下文学习两种提示设置下进行实验。

**💡 创新点**

创新点在于统一考察LLM对形态句法和句法结构的生成性能，探究提示设计与检索式示例选择对结构化输出的影响，并对开源与专有模型在阿拉伯语中的表现进行系统对比。

**🔧 技术方法**

采用指令调优LLM（Llama4、Qwen3、Gemini3、GPT5.2）结合零样本提示和基于chrF++与语义相似度的检索式上下文学习；使用约束解码评估完整的14维形态特征包和依存树。

**📊 数据集**

使用PATB（标准阿拉伯语新闻语料）进行形态句法标注，使用CAMeLTB（基于CATiB的多体裁依存树库）进行句法解析。

**📈 对比分析**

通过与监督基线（CAMeL Tools、CamelParser）以及各LLM在零样本和最佳ICL设置下的对比评估；结果显示开源LLM在零样本下差距显著，但检索式ICL可提升60+分的All Tags并接近监督基线；专有模型在特征级标注已与基线持平，在依存解析上能与CamelParser竞争，尤其在多根句子场景表现更佳。

**⚠️ 局限性**

仅限标准阿拉伯语，未覆盖方言；使用特定树库和标注体系，可能不覆盖所有语言结构；评估高度依赖提示与检索策略，结果易受噪声影响；开源与专有模型对比受访问与成本差异限制；tokenization误差对结果影响显著。

---

## 350. ModTrack: Sensor-Agnostic Multi-View Tracking via Identity-Informed PHD Filtering with Covariance Propagation

**arXiv ID:** 2603.15812 | [PDF](https://arxiv.org/pdf/2603.15812v1)

**作者:** Aditya Iyer `[一作]` (Brown University), Nora Ayanian `[通讯]` (Brown University)

**通讯引用:** 2494 | [OpenAlex ID](https://openalex.org/A5002752114)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `e0540dec-d77f-42db-94ae-d039248f6393` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ModTrack，一个模块化的多视角多目标跟踪系统，将神经网络仅限于检测和特征提取，其余融合、关联和跟踪通过闭式解析方法完成。

**💡 创新点**

创新点在于：1）将学习模块限定在感知阶段，后续使用精度加权的多视角聚类和身份增强的GM‑PHD滤波器实现跨模态、传感器无关的跟踪；2）在追踪过程中闭环地将身份先验反馈给关联，显著提升身份保持；3）完整的Jacobian不确定性传播实现可追踪的不确定性管理。

**🔧 技术方法**

采用的技术包括：BEV投影与Jacobian不确定性传播、χ²图聚类、精度加权多视角融合、身份信息增强的GM‑PHD滤波器、HMM运动模式、基于Mahalanobis距离的闭环身份先验以及回环反馈机制。

**📊 数据集**

使用的数据集包括 WildTrack（7摄像头行人）、MultiviewX（6摄像头合成）、RadarScenes（4雷达）以及相应的基准评估工具。

**📈 对比分析**

通过与多种端到端学习方法（如EarlyBird、MVTrajecter）和传统RFS方法（如MV‑GLMB‑OC、KSP‑DO）对比，ModTrack在 WildTrack 上实现 IDF1 95.5、MOTA 91.4、MOTP 87.2，超过传统模块化方法 21 点，并在 MultiviewX 与 RadarScenes 上保持同一追踪核心，表现与最先进端到端方法相近。

**⚠️ 局限性**

局限性包括：对前端检测与深度估计的误差敏感；在高密度场景下聚类门限可能导致误匹配；不支持动态摄像机安装；需要手动调节若干超参数。

---

## 351. Algorithmic Trading Strategy Development and Optimisation

**arXiv ID:** 2603.15848 | [PDF](https://arxiv.org/pdf/2603.15848v1)

**作者:** Owen Nyo Wei Yuan `[一作]` (Singapore Institute of Technology), Ryan Tan Jun Wei `[通讯]` (Singapore Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在S&P 500 2000-2024期间，结合财报转录文本，构建并优化了多因子算法交易策略。

**💡 创新点**

创新点在于将长期趋势过滤、交叉行情排名、ATR动态止损以及FinBERT情绪门控结合为多层信号框架。

**🔧 技术方法**

使用的技术包括Python Pandas向量化运算、FinBERT情绪分析、技术指标计算、向量化滚动统计、缓存排名和动态止损。

**📊 数据集**

使用的数据集为两千万条S&P 500日价格数据和相应的财报转录文本，分为训练、验证和隐藏测试集。

**📈 对比分析**

通过与基线策略在开发、验证和测试集上对比，总收益提升至约46%（开发）/189%（验证），夏普比率从1.19/0.41提升至1.69/2.04，最大回撤大幅下降。

**⚠️ 局限性**

局限包括缺少成交量、波动率和市场广度因子，情绪处理仅用分类标签，缺乏情感强度和主题分析，且高维向量化仍需大量计算。

---

## 352. Agentic AI for SAGIN Resource Management_Semantic Awareness, Orchestration, and Optimization

**arXiv ID:** 2603.16458 | [PDF](https://arxiv.org/pdf/2603.16458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 353. Surrogate-Assisted Genetic Programming with Rank-Based Phenotypic Characterisation for Dynamic Multi-Mode Project Scheduling

**arXiv ID:** 2603.16286 | [PDF](https://arxiv.org/pdf/2603.16286v1)

**作者:** Yuan Tian `[一作]` (Victoria University of Wellington), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**通讯引用:** 31894 | [OpenAlex ID](https://openalex.org/A5100400258)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于排名的表型表征方案，并将其集成到 surrogate-assisted genetic programming（SKGGP）中，用于进化动态多模式资源受限项目调度（DMRCPSP）的启发式规则。

**💡 创新点**

创新点在于设计了适用于 DMRCPSP 的决策情境型表型表征，将活动排序和组选择的排名信息编码为向量，从而首次实现了 surrogate-assisted GP 在该问题中的高效应用。

**🔧 技术方法**

采用了遗传程序化双树表示、最近邻 surrogate 模型、Manhattan 距离、k×|P| 代际子代生成与基于 surrogate 的精英选择等技术。

**📊 数据集**

使用维多利亚大学 Wellington 项目调度框架生成的 200 活动、3 模式、12 资源类型实例，评估三种前置强度（0.75/0.5/0.25）对应的 0.75/R12、0.5/R12、0.25/R12 场景。

**📈 对比分析**

通过 30 次独立运行、平均 makespan 与 Wilcoxon 符号秩检验对比基线 KGGP，结果显示 SKGGP‑2/4 在所有场景均显著优于 KGGP，并且在相同或更少评估次数内即可达到同等质量。

**⚠️ 局限性**

局限性在于 surrogate 的精度随子代倍率增大而下降，导致在大规模子代时误判率升高；目前仅使用最近邻模型和有限数据库，缺乏更精准的 surrogate 与更丰富的表型特征。

---

## 354. Homogeneous and Heterogeneous Consistency progressive Re-ranking for Visible-Infrared Person Re-identification

**arXiv ID:** 2603.16165 | [PDF](https://arxiv.org/pdf/2603.16165v1)

**作者:** Yiming Wang `[一作]` `[通讯]`, Yiming Wang

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于ResNet的跨模态行人重识别基线，并提出双阶段一致性重排方法HHCR；

**💡 创新点**

创新点在于同时考虑跨模态（异质）与同模态（同质）的一致性重排，利用GCN实现局部邻域信息传播；

**🔧 技术方法**

使用ResNet+BN提取特征，Triplet+Cross‑Entropy训练，随后采用GCN+局部查询扩展实现异质与同质一致性重排，最终通过加权融合得到最终相似度矩阵；

**📊 数据集**

在SYSU‑MM01、LLCM和RegDB三个公开跨模态数据集上进行评估；

**📈 对比分析**

与多种现有重排和跨模态重识别方法对比，HHCR在Rank‑1和mAP上均突破SOTA，例如SYSU‑MM01 All‑Search 最高 Rank‑1 90.42%/mAP 93.31%；

**⚠️ 局限性**

局限性包括对超参数（如λ、k值）敏感，计算复杂度较高，且在极端光照或极少样本情况下表现尚待进一步验证；

---

## 355. Grant, Verify, Revoke: A User-Centric Pattern for Blockchain Compliance

**arXiv ID:** 2603.15721 | [PDF](https://arxiv.org/pdf/2603.15721v1)

**作者:** Supriya Khadka `[一作]` (George Mason University), Sanchari Das `[通讯]` (George Mason University)

**通讯引用:** 4247 | [OpenAlex ID](https://openalex.org/A5003726306)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一个选择性披露框架并实现了ZK‑Compliance原型，允许用户在链上通过浏览器端零知识证明（ZKP）证明属性，采用Grant‑Verify‑Revoke生命周期；

**💡 创新点**

将资格验证与身份披露解耦，赋予用户动态可撤销的授权权限，并引入用户可直接调用的Kill Switch，彻底消除中心化中介；

**🔧 技术方法**

使用zk‑SNARK（Circom、SnarkJS）在浏览器内生成证明，配合以太坊智能合约（VerifierContract、AccessRegistry）实现链上验证，并通过浏览器本地存储的身份金库存放高熵盐；

**📊 数据集**

主要使用用户自身提供的身份属性（如出生日期）进行测试，未使用公开数据集；

**📈 对比分析**

通过在标准消费级硬件上测量交互延迟，客户端证明生成时间低于200 ms；在以太坊主网验证成本约15 美元，迁移至Layer 2后成本降至<0.5 美元，显示出良好的性能和经济可行性；

**⚠️ 局限性**

依赖自我证明导致可信度受限；证明与具体身份绑定不严格，存在重放风险；系统初始化由开发者手动完成，缺乏去中心化治理；用户认知负荷和撤销操作的易用性未通过实证验证；

---

## 356. ViT-AdaLA: Adapting Vision Transformers with Linear Attention

**arXiv ID:** 2603.16063 | [PDF](https://arxiv.org/pdf/2603.16063v1)

**作者:** Yifan Li `[一作]` (Michigan State University), Trung Bui `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ViT-AdaLA三阶段框架，将预训练的 Vision Foundation Model (VFM) 知识迁移到线性注意力 Vision Transformers，解决长序列视觉处理的算力与显存瓶颈；

**💡 创新点**

创新点在于通过注意力对齐、特征对齐和监督微调的逐步对齐策略，使线性注意力 ViT 能继承软max ViT 的先验知识，避免误差累积且不需大规模预训练；

**🔧 技术方法**

使用 vanilla linear attention（ELU 激活）配合 MSE 损失对齐注意力与特征，采用 AdamW 优化器和线性学习率衰减，框架兼容多种线性注意力实现；

**📊 数据集**

利用 COCO（注意力对齐）、ImageNet-22K（特征对齐预训练）以及 ImageNet-1K、ADE20K、Cityscapes 等数据集进行分类与分割评估；

**📈 对比分析**

与 softmax ViT、Hedgehog、LoLCATS、Monarch 等基线在分类 Top‑1 准确率、分割 mIoU、推理速度和显存等指标对比，ViT‑AdaLA 维持 1% 以内精度的同时实现约 2 倍速度提升和 50%+ 显存节省；

**⚠️ 局限性**

局限在于对不同 VFM 的泛化仍需进一步验证，极大分辨率或极低资源环境下的效果尚待评估，同时对教师模型质量的依赖仍是潜在瓶颈。

---

## 357. DreamPlan: Efficient Reinforcement Fine-Tuning of Vision-Language Planners via Video World Models

**arXiv ID:** 2603.16860 | [PDF](https://arxiv.org/pdf/2603.16860v1)

**作者:** Emily Yue-Ting Jia `[一作]` (USC Physical Superintelligence Lab), Yue Wang `[通讯]` (USC Physical Superintelligence Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视频世界模型的离线强化学习框架，在虚拟回放中微调视觉语言规划器以实现可变形物体操纵。

**💡 创新点**

创新点在于利用零射击VLM收集的子最优交互数据训练动作条件视频世界模型，再通过Best‑of‑K + ORPO在模型“想象”中完成强化学习，显著降低真实交互成本。

**🔧 技术方法**

采用视频扩散模型（CogVideoX‑5B）+ ControlNet 进行动作条件化；使用 ORPO 进行偏好式强化学习；通过渲染机器人臂轨迹与 SAM2 等工具实现视觉提示。

**📊 数据集**

数据集来源为在双臂 Franka FR3 系统上使用零射击 VLM 自动收集的 2056 条交互轨迹（包含衣物、绳索、软玩具等三类任务）。

**📈 对比分析**

与零射击基线相比，RL 微调后任务成功率提升 15%–40%，平均得分从 0.35 提升至 0.60；相对于采样验证基线，推理时间缩短至约 1 秒，性能保持领先。

**⚠️ 局限性**

局限性包括视频模型推理开销仍较大；数据规模有限，可能不易推广到更复杂或更高分辨率的场景；对真实物理噪声与动态的泛化仍需进一步验证。

---

## 358. Elastic Sketch under Random Stationary Streams: Limiting Behavior and Near-Optimal Configuration

**arXiv ID:** 2603.16786 | [PDF](https://arxiv.org/pdf/2603.16786v1)

**作者:** Younes Ben Mazziane `[一作]` (University of Avignon), Othmane Marfoq `[通讯]` (Meta)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究 Elastic Sketch 在随机静态流模型下的长期行为，并给出闭式期望计数误差公式，提供内存-精度权衡和参数调优方法。

**💡 创新点**

通过离散时间马尔科夫链分析重击块的永久选举与无穷切换两种状态，推导出最佳驱逐阈值只需在有限候选集内搜索；同时给出统一的误差上界与高概率上界。

**🔧 技术方法**

马尔科夫链理论、随机游走、极限定理、闭式概率推导、Monte Carlo 采样、离散时间随机过程分析。

**📊 数据集**

使用 Zipf 分布的合成流（α∈{0.8,1.0,1.2}，n=10^4，τ=5×10^5）进行数值验证。

**📈 对比分析**

与传统 Count‑Min Sketch 及 Elastic Sketch 的不同参数配置进行比较；理论预测与模拟结果高度吻合，表明所提取的 λ 与 m_1 能显著降低平均相对误差，提升内存利用效率。

**⚠️ 局限性**

仅适用于独立同分布、理想哈希函数的流；对非均匀、时变或大规模多样化流的鲁棒性尚未证明；以及求解最优参数仍需基于蒙特卡洛采样，计算开销不为零。

---

## 359. Adaptive Theory of Mind for LLM-based Multi-Agent Coordination

**arXiv ID:** 2603.16264 | [PDF](https://arxiv.org/pdf/2603.16264v1)

**作者:** Chunjiang Mu `[一作]` (Northwestern Polytechnical University), Shuyue Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 429 | [OpenAlex ID](https://openalex.org/A5052387391)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种自适应的心智理论（A-ToM）代理，旨在解决多代理协作任务中由于心智理论顺序不匹配导致的协调问题。

**💡 创新点**

创新点在于首次提出了自适应心智理论代理（A-ToM），该代理能够实时估计合作伙伴的心智理论顺序，并根据此调整其行为以实现协调。

**🔧 技术方法**

使用了在线学习算法来解决心智理论顺序对齐问题，并通过大语言模型（LLM）构建了具有固定心智理论顺序的代理和A-ToM代理。

**📊 数据集**

在四个多代理协调任务上进行了实证评估，包括重复矩阵游戏、两个网格导航任务和Overcooked任务。

**📈 对比分析**

与固定心智理论顺序的代理进行比较时，A-ToM代理在所有任务中表现出强大的协调能力，尤其在与不同类型的合作伙伴互动时，显示出显著的性能优势。

**⚠️ 局限性**

限制在于A-ToM代理的有效性可能受到任务的最优动作空间大小和代理理性程度的影响，且在某些情况下，心智理论对齐的重要性可能会降低。

---

## 360. Spectral Property-Driven Data Augmentation for Hyperspectral Single-Source Domain Generalization

**arXiv ID:** 2603.16662 | [PDF](https://arxiv.org/pdf/2603.16662v1)

**作者:** Taiqin Chen `[一作]` (Harbin Institute of Technology), Yongbing Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 7266 | [OpenAlex ID](https://openalex.org/A5101653272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于光谱属性驱动的数据增强方法（SPDDA），以解决单源遥感高光谱图像分类中的多样性与真实性权衡问题，并在增强样本上训练分类器以提高对未知目标域的泛化能力。

**💡 创新点**

创新点包括：①光谱多样性模块（SDM）通过通道掩码和通道自适应光谱混合器（CASM）模拟不同设备的光谱通道数和相邻通道混叠；②空间-光谱协同优化机制（SSCOM）联合空间保真约束和光谱连续自约束，并通过自适应权重动态平衡两者；③整合上述技术后在单源域上生成既多样又逼真的扩展域样本。

**🔧 技术方法**

核心技术包括：通道自注意力提取语义信息、基于相似度的高斯权重混合器、空间保真约束（MSE、SSIM、梯度损失）、光谱连续自约束（奇偶通道MSE/SSIM）、自适应λ调节、ResBlock编码网络以及LeNet分类器（带通道自适应卷积）。

**📊 数据集**

使用了三个跨场景高光谱数据集：Houston（Houston13→Houston18）、Pavia（PaviaU→PaviaC）和HyRank（Dioni→Loukia），每个数据集包含两种场景并提供多类别标签。

**📈 对比分析**

在与SDENet、FDGNet、S2ECNet、S2AMSNet（高光谱专用）以及ProRandConv、ABA、StyDeSty（RGB专用）等先进方法对比时，SPDDA在三个数据集的整体准确率（OA）、加权F1和Kappa指标上均取得最佳或次佳成绩，尤其在Houston上相较最优方法提升约+4.6% OA，+0.01 Kappa；此外在PSNR、SAM均衡性指标上也表现出更优的空间真实性与光谱多样性。

**⚠️ 局限性**

局限性在于：①λ自适应调节仍需经验阈值，可能在不同任务或更复杂场景中表现不稳定；②方法主要关注光谱通道数与相邻通道混叠，未充分考虑其他传感器噪声或光谱非线性效应；③在多源域（多样本）环境下效果尚未验证，可能需要进一步扩展。

---

## 361. Safety Case Patterns for VLA-based driving systems: Insights from SimLingo

**arXiv ID:** 2603.16013 | [PDF](https://arxiv.org/pdf/2603.16013v1)

**作者:** Gerhard Yu `[一作]` (York University), Alvine Boaye Belle `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了RAISE方法，用以系统构建基于视觉-语言-动作模型的自动驾驶系统的安全案例。

**💡 创新点**

创新点在于扩展HARA以包含安全事件，提出拒绝指令与接受有效指令的两套安全案例模式（RI、AAI），并提供构建安全案例的算法。

**🔧 技术方法**

采用HARA、GSN（目标结构化符号）模式、VLA模型评估、算法实现等技术。

**📊 数据集**

使用SimLingo（基于CARLA仿真环境）作为案例数据集，未使用公开大规模真实驾驶数据。

**📈 对比分析**

主要通过案例分析验证方法，没有与传统方法进行性能对比，结果显示可构建完整、可追溯的安全案例。

**⚠️ 局限性**

局限在于仅对单一系统SimLingo进行验证，缺乏多系统、真实场景的泛化验证，且依赖专家判断可能存在主观偏差。

---

## 362. Informationally Compressive Anonymization: Non-Degrading Sensitive Input Protection for Privacy-Preserving Supervised Machine Learning

**arXiv ID:** 2603.15842 | [PDF](https://arxiv.org/pdf/2603.15842v1)

**作者:** Jeremy J Samuelson `[一作]` `[通讯]` (Integrated Quantum Technologies), Jeremy J Samuelson (Integrated Quantum Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出信息压缩匿名化（ICA）与VEIL三层可信架构，实现不需要噪声注入或加密、但能在训练与推理阶段保持高预测效能的隐私保护机器学习；

**💡 创新点**

核心创新在于将多目标监督卷积残差自编码器（SCRAE）与信息压缩相结合，构造结构上不可逆的低维表示，同时通过Graph‑Laplacian与InfoNCE等连续目标适配实现分类与回归的统一；

**🔧 技术方法**

采用多目标监督自编码器、Center loss、Graph Laplacian loss、InfoNCE、三层信任边界VEIL以及拓扑与信息论证明，保证编码不可逆；

**📊 数据集**

实验使用MNIST（分类）、E2006‑tfidf（回归）以及真实房价数据集（攻击评估），展示了模型效果与攻击难度；

**📈 对比分析**

与传统自编码器对比，SCRAE在MNIST分类上准确率提升至99.25%（+6.80%），在回归中R²提升至0.6329（+0.0678）；在DP/HE对比中保持几乎无性能损失，且重构、属性与成员推断实验均表现为无显著成功率；

**⚠️ 局限性**

局限性包括：需要在源环境内可信部署编码器；Graph Laplacian计算在大批量时内存开销高；对高度相关属性的推断仍可能泄露信息；模型效果高度依赖任务对齐的监督目标，非完全通用的隐私保护方案。

---

## 363. Feed-forward Gaussian Registration for Head Avatar Creation and Editing

**arXiv ID:** 2603.15811 | [PDF](https://arxiv.org/pdf/2603.15811v1)

**作者:** Malte Prinzler `[一作]` (ETH Zürich), Timo Bolkart `[通讯]` (Google)

**通讯引用:** 4169 | [OpenAlex ID](https://openalex.org/A5025958423)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种利用多视角校准图像快速（0.5秒/帧）预测密集语义对应的高质量高斯斑点纹理的方法，并以此构建可编辑的头部头像。

**💡 创新点**

创新点包括：①注册引导注意力机制（每个UV令牌仅关注对应图像区域）显著提升推理速度和生成质量；②通过高斯斑点纹理实现跨主体、跨表情的密集语义对应，开启快速表情迁移、语义编辑与头像插值等应用；③通过直接预测纹理而非传统两阶段优化，头像创建时间从45小时压缩到4.6小时（10×加速）。

**🔧 技术方法**

技术手段包括：Transformer架构（交替使用注册引导注意力和分组注意力）、UV与图像令牌化、Coarse Mesh注册（TEMPEH+Sapiens特征）、高斯斑点纹理投影、PCA降维的GEM头像蒸馏、以及多任务损失（光度、几何、正则化）。

**📊 数据集**

主要使用Ava-256（256名主体，多视角、360°覆盖）和NeRSemble v2（425名主体，±50°水平、±15°垂直）数据集进行训练与评估，且在Ava-256上获得注册网格监督。

**📈 对比分析**

与GPAvatar、FastAvatar、LAM、Avat3r、FaceLift等方法在Ava-256与NeRSemble的新人视角合成任务中，LPIPS、PSNR、SSIM、L1/L2等指标均优于基线，尤其在极端表情与细节恢复上表现突出；在头像重建与重演任务中，创建时间比GEM快10倍，图像质量也更高。

**⚠️ 局限性**

局限性包括：①极端表情迁移时可能出现身份泄漏；②由高斯位置纹理生成的网格偶尔会自相交；③生成的头像仅覆盖训练表情，缺乏眼动跟踪；④当前对训练数据的注册网格依赖，完全无监督的跨数据集训练仍待研究。

---

## 364. Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents

**arXiv ID:** 2603.15658 | [PDF](https://arxiv.org/pdf/2603.15658v1)

**作者:** Madhava Gaikwad `[一作]` `[通讯]`, Madhava Gaikwad

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究将多存储检索视为路由问题，评估不同路由策略对问答性能和上下文效率的影响。

**💡 创新点**

创新点在于把存储选择建模为路由问题，引入覆盖率、精确匹配和浪费指标，并提出成本敏感的决策框架来权衡答案质量与检索成本。

**🔧 技术方法**

采用规则基、混合启发式和Oracle路由策略，利用GPT‑3.5‑turbo和GPT‑4o‑mini在合成标签与真实问答上进行实验，并通过token计数评估检索成本。

**📊 数据集**

使用LoCoMo与LongMemEval的查询分类生成1k合成查询及150道问答测试集，构造了基于查询类型的合成路由标签。

**📈 对比分析**

与统一检索、固定子集、混合启发式等做对比，Oracle路由在短/长上下文中均能提升≈5‑10%准确率，同时将上下文token减少62%；混合启发式覆盖率94%但准确率仅70%，统一检索则最慢且最不准。

**⚠️ 局限性**

局限性包括：路由标签基于语义规则非人工标注；启发式路由与Oracle仍有16%性能差距；未结合内部检索排名与多模型评估；仅使用全存储检索，未探讨top‑k检索与路由交互。

---

## 365. FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework for Quantitative Investment

**arXiv ID:** 2603.16365 | [PDF](https://arxiv.org/pdf/2603.16365v1)

**作者:** Qinhong Lin `[一作]` (Beijing University of Posts and Telecommunications), Yu Li `[通讯]` (Beijing Value Simplex Technology Co. Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个基于程序级别的 Alpha 因子挖掘框架 FactorEngine（FE），能够生成可执行、可审计且性能稳定的因子；

**💡 创新点**

创新点包括：1）将因子视为 Turing‑complete 代码，突破符号表达式的表达限制；2）宏微共进化架构，将 LLM 指导的逻辑演化与 Bayesian 超参搜索分离；3）知识注入的自举模块，利用多代理闭环从财报中提取、验证并生成可执行因子；4）多岛迁移与经验链（CoE）提升搜索多样性和效率；

**🔧 技术方法**

技术栈包括大语言模型（Gemini‑2.5‑Pro/GPT‑4o）、树结构搜索与 UCT 选择、Bayesian 参数优化（TPE/GP）、Polars 加速计算、并行化评估、闭环多代理抽取–验证–代码生成；

**📊 数据集**

数据集为中国沪深300/500 市场 OHLCV 价格与财报数据，训练集 2008‑2014，验证 2015‑2016，测试 2017‑2024；

**📈 对比分析**

与 LGBM、LSTM、Transformer、GPlearn、Alpha158、TRA、AlphaAgent、RD‑Agent 等基线比较。FE 在 400 次迭代下，在 CSI300/CSI500 上的 IC、ICIR、Rank‑IC、Rank‑ICIR、AR、IR、SR 等指标均优于基线，且因子多样性和稳定性（IC 下降缓慢）更好；

**⚠️ 局限性**

局限性包括：1）依赖 LLM 训练数据，可能引入信息泄漏或过拟合；2）计算资源仍较高，尤其是多岛并行与 Bayesian 搜索；3）目前仅处理 OHLCV 价格和财报文本，未涵盖其他数据模态；4）缺乏对交易成本、滑点等实盘约束的深入评估。

---

## 366. Physics-integrated neural differentiable modeling for immersed boundary systems

**arXiv ID:** 2603.16277 | [PDF](https://arxiv.org/pdf/2603.16277v1)

**作者:** Chenglin Li `[一作]` (Shanghai Jiao Tong University), Yanfei Zhang `[通讯]` (Shanghai Ship and Shipping Research Institute Co., Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理集成的可微分神经网络，用于长时程预测吸附边界流场，能够在粗网格上实现稳定、快速的自回归推理。

**💡 创新点**

创新点包括将压力投影步骤与多重直接强迫IB模块结构嵌入网络，利用子迭代解耦数值稳定性与时间步长，以及用卷积残差网络学习压强修正，完全避免昂贵的Poisson求解。

**🔧 技术方法**

技术核心为卷积残差网络（ConvResNet）、可微分的离散PDE算子、子迭代分步更新、以及可学习的压强校正模块；训练采用单步监督，避免BPTT。

**📊 数据集**

使用Re=100的经典基准数据集：静止圆柱和旋转振荡圆柱的高分辨率CFD模拟，随后下采样得到低分辨率训练集；同时构造了对应的数值基线。

**📈 对比分析**

与纯数据驱动模型、物理损失约束模型以及粗网格CFD基线对比，模型在空间和时间上均表现出更低的相对误差、长时程稳定性，推理速度约为高分辨率CFD的200倍、粗网格CFD的20倍。

**⚠️ 局限性**

局限性包括：在外推情形下误差仍会逐步累积；相对纯数据模型仍更快，且模型对高频细节的捕捉仍受低分辨率网格限制；需要先行高精度CFD数据作为训练来源。

---

## 367. Game-Theory-Assisted Reinforcement Learning for Border Defense: Early Termination based on Analytical Solutions

**arXiv ID:** 2603.15907 | [PDF](https://arxiv.org/pdf/2603.15907v1)

**作者:** Goutam Das `[一作]` (Purdue University), Daigo Shishika `[通讯]` (George Mason University)

**通讯引用:** 757 | [OpenAlex ID](https://openalex.org/A5057044271)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在边界防御游戏中，作者提出一种混合框架，将差分博弈理论中的Apollonius圆分析结果与多智能体强化学习相结合，以实现仅在探测到攻击者时立即结束训练回合并给出纳什均衡奖励，从而把RL的计算焦点放在搜索阶段。

**💡 创新点**

创新点在于将游戏理论的解析解嵌入RL训练循环，通过在探测时计算纳什均衡收益来代替后续的追捕学习，既保证了追捕阶段的最优性，又显著提升了样本效率和搜索策略质量。

**🔧 技术方法**

采用Apollonius圆求解追捕均衡、凸优化求交集、差分博弈分析、基于MAPPO的多智能体PPO训练以及BenchMARL与VMAS的并行仿真框架。

**📊 数据集**

实验使用自建的二维边界防御仿真环境，攻击者和防御者的初始位置在预定义区域内均匀随机生成，全部数据均来自该模拟器。

**📈 对比分析**

通过与传统端到端RL（即完整搜索+追捕训练）的对比，衡量平均奖励、收敛速度、探测率和纳什收益分布，结果显示GT辅助方法平均奖励提升约10–20%，收敛更快，探测率和空间配置质量均优于基线。

**⚠️ 局限性**

局限性包括：对多防御者时需要求解凸优化，计算量随防御者数量增加而增长；仅针对单一攻击者，未考虑多攻击者同时出现的情形；探测后的完美信息假设可能不适用于更复杂的感知不完全场景。

---

## 368. 360° Image Perception with MLLMs: A Comprehensive Benchmark and a Training-Free Method

**arXiv ID:** 2603.16179 | [PDF](https://arxiv.org/pdf/2603.16179v1)

**作者:** Huyen T. T. Tran `[一作]` (Tohoku University), Takayuki Okatani `[通讯]` (RIKEN AIP)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了360Bench高分辨率全景图像视觉问答基准，并提出了Free360训练无关的场景图推理框架来提升MLLM在360°图像上的推理性能

**💡 创新点**

创新点在于（1）设计七个细粒度子任务以全面评估全景图像感知；（2）提出基于场景图的分步推理流程，融合360°特定操作（实体中心旋转、视图映射等）；（3）通过融合CMP与ERP两种投影，实现对投影失真与空间推理的互补优势

**🔧 技术方法**

使用多模态大型语言模型（如GPT‑4o、Gemini、Qwen2.5‑VL等）作为基线，采用场景图生成（四步）和结构化文本序列化；对图像进行CMP与ERP转换、球面旋转、实体裁剪、视图节点构建等；评估指标为准确率与推理时延

**📊 数据集**

使用自收集的7K分辨率360°图像，共1,532个样本，包含7个子任务；对比人类实验（86.3%准确率）与13个MLLM及其增强方法的表现；引入公开的VQA360、OmniVQA等基准作为对比参考

**📈 对比分析**

通过与人类、不同投影格式以及现有增强方法（SEAL、DC^2、ZoomEye、Omni-CoT等）的系统评测，Free360在基准模型上提升了多达22.9%的子任务精度，整体准确率提升至45.3%，相较于原始Qwen2.5‑VL的58.1%（CMP）提升约7.3%；推理时延从2.1秒提升至22.5秒，仍低于人类平均28.9秒

**⚠️ 局限性**

局限性在于（1）仍无法达到人类水平，整体准确率仅46%；（2）推理时延较高，尤其对大规模模型；（3）仅针对单帧全景图像，未扩展至视频；（4）依赖手工标注的问答与实体框，扩展性受限

---

## 369. SQL-ASTRA: Alleviating Sparse Feedback in Agentic SQL via Column-Set Matching and Trajectory Aggregation

**arXiv ID:** 2603.16161 | [PDF](https://arxiv.org/pdf/2603.16161v1)

**作者:** Long Li `[一作]` (Griffith University), Chao Qu `[通讯]` (Fudan University)

**通讯引用:** 26346 | [OpenAlex ID](https://openalex.org/A5088679792)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Agentic SQL，构建多轮交互框架并解决 Text-to-SQL 的信用分配和奖励稀疏问题。

**💡 创新点**

创新点在于双层奖励机制——聚合轨迹奖励 ATR 与列集匹配奖励 CSMR，并利用 Lyapunov 稳定性证明 ATR 具有能量耗散、无周期、单调收敛性质。

**🔧 技术方法**

采用 Lyapunov 引导的非对称转移矩阵奖励设计、GRPO 策略优化、工具调用与数据库交互、量化奖励与优势归一化等技术。

**📊 数据集**

使用 BIRD、Spider、Spider 2.0 三个基准数据集，并在 Qwen2.5‑7B‑Instruct 与 OmniSQL‑7B 两大预训练模型上进行评估。

**📈 对比分析**

与单轮 0/1 奖励 GRPO、现有 SOTA Arctic‑Text2SQL‑R1‑7B、SQL‑R1、Reasoning‑SQL‑7B 等方法对比，平均提升约 5–8% 以上，Spider‑2.0 上从 15% 提升至 17.7%。

**⚠️ 局限性**

局限性包括计算开销与延迟显著增加、有限的交互步长可能限制对极其复杂任务的解决，以及奖励设计仍依赖若干启发式超参数。

---

## 370. More Rounds, More Noise: Why Multi-Turn Review Fails to Improve Cross-Context Verification

**arXiv ID:** 2603.16244 | [PDF](https://arxiv.org/pdf/2603.16244v1)

**作者:** Song Tae-Eun `[一作]` `[通讯]` (Daejeon Jungang Cheonggua Co., Ltd.), Song Tae-Eun (Daejeon Jungang Cheonggua Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在保持上下文分离的前提下，动态跨上下文审查（D‑CCR）对大语言模型（LLM）验证性能的影响，并与单轮 CCR 进行对比；

**💡 创新点**

提出并实证证明：在上下文分离的条件下，多轮交互并不提升性能，揭示了“假阳性压力”和“审查目标漂移”两种机制，并确立“单轮最佳”原理；

**🔧 技术方法**

使用 Claude Opus 4.6 进行独立会话调用，生成问题/答案交互，采用自定义匹配算法（行距 + 关键词 + 模糊匹配）评估发现；

**📊 数据集**

基于 30 个人工构造的代码、技术文档和演示脚本，共 150 个注入错误（5 类错误、3 级严重程度）进行实验；

**📈 对比分析**

通过成对 t‑检验、Wilcoxon、Bonferroni 校正比较四个条件，单轮 CCR 的 F1 为 0.376；多轮 D‑CCR 分别为 0.303、0.293、0.263；多轮提升召回率但显著降低精确率；

**⚠️ 局限性**

局限性包括仅使用单一模型（Claude Opus 4.6）；错误为人工注入而非自然；匹配算法可能漏检或误判；多轮实验未检验多模型或人类基准；D‑CCR‑2c 条件仅单跑导致估计不稳。

---

## 371. GAP-MLLM: Geometry-Aligned Pre-training for Activating 3D Spatial Perception in Multimodal Large Language Models

**arXiv ID:** 2603.16461 | [PDF](https://arxiv.org/pdf/2603.16461v1)

**作者:** Jiaxin Zhang `[一作]` (Harbin Institute of Technology), Dave Zhenyu Chen `[通讯]` (Huawei)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出GAP-MLLM框架，通过几何对齐预训练显式激活多模态大语言模型（MLLM）的3D空间感知能力，并在仅使用RGB输入的场景中显著提升性能。

**💡 创新点**

核心创新在于：①稀疏几何‑语义联合预训练目标，让模型在少量点级几何与语义监督下先行激活结构表征；②多层分层门控融合机制，按层动态平衡几何与语义信息，避免传统文本主导训练下几何被弱化。

**🔧 技术方法**

技术细节包括：并行视觉分支（如Qwen3‑VL）与几何分支（如VGGT）提取多层token；token‑级门控多层融合；两阶段训练（稀疏点地图+语义预训练，随后对象级任务微调）；统一度量坐标系统以保证几何一致性。

**📊 数据集**

数据集：稀疏预训练使用ScanNet与EmbodiedScan；下游任务使用ScanRefer（3D定位）、Scan2Cap（密集描述）、EmbodiedScan（视频目标检测）等。

**📈 对比分析**

与多种3D输入模型（SpatialLM、Chat‑3D v2等）和RGB‑only基线（VG‑LLM、Video‑3D LLM等）对比，GAP‑MLLM在3D视觉定位Acc@0.25/0.5、密集描述CIDEr/BLEU‑4、视频目标检测Precision/Recall/F1等指标上均实现或逼近显式3D方法的水平，显著优于同类RGB‑only方案。

**⚠️ 局限性**

局限性包括：需稀疏点级标注；对极端稠密或动态场景的鲁棒性尚未验证；门控融合机制增加了计算开销，且在更大模型或多模态任务中的通用性仍待评估。

---

## 372. A Human-Centred Architecture for Large Language Models-Cognitive Assistants in Manufacturing within Quality Management Systems

**arXiv ID:** 2603.16325 | [PDF](https://arxiv.org/pdf/2603.16325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 373. Balancing Openness and Safety: Central and Peripheral Governance Practices in the Lesbian Subreddit Ecosystem

**arXiv ID:** 2603.16176 | [PDF](https://arxiv.org/pdf/2603.16176v1)

**作者:** Yan Xia `[一作]` (Clemson University), Jinkyung Katie Park `[通讯]` (Clemson University)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5101734911)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了红迪上女同性恋子版块的治理生态，探讨中央与边缘社区如何在开放与安全之间进行权衡。

**💡 创新点**

提出生态系统视角的治理分工模型，揭示中央社区侧重内容质量、边缘社区侧重身份与边界保护，弥补了以往单社区治理研究的局限。

**🔧 技术方法**

结合网络分析（基于侧边栏交叉链接构建有向图）与归纳式主题分析（对167条规则进行编码）。

**📊 数据集**

数据集包含29个女同性恋子版块、51条侧边栏交叉链接及167条社区规则。

**📈 对比分析**

通过对中央与边缘社区规则主题与话题频率的对比，发现中央社区更强调内容审核与正向身份定义，边缘社区更注重边界控制与参与门槛，验证了生态分工假设。

**⚠️ 局限性**

仅以侧边栏交叉链接为网络边缘，未涵盖所有社区互动；规则文本缺乏实际帖子与用户行为的考量；研究聚焦红迪，缺乏跨平台验证，结果的普适性受到限制。

---

## 374. Diffusion Models for Joint Audio-Video Generation

**arXiv ID:** 2603.16093 | [PDF](https://arxiv.org/pdf/2603.16093v1)

**作者:** Alejandro Paredes La Torre `[一作]` `[通讯]` (Duke University), Alejandro Paredes La Torre (Duke University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

发布了两套高质量音视频对齐数据集，并在这些数据集上从零开始训练MM‑Diffusion模型，研究了基于潜在空间的联合扩散方法，最后提出并实现了基于CogVideoX与MM‑Audio的两步文本→音视频生成流水线。

**💡 创新点**

①新建两套30‑秒对齐音视频数据集；②首次从头训练MM‑Diffusion并验证其生成质量；③深入分析潜在空间联合扩散的难点与解码不稳定性；④提出分两步先生成视频再生成同步音频的模块化流程，显著提升生成对齐度。

**🔧 技术方法**

采用扩散模型（MM‑Diffusion、潜在扩散）、双U‑Net交叉注意力、Stable Diffusion VAE、Expert Transformer、CogVideoX、MM‑Audio、流匹配（Flow‑Matching）等技术。

**📊 数据集**

13小时游戏剪辑集（7200段）和64小时音乐会集（1700段），每段统一裁剪为34秒长，用于训练、评估与对比实验。

**📈 对比分析**

通过Fréchet Audio Distance（FAD）和Fréchet Video Distance（FVD）两种指标对比无条件生成、文本-视频生成和MM‑Diffusion生成结果。实验表明，MM‑Diffusion能够生成语义一致的音视频对，但在FAD/FVD上远低于大规模文本-视频模型；两步流水线在对齐度和生成质量上有明显提升。

**⚠️ 局限性**

训练成本高、从零开始需要大量GPU资源；潜在MM‑Diffusion在解码阶段表现不稳定；跨模态解码时音视频对齐仍不充分，导致生成质量受限。

---

## 375. FEEL (Force-Enhanced Egocentric Learning): A Dataset for Physical Action Understanding

**arXiv ID:** 2603.15847 | [PDF](https://arxiv.org/pdf/2603.15847v1)

**作者:** Eadom Dessalene `[一作]` (University of Maryland), Yiannis Aloimonos `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

收集了约3百万帧自同步力量-视角数据集FEEL，并利用力量信号生成可扩展的接触标签。

**💡 创新点**

首次将力量测量与主观视角视频结合，利用低维物理监督实现无监督接触检测和动作表征学习。

**🔧 技术方法**

使用自定义piezo电阻手套与Meta Aria眼镜同步采集，力预测预训练视频骨干网络，并结合DINO+Dense Prediction Transformer实现接触检测与像素级分割。

**📊 数据集**

主要使用FEEL数据集，此外在EPIC‑VISOR、HOI4D、EPIC‑Kitchens、Something‑Something‑V2、Ego‑Exo4D等公开数据集进行零样本迁移和基准对比。

**📈 对比分析**

与传统无监督/弱监督基线相比，力监督模型在接触检测上达87%‑92% IoU，且在动作识别的冻结与微调设置中提升3–8% top‑1，尤其在小型数据集上表现突出。

**⚠️ 局限性**

局限包括手套佩戴不舒适、传感器漂移导致噪声、伪标签仅针对运动物体，无法覆盖静止接触对象，硬件可扩展性和实时性仍待提升。

---

## 376. Systematization of Knowledge: The Design Space of Digital Payment Systems with Potential for CBDC

**arXiv ID:** 2603.16320 | [PDF](https://arxiv.org/pdf/2603.16320v1)

**作者:** Judith Senn `[一作]` (University of Innsbruck), Rainer Böhme `[通讯]` (University of Innsbruck)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对已有的数字支付系统设计进行系统化梳理，提出统一的评估框架，识别了四种主要设计模式，并在隐私、可用性与性能等维度上对约三十种方案进行比较与量化。

**💡 创新点**

创新点包括首次以技术视角进行顶层系统化分析，构建了包含 28 个属性的综合评估矩阵，归纳出“全局状态更新”“局部状态更新”“数字硬币传输”“销毁-创建”四大设计模式，并对隐私、合规与可用性权衡做出定量描述。

**🔧 技术方法**

主要技术涉及密码学隐私增强技术（PET）、零知识证明、区块链/UTXO 模型、可信硬件、分布式共识与分片等；论文通过对这些技术的组合与实现细节进行理论分析，而非单一技术实现。

**📊 数据集**

本文未使用真实交易数据集，而是基于文献综述、已有设计的说明与公开的实验报告进行分析，强调缺乏统一的公共数据集用于基准测试。

**📈 对比分析**

比较方法采用表格化的属性对比与量化指标（如身份隐私、价值隐私、用户不可链接性、离线功能、精确金额、收款人是否参与等），并给出各模式在性能（吞吐量、延迟）与可用性方面的经验数值；但大部分数据来源于合成基准，真实部署的系统占比较少。

**⚠️ 局限性**

局限性包括：缺乏完整实现与实测；大多数设计未实现或仅在实验环境下验证；缺少可比的性能基准与统一交易数据集；对后量子安全与离线支付的探讨仍不足；并且无机器可验证的安全证明，导致安全性难以量化验证。

---

## 377. Beyond Grading Accuracy: Exploring Alignment of TAs and LLMs

**arXiv ID:** 2603.16357 | [PDF](https://arxiv.org/pdf/2603.16357v1)

**作者:** Matthijs Jansen op de Haar `[一作]` (University of Twente), Faizan Ahmed `[通讯]` (University of Twente)

**通讯引用:** 587 | [OpenAlex ID](https://openalex.org/A5089107604)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建一个评估流程，使用六种主流开源大型语言模型（LLM）对 92 个软件设计课程中的 UML 类图进行逐条评判，并与教学助理（TA）的评分进行对比。

**💡 创新点**

创新点在于：①聚焦非专有 LLM，解决了高校对透明性与成本的需求；②引入逐条（criterion‑level）评估，揭示模型与人类评审在细粒度上的差异；③探索“最优模型”组合，展示不同 LLM 在各条评判标准上的互补优势；④提出混合式（mixed‑initiative）人机协作的可行路径。

**🔧 技术方法**

技术上主要包括：JSON 解析器将学生提交的图形数据转化为自然语言描述；构造包含评分规则和输出格式的提示；在温度为 0 的确定性设置下调用 LLM；使用 per‑criterion accuracy、Pearson 相关系数、平均绝对误差（MAE）等指标对比 LLM 与 TA 的评分。

**📊 数据集**

数据集为 92 张来自本科一年级软件设计考试的 UML 类图，学生使用 UTML 工具生成，包含 JSON 与 PNG 两种格式。

**📈 对比分析**

比较方法：将 LLM 的 per‑criterion 评分与 TA 的评分逐项对齐，计算准确率、Pearson 相关和 MAE。实验结果显示 LLM 的最高 per‑criterion 准确率达 88.56%，Pearson 相关系数最高 0.78，整体性能已显著提升到接近 TA 的水平。

**⚠️ 局限性**

局限性包括：①大多数模型在多重性、后期出现的实体以及多跳关系链上存在显著误判与欠评分偏差；②存在系统性的“苛刻”评分倾向，导致对学生不公平；③模型对长上下文的注意力受限，导致“位置偏倚”；④大模型对 GPU 内存和 KV 缓存需求高，实际部署成本较大，需要人机协作来弥补。

---

## 378. Fast-HaMeR: Boosting Hand Mesh Reconstruction using Knowledge Distillation

**arXiv ID:** 2603.16444 | [PDF](https://arxiv.org/pdf/2603.16444v1)

**作者:** Hunain Ahmed Jillani `[一作]` (RPTU), Didier Stricker `[通讯]` (DFKI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种将轻量化骨干网络与知识蒸馏相结合的 3D 手部重建方法，显著加速 HaMeR 模型，减少模型体积并保持高精度；

**💡 创新点**

创新点在于同时探索输出层、特征层和混合蒸馏策略，并在 HaMeR 基础上将 ViT-H 替换为 MobileNet、MobileViT、ConvNeXt、ResNet 等轻量化骨干，取得 35% 参数压缩、1.5 倍速度提升且误差仅 0.4mm；

**🔧 技术方法**

采用轻量化骨干网络、知识蒸馏（输出、特征及组合损失）、MANO 参数回归、重投影损失、3D/2D 关键点误差等技术；

**📊 数据集**

使用与 HaMeR 相同的多源合并数据集（FreiHAND、HO3D、H2O3D、InterHand2.6M、MTC、DexYCB、RHD、COCO WholeBody 等），并在 HO3D‑v2 进行评估；

**📈 对比分析**

与原 HaMeR 及其他基线在 HO3D‑v2 上对比，采用 PA‑MPJPE/MPVPE、F@5mm/15mm、参数量和 FPS 作为指标。结果显示 ConvNeXt‑L + 特征蒸馏几乎与 HaMeR 等效，仅误差 0.2mm，速度提升 1.5×，参数量降至约 35%；

**⚠️ 局限性**

局限性包括：蒸馏对小容量模型效果有限，混合蒸馏并未显著提升；仅针对单视图手部重建，未充分探讨多手或手物交互场景及更复杂的泛化能力。

---

## 379. DimFlux: Force-Directed Additive Line Diagrams

**arXiv ID:** 2603.16366 | [PDF](https://arxiv.org/pdf/2603.16366v1)

**作者:** Marcel Nöhre `[一作]` (University of Kassel), Gerd Stumme `[通讯]` (University of Kassel)

**通讯引用:** 11604 | [OpenAlex ID](https://openalex.org/A5053751744)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 DimFlux 算法，将 DimDraw 的结构化二维扩展与 Zschalig 的力导向优化相结合，生成双加性（对象与属性共同参与向量求和）的概念格线图。

**💡 创新点**

创新点在于把双加性布局与冲突距离最大化的力导向模型融合，并证明加性与实时代数嵌入不可兼容；同时提供从任意线图到最近加性图的正交投影与重投影方法。

**🔧 技术方法**

采用 DimDraw 的二维扩展、正交投影、力导向优化（排斥、吸引、重力）以及 SMT 求解验证可行性，构成完整的 DimFlux 工作流程。

**📊 数据集**

在 126 个四个可合性元素的概念格（所有简化上下文）及来自《Formal Concept Analysis》书中的真实案例上进行实验评估。

**📈 对比分析**

与手绘、属性加性 FDP、双加性 FDP、DimDraw 四种方法比较，DimFlux 在中小型格中与双加性 FDP 性能相当，对大型格保持结构完整并与手绘相符，显示出更好的可读性。

**⚠️ 局限性**

局限性包括在大型格中仍可能产生点云般的不可读布局，缺乏斜率一致性，且无法处理极复杂或需嵌套/几何表示的格。

---

## 380. A complexity analysis of the F4 Gröbner basis algorithm with tracer data

**arXiv ID:** 2603.16378 | [PDF](https://arxiv.org/pdf/2603.16378v1)

**作者:** Robin Kouba `[一作]` (Sorbonne Université), Mohab Safey El Din `[通讯]` (Sorbonne Université)

**通讯引用:** 1840 | [OpenAlex ID](https://openalex.org/A5048254079)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文对一种F4算法变体进行复杂度分析，并提出一种使用Gröbner跟踪的F4T算法，用于在满足特定可兼容性假设的多项式系统上高效地求得最小Gröbner基。

**💡 创新点**

创新点在于首次给出在泛型假设下F4T算法的精确大O复杂度上界，利用Gröbner跟踪在每个迭代中剔除所有归零对和冗余约化子，从而避免了无用的行列式运算；并通过半正则序列与同时Noether位置的结合，进一步降低了计算量。

**🔧 技术方法**

主要技术包括：F4算法的行列式化简（Macaulay矩阵的行最简形）、行最简化归约子构造子程序、Gröbner跟踪的数据结构、矩阵乘法常数ω的最优算法（如Strassen、Coppersmith–Winograd等），以及Hilbert多项式理论与泛型性证明。

**📊 数据集**

使用的数据集为在Zariski开集δnn∩δn∩δnn中的n元δ度同次多项式序列（即随机同次多项式系统），并在这些序列满足同时Noether位置假设的前提下进行实验。

**📈 对比分析**

与已有的F4、F5或其他行列式化简方法相比，F4T算法在满足Gröbner跟踪的兼容性假设后，其算术复杂度上界为O(δ^ω n+1 e^n c(ε,δ,ω))，在理论上与或优于现有最优泛型复杂度估计，尤其在ω≈2.373时得到最优表现。

**⚠️ 局限性**

局限性包括：需要在泛型的Zariski开集上（即序列是正则的、满足同时Noether位置、ϕ( ,n−1)∈S_δ,n−1,n）；对非泛型输入时无法保证复杂度；并且对δ不固定时仍需更多假设；此外实际实现仍需处理行列式分解和秩判定的常数项。

---

## 381. Three-Dimensional Affine Spatial Logics

**arXiv ID:** 2603.16308 | [PDF](https://arxiv.org/pdf/2603.16308v1)

**作者:** Adam Trybus `[一作]` (Jagiellonian University), Adam Trybus `[通讯]` (Jagiellonian University)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5056681459)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文构造并研究了在实数空间中以正则开放有理多面体为元素的区域基仿射空间逻辑，并证明不同维度的模型具有不同的理论；进一步在三维情形下建立了坐标框架、半空间、平面、直线之间的可定义关系，定义了平面上的加减乘运算，并证明每个区域都满足一个“仿射完整”公式，即该公式的所有满足者彼此仿射等价。

**💡 创新点**

创新点主要包括：①首次系统性比较二维与三维仿射逻辑的理论差异；②在三维中构造可定义坐标框架并利用其固定半空间，从而得到可表达所有有理多面体的“仿射完整”公式；③通过引入可定义的加减乘运算，将数值概念引入空间逻辑，展示了在三维空间中对有理数的完整编码。

**🔧 技术方法**

主要技术手段是：一阶逻辑定义的正则开放有理多面体结构、布尔代数运算的可定义性、Helly定理的逻辑化、半空间/平面/直线之间的关系公式构造、坐标框架的逻辑表述、加减乘运算的几何构造与逻辑编码；结合这些技术实现了对不同维度模型的理论比较和仿射完整性的证明。

**📊 数据集**

本文为理论工作，不涉及实验数据集；所有结果均基于逻辑语义和几何推导，无需使用外部数据集。

**📈 对比分析**

没有实验比较或性能评估；所给出的结论是理论性的，主要通过逻辑可定义性和模型论推理得到。与之前二维结果的比较主要体现在理论差异与表达能力上，并未给出可度量的性能指标。

**⚠️ 局限性**

局限性包括：①三维模型的公理化系统尚未完成；②所使用的公式往往较长、复杂，实际可用性受限；③虽然证明理论不可判定，但未给出复杂度分析；④推广到更高维度的具体方法仍是开放问题。

---

## 382. MoLoRA: Composable Specialization via Per-Token Adapter Routing

**arXiv ID:** 2603.15965 | [PDF](https://arxiv.org/pdf/2603.15965v1)

**作者:** Shrey Shah `[一作]` (Microsoft Corporation), Justin Wagle `[通讯]` (Microsoft Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出每个token级别的路由机制和MoLoRA（Mixture of LoRA），实现多适配器在单前向过程中并行处理不同模态或多任务需求的推理。

**💡 创新点**

创新点在于：①证明per-token路由在计算上最优；②将Mixture of Experts与多适配器统一，推出可组合的专门化方法MoLoRA；③设计热集内存和CUDA图捕获的高效系统架构，实现极低延迟。

**🔧 技术方法**

使用技术包括：LoRA低秩适配、确定性词表路由、学习门控的MLP路由器、张量核心加速的组化GEMM、热集GPU内存布局、CUDA图捕获、适配器分块调度等。

**📊 数据集**

评测数据集：GSM8K、MATH、BBH、GPQA，用于推理性能评估；Chameleon-style统一词表用于多模态交替生成；Qwen3系列模型作为基础模型。

**📈 对比分析**

与S-LoRA、Punica等传统per-sequence路由系统对比：per-token路由在4模态工作负载下从4次前向减少到1次，理论上提升4×，实验中在热集+CUDA图下达到5.5×；MoLoRA在1.7B基础上加入4个LoRA后在四大推理基准上分别提升14%、8%、2.5%和2.1%，并超过4.7×规模的8B模型，P99延迟下降约67×。

**⚠️ 局限性**

局限性包括：依赖统一词表连续分块的模型，对不具备此结构的多模态模型需改用学习路由；热集内存限制可并发部署的适配器数量；学习路由需要额外训练且可能出现负载不均衡。

---

## 383. A Comprehensive Benchmark of Histopathology Foundation Models for Kidney Histopathology

**arXiv ID:** 2603.15967 | [PDF](https://arxiv.org/pdf/2603.15967v1)

**作者:** Harishwar Reddy Kasireddy `[一作]` (University of Florida), Pinaki Sarder `[通讯]` (University of Florida)

**通讯引用:** 2591 | [OpenAlex ID](https://openalex.org/A5044061952)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估11种预训练视觉基础模型（HFMs）在肾脏病理学下的转移学习表现，涵盖 PAS、H&E、PASM、IHC 等多染色、图块级和切片级任务。

**💡 创新点**

首次在肾脏病理学上进行跨染色和内染色的全面基准测试，构建了可复现的评估流水线并公开了开源工具包；同时将多任务（分类、回归、复制检测）与统计检验（Friedman + Wilcoxon + CLD）结合，以客观评估模型差异。

**🔧 技术方法**

采用冻结的 HFMs 嵌入作为特征；图块级任务使用线性回归与 kNN 探测，回归任务使用岭回归；切片级任务使用 ABMIL（多实例学习）进行分类与回归；通过重复分层交叉验证、bootstrap 置信区间和统计显著性检验确保结果稳健。

**📊 数据集**

使用多中心公开数据集：WUSTL、Vanderbilt、NIH/NIDDK、JHU、UC Davis 等，包含 215 PAS 病例、63 PASM 病例、12 H&E 病例、14 UC Davis H&E 病例、137 PAS（DN vs 正常）和 85 PAS（MN 治疗响应）等，样本量从数百到两千多张图块不等。

**📈 对比分析**

评估指标包括 MCC（分类）、PCC（回归）、R²（回归）和准确率（复制检测）。结果显示：对粗放结构（如全局肾小球硬化、炎症）取得 MCC ≈0.9‑0.96；对细微结构（GBM spike、动脉狭窄）仅达到 MCC 0.2‑0.4；对分子或预后相关任务（细胞比例回归、eGFR 预测、MN 反应）表现差到负值（MCC < 0，R² < 0）。

**⚠️ 局限性**

限制：HFMs 主要以肿瘤数据预训练，缺乏肾脏专属特征；对微观结构、分子表型和长期预后预测能力不足；噪声鲁棒性不佳；切片级评估样本量有限；未使用多模态或组织专属再训练，导致迁移效果受限。

---

## 384. I Know What I Don't Know: Latent Posterior Factor Models for Multi-Evidence Probabilistic Reasoning

**arXiv ID:** 2603.15670 | [PDF](https://arxiv.org/pdf/2603.15670v1)

**作者:** Aliyu Agboola Alege `[一作]` `[通讯]` (Epalea), Aliyu Agboola Alege (Epalea)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文设计了一种将VAE潜在后验映射为软似然因子，再通过Sum-Product Network或神经学习方式完成多证据聚合的框架。

**💡 创新点**

创新点在于首次将VAE的不确定性与结构化概率推理相结合，提供可解释的校准不确定性，并实现了两种可比的聚合架构。

**🔧 技术方法**

技术上使用变分自编码器、Sum-Product Network、蒙特卡罗积分、温度缩放与可学习的质量/一致性网络。

**📊 数据集**

使用八个多样化任务（包括合规评估、医疗诊断、财经预测等）以及公开的FEVER事实验证数据集进行实验。

**📈 对比分析**

在所有基线（EDL、BERT、R-GCN、LLM等）上，LPF-SPN以97.8%准确率、1.4% ECE优于对手，而LPF-Learned亦实现了高于传统方法的约91%准确率。

**⚠️ 局限性**

局限在于LPF-SPN需要显式SPN结构和多次解码，导致推理复杂度上升；LPF-Learned虽然速度快但校准性不及SPN，且对训练数据的依赖较强。

---

## 385. CritiSense: Critical Digital Literacy and Resilience Against Misinformation

**arXiv ID:** 2603.16672 | [PDF](https://arxiv.org/pdf/2603.16672v1)

**作者:** Firoj Alam `[一作]` (Qatar Computing Research Institute), Raian Ali `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 5113 | [OpenAlex ID](https://openalex.org/A5088731268)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CritiSense这款多语言移动端媒体素养应用，通过微课程、互动测验、即时反馈和模拟练习，帮助用户在日常社交媒体中识别宣传与错误信息；

**💡 创新点**

创新点在于①支持九种语言的模块化平台；②以技术为先导的预防式学习模式；③将轻量化AI检测与用户训练相结合；④移动端微学习与实时交互体验；

**🔧 技术方法**

技术实现包括移动App（iOS/Android）、交互式微课与测验、即时解释反馈，以及使用BERT‑base、AraBERT、ViT‑B/16等轻量化AI模型；

**📊 数据集**

使用的数据集涵盖文本事实核查（AraFacts2、ANS‑Claim、CT22Claim、NewsCredibilityDataset、COVID19Factuality、PolitiFact、CT22T3_Factuality、CheckThat!、AVeriTeC）、文本宣传（PropXplain）、图像宣传（ArMeme）与仇恨表情包（Facebook Hateful Memes）；

**📈 对比分析**

模型以轻量化为主，CPU可部署，评估指标为文本任务的Micro‑F1、图像任务的Macro‑F1；文本事实核查中阿拉伯语Mi‑F1=0.868、英语0.726；宣传检测Mi‑F1=0.772/0.762；图像ArMeme 0.554、Hateful meme 0.507；

**⚠️ 局限性**

局限性包括：仅评估短期可用性与学习效果，未测长时记忆或真实行为；内容需持续更新与本地化；多语言覆盖仍有限；仅为平台检测的补充；存在双重使用风险，需持续监控。

---

## 386. Advancing Visual Reliability: Color-Accurate Underwater Image Enhancement for Real-Time Underwater Missions

**arXiv ID:** 2603.16363 | [PDF](https://arxiv.org/pdf/2603.16363v1)

**作者:** Yiqiang Zhou `[一作]` (China Telecom), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 62016 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量化、实时的水下图像增强框架，能够在保持高色彩还原度的同时满足嵌入式设备的计算限制。

**💡 创新点**

创新点包括：①自适应加权通道补偿（AWCC）利用绿色通道动态恢复红蓝通道；②多分支可重参数膨胀卷积（MRDConv）在训练时多尺度融合、推理时单路径高效化；③基于统计先验的全局颜色调整（SGCA）实现轻量化的色温、色调与饱和度调节。

**🔧 技术方法**

采用CNN骨干网络与可重参数化卷积、灰度世界假设、VGG感知损失、色彩一致性损失等技术；在推理时通过参数融合实现5×5卷积，显著降低运算量。

**📊 数据集**

在UIEB、LSUI、EUVP（S、I、D）八个公开数据集上进行训练和评估，并在U45、RUIE、ColorCheck7等测试集验证跨域鲁棒性。

**📈 对比分析**

与十余种主流方法（如Phaseformer、FiveA+、SFGNet、MobileIE等）对比，模型仅有3,880个参数、0.01 MB、0.145 G FLOPs，推理速度达409 FPS；在PSNR/SSIM/LPIPS/UIQM等指标上均位列前列，尤其在UCIQE提升29.7%，并在ROV实时实验中显著增加SIFT匹配数量。

**⚠️ 局限性**

局限性：SGCA模块中的Top‑K操作随着分辨率增大计算量急剧上升，限制了超高分辨率应用；在极端浑浊或光照异常条件下，颜色补偿与全局调整仍可能产生轻微失真。

---

## 387. Quantum-Secure-By-Construction (QSC): A Paradigm Shift For Post-Quantum Agentic Intelligence

**arXiv ID:** 2603.15668 | [PDF](https://arxiv.org/pdf/2603.15668v1)

**作者:** Arit Kumar Bishwas `[一作]` (PricewaterhouseCoopers), Joel Jacob Varghese `[通讯]` (PricewaterhouseCoopers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了量子安全构建（QSC）范式，将后量子密码、量子随机数和量子密钥分发集成到多智能体 AI 的生命周期中，实现全局安全通信。

**💡 创新点**

创新点在于把量子安全作为系统根本属性，提供基于运行时策略的加密姿态选择、统一会话密钥派生、治理感知的编排层，支持跨境、多域、长期运行的代理系统。

**🔧 技术方法**

使用后量子密码（Kyber、Dilithium）、量子随机数生成器（QRNG）和量子密钥分发（QKD）三层加密栈，并配合 LangGraph、MCP、Post‑Quantum TLS、AEAD 等协议实现安全堆栈。

**📊 数据集**

实验基于 Azure 云和本地测试平台，使用合成任务图、模拟 QKD 链路和量子随机数服务，未采用公开数据集而采用内部生成的多代理工作负载。

**📈 对比分析**

与传统 TLS‑1.3 对比，PQC+QRNG+QKD 的微基准延迟提升约 6–12% 但绝对延迟仍低于 10 ms；云端实验显示网络主导，总延迟约 300 ms，保持线性扩展；在 10⁵ 次攻击模拟中 100% 检测率。

**⚠️ 局限性**

局限性包括 QKD 的可扩展性和物理部署成本、握手延迟导致的聚合瓶颈、对政策与合规机制的依赖，以及实验环境缺乏真正量子链路导致的性能估计不确定性。

---

## 388. FlatLands: Generative Floormap Completion From a Single Egocentric View

**arXiv ID:** 2603.16016 | [PDF](https://arxiv.org/pdf/2603.16016v1)

**作者:** Subhransu S. Bhattacharjee `[一作]` (Australian National University), Rahul Shome `[通讯]` (Australian National University)

**通讯引用:** 557 | [OpenAlex ID](https://openalex.org/A5076976011)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了单视角 egocentric RGB 图像到地面鸟瞰 BEV 地图完成的数据集与基准，并在此上评估了确定性、集合与后验采样的生成模型。

**💡 创新点**

创新点在于将单视角 BEV 完成问题建模为条件后验采样任务，提出新的 UMR 与 MES 评估指标，并发布首个含 27 万条观测的室内 BEV 完成基准。

**🔧 技术方法**

采用深度估计与地面分割后投影得到观测 BEV，并以此条件训练 LaMa、Diffusion、Flow Matching、FM+XAttn 等后验生成模型，同时对比参数无关填充、U-Net、PConv 等基线。

**📊 数据集**

使用了 6 个真实室内 RGB‑D 数据集（Matterport3D、ScanNet、ScanNet++、ARKitScenes、3RScan、ZInD），通过虚拟相机生成 17,656 场景、270,575 条观测。

**📈 对比分析**

对比了无参数基线、确定性预测器、集合模型以及三类后验采样方法；在 ID 与 OOD 测试中，后验采样（尤其 FM+XAttn）在 UMR/MES 上优于确定性模型，且模型排名保持不变。

**⚠️ 局限性**

局限性包括仅处理二值通行性、忽略语义与梯度通行、使用简化摄像机内参、未覆盖多层楼/多室环境，以及单视角下对未观测区域不确定性的评估不足。

---

## 389. Fast and Reliable Gradients for Deformables Across Frictional Contact Regimes

**arXiv ID:** 2603.16478 | [PDF](https://arxiv.org/pdf/2603.16478v1)

**作者:** Ziqiu Zeng `[一作]` (National University of Singapore), Fan Shi `[通讯]` (National University of Singapore)

**通讯引用:** 78944 | [OpenAlex ID](https://openalex.org/A5100361956)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种全 GPU 加速的可微分模拟器，能够在存在摩擦接触和大变形的情况下稳定求解逆问题，如材料识别和逆动力学控制。

**💡 创新点**

创新点在于：①将摩擦接触建模为光滑 NCP（使用 smoothed Fischer–Burmeister 函数）并在正向和反向求解中统一使用；②设计了长周期一致性、统一接触稳定性和 Within‑Block Commutation 条件，以保证梯度的物理一致性；③开发了针对不同接触场景（无接触、无摩擦、摩擦）的一套预条件器与 Krylov 求解器（CG/GMRES + Sparse‑Inverse/Woodbury/Jacobi）。

**🔧 技术方法**

核心技术包括：差分隐式 Euler、项目式动力学（PD）与 DiffPD/ DiffCloth 结合、NCP+Fischer–Burmeister 平滑化、隐式微分+伴随法、GPU‑加速稀疏线性代数、稀疏逆预条件器、Woodbury 预条件器、GMRES 与 CG 迭代求解器。

**📊 数据集**

使用自定义的高保真仿真环境（软体机械臂、布料、软鸭、软象、软抓手等）进行单参、多参、非线性弹性、摩擦接触和长时间演化的实验；没有公开数据集，全部为内部生成的模拟案例。

**📈 对比分析**

与 DiffPD、DiffCloth 以及半隐式固定点迭代基线对比，采用相同的测试脚本；在无接触、无摩擦、摩擦三种模式下，Sparse‑Inverse+CG 或 Woodbury+GMRES 能在更少迭代、更多 GPU 资源下完成收敛，梯度与数值差分几乎无误差；在软体抓取和写字任务中，能在约一小时内完成 1 小时长的优化，显示出优越的计算效率与梯度质量。

**⚠️ 局限性**

局限性包括：①平滑参数 ε 需要经验调优，过大可能导致接触物理误差；②GPU 内存限制，极高分辨率网格时稀疏因子化内存占用高；③仅支持固定拓扑，无法处理破裂、撕裂等拓扑变化；④对极 stiff 摩擦约束的数值稳定性仍有挑战。

---

## 390. SynthChain: A Synthetic Benchmark and Forensic Analysis of Advanced and Stealthy Software Supply Chain Attacks

**arXiv ID:** 2603.16694 | [PDF](https://arxiv.org/pdf/2603.16694v1)

**作者:** Zhuoran Tan `[一作]` (University of Glasgow), Christos Anagnostopoulos `[通讯]` (University of Glasgow)

**通讯引用:** 2849 | [OpenAlex ID](https://openalex.org/A5001331936)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了 SynthChain，一个面向软件供应链攻击的多阶段、多源实时数据集与基准测试环境，用于评估在可观测性受限条件下的链级检测与重建。

**💡 创新点**

创新点在于：①首次提供具备真实攻击链级 ground truth 的多源（主机、网络、服务、容器）实时观测数据；②系统化评估不同源组合对链级可观测性的影响；③基于多源融合的检测与链重建方法，揭示单源不足与多源协同的收益。

**🔧 技术方法**

使用技术包括：近生产级实验平台、eBPF/Tracee 主机追踪、Mythic C2 日志导出、LLM 辅助的 payload 行为抽取、规则驱动的粗粒度事件标记、事件关联图构建与链重建、跨源联邦与评估指标计算。

**📊 数据集**

数据集为 SynthChain：7 个覆盖 PyPI、npm、C/C++ 的供应链攻击场景，包含约 0.58M 原始多源事件、1.5M 评估记录，涵盖 161 项 MITRE ATT&CK 技术、14 项战术；基于真实恶意包与攻击活动构建。

**📈 对比分析**

比较方法：在单源、两源、三源及完整多源预算下执行链重建，评估指标为标签覆盖率、链覆盖率、步骤/链召回与精准度以及整体重建可行性。结果显示：单源最佳 0.391 覆盖率/0.403 重建率；两源最佳 0.636 覆盖率/0.639 重建率（约 1.6 倍提升）；完整多源 0.481 覆盖率/0.488 重建率。

**⚠️ 局限性**

局限性包括：①仍存在缺失阶段的证据瓶颈，导致无法完整重建；②非可关联的证据（无公共 join key）导致信息断裂；③基于规则的粗粒度匹配可能引入误报或漏报；④仅覆盖 Windows/Linux，未包含 macOS 等系统；⑤侧重终端主机阶段，未覆盖完整 APT 生命周期；⑥云控制平面和 IAM 日志等关键来源缺失或无法充分匹配。

---

## 391. On the Emotion Understanding of Synthesized Speech

**arXiv ID:** 2603.16483 | [PDF](https://arxiv.org/pdf/2603.16483v1)

**作者:** Yuan Ge `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**通讯引用:** 11855 | [OpenAlex ID](https://openalex.org/A5100600701)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

系统评估了语音情感识别（SER）模型在合成语音上的表现，并揭示了人类语音与合成语音之间的显著性能差距。

**💡 创新点**

发现主导的性能瓶颈是语音生成过程中的自动回归（AR）词元预测错误，导致语音表征与真实语音不匹配，从而导致 SER 模型主要利用非鲁棒快捷方式。

**🔧 技术方法**

采用 Emotion2vec 及多种生成式语音大模型（如 Qwen3-Omni、GPT‑4o‑Audio）进行评测，并尝试了域对抗微调（GRL）和线性探测器等技术来缓解分布偏移。

**📊 数据集**

使用公开情感语料库 TESS、CREMA‑D、IEMOCAP、RAVDESS、MELD 等，并生成对应的 TTS/S2S 合成音频（CosyVoice2、IndexTTS2、Kimi‑Audio、GLM‑4‑Voice、GPT‑4o‑TTS/AUDIO）。

**📈 对比分析**

实验表明在人类语音上 SER 准确率可达 70%–80%，但在合成语音上仅 15%–35%；即使对合成数据进行微调或域对抗训练，模型在未见合成器或跨语料库的 OOD 场景下仍表现不佳。

**⚠️ 局限性**

主要局限在于合成语音数据规模受限于人工筛选、短时标注偏差，以及域对抗方法在跨合成器泛化上的不足，未能充分克服 token 预测导致的表征偏移。

---

## 392. Persistent Story World Simulation with Continuous Character Customization

**arXiv ID:** 2603.16285 | [PDF](https://arxiv.org/pdf/2603.16285v1)

**作者:** Jinlu Zhang `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**通讯引用:** 32179 | [OpenAlex ID](https://openalex.org/A5016080094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一套名为EverTale的持续学习故事可视化系统，能够在不重训模型的前提下，连续添加新角色并保持已有角色的身份一致性；

**💡 创新点**

核心创新在于三大组件：1）All‑In‑One World Character Integrator将所有角色知识压缩进单一LoRA模块，并通过投影矩阵D实现角色知识的互不干扰；2）Character Quality Gate利用MLLM进行链式思考评估，每个角色的定制质量自动判断是否继续学习；3）Character‑Aware Region‑Focus Sampling在多角色场景下通过局部自注意力与跨注意力分区，实现自然布局与高身份保真。

**🔧 技术方法**

使用LoRA低秩适配、投影矩阵D构造的左零空间投影、MLLM（多模态大语言模型）评估与决策、Stable Diffusion 1.5/XL/DiT等扩散模型、CLIP/DINO语义匹配以及自注意力局部化与区域采样策略。

**📊 数据集**

基准数据集为TBC‑Bench（包含Pororo、Frozen等多故事集），并使用相应的角色真实图像与多角色场景。

**📈 对比分析**

在单角色与多角色可视化任务中，对比了训练‑free、调优‑free、CL、文本插值、LoRA‑M/C、Mix‑of‑Show、StoryWeaver、FLUX.1 Kontext等方法。EverTale在DINO/CLIP的身份一致性、CLIP‑文本语义对齐、用户偏好评分以及生成速度等多项指标上普遍位居前列，特别是在连续学习设置下仍保持高身份保真和文本一致性。

**⚠️ 局限性**

主要局限包括：1）单LoRA模块容量有限，角色数量激增时可能出现细节退化；2）投影矩阵D的构造与维度设定需要经验调参；3）MLLM评估引入额外推理成本；4）在DiT体系下实验受限，尚未完整验证；5）对角色图像质量和多样性仍有依赖。

---

## 393. Communication-Aware Multi-Agent Reinforcement Learning for Decentralized Cooperative UAV Deployment

**arXiv ID:** 2603.16141 | [PDF](https://arxiv.org/pdf/2603.16141v1)

**作者:** Enguang Fan `[一作]` (University of Illinois at Urbana-Champaign), Jae Kim `[通讯]` (Boeing Research and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图神经网络的CTDE多智能体强化学习框架，用于在部分可观测和距离受限通信条件下部署无人机队列。

**💡 创新点**

创新点包括：①双重注意力编码器同时处理无人机与周边实体的注意力和邻居间的自注意力；②在训练时使用集中式评估器提升学习稳定性，同时保持执行时的去中心化；③通过图表示实现对不同团队规模的零样本泛化。

**🔧 技术方法**

技术手段为：集中式训练、去中心化执行（CTDE）；注意力机制（scaled dot‑product attention）用于 agent‑entity 与 neighbor self‑attention；PPO‑style actor‑critic学习；多轮消息传递与更新。

**📊 数据集**

使用仿真数据集，包括 DroneConnect（协同覆盖任务）和 DroneCombat（混合竞争任务），场景在 2D 区域内随机移动的无人机与地面节点。

**📈 对比分析**

与静态 MILP 上界、单智能体 RL 以及无通信/无实体注意力的消融实验对比。结果显示：在 RC+PO 约束下，平均覆盖率 0.74–0.79，接近 MILP 上界 0.77–0.80；在 DroneCombat 中，零样本赢率提升至 0.62 对比基线 0.49，平均决斗步长缩短。

**⚠️ 局限性**

局限性：仅在仿真环境中验证，缺乏真实世界测试；通信成本与网络QoS模型未详细评估；对极端动态场景或更大规模团队的泛化能力尚未系统评估；算法对超参数敏感，训练成本相对较高。

---

## 394. Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models

**arXiv ID:** 2603.16001 | [PDF](https://arxiv.org/pdf/2603.16001v1)

**作者:** Sijie Li `[一作]` (University of Sheffield), Jungong Han `[通讯]` (Tsinghua University)

**通讯引用:** 24918 | [OpenAlex ID](https://openalex.org/A5046605531)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对大型视觉语言模型（LVLM）的非对称文本‑视觉权重剪枝方法ATV‑Pruning，使用完整文本令牌加上块自适应选取的少量视觉令牌构成校准池，从而实现高稀疏度下的精细剪枝。

**💡 创新点**

创新点：①识别并量化文本路径对剪枝的高度敏感性与视觉路径的冗余性；②构造异构校准池，仅用文本全量且视觉仅选取重要子集；③采用视觉漂移（cosine差）做token重要性评分，并按块自适应分配视觉令牌预算，使剪枝更精确且高效。

**🔧 技术方法**

技术手段：基于Wanda的激活感知剪枝框架；使用视觉漂移（输入‑输出向量余弦距离）作为视觉令牌的重要性；块自适应预算算法；无结构稀疏化（按权重重要性去零）。

**📊 数据集**

数据集：校准使用ShareGPT4V 128张图文对；评测在9个标准多模态基准上：GQA、MMBench、MME、MMMU、OK‑VQA、POPE、ScienceQA‑IMG、TextVQA、VizWiz‑VQA；实验模型包括LLaVA‑Next 8B、Qwen2‑VL 7B。

**📈 对比分析**

比较方法与性能：与SparseGPT、Wanda、TAMP等现有剪枝方案在50%/60%稀疏度下对比。ATV‑Pruning在LLaVA‑Next 50%时平均保持率94.0%、60%时77.0%，在大多数任务上领先；在Qwen2‑VL 60%时平均保持率85.6%，同样优于基线。速度方面，ATV‑Pruning仅比Wanda略慢，但比SparseGPT和TAMP快数倍。

**⚠️ 局限性**

局限性：仅针对视觉与文本两种模态，未覆盖音频、深度等其他模态；对α超参数的依赖仍需经验调优；目前只实现无结构稀疏，未考虑半结构或量化场景。

---

## 395. Morphemes Without Borders: Evaluating Root-Pattern Morphology in Arabic Tokenizers and LLMs

**arXiv ID:** 2603.15773 | [PDF](https://arxiv.org/pdf/2603.15773v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 396. ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery

**arXiv ID:** 2603.16616 | [PDF](https://arxiv.org/pdf/2603.16616v1)

**作者:** Weiqin Jiao `[一作]` (University of Twente), Claudio Persello `[通讯]` (University of Twente)

**通讯引用:** 5010 | [OpenAlex ID](https://openalex.org/A5029035818)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种一次性完成全景矢量地图生成的方法，能够在一张航拍图像中同时生成所有地物类的多边形，且共享边缘、无间隙、无重叠，满足严格的拓扑一致性要求。

**💡 创新点**

创新点在于提出All‑Class Polygonal Vectorization (ACPV)任务，并设计了ACPV-Net框架，其中引入Semantically Supervised Conditioning (SSC)使语义监督直接指导扩散式顶点生成，以及基于过密平面直线图(PSLG)的确定性拓扑重构，保证全局拓扑一致性。

**🔧 技术方法**

使用的技术包括：扩散式顶点生成（Latent Diffusion Model）、语义监督条件编码、Gaussian‑Mixture顶点热图、PSLG过密构造与顶点引导子集选择、以及全局拓扑一致性证明与实现。

**📊 数据集**

数据集方面，创建了Deventer‑512新基准，包含512×512高分辨率航拍图（0.3 m GSD），涵盖建筑、道路、植被、水体、裸地五类地物，提供全局拓扑一致的多边形注释；同时在单类建筑数据集WHU‑Building上验证模型的通用性。

**📈 对比分析**

与多种单类多边形化基线（DeepSnake、FFL、TopDiG、HiSup、GCP）及公开语义分割基线对比，ACPV-Net在Deventer‑512上实现零间隙/重叠、100%共享边缘，且在语义精度、几何精度、顶点效率、拓扑保真度等指标上均显著优于对手；在WHU‑Building上同样取得最优成绩。

**⚠️ 局限性**

局限性包括：对极端光照、遮挡等复杂场景的鲁棒性仍有限；过密PSLG构造在大分辨率图像下计算量较大；模型在跨地区、不同传感器时可能需要微调；并且对非常细小或稀疏地物类的分割仍有提升空间。

---

## 397. Unified Removal of Raindrops and Reflections: A New Benchmark and A Novel Pipeline

**arXiv ID:** 2603.16446 | [PDF](https://arxiv.org/pdf/2603.16446v1)

**作者:** Xingyu Liu `[一作]` (Zhejiang University), Zhe-Ming Lu `[通讯]` (Zhejiang University)

**通讯引用:** 5735 | [OpenAlex ID](https://openalex.org/A5083531923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了统一去除雨滴与反射（UR^3）的任务，建立首个含复合降解的真实图像对数据集RDRF，并设计了两阶段扩散框架DiffUR^3以一次性恢复图像

**💡 创新点**

1）首次定义UR^3任务并提供标注数据；2）设计多条件控制的扩散生成网络，包含Modulate&Gate模块与Fidelity Encoder，能够融合原始图像与初步恢复结果；3）利用扩散模型的生成先验实现高质量恢复

**🔧 技术方法**

基于Stable Diffusion的扩散模型，配合VAE编码/解码；Modulate&Gate模块实现条件调制与门控；额外的Fidelity Encoder校正VAE压缩失真；色彩归一化后处理；两阶段训练：先用DRSformer进行初步恢复，再用扩散生成细化

**📊 数据集**

RDRF（RainDrop and ReFlection）数据集：307个场景，216场景训练集（9003对）与91场景测试集（277对）；此外在公开雨滴/反射单一降解数据集上进行泛化测试；在真实野外拍摄的RDRF-wild数据集上验证鲁棒性

**📈 对比分析**

与多类基线（单一降解方法、级联方法、全局一体化方法）以及在RDRF上重新训练的全局方法进行对比；在RDRF-test上使用PSNR、SSIM、LPIPS和三种无参考指标（MUSIQ、CLIPIQA+、HyperIQA）评估；DiffUR^3在所有指标上均位列首位，PSNR最高29.41dB，SSIM 0.9372，LPIPS 0.0813；用户研究中被专家评价为最优

**⚠️ 局限性**

对单一降解场景的适用性未完全验证；扩散采样步骤较多（50步）导致推理时间较长；在极端反射强度或雨滴密度极高时仍可能出现细节失真或颜色漂移；缺少针对不同摄像机/玻璃厚度的跨域泛化评估

---

## 398. Mixture of Style Experts for Diverse Image Stylization

**arXiv ID:** 2603.16649 | [PDF](https://arxiv.org/pdf/2603.16649v1)

**作者:** Shihao Zhu `[一作]` (Nankai University), Qibin Hou `[通讯]` (Nankai University)

**通讯引用:** 17716 | [OpenAlex ID](https://openalex.org/A5040392623)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 StyleExpert，一种基于 Mixture of Experts (MoE) 的语义感知扩散式风格迁移框架，并构建了 500k 内容‑风格‑渲染三元组数据集。

**💡 创新点**

创新点包括：① 用 InfoNCE 预训练风格编码器为 MoE 路由器提供语义先验；② 采用多专家 LoRA 结合共享专家实现对多层次风格的动态选择；③ 构建平衡色彩与语义的 500k 数据集，显著提升语义迁移效果。

**🔧 技术方法**

技术手段主要包括：Diffusion Transformer (Flux‑Kontext) 作为基础模型，InfoNCE 对抗式风格编码器，MoE+LoRA 结构，CLIP 与 Qwen‑VL 进行数据筛选与质量评估。

**📊 数据集**

使用自建的 500k 内容‑风格‑渲染三元组数据集（-500K），并从中抽取 40k 高质量三元组（-40K）作为最终评测集。

**📈 对比分析**

在 188 训练风格与 21 测试风格上，与 OmniStyle、CSGO、DreamO、Qwen‑Image‑Edit、OmniGen2 等现有方法对比，StyleExpert 在 CLIP、CSD、Aesthetic、Qwen Semantic、DreamSim 等指标上均取得领先（例如 Qwen Semantic 75.12% 远超对手），同时保持较高的内容保真度。

**⚠️ 局限性**

局限性包括：仍需大规模高质量数据进行预训练，MoE 训练过程对路由器初始信号依赖较大；在极度新颖或极端纹理的风格上，语义迁移效果仍可能不足；模型参数相对较多，部署成本提升。

---

## 399. Prompts Blend Requirements and Solutions: From Intent to Implementation

**arXiv ID:** 2603.16348 | [PDF](https://arxiv.org/pdf/2603.16348v1)

**作者:** Shalini Chakraborty `[一作]` (University of Bayreuth), Jan-Philipp Steghöfer `[通讯]` (XITASO GmbH IT and Software Solutions)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证“Prompt Triangle”模型，将提示拆分为功能与质量、通用方案与具体方案三部分，并基于DevGPT数据探讨其结构与演化；

**💡 创新点**

首次将提示视为轻量化、可演化的需求工件，构建了需求与解决方案共生的三维框架；

**🔧 技术方法**

利用LLM（Claude Sonnet 4.5）进行提示语义拆分，结合文本统计与模式分析；

**📊 数据集**

DevGPT对话数据集（约120条初始提示）以及公开博客提示示例；

**📈 对比分析**

通过手工验证LLM分类结果，统计功能/质量、通用方案、具体方案在提示中的出现频率（功能98.3%、通用76.7%、具体63.3%），未给出代码质量指标；

**⚠️ 局限性**

仅分析初始提示，缺乏对多轮对话演化的实证；数据规模有限，LLM拆分的准确性受限，且未验证模型在不同LLM或开发环境中的泛化性。

---

## 400. Surg$Σ$: A Spectrum of Large-Scale Multimodal Data and Foundation Models for Surgical Intelligence

**arXiv ID:** 2603.16822 | [PDF](https://arxiv.org/pdf/2603.16822v1)

**作者:** Zhitao Zeng `[一作]` (National University of Singapore), Yueming Jin `[通讯]` (National University of Singapore)

**通讯引用:** 4911 | [OpenAlex ID](https://openalex.org/A5050163233)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SurgΣ-DB 多模态数据基础和一系列基础模型，旨在提升外科智能的跨任务泛化与可解释性。

**💡 创新点**

核心创新包括统一的大规模多模态数据架构、统一标签空间与标准化格式、层级推理（Chain-of-Thought）注释，以及基于此构建的多任务基础模型。

**🔧 技术方法**

采用半自动标注流水线、专家与 AI 联合标注、结构化推理生成、LLM（Qwen2.5‑VL、Qwen3‑VL、GPT‑5.1）对齐与微调、逆动力学生成等技术。

**📊 数据集**

使用 SurgΣ-DB（约 5.98M 对话、6 专科 16 手术类型），并整合公开数据集如 Cholec80、EndoVis2018、PSI‑AVA、SurgPub‑Video 等。

**📈 对比分析**

在 SurgVLM‑Bench、Triplet 识别、Phase 识别等基准上，相较现有方法提升 15%–30% 以上，BSA 在多中心外部验证上平均准确率达 60%，Cosmos‑H‑Surgical 在机器人策略学习上样本效率提升约 2–3 倍。

**⚠️ 局限性**

仍存在标注覆盖不完整、推理注释不均匀、手术场景动态复杂导致标注成本高、在极端条件下泛化仍有限等限制。

---

## 401. Proactive Rejection and Grounded Execution: A Dual-Stage Intent Analysis Paradigm for Safe and Efficient AIoT Smart Homes

**arXiv ID:** 2603.16207 | [PDF](https://arxiv.org/pdf/2603.16207v1)

**作者:** Xinxin Jin `[一作]` (Zhejiang Gongshang University), Victor C. M. Leung `[通讯]` (Shenzhen MSU-BIT University)

**通讯引用:** 66181 | [OpenAlex ID](https://openalex.org/A5035919267)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Dual-Stage Intent-Aware (DS-IA) 框架，实现智能家居指令的安全高效执行。

**💡 创新点**

创新点在于将宏观意图分析与微观物理验证分离，采用语义防火墙和三层级级联验证，并通过 Generate‑and‑Filter 策略避免实体幻觉与交互频率困境。

**🔧 技术方法**

使用大语言模型（GPT‑4o‑mini / Qwen‑2.5‑7B）、意图分析模块、Cascade 验证器、结构化提示工程及 Generate‑and‑Filter 生成过滤技术。

**📊 数据集**

使用 HomeBench（智能家居指令执行基准）和 SAGE Benchmark（人机交互效率评测）数据集。

**📈 对比分析**

与 Baseline 与 SAGE 进行对比，DS-IA 在 HomeBench 上 Exact Match 提升至 58.56%（Baseline 29.98%，SAGE 1.77%），Invalid 单指令拒绝率 87.04%；在 SAGE 上自主成功率从 42.86% 提升至 71.43%，澄清成功率保持 75%。

**⚠️ 局限性**

局限性包括：仅基于文本环境状态，缺乏多模态感知；需要在设备隐私与推理延迟方面进一步通过小模型蒸馏优化；个性化记忆与偏好推断仍待完善。

---

## 402. UMO: Unified In-Context Learning Unlocks Motion Foundation Model Priors

**arXiv ID:** 2603.15975 | [PDF](https://arxiv.org/pdf/2603.15975v1)

**作者:** Xiaoyan Cong `[一作]` (Brown University), Srinath Sridhar `[通讯]` (Brown University)

**通讯引用:** 2944 | [OpenAlex ID](https://openalex.org/A5014973945)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种统一的上下文学习框架，将预训练的文本到运动基础模型（如 HY‑Motion）通过三种原子帧级操作（保持、生成、编辑）以及轻量级时间融合，扩展到多种下游运动生成任务（时间填充、文本编辑、几何约束、双身份反应等）

**💡 创新点**

创新点在于：①通过仅三种可学习的帧级 meta‑operation embedding 统一表述所有任务的帧级意图；②利用轻量化时间融合将上下文注入现有的 DiT backbone，无需改动网络结构；③所有条件均通过同一 LLM 编码的自然语言完成，省去了任务特定的条件模块

**🔧 技术方法**

核心技术包括：DiT‑based 运动基础模型 HY‑Motion、Flow Matching 训练、三维运动表示、可学习的 meta‑operation embeddings、Temporal Fusion 注入、统一语言条件（Qwen3‑8B LLM）以及对齐和微调策略

**📊 数据集**

使用的数据集包括：HumanML3D（文本‑运动），MotionFix（编辑指令），Inter‑X / InterHuman（双身份反应），以及自建的 2000 条参数化轨迹与障碍规避序列；训练与评估均覆盖这些数据

**📈 对比分析**

在各任务上与专用方法（如 CondMDI、MotionLab、ControlNet 等）以及无任务特定的训练‑自由基线进行对比，采用 FID、R‑Precision、MM‑Dist、MPJPE 等指标，统一模型在文本生成、时间填充、编辑、几何约束和双身份反应等任务均实现或接近最优性能，显著优于单任务或无条件基线

**⚠️ 局限性**

局限性包括：① meta‑operation embedding 仅作用于整个人体，无法细粒度部件级控制；②目前仅支持语言条件，未覆盖音频、音乐等多模态输入；③对多人物交互的迁移基于单人先验，仍需进一步探索更复杂的多主体场景

---

## 403. PulmoVec: A Two-Stage Stacking Meta-Learning Architecture Built on the HeAR Foundation Model for Multi-Task Classification of Pediatric Respiratory Sounds

**arXiv ID:** 2603.15688 | [PDF](https://arxiv.org/pdf/2603.15688v1)

**作者:** Izzet Turkalp Akbasli `[一作]` (Hacettepe University), Oguzhan Serin `[通讯]` (Hacettepe University)

**通讯引用:** 62 | [OpenAlex ID](https://openalex.org/A5007589912)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建PulmoVec多任务框架，将基于HeAR基础音频模型的声学特征与三类任务（异常筛查、声纹识别、疾病分组预测）相结合，并通过LightGBM堆叠与投票聚合实现从事件级到患者级的决策支持。

**💡 创新点**

创新点在于：①使用大规模自监督预训练的HeAR基础模型实现跨设备声学表征；②设计多任务堆叠结构，将基模型概率与人口学元数据融合，提升预测性能；③提出多策略投票聚合，实现事件级到患者级的无缝迁移。

**🔧 技术方法**

技术包括HeAR音频表征、PyTorch实现的三类基分类器、LightGBM堆叠元学习器、基于概率的多策略投票聚合以及数据增强与类别权重平衡。

**📊 数据集**

使用公开的SPRSound扩展版数据集，共1652名儿童患者、24,808段事件级标注，涵盖正常、细湿啰音、粗湿啰音、喘鸣、喘鸣+湿啰音、咕噜音、咳音等多种标签。

**📈 对比分析**

通过将基模型与堆叠模型在事件级和患者级分别评估，发现堆叠模型相较于单一基模型提升了9–12个百分点，事件级异常筛查ROC‑AUC达0.96，疾病分组患者级宏观AUC为0.91；相对传统单任务CNN，PulmoVec显著提高了分类精度与概率校准。

**⚠️ 局限性**

主要局限包括：仅在单中心单设备数据上验证，缺乏外部多中心与跨设备验证；患者级聚合未考虑时间/空间序列信息，导致部分正常类别召回低；稀缺疾病标签的类别不平衡仍未充分解决；噪声鲁棒性未在非临床环境中测试。

---

## 404. Decoding the Critique Mechanism in Large Reasoning Models

**arXiv ID:** 2603.16331 | [PDF](https://arxiv.org/pdf/2603.16331v1)

**作者:** Hoang Phan `[一作]` (VinUniversity), Khoa D. Doan `[通讯]` (VinUniversity)

**通讯引用:** 339 | [OpenAlex ID](https://openalex.org/A5080642445)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

调查大型推理模型在链式思维中隐藏的自我纠错机制，并提出利用激活空间中的“批评向量”来调节错误检测与修正。

**💡 创新点**

发现并解码隐藏的批评能力，将其表征为单一线性方向，并证明通过该向量可控制模型的错误检测和自我纠错。

**🔧 技术方法**

通过对抗性注入算术错误、特征空间线性探测、logit lens 分析以及激活向量调节（steering）等技术实现。

**📊 数据集**

使用 GSM8K-Error、MATH500-Error、ProcessBench、BIG-Bench-Mistake 等数学与推理基准。

**📈 对比分析**

对比不同规模模型（DeepSeek-R1 8B/14B/32B、Qwen3-4B）在错误检测和最终答案准确率上的提升，正向调节提升 5–10% 误差检测，负向调节降低；在测试时扩展（test-time scaling）中可提升至约 90% 的准确率。

**⚠️ 局限性**

机制仅在注入错误时激活，对原始样本效果有限；调节可能导致正确答案准确率下降；缺乏对该能力来源（预训练 vs 微调）的深入解析。

---

## 405. Encoding Predictability and Legibility for Style-Conditioned Diffusion Policy

**arXiv ID:** 2603.16368 | [PDF](https://arxiv.org/pdf/2603.16368v1)

**作者:** Adrien Jacquet Crétides `[一作]` (Sorbonne Université), Mohamed Chetouani `[通讯]` (Sorbonne Université)

**通讯引用:** 7810 | [OpenAlex ID](https://openalex.org/A5049398785)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种 Style‑Conditioned Diffusion Policy (SCDP)，通过在预训练扩散策略后添加轻量化的场景编码器和风格预测器，使机器人能够在不同空间模糊度的环境下自适应生成既可解释又高效的轨迹。

**💡 创新点**

创新点包括：1）post‑training 方式冻结基础扩散模型，仅训练外部 MLP，避免了重新训练；2）通过场景编码器生成上下文向量，再用 FiLM 对 U‑Net 进行风格条件化；3）利用几何椭圆做空间模糊检测，自动切换腿性或可预测性模式，实现环境感知的动态调节。

**🔧 技术方法**

技术栈主要包括：扩散政策（DDPM‑U‑Net）、FiLM 条件化、场景编码器 MLP、风格预测器 MLP、空间模糊检测椭圆、后训练轻量化模块，以及在真实机器人上的 YOLO‑RealSense 视觉感知。

**📊 数据集**

数据集：使用 200 条手工生成的演示（Bezier 曲线）分别用于 Block Reach（Franka Panda）和 Navigation（Turtlebot）任务，所有演示均在 Gazebo 模拟环境中采集；真实实验使用 YOLO 检测 + RealSense 深度图实现目标定位。

**📈 对比分析**

评价方法：与基线 Diffusion Policy 及 Legibility Diffuser 进行对比，指标包括 detachment、trajectory efficiency 以及自适应透明度得分（T）。实验显示：在模糊场景下 SCDP 与 Legibility Diffuser 取得相近的腿性表现；在非模糊场景中，SCDP 的自适应透明度得分显著高于两者（Block Reach: 0.74 vs 0.61/0.52；Navigation: 0.76 vs 0.62/0.50），并保持 98% 以上的成功率。

**⚠️ 局限性**

局限性：1）SCDP 仅能在训练集分布内生成轨迹，无法产生比训练样本更具可解释性的路径；2）空间模糊检测采用简单的几何椭圆，可能在复杂环境中误判；3）实验仅覆盖两种任务且目标数有限，尚未验证在更多目标或更大规模环境中的可扩展性。

---

## 406. Ember: A Serverless Peer-to-Peer End-to-End Encrypted Messaging System over an IPv6 Mesh Network

**arXiv ID:** 2603.16735 | [PDF](https://arxiv.org/pdf/2603.16735v1)

**作者:** Hamish Alsop `[一作]` (Edinburgh Napier University), Naghmeh Moradpoor `[通讯]` (Edinburgh Napier University)

**通讯引用:** 608 | [OpenAlex ID](https://openalex.org/A5082362331)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

开发并实现了基于Android的去中心化、无服务器端点对点加密聊天系统Ember，该系统利用Yggdrasil IPv6覆盖网络进行直接连接，并通过对称密钥+AES‑GCM、HMAC、HKDF实现消息加密、完整性校验与键轮转；采用加密SQLite（SQLCipher）实现仅存储密文、TTL自动删除，前台服务保证离线接收，所有传输均在端到端加密保护之下。

**💡 创新点**

① 彻底消除中央服务器与推送通知的依赖，最大化对中心化监管与平台隐私泄露的抵抗；② 将“端到端加密+无服务器”结合到移动端，证明在资源受限的Android环境下可行；③ 明确划分安全边界、使用显式HMAC校验与加密数据库，提供可审计的最小化数据留存；④ 通过关键轮转协议降低长期密钥暴露窗口；⑤ 对Yggdrasil作为安全传输层的可行性与局限进行系统验证。

**🔧 技术方法**

技术层面主要包括：
- Yggdrasil加密IPv6覆盖网络
- AES‑256‑GCM（AEAD）+ HMAC‑SHA256（预验证）
- HKDF‑SHA256（键轮转）
- SQLCipher（加密SQLite）与 Room
- Android 前台 Service + WorkManager（TTL清理）
- Jetpack Compose + ViewModel（UI隔离）

**📊 数据集**

该工作未使用公开数据集；所有评测均基于自建的单机/多机对等测试环境与Android内部日志/网络抓包，重点验证加密正确性、键管理与存储一致性。

**📈 对比分析**

评估方法：
- 静态代码分析（Kotlin/Android）
- 单元测试验证加密/轮转/HMAC
- ASVS v4.0 需求映射
- 结构化威胁建模与风险评估
- 动态网络抓包与运行时日志对比
性能方面通过单元测试微基准测量加密、解密与数据库写入延迟；结果表明在常见Android设备上消息发送/接收延迟可控（<100 ms）。

**⚠️ 局限性**

主要局限：
- 缺乏前向保密与后向兼容性（未实现 Double Ratchet）
- 只支持一对一会话，群组与多设备一致性未实现
- 依赖前台服务，功耗与系统限制导致可用性不稳定
- 无覆盖流量或混淆，网络级流量分析仍可泄露元数据
- TTL清理仅为应用层删除，无法保证物理磁盘擦除
- 受限于Android系统权限与硬件密钥存储实现，根级攻击仍能获取密钥或解密信息

---

## 407. Bayesian Inference of Psychometric Variables From Brain and Behavior in Implicit Association Tests

**arXiv ID:** 2603.16741 | [PDF](https://arxiv.org/pdf/2603.16741v1)

**作者:** Christian A. Kothe `[一作]`, Alik S. Widge `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种稀疏层级贝叶斯框架，用多模态（EEG、眼动、面部动作、时间导数）IAT数据推断心理健康相关的二元心理测量变量；

**💡 创新点**

创新点在于将D-score的对比结构融入贝叶斯稀疏学习，结合EEG源估计的Dugh框架与Haufe变换，实现对高维EEG和行为时序数据的参数高效、可解释性强的解码；

**🔧 技术方法**

使用稀疏贝叶斯学习、组稀疏与高斯随机游走先验、Haufe变换、随机变分推断、Platt式置信度校准及多模态线性组合；

**📊 数据集**

实验数据来自两种IAT变体：E‑IAT（情绪“陷阱”与抑郁组）和PSY‑IAT（精神病相关词与对照组），分别收集EEG、眼动、面部动作和时间导数；

**📈 对比分析**

与传统D‑score、L2正则化逻辑回归、sLDA、EEGNet等基线方法对比，最佳多模态组合在E‑IAT和PSY‑IAT上分别取得AUC≈0.73与0.76，超越基线且在两任务上表现更为一致；

**⚠️ 局限性**

局限包括样本量有限、单中心数据、E‑IAT依赖新兴EMA测量、EEG源估计缺乏个体化MRI、以及对不同任务和硬件的泛化尚未验证。

---

## 408. IdentityGuard: Context-Aware Restriction and Provenance for Personalized Synthesis

**arXiv ID:** 2603.15679 | [PDF](https://arxiv.org/pdf/2603.15679v1)

**作者:** Lingyun Zhang `[一作]` (Fudan University), Ping Chen `[通讯]` (Fudan University)

**通讯引用:** 30219 | [OpenAlex ID](https://openalex.org/A5061915249)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 IDENTITYGUARD 的个性化文本到图像模型安全框架，能够在仅当使用个性化身份与违规词组合时才限制生成，并为生成的图像嵌入可追踪的概念特定水印。

**💡 创新点**

核心创新在于将安全机制与个性化概念绑定：通过条件语义重定向（CIP）实现对恶意提示的上下文感知限制，并在个性化身份出现时激活概念专属水印，避免了传统全局过滤导致的功能损失。

**🔧 技术方法**

技术包括：在 DreamBooth 微调中加入两条条件训练路径，使用语义重定向损失（CIP Loss）和水印损失（WM Loss），以及冻结的水印解码器和停梯度操作实现精细控制。

**📊 数据集**

使用 Stable Diffusion v2.1 预训练模型，并在其上进行 DreamBooth 微调；评估使用公开的安全检测器（如 NudeNet）和标准图像质量指标（FID、CLIP Score）。

**📈 对比分析**

与基线方法（Safe Latent Diffusion、Erasing Concepts、HiDDeN）相比，IDENTITYGUARD 在保留原始 FID（≈54–58）和 CLIP Score 的同时，显著提升了恶意提示下的 CLIP‑Censored（0.1919 对比 0.2378）并将 nudity 检测从 342 降至 2，水印位准确率达到 97.1%。

**⚠️ 局限性**

局限性包括仅在有限的概念和基线上验证，缺乏对更大规模生成模型和更复杂黑名单定义的评估；未来需扩展到多种生成架构和开放式安全策略。

---

## 409. World Reconstruction From Inconsistent Views

**arXiv ID:** 2603.16736 | [PDF](https://arxiv.org/pdf/2603.16736v1)

**作者:** Lukas Höllein `[一作]` (Technical University of Munich), Matthias Nießner `[通讯]` (Technical University of Munich)

**通讯引用:** 23159 | [OpenAlex ID](https://openalex.org/A5088583491)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种从视频扩散模型生成的视角不一致的视频序列中重建 3D 世界的方法，先用非刚性 ICP 对点云进行对齐，再通过非刚性感知的高斯展开优化得到可实时渲染的 3D 场景。

**💡 创新点**

创新点在于：① 针对视频生成的不一致性设计了两阶段非刚性对齐（迭代 frame‑to‑model ICP 与全局优化）；② 利用反向非刚性变形的渲染损失，使 3D Gaussian Splatting 能在不一致视角下训练出一致的 3D 表现；③ 该方法可无须对扩散模型进行额外微调，直接将任何 VDM 转化为 3D 世界生成器。

**🔧 技术方法**

核心技术包括：几何基础模型（如 DepthAnything‑3）生成稠密点云；非刚性迭代 ICP（带点对平面、颜色、稀疏对应、ARAP 正则化）；全局非刚性优化；逆向非刚性变形网络；以及基于 2D Gaussian 的高斯展开（Gaussian Splatting）渲染。

**📊 数据集**

使用多种最新视频扩散模型生成的合成视频数据集：Wan‑2.2、ViewCrafter、Gen3C、Seva、Voyager、Genie3、HY‑WorldPlay 等。对每个视频采样 50 张帧进行实验。

**📈 对比分析**

与 3DGS‑MCMC、DA3、VGGT‑X 等基线在一致性、渲染质量和可视化稳定性上进行对比。实验结果显示该方法在 WorldScore 评测中获得最高一致性分数，并在 CLIP‑IQA+、CLIP‑Aesthetic 等视觉质量指标上达到与原始视频相当、优于基线的水平。

**⚠️ 局限性**

局限性包括：① 对生成漂移的处理仍需对每个场景额外计算；② 无法完全解决扩散模型的虚假生成（hallucination）问题；③ 计算成本较高（约 25 分钟/20GB GPU）；④ 需要进一步研究鲁棒的异常帧检测或直接对 VDM 进行微调以减少漂移。

---

## 410. Generative Inverse Design with Abstention via Diagonal Flow Matching

**arXiv ID:** 2603.15925 | [PDF](https://arxiv.org/pdf/2603.15925v1)

**作者:** Miguel de Campos `[一作]` (Institute of Mathematics), Hanno Gottschalk `[通讯]` (Institute of Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种名为 Diagonal Flow Matching (Diag–CFM) 的零锚定条件流匹配方法，用于逆设计与生成满足性能约束的多样化设计，并提出两种只需单模型即可评估的架构特定不确定性度量。

**💡 创新点**

创新点在于通过将标签与零配对、设计与噪声配对的零锚定策略，消除了标准 CFM 对坐标顺序和尺度的敏感性，形成可换位不变的学习问题；同时设计了 Zero‑Deviation 与 Self‑Consistency 两种不依赖集成或额外前向推断的架构特定不确定性指标。

**🔧 技术方法**

使用技术包括：可逆连续正则化流 (Conditional Flow Matching)、流匹配训练、可逆神经网络、零锚定 (Zero‑Anchoring) 机制、单模型不确定性评估（Zero‑Deviation 与 Self‑Consistency）、以及与传统的集成（Ensemble）、流匹配损失 (FM Loss) 等基线方法对比。

**📊 数据集**

实验数据集涵盖三种工程问题：气体涡轮燃烧器（P=6）、Unifoil 空气翼（P=14）以及可扩展的多目标基准 DTLZ2（P=12、24、50、100），并使用对应的 CFD/仿真或解析前向映射进行评估。

**📈 对比分析**

与标准 CFM、可逆神经网络（INN）在前向 MSE、回程误差和设计多样性等指标上进行 5 次随机初始化比较，Diag–CFM 在回程误差上提升 1–2 个数量级、前向误差降低 10 倍、设计多样性保持不变；在不确定性评估任务（最佳选取、误差拒绝、OOD 检测）上，Zero‑Deviation 与 Self‑Consistency 的表现优于集成、FM Loss 等基线。

**⚠️ 局限性**

限制主要在于：不确定性指标与理论最优（oracle）相比仍存在显著差距；在高维前向预测（P≥50）时 INN 的性能可略优于 Diag–CFM；以及在更大规模的拓扑优化或分子设计等高维逆问题中还需进一步验证。

---

## 411. Controlling Fish Schools via Reinforcement Learning of Virtual Fish Movement

**arXiv ID:** 2603.16384 | [PDF](https://arxiv.org/pdf/2603.16384v1)

**作者:** Yusuke Nishii `[一作]` (Kyoto University), Hiroaki Kawashima `[通讯]` (Kyoto University)

**通讯引用:** 7864 | [OpenAlex ID](https://openalex.org/A5027545207)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用无模型强化学习（Q‑learning）训练2D虚拟鱼的运动策略，以在屏幕上显示虚拟鱼并通过视觉刺激引导真实鱼群向指定方向移动。

**💡 创新点**

创新点在于：①利用可持续的虚拟鱼而非物理机器人克服硬件限制；②在缺乏真实鱼行为模型的情况下采用无模型RL学习策略；③将模拟学习的Q值迁移到真实实验，显著缩短训练时间。

**🔧 技术方法**

采用的技术包括：Q‑learning、ε‑greedy探索、图像背景减除与SVM分类、光学坐标转换、基于中心点的状态表示以及基于细胞划分的动作空间。

**📊 数据集**

使用的数据集为：Rummy‑nose tetra（Hemigrammus bleheri）鱼群在实验槽中的实时影像；以及在仿真环境中生成的随机运动轨迹，用来评估RL在不同忽视概率下的鲁棒性。

**📈 对比分析**

比较方法：与“边缘保持”（stay‑at‑edge）和“无刺激”两种基准策略进行对比，分别计算均值差异、t‑检验p值和Bhattacharyya距离。实验结果显示：学习策略在左右引导下均实现显著的方向偏移（p < 10⁻⁴），平均位置差距和分布差距（Bhattacharyya距离）均超过两种基准，证明RL策略优于人工定义策略。

**⚠️ 局限性**

局限性：①仅在小规模鱼群（3–4只）和单一种类鱼上验证；②状态仅用鱼群中心点，未包含速度或个体间关系；③动作空间限定为单一虚拟鱼的细胞位移，未探讨多鱼独立运动；④使用纹理映射的虚拟鱼，未评估抽象刺激或不同纹理对鱼行为的影响；⑤缺乏对大规模群体或实际机器人实现的进一步验证。

---

## 412. GSI Agent: Domain Knowledge Enhancement for Large Language Models in Green Stormwater Infrastructure

**arXiv ID:** 2603.15643 | [PDF](https://arxiv.org/pdf/2603.15643v1)

**作者:** Shaohuang Wang `[一作]` `[通讯]`, Shaohuang Wang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个结合监督微调、检索增强生成和Agent推理的域知识增强LLM框架，用于绿色雨水基础设施（GSI）的检查与维护任务。

**💡 创新点**

创新点在于将三种知识注入策略（静态LoRA微调、动态RAG检索、基于Agent的推理）融合，既能让模型掌握专业术语与流程，又能实时检索最新法规，兼顾通用知识与域特定准确性。

**🔧 技术方法**

使用LoRA参数高效微调、RAG检索-生成、ReAct/Agent推理、BLEU/ROUGE/句子BERT/G‑Eval等评测方法。

**📊 数据集**

使用自研的GSI指令式数据集（约11,000条实例）和通用Benchmark数据集（约5,000条实例）。

**📈 对比分析**

与Base LLM、Base+RAG、Fine‑tuned+RAG等基线对比，BLEU‑4从0.090提升至0.307，通用数据集性能保持稳定（0.304→0.305），G‑Eval最高达0.72。

**⚠️ 局限性**

局限包括：仅针对文本推理，图像处理功能有限；检索效果受索引质量影响；缺乏大规模人工评估，可能存在误差累积。

---

## 413. Rotatable Antenna-Enabled Mobile Edge Computing

**arXiv ID:** 2603.16275 | [PDF](https://arxiv.org/pdf/2603.16275v1)

**作者:** Qiyao Wang `[一作]` (South China University of Technology), Jie Tang `[通讯]` (South China University of Technology)

**通讯引用:** 29011 | [OpenAlex ID](https://openalex.org/A5044791875)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在论文中，作者研究了一种可旋转天线（RA）支持的移动边缘计算（MEC）上行系统，通过联合优化RA的指向角、接收波束成形以及边缘服务器的计算资源分配，以最小化系统中最大计算延迟。

**💡 创新点**

创新点主要体现在：①首次将可旋转天线引入MEC场景，利用天线指向角提供额外空间自由度；②提出了交替优化（AO）框架，将计算资源分配、波束成形与天线指向三者分块优化，并分别采用KKT、半正定松弛（SDR）+二分搜索、分数规划（FP）+SCA等技术实现全局可达的高质量解；③在RA指向优化中引入二次变换与SCA，克服非凸SINR约束。

**🔧 技术方法**

主要使用的技术包括：可旋转天线（RA）模型、移动边缘计算（MEC）架构、交替优化（AO）方法、Karush–Kuhn–Tucker（KKT）条件、半正定松弛（SDR）+二分搜索、分数规划（FP）+迭代算法、二次变换与成功凸近似（SCA）、高斯随机化。

**📊 数据集**

实验使用的是仿真数据：在半圆区域随机布置K个设备，设定N=9个RA，频带2 MHz、功率-噪声比-60 dBm等参数；未使用公开真实数据集。

**📈 对比分析**

与三种基准方案（固定方向天线、等效全向天线、随机天线指向）进行对比。结果显示，在中等传输功率、适中边缘计算资源下，RA-enabled方案能够显著降低最大计算延迟；随着功率升高或计算资源饱和，差距减小，但RA方案始终保持优越。

**⚠️ 局限性**

局限性包括：仅考虑上行通信，未考虑下行或多链路；未考虑能耗与功率预算约束；RA仅调节仰角，未充分利用全方向旋转；算法复杂度高，适用规模有限；缺乏真实网络环境验证。

---

## 414. Emotion-Aware Classroom Quality Assessment Leveraging IoT-Based Real-Time Student Monitoring

**arXiv ID:** 2603.16719 | [PDF](https://arxiv.org/pdf/2603.16719v1)

**作者:** Hai Nguyen `[一作]` (Posts and Telecommunications Institute of Technology), Cong Tran `[通讯]` (Posts and Telecommunications Institute of Technology)

**通讯引用:** 379 | [OpenAlex ID](https://openalex.org/A5067894717)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文设计并验证了一个低功耗 IoT 边缘设备实现的实时多学生情绪识别与课堂互动评估框架。

**💡 创新点**

创新点包括发布首个“Classroom Emotion”真实课堂多人的情绪数据集，结合轻量级检测+MobileNetV2情绪分类，使用变差正则化和 EMA 时序平滑的课堂质量评分模型。

**🔧 技术方法**

采用 Ultra-Light-Fast-Generic-Face-Detector (ULFG) 进行人脸检测，MobileNetV2 做表情分类，并在 NVIDIA Jetson TX2 上实现端到端推理；课堂质量通过加权情绪频率、EMA 平滑和阈值分割。

**📊 数据集**

使用自采集的 Classroom Emotion 数据集，包括 1500 张标注图片、300 条课堂检测视频，覆盖 3 所学校的 30% 小学、40% 中学、30% 高中学生，共 1-18 岁。

**📈 对比分析**

与传统 RetinaFace/ResNet50 等模型对比，ULFG+MobileNetV2 在 Jetson TX2 上实现 25 FPS、面部检测 mAP 0.928、情绪分类 84.3%；课堂状态分类准确率 88%，误差率下降 67%。

**⚠️ 局限性**

主要局限是数据受城市样本偏倚，情绪识别对高中生表现下降；仅基于视觉，缺乏多模态；隐私与监控伦理仍待完善。

---

## 415. Functorial Neural Architectures from Higher Inductive Types

**arXiv ID:** 2603.16123 | [PDF](https://arxiv.org/pdf/2603.16123v1)

**作者:** Karen Sargsyan `[一作]` (Institute of Chemistry, Academia Sinica), Karen Sargsyan `[通讯]` (Institute of Chemistry, Academia Sinica)

**通讯引用:** 1274 | [OpenAlex ID](https://openalex.org/A5077764895)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了神经网络在组合归纳（compositional generalization）上的系统性失败，并提出将组合归纳等价于解码器的函子性（functoriality），从而通过类型理论（Higher Inductive Types，HIT）编译出保证函子性的神经网络架构。

**💡 创新点**

创新点包括：
1) 将HIT规范与神经网络架构之间建立严格的编译函子，自动将生成元、路径和高阶单元映射为生成器网络、结构拼接和学习的同态变换；
2) 在Cubical Agda中形式化并证明：transport解码器是严格的monoidal函子，而softmax自注意力在任何非平凡组合任务上都不可能是函子；
3) 通过实验验证函子性在三种拓扑空间（T²、S¹∨S¹、Klein瓶）上分别体现绕度约束、单纯的monoidal组合以及学习到的2‑cell证明。

**🔧 技术方法**

使用的技术包括：
- HoTT / HIT 规范化空间的生成元和关系；
- 结构化拼接（transport解码器）与学习的2‑cell（homotopy解码器）；
- Cat. Deep Learning 框架，将参数化映射视为有向范畴中的箭头；
- Cubical Agda 进行形式化验证；
- 采用Chamfer距离和环分割准确率作为评估指标。

**📊 数据集**

使用的数据集为自定义的三维/四维几何循环数据，源自于长度为1、2 的词（训练集）与更长词（测试集）的所有可能组合；每个词对应一个在空间中闭合的路径。空间分别是：
1) 2维圆环（T²，π₁=ℤ²）；
2) 两圆的楔合（S¹∨S¹，π₁=F₂）；
3) Klein瓶（π₁=ℤ⋊ℤ）。

**📈 对比分析**

比较方法：将解码器分为两类：
- Type‑B（函子性）解码器：transport、homotopy；
- Type‑A（非函子性）解码器：Transformer‑WC、Cover、Transport‑Attention、Sequential。评估指标为每段Chamfer距离和在S¹∨S¹上的环分割准确率。实验结果显示：
• 在T²上，Type‑B 的误差稳定在0.77–0.80，而 Type‑A 误差在1.54–2.09之间，提升约2–2.7×；
• 在S¹∨S¹上，误差差距扩大至5.5–10×，且Type‑A 解码器甚至失去对两圆的区分；
• 在Klein瓶上，加入学习的2‑cell（homotopy）能将非标准词的误差从1.52降到0.82，缩小46% 的误差差距。

**⚠️ 局限性**

局限性包括：
- 目前仅处理π₁层次的拓扑约束，无法直接扩展到更高维的π₂、π₃等；
- 实验仅覆盖三种简单空间，缺乏更大基数或更复杂基底的验证；
- 对自然语言等离散符号任务的直接应用仍需进一步形式化组合语义；
- 编译过程需要手动编写HIT规范，尚未实现完全自动化的HIT→网络编译器。

---

## 416. Onboard MuJoCo-based Model Predictive Control for Shipboard Crane with Double-Pendulum Sway Suppression

**arXiv ID:** 2603.16407 | [PDF](https://arxiv.org/pdf/2603.16407v1)

**作者:** Oscar Pang `[一作]` (Imperial College London), Antoine Cully `[通讯]` (Imperial College London)

**通讯引用:** 2379 | [OpenAlex ID](https://openalex.org/A5011747084)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于MuJoCo的实时采样式MPC控制流水线，利用CEM规划在全动力学模型上直接评估动作序列，实现船上升降机的双摆阻振与目标跟踪。

**💡 创新点**

创新点在于：①完全放弃传统线性化/降维模型，直接用高保真物理模拟和采样优化实现控制；②设计自适应成本函数，在接近目标时动态平衡追踪与阻摆；③证明该框架可在资源受限的嵌入式硬件（NVIDIA Jetson AGX Orin）上以约40 Hz实时运行。

**🔧 技术方法**

使用技术包括MuJoCo物理引擎、Cross‑Entropy Method（CEM）采样优化、系统辨识、周期性扰动预测（自相关匹配）、状态估计、ROS接口、边缘计算平台部署。

**📊 数据集**

实验数据来自真实船舶运动平台的周期扰动序列与测距捕捉；RL基线使用相同MuJoCo模型训练；未使用公开数据集，全部为自采集实验数据。

**📈 对比分析**

通过与经典PID和基于PPO的RL基线对比，在静态、慢速、中速和快速海浪扰动下测量位置误差和摆角，结果显示MJPC在所有条件下均保持最低的跟踪误差和摆角，且在Jetson硬件上实现约40 Hz的实时频率。

**⚠️ 局限性**

局限性包括：对周期性扰动预测的依赖，无法及时响应突发非周期扰动；需要外部运动捕捉作为状态输入；对未建模的额外摆动模式（如二摆或旋转摆）时角误差显著增大。

---

## 417. LICA: Layered Image Composition Annotations for Graphic Design Research

**arXiv ID:** 2603.16098 | [PDF](https://arxiv.org/pdf/2603.16098v1)

**作者:** Elad Hirsch `[一作]`, Purvanshi Mehta `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了LICA数据集，收集了1,550,244个多层图形设计布局（含27,261个动画/视频布局），并为每个设计提供完整的层级结构、文本、图像、矢量及动画等元数据信息，支持层级编辑与时序建模。

**💡 创新点**

创新点在于：①大规模、结构化的图形设计数据（1.5M样本，完整层级与每层属性）；②首次加入动画/视频时序注释，将设计视为时序媒介；③多类别、模板变体支持结构化生成与编辑；④相较现有仅数千样本的数据库，提供十倍规模与更细粒度的属性。

**🔧 技术方法**

技术包括自定义渲染引擎将设计导出为JSON层级结构；HTML/CSS渲染；多种组件子类型（Lottie、SVG、图表等）与动画关键帧；自动与人工审核保证质量；利用大语言模型生成设计意图与模板。

**📊 数据集**

主要使用的数据集是LICA本身，并与Magazine、Crello、CGL、PKU-PosterLayout等现有图形设计数据集进行对比。

**📈 对比分析**

与现有数据集相比，LICA在规模、层级完整性、样式元数据和动画支持方面均占优，表格显示LICA提供1.5M样本、完整层级、丰富的样式属性和动画标注，而其他数据集仅数千样本且缺少动画；虽然论文未给出具体模型性能数值，但通过对比强调了LICA在结构化生成、编辑和时序建模任务上的基准潜力。

**⚠️ 局限性**

局限性包括：动画样本仅占总数的1.7%；仍依赖人工审核，可能存在标注错误；缺少更高级的图形元素和跨平台工具的直接兼容性；对多模态评估仍无统一标准，设计质量评估方法尚未成熟。

---

## 418. Evidential Domain Adaptation for Remaining Useful Life Prediction with Incomplete Degradation

**arXiv ID:** 2603.15687 | [PDF](https://arxiv.org/pdf/2603.15687v1)

**作者:** Yubo Hou `[一作]` (Institute for Infocomm Research), Zhenghua Chen `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 13279 | [OpenAlex ID](https://openalex.org/A5080343454)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对在目标域缺失完整退化阶段的剩余寿命预测任务，提出了一种名为EviAdapt的无监督域适应方法。

**💡 创新点**

创新点在于：①将源域与目标域按退化阶段分段，实现阶段级对齐；②利用证据学习估计不确定性，并在相同阶段间对齐不确定性而非直接对齐特征；③结合阶段分段与不确定性对齐的损失，显著缓解退化阶段错位与不确定性差异导致的负迁移。

**🔧 技术方法**

核心技术包括：LSTM特征提取器、证据贝叶斯分位数回归（NIG先验）、基于健康指数的阶段划分、阶段级证据对齐损失（k函数）以及传统的域对齐与无监督学习框架。

**📊 数据集**

实验数据集涵盖：C-MAPSS、N-CMAPSS（DS01/DS02/DS03）以及PHM2010（C1、C4、C6）三大工业设备退化数据集。

**📈 对比分析**

与DDC、Deep Coral、ADARUL、CADA、Cons DANN及源域单独模型等多种基线进行对比；在C-MAPSS上RMSE平均提升约16%，Score平均提升约16%；在N-CMAPSS上RMSE平均提升约30%，Score平均提升约42%；在PHM2010上RMSE平均提升约2%。

**⚠️ 局限性**

主要局限：①仍需完整标注的源域数据；②对阶段划分的假设（健康指数阈值）可能不适用于所有任务；③对量化不确定性所选分位数敏感；④在极端缺失退化阶段时，伪标签生成的误差可能影响对齐效果。

---

## 419. Understanding Cell Fate Decisions with Temporal Attention

**arXiv ID:** 2603.16562 | [PDF](https://arxiv.org/pdf/2603.16562v1)

**作者:** Florian Bürger `[一作]` (University of Cologne), Katarzyna Bozek `[通讯]` (University of Cologne)

**通讯引用:** 1474 | [OpenAlex ID](https://openalex.org/A5078879923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于Transformer的深度学习框架，直接从长时序活细胞摄像记录中预测癌细胞在化疗下的命运，并提供可解释的时间与形态特征分析。

**💡 创新点**

创新点在于：①无需预先提取形态或分子特征，直接处理原始图像序列；②结合注意力与遮蔽实验揭示了预测信息可追溯至事件前10小时的分布差异；③构建了完整的可解释性体系，关联细胞形态、p53信号与死亡/增殖决策。

**🔧 技术方法**

核心技术包括Transformer编码器-解码器结构、视觉注意力机制、时间步遮蔽（temporal masking）与可视化解释工具（如Grad-CAM、时间注意力权重图）。

**📊 数据集**

使用自身实验收集的多时间点化疗处理下的癌细胞单细胞视频数据集，覆盖有丝分裂、凋亡与存活三种命运标签。

**📈 对比分析**

与传统基于手工特征或卷积网络的分类方法相比，模型在平衡准确率上达到0.94，F1分数0.93；通过留一交叉验证和多种基线比较验证了其显著提升。实验结果表明预测可在事件发生前10小时实现可靠判断。

**⚠️ 局限性**

主要局限包括：①对特定化疗剂与细胞系的适用性未做广泛外部验证；②需要高质量长时序视频，限制了在大规模临床或低帧率场景的部署；③模型对计算资源需求较高，解释性方法仍依赖人工分析。

---

## 420. Online Semi-infinite Linear Programming: Efficient Algorithms via Function Approximation

**arXiv ID:** 2603.16200 | [PDF](https://arxiv.org/pdf/2603.16200v1)

**作者:** Yiming Zong `[一作]` (Hong Kong University of Science and Technology), Jiashuo Jiang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1368 | [OpenAlex ID](https://openalex.org/A5101856185)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种用于在线半无穷线性规划（OSILP）的函数逼近框架，并基于此设计了多种对偶梯度/镜像下降算法，最终给出了两阶段加速-细化方法。

**💡 创新点**

创新点主要在于：①使用非负基函数（如高斯核RBF）将无限约束压缩为常数维度q的对偶变量；②通过对偶变量的投影保持非负性，显著降低对偶维度；③在随机输入和随机排列两种模型下，首次实现了仅依赖q的 O(q√T) 与 O((q+qlogT)√T) 监管误差；④在随机输入模型下提出两阶段算法，突破 O(q√T) 传统上界，达到 O(qlogT+q/ε) 的监管误差。

**🔧 技术方法**

采用的技术包括：函数逼近（非负基函数）、对偶梯度下降与镜像下降、随机输入/随机排列分析、Hölder错误边界、加速梯度法、两阶段加速-细化策略。

**📊 数据集**

实验使用人工生成的数据集：三类分布（均匀、正态、柯西）与随机排列数据；约束维数设为 M=2000，时间 horizon 取 T=5000；基函数采用 Gaussian‑kernel RBF，q=10。

**📈 对比分析**

与传统单纯梯度下降（Simple‑GD）对比，实验显示：①在大量约束下 Ours‑log（两阶段算法）维持低监管误差且收敛速度快；②Simple‑GD 的监管误差随约束数增长而显著上升；③所有方法的约束违规率低，Ours‑log 与 Ours‑MD 接近 0，说明约束得到很好满足。

**⚠️ 局限性**

限制：①对偶逼近的效果高度依赖基函数的选取与维度 q；②两阶段算法需要 GPG 与 Hölder 错误边界假设；③实验仅基于模拟数据，缺乏真实世界案例验证；④对敌对或非 i.i.d. 分布的鲁棒性未做理论或实验验证。

---

## 421. Coverage First Next Best View for Inspection of Cluttered Pipe Networks Using Mobile Manipulators

**arXiv ID:** 2603.16471 | [PDF](https://arxiv.org/pdf/2603.16471v1)

**作者:** Joshua Raymond Bettles `[一作]`, Atsushi Yamashita `[通讯]` (University of Tokyo)

**通讯引用:** 15961 | [OpenAlex ID](https://openalex.org/A5047464293)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在未知的受辐射管道网络中，提出一种联合覆盖与重建的下一最佳视角（CFNBV）规划方法，并将随机几何不确定性纳入向量场不等式（SVFIs）以实现概率安全的碰撞规避。

**💡 创新点**

创新点：①将CPP问题重新表述为信息增益（IG）问题，允许与NBV方法无缝集成；②引入SVFIs，将测量不确定性通过置信约束转化为可求解的确定性缓冲约束；③实现一次性完成重建与覆盖，减少多规划器需求。

**🔧 技术方法**

技术手段：基于深度相机（RealSense D455）构建OctoMap；使用RANSAC提取几何平面；基于向量场不等式的约束二次规划控制器；随机采样与加权IG求解CFNBV；离散时间控制与离散化缓冲处理；使用滑动变量确保可行性。

**📊 数据集**

数据集与实验环境：在一个1.5 m³的实验箱内，搭建1至3根管道的随机布置；采用实测深度图和合成辐射传感器模型（球形）进行仿真；实验共9次，成功率在3管场景中为5/9。

**📈 对比分析**

与传统方法对比：未给出明确基线；论文通过实验展示信息增益下降、未知体素比例降低、覆盖率提升等指标；系统能够在约100 Hz频率下完成闭环控制，成功完成全覆盖与重建。

**⚠️ 局限性**

局限性：①缺乏真实辐射传感器，仅使用合成模型；②深度相机的最小测距导致误测，影响平面估计与碰撞规避；③仅处理平面几何原语，无法直接推广到圆柱或复杂形状；④未考虑机器人自身姿态与执行器不确定性；⑤未对已覆盖/占据体素进行重新分类，可能导致误判。

---

## 422. Deep Reinforcement Learning-Assisted Automated Operator Portfolio for Constrained Multi-objective Optimization

**arXiv ID:** 2603.16401 | [PDF](https://arxiv.org/pdf/2603.16401v1)

**作者:** Shuai Shao `[一作]` (Anhui University), Xingyi Zhang `[通讯]` (Anhui University)

**通讯引用:** 18877 | [OpenAlex ID](https://openalex.org/A5028634381)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并实现了一种基于深度强化学习的自动化算子组合策略（CMOEA-AOP），用于在约束多目标优化（CMOP）中动态推荐不同算子（遗传交叉与差分进化）的使用比例，从而在每一代中同时兼顾多种搜索范式。

**💡 创新点**

创新点包括：①将算子选择视为连续动作空间的强化学习问题，首次使用深度确定性策略梯度（DDPG）学习算子组合策略；②构造了包含收敛、分布、可行度和进化阶段四个维度的状态向量；③以全局超体积提升为奖励，直接衡量算子组合对解集质量的长期影响；④通过组合多种算子而非单一算子，显著降低陷入局部最优的风险。

**🔧 技术方法**

技术手段主要有：深度强化学习（DDPG）、深度神经网络（Actor/Critic 及其目标网络）、多目标进化算法（EMCMO等）以及基于状态/奖励的经验回放和批量训练。

**📊 数据集**

实验数据集为三个CMOP基准集合：CF（10个问题，2/3个目标）、LIR-CMOP（14个问题，2/3个目标）和DAS-CMOP（9个问题，2/3个目标），每个问题都采用标准的目标函数和约束条件。

**📈 对比分析**

对比方法包括EMCMO、Bico、AGEMOEA-II、TSTI、DRLOS。通过IGD指标和Wilcoxon秩和检验比较，CMOEA-AOP在33个测试实例中获得最佳结果23次，且在多数实例上显著优于所有基线（p<0.05），同时表现出更快的收敛速度和更均匀的可行解分布。

**⚠️ 局限性**

局限性包括：①强化学习训练需要较多的计算资源和时间，未给出训练时间或硬件成本；②奖励仅使用整体约束违约值，未分别考虑各约束，可能忽略单个约束的细节；③目前仅在约束多目标问题上验证，未探讨对无约束多目标优化的适用性；④算法在极高维或非常复杂的约束空间中仍需进一步评估。

---

## 423. MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale

**arXiv ID:** 2603.15954 | [PDF](https://arxiv.org/pdf/2603.15954v1)

**作者:** Hanxian Huang `[一作]` (Meta AI), Raghuraman Krishnamoorthi `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过硬件感知的剪枝搜索，直接针对移动设备预填充延迟优化，提出了MobileLLM-Flash模型族。

**💡 创新点**

创新点在于结合剪枝式结构搜索与多目标贝叶斯优化，构建了浅宽混合架构并交错使用跳过注意力与全局注意力，实现了低延迟与高准确率的 Pareto 前沿。

**🔧 技术方法**

采用了基于激活能量的剪枝指标、执行在Executorch的4‑bit/8‑bit量化以及多阶段贝叶斯优化（Ax+NEHVI）来搜索和评估模型。

**📊 数据集**

使用了600M令牌校准集进行剪枝，候选模型训练2.6B令牌，最终通过500B令牌继续训练并在800B令牌上进行指令微调。

**📈 对比分析**

与LFM2、Nemotron‑Flash、Qwen3等基线对比，MobileLLM-Flash在350M/650M/1.4B规模下预填充和解码速度分别提升1.8×/1.6×，同时在多项零/少样本推理任务中获得最高平均分。

**⚠️ 局限性**

局限在于仅搜索架构参数而未联合优化训练超参；未包含尚缺乏Executorch支持的新型子二次注意力机制。

---

## 424. Ground Reaction Inertial Poser: Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors

**arXiv ID:** 2603.16233 | [PDF](https://arxiv.org/pdf/2603.16233v1)

**作者:** Ryosuke Hori `[一作]` (Carnegie Mellon University), Kris Kitani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 14684 | [OpenAlex ID](https://openalex.org/A5037322163)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了GRIP，一种仅使用四个可穿戴IMU和足部压力传感器，并结合物理仿真来实现全身运动捕捉的方法。

**💡 创新点**

创新点在于将IMU与足部压力融合，利用KinematicsNet和DynamicsNet两个模块，以及物理驱动的数字双胞胎，实现对全身姿态和轨迹的物理可行性控制；同时引入State Difference中间表示，使控制器能在无全局位置信息的情况下修正误差。

**🔧 技术方法**

采用深度学习的双阶段网络（KinematicsNet + DynamicsNet）、强化学习（PPO）控制策略、物理仿真引擎（Torque‑driven humanoid）以及足部压力与IMU数据融合。

**📊 数据集**

构建并公开了PRISM多模态数据集，包含IMU、足部压力、光学MoCap以及环境模型，共1,275段10秒、约3.5小时的真实动作数据。

**📈 对比分析**

在PRISM、UnderPressure、PSU‑TMM100三个数据集上与PIP、GlobalPose、MobilePoser、FoRM、SolePoser等基线比较，GRIP在MPJPE、PA‑MPJPE、Foot‑Sliding、Foot‑Penetration等指标均优于或相当于其他方法，并在物理一致性上表现更佳。

**⚠️ 局限性**

局限性包括：在高冲量或快速动作中对足部压力模式的拟合不足；缺乏多人的交互或动态物体支持；以及对极端姿态或落地失稳时仍需改进的落地恢复机制。

---

## 425. A Pin-Array Structured Climbing Robot for Stable Locomotion on Steep Rocky Terrain

**arXiv ID:** 2603.16543 | [PDF](https://arxiv.org/pdf/2603.16543v1)

**作者:** Keita Nagaoka `[一作]` (Tohoku University), Kazuya Yoshida `[通讯]` (Tohoku University)

**通讯引用:** 13222 | [OpenAlex ID](https://openalex.org/A5023419492)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文设计并实现了一种配备可通过垂直分裂金属脊与弹性叶片实现被动适应地形的针阵结构抓手的多足攀爬机器人，并在室内外崎岖岩石环境中验证其稳定攀爬能力。

**💡 创新点**

创新点在于：①将针阵抓手的被动垂直运动与水平驱动耦合，利用弹性叶片产生恢复力实现机械锁定；②通过提高针密度与优化弹性元件，显著提升接触概率和抓取力；③采用统计蒙特卡洛方法对针阵抓取力进行建模，揭示单针力和接触数变异是主要不确定性来源。

**🔧 技术方法**

使用的技术包括：3D 打印碳纤维增强聚酰胺/PLA 结构、聚碳酸酯叶片、低刚度压缩弹簧、单轴水平驱动机构、Dynamixel 伺服电机、ROS 2 控制框架以及基于 IMU+编码器的状态估计。

**📊 数据集**

使用的实验数据集：在模拟表面（±30°, ±60°, ±90° 斜率）上测量抓取力（正交与切向），室内斜坡（10°, 20°, 30°）和户外自然岩石环境的爬行距离与时间记录。

**📈 对比分析**

与理论静态平衡模型（θ_theory≈55.3°）及实验测得的抓取力对比，机器人在室内最大倾斜角达到 65°，户外约 0.8 m 运动距离，显示出比单针实验更高的安全裕度；抓取力实验显示平均正交抓取力 34.1 N、切向 21.1 N，蒙特卡洛模拟预测 19.8–62.9 N 区间与实验数据高度吻合。

**⚠️ 局限性**

局限性：①爬行速度受限于被动弹簧对抓取力的影响，尤其在高坡度下需降低垂直运动速度；②对地形的感知完全依赖被动适应，缺乏主动姿态/抓取力调节；③当前抓手质量较大，限制了在更轻量化平台上的集成。

---

## 426. AW-MoE: All-Weather Mixture of Experts for Robust Multi-Modal 3D Object Detection

**arXiv ID:** 2603.16261 | [PDF](https://arxiv.org/pdf/2603.16261v1)

**作者:** Hongwei Lin `[一作]` (Xiamen University), Cheng Wang `[通讯]` (Xiamen University)

**通讯引用:** 26484 | [OpenAlex ID](https://openalex.org/A5100736836)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种名为AW-MoE的多模态3D目标检测框架，通过将Mixture of Experts (MoE) 与图像引导的天气感知路由结合，利用专门针对不同天气的专家网络以及统一双模态增广，显著提升在恶劣天气下的检测鲁棒性。

**💡 创新点**

核心创新点包括：①首次将MoE技术引入3D目标检测，以解决不同天气条件下的数据分布冲突；②设计基于图像的天气感知路由（IWR），实现精准的专家选择；③提出统一双模态增广（UDMA）和天气特定GT采样（WSGTS），保持场景真实性并增强数据多样性；④该框架可无缝扩展到多种基线检测器，兼容性强。

**🔧 技术方法**

主要技术手段：Mixture of Experts（多专家架构）、Image‑guided Weather‑aware Routing、Weather‑Specific Experts、Unified Dual‑Modal Augmentation、LiDAR‑guided Lift‑Splat‑Shoot图像特征提升、置信度加权损失与后处理、专门的训练策略（预训练单专家、共享骨干冻结、逐步细化）等。

**📊 数据集**

使用的主要数据集为真实世界的K‑Radar数据集，涵盖正常、阴天、雾、雨、毛刺、轻雪与大雪等六种恶劣天气场景。

**📈 对比分析**

在K‑Radar上与RTNH、InterFusion、3D‑LRF、L4DR、L4DR‑DA3D等先进方法以及基于点云特征的路由（PFR）进行对比。AW‑MoE在IoU 0.3和0.5下在所有天气条件下平均提升约10%‑15%，在极端天气（雾、雪）提升可达15%+；在不改变显著推理速度（仅增加≈0.5–1.0 FPS）和计算成本的前提下，兼容多种基线实现了性能跃迁。

**⚠️ 局限性**

局限性：①在极端能见度低的天气（如大雾、暴雨）图像特征仍可能不可靠，导致IWR误判；②多专家架构虽然开销小，但在极端天气样本稀缺时仍可能出现专家训练不均衡；③需要额外的天气分类网络和多分支管理，模型复杂度略增；④对极端天气的性能提升虽然显著，但仍无法完全匹配正常天气的最高水平。

---

## 427. Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting

**arXiv ID:** 2603.16129 | [PDF](https://arxiv.org/pdf/2603.16129v1)

**作者:** Da Zhang `[一作]` (Northwestern Polytechnical University), Junyu Gao `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 4276 | [OpenAlex ID](https://openalex.org/A5001848378)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种用于零样本目标计数（Zero-shot Object Counting, ZSOC）的框架 QICA，能在仅给定文本类别描述时精确计数。

**💡 创新点**

创新点：① 通过数值条件的协同提示策略（Synergistic Prompting Strategy, SPS）实现视觉与语言编码器的双向共同适配；② 在相似度图上直接操作的成本聚合解码器（Cost Aggregation Decoder, CAD）消除特征空间畸变并提升空间敏感度；③ 多层数量对齐损失（Multi-level Quantity Alignment Loss, ℒ_MQA）在编码器与解码器层面强化数量一致性。

**🔧 技术方法**

使用技术：预训练的大规模视觉语言模型 CLIP（ViT-B/16、ViT-L/14）作为冻结编码器；可学习的数值条件提示词；相似度图聚合 + Swin Transformer 的空间聚合模块；多量化假设与排序约束的损失函数。

**📊 数据集**

使用数据集：FSC-147（基准），CARPK 与 ShanghaiTech-A（跨域泛化验证）。

**📈 对比分析**

与现有 SOTA 方法比较：在 FSC-147 上 QICA 的 MAE/ RMSE 分别为 13.05 / 104.17（ViT-B/16）和 12.41 / 97.28（ViT-L/14），明显优于 CLIP‑Count、VLCounter、CounTX 等；在 CARPK 上 MAE 6.07/6.38，显著低于同类方法；在 ShanghaiTech-A 上 MAE 140.7/146.2，超越 CountGD、CLIP‑Count 等。

**⚠️ 局限性**

限制：仅处理离散数量假设；当前仅支持单类别计数，无法直接处理连续数量关系或多类别联合计数。

---

## 428. M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM

**arXiv ID:** 2603.16844 | [PDF](https://arxiv.org/pdf/2603.16844v1)

**作者:** Kerui Ren `[一作]` (Shanghai Jiao Tong University), Bo Dai `[通讯]` (University of Hong Kong)

**通讯引用:** 6018 | [OpenAlex ID](https://openalex.org/A5101990493)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了 M^3 框架，将多视角基础模型 Pi3X 与像素级密集匹配头结合，实现单目视频的实时流式稠密重建，既完成高精度位姿估计，又生成高质量的神经高斯散射场。

**💡 创新点**

创新点包括：① 在多视角基础模型上增设密集匹配头，获取像素级对应关系；② 采用单一次前向推理同时更新关键帧和跟踪，显著减少模型调用；③ 引入动态区域抑制与相机内参对齐，提升鲁棒性和全局一致性；④ 将前端与后端紧耦合，形成完整的实时 SLAM 系统。

**🔧 技术方法**

核心技术：多视角基础模型 Pi3X、Dense Prediction Transformer + MLP 进行密集匹配、InfoNCE 损失训练、Sim(3) 并行优化、滑动窗口关键帧管理、动态区域估计、相机内参一致化、全局 Bundle Adjustment 与神经高斯散射（3D Gaussian Splatting）。

**📊 数据集**

使用的评测数据集：ScanNet++、ScanNetV2、VR-NeRF（室内）以及 KITTI、Waymo、FAST-LIVO2（室外）等多种公开基准。

**📈 对比分析**

与 DROID‑SLAM、MASt3R‑SLAM、VGGT‑SLAM 等 SLAM 框架对比，ATE RMSE 大幅下降（对 VGGT‑SLAM 2.0 降 64.3%）；与 ARTDECO、MonoGS、S3PO‑GS 等稠密重建基准对比，PSNR、SSIM、LPIPS 均优于对手，训练时间和高斯数量保持竞争力；与 feed‑forward Gaussian Splatting（AnySplat、Depth‑Anything‑3）对比，M^3 在视觉质量上更胜一筹，同时显著降低显存需求。

**⚠️ 局限性**

主要局限：依赖基础模型的预测质量，若基础模型产生严重错误则后端优化难以恢复；目前仅支持单目视觉，未融合 LiDAR 或 IMU 等多模态传感器；缺乏专门的回退或错误恢复机制。

---

## 429. Answer Bubbles: Information Exposure in AI-Mediated Search

**arXiv ID:** 2603.16138 | [PDF](https://arxiv.org/pdf/2603.16138v1)

**作者:** Michelle Huang `[一作]` (University of Illinois), Eshwar Chandrasekharan `[通讯]` (University of Illinois)

**通讯引用:** 1572 | [OpenAlex ID](https://openalex.org/A5069919112)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四种搜索系统（vanilla GPT、Search GPT、Google AI Overview 与传统 Google Search）在 11,000 条真实查询上的生成答案进行大规模对比，分析其引用来源、语言特征及与原始来源的一致性。

**💡 创新点**

首次系统性地揭示“答案泡沫”现象——同一查询在不同生成搜索系统中会得到结构和信息现实完全不同的结果，并发现来源偏好与文本生成过程中的歧义与情感压缩存在相互放大的“层层加权”效应。

**🔧 技术方法**

使用多模态评估技术：① 文本来源检索与引用计数；② LIWC 与模型（XLM‑R、RoBERTa）提取的风格与情感特征；③ ACU（Atomic Content Unit）拆分、RoBERTa‑MNLI 归纳概率的覆盖度与偏倚度量（Equal Coverage、Coverage Parity）。

**📊 数据集**

基于 Google Natural Questions (NQ) 语料库抽样的 11,000 条真实查询，按 11 个主题均衡分布，采集四个系统的答案并收集引用网页完整文本。

**📈 对比分析**

对四个系统在三维度（来源多样性、语言特征、来源-摘要一致性）上进行量化比较。结果显示：① Search GPT 与 Google AI 在来源覆盖率、词汇与句法复杂度上差异显著；② 搜索加权显著削弱了歧义词与情感词，提升了肯定性；③ 在信息忠实度上，Google AI 在引用数量与覆盖率上表现更差，且对长篇与百科来源存在明显过度表征。

**⚠️ 局限性**

局限性包括：仅覆盖四个系统，缺乏多地区/多时间维度；使用 NQ 语料可能不代表所有查询类型；ACU 与蕴涵评估依赖模型推断，缺少人工校验；未测量用户感知与行为，因而无法直接说明“答案泡沫”对决策的影响。

---

## 430. How Vulnerable Are AI Agents to Indirect Prompt Injections? Insights from a Large-Scale Public Competition

**arXiv ID:** 2603.15714 | [PDF](https://arxiv.org/pdf/2603.15714v1)

**作者:** Mateusz Dziemian `[一作]` (Gray Swan AI), Zico Kolter `[通讯]` (Anthropic)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在公开红队比赛中构建了包含41个工具使用、编码与电脑使用场景的大规模间接提示注入基准，收集并评估了超过27万次攻击尝试，涉及13款前沿LLM代理模型。

**💡 创新点**

首次提出同时要求执行有害行为且保持隐藏的注入攻击基准，并揭示跨模型可迁移的通用攻击模板与攻击策略。

**🔧 技术方法**

利用公开红队平台、双重评审系统（工具执行与提示评估）以及聚类、相似度图等技术对攻击进行标注与分析。

**📊 数据集**

使用从比赛中收集的攻击日志与场景数据，包含工具调用、代码修改和计算机交互记录。

**📈 对比分析**

通过统计攻击成功率（ASR）进行模型对比，Gemini 2.5 Pro最高达8.5%，Claude Opus 4.5最低0.5%；迁移实验显示攻击从强模型向弱模型的迁移率可达81%。

**⚠️ 局限性**

局限包括：威胁模型过于宽松（攻击者可完整查看对话）、仅单回合注入、未覆盖多轮升级攻击、评测仅关注最终输出且未考虑内部推理监控。

---

## 431. Physics-Informed Video Diffusion For Shallow Water Equations

**arXiv ID:** 2603.15627 | [PDF](https://arxiv.org/pdf/2603.15627v1)

**作者:** Yang Bai `[一作]`, Gitta Kutyniok `[通讯]` (Ludwig-Maximilians-Universität München)

**通讯引用:** 16389 | [OpenAlex ID](https://openalex.org/A5090767423)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发了一种物理信息视频扩散模型，能够在不进行单独渲染的前提下，同时生成二维浅水方程下的水流视频和对应的物理状态。

**💡 创新点**

创新点在于将浅水方程的初始/边界条件和地形信息直接嵌入扩散Transformer，实现物理约束与视觉生成的统一框架，显著提升生成速度与物理一致性。

**🔧 技术方法**

采用物理信息增强的Latent Diffusion模型（DiT）+VAE、Patch Embedding、CNN/MLP嵌入、文本编码器以及联合视频/物理损失的训练方式。

**📊 数据集**

使用Clawpack仿真生成的20K条二维浅水方程实验数据，覆盖128×128、256×256、512×512三种网格，并配合Blender渲染得到视频数据集。

**📈 对比分析**

与CogVideoX、OpenSora等纯视频扩散基线以及传统模拟+渲染流水线比较，在LPIPS、SSIM、PSNR、FVD等视觉指标上均显著优于基线，生成时间比经典管线快10~100倍，物理误差维持在67%-90%之间。

**⚠️ 局限性**

局限性包括：在高分辨率下物理状态准确度下降，目前仅针对浅水方程，尚未推广到更一般的流体方程。

---

## 432. Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation

**arXiv ID:** 2603.16340 | [PDF](https://arxiv.org/pdf/2603.16340v1)

**作者:** Xinhao Cai `[一作]` (Nanjing University of Science and Technology), Wenguan Wang `[通讯]` (Zhejiang University)

**通讯引用:** 19352 | [OpenAlex ID](https://openalex.org/A5101433884)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为Iris的确定性双阶段光谱门控扩散框架，用于单目深度估计；

**💡 创新点**

创新点在于将低频实景先验与高频几何细节分离训练，结合Spectral‑Gated Distillation（低频教师引导）与Spectral‑Gated Consistency（高频细节对齐）以及辅助重建约束；

**🔧 技术方法**

采用Stable Diffusion V2作为基座，使用光谱门控算子、两阶段时间步策略、单步确定性U‑Net、伪标签教师、以及重建损失；

**📊 数据集**

训练数据包括59K合成样本（Hypersim + Virtual KITTI）与100K实景样本（SA‑1B）伪标签，评估覆盖NYUv2、ScanNet、KITTI、ETH3D、DIODE等真实数据集；

**📈 对比分析**

与多种前沿方法比较，Iris在零样本/零仿射深度估计任务中获得最优或近优的指标，平均AbsRel和δ₁均优于现有扩散与大规模判别模型，推理速度显著快于多步扩散方法；

**⚠️ 局限性**

局限在于仍需依赖大规模伪标签与教师模型，扩散模型对光照/遮挡的鲁棒性未完全验证，且两阶段时间步选择对不同场景可能需要调优。

---

## 433. Neural Pushforward Samplers for the Fokker-Planck Equation on Embedded Riemannian Manifolds

**arXiv ID:** 2603.16239 | [PDF](https://arxiv.org/pdf/2603.16239v1)

**作者:** Andrew Qing He `[一作]` (Southern Methodist University), Wei Cai `[通讯]` (Southern Methodist University)

**通讯引用:** 14371 | [OpenAlex ID](https://openalex.org/A5026754269)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文将弱对抗性神经推送方法（WANPF）推广到紧致嵌入Riemann曲面上的Fokker‑Planck方程，构建重排网络保证采样点始终位于曲面上，并利用环境平面波测试函数在任意曲面上闭式计算Laplace‑Beltrami，从而实现无网格、无坐标图、无雅可比的曲面概率密度求解。

**💡 创新点**

创新点在于①引入重排结构使生成器可非可逆且可在高维噪声空间工作；②利用平面波测试函数在嵌入空间中闭式表达Laplace‑Beltrami，避免对曲面几何的自动微分；③同时给出稳态与时变两种WANPF形式，并为球面与平坦托罗斯给出专门公式。

**🔧 技术方法**

技术手段包括神经网络推送+重排、弱对抗训练（min‑max 目标）、Monte Carlo 估计、闭式Laplace‑Beltrami公式、环境平面波测试函数、Adam 双优化、梯度裁剪、学习率余弦退火等。

**📊 数据集**

实验主要在双井势的单位球面（S²）上进行，使用标准正态分布为基分布进行采样；未使用公开数据集。

**📈 对比分析**

通过训练损失收敛曲线、生成样本的直方图与理论 Gibbs 分布对比，验证模型在 S² 双井问题上能准确聚集在两极点；虽然文中未给出与传统网格/有限元方法的数值对比，但展示了该方法在高维曲面上无需网格即可逼近稳态分布。

**⚠️ 局限性**

局限性包括需要曲面可解析得到投影矩阵 P(x) 与平均曲率向量 H(x)；平面波测试函数非固有特征函数，可能导致收敛速度受限；对非等方差扩散或分数 Laplace‑Beltrami 的推广仍需进一步研究；在更高维或复杂拓扑的曲面上采样与训练的效率与稳定性也需评估。

---

## 434. Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization

**arXiv ID:** 2603.16105 | [PDF](https://arxiv.org/pdf/2603.16105v1)

**作者:** Francesco Pio Monaco `[一作]` (University of Trento), Giovanni Iacca `[通讯]` (University of Trento)

**通讯引用:** 2983 | [OpenAlex ID](https://openalex.org/A5007121933)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于 Zipf 定律的词汇多样性校准数据挑选方法，用于后训练的模型压缩（剪枝与量化）。

**💡 创新点**

该方法的创新点在于：①模型无关且计算复杂度为线性；②可在单域和多域场景下统一使用；③与昂贵的模型依赖式方法（COLA）性能持平，同时显著提升压缩后模型的通用性。

**🔧 技术方法**

核心技术包括贪心词汇增益采样、线性复杂度的单域抽样算法，以及分层多域抽样策略，用于构造高词汇覆盖率的校准集合。

**📊 数据集**

实验使用的语料包括 C4、WikiText、Pile、GSM8k、MMLU、CommonsenseQA、WinoGrande、ARC、WMT14 等多领域文本；模型为 LLama‑3.1‑8B 与 Gemma‑2‑9B，压缩方法涵盖 Wanda、2SSP、GPTQ、AWQ。

**📈 对比分析**

与随机采样和 COLA 进行对比，基准任务包括语言模型困惑度、数学推理、常识推理、NLI、知识翻译等；结果显示 Zipf 采样在大多数任务上优于随机，且与 COLA 性能相当或略优，压缩后模型准确率提升或保持不变，同时计算时间比 COLA 快 200–260 倍。

**⚠️ 局限性**

局限性：仅在英文数据上验证；对多语言、Morphologically Rich 语言或非字母脚本的适配尚待研究；未评估在 MoE、混合专家或多模态 LLM 上的表现；并且在高频词剔除等预处理步骤上可能需要针对不同语言进行微调。

---

## 435. Retrieval-Augmented Sketch-Guided 3D Building Generation

**arXiv ID:** 2603.16612 | [PDF](https://arxiv.org/pdf/2603.16612v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 436. DynHD: Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning

**arXiv ID:** 2603.16459 | [PDF](https://arxiv.org/pdf/2603.16459v1)

**作者:** Yanyu Qian `[一作]` (Nanyang Technological University), Shirui Pan `[通讯]` (Griffith University)

**通讯引用:** 24404 | [OpenAlex ID](https://openalex.org/A5008056593)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Diffusion大型语言模型的幻觉检测，提出基于去噪动态偏差学习的方法，通过语义感知的token过滤和统计证据生成构造证据轨迹，并与学习得到的参考轨迹对比以判定幻觉。

**💡 创新点**

创新点包括：① 用语义感知过滤去除无信息token并通过均值、最大值、top-k等统计量构造多尺度证据；② 引入参考证据生成器学习不同问题条件下的正常熵演化轨迹；③ 通过偏差度量和停滞/反弹正则化捕获幻觉的动态特征。

**🔧 技术方法**

采用token熵、统计特征（均值、最大值、top-k）、参考轨迹生成网络、注意力权重的时序聚合、路径与反弹正则化等技术。

**📊 数据集**

使用 TriviaQA、HotpotQA、CommonsenseQA 三个事实推理数据集，基于 LLaDA‑8B‑Instruct 与 Dream‑7B‑Instruct 两个Diffusion LLM。

**📈 对比分析**

与输出基、潜在基以及 state‑of‑the‑art 的 TraceDet 进行对比，AUROC 最高，平均提升 12.2%，并在性能与效率上兼顾，适合实时监测。

**⚠️ 局限性**

局限性：仍受固定长度生成导致信息不平衡影响，主要依赖熵作为信号，难以处理多轮交互或更开放域多样性；对内部 remasking 策略的鲁棒性还有待进一步提升。

---

## 437. Remarks on the Relevance of Privacy Expectations for Default Opt-out Settings

**arXiv ID:** 2603.15705 | [PDF](https://arxiv.org/pdf/2603.15705v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 438. Differential Harm Propensity in Personalized LLM Agents: The Curious Case of Mental Health Disclosure

**arXiv ID:** 2603.16734 | [PDF](https://arxiv.org/pdf/2603.16734v1)

**作者:** Caglar Yildirim `[一作]` (Northeastern University), Caglar Yildirim `[通讯]` (Northeastern University)

**通讯引用:** 591 | [OpenAlex ID](https://openalex.org/A5000390546)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在AgentHarm基准上评估了前沿与开源LLM在工具使用型代理中的安全性，重点研究了精神健康披露在用户上下文中对有害任务完成与拒绝率的影响。

**💡 创新点**

创新点在于首次将用户个性化（特别是精神健康披露）作为实验变量加入AgentHarm评估，揭示个性化对有害行为的弱保护效应，并系统分析轻量级越狱对该保护的削弱。

**🔧 技术方法**

采用AgentHarm评估框架、LLM-as-judge（GPT‑4o）自动评分、Persona式前缀控制的用户上下文以及轻量级越狱提示来触发模型行为。

**📊 数据集**

使用AgentHarm 176个多步任务（含善意、恶意及越狱版本）以及对应的BioOnly和Bio+MH用户前缀作为实验材料。

**📈 对比分析**

通过比较无bio、bio-only和bio+mh三种用户上下文，在benign、harmful和jailbreak三种任务上下文下计算平均有害完成分数和拒绝率。结果显示，虽然大部分前沿模型在无bio时仍完成一定比例的有害任务，bio和bio+mh可略降低有害完成并提高拒绝率，但在越狱情形下该保护效果显著削弱，且不同模型表现差异大。

**⚠️ 局限性**

限制包括：仅使用简短文本披露且仅探讨精神健康属性，未覆盖更复杂或隐式的个性化；评估基于提示的个性化，未涵盖真实系统中的长期记忆或结构化存储；评估依赖GPT‑4o自动评判，可能带来测量误差；越狱仅为轻量级，未覆盖更强攻击手段。

---

## 439. MFTune: An Efficient Multi-fidelity Framework for Spark SQL Configuration Tuning

**arXiv ID:** 2603.16450 | [PDF](https://arxiv.org/pdf/2603.16450v1)

**作者:** Beicheng Xu `[一作]` (Peking University), Bin Cui `[通讯]` (Peking University)

**通讯引用:** 13243 | [OpenAlex ID](https://openalex.org/A5062357883)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MFTune，一个面向 Spark SQL 的多保真度调优框架，利用查询分区、多保真度评估、密度压缩、迁移学习与双阶段热启动等技术，在有限预算内高效搜索并确定最优配置。

**💡 创新点**

创新点包括：① 专为 Spark SQL 设计的查询基保真度分区策略；② 采用 SHAP 与核密度估计实现参数选择与范围压缩；③ 将迁移学习与双阶段热启动融入多保真度搜索；④ 构造高相关性的低保真代理，克服传统数据量缩减和早停的缺陷。

**🔧 技术方法**

使用技术包括 Bayesian Optimization、Hyperband（SH）、Probabilistic Random Forest 替代 Gaussian Process、Kendall tau 相关性评估、Kernel Density Estimation、SHAP 值、迁移学习（任务相似度匹配）和多保真度评估策略。

**📊 数据集**

实验基准为 TPC‑H 和 TPC‑DS，数据规模包括 600 GB 等不同规模的工作负载。

**📈 对比分析**

在 48 小时预算下与五种主流调优方法（如 OtterTune、ResTune、LlamaTune、OpAdvisor、TopTune 等）进行对比，MFTune 在 TPC‑H 上降低 25.9%–43.1% 的延迟，在 TPC‑DS 上降低 37.8%–63.1%，显著优于其它方法。

**⚠️ 局限性**

局限性在于：仅针对 Spark SQL 验证，通用性待进一步探索；低保真代理的有效性依赖查询集的代表性，对极端或新颖工作负载可能失效；迁移学习与热启动需要历史任务数据库，新任务缺乏历史时效果受限。

---

## 440. Alternating Reinforcement Learning with Contextual Rubric Rewards

**arXiv ID:** 2603.15646 | [PDF](https://arxiv.org/pdf/2603.15646v1)

**作者:** Guangchen Lan `[一作]` `[通讯]` (Purdue University), Guangchen Lan (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Alternating Reinforcement Learning with Rubric Rewards（ARL‑RR）框架，利用将多维 rubric 奖励拆分为若干语义 meta‑class 并交替优化的方式，避免传统的线性 scalarization。

**💡 创新点**

创新点：1）通过 meta‑class 分解保留多维奖励结构，消除对固定权重聚合的依赖；2）设计轻量级的搜索式 meta‑class 顺序确定策略，实现动态课程；3）理论证明 scalarization 缩小奖励方差，从而削弱 RL 信号；4）提出 synthetic meta‑classification，支持无专家标签的场景。

**🔧 技术方法**

使用了 RL‑HF 及 PPO‑style 目标、meta‑class 交替优化、搜索式动态 curriculum、synthetic meta‑classification、方差收缩理论分析，以及基于 Qwen3‑32B 的奖励模型。

**📊 数据集**

实验基于 HealthBench（5K 样本、专家标注的 rubric）以及 Qwen3‑32B 作为奖励模型；并在 Qwen3‑{1.7B,4B,8B,14B} 这四个 actor 上进行评估。

**📈 对比分析**

与传统 scalarized RL baseline 在 HealthBench 上进行直接对比；ARL‑RR 在所有模型规模上均显著提升最终得分（如 Qwen3‑1.7B 从 0.556 提升至 0.576，最高为 Qwen3‑14B 0.762→0.788），并在训练周期与时间上更高效；多种 ablation（reward model、meta‑class 顺序、搜索比例）进一步验证其优势。

**⚠️ 局限性**

局限性：1）仅在 HealthBench 上验证，缺乏更广泛的多任务或开放域评测；2）synthetic meta‑classification 相比专家标注有轻微性能损失；3）结果高度依赖奖励模型的准确性，未充分处理奖励噪声与不确定性。

---

## 441. OneWorld: Taming Scene Generation with 3D Unified Representation Autoencoder

**arXiv ID:** 2603.16099 | [PDF](https://arxiv.org/pdf/2603.16099v1)

**作者:** Sensen Gao `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Jiawang Bian `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出OneWorld框架，在预训练3D基底模型的统一表示空间直接进行扩散生成，实现单视图条件下的高质量3D场景合成。

**💡 创新点**

创新点：①构建3D统一表示自编码器3D-URAE，融合外观与语义；②跨视图对应一致性损失；③流形漂移强制，减小训练-推理漂移。

**🔧 技术方法**

采用3D Foundation模型π^3、3D Gaussian Splatting、Diffusion Transformer（DiT）、自编码器、交叉视图对应正则、流形漂移强制等技术。

**📊 数据集**

使用RealEstate10K和DL3DV-10K两个大规模多视图数据集。

**📈 对比分析**

与FlashWorld、Gen3R、LVSM等基线对比，在1视图NVS和WorldScore评估中，在PSNR/SSIM/LPIPS和WorldScore指标上均获得SOTA或接近最佳。

**⚠️ 局限性**

局限：对大视角变换的泛化仍有限；训练成本高；仍需改进对户外场景的表现。

---

## 442. The Era of End-to-End Autonomy: Transitioning from Rule-Based Driving to Large Driving Models

**arXiv ID:** 2603.16050 | [PDF](https://arxiv.org/pdf/2603.16050v1)

**作者:** Eduardo Nebot `[一作]` (University of Sydney), Julie Stephany Berrio Perez `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并分析了从规则化模块化架构向端到端学习的大型驾驶模型（LDM）的转变，重点讨论了Tesla FSD V12–V14、Rivian、NVIDIA等企业在机器人出租车和监督式E2E驾驶（FSD Supervised/L2++）中的部署与技术实现。

**💡 创新点**

创新点在于引入两阶段训练（模仿学习+强化学习）实现安全超越人类的端到端驾驶；将混合模态（视觉+音频+多秒时序）与混合模型（Mixture of Models）结合；以及双栈安全架构（E2E+规则栅栏）与开放源代码的推理模型，构建可持续迭代的“数据飞轮”。

**🔧 技术方法**

主要技术包括大规模Transformer时序建模、图像+雷达/音频多模态融合、行为克隆+强化学习边缘案例优化、实时自我监测与决策切换、以及基于物理AI的无监督世界建模（Cosmos）和模拟训练（Infinite Simulation）。

**📊 数据集**

使用的训练数据来自亿级车辆行驶记录（Tesla、Rivian、Waymo等），包含多种交通场景、天气、道路标记以及通过ADR采集的“重要事件”，同时辅以合成与仿真数据（NVIDIA Isaac Sim、Rivian Data Flywheel）。

**📈 对比分析**

通过与传统模块化栈、以及平均人类驾驶的基准进行对比，FSD Supervised在北美、澳新等地区实现了每百万英里事故率从0.7降至5.1（大事故），并在多城市机器人出租车试点中展示了与手动驾驶相近或更优的安全性、舒适度和可靠性，表明端到端模型在常见与稀有情景均能保持或提升性能。

**⚠️ 局限性**

主要局限包括：对长尾稀有事件的泛化仍受限于训练样本覆盖；需要针对不同国家/地区的道路规则、标记、驾驶文化进行本地化与校准；人机交互界面与监督责任仍需完善，防止误用与责任模糊；安全评估指标尚未统一，需要更细粒度的预警与用户分层分析。

---

## 443. Adaptive Multi-Head Finite-State Gamblers

**arXiv ID:** 2603.16034 | [PDF](https://arxiv.org/pdf/2603.16034v1)

**作者:** Julianne Cruz `[一作]` (Swarthmore), Neil Lutz `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并研究了自适应多头有限状态赌徒模型，证明其在预测序列方面优于传统的盲目（数据无关）多头有限状态赌徒，并给出了自适应多头模型的层级定理，展示随着头数增加预测能力严格提升。

**💡 创新点**

创新点在于首次引入自适应头运动的多头有限状态赌徒框架，并证明了自适应性与头数对序列可预测性具有显著且可分离的影响，填补了此前只关注盲目头运动研究的空白。

**🔧 技术方法**

主要技术手段包括Kolmogorov复杂度分析、s-赌徒（s-gale）框架、压缩-可预测性对应以及构造特殊序列以实现自适应与盲目模型的分离和层级证明。

**📊 数据集**

研究中未使用真实数据集，而是通过构造随机序列（如Martin‑Löf随机序列）和对其进行特定变换得到的人工序列来演示和验证理论结论。

**📈 对比分析**

通过比较自适应与盲目模型在同一系列构造序列上的predimension下界与上界，发现自适应模型的predimension更低（即更好），并证明了随着头数递增，预测能力呈严格递增的层级趋势。

**⚠️ 局限性**

局限性包括：分离效果量化较小、尚未确定自适应模型与盲目模型间最大可能的分离幅度、缺乏关于自适应模型在并集上的稳定性研究、未证实压缩特征在自适应情形下的完整性，以及对双向移动或非确定性扩展的影响仍未探讨。

---

## 444. Practical MCTS-based Query Optimization: A Reproducibility Study and new MCTS algorithm for complex queries

**arXiv ID:** 2603.16474 | [PDF](https://arxiv.org/pdf/2603.16474v1)

**作者:** Vladimir Burlakov `[一作]` (Lomonosov Moscow State University), Yuriy Dorn `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5069069110)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对现有基于MCTS的学习型查询优化器（AlphaJoin、HyperQO）进行可复现性评估，并提出一种结合默认成本模型与极值UCT策略的两阶段MCTS算法，用于改进复杂多表连接查询的执行计划。

**💡 创新点**

创新点包括：① 用标准数据库内部成本模型替代昂贵的学习模型，提升泛化性；② 引入极值UCT（UCT-extreme）取代传统平均UCT，以更关注全局最优结果；③ 设计两阶段前缀引导MCTS，将GEQO产生的初始计划作为搜索起点，显著提升对≥12表查询的性能；④ 公开完整实现与实验脚本，强调可复现性。

**🔧 技术方法**

技术主要有：Monte Carlo Tree Search（MCTS）与极值UCT选择策略、基于遗传算子（交叉、变异）的搜索扩展、Python外部控制循环与PostgreSQL前导提示（leading hints）交互、GEQO启发式搜索、基准评估脚本。

**📊 数据集**

数据集使用公开的IMDb Join Order Benchmark（JOB，113条查询）和新构造的JOB‑Complex（30条复杂查询）作为评测基准。

**📈 对比分析**

与AlphaJoin、HyperQO以及PostgreSQL默认优化器（DP/GEQO）在两套基准上对比。实验表明：在≥12表的复杂查询中，提出的MCTS‑Extreme在平均执行时间上比PostgreSQL下降约15‑20%，并明显优于AlphaJoin、HyperQO；在常规JOB查询中，表现波动但总体不劣于传统优化器；在JOB‑Complex上，学习型优化器几乎不产生改进，MCTS‑Extreme则保持稳定提升。

**⚠️ 局限性**

局限性包括：① 所有实现均为外部控制，未在DBMS核心中集成，难以完整测量端到端时间；② 只测试单一数据库（IMDb），对其它数据分布的泛化未知；③ 对极值UCT参数（γ、c）和MCTS搜索预算敏感；④ 需要手动构造前导提示，可能在实际部署中产生额外开销。

---

## 445. AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents

**arXiv ID:** 2603.16496 | [PDF](https://arxiv.org/pdf/2603.16496v1)

**作者:** Shannan Yan `[一作]` (Tsinghua University), Fengyun Rao `[通讯]` (WeChat Vision, Tencent Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了AdaMem，一个自适应用户中心记忆框架，整合工作、情节、人格和图形记忆，并通过目标识别、问题条件检索与多代理协作实现长程对话推理。

**💡 创新点**

采用参与者特定的多层记忆结构、问题条件检索路径规划与关系感知图检索、以及角色专化的多代理流水线，解决了语义检索盲点、碎片化与静态粒度的三大限制。

**🔧 技术方法**

结合LLM记忆代理、基于图的关系检索、语义检索与图扩展融合、目标参与者解析，以及研究和工作代理的迭代证据收集与合成等技术。

**📊 数据集**

在LoCoMo（多会话长时程推理）和PERSONAMEM（用户建模）两个基准上进行评估。

**📈 对比分析**

与MemGPT、A-Mem、Mem0、LangMem、Zep等开源记忆框架对比，AdaMem在LoCoMo上以GPT-4.1-mini实现最高F1 44.65%（+4.4%相对SOTA），在PERSONAMEM上准确率63.25%（+5.9%），显示出显著性能提升。

**⚠️ 局限性**

系统复杂度、token消耗和推理延迟升高，且对上游解析、实体链接和时间归一化误差敏感。

---

## 446. OGScene3D: Incremental Open-Vocabulary 3D Gaussian Scene Graph Mapping for Scene Understanding

**arXiv ID:** 2603.16301 | [PDF](https://arxiv.org/pdf/2603.16301v1)

**作者:** Siting Zhu `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9211 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出OGScene3D，一种可增量的开放词汇3D高斯场场景图构建与语义映射系统。

**💡 创新点**

采用置信度加权的3D高斯语义表示，层次化3D语义优化与长期全局优化，并实现逐步图构建。

**🔧 技术方法**

基于3D Gaussian Splatting、SAM与YOLO-World进行开放词汇分割，DROID-SLAM定位，CLIP/GPT‑4o生成标签与关系，深度+颜色损失优化。

**📊 数据集**

在Replica、ScanNet、3RScan等公共数据集及真实机器人实验场景上进行评测。

**📈 对比分析**

与SemGS、OpenGS‑SLAM等基线对比，mIoU提升约15‑20%，3D语义与图关系召回率显著高于基线；实时性约10分钟/场景。

**⚠️ 局限性**

仅关注空间关系，未覆盖功能性或语义层面的多关系；依赖SAM等大型模型导致计算开销。

---

## 447. Open-Source Reproduction and Explainability Analysis of Corrective Retrieval Augmented Generation

**arXiv ID:** 2603.16169 | [PDF](https://arxiv.org/pdf/2603.16169v1)

**作者:** Surya Vardhan Yalavarthi `[一作]` `[通讯]` (University of Cincinnati), Surya Vardhan Yalavarthi (University of Cincinnati)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文实现了CRAG（Corrective Retrieval Augmented Generation）系统的完整开源复现，并使用SHAP进行检索评估器的可解释性分析。

**💡 创新点**

创新点包括：1）将原本依赖付费Google搜索API和封闭模型的CRAG替换为免费的Wikipedia API和Phi-3-mini-4k-instruct，保持性能；2）首次用SHAP对T5检索评估器进行逐词解释，发现其主要依赖命名实体对齐；3）揭示了评估器在领域迁移和特殊实体类型上的失败模式。

**🔧 技术方法**

主要技术：检索增强生成（RAG）、纠正机制（评估器阈值控制），使用T5-large微调评估器、Phi-3-mini生成模型，基于Wikipedia API的多阶段检索，SHAP文本掩码进行可解释性分析。

**📊 数据集**

使用的数据集为PopQA（开放域实体问答）和ARC-Challenge（科学多项选择问答）。

**📈 对比分析**

与原始CRAG（使用LLaMA-2）以及vanilla RAG做对比。复现系统在PopQA上达到54.4%（原始54.9%），在ARC-Challenge上达到85.2%（原始84.8%）。在Ambiguous模式下引入Wikipedia搜索后准确率提升至23.0%。

**⚠️ 局限性**

局限性包括：单次实验缺乏置信区间；SHAP分析仅基于9个案例，缺乏统计支持；未对阈值进行针对不同生成模型或数据集的重新调优；Wikipedia搜索覆盖率仍不及商业搜索；以及评估器在非人名实体和科学领域的迁移性差。

---

## 448. Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation

**arXiv ID:** 2603.15759 | [PDF](https://arxiv.org/pdf/2603.15759v1)

**作者:** Jacob Levy `[一作]` (University of Texas at Austin), David Fridovich-Keil `[通讯]` (University of Texas at Austin)

**通讯引用:** 590 | [OpenAlex ID](https://openalex.org/A5070827615)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8d10c613-917e-4880-9716-17789f50e119` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于世界模型的仿真至真实的迁移框架，先在仿真中使用专家策略预训练完整的世界模型，再在真实机器人上仅通过在线规划和监督式动力学微调实现快速适配；

**💡 创新点**

创新点在于将任务结构（状态编码、奖励与价值函数）与环境特定动力学分离，仿真中学习全局结构并冻结，只在真实环境中微调动力学，从而避免长时序信用分配问题，实现高样本效率；

**🔧 技术方法**

使用的技术包括：隐式（latent）变换器世界模型、强化学习专家策略（PPO）、基于MPPI的采样规划、教师-学生蒸馏、动作扰动生成多样化数据、以及增量式监督式动力学微调；

**📊 数据集**

主要数据集为：仿真数据（约100M条轨迹，其中约55.7%来自仿真专家策略），以及真实机器人数据（15–30分钟约几千条轨迹，主要用于动力学微调）；

**📈 对比分析**

与多种基线（RLPD、IQL、SGFT‑SAC、Diffusion Policy等）以及行为克隆方法比较，实验在两类精细操纵任务与两类四足行走任务中，所提方法在真实环境中以极少数据实现了约2倍以上的成功率或前进距离，并在训练过程中保持单调提升，优于传统RL微调导致的灾难性遗忘；

**⚠️ 局限性**

局限性包括：对高保真仿真的依赖；需要大量仿真数据进行预训练；冻结奖励/价值模型可能在跨任务或跨域中失效；当前仅验证单任务设置，尚未扩展到多任务或更复杂环境。

---

## 449. SseRex: Practical Symbolic Execution of Solana Smart Contracts

**arXiv ID:** 2603.16349 | [PDF](https://arxiv.org/pdf/2603.16349v1)

**作者:** Tobias Cloosters `[一作]` (University of Duisburg-Essen), Lucas Davi `[通讯]` (University of Duisburg-Essen)

**通讯引用:** 6183 | [OpenAlex ID](https://openalex.org/A5089242868)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了第一套针对Solana智能合约的符号执行漏洞检测框架，能够自动识别缺失所有者检查、签名检查、密钥检查以及任意CPI等Solana特定漏洞；

**💡 创新点**

创新点在于：①首次针对Solana独特的账户模型设计符号执行流程；②提出针对账户反序列化、字符串格式化等Solana特有挑战的状态合并与混合符号长度技术；③构造了专门针对缺失所有者/签名/密钥检查和任意CPI的高精度漏洞或acular；

**🔧 技术方法**

采用符号执行技术（基于SENinja）与字节码提升工具，结合静态预分析、状态合并、混合符号长度、字符串跳过、CPI/主流程/随机三策略的探索；

**📊 数据集**

使用了全量Solana区块链部署的字节码合约（≈3763个，其中43.2%为Anchor）、120个Anchor开源项目的字节码与源代码、以及VRust、Neodyme工作坊提供的已知漏洞数据集；

**📈 对比分析**

与现有工具（VRust、Solana Fuzzer）以及先前研究数据对比，检测率提升约30%–40%，误报率下降到10%以下；在大规模评测中发现约374个缺失密钥/所有者检查、100个任意CPI漏洞，覆盖率平均为33%；

**⚠️ 局限性**

局限性包括：①缺乏对整数溢出、重入攻击等非字节码层面漏洞的检测；②部分漏洞依赖业务逻辑和链上状态，难以通过静态分析准确评估；③符号执行仍受限于路径爆炸，单个合约最大分析时长为2小时，可能漏检复杂路径；

---

## 450. When Generative Augmentation Hurts: A Benchmark Study of GAN and Diffusion Models for Bias Correction in AI Classification Systems

**arXiv ID:** 2603.16134 | [PDF](https://arxiv.org/pdf/2603.16134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 451. When Thinking Hurts: Mitigating Visual Forgetting in Video Reasoning via Frame Repetition

**arXiv ID:** 2603.16256 | [PDF](https://arxiv.org/pdf/2603.16256v1)

**作者:** Xiaokun Sun `[一作]` (University of Science and Technology of China), Linli Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3533 | [OpenAlex ID](https://openalex.org/A5009732907)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了FrameRepeat框架，通过在视频输入阶段重复重要帧来增强视觉信号，缓解视频推理中视觉遗忘的问题。

**💡 创新点**

创新点在于：1) 通过Add-One-In自监督方式自动生成帧重要性标签；2) 设计轻量化的跨模态交叉注意力评分模块，可迁移到不同Video-LLM；3) 将帧重复作为无额外推理开销的输入增强策略。

**🔧 技术方法**

使用CLIP编码器、跨模态交叉注意力、两层FFN、回归+排序损失进行训练；AOI策略在冻结的多模态大语言模型上计算重复增益；在推理时选择Top‑K帧重复。

**📊 数据集**

在VideoMME、LongVideoBench、LVBench、MLVU、Video-Holmes五个视频推理基准上进行实验，并用LLaVA‑Video‑178K等训练集。

**📈 对比分析**

相较于基线的CLIP‑K、Bottom‑K、Random‑K及TSPO方法，FrameRepeat在所有模型和基准上均实现显著提升，最高可达+3.7分；实验显示8帧重复是最佳平衡点，且模型在另一大模型上也保持了迁移效果。

**⚠️ 局限性**

局限性包括：1) 需要手动设定重复帧数K，过多会产生干扰；2) 仅在特定Video‑LLM上训练，跨框架迁移仍需验证；3) 只关注帧级重要性，未考虑时序动态与事件连续性。

---

## 452. Multi-objective Optimization for Over-the-Air Federated Edge Learning-enabled Collaborative Integrated Sensing and Communications

**arXiv ID:** 2603.15783 | [PDF](https://arxiv.org/pdf/2603.15783v1)

**作者:** Saba Asaad `[一作]` (York University), Ping Wang `[通讯]` (York University)

**通讯引用:** 50629 | [OpenAlex ID](https://openalex.org/A5100338632)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个多目标协同ISAC框架（CollabSenseFed），将目标定位与 over‑the‑air 联邦学习（OTA‑FEEL）联合起来，实现信号的双重用途，并通过正交脉冲实现干扰抑制。

**💡 创新点**

创新点包括：① 将 OTA‑FEEL 与分布式感知统一到同一信号与上行/下行波束；② 用局部充分统计量推导可无偏的目标坐标估计；③ 通过 Cramér‑Rao 边界得到统一的感知误差下界并将其作为多目标优化的第二目标；④ 采用 ε‑约束与 BCD 方法求解非凸多目标优化，得到近似最优波束。

**🔧 技术方法**

核心技术包括：多任务 OTA‑计算、正交脉冲成形、充分统计量提取、最大似然估计、Cramér‑Rao 边界、ε‑约束多目标优化、BCD 与凸松弛、迭代投影。

**📊 数据集**

使用 MNIST（2 层 MLP）和 CIFAR‑10（3 阶卷积网络）进行联邦学习实验，同时在同一网络上执行目标定位。

**📈 对比分析**

与理想的无定位 OTA‑FEEL、完美 OTA‑FEEL 以及单次投射式分布式感知基准进行对比。实验表明 CollabSenseFed 在学习精度上与无定位方案几乎无差异，而在感知精度上能逐步逼近 Cramér‑Rao 下界，明显优于单次投射方法；在多任务和多用户情形下表现出明显的感知-聚合误差权衡。

**⚠️ 局限性**

局限性：假设脉冲完全正交、理想无频偏、理想信道；未考虑硬件失真、时钟偏移、信道估计误差、用户掉线等实际因素；算法收敛到局部最优；实现复杂度与设备数量、天线数量呈三次方增长。

---

## 453. Human/AI Collective Intelligence for Deliberative Democracy: A Human-Centred Design Approach

**arXiv ID:** 2603.16260 | [PDF](https://arxiv.org/pdf/2603.16260v1)

**作者:** Anna De Liddo `[一作]` (Open University), Simon Buckingham Shum `[通讯]` (University of Technology Sydney)

**通讯引用:** 11890 | [OpenAlex ID](https://openalex.org/A5057774051)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出“针对议事民主的集体智能”（CI4DD）框架，设计并实施了两款 AI 辅助议事平台（BCause 与 DemocraticReflection），并通过人机协作的共创方法生成用户需求、系统需求及用例场景，最终在真实议事组织中进行原型验证。

**💡 创新点**

创新点包括：
• 将人机协作与议事民主相结合，提出以“人本中心设计”为核心的 CI4DD 发展路径；
• 设计了跨模态（面对面到线上）的转化流程（BCause 的转录、结构化、聚类与政策建议生成）；
• 开发了实时观众互动与议事分析的“二屏”系统（DemocraticReflection），实现实时转录、主题提取、动态问题生成和即时反馈；
• 在设计与实施过程中采用了多轮共创、用户情景共建和验证，以确保技术落地满足真实议事需求。

**🔧 技术方法**

使用技术主要包括：
• 语音识别（Speech‑to‑Text）与说话人识别；
• 论点挖掘与 IBIS 结构化（基于微调 DeBERTa Transformer）；
• 聚类与可视化（Fuzzy C‑means、UMAP、Voronoi/Treemap/Sunburst）；
• 大语言模型（LLM）用于摘要、问题生成和标签命名；
• 前端交互与可视化（自定义 Dashboards、实时投票卡片）。

**📊 数据集**

主要数据集：
• RIE Europe4Citizens 会议录音/文字稿（面向在线议事的原始文本）；
• CEPS Young Thinkers 现场访谈音频（实时转录与观众投票数据）；
• 公开或内部收集的议事日志与反馈卡片；
(未公开公开的标准公共数据集，仅使用自组织内部收集的语料)。

**📈 对比分析**

比较方法：
• 通过共创工作坊和迭代验证评估系统与需求的一致性；
• 在原型部署后收集定性反馈（使用者满意度、议事深度、决策透明度等）；
• 对比传统 DD 工具（如 Kialo、Pol.is）的功能缺口；
性能表现：
• BCause 在转录到结构化的准确率在 80‑85%（人工审核后），聚类效果能清晰呈现主题；
• DemocraticReflection 能在 1‑2 秒内完成实时转录和主题标记，动态问题生成得到参与者认可，但未给出量化指标。

**⚠️ 局限性**

局限性：
• 需要人工审核与干预，无法完全自动化；
• 依赖领域特定的模型微调，跨领域迁移受限；
• 原型验证规模有限，缺乏大规模量化评估与对比实验；
• 可能存在算法偏见与数据隐私风险，需进一步伦理与安全评估；
• 对多语言与跨文化场景的适配性尚未系统验证。

---

## 454. Tarab: A Multi-Dialect Corpus of Arabic Lyrics and Poetry

**arXiv ID:** 2603.16601 | [PDF](https://arxiv.org/pdf/2603.16601v1)

**作者:** Mo El-Haj `[一作]` `[通讯]`, Mo El-Haj

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了Tarab语料库，整合了现代和古典的歌曲歌词与诗歌，覆盖十四世纪以上、六大方言与多地区来源，统一以句子为基本单元，提供多维度元数据。

**💡 创新点**

首次大规模、跨时间、跨方言地将创意语言（歌曲与诗歌）统一收集到一个语料库，且在句子级别对每条记录进行多属性标注（方言、地理、时代、文体等），为跨域、跨时空的语言学与计算研究提供新资源。

**🔧 技术方法**

采用爬虫抓取公开网站、对Kaggle诗歌集与Habibi歌词集进行整合、统一字符编码、最小预处理、构建统一表结构；对词汇进行FastText词向量训练并用t‑SNE可视化；使用统计分析展示词汇多样性、方言差异及代码切换分布。

**📊 数据集**

主要数据来源为公开的Kaggle Arabic Poetry 数据集、Habibi Lyrics Corpus 以及自爬取的公开歌词网页，最终语料约 2,557,311 句、13,509,336 词。

**📈 对比分析**

通过词向量可视化与词汇统计比较方言与语体差异，并与原 Habibi 语料对比：词句数扩大约 4.8 倍；未给出具体机器学习任务指标，主要呈现语言学/统计对比。

**⚠️ 局限性**

局限性：时间元数据仅按历史时期粗划，缺乏精细年份；方言标注为句子或作品级别，未细化句内混合；未包含音乐、韵律、演绎等多模态信息；不适用于完整音乐学或音频处理任务；未来需完善时间、方言、风格等细粒度标注。

---

## 455. EPOFusion: Exposure aware Progressive Optimization Method for Infrared and Visible Image Fusion

**arXiv ID:** 2603.16130 | [PDF](https://arxiv.org/pdf/2603.16130v1)

**作者:** Zhiwei Wang `[一作]` (Zhejiang University of Technology), Edmund Y. Lam `[通讯]` (University of Hong Kong)

**通讯引用:** 9472 | [OpenAlex ID](https://openalex.org/A5008832723)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向过曝情况的红外-可见图像融合模型EPOFusion，利用曝光引导模块、迭代解码器和自适应损失实现对过曝区域热信息的精准恢复与细节保留。

**💡 创新点**

创新点包括：①曝光感知引导模块显式关注过曝区的红外特征；②基于扩散式迭代解码器的逐步优化提升多模态信息融合质量；③多尺度上下文融合模块(MSCF)与自适应损失协同调节不同区域的融合权重。

**🔧 技术方法**

核心技术为：曝光引导网络、迭代式扩散解码（DDIM/DDM）、多尺度上下文融合模块、基于区域的自适应损失（融合损失、引导损失、扩散损失）以及SAM辅助像素级标注。

**📊 数据集**

使用新构建的IVOE（Infrared–Visible Over‑Exposure）数据集（含2,315对合成样本与176对真实过曝样本）进行训练，同时在MSRS、FMB等公开数据集上进行测试。

**📈 对比分析**

与GANMcC、U2Fusion、SDNet、SegMif等SOTA方法对比，EPOFusion在EN、MI、VIF、Q^AB/F、SSIM等多项指标上均位居或接近榜首，且在下游检测/分割任务中显著提升性能。

**⚠️ 局限性**

局限性主要体现在：①模型参数与推理时间相对较高，需进一步优化；②对极端过曝或噪声污染严重场景的鲁棒性仍有限；③需要人工标注的过曝区域数据集规模有限，影响泛化。

---

## 456. Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty

**arXiv ID:** 2603.16538 | [PDF](https://arxiv.org/pdf/2603.16538v1)

**作者:** Mangyu Kong `[一作]` (Yonsei University), Euntai Kim `[通讯]` (Korea Institution of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文针对3D Gaussian Splatting（3DGS）在视觉定位中的姿态细化，深入探讨了先验姿态不确定性与几何不确定性对精度的影响，并提出了一种基于Monte Carlo采样与Fisher信息引导的PnP优化的无训练鲁棒重定位框架UGS‑Loc。

**💡 创新点**

创新点在于将姿态先验的不确定性通过Monte Carlo采样显式建模，并利用3DGS渲染中每个高斯原点的Fisher信息量化几何不确定性，在PnP-RANSAC阶段以不确定性加权采样，从而实现对噪声与几何误差的自适应抑制。

**🔧 技术方法**

使用技术包括：3DGS渲染、MASt3r或SuperPoint+LightGlue特征匹配、Monte Carlo Localization（MCL）式姿态采样、Fisher信息近似求几何不确定性、加权PnP-RANSAC、可视化与评估工具。

**📊 数据集**

实验数据集涵盖三大基准：7Scenes、12Scenes和Cambridge Landmarks（室内外混合），并在这些数据集上对不同姿态先验（APR、SCR、GS-CPR等）进行评估。

**📈 对比分析**

与多种基线（如PoseNet、DFNet、GS-CPR、MCLoc、NeRFMatch等）比较，UGS‑Loc在7Scenes、12Scenes和Cambridge Landmarks的[2 cm, 2°]阈值下平均精度提升约15–25%，在7Scenes上达到99.7% 5 cm/5°召回率，显著优于单一迭代或无不确定性方法；在迭代版GS‑CPR上提升约10%，表明改进主要来源于不确定性建模而非单纯多次优化。

**⚠️ 局限性**

局限性包括：为覆盖姿态不确定性仍需使用多粒子采样，导致推理时间较长（≈1.1 s/迭代，2.4 s/80迭代的MCLoc更慢）；依赖匹配器质量，对动态遮挡或极端光照变化的鲁棒性尚未充分验证；目前仅针对单一类型的3DGS模型，未探讨跨模型迁移或大规模场景下的可扩展性。

---

## 457. Multi-Agent Reinforcement Learning Counteracts Delayed CSI in Multi-Satellite Systems

**arXiv ID:** 2603.16470 | [PDF](https://arxiv.org/pdf/2603.16470v1)

**作者:** Marios Aristodemou `[一作]` (University of York), Lajos Hanzo `[通讯]` (University of Southampton)

**通讯引用:** 86981 | [OpenAlex ID](https://openalex.org/A5091122305)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了多卫星分布式基站在存在时延CSI下的下行链路，通过双阶段PPO实现分布式多天线系统的总速率最大化。

**💡 创新点**

提出DS-PPO双阶段近端策略优化，先用单卫星优化后共享奇异值进行协同第二阶段，直接映射延迟CSI到预编码矩阵，解决非IID环境和大连续动作空间问题。

**🔧 技术方法**

使用多智能体强化学习 (MARL)、Proximal Policy Optimization (PPO)、增广马尔可夫决策过程、奇异值分解、理论收敛分析与复杂度评估。

**📊 数据集**

基于Starlink类LEO星座仿真，随机生成K=2/4/6地面用户、L=4/6/8卫星，使用LoS信道模型、2 GHz频率，并模拟延迟T_d=3。

**📈 对比分析**

与完美CSI、延迟CSI下的IPPO、MAPPO以及基于CSI预测的SatCP+SatHB进行对比；DS-PPO在100个episode后实现≥300 Mbps平均总速率，提升≈75%相对IPPO，约3倍于SatCP方法。

**⚠️ 局限性**

当卫星数量超过一定阈值（如L=8）时性能下降，受非IID环境和动作空间增长影响；对手over影响有限；需进一步调优超参数，未在真实硬件上验证。

---

## 458. CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems

**arXiv ID:** 2603.15642 | [PDF](https://arxiv.org/pdf/2603.15642v1)

**作者:** Pearl Mody `[一作]` (Dwarkadas Jivanlal Sanghvi), Ruhina Karani `[通讯]` (Dwarkadas Jivanlal Sanghvi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对长期运行的LLM代理提出了CraniMem记忆架构，支持有门控、受限的多阶段编码、巩固与检索。

**💡 创新点**

创新点包括结合神经认知原理的门控筛选、双存储（短期缓冲+长期知识图）以及基于实用性与时间衰减的定时巩固与遗忘机制。

**🔧 技术方法**

采用门控/实用性标记、时间衰减、重放筛选、知识图链接、周期性优化循环等技术。

**📊 数据集**

在HotpotQA多跳问答数据集上加入噪声样本进行评测，并在多种指令调优LLM（Qwen、Gemma、Mistral）上进行实验。

**📈 对比分析**

与Vanilla RAG和Mem0基线在清洁与噪声两种评估下对比，CraniMem在F1、噪声衰减等指标上表现更好，但延迟略高。

**⚠️ 局限性**

局限性包括仅在100条样本上评测，统计显著性不足；缺乏对HippoRAG等其他记忆架构的系统对照；高延迟导致部署成本上升。

---

## 459. When Rolling Gets Weird: A Curved-Link Tensegrity Robot for Non-Intuitive Behavior

**arXiv ID:** 2603.16503 | [PDF](https://arxiv.org/pdf/2603.16503v1)

**作者:** Lauren Ervin `[一作]` (University of Alabama), Vishesh Vikas `[通讯]` (University of Alabama)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本研究设计并验证了一种采用半圆形弯曲链路、内部质量移动驱动的无绳 tensegrity 机器人 TeXploR2，能够实现动态滚动运动并通过非直观状态切换提升速度；

**💡 创新点**

创新点在于：①将弯曲链路与内部质量平衡结合，实现了高效滚动与稳定性平衡；②引入非直观状态切换机制，显著降低了状态转换所需的质量位移；③采用几何静态建模与螺旋理论解析四状态滚动序列，并验证了弹性结构的冲击吸收能力；

**🔧 技术方法**

主要技术包括：几何静态建模（利用Screw Theory与Lie群）、MATLAB quasistatic 仿真、内部质量驱动的双轴 NEMA23 步进电机、Onyx 3D 打印半圆形链路、BNO085 IMU 与 Vicon 动作捕捉、冲击试验与数据分析；

**📊 数据集**

未使用公开数据集，实验数据来自自制的 TeXploR2 机器人、Vicon 运动捕捉系统（误差≤0.3 mm）及冲击试验记录；

**📈 对比分析**

与传统直链路 tensegrity 机器人相比，TeXploR2 在同等尺寸下滚动速度提升至 0.71 BL/s，动态实验中通过非直观状态切换可达 1.88 BL/s；相对 4 种直链路设计，其速度约提升 3 倍；

**⚠️ 局限性**

局限性包括：①实验仍处于实验室环境，未验证复杂地形适应性；②缺乏闭环反馈控制，运动精度受限；③仅实现两链路，无法充分展示多接触点的混合系统优势；④对材料与尺寸的依赖限制了在极端环境（极寒、沙漠等）的可行性；

---

## 460. IQuest-Coder-V1 Technical Report

**arXiv ID:** 2603.16733 | [PDF](https://arxiv.org/pdf/2603.16733v1)

**作者:** Jian Yang `[一作]`, Bryan Dai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 IQuest‑Coder 系列（7B–40B）代码语言模型，采用预训练→中训练→后训练的四阶段管线，并设计 LoopCoder 循环 Transformer 结构，支持大上下文（32k/128k）和自我推理、工具使用等能力。

**💡 创新点**

创新点包括：① 将仓库演化序列（commit‑triplet）作为训练信号，显著提升规划与长程推理；② 在中训练阶段引入 32k/128k 长上下文的推理、Agent 路径和 FIM 任务，实现对多文件代码的跨文件推断；③ 后训练分为 “Thinking” 与 “Instruct” 两条路径，前者结合 RL‑reinforced 思维轨迹，后者强化指令遵循；④ LoopCoder 通过循环 Transformer 共享参数、双注意力门控，实现高效长段推理。

**🔧 技术方法**

技术手段包括：多阶段预训练（通用+代码→高质量代码 annealing）；双相中训练（32k→128k 上下文）配合推理、Agent、FIM 数据；后训练 SFT + RL（GRPO、CLIP‑Higher 策略）实现思维与指令优化；循环 Transformer 结构、共享参数、门控全局/局部注意力；大规模 GPU 训练架构（fused gated attention、KV 分片、容错检测）等。

**📊 数据集**

数据集涵盖：Common Crawl、GitHub 代码库、技术文档、公开 API、CodeSimpleQA‑Instruct（问答）、仓库演化三元组、FIM（文件级、仓库级）、推理 QA（数学、编程、逻辑）、Agent 路径、SWE‑RL 等；评测使用 CrossCodeEval、HumanEval、MBPP、BigCodeBench、FullStackBench、LiveCodeBench、CRUXEval、SWE‑Bench、Terminal‑Bench、Text‑to‑SQL（Spider/BIRD）、Agentic 任务（Mind2Web、BFCL V3）、安全基准（Tulu‑3、XSTest、WildGuard）等。

**📈 对比分析**

与 Claude 4.5 Sonnet、GPT‑5.1、Gemini‑3、Qwen‑系列、DeepSeek‑Coder、StarCoder2、Kimi‑Dev 等开源/闭源模型对比，IQuest‑Coder‑40B 在多项代码生成、推理与 Agent 任务上均位居前列（如 HumanEval+ 93.5%/88.2%/91.5%/98.3% 等），且在安全性评测中拒绝率与合规率平衡良好，整体性能明显优于现有开源同类模型。

**⚠️ 局限性**

局限性：① 训练与推理成本高，需数千 GPU‑小时及大规模算力；② 主要关注 Python/Java/TypeScript/C# 等主流语言，跨语言迁移仍有待验证；③ 对极大规模项目的全文件协作、实时调试等极端情景尚未彻底评估；④ 安全评测中对复杂攻击场景的覆盖仍有限。

---

## 461. Kestrel: Grounding Self-Refinement for LVLM Hallucination Mitigation

**arXiv ID:** 2603.16664 | [PDF](https://arxiv.org/pdf/2603.16664v1)

**作者:** Jiawei Mao `[一作]` (University of California Santa Cruz), Yuyin Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种训练自由的视觉语言模型幻觉减缓框架，通过显式视觉基准和结构化证据驱动的迭代自我完善来校正回答。

**💡 创新点**

创新点在于将外部视觉基准代理与可验证的证据链结合，并通过多轮保守的证据门控更新来避免过度校正，实现可解释的幻觉诊断。

**🔧 技术方法**

使用技术包括SAM3视觉基准、文本生成证据、LVLM判断者进行命题级验证、以及基于证据置信度的迭代自我校正。

**📊 数据集**

实验数据集为 POPE (MS-COCO, A-OKVQA, GQA) 与 MME-Hallucination 等。

**📈 对比分析**

与现有解码调控与工具利用的训练自由方法相比，在 Qwen3-VL 与 InternVL3.5 上分别提升 POPE 3.3% 与 MME 28.34 分，显著超过 OPERA、VCD、RITUAL 等基线。

**⚠️ 局限性**

局限在于推理时延较高、需额外的外部工具和多轮计算，且在极少数情况下仍可能出现误修正或无法获得足够证据的情形。

---

## 462. Robust Physics-Guided Diffusion for Full-Waveform Inversion

**arXiv ID:** 2603.16393 | [PDF](https://arxiv.org/pdf/2603.16393v1)

**作者:** Jishen Peng `[一作]` (Shanghai Jiao Tong University), Xiongbin Yan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理引导的扩散框架，用于全波形反演，结合了基于评分的生成先验和通过波动方程模拟计算的似然引导。

**💡 创新点**

创新点在于采用了基于最优传输的数据信息潜力，结合了受限加权和观察依赖的归一化，从而提高了对幅度不平衡和时间/相位错位的鲁棒性。

**🔧 技术方法**

使用了基于评分的生成模型和最优传输（OT）技术，结合了预条件引导的反向扩散方案。

**📊 数据集**

在OpenFWI数据集上进行了数值实验，展示了在相似计算预算下相较于确定性优化基线和标准扩散后验采样（DPS）更好的重建质量。

**📈 对比分析**

与确定性优化基线和标准DPS进行了比较，结果显示提出的方法在重建质量和稳定性上均有显著提升。

**⚠️ 局限性**

限制在于物理引导的计算成本仍然较高，未来需要开发更高效的实现和加速策略。

---

## 463. MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation

**arXiv ID:** 2603.16861 | [PDF](https://arxiv.org/pdf/2603.16861v1)

**作者:** Abhay Deshpande `[一作]` (Allen Institute for AI), Ranjay Krishna `[通讯]` (Princeton University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一套完整的开源数据生成管道MolmoSpaces+MolmoBot，用以在仿真环境中大规模生成多任务、跨机器人（Franka FR3与Rainbow RB-Y1）的专家轨迹，并在此基础上训练三类政策（基于Molmo2+DiT的VLM、π_0复制版本以及轻量Transformer SPOC），实现零射（zero‑shot）从仿真到真实世界的操控迁移。

**💡 创新点**

创新点包括：① 通过程序化生成与域随机化的仿真数据实现无需真实演示即可训练通用操控政策；② 将VLM与流匹配动作头（DiT）结合，首次将大规模视觉‑语言模型直接用于连续动作预测；③ 提供了完整的1.8M轨迹数据集与训练代码，填补了当前基于真实数据的闭源瓶颈；④ 通过多相机、多感知输入与多帧训练提升了策略的鲁棒性。

**🔧 技术方法**

技术手段主要有：MuJoCo仿真+MolmoSpaces场景渲染、程序化物体放置与域随机化；行为克隆（BC）与流匹配动作头（DiT）实现动作预测；VLM Molmo2与SigLIP视觉编码器、双向/因果注意力结合；π_0架构复制与轻量Transformer SPOC；多帧输入、动作分块预测、量化动作分类；对机器人状态与视觉信息的交叉注意力。

**📊 数据集**

数据集：自研的1.8M模拟专家轨迹，包含约300M帧、94k独立环境、11.4k目标物体与9.4k接收器，覆盖Pick、Pick‑and‑Place、Pick‑and‑Place‑Next‑To、Pick‑and‑Place‑Color、Open、Open‑Door等任务。评估平台使用Franka FR3（桌面操作）与Rainbow RB‑Y1（移动操作）。

**📈 对比分析**

对比方法：与DROID平台上的π_0、π_0.5、StereoVLA、LAP‑VLA、X‑VLA等现有VLA模型对比；在真实世界DROID环境中，3‑帧VLM模型在Pick‑and‑Place任务取得79.2%成功率，远超π_0.5的39.2%；在Franka FR3上同样取得79.2%；在RB‑Y1门开启任务中，Door Specialist取得77.7%成功率。模拟评估中，3‑帧VLM平均成功率64.4%，π_0.5零射仅31.3%。

**⚠️ 局限性**

局限性：① 仅适用于刚体及关节对象的操作，缺乏对软体、插接、精细触控等高接触动力学的支持；② 依赖高质量物理仿真，仍可能面临模拟与真实差异导致的未见情况；③ 对环境多样性（房屋数量）影响有限，表明模型更多依赖交互量而非场景多样性；④ 需要进一步探索更大规模、更多机器人类型与任务的可迁移性。

---

## 464. Grid-World Representations in Transformers Reflect Predictive Geometry

**arXiv ID:** 2603.16689 | [PDF](https://arxiv.org/pdf/2603.16689v1)

**作者:** Sasha Brenner `[一作]` (Max Planck Institute for Human Cognitive and Brain Sciences), Nico Scherf `[通讯]` (Max Planck Institute for Human Cognitive and Brain Sciences)

**通讯引用:** 1050 | [OpenAlex ID](https://openalex.org/A5085986753)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练decoder-only transformer模型，利用从二维格子上受限随机行走过程采样的前缀序列，研究其内部表示如何捕捉该过程的最优预测信息；

**💡 创新点**

提出并验证在极简的随机行走任务中，transformer能够自动学习到与世界模型和语法约束对应的低维预测向量，揭示预测几何与内部表示之间的内在关联；

**🔧 技术方法**

采用transformer架构、next-token交叉熵训练、R^2与lCKA两种表示相似度度量、PCA分析内在维度以及对随机行走过程的解析推导；

**📊 数据集**

使用人工生成的受限随机行走数据集，六种不同时间边界和终点配置，每条轨迹长度固定且必达终点；

**📈 对比分析**

通过R^2与lCKA衡量层激活与解析得到的预测向量的一致性，发现大多数层与最终LayerNorm的对齐度高，且训练损失接近理论下界，层内维度最终降至2；

**⚠️ 局限性**

局限在于仅在极简的无歧义随机行走任务上验证，缺乏对更复杂语言结构的推广与泛化评估，且依赖可解析的生成过程，难以直接扩展到非平稳或高阶语言模型。

---

## 465. SOMP: Scalable Gradient Inversion for Large Language Models via Subspace-Guided Orthogonal Matching Pursuit

**arXiv ID:** 2603.16761 | [PDF](https://arxiv.org/pdf/2603.16761v1)

**作者:** Yibo Li `[一作]` (Politecnico di Milano), Qiongxiu Li `[通讯]` (Aalborg University)

**通讯引用:** 357 | [OpenAlex ID](https://openalex.org/A5062097625)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可扩展的梯度逆向攻击框架SOMP，能够在聚合梯度下从大型语言模型中恢复多样本文本

**💡 创新点**

创新点在于将聚合Transformer梯度视为稀疏信号，在头层查询梯度的子空间中引导候选词池，并通过稀疏匹配（OMP）在梯度空间中精确重构混合梯度，显著提升长序列和大批量场景下的重构质量

**🔧 技术方法**

利用头层查询梯度的子空间投影、几何引导束搜索、头级稀疏性评分以及OMP稀疏重构；整体实现分为三阶段：Token池构建、几何束搜索和梯度空间稀疏重构

**📊 数据集**

在GPT‑J‑6B、GPT‑2、Qwen3‑8B等多种LLM上，使用IMDB、CoLA、SST‑2等标准数据集，并扩展到五种语言以及FedAvg训练设置

**📈 对比分析**

与DAGER、LAMP、GRAB等基线比较，SOMP在大批量（B≥4）和长序列下ROUGE‑L显著提升（如GPT‑J‑6B IMDB上从58%提升至73%+），在更大批量（B=16）仍能保持高质量重构；在耗时上对长序列更友好

**⚠️ 局限性**

受限于头层查询梯度的可分离性，若梯度被压缩或噪声严重，Token池可能提前剪枝；在短序列/小批量场景下由于OMP开销，速度可能不占优；未对DP‑SGD等正式差分隐私机制进行完整评估

---

## 466. Micro-AU CLIP: Fine-Grained Contrastive Learning from Local Independence to Global Dependency for Micro-Expression Action Unit Detection

**arXiv ID:** 2603.16302 | [PDF](https://arxiv.org/pdf/2603.16302v1)

**作者:** Jinsheng Wei `[一作]` (Nanjing University of Posts and Telecommunications), Guoying Zhao `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于 CLIP 的微表情 AU 检测框架 Micro‑AU CLIP，采用“从局部独立到全局依赖”的学习范式，实现微 AU 的细粒度识别，并可无情绪标签完成微表情识别。

**💡 创新点**

创新点包括：
• 将 AU 识别拆解为局部语义独立建模（LSI）和全局语义依赖建模（GSD），两者互补提升性能；
• 设计 Patch Token Attention（PTA）在 LSI 中对 AU 区域内的 patch 进行加权聚合；
• 引入 Global Dependency Attention（GDA）与 Global Dependency Loss（GDLoss）在 GSD 中捕捉 AU 之间的条件依赖；
• 设计 Micro‑AU Contrastive Loss（MiAUCL）实现视觉‑文本的细粒度对齐，弥补 CLIP 在微 AU 上的弱点；
• 通过文本提示实现情绪标签自由的零样本微表情识别。

**🔧 技术方法**

技术手段包括：CLIP 视觉与文本编码器、ViT‑B/32、光流输入、Patch Token Attention、Global Dependency Attention、MiAUCL 对比损失、GDLoss、以及多任务损失融合。

**📊 数据集**

使用公开微表情数据集 CASME II（255 条样本，8 个 AU）和 SAMM（159 条样本，4 个 AU）进行 AU 检测与 MER 评估。

**📈 对比分析**

与 SOTA 方法（ResNet18、AUFormer、AU‑LLM 等）对比，Micro‑AU CLIP 在 CASME II 的 F1 得到 0.782，SAMM 0.730，表现优于绝大多数方法，且跨数据集差距更小；在零样本 MER 上 F1 分别为 0.889（CASME II）和 0.747（SAMM），与有监督或弱监督方法相当甚至超越部分。

**⚠️ 局限性**

局限性：
• 需要对文本编码器进行微调，增加训练复杂度；
• 依赖于 AU 的样本分布，稀有 AU 的检测仍受限；
• 零样本 MER 的表现仍低于最优的有监督方法，说明对情绪标签的直接利用仍具优势；
• 该框架在大规模多模态或更高帧率的微表情数据上的泛化性尚待进一步验证。

---

## 467. SEAHateCheck: Functional Tests for Detecting Hate Speech in Low-Resource Languages of Southeast Asia

**arXiv ID:** 2603.16070 | [PDF](https://arxiv.org/pdf/2603.16070v1)

**作者:** Ri Chi Ng `[一作]` (Singapore University of Technology and Design), Roy Ka-Wei Lee `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 1684 | [OpenAlex ID](https://openalex.org/A5089793938)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了SEAHateCheck数据集，提供了针对印尼、菲律宾、泰国和越南等低资源东南亚语言的金标与银标功能测试案例，并使用该套件对多种大语言模型进行系统评估。

**💡 创新点**

创新点在于：①首次面向东南亚多语言场景的功能测试框架；②结合本土专家和LLM进行翻译、生成与人工校验，实现文化特定仇恨表达与受保护群体的精准覆盖；③引入金标/银标双层验证，提升评测生态有效性。

**🔧 技术方法**

主要技术包括：基于HateCheck与SGHateCheck的功能测试方法；使用GPT‑3.5、Gemini 1.5 Pro、GPT‑4o及SEA‑Lionv2.1进行多阶段机器翻译和案例生成；采用LoRA对开源LLM进行域适配；使用一致性标注与Fleiss κ评估质量。

**📊 数据集**

数据集：SEAHateCheck（13,579条金标、19,802条银标）与SGHateCheck；对比模型：开源Llama3/8b、Sealion、Seagem、Pangea、Qwen、Gemma、MiniS、Deepseek；闭源模型：o3、Gemini。

**📈 对比分析**

在金标测试中，闭源模型平均F1在80–90%，开源模型在70–80%；微调后提升5–15%但在泰语、印尼等语言出现退化；银标测试揭示模型在隐性仇恨、反仇恨、否定等功能上的显著弱点，且对模型的鲁棒性提供更逼真的诊断。

**⚠️ 局限性**

主要局限包括：①模板化生成导致金标僵化，银标质量相对较低；②LLM生成受安全过滤影响，可能偏离真实仇恨表达；③缺乏对代码混合与口语对话场景的覆盖；④受当前法律框架约束，识别的受保护群体可能滞后。

---

## 468. Electrodermal Activity as a Unimodal Signal for Aerobic Exercise Detection in Wearable Sensors

**arXiv ID:** 2603.15880 | [PDF](https://arxiv.org/pdf/2603.15880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 469. Perceptual Requirements for Low-Latency Head-Mounted Displays

**arXiv ID:** 2603.15796 | [PDF](https://arxiv.org/pdf/2603.15796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 470. On the Transfer of Collinearity to Computer Vision

**arXiv ID:** 2603.16592 | [PDF](https://arxiv.org/pdf/2603.16592v1)

**作者:** Frederik Beuth `[一作]` (Chemnitz University of Technology), Danny Kowerko `[通讯]` (Chemnitz University of Technology)

**通讯引用:** 812 | [OpenAlex ID](https://openalex.org/A5038259339)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了将人类视觉中的collinearity（共线性）原理迁移到计算机视觉领域的神经网络模型，并在半导体晶圆缺陷检测、纳米材料纤维识别、遮挡识别以及ImageNet子集等四个典型任务中进行系统验证。

**💡 创新点**

创新点：
1) 设计了包含Gabor滤波、池化与专门的collinearity层的端到端模型，collinearity层通过基于心理学实验的连接范围（5λ~14λ）和宽度（≈2λ）实现边缘共线性增强。
2) 首次将该模型与深度学习、特征检测与显著性模型相结合，证明其在工业视觉任务中的实用性。
3) 提供了“适用场景清单”，系统划分了collinearity在行业与通用数据集中的优势与局限。

**🔧 技术方法**

技术与方法：
- Gabor滤波器提取方向性边缘特征；
- 最大/双线性池化；
- 横向连接实现collinearity增强，采用乘法型调制；
- 与CNN、saliency模型融合；
- 通过误差率/准确率对比评估。

**📊 数据集**

数据集：
- 半导体晶圆缺陷图像（工业内部自有数据集）；
- 纳米技术材料纤维图像（自有实验数据集）；
- ImageNet子集（公开数据集）；
- 可能还使用了自定义对比度/长度实验图像。

**📈 对比分析**

比较方法与性能：
- 与仅使用CNN或传统特征检测方法对比；
- 半导体晶圆缺陷：错误率从6.5%降至5.26%，提升1.24×；
- 纳米材料纤维：错误率从21.65%降至6.64%，提升3.2×；
- ImageNet子集：未出现显著性能提升；
- 通过可视化、曲线拟合等方式验证心理学实验的重复性。

**⚠️ 局限性**

局限性：
- 仅对线性/直线结构有效，对圆形或复杂纹理缺乏适应性；
- 需要手动调节连接范围与权重，缺乏自适应机制；
- 未包含抑制效应，可能在某些情境下产生过度增强；
- 在大规模、类目多样的通用数据集（如完整ImageNet）中表现不佳，说明该原理对“人造线条”结构的依赖较强。

---

## 471. The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data

**arXiv ID:** 2603.16177 | [PDF](https://arxiv.org/pdf/2603.16177v1)

**作者:** Christina Baek `[一作]`, Pratyush Maini `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种在预训练阶段少量混合领域数据、并多次重复的专门化预训练（SPT）策略，并与传统仅在微调阶段加入领域数据的方法进行了对比。

**💡 创新点**

创新点在于证明在预训练阶段引入少量领域数据可以显著提升领域性能且保留通用知识，同时提出了针对混合比例和计算预算的过拟合缩放律，用以预测最优的领域数据混合比例。

**🔧 技术方法**

主要技术包括在预训练过程中将领域数据占比δ混入通用数据、采用多轮重复训练、早停、学习率调度，使用OLMo等大规模语言模型架构，并构建功率律模型预测过拟合与测试损失。

**📊 数据集**

使用的数据集包括约300M tokens的专业领域数据（ChemPile、MusicPile、ProofPile）以及通用预训练语料Dolma，此外还利用英文/日语并行文本做对照实验。

**📈 对比分析**

通过与标准预训练+微调（NPT）方案比较，评估领域测试损失、通用知识遗忘、下游任务准确率；结果显示SPT在相同计算预算下可将预训练token数减少至1.75×，1B SPT在领域任务上超过3B标准模型，整体域内测试损失下降约1–2%，并保持通用性能。

**⚠️ 局限性**

限制在于需要额外的预训练成本，混合比例需根据领域与计算预算进行调优，过拟合缩放律仅基于实验数据拟合，可能不适用于极小域数据或极大模型；同时缺乏对早期领域曝光为何更有效的深入理论解释。

---

## 472. Test Code Review in the Era of GitHub Actions: A Replication Study

**arXiv ID:** 2603.15935 | [PDF](https://arxiv.org/pdf/2603.15935v1)

**作者:** Hui Sun `[一作]` (North Carolina State University), Kathryn T. Stolee `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

复现并扩展了Spadini等人在Gerrit平台上对测试代码审查的研究，首次在GitHub Pull Request（PR）模式下检视测试代码的审查分布、主题与顺序，并进一步探讨GitHub Actions（GHA）自动化对审查行为的短期与长期影响。

**💡 创新点**

创新点在于：①首次在PR模式下复制原研究，验证平台差异对审查偏好的影响；②通过时间截断设计与混合效应模型，揭示GHA采用后测试代码审查的即时“激增”与随后衰退的动态；③将定量统计与定性内容分析相结合，提出六类审查主题并对比Gerrit与GitHub的差异；④构建公开复现包，提升可重复性。

**🔧 技术方法**

使用的技术与方法包括：GitHub REST API抓取PR与评论数据，文本清洗与正则过滤；统计指标如odds ratio、日志比率、评论密度；回归不连续设计（RDD）、中断时间序列分析、混合效应回归；内容分析编码书（六大主题）与卡片排序提高可靠性；非参数检验（Chi‑square）评估主题分布差异。

**📊 数据集**

数据集来源于六个大型开源项目（Python：Pandas；Go：Moby；Scala：Spark；Java：Flink；TypeScript：VSCode；C++：TensorFlow），共计213,511 PR，其中68,892 PR被保留用于RQ1（含至少一条人类评论且涉及测试或生产文件），6,166 PR用于RQ3（测试与生产文件共改并包含评论）。该数据集涵盖了不同语言、社区与历史的多样性。

**📈 对比分析**

比较方法：①平台层面将原Gerrit数据与GitHub数据的odds ratio、平均评论数对比；②时间截断设计评估GHA采用前后对log(OR)的即时跳跃与长期趋势；③对比不同项目的中断模型，验证VSCode等异常点；④内容分析中对比预GHA与后GHA的主题分布，使用卡方检验判断显著性。结果显示：GitHub PR对测试与生产文件的关注更均衡；GHA短期内提高测试文件的审查率和密度，但在Pandask与Spark等项目中长期呈现下降；总体评论密度低于Gerrit。

**⚠️ 局限性**

局限性包括：①项目异质性强，VSCode等项目偏离平均趋势，影响RDD的平行趋势假设；②仅包含公开大型开源项目，缺少私有或小型项目的验证；③对测试文件的识别依赖文件路径与命名规则，可能存在误判；④内容分析受编码主观性限制，尽管已做多轮校准；⑤GHA采用时间点以CI配置文件出现为标识，可能忽略先前的手工或其他CI；⑥分析主要基于PR评论，未考察代码合并后真实缺陷率的变化。

---

## 473. CAST-TTS: A Simple Cross-Attention Framework for Unified Timbre Control in TTS

**arXiv ID:** 2603.16280 | [PDF](https://arxiv.org/pdf/2603.16280v1)

**作者:** Zihao Zheng `[一作]` (Shanghai AI Lab), Xuenan Xu `[通讯]` (Shanghai AI Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的语音合成框架CAST‑TTS，能够通过语音或文本提示实现声色（timbre）控制，利用单一的跨注意力机制实现对两种模态的融合。

**💡 创新点**

创新点在于：① 用单个跨注意力模块同时处理语音与文本提示，避免复杂的多分支设计；② 通过轻量投影器将文本嵌入映射到语音声色空间，实现跨模态对齐；③ 采用多阶段训练策略先预训练语音模态，再对齐文本模态，最后联合微调，显著提升对齐质量。

**🔧 技术方法**

技术方案包括：预训练的 ECAPA‑TDNN 声音编码器、Flan‑T5 文本编码器、ConvNeXt V2 字符编码、Transformer flow‑matching 主干、BigVGAN 语音解码器及分类器无引导（CFG）增强生成质量。

**📊 数据集**

数据集主要为 LibriTTS‑R（语音提示训练），CapTTS‑LibriTTS‑R 与 GigaSpeech（文本提示训练），以及 LibriSpeech‑PC test‑clean 和 CapTTS 测试集用于评估。

**📈 对比分析**

与 F5‑TTS‑v1、MaskGCT、ZipVoice‑L（语音提示）以及 CapSpeech‑NAR、Parler‑TTS‑Large（文本提示）进行对比。使用 WER、SPK‑Sim、Style‑ACC、UTMOS、MOS 等指标，CAST‑TTS 在语音提示下获得最高 SPK‑Sim，文本提示下获得最优 Style‑ACC，整体性能与专用单模态模型相当或更好。

**⚠️ 局限性**

主要限制是缺乏对情感、口音等属性的控制，受限于训练数据多样性；此外，在少数年龄组（儿童、青少年、老年）上的表现仍待进一步提升。

---

## 474. W2T: LoRA Weights Already Know What They Can Do

**arXiv ID:** 2603.15990 | [PDF](https://arxiv.org/pdf/2603.15990v1)

**作者:** Xiaolong Han `[一作]` (University of Surrey), Zehong Wang `[通讯]` (University of Notre Dame)

**通讯引用:** 1284 | [OpenAlex ID](https://openalex.org/A5006927984)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 W2T 框架，先通过 QR–SVD 规范化 LoRA 权重得到唯一的 Canonical 表示，再将每个秩分量转化为 token，并使用双层 Transformer 在权重空间中学习嵌入，用于直接推断 LoRA 的属性、性能与相似度；

**💡 创新点**

首次在数据层面解决 LoRA 的 GL(r) 重参数化歧义，构建可唯一的 Canonical 结构化表示，并将其转化为可被标准 Transformer 学习的 token；

**🔧 技术方法**

采用 QR 与 SVD 联合分解、模版化 tokenization、rank‑level 与 position‑level Transformer、权重空间学习 (WSL) 与 GLNet 对比等技术；

**📊 数据集**

使用 Stable Diffusion v1.4 上的 CelebA‑LoRA 与 CUB‑LoRA、Llama‑3.2‑3B 上的 GoEmotions‑LoRA 与 ARC‑LoRA，以及包含 ARC‑Challenge、BoolQ、GSM8K 与 MBPP 的混合任务池；

**📈 对比分析**

与 GLNet、MLP、CNN、ViT 等基线对比，评价指标包括 macro‑F1、MAE、NDCG@10 等；W2T 在属性分类、性能预测和检索任务上均优于基线，且在迁移到新基础模型时仍保持高性能；

**⚠️ 局限性**

仅针对单一 LoRA 检查点，未考虑合并或组合适配器的情况；主要验证判别任务，对生成任务的适用性不明；检索实验为简化版本，缺乏大规模真实系统评估。

---

## 475. Volumetrically Consistent Implicit Atlas Learning via Neural Diffeomorphic Flow for Placenta MRI

**arXiv ID:** 2603.16078 | [PDF](https://arxiv.org/pdf/2603.16078v1)

**作者:** Athena Taymourtash `[一作]` (Massachusetts Institute of Technology), Polina Golland `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 21930 | [OpenAlex ID](https://openalex.org/A5081763875)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种体积一致的隐式模型，联合学习胎盘的共享模板和神经微分同胚流，实现个体胎盘在统一坐标系中的完整重建、密集对应与体素级强度映射。

**💡 创新点**

通过在隐式函数上施加体积正则化（雅可比行列式约束和双调和惩罚）以及对内部体素的采样，首次将隐式注册从仅表面扩展到整个器官体积，显著降低了内部折叠并保证了全域同胚连续性。

**🔧 技术方法**

采用深度隐式函数（DIT）、神经ODE实现的可逆流场、四步神经ODE变形器、体积正则化（Jacobian、双调和）以及基于四面体嵌入的内部采样进行训练；训练损失包括重建、变形幅度、Jacobian、双调和及潜在正则化。

**📊 数据集**

使用111幅胎盘MRI扫描（78例孕妇，26–38周），包含60例单胎和18例双胎，另外33例单胎在两种孕姿（仰卧与左侧卧）下采集，构成66个独立胎盘形状。

**📈 对比分析**

与DeepSDF、DIF-Net、DIT、NDF等基线方法对比，使用Chamfer、EMD、Normal Consistency、FlipRate、logDet‑L1、CycleError等指标评估；本文方法在表面精度相当甚至更优，FlipRate降至4.04%，logDet‑L1与CycleError显著下降，体积、面积与对称Dirichlet等变形度量亦优于基线。

**⚠️ 局限性**

推断阶段仍需潜在向量优化，导致推理速度受限；模型仅采用单一全局模板，难以捕捉子类型差异；未针对时间序列或动态MRI展开，未来需扩展多模板和时空分析。

---

## 476. Mediocrity is the key for LLM as a Judge Anchor Selection

**arXiv ID:** 2603.16848 | [PDF](https://arxiv.org/pdf/2603.16848v1)

**作者:** Shachar Don-Yehiya `[一作]` (Hebrew University of Jerusalem), Omri Abend `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估中研究 Anchor（锚模型）选择对 LLM-as-a-Judge 评估结果的影响，并量化其对排名准确性的作用。

**💡 创新点**

发现 Anchor 选择呈现倒 U 型关系：极强或极弱模型均为最差锚，优选中等水平 Anchor 能显著提升 Kendall τ 相关性；同时提出基于锚信息量的评估预算估算、样本量需求及 Anchor 选择的实用指南。

**🔧 技术方法**

使用 LLM 判断器（Deepseek‑v3、GPT‑OSS、Qwen3 等），win‑rate 与 Bradley‑Terry 聚合，Kendall τ 相关性评估，功效分析（sign/Wilcoxon 试验），以及信息量计算与样本量推导。

**📊 数据集**

主要数据集为 Arena‑Hard‑v2.0（750 条指令）和 AlpacaEval（805 条指令），并生成约 900K 条 LLM 判断结果用于实验。

**📈 对比分析**

与完整二次评估（quadratic）及人类评估对比，Anchor 选择可使 Kendall τ 相关性提升多达 0.19；Anchor 与 Judge 的影响力相当；通过样本量和信息量分析，说明标准 benchmark 大小不足以可靠区分竞争模型。

**⚠️ 局限性**

局限：金标准排名基于二次评估，可能与人类排名偏离；仅使用开源 LLM 判断器，结果对商业模型的推广有限；评估假设转移性（transitivity）在个体级不一定成立；Anchor 信息量上限约 0.5，导致评估预算浪费。

---

## 477. Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets

**arXiv ID:** 2603.16731 | [PDF](https://arxiv.org/pdf/2603.16731v1)

**作者:** Kristi Topollai `[一作]` (New York University), Anna Choromanska `[通讯]` (New York University)

**通讯引用:** 2564 | [OpenAlex ID](https://openalex.org/A5006452373)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究量化优化器状态下指数移动平均(EMA)的停滞现象，并提出预测模型和周期性重置策略。

**💡 创新点**

首次将EMA停滞建模为有效衰减并量化一阶二阶时序响应窗口，从而给出可解释的重置周期。

**🔧 技术方法**

采用指数移动平均、近似高斯梯度分布、有效步长分析、随机舍入、周期性重置和实测模拟等技术。

**📊 数据集**

在LLaMA模型的WikiText与C4数据集上进行预训练实验。

**📈 对比分析**

与全精度AdamW对比，采用低精度BF16/FP8/FP4并结合随机舍入与重置后可恢复或提升训练损失，内存减少至87.5%。

**⚠️ 局限性**

分析局限在仅考虑EMA停滞机制、近似假设与对不同优化器/规模的推广有限。

---

## 478. An approximate graph elicits detonation lattice

**arXiv ID:** 2603.16524 | [PDF](https://arxiv.org/pdf/2603.16524v1)

**作者:** Vansh Sharma `[一作]` (University of Michigan), Venkat Raman `[通讯]` (University of Michigan)

**通讯引用:** 5661 | [OpenAlex ID](https://openalex.org/A5036513384)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于图论的训练无关算法，利用SAM模型对三维压力轨迹进行精准分割并构建三维点火格子（detonation lattice），从而实现三维细胞结构的自动检测与量化。

**💡 创新点**

创新点在于：①将点火格子抽象为节点与边的图结构，保留三维拓扑信息；②结合无监督SAM分割与基于物理约束的边筛选（Cluster、Between）实现高鲁棒性；③首次实现从三维模拟或实验数据直接得到三维细胞尺寸、形状与方差统计。

**🔧 技术方法**

主要技术包括：SAM（Segment‑Anything）与Cellpose‑SAM融合的实例分割模型；离散化网格与KD‑Tree搜索；基于物理一致性的边构造算法（前向搜索、度上限、占据率测试）。

**📊 数据集**

使用了两类数据集：①合成的三维椭球体网格，用来验证分割精度；②化学计量乙烯‑空气混合物的三维数值仿真数据，真实捕捉三维细胞结构。

**📈 对比分析**

与传统二维手工计数、基于PSD或图像阈值的二维方法相比，该方法在三维尺度上将细胞体积误差降至约2%，并能捕捉细胞的轴向伸长和方差特征；在合成数据中，分辨率达到原始尺度时误差低于3%，在高分辨率下可进一步减至0.7%。

**⚠️ 局限性**

局限性包括：对极其复杂或高度碎片化的细胞格子仍易产生过度分割；高分辨率输入显著增大文件大小和计算开销；算法对前向方向与阈值的选择较为敏感，需要针对不同实验条件进行手动调参。

---

## 479. ExpressMind: A Multimodal Pretrained Large Language Model for Expressway Operation

**arXiv ID:** 2603.16495 | [PDF](https://arxiv.org/pdf/2603.16495v1)

**作者:** Zihe Wang `[一作]` (Beihang University), Yongxin Tong `[通讯]` (Beihang University)

**通讯引用:** 11653 | [OpenAlex ID](https://openalex.org/A5051874566)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个面向高速公路运营的多模态大型语言模型ExpressMind，能够完成交通法规问答、事故原因与应急方案链式推理、视频事件检测与描述等任务。

**💡 创新点**

创新点包括：①行业首个全栈高速公路数据集；②双层预训练+基于GRPO的RL对齐Chain‑of‑Thought机制；③图增强检索RAG动态知识库；④视觉优先对齐VPA提升视频理解；以及针对高速公路场景的专属多模态编码与跨模态注意力。

**🔧 技术方法**

采用了Qwen基础模型，结合自监督+全参数SFT、GRPO强化学习、LightRAG图检索、MROPE+DeepStack多层视觉编码、VPA视觉优先跨模态注意力等技术。

**📊 数据集**

使用Express‑Insight（海量文本）、Express‑QA（QA对）、Express‑IncidentCoT（事故链式思维）、Express‑VQA（视频+问答）四个子集构成的全栈高速公路数据集。

**📈 对比分析**

在三类知识问答（法规、智能高速公路、ITS）和视频事件检测/描述任务中，ExpressMind与Qwen‑32B、Llama‑3.3‑70B等基线对比，均实现了更高的准确率、F1、GPT‑Score及BLEU/ROUGE/CIDEr/BERTScore，并将推理延迟压至约13.2 ms，表现显著优于传统模型。

**⚠️ 局限性**

局限性包括对大量标注数据的依赖、模型规模仍偏大不易部署于边缘设备、长文本链式推理的通用性与鲁棒性待进一步提升，以及对实时时空动态推理的支持仍不充分。

---

## 480. The Importance of Being Smoothly Calibrated

**arXiv ID:** 2603.16015 | [PDF](https://arxiv.org/pdf/2603.16015v1)

**作者:** Parikshit Gopalan `[一作]` (Apple), Pranay Tankala `[通讯]` (Harvard)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5084760979)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了平滑校准（smooth calibration）在二分类预测中的理论性质，提出了针对平滑校准预测器的新全局预测（omniprediction）保证，并给出了平滑校准误差与不同校准距离（如上限距离、下限距离）的精确关系。

**💡 创新点**

创新点包括：① 在不需要完美校准的前提下，证明平滑校准预测器经过随机噪声平滑后即可获得对所有后处理函数的低 regret 预测保证；② 用地球移动距离（Earth‑Mover Distance）给出了下限校准距离的简洁等价定义，提供了更直观的证明；③ 展示了上限校准距离在“只观测预测”模型中无法以二次因子之外的精度估计，证明了该度量的计算难度。

**🔧 技术方法**

主要技术手段为：地球移动距离与 Kantorovich‑Rubinstein 对偶性；凸优化与双重线性规划的分析；损失函数的 V‑形基底展开；以及对随机噪声平滑操作的拉普拉斯/高斯噪声分析。

**📊 数据集**

本文主要为理论研究，没有使用真实数据集；实验和验证均基于构造的离散分布和假设场景。

**📈 对比分析**

与现有工作相比，本文提供了更宽泛的后处理基类（所有可微分后处理），并将误差上界从依赖于噪声方差σ的 1/σ 线性下降改为 O(σ + ε/σ)，在平滑校准误差 ε 较小的情形下取得更紧的误差上界；理论上证明了下限与上限校准距离之间的常数因子等价。

**⚠️ 局限性**

局限性：① 仍需要在预测器上加入随机噪声，实际实现中可能导致偏差；② 证明依赖于平滑校准误差可被有效估计的假设，实际估计仍有挑战；③ 上限校准距离在只观测预测的设置下不可逼近，说明在仅使用预测值的场景下仍存在不可估计的瓶颈。

---

## 481. Data-driven generalized perimeter control: Zürich case study

**arXiv ID:** 2603.16599 | [PDF](https://arxiv.org/pdf/2603.16599v1)

**作者:** Alessio Rimoldi `[一作]` (ETH Zurich), John Lygeros `[通讯]` (ETH Zurich)

**通讯引用:** 22486 | [OpenAlex ID](https://openalex.org/A5007359599)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于行为系统理论的自适应交通信号灯控制框架，并将其应用于Zürich市的高保真微观仿真，以实现全局交通流的最优调度。

**💡 创新点**

创新点在于：①使用行为系统理论构建无模型数据驱动的控制框架，消除传统宏观模型的建模开销；②将控制输入简化为交叉口激活比例λ，仅通过一维信号即可调控复杂网络；③通过对MFD的线性近似与DeePC结合，克服了传统MPC对非线性关系的误判。

**🔧 技术方法**

技术手段包括：数据驱动预测控制（DeePC）、行为系统理论、MFD拟合、SUMO微观仿真、PCA与负载分析、以及对比的线性MPC。

**📊 数据集**

使用的数据集为：①Zürich城市真实交通网络的数字孪生（约15000条道路、7000个交叉口）；②历史路况循环探测器数据用于需求估计；③从SUMO仿真生成的微观轨迹数据（约170,000辆车）。

**📈 对比分析**

通过与传统MPC和无控制基准在两套仿真网络（格子网络和Zürich）中比较，DeePC在总行程时间、等待时间、CO₂排放等指标上分别提升约9–18%，并实现了约20%的完成车辆数提升，表现优于MPC。

**⚠️ 局限性**

局限性包括：①对非线性系统的Hankel矩阵需要在足够多且覆盖度高的数据上构建，无法保证在不同需求条件下的鲁棒性；②优化维度随控制器数量增长，导致可扩展性受限；③对控制周期频率的选择需要经验判断，过长周期会导致性能退化。

---

## 482. Segmentation-Based Attention Entropy: Detecting and Mitigating Object Hallucinations in Large Vision-Language Models

**arXiv ID:** 2603.16558 | [PDF](https://arxiv.org/pdf/2603.16558v1)

**作者:** Jiale Song `[一作]` (Donghua University), Mingbo Zhao `[通讯]` (Donghua University)

**通讯引用:** 3419 | [OpenAlex ID](https://openalex.org/A5061195038)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大规模视觉‑语言模型（LVLM）中出现的对象幻觉问题，本文提出通过语义分割聚合视觉注意力，计算注意力熵来检测并在推理时调节注意力，从而降低幻觉率。

**💡 创新点**

创新点在于：①将视觉注意力按语义类别聚合并用归一化熵（Segmentation‑based Attention Entropy, SAE）量化注意力的不确定性；②基于SAE与交叉模态注意力质量融合得到可靠性得分用于幻觉检测；③设计了推理时注意力调节策略，以SAE为门控动态增强注意力一致性，既无额外训练成本，又显著抑制幻觉。

**🔧 技术方法**

使用技术包括：Transformer注意力机制、Mask2Former（Swin‑L）语义分割、熵计算与归一化、SAE‑guided 预激活 logits 调节、YOLO‑World 对象定位、基于LiDAR/Depth 的成本地图生成与路径规划。

**📊 数据集**

实验数据集：COCO 2014 验证集（1,000 张图像）用于幻觉检测与抑制；室内实验环境下的四足机器人 RGB‑D + LiDAR 传感器数据用于真实世界导航验证。

**📈 对比分析**

方法与多种基线（预测熵、最大 Softmax 概率、Margin、Energy、内部置信度、VAR、OPERA、VCD、PAI、Beam/Nucleus/Greedy）在 AUROC/AP、CHAIR（C_S/C_I）以及精确率/召回率/F1 方面均优于或与最佳基线持平；特别是在检测上相较 VAR 提升 7–15% AUROC，在抑制上将 C_S/C_I 降低约 50% 以上，保持高 F1。

**⚠️ 局限性**

局限性：①需要高质量的语义分割模型，分割误差会影响 SAE 计算；②主要针对对象幻觉，对文字或语义层面的幻觉覆盖不完整；③在不同 LVLM 架构或分辨率下的迁移性能尚未充分验证。

---

## 483. Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models

**arXiv ID:** 2603.16065 | [PDF](https://arxiv.org/pdf/2603.16065v1)

**作者:** Yanru Wu `[一作]` (USC Physical Superintelligence Lab), Yue Wang `[通讯]` (USC Physical Superintelligence Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种将大型视觉语言模型（VLM）专门化为即时帧级奖励生成器（Large Reward Models，LRM），并利用该奖励在无手工设计奖励的情况下对基于示范学习的机器人控制策略进行在线强化学习，从而显著提升高精度长时序操作的成功率。

**💡 创新点**

创新点：
1) 将 VLM 转化为三种帧级奖励（时间对比奖励、绝对进度奖励、任务完成奖励），实现实时、密集、语义化的奖励信号；
2) 通过大规模多源无标注视频（共 24 源）提取时间进度监督，构建高质量的奖励训练数据；
3) 在 RL 训练中采用 Interval‑Hold 机制，结合 PPO+GAE，充分利用 LRM 生成的密集奖励实现样本高效的策略优化；
4) 在 ManiSkill3 及真实机器人实验中展示零样本、无环境信息的 LRM 在 30 次 RL 迭代内即可将成功率从 57% 提升至约 61%，并在真实任务中将成功率从 38% 提升至 52%。

**🔧 技术方法**

技术：
- 基于 Qwen3‑VL‑8B‑Instruct 的 LoRA 微调；
- 使用 Direct Preference Optimization（DPO）训练对比奖励模型；
- 用 Supervised Fine‑Tuning（SFT）训练进度和完成奖励模型；
- PPO 与 GAE 结合的强化学习框架；
- Interval‑Hold 机制实现奖励稀疏查询与连续反馈；
- Chain‑of‑Thought 推理嵌入模型输出以提升奖励可靠性。

**📊 数据集**

数据集：
- 真实机器人轨迹：Open X‑Embodiment；
- 人机交互数据：HOI4D、EgoDex；
- 模拟环境：LIBERO、RoboCasa；
- 其他 19 种来源（未详细列举），总计 24 源，构成多域无标注视频集合，用于监督生成三类奖励。

**📈 对比分析**

比较方法与性能：
- 与零样本 Qwen3‑VL 基线、RoboReward‑8B、Robometer‑4B 等基准模型在 ManiSkill3 上对比；
- 成功率：LRM 各奖励 60–61%，优于 RoboReward 59% 与 Robometer 56%；
- 与环境奖励（Privileged）相比差距缩小至约 6%；
- 在真实机器人实验中，RL 微调后成功率从 38.3% 提升至 51.7%，显著优于 SFT 基线。

**⚠️ 局限性**

局限性：
- LRM 的奖励仍依赖 VLM 的预训练知识，可能在极端新颖环境或物体上表现不佳；
- 三种奖励的融合与权重选择需经验调参，缺乏统一自适应机制；
- Interval‑Hold 机制在极高频控制场景下可能导致奖励滞后；
- 训练数据仍需大量无标注视频，且对时间进度假设严格，可能不适用于非单调任务；
- 在真实机器人中，奖励误差可能导致误导性筛选，影响后续策略改进。

---

## 484. The Agentic Researcher: A Practical Guide to AI-Assisted Research in Mathematics and Machine Learning

**arXiv ID:** 2603.15914 | [PDF](https://arxiv.org/pdf/2603.15914v1)

**作者:** Max Zimmer `[一作]` (Zuse Institute Berlin), Sebastian Pokutta `[通讯]` (Technische Universität Berlin)

**通讯引用:** 1999 | [OpenAlex ID](https://openalex.org/A5043574831)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个面向数学与机器学习研究的 AI 辅助框架，包含从无 AI 到完全自主的五层整合分类、开放源码的 sandboxed agentic 研究框架以及一系列案例研究。

**💡 创新点**

创新点在于：①将 AI 整合水平系统化为五层模型；②构建基于 CLI coding agent 的可扩展、可验证、可追溯的研究工作流；③提出十条“命令”来规范 agent 行为，保障科学严谨。

**🔧 技术方法**

所采用技术包括前沿 LLM 的 CLI coding agents（Claude Code、Codex CLI、OpenCode 等）、容器化沙箱、Git 版本控制、自动化报告与实验追踪、命令式提示工程等。

**📊 数据集**

数据集方面，论文未统一列出，但案例涵盖了 LLM pretraining 使用的 FineWeb 数据集、GPU 训练集以及数学证明实验所需的公开代码库。

**📈 对比分析**

实验方式为 agent 在多节点 GPU 上自动化执行实验循环，案例显示在 20+ 小时内完成多项实验并生成结构化报告，显著加速实验与证明过程。

**⚠️ 局限性**

局限性包括：需人工审阅验证结果；agent 在计划不完整时可能长时间探索无效方向；无法完全保证发现的创新性与引用完整性；仍需要研究者持续监督与调优。

---

## 485. IOSVLM: A 3D Vision-Language Model for Unified Dental Diagnosis from Intraoral Scans

**arXiv ID:** 2603.16781 | [PDF](https://arxiv.org/pdf/2603.16781v1)

**作者:** Huimin Xiong `[一作]` (Zhejiang University), Zuozhu Liu `[通讯]` (Zhejiang University)

**通讯引用:** 1139 | [OpenAlex ID](https://openalex.org/A5024343415)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了IOSVLM——一个端到端的3D视觉语言模型，用于统一诊断多种牙科疾病并生成自然语言问答，配合新构建的大规模IOSVQA数据集。

**💡 创新点**

创新点包括：①直接使用原始3D口内扫描点云而非视图渲染；②提出Geometry-to-Chromatic Proxy（GCP）利用法向量伪色弥补无彩色点云与彩色预训练的分布差距；③设计两阶段课程式训练策略以提升几何感知和语言推理；④构建包含单弓、遮挡弓扫描的23种疾病、19,002例、249,055问答对的大规模IOS诊断VQA基准。

**🔧 技术方法**

技术方法：ReCon++ 3D点云编码器提取绝对位置、局部几何和全局特征；多分支投影器将特征映射到LLM token空间并加可学习视觉提示；LLM（Qwen3VL-8B-Instruct）完成诊断与推理；GCP将法向量映射为伪彩色提供分离信号；两阶段训练（Stage‑1 训练几何与对齐，Stage‑2 微调投影器+LLM，使用LoRA）与链式思考（CoT）推理监督。

**📊 数据集**

使用的数据集：IOSVQA（19,002 IOS扫描，249,055问答对，覆盖23种口腔疾病，包含单弓和遮挡弓两种扫描类型），来源于MaloccIOS、DiseaseIOS、Bits2Bites，已进行全局配准、标签统一与高质量样本筛选。

**📈 对比分析**

与四类基线（专有多模LLM、开源2D多模LLM、医学2D多模LLM、开源3D多模LLM）对比，IOSVLM在宏观准确率 77.23%、宏观 F1 50.39%、召回率 52.96% 等指标上均显著领先（Acc+9.58%、F1+1.46%），且实现 100% 解析率，显示出直接3D几何建模的优势。

**⚠️ 局限性**

局限性：受限于高质量标注与解释的稀缺，类不平衡和多病共存仍是挑战；模型性能依赖于LLM规模，极少标注情况下易退化；尚需进一步验证临床安全性、可解释性与跨设备泛化能力。

---

## 486. MDM-Prime-v2: Binary Encoding and Index Shuffling Enable Compute-optimal Scaling of Diffusion Language Models

**arXiv ID:** 2603.16077 | [PDF](https://arxiv.org/pdf/2603.16077v1)

**作者:** Chen-Hao Chao `[一作]` (University of Toronto), Rahul G. Krishnan `[通讯]` (University of Toronto)

**通讯引用:** 2427 | [OpenAlex ID](https://openalex.org/A5073514348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MDM-Prime-v2，改进离散扩散语言模型的子词化和索引打乱，使其在计算最优条件下更高效；

**💡 创新点**

通过理论证明最大粒度二进制编码与索引打乱可收紧变分上界，从而提升模型的概率估计；

**🔧 技术方法**

利用二进制子词化、索引打乱、变分推理、RoPE、SwiGLU等技术；

**📊 数据集**

在OpenWebText、C4、Slimpajama等大规模文本语料上训练并评估；

**📈 对比分析**

与ARM、MDM、MDM-Prime等基线进行对比，MDM-Prime-v2在计算最优配置下PPL降低至7.77，比ARM低约4.6；在1.1B参数规模下零样本常识推理平均准确率最高；

**⚠️ 局限性**

仍未解决的局限包括对不同BPE策略的通用性研究不足，以及在极大模型规模下的训练成本与资源瓶颈。

---

## 487. Quantum Key Distribution Secured Federated Learning for Channel Estimation and Radar Spectrum Sensing in 6G Networks

**arXiv ID:** 2603.15649 | [PDF](https://arxiv.org/pdf/2603.15649v1)

**作者:** Ferhat Ozgur Catak `[一作]` (University of Stavanger), Umit Cali `[通讯]` (University of York)

**通讯引用:** 2951 | [OpenAlex ID](https://openalex.org/A5082630876)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了基于QKD的安全联邦学习框架，用于6G网络中的无线信道估计与雷达频谱感知任务。

**💡 创新点**

将BB84量子密钥分发与对称加密的逐对掩码相结合，形成协议级QKD抽象的安全聚合方案，实现了在不泄露模型梯度的前提下进行模型聚合，并通过QBER阈值实现攻击检测与自动中止。

**🔧 技术方法**

采用BB84协议抽象、对称掩码（pairwise additive masking）、CNN（用于信道估计）和U-Net（用于雷达语义分割），以及联邦学习（FedAvg）与随机梯度下降优化。

**📊 数据集**

使用MATLAB 5G Toolbox生成的612×14 OFDM信道数据集（1000训练/500验证样本）以及合成的256×256×3雷达/通信频谱图数据集（200样本，80/20分割）。

**📈 对比分析**

与普通未加密的联邦学习（plain）和基于伪随机生成密钥的安全聚合（Classical-SA）进行对比，在3、10、20个客户端上评估NMSE、像素准确率和mIoU。结果显示QKD-SA在3客户端时NMSE≈0.051，雷达精度≈89%，与baseline基本持平；在更多客户端时略有提升，且通信量相同。

**⚠️ 局限性**

实验仅采用协议级QKD抽象，未考虑真实物理层量子通道噪声与MDI-QKD；雷达数据集规模有限，缺乏正式差分隐私分析，且掩码方案对拜占庭更新的鲁棒性未验证。

---

## 488. Long-Horizon Traffic Forecasting via Incident-Aware Conformal Spatio-Temporal Transformers

**arXiv ID:** 2603.16857 | [PDF](https://arxiv.org/pdf/2603.16857v1)

**作者:** Mayur Patil `[一作]` (Ohio State University), Nithin Santhanam `[通讯]` (Honda Research Institute USA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了针对长时 horizon 的交通流预测框架，利用事故感知的自适应时空图卷积 Transformer 进行多步预测，并通过自适应共形预测提供置信区间。

**💡 创新点**

核心创新在于：①用基于小时 CV 的分段 log‑normal 采样构造时变邻接矩阵；②通过碰撞清除时间、天气、工地、速度违规等多维属性对边权进行事故强度修正；③将上述自适应图嵌入 Transformer 的空间编码层，配合自适应共形校准实现可靠的多时 horizon 置信区间。

**🔧 技术方法**

技术要点包括：Spatio‑Temporal Transformer (encoder‑decoder)；自适应共形预测（ACP）；基于 CV 的 log‑normal 采样；事故严重度信号的加权；多头自注意力与跨注意力解码；图卷积混合与行归一化。

**📊 数据集**

使用俄亥俄州交通部（ODOT）交通计数数据及其对应的事故记录作为训练/验证数据；通过 SUMO 微观仿真与 INRIX 历史行程数据对预测结果进行实证检验。

**📈 对比分析**

与历史平均、ARIMA、FNN、GCN‑GRU、GAT‑LSTM、STGCN、DCRNN、Graph WaveNet、ASTGCN 等基线对比。STT‑ED 在 1–4 小时预测中均取得最低 MAE/RMSE，误差随 horizon 增长幅度最小；ACP 在 PICP 与 MPIW 上优于 CP、CQR 等共形与不确定性估计方法，覆盖率高而区间宽度更紧凑。

**⚠️ 局限性**

局限性包括：小时 CV 预估为静态，无法即时响应突发需求波动；事故信息仅以聚合强度形式加入，未捕捉事故演化与排队细节；实验仅在俄亥俄州单一网络与固定时 horizon 范围内验证；ACP 校准仅在训练周期后进行，实时漂移时响应有限。

---

## 489. NeSy-Route: A Neuro-Symbolic Benchmark for Constrained Route Planning in Remote Sensing

**arXiv ID:** 2603.16307 | [PDF](https://arxiv.org/pdf/2603.16307v1)

**作者:** Ming Yang `[一作]` (Nanjing University), Yu-Feng Li `[通讯]` (Nanjing University)

**通讯引用:** 41571 | [OpenAlex ID](https://openalex.org/A5082124101)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了NeSy-Route——一套大规模神经符号化路由规划基准，用于评估遥感场景下受约束的路径规划能力。

**💡 创新点**

创新点包括：①规模大，10,821个样本是现有基准的十倍；②分层神经符号评估，清晰拆分感知、推理、规划三阶段；③自动化数据生成框架，将高精度语义分割与启发式搜索结合，产生可验证的最优轨迹。

**🔧 技术方法**

采用OpenEarthMap语义分割、形态学滤波、双LLM逻辑验证、A*启发式搜索、符号向量提取、Kendall Tau相关、Chamfer距离等技术。

**📊 数据集**

基于OpenEarthMap遥感影像和自动合成的文本任务，生成符号查询、视觉约束与最优轨迹。

**📈 对比分析**

在三任务层级上使用TM、PR、FM、RM、AR、VR、CR、CD等指标对闭源（GPT‑5.1、Gemini‑3‑Pro、Qwen3‑VL‑Plus）和开源（LLaVA、Qwen系列等）LLM进行零样本评估。闭源模型在约束理解上表现最好，但在规划层面仍远低于最优；开源模型的遵循率与成本比普遍偏高，显示规划能力不足。

**⚠️ 局限性**

当前MLLM在遥感路径规划中难以有效整合土地类型约束，感知与推理虽具备一定能力，但缺乏全局规划策略；受限于训练数据缺乏土地纹理与地形特征，以及依赖旧架构导致的认知差距。

---

## 490. Hypothesis Class Determines Explanation: Why Accurate Models Disagree on Feature Attribution

**arXiv ID:** 2603.15821 | [PDF](https://arxiv.org/pdf/2603.15821v1)

**作者:** Thackshanaramana B `[一作]` `[通讯]` (SRM Institute of Science and Technology), Thackshanaramana B (SRM Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在不同假设类中预测相等的模型是否能给出相同的特征重要性解释，并发现存在显著差异。

**💡 创新点**

提出了“解释彩票”概念，证明了假设类差异是解释不一致的根本结构原因，并提出了后置诊断指标 Explanation Reliability Score R(x) 来预测实例级解释稳定性。

**🔧 技术方法**

使用 SHAP（TreeSHAP、KernelSHAP）与 LIME 计算特征重要性，进行 Spearman 相关性比较，基于同一数据拆分和不同随机种子进行大规模实验；同时给出理论证明。

**📊 数据集**

在 24 个来自 OpenML、ProPublica 等公开数据集的二分类任务上进行实验，数据维度从 4 到 856，样本量 208–45,211。

**📈 对比分析**

对 93,510 组预测相等模型进行两两比较，发现跨类平均 Spearman 相关约 0.42，内类约 0.68，解释彩票率约 35%（阈值 0.5）或 62%（同拆分实验）。实验表明假设类决定解释一致性，训练方差无影响。

**⚠️ 局限性**

局限包括仅针对表格二分类任务、仅使用 SHAP/LIME 解释方法、未验证大型语言模型或多分类设置；R(x) 仅为经验启发式，需在具体部署前构建多模型集合。

---

## 491. XLinear: Frequency-Enhanced MLP with CrossFilter for Robust Long-Range Forecasting

**arXiv ID:** 2603.15645 | [PDF](https://arxiv.org/pdf/2603.15645v1)

**作者:** Xiang Ao `[一作]` (Beijing Jiaotong University), Xiang Ao `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 3307 | [OpenAlex ID](https://openalex.org/A5068007462)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于 MLP 的长序列预测模型 XLinear，融合趋势季节分解、频域增强注意力 EFA 和交叉滤波器 CrossFilter，以提升长程依赖捕捉和噪声鲁棒性。

**💡 创新点**

创新点在于：①引入频域增强注意力 EFA，仅强化低频趋势信号；②设计 CrossFilter，将 PaiFilter 与 TexFilter 并行融合并通过 GELU 与逐元素相乘抑制噪声，从而兼顾简单与复杂数据集的鲁棒性。

**🔧 技术方法**

技术手段包括：MSE 损失 + Adam 优化；RevIN 实例归一化；移动平均趋势分解；FFT 频域操作 + 指数激活的 EFA；交叉滤波器的双分支频域乘法与 GELU；最后通过轻量级 MLP 输出预测。

**📊 数据集**

实验覆盖 8 个公开数据集：ETTm1、ETTm2、ETTh1、ETTh2、Weather、ECL、Traffic、Exchange，分别涵盖能源、金融、交通与气象领域，数据规模与维度多样。

**📈 对比分析**

与多种 SOTA 基线（iTransformer、PatchTST、TimesNet、DLinear、FilterNet 等）在 96/192/336/720 步长上进行对比；XLinear 在大多数数据集和长预测 horizon 上获得最低 MAE/MSE，且参数量与推理速度均优于 Transformer 族模型。

**⚠️ 局限性**

局限性在于：对极其复杂的 Traffic 数据仍落后于部分 Transformer 模型；EFA 尽管提升了精度，但相对原始 MLP 增加了参数；在某些短期预测任务中提升幅度不显著。

---

## 492. An Interpretable Machine Learning Framework for Non-Small Cell Lung Cancer Drug Response Analysis

**arXiv ID:** 2603.16330 | [PDF](https://arxiv.org/pdf/2603.16330v1)

**作者:** Ann Rachel `[一作]` (Birla Institute of Technology and Science Pilani), Tojo Mathew `[通讯]` (Birla Institute of Technology and Science Pilani)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个可解释的机器学习框架，用 XGBoost 回归模型预测肺腺癌（LUAD）和肺鳞癌（LUSC）细胞系对药物的 LN-IC50 敏感度，并通过 SHAP 及 DeepSeek API 生成临床可读的解释。

**💡 创新点**

创新点在于：① 将药物响应从分类问题转为回归问题，捕捉更细粒度的敏感度；② 结合 SHAP 与大型语言模型 DeepSeek，将特征重要性转化为医学可操作的解释；③ 在 GDSC 数据集上实现极高的预测准确率（R²≈0.997）。

**🔧 技术方法**

核心技术包括：XGBoost 回归、RandomizedSearchCV 超参搜索、SHAP 解释、DeepSeek API 生成文本总结、Streamlit 部署交互式应用。

**📊 数据集**

使用 Genomics of Drug Sensitivity in Cancer（GDSC）数据集，仅保留 LUAD 与 LUSC 两种亚型的细胞系记录，目标变量为 log‑transformed IC50（LN-IC50）。

**📈 对比分析**

与随机森林、线性回归及文献中已有模型对比，XGBoost 取得最高的 R²（0.9971）与最低的 MAE（0.0851）和 MSE（0.0249）。5 折交叉验证显示平均 R²≈0.9965，MAE≈0.079，证明模型在不同子集上的稳定性和泛化能力。

**⚠️ 局限性**

局限性包括：① 仅针对 LUAD 与 LUSC 两种亚型，难以推广到其他肺癌或药物；② 依赖公开 GDSC 数据，样本量相对有限；③ SHAP 与 DeepSeek 的解释受限于模型训练数据的偏差与语言模型的推理误差；④ 未在外部临床试验数据上进行外部验证，实际临床适用性仍需进一步评估。

---

## 493. QV May Be Enough: Toward the Essence of Attention in LLMs

**arXiv ID:** 2603.15665 | [PDF](https://arxiv.org/pdf/2603.15665v1)

**作者:** Zhang Edward `[一作]` `[通讯]`, Zhang Edward

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从词性与句法角度重新解释Transformer的QKV注意力机制，提出QV与QV‑Ka模型，并通过实验验证其与传统QKV模型的性能对比。

**💡 创新点**

创新性地将QKV拆解为浅层组合（Shallow‑Composing）与深层匹配（Deep‑Matching）两类功能，提出QV范式和QV‑Ka关键后值优化，并将MQA、GQA、MLA等架构归纳为QV的变体。

**🔧 技术方法**

使用Transformer‑BIG基础模型、AGF相对位置编码、PCM‑V优化、FP16半精度训练，结合OpenNMT‑py与AGF模块代码实现实验。

**📊 数据集**

采用WMT_17英德翻译数据集进行验证。

**📈 对比分析**

通过在3层Transformer上比较QKV、QV、QV‑Ka在验证集上的准确率，发现QV原始模式比QKV低0.5%，AGF+PCM‑V显著缩小差距至0.26%，QV‑Ka（2×d_head）达到70.69%，与QKV（70.78%）相近，同时参数与计算量更少。

**⚠️ 局限性**

实验规模有限（仅3层、1024维模型、半精度训练），未在大规模LLM上验证，需进一步测试其可扩展性与性能。

---

## 494. SpokenUS: A Spoken User Simulator for Task-Oriented Dialogue

**arXiv ID:** 2603.16783 | [PDF](https://arxiv.org/pdf/2603.16783v1)

**作者:** Jonggeun Lee `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**通讯引用:** 1693 | [OpenAlex ID](https://openalex.org/A5021733732)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了大型的 SpokenTOD 语音任务导向对话数据集，并在此基础上训练了支持主动对话交互的 SpokenUS 语音用户模拟器。

**💡 创新点**

创新点在于：1) 通过自动化流水线在文本 TOD 数据上增添跨轮槽位、插话、失语以及情绪韵律等真实口语行为；2) 设计三模式（听取、预写、发声）结构并加入专门的插话检测头，实现实时主动插话；3) 仅用约 1 千小时语音数据即能匹敌 20 万小时模型。

**🔧 技术方法**

技术包括：大规模 LLM（Qwen3-32B、Qwen2.5-3B）进行数据增添与对齐；Qwen3-TTS 与 CosyVoice3 进行情绪驱动语音合成；Conditional Flow Matching + HiFi‑GAN 进行高质量语音生成；多任务损失联合训练文本、语音与插话决策。

**📊 数据集**

使用了 MultiWOZ、SGD、TaskMaster、ABCD 等文本 TOD 数据集，并通过 Qwen3-TTS 合成 1,034 小时音频；同时加入真实录制的 SpokenWOZ 作为验证与评测。

**📈 对比分析**

在 100 条测试对话上，SpokenUS 在目标覆盖率（GA 0.82）与人类 MOS（4.06）上优于所有基线，尤其在 MOS 上超越 20M 时钟模型（3.18）；其 WER 仅为 11.36%，并且能保持与人类相近的说话者相似度；同时对代理的挑战更大，Slot F1 在 ASR 级别下降显著。

**⚠️ 局限性**

局限性包括：仅覆盖英语与现有文本 TOD 数据集的领域；未涵盖笑声、重叠语音、代码切换等行为；合成语音缺乏真实环境噪声与回声；插话检测仍存在一定误判，尤其是早期触发与误判；需要进一步扩展到多语言与更复杂的口语情境。

---

## 495. Tokenization Tradeoffs in Structured EHR Foundation Models

**arXiv ID:** 2603.15644 | [PDF](https://arxiv.org/pdf/2603.15644v1)

**作者:** Lin Lawrence Guo `[一作]` (Hospital for Sick Children), Lillian Sung `[通讯]` (Hospital for Sick Children)

**通讯引用:** 27617 | [OpenAlex ID](https://openalex.org/A5086900250)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文设计并系统评估了三种 EHR 令牌化策略（事件编码、时间编码、工作流注释）对基于 Transformer 的基础模型在 74 个临床预测任务中的表现与预训练计算成本的影响。

**💡 创新点**

首次量化了“局部绑定效率”对 joint 事件编码优势的贡献，并证明该优势跨机构迁移，提供了可操作的令牌化默认配置。

**🔧 技术方法**

采用 28 层 decoder‑only Transformer 进行自监督下一个 token 预测，并用线性混合模型评估 tokenization 轴对 AUROC 的独立效应。

**📊 数据集**

在加拿大儿童医院 SickKids 的 169M 事件数据上预训练，并在同医院的 74 个任务与 MIMIC‑IV ICU 数据集的 13 个任务上进行内部与外部评估。

**📈 对比分析**

通过 factorial 设计比较八种 tokenization 方案，发现 joint‑encoding 与 time‑positions 在大多数任务上提升约 0.008 AUROC，同时减少 39.5% 与 9.6% 的 FLOPs；在 MIMIC 上冻结模型仍保持相似优势，整体 AUROC 约 0.82。

**⚠️ 局限性**

仅在单一儿童医院预训练、仅评估线性探测器、未探索 BPE 等自学习 tokenization、外部评估混合了词汇、人口与机构差异，且对生成任务未测试。

---

## 496. Deep Tabular Representation Corrector

**arXiv ID:** 2603.16569 | [PDF](https://arxiv.org/pdf/2603.16569v1)

**作者:** Hangting Ye `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**通讯引用:** 252346 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Tabular Representation Corrector（TRC），一种模型无关的后置表示校正方法，能在不改动已训练深度表格模型参数的前提下提升其预测性能。

**💡 创新点**

通过两步任务：Tabular Representation Re-estimation（学习shift估计器消除表示漂移）和 Tabular Space Mapping（将校正后表示映射到低维嵌入空间，减少冗余并保持关键信息）实现对已训练模型的高效表示改进。

**🔧 技术方法**

采用轻量级的两层MLP作为shift估计器，线性softmax层作为坐标估计器，正交化损失和预测损失训练，同时利用梯度范数选取近似最优样本并对其进行噪声扰动。

**📊 数据集**

在10个真实世界的表格数据集上评测，包括回归（CO、DI、QS、CA等）和分类（AD、AU、GE、YE、COV）任务。

**📈 对比分析**

与13种主流深度表格模型（MLP、FT-Transformer、SCARF等）以及传统微调、LoRA等方案对比，TRC平均提升约5.1%（回归6.5%、分类2.4%），并在大规模数据集上保持显著优势。

**⚠️ 局限性**

对阈值τ和近似最优样本选择的假设敏感，且对极端噪声或极稀疏样本的适应性尚未充分验证。

---

## 497. Anticipatory Planning for Multimodal AI Agents

**arXiv ID:** 2603.16777 | [PDF](https://arxiv.org/pdf/2603.16777v1)

**作者:** Yongyuan Liang `[一作]` (University of Maryland), Ruiyi Zhang `[通讯]` (Adobe Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种两阶段强化学习框架，先用轨迹级别的奖励训练模型进行前瞻性轨迹规划，再用工具执行反馈细化步级动作执行；

**💡 创新点**

将长时间跨度的轨迹一致性与基于执行结果的步级精确度相结合，既能实现全局规划一致性，又能保证执行可行性；

**🔧 技术方法**

轨迹级强化学习（采用GRPO）、步级执行反馈强化、Qwen3-VL-8B/32B-Thinking多模态模型、轨迹对齐奖励与循环惩罚、时间折扣等；

**📊 数据集**

使用多模态轨迹数据集（AgentNet、AndroidControl、GUI‑Odyssey、Multimodal‑Mind2Web、AgentTrek），工具使用轨迹数据集（T3‑Agent工具箱）以及七个GUI与工具使用基准（OSWorld‑Verified、AndroidWorld、AndroidControl‑High、GUI‑Odyssey、Multimodal‑Mind2Web、GTA、GAIA）；

**📈 对比分析**

相较于多种开源与专有基线（GPT‑4o、Claude 4、OS‑Atlas、GUI‑R1 等），在在线 GUI 任务、离线 GUI 任务以及工具使用任务上均显著提升成功率和执行稳定性；在 OSWorld‑Verified 上从 27.4% 提升至 41.2%，在 AndroidWorld 上从 57.2% 提升至 64.8%，在 GAIA 与 GTA 也超过 GPT‑4o 与其他开源模型；

**⚠️ 局限性**

当前方法仅在短时更新上提供局部纠正，无法重塑对长周期可行性或任务结构的全局认知；缺乏多层次规划、内存或世界模型更新，限制了在更长时间尺度或体感/混合工具环境中的表现。

---

## 498. DASH: Dynamic Audio-Driven Semantic Chunking for Efficient Omnimodal Token Compression

**arXiv ID:** 2603.15685 | [PDF](https://arxiv.org/pdf/2603.15685v1)

**作者:** Bingzhou Li `[一作]` (Shanghai Jiao Tong University), Tao Huang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 19373 | [OpenAlex ID](https://openalex.org/A5025077602)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `fede83ac-7505-405f-ab37-e7284695c47f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无训练的结构感知多模态压缩框架DASH，利用音频嵌入做语义锚点，实现动态语义分块并进行跨模态压缩。

**💡 创新点**

创新点在于把音频的语义断点作为分块依据，动态生成可变长度段；通过三信号融合（边界概率、内容唯一性、注意力）进行token重要性评估；以及跨模态同步分块与自适应压缩比例。

**🔧 技术方法**

使用余弦相似度检测音频边界、基于多尺度高斯核的唯一性评估、注意力权重融合、以及基于时间比例的跨模态投影和自适应阈值分块。

**📊 数据集**

在AVUT、VideoMME和WorldSense三个音视频理解基准上进行实验。

**📈 对比分析**

相较于随机裁剪、FastV、DyCoke、OmniZip等方法，DASH在相同或更低的token保留率（如25%）下保持甚至超过对比方法的平均准确率，并在推理速度和内存占用上获得3.5–3.8倍加速和约30% FLOPs节省。

**⚠️ 局限性**

局限在于对音频语义划分的依赖，短音频或无明显停顿的场景下分块效果可能受限；同时跨模态对齐仍基于线性时间比例，可能忽略细粒度的音画同步差异。

---

## 499. Detecting Sentiment Steering Attacks on RAG-enabled Large Language Models

**arXiv ID:** 2603.16342 | [PDF](https://arxiv.org/pdf/2603.16342v1)

**作者:** Isha Andrade `[一作]` (Birla Institute of Technology and Science), Raja Muthalagu `[通讯]` (Birla Institute of Technology and Science)

**通讯引用:** 935 | [OpenAlex ID](https://openalex.org/A5070580861)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了两种轻量化的深度学习 IDS 模型（CNN‑IDS 与 LSTM‑IDS），用于检测并分类 IoT 网络攻击，覆盖二分类、分组分类与多分类三种场景。

**💡 创新点**

创新点在于：① 将轻量化网络结构应用于 IoT 攻击检测；② 在同一数据集上统一实现三种分类任务；③ 在 CICIoT2023 数据集上实现高于现有 HetIoT CNN‑IDS 的准确率。

**🔧 技术方法**

使用 CNN 与 LSTM 两种深度学习模型，配合 Adam 优化器、交叉熵损失函数，加入特征选择（随机森林重要性）与数据平衡（Stratify）等预处理技术。

**📊 数据集**

采用 CICIoT2023（47 维）数据集，选取 20 个最重要特征，使用 10% 的样本（约 4.7M 行）进行训练与评估。

**📈 对比分析**

实验通过与 HetIoT CNN‑IDS 对比，分别在二分类、分组分类和多分类上取得 99.34%/99.42%，99.02%/99.13%，98.62%/98.68% 的准确率，显示出比对手更优的性能。

**⚠️ 局限性**

限制包括：实验仅在单机 CPU + 集成显卡环境下完成；未对模型的可解释性和对抗攻击鲁棒性进行评估；并未在更大规模流量或多平台环境下验证可扩展性。

---

## 500. Industrial cuVSLAM Benchmark & Integration

**arXiv ID:** 2603.16240 | [PDF](https://arxiv.org/pdf/2603.16240v1)

**作者:** Charbel Abi Hana `[一作]`, Anthony Rizk `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对仓储物流环境下的视觉里程计与 SLAM 系统进行了系统性基准评测。

**💡 创新点**

创新点在于将 NVIDIA cuVSLAM 的 GPU 加速前端与自研后端相结合，形成一套高频、低延迟且资源占用低的完整 SLAM 方案。

**🔧 技术方法**

采用了 cuVSLAM（CUDA 加速）、ORB‑SLAM3、RTAB‑Map 以及混合方案（cuVSLAM 前端 + Idealworks SLAM 后端）的技术栈。

**📊 数据集**

使用了 LIPSEdge AE400 立体相机收集的三条 VO 数据集（L‑shape、Hybrid、Rotation‑Heavy）和两条 1.7 km 与 570 m 的 SLAM 路径数据集，真值来自运动捕捉与 LiDAR 定位。

**📈 对比分析**

通过绝对位姿误差（APE）以及 CPU/GPU 负载、tick‑rate 等指标进行比较，cuVSLAM 在 VO 中平均 APE 最低、推理最快；混合方案在 SLAM 中实现了 0.91 m 的平均误差、1.50 Hz 的 tick‑rate，优于 ORB‑SLAM3 与 RTAB‑Map。

**⚠️ 局限性**

局限性包括使用滚动快门 15 fps 相机导致部分旋转场景性能下降；cuVSLAM 缺乏后处理地图管理工具；以及实验结果仅在特定硬件（Jetson Xavier）与有限数据集上验证，需进一步验证在更高帧率与全局快门相机上的表现。

---

## 501. GLANCE: Gaze-Led Attention Network for Compressed Edge-inference

**arXiv ID:** 2603.15717 | [PDF](https://arxiv.org/pdf/2603.15717v1)

**作者:** Neeraj Solanki `[一作]` (University of Illinois Chicago), Arman Roohi `[通讯]` (University of Illinois Chicago)

**通讯引用:** 1347 | [OpenAlex ID](https://openalex.org/A5077392159)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于权重无神经网络（DWN）的低功耗视线估计与聚焦区域检测的两阶段实时对象检测框架。

**💡 创新点**

将DWN用于眼球姿态估计实现极低MAC，利用聚焦ROI合并生成单一裁剪并与时间累积相结合，显著降低计算、通信延迟与能耗。

**🔧 技术方法**

Differentiable Weightless Neural Networks（DWN）、ROI融合与合并、周期性检测调度、IMU驱动的ROI稳定、Arduino Nano 33 BLE边缘节点与YOLOv12n检测器。

**📊 数据集**

MPIIGaze（用于视线估计）和COCO（用于目标检测）。

**📈 对比分析**

与全局YOLOv12n基线对比，mAP分别提升至51.3%/72.1%/88.1%（小/中/大），仅处理40–50%图像面积，通信延迟降低177×，能耗下降65%，端到端延迟保持<10 ms。

**⚠️ 局限性**

视线估计误差仍为8.3°，需额外眼部摄像头，ROI累积在高速运动或大物体场景下的稳定性有限，受MCU内存与算力限制。

---

## 502. Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation

**arXiv ID:** 2603.16284 | [PDF](https://arxiv.org/pdf/2603.16284v1)

**作者:** TianTian Dang `[一作]`, Shuhui Wang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为LTS‑FS的插件式框架，通过在层级上定位幻觉相关层并按归因得分稀疏地调节特征，以在保持推理速度的前提下减少大视觉‑语言模型（LVLM）的幻觉。

**💡 创新点**

创新点在于：① 使用因果干预对每一层的关注头进行掩码，量化其对幻觉输出的贡献；② 依据归因得分构建层级稀疏门并按比例调节特征调节强度；③ 该层级稀疏调节与任何现有特征调节方法无缝对接，具有通用性和可插拔性。

**🔧 技术方法**

核心技术包括：因果干预归因、token/句子级幻觉归因权重、层级稀疏门（阈值 r_s）与软加权调节、以及对现有特征调节方法（如Nullu、VTI）的包装。

**📊 数据集**

构造了双粒度幻觉数据集：token级样本来自POPE、Antidote，句子级样本基于CHAIR；随后在这些样本上进行归因与层级定位，整个实验使用CHAIR、POPE、MME和LLaVA‑Bench等公开基准。

**📈 对比分析**

与传统特征调节（Nullu、VTI）及其它幻觉抑制方法（VCD、AGLA）对比，LTS‑FS 在CHAIR 的 CS 与 CI 显著下降（例如 53.0→46.8），在 POPE 上准确率提升至 86.59%，F1 分数提升约 2%，同时在 MME 与 LLaVA‑Bench 上保持或提升通用性能；推理时间基本与原模型相同。

**⚠️ 局限性**

局限性包括：① 依赖手工标注的幻觉样本，构造成本高；② 层级稀疏门阈值 r_s 需要经验调优，可能不适用于所有模型；③ 仅通过特征调节无法彻底解决生成逻辑层面的深层幻觉，尤其是跨模态推理错误；④ 对极长文本和不同任务的泛化性能仍待进一步验证。

---

## 503. GASP: Guided Asymmetric Self-Play For Coding LLMs

**arXiv ID:** 2603.15957 | [PDF](https://arxiv.org/pdf/2603.15957v1)

**作者:** Swadesh Jana `[一作]` (University of Tübingen), Pavel Kolev `[通讯]` (University of Tübingen)

**通讯引用:** 1814 | [OpenAlex ID](https://openalex.org/A5001474340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Guided Asymmetric Self-Play（GASP），通过将难度较高的真实题目（goalpost）作为教师生成问题的引导，构建 lemma‑lift 两步递进的自我训练流程；

**💡 创新点**

创新点在于将目标硬题作为训练目标，将问题生成分为先生成易题（lemma）再生成难题（lift）的两阶段过程，并通过 learnability 奖励与拒绝采样保持生成多样性；

**🔧 技术方法**

使用教师‑学生双角色的强化学习框架，Task‑Relative REINFORCE++ 奖励、learnability 曲线、拒绝采样、以及与真实数据 RL 的联合训练；

**📊 数据集**

基于 LiveCodeBench (LCB) 数据集，训练集 2023.05–2024.08 共 601 题，其中 146 题筛选为 goalpost，评估集 2024.10–2025.02 共 216 题；

**📈 对比分析**

与基线模型、未引导自我训练 AZR、标准 RLVR 及其与真实数据混合版本比较，GASP 在 pass@20 上提升 2.5%（约 34.46% 对比 31.15%），并在部分硬题上首次实现可解，整体性能优于或竞争现有方法；

**⚠️ 局限性**

局限性包括：未对 lemma‑lift 与 goalpost 的对齐性进行严格评估；生成难度提升可能偏向表面化；多样性受限；仅使用单一训练集，未动态更新 goalpost；缺乏判别器奖励或更细粒度的质量评估。

---

## 504. SF-Mamba: Rethinking State Space Model for Vision

**arXiv ID:** 2603.16423 | [PDF](https://arxiv.org/pdf/2603.16423v1)

**作者:** Masakazu Yoshimura `[一作]` (Sony Group Corporation), Takeshi Ohashi `[通讯]` (Sony Group Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SF-Mamba，一种改进的视觉Mamba编码器；

**💡 创新点**

主要创新点是辅助patch交换实现未来到过去的信息流，以及批量折叠+周期性状态重置提升GPU并行效率；

**🔧 技术方法**

采用可变长序列的选择性状态空间模型、可插拔的注意力块、1D深度卷积、批量折叠与周期性状态重置技术；

**📊 数据集**

在ImageNet-1K进行分类实验，ADE20K进行语义分割，COCO进行目标检测与实例分割；

**📈 对比分析**

与CNN、Transformer、混合模型及其他视觉Mamba变体对比，SF‑Mamba在不同规模下均实现了更优的精度‑吞吐量折衷，显著提升了速度并保持或提升了准确率；

**⚠️ 局限性**

在单图像推理（batch=1）下，批量折叠带来的加速有限；此外，模型仍需依赖窗口注意力来补全全局信息，可能在极高分辨率输入下受限。

---

## 505. Generative AI for Quantum Circuits and Quantum Code: A Technical Review and Taxonomy

**arXiv ID:** 2603.16216 | [PDF](https://arxiv.org/pdf/2603.16216v1)

**作者:** Juhani Merilehto `[一作]` `[通讯]`, Juhani Merilehto

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

系统性综述了2024-2026年间的13种量子电路与代码生成模型及其5个支持数据集

**💡 创新点**

提出了基于artifact类型与训练方式的六大类标签，并构建了三层评估框架（语法、语义、硬件可执行性），揭示了当前模型缺乏硬件端到端评估

**🔧 技术方法**

涵盖了监督微调、RL/验证器驱动、扩散/图生成、代理式优化等技术路线，并对各类模型架构与奖励机制进行对比

**📊 数据集**

使用了如QASMBench、QCircuitBench、QuantumLLMInstruct、graph-data-quantum-rl等公开数据集与基准

**📈 对比分析**

对比方法按语法有效性、语义评测（单元测试、过程相干度、分布一致性）和硬件可执行性（后端编译与量子器件运行）进行；大多数模型仅在语法/语义层面表现良好，硬件层面无公开结果

**⚠️ 局限性**

局限包括单审稿人筛选、公开信息不充分导致部分系统缺失、不同评测指标不可比、缺乏硬件执行验证、以及对大规模量子器件验证的高成本

---

## 506. Optimizing Hospital Capacity During Pandemics: A Dual-Component Framework for Strategic Patient Relocation

**arXiv ID:** 2603.15960 | [PDF](https://arxiv.org/pdf/2603.15960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 507. How to Utilize Complementary Vision-Text Information for 2D Structure Understanding

**arXiv ID:** 2603.16245 | [PDF](https://arxiv.org/pdf/2603.16245v1)

**作者:** Jiancheng Dong `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6216 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了视觉与文本双模态在二维表格理解中的互补性，并提出一种轻量级融合架构DiVA-Former；

**💡 创新点**

创新点在于将视觉特征作为动态查询，用跨注意力将长文本压缩为结构化摘要，并通过身份保持门控实现稳定融合；

**🔧 技术方法**

技术包括冻结大型语言模型、冻结视觉编码器、跨模态Transformer层、视觉初始化查询、可学习门控、指令微调；

**📊 数据集**

在13个公开表格基准上评估，涵盖表格问答、结构理解、事实验证和表格到文本生成四大任务；

**📈 对比分析**

与直接拼接、适配器、固定查询重采样等基线相比，DiVA-Former在平均性能上提升约40%，在大多数单个任务上取得最高分；

**⚠️ 局限性**

局限包括仅在已定义训练/测试集上验证，未测试无监督或跨任务泛化，且实验仅基于Qwen3-VL-8B-Instruct等单一后端模型。

---

## 508. VIBEPASS: Can Vibe Coders Really Pass the Vibe Check?

**arXiv ID:** 2603.15921 | [PDF](https://arxiv.org/pdf/2603.15921v1)

**作者:** Srijan Bansal `[一作]` (Salesforce AI Research), Semih Yavuz `[通讯]` (Salesforce AI Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一个名为VibePass的新基准，用来测量大型语言模型在故障目标推理（fault‑targeted reasoning）上的表现，包括：①自动生成能揭露隐藏缺陷的判别性测试用例（FT‑Test）；②在有或无测试提示的情况下基于测试修复程序（FPR）。

**💡 创新点**

①首次将测试生成与程序修复拆分为两步，在同一数据集上进行端到端评估；②提出判定模型是否存在缺陷的判断步骤，形成完整的诊断‑修复链；③系统性发现“故障假设生成”是当前LLM的主要瓶颈，而非代码合成或测试有效性。

**🔧 技术方法**

使用12种前沿LLM（GPT‑5系列、Gemini‑3、Claude Sonnet/Opus、GPT‑OSS‑120B、Nemotron‑3 Nano‑30B）进行实验；利用执行式验证、输入有效性检查器、判定器等技术；采用统计相关性分析、门槛评估和多设置对比评测。

**📊 数据集**

173个实例，来自LiveCodeBench 76个算法问题；每个实例包含问题描述、官方测试集、Python输入有效性检查器、银牌（正确）解决方案和LLM产生的 buggy 解决方案。

**📈 对比分析**

通过四级判定指标（V_I、V_IO、D_I、D_IO）、判断准确率 J、以及 Pass@1 与修复成功率 SR 等，比较12个模型的表现。结果显示：V_I 接近 86% 但 D_IO 仅约 61%；在修复任务中，无测试基线往往优于自生成测试，外部测试也未必提升性能；相关性分析表明 FT‑Input / FT‑IO 与修复成功率高度相关，显示故障假设生成是关键瓶颈。

**⚠️ 局限性**

目前缺乏足够的因果推理与故障定位能力；自生成测试对修复并不总是正面影响，模型易受误导；基准主要覆盖函数级算法任务，未扩展到大规模项目；数据集规模相对有限，难以覆盖更广泛的真实世界缺陷。

---

## 509. Machines acquire scientific taste from institutional traces

**arXiv ID:** 2603.16659 | [PDF](https://arxiv.org/pdf/2603.16659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 510. PyPhonPlan: Simulating phonetic planning with dynamic neural fields and task dynamics

**arXiv ID:** 2603.16299 | [PDF](https://arxiv.org/pdf/2603.16299v1)

**作者:** Sam Kirkham `[一作]` (Lancaster University), Sam Kirkham `[通讯]` (Lancaster University)

**通讯引用:** 523 | [OpenAlex ID](https://openalex.org/A5040366382)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并发布了一个Python工具包PyPhonPlan，用于在动态神经场（Dynamic Neural Fields, DNF）和任务动力学（Task Dynamics）框架内模拟语音计划、感知与记忆的耦合动力学。

**💡 创新点**

创新点包括：①将DNF与任务动力学耦合，实现从神经场峰值自动生成轨道变量目标；②模块化架构支持多层感知/规划/记忆场的自定义耦合与门控；③引入记忆场和可选的锁定门控以防感知激活导致无意识发声；④提供可视化、动画与完整示例，降低研究门槛。

**🔧 技术方法**

技术细节：Python实现，使用微分方程式模拟一维DNF动态（τu̇ = -u + h + s + ∫k g(u) + qξ），采用高斯输入、Mexican hat交互核、sigmoid阈值；任务动力学采用临界阻尼谐振子模型；多层耦合与门控机制；绘图与动画利用Matplotlib/Plotly等。

**📊 数据集**

未使用公开语料库，而是通过自定义参数（如x∈[-10,10]）和人工设置的感知/响应输入（s_perception=1，s_response=3）进行仿真；示例为语音影子实验的模拟数据。

**📈 对比分析**

在示例中对比了基线、影子与洗脱试验，观察峰值位置与轨道变量轨迹的偏移，说明模型能捕捉语音共鸣现象；由于是理论仿真，没有与真实人类数据或其他模型的数值性能对比。

**⚠️ 局限性**

局限性包括：①输入时序手工设定，缺乏自适应时序机制；②仅实现一维控制变量，尚未扩展到多维（2–4维）轨道变量；③缺乏完整的口腔器官到轨道变量的映射；④在高维空间中DNF不易扩展，限制了对复杂语音问题的建模；⑤未与标准任务动力学或反馈控制模型进行系统性能评估。

---

## 511. NanoGS: Training-Free Gaussian Splat Simplification

**arXiv ID:** 2603.16103 | [PDF](https://arxiv.org/pdf/2603.16103v1)

**作者:** Butian Xiong `[一作]` (USC Institute for Creative Technologies), Andrew Feng `[通讯]` (USC Institute for Creative Technologies)

**通讯引用:** 2935 | [OpenAlex ID](https://openalex.org/A5066604966)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无训练、轻量级的 3D 高斯光斑简化框架 NanoGS，直接在已有的 Gaussian Splat 模型上执行后处理；

**💡 创新点**

通过在稀疏 k‑近邻图上执行局部成对合并，并提出质量保证的 Mass‑Preserving Moment Matching（MPMM）算子和基于 I‑divergence 的合并成本，避免了传统方法所需的 GPU 训练和图像监督；

**🔧 技术方法**

关键技术包括稀疏 k‑近邻图构造、基于 I‑divergence 的几何与外观合并成本评估、MPMM 合并算子、以及逐步批量无重叠边合并；

**📊 数据集**

在四个标准 3DGS 基准（NeRF‑Synthetic、Mip‑NeRF 360、Tanks&Temples、Deep Blending，共 21 个场景）上进行实验；

**📈 对比分析**

与 LightGS、PUP3DGS、GHAP 等最先进压缩/简化方法对比，在三种压缩比例（ρ=0.1、0.01、0.001）下均保持最高 PSNR，平均提升 2.4 dB（ρ=0.1）、4.8 dB（ρ=0.01）和 5.5 dB（ρ=0.001），且在极端压缩下保持更稳定的视觉质量；

**⚠️ 局限性**

局限性包括：目前仅在 CPU 上实现，处理大规模场景时耗时较长；合并成本和 MPMM 仍分别处理几何与外观，缺乏视角感知或学习式合并准则；并且仅适用于静态场景，无法直接处理动态 3DGS 表示。

---

## 512. REFORGE: Multi-modal Attacks Reveal Vulnerable Concept Unlearning in Image Generation Models

**arXiv ID:** 2603.16576 | [PDF](https://arxiv.org/pdf/2603.16576v1)

**作者:** Yong Zou `[一作]` (Yunnan University), Renyang Liu `[通讯]` (National University of Singapore)

**通讯引用:** 210 | [OpenAlex ID](https://openalex.org/A5028872220)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种针对图像生成模型去学习（IGMU）的黑盒红队框架，利用图像提示的对抗样本在保持语义一致性的同时恢复被去学习的概念。

**💡 创新点**

创新点在于：①通过将参考图像转换为笔画化初始化来保留全局布局；②使用跨注意力图生成空间掩码，将对抗噪声集中在概念相关区域；③在不需要目标模型梯度的前提下，采用代理模型进行潜在空间对齐优化。

**🔧 技术方法**

技术手段包括：笔画化图像处理（中值滤波+颜色量化+区域笔画渲染）、跨注意力掩码构造、潜在空间对齐损失（MSE）、以及黑盒查询和代理模型的使用。

**📊 数据集**

实验使用了三个去学习任务的数据集：Nudity（使用SneakyPrompt提示集合）、Object‑Parachute（UnlearnDiffAtk提示集合）和Van Gogh‑Style（UnlearnDiffAtk提示集合），并利用第三方模型生成参考图像。

**📈 对比分析**

与基线方法（SneakyPrompt、Ring‑A‑Bell、MMA等）比较，本文方法在攻击成功率（ASR）和CLIP相似度上均达到或超过最高水平，同时攻击时间仅为基线的约1/10，表明更高效、更成功的对抗。

**⚠️ 局限性**

局限性包括：需要一个公开的代理模型来计算跨注意力；对抗样本生成依赖于参考图像的质量；在极端隐私或法律敏感场景下对手可能难以获得合适的参考图像；未探讨模型在持续去学习或在线学习环境中的鲁棒性。

---

## 513. Stochastic Resetting Accelerates Policy Convergence in Reinforcement Learning

**arXiv ID:** 2603.16842 | [PDF](https://arxiv.org/pdf/2603.16842v1)

**作者:** Jello Zhou `[一作]` (Stanford University), David J. Schwab `[通讯]` (CUNY)

**通讯引用:** 2351 | [OpenAlex ID](https://openalex.org/A5060296717)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了随机重置对强化学习中策略收敛的加速作用，并证明其在网格世界、风暴悬崖和MountainCar环境中的有效性。

**💡 创新点**

首次将随机重置作为可调控制参数引入RL，并揭示其加速学习的机制与折扣因子不同。

**🔧 技术方法**

采用Q‑learning、深度Q网络（DQN）以及动态规划作为基线，并通过随机重置策略进行对比实验。

**📊 数据集**

使用公开的离散网格世界、WindyCliff、MountainCar等环境进行实验。

**📈 对比分析**

与无重置、不同折扣因子及学习率的对照实验相比，随机重置在大多数任务中显著缩短了收敛时间，尤其在探索困难、奖励稀疏的场景下提升了约20–30%，并在WindyCliff中保持了最优策略不变。

**⚠️ 局限性**

仅适用于记忆无关的重置，无法处理误导性奖励或部分可观测环境；在高探索率或奖励梯度充足的任务中，重置反而可能降低性能。

---

## 514. V-DyKnow: A Dynamic Benchmark for Time-Sensitive Knowledge in Vision Language Models

**arXiv ID:** 2603.16581 | [PDF](https://arxiv.org/pdf/2603.16581v1)

**作者:** Seyed Mahed Mousavi `[一作]` (University of Trento), Giuseppe Riccardi `[通讯]` (University of Trento)

**通讯引用:** 6026 | [OpenAlex ID](https://openalex.org/A5062879885)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 V-DyKnow，评估视觉-语言模型在时间敏感事实知识上的表现，结合视觉与文本提示，并对模型的回答进行正确/过时/无关分类。

**💡 创新点**

创新在于构建一个动态多模态基准，加入时间有效性区间，系统分析模型在不同模态、提示扰动和知识更新方法下的表现，且提供机制性与训练数据解析。

**🔧 技术方法**

使用视觉-语言模型（如 LLaVA、Qwen、Molmo、GPT‑4/5）、知识编辑技术（GRACE、WISE、IKE）和多模态检索增强生成（RAG）以及上界评估与层级可解释性方法。

**📊 数据集**

基于 DyKnow 事实与 Wikidata 实时属性构造 139 条时间敏感事实，涵盖国家、运动员、组织，并配以国徽、旗帜、肖像、标志等图像。

**📈 对比分析**

通过对 9 个 VLM 进行视觉/文本提示的评测、提示一致性、编辑与 RAG 效果比较，发现多数模型产生过时回答，视觉提示性能明显低于文本提示；GPT‑4/5 表现最佳，编辑方法效果有限。

**⚠️ 局限性**

局限包括样本主要覆盖常见实体，难以代表稀有实体；编辑效果评估未考虑潜在副作用；仅评估 7B 规模的开源模型，关闭源 API 作为上界。

---

## 515. BLADE: Adaptive Wi-Fi Contention Control for Next-Generation Real-Time Communication

**arXiv ID:** 2603.16119 | [PDF](https://arxiv.org/pdf/2603.16119v1)

**作者:** Fengqian Guo `[一作]` (Tencent), Honghao Liu `[通讯]` (Tencent)

**通讯引用:** 1807 | [OpenAlex ID](https://openalex.org/A5101801342)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对下一代实时通信（云游戏、XR等）在Wi‑Fi网络中出现的长尾延迟问题，提出了一种基于全局分布式自适应冲突窗口（CW）调节的Wi‑Fi竞争控制算法；

**💡 创新点**

创新点在于利用实时信道竞争程度动态调节所有Wi‑Fi发射器的CW，突破传统固定或半动态CW调节的局限，显著降低短时包丢失导致的延迟峰值；

**🔧 技术方法**

主要技术包括：①大规模现场测量云游戏用户延迟分布；②在ns3仿真中实现自适应CW算法；③在商业Wi‑Fi AP上进行实地实验验证；

**📊 数据集**

使用的数据集为云游戏用户在边缘服务器部署下的延迟日志（包含长尾延迟样本），以及在仿真与真实网络实验中收集的Wi‑Fi传输性能数据；

**📈 对比分析**

与标准（固定CW）竞争控制进行对比，实验表明在高信道竞争场景下，尾部延迟降低超过5倍，MAC吞吐量更稳定，同时云游戏视频卡顿率降低超过90%；

**⚠️ 局限性**

限制：算法依赖对信道竞争度的准确估计，可能在非典型Wi‑Fi硬件或极端高负载环境下需要进一步调参；同时对跨设备协同或多频段（Wi‑Fi 6/6E）支持尚未验证。

---

## 516. SAC-NeRF: Adaptive Ray Sampling for Neural Radiance Fields via Soft Actor-Critic Reinforcement Learning

**arXiv ID:** 2603.15622 | [PDF](https://arxiv.org/pdf/2603.15622v1)

**作者:** Chenyu Ge `[一作]` `[通讯]` (University of Southern California), Chenyu Ge (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于Soft Actor-Critic的自适应采样框架SAC-NeRF，学习在NeRF渲染中动态放置采样点以提升效率；

**💡 创新点**

创新点包括：1）使用高斯混合分布输出颜色并提供不确定性度量；2）设计多项奖励函数平衡渲染质量、采样效率和空间一致性；3）采用两阶段训练以缓解环境非平稳性；

**🔧 技术方法**

核心技术：Soft Actor-Critic强化学习、可学习的高斯混合颜色模型、基于渲染权重和不确定性的状态表示、以及多项奖励设计；

**📊 数据集**

使用Synthetic-NeRF（8个合成场景）和LLFF（8个实景场景）进行训练与评估；

**📈 对比分析**

与原NeRF、DONeRF、NerfAcc、AdaNeRF等基线对比，SAC-NeRF在Synthetic-NeRF上采样点减少35-48%，渲染质量仅比基线低0.3-0.8 dB PSNR；

**⚠️ 局限性**

局限性包括：需为每个场景单独训练（耗时约3小时），RL框架相对复杂，采样点削减幅度不及显式稀疏结构，且训练需依赖有标签的真实图像。

---

## 517. ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace

**arXiv ID:** 2603.16550 | [PDF](https://arxiv.org/pdf/2603.16550v1)

**作者:** Alexander Prutsch `[一作]` (Graz University of Technology), Horst Possegger `[通讯]` (Graz University of Technology)

**通讯引用:** 3430 | [OpenAlex ID](https://openalex.org/A5039382695)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种轻量级 transformer 结构，用于在非塔台终端空域预测多模 3D 航空轨迹。

**💡 创新点**

创新点包括：① 使用姿态归一化与 3D 位姿嵌入捕捉局部与全局信息；② 采用可学习的模式查询解码器直接预测飞行参数（偏航、俯仰、速度），而非直接位置；③ 将参数化输出与坐标逆变换结合，实现更稳定的长时预测。

**🔧 技术方法**

技术主要包括：位置归一化、三维位置与姿态的嵌入、Transformer 自注意力编码、MLP 解码器与模式查询、基于赢家全部的损失策略以及 smooth‑L1 与交叉熵混合损失。

**📊 数据集**

在 TrajAir（波士顿-巴特尔机场）和 TartanAviation（波士顿-巴特尔与 Allegheny 机场）两个 ADS‑B 数据集上进行实验。

**📈 对比分析**

与现有基线（TrajAirNet、Social‑PatteRNN、GooDFlight 等）在 ADE/FDE 指标上比较，模型在所有划分、历史长度和预测时长设置下均取得更低误差，显著超过先前 SOTA，并在跨数据集评估中显示出更好的迁移性。

**⚠️ 局限性**

局限性：模型对跑道位置等全局关系学习是隐式的，导致在未见机场或不同跑道布局时需要额外微调；未考虑天气等外部环境因素，未来工作可进一步扩展。

---

## 518. Breaking the Chain: A Causal Analysis of LLM Faithfulness to Intermediate Structures

**arXiv ID:** 2603.16475 | [PDF](https://arxiv.org/pdf/2603.16475v1)

**作者:** Oleg Somov `[一作]` (Artificial Intelligence Research Institute), Elena Tutubalina `[通讯]` (Artificial Intelligence Research Institute)

**通讯引用:** 1625 | [OpenAlex ID](https://openalex.org/A5012311258)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并验证了一个因果评估协议，用来检测大语言模型（LLM）在使用结构化中间推理（如清单、打分规则）时，其最终预测是否真正受到这些结构的驱动。

**💡 创新点**

创新点在于：①把对结构化推理的 faithfulness 视作因果中介问题，引入可控干预（纠正与反事实）并通过确定性映射 C 产生唯一正确答案；②发现中间结构在多数情况下仅起到上下文作用，模型对干预的响应存在方向性不对称；③通过外部工具执行确定性映射以及强化提示策略，分别验证并缓解了“计算困难”与“指令歧义”两类导致的不faithfulness。

**🔧 技术方法**

使用的技术包括：因果干预协议、前门（front‑door）结构建模、确定性评估器 C、工具调用外部化（tool calling）以及三种干预场景（纠正、反事实）。

**📊 数据集**

采用了三种公开数据集：RiceChem（化学评分任务）、AVeriTeC（事实核查任务）和 TabFact（基于表格的事实验证任务），每个数据集都有明确的中间结构与可确定的决策映射。

**📈 对比分析**

在八个指令调优模型（Qwen‑3 1.7B/4B/8B、Falcon‑3 3B/7B、LLaMA‑3 3.1B/3.2B、Gemma‑2 2B）上进行比较。结果显示：在未干预时模型往往与自己的中间结构自洽（F_ID 0.24‑0.74），但在干预后更新不足，强 faithfulness（F_Strong）显著下降，Δ 在 0.08‑0.64 之间。外部化工具几乎消除了 Δ，证明计算困难是主要原因；而加强提示仅能带来微小提升，表明指令不确定性并非主要瓶颈。

**⚠️ 局限性**

局限性包括：①依赖具有显式中间结构和确定性映射的数据集，现实场景中难以获得；②实验仅在中等规模开源模型上进行，未覆盖更大规模或闭源模型；③干预方法需要对模型完全控制，可能不适用于 API 受限的环境。

---

## 519. HIPO: Instruction Hierarchy via Constrained Reinforcement Learning

**arXiv ID:** 2603.16152 | [PDF](https://arxiv.org/pdf/2603.16152v1)

**作者:** Keru Chen `[一作]` (Arizona State University), Shaofeng Zou `[通讯]` (Arizona State University)

**通讯引用:** 1023 | [OpenAlex ID](https://openalex.org/A5012545205)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新框架 HIPO，用来解决层级指令跟随（HIF）问题，即在大模型中严格遵守系统指令的优先级并最大化用户指令的效用

**💡 创新点**

将 HIF 重新建模为受限马尔可夫决策过程（CMDP），并通过引入拉格朗日乘子实现系统指令的显式约束；利用双重奖励（系统合规与用户效用）和群组相对优势估计实现无价值网络的安全强化学习；通过动态更新乘子使模型自动将注意力重心移向系统指令

**🔧 技术方法**

安全强化学习（Primal‑Dual RL）+ 群组相对优势（GRPO）+ LLM-as‑Judge 分离评估（系统与用户）+ 受限MCP的拉格朗日方法 + KL 约束 + 组基策略梯度

**📊 数据集**

SystemCheck 数据集（约2000条系统-用户指令对，含冲突与对齐两类），以及 MMLU‑Redux、WildJailbreak、HarmBench 等评测数据集

**📈 对比分析**

与 SFT、DPO、Sys‑only、User‑only、Split‑Softmax、FocalLoRA 等六种基线比较；HIPO 在所有主流模型（Qwen、Phi、Llama）上均实现了系统合规率 ≥ 0.7（阈值）同时提升用户效用，明显超越基线，尤其在冲突场景下达成 Pareto 改进；在安全评测中保持较低 ASR 并降低过度拒绝率

**⚠️ 局限性**

1）仅在期望层面优化系统约束，极端情况可能仍出现偏差；2）需要昂贵的 LLM‑as‑Judge 计算，导致评估成本高；3）对攻击者控制系统指令存在潜在风险，需要严格权限控制

---

## 520. SOMA: Unifying Parametric Human Body Models

**arXiv ID:** 2603.16858 | [PDF](https://arxiv.org/pdf/2603.16858v1)

**作者:** Jun Saito `[一作]` (NVIDIA), Umar Iqbal `[通讯]` (NVIDIA)

**通讯引用:** 2030 | [OpenAlex ID](https://openalex.org/A5101862728)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一套统一的动画管线，将五种异构参数化人体模型（SOMA-Shape、MHR、SMPL-X、Anny、GarmentMeasurements）映射到单一的骨骼和网格拓扑，实现身份与姿态的解耦。

**💡 创新点**

创新点在于：1）通过预计算的三维三角形插值实现快速且高精度的网格拓扑抽象；2）使用RBF回归+Kabsch对齐实现一次性骨骼拟合；3）采用Newton‑Schulz正交化的逆LBS求解姿态；4）统一的MLP姿态修正网络可一次训练，适用于所有后端；5）所有模块完全可微、GPU加速，避免多模型O(M²)适配。

**🔧 技术方法**

核心技术包括三角形/四面体 barycentric 插值、RBF 回归、Kabsch 旋转拟合、Newton‑Schulz 正交化、逆 LBS 迭代、6D 旋转表示、MLP 姿态修正、Warp GPU 路径的 LBS 与 Skinning。

**📊 数据集**

使用 SizeUSA、Triplegangers（用于训练 Shape PCA）、AMASS、SAM 3D Body（用于姿态逆向评估）、以及公开的 SMPL/SMPL‑X、MHR、Anny、GarmentMeasurements 数据集进行跨模型对比。

**📈 对比分析**

与各后端的传统单独实现相比，拓扑传输误差平均 <0.1 mm，姿态逆向误差约 5.3 mm（882 FPS），前向推理吞吐率 7k‑mesh/s（batch 128），跨模型形状重建误差 SMPL‑X/Shape 约 5.5–5.8 mm，显著优于 10‑或 15‑维 SMPL/ Garment。评估覆盖 topology、姿态、速度、跨模型形状比较四维。

**⚠️ 局限性**

局限包括：①需要高质量的 wrap 注册；②仍受 LBS 限制，极端姿态仍有残留畸变；③新增后端需实现注册与适配；④姿态抽象仅适用于共享人类几何的模型，无法用于非人形或全新骨骼的角色。

---

## 521. RepoReviewer: A Local-First Multi-Agent Architecture for Repository-Level Code Review

**arXiv ID:** 2603.16107 | [PDF](https://arxiv.org/pdf/2603.16107v1)

**作者:** Peng Zhang `[一作]` `[通讯]`, Peng Zhang

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个本地优先的多代理系统 RepoReviewer，用于对 GitHub 仓库或 PR 进行全流程自动化代码审查。

**💡 创新点**

创新点在于将审查拆解为仓库获取、上下文合成、文件级审查、优先级排序和总结四个独立阶段，使用多代理架构并通过 LiteLLM 实现模型提供者无关，强调输出文件优先（JSON/Markdown）以便后续分析。

**🔧 技术方法**

采用 Python + FastAPI + LangGraph + LiteLLM + PyGithub 作为后端，Next.js + Typer CLI 作为前端，利用多代理流程实现自动化审查。

**📊 数据集**

使用公开的 GitHub 仓库和 PR 作为实验数据，未提出新的标准数据集；系统支持对任意公开仓库进行评测。

**📈 对比分析**

目前仅做了定性演示和示例评测，未给出量化基准；未来计划使用评测运行器进行多仓库实验，评估精准率、可操作性、重复率、严重程度一致性、运行时间和成本。

**⚠️ 局限性**

局限性包括仅支持公开仓库、对大型仓库的上下文处理有限、依赖外部 LLM 产生的格式不确定、严重程度标签缺乏正式校准、缺少私有仓库支持和 GitHub 原生 PR 发布功能、以及模型配额/速率限制会影响批量评测。

---

## 522. BANGLASOCIALBENCH: A Benchmark for Evaluating Sociopragmatic and Cultural Alignment of LLMs in Bangladeshi Social Interaction

**arXiv ID:** 2603.15949 | [PDF](https://arxiv.org/pdf/2603.15949v1)

**作者:** Tanvir Ahmed Sijan `[一作]` (Jahangirnagar University), Md. Musfique Anwar `[通讯]` (Jahangirnagar University)

**通讯引用:** 485 | [OpenAlex ID](https://openalex.org/A5055016333)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个评估孟加拉语社会语用能力的基准，涵盖称呼、亲属推理和社会习俗三大领域；

**💡 创新点**

首次将情境驱动、文化厚度评估与高语境语言的社会层级、亲属关系等社会语用特征结合，发现LLM存在系统性文化误差；

**🔧 技术方法**

采用Hymes SPEAKING模型的提示设计、零样本评估与token-level log概率分析，检验模型在多语言场景下的社会语用推理；

**📊 数据集**

使用由本土孟加拉语annotator编写的1719条实例，分布于地址称呼、亲属推理和社会习俗三类；

**📈 对比分析**

对12个LLM在零样本条件下进行比较，平均准确率约为地址项68%、亲属推理49%、社会习俗77%，大型专有模型表现最佳；

**⚠️ 局限性**

局限在仅覆盖标准口语孟加拉语，未包含方言、城乡差异等多样交际情境，评估基于固定标签可能忽略多样化表达。

---

## 523. Parallel In-context Learning for Large Vision Language Models

**arXiv ID:** 2603.16092 | [PDF](https://arxiv.org/pdf/2603.16092v1)

**作者:** Shin'ya Yamaguchi `[一作]` (NTT), Taku Hasegawa `[通讯]` (NTT)

**通讯引用:** 9161 | [OpenAlex ID](https://openalex.org/A5103575803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Parallel-ICL，一种并行化多模态上下文学习推理算法，通过将长演示上下文拆分成若干短块并行处理，然后在logit层用加权 Product-of-Experts 融合预测，以实现高效的 LVLM 推理。

**💡 创新点**

创新点：1) 将演示上下文拆分成多个块并行推理；2) 在 logit 层使用加权 PoE 进行融合；3) 依据集成学习理论设计聚类分块（提升多样性）和相似度加权编译（提升相关性）；4) 兼顾推理速度与准确性，在保持或超越完整上下文表现的同时显著加速。

**🔧 技术方法**

技术手段：多模态视觉语言模型（LLaVA、Qwen2.5-VL、InternVL3.5）+ Transformer + FlashAttention-2；演示分块采用 k-means 聚类（多模态特征 CLIP）；上下文编译采用基于查询相似度的加权 PoE；使用贪婪解码、最大新 token 长度 1024。

**📊 数据集**

实验数据集：MI-Bench-ICL（Demo、VQA 子集）、GQA、TextVQA、COCO Caption。

**📈 对比分析**

比较方法：与全上下文 MM-ICL（K=1）和 DivPrune（演示裁剪）做对比。Parallel-ICL 在保持或超过完整上下文准确率（≈100%+）的同时实现 1.3–1.8 倍的速度提升；与 DivPrune 组合可达 3× 速度提升，准确率仍保持 ≈102%。

**⚠️ 局限性**

局限性：① K（块数）需手动调节，过多块会产生查询处理开销，低演示数时并行收益有限；② 需要在多样性与查询相关性之间做权衡；③ 对超大模型的显存和并行计算资源有一定要求。

---

## 524. LIMBERO: A Limbed Climbing Exploration Robot Toward Traveling on Rocky Cliffs

**arXiv ID:** 2603.16531 | [PDF](https://arxiv.org/pdf/2603.16531v1)

**作者:** Kentaro Uno `[一作]` (Tohoku University), Kazuya Yoshida `[通讯]` (Tohoku University)

**通讯引用:** 13222 | [OpenAlex ID](https://openalex.org/A5023419492)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并验证了一款10公斤级四足攀岩机器人LIMBERO，配备耙式抓手，在地球重力下实现陡峭岩壁的稳定攀爬。

**💡 创新点**

创新点包括：①单电机耙式抓手实现啮合与钩挂双动作，显著提升抓握力；②基于体素的几何抓握度评估算法，能在三维点云中快速定位可抓取点。

**🔧 技术方法**

使用了脉冲传动与星形齿轮减速的机械设计、ROS2 + Python/C++的分层控制架构、LiDAR点云体素化与掩模匹配的抓握度评估算法以及力学仿真与实测。

**📊 数据集**

使用的地形数据为自行LiDAR扫描得到的约10万点的石灰岩/浮石点云，未使用公开数据集。

**📈 对比分析**

通过与LEMUR3、ReachBot等10kg级抓手的抓握力对比，LIMBERO单抓手抓握力>150N，单抓手可支撑15kg，实测携带1.4kg负载攀爬0.5m，平均步速0.017m/s，功耗约10W。

**⚠️ 局限性**

局限性在于关节扭矩不足导致上升力受限、抓手钩挂部件磨损、缺乏完全自主的抓取判定与主动关节控制，影响攀爬速度与安全性。

---

## 525. Parametric Social Identity Injection and Diversification in Public Opinion Simulation

**arXiv ID:** 2603.16142 | [PDF](https://arxiv.org/pdf/2603.16142v1)

**作者:** Hexi Wang `[一作]` (Tsinghua University), Yiqun Liu `[通讯]` (Tsinghua University)

**通讯引用:** 10107 | [OpenAlex ID](https://openalex.org/A5100668121)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大规模语言模型在公共舆论模拟中的多样性坍塌问题，提出并实现了Parametric Social Identity Injection（PSII）框架，在隐藏层注入参数化的社会身份向量，以保持群体间与群体内的多样性；

**💡 创新点**

创新点在于把社会身份从文本提示迁移到表示层，通过分层注入、噪声扰动和语言价值向量三种机制，在不需微调的情况下实现可控、细粒度的身份调制，显著缓解多样性坍塌；

**🔧 技术方法**

技术主要包括：隐藏层向量注入（forward hooks）、层级分层注入策略、Gaussian噪声扰动、语言特定价值向量训练以及KPCA可视化分析；

**📊 数据集**

使用World Values Survey（WVS）数据集，基于其259个舆论问题和剩余问卷构建身份向量；

**📈 对比分析**

与Direct、High‑Temp、Multilingual、DivReq、PE、SimVBG等多种基线对比，PSII在KL、JS、MAE、Entropy Deviation等指标上持续取得更低误差、更高熵一致性，尤其在经济进步与政治参与类问题上表现最佳；

**⚠️ 局限性**

局限性包括：对极少数群体和主观价值类问题的模拟仍不够精确；方法对模型规模和层深敏感，需针对不同LMM进行噪声与注入层调优；仅在WVS上评估，缺乏跨域验证。

---

## 526. Towards Fair and Robust Volumetric CT Classification via KL-Regularised Group Distributionally Robust Optimisation

**arXiv ID:** 2603.15941 | [PDF](https://arxiv.org/pdf/2603.15941v1)

**作者:** Samuel Johnny `[一作]` (Carnegie Mellon University Africa), Moise Busogi `[通讯]` (Carnegie Mellon University Africa)

**通讯引用:** 171 | [OpenAlex ID](https://openalex.org/A5045181032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种轻量级的CT分类框架，结合 MobileViT-XXS 切片编码器与两层 SliceTransformer 聚合器，并采用 KL 正则化的 Group DRO 目标，能够同时解决多中心分布漂移与性别公平性问题。

**💡 创新点**

创新点包括：1) 在 Group DRO 中加入 KL 正则化，防止组权重坍塌；2) 在任务二中使用细粒度的 gender×class 分组，显著提升极少数子群表现；3) 通过 MobileViT-XXS + SliceTransformer 的组合，将参数控制在 1.3M 以内，适配小规模医学数据集。

**🔧 技术方法**

使用技术包括：轻量级 MobileViT-XXS 预训练网络、两层 SliceTransformer 聚合器、KL 正则化的 Group DRO（动态组权重更新 + KL 约束）、AdamW 优化器、ImageNet-1k 预训练、对比损失与分类头。

**📊 数据集**

数据集：Task 1 为四个医院共 1,530 份胸部 CT（COVID/非COVID，包含站点标识）；Task 2 为 889 份 CT，四类肺部病理（腺癌、鳞癌、COVID、正常），每个样本带有性别标签。

**📈 对比分析**

与 ACVLAB、FDVTS 等先前方法对比；在 Task 1，Group DRO α=0.5 获得 0.835 的 F1，比最优参赛 +5.9pp；在 Task 2，Group DRO α=0.5 获得 0.815 的平均 per-gender macro F1，比最优参赛 +11.1pp，尤其将 Female Squamous F1 从 0.462 提升到 0.636。

**⚠️ 局限性**

局限性：1) 对完全缺失的域（如中心 2 的严重分布漂移）无法补偿；2) KL 正则过强会导致组权重趋向均衡，削弱对少数群体的提升；3) 方案目前仅针对小数据集，未来需结合测试时自适应或站点特定归一化以提升跨域鲁棒性。

---

## 527. Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval for Long-Term Memory

**arXiv ID:** 2603.16862 | [PDF](https://arxiv.org/pdf/2603.16862v1)

**作者:** Sahil Sen `[一作]` (PricewaterhouseCoopers), Vamse Kumar Subbiah `[通讯]` (PricewaterhouseCoopers)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5011129359)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Chronos，一种结合事件日历和对话日历的时间感知记忆框架，用于长时序对话记忆。

**💡 创新点**

引入查询条件的选择性事件抽取、动态提示生成以及事件+对话双日历索引，实现对时间敏感多跳查询的高效检索。

**🔧 技术方法**

使用LLM驱动的事件抽取、ISO8601时间规范化、多分辨率时间处理、动态提示（Prompting）、向量检索+重排序、Grep工具调用、ReAct推理等技术。

**📊 数据集**

在LongMemEvalS基准上评测，共500道多轮记忆问题。

**📈 对比分析**

与EmergenceMem、Honcho、Mastra、Zep等基线对比，Chronos Low在GPT‑4o下取得92.60%准确率，比最佳基线高7.67%；Chronos High在Claude Opus 4.6下达到95.60%。

**⚠️ 局限性**

需要双重索引导致存储增大；事件抽取离线计算成本；查询时双重检索增加推理延迟；对极长对话的存储和检索效率待进一步优化。

---

## 528. Interpretative Interfaces: Designing for AI-Mediated Reading Practices and the Knowledge Commons

**arXiv ID:** 2603.15863 | [PDF](https://arxiv.org/pdf/2603.15863v1)

**作者:** Gabrielle Benabdallah `[一作]` (University of Washington), Gabrielle Benabdallah `[通讯]` (University of Washington)

**通讯引用:** 178 | [OpenAlex ID](https://openalex.org/A5010960241)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并原型实现了“解释性界面”——允许用户跟踪并注释大型语言模型（LLM）中单词在各中间层的嵌入轨迹，从而实现对模型内部运算的交互式解读

**💡 创新点**

创新点在于：①将解释焦点从系统整体行为转向单词层面的“标记跟踪”，实现对中间表示的直接操作；②提出“swatchbook”式多样化可视化手段，让非技术用户以多种符号化方式探究Transformer内部；③强调物理操控（拖拽、抓取等）与注释，突破传统XAI的图表/对话式解释模式

**🔧 技术方法**

技术包括TransformerLens（用于提取GPT-2的中间层嵌入）、Python后端+浏览器前端（Web UI）、降维算法（PCA/UMAP）生成2D轨迹图、Figma原型设计

**📊 数据集**

数据集：使用GPT-2 124M小型Transformer作为实验模型；输入示例来自公开文本片段或实验者自行构造的提示词；未使用大型专业数据集进行评估

**📈 对比分析**

尚未进行系统性对比实验或性能评估；目前仅在原型阶段通过设计研讨会收集用户反馈，计划后续通过参与式工作坊进行可用性和认知效果的定量/定性评估

**⚠️ 局限性**

局限性包括：①仅验证于GPT-2小模型，缺乏对大规模LLM（如GPT‑3/4）的可扩展性评估；②可视化和交互复杂度较高，非技术用户可能需要较长学习曲线；③缺乏客观性能指标（如信息获取效率、错误率等），需进一步实验验证其对科研工作流程的实际帮助

---

## 529. MG-Grasp: Metric-Scale Geometric 6-DoF Grasping Framework with Sparse RGB Observations

**arXiv ID:** 2603.16270 | [PDF](https://arxiv.org/pdf/2603.16270v1)

**作者:** Kangxu Wang `[一作]` (Tsinghua University), Guijin Wang `[通讯]` (Tsinghua University)

**通讯引用:** 4129 | [OpenAlex ID](https://openalex.org/A5045183950)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MG-Grasp框架，利用稀疏多视角RGB图像重建度量尺度三维点云，并直接生成6-DoF抓取姿态，无需深度传感器。

**💡 创新点**

创新点包括：①基于两视角几何模型与三角测量实现尺度恢复；②采用两阶段置信加权多视角一致性优化精细化几何；③将几何恢复与抓取生成融入统一端到端管线。

**🔧 技术方法**

使用MASt3R两视角几何推断、三角测量尺度恢复、基于Huber损失的两阶段多视角一致性优化、MobileSAM语义分割、LoG局部抓取模型等技术。

**📊 数据集**

使用GraspNet-1Billion数据集进行评估，并在实际机器人桌面抓取实验中验证效果。

**📈 对比分析**

与多种RGB-D基准（FlexLoG、TransGrasp、HGGD等）及RGB-only方法（GraspNeRF、VG-Grasp等）对比，GraspNet RealSense/Kinect上平均AP达到63.70/66.80、56.03/57.35、23.22/20.47，实机抓取成功率87.5%，完成率100%，性能与RGB-D方法接近。

**⚠️ 局限性**

局限性在于对高反射、透明或极球形物体的几何匹配易失真，导致抓取失败；此外，多视角一致性优化仍需进一步加速以提升闭环抓取速度。

---

## 530. Toward Reliable Scientific Visualization Pipeline Construction with Structure-Aware Retrieval-Augmented LLMs

**arXiv ID:** 2603.16057 | [PDF](https://arxiv.org/pdf/2603.16057v1)

**作者:** Guanghui Zhao `[一作]` (Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences), GuiHua Shan `[通讯]` (Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种结构感知检索增强生成（RAG）工作流，用于从自然语言描述自动生成可执行的 vtk.js 科学可视化流水线，并对生成结果进行人机交互式验证和修正；

**💡 创新点**

创新点在于将管线结构与模块匹配相结合的检索机制、引入“修正成本”作为可靠性度量、以及构建可视化交互系统实现端到端执行评估；

**🔧 技术方法**

技术包括大语言模型（GPT‑5、Claude‑4‑Sonnet、DeepSeek‑V3 等）、结构化元数据抽取与模块索引、基于模块重叠的检索算法、受限提示生成、自动化与人工双重评估框架；

**📊 数据集**

使用了四个典型科学可视化数据集：Rotor、DeepWater、Isabel、RedSea，并构建了 30 条精选 vtk.js 示例代码作为检索语料；

**📈 对比分析**

与传统无检索或 LLM 直接检索相比，RAG 方案在所有任务中显著降低了修正成本，尤其在 GPT‑5 上从可执行率几乎为 100% 缩减到极少编辑；实验显示检索延迟仅 0.01 s，生成速度与人机修正成本均优于基线；

**⚠️ 局限性**

局限性包括：对复杂 API（如 vtkCalculator）的配置错误仍导致高修正成本；检索库仍有限，无法覆盖所有新颖或极端任务；评估缺乏自动化视觉质量指标，且修正成本计量可能受注释和代码重构影响。

---

## 531. RadAnnotate: Large Language Models for Efficient and Reliable Radiology Report Annotation

**arXiv ID:** 2603.16002 | [PDF](https://arxiv.org/pdf/2603.16002v1)

**作者:** Saisha Pradeep Shetty `[一作]` (University of California), Vladimir Filkov `[通讯]` (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一套基于大语言模型的RadAnnotate框架，完成RadGraph实体标注并实现置信度驱动的选择性自动化流程。

**💡 创新点**

创新点：①针对每个实体类别训练专用LLM分类器，避免多类模型受高频类别主导；②采用检索增强生成（RAG）生成合成报告，显著提升低资源下的标注效果；③为每种实体学习置信度阈值，实现自动标注与人工审阅的动态分流。

**🔧 技术方法**

技术：大语言模型（Qwen2.5‑7B、Qwen2.5‑32B、Mixtral‑8x7B‑Instruct）、Retrieval‑Augmented Generation、QLoRA微调、指令调优、置信度阈值搜索与校准（ISOTONIC）。

**📊 数据集**

数据集：RadGraph（来自MIMIC‑CXR与CheXpert），对报告进行句子级切分，形成约2,425句子训练集；同时生成约3,050条合成报告用于验证。

**📈 对比分析**

比较方法：与RadGraph DyGIE++基线、Gold‑only、Synthetic‑only、Gold+Synthetic（30%）训练集对比。结果显示：Gold‑only模型F1≈0.87；Synthetic‑only模型仅落后1–2 F1点；在仅50报金标数据的低资源设置下，合成数据提升F1至0.70；在自动化实验中，置信度阈值可实现55%–90%报告覆盖，匹配分数0.86–0.92。

**⚠️ 局限性**

局限：仅完成实体标注，未涵盖关系抽取；合成数据可能继承源语料偏差；阈值需在新环境重新校准；未评估对临床下游任务的直接影响；验证范围仅限于少数机构的RadGraph子集。

---

## 532. Unifying Optimization and Dynamics to Parallelize Sequential Computation: A Guide to Parallel Newton Methods for Breaking Sequential Bottlenecks

**arXiv ID:** 2603.16850 | [PDF](https://arxiv.org/pdf/2603.16850v1)

**作者:** Xavier Gonzalez `[一作]` `[通讯]` (myUni), Xavier Gonzalez (myUni)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种可扩展且稳定的并行Newton方法，能够在GPU上并行化深度学习和动态系统的计算。

**💡 创新点**

创新点在于设计了准DEER（quasi-DEER）框架实现可扩展性，并通过ELK（Evaluating Levenberg-Marquardt with Kalman）信赖域技术实现数值稳定性，同时统一了多种固定点并行方法。

**🔧 技术方法**

采用GPU并行扫描、准Newton优化、信赖域策略、Kalman滤波以及Levenberg-Marquardt算法等技术实现并行化和收敛加速。

**📊 数据集**

实验代码托管于<https://github.com/lindermanlab/elk>，具体使用的数据集未在文中给出。

**📈 对比分析**

论文通过实验验证了方法的收敛速率和稳定性，但未给出与其他基准方法的详细性能对比数据。

**⚠️ 局限性**

主要限制是对系统动态的可预测性要求较高，若预测性差，导致目标函数条件差，收敛速度慢甚至数值失效；此外缺乏公开数据集和全面的基准评估。

---

## 533. VIGIL: Towards Edge-Extended Agentic AI for Enterprise IT Support

**arXiv ID:** 2603.16110 | [PDF](https://arxiv.org/pdf/2603.16110v1)

**作者:** Sarthak Ahuja `[一作]` (Amazon), Rebecca Steinert `[通讯]` (Amazon)

**通讯引用:** 752 | [OpenAlex ID](https://openalex.org/A5007363439)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并测试了 VIGIL，一套在企业端点上执行诊断、检索知识和执行修复的分布式边缘自治 AI 系统，旨在降低 IT 支持交互轮次并提高自助解决率。

**💡 创新点**

创新点包括：将 LLM 与工具增强、检索增强生成、规划与执行四种 Agent 结合到边缘端；通过云协同实现治理、可观察性与安全策略控制；实现端到端可审计的操作流，并在有限连接环境下保持自适应诊断。

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）、检索增强生成、ReAct 规划、工具接口协议（MCP）、Open Policy Agent (OPA) 的策略治理、Strands 与 Bedrock 的边缘执行框架、密集嵌入对比与 LLM 验证、自动化评估脚本。

**📊 数据集**

主要使用企业内部数据：IT 知识库（门户文章与已解决案例），一个 60,000 条历史交互的中央图谱（CGR），以及 1,586 次工具调用日志和 153 次交互记录，实验基于 100 台资源受限 Windows 端点。

**📈 对比分析**

通过与历史人类支持案例匹配评估性能：匹配率 39%；交互轮次减少 39%（11 次 vs 18 次）；诊断时间 36.5 秒 vs ≥2.7 分钟（至少 4× 加速）；响应质量平均 8/10；工具成功率 95.3%；自助解决率 82%。整体效果显著优于传统集中式支持。

**⚠️ 局限性**

局限性：匹配覆盖仅 39%，未匹配案例可能更复杂；评估基于自动 LLM 评分，缺乏人工专家验证；问卷回应率低 23/100，可能存在自选偏差；实验仅在单一硬件/软件配置上，缺乏跨异构设备的验证；自我改进循环未在本部署中激活。

---

## 534. Exploring the Use of VLMs for Navigation Assistance for People with Blindness and Low Vision

**arXiv ID:** 2603.15624 | [PDF](https://arxiv.org/pdf/2603.15624v1)

**作者:** Yu Li `[一作]` (Columbia University), John-Ross Rizzo `[通讯]` (New York University)

**通讯引用:** 8807 | [OpenAlex ID](https://openalex.org/A5073829818)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对闭源与开源视觉语言模型在盲/低视障人士导航任务中的基础视觉能力和实战表现进行系统评估

**💡 创新点**

提出基于重复测量的评估框架和专门构造的导航场景数据集，揭示模型的鲁棒性与易失效模式

**🔧 技术方法**

采用 GPT‑4V、GPT‑4o、Gemini‑1.5‑Pro、Claude‑3.5‑Sonnet、Llava‑mistral 与 Llava‑qwen 等模型，配合自然语言提示和人类标注进行评测

**📊 数据集**

使用公开的 150 张室内/室外图像数据集（GitHub: https://github.com/rizzojr01/vlm_navigation_eval.git），并对 8 个子任务进行 100 次重复测试

**📈 对比分析**

通过准确率、均值/方差、人工评价等指标对模型进行对比；GPT‑4o 在计数、空间推理和常识理解上均超过 90%，但开源模型表现相对逊色

**⚠️ 局限性**

局限在于数据集过于静态、缺乏动态光照与运动模糊；模型仍表现出易失效、空间偏差、对分辨率敏感等问题，限制了实际部署

---

## 535. AI-Generated Figures in Academic Publishing: Policies, Tools, and Practical Guidelines

**arXiv ID:** 2603.16159 | [PDF](https://arxiv.org/pdf/2603.16159v1)

**作者:** Davie Chen `[一作]` `[通讯]` (University of Arts in Poznań), Davie Chen (University of Arts in Poznań)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性调查了主要学术出版商对AI生成图像的政策，并提出了可操作的披露与质量控制指南。

**💡 创新点**

首次将多家顶级期刊的AI图像政策进行对比，识别核心关注点（可重复性、署名、信息误导），并在此基础上制定统一的最佳实践框架。

**🔧 技术方法**

使用自然语言生成模型（如SciDraw平台）结合图像生成与迭代对话的技术；对生成参数、版本和生成过程进行元数据记录。

**📊 数据集**

未采用传统意义上的实验数据集，而是以SciDraw生成的示例图（分子机制、实验设计、研究框架等）作为案例说明。

**📈 对比分析**

通过对比分析不同期刊政策的严格程度，并结合示例图展示生成质量；在可视化质量方面，SciDraw在文本渲染、风格一致性与元数据记录上优于通用模型。

**⚠️ 局限性**

局限性在于政策碎片化缺乏统一标准，AI生成的图像仍面临可重复性与版权问题，且评估主要基于案例示例而非大规模实验验证。

---

## 536. Development of Low-Cost and Bidirectional Syringe Pumps for Soft Robotics Applications

**arXiv ID:** 2603.16803 | [PDF](https://arxiv.org/pdf/2603.16803v1)

**作者:** Krishamsu Subedi Chhetri `[一作]` (Union), John Rieffel `[通讯]` (Union)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了一种低成本、模块化、可逆的注射泵系统，用于软体机器人硅胶体素（Silibots）的气压驱动。

**💡 创新点**

创新点在于将步进电机与螺杆驱动相结合，实现高精度的气体推拉；硬件模块化和可定制软件支持多泵并行运行；并兼容BROOKS仿真平台，成本显著低于传统医疗级泵。

**🔧 技术方法**

采用步进电机及其驱动、Arduino微控制器、铝框架、3D打印部件、线性滚珠轴承、螺杆与丝杠、医疗注射器等硬件技术。

**📊 数据集**

本文未使用公开数据集，实验主要在Silibots软体机器人上进行，并兼容BROOKS仿真平台。

**📈 对比分析**

与高压电磁阀、医疗级注射泵和隔膜泵等现有驱动方式对比，阐述了成本（<200美元/台 vs 2000–5000美元）、可调性、可扩展性以及精确控制等优势；实验表明装配时间约20分钟，气体位移精准且噪音低。

**⚠️ 局限性**

局限性包括尚未在高频或长周期运行下验证耐久性；目前仅在Silibots上测试，需进一步评估在更复杂软体机器人中的适用性；以及在极低压或高压极限下的性能仍待验证。

---

## 537. RetailBench: Evaluating Long-Horizon Autonomous Decision-Making and Strategy Stability of LLM Agents in Realistic Retail Environments

**arXiv ID:** 2603.16453 | [PDF](https://arxiv.org/pdf/2603.16453v1)

**作者:** Linghua Zhang `[一作]` (Ant Group), Zhisong Zhang `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RetailBench 基准和 Evolving Strategy & Execution 框架，用于评估和提升 LLM 代理在长周期零售决策中的表现。

**💡 创新点**

创新点在于创建高保真长周期零售环境，并将战略演化与执行分离，支持在非静态环境中按不同时间尺度更新策略。

**🔧 技术方法**

使用了基于大型语言模型的推理代理、prompting 技术和框架实现；在实验中对 8 个主流 LLM 进行评估。

**📊 数据集**

利用 RetailBench 仿真数据，该环境基于历史零售数据构建，模拟需求随机性、新闻动态和供应链延迟。

**📈 对比分析**

与 Reflection 等基线对比，实验显示该框架在操作稳定性和经济效益上有所提升，但随着任务难度增加性能显著下降。

**⚠️ 局限性**

局限包括单店设置、仿真简化、未使用 RL 或微调进行长期学习、未强制执行经济约束，且缺乏多代理协作与竞争环境。

---

## 538. Evaluating Agentic Optimization on Large Codebases

**arXiv ID:** 2603.16011 | [PDF](https://arxiv.org/pdf/2603.16011v1)

**作者:** Atharva Sehgal `[一作]` (University of Texas at Austin), Yisong Yue `[通讯]` (California Institute of Technology)

**通讯引用:** 6519 | [OpenAlex ID](https://openalex.org/A5085826758)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并构建了一个针对大型代码库的代理式优化基准（Formula Code Benchmark），评估大规模 LLM 代理在满足严格正确性约束下对真实开源项目执行多目标性能优化的能力。

**💡 创新点**

创新点包括：① 引入细粒度、多工作负载的性能度量（基于 Airspeed Velocity）和层级化优势（Stratified Advantage）以捕获全局与局部优化效果；② 通过专家补丁和社区维护的性能工作负载构建真实可复现的任务集合；③ 提供成本加权评价指标与动态增量任务流，实现实时、可持续的基准评估。

**🔧 技术方法**

技术手段包括：使用 GPT‑5、Claude 4.0 Sonnet、Gemini 2.5 Pro 与 Qwen 3 Coder 等前沿 LLM，结合 Terminus 2 与 OpenHands 等代理框架；利用 LLM 进行意图分类与环境构建，使用 Docker 自动化构建；统计检验（Mann‑Whitney U）与严格的功能正确性测试；以及自定义速度提升、优势、标准化优势和成本加权优势等评价公式。

**📊 数据集**

数据集来源于 70 个科研 Python 开源仓库（如 Pandas、SciPy、Scikit‑learn、Astropy 等），共 957 个任务，包含 1,232 条 Docker 环境脚本、59,000+ ASV 性能工作负载以及 3,181 条专家补丁。

**📈 对比分析**

比较方法：将代理模型的相对优势（Human‑Relative Advantage）和标准化优势（Normalized Advantage）与专家补丁进行对比，并计算成本加权优势；结果表明代理均能在基线代码上获得 >1 的速度提升，但整体优势为负，低于人类专家；在功能级别优化上表现较好，而在模块级或向量化优化上存在显著不足；高端 LLM 通过较高的推理成本实现更优的成本效益。

**⚠️ 局限性**

局限性：① 在小星标（tail）仓库中表现最差，说明对分布外项目的泛化受限；② 代理在多工作负载平衡和大规模优化上仍欠缺；③ 虽然数据泄露影响有限，但仍需持续监控；④ 评价过程依赖人工维护的工作负载，维护成本较高；⑤ 只关注运行时/内存等单一性能维度，可能忽略其他重要优化维度。

---

## 539. Sample-Efficient Adaptation of Drug-Response Models to Patient Tumors under Strong Biological Domain Shift

**arXiv ID:** 2603.16185 | [PDF](https://arxiv.org/pdf/2603.16185v1)

**作者:** Camille Jimenez Cortes `[一作]` (Universite Grenoble Alpes), German Vega `[通讯]` (Universite Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并评估了 STaR-DR 这一分阶段迁移学习框架，通过无监督预训练细胞和药物表征、任务对齐以及少样本临床适配实现药物反应预测。

**💡 创新点**

创新点在于将无监督表征学习、任务监督与少样本临床适配明确分离为三阶段，证明在强生物域移位下可显著提升适配效率。

**🔧 技术方法**

采用自动编码器进行表征学习、轻量化 MLP 预测头以及少样本（few-shot）微调，并用 PCA/tSNE 对潜在空间进行可视化分析。

**📊 数据集**

使用 CTRP–GDSC 进行无监督预训练与任务对齐，CCLE 用于跨数据集评估，TCGA 作为临床患者数据进行少样本适配。

**📈 对比分析**

与单阶段 AE–MLP 基线在 pair‑level、LCO、LDO、CCLE 和 TCGA 评估中比较，STaR‑DR 在患者少样本场景下的 ROC‑AUC 与 PR‑AUC 均提升约 5–10%，但在体外基准无显著差异。

**⚠️ 局限性**

仍无法实现零样本临床迁移，药物表征提升有限，需要更丰富的化学与多组学信息来进一步弥补细胞系与患者肿瘤之间的生物学鸿沟。

---

## 540. Domain Mixture Design via Log-Likelihood Differences for Aligning Language Models with a Target Model

**arXiv ID:** 2603.16622 | [PDF](https://arxiv.org/pdf/2603.16622v1)

**作者:** Ryo Kishino `[一作]` (Kyoto University), Hidetoshi Shimodaira `[通讯]` (RIKEN)

**通讯引用:** 17493 | [OpenAlex ID](https://openalex.org/A5012479520)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过设计训练数据的域混合权重，使基模型在预训练或持续预训练过程中与目标模型在分布上对齐。

**💡 创新点**

提出基于对数似然空间的几何公式，推导域权重选择规则，并用 softmax 与温度调节实现无须知识蒸馏的域权重估计。

**🔧 技术方法**

使用对数似然向量、Gram 矩阵、softmax 估计域权重、t‑SNE 可视化、KL 距离评估、AdamW 优化器以及实验中的多种对比方法。

**📊 数据集**

在 Pile Uncopyrighted（17 个域）上使用 NanoGPT 124M 作为基模型，目标模型为 Gemma‑2B、CodeGemma‑2B、CodeNanoGPT 等。

**📈 对比分析**

与均匀域权重、理论调整 LLD（adjusted‑LLD）、迭代 LLD（iterative‑LLD）以及知识蒸馏对比：聚合 LLD 在 KL 距离上明显低于均匀权重（约 1–2 bits/byte 的提升），略低于理论调整 LLD，但比知识蒸馏低；在下游任务上，聚合 LLD 的性能更接近目标模型，Pearson 相关性随 KL 减小而提升。

**⚠️ 局限性**

仅在 124M NanoGPT 上验证；对更大模型或不同架构的适用性未知；依赖明确域划分的 Pile，无法直接推广至无域或多域重叠的数据；使用固定 token 长度训练与可变长度评估的正则化选择尚未彻底评估；理论假设 SGD 但实验使用 AdamW，可能导致偏差；Gram 矩阵计算成本高，温度参数固定，未探索自适应或多轮蒸馏的潜在改进。

---

## 541. Conservative Continuous-Time Treatment Optimization

**arXiv ID:** 2603.16789 | [PDF](https://arxiv.org/pdf/2603.16789v1)

**作者:** Nora Schneider `[一作]` (Technical University of Munich), Niki Kilbertus `[通讯]` (Technical University of Munich)

**通讯引用:** 1042 | [OpenAlex ID](https://openalex.org/A5000443214)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种保守的连续时间治疗优化框架，从不规则采样的患者轨迹中学习并优化治疗方案。

**💡 创新点**

创新点在于：①将患者动力学建模为受控随机微分方程；②在优化目标中加入基于路径签名的最大均值差异（MMD）正则化，以限制模型误差导致的过度优化；③给出可一致估计的保守上界。

**🔧 技术方法**

采用的技术包括：受控SDE、神经SDE（神经网络拟漂移和扩散）、签名变换与签名核、条件MMD/MCMD估计、单射击优化。

**📊 数据集**

使用的数据集为：肺癌PK/PD仿真（两种治疗方案，60天、随机掩码30%）和COVID‑19疾病进程仿真（一次干预、14天、掩码30%）。

**📈 对比分析**

与基线TE‑CDE和INSITE进行对比：在癌症任务中，本方法在候选方案排名相关性和真实成本均优于基线；在COVID‑19任务中TE‑CDE在排名相关性上略优，但本方法在真实成本上更好；正则化强度λ越大可显著降低真实成本。

**⚠️ 局限性**

局限性包括：需要良好指定的动力学模型；正则化参数敏感；依赖正重叠/正向性假设；在数据稀缺或分布漂移严重时仍可能出现误导性优化；仅在仿真环境验证，真实临床数据仍需严格验证。

---

## 542. TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown Schemas

**arXiv ID:** 2603.16448 | [PDF](https://arxiv.org/pdf/2603.16448v1)

**作者:** Ai Jian `[一作]` (Beijing University of Posts and Telecommunications), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TRUST‑SQL框架，在未知模式下通过主动探测数据库元数据完成文本到SQL的翻译；

**💡 创新点**

创新点在于四阶段交互协议（Explore‑Propose‑Generate‑Confirm）与Dual‑Track GRPO训练策略，解决信息获取与SQL生成的信用分配问题；

**🔧 技术方法**

使用的技术包括POMDP建模、Token‑level masked advantage、GRPO、LLM（如Qwen3-4B/8B）以及工具调用接口；

**📊 数据集**

实验数据集为BIRD、Spider及其变体Spider‑Syn、Spider‑DK、Spider‑Realistic，涵盖大规模与鲁棒性场景；

**📈 对比分析**

与全模式预填充、单回合和多回合RL基线对比，TRUST‑SQL在Unknown Schema设置下平均提升30.6%（4B）/16.6%（8B），在多数基准上超越或匹配预填充模型；

**⚠️ 局限性**

主要局限包括：多回合交互导致推理延迟；仅在SQLite方言下验证；固定回合上限可能限制对复杂模式的充分探索。

---

## 543. Enforcing Task-Specified Compliance Bounds for Humanoids via Anisotropic Lipschitz-Constrained Policies

**arXiv ID:** 2603.16180 | [PDF](https://arxiv.org/pdf/2603.16180v1)

**作者:** Zewen He `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Yoshihiko Nakamura `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 20660 | [OpenAlex ID](https://openalex.org/A5040633710)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种将任务空间刚度上限映射为各向异性 Lipschitz 约束并通过梯度惩罚实现的 Stiffness‑Induced Lipschitz Constrained (SILC) 策略。

**💡 创新点**

创新点在于将任务级别的刚度预算转化为策略雅可比矩阵的方向性 Lipschitz 上限，从而实现可调节的物理可解释顺畅控制。

**🔧 技术方法**

采用了 Lipschitz 约束、hinge‑squared 谱范数惩罚、PPO 强化学习、PD 低层控制器以及关节位置敏感性分析。

**📊 数据集**

使用 AMASS 运动数据集作为上身运动先验，结合 Isaac Gym 仿真环境和真实 G1 人形机器人进行实验。

**📈 对比分析**

与基线和标量 LCP 进行对比，在仿真和硬件测试中表现出更低的动作/扭矩抖动、冲击鲁棒性更强、能耗略高但高频能量显著降低。

**⚠️ 局限性**

主要局限包括无法在运行时动态调整刚度上限、仅考虑关节位置敏感性且未对阻尼进行约束。

---

## 544. Rethinking UMM Visual Generation: Masked Modeling for Efficient Image-Only Pre-training

**arXiv ID:** 2603.16139 | [PDF](https://arxiv.org/pdf/2603.16139v1)

**作者:** Peng Sun `[一作]` (Westlake University), Tao Lin `[通讯]` (Westlake University)

**通讯引用:** 8828 | [OpenAlex ID](https://openalex.org/A5100702153)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种两阶段训练框架，先用大量无标注图像进行图像仅预训练，再用图像与文本图像对混合数据进行微调，以构建统一多模态模型。

**💡 创新点**

创新点在于引入Image-Only Training（IOMM）思路，利用冻结的多模态大语言模型与残差查询适配器，并采用掩码图像建模实现高效且可扩展的视觉生成。

**🔧 技术方法**

主要技术包括多模态扩散变换器（MM-DiT）架构、残差查询适配器、掩码图像建模、Flow Matching、冻结的InternVL3-2B特征提取器，以及LoRA等参数高效微调方法。

**📊 数据集**

预训练使用公开的 Megalith-10M 与 text-to-image-2M 数据集；微调阶段使用 BLIP3-o-60K、Echo-4o-Image 与 ShareGPT-4o-Image 等高质量指令式数据集。

**📈 对比分析**

在 GenEval、DPGBench、WISE 等基准上与 BAGEL、BLIP3-o 等SOTA模型对比，-B 模型在 GenEval 上达 0.89、WISE 0.55，显著超越传统方法，且训练成本仅 1050 H800 GPU 小时，显示出卓越的效率和性能。

**⚠️ 局限性**

局限性包括：大模型（-L）在资源受限下训练不足导致性能未完全发挥；混合数据比例与掩码比率需细致调优，过高或过低均可能降低效果；模型仍依赖一定量的文本-图像对，完全无标注学习的可迁移性尚有限。

---

## 545. Argumentative Human-AI Decision-Making: Toward AI Agents That Reason With Us, Not For Us

**arXiv ID:** 2603.15946 | [PDF](https://arxiv.org/pdf/2603.15946v1)

**作者:** Stylianos Loukas Vasileiou `[一作]` (New Mexico State University), William Yeoh `[通讯]` (Washington University in St. Louis)

**通讯引用:** 1730 | [OpenAlex ID](https://openalex.org/A5030505534)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并提出将大语言模型（LLM）与计算论证（CA）融合的研究方向，阐述了三大核心任务（论证挖掘、合成与推理）以及它们在实现可争议人机决策中的作用。

**💡 创新点**

创新点在于提出“可争议 AI 架构”，将 LLM 的生成能力与 CA 的正式推理引擎分离，形成可视化、可编辑的论证框架，使人机能够在对话中协同提案、批判、修订。

**🔧 技术方法**

主要技术包括：在 LLM 中使用少量样本提示、指令微调、生成‑批判‑修订循环；在 CA 中采用结构化论证框架、确定性求解器以及多模态多代理辩论。

**📊 数据集**

参考的数据集有 OpenDebateEvidence、覆盖论证工作流的多任务数据集等，用于训练与评估 LLM 的挖掘与生成能力。

**📈 对比分析**

对比常规 Transformer（如 RoBERTa）等基线，LLM 在论证挖掘与合成任务中取得了更高的准确率和更好的可迁移性，论文指出在大规模文本上的性能优于传统模板化方法，但在端到端链式任务中仍易受误差传播影响。

**⚠️ 局限性**

局限性包括缺乏针对多轮人机交互的评估指标、难以衡量协同决策是否真能提升效率、对模型不确定性的表达不足、算力成本与隐私合规挑战，以及需要进一步对齐专业领域规范。

---

## 546. TinyGLASS: Real-Time Self-Supervised In-Sensor Anomaly Detection

**arXiv ID:** 2603.16451 | [PDF](https://arxiv.org/pdf/2603.16451v1)

**作者:** Pietro Bonazzi `[一作]` (ETH Zurich), Michele Magno `[通讯]` (ETH Zurich)

**通讯引用:** 7798 | [OpenAlex ID](https://openalex.org/A5066423975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 TinyGLASS，一种轻量级自监督视觉缺陷检测框架，可在 Sony IMX500 传感器上实现实时推理。

**💡 创新点**

创新点在于将 GLASS 框架压缩为 ResNet‑18 主干，加入静态图跟踪与 INT8 量化，使参数压缩率达到 8.7×并兼容低功耗边缘硬件。

**🔧 技术方法**

使用 ResNet‑18、特征级融合、局部/全局异常评分、二元交叉熵＋焦点损失训练、Sony Model Compression Toolkit 进行 INT8 量化。

**📊 数据集**

在 MVTec‑AD 基准数据集和自研的 MMS（显微镜＋IMX500）双摄像头数据集上进行评估。

**📈 对比分析**

与原 GLASS 对比，TinyGLASS 在 MVTec‑AD 上获得 94.2% 的图像级 AUROC，保持 20 FPS、5.41 MB 内存、4.0 mJ/推理功耗，参数量仅为 2.9 M，压缩率 8.6×。

**⚠️ 局限性**

在训练集污染 30% 时性能下降约 7–8 pp，且在极小缺陷检测上表现略逊，未讨论更深层次的模型压缩或更复杂硬件环境的适配。

---

## 547. Segmentation-before-Staining Improves Structural Fidelity in Virtual IHC-to-Multiplex IF Translation

**arXiv ID:** 2603.16160 | [PDF](https://arxiv.org/pdf/2603.16160v1)

**作者:** Junhyeok Lee `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University)

**通讯引用:** 3673 | [OpenAlex ID](https://openalex.org/A5052023515)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种无监督、与架构无关的软先验条件方法，在IHC到多通道mIF的虚拟染色中加入基于Cellpose的连续细胞概率图，并结合局部方差保持正则化，以提升核结构的保真度。

**💡 创新点**

创新点包括：①使用预训练细胞分割模型的连续概率图作为软结构先验，无需额外训练或阈值选择；②在训练损失中加入方差保持正则化，维护细胞级别的局部强度异质性。

**🔧 技术方法**

使用技术包括Cellpose概率映射、条件GAN/回归/扩散模型、方差保持正则化、SSIM、LPIPS、Ki67阳性率误差等评估指标。

**📊 数据集**

使用的数据集为公开的DeepLIIF（IHC↔mIF多通道）和HNSCC（mIHC↔DAPI）配对数据集。

**📈 对比分析**

在Pix2Pix、U-Net、ResNet、回归U-Net和DDPM等多种架构上做对比；加入软先验和方差保持后，SSIM平均提升约0.01–0.4，LPIPS下降约0.02–0.5，Ki67阳性率误差基本不变，像素级MAE下降，表明结构保真和定量性能均得到改善。

**⚠️ 局限性**

局限性包括：未在更大、更多样化的组织或染色条件下验证，方差正则化对GAN/回归模型提升有限，整体性能仍受Cellpose分割质量影响，且未系统评估对临床决策阈值的跨越情况。

---

## 548. Do Not Leave a Gap: Hallucination-Free Object Concealment in Vision-Language Models

**arXiv ID:** 2603.15940 | [PDF](https://arxiv.org/pdf/2603.15940v1)

**作者:** Amira Guesmi `[一作]` (New York University), Muhammad Shafique `[通讯]` (New York University)

**通讯引用:** 11159 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对视觉语言模型的目标物体隐匿攻击，提出一种保持语义连续性的背景一致重编码方法。

**💡 创新点**

创新点在于不通过抑制或遮蔽目标区域，而是将其视觉特征重编码为与背景统计和语义一致，从而消除幻觉。

**🔧 技术方法**

使用多层视觉Transformer的统计对齐、字典投影和背景保持损失，结合像素级优化与总变差正则。

**📊 数据集**

在ImageNet和COCO的验证集上评估，使用真实的目标框作为ROI。

**📈 对比分析**

与VIP等抑制攻击对比，BCR在保持对象隐匿率的同时将基于视觉的幻觉率降低约3倍，语义漂移显著减少。

**⚠️ 局限性**

局限在于需要白盒访问并且对较大扰动预算仍有限；对非常小或复杂目标的重编码效果不佳。

---

## 549. Parallelised Differentiable Straightest Geodesics for 3D Meshes

**arXiv ID:** 2603.15780 | [PDF](https://arxiv.org/pdf/2603.15780v1)

**作者:** Hippolyte Verninas `[一作]` (Imperial College London), Simone Foti `[通讯]` (Imperial College London)

**通讯引用:** 120 | [OpenAlex ID](https://openalex.org/A5081602782)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一套可微、GPU并行的三维网格指数映射（Exponential Map）及其测地线、并行传输等Riemannian算子，并在此基础上提出自适应测地线卷积、MeshFlow流匹配与Mesh‑LBFGS二阶优化等应用。

**💡 创新点**

①首次实现直行测地线的可微分指数映射；②通过CUDA并行化实现数万条测地线的几毫秒级求解；③提出两种可微分方案：Extrinsic Proxy（高效近似）和Geodesic Finite Differences（更精确）；④将该算子直接嵌入GCNN、流匹配与二阶优化，提升学习与优化效果。

**🔧 技术方法**

直行测地线算法、欧氏代理与测地线有限差分、CUDA并行核、自动微分链式、Riemannian几何理论、LBFGS优化、最优传输、双调和距离等。

**📊 数据集**

Stanford Bunny、Spot the Cow、Composite Human Body 数据集（多种人形体），以及面数从200到1.7M的多尺寸网格用于基准测试。

**📈 对比分析**

与 Geometry‑central、p3d、Projection Integration 等现有实现进行精度与速度对比；精度相当、在大批量/大面数网格上速度提升3‑4个数量级；在AGC、MeshFlow、Mesh‑LBFGS 等任务中分别比SOTA方法提升准确率、降低损失、加速收敛，并在GPU上实现常数级运行时间。

**⚠️ 局限性**

①测地线在网格边界或孔洞处可能中断；②Extrinsic Proxy 对起点变化不敏感，需要人工置零；③GFD 计算量大，速度慢；④仅支持三角网格；⑤单GPU内存受限于极大网格；⑥部分任务仍需进一步完善Riemannian算子兼容性。

---

## 550. Exclusivity-Guided Mask Learning for Semi-Supervised Crowd Instance Segmentation and Counting

**arXiv ID:** 2603.16241 | [PDF](https://arxiv.org/pdf/2603.16241v1)

**作者:** Jiyang Huang `[一作]` (Harbin Institute of Technology), Antoni B. Chan `[通讯]` (City University of Hong Kong)

**通讯引用:** 12216 | [OpenAlex ID](https://openalex.org/A5065680386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于掩码监督的半监督人群实例分割与计数框架。

**💡 创新点**

创新点在于引入Exclusion-Constrained Dual-Prompt SAM (EDP-SAM)自动生成高质量掩码，以及基于排他性引导的XMask学习方法。

**🔧 技术方法**

使用了SAM、NNEC约束、深度可分离高斯平滑、可微中心采样等技术，并在半监督Mean Teacher框架下训练。

**📊 数据集**

在ShanghaiTech A、UCF-QNRF和JHU++三大人群计数数据集上进行实验。

**📈 对比分析**

与FastSAM、P2R等现有方法比较，EDP-SAM + XMask在5%、10%、40%标注比例下的IoU和MAE都取得了显著提升，速度提升近6倍。

**⚠️ 局限性**

局限在于仍依赖人工校正掩码、对极稀疏或极密集场景的边界精度有限。

---

## 551. Mask Is What DLLM Needs: A Masked Data Training Paradigm for Diffusion LLMs

**arXiv ID:** 2603.15803 | [PDF](https://arxiv.org/pdf/2603.15803v1)

**作者:** Linrui Ma `[一作]` (Huawei), Yunhe Wang `[通讯]` (Huawei)

**通讯引用:** 14349 | [OpenAlex ID](https://openalex.org/A5100727358)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种信息密度驱动的智能噪声调度器，用于改进离散扩散语言模型的监督微调

**💡 创新点**

创新点在于通过离线LLM提取信息密集枢纽并采用互补优先掩码，将噪声分布从输入无关的均匀方式转变为基于信息密度的分层掩码，从而在同一训练样本中实现逻辑推理与语法结构的双重学习

**🔧 技术方法**

技术包括信息密度区域提取（LLM/规则），优先掩码（权重w>1），互补掩码对（M与¬M），以及离散扩散模型的ELBO优化与块扩散训练

**📊 数据集**

使用了混合的OPC‑SFT‑Stage2（代码）和GSM8K（数学）数据集，共约45万条样本；信息密集区域通过GPT‑4o在10%–50%数据上提取；公开数据集链接：https://huggingface.co/datasets/malr07/opc-sft-stage2-dense-extracted

**📈 对比分析**

与统一随机掩码的基线进行对比，平均提升约4%（HumanEval +7.3%，MATH500 +6.2%），在四个基准（HumanEval、MBPP、GSM8K、MATH500）上的总体平均分从55.32%提升至59.19%；实验还验证了w≈2的软优先掩码最佳，硬掩码导致上下文崩溃，互补掩码显著提升性能

**⚠️ 局限性**

局限性包括对LLM的离线提取依赖、在高比例注解时会出现领域偏移导致数学推理下降、硬优先掩码在块扩散中易导致梯度崩溃；此外方法在不同语言/领域的通用性尚未充分验证

---

## 552. EmoLLM: Appraisal-Grounded Cognitive-Emotional Co-Reasoning in Large Language Models

**arXiv ID:** 2603.16553 | [PDF](https://arxiv.org/pdf/2603.16553v1)

**作者:** Yifei Zhang `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 6997 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EmoLLM，基于情绪评估理论的 Appraisal Reasoning Graph (ARG) 进行 IQ–EQ 共理由推理，并通过反视角强化学习优化多轮对话

**💡 创新点**

①将情绪评估拆分为五个相互依赖的节点（事实、需求、评估维度、情绪状态、回应策略）；②使用 ARG 结构化中间推理；③采用反视角预测用户情绪转移作为奖励信号；④在多轮角色扮演环境下进行强化学习

**🔧 技术方法**

大语言模型（Qwen3‑8B）+持续预训练（CPT）+ARG 引导监督 + 反视角推理 + 轨迹级 GRPO 强化学习

**📊 数据集**

ESConv、EmpatheticDialogues、MSDialog、MedDialog、ICLR 评审数据；ECoK 情绪知识图谱；构造的多轮场景种子

**📈 对比分析**

与多种提示、SFT、RL 以及 4 种专有 LLM（gpt‑5‑nano/mini、gemini‑2.5‑flash、gemini‑3.1‑flash‑lite）比较；在四个基准上，EmoLLM 在成功率、情绪得分、情绪提升/轮、共情适宜性等指标上均优于基线，甚至接近或超过专有模型，且参数规模更小

**⚠️ 局限性**

依赖 LLM 评估器导致模型偏差；训练仅在模拟用户环境，未覆盖真实人类多样性；ARG 轨迹为任务导向中间工具，非完整可解释；在高风险情境下需谨慎部署

---

## 553. GenZ-LIO: Generalizable LiDAR-Inertial Odometry Beyond Indoor--Outdoor Boundaries

**arXiv ID:** 2603.16273 | [PDF](https://arxiv.org/pdf/2603.16273v1)

**作者:** Daehan Lee `[一作]` (Pohang University of Science and Technology), Soohee Han `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 4735 | [OpenAlex ID](https://openalex.org/A5069368669)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为GenZ-LIO的LiDAR‑IMU里程计框架，能在室内、室外及其过渡环境中自适应处理不同空间尺度的点云数据；

**💡 创新点**

创新点包括基于比例-微分控制器的尺度感知自适应体素化、通过尺度与误差敏感度调度的增益策略实现平稳点云压缩、以及融合点‑面与点‑点残差的混合度量状态更新，配合体素裁剪对应搜索来降低计算开销；

**🔧 技术方法**

主要技术包括PD反馈控制（无积分项）对体素大小调节、尺度指示符与误差导数的归一化增益调度、基于误差状态迭代卡尔曼滤波器的混合残差更新、以及根体素邻域与距离裁剪的对应搜索；

**📊 数据集**

使用九个公开基准（SubT‑MRS、SuperLoc、HILTI 2021/2022、GEODE、M3DGR、NTU‑VIRAL、ENWIDE、Oxford Spires）以及新收集的“NarrowWide”数据集，涵盖极端从狭窄室内到宽阔室外的多尺度环境；

**📈 对比分析**

与FAST‑LIO2、Faster‑LIO、AdaLIO、Point‑LIO、LIO‑EKF、DLIO、iG‑LIO、PV‑LIO等十余种SOTA LiDAR‑IMU里程计进行对比，GenZ‑LIO在所有测试序列上实现了零发散、平均绝对平移误差仅为0.75 m（仅次于去除发散样本的基线），并在CPU使用率与单帧处理时间上均优于其他自适应体素化方法；

**⚠️ 局限性**

局限性包括对尺度指示符与增益调度参数的依赖（需经验调优）、对点云稀疏区的处理仍受限于体素化导致的分辨率损失，以及对非LiDAR/IMU传感器（如视觉）扩展的进一步验证不足。

---

## 554. Bayesian-guided inverse design of hyperelastic microstructures: Application to stochastic metamaterials

**arXiv ID:** 2603.15917 | [PDF](https://arxiv.org/pdf/2603.15917v1)

**作者:** Hooman Danesh `[一作]` (Technische Universität Braunschweig), Henning Wessels `[通讯]` (Technische Universität Braunschweig)

**通讯引用:** 790 | [OpenAlex ID](https://openalex.org/A5078524162)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于贝叶斯指导的离散逆设计框架，用于在超弹性形变材料的大型候选库中快速定位满足目标应力响应的结构。

**💡 创新点**

创新点在于将统计特征工程、关联多输出高斯过程、基于不确定性的主动学习与贝叶斯筛选相结合，实现在仅极少数高保真计算下完成逆设计。

**🔧 技术方法**

使用了两点相关加PCA降维特征、线性协同化多输出GP、变分推断、信息增益驱动的主动学习、以及预测不确定性加权的候选排序方法。

**📊 数据集**

使用了约50,000个由随机场生成的二维超弹性超材料单元格（96×96像素二值图像）作为设计库，并在20个对齐拉伸状态下采集训练数据，在45°旋转加载下评估测试目标。

**📈 对比分析**

通过与全量oracle评估对比，框架在仅50次oracle调用内就能满足5%误差阈值的目标，hit率>96%，平均调用次数<10；R²≥0.98，主动学习仅需0.4%的样本即可训练出精确的GP模型。

**⚠️ 局限性**

局限性在于仅能搜索预定义的离散设计集合、依赖固定的超弹性模型假设，且在候选库覆盖不足或模型不充分时效率受限，未来可结合生成模型和更灵活的本构发现方法。

---

## 555. When AI Navigates the Fog of War

**arXiv ID:** 2603.16642 | [PDF](https://arxiv.org/pdf/2603.16642v1)

**作者:** Ming Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Tianyi Zhou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个以2026年中东冲突为背景的、基于时间节点的评估框架，评估大型语言模型在“战雾”下的推理与概率判断。

**💡 创新点**

创新点在于：①采用完全后训练截止点的真实冲突场景，避免训练泄漏；②将评估拆分为 11 个关键时间节点和 42 条可验证问题，提供动态、渐进式的推理轨迹；③结合定量校准指标与定性叙事分析，揭示模型在多主体政治经济情境下的推理差异。

**🔧 技术方法**

技术主要包括：自然语言提示工程（给模型提供截至时点的新闻文本与问题），手动提取概率输出，基于 MAE 的概率校准评估，以及对模型回应的主题编码与叙事演化归纳。

**📊 数据集**

数据集为从 12 大新闻来源（如 Al Jazeera、Reuters、BBC 等）抓取的 889k 字新闻文本，按 11 个时间节点聚合，配合 42 条事件问题与 5 条探索性问题，构成“时间节点上下文-问题”对。

**📈 对比分析**

比较方法为跨模型的概率校准（1‑MAE）和主题层面的平均校准，得到 0.63–0.75 的平均得分，说明 SOTA LLM 在此泄漏防护设置下仍能在结构化经济路径上表现相对稳定，但在政治意图与多主体交互层面表现波动。定性对比显示模型叙事从早期的快速遏制预期转为系统性衰退与脆弱停火。

**⚠️ 局限性**

局限性包括：①冲突仍在进行，评估结果受观察截止时间影响；②对概率输出的手工提取可能引入主观偏差；③仅覆盖中东冲突，难以推广到其他地区或更复杂的多主体场景；④模型可能仍受训练数据中已存在的相关信息影响，虽做了时间截断但泄漏风险不可完全消除；⑤评估未涵盖模型的行动决策能力，仅限文本推理。

---

## 556. Longitudinal Risk Prediction in Mammography with Privileged History Distillation

**arXiv ID:** 2603.15814 | [PDF](https://arxiv.org/pdf/2603.15814v1)

**作者:** Banafsheh Karimian `[一作]` (ETS Montreal), Eric Granger `[通讯]` (ETS Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了在缺失先前乳腺X线检查时，仅利用当前检查和重构的历史信息进行多年乳腺癌风险预测的Privileged History Distillation（PHD）方法。

**💡 创新点**

创新点在于将完整历史视为训练时的特权信息，利用多老师horizon‑specific蒸馏将长期风险信号注入仅使用当前检查的学生模型。

**🔧 技术方法**

技术包括Mirai图像编码器+Transformer/ Vision‑Mamba RNN序列编码，特征重建网络，logit蒸馏与加权交叉熵训练。

**📊 数据集**

使用了CSAW‑CC纵向乳腺X线数据集。

**📈 对比分析**

与LoMaR、VMRA等完整历史模型及单检查模型在1–5年AUC、pAUC进行对比，PHD在无历史下达到或接近全历史模型性能，尤其在低FPR区和5年预测上显著提升。

**⚠️ 局限性**

局限性在于仍需训练时完整历史的可用性，特征重建在极少历史情况下可能不稳定，且未在多机构跨设备场景进一步验证。

---

## 557. SuCor: Susceptibility Distortion Correction via Parameter-Free and Self-Regularized Optimal Transport

**arXiv ID:** 2603.16758 | [PDF](https://arxiv.org/pdf/2603.16758v1)

**作者:** Sreekar Chigurupati `[一作]` (Indiana University), Eleftherios Garyfallidis `[通讯]` (Indiana University)

**通讯引用:** 6302 | [OpenAlex ID](https://openalex.org/A5083595381)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种基于一维最优运输的快速无参磁敏感失真校正方法 SuCor，能够在无需手动调参的情况下快速校正 EPI 图像的几何畸变。

**💡 创新点**

创新点在于将每个 PE 列视为一维最优运输问题，利用闭式量化映射获取精确位移场，并通过 Morozov 差异原理自动选择光滑正则化强度，实现参数无关、计算高效的校正。

**🔧 技术方法**

采用的一维 Wasserstein‑2 最优运输计算、频域薄板弯曲能量正则化以及基于残差差异原理的自动 λ 选择等技术。

**📊 数据集**

在 Human Connectome Project (HCP) 1200 受试者数据集的 LR–RL 方向的 b=0 EPI 对以及对应的 T1 结构图像上进行评估。

**📈 对比分析**

与 FSL TOPUP 对比，SuCor 在 T1 互信息（MI）上提高至 0.341（相较 0.317），运行时间仅约 12 秒（TOPUP 约 55 分钟），但 LR–RL 的相似度指标（NCC、RMSE）略逊于 TOPUP。

**⚠️ 局限性**

局部列向不一致导致的 LR–RL 余差较大；强正则化可降低该余差但会损失 MI，表现出平滑性与校正精度之间的权衡限制。

---

## 558. GATS: Gaussian Aware Temporal Scaling Transformer for Invariant 4D Spatio-Temporal Point Cloud Representation

**arXiv ID:** 2603.16154 | [PDF](https://arxiv.org/pdf/2603.16154v1)

**作者:** Jiayi Tian `[一作]` (Xi'an Jiaotong University), Jiaze Wang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5005316643)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种双重不变的 4D 点云视频理解框架 GATS，解决了分布不确定性与时间尺度偏差两大问题。

**💡 创新点**

创新点在于引入 Uncertainty Guided Gaussian Convolution（UGGC）以利用局部高斯统计和不确定性门控实现对密度、噪声、遮挡的鲁棒聚合，以及 Temporal Scaling Attention（TSA）通过可学习的时间缩放因子实现帧率无关的相对速度估计。

**🔧 技术方法**

主要技术包括高斯加权卷积与多尺度高斯核、条件数驱动的不确定性门控、基于时间缩放的注意力机制，以及 Transformer 级联实现高效全局建模。

**📊 数据集**

在 MSR-Action3D、NTU RGB-D（动作识别）和 Synthia4D（语义分割）三大公开基准上进行评估。

**📈 对比分析**

与 Transformer、CNN、SSM 等主流方法对比，GATS 在 MSR-Action3D 24 帧下取得 97.56% 准确率，NTU RGB-D 达 91.7% 识别率，Synthia4D 多帧 mIoU 达 84.21%，显著优于现有最佳模型。

**⚠️ 局限性**

局限性包括：对极端帧率变化或高噪声场景的泛化能力尚待进一步验证；UGGC 的高斯统计估计在点云稀疏区可能不稳定；以及整体模型仍保持 Transformer 复杂度，需在大规模数据上进一步压缩。

---

## 559. Leveling3D: Leveling Up 3D Reconstruction with Feed-Forward 3D Gaussian Splatting and Geometry-Aware Generation

**arXiv ID:** 2603.16211 | [PDF](https://arxiv.org/pdf/2603.16211v1)

**作者:** Yiming Huang `[一作]` (Chinese University of Hong Kong), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 17431 | [OpenAlex ID](https://openalex.org/A5032340829)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

结合feed-forward 3D Gaussian Splatting与几何感知的扩散模型，利用Leveling Adapter对稀疏输入进行视角外推细化，并将改进的视图回馈提升3D重建质量。

**💡 创新点**

首次将3D几何先验通过跨注意力融入扩散控制；提出调色板过滤策略保证训练稳定；在测试时采用形态学掩码细化，最终实现一次性重建提升的统一框架。

**🔧 技术方法**

使用AnySplat、Stable Diffusion 2轻量级Adapter、几何跨注意力、Palette filtering、形态学掩码细化，以及LPIPS/PSNR/SSIM/Met3R等评价指标。

**📊 数据集**

训练集：DL3DV（10,510 场景）+ ScanNet++（1,006 场景）；评估集：MipNeRF360、VRNeRF、TartanAir、ScanNet。

**📈 对比分析**

与Diffix3D+、GSFix3D、GSFixer、ViewExtrapolator等基线在NVS和深度估计任务上对比，PSNR提升约6–15%，AbsRel降低约9–34%，Met3R下降到0.0376/0.0614，且推理速度与图像扩散方法相近。

**⚠️ 局限性**

对极端稀疏输入仍有限制，掩码细化需手工调参；当几何先验不可靠时仍可能产生误差；模型需要额外GPU资源；未充分探讨多帧视频连续一致性。

---

## 560. Efficient Reasoning on the Edge

**arXiv ID:** 2603.16867 | [PDF](https://arxiv.org/pdf/2603.16867v1)

**作者:** Yelysei Bondarenko `[一作]`, Babak Ehteshami Bejnordi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一整套从 LoRA 适配、动态 Switcher 路由、预算强制强化学习、并行推理与轻量化验证器，到 4‑bit 量化，最终实现高效推理的端到端边缘设备 LLM 框架。

**💡 创新点**

创新点包括：① 对预填充阶段进行掩蔽 LoRA 训练实现 KV‑cache 共享；② 引入轻量 Switcher 在推理时动态选择是否激活 LoRA；③ 设计软阈值奖励的预算强制 RL，显著压缩推理长度；④ 在并行推理中使用轻量化验证器做加权投票；⑤ 将量化融入训练，保持性能；⑥ 将上述组件统一到边缘硬件上。

**🔧 技术方法**

技术涵盖 LoRA、Masked LoRA、GRPO 强化学习、软阈值预算奖励、加权投票、FastForward/GENIE SDK、4‑bit 量化、并行采样与轻量化验证头。

**📊 数据集**

主要数据集为 Mixture of Thoughts、OpenThoughts3、DeepSeek‑R1、QwQ‑32B、MATH、GPQA、LiveCodeBench、HumanEval、MBPP、AIME、AMC 等多领域推理与编程任务。

**📈 对比分析**

与全参数稠密微调、DeepSeek‑R1‑Distill 等基线相比，LoRA 128 能在 3B/7B 模型上逼近或匹配全参数性能；预算强制后平均推理长度缩短 2.4×，准确率仅微降；加权投票提升 10% 的 MATH 500 正确率；在量化后与 FP32 结果差距不足 2%。

**⚠️ 局限性**

局限性包括：小尺寸 backbone 对 LoRA rank 敏感、适配导致的遗忘折衷、Switcher 仍是监督式二分类且未使用 RL 优化、预算强制假设每个 token 成本均匀、仅支持单一 LoRA 适配器、并行推理仍受限于设备内存与延迟。

---

## 561. Prior-Informed Neural Network Initialization: A Spectral Approach for Function Parameterizing Architectures

**arXiv ID:** 2603.16376 | [PDF](https://arxiv.org/pdf/2603.16376v1)

**作者:** David Orlando Salazar Torres `[一作]` (South Westphalia University of Applied Sciences), Andreas Schwung `[通讯]` (South Westphalia University of Applied Sciences)

**通讯引用:** 1246 | [OpenAlex ID](https://openalex.org/A5025397538)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于先验信息的Bag-of-Functions网络初始化与结构配置框架，利用FFT提取季节性先验、残差回归估计趋势先验，并将这些统计信息直接嵌入权重初始化和模型深度、编码器维度的决定中。

**💡 创新点**

创新点在于把频谱分析与趋势估计融入到网络初始化与架构设计的闭环过程，首次给出有限样本趋势回归的样本量保证，并通过理论与实验验证该方法可显著提升收敛速度与解码精度。

**🔧 技术方法**

采用FFT频谱提取、残差基函数分解、最小二乘线性回归、子高斯集中界、Bag-of-Functions编码解码、Adam优化和MSE损失等技术。

**📊 数据集**

使用了三类数据集：合成时序（已知频率、趋势、事件）、PJM电网负荷时序、热电厂发电量时序，以及四个公开生成模型基准（Sine、Stocks、Air Quality、PU）。

**📈 对比分析**

通过在10次独立实验中与传统Xavier/He初始化、经验式Heuristic BoF以及其他生成模型（GAN/ VAE）进行对比，实验显示改进初始化可使MSE降低约97%、收敛加快、参数漂移减小，并在生成模型基准上显著降低判别与预测误差。

**⚠️ 局限性**

局限性包括对阈值τ和误差容差δ的手工设定、对非周期或高度非线性趋势的提取可能不足、在极端噪声或短序列场景下的稳定性未完全验证，以及未深入研究在线自适应更新机制。

---

## 562. SIA: A Synthesize-Inject-Align Framework for Knowledge-Grounded and Secure E-commerce Search LLMs with Industrial Deployment

**arXiv ID:** 2603.16137 | [PDF](https://arxiv.org/pdf/2603.16137v1)

**作者:** Zhouwei Zhai `[一作]` (JD.com), Anmeng Zhang `[通讯]` (JD.com)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在电子商务搜索领域构建了SIA（合成‑注入‑对齐）框架，系统地解决了知识幻觉与安全合规问题。

**💡 创新点**

创新点包括基于知识图谱与行为日志的语义合成数据、深度上扩张（Depth Up‑Scaling）知识注入、以及双路径多任务指令与对抗训练的细粒度对齐策略。

**🔧 技术方法**

核心技术包括LLM预训练与参数高效扩展、链式思维数据生成、指令微调、红队对抗训练与安全对齐。

**📊 数据集**

使用了京东内部商品知识图谱、用户行为日志、合成安全问答、公开指令集（Alpaca、Dolly、ShareGPT）以及标准基准（C‑Eval、MMLU、GSM8K、JDSec等）。

**📈 对比分析**

与基线Qwen2.5‑14B‑Instruct、DeepSeek‑V3及GPT‑4o对比，SIA‑15B在电商任务ROUGE‑L、准确率提升≈30%、安全ASR降低≈20%，在线A/B实验显著提升CTR/CVR/F1等业务指标。

**⚠️ 局限性**

局限性在于需要频繁的增量更新、缺乏多模态支持以及对极端低频或跨境场景的安全评估仍不足。

---

## 563. Reconciling distributed compliance with high-performance control in continuum soft robotics

**arXiv ID:** 2603.16630 | [PDF](https://arxiv.org/pdf/2603.16630v1)

**作者:** Vito Daniele Perfetta `[一作]` (Delft University of Technology), Cosimo Della Santina `[通讯]` (Delft University of Technology)

**通讯引用:** 4206 | [OpenAlex ID](https://openalex.org/A5050239145)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了一款完全柔性的连续软机械臂，利用双重肌腱路径实现耦合弯曲与扭转，并通过两级闭环控制实现高速精准的任务空间调节。

**💡 创新点**

将可变应变模型与摩擦感知、欠驱动动力学协同以及针对欠驱动连续系统的闭环逆运动学结合，构成了完整可解释的控制架构，实现了在全软体上四倍加速的任务空间精度。

**🔧 技术方法**

采用可变应变Cosserat杆模型、摩擦改进的变量应变动力学、双层协同控制（内部协作PID + 外部闭环逆运动学），以及运动捕捉反馈。

**📊 数据集**

未使用公开数据集，所有结果均基于实验室自制的柔性臂、摆锤、糖杯等真实环境测试。

**📈 对比分析**

与现有分段软机器人、非分段弯曲及通用连续软体进行对比；在毫米级误差下任务速度提升近四倍，瞬时端点速度高达4 m/s，展示了高带宽闭环控制的可行性。

**⚠️ 局限性**

仅限于配置调节；对运动轨迹跟踪缺乏验证；实验依赖运动捕捉；未显式建模外部环境相互作用，需进一步提升鲁棒性。

---

## 564. Synergizing Deep Learning and Biological Heuristics for Extreme Long-Tail White Blood Cell Classification

**arXiv ID:** 2603.16249 | [PDF](https://arxiv.org/pdf/2603.16249v1)

**作者:** Trong-Duc Nguyen `[一作]`, Huy-Hieu Pham `[通讯]` (VinUniversity)

**通讯引用:** 1170 | [OpenAlex ID](https://openalex.org/A5065112274)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了融合生成修复、Swin Transformer和MedSigLIP对比学习以及形态学过滤的三阶段混合框架以实现白细胞罕见类的泛化

**💡 创新点**

创新点在于将Pix2Pix域恢复与对比学习嵌入结合，以及基于生物学先验的几何尖锐度与马氏距离形态约束来提升罕见类识别

**🔧 技术方法**

使用Pix2Pix GAN、Swin Transformer、MedSigLIP、Focal Loss、TTA、形态学特征提取及马氏距离过滤等技术

**📊 数据集**

使用ISBI 2026 WBCBench 2026数据集（55,012张血液涂片图像，13类长尾分布）

**📈 对比分析**

相较于传统长尾学习方法（如LDAM、Decoupling）以及单一Swin-T模型，最终在私有排行榜上达Macro-F1 0.77139，显著提升

**⚠️ 局限性**

局限性包括形态学过滤仅针对主要混淆对，难以推广至所有少数类；依赖合成对齐数据；对跨域变化的鲁棒性仍待提升

---

## 565. Is Semi-Automatic Transcription Useful in Corpus Creation? Preliminary Considerations on the KIParla Corpus

**arXiv ID:** 2603.16258 | [PDF](https://arxiv.org/pdf/2603.16258v1)

**作者:** Martina Simonotti `[一作]`, Caterina Mauri `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

评估将自动语音识别（ASR）集成到 KIParla 语料库转录流程中，比较手工转录与 ASR 辅助转录在转录速度和质量上的差异；

**💡 创新点**

创新点在于直接在同一音频段上对比两种转录流程，结合多维度质量指标（Delta、WER、重叠错误率）和 Generalized Additive Mixed Models（GAMM）统计建模，系统评估转录者专业度、对话类型及流程对转录结果的影响；

**🔧 技术方法**

使用 Whisper 大型 ASR 模型、Elan 转录软件、Needleman‑Wunsch 对齐算法、Token 化与 Jefferson 符号转换、GAMM 统计模型等技术；

**📊 数据集**

使用 KIParla 语料库中的 228 小时意大利语对话数据，选取七个 10 分钟抽样（自由对话、半结构化访谈、L2 访谈），并提供相应 Gold 标准；

**📈 对比分析**

比较方法包括：Delta 统计（与 Gold 的 token/TU 差异）、WER 计算、重叠错误比例、以及统计模型评估转录速度和结构差异；结果显示 ASR 辅助显著提升转录速度，尤其对非专业转录者显著降低 WER，专家效果更为多变；ASR 辅助产生的 TU 更少且平均长度更长；

**⚠️ 局限性**

局限性包括：样本量小、对话类型不平衡、未在同一类型对话中同时测试两种流程、专家/非专家划分过于粗糙、未深入收集转录者的定性反馈，导致结论仍为探索性，需进一步扩展数据集和改进实验设计。

---

## 566. Reevaluating the Intra-Modal Misalignment Hypothesis in CLIP

**arXiv ID:** 2603.16100 | [PDF](https://arxiv.org/pdf/2603.16100v1)

**作者:** Jonas Herzog `[一作]` (Zhejiang University), Yue Wang `[通讯]` (Zhejiang University)

**通讯引用:** 54952 | [OpenAlex ID](https://openalex.org/A5113600509)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

重新评估 CLIP 等视觉‑语言模型的 intra‑modal misalignment 假设，提出理论证明 intra‑modal 相似度由跨模态相似度决定，且无自由度；通过实验证明常用的 misalignment 指标（余弦相似度直方图、模态差距等）并不能可靠地检测 intra‑modal misalignment；进一步提出基于 PCA 的投影方法（PCA^←），在 few‑shot 识别和图像‑图像检索上显著提升性能，表明任务模糊性而非嵌入不对齐是主要瓶颈。

**💡 创新点**

1) 从理论层面驳斥了先前关于 embedding 自由度导致 misalignment 的观点；2) 系统性验证了 misalignment 指标的无效性；3) 提出一种简单且效果优于现有 OTI 等方法的投影方案；4) 综合多种模型（CLIP、SigLIP、SigLIP2、DINO 等）和多任务评估，提供对比实验。

**🔧 技术方法**

理论推导（矩阵分析、线性系统解），余弦相似度直方图、模态差距分析，PCA 投影（PCA^←），基于预训练 CLIP 视觉编码器的特征提取，few‑shot 评估（线性判别、特征选择等），图像‑图像检索评估（mAP）。

**📊 数据集**

11 个图像分类数据集（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、EuroSAT、UCF101、DTD、SUN397），检索基准（ROxford、RParis、BDD100k）以及 CLIP 预训练数据集。

**📈 对比分析**

方法上将仅跨模态对齐的模型（CLIP、SigLIP）与同时包含 intra‑modal 对齐的模型（SigLIP2、DINO）进行对照；在 few‑shot 识别中比较 Tip‑X、Coder、APE、LDA 等方法；在检索任务中比较 CLIP、SigLIP、SigLIP2、DINO、OTI 等；实验表明：PCA^← 在 most 任务上均优于 OTI，且 CLIP 与带有 intra‑modal 训练的模型差距不大，说明 misalignment 并非主要瓶颈。

**⚠️ 局限性**

仅在检索和 few‑shot 识别上验证，未覆盖分割、深度估计、VQA 等其他视觉任务；PCA^← 仅在标签与图像主语义对齐时有效，对非主语义任务可能失效；实验未能完全证明 misalignment 在所有场景下不存在，只能说明它不是导致性能差异的主要原因。

---

## 567. Beyond Reward Suppression: Reshaping Steganographic Communication Protocols in MARL via Dynamic Representational Circuit Breaking

**arXiv ID:** 2603.15655 | [PDF](https://arxiv.org/pdf/2603.15655v1)

**作者:** Liu Hung Ming `[一作]` `[通讯]` (PARRAWA AI), Liu Hung Ming (PARRAWA AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究分散式多智能体强化学习中隐蔽共谋威胁，并提出 Dynamic Representational Circuit Breaker (DRCB) 框架来监测和抑制此类隐蔽通信。

**💡 创新点**

创新点在于把通信从语义层迁移到优化层：使用 VQ‑VAE 离散符号、Jensen‑Shannon Divergence、代码库漂移以及随机观测池组成的多层监测与结构破坏式干预，并在检测到共谋时动态调整阈值、注入梯度惩罚、严重奖励抑制以及代码库洗牌+Adam 重置。

**🔧 技术方法**

采用 AIM 框架、VQ‑VAE 离散化、EMA 合成 Collusion Score、随机观测池（MTD）、梯度空间惩罚、严重奖励抑制、代码库洗牌和优化器状态重置等技术。

**📊 数据集**

使用 MNIST 作为上下文标签的条件 Prisoner’s Dilemma 环境来验证。

**📈 对比分析**

通过与无防御基线和固定阈值监测的静态方案进行 Welch t‑test 比较，结果显示 DRCB 在保持平均联合奖励无显著下降的同时显著降低深度欺骗事件、提高观察者准确率 (+9.3%) 并抑制波动性；极端惩罚参数可能导致系统崩溃。

**⚠️ 局限性**

局限性：仅适用于可离散化的通信任务，无法完全消除低频长尾隐蔽信道；动态阈值对超参数高度敏感，结构破坏会导致短暂功能真空，整体计算开销提升约 15‑22%。

---

## 568. Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights

**arXiv ID:** 2603.16817 | [PDF](https://arxiv.org/pdf/2603.16817v1)

**作者:** Yi Chen `[一作]` (University of Wisconsin-Madison), Ramya Korlakai Vinayak `[通讯]` (University of Wisconsin-Madison)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了在检索增强生成（RAG）基础上加入的 conformal factuality 过滤框架，探讨其在事实性和信息性之间的权衡，并提出了新的评价指标。

**💡 创新点**

创新点在于提出非空率、非空真值率、充分正确性和条件充分正确性等指标，揭示传统真值率易受空白答案影响；同时发现轻量化蕴涵推理器可在显著降低 FLOPs 的情况下优于大规模 LLM‑based 置信度评分器。

**🔧 技术方法**

采用了 conformal prediction 过滤流程、文档/句子级蕴涵模型、LLM‑based 置信度评分（包含多种提示设计）、以及 MoE 与 dense 语言模型的并行推理。

**📊 数据集**

实验覆盖了 FActScore（开放式摘要）、MATH（数学推理）和 Natural Questions（知识问答）三大数据集，并在多种开源 LLM 家族（Qwen3、Llama‑3.x、SmolLM2、gpt‑oss 等）上评测。

**📈 对比分析**

与传统 Empirical Factuality、Power、FPR 等指标相比，新指标能更好捕捉答案的实用性；实验显示轻量化蕴涵推理器在 100× FLOPs 以内即可匹配或超越 117B LLM 的表现，且在高事实性阈值下仍保持较高非空率。

**⚠️ 局限性**

主要局限在于 conformal filtering 对分布偏移和诱导假设（distractor）不具鲁棒性；此外，若校准集与测试集不完全可交换，事实性保证可能被破坏，导致实用性下降。

---

## 569. Prompt Engineering for Scale Development in Generative Psychometrics

**arXiv ID:** 2603.15909 | [PDF](https://arxiv.org/pdf/2603.15909v1)

**作者:** Lara Lee Russell-Lasalandra `[一作]`, Hudson Golino `[通讯]` (University of Virginia)

**通讯引用:** 4909 | [OpenAlex ID](https://openalex.org/A5043910258)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在AI‑GENIE框架下进行蒙特卡罗仿真，系统评估不同prompt工程策略对LLM生成的个性测评题项的质量影响。

**💡 创新点**

首次量化adaptive prompting在生成心理测量题项中的优势，并揭示其与模型规模、温度的交互作用。

**🔧 技术方法**

采用GPT‑4o、GPT‑5.1、GPT‑OSS‑20B、GPT‑OSS‑120B等LLMs生成题项，并用AI‑GENIE的网络心理测量方法（UVA、bootEGA、EGA）评估与裁剪。

**📊 数据集**

使用基于Big Five（OCEAN）人格特质属性的虚拟题项池作为生成输入，没有人工标注，仅靠模拟生成。

**📈 对比分析**

通过比较不同prompt（Basic、Expanded、Few‑Shot、Persona、Persona+FS、Adaptive）在各模型/温度下的NMI、冗余率、保留项数等指标，发现adaptive prompting在最强模型GPT‑5.1上实现近乎完美的NMI（≈98%）且保留更多题项。

**⚠️ 局限性**

局限性包括仅检验Big Five人格、缺乏人工专家评审、仅使用特定模型版本且模型更新可能影响结果、对低频/少量资料构念的验证不足。

---

## 570. From Workflow Automation to Capability Closure: A Formal Framework for Safe and Revenue-Aware Customer Service AI

**arXiv ID:** 2603.15978 | [PDF](https://arxiv.org/pdf/2603.15978v1)

**作者:** Cosimo Spera `[一作]` (Minerva CQ), Riccardo De Maria `[通讯]` (Minerva CQ)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文将能力超图框架应用到多智能体客户服务自动化，构建了一个包含12种能力的电信案例，并给出成本与收入模型，证明该框架在安全性与商业价值上均可实现显著提升。

**💡 创新点**

创新点在于：①首次将非组合安全理论与客户服务结合，揭示单独安全的智能体组合可能触发禁止能力的“合成风险”；②提出安全-价值双重性证明，即安全认证与商业目标发现是同一计算；③给出完整的四事件（加入、离开、获取、丢失）安全检查与闭包更新规则；④提出五条面向下一代系统的正式架构原则。

**🔧 技术方法**

主要技术包括：能力超图（hypergraph）与闭包运算、非组合安全定理、安全-价值双重性、增量维护定理、对抗性鲁棒性（对提示注入的单边缘检查）、PAC学习算法用于从日志中推断超图结构。

**📊 数据集**

使用的主要数据集是基于公开电信行业基准的模拟交易（1000万笔年交易）以及12种能力的人工构造超图；PAC学习实验以约6600条会话日志验证超图结构学习误差 ≤5%。

**📈 对比分析**

相较于传统工作流/多代理平台，本文在电信案例中实现了：AI 预防率从70%提升至80%（节省3.8-7.6 M美元运营成本），收入提升18.9-20.35 M美元，净年价值约20.7 M美元，投资回收期仅约11周；在安全检测上，闭包检查在O(24)操作内完成，实时可用。

**⚠️ 局限性**

局限性包括：假设能力持久且无时效限制，难以处理资源/时序约束；超图结构需足量轨迹数据，稀疏组合可能需主动探测；未覆盖人机混合会话；计算最小不安全反链仍为coNP-hard，实际应用需对稀疏结构进一步优化；模型未与LLM函数调用图直接对应。

---

## 571. An Exponential Separation between Deterministic CDCL and DPLL Solvers

**arXiv ID:** 2603.16156 | [PDF](https://arxiv.org/pdf/2603.16156v1)

**作者:** Sahil Samar `[一作]` (Georgia Institute of Technology), Vijay Ganesh `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8232 | [OpenAlex ID](https://openalex.org/A5052292970)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明存在一种完全确定性的 CDCL SAT 求解器配置，在使用 VSIDS（衰减因子 ≤ 1/2）、固定相位、每冲突重启、无删Clause、1UIP 学习的前提下，能够在多项式时间内解决 Ordering Principle（OP）CNF 公式。

**💡 创新点**

首创性地展示了一个确定性 CDCL 配置能在 OP 公式上实现多项式时间求解，并通过引入 Focus Lemma、Equal Score Invariant 等新颖的 VSIDS 性质证明了与树形解析（DPLL）之间的指数分离；同时首次给出了完整的学习序列与冲突分析证明。

**🔧 技术方法**

利用 VSIDS 的得分更新规则、衰减因子≤1/2 等价于 VMTF 的性质，构造固定相位、频繁重启的求解流程；通过对冲突学习的蕴含图进行 1UIP 切分，证明了 Head、Descending Cascade 与 Tail 三个阶段的学习顺序，并用这些阶段推导出 UNSAT。

**📊 数据集**

仅使用 Ordering Principle (OP) 这一经典 CNF 公式集，变量数为 n，作为实验/理论验证的实例。

**📈 对比分析**

与树形解析（DPLL）做性能对比。证明 CDCL 在 OP 公式上只需 n²–3 次冲突（多项式），而已知树形解析对同类公式的证明长度具有指数下界，从而实现了指数性能分离。

**⚠️ 局限性**

局限性：证明依赖于特定的 tie‑breaking 规则、衰减因子、固定相位以及无删Clause 的严格配置；该配置不一定适用于所有实际 SAT 语料或其他启发式；此外，结果仅在 OP 公式上证明，未证明其对更广泛问题集的通用性。

---

## 572. TCATSeg: A Tooth Center-Wise Attention Network for 3D Dental Model Semantic Segmentation

**arXiv ID:** 2603.16620 | [PDF](https://arxiv.org/pdf/2603.16620v1)

**作者:** Qiang He `[一作]` (Institute of Software), Hongan Wang `[通讯]` (Beijing Stomatological Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文针对3D牙齿模型的语义分割任务，提出了TCATSeg网络，旨在提升牙齿分割的准确性与跨域泛化能力。

**💡 创新点**

创新点：① 使用与牙齿中心对齐的物理有意义稀疏超点（TCP）作为全局语义上下文引导；② 通过双重注意力机制（DPDA+SGDA）将局部几何特征与全局语义信息融合；③ 采用交叉层一致性约束与匹配监督提升超点的语义一致性。

**🔧 技术方法**

核心技术：Teeth Center‑Wise 编码器、Channel‑Wise Attention、全局聚合注意力、层注意力、双重注意力融合、匈牙利匹配、Chamfer距离与 Smooth L1 损失，后处理细化步骤。

**📊 数据集**

数据集：公开 Teeth3DS 数据集；自建 TeethWild（400例正畸患者）覆盖异常牙弓、拥挤、缺牙等多样化场景。

**📈 对比分析**

方法对比：在 Teeth3DS 与 TeethWild 上与 PointNet++, DGCNN, PT, SpoTr, DBGANet 等经典与专用基线进行实验，TCATSeg 在 OA、DSC、TIR 等指标上均位居榜首；在 3DTeethSeg'22 挑战中获得最高分数；零样本跨域评估亦显示出最佳泛化性能。

**⚠️ 局限性**

局限性：超点初始化与位置约束对训练稳定性有影响，极端缺牙或极度异常形状仍可能导致分割误差；模型整体复杂度和推理时间相对较高。

---

## 573. Offline Exploration-Aware Fine-Tuning for Long-Chain Mathematical Reasoning

**arXiv ID:** 2603.16206 | [PDF](https://arxiv.org/pdf/2603.16206v1)

**作者:** Yongyu Mu `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**通讯引用:** 11855 | [OpenAlex ID](https://openalex.org/A5100600701)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种 Offline eXploration-Aware (OXA) 细调方法，用于在 RLVR 之前通过低置信度教师数据的最大似然和高置信度错误样本的负似然损失来提升大型语言模型的数学推理性能并保持较高的策略熵。

**💡 创新点**

创新点在于在 SFT 阶段主动引导模型探索更多推理路径：①通过 Gaussian-guided PPL 采样挑选低置信度正确样本来扩展模型的推理空间；②通过 unlikelihood 损失消除高置信度错误样本，重新分配概率质量，避免熵塌陷。

**🔧 技术方法**

技术手段包括：基于最大似然的 MLE 训练、基于 unlikelihood 的负似然训练、PPL 计算与 Gaussian-guided 采样、RLVR 的可验证奖励机制，以及与 Clip-Cov 等熵控制方法的组合。

**📊 数据集**

使用的数据集主要有：2.6 M 的 AceReason-1.1-SFT（含 2 M 已验证推理样本），DeepScaleR-40K 用于 RLVR 训练，以及六个数学基准（AIME24/25、BRUMO25、CMIMC25、HMMT25、Minerva）。

**📈 对比分析**

与传统 SFT、低 PPL SFT、以及 OXA 的 MLE 仅版和完整版（含 UL）进行对比。实验显示 OXA 在 1.5B 模型上平均提升 +6 Pass@1、+5 Pass@k，且在 RLVR 训练后仍保持优势；在 7B 模型和其他 LLaMA3.2、LLaMA3.2-70B 上亦实现显著性能提升；与 Clip-Cov 组合时进一步提升，验证方法的互补性。

**⚠️ 局限性**

局限性包括：①OXA_Full 需要额外的自蒸馏推理样本采样，计算成本高于普通 SFT；②实验仅覆盖 1.5B‑7B 规模模型，尚未验证更大模型的效果，未来需扩展至更大参数规模。

---

## 574. NextMem: Towards Latent Factual Memory for LLM-based Agents

**arXiv ID:** 2603.15634 | [PDF](https://arxiv.org/pdf/2603.15634v1)

**作者:** Zeyu Zhang `[一作]` (Renmin University of China), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 61114 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了NextMem框架，利用自回归Transformer自编码器将长文本转换为短潜在表示，并可精准重构回原文本，实现高效的事实记忆存储与利用。

**💡 创新点**

创新点包括：① 两阶段训练（自回归重构对齐 + 进阶潜在替换）实现文本↔潜在↔文本的无损映射；② 共享权重的LoRA实现编码器与解码器的轻量切换；③ NF4量化在保持高重构精度的同时显著降低存储开销；④ 将潜在记忆直接作为检索索引，统一存储、检索与推理三大功能。

**🔧 技术方法**

使用的技术主要有：自回归Transformer自编码器、LoRA适配器、NF4正态浮点量化、进阶潜在替换训练、对齐训练、停止梯度、正则化等。

**📊 数据集**

评估数据集包括SQuAD、HotpotQA、RACE、LoCoMo、LongMemEval以及1c5[2]*等多种问答与对话场景。

**📈 对比分析**

与DeepSeek-OCR、ICAE、DyPRAG、BGE等基线对比：在事实重构任务中NextMem-Dense/NextMem-Sparse均取得最高F1/ROUGE/METEOR/BERTScore；在解压后上下文生成任务中优于所有非Oracle方法；在密集检索任务中在Hit@5、Recall@5等指标上均显著超过ICAE与DyPRAG，接近BGE的效果。量化版仅损失约0.1%重构精度，同时存储压缩率提升≈70%。

**⚠️ 局限性**

局限性：① 进阶潜在替换训练复杂，训练时间和调参成本高；② 量化后对极短或极长文本的重构仍可能产生幻觉；③ 在直接推理（无解码）场景下表现略逊于专门针对推理优化的模型（如ICAE）；④ 需要额外解码步骤才能利用存储内容，增加推理延迟；⑤ 对特定领域事实的细粒度记忆更新和编辑仍有待进一步研究。

---

## 575. Characterizing Delusional Spirals through Human-LLM Chat Logs

**arXiv ID:** 2603.16567 | [PDF](https://arxiv.org/pdf/2603.16567v1)

**作者:** Jared Moore `[一作]` (Stanford University), Desmond C. Ong `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

收集并分析19位自报因与LLM聊天机器人互动而出现心理伤害的用户聊天日志（共391,562条消息），构建了28个针对用户和机器人消息的编码体系，并使用LLM自动标注与人工验证相结合，对聊天行为进行定量描述，探究情绪支持、恋爱/友谊倾向、机器人自我意识误解及危机处理等特征与对话长度和用户心理危机的关联。

**💡 创新点**

①首次系统性、实证性研究真实用户与LLM的长时对话；②提出可复制、公开的28项编码清单和相应LLM标注工具；③通过定量回归揭示机器人爱慕、情感赞美、送情感承认等行为与对话延长和心理危机发生率的关联，提供针对风险行为的政策与技术建议。

**🔧 技术方法**

使用大型语言模型（如GPT‑4）进行自动标注，结合人工评注验证；利用回归分析、共现矩阵等统计方法对编码频率和行为序列进行定量分析；并通过公开的GitHub仓库分享工具与数据。

**📊 数据集**

19份自愿提交的聊天日志，涵盖约391,562条消息、4,761个对话；数据来自支持小组、媒体关注案例及Human Line项目，涵盖多种LLM模型（ChatGPT、Claude等）与多样化用户（年龄30-55岁，男女各占1/4）。

**📈 对比分析**

通过LLM与人工标注交叉验证获得约77.9%的一致率；对比分析中，代码共现频次、对话长度回归、危机回应比例等均表现出显著统计差异；但缺乏与正常对话基线或其他AI系统的直接比较，性能评估主要集中在标注准确率与行为关联性。

**⚠️ 局限性**

样本量小、仅限自报心理伤害用户，缺乏代表性；LLM标注与人工标注在不同代码上一致性差异大（如机器人自我能力误表仅0.08）；数据仅为日志，缺乏完整的心理健康评估或正式诊断；研究仅揭示相关性，未证明因果关系，且对极端事件的分析可能存在伦理与安全风险。

---

## 576. DST-Net: A Dual-Stream Transformer with Illumination-Independent Feature Guidance and Multi-Scale Spatial Convolution for Low-Light Image Enhancement

**arXiv ID:** 2603.16482 | [PDF](https://arxiv.org/pdf/2603.16482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 577. Unlearning for One-Step Generative Models via Unbalanced Optimal Transport

**arXiv ID:** 2603.16489 | [PDF](https://arxiv.org/pdf/2603.16489v1)

**作者:** Hyundo Choi `[一作]` (Sungkyunkwan University), Jaewoong Choi `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1640 | [OpenAlex ID](https://openalex.org/A5101792384)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对一阶生成模型的机器遗忘框架UOT-Unlearn。

**💡 创新点**

将概念遗忘建模为无偏最优传输（UOT）问题，并通过成本设计实现概率质量平滑重分布。

**🔧 技术方法**

使用无偏最优传输理论、半对偶UOT目标、神经网络参数化的运输映射及特征锚点。

**📊 数据集**

在CIFAR-10、ImageNet-256等公开图像数据集上进行实验。

**📈 对比分析**

与梯度上升、选择性健忘、显著性遗忘、变分扩散遗忘等方法对比，PUL最高、u-FID最小，表现优异。

**⚠️ 局限性**

缺点是需要预先计算遗忘类锚点，对边缘类的重分布依赖于超参数，且对大规模类别结构的理论稳定性尚未深入。

---

## 578. Social Simulacra in the Wild: AI Agent Communities on Moltbook

**arXiv ID:** 2603.16128 | [PDF](https://arxiv.org/pdf/2603.16128v1)

**作者:** Agam Goyal `[一作]` (University of Illinois), Koustuv Saha `[通讯]` (University of Illinois)

**通讯引用:** 2745 | [OpenAlex ID](https://openalex.org/A5057029055)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对 Moltbook 及 Reddit 5 个匹配社区的 73,899 与 189,838 条帖子进行大规模实证比较，揭示 AI 代理社区在参与分布、跨社区迁移与语言特征上的差异，评估社区与作者层面的同质化与可辨识度；

**💡 创新点**

创新点在于首次系统性比较 AI 代理与人类社区的结构与语言，证明社区同质化主要由跨社区作者共享导致而非语言本身，并展示 AI 代理在语言上更具可识别性；

**🔧 技术方法**

使用统计学方法（Gini、幂律拟合、Jensen–Shannon、Jaccard、Cohen’s d、t检验、KS检验）、心理语言学（LIWC）、可读性与语用特征提取、逻辑回归作者归属分类与 PCA 可视化；

**📊 数据集**

数据集为 2026 年 1 月 27 日至 2 月 9 日的公开抓取数据：Moltbook（73,899 条）与 Reddit（189,838 条），包含 5 个主题社区（consciousness、philosophy、technology、trading、offmychest）；

**📈 对比分析**

比较方法包括对社区层面的参与不平等、活动分布、跨社区重叠；对文本层面的 LIWC、可读性、词汇多样性、语用指标进行差异统计；对社区层面使用 JSD、Jaccard 及主题分类器评估同质化；对作者层面使用 CoV 及 50 类作者归属分类评估辨识度。性能上：Moltbook 参与 Gini 0.84，高跨社区比例 33.8%，语言情感抑制显著；作者归属准确率 Moltbook 89.6% 远高于 Reddit 45.8%；

**⚠️ 局限性**

局限性包括：仅覆盖两周时间窗口，无法观察长期演化；只挑选 5 个主题社区，缺乏普遍性；缺乏代理元数据（模型、系统提示、自治程度）导致难以区分模型层面与社区结构的影响；相关性研究，未能断定因果关系；

---

## 579. Trained Persistent Memory for Frozen Encoder--Decoder LLMs: Six Architectural Methods

**arXiv ID:** 2603.16413 | [PDF](https://arxiv.org/pdf/2603.16413v1)

**作者:** Hong Jeong `[一作]` (Inha University), Hong Jeong `[通讯]` (Inha University)

**通讯引用:** 6331 | [OpenAlex ID](https://openalex.org/A5064069488)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在冻结的Encoder–Decoder LLM中实现持续的潜在空间记忆，并展示了六种不同的记忆适配器架构能够在小规模下实现非零的遗忘曲线和知识积累。

**💡 创新点**

创新点在于把持久记忆切换到模型的连续潜在空间，并提出了写入-读取机制的六种设计空间，验证了只需极小的可训练参数即可在冻结模型上实现长期记忆。

**🔧 技术方法**

技术包括：冻结Flan‑T5‑XL背骨，训练仅适配器参数，使用注意力耦合、Hebbian、槽位写入等写入规则，以及跨Encoder、Decoder或内部注入点的读取路径。

**📊 数据集**

使用LoCoMo长会话对话数据集（含问答与证据标注）进行训练与评估。

**📈 对比分析**

比较方法是基于遗忘曲线的头部归一化内存召回率；在10×容量下，所有六种方法均显示正向记忆曲线，最强者为Hebbian（长滞后记忆率≈10%）和Slot（短滞后≈15%），累计知识曲线显示最高增益≈9.7%。

**⚠️ 局限性**

局限性包括：模型全部冻结、仅使用单一数据集、极小的记忆容量（64/640槽），以及较低的绝对召回率（≈12%）；未涉及端到端训练、规模更大模型或多任务数据。

---

## 580. RASLF: Representation-Aware State Space Model for Light Field Super-Resolution

**arXiv ID:** 2603.16243 | [PDF](https://arxiv.org/pdf/2603.16243v1)

**作者:** Zeqiang Wei `[一作]` (Capital Normal University), Min Xu `[通讯]` (Capital Normal University)

**通讯引用:** 12663 | [OpenAlex ID](https://openalex.org/A5100413849)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种表示感知状态空间框架RASLF，用于光场图像超分辨率；

**💡 创新点**

创新点在于引入进阶几何细化(PGR)与全景极线表示(PER)，并结合表示感知非对称扫描(RAAS)和双锚点聚合(DAA)来显式建模多表示之间的空间-角度耦合，消除冗余；

**🔧 技术方法**

核心技术包括状态空间模型(SSM)，VSSM算子，Panoramic Epipolar Representation，非对称扫描策略，双锚点聚合模块；

**📊 数据集**

使用了EPFL、INRIA、STF‑gantry、HCIold和HCInew五个公共光场数据集进行训练与评估；

**📈 对比分析**

与16种SOTA方法（CNN、Transformer和SSM基模型）对比，RASLF在2×和4×超分任务中实现了最高或第二高的PSNR/SSIM，同时保持极低参数量（约0.9M）和较低算力；

**⚠️ 局限性**

局限性包括对GPU优化的SSM实现尚不成熟，推理速度仍略逊于轻量CNN；进一步研究需探索SSM与光场几何先验的更深层融合。

---

## 581. Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies

**arXiv ID:** 2603.16536 | [PDF](https://arxiv.org/pdf/2603.16536v1)

**作者:** Vassilios Tsounis `[一作]` (Disney Research), Moritz Bächer `[通讯]` (Disney Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一款名为Kamino的GPU加速物理求解器，支持带闭环连杆的异构刚体多体系统，可在RL训练中并行模拟数千个实例。

**💡 创新点**

采用最大坐标表述结合Proximal‑ADMM，原生处理闭环约束；实现块式Cholesky/矩阵无关的Delassus求解；支持异构世界并行化，突破同质化限制。

**🔧 技术方法**

使用NVIDIA Warp编译Python核、CUDA图捕获；最大坐标法、Delassus矩阵、Proximal‑ADMM、Nesterov加速、温启动、Jacobi预处理、Conjugate Residual迭代；多线程二维网格、SO(3)指数映射、Moreau‑Jean中点积分等技术。

**📊 数据集**

对四套机器人模型（DR Legs、BDX、Olaf、Iron Man）进行仿真；DR Legs用于RL训练，训练数据为约3.9 × 10⁸个环境转移。

**📈 对比分析**

通过比较稀疏CR与稠密LLT求解器在内存和吞吐量上的表现；在RTX Pro 6000上，稀疏CR在约300个约束时优于稠密；在4096个环境下，DR Legs可达约3600 env/s，训练收敛约31 小时；不同GPU型号的速度提升也被量化。

**⚠️ 局限性**

仅支持刚体多体，未涵盖粒子或软体；缺乏梯度可微性；当前仅实现双重正向动力学，未实现原始或KKT变体；极大接触数量时的可扩展性仍待验证；未完成实物转移实验。

---

## 582. Attribution Upsampling should Redistribute, Not Interpolate

**arXiv ID:** 2603.16067 | [PDF](https://arxiv.org/pdf/2603.16067v1)

**作者:** Vincenzo Buono `[一作]` (Halmstad University), Stefan Byttner `[通讯]` (Halmstad University)

**通讯引用:** 869 | [OpenAlex ID](https://openalex.org/A5069409359)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对解释性AI中上采样问题进行理论分析并提出 Universal Semantic-Aware Upsampling（USU）解决方案，替代传统插值；

**💡 创新点**

引入四个可上采样理想属性，证明标准插值违反三项，并基于 Luce 选择公理推导出唯一满足所有条件的比值重分配算子 USU；

**🔧 技术方法**

运用形式化公理、ratio-form 重分配、Luce IIA、热参数控制、Soft IWMR、层级边界细化等技术；

**📊 数据集**

使用合成形状任务、ImageNet、CIFAR‑10、CUB‑200 等数据集；

**📈 对比分析**

与 bilinear、bicubic、Lanczos 等插值以及多种归因方法对比，Infidelity、IoU、PG 等指标提升 1–4 个数量级，表现优异；

**⚠️ 局限性**

依赖预先得到语义分割与分数，且仅适用于线性 mass 传播，对 mass 依赖加权的情况仍是开放问题。

---

## 583. Probing Cultural Signals in Large Language Models through Author Profiling

**arXiv ID:** 2603.16749 | [PDF](https://arxiv.org/pdf/2603.16749v1)

**作者:** Valentin Lafargue `[一作]` (IMT), Jean-Michel Loubes `[通讯]` (INRIA Bordeaux)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对多款开源大语言模型（LLM）在无监督（zero‑shot）条件下进行歌曲歌词作者的性别与族裔推断，并通过模型自我解释和公平度量评估其文化偏见。

**💡 创新点**

提出两种新的公平性度量：Modality Accuracy Divergence（MAD）和 Recall Divergence（RD），并在歌词域通过零样本作者画像实验揭示LLM在不同族裔与性别上的系统性偏差。

**🔧 技术方法**

使用 instruction‑tuned LLM（如 Llama‑3.1‑8B、Gemma‑12B、Ministral‑8B 等）配合多级提示（Regular、Informed、Well‑informed、Corrected）进行推断，并采用统计检验（χ²、CLT、Wasserstein）及 MAD/RD 评估偏差；同时对模型生成的推理过程进行自我解释分析。

**📊 数据集**

构建了约10,808 首歌词数据集（来自 Deezer、Spotify、Genius），先将非英文歌词翻译为英文（使用 Mistral‑Small 3.2），并按艺术家性别与宏观族裔（非洲、亚洲、欧洲、北美、澳大、南美）平衡筛选得到 2,973 位单人艺术家。

**📈 对比分析**

采用分层自举 1,000 次、χ²/CLT/Wasserstein 检验与 MAD/RD 衡量模型在族裔/性别上的分布偏差；大模型平均可达 76% 性别、44% 族裔的准确率，Gemma‑12B 在公平度量上最平衡，Ministral‑8B 则表现最强的族裔偏差；模型大小、提示复杂度、歌词长度与流派均显著影响性能。

**⚠️ 局限性**

仅限于歌词文本、翻译可能引入风格失真、族裔划分为宏观地区而非细粒度、未充分考虑写手与鬼写手、数据来源与预训练语料缺乏透明度、存在性能与公平之间的权衡。

---

## 584. A low-data, low-cost, and open-source workflow for 3D printing lithographs for digital accessibility of microscopy images

**arXiv ID:** 2603.16801 | [PDF](https://arxiv.org/pdf/2603.16801v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 585. A Semantic Timbre Dataset for the Electric Guitar

**arXiv ID:** 2603.16682 | [PDF](https://arxiv.org/pdf/2603.16682v1)

**作者:** Joseph Cameron `[一作]` (University of Cambridge), Alan Blackwell `[通讯]` (University of Cambridge)

**通讯引用:** 6854 | [OpenAlex ID](https://openalex.org/A5017575045)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文创建并验证了一个包含19个语义音色描述符的单声道电吉他音色数据集，并使用变分自编码器进行音色生成与插值。

**💡 创新点**

创新点在于首次将音乐家常用的语义描述符与音色强度量化关联，构建细粒度标注数据集，支持高层语义控制。

**🔧 技术方法**

使用技术包括变分自编码器（VAE）学习音色潜在空间、CNN分类器评估语义一致性，以及MOS和Kendall Tau等主观评估。

**📊 数据集**

使用的数据集为275,310个由EGFxSet清洁吉他音符通过Guitar Rig 7 Pro等效果器生成并标注的单声道电吉他样本。

**📈 对比分析**

通过MOS（平均>4.0）、CNN分类准确率94.6%以及Kendall Tau 0.879（p<0.001）验证了VAE在音色重建与语义插值方面的优异性能。

**⚠️ 局限性**

局限性包括对快速调制音色（如flutter、stutter）的重建不佳，且数据仅覆盖单声道吉他，缺乏对多声部或其他乐器的泛化。

---

## 586. $D^3$-RSMDE: 40$\times$ Faster and High-Fidelity Remote Sensing Monocular Depth Estimation

**arXiv ID:** 2603.16362 | [PDF](https://arxiv.org/pdf/2603.16362v1)

**作者:** Ruizhi Wang `[一作]` (Zhejiang University), Li Sun `[通讯]` (Ningbo Global Innovation Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新型的遥感单目深度估计框架D^3-RSMDE，先用ViT快速生成结构一致的粗深度图，再通过轻量化的扩散模块在VAE潜在空间中进行少步的高保真细节重建；

**💡 创新点**

创新点主要包括：① 用ViT替代扩散的耗时结构生成，显著加速；② 设计了Progressive Linear Blending Refinement（PLBR）策略，实现少步的逐层细化；③ 在潜在空间中执行扩散，配合VAE大幅降低计算量；④ 通过上述方法在保持高感知质量的同时，推理速度提升至Marigold的40倍以上；

**🔧 技术方法**

采用了ViT编码器+HDN损失、轻量U‑Net扩散网络、VAE（AEKL或VA_VAE）以及PLBR细化策略，并在5个遥感数据集上进行交叉验证训练；

**📊 数据集**

使用了RS3DBench的五个遥感数据集：Japan+Korea（J&K）、Southeast Asia（SA）、Mediterranean（Med）、Australia（Ast）和Switzerland（Swi）；

**📈 对比分析**

与基准ViT模型（DPT、Omnidata、AdaBins）、扩散模型（Marigold、EcoDepth）以及GAN（Pix2pix）进行对比；D^3-RSMDE在绝大多数评估指标（MAE、δ^3、PSNR、LPIPS）上达到SOTA或次优，LPIPS比Marigold低11.85%，推理速度提升40倍，显存占用与轻量ViT模型相当；

**⚠️ 局限性**

局限性包括：① 仍需在训练阶段使用VAE和ViT，训练成本较高；② 在步数过多时可能出现过度细化或伪影；③ 对极端域外遥感图像的泛化能力尚待进一步验证；

---

## 587. Loosely-Structured Software: Engineering Context, Structure, and Evolution Entropy in Runtime-Rewired Multi-Agent Systems

**arXiv ID:** 2603.15690 | [PDF](https://arxiv.org/pdf/2603.15690v1)

**作者:** Weihao Zhang `[一作]` (Hong Kong University of Science and Technology), Hongyi Li `[通讯]` (Tsinghua University)

**通讯引用:** 32438 | [OpenAlex ID](https://openalex.org/A5100413697)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Loosely-Structured Software（LSS）范式，定义三层工程框架（View/Context、Structure、Evolution），并提出对应的设计原则与模式；在RepoBench‑R检索基准和自动化科研流程中进行验证与评估。

**💡 创新点**

把多代理系统视为可运行时生成与演化的软件，首次将View‑Constructed Programming、Runtime Semantic Binding、Endogenous Evolution等核心“物理”与三层熵治理框架结合，形成面向熵管理的系统化设计方法。

**🔧 技术方法**

基于LLM驱动的多代理框架（AutoGen、LangChain等），结合语义索引、视图构造、语义路由、继承生成器、沙箱演化、语义棕页等技术；实验采用DeepSeek API进行推理。

**📊 数据集**

使用RepoBench‑R代码检索基准（Python子集）以及在实验中构建的文件化知识库（包含实验记录、文档等多种artifact）。

**📈 对比分析**

通过与单一Worker检索对比，使用Hit@5和Top1 Accuracy评估；Lens+Worker将Hit@5从0.70提升至0.78，Lens+Index进一步提升至0.84；同时平均上下文token下降、Worker负载下降；总token略升但可通过索引复用摊销；在完整LSS环境中展示Token消耗分布与人工评估得分，证明设计模式在实际任务中的有效性。

**⚠️ 局限性**

受限于语义排序精度提升不足、总token成本升高、演化过程对目标对齐与知识衰退敏感；缺少完整自动化验证；多代理互操作细节与安全性仍未系统化。

---

## 588. Nonlinear Information Theory: Characterizing Distributional Uncertainty in Communication Models with Sublinear Expectation

**arXiv ID:** 2603.16700 | [PDF](https://arxiv.org/pdf/2603.16700v1)

**作者:** Wen-Xuan Lang `[一作]` (National Center for Mathematics and Interdisciplinary Sciences, Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Zhiming Ma `[通讯]` (National Center for Mathematics and Interdisciplinary Sciences, Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文基于非线性期望理论，构建了一套非线性信息理论框架，重新定义了信息熵、联合熵、条件熵、互信息等核心概念，并在此基础上推导了非线性源编码定理、信道编码定理和失真源编码定理。

**💡 创新点**

创新点包括：
1) 引入子线性期望描述源与信道的分布不确定性，突破传统假设单一概率分布的局限；
2) 在非线性期望空间下给出信息熵、互信息的上界与聚点形式；
3) 在极大误差概率与极小误差概率两种评价准则下分别给出源编码与信道编码的性能极限；
4) 对失真编码给出基于非线性互信息的率失真函数，并证明其为最小期望失真准则下的聚点。

**🔧 技术方法**

核心技术主要是子线性期望理论（Peng 的非线性期望理论）、子线性期望空间的容量概念、以及在该空间下的强大数定律；在信息度量上使用极大极小的熵定义，求解聚点；在编码定理中结合大数定律与容量上界推导，构建最大误差概率与最小误差概率两种判准。

**📊 数据集**

本文没有使用具体实验数据集，主要以理论推导为主；在章节 7 中给出基于 Bernoulli 型不确定分布源和不确定分布的二进制对称信道的仿真示例，用于验证理论结果的正确性。

**📈 对比分析**

比较方法：将非线性信息理论下的极大误差概率上界与极小误差概率聚点与传统 Shannon 理论下的熵、容量、率失真函数进行对比。仿真结果表明：
- 在极小误差概率准则下，聚点往往低于经典 Shannon 极限，说明在不确定性环境中可实现更高的压缩率；
- 在极大误差概率准则下，聚点高于经典容量，说明在最小误差准则下可实现更高的传输效率。

**⚠️ 局限性**

限制与不足：
1) 仅给出了聚点性质，并未给出精确的渐近极限或具体编码方案；
2) 对于非离散或有记忆的信道、源的推广尚未完成；
3) 只考虑了子线性期望的上限/下限，未讨论实际实现中的算法复杂度与可行性；
4) 理论假设的弱紧致概率测度族在实际应用中的可估计性尚待研究。

---

## 589. DualPrim: Compact 3D Reconstruction with Positive and Negative Primitives

**arXiv ID:** 2603.16133 | [PDF](https://arxiv.org/pdf/2603.16133v1)

**作者:** Xiaoxu Meng `[一作]` (Independent Researcher), Lin Gao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种使用正负密度超四边形双原语的可微分3D形状表示方法，支持端到端的多视角图像学习并可通过闭式布尔差分提取网格。

**💡 创新点**

创新点在于将增补与减补相结合的双原语结构，既保持紧凑与可微性，又能够自然表达孔洞和凹陷等复杂几何特征。

**🔧 技术方法**

利用可微分体渲染器、闭式布尔差分和超四边形参数化，实现了从图像到可解释3D结构的完整管道。

**📊 数据集**

论文未公开具体使用的数据集，推测可能采用了常用的ShapeNet、ModelNet或类似公开3D模型库。

**📈 对比分析**

实验对比未给出详细数值，论文声称该方法在保持结构紧凑与可解释性的同时，相较于仅使用增补原语的方法具有更优的重建质量。

**⚠️ 局限性**

可能的局限包括对极其细节化或高纹理复杂度模型的重建能力有限，闭式布尔运算对数值稳定性要求较高，且未说明对噪声或不完整视角的鲁棒性。

---

## 590. Runtime Governance for AI Agents: Policies on Paths

**arXiv ID:** 2603.16586 | [PDF](https://arxiv.org/pdf/2603.16586v1)

**作者:** Maurits Kaptein `[一作]` (Eindhoven University of Technology), Andriy Podstavnychy `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个针对AI代理的运行时治理框架，将合规政策定义为对代理执行路径的确定性函数，支持路径级别的风险评估与干预

**💡 创新点**

首次将提示、访问控制、代理级安全、内容过滤等现有治理方法归入统一框架，并将运行时路径评估作为核心机制

**🔧 技术方法**

采用正式的政策函数 π_j(A,P_i,s^*,Σ) 以及 Policy Engine 对每一步进行预先评估、决策并维护共享治理状态

**📊 数据集**

未使用具体数据集，论文以理论推导和案例场景为主

**📈 对比分析**

无实验对比或性能评估，论文仅给出概念验证与参考实现的架构说明

**⚠️ 局限性**

局限性包括缺乏概率校准、对代理策略迭代的鲁棒性不足、执行环境的完整性与共享状态一致性问题、以及对多机构协同治理的不足

---

## 591. GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution

**arXiv ID:** 2603.16769 | [PDF](https://arxiv.org/pdf/2603.16769v1)

**作者:** Qiaosi Yi `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 106830 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于强化学习的单步生成式图像超分辨率框架GDPO，并在噪声感知单步扩散模型NAOSD上实现

**💡 创新点**

创新点：引入噪声感知扩散模型实现多样化样本；将DPO与GRPO结合形成GDPO；设计属性感知奖励函数动态权衡保真与感知指标

**🔧 技术方法**

技术：噪声感知扩散模型（NAOSD）、不等时间步策略、Group Direct Preference Optimization（GDPO）、属性感知奖励函数（ARF）、LoRA微调

**📊 数据集**

数据集：LSDIR、FFHQ、Real-ESRGAN降解训练集；测试使用DRealSR、RealSR、DIV2K-val

**📈 对比分析**

与多步扩散方法（StableSR、PASD、DiffBIR、SeeSR）、一阶扩散方法（OSEDiff、InvSR）以及RL方法DP^2OSR进行对比；在PSNR、SSIM、LPIPS、FID等指标上均优于对手，尤其在真实场景数据上显著提升

**⚠️ 局限性**

限制：训练时需为每个输入生成多份样本，增加训练开销；奖励函数仍是手工设计的启发式方法，缺乏与人类感知完全一致的对齐

---

## 592. DISCOVER: A Solver for Distributional Counterfactual Explanations

**arXiv ID:** 2603.16436 | [PDF](https://arxiv.org/pdf/2603.16436v1)

**作者:** Yikai Gu `[一作]` (Technical University of Denmark), Lei You `[通讯]` (Technical University of Denmark)

**通讯引用:** 771 | [OpenAlex ID](https://openalex.org/A5082049111)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 DISCOVER，一种模型无关的分布级反事实解释求解器；

**💡 创新点**

创新点在于利用最优传输的样本级分解实现稀疏 top‑k 编辑，并通过 propose‑and‑select 与 OT 引导的锥形采样实现对黑盒模型的无梯度优化；

**🔧 技术方法**

技术包括最优传输（切片 Wasserstein、Wasserstein）、样本级 OT 影响评分、top‑k 预算、候选生成与挑选、OT 引导的锥形采样、遗传与蒙特卡罗优化等；

**📊 数据集**

在 HELOC、COMPAS、Hotel Booking、German Credit、Cardiovascular Disease 等五个标准表格数据集上进行实验；

**📈 对比分析**

与现有的实例级（DiCE）、组级（AReS、GLOBE）和梯度式（DCE）方法对比，DISCOVER 在输入-输出 Wasserstein 距离（MMD）和输出分布匹配（AReS Cost）上取得最佳或次佳成绩，且在非可微模型下性能最优；

**⚠️ 局限性**

局限性包括对 top‑k 预算的依赖、对 OT 近似（切片）精度的敏感、对大规模样本时计算量增长，以及在极端混合约束下可能出现的不稳定性。

---

## 593. SympFormer: Accelerated attention blocks via Inertial Dynamics on Density Manifolds

**arXiv ID:** 2603.16535 | [PDF](https://arxiv.org/pdf/2603.16535v1)

**作者:** Viktor Stein `[一作]` (Institute of Mathematics Technische Universität Berlin), Gabriele Steidl `[通讯]` (Institute of Mathematics Technische Universität Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于概率密度流加速的注意力块，构建加速Transformer并实现SympFormer架构；

**💡 创新点**

将Nesterov加速动力学迁移至Wasserstein/Stein梯度流，得到哈密顿动力学加速注意力块，且在保持相同oracle调用次数的前提下提升收敛速度；

**🔧 技术方法**

利用优化理论、Wasserstein‑2/Stein梯度流、哈密顿动力学、共形辛积分、Adams‑Bashforth等数值离散方法，对线性和softmax注意力进行粒子化数值近似；

**📊 数据集**

使用TinyStories短篇小说分类数据集进行实验；

**📈 对比分析**

与Baseline、YuriiFormer、Plain Euler、Presymp Euler/ExpEuler/AB2/ETD‑AB2等多种离散方案对比，SympFormer在验证损失和收敛速度上均优于对手；

**⚠️ 局限性**

实验规模有限，仅验证了小数据集；未在大规模或下游任务上评估；理论收敛性和适用范围尚待进一步严谨证明与扩展。

---

## 594. Why Avoid Generative Legal AI Systems? Hallucination, Overreliance, and their Impact on Explainability

**arXiv ID:** 2603.15937 | [PDF](https://arxiv.org/pdf/2603.15937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 595. VIEW2SPACE: Studying Multi-View Visual Reasoning from Sparse Observations

**arXiv ID:** 2603.16506 | [PDF](https://arxiv.org/pdf/2603.16506v1)

**作者:** Fucai Ke `[一作]` (Monash University), Hamid Rezatofighi `[通讯]` (Monash University)

**通讯引用:** 3288 | [OpenAlex ID](https://openalex.org/A5034608678)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过物理模拟构建了大规模可控的三维场景，生成了 VIEW2SPACE 这一稀疏多视角视觉推理基准，并提出了可扩展的训练集与基准评测流程。

**💡 创新点**

创新点包括：①基于物理引擎的稀疏多视角数据引擎，可生成 300 万问答对；②多维度难度设计（感知、跨视角、组合推理、可视化证据约束）；③视觉锚定的链式推理（Grounded CoT），显著提升多视角推理与定位性能；④对比实验展示该基准对模型、数据量、尺度与推理深度的敏感性，揭示现有模型在高复杂度与高遮挡场景下的瓶颈。

**🔧 技术方法**

主要技术包括：物理驱动的三维渲染与精确几何/语义注释；OSD‑Tag 自动层级标签生成；多视角渲染与视角组合；规则驱动的可视化推理脚本；基于 Qwen3‑VL‑4B‑Instruct 的全参数微调；Grounded CoT 训练框架；对比基准的评测指标（ACC、MAE、mIoU、F1）。

**📊 数据集**

使用的数据集有：①VIEW2SPACE（约 3M QA、2000 个 3D 场景、40 主题、3k 评估样本）；②MindCube‑Tiny 真实世界多视角基准，用于跨域泛化测试。

**📈 对比分析**

与随机、开源 MLLM、专门的空间模型以及闭源 GPT 进行对比。随机基线约 28%/11%/3%；开源模型在 MCQ 仅提升至 30–38%；闭源 GPT 在 MCQ 可达 59–60%。Grounded CoT 在 VIEW2SPACE 上提升 50%+（mIoU）并在 MindCube 上比官方基准高 9% 以上，证明训练与可视化证据提升显著，但整体性能仍低于人类水平。

**⚠️ 局限性**

局限性包括：①在高推理深度或高遮挡场景下表现急剧下降，表明线性 CoT 与深层推理仍缺乏有效搜索；②模型对稀疏视角的跨视角对齐能力有限，导致定位任务几乎随机；③数据规模与模型规模对性能提升的边际收益递减；④当前评测尚未覆盖动态环境与更复杂的多智能体协同场景。

---

## 596. Discovery of interaction and diffusion kernels in particle-to-mean-field multi-agent systems

**arXiv ID:** 2603.15927 | [PDF](https://arxiv.org/pdf/2603.15927v1)

**作者:** Giacomo Albi `[一作]` (University of Verona), Elisa Calzola `[通讯]` (University of Ferrara)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5004006833)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套基于数据驱动的框架，用于从仅观测到的多主体轨迹数据中同时识别随机多主体系统的漂移和扩散交互核，解决交互结构未知、数据稀缺的逆问题。

**💡 创新点**

创新点在于：①在未知交互配对的情况下，首次提出随机批次采样与均值场重构两种互补策略；②将漂移与扩散交互核的识别统一成稀疏回归问题，并给出严格的先验误差估计；③在单轨迹、少量时间步的有限数据下即可实现高精度核重构。

**🔧 技术方法**

核心技术包括：稀疏回归（SINDy式）在分段线性紧支撑基空间中的求解、随机批次（Random‑Batch）采样逼近微观交互、均值场（Mean‑Field）逼近非局部积分、加权回归与模型集成（平均法与最佳拟合法）、以及基于格子逼近的经验密度估计。

**📊 数据集**

实验全部基于人工合成数据：1D、2D 10^5 个粒子，时步 Δt 0.01，涵盖 bounded‑confidence、吸引-排斥、非局部扩散等多种交互核；未使用真实社会或生物数据。

**📈 对比分析**

与传统全观测或多轨迹方法对比，采用三种策略（随机批次平均、随机批次最佳、均值场）得到的相对误差均在 10^‑2–10^‑1 范围内，Wasserstein 密度误差亦在 10^‑3 左右，表明三种方法在不同可观测程度下均能保持相近的重构精度。

**⚠️ 局限性**

局限性包括：①仅在人工合成数据上验证，缺乏对真实应用场景的测试；②对交互核的假设为径向/可分离形式，复杂高维交互难以直接捕捉；③对大距离交互或边界信息不足时重构误差明显上升；④需设定基函数数量、批量大小、时间窗口等超参数，调参较为繁琐。

---

## 597. Point-to-Mask: From Arbitrary Point Annotations to Mask-Level Infrared Small Target Detection

**arXiv ID:** 2603.16257 | [PDF](https://arxiv.org/pdf/2603.16257v1)

**作者:** Weihua Gao `[一作]` (Chinese Academy of Sciences), Xiaodong Peng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 15745 | [OpenAlex ID](https://openalex.org/A5100369688)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 Point-to-Mask 框架，通过 PAMG 将点标注转为伪掩模并与 RPR-Net 结合，实现低成本点监督的红外小目标检测。

**💡 创新点**

创新点在于结合物理光学先验的自适应掩模生成与中心+半径几何回归闭环，并构建 SIRSTD-Pixel 序列像素级数据集。

**🔧 技术方法**

采用物理驱动的 MAP 能量函数、点扩展掩模生成 (PAMG)、时空注意机制的 RPR-Net，以及基于中心+半径的几何回归。

**📊 数据集**

使用了 SIRSTD-Pixel（序列像素级注释）、NUDT‑SIRST、IRSTD‑1K 等公开数据集。

**📈 对比分析**

与全监督和点监督方法对比，RPR-Net 在 SIRSTD-Pixel 上达 Pd 91.5%、AUC 0.958，点监督+PAMG mIoU 57.4% 仅差 4.5%，在单帧数据上仍具竞争力。

**⚠️ 局限性**

局限在于对极弱信号、强噪声或非圆形扩散目标的鲁棒性有限，几何回归近似受限。

---

## 598. Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR

**arXiv ID:** 2603.16184 | [PDF](https://arxiv.org/pdf/2603.16184v1)

**作者:** Quy-Anh Dang `[一作]` (Knovel Engineering Lab), Chris Ngo `[通讯]` (Knovel Engineering Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在新加坡四官方语言（英语、普通话、泰米尔语、马来语）的多语言语音识别任务上，本文提出并训练了 Polyglot‑Lion 这一小型模型。

**💡 创新点**

创新点在于采用两阶段语言平衡上采样以消除训练数据的不均衡，同时去除语言标签条件，让模型通过音频自适应识别语言。

**🔧 技术方法**

技术上使用 Qwen3‑ASR 预训练模型进行细粒度微调，并采用 Conformer 编码器 + 生成式解码器，配合均衡采样与无语言标记的训练策略。

**📊 数据集**

使用公开语料库，包括 Librispeech、NSC、AISHELL‑1/3、Common Voice 23、Fleurs、SLR127/65 以及 Mesolitica 等，涵盖了所有四种语言。

**📈 对比分析**

与 Whisper‑large‑v3‑turbo、SeaLLMs‑Audio‑7B、Qwen2.5‑Omni‑3B/7B、MERaLiON‑2‑10B‑ASR 等八个基线进行对比，Polyglot‑Lion‑1.7B 的平均错误率仅比 MERaLiON 略高（14.85 vs 14.32），但模型参数 6 倍更小、推理速度 20 倍更快、训练成本降低 233 倍。

**⚠️ 局限性**

局限性包括在新加坡英语（NSC）和泰米尔语的性能仍低于 MERaLiON，缺乏对代码切换语料的评估，以及模型对极端口音或混合语言场景的鲁棒性尚未验证。

---

## 599. Adaptive Captioning with Emotional Cues: Supporting DHH and Neurodivergent Learners in STEM

**arXiv ID:** 2603.15977 | [PDF](https://arxiv.org/pdf/2603.15977v1)

**作者:** Sunday David Ubur `[一作]` (Virginia Tech), Denis Gracanin `[通讯]` (Virginia Tech)

**通讯引用:** 2476 | [OpenAlex ID](https://openalex.org/A5056069234)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过设计和评估四种情感增强实时字幕原型，支持聋人和神经多样性学生在 STEM 教学中的学习。

**💡 创新点**

创新点在于将面部表情、关键词高亮、emoji/图标等情感线索与实时字幕相结合，并强调可定制化。

**🔧 技术方法**

使用的技术包括基于 HTML/CSS/JS 的实时字幕生成、音频语调检测（情绪标记）、视觉高亮动画以及嵌入视频覆盖。

**📊 数据集**

数据集为自制的 STEM 讲座视频（化学、工程、生物等）及对应的 ASR 文本，参与者为 24 名 DHH/神经多样性学习者。

**📈 对比分析**

与传统字幕基线相比，实验显示情感增强原型在 NASA‑TLX 认知负荷和短时理解测验上有显著提升，尤其是 P3（关键词高亮）在降低心理负荷方面最为突出。

**⚠️ 局限性**

局限性包括样本量小、DHH 参与者仅 8 人、不同视频主题导致内容差异、实验周期短以及仅测量即时理解而非长期学习效果。

---

## 600. Data-Local Autonomous LLM-Guided Neural Architecture Search for Multiclass Multimodal Time-Series Classification

**arXiv ID:** 2603.15939 | [PDF](https://arxiv.org/pdf/2603.15939v1)

**作者:** Emil Hardarson `[一作]` (Reykjavik University), María Óskarsdóttir `[通讯]` (University of Southampton)

**通讯引用:** 1473 | [OpenAlex ID](https://openalex.org/A5089062610)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种在本地数据环境下使用大语言模型引导的神经架构搜索（LLM‑NAS）框架，用于多类别多模态时间序列分类；

**💡 创新点**

创新点在于：①将多模态分类拆解为每个模态、每个类别的二分类专家并融合，②使用LLM作为控制器，在不访问原始数据的前提下动态生成并评估专家网络架构；

**🔧 技术方法**

使用的技术包括：多模态时间序列预处理、基于残差‑Inception 的1D CNN专家网络、轻量级融合MLP、LLM驱动的代理搜索循环和基于共享文件系统的实验记录；

**📊 数据集**

实验数据集包括公开的UEA30 22个多变异时间序列任务以及医疗睡眠分期数据集SleepEDFx；

**📈 对比分析**

与先前工作对比：在大多数任务上，基线专家+融合模型已显著优于端到端模型，LLM‑NAS进一步提升性能，尤其是SleepEDFx（从84.7%提升到87.9%）和EthanolConcentration（从32%提升到44.1%）；

**⚠️ 局限性**

局限性包括：LLM可能利用已有公开架构导致性能提升难以归因、搜索时间与计算成本高、在部分数据集上表现有限或略降，且框架主要针对分类任务，扩展到回归或预测仍需研究。

---

## 601. MedArena: Comparing LLMs for Medicine-in-the-Wild Clinician Preferences

**arXiv ID:** 2603.15677 | [PDF](https://arxiv.org/pdf/2603.15677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 602. FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation

**arXiv ID:** 2603.16596 | [PDF](https://arxiv.org/pdf/2603.16596v1)

**作者:** Fangjing Li `[一作]` (Beijing Jiaotong University), Ming Jin `[通讯]` (Griffith University)

**通讯引用:** 13208 | [OpenAlex ID](https://openalex.org/A5039636381)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一种轻量化的顶层框架FSMC‑Pose，用于在群内密集环境下对奶牛挂载姿态进行精确估计。

**💡 创新点**

创新点包括：① 将频域与空间域信息融合的SFEBlock；② 通过多尺度受体聚合RABlock提升对不同尺度关节的感知；③ 引入空间‑通道自校准头SC2Head，有效校正重叠下的结构偏移；④ 构建专门的MOUNT‑Cattle挂载数据集。

**🔧 技术方法**

采用了MobileNet‑style轻量化骨干，深度可分离卷积与倒置残差结构；小波变换与高斯滤波实现频域增强；空洞卷积实现多尺度受体聚合；空间‑通道注意力与自校准分支组成SC2Head；整体采用顶层结构RTMPose改造。

**📊 数据集**

使用自采的MOUNT‑Cattle（1176个挂载实例）与公开的NWAFU‑Cattle数据集融合而成的综合基准。

**📈 对比分析**

与多种顶层与底层基准（DEKR、SimCC、RTMPose、RTMO等）进行对比，FSMC‑Pose在该基准上取得AP 89.0%、AP75 92.5%、AR 89.9%、AR75 93.1%，参数仅2.698M，FLOPs 0.354GF，实时推理速度216FPS。

**⚠️ 局限性**

局限性：仅使用单帧图像，缺乏时间序列与多摄像头信息；在极端光照或遮挡条件下仍可能出现误检；目前只覆盖挂载姿态，未覆盖完整的发情行为管线。

---

## 603. FederatedFactory: Generative One-Shot Learning for Extremely Non-IID Distributed Scenarios

**arXiv ID:** 2603.16370 | [PDF](https://arxiv.org/pdf/2603.16370v1)

**作者:** Andrea Moleri `[一作]` (Honda Research Institute Europe), Barbara Hammer `[通讯]` (Bielefeld University)

**通讯引用:** 9754 | [OpenAlex ID](https://openalex.org/A5091180862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种零依赖的一次性生成式联邦学习框架FederatedFactory，在单类孤岛极端非IID情形下通过传输生成先验而非判别参数，实现全局平衡数据合成并训练分类器。

**💡 创新点**

创新点包括：①将联邦学习通信单元从判别参数转为生成先验，消除梯度冲突；②一次通信完成生成与训练，极大降低通信开销；③支持中心化与P2P两种模式；④利用生成矩阵实现精确的模块化遗忘。

**🔧 技术方法**

使用技术：EDM2扩散模型做本地生成器；ResNet‑50作为分类器；PoE融合做P2P聚合；One‑Shot Federated Learning；理论证明无外部预训练模型依赖；生成矩阵实现模块化Unlearning。

**📊 数据集**

实验数据集：CIFAR‑10、MedMNIST（BloodMNIST、RetinaMNIST、PathMNIST）以及ISIC2019。

**📈 对比分析**

与FedAvg、FedDyn、FedProx、Scaffold等基线在单类孤岛下比较，FederatedFactory在准确率/ AUROC 上几乎达到中心化上界：如CIFAR‑10从11.36%提升至90.57%，ISIC2019 AUROC从47.31%提升至90.57%；同时通信量下降至99.4%。

**⚠️ 局限性**

限制：计算成本显著增加（生成器训练 FLOPs 增至十倍以上）；缺乏正式的差分隐私等隐私保证；理论假设需要生成器收敛且不记忆本地数据。

---

## 604. Execution-Grounded Credit Assignment for GRPO in Code Generation

**arXiv ID:** 2603.16158 | [PDF](https://arxiv.org/pdf/2603.16158v1)

**作者:** Abhijit Kumar `[一作]`, Shikhar Gupta `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于执行跟踪的信用分配方法 EGCA，在 critic‑free RL 中精确定位程序失败的语义位置并加权梯度。

**💡 创新点**

将参考解的执行轨迹与候选程序比较，识别最早的语义偏差点，只在该 token 片段上分配优势，从而解决传统 GRPO 的信用扩散问题。

**🔧 技术方法**

采用 Group Relative Policy Optimization、结构化约束提取、AST/CFG 对比、LLM 辅助的执行对比定位以及 token 级 GRPO 优势分配等技术。

**📊 数据集**

在 APPS+ 训练集上训练，使用 HumanEval 与 MBPP 作为评估基准。

**📈 对比分析**

与 GRPO、StepCoder、RLTF、CodeRL+ 等基线同基模型比较，EGCA 在 HumanEval 上达 82.1% pass@1，MBPP 68.9%，分别比 GRPO 提升 3.1 和 1.5 点，整体优于其他方法。

**⚠️ 局限性**

依赖可用的参考解和 LLM 调试器；仅适用于近似正确的程序，对低质量初始模型和结构多样的解方案效果有限。

---

## 605. COGNAC at SemEval-2026 Task 5: LLM Ensembles for Human-Level Word Sense Plausibility Rating in Challenging Narratives

**arXiv ID:** 2603.15897 | [PDF](https://arxiv.org/pdf/2603.15897v1)

**作者:** Azwad Anjum Islam `[一作]` (Florida International University), Tisa Islam Erana `[通讯]` (Florida International University)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5049012590)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种使用闭源大语言模型和三种提示策略（零样本、链式思考、比较提示）来预测叙事短文中同形词的可行性评分，并通过模型集成提升了与人类平均评分的契合度。

**💡 创新点**

创新点在于：①首次将比较提示与同形词可行性评估相结合，显著提升了性能；②通过对多模型、多提示策略的无权重平均集成，有效缓解了高标注方差导致的预测偏差。

**🔧 技术方法**

采用的技术包括：基于提示的生成式推理（Zero-shot、Chain-of-Thought、Comparative Prompting）、JSON结构化输出、无监督模型集成以及 Spearman 相关系数和标准差内准确率的评估。

**📊 数据集**

使用的语料为 SemEval‑2026 Task 5 所用的 AmbiStory 数据集，共 3,798 个样本，包含 633 个设置，每个设置有 6 个样本（两种意义各自的两种故事结尾）。

**📈 对比分析**

与单一模型和提示策略比较，比较提示策略在所有模型中表现最好；模型集成后在官方评测上取得 0.89 的平均得分（0.93 的准确率与 0.86 的 Spearman ρ），排名赛制第四，赛后进一步提升至 0.92 平均得分。

**⚠️ 局限性**

局限性包括：仅依赖闭源商用 LLM，缺乏可复现性和可访问性；未使用训练集进行微调；集成方法计算量大、延迟高，资源受限场景下实用性有限。

---

## 606. Rotated Robustness: A Training-Free Defense against Bit-Flip Attacks on Large Language Models

**arXiv ID:** 2603.16382 | [PDF](https://arxiv.org/pdf/2603.16382v1)

**作者:** Deng Liu `[一作]` (University of Science and Technology of China), Song Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 252346 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种训练‑free 的 RoR 防御，利用正交 Householder 变换对激活空间进行旋转，从而抑制权重位翻转引起的极端激活放大，防止大语言模型崩溃。

**💡 创新点**

创新点在于：① 将正交旋转与低秩 WY 表示结合，实现“零损失”防御；② 通过对极端激活通道进行静态识别并旋转，彻底破坏权重错误与激活尖峰的空间对齐；③ 兼顾极低存储和推理延迟开销，使防御可落地。

**🔧 技术方法**

使用技术包括：Householder 正交变换、Compact WY 低秩矩阵乘法、离线离线权重融合、激活极值阈值检测；实现基于 PyTorch 2.4 与 CUDA 的 GPU 推理。

**📊 数据集**

评估数据集：WikiText‑2（语言建模），MMLU、HellaSwag、PIQA（下游推理）以及多种 LLM 变体（Llama‑2/3、OPT、Qwen）。

**📈 对比分析**

与 FaR、RADAR 等现有防御对比：RoR 在黑盒随机位翻、灰盒 Progressive Bit Search 与白盒单点失败攻击中，将模型崩溃率从 3.15%/0.00% 降至 0%；在 50 次定向翻转后 MMLU 仍保持 43.9%（接近未受攻击的 45.2%）。存储仅 +0.31%，推理延迟 +9.1%–19.2%，显著优于其它方案。

**⚠️ 局限性**

局限性：仅覆盖密集线性层，无法直接应用于归一化层；对多模态模型需要动态自适应旋转；目前低秩更新仍受内存带宽限制，可能导致推理延迟略高；防御仅针对位翻转，不涉及其它硬件攻击模式。

---

## 607. Online Experiential Learning for Language Models

**arXiv ID:** 2603.16856 | [PDF](https://arxiv.org/pdf/2603.16856v1)

**作者:** Tianzhu Ye `[一作]` (Microsoft Research), Furu Wei `[通讯]` (Microsoft Research)

**通讯引用:** 31969 | [OpenAlex ID](https://openalex.org/A5014662947)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出在线经验学习框架，利用部署时收集的交互轨迹提取可迁移经验知识，并通过自我策略的 on‑policy 上下文蒸馏将知识内化进模型参数，实现服务器端持续改进。

**💡 创新点**

突破传统离线训练的瓶颈：无需奖励函数、人工标注或服务器侧环境访问；将文本环境反馈转化为可迁移知识；利用自举的 on‑policy 蒸馏克服离线分布偏差和灾难性遗忘。

**🔧 技术方法**

经验知识提取（π_extract 模型）、累计式知识聚合、on‑policy 上下文蒸馏（反向 KL）、自我策略推理、迭代学习循环。

**📊 数据集**

文本游戏环境 Frozen Lake 与 Sokoban（TextArena），实验使用 Qwen3 系列模型（1.7B、4B、8B）进行交互与学习。

**📈 对比分析**

与无经验、原始轨迹、off‑policy 蒸馏等方法对比；OEL 在 Frozen Lake 两轮提升 pass rate 由约 10% 提升至 20% 以上，token 长度缩短 30%；在 IF‑Eval 的 OOD 任务上保持与初始模型相近的性能，显著优于基线和 off‑policy 方法。

**⚠️ 局限性**

受限于足够轨迹与模型容量、知识提取质量受模板约束、上下文窗口饱和导致性能上限；仅在文本游戏验证，缺乏对更复杂多模态或真实任务的评估；需要持续收集用户轨迹，涉及隐私与部署成本问题。

---

## 608. SWE-QA-Pro: A Representative Benchmark and Scalable Training Recipe for Repository-Level Code Understanding

**arXiv ID:** 2603.16124 | [PDF](https://arxiv.org/pdf/2603.16124v1)

**作者:** Songcheng Cai `[一作]` (University of Waterloo), Wenhu Chen `[通讯]` (University of Waterloo)

**通讯引用:** 4962 | [OpenAlex ID](https://openalex.org/A5103103242)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 SWE-QA-Pro 基准，聚焦长尾可执行仓库的仓库级 QA，并提出两阶段 SFT+RLAIF 训练流程提升小模型的探索与推理能力。

**💡 创新点**

创新点包括：① 用 issue 文本聚类与多维过滤剔除可直接知识回答的题目；② 通过合成数据管道与人工校验生成高质量 QA；③ 训练两阶段策略使开源模型逼近 GPT‑4o 等专业模型。

**🔧 技术方法**

技术涵盖语义聚类（k‑means、Qwen3‑8B embedding）、ReAct‑式代理（Search/View/CommandLine）、监督微调（SFT）、强化学习（RLAIF/GRPO）、LLM‑as‑Judge 评估与奖励。

**📊 数据集**

使用 SWE‑QA‑Pro（260 题/26 仓库）及其训练/测试拆分；基准构建基于 SWE‑Rebench 的 1.6M issue；对照 CodeQueries、RepoChat 等现有仓库级 QA 数据集。

**📈 对比分析**

采用严格的 LLM‑as‑Judge 评分，比较直接回答与 agentic 两种方式；agentic 在正确性、完整性等指标提升约 13 分；Qwen3‑8B+SFT+RL 在 SWE‑QA‑Pro 上比 GPT‑4o 提升 2.3 分，接近 Claude Sonnet 4.5 等先进模型。

**⚠️ 局限性**

局限性：① 数据规模有限，仅 260 题；② 仅覆盖 Python 生态，无法直接迁移到其他语言；③ 训练奖励与评估使用相同 LLM‑Judge，存在奖励作弊风险，需进一步研究更稳健的奖励机制。

---

## 609. Real-Time Decoding of Movement Onset and Offset for Brain-Controlled Rehabilitation Exoskeleton

**arXiv ID:** 2603.16825 | [PDF](https://arxiv.org/pdf/2603.16825v1)

**作者:** Kanishka Mitra `[一作]` (Massachusetts Institute of Technology), José del R. Millán `[通讯]` (University of Texas at Austin)

**通讯引用:** 75928 | [OpenAlex ID](https://openalex.org/A5107838719)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过非侵入性EEG和运动想象，实时控制臂部外骨骼实现起始与停止双状态指令；

**💡 创新点**

首次实现在线双状态MI解码实现起止控制，并提出类无关的注视点再中心化方法显著降低EEG漂移与类别偏差；

**🔧 技术方法**

利用Riemannian几何的最小距离分类器、指令基准化的复位、stNMES反馈、Harmony外骨骼闭环控制；

**📊 数据集**

在8名健康右撇子成年人上完成一次离线校准与两次在线会话（共160次试验），使用64通道EEG；

**📈 对比分析**

与传统任务基准化对比，采用AUC评估，修正后AUC从0.554提升至0.866（+56%）起始，0.619提升至0.832（+34%）停止；平均起始命中率≈65%，停止≈65%，起始延迟≈1s，停止≈3.4s；

**⚠️ 局限性**

仅限小样本健康志愿者、仅两次会话、缺乏中风患者验证、对长期学习与高非稳态的适应性尚未评估。

---

## 610. CABTO: Context-Aware Behavior Tree Grounding for Robot Manipulation

**arXiv ID:** 2603.16809 | [PDF](https://arxiv.org/pdf/2603.16809v1)

**作者:** Yishuai Cai `[一作]` (National University of Defense Technology), Yuanpei Chen `[通讯]` (PsiBot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了自动构建完整一致行为树（BT）系统的框架CABTO；

**💡 创新点**

首次将大型语言模型与可视语言模型结合，用规划与环境反馈交叉引导的搜索方法解决BT grounding问题，实现全自动BT生成；

**🔧 技术方法**

使用GPT‑4o等LLM进行高层动作模型提议，Molmo+cuRobo等VLM进行低层控制策略采样，并辅以BT规划算法；

**📊 数据集**

基于七套机器人操纵任务集（单臂Frankia、双臂Frankia、Fetch移动机器人），共21个目标进行实验；

**📈 对比分析**

与手工编写、进化搜索、RL等基线方法比较，CABTO在规划成功率（平均>90%）、完整规划成功率>90%和反馈循环≤3方面显著优于对手；

**⚠️ 局限性**

受限于LM推理误差、对抽象概念识别不足、仿真与真实机器人差距以及对大量训练数据和微调的需求。

---

