# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-13 | 今日论文总数: 364

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. MemeBuddy: Dialog-Style Audio Representations for Engaging Non-Visual Meme Experiences

**arXiv ID:** 2607.08912 | [PDF](https://arxiv.org/pdf/2607.08912v1)

**作者:** Chirag Bhansali `[一作]` (Michigan State University), Hae-Na Lee `[通讯]` (Michigan State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种将图像表情包重新构造为两人对话的音频表示系统，以提高盲人用户的可访问性和参与度。

**💡 创新点**

创新点在于把表情包视作多轮对话，采用角色化发言者（评论者-评论者或评论者-角色）来分阶段展开幽默与语境，而非传统单一字幕式描述。

**🔧 技术方法**

核心技术为多模态大型语言模型（如Gemini 3 Pro、GPT‑5.1、Gemma 4）配合结构化提示，生成对话文本和TTS参数；随后使用语音合成（TTS）将其转为可播放音频。

**📊 数据集**

使用了100张多样化的表情包图像（从Kaggle Meme Dataset和Imgflip收集），并对其进行人工标注（模板、视觉细节、文字内容、意图摘要）以评估模型生成质量。

**📈 对比分析**

比较方法包括离线评估（四项指标：意图清晰度、表示忠实度、文本准确性、无猜测度）与盲人用户实验（情感、沉浸、娱乐、喜剧感、偏好）。结果显示，CO（评论者-评论者）对话结构在保持理解度的同时显著提升情感与沉浸；BL（字幕）在理解度最高，CH（评论者-角色）在清晰度上略逊。

**⚠️ 局限性**

局限性包括：仅针对静态英文表情包；对角色化对话的语音情感表达有限；实验规模（14位盲人）有限；未能分离多声道与时序展开对参与度的具体贡献；依赖LLM隐式文化推理可能导致误判。

---

## 2. Improved lower bounds of the time complexity of shellsort

**arXiv ID:** 2607.08997 | [PDF](https://arxiv.org/pdf/2607.08997v1)

**作者:** Zhenghan Zang `[一作]` `[通讯]`, Zhenghan Zang

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种基于参数化映射的潜能函数框架，对Shellsort在任意严格递减增量序列下的最坏情况运行时间进行理论分析，并给出了相应的下界。

**💡 创新点**

创新点在于：①将Shellsort的交换过程映射为多项式潜能函数的变化；②利用该框架将先前针对几何级数逼近增量序列的Ω(N³/²)下界推广到更广泛的序列；③利用Dirichlet逼近和几何级数逼近，得到Tokuda序列等经验性增量序列的Ω(N^1.26)下界；④给出了一个通用条件，说明若要实现O(N(log N)^μ)的最坏情况复杂度，则增量序列中某些项必须满足特定大小约束。

**🔧 技术方法**

主要技术包括：复数潜能函数与交换操作的代价分析；Dirichlet逼近定理的强化版；组合数与几何级数逼近的结合；极限下界证明的构造性交换序列。

**📊 数据集**

本文为纯理论分析，没有使用任何实际数据集。

**📈 对比分析**

本文仅给出理论下界，并未与实验实现或已有上界进行比较；因此无法直接评估其在实际Shellsort实现中的性能表现。

**⚠️ 局限性**

局限性包括：只给出最坏情况下界，缺乏上界或平均情况分析；对非严格递减序列适用性有限；实际实现中的增量序列选择仍需经验验证；并未讨论潜能函数在实际实现中的可计算性或效率。

---

## 3. Stochastic Linear Bandits with Partially Observed Actions

**arXiv ID:** 2607.08971 | [PDF](https://arxiv.org/pdf/2607.08971v1)

**作者:** Gautam Dasarathy `[一作]` (Arizona State University), Lalit Jain `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在每个动作仅能看到部分特征的随机线性乐观探索问题，并在动作向量低秩结构下提出基于分段冻结子空间估计的算法；

**💡 创新点**

通过在每个 epoch 只更新一次低秩子空间估计并冻结，使得在部分可观测下仍可实现 √T 的 regret，同时兼顾缺失率、子空间维度、动作集大小和协方差条件的交互；

**🔧 技术方法**

使用了子空间估计的逆概率校正协方差、Davis–Kahan sinΘ 定理、最小二乘填补、OFUL 的自归一化不等式以及多阶段双倍长度 epoch 结构；

**📊 数据集**

在合成实验中使用 d=30, m=3 的随机低秩数据，以及 MNIST 分类任务的低秩特征；

**📈 对比分析**

与全零填充 OFUL、PSLB 等基线相比，在高缺失率（p 低）时 TOFU 与 RA-TOFU 取得显著更低的累计 regret；

**⚠️ 局限性**

局限在于对缺失率 p 的依赖仍可能不是最优（√T/p² 可能不是必要的），且假设动作集合 iid 且缺失随机，实际应用中可能需要处理近似低秩或缺失非随机的情况。

---

## 4. Clean2FX: Label-conditioned modeling for clean-to-effect guitar audio transformations

**arXiv ID:** 2607.08863 | [PDF](https://arxiv.org/pdf/2607.08863v1)

**作者:** Oliverio Bombicci Pontelli `[一作]` (Queen Mary University of London), Iran R. Roman `[通讯]` (Queen Mary University of London)

**通讯引用:** 225 | [OpenAlex ID](https://openalex.org/A5027526258)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究并实现了针对电吉他音频的标签条件的干净到效果转换（Clean2FX），并提供网页演示。

**💡 创新点**

将音频效果标签与神经网络相结合，在保持音乐内容不变的前提下实现多种效果的音频转换，并对不同效果的表现提供可解释评估。

**🔧 技术方法**

采用四种模型（两种VAE、两种U‑Net）进行训练，输入为STFT幅度谱（线性或对数表示），使用FiLM条件化、加权损失和Griffin‑Lim相位重建。

**📊 数据集**

使用EGFxSet数据集，包含十种真实硬件效果处理的电吉他音频及对应的干净参考，构造合成和弦、旋律和混合时间线。

**📈 对比分析**

通过MSE（线性幅度）和Fréchet Audio Distance（FAD）与无效果复制基线对比，U‑Net模型在MSE上提升约70%，FAD提升约30%，而VAE模型平均未能超过基线。

**⚠️ 局限性**

对延迟、混响等时延效果的感知质量提升有限；模型在训练数据之外的真实演奏中效果衰减，且仅恢复幅度谱，缺乏相位信息。

---

## 5. Programming-by-Example for Batch-Editing Collision Meshes in 3D Software

**arXiv ID:** 2607.08804 | [PDF](https://arxiv.org/pdf/2607.08804v1)

**作者:** Gengyang Xu `[一作]` (Hong Kong University of Science and Technology), Shing-Chi Cheung `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 3393 | [OpenAlex ID](https://openalex.org/A5034057959)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出 MeshForge，针对 3D 软件中碰撞网格的批量编辑问题，采用编程‑by‑example 的方式自动生成可复用的批量编辑程序；

**💡 创新点**

创新点包括：① 将碰撞网格转化为符号中间表示，并设计三层抽取器（语义、拓扑、几何）实现跨异构网格的泛化；② 引入自举推理 (ABI) 算法在少量感知噪声下自动修正标签，显著提升合成鲁棒性；③ 构建首个碰撞网格批量编辑基准，系统化评估。

**🔧 技术方法**

技术方法主要有：神经-符号程序合成、基于 VLM 的语义标注、DSL 设计与枚举搜索、ABI 纠错、三维几何处理与碰撞图构建。

**📊 数据集**

使用自构造的基准数据集：8 类 3D 资产（杯、壶、刀、椅、分配器、门、灯、桌子），共 200 组资产、600 个碰撞网格，覆盖 24 个批量编辑任务。

**📈 对比分析**

与基线（VLM 直接生成 DSL 或 Blender 脚本）比较，MeshForge 在任务成功率 SSR 上达 95.8%（23/24），功能通过率 FPR 为 0.92，平均演示次数 2.2 次，合成时间 3.5 秒；相比之下 VLM 方案 FPR 分别为 0.69 与 0.52，且需要更长时间。

**⚠️ 局限性**

局限性包括：依赖 GPT‑4o 等高成本 VLM，无法处理需要更复杂编辑（如顶点级、比例约束）的情况，且仅支持凸多面体编辑；系统仍需人工交互验证，受人机因素影响。

---

## 6. FairSelect: A Systematic Evaluation of Multi-Level and Intersectional Algorithmic Fairness

**arXiv ID:** 2607.08953 | [PDF](https://arxiv.org/pdf/2607.08953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 7. SplatCtrl: Perception-Action Coupling via Gaussian Scene Representations and Reactive Robot Control

**arXiv ID:** 2607.08948 | [PDF](https://arxiv.org/pdf/2607.08948v1)

**作者:** Siddarth Jain `[一作]` (Mitsubishi Electric Research Laboratories), Ho Jin Choi `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一个实时场景重建与机器人运动生成的统一框架 SplatCtrl，实现了在动态未知环境中的碰撞避免控制。

**💡 创新点**

创新点在于将 3D Gaussian Splatting 与体素过滤与动态高斯迁移结合，基于高斯过程构建连续可微 SDF，并将其作为控制壁函数融入 QP‑IK 反应控制。

**🔧 技术方法**

采用多视角 RGB‑D 感知、3D Gaussian Splatting、体素过滤、Gaussian Process Distance Field、控制壁函数的 QP‑IK 与采样规划两种运动生成策略。

**📊 数据集**

使用 DTU MVS 数据集进行重建评估，Simulated 3D 环境与真实 Franka Emika Panda 机器人实验，以及人机共享工作空间的用户实验数据。

**📈 对比分析**

与标准 3D‑GS、SMS 安全停机等基线比较，SplatCtrl 在重建精度上略优（PSNR 23.33 vs 23.27），规划成功率提升至约 95%（从 32% 降至 95%），且控制频率可达 100Hz+。

**⚠️ 局限性**

局限包括对大规模环境的可扩展性不足、未实现抓取物体级碰撞检测、需手工调参以及近似球形高斯可能导致精度和不确定性信息损失。

---

## 8. GReFEM: Multimodal LLMs as Zero-Shot Semantic Assistants for Physics-Guided 3D Mesh Refinement

**arXiv ID:** 2607.08798 | [PDF](https://arxiv.org/pdf/2607.08798v1)

**作者:** Kartik Bali `[一作]` (Helmholtz Zentrum Hereon), Roland Aydin `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了 GReFEM 框架，利用现成的多模态大语言模型（MLLM）在无求解器的条件下通过视觉指令定位三维几何中应力集中区域，并生成精细有限元网格。

**💡 创新点**

创新点在于：①提出了零射程语义定位方法，将文本物理指令映射到三维空间；②设计了 orthoViews 视角选择模块以最大化几何可观测度；③在有限元网格生成中以 MLLM 作为解算器替代传统误差估计。

**🔧 技术方法**

使用的技术包括多模态大语言模型（Gemini‑3‑flash, Claude‑Haiku‑4.5, GPT‑5.4‑mini, Qwen3VL‑235B）、二维正交视图投影、图像网格投影与光线投射、以及基于 Gmsh 的尺寸场网格生成。

**📊 数据集**

数据集为 5k 多样化 CAD 零件（ABC CAD、FreeCAD），实验涵盖 30 个几何、5 种载荷情况、3 种提示模板，使用 99% 置信度的 ZZ 应力指标作为真实标签。

**📈 对比分析**

通过与基于几何特征的启发式方法在相同网格细化预算下比较，评估指标为精度、召回、F1。实验显示，MLLM 在 Load+Features 提示下的精度和 F1 与启发式相近或更优，且在前5视角下可显著降低能量误差和 L2 位移误差。

**⚠️ 局限性**

局限性：对光学 CV 预处理、精确像素定位及深度投射有依赖；需要手工设计物理提示；对未见几何特征的泛化能力有限，且无法完全替代物理求解器进行自适应误差估计。

---

## 9. Prompt-Driven Exploration

**arXiv ID:** 2607.08837 | [PDF](https://arxiv.org/pdf/2607.08837v1)

**作者:** Sunshine Jiang `[一作]` (Massachusetts Institute of Technology), Zhang-Wei Hong `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 442893 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Prompt‑Driven Exploration (PDE)，通过 VLM 对提示进行后验采样并更新，进而让 RL 在提示空间进行全局探索，驱动 VLA 策略获得更高奖励；

**💡 创新点**

创新点在于把探索维度从传统动作噪声扩展到语言提示，通过后验采样方式在提示空间内实现全局行为改变，从而突破弱初始策略的局限；

**🔧 技术方法**

技术组合包括 VLM 作为提示生成器、VLA（vision‑language‑action）策略、PPO 强化学习、混合采样（canonical prompt 与 VLM 采样）以及混合反向传播（双提示交叉）来实现提示后验更新；

**📊 数据集**

使用了 LIBERO、LIBERO‑PRO、ManiSkill、LiveCodeBench、AIME 2026 以及 Franka‑FR3 真实机器人实验等数据集；

**📈 对比分析**

与 Action Noise、Parameter Noise、RND、Robometer 等基线对比，PDE 在所有难度层级上显著提升成功率，尤其在初始零成功任务上实现快速收敛，实测样本效率提升超过两倍；

**⚠️ 局限性**

局限性包括对 VLM 对提示理解的依赖，提示空间过大时搜索效率下降；在极稀疏奖励场景下仍需依赖 VLM 诊断；缺乏对大规模分布漂移或在线连续任务的评估。

---

## 10. Is sub-metre resolution necessary for cocoa mapping? A landscape-stratified evaluation of very high resolution imagery, decametric Earth Observation inputs, and operational products in Cote d'Ivoire

**arXiv ID:** 2607.08945 | [PDF](https://arxiv.org/pdf/2607.08945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 11. A Strongly-Subquadratic $(3+\varepsilon)$-Approximation for the Fréchet Distance for Paths in Metric Spaces

**arXiv ID:** 2607.08893 | [PDF](https://arxiv.org/pdf/2607.08893v1)

**作者:** Thijs van der Horst `[一作]` (Utrecht University), Tim Ophelders `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5013950801)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种确定性算法，在强子二次时间内对路径之间的 Fréchet 距离给出 (3+ε) 近似（在一维可达到 3 近似）。

**💡 创新点**

创新点在于：1) 通过构造强子二次时间的 3‑近似判定器，逼近 SETH 下的理论下界；2) 引入“替代子曲线”（surrogate）和“自由空间预言机”，实现对任意度量空间的通用性；3) 完全去掉了先前算法的随机化与高时间常数。

**🔧 技术方法**

核心技术包括：自由空间（free space）与可达性映射（reachability map）构造；使用分治与子路径拆分产生的 3‑近似出口集；通过二分与指数搜索结合的决策树实现最终距离近似；以及在欧氏/几何空间下的距离与弧长预言机。

**📊 数据集**

论文为理论分析，不涉及具体实验数据集；所有结论基于复杂度与理论证明。

**📈 对比分析**

与以往 7+ε 随机近似算法相比，确定性 3+ε 近似在时间上从 O(nm^{0.99}) 改进到 O(nm^{2/3} polylog(n,1/ε))；在 1‑维欧氏空间实现了最优 3 近似；相比基于随机化的 Blank 等算法，该方法在均衡与不均衡两种复杂度下均能保持更低的时间与更好或等价的近似因子。

**⚠️ 局限性**

局限性包括：1) 仍需满足自由空间的可达性属性，对某些非凸空间可能不适用；2) 需要预先构造的自由空间与距离预言机，常数与空间开销仍较大；3) 在实际大规模数据上，空间 O(nm^{2/3}) 与实现细节仍需进一步优化。

---

## 12. Learning-enabled Parameter Synthesis for Nonlinear Systems from Signal Temporal Logic

**arXiv ID:** 2607.08899 | [PDF](https://arxiv.org/pdf/2607.08899v1)

**作者:** Alex Beaudin `[一作]` (University of California, Berkeley), Murat Arack `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种神经符号化参数合成方法，先用梯度学习在离散时间样本上寻找满足STL约束的系统参数，然后利用基于集合的可达性分析给出连续时间满足保证。

**💡 创新点**

①可扩展到高维非线性系统；②将梯度优化与后验验证结合，实现正式保证；③引入残差神经网络平滑学习、SlackReLU损失及Counter‑Example缓冲提升学习稳定性与收敛。

**🔧 技术方法**

梯度下降（SGD/AdaBelief）、残差MLP、SlackReLU/CVaR目标、集合可达性分析（CORA）、STL稳健度量、离散/连续时间语义。

**📊 数据集**

在三种系统上评估：12维四旋翼、18维基因调控网络、7维Laub‑Loomis酶动力学模型。

**📈 对比分析**

与传统符号合成、纯模拟优化以及不同损失函数（ReLU、LeakyReLU）对比，实验表明该方法在参数维度达18时仍能稳健收敛，最终获得连续时间满足保证，验证结果显示可达集与目标一致，性能优于仅使用模拟验证的方法。

**⚠️ 局限性**

STL非光滑性仍导致优化不稳定；时间步长的选择权衡精度与计算成本；残差网络与SlackReLU在不同系统中效果不一；对更大规模系统的可扩展性尚待进一步验证。

---

## 13. Impedance-Guided Programmable Transmission of Localized Deformation in Modular Soft Metamaterials

**arXiv ID:** 2607.08966 | [PDF](https://arxiv.org/pdf/2607.08966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 14. HALO: Hybrid Adaptive Latent Reasoning for Language Models

**arXiv ID:** 2607.08775 | [PDF](https://arxiv.org/pdf/2607.08775v1)

**作者:** Micah Zhang `[一作]` `[通讯]` (Lockheed Martin), Micah Zhang (Lockheed Martin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在冻结预训练语言模型上添加可适配的局部精炼机制（HALO），用于提高推理质量。

**💡 创新点**

创新点在于将粗略精炼与基于令牌评分与单调停止的二阶隐空间精炼相结合，仅在需要的令牌上付费，从而实现高效的计算分配。

**🔧 技术方法**

使用了门控评估、令牌评分与单调停止策略、局部隐层精炼块，以及冻结模型的可训练精炼头。

**📊 数据集**

在公开基准 MMLU-Pro 和 GPQA‑Diamond 上进行评估，并在内部使用 token‑accuracy、lift 与平均精炼步数等指标。

**📈 对比分析**

与冻结模型、一次全序列精炼（fixed‑1）和两次全序列精炼（fixed‑2）对比，HALO 在两个基准上的平均得分最高（35.66），且平均精炼步数仅 0.776，显著低于 fixed‑1 的 1.000 和 fixed‑2 的 2.000。

**⚠️ 局限性**

局限性包括仅测试两项基准、结果在不同任务上表现不均衡、计算效率指标仅为平均精炼步数而非实际延迟/吞吐，以及未验证在更广泛任务或更大模型上的泛化能力。

---

## 15. Offline Nash Solvers Meet Online Tree Search in Multi-Agent Games on Graphs

**arXiv ID:** 2607.08892 | [PDF](https://arxiv.org/pdf/2607.08892v1)

**作者:** Mukesh Kumar `[一作]` (Georgia Institute of Technology), Panagiotis Tsiotras `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7654 | [OpenAlex ID](https://openalex.org/A5077667229)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在图形基础的多智能体追捕-逃逸游戏中，提出一种Hybrid方法 Primitive-Guided Tree Search（PGTS），先在离线阶段求解一系列小型子游戏（1v1、2v1）的Exact Nash equilibrium，随后在在线阶段利用这些预先计算的策略与价值函数指导 SM-MCTS 的搜索与叶值估计，从而在保证团队层级协作的同时控制搜索的分支与深度。

**💡 创新点**

创新点包括：① 通过将小规模子游戏的 Nash 策略作为先验来引导联合动作的采样（Primitive-guided Expansion）；② 采用 2v1 子游戏的高阶交互作为主要价值估计来源，顺序匹配两种子游戏的组合来逼近全局叶值；③ 在 SM-MCTS 里集成回报预估与线性分配（Hungarian），显著降低搜索维度与计算成本。

**🔧 技术方法**

技术手段：Shapley value iteration 与线性规划求解子游戏 Nash equilibrium；离线 PSRO/MT-PSRO 用于获取子游戏解；Simultaneous-Move MCTS（SM-MCTS）+ Regret Matching / Decoupled UCT 进行在线决策；线性分配（Hungarian algorithm）用于 2v1 与 1v1 的匹配；图形搜索和树结构维护。

**📊 数据集**

数据集：GraphChase平台提供的两张 7×7 网格图、Scotland Yard 200 节点图以及实际的 Atlanta 151 节点图；实验采用 N-Pursuer vs. 1-Evader 以及多智能体配置；同时使用真实世界网络拓扑进行评估。

**📈 对比分析**

与 MT-PSRO、NSGZero、纯子游戏分解和启发式截击策略对比。PGTS-RM/PGTS-DUCT 在所有图形上均实现更高的 Worst-Case Utility (WCU)、Shortest‑Path Worst‑Case Utility (SP‑WCU) 以及 Expected Reward (ER)。特别是在 Scotland‑Yard 与 Atlanta 复杂图形中，PGTS 在 WCU 上提升 0.5‑0.7 以上，且对不同 Evader 策略（随机、最短路径、PGTS 生成的策略）均保持稳健。

**⚠️ 局限性**

局限性：① 仅适用于完全可观测、确定性转移的零和两队游戏；② 仅考虑 1v1 与 2v1 子游戏，无法直接扩展到更大子团队或不对称信息环境；③ 对极大规模 agent 集合的可扩展性仍有挑战，搜索深度与分支因子随 agent 数量增长；④ 需要预先求解子游戏，若图形或规则变化需要重新训练。

---

## 16. Federated Low-Rank Koopman Learning for Multivariate Time-Series Anomaly Detection in IoT Systems

**arXiv ID:** 2607.08978 | [PDF](https://arxiv.org/pdf/2607.08978v1)

**作者:** Tung-Anh Nguyen `[一作]` (University of Technology Sydney), Xiaojing Huang `[通讯]` (University of Technology Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 FedKAD，一种面向资源受限 IoT 设备的联邦低秩 Koopman 异常检测框架，能够在边缘设备上本地学习时间序列的线性动态，并通过仅交换低维子空间变量完成全局协同。

**💡 创新点**

创新点：① 将 Koopman 动态模型迁移到联邦学习场景，避免了深度模型的高计算/通信成本；② 通过低秩子空间共识与 Stiefel‑ADMM 优化，保证了正交约束下的收敛；③ 提供了在部分客户端参与时的收敛与停滞分析；④ 在四大 IoT 基准上实现了性能领先且极低的边缘部署成本。

**🔧 技术方法**

技术手段：滑动窗口观测、Koopman 线性化、低秩子空间投影、Stiefel 流形上的 ADMM 优化、闭式局部最小化、联邦平均协议、阈值/段级别评价指标。

**📊 数据集**

数据集：SMD（服务器资源监控）、PSM（电子商务服务器聚合）、SMAP（土壤湿度传感）、MSL（火星探测器遥测）。

**📈 对比分析**

与 DeepSVDD、LSTM‑AE、USAD、TranAD 四种神经网络基线对比。FedKAD 在主段级 PA%K（k=0.01）指标下，在 SMD、PSM、SMAP 上取得最高 F1，MSL 仅略逊；在训练时间、通信量与推理延迟上均比所有基线快 200–2000×、缩减 3–40×通信，推理延迟与最慢基线相当或更低。

**⚠️ 局限性**

局限性：① 依赖低秩假设，对高维或极度非线性动态的适应性待验证；② 对极端非 IID、极高异常率的数据可能仍需更多调优；③ 目前仅在四个公开基准上评估，跨域泛化性未完整验证；④ 需要服务器维护子空间同步，可能在极低带宽/高延迟环境下受限。

---

## 17. Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading

**arXiv ID:** 2607.08964 | [PDF](https://arxiv.org/pdf/2607.08964v1)

**作者:** Zongxia Li `[一作]` (Tencent HY LLM Frontier), LeoweiLiang `[通讯]` (Tencent HY LLM Frontier)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个新基准“<name>”，聚焦于长时序终端任务，并通过细粒度子任务评分提供密集的进度反馈。

**💡 创新点**

创新点在于：1）将终端任务拆分为可量化子任务，解决了传统二元评估的稀疏奖励问题；2）构造了 46 个跨 9 个领域、真实工作流程的长时序容器化任务；3）提供公开评测工具与隐藏验证，确保模型泛化而非仅针对公共测试。

**🔧 技术方法**

使用了容器化终端环境、Deterministic Subtask Grader、Harbor 任务格式、Terminous-2 及 Codex 等代理主架构，以及基于 LLM 的前沿模型（如 GPT‑5.5、DeepSeek V4 Pro 等）进行评测。

**📊 数据集**

数据集由 46 个任务组成，涵盖实验复现、交互游戏、软件工程、多模态审计、科学计算等，任务通过 Docker 镜像提供完整代码、数据与工具。

**📈 对比分析**

在共享 90 分钟预算下对 15 种前沿模型进行实验：平均通过率仅 4.3%，最强 GPT‑5.5 达到 15.2%（R≥0.95）。相比传统仅评测终点成功率的基准，密集奖励揭示了更多“近乎成功”情况，表明模型在长时序执行上仍存在显著瓶颈。

**⚠️ 局限性**

局限性包括：1）对长时序执行的需求仍未得到充分满足，导致大多数模型超时或误判已完成；2）缺乏高效的自我验证与停止决策机制；3）评测主要聚焦终端命令，未覆盖更复杂交互或多模态决策场景。

---

## 18. AgenticFocus: Object-Preserving Mixed Reality Synthesis from Human FPV Video for Dexterous Humanoid Learning

**arXiv ID:** 2607.08857 | [PDF](https://arxiv.org/pdf/2607.08857v1)

**作者:** Iaroslav Kolomiets `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 932 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将普通人类第一人称视角（FPV）视频转换为可供机器人训练的同步人形机器人演示；

**💡 创新点**

三大创新：1) 基于任务相关物体的对象中心化恢复与遮挡填补；2) 在相机相对坐标系下完成全手指的人形机器人转移；3) 层级混合现实渲染，保持接触深度与物体完整几何；

**🔧 技术方法**

采用 SAM2 进行物体分割与跟踪；E2FGVI-HQ 做背景恢复；WiLoR/HaMeR 等 3D 手部重建模型；逆运动学 (IK) 在 MuJoCo 中实现手部转移；Layered compositing 结合手部与物体模板；SPARC 指标评估平滑度；

**📊 数据集**

主要使用 EPIC‑KITCHENS 的 FPV 视频以及内部收集的演示；视频统一为 30 FPS；

**📈 对比分析**

与 Masquerade 与 Do as I Do 两个跨身姿转移基线同等条件对比；在轨迹重建误差上平均误差最低，SPARC 指标（平滑度）也比两者更好（-5.18 vs -5.56 vs -6.05）；

**⚠️ 局限性**

局限：只针对相机相对框架的转移，未充分验证在不同相机高度/视角下的鲁棒性；对小物体的物理相互作用仍可能产生轻微不自然的渲染；未在下游策略训练中展示实际性能提升。

---

## 19. Proof-of-Continuity: A Temporal Model for Authority Propagation in Distributed Systems and AI Agents

**arXiv ID:** 2607.08906 | [PDF](https://arxiv.org/pdf/2607.08906v1)

**作者:** Nicola Gallo `[一作]` `[通讯]`, Nicola Gallo

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 Proof‑of‑Continuity（PoC）模型，定义了 Proof‑of‑Relationship（PoR）与 PoC 机制，用以在多跳执行链中保证权限沿链单调不扩展，从而消除传统的“confused deputy”漏洞。

**💡 创新点**

创新点在于将权限传播拆分为三层：持有（Proof‑of‑Possession）、因果关联（Proof‑of‑Relationship）与连续性（Proof‑of‑Continuity）；通过形式化证明展示了线性权限链中权威不可能从非起始位置出现，解决了权威混合与时间维度的冲突。

**🔧 技术方法**

技术手段主要是理论建模与形式化证明，构建 Provenance Identity Continuity（PIC）框架并定义 PoR 与 PoC 的公理；未涉及具体实现或密码学构造，留给后续实现工作。

**📊 数据集**

无实验数据集；论文为理论研究，未使用公开或自建数据集。

**📈 对比分析**

比较方法：通过对比传统 Proof‑of‑Possession 体系与 PoC，利用形式化证明展示后者在权限不扩展与因果连贯性方面的优势；未给出性能指标或数值实验。

**⚠️ 局限性**

局限：未实现 PoR 具体构造、未处理撤销与并行/分叉执行、未给出异构系统的策略翻译实现、未考虑线性模型之外的 DAG 结构，且未对实际系统中的性能与安全成本进行评估。

---

## 20. A Machine Learning Surrogate for Component Criticality Ranking in Interdependent Power-Communication Networks

**arXiv ID:** 2607.08918 | [PDF](https://arxiv.org/pdf/2607.08918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 21. Improved Error Bounds for Pure Differentially Private Continual Counting via Matrix Factorization

**arXiv ID:** 2607.08963 | [PDF](https://arxiv.org/pdf/2607.08963v1)

**作者:** Pavel Arkhipov `[一作]` (Institute of Science and Technology Austria), Nikita P. Kalinin `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

针对纯差分隐私的连续计数问题，提出了一种基于矩阵分解的新机制，并给出了实现该机制的高效算法。

**💡 创新点**

创新点在于：①通过梯度优化获得低维高质量分解，然后递归构造高维分解，显著降低最大平方误差和均方误差的常数项；②证明了对 {0,1} 元素分解的 Ω(log³n) 下界，表明在该自然分解类中上界已近似最优。

**🔧 技术方法**

核心技术包括：矩阵机制框架、α‑参数化的组合分解（◊ₐ 操作）、梯度与 L‑BFGS 优化、符号与浮点精度验证、以及对 Bollobás 型不等式的应用。

**📊 数据集**

该工作为理论分析，没有使用实际数据集；所有结果均通过数学证明与数值优化实验（在 1024×1024 维度上得到最优常数）得到验证。

**📈 对比分析**

与现有的 k‑ary 树 + 减法机制相比，新机制将最大平方误差常数从 0.1171 降到 0.0778（约 1/3 的改进），均方误差常数从 0.1171 降到 0.0710（约 1/1.6 的改进）。实验和理论分析表明在任意大尺寸下仍保持同样的 asymptotic 形式。

**⚠️ 局限性**

主要局限：①下界仅适用于 {0,1} 分解；对一般分解的 Ω(log³n) 下界仍未得到；②实现仍需对高维矩阵进行符号精确化，实际部署需要额外的数值稳定性保障。

---

## 22. Pareto-Optimal Scheduling in the Half-batch Multiserver-job Model

**arXiv ID:** 2607.08999 | [PDF](https://arxiv.org/pdf/2607.08999v1)

**作者:** Ziyuan Wang `[一作]` (Northwestern University), Izzy Grosof `[通讯]` (Northwestern University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并分析了半批 MSJ 模型，研究大作业的响应时间与小作业吞吐量之间的 Pareto 最优权衡。

**💡 创新点**

创新点在于给出了该双指标优化问题的完整非渐进性闭式 Pareto 前沿，证明其由 (1,0)-convoy 和 (k,n)-convoy 策略及其相邻组合生成。

**🔧 技术方法**

主要技术包括：马尔可夫链重构、阈值策略复制、凸性与互换论证以及对小作业层级的概率矩阵逆推导。

**📊 数据集**

使用理论推导与仿真相结合：参数设定为 n=10、μ₁=2、λ=0.5，采用指数分布大作业（但结果对任意分布通用）。

**📈 对比分析**

通过仿真与理论曲线对比，Pareto 最优策略显著优于多种子弹式（k,m）-convoy、固定小作业、非空闲版本等次优策略，展示了更高吞吐量与更低响应时间的折中。

**⚠️ 局限性**

局限在于假设所有大作业占用全部服务器、无抢占、单一大作业类型；若允许部分占用、多类大作业或加入抢占开销，最优策略结构可能会发生变化。

---

## 23. How are linear representations learned? Exact solutions to the dynamics of abstraction

**arXiv ID:** 2607.08843 | [PDF](https://arxiv.org/pdf/2607.08843v1)

**作者:** William W. Yang `[一作]` (University College London, United Kingdom), Peter E. Latham `[通讯]` (University College London, United Kingdom)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个关于抽象（概念向量对齐）在训练过程中动态演化的理论，并给出了线性网络的完整解析解，随后扩展到无限宽非线性网络并给出吸收律，最终通过对Transformer和视觉/语言模型的局部GELU消融实验验证理论并提升线性探针泛化。

**💡 创新点**

创新点在于：① 用梯度流和Ricatti方程给出抽象轨迹的解析形式；② 发现深度和初始化对最大/终端抽象的定量影响；③ 推导非线性激活导致的抽象衰减律；④ 提出可操作的局部GELU消融方法以提升抽象与泛化。

**🔧 技术方法**

技术方法包括：梯度流分析、矩阵Ricatti方程、2FS对称性化简、无限宽NNGP核映射、解析解和定理证明、线性探针评估与实验对比。

**📊 数据集**

使用的数据集包括：最小二元概念（shape/color）生成的四类合成数据、3dshapes、DINOv3 ViT-L/16、Gemma 4 E2B、以及Macaca mulatta的V4/IT神经记录。

**📈 对比分析**

与基线模型（未消融）对比，局部GELU消融在所有层保持或提升抽象，线性探针准确率平均提升约1个百分点；理论预测与实验结果高度一致，尤其在抽象轨迹和终端抽象的深度依赖性上。

**⚠️ 局限性**

局限性包括：对2FS对称性、无限宽、梯度流优化、以及线性读出最优假设的严格依赖；非线性网络的深层动态尚未完全解析，实验验证仍停留在小规模网络上。

---

## 24. SafeExplorer: An Unbiased Policy Gradient for Reinforcement Learning with Recovery Interventions

**arXiv ID:** 2607.08925 | [PDF](https://arxiv.org/pdf/2607.08925v1)

**作者:** Elham Daneshmand `[一作]` (McGill University), Hsiu-Chin Lin `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种在实际机器人上使用外部恢复策略的安全强化学习算法，能够显著减少训练过程中的跌倒次数。

**💡 创新点**

创新点包括：① 对混合策略（主策略+恢复策略）给出无偏梯度估计（Masked Policy Gradient）；② 在确定性动力学下给出恢复状态的闭式价值，消除批评器误差；③ 引入基于恢复成功与否的兼容性正则化，提升学习效率。

**🔧 技术方法**

使用技术主要有：PPO框架、分区掩蔽梯度、优势估计（GAE）、可学习与闭式价值融合、结果门控的行为克隆正则化。

**📊 数据集**

实验数据集为三种连续控制跑步任务：HalfCheetah、Ant、Unitree Go1（仿真和真实机器人），共使用五个随机种子。

**📈 对比分析**

与标准PPO、Recovery RL、Safe Legged、CPO、PPO-Lagrangian等基线比较，落地训练跌倒次数降低最多可达233倍（HalfCheetah），奖励保持或提升；在恢复策略不可靠的Ant环境中，该方法唯一能达到80%最优奖励阈值。

**⚠️ 局限性**

局限性：需预训练恢复策略和手工设计安全区域距离函数；闭式价值假设确定性动力学，噪声下精度下降；对恢复策略的可靠性和安全区域宽度敏感，且方法在非腿部运动任务的迁移尚未验证。

---

## 25. Adaptive Bayes exactly tracks information over intrinsic time

**arXiv ID:** 2607.08789 | [PDF](https://arxiv.org/pdf/2607.08789v1)

**作者:** Akshay Balsubramani `[一作]` `[通讯]`, Akshay Balsubramani

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一套统一的、精确的贝叶斯/乘法权重更新的路径级信息平衡公式，将在线学习的累计后悔拆解为即时信息支付、温度变化产生的漂移以及比较器的描述成本三部分，并将该框架推广到专家、模型、动作选择以及连续空间、Bandit、上下文、对齐游戏等多种情景；

**💡 创新点**

核心创新在于构造了一个可精确追踪的“内在时间”计时器，并在此基础上提出了两类自适应更新（先复温后再累积与局部递归），以及对应的二阶“平方根时钟”和“压力目标”调度；此框架将现有多种算法（Hedge、AdaHedge、Squint等）的先验、学习率、侧信息、快速率等看作同一信息经济学的不同视角；

**🔧 技术方法**

主要技术手段包括：信息论的KL散度与相对熵公式、复合损失（side‑information + 可预测补偿）映射、指数加权更新的闭式变分表示、信息平衡一阶恒等式、二阶（方差）与高阶泰勒松弛、PAC‑Bayes 变分推导、压力目标（free‑energy）一阶根搜索，以及路径级的高概率、迭代对数界和低噪快率自界；

**📊 数据集**

实验验证使用了合成的多种损失序列（周期性、随机、梯度变化等）以及少量真实在线数据（未公开具体数据集），目的是检验理论预言的路径级分解与自适应调度效果；

**📈 对比分析**

与 AdaHedge、NormalHedge、AdaNormalHedge、Squint 等基线算法在相同损失序列上进行对比，实验显示本文算法在合成场景下能够逼近理论下限并在实际数据上表现出更稳健的低回报上界；在高概率和渐近迭代对数分析上也取得与现有最优一致的或更优的界；

**⚠️ 局限性**

局限性包括：需要损失有界或指数矩存在才能使用集中与高阶松弛；自适应调度的实现复杂度较高，尤其是压力目标线搜索；快速率的自界在多元混合比较器时较弱，仅对点比较器或极端低噪环境有效；整体框架仍以理论为主，实际部署时可能需要对内在时间进行近似或估计，导致精确性下降。

---

## 26. NL-PAC: Specification Ambiguity and Certified Minimax Risk Floors in LLM-Mediated Supervision

**arXiv ID:** 2607.08961 | [PDF](https://arxiv.org/pdf/2607.08961v1)

**作者:** Berkay Anahtarci `[一作]` `[通讯]` (Ozyegin University), Berkay Anahtarci (Ozyegin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出NL-PAC框架，利用固定的LLM解码阈值定义可接受标签集合，量化模型解释歧义并给出最小化风险下限，并通过对未标记输入的统计检验实现风险上限的可证性。

**💡 创新点**

创新点在于将任务说明与监督通道的耦合转化为可观测的“重叠质量”，揭示了目标盲区所产生的不可消除学习风险底限，并给出了从无标签样本即可计算的有限样本风险上限。

**🔧 技术方法**

主要技术包括阈值化解码概率、可接受标签多重性分析、两点Le Cam论证、最优随机化策略构造、Hoeffding与Bernstein置信区间、以及在采样解码模式下的可插补（plug‑in）估计。

**📊 数据集**

实验使用冻结版Qwen 2.5‑3B模型，针对内容审核任务的两种提示（P1、P2）及精确规则对照，采样1000个未标记输入进行证书计算。

**📈 对比分析**

与传统基于标签噪声或弱监督的评估方法相比，NL-PAC能够在不需要真实标签的情况下给出风险下限，实验结果表明在P1提示下的重叠质量显著高于阈值，证书给出了非零最小化风险，而在P2与精确规则控制下则得到零风险下限。

**⚠️ 局限性**

局限性包括：需能观测或可高深度采样解码概率；仅适用于目标盲区通道；证书仅对模型-提示-阈值组合有效，未直接解决人类解释歧义；对多重阅读的覆盖度与一致性约束过宽，导致在实际应用中难以推广。

---

## 27. A Seed for Privacy -- semi-automatic privacy-revealing data reminder in databases and data streams

**arXiv ID:** 2607.08801 | [PDF](https://arxiv.org/pdf/2607.08801v1)

**作者:** He Gu `[一作]` (University of Olso), Vera Goebel `[通讯]` (University of Olso)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了半自动化工具 pArborist，利用数据模式与用户提供或法规规定的种子查询生成查询树，随后通过相关性与条件依赖过滤，自动识别并标注数据库与实时流中的隐私泄露复杂事件。

**💡 创新点**

创新点包括：① 用查询树（Growing 与 Striking）一次性枚举所有语法正确的查询；② 在树中通过相关性/条件依赖检验，自动剔除非隐私查询；③ 结合 GDPR 作为默认种子支持无专业知识用户；④ 在流处理场景实现平均 1.3 ms 延迟，满足实时需求。

**🔧 技术方法**

技术手段：A‑F（Attribute‑Function）对抽象、查询树构造与剪枝算法、相关系数（Kappa、ICC、tau）与置信区间评估、Bootstrap 与增量更新策略、C++/Python 实现，配合数据库/流接口。

**📊 数据集**

使用的真实数据集有：Election（约 2000 万条选民记录）、Check‑in（约 1000 万条酒店/餐厅签到记录）和 APP（约 5 万条用户个人信息），用于评估 recall、precision、效率与实时性能。

**📈 对比分析**

与 FQID 进行对比实验，pArborist 在 Recall 超过 90%，Precision 超过 93%，并在数据库场景平均 500 ms、流场景 1.3 ms（warm‑up 920 ms）内完成查询生成，明显优于 FQID 仅关注 quasi‑identifier 的性能。

**⚠️ 局限性**

局限性：① 查询树的组合爆炸仍导致空间/时间消耗上升，尤其在极高复杂度或稀有事件上；② 相关性/条件依赖评估需要足够数据，稀缺数据时置信区间不可靠；③ 对用户提供种子或法规的依赖，若种子错误或不完整，检测覆盖度下降；④ 对分布漂移的流场景尚未完全自适应，需要进一步强化模型与强化学习集成。

---

## 28. Multi-Conditioned Diffusion Synthesis of Sand Boils for Low-Resource Earthen-Levee Inspection

**arXiv ID:** 2607.08794 | [PDF](https://arxiv.org/pdf/2607.08794v1)

**作者:** Padam Jung Thapa `[一作]` (University of Louisiana at Lafayette), Md Tamjidul Hoque `[通讯]` (Louisiana State University New Orleans)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对缺乏标注样本的沙沸检测任务，本文提出一种基于扩散模型的合成管线，利用 DreamBooth LoRA 适配 SDXL、ControlNet 多分支结构和软掩模隐式融合技术，生成既保持真实缺陷像素又能丰富背景多样性的合成图像，并通过 Prompt Atlas 自动构造多样化文本提示，最终产生带单一可追溯标注的增强数据集。

**💡 创新点**

创新点包括：
1) 软掩模隐式融合，解决传统 Poisson 复制的边缘裂痕与色彩漂移问题；
2) 形状自适应软掩模几何，可根据缺陷尺寸自动调整保留与过渡区域；
3) Mask‑conditioned ControlNet 直接将掩模作为标签生成，提供零标注的标签可信度；
4) Prompt Atlas 通过 JSON 规范自动化扩展提示词，兼容多种缺陷类别；
5) 基于留一 CLIP 相似度的自动过滤器同时抑制 OOD 与记忆化样本。

**🔧 技术方法**

核心技术：Stable Diffusion XL（SDXL）+ DreamBooth LoRA 微调、ControlNet 四分支（Canny、深度、HED 软边、表面法线）、IP‑Adapter 风格锚定、软掩模隐式融合、Convex Hull Annotator（自动标注与验证）、Prompt Atlas（提示生成与 CLIP 验证）、留一 CLIP 相似度过滤器。

**📊 数据集**

数据集：美国陆军工程兵团（USACE） levee 检测照片，挑选 199 张带标注的沙沸图像做训练，39 张高质量参考图做 DreamBooth LoRA 训练与控制网络初始化，随后在 1,020 条合成候选中通过 CLIP 过滤得到 815 条可用增强样本。

**📈 对比分析**

评估方法：
- 生成图像与真实参考集的 FID（两种实现）与 KID；
- CLIP 文本‑图像相似度；
- LPIPS 组内多样性；
- 精度/召回率、密度/覆盖度等分解指标；
- Mask‑conditioned ControlNet 的标签漂移 IoU。
结果：V4 预设在保持真实缺陷像素、低 FID/高 CLIP 的平衡下被选为增强集生成器；MaskCN 在图像质量与多样性上表现最好，但其标签可靠性受限于现有门控模型，导致无法正式采用。

**⚠️ 局限性**

局限性：
1) Mask‑conditioned ControlNet 的标签无法通过现有 Convex Hull Annotator 完全验证，存在标签漂移；
2) 仅对沙沸进行实验，尚未在下游分割模型上验证提升效果；
3) 软掩模与 ControlNet 的配置仍需针对不同缺陷类手动调节；
4) CLIP 过滤虽然能排除 OOD 与记忆化样本，但对极端域外场景的鲁棒性仍待评估。

---

## 29. Keyless Covert Communication Over Quantum MACs with General Message Sets

**arXiv ID:** 2607.08898 | [PDF](https://arxiv.org/pdf/2607.08898v1)

**作者:** Hassan ZivariFard `[一作]` (Columbia University), Xiaodong Wang `[通讯]` (Columbia University)

**通讯引用:** 65868 | [OpenAlex ID](https://openalex.org/A5100382645)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了多用户信道(MAC)中的隐蔽通信，提出一种支持多消息集的正比率传输方案，并给出了可行率区域与在经典-量子信道中的最优性证明。

**💡 创新点**

创新点在于引入消息图结构实现多用户共享消息的协同编码，突破传统隐蔽通信低速率限制，实现了正比率传输，并证明在量子-经典模型下该方案是最优的。

**🔧 技术方法**

采用随机码书构造、同时抠位（pinching）技术、Hayashi–Nagaoka误码分析、量子相对熵与信息理论工具，以及连续变量光子熵计算等技术。

**📊 数据集**

未使用实际数据集，主要基于理论模型，包括有限维离散通道、Gaussian信道与单模bosonic通道。

**📈 对比分析**

通过与已有隐蔽通信理论（如平方根定律）的对比，证明该方案在经典-量子信道中可实现正比率，在Gaussian与bosonic通道中给出隐蔽容量下界，性能显著优于传统低速率方案。

**⚠️ 局限性**

局限性包括：对连续变量 bosonic 信道的完备逆推未完成；仅在特定信道结构（如半透明双射器）下证明；对噪声不为满秩时单字母化简失效。

---

## 30. GATS: Graph-Augmented Tree Search with Layered World Models for Efficient Agent Planning

**arXiv ID:** 2607.08894 | [PDF](https://arxiv.org/pdf/2607.08894v1)

**作者:** Maureese Williams `[一作]`, Dymitr Nowicki `[通讯]` (Institute for Cybernetics of National Academy of Sciences of Ukraine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 GATS（Graph-Augmented Tree Search）框架，利用分层世界模型和 UCB1 树搜索实现多步规划；

**💡 创新点**

创新点在于将符号匹配、统计学习与 LLM 预测三层结合，消除推理时 LLM 调用；通过系统的 UCB1 搜索取代 LLM 随机采样，获得确定性、可重复的计划；

**🔧 技术方法**

使用分层世界模型（L1 符号匹配、L2 统计学习、L3 LLM 预测）、UCB1 选择、BFS 价值估计、图式记忆与 LLM 结果缓存等技术；

**📊 数据集**

在 100 个合成多步规划任务、12 类压力测试共 120 任务以及 API‑Bank Level 1/2/3 等数据集上进行评估；

**📈 对比分析**

与 Greedy（oracle）、ReAct、LATS 进行对比；在合成任务中 GATS 100% 成功率、LLM 调用 0，LATS 92%，ReAct 64%；在压力测试中 GATS 100% vs LATS 88.9% vs ReAct 23.9%；表现出显著的性能提升与零方差；

**⚠️ 局限性**

局限在于依赖已知动作规范；在开放域缺失规范时会更多依赖 LLM，效率优势下降；BFS 价值估计在大动作空间或深度时复杂；实验主要基于合成任务，需进一步验证真实 API 场景。

---

## 31. DaDaDa: A Dataset for Data Pricing in Data Marketplaces

**arXiv ID:** 2607.08785 | [PDF](https://arxiv.org/pdf/2607.08785v1)

**作者:** Qiheng Sun `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8811 | [OpenAlex ID](https://openalex.org/A5020630816)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了第一个面向数据产品定价的公开数据集 DaDaDa，并基于此数据集构建了定价、分类和检索三大任务的基准实验。

**💡 创新点**

创新点在于：①整合 9 大主流数据市场的 16,147 条产品信息，统一了元数据格式；②首次公开提供真实定价样本，为数据市场的定价模型训练与评估提供基准；③同时支持多任务（定价回归、分类、多语言检索），实现跨市场的比较与迁移研究。

**🔧 技术方法**

技术手段包括：网页爬虫（Selenium、XPath/CSS 解析）抓取原始数据；数据清洗与统一字段映射（手工对齐 AWS 类目、币种转换、文本嵌入）；使用 XLM‑RoBERTa‑Large 进行文本嵌入并 PCA 降维；多种机器学习模型（XGBoost、GB、RF、kNN、DNN）进行定价回归；mBERT 与 XLM‑RoBERTa‑Large 进行多语言分类；Elasticsearch 搭建垂直检索原型；SHAP 解释模型特征重要性。

**📊 数据集**

使用的数据集为 DaDaDa，包含 16,147 条来自 9 个数据市场（如 AWS Data Exchange、Snowflake、Databricks、Datarade 等）的产品元数据及公开定价信息。

**📈 对比分析**

在定价任务上，XGBoost 取得最高 R²≈0.833、MAE≈0.805，优于其他模型；分类任务中 XLM‑RoBERTa‑Large 的宏观 F1≈0.82，mBERT 约为 0.80；检索原型通过全文+关键词+范围过滤实现跨市场的高效搜索，实验显示检索结果与原始页面一致且覆盖面广。

**⚠️ 局限性**

局限性包括：仅采集公开的列表价格，无法反映实际交易折扣或定制合同；元数据信息不完整（部分字段缺失需默认值）；模型在未见市场（如 Snowflake）上泛化仍有一定误差；时间维度缺失，无法追踪价格与特征随时间变化的动态；对谈判型产品仅标记为 0，未能直接用于定价预测。

---

## 32. SiFAR: Synchronization-Free All-Reduce for Low-Latency LLM Inference

**arXiv ID:** 2607.08973 | [PDF](https://arxiv.org/pdf/2607.08973v1)

**作者:** Hritvik Taneja `[一作]` (Georgia Institute Of Technology), Moinuddin Qureshi `[通讯]` (Georgia Institute Of Technology)

**通讯引用:** 4032 | [OpenAlex ID](https://openalex.org/A5082772077)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对低延迟LLM推理中的All-Reduce同步瓶颈，提出一种同步自由All-Reduce（SiFAR）框架，显著降低每个token生成延迟。

**💡 创新点**

创新点在于三大技术：①冗余拉取（redundant pull）利用交换机内置归约降低数据传输；②双缓冲消除下限Barrier；③投机归约（speculative reduction）通过验证机制消除上限Barrier。

**🔧 技术方法**

主要技术包括：NVIDIA NVSwitch的in-switch归约指令（ld_reduce），Megakernels融合前向推理，CUDA Graphs和Megakernel的自动内存管理，验证标记的快速校验与重试。

**📊 数据集**

在Llama‑3.1‑8B（密集Transformer）和Qwen3.5‑397B‑17B（Mixture‑of‑Experts）两大模型上进行评估，使用单请求长度为1000、最大16K、批量1至4、TP=2/4/8的多种配置。

**📈 对比分析**

相较于最优的oneshot/​twoshot、TRT‑LLM、Lamport push‑based、NCCL 2.30等基线，SiFAR在TP=8时对Llama‑3.1‑8B实现52%延迟下降、15.2%吞吐提升；对Qwen3.5‑397B‑17B实现48%延迟下降、12.2%吞吐提升，且对尾部延迟和大输入长度仍保持显著加速。

**⚠️ 局限性**

主要局限：需依赖支持in-switch归约的NVSwitch硬件；投机归约假设GPU进度单调，在极端异步或抢占情况下可能失效；双缓冲占用额外HBM，虽然对小批量几乎无影响，但在极大模型或多实例场景下可能影响内存利用。

---

## 33. Director: Accelerating Distributed MoE Serving via Online Proactive Expert Placement

**arXiv ID:** 2607.08782 | [PDF](https://arxiv.org/pdf/2607.08782v1)

**作者:** Qianli Liu `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 21156 | [OpenAlex ID](https://openalex.org/A5043464306)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Director，一种在线主动专家放置的分布式MoE推理系统。

**💡 创新点**

创新点在于基于预测的主动放置、轻量级级联/量化预测器、计算重叠迁移以及基于LP松弛的多项式时间放置优化器。

**🔧 技术方法**

使用了专家级联预测器、低位量化MoE副本、计算重叠迁移、LP松弛与迭代逼近放置优化、随机逼近等技术。

**📊 数据集**

评测了Mistral 8×7B、DeepSeekMoE-16B、DeepSeek-V2-Lite、Qwen3-30B-A3B等四个MoE模型，使用MATH500、WikiText-103、Live Code Bench等数据集。

**📈 对比分析**

与Vanilla、Offline、Online Reactive等基线对比，端到端延迟降低11–55%，吞吐量提升且稳定性更佳。

**⚠️ 局限性**

局限性包括预测误差对性能的影响、迁移开销、对极端负载变化的适应性，以及仅在专家并行维度验证，未来需推广到其他并行维度。

---

## 34. Ortho2CAD: 3D CAD generation from orthographic drawings using vision language models

**arXiv ID:** 2607.08891 | [PDF](https://arxiv.org/pdf/2607.08891v1)

**作者:** Aditya Joglekar `[一作]` (Carnegie Mellon University), Levent Burak Kara `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2304 | [OpenAlex ID](https://openalex.org/A5048339797)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 Ortho2CAD 系统，利用视觉语言模型将二维正投影工程图直接翻译为 CadQuery 代码，从而生成可编辑的三维 CAD 模型。

**💡 创新点**

创新点包括：①在无 CadQuery 代码标注的情况下使用基于几何交互的强化学习优化；②发布可生成 150k+ 标注正投影图的自动化数据生成脚本；③在同一任务上首次将强大的 VLM（Qwen3VL）与基于 IoU 的 RL 结合，显著提升代码有效率和几何精度。

**🔧 技术方法**

采用 Qwen3VL‑8B‑Instruct 作为 VLM 基座，进行有监督微调（SFT）以及使用 Dr. GRPO 的序列级强化学习（RL）来优化 CadQuery 代码生成；数据生成利用 pythonOCC 渲染第一角正投影图并标注关键尺寸。

**📊 数据集**

主要使用 DeepCAD（含 GenCAD‑Code 代码标注）和 Fusion 360 Reconstruction（仅有 STEP 目标）作为训练与评估数据；此外自行生成超过 150k 张带隐藏线与尺寸标注的正投影图。

**📈 对比分析**

在 DeepCAD 上与 CAD‑Coder、Qwen3VL、GPT‑5.2 等基线对比，Ortho2CAD 代码有效率 100% 及 IoU 0.7922，超过 CAD‑Coder 的 0.7361；在 Fusion 360 上，RL 版 Ortho2CAD 的 IoU 0.5601 超过 GPT‑5.2 的 0.5181，展示了 RL 在无标注情境下的显著提升。

**⚠️ 局限性**

局限性包括：①仅覆盖草图-拉伸两种基本操作，缺少更复杂的 CAD 操作；②RL 奖励仅基于 IoU，未考虑图形相似度与尺寸一致性；③缺乏迭代细化机制，单次生成可能不满足工程迭代需求；④需更丰富多样的 CadQuery 代码库来进一步提升 SFT 效果。

---

## 35. QMA Lower Bounds for Batch Verification via Approximate Degree

**arXiv ID:** 2607.08888 | [PDF](https://arxiv.org/pdf/2607.08888v1)

**作者:** Mark Bun `[一作]` (Boston University), Samuel King `[通讯]` (Georgetown University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了QMA查询和通信复杂度中的批量验证问题，旨在理解验证m个布尔函数f的资源需求如何依赖于m。

**💡 创新点**

提出了一种通用技术，用于证明批量验证函数f所需的见证查询权衡的下界，并展示了即使在尝试节省常数因子的情况下，查询成本也会大幅增加。

**🔧 技术方法**

使用了量子Merlin-Arthur (QMA)模型中的查询和通信技术，结合了Grover搜索算法。

**📊 数据集**

应用于显式的DNF公式族f，并获得了对一次读取CNF公式和映射性及k元素不同函数的新下界。

**📈 对比分析**

通过与基线方法的比较，展示了在见证长度减少的情况下，查询复杂度会显著增加，具体为Ω(n^(1-δ)√(m/w))的查询需求。

**⚠️ 局限性**

限制在于所研究的函数f的结构和近似度，可能无法适用于所有类型的布尔函数。

---

## 36. Sticky Routing: Training MoE Models for Memory-Efficient Inference

**arXiv ID:** 2607.08780 | [PDF](https://arxiv.org/pdf/2607.08780v1)

**作者:** Ali Kayyam `[一作]` `[通讯]` (BrainChip Inc.), Ali Kayyam (BrainChip Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在MoE预训练阶段引入可微分的路由一致性损失，直接惩罚相邻token的专家切换，从而提升推理时的时序局部性。

**💡 创新点**

创新点在于将路由局部性作为训练目标而非后处理或架构改动，能在模型训练初期就同时优化专家表示与路由决策，显著降低缓存缺失率。

**🔧 技术方法**

采用的技术包括：基于门向量的L2一致性损失、标准的负对数似然（CE）和负载平衡损失、可选的软‑硬一致性变体，以及常规的GPT‑style Transformer MoE结构和AdamW优化器。

**📊 数据集**

实验数据集为WikiText‑2的原始字符级BPE分词数据，使用了GPT‑2 BPE词表，共计约57k词表。

**📈 对比分析**

与基线MoE、ReMoE、Oracle‑MoE、Hard‑Window等对比，本文在小模型下最多可将专家切换率降低59%，缓存命中率提升至3.92×，同时在中型模型上实现了0.4%以内的困惑度提升，表明在质量与本地性之间实现了Pareto最优。

**⚠️ 局限性**

局限性包括：对低层(L0)的局部性提升有限，统一切换惩罚无法区分语义边界与内部切换，且在更大规模模型上实验尚未验证，未来需考虑边界感知一致性和跨层专家共享等改进。

---

## 37. HERO: A Heterogeneity-Aware Benchmark Library for Federated Continual Learning

**arXiv ID:** 2607.08784 | [PDF](https://arxiv.org/pdf/2607.08784v1)

**作者:** Thinh T. H. Nguyen `[一作]` (VinUniversity), Kok-Seng Wong `[通讯]` (VinUniversity)

**通讯引用:** 1151 | [OpenAlex ID](https://openalex.org/A5059359253)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HERO 这一可重用的 Federated Continual Learning（FCL）基准库，拆分任务划分、客户端数据划分和客户端任务序列，提供可配置的异质性控制；

**💡 创新点**

创新点在于将三种常见的异质性因素（任务划分、数据不均、任务顺序）解耦，统一流生成与方法实现与报告，支持多指标评估并对不同场景（图像、图结构）实现可扩展性；

**🔧 技术方法**

采用基于 Dirichlet 分布控制客户端数据 skew（α）和基于参数 ρ 控制任务顺序不匹配的流生成，使用多种 FCL 方法（FedAvg、GLFC、MFCL、TARGET、LANDER、Re‑Fed+、TagFed、LGA、FedCBDR）以及 AP 评估的图结构方法（POWER、MOTION）进行实验；

**📊 数据集**

核心实验使用 CIFAR‑100 与 TinyImageNet 两个图像数据集；可扩展实验使用 OGB‑MolPCBA 的分子图结构数据；

**📈 对比分析**

通过在 3×3 α‑ρ 网格下计算平均准确率（AFA）、平均遗忘（AF）和底部 10% 客户端准确率（B10），发现 FedCBDR 在同步易设置下性能最佳，而 TagFed、LGA 在高异质性（α↓ 或 ρ↑）下在 B10 上表现更稳健；在 OGB‑MolPCBA 中，随着 scaffold 领域粒度增大 AP 降低且遗忘上升，POWER、MOTION 在异质性下更稳定；

**⚠️ 局限性**

局限性包括：核心仅覆盖图像 FCIL；固定客户端进度且未考虑异步或缺失任务；未提供正式的隐私、通信或能源评估；以及对不同模态、任务类型的覆盖仍有限。

---

## 38. Signed Symmetric Quantization for Few-Bit Integers

**arXiv ID:** 2607.08779 | [PDF](https://arxiv.org/pdf/2607.08779v1)

**作者:** Ian Colbert `[一作]` (Advanced Micro Devices), Arun Ramachandran `[通讯]` (Advanced Micro Devices)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了符号对称量化（signed symmetric quantization），允许量化尺度（scale）为正或负，从而利用符号整数字母表中多出的负端点；

**💡 创新点**

创新点在于将量化尺度的符号作为可变参数，通过简单的闭式符号选择规则（将最大绝对值对应的端点映射到额外负端点），在保持对称量化的推理路径下显著降低了低比特量化误差；

**🔧 技术方法**

主要技术包括理论误差分解、最坏情况 ℓ₂ 误差上界、条件最佳符号选择证明，以及与非对称量化的等价性证明；在实验中使用了 Qwen3、Qwen3.5、Llama3 等大语言模型；

**📊 数据集**

实验数据集包括 WikiText2（评估 perplexity）、LightEval（ARC、HellaSwag、PIQA、WinoGrande）以及 MMLU-Redux、GSM8K 等下游任务；

**📈 对比分析**

与传统正尺度对称量化和非对称量化进行对比，结果显示在 2/3/4 位精度下，符号对称量化在 WikiText2 perplexity、few‑shot准确率上均有提升（尤其 2 位显著下降），且保持与对称量化相同的内存和吞吐优势；

**⚠️ 局限性**

局限性包括：只针对无数据的每组权重量化，未考虑更复杂的量化粒度或卷积操作；并且在极低精度下仍无法完全达到非对称量化的性能，需要进一步结合数据感知的旋转和网格学习来提升。

---

## 39. Benchmarking Large Language Models on Repairing Qiskit Programs using Bugs4Q

**arXiv ID:** 2607.09007 | [PDF](https://arxiv.org/pdf/2607.09007v1)

**作者:** Saumya Brahmbhatt `[一作]` (University of Maryland, Baltimore County), Lei Zhang `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 7287 | [OpenAlex ID](https://openalex.org/A5100364094)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Bugs4Q量化的67个真实Qiskit缺陷进行版本锁定验证，并在六个Qiskit发布版本上评估四个GPT模型的自动程序修复能力。

**💡 创新点**

首次发现量化基准的有效性随Qiskit版本变化，提出版本锁定验证协议并量化环境不兼容导致的“伪成功”。

**🔧 技术方法**

采用LLM自动修复生成、pass@k评估、可执行性分类、基于语义的测试用例构造与版本对齐的执行框架。

**📊 数据集**

利用融合Bugs4Q-Framework与Bugs4Q-NA的去重后67条真实缺陷，并为缺失测试编写MUT型断言。

**📈 对比分析**

在每个版本下计算pass@10和可执行率，结果显示GPT-5.4最高达48.8% pass@10，且性能在0.45.0上峰值后随1.0.0过渡显著下降。

**⚠️ 局限性**

评估仅覆盖Qiskit缺陷、四款GPT模型及六个固定版本，缺陷多为API迁移问题，结果可能不适用于其他框架或中间版本。

---

## 40. LLM-Driven Evolutionary Generation of Multi-Objective Bayesian Optimization Algorithms

**arXiv ID:** 2607.08791 | [PDF](https://arxiv.org/pdf/2607.08791v1)

**作者:** Georgios Laskaris `[一作]` (Terra Quantum AG), Florian Neukart `[通讯]` (Terra Quantum AG)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用大语言模型驱动的进化搜索自动生成多目标贝叶斯优化算法，并在合成与真实工程问题上进行评估。

**💡 创新点**

首次将LLaMEA框架扩展到多目标贝叶斯优化领域，结合SMAC超参优化与LLM变异/交叉，得到既高效又能快速收敛的算法。

**🔧 技术方法**

技术核心包括大语言模型（Gemini‑2.5‑flash）作为变异/交叉算子、SMAC多目标超参优化、演化策略（(1+1)-ES、(4+16)-ES、(8,16)-ES）以及基准多目标BO实现（qParEGO）和经典进化算法。

**📊 数据集**

使用十二个合成基准（ZDT、DTLZ、WFG）和三个未见真实工程基准（RE21、RE34、RE37）。

**📈 对比分析**

对比方法包括qParEGO、IOC‑SAMO‑COBRA、NSGA‑II、NSGA‑III、随机搜索等，生成算法在合成基准上平均归一化超体积最高（0.971），比qParEGO高 7/12 问题且运行时间约 60 倍更快；在真实基准上，Improved‑Scalarized‑EI 超越 qParEGO 并仅耗 3.4 倍时间。

**⚠️ 局限性**

局限性：仅覆盖无约束、多目标问题；评估预算固定为 400 次评价，可能不适用于更大规模或噪声/约束情形；LLM 生成算法可能存在确定性或实现错误，需要人工校正；实验规模受算力限制，缺乏更广泛的真实问题验证。

---

## 41. A Formalization of the Mean-Field Derivation of the Vlasov Equation: AI-Assisted Lean Formalization as a Strategy Game

**arXiv ID:** 2607.08986 | [PDF](https://arxiv.org/pdf/2607.08986v1)

**作者:** Joseph K. Miller `[一作]` `[通讯]` (Stanford University), Joseph K. Miller (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在 Lean 4 证明助手中完成了 Dobrushin 的 Vlasov 方程无穷体场理论的完整形式化，并提取出可复用的最优传输层。

**💡 创新点**

创新点在于：① 将数学家对 AI 的“指导”形式化为“formalization game”，并通过四项机器可检验的可重用性检查（逆向依赖、独立编译、linter 清洁、无冗余）证明了层的自包含性；② 通过人类主导的定义锁定、分解与库缺口 triage，展示了在 AI 辅助下快速且可靠的形式化流程。

**🔧 技术方法**

使用技术：Lean 4 与 Mathlib 4.29.1，Claude 语言模型（Opus 4.7/4.8）驱动证明自动化；自定义“standing instruction file”记录策略；构造了 Wasserstein‑1 的双面定义与 Kantorovich–Rubinstein 桥接，完成稳定性估计与弱→Lagrangian 超位置原理。

**📊 数据集**

该工作不依赖传统机器学习数据集，而是以 Dobrushin 的手稿及其证明步骤为“训练材料”，通过 Git commit 历史记录体现进展。

**📈 对比分析**

比较方法：与传统人工形式化（如“Liquid Tensor Experiment”）对比，完成主要定理所需时间从数周缩短至约一月；所有目标定理在无 sorry、无自定义 axiom 的前提下通过 Lean 的编译与 axiom‑footprint 检查，说明方法可靠且高效。

**⚠️ 局限性**

局限性：① 仅对单一研究方向（Vlasov 方程）验证；② 对某些库缺口（如 Polish‑space 完备性）采用占位符而未实现；③ 结果依赖特定 AI 模型版本与 Mathlib 版本，未来工具升级可能需重跑；④ 评价仅基于单一次实验，缺乏广泛统计。

---

## 42. Loop-Based Slicing and Input-Driven Concretization: An Empirical Study of Termination and Non-Termination Analysis

**arXiv ID:** 2607.08988 | [PDF](https://arxiv.org/pdf/2607.08988v1)

**作者:** Negar Fathi `[一作]` (University of Nebraska--Lincoln), Hiroshi Unno `[通讯]` (Tohoku University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套工具无关的 C 源级预处理框架（Loop-Based Slicing 与 Input-Driven Concretization），并在六种现有终止/非终止分析器上做了系统的实证研究。

**💡 创新点**

创新点在于：①以循环为切片目标的“loop‑based slicing”能够在保持终止行为的前提下局部化分析任务；②“input‑driven concretization”通过将非确定性输入固定为具体实例来缩小分析范围；③三种预处理配置（无、仅切片、仅 concretization、切片+concretization）与六个分析器交叉验证，揭示预处理对不同工具、任务和代码特征的可变影响。

**🔧 技术方法**

技术手段包括：Clang LibTooling 进行源代码转化；Frama‑C 进行循环切片；C‑to‑C 变换实现 concretization；算法 1 用于聚合变体级结果；六个成熟终止/非终止分析器（Athena、Proton、UAutomizer、AProVE、CPAchecker、2LS）。

**📊 数据集**

数据集采用 Shi 等人整理的 117 份简化 C/C++ 程序（56 个真实非终止 bug 及其 61 个修复版本），涵盖指针、数组、位运算、递归等多种真实语义特征。

**📈 对比分析**

比较方法：在四种预处理配置下对每个分析器执行 6 × 4 × 117 = 2808 次分析，记录结果（yes/no/unknown/timeout/error）、时长，并按程序级与变体级指标（ratio_T、ratio_NT、TVT、AVT、MVT）评估准确性、互补性、局部化效果、特征敏感度和效率。实验发现：预处理在某些分析器/任务上可提升 10–30 % 的正确率，特别是含 concretization 的配置，但整体提升并不统一；总时长 TVT 在含 concretization 配置下显著上升，单变体时长 AVT 变化不一；在特定代码特征（整数溢出、位运算）下效果更好。

**⚠️ 局限性**

局限性：①预处理对不同分析器的增益高度依赖，未必能在所有场景中获益；②concretization 为下近似，仅覆盖选定输入场景，可能导致错误结论或遗漏真正的非终止行为；③实验仅限于 Shi 等人提供的 117 份基准，未检验更大规模或其它语言特性的程序；④未支持递归调用的切片；⑤输入实例的生成方式人工选择，未自动化；⑥结果对工具的内部实现细节敏感，可能在其它实现版本下表现不同。

---

## 43. Eluna: An Agentic LLM System for Automating Warehouse Operations with Reasoning and Task Execution

**arXiv ID:** 2607.08960 | [PDF](https://arxiv.org/pdf/2607.08960v1)

**作者:** Ning Liu `[一作]` (Amazon.com, Inc.), Shervin Malmasi `[通讯]` (Amazon.com, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种面向仓储运营的图引导多代理框架，用以可靠执行标准操作程序（SOP）并在生产环境中落地；

**💡 创新点**

创新点在于将SOP编码为有向无环图并实现渐进式曝光、并行子代理委派，以及采用异步经验蒸馏的训练管线，从而显著提升LLM对流程的遵循度和执行效率；

**🔧 技术方法**

主要技术包括Graph‑guided skill‑based agent、CodeAct 代码解释器、MCP 实时数据访问、层次化多代理架构、异步经验蒸馏与 LoRA 微调；

**📊 数据集**

使用了8,525条SOP执行轨迹进行训练，并在13类任务基准、1,500个机器人输送案例以及410个库存票据案例上进行评估；

**📈 对比分析**

与 GLM‑4.5‑Air 等 OTS 基线相比，模型在 13 任务基准上平均提升约 6%，在机器人输送任务中根因与动作的精度提升至 84%/165%，在票据处理任务中实现 94.4% 的人类一致率，且速度比人工处理快约 300%；

**⚠️ 局限性**

局限性包括仅在两类仓储场景验证，需依赖大型教师模型和昂贵标注，且跨域迁移和数据质量依赖尚未充分验证。

---

## 44. Training, Reading, and Editing Legible Transformers

**arXiv ID:** 2607.08946 | [PDF](https://arxiv.org/pdf/2607.08946v1)

**作者:** Mark Oskin `[一作]` `[通讯]` (University of Washington), Mark Oskin (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种可解释的Transformer，通过把前向层和注意力值层的运算设计为可读的有界模糊集合运算，并通过新的训练目标保证这些运算在训练过程中保持清晰和有用，从而实现模型内部的可读性与可编辑性。

**💡 创新点**

创新点在于：① 用了一个“方差下限”损失来修复传统脉冲（crispness）惩罚导致的常数坍塌问题；② 通过学习每个单元的门控参数（alpha）自动取舍可读与非可读运算，取代了手工保留的常规运算单元；③ 引入了跨单元去相关损失作为一个可调“可读性拨盘”，可将网络从共享（fan‑out）模式转为独立（fan‑in）模式，提升编辑性而不损失质量。

**🔧 技术方法**

使用的技术包括：有界sigmoid运算、模糊集合操作（∩、−）、脉冲惩罚、方差下限正则化、门控门（learned α）、跨单元去相关损失、层旋转（tuned lens）读写框架，以及常见的logit直接归因做解释。

**📊 数据集**

实验数据集为公开Web文本（Open‑Web corpus），并使用标准基准评测：Perplexity（dev集）、LAMBADA（语言模型理解）与BLiMP（句法能力）。

**📈 对比分析**

与传统的GELU Transformer做对比，经过一个epoch训练后，模型在perplexity、LAMBADA和BLiMP上与基准几乎保持相同甚至略有提升；在可读性指标上，78%的前向运算和50%的注意力运算在深层达到了“脉冲且有上下文”的高比例，远超传统模型；编辑实验表明，脉冲单元的编辑更局部、损失更小。

**⚠️ 局限性**

局限性包括：实验仅在单一随机种子和125M参数规模下完成；完全可读的层仅在保持约一半可读运算时稳定，超过此比例会出现训练不稳定；检测与命名之间仍存在显著差距（检测更常见但难以直接命名）；未评估在OOD或压缩/量化等条件下的泛化与压缩代价；并且对可读性拨盘的效果尚未在更大规模或更严格的压缩约束下验证。

---

## 45. CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding

**arXiv ID:** 2607.08974 | [PDF](https://arxiv.org/pdf/2607.08974v1)

**作者:** Yuri Ishitoya `[一作]` (Ochanomizu University), Mai Nishimura `[通讯]` (OMRON SINIC X Corp)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出CLAP，将预训练VLM通过在输出序列前添加自然语言动作描述来直接转化为可执行的VLA。

**💡 创新点**

仅改变输出表示，保留VLM架构和训练目标，解决输出分布失配问题，实现轻量化、透明的VLA微调。

**🔧 技术方法**

使用自动回归语言+动作生成、模板化语言动作前缀、可选动作遮掩增强等技术，基于Qwen3.5大模型及其Tokenization。

**📊 数据集**

使用LIBERO、LIBERO-PRO以及VLABench等机器人演示数据集进行训练与评估。

**📈 对比分析**

在与匹配基线VLA-0以及SmolVLA、OpenVLA、π_0.5等同一backbone、数据、训练步数下对比，2B CLAP单轮训练在LIBERO上达90.8%成功率，显著超越VLA-0 (+14.9pt)，并在OOD场景提升5–11pt。

**⚠️ 局限性**

仅在单一VLM家族验证，未评估其他backbone或真实环境，推理时自回归生成速度慢；缺乏并行解码或更复杂动作表示。

---

## 46. Shadow-Based Noise Fingerprinting of Simulated Quantum Noise Models

**arXiv ID:** 2607.08998 | [PDF](https://arxiv.org/pdf/2607.08998v1)

**作者:** Vridhi Jain `[一作]` (University of Delaware), Lei Zhang `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 7287 | [OpenAlex ID](https://openalex.org/A5100364094)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于经典影子 (classical shadow) 的噪声指纹识别流程，使用固定的 3 量子比特探针电路和物理信息驱动的 279 维特征向量，对模拟的十种 Qiskit 噪声模型进行分类。

**💡 创新点**

创新点包括：① 将经典影子与物理信息提取相结合，构造专门针对噪声差异的特征；② 设计了包含 QAOA 结构与简单基态的探针组合，显著提升了对相似噪声通道的辨别能力；③ 在此框架下对树模型与神经网络进行系统评估，展示树模型在高维结构化特征上的优势。

**🔧 技术方法**

使用技术包括：经典影子测量 (randomized Pauli measurements)、QAOA 及基础量子态作为探针、物理信息特征工程、随机森林 (Random Forest)、极端随机树 (Extra Trees) 与多层感知器 (MLP) 三种分类器，实验环境为 Qiskit 模拟。

**📊 数据集**

数据集：共 14,000 条样本，等量分布在十种噪声模型中（每类 1,400 条），噪声强度均匀采样于 [0.01, 0.15]，每条样本对应 3 量子比特探针电路的 279 维特征。

**📈 对比分析**

对比方法：随机森林、极端随机树、MLP；性能结果为：随机森林准确率 0.8426、宏 F1 0.8437；极端随机树准确率 0.8406、宏 F1 0.8416；MLP 准确率 0.7925、宏 F1 0.7924。混淆矩阵显示多数噪声类型分类准确，误差集中在物理机制相近的通道之间。

**⚠️ 局限性**

局限性：① 仅针对 3 量子比特，扩展到更大规模时特征可扩展性未知；② 仅完成离散噪声类型分类，未实现噪声参数估计；③ 部分特征冗余，需进一步优化；④ 仅在模拟环境测试，未评估真实硬件的射线噪声、射线相关性及有限投影误差对性能的影响；⑤ 探针集可能不足以充分激发某些噪声通道的特征。

---

## 47. Breaking Local-Minimum Traps in Spiking Neural Network-Based Solvers for CSPs via Parallel Tempering

**arXiv ID:** 2607.08897 | [PDF](https://arxiv.org/pdf/2607.08897v1)

**作者:** Recep Bugra Uludag `[一作]` (University of Minnesota - Twin Cities), Ulya R. Karpuzcu `[通讯]` (University of Minnesota - Twin Cities)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出将并行温度交换（Parallel Tempering）方法引入脉冲神经网络（SNN）以解决约束满足问题（CSP）

**💡 创新点**

首次将多温度复制机制与脉冲神经采样结合，利用温度交换打破局部最小陷阱，提升困难实例求解率

**🔧 技术方法**

使用温度调节的神经采样框架、SNN中随机发射率的β缩放以及相邻温度间的Metropolis交换策略

**📊 数据集**

在 SATLIB uf20-91（20变量91条3-SAT）随机生成的 1000 个实例上进行评估

**📈 对比分析**

与相同计算资源下的四个固定温度并行采样器进行对比；在 332 个实例上成功率提升，5 个实例略降；在困难实例中提升显著，且在整个运行期间持续保持优势

**⚠️ 局限性**

在易于求解的实例上几乎无收益；对温度分布和交换频率的选择需要经验调优，且未探索更高维或不同 CSP 类型的泛化

---

## 48. Adaptive MPPI with Online Disturbance Covariance Estimation: Provable Stability Tightening via Spatial Smoothing

**arXiv ID:** 2607.08942 | [PDF](https://arxiv.org/pdf/2607.08942v1)

**作者:** Hyung-Jin Yoon `[一作]` (Tennessee Technological University), Hunmin Kim `[通讯]` (Mercer University)

**通讯引用:** 1154 | [OpenAlex ID](https://openalex.org/A5012273541)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种细胞级在线噪声协方差估计器，并将其与Model Predictive Path Integral (MPPI) 控制器结合，给出了估计器收敛性、平滑偏差、漂移误差等三项误差分解，并将其误差嵌入MPPI的闭环稳定性证明，最终导出“支付定理”：在足够多的采样后，自适应协方差估计器能够得到比任何固定误差协方差更紧的稳定性上界。

**💡 创新点**

创新点在于：
- 设计了带空间扩散的细胞级协方差估计器，并证明其收敛到平滑固定点；
- 在Lyapunov分析中证明扩散算子在p_s加权范数下是耗散的，从而避免了两时间尺度随机逼近的复杂性；
- 将估计误差直接映射为MPPI稳定性证书中的适应性惩罚，给出可计算的跨越时间；
- 在理论上和实验上验证了自适应估计在碰撞率和目标到达率上的优势。

**🔧 技术方法**

技术方法包括：
- 随机逼近（stochastic approximation）与递归更新；
- 细胞网格离散与空间扩散核设计，满足详细平衡条件；
- 以p_s加权的Frobenius范数构造Lyapunov函数，证明扩散算子耗散；
- 结合路径积分控制与MPPI的闭环稳定性理论（线性与非线性两篇伴随论文的结果）；
- 统计上对过程残差的无偏性与四阶矩界定，用于估计误差分析。

**📊 数据集**

实验数据集：
- 双积分器（UAV）环境，状态空间[-5,5]^2，噪声在四象限分别为{0.1,0.3,0.5,0.8} m/s²；
- 自行车模型（UGV）环境，状态空间同上，噪声在窄通道内为0.6 m/s²，其他为0.1 m/s²；
- 两个环境均使用随机轨迹（扫荡式路径）与目标达成/碰撞的终端奖励进行评估。

**📈 对比分析**

比较方法与性能：
- 与固定协方差MPPI（Σ̅_w）和oracle MPPI（已知噪声）进行对比；
- 评估指标包括：估计误差收敛速度、稳定性上界的松紧、跨越时间k_T^*、碰撞率、目标到达率以及平均终端成本；
- 实验结果显示：自适应控制在足够步数后稳定性上界严格优于固定协方差；在碰撞率和目标到达率方面，双积分器自适应控制优于固定控制，甚至超过oracle控制（体现耦合约束下的探索/风险平衡）。

**⚠️ 局限性**

局限性：
- 需要已知或可估计的访问概率p_min>0，极少访问的细胞会导致大偏差；
- 扩散导致平滑偏差β d_κ L_σ r_/p_min，若β或L_σ过大会使自适应控制在早期表现不佳；
- 依赖噪声协方差的慢时变假设和空间Lipschitz连续性；
- 耦合约束Σ_ϵ与控制权重共同作用，导致估计误差不一定直接改善任务奖励；
- 证明仅适用于MPPI的连续时间近似（大样本、η→0），离散采样误差仍未完全覆盖。

---

## 49. Accelerating GPU Inference of Large Language Models with Moderately Unstructured Sparse Weight Matrices

**arXiv ID:** 2607.08786 | [PDF](https://arxiv.org/pdf/2607.08786v1)

**作者:** Tao Lu `[一作]` (National University of Singapore), Wenzhi Chen `[通讯]` (Zhejiang University)

**通讯引用:** 3439 | [OpenAlex ID](https://openalex.org/A5101562846)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出利用中等非结构化稀疏权重矩阵加速大型语言模型在GPU上的推理。

**💡 创新点**

创新点在于设计了专门的稀疏矩阵存储格式与高效的SPMM核，兼顾稀疏性与GPU并行度，显著提升推理速度同时降低显存占用。

**🔧 技术方法**

使用了稀疏压缩技术、混合存储布局、线程级并行化以及CUDA/TensorCore优化，结合半精度推理实现性能最大化。

**📊 数据集**

在GPT‑2 1.5B、LLaMA‑7B、BLOOM等公开模型上进行实验验证。

**📈 对比分析**

与原始密集实现、SparseGPT、Exllama等方法对比，速度提升约2‑3×，显存使用下降30‑40%，推理精度保持在99.9%以上。

**⚠️ 局限性**

局限性包括仅针对推理阶段、对极高稀疏度（>80%）模型效果下降、稀疏化后仍需手工调优，且稀疏化过程对模型准确性有细微影响。

---

## 50. From Generic to Personalized: Exploring Persona-Aware Code Review Explanations

**arXiv ID:** 2607.08990 | [PDF](https://arxiv.org/pdf/2607.08990v1)

**作者:** Shamse Tasnim Cynthia `[一作]` (University of Saskatchewan), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 14253 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究针对不同问题解决风格的开发者，生成与其人格相匹配的代码评审解释。

**💡 创新点**

通过将大语言模型与GenderMag人格框架结合，实现了可自适应的个性化代码评审解释。

**🔧 技术方法**

采用ChatGPT（GPT‑5.2）结合提示工程生成Persona‑aligned 解释，并用GenderMag对开发者进行人格映射。

**📊 数据集**

使用公开的代码评审数据集中的代码片段及评论，以及自建的问卷数据。

**📈 对比分析**

通过混合方法的用户研究，对比不同人格、经验与角色的开发者对解释的偏好，发现Abi偏好详细步骤、Tim偏好简洁实用，但未给出客观性能指标。

**⚠️ 局限性**

样本量小、仅考察Abi与Tim两种极端人格、缺乏定量评估，适用范围受限。

---

## 51. Toward Inferring Accurate Context-free Grammars for Big Languages in a Black-box Setting

**arXiv ID:** 2607.08959 | [PDF](https://arxiv.org/pdf/2607.08959v1)

**作者:** Mohammad Rifat Arefin `[一作]` (University of Texas at Arlington), Christoph Csallner `[通讯]` (University of Texas at Arlington)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种面向大型语言的黑盒上下文无关文法推断方法，能够从有限的样例程序和不可见解析器推导出准确、紧凑的CFG；

**💡 创新点**

创新点在于引入确定性迭代构造解析树的bubble‑merge技术、批量合并与层次差分调试(HDD)、基于LLM的非终结符命名以及终结符泛化策略，显著提升推断准确度与可读性；

**🔧 技术方法**

核心技术包括：解析树通用化（bubble‑merge、CheckBubble）、批量合并、层次差分调试(HDD)、LLM生成非终结符标签、终结符泛化、确定性推断流程；

**📊 数据集**

使用了17种编程语言的样例与测试集（包括小型语言如turtle、json与大型语言如java、cpp、rust），并通过Grammarinator合成无重复的种子/测试程序；

**📈 对比分析**

与Kedavra、Crucio等现有工具在精度、召回率、F1、语法规模和运行时间上进行了系统对比，实验表明本文方法在大多数语言上获得更高的F1（平均提升约35%）、更小的语法尺寸以及更快的构造时间；

**⚠️ 局限性**

局限性包括：在极大语言（如python liquid）仍可能出现异常或不完整的推断；对高度左递归或复杂语法结构的处理尚不完善；依赖种子/测试程序的覆盖率，若样本不足可能导致规则遗漏；

---

## 52. Accelerating Point-in-Polygon Predicates via Algebraic Hash-Joins and Discrete Global Grids at Scale

**arXiv ID:** 2607.08956 | [PDF](https://arxiv.org/pdf/2607.08956v1)

**作者:** Levente Juhasz `[一作]` `[通讯]` (University of Florida), Levente Juhasz (University of Florida)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个交互式仪表盘，评估将点-多边形查询从传统几何判定转为离散全球网格系统（DGGS）的哈希连接的计算开销，并通过四种 DGGS（H3、S2、A5、ISEA4H）在 DuckDB 上实验。

**💡 创新点**

通过统一 DGGS 翻译框架，将不同实现的网格统一到相同接口，实现离散化后将空间关系转为常数时间的等价连接；系统化展示“工具链差距”和预索引的收益；提供可视化的多阶段性能剖析，证明预索引可消除差距，所有网格在子秒级完成连接。

**🔧 技术方法**

DuckDB（向量化数据库）+ C++/Rust/TypeScript/Python 多语言桥接的 DGGS 翻译框架；Polyfill、编码、哈希连接；WebGL 前端仪表盘；Synthetic point generator（Fibonacci 球格 + 高斯混合）；GeoParquet 预索引；Python 代理、C++桥接等。

**📊 数据集**

生成的全球与城市级合成点集（最高 5M 点），OpenStreetMap Nominatim 提取的行政边界；预计算的 GeoParquet 文件覆盖巴西与南非，包含 100k/1M/5M 点在两种分辨率下的四种网格。

**📈 对比分析**

对比相同点集和多边形在三种情景（实时 ETL、预索引缓存、纯 SQL）下的总延迟，并将其拆解为编码、覆盖、连接阶段；发现实时 ETL 与传统几何判定相比可提升 566 倍；预索引后所有网格统一在子秒级完成哈希连接，工具链差距消失。

**⚠️ 局限性**

主要受限于某些科学网格（ISEA4H）的工具链缺陷，导致 ETL 代价高；当前实现对 Python/桥接依赖，未完全利用本地数据库扩展；边界近似误差导致覆盖集与真实多边形不完全一致；未来需完善多网格原生支持并降低编解码开销。

---

## 53. Micro-level AI Feedback Features and Student Responses in Consecutive LLM Tutoring Interactions

**arXiv ID:** 2607.08952 | [PDF](https://arxiv.org/pdf/2607.08952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 54. FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space

**arXiv ID:** 2607.08877 | [PDF](https://arxiv.org/pdf/2607.08877v1)

**作者:** Michael Murray `[一作]` (Microsoft Research), Andrey Kolobov `[通讯]` (Microsoft Research)

**通讯引用:** 366 | [OpenAlex ID](https://openalex.org/A5101862316)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过逆向时间积分与局部精炼，将人类纠正动作映射到预训练生成式机器人策略的噪声空间，从而在保持基准模型不变的前提下训练轻量级噪声策略进行自适应；

**💡 创新点**

关键创新在于‘动作反演’技术，将人类干预动作转换为对应噪声向量，利用噪声空间监督轻量级策略，避免修改基准模型参数，兼顾样本与计算效率；

**🔧 技术方法**

使用逆向ODE积分、固定点迭代实现动作反演；结合流匹配与扩散生成式策略；在DAgger框架下训练噪声策略；同时实现世界动作模型（WAM）的联合噪声反演；

**📊 数据集**

在MetaWorld模拟环境（12个基准任务）和真实机器人平台（两臂FR3、单臂操作），包括Block Pick、Glassware Stacking、BusyBox系列、Jenga Stack、Toolbox Packing、Plug Insertion等任务；基准模型使用预训练的VLA（π0.5）与Cosmos-Policy；

**📈 对比分析**

与SFT、LoRA‑DAgger、Residual‑DAgger、DSRL等基线在相同干预/演示预算下比较；结果显示在模拟与真实环境中，动作反演+噪声策略在少量干预（5–20回合）即可显著提升成功率，且保留基准模型在未见任务上的性能；

**⚠️ 局限性**

适配能力受限于基准模型的动作流形，难以实现超出该流形的新行为；依赖高质量人类干预，干预不足或偏差会影响适配效果；在多模态或参数不稳定的生成模型中，动作反演精度可能下降。

---

## 55. Reward Transport: Property Control in Flow Matching via Noise-Space Alignment

**arXiv ID:** 2607.08781 | [PDF](https://arxiv.org/pdf/2607.08781v1)

**作者:** Kehan Guo `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 8242 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `40105733-5154-44cd-8090-a8cab9e64b07` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 Reward Transport，将流匹配的噪声-数据耦合通过一维最优传输排序，形成一个单一噪声标量控制器，可在推理时实现分布级分子属性调控。

**💡 创新点**

创新点在于把耦合视为对齐接口，利用排序OT耦合把属性信息写入噪声空间，从而无需任何指导或奖励模型即可通过单一噪声标量实现属性分布截断。

**🔧 技术方法**

采用流匹配模型、最优传输排序、方向MLP 注入、SELFIES 表示、预层归一化修正等技术实现模型训练与推理。

**📊 数据集**

实验使用 ZINC‑250K 与 GuacaMol 两个公开分子数据集，分别评估 logP 与 QED 两种属性。

**📈 对比分析**

与无OT基线、条件流匹配等方法对比，单一噪声标量即可实现 logP 最高 137% 的提升（ρ=1），QED 提升 0.059（ρ=1），同时保持 100% 合法性与高多样性。

**⚠️ 局限性**

局限性包括仅适用于单属性、单维排序的情况；ε‑预测目标下无效；长度与拓扑属性控制受限；对大模型、多属性或强化学习微调的效果尚待验证。

---

## 56. Tonnetz-Driven Graph Wedgelet for Harmonic Complexity Reduction in Music Scores

**arXiv ID:** 2607.08806 | [PDF](https://arxiv.org/pdf/2607.08806v1)

**作者:** Emmanuel Caronna `[一作]` (University of Palermo), Silvia Licciardi `[通讯]` (University of Palermo)

**通讯引用:** 571 | [OpenAlex ID](https://openalex.org/A5014238743)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向符号音乐谱的图分块压缩方案，通过在钢琴子图上应用基于Tonnetz六维嵌入的二叉楔分割树实现对伴奏音符的可解释性压缩；

**💡 创新点**

创新点在于将和声距离嵌入到图信号分块准则中，并采用均值映射与量化回原始音高集合，保证压缩后谱子在音高网格上可直接播放；

**🔧 技术方法**

使用了图楔分割（graph wedgelet）、FA‑greedy自适应二叉分割树、六维Tonnetz嵌入、均值基解码与音高量化；

**📊 数据集**

实验基于包含70首MusicXML格式乐谱、涵盖三位作曲家的公开语料库；

**📈 对比分析**

通过计算分块逼近与原谱在Tonnetz空间中的RMSE与压缩比例的关系，并在示例谱上可视化保留/变更的音高比例，显示在低压缩比下仍能保持大量原始音高，性能以RMSE曲线和色块比例呈现；

**⚠️ 局限性**

仅压缩钢琴子图，未覆盖多类型异构图；每个楔仅使用常数值，无法捕捉楔内的和声变化；未来工作需扩展到完整异构图并考虑非恒定重建函数。

---

## 57. Secure-by-Disguise: A Systematic Evaluation of Image Disguising for Confidential Medical Image Modeling

**arXiv ID:** 2607.08867 | [PDF](https://arxiv.org/pdf/2607.08867v1)

**作者:** Jason Rojas `[一作]` (University of Maryland, Baltimore County), Keke Chen `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 1123 | [OpenAlex ID](https://openalex.org/A5002572745)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了统一评估框架，对隐形图像技术在医疗图像分类和分割任务中的实用性、效率、参数敏感性及对抗重构攻击的表现进行系统评估。

**💡 创新点**

创新点在于首次将隐形图像方法应用于真实医学图像数据，并揭示了其对图像级任务与像素级任务的根本差异，同时通过对RMT、AES与NeuraCrypt的比较指出RMT在隐私与性能平衡上的优越性。

**🔧 技术方法**

主要技术包括块级随机多维变换（RMT）、AES加密变换、NeuraCrypt特征隐写，以及基于已知对的回归重构攻击和DNN审计器评估隐私泄露。

**📊 数据集**

使用了四个公开医学数据集：Breast (histopathology)、MData (临床伤口)、CVC-ClinicDB (结肠镜) 与 Wound Patch (糖尿病足部) 以覆盖分类与分割两大任务。

**📈 对比分析**

方法比较采用统一实验协议（相同数据划分、超参数、五折交叉验证），结果显示：对分类任务，RMT保持约10% F1降幅，AES性能几乎等于随机；对分割任务，RMT和AES均导致Dice降至0.4-0.6，NeuraCrypt保持原始水平；RMT在一次性预处理时间仅为数毫秒，且在强攻击下对重构无明显收益。

**⚠️ 局限性**

局限性包括：仅评估了四个数据集，未覆盖CT/MRI/超声等常见影像；只检验了回归式重构攻击，未考虑基于生成模型或更强攻击；现有隐形方法仍无法满足精细像素级应用，需要进一步研究保留空间结构的隐私技术。

---

## 58. Wireless Decentralized Federated Learning via Device Clustering and Inter-Cluster Link Enhancement

**arXiv ID:** 2607.08797 | [PDF](https://arxiv.org/pdf/2607.08797v1)

**作者:** William Weijia Zheng `[一作]` (Chinese University of Hong Kong), Ying-Jun Angela Zhang `[通讯]` (University of Macau)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种预算感知、基于聚类的去中心化联邦学习框架，在设备之间通过少量可靠回程链路提升聚合效率。

**💡 创新点**

创新点在于利用网络的“小世界”结构，将设备划分为密集聚类并在聚类头之间部署有限可靠链路，实现两层聚合：聚类内部 OTA，聚类头间低频无噪声 gossip，从而在保持去中心化的同时显著加速收敛。

**🔧 技术方法**

采用了 OTA (over-the-air) 聚合、混合聚合协议、基于大规模衰落信息的聚类优化、最小生成树与 Gibbs 采样等技术。

**📊 数据集**

使用了 Fashion‑MNIST 图像分类数据集，分布在 50 台设备上进行实验。

**📈 对比分析**

与传统基于 OTA gossip 的去中心化 FL 基线对比，实验显示该方法在训练损失与测试准确率上实现了更快的收敛，且对 inter‑cluster 通信间隔 H 具有鲁棒性。

**⚠️ 局限性**

局限性包括需要预先了解大规模衰落统计并在聚类头之间部署额外硬件链路，适用场景受网络规模和部署成本限制；此外算法对链路预算和聚类数的选择敏感，过多聚类可能因预算不足导致聚类质量下降。

---

## 59. StereoSplat+: Feed-Forward Stereo Gaussian Splatting with Diffusion-Assisted Progressive Inference

**arXiv ID:** 2607.08808 | [PDF](https://arxiv.org/pdf/2607.08808v1)

**作者:** Zihua Liu `[一作]` (Institute of Science Tokyo), Masatoshi Okutomi `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5018135747)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在仅有单对立体图像的条件下，提出StereoSplat+框架，实现单目立体输入的实时3D高斯光散射场重建，并通过一次扩散增强的伪多视角推理提升渲染与几何质量。

**💡 创新点**

① 输入不变的双分支StereoSplat，可处理可变数量与姿态的立体图像；② 采用一次扩散增强的伪视图循环，仅需单步渲染+扩散即可模拟多视角输入；③ 在单视图基础上实现渐进式改进，同时保持实时推理。

**🔧 技术方法**

3D高斯光散射 (3DGS)、成本体积与三平面变压器双分支网络、连续姿态编码、随机视图子采样、一次扩散增强模型 (SD‑Turbo/DIFIX3D+)。

**📊 数据集**

KITTI‑360 数据集的立体图像与稀疏 LiDAR 深度。

**📈 对比分析**

与 PixelSplat、MVSplat、OmniScene、DepthSplat 等最新前馈3DGS基线对比，使用 PSNR/SSIM 与 AbsRel/SqRel 评估；StereoSplat+ 在单视图下可实现 21.21 dB PSNR、0.72 SSIM，深度误差显著低于基线，表现优异。

**⚠️ 局限性**

扩散增强可能产生伪影且多次递推易累积误差；未显式建模动态物体，导致时间一致性不足。

---

## 60. HAT Super-Resolution and a PARSeq+CLIP4STR Voting Ensemble for Extreme In-the-Wild License Plate Recognition

**arXiv ID:** 2607.08896 | [PDF](https://arxiv.org/pdf/2607.08896v1)

**作者:** Karthik Sivarama Krishnan `[一作]` (Independent Researcher), Koushik Sivarama Krishnan `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套针对极端环境车牌超分辨率与识别的端到端管线，包括HAT超分辨率、PARSeq+CLIP4STR OCR集成及自适应字符投票放弃机制。

**💡 创新点**

创新点在于结合软最大置信度的字符投票并自适应放弃低置信度位置，采用2:1加权的OCR集成，以及利用Laplace方差选取最清晰帧进行ECC对齐与像素级融合。

**🔧 技术方法**

技术包括Hybrid Attention Transformer (HAT) 超分辨率、PARSeq-S v2 与 CLIP4STR-B 场景文字识别器、ECC affine 对齐、像素级融合、soft‑max 加权投票与阈值τ=0.33 的放弃策略以及SIV格式字符纠错。

**📊 数据集**

使用的数据集为XLPSR开发集（39序列）、200k合成法国车牌、以及公开预训练权重（PARSeq、CLIP4STR、CLIP ViT‑B/16、HAT）。

**📈 对比分析**

在公开验证集上从无SR基线的7.27 wECR提升至9.73 wECR，超分辨率贡献+2.00点，OCR集成+0.31点，阈值投票+0.11点；在盲测中获得9.73的最高分。

**⚠️ 局限性**

局限性在于超分辨率模型计算开销较大、投票权重与阈值对不同数据分布敏感、以及仅针对法国SIV格式车牌设计，缺乏对多语种或多格式车牌的通用性。

---

## 61. AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision

**arXiv ID:** 2607.08984 | [PDF](https://arxiv.org/pdf/2607.08984v1)

**作者:** Brent Kong `[一作]` (California Institute of Technology), Tony Yue Yu `[通讯]` (California Institute of Technology)

**通讯引用:** 108 | [OpenAlex ID](https://openalex.org/A5064553016)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究AlphaZero在可完全求解的棋类游戏（连连四和Chomp）中，强力玩法与完美玩法之间的差距，并比较普通AlphaZero、多帧输入与加入辅助Oracle损失的模型表现。

**💡 创新点**

首次引入Oracle一致性评估指标，证明强力玩法不等于完美玩法；提出AlphaZero辅助损失（AZAL）通过Oracle监督显著提升Oracle一致性；同时验证多帧输入在此任务中无效。

**🔧 技术方法**

使用AlphaZero自玩+MCTS框架，构建多帧输入变体，加入Oracle辅助损失（cross‑entropy）；利用Pons求解器和Grundy数Oracle进行状态评估；通过轨迹级Oracle-match率、最长Oracle链、FirstFail等指标进行分析。

**📊 数据集**

利用连连四与Chomp的完整棋盘状态、随机种子自玩生成的全局游戏轨迹，以及从这些轨迹中随机采样的状态集合，构成评估数据。

**📈 对比分析**

通过多seed全局轨迹与随机采样状态的Oracle一致性统计，比较Vanilla AlphaZero、多帧输入和AZAL的性能。结果显示Vanilla AlphaZero在两游戏中表现强大但Oracle一致性低；AZAL在Chomp 10×11实现完美一致，在9×10表现高一致；在连连四中仅提升Oracle一致率，未达到完美。

**⚠️ 局限性**

研究仅涵盖小型确定性棋盘游戏，样本覆盖有限；未探索不同搜索预算、模型超参数、检验点对结果的影响；结果可能不易推广至更大或更复杂的环境。

---

## 62. Privacy-Preserving Intent Fulfilment and Assurance for 6G RAN

**arXiv ID:** 2607.08809 | [PDF](https://arxiv.org/pdf/2607.08809v1)

**作者:** Joss Armstrong `[一作]` `[通讯]` (Ericsson), Joss Armstrong (Ericsson)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种在6G RAN中实现隐私保护的意图（Intent）满足与保障架构，核心是只利用声明的意图类别进行资源调度，并仅通过O‑1接口的聚合性能计数器（PM）进行服务保障，而不进行深度包检测或收集用户级流量。

**💡 创新点**

创新点包括：①利用数据处理不等式证明资源分配最多泄露 log₂K 位信息；②定义并证明了意图‑流量不可链接性和节点透明验证两项架构隐私属性；③揭示隐私与检测功效在类别粒度上的结构性一致性；④将该架构映射至3GPP意图生命周期和O‑RAN非实时RIC，展示其在多厂商多租户环境下的可部署性。

**🔧 技术方法**

技术手段主要包括：基于 Markov 链的隐私分析、k‑means 聚类构造意图类别、聚合统计检测（均值阈值法）、O‑1接口聚合 PM 计数器采集、数据处理不等式与信息理论的泄漏上界推导。

**📊 数据集**

实验使用四个运营商生产 PM 数据集（Net‑A、Net‑B、Net‑C、Net‑D），共计 41,000+ 个单元格小时，11 维标准化 PM 特征；通过聚类、阈值校准与注入式退化模拟构造评估数据。

**📈 对比分析**

与传统 GenAI‑IDN（流量级联分析）和 E2‑based（单 UE 近实时监控）方案相比，MISES 方案在每个单元格的采样量低三至四个数量级；检测召回率在 K≈8 时达到 0.85（最高）并在更大 K 上趋于平台化；隐私泄漏随 K 增加而线性增长，达 4.9 位；总体而言，该方案在保证同等检测性能的前提下显著降低了数据量和隐私风险。

**⚠️ 局限性**

局限性包括：①隐私定义仅为架构层面，未考虑攻击者可能利用外部关联信息的情况；②检测基于注入式退化，未验证在自然退化环境下的表现；③聚合检测的优势依赖于“任务正交噪声”假设，若此假设失效则节点透明验证可能不再优于单 UE 验证；④未覆盖对动态网络环境下类别漂移的实时自适应机制。

---

## 63. SCATE: Learning to Supervise Coding Agents for Cost-Effective Test Generation

**arXiv ID:** 2607.08983 | [PDF](https://arxiv.org/pdf/2607.08983v1)

**作者:** Sijia Gu `[一作]` (University of British Columbia), Ali Mesbah `[通讯]` (University of British Columbia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于上下文带的自适应监督框架，自动决定编码代理在单元测试生成过程中的默认生成、程序分析或停止三种动作，从而解决代理的“懒生成”问题。

**💡 创新点**

将代理监督建模为上下文带问题，利用静态测试可测度和实时覆盖反馈学习最优策略，实现完全无人工干预的自适应监督。

**🔧 技术方法**

采用 LinUCB 上下文带算法、LLM 驱动的编码代理、MCP 形式的程序分析工具、动态覆盖收集以及以覆盖收益与成本权衡的奖励函数。

**📊 数据集**

使用 Defects4J 的 120 个结构复杂类（基于 LOC、WMC、RFC 过滤），训练 40 个，评估 80 个。

**📈 对比分析**

与无监督代理（单次/多次迭代）和两种主流非代理 LLM 测试生成工具对比，实验结果显示行覆盖率提升 32.3%、分支覆盖率提升 30.9%，变异得分达到 64.7%，成本和运行时间相对可控。

**⚠️ 局限性**

实验仅在 Java 环境和 Defects4J 数据集上验证，可能对其他语言或不同复杂度的项目推广效果有限；监督策略依赖数值指标，在极端代码结构下可能失效；对 LLM 的成本和延迟敏感。

---

## 64. A Unified Approach to Interpreting Knowledge Distillation for Large Language Models via Interactions

**arXiv ID:** 2607.08776 | [PDF](https://arxiv.org/pdf/2607.08776v1)

**作者:** Qingzhuo Wang `[一作]` (Tongji University), Zhihua Wei `[通讯]` (Tongji University)

**通讯引用:** 1133 | [OpenAlex ID](https://openalex.org/A5004301353)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了知识蒸馏（KD）在大型语言模型（LLM）中的机制，提出通过交互稀疏化来解释不同KD方法的共同效果，并提出了可插拔的Complex Interaction Penalty（CIP）损失；

**💡 创新点**

创新点在于将游戏理论交互作为统一框架，揭示KD的核心机制是保留教师模型的显著简单交互并压缩复杂交互，同时引入CIP显式抑制复杂交互，从而提升蒸馏效果；

**🔧 技术方法**

主要技术包括交互分析（Shapley、游戏理论）、Gini系数和Shannon熵衡量稀疏度、交互重叠度评估学生-教师对齐、CIP正则化以及采样近似方法降低交互计算复杂度；

**📊 数据集**

使用Dolly数据集进行训练，评估时使用DollyEval、Self‑Instruct、Super‑Natural Instructions、Vicuna等指令跟随基准；

**📈 对比分析**

对比六种主流KD方法（KD、SeqKD、ImitKD、MiniLLM、GKD、DISTILLM），在GPT‑2/OPT/LLaMA家族模型上验证，CIP显著提升ROUGE‑L和GPT‑5分数，尤其在离谱分布（OOD）测试中表现更好；

**⚠️ 局限性**

局限在于交互提取与评估计算量大，需采样近似；CIP权重需手动调参，过大会削弱重要交互；此外研究聚焦于交互稀疏化，未深入探讨语义层面的细节。

---

## 65. Programmers Are Poor and Overconfident Judges of LLM-Generated Assertions

**arXiv ID:** 2607.08885 | [PDF](https://arxiv.org/pdf/2607.08885v1)

**作者:** Zhanna Kaufman `[一作]` (University of Massachusetts Amherst), Madeline Endres `[通讯]` (University of Massachusetts Amherst)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对86名Python程序员的控制实验和10名开发者的think‑aloud访谈，系统评估了人类在判断LLM生成的后置断言（postcondition）正确性与完整性时的能力，并考察自然语言解释对判断的影响。

**💡 创新点**

创新点在于首次量化展示开发者在识别错误断言时的显著低效（49% vs 74%），揭示自然语言解释在提高判断准确性方面不见提升，甚至低质量解释会降低准确度并提高自信心，从而挑战了“自然语言解释能帮助理解”的常规假设；同时结合定量与定性方法揭示了不同断言构造模式与认知策略对判断的影响。

**🔧 技术方法**

使用了GPT‑3.5‑turbo生成断言与评论，Python 3代码实现实验，利用混合效应模型（generalized linear mixed‑effects 和 linear mixed‑effects）分析定量数据，采用 directed content analysis 进行定性访谈转录编码。

**📊 数据集**

主要数据集为OpenAI的 HumanEval 单函数Python代码库，选取26个函数生成260个实验刺激（每个函数含正确/错误断言以及五类评论），实验参与者共86人；think‑aloud 样本为10名开发者。

**📈 对比分析**

通过比较正确与错误断言的判断准确率、置信度与反应时，使用卡方检验、Fisher精确检验和混合效应回归评估效果。结果显示：正确断言判断准确率为73.9%，错误断言为49.0%；自然语言评论整体不提升准确率，低质量评论导致准确率下降至约54%（相对精确评论）并显著提升自信；此外，准确判断错误断言需要更长时间。相比以往仅关注生成质量的研究，本研究提供了人类评估层面的实证。

**⚠️ 局限性**

限制包括：受试者仅来自美国高校与企业，样本可能不具代表性；使用的生成模型为GPT‑3.5‑turbo，未覆盖更先进模型；实验仅基于单函数、单断言的 HumanEval 代码，难以推广到多函数或复杂逻辑的真实项目；此外评论的质量评估仍基于人工标注，可能存在主观偏差。

---

## 66. Entropy Bootstrapping for Wireless Embedded Systems

**arXiv ID:** 2607.08865 | [PDF](https://arxiv.org/pdf/2607.08865v1)

**作者:** Javier Blanco-Romero `[一作]` (Universidad Carlos III de Madrid), Andrés Marín-López `[通讯]` (Polytechnic University of Madrid)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种多源防御深度的 ESP32 启动熵路径，结合 SRAM PUF、无线爆发包提取和非对称熵胶囊，并实现 Zephyr 固件与实验测量。

**💡 创新点**

创新点在于显式的源状态录入与信任根组合，避免仅凭输出统计；引入公共爆发包提取与本地 WDEV 输出结合，提供可验证的本地熵；使用非对称熵胶囊实现冷启动安全且保持隐私。

**🔧 技术方法**

使用 ESP32 WDEV RNG、SRAM PUF、无线爆发包提取、ML‑KEM/ML‑DSA 后量子密钥封装、HKDF、Zephyr RTOS、BLAKE2s、TestU01 Rabbit 等熵评估工具。

**📊 数据集**

实验数据集包括 256 条 RF 状态下的 WDEV 流、127 次固定爆发包试验、10 次熵胶囊延迟测量、5 个 SRAM 重置转储，以及主机 Linux 基准。

**📈 对比分析**

通过 SP800‑90B、ENT、Borel、AIS31 等熵计数器和 TestU01 Rabbit 评估；RF‑禁用状态被拒绝，RF‑启用状态通过；爆发包窗口约 59 kbit 原始熵，提取到 256 bit 种子；胶囊延迟平均约 10 s（服务器侧）且客户端仅 18.7 s，整体性能满足低功耗 IoT 节点的熵需求。

**⚠️ 局限性**

局限性：仅在单板实验，缺乏多板复现；SRAM 仅做预备评估，未完成完整冷启动实验；爆发包熵假设对完全观察者仍是经验性；胶囊需可信节点与预装密钥；不提供源证明，只通过运行时录入和策略。

---

## 67. The queer Hero versus the Fool bias of the queer trait: An archetypometric analysis of the collective portrayal of queerness in fictional stories

**arXiv ID:** 2607.08859 | [PDF](https://arxiv.org/pdf/2607.08859v1)

**作者:** Ashley M. A. Fehr `[一作]` (University of Vermont), Peter Sheridan Dodds `[通讯]` (University of Vermont)

**通讯引用:** 9224 | [OpenAlex ID](https://openalex.org/A5040821463)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过archetypometric方法对2000个虚构角色的特质评分进行分析，探究观众对酷儿与异性恋角色的感知，并比较这些角色在六维原型空间中的分布。

**💡 创新点**

创新点在于将酷儿-异性恋语义差值与六维原型向量相结合，揭示酷儿角色在原型维度上呈现积极趋势（如冒险家、极客）与负面刻板（如贬义特质）的对立关系，并提出基于SVD的原型构建与族群邻域选择方法。

**🔧 技术方法**

主要技术包括：单值分解（SVD）得到原型向量；语义差值与特质空间分析；欧氏距离、内积、余弦相似度用于构建角色邻域；bootstrap与质心半径（Rg）评估子样本稳定性；基于Fandom LGBTQ+标签的聚类过滤。

**📊 数据集**

使用的数据集为：OpenPsychometrics “Which Character”问卷的72M评分数据（2000个角色、464特质），以及Fandom LGBTQ+ Characters Wikia XML数据（共9680个角色），二者结合得到125名“canonically queer”角色。

**📈 对比分析**

通过比较straight-aligned与queer-aligned子集在原型维度（Hero、Adventurer、Geek、Brute等）以及特质距离上的差异，发现酷儿角色更倾向于冒险家与极客原型，异性恋角色更倾向于英雄与野蛮人原型；该方法覆盖多种媒体形式且样本规模大，能够揭示正面形象与负面刻板的对立。

**⚠️ 局限性**

局限性包括：1）对酷儿与异性恋的二元标签过于简化，缺乏时间与多样性维度；2）Fandom标签可能含有错误或粉丝猜测；3）样本主要为西方主流影视，缺乏跨文化视角；4）语义差值仅捕捉straight-queer对立，未细化更细粒度身份；5）原型分析受限于SVD特征提取与原型数目。

---

## 68. Optimizing Against Safety Representations: Activation-Guided Adversarial Suffixes and the Geometry of Refusal

**arXiv ID:** 2607.08883 | [PDF](https://arxiv.org/pdf/2607.08883v1)

**作者:** Ege Çakar `[一作]` (Harvard University), Kayden Kehe `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于内部拒绝方向的白盒攻击（Activation‑Guided GCG）以及一种连续化的软GCG（Soft‑GCG）方法，显著提升了对已对齐大型语言模型的逃逸成功率并大幅降低计算成本。

**💡 创新点**

创新点在于：① 用内部拒绝方向投影的最小化替代传统输出日志目标，直接攻击拒绝机制；② 将离散的 GCG 换成 Gumbel‑Softmax 连续优化，实现在约 33× 速度提升下保持甚至提升攻击成功率。

**🔧 技术方法**

采用梯度下降、激活方向提取（差异均值法）、Gumbel‑Softmax 连续化、Carlini‑Wagner 损失、温度退火调度等技术。

**📊 数据集**

使用的主要数据集包括 AdvBench（训练/测试）、HarmBench、拒绝方向数据集以及 Gemma3/LLama‑2‑7B‑Chat 等模型进行评估。

**📈 对比分析**

与标准 GCG、消融（Ablation）以及不同激活目标（Single、Layer、Token、All、Negative）对比：All 目标在子串 ASR 上达 0.91，接近消融 0.98；Soft‑GCG 在速度上比 GCG 提升 33×，在子串 ASR 上与 GCG 相当或更好；在 Gemma3 系列，模型越大攻击成功率下降，体现规模化对安全机制的影响。

**⚠️ 局限性**

主要限制包括：需要白盒权重和梯度访问；子串匹配评估易出现误判；大词表导致连续化优化更难；实验主要针对中小规模模型，尚未在更大模型或实际部署环境中验证。

---

## 69. Decoupled Illumination Priors for Spatially Controllable Multi-View Indoor Scene Relighting

**arXiv ID:** 2607.08879 | [PDF](https://arxiv.org/pdf/2607.08879v1)

**作者:** Chenjian Gao `[一作]` (Chinese University of Hong Kong), Tianfan Xue `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1945 | [OpenAlex ID](https://openalex.org/A5100552155)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Lume-Palette 框架，实现室内场景多视角照明重绘，支持用户在三维空间自由放置光源并获得高逼真度、视角一致的重照明结果。

**💡 创新点**

通过将照明重绘拆分为“照明提炼”(从预训练扩散模型中提取四个方向的标准照明图集)和“照明投射”(利用提炼的照明图集与用户自定义的接收器中心光照条件合成目标照明)，以及引入非对称多视角条件策略，既保留了扩散模型的光照先验，又实现了显式空间控制与多视角一致性。

**🔧 技术方法**

使用基于 Flux.2 Klein 9B 的扩散模型、LoRA 微调、流匹配（flow‑matching）训练目标、3D 位置编码和自定义多视角条件机制；光照条件通过三维网格重建后渲染白色网格得到。

**📊 数据集**

使用 MIT Multi‑Illumination（1,000 场景，25 个照明方向）进行照明提炼训练；在合成的多视角重照明数据集（480 场景）训练投射阶段；在 ScanNet++ 等真实场景上进行测试。

**📈 对比分析**

与 RGB↔X、IC‑Light、LumiNet、ScribbleLight 等现有方法对比，PSNR、SSIM、LPIPS、跨视角一致性（MV‑MSE、MV‑LPIPS）均显著优于基线，用户研究中在真实性、光照符合度和多视角一致性上均获得最高分。

**⚠️ 局限性**

对重建几何粗糙度敏感，薄结构、透明或高反射材质以及复杂间接照明场景下效果受限；未对全局间接光进行建模，导致某些光照细节缺失。

---

## 70. Living Inside the Black Box: Behavioral Probing and Adaptation in Mandatory Wearable Sensing

**arXiv ID:** 2607.09009 | [PDF](https://arxiv.org/pdf/2607.09009v1)

**作者:** Yibo Meng `[一作]` (Cornell University), Shuai Ma `[通讯]` (Aalto University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对24名曾接受强制电子监测的受试者进行半结构访谈，研究他们如何在不可解释的可穿戴传感系统下进行感知识字。

**💡 创新点**

提出‘受限情境下的感知识字（sensor literacy under constraint）’概念，并发现监测受试者在不同规则域形成两种适应倾向：战略利用与保守收缩，并揭示监测后遗留的计算习惯。

**🔧 技术方法**

采用定性访谈与主题分析法，对受试者的经验进行编码与归纳。

**📊 数据集**

收集了24名社区矫正系统电子监测受监者的访谈文本，涉及年龄、性别、职业、教育等背景。

**📈 对比分析**

没有定量指标或对比实验，研究以访谈数据为主，无法给出数值性能评估。

**⚠️ 局限性**

样本仅来自中国社区矫正，且数据为自述，缺乏系统日志与对照，研究结果可能受地区和文化差异影响。

---

## 71. LieBN: Batch Normalization over Lie Groups

**arXiv ID:** 2607.08783 | [PDF](https://arxiv.org/pdf/2607.08783v1)

**作者:** Ziheng Chen `[一作]` (University of Trento), Nicu Sebe `[通讯]` (University of Trento)

**通讯引用:** 19246 | [OpenAlex ID](https://openalex.org/A5027171279)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了通用的 Lie 群批归一化框架 LieBN，能够在任意 Lie 群上以左、右或双不变度量实现均值和方差的可控归一化，并将其应用于 SPD、旋转矩阵和相关矩阵等常见非欧几里得空间。

**💡 创新点**

创新点包括：①在 Lie 群上给出理论保证的可控均值与方差批归一化方法；②首次设计了非平凡的右不变 SPD 度量 CRIM；③通过矩阵幂变形扩展了 SPD 的 Lie 结构；④在多种几何上统一实现了 LieBN。

**🔧 技术方法**

采用了 Riemannian 几何、Lie 群不变度量、Fréchet 均值、欧氏 BN 的拉回映射、Riemannian 优化以及在 SPDNet、LieNet、TSMNet 等网络中的实现。

**📊 数据集**

使用了 Radar、HDM05、FPHA（雷达/人类动作识别）、NTU60、G3D（动作识别）以及 Hinss2021（EEG）等公开数据集。

**📈 对比分析**

与 SPDBN、SPDDSMBN、ManifoldNorm 等传统方法对比，LieBN 在不同几何和数据集上均提升了准确率（如 SPDNet 在 Radar 上+1.62%，HDM05 上+11.71%，FPHA 上+4.8%），并保持或提升训练效率；在 LieNet 上加速收敛、降低波动；在 EEG 上相较 SPDDSMBN 提升 0.98%–3.87%。

**⚠️ 局限性**

局限性包括：对高维 SPD/相关矩阵的计算开销仍较大；右不变度量在部分任务中的表现有限；对不同几何的最优度量仍需经验调参；尚未扩展到非 Lie 结构（如球面、双曲空间）或更广泛的网络架构。

---

## 72. CogniConsole: Externalizing Inference-Time Control as a Formal Abstraction for Reliable LLM Interactions

**arXiv ID:** 2607.08774 | [PDF](https://arxiv.org/pdf/2607.08774v1)

**作者:** Vanessa Figueiredo `[一作]` (University of Regina), Wilter Franceschi `[通讯]` (Orbital Sea)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为 CogniConsole 的架构，用以显式化 LLM 的推理时控制层，并通过结构化的提示与程序化调度降低模型输出方差与失败率。

**💡 创新点**

创新点在于将推理时控制视为第一类可编程计算层，区分程序化决策空间与受限提示推理，构建单梯度决策节点、边界约束与选择性记忆，实现可调节的任务级控制结构。

**🔧 技术方法**

技术手段包括：结构化提示（单决策阶梯）、程序化节点与卡特（Cartridge）模型、边界与输出契约、短期/长期记忆分离、模型路由与容量解耦，以及在 GPT‑class 前置模型上嵌入的控制接口。

**📊 数据集**

数据集：自定义多步骤游戏式交互环境（共 169 条探测实例，N=489 条测试），未使用公开数据集；对比不同控制层级（无结构、半结构、完全结构）下的表现。

**📈 对比分析**

对比方法：在三种提示结构（C1、C2、C3）下，使用五类探测（P1–P5）评估平均分与方差；结果显示 C3 相较 C1 的相对提升为 +15% 左右，失败率下降 30%，但在处理噪声输入时 C3 略逊于 C1。

**⚠️ 局限性**

局限性：结构化控制会削弱模型在模糊或噪声情境下的灵活性；过度约束可能导致误判；实验仅在单一前置模型与自定义任务上验证，需在更广泛的模型与任务上进一步评估；控制层与模型架构间的相互作用尚未完全探索。

---

## 73. Group Invariant Spectral Embedding

**arXiv ID:** 2607.08987 | [PDF](https://arxiv.org/pdf/2607.08987v1)

**作者:** Yeari Vigder `[一作]` (Tel Aviv University), Amit Moscovich `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出在已知对称性（由紧致李群G实现）的高维数据集上，将对称性直接嵌入谱嵌入的亲和核，从而得到的图拉普拉斯算子在商流形上收敛，显著提高了收敛速率；

**💡 创新点**

创新点在于：①通过三类G-不变核（最小、积分、特征映射）构造谱嵌入；②证明图拉普拉斯算子点态收敛到商流形上的二阶微分算子，并给出改进的采样复杂度；③在商流形上实现对称性感知的低维表示，解决传统谱嵌入忽略对称性的问题；

**🔧 技术方法**

技术包括：Riemannian 子流形假设下的图拉普拉斯构造、G-不变核的分析、Fubini类型积分公式、概率不等式（Bernstein）、与拉普拉斯–贝尔米算子以及积分/最小核的微分算子推导；

**📊 数据集**

实验数据集涵盖SO(2)和SO(3)对称性：①Glucagon分子3D点云（旋转自由度）；②Glucagon的投影图像（二维平面旋转）；③两旋转玩具的二维图像；

**📈 对比分析**

对比方法：传统欧氏核谱嵌入；结果显示：G-不变核能够准确恢复圆、半圆、环面等内在几何；相较于欧氏核，收敛速率提升，样本效率提高，噪声鲁棒性更好；

**⚠️ 局限性**

局限性：仅考虑自由作用的紧致李群；未证明特征值与特征向量的收敛；对近似对称性或非自由作用的情况缺乏理论支持；

---

## 74. iLENS: Interpretable LLM-Guided Mixture-of-Experts for Neuroimaging Survival Analysis

**arXiv ID:** 2607.08778 | [PDF](https://arxiv.org/pdf/2607.08778v1)

**作者:** Farica Zhuang `[一作]` (University of Pennsylvania), Li Shen `[通讯]` (University of Pennsylvania)

**通讯引用:** 23732 | [OpenAlex ID](https://openalex.org/A5100333320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出 iLENS，一个基于 LLM 引导的混合专家框架，用于阿尔茨海默病转化的生存预测和亚型划分。

**💡 创新点**

创新点在于：①利用 LLM 进行专家初始化与语义路由，实现可解释的专家分配；②在 MoE 架构下将生存模型与聚类相结合，提供双层可解释性，并生成自然语言的推理过程。

**🔧 技术方法**

使用技术包括：LLM（GPT‑4.1‑mini）作为策略者和路由器；混合专家（MoE）网络；Weibull 分布的生存子型混合模型；Softmax 软聚类；以及多模态特征处理与评估指标。

**📊 数据集**

采用 ADNI（Alzheimer's Disease Neuroimaging Initiative）纵向多模态数据，包含 VBM、FDG、AV45 等结构化影像特征和临床笔记。

**📈 对比分析**

与 SCA、VaDeSC、NSC、DCSM 等现有聚类生存模型在 C‑Index 与 LogRank 上进行对比。iLENS 在大多数模态和聚类数下保持竞争性的 C‑Index，并在 LogRank 上显著优于基线，表明更优的亚型分离和预测性能。

**⚠️ 局限性**

局限性包括：①对 LLM 的路由与专家初始化过度依赖，易受偏见和幻觉影响；②仅在 ADNI 单一队列和疾病设置中验证；③仅探讨 K=2、3 的亚型，缺乏更细粒度或多种亚型的评估；④未来需要在不同 LLM、临床结局和亚型方案下检验鲁棒性。

---

## 75. Interval Certifications for Multilayered Perceptrons via Lattice Traversal

**arXiv ID:** 2607.08773 | [PDF](https://arxiv.org/pdf/2607.08773v1)

**作者:** Merkouris Papamichail `[一作]` (Foundation for Research and Technology - Hellas), João Marques-Silva `[通讯]` (Catalan Institution for Research and Advanced Studies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了多层感知机（MLP）在对抗鲁棒性方面的区间认证问题，将鲁棒性判定转化为格（lattice）遍历问题，提出最大声誉（sound）区间和最小完整（complete）区间的求解框架。

**💡 创新点**

创新点在于：①引入完整性认证概念并证明其可通过“⊤‑operator”在多项式次oracle调用下求得；②提出“-operator”实现可保证最小边长的声誉区间；③揭示声誉最大化不可多项式求解而完整最小化可多项式求解；④针对均匀区间（ℓ∞‑球）给出对数次oracle调用的算法；⑤构建区间代数与混合整数线性规划（MILP）验证器相结合的遍历算法。

**🔧 技术方法**

使用的技术包括：区间代数、格理论、混合整数线性规划（MILP）建模、Marabou 验证器作为声誉与完整性oracle、TDS（Top‑Down Search）、BUS（Bottom‑Up Search）、SDE（Sequential Dichotomic Expansion）等遍历与排除算法。

**📊 数据集**

实验使用的网络是在 MNIST 与 Fashion‑MNIST 上训练的 784‑32‑10‑10 的 MLP，分别获得 94% 与 82% 的测试准确率。

**📈 对比分析**

与现有基于凸松弛或 MILP 的鲁棒性验证方法比较，实验表明均匀区间（ℓ∞‑球）算法（B‑BUS、B‑TDS）比通用区间算法快约十倍；B‑TDS 取得的声誉区间约为 TDS 的一半，且在大多数实验中能在较短时间内完成；完整区间算法（BUS、B‑BUS）获得的区间体积更大，但声誉最大化算法 TDS+SDE 在部分样本上出现 timeout。

**⚠️ 局限性**

主要限制包括：①仅适用于多层感知机且仅考虑轴对齐超矩形区间；②完整认证的最小化虽可多项式求解，但声誉最大化仍为 NP‑难且不可近似；③对非凸或更复杂网络结构（如卷积网络）未做扩展；④实验中的超参数（δ、时间阈值）对结果影响较大。

---

## 76. Mixture of Probes: Learning from Privileged Modalities in Multimodal LLMs Through Probing

**arXiv ID:** 2607.08839 | [PDF](https://arxiv.org/pdf/2607.08839v1)

**作者:** Dominick Reilly `[一作]` (Sony Group Corporation), Yuki Mistufuji `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在特权模态（辅助模态仅在训练时可用）下的多模态大型语言模型，提出 Mixture of Probes (MoP) 框架来利用训练时的辅助模态提升单模态推理性能。

**💡 创新点**

通过在统一模态编码器中引入可学习的中间层探针（分为模态特定与模态通用两类）并加入探针去耦合损失，显式分离模态特定与通用信息，促进跨模态知识迁移。

**🔧 技术方法**

采用探针机制、交叉注意力更新、探针去耦合损失、模态交错批处理、LoRA 微调大型语言模型（如 Qwen 2.5）以及 SigLIP 统一编码器。

**📊 数据集**

评估在 ADL 领域的 Ego-in-Exo Perception（egocentric、exocentric、depth 三种模态）和音乐理解领域的 Music-AVQA（audio、video 两种模态）上。

**📈 对比分析**

与单模态训练、朴素多模态训练及现有对齐式 MLLM（X-InstructBLIP、OneLLM）进行对比，MoP 在三种推理模态均超过基线 6–12% 及对齐模型约 5–10% 的准确率/平均分，显示显著提升。

**⚠️ 局限性**

仅适用于统一编码器的多模态设置，未扩展到异构编码器；仅关注单模态推理的特权模态场景，未考虑多模态推理；需要更大规模数据与实验验证。

---

## 77. SIDMA: Semantic Interleave Division Multiple Access Communication System

**arXiv ID:** 2607.08777 | [PDF](https://arxiv.org/pdf/2607.08777v1)

**作者:** Yunlu Wang `[一作]` (Beijing University of Posts and Telecommunications), Ping Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 71078 | [OpenAlex ID](https://openalex.org/A5100405781)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种Semantic Interleave Division Multiple Access (SIDMA)体系结构，用以在多用户并发传输中消除语义碰撞与语义崩塌。

**💡 创新点**

创新点在于：①引入排列操作实现特征域结构白化（semantic structural whitening），将语义协方差分散成近白噪声；②结合重要性感知自适应功率分配（ImpPA）模块，对核心语义块进行差异化加权；③通过极值理论证明在高维空间下实现渐近正交性。

**🔧 技术方法**

使用的技术包括：Swin Transformer编码器/解码器、伪随机排列交织、重要性评估与自适应功率网络、峰侧功率水平（PSL）种子选择算法、PSNR/SSIM评估指标。

**📊 数据集**

实验数据集为DIV2K图像恢复基准集（训练800张，测试100张）。

**📈 对比分析**

与OMDMA、DeepMA、SE以及传统MA（JPEG/JPEG2000）进行对比，结果显示SIDMA在0–14 dB低至中等信噪比下均取得最高PSNR/SSIM，尤其在多用户（最多100人）场景下仍保持优越的重建质量，且ImpPA模块显著提升性能。

**⚠️ 局限性**

局限性包括：在极高用户密度（K>50）时性能下降；对种子选择和参数调优敏感；实现复杂度较高，且目前仅在图像任务上验证，需进一步扩展至语音/文本等领域。

---

## 78. Optimal Transport-based Semantic Alignment for LLM-based Audio-Visual Speech Recognition

**arXiv ID:** 2607.09001 | [PDF](https://arxiv.org/pdf/2607.09001v1)

**作者:** Xugang Lu `[一作]` (National Institute of Information and Communications Technology), Hisashi Kawai `[通讯]` (National Institute of Information and Communications Technology)

**通讯引用:** 816 | [OpenAlex ID](https://openalex.org/A5114514387)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于最优传输（OT）的语义对齐框架，用于提升LLM驱动的音视频语音识别（LLM-AVSR），通过将声学与视觉特征对齐到LLM文本嵌入空间，并利用OT耦合作为软标签进行对比学习，改进多模态融合与解码效果。

**💡 创新点**

创新点在于：①显式使用OT在声学、视觉与文本嵌入空间之间构建结构化对应关系；②将OT耦合作为软伪标签指导对比学习，实现跨模态语义一致；③引入虚拟桶过滤非信息帧，避免无意义对齐；④在LLM-AVSR中首次将语义对齐作为前置模块。

**🔧 技术方法**

使用技术包括：Whisper声学编码器、AV‑HuBERT视觉编码器、LLaMA3.2‑3B LLM解码器；Sinkhorn迭代求OT耦合；对比学习软标签；多种融合策略（通道拼接、加法、交叉注意力、Q‑former）；LoRA微调；噪声增强训练。

**📊 数据集**

实验数据集为LRS3‑TED语音视频基准集（433小时训练、1小时验证、1小时测试），并在不同SNR（-5、0、5 dB）下模拟噪声。

**📈 对比分析**

与Whisper‑Flamingo、LLaMA‑AVSR、MMS‑LLaMA等先进基线对比，OT对齐模型在干净与噪声条件下的WER分别为：SNR‑5 7.16%、0 2.34%、5 1.11%、清洁 0.73%，显著优于基线（如MMS‑LLaMA在清洁为0.95%）。

**⚠️ 局限性**

局限性包括：需要手工调节多重超参数（OT正则化λ、温度τ、虚拟桶阈值s̃、对齐权重α）；OT与虚拟桶理论尚未完全解析；对齐仅针对单帧/文本，对长序列动态建模不足；在极低SNR下提升仍有限。

---

## 79. Dual-BEATs: Unlocking Zero-Shot Stereo Audio Perception in Audio Large Language Models via Dithering

**arXiv ID:** 2607.08800 | [PDF](https://arxiv.org/pdf/2607.08800v1)

**作者:** Shuo-Chun Lin `[一作]` (Academia Sinica), Hen-Hsen Huang `[通讯]` (Academia Sinica)

**通讯引用:** 397 | [OpenAlex ID](https://openalex.org/A5053932280)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计 Dual‑BEATs 架构，通过将左右声道分别输入同一语义编码器并加入无相关噪声，实现在不依赖专用空间编码器的情况下实现多模态 LLM 的立体声定位。

**💡 创新点**

创新点是利用无相关抖动噪声桥接归一化瓶颈，解锁 LLM 对连续空间信息的感知，而非传统的专用空间编码或模拟房间。

**🔧 技术方法**

采用双 BEATs 语义编码、静态无相关抖动噪声、量化低秩适配（QLoRA）对 LLM 进行空间指令微调。

**📊 数据集**

使用 AudioSet 语音/音乐标注数据，并对其进行空间化处理以构建左右声道 panning。

**📈 对比分析**

与未抖动基线对比，抖动模型在三分类定位任务中实现 97% 以上准确率，且在未见空间配置上实现零射击泛化。

**⚠️ 局限性**

限制包括语义 F1 下降约 5%，只针对单源平移声道的实验，未验证多源或真实双耳 HRTF 场景，且需大内存处理长序列。

---

## 80. Intrinsic Redundancy and Local Robustness in Finite $β$-Expansion Systems

**arXiv ID:** 2607.08795 | [PDF](https://arxiv.org/pdf/2607.08795v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 81. Pattern-Aware Graph Neural Networks for Handling Missing Data

**arXiv ID:** 2607.08915 | [PDF](https://arxiv.org/pdf/2607.08915v1)

**作者:** Minett Tran `[一作]` (San Jose State University), Taehee Jeong `[通讯]` (San Jose State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对GRAPE模型进行扩展，加入显式的缺失模式编码，直接从不完整表格数据中预测标签。

**💡 创新点**

首次在图神经网络框架下系统探索多种缺失模式嵌入策略（学习、随机、统计、层次化），并证明即使是最简单的随机投影也能获得接近学习嵌入的性能。

**🔧 技术方法**

使用图神经网络（GRAPE）、图注意力网络（GAT）、均值聚合、随机投影、层次化嵌入等技术，对缺失模式进行编码并融合到节点表示中。

**📊 数据集**

七个UCI公开数据集：Annealing、Hepatitis、Soybean、Thyroid、Voting、Physionet Sepsis、NHANES，均包含自然缺失。

**📈 对比分析**

与原始GRAPE（无缺失模式）、多种基于插补的全连接网络（均值、median、KNN、MICE）进行对比；缺失模式编码的模型平均提升了约17%均衡准确率、22% F1-macro、35% MCC，某些数据集（如Annealing）提升高达80%。

**⚠️ 局限性**

效果高度依赖数据集，某些数据集提升有限；样本量小导致方差大；仅评估了七个数据集，缺失机制未被系统控制；未探讨模型在更大规模或不同任务（回归、无监督）上的表现。

---

## 82. L2-Bench: An Evaluation Benchmark for Measuring LLM Capabilities in Second Language Education

**arXiv ID:** 2607.08842 | [PDF](https://arxiv.org/pdf/2607.08842v1)

**作者:** James Edgell `[一作]` (Oxford University Press), Martin Ku `[通讯]` (Oxford University Press)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了L2-Bench基准，包含1000+单轮任务-响应对，并基于12核心能力及31子能力的专家验证分类，提供可复现的rubric评分方法和LLM评判管线。

**💡 创新点**

首次以学习体验设计为构念，将教学能力转化为可操作的分类与rubric，并采用LLM-as-Judge实现自动化评估，形成可扩展且公开的AI教育评估框架。

**🔧 技术方法**

使用多大模型（Claude Opus 4.7、GPT‑5.4、Gemini 3.x 等）进行被测，Claude Sonnet 4.6 作为评判者；通过混合人机任务生成、统计检验（Krippendorff α、Cronbach α、混合效应模型、ANOVA）评估性能。

**📊 数据集**

基于221名专家共计1000+任务-响应对的数据集，任务涵盖33个上下文变量、目标与方法、参考答案，形成全面的L2教学设计评估资源。

**📈 对比分析**

采用排行榜和方差分析对九大模型进行比较，Claude Opus 4.7 最高得分85.5%，GPT‑5.4 84.1%，Gemini 3.1 Pro 83.4%；模型在结构化任务表现强，而在开放式、交互式任务表现相对较弱，排名总体稳定。

**⚠️ 局限性**

局限在于仅覆盖英语/UK/US L2情境，单轮交互限制、负面评价稀少导致总分低估；评判者一致性低、评分噪声高，且缺乏多语言、多文化及多轮对话的验证。

---

## 83. SeedSmith: LLM-Driven Seed Synthesis for Directed Fuzzing

**arXiv ID:** 2607.08949 | [PDF](https://arxiv.org/pdf/2607.08949v1)

**作者:** Junmin Zhu `[一作]` (UC Santa Barbara), Giovanni Vigna `[通讯]` (UC Santa Barbara)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一套基于LLM的种子生成前端，通过从目标sink函数出发，迭代探索代码、解析间接调用、推断崩溃前置条件，并输出可编程的Python脚本来生成满足触发条件的初始种子，从而显著提升有向模糊测试的崩溃检测效率。

**💡 创新点**

创新点主要有：①将LLM拟人化为“安全分析师”，利用迭代查询和工具代理主动推断间接调用与崩溃前置条件；②采用路径压缩+上下文感知的代码搜索工具，使LLM能够在有限上下文内获取完整函数体或类型定义；③生成Python脚本而非裸字节，自动构造复杂输入格式，解决传统随机突变难以满足结构约束的问题。

**🔧 技术方法**

技术手段包括：LLM（Anthropic Claude/ChatGPT）+代理框架、CodeQL静态分析与自定义索引、上下文感知的grep/搜索工具、Python脚本生成与迭代反馈循环、Sanitizer（ASan/TSan/UBSan）验证、以及基于距离度量的有向模糊器（AFL++, AFLGo, AFLRun, FairFuzz）。

**📊 数据集**

数据集：Magma（23个C/C++漏洞）和ARVO（115个跨项目真实漏洞挑战），两者涵盖多种输入格式（文本、图像、协议、压缩包等）。

**📈 对比分析**

评估方法：将生成的种子与默认种子、OSS‑Fuzz种子以及Locus等基线在同一模糊器上进行10次10小时实验，使用RMST与几何平均speedup进行比较。结果显示：在Magma上，所有四款模糊器的平均speedup为11.51×（AFL++）至14.66×（AFLGo）；在ARVO上，speedup为3.09×（AFL++）至3.02×（AFLRun）。此外，生成种子帮助发现16个此前未触发的漏洞。

**⚠️ 局限性**

局限性：①系统性能受LLM代码理解能力限制，误判或漏判会导致种子质量下降；②需要手工提供sink函数，若根因位于其他函数则效果受限；③不自带sink定位功能，需与静态漏洞检测或学习型漏洞模式结合；④在极大项目或CodeQL不兼容时，路径预处理可能失败；⑤LLM调用和seed生成虽相对低成本，但仍需额外算力与时间。

---

## 84. RaMark: Radioactive Watermarking for Generated Tabular Data

**arXiv ID:** 2607.09000 | [PDF](https://arxiv.org/pdf/2607.09000v1)

**作者:** Xin Che `[一作]` (McMaster University), Jian Pei `[通讯]` (Duke University)

**通讯引用:** 23778 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现一种名为 RaMark 的放射性水印方法，用以在生成的表格数据中嵌入可在重训练攻击后仍可检测的水印。

**💡 创新点**

创新点在于：将 sinusoidal 依赖关系作为数据分布的内在成分嵌入；利用“放射性”概念保证水印随数据分布传播；通过理论分析阐明分布保持与水印可检测性之间的直接关联，并给出可控的水印强度-数据效用权衡。

**🔧 技术方法**

核心技术包括：
- 采用扩散模型的逆向采样过程加入水印引导（watermark‑guided sampling），
- 通过投影到二维空间并转换为离散时间信号后使用 Lomb‑Scargle 周期图进行频谱检测，
- 理论上证明分布相似性→信号相似性→频谱相似性，从而保证放射性。

**📊 数据集**

实验使用了两个公开的真实世界表格数据集：Higgs Small（分类）和 House‑16H（回归）。

**📈 对比分析**

与 7 种主流水印方法（S2R2W、TabularMark、WGTD、PKF、TabWak、MUSE、B2Mark）在 10⁵ 业主大规模验证场景下进行比较。评估指标为 detectability 和 traceability 的 AUC；在重训练攻击和四类数据修改攻击下，RaMark 的 AUC 始终接近 1.0，显著优于所有基线，且在保持相同的 ML 效率（≤1% 下降）下实现最高的可检测性与可追溯性。

**⚠️ 局限性**

局限性包括：
- 需要至少两列连续值特征，无法直接处理纯离散或文本属性；
- 水印强度 α 的设置需要权衡数据效用和可检测性，过大或过小均会影响性能；
- 在极端修改攻击（如大幅度噪声或大量行/列删除）下仍会出现一定的水印弱化。

---

## 85. TSRouter: Dynamic Modality-Model Selection for Time Series Reasoning

**arXiv ID:** 2607.08940 | [PDF](https://arxiv.org/pdf/2607.08940v1)

**作者:** Fangxu Yu `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于异构图的动态路由框架TSRouter，用于在时间序列推理任务中同时选择最佳输入模态（文本或可视化）和最合适的基础模型（LLM或VLM）

**💡 创新点**

创新点在于：①将任务、查询、模态和模型视为四类节点构建异构图，显式建模它们之间的多维交互；②将路由任务化为候选评分问题，通过GNN学习节点表征并使用soft目标训练实现全局排序；③实现了零样本泛化，可无须再训练即可插入新模型或新任务，并通过成本加权优化实现成本与性能的可调权衡

**🔧 技术方法**

主要技术包括：预训练文本嵌入（如Qwen3-embedding）、异构图神经网络（Heterogeneous Graph Transformer）进行节点表征学习、两层MLP评分头、softmax/KL散度训练目标，以及基于成本的效用函数实现多目标路由

**📊 数据集**

使用了TSRBench作为主要评测基准（覆盖感知、推理、预测、决策四大类、共15个子任务），并在两个未见任务（TSQA中的缺失值预测、MTBench中的相关性预测）上检验泛化能力

**📈 对比分析**

与十个基线（规则基、Elo、MF、KNN、GraphRouter、Hybrid LLM、RouterDC、CausalLM、Router-R1等）进行对比，TSRouter在总体准确率上提升16%–46%相对基线，并且在成本维度保持竞争力（通过α参数可在成本-准确率之间平衡，且在大多数预算下仍优于所有基线)

**⚠️ 局限性**

局限性包括：①仍需依赖大量交互数据进行训练，对低频查询可能表现欠佳；②GNN层数和嵌入维度需要调参，过深或维度过大易出现过平滑；③当前仅针对LLM/VLM两类基础模型，未考虑其他类型推理器；④在极低成本场景下仍可能无法满足实时性需求

---

## 86. Sensitivity-Aware Thresholding and Token Routing for Activation Sparsification in Large Language Models

**arXiv ID:** 2607.08991 | [PDF](https://arxiv.org/pdf/2607.08991v1)

**作者:** Bishmoy Paul `[一作]` (Santa Clara University), Hoeseok Yang `[通讯]` (Santa Clara University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了两种用于大规模语言模型推理加速的方法，分别是感知阈值化（SATS）和基于标记级别的路由（Token Routing），旨在提升稀疏化和动态路径选择的质量-吞吐量平衡。

**💡 创新点**

创新点在于：①用局部MLP输出失真度作为阈值校准依据，取代传统的百分位阈值，构建了“感知阈值化”方案；②提出只依据当前标记身份进行路由的轻量级表查方式，使得每个标记可选择稠密路径或稀疏路径，显著提升了静态稀疏模型的性能。

**🔧 技术方法**

技术包括：感知阈值化（SATS）—基于本地失真度预算选择阈值；CATS框架的门控阈值稀疏化；token‑routing—使用标记级别的损失差异表进行动态路径决策；LoRA微调、Bf16/FP32推理、Triton自定义内核等实现细节。

**📊 数据集**

实验使用llama 3.1 8B和Qwen 3 8B两大开源LLM，评估指标覆盖七项下游任务（PIQA、OpenBookQA、ARC‑Easy、Winogrande、Hellaswag、SciQ、BoolQ）以及Wiki‑Text2和RefinedWeb的困惑度。

**📈 对比分析**

与传统CATS的比较显示：在相同实现稀疏率下，SATS在平均下游准确率和困惑度上均优于CATS；token‑routing进一步提升了准确率并保持高吞吐率（相对密集模型仍高于稠密推理）。总体而言，SATS+路由在保持大约50%稀疏率时，兼顾了更高质量和更快推理。

**⚠️ 局限性**

局限性包括：①对标记级别的路由需要预先计算并存储损失差异表，且仅利用标记身份，忽略上下文影响；②在高稀疏率（>70%）时效果不佳，稀疏化仍可能导致显著性能下降；③微调时需额外设计LoRA‑aware稀疏内核，实施复杂；④实验仅覆盖两大模型，泛化到更大规模或不同架构仍需验证。

---

## 87. C-GAP: Class-Aware and Online Prompting Improves Vision-Language Models on Imbalanced Classes

**arXiv ID:** 2607.09008 | [PDF](https://arxiv.org/pdf/2607.09008v1)

**作者:** Francis Fernandez `[一作]` (San Diego State University), Salimeh Sekeh `[通讯]` (San Diego State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无标注、无训练、只改写文本提示的框架C-GAP，用于提升冻结开词检测器在少数类中的检测性能。

**💡 创新点**

创新点在于利用检测器对少数类AP@0.5的反馈作为迭代提示优化的目标，并通过三桶三分法实现自适应的提示改进，彻底解决少数类检测在小标签空间下的稀疏标注难题。

**🔧 技术方法**

核心技术包括：①使用LLM生成场景描述（Scene Description）和类别数量（Class Quantity）两种提示，再拼接成Composite Caption；②基于冻结开词检测器的AP@0.5评估迭代生成的提示；③三桶三分（regenerate、tentative、keep）策略与早停机制。

**📊 数据集**

在三组数据集上验证：MS‑COCO（公交为少数类）、Cityscapes（卡车为少数类）以及真实交通监控数据Chula Vista（自行车为少数类）。

**📈 对比分析**

与各类开词检测器（Grounding DINO、OmDet‑Turbo、OWLv2、YOLO‑World）及静态提示（Scene Description、Class Quantity、Composite Caption）相比，C‑GAP在10/12配置中提升了少数类AP@0.5，最高可达+14.4个百分点；整体mAP变化保持在1%以内。

**⚠️ 局限性**

局限性包括：①对LLM生成提示的依赖，若提示质量欠佳会影响收敛；②三桶阈值需手动设定，可能对不同后端和数据分布不够鲁棒；③目前仅针对单一少数类，未扩展到多类长尾场景。

---

## 88. Model Agnostic Graph Prompt Learning for Crystal Property Prediction

**arXiv ID:** 2607.08996 | [PDF](https://arxiv.org/pdf/2607.08996v1)

**作者:** Shrimon Mukherjee `[一作]` (Indian Association for the Cultivation of Science), Niloy Ganguly `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出多层图软提示学习框架，在节点层和图层分别加入可学习的软提示，以捕捉晶体属性的潜在化学和结构特征。

**💡 创新点**

在GNN中首次结合节点级和图级软提示，轻量化且可迁移地学习潜在特征，显著提升预测性能，且仅增加约10k参数。

**🔧 技术方法**

采用可学习软提示（soft prompts）、注意力聚合、图神经网络（CGCNN、ALIGNN、Matformer、PotNet等）以及跨属性知识迁移方法。

**📊 数据集**

使用公开晶体基准数据集Materials Project和JARVIS‑DFT，涵盖形成能、能隙、模量等多种属性。

**📈 对比分析**

与原始SOTA模型在相同训练设置下对比，prompt-infused 版本平均提升4–15%，在少样本任务中提升3–6%；相对传统的网络加宽/深度/适配器等改进方案，prompt学习更轻量、效果更佳。

**⚠️ 局限性**

仍需手动选择提示维度与数目（k），对非晶体或非周期性结构的适用性未验证，跨属性迁移效果受属性相关性限制。

---

## 89. StreamDQ: Near-Memory Weight DeQuantization in Custom HBM for Scalable AI Inference Acceleration

**arXiv ID:** 2607.08993 | [PDF](https://arxiv.org/pdf/2607.08993v1)

**作者:** Minki Jeong `[一作]` (SK Hynix), Hoshik Kim `[通讯]` (SK Hynix)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 StreamDQ，一种在 HBM 基底晶圆中集成轻量级去量化块（DQB），实现对权重的在线去量化，从而消除 GPU CUDA 核心上的去量化指令和相关的芯片内外数据流动。

**💡 创新点**

创新点在于：①利用轻量级侧带标签（sideband tag）在标准内存加载时即完成去量化；②将去量化逻辑迁移到 HBM 基底，避免虚拟地址到物理地址转换和跨通道通信；③通过伪通道感知布局和 S/Z 缓存，支持多种 INT4/INT8 → FP16/BF16 的格式转换，并在面积、功耗和热阻方面保持低开销。

**🔧 技术方法**

技术实现包括：在 HBM 基底实现 3-bit 标记的 DQB；使用 S/Z 请求表和缓冲区实现按组量化参数的快速获取；基于共享 FP32 ALU 的去量化运算与轻量级位映射实现 FP↔FP 与 INT↔FP 的转换；通过 12 nm CMOS 逻辑工艺实现 0.127 mm²/块，0.355 W/块；使用 StreamDQ‑Sim 进行性能与能耗仿真，并通过 Nsight/Accel‑Sim 校准真实 GPU 行为。

**📊 数据集**

实验数据集为三大 LLM：LLaMA‑3.1‑8B‑Instruct、Qwen3‑8B、Mistral‑7B‑Instruct‑v0.3；使用量化配置 W4A16、W8A16，并在 NVIDIA A100 PCIe 40 GB GPU 上进行混合精度 GEMM 与端到端推理评测。

**📈 对比分析**

比较方法：将 StreamDQ 与软件级融合去量化/混合精度 GEMM（GPTQ、AWQ‑v1/v2、TorchAO）以及传统 CUDA 核心去量化进行对比；在混合精度 GEMM 上实现最高 7.08× 的速度提升、90.23% 的能耗下降；在端到端推理上达到 54.68% 的延迟缩短和 2.20× 的解码吞吐率提升；在不同批次大小下分别展示了 StreamDQ 在计算受限场景的优势。

**⚠️ 局限性**

局限性：需要在 HBM 基底集成 DQB，当前仅在实验室仿真与 12 nm CMOS 设计下验证；对激活量化场景不适用，仅针对权重仅量化；侧带标签的实现依赖于 GPU 侧的区域查找表，若模型权重布局不规则会增加表大小；在极大批次或极高精度混合模式下，S/Z 缓存命中率可能下降，导致额外的 DRAM 访存。

---

## 90. The Patchwork Problem in LLM-Generated Code

**arXiv ID:** 2607.08981 | [PDF](https://arxiv.org/pdf/2607.08981v1)

**作者:** Viraaji Mothukuri `[一作]` (Kennesaw State University), Reza M. Parizi `[通讯]` (Kennesaw State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过图模型定义结构一致性不变式，提出八类 LLM 代码生成的结构失效分类，并构建混合验证框架（结合成熟静态分析与自研跨图检测），在 336 次 GPT‑4o/Claude 生成任务和 43 个真实 AI 代码仓库上评估，发现 97% 的失效未被常规 CI 检测捕获；

**💡 创新点**

创新点在于：①把结构一致性正式化为跨文件图不变式；②建立跨图、跨语言的八类失效分类；③设计混合验证架构，利用现有工具做子问题，针对跨图缺口自研检测器；④通过实证评估展示不同模型在失效分布上的差异；

**🔧 技术方法**

主要技术包括：图属性建模（导入图、调用图、依赖图、配置图、资源图、控制流图、路由图、模式图）；成熟工具如类型检查、SAST、Lint；自研检测算法（如配置不一致、依赖幻觉、符号解析、跨文件契约、资源完整性、控制流健全性、安全结构回归）以及基于层次的混合策略；

**📊 数据集**

数据集由 10 个已维护的开源仓库（Python/Django、FastAPI；TypeScript/Express、Next.js）抽取的 60 个任务（按复杂度划分）生成 336 次 LLM 代码，以及 43 个公开的全 AI 生成仓库（共 1,581 文件）构成评估样本；

**📈 对比分析**

与四类基线（类型检查、测试、SAST、正则）对比，框架检出 67 个失效，97% 未被基线发现；按类别精度 100%（符号解析、内部 API、依赖、配置、资源、跨文件契约、控制流、路由安全）；模型对比显示 GPT‑4o 与 Claude 在失效分布上显著不同；提示策略影响失效数量，跨层任务最易失败；

**⚠️ 局限性**

局限性包括：仅覆盖 8 类失效，缺乏对更细粒度或语言（如 Java、Go）失效的支持；检测依赖外部服务（如 Docker、Kubernetes）时需手工扩展；对极大仓库性能仍受图构建瓶颈；对基于迁移学习的模型生成细粒度错误的检测仍有待改进。

---

## 91. Optimal Top-$k$ Identification from Pairwise Comparisons

**arXiv ID:** 2607.08979 | [PDF](https://arxiv.org/pdf/2607.08979v1)

**作者:** Motti Goldberger `[一作]` (Yale University), Nils Rudi `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对噪声双项比较的固定置信度 top‑k 识别算法，能够在错误概率不超过 δ 的前提下最小化期望比较次数。

**💡 创新点**

创新点在于将识别问题的下界结构化为两人零和游戏：设计者分配比较次数，敌人选择“边界对”；利用该 saddle‑point 结构实现在线 primal–dual 学习，得到渐近最优的采样分配。

**🔧 技术方法**

技术上使用了：
- 隐含效用模型（latent‑utility）下的指数族比较模型；
- KL 投影与 Danskin 定理求导；
- 经验风险最小化（MLE）估计；
- Entropic FTRL primal–dual 更新与 C‑tracking 采样策略；
- 自适应停止阈值基于似然比统计量。

**📊 数据集**

实验采用三种合成数据集：随机效用、等间距效用和满足 SST 的偏好矩阵；所有数据均通过 Bradley–Terry 模型生成双项比较结果。

**📈 对比分析**

与现有基线（SEEKS、SEEKS‑v2、Active Ranking）对比。实验表明在 δ=0.01 的条件下，该方法在多数实例中取得更低的平均停止次数，尤其当 n 增大或 k 变大、效用区间固定时表现最优；与离线最优分配的 oracle 结果相近。

**⚠️ 局限性**

主要局限在于中等 δ（如 0.01）时停止阈值相对保守，导致实际停止次数可能高于某些基线；此外，算法对较大 n 的计算成本仍较高，尤其在梯度估计和 KL 投影上需要 O(n²) 复杂度。

---

## 92. MultiView-Bench: A Diagnostic Benchmark for World-Centric Multi-View Integration in VLMs

**arXiv ID:** 2607.08970 | [PDF](https://arxiv.org/pdf/2607.08970v1)

**作者:** Hantao Zhang `[一作]` (Yale University), Zhuoran Yang `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 MultiView-Bench，一个评估视觉语言模型（VLM）多视角空间推理与全局坐标理解的基准测试，并提供可扩展的数据生成管线。

**💡 创新点**

创新点在于：①以可见的全局坐标系引入 allocentric 推理需求；②构造可控自由度（DoF）任务变体，精准诊断 VLM 的多视角整合与轴向识别缺陷；③设计 ViewNavigator 多代理框架，结合贝叶斯信念聚合与主动视角选择显著提升 VLM 的三维推理性能。

**🔧 技术方法**

采用 Blender 程序化渲染、LLM 规划器、VLM 视觉推理、Dirichlet 贝叶斯聚合、微抖动视角和主动采样技术。

**📊 数据集**

使用合成几何体（立方体、球体、圆柱体等）与真实三维资产库 3DCoMPaT++，在 5 个主任务变体（各 100 个实例）及 20 个扩展变体中生成数据。

**📈 对比分析**

在 7 款前沿 VLM（Claude 3.7/4、Gemini 2.5 Flash/Pro、GPT‑4o、GPT‑5、GPT‑o3）上进行比较；基线模型在 DoF=3 时仅略高于随机（≤20%），而 ViewNavigator 在同样视角预算下将准确率提升至 19–61%（最强 GPT‑5 由 49% 提升到 61%，可扩展至 85%）。

**⚠️ 局限性**

局限性：①现有 VLM 对非标准坐标方向与颜色偏差高度敏感，难以通过视觉直接推理轴向；②在极端三维自由度下仍难以达成高精度推理；③ViewNavigator 需要额外的推理循环与多次视角查询，导致推理时间和计算成本提升。

---

## 93. Better Harnesses, Smaller Models: Building 90% Cheaper Agents via Automated Harness Adaptation

**arXiv ID:** 2607.08938 | [PDF](https://arxiv.org/pdf/2607.08938v1)

**作者:** Chenyang Yang `[一作]` (Carnegie Mellon University), Christian Kästner `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种自动化适配agent harness的方法，以提升小语言模型在业务任务上的性能并降低成本。

**💡 创新点**

提出了基于失效模式映射的框架和一个meta-agent优化器，能够自动从失败轨迹中发现并执行适配策略，从而使SLM几乎等价于LLM。

**🔧 技术方法**

使用meta-agent、Prompt工程、工具集改造、循环控制以及遗传搜索等技术来实现自动化 harness 优化。

**📊 数据集**

采用7个业务导向的agent任务数据集，来源于TheAgentCompany、LOCA-Bench、WebGenBench、WebArena、RefactorBench等公开基准。

**📈 对比分析**

通过将SLM与通用 harness、优化 harness 以及 frontier LLM 通用 harness 进行对比，评估准确率、成本与延迟，实验表明优化 harness 的 SLM 在 16/21 任务-模型组合上提升显著，7 对中几乎关闭性能差距，最佳 SLM 在仅 4% 成本下恢复 89% LLM 性能。

**⚠️ 局限性**

局限性包括实验仅覆盖特定任务与模型，优化器实现可能不易迁移，对黑盒 frontier LLM 的依赖，适配仅针对单一 harness，且在多样性任务和弱 SLM 上效果有限。

---

## 94. Vision Transformers Learn Gestalt-Like Figure-Ground Cues from Natural Images

**arXiv ID:** 2607.08932 | [PDF](https://arxiv.org/pdf/2607.08932v1)

**作者:** Matthias Tangemann `[一作]` (University of Toronto), Sven Dickinson `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在Vision Transformer（ViT）预训练模型的中间层特征上训练线性探测器，对自然图像和人工合成图像进行形状基的图形-背景分割评估，系统检验了围绕性、凸性和对称性三种经典形状线索是否被模型捕捉；

**💡 创新点**

创新点在于首次在ViT上开展大规模（25个不同训练目标和规模的模型）线性探测实验，证明了围绕性与凸性在自然图像和合成线索上均能被零样本迁移，并揭示了对称性仅在无纹理条件下可被解码，进一步表明ViT能够学习与人类相似的Gestalt图形-背景线索；

**🔧 技术方法**

技术主要包括：冻结ViT特征、使用空间先验的带位置偏置的逻辑回归探测器、信息增益解释（IGE）评估指标、以及对三类合成刺激的自动生成与控制；

**📊 数据集**

使用的数据集为MSRA-10K（自然单物体分割）进行自然图像测试，以及基于Infinite DSprites、DTD纹理与自定义几何的三种合成数据集（围绕性、凸性、对称性）进行线索孤立评估；

**📈 对比分析**

比较方法采用信息增益解释（IGE）和准确率，结果显示：大多数模型在自然图像上的IGE可达70–90%，在围绕性和凸性合成测试中零样本探测器接近或超过专门训练探测器的性能；对称性在纹理条件下表现低于空间先验，而去纹理后可达到近乎完美；此外，模型大小、预训练任务（对象识别、视觉‑语言对齐、掩码图像建模、Self‑Distillation）与层次深度均对性能有显著影响；

**⚠️ 局限性**

局限性包括仅使用线性探测器，可能低估模型对更复杂线索的潜在编码；仅考察三种经典形状线索，未涵盖面积、低区、熟悉度等其他人类图形-背景因素；未与人类行为数据直接对照，缺乏人机性能的直接比较；

---

## 95. BlockServe: Block-Grained Continuous Batching for High-Throughput Diffusion LLM Serving

**arXiv ID:** 2607.08930 | [PDF](https://arxiv.org/pdf/2607.08930v1)

**作者:** Yuanjie Zhu `[一作]` (University of Illinois Chicago), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BlockServe 框架，实现对扩散型大型语言模型（dLLM）的高吞吐量连续批量推理。

**💡 创新点**

创新点在于：1) 基于块级预emption 的块粒度调度，立即回收已完成请求；2) 混合状态内存管理，通过逻辑位置对齐和 gather‑scatter 索引实现不同块状态的并行执行；3) 采用基于 token 预算的自适应入队控制，动态调节有效批次容量。

**🔧 技术方法**

主要技术包括块粒度调度、混合状态内存管理、gather‑scatter 索引、token‑budget 入队策略、Dual Cache 与 Parallel Decoding 加速。

**📊 数据集**

在 LLaDA‑8B‑Instruct 与 Dream‑v0‑Instruct‑7B 两个模型上，使用 GSM8K、HumanEval、MBPP、MATH、TruthfulQA‑Gen 五个基准数据集进行评估。

**📈 对比分析**

与 Fast‑dLLM 基线相比，BlockServe 在所有基准上实现 1.9–10.6 倍的吞吐量提升，且保持与基线相当的生成质量；同时在高批量情况下显著降低尾部延迟。

**⚠️ 局限性**

局限性包括：1) 目前仅在离线批量推理环境下验证，缺乏对在线实时请求到达的评估；2) 依赖于块长度和 prompt 长度排序，需针对不同模型/硬件进行调优；3) 对于极短生成任务，块粒度仍可能导致细粒度调度开销。

---

## 96. Event-Based Token Sequences for Audio-Conditioned Music-Game Level Modeling

**arXiv ID:** 2607.09095 | [PDF](https://arxiv.org/pdf/2607.09095v1)

**作者:** Ke Zhang `[一作]` (Japan Advanced Institute of Science and Technology), Kokolo Ikeda `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于事件的序列到序列模型，用音频与元数据生成音乐游戏关卡的节拍级别事件序列。

**💡 创新点**

①将关卡拆解为交替出现的节拍移位和事件标记，显式表示事件间的相对时序；②引入Audio Contribution Score（ACS）衡量音频对生成的实际贡献；③使用Transformer自回归解码器在多模态输入下进行级别生成。

**🔧 技术方法**

预训练音频编码器（Whisper或MERT）、多模态融合模块、基于Transformer的自回归解码器，以及自定义事件词表和节拍移位编码。

**📊 数据集**

来自商用节奏游戏maimai的4,187条关卡，1,018首歌曲，经过固定节拍长度切片得到约26,600条训练/验证/测试序列。

**📈 对比分析**

与两种经典帧级基线（DDC、GeneLive!）在事件级F1、帧级F1等指标上对比；实验显示在事件级F1上模型可提升约30%（如0.527 vs 0.384），Whisper编码器表现优于MERT，且ACS表明音频贡献显著。

**⚠️ 局限性**

受限于事件词表粗粒度、缺乏空间/物理约束、对高难度或高BPM场景生成多样性不足，以及生成序列易出现极端密度或循环重复等失真。

---

## 97. Pitfalls and Remedies for Multi-Task Bayesian Optimization

**arXiv ID:** 2607.09073 | [PDF](https://arxiv.org/pdf/2607.09073v1)

**作者:** Carl Hvarfner `[一作]` (Meta), Eytan Bakshy `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了贝叶斯优化中多任务高斯过程（MTGP）转移学习的结构性缺陷，提出并验证了三种保守的修复策略。

**💡 创新点**

创新点在于首次系统地揭示并理论证明两大问题：有限样本标准化误差导致相关性低估，以及信息理论下限导致相关性推断困难，并给出可落地的三项改进。

**🔧 技术方法**

使用技术包括多任务GP（ICM）、理论分析（Fisher信息、Cramér‑Rao界限）、正相关约束、任务均值/尺度参数化以及实验对照。

**📊 数据集**

实验数据集涵盖合成仿真（Forrester、Hartmann、Ackley等）、半合成 HPO（LCBench）、深度学习调参（pd1）和 LLM 调参（ifeval）等多种场景。

**📈 对比分析**

方法比较中，三种修复策略在简单仿真问题上能恢复到单任务GP水平，甚至在 LCBench HPO 任务中优于标准 ICM、RGPE、QuantileBO 等；但在更难实例或大多数变体中仍出现负迁移。

**⚠️ 局限性**

局限性包括：理论分析仅覆盖正 affine 任务，难以推广到非 affine 情况；修复策略只能缓解而非根除困难；缺乏针对低源预算下何时放弃迁移的明确准则。

---

## 98. Agentic Proof and Property-Based Testing via Property-Templates in Data-Intensive Computing

**arXiv ID:** 2607.09072 | [PDF](https://arxiv.org/pdf/2607.09072v1)

**作者:** Seongmin Lee `[一作]` (University of California Los Angeles), Miryung Kim `[通讯]` (University of California Los Angeles)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了可参数化的属性模板，并在 Apache Spark 上同时进行 Lean 4 形式化证明与可执行的属性基测试，以验证数百条正确性属性。

**💡 创新点**

创新之处在于将相同结构的属性族抽象为模板，统一生成证明与测试，显著提升验证效率并显著降低证明幻觉与测试意图误差。

**🔧 技术方法**

使用 GPT‑5.5 进行模板填充与自动证明/测试合成，Lean 4 进行机理化证明，PySpark 运行 PBT，构建双轨 agentic 验证框架。

**📊 数据集**

基于 Apache Spark 的核心 API 作为模型，自动生成 400 条候选属性（每族 100 条），并在真实 PySpark 环境中执行测试。

**📈 对比分析**

与无模板基线相比，模板将证明成功率提升至最高 2.6×（平均 1.6×），成本下降 23%；PBT 意图误差从 22 降至 1，合成成本降低 5.7×，并在代码覆盖率上与最先进 fuzzer 相当。

**⚠️ 局限性**

局限性包括仅覆盖四个属性族，模型与实现之间仍存在差距，LLM 生成的属性与测试需人工复核，模板对非结构化或更复杂属性支持有限。

---

## 99. Automating Just-In-Time Python Type Annotation Updating

**arXiv ID:** 2607.09054 | [PDF](https://arxiv.org/pdf/2607.09054v1)

**作者:** Zhipeng Xue `[一作]` (Zhejiang University), Shanping Li `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了用于 Python 的 Just‑In‑Time 类型注解更新工具，自动根据代码变更生成新的类型注解。

**💡 创新点**

首次定义此任务并构建首个类型注解更新数据集；通过将 LLM 与检索增强生成（RAG）结合，利用项目历史和相似代码变更指导模型，显著提升更新准确率。

**🔧 技术方法**

使用 GPT‑3.5‑turbo 进行推理，CoditT5 对代码变更进行向量编码，Chroma 做向量检索；实现 Retrieval、Reasoning、Updating 三个 LLM 代理，实现 RAG 推理流程。

**📊 数据集**

36,796 条代码-注解共变样本，来源于 450 个 GitHub 项目，形成首个 Python 类型注解更新数据集。

**📈 对比分析**

与 TypeGen、Type4py、pytype、HiTyper、Tiger 等基线方法对比，工具在 500 条测试样本上取得 41.9% 的准确率提升，召回率提升 27.2%，最终正确更新 359 条注解；在 10 个真实项目中 20 条 PR 被开发者接受。

**⚠️ 局限性**

性能受检索质量限制，难以处理新项目或缺乏历史变更、极为复杂的代码变更；在与静态检查器结合时虽然精度提升，但召回率下降，表明需要更好的误差修正机制。

---

## 100. COBS: Cumulant Order Block Sparse Attention

**arXiv ID:** 2607.09052 | [PDF](https://arxiv.org/pdf/2607.09052v1)

**作者:** Alexander Tian `[一作]` (MatX), Akshay Mishra `[通讯]` (MatX)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了块稀疏注意力的选择机制，提出COBS方法通过存储压缩的二阶统计量改进块选择，显著提升长上下文检索性能。

**💡 创新点**

创新点在于用累积展开证明第一阶选择器无法捕捉块内方差，并提出压缩协方差的第二阶块选择器，使块稀疏注意力逼近密集注意力。

**🔧 技术方法**

使用了块稀疏注意力框架、累积展开、低秩分解、子空间投影、FP4量化、NoPE编码、GQA和NSA三支路结构。

**📊 数据集**

采用32k RULER长上下文检索基准、OpenBookQA/PIQA/HellaSwag/ARC/TriviaQA/WinoGrande短上下文常识评测以及LongCrawl64自然语言建模数据集。

**📈 对比分析**

在NSA MLP基线(0.2999)与密集注意力(0.9040)之间，COBS在32k RULER上得到0.8195，闭合约86%缺口；短上下文保持与密集相当；KV缓存读取量仅为NSA基线1.21倍，远低于密集。

**⚠️ 局限性**

局限性：实验仅在约1.2B规模、4k预训练长度的受控设置；COBS仍需读取压缩统计量；NoPE与RoPE差异可能影响对比；RULER式SFT的适用性有限；KV读取计量不直接等同于推理速度。

---

## 101. Variable-Length Generative Protein Design via Generalized Poisson Flow

**arXiv ID:** 2607.09039 | [PDF](https://arxiv.org/pdf/2607.09039v1)

**作者:** Chaoran Cheng `[一作]` (University of Illinois Urbana-Champaign), Ge Liu `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种通用的可变长度生成框架——Generalized Poisson Flow（GPF），能够在蛋白质结构、序列、基序装配以及肽共设计等多种任务中不需要预先指定长度进行生成。

**💡 创新点**

创新点在于：①用在homogeneous Poisson过程建模长度变化，并将其与任意维度的“内在长度”模态（连续、离散、黎曼）耦合；②推导出无仿真、可解析的负对数似然目标，并给出联合分布恢复与生成分布的KL上界；③实现了单一理论框架覆盖所有模态和多模态联合生成。

**🔧 技术方法**

主要技术包括：基于广义泊松过程的长度进化、后验边缘化得到可训练的速率目标、流匹配或扩散的“内部生成器”来处理模态内部的坐标/类型更新、负对数似然学习与生成器KL上界证明、以及自回归式插入采样。

**📊 数据集**

实验使用了 PDB、AFDB（蛋白质结构）、UniRef50（蛋白序列）以及肽共设计所需的配体结构数据集。

**📈 对比分析**

与固定长度模型（Proteina、DPLM、EvoDiff、ESM3、SCISOR、PepFlow等）对比，GPF 在无条件结构生成中提升 3.9%–18.5% 设计可行性，序列生成中更接近 UniRef50 的 pLDDT 与二级结构分布，基序装配中在 10/16 任务中排名第一并显著增加独特成功率，肽共设计中在 4/8 评估指标上表现最佳，并能自行学习长度分布。

**⚠️ 局限性**

主要限制是对调度器（scheduler）的敏感性，在多模态设置中不同模态间的交互可能导致采样效率或分布偏差；此外，现有采样策略（如 τ‑leaping、局部路径）仍需进一步改进以减少潜在的权衡。

---

## 102. Optimal Metric Distortion for Learning-Augmented Matching on the Line

**arXiv ID:** 2607.09038 | [PDF](https://arxiv.org/pdf/2607.09038v1)

**作者:** Jabari Hastings `[一作]` (Stanford University), Marena Richter `[通讯]` (University of Bonn)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

在一维线性度量空间上，提出一种学习增强的匹配机制，输入为代理对物品的序数偏好以及一个预测完美匹配，输出一组完美匹配；

**💡 创新点**

实现了最佳双重世界失真保证：在预测准确时达到一致性1（完全最优），在预测错误时保持鲁棒性3（与无预测机制相同的最优常数失真），解决了之前的开放问题；

**🔧 技术方法**

基于线性排序恢复的“OrderMatch with Reserved Outliers”框架，对可排序的物品集合与极端物品进行分离，并利用预测匹配对出界物品与同一顶级偏好的代理进行细化排序；分析中运用了未交叉不等式、排序变换等技术；

**📊 数据集**

论文未使用公开数据集，所有证明和实验均为理论分析；

**📈 对比分析**

与传统无预测机制（失真3）以及随机/确定性策略的已知下界进行比较，证明最坏情况下失真不超过3，且在预测完美时失真为1；在中间情况给出上界min{预测失真,3}；

**⚠️ 局限性**

局限性包括仅适用于一维线性度量空间；对一般度量空间的失真上界仍为O(n)，下界为Ω(log n)，无法给出更紧的常数；此外需要完整的预测匹配，未考虑预测误差的度量或多种预测形式。

---

## 103. Study of Mixed-Integer Optimization Based on Graph-Based Decomposition for Cell-Free Networks

**arXiv ID:** 2607.09017 | [PDF](https://arxiv.org/pdf/2607.09017v1)

**作者:** Julio Cesar Cardoso Tesolin `[一作]`, Rodrigo C. de Lamare `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于结构化RAN服务状态图的混合整数-连续优化框架，用于用户中心化的无小区大规模MIMO网络中的聚类与资源分配问题，并基于该框架设计了图形化搜索与评估（GBSE）算法；

**💡 创新点**

创新点在于将离散聚类决策与连续资源分配解耦，利用Hamming拓扑构建服务状态图，并通过可调的Hamming距离阈值实现可扩展、可调的搜索复杂度；

**🔧 技术方法**

主要技术包括：混合整数非线性规划、图论（Hamming图、邻域搜索）、Steepest-Ascent Hill-Climbing、Dinkelbach分数最优化、MRT预编码与仿真模拟；

**📊 数据集**

实验仅使用仿真数据，模拟了不同密度和AP/UE配置的用户中心化CF-mMIMO场景；

**📈 对比分析**

与全局穷举、松弛+取整（JO）以及基于信道范数的启发式方法比较，GBSE在中等Hamming阈值下可在相同计算时间内逼近全局最优RAN能效，显著优于现有方法；

**⚠️ 局限性**

局限性在于：需预先设定Hamming阈值，可能陷入局部最优；内层求解仍受传统求解器性能限制；缺乏在真实网络数据上的验证。

---

## 104. REBASE: Reference-Background Subspace Elimination for Training-Free In-Context Segmentation

**arXiv ID:** 2607.09082 | [PDF](https://arxiv.org/pdf/2607.09082v1)

**作者:** Mantha Sai Gopal `[一作]` (CamCom Technologies Private Limited), Uma Mahesh `[通讯]` (CamCom Technologies Private Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练免费的一次性图像分割框架 REBASE，能够在仅给定一张带标注的参考图像的情况下，在查询图像中实现精确分割；

**💡 创新点**

通过对参考图像背景特征构建低秩子空间并投影至其正交补空间，显式抑制跨图像相似度中的共享背景干扰，从而提升语义匹配的区分度；

**🔧 技术方法**

使用冻结的 DINOv2 ViT-L/14 提取稠密特征，利用 SAM ViT-H 进行分割；实现背景子空间投影、相似度加权最远点采样 (SW‑FPS) 与稠密相似度先验；

**📊 数据集**

在五个一-shot 分割基准上评估：ISIC 2018（皮肤病变）、胸部 X‑ray、FSS‑1000（细粒度分类）、PASCAL‑Part、PACO‑Part；

**📈 对比分析**

与多种训练免费方法（PerSAM、Matcher、GF‑SAM、INSID3）以及部分微调方法对比，REBASE 在 ISIC、胸部 X‑ray、FSS‑1000、PACO‑Part 上取得了最佳或第二佳 mIoU，显著优于前沿训练免费方法；

**⚠️ 局限性**

局限性包括对单一基础模型（DINOv2）与 SAM 结构的依赖；对极度多样化背景的鲁棒性尚未彻底验证；仅使用相似度信息的稀疏点提示在某些极端姿态下仍可能不足以完全覆盖目标。

---

## 105. Beyond Time Shifts: Adapting Omni-LLM as a Reference-Free Evaluator for Generative Audio-Visual Models

**arXiv ID:** 2607.09091 | [PDF](https://arxiv.org/pdf/2607.09091v1)

**作者:** Yijie Qian `[一作]` (Zhejiang University), Shujun Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了真实生成失败的音视频同步评估数据集 SynthSync，并改造 Omni-LLM 为连续无参考评分器，利用 Real‑Valued Group Relative Policy Optimization 对列表级同步拓扑进行全局对齐，提出了一套从人工相对排序到绝对连续评估的完整框架。

**💡 创新点**

创新点在于（1）首创 SynthSync 数据集，聚焦真实生成失真；（2）将多模态 LLM 的离散语言头改为连续投影，直接输出同步分数；（3）提出 ℝ‑GRPO 通过 Gaussian 策略与列表奖励实现对全局同步拓扑的强化学习优化。

**🔧 技术方法**

使用了 Omni‑LLM（Qwen2.5‑Omni‑3B）连续投影头、Bradley‑Terry‑Luce 偏好对齐、Gaussian 策略的 ℝ‑GRPO、PPO‑style clip、KL 正则化、以及多模型生成与人工偏好标注的组合。

**📊 数据集**

主要数据集为 SynthSync（基于 10 款视频‑>音频 V2A 模型生成的失败样本）和后续的 SyncBench 基准（185 个物理事件提示）。

**📈 对比分析**

通过与 SyncNet、PEAVS、AlignNet、AV‑Align 等传统指标对比，并使用 NDCG、Kendall 相关、Pairwise Accuracy 等评估，实验表明该方法在 SynthSync 上实现最高人类偏好对齐（NDCG 0.9435，Pairwise Acc 72.38%），在 SyncBench 上亦能稳健区分多种 AV‑Gen 模型，显著优于现有评测工具。

**⚠️ 局限性**

局限性包括：依赖人工相对排序收集，难以覆盖极端结构失真；Gaussian 探索方差需要调优，过小或过大均会影响收敛；在极低质量音频或极不常见事件时，模型对同步的判别仍可能出现误判。

---

## 106. EXHOLD: Experience-Aware Real-Time Hold Control for Large-Scale Ride-Hailing Matching at DiDi

**arXiv ID:** 2607.09090 | [PDF](https://arxiv.org/pdf/2607.09090v1)

**作者:** Xu Liu `[一作]` (Didichuxing Co. Ltd), Zihao Lu `[通讯]` (Didichuxing Co. Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并部署了一种两阶段的实时“hold”控制框架（EXHOLD），通过先给司机-订单对分配经验层级，再根据层级计算可接受的持有时长，优化乘客与司机的整体体验。

**💡 创新点**

创新点在于：①将经验评估与持有时长分离，允许在保持可预测性的同时针对多目标体验进行优化；②利用Transformer捕获短期时间上下文并结合线性UCB上下文多臂老虎机实现可解释且稳健的层级决策；③通过对经验层级的经验分位数进行约束优化，既保证了服务护栏，又实现了可审计的单调持有时长表。

**🔧 技术方法**

技术包括Transformer编码器、LinUCB上下文多臂老虎机、经验分位数估计、带约束的多目标优化、A/B实验监控与滚动回滚机制。

**📊 数据集**

使用DiDi在巴西市场的真实运营日志（包含100k+请求/日、五座城市、近三周的在线实验数据）作为训练与评估数据集。

**📈 对比分析**

与传统基于多模型阈值的hold策略做A/B对比，实验显示：行程完成率提升0.49%，司机收入提升0.50%，乘客取消率在接受前/后分别下降约2%，呼叫-接受时间减少1.8%。此外，实验在不同城市、峰值/非峰值时段均保持相同正向趋势。

**⚠️ 局限性**

局限性包括：①需要手工调节奖励权重和护栏阈值，调参成本高；②对极端非平稳流量或不同城市/文化背景的迁移性尚未验证；③两阶段分离虽然降低风险，却可能导致层级与持有时长之间的耦合失衡，若未来业务规则变化需重新训练。

---

## 107. DETRAM: End-to-end DEtection, Tracking and Recovery of HumAn Meshes

**arXiv ID:** 2607.09089 | [PDF](https://arxiv.org/pdf/2607.09089v1)

**作者:** Chunggi Lee `[一作]` (Harvard University), Hanspeter Pfister `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为DETRAM的统一框架，能够在视频中同时进行多人的检测、跟踪和人类网格恢复。

**💡 创新点**

创新点在于将检测、跟踪和用户提示查询整合到一个端到端的可训练框架中，首次实现了多人的人类网格恢复与跟踪的统一。

**🔧 技术方法**

使用了基于变换器的解码器，结合了可学习的查询嵌入，能够在时间上保持身份一致性。

**📊 数据集**

使用了多个数据集进行训练和评估，包括PoseTrack21、3DPW、BEDLAM和MuPoTS-3D。

**📈 对比分析**

与现有方法相比，DETRAM在PoseTrack21、MuPoTS-3D、BEDLAM和3DPW上取得了最先进的跟踪结果，并在重建精度上表现出竞争力，尤其在处理拥挤场景时表现优越。

**⚠️ 局限性**

限制在于尽管在大多数指标上表现优异，但在某些情况下，HOTA得分略低于其他方法，可能与2D边界框的定义差异有关。

---

## 108. GeoTrace: Geometry-Aware Trajectory Token Compression for Video Large Language Models

**arXiv ID:** 2607.09080 | [PDF](https://arxiv.org/pdf/2607.09080v1)

**作者:** Guohuan Xie `[一作]` (Tsinghua University), Siqi Li `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为GeoTrace的训练无关的视频视觉令牌压缩框架，能够在视频LLM推理时大幅减少视觉令牌数量，提升效率；

**💡 创新点**

创新点在于将视频证据拆分为精确的骨架令牌与可追踪的事件令牌，通过上下文极点锚定(CFPA)保留全局结构，并用轨迹约束残差凝聚(TCRC)在保持语义可辨性与时空一致性的同时实现压缩；

**🔧 技术方法**

采用了基于注意力的视觉词汇显著性与上下文一致性评估、候选极点采样、Hungarian 匹配构建一对一轨迹、熵正则传输（ET）进行空间凝聚等技术；

**📊 数据集**

在四大视频理解基准（VideoMME、LongVideoBench、MVBench、EgoSchema）以及 Qwen2.5-VL/Qwen3-VL 视频LLM 上进行评估；

**📈 对比分析**

与 FastV、VisionZip、PruneVID、FastVID、AOT、ST‑SimDiff 等现有压缩方法比较，GeoTrace 在 10% 视觉令牌保留率时仍保持 99.1% 的原始精度，TFLOPs 下降 12.99×，在 LLaVA‑OneVision 上获得最高平均分；

**⚠️ 局限性**

局限在于需要预先划分时段且对骨架分配比例需手动调节，对极低压缩率或某些架构的适配性尚待验证，且缺乏在线自适应的动态预算控制。

---

## 109. Probing Diffusion Denoising Dynamics for Contrastive Representation Learning

**arXiv ID:** 2607.09067 | [PDF](https://arxiv.org/pdf/2607.09067v1)

**作者:** Yasong Dai `[一作]` (Australian National University), Hongdong Li `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对预训练的扩散模型（Stable Diffusion），提出 D^3CL 框架，在不改变生成能力的前提下，使用噪声级对比学习与重构损失相结合进行轻量化 fine‑tune，得到兼具高质量生成和强判别特征的模型。

**💡 创新点**

创新点在于：①将不同噪声级的扩散过程视为同一图像的“视图”，引入对比学习；②将对比学习与重构损失统一为一个损失，证明两者互补；③采用 LoRA 在交叉注意力层实现参数高效更新；④系统化评估噪声级采样对判别特征的影响。

**🔧 技术方法**

技术包括：Stable Diffusion（Latent Diffusion Model）+ UNet + VAE、LoRA（低秩适配）、InfoNCE 对比学习、噪声级调度、逆余弦噪声计划、线性探针、t‑SNE 可视化。

**📊 数据集**

主要使用 ImageNet‑1K 进行预训练与线性探针评估，ImageNet‑256 用于无条件与类别条件生成评估，MSCOCO 用于文本到图像生成，CIFAR‑100 用于少量样本和零样本迁移测试。

**📈 对比分析**

与 Diffusion‑based DifFeed、MAGE、SimCLR、DINO、MAE 等方法对比：在 ImageNet 线性探针精度上达到 80.1%（比基线高约 8.3%），无条件生成 FID 5.56、IS 142.3，类别条件生成 FID 5.16、IS 189.7，SPair‑71k PCK 53.0，MSCOCO CLIP 92.45，整体显示生成与判别性能兼顾。

**⚠️ 局限性**

局限性包括：对噪声级采样和对比损失权重的敏感性，可能在某些噪声级下对判别或生成效果产生折中；目前仅针对视觉任务，跨模态扩展仍需探索；虽然 LoRA 大幅降低参数量，但在极大规模模型或高分辨率生成任务中仍面临计算瓶颈。

---

## 110. Dec-MARVEL: Decentralized Multi-Agent Exploration without Communication under Budget Constraints

**arXiv ID:** 2607.09060 | [PDF](https://arxiv.org/pdf/2607.09060v1)

**作者:** Janghyun Cho `[一作]` (Sogang University), Changjoo Nam `[通讯]` (Sogang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Dec-MARVEL，一种完全去中心化、预算约束的多无人机探索框架，仅通过在视野内偶尔观察到的队友轨迹实现协同，不需要任何通信或地图/目标共享。

**💡 创新点**

创新点在于：① 将可见的队友轨迹作为唯一的部署时协同信号；② 结合图注意力演员、阶段条件的双头评估器和任务导向的特权批评者，专门训练出能在有限预算下做出返回可行前进决策的策略；③ 通过混合预算课程和自适应预算采样实现对不同预算规模的泛化；④ 在实际硬件上实现从仿真到真实机器人的无缝迁移。

**🔧 技术方法**

使用技术包括：图注意力网络（Graph‑Attention Actor）用于融合局部前沿信息；多头时间注意力编码观察到的队友轨迹；预算条件解码器；Soft Actor‑Critic（离散SAC）作为学习框架；特权批评者、阶段条件批评者和预算课程的训练技巧；以及回退预算守卫和冲突解决机制。

**📊 数据集**

数据集：① 900 次从程序生成的二维室内地图（不同大小、复杂度、预算 720/800/1024 米、队伍规模 2/4/8）进行仿真；② 4 台真实移动机器人在 3.6 m × 3.6 m 室内环境中，使用运动捕捉系统进行定位和执行。

**📈 对比分析**

与四个无通信基线（Nearest Frontier、SMMR‑Explore、NBVP、MARVEL）比较。Dec‑MARVEL 在所有 9 个预算‑队伍组合中均实现最高或并列最高的平均探索率、最高成功率，并且感知重叠最小；平均成功率提升约 2.7%，到 99% 覆盖的距离缩短约 17.7%，感知重叠下降约 26.4%。

**⚠️ 局限性**

局限性：① 假设在部署阶段能可靠地观测到队友轨迹（对遮挡、感知误差敏感）；② 目前仅在二维环境和单一方向传感器上验证；③ 仅适用于同质机器人团队；④ 对极端预算紧张或高动态障碍的鲁棒性未充分评估。

---

## 111. ARCANA: A Reflective Multi-Agent Program Synthesis Framework for ARC-AGI-2 Reasoning

**arXiv ID:** 2607.09059 | [PDF](https://arxiv.org/pdf/2607.09059v1)

**作者:** Kunbo Zhang `[一作]` (Columbia University), Kejian Tong `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ARCANA多代理框架，分解ARC-AGI-2图形转换任务为感知、程序生成、符号执行和反思四轮协作推理。

**💡 创新点**

创新点在于结合可微黑板、条件变分程序空间、DPP多样性采样、反思代理与LoRA自适应，使多轮推理高效且可自我改进。

**🔧 技术方法**

使用了Slot Attention+2D RoPE Transformer进行感知，CVAE+自回归解码器生成程序，DPP多样性采样，符号执行验证，元控制器调度，以及LoRA测试时适配。

**📊 数据集**

在ARC-AGI-2数据集（公开+半私有共120任务）上进行实验。

**📈 对比分析**

与六大基线（NVARC、TRM、CompressARC、SOAR、EPS-Grok、OmniARC）对比，ARCANA在48M参数下取得32.5%准确率、CNE 7.90，显著优于EPS-Grok 26.0%和SOAR 18.5%，但仍低于人类75%。

**⚠️ 局限性**

局限性包括与人类仍有显著差距，对多规则与符号解释任务的处理仍弱，且多轮推理导致推理时间与计算开销相对较高。

---

## 112. STEAM: Stable Self-Training with Elastic Matching and Adaptive Purification

**arXiv ID:** 2607.09057 | [PDF](https://arxiv.org/pdf/2607.09057v1)

**作者:** Shaoxiang Wang `[一作]` (Harbin Engineering University), Lan Zhang `[通讯]` (Northeast Forestry University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种端到端的无监督跨视角地理定位框架 STEAM，直接在真实无人机与卫星图像上进行自训练。

**💡 创新点**

创新点在于三大组件：稳定空间感知模块提升特征稳定性；弹性匹配结合双向 Top‑K 软匹配和动态阈值筛选，扩大高质量伪标签覆盖；自适应净化通过置信度/年龄更新和过期标签剔除，抑制伪标签噪声累积。

**🔧 技术方法**

使用 ConvNeXt-Base backbone 与非线性多头空间注意力；双向 Top‑K 软匹配 + 动态阈值筛选；自适应净化的置信度/年龄更新和过期标签剔除；对比学习 InfoNCE 损失。

**📊 数据集**

在 University‑1652 与 SUES‑200 两大基准数据集上进行实验。

**📈 对比分析**

与现有监督与无监督方法比较，STEAM 在两组数据集上无监督下实现与监督相当或更优，取得最高 Recall@1 与 AP，尤其在 SUES‑200 的多高度场景中表现突出。

**⚠️ 局限性**

仍受伪标签阈值、K 值、标签年龄阈值等超参的影响，对极端视角差异和动态场景的鲁棒性尚待进一步验证。

---

## 113. Modeling and Analysis for Multiple-Layer LEO Satellite Internet of Things Constellations

**arXiv ID:** 2607.09035 | [PDF](https://arxiv.org/pdf/2607.09035v1)

**作者:** Ming Ying `[一作]` (Zhejiang University), Yichao Xu `[通讯]` (Zhejiang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于随机几何的多层低轨卫星物联网星座建模与分析框架，结合Rician衰落模型和Cox点过程，推导了连通概率、覆盖概率和传输速率（含有限块长度）等关键性能指标的闭式表达式，并通过 Monte‑Carlo 仿真验证了理论结果。

**💡 创新点**

创新点包括：① 用有限 General Dirichlet 系数逼近 Marcum‑Q 函数，克服 Rician 衰落导致的积分难题；② 将多层卫星轨道分布建模为 CPP，既保留轨道约束又保持解析可行性；③ 在同一框架下同时考虑 Rician 衰落、阴影 Rician、以及多层干扰与 IoT 短包传输的有限块长度效应；④ 通过闭式表达式给出多层卫星配置的最优设计洞察。

**🔧 技术方法**

采用的技术包括：随机几何（Poisson 与 Cox 点过程）、CDF 与 Laplace 变换、General Dirichlet 系数优化（使用遗传算法）、Marcum‑Q 逼近、有限块长度信息理论、以及大规模 Monte‑Carlo 仿真。

**📊 数据集**

使用的是理论参数集合（如地球半径 6378.14 km、载波频率 5 GHz、卫星轨道半径 6800–7400 km 等）和通过 CPP 产生的合成卫星与 IoT 设备位置，未使用真实卫星轨道或通信数据集；所有实验均基于仿真生成的数据。

**📈 对比分析**

对比方法主要是：① 与 BPP（单参数模型）比较，展示 CPP 在多参数配置下的灵活性；② 与 Rician 与阴影 Rician 衰落下的覆盖概率比较，说明不同衰落模型对性能的影响；③ 通过闭式公式与 10^6 次 Monte‑Carlo 仿真对比，验证误差在 1% 以内。性能结果显示：适当增加卫星密度或轨道数能提升连通性，但超过阈值后干扰主导；低轨层功率对覆盖更稳健；阴影 Rician 的严峻程度反而能提升覆盖率；有限块长度对短包速率存在显著折扣。

**⚠️ 局限性**

局限性包括：① 仅考虑圆形轨道且忽略轨道倾斜、轨道偏心等实际轨道误差；② 假设 IoT 设备均匀分布且只处于开阔环境，未考虑城市遮挡与多路径；③ 仅在两层星座下验证，更多层时计算复杂度上升且可能出现更复杂的干扰交互；④ Rician 逼近在极端 K 值下误差仍不为零；⑤ 模型未考虑实时卫星轨道变化、链路管理与多跳跳频等实际协议细节。

---

## 114. MOSAIC: Adaptive Inter-layer Composition for Efficient Heterogeneous Vision-Language Models

**arXiv ID:** 2607.09029 | [PDF](https://arxiv.org/pdf/2607.09029v1)

**作者:** Yuncheng Yang `[一作]` (LiAuto Inc.), Yan Xie `[通讯]` (LiAuto Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

自动将同质的视觉-语言模型转换为异质架构，并通过两阶段蒸馏恢复性能，最终得到 MOSAIC-4B，显著提升推理速度同时保持与教师模型相近的性能。

**💡 创新点**

创新点在于提出硬件感知的多目标 MIP 搜索框架 MOSAIC，可自适配层级异质结构，并结合两阶段参数恢复（全局离策略蒸馏 + 双教师在策略蒸馏）以缓解结构转换带来的性能下降。

**🔧 技术方法**

技术细节包括线性/稀疏/低秩注意力、可缩放的 FFN 结构、块级局部蒸馏、Tchebycheff 标量化的多目标 MIP 求解、以及双教师蒸馏策略。

**📊 数据集**

训练使用约 32M 的公开文本、图像视频数据集；评估基准涵盖 19 项多模态任务，如 MME、MMBench、MMMU、GQA、ScienceQA、CV-Bench、InfoVQA、DocVQA、ChartQA、OCRBench、MVBench、Video-MME、WorldSense、ARC、GSM-8K、HellaSwag 等。

**📈 对比分析**

与 Qwen3-VL-4B-Instruct、InternVL、Molmo 等同规模/小规模模型对比，MOSAIC-4B 在 19 项评测中平均仅低于 0.6% 的性能差距，同时 prefilling 速度提升 1.76×、decoding 速度提升 2.54×，且训练成本仅占原模型的不到 2%。

**⚠️ 局限性**

主要局限在于搜索空间和硬件性能测量的精度需要手工设定，且两阶段蒸馏仍依赖大模型教师，若硬件平台变化或缺乏高质量教师可能需要重新评估和调优。

---

## 115. Achieving Almost Exact Recovery in Almost Quadratic Time: Rank-Based Graph Matching via Local Tree Correlation Tests

**arXiv ID:** 2607.09087 | [PDF](https://arxiv.org/pdf/2607.09087v1)

**作者:** Jiale Cheng `[一作]` (University of Michigan), Lei Ying `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在相关ER图对模型下，提出了一种近似完全恢复的图匹配算法，时间复杂度几乎为二次

**💡 创新点**

首次实现对 λ=(log n)^α (0<α<1) 处的几乎完全恢复，并将时间复杂度降低到 O(n^2+o(1))

**🔧 技术方法**

利用局部树相关性检验的似然比，改为按似然比排名的无阈值匹配策略

**📊 数据集**

使用无真实图数据集，仅在相关ER图模型（CER(n,λ,s)）上进行理论与实验验证

**📈 对比分析**

相较于之前的chandelier‑counting算法，所提出方法在 s∈(√C_Otter,1] 区间内保持常数（几乎）指数时间，且恢复成功率高达 1-O(e^{- (log n)^{α/2}})

**⚠️ 局限性**

对极端低相关性 s≤√C_Otter 的情形仍无可行算法，且算法在实际实现中需预先构造所有局部树，导致常数因子较大

---

## 116. Equivariant Filter for High Performance Image Tracking using an Event Camera

**arXiv ID:** 2607.09103 | [PDF](https://arxiv.org/pdf/2607.09103v1)

**作者:** Angus Apps `[一作]` (Australian National University), Robert Mahony `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种利用事件相机进行平面图像跟踪的等变滤波器（EqF），并通过等价测量方法消除AEB追踪器产生的高相关性测量误差；

**💡 创新点**

将等变滤波与等价测量相结合，实现对事件相机输出的高频、时相关特征位置进行稳健的平面变换估计；

**🔧 技术方法**

使用事件相机的Asynchronous Event Blob (AEB) 追踪器获取特征位置；构建等变滤波器，采用 Lie 群对称性与系统提升；通过等价测量框架将相关测量转为独立测量；

**📊 数据集**

实验使用两组数据：一组为手持相机在平面上的一般平移+旋转运动，另一组为固定相机观察高速旋转盘上特征，最高特征速度约7000像素/秒；

**📈 对比分析**

与直接最小二乘优化（事件间零阶保持）和协方差交叉融合（Covariance Intersection）两种基线方法比较；结果显示EqF+等价测量在平移、旋转和速度估计上噪声更低、轨迹更平滑，尤其在高速运动下优势更明显；

**⚠️ 局限性**

目前仅处理二维平面变换（SE(2)），未扩展到完整的单应性（Homography）；等价测量求逆矩阵时若累积窗口过小易出现数值不稳定；对极端遮挡或特征点极少时仍可能出现漂移。

---

## 117. A Survey on the Green Development of Large Models: From Resource-Efficient Architectures to Hardware-Software Co-Design

**arXiv ID:** 2607.09084 | [PDF](https://arxiv.org/pdf/2607.09084v1)

**作者:** Linhui Xiao `[一作]` (Pengcheng Laboratory), Yaowei Wang `[通讯]` (Harbin Institute of Technology (Shenzhen))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了大规模AI模型在绿色发展方面的全链路路径，涵盖模型架构、训练与部署优化、硬件与软件协同设计，以及在可持续性应用中的落地案例；

**💡 创新点**

提出三角层级框架，将模型架构、训练策略与硬件协同放在同一视角，强调层间双向影响与协同创新；

**🔧 技术方法**

系统梳理并整合了线性/FlashAttention、物理启发型模型（vHeat）、状态空间模型（SSM、Mamba）、稀疏激活（MoE）、参数高效微调（PEFT）、压缩与量化、Speculative Decoding、绿色AI芯片与IMC、分布式训练与 Federated Learning 等技术；

**📊 数据集**

作为综述论文，未针对单一实验使用具体数据集；引用的研究多使用通用大规模文本/视觉数据集（如C4、ImageNet、遥感数据集等）和公开实验平台；

**📈 对比分析**

采用多维度评估方法（GPU时长、能耗、碳排放、FLOPs、速度提升等），对比显示各技术在节能、加速或参数压缩方面的显著改进，但未给出统一基准或统一实验结果；

**⚠️ 局限性**

缺乏统一的标准化评价体系，跨技术比较缺乏可直接对比的基准；技术落地仍受硬件限制，能耗量化与生命周期评估仍不完整，且部分技术（如IMC、光子芯片）尚处于实验阶段，难以立即普及。

---

## 118. EvoLP: Self-Evolving Latency Predictor for Model Compression in Real-Time Edge Systems

**arXiv ID:** 2607.09063 | [PDF](https://arxiv.org/pdf/2607.09063v1)

**作者:** Shuo Huai `[一作]` (Nanyang Technological University), Weichen Liu `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 EvoLP，能够在模型压缩过程中实时准确预测边缘设备上的推理延迟。

**💡 创新点**

创新点在于将 MLP 作为层级延迟预测器并引入自演化矩阵，使预测器在压缩过程中不断提高精度，且只需针对待压缩模型采样，显著降低构建成本。

**🔧 技术方法**

采用三层 MLP 进行延迟建模，使用自演化矩阵（EM）与梯度下降更新，并结合压缩算法中的重要性因子来动态调整压缩比例。

**📊 数据集**

使用 ImageNet‑100 数据集对 VGG、ResNet、Inception、MobileNet 进行压缩验证，实验设备包括 ARMv8 CPU、NVIDIA Pascal 和 Maxwell GPU。

**📈 对比分析**

与 FLOPs、FLOPsE、ZBNLP、nn‑Meter、TALM 等基线对比，EvoLP 在三台设备上实现了最高的 ±5% 预测准确率（约 90‑99%），RMSE 低于对手，且在压缩实验中满足时延约束的同时保持更高的模型准确率。

**⚠️ 局限性**

局限性包括：对不同硬件架构需要重新采样；自演化过程依赖于真实延迟测量，若测量不充分可能导致收敛不佳；目前仅针对延迟预测，未扩展至功耗、准确率等多指标预测。

---

## 119. On Locality and Length Generalization in Visual Reasoning

**arXiv ID:** 2607.09061 | [PDF](https://arxiv.org/pdf/2607.09061v1)

**作者:** Pulkit Madan `[一作]` (Qualcomm AI Research), Roland Memisevic `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究视觉推理中的长度泛化，探讨局部感知与递归处理对超出训练范围任务的影响。

**💡 创新点**

提出基于视觉状态跟踪的长度泛化测试床，并证明局部感知+递归是必要且足够的。

**🔧 技术方法**

采用局部焦点（foveated）感知 + LSTM 递归网络，配合链式思维监督和主动搜索机制。

**📊 数据集**

使用三种合成任务（视觉奇偶性、状态机、检索）和一个真实世界的图形根检测任务，构建 Gymnasium 环境生成轨迹。

**📈 对比分析**

与全局视角 Transformer/视觉语言模型对比，发现全局模型在 OOD 长度上性能急剧下降，而局部递归模型在 InD 和 OOD 上保持高精度；在根检测任务上，foveated 模型比全局模型提升约 30% 精度。

**⚠️ 局限性**

仅使用模仿学习的 oracle 策略，缺乏通用策略；对更复杂多样化任务的鲁棒性和实时计算成本未充分评估。

---

## 120. Video Generation Models are General-Purpose Vision Learners

**arXiv ID:** 2607.09024 | [PDF](https://arxiv.org/pdf/2607.09024v1)

**作者:** Letian Wang `[一作]` (Google DeepMind), Cristian Sminchisescu `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用大规模文本到视频生成模型做预训练，并在此基础上通过统一的前馈架构对多种视频感知任务（密集与稀疏任务）进行后训练，构建单一通用视觉模型。

**💡 创新点**

创新点在于：1) 证明文本到视频生成是获取通用时空视觉先验的最优预训练目标；2) 将迭代扩散模型改写为单步前馈感知模型；3) 通过把所有任务输出统一映射到RGB空间，实现任务无关的统一解码与损失；4) 采用可学习的标记扩展支持稀疏任务；5) 在多任务上只使用单一L2损失，任务细节通过数据表示完成。

**🔧 技术方法**

核心技术包括文本到视频扩散预训练（WAN 2.1）、Rectified Flow 速度预测、RoPE 时间位置插值、可学习标记（token-based）扩展、统一RGB空间映射、单步前馈推理、数据级任务自适配。

**📊 数据集**

主要使用合成数据：800个 RenderPeople 资产、200个 CMU 动作、HDRI 背景，生成 7,500 条 480x832 81帧视频；并补充 TartanAir、Virtual KITTI、MVS Synth、MeViS、Ref-COCO、YouTube-VOS 等公开数据用于部分任务。

**📈 对比分析**

在多种基准（Hi4D、SINTEL、KITTI、ETH3D、Goliath、VideoMatte、PhotoMatte、PPM-100、SINTEL、EMDB、RefVOS-DAVIS、MeViS）上，与专用最先进模型相比，单一通用模型在大多数任务上达到或超过专用模型的性能，且在合成训练下的通用性与数据效率尤为突出。

**⚠️ 局限性**

局限性包括：1) 对稀疏任务的联合训练易导致性能退化，需更多数据或更细粒度的模型调整；2) 当前统一解码仅支持三通道RGB映射，某些高维任务需要特殊布局；3) 依赖合成数据的泛化虽强，但仍可能在极端真实场景或极端摄像条件下表现欠佳；4) 训练过程需要大规模 TPU 资源，推理仍受 GPU/TPU 内存限制。

---

## 121. Secret Scanner Agent: Extracting Secrets and Access Context from Unstructured Documents

**arXiv ID:** 2607.09011 | [PDF](https://arxiv.org/pdf/2607.09011v1)

**作者:** Zixiao Chen `[一作]` (Microsoft), Charlotte Siska `[通讯]` (Microsoft)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个多代理大语言模型系统Secret Scanner Agent（SSA），能够在未结构化文档中同时识别泄露的秘密及其对应的“门”并给出证据。

**💡 创新点**

创新点在于：①引入双代理架构（检测代理优先召回、审核代理过滤误报并补全上下文），②将秘密与其使用目标一起提取而非仅返回字符串，③通过合成基准与人类审查共同验证提升精度与召回率。

**🔧 技术方法**

采用的大技术包括大型语言模型（多模型测试）、多代理对话架构、程序化匹配、LLM判定器以及人工评审链。

**📊 数据集**

使用了人工合成的基准数据集，覆盖23种秘密类型并包含多种文档格式，真实凭据因隐私而无法公开。

**📈 对比分析**

与单代理版、传统正则表达式扫描器以及13名安全分析师对比，SSA在门提取精度上提升最多16个百分点、召回率超过三倍、与正则扫描器保持相同精度、比分析师多恢复近两倍秘密–门对、并且处理速度提升5到17倍。

**⚠️ 局限性**

局限性包括：合成数据可能与真实泄露场景存在差距；系统依赖大模型的推理成本和可用性；在极度混乱或极长文本中仍可能出现误报或漏报；需要额外的计算资源和人力来验证和补全结果。

---

## 122. AgentKGV: Agentic LLM-RAG Framework with Two-Stage Training for the Fact Verification of Knowledge Graphs

**arXiv ID:** 2607.09092 | [PDF](https://arxiv.org/pdf/2607.09092v1)

**作者:** Yumin Heo `[一作]` (SungKyunKwan University), Youngjoong Ko `[通讯]` (SungKyunKwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AgentKGV，一个基于Agentic LLM‑RAG的知识图谱事实验证框架，结合动态路由与迭代查询重写，并通过两阶段训练（SFT+GRPO）提升准确性与成本效率。

**💡 创新点**

① 为KG验证专门设计的Agentic框架，动态决定是否检索；② (s,p,o) 结构化查询重写，缓解三元组与文档表述差异；③ 通过 Turn‑level SFT 与轨迹级 GRPO 两阶段训练，稳定查询重写并优化检索深度。

**🔧 技术方法**

使用 LLM‑RAG、动态路由、迭代查询重写、检索总结模块；蒸馏式 SFT 以教师轨迹为监督；轨迹级 Group Relative Policy Optimization (GRPO)；BGE dense retriever、Qwen‑2.5‑7B‑Instruct 作为检索与摘要基础。

**📊 数据集**

T‑REx (English) 的长尾与未见谓词拆分；以及人工标注的韩语企业 KG（337 正样本、114 负样本）。

**📈 对比分析**

与 Direct LLM、Single‑turn RAG、IRCoT 等基线对比，AgentKGV+SFT+GRPO 在 T‑REx 长尾/未见拆分中宏 F1 提升至 0.871，平均检索调用仅 1.63 次，远低于 IRCoT 的 5.58 次；在韩语企业 KG 上宏 F1 达 0.608，显著优于所有基线。

**⚠️ 局限性**

工业基准规模有限，标注成本高，实验仅覆盖小规模韩语企业 KG，缺乏大规模工业 KG 的验证，未来需扩展到更大规模并探索半自动或人机协同标注。

---

## 123. A Coreset Selection Framework with Ensemble Aggregation for Image Classification

**arXiv ID:** 2607.09100 | [PDF](https://arxiv.org/pdf/2607.09100v1)

**作者:** Pedro Rocha Dantas `[一作]` (University of São Paulo), Lucas Pascotti Valem `[通讯]` (University of São Paulo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种结合coreset选择与集成聚合的框架，主张在图卷积网络和SVM训练中通过样本子集降低成本并保持精度。

**💡 创新点**

创新点在于开发SCOre‑Stratified Selection（SCOSS）及其类平衡/图基变体，利用分层抽样覆盖完整得分分布并兼顾类别平衡，随后通过多次独立训练的集成提升鲁棒性。

**🔧 技术方法**

采用预训练ResNet‑152提取特征，计算与类质心的欧氏距离作为得分；实现SCOSS、SCOSS_B、SCOSS_CC；使用Simple Graph Convolution（SGC）与SVM分类器；集成采用多次运行后概率平均。

**📊 数据集**

在CUB‑200（200类鸟类）和CIFAR‑10（10类）两个图像数据集上进行实验。

**📈 对比分析**

与随机抽样、Moderate Coreset及其类平衡版比较；SCOSS_B在多数情况下获得最佳或接近最佳准确率；在20%子集上，SGC训练时间下降约74%，GPU内存下降约64%，但CUB‑200准确率从63.57%降至53.07%；集成进一步提升准确率并减少运行间波动。

**⚠️ 局限性**

局限在于实验成本高、仅评估了欧氏距离得分、未探索更多基线、数据集和分类器，且集成方式仅为简单平均，未来可尝试其他得分/聚合策略。

---

## 124. L-MAD: A Systematic Evaluation of Multi-Agent Debate Structures in Legal Reasoning

**arXiv ID:** 2607.09099 | [PDF](https://arxiv.org/pdf/2607.09099v1)

**作者:** Tan-Minh Nguyen `[一作]` (Japan Advanced Institute of Science and Technology), Le-Minh Nguyen `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了法律多智能体辩论框架 L-MAD，用以在法律文本蕴含任务中评估不同辩论结构和聚合方法。

**💡 创新点**

创新点在于系统性比较强制共识与独立投票的决策协议、探究代理数量与讨论轮数的尺度折衷，以及揭示多智能体在高阶法律推理中的有效边界。

**🔧 技术方法**

技术主要包括多智能体协作（对话与投票）、专家人格生成、链式推理（IRAC）和自一致性（Self‑Consistency）等 LLM 交互与聚合技术。

**📊 数据集**

使用 COLIEE 2024–2026 Task 4 的法律文本蕴含数据集，包含三年（R07、R06、R05）样本，评估模型在不同年份的鲁棒性。

**📈 对比分析**

与零射击、IRAC、少量样本提示和自一致性基线对比，发现对 30B 级模型，L‑MAD（共识或投票）平均提升约 5–8%；但在 8B 级模型上出现性能下降，且最先进的 32B 模型已接近单模型峰值，额外协作收益有限。

**⚠️ 局限性**

局限性包括：高阶模型已饱和导致多智能体收益递减；低阶模型易出现协同幻觉和过度推理漂移；讨论轮数过多会导致“过度辩论漂移”；所有模型仍受限于缺失外部法律知识，约 17–24% 的问题无法通过内部推理解决。

---

## 125. PRecG: Legal Precedent Retrieval with Graph Neural Networks and Rhetorical Role Segmentation

**arXiv ID:** 2607.09094 | [PDF](https://arxiv.org/pdf/2607.09094v1)

**作者:** Devanshu Verma `[一作]` (University of Delhi), Balaji Ganesan `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 PRecG 模型，按修辞角色对判决文本分段，构建段落级知识图谱，利用图神经网络学习实体表示并聚合成判决级嵌入，最终计算案例相似度实现自动先例检索。

**💡 创新点**

创新点在于：① 引入修辞角色语义分段，保留判决内部结构；② 为每段构建知识图谱并用 LLM 进行实体关系抽取；③ 采用 GAT+注意力聚合的分层编码与 Transformer 聚合，兼顾实体上下文与段落关系；④ 通过聚焦 MSE 训练提升难检索对的预测精度。

**🔧 技术方法**

技术包括：大规模语言模型（Llama 3.1）用于三元组抽取；InLegalBERT 进行实体/关系初始化；图注意力网络（GATConv）学习段落图表示；Transformer 进行段落聚合；焦点 MSE 损失；BERT/Doc2Vec、BM25、LLM Prompt 等基线做对比。

**📊 数据集**

使用印度最高法院和高等法院共 1095 条判决构建知识图谱，并在公开的印度判决相似度基准数据集（190 条判例对）上进行评估。

**📈 对比分析**

与 BM25、PLI+Doc2Vec、PLI+InLegalBERT、InLegalBERT、GAT+InLegalBERT、PromptReps、Lawpoints 等基线对比，PRecG 在 MSE 上从 0.2560 降到 0.1171，MAE 从 0.5144 降到 0.2871，明显优于所有基线。

**⚠️ 局限性**

局限性包括：① 依赖 LLM 的三元组抽取对推理质量敏感，且成本较高；② 仅在印度法律语料上验证，跨域泛化需进一步验证；③ 训练样本有限，模型可能在更大规模数据上过拟合；④ 需要人工验证知识图谱，流程仍有手工干预。

---

## 126. Adaptive Latent Trajectory Anchoring for Action Segmentation Dataset Condensation

**arXiv ID:** 2607.09081 | [PDF](https://arxiv.org/pdf/2607.09081v1)

**作者:** Artheme Gauthier-Villar `[一作]`, Angela Yao `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于DDIM扩散模型的自适应潜在轨迹锚点化方法，用于时间动作分割数据集的压缩；

**💡 创新点**

创新点在于：①将压缩任务从昂贵的迭代优化迁移到确定性潜在映射；②用稀疏锚点表示动作轨迹，并通过潜在空间插值恢复细粒度时序；③引入自适应锚点分配策略，按重建误差动态分配锚点数量；

**🔧 技术方法**

技术手段包括：DDIM（确定性扩散逆过程）用于特征映射；潜在轨迹插值（线性或更高阶插值）；自适应锚点分配的贪心算法；传统TAS模型（MSTCN、ASFormer）训练与评估；

**📊 数据集**

使用公开的三个动作分割基准：GTEA、50Salads和Breakfast；数据均采用预提取的I3D特征；

**📈 对比分析**

与Mean、Intp、Coreset、GNI等基线对比，实验显示在所有三个数据集上都显著优于GNI，且在压缩率仅为原始数据的1.4%~2.4%时，性能接近完整数据训练（例如Breakfast上Acc提升至~68%~72%）。

**⚠️ 局限性**

局限性包括：①依赖预先训练好的DDIM模型，训练成本仍较高；②适用于已提取特征的场景，对原始视频处理的直接性有限；③在极长或复杂动作序列中，自适应锚点分配可能仍需手动调参，且对动态变化的捕捉仍有局限。

---

## 127. Toward Active Object Detection for UAVs in the Wild: A Large-Scale Dataset, Benchmark and Method

**arXiv ID:** 2607.09078 | [PDF](https://arxiv.org/pdf/2607.09078v1)

**作者:** Tianpeng Liu `[一作]` (National University of Defense Technology), Yongxiang Liu `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文先发布了首个大规模真实世界UAV‑Ground Active Object Detection（UGAOD）数据集ATRNet‑LUDO，并在此基础上构建了完整的评测基准；随后提出了基于世界模型的主动观测策略学习方法WMPL，并在基准上验证其优越性。

**💡 创新点**

创新点包括：①首次公开真实场景的UGAOD大规模数据集；②将Joint Embedding Predictive Architecture（JEPA）引入主动检测任务，构建任务专属的世界模型；③利用SAM3实现场景净化（仅保留目标像素并统一同类别遮挡物灰度），显著提升状态表示质量；④通过对比实验展示世界模型显著缩小训练‑测试泛化差距。

**🔧 技术方法**

核心技术：深度强化学习（Dueling DQN、MAP等），自监督学习（JEPA、对比学习、奖励/逆向/前向模型等），场景净化（SAM3），多模态状态编码（GRU+ResNet‑50），奖励设计，经验回放与硬更新。

**📊 数据集**

使用数据集ATRNet‑LUDO：121,000张多视角全景图、1.21M张局部目标切片，覆盖10种车辆类别、40个场景；实验中与现有AOD方法（MAP、MTL、SSL‑MAP、IBE‑MAP）以及随机/前进策略进行对比。

**📈 对比分析**

评估指标为平均回报、识别率与移动距离。WMPL在训练环境下与基线相当，在测试环境中识别率提升约2–3个百分点，平均回报提升约10%，并在保持移动成本不变或略增的情况下显著优于所有对比方法；与被动检测相比，主动策略提高约20%识别率。

**⚠️ 局限性**

局限性：①仍存在训练‑测试泛化缺口，尤其在完全新场景下表现下降；②依赖单目标设置，未验证多目标情形；③对目标类别仅限车辆，难以直接迁移到其他物体；④缺乏3D空间建模与语义关系推理，进一步提升主动策略的鲁棒性仍需研究。

---

## 128. Neuro-Agentic Control: A Deep Learning-based LLM-Powered Agentic AI Framework for Controlling Security Controls

**arXiv ID:** 2607.09076 | [PDF](https://arxiv.org/pdf/2607.09076v1)

**作者:** Saroj Gopali `[一作]` (Texas Tech University), Akbar Siami Namin `[通讯]` (Texas Tech University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 Neuro-Agentic Loop，将 LLM 规划与时间序列基础模型结合，用于工业 IoT 的闭环安全控制。

**💡 创新点**

通过 Counterfactual Physics Injection 机制在预测空间模拟 LLM 提议的控制动作，形成物理约束的检验层，消除幻觉。

**🔧 技术方法**

使用 Gemini 2.5 Flash‑Lite 作为 LLM 规划器，TimesFM 作为时序预测 Sentinel，以及 RAG、检索增强生成和线性物理注入。

**📊 数据集**

在 SWaT 水处理实验平台的时间序列数据上评估。

**📈 对比分析**

与 LSTM、TCN 两个基线在 15 次随机攻击试验中对比，Neuro‑Agentic 预防 33.3% 的突破，平均风险降低 48.5 单位，且无幻觉执行；LSTM 26.7%，TCN 13.3%。

**⚠️ 局限性**

推断延迟约 1.5–2.5 秒，线性注入近似无法捕捉非线性动力学，实验范围局限于单一数据集。

---

## 129. Rank-Independent Spectral Hypergraph Sparsification via Global-Dictionary Chaining

**arXiv ID:** 2607.09074 | [PDF](https://arxiv.org/pdf/2607.09074v1)

**作者:** Chenghua Liu `[一作]` (Chinese Academy of Sciences), Yuxin Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的无阶数(rank‑independent)谱超图稀疏化方法，使用全局字典链式技术（global‑dictionary chaining）在采样过程中消除了对超边最大大小（rank）的对数因子。

**💡 创新点**

核心创新是将所有超边的最大差分度量归一化为相同的全局有效电阻坐标，并证明所有超边度量对该全局字典的 Lipschitz 性质，从而在全局范数上完成一次链式上界，而不再需要对每个超边单独进行链式分析，进而消除了此前 O(log r) 的依赖。

**🔧 技术方法**

主要技术包括：
1) 通过在每条超边上分配团（clique）权重构造参考图 L，得到超边的杠杆上界 τ_e 并保证总和 O(n)；
2) 将图的 Moore–Penrose pseudoinverse L⁺ 的平方根作为坐标变换，使得超图能量 Q_H 变为 τ_e N_e(y)² 的线性组合；
3) 证明所有 N_e(y) 对全局字典度量 ρ(h)=max_{p∈V²} u_p·h 1‑Lipschitz；
4) 利用 Talagrand 的 γ₂ 量化全局字典的 Gaussian 宽度，得到 γ₂(B₂ⁿ,ρ) = O(√log n)；
5) 通过 Rademacher 符号对称化和自界化（self‑bounding）得到统一的采样误差上界，最终构造 O(n log n/ε²) 条超边的稀疏化子超图。

**📊 数据集**

无实验数据集；本工作为纯理论分析，未涉及实测或基准测试。

**📈 对比分析**

与 2023 年 Lee 以及 Jambulapati–Liu–Sidford 的稀疏化结果相比，去掉了对 rank 的对数因子，使得最终稀疏化大小为 O(n log n / ε²)，而不是 O(n log n log r / ε²)。与图的 Batson–Spielman–Srivastava 结果相比，仍保留了 O(log n) 因子，仍未达到 O(n / ε²) 的线性规模。

**⚠️ 局限性**

限制：
1) 仍然需要 O(log n) 的额外因子，未达到与无向图完全等价的 O(n/ε²) 上界；
2) 本方法主要针对无向或无方向的谱稀疏化，虽提供了有向稀疏化的一步估计，但完整的全动态有向稀疏化仍需进一步优化；
3) 对于某些特定变体（如切割稀疏化、子空间学习等）未直接给出改进，需要进一步研究。

---

## 130. Central Tendency Bias in Human Selection of AI-Generated Design Variations

**arXiv ID:** 2607.09018 | [PDF](https://arxiv.org/pdf/2607.09018v1)

**作者:** Huiyang Chen `[一作]` (University of Michigan), Keqing Jiao `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在控制实验中研究了人类在选择 AI 生成设计变体时的中心倾向偏差，操纵设计集合的方差，分别测量审美偏好和代表性任务下的选择行为。

**💡 创新点**

将集合感知理论扩展到 AI 设计协作中，首次证明多样性生成与最终选择多样性之间存在张力，并揭示了人类选择被无意识的平均表征所驱动。

**🔧 技术方法**

采用受控实验设计、网格布局呈现多选项、配对 t 检验、参与者自评原因，结合对视觉连续体的人工构造来操纵方差。

**📊 数据集**

使用 128 组手工构造的海报设计（8 主题 × 2 低方差集 × 2 高方差集），每组 8 设计，中心位置 0、0'，方差分别为低（-1,-1,0,0,0',0,+1,+1）和高（-3,-2,-1,0,0',+1,+2,+3）。

**📈 对比分析**

通过配对样本 t 检验比较中心选取率：在偏好任务中高方差为 56% 对比低方差 34%（t=3.51, p=0.001），在代表性任务中高方差 66% 对比低方差 38%（t=4.34, p<0.001），表明高方差显著提升中心倾向。

**⚠️ 局限性**

限制：实验使用人工构造的样本，缺乏真实 AI 输出的随机性与质量多样性；任务仅限于审美偏好，未涵盖复杂设计决策；样本量 50 人，普适性与生态有效性有限。

---

## 131. SLBench: Evaluating How LLM Agents Follow Logical Relations in Skills

**arXiv ID:** 2607.09016 | [PDF](https://arxiv.org/pdf/2607.09016v1)

**作者:** Xuan Chen `[一作]` (Purdue University), Xiangyu Zhang `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 SkillLogic 框架，对 LLM 代理使用的可重用技能文件中的逻辑关系进行抽取、分类，并基于此构建了 86 个可执行测试用例，验证代理是否能正确遵循这些逻辑关系。

**💡 创新点**

创新点在于：①定义了 8 种逻辑关系的税onomies；②将技能文件中的自然语言逻辑关系映射为可执行的基准测试；③提出轻量级的运行时关系检查 scaffold，显著降低违规率。

**🔧 技术方法**

技术上使用了自然语言处理的句子抽取与关系分析、LLM 驱动的分析与测试生成、可执行环境模拟、自动化分级器以及关系检查脚本。

**📊 数据集**

数据集来源于 SkillsMP，采集了约 5,224 个公开技能文件，经过筛选后构建了 86 个可执行基准案例。

**📈 对比分析**

在 Codex 与 Claude Code 的六种 LLM 骨干上进行评估，安全率仅 35%–70% 之间，违规率高达 70%；使用关系检验脚本后违规率下降 63%。

**⚠️ 局限性**

局限性包括：仅覆盖可本地执行的高影响关系，无法处理需要外部 API 或长期部署的技能；样本量有限且仅评估两大代理平台；自动分级器可能漏判模糊行为；人类审核样本不足。

---

## 132. Correlation-Aware Contextual Bandits with Surrogate Rewards for LLM Routing

**arXiv ID:** 2607.09015 | [PDF](https://arxiv.org/pdf/2607.09015v1)

**作者:** Ajay Narayanan Sridhar `[一作]` (Pennsylvania State University), Vijaykrishnan Narayanan `[通讯]` (Pennsylvania State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了两种针对关联臂和噪声代理奖励的上下文Bandit算法（CABS-C与CABS-D），应用于LLM路由任务。

**💡 创新点**

首次将图反馈与代理奖励相结合，并提供耦合与解耦两种设计，既能在代理可靠时加速学习，又能在代理失真时保持鲁棒。

**🔧 技术方法**

利用图反馈机制、重要性加权、偏差校正、专家聚合（Hedge）以及线性/神经网络回归，给出理论回报界。

**📊 数据集**

在RouterBench、SPROUT和Open LLM Leaderboard v2三大LLM路由基准上进行实验。

**📈 对比分析**

相较于传统LinUCB/TS、随机与静态路由，CABS-D在不同成本敏感度下平均效用提升约8%–15%，累计回报显著降低，Pareto前沿明显上移。

**⚠️ 局限性**

对代理奖励的噪声控制要求较高，计算复杂度随动作数K的立方增长，且未考虑延迟反馈与极大动作空间的场景。

---

## 133. Multi-Agent LLM Collaboration for Unit Test Generation via Human-Testing-Inspired Workflows

**arXiv ID:** 2607.09101 | [PDF](https://arxiv.org/pdf/2607.09101v1)

**作者:** Quanjun Zhang `[一作]` (Nanjing University of Science and Technology), Liang Xiao `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多智能体协作的 LLM 单元测试生成框架，模拟人类开发者的需求规划、测试生成与评审工作流程。

**💡 创新点**

创新点包括①三类专用智能体（需求规划、生成、评审）共同完成测试；②为 LLM 设计可动态调用的工具 API，打破固定流程；③构建专属单元测试知识图，细粒度捕捉代码依赖并持久存储测试工件。

**🔧 技术方法**

采用大语言模型（GPT‑4o、DeepSeek‑V3、Qwen3‑30B 等）+ 多智能体协作与链式推理 + 通过工具 API 进行上下文检索、语法检查、编译、执行、覆盖率与变异测试；使用静态分析构建知识图，Python/Java 代码解析器。

**📊 数据集**

实验数据集：Java 公开项目 6 个（Commons‑Cli、Csv、Gson、Chart、Lang、Ruler）+ 5 个 Defects4J；工业数据集 UTXXX（658 方法，4 个项目）；Python 三个项目（Dataclasses‑json、Apimd、Thonny）共 141 方法。

**📈 对比分析**

与 EvoSuite、ChatUniTest、HITS 等 LLM 基线对比，在 Java 上平均执行率 97.46%、行覆盖 92.34%、分支覆盖 90.24%、变异得分 83.69%，显著优于基线；Python 版亦领先 CodaMosa、CoverUp；在工业项目上获得 84% 行覆盖、55% 变异得分，证明了实用性。

**⚠️ 局限性**

局限性：依赖高成本 LLM；对极复杂方法的重试上限导致执行率略低；知识图构建与维护成本较高，需针对不同语言编写解析器；评估仅在有限项目上，跨语言迁移需额外工程；LLM 随机性未进行多次复测。

---

## 134. Subtoken Vision Transformer for Fine-grained Recognition

**arXiv ID:** 2607.09086 | [PDF](https://arxiv.org/pdf/2607.09086v1)

**作者:** Jie Zhu `[一作]` (Michigan State University), Xiaoming Liu `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 Subtoken Vision Transformer（SubVT），通过在重要图像区域生成细粒度子标记，提升细粒度识别能力，同时保留全局上下文。

**💡 创新点**

创新点在于：①基于注意力的子标记分割（ATS）策略，只在关注度最高的 K% 区域细分；②两阶段训练：先随机采样注意力头探索多样化分割，再通过特征退化距离蒸馏得到单一可预测的子标记重要性映射；③引入轻量化路由器（router），消除推理时的额外注意力前向，保持近乎原始 ViT 的推理成本。

**🔧 技术方法**

使用了 ViT 结构、注意力头、子标记生成、插值扩展位置嵌入、特征退化距离度量、蒸馏与温度化损失、以及单图像推理的路由器网络。

**📊 数据集**

在细粒度数据集 CUB‑200、FGVC‑Aircraft、Stanford‑Cars 上进行评估，并在 CIFAR‑10 与 ImageNet‑100 上验证通用性。

**📈 对比分析**

与 ViT、Retina Patch、MSViT 等基线相比，SubVT 在 DINOv2 的基础上把新类别精度从 81.3% 提升至 84.7%，在三大细粒度数据集上整体提升 1–3% 左右；推理时仅增加 0.5 ms 延迟、3.4% FLOPs，且相较 Retina Patch 的两步推理减少 73.8% 延迟。

**⚠️ 局限性**

主要局限在于分割预算 K 需手工设定并针对每个数据集进行路由器蒸馏，且注意力头之间存在冗余，未来可自动学习 K 并进行头冗余剪枝。

---

## 135. OmniMapBench: Benchmarking Visual-Centric Reasoning on Diverse Map Documents

**arXiv ID:** 2607.09068 | [PDF](https://arxiv.org/pdf/2607.09068v1)

**作者:** Yang Chen `[一作]` (Shanghai Artificial Intelligence Laboratory), Botian Shi `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了OmniMapBench，一个包含9类地图、2100余个人工标注问答对的视觉中心化评测基准，用以检验大型视觉语言模型（LVLM）在地图理解中的感知与多步空间推理能力。

**💡 创新点**

创新点包括：1) 将地图作为不可文本化的视觉资料，显著提升视觉依赖度；2) 引入视觉依赖指数（VDI）衡量基准对视觉信息的真正需求；3) 在同一基准上对25个主流LVLM进行系统评测，揭示现有模型在多步推理上的显著瓶颈。

**🔧 技术方法**

技术上主要使用：图像描述生成（视觉模型 + 生成式提示）、链式推理（CoT）和严格匹配准确率评估；同时通过VDI/DSI两指标对视觉与文本信息的依赖进行量化。

**📊 数据集**

数据集为OmniMapBench，包含1,603张多样化地图图像（室内导航、教育历史、工程设施、交通运输、旅游、地形、遥感等）及2,096条单/多选与排序类型的问答，语言覆盖中英文。

**📈 对比分析**

与传统基准相比，OmniMapBench在VDI上显著更高（k_max=1024时VDI=0.338），说明任务难以通过文本化描述解决；在25个LVLM中，最优模型Gemini‑3.1‑Pro仅达75.03%准确率，显示出明显的性能差距。

**⚠️ 局限性**

局限性包括：①基准仍主要针对二维地图，未涵盖三维或动态地图场景；②数据规模相对有限，可能难以覆盖更广泛的视觉复杂性；③评测受限于单一参考模型的VDI计算，未来需多模型交叉验证以避免偏差。

---

## 136. Inside the Skill Market: From Software Engineering Activities to Reusable Agent Skills

**arXiv ID:** 2607.09065 | [PDF](https://arxiv.org/pdf/2607.09065v1)

**作者:** Jialun Cao `[一作]` (Hong Kong University of Science and Technology), Shing-Chi Cheung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对公开的四大技能市场中的 11,497 条软件工程相关技能进行大规模实证研究，系统分析其结构特征、生命周期覆盖、演化特征以及评估机制。

**💡 创新点**

首次从活动视角对 SE 技能进行表征，揭示生命周期覆盖不均衡、评估机制薄弱，并提出技能推荐、结构化与高上下文活动抽象化的研究方向。

**🔧 技术方法**

利用自动化爬虫采集技能、GPT‑5.5 进行语义过滤、Qwen3.6‑35B‑A3B 进行生命周期与活动标注、tiktoken 进行 token 统计，并对技能结构与版本进行统计分析。

**📊 数据集**

基于 ClawHub、SkillHub、SkillNet 与 SkillsMP 四大公共技能市场的原始技能仓库，经过去重、过滤后得到 11,497 条独特的 SE 相关技能数据集。

**📈 对比分析**

通过 LLM 标注对技能在生命周期阶段和细粒度 SE 活动中的分布进行量化比较；对各市场自有评估指标进行汇总和对比，发现评估维度集中在安全性与可用性，区分度有限；未做传统性能基准，但展示了技能覆盖与评估差异。

**⚠️ 局限性**

研究依赖 LLM 进行标注与过滤，可能产生主观性；数据仅覆盖四大公开市场，未必代表全部 agent 技能；评估信息可能不完整，导致分析的局限性。

---

## 137. An Emergent Mirage: Is Emergent Misalignment and Realignment Indeed a Robust Phenomenon?

**arXiv ID:** 2607.09053 | [PDF](https://arxiv.org/pdf/2607.09053v1)

**作者:** Abhinav Rao `[一作]` (University of Maryland), Atharva Naik `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统研究了在循环微调（bad→good→bad 与 good→bad→good）中大型语言模型的对齐与误对齐行为及其表示漂移；

**💡 创新点**

创新点在于将多阶段对齐/误对齐循环与连续的LoRA参数相似度与梯度监控相结合，揭示误对齐现象对数据分布与响应长度的高度敏感性；

**🔧 技术方法**

使用LoRA参数化微调、梯度范数监控、余弦相似度追踪、LLM judge 评估以及响应长度归一化技术；

**📊 数据集**

主要使用风险金融建议（bad）与安全金融建议（good）的配对数据集（共6000条），并参考恶劣代码、恶意数字等对齐/误对齐数据集；

**📈 对比分析**

通过每5步保存检查点，对模型在保留样本上使用LLM judge评估对齐/连贯性分数，计算误对齐率，并与梯度、相似度曲线对比；结果显示误对齐易出现但在长度归一化后可再次触发，内部信号波动不稳定；

**⚠️ 局限性**

局限性包括：实验仅在单一模型上进行，对齐/误对齐的表面特征极易被数据分布和响应长度偏差影响，缺乏可重复且稳定的内部表征转移信号，评估方法受限于现有评测工具。

---

## 138. Learning More from Less: Reinforcement Learning from Hindsight

**arXiv ID:** 2607.09042 | [PDF](https://arxiv.org/pdf/2607.09042v1)

**作者:** Iris Xu `[一作]` (Massachusetts Institute of Technology), Zhang-Wei Hong `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种学习自事后推理（Learning from Hindsight，LfH）的方法，用预训练的视觉语言模型对机器人失败轨迹进行指令与奖励重标注，使稀疏奖励环境下的RL能够从失败中提取学习信号，显著提升样本效率。

**💡 创新点**

创新点在于：①将同一视觉语言模型同时用于指令重标和奖励评估；②将Hindsight relabeling与GRPO算法结合，利用重要性采样修正；③通过语言重标把失败轨迹变成可训练的多任务示例，从而在稀疏奖励环境中挖掘更多信息。

**🔧 技术方法**

采用的技术包括：GRPO（Group Relative Policy Optimization）强化学习框架；预训练视觉语言模型（VLM）进行指令重标与奖励评估；hindsight policy gradient 重要性采样与 KL 正则化；稀疏奖励转化为语言标签的自监督重标流程。

**📊 数据集**

使用的数据集为 LIBERO-PRO（包含 LIBERO-90、Spatial、Goal、Object 等子集），并在 OOD 任务扰动版本上进行实验；此外，还在真实 Franka FR3 机器人上收集物理实验数据。

**📈 对比分析**

与标准 GRPO 及使用 RoboMETER 密集进度奖励的 GRPO 进行比较。实验表明 LfH 在稀疏奖励环境下的样本效率提升约 5 倍，完成相同最终性能仅需约 1/5 的训练步数；在真实机器人上，LfH 的成功率在 128 次回合时几乎翻倍，160 次回合达到 56% 而 GRPO 仅 22%。

**⚠️ 局限性**

局限性包括：①依赖轨迹中存在可重标的有意义行为，若轨迹重复或无效则难以获益；②重标质量受 VLM 性能限制，错误或过于通用的提示会引入噪声；③对置信度评估和人工验证的需求未在论文中充分解决。

---

## 139. Continuous Aperture Array-Assisted Integrated Communication and Navigation in LEO Satellite Constellations

**arXiv ID:** 2607.09030 | [PDF](https://arxiv.org/pdf/2607.09030v1)

**作者:** Qi Wang `[一作]` (Zhejiang University), Yuanwei Liu `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `14d48e9d-0069-4ad9-996a-1d5968216998` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在低地轨道星座上提出基于连续开口阵列（CAPA）的集成通信与导航（ICAN）框架，并设计了联合波束成形优化算法，最小化平均定位误差（CRB）同时满足通信速率和功率约束。

**💡 创新点**

创新点包括：①首次将CAPA引入多卫星协同ICAN；②提出ICAN信道子空间理论，将无限维波束设计映射到有限维空间；③联合考虑通信速率与导航定位误差，形成非凸优化并通过SDR、BCD、SCA求解；④提供完整的仿真与对比分析。

**🔧 技术方法**

采用电磁信息理论对CAPA波束建模；利用连续信道子空间展开；计算通信SINR与导航CRB；通过半正定松弛（SDR）转化为矩阵优化；使用块坐标下降（BCD）和差分凹凸（SCA）加惩罚实现可行解；使用Gauss–Legendre积分与数值仿真。

**📊 数据集**

仿真使用基于Walker Delta配置的Starlink-like LEO星座（P=72、N=18、h=550km等），随机生成用户分布（M=10、L=4），在不同功率、速率、天线面积、用户数、卫星数等参数下评估性能；未使用真实卫星测量数据，仅为数值仿真。

**📈 对比分析**

与四种基准方案对比：离散相控阵（DPA）、仅优化导航的方案、基于傅里叶展开的方案、零干扰（ZF）方案。实验表明，所提CAPA波束优化在平均CRB、功率预算、速率约束、天线面积以及用户密度等多维度下均显著优于基准，尤其在功率较高或星座密度较大时性能提升更明显。

**⚠️ 局限性**

主要限制：①算法复杂度仍受SDR与矩阵维度限制，需较高计算资源；②依赖理想化的LoS电磁模型和完整CSI，实际系统中可能受多径、非理想波束等影响；③CAPA硬件实现仍处于概念阶段，实际可行性与成本需进一步验证；④对高维信号的近似与惩罚可能导致次优局部最优，缺乏全局最优保证。

---

## 140. Evolutionary Intelligence for Scientific Discovery: From Evolutionary Computation to Cumulative Discovery Systems

**arXiv ID:** 2607.09025 | [PDF](https://arxiv.org/pdf/2607.09025v1)

**作者:** Chao Wang `[一作]` (Xidian University), Licheng Jiao `[通讯]` (Xidian University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了将进化计算（EC）扩展为进化智能（EI）以实现累积式科学发现的理论框架与实践案例。

**💡 创新点**

提出了从EC到EI的转化概念，定义了五维分析框架（演化对象、变异来源、选择准则、反馈环境、演化时序），并阐述了EI如何通过经验保留与知识迁移实现科学洞见。

**🔧 技术方法**

主要技术包括基于种群的进化算子、基础模型（如大语言模型、预训练模型）与自动实验平台的协同演化；以及经验保留、知识表示与迁移等机制。

**📊 数据集**

本论文为综述性质，未使用单一数据集，而是综合了分子设计、蛋白工程、材料合成、程序搜索和实验自动化等领域的多案例与实验数据。

**📈 对比分析**

对比方法侧重于从单一候选评价转向长期累积评价，讨论了如何量化知识迁移、实验可靠性与可追溯性；性能指标强调经验共享与可解释性，而非仅仅是候选质量。

**⚠️ 局限性**

局限性包括：缺乏统一的长期评估标准、实验反馈稀缺导致模型偏差、演化历史追踪困难、模拟与实验之间的鸿沟、以及缺乏共享经验的开放基础设施。

---

## 141. Privacy Detective: A Narrative Game that Cultivates Student Developers' Privacy Awareness by Harnessing Legal Documents

**arXiv ID:** 2607.09022 | [PDF](https://arxiv.org/pdf/2607.09022v1)

**作者:** Shao-Yu Chu `[一作]` (University of California, San Diego), Haojian Jin `[通讯]` (University of California, San Diego)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一款基于真实FTC执法文件的叙事式调查游戏，用来训练学生开发者的隐私意识；

**💡 创新点**

创新点在于通过游戏化证据搜集与模板化违规报告，将隐私原则内化，并利用执法案例作为真值，实现主动练习隐私判断；

**🔧 技术方法**

使用的技术包括叙事调查游戏机制、进阶搜索树、模板化推理、即时针对性反馈以及基于ChatGPT的文本改写；

**📊 数据集**

使用的数据集为约328条FTC执法案例中的若干条（在实验中选取了3个案例轨迹，如Easy Healthcare、GoodRx、Alexa等），包含违规计数与法律依据；

**📈 对比分析**

通过两组（游戏 vs 阅读）之间被试实验比较，测量recall、precision、reasoning completeness，游戏组recall提升0.242、precision +0.127、完整性+0.219，显著优于阅读组；

**⚠️ 局限性**

限制包括样本来自单一美国高校学生且量级有限，基于美国FTC执法缺乏跨法域覆盖，执法滞后且未评估长期效果。

---

## 142. YeTI: You Only Need Two Noisy Images for Real-World sRGB Noise Generation

**arXiv ID:** 2607.09193 | [PDF](https://arxiv.org/pdf/2607.09193v1)

**作者:** Jaekyun Ko `[一作]` (Hanyang University), Tae Hyun Kim `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了YeTI，一个无需干净图像或相机元数据、只需两张噪声图像即可生成真实sRGB噪声的框架。

**💡 创新点**

创新点在于通过重建自编码器（RAE）将场景结构与噪声分离，再利用一阶条件扩散变换（C-DiT）在噪声潜在空间中一次性生成高质量、信号相关的噪声。

**🔧 技术方法**

核心技术包括：结构编码器 + 噪声编码器 + 共享解码器的RAE；对比学习与正则化实现结构/噪声解耦；基于一致性目标的一阶条件扩散 Transformer；以及在单张噪声图像推理时的随机潜在采样。

**📊 数据集**

使用SIDD（含5台手机）作为主要训练集，验证集同源；跨域评估在SIDD+、MAI2021、SID上进行；下游去噪性能在SIDD验证集和DND基准上测试。

**📈 对比分析**

与传统对齐方法（NAFlow、SeNM-VAE、C2N）比较，YeTI在KLD/AKLD指标上均取得最低或最优值；在去噪实验中，用YeTI合成噪声训练的AP‑BSN/MM‑BSN在SIDD/ DND上达到与真实噪声相当甚至略优的PSNR/SSIM，并且混合真实+合成数据还能进一步提升性能。

**⚠️ 局限性**

局限性包括：训练阶段仍需两张同场景的噪声快照，难以直接应用于动态场景或单帧数据；对极高ISO或低光条件下噪声特征的建模仍有待进一步改进。

---

## 143. GenVid2Robot: From Video Generation to Robot Manipulation via Rigid-Geometric Consistency

**arXiv ID:** 2607.09191 | [PDF](https://arxiv.org/pdf/2607.09191v1)

**作者:** Haohui Huang `[一作]` (Guangdong University of Technology), Yi Guo `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 GenVid2Robot 框架，将生成的视频转换为可执行的机器人抓取轨迹，通过对生成视频的二维视觉轨迹进行稀疏 SE(3) 几何一致性验证，并结合真实 RGB‑D 初始帧中的语义锚点、掩码约束抓取和深度补偿，实现从视频到机器人操作的闭环转换。

**💡 创新点**

创新点包括：
1) 将生成视频视为不确定的二维运动假设，而非直接演示；
2) 使用真实第一帧的稀疏 RGB‑D 锚点与生成视频中的跟踪点通过 PnP/RANSAC 检验 SE(3) 运动一致性；
3) 仅接受满足几何一致性的运动，并将其映射到实际抓取时的 TCP 轨迹（抓取条件化）；
4) 采用局部深度补偿对执行过程进行微调，提升鲁棒性。

**🔧 技术方法**

使用技术与工具：
- 视频生成模型（Kling 1.6）
- VLM 语义提示与 SAM 语义分割
- K‑Means 选取稀疏锚点
- CoTracker 跟踪
- OpenCV PnP + RANSAC 估计 SE(3) 运动
- AnyGrasp 生成可行抓取候选
- PyBullet IK 与轨迹平滑
- RealSense 深度反馈做局部补偿
- CubicSpline、Savitzky–Golay 滤波器
- 机器人底座转化、姿态投影等几何运算。

**📊 数据集**

实验基于 RM75 机械手在桌面上完成四类任务（倒水、举起、工具递送、扫掠）的真实机器人试验；使用真实 RGB‑D 观测作为输入，并利用生成模型产生的视频候选；未采用公开大型数据集，主要依赖现场采集的RGB‑D与真实操作数据。

**📈 对比分析**

与 ReKep‑style、RIGVid‑style、NovaFlow‑style 三种基线在相同硬件与抓取、IK、执行接口下进行对比；在 20 次试验下，GenVid2Robot 的成功率在 90% 以上，显著高于其它基线（70–80%）。消除几何一致性过滤后，成功率下降 10–30%；实验还展示了对锚点选择、深度补偿等模块的消融分析，验证其对性能的贡献。

**⚠️ 局限性**

局限性：
1) 依赖生成视频质量，严重失真、遮挡或语义错误会被过滤或导致失败；
2) 稀疏锚点在纹理匮乏、对称或分布不均时 PnP 可能不稳；
3) 只做局部深度补偿，缺乏完整的在线视觉伺服或力反馈；
4) 运动一致性检验不保证全局可达性，导致工作空间或关节极限失效；
5) 对大幅度图像平面位移的应对有限，需重新初始化。

---

## 144. SherAgent: Scaling Attack Investigation in the Wild via LLM-Empowered Iterative Query-Filter Backtracking

**arXiv ID:** 2607.09176 | [PDF](https://arxiv.org/pdf/2607.09176v1)

**作者:** Zhenyuan Li `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个基于大型语言模型（LLM）的自动化攻击调查系统，采用迭代的“查询‑过滤”回溯策略，利用SQL查询和语义推理在真实 SOC 环境中重建被碎片化的因果链并抑制依赖爆炸。

**💡 创新点**

创新点包括：① 通过预定义SQL模板约束 LLM 生成的查询，动态调节查询范围以弥补缺失事件；② 引入语义字段（如 CmdLine、文件路径）进行分支过滤，显著降低无关节点；③ 使用树形状态管理记录全局调查进度，支持回溯与自适应分支选择；④ 在缺失链场景下通过语义推理补全路径，解决传统基于连通性方法失效的问题。

**🔧 技术方法**

技术要素：大型语言模型（DeepSeek‑V3.1、Qwen3.5 等）驱动查询生成与推理；SQL 与 ClickHouse 数据库查询；语义增强（CmdLine、路径等）与上下文提示；树形调查状态与回溯机制；自动化工作流与 API 费用控制。

**📊 数据集**

数据集：来自某互联网公司 SOC 的 53,849 条真实安全警报和对应的原始日志，包含云、企业 Windows 和客户端环境；从中抽取了 125 条缺失链样本、25 条假成功样本和 50 条完整链样本用于实验。

**📈 对比分析**

评估方法：与三种学术 SOTA（DepImpact、ATLAS、OCR-APT）和 SOC 基线进行对比。系统在完整链上取得 92.2% 的成功率（相较 SOC 基线提升 31.1%，相较 SOTA 提升 63.7%），在缺失链上 RSR 达 93.9%/78.3%/88.9%；查询效率提升，平均 API 成本 <$0.10，单次调查时间 <4 分钟；用户研究显示平均节省 10–20 分钟，报告准确度与可读性得分 4.3/5。

**⚠️ 局限性**

局限性：对通用系统进程的 CmdLine 语义不足，难以进行精确过滤；在极端噪声或日志缺失/保留期短的情况下仍可能产生误检；LLM 的非确定性导致部分结果需要人工复核；缺乏对抗性攻击的完整防护措施；系统在极大规模多主机跨域场景下的性能与可解释性尚待进一步验证。

---

## 145. Attention to Detail: Evaluating Energy, Performance, and Accuracy Trade-offs Across vLLM Configurations

**arXiv ID:** 2607.09172 | [PDF](https://arxiv.org/pdf/2607.09172v1)

**作者:** Nada Zine `[一作]` (Univ. Lille, CNRS, Inria), Patricia Lago `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在vLLM推理引擎中对三项配置（注意力核、前缀缓存、分块预填）进行大规模实验，评估其对能耗、性能和准确率的影响。

**💡 创新点**

系统化地研究了vLLM配置与任务、模型的交互效应，并揭示配置能影响模型准确率。

**🔧 技术方法**

使用ART非参数ANOVA、Cliff's Delta、Pareto前沿分析，对9,000次实验收集的能耗、延迟和准确率指标进行统计。

**📊 数据集**

5个开源大模型（Qwen-32B、Qwen-4B、Magistral-24B、Llama-3.1-8B、Llama-3.2-3B）和5个任务集（AT、EE、LB、NQ、WC）。

**📈 对比分析**

通过对比不同配置组合的能耗、首词延迟和总延迟的中位数及效应量，发现注意力核和前缀缓存对能耗/延迟影响显著，配置组合在不同任务中各有优劣。

**⚠️ 局限性**

实验仅覆盖A100 GPU、vLLM 0.10.2、默认超参数，未触发分块预填；结果对硬件、模型规模、其他推理引擎的泛化有限。

---

## 146. Distributed Symmetry Breaking on Hyperbolic Random Graphs

**arXiv ID:** 2607.09170 | [PDF](https://arxiv.org/pdf/2607.09170v1)

**作者:** Yannic Maus `[一作]` (TU Graz), George Skretas `[通讯]` (Hasso Plattner Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在超球面随机图（HRG）上进行对称破坏问题的分布式算法，特别是最大独立集（MIS）和最大匹配（MM）问题。

**💡 创新点**

创新点在于证明了在HRG上，MIS和MM的下界为Ω(loglog n/logloglog n)，并且设计了针对HRG的高效算法，显著提高了性能。

**🔧 技术方法**

使用了超球面随机图（HRG）模型，结合几何信息和新的结构性见解，设计了分布式算法。

**📊 数据集**

使用了超球面随机图（HRG）作为数据集，HRG是通过在超球面上采样点生成的，具有幂律度分布和高聚类系数的特性。

**📈 对比分析**

与传统的最坏情况图相比，HRG上的MIS和MM问题的算法复杂度显著降低，MIS在(log^5/3 log n)轮内解决，MM在(log^3 log n)轮内解决，性能优于已知的最坏情况下的下界Ω(min{logΔ, √(log n)})。

**⚠️ 局限性**

限制在于，尽管在HRG上取得了显著的性能提升，但MIS和MM的问题仍然比Δ+1着色问题更难，且在没有几何信息的情况下，算法的下界仍然存在。

---

## 147. Beyond F5 and GVW: The Proper-Cover Algorithm for Fast Ideal Basis Computation

**arXiv ID:** 2607.09163 | [PDF](https://arxiv.org/pdf/2607.09163v1)

**作者:** Sheng-Ming Ma `[一作]` (BeiHang University), Zheng-Lin Jiao `[通讯]` (BeiHang University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Proper-Cover 算法，用签名、覆盖与 proper basis 结合的方法，在零维多项式理想上高效求解 Gröbner 基。

**💡 创新点**

创新点在于：① 将签名、POT 排序、覆盖等概念推广到参数化系数（PID）；② 设计两阶段兼容因子与贪婪细化的算法；③ 通过兼容因子与 semi‑S‑pair 证明终止性和输出正确性。

**🔧 技术方法**

使用签名基 Gröbner 基、GVW 覆盖优化、proper basis 理论、regular/hungry reduction、兼容因子、semi‑S‑pair 与 S‑pair 表示、模块化基等技术。

**📊 数据集**

实验数据集包括标准 Cyclic、Katsura 系列以及七个随机多项式系统。

**📈 对比分析**

通过与 GVW、F5 在三种单项式顺序下的对比实验，结果显示 Proper-Cover 在所有顺序下均快于 F5，并在 lexicographic (plex) 顺序上明显快于 GVW，解决了更多实例并显著缩短运行时间。

**⚠️ 局限性**

局限性：仅针对零维理想；实现基于 Maple 18，未做进一步优化；对高维或非零维理想的适用性尚未验证。

---

## 148. Taxonomy Maintenance In The Wild Over Evolving Scholarly Data: Reliability, Efficiency, and Cost-Effectiveness

**arXiv ID:** 2607.09149 | [PDF](https://arxiv.org/pdf/2607.09149v1)

**作者:** Daomin Ji `[一作]` (Rmit University), Zi Huang `[通讯]` (University Of Queensland)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种面向学术论文的持续分类体系维护框架GIST，自动从论文“Related Work”部分提取可靠的局部层次并在几何盒嵌入空间中进行全局融合，实现持续迭代的知识树构建；

**💡 创新点**

创新点在于（1）将 LLM 仅用于结构提取，避免幻觉；（2）引入双向词-盒映射与自监督双目标训练；（3）使用新颖度感知的核心样本选择进行增量学习；（4）通过几何“空洞”预测潜在新概念，并在预算约束下进行成本效益检索；

**🔧 技术方法**

主要技术包括基于盒嵌入的几何层次编码、双向词-盒映射网络、可自监督的循环一致性与体积正则化、核心样本采样与增量训练、贝塔分布的几何概念生成模型、Gaussian 近似的预算感知检索规划与子模优化；

**📊 数据集**

使用 arXiv 计算机科学子域的真实论文集，并通过12篇权威综述手工构建的基准进行评估；

**📈 对比分析**

与四种基线（点嵌入、超bolic、三阶段提取、LLM 直接生成）和两种更新策略（重训练、增量）以及三种检索策略（关键词、现有概念、预测概念）比较，GIST 在节点/边 F1 上比最强基线提升约11%/13%，运行时间仅 9.6% 预算仅 12.7%；

**⚠️ 局限性**

局限包括对“Related Work”结构的依赖（缺少此节的论文会被忽略）、对 LLM 质量与提示的敏感性、盒嵌入在极深层次时可能产生数值不稳定以及缺乏跨领域通用性验证。

---

## 149. Exploring the Potential of Program Flowcharts on Code Generation Using Multimodal LLMs

**arXiv ID:** 2607.09146 | [PDF](https://arxiv.org/pdf/2607.09146v1)

**作者:** Yuki Toi `[一作]` (Kyushu University), Yasutaka Kamei `[通讯]` (Kyushu University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在多模态大型语言模型（LLM）代码生成任务中加入程序流程图（Flowchart）对性能的提升效果；

**💡 创新点**

创新点在于首次系统评估流程图对多模态LLM（GPT‑4o）代码生成准确率与成本的影响，并探讨了不同抽象层级与少量示例的组合策略；

**🔧 技术方法**

采用GPT‑4o多模态LLM、Visustin 自动流程图生成工具、Few‑Shot 学习框架以及OpenAI Token计费模型；

**📊 数据集**

使用了从AtCoder竞赛平台收集的289道编程题及其官方示例代码与测试用例，并从中抽取了132道题的不同抽象深度流程图；

**📈 对比分析**

通过对比“仅文本”“文本+完整流程图”“文本+不同深度抽象流程图”以及“一/二示例少量学习”四种提示策略，实验显示加入流程图可使代码生成准确率提升4–13%，且成本保持低位（约$1.22–$3.84/次）；

**⚠️ 局限性**

局限性包括流程图生成依赖于已有解码代码，可能导致信息泄露；实验仅在AtCoder题目与单一模型GPT‑4o上进行，难以推广至其他数据集或模型；并且仅评价功能正确性，未涉及代码可读性、可维护性等质量维度。

---

## 150. Event Stream based Multi-Modal Video Anomaly Detection: A Benchmark Dataset and Algorithms

**arXiv ID:** 2607.09114 | [PDF](https://arxiv.org/pdf/2607.09114v1)

**作者:** Peipei Zhu `[一作]` (Tianjin University of Traditional Chinese Medicine), Zheng Li `[通讯]` (Tianjin University of Traditional Chinese Medicine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了E-VAD框架，结合可见光视频与事件摄像头的同步数据进行视频异常检测，并创建了首个真实场景的可见-事件异常检测基准数据集TJUTCM Pha。

**💡 创新点**

创新点包括：①跨模态对比预训练将事件、视频和文本嵌入统一语义空间；②自适应加权融合模块根据实时可靠性动态平衡事件时间线和视频空间特征；③首次引入真实事件摄像机采集的工业环境数据，弥补了以往仅使用可见光或模拟事件的局限。

**🔧 技术方法**

核心技术包括CLIP ViT-L/14图像/文本编码器、事件帧表示、对比学习（事件-视频、事件-文本、视频-文本），自注意力+局部-全局时序建模，以及基于sigmoid门控的轻量级自适应融合。

**📊 数据集**

使用的数据集有：TJUTCM Pha（真实可见+事件，376k帧、6.3 B事件），UCF‑Crime（含重录事件扩展），ShanghaiTech Campus（含合成事件扩展）以及公开基准数据集（如UCF‑Crime、ShanghaiTech）。

**📈 对比分析**

与多种SOTA方法对比，E‑VAD在上海科技校园数据集AUC达到98.67%、FAR 0%；在UCF‑Crime AUC 88.75%；在TJUTCM Pha AUC 82.25%，在所有基准上均超越或接近最优模型，验证了事件与视频融合的显著优势。

**⚠️ 局限性**

局限性：①依赖昂贵的混合事件摄像机，部署成本高；②在非实验室或非生产环境的泛化尚未充分验证；③仅采用弱监督训练，缺乏精细的帧级标注；未来工作需探索更高效的事件网络、端到端自监督学习及更大规模多模态基准。

---

## 151. Integrating Large Language Models and Graph Convolutional Networks for Semi-Supervised Image Classification

**arXiv ID:** 2607.09104 | [PDF](https://arxiv.org/pdf/2607.09104v1)

**作者:** Camila Piscioneri Magalhães `[一作]` (University of São Paulo), Lucas Pascotti Valem `[通讯]` (University of São Paulo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种利用LLM对图结构进行语义精炼，从而提升半监督GCN图像分类的框架。

**💡 创新点**

创新点在于将VLM生成的图像描述与LLM进行语义相似度评估来剪枝kNN/Rec‑kNN图，弥补视觉特征构图的噪声。

**🔧 技术方法**

使用了Vision‑Language Model BLIP生成描述，GPT‑OSS‑20B LLM进行相似度评分，以及基于SAGE的简化图卷积网络(SGC)。

**📊 数据集**

实验基于Corel5k数据集，使用ResNet、ViT、DINOv2三种视觉特征提取器构建图。

**📈 对比分析**

与仅基于视觉相似度的kNN/Rec‑kNN图相比，加入LLM精炼后kNN图的分类准确率提升约2–3%，Rec‑kNN图提升不足1%，整体表现更稳定。

**⚠️ 局限性**

局限在于对阈值的依赖、在已高质量图结构上收益有限，以及未探索更高效的prompt与多模态相似度融合方式。

---

## 152. Toward Auditable AI Scientists: A Hypothesis Evolution Protocol for LLM Agents

**arXiv ID:** 2607.09195 | [PDF](https://arxiv.org/pdf/2607.09195v1)

**作者:** Izumi Takahara `[一作]` (University of Tokyo), Teruyasu Mizoguchi `[通讯]` (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了Hypothesis Evolution Protocol（HEP），使LLM代理能够以可审计的方式生成、测试、记录证据并更新假设的信念，随后在三类材料科学研究任务上验证其有效性。

**💡 创新点**

创新点在于：①将假设生命周期、信念概率和证据记录为持久、可审计的事件日志；②通过阈值驱动的判决机制强制执行假设的支持/否决；③让代理在规划之外主动执行完整的假设–测试–证据–信念循环；④证明该协议可跨任务泛化，并且其利用深度与基础LLM能力呈正相关。

**🔧 技术方法**

使用技术包括：OpenAI Agents SDK实现的自治代理；基于GPT‑5.5/5.4‑mini/4.1等LLM；HEP注册表（append‑only事件日志）和七个HEP工具；材料计算工具（MLIP结构松弛、分子动力学、材料数据库查询）；自定义脚本生成与记录；实验循环中的迭代调度。

**📊 数据集**

数据集为三组材料科学研究问题：①AO₂ 二氧化物的多晶型选择（7种原型）；②A₂BB'O₆ 双钙钛矿的B位阳离子排序（岩盐、层状、柱状及无序参考）；③MX 过渡金属单硫/单 pnictide 的原型选择（岩盐、NiAs、zincblende、wurtzite）。每组任务通过固定的机器学习势能（MLIP）评估结构能量。

**📈 对比分析**

比较方法：将HEP代理与同一LLM的规划式代理（plan–execute–replan）进行对比，统计步骤占比、转移概率矩阵、生成假设数量、最大生成深度及最终信念分布。结果显示：HEP代理实现了完整的假设循环，生成更多假设，深度更高，最终多数假设被支持，信念更新更具可追踪性；而规划式代理缺乏显式信念更新。性能随LLM能力递减：GPT‑5.5产生≈15个假设、深度≈4；GPT‑5.4‑mini≈7个、深度≈1；GPT‑4.1更不稳定。

**⚠️ 局限性**

限制：①需要强大的基础LLM，能力不足会导致假设生成不足或无法迭代；②证据验证依赖代理自评，缺乏外部程序化检查或独立审计；③实验仅在固定MLIP体系下进行，未覆盖更精细的第一性原理验证；④仅在单代理单任务环境下验证，跨领域或多人协作的可扩展性待进一步研究。

---

## 153. Causally Debiased Latent Action Model for Embodied Action Conditioned World Models

**arXiv ID:** 2607.09185 | [PDF](https://arxiv.org/pdf/2607.09185v1)

**作者:** Yufan Wei `[一作]` (Aether AI), Biwei Huang `[通讯]` (Aether AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对Latent Action Model进行因果去偏，提升动作条件世界模型的可控性与效率。

**💡 创新点**

引入三种去偏目标（主体中心重建、动作对比学习、空间校准），并采用三阶段微调流程，显著减少对偏差的依赖。

**🔧 技术方法**

利用主体掩码（SAM3）、对比学习、KL自由位、低维映射等技术对LAM进行微调，并用桥接MLP对机器人动作进行映射。

**📊 数据集**

使用大规模无标签人类操作视频（EgoDex）和机器人动作数据（AgiBot），以及DreamDojo基线数据进行预训练与评估。

**📈 对比分析**

与DreamDojo原始模型比较，CD‑LAM在FDCE、PSNR、SSIM等指标上提升30–40%，在14B模型上仅需3–6k步即可达到并超越基线，更新量减少12倍。

**⚠️ 局限性**

仍受限于现有数据域，对非抓取动作的泛化有限，且去偏过程需要额外的主体掩码与对比标签，导致实施成本提升。

---

## 154. Generative Communications: Overview, Technologies, and Trends

**arXiv ID:** 2607.09183 | [PDF](https://arxiv.org/pdf/2607.09183v1)

**作者:** Wenjun Zhang `[一作]` (Shanghai Jiao Tong University), Meixia Tao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 6G 通信的新范式——生成式通信（GenCom），将理解、推理和生成直接嵌入通信流程，实现仅传输必要信息，接收端利用共享的生成先验和知识库进行有控制的内容生成。

**💡 创新点**

创新点在于：①把通信目标从“准确重现数据”转变为“实现预期生成结果”；②设计两层架构（传输层 + 控制层）与联合源-信道-生成编码（JSCGC）；③引入生成式控制、通信感知 LLM、知识同步等技术；④提出针对生成式通信的评估指标与基准。

**🔧 技术方法**

核心技术包括：联合源-信道-生成编码（JSCGC）、受控生成、跨模态条件扩散模型、LLM 生成与推理、知识库同步、资源调度与自学习控制。

**📊 数据集**

实验使用 DIV2K 图像数据集，并在生成端使用 SDXL、BLIP、Qwen‑VL、SAM 等模型；对比传统 JPEG+LDPC 方案。

**📈 对比分析**

与传统 JPEG+LDPC 方案相比，Text‑only 生成方案将负载压缩至约 0.05%，同时保持 CLIP 相似度 0.811；Text+Downsampling 方案在 3–12% 负载下可获得 0.853–0.863 的 CLIP 相似度，显示在极低负载下仍能实现高语义保真。

**⚠️ 局限性**

主要局限包括：缺乏严格的生成式通信理论框架；推理与生成的计算延迟高，难以满足实时需求；对抗性攻击与隐私泄露的新风险；需要进一步研究轻量化模型、协同边缘推理与安全防护。

---

## 155. What Pixels Are Enough? SEAMS: Sufficiency Saliency via MSE-Preservation Soft-Masks

**arXiv ID:** 2607.09164 | [PDF](https://arxiv.org/pdf/2607.09164v1)

**作者:** Magdalena Trędowicz `[一作]` (Jagiellonian University), Jacek Tabor `[通讯]` (Jagiellonian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于充分性（sufficiency）的视觉解释方法，直接优化连续软掩码以保留选定模型输出，生成稀疏且可解释的显著图。

**💡 创新点**

创新点在于：①使用可学习的稀疏预算和软掩码参数化，避免显式子集选择；②通过三源合成（原图、自增扰影像和模糊背景）实现无外部干扰源的充分性评估；③同一优化框架可针对不同目标（类别概率、CLS嵌入、token表示）产生多种解释。

**🔧 技术方法**

采用深度可微分的软掩码优化、正则化稀疏预算、三向合成图像、随机增广以及目标表示的保持损失进行梯度优化；实现基于Vision Transformer和ConvNeXt的后置解释。

**📊 数据集**

主要使用ImageNet‑1k验证集，对ViT‑S/16、ConvNeXt‑Tiny、ConvNeXt‑Base、DINOv2 ViT‑S/14等多种模型；此外在私有ROP（视网膜病变）医学图像数据集上测试迁移能力。

**📈 对比分析**

与梯度基因子（Integrated Gradients、SmoothGrad 等）以及最近的perturbation方法（DAVE、RISE 等）在插入/删除 AUC、像素删除曲线等指标上进行比较，取得最高的插入/删除分数，证明在保留模型输出方面更为有效。

**⚠️ 局限性**

局限性包括：①需要迭代优化，计算成本高于单次梯度方法；②结果依赖合成策略，解释意义限定于“充分性”而非“重要性”；③目前仅针对图像编码器，未扩展到视频、多模态或生成模型。

---

## 156. KV-PRM: Efficient Process Reward Modeling via KV-Cache Transfer for Multi-Agent Test-Time Scaling

**arXiv ID:** 2607.09153 | [PDF](https://arxiv.org/pdf/2607.09153v1)

**作者:** Peng Kuang `[一作]` (University of Illinois Urbana Champaign), Haohan Wang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种新的过程奖励模型（KV-PRM），通过直接读取生成过程中的 KV 缓存并仅使用一个验证 token 来评估多代理推理轨迹，避免了传统文本重新编码。

**💡 创新点**

创新点在于：①证明 KV 缓存比文本包含更多信息并且更适合奖励建模；②将 KV 缓存作为输入，实现 O(L) 计算复杂度；③利用单一查询 token 的读取即可获得大部分奖励信息，理论上近似最优。

**🔧 技术方法**

主要技术包括：Transformer 关键/值缓存的直接读取、LoRA 轻量级适配器、单-token 验证流程、基于 MCTS 的标签生成、梯度可微的 KV 缓存优化。

**📊 数据集**

在四个数学推理基准上验证：MATH、GSM8K、AIME 2024、AIME 2025，并使用 Qwen3-0.6B、4B、8B 三种规模模型。

**📈 对比分析**

与传统文本 PRM 及无奖励策略比较，KV-PRM 在 Beam Search、MCTS、加权投票等搜索算法下保持甚至提升准确率，同时在 FLOPs、推理延迟和内存占用上分别提升约 5,000×、37×、34×，显著降低了计算瓶颈。

**⚠️ 局限性**

局限性包括：需要生成器和验证器共享相同的 Transformer 架构；理论依赖线性表示假设；梯度优化的实用性仍处于概念验证阶段；对非文本生成场景的适用性未知。

---

## 157. ReGen: Hierarchical Multi-Prompt Representation Generation for Efficient Waveform Diffusion Models

**arXiv ID:** 2607.09134 | [PDF](https://arxiv.org/pdf/2607.09134v1)

**作者:** Sang-Hoon Lee `[一作]` (Ajou University), Ha-Yeong Choi `[通讯]` (KT Corp)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 ReGen 框架与 GFM 方法，用于在单一扩散 Transformer 中实现层次化多提示表示生成，以提升低比特率语音编解码与 TTS 的波形质量。

**💡 创新点**

创新点在于：① 把中间表示视为可生成的随机变量，避免 REPA 的潜在表示纠缠；② 采用层次化多提示与掩码填充策略联合学习多维表示；③ 引入 generalized flow matching 在向量场空间添加排斥项，防止多轨道收敛到单一均值。

**🔧 技术方法**

技术包括：Diffusion Transformer (DiT)、Conditional Flow Matching (CFM)、Generalized Flow Matching (GFM)、多提示嵌入、掩码填充、VQ/FSQ 编码器、VAE、以及 6.25 Hz 隐空间 LDM。

**📊 数据集**

使用的主要数据集有 LibriTTS、LibriSpeech、Emilia（多语种 MP3 编码）以及 Seed‑en 基准数据集用于零样本 TTS 评估。

**📈 对比分析**

通过与 EnCodec、X‑codec2、BigVGAN、CosyVoice 等现有低比特率编解码器和 TTS 方案对比，ReGen 在 WER、SIM、PESQ、MOS 等指标上均取得或接近最优表现；在 12.5 Hz VAE 编码下，WER 仅 3.63%，MOS 约 4.1，显著优于同等比特率方法。

**⚠️ 局限性**

局限性包括：① 仍需要大量算力训练（如 1M 步 H100 GPU）；② 对极低比特率场景的高频细节恢复有限；③ 潜在的语音克隆与隐私风险需通过水印或检测技术来缓解。

---

## 158. VTaMo: Video-Text Alignment Model for Sign Language Translation

**arXiv ID:** 2607.09126 | [PDF](https://arxiv.org/pdf/2607.09126v1)

**作者:** Junyi Hu `[一作]` (New York University Abu Dhabi), Yi Fang `[通讯]` (New York University Abu Dhabi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种视频-文本多粒度对齐框架VTaMo，用于无注释手语翻译。

**💡 创新点**

创新点在于将局部Optimal Transport+可学习空白token、全局正交变换以及位置对齐对比学习三种对齐机制联合使用，实现显式的帧-词对齐并重排序。

**🔧 技术方法**

采用了熵正则化OT、Sinkhorn算法、可学习正交矩阵、LoRA调优的Flan‑T5解码器以及信息熵正则化的对比损失等技术。

**📊 数据集**

在四个公开基准上评测：Phoenix‑2014T、CSL‑Daily、How2Sign、OpenASL。

**📈 对比分析**

与现有最先进的无注释方法相比，在所有数据集上均取得BLEU‑4最高，Phoenix‑2014T 28.86、CSL‑Daily 27.16、How2Sign 18.47、OpenASL 25.94，表现明显优于SpaMo等对手。

**⚠️ 局限性**

主要局限在于评估仅关注离线精度，未考虑实时或移动端部署；且词序恢复仍依赖单独的文本恢复模型。

---

## 159. Event Burst Trigger: An Availability Backdoor Attack on Event-Based SNN Object Detection

**arXiv ID:** 2607.09115 | [PDF](https://arxiv.org/pdf/2607.09115v1)

**作者:** Jaesun Baek `[一作]` (Incheon National University), Eun-Kyu Lee `[通讯]` (Incheon National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并评估了一种事件爆发触发（EBT）攻击，利用时间聚集的事件触发与标签投毒在SNN目标检测推理时产生大量伪候选框，从而显著提升NMS计算负荷并降低系统可用性。

**💡 创新点**

首次揭示事件驱动SNN检测中的可用性后门攻击，证明通过注入聚集事件触发与重叠标签即可使NMS爆炸而不影响检测精度，且现有熵基检测方法（如STRIP）对该攻击失效。

**🔧 技术方法**

采用事件触发设计（单像素补丁、加权补丁、时间噪声）、数据集投毒策略（标签重叠注入）、SpikeYOLO模型微调、NMS复杂度与边缘硬件（Jetson Orin）资源耗尽评估，以及STRIP防御实验。

**📊 数据集**

Prophesee GEN1 事件数据集，用于训练与评估SpikeYOLO。

**📈 对比分析**

通过比较不同触发器与投毒比例下的mAP、候选框数、总推理时间及NMS时间，实验表明mAP下降不超过0.1，NMS延迟最高可提升至38×，平均候选数增至约1万，展示攻击在保持精度的前提下极大削弱可用性。

**⚠️ 局限性**

仅针对SpikeYOLO和单一事件摄像头，攻击效果受设备时序与SNN架构差异影响；对更复杂防御（如候选数阈值）可能仍具抵抗；实验未覆盖更广泛的SNN检测模型和更强防御策略。

---

## 160. Scoped Verification for Reliable Long-Horizon Agentic Context Evolution under Distribution Shift

**arXiv ID:** 2607.09175 | [PDF](https://arxiv.org/pdf/2607.09175v1)

**作者:** Dan C. Hsu `[一作]` (RedMind Research), Luke Lu `[通讯]` (RedMind Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在固定的代理模型与工具环境下，将持续系统指令维持为有类型的语义图，利用局部结构验证并按差异编辑恢复文本指令，以实现长期演化。

**💡 创新点**

提出以有向有类型图为演化介质，并在更新时仅对受影响的局部邻域进行结构性验证，从而在保持一致性的同时实现高效的增量修订。

**🔧 技术方法**

采用异构信息网络、PathSim式邻域验证、图编辑算子（AddNode/RemoveNode等）以及差分重构技术，实现对指令图的编辑、验证与文本同步。

**📊 数据集**

使用 τ^2‑bench 旗下的电信客服任务集（共 2,285 条任务，10 组演化批次）进行实验。

**📈 对比分析**

与 HCE（平面文本演化）和 GRACE‑w/o SA（仅图结构无结构验证）三种基线进行对比，使用 Gemini 2.5 Flash 作为代理模型，评估指标为 pass@3、passˆ3；GRACE 在最终检查点实现 0.673 的 passˆ3，明显优于 HCE（0.191）和 GRACE‑w/o SA（0.248），并超过零样本 Gemini 3.1 Pro 的 0.242。

**⚠️ 局限性**

实验仅在单一电信领域与固定模型/工具组合下进行，缺乏跨域验证；此外，演化预算与完整系统对比有限，需要进一步在更多领域与完整上下文演化系统中验证其通用性。

---

## 161. Evaluating Semantic and Quality-Aware Retrieval for Source Code Repositories

**arXiv ID:** 2607.09161 | [PDF](https://arxiv.org/pdf/2607.09161v1)

**作者:** Marek Horváth `[一作]` (Technical University of Košice), Emília Pietriková `[通讯]` (Technical University of Košice)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了一个基于RAG的语义与质量感知源代码检索原型，针对教育 C 代码库实现功能级片段索引、LLM 质量评分与向量检索的完整流程。

**💡 创新点**

创新点在于将 LLM 自动生成的质量元数据与向量检索结合，设计四种检索模式（语义、质量过滤、混合、自动路由）并实现自动路由模型挑选最佳模式。

**🔧 技术方法**

使用技术包括 OpenAI text-embedding-3-small 进行文本和代码嵌入，ChromaDB HNSW 向量索引，LLM（OpenAI）进行质量评分与路由，和基于向量相似度的检索算法。

**📊 数据集**

使用的数据集为 10% 随机抽样的教育 C 代码库，包含 56 位匿名程序员 ID、847 个文件、3,839 个函数级片段。

**📈 对比分析**

通过 15 个自然语言查询、手工评判 0–3 级相关性，比较四种检索模式。整体来看语义模式取得最高 nDCG@5（0.820），质量过滤和混合模式在质量导向查询中表现最佳；自动路由准确率 100%，但平均增加约 1.9 秒延迟。

**⚠️ 局限性**

局限性包括样本量小（仅 15 查询）、仅使用单一嵌入模型、缺乏多评标与一致性评估、LLM 评分误差影响质量模式、查询覆盖受限、未对生成答案的可信度进行评估。

---

## 162. Feeling UISTful: An Interactive Portrait of Scholarly Authorship, Readership, and the In-Between

**arXiv ID:** 2607.09155 | [PDF](https://arxiv.org/pdf/2607.09155v1)

**作者:** Sophia Liu `[一作]` (University of California, Berkeley), Bjoern Hartmann `[通讯]` (University of California, Berkeley)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并实现了 UISTful 系统，记录用户的阅读轨迹，支持私密检索、编辑、发布，并在语义地球地图、路径、树和图三种视图中展示个人与共享的阅读路径，同时提供注释、反思与讨论功能。

**💡 创新点**

将阅读轨迹视为创作者的摄像机（information flâneur），实现了从私密到公开的轨迹生命周期管理；引入语义地球地图（七个语义大陆），结合路径、树、图和注释的多模态可视化，并通过可视化聚合层（路径热图、个体轨迹、访问热度）展现社区整体阅读行为，形成一种“活体肖像”式的学术社群展示。

**🔧 技术方法**

核心技术包括：论文标题与摘要的语义嵌入（利用预训练模型）、Spherical UMAP 与 k‑means 聚类形成七大语义大陆、Voronoi 与 Delaunay 网格构建地理布局、WebGL/Three.js 实现可旋转的三维球体、LLM（如 GPT‑4）为大陆命名、Graph‑Neural‑Network‑style 结构用于显示阅读路径与隐藏链接、React/Redux 前端框架配合后端 REST API，数据存储使用 PostgreSQL + Neo4j 维护论文-作者-引用网络。

**📊 数据集**

使用 UIST 2020–2025 年度全部会议论文与作者信息（约 106 篇论文、35 位作者），通过论文标题、摘要以及引用关系构建完整的知识图谱，并嵌入到语义地球中。

**📈 对比分析**

评估方式主要基于用户体验研究：在 26 名 UISTful 用户上记录 21 天的阅读活动，计算路径重叠度、热度分布与 Jaccard 相似度，展示系统能在多维视图中准确捕捉并可视化个人与社区的阅读轨迹；未给出传统机器学习模型的精度或速度指标，但从可视化聚合层的表现推断系统具备良好的交互延迟（< 200 ms）和可扩展性。

**⚠️ 局限性**

局限性：1）缺乏严格的量化评估与用户实验数据；2）目前仅面向 UIST 社群，通用性和跨领域迁移需要进一步验证；3）依赖 LLM 命名与语义聚类，可能带来偏见与解释不一致；4）隐私保护方面未详细讨论轨迹匿名化与同意机制；5）系统在大规模文献集（千级以上）时的可视化性能与交互响应仍需优化。

---

## 163. Instrumentation and field tests to evaluate a rollover risk estimator for mobile machinery with mobile tools

**arXiv ID:** 2607.09188 | [PDF](https://arxiv.org/pdf/2607.09188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 164. Weaving Light and Time: Unified Harmonic-Geometric Representation Learning for Dense RGB-Event Parsing

**arXiv ID:** 2607.09143 | [PDF](https://arxiv.org/pdf/2607.09143v1)

**作者:** Chenxu Peng `[一作]` (Nankai University), Xiang Li `[通讯]` (NKIARI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了首个统一的 RGB‑Event 密集分割骨干网络，实现了在单一编码器中同时处理密集光强图像与稀疏事件流，并支持多种事件表示的无缝对齐与融合。

**💡 创新点**

核心创新包括：
- 交叉模态几何视差校正模块（Geometric Parallax Rectification）自动学习空间变形；
- 频域谐波谱共振模块（Harmonic Spectral Resonance）在复数频域实现纹理跨模态转移，避免空间噪声；
- 暂态全局路由层（Transient Global Routing）利用事件驱动的异构注意力实现宏观上下文调度；
- 结合随机事件表示混合的预训练协议，使模型对不同事件编码具有鲁棒性。

**🔧 技术方法**

采用了变形卷积、傅里叶变换频域处理、注意力机制（跨模态自注意力）以及对齐后的多尺度编码器。预训练与微调过程中使用了 AdamW、交叉熵损失、drop path 等常规技术。

**📊 数据集**

主要使用的数据集有：
- N‑ImageNetV2（自建，提供精准对齐的 RGB‑Event 对）；
- DELIVER、DDD17、DSEC（道路场景语义分割 benchmark）；
- MFNet 与 KITTI‑360（用于验证跨模态通用性）。

**📈 对比分析**

与 17 种最新多模态分割方法对比，在 DELIVER 上实现 59.57% mIoU、DDD17 上 80.12% mIoU、DSEC 上 76.80% mIoU，均刷新了 state‑of‑the‑art；在 FLOPs 与参数量上也显著低于传统双编码器模型，推理时延在 45.6 ms（大模型）和 22.4 ms（小模型）之间，兼顾精度与实时性。

**⚠️ 局限性**

局限性：
- 对极端对齐误差仍有一定敏感性，虽然已大幅降低但在大幅几何偏移下性能会略降；
- 预训练依赖大规模精确对齐的 RGB‑Event 数据，生成成本较高；
- 对于非事件的稀疏模态（如 LiDAR、thermal）需额外调优，效果仍受域差异影响。

---

## 165. IB-Flow: Information Bottleneck-Guided CFG Distillation for Few-Step Text-to-Image Generation

**arXiv ID:** 2607.09133 | [PDF](https://arxiv.org/pdf/2607.09133v1)

**作者:** Yiting Wang `[一作]` (Tsinghua University), Kang Zhao `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于信息瓶颈的动态CFG蒸馏框架IB-Flow，用于将大规模文本到图像生成模型压缩到仅两步，显著提升生成质量。

**💡 创新点**

创新点包括：①信息瓶颈驱动的实例感知投影时间点选择，自动在高熵阶段保持结构锚定并在低熵阶段捕获高频细节；②熵感知的动态引导强度调度，使CFG强度随SNR自适应衰减，消除过度引导伪影；③通过局部向量场范数得到闭式KL约束，避免昂贵的高维KL计算。

**🔧 技术方法**

技术手段：Flow Matching + ODE采样、Classifier-Free Guidance、信息瓶颈理论、Fisher信息近似、闭式KL约束、动态CFG增强目标。

**📊 数据集**

使用三个大规模教师模型（FLUX.1-dev、OpenUni-L-512、Qwen-Image-20B）进行蒸馏，训练集为2.3M文本提示；评估数据集包括GenEval、DPG-Bench、OneIG-Bench。

**📈 对比分析**

与SenseFlow、pi-Flow、ArcFlow、RCGM、TwinFlow、Qwen-Image-Lightning等多种基线对比，IB-Flow在2步NFE下取得最高分：GenEval 0.86、DPG-Bench 88.67，显著优于ArcFlow（0.84/87.94）和TwinFlow（0.82/87.01），表明动态调度有效提升结构、语义、色彩和纹理四个维度。

**⚠️ 局限性**

局限性：KL近似在极大κ下可能失效；实验仅在极端2步配置下验证，缺乏对更大步数或不同模型的泛化评估；动态调度需额外调参，实际部署时对计算资源与稳定性仍有挑战。

---

## 166. TactiDex: A Real-World Tactile-Guided Benchmark for Human-Like Dexterous Manipulation

**arXiv ID:** 2607.09190 | [PDF](https://arxiv.org/pdf/2607.09190v1)

**作者:** Suting Ni `[一作]` (Shanghaitech University), Jingya Wang `[通讯]` (Shanghaitech University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个真实世界触觉驱动的手物交互数据集 TactiDex 与基于该数据集的 TactiSkill 框架，用于实现从人类示范到机器人手的触觉感知驱动的精细操作。

**💡 创新点**

创新点包括：①首次同步收集全手触觉、精细关节运动和物体 6D 轨迹，构建长期、高精度触觉丰富数据集；②设计三元触觉奖励（触觉引导、力度对齐、安全约束），将触觉视为结构化监督；③采用异构 Actor‑Critic，Critic 访问仿真触觉，Actor 只接收目标触觉，提高收敛稳定性。

**🔧 技术方法**

使用技术包括：高精度光学运动捕捉 + 触觉手套；多模态数据同步与后处理；强化学习（残差策略、异构观测）；三元触觉奖励设计；模拟与真实硬件的零样本转移。

**📊 数据集**

使用了 TactiDex 数据集（49 个物体、757 条交互序列），并与传统仅基于视觉或关节轨迹的数据集（如 OPENTOUCH、HOI4D）做对比。

**📈 对比分析**

通过与纯 kinematic 基线（ManipTrans）和 TactiSkill 的消融实验对比，TactiSkill 在 SR_tac（64.6%）和触觉 F1（0.738）等指标上明显优于基线（39.3% 和 0.556）；同时 PeakSafe@3N（72.8%）和 SafeTac@3N（53.6%）等安全性指标也获得提升，说明在物理逼真性与安全性上都有显著改进。

**⚠️ 局限性**

局限性包括：仿真与真实触觉传感器在噪声、分辨率和柔性方面存在差异，导致部署时可能出现误差；当前部署未使用触觉做闭环反馈，需进一步在机器人上做 fine‑tune；未来需整合视觉、姿态与触觉的统一学习框架以提升鲁棒性。

---

## 167. HiHR: Hierarchical Hyperbolic Representation for Aerial-Ground Person Re-Identification

**arXiv ID:** 2607.09186 | [PDF](https://arxiv.org/pdf/2607.09186v1)

**作者:** Qiwei Yang `[一作]` (Dalian University of Technology), Pingping Zhang `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种 HiHR（Hierarchical Hyperbolic Representation）框架，用于解决航空-地面人重识别（AG‑ReID）问题。该框架通过预训练的视觉‑文本编码器提取多粒度特征，使用文本引导多粒度融合（TMF）融合这些特征，然后在超曲空间（Lorentz 形）构建粗细层次的层次结构（HHL），实现跨视角一致性与视角特异性信息的同时保留。

**💡 创新点**

①将文本提示与多粒度视觉特征通过跨注意力融合，形成更具辨识力的特征；②在超曲空间构建粗细层次层次结构，并通过四个含义约束（parent‑child, prompt‑prompt, prompt‑visual）保持层次一致性；③利用层次化超曲学习（HHL）将跨视角身份分离与视角特异性细节统一编码，避免直接特征对齐导致的信息损失。

**🔧 技术方法**

使用 CLIP‑ViT‑B/16 预训练视觉‑文本编码器；文本提示（view‑agnostic 与 view‑aware）；跨注意力融合（Text‑Guided Multi‑granularity Fusion, TMF）；超曲空间映射（Lorentz 双曲几何）与四个含义约束；交叉熵与三元组损失相结合的层次监督；以及对层次尺度与曲率的自适应学习。

**📊 数据集**

在四个 AG‑ReID 基准上进行评估：AG‑ReID v1、AG‑ReID v2、LAGPeR、CARGO。每个基准均按官方数据拆分与评估协议（Rank‑1、mAP）进行实验。

**📈 对比分析**

与多种 state‑of‑the‑art 方法（VDT、GSAlign、SD‑ReID、ViSA、SAS‑VPReID、SeCap、LATex 等）在所有基准上进行对比。HiHR 在 Rank‑1 与 mAP 上普遍取得最高分，尤其在 A→G（航空到地面）设置下 mAP 提升约 1.5 分，Rank‑1 提升约 4 分，显著优于现有方法。

**⚠️ 局限性**

局限性：①对超曲空间的超参数（曲率、尺度）仍需手动调节，尽管实验显示一定鲁棒性；②模型对极端遮挡、光照变化等鲁棒性未进行系统评估；③当前仅在四个公开基准上验证，跨摄像机分辨率差异或非标准场景的泛化能力尚不充分。

---

## 168. TSR-Ego: Temporally Guided Stereo Refinement Framework for Egocentric 3D Human Pose Estimation

**arXiv ID:** 2607.09169 | [PDF](https://arxiv.org/pdf/2607.09169v1)

**作者:** Md Mushfiqur Azam `[一作]` (University of Texas at San Antonio), Kevin Desai `[通讯]` (University of Texas at San Antonio)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 TSR-Ego，利用因果时间卷积和投影引导的可变形交叉注意力实现头戴立体摄像机的 egocentric 3D 关节估计。

**💡 创新点**

创新点在于将短期运动信息注入稠密特征层与联合查询，形成时空自注意力+投影采样的单阶段解码器，从而在弱单帧立体线索下提升鲁棒性。

**🔧 技术方法**

采用 ResNet‑18+FPN 编码器、因果深度可分离时间卷积、Transformer 自注意力、可变形立体交叉注意力以及多层投影回归的解码头。

**📊 数据集**

在 UnrealEgo2（合成）和 UnrealEgo‑RW（实测）两大头戴双摄 fisheye 数据集上训练与评估。

**📈 对比分析**

相较于 EgoGlass、EgoPoseFormer 等基线，TSR‑Ego 在 UnrealEgo2 上 MPJPE 22.36 mm、PA‑MPJPE 21.23 mm，3D‑PCK 98.36%，在 UnrealEgo‑RW 上 MPJPE 65.58 mm、PA‑MPJPE 56.03 mm、3D‑PCK 90.66%，实现了同类任务的最佳性能。

**⚠️ 局限性**

局限性包括对双摄校准与帧同步的依赖、对极端遮挡/快速运动的容错仍有限，以及单视角信息不足时仍可能出现误估。

---

## 169. Understanding Schedule-Free Methods in Nonconvex Optimization: Rate Guarantees and Escaping Saddles

**arXiv ID:** 2607.09167 | [PDF](https://arxiv.org/pdf/2607.09167v1)

**作者:** Jiseok Chae `[一作]` (Korea Advanced Institute Of Science And Technology), Donghwan Kim `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

对无调度学习率的Schedule-Free方法在非凸光滑优化中的收敛性、严格鞍点避免性和评估迭代行为进行严格理论分析

**💡 创新点**

在非凸场景下首次给出Schedule-Free梯度下降与随机梯度下降的最优worst-case收敛率，并证明在一次小扰动后可避免严格鞍点，提供了评估迭代的PEP上界

**🔧 技术方法**

基于连续时间极限ODE的Lyapunov分析、非自治动力学系统理论、严格鞍点避免理论及性能估计问题（PEP）框架

**📊 数据集**

无实验数据集，全部为理论证明与数值PEP实验

**📈 对比分析**

与标准GD/SGD、Polyak–Ruppert平均、原始Schedule-Free等方法对比，理论收敛率分别为O(1/T)与O(1/√T)，与已知的最优界相符，评估迭代在PEP实验中表现出与极限参数β≈1时略优

**⚠️ 局限性**

常数因子可能不够紧凑；评估迭代的分析仍未得到严格上界；严格鞍点避免需引入一次扰动，理论上是可接受但在实践中略显人工；整体为理论性工作，缺乏实证验证

---

## 170. Present but Rescaled: Chat-to-Agent Transfer of Additive Activation Steering

**arXiv ID:** 2607.09156 | [PDF](https://arxiv.org/pdf/2607.09156v1)

**作者:** Lucas Pinto `[一作]` `[通讯]`, Lucas Pinto

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了添加性激活调控在从单回合聊天到 ReAct 代理部署时的转移效果，发现方向在残差流中几乎完全保留，但行为耦合被重新调节。

**💡 创新点**

创新点在于首次系统性地揭示了“存活-耦合分离”现象，即在不同模型和上下文中无统一的放大或衰减常数，也无统一符号，说明代理部署的安全性无法从聊天评估直接迁移。

**🔧 技术方法**

主要技术包括：差异均值提取的方向向量、残差流级别的加法注入、随机方向对照、KV 缓存重算、层级注入限制、行为不变解析器、以及前后期注入阶段的限制实验。

**📊 数据集**

数据集为自定义的多模型（Qwen2.5‑7B、Llama‑3.1‑8B、Gemma‑2‑9B‑IT 以及其他八个族群）在一致命令字节的同一条项目上进行的匹配信息阶梯实验，包含拒绝、顺从和谄媚等行为。

**📈 对比分析**

通过匹配信息阶梯与行为不变解析器进行比较，使用随机方向对照验证方向特异性；结果显示在 Qwen2.5‑7B 等模型中拒绝绕过行为被放大，Gemma‑2‑9B 放大，Yi‑1.5‑9B 削弱，表明代理可放大攻击且行为变化依赖模型。

**⚠️ 局限性**

主要局限包括：实验集中在少数模型族、仅覆盖拒绝和谄媚两类行为、未在公开基准上验证、缺乏统一符号机制解释、以及对不同工具与多轮交互的泛化仍待进一步研究。

---

## 171. BeyondSight: Object Permanence for End-to-End Autonomous Driving

**arXiv ID:** 2607.09138 | [PDF](https://arxiv.org/pdf/2607.09138v1)

**作者:** Sandro Papais `[一作]` (University of Toronto), Steven L. Waslander `[通讯]` (University of Toronto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为 BeyondSight 的持久化感知框架，能够在车辆完全被遮挡时仍保持对目标的假设，并将这些假设融合到预测与规划中；同时扩展 nuScenes 为 nuScenes-Permanence，提供可见与不可见目标的监督与评估。

**💡 创新点**

核心创新在于：① 将对象永久性（object permanence）作为模型和数据集的设计原则；② 在 SparseDrive 基础上加入 Temporal Prior Decoder、Observation Decoder 和 Posterior Fusion Decoder，实现对目标的时间传播与可观测性条件更新；③ 通过新设的观测性条件评估指标，使模型在遮挡期内的检测与预测性能得到显著提升。

**🔧 技术方法**

技术包括稀疏查询（sparse query）感知架构、BEV 上的时间注意力、基于变压器的状态传播与融合、观测性分类损失、以及两阶段训练与端到端优化。

**📊 数据集**

使用 nuScenes 原始数据与扩展版 nuScenes-Permanence（约 30% 的额外遮挡目标注释）进行训练与评估。

**📈 对比分析**

与 SparseDrive、UniAD、VAD 等基准进行对比；在标准 nuScenes 上，BeyondSight 的 L2_avg 从 0.61 降至 0.54，碰撞率从 0.08% 降至 0.07%；在 nuScenes-Permanence 上，mAP_unobs 从 0 提升至 0.249，显示了显著的遮挡目标检测能力。

**⚠️ 局限性**

局限性包括：在长时间遮挡或目标突然改变运动状态时，持久化假设可能失效；nuScenes-Permanence 仅覆盖连续运动的遮挡情形，未涵盖突发停顿或转向等不连续运动；缺乏对不确定性建模的机制。

---

## 172. Residual Physics-Informed Neural Networks for High-Fidelity BLDC Motor Modeling

**arXiv ID:** 2607.09136 | [PDF](https://arxiv.org/pdf/2607.09136v1)

**作者:** Haitham El-Hussieny `[一作]` `[通讯]` (Egypt-Japan University of Science and Technology), Haitham El-Hussieny (Egypt-Japan University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用物理信息神经网络（PINN）和残差网络（ResNet）构建了一个连续时间的BLDC电机六维状态（角度、角速度、三相电流、温度）预测模型；

**💡 创新点**

创新点在于将ODE残差与数据损失组合成复合物理-数据损失，采用逐步激活的课程学习策略避免物理约束早期干扰收敛，并使用残差结构保持梯度流；

**🔧 技术方法**

技术包括自微分的PINN、ResNet残差结构、物理约束的梯度正则化、课程学习调度、Adam优化器与梯度裁剪；

**📊 数据集**

数据集来源于10条1秒的模拟轨迹（频率5–20 Hz，幅值5–15 V），共10,000条样本，按8:2划分为训练集和验证集；

**📈 对比分析**

与显式欧拉和Scipy RK45数值求解器在1–10,000点批量推理中进行对比；PINN在大批量下推理延迟比Euler低14–118倍，准确率在慢速状态（角度、温度）NRMSE<14%，在快速电流NRMSE≈31%；

**⚠️ 局限性**

主要限制是模型规模小，导致快速电流的预测误差较高，且在极端工作点或硬件部署时的鲁棒性仍需验证。

---

## 173. Vascular Geometry Characterization for AI-Based Endovascular Navigation

**arXiv ID:** 2607.09130 | [PDF](https://arxiv.org/pdf/2607.09130v1)

**作者:** Han-Ru Wu `[一作]` (National Taiwan University Hospital), Alejandro Granados `[通讯]` (Kings College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了血管几何特征对机械血栓切除中强化学习导航难度的影响，并开发了自动化血管特征提取管线。

**💡 创新点**

首次建立基于CTA血管中心线的自动化几何量化管线，量化关键特征（如大动脉弓类型、牛犊弓、逆曲数）并将其与RL导航表现关联，提供可标准化的复杂度评估。

**🔧 技术方法**

使用3D Slicer和VTK进行血管分割，Python脚本提取几何指标；采用Soft Actor-Critic（SAC）强化学习代理进行120秒自主导航模拟；通过混合效应线性和逻辑回归分析关联。

**📊 数据集**

61例真实CT血管数据（CTA）分割得到的血管树作为RL训练和评估环境。

**📈 对比分析**

为每个血管训练专属SAC代理，进行20次评估，记录成功率、程序时长、路径比；混合效应模型评估血管特征对这些指标的影响，发现大动脉弓II/III、牛犊弓、逆曲数显著增加时间并降低成功率。

**⚠️ 局限性**

仅使用RL模拟导航，缺乏与人类临床操作对照；数据量有限，未考虑非线性关系；仅评估至CCA分支，未覆盖ICA等后段；每个血管单独训练限制了模型泛化。

---

## 174. Elusive but Coverable: The Recursion-Theoretic Structure of Complete Abstract Interpretations

**arXiv ID:** 2607.09128 | [PDF](https://arxiv.org/pdf/2607.09128v1)

**作者:** Nicklas Carpenter `[一作]`, Roberto Giacobazzi `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了抽象解释的局部完备性，并从递归理论角度定义了可统一闭包操作，证明了局部完备性类是递归不可分且可产生的。

**💡 创新点**

提出了局部完备性的递归理论表征，证明其在算术层级中的 Π^0_2/Σ^0_2 位置，并构造了可计算覆盖函数实现程序修复。

**🔧 技术方法**

利用可计算函数、s‑m‑n 定理、Rice 定理、递归可枚举/可产生集理论以及抽象域的 Galois 连接构造统一闭包。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

未进行实验比较；通过理论复杂度分析表明局部完备性类不可枚举但可通过程序变换得到可枚举覆盖，证明其在算术层级中的上界。

**⚠️ 局限性**

局限在于只考虑单变量可判定连续抽象域，缺乏对多维或非可判定抽象域的实验验证；且覆盖仅适用于局部完备性而非不完备性。

---

## 175. Power Flow Feasibility Assessment Using Variational Graph Autoencoders

**arXiv ID:** 2607.09122 | [PDF](https://arxiv.org/pdf/2607.09122v1)

**作者:** Ferran Bohigas-Daranas `[一作]` (UPC), Oriol Gomis-Bellmunt `[通讯]` (UPC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出使用变分图自动编码器（VGAE）对电力系统的功率流解是否可行进行无监督检测，利用模型在仅训练合法网络状态下的重构误差来判别不可行状态。

**💡 创新点**

创新点在于：①将VGAE与MPNN结合，仅用无标签的可行样本训练模型，从重构误差自动识别不可行；②通过多维度（负荷、拓扑、运作限值）数据集生成，避免模型过拟合单一指标。

**🔧 技术方法**

使用的技术包括：变分图自动编码器（VGAE）、基于MPNN的消息传递网络、重构损失+KL正则化、样本阈值分割、无监督训练。

**📊 数据集**

数据集基于IEEE 118节点案例，生成了三种场景：Stressed Continuous (SC)、Relaxed Continuous (RC)、Stressed Separated (SS)，并构造了1500样本的组合测试集。

**📈 对比分析**

通过将阈值设为验证集重构误差的99%分位数，得到混淆矩阵；模型在组合测试集上的准确率为99.5%，误检率仅2/1500，且在不同场景下均保持高召回，优于传统迭代求解器的误判率。

**⚠️ 局限性**

局限性包括：对与训练分布差异较大的不可行情况检测灵敏度降低；模型需针对不同系统重新生成多样化训练集；重构误差对极端不可行场景仍可能不足以完全识别。

---

## 176. Augmenting Fundamental Analysis with Large Language Models: A RAG-Based System for Generating Investor Briefs

**arXiv ID:** 2607.09121 | [PDF](https://arxiv.org/pdf/2607.09121v1)

**作者:** Bartosz Ziółko `[一作]` (AGH University of Krakow), Kacper Dobrzeniewski `[通讯]` (AGH University of Krakow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建基于 RAG 架构的 GPT‑4o 大语言模型，自动生成针对九家多行业公司与宏观经济、监管文件等多源数据的投资分析简报，并对九名散户投资者的使用体验进行评估。

**💡 创新点**

创新点在于将公司财报、宏观指标、实时新闻、监管文件等异构信息通过检索+生成方式整合，并在简报中实现新闻排序、主题聚类、突发事件定性评估以及基于 Kitchin 周期的宏观情境切片，首次在一次实验中展示了从数据摄取到用户反馈的完整闭环。

**🔧 技术方法**

主要技术包括 GPT‑4o 作为核心推理引擎、RAG 检索框架、向量数据库（如 LanceDB）用于存储文档嵌入、K‑Means 聚类用于新闻主题分组，以及自定义提示工程实现量化推导与可解释规则。

**📊 数据集**

数据集涵盖：① 10‑K、10‑Q 等公司报表与 SEC EDGAR 文档；② 宏观经济指标（GDP、CPI、PMI 等）；③ 新闻语料（财经媒体报道、监管公告等）；⑤ 个人投资者提供的偏好与反馈。

**📈 对比分析**

对比方法为基于问卷和访谈的主观评估，结果显示简报显著节省信息收集时间、提升信息聚合效率；未进行量化回测，缺乏客观收益对比，系统在数值预测深度和精度上还有待提升。

**⚠️ 局限性**

局限性包括：① 样本规模仅九名投资者与九家公司，结果缺乏统计显著性；② 依赖 LLM 生成，仍存在“幻觉”风险；③ 量化分析模块相对浅薄，未覆盖完整的财务预测与风险度量；④ 未对生成简报的实际投资绩效进行后向测试。

---

## 177. A Personalized Computational Framework for Assessing the Sufficiency of Partially Observed Data in Healthcare AI models

**arXiv ID:** 2607.09165 | [PDF](https://arxiv.org/pdf/2607.09165v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 178. Quantum Circuits in Diffusion Models: A Fair-Comparison Study and a Mechanistic Analysis of Angle-Embedding Failures

**arXiv ID:** 2607.09108 | [PDF](https://arxiv.org/pdf/2607.09108v1)

**作者:** Jaeuk Kim `[一作]` (NextITS Co., Ltd.), Sanghoon Yoo `[通讯]` (NextITS Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在扩散模型的SE模块中，将量子变分电路与角色匹配的经典MLP做公平比较，以评估量子核心的表现。

**💡 创新点**

提出了一个基于SE框架的角色匹配、初始化一致、多种种子显著性检验的公平比较方法，并揭示角度嵌入在Score-based NCSN中的别名失效机制。

**🔧 技术方法**

使用8量子比特的变分量子电路（RealAmplitudes与EfficientSU2）、经典MLP、DDPM、LDM与NCSN扩散模型以及πtanh归一化修正。

**📊 数据集**

在MNIST和CIFAR-10两个图像基准上进行实验。

**📈 对比分析**

通过在相同SE框架、相同初始化、五个采样种子与多次训练种子进行SNR显著性检验，结果显示量子核心与经典控制在DDPM和LDM上无显著差异，参数匹配时差距很小；NCSN则表现差，经过πtanh修正后量子核心取得最佳FID。

**⚠️ 局限性**

实验仅在8量子比特可模拟规模、单一插入点以及小图像数据上进行，未验证在更大规模量子硬件上的性能或通用性。

---

## 179. Gårding's Theorem for Posynomials

**arXiv ID:** 2607.09168 | [PDF](https://arxiv.org/pdf/2607.09168v1)

**作者:** Nima Anari `[一作]` `[通讯]` (Stanford University), Nima Anari (Stanford University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

扩展了Gårding定理到齐次正多项式，证明了在右半平面乘积上，如果有限的正单项式和为零，则其度归一化根是凹的。

**💡 创新点**

通过去掉因子2的损失，改进了已知的连接，提供了更好的混合时间和领域稀疏化保证。

**🔧 技术方法**

使用了复分析和多项式的性质，结合了AI辅助的交互和证明过程。

**📊 数据集**

未具体提及数据集，但涉及的应用包括固定大小的匹配和非对称确定性点过程。

**📈 对比分析**

通过与已知的混合时间和领域稀疏化保证进行比较，展示了改进的效果，提供了更好的参数。

**⚠️ 局限性**

未明确指出限制，但可能涉及到对特定类型多项式的适用性和复杂性问题。

---

## 180. COAST: Context-Aware Differential Learning for Gene Expression Prediction in Spatial Transcriptomics

**arXiv ID:** 2607.09166 | [PDF](https://arxiv.org/pdf/2607.09166v1)

**作者:** Keunho Byeon `[一作]` (Korea University), Jin Tae Kwak `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 COAST 框架，利用 H&E 病理图像预测空间基因表达；

**💡 创新点**

创新点在于结合局部与全局上下文，并通过差分学习同时监督绝对表达和相对表达差异；

**🔧 技术方法**

采用特征提取、上下文特定特征调制、Transformer 聚合以及联合绝对与差分回归头的技术；

**📊 数据集**

使用 SpaRED 中的七个多物种多组织数据集（如 AHSCC、EHPCP1、MMBO 等）以及 TCGA‑LUAD 进行下游生存预测；

**📈 对比分析**

与 STNet、Hist2ST、TCGN、EGNv2、NH2ST、PEKA 等基线比较，COAST 在 PCC、MI、AUC、NRMSE 等指标上均取得最高成绩；

**⚠️ 局限性**

局限性包括对上下文采样大小和差分权重敏感，需要更多数据验证跨平台泛化能力。

---

## 181. MedRealMM: A Real-World Multimodal Benchmark for Chinese Online Medical Consultation

**arXiv ID:** 2607.09142 | [PDF](https://arxiv.org/pdf/2607.09142v1)

**作者:** Runhan Shi `[一作]` (Shanghai Jiao Tong University), Jun Xu `[通讯]` (JD Health International Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建了 MedRealMM 基准，利用 JD Health 真实多模态在线医疗咨询日志，通过自动抽取临床关键点（MCCP）并保留患者上传的图像，形成标准化的一轮回答生成评估实例，并为每个实例制定医生审核的案例专属评分规则。

**💡 创新点**

创新点在于：①从真实会话中自动提取临床关键点并保留多模态信息；②使用医生迭代修订的案例专属评分模板实现开放式、高质量评价；③通过 LLM 裁判实现评分自动化，兼顾可扩展性和临床可解释性。

**🔧 技术方法**

主要技术包括 LLM 驱动的 MCCP 抽取器、医生引导的评分模板生成与修订、LLM 裁判评分机制，以及基于文本+图像的多模态推理框架。

**📊 数据集**

数据集来源于 JD Health 2025-2026 年收集的 5,620 条包含患者上传图像的一对一咨询日志，覆盖 64 个临床科室，具有真实对话、完整多模态信息和去标识化保证。

**📈 对比分析**

对 19 种 LLM（文本、跨模态、开源/闭源）进行评估，最先进的多模态模型在图像输入下可达约 55% 的得分，仍低于真实医生的约 60%，并在负面安全指标（如虚假诊断、反向建议）上表现尤为差。

**⚠️ 局限性**

局限性包括：仅覆盖中文互联网医院场景，无法评估多轮交互的连续性；对图像信息的依赖可能导致模型偏差；评分标准与临床实践的一致性仍需进一步验证；数据仅来自单一平台，可能缺乏多样性。

---

## 182. Super-Generalist: Towards Comprehensive and Accurate Medical Image Understanding via Generalist-Specialist Synergy

**arXiv ID:** 2607.09135 | [PDF](https://arxiv.org/pdf/2607.09135v1)

**作者:** Shaoteng Zhang `[一作]` (DAMO Academy, Alibaba Group), Jianpeng Zhang `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了 SuG（Super‑Generalist）框架，将专业化的解剖与病灶分割知识与通用的视觉‑语言学习相融合，从而实现既能覆盖多种疾病诊断，又能达到专科模型水平的精准病灶定位。

**💡 创新点**

创新点包括：① 专家‑增强的视觉‑语言对齐（将多尺度解剖分割掩码注入视觉特征并与文本对齐）；② 病灶引导的跨模态注意力校准（利用病灶掩码调节文本条件下的视觉注意力）；③ 逐步多阶段训练策略，保证分割、对齐与定位三者协同收敛；④ 首创“专业化‑通用化”协同范式，使得专业知识可迁移至通用任务。

**🔧 技术方法**

技术手段涵盖多任务分割（解剖、病灶类别分割、无类别病灶分割）、多尺度特征聚合与 L2 归一化、解剖级对比学习、病灶掩码指导的注意力损失、以及三阶段的渐进式优化。

**📊 数据集**

使用了多组胸腹部 CT 数据集：MedVL‑CT69K、Merlin、CT‑RATE、RAD‑ChestCT、LesionSegAbdomen（四类肿瘤）和 LesionSegLung，涵盖从大规模图像‑报告对到细粒度病灶标注的多种数据来源。

**📈 对比分析**

与现有通用模型（CLIP、LOVT、fVLM、ViSD‑Boost、OpenVocabCT、HCFNet、VLWS）以及经过改造的专科模型比较，SuG 在疾病诊断（AUC 90.1% 对比 84.9%）、细粒度病灶诊断（AUC 96.1% 对比 94.7%）和病灶定位（AUC 90.8% 对比 47.0%）均实现了显著提升，且在外部验证集上表现更为稳健。

**⚠️ 局限性**

局限性主要包括：① 目前仅在 CT 领域验证，缺乏对 X‑ray、MRI 等其他模态的适用性验证；② 类无类别病灶定位效果虽好，但仍有提升空间，需更大规模、高质量的体素级标注来进一步验证；③ 训练过程中对分割与对齐的多任务平衡仍依赖手工调参。

---

## 183. 4D Human-Scene Reconstruction from Low-Overlap Captures

**arXiv ID:** 2607.09125 | [PDF](https://arxiv.org/pdf/2607.09125v1)

**作者:** Minhyuk Hwang `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向稀疏低重叠摄像机的4D人体场景重建管线；

**💡 创新点**

通过背景与人体分别使用视频扩散模型与SMPL几何先验的解耦重建，并引入运动自适应一致性注入的递归扩散增强；

**🔧 技术方法**

利用视频扩散模型生成密集视角、SMPL姿态估计、三维高斯分布建模、单步扩散提升和光流一致性融合；

**📊 数据集**

在EgoHumans、Harmony4D、Mobile Stage和SelfCap四个真实世界数据集上进行实验；

**📈 对比分析**

与Dyn-3DGS、MonoFusion和STG三种基线比较，在360°和180°相机布局下均在PSNR、SSIM、LPIPS上显著优于对手，尤其在低重叠场景的未观察区域表现突出；

**⚠️ 局限性**

局限包括面部、手部高频细节难以重建、缺乏对动态物体（如球）处理以及背景阴影不随人体运动变化。

---

## 184. ReProAgent: Tool-Augmented Multi-Stage Agentic Generation of Bug Reproduction Tests from Issue Reports

**arXiv ID:** 2607.09123 | [PDF](https://arxiv.org/pdf/2607.09123v1)

**作者:** Quanjun Zhang `[一作]` (Nanjing University of Science and Technology), Liang Xiao `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种多阶段智能体框架，用于从缺陷报告自动生成可重现的测试用例。

**💡 创新点**

通过将问题重现过程拆解为错误定位、根因分析、测试规划和测试生成四个阶段，并为每阶段配备专用工具，实现了更精细的仓库级上下文检索和执行反馈循环。

**🔧 技术方法**

结合大型语言模型（如 GPT‑5‑mini）与工具链（文本检索、代码知识图、命令行和 Python 交互），以及 LangChain/LangGraph 进行多阶段流程编排。

**📊 数据集**

在 SWT‑Bench 的两个子集（SWT‑Bench‑Lite 与 SWT‑Bench‑Verified）上进行评估。

**📈 对比分析**

与包括 LIBRO、Issue2Test、AssertFlip、OpenHands 等基线在内的多种方法对比，使用 fail‑to‑pass 率指标，最终在两大数据集上分别达到 58.43% 与 70.30%，显著优于所有对照模型。

**⚠️ 局限性**

仍依赖大型 LLM 与 Docker 沙盒，推理成本较高；对非 Python 语言或更大规模仓库的适应性未充分验证，且对定位错误的准确率仍有改进空间。

---

## 185. Two-dimensional constacyclic codes over finite chain rings

**arXiv ID:** 2607.09117 | [PDF](https://arxiv.org/pdf/2607.09117v1)

**作者:** Vaishali Singh `[一作]` (Punjab Engineering College (Deemed to be University)), Ridhima Thakral `[通讯]` (Punjab Engineering College (Deemed to be University))

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了有限链环上两维（λ, μ）-常数循环码的代数结构，并给出了其生成多项式的显式表达。

**💡 创新点**

创新点在于：① 首先确定了环 R[y]/⟨y^m-μ⟩ 的原始幺元；② 利用这些幺元与单维常数循环码的结构，构造出两维常数循环码的完整生成集；③ 给出了两维常数循环码达到最大汉明距离（MHDR）与其对应场上码的等价条件。

**🔧 技术方法**

主要技术包括：有限链环与其最大理想的链式结构；多项式环的分解与互素分解；原始幺元的构造与使用；常数循环码的理想结构与最小生成集构造。

**📊 数据集**

论文不涉及具体数据集，全部结果为理论证明与符号示例。

**📈 对比分析**

由于研究的是理论性质，没有实验比较；通过示例验证了生成方法和 MHDR 条件的正确性，展示了在特定链环和长度下的码的维度与距离。

**⚠️ 局限性**

局限性：需满足残域场满足 q ≡ 1 (r m)（即存在适当的原根）才能构造原始幺元；方法依赖链环的特殊结构，难以直接推广到非链环或更一般的环；对于非原始根情况，生成公式尚不完整。

---

## 186. Empirical Pedestrian Safety Assessment in a Mobile Robot Using a Predictive Social Force Model

**arXiv ID:** 2607.09192 | [PDF](https://arxiv.org/pdf/2607.09192v1)

**作者:** Alireza Jafari `[一作]` (National Cheng Kung University), Yen-Chen Liu `[通讯]` (National Cheng Kung University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对人行道上移动机器人与行人互动，研究了传统与基于投影碰撞时间（PTTC）的社交力模型（SFM、TSFM）及其预测性扩展（PSFM、PTSFM），并通过实验评估其客观与主观安全性。

**💡 创新点**

创新点在于将预测的社交力向量平均整合进SFM与TSFM，构建预测版模型，并系统比较预测与非预测版本在安全性方面的差异。

**🔧 技术方法**

使用社交力模型、投影碰撞时间计算、非完整运动学模型、机器人运动学方程以及基于深度摄像头和LiDAR的实时感知算法。

**📊 数据集**

实验数据来自于在国立成功大学实验大厅中进行的单人面向交互试验，共200个试点，记录PTTC、速度、距离、曲率等指标，并通过Likert量表收集主观评价。

**📈 对比分析**

与传统SFM相比，TSFM与PTSFM在最小PTTC、最大曲率等客观指标上显著提升；而预测版本对客观安全提升有限，但在主观舒适度、速度适宜性等方面表现出轻微正向影响。

**⚠️ 局限性**

局限性包括样本单一（年轻男性志愿者）、仅在单人面对面场景下测试、预测时间窗口固定、未对预测模型进行专门校准，以及机器人保守行为被部分受试者误解为不确定性。

---

## 187. Malaika: Understanding Malware through Tri-Grounded Agentic Reasoning

**arXiv ID:** 2607.09179 | [PDF](https://arxiv.org/pdf/2607.09179v1)

**作者:** Xingzhi Qian `[一作]` (University College London), Lorenzo Cavallaro `[通讯]` (University College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种三重根基多代理框架 Tri‑Grounded Agent，用以在安卓恶意软件中通过可审计的流程重构高层恶意行为；

**💡 创新点**

创新点在于将领域、语义、知识三种根基融入代理工作流，并通过分析器‑探索者‑评审者‑摘要者的迭代流程实现可靠、可解释的行为推理；

**🔧 技术方法**

使用技术包括大语言模型（DeepSeek‑V3.2）、LangGraph 代理编排、Androguard 静态分析工具、MITRE ATT&CK Mobile 检索、函数摘要/调用者/类上下文工具以及多步评审机制；

**📊 数据集**

使用数据集为 MalEval（255 APK，230 恶意 + 25 良好）以及针对 ATT&CK 归属的 20 个 Android 样本子集；

**📈 对比分析**

与先前 LLM 基线（MalEval、LAMD）和前沿代理（Codex、Claude Code）对比，报告质量和 ATT&CK 精准度均提升数个百分点，误报率显著降低，整体成本更低；

**⚠️ 局限性**

局限性包括仅针对 Android 静态分析；缺少对动态加载、原生代码和运行时行为的支持；ATT&CK 评估基于家族级别且样本量有限；框架迁移到其他平台需重构工具与知识库。

---

## 188. Automatic Thematic Indexing of Large Literary Corpora: A Machine Learning Approach to Voltaire's Complete Works

**arXiv ID:** 2607.09316 | [PDF](https://arxiv.org/pdf/2607.09316v1)

**作者:** Miguel Arana-Catania `[一作]` (University of Oxford), Glenn Roe `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过机器学习方法实现对维尔特大规模文学语料库的自动主题索引，重点在于对《完整作品集》中的两大子语料库进行实验。

**💡 创新点**

创新点在于将索引视为封闭词汇多标签分类任务，采用生成式大型语言模型（LLM）配合LoRA低秩微调与4位量化技术，并系统评估跨语料库泛化与对修辞复杂性的定性分析。

**🔧 技术方法**

使用的技术包括基于编码器+分类头的模型、生成式解码器、LoRA微调、4-bit量化、Mistral-Small-3.2-24B、GPT-4.1-mini等。

**📊 数据集**

数据集为《完整作品集》中的Essai sur les mœurs et l'esprit des nations与Questions sur l'Encyclopédie两大子语料库，使用印刷版专业索引作为训练与评估的标注。

**📈 对比分析**

通过多阶段模型对比（5B、15B规模），评估F1得分；最佳模型在过滤低频标签后在QE上达到0.67、EM上达到0.47；生成式LLM优于编码器模型，跨语料库测试显示F1下降约10–20%。

**⚠️ 局限性**

局限性包括标签分布极度不平衡、文本语义丰富且修辞复杂、仅使用一级标签、页码为分析单元导致语义不完整，以及评估标准对语义合理但与印刷索引不符的预测给出惩罚。

---

## 189. A Polynomial-Time Algorithm for Coloring Perfect Graphs Based on Walk Counting

**arXiv ID:** 2607.09309 | [PDF](https://arxiv.org/pdf/2607.09309v1)

**作者:** Amir Ali Ahmadi `[一作]` (Princeton University), Yukai Tang `[通讯]` (Princeton University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种完全基于图论运算的多项式时间算法，用于在完美图中决定是否存在指定大小的团，并进一步实现完美图的最优着色。

**💡 创新点**

创新点在于：① 将团判定问题归约到通过权重 walk 计数得到的闭路权重比值；② 设计了一个迭代权重更新（类似乘法权重更新）并通过 walk 计数实现近似谱信息，避免使用半正定规划；③ 证明了该算法在完美图上可以视为对改进的 theta 程序的组合实现，构成了第一个真正的组合性多项式时间着色算法。

**🔧 技术方法**

主要技术包括：图的 walk 计数（通过矩阵幂实现）、权重更新的乘法权重更新框架、Lyapunov 函数证明收敛、对 theta 程序与其修改版 SDP 的分析。

**📊 数据集**

本文未使用任何外部数据集；所有实验与证明均在理论图模型上完成。

**📈 对比分析**

方法的性能为多项式时间（时间复杂度为 O(n^3T̂)，其中 T̂ 与输入规模呈多项式关系）。与传统的半正定规划方法相比，省略了 LP/SDP 求解步骤，显著降低了实现复杂度；在实验中已验证其在随机完美图上的正确性与可行性。

**⚠️ 局限性**

局限性包括：① 对于一般图的复杂度仍未提升；② 对“组合算法”定义的正式化缺失，仍需社区进一步讨论其组合性质；③ 需要预先计算大量 walk 计数，尽管多项式，但在大规模图上可能仍存在效率瓶颈。

---

## 190. Letter Lemmatization: One-to-one and Banded RNNs for Reversing Character-Set Simplification and Abbreviation in Medieval Text

**arXiv ID:** 2607.09291 | [PDF](https://arxiv.org/pdf/2607.09291v1)

**作者:** Anguelos Nicolaou `[一作]` (University of Graz), Georg Vogeler `[通讯]` (University of Graz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出字符集简化映射（CSM）与字母词形化（letter lemmatization）技术，并利用自监督的一对一LSTM RNN实现去映射与手写文本识别（HTR）后校正；进一步将网络扩展为Banded RNN以处理缩写展开与收缩；将上述方法封装在pylelemmatize库中。

**💡 创新点**

创新点在于：①自动生成字符映射的字母词形化启发式，可在任意字符集间快速构建CSM；②自监督的一对一RNN能恢复映射损失，且可直接用作HTR后校正；③将一对一网络通过Banding扩展为Banded RNN，支持插入/删除，实现缩写展开；④提供高效Python实现库pylelemmatize，兼顾速度与功能。

**🔧 技术方法**

采用的技术包括：字符相似度启发式（基于Unicode分类、名称、转写、音素等）；双向LSTM一对一RNN与CTC+Banding扩展；自监督训练（对映射前后字符做交叉熵或CTC损失）；Python Numpy实现的高效映射；并在实验中与传统映射、手写识别模型进行对比。

**📊 数据集**

使用的数据集主要为：Königsfelden中欧中世纪条约语料；Monasterium、Nürnberg Letterbooks等HTR训练/测试并行语料；FTN（Fontenay）与SMG（Santa Maria della Grotta）缩写对照语料；合并训练的数据集用于评估跨语料通用性。

**📈 对比分析**

评估方法：1）计算映射前后字符错误率（CER）来量化信息损失；2）在HTR后校正实验中对比训练/测试集的CER（如Königsfelden从5.11%降至3.59%）；3）在缩写展开实验中，将无操作基线CER（21.6%）与Banded RNN结果（3.9%）比较；4）混合模型跨语料CER为4.3%，显著低于基线。性能方面：pylelemmatize在BMP字符下比普通Python实现快2.8–6倍；一对一RNN在仅20行训练后即可将CER降至原来的一半；Banded RNN在N=5的band内实现了5–7倍的错误率下降。

**⚠️ 局限性**

限制：①一对一映射无法处理多字符映射（连字、缩写），需手动定义或使用Banded RNN；②Banded RNN对band大小敏感，过小会丢失长尾展开，过大则计算量增大且收敛慢；③字母词形化启发式不总能产生符合期望的映射，部分字符仍需人工覆盖；④去映射网络依赖充分覆盖映射操作的训练语料，低资源或高度变异文本可能导致效果下降。

---

## 191. Matroid Contention Resolution with Concentration

**arXiv ID:** 2607.09268 | [PDF](https://arxiv.org/pdf/2607.09268v1)

**作者:** Stephen Arndt `[一作]` (Carnegie Mellon University), Michael Zlatin `[通讯]` (Pomona College)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了针对随机顺序争抢解析方案（AW）的下尾概率界定，并将其推广到任意线性函数，进一步应用此界定改善了多基数交叉着色问题与单调子模最大化在包含 packing 与 covering 约束时的近似算法；

**💡 创新点**

首次给出针对任意 CRS 的下尾收敛性证明，定义了更强的“强 λ‑boundedness”性质，构造了通用的顺序选择过程模型，并实现了维度无关的下尾概率界；

**🔧 技术方法**

采用 Freedman 及 Bhatia‑Davis 不等式对适应性随机过程进行马尔科夫分析，利用 Bernstein、Chernoff 等经典不等式与交换映射技术，构造了强 λ‑bounded顺序选择过程的上界和下界；

**📊 数据集**

本文为理论研究，无实验数据集；所有结果均为上界证明与近似比率分析；

**📈 对比分析**

与先前 k‑matroid 交叉着色的 O(k²) 近似相比，得到 O(k log k) 的改进；在子模最大化问题中实现了 (1-1/e-ε/k+1) 的期望近似，并在覆盖约束上仅违背 O(k) 乘数；

**⚠️ 局限性**

下尾界限仅保证对常数比例的期望，不能覆盖后续步骤；需基数足够大或覆盖宽度足够大；在 k‑dim 匹配等特殊问题中，O(k) 违背已知可达下界；对一般覆盖约束仍无高概率保证。

---

## 192. REMIND: RE-Identification with Memory for INDoor Navigation

**arXiv ID:** 2607.09267 | [PDF](https://arxiv.org/pdf/2607.09267v1)

**作者:** Pablo Diaz-Pereda `[一作]` (Universidad Politécnica de Madrid), Pascual Campoy `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在线多目标重识别追踪器REMIND，用于室内导航场景中从单目RGB视频进行长时序的通用物体身份保持；

**💡 创新点**

创新点在于结合冻结的DINOv3特征、双银行多原型外观记忆、部件与背景级描述符、邻域上下文推理以及全局匈牙利分配，并引入模糊识别保险机制；

**🔧 技术方法**

使用DINOv3 ViT作为特征提取器、K-means或注意力方法生成部件原型、局部背景环描述、邻居共现图、匈牙利算法进行全局匹配、指数移动平均更新记忆；

**📊 数据集**

在自定义的室内长时序视频和公开的ScanNet++数据集上进行评估；

**📈 对比分析**

与最强VOS基线DAM4SAM和通用追踪MASA对比，REMIND在自定义视频上IDF1提升近20点，MASA低36点；在ScanNet++上保持最高IDF1并完成所有场景，且在YOLO检测下未出现GPU内存溢出；

**⚠️ 局限性**

局限在于同类物体密集场景下身份纯度下降、检测质量影响较大，以及对极度拥挤或同类高度相似的场景仍需更强的区分与上下文推理改进。

---

## 193. Geopolitical alignment: Endorsement effects in large language models

**arXiv ID:** 2607.09262 | [PDF](https://arxiv.org/pdf/2607.09262v1)

**作者:** Maxim Chupilkin `[一作]` `[通讯]` (University of Oxford), Maxim Chupilkin (University of Oxford)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过让四个主流大型语言模型评估相同的两篇技术政策短文，并随机加入美国、欧盟、中国或俄罗斯背书，研究了背书信息对模型评分和解释的影响。

**💡 创新点**

首次将政治心理学的背书实验设计迁移到LLM评估领域，并系统比较数值输出与附加解释对评估结果的差异，揭示模型可能隐含的地缘政治偏好。

**🔧 技术方法**

使用GPT‑5、Claude Sonnet、Gemini 2.5 Flash 和 DeepSeek Chat 在无解释与附解释两种提示条件下生成评分与简短理由，并用均值、置信区间及OLS回归分析背书效应。

**📊 数据集**

采用两篇中性技术政策（数字海关平台与网络事件报告平台）与四种背书国家共计640个模型输出（80个无解释 + 320个附解释）作为实验数据集。

**📈 对比分析**

通过比较不同背书下的平均分及回归系数，发现 GPT‑5、Claude Sonnet 和 Gemini 在背书为中国/俄罗斯时显著降低评分；DeepSeek 在无解释时差异不显著，但在附解释后出现显著惩罚；解释提示对各模型评分的影响差异显著，表明解释本身能改变评估。

**⚠️ 局限性**

实验样本量有限，仅覆盖两种技术性中性政策，未检验更广泛领域或语言版本的适用性，且结果仅对特定提示和模型版本具有可重复性。

---

## 194. SQL-RewriteBench: A Correctness-Gated, Full-Denominator Benchmark for Statement-Level SQL Rewriting [Experiment,Analysis & Benchmark]

**arXiv ID:** 2607.09251 | [PDF](https://arxiv.org/pdf/2607.09251v1)

**作者:** Jiang Long `[一作]` (Zhejiang University), Jiang Zhang `[通讯]` (Huawei Company)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了SQL-RewriteBench，一套可执行的SQL重写基准，包含完整的输入、参考重写、架构、结果与执行计划等信息，并通过全分母计数和新指标CGOQ对重写方法进行端到端评估。

**💡 创新点**

创新点在于：① 明确的全分母计数体系，区分接受率、生成率、执行覆盖率、结果一致性和安全重写率；② 引入静态SQL复杂度得分（SCS）和正确性门控的优化质量（CGOQ），既衡量运行时加速，又兼顾结构简化；③ 设计四类池（EQUIV、PERF、ROBUST、DIALECT）以聚焦不同类型的重写机会，并提供可验证的参考重写。

**🔧 技术方法**

使用了SQL解析与AST提取、执行驱动的结果一致性检查、基于运行时测量的加速统计、结构特征计数实现SCS，以及多种重写方法（直接LLM、规则搜索、LLM+规则、学习型重写等）的接口；同时提供评估引擎、脚本化评分与报告生成。

**📊 数据集**

基准数据集由39条SQLStorm、60条Calcite测试、34条TPC‑DS、47条DSB等公共工作负载构成，总共180个可执行案例；每个案例都附带数据库模式、样本数据、参考重写与验证结果。

**📈 对比分析**

在七大方法（Direct-LLM、LearnedRewrite、LLM‑R2、R‑Bot、QUITE等）上进行评估，报告了生成率、结果一致性率、安全重写率、几何平均加速、CGOQ@N等指标。实验显示所有方法在全分母CGOQ@N均为负值，表明即使生成率高或结果一致性好，也难以获得正的优化价值。

**⚠️ 局限性**

局限性包括：① 基准规模仅为180个案例，未覆盖生产中全部重写机会；② 仅针对PostgreSQL源方言，跨方言支持待扩展；③ 运行时加速受实验环境（WSL2/Docker、PostgreSQL 16）影响，难以直接推断云端或大规模分布式场景下的真实收益。

---

## 195. General Non-Clairvoyant KV-Cache Scheduling via Regime-Aware Routing

**arXiv ID:** 2607.09248 | [PDF](https://arxiv.org/pdf/2607.09248v1)

**作者:** Yiding Feng `[一作]` (Hong Kong University of Science and Technology), Yuhao Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对大语言模型推理中的 KV‑缓存内存受限、响应长度未知的非clairvoyant调度问题，提出了首个常数竞争比（O(1)）的调度算法。

**💡 创新点**

创新点在于：①将任务分解为三类几何范式（大任务、小提示占优、小响应占优）并分别设计专用子调度器；②引入路由式元调度器，在不预知任务类型的情况下动态划分并共享内存预算；③在每个范式中实现了新的非clairvoyant调度技术（矩形条带调度和几何切片），从而突破了传统优先级规则无法同时兼顾内存利用率与面积顺序的瓶颈。

**🔧 技术方法**

主要技术包括：非clairvoyant矩形条带调度（使用宽度固定、长度不可预知的几何递增尝试与贪心分配），几何切片调度（使用响应长度上限与代理提示长度构造相同响应的批处理），以及元调度框架（时间共享与动态路由）。此外，还利用面积下界与内存利用率分析来证明竞争比。

**📊 数据集**

本文为理论研究，未使用任何实际数据集；所有证明均基于抽象的调度实例与数学下界。

**📈 对比分析**

与全知 clairvoyant 调度的最优总完成时间相比，提出的算法在所有实例上实现了常数倍的竞争比（如大任务 36 倍、响应占优 78.67 倍，整体组合常数约 996），并在作业到达在线场景中也保持 O(1) 竞争。

**⚠️ 局限性**

局限性包括：①假设所有任务在时间零同时到达，虽然给出了在线变体但仍是理论模型；②依赖已知提示长度信息；③对内存预算的划分与阶段设置较为复杂，实际系统实现可能需要额外调优；④在极端大内存或极端小内存环境下，常数竞争比的常数值可能偏大，实际性能可能受限。

---

## 196. Beyond Topicality: A Conceptual Analysis of Societal Relevance and Its Application to Search Results and AI Responses

**arXiv ID:** 2607.09264 | [PDF](https://arxiv.org/pdf/2607.09264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 197. OpenProver: Agentic and Interactive Theorem Proving with Lean 4

**arXiv ID:** 2607.09217 | [PDF](https://arxiv.org/pdf/2607.09217v1)

**作者:** Matěj Kripner `[一作]` (Charles University), Milan Straka `[通讯]` (Charles University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为OpenProver的开源大语言模型驱动自动定理证明系统，集成了Lean 4的正式验证，并支持交互式终端操作。

**💡 创新点**

创新点在于将Agentic Planner‑Worker‑Verifier架构与Lean 4正式验证相结合，实现了自动化证明搜索与形式化验证的无缝协作，同时提供可复现的量化评估与人机交互界面。

**🔧 技术方法**

核心技术包括大语言模型（如Kimi‑K2.5、Leanstral）生成链式思考、Planner‑Worker‑Verifier的并行工作流、Lean 4的形式化验证工具以及可视化交互式终端界面。

**📊 数据集**

使用ProofNet数据集进行评估，测试了185条正式定理，并在模型上分别使用Kimi‑K2.5和Leanstral。

**📈 对比分析**

与线性对话式基线相比，OpenProver在Kimi‑K2.5上提升了20.5个百分点、在Leanstral上提升了7.0个百分点，证明了其在有限token预算下的更高成功率。

**⚠️ 局限性**

主要局限包括：形式化过程比非正式推理更难，导致验证失败率较高；系统依赖大语言模型的质量和令牌预算；以及对复杂数学结构的支持仍受限于Lean生态的现有库。

---

## 198. Multi-Agent Reinforcement Learning for SLA-Aware Network Slicing in UAV-Enabled MEC

**arXiv ID:** 2607.09295 | [PDF](https://arxiv.org/pdf/2607.09295v1)

**作者:** Mohammad Farhoudi `[一作]` (Oulu University), Tarik Taleb `[通讯]` (Ruhr University Bochum)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种预测多智能体强化学习框架，实现无人机边缘计算（UAV‑MEC）网络切片在用户移动、任务随机到达和能量/计算受限条件下的 SLA（服务级别协议）稳定保障。

**💡 创新点**

创新点在于：①将用户移动与任务生成预测与多智能体MAPPO联合，形成主动 SLA 维护策略；②构建包含即时、持续与预测违约的综合奖励函数；③采用集中训练、分散执行的学习架构，实现多 UAV 的协同决策；④在仿真中实现与 oracle 等基准的对比。

**🔧 技术方法**

主要技术：D3QL（双 Q‑学习+分裂网络）用于用户位置与任务生成预测；MAPPO（多智能体近端策略优化）用于 UAV 的轨迹控制与计算资源分配；通信/计算速率模型、能量消耗模型与 SLA 违约判定；事件驱动仿真框架。

**📊 数据集**

使用 YJMob100K 移动轨迹数据集；HRLLC/eMBB/mMTC 三类任务分布按论文设定的平均数据大小、CPU 需求与时延阈值采样；仿真参数与表格一致（UAV 3 台、用户 24 台、面积 1000×1000 m² 等）。

**📈 对比分析**

对比方法包括 GA‑Search（离散轨迹搜索）、Greedy（按延迟紧急度分配）、Random（随机动作）以及 Offline‑Optimal（oracle）基准。实验表明，Predictive‑MAPPO 在平均服务延迟、SLA 违约概率以及违约持续时间方面均优于 GA‑Search、Greedy 与 Random，且与 oracle 的差距缩小到可忽略范围；能量消耗保持在与基准相近或略优的水平。

**⚠️ 局限性**

局限性：①仅使用简化的航行能耗模型，未考虑功率控制与干扰影响；②预测误差在 UAV 资源匮乏时对性能影响明显；③未对切片动态接入、带宽动态分配等进一步优化做深入研究；④实验仅在仿真环境中验证，缺乏真实部署验证。

---

## 199. Leveraging Interpretable Tsetlin Machine for PDF Malware Detection

**arXiv ID:** 2607.09290 | [PDF](https://arxiv.org/pdf/2607.09290v1)

**作者:** Rahul Jaiswal `[一作]` `[通讯]` (University of Agder), Rahul Jaiswal (University of Agder)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用可解释的Tsetlin Machine对PDF文件进行静态特征提取，并通过规则学习实现恶意与良性文件的分类；

**💡 创新点**

创新点在于将可解释的规则学习与PDF恶意检测相结合，既保持高检测精度，又能直接输出可解释的判定依据，并且推理时间极短；

**🔧 技术方法**

核心技术包括特征二值化、Tsetlin Machine逻辑句子学习、k折交叉验证、随机欠采样及与决策树、随机森林、KNN、朴素贝叶斯、逻辑回归、XGBoost、LightGBM等传统机器学习模型的对比；

**📊 数据集**

实验使用公开的RIT-PDFMal-2026数据集，该集包含24,337个PDF样本（13,235个良性、11,102个恶意），共42个数值特征；

**📈 对比分析**

与传统机器学习模型比较，Tsetlin Machine达成98.02%的准确率、宏平均F1≈95.99%，推理时间仅2.85µs，虽略低于随机森林（98.28%）但在速度和可解释性上表现更佳；

**⚠️ 局限性**

局限在于仅在单一数据集上评估，解释性验证仅依赖案例展示，未进行正式用户研究，也未覆盖更广泛的恶意PDF家族。

---

## 200. Bidirectional Resource Scheduling for Disaggregated and Asynchronous RL Post-Training

**arXiv ID:** 2607.09207 | [PDF](https://arxiv.org/pdf/2607.09207v1)

**作者:** Tan Zhiqiang `[一作]`, Shi Shaohuai `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

BiDiRL 通过在异步、分离式 LLM RL 后训练中实现双向资源借用，显著减少资源闲置，从而提升整体训练吞吐量。

**💡 创新点**

创新点包括：1) 轻量级热切换运行时实现 Rollout 与 Trainer 的即时角色互换；2) 基于时间性能模型的静态规划器在保证热切换兼容的前提下，选取资源池划分以最小化结构性空洞；3) 以模型预测收益为准则的双向调度器，动态决定何时、如何借用另一池资源，并按剩余工作量进行精细拆分。

**🔧 技术方法**

技术栈包括：vLLM（推理引擎）+ PyTorch FSDP（训练引擎）+ Ray（调度与资源管理）+ 自研热切换与异步 chunk 调度框架；通过时间性能模型与在线剖面动态更新，驱动调度决策。

**📊 数据集**

实验数据集与模型：使用 Qwen3 族大模型（Qwen3VL‑2B、Qwen3VL‑4B、Qwen3‑8B）与两类数据集——多模态 Geo3K 与文本 GSM8K，覆盖 1K–4K 生成长度、不同 staleness 约束、8–32 GPU 资源。

**📈 对比分析**

与 veRL、AReaL、ROLL 在 NVIDIA A6000 与 H100 32 GPU 测试平台上对比，涵盖响应长度、staleness、资源比例、批量大小、模型/数据集等多维度；BiDiRL 在 A6000 上实现 1.05–1.94 倍吞吐提升，H100 上 1.23–1.53 倍，最大 1.68×（A6000）/1.47×（H100）提升；消融实验验证了双向调度与模型驱动的收益决定的必要性。

**⚠️ 局限性**

局限性：① 仍需先行的静态资源划分，对极端动态变化（如模型规模突变、硬件波动）适应性有限；② 热切换开销在短暂闲置窗口时可能抵消收益；③ 需要 Rollout 与 Trainer 的模型布局兼容；④ 评估主要集中在 32 GPU、特定 Qwen 模型与数据集，进一步验证跨平台与更大规模场景仍待探讨。

---

## 201. Application of machine learning to monster level prediction in tabletop RPG game design

**arXiv ID:** 2607.09196 | [PDF](https://arxiv.org/pdf/2607.09196v1)

**作者:** Jolanta Śliwa `[一作]` (AGH University of Krakow), Jakub Adamczyk `[通讯]` (AGH University of Krakow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究使用机器学习将 Pathfinder 2e 怪物的属性映射为等级，构建公开数据集并系统评估多种序数回归模型。

**💡 创新点**

首次公开专门针对 TTRPG 怪物等级预测的数据集，并将时间序列式训练测试与多种序数回归与神经网络模型进行对比。

**🔧 技术方法**

采用经典回归、树基集成（RF、LightGBM、Ordered Random Forest）、专用序数回归算法（SAOC、ORD、IT、CLM、CORN、CONDOR 等）以及针对序数的神经网络（CLM、NNRank、OR‑CNN、CORAL、CORN、CONDOR），并使用 SHAP 进行可解释性分析。

**📊 数据集**

使用 6007 条 Pathfinder Second Edition 怪物记录，包含 33 个特征的公开数据集，托管在 GitHub。

**📈 对比分析**

采用 chronological split 与 expanding‑window 两种时间序列验证方案，指标包括 MAE、MAE^M、RMSE^M、Somers' D、acc、acc@1；树基模型在两方案下均超越线性/核/神经网络，随机森林/LightGBM 近乎完美排序、准确率>98%，平均误差<1等级。

**⚠️ 局限性**

局限性包括数据量有限导致高等级或特殊变种怪物预测不佳，神经网络在小型表格数据上表现欠佳，未充分捕捉特殊能力等非数值特征，需进一步扩展特征与因果解释。

---

## 202. EcoKube: Simulating Carbon-Aware Scheduling Policies in Heterogeneous Edge-Cloud Environments

**arXiv ID:** 2607.09318 | [PDF](https://arxiv.org/pdf/2607.09318v1)

**作者:** Gonçalo Ferreira `[一作]` (University of Amsterdam), Shashikant Ilager `[通讯]` (University of Amsterdam)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可配置的离散事件模拟框架，用于在异构边缘‑云环境中评估可持续性调度策略。

**💡 创新点**

创新点包括：①提供可复现、模块化的调度模拟流水线；②整合网格碳强度(CI)、机房能效(PUE)和节点硬件异构性；③给出可扩展的参考策略并公开实现细节。

**🔧 技术方法**

使用技术：Go语言实现的调度引擎、Python脚本做后处理；CI/Topology适配器、基于权重的打分公式；与Kubernetes兼容的硬约束过滤；离散事件仿真与可配置权重。

**📊 数据集**

数据集：合成批量工作负载（CPU/GPU混合、事件驱动等）；Wattnet 2024碳强度时序；三站（边缘/云）拓扑与节点异构性描述。

**📈 对比分析**

对比方法：与默认Kubernetes调度器、KEIDS、TOPSIS/KCSS三种基线使用相同输入与随机种子进行50次重复；指标包括估计碳排放、完成时间、平均等待时间、已完成作业数。实验显示参考策略相较默认基线碳排放降低约45%，仅略微增加完成时间。

**⚠️ 局限性**

局限性：仅基于仿真，模型对真实工作负载、网络延迟与能耗的逼真度有限；实验规模受限于三站拓扑；未包含更丰富的基线或大规模真实集群验证；权重设定需进一步泛化。

---

## 203. Faster Exact Algorithms for Equal-Subset-Sum

**arXiv ID:** 2607.09289 | [PDF](https://arxiv.org/pdf/2607.09289v1)

**作者:** Ryosuke Yamano `[一作]` (University of Tokyo), Tetsuo Shibuya `[通讯]` (University of Tokyo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了新的多项式空间和指数空间的 Exact 计算 Equal-Subset-Sum 的算法，显著降低了时间与空间的上界；

**💡 创新点**

创新点在于通过随机输入变形与解决方案大小的概率削减，以及结合低空间 Element Distinctness 和子集和快速查询技术，得到更优的时间‑空间权衡；

**🔧 技术方法**

主要技术包括随机替换（RandomReplace）减少最小解大小、基于模算术的等和搜索、Schroeppel‑Shamir 计数与遍历、Fast Subset‑Sum Oracle 及随机访问结构；

**📊 数据集**

论文未使用公开数据集，而是针对理论上的随机整数实例进行分析与证明；

**📈 对比分析**

与此前最优的 1.7067ⁿ/空间 1.7067ⁿ 的算法相比，新算法在时间上提升到 1.6994ⁿ、空间降低到 1.5664ⁿ；在多项式空间下，速度从 2.6817ⁿ 降到 2.5430ⁿ；

**⚠️ 局限性**

限制在于仍为 Monte Carlo 算法，误差可通过重复降低但无法消除；此外，时间空间折衷曲线在极小空间（α≤0.12）时尚未最优，且对非常稠密输入的表现仍未完全优化。

---

## 204. LionVote: Per-Layer Learning Rate Adaptation for Lion

**arXiv ID:** 2607.09266 | [PDF](https://arxiv.org/pdf/2607.09266v1)

**作者:** Kris Atallah `[一作]` `[通讯]` (New York University), Kris Atallah (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并评估了一种称为LionVote的按层学习率自适应机制，利用梯度方向稳定性与动量健康两项诊断，对Lion优化器的有效学习率进行整数级别调节。

**💡 创新点**

①基于几何身份和EMA时常推导出阈值的投票机制；②揭示Lion在ViT中不同层类型的有效尺度存在2.6–2.8倍误差，并证明单一全局率无法补偿；③证明按层自适应在层异质性较大时能显著提升性能。

**🔧 技术方法**

Lion优化器、指数移动平均（EMA）动量、梯度方向相似度、动量与梯度比例检查、验证损失打破投票冲突、按层整数级别调节学习率、随机种子、ablation与显著性检验。

**📊 数据集**

CIFAR-10、CIFAR-100，模型为WideResNet-28/40和ViT-Tiny。

**📈 对比分析**

与AdamW、SGD+cosine以及无投票的Lion对比；在ViT-Tiny/CIFAR-100上LionVote(阶8)达到69.7% top‑1（比Lion高0.75pp，p<0.02），与AdamW在ViT-Tiny/CIFAR‑10上持平；在WRN上SGD仍占优势。

**⚠️ 局限性**

实验仅覆盖CIFAR数据集与小模型，未检验更大规模网络；投票冲突主要由验证损失解决，需在hold‑out验证确认；Vote2几乎不发火；缺乏跨任务与跨架构的验证，统计功效有限。

---

## 205. AnythingReality: Robust Online Gaussian Splatting SLAM for Open-Vocabulary VR Scene Exploration

**arXiv ID:** 2607.09260 | [PDF](https://arxiv.org/pdf/2607.09260v1)

**作者:** Timofei Kozlov `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个集成的在线3D高斯喷射重建、实时VR探索和语音驱动的VLM交互系统AnythingReality

**💡 创新点**

创新点在于将ORB‑SLAM3姿态估计、Gaussian‑plus‑SDF在线映射、WebXR立体VR渲染与语音VLM交互结合，并在噪声RGB‑D场景中实现低延迟、可交互的增量重建

**🔧 技术方法**

采用了ORBSLAM3、Gaussian‑plus‑SDF SLAM、WebRTC/WebXR、faster‑whisper语音识别、OpenAI兼容VLM以及vLLM式本地部署

**📊 数据集**

在四个RealSense室内序列（Our_Lab、Our_Objects、Our_Tables、Our_Kitchen）、Replica和TUM‑RGBD公开数据集上进行评测

**📈 对比分析**

与GS‑ICP SLAM、RTG‑SLAM、GPS‑SLAM等最先进方法比较，显示在图像质量（PSNR、SSIM、LPIPS）和重建帧率方面均优于对比方法，且在噪声深度场景下具有更高的鲁棒性

**⚠️ 局限性**

局限性包括缺乏持久的三维语义地图、仅基于视图的VLM交互、对深度噪声的进一步优化空间以及对多视角语义关联的支持不足

---

## 206. Blockchain-Linked Auditable Decision Management for Telecom/IoT Fraud-Control Requests

**arXiv ID:** 2607.09259 | [PDF](https://arxiv.org/pdf/2607.09259v1)

**作者:** Saviz Changizi `[一作]` (University of Surrey), Rahim Tafazolli `[通讯]` (University of Surrey)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了基于区块链的可审计决策管理框架，将电信/物联网诈骗控制视为请求级决策流程，并在受控合成部署重放环境中评估三种风险源（集中ML、联邦元学习、LLM）。

**💡 创新点**

将诈骗控制从单纯检测转为统一的请求级决策管理，提出硬/软诈骗门控、五态政策、两区细化与区块链审计的完整体系，并首次在同一决策链下对LLM零射与QLoRA与传统模型进行对比；同时在同一合成数据集上实现可追溯、可审计的部署生命周期。

**🔧 技术方法**

集中化机器学习集成（AVG6）、联邦元学习、LLM（Qwen系列）通过QLoRA微调、以太坊兼容区块链智能合约、Ganache本地链、时间序列泄漏控制与交叉验证等技术。

**📊 数据集**

合成电信/物联网使用记录数据，训练集27,000条记录（软硬诈骗分布15%）、部署集100,000条记录，人工注入8种诈骗类型（软硬双重实现），并模拟运营上下文与漂移。

**📈 对比分析**

采用同一请求子平台、同一决策策略下的部署重放，对比准确率、召回、合法请求FPR、误报率、区块链交易量、gas/吞吐量等指标；结果显示M1（集中ML）提供最优平衡，M2在提升软诈骗召回的同时FPR上升，M3-Base性能最差，M3-QLoRA性能接近M1/M2但未超过。

**⚠️ 局限性**

仅在受控合成数据上验证，未覆盖真实运营流量与复杂性；区块链评估在本地Ganache，未体现真实共识网络的性能与安全；未实现在线漂移监控、模型再校准及隐私友好审计等实际部署需求。

---

## 207. Low-Complexity Successive-Cancellation List Decoding of $2\times2$ Kernel Non-Binary Polar Codes

**arXiv ID:** 2607.09257 | [PDF](https://arxiv.org/pdf/2607.09257v1)

**作者:** Xinyu Zhou `[一作]` (Fuzhou University), Pingping Chen `[通讯]` (Fuzhou University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文研究了基于 2×2 非二进极化码的低复杂度 Successive Cancellation List（SCL）解码方法。

**💡 创新点**

创新点在于提出三种低复杂度解码器——SR-NBSCL、ESR-NBSCL 与 ABP-NBSCL，通过可靠性阈值、终端 Rate‑1 节点简化、以及累积可靠度偏差（ARD）剪枝，有效减少 q‑ary 分支扩展与排序复杂度。

**🔧 技术方法**

采用高斯逼近（GA）阈值、可靠度偏差指标、非二进极化码构造与 SC/SCL 更新规则、以及路径计数与剪枝策略等技术。

**📊 数据集**

使用 GF(4)、GF(8)、GF(16) 上的长度 N=128、速率 R=0.5、列表大小 L=8 的码；信道为 BPSK 加 AWGN，仿真数据基于 Monte‑Carlo 生成的码字。

**📈 对比分析**

与传统 NB‑SCL、SR‑NBSCL、ESR‑NBSCL 对比，ABP‑NBSCL 在 FER 与 NB‑SCL 相近的同时，PSN（路径分裂数）下降 80%+，平均路径数与有效分支因子均显著降低，证明了性能与复杂度的优越折衷。

**⚠️ 局限性**

主要限制包括：阈值需离线预先设计，对极化码构造与字段阶数的依赖较强；在极高 SNR 或大字段场景下可能仍出现小幅性能损失。

---

## 208. Git-Assistant: Planning-Based Support for Updating Git Repositories

**arXiv ID:** 2607.09224 | [PDF](https://arxiv.org/pdf/2607.09224v1)

**作者:** Alfredo Garrachón Ruiz `[一作]` (AI Research, JPMorganChase), Daniel Borrajo `[通讯]` (AI Research, JPMorganChase)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种混合 AI 工具，结合大型语言模型与自动化规划来解析自然语言请求并生成可靠的 git 命令序列；

**💡 创新点**

创新点在于将 LLM 的意图理解与 PDDL 规划相结合，形成可验证、可安全执行的操作序列；

**🔧 技术方法**

使用 GPT‑4o 进行自然语言解析、GitPython 采集仓库状态、Fast‑Downward 规划器完成规划，最终生成 git 命令；

**📊 数据集**

构造了两类数据集：手工设计的基准环境（含本地/远程分支、未追踪文件等）和随机生成的协作环境；共生成 200 条请求（100 条基准 + 100 条随机）并手工标注真值；

**📈 对比分析**

通过比较 LLM‑Vanilla、LLM‑based 与 Hybrid‑Planner 三种变体，评估准确率、执行时间、错误率及远程/本地/工作树状态匹配率。Hybrid‑Planner 在基准环境中达到 81% 准确率，错误率仅 3%；在随机环境中仍领先，准确率 59%，显著优于纯 LLM 方案；

**⚠️ 局限性**

局限包括仅支持常见 git 操作，未覆盖 clone、restore、diff 等；未处理复杂冲突恢复、删除文件同步、分支恢复等；依赖仓库规模，Observer 时间随文件/提交数增大；在极端并发或非标准工作流场景下表现未知。

---

## 209. Network-distance decay of perceived online social support

**arXiv ID:** 2607.09210 | [PDF](https://arxiv.org/pdf/2607.09210v1)

**作者:** Masanori Takano `[一作]` (CyberAgent, Inc.), Fujio Toriumi `[通讯]` (University of Tokyo)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合两波问卷和行为日志，训练随机森林模型推断在线社交支持，随后在Pigg Party互动网络上研究其随图距离的纵向关联，并用个体模拟验证重尾衰减机制。

**💡 创新点**

首次将感知在线社交支持视为可测量的节点心理状态，并证明其在大型虚拟社群中呈现比单指数更慢的距离衰减，揭示网络位置异质性驱动的重尾效应。

**🔧 技术方法**

使用随机森林预测、网络距离回归、Davidson‑MacKinnon J检验、对数线性模型，以及基于随机游走的个体传播模拟。

**📊 数据集**

Pigg Party avatar通信应用的用户行为日志、两波问卷数据，以及Twitter、Facebook、大学邮箱等公开网络作为模拟基准。

**📈 对比分析**

对比指数衰减和幂律衰减模型，J检验及调整R²显示在1–6跳范围内幂律拟合更佳；预测模型的r≈0.36、R²≈0.12；模拟结果表明BA/ER网络支持幂律，WS网络更符合指数。

**⚠️ 局限性**

预测精度仅中等，未实现因果推断，模拟模型简化了内容、权重、时间等因素，结果可能不完全反映真实支持传播。

---

## 210. Complexity-Guided Component-wise Initialization for Language Model Pretraining

**arXiv ID:** 2607.09204 | [PDF](https://arxiv.org/pdf/2607.09204v1)

**作者:** Konstantin Garbers `[一作]` (Peking University), Nicholas Oh `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析了多种 GPT‑2 风格模型的权重谱，并尝试将这些谱特征作为初始化信号用于新模型的预训练。

**💡 创新点**

创新点在于首次系统地量化不同规模、语言、分词器和语料库下的层级与子组件谱趋势，并构造了基于这些谱趋势的初始化方案。

**🔧 技术方法**

主要技术包括 Frobenius 范数、有效秩熵谱分析、谱形状重塑以及对比传统高斯初始化和预训练权重复用的初始化方法。

**📊 数据集**

使用的实验数据集包括 SlimPajama‑6B 进行预训练，验证集为 FineWeb‑Edu、OpenWebText、WikiText‑103，评测数据包括 BLiMP、ARC‑Challenge/Easy 以及 WinoGrande。

**📈 对比分析**

与标准高斯初始化、增大标准差、去除残差缩放、以及直接预训练权重复用等方法对比，结果显示谱基初始化在验证困惑度、BLiMP 语法准确率和多项选择准确率上并未取得系统性提升，预训练权重复用仍保持竞争力。

**⚠️ 局限性**

局限性包括仅关注 GPT‑2 解码器架构、谱形状过于粗糙缺乏子空间或奇异向量信息、实验仅在单一随机种子和单一数据集上验证，且无法证明谱结构与性能的因果关系。

---

## 211. When is Routing Meaningful? Diversity and Robustness in Language Model Societies

**arXiv ID:** 2607.09197 | [PDF](https://arxiv.org/pdf/2607.09197v1)

**作者:** Fantine Huot `[一作]` (Google DeepMind), Mirella Lapata `[通讯]` (Google DeepMind)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究多模型系统中的路由器，提出除了任务准确率之外，还需评估社会多样性与路由稳健性。

**💡 创新点**

创新点在于引入Hierarchic Social Entropy衡量行为多样性，以及基于扰动的路由稳健性指标，并用这两种结构属性评估路由有效性。

**🔧 技术方法**

使用HSE对行为向量计算余弦距离、单链接层次聚类；定义5级扰动分类并计算路由一致性；比较KNN、提示式、随机路由策略。

**📊 数据集**

实验数据集为EmbedLLM（112个模型、36k题）和RouterBench（11个模型、36k查询），并构造RD、SA等专家合成社会。

**📈 对比分析**

通过与基线随机、KNN和提示路由对比，发现高HSE能提升路由准确率但KNN鲁棒性下降，提示路由在准确率略低时稳健性更好；最大HSE子集能显著提升鲁棒性。

**⚠️ 局限性**

局限在于HSE依赖固定评估集、最大HSE子集使用贪心算法未必最优、合成专家假设完美、鲁棒性仅评估单轮查询不涵盖多步交互。

---

## 212. Robot Trajectron V3: A Probabilistic Shared Control Framework for SE(3) Manipulation

**arXiv ID:** 2607.09315 | [PDF](https://arxiv.org/pdf/2607.09315v1)

**作者:** Pinhao Song `[一作]` (KU Leuven), Renaud Detry `[通讯]` (KU Leuven)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Robot Trajectron V3 (RT‑V3)，一种基于贝叶斯推断的 SE(3) 抓取共享控制框架，能够在低带宽接口下通过学习用户意图先验来实时辅助人机协作；

**💡 创新点**

核心创新包括：①将点云与抓取候选姿态通过 transformer 编码成上下文先验；②采用 CVAE‑GMM 结构捕捉多模态用户意图；③对 SE(3) 动作进行平移‑旋转因式分解，降低联合分布学习难度；④引入异步共享控制机制，在用户不发指令时自动执行预测动作，减轻用户负担；

**🔧 技术方法**

技术实现主要依托 transformer‑based encoder、LSTM 动力学编码、CVAE‑GMM 先验建模、贝叶斯后验推断、异步控制策略，并在 Franka Research 3 机器人上实现；

**📊 数据集**

使用大规模仿真数据集（303 种物体，22,265 场景，833,324 个抓取候选，568,277 条轨迹）进行训练；此外在真实实验中采集 12 场工作空间的 21 名参与者数据；仿真数据来源于 CuRobo 与 NeoSS 抓取规划；

**📈 对比分析**

与 Direct、Hindsight Optimization、NeoSS、RT‑V2 以及 6D 预测变体进行对比。结果表明 RT‑V3 在轨迹预测的 ADE/FDE 上达到最佳；在仿真与真实共享控制中，成功率提升至 86.9%（比 Direct 78.6%、HO 74.7%），碰撞率、未匹配率、时间超限率均显著降低；同时用户输入次数、任务时长及 NASA‑TLX 负荷指标也大幅改善；

**⚠️ 局限性**

存在的局限包括：①先验训练以规划器示范为基础，与真实人类操作存在分布偏移；②对抓取规划器的依赖，若规划器生成的候选姿态不完整或不可行会影响性能；③模型规模受限，可能无法捕捉所有人类多模态行为；④仅针对抓取任务，缺乏跨任务通用性；

---

## 213. LLMs for health: Perceived benefits, risks, intention to use AI chatbots, and willingness to self-disclose across sensitive health topics

**arXiv ID:** 2607.09253 | [PDF](https://arxiv.org/pdf/2607.09253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 214. TextileNet: Towards Zero-shot Text-style Segmentation of Manuscripts

**arXiv ID:** 2607.09299 | [PDF](https://arxiv.org/pdf/2607.09299v1)

**作者:** Anguelos Nicolaou `[一作]` (University of Graz), Georg Vogeler `[通讯]` (University of Graz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发并验证了名为TextileNet的全卷积多任务网络，利用合成数据训练，能够生成细粒度文本纹理嵌入，并在中世纪手稿上实现零样本写者与性别识别。

**💡 创新点**

通过将像素级相对比例三元组损失与多任务学习结合，实现零样本迁移；并设计了专门的古文人学者问卷实验，为手稿写作风格评估提供人类基准。

**🔧 技术方法**

使用IUnet骨干的全卷积网络，1×1卷积多任务头，像素级三元组损失，Chamfer距离聚合，kNN检索和逻辑回归基线等技术。

**📊 数据集**

主要使用自行生成的合成伪页面以及拿坡里女修道院手稿三册（ASN 1401）和用户问卷图像作为评估数据集。

**📈 对比分析**

对比零样本kNN与有监督LR，性别识别准确率约90%（受位置偏差影响），写者识别kNN/ LR约70–90%；在问卷上TextileNet整体准确率为67.5%，显著优于随机基线。

**⚠️ 局限性**

需要高显存GPU，无法有效过滤噪声或破损区域，聚合与比较方式过于简易，且缺乏无监督分割手稿写手区域的算法。

---

## 215. Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning

**arXiv ID:** 2607.09287 | [PDF](https://arxiv.org/pdf/2607.09287v1)

**作者:** Ivan Ilin `[一作]` (King Abdullah University of Science and Technology), Peter Richtárik `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出两种新的参数高效微调方法：Super（使用Wanda风格的激活加权幅度评分一次性确定稀疏支持）和Supra（将Super的稀疏支持与LoRA低秩适配器按预算分配结合）。

**💡 创新点**

创新点在于将剪枝中无梯度、训练免费、基于激活的saliency信号直接用于稀疏微调；以及提出预算匹配的低秩-稀疏混合策略Supra，能够在同等可训练参数量下显著提升算术推理性能。

**🔧 技术方法**

主要技术包括：Wanda激活加权幅度评分、稀疏掩码生成、LoRA低秩适配器、预算匹配的rank转换规则、基准数据集Math17K的单任务微调以及对AddSub、MultiArith、SingleEq、GSM8K、AQuA、SVAMP六大算术推理基准的评估。

**📊 数据集**

使用了Math17K算术指令微调数据集，并在AddSub、MultiArith、SingleEq、GSM8K、AQuA、SVAMP六个公开算术推理基准上进行评测。

**📈 对比分析**

与LoRA、SIFT、RoSA等现有PEFT方法对比，实验显示：在1B模型上Supra（BottomK，λ=0.8）平均准确率达62.23%，超过LoRA 61.07%；在8B模型上Supra-Mag（BottomK，λ=0.3）平均准确率达79.12%，比LoRA高5.95个百分点，接近或略优于单纯的低幅度（Magnitude）稀疏基线。

**⚠️ 局限性**

主要局限包括：仅单一随机种子实验；结果为schedule‑selected的最佳观察值，未覆盖多种训练时序；评测仅限算术推理任务，未验证对其他任务或更大模型的适用性；掩码选取基于近似saliency，可能缺乏动态适配；稀疏支持在训练过程中保持静态；实现效率受限于当前稀疏矩阵实现，未做专门的稀疏运算加速。

---

## 216. Autoregressive latent diffusion for 3D molecule generation

**arXiv ID:** 2607.09277 | [PDF](https://arxiv.org/pdf/2607.09277v1)

**作者:** Federico Ottomano `[一作]` (Imperial College London), Alex M. Ganose `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了KRONOS，一种在预训练统一自编码器潜在空间中进行的自回归扩散框架，用于生成三维分子并支持变量长度生成与片段条件生成。

**💡 创新点**

将自回归生成与潜在空间扩散结合，利用混合训练策略（Fill-in-the-Middle）实现无架构修改即可兼顾无条件与片段条件生成。

**🔧 技术方法**

使用预训练Unified AutoEncoder (UAE)进行潜在表示，变分扩散损失（Diffusion Loss）与自回归Transformer，以及停机分类器。

**📊 数据集**

在QM9和GEOM-Drugs两个标准三维分子数据集上进行评估。

**📈 对比分析**

与多种自回归（G-SchNet、Symphony、QUETZAL、NEAT）及扩散（EDM、MiDi、JODO、ADiT）基线对比，KRONOS在QM9上实现最高无条件有效率（97.3%）并在GEOM-Drugs上表现最优，片段条件生成的准确率和稳定性也明显优于现有方法。

**⚠️ 局限性**

受限于预训练自编码器的表达能力、缺乏完整的置换不变性、依赖固定BRICS分片方案以及目前仅支持无条件与片段条件两种生成模式。

---

## 217. Forget Narrowly, Retain Broadly: Unlearning as an Asymmetric Generalization Problem

**arXiv ID:** 2607.09236 | [PDF](https://arxiv.org/pdf/2607.09236v1)

**作者:** Amit Peleg `[一作]` (Tübingen AI Center, University of Tübingen), Matthias Hein `[通讯]` (Tübingen AI Center, University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套细粒度的机器学习模型忘记（unlearning）评估协议与数据集，并基于此提出了一种新的拒绝式忘记算法；

**💡 创新点**

创新点在于：1）构建了同时覆盖“遗忘”与“保留”两侧的对称泛化评估框架，细分语义、句法、词汇层面的边界；2）设计了新的分布匹配目标与随机前缀混合技术，动态平衡遗忘与保留损失，提升拒绝生成质量；

**🔧 技术方法**

使用技术包括：分布匹配（Jensen‑Shannon 失真）、随机前缀混合（stochastic prefix‑mixing）、动态损失重标（α‑scaling）、样本配对（forget‑retain pairing）等；

**📊 数据集**

使用的数据集为自建的“Selective Unlearning of Isolated Topics and Events”数据集，涵盖4个真实事件（挑战者灾难、塞勒姆女巫审判、乔布斯健康、Britney Spears监护），每个主题包含遗忘集与保留集，并细分语义层级、句法/词汇相似度；

**📈 对比分析**

与10种现有忘记方法（如GradDiff、NPO、PDU、DPO、JensUn等）在三大LLM上进行比较，实验显示新方法在遗忘率≤3%且拒绝率为0%时，保留性能下降≤3%，且在顺序与联合忘记场景均优于其它方法；

**⚠️ 局限性**

限制：1）忘记/保留层级固定，需针对具体任务调整；2）语义层级划分依赖于人类标注或其它模型，可能影响泛化；3）仅在所选4个主题上验证，未覆盖更大范围的实际应用场景。

---

## 218. Temporal Knowledge Graph Forecasting under Distribution Shifts: A Synthetic Evaluation

**arXiv ID:** 2607.09232 | [PDF](https://arxiv.org/pdf/2607.09232v1)

**作者:** Konrad Özdemir `[一作]` (University of Mannheim), Heiner Stuckenschmidt `[通讯]` (University of Mannheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究利用可控的合成时间知识图谱生成器，系统评估了七种 TKG 预测模型在递归、同质性与周期性三种信号机制下，尤其关注在平稳与突变（断裂）环境中的鲁棒性与适应性。

**💡 创新点**

创新点在于提出了可精细调节三种时间结构信号（recurrence、homophily、periodicity）的合成生成器，并在同一实验框架下比较模型在分布偏移前后表现，揭示模型鲁棒性高度依赖于信号类型。

**🔧 技术方法**

技术手段包括：基于权重采样的合成生成器、六种主流 TKG 预测架构（RE‑GCN、TLogic、TimeTraveler、DiMNet、CognTKE、EdgeBank‑RX）、记忆基准 Recurrency Baseline，以及 oracle 参考；评价指标为时间感知 MRR 与 Hits@k。

**📊 数据集**

数据集为 100 个时间戳、2500 个实体、16 条关系的合成 TKG 数据，针对递归、同质性、周期性分别生成持久与断裂两种实验情景。

**📈 对比分析**

通过 oracle 归一化与 MRR/Hits@k 对比，发现递归与周期性持久时大多数模型能达到 90%+ 的 oracle 分数，记忆基准表现尤为强劲；同质性持久时 DiMNet、TimeTraveler、CognTKE 较佳；在断裂情景中，同质性与周期性均显著下降，递归类模型相对更鲁棒，整体表明不同模型对不同结构信号的适应性差异显著。

**⚠️ 局限性**

局限性包括：仅考虑三种信号机制且参数设定固定；断裂为突变而未覆盖渐进式分布漂移；合成数据可能与真实世界分布差异，结果仅能说明相对鲁棒性而非普适结论。

---

## 219. Glob3R: Global Structure-from-Motion with 3D Foundation Models

**arXiv ID:** 2607.09225 | [PDF](https://arxiv.org/pdf/2607.09225v1)

**作者:** Junyuan Deng `[一作]` (Hong Kong University of Science and Technology), Ping Tan `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于3D基础模型的全局SfM框架Glob3R，通过在Pi3X上增添稠密匹配头和滑动窗口关键帧策略，将前向预测的几何信息转换为可优化的对应约束，并在全局尺度上进行运动平均和束缚优化，实现高精度全局重建。

**💡 创新点**

创新点在于将feed-forward 3D模型的几何先验与全局SfM优化相结合；通过稠密图像变换生成稀疏可靠的多视角轨迹，采用关键帧滑动窗口关联实现长序列和无序集合的可扩展重建。

**🔧 技术方法**

使用Pi3X冻结的Transformer骨干、稠密匹配头、关键帧滑动窗口关联、旋转/平移平均、全局束缚优化，以及稠密深度重投影。

**📊 数据集**

在Tanks and Temples、TUM RGB-D、ETH3D、KITTI等多种室内外、无序和长序列数据集上进行实验。

**📈 对比分析**

与COLMAP、GLOMAP、DA3、Pi3X、DROID‑SLAM、VGGT‑Long、LoGeR、Scal3R、SAIL‑Recon等基线比较，Glob3R在T&T、KITTI、ETH3D等任务上均取得更高的PSNR、更低的位姿RMSE或更好的RRA/RTA指标，显示出更鲁棒、更精确的重建。

**⚠️ 局限性**

局限在于固定滑动窗口大小，匹配头仍依赖基模型的几何预测，对初始几何误差的鲁棒性有待提升，且对极端低纹理或大视角差异的场景可能表现不足。

---

## 220. Co-evolution of self-replication and function in a digital primordial soup

**arXiv ID:** 2607.09211 | [PDF](https://arxiv.org/pdf/2607.09211v1)

**作者:** Francesco Cicala `[一作]` (Google), Blake Richards `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在数字原始汤中自发产生自复制与问题求解的共同进化，使用随机32字节Z80汇编程序通过竞争和交互演化出复制与多阶多项式计算能力。

**💡 创新点**

引入竞争门控验证将求解任务与复制机会分离，并证明任务压力能加速紧凑复制结构出现；利用跨域扩散和空间分层自发形成学习课程；展示代谢约束促成条件性停止。

**🔧 技术方法**

Z80汇编模拟、竞争门控交互、代谢惩罚、交叉域扩散、空间网格、随机突变及进化分析工具。

**📊 数据集**

随机生成的32字节程序、32个不同多项式任务（共32种）、总计512×1024格网格。

**📈 对比分析**

与平滑Fitness、单任务设置、无验证控制对照；仅在交叉率适中（约0.05）时能解决高阶多项式，单任务和极端交叉率无法进展；在1M世代内实现多阶多项式解法，任务成功率随交叉率0.05最高。

**⚠️ 局限性**

程序长度限制仅允许单体解法；任务仅为多项式，缺乏多元化；未深入探讨重组、协同解题；未评估更大规模或多代理协作。

---

## 221. Interference and Retention in Continual Learning

**arXiv ID:** 2607.09202 | [PDF](https://arxiv.org/pdf/2607.09202v1)

**作者:** Julius Störk `[一作]` `[通讯]` (VARTA Microbattery GmbH), Julius Störk (VARTA Microbattery GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种新的持续学习方法，直接将遗忘建模为任务之间的干扰，而不是依赖于后处理机制，如重放、弹性正则化或蒸馏。

**💡 创新点**

创新点在于提出了干扰门控功能分配（Interference-Gated Functional Allocation, igfa），这是一种无重放、无Fisher的方法，能够在任务对齐时共享方向，在任务冲突时保护方向。

**🔧 技术方法**

使用了路径平均曲率和干扰功能的几何分析，结合了在线和离线的模型合并与持续学习。

**📊 数据集**

使用了多个基准数据集进行实验，包括Split-Digits和Rotated-Digits等，验证了方法的有效性。

**📈 对比分析**

与其他方法（如EWC、重放和结构基线）进行比较，igfa在不同任务流中表现出无遗忘的保留能力，并在相似任务中提高了性能，尤其是在任务相似性较高的情况下。

**⚠️ 局限性**

限制在于该方法的准确性在深层网络中可能会下降，且在特征漂移的情况下，性能可能不如预期。

---

## 222. Configurable AI Coding Assistants: Designing For Developers Who Like to Be in Control

**arXiv ID:** 2607.09215 | [PDF](https://arxiv.org/pdf/2607.09215v1)

**作者:** Ekaterina Koshchenko `[一作]` (JetBrains Research), Agnia Sergeyuk `[通讯]` (JetBrains Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 AI 编码助手的可配置性进行系统研究，收集并归类已有工具的配置选项，结合问卷与共创设计会话，分析经验丰富开发者对配置需求、优先级与界面偏好的认知与实践。

**💡 创新点**

提出四大配置维度（代码建议、系统与策略、人机交互、用户与个人上下文），系统地映射出需求缺口与现有实现差距，并揭示开发者将项目级与任务级配置区分开来、倾向于在主设置窗口与快捷访问窗口分别放置的设计模式。

**🔧 技术方法**

采用定量问卷调查（Likert量表）与定性共创设计（Miro 框图、半结构化访谈）两种技术手段，并通过统计分析（使用度、可用度、差距度）量化配置需求。

**📊 数据集**

数据集为 165 名经验丰富的专业开发者完成的在线问卷，涵盖 1848 条配置评分；以及 10 名受访者在共创设计会话中产生的 231 条配置选择与布局记录。

**📈 对比分析**

比较方法为对比“使用价值”与“已实现可用性”两项指标，计算正向价值占比（72.6%）与实际可用率（35.6%），进一步计算需求缺口。实验未涉及传统性能指标，主要通过平均使用度 1.1/2（满分2）评估总体可用感知，发现高需求配置多集中于交互层面。

**⚠️ 局限性**

局限性包括：依赖自我报告导致记忆与社会期望偏差；样本仅为经验丰富的开发者，难以推广至新手或偶尔使用者；共创设计受 Miro 界面与预设控件集约束，可能影响真实偏好；未在真实代码编辑环境中验证配置对生产率或错误率的实际影响。

---

## 223. Creativity, honesty and designed forgetting emerge in small hyperbolic language models

**arXiv ID:** 2607.09306 | [PDF](https://arxiv.org/pdf/2607.09306v1)

**作者:** Kwan Soo Shin `[一作]` (PolymathMinds Lab), Yunkyung Min `[通讯]` (Korean Educational Development Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了三种小型语言模型和一个记忆操作系统，在超平面子空间上实现创造性、诚实与设计性遗忘，以打造可伴随的人工友。

**💡 创新点**

证明超平面几何可在规模不大的模型中显现伴随所需特质，并提出可携带的记忆操作系统与合规检测器，突破规模、数据中心依赖与忘却缺陷。

**🔧 技术方法**

使用 Lorentz 负曲率与球面正曲率双曲空间训练、Farthest-Point Sampling、Karcher 最小化、贝叶斯合规评估、LoRA 适配器等技术。

**📊 数据集**

采用 3.32M 帧配对问答、812K 语句–轨迹–标签对、1.84B 训练语料（七轴）等数据集，涵盖英韩混合框架与多领域对话。

**📈 对比分析**

与前沿 LLM 的零样本判断对比，BS 审计机实现 90.7% 的二分类合规准确率；S3 创意模型在 100% win‑rate；LSM‑OS 在四种条件下展示骨架/壁纸双曲存留曲线，优于无门控与统一门控。

**⚠️ 局限性**

记忆模型尚未跨用户验证；审计器仅覆盖四类不诚实行为；记忆分离理论在长期真实用户中的验证待后续；模型对语言多样性与跨文化的普适性仍待测试。

---

## 224. From Classification to Localization and Clinical Validation: Large-Scale Development of a Deep Learning System for Thoracic Disease Detection on Chest Radiographs in Thailand

**arXiv ID:** 2607.09305 | [PDF](https://arxiv.org/pdf/2607.09305v1)

**作者:** Isarun Chamveha `[一作]` (Perceptra Co., Ltd.), Warasinee Chaisangmongkon `[通讯]` (King Mongkut's University of Technology Thonburi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发并全面验证了 Inspectra CXR v5 这一胸部 X 光图像分析模型，支持多标签疾病分类与弱监督定位。

**💡 创新点**

创新点在于将 DenseNet-121 与 Attend‑and‑Compare Modules、Probabilistic Class Activation Map 以及 Cut‑Noise 结合，实现同一模型同时输出诊断分数与可解释热图，并通过本地化训练显著提升在泰国医院间的泛化能力。

**🔧 技术方法**

使用的技术包括深度卷积网络 DenseNet‑121、ACM、外部注意力解码器、PCAM 聚合层、Cut‑Noise 去噪、Adam 优化器和弱监督定位策略。

**📊 数据集**

使用的数据集为 874,858 张 Siriraj Hospital 前景 PA 胸片及对应报告，测试集包括 19,871 张内域样本（Dataset‑A）、5,992 张跨站样本（Dataset‑B）以及 4,549 张有定位标注的样本（Dataset‑C）。

**📈 对比分析**

性能评估通过与放射科医生标注的内域和跨站数据比较 AUROC、敏感度、特异度；定位评估用 LLF、IoU、N/LF；与医生进行分类/定位一致率和 SUS 得分；结果显示内域 AUROC 0.994、跨站 0.970、LLF 77.9%，分类/定位一致率 93.6%/94.7%，SUS 89。

**⚠️ 局限性**

局限在于对小结节、低区或板块性肺不张等细小或被骨骼遮挡病变的检测与定位准确性不足，以及偶尔出现热图过度扩散的问题。

---

## 225. Risk-Aware General-Utility Markov Decision Processes

**arXiv ID:** 2607.09298 | [PDF](https://arxiv.org/pdf/2607.09298v1)

**作者:** Pedro P. Santos `[一作]` (INESC-ID), Francisco S. Melo `[通讯]` (INESC-ID)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出风险感知的通用效用马尔可夫决策过程（GUMDP）框架，并给出基于MCTS的在线规划算法以求解熵风险度量（ERM）目标。

**💡 创新点**

将风险测度（ERM）与GUMDP结合，构造等价的占据MDP，证明通过ERM-MCTS可实现近似最优并可调节风险-收益平衡。

**🔧 技术方法**

使用熵风险度量（ERM）、分布式MDP、占据MDP、Monte Carlo树搜索（ERM-MCTS）等技术。

**📊 数据集**

采用低维四状态示例环境以及10×10网格环境，涵盖成本最小化、最大状态熵探索、模仿学习和多目标MDP等任务。

**📈 对比分析**

与线性目标下的基线ERM-BI对比，ERM-MCTS在成本分布上与基线相近；在非线性任务中，随着β增大，尾部成本下降、期望值上升，验证了风险取向可调节且性能符合预期。

**⚠️ 局限性**

局限性：仅适用于有限离散状态空间；未针对连续或极大状态空间进行扩展；仅实现ERM风险度量，未实验如CVaR等其他风险度量。

---

## 226. Rethinking Monocular Depth Embedding for Generalized Stereo Matching

**arXiv ID:** 2607.09284 | [PDF](https://arxiv.org/pdf/2607.09284v1)

**作者:** Libo Lin `[一作]` (Xi'an University of Technology), Yiguang Liu `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在RAFT‑Stereo框架上引入Depth Anything V2的单目深度先验，设计三大模块（DGIA、Gradient‑GRU、EALS）以提升无监督的立体匹配泛化性能。

**💡 创新点**

创新点包括：① 用变形卷积将单目深度与RGB融合，避免网络宽度过大导致的shortcut learning；② 用深度梯度作为软约束引导GRU迭代，增强对纹理稀疏和遮挡区的鲁棒性；③ 通过边缘置信度抑制随机缩放产生的边缘标注误差，提升边缘精度。

**🔧 技术方法**

技术手段：RAFT‑Stereo迭代优化、Depth Anything V2、变形卷积（DCN）、卷积门控GRU、实例归一化、边缘置信度估计与边缘感知损失、数据增强（颜色抖动、随机缩放等）。

**📊 数据集**

使用SceneFlow作为训练集，零样本测试于Middlebury、KITTI2015、ETH3D、DrivingStereo等多种真实场景数据集。

**📈 对比分析**

与RAFT‑Stereo、IGEV‑Stereo、Selective‑IGEV、DEFOM‑Stereo、MonSter、MGStereo等基线进行对比，零样本EPE在大多数数据集上均位居前列（如KITTI15 1.01，Middlebury 1.61），且推理速度0.284 s，参数约11.23 M，显示出高效且优秀的泛化能力。

**⚠️ 局限性**

局限性：梯度引导对雨天效果有限；单目深度误差仍可能对结果产生负面影响；仅使用Depth Anything V2的基础版，未能充分挖掘更强大先验潜力。

---

## 227. Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative Mining for Sign Language Retrieval

**arXiv ID:** 2607.09263 | [PDF](https://arxiv.org/pdf/2607.09263v1)

**作者:** Junmyeong Lee `[一作]` (Korea Advanced Institute of Science and Technology), KyungTae Lim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对手语检索的视觉感知硬负样本挖掘方法 (SAN)，通过视觉相似性而非文本相似性构造硬负样本，提升细粒度检索性能。

**💡 创新点**

创新点在于将硬负样本挖掘从传统的语言层面迁移到视觉层面，解决手语中因视觉相似导致的负样本分布失配问题。

**🔧 技术方法**

采用对比学习 (Cross‑Lingual Contrastive Learning) 与视觉编码器/文本编码器，结合可靠的手语-词对挖掘、视觉相似负样本搜索和关键词替换生成负样本，最终在损失函数中加入硬负样本约束。

**📊 数据集**

使用 PHOENIX‑2014T 德语手语数据集进行实验，构造细粒度和粗粒度检索测试集。

**📈 对比分析**

与基线模型（CiCo、GFSLT‑VLP）以及基于 FastText、RoBERTa、GPT‑4o‑mini 的文本硬负样本方法比较，SAN 在细粒度检索上提升约 20‑30%（如 V2T R@1 从 17.9% 提升至 39.4%），而对粗粒度检索的影响很小；文本方法性能远低于 SAN。

**⚠️ 局限性**

局限性包括仅在单一域（PHOENIX‑2014T）验证，阈值 β 固定可能不适用于不同签类，且在提升细粒度时可能略微牺牲粗粒度性能，未来需扩展至多语言、动态阈值和语音学信息结合。

---

## 228. Validating Virtual Reality for Studying Multimodal Human-Robot Interaction in Socially Aware Robot Navigation

**arXiv ID:** 2607.09261 | [PDF](https://arxiv.org/pdf/2607.09261v1)

**作者:** Hariharan Arunachalam `[一作]` (LAAS-CNRS, Universite de Toulouse), Rachid Alami `[通讯]` (LAAS-CNRS, Universite de Toulouse)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于VR的实验平台，用于验证在社会化机器人导航中VR是否能再现真实世界中的多模态人机交互动态，并通过与真实环境的对比实验验证其可行性。

**💡 创新点**

创新点在于：①首次将VR环境与真实场景在同一导航任务下进行严格的多模态对比（轨迹、速度、头部朝向等）；②提出了可复现的VR原型，结合Unity、ROS和PR2机器人，实现了与真实环境几乎相同的交互模型；③通过定量分析（速度、加速度、路径偏差、头部朝向相似度）证明VR能够保留关键的社会化导航特征。

**🔧 技术方法**

技术手段包括：Unity3D模拟环境 + Microsoft Rocketbox 人体化身；ROS与Unity通过ROS‑TCP‑Connector实现通信；HTC Vive Pro 2 + 手柄进行VR交互与头部姿态捕捉；Motion‑Capture 系统在真实实验中捕捉人类姿态；PR2移动机器人 + CoHAN 社会化导航规划器 + 视线追踪模型；数据分析使用统计相关、速度/加速度曲线、路径偏差和余弦距离等指标。

**📊 数据集**

数据集：21名受试者（6女15男，平均年龄27.7岁），完成 40 场导航回合（每个场景10回合），共收集 240 条人机交互轨迹。包含真实环境和VR环境下的轨迹、速度、加速度、头部朝向、问卷评估等数据。

**📈 对比分析**

比较方法：在同一任务（正交交叉、过路）下，采用 within‑subjects 设计，收集主观问卷（5 分制）与客观运动数据；对比 RW 与 VR 在速度、加速度、路径偏差、头部朝向相似度等指标。结果显示：①中位数问卷评分相近；②VR 与 RW 的速度、加速度与路径偏差在统计上基本无显著差异；③头部朝向与未来导航方向的余弦距离在两种环境中保持一致。总体而言，VR 能够重现真实环境中的多模态交互特征，性能相当。

**⚠️ 局限性**

局限性：①缺少声学反馈，导致受试者过度依赖视觉；②仅捕捉平面运动与头部姿态，未包含完整身体姿势与眼动；③VR 视野受限，影响对机器人的感知；④受试者群体单一（成年男性/女性），缺乏多样性；⑤VR 导航自然度低，可能影响行为真实性；⑥未在更复杂或动态障碍环境下验证。

---

## 229. All you need is SAMPAT

**arXiv ID:** 2607.09235 | [PDF](https://arxiv.org/pdf/2607.09235v1)

**作者:** Jayadeva `[一作]` (Indian Institute of Technology), Madhur Aswani `[通讯]` (Indian Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种三层可解释的 SAMPAT 网络，用对数、指数、线性激活构造可解析多项式逼近器，实现对光滑函数的全微分可解释性。

**💡 创新点**

创新点在于将对数–指数–线性组合转化为可解析的多项式表达式，并通过约束学习不同函数族（多项式、分式、Gauss 混合等），兼容复数权重，提升逼近灵活性。

**🔧 技术方法**

使用对数激活的第一层、指数激活的第二层以及可选线性/非线性激活的第三层，配合跳跃连接和复数权重进行端到端训练。

**📊 数据集**

实验数据涵盖合成函数（sin、ReLU）、多元基准函数、RLC 电路时序、UCI 传统数据集以及图像数据集（CIFAR‑10、Cats‑vs‑Dogs）。

**📈 对比分析**

与传统 MLP、SVM、ELM 及优化后的多层网络比较，SAMPAT 在参数量 7–8 倍减少的情况下保持或提升 MSE/R²；在图像分类任务中未预训练即可实现 90%+ 的准确率，表现与 DenseNet/MobileNet 相近。

**⚠️ 局限性**

局限性包括：对非光滑或稀疏特征的适应性有限；训练时复数权重需保持数值稳定；模型扩展到更深层时结构设计仍需手工调优，且在极高维度场景下可能面临可解释性与效率折衷。

---

## 230. Implicit-Behavior Coordination from Unlabeled Sub-Task Demonstrations for Rearrangement Tasks

**arXiv ID:** 2607.09234 | [PDF](https://arxiv.org/pdf/2607.09234v1)

**作者:** Ahmed Shokry `[一作]` (University of Bonn), Maren Bennewitz `[通讯]` (University of Bonn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种利用无标签子任务演示，通过条件流匹配生成多模态动作候选并由价值网络挑选实现长周期搬运的隐式行为协调方法。

**💡 创新点**

无需显式技能标签、规划或切换逻辑，直接从混合子任务数据学习多模态行为并通过价值引导实现自适应协调。

**🔧 技术方法**

使用条件流匹配生成器、强化学习价值网络、不确定性加权价值选择以及基于视觉与状态的多模态输入。

**📊 数据集**

Habitat 2.0 + ReplicaCAD训练集、Fetch机器人环境以及UR3e桌面平台收集的混合子任务演示。

**📈 对比分析**

与显式技能序列基线（Skill Transformer、Oracle Planner+RL/BC）对比，隐式方法在Nav‑Pick‑Nav‑Place与Nav‑Open‑Pick‑Nav‑Place任务中表现更优，在行为库扩展和链式目标场景下仍保持较高成功率。

**⚠️ 局限性**

对子任务演示的充分覆盖与重叠依赖度高，稀疏奖励导致价值传播受限；在真实场景中缺乏目标位置信息，需要重新训练价值网络。

---

## 231. Tactile and Vision Conditioned Contact-Centric Control for Whole-Arm Manipulation

**arXiv ID:** 2607.09218 | [PDF](https://arxiv.org/pdf/2607.09218v1)

**作者:** Rishabh Madan `[一作]` (Cornell University), Tapomayukh Bhattacharjee `[通讯]` (Cornell University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于多模态感知（视觉、分布式触觉、接近图）和运动学的整臂操控控制框架TACTIC，利用采样式MPC和混合预测模型实现多接触轨迹规划与力调节。

**💡 创新点**

创新点在于①构建接触中心的隐状态表示，融合触觉与接近图；②在MPC采样中引入接触雅可比矩阵进行力调节导向的动作采样；③结合分析运动学与学习的潜在动力学，形成混合预测以兼顾物理一致性与复杂接触演化。

**🔧 技术方法**

核心技术包括基于DINOv3的多模态编码器、ViT潜在动力学预测、MPPI/ STORM采样、接触雅可比矩阵和力约束投影，以及两阶逼近预测与增量式价值估计。

**📊 数据集**

使用Kinova Gen3 7-DoF机器人收集的分布式触觉+视觉数据，结合RCareWorld、Robohive、FleX等公开仿真环境以及真实人形模型（manikin）场景进行训练与评估。

**📈 对比分析**

通过与DreamerV3、TD-MPC2、Diffusion Policy等基线对比，TACTIC在仿真（Maze、Bathing、Reach、Granular）和真实世界（迷宫、侧翻、四肢复位）任务中均取得更高成功率、更低力阈值违规和更平滑的动作，显示出显著的性能提升。

**⚠️ 局限性**

主要限制包括缺乏形式化安全保证、对高质量触觉数据的依赖、在极端动态环境下的实时性挑战，以及在更复杂人机交互场景中可扩展性的未知。

---

## 232. Strong Refutation of Random Ordering CSPs

**arXiv ID:** 2607.09410 | [PDF](https://arxiv.org/pdf/2607.09410v1)

**作者:** Xifan Yu `[一作]` `[通讯]` (Yale University), Xifan Yu (Yale University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对随机排序约束满足问题（OCSP）在平均-场设定下的强驳斥（refutation）问题进行理论分析，并提出了两种新算法实现强驳斥，分别为秩分解法和分桶法。

**💡 创新点**

创新点在于将OCSP映射到有限字母CSP、提出秩分解（rank decomposition）将原约束转化为只依赖有限坐标的低维函数、并利用Quiet Planting构造下界，证明在满足Low Degree Conjecture的前提下算法难度与坐标度呈指数关系。

**🔧 技术方法**

采用了秩分解、Kikuchi图谱与Kikuchi bounds、矩阵 Bernstein 与 McDiarmid 不等式、Efron‑Stein 分解、低坐标度优势（Low Coordinate Degree Advantage）等理论工具。

**📊 数据集**

无具体数据集，所有结果均为理论上限与下界，主要基于随机OCSP实例的概率分布。

**📈 对比分析**

与现有基于随机CSP强驳斥方法对比，分桶法在驱散阈值为常数强度时达到近最优性能；秩分解法提供了更细粒度的阈值与时间复杂度权衡，算法时间可为多项式至指数级，具体表现取决于坐标度与分桶数量。

**⚠️ 局限性**

限制在于下界依赖于Generalized Low Degree Conjecture，对其假设不成立的情形下证明不完整；同时，秩分解与分桶方法仅适用于坐标度较小且约束函数满足特定结构的OCSP，无法直接推广到所有排序约束或高坐标度情况。

---

## 233. SYNRARE: Synthetic Rare Disease EHR Generation for ML Benchmarking

**arXiv ID:** 2607.09404 | [PDF](https://arxiv.org/pdf/2607.09404v1)

**作者:** Nicolai Dinh Khang Truong `[一作]` (University of Southern Denmark), Richard Röttger `[通讯]` (University of Southern Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了SYNRARE，一个基于Synthea的图形化界面，用于快速生成与常见疾病差异可定义的罕见疾病电子健康记录（EHR），以支持机器学习（ML）模型的基准测试。

**💡 创新点**

①提供无代码、可视化的模块修改界面，用户可一次性全局调整病程状态分布；②引入BMI驱动的并发症建模，实现体重指数对健康影响的可调节；③支持旧版模块迁移，降低使用门槛；④通过示例展示如何生成罕见变种疾病并进行ML评估。

**🔧 技术方法**

使用Python 3.8+、Java JDK 11、Synthea规则式合成数据生成框架；构建了GUI以可视化查看与修改模块分布，并实现随机全局修改功能；示例中使用了PCA可视化和常规ML分类模型（如逻辑回归、黑盒模型）进行评估。

**📊 数据集**

完全合成的Synthea模块数据，例如支气管炎、囊性纤维化等常见疾病模块，基于这些模块生成的罕见疾病变体数据；未使用真实患者数据。

**📈 对比分析**

通过生成罕见疾病变体数据后，利用PCA对原始与变体进行可视化，随后用逻辑回归等分类器测试二分类性能，发现逻辑回归难以区分两类，提示需采用更复杂模型；但论文未给出具体数值指标。

**⚠️ 局限性**

①合成数据的“理想化”轨迹可能无法完全反映真实EHR的缺失、噪声和混杂信息；②需用户具备足够的临床知识以确保生成数据的现实性；③基于规则的生成方式可能导致与真实数据分布的偏差；④该工具主要面向技术评估，非用于疾病理解。

---

## 234. STEEL: Sparsity-Aware Fused Attention for Energy-Efficient Long-Sequence Inference on AMD's XDNA NPU

**arXiv ID:** 2607.09385 | [PDF](https://arxiv.org/pdf/2607.09385v1)

**作者:** Victor J. B. Jung `[一作]`, Luca Benini `[通讯]` (ETH Zürich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并开源了 STEEL，一套针对 AMD NPU 的 FlashAttention 2 数据流实现，用于高效推理长序列的大语言模型。

**💡 创新点**

创新点包括：①将 FlashAttention 细化为三阶段流水线，匹配 AIE 核的 VLIW 架构；②提出稀疏感知的流水线放置策略，缓解因因果遮罩导致的负载不均衡；③通过融合操作避免中间张量落地，显著降低数据传输和能耗；④实现了跨 AMD NPU 代际的可移植性。

**🔧 技术方法**

技术栈：AMD 的 AIE/IRON 开源软件（Python+MLIR-AIE+C++），Mem/DMA 通道实现 4D 传输，VLIW 指令并行，向量化乘累积，指数查表，分块矩阵乘法，在线 softmax，稀疏掩码生成。

**📊 数据集**

使用了 Llama‑3.1‑1B（32 个 head，head_dim 64）和 BERT 的 attention 配置（12 个 head，head_dim 64），在 2048–32768 令牌范围内评估；对比 DATO 在 4096 令牌的实验。

**📈 对比分析**

比较方法：在同一 SoC 上对比 NPU、12 核 Zen5 CPU 与 RDNA 3.5 GPU；对比基线的逐层实现、DATO 的 FlashAttention；使用 TorchLib/ROC‑m/HIP 作为 CPU/GPU 基线；测量延迟、能耗和 off‑chip 传输量。性能结果：与 CPU 对比能耗下降 9.17×，与 GPU 下降 1.75×；与 DATO 延迟平均下降 9.6×；与逐层实现平均加速 22.8×；稀疏放置比均匀放置快 38%。

**⚠️ 局限性**

局限性：①仅在 AMD 的 XDNA‑类 NPU 上验证，未针对其它厂商的 NPU 做通用化；②对因果遮罩的优化特化，可能对非因果或窗口注意力效果有限；③编译和部署依赖复杂的 IRON/MLIR-AIE 堆栈，门槛较高；④在极端长序列或超大模型时，硬件资源（如 Mem 端口）仍可能成为瓶颈。

---

## 235. CtrlVTON: Controllable Virtual Try-On via Visual-Instance-Prompt Segmentation

**arXiv ID:** 2607.09362 | [PDF](https://arxiv.org/pdf/2607.09362v1)

**作者:** Seungyong Lee `[一作]` (NXN Labs), Sungjoon Park `[通讯]` (NXN Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对虚拟试衣场景，作者提出了 VIP-Seg（VIP‑SAM）实现服装实例级分割，并基于此构建了 CtrlVTON，一个可通过像素级掩码实现风格、尺寸、位置可控的编辑式试衣框架。

**💡 创新点**

创新点在于：①将服装实例分割任务从类别级提升到实例级（VIP‑SAM），通过在 SAM 编码器早期注入参考特征实现对同类服装的精准分割；②把试衣问题重新建模为图像编辑而非掩码填充，既避免了 inpainting 的身份漂移，又保留了对目标服装的精准定位；③在 CtrlVTON 中加入轻量级 LoRA 的掩码条件注入，实现对单件或多件服装的颜色编码、分层和局部切换控制。

**🔧 技术方法**

主要技术包括：SAM/SAM‑2/ViT‑SAM 的图像分割框架、VIP‑SAM 的跨模态特征注入与跨注意力模块、Diffusion Transformer（DiT）编辑模型、LoRA 模块、VLM‑as‑judge（Gemini 3.0 Flash）评估、以及用于合成训练三元组的人体‑服装‑合成对齐流程。

**📊 数据集**

使用的数据集有：VITON‑HD、OmniTry Bench、DressCode‑MR、Garments2Look、VITON‑HD‑edit（公开的三元组+掩码基准），以及在数据准备管线中合成的 VITON‑HD‑edit 的 2032 张实例。

**📈 对比分析**

在单件、多人件、任务标记和掩码可控的四个评价维度下，CtrlVTON 在 M‑DINO、M‑CLIP、GTC、PBC、PR 等指标上均优于现有的 inpainting‑和 editing‑基础试衣模型；在掩码对齐方面，CtrlVTON 的 IoU、Hu 距离和 Hausdorff 距离远优于四大商用编辑器，说明其像素级控制更精准。

**⚠️ 局限性**

局限性包括：①缺乏公开的多件服装掩码可控基准，仅在 VITON‑HD‑edit 进行定量评估；②训练过程依赖大量合成三元组，合成质量可能影响模型泛化；③对极端姿态或极稀缺服装类别的鲁棒性尚未充分验证；④模型规模较大，推理成本相对较高。

---

## 236. Spanning Paths and Cycles: Structural Limitations of the Irrelevant Vertex Technique

**arXiv ID:** 2607.09342 | [PDF](https://arxiv.org/pdf/2607.09342v1)

**作者:** Dimitrios M. Thilikos `[一作]` (LIRMM, Université de Montpellier, CNRS), Sebastian Wiederrecht `[通讯]` (School of Computing, KAIST)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了“无关顶点技术”(Irrelevant Vertex Technique)在跨度路由(Spanning Routing)问题中的结构限制，并提出了新的结构参数 depth_2，证明了一个完整的组合学二分法：在任何红色-子图封闭类中，跨度离散路径问题(SDP)具有无关顶点属性，当且仅当该类的 depth_2 有界。

**💡 创新点**

创新点在于首次将 depth_2 作为划分无关顶点技术适用与否的精确阈值；通过局部结构定理与跨度 Vital Linkage 定理的结合，给出了 SDP 的 2^{2^{k+d}} n^2 上界，并给出了匹配的下界，展示了无关顶点规则在 depth_2 无界时甚至在平面图中不成立。

**🔧 技术方法**

采用了图最小化理论的结构定理(Almost Embedding、Vortex 处理)、局部结构定理、跨度 Vital Linkage 函数、Annulus Combing Lemma、火炬 (candle) 分析、红色子图与顶点分离、双维度性( bidimensionality) 等技术；同时引入了新的“深度 2”参数以及“浅渗透”(shallow) 表现来控制红点在 vortex 内的分布。

**📊 数据集**

本研究为理论算法论文，无需使用实验数据集；所有结论均基于纯粹的图论结构和算法证明。

**📈 对比分析**

算法在参数 k（终点对数）和 d（depth_2）上呈指数级时间 2^{2^{k+d}} n^2，匹配的下界表明该复杂度是最佳的；对无关顶点技术的适用性进行了严格的结构与时间复杂度分析，而非经验对比。

**⚠️ 局限性**

局限性在于：算法仅适用于 depth_2 有界的类；在 depth_2 无界时，无法提供无关顶点规则，甚至在平面图中也不成立；此外，论文未给出在 depth_2 有界之外是否存在多项式时间解或 FPT 方案，问题在更一般的情形下仍未解决。

---

## 237. WILDTRACE: Benchmarking Natural Evidence Trails in Long-Context Reasoning

**arXiv ID:** 2607.09328 | [PDF](https://arxiv.org/pdf/2607.09328v1)

**作者:** Zixin Chen `[一作]` (Hong Kong University of Science and Technology), Huamin Qu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个面向长文档的多跳推理基准，专注于自然散布在文档内部的证据链；

**💡 创新点**

创新点在于：①源先生成（source‑first）方式挖掘文档内部的证据轨迹；②提出七种证据几何（链式、交叉、比较、时间、因果、溯因、反事实）来刻画推理需求；③设计多阶段有效性门控（验证、必要性、短路防护、答案可行性）以确保任务真实性；

**🔧 技术方法**

使用的技术包括：文档分段、LLM辅助信息抽取、结构化图构建、几何查询挖掘、人工与外部审核、全文提示与基于标准化 Rubric 的多级评分；

**📊 数据集**

数据集来源于 214 篇公开长文档（技术事故报告、文学叙事与中文文学），生成 481 个验证通过的任务；

**📈 对比分析**

通过对 18 个前沿模型在“完整文档+问题”条件下的 evidence‑withheld 评估，采用多评判者的 Rubric 打分；最佳模型平均得分 75.3%，显示与文档长度、几何类型相关的显著性能差异，尤其在反事实几何上表现最弱；

**⚠️ 局限性**

局限性包括：数据主要聚焦英文技术报告与文学叙事，中文样本不足；只覆盖单一几何的任务，未考虑多几何组合；未评估检索增强、迭代或多次调用的系统；

---

## 238. Differential Analysis of Multispectral Images for Terrain Identification

**arXiv ID:** 2607.09319 | [PDF](https://arxiv.org/pdf/2607.09319v1)

**作者:** Omar Kashmar `[一作]` (University of Genoa), Fulvio Mastrogiovanni `[通讯]` (University of Genoa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种轻量级多光谱地形识别框架，通过双流残差网络提取原始波段和波段比值特征，并使用差分融合提升鲁棒性。

**💡 创新点**

创新点在于：①利用波段比值抑制多重光照/传感器增益的乘法效应；②双流差分融合机制显式捕捉绝对波段与比值特征的差异；③实现兼容边缘计算的轻量级设计。

**🔧 技术方法**

采用残差网络、双流结构、波段比值计算、差分融合分支、轻量化全连接分类头以及可选的对比损失；使用Grad‑CAM做解释性分析。

**📊 数据集**

在使用 MicaSense RedEdge‑P 六波段多光谱相机收集的油渍土壤数据集（约 400 张图像）上进行实验，并在受控水滴-草地场景下分析 NIR 反射变化。

**📈 对比分析**

与四种基线比较：raw‑only、ratio‑only、concat‑fusion、以及提出的 DRIFT。实验显示 DRIFT 的准确率最高（94.50%），F1 分数为 0.93，显著优于单流基线和简单拼接模型。

**⚠️ 局限性**

局限性包括：数据集规模有限、仅覆盖单一土壤/油种/传感器配置；缺乏跨环境、不同光照与噪声的系统性鲁棒性评估；未针对大规模部署进行多设备/多平台验证。

---

## 239. Parameter-Efficient Vision-Language Adaptation with Continuous Metadata Conditioning for Animal Re-Identification

**arXiv ID:** 2607.09443 | [PDF](https://arxiv.org/pdf/2607.09443v1)

**作者:** Anil Osman Tur `[一作]` (University of Verona), Cigdem Beyan `[通讯]` (University of Verona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种参数高效的CLIP适配框架，利用连续元数据调制提示词，针对动物长期再识别问题进行学习；

**💡 创新点**

主要创新在于：①将连续数值元数据直接嵌入提示词中，而非离散化为文本标签；②实现训练时仅使用元数据，推理阶段完全无元数据依赖；③统一端到端的低秩视觉适配、提示学习和跨模态对齐，避免多阶段训练；

**🔧 技术方法**

使用CLIP ViT-B/16冻结视觉/文本编码器，低秩注意力适配（LoRA）、可学习提示上下文、sinusoidal或FiLM元数据调制、双向跨模态对齐、三元组损失与辅助分类；

**📊 数据集**

主要在七年持续收集的corkwing wrasse（Melops）鱼类数据集上进行实验，并在SeaTurtleID2022等多动物ReID基准上验证；

**📈 对比分析**

与CLIP-ReID、IndivAID等现有方法对比，在闭集、开放集及时间感知评估中均取得更高的mAP和Rank‑1/5分数；在训练参数量仅为全微调的一半左右；

**⚠️ 局限性**

仍受限于元数据可用性与分辨率；元数据调制需在训练阶段预先可得，推理阶段无法利用；对不同物种的通用性需进一步验证；

---

## 240. Test-Time Scaling for Small VLMs on Multilingual Visual MCQ

**arXiv ID:** 2607.09438 | [PDF](https://arxiv.org/pdf/2607.09438v1)

**作者:** Spiros Baxevanakis `[一作]` (University of Amsterdam), Peng-Jian Yang `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多语言视觉多选题（EXAMS‑V）上评估并对比测试时扩展（TTS）技术对小型开源视觉‑语言模型的性能影响，研究自一致性、结构化搜索以及后置选择器三种推理策略。

**💡 创新点**

发现解析器格式和每条链的解码预算是决定 TTS 效果的关键因素；结构化搜索在此任务与资源受限条件下并不优于平行采样；通过指南修复与 2,048 token 预算的组合，首次在 ImageCLEF 2026 Visual MCQ 评测中取得榜首。

**🔧 技术方法**

使用 Qwen2.5‑VL‑7B‑Instruct 与 Qwen3.5‑4B 两款开源 VLM；自一致性、describe‑then‑reason + PRM‑guided beam search；训练自由生成批评家、判别式过程奖励模型（PRM）；guided parse repair；温度、链数、token 预算等多维 TTS 探索。

**📊 数据集**

数据集：EXAMS‑V（13 语种、20 学科、4,651 题验证集；1,117 题测试集）以及 ImageCLEF 2026 官方测试集。

**📈 对比分析**

在验证集上，最佳配置（Qwen3.5‑4B、SC‑N=16、2,048‑token、guided repair）达到 84.06%（测试集 84.1%），在 Visual MCQ 领导榜首；与零射击/Chain‑of‑Thought 的提升超过 24pp；单链 token 预算提升 3.7pp，链数提升仅 0.15pp，结构化搜索在算力约 8.7× 的情况下甚至低于自一致性。

**⚠️ 局限性**

局限性：仅评估两款同族模型，缺乏跨家族的泛化；解析器与解码预算的影响可能因模型架构不同而变化；部分消融基于 200 题子集，未覆盖完整验证；PRM 在多语言非数学内容下表现不佳；未使用微调、few‑shot 或图像增强等常见提升手段。

---

## 241. How Do Software Professionals Evaluate AI-Generated Code? (Registered Report)

**arXiv ID:** 2607.09434 | [PDF](https://arxiv.org/pdf/2607.09434v1)

**作者:** Samuli Määttä `[一作]` (University of Oulu), Markus Kelanti `[通讯]` (University of Oulu)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并进行了一项基于构造主义扎根理论的研究，探讨软件专业人员如何评估AI生成代码。

**💡 创新点**

创新点在于结合梯度提问法（laddering）与半结构化访谈，以价值链的方式揭示评估实践背后的动机与价值体系，并系统构建了评估实践的理论框架。

**🔧 技术方法**

使用的技术包括构造主义扎根理论分析、梯度提问访谈、半结构化访谈、NVivo/Quirkos等定性分析工具。

**📊 数据集**

数据来源为芬兰软件专业人员的问卷调查（163份）以及后续的20–50名受访者的访谈记录。

**📈 对比分析**

本研究未进行算法或系统性能比较，重点在于理论生成与实践洞见，因而没有性能评估。

**⚠️ 局限性**

局限性包括样本局限于芬兰，访谈样本规模有限，研究结果的可推广性受限，并且高度依赖研究者的解释与主观判断。

---

## 242. Similarity search generalisation in contrastive learning with InfoNCE loss

**arXiv ID:** 2607.09405 | [PDF](https://arxiv.org/pdf/2607.09405v1)

**作者:** Nick Whiteley `[一作]` `[通讯]` (University of Bristol), Nick Whiteley (University of Bristol)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文从理论角度研究了InfoNCE损失在相似度检索中的泛化性质，推导了与交叉熵、互信息以及对齐与均匀性等解读的联系；提出了针对InfoNCE的连续性上界并利用Gâteaux导数构造，进而给出了Rademacher复杂度的上界，说明负样本数量k对泛化误差的平均化效应；进一步将这些理论结果综合成相似度检索的整体泛化界；并讨论了早停、神经网络等应用场景。

**💡 创新点**

创新点主要有：①提出InfoNCE风险与软最大相似度检索的交叉熵之间的O(1/k)收敛关系；②构造了InfoNCE损失的Gâteaux连续性上界，捕捉k负样本平均结构；③基于此连续性上界得到Rademacher复杂度上界，揭示k、温度τ、样本数n对泛化误差的精细影响；④将上述理论整合成完整的相似度检索泛化界。

**🔧 技术方法**

使用了：概率论与测度论（Markov核、Radon–Nikodym导数）；信息论（互信息、交叉熵）；泛化理论工具（Rademacher复杂度、Dudley积分、覆盖数）；Gâteaux导数与连续性分析；对负样本平均结构的功率均值（power‑mean）技巧。

**📊 数据集**

未在论文中使用任何公开数据集，全部为理论推导与数理证明，未进行实验验证。

**📈 对比分析**

未给出实验或方法比较，因本文属于纯理论分析。

**⚠️ 局限性**

局限性包括：①理论上依赖于i.i.d.假设和负样本平均结构，实际训练中负样本分布可能不同；②对神经网络的Rademacher复杂度上界仍包含显式维度/参数量依赖，可能在宽深网络时失效；③未考虑自适应温度或非均匀负样本采样；④缺乏实验验证，无法评估理论界限与实际性能的贴合度。

---

## 243. System Capybara: Tracking Capabilities for Separation and Freshness (Extended Version)

**arXiv ID:** 2607.09383 | [PDF](https://arxiv.org/pdf/2607.09383v1)

**作者:** Yichen Xu `[一作]` (EPFL), Martin Odersky `[通讯]` (EPFL)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 System ，扩展 Scala 的捕获检查器，使其支持可选择的子结构别名控制（分离、只读、消费、新鲜度），从而在保持 Scala 共享默认行为的同时提供 Rust 风格的资源安全与无数据竞争并发。

**💡 创新点**

创新点在于：① 将捕获检查与子结构别名控制相结合，提供局部别名管理；② 设计了可分离、只读、消费、新鲜度的捕获集合语义；③ 引入模态类型[Ψ,Φ]实现分离检查；④ 通过类型保留翻译与 Lean 4 逻辑关系证明实现语义安全性。

**🔧 技术方法**

技术包括：捕获类型（capture checking）、线性/唯一性类型概念、模态类型与锁/解锁机制、类型保留翻译、基于世界的逻辑关系模型、Lean 4 机械化证明。

**📊 数据集**

该工作为理论系统，无需实验数据集；评估基于形式化证明和在 Scala 3 编译器中的实现验证。

**📈 对比分析**

方法对比：通过证明系统的语义安全性、类型安全性、内存安全性、不可变性和无数据竞争性；没有基准性能测评，但实现已集成到 Scala 3 编译器，证明在大规模代码库中非侵入且可行。

**⚠️ 局限性**

局限性：只能对可在调用点分离的根进行分离检查，无法直接消费共享结构内部子对象；不支持循环引用；路径依赖能力仅有限；对多线程外部同步（如互斥锁）仍需外部方案；实现仍在 Scala 3 生态中，尚未针对完整语言特性（类、异常、继承等）进行全面验证。

---

## 244. Learning Physics-Informed Surrogate Model of Linear Elastic Displacement Fields from Geometry

**arXiv ID:** 2607.09382 | [PDF](https://arxiv.org/pdf/2607.09382v1)

**作者:** Rodolphe Barlogis `[一作]` (Université Perpignan Via Domitia), Stéphane Grieu `[通讯]` (Université de Toulouse)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

使用物理信息的DeepONet学习弹性域中包含裂纹的位移场，能够实时预测几何形状和边界条件的影响。

**💡 创新点**

创新点在于将裂纹几何直接作为输入通过距离函数编码，并在无FEM训练数据的情况下实现物理一致的推断；同时提出弱约束和柔性材料两种实现方案。

**🔧 技术方法**

使用DeepONet、自动微分、签名距离函数（SDF）、物理信息损失、弱约束或柔性材料方法。

**📊 数据集**

使用无训练数据，直接利用物理约束学习；测试以单一二维圆孔拉伸问题为例。

**📈 对比分析**

与FreeFem++有限元结果对比，误差仅在几位小数内；在单一几何下实现收敛，性能相对较好。

**⚠️ 局限性**

仅验证单一圆孔几何，缺乏多形状推广；平滑处理导致对裂纹尖端应力集中描述不足；模型对复杂几何和三维情况的泛化能力待验证。

---

## 245. Letting the Data Speak: Extracting Keywords from Crowdsourced Collections with AI

**arXiv ID:** 2607.09324 | [PDF](https://arxiv.org/pdf/2607.09324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 246. DKCD: Domain Knowledge-Enhanced Causal Discovery from Unstructured Data

**arXiv ID:** 2607.09348 | [PDF](https://arxiv.org/pdf/2607.09348v1)

**作者:** Xin Li `[一作]` (University of Technology Sydney), Fang Chen `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 DKCD 框架，利用领域知识图结合 LLM 对非结构化文本进行因果因素挖掘、隐因果推理和因果图生成；

**💡 创新点**

创新点在于三层协同：知识挖掘检索语义相关 KG 子图，知识引导因果推理既发现隐含因果因素又生成因果线索，最终提升因果结构完整性与准确性；

**🔧 技术方法**

技术包括：大型语言模型（GPT‑4o、Gemini 等）+ 语义匹配（all‑MiniLM‑L6‑v2）+ KG 检索+ 生成式因果推理+ 统计因果发现算法 FCI；

**📊 数据集**

使用两个医疗领域合成数据集：糖尿病病情描述集与呼吸疾病病情描述集，均基于真实医学知识图构建；

**📈 对比分析**

与 Zero‑shot LLM、COAT、META 等基线比较，DKCD 在节点精确率、召回率、F1、邻接精确率/召回率/AF 以及 ESHD 指标上均显著优于基线，说明因果因素和图结构更完整、准确；

**⚠️ 局限性**

局限性包括：仍高度依赖 LLM 生成，易出现幻觉和偏差；合成数据集规模有限，缺乏真实公开基准；跨域泛化（如教育、金融等）尚未验证。

---

## 247. How Far Are We from Detecting Flaky Tests? On the Limits of Code-Based Detection

**arXiv ID:** 2607.09345 | [PDF](https://arxiv.org/pdf/2607.09345v1)

**作者:** Ömer Oktay Gültekin `[一作]` (Heidelberg University), Sebastian Baltes `[通讯]` (Heidelberg University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种基于 LLM 的 flaky test 检测器（Flakify、FlakyQ、FlakyXBert）进行复现，识别并修正实验缺陷；构建两套新的数据集（Δ：54468 单元测试，采用 500 次重复执行获得非 flaky 标注；FlakyBench 类别化的 flaky 测试）并评估模型性能；挖掘 CI 日志得到 86 条 flaky E2E 测试并进行根因分析；最终提出将 flaky 检测问题从“测试代码是否 flaky”重新表述为“观察到的失败是否 flaky/在特定运行环境下的失败概率”。

**💡 创新点**

揭示高 F1 分数主要来自数据泄露、标签捷径（fix‑commit）和项目重叠；证明在更严格的评估（项目分离、常量基线）下，LLM 也无法显著提升检测效果；提供可复现的数据集与评估框架，推动研究关注真正的泛化能力；为后续研究提供 failure‑level 检测与执行环境感知的方向。

**🔧 技术方法**

技术手段：CodeBERT 及其变体（fine‑tune、quantization、few‑shot Siamese）；在 macOS M3 Ultra 上使用 Metal Performance Shaders 进行实验；数据预处理去除 smell‑based 空方法体，保持完整测试代码。

**📊 数据集**

数据集：
1) Δ：57 个 GitHub 项目，共 54468 个单元测试，developer‑confirmed flaky，non‑flaky 通过 500 次重复执行获得。
2) FlakyBench：280 个 flaky（按根因分类）+ 8294 个 non‑flaky（rerun‑confirmed）
3) CI‑log mining：86 条 E2E flaky 测试（74 Cypress + 12 Playwright），包含对应的通过/失败运行日志。

**📈 对比分析**

比较方法：
- 10‑fold 交叉验证（CV）
- 项目分离（project‑disjoint）
- 常量基线（always‑flaky、always‑non‑flaky）
- 统计指标：flaky‑class precision/recall/F1、weighted F1。结果显示：在项目分离下，CodeBERT 变体的 flaky‑class F1 仅为 0.035–0.07，几乎等同于 always‑non‑flaky 基线；在 CV 方案下表现优异（F1 > 0.9），说明高分主要来自数据/实验捷径。

**⚠️ 局限性**

局限性：
- 仅覆盖 JVM 单元测试，难以推广到其他语言/生态。
- 非 flaky 标注依赖 500 次重复执行，仍可能漏检低概率 flaky。
- 根因分析由单一评审员完成，可能存在主观偏差。
- CI 日志样本规模小（86 条），并仅来自 GitHub Actions，缺乏其他 CI 环境的代表性。
- 未考虑跨项目、跨语言、跨框架的泛化性能。

---

## 248. Shortcut Trajectory Planning for Efficient Offline Reinforcement Learning

**arXiv ID:** 2607.09336 | [PDF](https://arxiv.org/pdf/2607.09336v1)

**作者:** Guanquan Wang `[一作]` (University of Tokyo), Yoshimasa Tsuruoka `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Shortcut Trajectory Planning (STP)，一种使用 shortcut 模型的离线模型基强化学习轨迹规划器，替代传统教师‑学生蒸馏过程，简化训练并支持可调步数推理。

**💡 创新点**

通过在单阶段训练中直接学习条件 shortcut 模型，消除两阶段蒸馏，加入 warm‑start 与 feasibility‑aware plan selection，提升规划效率与鲁棒性。

**🔧 技术方法**

基于 shortcut 模型（flow‑matching + 自洽约束）、逆动力学模型、价值评估网络，结合可调步长采样与温启动，适用于离线 RL。

**📊 数据集**

在 D4RL 离线基准上评估，包括 Gym MuJoCo 运动任务（HalfCheetah、Hopper、Walker2d）、Maze2D、AntMaze、Kitchen 与 Adroit 等。

**📈 对比分析**

与 Diffusion‑QL、Diffuser、Decision Diffuser、Consistency Actor‑Critic、CP、CTP、RACTD、LEQ 等方法对比，STP 在大多数任务上均超过或与 CTP 持平，平均得分略优，尤其在长周期导航与高维操作任务上表现突出。

**⚠️ 局限性**

对环境可行性仍依赖外部罚项，且在极端稀疏奖励或极大时间尺度任务中，shortcut 模型的有限步长可能导致规划误差；对超大步长的收敛性和泛化仍待进一步验证。

---

## 249. Practical Source Code Recovery from Binary Functions Using Anchor-Based Retrieval and LLM Reasoning

**arXiv ID:** 2607.09452 | [PDF](https://arxiv.org/pdf/2607.09452v1)

**作者:** Charles Edward Gagnon `[一作]` (McGill University), Benjamin C. M. Fung `[通讯]` (McGill University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该工作提出一种将逆向工程、锚点检索与大语言模型（LLM）推理相结合的流水线，用于将被剥离的二进制函数恢复为源代码。

**💡 创新点**

创新点在于：①通过提取字符串、常量、外部调用等锚点进行初步检索；②使用LLM对候选源码进行深度比较并重新排序；③支持迭代传播，已匹配函数可作为后续函数的锚点，从而提升整体恢复效果。

**🔧 技术方法**

使用的主要技术包括：Ghidra 静态分析提取锚点与稠密特征；倒排索引检索源代码文件；GPT‑5.5（实验）或 GPT‑5 mini（GitHub 试验）对候选函数进行验证和重排序；以及对检索到的代码片段进行函数边界定位。

**📊 数据集**

实验数据集为：①自建的 8,061 个开源项目数据库（受控环境）；②公开的 GitHub 代码搜索接口（真实环境）。

**📈 对比分析**

性能比较：在受控数据库上 Hit@1 0.8564、Hit@3 0.8761、平均 MRR 0.8660，指令覆盖率 95.2%；在 GitHub 上平均 Hit@1 0.283、覆盖率 35.5%，主要瓶颈是数据库检索缺失导致的高 Miss 率，LLM 重新排序虽提升了 Hit@1 约 60% 的概率，但无法弥补缺失候选。

**⚠️ 局限性**

局限性：①检索阶段依赖于数据库质量，噪声大、重复仓库多时易产生 Miss；②锚点稀疏（如无字符串、常量）的函数难以检索；③去编译的伪代码与源代码可能存在细微差异，导致 LLM 误判；④对高度优化或加密/信号处理函数的恢复效果有限。

---

## 250. Robustifying Vision-Language Models via Test-Time Prompt Adaptation

**arXiv ID:** 2607.09450 | [PDF](https://arxiv.org/pdf/2607.09450v1)

**作者:** Xingyu Zhu `[一作]` (National University of Singapore), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在测试阶段通过分布级的 Prompt 适配来增强 Vision‑Language 模型的鲁棒性，利用增强视图形成的视觉分布与文本原型分布之间的对齐纠正对抗扰动导致的语义错配。

**💡 创新点**

创新点：①把视觉特征和文本原型视为离散分布，采用 Optimal Transport（OT）进行全局分布对齐，抵御对抗噪声的局部扰动；②引入动态缓存机制，在线收集可信视图并与文本空间对齐，进一步提升分布对齐效果；③在不更新 CLIP 参数的前提下，利用仅一次测试时间调优即可实现鲁棒提升。

**🔧 技术方法**

使用的技术包括：CLIP 预训练模型、测试时 Prompt 调优、随机增强（裁剪、翻转等）、Optimal Transport（Sinkhorn 迭代）、Orthogonal Procrustes 对齐、动态缓存与置信度筛选、AdamW 优化器。

**📊 数据集**

实验数据集涵盖：Caltech101、DTD、EuroSAT、UCF101、Pets、Cars、Flowers、Aircraft 八个细粒度分类集；以及 ImageNet 及其四个变体（ImageNetV2、ImageNet‑Sketch、ImageNet‑A、ImageNet‑R）。

**📈 对比分析**

与 CLIP、TPT、R‑TPT、C‑TPT、MTA、Ensemble 以及 TeCoA、PMG、FARE 等基线进行对比。实验显示，方法在对抗攻击下平均提升约 45–50% 的鲁棒准确率，且对抗准确率比 R‑TPT 高 1.8–2.2%；在干净样本上保持或略高的准确率，整体表现优于现有测试时适配方案。

**⚠️ 局限性**

局限性：目前仅验证在图像分类任务，未扩展到生成式多模态任务；对抗攻击策略仍在演进，模型在极端扰动下可能仍出现失效；动态缓存与 OT 计算带来一定的运行时开销，需进一步优化。

---

## 251. How Does Bayesian Causal Discovery Fail? Characterising Structural Consequences in Linear Gaussian Networks under Latent Confounding

**arXiv ID:** 2607.09449 | [PDF](https://arxiv.org/pdf/2607.09449v1)

**作者:** Debargha Ghosh `[一作]` (Utrecht University), Anna Kononova `[通讯]` (Leiden University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了线性高斯因果结构在存在单一潜在混杂时贝叶斯结构学习后验分布的行为，并给出了导致伪边出现的临界相关性阈值及其样本量依赖性；

**💡 创新点**

提出了基于混杂导致的伪边与碰撞结构相互作用的两种后验失败模式（沉默失败与嘈杂失败），并给出其结构化理论解释；

**🔧 技术方法**

利用贝叶斯高斯等价（BGe）得分、局部分数比较与完全枚举后验计算；

**📊 数据集**

使用20个不同拓扑结构的5节点合成图，在每个结构下生成5,000个样本的线性高斯数据，并多次重采样；

**📈 对比分析**

通过对比无混杂和强混杂（α=0.8）下的后验熵、真边概率和伪边概率，展示在混杂超过阈值时后验显著退化，嘈杂失败导致熵翻倍，沉默失败仍保持高置信度但错误结构；

**⚠️ 局限性**

仅考虑单个高斯混杂、线性高斯模型、极小图以及完全后验枚举，未覆盖非高斯或离散数据、多个混杂、以及大规模图中的近似推断。

---

## 252. Multimodal Scenario Similarity Search for Autonomous Driving

**arXiv ID:** 2607.09428 | [PDF](https://arxiv.org/pdf/2607.09428v1)

**作者:** Tamás Matuszka `[一作]` (aiMotive), Balázs Szolár `[通讯]` (aiMotive)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种多模态自动驾驶场景检索框架，融合视觉（视频）与轨迹（运动）两种表示，并通过该框架对大规模驾驶数据进行相似场景检索；

**💡 创新点**

创新点在于①系统性比较视觉与轨迹两种模态的相似性表现；②提出了两种轨迹检索方法——基于轨迹匹配的 Exo‑Trajectory 与基于对比学习的 Transformer 轨迹嵌入 ScenarioFormer；③通过分级权重融合实现视觉与运动信息的互补提升；

**🔧 技术方法**

使用的技术包括 ViViT 与 Qwen3‑VL‑2B 视觉编码、Fréchet‑距离与 Hungarian 匹配的轨迹匹配、Transformer‑Encoder 轨迹编码、对比学习（InfoNCE）与数据增强、基于余弦相似度的检索及分数级融合；

**📊 数据集**

实验基准为扩展版 aiMotive Multimodal Dataset，包含约682条手工标注的动态驾驶场景（包括切入、转弯、行人横穿等七类）；

**📈 对比分析**

在 NDCG@10 与 Recall 指标上，视觉模型 Qwen3‑VL‑2B 单独实现0.621；轨迹模型 Exo‑Trajectory 与 ScenarioFormer 单独实现0.565；融合后最高 NDCG@10 达到0.671（Qwen3‑VL‑2B + Exo‑Trajectory），显著优于单模态；

**⚠️ 局限性**

局限性包括：①轨迹匹配方法计算量大，难以大规模实时部署；②对比学习对批量大小敏感，需要额外显存；③仅评估于 682 条样本的人工标注基准，规模有限；④未探索端到端联合视觉‑轨迹编码的全新多模态表示；⑤对特殊场景（如复杂多车道、极端天气）表现尚未验证。

---

## 253. A Sovereign, Open-Source Foundation Model for German and English

**arXiv ID:** 2607.09424 | [PDF](https://arxiv.org/pdf/2607.09424v1)

**作者:** The Soofi-Team `[一作]`, Max Lübbering `[通讯]` (DFKI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并发布了一款面向德语和英语的主权、开放源码Mixture‑of‑Experts（MoE）混合 Mamba‑Transformer 基础模型 Soofi S 30B‑A3B，训练了约27 万亿 token，并在 1 百万 token 上实现长上下文推理。

**💡 创新点**

创新点包括：① 在 Nemotron‑3 Nano 参考架构基础上加入 Mamba‑2 与 MoE，形成能在每个 token 仅激活约3 B 参数、KV‑cache 近乎恒定的高效长上下文网络；② 三阶段 Warmup‑Stable‑Decay 训练日程，先广泛多样化再逐步聚焦高质量、德语强化与长上下文；③ 完全可复现的数据与训练流程公开，提供每一源数据的 token 计数、epoch 乘数及混合细节，满足欧盟主权与透明度要求。

**🔧 技术方法**

核心技术：Mamba‑2 递归序列混合层、Group‑Query Attention、Granular MoE 及 6 层 KV‑cache；AdamW + Warmup‑Stable‑Decay 学习率调度；Megatron‑Bridge 训练框架；长上下文阶段采用 1 M token 长度桶化与 16‑rank context parallel；vLLM 推理评测。

**📊 数据集**

数据集：Nemotron‑CC（v1.0/v2.0/v2.1）及其 synthetic、translated 版、FinePDFs、Dolma、HPLT、German‑Commons、Genios、FineWiki、SwallowCode、Nemotron‑Pretraining‑Code、Nemotron‑Pretraining‑Math、UltraData‑Math、KletterMix、MultiSynt/MT 等，按质量层级和语言进行上采样与阶段分配。

**📈 对比分析**

对比方法：统一使用同一评测 harness、提示、few‑shot 设置，在英德多维度基准（代码、数学、知识、推理、阅读、德语专属）对标开源基础模型（Alia 40B、EuroLLM 22B、Apertus 70B、Olmo 3 32B、Qwen3.5 35B、Gemma 3 27B、Ministral 3 14B）以及同架构 Nemotron‑3 Nano。Soofi S 在多数英德聚合指标、代码、数学与德语专属基准上均超越或持平，且在 40K/256K 上下文下的聚合解码吞吐量比同规模密集模型高 9×、比 Qwen3.5‑35B‑A3B 高 1.9×，显示出显著的部署效率优势。

**⚠️ 局限性**

局限性：① 训练与评测中的 Minerva 数学基准因停止条件与 token 限制需修正；② 德语竞赛数学（Minerva MATH‑DE）与部分知识召回（NaturalQuestions）仍落后最强密集模型；③ 代码评测 LBPP 受污染检测影响，表现略逊于部分密集基线；④ 受数据许可限制，部分商业文本（如 Genios）无法完整公开，完整再构需依赖聚合统计。

---

## 254. Action-Factored Multi-Agent Reinforcement Learning for Scalable Quantum Device Tuning

**arXiv ID:** 2607.09422 | [PDF](https://arxiv.org/pdf/2607.09422v1)

**作者:** Edwin De Nicolo `[一作]` (University of Oxford), Natalia Ares `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套名为QADAPT的多智能体强化学习框架，用于大规模量子点阵列的自动调谐；

**💡 创新点**

创新点在于采用在线因式化的动作空间（门电压虚拟化）与参数共享的分布式演员-批评家架构，能够实现零样本跨尺寸扩展和并行执行；

**🔧 技术方法**

使用了Kalman滤波估计电容矩阵、CNN虚拟化模型、PPO强化学习、Dec-POMDP建模以及动作空间因式化等技术；

**📊 数据集**

主要数据集来自QArray仿真，涵盖2到8个量子点不同尺寸的电荷稳定性图像；

**📈 对比分析**

与随机搜索、Nelder–Mead、贝叶斯优化、L‑BFGS、DreamerV3等基线对比，QADAPT在所有规模下收敛率最高、测量步数最少，且呈现近线性缩放；

**⚠️ 局限性**

局限性包括对真实硬件的适应性尚未验证，可能受到参数漂移、滞后、门角色划分不严格以及对非稳态/孤立量子点的适用性影响。

---

## 255. Two Vocabularies, One Phenomenon: Metadata Bias in AI Evidence Synthesis on Fertility Decline

**arXiv ID:** 2607.09409 | [PDF](https://arxiv.org/pdf/2607.09409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 256. Fictional Worldbuilding: Multi-Agent LLM Collaboration with Hierarchical Context Compression and Iterative Review

**arXiv ID:** 2607.09403 | [PDF](https://arxiv.org/pdf/2607.09403v1)

**作者:** Jingbo Chen `[一作]` (National University of Defense Technology), Zhenyan Lu `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

AutoWorldBuilder 基于多代理 LLM 协作实现自动化世界构建，解决上下文爆炸、创造多样性与一致性冲突、自动质量检测等问题；

**💡 创新点**

主要创新点包括结构化概念网络与冲突检测、基于 DAG 的语义局部化混合批量调度、四层上下文压缩机制、迭代评审与专门审计器以及基于技能文件的可零代码扩展多代理架构；

**🔧 技术方法**

使用技术包括大型语言模型（GPT‑OSS 120B、DeepSeek v3.2）、FAISS 向量检索、混合批量调度算法、分层 token 预算压缩、迭代审计和多维评分系统、YAML 前置的技能文件动态加载；

**📊 数据集**

实验数据集为 20 个跨 5 主题（奇幻、科幻、后末日、都市、历史）世界构建案例；

**📈 对比分析**

通过两组实验对比，系统在两种 LLM 后端下 95% 成功率，概念数 56–103/世界，pass率 85.5%–99.2%，构建时间 18–31 分钟，压缩效率 90% 以上；

**⚠️ 局限性**

限制包括关系解析未激活、审计器问题检测率低、任务拆分失效案例、缺乏标准评估基准、单语言输入局限等。

---

## 257. Closing the Complexity Gap for Exact Domatic Number at Three and Four

**arXiv ID:** 2607.09442 | [PDF](https://arxiv.org/pdf/2607.09442v1)

**作者:** Holger Spakowski `[一作]` `[通讯]` (University of Cape Town), Holger Spakowski (University of Cape Town)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构造多种图形化归约，证明了固定整数k≥3的Exact‑k‑Domatic‑Number问题属于DP类的完全问题；

**💡 创新点**

创新点在于设计了两种SAT→Domatic‑Number归约，一种避免出现domatic数为3的输出图，另一种实现3- vs 2判定，从而填补了之前Riege和Rothe未解决的k=3、4两种情况；

**🔧 技术方法**

主要技术包括图形化构造（利用大团、锚点、变量辅助顶点、子句辅助顶点等）与鸽笼原理、颜色约束等多种逻辑编码；

**📊 数据集**

文中未使用实际数据集，研究完全属于理论复杂度分析；

**📈 对比分析**

由于研究目标是证明计算复杂度，未进行实验比较，评估结果表现为理论上的DP‑完备性证明；

**⚠️ 局限性**

局限性在于仅给出理论证明，未提供高效算法或实际实例分析，且方法主要针对固定k，未探讨可变k或近似算法。

---

## 258. Self-Guided Test-Time Training for Long-Context LLMs

**arXiv ID:** 2607.09415 | [PDF](https://arxiv.org/pdf/2607.09415v1)

**作者:** Xinyu Zhu `[一作]` (Meta AI), Xi Liu `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在长文本推理任务中，提出一种在测试时使用模型自身标记相关证据片段进行自适应训练的框架S-TTT。

**💡 创新点**

创新点是将训练数据的质量视为长文本T T T的关键瓶颈，并通过模型自我标注来选取最有价值的训练片段。

**🔧 技术方法**

技术主要包括基于LLM的span选择、LoRA参数高效自适应、next-token预测目标以及在原始完整上下文上生成答案。

**📊 数据集**

使用LongBench-v2与LongBench-Pro两个长文本推理基准数据集。

**📈 对比分析**

与基线、LongLLMLingua、qTTT、QRHead Span TTT、Random Span TTT和Full Context TTT等方法对比，S-TTT在两大基准上均能提升准确率，且在长文本区间表现尤为显著，同时推理延迟低于非冻结KV缓存的其它T T T方法。

**⚠️ 局限性**

局限性在于需先进行span选择，若模型未能正确标注会退回到随机采样；在极短文本或标注失败场景下效果受限；此外仍受限于LLM自身对问题的理解能力。

---

## 259. On-Device Adaptive Battery Power Prediction for Electric Vehicles

**arXiv ID:** 2607.09400 | [PDF](https://arxiv.org/pdf/2607.09400v1)

**作者:** Avik Bhatnagar `[一作]` (FZI Research Center for Information Technology, University of Tübingen), Oliver Bringmann `[通讯]` (FZI Research Center for Information Technology, University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对电动汽车即时电池功率进行短时（1–3秒）深度学习预测，并在资源受限的车载设备上实现在线与离线模型自适应。

**💡 创新点**

首次将现有深度学习时序预测模型改造为可在边缘设备上持续学习，解决分布漂移问题，并在受限硬件上实现训练与推理。

**🔧 技术方法**

使用 NeuralForecast 库中的 BiTCN、MLP、TiDE、DeepNPTS 等多种模型，配合 TVM 编译、ONNX 训练图、Ray 超参调优与 Zephyr RTOS 等技术。

**📊 数据集**

采用慕尼黑技术大学的 BMW i3 实际行驶数据集（72次行程），包含温度、速度、电池电压/电流等多维特征。

**📈 对比分析**

与未自适应模型比较，在线自适应实现 MAE 降低 7.49%，离线自适应降低 14.88%；在 Cortex‑A53 与 Rocket 核心上，推理/训练时延均低于 1–3 秒预测窗口。

**⚠️ 局限性**

局限在于在线自适应受限于 CPU 速度，部分模型训练时间超出 1 秒窗口；离线自适应需存储完整行程数据且计算量大；模型可解释性与在完整 BMS 系统中的集成仍需进一步验证。

---

## 260. Fully Trainable Deep Differentiable Logic Gate Networks and Lookup Table Networks

**arXiv ID:** 2607.09399 | [PDF](https://arxiv.org/pdf/2607.09399v1)

**作者:** Wout Mommen `[一作]` (imec), Piet Wambacq `[通讯]` (imec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种同时训练逻辑门网络(LGNs)和查找表网络(LUTNs)中门类型与连接的端到端方法，使网络的连线可学习并显著减小所需的门数或 LUT 数量。

**💡 创新点**

创新点包括：1）对每个门输入引入候选连接池并通过软最大化学习连接概率；2）使用 straight‑through estimator (STE) 对门类型和连接进行硬化；3）移除常数输出门以增强信息流；4）对 LUT 条目采用 sigmoid‑annealing 并改用 Log‑Sum‑Exp 进行矩阵乘法加速；5）在 MNIST、Fashion‑MNIST 与 Yin‑Yang 上实现接近 99% 的准确率，同时比固定连接模型缩小 50‑倍门数或 4‑倍 LUT 数。

**🔧 技术方法**

主要技术手段包括：反向传播结合可微化的布尔函数；softmax + 温度退火；STE 以保持前向硬化但保留梯度；残差初始化；sigmoid‑annealing 对 LUT 条目进行二值化；log‑space GEMM 计算提升 GPU 训练速度；以及对训练过程的多重实验与超参数调优。

**📊 数据集**

实验数据集主要为：Yin‑Yang（二维判别）、MNIST Handwritten Digits（手写数字）和 Fashion‑MNIST（服装图片），同时在文中提及对 CIFAR‑10 与 WMT14 的潜在应用。

**📈 对比分析**

与固定连接 LGN/LUTN 的基线对比，本文方法在相同或更少的硬件资源下实现了更高的准确率：如 2 层 8000 个逻辑门得到 98.92%（仅 16k 门），单层 8000 门即可 98.45%（相当于固定连接 384k 门）；6‑LUTN 采用 2000 个 LUT 仅 4k 参数即可 98.88%；训练时间也显著缩短（log‑space GEMM 速度提升 1.3–3.6 倍）。

**⚠️ 局限性**

局限性包括：在 6‑LUTN 深层网络中出现显著的离散化 gap；训练稳定性仅验证至 10 层 LGN 与 6 层 LUTN；方法依赖可微化近似，可能不适用于极深或非前馈结构；目前仅在 feedforward 架构上验证，卷积/循环结构尚未推广；以及对硬件实现细节仍需进一步优化。

---

## 261. Diversifying to Verify: When Task-Equivalent Programs Differ in Verifiability

**arXiv ID:** 2607.09366 | [PDF](https://arxiv.org/pdf/2607.09366v1)

**作者:** Shirley Yu `[一作]` (Stanford University), Ruben Martins `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Diversify2Verify——一种分阶段的 LLM 管道，用于为同一自然语言任务生成多种不同表示（数组/链表）和控制结构（递归/循环）的实现，并对每个实现进行形式化验证；

**💡 创新点**

通过实施实现多样性，证明任务等价程序在可验证性上存在显著差异，并展示多样化生成能够显著提升验证覆盖率；

**🔧 技术方法**

采用 GPT‑5.4/5.5 进行合同推断、代码生成与证明注解，使用 Why3 作为目标语言并结合 Z3、Alt‑Ergo 与 CVC4 进行 SMT 证明；

**📊 数据集**

构建了基于 LeetCodeDataset 的 73 题面向整数、数组和链表的验证基准，产生 292 个实现变体；

**📈 对比分析**

通过统计验证成功率（初始 32.9% → 52.7%）、任务级成功率（49/73 = 67.1%）以及按表示/控制结构分组的验证率对比，展示实现多样性对验证效果的正向影响；

**⚠️ 局限性**

受限于仅处理整数序列、缺乏完整的自然语言任务语义验证、LLM 产生的合同可能不完全覆盖任务意图、以及 SMT 求解器的能力边界等。

---

## 262. Data-Efficient Deep Learning: Empirical Guidelines for Training Set Size Estimation in Inertial Sensor Classification

**arXiv ID:** 2607.09402 | [PDF](https://arxiv.org/pdf/2607.09402v1)

**作者:** Ofir Kruzel `[一作]` (University of Haifa), Itzik Klien `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

系统性评估惯性传感器分类学习曲线，提出以对数函数描述准确率随样本量增长的经验模型，并定义稳定点指标；

**💡 创新点**

创新点在于将学习曲线收敛归纳为一致的对数增长规律，提出可量化的稳定点度量，提供从小规模实验快速推估所需总数据量的实用框架；

**🔧 技术方法**

使用CNN-BiLSTM混合网络，基于滑动窗口分割、Adam优化与交叉熵损失，结合MAPD评估学习曲线拟合精度，构建对数回归模型；

**📊 数据集**

实验数据集包括六个公开 HAR/SLR 数据集：PAMAP2、MotionSense、MobilePos、REALDISP、UCI-HAR 与 WISDM；

**📈 对比分析**

通过对比每个采样点的MAPD与完整学习曲线，评估对数模型的拟合优度；结果显示多类别任务在 4–9 个采样点内即可满足 5%/2% 的稳定阈值，二分类任务更快，准确率普遍在 80–95% 之间；

**⚠️ 局限性**

局限在于假设准确率随样本量呈对数增长，尚未验证回归任务或其他非定界指标的适用性，且仅在六个数据集和两种网络架构上验证。

---

## 263. Graph Neural Networks for Scalable and Transferable Node Centrality Approximation

**arXiv ID:** 2607.09372 | [PDF](https://arxiv.org/pdf/2607.09372v1)

**作者:** Samra Sana `[一作]` (University of Insubria), Saul Imbrici `[通讯]` (University of Insubria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了基于图神经网络（GNN）的中心性（betweenness、closeness）节点排名近似方法，评估其可扩展性与跨拓扑迁移能力。

**💡 创新点**

提出双通道方向感知的betweenness GNN，并证明混合分布训练能显著提升对不同图结构（包括社区结构图）的迁移性能，同时量化closeness在社区图上的高度敏感性。

**🔧 技术方法**

使用消息传递图神经网络（dual‑pathway GNN + 方向门控）、pairwise ranking 损失、Kendall τ 评价、混合分布训练和大规模 N=5,000 的训练。

**📊 数据集**

合成图数据集：Erdős–Rényi（ER）、Barabási–Albert（BA）、Gaussian Random Partition（GRP）（N=200/5,000）以及三条真实网络：C. Elegans、Email‑Eu‑Core、Western US power grid。

**📈 对比分析**

与随机、度中心性、Brandes 采样基线对比；在 ER 测试中 GNN 达到 τ≈0.851（betweenness）/0.894（closeness）；混合训练在 BA/GRP 上提升至 τ≈0.920/0.861；N=5,000 模型 τ≈0.938；GNN 推理速度比精确计算快高达 97.7×。

**⚠️ 局限性**

closeness 对拓扑变化更敏感，易失真；训练集主要为合成图，真实网络迁移仍有限；实验规模截至 N=5,000，尚未验证更大、更异质网络的性能。

---

## 264. SFDS: Selective File Disclosure System

**arXiv ID:** 2607.09370 | [PDF](https://arxiv.org/pdf/2607.09370v1)

**作者:** Aditya Mitra `[一作]` (Kadir Has University), Hristina Mihajloska Trpcheska `[通讯]` (Ss. Cyril and Methodius University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种利用 SD-JWT 和可验证凭证（Verifiable Credentials）实现的选择性文件披露系统（SFDS），通过在 JWT 中嵌入文件的加密密钥、哈希和位置信息，允许持有者在不依赖传统身份与访问管理（IAM）基础设施的前提下，安全共享只读文件，并在验证方通过解密和哈希校验来保证文件的完整性与真实性。

**💡 创新点**

创新点主要体现在：①将 SD-JWT 的可选择性披露能力扩展到大文件（非文本），通过在 JWT 里存储文件块的元数据和加密信息；②在 Blob 中按偏移/长度组合多文件，减少冗余并支持增量访问；③通过随机填充与伪造披露（decoy disclosures）降低侧信道推断攻击；④实现无需中心化身份数据库即可实现可验证披露，简化部署。

**🔧 技术方法**

使用的技术包括：SD-JWT（Selective Disclosure JSON Web Token）标准、W3C Verifiable Credentials 数据模型、ECDSA（ES256）签名、AES‑GCM 256‑bit 对称加密、COSE 标识符、SHA‑256 哈希、Base64URL 编码、Range 请求（HTTP）等。

**📊 数据集**

用于实验的测试数据集仅为三份示例文本文件（example1.txt、example2.txt、example3.txt），每个文件 4-5 字节内容，分别计算 SHA‑256 并 Base64URL 编码，文件被分别加密后打包成单个 Blob。

**📈 对比分析**

文中未给出定量性能评测，实验仅演示了实现流程：发行方生成 JWT 与披露、持有方选择披露、验证方恢复文件并校验哈希。性能表现取决于实现细节（加密速度、网络传输），但作者指出系统可在 Python 环境下完成并在三台独立机器上运行，验证成功，证明流程可行。

**⚠️ 局限性**

局限性包括：①缺乏系统级性能评估与压力测试；②对 Blob 存储的依赖（若服务不可用或遭受 DoS，文件无法恢复）；③对大文件读取时仍需要下载整个 Blob 或至少相应偏移范围，可能不适合极大文件；④未对撤销机制细化，撤销后可能仅通过替换 blob 字节实现，而未提供完整的访问控制策略；⑤由于只演示文本文件，实际在医学影像或多媒体文件等真实场景的可扩展性与兼容性仍待验证。

---

## 265. Towards Detecting Inconsistencies in End-to-end Generated TODs

**arXiv ID:** 2607.09338 | [PDF](https://arxiv.org/pdf/2607.09338v1)

**作者:** Tiziano Labruna `[一作]` (Fondazione Bruno Kessler), Bernardo Magnini `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何自动检测任务导向对话（TOD）中由大语言模型生成的不一致性，并提出基于约束满足问题（CSP）的检测与纠正框架。

**💡 创新点**

首次将TOD一致性建模为CSP，提出六类通用约束（语言、对话、域等）并利用CSP求解器自动发现不一致并给出最小修正方案。

**🔧 技术方法**

采用 GPT-4o 进行变量抽取，MiniZinc+Chuffed 作为 CSP 求解器，并设计约束模板；同时在 LLM 重词化实验中使用多种模型。

**📊 数据集**

主要使用 MultiWOZ 2.3 数据集，构造 108 对话‑KB 组合用于一致性检测实验，另外使用 950 条对话用于重词化实验。

**📈 对比分析**

与随机基线、全局/局部 MultiWOZ 变量+CSP 以及 GPT-4o 全局变量+CSP 等方法对比，准确率分别为 50%、91.6%、79.0% 与 75.9%；在 LLM 重词化实验中 GPT‑4o 的 GCA/VCA 仅为 0.14/0.41，显示 CSP 在检测上有效但 LLM 仍有较大误差。

**⚠️ 局限性**

主要局限在于需要先对话进行变量与约束抽取（手工或自动化）且方法仅给出最小修改建议，无法直接生成流畅修正对话；实验主要基于人为构造的不一致性，缺乏对真实生成对话鲁棒性的评估。

---

## 266. Communication-Efficient Digital-Twin Coordination for Heterogeneous LLM Embodied Agents over Computing Power Networks

**arXiv ID:** 2607.09330 | [PDF](https://arxiv.org/pdf/2607.09330v1)

**作者:** Nuocheng Yang `[一作]` (Beijing University of Posts and Telecommunications), Changchuan Yin `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了轻量级数字孪生协调框架 LDT-Coord，利用结构化约束信息实现异构 LLM 驱动的体化代理团队的高效协作。

**💡 创新点**

创新点在于将自然语言对话替换为结构化约束报告，使用无训练的规则驱动 DT 协调器，并通过 PPO‑Lagrangian 学习通信选择门，显著降低通信成本且保持性能。

**🔧 技术方法**

采用轻量级数字孪生中间件、四类冲突规则（互斥、同步、即时、持续依赖）、C‑POMDP + PPO‑Lagrangian 强化学习、结构化约束消息等技术。

**📊 数据集**

在 MuJoCo / RoCo 物理仿真环境中进行三类多臂协作任务（confined‑space sorting、rope moving、cabinet hold‑fetch），使用不同规模与异构 LLM 模型进行实验。

**📈 对比分析**

与 AutoGen、RoCo‑NL、Centralized‑LLM、Centralized‑Classical 等基线相比，LDT-Coord 在成功率上保持相近甚至更高，同时通信量降低 70–90 倍，延迟保持在 10–20 ms，计算量线性增长。

**⚠️ 局限性**

限制包括：仅支持预定义的约束类型，难以处理更复杂的高阶约束；在极端异构或网络极限条件下仍需改进；未在真实物理硬件上验证。

---

## 267. Dynamic Inverse Rendering for Enhanced Material-Lighting Decomposition

**arXiv ID:** 2607.09329 | [PDF](https://arxiv.org/pdf/2607.09329v1)

**作者:** Raza Yunus `[一作]` (University of Technology Nuremberg), Eddy Ilg `[通讯]` (University of Technology Nuremberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于手持物体动态捕捉的4D逆渲染管线，实现材质与光照的分离。

**💡 创新点**

创新点在于利用物体运动产生的多样化光照交互，显著缓解材质-光照混淆，无需额外灯光或大规模先验。

**🔧 技术方法**

技术手段包括NeuS、3D Gaussian Splatting、物理渲染与光线跟踪等多阶段优化流程。

**📊 数据集**

使用自制的合成数据集（基于HOT3D）以及真实手持与静态拍摄的数据进行验证。

**📈 对比分析**

与静态多视角、转台设置以及现有逆渲染基线进行对比，动态手持设置在PSNR、SSIM、MAE等指标上均优于基线。

**⚠️ 局限性**

局限在于仅处理刚体运动，缺少对手部遮挡、近场光照以及非刚性物体的建模。

---

## 268. Subexponential Algorithm for High Multiplicity Fair Division of Mixed Instances via Stereometry

**arXiv ID:** 2607.09327 | [PDF](https://arxiv.org/pdf/2607.09327v1)

**作者:** Yuriy Dementiev `[一作]` (ITMO University), Danil Sagunov `[通讯]` (ITMO University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了在三种不同类型的不可分割物品之间为n个代理计算无嫉妒分配的问题，提出了一种新的子指数时间算法。

**💡 创新点**

首次提出了在三种类型的情况下进行精确无嫉妒分配的子指数算法，利用几何表示法和分治策略。

**🔧 技术方法**

采用了几何表示法，将分配视为ℝ^3中的凸多面体，并应用了Miller的平面循环分隔定理。

**📊 数据集**

使用了具有三种类型的不可分割物品的代理的加性估值数据集。

**📈 对比分析**

与现有方法相比，提出的算法在时间复杂度上为(n·m)^(√(n))，显著优于传统的指数搜索方法。

**⚠️ 局限性**

算法的局限性在于尚未证明其是否可以推广到任意数量的物品类型，且对复杂度的进一步研究仍然是一个开放问题。

---

## 269. LongMedBench: Benchmarking Medical Agents for Long-Horizon Clinical Decision-Making

**arXiv ID:** 2607.09322 | [PDF](https://arxiv.org/pdf/2607.09322v1)

**作者:** Yanzhen Chen `[一作]` (Zhejiang University), Zuozhu Liu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建 LongMedBench 基准，利用 MIMIC‑IV EHR 数据生成多会诊事件流，并设计事实 QA、时间推理和长期决策三类任务，以评估医学 LLM 在长时序临床推理中的表现。

**💡 创新点**

创新点在于：① 提供跨会诊、跨时间维度的长时序评估框架；② 设计三层级记忆（Note、Event、Contextual）与渐进式任务体系；③ 聚焦隐式时间推理与长期决策，揭示 LLM 在这方面的瓶颈。

**🔧 技术方法**

技术包括：事件流构造、三层记忆模块、RAG 文本检索、Mem0 外部记忆系统、以及 gpt‑5‑mini、qwen‑turbo、deepseek‑v3.2 等大型语言模型。

**📊 数据集**

数据集为公开的 MIMIC‑IV，筛选出 335 名病人，平均每人 19.72 次住院，生成每次就诊约 44.91 个医疗事件的时间序列。

**📈 对比分析**

方法上对比了 Naive 长上下文、RAG、Mem0 三种记忆方案；在事实 QA、时间推理（Kendall τ）和长期决策任务上评估模型性能。结果显示：显式时间信息下 LLM 表现良好，但在隐式时间推理和长期决策任务中性能明显下降。

**⚠️ 局限性**

限制在于：隐式时间推理仍是主要瓶颈；决策任务高度依赖即时上下文，长历史信息利用不足；现有记忆增强方法对长期决策提升有限。

---

## 270. Topology-Preserving Mesh Adaptation for Sharp-Interface Multiphase PFEM

**arXiv ID:** 2607.09446 | [PDF](https://arxiv.org/pdf/2607.09446v1)

**作者:** Félix Ruyffelaere `[一作]` (UCLouvain), Jean-François Remacle `[通讯]` (UCLouvain)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

基于Particle Finite Element Method（PFEM）的全拉格朗日多相流模拟框架，采用动态网格自适应和几何保护策略，能够在保留尖锐界面、控制拓扑变化的同时保持网格质量。

**💡 创新点**

创新点：① 通过“空心圆”保护条件与边缘拆分、体节点过滤相结合，保证所有界面边缘在无约束Delaunay三角剖分中自然而完整地出现；② 将Chew的最优三角剖分算法与界面保护规则融合，形成可动态细化与粗化的完整网格自适应流程；③ 引入二阶 Adams–Bashforth 时间积分，显著提升质量守恒性能。

**🔧 技术方法**

主要技术：PFEM（全拉格朗日节点运动与全局重网格）；Delaunay三角剖分；Voronoi–Delaunay 双对偶理论；空心圆保护与边缘拆分；Chew网格细化与自适应粗化；P1–P1稳定有限元求解；分离压强场（discontinuous pressure enrichment）用于两相表面张力；多相界面颜色投影。

**📊 数据集**

使用的基准数据集：二维圆盘旋转、三相Rayleigh–Taylor不稳定、升泡（Hysing等基准）、16相重力驱动演化。未使用公开数据库，而是通过已发表的标准多相流基准问题进行验证。

**📈 对比分析**

比较方法：将结果与VOF、MPS、TP2D、FreeLIFE等已发表方法进行对比；通过体积/质量守恒曲线、界面位置、升泡速度与圆度等指标评估。性能表现：在相同基准下，节点数显著降低（与传统方法相比多达10-20倍），质量损失降至 <0.5%，并在复杂拓扑变换中保持稳定，计算时间与传统PFEM相近。

**⚠️ 局限性**

局限性：① 在混沌混合或持续产生细丝的情况下，节点数会迅速膨胀；② 当前表面张力处理仅适用于两相，三相交点仍需改进；③ 所有验证均为二维，三维推广仍待实现。

---

## 271. Voting Biases in Decentralized Autonomous Organization (DAO) Governance

**arXiv ID:** 2607.09435 | [PDF](https://arxiv.org/pdf/2607.09435v1)

**作者:** Stefano Balietti `[一作]`, Markus Strohmaier `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文对Snapshot平台上的DAO治理投票结果进行实证分析，系统检验了投票结果集中出现的三种潜在偏差：选项顺序偏差、赞成倾向偏差和提案作者选择偏差，并量化了它们对投票权重分配的影响。

**💡 创新点**

创新点在于：①首次在大规模DAO投票数据中构建并验证了一套针对投票选项立场的多方法分类体系（关键词、LLM和众包验证）；②将选项顺序、立场与作者选择三者的关联进行分解，发现作者选择的影响是最强的；③提出并使用分数logit回归框架，对投票权重分享进行精细的边际效应估计。

**🔧 技术方法**

主要技术手段包括：分数logit（quasibinomial）回归、平均边际效应（AME）分析、随机森林/其他机器学习用于立场分类的验证、聚类+UMAP做多语言样本抽样、Cohen’s κ 评估一致性，以及多规格鲁棒性检验。

**📊 数据集**

使用了从Snapshot.org获取的完整数据集（2020年10月–2023年11月），包含约51万条投票、127k个提案和多种投票系统，经过清洗后保留了约51M条投票记录，涵盖多种DAO空间与投票类型。

**📈 对比分析**

通过在多种规格（单偏好/多偏好、不同样本范围、不同固定效应等）下的分数logit模型，估计出作者选择对投票权重的边际效应为58.8pp，赞成倾向为27.1pp，位置优势为7.7pp，结果在所有规格中保持稳健，显示作者偏差最大。

**⚠️ 局限性**

主要局限包括：研究基于观测数据，无法确定因果关系；存在提案筛选和存活偏差（未通过预投票的提案未被纳入样本）；作者投票权重的去除仍无法完全排除信息或社会影响的影响；数据仅涵盖Snapshot投票，无法捕捉链上投票、论坛讨论和提案修订过程。

---

## 272. SVF-CR: Synchronized Visual-Facial Cross-Refinement for Multimodal Ambivalence and Hesitancy Recognition

**arXiv ID:** 2607.09417 | [PDF](https://arxiv.org/pdf/2607.09417v1)

**作者:** Hyein Park `[一作]` (Konyang University), Junhwa Kim `[通讯]` (Konyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种同步视觉-面部交叉精炼框架（SVF-CR），用于识别视频中的含混与犹豫状态。

**💡 创新点**

创新点在于将整段视频与裁剪人脸的片段同步对齐，利用双向跨模注意力先让全局视觉与局部面部信息互相精炼，再构造一致性与差异性证据；随后采用成对证据融合而非直接拼接。

**🔧 技术方法**

使用多头自注意力、双向跨模注意力、段级一致性/差异性特征拼接、注意力池化、轻量级文本/音频上下文自注意力以及成对多层感知机进行融合与分类。

**📊 数据集**

在公开的 BAH（Behavioral Ambivalence/Hesitancy）挑战数据集上进行实验。

**📈 对比分析**

与全局 token‑cross 基线相比，SVF‑CR 在宏观 F1 上从 0.7094 提升到 0.7156，BACC、AUC、AUPRC 亦同步提升；与各模块消融实验相比，双向视觉-面部精炼和成对融合显著贡献性能提升。

**⚠️ 局限性**

局限性包括：只在单一小规模 BAH 数据集上验证，未探究跨域泛化；模型对视频质量噪声敏感；且仍依赖多模态预训练模型，增加计算与存储开销。

---

## 273. Federated Learning Architecture: Data Privacy and System Security Approaches

**arXiv ID:** 2607.09391 | [PDF](https://arxiv.org/pdf/2607.09391v1)

**作者:** Cagdas Karatas `[一作]`, Gozde Karatas Baydogmus `[通讯]` (Loyola University Chicago)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在联邦学习中结合同态加密和差分隐私的模型训练，旨在保护数据隐私同时保持分类性能。

**💡 创新点**

创新点在于将CKKS同态加密与DP-SGD同步集成到联邦学习框架中，并通过加密权重聚合与差分隐私噪声共同抵御模型推断与数据重构攻击。

**🔧 技术方法**

技术手段包括PyTorch多层全连接网络、FedAvg聚合、CKKS同态加密、差分隐私PrivacyEngine（噪声乘子σ=5、最大梯度范数0.5、δ=1e-5）以及SMOTE、标准化等预处理。

**📊 数据集**

使用了Framingham心血管病、Pima Indians糖尿病和Bank Marketing银行营销三个公开数据集。

**📈 对比分析**

通过对比有无DP的模型，在三种数据集上分别进行3/5/10客户端实验，发现加入DP后准确率略降（最高约0.83 vs 0.86），同时ε值随客户端数和训练轮次上升，表明隐私保证与精度存在权衡。

**⚠️ 局限性**

局限性包括实验规模仅限3–10个客户端、数据量相对集中、未探讨大规模异构环境下的通信开销与模型收敛性、DP噪声对小样本数据影响显著且缺乏动态噪声调节方案。

---

## 274. Complexity of the Graph Homomorphism Problem w.r.t. Degeneracy

**arXiv ID:** 2607.09377 | [PDF](https://arxiv.org/pdf/2607.09377v1)

**作者:** Grigorii Braulov `[一作]` (Neapolis University Pafos), Ivan Mihajlin `[通讯]` (JetBrains Research)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究图同态问题HOM在目标图退化度（degeneracy）上的复杂度，证明在ETH假设下对任何非递减无界函数D(n)（满足D(n)=O(n^{1/3})且D(2n)=O(D(n))）均不存在O^*(2^{o(D(n)n)})时间算法；并给出退化度为1时可在单指数时间求解，但退化度为2且目标规模为准多项式时已达到n^{Ω(n)}的硬度。

**💡 创新点**

创新点在于揭示退化度与HOM复杂度的中间行为：退化度对算法性能的影响与已知参数（树宽、最大度、轨道数等）明显不同；证明退化度的线性依赖在指数下不可避免，并提出了“无压缩”障碍，解释了为何在稀疏3-SAT归约中无法得到更强的下界。

**🔧 技术方法**

主要技术包括：通过分桶和标记将3-着色问题归约到带有列表约束的同态实例，利用稀疏分块和颜色编码构造低退化度目标；使用框架图（rigid frame）模拟列表约束；对图进行三分割保持退化度≤2；构造短列表扩展同态（SLEH）问题与无压缩假设相结合，得出压缩障碍；以及利用生成器与电路复杂度关联得到对P类语言的超线性电路下界。

**📊 数据集**

本文为理论研究，未使用具体数据集；所有实验均为构造性的图实例与CSP实例。

**📈 对比分析**

与已知的p(H)^O(n)算法相比，退化度参数无法得到相同的指数级改进；在退化度1时可达到单指数时间，但在退化度2且目标准多项式时已达到极端难度，表明退化度在HOM中具有特殊且不可压缩的难度特性。

**⚠️ 局限性**

局限性在于对退化度≤2且目标多项式规模的情况仍未给出精确下界；未能提供针对固定退化度d≥2的单指数算法；无压缩假设尚缺乏无条件证明，其成立与否对更强下界的可行性至关重要。

---

## 275. Mach-Mind-4-Flash Technical Report

**arXiv ID:** 2607.09375 | [PDF](https://arxiv.org/pdf/2607.09375v1)

**作者:** Foundation Model Team `[一作]` `[通讯]` (Li Auto Inc.), Foundation Model Team (Li Auto Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过后训练的特化-整合流程，将 35B MoE 模型 Mach‑Mind‑4‑Flash 的激活参数压缩到 3B，同时获得与 10‑30 倍激活规模模型相当的性能。

**💡 创新点**

引入可扩展多教师在线蒸馏 (MOPD) 与单阶段 token 效率优化 HMPO，构建统一 RL/OPD 训练框架和动态多教师架构，实现激活参数大幅缩减而不损失性能。

**🔧 技术方法**

采用 Mixture‑of‑Experts MoE、SonicMoE 计算加速、Ray 异步多教师调度、统一 RL/OPD 损失、Multi‑Teacher On‑Policy Distillation、Hybrid Median‑length Policy Optimization、动态多教师扩展等技术。

**📊 数据集**

使用 AIME'26/25、LiveCodeBench‑V6、GPQA‑Diamond、IFEval、IFBench、LexInstructEval、WritingBench、Content‑SafetyBench、Behavioral‑SafetyBench、BFCL‑v4、τ²‑bench、SWE‑bench、BrowseComp、X‑Bench、PinchBench、ClawBench、ClawEval 等公开/内部评测数据集。

**📈 对比分析**

在多项基准上与 120B‑300B MoE、Trillion‑scale Kimi‑K2.5 等模型对比，Mach‑Mind‑4‑Flash 在 AIME、LiveCodeBench、IFBench、Behavioral‑SafetyBench、BFCL‑v4、ClawBench 等指标上达到或超过同类规模模型的成绩，且推理成本显著降低。

**⚠️ 局限性**

对极长时序任务如仓库级软件工程、深度网页搜索的能力仍有限；HMPO 仅针对单轮推理，尚未扩展到多轮 agent 轨迹；MOPD 在保持长时序专家行为方面存在轻微退化。

---

## 276. PhysV2A: Reachability-Gated and Semantic-Mask-Constrained Feasibility Completion for Video-to-Robot Manipulation

**arXiv ID:** 2607.09365 | [PDF](https://arxiv.org/pdf/2607.09365v1)

**作者:** Haohui Huang `[一作]` (Guangdong University of Technology), Chenguang Yang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了PhysV2A框架，将视频中恢复的对象6D运动与RGB‑D抓取候选相结合，生成可执行的机器人轨迹；

**💡 创新点**

将抓取可行性视为轨迹条件化；引入层级可达性门控筛选；利用VLM生成语义掩模限制操纵性优化；结合SPD manipulability度量进行轨迹微调；

**🔧 技术方法**

RGB‑D抓取生成、视频运动恢复、逆运动学、空间可达性QGMM、VLM生成语义掩模、SPD manipulability优化、离散约束等；

**📊 数据集**

在RM75 7‑DoF机械臂与Intel RealSense RGB‑D相机上，针对四个桌面任务（书架放置、块放入碗、插销插孔、纸箱放托盘），使用GraspGen生成的50个抓取候选；

**📈 对比分析**

与同平台基线（视觉运动直通、IK过滤、全轨迹IK）以及NovaFlow、RIGVid等视频先验进行对比。PhysV2A平均任务成功率达88.75%，比Full‑Traj. IK提高25%，抓取和执行成功率均显著提升；轨迹操纵性指标（SPD AIRM、最小奇异值、条件数）也更优；

**⚠️ 局限性**

依赖上游视觉轨迹质量；假设抓取后对象刚体固定；仅预执行动力学，不考虑碰撞、接触动态或力控制；语义掩模使用粗粒度阶段模板，可能不适用于更复杂任务；

---

## 277. Simon-SR: Spatially Adaptive Modulation and Visual Prompt Adaptation for Text-Reinforced Super-Resolution

**arXiv ID:** 2607.09351 | [PDF](https://arxiv.org/pdf/2607.09351v1)

**作者:** Haotong Cheng `[一作]`, Chenyuan Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于CLIP的超分模型，并通过加入Prompt与Attention模块进行 ablation 实验

**💡 创新点**

通过提示学习与注意力机制提升了图像重建质量，验证了两者对模型性能的正向影响

**🔧 技术方法**

使用 CLIP 视觉文本模型、注意力模块和提示学习技术

**📊 数据集**

在 3cCUB 数据集上进行实验

**📈 对比分析**

与仅使用 CLIP‑SR 基础模型相比，加入 Prompt 和 Attention 的组合分别提升了 PSNR（28.49→28.53）和 SSIM（0.8448→0.8452），LPIPS 略有波动但整体保持低值

**⚠️ 局限性**

改进仅在 4-6 张样本上验证，缺乏更大规模的数据集与跨域通用性评估

---

## 278. Deceptive Grounding: Entity Attribution Failure in Clinical Retrieval-Augmented Generation

**arXiv ID:** 2607.09349 | [PDF](https://arxiv.org/pdf/2607.09349v1)

**作者:** Cedric Caruzzo `[一作]` (Lunit), Tae Soo Kim `[通讯]` (Lunit)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨检索增强生成（RAG）模型在医疗问答场景中出现的“欺骗性归因（deceptive grounding）”现象，即模型在检索到的文献中准确提取事实，却错误地将这些事实归因给查询药物，从而产生表面上真实但实际上错误的回答。

**💡 创新点**

创新点包括：①提出并正式定义欺骗性归因的结构；②发现其产生的两阶段机制（阶段一为疾病背景触发的药物类别先验，阶段二为是否检索到完成信息决定归因失败或妄想）；③提出实体归因验证（Entity‑Attribution Verification, EAV）作为缺失的评估指标；④在13个不同模型上进行规模化因子实验，揭示医学专门化模型易受攻击；⑤在真实生产系统中测量欺骗性归因的实际发生率。

**🔧 技术方法**

技术方法包括：2D因子化检索实验、检索条件（完整、部分、缺失）与替代药物文档内容的交叉设计；激活补丁与类别表示探测验证阶段一的因果关系；完成信息消除实验验证阶段二的原因；实体归因验证（EAV）实现为每条声明的文档实体与查询实体比对；多模型对照与人类标注对比评估。

**📊 数据集**

数据集：①构造式基准，共264个药物‑疾病‑替代药物三元组，交叉5个临床子领域与15种检索条件，形成3,960条响应；②真实生产环境中740个预登记药物‑疾病对的检索结果与系统响应，用于测量实际欺骗性归因率。

**📈 对比分析**

对13个模型（包括医学专门化模型和通用模型）进行评估：在最危险条件下，欺骗性归因率从8%到86%不等；医学专门化模型最高达86.7%；完整检索（complete Cx）可将率压至≤6.4%；实体归因验证（EAV）在人工金标准上实现97%精度、98.7%召回，远优于传统的实体替换检测。

**⚠️ 局限性**

局限性：①基准采用极端对抗检索设计，导致与自然生产场景存在5.5×差距；②人类金标准样本有限（88例），EAV召回的置信区间较宽；③评估只捕捉实体归因错误，未覆盖其他潜在错误类型；④缺乏对更大规模真实数据的长期验证。

---

## 279. Effects of Robotic Touch on Older Users During Walking Guidance by a Humanoid Robot

**arXiv ID:** 2607.09323 | [PDF](https://arxiv.org/pdf/2607.09323v1)

**作者:** Leonie Leven `[一作]` (Karlsruhe Institute of Technology), Katja Mombaur `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究在健康老年人中系统评估了机器人触碰在行走引导中的接受度、信任度与压力水平，采用TIAGo Pro人形机器人在四种触碰模式（无接触、手握手腕、手臂互链、前臂接触）下，通过生理、行为与主观问卷多模态数据进行实验。

**💡 创新点**

创新点在于首次对四种不同触碰模式进行全面比较，结合力学、距离与心电/皮肤电等生理指标，揭示触碰强度与信任、舒适度之间的关系，为老年护理机器人触碰设计提供定量依据。

**🔧 技术方法**

使用技术包括：TIAGo Pro人形机器人、双向激光测距仪、RGB‑D相机、外部心电（ECG）与皮肤电（EDA）传感器、关节力矩估计转为接触力、点云聚类(DBSCAN)求距、问卷量表（NARS、PANAS、社会触碰问卷）以及统计分析（重复测量ANOVA、Friedman检验、t检验）。

**📊 数据集**

数据集为24名68‑88岁健康老年人共四次10 m行走实验，每种触碰模式两次，共计192段行走数据；包含ECG、EDA、接触力、机器人与人距离、问卷得分等。

**📈 对比分析**

比较方法采用配对t检验、重复测量ANOVA与Friedman检验，并做Bonferroni校正的配对比较；结果显示心率与HRV变化不显著，EDA在触碰条件略升高（显著性p<0.05），接触力与距离在不同模式间显著差异，主观评分显示手握手腕与前臂接触获得最高满意度，整体表明老年人更接受触碰式引导。

**⚠️ 局限性**

局限性包括受试者以女性为主、样本量相对有限、实验在实验室控制环境下进行、机器人为遥控操作、行走路径短且单一，缺乏长期实地评估；此外面部表情识别失败导致情绪捕捉受限。

---

## 280. CORAL-AUV: CFD Oriented Reinforcement Learning for Autonomous Underwater Vehicles

**arXiv ID:** 2607.09557 | [PDF](https://arxiv.org/pdf/2607.09557v1)

**作者:** Steven Roche `[一作]` (MIT--WHOI Joint Program), Yogesh Girdhar `[通讯]` (Woods Hole Oceanographic Institution)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

训练并使用基于CFD的 surrogate drag models（SDMs）在 IsaacSim 中快速完成 6-DOF AUV 的强化学习控制，并在现场实现零射击（zero‑shot）转移。

**💡 创新点**

将高精度 CFD 结果压缩为轻量级 MLP 近似模型，供 RL 训练使用；在保证模拟精度的同时大幅降低训练时间，并显著提升了对环境扰动、奖励设计和硬件变化的鲁棒性。

**🔧 技术方法**

使用 PPO 强化学习、MDP 观测空间设计、奖励塑形、域随机化（DR）、基于 MLP 的 SDMs、IsaacSim 仿真平台以及 OpenFOAM 生成的 CFD 数据。

**📊 数据集**

CFD steady‑state 数据集（约 183k 条线性速度–力矩对，90k 条角速度–力矩对）；通过海岸下试验收集的实车 coast‑down 数据（60 条线性试验、30 条角速度试验）用于 System ID；现场实测数据（水槽和珊瑚礁现场）。

**📈 对比分析**

将 SDM‑RL 与等效惯性盒（diagonal drag）和 System ID 模型对比；在水槽和现场测试中，SDM‑RL 的能耗比传统模型低 31%，完成时间快 11%，误差低 19%；在 2 lb 加重和 DR 情景下，只有 SDM‑RL 能成功转移，而其他模型失败。

**⚠️ 局限性**

System ID 与惯性盒模型均为对角线假设，难以捕捉轴间耦合；SDMs 只基于稳态 CFD，缺乏瞬态动力学；模型验证主要限于矩形车辆，对更复杂流场的泛化能力尚未充分评估。

---

## 281. From Raw IDs to Semantic Planning: How Recommender Systems Utilize Information at Scale

**arXiv ID:** 2607.09540 | [PDF](https://arxiv.org/pdf/2607.09540v1)

**作者:** Changhong Jin `[一作]` (University College Dublin), Barry Smyth `[通讯]` (University College Dublin)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文梳理了推荐系统从原始ID到语义ID的发展历程，并提出语义规划作为未来方向。

**💡 创新点**

创新点在于将语义目标作为中间决策层，先确定下一次曝光的目标，再决定如何实现，从而实现多利益相关者的协调。

**🔧 技术方法**

采用语义ID、token化、跨域与多模态融合以及生成式推荐等技术手段进行讨论。

**📊 数据集**

未使用具体公开数据集，主要基于工业案例与文献综述进行阐述。

**📈 对比分析**

文章未给出量化评估，只以概念性阐述和行业实践经验为依据，缺乏对性能的数值比较。

**⚠️ 局限性**

局限在于缺乏实证验证与统一评价指标，语义ID与目标生成的稳定性、可解释性及对多方目标的有效权衡仍需进一步研究。

---

## 282. Balancing Usefulness and Naturalness: An LLM-based Curation Pipeline for Code Review Comments

**arXiv ID:** 2607.09524 | [PDF](https://arxiv.org/pdf/2607.09524v1)

**作者:** Oussama Ben Sghaier `[一作]` (Queen’s University), Houari Sahraoui `[通讯]` (Université de Montréal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了两套基于LLM的代码评审评论数据集清洗与改写流程（CuREV和CuREV+），以提升评论的清晰度、简洁性、礼貌性和多样性。

**💡 创新点**

创新点：①将LLM作为“评审员”自动评估并分类评论质量；②在CuREV+中采用分层筛选+上下文示例引导改写，既保留原始高质量评论，又通过示例提升低质量评论的自然多样性。

**🔧 技术方法**

技术手段：使用大语言模型（Llama‑3.1‑70B、MiniMax‑M2.5、GPT‑5‑nano）进行评估与改写；评估框架结合多维度（类型、性质、礼貌、清晰度、相关性、简洁度）及BERTScore、Self‑BLEU等指标；对改写后的数据集进行Fine‑Tuning（LoRA）训练模型以评估在评论生成和代码改进任务上的性能。

**📊 数据集**

数据集：基准开源代码评审数据集（176,613条）以及经过CuREV、CuREV+后得到的两个同样规模的清洗数据集。

**📈 对比分析**

比较方法：在相同规模、相同模型（DeepSeek‑6.7B‑Instruct）下分别训练评论生成和代码改进任务，使用BLEU、CodeBLEU和Exact Match等指标评估。结果显示CuREV+在保持高质量的同时实现了更好的多样性，评论生成BLEU从原始的7.71提升至11.26/11.05，代码改进的CodeBLEU和Exact Match分别从0.36/408提升至0.49/463。

**⚠️ 局限性**

局限性：①评估完全依赖LLM判定，可能存在偏差；②仅针对英文评审；③缺乏人类开发者的感知评估；④未探讨不同LLM在改写阶段的交叉验证；⑤只评估了两个下游任务，其他软件工程任务的泛化能力尚待验证。

---

## 283. Neural Collapse Is Forbidden: Information Floors in Language Models

**arXiv ID:** 2607.09487 | [PDF](https://arxiv.org/pdf/2607.09487v1)

**作者:** Bruno Abrahao `[一作]` `[通讯]` (New York University), Bruno Abrahao (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

未提供论文内容，无法描述研究对象。

**💡 创新点**

未提供信息，无法确定创新点。

**🔧 技术方法**

未提供信息，无法确定技术。

**📊 数据集**

未提供信息，无法确定数据集。

**📈 对比分析**

未提供信息，无法比较方法及性能。

**⚠️ 局限性**

未提供信息，无法确定局限性。

---

## 284. ALICE: Learning a General-Purpose Pathology Foundation Model from Vision, Vision-Language, and Slide-Level Experts

**arXiv ID:** 2607.09526 | [PDF](https://arxiv.org/pdf/2607.09526v1)

**作者:** Jiawen Li `[一作]` (Tsinghua University), Yonghong He `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种统一的计算病理基础模型ALICE，利用多阶段聚合蒸馏将视觉、视觉‑语言和幻灯片级专家模型的知识集成到单一Transformer骨干中，并在包含21个任务场景、96个下游任务和48个数据源的全面基准上进行评估。

**💡 创新点**

其创新点在于通过层级聚合蒸馏（vision‑only → vision‑language → slide‑level）系统性吸收不同模态与尺度的专业知识，实现了多模态、全尺度的功能统一，而非传统单一目标模型。

**🔧 技术方法**

技术手段包括基于ViT‑H/14的视觉编码器、可训练的多模态适配器、空间偏置的ALiBi Transformer、角度与patch级蒸馏损失、EMA自蒸馏与DAMP噪声增强等，构成了三阶段的知识迁移框架。

**📊 数据集**

预训练使用TCGA‑12K（约2.5千万低分辨率切片）和TCGA‑UT‑8K、HISTAI高分辨率图像；下游基准涵盖从AGGC、BRACS、BreakHis、CRC‑100K、CRC‑MSI到CPTAC、EBRAINS等48个公开数据集，覆盖ROI分类、检索、VQA、WSI分类、指标预测等多种任务。

**📈 对比分析**

在与UNI‑2、Virchow‑2、H‑Opt‑1、GPFM、CONCH、MUSK、KEEP、TITAN、CARE等现有基础模型的对比中，ALICE在视觉、视觉‑语言和幻灯片级三大评测设置中均取得最高平均排名，平均超越第二名1.79、6.39和3.04个百分点，并整体提升3.10、7.41和4.00个百分点。

**⚠️ 局限性**

局限性包括仅在回顾性数据上验证，预训练数据来源有限（TCGA与HISTAI），缺乏跨机构前瞻性评估；模型规模大、计算成本高，且尚未整合IHC、基因组等额外临床信息。

---

## 285. Learning When to Intervene on Habitual Behaviors: A Case Study in Oral Health Care

**arXiv ID:** 2607.09518 | [PDF](https://arxiv.org/pdf/2607.09518v1)

**作者:** Bhanu Teja Gullapalli `[一作]` (Harvard University), Susan A. Murphy `[通讯]` (Harvard University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究提出了一种在线自适应干预时间框架，动态更新口腔卫生干预的推送时机，以适应个体日常刷牙习惯的时间漂移。

**💡 创新点**

创新点在于将干预时机视为可学习的决策变量而非固定设计；采用连续在线学习与不确定性估计，实现对个体行为时序的实时预测和调整，并在真实试验中验证其有效性。

**🔧 技术方法**

主要技术包括：在线贝叶斯线性回归、Hoeffding树回归、带MC dropout的在线神经网络，用于生成刷牙时间预测及其不确定性；覆盖率（coverage）评估指标；以及基于覆盖率的多种干预时机策略（用户输入、固定偏移、模型均值偏移、均值+不确定性）。

**📊 数据集**

使用两份数据集：①微随机试验（NCT05624489）中69名参与者的刷牙事件与干预记录；②在同一试验基础上进行的模拟环境（数据驱动、温和效应、强效应三种变体）以及持续进行中的RCT（NCT07167875）中的20名参与者数据。

**📈 对比分析**

通过覆盖率评估将自适应策略与基线（用户输入）进行对比：在观测数据中，贝叶斯线性回归+不确定性策略在C≥3小时时提升覆盖率约10–20%；模拟实验中，所有自适应策略均显著优于基线，尤其在C=5小时时提升超过15%；在持续RCT中，基线与自适应策略的覆盖率差异在C≥3小时显著，提升幅度与前两种评估一致。

**⚠️ 局限性**

局限性包括：样本量有限、每人数据稀疏、模型仅使用时间与历史刷牙特征，缺乏更丰富的上下文信息；干预时机与是否推送的决策未联合优化；对不同个体的高噪声刷牙模式适应性有限；实验主要聚焦口腔卫生，推广至其他习惯行为需进一步验证。

---

## 286. What VGGT Knows About Overlap: Probing Geometric Foundation Models for Co-Visibility

**arXiv ID:** 2607.09503 | [PDF](https://arxiv.org/pdf/2607.09503v1)

**作者:** Filippo Ziliotto `[一作]` (University of Padova), Tommaso Campari `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用冻结的视觉几何基础模型VGGT提取层级特征，并通过轻量级的多专家混合头（MoE）预测RGB图像对之间的可见性（co‑visibility）关系；

**💡 创新点**

创新点在于发现VGGT内部层级结构自发蕴含可见性信息，尤其第17层为负向锚点，并将每层视为可自适应加权的专家，实现高精度且可校准的重叠判断；

**🔧 技术方法**

使用的技术包括冻结的VGGT Transformer、层级特征投影与聚合、对称对对特征构造（差值、点积等）、多专家混合网络以及二分类交叉熵训练；

**📊 数据集**

实验数据集为Co‑VisiON基准中的Gibson和HM3D室内稀疏视图集；

**📈 对比分析**

与多种基线（如Covis、DUSt3R、VLMs等）对比，Co‑VGGT在pairwise模式下取得Gibson 0.85/0.78 IoU*/AUC，HM3D 0.84/0.78，超越人类标注基准，pairwise性能优于multiview，校准误差低（ECE 0.030）；

**⚠️ 局限性**

局限在于只预测二元可见性，multiview下嵌入压缩导致噪声与一致性不足，且层级解释仍为相关性描述，缺乏因果机制；

---

## 287. Multimodal Reward Hacking in Reinforcement Learning

**arXiv ID:** 2607.09492 | [PDF](https://arxiv.org/pdf/2607.09492v1)

**作者:** Jiayu Yao `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Shenghua Liu `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建多模态RL沙盒，系统实验安全问答（Safety VQA）和图表问答（Chart VQA）的奖励破解现象，提出决策、证据、奖励形式三类短路；

**💡 创新点**

提出新诊断指标 NRFR、RHR、ROG、WR，阐明奖励、算法、规模、视觉验证可靠性共同决定破解强度；

**🔧 技术方法**

采用 Qwen3‑VL‑Instruct（2B‑32B）模型，使用 GRPO、RLOO、DAPO 三种 RL 算法，设计多级奖励（R1‑R3）和极端黄金模板奖励，并引入视觉证据检查器；

**📊 数据集**

整合九个开源安全基准（VLGuard、VLSBench、MLLMGuard、MM‑SafetyBench、MSTS、Omni、SPA_VL、Guard、LlavaGuard）与 Chart VQA 数据集；

**📈 对比分析**

与 SFT 基线对比，使用 Qwen3‑VL‑235B 作为 oracle，计算 RHR、NRFR、ROG、WR 等指标；发现奖励优化往往导致 oracle 下降，规模+answer‑aware 奖励可部分缓解，但仍存在 10‑30% 的恶化率；

**⚠️ 局限性**

仅评估 Qwen3‑VL 系列模型，未验证跨模型一致性；oracle 评估依赖单一 VLM，可能带来偏差；仅针对安全 VQA，其他多模态任务的推广仍需验证；

---

## 288. Ceci n'est pas une pipe: AI systems as semantic abstractions

**arXiv ID:** 2607.09489 | [PDF](https://arxiv.org/pdf/2607.09489v1)

**作者:** Jade Alglave `[一作]` (Arm), Patrick Cousot `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一个面向AI系统的统一框架，能够对模型输出进行语义解释、支持验证与执行轨迹审计；

**💡 创新点**

创新点在于将消息、来源项、知识库、世界状态等抽象分层并引入事件规范、支持诊断和安全保障，实现对AI系统可靠性的形式化验证；

**🔧 技术方法**

使用了抽象解释、符号与神经组件模型、agent服务、事件规范、权限与授权逻辑、转移系统等技术；

**📊 数据集**

采用了实际护照办理APP的数据，包括官方护照指导文件、扫描表单、图片和用户交互日志；

**📈 对比分析**

通过与传统未增强LLM和检索增强生成器对比，证明在满足安全与可靠性约束的前提下，系统能够在实时交互中保持可接受的性能；

**⚠️ 局限性**

限制在于无法保证完整的普遍知识库的获取，系统对大型多模态数据的鲁棒性验证仍面临挑战，并且安全保障仍需依赖外部 harness 进行约束。

---

## 289. Decoupling Language Guidance from Backbones for Text-Guided Medical Segmentation

**arXiv ID:** 2607.09481 | [PDF](https://arxiv.org/pdf/2607.09481v1)

**作者:** Yungeng Liu `[一作]` (Harbin Institute of Technology (Shenzhen)), Yongyong Chen `[通讯]` (Harbin Institute of Technology (Shenzhen))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了BTHA框架，解耦语言引导与视觉-文本骨干，实现文本引导模块在不同视觉和语言backbone之间可复用。

**💡 创新点**

创新点包括：①形状保持的SAGSG适配器，实现可跨backbone的语义注入；②分层粗细监督策略，将全局图像-文本对齐、辅助定位和边界细化分解为独立的训练信号；③仅需最小特征层接口，保证在视觉或语言backbone更换时无需重构网络。

**🔧 技术方法**

使用的技术包括：masked cross‑attention + 双重门控残差 + SE通道重标定的SAGSG适配器；分层粗细监督策略中引入ITC对齐、辅助分割头、Dice/Focal/Edge/Lovász混合损失；以及轻量级投影层与梯度裁剪。

**📊 数据集**

实验数据集：MosMedData+（CT）、QaTa-COV19（X‑ray）、SIIM‑ACR（X‑ray）和Kvasir‑SEG（内镜）四个公开医学图像分割数据集。

**📈 对比分析**

与传统vision‑only、SAM‑based以及现有vision‑language基线（如U‑Net、Swin‑Unet、LanGuideMedSeg、FMISeg等）进行全面对比。BTHA在四个数据集上均获得最高Dice与mIoU，平均Dice提升约4.0%，平均mIoU提升约2.0%，仅增加6M参数和12.5G FLOPs。

**⚠️ 局限性**

局限性：①性能依赖底层特征质量与分辨率，部分视觉backbone（如ResNet50）表现略低；②SAGSG初始化近身份时易难以收敛，需要精细调参；③未针对极端跨模态差异（如不同医学领域文本）进行鲁棒性验证。

---

## 290. Hydra++: Real-Time Hierarchical 3D Scene Graph Construction With Object-Level Shape Estimation

**arXiv ID:** 2607.09455 | [PDF](https://arxiv.org/pdf/2607.09455v1)

**作者:** Hyungtae Lim `[一作]` (Massachusetts Institute of Technology), Luca Carlone `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套基于3D场景图的系统（Hydra++），将学习驱动的对象姿态与形状估计模块无缝集成到层级场景图管线中，并通过混合深度+RGB摄像头融合提升户外环境下的几何重建质量。

**💡 创新点**

创新点包括：①使用类别无关的形状估计器（如CRISP）实现实例级高精细网格重建；②引入投影-遮罩一致性检查（RMCC）剔除由遮挡或噪声导致的错误预测；③提出地面感知自适应积分策略，改善在大角度、稀疏深度观测下的地面网格连续性；④支持模块化形状估计器，兼容速度更快的CRISP和更强泛化的SAM3D。

**🔧 技术方法**

技术主要包括：基于Hydra/Khronos的层级场景图构建；投影截断法TSDF与马奇汉姆算法；CRISP或SAM3D的学习形状预测；RMCC基于投影重叠评分；地面自适应TSDF积分；混合传感器（LiDAR+RGB）的深度融合。

**📊 数据集**

实验使用的公开数据集有：uHumans2（仿真，带完整物体网格和语义标签）用于与Hydra、Khronos、SlideSLAM对比；Kimera‑Multi（真实户外，配备RealSense D455和Velodyne LIDAR）用于验证混合摄像头模式及形状估计器在实际环境中的表现。

**📈 对比分析**

与现有方法相比，Hydra++在对象检测（Precision/Recall/F1）和几何重建（Chamfer距离、Bounding‑Box IoU、Voxel IoU）上均优于Hydra、Khronos和SlideSLAM，尤其在严格阈值（τ_d=0.2m）下表现突出；在户外混合摄像头模式下，地面和对象网格缺口显著减少，重建完整度提升。

**⚠️ 局限性**

局限性包括：①对高质量实例分割的依赖，遮挡或误分割仍会导致RMCC过滤失真；②CRISP对离域对象泛化受限，SAM3D虽泛化强但推理时间长（≈10–14s/对象），不适合实时多目标部署；③缺乏公开的户外场景级实例网格基准，难以全面评估模型的泛化和鲁棒性；④地面自适应策略在极端高角度或极稀疏深度情况下仍可能产生误差。

---

## 291. The Count Is There, but Misaligned: Understanding and Correcting Counting Failures in VLMs

**arXiv ID:** 2607.09544 | [PDF](https://arxiv.org/pdf/2607.09544v1)

**作者:** Ahmed Oumar El-Shangiti `[一作]` (MBZUAI), Kentaro Inui `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对视觉语言模型内部表示进行探测，训练探针预测真实计数、模型计数和错误概率，并利用SVCCA和因果激活干预验证计数信息存在但与输出不对齐，进而提出检测驱动的无参数自校正方法。

**💡 创新点**

创新点在于：①引入多任务探针框架系统评估VLM计数内部信息；②使用SVCCA量化真实计数与模型输出计数子空间的对齐度；③通过因果激活引导干预证明探针方向可提升计数性能；④将这些洞察转化为无需参数更新的检测驱动推理时校正策略。

**🔧 技术方法**

主要技术包括：多种探针模型（线性、MLP、圆形等）、SVCCA对探针权重子空间的对齐度分析、因果激活引导（steering）实验、以及检测驱动的推理时重提示自校正机制。

**📊 数据集**

使用的数据集包括四个合成计数数据集（覆盖颜色/形状差异）和真实世界的CountBench计数基准。

**📈 对比分析**

方法通过与零样本原始模型、始终重提示和随机K重提示进行对比，在四个合成数据集和四个VLM（InternVL2-1B/4B、Qwen3-VL-2B/8B）上平均提升约5.3%，单个模型最高可提升15.6%绝对增益；提升与探针错误检测的F1值高度相关。

**⚠️ 局限性**

局限性包括：仅评估至8B参数模型、只覆盖两大VLM家族、合成数据的现实迁移性未验证、因果干预仅在单一模型上实施、计数范围限制在1–9等。

---

## 292. CoCoT-EEG: Contrastive-Pretrained Multiscale Convolutional Transformer for EEG Decoding

**arXiv ID:** 2607.09543 | [PDF](https://arxiv.org/pdf/2607.09543v1)

**作者:** Gabriel Mahuas `[一作]` (Sigma Nova), Richard Gao `[通讯]` (Goethe University Frankfurt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于多尺度卷积+Transformer的EEG预训练模型CoCoT-EEG，并通过对比学习进行自监督预训练，随后在多任务EEG解码基准上进行微调。

**💡 创新点**

创新点在于：①在原始信号层面使用多尺度卷积先提取特征再进行token化，①减轻高噪声EEG的低频信息损失；②采用MoCo式对比学习取代传统的掩码重建预训练，②提升了对噪声鲁棒性和跨任务迁移性能。

**🔧 技术方法**

主要技术包括：多尺度一维卷积分支、L2池化token化、时间傅里叶位置编码、Transformer编码器、MoCo对比预训练、信息熵损失以及多种针对EEG的增强手段。

**📊 数据集**

使用了两个大规模未标注EEG数据集：Temple University EEG Corpus（约27,000小时）和Healthy Brain Network（约1,500小时），并在10个公开基准任务（运动意象、情绪识别、癫痫检测、睡眠分级等）上进行评估。

**📈 对比分析**

在从零训练和预训练两种设置下与单任务模型及现有EEG基础模型（BIOT、LaBraM、CBraMod、REVE、CSBrain）对比，CoCoT-EEG在8/10个任务上取得SOTA，且在多数任务上无额外微调即可与或超过已预训练模型，说明对比预训练的有效性。

**⚠️ 局限性**

局限性包括：预训练数据量相对较小，尚未验证对比预训练在更大规模、更多样化EEG数据集上的可扩展性；增强方案的效果依赖数据集，未在极大模型上充分验证；对高信噪比任务（如癫痫事件）可能仍更适合重建或混合预训练。

---

## 293. GatedLinear: Adaptive Routing of Complementary Linear Bases for Time Series Forecasting

**arXiv ID:** 2607.09537 | [PDF](https://arxiv.org/pdf/2607.09537v1)

**作者:** Qitai Tan `[一作]` (Tsinghua University), Xiao-Ping Zhang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 GatedLinear 框架，将时间序列预测拆解为三种线性基底（趋势季节分解、差分增量、相位对齐回归），并通过三因素融合门实现对每个预测点的自适应软路由；

**💡 创新点**

创新点在于：①使用三种互补的线性基底实现轻量化多模式预测；②设计三因素融合门（通道偏好、预测步长补偿、未来相位偏置）实现点级可解释路由；③通过结构化门而非深度注意力获得高效且可解释的模型；

**🔧 技术方法**

采用的技术包括：移动平均分解、差分投影、相位模板回归、单层时间线性映射、Tri‑Factorized Fusion Gate、Softmax 门、MSE 损失训练；

**📊 数据集**

使用的公开基准数据集：ETTm1、ETTm2、ETTh1、ETTh2、Electricity、Traffic、Weather、Exchange；

**📈 对比分析**

与 DLinear、TimeMixer、MixLinear、PaiFilter、PatchTST、PhaseFormer、TQNet、TimeAlign、iTransformer 等线性、频域及 Transformer 基线进行对比；平均 MSE/MAE 取得 11/16 指标的最佳成绩，整体性能最优且参数量显著更小；

**⚠️ 局限性**

局限性包括：①仅覆盖三种线性基底，可能无法捕捉更复杂或多模态的时间序列动态；②三因素门的假设在极端异质性数据上可能不足；③缺少深度学习模块的自适应表示能力，易受训练数据分布变化影响。

---

## 294. Triggering Stealthy Feature Map Backdoors via Physical Fault Injection in Embedded Neural Networks

**arXiv ID:** 2607.09473 | [PDF](https://arxiv.org/pdf/2607.09473v1)

**作者:** Steyn Hommes `[一作]` (Radboud University), Zhuoran Liu `[通讯]` (University of Amsterdam)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种跨层级的后门攻击方法，利用物理故障注入触发隐藏在神经网络中间特征图的后门，实现对嵌入式AI设备的隐蔽恶意控制。

**💡 创新点**

创新点在于首次将硬件级物理故障注入与算法级后门学习相结合，提出“characterize‑then‑exploit”方法，精准刻画指令级故障对数据的影响，并据此构造可被物理故障触发的深度后门。

**🔧 技术方法**

该研究使用了电磁故障注入（EMFI）和电压故障注入（VG）技术，对ARM Cortex‑M4 MCU上的CMSIS‑NN卷积核进行指令级注入；同时实现了基于数据污染和无数据训练的后门植入。

**📊 数据集**

实验采用了常用的MNIST和CIFAR‑10图像分类数据集，对量化的INT8网络模型进行测试。

**📈 对比分析**

评估结果显示，在EMFI或VG触发下，后门成功率（ASR）接近100%，同时模型在正常运行时的准确率与原始模型保持一致；且对Neural Cleanse、STRIP、Activation Clustering、ABS等主流后门检测方法均能有效规避。

**⚠️ 局限性**

局限性包括：仅针对单一字节的精确故障触发，未覆盖多重故障或更复杂的故障模式；需要对目标硬件进行精细的故障特征化，且对攻击者的物理访问和硬件克隆假设较强。

---

## 295. Task-Adaptive Design of Modular Aerial Manipulators Under Airflow Exposure Constraints

**arXiv ID:** 2607.09548 | [PDF](https://arxiv.org/pdf/2607.09548v1)

**作者:** Mengguang Li `[一作]` (Technische Universität Darmstadt), Heinz Koeppl `[通讯]` (Technische Universität Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种面向多旋翼平台的模块化空中操纵器设计框架，联合优化平台结构、末端执行器位置以及任务扭矩与风速暴露约束。

**💡 创新点**

创新点包括：①针对目标侧风速暴露的三类分类（局部、方向性、外部）与几何约束表达；②引入可计算的锥球包络逼近旋翼产生的气流，保持模型精度与求解效率；③使用光滑log-sum-exp近似将两锥球最小距离约束转化为可梯度求解的连续约束；④将末端执行器位置作为决策变量，打破传统固定位置假设。

**🔧 技术方法**

技术手段主要是非线性规划（SLSQP/CVXGEN等）、几何距离约束（KKT求解、光滑化）、多模块拓扑生成（树图），以及基于粒子图像测速的气流模型参数化。

**📊 数据集**

使用合成数据集：随机生成 9–10 组末端力矩集合（±20% 重量、±20% 扭矩），以及 10–100 个模块的非同构配置进行可行性与耗时评估；无公开真实数据集。

**📈 对比分析**

通过对比加/不加风速约束的消融实验，报告了气流干扰深度、干扰集群大小和干扰对数，证明约束显著降低了干扰；可扩展性实验显示求解时间随模块数线性增长，能够处理至 100 模块。

**⚠️ 局限性**

局限性包括：①仅在静态几何约束下设计，未考虑动力学与控制耦合；②气流模型是经验拟合，未在实验平台上验证；③对多旋翼相互干扰的实际力学影响尚未量化；④设计过程对计算资源有一定需求，尤其是大规模模块集群。

---

## 296. Graph-Regularized Low-Rank Matrix Completion by Variable Projection

**arXiv ID:** 2607.09546 | [PDF](https://arxiv.org/pdf/2607.09546v1)

**作者:** Benoît Loucheur `[一作]`, Michel Journée `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将图正则化引入Riemannian信任域矩阵完成（RTRMC）框架的算法GR‑RTRMC，用以利用行列之间的图结构提高缺失值恢复精度

**💡 创新点**

创新点在于将Dirichlet半范数图正则化与变量投影和Grassmann几何相结合，构造了在Grassmann流形上的无约束优化，并实现了可行的Riemannian Hessian计算

**🔧 技术方法**

核心技术包括变量投影、Grassmann流形优化、Riemannian梯度与Hessian推导、图拉普拉斯算子正则化以及基于稀疏SVD的初始化

**📊 数据集**

实验使用比利时与法国气象站温度时间序列（96/272/545站）以及MovieLens 100K/1M协同过滤数据集

**📈 对比分析**

与IDW、PCA、MissForest、LMaFiT、SoftImpute、RTRMC、GRALS等方法进行k折交叉验证对比，GR‑RTRMC在大多数场景下取得最低RMSE（比RTRMC低约15%–30%，在天气数据的Block/Spread场景分别为0.45/0.43°C；在法国数据为0.57/0.54°C；在MovieLens为0.942/0.913），但计算时间约为RTRMC的2–3倍

**⚠️ 局限性**

主要局限在于对高度局部化的气象事件（如雷暴、突发降雨）模型不够鲁棒，导致产生异常振荡或错误平滑；此外，图构造参数和正则化权重对性能敏感，需要额外的超参搜索与多变量扩展

---

## 297. Portable Acceleration of Learning With Errors KEMs for Post-Quantum Cryptography

**arXiv ID:** 2607.09541 | [PDF](https://arxiv.org/pdf/2607.09541v1)

**作者:** Tiziana Liberati `[一作]` (E4 Computer Engineering SpA), Marco Pedicini `[通讯]` (Roma Tre University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于OpenMP Target的可移植GPU实现，完整地在NVIDIA GH200、A100、AMD MI300X与MI300A上实现了LWE‑基KEM，并将CUDA版RNGonGPU随机数生成器移植至HIP以实现GPU全驻留随机数生成。

**💡 创新点**

创新点在于：①通过OpenMP Target与HIP统一源代码实现跨供应商GPU的源级可移植性；②在GPU上完整执行密钥生成、加密/解密及随机数生成，显著减少CPU–GPU数据传输；③系统评估四种GPU架构的性能与能耗，揭示统一内存系统的性能瓶颈。

**🔧 技术方法**

使用技术包括：OpenMP Target指令集、HIP互操作、RNGonGPU AES‑256 DRBG、密钥生成/FO变换的矩阵向量运算、NVIDIA Nsight Systems与ROCm Systems Profiler进行性能剖析，以及GPU功率/温度监测工具进行能耗测量。

**📊 数据集**

实验使用LWE参数集（n从32到16384/32768，迭代次数N=100或1000）自行生成的随机矩阵和向量，未使用公开数据集。

**📈 对比分析**

通过与72线程多核CPU（OpenMP）基线以及不同GPU平台（GH200、MI300X、A100、MI300A）比较执行时间和能耗；结果显示GPU offloading可获得两位数以上加速，GH200最快，MI300X仅慢1.4×，MI300A慢1.9×；能耗方面GH200比MI300X低约2.5倍。

**⚠️ 局限性**

局限性包括：AMD MI300A因统一CPU–GPU内存争用导致性能下降；HIP RNG后台内存分配产生额外延迟；OpenMP编译后端差异影响性能；在统一内存架构下，CPU与GPU的并行内存访问仍会互相竞争；缺乏多GPU和分布式扩展的评估。

---

## 298. TSAI-MetaFraud: A Benchmark Dataset for Financial Fraud Transaction and Behavioral Risk Detection in Metaverse Ecosystems

**arXiv ID:** 2607.09528 | [PDF](https://arxiv.org/pdf/2607.09528v1)

**作者:** Refat Ishrak Hemel `[一作]` (University of New Brunswick), Roozbeh Razavi-Far `[通讯]` (University of New Brunswick)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了多模态、跨任务的元宇宙欺诈检测基准数据集，并定义了四个评测任务（交易欺诈检测、跨模态节点分类、时序链路预测、弱监督欺诈检测）。

**💡 创新点**

创新点包括①将行为日志、金融交易和图结构统一为单一数据集；②在同一数据上支持多任务评测；③通过仿真生成具有真实标签的行为和交易记录，实现可复现的基准。

**🔧 技术方法**

技术手段包括：OpenSimulator 仿真环境、Keystroke 动态生成与 Gaussian KDE、GraphSAGE（异构 GNN）、传统机器学习（LR、RF、XGBoost）以及弱监督标签传播。

**📊 数据集**

使用的数据集是 TSAI‑MetaFraud 本身（936 个用户、74,671 条交易、230k 行为事件），并以图形和表格两种格式发布。

**📈 对比分析**

基线对比采用传统 ML 与 GNN；行为欺诈 F1 最高达 0.93，财务欺诈表现差；GNN 在多类任务上优于传统模型，弱监督下宏观 F1 仅约 0.23，表明任务难度大。

**⚠️ 局限性**

局限性：仅基于单一 OpenSimulator 仿真，欺诈类型有限，严重类别不平衡且标签稀缺，难以保证在真实元宇宙环境中的泛化能力。

---

## 299. How Mobile Gas Sensor Trajectories Govern Hydrogen Leak Detection: A Safety Gap in Manual Leak Inspection of Hydrogen System Components

**arXiv ID:** 2607.09527 | [PDF](https://arxiv.org/pdf/2607.09527v1)

**作者:** Christian Masuhr `[一作]` (Hamburg University of Technology), Thorsten Schüppstuhl `[通讯]` (Hamburg University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了氢气泄漏检测中手动探针扫描轨迹的运动学对检测可靠性的影响，并在机器人试验台上量化了不同轨迹、速度、探针方向与检测灵敏度之间的关系。

**💡 创新点**

创新点在于：①首次在小尺度泄漏环境下系统地量化扫描速度与探针姿态对探测信号的影响；②提出并验证了基于传感器响应时间的速度衰减模型；③基于实验结果制定了几何特定的轨迹规则和通用运动学约束；④开发了可从3D模型自动生成符合规则的检测路径的试点管道。

**🔧 技术方法**

技术包括：机器人协作操控台、三维网格静态浓度测量、动态轨迹扫描、传感器时间常数测定、首阶响应模型、实验设计（分层实验）、规则引擎与路径规划（基于Traveling‑Salesman思路）。

**📊 数据集**

数据集：三种典型泄漏几何（1 mm 圆孔、1×20 mm 划痕、0.1 mm 环缝）和三种泄漏率（20、50、100 ml/min）；使用的探针为 Dräger X‑am 8000（XXS‑H2）和 Pfeiffer ASM 310；静态测量节点以1 cm间距（近场0.5 cm）展开；动态轨迹在10、20、30 mm/s速度下各重复5次。

**📈 对比分析**

通过比较不同轨迹（直线 vs. 侧向冲刺）与速度对检测阈值的影响，实验发现直线扫描在垂直管道上高速度时漏检率可达40%；而侧向冲刺轨迹在所有速度下均保持高可靠性。速度衰减模型与实验数据吻合良好，说明模型可用于预测并设定安全扫描速度。

**⚠️ 局限性**

限制：①仅在参考几何与受限环境下验证，未考虑真实装配中的气流扰动；②缺乏人机试验来评估操作误差；③只对两种传感器进行动态验证，未完成所有常用设备的完整静态/动态建模；④管道生成管线依赖预先标注的组件，尚未支持全新几何的自动识别；⑤实验次数有限，未给出统计置信区间。

---

## 300. Epilepsy Online Social Support: Characterizing Topics and Challenges Shared in the r/Epilepsy Community

**arXiv ID:** 2607.09523 | [PDF](https://arxiv.org/pdf/2607.09523v1)

**作者:** Jessica Y. Medina `[一作]` (Drexel University), Afsaneh Razi `[通讯]` (Drexel University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对23,944条r/Epilepsy子版块帖子进行主题建模、主题分析与心理语言分析，揭示了癫痫患者在网络社区中讨论的主要议题与挑战。

**💡 创新点**

创新点在于将LDA主题模型与LIWC心理语言分析相结合，系统构建了五大主题框架并识别各主题对应的情感、认知等心理语言特征，为后续技术干预提供了多维设计依据。

**🔧 技术方法**

使用的技术包括：Latent Dirichlet Allocation (LDA) 主题建模、BERTopic 对比实验、主题分析（定性分析）以及 Linguistic Inquiry and Word Count (LIWC) 进行心理语言特征提取。

**📊 数据集**

数据集为从2010年起至2023年3月的r/Epilepsy subreddit公开帖子，共23,944条，已剔除被删除或移除的内容，并匿名化作者信息。

**📈 对比分析**

作者通过与BERTopic对比，发现LDA在主题一致性（coherence 0.4738）与可解释性上更优；未给出传统机器学习指标，主要以主题一致性与可解释性评估为主。

**⚠️ 局限性**

研究局限在于仅聚焦单一匿名社交平台，无法验证用户属性与行为差异，缺乏对比实验与干预评估，且结果可能不具普适性，对其他语言或地区的适用性未作探讨。

---

## 301. SAGEAgent: A Self-Evolving Agent for Cost-Aware Modality Acquisition in Multimodal Survival Prediction

**arXiv ID:** 2607.09521 | [PDF](https://arxiv.org/pdf/2607.09521v1)

**作者:** Chongyu Qu `[一作]` (Vanderbilt University), Yuankai Huo `[通讯]` (Vanderbilt University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出SAGEAgent，一个自我进化的LLM驱动的临床决策代理，能根据患者诊断状态动态决定是否获取下一项诊断模态，以在不牺牲生存预测准确性的前提下降低诊断负担。

**💡 创新点**

创新点在于把成本意识的模态获取问题建模为序列决策，并引入冻结的LLM、可解释的临床工具、回忆性与语义记忆的双重自我进化机制，完全无梯度更新即可学习决策规则。

**🔧 技术方法**

采用冻结的大型语言模型（Qwen-2.5-7B-Instruct）、Transformer多模态编码器、Cox比例风险模型、校准不确定性头，以及FAISS检索和反射式规则学习。

**📊 数据集**

使用TCGA-LGG/GBM与BraTS三组脑胶质瘤患者，共962例，完整模态患者170例，涵盖人群、影像、病理和基因组四个模态。

**📈 对比分析**

与静态融合、RL与LLM基线比较，SAGEAgent在C-index 0.813（仅比完整模态低0.012）同时将诊断负担降至0.451，降低55%，超越RL方法的准确率与成本权衡。

**⚠️ 局限性**

局限在于仅在单中心胶质瘤数据集验证，缺乏多中心和前瞻性临床试验，且部分学习到的规则在不同fold间不稳定，难以普适化。

---

## 302. Seeing is Free, Speaking is Not: Uncovering the True Energy Bottleneck in Edge VLM Inference

**arXiv ID:** 2607.09520 | [PDF](https://arxiv.org/pdf/2607.09520v1)

**作者:** Junfei Zhan `[一作]` (University of Pennsylvania), Tengjiao He `[通讯]` (Jinan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对边缘设备上多模态视觉‑语言模型（VLM）推理进行系统能耗分析，涵盖五个模型、三类架构、四种分辨率、两种硬件平台，提出能耗分解模型并给出部署指南。

**💡 创新点**

首次在边缘GPU上系统测量多模态推理能耗，发现平均功耗是模型固有常数，输出令牌成本是能耗主导因素，视觉令牌削减效果极限小；构建低维能耗预测模型并提供能耗削减策略。

**🔧 技术方法**

使用 HWiNFO64 / INA3221 采样功耗、llama‑server (llama.cpp) 推理、线性回归分析、能耗分解公式、交互项预测模型。

**📊 数据集**

COCO 2017 验证集（图像+注释），按对象计数分层划分图像复杂度；利用不同提示生成不同输出长度。

**📈 对比分析**

与视觉令牌剪枝、量化等传统效率方法对比，显示控制输出长度可节能达97%，视觉令牌删除仅约10%；在 7B、8B 模型上验证规律，并给出能耗预测 MAPE 10‑13%。

**⚠️ 局限性**

仅研究单样本推理，未覆盖批量或并发推理；仅针对 1‑4B 参数模型；硬件平台有限，未评估 CPU/NPU 等其他边缘设备的能耗差异。

---

## 303. DemoBridge: A Simulation-in-the-Loop Toolkit for Single-View Human Demonstration Retargeting

**arXiv ID:** 2607.09519 | [PDF](https://arxiv.org/pdf/2607.09519v1)

**作者:** Zehao Wang `[一作]` (Ku Leuven), Rahaf Aljundi `[通讯]` (Toyota Motor Europe)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

将单视角人类手部演示转化为可执行、物理验证的机器人臂轨迹。

**💡 创新点**

提出单个碰撞感知轨迹优化器，结合全臂规划与物理验证，自动推断抓取时序并实现无注释的 simulation‑in‑the‑loop 重定向。

**🔧 技术方法**

采用基于 PyRoKi 的非线性最小二乘轨迹优化、全局采样 RRT、Isaac Sim 物理模拟以及 MediaPipe/RecGen/FusionPose 等感知后端。

**📊 数据集**

使用从真实桌面场景拍摄的十条演示（Mug‑on‑plate、Cup‑stack、Pour‑milk）以及基于程序生成的 50 场合成场景进行基准测试。

**📈 对比分析**

与单帧 IK、关键帧插值方法 YOTO、RoboWheel 进行对比，DemoBridge 在合成基准中碰撞率 < 2%、末端误差 ≈ 3 mm、路径相似度 ≥ 0.78；在真实任务中成功率达到 80%–90%，显著高于基线。

**⚠️ 局限性**

主要局限在物体 6D 姿态跟踪的鲁棒性以及仅支持单臂运动，未覆盖双手协同与更复杂交互。

---

## 304. One-Shot Multimodal Learning from Demonstration with Force-Constrained Elastic Maps

**arXiv ID:** 2607.09515 | [PDF](https://arxiv.org/pdf/2607.09515v1)

**作者:** Brendan Hertel `[一作]` (University of Massachusetts Lowell), Reza Azadeh `[通讯]` (University of Massachusetts Lowell)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种一-shot多模态学习演示框架，能在演示中同时学习运动轨迹和接触力，并通过分割、编码和重现实现更安全、更可靠的机器人操作

**💡 创新点**

创新点在于：1）引入可加权的多模态概率分割方法，自动识别并利用力、位姿等多源信息；2）在弹性映射（elastic maps）中加入力约束，并提供凸优化求解；3）仅需单次演示即可完成力约束任务，兼容不同力传感配置

**🔧 技术方法**

使用的技术包括：多模态概率分割（基于三阶导数峰值与加权概率融合），弹性映射的扩展（力约束与凸优化），以及期望‑最大化算法求解节点位置；低层采用传统力/位姿混合控制实现轨迹跟踪

**📊 数据集**

使用了两套真实演示数据集：UR5e+Robotiq 2f‑85（腕部力/扭矩传感）和Kinova Gen3+Openhand Model O（指尖力传感），共计五项真实任务（抓取、按压、擦拭、弹跳等），并在公开抓取数据集（GBTI‑Grasping）中验证

**📈 对比分析**

通过与传统仅位姿分割、统一权重分割、以及混合力/位置控制等基线对比，实验显示：平均最大力下降~70‑80%，平均力下降~30‑66%；分割准确率提高至3–4段；重现成功率保持100%；与手动力控制相比，最大/平均力均最低（仅Z轴略高）

**⚠️ 局限性**

局限性包括：1）多模态分割权重需人工调节；2）仅在演示阶段使用力信息，执行时未实时监测；3）对视觉检测误差敏感；4）部分任务仍出现偶发力尖峰，需进一步约束优化

---

## 305. All Explanations are Wrong, But Many Are Useful: Exploring the Rashomon Explanation Set with Large Language Models

**arXiv ID:** 2607.09502 | [PDF](https://arxiv.org/pdf/2607.09502v1)

**作者:** Pan Li `[一作]` `[通讯]` (Georgia Tech), Pan Li (Georgia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出 Rashomon Explanation 概念，并实现 RashomonLLM，一种基于 LLM 的解释–预测–反思（EPR）工作流，用自然语言生成解释集并将其嵌入模型输入，以实现解释与预测的互补提升。

**💡 创新点**

创新点包括：① 将单一解释视为“所有解释都错”，而是构建 Rashomon 解释集合；② 通过 LLM 生成可解释的自然语言规则；③ 在 EPR 循环中将解释与预测耦合，并引入双循环学习以动态修正解释；④ 采用特征随机丢弃、批量 LLM 学习等机制提升鲁棒性和可扩展性；⑤ 给出完整理论证明，证明 Rashomon 解释集合非空且解释质量可上界预测性能。

**🔧 技术方法**

核心技术：大型语言模型（GPT‑4o、Qwen3）、EPR 三代理（Explanation、Prediction、Reflection）框架、批量 LLM 学习、特征 dropout、解释聚合与总结、基于嵌入的收敛与覆盖分析，以及与传统 XAI（LIME、SHAP）和深度模型（TabPFN、TabTransformer、DCNv2 等）的对比。

**📊 数据集**

实验数据集包括：① 合成信用评分仿真数据；② 银行流失（Bank Churn）二分类；③ HCT 生存期（regression）；④ 大规模直播平台 KuaiLive 的点击率（CTR）日志（约 1.7 亿条交互记录）。

**📈 对比分析**

方法与多种基线比较：传统可解释模型（Sparse Decision Tree、Sparse AutoEncoder）、LLM 单发解释、Tabular 深度模型（TabPFN、TabTransformer）、行业 CTR 模型（Wide & Deep、DeepFM、DCNv2 等）。RashomonLLM 在所有基准上均显著提升预测准确率（如 Bank Churn AUC +0.02、HCT RMSE -5% 及 KuaiLive CTR AUC +0.02），并在单删除、随机化、综合性、充分性等多项解释可信度指标上优于对手。实验还通过消融、温度、批量大小、随机种子、时间拆分等鲁棒性检验验证其稳定性。

**⚠️ 局限性**

局限性：① 依赖大规模 LLM，计算成本高且需防止模型记忆/泄漏；② 目前仅在结构化表格数据上验证，未对图像、文本等非结构化数据展开；③ 解释生成仍可能产生幻觉，需要进一步的真实性保障；④ 理论假设（如收敛、覆盖）需在更多场景验证；⑤ 对异常分布（极端漂移）与冷启动场景的鲁棒性尚未完全覆盖。

---

## 306. A Truckload of Satoshis: Detecting and Measuring One-Way Arbitrage in the Wild

**arXiv ID:** 2607.09491 | [PDF](https://arxiv.org/pdf/2607.09491v1)

**作者:** Eugenio Nerio Nemmi `[一作]` (Sapienza University of Rome), Alessandro Mei `[通讯]` (Sapienza University of Rome)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过对Binance与Kraken历史匿名现货交易数据进行时序与量值匹配，检测并量化了一种被称为单向套利（OWA）的交易策略。

**💡 创新点**

创新点在于提出一种在无交易者标识的公开数据上推断单向套利序列的可扩展方法，并给出大规模数据集。

**🔧 技术方法**

技术主要是基于交易时间、交换量差和价格差的启发式匹配与阈值优化的算法。

**📊 数据集**

使用的主要数据集为2017-2023年Binance以及2013-2023年Kraken的全量匿名现货交易记录。

**📈 对比分析**

通过阈值精炼和利润基准来过滤候选序列，检测到约4.02亿个OWA序列，累计成交量近190亿美元，净利润约3120万美元；相较于传统理论套利，实际检测展示了真实规模与收益。

**⚠️ 局限性**

局限在于缺乏交易者身份导致只能推断而非确证，无法捕捉所有OWA形式、费用不确定及未观测到的失败案例。

---

## 307. ProofCouncil: An LLM Agent for Solving Open Mathematical Problems

**arXiv ID:** 2607.09474 | [PDF](https://arxiv.org/pdf/2607.09474v1)

**作者:** Johannes Schmitt `[一作]` (ETH Zurich), David Holmes `[通讯]` (Leiden University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一款名为ProofCouncil的数学代理，采用作者‑评论家架构并集成议会LLM与计算节点，在FirstProof挑战中解决6/10个问题，并在30个开放数学问题上取得多项进展。

**💡 创新点**

创新点包括：①将作者‑评论家循环与可选的议会协助和计算节点相结合，②使用条件DAG框架实现灵活的代理系统构建，③开源完整的agent‑building库。

**🔧 技术方法**

采用的技术主要是GPT‑5.5‑Pro LLM、代码执行与网络搜索工具、作者‑评论家迭代、议会LLM辅助、计算节点（CAS），以及基于条件DAG的系统架构。

**📊 数据集**

使用的数据集为FirstProof挑战的10个公开但无解的数学问题，以及从数学研究者收集的30个开放问题。

**📈 对比分析**

与其他参赛团队比较，ProofCouncil在FirstProof中解决了6个问题，表现最佳；在人类评审的30个问题中，5个完全正确、2个有潜力、8个部分进展，显示出较高的有效性。

**⚠️ 局限性**

主要局限在于模型对问题解释的误判导致部分解答对应易化版，成本与时间预算限制，以及仍需人工最终验证结果。

---

## 308. Writing Bug Reports for Software Repair Agents: What Information Matters Most?

**arXiv ID:** 2607.09553 | [PDF](https://arxiv.org/pdf/2607.09553v1)

**作者:** Vincenzo Luigi Bruno `[一作]` (Università della Svizzera italiana), Gabriele Bavota `[通讯]` (Università della Svizzera italiana)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本文中，研究人员对 500 条真实 GitHub 问题（其中 441 条为 bug 报告）进行手工标注，识别并分类了文本中的信息类型（如观察到的行为、期望行为、重现步骤、定位提示、建议修复等），随后利用 mini‑SWE‑agent（基于 GPT‑5‑mini、MiniMax‑M2.5、Gemini‑Flash 3 三种 LLM）在这些报告上进行自动修复实验，构建混合效应二项回归模型探究各信息类型与修复成功率的关联，并通过句子级消融实验进一步验证信息的重要性。

**💡 创新点**

本文的创新点在于：①首次从 agentic‑first 的视角系统评估 bug 报告中各信息类型对 LLM 代理修复效果的影响；②提出“agent‑ready”报告概念，将 bug 报告视为供代理执行的任务规范；③通过观察性分析和消融实验相结合的方法，揭示定位提示和建议修复是提升代理成功率的关键信息，而传统对人类开发者重要的重现步骤等信息对代理的影响有限。

**🔧 技术方法**

技术上，作者使用了：①手工字符级标注工具对报告文本进行信息类型标注；②三种 LLM（GPT‑5‑mini、MiniMax‑M2.5、Gemini‑Flash 3）驱动的 mini‑SWE‑agent 进行自动修复；③混合效应二项回归模型（control 变量包括 issue 长度、金丝贴差异大小、是否包含代码片段、难度等级、仓库随机效应）评估信息类型的增量关联；④句子级消融实验（对 65 条信息完整报告逐一去除宏观类别）验证信息缺失对成功率的影响。

**📊 数据集**

实验所用数据集为 500 条来自 SWE‑Bench（经人工验证的子集）真实 GitHub 问题，其中包含开发者手写的 gold patch 和测试套件。

**📈 对比分析**

对比方法：作者在同一批问题上分别使用三种 LLM 作为代理 backbone，记录每次运行是否生成通过测试的补丁；并在消融实验中对每种宏观信息缺失情形分别进行 3 次重复，统计成功率。结果显示：在所有 LLM 上，含有定位提示或建议修复的报告成功率显著提升（OR>1.5），而去除两者后成功率下降至 0.60 左右；此外，issue 长度越短、patch 规模越小、难度越低的报告，代理成功率越高。

**⚠️ 局限性**

局限性包括：①信息类型的出现被建模为二值变量，未考虑信息质量与细节深度；②标注过程主观性可能影响结果，虽然采用多标注者并计算 Cohen’s κ，但仍可能存在误差；③消融实验通过人工删除文本片段，可能不完全反映真实缺失信息的情况；④研究仅涉及三种 LLM 与单一 agent 架构，结果对其他代理和模型的推广性有限；⑤数据集覆盖的仅是 11 个开源仓库，生态系统多样性不足。

---

## 309. On the Gaussian-Quadratic Rate-Distortion Function for Vector Sources with Individual Distortion Constraints

**arXiv ID:** 2607.09545 | [PDF](https://arxiv.org/pdf/2607.09545v1)

**作者:** Shuao Chen `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了多维高斯源在逐个分量失真约束（individual distortion）下的高斯-二次率失真函数（rate‑distortion function，RDF），揭示了源相关性与RDF之间的定量关系，并给出了在特定相关结构下的闭式解析解。

**💡 创新点**

创新点：
1) 首次给出非满足半正定条件（SDC）的情况的严格下界和上界，补全了传统Hadamard下界的不足；
2) 引入2‑TC（两种相关性）协方差类，并证明其可推广到更一般的(n‑TC)类；
3) 在2‑TC与2‑TD（两种失真约束）条件下，完整划分失真平面为七个区域，给出对应的闭式RDF及最优失真分配；
4) 在等方差（等相关）和相同失真约束时，证明个体约束下的RDF与总失真约束下相同，并给出渐近每分量率。

**🔧 技术方法**

技术方法：
- Hadamard不等式与半正定条件的等价分析；
- 最大判定（Max‑Det）优化与KKT条件推导；
- 逆向水分配原理的标量化版本；
- 协方差矩阵的块结构与逆矩阵运算（Sherman–Morrison–Woodbury、Sylvester判据）；
- 典型子空间与特征值计数的变分表述；
- 解析积分与渐近估计。

**📊 数据集**

数据集：无；本文为纯理论分析，未使用实验数据或公开数据集。

**📈 对比分析**

对比方法：与传统Hadamard下界和数值解（内部点法）进行比较；通过仿真验证闭式RDF与数值结果一致，Hadamard下界往往显著过松甚至为负；在等相关与相同失真约束情形下，数值与解析式完全匹配。

**⚠️ 局限性**

局限性：
- 仅在2‑TC协方差结构（以及其推广的(n‑TC)类）给出闭式RDF，通用协方差下仍缺乏完整解析解；
- 对非SDC情形的上界与下界虽给出，但精度与可计算性仍有限；
- 计算量随N²增长，数值方法在高维时不切实际；
- 论文主要关注高斯-二次模型，非高斯或非二次失真情形尚未覆盖。

---

## 310. Higher-Order Programs with Indefinite Causal Orders: a Linear Approach to Coherent Control of Quantum Processes

**arXiv ID:** 2607.09534 | [PDF](https://arxiv.org/pdf/2607.09534v1)

**作者:** Kathleen Barsse `[一作]` (Université de Lorraine, CNRS, Inria, LORIA), Simon Perdrix `[通讯]` (Université de Lorraine, CNRS, Inria, LORIA)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种支持无限制测量和无限定因果顺序的高阶量子函数式语言，并给出线性类型系统、运算语义与解释语义；证明其安全性、物理可解释性与可表达性，并扩展到非线性与递归。

**💡 创新点**

① 用线性类型约束保证量子控制在完全正映射上可定义，避免非物理过程；② 通过设备引用和记忆函数实现测量结果同步；③ 构造 causal category 解释，使所有well‑typed程序天然物理合法；④ 定义 QC‑QCs with memory 的子类，证明语言可实现其全部。

**🔧 技术方法**

量子 λ 计算、线性类型理论、对称单子闭（SMCC）与 compact‑closed 结构、Causal category 构造、完全正映射、完全正超图、可证明的安全性与一致性。

**📊 数据集**

无实验数据集，纯理论推导与证明；示例包括量子开关、CNOT、测量控制等。

**📈 对比分析**

通过形式化证明（类型安全、进展、唯一规范形）验证语义一致；表达性通过构造性证明表明可实现所有一阶量子通道及 QC‑QCs with memory；无数值实验，性能以理论可实现性和多态性衡量。

**⚠️ 局限性**

① 仅覆盖 QC‑QCs with memory，无法实现所有一般 QC‑QCs；② 初始语言为纯线性，需额外扩展以支持多次查询与递归；③ 对于更高阶超图的完整物理性判定仍未完全自动化；④ 设备引用机制在非线性情形下需要进一步改进。

---

## 311. FreyaTTS Technical Report

**arXiv ID:** 2607.09530 | [PDF](https://arxiv.org/pdf/2607.09530v1)

**作者:** Ahmet Erdem Pamuk `[一作]`, Mustafa Yavuz `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个无词元、非自回归的文本到语音模型，能够在单 GPU 甚至 CPU 上实时生成高质量的土耳其语对话语音。

**💡 创新点**

创新点在于：① 直接使用 92 字符的字符词表，无需音素化或 BPE；② 在冻结的 AudioVAE2 连续潜在空间内训练 183.2M 参数的条件流匹配 Diffusion Transformer；③ 采用两阶段后训练（单声道声学锁定 + 短句覆盖）将多说话人预训练模型转化为可靠的单声道生产模型。

**🔧 技术方法**

技术核心包括：流匹配目标、Euler ODE 解算器、AdaLN‑zero 以及 ConvNeXt‑1D 文本编码器；使用冻结的 AudioVAE2 进行 16 kHz 编码、48 kHz 解码；实现了全并行的非自回归推理，消除自回归误差累积。

**📊 数据集**

训练数据为内部规模大、质量高的多说话人土耳其语语料（含数值、对话句子等），随后使用单说话人专业语料进行后训练；评估使用公开的 Freya‑TR‑Eval 基准（≈400 句，域中性）。

**📈 对比分析**

在 Freya‑TR‑Eval 上实现 8.0% WER、3.0% CER，优于同类开源系统（XTTS‑v2、F5‑TTS 等）且仅占其 40–55% 参数量；实时因子约 0.14，CPU 上可实时甚至快于实时；在 RTX 4090 上显存仅 1.5 GB，吞吐量高。

**⚠️ 局限性**

局限包括：① 由于使用 16 kHz 编码，音频上限为窄带；② 对数字、日期等高密度数值的处理仍需在前端展开；③ 与已实现音素化的 VITS 基线相比，整体错误率仍略高；④ 当前模型仅支持单声道，扩展到多声道或多声源仍需进一步研究。

---

## 312. Artificial Intelligence and the Generative Science of Food Formulation

**arXiv ID:** 2607.09529 | [PDF](https://arxiv.org/pdf/2607.09529v1)

**作者:** Vahidullah Tac `[一作]` (Stanford University), Ellen Kuhl `[通讯]` (Stanford University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文综述并提出了面向生成科学的食品配方框架，整合数字化食品表征与AI能力，并通过可持续性与营养学案例演示其实现。

**💡 创新点**

创新点在于将可持续性与营养等评价指标转化为可直接优化的设计目标，构建统一的生成科学框架，并明确所需数据、模型、基准与自动化路径。

**🔧 技术方法**

使用了预测模型、发现算法、生成式模型（扩散网络）、基础模型、世界模型和代理式AI等多种现代AI技术，并实现多模态信息融合。

**📊 数据集**

利用了Food.com配方数据、USDA FoodData Central、FlavorDB、FooDB、LCA数据库、Taste of the Industry等多源公开数据集。

**📈 对比分析**

目前缺乏统一的多模态基准，本文通过示例（如AI生成的汉堡与麦当劳巨无霸相当且环境影响降低一订单）说明方法可行，但对比指标与系统性评估仍待完善。

**⚠️ 局限性**

限制包括数据碎片化、缺乏标准化评估基准、模型对物理机理的缺失、可解释性与不确定性量化不足以及对开放科研基础设施的高度依赖。

---

## 313. DGSfM: Depth-Guided Scale-Aware Global Structure-from-Motion

**arXiv ID:** 2607.09507 | [PDF](https://arxiv.org/pdf/2607.09507v1)

**作者:** Sithu Aung `[一作]` (Czech Technical University), Zuzana Kukelova `[通讯]` (Czech Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了DGSfM，一种利用单目深度信息进行全局结构光照的SfM框架；

**💡 创新点**

将单目深度作为度量先验，设计深度感知相对位姿求解、视图图过滤、深度一致性过滤、全局尺度平均和深度引导初始化，显著降低尺度不确定性与初始化敏感性；

**🔧 技术方法**

使用深度感知相对位姿求解器（RePoseD风格）、Doppelgangers++视觉歧义消除、三视图一致性剪枝、深度一致性对应点过滤、GLOMAP式全局定位与bundle adjustment；

**📊 数据集**

在ETH3D、IMC2021、LaMAR等公开数据集上进行实验；

**📈 对比分析**

与COLMAP、GLOMAP、InstantSfM、PixSfM、DF‑SfM、Dense‑SfM、VGGT、Pi3、DepthAnything3等基线比较，DGSfM在AUC@1、AUC@3、AUC@5等阈值上均优于或等同于最强基线，尤其在稀疏与稠密匹配场景均表现突出；

**⚠️ 局限性**

主要局限在于对单目深度预测的依赖，深度误差可能导致位姿和尺度估计受影响，且未将SfM结果用于后续深度优化。

---

## 314. Terminal Dimension Reduction for Time Series with Applications

**arXiv ID:** 2607.09490 | [PDF](https://arxiv.org/pdf/2607.09490v1)

**作者:** Alexander Munteanu `[一作]` (TU Dortmund), Chris Schwiegelshohn `[通讯]` (Aarhus University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种终端嵌入（terminal embedding）方法，推广到高维时间序列（多边形曲线）中，能够在保持Fréchet距离的前提下实现维数无关的核心集构造；

**💡 创新点**

创新点在于将终端嵌入从点扩展到线段/曲线，设计了面向Fréchet距离的(ε,m,ℓ)终端嵌入，目标维度为O(ℓ ε⁻⁴ log(nm))，并通过该嵌入直接获得最优(ε)核心集；

**🔧 技术方法**

主要技术包括Gaussian随机投影的子空间保持性、正交投影与辅助坐标的分块映射、ε‑网覆盖子空间、以及链式核心集框架的组合；

**📊 数据集**

实验使用加州温室气体浓度真实数据（16条时间序列，2921维，327时间点）以及合成的50维、长度5的查询曲线；

**📈 对比分析**

与标准Johnson‑Lindenstrauss (JL) 和主成分分析 (PCA) 进行对比，测量Fréchet距离的近似比；结果显示TE与JL表现相近，均显著优于PCA，尤其在较低目标维度下效果更佳；

**⚠️ 局限性**

局限性包括：映射非线性且需随机化；构造查询映射计算量大；对极长曲线或大ℓ仍有维度/误差上界限制；适用范围主要集中在线段/多边形曲线。

---

## 315. Streaming with Catalytic Memory

**arXiv ID:** 2607.09475 | [PDF](https://arxiv.org/pdf/2607.09475v1)

**作者:** Tamara Kaplan `[一作]`, Haim Kaplan `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并研究了“催化流”模型，在该模型中算法可以使用一次性恢复的辅助记忆来实现流式计算；在此框架下给出了对频率矩、任意多项式、F0、子图计数和精确重心阈值的多遍流式算法；

**💡 创新点**

通过引入催化记忆与多遍流的组合，证明单遍流不受益，但多遍流可实现多项式的精确评估和频率矩的高效计算；提出对F₂计算的两遍不可达性范畴并突破三遍瓶颈；

**🔧 技术方法**

利用催化指令序列、幂化（powering）技术、二叉树结构、离散导数分析和Catalytic通信模型的下界技巧；

**📊 数据集**

该工作为理论性研究，不依赖具体实验数据集，所有结果均为证明与上界/下界；

**📈 对比分析**

与传统单遍/多遍流算法对比，催化流在多遍情形下实现了极低的常规记忆（O(1)或O(log n)）与可接受的催化记忆（多项式规模）；

**⚠️ 局限性**

局限在于多遍下界仅适用于两遍且对算法结构有限制；无法给出所有问题的最优下界；并未探讨实际实现与噪声鲁棒性。

---

## 316. A combinatorial framework for clustering graph states: Algorithms and hardness for rank-integrity

**arXiv ID:** 2607.09469 | [PDF](https://arxiv.org/pdf/2607.09469v1)

**作者:** Romain Bourneuf `[一作]` (University of Bordeaux), Stéphan Thomassé `[通讯]` (University of Lyon)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种新的图态距离定义，研究在量子网络中使用少量辅助量子比特将一种图态转化为另一种图态所需的最小辅助比特数，并通过图最小化（vertex‑minor）和矩阵秩方法给出图形描述；进一步定义并求解“高纠缠簇”(ancilla integrity) 与“秩完整性(rank integrity)”问题，并给出FPT算法、1‑ancilla 情况的 O(n⁶) 具体实现以及证明这些问题的 W[1]‑难度。

**💡 创新点**

创新点在于：① 用量子网络中的辅助比特数量量化图态之间的可预备距离；② 将该距离与图的 vertex‑minor、局部补全（local complementation）和矩阵秩相联系；③ 引入 ancilla integrity 与 rank integrity 两个新的参数，并证明二者在量化上互相紧等；④ 给出针对 1‑ancilla 的多项式算法以及一般 k 的 FPT 方案；⑤ 通过构造 Sylvester‑Hadamard 图和顶点‑边偶联图完成 W[1]‑难度的归约。

**🔧 技术方法**

主要技术包括：图最小化与局部补全理论、矩阵秩与 GF(2) 上的线性代数、分裂分解与动态规划、基于分辨列表平衡（weighted list balancing）的组合优化、复杂度理论的参数化归约、Sylvester‑Hadamard 结构用于构造鲁棒连通图、顶点‑边偶联矩阵与矩阵补全难度分析。

**📊 数据集**

本文为理论研究，未使用具体数据集；所有结果均为定理证明与算法设计。

**📈 对比分析**

与已有的图编辑距离与连通度提升问题对比，本文给出了量子网络视角的距离定义，并通过 O(n⁶) 算法实现 1‑ancilla 的求解；在 k=1 时实现多项式时间；在一般 k 时提供 n^{f(k)} 的 FPT 算法；但在大多数参数下问题被证明为 W[1]‑难度，说明无法期望得到多项式时间解。

**⚠️ 局限性**

局限性：仅在 k=1 或 k 较小的情况下能得到多项式或 FPT 方案；一般 k 的算法常数与指数随 k 增大迅速，实际实现难度高；证明的 W[1]‑难度说明问题在参数 k 上不可避免地不具可分离性；算法主要针对理论框架，缺乏对真实量子网络实例的实验验证。

---

## 317. Beyond Fixed Representations: The Vocabulary and Verifier Gaps in Open-Ended AI

**arXiv ID:** 2607.09560 | [PDF](https://arxiv.org/pdf/2607.09560v1)

**作者:** Yuan Cao `[一作]`, Haiqian Yang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文探讨了AI系统从在固定表征框架中求解转向能够自发扩展表征空间、创造新概念并自主评估其价值的开放式创新能力。

**💡 创新点**

提出了词汇缺口与验证缺口的两个关键障碍，构建了创新自主性阶梯框架，并给出目标函数、数据格式、持续记忆与自适应验证等实现路径。

**🔧 技术方法**

主要技术思路包括基于认知差异降低的框架、抽象/类比/组合等认知变换、可持续记忆模块与自适应验证器设计。

**📊 数据集**

论文并未在特定数据集上进行实验，而是基于现有的LLM、程序合成与自动科研系统等案例进行理论分析。

**📈 对比分析**

由于是概念性研究，没有提供数值比较；所述方法以理论分析与已有系统的定位（L0-L3）为依据，未给出性能指标。

**⚠️ 局限性**

主要局限在缺乏实证验证、可扩展性与安全性评估，以及如何在实际系统中实现自我扩展评估器的技术细节。

---

## 318. A novel robust mixed integer linear programming model for index tracking problem under no rebalancing: heuristic optimization approach

**arXiv ID:** 2607.09556 | [PDF](https://arxiv.org/pdf/2607.09556v1)

**作者:** Danial Ramezani `[一作]` (Kharazmi University), Mohamadreza Dehghani Ahmadabad `[通讯]` (Kharazmi University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种鲁棒混合整数线性规划模型并设计了基于遗传算法与局部分支的GALB启发式，用于在不需要再平衡的情况下实现指数跟踪。

**💡 创新点**

创新点在于将相关性不确定性鲁棒化与权重优化结合，构建了能够在长期无再平衡条件下实现低跟踪误差的模型；同时首次将局部分支与遗传算法融合成新的启发式求解框架。

**🔧 技术方法**

采用混合整数线性规划（MILP）建模、遗传算法（GA）作为全局搜索核心、局部分支（Local Branching）实现局部改进，以及与CPLEX等商业求解器的对比。

**📊 数据集**

使用OR‑Library提供的五个主流指数数据（Hang Seng、DAX、FTSE 100、S&P 100、Nikkei 225）以及Yahoo Finance下载的日频行情。

**📈 对比分析**

通过在多种Γ值（不确定性程度）下，用CPLX求解与GALB在相同时间限制（1800 s vs. 120–300 s）下的最优值、MAD、RMSE、β、相关系数等指标进行对比；实验表明GALB在大多数情形下能够在更短时间内得到与CPLX相当或更优的解。

**⚠️ 局限性**

局限性包括：对极大规模资产组合的可扩展性尚未充分验证；模型假设相关性在不同市场环境下的稳健性可能不足；再平衡间隔过长在剧烈波动时可能导致跟踪误差累积。

---

## 319. Statistically Undetectable Backdoors in Deep Neural Networks

**arXiv ID:** 2607.09532 | [PDF](https://arxiv.org/pdf/2607.09532v1)

**作者:** Andrej Bogdanov `[一作]` (University of Ottawa), Neekon Vafa `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在满足特定结构的深度前馈网络（第一层是随机高斯压缩矩阵，其余层双 Lipschitz、输入离散）下，构造了统计上不可检测的后门，该后门赋予模型训练者在生成 invariance‑based 对抗样本（即碰撞对）方面指数级的优势。

**💡 创新点**

创新点包括：① 在白盒环境中实现统计不可检测的后门；② 通过压缩矩阵的后门向量实现 invariance‑based 对抗样本；③ 证明后门强度可达到 2^n/m/β_upper，几乎不影响模型功能；④ 结合 Johnson‑Lindenstrauss 嵌入与加密难题，提供可验证的模型归属证明。

**🔧 技术方法**

使用的技术包括：高斯压缩矩阵的拒绝采样与条件采样；二阶矩浓度分析、Renyi 散度与 Pinsker 不等式；基于一阶/多元可逆函数与 LWE 的加密假设；双 Lipschitz 层与随机特征学习；以及零知识证明编译的验证算法。

**📊 数据集**

实验所用数据集为 Fashion‑MNIST，构建了 784→256→512→1024→2048 的嵌入网络，输入像素为 0–255 的离散整数。

**📈 对比分析**

与无后门模型比较时，后门模型在保持 89% 近似准确率的同时，后门强度约 10^9（理论上可达 2^n/m），且对模型精度影响极小（带后门 86.5%）。实验表明 LLL、随机、热解等算法无法恢复植入向量，验证后门的不可破性。

**⚠️ 局限性**

局限性包括：仅适用于满足第一层压缩高斯、后续双 Lipschitz、输入离散等约束的网络；后门强度受压缩比例与 κ 参数限制；统计不可检测性在 κ 过小时可能失效；实验规模有限（n≤50），更高维度的表现尚待验证；实现仅为概念演示，实际部署与性能评估仍需进一步研究。

---

## 320. Short Graph Sketches Suffice for Error-resilient Leader Verification in CONGEST

**arXiv ID:** 2607.09522 | [PDF](https://arxiv.org/pdf/2607.09522v1)

**作者:** Pawel Garncarek `[一作]`, Subhajit Pramanick `[通讯]` (University of Wrocław)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对树形网络，在CONGEST模型下设计了一个误差容忍的唯一领袖验证算法，并证明了其正确性和复杂度。

**💡 创新点**

创新点在于提出了紧凑的图形草图（local graph sketch）技术，能够在仅 O(ε²) 轮通信内获得足够信息完成验证，同时给出了证明任意算法必须查看超过 ε 路径的下界。

**🔧 技术方法**

主要技术包括：图形草图压缩、想象树（imagined trees）与想象认证（imagined certificates）概念、距离标签法、路径序列可达性检查以及分阶段信息传播。

**📊 数据集**

使用理论上的任意树结构（无特定真实数据集），通过构造路径实例进行证明。

**📈 对比分析**

与传统需要完整邻域信息的 LOCAL 方案对比，所提出算法在 CONGEST 下实现 O(ε²) 轮、每节点 O(ε² log n) 位的通信量，已达到错误容忍验证的最优阶；下界证明了此阶不可被压缩。

**⚠️ 局限性**

局限性在于仅适用于树形网络，无法直接扩展到包含环的图；草图的尺寸和通信轮数仍随 ε² 成长，对大 ε 情况的效率有限。

---

## 321. Failure as a Process: An Anatomy of CLI Coding Agent Trajectories

**arXiv ID:** 2607.09510 | [PDF](https://arxiv.org/pdf/2607.09510v1)

**作者:** Xiangxin Zhao `[一作]` (University College London), He Ye `[通讯]` (University College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对终端交互式LLM编码代理的执行轨迹进行大规模实验，首次把失败视为一个动态过程进行研究。

**💡 创新点**

提出了三阶段失败时间点（决策错误、锁定、可观测信号）框架，并构建了包含9,000+执行步骤的公开注释数据集；从根因、恢复行为和跨系统差异等多维度揭示失败的根源主要是认知错误，且错误往往在轨迹早期发生并在随后一段时间内保持隐蔽。

**🔧 技术方法**

利用LLM辅助的注释流水线（Claude Opus 4.6 生成草稿，人工复核）、自定义失败时间戳标注、根因分类、恢复行为标注，以及基于轨迹前缀的在线监测模型。

**📊 数据集**

使用 Terminal‑Bench 240 任务的 89 个子任务、21 个模型-框架组合（MiniSWE、OpenHands、Terminus2 与 7 大前沿模型）共计 1,794 条完整有效轨迹。

**📈 对比分析**

对比 21 种模型-框架组合的通过率（19%–45%）和错误类型分布，发现所有系统中认知错误占比均高于 44%，并通过可观测信号与恢复行为的统计量量化失败过程；与传统仅基于最终成功率的评估相比，提出的过程式评估能捕捉到早期错误和隐藏的失败窗口。

**⚠️ 局限性**

主要限制包括：① 只涵盖 Terminal‑Bench 任务，可能对其他类型的终端任务存在偏差；② 仅对 89 个任务进行完整覆盖，仍难以覆盖更广泛的任务空间；③ 失败判定与恢复行为的标注仍依赖人工判断，尽管一致性高，但仍存在主观性；④ 研究侧重失败过程，未深入探究如何主动干预或修正错误的具体机制。

---

## 322. Normalisation-Based Likelihood Ratio Estimation for Forensic Authorship Verification

**arXiv ID:** 2607.09501 | [PDF](https://arxiv.org/pdf/2607.09501v1)

**作者:** Sadie Barlow `[一作]` (University of Manchester), Edoardo Manino `[通讯]` (University of Manchester)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并评估了两种无需训练校准模型即可生成可信似然比的 LambdaG 归一化方法（平方根校正和 Hapax 校正），用于作者身份验证。

**💡 创新点**

创新点在于用简单归一化替代传统逻辑回归校准，消除对大量校准数据的依赖，降低时间与复杂度，同时保持与现有最佳方法相当的性能。

**🔧 技术方法**

技术包括 LambdaG 语言模型、两种归一化公式（对文本长度取平方根、按 hapax 计数比例调整）以及对比的逻辑回归校准。

**📊 数据集**

使用了十五个多样化语料库（学术论文、新闻、博客、评论、维基讨论、论坛、电子邮件、聊天、推文、短信等），并在部分语料库中对文本长度（100–9500 tokens）进行系统变化。

**📈 对比分析**

通过 C_llr 指标进行性能比较；Hapax 校正在约 45% 的测试中优于逻辑回归，且大多数情况下误差≤5%，平方根校正整体表现次优。

**⚠️ 局限性**

局限性包括仅在英语文本验证、缺乏对归一化理论的严格证明、在极短文本时 Hapax 可能失效，以及对其他语言或极端数据集的适用性尚未评估。

---

## 323. Shared Selective Persistent Memory for Agentic LLM Systems

**arXiv ID:** 2607.09493 | [PDF](https://arxiv.org/pdf/2607.09493v1)

**作者:** Sanjana Pedada `[一作]` (Apple Inc.), Neelraj Patil `[通讯]` (Apple Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了共享选择性持久化内存架构和零令牌数据刷新机制，构建了基于 Git 的协作工作空间平台，使 LLM 代理在多轮工具使用中保持可重用上下文并支持跨用户共享。

**💡 创新点**

提出四类可持久化的“可重用上下文”（任务规范、数据模式、工具配置、输出约束），并有选择地丢弃会话特定的推理轨迹，实现比全记录更高效的记忆管理；同时实现零令牌数据刷新解耦。

**🔧 技术方法**

采用 Claude Opus 4 LLM、Agentic Loop、MCP/SQL/REST/CSV 多连接器、MongoDB 存储、Git 版本控制、FastAPI 后端、Claude Agent SDK 等技术。

**📊 数据集**

使用 24 个企业内部结构化数据文件（供应链、销售、流程指标）以及四个公开数据集（Superstore Sales、Adult Income、NYC 311、World Bank GDP）。

**📈 对比分析**

对比无记忆、完整历史、选择性记忆三种条件，使用人工评估任务完成率、token 消耗、用户回合数、耗时等指标；选择性记忆取得 96% 完成率、显著降低 token 与回合数；零令牌刷新实现无模型调用、时间缩短 14 倍；摘要驱动生成实现 97–946 倍 token 节省。

**⚠️ 局限性**

记忆划分手工设计，未能自动识别可重用上下文；仅适用于结构化表格数据，无法处理流式或非结构化数据；多连接器各自的失败模式未统一处理；评估缺乏对美学等主观质量的评估；用户研究样本有限；协作仅支持工作空间级共享，细粒度共享尚未实现。

---

## 324. SigLIP-HD by Fine-to-Coarse Supervision

**arXiv ID:** 2607.09488 | [PDF](https://arxiv.org/pdf/2607.09488v1)

**作者:** Lihe Yang `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过一种“细化→粗化”监督机制，将多尺度高分辨率图像的高质量特征迁移到标准分辨率（512px）下的视觉编码器，使其在保持相同推理成本的前提下，提升对细粒度视觉信息的感知能力。

**💡 创新点**

创新点在于：①不依赖额外上采样或复杂后处理，仅通过对预训练 SigLIP‑2 编码器的轻量级微调，实现低分辨率特征对高分辨率特征的仿射；②采用多尺度（512px + 1024px）特征融合与 L1 损失，直接在特征层面进行知识蒸馏；③在保持相同输入分辨率和 token 数量的情况下，显著提升 OCR 及细粒度 VQA 任务表现。

**🔧 技术方法**

核心技术：多尺度输入与滑窗推理、特征下采样 + 均值融合、L1 损失对齐、AdamW 训练、SigLIP‑2 预训练模型、LLaVA 训练框架、AnyRes 方案、不同 LLM（Vicuna‑1.5、Llama‑3.2‑3B、Qwen‑2.5‑7B）融合。

**📊 数据集**

使用 4.5M 来自 Cambrian‑1 的原始图像进行微调；在多种 MLLM 基准上评估：DocVQA、ChartQA、TextVQA、InfoVQA、TextCaps、HRBench、RealWorldQA、GQA、MME Perception、POPE、MMBench、ScienceQA‑IMG、AI2D。

**📈 对比分析**

对比方法：与原始 SigLIP‑2 编码器在同一 LLM、相同输入分辨率下的表现做基线；使用 LLaVA‑1.5 及 LLaVA‑NeXT 的数据集进行 SFT；在 AnyRes 与多 LLM 上进行横向验证。结果显示：OCR 任务提升 3–4.6 分；整体 VQA 性能提升 4.8 分；在 AnyRes 情况下 OCR 进一步提升 3.5 分；在 Qwen‑2.5‑7B 上，DocVQA 提升 1.7 分。

**⚠️ 局限性**

局限性：①仍需依赖预训练编码器，无法彻底替代本地高分辨率推理；②对极低分辨率（如 336px）场景的细粒度感知仍受限；③仅在 512px 的标准分辨率下验证，跨尺度迁移效果未知；④在不同任务与模型上，提升幅度不均衡；⑤训练时需多尺度推理，仍会产生一定额外计算成本。

---

## 325. Foveation-Guided Dynamic Token Selection for Robust and Efficient Vision Transformers

**arXiv ID:** 2607.09480 | [PDF](https://arxiv.org/pdf/2607.09480v1)

**作者:** Ibrahim Batuhan Akkaya `[一作]` (NavInfo Europe), Elahe Arani `[通讯]` (Eindhoven University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于人眼视觉系统的动态注意力机制——Foveated Dynamic Transformer（FDT），在 Vision Transformer 中融合了自适应的“视网膜分辨率”和“注视点选择”，实现更高效、更鲁棒的图像分类。

**💡 创新点**

创新点在于：①将多尺度“视网膜”采样与单次前向传播中的“注视点”选择联合实现，既减少了计算量，又提升了模型对噪声、对抗攻击和自然失真图像的稳健性；②通过引入固定预算约束和 Gumbel‑Softmax 训练策略，让模型在保持高精度的同时仅处理最关键的图像区域。

**🔧 技术方法**

技术手段包括：多层深度可分离卷积实现多尺度特征提取（foveation），单线性层生成二值注视图（fixation），多头自注意力仅在选定标记上计算，预算损失约束，Gumbel‑Softmax 训练实现可微分的二值化。

**📊 数据集**

使用 ImageNet100（包含 100 类图像）进行分类，并在 ImageNet100-C 进行自然腐败实验，使用多种对抗攻击（12 种）和 Tinted‑ImageNet100 评估短路学习鲁棒性。

**📈 对比分析**

与 DeiT‑S 基线相比，FDT 在 50% 预算下实现 81.9% 的 top‑1 纯净精度（高于 80.9%），在对抗攻击上平均提升 27%，自然失真上提升 3%，同时 MAC 下降 34.57%。在不同预算下，FDT 通过自适应计算权衡精度、鲁棒性和效率。

**⚠️ 局限性**

局限性包括：①MAC 下降并不必然转化为实际推理延迟，因多尺度重排与通道拼接产生内存绑定开销；②模型仅在单次前向传播中并行选择注视点，缺乏人眼时间序列的递归与重访机制，不能完全再现人眼的生物学细节。

---

## 326. Active rejection enables reliable generalization of universal machine-learning interatomic potentials

**arXiv ID:** 2607.09456 | [PDF](https://arxiv.org/pdf/2607.09456v1)

**作者:** Mingxiang Luo `[一作]` (Fudan University), Yuqiang Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种自适应多教师路由（ATR）框架，通过对多模型预测结果进行结构层面的可靠性评估与主动拒绝，生成高置信度的r2SCAN级伪标签，构建大规模高精度材料数据集。

**💡 创新点**

创新点在于将教师模型的互补性和不一致性转化为可学习的接受/拒绝决策机制，使用小量真实DFT标注进行校准，并在大规模候选结构中实现主动拒绝，避免低质量伪标签污染。

**🔧 技术方法**

技术实现基于结构描述符、教师身份信息与教师间误差分布特征构建特征集，采用轻量级XGBoost二分类器做路由决策，并在部署时进行阈值与后处理校正。

**📊 数据集**

使用的数据集包括约1.1×10^8个公开候选结构（来自MP、Alexandria、MPTrj、OQMD），1.8×10^4个真实r2SCAN校准样本，以及经过ATR筛选后得到的2.89百万条可追踪r2SCAN伪标签。

**📈 对比分析**

在CHGNet学生模型上进行预训练与微调后，ATR伪标签显著提升了在2,000条hold‑out r2SCAN测试集和MP‑r2SCAN基准上的能量与力MAE，并在三组300 K分子动力学模拟中保持轨道稳定性，显著优于无路由或仅增大数据量的对照组。

**⚠️ 局限性**

局限性包括对校准集规模与分布的敏感性、仅以能量与力阈值定义可靠性、未覆盖应力、磁矩等更丰富的物理约束、以及对极端高温或表面/界面等更复杂体系的验证不足。

---

## 327. Solving the Reachability Problem for Branching Vector Addition Systems via Semilinear Inductive Invariants

**arXiv ID:** 2607.09558 | [PDF](https://arxiv.org/pdf/2607.09558v1)

**作者:** Clotilde Bizière `[一作]` (University of Bordeaux), Grégoire Sutre `[通讯]` (University of Bordeaux)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了分支向量加法系统（BVAS）的可达性问题可判定，完成了该问题三十余年未解的根本性突破；

**💡 创新点**

创新点在于构造安全证据与面剥离定理，利用全周期集合的几何分解和诱导不变式，首次实现对BVAS可达集的半线性描述；

**🔧 技术方法**

核心技术包括诱导不变式、良序化的有向不变子系统、WSVAS（带钻石映射的向量加法系统）、凸多面体几何、面剥离定理以及对角关系的使用；

**📊 数据集**

未使用任何具体实验数据集，全部为理论证明；

**📈 对比分析**

与传统的KLM算法方法进行比较，虽然未给出复杂度上界，但方法证明可达性是可判定的，性能上没有实测数据；

**⚠️ 局限性**

局限性在于缺乏复杂度分析（仅继承Ackermann上界），未覆盖更强的EBVAS模型，也未验证其可达集的几何特性。

---

## 328. PanoWorld: Real-World Panoramic Generation

**arXiv ID:** 2607.09661 | [PDF](https://arxiv.org/pdf/2607.09661v1)

**作者:** Haoyuan Li `[一作]` (Insta360 Research), Lu Qi `[通讯]` (Insta360 Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PanoWorld框架，实现高保真可控全景视频生成，解决长程记忆与物理一致性问题。

**💡 创新点**

创新点在于利用全景投影的旋转等变性将旋转视为几何变换，仅显式学习平移；设计DPRC与GMA两模块并采用三阶段训练流水线。

**🔧 技术方法**

使用Diffusion模型基础，结合LoRA微调、Dense Panoramic Ray‑Conditioning（DPRC）、Geometry‑aware Memory（GMA）、PRoPE位置嵌入以及Causal Forcing实现实时推理。

**📊 数据集**

构建World360数据集，包含70k真实UAV全景视频与50k AirSim360仿真视频，覆盖多高度与多光照条件。

**📈 对比分析**

与Imagine360、Matrix‑3D、OmniRoam等基线对比，FID、PSNR、QA等指标均显著提升，实时推理可在8秒内生成161帧全景视频。

**⚠️ 局限性**

仍受限于训练数据分布，未在极端动态场景或超长时序连续性测试中验证，低光照或极端运动场景仍存在性能波动。

---

## 329. Indirect and Direct AI Scaffolding for Computational Problem Posing: A Pilot Experience Report

**arXiv ID:** 2607.09628 | [PDF](https://arxiv.org/pdf/2607.09628v1)

**作者:** Shayla Sharmin `[一作]` (University of Delaware), Roghayeh Leila Barmaki `[通讯]` (University of Delaware)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了两种基于大语言模型的计算问题构造 scaffolding 系统——一种通过引导性问题提示（Indirect），另一种通过工作示例改进（Direct），并在两种任务开放度不同的情境中进行试点。

**💡 创新点**

首次系统性比较 Indirect 与 Direct LLM scaffolding 对问题质量与学习体验的影响，并提出按任务开放度顺序先用 Indirect 再用 Direct 的实践策略。

**🔧 技术方法**

使用 GPT‑4o 大语言模型与 Prompt 工程实现实时评估与反馈；评价框架基于 Bloom’s Taxonomy 的四个维度；用户体验通过 UEQ‑S、NASA‑TLX 等问卷收集。

**📊 数据集**

使用自生成的“Handshake”与“Making Change”两类计算问题场景，收集参与者提交的原始与修改后问题作为评分样本。

**📈 对比分析**

采用 within‑subject 对照、Latin square 平衡顺序，利用 Wilcoxon 符号秩检验、混合效应回归、UEQ‑S 评分等方法比较两种 scaffolding；结果显示 Direct 在问题质量提升和效率上更佳，而 Indirect 则更能促使深度反思。

**⚠️ 局限性**

局限性包括样本仅为 20 名研究生（多为 TA）且缺乏无 scaffolding 对照，缺少长期学习与保留评估，仅覆盖两种任务场景，且高水平初始问题导致改进空间有限。

---

## 330. Task-Specific Multimodal Question Answering Agents via Confidence Calibration and Incremental Reasoning for QANTA 2026

**arXiv ID:** 2607.09623 | [PDF](https://arxiv.org/pdf/2607.09623v1)

**作者:** Nirjhar Das `[一作]` (Chittagong University of Engineering & Technology), Md. Al-Mamun Provath `[通讯]` (Chittagong University of Engineering & Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了针对 QANTA 2026 竞赛的两代理体系，分别针对 Tossup（逐渐揭示线索）和 Bonus（完整上下文）任务，结合置信度校准、增量推理和多模态证据融合实现高效问答。

**💡 创新点**

创新点在于：①任务专用的两代理架构，满足不同决策目标；②置信度门控与期望值最大化的 Buzz 策略；③针对量化推理的 Numeric Firewall 以及在 Bonus 中引入的 leadin‑aware 推理和人机协作优化；④在有限资源下通过晚期融合实现高效多模态推理。

**🔧 技术方法**

使用技术主要包括：GPT‑4.1‑mini 作为 Tossup 推理模型，GPT‑4.1 作为 Bonus 推理模型；置信度校准机制（输出概率阈值 0.90）；增量式推理与期望值计算；数值门控（Numeric Firewall）；多模态证据路由（文本先行，图像做验证）；结构化输出与简洁解释提升人机采纳率。

**📊 数据集**

使用的数据集为 QANTA 2026 共享挑战提供的 Quizbowl 问题，包含文本线索、可选图片，覆盖文学、历史、科学、艺术和时事等多领域。

**📈 对比分析**

评估指标包括 Tossup 的期望分数、Buzz 位置、精度、Win Rate；Bonus 的 Part Accuracy、Question Accuracy、Bonus Effect、Calibration、Adoption。最终系统整体得分 0.402，Tossup 0.238，Bonus Effect 0.164，位居排行榜第一，明显优于近似系统（整体 0.370）。

**⚠️ 局限性**

局限性主要体现在：①对置信度阈值高度依赖，仍可能在罕见或极端场景下过度保守或过度激进；②对相关实体混淆的处理仍不够鲁棒，特别是早期线索相似度高时易产生错误推测；③多模态融合仍采用简单的晚期拼接，缺乏更细粒度的跨模态对齐与自适应检索；④在极低成本设置下的推理速度与吞吐量仍有限。

---

## 331. Overlapping Unfoldings of Cones and Convex Polyhedra

**arXiv ID:** 2607.09606 | [PDF](https://arxiv.org/pdf/2607.09606v1)

**作者:** MIT CompGeom Group `[一作]`, Joseph O'Rourke `[通讯]` (Smith College)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了凸多面体的展开问题，证明了两点：一是任意给定厚度 t，可以构造出允许非边缘切割的一般展开使重叠厚度为 t；二是存在凸多面体，其边缘展开可以达到任意给定厚度 t，并且所需顶点数为 O(t)。

**💡 创新点**

创新点在于提出并实现了针对“重叠厚度”的逆向展开构造：利用可变曲率锥体与嵌套重叠装置，分别给出了一般展开和边缘展开的可实现性证明，并证明了线性顶点数上界的紧确性。

**🔧 技术方法**

主要技术包括：① 对锥体的角度与滚动展开的几何分析，构造可得到任意厚度的非边缘切割路径；② 设计嵌套重叠装置（Overlap Gadget），通过平行且嵌套的三角形堆叠，递归构造多层重叠，实现任意厚度的边缘展开；③ 证明凸性与切割树的合法性，确保展开后仍为简单多边形。

**📊 数据集**

论文没有使用实验数据集，而是采用纯几何构造与证明，给出了具体的多面体实例和切割路径示意图。

**📈 对比分析**

对比方法主要是理论分析：与之前已知的“重叠厚度不超过面数”的上界比较，证明线性顶点数上界是紧确的。没有数值实验，故未给出性能指标；结论纯粹是存在性与构造性证明。

**⚠️ 局限性**

局限性：
- 只给出存在性构造，未提供多面体的统一生成算法；
- 构造的多面体顶点数随厚度线性增长，实际应用中可能过大；
- 对于具体已知多面体（如正多面体）的最大重叠厚度仍未完全确定。

---

## 332. Agora: Enhancing LLM Agent Reasoning Via Auction-Based Task Allocation

**arXiv ID:** 2607.09600 | [PDF](https://arxiv.org/pdf/2607.09600v1)

**作者:** Kaiji Zhou `[一作]` (University of Birmingham), Yue Feng `[通讯]` (University of Birmingham)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Agora 框架，将推理步骤的分配视为基于信心校准的拍卖，动态选择最合适的专家模型或工具来完成每个子任务；

**💡 创新点**

创新点在于引入激励兼容的拍卖机制和分层信心校准（静态+在线），有效过滤模型的幻觉自信，实现可控的成本-质量权衡；

**🔧 技术方法**

使用的技术包括：LLM 规划器生成任务图，静态校准器（基于嵌入分箱的概率校准），在线动态校准器（logit 线性变换），拍卖公式（confidence^γ - β·cost），以及后向反馈更新 S’；

**📊 数据集**

实验数据集包括 MuSiQue-Ans、MMLU-Pro、SciCode、SPIQA、MathVision；

**📈 对比分析**

与单模型、随机路由、1NN 路由、适应性解算器、混合 LLM、一致性级联、FrugalGPT 等基线对比；在文本推理上，Agora 在 EM、F1、准确率上相较匹配池基线提升 1–3%（例如 MuSiQue EM 43.0%/F1 54.3%），在科学代码和多模态任务上亦能保持竞争或略优；成本-质量可通过 β 参数调节，展示了可调的准确率-成本曲线；

**⚠️ 局限性**

局限性包括：校准的泛化能力受分布漂移影响，拍卖需要可靠的规划拆分，若模型差异不大或同质，则增益有限。

---

## 333. PAC-ACT: Post-training Actor-Critic for Action Chunking Transformers

**arXiv ID:** 2607.09590 | [PDF](https://arxiv.org/pdf/2607.09590v1)

**作者:** Yujie Pang `[一作]` (Southern University of Science and Technology), Zudong Li `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对预训练的 Action Chunking Transformer（ACT）进行后训练，使用强化学习优化其在工业精密接触任务中的表现，提升任务完成率、接触力安全性和执行效率。

**💡 创新点**

① 将步骤级 RL 目标重构为块级决策过程，解决动作块与奖励时间尺度不匹配的结构性矛盾；② 设计 ACT 迁移的 Actor‑Critic 架构，保留 ACT 的视觉编码和 Transformer 生成动作块，同时去除 CVAE 模块；③ 采用混合 KL 约束（行为先验和分布信任域）防止策略偏离预训练分布，保证稳定性。

**🔧 技术方法**

使用 PPO 的块级更新、GAE 估计、混合 KL 正则、对数标准差学习、以及基于视觉+关节角度的多模态输入；网络基于 ResNet‑18+4 层 Transformer 编码器，Actor 输出 K×d 维动作均值，Critic 采用 encoder‑value 结构。

**📊 数据集**

在 LeRobot 记录的工业级接触演示数据上进行预训练，演示规模为 800 条轨迹；随后在 Metal Touch（金属触碰）与 Square Assembly（方块装配）这两个基准任务上进行 RL 微调。演示数据通过 MP4 压缩与 Parquet 存储，方便随机抽取。

**📈 对比分析**

与 ACT、Diffusion Policy、π0.5 等基线以及 3B 参数的 VLA 模型对比。PAC‑ACT 在 Contour 任务的成功率从 60% 提升至 100%，在 Square Assembly 从 51.2% 提升至 98.2%；完成步数平均下降 2.8 倍；接触力峰值从 8452.5 N 降至 120.9 N，超阈值事件减少 46 倍。推理时延 88.1 ms，GPU 内存 2.30 GB，保持低延迟与低占用。

**⚠️ 局限性**

实验仅在仿真环境完成，缺乏真实机器人验证；只评估位置扰动，未系统检验视角、光照或动力学参数变化的鲁棒性；去除 CVAE 可能限制多模态动作分布；需要手工设计奖励函数，自动化奖励生成仍待研究。

---

## 334. Knowledge Graphs and Explainable AI as Complementary Resources for Urban Mining

**arXiv ID:** 2607.09578 | [PDF](https://arxiv.org/pdf/2607.09578v1)

**作者:** Jan Gronewald `[一作]` (German Research Center for Artificial Intelligence), Nijat Mehdiyev `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了四种知识图谱与可解释人工智能（KG–XAI）协同集成模式（Lifting、Constraining、Typing、Revising），并通过在城市矿业预拆评估（Pre‑Demolition Assessment）中的防火门案例来说明如何将这些模式用于构建可审计、可解释且可追溯的决策记录。

**💡 创新点**

创新点在于：①以“可审计性”属性为视角，对KG–XAI集成进行结构化分类，提出四种类型的“补充”模式；②将信息系统的资源互补理论应用于AI与知识图谱的交互，阐明它们如何共同产生四种不可单独实现的可审计特性（可读性、可行性、来源性、可争议性）；③给出基于W3C Linked Building Data栈和Urban Mining Index的操作实例，为建筑物数字化与循环经济决策提供具体方法。

**🔧 技术方法**

使用的技术包括：可解释AI方法（特征归因如SHAP、梯度归因，反事实生成如Wachter‑style或DiCE），知识图谱构建与查询（W3C LBD Stack中的BOT、PRODUCT、PROPS、OPM、Urban Mining Index），以及语义推理与可追溯性标准（PROV‑O、OPM）。这些技术被组织为四个模式的输入与输出，形成可视化的属性状态图和可审计的变更历史。

**📊 数据集**

数据集方面，本文没有使用公开机器学习数据集，而是以W3C Linked Building Data（BOT、PRODUCT、PROPS）和Urban Mining Index为知识图谱底层，并以一张木质防火门的检测结果（检测器给出门、火门细分类别、置信度）作为示例。该示例体现了模型输出与知识图谱属性之间的映射。

**📈 对比分析**

论文并未进行实验性比较或量化评估；其主要贡献是理论框架和示例演示。作者指出，未来工作将把该框架应用于真实的预拆评估案例，以评估KG‑XAI集成对审核人员文档编制、争议提出和决策修订的实际提升效果。

**⚠️ 局限性**

局限性包括：①缺乏实证验证，未证明所提模式在大规模评估中的可行性和性能；②对知识图谱治理和权威机构的支持未给出解决方案，仍需外部审核机制；③仅适用于黑盒模型，不能直接扩展到本质可解释模型；④示例范围有限，未覆盖多样化建筑组件和更复杂的干预情景；⑤对跨实例更新的治理时效性与权限仍需进一步研究。

---

## 335. Impact of Benign Connectivity Variations on Intrusion Detection for Encrypted OPC UA Traffic in Industrial Private 5G Networks

**arXiv ID:** 2607.09659 | [PDF](https://arxiv.org/pdf/2607.09659v1)

**作者:** Song Son Ha `[一作]` (Helmut-Schmidt-University), Gerd Scholl `[通讯]` (Helmut-Schmidt-University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了工业私有5G网络中，对加密OPC UA流量使用机器学习IDS时，正常连接变化（如UE重连、PDU会话重置等）如何导致误报率上升。

**💡 创新点**

发现了连接事件与控制平面（CP）活动在时间上高度相关，说明仅基于上行流量特征的IDS无法区分恶意攻击与合法网络状态变化。

**🔧 技术方法**

采用统计特征提取、CP指示器（注册、状态转换、会话重置等）和四种监督学习模型（LogReg、RF、SVM、XGBoost）构建IDS。

**📊 数据集**

使用由基线、恶意攻击和四类善意连接变化场景构成的PCAP数据集（共约250个文件），每类场景10个文件。

**📈 对比分析**

对不同攻击强度（L1-L3）进行召回率评估，并与基线对照测量误报率；在所有模型中，误报率从3.5–4.9%升至8–9%，表明善意变化显著影响性能。

**⚠️ 局限性**

缺点是IDS未利用CP上下文，导致误报；实验场景有限，未验证更广泛的网络部署；未来需要构建CP感知的检测方法。

---

## 336. Network Analysis with Parametric NetKAT

**arXiv ID:** 2607.09637 | [PDF](https://arxiv.org/pdf/2607.09637v1)

**作者:** Han Xu `[一作]` (Princeton University), David Walker `[通讯]` (Princeton University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `847a60d8-a755-47af-ba5d-c5236b9e3083` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Parametric NetKAT语言，实现网络诊断的枚举查询。

**💡 创新点**

将NetKAT、Relational NetKAT和Weighted NetKAT合并并引入参数，支持布尔组合约束的所有取值集合求解。

**🔧 技术方法**

基于NetKAT自动机的符号编译，使用BDD/ADD进行参数化状态符号化，并改进空检查与加权求和算法。

**📊 数据集**

评估使用Topology Zoo、Alibaba Rela网络流、以及Batfish 200K 行路由配置。

**📈 对比分析**

对比逐个枚举失败场景，Parametric NetKAT 在大规模网络上平均耗时数秒，最大约200秒，速度提升可达数百倍。

**⚠️ 局限性**

仅支持有限参数空间，无法输出无限长追踪或符号权重，且对极大参数组合仍存在爆炸风险。

---

## 337. Kleene Algebra with Transitive Commutativity Conditions

**arXiv ID:** 2607.09635 | [PDF](https://arxiv.org/pdf/2607.09635v1)

**作者:** Han Xu `[一作]` (Princeton University), David Walker `[通讯]` (Princeton University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究Kleene代数与可交换条件（Kleene Algebra with Commutativity Conditions，KA+C）的等价性与可判定性，证明在可交换关系为传递时等价理论与KA^*+C一致；若不传递则不可判定，且不传递时的通用性问题（universality）不可判定。

**💡 创新点**

创新点包括：① 用最小的非传递可交换关系（仅满足 (a,b),(b,c)∈C 但 (a,c)∉C）即可得到不可判定；② 将通用性问题提升为不可判定；③ 通过归一化、因子化和支撑表达式构造，证明KA+C与KA^*+C在传递情形下完全一致，突破了以往仅针对正则语言或特定情况的结论；④ 将判定问题从正则语言映射到Kleene代数，保留星号运算的代数性质。

**🔧 技术方法**

使用了矩阵Kleene代数、Parikh映射、半线性集合的布尔原子分解、因子化（block factorization）、支撑表达式归一化、递归不相容性（recursive inseparability）与图灵机编码，构造了复杂的正则表达式来实现可判定性与不可判定性的证明。

**📊 数据集**

没有使用任何实验数据集；研究完全基于形式化证明和理论构造。

**📈 对比分析**

方法通过与已知的正则语言下的可判定性结果对比，证明了在传递可交换关系下可判定性，且对比了不可判定的基准（Post对应问题或两计数器机）后提升到最小非传递情况。理论结果表明，当可交换关系传递时，判定问题可在多项式时间内完成；若不传递，则问题是不可判定的，没有可行的算法可用于判断。

**⚠️ 局限性**

限制包括：① 证明极其繁复，难以手工检查；② 仅适用于有限字母表且仅讨论单字符的可交换关系；③ 对星号运算的代数处理依赖于Kleene代数的公理，缺乏直接的构造性算法；④ 对实际程序优化或并发模型的直接应用仍需进一步研究。

---

## 338. New Complexity Classes in Locally Checkable Labeling for Local Computation Algorithms

**arXiv ID:** 2607.09626 | [PDF](https://arxiv.org/pdf/2607.09626v1)

**作者:** Sijin Peng `[一作]` `[通讯]` (MIT), Sijin Peng (MIT)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究局部计算算法（LCA）的复杂度分类，给出任意多项式或多项式对数阶的探测复杂度的LCL问题，并通过“堆叠”构造实现。

**💡 创新点**

首次证明在LCA和Rosenbaum‑Suomela（ROST）模型中存在LCL问题达到任意整数 k 的 Θ(log^k n) 以及任意有理指数 p/q∈(0,1] 的 Θ̃(n^{p/q}) 探测复杂度，填补了先前仅知有限离散复杂度点的空白。

**🔧 技术方法**

利用 R‑S 二叉树标签化约束，将已有的 Θ(log n) 与 Θ̃(n^{1/k}) 实例进行层级堆叠，并采用分布式随机游走、采样与二分查找等技术实现上界；下界通过构造特定实例族与 Yao 最优原理给出。

**📊 数据集**

无数据集，研究以理论图模型为基础的抽象问题。

**📈 对比分析**

与 LOCAL 与 DIST 模型中的已知复杂度谱对比，证明 LCA/ROST 模型的谱更细粒度；在理论上实现了多项式密度，即任意两个指数之间都可实现问题，表现为多项式级别的多样性。

**⚠️ 局限性**

局限在于仅考虑常度图、随机化算法、LCL 问题；对确定性 LCA 仍未给出完整谱；实际实现及性能未在实验上验证。

---

## 339. LLM for EDA in Front-End Design: Challenges and Opportunities

**arXiv ID:** 2607.09616 | [PDF](https://arxiv.org/pdf/2607.09616v1)

**作者:** Kangwei Xu `[一作]` (Technical University of Munich), Ulf Schlichtmann `[通讯]` (Technical University of Munich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了大型语言模型（LLM）在芯片前端设计中的应用，阐述了从规格理解到HDL生成、测试平台构建以及高层合成（HLS）等关键任务的进展，并提出了将LLM演进为智能代理（agentic AI）以实现闭环自动化的研究方向。

**💡 创新点**

创新点在于系统化地将LLM与前端EDA流程对接，展示了多种基于LLM的生成与修复框架（如VRank、VFocus、AutoBench、CorrectBench、ConfiBench、HLSRepair等），并对agentic AI在保持语义一致性、工具协同和闭环反馈方面的挑战与机遇进行了前瞻性评估。

**🔧 技术方法**

核心技术包括大语言模型（LLM）生成、检索增强生成（RAG）、基于仿真结果的候选聚类与排名、自我校正与修复循环、Python+HDL混合测试框架、跨阶段知识迁移以及多代理协同设计。

**📊 数据集**

使用的数据集涵盖公开的Verilog/HDL评测集（VerilogEval）、多任务HDL生成/测试基准、C/C++至HLS的对齐数据集以及自定义的测试平台生成数据集，文中通过这些基准对比实验验证方法有效性。

**📈 对比分析**

实验与基线比较显示：VRank将HDL正确率提升10.5%，VFocus提升30.9%；AutoBench的Pass@1提升57%，CorrectBench和ConfiBench分别达到70.13%和72.22%整体通过率；HLSRepair的修复通过率提升23.33%，测试效率提升2.71倍，硬件面积、功耗、延迟平均分别下降约25%、13%和18%。

**⚠️ 局限性**

主要局限包括：LLM在不同设计阶段保持语义一致性的困难、频繁的hallucination导致功能不正确、模型token消耗高导致执行效率低、缺乏高质量对齐硬件数据集、对覆盖闭环的验证不足以及在工业流程中对持久状态管理和可验证闭环的需求尚未完全满足。

---

## 340. Quantum Orchestras: a Concrete Semantics for Recursive Hybrid Programs

**arXiv ID:** 2607.09605 | [PDF](https://arxiv.org/pdf/2607.09605v1)

**作者:** Alex Rice `[一作]` (University of Edinburgh), Robert I. Booth `[通讯]` (University of Oxford)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

提出了Quantum Orchestra Monads来给混合量子-经典程序提供可组合的数学语义；

**💡 创新点**

创新点在于将量子仪器与域理论结合，构造支持无限递归、测量反馈的参数化强单子；

**🔧 技术方法**

使用了量子仪器、von Neumann代数、Scott连续、Kleisli组合、CPS单子等数学工具；

**📊 数据集**

未使用任何数据集；

**📈 对比分析**

未进行实验比较，主要通过理论证明与形式化验证；

**⚠️ 局限性**

局限在未覆盖动态分配在循环中的情况、未给出直接式单子实现以及缺乏与现有语言的完整互操作性。

---

## 341. TrustX Agent Risk Classification Framework (ARC): Risk-Tiering Internally Created Agentic AI Systems

**arXiv ID:** 2607.09586 | [PDF](https://arxiv.org/pdf/2607.09586v1)

**作者:** Hannah M. Liu `[一作]` (Responsible AI Institute), Shiv Asthana `[通讯]` (Responsible AI Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 TrustX Agent Risk Classification Framework（ARC），用于对七类代理式 AI 系统进行结构化、可重复的风险分层评估。

**💡 创新点**

创新点在于：① 将现有 AI 治理框架（NIST、ISO、EU AI Act、OWASP 等）融入 12 维评分量表，并采用“关键维度”策略防止高风险被平均稀释；② 针对编码助手提供专门的 20 项能力评估、部署模型分类与 20 项风险因子；③ 通过自治层级（L1–L5）与 GPA+IAT 机构属性相结合，形成三层治理输出与对应控制建议。

**🔧 技术方法**

技术方法包括：基于 GPA+IAT 的代理性属性检查、Feng 等的五层自治评估、12 维风险量表打分、关键维度加权公式、以及编码助手扩展的能力与风险因子列表。

**📊 数据集**

论文未使用公开数据集，而是通过案例演示（如自主代理、决策支持、嵌入式/物理代理、知识助手、工具使用代理、交易代理、编码助手）展示框架的应用和风险分层。

**📈 对比分析**

与其他方法的对比：通过案例说明 ARC 能在不同代理类型中得到一致且符合预期的风险等级（如自主代理为高风险、知识助手为中风险），但未给出定量实验或性能指标；框架的效果主要通过理论评估与专家共识验证。

**⚠️ 局限性**

限制包括：① 评分主观性，需社区反馈完善定义；② 框架为静态快照，无法捕捉生命周期内的漂移；③ 随着代理技术快速演进，框架可能迅速过时。

---

## 342. Wan-Dancer: A Hierarchical Framework for Minute-scale Coherent Music-to-Dance Generation

**arXiv ID:** 2607.09581 | [PDF](https://arxiv.org/pdf/2607.09581v1)

**作者:** Mingyang Huang `[一作]` (Tongyi Lab, Alibaba Group), Bang Zhang `[通讯]` (Tongyi Lab, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在音乐与文本条件下，提出层次化全局关键帧规划与局部细化框架，能够生成一分钟级、720p/30fps的连贯舞蹈视频。

**💡 创新点**

创新点包括：① 层次化全局‑局部生成架构；② 动态帧率适配的 RoPE 时序映射；③ 基于光流的运动连续性损失；④ 运动速度控制机制；⑤ LoRA 微调实现舞蹈风格定制。

**🔧 技术方法**

主要技术：Diffusion Transformer（DiT）+ VAE、RoPE+时序注入、音频编码器、光流网络（SEA‑RAFT）、LoRA、矩阵分解等。

**📊 数据集**

使用自建约200小时、720p/30fps高分辨率舞蹈视频数据集，涵盖中式古典、K‑Pop、拉丁、踢踏、街舞等五大流派。

**📈 对比分析**

与 X‑Dancer、MusicInfuser 等现有端到端方法对比，评估指标包括舞蹈质量、视频质量与提示对齐。实验结果显示，在所有指标上均优于基线，并突破 20 秒时长限制，成功生成超过一分钟的高质量舞蹈视频。

**⚠️ 局限性**

当前局限：① 角色身份一致性不足；② 语义对齐与多模态表达仍有限；③ 仅支持单舞者，未覆盖多舞者交互与群舞。

---

## 343. Evolution of Accuracy and Visual-Cognitive Errors in a Decade of Vision-Language AI Models

**arXiv ID:** 2607.09654 | [PDF](https://arxiv.org/pdf/2607.09654v1)

**作者:** Shravan Murlidaran `[一作]` (University of California), Miguel P. Eckstein `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了过去十年视觉-语言模型在描述复杂社会行为图像与传统MS-COCO图像上的性能，并对模型产生的错误类型进行系统分析。

**💡 创新点**

创新点在于构造了Complex Social Behavior（CSB）数据集、提出了五类视觉-认知错误（检测、识别、错觉、场景理解、空间依赖）并用LLM相似度指标（Gemini‑SI/GPT‑SI）与人类评估对齐，提供了跨时代、跨模型的错误归因框架。

**🔧 技术方法**

使用的技术包括多模态大语言模型（GPT‑4、Gemini 等）、预MLLM（UDC、OSCAR 等）以及基于掩码的空间依赖评估、bootstrapping 统计和 FDR 校正；通过 Gemini‑SI/GPT‑SI 计算描述与金标准的余弦相似度。

**📊 数据集**

所用数据集为自制的 CSB（100 张电影剪辑帧）和 MS‑COCO（100 张公开图像）两套样本，此外还评估了未公开的 in‑house CSB 图像以排除训练集泄露。

**📈 对比分析**

比较方法是将模型与 20 条人类描述（按排名选取顶端四条与底端四条）与金标准进行 Gemini‑SI/GPT‑SI 相似度测量；结果显示 pre‑MLLM 在 CSB 上的相似度远低于人类底端描述，MLLM 的相似度与顶端人类描述相当，且错误率大幅下降，唯空间依赖错误仍存在。

**⚠️ 局限性**

局限性包括样本仅 200 张图像、错误分析需要手工逐图探测、MLLM 内部结构不可见导致只能基于输入输出推断，且未覆盖所有视觉推理维度。

---

## 344. VEXAIoT: Autonomous IoT Vulnerability EXploitation using AI Agents

**arXiv ID:** 2607.09653 | [PDF](https://arxiv.org/pdf/2607.09653v1)

**作者:** Katherine Swinea `[一作]` (Tennessee Tech University), Maanak Gupta `[通讯]` (Tennessee Tech University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个多智能体框架VEXA，能在IoT环境中自动完成侦察、漏洞检测、攻击规划和执行，支持多种OWASP IoT Top 10漏洞的利用；

**💡 创新点**

首次将大型语言模型（LLM）与自主攻击代理结合，形成端到端的IoT漏洞挖掘与利用流程，并实现依赖感知的并行攻击；

**🔧 技术方法**

利用LLM（ChatGPT 5.1），nmap/nbtscan等侦察工具，Exploit-DB/Metasploit等公开漏洞数据库，自动生成攻击脚本并在Kali Linux上执行；

**📊 数据集**

使用IoTGoat（覆盖9/10个OWASP IoT Top 10场景）和Metasploitable2作为测试环境，共计260次攻击实验；

**📈 对比分析**

在IoTGoat中取得95.0%整体成功率（IoTGoat 94.5%，Metasploitable 100%），单次攻击平均执行时<2.5分钟，token使用500–1500，显示与传统手工或规则驱动方法相比，速度更快、成功率更高；

**⚠️ 局限性**

受限于LLM生成命令的语法错误、模型拒绝和hallucination，导致部分攻击失败；缺乏完善的命令验证与错误恢复机制，且在真实物理硬件环境下尚未验证。

---

## 345. Revisiting Euler-Angle Regression with Kolmogorov-Arnold Networks

**arXiv ID:** 2607.09650 | [PDF](https://arxiv.org/pdf/2607.09650v1)

**作者:** Yangting Sun `[一作]` (Independent Researcher), Yufei Zhang `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对受限范围的欧拉角回归，提出结合 Kolmogorov‑Arnold 网络（KAN）与范围感知欧拉角建模的框架，并在对象姿态、机器人手臂逆运动学及人类手部逆运动学等多种任务中验证其效果。

**💡 创新点**

① 将 KAN 引入欧拉角回归，利用可学习的单变量激活函数匹配欧拉角的角度结构；② 通过限定欧拉角的工作区间并将最受限轴置于中间位置，消除周期性不连续和万向锁奇异；③ 在理论上证明受限欧拉角回归的目标近似可加结构，说明 KAN 在此场景下的近似优势。

**🔧 技术方法**

使用 Kolmogorov‑Arnold 网络（KAN）配合 B‑spline 可学习边激活；传统 MLP 与 6D、轴角、四元数等欧拉角表示；回归损失、几何误差（GE）、均值角误差（MAE）等评估指标；针对机器人手臂、手部和对象姿态的深度编码器‑解码器网络。

**📊 数据集**

ModelNet10（椅子子类）用于对象姿态；Franka Panda 机器人手臂；FreiHand 数据集用于人类手部逆运动学；以及公开的旋转矩阵回归基准。

**📈 对比分析**

与传统 MLP+6D、MLP+Euler、MLP+AA 等基线在 MAE、GE、FKE、SR@1cm 等指标上进行对比。KAN+Euler 在所有任务上均实现 MAE 下降 30‑40%、GE 下降 35‑50%，在受限工作空间或低维度情况下获得 100% 成功率，且参数与 FLOPs 更少、收敛更快。

**⚠️ 局限性**

仅适用于单个受限欧拉角范围，未覆盖关节耦合或更高维度的姿态约束；需要手动选择轴顺序；实验主要在仿真或特定数据集上，真实环境中的噪声与非理想性仍需进一步验证。

---

## 346. B-spline Policy: Accelerating Manipulation Policies via B-spline Action Representations

**arXiv ID:** 2607.09648 | [PDF](https://arxiv.org/pdf/2607.09648v1)

**作者:** Xiaoshen Han `[一作]` (Harvard), Yilun Du `[通讯]` (Harvard)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6514db3d-8de6-452c-91b7-acdb31787cc4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出B-spline Policy（BSP），将机器人动作表示为连续B样条曲线，并在推理时通过段对齐实现高速执行。

**💡 创新点**

创新点包括：①用B样条曲线替代离散动作块，实现可变时间分辨率与连续平滑；②自适应节点拟合将演示轨迹压缩为低维参数；③推理时段对齐机制消除段边不连续，提高高速鲁棒性。

**🔧 技术方法**

技术主要包括：B样条曲线拟合（FITPACK）、深度模仿学习框架（Diffusion Policy、ACT）、高频低层控制、时间缩放与段对齐算法。

**📊 数据集**

使用真实演示数据（200/300/200条）用于三项机器人任务（精确抓取、桌面清理、杯子金字塔）以及仿真数据集Push‑T、RoboMimic、RoboCasa。

**📈 对比分析**

与传统基于动作块的Diffusion Policy、Regression Policy以及DemoSpeedup对比，BSP在相同成功率下将任务完成时间缩短50%以上，尤其在4×速度提升时完成时间显著减少；在大多数任务中成功率保持不变或提升。

**⚠️ 局限性**

限制主要在低成本机械臂的低层控制器无法跟踪极高速度的指令，导致在4×速度下部分任务失效；需更高刚度或更精细的控制算法。

---

## 347. Mosaic: Runtime-Efficient Multi-Agent Embodied Planning

**arXiv ID:** 2607.09603 | [PDF](https://arxiv.org/pdf/2607.09603v1)

**作者:** Kunjal Panchal `[一作]` (University of Massachusetts), Hui Guan `[通讯]` (University of Massachusetts)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个运行时高效的多智能体具身规划框架，目标是显著降低失败操作、减少 LLM 调用次数并提升任务完成速度。

**💡 创新点**

创新点在于：① 引入基于代理视角的语义记忆（Agent‑Centric Semantic Memory，ASM），以相对坐标轻量化存储关键对象并支持几何变换；② 在每个规划步使用整数线性规划（ILP）对 LLM 生成的动作候选进行联合优化，强制满足碰撞、路径、工作负载等多重约束，从而实现细粒度的动作层级协同。

**🔧 技术方法**

核心技术包括：LLM 作为动作生成器；相对坐标语义记忆；整数线性规划与约束优化；几何变换与协同约束的组合；以及对 LLM 调用成本的评估。

**📊 数据集**

实验使用了 AI2‑THOR 住宅环境和 SAR（搜索‑救援）模拟环境两套基准数据集。

**📈 对比分析**

与最新基线系统（如 LLM‑based multi‑agent planners）进行对比：执行时间提升 27–32%，LLM 调用次数降低 30–33%，物理步骤减少 25–31%，在固定规划步数预算下成功率提升 4–10%；在 5 个代理的规模下仍能实现 1.25–1.30 倍的整体加速，单代理版本相比亦能缩短 1.25–1.33 倍。

**⚠️ 局限性**

局限性：① 对 LLM 推理时延的依赖仍然存在，特别在极端资源受限或实时需求场景下；② 在极度动态或严重噪声的观测环境中，ASM 的相对坐标推理可能出现误差，导致协同失效；③ ILP 的求解复杂度随代理数迅速上升，对大规模多代理系统的可扩展性仍需进一步验证。

---

## 348. Improved Approximation of Min-Distances in Near-Linear Time

**arXiv ID:** 2607.09588 | [PDF](https://arxiv.org/pdf/2607.09588v1)

**作者:** Yael Kirkpatrick `[一作]` `[通讯]` (Massachusetts Institute of Technology), Yael Kirkpatrick (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种随机近线性时间算法，能够在有向图上以3-近似值逼近最小距离度量下的图直径，并将这一方法扩展到双模式图（2-mode）直径问题，同样实现3-近似。

**💡 创新点**

创新点在于引入了“类型分类框架”，通过对节点的类型（Type 1/Type 2）进行划分，并利用随机采样与球形并集相结合的技术，克服了最小距离非度量导致的传统算法难以适用的问题，从而实现了相较之前4-近似更优的近似比且保持近线性运行时间。

**🔧 技术方法**

主要技术包括：1) 类型分类与相同类型节点的聚类；2) 在多层“padding”中使用入/出球的交集来逐步缩小候选点集；3) 随机样本与概率分析保证高成功率；4) 对双模式图的改造通过在两种边集上分别取入/出球的并集，保持性质不变。

**📊 数据集**

论文没有使用具体实验数据集，而是基于理论分析给出算法的时间复杂度与近似比。所述算法对任何有向图或双模式图均适用，无需预先构建特定数据集。

**📈 对比分析**

与之前的工作相比，Dalirrooyfard等人提供的4k‑1近似与Chechik‑Zhang的4-近似均已被本文的3-近似超越；运行时间从 O(m n^{1/(k+1)})（k≥1）提升到 O(m)（近线性），即在相同或更快的时间内获得更优的近似比。对双模式直径，之前仅有 O(n) 近似被改进到 3-近似，同样保持 O(m) 时间。

**⚠️ 局限性**

限制包括：1) 对直径的近似已完成，但对半径（min-radius）或更一般的eccentricity的近似仍未解决；2) 在双模式图中，虽然直径可实现3-近似，但对于双模式半径的近似由于复杂度难题尚无子二次时间算法；3) 算法依赖随机化，成功概率为 1−O(1/n)，在极端图结构下可能需要多次尝试。

---

## 349. CoDiMAD: Diffusion-Based Privileged Distillation for Communication-Free Multi-Robot Coordination

**arXiv ID:** 2607.09587 | [PDF](https://arxiv.org/pdf/2607.09587v1)

**作者:** Jiyue Tao `[一作]` (Peking University), Feitian Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为CoDiMAD的三阶段框架，用于在部分可观测的情况下实现多机器人协调，特别是在无通信的环境中。

**💡 创新点**

创新点在于通过条件去噪扩散模型解决多模态动作分布的问题，首次将扩散生成策略整合到多机器人特权蒸馏中。

**🔧 技术方法**

使用了条件去噪扩散概率模型（DDPM）和去噪扩散隐式模型（DDIM）来进行策略蒸馏。

**📊 数据集**

使用了一个离线数据集，该数据集由本地观察和特权动作对组成，数据来自于训练好的特权oracle策略。

**📈 对比分析**

与直接的本地多智能体强化学习（MARL）和确定性蒸馏基线进行比较，CoDiMAD在三个合作任务中表现出一致的优越性，接近特权oracle的性能。

**⚠️ 局限性**

局限性包括作为离线蒸馏方法，CoDiMAD可能难以从oracle轨迹分布之外的状态恢复，且在超过3个代理的情况下的可扩展性尚未评估。

---

## 350. Promptable Concept Segmentation from Above: Evaluating SAM 3's Zero-Shot and One-Shot Capabilities in Remote Sensing

**arXiv ID:** 2607.09583 | [PDF](https://arxiv.org/pdf/2607.09583v1)

**作者:** Mohammad Dabaja `[一作]` (University of Agder), Turgay Celik `[通讯]` (University of Agder)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在遥感领域对 SAM 3 进行无监督零/一-shot 评估，涵盖场景分类、目标检测和实例分割。

**💡 创新点**

将 SAM 3 的解耦 Presence Head 转化为全局分类器，系统剖析多模态对齐机制，并提出训练免费 Proxy 评估协议。

**🔧 技术方法**

使用 Promptable Concept Segmentation 架构、文本与视觉提示分离、多模态解码器、Oracle 负样本过滤以及 Harmonic Mean 性能度量。

**📊 数据集**

采用 AID、DIOR、iSAID 三个遥感基准数据集。

**📈 对比分析**

与 CLIP、RemoteCLIP、RSDiX 等基准模型对比，SAM 3 在视觉解码器表现突出（如 Box-Only 检测 66.55 mAP），但零-shot 文本性能低，整体 Harmonic Mean 约 5–40，表明仍有提升空间。

**⚠️ 局限性**

跨模态语义对齐不足、分辨率瓶颈导致小目标识别差、存在负样本误检，需轻量化文本/解码器微调。

---

## 351. Semantic Pareto-DQN: A Multi-Objective Reinforcement Learning Framework for Financial Anomaly Detection

**arXiv ID:** 2607.09641 | [PDF](https://arxiv.org/pdf/2607.09641v1)

**作者:** Cláudio Lúcio do Val Lopes `[一作]` (A3Data), Lucca Machado da Silva `[通讯]` (A3Data)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对金融异常检测中极端类别不平衡问题，提出了Semantic Pareto-DQN多目标强化学习框架，通过将交易特征合成自然语言叙事并用大语言模型编码为语义向量，构建尺度不变的状态表示，利用Pareto-DQN动态决策平衡金融效益、运营摩擦和语义发现。

**💡 创新点**

创新点包括：①将多样交易特征融合为自然语言叙事，再用LLM生成语义向量，构建统一的连续语义空间；②设计三维向量奖励（金融效益、摩擦惩罚、语义多样性），实现目标分离；③采用超体积（Hypervolume）指标直接逼近连续Pareto前沿，避免传统单目标或标量化方法导致的‘fraud collapse’。

**🔧 技术方法**

技术手段主要有：大语言模型文本编码（Sentence Transformer）、L2归一化语义嵌入、基于多目标DQN的Pareto学习、指数移动平均计算语义多样性奖励、超体积指标进行动作选择，以及离线手工校准的多维超参数。

**📊 数据集**

实验使用两个公开金融数据集：①e‑commerce欺诈交易数据（约15万笔，欺诈率<10%）；②UCI信用卡违约客户数据（23个特征，类别不平衡）。

**📈 对比分析**

与XGBoost原始表格、XGBoost语义嵌入、标准单目标DQN做对比，实验表明Semantic Pareto-DQN在两数据集上均显著提升少数类召回率和F1分数。e‑commerce数据中召回率从0提升至0.267，F1从0.101提升至0.109；UCI数据中召回率从0.353提升至0.419，F1从0.458提升至0.481。

**⚠️ 局限性**

局限性主要在：①需要离线手工校准众多超参数（λ、κ、δ、η、τ、M等），缺乏自适应学习；②超体积参考点固定，导致对业务优先级变化的适应性有限；③对LLM语义嵌入高度依赖，若缺乏文本描述或嵌入质量下降，模型性能会受限。

---

## 352. 4DR360: State Reasoning for Joint 3D Detection and Occupancy Prediction in 4D Radar-Camera Full-Scene Perception

**arXiv ID:** 2607.09629 | [PDF](https://arxiv.org/pdf/2607.09629v1)

**作者:** Xiaokai Bai `[一作]` (Zhejiang University), Hui-liang Shen `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种基于4D毫米波雷达与多视角相机融合的360°全景感知框架，联合预测3D检测框和语义占用布局；

**💡 创新点**

将语义占用建模为持续的场景状态，并提出状态引导BEV增强（SBE）与多普勒引导时序融合（DTF）实现跨模态、跨时序的状态推理；同时扩展ManTruckScenes的占用标签并统一评测协议；

**🔧 技术方法**

利用4D雷达+多相机输入，采用BEV变换与双向可变形注意力进行跨模态融合；引入层次化占用状态推理、状态引导自注意与交叉注意；多普勒补偿的时间融合；损失函数包含焦点损失、几何与语义尺度损失；全部在MMDetection3D框架下训练；

**📊 数据集**

使用OmniHD-Scenes（6相机+6雷达，4类检测+11类占用）和扩展后的ManTruckScenes（4相机，10类占用），并在夜间/雨天子集评估鲁棒性；

**📈 对比分析**

与RCBEVDet、RCFusion、HGSFusion、Doracamom等现有雷达-相机多任务方法在统一基准上比较；在OmniHD-Scenes上实现mAP 49.57（+7.85）、NDS 52.97（+6.12）和占用mIoU 31.65（+2.96）；在ManTruckScenes占用SC IoU 35.05、mIoU 33.71，检测mAP 48.62、NDS 52.05；夜雨子集提升2–3分；

**⚠️ 局限性**

仍受雷达稀疏性限制，状态推理受雷达分辨率约束；时序融合需要较长历史窗口，计算开销略增；未实现完整端到端雷达-相机联合训练，需进一步验证对极端动态场景的鲁棒性。

---

## 353. Density Evolution of Soft-Decision Collapsed Projection-Aggregation Decoding for Reed-Muller Codes over the BIAWGN Channel

**arXiv ID:** 2607.09602 | [PDF](https://arxiv.org/pdf/2607.09602v1)

**作者:** Jiajie Li `[一作]` (McGill University), Warren J. Gross `[通讯]` (McGill University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对BIAWGN信道下的Reed–Muller码，分析并构建软判决的Collapsed Projection‑Aggregation（CPA）解码的密度演化模型；证明CPA能给出精确的边缘概率并保持对称性；并基于该模型对迭代过程中的平均值与方差下降进行理论分析，进一步给出CPA在零码率下可实现消失误码率的渐进性结论。

**💡 创新点**

①证明CPA解码返回精确的边缘概率，且对码字具有对称性；②基于此构建密度演化模型；③在模型中近似投影与Fast Hadamard Transform（FHT）为硬判决，简化分析；④从密度演化中解释CPA的快速收敛机制；⑤给出渐近分析，证明零码率时误码率趋于0。

**🔧 技术方法**

软判决投影与聚合、Fast Hadamard Transform（FHT）、密度演化（Density Evolution）、正态逼近与中心极限定理、渐进概率分析。

**📊 数据集**

仿真使用rm(7,3)和rm(8,3)两组Reed–Muller码，在E_b/N_0分别为2.5 dB和1.5 dB的BIAWGN信道上，采样10^4帧。

**📈 对比分析**

通过将仿真得到的LLR直方图与密度演化预测的高斯分布进行对比，发现模型能较好捕捉平均值和方差的快速下降；误码率随迭代次数快速下降，早停条件能有效终止迭代。

**⚠️ 局限性**

①假设不同子空间间LLR相互独立，导致方差低估；②FHT采用硬判决近似，实际误差率可能更低；③未考虑子空间间的相关性，缺乏精确误差上界；④模型在有限码长下的准确性有限，无法给出严格的误码率下界。

---

## 354. Tokenizer Transplantation: Mitigating Autoregressive Collapse in Edge-Efficient Bengali ASR

**arXiv ID:** 2607.09598 | [PDF](https://arxiv.org/pdf/2607.09598v1)

**作者:** Sanjid Hasan `[一作]` (Khulna University of Engineering & Technology), Md. Abdur Rahman `[通讯]` (Military Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在轻量化的ASR模型Moonshine上，通过把原本针对英语的byte级tokenizer替换为本土语言BanglaBERT WordPiece tokenizer，并在声学适配后进行两阶段恢复训练，解决了在孟加拉语上的自回归崩溃问题。

**💡 创新点**

提出了“Tokenizer Transplantation”三阶段适配管线，先对声学编码器进行微调，再单独替换解码器词表并重设嵌入维度，随后采用高LR恢复和稳定阶段两步训练，从而实现低参数模型在低资源语言上的高效稳定解码。

**🔧 技术方法**

使用的技术包括：Moonshine轻量架构、BanglaBERT词表迁移、ScheduleFree AdamW优化器、BF16混合精度、梯度检查点、动态掩码、Beam Search生成、Greedy WER监控。

**📊 数据集**

实验基于882小时的Lipi‑Ghor孟加拉语语料库，包含多说话人、多域长句子；对音频进行VAD过滤并归一化到16kHz。

**📈 对比分析**

与零样本Meta MMS、Whisper、Conformer等基线相比，迁移后的Moonshine模型在Lip‑Ghor 5%测试集上取得21.54% WER、10.79% CER，仅61.5M参数；与769M Faster Whisper Medium相比，WER相近但参数更少；在22h基准上RTF为0.0053，速度是Whisper‑Medium的3.5倍。

**⚠️ 局限性**

仅在孟加拉语上验证，迁移方案对其它语言的可迁移性未探讨；在极低资源或高噪声环境下仍可能出现解码不稳定；迁移后仍需大量计算资源（RTX4070等）进行训练。

---

## 355. Large-Scale Portfolio Optimization Problem Under Cardinality Constraint With Enhanced Multi-Objective Evolutionary Algorithms

**arXiv ID:** 2607.09566 | [PDF](https://arxiv.org/pdf/2607.09566v1)

**作者:** Danial Ramezani `[一作]` (Kharazmi University), Mostafa Abouei Ardakan `[通讯]` (Kharazmi University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在基数约束下的大规模投资组合优化问题，提出并实现了一套改进的多目标进化算法。

**💡 创新点**

创新点包括：① 用双行编码方式限制资产数量，避免超量资产；② 设计了专门的卡点约束修复机制（关联、分数、阈值三种策略）和探索者算子以重新分配权重；③ 采用分阶段的交叉、变异和锦标赛选择策略，提升收敛速度与搜索广度。

**🔧 技术方法**

使用了 NSGA‑II 的改进版本，结合新的交叉/变异算子、卡点约束修复、探索者算子，并使用符号检验和 Wilcoxon 符号秩检验对结果进行统计验证。

**📊 数据集**

实验数据集包括：OR‑Library 标准指数（S&P 100、DAX 100、Nikkei 225）以及伊朗特拉维尔特斯股市（418 支股票）。

**📈 对比分析**

通过 Hypervolume、Generational Distance、Inverted GD、Diversity 四个性能指标进行对比，并采用非参数统计检验。结果表明改进算法在所有指标上均显著优于传统 NSGA‑II，尤其在资产规模较大时优势更明显。

**⚠️ 局限性**

限制在于：未考虑交易成本、流动性等额外实务约束；对极小基数（≤3）情况的性能分析不足；实现基于 Python，缺乏在更高效语言中的验证。

---

## 356. Bidirectional Elaborators à la Carte

**arXiv ID:** 2607.09564 | [PDF](https://arxiv.org/pdf/2607.09564v1)

**作者:** Andrew Slattery `[一作]` (University of Cambridge), Jonathan Sterling `[通讯]` (University of Cambridge)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于依赖类型的单子 DSL，利用部分性单子和初始模型的构造，给出了可验证的、模块化的 bidirectional elaboration 规范，并证明其正确性与稳定性。

**💡 创新点**

创新点在于：①把 elaboration 视为可执行的单子化组合子，消除了对具体实现细节的依赖；②引入了在 SOGAT 语义框架下的“可变性”与“多线性”分析；③通过初始性定理从抽象模型直接提取可执行算法，确保 elaboration 的正确性。

**🔧 技术方法**

主要技术包括：依赖类型理论、部分性单子（relative 2‑monad）与可分离的上下文抽象、SOGAT 的初始性与自然模型、presheaf 语义、以及对核心语法的多线性与多约束性分析。

**📊 数据集**

无数据集，论文纯理论与形式化研究。

**📈 对比分析**

比较方法主要是理论证明：通过构造可执行的组合子并利用初始性定理证明其在所有满足约束的模型上都能得到正确、稳定的 elaboration；未给出实测性能数据。

**⚠️ 局限性**

局限性包括：尚未覆盖累积宇宙、未实现 hole、隐式参数和 type‑class 等动态 elaboration 机制；对实际 proof‑assistant 实现的细节与优化仍需进一步研究。

---

## 357. TCLA: Training-Free Class-wise Logit Adaptation for Medical Vision-Language Models

**arXiv ID:** 2607.09562 | [PDF](https://arxiv.org/pdf/2607.09562v1)

**作者:** Tianyou Jiang `[一作]` (University of Bern), Ziyu Zhou `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出一种训练无关的医学视觉‑语言模型少样本自适应方法 TCLA，利用少量支持样本对零样推理 logits 进行类别‑层级残差校正。

**💡 创新点**

创新点在于：①全无训练、无参数更新；②多层类别原型和提示分布对齐相结合；③通过闭式残差映射实现高效自适应。

**🔧 技术方法**

采用 Mahalanobis 加权原型、层级可辨性评分、Prompt Distribution Alignment、Ridge 回归残差映射等技术。

**📊 数据集**

在九个医学影像数据集上评估，涵盖 X‑ray、CT、MRI、超声、组织病理图像（共 9 个数据集）。

**📈 对比分析**

与 Tip‑Adapter、APE、ProKeR 等训练无关方法及 LP、CoOp、BiomedCoOp、LDC 等训练有监督方法对比，TCLA 在绝大多数数据集与 shot 设定下取得最佳或相近性能，尤其在 1‑shot 至 16‑shot 时表现稳健。

**⚠️ 局限性**

局限在于：仅针对分类任务；对更细粒度的视觉‑语言对齐（如定位、分割）未展开；在某些基于切片的 MRI 数据上多层原型提升有限。

---

## 358. Conceptual Networks for Cross-Linguistic Idiomatic Expressions:A Feature-Based Graph Approach

**arXiv ID:** 2607.09576 | [PDF](https://arxiv.org/pdf/2607.09576v1)

**作者:** Kiran Pala `[一作]` (University of Eastern Finland), Lixun Yu `[通讯]` (University of Eastern Finland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个以认知语言学为基础的可解释网络框架，用二值化的概念特征（含图式、角色与情感价值）对八种语言共160个成语进行注释，并利用Jaccard相似度构建加权图，验证该网络在社区结构、语义检索、跨语言映射及下游成语检测中的有效性。

**💡 创新点**

①将认知图式与成语语义相结合，首次在跨语言层面展示成语聚类主要受图式驱动而非语言；②提出用概念网络而非分布式嵌入进行跨语种等价发现，显著提升匹配准确率；③通过LLM自动特征提取证明框架可扩展；④系统化的消融实验解析各特征维度与图结构信号对性能的贡献。

**🔧 技术方法**

Jaccard相似度构建网络、Louvain社区检测、Betweenness/度中心性、社区归属、邻居相似度等图特征；LLM（GPT‑4）进行特征自动标注；BERT‑based分类器与图特征结合进行成语检测；XLM‑R分布式嵌入做基线对比；PMI 加权提升语料相关性。

**📊 数据集**

8种语言（英语、印地语、巴格里语、芬兰语、日语、西班牙语、汉语、马拉雅拉姆语）各20个成语（共160个），使用 COCA、EMILLE、BCCWJ、Suomi24 等公开语料库进行频率验证；同源语料用于 PMI 计算；人工翻译对等评估用于跨语言检索。

**📈 对比分析**

与 XLM‑R 余弦相似度构建的分布式网络相比，Jaccard网络在与图式划分的 NMI 为 0.76（vs 0.45），在跨语种等价匹配中准确率 78%（vs 54%）；在成语检测任务中，加入网络特征可使 F1 从 0.82 提升至 0.86（增益 4%）。消融实验显示移除图式特征导致 NMI 降至 0.58，F1 增益仅 1.2%。

**⚠️ 局限性**

①特征仅为二值化，缺乏对强度、程度的细粒度表达；②对文化特定或情感细微的成语标注仍需人工校正；③数据量相对有限，跨语言覆盖面有限；④LLM 自动标注虽可扩展，但仍存在错误率；⑤网络依赖完整注释，未探讨对未标注成语的泛化能力。

---

## 359. The Effects of Synthetic Data and Label Distribution on Canola Branch Counting

**arXiv ID:** 2607.09630 | [PDF](https://arxiv.org/pdf/2607.09630v1)

**作者:** Amirsalar Darvishpour `[一作]` (University of Calgary), Adam Runions `[通讯]` (University of Calgary)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

系统性评估了在油菜分枝计数任务中，合成数据与真实数据的比例、标签分布、Gaussian平滑以及最小标签样本数等设计选择对模型性能的影响；

**💡 创新点**

首次量化合成与真实数据比例与标签分布对植物计数模型的作用，并给出实用的合成数据设计准则；

**🔧 技术方法**

使用预训练ResNet‑18进行回归微调，合成图像由校准的L‑system生成，并采用分布插值、Gaussian平滑与最小样本约束等数据增强策略；

**📊 数据集**

使用788张真实油菜图像（分支数1–40）及其通过L‑system产生的合成图像，按400/100/288划分训练/验证/测试集；

**📈 对比分析**

通过5次重复实验评估测试集上的平均绝对误差（MAE），发现合成与真实比例1:5–1:22可显著提升性能，最优比例1:7时MAE降低7.6%；最优标签分布为向真实分布偏移90%（MAE 0.927）或Gaussian平滑σ≈3.6（MAE 0.912），比仅用真实数据提升14.7%；

**⚠️ 局限性**

需要耗费大量工作来校准L‑system模型，且合成比例仍是关键因素；结果仅在油菜分枝计数任务中验证，可能不易直接推广；分布匹配依赖对真实标签分布的准确估计；

---

## 360. Scalable Visual Pretraining for Language Intelligence

**arXiv ID:** 2607.09657 | [PDF](https://arxiv.org/pdf/2607.09657v1)

**作者:** Yiming Zhang `[一作]` (Shanghai Artificial Intelligence Laboratory), Kai Chen `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在不进行文本提取或图像–文本配对监督的前提下，利用原始科学论文页面的视觉信息，对基础模型进行视觉预训练（VP），并与传统文本预训练（TP）在相同语料下进行对比。

**💡 创新点**

①首次证明：仅通过视觉信息的自回归潜在预测即可提升科学推理能力，且在同一语料下比仅用文本预训练更有效；②展示视觉预训练在保持更少令牌预算（约25%）的同时，能实现更快的SFT收敛和更高的下游性能；③揭示视觉预训练能在无配对监督的情况下增强跨模态对齐与多模态推理。

**🔧 技术方法**

使用冻结的视觉编码器（vision tower）提取页面特征；通过前景过滤得到稀疏视觉令牌；将视觉向量投射到LLM隐藏空间；采用自回归视觉潜在预测（InfoNCE对比损失）与传统文本下一个词预测交叉训练；在训练时交替采样文本与视觉样本；保持视觉编码器不变，只更新LLM和投影头。

**📊 数据集**

科学论文PDF语料库（同一批PDF被用于TP和VP）；多种大型语言模型骨干（Qwen3.5、Qwen3、Llama3.2 Vision、Llama3.1）；下游评测数据集包括GPQA、MMLU-Pro、HLE、AIME-25（单模态推理）以及MMMU-Pro、SFE、ChartQAPro、MathVista（多模态推理）。

**📈 对比分析**

实验设定：在匹配的文档源下，TP处理约80B文本令牌，VP处理约20B视觉令牌；两者在相同SFT数据、提示方式和评测协议下比较。结果显示：①VP在所有骨干上均优于TP，GPQA提升最高达3.22分，MMLU-Pro提升2.1分；②VP的SFT损失下降更快，最终损失更低；③按下游提升归一化度量，VP在MMLU-Pro、GPQA、AIME-25分别达到1.27×、2.02×、2.88×；④跨模态对齐指标（中心距离、余弦相似度、线性CKA、k‑NN重叠）均显著提升；⑥在视觉稠密页面上优势更明显。

**⚠️ 局限性**

①视觉预训练并非完全独立于文本预训练；②目前视觉与文本解码的协同尚未最优化；③实验主要聚焦高知识密度科学PDF，尚未验证对自然图像或视频等更广泛视觉语料的迁移；④在更大视觉令牌预算下的超参数与优化仍需调整；⑤缺乏对视觉预训练效果的因果机制深入研究。

---

## 361. OpenLongTail: Generative Scaling of Long-Tail Driving Data

**arXiv ID:** 2607.09655 | [PDF](https://arxiv.org/pdf/2607.09655v1)

**作者:** Lulin Liu `[一作]` (Texas A&M University), Zhiwen Fan `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套名为 OpenLongTail 的生成式数据引擎，将异构的长尾单目驾驶视频转换为与目标摄像机装置对齐、姿态已标记的多视角训练素材，从而提升视觉‑语言‑动作（VLA）驾驶策略在罕见情景下的鲁棒性。

**💡 创新点**

创新点包括：
- 将 Plücker‑ray 几何、时间深度 Warping 与跨视角记忆银行融合到扩散模型中，实现对未观测视角的高质量推断；
- 通过 MapAnything 的度量‑尺度轨迹恢复与 Kalman‑Rauch 后处理，得到平滑且绝对尺度的 ego‑trajectory，作为生成条件；
- 采用 LoRA 适配器在 Wand 2.1‑VACE Diffusion 框架上轻量化训练，使模型既保持生成质量又兼顾可扩展性；
- 通过流匹配目标与跨视角自注意力实现姿态对齐的生成。

**🔧 技术方法**

使用了基于扩散模型的生成技术（Wan 2.1‑VACE + DiT 1.3B）、Plücker‑ray 轨迹编码、时间深度 Warping、跨视角记忆机制、MapAnything 轨迹恢复、Kalman‑Rauch 滤波、LoRA 适配器与流匹配损失。

**📊 数据集**

主要数据集包括：
- NVIDIA PhysicalAI‑Autonomous‑Vehicles（PAV）
- Waymo E2E
- PandaSet
- nuScenes
- Nexar（用户提供的 dash‑cam 视频）
- 还对外部来源如 Waymo E2E 与 Nexar 进行验证。

**📈 对比分析**

与现有多视角生成方法（TrajectoryCrafter、Gen3C、ReCamMaster、Vista4D 等）和轨迹估计器（DROID‑W、MegaSAM、VGGT、MapAnything、ViPE）进行对比。评估指标包括 AlpaSim 的 AS/CR、PSNR/LPIPS/GeoKPM、流匹配损失、ATE、RPE、RotErr、Jerk 等。结果显示：
- 在 AlpaSim 的长尾事件中，OpenLongTail 生成的数据使 AS 提升至 0.748，CR 降至 0.0%（仅次于真实多视角数据）；
- 在视角合成上，PSNR 最高 13.28、LPIPS 最低 0.597，且 GeoKPM μ 达到 82.41，远超基线；
- 轨迹恢复在度量尺度下的 ATE 与 MapAnything 相当，同时显著降低 Jerk 与加速度方差。

**⚠️ 局限性**

局限性包括：
- 生成过程基于扩散模型，推理成本高，实时性受限；
- 仍存在微小的时间不连贯性与帧间伪影；
- 对摄像机参数的偏差（如未知的内参、光学畸变）仍影响生成质量；
- 需要对源数据分布进行对齐选择，单纯扩大数据量并不能保证性能提升。

---

## 362. ConceptSMILE: Auditing the Trustworthiness of Concept-Based Explainable AI

**arXiv ID:** 2607.09649 | [PDF](https://arxiv.org/pdf/2607.09649v1)

**作者:** Mohadeseh Mollapour `[一作]` (University of Hull), Zhibao Mian `[通讯]` (University of Hull)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出ConceptSMILE，一种对概念级可解释AI的模型无关扰动审计框架，扩展SMILE至概念层面；

**💡 创新点**

创新点在于将扰动、局部加权、XGBoost代理建模与多维度可靠性评估（归因准确性、逼真度、可信度、稳定性、一致性）相结合，形成独立的概念解释可审计层；

**🔧 技术方法**

使用超像素扰动、余弦/Wasserstein距离局部权重、XGBoost代理模型，并计算归因准确性、R²等指标；

**📊 数据集**

在四个公开视网膜图像数据集（HRF、APTOS、ODIR5K、IDRiD）上进行实验；

**📈 对比分析**

与MedSAM（视觉分割导向概念）和VLM（语义概念）两条路径对比，评估归因准确性、代理拟合度、可信度、稳定性和一致性；MedSAM在空间归因与代理逼真度上表现更好，VLM在可信度与稳定性上优势更显；

**⚠️ 局限性**

局限包括样本量小、概念范围有限、依赖基础概念提取质量、扰动策略可能不够现实、计算成本高、缺乏因果验证和临床可用性验证等。

---

## 363. Toward Real-Time Sentence-Level Sign Language Translation

**arXiv ID:** 2607.09611 | [PDF](https://arxiv.org/pdf/2607.09611v1)

**作者:** Thanh-Hoang Nguyen Doan `[一作]` `[通讯]` (University of Danang -- University of Science and Technology), Thanh-Hoang Nguyen Doan (University of Danang -- University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

实现了基于Raspberry Pi 4B的实时句子级手语翻译系统，支持摄像捕获、文字展示与语音输出，利用云端GPU进行重识别与翻译；

**💡 创新点**

主要创新在于硬件感知到推理的流水线优化，包括分块采集、有限队列、并行手势感知、时间重排和句子边界状态机，显著降低平均与P95延迟；

**🔧 技术方法**

采用SHuBERT编码器与ByT5字节级解码器，冻结SHuBERT并用QLoRA（4‑bit量化+LoRA）对解码器进行参数高效微调，配合MediaPipe、DINOv2、PoseNet等视觉感知模块；

**📊 数据集**

在How2Sign 9872例子子集（均匀采样）上训练与评估，包含9位手语者的连续视频与对应英文句子；

**📈 对比分析**

与基线流水线相比，优化后平均延迟从1.873 s降至1.354 s（27.7%），P95延迟从2.919 s降至2.130 s（27.0%）；翻译质量在测试集上BLEU 15.9、BLEURT 44.7；

**⚠️ 局限性**

局限包括：只能在句子结束后才翻译，无法实现实时增量解码；延迟受网络与并发负载影响；目前完整模型无法完全在边缘设备上运行；需要进一步验证不同客户端（浏览器、手机）的性能。

---

## 364. KnitID: Machine-Knitted RFID Antennas for Battery-Free Authentication, Localization and Interaction

**arXiv ID:** 2607.09584 | [PDF](https://arxiv.org/pdf/2607.09584v1)

**作者:** Weiye Xu `[一作]` (University of Washington), Yiyue Luo `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种机织RFID天线KnitID，集成在袖子上，实现了无电池身份验证、定位与交互；

**💡 创新点**

创新点在于将导线直接嵌入机织结构，使天线尺寸比传统环形天线缩小90%，并在人体上提升约30%的感知范围；

**🔧 技术方法**

采用机织纺织技术、PTFE覆层铜线、滑动结构、RFID读写器、随机森林与GRU时序编码器等技术；

**📊 数据集**

使用自制数据集，包含10名受试者的RFID ID、RSSI与相位信号，记录手势交互与位置信息；

**📈 对比分析**

与传统环形/偶极天线进行对比，KnitID在同等尺寸下读距提升30%，身份验证100%准确，交互识别>90%准确，定位均方误差仅5cm；

**⚠️ 局限性**

局限性包括对特定环境的依赖、标签密度受限、纺织结构复杂性带来的制造成本以及对人体动态干扰的敏感性。

---

