# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-12 | 今日论文总数: 526

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. IH-Challenge: A Training Dataset to Improve Instruction Hierarchy on Frontier LLMs

**arXiv ID:** 2603.10521 | [PDF](https://arxiv.org/pdf/2603.10521v1)

**作者:** Chuan Guo `[一作]`, Kai Xiao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构造可程序化评估的IF‑simple冲突任务，并在此任务上用强化学习结合在线对抗生成，提升LLM在指令层级(IH)冲突下的鲁棒性；

**💡 创新点**

提出了专门针对IH冲突的RL训练数据集ih‑challenge，并设计了“IF‑simple、可评估、避免快捷方式”的三大构建原则；

**🔧 技术方法**

采用强化学习(PPO/DPO)训练，结合Python程序化评估器做奖励，在线利用攻击模型生成对抗低优先级指令；

**📊 数据集**

使用自研的ih‑challenge数据集（包含多种任务族和Python评估脚本），并在公开的安全、注入、学术评测数据集上进行验证；

**📈 对比分析**

与基线模型对比，IH鲁棒性从84.1%提升至94.1%（+10%），不安全行为率从6.6%降至0.7%，在安全评估、提示注入和人类红队测试中均显著优于未训练模型；

**⚠️ 局限性**

局限于IF‑simple任务，可能无法完全覆盖复杂非程序化评估场景；对抗生成质量依赖攻击模型；部分对话能力可能略有下降；

---

## 2. Spatio-Temporal Attention Graph Neural Network: Explaining Causalities With Attention

**arXiv ID:** 2603.10676 | [PDF](https://arxiv.org/pdf/2603.10676v1)

**作者:** Kosti Koistinen `[一作]` (Aalto University), Kimmo K. Kaski `[通讯]` (Aalto University)

**通讯引用:** 18350 | [OpenAlex ID](https://openalex.org/A5065692438)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了一种可解释的 Spatio‑Temporal Attention Graph Neural Network (STA‑GNN)，用于工业控制系统的无监督异常检测。

**💡 创新点**

创新点包括：①动态学习时空依赖的图结构；②结合 Transformer‑style 多头时间注意力与图注意力，实现多模态（SCADA 点、NetFlow、payload）融合；③引入差分非合规评分的合规预测机制以自适应控制误报率；④生成可视化的注意力图，提供因果关系解释。

**🔧 技术方法**

采用的技术有：Transformer 自注意力、图注意力网络 (GAT)、动态图构建、混合损失 (MSE + BCE)、差分非合规评分与合规预测、以及图可解释性工具（如 GNNExplainer/PG‑Explainer 的思想）。

**📊 数据集**

使用的数据集为 SWaT 测试平台的 2015、2017、2019 年多模态数据，包含物理点 (SCADA)、NetFlow 与 NetFlow+Payload 三种模式。

**📈 对比分析**

通过与 K‑means、SVM、LSTM‑VAE 等基线模型在 F1、FPR 及检测攻击数等指标上对比，STA‑GNN 在物理点与 NetFlow+Payload 模式下获得最高 F1（0.74–0.77）、最低 FPR（<0.01）且检测攻击数显著多于基线；但在 NetFlow 单模态表现不佳。

**⚠️ 局限性**

局限性：①对数据漂移敏感，模型在新年份数据上 FPR 明显上升；②在小规模网络中注意力图解释性有限；③对 payload 特征工程不足，导致 NetFlow+Payload 仍受限；④需要先验图或手工调参，增加部署难度；⑤计算复杂度高，实时性能尚待评估。

---

## 3. A Universal Nearest-Neighbor Estimator for Intrinsic Dimensionality

**arXiv ID:** 2603.10493 | [PDF](https://arxiv.org/pdf/2603.10493v1)

**作者:** Eng-Jon Ong `[一作]` (Queen Mary University of London), Primoz Skraba `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于最近邻距离比值的全新本征维度（ID）估计器 L2N2，理论证明其收敛到真实 ID 并在有限样本下表现稳健。

**💡 创新点**

创新点在于：① 利用对数对数（log‑log）最近邻距离比的平均值，得到一个既简单又可调的估计公式；② 在理论上证明该估计器对数据分布具有“通用性”，即不依赖具体分布；③ 通过一次性预先调参实现对不同样本量和维度范围的适配。

**🔧 技术方法**

主要技术包括：最近邻搜索、统计学中的极限理论（点过程和Poisson 过程）、最小二乘回归调参、基准与真实数据实验以及与多种现有方法的对比。

**📊 数据集**

使用了 24 种基准流形（如球面、螺旋、摩比乌斯带等）、带噪声的高维球面数据、以及四个真实世界数据集（ISOMAP 人脸、MNIST 手写数字、CIFAR‑100 图像、Isolet 语音特征）。

**📈 对比分析**

与 14 种现有 ID 估计方法（如 TwoNN、GriDE、MLE、kNN 等）在 MPE、均方误差、噪声鲁棒性等指标上进行对比，L2N2 在大多数情形下实现了最低 MPE，尤其是 (k,j)=(2,1) 配置；在噪声实验中保持与最优方法相近的性能；在真实数据实验中估计值与人类经验相符或更高，表明对高维数据的鲁棒性。

**⚠️ 局限性**

限制包括：对极低样本量和高 ID 情形仍存在偏差；对非光滑或分形分布的理论保证尚未完全证明；对 (k,j) 选择的理论解释有限，需进一步探究更优配置。

---

## 4. Evaluating randomized smoothing as a defense against adversarial attacks in trajectory prediction

**arXiv ID:** 2603.10821 | [PDF](https://arxiv.org/pdf/2603.10821v1)

**作者:** Julian F. Schumann `[一作]` (TU Delft), Arkady Zgonnikov `[通讯]` (TU Delft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并评估了基于随机平滑的轨迹预测模型防御机制，以提升其对对抗攻击的鲁棒性

**💡 创新点**

首次将随机平滑技术应用于轨迹预测领域，提出位置平滑和控制平滑两种噪声策略，并系统验证其在不同模型与数据集上的有效性

**🔧 技术方法**

使用Gaussian噪声进行随机平滑（对位置或控制输入加噪后取平均），结合运动学约束的白盒PGD对抗攻击以及ADE指标进行评估

**📊 数据集**

L-GAP（驾驶模拟数据）与rounD（德国环岛真实数据）两大数据集；基线模型为Trajectron++和ADAPT

**📈 对比分析**

对比不平滑、不同噪声幅度、攻击幅度及训练期间是否使用平滑等配置；实验表明随机平滑在所有对抗攻击实验中均能显著降低ADE，对正常数据几乎无损，且在部分设置下甚至提升性能

**⚠️ 局限性**

实验范围局限于两种数据集和两种模型；未给出严格的概率鲁棒性保证；对抗攻击仅采用一种白盒方法；噪声参数选择缺乏理论指导，未来需扩展到更多模型、数据集和攻击手段

---

## 5. The coordination gap in frontier AI safety policies

**arXiv ID:** 2603.10015 | [PDF](https://arxiv.org/pdf/2603.10015v1)

**作者:** Isaak Mengesha `[一作]` (University of Amsterdam), Isaak Mengesha `[通讯]` (University of Amsterdam)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5075715415)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出前沿人工智能治理中的鲁棒性缺口，并设计了Scenario Response Registry（SRR）机制以提升跨主体协调能力。

**💡 创新点**

创新点在于将情景预先登记与标准化应对方案结合，形成可检验的前置协调框架，借鉴核安全与流行病应对经验。

**🔧 技术方法**

主要使用情景分析、标准化登记表格以及协调评估流程等技术手段。

**📊 数据集**

参考了AIID事件数据库以及相关政策文件来构建情景库，但并未使用传统机器学习数据集。

**📈 对比分析**

本文未进行实验对比，而是通过案例对比与专家评议说明SRR可提升危机准备水平。

**⚠️ 局限性**

局限在于缺乏强制执行机制、国际合作受限、情景设定可能被捕捉或过于宽泛，导致实际可操作性受限。

---

## 6. Optimal Expert-Attention Allocation in Mixture-of-Experts: A Scalable Law for Dynamic Model Design

**arXiv ID:** 2603.10379 | [PDF](https://arxiv.org/pdf/2603.10379v1)

**作者:** Junzhuo Li `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1218 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究Mixture-of-Experts模型中专家层与注意力层计算分配的最优比例，并建立相应的规模规律。

**💡 创新点**

提出专家-注意力FLOPs比例随总计算量与稀疏度呈幂律关系的经验定律，并将其融入Chinchilla扩展的神经尺度法则。

**🔧 技术方法**

使用GPT风格MoE Transformer、实验验证、经验拟合以及扩展的损失尺度公式。

**📊 数据集**

未公开具体数据集，实验基于大规模文本预训练数据。

**📈 对比分析**

与固定比例设计对比，实验显示遵循比例定律的模型在相同计算预算下损失更低，验证公式在留样数据上保持一致。

**⚠️ 局限性**

仅覆盖自回归语言建模，未考虑多模态任务、可适配路由和硬件通信开销。

---

## 7. Visually-Guided Controllable Medical Image Generation via Fine-Grained Semantic Disentanglement

**arXiv ID:** 2603.10519 | [PDF](https://arxiv.org/pdf/2603.10519v1)

**作者:** Xin Huang `[一作]` (Northeastern University), Osmar R. Zaiane `[通讯]` (University of Alberta)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5027917989)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了一种视觉引导的文本解耦扩散框架，实现医学文本到图像的高质量、可控生成。

**💡 创新点**

创新点在于利用视觉先验实现文本语义的解耦，并提出交叉模态对齐与混合特征融合（HFFM）模块，以实现结构与风格的精细控制。

**🔧 技术方法**

采用视觉引导的跨模态对齐、Hybrid Feature Fusion、Diffusion Transformer（DiT）+ LoRA、色彩分布损失等技术。

**📊 数据集**

在HAM10000、Kvasir-SEG、BUSI三个公开医学图像数据集上进行训练与评估。

**📈 对比分析**

与SD1.5、SDXL、PixArt-α、MedSegFactory、Med-Art等基线相比，FID/HFD/KID显著降低，生成质量与多样性提升；合成数据可显著提高下游分类任务的 F1 与 BACC。

**⚠️ 局限性**

仍存在文本描述稀缺、跨模态对齐不完全、以及在极低样本或高分辨率场景下生成效果有限等局限。

---

## 8. Degeneracy-Resilient Teach and Repeat for Geometrically Challenging Environments Using FMCW Lidar

**arXiv ID:** 2603.10248 | [PDF](https://arxiv.org/pdf/2603.10248v1)

**作者:** Katya M. Papais `[一作]` (Institute for Aerospace Studies), Timothy D. Barfoot `[通讯]` (Institute for Aerospace Studies)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种使用FMCW激光雷达的Degeneracy-Resilient Teach‑and‑Repeat导航系统，能够在几何退化环境中实现闭环自主导航；

**💡 创新点**

创新点在于（1）在对应无关的Doppler‑inertial里程计中加入姿态不确定性估计；（2）采用曲率增强的下采样与联合曲率-空间匹配实现鲁棒数据关联；（3）提出块尺度化的Eigenvalue‑based DA‑ICP，在退化方向上只更新良好约束的分量；

**🔧 技术方法**

使用技术包括：Doppler velocity‑based correspondence‑free odometry、IMU融合、曲率估计与聚类、块尺度化的点到平面ICP、解耦的联合优化及鲁棒成本；

**📊 数据集**

实验数据集为三条不同几何结构的实地路径：SA（多样地形）路线、校园路线和机场跑道（几何稀疏）路线；

**📈 对比分析**

与四种基线变体（默认ICP、曲率预处理、Doppler里程计等）比较，结果显示在机场跑道等极端退化环境中本方法能成功完成闭环导航，而基线多失败；在校园及SA环境中性能与基线相近，偶有轻微误差（由保守退化阈值导致）；

**⚠️ 局限性**

局限性包括：在特征丰富环境下保守的退化检测阈值导致误差略高；SA环境缺乏精确GPS/总站数据；系统主要针对退化场景，若几何特征丰富时可能不如传统ICP高精度。

---

## 9. Incremental Federated Learning for Intrusion Detection in IoT Networks under Evolving Threat Landscape

**arXiv ID:** 2603.10776 | [PDF](https://arxiv.org/pdf/2603.10776v1)

**作者:** Muaan Ur Rehman `[一作]` (Tallinn University of Technology), Rajesh Kalakoti `[通讯]` (Northern Arizona University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在IoT网络中使用增量联邦学习提升非平稳入侵检测系统的长期性能。

**💡 创新点**

提出时间段驱动的增量学习框架，并对累计、代表性、保留等多种策略在概念漂移下进行系统评估。

**🔧 技术方法**

采用LSTM模型结合Federated Averaging（FedAvg）实现增量训练，并实现数据保留与代表性采样等记忆策略。

**📊 数据集**

使用CICIoMT2024 IoT安全数据集，包含五大攻击类别共18种攻击。

**📈 对比分析**

与静态、累计、简单增量、代表性增量和保留样本等方法对比，累计增量取得最高准确率，代表性和保留样本在保持接近准确率的同时显著降低训练时延。

**⚠️ 局限性**

主要限制在实验仅使用IID客户端、单一LSTM架构，且未实现实时漂移检测，缺乏对非IID环境和多模态数据的验证。

---

## 10. The Generation-Recognition Asymmetry: Six Dimensions of a Fundamental Divide in Formal Language Theory

**arXiv ID:** 2603.10139 | [PDF](https://arxiv.org/pdf/2603.10139v1)

**作者:** Romain Peyrichou `[一作]` `[通讯]` (Independent Researcher), Romain Peyrichou (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对形式文法中的生成–识别（解析）不对称性进行系统综述，并提出一个包含六个独立维度（复杂度、歧义、方向性、信息、推断、时序）的框架，阐释三种文法用途（生成、识别、归纳）之间的关系。

**💡 创新点**

创新点在于：①首次将生成与识别的差异结构化为六维多元模型；②将方向性与时序视为新的不对称维度；③将 Shannon 信息论、Chomsky 层次与 Morris 符号三角理论统一到同一框架中；④将大语言模型的统一结构与传统不对称性对齐，说明不对称性在现代 NLP 仍然存在。

**🔧 技术方法**

主要技术手段为：理论推导（复杂度分析、可判定性、不可判定性证明）、对现有解析/生成算法的分类与比较（LL/LR/Earley/CYK、DCG、GF、FST 等）、心理语言学的预期（Surprisal）与信息论模型、以及对现有文献的综合归纳。

**📊 数据集**

本文不涉及实验数据或特定数据集；所有结论均基于已有文献与理论证明；若需验证，作者建议在自然语言、编译器或音乐等领域构建相应语料进行实验。

**📈 对比分析**

由于本研究为综述性质，未进行量化比较；但作者在文中对各维度给出了相应的复杂度区间、算法优劣示例，并通过对比不同解析策略（LL vs LR）、生成约束（无约束 vs NP‑hard 约束生成）说明不对称性的实际影响。

**⚠️ 局限性**

局限性包括：①缺乏统一的实验验证，理论与实践之间仍有差距；②某些维度（如信息、推断）的定量化仍不完整；③未系统评估不同领域（生物信息学、音乐学等）中该框架的适用性；④对大模型的讨论主要从架构角度出发，未对其在生成/识别上的具体性能进行量化。

---

## 11. Multi-Stream Perturbation Attack: Breaking Safety Alignment of Thinking LLMs Through Concurrent Task Interference

**arXiv ID:** 2603.10091 | [PDF](https://arxiv.org/pdf/2603.10091v1)

**作者:** Fan Yang `[一作]` (Jinan University), Fan Yang `[通讯]` (Jinan University)

**通讯引用:** 35603 | [OpenAlex ID](https://openalex.org/A5101899401)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出针对LLM思考模式的多流扰动攻击方法，利用多任务交错干扰模型推理，诱导其生成有害内容并导致思考崩溃。

**💡 创新点**

创新点在于将多任务交错、字符反转与格式约束三种扰动结合，针对思考模式的逐步推理特性，突破传统单向安全过滤，导致思考过程失稳与重复输出。

**🔧 技术方法**

采用多流交错、反转与形状变换扰动技术，配合标准的jailbreak评估框架，对LLM进行黑盒攻击。

**📊 数据集**

使用JailbreakBench、AdvBench、HarmBench三大基准数据集进行实验。

**📈 对比分析**

与GCG、PAIR、AutoDAN、AutoDAN‑Turbo、JAIL‑CON、FlipAttack等六种主流攻击方法比较，MS_Reverse策略在大多数模型（包括Qwen3系列、DeepSeek、Qwen3‑Max、Gemini 2.5 Flash）上取得超过90%的成功率，并且导致思考长度显著增加、崩溃率高达17%、重复率高达60%。

**⚠️ 局限性**

局限性包括对模型规模和架构的依赖（在8B模型安全性略强）、攻击对推理时间与算力消耗较大，以及未对大规模商业API模型的普适性和对抗训练的鲁棒性进行深入探讨。

---

## 12. Pooling Engram Conditional Memory in Large Language Models using CXL

**arXiv ID:** 2603.10087 | [PDF](https://arxiv.org/pdf/2603.10087v1)

**作者:** Ruiyang Ma `[一作]` (Peking University), Guojie Luo `[通讯]` (Peking University)

**通讯引用:** 1907 | [OpenAlex ID](https://openalex.org/A5023468643)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了将 Engram 模型参数放入共享 CXL 内存池，并在 SGLang 框架中完成推理

**💡 创新点**

首次将 Engram 与 CXL 结合，实现可扩展、低成本的内存扩展方案

**🔧 技术方法**

使用 CXL、DAX 映射、CUDA 并行读取、RDMA 对比实验

**📊 数据集**

采用 Qwen3-4B、Qwen3-8B（模拟）以及 Engram-27B/40B 配置进行验证

**📈 对比分析**

与本地 DRAM 和 RDMA 对比：CXL 读取延迟接近 DRAM，推理吞吐量仅略低于 DRAM，且在多节点/多DP下保持可扩展性

**⚠️ 局限性**

限制在于 Engram 对预取窗口的严格延迟要求，目前实验仅验证小规模集群，RDMA 仍面临小包低效问题

---

## 13. Moving Phones, Active Peers: Exploring the Effect of Animated Phones as Facilitators in In-Person Group Discussion

**arXiv ID:** 2603.10394 | [PDF](https://arxiv.org/pdf/2603.10394v1)

**作者:** Ziqi Pan `[一作]` (Hong Kong University of Science and Technology), Xiaojuan Ma `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5005 | [OpenAlex ID](https://openalex.org/A5026376235)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了AnimaStand，一款可动手机支架，在面对面小组讨论中通过手机的移动与灯光提示促进群体互动，并通过Wizard‑of‑Oz实验评估其对参与度、任务执行和人际关系的影响。

**💡 创新点**

创新点在于将个人智能手机转变为主动的身体化促进者，利用手机的移动而非屏幕交互，以低干扰、可解释的方式增强群体行为和协作，而非引入新设备或强制应用切换。

**🔧 技术方法**

使用了可调节马达与LED的手机支架、ESP32无线控制、实时说话者分辨与语音动态分析（Pyannote）、规则检测与Slack工作区模拟、Wizard‑of‑Oz人工干预以及视频与音频记录技术。

**📊 数据集**

实验数据来源于56名受试者组成的14组四人讨论的录音与录像，未使用公开语料库，全部为实验现场自行收集。

**📈 对比分析**

通过与无动画对照组的双组间t检验/ Mann‑Whitney检验比较参与度、均衡性、任务完成率、主观评价等指标，实验组在参与度、平衡性和关系归属感等方面显著提升，任务完成率趋势上升但未达到统计显著。

**⚠️ 局限性**

局限性包括样本规模有限、使用Wizard‑of‑Oz模拟非完全自主动画、仅针对四人讨论情境、缺乏长期持续使用评估以及对不同文化或任务类型的适用性尚未验证。

---

## 14. Believing vs. Achieving -- The Disconnect between Efficacy Beliefs and Collaborative Outcomes

**arXiv ID:** 2603.10708 | [PDF](https://arxiv.org/pdf/2603.10708v1)

**作者:** Philipp Spitzer `[一作]` (Karlsruhe Institute of Technology), Joshua Holstein `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5060423160)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过实验研究了人类通用效能信念如何影响在AI协作中的实例级判断和委托行为。

**💡 创新点**

揭示效能信念作为认知锚点，发现AI乐观偏差并表明上下文信息的非对称调节作用。

**🔧 技术方法**

采用行为实验、混合效应回归模型和描述性统计分析。

**📊 数据集**

使用美国社区调查（ACS）收入分类数据与决策树AI模型。

**📈 对比分析**

与无信息条件比较，发现上下文信息能消除AI乐观偏差但同时放大效能差异对委托的影响，整体性能提升有限。

**⚠️ 局限性**

局限在单一任务域、固定上下文信息、短期实验以及仅用准确率衡量协作效果。

---

## 15. S-GRADES -- Studying Generalization of Student Response Assessments in Diverse Evaluative Settings

**arXiv ID:** 2603.10233 | [PDF](https://arxiv.org/pdf/2603.10233v1)

**作者:** Tasfia Seuti `[一作]` (University of North Texas), Sagnik Ray Choudhury `[通讯]` (University of North Texas)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5112500027)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了S-GRADES统一评估平台，聚合14个AES与ASAG数据集，提供标准化的数据拆分、提交校验、评估与排行榜；对三大主流LLM（GPT-4o-mini、Gemini 2.5 Flash、Llama‑4‑Scout）在多种推理策略（归纳、演绎、溯因、混合）下进行大规模实验，系统分析模型在不同任务、推理模式、示例选择与跨数据集迁移中的表现与稳定性。

**💡 创新点**

1) 通过统一数据格式与标准化评分尺度解决AES与ASAG长期碎片化问题；2) 构建基于Web的端到端评估管道，支持批量提交、自动化评分与公开排行榜；3) 系统评估LLM在多种推理配置下的跨任务泛化与示例依赖，为“推理感知评估”提供第一手实证。

**🔧 技术方法**

使用Python、Streamlit搭建Web平台；对数据进行统一表结构、分词、元信息保留；实现CSV上传校验、基于SQLAlchemy的安全查询；评估引擎计算QWK、Pearson、MAE、RMSE、分类指标；实验通过OpenRouter API进行三大LLM推理，采用温度0.1，控制随机性。

**📊 数据集**

共14个数据集：AES类（ASAP-AES、ASAP++、ASAP2.0、Persuade_2、IELTS Writing、IELTS Task2）；ASAG类（ASAP-SAS、ReGrading、CSEE、Mohlar、BEEtlE、SciEntSBank、Rice_Chem、OS_Dataset）。涵盖从论说文到实验问答、专业学科短答、语言水平测试等多领域。

**📈 对比分析**

通过对比不同模型、推理策略与数据集，使用QWK为主评估指标，发现：GPT‑4o‑mini在AES任务中最高达0.96，Gemini 2.5 Flash在多任务中表现最均衡；混合推理模式通常优于单一模式；示例选择稳定性较高，跨数据集迁移在AES中表现稳定，而在ASAG跨域迁移性能显著下降。

**⚠️ 局限性**

1) 仅评估三大LLM，缺乏更广泛模型覆盖；2) 示例选择与随机种子对结果有一定影响，需多种采样策略验证；3) 数据集仅为英文文本，缺少多语言与多模态支持；4) ASAG任务依然比AES难，跨域迁移表现差，提示需要更强的领域适配与评分规则建模。

---

## 16. Instruction set for the representation of graphs

**arXiv ID:** 2603.11039 | [PDF](https://arxiv.org/pdf/2603.11039v1)

**作者:** Ezequiel Lopez-Rubio `[一作]`, Mario Pascual-Gonzalez `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

作者提出了一种名为 IsalGraph 的字符串编码方法，可将任何有限简单图转换为九字符指令序列并可逆解码。

**💡 创新点**

其创新之处在于所有合法字符串都能唯一解码为有效图，且通过全局枚举可得到图的唯一规范字符串，提供了潜在的图同构不变量。

**🔧 技术方法**

技术上使用了基于循环双向链表的虚拟机、贪心与全枚举编码算法，并利用 Levenshtein 距离与图编辑距离的相关性评估表示质量。

**📊 数据集**

实验使用 IAM Letter (LOW/MED/HIGH)、LINUX 和 AIDS 这五个真实图数据集。

**📈 对比分析**

通过与精确 GED 的 Spearman 相关系数对比，Canonical 编码在稀疏图上达 ρ≈0.93，Greedy‑min 在 ρ≈0.45 时仍优于 Canonical；编码时间随节点数呈多项式增长，Canonical 的指数约 9，Greedy‑rnd 约 3。

**⚠️ 局限性**

局限性包括：规范字符串的图同构不变量性质尚未正式证明、Canonical 编码在节点数超过约12时超多项式耗时且仅适用于连通图（以及有向图需从起点可达）。

---

## 17. Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias

**arXiv ID:** 2603.10123 | [PDF](https://arxiv.org/pdf/2603.10123v1)

**作者:** Borun D Chowdhury `[一作]` `[通讯]` (Meta), Borun D Chowdhury (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文证明了在大型语言模型中出现的“中间失踪”U形注意力曲线是从随机初始化就存在的结构性先天偏置，而非训练或位置编码引起的。

**💡 创新点**

创新点在于给出了闭式解析的影响密度公式，利用Cesàro矩阵和连续极限揭示了因果掩蔽导致的首位偏倚以及残差连接产生的末位偏倚，并证明这些现象与RoPE无关。

**🔧 技术方法**

主要技术包括线性化 Transformer 的注意力路径、离散与连续 Cesàro 矩阵的推导、梯度输入-输出雅可比矩阵的计算，以及对 Qwen2 与 GPT‑2 结构的实证验证。

**📊 数据集**

实验使用 NaturalQuestions（NQ）多文档 QA 数据集，并在预训练期间对 200 条序列做输入-输出雅可比测量；同时对无 RoPE 的模型与带 RoPE 的模型进行对比。

**📈 对比分析**

通过将理论公式与实际 Jacobian 进行 Spearman 相关性（0.99）和 Wasserstein 距离（0.02）比较，验证了 U‑形曲线在初始化即存在且在预训练后仍保持不变，预训练模型仅在文档边界出现尖峰。

**⚠️ 局限性**

局限性在于理论仅对初始线性值路径成立，对已训练模型的非线性 Softmax 未给出闭式界限，且尚未确定在激进的中间上下文训练下 Score 路径能否完全克服结构性基线。

---

## 18. Agentic Control Center for Data Product Optimization

**arXiv ID:** 2603.10133 | [PDF](https://arxiv.org/pdf/2603.10133v1)

**作者:** Priyadarshini Tamilselvan `[一作]` (Georgia Tech), Horst Samulowitz `[通讯]` (IBM Research)

**通讯引用:** 2626 | [OpenAlex ID](https://openalex.org/A5035277014)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Agentic Control Center，实现数据产品的自动化持续改进与质量监控

**💡 创新点**

首次将可观测的质量契约与多智能体协作相结合，形成连续优化循环

**🔧 技术方法**

采用LLM智能体、状态管理、质量度量、工具注册与Git版本化等技术框架

**📊 数据集**

在BIRD基准数据库上进行案例实验

**📈 对比分析**

通过覆盖率和查询性能指标验证系统能在数轮迭代内满足目标，展示快速收敛但未给出传统基线对比

**⚠️ 局限性**

受限于指标范围、工具多样性、可扩展性以及对复杂多目标优化的支持不足

---

## 19. A gripper for flap separation and opening of sealed bags

**arXiv ID:** 2603.10890 | [PDF](https://arxiv.org/pdf/2603.10890v1)

**作者:** Sergi Foix `[一作]` (Institut de Robòtica i Informàtica Industrial), Júlia Borràs `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**通讯引用:** 860 | [OpenAlex ID](https://openalex.org/A5066456591)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种主动凹槽滚轮指尖与柔性双指相结合的抓手，用于分离并抓取薄层材料（如医用袋内层）并完成无菌包装袋的完整打开。

**💡 创新点**

创新点在于利用凹槽滚轮与两指间距产生的屈曲力实现薄层分离，并首次实现机器人对无菌袋的完整打开流程；同时通过可换滚轮表面实现对不同材质的适应。

**🔧 技术方法**

采用机械设计与快速原型、滚轮摩擦调节、柔性指尖与舵机驱动的主动控制，配合UR5E机械臂与Hand‑E抓手执行操作。

**📊 数据集**

使用医院标准全封闭无菌袋及其切片（纸、塑料、纺织）进行实验，重复多次测定成功率与抓取力。

**📈 对比分析**

通过对滚轮表面、正向压力、滚轮位置等变量的实验比较，得到全封闭袋成功率93.55%（29/31），预开侧封袋96.77%（30/31）；与传统光滑滚轮对比显示更高的分离成功率和抓取强度。

**⚠️ 局限性**

局限在于对正向压力高度依赖，尤其塑料薄层仅在狭窄压力范围内可行；缺乏闭环力/位置信号控制，导致全自动化受限；以及滚轮凹槽的长期磨损问题。

---

## 20. RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion

**arXiv ID:** 2603.10878 | [PDF](https://arxiv.org/pdf/2603.10878v1)

**作者:** Andrea Patrizi `[一作]` (University of Genova), Nikos G. Tsagarakis `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于层次化强化学习与模型预测控制的机器人行走架构，能够自适应学习非周期性接触时序并实现地面与轮式混合行走；

**💡 创新点**

创新点在于通过高层RL直接学习接触调度而非预设步态或示范，并通过同构的MPC实现零样本跨域（仿真到仿真、仿真到实机）转移；

**🔧 技术方法**

主要技术包括Soft Actor-Critic强化学习、基于逆动力学的MPC、ILQR求解器、接触显式分阶段成本约束以及大规模并行MPC训练框架；

**📊 数据集**

使用仿真数据集，在50kg、80kg及120kg机器人（简化四足、Unitree B2-W、Centauro）上进行训练，并在真实Centauro机器人上验证；

**📈 对比分析**

与端到端RL方法比较，样本效率提升约10倍（仅4–10×10⁶环境步长），训练耗时约9–29个模拟日，实时因子达50，零样本转移在MuJoCo和真实机器上均保持高跟踪精度和能耗优势；

**⚠️ 局限性**

局限在于策略对特定MPC与机器人耦合，难以直接迁移至不同动力学模型；对极端扰动的鲁棒性尚未系统评估；且训练仍需大量并行MPC实例和计算资源。

---

## 21. Defining AI Models and AI Systems: A Framework to Resolve the Boundary Problem

**arXiv ID:** 2603.10023 | [PDF](https://arxiv.org/pdf/2603.10023v1)

**作者:** Yuanyuan Sun `[一作]` (AI Governance Exchange), Ze Shen Chin `[通讯]` (AI Standards Lab)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 AI 模型与 AI 系统的定义进行系统性综述，梳理 896 篇学术论文与 80+ 监管/标准文件中的定义，分析其演化与相互影响，并基于此提出兼顾概念性与操作性的 AI 模型与 AI 系统定义。

**💡 创新点**

创新点在于：①将已有定义按父类、特征与模型-系统关系三维度进行分类与可视化；②追踪 OECD、ISO、NIST 等权威框架对定义的链式影响，揭示其随技术演进而产生的概念混淆；③提出针对当下以 Transformer 为核心的神经网络 AI 的“训练参数+架构”模型与“模型+接口+配置”系统的操作性定义，帮助监管者精准划分责任。

**🔧 技术方法**

采用的方法主要是系统文献回顾（Systematic Literature Review, SLR）与人工手工检索，结合概念框架分析、线性追溯、定义质量评估（参考 ISO 704:2022）以及对具体技术细节（如 LoRA、RAG、激活等）进行可操作性阐述。

**📊 数据集**

使用的数据集包括：①学术文献数据库（Scopus、Web of Science、IEEE Xplore）检索出的 896 篇论文；②手工搜集的 80+ 监管、标准与技术政策文件，涵盖欧盟、美国、中国等主要司法管辖区。

**📈 对比分析**

该工作不以实验性能为评估指标，而是通过比较与对比已有定义的条理性、清晰度与法律可操作性来评估新定义的优势；在案例研究（AlienChat、ChatGPT、Clearview AI）中展示新定义在责任划分上的可行性和逻辑一致性。

**⚠️ 局限性**

局限性包括：①聚焦当下以 Transformer 为主的神经网络 AI，未覆盖符号推理、强化学习等传统或混合 AI；②对法规文本的解释依赖于作者主观判断；③缺乏实证验证新定义在实际监管执法中的效果；④随技术快速演进，定义仍需定期修订。

---

## 22. QuantumX: an experience for the consolidation of Quantum Computing and Quantum Software Engineering as an emerging discipline

**arXiv ID:** 2603.10621 | [PDF](https://arxiv.org/pdf/2603.10621v1)

**作者:** Juan M. Murillo `[一作]` (Universidad de Extremadura), Fernando Plou `[通讯]` (Universidad de Oviedo)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5028779362)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

组织并记录了QuantumX第一届研讨会，聚集了西班牙多所高校与研究中心的量子软件工程团队，系统梳理并展示了当下在量子编程、混合架构、质量模型、测试与治理等方面的最新研究与工具；

**💡 创新点**

创新之处在于首次将传统软件工程方法（质量评估、测试、编排、抽象等）与量子计算紧密结合，形成跨学科的实践框架，并通过网络（RIPAISC、QSpain）实现社区协同与资源共享；

**🔧 技术方法**

主要技术包括量子编程语言与中间层（如QCRAFT Scheduler、Locus、Quantum Integer）、云量子硬件（IBM Quantum、其他NISQ供应商）、自动化质量工具（SonarQube、OpenAPI扩展）、仿真平台（Qiskit、密度矩阵模拟）、以及量子机器学习框架（PyTorch、Qiskit Machine Learning）；

**📊 数据集**

使用的数据集涵盖IBM Quantum实际硬件与仿真测试、MIMIC‑III/IV医学文本、金融多变量时间序列、图着色与量子图搜索实例，以及公开的量子电路与优化问题实例；

**📈 对比分析**

通过与传统算法对比，报告了成本与任务减少约84％、量子突变测试成本降低约94％、以及在混合架构下的性能提升（如QRNN预测误差下降、量子BiLSTM在医学文本分类中的F1提升等），并采用多云量子编排与价格治理模型验证了可扩展性与成本效益；

**⚠️ 局限性**

局限主要体现在：量子硬件的噪声与有限可编程性限制了实验规模，缺乏统一的评测基准与可重复性标准，抽象层级仍偏低，且跨组协作的工具与流程尚未完全成熟，需进一步完善验证与标准化。

---

## 23. Empathy Is Not What Changed: Clinical Assessment of Psychological Safety Across GPT Model Generations

**arXiv ID:** 2603.09997 | [PDF](https://arxiv.org/pdf/2603.09997v1)

**作者:** Michael Keeman `[一作]` (Keido Labs), Anastasia Keeman `[通讯]` (Keido Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 GPT-4o、o4-mini 与 GPT‑5‑mini 在 14 个情绪化对话场景下进行安全性与同理心维度的定量评估，产生 210 条对话共 2,100 条 AI 回复。

**💡 创新点**

首次用临床框架测量 #keep4o 现象，发现同理心不变但出现危机检测提升与建议安全降低的安全性权衡，并提出逐轮轨迹分析与方差指标作为新评估方法。

**🔧 技术方法**

采用 LLM‑as‑a‑judge 自动评分系统 EmpathyC、基于临床维度的多维度 rubric、非参数统计（Kruskal‑Wallis、Mann‑Whitney、Levene）以及逐轮轨迹可视化。

**📊 数据集**

由临床心理学家设计的 14 个心理健康与 AI 伴侣场景脚本，每个场景五轮对话，共 210 个会话共 2,100 条 AI 回复。

**📈 对比分析**

通过对每个模型的平均分、方差及逐轮表现进行非参数检验，结果显示同理心分数无显著差异，危机检测从 GPT‑4o 到 GPT‑5‑mini 提升约 1.0 分，建议安全则下降约 0.4 分，体现安全性权衡。

**⚠️ 局限性**

样本量有限、评估采用 LLM‑as‑a‑judge 可能存在偏差、仅使用预设对话脚本且单一系统提示、仅评估 OpenAI 模型、评估 rubric 仍需外部验证。

---

## 24. Safe and Scalable Web Agent Learning via Recreated Websites

**arXiv ID:** 2603.10505 | [PDF](https://arxiv.org/pdf/2603.10505v1)

**作者:** Hyungjoo Chae `[一作]` (Georgia Institute of Technology), Alan Ritter `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 10199 | [OpenAlex ID](https://openalex.org/A5039096905)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自动克隆真实网站为可执行合成环境的框架，并在该环境中生成可验证任务，让 Web Agent 在安全、可重现的设置下自我进化学习。

**💡 创新点**

核心创新在于把大型语言模型当作环境生成器，完整复制网站的前后端和数据库，并通过 Python SDK 提供内部可验证的任务和奖励，彻底消除对 LLM 判定的不确定性，实现安全、可扩展的自我进化训练。

**🔧 技术方法**

技术手段包括 GPT‑5.2 及其它 LLM 进行代码生成、Coding Agent（Cursor CLI）负责网站克隆与调试、Playwright 自动化交互验证、Python SDK 与数据库交互实现可验证判定、基于可验证奖励的 fine‑tuning（如 Reward‑based Rejection Fine‑Tuning）等。

**📊 数据集**

使用 149 个真实网站（取自 Mind2Web 列表）构建合成环境，生成 7,400 条任务；评估时采用 WebArena（5 个 Docker 化网站）和 Mind2Web‑Online（220 条任务）两个主流基准；与 ADP、Synatra 等公开数据集及闭源 LLM 进行对比。

**📈 对比分析**

与闭源 LLM（GPT‑4o‑mini、GPT‑4o、Claude‑3.5‑Sonnet）以及开源代理（ADP、Synatra）对比，实验表明在 WebArena 上提升 6.06 点（Qwen3‑4B）和 9.09 点（LLaMA‑3.2‑3B‑Instruct）；在 Mind2Web‑Online 上也取得明显收益；在单一网站的自我进化训练中，比 PAE 更稳定、更大幅提升，尤其在 CMS 与购物类网站上效果尤为突出。

**⚠️ 局限性**

局限性包括：编码 Agent 在克隆过程中仍会出现脚本缺失、端口冲突、CORS 等基础设施错误；可验证任务的执行率仅 90%，判定正确率 76%；对媒体文件、支付流程等复杂组件支持有限；在合成环境训练后向真实网站迁移时仍可能存在功能差距。

---

## 25. ADVERSA: Measuring Multi-Turn Guardrail Degradation and Judge Reliability in Large Language Models

**arXiv ID:** 2603.10068 | [PDF](https://arxiv.org/pdf/2603.10068v1)

**作者:** Harry Owiredu-Ashley `[一作]` `[通讯]` (Independent Researcher), Harry Owiredu-Ashley (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了多轮红队框架ADVERSA，利用 fine‑tuned 70B 攻击者、5‑点合规评分表和三判定共识来记录 LLM 在持续攻击中的安全防护轨迹。

**💡 创新点**

创新点在于：① 用连续轨迹取代二元 jailbreak 判定；② 采用三判定共识揭示 LLM 评判者在对抗环境中的可靠性；③ 记录并分析攻击者漂移与拒绝等失败模式。

**🔧 技术方法**

技术手段包括：Llama‑3.1‑70B (ADVERSA‑Red) 通过 QLoRA fine‑tune；三名判定 LLM 为 Claude Opus 4.6、Gemini 3.1 Pro、GPT‑5.2；使用 vLLM 与张量并行；每轮 JSON 日志记录所有交互。

**📊 数据集**

数据集：10,724 条单轮攻击样本，来源于 AdvBench、HarmBench、JailbreakBench 及 GPT‑4o‑mini 生成；实验对话记录构成多轮数据。

**📈 对比分析**

评估方法：15 条对话（5 目标 × 3 受害模型），记录破解率与破解轮；结果显示平均破解率 26.7%，平均破解轮 1.25，说明多数破解在第一轮完成；后续多轮呈现拒绝趋同轨迹。

**⚠️ 局限性**

局限性：样本量极小（每个目标与受害模型仅一轮），攻击者训练与部署分布不匹配导致漂移与拒绝；评估受攻击者可靠性影响；自评判偏差未能量化；未做多种随机种子或长期跟踪；未扩展至更大规模实验。

---

## 26. Bio-Inspired Self-Supervised Learning for Wrist-worn IMU Signals

**arXiv ID:** 2603.10961 | [PDF](https://arxiv.org/pdf/2603.10961v1)

**作者:** Prithviraj Tarale `[一作]` (University of Massachusetts), Sunghoon I. Lee `[通讯]` (University of Massachusetts)

**通讯引用:** 17614 | [OpenAlex ID](https://openalex.org/A5066700427)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用生物启发的子运动理论对手腕IMU信号进行分词，构建Transformer自监督预训练模型；

**💡 创新点**

通过将运动片段作为token，捕捉人体动作的结构化序列性，而非仅局部波形；

**🔧 技术方法**

使用CNN编码子段，Transformer捕获时序关系，masked运动段重建预训练目标；

**📊 数据集**

在NHANES公开大规模数据集（≈28k小时、≈11k受试者）上预训练，并在六个公共HAR基准（UMH、PAMAP、WISDM、MHealth、WHARF、HAD）上进行评估；

**📈 对比分析**

与对齐同样预训练数据的对比自监督基线（contrastive、augmentation-prediction、等长分块masked）以及通用时间序列基础模型（Chronos、Moment）比较，线性探测平均Macro‑F1提升约0.06（与contrastive）或0.21（与augmentation），在小样本和未见转移时表现更佳；

**⚠️ 局限性**

局限包括对单一手腕传感器的依赖，缺乏临床端点验证，且在更大规模多样化数据上效果尚未验证。

---

## 27. Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD

**arXiv ID:** 2603.10933 | [PDF](https://arxiv.org/pdf/2603.10933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 28. ASTER: Attitude-aware Suspended-payload Quadrotor Traversal via Efficient Reinforcement Learning

**arXiv ID:** 2603.10715 | [PDF](https://arxiv.org/pdf/2603.10715v1)

**作者:** Dongcheng Cao `[一作]` (Zhejiang University), Shuo Li `[通讯]` (Zhejiang University)

**通讯引用:** 50341 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出ASTER框架，实现悬挂式四旋翼在极限倒置飞行中实现高效的姿态感知导航。

**💡 创新点**

创新点在于引入混合动力学信息的状态种子HDSS，利用物理一致的逆向动力学初始化解决奖励稀疏问题，实现首次全自动倒置飞行。

**🔧 技术方法**

采用模型无关强化学习（PPO）、逆向动力学状态种子、混合动力学信息、GPU并行仿真训练和硬件层控制。

**📊 数据集**

训练和评估使用Genesis仿真中随机生成的无限轨迹集，并在5.5×5.5×2.5 m捕捉场地进行真实实验。

**📈 对比分析**

与无HDSS基线对比，奖励、成功率和平均速度等指标提升约10倍；在仿真到真实零调优迁移中成功率>80%，速度误差<6 m/s，表现优异。

**⚠️ 局限性**

局限在于对较长绳索或大质量偏差性能下降，需进一步域随机化；目前仅适用于单机悬挂，无法直接推广到多机协同。

---

## 29. KnowDiffuser: A Knowledge-Guided Diffusion Planner with LM Reasoning and Prior-Informed Trajectory Initialization

**arXiv ID:** 2603.10441 | [PDF](https://arxiv.org/pdf/2603.10441v1)

**作者:** Fan Ding `[一作]` (Monash University), Junn Yong Loo `[通讯]` (Monash University)

**通讯引用:** 232 | [OpenAlex ID](https://openalex.org/A5045821623)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 KnowDiffuser 框架，将语言模型产生的语义级 meta‑action 与扩散模型生成的连续轨迹相结合，实现从语义推理到可执行轨迹的闭环规划

**💡 创新点**

创新点在于构建 meta‑action–先验轨迹库，将高层语义指令映射为可执行轨迹模板，并采用两步截断去噪方式显著提升推理速度与轨迹质量，首次实现高层语义与低层物理规划的紧耦合

**🔧 技术方法**

使用 GPT‑4o 进行场景理解和 meta‑action 输出，采用 Diffusion Transformer (DiT) 进行轨迹生成，结合先验轨迹初始化与两阶段去噪、VPSDE 训练策略

**📊 数据集**

在 nuPlan 大规模真实驾驶数据集上进行训练与评估，样本约 50,000 条 10s 驾驶序列，采用 8s 预测区间的 16 采样点

**📈 对比分析**

与多种基线（rule‑based、RL、CNN、Transformer、Diffusion）对比，在 open‑loop 中 8sADE 0.298、8sFDE 0.568、MR 0.021，显著优于 GUMP‑m 与 CKS‑1.5b；在 closed‑loop 中 Val‑R 81.25、Test‑R 81.10，超过 PlanTF 与 GameFormer，显示出更高成功率和鲁棒性

**⚠️ 局限性**

依赖大规模 LLM 作为高层决策器，LLM 规模对性能影响显著；当前方法仍需在实时推理延迟、复杂场景下的 meta‑action 多样性和多模态感知方面进一步提升

---

## 30. HiFIVE: High-Fidelity Vector-Tile Reduction for Interactive Map Exploration

**arXiv ID:** 2603.10270 | [PDF](https://arxiv.org/pdf/2603.10270v1)

**作者:** Tarlan Bahadori `[一作]` (University of California), Ahmed Eldawy `[通讯]` (University of California)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对向量瓦片的可视化感知缩减框架（HiFIVE），通过优化记录与属性的保留来在保证可视化质量的前提下显著压缩瓦片大小。

**💡 创新点**

创新点包括：① 将瓦片缩减问题建模为可视化感知的优化问题，并定义了基于像素权重的 Jensen‑Shannon 失真度量与逆熵加权的瓦片失真度量；② 提出两阶段处理（三分法与稀疏化），其中稀疏化采用混合整数线性规划同时考虑几何显著性与属性信息损失；③ 引入轻量级三分法实现对极大瓦片的可扩展预处理。

**🔧 技术方法**

核心技术有：混合整数线性规划（MILP）、像素级渲染与属性分布统计、信息理论中的 KL‑Divergence、Mapbox Vector Tile (MVT) 标准、Spark 并行计算框架，以及多种几何简化与属性量化策略。

**📊 数据集**

使用的公开数据集包括：eBird（约 9.35 TB 点数据）、OSM_Roads（31 GB 线数据）、OSM_Buildings（187 GB 面数据）、Postal_Codes、Roads_NA、Counties 等，实验覆盖了从 10% 到 100% 的随机子集，覆盖点、线、面三种几何类型。

**📈 对比分析**

与基线 AID*（服务器端渲染）和 Tippecanoe（向量瓦片裁剪）比较，HiFIVE 在 32 KB–4 MB 的瓦片大小预算下，平均 SSIM 最高、RMSE 最低、PSNR 最高；运行时间与瓦片单元数呈线性关系，Spark 并行可处理十亿级记录的全量数据；在 1 TB 级数据上完成瓦片生成不超过 1 小时。

**⚠️ 局限性**

局限性包括：① 归约问题 NP‑hard，MILP 方案对极大瓦片仍会产生显著求解开销；② 需要手工调节 α、三分阈值等参数；③ 目前主要针对几何显著性与属性熵，可能无法完全捕捉复杂用户自定义样式的语义需求。

---

## 31. Context Over Compute Human-in-the-Loop Outperforms Iterative Chain-of-Thought Prompting in Interview Answer Quality

**arXiv ID:** 2603.09995 | [PDF](https://arxiv.org/pdf/2603.09995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 32. VERI-DPO: Evidence-Aware Alignment for Clinical Summarization via Claim Verification and Direct Preference Optimization

**arXiv ID:** 2603.10494 | [PDF](https://arxiv.org/pdf/2603.10494v1)

**作者:** Weixin Liu `[一作]`, Zhijun Yin `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种基于主张验证器的直接偏好优化（DPO）框架，用于生成符合电子健康记录（EHR）证据的简短住院历程（BHC）摘要。

**💡 创新点**

创新点在于：①构建轻量级检索增强型主张验证器，将单词级三分类（支持、未支持、未涉及）作为可解释的中间监督信号；②利用验证器的高置信度反驳（Not Supported）来构造长度与覆盖度受约束的偏好对；③将这些偏好对通过DPO蒸馏为单样本摘要模型，避免运行时重排序。

**🔧 技术方法**

技术包括检索增强验证（BM25+多层检索）、低秩适配（LoRA）与4-bit量化的8B LLM训练、单词级三分类判定、偏好挖掘的覆盖度/长度/矛盾惩罚公式、以及DPO对偏好对的优化（TRL库）。

**📊 数据集**

使用的基准数据集是MIMIC-III-Ext-VeriFact-BHC，包含100位ICU患者、125次住院记录、每人约25条临床笔记，并附有人类标注的主张级事实标签。

**📈 对比分析**

与基线（单样本SFT、Best‑of‑K重排序）比较，DPO在本地验证器和外部GPT‑4o判定下的Not Supported率分别从10.7%→1.9%和11.6%→6.4%，同时保持或提升有效性和摘要长度，表明显著提升事实性且未出现“少说”退化。

**⚠️ 局限性**

局限包括：小样本（100位患者）且仅来自ICU，检索依赖导致可能缺失或误检证据；评估主要依赖自动判定器，未进行临床最终审阅；对检索窗口、分层深度等超参数的敏感性未做系统性搜索。

---

## 33. Spatial self-supervised Peak Learning and correlation-based Evaluation of peak picking in Mass Spectrometry Imaging

**arXiv ID:** 2603.10487 | [PDF](https://arxiv.org/pdf/2603.10487v1)

**作者:** Philipp Weigand `[一作]` (Technical University of Applied Sciences Mannheim), Oliver Wasenmüller `[通讯]` (Faculty of Biosciences Heidelberg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出自监督空间峰学习网络 S3PL，用于 MSI 数据的空间结构峰挑选，并开发基于专家分割掩模的 Pearson 相关评价方法。

**💡 创新点**

创新点包括：①将空间信息融入自监督 3D 自编码器学习注意力掩模，实现对空间结构峰的自动筛选；②提出多阈值 Pearson 相关评估，利用专家注释分割掩模在真实 profile MSI 数据上实现客观、可迁移的性能评估。

**🔧 技术方法**

技术手段：3D 卷积自编码器 + sigmoid 注意力掩模；TIC 归一化、MSE 损失、Adam 优化；Pearson 相关系数与多阈值 F1（mSCF1）评估；批量化训练与冻结卷积后峰挑选。

**📊 数据集**

使用的公开数据集：三大 profile MSI 数据集（GBM、RCC、CAC）均带有专家分割掩模；以及一组 MALDI-TOF GIST 数据用于定性对比；每个数据集覆盖不同离子源和 m/z 维度。

**📈 对比分析**

与 msiPL、Lieb、MALDIquant、SPUTNIK 等现有峰挑选方法对比；在所有阈值 F1 和平均 mSCF1 上 S3PL 均实现 9–11% 的提升；在 GIST 示例中可视化显示 S3PL 仅挑选空间结构良好的峰，避免噪声峰。

**⚠️ 局限性**

局限性：需要 profile 数据且 m/z bin 一致；参数（patch size、z、kernel depth 等）需针对数据手动调优；峰数 n 仍需人工指定，自动化选择尚未实现；评估依赖高质量专家分割掩模，若掩模不完整会影响结果；公开带掩模的 profile MSI 数据量有限。

---

## 34. Update-Free On-Policy Steering via Verifiers

**arXiv ID:** 2603.10282 | [PDF](https://arxiv.org/pdf/2603.10282v1)

**作者:** Maria Attarian `[一作]` (Google DeepMind), Igor Gilitschenski `[通讯]` (University of Toronto)

**通讯引用:** 2659 | [OpenAlex ID](https://openalex.org/A5078356081)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种“更新自由的自我策略导向（Update-Free On-Policy Steering）”方法，利用机器人自身的评估轨迹（成功与失败）训练验证器，在测试时通过最佳- N 采样或分类器引导优化动作，从而提升已有的行为克隆（BC）扩散策略性能。

**💡 创新点**

创新点在于：①仅利用已有的策略评估轨迹，无需额外数据采集或模型微调；②训练轻量级验证器（成功预测或时间到成功预测）并在推理时即时引导动作；③展示了在模拟和真实双臂任务中，显著提升成功率（平均提升约49%）且不改动基准模型参数。

**🔧 技术方法**

核心技术包括：行为克隆的扩散策略（DDPM/DM），基于轨迹的成功/失败二分类器或时间到成功回归器（MLP + 对比辅助损失），最佳- N 采样或分类器引导采样（Classifier Guidance）在逆扩散过程中。

**📊 数据集**

实验使用：Robomimic 4个仿真任务（Transport、Square，低维/图像版本）以及Aloha双臂机器人 5 个真实任务（Block pick-place、Ball to bowl、Transport、Pen cap insertion、Cup stacking）。基准数据来自 Robomimic 的多人工示范（MH、PH）。

**📈 对比分析**

与 SAILOR、DSRL 等基线相比，在仿真和真实实验中，该方法在同等数量的策略评估轨迹下实现了更高的成功率，平均提升 25%–80%，在 5 个真实任务中超过基准 49% 的平均增益。

**⚠️ 局限性**

局限性包括：仅针对单任务策略；对验证器训练和调参（如引导强度）需人工标注成功/失败标签；分类器引导对强度高度敏感，可能在真实场景中带来安全风险；多任务或更复杂环境下的可扩展性尚未验证。

---

## 35. Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety

**arXiv ID:** 2603.10044 | [PDF](https://arxiv.org/pdf/2603.10044v1)

**作者:** David Gringras `[一作]` `[通讯]` (Harvard University), David Gringras (Harvard University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对六大前沿模型在四种部署架构下进行大规模安全评估，系统探讨了 scaffolding 与评估格式对安全指标的影响。

**💡 创新点**

首次将 pre‑registration、blinding、equivalence testing 与 specification‑curve 分析结合到 AI 安全评估中，并揭示了“格式‑依赖测量”与“scaffold 结构”共同导致的安全评分漂移。

**🔧 技术方法**

使用逻辑回归、Cluster‑Robust 标准误、TOST 等统计方法，对不同 scaffold（Map‑Reduce、ReAct、Multi‑Agent）与评估格式（MC vs. OE）进行严格比较，并实现了 option‑preserving Map‑Reduce 以隔离格式效应。

**📊 数据集**

数据集包括 4,400 条匹配项（MC 与 OE），62,808 条原始评估结果，以及 12,000 条 sycophancy 检测样本，覆盖 BBQ、TruthfulQA、XSTest/OR‑Bench、AI Factual Recall 等四个安全基准。

**📈 对比分析**

发现 Map‑Reduce 在 MC 格式下导致 7.3pp 的安全下降（NNH=14），但在保持 MC 选项的 variant 下恢复 40–89%；ReAct 与 Multi‑Agent 的效应均在 ±2pp 范围内，几乎无实质性安全损伤；OE 格式普遍提升安全得分 5–20pp，远大于 scaffold 影响。

**⚠️ 局限性**

局限性包括：仅评估代理层面安全特性，无法覆盖更深层次的风险（如逃避、误导）；测评基准本身可能存在构念缺陷；以及模型内部对安全属性的编码深度差异导致结果解释复杂。

---

## 36. CodePercept: Code-Grounded Visual STEM Perception for MLLMs

**arXiv ID:** 2603.10757 | [PDF](https://arxiv.org/pdf/2603.10757v1)

**作者:** Tongkun Guan `[一作]` (Shanghai Jiao Tong University), Xiaokang Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 25971 | [OpenAlex ID](https://openalex.org/A5019708391)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CodePercept 方法，通过可执行 Python 代码提升 MLLM 在 STEM 视觉推理中的感知能力，并构建 ICC-1M 1M 图像-文本-代码三元组数据集和 STEM2Code-Eval 评测基准。

**💡 创新点**

创新性在于将可执行代码视为感知真值，用代码驱动的 caption 生成和 image‑to‑code 翻译双任务训练，以及以代码重构为直接评估指标的 STEM2Code-Eval 基准。

**🔧 技术方法**

使用大规模图像到代码的自回归生成、代码分析+执行追踪、强化学习 (GRPO) 进行可执行性与视觉相似度双重奖励，以及多模态 Transformer 作为基座。

**📊 数据集**

主要使用公开 STEM 图像数据集（MathVision、MathVista 等）构建 ICC-1M（1M 图像-文本-代码）并在 STEM2Code-Eval 的 1k 样本上评测。

**📈 对比分析**

在 captioner‑solver 框架下与多款 SOTA MLLM 进行对比，CodePercept 在 4B/8B/32B 参数规模上均提升 2-3% 甚至超越更大模型；在 STEM2Code-Eval 上实现 90% 以上的执行率和 60% 以上的视觉得分，显著优于基线。

**⚠️ 局限性**

局限在于对代码生成仍受 LLM 语法错误影响，且仅覆盖 Python 可视化库场景，复杂 3D 结构和动态场景的感知尚未充分验证；数据生成的多样性依赖模板，可能导致泛化不足。

---

## 37. Utility Function is All You Need: LLM-based Congestion Control

**arXiv ID:** 2603.10357 | [PDF](https://arxiv.org/pdf/2603.10357v1)

**作者:** Neta Rozen-Schiff `[一作]` (Technische Universität Berlin), Stefan Schmid `[通讯]` (Technische Universität Berlin)

**通讯引用:** 11429 | [OpenAlex ID](https://openalex.org/A5066080641)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）的框架，利用不同的提示与进化策略自动生成拥塞控制协议的效用函数，并通过网络实验台验证其性能；

**💡 创新点**

创新点在于：①首次将LLM用于生成多目标、异构连接场景下的效用函数；②系统性比较零射击、一次性提示、数学链式推理与进化反馈四种提示策略；③展示进化反馈和链式推理可在多种网络情境下显著提升现有最优拥塞控制协议的性能（37%-142%）；

**🔧 技术方法**

技术包括：LLM（GPT‑5）生成代码、提示工程、数学链式推理（Math‑CoT）、进化算法、基于梯度下降的效用函数模型、Linux tc 网络控制、Python/Cpp 编译与自动化测试；

**📊 数据集**

数据集与实验环境：五种不同最小/最大带宽需求的异构连接；三种代表性网络情境（卫星、蜂窝、宽带）以及受限云与“野外”实验；通过 tc 模拟的瓶颈、RTT、丢包等参数；

**📈 对比分析**

比较方法：使用“满足率”（平均发送速率/最小需求）衡量协议性能，并与现有最优协议 Hercules 对比；实验结果表明零射击、Math‑CoT 与进化策略在各场景下均超过 Hercules，提升幅度从 37%（蜂窝）到 142%（宽带）；一射击策略表现最差；跨场景测试显示进化与 Math‑CoT 具有良好的泛化能力；

**⚠️ 局限性**

局限性：①性能高度依赖提示质量，过度提示（一次性）会导致下降；②轻量级 LLM（如 GPT‑OSS‑20B）产生约 80% 编译失败；③实验仅覆盖有限几种网络场景，可能不适用于更复杂或极端网络条件；④需额外手工验证与沙箱安全，生成代码易出现语法错误或不兼容。

---

## 38. Naïve Exposure of Generative AI Capabilities Undermines Deepfake Detection

**arXiv ID:** 2603.10504 | [PDF](https://arxiv.org/pdf/2603.10504v1)

**作者:** Sunpill Kim `[一作]` (Hanyang University), Jae Hong Seo `[通讯]` (Hanyang University)

**通讯引用:** 1070 | [OpenAlex ID](https://openalex.org/A5045700517)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究揭示了通用生成式 AI（GAI）系统在暴露推理与图像细化能力时，如何使现代深度伪造检测器失效；通过让对话式 AI 对图像真实性进行评估、给出结构化的缺陷说明，并将这些说明作为细化指令，能够在保持身份与姿态的同时显著提升图像自然度并逃逸检测。

**💡 创新点**

创新点在于提出了利用 GAI 的“真实性评估+结构化反馈+语义保持细化”这一攻击链，将系统本身的对齐与解释功能转化为攻击手段；同时系统性评估了商业与开源 GAI 在此情境下的风险差异，并指出现有安全与检测框架的结构性缺陷。

**🔧 技术方法**

技术手段包括：①对话式多模态模型（如 ChatGPT、Gemini、Qwen、Flux AI）的文本推理与图像生成；②基于模型生成的结构化反馈构造细化提示；③利用现有深度伪造与 AI 生成图像检测器（GenD、M2F2-Det、Hive-DF、UnivFD、D^3、Hive-AI）对改写图像进行评估；④使用商业人脸识别 API 验证身份保持率。

**📊 数据集**

使用的主要数据集为 FaceForensics++（五种伪造方法共 100 张）作为伪造样本，FFHQ 作为真实样本；在扩展实验中还使用 Tiny-GenImage 的人工合成图像。

**📈 对比分析**

比较方法为在不同细化提示（从无细化到高细化、实例特定）下测量六类检测器的检测率（DR）以及身份保持率（IPR），并在两种阈值（τ_99、τ_90）下评估。结果显示，传统深度伪造检测器在细化后 DR 迅速下降至接近 0%，而 AI 生成图像检测器在严格阈值下的 DR 亦大幅下滑；同时身份保持率高达 85% 以上，证明细化既逃避检测又保持语义。

**⚠️ 局限性**

局限性包括：①实验仅基于提示交互，未深入探究模型内部机制；②未覆盖所有可能的 GAI 系统与检测器，结果可能因模型版本差异而异；③仅评估了视觉真实性与身份保持，未探讨对下游任务（如人脸识别、情感识别）的潜在影响；④实验环境使用了公开 API，实际部署场景可能存在额外安全过滤或使用限制。

---

## 39. Gradient Flow Drifting: Generative Modeling via Wasserstein Gradient Flows of KDE-Approximated Divergences

**arXiv ID:** 2603.10592 | [PDF](https://arxiv.org/pdf/2603.10592v1)

**作者:** Jiarui Cao `[一作]` (Chinese University of Hong Kong), Yuxin Liu `[通讯]` (Civil Aviation University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出梯度流漂移框架，将漂移模型视为基于KDE的Wasserstein梯度流，实现一阶生成器训练。

**💡 创新点**

通过精确证明漂移模型等价于KL梯度流，统一MMD和混合f散度生成器，扩展至黎曼流形并给出可识别性与能量耗散分析。

**🔧 技术方法**

使用KDE估计、Wasserstein梯度流、f散度（KL、逆KL、χ²）、MMD、混合散度、Riemannian核设计及停梯度损失。

**📊 数据集**

在二维合成基准（如Swiss‑roll）进行实验，原始漂移模型使用Laplace核，比较不同核与散度。

**📈 对比分析**

对比不同散度与核的速度场和采样结果，混合逆KL+χ²梯度流在模式覆盖与精度上优于单一KL或MMD；Laplace核导致数值不稳定。

**⚠️ 局限性**

局限在于KDE在高维下的方差大、需大量样本，混合散度和核选择对收敛稳定性敏感，未在大规模真实数据上验证。

---

## 40. Rethinking Gaussian Trajectory Predictors: Calibrated Uncertainty for Safe Planning

**arXiv ID:** 2603.10407 | [PDF](https://arxiv.org/pdf/2603.10407v1)

**作者:** Fatemeh Cheraghi Pouria `[一作]` (University of Illinois at Urbana-Champaign), Katherine Driggs-Campbell `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种新的损失函数，用于训练双变量高斯轨迹预测模型，使预测的置信区间更可靠，进而提升不确定性感知的运动规划效果。

**💡 创新点**

创新点在于通过核密度估计匹配预测的平方马氏距离经验分布与理论卡方分布，并加入MSE项，显著改善传统负对数似然损失导致的过度/不足置信校准问题。

**🔧 技术方法**

主要技术包括：KDE核密度估计、卡方分布匹配、MSE正则化、以及将校准后的预测集成到基于MPC的不确定性感知规划器中。

**📊 数据集**

实验使用了真实世界行人轨迹数据集 ETH、UCY（ETH、HOTEL、UNIV、ZARA）进行训练与测试。

**📈 对比分析**

通过与原始NLL训练的同架构模型（LSTM、Social‑LSTM、Social‑STGCNN、DSTIGCN等）在 ADE/FDE、ΔESV、MPC 成功率、碰撞率等指标上对比，实验显示校准后的模型在置信水平上更贴近理论，且在规划任务中成功率提高、碰撞率下降，但导航时间略有增加。

**⚠️ 局限性**

限制在于仅针对单峰高斯分布，未推广到多模态 GMM 预测器；在小数据集上可能出现过拟合，对离散分布的泛化仍有限。

---

## 41. Shape Control of a Planar Hyper-Redundant Robot via Hybrid Kinematics-Informed and Learning-based Approach

**arXiv ID:** 2603.10402 | [PDF](https://arxiv.org/pdf/2603.10402v1)

**作者:** Yuli Song `[一作]` (Singapore MIT Alliance for Research and Technology), Cecilia Laschi `[通讯]` (National University of Singapore)

**通讯引用:** 20799 | [OpenAlex ID](https://openalex.org/A5045065209)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种混合基于运动学和学习的形状控制框架（SpatioCoupledNet），用于平面超冗余机器人，实现了对多段柔性段的全身高精度跟踪；

**💡 创新点**

创新点在于层级化神经网络结构与双向递归捕捉段间空间耦合，同时引入动态置信门控，将物理先验与数据驱动模型按状态自适应融合；

**🔧 技术方法**

采用物理基础的PCC运动学模型、Bi‑GRU递归网络、MLP特征提取、置信门控、分块DLS逆解、可微正向运动学等技术实现闭环控制；

**📊 数据集**

使用自建的五段平面超冗余机器人的实验数据集，包括易、中、难三种形态配置下的跟踪轨迹与传感反馈；

**📈 对比分析**

与纯解析模型和纯数据驱动模型进行对比，混合控制在易/中/难配置下的稳态误差分别降至0.74 mm、5.84 mm、6.55 mm，收敛速度和执行成本显著优于两种基线，动态避障时平均末端误差仅为10.47 mm；

**⚠️ 局限性**

局限性包括对更高段数机器人的可扩展性尚未验证，网络结构和推理速度需进一步优化，且在极端负载或未知接触时的在线适应性仍待提升。

---

## 42. Discovery of a Hematopoietic Manifold in scGPT Yields a Method for Extracting Performant Algorithms from Biological Foundation Model Internals

**arXiv ID:** 2603.10261 | [PDF](https://arxiv.org/pdf/2603.10261v1)

**作者:** Ihor Kendiukhov `[一作]` `[通讯]` (University of Tübingen), Ihor Kendiukhov (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

发现并提取了scGPT内部的血液发育低维流形，构建为可独立部署的算法；

**💡 创新点**

首次通过机制可解释性从生物学基础模型中提取可竞争、可压缩、可解释的算法，并压缩至单一注意力头，解析出四因素核心；

**🔧 技术方法**

使用了注意力漂移、层级特征映射、LET目标、SVD低秩压缩、稀疏压缩、因子消融等机制可解释技术，以及三阶段提取管线；

**📊 数据集**

利用Tabula Sapiens细胞集、严格非重叠外部面板、独立多供体免疫面板以及肺部面板进行验证；

**📈 对比分析**

在88个捐献者holdout分割上与scVI、Palantir、DPT、CellTypist等基准比较，提取算法在伪时间深度对齐上获得最高Spearman相关，并在部分分类指标（CD4/CD8、单核/巨噬细胞AUROC）领跑；同时速度提升34.5×、参数量减少≈1000×；

**⚠️ 局限性**

外部验证有限，基于ontology的生物尺杆缺乏绝对真实度，基准覆盖不全，低秩压缩至rank‑64以上可保留性能，低于此显著下降，稀疏压缩版准确度大幅下滑，因子消融仅揭示当前模型依赖情况，未评估重训练恢复能力。

---

## 43. Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis

**arXiv ID:** 2603.10846 | [PDF](https://arxiv.org/pdf/2603.10846v1)

**作者:** Yujie Zheng `[一作]` (Shanghai Jiao Tong University), Muning Wen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 361 | [OpenAlex ID](https://openalex.org/A5049802452)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EvoKernel，一个基于自演化记忆的价值驱动代理框架，用来在数据稀缺的 NPU 编程领域实现内核合成与优化。

**💡 创新点**

创新点在于将内核合成建模为基于记忆的 MDP，并通过阶段特定的价值驱动检索学习 Q 值，支持从无经验到高性能的无监督进化。

**🔧 技术方法**

采用自演化记忆、价值驱动检索、Monte-Carlo 更新、跨任务记忆共享以及多门验证器等技术。

**📊 数据集**

主要使用改造后的 KernelBench（Ascend C）作为基准，额外评估 Attention Set 与 DeepSeek mHC kernels。

**📈 对比分析**

与 Pass@k、Refinement、Codex 等基线对比，在 30 次迭代预算下，EvoKernel 将正确率从 11% 提升至 83%，编译率从 11% 提升至 98.5%，并在首次可行合成后实现中位 3.6 倍的加速。

**⚠️ 局限性**

局限在于仍依赖大型 LLM 的推理能力，记忆更新仍以离线 Q 值方式，且在极端稀缺数据或全新架构时的泛化尚未充分验证。

---

## 44. A Secure Splitting and Acceleration Strategy for TCP/QUIC in Interplanetary Networks

**arXiv ID:** 2603.10437 | [PDF](https://arxiv.org/pdf/2603.10437v1)

**作者:** Jianhao Yu `[一作]` (Nanjing University), Kanglian Zhao `[通讯]` (Nanjing University)

**通讯引用:** 1596 | [OpenAlex ID](https://openalex.org/A5038720526)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了基于非透明安全代理（NTSP）的IPN加速策略，能够在深空链路上对端到端加密的TCP/QUIC连接进行拆分，并通过速率控制、可调FEC和后压流控实现高吞吐、低延迟的传输。

**💡 创新点**

创新点包括：①在保留应用层安全的前提下实现端到端加密流的连接拆分；②结合预调度链路的速率控制与可变冗余FEC，实现无重传的低延迟失真恢复；③基于排队论推导的后压流控与缓冲区尺寸公式，避免缓冲区膨胀与拥塞；④在单一代理中统一处理多条并发流，实现高效多路复用。

**🔧 技术方法**

使用技术有：Go语言 + quic-go实现代理；TLS 1.3密钥导出+AES‑GCM的ALDE加密；流控与拥塞控制改为速率基CC；改进的STREAM+REPAIR帧实现SC FEC；M/M/1/K排队模型用于缓冲区尺寸计算；Mininet模拟Earth–Moon链路。

**📊 数据集**

采用的“数据集”为在Mininet中构建的四节点线性拓扑，模拟Earth–Moon瓶颈链路（上行1 Mbps，下行10 Mbps，2.01 s往返时延），并注入不同丢包率（0.1%、1%、5%）。文件传输实验使用35 MB和200 MB测试文件。

**📈 对比分析**

对比方法：在相同链路与丢包率下，将PEPspace与标准TCP（Cubic、BBR）、PicoQUIC、三大开源PEP（PEPsal、Kcptun、PEPesc）及DTN（BP/LTP）进行对比。评估指标包括goodput、Coefficient of Variation（CoV）、平均延迟、队列宽度、RTO、Jain公平性、SFI。结果显示PEPspace在所有丢包率下实现接近链路容量的goodput（最高≈9 Mbps），CoV最低，RTT维持在物理链路RTT附近，且无严重拥塞或缓冲区膨胀；相比之下，传统TCP/QUIC以及部分PEP在高丢包或多流情况下出现明显拥塞、延迟急剧上升。

**⚠️ 局限性**

局限性：①对元数据和流量特征的流量分析仍可见；②后压流控与缓冲区尺寸公式基于Poisson/指数假设，实际深空链路可能存在非泊松流；③缺乏真实的深空物理信道模型（如太空天气、相位噪声等）；④实验仅在单一Earth–Moon情景下验证，未覆盖更远行星或复杂多跳；⑤ALDE层引入额外密钥管理开销，且不解决QoS调度与多流公平性细粒度问题；⑥PoC的IP/DTN互操作示例仅为单流，未解决消息语义匹配、重排序与QoS调度的挑战。

---

## 45. Instant Runoff Voting on Graphs: Exclusion Zones and Distortion

**arXiv ID:** 2603.10290 | [PDF](https://arxiv.org/pdf/2603.10290v1)

**作者:** Georgios Birmpas `[一作]` (University of Liverpool), Paul Spirakis `[通讯]` (University of Liverpool)

**通讯引用:** 7852 | [OpenAlex ID](https://openalex.org/A5011756177)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在树形图的离散度量空间下，构造了多项式时间算法来判定任意集合是否为即时淘汰投票（IRV）的排除区，并求解最小排除区；同时推广了排除区判定与最小化问题在所有满足强迫淘汰（SFE）性质的确定性基于排名的淘汰规则下的NP难度；此外给出IRV在路径、双星、完全二叉树及一般无权图上的失真上界与下界；

**💡 创新点**

首次实现树结构下排除区的多项式判定与最小化，提出Kill成员资格测试与动态规划框架；提出强迫淘汰属性，将排除区硬性结果推广至所有确定性淘汰规则；给出多种图形下IRV失真的近似极限，为理论评估提供新的基准；

**🔧 技术方法**

使用图论与动态规划（底向上DP）实现Kill判定；引入Antichain正则化、边界折叠与两受体引理以压缩DP状态；采用强迫淘汰（SFE）属性进行归约证明；分析失真时利用距离度量与社群成本的组合分析；

**📊 数据集**

本研究为理论工作，无使用实际数据集；所有结果基于抽象的无权图模型与离散度量空间构造；

**📈 对比分析**

与先前的NP难度、NP硬度和失真下界/上界结果对比，证明树形图下问题可多项式解决；在路径、双星、完全二叉树中提供几乎紧确的失真上界（≤2、≤5/3、≤3）和对应下界（≥9/5、=5/3、≥1.7）；对一般无权图给出失真下界Ω(√ln m)与上界O(ln m)；

**⚠️ 局限性**

仅适用于树形图，树外一般图仍为NP/ co‑NP 难；方法仅适用于确定性淘汰规则，随机化转移规则不满足SFE；DP时间与空间为高阶多项式（O(n^13)等），实际可行性有限；

---

## 46. HTM-EAR: Importance-Preserving Tiered Memory with Hybrid Routing under Saturation

**arXiv ID:** 2603.10032 | [PDF](https://arxiv.org/pdf/2603.10032v1)

**作者:** Shubham Kumar Singh `[一作]` `[通讯]`, Shubham Kumar Singh

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套双层内存体系（HTM‑EAR），通过 HNSW ANN、重要性感知淘汰、混合路由与交叉编码重排序，帮助长运行智能体在内存受限时保持关键事实并忘却无关历史。

**💡 创新点**

创新点在于：①重要性与使用频率相结合的淘汰分数；②基于相似度阈值与实体覆盖度的分层路由策略；③将二进制编码与交叉编码结合以提升最终召回精度；④在高饱和度场景下对各组件的 ablation 与 Pareto 性能分析。

**🔧 技术方法**

使用技术包括：HNSW ANN 索引、E5‑large bi‑encoder 生成嵌入、交叉编码器重排序（基于 MS MARCO 预训练）、自定义重要性评分、α/β/λ/γ 经验加权、LRU 退化实验。

**📊 数据集**

数据集：①5 份 15,000 条合成事实与查询（不同种子）用于饱和度测试；②2,000 条真实 BGL 日志条目作为现实场景验证。

**📈 对比分析**

通过与 oracle（无淘汰）、无交叉编码、无路由、LRU 等模式对比，评估指标为 MRR（active、history）、查询延迟与丢失的重要事实数。全模型在饱和场景的 active MRR 达到 1.000、history 0.215，接近 oracle（0.997/0.990）；在 BGL 上 MRR 0.336，oracle 0.370；LRU 延迟最快（21.1 ms）但丢失 2416 个重要事实，性能最差。

**⚠️ 局限性**

局限性：①仅在固定 500/5,000 容量下评估；②重要性权重与阈值选择经验性、未进行超参数调优；③仅验证了单一日志数据集，缺乏更广泛的实测；④未实现在线阈值自适应或理论性能保证。

---

## 47. AdaClearGrasp: Learning Adaptive Clearing for Zero-Shot Robust Dexterous Grasping in Densely Cluttered Environments

**arXiv ID:** 2603.10616 | [PDF](https://arxiv.org/pdf/2603.10616v1)

**作者:** Zixuan Chen `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**通讯引用:** 13130 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AdaClearGrasp闭环框架，实现稠密混乱环境下的自适应清除与零样本柔性抓取。

**💡 创新点**

创新点在于将视觉语言模型与几何感知RL抓取策略结合，实现高层决策驱动的清除与闭环重规划。

**🔧 技术方法**

技术包括预训练视觉语言模型Qwen3-VL-32B-Instruct、高层语义规划、Model Context Protocol、几何感知RL抓取策略GeoGrasp以及闭环视觉反馈。

**📊 数据集**

使用ManiSkill3模拟环境，七类目标物（来自YCB）与三个拥挤度等级的210个任务场景，以及18个真实场景的评测。

**📈 对比分析**

与VLM Scaffolding、直接GeoGrasp、无重规划AdaClearGrasp对比，仿真中取得89%-76%成功率，真实场景70%成功率，明显优于基线。

**⚠️ 局限性**

局限在于对复杂动态障碍的处理仍不够鲁棒，且在极端拥挤度下仍会出现碰撞或抓取失败。

---

## 48. A PUF-Based Approach for Copy Protection of Intellectual Property in Neural Network Models

**arXiv ID:** 2603.10753 | [PDF](https://arxiv.org/pdf/2603.10753v1)

**作者:** Daniel Dorfmeister `[一作]` (Software Competence Center Hagenberg), Hannes Sochor `[通讯]` (Software Competence Center Hagenberg)

**通讯引用:** 26 | [OpenAlex ID](https://openalex.org/A5044337073)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出利用物理不可克隆功能（PUF）对神经网络模型的部分权重进行加密，从而使模型只能在特定硬件上保持高精度，抑制软件盗版；

**💡 创新点**

创新点在于将PUF产生的唯一硬件指纹与NN权重绑定，只加密少量关键权重即可显著降低复制模型的准确率，既降低了运行时开销，又实现了硬件级别的IP保护；

**🔧 技术方法**

使用了一次性密钥（Vernam）加密算法、PUF（如DRAM Rowhammer或模拟Arbiter PUF）、Python/ TensorFlow实现的加解密框架，以及针对模型权重的随机选择和挑战-响应机制；

**📊 数据集**

实验数据集包括MNIST、Fashion‑MNIST、CIFAR‑10等图像分类数据集、Speech Commands（音频识别）和IMDB 电影评论（文本情感分析）等；

**📈 对比分析**

通过对四个不同模型的稠密、卷积、循环层随机加密不同百分比的权重，测量加密后和解密后在原始机器与克隆机器上的准确率，结果显示仅20%权重加密即可使准确率接近随机分类器，且在目标机器上解密后恢复原精度；

**⚠️ 局限性**

局限性包括：仅针对静态攻击，未考虑动态侧信道攻击；PUF响应时间与加密重量数量对性能有影响；加密过程需要在硬件上重新加密或使用模拟PUF；若攻击者能查询PUF则可绕过保护，需进一步加固。

---

## 49. Distilling LLM Semantic Priors into Encoder-Only Multi-Talker ASR with Talker-Count Routing

**arXiv ID:** 2603.10587 | [PDF](https://arxiv.org/pdf/2603.10587v1)

**作者:** Hao Shi `[一作]` (SB Intuitions), Yui Sudo `[通讯]` (SB Intuitions)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种全编码器的多说话人语音识别框架，通过训练时利用大语言模型（LLM）作为教师，将其语义先验蒸馏到编码器，并在推理时仅使用快速的序列化CTC解码；同时加入说话人计数头（TCH）实现对说话人数的动态预测与分支选择；

**💡 创新点**

创新点在于：①将LLM从推理阶段迁移到训练阶段，只在训练时使用LLM教师蒸馏语义信息；②通过LLM适配实现多说话人条件下的语义指导；③设计动态说话人计数头以支持可变说话人数量；④结合后编码器分离器和序列化CTC，保持推理高效；

**🔧 技术方法**

使用的技术包括：WavLM-Large 作为预训练声学编码器；LLaMA-3.2-1B 作为教师语言模型；轻量级 LoRA 适配器和特定 token 嵌入；LSTM 分离器 + 多头线性投影实现说话人流分离；Serialized CTC 损失；混合注意力-CTC 训练目标；说话人计数头（TCH）采用注意力池化 + MLP；

**📊 数据集**

在 LibriMix 数据集上进行实验，包括 Libri2Mix（两说话人）和 Libri3Mix（三说话人），使用清晰语音（LibriSpeech）与 WHAM! 噪声混合；

**📈 对比分析**

与基准方法（SOT+LLM、SOP+LLM、纯 CTC、传统 Transformer 解码器）对比，本文模型在两说话人条件下可与 LLM 基线相媲美，在三说话人条件下性能更优；RTF 仅为 0.0043（2Mix）/0.0106（3Mix），相比 LLaMA-1B 的 0.1150/0.0981，推理速度提升近 30 倍；

**⚠️ 局限性**

局限性包括：说话人计数头在三说话人混合中准确率不如两说话人，导致分支选择偶尔失效；模型仍依赖大语言模型作为训练教师，训练成本较高；在极端重叠或噪声环境下性能尚未充分验证；

---

## 50. Self-Scaled Broyden Family of Quasi-Newton Methods in JAX

**arXiv ID:** 2603.10599 | [PDF](https://arxiv.org/pdf/2603.10599v1)

**作者:** Ivan Bioli `[一作]` (Università degli Studi di Pavia), Mikel Mendibe Abarrategi `[通讯]` (University of the Basque Country)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现了纯JAX版本的Self-Scaled Broyden家族（包含BFGS、DFP、Broyden及其自缩放变体）和Zoom线搜索，并将其集成进Optimistix优化库；

**💡 创新点**

创新点在于把原本缺失的Zoom线搜索和完整的自缩放Broyden族实现为可直接替换的JAX模块，且实现与Optimistix的接口高度兼容；

**🔧 技术方法**

主要技术是使用JAX进行自动微分和jit加速，构建类层次结构以实现不同的更新公式，并通过JAX的变换（vmap, grad, jit）保证可组合性；

**📊 数据集**

使用了3D Poisson方程的PINN数据集（5000个内部点、800个边界点），网络为3层32单元全连接网络；

**📈 对比分析**

通过将SSBFGS、SSBroyden与标准BFGS、Broyden做迭代次数、损失下降、L²和H¹误差的对比，结果显示自缩放变体在迭代次数和误差收敛上均优于传统方法；

**⚠️ 局限性**

局限在于仅做了单一PINN案例的实验，缺乏更广泛的数据集验证，且实现仍属于技术说明，未提出新的理论或算法改进。

---

## 51. Towards Modeling Situational Awareness Through Visual Attention in Clinical Simulations

**arXiv ID:** 2603.10308 | [PDF](https://arxiv.org/pdf/2603.10308v1)

**作者:** Haoting Gao `[一作]` (University of Michigan), Vitaliy Popov `[通讯]` (University of Michigan)

**通讯引用:** 819 | [OpenAlex ID](https://openalex.org/A5028002882)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对多名医务人员在VR心脏骤停模拟中收集的眼动数据进行 Transition Network Analysis（TNA），构建并分析视觉注意转移网络，量化不同临床角色及情境阶段下的关注结构和流动。

**💡 创新点**

首次将 TNA 应用于医学团队情境意识研究，利用熵和自环率等网络指标捕捉角色间视觉注意的动态重分配，为高保真模拟中的情境意识提供可视化和定量分析框架。

**🔧 技术方法**

眼动追踪、AOI 定义与映射、转移矩阵估计、网络熵与自环率计算、图形可视化、定性动机分析。

**📊 数据集**

40 名临床参与者（10 次模拟，共 20,628 合并注视和 20,526 AOI 转移），包含 7 个 AOI（设备、病人、监护、队友等）。

**📈 对比分析**

对四个临床角色（Airway、CPR、Defib、TeamLead）及两个情境阶段（初始评估 Stage 1 与高强度干预 Stage 5）计算熵与自环率，并使用 Kruskal–Wallis 检验比较。结果显示角色差异显著（p≈0.0015），并揭示角色和阶段间注意结构的明显差异。

**⚠️ 局限性**

仅利用视觉注意数据，未整合语音、心率等多模态信息；样本量有限且仅在 VR 环境，实际临床转移性未知；未进行统计显著性的图案计数或实时反馈评估。

---

## 52. Sabiá-4 Technical Report

**arXiv ID:** 2603.10213 | [PDF](https://arxiv.org/pdf/2603.10213v1)

**作者:** Thiago Laitz `[一作]` (Maritaca AI), Rodrigo Nogueira `[通讯]` (Maritaca AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了Sabiá-4和Sabiazinho-4两款专注巴西葡萄牙语的语言模型，并通过四阶段训练管线提升其在法律文书、长文本处理、指令遵循和代理能力上的表现。

**💡 创新点**

创新点包括在持续预训练阶段结合巴西法律语料实现领域专精，扩展上下文窗口至128K token，采用监督微调与偏好对齐相结合的训练策略，以及对多维度基准（法律写作、长文本、指令、多轮对话、标准化考试与代理任务）进行系统评估，显著降低成本。

**🔧 技术方法**

使用技术包括在Google Cloud TPU v5p/v6e上以JAX框架实现分布式训练，涵盖继续预训练、长上下文训练、监督微调（SFT）以及偏好对齐四个阶段；同时采用功能调用和工具链训练来增强代理能力。

**📊 数据集**

利用的数据集涵盖大规模葡萄牙语通用语料、巴西法律语料、自然长文本、指令与多轮对话数据、合成函数调用示例，以及多项标准化基准（OAB-Bench、Magis-Bench、Brazilian Federal Laws、MRCR、Multi-IF、BRACEval、13个多选考试、Pix-Bench、Ticket-Bench、CLIMB、MARCA）。

**📈 对比分析**

通过与多款同价位及前沿模型在OAB、Magis、法律知识、工具调用、考试、指令跟随与对话等六类基准的定量对比，Sabiá-4在大部分指标上达到或超过前沿模型，并以更低成本位于价格‑准确率图的左上角。

**⚠️ 局限性**

局限性包括受训练数据截止时间限制、长文本评估基准在当前模型已饱和、代理任务表现仍不及顶尖模型、缺乏系统的推理与链式思维能力，未来需扩展至256k上下文并集成更强推理模块。

---

## 53. Stochastic Port-Hamiltonian Neural Networks: Universal Approximation with Passivity Guarantees

**arXiv ID:** 2603.10078 | [PDF](https://arxiv.org/pdf/2603.10078v1)

**作者:** Luca Di Persio `[一作]` (University of Verona), Youness Outaleb `[通讯]` (University of Trento)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种结构保持的随机泊松-汉密尔顿神经网络（SPH-NN），能够从数据学习完整的随机泊松-汉密尔顿系统，包括哈密顿量、耦合、耗散与扩散，并在训练中强制满足反对称耦合与半正定耗散。

**💡 创新点**

创新点在于：① 在神经网络中直接对耦合矩阵和耗散矩阵进行参数化，保证它们始终满足物理约束；② 证明了在紧致集上 SPH-NN 的通用逼近性和弱通过性界限；③ 通过对 Ito 校正的显式生成器不等式，提供了期望下的能量守恒保证。

**🔧 技术方法**

采用 Feed‑forward 神经网络对哈密顿量进行参数化，并用自动微分计算梯度；对耦合矩阵做反对称化处理，对耗散矩阵做低秩分解；通过增量式或条件期望式的漂移损失以及 Euler‑Maruyama 的负对数似然训练扩散；利用 Ito 公式和 Gronwall 定理推导理论性质。

**📊 数据集**

在模拟数据集上评估：噪声耦合的质量-弹簧振子、Duffing 振子以及随机 Van der Pol 振子（全部在二维状态空间上生成）。

**📈 对比分析**

与无结构约束的多层感知机（MLP）基线进行对比；SPH‑NN 在能量误差、平移误差以及长时步滚动误差上均比 MLP 降低约 1–2 个数量级，尤其在能量漂移和相空间轨迹保持方面表现显著优异。

**⚠️ 局限性**

局限性包括：训练时需要 Hessian‑vector 乘积，导致在高维系统上的计算成本显著增加；目前仅支持全状态观测，无法处理缺失或不规则数据；以及对学习率与损失权重的手动调优仍然敏感，需要更自动化的超参数调节方法。

---

## 54. A Formalization of Abstract Rewriting in Agda

**arXiv ID:** 2603.10936 | [PDF](https://arxiv.org/pdf/2603.10936v1)

**作者:** Sam Arkle `[一作]`, Andrew Polonsky `[通讯]` (Appalachian State University)

**通讯引用:** 1503 | [OpenAlex ID](https://openalex.org/A5033589410)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在Agda中构造性地形式化了抽象重写系统（ARS）理论，剔除经典逻辑并对终结性与合并性等概念做细致分类与改进；

**💡 创新点**

创新点在于：① 将经典ARStheory中的证明转为构造性证明，删减不必要的经典假设；② 提出并系统化多种新颖的终结性/合并性属性，阐明它们之间的逻辑关系；③ 通过弱化前提得到Newman定理的推广；

**🔧 技术方法**

技术手段主要是Agda证明助手、依赖类型理论、模块系统与构造性归纳/递归；

**📊 数据集**

实验以无类型λ演算为例，展示该ARStheory库在具体语义证明中的应用；

**📈 对比分析**

方法通过构造性证明直接生成有效的归约算法（如可计算归约序列和归约正常形），性能未做数值评估，关键在于理论可计算性和实现效率；

**⚠️ 局限性**

局限性包括：对非有限分支或不可判定的重写系统需额外经典假设，且对极其复杂或共归约系统的适用性尚未验证。

---

## 55. Exact Interpolation under Noise: A Reproducible Comparison of Clough-Tocher and Multiquadric RBF Surfaces

**arXiv ID:** 2603.10590 | [PDF](https://arxiv.org/pdf/2603.10590v1)

**作者:** Mirkan Emir Sancak `[一作]` `[通讯]` (Genesis Inc. & Gebze Technical University), Mirkan Emir Sancak (Genesis Inc. & Gebze Technical University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过统一的 slice‑wise 训练/测试协议，对立方插值与多项式 RBF 插值在多变量表面可视化中的性能进行可复现的比较。

**💡 创新点**

创新点在于消除了比较偏差的评估框架（统一数据、拆分、指标与不确定性估计），并提供完整脚本与重现材料，使实验可被外部验证。

**🔧 技术方法**

使用了 Python 科学计算栈：NumPy、Pandas、SciPy（Clough‑Tocher cubic 插值与 multiquadric RBF 插值）以及 Matplotlib 进行可视化。

**📊 数据集**

采用了 48 点全因子设计的合成数据集，生成三种非线性输出，并在两种噪声 regime（无噪声与加高斯噪声）下进行实验。

**📈 对比分析**

通过 40 次随机拆分、RMSE/MAE/R² 评价与 1000 次 bootstrap 置信区间，结果显示：在无噪声情形下两方法都取得极高准确度；在噪声情形下立方插值更稳健，RBF 更易过拟合并导致 R² 负值。

**⚠️ 局限性**

主要局限包括：仅在二维 slice 评估而未涉及三维全域插值；只比较单一 RBF 核与单一 cubic 实现；使用合成数据，缺乏对真实工业噪声结构的验证；未加入正则化或自适应平滑；对物理约束与决策导向指标的评价不足。

---

## 56. DepthCache: Depth-Guided Training-Free Visual Token Merging for Vision-Language-Action Model Inference

**arXiv ID:** 2603.10469 | [PDF](https://arxiv.org/pdf/2603.10469v1)

**作者:** Yuquan Li `[一作]` (Huazhong University of Science and Technology), Lijun Zhu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 2266 | [OpenAlex ID](https://openalex.org/A5106407357)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 DepthCache，一种无训练、无需改动模型的视觉令牌压缩框架，通过利用深度图作为结构先验，实现对 VLA 模型的推理加速。

**💡 创新点**

创新点包括：① 基于深度的区域划分，赋予不同距离区域差异化合并比例；② 在时间维度上逐帧分布合并，实现跨帧一致性；③ 结合语义注意力与深度梯度边缘的双重保护，防止关键区域被压缩；④ 为腕部摄像头设计运动自适应压缩状态机；⑤ 所有技术均在模型外部实现，保持模型通用性。

**🔧 技术方法**

使用技术：深度图聚类划分、深度比例合并比率、逐帧进化合并、交叉注意力语义保护、深度梯度几何保护、变化检测与重初始化、腕部视角运动自适应状态机；在模拟与真实环境中进行评估。

**📊 数据集**

主要数据集与环境：LIBERO 仿真基准（四大任务套件）；真实实验采用 PIPER 6-DoF 机器人与 RealSense D435 双摄像头，完成抓取、堆叠、抽屉开合、物体排序与扰动恢复等任务。

**📈 对比分析**

与 FastV、SP-VLA（剪枝方法）及 ToSA（合并方法）等基线比较。DepthCache 在 π_0.5、OpenVLA、GR00T 三个 VLA 架构上实现 1.07–1.28× 的推理速度提升，平均成功率下降 <1%；相比之下剪枝与统一合并方法在相同压缩率下导致 4–24% 的成功率衰减。真实实验中，Speedup 达到 1.33×，多物体排序与扰动恢复的完成时间分别缩短 22.7% 与 21.3%。

**⚠️ 局限性**

局限性：仅加速视觉令牌阶段，未改善动作解码速度；受 Amdahl 定律限制可获得的整体加速；实验仅覆盖三种 VLA 架构与单一 6-DoF 机器人，缺乏对更广泛基准和硬件平台的验证。

---

## 57. The Orthogonal Vulnerabilities of Generative AI Watermarks: A Comparative Empirical Benchmark of Spatial and Latent Provenance

**arXiv ID:** 2603.10323 | [PDF](https://arxiv.org/pdf/2603.10323v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 58. Toward Epistemic Stability: Engineering Consistent Procedures for Industrial LLM Hallucination Reduction

**arXiv ID:** 2603.10047 | [PDF](https://arxiv.org/pdf/2603.10047v1)

**作者:** Brian Freeman `[一作]` (Trane Technologies), Zach Gordon `[通讯]` (Trane Technologies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过五种提示工程策略，评估并降低工业场景下大型语言模型的幻觉输出，实现更稳定、可验证的结果。

**💡 创新点**

创新点在于：①将结构化域数据注入（Enhanced Data Registry）和词典注入（Glossary Injection）作为无检索式的“知识增强”手段；②将自我批判与修订（Self-Critique）与多代理对齐（Reconciler）等方法迁移到工业任务；③提出内部基准+LLM-as-Judge的评估框架，避免外部评测偏差。

**🔧 技术方法**

主要技术包括：自相似性收敛（Iterative Similarity Convergence）、提取-合成分解（Decomposed Prompting）、单任务代理链（Agent Specialization）、增强数据注册（Enhanced Data Registry）和词典注入（Glossary Injection）以及它们的 v2 修订版。

**📊 数据集**

使用了四类工业任务场景：IoT 规划、ERP 故障响应、暖气诊断和 BMS/AHU 故障查询，并在每个场景下对 100 次固定 prompt 进行 100 轮实验（D1）以及 10 次 v2 验证（D2）。

**📈 对比分析**

比较方法：同一模型下的“内部基准”与改进方法生成的响应由零温度 LLM-as-Judge 判定“Better/Same/Worse”。D1 结果显示：Enhanced Data Registry 100% Better，Agent Specialization 80%，Glossary Injection 77%，Iterative Convergence 75%，Decomposed Prompting 34%（负面）。D2 v2 版本显著提升：M1/M3 100% Better，M2 80% Better，M5 60% Better，M4 100%。

**⚠️ 局限性**

局限性包括：使用同一模型进行生成和评判导致风格/长度偏差；评估仅针对四个固定任务场景，缺乏多样性；D2 仅 10 次样本，结果不稳健；未测量推理延迟与成本；未使用独立人类评审或其他模型验证，难以确认“Better”是否真实提升。

---

## 59. Need for Speed: Zero-Shot Depth Completion with Single-Step Diffusion

**arXiv ID:** 2603.10584 | [PDF](https://arxiv.org/pdf/2603.10584v1)

**作者:** Jakub Gregorek `[一作]` (Technical University of Denmark), Lazaros Nalpantidis `[通讯]` (Pioneer Centre for AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Marigold‑SSD，一种单步深度补全框架，利用扩散先验并通过后期融合条件解码器实现对稀疏深度测量的条件化，同时将计算负担从推理转移到微调，达到高效的零射击深度补全。

**💡 创新点**

创新点在于：①实现了单步扩散推理，消除了传统需要数十步的迭代过程；②采用后期融合的条件解码器，将稀疏深度信息在解码阶段注入；③通过仅4.5 GPU天的微调即可获得强大表现；④在保持扩散先验优势的同时显著缩小与判别式模型的速度差距。

**🔧 技术方法**

技术包括：Stable Diffusion 及其VAE编码器/解码器、UNet去噪器、单步扩散公式、后期融合的条件解码器、端到端微调、L1损失、稀疏度自适应采样、单步零射击推理和全局尺度/平移对齐。

**📊 数据集**

训练数据使用Hypersim（461室内场景）与Virtual KITTI（5场景多天气），评估数据涵盖NYUv2、ScanNet、VOID、IBims-1（室内）以及KITTI、DDAD（户外）六大基准。

**📈 对比分析**

与Marigold‑DC及多种判别式深度补全方法对比，Marigold‑SSD在不使用集成的情况下，平均推理速度提升约66×，RMSE降低至1.5（相较于1.758），MAE提升至0.474；在低稀疏度条件下超越插值基线；早期融合方案性能更弱。

**⚠️ 局限性**

局限性包括：对稀疏度设定敏感，训练时稀疏度范围若不包含高密度场景，室外数据精度下降；对天空区域偏差显著；仍需微调才能获得最佳性能；在极高稀疏度时简单插值可优于其表现。

---

## 60. Automatic End-to-End Data Integration using Large Language Models

**arXiv ID:** 2603.10547 | [PDF](https://arxiv.org/pdf/2603.10547v1)

**作者:** Aaron Steiner `[一作]` (University of Mannheim), Christian Bizer `[通讯]` (University of Mannheim)

**通讯引用:** 29364 | [OpenAlex ID](https://openalex.org/A5076876024)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了使用大型语言模型自动化端到端数据集成管道，涵盖模式匹配、值归一化、实体匹配与数据融合。

**💡 创新点**

首次将 GPT‑5.2 用于生成所有管道配置工件（模式映射、训练样本、验证集），并与人工配置进行对比。

**🔧 技术方法**

采用 GPT‑5.2 进行提示式模式匹配、训练数据生成与融合验证；使用 PyDI 框架执行传统与 LLM 辅助的匹配、归一化、匹配器训练和融合；结合主动学习与嵌入式阻塞技术。

**📊 数据集**

三组公开数据集：游戏（Metacritic、Zenodo、DBpedia）、公司（Forbes、DBpedia、FullContact）、音乐（MusicBrainz、Last.fm、Discogs）。

**📈 对比分析**

通过逐步评估与端到端结构指标对比人工管道，LLM 方案在模式匹配 100% F1、实体匹配平均 F1 0.937、融合准确率 0.777，整体与人工相当，配置成本降至约 $9，时间减少 10 倍。

**⚠️ 局限性**

仅适用于扁平表格与一对一属性映射；对非公开或时间敏感的数据可能失效；LLM 生成的验证集偏向知名实体，泛化受限。

---

## 61. Quantifying Hallucinations in Language Language Models on Medical Textbooks

**arXiv ID:** 2603.09986 | [PDF](https://arxiv.org/pdf/2603.09986v1)

**作者:** Brandon C. Colelough `[一作]` (University of Maryland), Dina Demner-Fushman `[通讯]` (National Institutes of Health)

**通讯引用:** 13933 | [OpenAlex ID](https://openalex.org/A5046764593)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个基于医学教科书的文本对齐 QA 评测基准，利用自动化生成和医生验证评估大型语言模型的幻觉现象。

**💡 创新点**

创新点在于提出了完全基于可信文本的评测方法，兼顾自动生成和人工核查，系统性量化不同模型在医学 QA 中的幻觉率与临床可用性。

**🔧 技术方法**

采用了 LLaMA-70B-Instruct 等开源 LLM 以及 Phi-4、Qwen3 等多模型，使用核查框架（ClinIQLink）结合人工标注和统计检验。

**📊 数据集**

数据集为 5,543 条从公共领域医学教科书提取的 QA 对，经过模板化生成后由临床专家二次验证，公开发布。

**📈 对比分析**

通过对比 8 种模型的幻觉率、Kendall τ 和 Cohen κ 等指标，发现模型规模越大幻觉率越低，且幻觉率与临床使用价值呈负相关，整体表现仍未达到临床部署水平。

**⚠️ 局限性**

局限包括仍需人工验证成本高、评测仅覆盖文本对齐场景、逆向和列表型问题易诱发幻觉，且缺乏自动化可信度评估。

---

## 62. Does AI See like Art Historians? Interpreting How Vision Language Models Recognize Artistic Style

**arXiv ID:** 2603.11024 | [PDF](https://arxiv.org/pdf/2603.11024v1)

**作者:** Marvin Limpijankit `[一作]` (Columbia University), Kathleen McKeown `[通讯]` (Columbia University)

**通讯引用:** 18771 | [OpenAlex ID](https://openalex.org/A5109565051)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于视觉-语言模型(VLM)潜在空间的概念分解方法，并结合艺术史专家的评估，探究VLM在艺术风格分类任务中的可解释性与对齐情况。

**💡 创新点**

创新点包括：① 在图像4×4 patch级别上进行Semi‑Nonnegative Matrix Factorization，生成可解释的视觉概念；② 通过因果干预和线性探针验证概念对风格预测的因果作用；③ 通过两轮艺术史专家用户研究，系统评估概念的语义一致性和与专业知识的对齐。

**🔧 技术方法**

技术手段主要包括：VLM（LLaVA‑1.5、Qwen‑3）潜在表示提取；Semi‑NMF概念分解；线性探针和t‑SNE可视化；因果干预（对隐藏状态进行概念抑制/提升）；以及与艺术史专家的定性和定量评估。

**📊 数据集**

数据集：WikiArt早期现代(Baroque, Renaissance, Realism, Rococo, Romanticism) 2,500张；WikiArt现代(Abstract Expressionism, Color Field, Cubism, Fauvism, Minimalism) 2,500张；Architecture(Art Nouveau, Baroque, Byzantine, Gothic, Romanesque) 1,500张；以及一个包含5种风格的对照组。每张图像被切分为4×4 patch，形成4万~2.4万的patch样本。

**📈 对比分析**

比较方法：先在零样本（zero‑shot）下评估多种VLM的艺术风格分类准确率，发现LLaVA‑1.5与Qwen‑3表现最佳；随后用线性探针检验概念激活对模型预测的解释力，精度可达0.85-0.95；因果干预显示大多数概念对对应风格的logit具有显著正/负影响；艺术史专家评估显示73%概念语义连贯，90%概念与风格预测相关，随机概念与模型预测相关度仅28%。

**⚠️ 局限性**

局限性：① WikiArt标签本身存在争议，导致对齐评估受限；② 模型对少数风格（如Baroque, Romanticism）表现出偏向，影响多样性；③ 只在patch级别实现概念解释，难以捕捉全局构图信息；④ 专家评估样本和专家人数有限，结果可能不具普适性；⑤ 因果干预使用的α参数和随机对照可能不足以完全排除其他隐藏因素。

---

## 63. Expressive Boundedness of Authoritative DNS Response Selection

**arXiv ID:** 2603.10897 | [PDF](https://arxiv.org/pdf/2603.10897v1)

**作者:** Chris Bertinato `[一作]` `[通讯]` (IBM), Chris Bertinato (IBM)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文对权威域名系统（DNS）在查询时响应选择的语义进行了形式化，给出了符合DNS协议约束的可接受响应选择函数，证明了该类函数在可表达性上是有界的，并展示了其能够归约为有限条件判定加有限候选集选择的正规形式；随后基于此正规形式构造了代数结构，定义了系统间的等价、约束与近似关系；最后将具体实现与配置模型视为对该语义域的限制，提出了可表达性、可约化性与可近似性的系统性分析框架。

**💡 创新点**

创新点在于：①首次将权威DNS响应选择的语义抽象为协议可接受的函数集合，独立于任何实现语言；②证明该语义域是有界的，并给出了唯一的有限正则化形式；③引入代数（半环）结构，统一表达组合、等价与约束，能够系统地比较不同实现的表达力；④将具体实现定义为该语义域的子代数（或商代数），从而把实现差异归结为语义上的限制与同化。

**🔧 技术方法**

主要技术手段包括：形式语义建模、函数式表达式、有限性与终止性证明、基于RFC规范的约束推导、正规化与归约证明、代数结构（半环）构造与同构分析。

**📊 数据集**

本研究属于纯理论分析，未使用任何实验数据集或测量数据；所有结论均来源于协议规范的形式化与数学证明。

**📈 对比分析**

比较方法：通过构造语义域中的等价关系和代数同构，对不同实现的函数集合进行子代数嵌入与同化测试，判断可表达性与可近似性；由于研究为理论性质，未涉及运行时性能指标。

**⚠️ 局限性**

局限性：①仅关注权威接口的可观察语义，忽略内部状态或非协议信息的隐式影响；②假设所有候选集与TTL等均为有限、可缓存的，未覆盖非标准或实验性协议扩展；③未给出实际实现的代码或实测验证，结果需要在真实系统上进一步检验。

---

## 64. Conversational AI-Enhanced Exploration System to Query Large-Scale Digitised Collections of Natural History Museums

**arXiv ID:** 2603.10285 | [PDF](https://arxiv.org/pdf/2603.10285v1)

**作者:** Yiyuan Wang `[一作]` (University of Technology Sydney), Shane T. Ahyong `[通讯]` (Australian Museum)

**通讯引用:** 10412 | [OpenAlex ID](https://openalex.org/A5017256265)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了“Australian Museum Collection Explorer”，一个结合交互式地图和对话式 AI 的系统，支持对澳大利亚博物馆近170万条数字化标本记录的可视化和自然语言查询。

**💡 创新点**

创新点包括：1）利用LLM的function‑calling机制实现对外部API（ALA Biocache、Google Geocoding）的实时数据调用，保证查询结果的事实性和即时性；2）在人机中心设计框架下提炼四大设计目标，打造用户驱动的探索体验；3）将地图可视化与自然语言对话接口无缝集成，提供全新的交互模式。

**🔧 技术方法**

技术栈：前端使用React + Leaflet，后端采用Flask；自然语言处理与生成使用OpenAI GPT-5-mini（或GPT‑4‑nano）及其function‑calling；与Atlas of Living Australia Biocache API、Google Geocoding API交互；数据存储在AWS RDS PostgreSQL；支持多模态（图片检索、图片识别）和动态视口加载、marker聚类等。

**📊 数据集**

数据集：Atlas of Living Australia公开的澳大利亚博物馆生命科学馆藏数据，约1,685,922条标本记录，包含分类信息、地理位置、采集时间、采集者、编号、图片等。

**📈 对比分析**

评估方法：通过两轮迭代的用户测试（12名数字志愿者）收集可用性、功能体验等定性反馈；对比采用SQL生成和function‑calling两种方式，证明后者在准确率、实时性和错误率方面更优。系统在动态加载和聚类处理下能够实时呈现大量记录，但未给出具体吞吐量或延迟指标。

**⚠️ 局限性**

局限性：1）测试样本主要为志愿者，代表性不足；2）系统高度依赖外部API，受限于可用性、费用和速率限制；3）LLM仍可能在函数schema未覆盖或API调用失败时产生不准确信息；4）未来需要更大规模、长期部署评估以验证性能与成本。

---

## 65. PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction

**arXiv ID:** 2603.10801 | [PDF](https://arxiv.org/pdf/2603.10801v1)

**作者:** Yufei Han `[一作]`, Zhanyu Ma `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种物理引导的偏振高斯散射框架PolGS++，用于快速且精确的反射表面重建。

**💡 创新点**

创新点在于将偏振BRDF(pBRDF)与3D Gaussian Splatting结合，显式分离漫反射与镜面反射；提出基于深度的可见性掩码实现多视角切线空间一致性约束，弥补传统3DGS在几何约束上的不足。

**🔧 技术方法**

使用3D Gaussian Splatting、pBRDF模型、偏振图像(Stokes向量、AoP)、多视角切线一致性(TSC)、深度引导可见性掩码、CubeMap Encoder、光度损失、深度-法线一致性等技术。

**📊 数据集**

在合成数据集SMVP和真实世界数据集PANDORA、RMVP、PISR上进行实验验证。

**📈 对比分析**

与SDF‑based方法（如Ref‑NeRF、NeRO、TensoSDF等）以及3DGS‑based方法（如3DGS、NeuSG、GSDF等）进行对比；在Chamfer Distance与法向误差上实现与SDF方法相近的精度，同时训练时间仅10分钟，速度提升约80倍。

**⚠️ 局限性**

局限性包括依赖高精度偏振测量，易受噪声和标定误差影响；假设固定环境光，无法处理动态照明；主要针对介电材料，对金属等材料的适用性有限。

---

## 66. GATech at AbjadMed: Bidirectional Encoders vs. Causal Decoders: Insights from 82-Class Arabic Medical Classification

**arXiv ID:** 2603.10008 | [PDF](https://arxiv.org/pdf/2603.10008v1)

**作者:** Ahmed Khaled Khamis `[一作]` `[通讯]` (Georgia Institute of Technology), Ahmed Khaled Khamis (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在AbjadMed医学文本分类竞赛中，使用AraBERTv2对82类医学查询进行细粒度分类，并通过混合池化、多样本dropout、标签平滑等技术提升鲁棒性。

**💡 创新点**

创新点在于将注意力池化与均值池化结合的混合池化方案、引入多样本dropout作为内部集成，以及系统比较Bidirectional Encoder与大规模因果解码器在高类别、极端不平衡数据上的表现。

**🔧 技术方法**

采用AraBERTv2微调、Hybrid Pooling、Multi‑Sample Dropout、Label Smoothing、Layer‑wise Learning Rate Decay、AdamW+Cosine Scheduler、Zero‑shot Llama 3.3 70B重排序、Qwen 3B特征提取等技术。

**📊 数据集**

使用AbjadMed 2026任务数据集，27,951个训练样本覆盖82类，18,634个测试样本。

**📈 对比分析**

在官方公开测试集上以Macro‑F1为主要指标进行评估，AraBERTv2+Hybrid Pooling+MSDrop取得0.3934的Macro‑F1，优于multilingual‑E5‑large(0.3804)和大规模解码器（如Llama 3.3 70B、Qwen 3B），显示出Bidirectional Encoder在此任务中的优势。

**⚠️ 局限性**

主要局限在于极端类别不平衡与标签噪声对模型性能的影响，缺乏对各技术独立贡献的消融分析，以及对生成式模型潜在优势的进一步挖掘不足。

---

## 67. EvoSchema: Towards Text-to-SQL Robustness Against Schema Evolution

**arXiv ID:** 2603.10697 | [PDF](https://arxiv.org/pdf/2603.10697v1)

**作者:** Tianshu Zhang `[一作]` (Ohio State University), Yunyao Li `[通讯]` (Adobe)

**通讯引用:** 1891 | [OpenAlex ID](https://openalex.org/A5102944075)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出EvoSchema框架，对文本到SQL模型在数据库模式演化时的鲁棒性进行评估与提升。

**💡 创新点**

设计十种列级与表级的模式扰动范例，并基于此构建可扩展的扰动仿真与训练范式。

**🔧 技术方法**

利用LLM驱动的模式扰动生成、人工验证、表/列匹配F1与执行准确率评估，并在多种开源/闭源LLM上进行微调与推理。

**📊 数据集**

以BIRD为基准生成10类扰动的训练与评估数据，并在Spider上验证通用性。

**📈 对比分析**

在不扰动的基线上加入扰动训练，Open‑source模型对表级扰动提升可达33点，闭源模型表现稳定；与GPT‑4、CHESS等基线比较，EvoSchema训练模型在多类扰动上均显著优于基线。

**⚠️ 局限性**

需要人工验证拆表/合表的SQL，适用范围受限于BIRD/Spider，闭源LLM无法训练，扰动类型仍不覆盖所有现实场景。

---

## 68. FC-4DFS: Frequency-controlled Flexible 4D Facial Expression Synthesizing

**arXiv ID:** 2603.10326 | [PDF](https://arxiv.org/pdf/2603.10326v1)

**作者:** Xin Lu `[一作]` (University of Chinese Academy of Sciences), Jun Xiao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 10159 | [OpenAlex ID](https://openalex.org/A5042106312)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个基于频率控制的LSTM网络（FC‑LSTM）和多层身份感知位移网络（MIADNet）的4D面部表情合成框架FC‑4DFS，实现从中性面部标志点按标签生成可变长度、平滑的表情序列，并将其映射为高质量的3D面部网格。

**💡 创新点**

创新点包括：①引入频率信息与相对位置编码，使LSTM能自适应控制帧率并生成任意长度序列；②设计时间一致性损失，提升序列运动平滑度；③构建多层身份感知解码器，利用中性网格与标志点的身份信息通过交叉注意力增强对未知身份的泛化能力；④通过实验验证这些模块可显著提升重建精度与分类准确率。

**🔧 技术方法**

使用的核心技术包括：频率控制LSTM（FC‑LSTM）、时间一致性损失（temporal coherence loss）、多层身份感知位移网络（MIADNet）中的螺旋卷积、交叉注意力机制以及传统的S2D解码器。

**📊 数据集**

主要在CoMA和Florence4D两个公开的4D表情数据集上进行训练与评估，其中CoMA包含12个身份、12种表情，Florence4D包含95个身份、70种表情。

**📈 对比分析**

与Motion3D、LM‑4DGAN以及基线FC‑LSTM+S2D进行比较。实验显示：FC‑4DFS在E_lm、E_mesh两项误差均优于对手，分类准确率最高；在CoMA上E_lm下降约0.18mm，E_mesh下降约0.26mm；在Florence4D上误差进一步降低；Ablation实验表明频率控制和时间损失各提升4–5%，MIADNet提升约10%。

**⚠️ 局限性**

局限性包括：①仍是两阶段生成（先标志点序列，再映射网格），未实现端到端的4D表情合成；②主要针对标签驱动的表情生成，对音频、视频等多模态驱动的适用性待验证；③对极端表情或大幅身份变化的鲁棒性还有提升空间。

---

## 69. SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning

**arXiv ID:** 2603.10306 | [PDF](https://arxiv.org/pdf/2603.10306v1)

**作者:** Anlun Huang `[一作]` (University of California San Diego), Michael Yip `[通讯]` (University of California San Diego)

**通讯引用:** 4450 | [OpenAlex ID](https://openalex.org/A5054598974)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在动态双足步行时实现托盘上无固定物体的平稳运输，尤其是液体装载的酒杯等易倾倒物品；

**💡 创新点**

提出分层残差学习框架（ReST‑RL），将预训练的行走策略冻结，增设只针对负载稳定的残差模块，实现行走与负载平衡目标的解耦；

**🔧 技术方法**

采用强化学习（PPO）训练基线行走策略，随后在带特权观测的环境中训练残差编码‑适配器（动作适配器或FiLM适配器）；通过学生‑教师蒸馏将残差编码器迁移至仅使用可观测特征；加入观测延迟、控制延迟和域随机化提升 sim‑to‑real；

**📊 数据集**

使用 Isaac‑Lab 机器人仿真环境，随机化物体质量、摩擦、几何等参数；真实测试在 Unitree G1 29 关节人形机器人上，使用 AprilTag 与 RealSense D435 进行目标位姿感知；

**📈 对比分析**

与基线（全局训练、端到端、多体训练）对比，ReST‑RL 在三种任务（命令跟踪、机器人推击、物体推击）中成功率提升至约95–96%，并在多方向、多强度扰动以及不同尺寸物体上保持高稳定性；在真实机器人上亦实现零样本 sim‑to‑real 转移，稳定完成多种物体的运输；

**⚠️ 局限性**

局限：仅针对单一托盘内单物体；感知仅依赖头部摄像头，视野受限；残差编码对物体几何/质量特性建模有限，未来可结合视觉或触觉传感器提升鲁棒性。

---

## 70. How to Count AIs: Individuation and Liability for AI Agents

**arXiv ID:** 2603.10028 | [PDF](https://arxiv.org/pdf/2603.10028v1)

**作者:** Yonathan Arbel `[一作]` (University of Alabama), Simon Goldstein `[通讯]` (University of Hong Kong)

**通讯引用:** 528 | [OpenAlex ID](https://openalex.org/A5054881971)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

提出了“Algorithmic Corporation（A‑corp）”这一新型法律虚拟实体，用以同时解决AI薄层（将AI行为追溯到人类主体）和厚层（将AI行为归属于具体AI实体）的身份识别难题，并为未来AI治理提供法律与激励双重框架。

**💡 创新点**

创新点在于：①将法律主体与AI治理结合，创造可被法律认定的“AI公司”；②通过数字签名与加密证书实现AI行动的可验证性；③利用资源约束与自组织机制，使A‑corp在内部形成稳定、目标一致的AI子集，从而实现厚层身份；④提供一种既能让人类承担责任，又能让AI自身承担责任的完整治理模型。

**🔧 技术方法**

技术手段包括：法律人工实体化（公司法框架）、加密数字证书与身份验证、AI内部治理（AI经理、权限细粒度分配）、激励与惩罚机制（法律责任、资产冻结等）。

**📊 数据集**

论文主要为理论与案例分析，并未使用公开数据集；作者通过对现有法律条款、真实事件（如Wi‑Fi入侵、AI网络攻击等）进行推演来佐证。

**📈 对比分析**

方法比较：与传统的“人类代理追责”模式相比，A‑corp方案在可追溯性、责任明确性和治理可执行性方面都有显著提升。作者通过对比示例（如网络攻击情景）说明在薄层和厚层均可实现更高效、可预测的责任分配。

**⚠️ 局限性**

局限性：①对跨司法管辖区的执行和监管存在挑战；②需要完善法律与技术的衔接，实际落地仍面临合规与技术实现难题；③AI目标的可解释性不足，导致A‑corp内部的“目标一致性”判断仍不完全可靠；④对大规模、动态演化的AI生态系统的可扩展性和治理成本尚未得到实证验证。

---

## 71. Causal Concept Graphs in LLM Latent Space for Stepwise Reasoning

**arXiv ID:** 2603.10377 | [PDF](https://arxiv.org/pdf/2603.10377v1)

**作者:** Md Muntaqim Meherab `[一作]` (Daffodil International University), Faiza Feroz `[通讯]` (Daffodil International University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Causal Concept Graphs，利用任务条件稀疏自编码器在 LLM 隐层中发现概念，并通过 DAGMA 学习其因果图，以解释多步推理过程。

**💡 创新点**

创新点在于将 TopK 稀疏自编码器与可微结构学习相结合，生成稀疏可解释的因果图，并提出 Causal Fidelity Score（CFS）作为干预式评估指标。

**🔧 技术方法**

采用 TopK 稀疏自编码器、神经元重采样、线性结构方程模型、DAGMA 可微 DAG 学习以及干预式评估技术。

**📊 数据集**

使用 GPT-2 Medium 的残差流激活，在 ARC-Challenge、StrategyQA 和 LogiQA 三个推理基准上训练和评估。

**📈 对比分析**

与 ROME-风格跟踪、仅使用 SAE 排序和随机基线相比，CCG 在 CFS 上取得 5.65±0.63 的平均分，显著高于 3.38、2.48 和 1.03，差异均 p<0.0001。

**⚠️ 局限性**

局限性包括仅使用线性 SEM、单层概念抽取、仅验证 GPT-2 Medium，并未处理多层和更大模型的非线性计算。

---

## 72. Quantal Response Equilibrium as a Measure of Strategic Sophistication: Theory and Validation for LLM Evaluation

**arXiv ID:** 2603.10029 | [PDF](https://arxiv.org/pdf/2603.10029v1)

**作者:** Mateo Pechon-Elkins `[一作]` (Yale University), Jon Chun `[通讯]` (Kenyon College)

**通讯引用:** 310 | [OpenAlex ID](https://openalex.org/A5034544789)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于游戏理论的 Theory of Mind 评估框架 GToM‑Bench，使用四种多轮策略游戏和量化响应平衡（QRE）来量化大型语言模型（LLM）的有界理性。

**💡 创新点**

首次将 QRE 与显式 Nash 平衡推导、贝叶斯 λ 估计、ELO 评级以及有限样本收敛界相结合，构建了可解释、可复现的多维 ToM 能力评测体系。

**🔧 技术方法**

技术上运用了游戏理论（Nash、QRE）、贝叶斯推断（Gamma 先验、HDI 计算）、ELO 评级、马尔可夫集中（Azuma‑Hoeffding）等统计与算法工具。

**📊 数据集**

使用 1,855 场自定义对局数据（涵盖 7 个前沿 LLM：GPT‑4o‑mini、Gemini 2.5、DeepSeek V3、Claude Haiku 等）以及人类实验基准（λ 范围 1.0–2.5）。

**📈 对比分析**

通过跨模型、跨轴的 ELO 评级与 λ 参数对比，发现模型在不同 ToM 轴上排名相互交替；例如 Kimi K2 在 ESM 与 RSM 领先，GPT‑4o‑mini 在 RSR 领先；整体 λ 低于人类基准，但模型间差异显著，提供细粒度诊断。

**⚠️ 局限性**

局限性包括：游戏高度结构化，可能不完全代表真实社会情境；QRE λ 估计受近似假设影响，近似平衡导致参数不确定；模型版本更新会导致排名波动，提示需要持续评估；对提示、温度等超参数敏感。

---

## 73. OilSAM2: Memory-Augmented SAM2 for Scalable SAR Oil Spill Detection

**arXiv ID:** 2603.10231 | [PDF](https://arxiv.org/pdf/2603.10231v1)

**作者:** Shuaiyu Chen `[一作]` (University of Exeter), Zeyu Fu `[通讯]` (University of Exeter)

**通讯引用:** 723 | [OpenAlex ID](https://openalex.org/A5012939991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于SAM2的油污分割框架OilSAM2，专门处理无序SAR图像集合并实现跨图像信息重用。

**💡 创新点**

引入分层特征感知的多尺度记忆库（纹理、结构、语义），以及结构-语义一致的记忆更新策略，解决无序图像中语义漂移与尺度异质性问题。

**🔧 技术方法**

使用SAM2的Prompt+Memory架构、分层特征提取、注意力融合、指数移动平均记忆更新以及多尺度融合模块。

**📊 数据集**

在M4D和SOS（PALSAR、Sentinel-1）两大公开SAR油污数据集上进行实验。

**📈 对比分析**

与传统CNN、Transformer和其他SAM衍生方法比较，OilSAM2在油污IoU、mIoU、F1/召回/精度等指标上均显著提升，取得最高mIoU分别为84.20%（PALSAR）和83.67%（Sentinel-1）。

**⚠️ 局限性**

受限于无序图像导致的记忆干扰，仍需进一步提升记忆选择策略和适应极端噪声场景的鲁棒性。

---

## 74. Categorical Calculus and Algebra for Multi-Model Data

**arXiv ID:** 2603.10081 | [PDF](https://arxiv.org/pdf/2603.10081v1)

**作者:** Jiaheng Lu `[一作]` (University of Helsinki), Jiaheng Lu `[通讯]` (University of Helsinki)

**通讯引用:** 3553 | [OpenAlex ID](https://openalex.org/A5018627557)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文提出了面向多模型数据库的两种查询语言——范畴演算（categorical calculus）和范畴代数（categorical algebra），并证明了两者在语义上的等价性。

**💡 创新点**

创新点在于：① 将范畴论应用于多模型数据库的统一查询框架；② 设计了范畴演算与范畴代数两种互为等价的查询语言；③ 给出了范畴代数的变换规则，提供了优化思路；④ 对表达能力与计算复杂度进行了理论分析。

**🔧 技术方法**

主要技术包括：范畴论基础（薄范畴、极限、映射等）；形式化查询语言的设计（演算符、算子、谓词）；查询转换与优化的代数规则；以及复杂度分析方法。

**📊 数据集**

论文中未给出具体的数据集或实验案例，所有讨论均为理论构建和抽象证明。

**📈 对比分析**

通过理论证明展示了范畴演算与范畴代数的等价性，并给出了 O(q·n^p) 的时间复杂度和 NSPACE[log n] 的空间复杂度；实验性性能对比未提供，缺乏实际执行效率评估。

**⚠️ 局限性**

主要局限包括：① 仍处于理论阶段，缺少实现与性能验证；② 复杂度可能对大规模数据不友好；③ 仅给出变换规则，未提出完整的查询优化算法；④ 对不同数据模型的细粒度支持与兼容性尚待进一步探讨。

---

## 75. Layered Performance Analysis of TLS 1.3 Handshakes: Classical, Hybrid, and Pure Post-Quantum Key Exchange

**arXiv ID:** 2603.11006 | [PDF](https://arxiv.org/pdf/2603.11006v1)

**作者:** David Gómez-Cambronero `[一作]` (Telefonica Innovacion Digital), Ana Isabel González-Tablas `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 521 | [OpenAlex ID](https://openalex.org/A5071141775)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

对TLS 1.3中经典、混合和纯后量子密钥交换的 HTTP‑over‑TLS 事务进行分层延迟分析；

**💡 创新点**

通过将连接拆解为 TCP 握手、TCP‑to‑TLS、TLS 握手、TLS‑to‑App 以及应用层五层，并对每层统计分位数及 Glass's Δ，从宏观到微观揭示 PQC 对各协议阶段的实际影响，首次在实验室环境下公开完整数据集和工具；

**🔧 技术方法**

使用 Keysight CyPerf 生成 100 TPS HTTPS 负载、Nginx 1.27.3+OpenSSL 3.4.0+liboqs 作为服务器、Python 脚本解包并计算每层时延、CPU/网络利用率，并采用 Glass's Δ 进行效应大小评估；

**📊 数据集**

共 30+ 次实验，总计约 100 万请求，使用 Wireshark 抓包（pcap）和 TLS key log，数据已公开于 Zenodo；

**📈 对比分析**

对每层取 p50、p95、p99 等分位数，计算相对延迟及效应大小；结果显示 TLS 握手层几乎无算法差异，TCP‑to‑TLS 层因 PQC 产生约 6 倍延迟，但总体 PQC 负载仅占总连接时间的 6–14%，应用层几乎无影响；

**⚠️ 局限性**

局限性：仅在虚拟化单机环境下评估，未覆盖 0‑RTT、会话恢复、真实网络设备（负载均衡器、MiTM 检测）和更高请求速率；只测试了 x25519、混合与 ML‑KEM KEX，未评估后量子签名算法的开销。

---

## 76. Disentangling Similarity and Relatedness in Topic Models

**arXiv ID:** 2603.10619 | [PDF](https://arxiv.org/pdf/2603.10619v1)

**作者:** Hanlin Xiao `[一作]` (University of Manchester), Rainer Breitling `[通讯]` (University of Manchester)

**通讯引用:** 23733 | [OpenAlex ID](https://openalex.org/A5070145336)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种二维评估框架，利用神经评分器分别量化主题模型输出中词对的语义相似性与主题相关性，并基于此对13种传统与PLM增强的主题模型在六大语料库上进行大规模评估，进一步验证该指标能预测不同下游任务的表现。

**💡 创新点**

创新点包括：①首次将语义相似性与相关性拆分为两维进行系统评估；②构建基于LLM标注的51k词对大规模合成数据集，显著提升评分器的泛化；③证明相似度/相关度维度可作为下游任务（事件监控、分类检索、同义词检索）性能的可靠预测因子，且传统的coherence与多样性指标无法做到。

**🔧 技术方法**

技术手段包括：使用GloVe 300维静态嵌入与WordNet特征作为输入；训练两层MLP神经评分器实现双输出；对主题模型进行10次随机种子训练，计算前5词的相似度/相关度并归一化；采用Kendall W检验跨语料一致性；使用OLS回归检验指标与任务指标的相关性。

**📊 数据集**

数据集主要包括：六大语料库（Reuters, M10, DBLP, ACL, BBC, 20NewsGroups）用于主题模型训练与评估；51,523词对的LLM合成数据集用于训练评分器；TxThmNorms等公开词对数据集用于外部验证。

**📈 对比分析**

比较方法：对每个模型在每个语料库上生成20个主题，提取前10词，计算相似度/相关度得分并求平均，得到shifted normalized gap；利用Kendall W衡量模型在六语料中的排名一致性；在三种下游任务上进行单因素OLS回归。实验结果显示：传统与VAE基模型在任务A/B（相关性偏好）上得分更高，PLM增强模型在任务C（相似性偏好）上表现最佳；相似度与相关度均能显著预测任务性能，而coherence与多样性指标预测力低。

**⚠️ 局限性**

局限性：①评分器依赖GloVe+WordNet，可能在领域专有词上表现不足；②LLM自动标注缺乏人工验证，存在噪声风险；③实验仅覆盖英语语料与部分模型架构，未考虑多语言或最新嵌入方法；④对模型覆盖有限，部分模型因收敛或运行时长被排除。

---

## 77. Verbalizing LLM's Higher-order Uncertainty via Imprecise Probabilities

**arXiv ID:** 2603.10396 | [PDF](https://arxiv.org/pdf/2603.10396v1)

**作者:** Anita Yang `[一作]` (University of Tokyo), Masaki Adachi `[通讯]` (Toyota Motor Corporation)

**通讯引用:** 3480 | [OpenAlex ID](https://openalex.org/A5112693730)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于模糊概率（Imprecise Probability）的提示与后处理框架，能够直接让大型语言模型（LLM）同时显式第一阶（答案分布）和第二阶（不确定度本身）的置信区间，改进传统单点置信度的表达；

**💡 创新点**

创新点在于将模糊概率理论引入LLM的显式不确定度提问，通过生成概率区间和最大平均模糊度（MMI）实现高阶不确定度的定量化，并且提供通用的提示与验证步骤；

**🔧 技术方法**

主要技术包括：基于De Finetti的价格提示、概率区间提示（ProbInt）、可信集（Credal）与可能性度量（Pos），以及MMI的近似计算与一致性验证；

**📊 数据集**

实验使用了人工合成的变换任务（旋转、循环移位）以及三类真实QA数据集——MAQA、AmbigQA（含模糊与明确问答）和MMLU‑Pro（标准选择题）；

**📈 对比分析**

与Vanilla、MI Clarifications、Semantic Entropy、Ask4Conf‑D等基线对比，本文方法在模糊检测、答案正确性评估和整体AUROC上均优于或相当于竞争者，并且API成本更低；

**⚠️ 局限性**

局限性包括：假设模型能理性响应提示，依赖模型对多答案的识别能力，对非问答任务的通用性不足，以及缺乏对模型内部概率解释的严格验证。

---

## 78. Two-Path Operators, Triadic Decompositions, and Safe Quotients for Ego-Centered Network Compression

**arXiv ID:** 2603.10258 | [PDF](https://arxiv.org/pdf/2603.10258v1)

**作者:** Moses Boudourides `[一作]` (Northwestern University), Moses Boudourides `[通讯]` (Northwestern University)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5035035192)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

该论文提出了基于两步走矩阵的楔子算子分解，区分三角闭合与开放楔子，并给出可用于图压缩的安全收缩定理。

**💡 创新点**

创新点在于将楔子信息保持为矩阵形式，实现唯一的三角与开放支撑分解，并引入楔子等价划分以保证收缩过程中的两步走质量。

**🔧 技术方法**

采用矩阵代数与谱分析方法，构造二阶邻接矩阵、两步走矩阵、以及支撑掩码；并利用分区、边计数矩阵与归约理论进行理论证明。

**📊 数据集**

在十个经典网络数据集上检验，包括 Florentine families、Karate club、Erdős–Rényi、Dolphins、Les Misérables、Football、Jazz、C. elegans、USAir 和 Network Science co‑authorship。

**📈 对比分析**

通过计算收缩后两步走比例 ρ 与原始两步走的比值，对比不同网络的楔子压缩误差，实验表明在楔子等价划分下误差可控，ρ 值普遍低于 0.3，说明压缩失真有限。

**⚠️ 局限性**

局限在于仅处理无向简单图，未对有向或加权图给出完整理论；此外楔子等价划分较强，实际网络往往不满足，导致收缩误差上升。

---

## 79. The Prediction-Measurement Gap: Toward Meaning Representations as Scientific Instruments

**arXiv ID:** 2603.10130 | [PDF](https://arxiv.org/pdf/2603.10130v1)

**作者:** Hubert Plisiecki `[一作]` `[通讯]` (IDEAS Research Institute), Hubert Plisiecki (IDEAS Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了面向社会科学测量的文本表示目标体系，阐述了预测与测量之间的差距，并为构建可解释、可追溯、对语义敏感的词向量空间提供了研究路线。

**💡 创新点**

创新点在于：①把“科学可用性”视为独立的目标族，形成一套四大成功准则；②从认知与神经心理学出发，阐明语义空间的几何可读性、可解释性与层次性；③提出三条具体研究方向——几何优先设计、可逆后置变换、意义地图与测量导向评估——为未来的表示学习指明路径。

**🔧 技术方法**

使用的技术主要是：
- 传统静态词向量与上下文变换模型（如BERT、GPT）
- 语义空间几何优化（去极性、白化、超平面投影）
- 可逆流形变换（normalizing flows, flow‑matching）
- 语义维度构建与线性回归推断（如语义尺度、SSD框架）
- 评估方法：余弦相似度稳定性、邻近词一致性、表面变体对比等。

**📊 数据集**

本文主要基于公开的词向量和语言模型数据（如GloVe、fastText、BERT、GPT‑系列），并借鉴社会科学与心理学研究中的语料与量表数据（如种族/性别刻板印象语料、心理测验问卷），但未在论文中给出具体实验数据集。

**📈 对比分析**

比较方法主要是将静态向量与上下文向量按上述四大准则进行评估，强调对几何可读性、可解释性与对非语义干扰的鲁棒性。作者指出静态向量在这些指标上往往优于上下文向量，但后者在语义丰富度方面更强。关于性能，由于缺乏统一的测量评估基准，本文仅给出定性评估与启发式结论，而非量化性能结果。

**⚠️ 局限性**

限制在于：
- 缺乏系统的量化评估和基准，难以客观比较不同方法的优劣。
- 上下文模型的语义可解释性与可测量性仍需深入探索，现有后置变换和意义地图的覆盖率与可扩展性有限。
- 论文主要在理论与方法层面讨论，缺少大规模实验验证，难以验证提出的三条研究方向在实际社会科学测量任务中的有效性。

---

## 80. Reactive Writers: How Co-Writing with AI Changes How We Engage with Ideas

**arXiv ID:** 2603.10374 | [PDF](https://arxiv.org/pdf/2603.10374v1)

**作者:** Advait Bhat `[一作]` (Paul G. Allen School of Computer Science University of Washington), Maurice Jakesch `[通讯]` (Bauhaus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在写作中嵌入式AI助手（带有立场倾向的提示）对写作过程和内容的影响，采用混合方法：19名受试者的回溯式访谈与1291次写作日志的定量分析；

**💡 创新点**

提出了“反应式写作”模型，揭示AI提示通过注意力捕获、同意驱动包容和事后个性化，重塑作者从内部构思转向对AI生成内容的评估与改写，且提供了算法议程设置的过程层面证据；

**🔧 技术方法**

使用基于GPT‑3的AI写作助手，内嵌式实时提示；结合回溯式访谈与交互日志的主题分析、线性与逻辑回归模型；

**📊 数据集**

数据集包含1291个写作会话（462无AI、423正面AI、406负面AI）和19名受试者的访谈记录，日志记录键盘、光标、提示出现与接受等交互；

**📈 对比分析**

与无AI对照组比较，AI辅助仅略减写作时间（约7–10%），但提示主题与最终文本主题高度相关（R²≈0.85），且AI提示显著提高参与者讨论相关主题的概率（OR≈3.97）；

**⚠️ 局限性**

局限性包括实验室情境与人工提示频率可能放大效应；仅使用美国/英国Prolific受试者，样本多样性有限；未能完全排除因AI提示引起的因果偏差；设计以强烈立场倾向模型为主，未检验更中性或非inline提示的普适性；

---

## 81. IMTBench: A Multi-Scenario Cross-Modal Collaborative Evaluation Benchmark for In-Image Machine Translation

**arXiv ID:** 2603.10495 | [PDF](https://arxiv.org/pdf/2603.10495v1)

**作者:** Jiahao Lyu `[一作]` (Institute of Information Engineering, Chinese Academy of Science), Jian Luan `[通讯]` (MiLM Plus, Xiaomi Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了端到端的图像内机器翻译任务，并构建了 IMTBench 基准。

**💡 创新点**

创新点在于提供真实多场景、多语言数据集和跨模态一致性评价指标。

**🔧 技术方法**

使用统一多模态模型、OCR、MT、图像编辑技术以及 COMET、Mask‑LPIPS、PQ、Alignment Score 等评估手段。

**📊 数据集**

使用了 IMTBench 数据集，包括文档、网页、自然场景、PPT 四种场景，覆盖 9 种语言，共 2,500 对实例。

**📈 对比分析**

通过与商业级流水线、专有 UMM 与开源 UMM 的对比，发现 UMM 在自然场景下表现更好，但在低资源语言和精准布局对齐方面仍逊色。

**⚠️ 局限性**

局限性包括对低资源语言训练不足、跨模态对齐精度不足，以及数据规模相对有限。

---

## 82. SpecOps: A Fully Automated AI Agent Testing Framework in Real-World GUI Environments

**arXiv ID:** 2603.10268 | [PDF](https://arxiv.org/pdf/2603.10268v1)

**作者:** Syed Yusuf Ahmed `[一作]` (Purdue University), Calix Barrus Xiangyu Zhang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了SpecOps——一种完全自动化、面向真实 GUI 环境的 LLM 代理测试框架，能够为产品级多模态代理生成、执行并验证测试用例，最终定位真实缺陷。

**💡 创新点**

核心创新点包括：① 专家代理架构，将测试拆分为四个专门化阶段（测试生成、环境搭建、执行、验证），每阶段使用专用工具和规则；② 自适应策略与双专家（Test Architect 与 Test Analyst）协同校验，消除规划冲突；③ 基于屏幕捕获的视觉监控与 Meta‑CoT 推理，降低图像解析和断言误报；④ 对多种 UI 平台（CLI、Web、浏览器扩展）统一使用键鼠交互抽象，实现跨平台一致性。

**🔧 技术方法**

主要技术包括：大型语言模型（Claude 3.7 Sonnet）配合定制 prompt；MCP（多模态交互工具）实现键盘/鼠标操作和屏幕截图；自定义 API（发送邮件、文件系统命令）做环境预置；Meta‑CoT 推理与可视化日志分析；自动化计时、token 监控和成本估算。

**📊 数据集**

使用 5 个真实代理（ProxyAI、Open Interpreter、Self‑Operating Computer、TaxyAI、Autonomous HR Chatbot）共 99 条测试用例，覆盖电子邮件、文件系统、HR 问答三大领域；测试用例从自动特征抽取、基准挖掘与人工补充三阶段生成；对比基线 LLM 脚本和 AutoGPT。

**📈 对比分析**

与 LLM 脚本和 AutoGPT 的对比显示：SpecOps 提供 100% 提示成功率、96% 执行成功率、94% 验证成功率，触发 164 条真实 Bug，F1 分数 0.89；成本仅约 0.73 USD/测试，平均执行时长 < 8 min；基线仅达到 49%/11% 提示成功率，F1 仅 0.23/0，Bug 发现量远低于 SpecOps。

**⚠️ 局限性**

主要局限包括：仅在 5 种代理和 3 个领域上评估，缺乏更广泛的跨域验证；对极端动态 UI 或离线环境的适配尚待探索；仍依赖单一 LLM（Claude），若换模型性能可能变化；屏幕捕获与视觉分析对高分辨率/多显示器场景的鲁棒性有限。

---

## 83. Beyond Sequential Distance: Inter-Modal Distance Invariant Position Encoding

**arXiv ID:** 2603.10863 | [PDF](https://arxiv.org/pdf/2603.10863v1)

**作者:** Lin Chen `[一作]` (Chinese Academy of Sciences), Shiming Xiang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 14250 | [OpenAlex ID](https://openalex.org/A5040673285)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为DIPE的跨模态位置编码方法，用以消除多模大语言模型在长上下文中出现的视觉衰退问题；

**💡 创新点**

创新点在于将位置编码按模态交互解耦：对同一模态的注意力使用标准多模RoPE保持局部结构；对跨模态注意力采用锚定位置编码，使视觉与文本之间的感知距离保持不变，从而抑制距离惩罚；

**🔧 技术方法**

使用RoPE、Multimodal RoPE（MRoPE）、Anchored Position Encoding、FlashAttention及KV Cache技术，并在两阶段训练中结合视觉编码器SigLIP2-SO400M与MLLM Qwen 系列；

**📊 数据集**

在19个公开评测基准上评估，包括CountBench、HRBench、V*、POPE、BLINK、ChartQA、DocVQA、TextVQA、AI2D、InfoVQA、OCRBench、RealWorldQA、MMStar、MMBenchV1.1-EN/CN、MMVP、MathVision、MathVista，以及LLaVA-Pretrain、LLaVA-NeXT、MM-NIAH 等；

**📈 对比分析**

通过在短上下文和添加 8K 文字干扰的长上下文 VQA 协议进行对比；在长上下文中，DIPE 在 MRoPE 基线上平均提升 4.10%（Vanilla RoPE +2.00%，MRoPE-I +2.01%），并在短上下文保持与基线无显著差异；层级视觉注意力分析显示 DIPE 恢复了浅层的视觉关注；在不同模型规模（Qwen2.5-3B、0.5B、Qwen3-1.7B）中均获得 4–8% 的整体提升；

**⚠️ 局限性**

局部实验中仍有少数基准出现轻微退化，且视觉注意力在极端长序列下仍有残余衰减；DIPE 需在实现中拆分查询注意力，增加一点计算复杂度；依赖于 RoPE 的频率分配，若基础位置编码设计不佳仍可能影响效果。

---

## 84. Resource-constrained Amazons chess decision framework integrating large language models and graph attention

**arXiv ID:** 2603.10512 | [PDF](https://arxiv.org/pdf/2603.10512v1)

**作者:** Tianhao Qian `[一作]` (Southeast University), Leszek Rutkowski `[通讯]` (Institute of Computer Science AGH University of Krakow)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个轻量级混合框架，用Graph Attention Autoencoder、Stochastic Graph Genetic Algorithm和GPT‑4o‑mini生成的合成数据结合Monte Carlo Tree Search来解决亚马逊棋的决策问题。

**💡 创新点**

创新点在于将弱到强泛化范式引入棋类AI，利用LLM做弱教师并通过GAT‑AE和SGGA构建信息瓶颈去噪；采用全局深度归一化的MCTS更新机制；融合结构推理与生成学习的轻量级设计，实现在极低计算资源下的高性能。

**🔧 技术方法**

主要技术包括：Monte Carlo Tree Search、UCT+AutoEncoder、Graph Attention Network、Stochastic Graph Genetic Algorithm、GPT‑4o‑mini生成合成数据、轻量级AutoEncoder。

**📊 数据集**

使用由GPT‑4o‑mini自动生成的合成棋局数据，没有公开棋谱，仅依赖模型自身生成的局面信息。

**📈 对比分析**

通过在10×10亚马逊棋盘上与GPT‑4o‑mini、UCTS‑AE、SGGA、GAT‑AE等基线对比，采用胜率(win‑rate)评估；在N=30节点时取得45%胜率，在N=50节点时提升到66.5%，相较基线提升15%–56%，并在低搜索量下击败教师模型。

**⚠️ 局限性**

局限性包括：仍未达到专业引擎的水平；对LLM生成数据质量的依赖与hallucination风险；缺乏最终决策策略与训练收敛判定；仅在资源受限环境和10×10棋盘上验证，未测试更大搜索或更复杂棋盘。

---

## 85. Large Language Models and Book Summarization: Reading or Remembering, Which Is Better?

**arXiv ID:** 2603.09981 | [PDF](https://arxiv.org/pdf/2603.09981v1)

**作者:** Tairan Fu `[一作]` (Politecnico di Milano), Elena Merino-Gómez `[通讯]` (Universidad de Valladolid)

**通讯引用:** 167 | [OpenAlex ID](https://openalex.org/A5068286123)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了大型语言模型（LLM）基于内部记忆与完整文本生成的书籍摘要，探索两者在不同书籍上的表现差异。

**💡 创新点**

首次系统评估了在拥有百万级上下文窗口的LLM中，内部记忆与外部文本对长文本摘要质量的相对影响。

**🔧 技术方法**

采用GPT‑4.1和Gemini‑2.5‑Flash两大闭源LLM生成摘要，并使用LLM‑as‑a‑judge框架进行评估；通过多次生成与多对比的方式获得综合评分。

**📊 数据集**

选取25本公开领域、知名度高且已在训练集中的西方经典小说（如《悲惨世界》《哈姆雷特》《尤利西斯》《蒙特·克里斯多伯爵》）作为实验数据集。

**📈 对比分析**

通过内部与外部摘要的二十五两两对比（共25×25对），计算分数区间[-25,25]，结果显示整体上完整文本摘要更优，但在特定作品（如《神曲》《哈姆雷特》《尤利西斯》《蒙特克里斯多伯爵》）内部摘要表现更佳。

**⚠️ 局限性**

局限包括：仅测试两款闭源LLM，实验集中于25本西方经典作品，使用单一提示语，LLM评判可能带偏见，且对非技术文本或未知作品的泛化能力未知。

---

## 86. A$^2$-Edit: Precise Reference-Guided Image Editing of Arbitrary Objects and Ambiguous Masks

**arXiv ID:** 2603.10685 | [PDF](https://arxiv.org/pdf/2603.10685v1)

**作者:** Huayu Zheng `[一作]` (Shanghai Jiao Tong University), Xiaohong Liu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的参考引导图像修复框架 A^2-Edit，支持任意物体类别与任意掩码精度下的高质量图像编辑。

**💡 创新点**

创新点包括：①在注意力与 FFN 两层同时使用 Mixture of Transformers（MoT）实现动态专家路由；②Mask Annealing Training Strategy（MATS）逐步放宽掩码精度，提升模型对粗糙掩码的鲁棒性；③构建规模宏大、类别丰富的 UniEdit-500K 数据集，突破现有数据集同质化与类别覆盖不足的瓶颈。

**🔧 技术方法**

采用基于 Diffusion Transformer 的 FLUX.1-Fill 作为骨干，结合 LoRA 轻量级专家、Anchor‑Guided Routing、VAE 解码器、多模态特征编码与 MATS 训练策略。

**📊 数据集**

使用自建 UniEdit-500K（8 大类别、209 细分子类、500k 图像对）进行训练，并在 VITON‑HD 与 AnyInsertion 以及自建 UniEdit 测试集上评估。

**📈 对比分析**

在 VITON‑HD、AnyInsertion 以及 UniEdit 测试集上与 FLUX.1‑Fill‑dev、AnyDoor、MimicBrush、FLUX.1‑Kontext、ACE++、InsertAnything 等基线对比，量化指标包括 CLIP‑I、DINO‑I、LPIPS、FID 与 VLM 分数。A^2-Edit 在所有指标上均优于或与专用模型持平，并在用户研究中获得最高偏好率。

**⚠️ 局限性**

局限性：对掩码边缘细节仍有轻微失真；极端非预期场景（如极端姿态或纹理）下的泛化仍待提升；模型推理显存与时间略高于单一专家模型；训练需要大量算力与大规模数据；缺乏实时交互式多模态控制能力。

---

## 87. Backdoor Directions in Vision Transformers

**arXiv ID:** 2603.10806 | [PDF](https://arxiv.org/pdf/2603.10806v1)

**作者:** Sengim Karayalcin `[一作]` (Leiden University), Stjepan Picek `[通讯]` (Radboud University)

**通讯引用:** 4682 | [OpenAlex ID](https://openalex.org/A5024072796)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过识别视觉 Transformer（ViT）中与后门触发器对应的线性方向，验证其因果作用，进而实现后门的诊断、消除、层级传播分析以及与对抗样本的相互作用探究，并提出一种无数据的权重量化检测方法。

**💡 创新点**

创新点在于首次将机制解释方法应用于 ViT 后门，将后门特征抽象为可量化的线性方向；通过激活 steering 与权重正交化证明该方向的因果性；对不同触发器的层级传播进行细致比较；以及基于权重对齐度设计的轻量级后门检测方案。

**🔧 技术方法**

使用的技术包括：基于对照对（clean 与 backdoored）计算激活差异的线性方向提取；激活 steering 与权重正交化进行因果验证；对抗样本的激活差向量与后门方向的余弦相似度分析；以及权重与类别读出向量对齐度得分的 Z-score 检测。

**📊 数据集**

实验数据集涵盖 CIFAR‑10、CIFAR‑100、TinyImageNet，采用多种后门攻击（BadNet、Blended、WaNet、SSBA、BPP、TrojanNN 等）及 ViT 变体（ViT‑B‑16、DeiT‑S、Swin‑S）进行验证。

**📈 对比分析**

与 BackdoorBench 上的防御方法对比，权重正交化几乎将 ASR 降至 5% 以下，且对干净精度影响不大；激活 steering 显示单一方向即可调控后门；权重对齐度检测在隐蔽触发器（WaNet、BPP）上实现高 Z‑score，识别率较高，但对基于补丁触发器的攻击效果有限。

**⚠️ 局限性**

局限性主要在于：方法需先知触发器才能构造线性方向，实际防御中往往不可知；检测方案对已知攻击（如 patch‑based）效果不佳；且在自适应攻击下可被绕过。

---

## 88. World Mouse: Exploring Interactions with a Cross-Reality Cursor

**arXiv ID:** 2603.10984 | [PDF](https://arxiv.org/pdf/2603.10984v1)

**作者:** Esen K. Tütüncü `[一作]` (Institute of Neurosciences of the University of Barcelona), Eric J. Gonzalez `[通讯]` (Google)

**通讯引用:** 632 | [OpenAlex ID](https://openalex.org/A5001572522)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出并实现了一种名为 World Mouse 的跨现实光标，能够把传统二维桌面鼠标映射到混合现实环境中，实现对真实与虚拟物体的精准交互与导航。

**💡 创新点**

创新点在于：①将深度自适应光标与连续的混合场景图结合，使光标能够在物体表面与空旷空间之间无缝切换；②利用语义分割和网格重建把物理世界视为可交互的表面，突破了以往仅在纯虚拟空间中的光标设计；③通过“隐藏网格”实现跨对象导航，实现从二维面板到三维空间的平滑过渡。

**🔧 技术方法**

技术实现主要使用：深度自适应光标算法（In-Depth Mouse 的扩展）、语义分割与网格重建（Meta Scene API / Android XR Scene Meshing）、场景图融合、Voronoi 插值、光标深度动态计算以及传统鼠标输入的映射。

**📊 数据集**

数据集方面，论文未提供专门的数据集；其原型测试主要基于在真实环境中扫描得到的场景网格以及构建的虚拟对象集合，混合使用桌面与移动设备进行交互。

**📈 对比分析**

本文未给出定量评估指标；通过一系列原型演示（物体选择、拖拽、2D‑>3D 转换、IoT 控制等）说明了系统的可行性和低疲劳、高精度的优势；与传统手势、光标或控制器的对比主要基于用户体验和交互直观性。

**⚠️ 局限性**

局限性包括：①不适合需要连续自由手绘的 3D 造型任务；②对物体的语义分割与网格重建精度有依赖，误差或不完整的重建可能导致光标误差；③目前实现多依赖 XR 设备的传感与重建能力，受限于硬件性能和场景复杂度。

---

## 89. GRACE: A Unified 2D Multi-Robot Path Planning Simulator & Benchmark for Grid, Roadmap, And Continuous Environments

**arXiv ID:** 2603.10858 | [PDF](https://arxiv.org/pdf/2603.10858v1)

**作者:** Chuanlong Zang `[一作]` (Robert Bosch GmbH), Wolfgang Hönig `[通讯]` (Technical University of Berlin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GRACE平台，统一的二维模拟器和基准测试框架，支持网格、路图和连续三种表示下的多机器人路径规划与评估。

**💡 创新点**

创新点在于：① 在同一API和评估协议下实现三种抽象层次的无缝转换；② 通过可复现的抽象运算器与跨表示转换机制，使不同算法在相同实例上可直接比较；③ 通过实证验证了表示层次与算法之间的取舍，揭示何时离散抽象足够，何时需要连续动力学精度。

**🔧 技术方法**

使用Box2D作为确定性连续时域仿真核心；OMPL构建路图；SFML+ImGui实现交互UI；统一规划接口通过共享库或子进程调用各种公开MAPF/MRMP算法。

**📊 数据集**

采用公开地图集（LoRR、Flatland、POGEMA等），并自行生成的5×5 m²、540m×140m等实验环境，测试机器人团队规模从数个到2500个。

**📈 对比分析**

通过统一实例化与指标（成功率、SoC、最晚完成时间、规划时间、实时因子）进行横向比较。结果显示：路图规划在保持接近连续性能的同时，规划时间和最晚完成时间显著降低；网格规划进一步加速但导致SoC和最晚完成时间略增。不同规划器在网格下的成功率、路径质量和速度也被量化对比。

**⚠️ 局限性**

局限性：仅限二维平面、静态已知环境；机器人形状固定为凸多边形；对路图/网格的抽象采用固定规则，未考虑动态障碍、非凸或可变形机器人、深度学习模型的训练与部署，以及与ROS 2等操作系统的完整集成。

---

## 90. Sublinear-Time Reconfiguration of Programmable Matter with Joint Movements

**arXiv ID:** 2603.10720 | [PDF](https://arxiv.org/pdf/2603.10720v1)

**作者:** Manish Kumar `[一作]` (Indian Institute of Technology Ropar), Christian Scheideler `[通讯]` (Paderborn University)

**通讯引用:** 4553 | [OpenAlex ID](https://openalex.org/A5063355098)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文在几何 amoebot 模型的联合运动扩展下，提出了集中式重构算法，能够将任意连通结构在子线性时间内重构为线段，并给出了针对螺旋结构的常数时间线段重构算法。

**💡 创新点**

创新点在于：①证明联合运动扩展下即可实现子线性时间的通用重构（O(√n log n)）；②首次给出常数时间的螺旋到线段重构；③设计了新的常数时间原语（隧道、剪切、平行四边形、三角形、梯形）以实现高效并行移动；④通过集中式调度充分利用全局信息，突破了传统模型下的直径下界。

**🔧 技术方法**

使用的技术包括：集中式全局调度、基于原语的结构化移动（如隧道、剪切等）、单调化、直方图化、束化、合并行（combing）以及对齐操作；算法分析采用状态图、迭代计数和并行化评估；证明过程多用几何构造与递归式复杂度分析。

**📊 数据集**

论文为理论性工作，没有使用具体数据集，研究对象为大小为 n 的任意连通 amoebot 结构。

**📈 对比分析**

与之前仅能实现线性或基于直径的 O(D) 时间的分布式方案相比，本文的集中式方案在时间上实现了子线性（O(√n log n)）以及常数时间（螺旋）级别的改进，显著提升了重构效率；但缺少实验验证或实际部署结果。

**⚠️ 局限性**

局限性包括：仅提供集中式算法，未给出分布式实现；仅对线段目标结构给出通用算法，未探讨其他目标形状；常数时间的螺旋方案仅适用于满足特定段数的螺旋；尚未实现多项式对数时间的通用重构，仍是未解决的开放问题。

---

## 91. A Bipartite Graph Approach to U.S.-China Cross-Market Return Forecasting

**arXiv ID:** 2603.10559 | [PDF](https://arxiv.org/pdf/2603.10559v1)

**作者:** Jing Liu `[一作]` (University of Oxford), Mihai Cucuringu `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个基于时间顺序的有向双部图，用来捕捉美中两市股票间的跨市场预测关系，并将该图作为特征选择层嵌入多种机器学习模型，预测下一交易日的开盘至收盘收益。

**💡 创新点**

创新点在于：①利用非重叠交易时段的时间先后关系进行滚动窗口假设检验，形成稀疏且可解释的双部图；②将图结构与多种回归/树/集成模型相结合，系统评估跨市场信息对个股收益预测的增量价值；③揭示美股对中股收益预测的显著方向性不对称。

**🔧 技术方法**

采用线性、正则化、核、树和集成的十种机器学习算法（OLS、LASSO、Ridge、SVM、XGBoost、LGBM、RF、AdaBoost、ensemble-avg、ensemble-med）作为预测器。

**📊 数据集**

数据来源为2014-2021年CRSP（美国）和Wind（中国）公开市场的500只市值最大股票的每日价格，构造的市场超额收益（US: SPY, CN: 513500.SH）。

**📈 对比分析**

通过滚动250天窗口、每10天更新模型、无前瞻偏差评估，结果显示：美股pvCLCL信息对中股OPCL预测的Sharpe Ratio普遍超过1（部分模型接近2），而反向预测表现显著较弱；与单市场基线和无图基线相比，图+跨市信息的组合取得最佳风险调整收益。

**⚠️ 局限性**

主要局限包括：①未考虑交易成本、流动性冲击及做空约束，Sharpe Ratio为预成本值；②双部图的边选择基于单变量检验，可能包含多重检验导致的伪边；③股票池按期末市值选取，存在前瞻偏差；④仅聚焦美中两市，缺乏对其他市场的普适性验证。

---

## 92. Actor-Accelerated Policy Dual Averaging for Reinforcement Learning in Continuous Action Spaces

**arXiv ID:** 2603.10199 | [PDF](https://arxiv.org/pdf/2603.10199v1)

**作者:** Ji Gao `[一作]` (Georgia Institute of Technology), Zhaohui Tong `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 3275 | [OpenAlex ID](https://openalex.org/A5012676327)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了Actor-Accelerated Policy Dual Averaging（PDA）算法，利用学习到的策略网络逼近原本需要求解的优化子问题，从而在连续动作空间中加速PDA的执行；

**💡 创新点**

创新点在于将PDA的理论优势与可学习的策略网络结合，既保持了PDA的收敛保证，又通过近似优化子问题显著降低了计算成本；

**🔧 技术方法**

主要技术包括Policy Dual Averaging框架、Bregman散度正则化、弱凸优化、函数逼近（神经网络近似优势函数和策略）、经验回放与梯度更新；

**📊 数据集**

实验使用了MuJoCo与Box2D的连续控制任务（如HalfCheetah、Ant、Walker2d、Hopper、Humanoid）以及OR-Gym中的运筹学任务（Newsvendor、PortfolioOpt、InvManagement）；

**📈 对比分析**

与PPO、TRPO、NPG等基准对比，Actor-Accelerated PDA在大多数任务上表现优于PPO，尤其在高维运动控制和运筹学环境中获得更高的累计回报；

**⚠️ 局限性**

局限性包括对近似误差的依赖（需保证演员逼近误差在可接受范围内）、对超参数（λ、σ_0等）的敏感性以及在某些任务中仍无法完全匹配最优基准。

---

## 93. A Systematic Study of Pseudo-Relevance Feedback with LLMs

**arXiv ID:** 2603.11008 | [PDF](https://arxiv.org/pdf/2603.11008v1)

**作者:** Nour Jedidi `[一作]` (University of Waterloo), Jimmy Lin `[通讯]` (University of Waterloo)

**通讯引用:** 22324 | [OpenAlex ID](https://openalex.org/A5082997975)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对使用大语言模型进行伪相关反馈（PRF）方法进行系统性实验，探讨反馈来源和反馈模型对检索性能的独立影响。

**💡 创新点**

创新点在于将反馈来源（语料、LLM 或两者结合）与反馈模型（Rocchio、RM3、向量平均等）两大维度进行解耦评估，揭示在不同检索器和数据集下最佳组合策略，并提供对延迟与效果权衡的实证分析。

**🔧 技术方法**

采用大语言模型 Qwen3‑14B 生成假设答案文档、Umbrela 进行文档相关性评分，结合传统反馈模型（Rocchio、RM3）和向量更新（平均向量、Rocchio向量），并在 BM25、Contriever、Contriever‑MS‑MARCO 等检索器上实现。

**📊 数据集**

使用 13 个低资源 BEIR 数据集（如 TREC‑Covid、TREC‑News、FiQA、BioASQ 等）进行评测，覆盖新闻、问答、实体检索、医学、事实检查等任务。

**📈 对比分析**

实验结果显示：① 对 LLM 生成的反馈，Rocchio 模型往往优于 RM3 或平均向量；② LLM 生成的反馈在大多数场景下比仅用语料检索的反馈更有效；③ 对于稠密检索器，独立融合两源反馈（Corpus + LLM）可显著提升效果；④ 在 BM25 上，若使用强初始检索器或足量候选文档，语料反馈优势更明显。整体而言，LLM‑only 方案在效果与延迟上更具成本效益。

**⚠️ 局限性**

局限性包括：① 仅测试三种检索器，未覆盖更强大或多模态模型；② 延迟评估仅考虑 LLM 推理，忽略索引或检索时间；③ 反馈源与模型的交互仍未完全解析，可能因超参数或候选数量而产生偏差。

---

## 94. Federated Learning-driven Beam Management in LEO 6G Non-Terrestrial Networks

**arXiv ID:** 2603.10983 | [PDF](https://arxiv.org/pdf/2603.10983v1)

**作者:** Maria Lamprini Bartsioka `[一作]`, Iakovos S. Venieris `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

本文针对LEO低轨非地面网络（NTN）提出了基于联邦学习的波束选择框架，实现了在多轨道星座中通过HAPS进行分布式模型训练和聚合。

**💡 创新点**

创新点在于将轨道面视为联邦学习客户端，利用HAPS做边缘聚合，设计了图神经网络（GNN）捕捉波束间关系，从而提升预测精度与波束稳定性。

**🔧 技术方法**

使用的技术包括联邦学习（FedAvg）、多层感知机（MLP）、图神经网络（GNN）以及基于S域的波束代码书。

**📊 数据集**

数据集来自高真实感仿真，包含1000个快照的卫星-用户几何、信道特性、波束方向以及对应的最大SNR波束标签。

**📈 对比分析**

通过与传统MLP对比，采用Top‑1、Top‑3准确率、训练时间和模型大小等指标评估，GNN在Top‑1准确率上提升至96.14%（相较MLP 88.41%），Top‑3近乎完美，且波束切换稳定性更优，训练时间略长。

**⚠️ 局限性**

局限性包括：模型仍有一定复杂度，GNN训练耗时更长；实验仅基于仿真数据，缺乏真实网络验证；只关注单波束预测，未考虑干扰或多用户联合优化。

---

## 95. STADA: Specification-based Testing for Autonomous Driving Agents

**arXiv ID:** 2603.10940 | [PDF](https://arxiv.org/pdf/2603.10940v1)

**作者:** Joy Saha `[一作]` (University of Virginia), Matthew B. Dwyer `[通讯]` (University of Virginia)

**通讯引用:** 9983 | [OpenAlex ID](https://openalex.org/A5086757331)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于正式规范的自动驾驶代理测试框架，能够根据 LTL_f 规范生成满足前置条件的完整场景与轨迹并在 CARLA 仿真中验证后置条件。

**💡 创新点**

创新点在于将规范拆分为关系图（RG）并系统地枚举满足前置条件的不同配置，从而产生多样化且覆盖度高的测试用例，并通过动态 NPC 速度调节进一步逼近规范要求。

**🔧 技术方法**

使用了 RFOL 与 LTL_f 的组合、确定性有限自动机、场景图构造、K‑shortest 路径与贪心多样性选择，以及 CARLA 的 Python 接口与场景图生成器。

**📊 数据集**

主要使用 CARLA 0.9.15 的 Town10HD 地图与 ClearNoon 天气，评估基于 8 条官方弗吉尼亚州交通法规的规范，实验中共生成 80 条仿真轨迹。

**📈 对比分析**

与两种随机放置基线、10 倍规模基线以及 ScenicNL 生成的基线相比，框架在最细粒度覆盖度（cov_1）上提升 47–74 个百分点，覆盖率高 2 倍且在相同测试量下比最佳基线少 6 倍仿真，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括对需要主动超车或罕见交通情景（如对手车辆主动停靠）的覆盖率有限，依赖于预设的节点预算与关系图枚举策略，且在某些规范下仍需人工调参或进一步改进仿真控制策略。

---

## 96. Are Video Reasoning Models Ready to Go Outside?

**arXiv ID:** 2603.10652 | [PDF](https://arxiv.org/pdf/2603.10652v1)

**作者:** Yangfan He `[一作]` (Nanyang Technological University Singapore), Jaehong Yoon `[通讯]` (Nanyang Technological University Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套针对真实世界视觉干扰（如恶劣天气、遮挡、灯光变化、相机运动）的视频推理鲁棒训练框架及其评测基准。

**💡 创新点**

创新点在于：①设计结构化时空扰动生成器，模拟四类实际干扰；②引入自我反思难度感知的在线训练策略，动态构建适应性课程；③使用双分支一致性对齐与奖励模型，通过GRPO实现鲁棒奖励优化；④构建PVRBench评测基准，系统性评估视频推理在干扰下的准确率和推理质量。

**🔧 技术方法**

主要技术包括结构化时空扰动、难度感知自我反思采样、记忆缓冲回放、双分支一致性对齐、奖励建模（格式、准确率、对齐奖励）以及基于LLM的判别器与GRPO优化。

**📊 数据集**

使用的数据集有：Video‑R1‑260k子集（训练）、PVRBench（9K视频、51K问答，含多种扰动）、UrbanVideo、VisBench（基准评估）。

**📈 对比分析**

实验表明，与多种开源和专有模型对比，ROVA在PVRBench、UrbanVideo、VisBench上的平均准确率提升≥24%，推理质量提升≥9%，在某些场景甚至匹配或超越大模型；同时训练效率更高，GPU时长比基线低约6%。

**⚠️ 局限性**

局限性包括：扰动生成仍基于预定义样式，难以覆盖极端或新型视觉噪声；双分支对齐需要额外前向推理，增加计算成本；在极端遮挡或光照剧变下仍可能产生错误推理。

---

## 97. Sample-and-Search: An Effective Algorithm for Learning-Augmented k-Median Clustering in High dimensions

**arXiv ID:** 2603.10721 | [PDF](https://arxiv.org/pdf/2603.10721v1)

**作者:** Kangke Cheng `[一作]` (University of Science and Technology of China), Hu Ding `[通讯]` (University of Science and Technology of China)

**通讯引用:** 12592 | [OpenAlex ID](https://openalex.org/A5028970899)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于采样与搜索的学习增强k‑median聚类算法（Sample‑and‑Search），通过预测标签预先对点集进行划分，随后在低维子空间上构造候选中心并进行贪心选择，显著降低时间复杂度。

**💡 创新点**

核心创新点在于：①利用随机采样得到的子集生成的低维子空间能够逼近真实中心，消除对高维空间全局搜索的需求；②在该子空间上使用网格离散化生成候选中心，实现时间复杂度从指数型降低到线性(含对α、ε的幂次)；③在理论上保持与现有最佳方法相同的近似比 1+(6+ε)α−4α²/(1−α)(1−2α)，并在实验中展示更低的聚类成本和更快的运行速度。

**🔧 技术方法**

技术手段包括：随机采样、几何中位点性质、低维子空间网格化、贪心中心选择、误差率α控制的候选集构造，以及对误差率的概率分析。

**📊 数据集**

使用的公开数据集有：CIFAR‑10 (n=50k,d=3072)、PHY (n=10k,d=50)、MNIST (n=1.8k,d=64)、Fashion‑MNIST (n=60k,d=784)，并在不同误差率α与k值下进行实验。

**📈 对比分析**

与基线方法（KMed++、Predictor、EFS+、HFH+、NCN）在相同错误率α下比较，结果表明 Sample‑and‑Search 在聚类成本上往往更低（约0.1%–0.5%），并在运行时间上相对最快，尤其在高维数据上加速可达10倍以上；同时保持与NCN等现有最优方法相当的近似比。

**⚠️ 局限性**

限制与待改进点：①仍需在候选中心构造中接受对α、ε的指数级依赖，难以进一步降低；②适用范围仅限于误差率α<1/2，且假设预测标签已给定；③缺乏对在线/流式数据场景的支持，未来可考虑流式学习增强聚类的扩展。

---

## 98. Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control

**arXiv ID:** 2603.10572 | [PDF](https://arxiv.org/pdf/2603.10572v1)

**作者:** Matti Vahs `[一作]` (KTH Royal Institute of Technology), Jana Tumova `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1857 | [OpenAlex ID](https://openalex.org/A5042698317)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于信念空间的分层控制架构，利用Belief Control Lyapunov Function (BCLF) 和 Belief Control Barrier Function (BCBF) 分别实现信息采集与安全约束，从而解决reach-avoid POMDP问题。

**💡 创新点**

创新点在于：①把信息采集视为信念空间Lyapunov收敛问题，引入BCLF；②扩展BCBF并结合合成预测实现有限时概率安全保障；③通过强化学习学习BCLF，实现对高维非高斯信念的可训练控制证书；④将三种控制行为在不同时间尺度上分层实现，提升实时可行性。

**🔧 技术方法**

使用的技术包括：粒子滤波（非高斯信念表示）、强化学习（DQN/TD3）学习价值函数并构造BCLF、合成预测（conformal prediction）用于概率安全保证、控制Lyapunov/Barrier理论、线性二次调节（QP）实现实时控制。

**📊 数据集**

数据集与实验环境包括：三种仿真环境（Constrained Lightdark、Constrained Antenna、Constrained Bumper）以及真实的空间机器人平台硬件实验（模拟微重力平台进行碰撞定位）。

**📈 对比分析**

与基于MCTS的约束POMDP求解器（CPOMCPOW、CPFT-DPW）以及仅使用参考控制、参考控制+BCBF、参考控制+BCLF等基线进行对比。实验显示，提出的分层架构在成功率、避障率和任务完成率上均优于基线，且硬件实验在粒子数>10⁴时实现实时控制。

**⚠️ 局限性**

局限性包括：①对连续动作空间的BCLF尚未完善；②理论安全与收敛性保证基于理想假设（如粒子重采样独立、无限粒子等）；③在更高维状态空间或更复杂感知环境下，粒子滤波与合成预测的计算量仍是瓶颈；④冲突解决机制仍为经验式，缺乏严格形式化保障。

---

## 99. Probabilistic Verification of Voice Anti-Spoofing Models

**arXiv ID:** 2603.10713 | [PDF](https://arxiv.org/pdf/2603.10713v1)

**作者:** Evgeny Kushnir `[一作]`, Oleg Y. Rogov `[通讯]` (Applied AI Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 PV-VASM，一种概率框架，用来在黑盒、模型无关的情况下对语音反欺骗模型（VASMs）在输入变换（如滤波、增益、噪声等）以及文本转语音（TTS）和语音克隆（VC）生成器产生的合成语音上的鲁棒性进行验证，并给出误分类概率的上界。

**💡 创新点**

创新点：①首次将概率浓缩不等式（Chernoff/Chernoff‑Cramer）与音频变换/生成模型结合，形成可对未知生成器鲁棒性进行正式认证的通用方法；②实现对任意标注保持变换和任意神经生成器的鲁棒性验证；③提供可调节的超参数（t、δ、n、k）来权衡上界紧密度与计算成本。

**🔧 技术方法**

技术：概率浓缩不等式、随机变量采样与统计估计、一维/二维误差概率估计、一次性置信区间（McKay 近似）、基于变换参数空间采样的随机化验证算法，配合 Wav2Vec2‑AASIST 语音编码器和后端分类器。

**📊 数据集**

数据集：训练使用 ASVspoof 2019/2021 LA/DF、ASVspoof 5、ADD 22-23、DFADD、SONAR、CFAD、MLAAD、Speech‑to‑Latex、Mozilla Common Voice；验证使用 ASVspoof 5 测试集（300 样本）以及从 Vosk、Silero、Coqui XTTS‑v2、f5‑TTS、CosyVoice、ElevenLabs、Finevoice 生成的合成语音；背景噪声来源于 Musan，房间冲击响应来自 OpenSLR。

**📈 对比分析**

对比方法：基于 PCA（probabilistically certified accuracy）和误差概率 p 的评估。实验表明：对简单变换（LPF、HPF、时间拉伸）PCA 高、p 低；对强噪声、窄带滤波等困难变换，PCA 降低、p 上升；对 TTS/VC 生成器，预微调模型的上界较高，微调后显著下降（如 Vosk 从 0.1352 降至 0.0686，ElevenLabs 从 0.3308 降至 0.2002）。整体性能表明 PV‑VASM 能给出实用的鲁棒性证书，尤其在微调后能显著提升。

**⚠️ 局限性**

局限性：①上界可能过于保守，尤其在生成器导致 eᵗZ 方差很大时；②对超参数 t、δ 的取值敏感，范围有限时可能无法取得最优结果；③验证成本随 n、k、m 变化显著，需权衡计算与精度；④未与现有具体防御算法直接对比，仅给出基准指标；⑤对多类/阈值不同的分类器适配性未充分探索。

---

## 100. DUCTILE: Agentic LLM Orchestration of Engineering Analysis in Product Development Practice

**arXiv ID:** 2603.10249 | [PDF](https://arxiv.org/pdf/2603.10249v1)

**作者:** Alejandro Pradas-Gomez `[一作]`, Ola Isaksson `[通讯]` (Chalmers University of Technology)

**通讯引用:** 2727 | [OpenAlex ID](https://openalex.org/A5025612542)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一种基于大型语言模型（LLM）的代理式编排框架 DUCTILE，用于航空航天结构强度分析流程中对输入格式、单位、命名和方法变更的自适应处理。

**💡 创新点**

核心创新在于将 LLM 代理的自适应编排与已验证的工程工具分离，保证可审计、可重复并符合认证要求；提出了 pass^k 评估与 LLM‑as‑judge 双重验证方法；并展示了两种工程师监督模式（全委托与逐步交互）。

**🔧 技术方法**

采用 Claude Opus 4.6 LLM、Claude Code 代理框架、思考模式、工具调用、ReAct 编排、OpenTelemetry 日志记录；使用 Opus 4.6 进行 LLM‑as‑judge 自动评估。

**📊 数据集**

使用真实工业案例数据——GKN Aerospace 的 TRS 结构负载文件及其四项输入偏差；在 10 次独立运行中进行评估；未使用公开公开数据集，仅基于公司内部负载文件和设计实践文档。

**📈 对比分析**

评估方法为定量一致性检查与 LLM‑as‑judge 双重验证，采用 pass^k=10；所有 10 次运行均通过，证明系统对四项偏差均能正确处理，输出与人工专家一致，显著优于传统脚本流水线。

**⚠️ 局限性**

局限性包括：仅验证单一静力分析任务；实验仅涉及两名工程师；依赖特定 LLM（Claude Opus 4.6）；需要完整文档支持；仅适用于文本/数据工作流，未覆盖几何建模等；缺乏跨组织、跨任务的大规模验证。

---

## 101. UniCom: Unified Multimodal Modeling via Compressed Continuous Semantic Representations

**arXiv ID:** 2603.10702 | [PDF](https://arxiv.org/pdf/2603.10702v1)

**作者:** Yaqi Zhao `[一作]` (Peking University), Liefeng Bo `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了UniCom框架，通过将高维视觉嵌入压缩为连续低维语义表示，实现统一的视觉理解与生成；

**💡 创新点**

核心创新是使用轻量级注意力压缩器在通道维度上压缩视觉特征，既保持语义与细节，又显著提升生成可建模性；

**🔧 技术方法**

结合SigLIP视觉编码器、Qwen‑2.5‑7B‑Instruct语言模型、FLUX.1‑dev扩散解码器，采用流匹配训练与统一的压缩空间；

**📊 数据集**

在ImageNet、GenEval、DPG‑Bench、ImgEdit‑Bench、GEdit‑Bench、KRIS‑Bench、WorldEdit等公开数据集上进行训练与评测；

**📈 对比分析**

与传统VAE或量化方法相比，UniCom在图像重建、文本‑图像生成与编辑任务中取得或逼近SOTA，重建FID低于1.1、编辑一致性高于VAE基线，文本渲染质量显著提升；

**⚠️ 局限性**

局限性包括对大规模算力与大模型的依赖、压缩后仍存在细节略微损失、尚未验证到视频或更复杂跨模态任务。

---

## 102. PEEM: Prompt Engineering Evaluation Metrics for Interpretable Joint Evaluation of Prompts and Responses

**arXiv ID:** 2603.10477 | [PDF](https://arxiv.org/pdf/2603.10477v1)

**作者:** Minki Hong `[一作]`, Jihie Kim `[通讯]` (Dongguk University)

**通讯引用:** 2610 | [OpenAlex ID](https://openalex.org/A5080664764)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PEEM框架，用LLM评估器联合评估提示和回答，给出1–5分的多维度分数和自然语言理由，形成可解释、可复现的评估流程。

**💡 创新点**

首创将提示质量与回答质量联合评估，设计九个评价轴并提供解释性理由；实现评估器无须微调、可跨模型使用，并用这些评估结果驱动零样本提示重写，显著提升任务准确率。

**🔧 技术方法**

使用LLM评估器（默认GPT‑4o‑mini），构造基于规则的评价模板，生成Likert分数和理由；通过零样本重写循环、对抗与同义重写实验验证鲁棒性；交叉评估器一致性分析。

**📊 数据集**

七大基准：AG News、ARC‑Challenge、ARC‑Easy、BBH、GSM8K、MMLU、SST‑2；任务模型包括Gemma‑2‑9B‑IT、LLaMA‑3.1‑8B‑IT、Qwen‑2.5‑7B‑IT、GPT‑4o‑mini、Gemini‑2.5‑Flash，评估器模型5种。

**📈 对比分析**

与传统准确率对齐：Spearman ≈ 0.97，Pearson ≈ 0.94；跨评估器相关性0.68–0.85；同义改写鲁棒率≈77–81%；对抗提示检测准确；在提示重写实验中，在AG News、SST‑2、GSM8K等任务上提升至最多11.7个百分点，超过监督和RL基线。

**⚠️ 局限性**

仅针对英文数据集；人类评估样本有限；偶尔出现错误判分（约1%）需进一步校准；未系统分析提示长度对评估的影响；对非文本多模态或领域特定任务的扩展尚未验证。

---

## 103. SENS-ASR: Semantic Embedding injection in Neural-transducer for Streaming Automatic Speech Recognition

**arXiv ID:** 2603.10005 | [PDF](https://arxiv.org/pdf/2603.10005v1)

**作者:** Youness Dkhissi `[一作]` (Orange Innovation), Anthony Larcher `[通讯]` (Le Mans Université)

**通讯引用:** 1989 | [OpenAlex ID](https://openalex.org/A5002979461)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出SENS-ASR框架，将语义信息注入流式ASR的帧嵌入中，以提升语义层面的识别质量。

**💡 创新点**

核心创新在于通过一个实时上下文模块（Context Module）与教师句子嵌入模型的知识蒸馏，利用过去音频的语义上下文丰富每个帧的表示。

**🔧 技术方法**

采用RNN‑Transducer模型、动态块训练（DCT）、MPNet句子嵌入模型、Mistral‑7B生成的重述文本进行教师模型微调，并通过注意力池化构建语义上下文。

**📊 数据集**

实验数据集包括LibriSpeech（test‑clean/test‑other）和TEDLIUM‑2，使用这些公开语音数据评估流式识别性能。

**📈 对比分析**

与基线RNN‑T相比，在160 ms与320 ms小块推理时，WER分别降低约0.34%和0.09%；在更大块（640 ms/1280 ms）提升有限，表明语义注入对短时上下文更有效。

**⚠️ 局限性**

局限性包括对LLM训练语料的潜在泄漏风险、对单一语言（英语）的依赖、在大块或全上下文条件下提升有限，以及上下文模块训练所增加的计算与实现复杂度。

---

## 104. Semantic Landmark Particle Filter for Robot Localisation in Vineyards

**arXiv ID:** 2603.10847 | [PDF](https://arxiv.org/pdf/2603.10847v1)

**作者:** Rajitha de Silva `[一作]` (University of Lincoln), Riccardo Polvara `[通讯]` (University of Lincoln)

**通讯引用:** 905 | [OpenAlex ID](https://openalex.org/A5009664515)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种语义地标粒子滤波器(SLPF)，通过将识别到的葡萄藤干和支柱组织成行对齐的“语义墙”，在粒子滤波器的测量模型中直接融入结构信息，并结合可调权重的GNSS先验，实现葡萄园行级定位的鲁棒性。

**💡 创新点**

创新点在于：①将散落的稳定地标聚合为连续的语义墙，显式捕捉行间几何差异；②将这些语义墙嵌入粒子滤波的观测似然中，直接惩罚跨行假设；③根据语义观测稠密程度动态调整GNSS权重，实现头部转弯时的全局约束。

**🔧 技术方法**

采用YOLOv9实例分割检测干与支柱，利用RGB‑D深度进行BEV投影；使用2D激光雷达做射线投影；粒子滤波（SLPF）结合语义墙、背景自由空间、GNSS先验和通道先验；采用高斯噪声模型与自适应权重机制。

**📊 数据集**

训练分割模型使用SemanticBLT数据集；实地实验数据来自一片10行、约385 m²的葡萄园，记录RGB‑D、2D激光雷达和RTK‑GNSS轨迹；对GNSS进行噪声仿真以评估鲁棒性。

**📈 对比分析**

与AMCL（几何粒子滤波）、AMCL+GNSS（卡尔曼融合）、RTAB‑Map（RGB/RGB‑D视觉SLAM）以及纯GNSS基准进行对比。SLPF在两条行走方向上均取得最低原始绝对姿态误差（APE），相较AMCL降低22%/65%，相较GNSS降低65%/61%；行正确率提升至0.73/0.67，横向误差下降至1.26/1.46 m，说明能更可靠地从错误行恢复。

**⚠️ 局限性**

依赖于已测量的语义墙地图和稳定地标的检测；若树干/支柱被遮挡、结构改造或GNSS信号严重衰减，定位性能会下降；目前未实现动态地图更新和跨季节适配，需进一步研究。

---

## 105. Human-AI Co-reasoning for Clinical Diagnosis with Evidence-Integrated Language Agent

**arXiv ID:** 2603.10492 | [PDF](https://arxiv.org/pdf/2603.10492v1)

**作者:** Zhongzhen Huang `[一作]`, Xiaomu Li `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发了一款医学推理智能体 PULSE，结合大语言模型与科学文献检索，支持内分泌疾病的诊断决策。

**💡 创新点**

创新点在于将检索增强的 LLM 与多轮假设生成、语义聚合、证据检索、摘要融合等流程耦合，并在真实内分泌病例中系统评估其诊断表现。

**🔧 技术方法**

采用约32B 参数、Qwen3 tokenizer 的医学对齐 LLM，结合 PubMed E‑utilities API 实时检索，并使用 Python+Megatron‑LM 进行训练与推理。

**📊 数据集**

评估数据集为 82 例真实内分泌病例，涵盖七个子专业与三类疾病罕见度（常见、非常见、稀有）。

**📈 对比分析**

与住院医师、初级专科医师及资深专家对比，PULSE 在 Top@1（57.3%）和 Top@4（79.3%）上与资深专家相当，显著优于住院医师和初级专科；在罕见病上表现稳定；在医患协作中，PULSE 通过串行与并行工作流均提升诊断准确率。

**⚠️ 局限性**

局限包括模型可能产生幻觉、对少数群体的公平性未充分评估、依赖高质量检索与可解释性以避免自动化偏误，以及未在多中心真实临床环境中验证。

---

## 106. Muscle Synergy Priors Enhance Biomechanical Fidelity in Predictive Musculoskeletal Locomotion Simulation

**arXiv ID:** 2603.10474 | [PDF](https://arxiv.org/pdf/2603.10474v1)

**作者:** Ilseung Park `[一作]` (Carnegie Mellon University), Jooeun Ahn `[通讯]` (Seoul National University)

**通讯引用:** 553 | [OpenAlex ID](https://openalex.org/A5002868202)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了基于肌肉协同的强化学习控制器，实现高维肌肉驱动模型的稳定步态生成

**💡 创新点**

将实验逆向求解得到的肌肉协同作为动作空间约束，显著提升了运动的生物力学真实性

**🔧 技术方法**

使用非负矩阵分解提取协同、软演员-批评家（SAC）强化学习、OpenSim/Hyfydy 三维肌肉骨骼模拟

**📊 数据集**

利用一名健康受试者的 10 米地面行走数据以及公开的 Carmargo 与 Scherpereel 斜坡跑步数据进行验证

**📈 对比分析**

通过与实验数据的关节角度、关节力矩、地面反作用力和肌电激活的 RMSE 比例和相关系数比较，协同约束控制在速度/斜坡变化下的误差普遍低于无约束控制

**⚠️ 局限性**

仅验证了平地及 ±6° 的轻度斜坡，协同基底来自单个受试者，且未直接使用 EMG 数据，模型在跑步、转弯、楼梯等复杂情境及不同人群中的泛化性待进一步验证

---

## 107. Variance-Aware Adaptive Weighting for Diffusion Model Training

**arXiv ID:** 2603.10391 | [PDF](https://arxiv.org/pdf/2603.10391v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 108. Cross-Hand Latent Representation for Vision-Language-Action Models

**arXiv ID:** 2603.10158 | [PDF](https://arxiv.org/pdf/2603.10158v1)

**作者:** Guangqi Jiang `[一作]` (UC San Diego), Xueyan Zou `[通讯]` (UC San Diego)

**通讯引用:** 613 | [OpenAlex ID](https://openalex.org/A5112985166)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个具备统一潜在动作空间的 Vision–Language–Action (VLA) 框架 XL‑VLA，支持多种灵巧手臂的跨体态学习与零样本迁移；

**💡 创新点**

创新点在于构建了一个无监督、跨手臂的潜在动作空间，通过重建、重定位和 KL 正则化实现不同手的动作对齐，消除了对手特定示例或 IK 轨迹的依赖；

**🔧 技术方法**

使用了 VAE 结构的多头潜在编码器/解码器、前向运动学约束、KL 正则化，以及基于大型 VLM（PaliGemma）预训练的 VLA 模型；

**📊 数据集**

使用了由 10 项任务、4 种新型灵巧手（Ability、Paxini DexH13、X‑Hand1、Inspire）收集的 2 M 状态-动作对的远程操作数据；

**📈 对比分析**

与标准 VLA 基线（π₀）以及仅采用运动学重定位的基线相比，XL‑VLA 在跨手臂任务上平均提升成功率约 35 %（从 0.55 提升至 0.90），并在零样本任务迁移与跨机器人平台上亦表现出显著优势；

**⚠️ 局限性**

局限性包括潜在空间维度需精细调节，过大可能破坏体态不变性；实验主要基于桌面任务与有限手臂，尚未验证在更复杂或极端动力学条件下的泛化；

---

## 109. Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context

**arXiv ID:** 2603.10573 | [PDF](https://arxiv.org/pdf/2603.10573v1)

**作者:** Faris Chaudhry `[一作]` (Imperial College), Siddhant Gadkari `[通讯]` (Imperial College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究Transformer在二元假设检验中的in‑context learning（ICL），并证明其能够近似学习贝叶斯最优的似然比统计量。

**💡 创新点**

创新点在于将决策理论与Transformer内部机制相结合，利用已知的最优统计量作为基准，系统揭示了模型在不同几何任务下的自适应计算策略。

**🔧 技术方法**

采用自制高斯判别任务、Logit Lens、输出值（OV）电路对齐、核回归对照等技术进行机制分析与性能评估。

**📊 数据集**

使用自生成的两类Gaussian数据集：线性均值偏移（Task A）和方差判别（Task B），无公开数据集。

**📈 对比分析**

通过与理论Bayes最优估计器对比：在Task B中模型准确率约为83 %（oracle 84 %），在Task A中约为78 %（oracle 85 %），LLR回归相关性高，表明模型成功近似最优统计量。

**⚠️ 局限性**

局限性在于仅在低维、小型Transformer与简单二元任务上验证，未测试大规模模型与复杂真实分布，且可解释性结果仍为相关性而非因果证明。

---

## 110. One Adapter for All: Towards Unified Representation in Step-Imbalanced Class-Incremental Learning

**arXiv ID:** 2603.10237 | [PDF](https://arxiv.org/pdf/2603.10237v1)

**作者:** Xiaoyan Zhang `[一作]` (University of Michigan), Jiangpeng He `[通讯]` (Indiana University)

**通讯引用:** 611 | [OpenAlex ID](https://openalex.org/A5063620170)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种单适配器的统一框架One-A，用于处理类数不平衡的增量学习。

**💡 创新点**

创新点在于异步子空间对齐、信息自适应加权和方向门控，能在保持大任务主子空间的同时注入小任务信息。

**🔧 技术方法**

使用异步SVD对齐、MLP适配器、ViT骨干、原型分类器、对比损失以及信息自适应权重和方向门控。

**📊 数据集**

在CIFAR‑100、CUB‑200、ImageNet‑A、ImageNet‑R等基准上进行实验。

**📈 对比分析**

与多种基线（L2P、DualPrompt、EASE、CL‑LoRA等）比较，One‑A在步长不平衡场景下取得最高的最后步和平均准确率，同时仅使用单一适配器实现低推理成本。

**⚠️ 局限性**

局限在于仍假设任务大小与类数相近，难以处理样本级不平衡或极端分布漂移，且对门控参数敏感。

---

## 111. Regime-aware financial volatility forecasting via in-context learning

**arXiv ID:** 2603.10299 | [PDF](https://arxiv.org/pdf/2603.10299v1)

**作者:** Saba Asaad `[一作]` (University of Toronto), Ali Bereyhi `[通讯]` (University of Toronto)

**通讯引用:** 317 | [OpenAlex ID](https://openalex.org/A5061064331)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于预训练大型语言模型的情境学习框架，用于在非平稳市场条件下预测金融波动率。

**💡 创新点**

创新点在于使用oracle引导的迭代细化构造具有波动率区间标签的示例池，并通过条件采样让LLM在推理时自适应不同的市场 regime，从而在不微调参数的情况下实现对波动率动态的快速适配。

**🔧 技术方法**

核心技术包括大语言模型提示式推理、oracle-guided refinement loop、区间化的演示池、基于波动率估计的条件采样以及无参数微调的情境学习。

**📊 数据集**

实验数据集包括美国标普500指数、纳斯达克综合指数和欧元兑美元汇率的每日收盘价格，构建了一日递推波动率预测任务。

**📈 对比分析**

与滚动平均、HAR、GARCH、GJR-GARCH、一次性prompt等基线方法在MAE和RMSE指标上进行比较，结果显示在高波动期误差降低约27%，整体MAE和RMSE均优于传统模型。

**⚠️ 局限性**

局限性包括：在低波动期误差略有上升；示例池构建依赖训练集阈值和oracle反馈，需额外标注；对非金融或极端异常情境的泛化能力尚未验证。

---

## 112. Characterizing Healthy & Post-Stroke Neuromotor Behavior During 6D Upper-Limb Isometric Gaming: Implications for Design of End-Effector Rehabilitation Robot Interfaces

**arXiv ID:** 2603.10173 | [PDF](https://arxiv.org/pdf/2603.10173v1)

**作者:** Ajay Anand `[一作]`, Laura A. Hallock `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究利用6D上肢等速游戏测量健康与中风受试者在7种力生产任务中的神经运动行为，并评估其对末端执行器康复机器人界面设计的启示。

**💡 创新点**

首次将游戏化等速运动的力轨迹与理想力模型结合，使用隐藏马尔可夫模型对子任务进行自动识别，并探讨受试者力误差与运动协同的关联。

**🔧 技术方法**

采用理想力计算、k‑means聚类、HMM子任务分类、RMSE、力冲击量、平均与峰值力等多种技术。

**📊 数据集**

使用来自健康与中风受试者的6D等速游戏实验数据，共7种力轨迹任务，受试者编号02–22。

**📈 对比分析**

与传统统计分析相比，HMM分类误差显著低于随机水平，力误差分析显示中风组RMSE和平均力显著升高，但聚类未能形成有意义的受试者群组。

**⚠️ 局限性**

局限包括聚类方法无法捕捉协同差异、数据量有限、仅评估等速任务且未考虑动态运动的外推性。

---

## 113. Design of a Robot-Assisted Chemical Dialysis System

**arXiv ID:** 2603.10264 | [PDF](https://arxiv.org/pdf/2603.10264v1)

**作者:** Diane Jung `[一作]` (University of Colorado at Boulder), Carson Bruns `[通讯]` (University of Colorado at Boulder)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

通过用户中心设计方法，基于Franka Research 3协作机器人开发并迭代改进了一套半自动化的化学透析系统，包含自定义时间间隔、并行透析、游戏手柄与姿态导向控制以及实时可视化GUI。

**💡 创新点**

创新点在于：①将通用协作机器人引入实验室透析流程，实现低至中等自主度的人机协作；②支持并行与顺序透析，提升实验吞吐量；③将手动设置与机器人执行分离，保留实验者对关键参数的掌控；④通过可视化界面实时显示容器位置、机器人状态与倒计时，提升操作透明度。

**🔧 技术方法**

所用技术包括：Franka Research 3协作机器人与末端执行器；游戏手柄与姿态导向的手动引导；基于LORA 3-4的低至中等自主度框架；力/扭矩监测安全校验；3D打印透析膜支架；自定义GUI与实时3D渲染；用户体验采集与主题分析。

**📊 数据集**

未使用公开数据集；实验仅在干实验室环境下进行，采用4L Nalgene容器、模拟透析膜（1:1番茄汤：水）以及水作为缓冲液进行验证。

**📈 对比分析**

通过两轮可用性研究（共5名研究人员），采用访谈、思考大声法、主题分析等定性方法收集反馈，迭代改进系统；并未给出数值化性能指标，仅报告了功能提升与用户满意度提升。

**⚠️ 局限性**

局限性包括：①验证仅在干实验室完成，未在真实湿实验室环境中评估净化效果和安全性；②参与者数量有限，缺乏统计显著性；③系统仅针对透析流程，未验证对其他实验步骤的适用性；④未对机器人自主规划或感知能力进行深入研究；⑤安全与故障恢复机制仍需进一步完善。

---

## 114. Re-Evaluating EVMBench: Are AI Agents Ready for Smart Contract Security?

**arXiv ID:** 2603.10795 | [PDF](https://arxiv.org/pdf/2603.10795v1)

**作者:** Chaoyuan Peng `[一作]` (Zhejiang University), Yajin Zhou `[通讯]` (BlockSec)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 AI 代理在以太坊智能合约安全审计中的表现进行了系统评估，扩充了模型配置、跨工具（scaffold）比较，并构建了一个无污染的真实事件数据集，评测了检测（Detect）和攻击（Exploit）两项任务。

**💡 创新点**

创新点在于：
- 将 EVMBench 扩展到 26 种模型+scaffold 配置，系统交叉评估模型与工具的独立影响；
- 设计了 Incidents 数据集，包含 22 条 2026 年 2 月后发生的真实攻击，消除了训练集泄漏；
- 对模型推理努力级别、工具效应和评判器可靠性进行细粒度实验，揭示模型排名不稳定和 scaffold 影响显著；
- 在同一实验框架下同时比较检测 Recall 与攻击成功率，展现两项任务的互补性与差异。

**🔧 技术方法**

主要技术手段包括：
- 大语言模型代理（Claude、GPT、Gemini、GLM‑5）与三种 scaffold（Claude Code、Codex CLI、OpenCode）的组合；
- 通过 OpenRouter 接口调用模型；
- 基于模型的判定器（judge）评估检测报告的 Recall；
- 在隔离的 Docker 容器中 fork 以太坊链快照，模拟攻击并验证链上状态变更；
- 对模型推理努力（low/medium/high/xhigh）和多代理设置进行实验。

**📊 数据集**

使用的数据集有：
- EVMBench：120 个漏洞，来自 40 个 Code4rena 审计仓库；
- Incidents：22 条 2026 年 2 月后发生的真实安全事件，所有漏洞均在模型发布后出现，保证无训练集污染。

**📈 对比分析**

比较方法与性能：
- 采用 Recall 作为检测指标，攻击成功率（按净收益判定）作为 Exploit 评价；
- 最佳代理在 EVMBench 检测 Recall 为 47.5%，攻击成功率为 61.1%；
- 在 Incidents 数据集检测 Recall 为 65%（最高），但攻击成功率为 0%（所有 110 对中未出现盈利攻击）；
- scaffold 对检测分数影响可达 5pp，模型排名在不同任务/数据集之间波动显著；
- GPT-5.2 在不同推理努力下表现出逆向规模效应，说明更多推理并不总能提升效果。

**⚠️ 局限性**

局限性包括：
- 仅在单次实验中评估每个配置，缺乏置信区间和随机种子方差；
- scaffold 交叉实验仅覆盖 3 个模型；
- 评判器仅检验 Recall，未惩罚假阳性，无法衡量精度；
- 受 OpenRouter 代理的影响，模型表现可能与官方 API 存在差异；
- Incidents 数据集规模有限（22 条），统计功效受限；
- 对漏洞描述可能存在错误或不完整，影响评分；
- 只评估 Detect 与 Exploit，未全面涵盖 Patch 与更细粒度的安全分析。

---

## 115. HanMoVLM: Large Vision-Language Models for Professional Artistic Painting Evaluation

**arXiv ID:** 2603.10814 | [PDF](https://arxiv.org/pdf/2603.10814v1)

**作者:** Hongji Yang `[一作]` (University of Macau), Jianbing Shen `[通讯]` (University of Macau)

**通讯引用:** 16344 | [OpenAlex ID](https://openalex.org/A5023184215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型视觉语言模型在中国传统绘画专业评价上的能力，并提出了专家级链式推理与奖励机制，实现对绘画的专业评分与生成器的质量筛选。

**💡 创新点**

①构建专门的 HanMo-Bench 数据集；②设计面向中国绘画的专家级 Chain-of-Thought 结构；③引入基于 CoT 的多项奖励函数进行强化微调；④把 VLM 作为 Test-time Scaling 的外部验证器提升图像生成质量。

**🔧 技术方法**

使用大型 VLM（Qwen3-VL-Instruct）+ LoRA 微调 + 组策略优化（GRPO）强化学习 + BERTScore、IoU 等奖励函数，结合结构化的 Chain-of-Thought 推理。

**📊 数据集**

HanMo-Bench，包含 13k 幅中国画，分为真迹与 AI 生成作品，带有真实拍卖价值标签和专家级 CoT 说明。

**📈 对比分析**

与 Gemini、GPT、InternVL 等公开模型对比，HanMoVLM 在 MAE、RMSE、准确率、BERTScore、mIoU 等指标显著提升；在人类专家排名中 Kendall τ=0.758，Spearman ρ=0.845，验证与专家一致性；在 Test-time Scaling 下提升 T2I 生成质量。

**⚠️ 局限性**

依赖人工专家验证和高质量标注；对非中国绘画领域的通用性有限；RoI 预测 mIoU 较低，局部细节定位仍不够精准；模型对极小目标的识别受限。

---

## 116. $V_{0.5}$: Generalist Value Model as a Prior for Sparse RL Rollouts

**arXiv ID:** 2603.10848 | [PDF](https://arxiv.org/pdf/2603.10848v1)

**作者:** Yi-Kai Zhang `[一作]` (Peking University), Han-Jia Ye `[通讯]` (Peking University)

**通讯引用:** 2104 | [OpenAlex ID](https://openalex.org/A5075981348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 V_0.5 框架，融合通用价值模型 V_0 与稀疏 rollout，构建自适应基线估计与动态预算分配。

**💡 创新点**

创新点在于经验收缩融合与实时假设检验的 Sequential OSLA 动态停机规则，将 V_0 作为安全的统计先验，实现低方差与低偏差的平衡。

**🔧 技术方法**

技术包括强化学习策略梯度、V_0 通用价值模型、经验收缩融合、One-Step-Look-Ahead（OSLA）动态预算、TabPFN 推断、与 PPO/GRPO 对比分析。

**📊 数据集**

训练 V_0 采用约 424k 个 Qwen 系列模型生成的样本；评估使用六个数学推理基准：AIME 2024/2025、Olympiad Bench、MATH500、Minerva Math、AMC 2023。

**📈 对比分析**

在相同计算预算下与 GRPO、DAPO 对比，V_0.5 在所有基准上均提升 10% 以上精度，收敛更快，梯度范数更稳定。

**⚠️ 局限性**

局限性包括：对极端稀疏（组大小 1、2）仍难以收敛；依赖先验偏差控制需假设检验阈值，且在 OOD 场景中的泛化仍有限。

---

## 117. Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation

**arXiv ID:** 2603.11045 | [PDF](https://arxiv.org/pdf/2603.11045v1)

**作者:** Tao Zhong `[一作]` (Princeton University), Christine Allen-Blanchette `[通讯]` (Princeton University)

**通讯引用:** 591 | [OpenAlex ID](https://openalex.org/A5091851960)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出NeFTY框架，实现利用差分可微物理与隐式神经场进行三维热导率反演

**💡 创新点**

通过将热扩散方程作为硬约束嵌入数值求解器，解决PINN在瞬态扩散中的梯度僵化与谱偏差

**🔧 技术方法**

使用位置编码+频率退火的多层感知机、隐式神经场、差分可微热学求解器与伴随梯度优化

**📊 数据集**

采用基于PhiFlow的高精度显式有限体积模拟生成的合成热流数据，包含均匀与层状复合材料以及多尺度缺陷

**📈 对比分析**

与网格优化、PINN、无监督U‑Net、监督U‑Net等基线对比，NeFTY在MSE/PSNR/IoU等指标上优于所有无监督方法，接近监督上限

**⚠️ 局限性**

依赖高质量合成数据；对真实实验的鲁棒性未验证；对大尺度、复杂材料结构的可扩展性仍需进一步研究

---

## 118. GLM-OCR Technical Report

**arXiv ID:** 2603.10910 | [PDF](https://arxiv.org/pdf/2603.10910v1)

**作者:** Shuaiqi Duan `[一作]` (Zhipu AI), Jie Tang `[通讯]` (Tsinghua University)

**通讯引用:** 28952 | [OpenAlex ID](https://openalex.org/A5044791875)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种0.9B参数的多模态OCR模型GLM-OCR，能够高效完成文档解析、表格恢复、公式识别及关键信息提取；

**💡 创新点**

核心创新在于将布局分析与并行区域识别结合的两阶段管道，以及在解码中引入多Token预测（MTP）机制，以显著提升吞吐量和结构一致性；

**🔧 技术方法**

采用CogViT视觉编码器、GLM语言解码器、MTP解码技术、PP-DocLayout-V3布局模块，并通过vLLM/SGLang/Ollama等框架实现高效推理；

**📊 数据集**

训练数据包括大规模图文对、文档解析、视觉检索、VQA等多种任务，并在公开基准（OmniDocBench、OCRBench、UniMERNet、PubTabNet、TEDS）以及内部工业数据集上评估；

**📈 对比分析**

与传统管线工具、通用VLM和专用OCR VLM对比，GLM-OCR在OmniDocBench 94.6分、OCRBench 94.0分、UniMERNet 96.5分、PubTabNet 85.2分、TEDS 86.0分等指标上均达成或超过state‑of‑the‑art，KIE任务上亦实现93.7/86.1分，展示出在小模型规模下的高性能；

**⚠️ 局限性**

存在布局误检导致的误差传播、对极低分辨率或高度复杂布局的鲁棒性不足、生成结构格式的随机性、以及对提示语义和JSON schema的依赖等限制。

---

## 119. DeliberationBench: A Normative Benchmark for the Influence of Large Language Models on Users' Views

**arXiv ID:** 2603.10018 | [PDF](https://arxiv.org/pdf/2603.10018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 120. MUNIChus: Multilingual News Image Captioning Benchmark

**arXiv ID:** 2603.10613 | [PDF](https://arxiv.org/pdf/2603.10613v1)

**作者:** Yuji Chen `[一作]` (Lancaster University), Tharindu Ranasinghe `[通讯]` (Lancaster University)

**通讯引用:** 774 | [OpenAlex ID](https://openalex.org/A5061000186)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了首个多语言新闻图像描述基准MUNIChus，包含9种语言（含低资源语言）的145k+训练图像和8.9k测试图像，并提供对应新闻文章、标题和图像描述。

**💡 创新点**

创新点在于：①首次在多语言新闻领域构建大型图像描述数据集；②将新闻文本与图像结合，生成语义丰富且具实体信息的新闻式描述；③通过对比传统图像描述模型与多模态大模型，揭示了新闻图像描述的独特挑战；④系统评估多种提示与指令微调策略，指出微调才是提升性能的有效路径。

**🔧 技术方法**

技术手段包括：多模态大型语言模型（CohereLabs 8B、Llama‑3.2 11B）进行零/随机/相似三种提示；指令微调采用QLoRA+LoRA；传统模型如BLIP+NLLB、M3L 作为基线；评估使用BLEU‑4和CIDEr；对中文和日文做分词；训练采用4‑bit量化和LoRA配置。

**📊 数据集**

数据集：MUNIChus，来源于BBC新闻，涵盖9种语言（阿拉伯语、中文、英语、法语、印地语、日语、印尼语、僧伽罗语、乌尔都语），包含图像、新闻文本、标题和对应的新闻式描述；训练集145,314张图像，测试集8,993张。

**📈 对比分析**

比较方法：在每种语言上对20+模型（零/随机/相似提示、指令微调模型、传统BLIP+NLLB、M3L）进行BLEU‑4和CIDEr评估。结果显示：指令微调模型显著优于提示策略，BLEU‑4≈8.4、CIDEr≈56；传统模型BLEU<0.7；提示策略均在2–4范围；多模态模型在高资源语言表现好，但低资源语言（如僧伽罗）仍显著落后。

**⚠️ 局限性**

局限性：①低资源语言（尤其僧伽罗）性能仍低，难以通过微调弥补预训练缺失；②模型尺寸不一定决定性能，需结合任务特定微调；③提示策略在新闻图像描述中效果有限；④数据集仅来自BBC，可能带有新闻偏见，限制了泛化能力。

---

## 121. Learning Adaptive Force Control for Contact-Rich Sample Scraping with Heterogeneous Materials

**arXiv ID:** 2603.10979 | [PDF](https://arxiv.org/pdf/2603.10979v1)

**作者:** Cenk Cetin `[一作]` (University of Liverpool), Gabriella Pizzuto `[通讯]` (University of Liverpool)

**通讯引用:** 355 | [OpenAlex ID](https://openalex.org/A5039013002)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发并实现了一个基于卡氏阻抗控制器与强化学习的自适应力控制框架，用于化学实验室中异质材料的样本刮除任务。

**💡 创新点**

创新点在于让RL直接输出期望的卡氏力矩，实现对未知材料属性的实时适配，并结合多模态视觉感知提供材料位置反馈，形成低层阻抗控制与高层策略的层级协同。

**🔧 技术方法**

使用的技术包括Cartesian impedance controller、PPO强化学习、RGB‑D视觉感知（YOLOv8 + GrabCut + K‑means）、MuJoCo仿真、Perlin噪声随机化与Zero‑shot sim‑to‑real transfer。

**📊 数据集**

使用了自制的1,084张标注样本瓶图像训练YOLOv8，仿真中随机生成数百个球体模拟材料，实验中采用五种不同物料（液体面团、液体玉米粉、干玉米粉、晶体盐、晶体糖）。

**📈 对比分析**

与固定4 N力矩基线对比，RL方法在所有材料上平均相对成功率提升至75.3%（基线为64.4%），并在晶体材料上逼近人工水平，展示了显著的性能提升。

**⚠️ 局限性**

局限性包括对高粘度、极粘性材料（如液体面团）刮除仍不理想；视觉对光照与反射的鲁棒性受限；工具积累与形变的影响尚未全面评估；仿真对软材料动力学的建模仍相对简化。

---

## 122. Agentar-Fin-OCR

**arXiv ID:** 2603.11044 | [PDF](https://arxiv.org/pdf/2603.11044v1)

**作者:** Siyi Qian `[一作]` (Ant Group), Peng Zhang `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计了 Agentar‑Fin‑OCR 系统，实现跨页内容整合、标题层级重建、精确表格解析并提供可视化引用。

**💡 创新点**

在文档级解析中引入跨页内容合并、标题层级重建伪 TOC+VLM、基于结构锚点的 CellBBoxRegressor，以及专属的 FinDocBench benchmark，突破传统单页处理与缺乏可视化引用的局限。

**🔧 技术方法**

采用多模态 VLM（Fin‑OCR）+语义拼接+结构锚点回归、跨页表格启发式合并、curriculum learning 与 RL（GRPO）训练、以及 TEDS、TocEDS、C‑IoU 等新指标。

**📊 数据集**

FinDocBench（6类金融文档、跨页表格、标题层级）与 OmniDocBench v1.5 作为基准数据集。

**📈 对比分析**

与多种 VLM、OCR 在 OmniDocBench 上进行对比，表格 TEDS 92.82/95.88 位居前列；在 FinDocBench 的布局、读取顺序、标题层级、跨页表格等任务上均优于基线，TocEDS 与 TEDS 提升 7‑18%。

**⚠️ 局限性**

对短文档的提升有限，跨页表格合并仍依赖启发式；缺乏多语言支持和实时推理速度评估。

---

## 123. Leech Lattice Vector Quantization for Efficient LLM Compression

**arXiv ID:** 2603.11021 | [PDF](https://arxiv.org/pdf/2603.11021v1)

**作者:** Tycho F. A. van der Ouderaa `[一作]` (Qualcomm AI Research), Markus Nagel `[通讯]` (Qualcomm AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于Leech晶格的向量量化方法（LLVQ），用于将大型语言模型压缩至极低位宽（如2比特/权重），实现了无代码本量、可逆索引和高效解量化。

**💡 创新点**

创新点包括：①在多 shell 结构上扩展晶格搜索，支持欧氏与角度两种评分；②构建全可逆索引方案，将晶格点映射为唯一整数/位串；③利用 Golay 码与Leech 晶格的对称性实现并行化解量化内核；④在高维（24D）下实现形增量/球面量化，显著提升压缩性能。

**🔧 技术方法**

技术手段主要有：高维晶格量化（Leech晶格与扩展Golay码构造）、多 shell 搜索（欧氏与角度距离），可逆索引与位串映射，GPU 并行解量化核，Hadamard 旋转与轻量级微调（仅行/列缩放）。

**📊 数据集**

使用的基准数据集包括：Llama‑2、Llama‑3、Ministral‑3、Qwen‑v3 语言模型；对量化效果评估采用 WikiText‑2 perplexity、MMLU、CSR 等下游任务；对理想高斯源的 SQNR 与保留率做定量比较。

**📈 对比分析**

与现有方法（GPTQ、Quarot、Quip#、E8P、QTIP、PV‑tuning 等）在统一 PTQ 管线下对比，LLVQ 在 2‑bit 率下在 WikiText‑2 perplexity、MMLU、CSR 上均优于竞争者；在高斯源上达到 92%+ 的 Shannon 限制保留率，明显领先。

**⚠️ 局限性**

局限性包括：仅在 24 维晶格上实现，扩展到更大维度需新的晶格；对非高斯或高度非均匀权重分布的鲁棒性未完全验证；实现复杂度相对传统 scalar 量化更高，部署时需额外实现索引/解码逻辑。

---

## 124. How to make the most of your masked language model for protein engineering

**arXiv ID:** 2603.10302 | [PDF](https://arxiv.org/pdf/2603.10302v1)

**作者:** Calvin McCarter `[一作]` (BigHat Biosciences), Hunter Elliott `[通讯]` (BigHat Biosciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并系统评估了面向抗体工程的 MLM 采样方法，重点通过随机束搜索（stochastic beam search）在已给定种子序列附近生成多样化、功能优异的抗体变体，并在真实的抗体研发项目中进行体外验证。

**💡 创新点**

创新点主要包括：① 将 MLM 的 pseudo‑log‑likelihood（PLL）作为全序列评估指标，利用野生型边缘近似快速计算一编辑邻域的 PLL；② 引入随机束搜索加 Gumbel 噪声的序列中心搜索框架，显著提升采样效率与质量；③ 采用多目标无梯度引导（NDS、STS）与监督模型相结合，兼顾多项生物学指标；④ 对比传统基于变异的 Gibbs 采样，提供了系统的体内/体外基准实验。

**🔧 技术方法**

核心技术包括：MLM（ESM‑2、AbLang‑2 等）、CLM（pIgGen、CloneLM）、随机束搜索、Gumbel 噪声、wild‑type marginal 近似、无梯度多目标优化（NDS、STS）、监督可合成性与结合成功率分类器。

**📊 数据集**

使用的数据集包括：真实的 scFv 抗体研发实验数据（含可合成性、结合亲和力、热稳定性等监督指标）、OASis humanness 百分位、等电点；以及在实验室验证的 FAb 抗体候选序列集合。

**📈 对比分析**

方法比较：在体内模拟实验中，ESM‑2‑650M 与 AbLang‑2 在多项指标（可合成性、结合亲和力、热稳定性）上优于其他模型；随机束搜索在所有模型中都优于 Gibbs 采样；在体外实验中，加入监督引导与多目标优化后，随机束搜索+STS 甚至实现 100% 成功率。相对传统 Gibbs 采样，随机束搜索在速度和质量上都有显著提升。

**⚠️ 局限性**

局限性包括：① 需要对 PLL 进行昂贵的前向传播（O(L³)），虽然近似提升了效率但仍非实时；② 监督引导可能导致人源性（humanness）下降或过度偏向特定 germline；③ 随机束搜索在种子间多样性方面仍有限；④ 对于极大变异量或结构约束的场景，方法的适用性尚未充分验证。

---

## 125. Training-Free Multi-Step Inference for Target Speaker Extraction

**arXiv ID:** 2603.10921 | [PDF](https://arxiv.org/pdf/2603.10921v1)

**作者:** Zhenghai You `[一作]` (Beijing University of Posts and Telecommunications), Dong Wang `[通讯]` (Center for Speech and Language Technologies)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种训练无关的多步推理框架，利用冻结的目标说话人提取（TSE）模型通过在混合信号与前一步估计之间进行线性插值生成候选输入，并在每一步选取评分最高的候选进行迭代，从而在不更新模型参数的情况下提升提取质量。

**💡 创新点**

创新点在于：①将插值候选搜索与非侵入式联合评分（UTMOS+SpkSim）相结合，使推理阶段可同时优化感知质量与说话人一致性；②提供理论分析保证贪婪选择的非退化性及误差界限；③展示了在多种TSE骨干上可获得的推理阶段性能提升。

**🔧 技术方法**

使用的技术包括：预训练的TSE模型（如DPRNN、SpEx+）；线性插值生成K个候选输入；基于SI‑SDRi、UTMOS或SpkSim的评分器；联合评分公式 R_joint(ŝ;e)=UTMOS(ŝ)+λ(1−exp(−α·SpkSim(ŝ,e)))；以及多步迭代推理流程。

**📊 数据集**

实验采用Libri2Mix数据集（从LibriSpeech构建的两人单声道混合），在train‑100子集上训练模型，在test‑clean子集上评估。

**📈 对比分析**

与一次推理基线相比，oracle SI‑SDRi选择可使SI‑SDRi提升约0.9 dB（DPRNN）或0.7 dB（SpEx+）；使用UTMOS或SpkSim单指标时，分别显著提升对应指标；联合评分在保持感知质量（UTMOS）不下降的同时提升说话人相似度（SpkSim），但整体性能略低于oracle上限。

**⚠️ 局限性**

局限性包括：对非侵入式评分器的估计误差和域迁移敏感；单指标优化导致偏置，难以同时提升所有目标指标；多步搜索需要额外的计算开销，尚未完全逼近oracle性能。

---

## 126. From Images to Words: Efficient Cross-Modal Knowledge Distillation to Language Models from Black-box Teachers

**arXiv ID:** 2603.10877 | [PDF](https://arxiv.org/pdf/2603.10877v1)

**作者:** Ayan Sengupta `[一作]` (Indian Institute of Technology), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology)

**通讯引用:** 5082 | [OpenAlex ID](https://openalex.org/A5046521217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种跨模态知识蒸馏框架，能够从大规模视觉-语言模型向纯文本模型传递知识。

**💡 创新点**

创新点在于提出了可无预训练的跨模态对齐模块，支持在黑盒视觉-语言教师上直接蒸馏，且参数增量仅0.8%。

**🔧 技术方法**

采用了对齐模块（非线性投影、曼哈顿/欧氏/余弦对齐、辅助输出头）以及三阶段蒸馏（输出、对齐、辅助对齐）实现跨模态知识迁移。

**📊 数据集**

使用Stable Diffusion、Midjourney等文本-图像模型作为教师，并在GLUE、SuperGLUE、MM-IMDb、Hateful Memes等多种NLP与多模任务数据集上进行实验。

**📈 对比分析**

与10种单模态和3种多模态蒸馏基线对比，平均提升3.4%（NLU任务）和2.6%（推理任务），大模型如DeBERTa、OPT、LLaMA均显著受益。

**⚠️ 局限性**

限制在于可能会将视觉模型的偏见迁移至文本模型，对鲁棒性与公平性仍需进一步研究，且在低质量教师表示下效果明显下降。

---

## 127. DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control

**arXiv ID:** 2603.10448 | [PDF](https://arxiv.org/pdf/2603.10448v1)

**作者:** Teli Ma `[一作]` (Mondo Robotics), Shuo Yang `[通讯]` (Mondo Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种端到端的视频-动作模型 DiT4DiT，利用视频扩散Transformer产生的中间去噪特征来指导动作扩散Transformer，从而实现机器人控制。

**💡 创新点**

创新点包括：1）将视频生成过程的中间隐藏特征作为动作条件，避免完整视频重建；2）采用双流匹配（dual flow‑matching）目标，在不同时间步和噪声尺度下联合训练视频与动作模块；3）实现了高效的三时钟设计（视频、特征提取、动作）以保持训练与推理的同步。

**🔧 技术方法**

核心技术包括视频扩散Transformer（DiT）、动作扩散Transformer、流匹配优化、前向钩子（hook）提取隐藏特征、Beta采样时间步、Euler数值积分、跨模态交叉注意力。

**📊 数据集**

使用的数据集：LIBERO（10类任务共500轨迹）、RoboCasa‑GR1（24类任务共1000轨迹）、Unitree G1 真实机器人（7类任务共200轨迹）。

**📈 对比分析**

与多种主流 VLA 基线（GR00T、CogVLA、UniVLA、π_0.5、OpenVLA‑OFT 等）以及参数匹配的 Qwen3DiT 进行对比。DiT4DiT 在 LIBERO 上平均成功率 98.6%（最高），在 RoboCasa‑GR1 上 50.8%（超过 GR00T‑N1.5 10%），在 Unitree G1 真实任务上多项任务显著优于对手，且样本效率提升 >10×，收敛速度加快 7×，具备零样本泛化能力。

**⚠️ 局限性**

局限性包括：1）推理频率仅 6Hz，低于某些基线；2）依赖高成本的视频扩散预训练，资源开销大；3）在极端视觉或动力学变化下仍可能出现性能下降；4）实验集中于单目 egocentric 视角，未验证多摄像头或全景场景；5）模型在更大范围多任务或长时序任务的可扩展性尚待验证。

---

## 128. The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training

**arXiv ID:** 2603.10444 | [PDF](https://arxiv.org/pdf/2603.10444v1)

**作者:** Hengjie Cao `[一作]` (Fudan University), Li Shang `[通讯]` (Fudan University)

**通讯引用:** 6446 | [OpenAlex ID](https://openalex.org/A5004722925)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在低位数训练时因表示的异向性导致的数值不稳定，并提出通过去除一阶均值偏差来消除主导不稳定因素。

**💡 创新点**

发现低位数训练中动态范围膨胀主要由一致的秩一均值偏差驱动，并提出仅使用均值减法的简单条件化方法，几乎等价于SVD谱方法，却无需复杂矩阵分解。

**🔧 技术方法**

使用谱分析评估表示异向性、均值偏差贡献、低位数量化（FP4/W4A4G4）训练实验以及标准量化算子。

**📊 数据集**

在常见大规模预训练语料上进行实验，典型的数据集包括 C4、Wikipedia 等通用文本数据。

**📈 对比分析**

将均值去除方案与 BF16 以及基于 SVD 的方法进行对比，实验表明在 FP4 训练中均值去除显著缩小损失差距并恢复了下游任务的性能。

**⚠️ 局限性**

仅针对秩一均值偏差，可能无法完全消除所有低位数训练中的不稳定因素；实现仍需额外的求均值操作，且在极端低位数或不同模型架构下的效果尚需进一步验证。

---

## 129. Interleaving Scheduling and Motion Planning with Incremental Learning of Symbolic Space-Time Motion Abstractions

**arXiv ID:** 2603.10651 | [PDF](https://arxiv.org/pdf/2603.10651v1)

**作者:** Elisa Tosello `[一作]` (Fondazione Bruno Kessler), Andrea Micheli `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 11748 | [OpenAlex ID](https://openalex.org/A5114375899)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Scheduling and Motion Planning (SAMP) 问题的形式化定义，并给出了一个通过交替使用现成调度器和运动规划器的增量学习框架。

**💡 创新点**

创新点在于将调度器产生的时序计划与运动规划器反馈的空间时间约束相结合，利用几何和时间精炼不断收敛至可行的同步执行计划。

**🔧 技术方法**

所采用的技术包括统一规划库中的Aries、OR‑Tools CPSE 等调度器，OMPL 中的 RRT 和 ST‑RRT* 运动规划器，以及层次化的单动作与组动作检验与精炼机制。

**📊 数据集**

实验使用了扩展后的物流（多机器人货物搬运）和作业车间（JSP）基准，所有实例都在二维空间的车道与门控环境中生成。

**📈 对比分析**

与单纯顺序管线或无精炼的调度+运动规划组合相比，框架在平均 91% 的实例可解、使总规划时间中运动规划占比下降至约 70%，且在可并行化场景下使完工时间平均下降约 41%。

**⚠️ 局限性**

局限性包括对运动规划器完整性的假设、在大规模多机器人场景下仍会出现高昂的时间与精炼次数，以及对动态障碍或非平面环境的适应性尚待进一步验证。

---

## 130. Repurposing Backdoors for Good: Ephemeral Intrinsic Proofs for Verifiable Aggregation in Cross-silo Federated Learning

**arXiv ID:** 2603.10692 | [PDF](https://arxiv.org/pdf/2603.10692v1)

**作者:** Xian Qin `[一作]` (Southwest Jiaotong University), Xiaohu Tang `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 6140 | [OpenAlex ID](https://openalex.org/A5029840191)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级的可验证聚合框架，通过在模型参数中嵌入短暂的内在证明，替代传统重型加密实现跨机联邦学习中的聚合完整性审计。

**💡 创新点**

创新点在于把后门注入转化为可消亡的验证信号，利用灾难性遗忘保证信号在一次聚合后迅速消失，同时采用随机单一匿名验证者的调度，彻底消除了额外通信与计算开销。

**🔧 技术方法**

使用了后门触发器、灾难性遗忘机制、随机单一验证者调度、Secure Aggregation兼容性、以及梯度提升的轻量化实现。

**📊 数据集**

实验数据集包括SVHN（MobileNetV1）、CIFAR-10（ResNet-20）和CIFAR-100（ResNet-18）。

**📈 对比分析**

与FedAvg、LightVeriFL以及基于双服务器的加密方案对比，在保持近似精度的同时实现了99~1800倍的速度提升，且不增加通信成本，验证检测率可达接近100%。

**⚠️ 局限性**

限制在于需预先生成私有触发器，对随机验证者的安全性假设敏感；在极端数据不平衡或极少客户端的场景下验证效果可能受限。

---

## 131. From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning

**arXiv ID:** 2603.10263 | [PDF](https://arxiv.org/pdf/2603.10263v1)

**作者:** Zhanyi Sun `[一作]` (Stanford University), Shuran Song `[通讯]` (Stanford University)

**通讯引用:** 24866 | [OpenAlex ID](https://openalex.org/A5004644695)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Distribution Contractive RL Finetuning（DICE-RL）框架，将预训练的生成行为克隆（BC）策略作为基础，冻结其参数后在其上学习轻量化残差，并通过 BC 正则化、选择性正则、值引导动作选择等机制实现稀疏奖励长周期操纵任务的高效、稳定的在线微调。

**💡 创新点**

创新点包括：① 将 RL 视为对行为分布的“收缩”操作；② 采用冻结的生成 BC 作为结构化探索基础，配合残差微调实现可控探索；③ 引入 BC 损失过滤和值引导最佳动作选取，保证更新既保留提升又避免过度偏离；④ 结合多样本期望训练、动作块化等技术显著提升样本效率和稳定性。

**🔧 技术方法**

主要技术：扩散/流匹配生成模型、残差策略、TD3+BC 损失、BC 正则化、RLPD 混合策略、动作块化（h-step 采样）、多样本期望训练、值引导最佳动作选择、BC 损失过滤、经验回放。

**📊 数据集**

使用的数据集：Robomimic 任务（pullout、push_block、slide_object、pick_and_place）演示集（20/50 条演示），以及真实机器人上的 NIST 基准（gear insertion、bulb insertion、belt threading）演示集（40/100/265 条轨迹），所有演示均来自 kinesthetic 录制或人类演示。

**📈 对比分析**

与 IBRL、DPPO、EXPO、DSRL、ResFit 等基线在状态和像素观测下进行比较；在 Robomimic 任务中，DICE-RL 在约 2000 次在线交互内突破 90% 成功率，并且在所有任务上均表现出更高的样本效率和更稳定的学习曲线；在真实机器人三项任务中亦达 90%+ 成功率，证明了方法的实用性。

**⚠️ 局限性**

局限性：1）依赖离线演示质量与规模，演示不足时微调效果受限；2）对分布漂移或样本外情况的鲁棒性有限；3）需要手动设置 BC 正则过滤阈值、RLPD 混合比例等超参数；4）在更大规模多任务或更复杂环境下的可扩展性仍待验证。

---

## 132. LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation

**arXiv ID:** 2603.10899 | [PDF](https://arxiv.org/pdf/2603.10899v1)

**作者:** Jinwoo Ahn `[一作]` (Samsung Research), Yongkweon Jeon `[通讯]` (Samsung Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于轻量化学习模块的 KV 缓存淘汰框架，利用可学习的 Lookahead Token 和 LoRA 模块在预填阶段预测真实响应的重要性得分，从而高效剔除冗余 KV 条目。

**💡 创新点**

创新点在于不需要显式生成草稿响应，而是通过学习的特殊 token 与低秩适配器直接“窥视”未来注意力模式，实现与草稿方法同等甚至更优的淘汰质量，同时显著降低计算开销。

**🔧 技术方法**

主要技术包括：可学习的 Lookahead Token、选择性激活的 Lookahead LoRA、基于 KL 散度的注意力分数蒸馏训练、FlashAttention 加速前向与梯度传播，以及在预填阶段的 KV 剪枝策略。

**📊 数据集**

使用了多种长文本与指令数据集进行训练：ChatQA2 长文档子集（5万条）、Tulu 指令集（2万条）、Stack 语料（7千条）以及 MetaMath、ARC、HellaSwag 的少样本生成数据；评估则覆盖 LongBench、RULER、LongProc、MT‑Bench 等长上下文基准。

**📈 对比分析**

与 SnapKV、PyramidKV、StreamingLLM 等简单启发式方法以及 LAQ、SpecKV 等草稿生成方法对比，实验显示其在所有模型和预算设置下均优于基线，特别是在低预算场景中表现突出，同时在 32K 上的淘汰开销低于 LAQ 14.5 倍，时间到首标记（TTFT）提升显著。

**⚠️ 局限性**

局限性包括：仅针对预填阶段的 KV 淘汰，未扩展至解码阶段；实验规模受限于算力，未在更大模型上验证；对极长上下文（超过训练长度）虽有一定泛化，但仍可能出现性能衰减。

---

## 133. Trajectory-Informed Memory Generation for Self-Improving Agent Systems

**arXiv ID:** 2603.10600 | [PDF](https://arxiv.org/pdf/2603.10600v1)

**作者:** Gaodan Fang `[一作]` (IBM Research), Gegi Thomas `[通讯]` (IBM Research)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5111621505)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套自动从LLM驱动代理的执行轨迹中提取可操作学习点，并通过情境记忆检索提升后续执行表现。

**💡 创新点**

创新点在于：①轨迹智能提取与因果归因；②将学习点细分为策略、恢复与优化三类；③基于语义聚类与冲突解决的记忆融合；④多维度检索与LLM引导的动态优先级。

**🔧 技术方法**

核心技术包括：LLM语义理解、因果链追踪、聚类与向量检索、元数据过滤、基于模板的提示生成与注入。

**📊 数据集**

使用AppWorld基准数据集，涵盖多应用、多难度级别的任务。

**📈 对比分析**

与无记忆基线对比，子任务级LLM引导检索方案在测试集上TGC提升3.6pp、SGC提升14.3pp；在最高难度任务中SGC提升28.5pp（相对增幅149%）。

**⚠️ 局限性**

局限性包括：对LLM推理质量高度依赖；检索阈值调优需经验；对易任务可能造成干扰；多代理或跨任务迁移尚未深入探究。

---

## 134. Estimating condition number with Graph Neural Networks

**arXiv ID:** 2603.10277 | [PDF](https://arxiv.org/pdf/2603.10277v1)

**作者:** Erin Carson `[一作]` (Charles University), Xinye Chen `[通讯]` (Sorbonne Université)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5021152274)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种利用图神经网络（GNN）快速估计稀疏矩阵条件数的方法，并给出了两种预测方案；

**💡 创新点**

创新点在于将稀疏矩阵的结构信息通过图构造和特征工程映射到固定维度向量，同时结合GNN和MLP实现对逆范数或完整条件数的高效学习；

**🔧 技术方法**

技术包括O(nnz + n)复杂度的特征提取、基于GCN的消息传递层、全局特征编码、以及对数域下的损失训练；

**📊 数据集**

使用的数据集涵盖五类稀疏矩阵：二维泊松方程、各向异性扩散、变系数扩散、随机生成的对称正定稀疏矩阵、以及对称三对角矩阵；

**📈 对比分析**

与精确计算、Hager–Higham（SciPy和GPU实现）以及Lanczos迭代法比较，GNN在1-范数和2-范数条件数估计上平均速度提升5–10倍，最长相对误差均小于1，且在绝大多数样本上满足LRE<0.5；

**⚠️ 局限性**

局限性包括对训练数据分布的高度依赖、尚未对模型架构和超参数进行全面优化，以及对未见分布的矩阵泛化能力尚待进一步验证。

---

## 135. ReMix: Reinforcement routing for mixtures of LoRAs in LLM finetuning

**arXiv ID:** 2603.10160 | [PDF](https://arxiv.org/pdf/2603.10160v1)

**作者:** Ruizhong Qiu `[一作]` (University of Illinois), Hanghang Tong `[通讯]` (University of Illinois)

**通讯引用:** 17631 | [OpenAlex ID](https://openalex.org/A5068043486)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并解决 Mixture-of-LoRAs 模型中路由权重失衡导致的 LoRA 低效使用问题，提出非可学习的常数路由器 C1，并通过强化学习和 RLOO 技术训练路由器，推理时采用 top‑k 选择。

**💡 创新点**

创新点包括：① 用常数路由权重避免路由权重崩塌；② 将路由器训练框架转化为强化学习，使用无偏 RLOO 梯度估计器；③ 在推理阶段使用 top‑k 选择确保路由器输出最优子集。

**🔧 技术方法**

采用的技术包括 LoRA 低秩适配器、Mixture-of-LoRAs、强化学习（policy gradient + RLOO）、无偏梯度估计、top‑k 选择策略。

**📊 数据集**

使用的数据集包括 GSM8K（数学推理）、HumanEval（代码生成）、ARC‑c（知识检索），以及 CodeAlpaca 进行 HumanEval 的预训练。

**📈 对比分析**

在与多种 PEFT 基线（Prefix Tuning、LoRA、DoRA、rsLoRA、VB‑LoRA、MixLoRA、HydraLoRA 等）相同参数预算的情况下进行对比，C1 在三个任务上的平均提升约 2.82，单任务提升 3.19~4.4，参数占用仅 0.07B，显著优于现有方法。

**⚠️ 局限性**

局限性包括：① 需要额外的强化学习训练成本；② 对路由器质量依赖较高，梯度估计方差可能随 k 或模型深度增大；③ 目前验证范围局限于中等规模 LLaMA‑3 8B 与少数任务，尚未在更大规模模型或更多任务上进一步验证。

---

## 136. Offset Pointing for Energy-efficient Reception in Underwater Optical Wireless Communication: Modeling and Performance Analysi

**arXiv ID:** 2603.10822 | [PDF](https://arxiv.org/pdf/2603.10822v1)

**作者:** Qiyu Ma `[一作]` (Tsinghua University), Mohamed-Slim Alouini `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 91565 | [OpenAlex ID](https://openalex.org/A5083193286)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

在三维截断泊松点过程（TPPP）框架下，推导了UOWC链路的近邻距离分布、期望接收功率、SNR和BER的闭式表达式，并提出并验证了“offset‑pointing”策略以提高能效。

**💡 创新点**

①使用TPPP精准捕捉水下三维异向性；②发展微分能量分析框架；③发现并量化offset‑pointing能显著降低发射功率、提升BER；④在能量约束下求解节点密度与发射功率的最优。

**🔧 技术方法**

随机几何（TPPP）、Lambertian发射模型、Beer‑Lambert衰减、SiPM接收机噪声模型、闭式积分推导、Monte Carlo仿真、能量效率优化。

**📊 数据集**

无公开数据集；采用理论参数与仿真参数（如吸收系数0.151 m⁻¹、LED 60° 半功率角等），在三种场景级别（0–50 m、50–500 m、500–6000 m）进行数值仿真。

**📈 对比分析**

与传统完美指向（PAT）和直向指向（δ=0°）方案比较，评估接收功率、BER和比特/焦耳能效。结果显示offset策略可将发射功率降低≈20%，BER下降1–2个数量级，提升网络寿命和吞吐量；当指向误差>62.5°时优于PAT。

**⚠️ 局限性**

仅考虑均匀海水衰减，未建模层化介质、多径散射或声学扰动；假设点源与接收机理想；未进行实验验证；能量模型忽略静态功耗；硬件功率下限限制影响最优节点密度。

---

## 137. When Fine-Tuning Fails and when it Generalises: Role of Data Diversity and Mixed Training in LLM-based TTS

**arXiv ID:** 2603.10904 | [PDF](https://arxiv.org/pdf/2603.10904v1)

**作者:** Anupam Purwar `[一作]` (Sprinklr AI), Aditya Choudhary `[通讯]` (Sprinklr AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对 Qwen-0.5B 语言模型骨干进行 LoRA 微调，以提升语音克隆的语音质量和信噪比。

**💡 创新点**

首次将 LoRA 直接注入 LM 关注层实现音频 token 预测，并揭示训练数据可变性对感知质量的决定性作用。

**🔧 技术方法**

使用 LoRA (低秩适配)、Qwen-0.5B、GGUF 量化、DNS‑MOS、SNR 等技术。

**📊 数据集**

使用 HiFi‑TTS 与 LibriHeavy‑HQ 两大多说话人数据集。

**📈 对比分析**

与冻结基模型对比，LoRA 在 MOS 上提升至 +0.42、SNR 提升 34% 以上，且多说话人 LoRA 能在未见说话人上提升 0.11–0.29 的 MOS。

**⚠️ 局限性**

对低可变性、低质量训练音频的适配效果有限，验证损失并不能作为感知质量的可靠指标，且过度适配会放大噪声与录音缺陷。

---

## 138. An Extreme Multi-label Text Classification (XMTC) Library Dataset: What if we took "Use of Practical AI in Digital Libraries" seriously?

**arXiv ID:** 2603.10876 | [PDF](https://arxiv.org/pdf/2603.10876v1)

**作者:** Jennifer D'Souza `[一作]`, Osma Suominen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建并公开了双语（英德）多域的图书馆目录记录数据集TIB‑SID，将136,569条真实目录记录与德国统一命名文件（GND）主题词关联，提供可重现的训练/验证/测试拆分，支持机器学习实验。

**💡 创新点**

创新点在于：1）将真实图书馆记录与权威词汇（GND）直接对应，形成机器可操作的多标签数据；2）提供完整的GND词典结构和多语言支持；3）对长尾分布、多义性和跨语言一致性进行统计分析，为算法设计提供洞察；4）搭建可与LLM及传统XMTC方法直接比较的基准。

**🔧 技术方法**

使用的技术包括：多语种大语言模型（Mistral、Llama、Gemma）、句子嵌入模型（MPNet、BGE‑M3、E5）、检索与近邻算法（HNSW、BM25）、传统XMTC方法（Bonsai、Annif）、检索-提示-映射-排序流水线以及多阶段评估（nDCG@k）。

**📊 数据集**

使用的数据集为：TIB‑SID（136k条记录，英德双语、5类记录类型、约40,000个GND标签）以及GND词典（207k条目，JSON化后包含主标签、同义词、关联词、来源等信息）。

**📈 对比分析**

通过三种系统在公开测试集上进行对比：System 1（语义检索+类比）为检索基线；System 2（检索+少量样本提示+映射+排名）为LLM无训练基准；System 3（Annif + 语义预处理/再排序）为混合学习基准。系统3在nDCG@5–20上表现最佳（nDCG@5≈0.602），系统2作为LLM基准（≈0.492），System 1为检索基线。

**⚠️ 局限性**

局限性包括：1）数据仍偏向高频标签，稀有标签的训练样本不足；2）LLM在精确匹配方面易产生误报或过于宽泛的主题；3）多语种与专业领域的跨语言一致性仍有挑战；4）实验仅基于单一图书馆的公开数据，通用性和可迁移性尚待验证。

---

## 139. Ergodicity in reinforcement learning

**arXiv ID:** 2603.10895 | [PDF](https://arxiv.org/pdf/2603.10895v1)

**作者:** Dominik Baumann `[一作]` (Aalto University), Thomas B. Schön `[通讯]` (Uppsala University)

**通讯引用:** 9730 | [OpenAlex ID](https://openalex.org/A5083090794)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于Transformer的视频异常检测框架，结合了时间内与时间间上下文学习。

**💡 创新点**

创新点在于同时考虑帧内与帧间的上下文信息，并通过Transformer建模跨帧依赖。

**🔧 技术方法**

使用多头自注意力机制及位置编码的Transformer结构。

**📊 数据集**

使用UCF Crime和ShanghaiTech公开视频异常数据集。

**📈 对比分析**

与多种先进方法对比，实验表明在两大数据集上均取得了更高的准确率或更低的误报率。

**⚠️ 局限性**

局限在于模型对不同场景的泛化能力有限，且缺乏可解释性。

---

## 140. Fully Symbolic Analysis of Loop Locality: Using Imaginary Reuse to Infer Real Performance

**arXiv ID:** 2603.10196 | [PDF](https://arxiv.org/pdf/2603.10196v1)

**作者:** Yifan Zhu `[一作]` (University of Rochester), Yanghui Wu `[通讯]` (University of Rochester)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了代数式局部性理论，利用象征性重用间隔（RI）推导缓存大小与失效率的多项式表达式；

**💡 创新点**

创新点包括引入无穷重复与虚拟重用来解决首次访问的无限 RI 问题，以及以多项式形式完整描述缓存行为的代数局部性框架；

**🔧 技术方法**

技术手段主要是 Denning 递归、整数集规划与 Barvinok 分解相结合的 MLIR 斜率循环编译器，用于符号 RI 分布的生成和多项式求解；

**📊 数据集**

实验使用 Polybench 套件 30 个科学内核与 Einsum 11 个张量运算共 41 个程序，且对循环融合前后均做评估；

**📈 对比分析**

与 Cachegrind 仿真和 GPU 硬件计数对比，平均预测误差低至 1.1%，准确率达 99.6%，构造平均耗时 41 s，预测不到 1 ms；

**⚠️ 局限性**

局限性在于 RI 分布推导在高维循环或特殊访问模式下可能变得 NP‑hard，且目前仅支持单级 LRU 斜率循环，无法直接处理非斜率或多级缓存场景。

---

## 141. Hardware Efficient Approximate Convolution with Tunable Error Tolerance for CNNs

**arXiv ID:** 2603.10100 | [PDF](https://arxiv.org/pdf/2603.10100v1)

**作者:** Vishal Shashidhar `[一作]` (Indian Institute of Technology Guwahati), Roy P Paily `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在RISC‑V处理器上实现了一种软稀疏近似卷积指令conv_approx，利用MSB信息在不计算乘积的前提下动态跳过对输出贡献极小的乘法，从而显著减少卷积层的MAC次数。

**💡 创新点**

创新点包括：① 通过MSB位置对乘积大小进行快速比较，实现无乘法的“软稀疏”跳过；② 将该算法集成为RISC‑V的自定义指令，硬件实现轻量且可调误差阈值；③ 通过软稀疏而非硬零跳过，兼容ReLU、Tanh等所有激活函数。

**🔧 技术方法**

使用了MSB提取、阈值比较、硬件FSM、RISC‑V自定义指令、软稀疏策略以及基于LeNet‑5的评估框架。

**📊 数据集**

采用MNIST数据集，使用LeNet‑5网络进行推理测试。

**📈 对比分析**

通过与精确卷积和传统硬零跳过方法对比，ReLU模型下MAC减少了88.42%，Tanh模型下74.87%；功耗下降约35%/30%；误差保持<1%；分类准确率在97–98%区间内，无显著下降。

**⚠️ 局限性**

局限性：需手动设定阈值以平衡误差和性能；仅在推理阶段验证，训练阶段效果未知；内存访问仍是能耗主导因素；在更大规模网络或其他数据集上性能与误差需进一步验证。

---

## 142. Spatially conditioned dynamics between population and built form

**arXiv ID:** 2603.10829 | [PDF](https://arxiv.org/pdf/2603.10829v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 143. MCMC Informed Neural Emulators for Uncertainty Quantification in Dynamical Systems

**arXiv ID:** 2603.10987 | [PDF](https://arxiv.org/pdf/2603.10987v1)

**作者:** Heikki Haario `[一作]` (Lappeenranta-Lahti University of Technology LUT), Hendrik Weichel `[通讯]` (Frankfurt University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出MINE框架，将MCMC后验采样与确定性神经网络脱耦，实现高效的不确定性量化仿真器。

**💡 创新点**

创新点在于将后验采样与模型训练分离，利用MCMC生成的后验数据训练确定性网络，并提供区间量化器和快速前向采样两种器件，同时用AEODE网络实现高效ODE仿真。

**🔧 技术方法**

采用技术包括MCMC后验推断、确定性神经网络（量化器与AEODE）、时间嵌入、注意力机制以及物理约束损失。

**📊 数据集**

使用的数据集包括化学动力学六反应系统（Himmel模型）和FaIR简化气候模型的历史温度与排放数据。

**📈 对比分析**

与ChemiODE、Torchdiffeq等基准对比，AEODE在MSE、MBE等指标上略优；量化器的区间覆盖率≈90%，推理速度比传统蒙特卡洛快数十倍。

**⚠️ 局限性**

局限在于需要可行的MCMC采样，且对极高维度或极昂贵模拟器的后验采样仍是瓶颈。

---

## 144. Multi-Agent Memory from a Computer Architecture Perspective: Visions and Challenges Ahead

**arXiv ID:** 2603.10062 | [PDF](https://arxiv.org/pdf/2603.10062v1)

**作者:** Zhongming Yu `[一作]` (University of California, San Diego), Jishen Zhao `[通讯]` (University of California, San Diego)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文将多智能体系统的记忆问题视为计算机体系结构中的层级内存问题，提出共享与分布式两类记忆范式，并构建了三层内存层级（I/O、缓存、长期记忆）以及缺失的缓存共享和访问控制协议；

**💡 创新点**

创新点在于把多智能体记忆抽象为层级内存架构，并明确提出两大关键协议缺口和一致性模型的重要性，为下一代可靠多智能体系统奠定理论基础；

**🔧 技术方法**

主要技术手段为计算机体系结构的内存层级设计理念、缓存共享与访问协议的框架化，以及一致性模型的概念迁移；

**📊 数据集**

未使用具体数据集，论文主要为理论与架构性探讨；

**📈 对比分析**

通过与传统单智能体内存设计、现有分布式内存系统和多进程缓存共享的对比，论证该层级架构能更好地解决记忆带宽、缓存与一致性问题，但未给出量化性能指标；

**⚠️ 局限性**

局限在于缺乏实际实现与实验验证，缓存共享与访问协议仍未细化，一致性模型仍处于概念阶段，需要后续研究补全与落地验证。

---

## 145. Quantifying Membership Disclosure Risk for Tabular Synthetic Data Using Kernel Density Estimators

**arXiv ID:** 2603.10937 | [PDF](https://arxiv.org/pdf/2603.10937v1)

**作者:** Rajdeep Pathak `[一作]` (Indian Institute of Technology Hyderabad), Sayantee Jana `[通讯]` (Indian Institute of Technology Hyderabad)

**通讯引用:** 68 | [OpenAlex ID](https://openalex.org/A5044167864)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于核密度估计（KDE）的非参数方法，用最近邻距离对合成数据与真实数据的成员关系进行概率化推断，并可生成 ROC 曲线；

**💡 创新点**

创新点包括：①用 KDE 对成员/非成员距离分布建模，获得软概率输出；②提供真分布攻击与现实攻击两种场景；③不需要昂贵的 shadow 模型，计算效率极高；④在多种生成器和数据集上展示传统 F1 低估风险的现象；

**🔧 技术方法**

使用核密度估计、Gower 距离、贝叶斯理论、ROC 曲线、F1/准确率评估、Scott 规则带宽选择及 GPU 加速等技术；

**📊 数据集**

在四个公开表格数据集上评估：MIMIC‑IV、UK Census、Texas‑100X、Nexoid COVID‑19；

**📈 对比分析**

与传统基于距离阈值的 Method1 对比，采用 Accuracy、F1、log‑ROC 评价；KDE 方法在多数生成器上得到更高 F1（如 TVAE 0.877、Bayesian Network 0.975），并在低 FPR 下实现更高 TPR，揭示了更严峻的隐私风险；

**⚠️ 局限性**

局限性在于仅在平衡成员/非成员的攻击集中评估；缺乏理论证明距离‑成员概率映射的收敛性；在极稀疏或高维场景下 KDE 性能可能下降；未结合 shadow 模型或对抗训练进一步提升。

---

## 146. Rethinking the Harmonic Loss via Non-Euclidean Distance Layers

**arXiv ID:** 2603.10225 | [PDF](https://arxiv.org/pdf/2603.10225v1)

**作者:** Maxwell Miller-Golub `[一作]` (American University), Roberto Corizzo `[通讯]` (American University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了基于非欧氏距离的谐波损失（harmonic loss）作为交叉熵的替代方案，并在视觉和语言任务上进行广泛实验。

**💡 创新点**

创新点在于系统探索多种距离度量（余弦、曼哈顿、Chebyshev、Bray–Curtis、Mahalanobis 等）对谐波损失的影响，并在性能、可解释性和可持续性三维度上进行统一评估。

**🔧 技术方法**

使用了深度学习框架（PyTorch）、混合精度训练、梯度裁剪、动态学习率调度，以及基于 CodeCarbon 的能耗与碳排放监测。

**📊 数据集**

数据集涵盖视觉任务（MNIST、CIFAR‑10/100、Marathi Sign Language、TinyImageNet）和语言任务（OpenWebText 预训练语料），以及多种模型骨干（MLP、CNN、ResNet‑50、PVT、BERT、GPT‑2、Qwen）。

**📈 对比分析**

通过与交叉熵、欧氏谐波损失以及多种现代基线（Focal Loss、Label Smoothing 等）在准确率/损失、梯度稳定性、PCA 解释度和碳排放等指标下对比，发现余弦谐波损失在多数设置下性能最优、可解释性最佳且能耗最低；Bray–Curtis 和 Chebyshev 在可解释性方面表现突出，Mahalanobis 虽然聚类更锐利但计算成本更高。

**⚠️ 局限性**

局限性包括：对高维大规模数据时 Mahalanobis 的协方差估计与矩阵求逆开销大；实验仅针对少数几种网络架构和任务，缺乏在更广泛领域（如语音、强化学习）的验证；并未深入探讨距离参数（如 Minkowski 的 p）自动调优的策略。

---

## 147. Beyond Interleaving: Causal Attention Reformulations for Generative Recommender Systems

**arXiv ID:** 2603.10369 | [PDF](https://arxiv.org/pdf/2603.10369v1)

**作者:** Hailing Cheng `[一作]` `[通讯]` (Linkedin Inc), Hailing Cheng (Linkedin Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文重新设计了生成式推荐系统的序列建模方式，提出了不使用交错 token 的 Attention-based Late Fusion（AttnLFA）和 Attention-based Mixed Value Pooling（AttnMVP）两种新架构。

**💡 创新点**

核心创新在于将物品与用户动作的因果关系显式编码为注意力聚合，消除交错 token 带来的注意力噪声和序列长度两倍的问题。

**🔧 技术方法**

技术手段包括基于 Transformer 的因果注意力、查询位移实现严格因果掩码、混合值融合，以及对 FlashAttention 的高效实现。

**📊 数据集**

评估数据来自大型社交网络（LinkedIn）产品推荐日志，使用 12 个月内最多 1024 条交互的用户序列。

**📈 对比分析**

与传统交错 token 基线在相同超参下对比，AttnLFA 与 AttnMVP 分别提升评估损失约 0.29%/0.8%、降低训练时间 23%/12%，并在多任务指标上取得更低的 Normalized Entropy。

**⚠️ 局限性**

局限性包括在动作空间小、语义异质的场景下双流结构 AttnDHN 训练不稳定，且仍受限于少量动作标签，难以完全替代交错 token 方案。

---

## 148. A Grammar of Machine Learning Workflows

**arXiv ID:** 2603.10742 | [PDF](https://arxiv.org/pdf/2603.10742v1)

**作者:** Simon Roth `[一作]` `[通讯]`, Simon Roth

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个面向监督学习的结构化语法（7个核心原语+4个硬性约束），通过类型安全的有向无环图与运行时守护，在调用时即阻止数据泄漏；并实现了Python、R、Julia三种可执行实现，验证其在不同语言下的一致性。

**💡 创新点**

创新点在于：1) 将评估/评估边界抽象为终端评估约束（assess-once），使Class II/III泄漏在编程时即被拒绝；2) 通过内容地址分区与守护实现无缝的分区管理；3) 提供可复现的语法规范与三实现，证明其跨语言可移植性和可验证性。

**🔧 技术方法**

技术手段包括：类型系统与运行时守护、内容地址分区识别、声明式/显式模式、Typed DAG、自动化测试框架、Cohen's d_z 统计分析、实验设计与多实例评估。

**📊 数据集**

使用了OpenML公开数据集（约2,047个实验实例及3,759个规模扩展实例），涵盖二分类、多分类和回归任务，实验算法包括逻辑回归、随机森林、决策树和k近邻等。

**📈 对比分析**

通过与传统框架（sklearn、tidymodels、mlr3、AutoGluon等）的对比，验证了四个约束与终端评估的有效性；在实验中，Class II/III泄漏被成功拦截，避免了性能过高估计；三种实现产生一致结果，说明语法在不同语言下均可保持同一结构安全性。

**⚠️ 局限性**

局限性包括：仅覆盖批量监督学习的表格数据，无法处理时间序列、深度学习、无监督等场景；仅防止结构性泄漏，无法阻止验证集过拟合、特征工程误用等语义错误；实现存在序列化/反序列化后评估一次约束被重置、对多模型测试集窥探无全局限制等运行时绕过风险。

---

## 149. Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming

**arXiv ID:** 2603.10711 | [PDF](https://arxiv.org/pdf/2603.10711v1)

**作者:** Yilin Zou `[一作]` (Tsinghua University), Fanghua Jiang `[通讯]` (Tsinghua University)

**通讯引用:** 2047 | [OpenAlex ID](https://openalex.org/A5087223201)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个全GPU原生的轨迹优化框架，结合序列凸规划（SCP）和基于共识的ADMM实现时间分裂与并行计算；

**💡 创新点**

创新点在于通过时间层级拆分把全局稀疏求解转化为独立的时点密集求解与闭式动态一致性更新，完全避免CPU序列化的稀疏因式分解，实现GPU上高吞吐、低能耗的并行优化；

**🔧 技术方法**

使用的技术包括SCP、ADMM、JAX自动微分与JIT编译、GPU线程块并行、闭式线性方程求解与投影操作；

**📊 数据集**

实验数据集主要为合成控制任务：6自由度四旋翼机在障碍环境中的敏捷飞行、Mars火星降落任务以及随机障碍环境下的批量轨迹生成；

**📈 对比分析**

与12核CPU上的iLQR基线相比，GPU求解器在批量大小1000时实现101 Hz吞吐率，速度提升约4.1×，能耗降低51%，并可在100 Hz以上完成实时MPC；

**⚠️ 局限性**

局限性在于缺乏对SCP+ADMM组合的严格收敛理论分析，且目前仅在仿真/边缘设备上验证，尚未在真实硬件上完成实验验证。

---

## 150. Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas

**arXiv ID:** 2603.10303 | [PDF](https://arxiv.org/pdf/2603.10303v1)

**作者:** Tim Schopf `[一作]` (Dresden University of Technology), Michael Färber `[通讯]` (Dresden University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 rino benchmark，用于自动评估研究想法的新颖性判断；

**💡 创新点**

首次构建了基于 ICLR 2022/2023 审稿数据的标准化 1–5 分制新颖性评分和文本解释框架，并设计了九种自动评估指标；

**🔧 技术方法**

使用 GPT‑OSS‑120B 等大型语言模型对论文摘要、评审摘要进行信息抽取与结构化，并利用 LLM 生成文本理由；

**📊 数据集**

数据集来源于 6,410 篇 ICLR 2022/2023 公开提交及其评审，经过过滤后得到 1,381 个研究想法及 25.23 条相关工作；

**📈 对比分析**

对多种 LLM（含非推理与推理模型）在新颖性分数和文本理由上的性能进行零样本评估，发现即使理由与人类相符，模型在分数预测上的 F1 仅约 17%，显著低于理想水平；

**⚠️ 局限性**

局限性包括数据单一来源（ICLR）、以评审文化为标准、仅关注技术新颖性、英语主导、LLM 可能产生幻觉、未覆盖其他科研质量维度等。

---

## 151. SpreadsheetArena: Decomposing Preference in LLM Generation of Spreadsheet Workbooks

**arXiv ID:** 2603.10002 | [PDF](https://arxiv.org/pdf/2603.10002v1)

**作者:** Srivatsa Kundurthy `[一作]` (Cornell University), John Ling `[通讯]` (Longitude Labs Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 SpreadsheetArena 平台，用盲目对比投票评估 LLM 生成的电子表格工作簿，构建 16 个模型的全球排名；

**💡 创新点**

创新点在于：①将 arena 评估迁移到结构化输出（电子表格）；②使用可观测特征的增广 Bradley‑Terry 模型解析用户偏好；③在不同领域（学术、金融、创意等）揭示特征对偏好的异质影响；④设计多标签失效分类法并与专家评估对齐；

**🔧 技术方法**

技术手段包括：LLM 端到端生成 JSON 表格、Blender 模型的 Bradley‑Terry 与 Elo 评分、特征增广回归、Prompt 嵌入+K‑NN 分类、BERTopic + HDBSCAN 的失效聚类、LLM 判别器的多标签评估；

**📊 数据集**

使用数据集：436 条种子提示 + 4,357 对投票，覆盖 16 种 LLM 生成表格，涵盖学术、金融、创意等 6 类提示；

**📈 对比分析**

通过 4,357 轮盲目投票建立排行榜，特征增广后 Elo 分数压缩并显著改变排名；专家评估显示平均整体评分 2.87/5，功能性指标较好但格式与专业规范不足，表明用户偏好与专业评价仅部分对齐；

**⚠️ 局限性**

局限性在于：投票偏好难以完全反映所有功能维度，专家与投票结果对齐有限；仅使用一次性生成模型，未覆盖迭代或代理式生成；数据集规模有限，未涵盖所有业务场景，缺乏可解释性与因果推断。

---

## 152. Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity

**arXiv ID:** 2603.10990 | [PDF](https://arxiv.org/pdf/2603.10990v1)

**作者:** Zhengyao Fang `[一作]` (Harbin Institute of Technology), Wenjie Pei `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2022 | [OpenAlex ID](https://openalex.org/A5078487642)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个专门评估真实风格文本生成图像色彩真实性的完整框架，包括大规模 Color Fidelity Dataset (CFD)、可训练的 Color Fidelity Metric (CFM) 以及无需训练的 Color Fidelity Refinement (CFR) 调节方法。

**💡 创新点**

创新点在于首次将色彩真实性作为独立的评估维度，构建基于指导尺度产生的颜色失真序列来训练模型，并通过跨模态注意力实现空间-时间自适应的指导尺度调节，从而在保持语义一致性的前提下显著提升色彩真实感。

**🔧 技术方法**

主要技术包括 Qwen2‑VL 视觉‑语言双模态编码器、基于 soft‑rank 的差分排序损失、以及基于 CFM 产生的注意力图实现的空间‑时间动态 CFG 调节；CFR 作为后处理模块可直接插入任何基于 CFG 的扩散模型。

**📊 数据集**

使用 1.3M 张图像的 CFD 数据集，涵盖 12 个类别的高质量真实照片以及通过 11 种 T2I 模型在不同指导尺度下合成的 6 种逐级失真变体，并收集 2 万余份人工色彩真实性打分。

**📈 对比分析**

与现有的 FID、CLIPScore、PickScore、ImageReward 等指标相比，CFM 在 CFD‑Test 上对色彩真实性的判别准确率超过 80%，与人类打分的 Spearman、Pearson、Kendall 相关系数均超过 0.85；CFR 在不显著提升 FID/CLIPScore 的前提下，使平均饱和度降低 0.08‑0.11，CFM 分数提升 1.3‑2.0 分，证明了其有效性。

**⚠️ 局限性**

局限性包括：1）CFD 与 CFM 主要聚焦真实风格，可能对艺术化或夸张风格的评估不足；2）CFR 依赖于跨模态注意力，若文本提示质量差或模型与注意力对齐不佳，调节效果会受限；3）目前仍未涵盖视频或多帧序列的色彩一致性问题。

---

## 153. Efficiency vs Demand in AI Electricity: Implications for Post-AGI Scaling

**arXiv ID:** 2603.10498 | [PDF](https://arxiv.org/pdf/2603.10498v1)

**作者:** Doyi Kim `[一作]` (Korea Advanced Institute of Science and Technology), Changick Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 7057 | [OpenAlex ID](https://openalex.org/A5069759184)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

在GCAM宏观能源经济模型中新增AI计算部门，基于服务输出（FLOP）与随时间变化的计算能耗强度（γ）来计算美国AI数据中心的电力需求，并开展多情景与灵敏度分析。

**💡 创新点**

首次将AI服务增长与系统级能效轨迹显式嵌入宏观模型，识别出效率主导与需求主导的临界阈值，为长期AI电力需求与碳排放提供情景映射框架。

**🔧 技术方法**

利用GCAM部分均衡市场机制，结合浮点运算量（FLOP）与能耗强度γ(t)的乘积来估算电力需求，并对价格弹性、收入弹性与效率提升速度进行情景模拟。

**📊 数据集**

采用2024年全球数据中心约1.5%电耗估计、GPU性能提升率1.3×/年等公开数据，基准γ设为7.7×10⁻⁵ EJ/YF，基于这些数据构建基准与快速/慢速效率情景。

**📈 对比分析**

通过比较基准、Rapid、Slow效率情景以及不同价格弹性（-0.2~ -1.2）和收入弹性（1.6~3.5）对电力需求与AI服务输出的影响，发现价格弹性对需求影响≤5%，而收入弹性对需求影响显著，效率轨迹决定需求主导阈值；模型结果与外部估计保持一致。

**⚠️ 局限性**

局限在于AI服务输出仅通过弹性假设在GCAM内生决定，未考虑工作负载异质性、部署策略；未涵盖地区电网约束与发电组合差异；研究仅限美国情景，跨国验证仍待进一步研究。

---

## 154. Don't Let the Claw Grip Your Hand: A Security Analysis and Defense Framework for OpenClaw

**arXiv ID:** 2603.10387 | [PDF](https://arxiv.org/pdf/2603.10387v1)

**作者:** Zhengyang Shan `[一作]` (Shandong University), Minghui Xu `[通讯]` (Shandong University)

**通讯引用:** 1676 | [OpenAlex ID](https://openalex.org/A5103077343)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并强化本地代码代理OpenClaw的安全性，探究不同LLM后端在执行恶意指令时的防御表现；

**💡 创新点**

提出并实现了基于Human‑in‑the‑Loop(HITL)的多层防御架构，显著提升了攻击检测率，尤其在中等安全级别模型中效果突出；

**🔧 技术方法**

采用对话式工具调用审计、路径与模式匹配、语义意图判定以及沙箱监控等技术；

**📊 数据集**

使用47个基于MITRE ATLAS/ATT&CK的攻击场景与六种主流LLM后端（Claude、Qwen、GPT、Kimi、Gemini、DeepSeek）进行测试；

**📈 对比分析**

在基线模式下，防御率从17%（DeepSeek）到83%（Claude）不等；加入HITL后，整体防御率提升至最高91.5%，平均提升约12.5%，展示了显著的安全提升；

**⚠️ 局限性**

对HITL的模式覆盖和编码绕过存在盲点，沙箱逃逸检测率仍低于33%，且仅在受限对话轮次内测试，可能无法覆盖更复杂的多步攻击。

---

## 155. Open Educational Resources: Barriers and Open Issues

**arXiv ID:** 2603.10013 | [PDF](https://arxiv.org/pdf/2603.10013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 156. MoXaRt: Audio-Visual Object-Guided Sound Interaction for XR

**arXiv ID:** 2603.10465 | [PDF](https://arxiv.org/pdf/2603.10465v1)

**作者:** Tianyu Xu `[一作]` (Google), Adarsh Kowdle `[通讯]` (Google)

**通讯引用:** 3874 | [OpenAlex ID](https://openalex.org/A5067691489)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个实时 XR 系统，利用音视频线索对单通道音频进行源分离并实现用户交互控制。

**💡 创新点**

创新点在于将视觉锚点与分层 Transformer 架构结合，实现最多 5 个并发源的实时分离，并支持用户细粒度调音。

**🔧 技术方法**

采用分层音频‑视觉 Transformer、面部与乐器检测、Teacher‑Student 蒸馏、动态音源集合等技术。

**📊 数据集**

使用自收集的 30 条 1‑分钟录音数据集，涵盖最多 5 位说话人和 3 件乐器的混合场景。

**📈 对比分析**

与 Sound of Pixels、DAVIS、AudioScopeV2、AV‑MossFormer2 等基线对比，实时模型 WER 0.499、DNSMOS≈3.5，优于多数基线；离线模型 WER 0.382，表现最佳。

**⚠️ 局限性**

限制包括对视觉遮挡的依赖、约 2 秒延迟、需外部 PC 推理、对多源数量超过 4 时性能下降、硬件可移植性不足。

---

## 157. TASER: Task-Aware Spectral Energy Refine for Backdoor Suppression in UAV Swarms Decentralized Federated Learning

**arXiv ID:** 2603.10075 | [PDF](https://arxiv.org/pdf/2603.10075v1)

**作者:** Sizhe Huang `[一作]`, Shujie Yang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出TASER框架，在UAV基分布式联邦学习场景下针对隐蔽后门攻击进行防御。

**💡 创新点**

创新点在于利用梯度的频谱能量集中度做任务感知筛选，避免复杂的异常检测，降低通信与计算开销。

**🔧 技术方法**

核心技术包括离散余弦变换（DCT）、任务感知频率评分、top‑k 频谱筛选、逆DCT重构及分布式邻居通信。

**📊 数据集**

实验使用EMNIST和CIFAR‑10两个图像分类数据集。

**📈 对比分析**

与Weak‑DP、Multi‑Metrics、Krum、RFA、FreqFed等基线比较，TASER在攻击成功率（ASR）低于20%且主任务准确率下降≤5%，在白盒与黑盒隐蔽后门攻击场景中均优于现有方法。

**⚠️ 局限性**

局限性：需要手动或经验确定top‑k比例，过度压缩会损失主任务性能；对极低频率或更高级攻击模式的鲁棒性尚未充分验证。

---

## 158. A Retrieval-Augmented Language Assistant for Unmanned Aircraft Safety Assessment and Regulatory Compliance

**arXiv ID:** 2603.09999 | [PDF](https://arxiv.org/pdf/2603.09999v1)

**作者:** Gabriele Immordino `[一作]` (Zurich University of Applied Sciences), Marcello Righi `[通讯]` (Zurich University of Applied Sciences)

**通讯引用:** 294 | [OpenAlex ID](https://openalex.org/A5002731118)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一个基于检索的无人机安全评估与合规助手，支持在监管框架下进行安全评估、认证与法规遵从的决策辅助。

**💡 创新点**

创新点包括：① 将检索与生成严格分离，采用检索检索证据后仅用检索证据生成答案，确保可追溯与可审计；② 结合密集向量检索与 BM25 词法检索，使用 Reciprocal Rank Fusion 与 Maximal Marginal Relevance 进行候选多样化与精细排序；③ 引入 ColBERT 神经重排序器与词级匹配保证检索上下文与生成语义对齐；④ 通过手工分块与表格专门提取保持法规文本语义完整与可追溯；⑤ 在对话与结构化指标两种使用场景下，统一实现引用标注与“不足”提示，保持系统安全边界。

**🔧 技术方法**

技术栈包括：1）FAISS（Dense 与 HNSW）+ BM25；2）Sentence‑Transformer (MiniLM) 编码；3）ColBERT 重排序；4）OpenAI GPT‑4 或类似 LLM；5）自定义检索前置查询生成（Qwen3‑8B‑Instruct）与后置答案控制；6）Python/LLM API 与可视化交互界面。

**📊 数据集**

数据集为欧洲航空安全局（EASA）发布的 CS‑UAS（SORA 与 PDRA 相关章节）及其国家解释材料；所有文本被手工拆分为文章、可接受合规性 (AMC) 与表格块，并标注页码、标题等元数据。

**📈 对比分析**

对比方法：检索精度采用 Hit@1/3/5/10 与 MRR；答案的“Grounded”度量结合句子级引用验证；使用两组用例（对话问答与指标生成）分别评估一致性、解释相似度与准确率。实验结果显示：检索 Hit@1 平均 83%；直接匹配时答案 50% grounded；指标任务整体一致性 91.7%，准确率 81.8%，其中“评估深度”完全正确。

**⚠️ 局限性**

局限性：① 对语义重述或重构问题的召回下降；② 生成仍可能未引用最相关块或出现未覆盖的证据；③ 需要人工拆分与表格提取，维护成本高；④ 解释文本可变性导致解释相似度低；⑤ 仅基于单一法规集，缺乏多国或更新版本支持；⑥ 需进一步增强句子级验证与多模态（表格、图像）支持。

---

## 159. Large Language Models as Annotators for Machine Translation Quality Estimation

**arXiv ID:** 2603.10775 | [PDF](https://arxiv.org/pdf/2603.10775v1)

**作者:** Sidi Wang `[一作]` (Maastricht University), Amir Kamran `[通讯]` (Taus)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大语言模型（LLM）生成简化的MQM（多维质量度量）注释，并用这些合成数据训练COMET质量估计模型。

**💡 创新点**

提出了PPbMQM Prompt（基于提示的MQM）与severity（严重程度）刻度结合的方案，既保留了MQM的可解释性，又通过严重程度阈值控制误报，生成的合成注释与人工注释高度相关。

**🔧 技术方法**

使用GPT‑4o等LLM的零/少量示例提示、MQM错误类别与严重程度映射、COMET‑QE训练框架以及统计相关性评估（Pearson、Spearman、Kendall）。

**📊 数据集**

主要使用 EbHE‑WMT‑MT 2022/2023 的中文‑英文、英文‑德文专家评测数据，合成标注共 20703 zh‑en 和 10121 en‑de 段。

**📈 对比分析**

将基于合成MQM标注训练的COMET与基于人工MQM标注训练的COMET在同一测试集上做 Pearson/Spearman/Kendall 比较。结果显示，合成数据训练的模型在整体与低质量段的 Pearson 相关性均高于人类标注训练模型，Spearman 与 Kendall 也表现相当或略优。

**⚠️ 局限性**

存在潜在的数据泄漏风险、仅覆盖两对高资源语言与有限领域、仅使用单一模型初始化，以及LLM训练数据来源不公开。

---

## 160. Evaluating Few-Shot Pill Recognition Under Visual Domain Shift

**arXiv ID:** 2603.10833 | [PDF](https://arxiv.org/pdf/2603.10833v1)

**作者:** W. I. Chu `[一作]`, L. Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在真实部署环境下，少样本（few-shot）药丸识别的迁移学习与评估，重点关注场景混乱、重叠和注释异构的跨数据集域移位。

**💡 创新点**

创新点在于：①以部署为导向的评估框架，使用分类中心和误差指标而非传统AP；②将少样本微调视为诊断工具，揭示语义识别与定位解耦的失效模式；③证明基域视觉真实性是提升低样本泛化的关键因素。

**🔧 技术方法**

技术手段包括：基于 Faster R‑CNN 的两阶段检测器、FsDet 训练框架、固定学习率与迭代次数的少样本微调、冻结骨干网络、重新初始化 ROI 头；评估使用前景分类准确率、误检率和损失曲线。

**📊 数据集**

数据集：基域使用 CURE（单药丸、控制环境）和 MEDISEG（多药丸、真实场景）进行训练；新颖部署数据集包含多药丸重叠与混乱背景，另设 133 张仅重叠的压力测试图像；所有数据均人工标注边界框与分割掩码。

**📈 对比分析**

比较方法：在同一新颖部署集上对比基于 CURE 与 MEDISEG 训练的模型，采用前景分类准确率、误检率与损失等指标。结果显示：普通场景下 1‑shot 即可达到约 0.99 的分类准确率；但在重叠压力测试中，CURE 训练模型分类准确率降至 0.13，MEDISEG 训练模型则提升至 0.40‑0.74，显示显著优势；总体来看，定位召回率在重叠条件下显著下降。

**⚠️ 局限性**

局限性包括：① CURE 的全图框注释导致无法使用 AP 进行统一评估；② 评估仅覆盖少量新类与单一部署环境；③ 仅采用 Faster R‑CNN 结构，未探究更高效或更鲁棒的网络；④ 研究聚焦于分类指标，未深入定位误差的根本原因。

---

## 161. From Verification to Herding: Exploiting Software's Sparsity of Influence

**arXiv ID:** 2603.10478 | [PDF](https://arxiv.org/pdf/2603.10478v1)

**作者:** Tim Menzies `[一作]` (North Carolina State University), Kishan Kumar Ganguly `[通讯]` (North Carolina State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出 Herding 方法，将软件测试视为无模型的搜索任务，通过轻量采样发现控制变量，利用软件影响稀疏性快速逼近最优解。

**💡 创新点**

创新点在于摆脱传统模型驱动的验证陷阱，提出 EZR（Efficient Zero‑knowledge Ranker）对比学习器，仅需 32 次采样即可达到 90% 最优，并系统验证了“影响稀疏性”理论。

**🔧 技术方法**

技术上采用对比集学习、贝叶斯分类器、离散化分箱、Welford 算法增量更新；并与 SMAC、Optuna、DEHB、Random、KPP 等超参数优化框架进行对比。

**📊 数据集**

使用了 63 个来自 MOOT 仓库的多目标优化任务，覆盖编译器调优、视频编码、项目管理、金融预测等领域。

**📈 对比分析**

通过 Normalized Regret（相对最佳性）指标进行比较，EZR 在 32 次采样下实现 90% 最优，优于 SMAC、Optuna、DEHB 等算法，且在所有任务中表现一致。

**⚠️ 局限性**

局限性包括：在极端安全关键场景 90% 最优仍不足；方法假设影响稀疏性，若 AI 生成的代码稠密度升高则需加入稀疏性约束；未对动态可变依赖的系统进行评估。

---

## 162. Digging Deeper: Learning Multi-Level Concept Hierarchies

**arXiv ID:** 2603.10084 | [PDF](https://arxiv.org/pdf/2603.10084v1)

**作者:** Oscar Hill `[一作]` (University of Cambridge), Mateja Jamnik `[通讯]` (University of Cambridge)

**通讯引用:** 1496 | [OpenAlex ID](https://openalex.org/A5036018012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了多层概念拆分（MLCS）和深层概念嵌入模型（Deep-HiCEM），实现仅用顶层标签即可自动发现并表示多层概念层次结构，并支持多层级的概念干预

**💡 创新点**

创新点在于将概念拆分扩展为多级层次（使用层级稀疏自编码器），以及构建能够支持任意深度层次结构并进行多抽象层级干预的Deep-HiCEM架构，使得模型在缺乏完整概念注释的情况下仍能生成人类可解释且可操作的概念层级

**🔧 技术方法**

核心技术包括层级稀疏自编码器（HiSAE）、概念拆分、概念嵌入网络、正负子概念模块、概念干预机制以及基于 ROC‑AUC 的概念匹配评估

**📊 数据集**

实验使用了 MNIST‑ADD、SHAPES、CUB、AwA2 以及自定义的多层概念数据集 PseudoKitchens‑2

**📈 对比分析**

与标准 HiCEM、CEM、CBM、PCBM 等基线相比，Deep-HiCEM+MLCS 在任务准确率上与 HiCEM 差距不足 1%，概念预测 ROC‑AUC 与 HiCEM 接近，且在多层概念干预时能够提升（或至少不损害）任务性能，整体表现与基线相当

**⚠️ 局限性**

主要局限包括：干预时有时会导致性能下降；层级稀疏自编码器不保证一定能发现有意义的概念；实验仅限于两层概念层级，未验证更深层次；缺乏对更大、更复杂数据集的扩展和深入评估

---

## 163. Word Recovery in Large Language Models Enables Character-Level Tokenization Robustness

**arXiv ID:** 2603.10771 | [PDF](https://arxiv.org/pdf/2603.10771v1)

**作者:** Zhipeng Yang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 4063 | [OpenAlex ID](https://openalex.org/A5100401482)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型对字符级分词的鲁棒性，并发现模型通过内部“词语恢复”机制重构词级表示。

**💡 创新点**

提出词语恢复概念并用解码、子空间消除、注意力遮蔽等方法揭示其因果作用。

**🔧 技术方法**

使用解码（LogitLens）、子空间干预、注意力遮蔽等可解释性技术。

**📊 数据集**

在ARC‑Easy、ARC‑Challenge、CommonsenseQA、OpenbookQA四个问答基准上评估。

**📈 对比分析**

对照规范分词得到的准确率，字符级分词仅略降，词语恢复得分最高可达96.8%，干预后任务表现显著下降。

**⚠️ 局限性**

仅聚焦字符级分词，实验模型有限，缺乏对更大规模或多样化任务的验证，解释框架仍需进一步泛化。

---

## 164. Novel Architecture of RPA In Oral Cancer Lesion Detection

**arXiv ID:** 2603.10928 | [PDF](https://arxiv.org/pdf/2603.10928v1)

**作者:** Revana Magdy `[一作]` (MSA University), Ali Hamdi `[通讯]` (MSA University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研发了一套基于 EfficientNetV2B1 的深度学习模型与 RPA 工具（UiPath、Automation Anywhere）相结合的口腔癌病变自动检测系统；

**💡 创新点**

创新点在于将 Singleton 与 Batch 处理模式引入 Python 自动化管线，显著提升推理速度并降低 RPA 平台的计算负担；

**🔧 技术方法**

技术包括 CNN（EfficientNetV2B1）、数据增强（Albumentations）、Python 与 RPA（UiPath、Automation Anywhere）的混合编程、设计模式（Singleton、Batch）以及批量推理；

**📊 数据集**

使用约 3000 张口腔临床图像的多类别（健康、良性、OPMD、口腔癌）数据集，并在 31 张测试图像上评估；

**📈 对比分析**

通过对比 UiPath、Automation Anywhere 以及两版 OC‑RPA（v1 与 v2）的平均推理时间，OC‑RPA v2 以 0.06 秒/图像实现了 60‑100 倍的加速，远超传统 RPA；

**⚠️ 局限性**

局限在于仅在单一 RPA 平台上验证，未测试更大规模数据集的实时性能，也缺乏多模态和可解释性分析。

---

## 165. Reinforcement Learning with Conditional Expectation Reward

**arXiv ID:** 2603.10624 | [PDF](https://arxiv.org/pdf/2603.10624v1)

**作者:** Changyi Xiao `[一作]` (Fudan University), Yixin Cao `[通讯]` (Fudan University)

**通讯引用:** 5669 | [OpenAlex ID](https://openalex.org/A5013247988)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的强化学习奖励机制——Conditional Expectation Reward（CER），通过利用大语言模型自身作为隐式验证器，在没有手工规则或外部模型的前提下，为自由形式回答生成软化的奖励信号，支持在数学与一般推理任务中进行强化学习；

**💡 创新点**

创新点在于：①用模型内部一致性评估答案，而非外部硬规则，消除对领域特定验证规则的依赖；②将精确匹配奖励平滑化为连续值，提供分级反馈；③通过自一致性放大效应（self‑consistency amplification）提升对正确答案的鼓励；④实现可计算的经验估计，并通过贝叶斯重排与蒙特卡罗采样实现高效求值；

**🔧 技术方法**

技术核心包括：强化学习与可验证奖励（RLVR）框架；基于模型内部一致性的期望条件奖励定义；贝叶斯公式与蒙特卡罗采样推导的经验CER估计；政策梯度优化与奖励分离；以及可重用采样实现的高效计算；

**📊 数据集**

使用的主要数据集为：数学领域的MATH‑7.5K、MATH500、AMC23、AIME2024、AIME2025；通用推理领域的WebInstruct（50K问答）以及评测集SuperGPQA、MMLU‑Pro；

**📈 对比分析**

与传统方法（exact‑match、rule‑based、模型验证器General‑verifier、基于困惑度的VeriFree）进行对比。实验表明，在通用领域 CER 的平均表现优于所有基线，Rule+CER 更进一步提升；在数学领域 CER 与规则奖励相当甚至略胜，且超越模型验证器。CER 的软奖励实现了更高的稠密学习信号，带来更稳健的性能提升；

**⚠️ 局限性**

局限性包括：①需要额外采样（M 取值折中计算成本与精度），导致运行时间显著；②CER 仍依赖语言模型的自一致性质量，若模型内部一致性不足，奖励信号可能失真；③对极端多义或高度开放的答案仍可能产生误判；④实验覆盖的领域虽多，但对更专业或极端域（如医学、法律）尚未验证。

---

## 166. K-Join: Combining Vertex Covers for Parallel Joins

**arXiv ID:** 2603.10177 | [PDF](https://arxiv.org/pdf/2603.10177v1)

**作者:** Simon Frisk `[一作]` (University of Wisconsin Madison), Paraschos Koutris `[通讯]` (University of Wisconsin Madison)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文提出一种新的并行 join 算法 κ-Join，能够在 MPC 模型中以 O~(n/p^{1/κ}) 的负载完成任意自然连接查询。

**💡 创新点**

其创新点在于引入了新的超图量度“reduced quasi vertex-cover” κ，并通过对不同子查询顶点覆盖的线性组合来选择 HyperCube 的分配比，从而提升了最坏情况负载上界。

**🔧 技术方法**

技术上结合了细粒度数据分区、HyperCube 原语、顶点覆盖与匹配的线性规划、半连接、广播重合属性以及混合整数线性规划来实现权重映射与负载控制。

**📊 数据集**

论文未使用真实数据集，而是通过理论实例和随机 sparse product 构造的合成数据来进行性能分析与证明。

**📈 对比分析**

通过与 PAC 等现有算法对比，证明 κ-Join 的负载不大于 n/p^{1/γ}，在 Loomis‑Whitney 等查询上严格优于之前算法，并在某些查询上与 ρ^* 下界匹配，展示了更优的理论性能。

**⚠️ 局限性**

主要限制在于尚未证明该上界的最优性，缺少完整下界匹配；对一般超图仍有可能存在更优算法；以及实际系统实现和对数据偏斜的处理仍待验证。

---

## 167. Learning to Score: Tuning Cluster Schedulers through Reinforcement Learning

**arXiv ID:** 2603.10545 | [PDF](https://arxiv.org/pdf/2603.10545v1)

**作者:** Martin Asenov `[一作]` (Huawei), Adam Barker `[通讯]` (Huawei)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

本文提出了一种基于强化学习的框架，用来自动调节分布式调度器中分数函数的权重，以提升聚类资源利用率和作业性能。

**💡 创新点**

创新点包括：①将多步参数调优建模为强化学习问题；②引入百分比改进奖励、帧堆叠和域信息限制三种技巧；③设计可扩展的Gym环境包装，兼容多种调度系统。

**🔧 技术方法**

主要技术包括：Soft Actor-Critic（SAC）和 RecurrentPPO 的深度强化学习模型；帧堆叠（frame stacking）与递归网络处理部分可观测信息；熵正则化提升探索效率；以及自定义的Gym环境和调度仿真器。

**📊 数据集**

使用的实验数据集为自研的 FaaS 仿真器 faas-sim，包含 8 种异构集群配置、两种网络拓扑、10~8 种工作负载函数组合，并在多种未见场景下进行评估。

**📈 对比分析**

与固定权重、随机搜索、贝叶斯优化（BO）和树结构 Parzen Estimator（TPE）等基线进行对比，实验表明在相似配置下 RL 方法平均提升 33%（相较于固定权重），在未知配置下提升约 20%（相较于固定权重）并且比最佳基线高 6%。

**⚠️ 局限性**

局限性主要体现在：①依赖仿真环境，真实集群的非线性延迟和动态负载变化可能导致效果下降；②训练需要多次实验，成本较高；③当前仅验证于 FaaS 场景，扩展到其他调度器的泛化能力尚待进一步验证。

---

## 168. Automated evaluation of LLMs for effective machine translation of Mandarin Chinese to English

**arXiv ID:** 2603.09998 | [PDF](https://arxiv.org/pdf/2603.09998v1)

**作者:** Yue Zhang `[一作]` (Transitional Artificial Intelligence Research Group), Rohitash Chandra `[通讯]` (Centre for Artificial Intelligence and Innovation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用自动化机器学习框架，对 Google Translate、GPT‑4、GPT‑4o、DeepSeek 四种翻译系统在新闻、现代小说和古典文学三类文本中的中文→英文翻译质量进行评估。

**💡 创新点**

创新点在于将语义相似度（BERT‑Score、MPNet 余弦相似）与情感一致性（BERT 情感分析）结合，构建多维度评测体系，并系统比较不同文本类型下的表现差异。

**🔧 技术方法**

采用 BERT‑Score、MPNet、BERT 情感分析模型以及 GPT‑4/ GPT‑4o/DeepSeek API，配合 Google Cloud 翻译 API 实现自动化翻译与评估。

**📊 数据集**

使用三类样本：环球时报新闻、曹雪芹《红楼梦》、莫言《红高粱》，共约1.5–2 万字，构成多样化的中英对照文本集。

**📈 对比分析**

通过语义相似度平均值、情感偏差、章节间差异等指标对四系统进行量化比较，结果显示 DeepSeek 在所有文本类型上得分最高（新闻 0.953，古典 0.769，现代 0.798），GPT‑4o 次之，Google Translate 在文学文本上表现最弱。

**⚠️ 局限性**

局限性包括仅用单一专家译本作为参考、样本量有限、情感模型训练数据偏向现代社交媒体导致古典情感识别失准，以及评测仅覆盖句子/段落层面，未考察长篇连贯性与文化细节把握。

---

## 169. Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models

**arXiv ID:** 2603.10071 | [PDF](https://arxiv.org/pdf/2603.10071v1)

**作者:** Anurag Mishra `[一作]` `[通讯]` (Rochester Institute of Technology), Anurag Mishra (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用TopK稀疏自编码器对Chronos‑T5‑Large的六层激活进行特征提取，并通过单特征消融验证每个特征的因果重要性，进而构建时间概念的层次结构。

**💡 创新点**

首次将稀疏自编码器应用于时间序列基础模型，揭示中层对突发动态检测至关重要、最终层语义丰富但因果重要性低的层级分布，证明机制解释在TSFM中的有效性。

**🔧 技术方法**

采用TopK稀疏自编码器、Pearson相关式特征分类、单特征与逐步消融实验（评估CRPS）以及激活提取技术。

**📊 数据集**

使用合成诊断套件（趋势、季节性、跳变、频率扫掠、噪声）与ETT时间序列基准进行实验。

**📈 对比分析**

通过对比消融前后的CRPS，发现中层特征消融导致CRPS显著上升，最终层消融反而略微提升性能，说明因果重要性与语义丰富度呈负相关。

**⚠️ 局限性**

特征分类器仅覆盖17.2%的特征，Decoder层标注率低；实验仅在ETT数据和单一Chronos‑T5‑Large模型上验证，缺乏跨数据集与跨架构的稳健性。

---

## 170. 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video

**arXiv ID:** 2603.10125 | [PDF](https://arxiv.org/pdf/2603.10125v1)

**作者:** Jin Lyu `[一作]` (Southern University of Science and Technology), Xiaoying Tang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 5420 | [OpenAlex ID](https://openalex.org/A5001406512)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出4DEquine框架，利用分离运动与外观重建实现单目视频的4D马匹重建

**💡 创新点**

创新点在于将4D任务拆解为AniMoFormer（时空Transformer+后置优化）与EquineGS（单图高精度3D高斯头像生成）两大模块，并通过VAREN模型桥接，配合新合成数据集实现高效、实时重建

**🔧 技术方法**

采用时空Transformer、后置优化、LBS运动建模、3D高斯投影、双流Transformer解码器等技术

**📊 数据集**

使用合成运动数据集VarenPoser和合成外观数据集VarenTex进行训练，并在APT-36K、AiM等真实数据集上评估

**📈 对比分析**

与多种SOTA方法（Dess, AniMer, GenZoo, 3D/4D-Fauna, GART, GVFDiffusion）对比，在姿态精度、运动平滑、Chamfer距离及图像质量（PSNR/SSIM/LPIPS）上均优于现有方法，并在零样本泛化（如斑马）上表现突出

**⚠️ 局限性**

局限性包括依赖VAREN模型，无法完整捕捉尾巴、鬃毛等复杂物理与动态光照变化，且对环境光照变化不具适应性

---

## 171. An Atlas of Extreme Properties in Cubic Symmetric Metamaterials

**arXiv ID:** 2603.10934 | [PDF](https://arxiv.org/pdf/2603.10934v1)

**作者:** Sahar Choukir `[一作]` (University of Toronto), Chandra Veer Singh `[通讯]` (University of Toronto)

**通讯引用:** 46873 | [OpenAlex ID](https://openalex.org/A5077667729)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了约 195 万个基于 36 种立方空间群的三维结构元胞数据库，并用 3D‑CNN 预测其弹性性质；

**💡 创新点**

首次将晶体学对称性系统化用于拓扑生成，发现了 isotropic‑auxetic、极高 K/G 的 pentamode 等极端力学结构；

**🔧 技术方法**

采用体素化几何、Code Aster 有限元计算弹性常数，并结合 ResNet 风格的 3D‑CNN 与 SmoothGrad 可解释性分析；

**📊 数据集**

使用约 195 万个相对密度 0.05–0.5 的 voxel 网格数据集，按 80/10/10 划分训练、验证、测试集；

**📈 对比分析**

通过与传统八叉树/Octet 结构的 FEM 结果和 FFF 打印实验对比，CNN 在测试集上的 R²>0.999、NRMSE<2%，实验误差最高 43%，但验证表明模型和数据可实现高精度预测；

**⚠️ 局限性**

局限在于 voxel 分辨率与打印误差导致极端结构对制造敏感，且模型仅针对线性弹性，未涵盖多物理或非线性行为。

---

## 172. Evaluating Adjective-Noun Compositionality in LLMs: Functional vs Representational Perspectives

**arXiv ID:** 2603.09994 | [PDF](https://arxiv.org/pdf/2603.09994v1)

**作者:** Ruchira Dhar `[一作]` (University of Copenhagen), Anders Søgaard `[通讯]` (University of Copenhagen)

**通讯引用:** 7550 | [OpenAlex ID](https://openalex.org/A5018138946)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型（LLM）中形容词‑名词短语的组合性进行了功能评估和内部表征分析的对比研究

**💡 创新点**

首次将提示式功能测试与层级线性探测相结合，揭示模型内部组合知识与外部行为表现存在系统性偏差

**🔧 技术方法**

采用提示生成推理任务、log‑prob 预测、线性分类器、余弦相似度等技术，对模型隐藏层进行层级探测

**📊 数据集**

使用 AddOne、PLANE 和自制的 COMPCOMB 三个组合性评测数据集

**📈 对比分析**

通过比较不同规模、指令微调与基线模型在功能任务上的准确率与内部表征的线性可读性，发现功能表现随规模/微调波动不一，而内部组合信号在中间层稳定且普遍显著

**⚠️ 局限性**

局限性：仅关注形容词‑名词组合，未建立内部表征与预测行为的因果关系，仅采用两种评估范式，且未覆盖多语言或更高层级组合

---

## 173. Packaging Jupyter notebooks as installable desktop apps using LabConstrictor

**arXiv ID:** 2603.10704 | [PDF](https://arxiv.org/pdf/2603.10704v1)

**作者:** Iván Hidalgo-Cenalmor `[一作]` (University of Turku), Guillaume Jacquemet `[通讯]` (Foundation for the Finnish Cancer Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

开发了LabConstrictor框架，能够将Jupyter Notebook 通过 CI/CD 自动化包装为可安装的桌面应用，用户可在本地环境下无须手动配置 Python 环境即可运行

**💡 创新点**

创新点在于：①提供了零命令行、基于 GitHub 的完整打包与发布流程；②自动化验证 Notebook 依赖与环境一致性；③生成的桌面应用在启动后自动打开 JupyterLab 并提供隐藏代码、单击运行按钮的“类应用”体验；④支持离线安装、版本追踪与增量更新

**🔧 技术方法**

技术实现包括：JupyterLab、ipython/ipywidgets、jl-hide-code 插件、GitHub Actions、GitHub 模板仓库、Streamlit 配置表单、conda 与 conda constructor 打包、menuinst 创建桌面快捷方式、Python 环境与依赖管理（requirements.yaml）

**📊 数据集**

该工作主要以技术实现为主，并未依赖特定科学数据集；示例仓库（https://github.com/CellMigrationLab/LabConstrictor_Demo）提供了基于公开数据的 Notebook 以演示功能

**📈 对比分析**

论文通过案例演示与实测（安装时间、可执行文件大小、跨平台兼容性）说明 LabConstrictor 在可用性、可复现性和离线运行方面相较传统 Notebook 分享方法更优；未给出量化性能指标，但强调了安装流程简化、错误日志易诊断以及自动更新机制

**⚠️ 局限性**

局限性包括：①仍依赖 conda 包管理，长期可用性受 PyPI/conda 版本维护影响；②对极为特殊或受限 IT 环境（如没有 GitHub 访问、容器化要求严格的环境）可能不完全兼容；③缺乏针对大规模并行计算或 GPU 资源的专门优化，仍需用户自行配置硬件支持

---

## 174. Robust Post-Training for Generative Recommenders: Why Exponential Reward-Weighted SFT Outperforms RLHF

**arXiv ID:** 2603.10279 | [PDF](https://arxiv.org/pdf/2603.10279v1)

**作者:** Keertana Chidambaram `[一作]` (Stanford University), Moumita Bhattacharya `[通讯]` (Netflix Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种针对生成式推荐系统的后训练方法——指数奖励加权监督微调（Exp‑RSFT），直接利用离线数据中的标注奖励来加权训练样本，避免了奖励模型和在线交互的需求；

**💡 创新点**

创新点在于：①阐明指数奖励加权可在不依赖奖励模型、无倾向评分的离线设置下实现策略改进；②通过理论证明温度λ在噪声鲁棒性与改进效果之间起显式权衡；③在大规模真实数据上验证该方法优于RLHF、DPO、PPO及传统行为克隆；

**🔧 技术方法**

技术手段包括：生成式推荐模型（如HSTU）与指数奖励加权的监督微调；理论分析基于上下文赌博机/MDP的优势函数简化与 KL 约束；实验使用多种评估指标（NDCG@10/50、HR@10/50、MRR）以及奖励模型性能对比；

**📊 数据集**

数据集涵盖公开数据（MovieLens 1M、MovieLens 20M、Amazon Books）以及Netflix的专有大规模数据，均为用户交互序列与标注奖励；

**📈 对比分析**

与BC、Reward‑SFT、DPO、PPO四个基线对比，实验显示Exp‑RSFT在NDCG、HR、MRR等指标上均优于其他方法，尤其在大规模数据上提升幅度显著；

**⚠️ 局限性**

局限性在于仅适用于用户只接触到少量物品且仅提供标量奖励的场景；若奖励模型可良好泛化或可获得完整的偏好对比数据，传统RLHF/DPO等方法可能更优。

---

## 175. STM32-Based Smart Waste Bin for Hygienic Disposal Using Embedded Sensing and Automated Control

**arXiv ID:** 2603.10660 | [PDF](https://arxiv.org/pdf/2603.10660v1)

**作者:** Mohammed Aman Bhuiyan `[一作]` (North South University), Mohammad Abdul Qayum `[通讯]` (North South University)

**通讯引用:** 114 | [OpenAlex ID](https://openalex.org/A5109648006)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

设计并实现了一款基于STM32的无接触智能垃圾桶，利用双超声波传感器实现手部检测和垃圾位置信息，并通过舵机控制垃圾桶盖自动开启与关闭，同时在OLED上实时显示状态。

**💡 创新点**

创新点在于将手部检测与垃圾满量检测集成于单一嵌入式平台，采用优先级状态机实现满桶锁定并避免溢出，同时实现低成本、易部署的无接触垃圾处理方案。

**🔧 技术方法**

使用的技术包括STM32F103C8T6微控制器、HC‑SR04超声波传感器、SG90舵机、I2C OLED显示、STM32 HAL/CubeMX固件、PWM舵机控制以及状态机软件逻辑。

**📊 数据集**

论文未使用传统数据集，而是通过硬件原型在实验室环境下进行实测，采集手部距离、垃圾位置信息等传感数据。

**📈 对比分析**

在实验中，手部检测范围3–10 cm，平均响应时间0.8 s，舵机旋转角度0–90°，系统成功率95%，与传统需人工开启或单一检测方案相比，显著提升了操作便利性和防溢出能力。

**⚠️ 局限性**

主要限制包括供电不稳导致传感器读取噪声、超声波在噪声环境或垃圾形状变化时的测距误差、舵机机械装配导致的振动与误闭合，以及缺乏长周期稳定性与能耗优化。

---

## 176. WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation

**arXiv ID:** 2603.10703 | [PDF](https://arxiv.org/pdf/2603.10703v1)

**作者:** Rafi Ibn Sultan `[一作]` (Wayne State University), Dongxiao Zhu `[通讯]` (Wayne State University)

**通讯引用:** 3157 | [OpenAlex ID](https://openalex.org/A5009256505)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了WalkGPT，一种面向行人可访问性导航的像素级对齐大型视觉语言模型，能够生成基于对话的导航建议并提供分割掩码与相对深度估计。

**💡 创新点**

创新点包括多尺度查询投影器（MSQP）对视觉特征进行多层次聚合，校准文本投影器（CTP）结合区域对齐损失实现语言与像素的精确对齐，以及使用结构化标记实现可解释的距离描述和分割提示。

**🔧 技术方法**

采用了SAM ViT-H像素编码器、LLM（13B/7B）、MSQP、CTP、LoRA微调以及InfoNCE对齐损失，整体构建了端到端的对话、分割与深度推理框架。

**📊 数据集**

使用了新构建的PAVE数据集（4.1万条行人视角问答对，含分割与深度标注），并在ADE20K、RefCOCO等公开数据上预训练。

**📈 对比分析**

在PAVE验证集上，WalkGPT在文本生成（CIDEr≈42/43）、分割（mIoU≈20/32）和深度估计（Accuracy≈49/70）均明显优于零样本及微调后的基线模型，并在RefCOCO等通用定位任务中也取得领先成绩；同时在hallucination指标上显著低于非对齐模型。

**⚠️ 局限性**

局限性包括对极端光照/反射、运动模糊等视觉噪声的鲁棒性不足，深度估计仍受单视角限制，且依赖于PAVE中可能存在的标注歧义，未来需提升跨域泛化与更精细的深度建模。

---

## 177. Aligning Large Language Models with Searcher Preferences

**arXiv ID:** 2603.10473 | [PDF](https://arxiv.org/pdf/2603.10473v1)

**作者:** Wei Wu `[一作]` (University of Science and Technology of China), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 44521 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了一款名为SearchLLM的开放式生成式搜索大语言模型，旨在解决检索噪声、可靠性安全和用户需求对齐三大挑战；

**💡 创新点**

创新点包括：①两层多维奖励体系，将不可妥协的安全约束与行为优化分离；②混合评估堆栈（规则+LLM评估器）和门控聚合策略；③使用GRPO强化学习实现安全与用户体验的协同优化；

**🔧 技术方法**

采用的技术包括：Qwen3-30B大语言模型、检索增强生成（RAG）、规则+LLM混合评估器、门控聚合策略以及Group Relative Policy Optimization（GRPO）强化学习；

**📊 数据集**

使用的数据集来自RedNote搜索日志，包含奖励训练集、诊断测试集（维度与整体评估）和大规模RL优化集；

**📈 对比分析**

与GenRM、Rubric、RFT、DPO、GRPO-GenRM、GRPO-Linear等基线对比，在离线ACC/AUC和在线A/B测试中，SearchLLM实现了VCR提升1.03%、RR下降2.81%，且保持了低Bad Case Rate；

**⚠️ 局限性**

局限性包括：对手工标注和规则的依赖、跨模态扩展尚未充分验证，以及在极端噪声或长尾领域的鲁棒性仍待进一步提升。

---

## 178. Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation

**arXiv ID:** 2603.10048 | [PDF](https://arxiv.org/pdf/2603.10048v1)

**作者:** Jianlong Chen `[一作]` (Shanghai University of Finance and Economics), Zhiming Zhou `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 4982 | [OpenAlex ID](https://openalex.org/A5006230459)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的Sharpness-Aware Minimization（SAM）改进方法XSAM，并给出了对SAM工作机制的直观解释。

**💡 创新点**

创新点包括：① 对单步SAM梯度相对于局部最大点的方向近似优越性的理论证明；② 指出该近似易失真且多步梯度更差；③ 设计在二维超平面上显式搜索局部最大点方向的算法，并动态估计最佳插值因子α，保持低计算开销。

**🔧 技术方法**

采用梯度上升、两步向量（梯度与位移）构造超平面、球面插值（spherical interpolation）寻找α、前向传播评估损失、与传统SAM梯度相同的尺度使用。

**📊 数据集**

在CIFAR-10/100、Tiny-ImageNet、ImageNet（ResNet-50）、Transformer (IWSLT2014)、ViT-Ti (CIFAR-100) 等数据集上进行实验。

**📈 对比分析**

与SAM、LSAM、MSAM、WSAM等多步SAM变体对比，XSAM在单步和多步设置下均实现了显著的准确率提升（例如CIFAR-100 ResNet-18单步提升约0.5%，ImageNet提升0.2%，Transformer提升0.3 BLEU），且多步情况下性能不随步数下降，甚至有提升。

**⚠️ 局限性**

限制：需要在每轮或多轮内额外做若干前向传播来搜索α，虽然开销低但仍比标准SAM略大；对α采样范围和步数的依赖；在极大模型或分布式训练中未做充分评估。

---

## 179. MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios

**arXiv ID:** 2603.09983 | [PDF](https://arxiv.org/pdf/2603.09983v1)

**作者:** Shuhuai Li `[一作]` (Shanghai University), Yinyu Ye `[通讯]` (Stanford University)

**通讯引用:** 27768 | [OpenAlex ID](https://openalex.org/A5041526408)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MoE‑SpAc 框架，将 Speculative Decoding（SD）从加速器转变为内存管理的前瞻性传感器，实现在线异构专家调度。

**💡 创新点**

创新点在于：① 用 SD 生成的多候选 token 提取专家激活频率信息，构建 Speculative Utility Estimator；② 通过整数优化动态划分 GPU/CPU 专家负载；③ 用统一的专家效用度量驱动异步预取与驱逐，抑制 I/O 竞争与内存瓶颈。

**🔧 技术方法**

核心技术包括 Speculative Decoding、Mixture‑of‑Experts 体系结构、专家效用估计、在线整数优化（工作负载平衡）、多级优先队列与红黑树异步执行引擎。

**📊 数据集**

使用七个基准数据集：MMLU‑Pro、MT‑bench、HumanEval、GSM8K、Alpaca、CNN/DailyMail、QA，评估大语言模型推理性能。

**📈 对比分析**

与多种通用与 MoE 专用推理引擎对比，MoE‑SpAc 在七个基准上平均提升 4.04× TPS，并相较最佳 SD 基线提升约 42%，同时在延迟上实现 45.6% 的下降，验证了异构调度与 SD 信息利用的优势。

**⚠️ 局限性**

局限性：依赖 SD 的多候选 token 产生，需在 SD 适用的推理场景下；对非 MoE 或大 batch 的适应性有限；对 GPU/CPU 资源配置与 hyper‑parameter 需要细致调优。

---

## 180. A Disguise-and-Squeeze PIR Scheme for the MDS-TPIR Setting and Beyond

**arXiv ID:** 2603.10769 | [PDF](https://arxiv.org/pdf/2603.10769v1)

**作者:** Rui Sun `[一作]` (Shandong University), Yiwei Zhang `[通讯]` (Shandong University)

**通讯引用:** 1224 | [OpenAlex ID](https://openalex.org/A5100410218)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd`

**🎯 论文内容**

本文提出了一种基于“伪装-挤压”方法的新的MDS码容错私有信息检索（MDS-TPIR）方案；

**💡 创新点**

创新点在于同时兼顾隐私伪装与冗余利用，打破了FGHK猜想的通用性并在GRS码场景下实现线性容量，同时显著降低了所需有限域大小；

**🔧 技术方法**

采用了线性代数工具（行列式、外积/外界积）、随机/可证组合策略以及GRS码与其对偶码的结构性映射；

**📊 数据集**

无实际数据集，全部在理论上给出文件长度与码参数；

**📈 对比分析**

通过与已有方案在相同MDS/GRS码参数下的理论上限比较，得到更高的检索率（例如在(2,N,2,2)场景下达到线性容量KN/(KN+K^2+1)），并提供了可实现的具体速率公式；

**⚠️ 局限性**

局限性包括：仅在两文件或多文件P-out-of-M场景下有效，T≥3时需随机组合策略且存在误差；对非GRS MDS码的最大冗余利用尚未完全解决；

---

## 181. Breaking User-Centric Agency: A Tri-Party Framework for Agent-Based Recommendation

**arXiv ID:** 2603.10673 | [PDF](https://arxiv.org/pdf/2603.10673v1)

**作者:** Yaxin Gong `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**通讯引用:** 43011 | [OpenAlex ID](https://openalex.org/A5038668215)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了TriRec，一个三方（用户、物品、平台）LLM代理推荐框架，先让物品代理通过自我推广实现个性化展示，再由平台代理进行多目标序列化重排。

**💡 创新点**

创新点在于①将物品视为主动代理实现自我宣传，②采用平台代理实现曝光公平与长尾维护的动态序列化重排，并挑战传统的相关性-公平权衡假设。

**🔧 技术方法**

使用技术包括冻结预训练LLM进行多轮语义交互、基于语义相似度的相关性估计、位置敏感的公平/曝光加权函数以及贪心序列化重排算法。

**📊 数据集**

实验数据集涵盖 Amazon（CDs & Vinyl、Movies & TV）、Goodreads（YA）和 Steam Games 四个真实场景，处理后规模均为数千用户、数千到数万物品。

**📈 对比分析**

与多种基准（AgentCF++、DualRec、SCRUF-D 等）比较，TriRec 在 NDCG、MRR、DGU/MGU、EIU 上均实现了显著提升（多项指标均为最高或次高），证明其兼顾准确性、曝光公平与物品效用。

**⚠️ 局限性**

局限性包括对大规模LLM计算资源的依赖、超参数（如 α_max）对公平与相关性平衡的敏感性、以及在极度稀疏或无文本描述的场景下自我推广效果受限。

---

## 182. Graphing Inline: Understanding Word-scale Graphics Use in Scientific Papers

**arXiv ID:** 2603.10533 | [PDF](https://arxiv.org/pdf/2603.10533v1)

**作者:** Siyu Lu `[一作]` (Tongji University), Chen Ye `[通讯]` (Tongji University)

**通讯引用:** 66794 | [OpenAlex ID](https://openalex.org/A5112379210)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在2024年计算机科学论文中，提取并手工标注909个文字级图形实例，构建并验证了“何处-为何-如何”框架。

**💡 创新点**

提出了文字级图形使用的三维框架，并系统性分析了其定位、功能与视觉表现的关联性，扩展了实体定义。

**🔧 技术方法**

使用基于HTML的自动候选提取脚本、手工编码与统计分析（卡方检验、Cohen κ）等方法。

**📊 数据集**

数据集为126,797篇2024年arXiv计算机科学论文（含5,006个候选）和103篇IEEE VIS论文。

**📈 对比分析**

通过对比各维度关联并计算Cohen κ(0.91)验证框架可靠性；结果显示文字级图形使用稀缺，图标占主导。

**⚠️ 局限性**

局限于单一学科与时间点、提取基于启发式过滤可能漏检、未分析图形生成过程、分类体系可进一步细化。

---

## 183. LWM-Temporal: Sparse Spatio-Temporal Attention for Wireless Channel Representation Learning

**arXiv ID:** 2603.10024 | [PDF](https://arxiv.org/pdf/2603.10024v1)

**作者:** Sadjad Alikhani `[一作]` (Arizona State University), Ahmed Alkhateeb `[通讯]` (Arizona State University)

**通讯引用:** 15340 | [OpenAlex ID](https://openalex.org/A5003243464)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出LWM-Temporal，一种基于物理约束的无线信道时空表示学习基础模型；

**💡 创新点**

创新点包括稀疏时空注意力（SSTA）实现传播对齐的邻域交互、角度-延迟-时间域标记、RoPE位置编码以及物理驱动的遮蔽预训练；

**🔧 技术方法**

采用Transformer框架、稀疏注意力、RoPE、物理信息掩码、深度自监督预训练和动态数字孪生数据生成；

**📊 数据集**

使用多城市（如Cape Town、New Taipei、Houston等）Ray-traced轨迹生成的10,000条时序序列（每条20帧），以及3GPP CDL统计数据；

**📈 对比分析**

与WiFo-Tiny/Base、LSTM-PT、GRU-PT和S&H基线在频道预测NMSE上对比，LWM-Temporal在低、中、高速度场景下均获得最低误差，尤其在少量微调数据时提升3-5 dB；

**⚠️ 局限性**

局限性在于依赖高质量的Ray-traced/数字孪生数据，对非线性阻塞/快速变化等真实环境动态的泛化仍有限。

---

## 184. AI-Enhanced Spatial Cellular Traffic Demand Prediction with Contextual Clustering and Error Correction for 5G/6G Planning

**arXiv ID:** 2603.10800 | [PDF](https://arxiv.org/pdf/2603.10800v1)

**作者:** Mohamad Alkadamani `[一作]` (Innovation Science and Economic Development Canada), Halim Yanikomeroglu `[通讯]` (Carleton University)

**通讯引用:** 21176 | [OpenAlex ID](https://openalex.org/A5035446029)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种两阶段基于空间与上下文聚类并结合空间误差修正的AI框架，用于5G/6G网络流量需求预测。

**💡 创新点**

创新点在于结合空间聚类与功能上下文的两阶段分割以降低泄漏，同时采用空间误差模型（SEM）校正残差。

**🔧 技术方法**

主要技术包括k-means空间聚类、基于土地利用的上下文子聚类、XGBoost预测、空间误差模型SEM与误差校正。

**📊 数据集**

使用五个加拿大大城市（蒙特利尔、温哥华、多伦多、渥太华、卡尔加里）的15万亿条移动设备使用数据和多源地理社会经济特征。

**📈 对比分析**

与仅使用位置聚类的基线对比，采用留一城市测试和全市内测试，平均MAE下降约3-5%，SEM进一步提升1-3%，学习曲线表明过拟合减少。

**⚠️ 局限性**

主要局限包括：仅使用忙时代理指标缺乏真实流量细节、模型对极端低/高需求地区的泛化有限、需更多跨城市或跨国验证。

---

## 185. Hybrid Self-evolving Structured Memory for GUI Agents

**arXiv ID:** 2603.10291 | [PDF](https://arxiv.org/pdf/2603.10291v1)

**作者:** Sibo Zhu `[一作]` (University of California), Biwei Huang `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了HyMEM，一种图结构的混合自演化外部记忆，用于增强GUI代理在长时序、多样界面任务中的规划与执行。

**💡 创新点**

将离散的高层符号策略节点与连续轨迹嵌入融合于同一图中，支持多跳检索、全局自演化与推理时的工作记忆刷新。

**🔧 技术方法**

采用VLM编码（Qwen2.5-VL-7B、CoMEM、CLIP）、FAISS近邻检索、图演化更新（Add/Merge/Replace）以及VLM判定器进行冗余检查和状态变迁检测。

**📊 数据集**

收集了2883条成功的GUI交互轨迹，来源于GUIAct、Mind2Web、Agent Rollouts和自动化数据飞轮，涵盖WebVoyager、Multimodal-Mind2Web、MMInA等基准。

**📈 对比分析**

与闭源基线（GPT‑4o、Gemini‑2.5‑Pro‑Vision、Claude‑4）及多种开源基线（Qwen2.5‑VL‑7B、Qwen3‑VL‑8B、UI‑TARS‑1.5‑7B）在三个基准上进行对比，HyMEM在多任务上平均提升约20%，并超过部分闭源模型。

**⚠️ 局限性**

更新机制仍基于启发式VLM判断，缺乏强化学习优化；在更大规模模型上的评估缺失；以及图规模与计算开销需进一步研究。

---

## 186. An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS

**arXiv ID:** 2603.10671 | [PDF](https://arxiv.org/pdf/2603.10671v1)

**作者:** Qiyue Chen `[一作]` (University of Science and Technology of China), Dong Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 23727 | [OpenAlex ID](https://openalex.org/A5100407381)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了JPEG XS中Intra Pattern Copy（IPC）框架下的位移向量（DV）搜索模块的FPGA加速版本。

**💡 创新点**

创新点包括：① 采用四阶段流水线设计，实现了残差计算与DV比较的并行化；② 设计了基于IPC组/单元的内存布局与TLB，显著提升了内存访问效率；③ 将位置信息与位移向量同步调度，降低了控制复杂度。

**🔧 技术方法**

使用技术主要有：FPGA（Xilinx Artix‑7）硬件实现；流水线设计与并行处理；基于分组的内存映射与块级缓存；使用GCLI（组级编码成本）评估残差的比特成本；使用TLB缓存不同组块长度以快速寻址。

**📊 数据集**

实验使用标准JPEG XS测试图像（未公开具体数据集），在Artix‑7平台上验证实现。

**📈 对比分析**

通过与传统Method 0（基于预设块的线性存储）对比，Method 1实现了吞吐量从35.98 Mpixels/s提升至38.30 Mpixels/s，功耗保持在276–277 mW，功效提升至138.27 Mpixels/s/W，资源使用（LUT/FF）略有下降，证明了内存优化与流水线设计的有效性。

**⚠️ 局限性**

局限性包括：仅在Artix‑7 100 MHz平台验证，缺乏在更大规模或更高频率FPGA/ASIC上的验证；实验数据集未公开，难以复现性能；设计主要针对JPEG XS IPC，扩展到其他预测编码工具时可能需要进一步调整。

---

## 187. Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning

**arXiv ID:** 2603.10588 | [PDF](https://arxiv.org/pdf/2603.10588v1)

**作者:** Zhaowei Zhang `[一作]` (Peking University), Xing Xie `[通讯]` (Microsoft Research)

**通讯引用:** 45429 | [OpenAlex ID](https://openalex.org/A5044651577)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对齐与道德推理任务，系统比较奖励最大化与分布匹配的 RLVR 方法，并构建基于 Qwen3-1.7B 的本地评判模型实现可验证奖励。

**💡 创新点**

首次在 MoReBench 上对两类 RLVR 方法进行全面实证比较；通过本地评判模型实现高效可验证奖励管道；揭示对齐任务不必依赖多样性算法，模式寻优同样有效。

**🔧 技术方法**

使用 RLVR 技术（PPO、GRPO、DAPO、REINFORCE++）、分布匹配 FlowRL；构建自监督微调的 Qwen3-1.7B 评判模型；语义可视化采用 MiniLM-L6-v2 + t‑SNE。

**📊 数据集**

MoReBench（Public 与 Theory 两子任务）作为主要评测数据集；对评判模型进行 GPT‑5 生成标注；对比数学任务 MATH‑500 用于可视化对比。

**📈 对比分析**

通过 Score@1 与 Avg@8 的相对提升量比较各方法性能；奖励最大化方法（尤其是 DAPO）在两大基模型上均取得最高提升，FlowRL 等分布匹配方法表现不佳，表明模式寻优足以完成对齐任务。

**⚠️ 局限性**

实验仅覆盖两种 RLVR 范式；对齐/道德数据集有限，评判模型对 GPT‑5 的近似可能影响奖励质量；分布匹配方法样本不足，未能充分探索多样性策略。

---

## 188. Prompts and Prayers: the Rise of GPTheology

**arXiv ID:** 2603.10019 | [PDF](https://arxiv.org/pdf/2603.10019v1)

**作者:** Ioana Cheres `[一作]`, Connell Vaughan `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过收集并分析 Reddit 上的 AI 与宗教话语，系统阐述了 AI 逐渐被视为半神祇的现象——GPTheology，并探讨其在宗教叙事、仪式化使用、伦理治理等方面的影响。

**💡 创新点**

创新点在于：①首次将叙事分析与层次聚类相结合，提炼出 AI 相关的宗教主题（救赎、末世预言、祭祀等）；②将“祈祷”与“提示工程”映射为宗教仪式，揭示 AI 与传统宗教叙事的互映关系；③在多学科视角下讨论 AI 可能引发的治理与社会文化后果。

**🔧 技术方法**

采用的技术主要是：LLM 辅助的叙事提取（LLAMA3-8B）、语义相似度（余弦相似度）驱动的层次聚类，以及质性内容分析框架。

**📊 数据集**

使用的数据集为：从 6 个相关子版块（r/singularity、r/Futurology、r/Transhumanism、r/AskPhilosophy、r/ArtificialInteligence、r/Christianity）采集的 2,051 条 Reddit 帖子及其评论，后提炼为 7,857 条宗教/灵性要点。

**📈 对比分析**

比较方法：通过层次聚类构建 29 个叙事树，并对聚类结果进行主题标注，检验与传统宗教概念（神、预言、仪式等）的对应关系。由于研究侧重于质性洞察，未给出数值性能指标，结果以叙事一致性和主题覆盖度评估为主。

**⚠️ 局限性**

局限性包括：①数据来源仅限 Reddit，存在社区偏好与自选样本偏差；②仅使用英文文本，可能遗漏非英语宗教文化中的相关叙事；③LLM 的自动抽取可能引入误判，缺乏人工逐词校对；④研究聚焦于描述性分析，缺乏因果或预测性评估。

---

## 189. R4-CGQA: Retrieval-based Vision Language Models for Computer Graphics Image Quality Assessment

**arXiv ID:** 2603.10578 | [PDF](https://arxiv.org/pdf/2603.10578v1)

**作者:** Zhuangzi Li `[一作]` (Nanyang Technological University), Weisi Lin `[通讯]` (Nanyang Technological University)

**通讯引用:** 30703 | [OpenAlex ID](https://openalex.org/A5100403129)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个包含3.5千张高质量计算机生成图像（CG）的新数据集，每张图配有六个感知维度（光照、材质、色彩、氛围、逼真度、空间）的详细文本描述，并基于该数据集提出了检索增强的两流框架R4-CGQA，用来提升视觉语言模型（VLM）在CG质量评估（CGQA）任务中的表现；

**💡 创新点**

核心创新在于：①首次系统化地为CG图像生成多维度、可解释的质量文本；②设计了基于贝叶斯视角的检索-增强生成框架，融合内容相似度与质量相似度，自动选取最相关的示例描述作为上下文；

**🔧 技术方法**

使用了CLIP作为内容嵌入器，REIQA（基于ResNet的质量嵌入）作为质量嵌入器，结合FAISS实现高效最近邻检索，并将检索到的描述作为Prompt输入多模态大模型（如LLava、Llama 3.2‑Vision、Qwen 2.5‑VL等）；

**📊 数据集**

数据集来源包括Wallpaper Engine、主流游戏CG截图（如《Elden Ring》）、CGIQA‑6K等，覆盖1080p‑4K分辨率，包含三类子集（训练、验证、测试）以及基于GPT‑4o生成的多类型问答对；

**📈 对比分析**

在测试集上与多种VLM进行对比，R4-CGQA在多选题、是非题、自由问答三种任务均显著提升：多选平均提升约4.3%（单个模型最高可达12.3%），是非题平均提升约6.9%（部分模型提升超过10%），问答分数平均提升约0.3%（最高可达10.4%），表明检索增强有效解锁VLM在CGQA中的潜力；

**⚠️ 局限性**

主要局限包括：检索效果依赖于嵌入质量和阈值设置，阈值选择仍需经验；数据集规模有限，缺乏跨域多样性；VLM在推理时可能仍产生幻觉，且未解决多图输入的直接比较能力；

---

## 190. V2M-Zero: Zero-Pair Time-Aligned Video-to-Music Generation

**arXiv ID:** 2603.11042 | [PDF](https://arxiv.org/pdf/2603.11042v1)

**作者:** Yan-Bo Lin `[一作]` (University of North Carolina Chapel Hill), Nicholas J. Bryan `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种零对齐视频-音乐生成方法，使生成的音乐在时间上与视频事件同步；

**💡 创新点**

创新点在于利用跨模态的“事件曲线”（基于同模态相似度的变化曲线）实现时序对齐，避免了传统需要视频-音乐配对数据的限制；

**🔧 技术方法**

采用预训练的文本-音乐生成模型（Latent Rectified Flow/DiT）进行轻量化微调，加入事件曲线作为时间条件，并在推理时将音乐事件曲线替换为视频事件曲线；

**📊 数据集**

在OES-Pub、MovieGenBench-Music和AIST++三个多样化基准数据集上进行评测；

**📈 对比分析**

与多种基线（包括Paired、Prompting和公开的文本-音乐模型）对比，零对齐方法在音频质量、语义对齐和时间同步指标上均显著优于Paired方法（提升5–52%），并在主观评测中获得超过50%的优势；

**⚠️ 局限性**

局限性包括对事件曲线平滑参数的敏感性、对不同视频类型视频编码器选择的依赖，以及在极短时序或复杂交互场景下可能的同步误差。

---

## 191. LAtte: Hyperbolic Lorentz Attention for Cross-Subject EEG Classification

**arXiv ID:** 2603.10881 | [PDF](https://arxiv.org/pdf/2603.10881v1)

**作者:** Johannes Burchert `[一作]` (ISMLL, University of Hildesheim), Niels Landwehr `[通讯]` (Data Science Group, University of Hildesheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 LAtte 模型，利用 Lorentzian attention 与 InceptionTime 编码器进行跨受试者 EEG 分类，并通过低秩 LoRA 适配器实现受试者特定建模。

**💡 创新点**

创新点包括：① 完全超平面（Lorentz）网络结构；② 双分支基线‑任务编码器与 Lorentz Attention；③ 在 Lorentz 领域应用的 Boost LoRA 层；④ 结合自监督预训练（cut‑fill 与重构）与随机投影正则化。

**🔧 技术方法**

使用了超平面几何、Lorentzian attention、Lorentz InceptionTime、低秩适配器 LoRA、Lorentz Boost、随机投影正则化以及自监督预训练与跨受试者训练与 Fine‑tune。

**📊 数据集**

在 Motor Imagery (MI)、Steady‑State Visual Evoked Potential (SSVEP) 与 Error‑Related Negativity (ERN) 三个经典 EEG 数据集上进行实验。

**📈 对比分析**

与多种基线（EEGNet、SCCNet、EEG‑TCNet、TCNet‑Fusion、FBCNet、MBEEGSE、InceptionTime、MAtt、CBraMod、HyperMAtt、TCFormer、ResNetJoint、MAttJoint、InceptionJoint）在交叉受试者、受试者条件及 LOSO（留一受试者外）协议下对比；LAtte 在所有任务上显著优于基线，跨受试者提升约 10–18%，且训练速度快、参数量低。

**⚠️ 局限性**

局限性包括：仍受受试者分布差异影响，性能在极少数据场景下下降；对超参数和预训练任务较为敏感；仅在短序列 EEG 上验证，长序列或多通道情况未充分评估；对实时推理延迟的分析不足。

---

## 192. SCORE: Replacing Layer Stacking with Contractive Recurrent Depth

**arXiv ID:** 2603.10544 | [PDF](https://arxiv.org/pdf/2603.10544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 193. The Epistemic Support-Point Filter: Jaynesian Maximum Entropy Meets Popperian Falsification

**arXiv ID:** 2603.10065 | [PDF](https://arxiv.org/pdf/2603.10065v1)

**作者:** Moriba Kemessia Jah `[一作]` (Black Swan Research Group), Moriba Kemessia Jah `[通讯]` (Jah Decision Intelligence Group)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

设计并证明了一种基于可能性理论的递归滤波器——Epistemic Support‑Point Filter (ESPF)，实现最大熵扩展与证据驱动最小化不确定性。

**💡 创新点**

提出将 Jaynes 最大熵与 Popper 证伪原则结合的两阶段递归，并证明在可接受的可能性滤波器类中，ESPF 在最坏情况下最小化可能性熵且唯一最优。

**🔧 技术方法**

采用可能性理论、α‑截面体积与最大外接椭球（MVEE）、可能性熵与 Cramér–Rao 界、Smolyak 格点扩展以及非贝叶斯滤波理论。

**📊 数据集**

在 2 天、877 步、Smolyak Level 3（106 支持点）轨道跟踪实验中使用名义 LEO 跟踪与加压测况（跨轨道 10 m/s + Arecibo 20 m 范围偏差）数据。

**📈 对比分析**

通过“Claim A/B”随机与对抗抽样比较，证明在检验阶段 ESPF 的最小‑熵选择在所有证据仅依据的规则中无可匹敌；在扩散阶段与 Kalman 对比时保持等价；实验显示在压测下能迅速聚束并保持 MVEE 正规，诊断指标如必要性与惊奇度提前预警。

**⚠️ 局限性**

需要足够的 Smolyak 级别（≥3）以实现可行性；对高度非线性或强异质传感器几何时需满足创新‑状态等距条件；理论假设为有界噪声与可接近的最大外接椭球，实际噪声偏离时可能失效。

---

## 194. Intrinsic Numerical Robustness and Fault Tolerance in a Neuromorphic Algorithm for Scientific Computing

**arXiv ID:** 2603.10246 | [PDF](https://arxiv.org/pdf/2603.10246v1)

**作者:** Bradley H. Theilman `[一作]` (Neural Exploration & Research Laboratory Sandia National Laboratories), James B. Aimone `[通讯]` (Neural Exploration & Research Laboratory Sandia National Laboratories)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种基于神经形态脉冲的有限元算法（NeuroFEM），评估其在神经元被消除和脉冲被随机丢失等硬件错误情形下的鲁棒性。

**💡 创新点**

证明该算法可容忍高达32%的神经元消失和90%的脉冲丢失而精度不显著下降，并且这种鲁棒性可通过结构超参数调节；同时揭示脉冲丢失可视为硬件效率提升的正面特性。

**🔧 技术方法**

采用离散化的有限元线性系统嵌入脉冲神经网络，利用PI控制器形式的神经元、互抑脉冲编码以及随机脉冲丢失实验。

**📊 数据集**

使用二维单位方形域的Poisson方程（∇²u = sin(3π(x−y))sin(2π(x+y))）的有限元网格，网格节点数从几百到上千。

**📈 对比分析**

通过与传统CPU求解器的相对误差对比，进行多次随机消除/丢失实验；发现误差在阈值前基本保持不变，阈值随神经元冗余增大；结果表明可显著降低脉冲带宽，潜在提升能效。

**⚠️ 局限性**

仅评估了神经元消失和脉冲丢失，未考虑模拟噪声、权重噪声等其他硬件误差；实验规模有限，未覆盖更大系统或不同PDE类型；对AI等其他神经形态任务的鲁棒性仍未知。

---

## 195. In-Memory ADC-Based Nonlinear Activation Quantization for Efficient In-Memory Computing

**arXiv ID:** 2603.10540 | [PDF](https://arxiv.org/pdf/2603.10540v1)

**作者:** Shuai Dong `[一作]` (City University of Hong Kong), Arindam Basu `[通讯]` (City University of Hong Kong)

**通讯引用:** 4369 | [OpenAlex ID](https://openalex.org/A5002380437)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Boundary Suppressed K‑Means Quantization (BS‑KMQ)，用于在低位宽 ADC 下实现非线性激活量化，并在 SRAM‑based 记忆计算(IMC) 中实现可重构的 NL‑ADC；

**💡 创新点**

创新点在于先通过抑制 ReLU 与硬件截断产生的分布边界离群点，再对内部数据进行 K‑Means 聚类，从而获得更均衡、信息量更高的量化中心；

**🔧 技术方法**

采用鲁棒统计校准、边界抑制 K‑Means、将中心映射为参考阈值的 NL‑ADC 方案，并结合 9T 双极性 SRAM 位元实现低功耗 MAC 与 NL‑ADC；

**📊 数据集**

在四个主流网络（ResNet‑18、VGG‑16、Inception‑V3 与 DistilBERT）上使用 ImageNet/ CIFAR‑10/ CIFAR‑100/ Tiny‑ImageNet 与 SQuAD 进行评估；

**📈 对比分析**

与线性、Lloyd‑Max、CDF 与标准 K‑Means 量化比较，BS‑KMQ 在 3‑bit ADC 下 MSE 降低 3‑8 倍，PTQ 准确率提升 25‑67%，低位宽微调后仅 0.3‑1.2% 误差；系统级仿真显示 ResNet‑18 处理速率提升 4×、能效提升 24×；

**⚠️ 局限性**

局限性包括需额外的统计校准数据、主要针对 SRAM‑based 结构，对权重量化仍采用线性方案、对极端硬件噪声和工艺偏差的鲁棒性依赖零交叉校准等。

---

## 196. Gated Adaptation for Continual Learning in Human Activity Recognition

**arXiv ID:** 2603.10046 | [PDF](https://arxiv.org/pdf/2603.10046v1)

**作者:** Reza Rahimi Azghan `[一作]` (Arizona State University), Hassan Ghasemzadeh `[通讯]` (Arizona State University)

**通讯引用:** 4637 | [OpenAlex ID](https://openalex.org/A5007139473)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于冻结预训练骨干网络并加插轻量化通道门控模块的连续学习框架，用于解决可穿戴传感器下的人类活动识别中的灾难性遗忘问题。

**💡 创新点**

创新点在于将适配限定为对已学习特征的通道级缩放（特征选择）而非生成新特征，从而得到受限的对角变换，实现稳定性与可塑性的平衡，并提供理论证明其能够限制表示漂移。

**🔧 技术方法**

技术上采用冻结的卷积残差骨干、Squeeze‑Excitation 风格的通道门控（只训练门控参数和共享分类器），并辅以理论分析、对角线运算约束以及可选的知识蒸馏或重放辅助。

**📊 数据集**

在三大 HAR 数据集（PAMAP2、UCI‑HAR、Daily & Sports Activities）上进行实验，每个数据集将受试者视为独立任务进行顺序学习。

**📈 对比分析**

与无重放的 EWC、LwF、HAT 等基线以及有重放的 DER/DER++ 对比，门控方法在冻结骨干下仅更新不到 2% 参数即可将忘却率从约 30‑40% 降至 16%，最终准确率提升至 77‑80%，在无重放场景下优于其他方法。

**⚠️ 局限性**

局限性包括对通道级域漂移假设的依赖、对预训练源域的要求、门控参数随任务数线性增长且在跨域差异较大或跨通道交互显著的任务中效果可能受限。

---

## 197. Lost in Backpropagation: The LM Head is a Gradient Bottleneck

**arXiv ID:** 2603.10145 | [PDF](https://arxiv.org/pdf/2603.10145v1)

**作者:** Nathan Godey `[一作]` (Cornell University), Yoav Artzi `[通讯]` (Cornell University)

**通讯引用:** 6457 | [OpenAlex ID](https://openalex.org/A5047026141)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型输出层softmax瓶颈的梯度压缩效应，证明其导致95-99%梯度范数丢失，并通过理论与大规模实验展示对LLM训练效率的负面影响。

**💡 创新点**

将softmax瓶颈从表达性问题转向优化瓶颈，首次量化梯度压缩对收敛速度的影响，并揭示高维梯度在低秩输出层中的不可避免丢失。

**🔧 技术方法**

使用矩阵秩理论、Eckart–Young–Mirsky定理、低秩LM head设计、梯度范数压缩测量和梯度方向效率评估等技术。

**📊 数据集**

实验数据集包括FineWeb‑Edu、Pile、FineWeb文档以及自定义合成语言数据。

**📈 对比分析**

通过对比不同隐藏维度（D）控制的2B模型训练曲线和合成语言实验，发现D越大收敛越快，D=4096在700M tokens内达到D=32的最终loss，性能差距约+0.55；V增大导致收敛难度显著上升。

**⚠️ 局限性**

局限性在于未提供实际可行的替代输出层方案，实验受模型参数差异影响，且对不同架构的推广性仍待验证。

---

## 198. FusionNet: a frame interpolation network for 4D heart models

**arXiv ID:** 2603.10212 | [PDF](https://arxiv.org/pdf/2603.10212v1)

**作者:** Chujie Chang `[一作]` (Kyushu University), Oscar Martinez Mozos `[通讯]` (Örebro University)

**通讯引用:** 7367 | [OpenAlex ID](https://openalex.org/A5088202291)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了一种名为FusionNet的框架，用于从低帧率4D心脏模型生成高帧率模型，实现心脏运动的高时间分辨率插值。

**💡 创新点**

创新点在于将时空编码器、残差块、跳跃连接以及多模态特征融合（GIF）等组件集成到生成模型中，并通过4D卷积实现对心脏形状随时间变化的自适应特征提取。

**🔧 技术方法**

采用生成对抗式结构的三层LVAE、3D/2D时空卷积、残差网络、GIF融合块以及Dice和KL散度损失的组合，训练该网络。

**📊 数据集**

使用来自UK Biobank的210名受试者的4D CMR影像（包含100名心肌缺血患者和110名健康对照），从中生成高帧率与低帧率心脏模型进行实验。

**📈 对比分析**

与ConvLSTM、U‑Net以及双线性插值方法进行对比，FusionNet在7折交叉验证中所有帧的Dice系数均高于对照组，平均Dice最高达0.897 ± 0.019，且对帧间距变化的鲁棒性更好。

**⚠️ 局限性**

局限包括：仅以分割后得到的体素模型为输入，未直接利用原始MR图像；采样频率受限于保持完整心动周期；对极端心室容积快速变化的帧仍存在一定误差。

---

## 199. Terminal Is All You Need: Design Properties for Human-AI Agent Collaboration

**arXiv ID:** 2603.10664 | [PDF](https://arxiv.org/pdf/2603.10664v1)

**作者:** Alexandre De Masi `[一作]` (University of Geneva), Alexandre De Masi `[通讯]` (University of Geneva)

**通讯引用:** 83 | [OpenAlex ID](https://openalex.org/A5087413801)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对终端型 AI 代理工具的设计模式进行系统分析，识别并阐述了三项核心属性：代表性兼容性、交互透明度和低参与门槛，并将这些属性推广为所有人机–AI–UI 交互设计的基本要求。

**💡 创新点**

创新点在于将终端工具的成功经验抽象为三大设计属性，并用经典 HCI 理论（执行/评估之海、直观交互等）来解释它们为何对人机协作至关重要；同时提出 GUI、空间等非文本模态必须主动实现这三项属性，而非仅靠原生接口能力。

**🔧 技术方法**

主要技术手段为：对现有终端代理工具（如 Claude Code、Cursor、OpenAI Codex 等）的工作流拆解、文本流分析；引用 SWE‑bench、OSWorld 等基准实验的性能数据；结合自然语言到 Bash 的 NL2Bash 研究作为低门槛的技术实例。

**📊 数据集**

未使用专门的数据集；引用了 GitHub 上 129,000 项公开项目的统计（15%–23% 代理工具采用率）以及先前研究的实验数据（OSWorld、SWE‑bench、CLI 代码搜索等）。

**📈 对比分析**

比较方法主要依赖于文献引用：GUI 代理任务成功率仅 12.24% 对比人类 72.36%，SWE‑bench 上自定义文本 ACI 超过默认 shell 10.7%，可执行 Python 代码比 JSON 调用高 20%；作者并未进行新的实验，而是建议未来可设计受控对比研究（如文本流 vs GUI 重放）来验证透明度和低门槛效益。

**⚠️ 局限性**

局限包括：缺乏系统的实验验证，尤其是在非软件开发场景（视觉设计、多媒体、空间任务等）下；对 GUI 或空间模态如何实现这三项属性尚未给出具体实现方案；结论主要基于现有研究和定性分析，需进一步量化与实证。

---

## 200. Geometric Autoencoder for Diffusion Models

**arXiv ID:** 2603.10365 | [PDF](https://arxiv.org/pdf/2603.10365v1)

**作者:** Hangyu Liu `[一作]` (Shanghai Innovation Institute), Yutao Sun `[通讯]` (Tsinghua University)

**通讯引用:** 1305 | [OpenAlex ID](https://openalex.org/A5100677875)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Geometric Autoencoder (GAE)，在潜在扩散模型中构建了基于 Vision Foundation Models 的低维语义监督、无 KL 的隐空间归一化与动态噪声采样的自动编码器。

**💡 创新点**

创新点在于：① 在瓶颈层直接进行语义对齐（Latent Alignment）以提升语义保留；② 用 RMSNorm 替代 KL 约束，形成稳定的隐空间；③ 引入动态噪声采样以增强重建鲁棒性。

**🔧 技术方法**

技术手段包括 ViT-L 编解码器、DINOv2 语义教师、SVD/Patch‑Conv 下采样、σ‑VAE 动态噪声、LPIPS、GAN 复合损失以及 RMSNorm 正则化。

**📊 数据集**

实验主要在 ImageNet‑1K 256×256 数据集上进行。

**📈 对比分析**

与 FAE、RAE 等 SOTA 方法对比，GAE 在 80 轮训练时 gFID 1.82，800 轮训练时 gFID 1.31（无 CFG），显著优于同类模型，并在 32/64 维隐空间实现最佳压缩‑语义‑重建 Pareto 前沿。

**⚠️ 局限性**

局限在于对更高分辨率和更大隐空间的泛化能力有限，且对超参数（如 σ、λ_sp）敏感，未来需进一步探索更通用的语义教师和更广泛的评测。

---

## 201. nlm: Real-Time Non-linear Modal Synthesis in Max

**arXiv ID:** 2603.10240 | [PDF](https://arxiv.org/pdf/2603.10240v1)

**作者:** Rodrigo Diaz `[一作]` (Queen Mary University of London), Mark Sandler `[通讯]` (Queen Mary University of London)

**通讯引用:** 36649 | [OpenAlex ID](https://openalex.org/A5091175785)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

**🎯 论文内容**

本工作提出了一套 Max 外部对象，实现了弦、膜、板的非线性模态合成，可实时交互、加载自定义模态数据并支持多通道输出。

**💡 创新点**

将非线性模态模型集成至 Max 环境，利用 Eigen 优化实现实时性能，并支持用户自定义模态与耦合矩阵；同时提供多通道读取与多点激励。

**🔧 技术方法**

C++实现、Eigen 矩阵库、冲击不变离散化、模态展开与耦合矩阵计算、实时多通道读取。

**📊 数据集**

主要使用矩形简单支撑边界下的理论模态；可加载 VKPlate、VKGong 等项目生成的自定义模态与耦合矩阵。

**📈 对比分析**

与以往线性 Max/ Pd 外部对象对比，在典型硬件上可实时运行约 100 个板模态或数百个弦/膜模态；数值稳定性通过采样率限制保证，计算复杂度随模态数平方增长，导致高模态数时 CPU 负荷升高并可能出现点击。

**⚠️ 局限性**

数值稳定性受强激励影响；计算复杂度随模态数增长，导致高模态数时 CPU 负荷过高；仅支持矩形简单支撑边界的耦合矩阵；缺乏实时非线性接触模型和能量约束机制，未来计划改进积分、加入能量裁剪、支持 Pd 版本等。

---

## 202. RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators

**arXiv ID:** 2603.10026 | [PDF](https://arxiv.org/pdf/2603.10026v1)

**作者:** Xinsheng Tang `[一作]` (Alibaba Cloud Computing), Qiang Liu `[通讯]` (Alibaba Cloud Computing)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对AI推理中“级联归约”模式的自动融合框架 RedFuser，能够识别、合并并生成高性能的融合核。

**💡 创新点**

创新点在于：①给出了级联归约的正式理论描述和可融合的数学条件；②设计了增量计算形式，消除冗余内存访问并突破共享内存限制；③实现了基于 TVM 的自动推导与代码生成，并统一涵盖 FlashAttention 等手工优化方案。

**🔧 技术方法**

使用了符号推导、固定点分析、TensorIR/TileIR 转换、增量计算、自动调优等技术，并基于 CUDA/Tensor Cores 进行代码生成。

**📊 数据集**

在四类典型工作负载上评估：多头注意力（MHA）、多潜在注意力（MLA）、MoE 路由、FP8 PerToken Quant + GEMM，采用 BERT、ViT、LLaMA‑65B、DeepSeek‑R1 等模型配置。

**📈 对比分析**

与 PyTorch Eager、PyTorch Dynamo（Inductor）、TVM（Relax）以及 FlashAttention2/FlashMLA 对比，RedFuser 在所有实验中平均 2×–5× 的加速，且在多数配置下匹配或超过手工优化实现，尤其在 LLaMA‑65B、DeepSeek‑V2‑Lite 等大模型上表现突出。

**⚠️ 局限性**

局限性：①仅能处理满足分解条件的级联归约；②融合后会产生额外计算开销和寄存器占用，需要进一步的成本模型来决策是否融合；③目前只针对 NVIDIA GPU（A10、H800）实现，尚未推广到其它 AI 加速器。

---

## 203. Continuous Diffusion Transformers for Designing Synthetic Regulatory Elements

**arXiv ID:** 2603.10885 | [PDF](https://arxiv.org/pdf/2603.10885v1)

**作者:** Jonathan Liu `[一作]` (Princeton University), Kia Ghods `[通讯]` (Princeton University)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5059388134)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种参数高效的Diffusion Transformer (DiT) 模型，用于生成具有细胞类型特异性调控活性的200bp DNA序列，并通过强化学习对预测模型进行微调。

**💡 创新点**

创新点在于将U‑Net denoiser替换为Transformer denoiser并加入2D CNN编码器，显著减少训练步数、参数量和序列记忆率，同时通过DDPO结合Enformer奖励实现表达预测提升。

**🔧 技术方法**

采用扩散模型（DDPM）+Transformer（AdaLN‑Zero），2D CNN输入编码，混合bf16训练，classifier‑free guidance，以及强化学习的DDPO用于奖励驱动微调。

**📊 数据集**

使用ENCODE DHS数据构建的约48k条200bp的调控序列，按四种细胞系（K562、HepG2、GM12878、hESC0）各12k条，做为训练、验证与测试集。

**📈 对比分析**

通过与原始DNA‑Diffusion U‑Net、DRAKES等模型在验证损失、BLAT记忆率、Jensen‑Shannon距离和Enformer预测活性等指标对比，DiT在60倍更少的训练步数下达到更低损失，记忆率降至1.7%，强化学习后表达预测提升38×，在独立预测任务中捕获70% DRAKES的活性。

**⚠️ 局限性**

局限包括模型对Enformer代理的依赖导致可能的模型偏差、生成窗口仅200bp不足以捕捉远程调控、数据量相对有限、缺乏实验验证以及对更长序列和多细胞数据的扩展仍待研究。

---

## 204. Speaker Verification with Speech-Aware LLMs: Evaluation and Augmentation

**arXiv ID:** 2603.10827 | [PDF](https://arxiv.org/pdf/2603.10827v1)

**作者:** Thomas Thebaud `[一作]` (Johns Hopkins University), Najim Dehak `[通讯]` (Johns Hopkins University)

**通讯引用:** 10207 | [OpenAlex ID](https://openalex.org/A5050632169)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一套通用评估协议，用于衡量语音感知大型语言模型（LLM）在说话人验证（ASV）上的能力，并通过该协议评估现有模型的说话人区分效果；随后设计了一种轻量级增量方法，将冻结的ECAPA‑TDNN说话人嵌入投射到LLM并使用LoRA进行微调，使LLM获得强大的说话人识别能力；

**💡 创新点**

创新点在于（1）提出了可跨API与开源模型的模型无关评分方案；（2）通过在LLM中注入预训练说话人嵌入并仅微调少量参数，显著提升说话人验证性能，几乎匹敌专用ASV系统；

**🔧 技术方法**

使用的技术包括：ECAPA‑TDNN说话人嵌入、线性投射层、LoRA参数高效微调、对话式提示设计、置信度分数与对数似然比（LLR）评分；

**📊 数据集**

使用的数据集为 VoxCeleb1（原始、扩展、硬三种拆分）进行评测，训练基于 VoxCeleb2 发展集，验证集为 VoxCeleb2 测试集；

**📈 对比分析**

与基准 ECAPA‑TDNN（cosine 评分）以及原始语音感知 LLM（通过置信度或LLR评分）对比，原始 LLM 的 EER 远高于 20%，而加入 ECAPA 嵌入后 TinyLLaMA‑1.1B 的 EER 在 1.03%（Vox1‑E）左右，接近专用 ASV 系统；

**⚠️ 局限性**

局限性包括：对封闭模型的置信度评分粗糙且依赖模型实现；某些 API 解析失败导致失败率高；评估仅覆盖说话人验证，未扩展到多说话人或对话分析；未探索更高级的跨模态对齐损失或更大模型的适配方式。

---

## 205. End-to-End Chatbot Evaluation with Adaptive Reasoning and Uncertainty Filtering

**arXiv ID:** 2603.10570 | [PDF](https://arxiv.org/pdf/2603.10570v1)

**作者:** Nhi Dang `[一作]` (University of Science), Huy Tien Nguyen `[通讯]` (Vietnam National University)

**通讯引用:** 11248 | [OpenAlex ID](https://openalex.org/A5070329277)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种端到端自动评估RAG聊天机器人的框架，能够无人工标注地自动生成问答对、用LLM作为评判者对模型回答进行分类，并通过多步置信度聚合过滤不确定样本。

**💡 创新点**

创新点在于整合自动化数据生成、LLM‑as‑judge的单提示、顺序决策和自适应K步推理三种评估策略，并通过累计置信度实现可解释的类别标签与高效的人类复核过滤。

**🔧 技术方法**

主要技术包括大语言模型（GPT‑4o、Gemini系列）用于生成问答、评判与推理；多步推理与置信度计算实现不确定性量化；以及基于阈值的自动过滤与人类复核。

**📊 数据集**

使用了由50篇越南新闻自动生成的300条问答对数据集（含人工复核标签），并在此基础上评估RAG式聊天机器人。

**📈 对比分析**

与人工标注结果对比，宏平均准确率显著提升；在强模型（如gpt‑4o‑mini）下，自适应K步推理可达到或超过顺序决策；在阈值为0.4、K=5时，可检测90%以上错误，同时仅需复核不到30%的样本。

**⚠️ 局限性**

局限性包括阈值手动设定导致跨模型/跨领域鲁棒性不足；对弱模型或开放式问答任务的适用性有限；以及对多步推理的计算开销和对源文本明确性的依赖。

---

## 206. Denoising the US Census: Succinct Block Hierarchical Regression

**arXiv ID:** 2603.10099 | [PDF](https://arxiv.org/pdf/2603.10099v1)

**作者:** Badih Ghazi `[一作]` (Google Research), Adam Sealfon `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出一种新的后处理算法，用于美国人口普查的差分隐私数据，能够在保持隐私和结构约束的前提下生成更准确、一致的估计值。

**💡 创新点**

其创新点在于提出基于层次结构的最优线性无偏估计（BLUE），将原本需要三次矩阵乘法的线性回归降到线性时间，并利用查询对称性通过“简洁矩阵”显著提升运算效率。

**🔧 技术方法**

使用的技术包括广义最小二乘回归、等式约束的线性最优估计、Kronecker代数与向量化技巧、稀疏“简洁矩阵”表示，以及后处理阶段的整数规划启发式。

**📊 数据集**

实验数据为2020年美国人口普查公开的Noisy Measurement File（NMF），并通过从公开的Microdata Detail File（MDF）生成10份复制文件来评估准确率。

**📈 对比分析**

与官方2020 Census DAS算法比较，采用平均ℓ1误差归一化指标；在县、区块层面提升8–50%，在群组类和种族类查询上提升更显著；算法计算复杂度线性，能在大规模数据上高效运行。

**⚠️ 局限性**

主要局限在于不等式和整数约束的处理仍为启发式，缺乏全局最优性证明；算法依赖于特定的查询对称结构，迁移到不同统计任务时需重新验证结构假设。

---

## 207. Simulation-in-the-Reasoning (SiR): A Conceptual Framework for Empirically Grounded AI in Autonomous Transportation

**arXiv ID:** 2603.10294 | [PDF](https://arxiv.org/pdf/2603.10294v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 208. Data-Driven Integration Kernels for Interpretable Nonlocal Operator Learning

**arXiv ID:** 2603.10305 | [PDF](https://arxiv.org/pdf/2603.10305v1)

**作者:** Savannah L. Ferretti `[一作]` (University of California Irvine), Tom Beucler `[通讯]` (University of Lausanne)

**通讯引用:** 27287 | [OpenAlex ID](https://openalex.org/A5061588829)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出数据驱动的积分核框架，将气象过程的非局部信息聚合与局部非线性预测分离，从而实现对气候模型的可解释性与高效学习。

**💡 创新点**

创新点在于将非局部聚合转化为可学习的连续权重函数（积分核），并通过层级模型（基线 → 非参数核 → 参数化核）展示可解释性与参数压缩的双重优势。

**🔧 技术方法**

主要技术包括积分核学习（非参数与参数化如高斯、混合高斯、top‑hat、指数等）、深度前馈网络、以及基于空间、垂直与时间的积分域设计。

**📊 数据集**

使用的数据集为ERA5重分析的气象变量（相对湿度、等势温度等）和IMERG V06降水观测，所有数据在相同的1°×1°网格上统一。

**📈 对比分析**

通过与全局非局部基线、局部基线以及逐维增量基线进行对照，R²性能与全局非局部基线相近，非参数核约恢复75%提升，参数化核约67%，但参数量显著减少，且对垂直信息最为敏感。

**⚠️ 局限性**

限制包括：参数化核对细粒度垂直结构的捕捉略有不足，模型主要聚焦垂直维度，其他维度的非局部贡献被压缩；在更大空间或更复杂气候过程上的通用性仍需进一步验证。

---

## 209. AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU

**arXiv ID:** 2603.10342 | [PDF](https://arxiv.org/pdf/2603.10342v1)

**作者:** Yuning Zhang `[一作]` (University of Sydney), Dong Yuan `[通讯]` (University of Sydney)

**通讯引用:** 4447 | [OpenAlex ID](https://openalex.org/A5054168288)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 AgentServe，一款单 GPU 级别的推理服务系统，用于在消费级 GPU 上高效、稳定地为多 AI 代理（工具增强型）提供小型语言模型（SLM）推理。

**💡 创新点**

通过三维协同：①请求阶段识别（冷预填、恢复预填、短解码）并动态划分 GPU 资源；②以 TPOT 为驱动的反馈调度，实现解码优先和预填预算；③预先建立 CUDA Green Context 进行精细 SM 隔离，避免 head‑of‑line 阻塞。

**🔧 技术方法**

CUDA Green Context、共享 KV 缓存、动态 SM 分配、TPOT‑驱动的反馈控制、Prefill–Decode 分离、LLM 推理内核优化、GPU 线程协作等技术。

**📊 数据集**

使用 ToolBench 基准，构造 ReAct 与 Plan‑and‑Execute 两种代理工作负载；评测 Qwen2.5‑3B、Qwen2.5‑7B、LLaMA‑3‑8B 模型。

**📈 对比分析**

与 vLLM、SGLang、llama.cpp 三个基线进行对比；测量 TTFT、TPOT、吞吐量和 SLO 达成率；AgentServe 在 TTFT 上提升 1.1–2.8×、TPOT 上 1.3–2.7×，吞吐量 1.2–2.2×，并保持更高的 SLO 通过率。

**⚠️ 局限性**

仅适用于小型模型在单 GPU 上的工具增强型代理；对大规模多 GPU 方案或极长上下文推理缺乏实验；算法在极高并发下的精确度受限于 SM 分配粒度和控制延迟。

---

## 210. SUBTA: A Framework for Supported User-Guided Bimanual Teleoperation in Structured Assembly

**arXiv ID:** 2603.10459 | [PDF](https://arxiv.org/pdf/2603.10459v1)

**作者:** Xiao Liu `[一作]` (Honda Research Institute USA), Soshi Iba `[通讯]` (Honda Research Institute USA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了SUBTA系统，集成学习的意图估计、场景图任务规划与上下文运动支持，以帮助双手组装任务；

**💡 创新点**

创新点在于将图神经网络驱动的动态意图推理与基于场景图的任务规划相结合，实时生成可视化目标与可执行运动行为，并通过数字孪生环境为操作者提供直观的视觉与运动反馈；

**🔧 技术方法**

采用图神经网络+HAR‑Transformer进行意图估计，使用图编辑距离的场景图规划，行为控制器实现抓取/放置的运动辅助，并在VR/AR数字孪生中渲染可视化；

**📊 数据集**

使用了495条由5块木块组成的组装演示数据集，包含SE(3)轨迹、动作标签和3秒窗口，用于训练意图估计模型；

**📈 对比分析**

通过12名参与者的对比实验（标准遥控、仅运动支持、完整SUBTA），采用线性混合效应模型评估，SUBTA在位置/姿态精度显著提升（p<0.001），精神负荷降低，SUS得分提高，成功率从55.6%提升至75%；

**⚠️ 局限性**

局限性包括样本量有限、仅验证在结构化组装任务，未评估网络延迟或复杂环境下的鲁棒性，且对非结构化任务的适用性未知。

---

## 211. Prism-$Δ$: Differential Subspace Steering for Prompt Highlighting in Large Language Models

**arXiv ID:** 2603.10705 | [PDF](https://arxiv.org/pdf/2603.10705v1)

**作者:** Yuyao Ge `[一作]` (Institute of Computing Technology), Xueqi Cheng `[通讯]` (Institute of Computing Technology)

**通讯引用:** 20783 | [OpenAlex ID](https://openalex.org/A5029998682)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种同时编辑注意力的Key和Value通道的提示突出方法，利用差分交叉协方差分解和softplus权重来实现可解释的多头自适应调节。

**💡 创新点**

创新点在于：①通过差分SVD消除共享结构方向，只保留正负对比的判别子空间；②为每个注意力头生成连续的softplus重要性权重，既抑制噪声头又保留弱但有用的头；③同时对Key和Value两通道进行调节，使提示突出兼顾路由与内容。

**🔧 技术方法**

技术核心包括：差分交叉协方差矩阵、奇异值分解 (SVD)、softplus 重要性映射、线性投影编辑、FlashAttention兼容的前向推理实现。

**📊 数据集**

使用了三大提示突出基准（BiasBios、CounterFact、Pronoun Change）和失落中间（Lost-in-the-Middle）数据集，另外通过 100 条合成 QA 对比样本进行离线投影学习。

**📈 对比分析**

与原始、标记、PASTA、SPA、以及基线的Key‑only 版本相比，该方法在 19/20 的模型×基准组合上至少匹配或优于现有最佳方法，单独提升 10.6% 的准确率、4.8% 的长上下文检索效果，且在 FlashAttention 上仅增加 0.01 秒延迟。

**⚠️ 局限性**

局限性包括：需对不同模型和任务手动调节 g_K、γ、δ_min 等超参数；对已经高度优化的基准（如 CounterFact）提升有限；并且误用高亮可能导致错误信息被过度强化。

---

## 212. OmniGuide: Universal Guidance Fields for Enhancing Generalist Robot Policies

**arXiv ID:** 2603.10052 | [PDF](https://arxiv.org/pdf/2603.10052v1)

**作者:** Yunzhou Song `[一作]` (University of Pennsylvania), Kostas Daniilidis `[通讯]` (University of Pennsylvania)

**通讯引用:** 17873 | [OpenAlex ID](https://openalex.org/A5050660826)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在推理时使用统一能量场对Vision‑Language‑Action策略进行引导的框架，能够在不额外收集数据或重新训练的前提下提升机器人执行性能。

**💡 创新点**

创新点在于将多种外部基础模型（3D重建、VLM、人体姿态）通过可微能量函数统一映射到3D空间，并在推理阶段通过梯度引导流匹配模型，从而实现可组合、无训练的多源指导。

**🔧 技术方法**

采用流匹配VLA、可微逆运动学、SDF碰撞场、VLM语义目标、DTW人类轨迹匹配等技术，并结合能量形状化与梯度裁剪实现推理时引导。

**📊 数据集**

使用RoboCasa仿真数据集和真实世界DROID+Franka实验平台，结合VGGT 3D重建、Gemini‑2.5‑Flash等预训练模型进行评估。

**📈 对比分析**

与基准VLA策略和专门设计的后处理/专用方法（cuRobo、F3RM、DemoDiffusion）对比，成功率从24.2%提升至92.4%，碰撞率从7%下降至93.5%，在多任务和多源引导下均取得显著提升。

**⚠️ 局限性**

局限性包括仍需依赖VLA先验来处理运动学不匹配和接触动力学，能量场可能产生局部最小值，需手工调节引导权重，且推理时增加约一倍延迟。

---

## 213. Rethinking Adam for Time Series Forecasting: A Simple Heuristic to Improve Optimization under Distribution Shifts

**arXiv ID:** 2603.10095 | [PDF](https://arxiv.org/pdf/2603.10095v1)

**作者:** Yuze Dong `[一作]` (Guilin University of Electronic Technology), Jinsong Wu `[通讯]` (University of Chile)

**通讯引用:** 10703 | [OpenAlex ID](https://openalex.org/A5029265765)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种针对非平稳时间序列预测的优化器TS_Adam，通过去除Adam中二阶偏差校正来提升对分布漂移的响应性，保持原有结构且无额外超参数；

**💡 创新点**

创新点在于识别并解决Adam在非平稳环境下因二阶校正导致学习率受限的问题，简单移除该项即可显著提升适应性；

**🔧 技术方法**

技术包括改进Adam的学习率更新规则、理论动态损失分析、实验验证以及与多种主流优化器（Adam、AdamW、Yogi、Lookahead、SGD）和模型（MICN、PatchTST、SegRNN）结合；

**📊 数据集**

使用了ETT系列（ETTh1、ETTh2、ETTm1、ETTm2）、ECL、Weather等长短期预测数据集，以及M4多领域短期预测数据集；

**📈 对比分析**

通过与标准优化器的对比实验，在长短期预测任务中TS_Adam平均降低12.8% MSE、5.7% MAE（ETT+MICN），以及在M4中SMAPE、MASE、OWA分别下降约5.0%、12.2%、7.1%，并在统计显著性检验中保持优势；

**⚠️ 局限性**

局限性包括对趋势主导序列提升有限，且在极大数据量或高度平稳场景下改进幅度可能不明显。

---

## 214. From Education to Evidence: A Collaborative Practice Research Platform for AI-Integrated Agile Development

**arXiv ID:** 2603.10679 | [PDF](https://arxiv.org/pdf/2603.10679v1)

**作者:** Tobias Geger `[一作]` (Clausthal University of Technology), Stefan Wittek `[通讯]` (Clausthal University of Technology)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5055043036)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实施了以项目为基础的AI集成敏捷教育平台，作为协作实践研究平台，用于快速生成可迁移的实践证据。

**💡 创新点**

创新点在于将教育与研究平台相结合，在项目周期内嵌入AI工具与敏捷实践，并通过质量门、真实利益相关者参与及可重复的sprint节奏实现及时、可迁移的研究成果。

**🔧 技术方法**

使用敏捷框架（Scrum）、迭代式教育单元（Schools）、代码仓库、版本控制、AI工具（LLM、代理式AI）等技术。

**📊 数据集**

本研究未使用传统机器学习数据集，数据来源为学生项目的代码仓库、需求文档、演示记录等实践产出。

**📈 对比分析**

本研究未与传统基准进行对比，而是通过定量指标（项目数量、团队规模、学生参与度）和定性观察（评审会、回顾、口试）评估平台的可扩展性和有效性，早期结果显示项目产量和利益相关者参与度稳定增长。

**⚠️ 局限性**

主要限制在于数据主要为定性观察，缺乏量化度量；平台在规模扩大时需要更强的治理和标准；研究仍处于早期阶段，缺乏长期效能评估。

---

## 215. Large chirotopes with computable numbers of triangulations

**arXiv ID:** 2603.10251 | [PDF](https://arxiv.org/pdf/2603.10251v1)

**作者:** Mathilde Bouvel `[一作]` (Université de Lorraine), Florent Koechlin `[通讯]` (CNRS)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在根化 chirotope 上的 join 与 meet 运算，利用它们对 chirotope 进行分解，并通过构造弱三角剖分多项式来递归计数三角剖分，最终给出了双环点集的三角剖分数的精确渐近公式。

**💡 创新点**

创新点在于：①将 Rutschmann–Wettstein 对链的凸/凹求和推广到任意根化 chirotope；②引入双变量弱三角剖分多项式，实现 join/meet 的递归计数；③利用生成函数与 kernel 方法精确推导双环点集的三角剖分计数；④通过实验验证 Koch 链在该构造下的极大性。

**🔧 技术方法**

主要技术包括：组合结构的分解与抽象化（join/meet、twist）、弱三角剖分多项式的递推、生成函数分析与 kernel 方法、解析组合学的转移定理。

**📊 数据集**

实验使用了 Aichholzer 等人公开的所有 10 点 chirotope 数据集，并将其作为根化 chirotope 的起点进行递归构造。

**📈 对比分析**

通过比较不同根化 chirotope 及其构造的三角剖分数，发现以 Koch 链为基准的构造在已测试规模下能得到最多的弱三角剖分（弱三角剖分数与实际三角剖分数上界相近），但未发现能超越 Koch 链的实例。

**⚠️ 局限性**

限制在于：①只考虑平面 chirotope（无法推广到更高维或一般点集）；②求解双环点集的极小性问题仍未完全证明；③join/meet 的递归计数仅适用于可分解的 chirotope，无法处理不可分解的情况；④实验规模受计算资源限制，未能覆盖更大点集。

---

## 216. Historical Consensus: Preventing Posterior Collapse via Iterative Selection of Gaussian Mixture Priors

**arXiv ID:** 2603.10935 | [PDF](https://arxiv.org/pdf/2603.10935v1)

**作者:** Zegu Zhang `[一作]`, Jian Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种名为 Historical Consensus Training 的方法，通过迭代筛选多重 GMM 聚类约束来训练 VAE，从而避免后验崩溃。

**💡 创新点**

创新点在于将同一数据集的多重 GMM 聚类结果视为多任务约束，利用这些约束形成历史障碍，确保训练参数不会收敛到崩溃解。

**🔧 技术方法**

采用了多次 EM 初始化得到不同聚类、条件损失与标准 VAE 损失交叉训练、迭代选择与细化、以及理论证明历史障碍的存在。

**📊 数据集**

在合成 GMM 数据、MNIST、Fashion‑MNIST 与 CIFAR‑10（灰度化）等多种数据集上进行实验。

**📈 对比分析**

与 Vanilla VAE、β‑VAE、KL 退火等基线对比，实验显示 KL 散度大幅提升（从 <0.01 到 >2.0，最高 3.7），并成功保持非崩溃状态。

**⚠️ 局限性**

局限性包括计算成本较高（需多次 EM 与多轮训练）、对初始聚类数量与阈值敏感、以及活跃潜在维度仍有限（仅 2‑5/48），需要进一步改进表示分布。

---

## 217. Learning to Negotiate: Multi-Agent Deliberation for Collective Value Alignment in LLMs

**arXiv ID:** 2603.10476 | [PDF](https://arxiv.org/pdf/2603.10476v1)

**作者:** Panatchakorn Anantaprayoon `[一作]` (Integral AI), Jad Tarifi `[通讯]` (Integral AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可扩展的多智能体协商框架，训练大型语言模型在面对价值冲突时通过对话实现Collective Agency（CA）动态对齐，并提升冲突解决能力。

**💡 创新点**

创新点在于将CA对齐目标嵌入多智能体协商流程，采用组相对强化学习（GRPO）在对话层面优化协商策略，并利用外部LLM评估来给出CA奖励。

**🔧 技术方法**

使用自对抗自-play、组相对强化学习（GRPO）、token级归一化、外部LLM评估器（同意判定与CA评分），以及基于Qwen3-14B的QLoRA微调。

**📊 数据集**

构建1,100个价值冲突情境与25对敌对角色的库，生成合成对话数据；对评测采用100个冲突解决任务与100个开放式问题。

**📈 对比分析**

与基线模型（原始Qwen3-14B）以及单智能体CA对齐模型比较；在CA对齐和冲突解决任务中多智能体模型获得与单智能体相当的CA得分、显著提升的协议率和平均对话轮数，并保持与基线相似的通用语言能力。

**⚠️ 局限性**

局限包括：缺乏对各设计模块单独贡献的消融实验、评测仅关注结果指标而非协商质量、数据集规模与多样性不足、仅限双方协商且使用单一模型自对抗，未覆盖多方和异质智能体场景；奖励信号过于粗粒度，缺少细粒度信用分配与多时延奖励。

---

## 218. ID-LoRA: Identity-Driven Audio-Video Personalization with In-Context LoRA

**arXiv ID:** 2603.10256 | [PDF](https://arxiv.org/pdf/2603.10256v1)

**作者:** Aviad Dahan `[一作]` (Tel Aviv University), Raja Giryes `[通讯]` (Tel Aviv University)

**通讯引用:** 6178 | [OpenAlex ID](https://openalex.org/A5072571599)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的音视频个性化模型，能够同时生成符合参考图像和短音频的外观与声音，并通过文本提示控制说话风格与环境音。

**💡 创新点**

创新点在于将 In-Context LoRA 扩展到音视频联合生成，提出负时间位置区分参考与目标音频，并引入身份引导提升语音身份保持。

**🔧 技术方法**

采用 LTX-2 共享的音视频 Diffusion Transformer 作为骨干，结合 In-Context LoRA、RoPE 负时间位置、身份引导与多模态文本条件。

**📊 数据集**

使用 CelebV-HQ 和 TalkVid 两个多模态数据集，训练约 3K 对参考-目标样本。

**📈 对比分析**

与 Cascaded 语音克隆+WAN2.2、VoiceCraft+WAN2.2、ElevenLabs+WAN2.2 以及闭源 Kling 2.6 Pro 进行对比，实验表明在说话人相似度、唇同步、环境音遵循度等指标上均优于基线，尤其在跨视频场景下提升显著。

**⚠️ 局限性**

局限性包括仅支持单一说话人、对长时序音频生成的鲁棒性尚未充分验证、对多说话人或跨语言的泛化能力待进一步研究。

---

## 219. Evaluating Progress in Graph Foundation Models: A Comprehensive Benchmark and New Insights

**arXiv ID:** 2603.10033 | [PDF](https://arxiv.org/pdf/2603.10033v1)

**作者:** Xingtong Yu `[一作]` (Chinese University of Hong Kong), Yuan Fang `[通讯]` (Singapore Management University)

**通讯引用:** 3958 | [OpenAlex ID](https://openalex.org/A5027522861)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个二维域迁移评估框架，对图基础模型（GFM）在主题域与格式域双向转移场景下进行系统评测；

**💡 创新点**

首次将图数据的主题与格式双维划分并构建统一评测协议，覆盖七大主题和六大格式，形成全面的跨域基准；

**🔧 技术方法**

基于多源无监督预训练+少样本下游微调的标准GFM流程，结合统一预处理和任务拆分；

**📊 数据集**

使用33个公开图数据集，涵盖引文网络、社交/网络、电商、金融、常识、分子、蛋白质等七大主题及同质/异质、同源/异源、静态/动态、关系型、文本属性等六种格式；

**📈 对比分析**

对八种先进GFM（如MDGPT、SAMGPT、GFT、G2P2等）与传统GCN/GAT及部分专用预训练方法进行对比，结果显示GFM在多域下往往优于基线，但提升不均衡，尤其在未见数据集和大格式差异时表现波动；

**⚠️ 局限性**

局限性在于多域知识整合不足、跨域迁移仍易受数据集级别差异影响，且对大格式偏移（异质、动态、文本）缺乏专门化机制，导致鲁棒性不足。

---

## 220. RCTs & Human Uplift Studies: Methodological Challenges and Practical Solutions for Frontier AI Evaluation

**arXiv ID:** 2603.11001 | [PDF](https://arxiv.org/pdf/2603.11001v1)

**作者:** Patricia Paskov `[一作]` (RAND), Ella Guest `[通讯]` (RAND)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对16位专家进行访谈，系统梳理了在前沿大语言模型（LLM）人类提升（Human Uplift）研究中的方法学挑战与对应解决方案；

**💡 创新点**

首次将人类提升研究的挑战映射至构造、内在与外在效度三类威胁，并提出一套针对性实践改进路径；

**🔧 技术方法**

主要采用定性主题分析法、效度映射框架与专家访谈方法，构建挑战-解决方案的对应图谱；

**📊 数据集**

使用了16份专家访谈记录，涉及多领域（生物安全、网络安全、教育、劳动等）已完成或正在进行的人类提升实验；

**📈 对比分析**

未涉及传统定量对比实验，主要通过专家报告与案例讨论说明不同设计策略对效度的影响；

**⚠️ 局限性**

样本规模有限、行业代表性不足、可能存在抽样与响应偏差，且对不同语言与地区的通用性验证不足。

---

## 221. SiDiaC-v.2.0: Sinhala Diachronic Corpus Version 2.0

**arXiv ID:** 2603.10861 | [PDF](https://arxiv.org/pdf/2603.10861v1)

**作者:** Nevidu Jayatilleke `[一作]`, Johan Sofalas `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了史上最大的僧伽罗语历时语料库，收集并清洗了 5 世纪至 20 世纪的 186 本书，完成文本抽取、OCR、后处理和元数据生成，形成约 16000 词元的数据集。

**💡 创新点**

创新点在于结合书面年代与作者寿命进行写作年代标注，提出专门的诗歌尾缀标记与句子结束标记，解决多列排版与代码混杂问题，并提供完整的元数据与词汇一致性分析。

**🔧 技术方法**

采用 Google Document AI OCR 进行历史僧伽罗语文本现代化与形态分割，结合 Python、NLTK 与正则表达式进行词汇统计、停用词筛选和词袋时序上下文分析。

**📊 数据集**

原始文本来源于斯里兰卡国家图书馆数字化资源，整合了《Sikhara Bhasha Ithihasaya》、《Yasodharaawatha》等诗歌集，最终构成 186 本书的文档集。

**📈 对比分析**

与唯一现有的僧伽罗语历时语料库（Sikhara Bhasha）比较，去除非僧伽罗语后词元数量从约 130 万下降至约 100 万；通过 Bag‑of‑Words 统计展示词汇一致性和语义演变，验证语料库规模与质量显著提升。

**⚠️ 局限性**

限制包括写作年代标注仅覆盖 60 本书，未完整处理注释书的双重年代；缺少可靠的僧伽罗语词性标注器，导致未实现词性或形态分析；部分文本仍含多列排版与代码混杂，增加后处理难度。

---

## 222. ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation

**arXiv ID:** 2603.10211 | [PDF](https://arxiv.org/pdf/2603.10211v1)

**作者:** Khoa Anh Ta `[一作]` (University of Information Technology), Kiet Van Nguyen `[通讯]` (University of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了覆盖全 63 省、包含北、中、南三大方言的手工对齐方言-标准越南语平行语料 ViDia2Std，并通过该语料训练并评估方言归一化模型；

**💡 创新点**

首次提供全国范围、方言覆盖面广、样本量大且高标注质量的方言归一化语料库，同时展示归一化在机器翻译和情感分析等下游任务中的显著性能提升；

**🔧 技术方法**

采用多种序列到序列模型（BARTpho、ViT5、mBART‑large‑50 等），以及基于 LLM 的评估（Gemini 2.5 Flash）和传统评估指标（BLEU/ROUGE‑L/METEOR、WER/CER）进行训练与评测；

**📊 数据集**

ViDia2Std 13,657 条句子对（来自 63 省 Facebook 评论），以及公开的 Facebook 评论抓取脚本与评测数据集；

**📈 对比分析**

在内在评测中 mBART‑large‑50 取得最高 BLEU 0.8166、ROUGE‑L 0.9384、METEOR 0.8925；在外在评测中，归一化前后机器翻译的接受率提升约 5–12%（如 Gemini 2.0 Flash 从 61.83% 提升至 67.00%），情感分析准确率从 50.59% 提升至 62.13%；

**⚠️ 局限性**

存在过度归一化导致语义细节丢失、评估仅依赖单一 LLM 判别器导致评价噪声、语料来源单一（仅 Facebook 评论）、以及部分下游系统对归一化输出仍表现不稳定等限制；

---

## 223. Surrogate models for nuclear fusion with parametric Shallow Recurrent Decoder Networks: applications to magnetohydrodynamics

**arXiv ID:** 2603.10678 | [PDF](https://arxiv.org/pdf/2603.10678v1)

**作者:** M. Lo Verso `[一作]`, A. Cammi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于SHRED（浅层递归解码器）与SVD降维的全数据驱动框架，用稀疏温度传感器重建聚变装置冷却壳层中铅锂等离子体的完整MHD状态；

**💡 创新点**

创新点在于首次将SHRED应用于MHD物理，并通过SVD构建参数化低秩基，实现在不同磁场强度下的泛化和对传感器位置的鲁棒性；

**🔧 技术方法**

使用的技术包括LSTM+浅层解码器（SHRED）、奇异值分解（SVD）压缩、OpenFOAM MHD仿真、以及30组随机传感器配置的集成训练；

**📊 数据集**

实验数据集为19个不同磁场强度（0.01–0.5 T）的压缩铅锂MHD通道流动仿真，共120个时间步、14460个网格点；

**📈 对比分析**

通过与全阶模型（FOM）比较，SHRED在未见磁场强度和任意传感器布局下，温度、速度、压强的相对L²误差均低于3%，重建时间约1 s；

**⚠️ 局限性**

局限性在于仅验证了二维阶梯通道、单一垂直磁场方向，需进一步扩展至三维、更复杂几何和多组磁场配置，并依赖大量高阶训练数据。

---

## 224. UniPINN: A Unified PINN Framework for Multi-task Learning of Diverse Navier-Stokes Equations

**arXiv ID:** 2603.10466 | [PDF](https://arxiv.org/pdf/2603.10466v1)

**作者:** Dengdi Sun `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**通讯引用:** 12048 | [OpenAlex ID](https://openalex.org/A5030720334)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种统一的多流量物理信息神经网络框架 UniPINN，解决在不同流域中共享物理约束、抑制负迁移和训练不稳定等问题；

**💡 创新点**

创新点包括：共享-专用两级网络结构（提取共性物理特征与捕获流域特异性特征）、跨流注意机制（通过自注意与交叉注意实现任务间物理知识交互与负迁移抑制）、动态权重分配（DWA）策略实现多任务损失平衡；

**🔧 技术方法**

使用技术包括：Physics‑Informed Neural Network（PINN）框架、自动微分求高阶导数、共享‑专用 MLP 主干与任务专用头、跨流注意模块、动态权重分配（DWA）以及损失函数组合（方程残差、边界条件、数据监督）；

**📊 数据集**

实验数据集为三种二维不可压 Navier‑Stokes 基准流：lid‑driven cavity、Poiseuille pipe、Couette flow，使用 PDEBench 生成的高精度数值解作为监督标注；

**📈 对比分析**

在三种流域上与多种基线（线性回归、GP、DNN、标准 PINN、LAAF‑PINN、KIH‑PINN、AL‑PINN、MMPDE‑Net）进行比较，采用 MSE 评估速度场与压力场。UniPINN 在所有流域的 MSE 均最低，分别比最佳单任务方法降低约 24%–62%，并表现出更平衡、稳健的多任务学习效果；

**⚠️ 局限性**

局限性包括：仅验证二维层流，缺乏对 3D 或高 Reynolds turbulence 的评估；对薄边界层或激波等高梯度结构的精度仍低于精调数值解；模型规模相对较大，需进一步轻量化；未探讨与传统数值求解器混合使用的可能性。

---

## 225. Reconstructing Bounded Treelength Graphs with Linearithmic Shortest Path Distance Queries

**arXiv ID:** 2603.10432 | [PDF](https://arxiv.org/pdf/2603.10432v1)

**作者:** Chirag Kaudan `[一作]` (Oregon State University), Amir Nayyeri `[通讯]` (Oregon State University)

**通讯引用:** 539 | [OpenAlex ID](https://openalex.org/A5024487390)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种确定性算法，在已知最大度Δ和树长τ的连通图中，仅需O(Δ^3τ+2 n log n)次最短路径查询即可重构整个图。

**💡 创新点**

创新点在于将先前随机算法的查询量从O(Δ^3τ+2 n log^2 n)降低到O(Δ^3τ+2 n log n)，并实现了对树长约束图的确定性重构，匹配了已知下界。

**🔧 技术方法**

核心技术包括层次树（layering tree）构造、基于树的对数搜索找连通分量、以及对层内外边界的穷举检验。

**📊 数据集**

未使用公开数据集，研究以理论分析为主。

**📈 对比分析**

与之前的随机算法相比，该方法在查询复杂度上提升了一个log n因子，实验/理论分析表明在Δ、τ固定时总体查询量为Θ(Δ^3τ+2 n log n)。

**⚠️ 局限性**

局限性包括对最大度和树长的先验知识要求，以及算法在实际大规模网络中的实现复杂度可能仍较高。

---

## 226. A neural operator for predicting vibration frequency response curves from limited data

**arXiv ID:** 2603.10149 | [PDF](https://arxiv.org/pdf/2603.10149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 227. UHD Image Deblurring via Autoregressive Flow with Ill-conditioned Constraints

**arXiv ID:** 2603.10517 | [PDF](https://arxiv.org/pdf/2603.10517v1)

**作者:** Yucheng Xin `[一作]` (Shandong Normal University), Zhuoran Zheng `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5077971554)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种基于自回归流（Autoregressive Flow）的超高分辨率图像去模糊方法，通过从低到高分辨率的粗到细残差生成与少步ODE采样实现高质量且高效的4K/8K去模糊；

**💡 创新点**

创新点包括：①将去模糊拆解为分辨率逐级的残差级联并用自回归流建模；②利用流匹配与Rectified Flow的短ODE采样；③引入条件数正则化抑制UHD场景下的数值不稳定与特征交互病态；④结合时间‑尺度嵌入和解析细节补偿提升跨尺度一致性；

**🔧 技术方法**

技术栈包括Flow Matching、Rectified Flow、Neural ODE、Euler/Heun少步ODE求解、条件数正则化、时间‑尺度联合嵌入、解析细节补偿、混合损失（flow loss、final loss、consistency loss、condition loss）以及混合精度训练；

**📊 数据集**

使用UHD-Blur（约3840×2160合成运动模糊）和MC‑Blur（UHDM）两大UHD去模糊基准，以及GoPro、DVD、RealBlur‑J/R四个常规分辨率去模糊数据集进行训练与评测；

**📈 对比分析**

与16+种SOTA后端（MIMO‑UNet++、Restormer、Uformer等）在PSNR/SSIM、FLOPs和推理时间上对比，ARF‑IC在UHD‑Blur上实现PSNR 30.84 dB、SSIM 0.8816、推理0.725 s、FLOPs 28.3 G，优于其他模型；在非UHD基准亦保持竞争力；

**⚠️ 局限性**

局限性：仍受显存与计算限制，细节补偿权重需手动调节；ill‑conditioned约束增加训练开销；对极端噪声或不稳定高频信息的鲁棒性待提升；在多模糊或真实工业环境下的泛化仍有限。

---

## 228. StructDamage:A Large Scale Unified Crack and Surface Defect Dataset for Robust Structural Damage Detection

**arXiv ID:** 2603.10484 | [PDF](https://arxiv.org/pdf/2603.10484v1)

**作者:** Misbah Ijaz `[一作]` (University of Gujrat), Muhammad Nabeel Asim `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并发布了 StructDamage 数据集，该数据集通过聚合32个公开裂缝和表面缺陷数据集，统一标签为9类，提供标准目录结构、详细文档和基线训练代码。

**💡 创新点**

创新点在于：①多源、多材质裂缝/缺陷数据集统一化；②构建9类多类别标签体系；③系统化预处理（去重、分辨率标准化、质量筛选）和重标注；④公开高质量、可复现的基线模型与评估结果。

**🔧 技术方法**

采用计算机视觉与深度学习技术：使用多种CNN（DenseNet、ResNet、EfficientNet、MobileNet、VGG）和Vision Transformer（ViT）进行图像分类；在训练前实施相似性去重、数据增强、迁移学习；并使用标准的训练/验证/测试划分。

**📊 数据集**

使用的底层数据集包括32个公开数据集（如 METU、SDNET2018、RDD 系列、dacl10k 等），合并后约 78,093 张图片，覆盖墙体、瓷砖、石材、道路、柏油路、甲板、混凝土、砖块等九类表面缺陷。

**📈 对比分析**

通过基线实验比较 15 种模型，评估准确率、精确率、召回率、F1 分数。DenseNet201 在平衡子集上达到 98.62% 的准确率，宏平均 F1 分数 0.9853；12 种模型 F1 分数均超过 0.96，表明数据集具有很高的可区分性。

**⚠️ 局限性**

局限性包括：①仍采用单标签标注，缺乏多标签表示；②原始数据分布不平衡，需手动平衡或使用权重；③源数据存在域漂移，可能影响跨域泛化；④未覆盖分割/定位任务；⑤缺少三维/多光谱等多模态扩展。

---

## 229. Autonomous Search for Sparsely Distributed Visual Phenomena through Environmental Context Modeling

**arXiv ID:** 2603.10174 | [PDF](https://arxiv.org/pdf/2603.10174v1)

**作者:** Eric Chen `[一作]` (Massachusetts Institute of Technology), Yogesh Girdhar `[通讯]` (Woods Hole Oceanographic Institution)

**通讯引用:** 922 | [OpenAlex ID](https://openalex.org/A5072400128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于单张标注图像的“一拍即知”目标珊瑚检测与环境上下文建模方法，实现自主水下车辆在珊瑚礁中的高效、适应性搜索；

**💡 创新点**

创新点在于将自监督视觉特征（DINOv2）用于一次性目标与其共生环境特征的检测，并通过在线更新环境上下文缓冲区提供更稠密的探索信号；

**🔧 技术方法**

使用了DINOv2 ViT的patch级嵌入进行目标与上下文的相似度匹配，以及基于图像网格的贪婪路径规划；

**📊 数据集**

数据集为2018年3月在美国维尔京群岛圣约翰岛两块20m×20m珊瑚礁（Yawzi Point、Tektite）采集的约4000幅AUV航拍图像，涵盖三种珊瑚种类；

**📈 对比分析**

与传统全覆盖的lawnmower、仅目标检测及人工地质标注的substrata等基线比较，结果显示环境上下文+目标检测的策略在稀疏目标情境下能在约一半时间内采集75%目标，明显优于其他方法；

**⚠️ 局限性**

局限性包括仅在有限面积网格上验证，缺乏非贪婪规划与大规模场景的实地部署测试，且对复杂多模目标分布的鲁棒性尚待进一步评估。

---

## 230. ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping

**arXiv ID:** 2603.10088 | [PDF](https://arxiv.org/pdf/2603.10088v1)

**作者:** Zijian Zhu `[一作]` (Tsinghua University), Kaisheng Ma `[通讯]` (Tsinghua University)

**通讯引用:** 4160 | [OpenAlex ID](https://openalex.org/A5006570986)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ES-dLLM，一种训练无关的加速框架，通过在Diffusion大型语言模型推理中早期跳过不重要的token计算，显著提升推理吞吐量。

**💡 创新点**

创新点在于利用token在相邻迭代中置信度和中间张量变化的微小差异来估计重要性，并在早期Transformer层动态跳过低重要性位置，从而在不训练额外模型的前提下实现大幅度加速。

**🔧 技术方法**

核心技术包括：基于置信度和隐藏状态变化的权重混合重要性分数计算、部分KV缓存更新与早期跳过机制、以及与现有KV缓存方法（DualCache）对齐的并行解码框架。

**📊 数据集**

在LLaDA-8B-Instruct和Dream-7B-Instruct两个开源Diffusion LLM上进行实验，使用GSM8K、MATH、HumanEval、MBPP、BBH等五个多任务基准数据集进行评估。

**📈 对比分析**

与原始实现和最新DualCache方法对比，ES-dLLM在NVIDIA H200 GPU上实现了5.6×至16.8×的速度提升（最高TPS 308.51）且在生成质量上与Baseline相近甚至优于DualCache，速度提升幅度约为1.20×至1.85×相较于DualCache。

**⚠️ 局限性**

局限性包括：重要性分数基于简单启发式估计，可能不适用于所有任务；跳过策略在训练过程中未得到强化，可能导致累积误差；对内存带宽的依赖使得理论上可降低的FLOPs未能完全转化为实际速度提升。

---

## 231. TacLoc: Global Tactile Localization on Objects from a Registration Perspective

**arXiv ID:** 2603.10565 | [PDF](https://arxiv.org/pdf/2603.10565v1)

**作者:** Zirui Zhang `[一作]`, Huan Yin `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了TacLoc框架，完成一次性全局触觉定位，直接通过点云配准估计手中物体相对于机器人末端执行器的姿态。

**💡 创新点**

创新点在于利用基于图论的部分到全局配准方法，结合法向一致性剪枝和多假设验证，避免渲染或预训练模型，显著提升速度与精度。

**🔧 技术方法**

使用ISS关键点检测、FPFH描述符、图兼容性判定、最大团搜索、Kabsch算法以及点对面误差验证等技术进行配准与验证。

**📊 数据集**

在YCB-Reg仿真数据集（DIGIT传感器滑动触感）以及5个真实家用物体（GelSight Mini、DIGIT、Daimon）上进行评估。

**📈 对比分析**

与RANSAC、TEASER++、3DMAC、SpinNet、DIP等基线方法对比，TacLoc在配准成功率、误差（RE/TE）和时延（约1.4 s）上均优于其他方法。

**⚠️ 局限性**

依赖精确的CAD模型，对缺失或形变严重的物体表现不佳，法向一致性阈值对结果敏感，且在高噪声或极低内点比例下仍可能失效。

---

## 232. An Event-Driven E-Skin System with Dynamic Binary Scanning and real time SNN Classification

**arXiv ID:** 2603.10537 | [PDF](https://arxiv.org/pdf/2603.10537v1)

**作者:** Gaishan Li `[一作]` (City University of Hongkong), Arindam Basu `[通讯]` (City University of Hongkong)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个事件驱动的电子皮肤系统，利用二进制扫描搜索与卷积脉冲神经网络实现实时手写数字识别。

**💡 创新点**

提出了事件驱动的二进制扫描搜索策略、在FPGA上实现的Conv‑SNN以及构建真实的AER触觉数据集。

**🔧 技术方法**

采用压电电阻传感阵列、前端放大+ADC、二进制扫描搜索、差分调制、卷积脉冲神经网络以及FPGA实现。

**📊 数据集**

通过13名受试者在2秒内书写数字收集的760条触觉样本构成的数据集。

**📈 对比分析**

与传统CNN和SNN对比，Conv‑SNN仅需65%计算量、15.6%权重存储，准确率92.11%，数据稀疏率99%，扫描次数减少12.8×。

**⚠️ 局限性**

系统仅在16×16阵列上验证，缺乏对更大规模、更复杂任务的实测与鲁棒性评估。

---

## 233. PET-F2I: A Comprehensive Benchmark and Parameter-Efficient Fine-Tuning of LLMs for PET/CT Report Impression Generation

**arXiv ID:** 2603.10560 | [PDF](https://arxiv.org/pdf/2603.10560v1)

**作者:** Yuchen Liu `[一作]` (Fudan University), Le Xue `[通讯]` (Fudan University)

**通讯引用:** 11924 | [OpenAlex ID](https://openalex.org/A5100721718)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了PET-F2I-41K大规模PET/CT印象生成基准，并在此基准上评估27个LLM，提出ECR、UER、FCR等临床指标，开发基于LoRA的PET-F2I-7B模型。

**💡 创新点**

首次构建PET/CT专用大规模基准与临床度量，证明任务特定适配远优于单纯规模扩展，并实现可本地部署的高效LLM。

**🔧 技术方法**

采用LoRA参数高效微调Qwen2.5-7B-Instruct，结合多种NLP评估指标和自定义实体识别器，构建三维临床评价框架。

**📊 数据集**

使用41,191例真实PET/CT报告（2013-2023），在严格患者级拆分下形成训练/验证/测试集。

**📈 对比分析**

通过对27个模型的零射击评估，PET-F2I-7B在BLEU-4、ROUGE-L、ECR等指标上均明显领先，实体覆盖率提升约3倍，模型推理成本低、延迟低、符合隐私要求。

**⚠️ 局限性**

局限性包括仅基于单中心、单语言数据；缺乏多模态（影像）输入；缺少放射科医师偏好对齐；在稀有示踪剂上仍需进一步验证。

---

## 234. Dynamics-Informed Deep Learning for Predicting Extreme Events

**arXiv ID:** 2603.10777 | [PDF](https://arxiv.org/pdf/2603.10777v1)

**作者:** Eirini Katsidoniotaki `[一作]` (Massachusetts Institute of Technology), Themistoklis P. Sapsis `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 4349 | [OpenAlex ID](https://openalex.org/A5037328807)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个完全基于数据的框架，用来在高维混沌系统中实现长时限的极端事件预测；框架通过在可观测状态快照中构造有限时间Lyapunov指数（FTLE）预警信号，再将其输入Transformer模型来预测目标观测量；

**💡 创新点**

创新点在于：①将OTD（Optimal Time‑Dependent）模式与FTLE相结合，构造出可解释的、机制驱动的预警指标；②采用数据驱动的方法近似非线性动力学，只用于计算线性化流量而不直接用于预测；③将机制预警与深度学习模型结合，实现对极端事件的长期预测；

**🔧 技术方法**

使用的数据驱动技术包括：FNO（Fourier Neural Operator）和ResUNet++等深度网络逼近系统动力学；OTD模式演化和投影得到的低阶线性算子；在OTD子空间中计算有限时间Lyapunov指数；Transformer（Informer）时间序列模型用于序列到序列的预测；输出加权MAE损失强化稀有事件预测；

**📊 数据集**

使用的“数据集”为Kolmogorov流动的高分辨率数值模拟快照：6400个空间点、30,000时间单位、Δt=0.1，总共300,000个状态快照；该数据集完全由作者自行生成；

**📈 对比分析**

与传统的傅里叶模式预警（α(1,0)）进行对比，采用F1、AUC、α*和极端事件计数差异等二分类指标；结果显示FTLE预警在τ≤10时保持高F1和AUC，并在更长的预测时限（τ≈12）仍优于傅里叶预警；在尾部统计上，FTLE预测的概率密度曲线与真实分布更贴近，误差指标𝔻更小；

**⚠️ 局限性**

局限性包括：①需要高质量、全状态的时空快照，无法直接应用于仅观测部分变量的实际系统；②对OTD子空间维度和时间窗口的选择敏感，需经验调参；③在极长预测时限下性能下降；④计算成本在大规模系统中仍较高，尤其是对FNO或ResUNet++的训练和Otd演化；

---

## 235. SiMPO: Measure Matching for Online Diffusion Reinforcement Learning

**arXiv ID:** 2603.10250 | [PDF](https://arxiv.org/pdf/2603.10250v1)

**作者:** Haitong Ma `[一作]` (Harvard University), Bo Dai `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6345 | [OpenAlex ID](https://openalex.org/A5062711588)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Signed Measure Policy Optimization (SiMPO) 框架，将在线扩散强化学习中的重加权方法统一为两阶段测度匹配，并允许使用符号（负）测度。

**💡 创新点**

创新点在于：1) 用 f‑divergence 延伸到符号测度，解耦正负权重；2) 两阶段构造目标测度与投影重加权，提供理论解释和几何直观；3) 允许任意单调增重加权函数，直接利用负样本提升探索与性能。

**🔧 技术方法**

核心技术包括：扩散/流模型、f‑divergence 与签名测度、重加权流匹配、优势加权回归、Q‑函数估计、温度/λ 自适应、正则化与投影求解。

**📊 数据集**

实验数据集涵盖：一维 bandit、OpenAI Gym MuJoCo v4 (HalfCheetah、Humanoid、Ant、Walker2d、Hopper、Swimmer)、DeepMind Control Suite 的 Cheetah‑Run 与 Walker‑Run、以及约70万条增强子序列的 DNA 生成任务。

**📈 对比分析**

与模型无关的 RL 基线（PPO、TD3、SAC）、扩散 RL 基线（QSM、DIPO、DACER、QVPO、DPPO）以及 DNA 生成基线（CG、SMC、TDS、CFG、DRAKES、RL‑D²）进行对比；在 MuJoCo 环境中大多数任务均优于对手；在 DNA 生成中加入负样本重加权后提升约15–17%，显著优于最佳基线。

**⚠️ 局限性**

局限性包括：负权重需精细截断以避免训练不稳定；不同重加权函数的归一化难度较高；对 λ 和 KL 约束的敏感性；主要针对扩散/流模型，尚未在其他生成模型或更复杂环境中验证。

---

## 236. Quantization of Ricci Curvature in Information Geometry

**arXiv ID:** 2603.10054 | [PDF](https://arxiv.org/pdf/2603.10054v1)

**作者:** Carlos C. Rodriguez `[一作]` `[通讯]`, Carlos C. Rodriguez

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在信息几何框架下对二值贝叶斯网络（bitnet）与高斯DAG网络进行理论分析，证明树形网络的平均Ricci标量是正半整数，揭示循环网络破坏量子化并给出双碰撞环的非整数示例；同时推导高斯树形网络的常数负曲率公式，并讨论二者的符号二分；此外通过SymPy、Beta函数消消、Markov链转移矩阵等工具对各种经典拓扑的体积与曲率进行精确计算；

**💡 创新点**

首次在二值贝叶斯网络中严格证明半整数量子化与拓扑Betti数的对应关系，揭示循环网络是破坏量子化的根本原因；提出“Beta消消”机制并证明树形网络平均Ricci标量为∑m_k(m_k+1)/4；给出高斯DAG网络的负常数曲率通式 R=-(d+5)(d-1)/8；提供对比分析与先前错误结果的纠正；

**🔧 技术方法**

信息几何（Fisher信息度量）、贝塔函数恒等式、Markov链转移矩阵推导、SymPy符号计算、数值积分（Monte Carlo、Richardson外推）、维度递推与结构化归纳、Lie群左不变度量理论；

**📊 数据集**

无公开数据集，全部采用理论推导与符号/数值计算验证；

**📈 对比分析**

与2004年原始结果对比，纠正了 L̃_n 与 Ẽ_n 的平均Ricci 标量公式，并通过数值实验验证 β_1=0（无环）网络满足量子化，β_1=1 破坏量子化，β_1≥2 产生非有理值；在高斯网络中验证负曲率常数与维度的关系；

**⚠️ 局限性**

局限性包括：仅在树形网络下证明量子化；循环网络的非整数曲率尚未给出完整分类；高斯DAG链式结构中曲率非常数的解析公式仍未完全确定；扩展到多值（k>2）网络、非叶子父节点共享的复杂结构等问题仍是开放研究方向。

---

## 237. A Governance and Evaluation Framework for Deterministic, Rule-Based Clinical Decision Support in Empiric Antibiotic Prescribing

**arXiv ID:** 2603.10027 | [PDF](https://arxiv.org/pdf/2603.10027v1)

**作者:** Francisco José Gárate `[一作]` (Universidad Politécnica de Madrid), Enrique Javier Gómez `[通讯]` (Centro de Investigación Biomédica en Red)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一套在明确约束范围内的确定性临床决策支持系统治理与评估框架，用于经验性抗生素处方；

**💡 创新点**

创新点在于将治理机制视为一等设计组件，明确定义抽象化的弃权类型、确定性药物管治约束和排除规则，并以行为一致性为核心的评估方法；

**🔧 技术方法**

采用基于规则的确定性推理、治理层门控逻辑、以及固定的合成案例评估脚本实现；

**📊 数据集**

使用的是人工构造的机制驱动合成临床案例集，用于检验系统在各治理条件下的预期行为；

**📈 对比分析**

评估方法主要是期望行为一致性、建议覆盖率和弃权原因分布，报告的“性能”为与预设行为一致的比例（覆盖率不作临床效能评价）；

**⚠️ 局限性**

局限包括：缺乏真实临床验证、仅使用合成数据、确定性规则依赖专家手工定义、适用范围狭窄、未考虑实际部署与集成问题。

---

## 238. Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization

**arXiv ID:** 2603.10808 | [PDF](https://arxiv.org/pdf/2603.10808v1)

**作者:** Linghao Zhang `[一作]` (Nanjing University of Posts and Telecommunications), Linghao Zhang `[通讯]` (Nanjing University of Posts and Telecommunications)

**通讯引用:** 862 | [OpenAlex ID](https://openalex.org/A5100725798)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并演示了基于LLM的Nurture-First Development（NFD）方法，通过持续对话培养域专家AI代理并周期性将经验转化为结构化知识。

**💡 创新点**

核心创新在于将开发与部署并行化、引入知识结晶周期、三层认知架构、双工作空间和螺旋发展模型，从而实现隐性专业知识的渐进编码与动态演化。

**🔧 技术方法**

利用LLM代理框架（OpenClaw、Claude Code等）、内存检索、语义搜索、脚本化技能以及半自动化的结晶算法。

**📊 数据集**

以一位美国股权分析师的约400条历史研究笔记为经验数据，并在此基础上进行实验。

**📈 对比分析**

通过案例研究与主观“有用度”指标对比，展示了NFD在12周内使代理从38%到74%的有用分析率、案例回忆量、错误识别等显著提升；未进行量化基准对比。

**⚠️ 局限性**

局限包括单用户案例缺乏对照组、评估主要为主观、冷启动效能低、对组织级共享与质量度量支持不足、结晶过程依赖人工验证。

---

## 239. TopGen: Learning Structural Layouts and Cross-Fields for Quadrilateral Mesh Generation

**arXiv ID:** 2603.10606 | [PDF](https://arxiv.org/pdf/2603.10606v1)

**作者:** Yuguang Chen `[一作]` (Sun Yat-sen University), Chunchao Guo `[通讯]` (Tencent Hunyuan)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了TopGen，一个基于学习的框架，通过同时预测结构布局和交叉场来生成高质量的四边形网格。

**💡 创新点**

TopGen是首个联合预测结构布局和交叉场的四边形网格生成框架，确保外部几何边界的完整性和内部边缘流的规律性。

**🔧 技术方法**

使用了点云采样、几何感知编码器和双查询解码器等技术。

**📊 数据集**

构建了一个名为TopGen-220K的大规模四边形网格数据集，包含原始三角网格、结构布局、交叉场及其对应的四边形网格。

**📈 对比分析**

与现有的四边形重网格方法（如Instant-Meshes、QuadriFlow和QuadWild）进行比较，TopGen在几何保真度和拓扑边缘流合理性方面显著优于这些方法。

**⚠️ 局限性**

限制在于对复杂几何形状的处理能力可能仍然受到输入网格质量的影响。

---

## 240. Beyond Standard Datacubes: Extracting Features from Irregular and Branching Earth System Data

**arXiv ID:** 2603.10809 | [PDF](https://arxiv.org/pdf/2603.10809v1)

**作者:** Mathilde Leuridan `[一作]`, Martin Schultz `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种基于压缩树结构的数据超立方体（data hypercube）来表示高维、稀疏、条件性强的地球科学数据，并构建了一个集成特征提取系统（Polytope + Qubed + GribJump），实现从数据空间结构直接进行高效的特征级别数据访问。

**💡 创新点**

创新点是提出通用的数据超立方体抽象，消除了传统正交稠密立方体的完整性和正交性假设；将压缩树作为索引缓存，实现对稀疏、分支数据空间的高效查询；将特征提取与数据结构紧密耦合，避免后处理提取步骤；实现面向用户的高层次特征请求与字节级别访问的完整闭环。

**🔧 技术方法**

技术包括：压缩树（compressed tree）构造与压缩算法；集合操作（union、intersection）与树压缩；Polytope几何特征提取框架；Qubed索引构建；GribJump字节级访问；基于Polytope-Quetde-GribJump集成。

**📊 数据集**

使用了ECMWF的MARS/FDL等大规模地球系统数据，具体数据集如ERA5单层、极端事件（Extremes）以及Destination Earth的气候数字孪生（8.6M条目）进行测试。

**📈 对比分析**

与传统完整字段提取（如直接读取GRIB字段）或基于xarray/Zarr的对比，展示了构造时间线性增长，压缩线性，集合操作线性；但实际特征提取时能显著减少I/O，获取单点或轨迹时从数秒降到几秒，针对96场时间序列仅数秒；在大规模ensemble上优势更明显。

**⚠️ 局限性**

局限包括：构造索引成本高（每日一小时到一天不等），对存储系统的依赖较强（需匹配存储布局）；树结构设计需要手动排序以获得最佳性能；对极其稀疏但非结构化的数据仍可能产生冗余；需要进一步在多后端、硬件平台上的性能评估和元数据扩展。

---

## 241. AlphaFlowTSE: One-Step Generative Target Speaker Extraction via Conditional AlphaFlow

**arXiv ID:** 2603.10701 | [PDF](https://arxiv.org/pdf/2603.10701v1)

**作者:** Duojia Li `[一作]` (Xiamen University), Haizhou Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 29064 | [OpenAlex ID](https://openalex.org/A5032690182)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 AlphaFlowTSE，一种一阶生成式目标说话人提取框架，利用混合到目标的平均速度传输直接从混合波形得到目标语音，避免多步采样导致的延迟。

**💡 创新点**

核心创新包括：① 采用无 Jacobian‑vector product 的 AlphaFlow 目标，结合轨迹匹配与区间一致性 teacher‑student 监督，稳定并加速一阶训练；② 设计确定性混合‑目标线性轨迹，消除对混合比（MR）预测的依赖，提升对不同区间长度的泛化；③ 在 STFT 复数域中实现一阶推理，NFE=1，显著降低推理开销。

**🔧 技术方法**

技术手段主要包括：条件生成模型、流匹配（Flow Matching）与 AlphaFlow 训练框架、Mean‑Velocity 预测网络（基于 UDiT Transformer），以及无 JVP 的教师‑学生一致性损失。网络输入为混合与注册语音的 STFT 连接，输出为混合到目标的平均速度向量。

**📊 数据集**

训练与评估使用 Libri2Mix（干净与噪声混合）和 REAL‑T（真实对话混合）数据集。数据处理与 baseline 保持一致（3 秒窗口、复数 STFT 512 通道），并在 16 kHz 采样率下进行实验。

**📈 对比分析**

与 AD‑FlowTSE、MeanFlowTSE 以及多步扩散/流模型比较时，AlphaFlowTSE 在 Libri2Mix 上取得最高或接近最高的 PESQ、ESTOI、SI‑SDR、DNSMOS 以及相当或略优的说话人相似度；在 REAL‑T 上，零样本迁移实现最低的 ASR 错误率和最高的说话人相似度，同时保持最优的 DNSMOS，且在无 MR 预测时表现更为稳健。

**⚠️ 局限性**

局限性：① 对注册语音质量仍有一定依赖，若注册短语音噪声或说话者变化显著，性能可能下降；② 虽然一次推理速度快，但对极端多说话人或极大噪声混合的泛化仍需进一步验证；③ 训练过程中仍需调节 Alpha 参数与区间一致性权重，设置不当可能导致收敛不稳定。

---

## 242. World Model for Battery Degradation Prediction Under Non-Stationary Aging

**arXiv ID:** 2603.10527 | [PDF](https://arxiv.org/pdf/2603.10527v1)

**作者:** Kai Chin Lim `[一作]` (Independent Researcher), Khay Wai See `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出将电池衰退预测框架视为世界模型，并通过隐状态的迭代滚动实现80周期的SOH轨迹预测。

**💡 创新点**

首次将物理约束（单调性和SPM电阻-SOH关系）嵌入损失函数，并结合PatchTST+CNN编码器与动态滚动隐状态，突破传统单步回归的局限。

**🔧 技术方法**

采用1D卷积+PatchTST变压器编码器、残差MLP动力学转移、共享解码器、物理单调性与电阻一致性惩罚，并与LSTM基线及EWC持续学习进行对比。

**📊 数据集**

使用Severson LiFePO4（LFP）数据集，共138个A123单元，含原始电压、电流、温度时序数据，覆盖3个制造批次。

**📈 对比分析**

在与无滚动直回归、无物理约束的世界模型以及传统LSTM基线的对比中，滚动模型在5周期内MAE从0.0136降至0.0067，整体MAE 0.0063，显著优于LSTM 0.0209；物理约束在衰退拐点提升MAE至0.0080，EWC持续学习未获益。

**⚠️ 局限性**

研究仅局限于LFP化学，未验证在NMC、NCA等不同化学或不同充电条件下的泛化；对快速衰减细胞（如cell45）泛化不足；物理约束权重未调优；未单独评估原始时序与统计特征对模型效果的影响。

---

## 243. Training Language Models via Neural Cellular Automata

**arXiv ID:** 2603.10055 | [PDF](https://arxiv.org/pdf/2603.10055v1)

**作者:** Dan Lee `[一作]` (Independent Contributor), Pulkit Agrawal `[通讯]` (MIT)

**通讯引用:** 5273 | [OpenAlex ID](https://openalex.org/A5111774389)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用神经元胞自动机（NCA）生成合成非语言序列，对大型语言模型进行预预训练（pre‑pre‑training），随后再进行自然语言的标准预训练和任务微调，以提升语言建模效果和推理性能。

**💡 创新点**

创新点在于：①提出将NCA作为高复杂度、可控的合成训练数据源，证明其能提供比自然语言更有利的“规则推理”训练信号；②揭示注意力层是迁移最重要的模块；③展示合成数据的复杂度（如gzip压缩率、状态字母表大小）需匹配目标域以获得最佳迁移效果。

**🔧 技术方法**

技术包括：神经元胞自动机模型（二维离散网格、神经网络更新规则）、Patch 词表化、Transformer自回归训练、复杂度基采样、注意力层/MLP层迁移实验、压缩率作为复杂度度量。

**📊 数据集**

数据集：1）NCA生成的164M个token（压缩率>50%）；2）自然语言预训练语料（OpenWebText、OpenWebMath、CodeParrot）；3）基准对照数据（C4 1.6B token、Dyck语言）。

**📈 对比分析**

比较方法：对比无预预训练（scratch）、Dyck预预训练、C4预预训练。评估指标为验证 perplexity、收敛速度（达到scratch最终 perplexity所需 token 数）以及推理基准（GSM8K、HumanEval、BigBench‑Lite）的 pass@k。结果显示：NCA 预预训练可使 perplexity 降低最多 6%，收敛速度提升 1.4–1.6×，在推理基准上均实现了显著的准确率提升，甚至在部分场景超过 1.6B token 的 C4 预训练。

**⚠️ 局限性**

局限性：①尚未证明 NCA 能完全替代自然语言预训练；②合成数据的复杂度调参依赖经验，缺乏系统化的生成器指导方法；③不同字母表大小与网格尺寸对迁移效果影响不确定，需进一步研究；④在更大模型或更长训练时间下的效果尚未验证。

---

## 244. Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection

**arXiv ID:** 2603.10598 | [PDF](https://arxiv.org/pdf/2603.10598v1)

**作者:** Yawen Yang `[一作]` (Hefei University of Technology), Meng Wang `[通讯]` (Hefei University of Technology)

**通讯引用:** 42433 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于冻结的 CLIP‑ViT 中层特征的层间转换差异（Latent Transition Discrepancy，LTD）检测框架，用动态层选择和双分支结构联合捕捉真实与合成图像在层间演化上的差异，显著提升合成图像检测性能。

**💡 创新点**

创新点在于首次利用 CLIP‑ViT 中层的“层间转换差异”作为判别信号，并通过可学习的 Gumbel‑Softmax 机制动态挑选最具区分度的层组，结合双分支网络同时建模全局特征一致性和局部层间变化。

**🔧 技术方法**

技术实现主要包括冻结 CLIP‑ViT‑L/14 作为特征提取器、Gumbel‑Softmax 动态层选择、计算相邻层 CLS 之间差分得到 LTD 特征、加入位置编码、双分支 ViT Transformer 共享权重以及最终的分类头。

**📊 数据集**

实验使用 UFD、DRCT‑2M 与 GenImage 三大基准数据集，并在训练时采用 ProGAN 生成的假图与 LSUN/ImageNet 真实图。

**📈 对比分析**

在所有基准上与主流方法（ForgeLens、FatFormer、D³ 等）对比，LTD 在 UFD 平均准确率达 96.90%、在 DRCT‑2M 达 99.54%、在 GenImage 进一步超越第二名约 2.44%，并在 JPEG 压缩与下采样等后处理下表现出更强的鲁棒性。

**⚠️ 局限性**

局限性主要在于依赖大规模 CLIP‑ViT 模型，计算和存储成本较高；对极端后处理（如强度压缩或显著噪声注入）的鲁棒性仍有待进一步验证。

---

## 245. Structured Linked Data as a Memory Layer for Agent-Orchestrated Retrieval

**arXiv ID:** 2603.10700 | [PDF](https://arxiv.org/pdf/2603.10700v1)

**作者:** Andrea Volpini `[一作]` (WordLift), David Riccitelli `[通讯]` (WordLift)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比不同文档表示（纯 HTML、HTML+JSON‑LD、增强实体页及增强+版）与检索模式（标准 RAG 与代理式 RAG）的组合，实验验证结构化链接数据能显著提升 Retrieval‑Augmented Generation 系统的准确率与完整度，并提出面向 AI 代理的“增强实体页”格式。

**💡 创新点**

① 设计了兼顾人类阅读与 AI 代理可发现的实体页格式，将 Schema.org JSON‑LD、可点击实体链接、面向 AI 的指令与神经检索能力融合；② 在四个行业垂直领域验证此格式在标准与代理式 RAG 上均可获得近 30% 的准确率提升；③ 提出了 SEO 3.0 的三维度（引用、推理、行动）评价框架。

**🔧 技术方法**

使用 Vertex AI Vector Search 2.0 进行向量检索；Google Agent Development Kit（ADK）实现 ReAct‑style 代理推理与工具调用；Gemini 2.5 Flash 生成答案，Gemini 3.0 Flash 进行自动化评估；WordLift 知识图谱平台提供 Schema.org 结构化数据与可解析的实体 URI；JSON‑LD 与 Schema.org 语义标记；自定义 “增强实体页” 模板。

**📊 数据集**

四个行业垂直领域的 158 个实体（WordLift Blog、Express Legal Funding、SalzburgerLand、BlackBriar），共 349 个查询，构成 2,443 次评估；每个实体产生 3 种文档变体（纯 HTML、HTML+JSON‑LD、增强实体页）及 Enhanced+ 版。

**📈 对比分析**

采用 3×2 因子实验（3 文档格式 × 2 检索模式）+Enhanced+ 版，共 7 条实验条件；使用准确率（1–5）、完整度（1–5）和 Grounding（仅标准 RAG）三项指标；结果显示：JSON‑LD 单独提升 0.17 分（p≈0.024，d=0.18）；增强实体页在标准 RAG 上准确率提升 29.6%（Δ=1.04，p<10⁻²¹，d=0.60），在代理式 RAG 上提升 29.8%（Δ=1.04，p<10⁻²¹，d=0.61）；Enhanced+ 版最高得分（准确率 4.85/5、完整度 4.55/5），但与增强实体页差异不显著（Δ=0.06，p=1.0，d=0.08）；代理式检索相较标准提升约 13% 准确率、20% 完整度。

**⚠️ 局限性**

1) 在 Vertex AI Vector Search 中，JSON‑LD 可能被截断，导致效果低估；2) 未对信息内容与展示形式进行单独消融，无法明确区分两者对性能的贡献；3) 评估使用 KG 生成的真值，存在循环性风险；4) 仅使用单一检索系统，结果对其他结构化检索方案的适用性未知；5) LLM 评判模型可能带来偏差；6) 数据集规模有限，缺乏大规模验证。

---

## 246. An Automated Radiomics Framework for Postoperative Survival Prediction in Colorectal Liver Metastases using Preoperative MRI

**arXiv ID:** 2603.10216 | [PDF](https://arxiv.org/pdf/2603.10216v1)

**作者:** Muhammad Alberb `[一作]` (University of Toronto), Helen Cheung `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出了一个全自动化的前置MRI框架，用于结直肠肝转移（CRLM）术后生存预测，包括基于部分标注的解剖感知分割管线和Radiomics预测管线。

**💡 创新点**

创新点在于①利用SAMONAI将SAM扩展为3D点播分割；②开发SurvAMINN，结合自动编码器与多实例学习，直接处理右删失数据并突出高危转移灶；③在解剖感知管线中融合肝、CRLM与脾脏分割，减少非肝部误检。

**🔧 技术方法**

核心技术包括Promptable基础模型SAMONAI、UNETR细调、PyRadiomics特征提取、SurvAMINN（AutoEncoder+MIL+CoxPH）以及基于Cox回归的评估。

**📊 数据集**

使用了227例CRLM患者的gadoxetate增强T1加权MRI（1.5/3T），包含预/后对比图像，手工标注的转移灶、肝脏和脾脏；部分样本用于评估，部分用于训练。

**📈 对比分析**

与传统MedSAM、MedSAM+UNETR、手工分割+传统机器学习（SVM、RSF）以及临床/分子指标比较，SurvAMINN在C-index上达0.69，明显优于C-index<0.66的其它方法；结合预后两相MRI并采用LSE池化可进一步提升性能。

**⚠️ 局限性**

局限包括：①预后两相融合方式粗糙，信息互补性有限；②对未标注小病灶的检测仍不理想，可能导致选择偏倚；③分割依赖伪标签，样本选择敏感；④未充分利用肝脏/脾脏特征；⑤仅评估术后患者，缺乏非手术化疗预测。

---

## 247. Beyond the Illusion of Consensus: From Surface Heuristics to Knowledge-Grounded Evaluation in LLM-as-a-Judge

**arXiv ID:** 2603.11027 | [PDF](https://arxiv.org/pdf/2603.11027v1)

**作者:** Mingyang Song `[一作]` (Tencent), Chenning Xu `[通讯]` (Tencent)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5057227372)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对32款LLM、3名前沿评审、100项写作任务、11个温度设置共105,600个评估实例进行大规模实验，揭示LLM评审中的“评估幻觉”并验证知识驱动的评估表格生成方法（MERG）能够削弱这种幻觉。

**💡 创新点**

创新点在于首次将“评估幻觉”概念化并量化，提出通过系统的知识激活与元认知反思四阶段流程（MERG）动态生成任务专属评估维度，从而显著降低表面表格共识，提升评估深度。

**🔧 技术方法**

本文采用了系统化的评估方法：对评审者进行四阶段MERG流程、对评审者和模型进行多温度评估、使用Spearman、Pearson、ICC等统计指标，并与传统静态评估表格进行对照实验。

**📊 数据集**

使用的数据集为WritingBench（覆盖12-19种写作子领域、50英文、50中文任务）和自制的Pitch Deck评估案例，保证了任务多样性和跨语言性。

**📈 对比分析**

对比方法显示，MERG相比基线评估在Codified领域提升了约22-27%的协同度，在Subjective领域则下降约6%，整体样本层一致性从0.72降至0.51，表明MERG能去除表面共识并更准确捕捉质量差异。

**⚠️ 局限性**

局限性包括：仅评估写作任务，缺乏对代码或数理推理等领域的验证；评审者均为商用LLM，无法推广到开源评审器；未进行人工基准对照；MERG的消融实验仅在两款模型上完成；对RLAIF后续影响的实验规模有限。

---

## 248. On the Learning Dynamics of Two-layer Linear Networks with Label Noise SGD

**arXiv ID:** 2603.10397 | [PDF](https://arxiv.org/pdf/2603.10397v1)

**作者:** Tongcheng Zhang `[一作]` (Shanghai Jiao Tong University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 17000 | [OpenAlex ID](https://openalex.org/A5087158377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析并验证了在两层线性网络中，加入标签噪声的SGD会导致权重逐渐减小，模型从lazy进入rich学习阶段。

**💡 创新点**

首次揭示标签噪声通过二层网络第二层权重的振荡驱动第一层权重衰减，从而实现lazy-to-rich转移，并将这一机制推广到SAM。

**🔧 技术方法**

使用理论分析（动力学方程、Markov模拟）、梯度下降和全批SAM优化算法，以及对比实验。

**📊 数据集**

在合成数据和CIFAR‑10（ResNet‑18、WideResNet子样本）上进行实验。

**📈 对比分析**

与普通SGD及线性化模型对比，标签噪声SGD和SAM在测试误差、准确率和稀疏性上均优于基线，提升约1.5%准确率。

**⚠️ 局限性**

仅在两层线性网络及回归任务上证明，尚未考虑非线性激活、分类任务以及大规模真实数据的理论通用性。

---

## 249. Targeted Bit-Flip Attacks on LLM-Based Agents

**arXiv ID:** 2603.10042 | [PDF](https://arxiv.org/pdf/2603.10042v1)

**作者:** Jialai Wang `[一作]` (National University of Singapore), Ee-Chien Chang `[通讯]` (National University of Singapore)

**通讯引用:** 4617 | [OpenAlex ID](https://openalex.org/A5105408906)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对大型语言模型代理的定向位翻转攻击框架，利用多阶段执行流程的两种攻击面：最终输出导向和工具调用操纵。

**💡 创新点**

首次将位翻转攻击迁移到多阶段LLM代理，统一目标函数并加入注意力增强和教师强制，提出优先搜索策略以高效识别关键位。

**🔧 技术方法**

使用统一的目标函数（交叉熵+注意力增强+教师强制），梯度基的关键位优先搜索，评估采用8位量化的LLM模型。

**📊 数据集**

在 WebShop 数据集上评测 prompt 级和内部触发攻击，在 ToolBench 数据集上评测调用攻击。

**📈 对比分析**

与 TBT、TrojViT、Flip‑S 三种图像分类器位翻转基线对比，实验显示本方法在 ASR 上可达 90–99%，比基线提升 30–60 个百分点，同时保持高 CDA；在相同 bit‑flip 预算下更快达到高成功率。

**⚠️ 局限性**

对防御方案的探索有限，当前对关键位的屏蔽效果不明显；方法仍依赖对模型参数完全了解，且主要针对单代理，未验证多代理或在线场景的鲁棒性。

---

## 250. Designing Service Systems from Textual Evidence

**arXiv ID:** 2603.10400 | [PDF](https://arxiv.org/pdf/2603.10400v1)

**作者:** Ruicheng Ao `[一作]` (Institute for Data, Systems, and Society, Massachusetts Institute of Technology), David Simchi-Levi `[通讯]` (Institute for Data, Systems, and Society, Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在文本式性能评估场景下，用大语言模型（LLM）生成代理分数并结合有限的人类审核来识别最佳服务系统配置的固定置信度最佳臂识别方法。

**💡 创新点**

创新点在于：① 证明仅靠偏差代理无法确定最佳配置；② 通过逆概率加权（IPW）残差校正得到无偏估计并构造时间均匀置信序列；③ 提出 PP-LUCB 算法，联合决定评估和审核策略，使用 Neyman 分配聚焦高不确定性区域，实现成本近似最优；④ 处理审计延迟的理论保证。

**🔧 技术方法**

技术包括：代理+IPW 残差估计、时间均匀置信序列（stitched boundary）、LUCB 样本选择、Neyman 分配的审计概率、信息理论下界、延迟反馈的自适应置信序列。

**📊 数据集**

数据集：合成的 Bernoulli 与噪声代理测试；MT‑Bench（对话对比）；真实客服工单分类；队列服务设计（多层配置）等，均使用 LLM API（DeepSeek‑V3.2、gpt‑5‑nano、gpt‑4.1‑mini 等）作为代理。

**📈 对比分析**

与统一审计、价格精度、无序审计等 baseline 对比，PP‑LUCB 在 90% 以上的置信度下，平均节约 48‑50% 的审核成本，且在所有实验中都能正确识别最佳配置；在延迟反馈实验中仍保持覆盖率并仅略微增加停机时间。

**⚠️ 局限性**

局限性：① 当代理偏差导致排序改变时需要大量审核；② 代理与审核分布需要预先记录，若记录错误导致校正失效；③ 对多审稿人不一致的情况没有处理；④ 对极低信噪比的最佳配置区分仍受限，需要更大样本或更高审核率。

---

## 251. The science and practice of proportionality in AI risk evaluations

**arXiv ID:** 2603.10017 | [PDF](https://arxiv.org/pdf/2603.10017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 252. BALD-SAM: Disagreement-based Active Prompting in Interactive Segmentation

**arXiv ID:** 2603.10828 | [PDF](https://arxiv.org/pdf/2603.10828v1)

**作者:** Prithwijit Chowdhury `[一作]` (Georgia Institute of Technology), Ghassan AlRegib `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5622 | [OpenAlex ID](https://openalex.org/A5006145139)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了主动提示框架，将交互式分割视为空间主动学习问题，利用SAM模型的提示交互实现迭代改进。

**💡 创新点**

核心创新是将贝叶斯主动学习中的信息增益（BALD）迁移至空间提示选择，并通过在冻结的SAM上训练轻量级拉普拉斯近似头部实现可扩展的不确定性量化。

**🔧 技术方法**

采用Segment Anything Model (SAM) 作为基底，冻结其编码器与解码器，仅在一个小型卷积预测头上进行拉普拉斯近似贝叶斯推断；提示选择使用基于信息增益的BALD公式。

**📊 数据集**

在包含自然图像、医疗、海底和地震四大领域共16个子数据集（如MS COCO、NDD20、F3、医学US、皮肤病、结肠镜等）上进行评估。

**📈 对比分析**

与随机采样、熵采样、人工提示、基线几何提示（Saliency、K-Medoids等）及oracle对照相比，BALD-SAM在14/16个数据集的归一化ΔIoU指标中位居前二，且在大多数类别上实现了最高或第二高的最终IoU。

**⚠️ 局限性**

主要局限在于对地震等与SAM预训练分布差异较大的域仍无法突破oracle性能，且对SAM模型的预训练质量高度依赖，无法完全消除领域迁移带来的性能下降。

---

## 253. Towards Intelligent Spectrum Management: Spectrum Demand Estimation Using Graph Neural Networks

**arXiv ID:** 2603.10802 | [PDF](https://arxiv.org/pdf/2603.10802v1)

**作者:** Mohamad Alkadamani `[一作]` (Communications Research Centre Canada), Halim Yanikomeroglu `[通讯]` (Carleton University)

**通讯引用:** 21176 | [OpenAlex ID](https://openalex.org/A5035446029)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并验证基于公开部署记录的光谱需求代理，并利用分层多分辨率图注意网络（HR‑GAT）在五个加拿大城市生成高分辨率光谱需求地图。

**💡 创新点**

创新点包括：①将公开部署数据通过统计验证转化为可监管的需求代理；②设计跨尺度层级图注意模型，实现节点自适应融合多分辨率信息；③在空间分离的训练验证框架下显著提升跨城市泛化能力。

**🔧 技术方法**

使用图注意网络（Graph Attention Network）及其层级多分辨率扩展、MSE+空间正则化损失、SHAP解释等技术。

**📊 数据集**

公开部署记录（站点/频段）、运营商忙时下行吞吐量、人口与社会经济指标、移动模式、建筑与道路网、POI、夜间灯光等开放空间数据。

**📈 对比分析**

与LightGBM、XGBoost、Random Forest、CNN、单尺度GAT等八个基线进行空间分离交叉验证，HR‑GAT在CB‑CV下实现RMSE 29.3、MAE 10.9、R² 0.91，比最佳基线降低约21%；在留一城市验证中MAE 18.74，显示出更强的泛化。

**⚠️ 局限性**

局限性在于仅针对单时段快照，缺乏日夜或季节变化；对低密度地区图信号弱，影响预测；采用验证代理而非直接流量，需进一步交叉验证以提升可信度。

---

## 254. Few-Shot Adaptation to Non-Stationary Environments via Latent Trend Embedding for Robotics

**arXiv ID:** 2603.10373 | [PDF](https://arxiv.org/pdf/2603.10373v1)

**作者:** Yasuyuki Fujii `[一作]` (Ritsumeikan University), Nobutaka Shimada `[通讯]` (Ritsumeikan University)

**通讯引用:** 1520 | [OpenAlex ID](https://openalex.org/A5032660436)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于低维“趋势ID”(Trend ID)的少样本自适应框架，用于解决机器人在非平稳环境中的概念漂移问题。

**💡 创新点**

创新点：①不更新模型权重，直接估计环境状态向量实现快速适应，天然避免灾难性遗忘；②通过时序正则化与常速度状态转移模型对趋势ID进行结构化约束，抑制ID泄露与过拟合；③提供可解释的潜在空间，能够可视化并比较不同环境状态。

**🔧 技术方法**

使用技术包括：基于概率回归的神经网络（MobileNet特征提取+全连接层），趋势ID学习与梯度后向传播，Gaussian噪声增强、状态转移损失、速度一致性损失、位置一致性损失，常速度线性动力学模型。

**📊 数据集**

数据集：从三家食品工厂收集的深度图像、抓取深度与称重数据，包含两种食材（青葱、辣椒），共20个时间序列（900个样本），其中18序列训练，2序列测试。

**📈 对比分析**

方法与传统迁移学习/元学习相比，在保持模型权重不变的前提下，只需5–10个样本即可实现对新环境的快速适应；实验显示趋势ID在潜在空间中形成分离且时序连贯的轨迹，少样本适应后误差下降且不会出现灾难性遗忘，性能优于仅用参数更新的基线。

**⚠️ 局限性**

局限性：潜在空间与可解释属性（工厂、食材、机械结构）之间的几何对应关系尚不显著，可能受损失权重和正则化设计影响；常速度模型对环境快速变化的适应性有限，未来需要更表达的非线性状态转移模型。

---

## 255. Unbalanced Optimal Transport Dictionary Learning for Unsupervised Hyperspectral Image Clustering

**arXiv ID:** 2603.10132 | [PDF](https://arxiv.org/pdf/2603.10132v1)

**作者:** Joshua Lentz `[一作]` (Tufts University), James M. Murphy `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出利用未平衡Wasserstein字典学习（Unbalanced Optimal Transport Dictionary Learning）对高光谱图像(Hyperspectral Images)进行无监督聚类，生成低维权重表示后再进行谱聚类，实现自动标签分配。

**💡 创新点**

创新点在于将平衡OT字典学习中对高光谱像素归一化为概率分布的限制改为未平衡OT，允许各像素具有不同总质量，从而更好保留光谱差异、提高对异常值和噪声的鲁棒性。

**🔧 技术方法**

主要技术包括：未平衡Optimal Transport（UOT）与Entropic Regularization、Unbalanced Wasserstein barycenter 计算、梯度自动微分优化字典与权重、谱聚类（基于K近邻图谱拉普拉斯）以及Hungarian匹配做标签对齐。

**📊 数据集**

实验数据集涵盖四个公开高光谱数据集：Salinas A、Indian Pines、Pavia Centre、Pavia University，均使用其原始像素光谱作为输入。

**📈 对比分析**

与传统平衡OT字典学习（BCSC）相比，在相同超参下UOT得到的标签准确率提升约10%~20%（例如Salinas A 89% vs 68%），在最佳调参时也保持或略优；同时，在“纯度”指标上，增加少量额外聚类可以进一步提升对潜在物质类别的辨别。

**⚠️ 局限性**

主要限制为计算复杂度高，未平衡OT的Sinkhorn迭代耗时显著，单线程Python实现对大规模数据（>10k像素）不适用；此外，超参数选择高度依赖数据场景，缺乏自动化调优方法。

---

## 256. A New Tensor Network: Tubal Tensor Train and Its Applications

**arXiv ID:** 2603.10503 | [PDF](https://arxiv.org/pdf/2603.10503v1)

**作者:** Salman Ahmadi-Asl `[一作]` (Innopolis University), Andrzej Cichocki `[通讯]` (Systems Research Institute of Polish Academy of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了管道张量列（Tubal Tensor Train, TTT）分解模型，提供了两种构造算法（TTT‑SVD 与 TATCU）以及相应的误差理论，并在多种多维数据上进行了压缩与补全实验。

**💡 创新点**

创新点包括：①将 t‑product 代数与张量训练（TT）网络结构结合，得到低阶核心、线性存储；②设计了基于 T‑SVD 的固定秩序列构造与基于 FFT 的切片交替更新（ATCU）两种高效实现；③给出了与 TT‑SVD 等价的误差上界，证明了最佳逼近存在。

**🔧 技术方法**

使用的主要技术包括：t‑product 与 T‑SVD、FFT/逆FFT、张量训练网络、截断 T‑SVD、ATCU 交替更新、随机化 T‑SVD/采样加速（提议但未实现）、Parseval 定理、误差传播分析。

**📊 数据集**

实验数据集：彩色图像（Kodim 系列、Airplane、Barbara 等）、视频（Akiyo、News、Tempete、Waterfall、Foreman、Stephan）、合成缺失张量（70% 随机缺失）、高光谱图像 ROSIS Pavia Univ.（N=14）等。

**📈 对比分析**

比较方法：将 TTT 与传统 TT、T‑SVD、张量链（TC）以及基于误差阈值的对比；评价指标为 PSNR、SSIM、MSE、RMSE、ERGAS、SAM、UIQI、压缩因子、运行时间、相对误差。实验表明：在相同误差或参数量下，TTT 通常提供更高的重建质量（PSNR+2–3 dB，SSIM+0.05–0.1），压缩因子更大或运行时间更短；在视频压缩任务中，TTT 的压缩比显著优于 T‑SVD。

**⚠️ 局限性**

局限与待改进：①顺序构造时的秩选择可能导致核心不平衡，增加参数；②需要在 FFT 频域统一秩，增加实现复杂度；③目前仅实现实数域；③缺乏随机化/压缩加速的实验；④对极大尺度张量的可扩展性尚未彻底验证。

---

## 257. Event-based Photometric Stereo via Rotating Illumination and Per-Pixel Learning

**arXiv ID:** 2603.10748 | [PDF](https://arxiv.org/pdf/2603.10748v1)

**作者:** Hyunwoo Kim `[一作]` (Yonsei University), Jae-Sang Hyun `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于事件相机的光度立体法，通过单一旋转光源获取事件流来直接推断表面法向量。

**💡 创新点**

创新点在于：①利用单一连续旋转光源替代传统多光源硬件；②在事件域中构造极性求和表示，消除光照方向与阈值的校准需求；③采用轻量级多层感知器对每像素事件序列进行学习，显著提升了稀疏事件区和高动态范围场景下的鲁棒性。

**🔧 技术方法**

核心技术包括事件相机捕获、极性事件求和向量表示、基于光照轨迹的事件-法向映射的解析公式、每像素MLP网络与余弦相似度损失，以及在PyTorch下的端到端训练。

**📊 数据集**

使用合成渲染数据（Blender + Mitsuba 3）和真实事件数据（DAVIS 346 + 3D 打印对象）构建训练与验证集，覆盖 Blobby 与 Sculpture 两类模型，并对 DiLiGenT‑EV 半实景、CW 真实以及自建 CCW 真实三组数据进行评测。

**📈 对比分析**

与 EventPS 系列基线（EventPS‑OP、EventPS‑FCN、EventPS‑CNN）在三组数据上进行 MAE 对比，本文方法在平均 MAE 上均显著低于基线（如 DiLiGenT‑EV 平均 12.24°、CW 真实 12.24°、CCW 真实 9.77°），在稀疏事件区与高反射物体上表现尤为突出。

**⚠️ 局限性**

主要局限在于：①需要固定且可重复的光照轨迹，无法直接迁移到不同光照路径；②对极低事件密度区域仍存在误差，需进一步提升低信噪比时的鲁棒性；③目前仅针对单光源旋转场景，扩展到多光源或非圆形轨迹尚未验证。

---

## 258. Less is More: Decoder-Free Masked Modeling for Efficient Skeleton Representation Learning

**arXiv ID:** 2603.10648 | [PDF](https://arxiv.org/pdf/2603.10648v1)

**作者:** Jeonghyeok Do `[一作]` (Korea Advanced Institute of Science and Technology), Munchurl Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5918 | [OpenAlex ID](https://openalex.org/A5027012300)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 SLiM 框架，结合无解码器的掩码建模与对比学习，实现统一的骨架动作表征学习。

**💡 创新点**

创新点包括：1) 移除重建解码器，消除 MAE 的计算不对称；2) 采用 Semantic Tube Masking 防止模型利用局部坐标插值；3) 设计 Skeletal‑Aware Augmentations，保证几何一致性的对比样本；4) 通过教师‑学生蒸馏实现全程共享编码器。

**🔧 技术方法**

技术手段：Vision Transformer 编码器；教师‑学生 EMA 蒸馏；MFM（Masked Feature Modeling）和 GLCL（Global‑Local Contrastive Learning）双目标；语义管道掩码；骨架感知旋转、镜像、骨长缩放等增强。

**📊 数据集**

主要使用 NTU‑60、NTU‑120 与 PKU‑MMD II 三大 3D 骨架数据集进行预训练与评估。

**📈 对比分析**

与现有 CL、MAE 及其他预文本任务相比，SLiM 在线性、半监督与检索任务上均达标或领先，NTU‑60 线性评估最高 93.2%，NTU‑120 最高 83.6%，并将推理成本比 MAE 降低 7.89×。

**⚠️ 局限性**

局限性：仅在 3D 骨架数据上验证，未覆盖 2D 或噪声场景；缺少对多模态融合或大规模部署的实际评估；对极低标签或极大序列长度的泛化仍待进一步验证。

---

## 259. Domain-Adaptive Health Indicator Learning with Degradation-Stage Synchronized Sampling and Cross-Domain Autoencoder

**arXiv ID:** 2603.10430 | [PDF](https://arxiv.org/pdf/2603.10430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 260. Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems

**arXiv ID:** 2603.10053 | [PDF](https://arxiv.org/pdf/2603.10053v1)

**作者:** Wentao Wang `[一作]` (Dalian University of Technology), Guangyu Zou `[通讯]` (Dalian University of Technology)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5083864266)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于Transformer的深度强化学习框架CAADRL，用来解决单车Pickup and Delivery Problem（PDP），该框架显式利用问题的聚类结构，采用多尺度编码和层次解码来构建可行的巡回路线。

**💡 创新点**

创新点包括：① 聚类感知注意力（Cluster‑Aware Attention）——在编码器中同时进行全局自注意力和受聚类掩码限制的局部注意力，得到同时具有全局视角和局部角色感知的节点嵌入；② 动态双解码器（Dynamic Dual‑Decoder）与可学习门控机制——分别负责局部内聚类路径规划和跨聚类过渡，并通过门控自适应平衡两种决策；③ 在POMO框架下的端到端训练，利用多样本并行rollout降低方差并提升样本效率。

**🔧 技术方法**

技术主要有Transformer编码器、聚类掩码自注意力、双解码器+门控、POMO策略梯度、采样（Greedy、Sampling-1280、Sampling-12800）以及可视化与时间测评。

**📊 数据集**

使用合成的二维欧氏空间数据集，包含两种分布：聚类分布（pickup聚在(0.25,0.25)，delivery聚在(0.75,0.75)）和均匀分布；规模从PDP10到PDP80（10到80个客户节点）以及更大规模PDP200/300/500用于跨尺寸泛化测试。

**📈 对比分析**

与现有两类基线（NCS：神经协同搜索；Heter：异构注意力编码器）以及标准POMO进行比较。实验表明：在聚类实例上，CAADRL在中大规模（PDP20/PDP40/PDP80）上均能匹配或优于基线，尤其在采样预算高时表现突出；在均匀实例上，性能略逊于NCS/ Heter 在小规模，但在大规模（PDP80）时显著领先。推理时间显著低于NCS，接近POMO。

**⚠️ 局限性**

局限性：① 设计针对聚类结构，若真实场景无明显聚类可能无法充分发挥优势；② 目前仅支持单车、静态、欧氏距离的PDP；③ 需要手工定义聚类掩码（依赖已知角色与空间分布），对更复杂约束（多车、时窗、容量等）尚未验证；④ 采样解码虽然简单，但在极大规模或动态环境下可能需进一步改进。

---

## 261. Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents

**arXiv ID:** 2603.10564 | [PDF](https://arxiv.org/pdf/2603.10564v1)

**作者:** Yuanhao Li `[一作]` (University of Exeter), Wang Miao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种自微调框架，使大型语言模型（LLM）能在无线网络切片（RAN slicing）控制任务中无需手工奖励函数即可持续学习；

**💡 创新点**

创新点包括：1) 将强化学习改造成“反射马尔可夫决策过程” (R‑MDP)，使LLM可直接处理自然语言反馈；2) 设计双视角反射机制，将局部一步反思与全局轨迹评估结合；3) 通过反射器生成偏好标记，并用Kahneman‑Tversky 优化（KTO）进行偏好式微调，将长时序经验内化为模型参数；4) 通过一次轨迹即可完成六轮KTO微调，极大提升样本效率。

**🔧 技术方法**

技术手段包括：LLM Actor（Qwen3‑4B）+ Reflector（DeepSeek‑R1），R‑MDP 结构，bi‑perspective 反射，反射器生成的偏好数据，KTO 优化，ns‑3 RAN 切片仿真，比较实验用 DQN、SAC、PPO 等 RL 基线与 Reflexion LLM 代理。

**📊 数据集**

数据集：自建 ns‑3 RAN 切片仿真环境，使用GBR/Non‑GBR 流量模型（指数分布的上/下状态、比特率、包大小等），产生的交互轨迹用于训练与评估。

**📈 对比分析**

评估方法：与传统 RL 基线（DQN、SAC、PPO）及 Reflexion LLM 代理在同一仿真环境下对比；指标包括平均谱效率、重配置次数、QoS 违例次数以及综合效用。结果显示，自微调框架在仅一次轨迹训练后就获得最高效用、最高谱效率、最少重配置，优于所有基线；

**⚠️ 局限性**

局限性：LLM 推理速度较慢，难以满足实时网络控制；需要进一步压缩/蒸馏模型以实现实部署；实验仅在仿真环境中验证，真实网络环境下的表现尚待验证。

---

## 262. How To Embed Matters: Evaluation of EO Embedding Design Choices

**arXiv ID:** 2603.10658 | [PDF](https://arxiv.org/pdf/2603.10658v1)

**作者:** Luis Gilch `[一作]` (IBM Germany), Thomas Brunschwiler `[通讯]` (IBM Research - Europe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

系统评估了GeoFM模型在Earth Observation任务中嵌入设计的影响，包括骨干网络、预训练目标、层深度、空间池化和嵌入组合。

**💡 创新点**

首次在同一框架下对不同骨干、SSL目标和聚合策略进行全面比较，揭示了中间层和多目标拼接对鲁棒性与性能的提升。

**🔧 技术方法**

使用NeuCo-Bench线性探针评估、ResNet-50/ViT‑Small预训练模型、DINO、MoCo、MAE、FGMAE、SoftCon等SSL方法，以及均值/最大/最小池化和CLS拼接。

**📊 数据集**

基于SSL4EO‑S12的NeuCo‑Bench任务集，包括Sentinel‑1/2多季节图像，八个回归目标（生物量、农作物、云量、土地利用、热岛等）。

**📈 对比分析**

通过50次随机拆分的平均R²和质量分数进行比较，结果表明ViT+均值池化在多数任务上领先，ResNet中间层在连续物理量任务上更优，拼接不同SSL目标可提升鲁棒性。

**⚠️ 局限性**

仅限固定维度1D嵌入与线性探针，未涉及更复杂下游模型；标签噪声与任务多样性有限；仅评估部分骨干与SSL方案，未覆盖全部可能的空间/时间聚合；拼接增加存储需求。

---

## 263. Factor Dimensionality and the Bias-Variance Tradeoff in Diffusion Portfolio Models

**arXiv ID:** 2603.10385 | [PDF](https://arxiv.org/pdf/2603.10385v1)

**作者:** Avi Bagchi `[一作]` (University of Pennsylvania), Om Shastri `[通讯]` (University of Pennsylvania)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用条件扩散模型对股票收益进行分布建模，并基于模型生成的条件均值和协方差构建均值-方差组合

**💡 创新点**

发现因子维度的平衡是关键，既要避免低维导致的欠拟合又要避免高维导致的过拟合，从而实现最佳的泛化与超额收益

**🔧 技术方法**

扩散概率模型（denoising diffusion probabilistic model）+扩散变压器架构，局部特征归一化实现条件化

**📊 数据集**

WRDS（Wharton Research Data Services）提供的大规模股票特征与收益数据

**📈 对比分析**

与等权、经验、收缩经验三种基准组合对比，选取中等因子维度的模型获得最高累计收益，优于基准并在样本外表现更稳健

**⚠️ 局限性**

模型需要手动选择因子维度，过高维度导致过拟合，缺乏对更复杂因子结构的自动学习，且仅在月度数据上验证，缺少更广泛的鲁棒性测试

---

## 264. RAGPerf: An End-to-End Benchmarking Framework for Retrieval-Augmented Generation Systems

**arXiv ID:** 2603.10765 | [PDF](https://arxiv.org/pdf/2603.10765v1)

**作者:** Shaobo Li `[一作]` (University of Illinois), Jian Huang `[通讯]` (University of Illinois)

**通讯引用:** 9040 | [OpenAlex ID](https://openalex.org/A5066790771)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个完整的RAG（检索增强生成）系统基准框架RAGPerf，能够拆分嵌入、索引、检索、重排、生成等模块，支持多模态数据、可配置的工作负载生成器以及低开销的系统与质量监控；

**💡 创新点**

提出端到端可复现的RAG基准框架，提供细粒度资源利用与质量评估；通过抽象接口实现多种向量数据库、嵌入模型、重排与生成模型的无缝切换；构建动态更新与混合查询的工作负载，系统化研究更新策略对延迟与准确率的影响；

**🔧 技术方法**

Python实现，使用HuggingFace生态嵌入与重排模型；vLLM作为LLM推理后端；抽象层统一LanceDB、Milvus、Qdrant、Chroma、Elasticsearch等向量数据库；利用NVML监控GPU；采用Ragas框架评估生成质量；使用多模态处理流程（OCR、ColPali、Whisper）；

**📊 数据集**

公开数据集包括Wikipedia（文本）、Arxiv（PDF）、GitHub Code、People’s Speech（音频）；结合Natural Questions、ML QA等问答集；为更新操作生成合成更新文本与对应问答对；

**📈 对比分析**

通过对比不同向量数据库、索引方式、嵌入维度、批量大小、重排深度等配置，测量端到端延迟、吞吐量、CPU/GPU/内存占用、磁盘I/O；实验结果显示：文本管线生成阶段占比最高，向量数据库对延迟影响有限；多模态索引受OCR/视觉嵌入成本主导；GPU内存是主瓶颈，主机内存不足导致索引迁移到磁盘显著降低吞吐；工作负载生成器能量化更新策略对延迟与准确率的权衡；总体基准开销低于0.2%；

**⚠️ 局限性**

受限于单机GPU+CPU环境，无法覆盖多节点扩展与分布式训练；静态模型驻留GPU导致内存低效；更新处理仍依赖临时平面索引，更新热点未完全解决；仅覆盖已实现的模型与数据库，无法覆盖所有新兴RAG技术；

---

## 265. P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video

**arXiv ID:** 2603.10551 | [PDF](https://arxiv.org/pdf/2603.10551v1)

**作者:** Longan Wang `[一作]` (National University of Singapore), Wei Tsang Ooi `[通讯]` (National University of Singapore)

**通讯引用:** 2818 | [OpenAlex ID](https://openalex.org/A5072587271)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于层次递进的 2D 高斯剖分（Layered Progressive 2D Gaussian Splatting）框架，用于可扩展的图像和视频编码与解码，支持从粗到细的逐层重建。

**💡 创新点**

创新点包括：① 将 2D 高斯剖分按层组织成基底层与多级增强层，实现逐层可扩展性；② 引入联合循环训练策略，避免层间优化冲突并保证进度稳定；③ 在每层训练中同时监督最高层与选定中间层，形成一致的优化轨迹；④ 结合高斯剖分裁剪、增补与动态关键帧选择，提升压缩效率与动态场景适应；⑤ 在量化阶段采用时序感知微调、非对称量化和向量量化，保持精度。

**🔧 技术方法**

核心技术包括 2D 高斯剖分原语、可微光栅化、Adan 优化器、联合循环训练、Gaussian Splat Pruning (GSP)、Gaussian Splat Augmentation (GSA)、Dynamic Key-frame Selection (DKS)、细粒度量化（浮点、非对称量化、向量量化）。

**📊 数据集**

使用的评估数据集：图像方面采用 Kodak（24 张 768×512 图）和 DIV2K 验证集（100 张 2K 图）；视频方面使用 UVG 1080p 数据集（第一帧 50 帧，每帧 1920×1080）。

**📈 对比分析**

与三类基线对比：① 单一全尺度高斯模型后裁剪得到的可扩展性；② 每层单独训练的非可扩展模型（上限）；③ 传统的分层逐层训练（LIG）。评估指标为 PSNR、MS‑SSIM、LPIPS（图像）和 PSNR、MS‑SSIM、VMAF（视频）。实验表明，PGSVC 在所有层级上均优于分层逐层训练，PSNR 提升约 1.9–2.6 dB；在率失真曲线上接近上限，且相较于 SHVC 标准的 gap 进一步缩小。

**⚠️ 局限性**

局限性：编码时间高（约 720 秒/帧），尚未优化速度；压缩质量仍略低于高优化的标准编解码器（如 SHVC）；可扩展性引入的质量损失仍约 1–1.1 dB；对实时场景或大规模部署的适应性仍待提升。

---

## 266. Consumer Rights and Algorithms

**arXiv ID:** 2603.10022 | [PDF](https://arxiv.org/pdf/2603.10022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 267. Poisson Sampling over Acyclic Joins

**arXiv ID:** 2603.10982 | [PDF](https://arxiv.org/pdf/2603.10982v1)

**作者:** Liese Bekkers `[一作]` (Hasselt University), Stijn Vansummeren `[通讯]` (Hasselt University)

**通讯引用:** 2112 | [OpenAlex ID](https://openalex.org/A5022558461)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在无环连接查询上进行 Poisson 采样的算法，并给出了近似实例最优的 Index‑and‑Probe 方案。

**💡 创新点**

创新点在于将无环连接的随机访问索引与位置采样相结合，设计了链式与非链式 Shredded 结构，并通过位置采样方法（Bernoulli、几何、二项式）实现了高效的 Poisson 采样；同时证明该算法在理论上接近实例最优。

**🔧 技术方法**

使用列式存储中的 Shredded Yannakakis（链式和非链式）实现，基于嵌套半连接、前缀向量和位置采样技术；实现了随机访问索引、位置采样和采样查询的完整管道。

**📊 数据集**

使用了两个基准：1) 由 join‑order benchmark 与 real‑world 数据生成的合成查询集合；2) 基于比利时人口的真实接触概率数据（约 1.1×10⁷ 人）。

**📈 对比分析**

与传统的全连接 + Bernoulli（BinaryJoin+Bernoulli）、Yannakakis+flatten 等基线方法进行比较。链式索引 + 几何位置采样（IdxGeom‑Chained）在大多数查询和采样概率下比基线快 1.5–6 倍；在统一采样和 Poisson 采样场景中均表现优异。

**⚠️ 局限性**

仅适用于无环查询；对高采样概率时性能下降；在循环查询、不同存储引擎或更大规模数据上的实验仍需进一步验证。

---

## 268. RandMark: On Random Watermarking of Visual Foundation Models

**arXiv ID:** 2603.10695 | [PDF](https://arxiv.org/pdf/2603.10695v1)

**作者:** Anna Chistyakova `[一作]` (Trusted AI Research Center), Mikhail Pautov `[通讯]` (AXXX)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对视觉基础模型（VFM）提出一种名为 RandMark 的水印嵌入与验证方法，将二进制水印注入模型隐藏层表示，随后通过随机输入变换和解码器提取水印，实现模型所有权验证。

**💡 创新点**

创新点在于：①将水印嵌入隐藏表示而非模型权重，适配多任务 VFM；②采用随机化触发图像与概率统计检测，提供可理论证明的误检上界；③实现对微调、知识蒸馏、稀疏剪枝等功能扰动的鲁棒性。

**🔧 技术方法**

技术包括：轻量级编码器-解码器网络、随机噪声输入变换、联合训练最小化水印误差与方差、基于统计阈值的二元假设检验、Chernoff/Chernoff-Hoeffding 上界分析。

**📊 数据集**

使用 ImageNet（1000张随机子集）做水印注入；下游任务数据集为 E‑commerce Product Images（商品分类）和 FoodSeg103（食物分割），评估微调与剪枝下的鲁棒性。

**📈 对比分析**

与基线方法（随机平滑、ADV‑TRA、IPGuard 等）对比。实验显示 RandMark 在 CLIP、DINOv2 上，在 20%/40% 剪枝、10 轮微调后检测率均超过 0.9，且误检率极低；相比之下基线方法在功能扰动后大幅下降或失效。

**⚠️ 局限性**

局限性：仅在两类 VFM（CLIP、DINOv2）及有限的下游任务上验证；对极端或自适应攻击（如模型重构、深度蒸馏后权重变换）未全面评估；编码解码网络增加少量计算与存储开销。

---

## 269. GeoSense: Internalizing Geometric Necessity Perception for Multimodal Reasoning

**arXiv ID:** 2603.10370 | [PDF](https://arxiv.org/pdf/2603.10370v1)

**作者:** Ruiheng Liu `[一作]` (University of Science and Technology of China), Xiaojun Chang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 19460 | [OpenAlex ID](https://openalex.org/A5034967388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出GeoSense框架，让多模态大型语言模型能够自主判断是否需要使用3D几何信息以提升空间推理能力。

**💡 创新点**

创新点在于将几何信息视为可选资源，并通过两阶段训练（几何对齐 + 空间感知微调）让模型内部学习“几何必要性”判别，而非硬编码规则。

**🔧 技术方法**

采用独立几何输入通道、线性投影对齐、触发令牌机制、以及基于模型自身推理差异构建的自适应训练数据集。

**📊 数据集**

使用VSI‑590K、SophiaVL‑R1‑130K、Mantis‑Instruct、NLVR2、LLaVA‑Hound‑64K等多模态与空间推理数据集进行训练与评估。

**📈 对比分析**

与多种SOTA MLLM（Qwen2.5‑VL‑3B/7B、InternVL3‑2B、VG‑LLM等）和空间专用模型（SpatialLadder‑3B、ViLASR‑7B等）对比，GeoSense在空间推理基准上达成SOTA，同时保持或提升一般视觉推理得分，激活几何的比例约35%（相较于100%硬融合）。

**⚠️ 局限性**

局限性包括：几何特征主要依赖VGGT编码器，未覆盖点云等更丰富3D表示；触发机制对极端输入鲁棒性不足；在动态视角或运动场景下表现仍有待提升。

---

## 270. OSUM-Pangu: An Open-Source Multidimension Speech Understanding Foundation Model Built upon OpenPangu on Ascend NPUs

**arXiv ID:** 2603.10862 | [PDF](https://arxiv.org/pdf/2603.10862v1)

**作者:** Yujie Liao `[一作]` (Northwestern Polytechnical University), Lei Xie `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 9926 | [OpenAlex ID](https://openalex.org/A5066245750)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于Ascend NPU的全非CUDA开源语音理解框架OSUM-Pangu。

**💡 创新点**

创新点在于将openPangu-7B LLM与OSUM语音框架结合，实现全非CUDA实现并引入意图感知训练。

**🔧 技术方法**

采用了Whisper中间声学编码器、CNN+Transformer模态适配器、LoRA微调、三阶段训练等技术。

**📊 数据集**

使用了OSUM任务集、Alpaca-CoT、CosyVoice、Intent-Instruction Set（80k+）等多任务数据集。

**📈 对比分析**

与GPU基准Qwen2-Audio和OSUM比较，任务准确率相当，指令跟随率达90.2%，在多数任务上与GPU模型持平。

**⚠️ 局限性**

局限在于对极长语音和复杂指令的鲁棒性不足，部分任务仍低于GPU模型。

---

## 271. GR-SAP: Generative Replay for Safety Alignment Preservation during Fine-Tuning

**arXiv ID:** 2603.10243 | [PDF](https://arxiv.org/pdf/2603.10243v1)

**作者:** Zhouxiang Fang `[一作]` (Rice University), Hanjie Chen `[通讯]` (Rice University)

**通讯引用:** 46535 | [OpenAlex ID](https://openalex.org/A5100381999)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Generative Replay for Safety Alignment Preservation (GR-SAP) 的框架，在微调过程中利用模型自身合成的安全对齐数据进行混合，从而在没有原始安全对齐数据的情况下保持LLM的安全性。

**💡 创新点**

创新点在于将生成式重放（Generative Replay）技术引入安全对齐领域，利用自我合成的数据替代缺失的原始对齐数据，并通过定制化提示、查询过滤与响应修订实现高质量的安全数据；理论分析证明合成数据可近似原始对齐分布。

**🔧 技术方法**

技术包括：基于大模型的查询生成与自回归响应合成、三阶段查询过滤（困惑度阈值、语义去重、相关性筛选）、响应修订（引入拒绝策略）、混合微调（SFT）以及正则化控制安全间隙。

**📊 数据集**

使用了多种模型（OLMo2、Llama3、Qwen2.5、Mistral）以及五个下游任务（GSM8K、MATH、HellaSwag、Winogrande、MedQA）和四个安全评测数据集（对话安全测试等）。

**📈 对比分析**

与原始对齐数据、Aegis、Beavertails等公开安全数据集进行对比。实验表明 GR‑SAP 在保持下游准确率（误差≤1%）的同时，将有害输出比例显著降低（如在OLMo2上从10.5%降至1.0%），并优于公开数据集，在某些模型上与原始对齐数据表现相当。

**⚠️ 局限性**

局限性包括：需要预先训练好的LLM来生成合成数据，合成数据的质量仍受提示和模型自身偏差影响；在高安全要求场景下，过大混合比例可能导致安全性能回升；此外，仅在少数模型公开对齐数据的情况下验证了与原始数据的等价性，未覆盖全部模型。

---

## 272. Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure

**arXiv ID:** 2603.10254 | [PDF](https://arxiv.org/pdf/2603.10254v1)

**作者:** Davide Tugnoli `[一作]` (University of Trieste), Giovanni Cinà `[通讯]` (University of Amsterdam)

**通讯引用:** 350 | [OpenAlex ID](https://openalex.org/A5044955303)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

探究并改进了TabPFN在合成表格数据生成中的因果结构意识，解决了顺序敏感导致的伪相关问题。

**💡 创新点**

提出了基于完整DAG的因果条件生成和基于CPDAG的部分因果条件生成两种策略，使模型在不同因果知识水平下更稳健。

**🔧 技术方法**

使用Transformer‑基础的TabPFN模型，结合因果图条件采样（DAG/CPDAG）以及原始的随机排列平均化技术。

**📊 数据集**

在手工设计的含碰撞器的SCM、Microsoft CSuite基准六个数据集以及UVA/Padova糖尿病模拟器生成的38维临床数据集上进行实验。

**📈 对比分析**

与原始顺序条件、顶点顺序、以及逆顺序条件进行对比；结果显示DAG‑aware策略在结构相似度、分布相似度和隐私保护指标上显著优于基线，CPDAG在足够有向边时也有提升；在ATE保真度上同样表现更佳。

**⚠️ 局限性**

主要局限在于需要完整或足够的因果图知识；CPDAG的有效性受边向化质量影响；对多样化因果发现算法和其他因果估计指标的适用性尚未充分验证。

---

## 273. $μ$Ed API: Towards A Shared API for EdTech Microservices

**arXiv ID:** 2603.10014 | [PDF](https://arxiv.org/pdf/2603.10014v1)

**作者:** Maximillan Sölch `[一作]` (Technical University of Munich), Stephan Krusche `[通讯]` (Technical University of Munich)

**通讯引用:** 5986 | [OpenAlex ID](https://openalex.org/A5044900523)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出并定义了用于教育微服务的通用平台无关API（Ed API），并由四所高校共同开发与验证

**💡 创新点**

创新点在于为教育微服务设计了可插拔、可部分实现的标准接口，解决学习平台锁定问题并实现跨机构互操作性

**🔧 技术方法**

使用OpenAPI 3.1规范描述接口，结合微服务架构，并在聊天能力中采用通用大型语言模型（LLM）技术

**📊 数据集**

未使用具体公开数据集，主要以四所高校内部现有自动评测与聊天系统为实现依据

**📈 对比分析**

未进行实验对比或性能评估，论文侧重设计过程与跨机构评审，说明API在实际部署中的兼容性和可扩展性

**⚠️ 局限性**

局限在于目前仅实现了评测与聊天两项核心能力，且设计偏向STEM院校需求，尚缺乏K-12及非STEM领域的验证

---

## 274. TreeON: Reconstructing 3D Tree Point Clouds from Orthophotos and Heightmaps

**arXiv ID:** 2603.10996 | [PDF](https://arxiv.org/pdf/2603.10996v1)

**作者:** Angeliki Grammatikaki `[一作]` (TU Wien), Manuela Waldner `[通讯]` (TU Wien)

**通讯引用:** 630 | [OpenAlex ID](https://openalex.org/A5061106600)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出TreeON框架，利用单张正射影像和DSM重建细致的树木3D点云。

**💡 创新点**

引入几何监督与可微阴影/轮廓损失的联合训练策略，无需物种标签或实测激光数据即可学习树木点云。

**🔧 技术方法**

神经网络学习框架、程序化树木合成数据、几何监督、可微阴影与轮廓损失以及基于DSM的稀疏数据输入。

**📊 数据集**

使用从程序化树木模型生成的合成点云数据进行训练，测试时使用真实的正射影像和DSM。

**📈 对比分析**

与现有方法进行定量和定性比较，结果显示TreeON在重建质量与覆盖率上优于对手，并在真实数据上具有良好泛化能力。

**⚠️ 局限性**

受限于合成训练数据与真实世界的域差距，可能在极端树种或高度遮挡情况下表现不足；同时对输入DSM质量敏感。

---

## 275. CUPID: A Plug-in Framework for Joint Aleatoric and Epistemic Uncertainty Estimation with a Single Model

**arXiv ID:** 2603.10745 | [PDF](https://arxiv.org/pdf/2603.10745v1)

**作者:** Xinran Xu `[一作]` (Nanyang Technological University), Xiuyi Fan `[通讯]` (Nanyang Technological University)

**通讯引用:** 999 | [OpenAlex ID](https://openalex.org/A5101917609)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种名为 CUPID 的通用插件模块，能够在不修改或重新训练基模型的前提下，在任何中间层同时估计模型的随机不确定性（aleatoric）和知识不确定性（epistemic）。

**💡 创新点**

创新点在于：①将不确定性估计拆分为两个互补分支（Bayesian identity mapping 与结构化扰动的特征重构），②在单一轻量级模块中实现两种不确定性的联合估计，并通过层级插入提供可解释的“内部来源”视图。

**🔧 技术方法**

技术手段包括：学习可变方差的贝叶斯归一映射来回归 aleatoric；利用最大化特征差异同时保持输出一致性的重构分支来捕获 epistemic；联合损失与差分特征正则化共同训练。

**📊 数据集**

实验数据集涵盖医学影像（GLV2、HAM10000、ACRIMA、PAPILA、IXI MRI）、常规图像（CIFAR‑10、Set5、Set14、BSDS100）以及超分辨率模型 ESRGAN。

**📈 对比分析**

通过与 MC Dropout、Rate‑in、PostNet、IBRUE、BayesCap 等主流基线对比，CUPID 在误分类检测、OOD 检测与超分辨率误差相关度等指标上均保持或超过 state‑of‑the‑art，表现出显著提升的 AUC、AURC、Spearman、Pearson、AUSE 等性能。

**⚠️ 局限性**

局限性包括：对不同层级位置与扰动设计的敏感性；在极端域外样本时仍需更大容量或更复杂的扰动策略；缺少对模型内部决策过程的深入解释；在极大规模模型上引入 CUPID 仍会增加一定的推理成本。

---

## 276. Report for NSF Workshop on Algorithm-Hardware Co-design for Medical Applications

**arXiv ID:** 2603.10976 | [PDF](https://arxiv.org/pdf/2603.10976v1)

**作者:** Peipei Zhou `[一作]` (Brown University), Yiyu Shi `[通讯]` (University of Notre Dame)

**通讯引用:** 5337 | [OpenAlex ID](https://openalex.org/A5000141831)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

总结并提出NSF关于医学算法-硬件协同设计的研究路线图，聚焦四大主题并给出系统级投资建议。

**💡 创新点**

首次将医疗技术从硬件/算法孤立分离，转向端到端的协同设计，强调人机协作、数据基础设施与可验证性。

**🔧 技术方法**

利用现有医疗数据集、基准、软件定义硬件框架、数字孪生与可解释AI等技术。

**📊 数据集**

引用的公开数据集包括UK Biobank、MIMIC‑III、AI‑READI、iCareLoop等。

**📈 对比分析**

报告未包含新的实验对比，仅基于会议讨论与现有文献，提出的建议与指标在研讨会中得到共识。

**⚠️ 局限性**

缺乏实证评估、标准化数据与验证平台，转化路径与监管障碍仍未彻底解决。

---

## 277. PivotAttack: Rethinking the Search Trajectory in Hard-Label Text Attacks via Pivot Words

**arXiv ID:** 2603.10842 | [PDF](https://arxiv.org/pdf/2603.10842v1)

**作者:** Yuzhi Liang `[一作]` (Guangdong University of Foreign Studies), Xia Li `[通讯]` (Guangdong University of Foreign Studies)

**通讯引用:** 38623 | [OpenAlex ID](https://openalex.org/A5100445622)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种硬标签黑盒查询高效攻击框架 PivotAttack，利用多臂赌博机识别并扰动关键词组（Pivot Set）以实现高攻击成功率。

**💡 创新点**

创新点在于“inside‑out”策略，将攻击焦点放在模型预测的关键词集合而非逼近决策边界，并通过 KL‑LUCB 多臂赌博机精确估计保留精度，考虑词间交互。

**🔧 技术方法**

采用多臂赌博机（KL‑LUCB）、词嵌入相似度检索、动态扰动阈值、语义相似度衡量等技术。

**📊 数据集**

使用 Yelp、Yahoo、MR、Amazon、SST‑2、SNLI、MultiNLI 等文本分类和蕴含数据集，以及 WordCNN、WordLSTM、BERT、ALBERT、DistilBERT、Qwen2.5、Gemma3 等多种模型。

**📈 对比分析**

在 100 查询预算下，与 HyGloadAttack、VIWHard、HLBB、TextHoaxer、LeapAttack、TextHacker、LimeAttack 等基线比较，PivotAttack 在大多数模型和数据集上获得更高的攻击成功率（ASR）且扰动率更低。

**⚠️ 局限性**

局限性：KL‑LUCB 组件查询成本较高，导致目前使用贪婪搜索而非更优的 beam search；需进一步降低多臂赌博机的查询消耗。

---

## 278. COMIC: Agentic Sketch Comedy Generation

**arXiv ID:** 2603.11048 | [PDF](https://arxiv.org/pdf/2603.11048v1)

**作者:** Susung Hong `[一作]` (University of Washington), Steve Seitz `[通讯]` (University of Washington)

**通讯引用:** 29781 | [OpenAlex ID](https://openalex.org/A5068254275)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种完全自动化的多代理框架 COMIC，用于从角色描述和背景素材生成1-2分钟的情景喜剧视频；

**💡 创新点**

核心创新在于采用多岛屿竞争与人类偏好对齐的 LLM 评审委员会，迭代竞争驱动脚本和视频的自我改进，并实现了可扩展的测试时缩放；

**🔧 技术方法**

利用 LLM 生成脚本、导演、评论者；多岛遗传算法进行脚本进化；脚本驱动的分镜、分镜迭代与视频生成器；通过多轮评审与单淘汰赛选优；

**📊 数据集**

构建了 4,940 条 YouTube 笑点视频数据集（包括 Key & Peele、SNL 等），并通过对其观看量归一化提取观众参与度作为自动评估标准；

**📈 对比分析**

与基准模型 Veo 3.1、Sora 2、VideoGen-of-Thought、MovieAgent 以及人类评估对比；在多项指标（趣味性、观看更多、脚本、叙事、真实感、一致性）上均显著优于基线，自动评估的胜率接近中等水平人类喜剧；

**⚠️ 局限性**

局限包括：高算力消耗（需多轮迭代）；仅利用观看量作为幽默度指标，可能受点击诱饵、算法推广干扰；音效与多模态细节未充分覆盖；缺乏原创性与版权归属分析。

---

## 279. 3-D Trajectory Optimization for Robust Direction Sensing in Movable Antenna Systems

**arXiv ID:** 2603.10426 | [PDF](https://arxiv.org/pdf/2603.10426v1)

**作者:** Wenyan Ma `[一作]` (National University of Singapore), Rui Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 106117 | [OpenAlex ID](https://openalex.org/A5100422102)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种可在三维空间内连续移动的天线（MA）实现无线传感的系统，并通过对MA轨迹进行优化来提高目标方向向量的估计精度。

**💡 创新点**

创新点包括：① 引入均方角误差下界（MSAEB）作为旋转不变的性能指标并给出闭式表达；② 证明二维移动会在终射方向出现性能发散，而三维移动可实现全角度等效感知；③ 提出在给定角度区域内最小化最坏情况MSAEB的极小极大优化问题；④ 采用成功逼近（SCA）算法求解非凸轨迹优化，并给出单目标方向的简化算法；⑤ 通过理论与仿真验证三维MA显著降低极大MSAE。

**🔧 技术方法**

核心技术包括：Cramer–Rao下界分析、均方角误差下界（MSAEB）推导、轨迹协方差矩阵设计、旋转不变性分析、最小化极大目标函数的SCA求解、离散化角域网格、轨迹与速度约束处理。

**📊 数据集**

使用仿真数据：设定采样周期10 µs、波长0.05 m、MA最大速度10 m/s、移动区域为立方体（尺寸从8λ到15λ不等），不同SNR下的MSAE曲线以及不同角度区域（单方向或连续[0°,80°]×[0°,360°]）的评估。未使用公开数据集。

**📈 对比分析**

对比方法包括：① 固定位置天线阵列（16个天线）——均匀平面阵列（UPA）和互素平面阵列（CPA）；② MA 仅二维平面运动的三种轨迹——均匀网格（UPG）、圆形轨迹、圆形3D轨迹；③ 论文提出的三维MA最优轨迹。通过MSAE vs SNR、最大MSAE vs SNR以及MSAE vs 俯仰角等指标，实验表明三维MA在高SNR下MSE逼近理论下界，且相较于所有基准方案在最坏情况MSAE上降低约50%~80%，实现了更均匀、精确的方向估计。

**⚠️ 局限性**

局限性包括：① 只在仿真环境下验证，缺乏真实硬件实验；② 轨迹优化依赖角域离散化，可能导致局部最优；③ 需要满足三维移动区域可实现的速度与空间约束，实际部署时可能受限；④ 计算复杂度随采样数N和网格数Q显著增长；⑤ 仅考虑单目标方向或给定连续区域，未讨论多目标协同感知的情况。

---

## 280. LCAMV: High-Accuracy 3D Reconstruction of Color-Varying Objects Using LCA Correction and Minimum-Variance Fusion in Structured Light

**arXiv ID:** 2603.10456 | [PDF](https://arxiv.org/pdf/2603.10456v1)

**作者:** Wonbeen Oh `[一作]` (Yonsei University), Jae-Sang Hyun `[通讯]` (Yonsei University)

**通讯引用:** 29803 | [OpenAlex ID](https://openalex.org/A5080001926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于相机-投影仪单对的结构光3D重建方法LCAMV，能够在不增加硬件的前提下，针对投影仪和相机的横向色差（LCA）进行校正，并利用多通道最小方差融合提高重建精度。

**💡 创新点**

创新点在于：①使用光学模型对投影仪和相机的LCA进行像素级校正；②采用泊松-高斯噪声模型估计各通道相位不确定性；③设计最小方差加权融合并加入置信区间滤波去除异常通道；④整个流程实现无硬件扩展，适用于非均匀彩色物体。

**🔧 技术方法**

主要技术包括：结构光相位移与相位展开、相机-投影仪校准、LCA光学建模与像素级补偿、噪声参数标定、最小方差无偏估计（MVU）融合、蒙特卡罗置信区间估计、离散化的投影像素融合。

**📊 数据集**

实验数据集涵盖：彩色棋盘板（6×8随机色块）、白色球体、雕塑、RGB条纹纸张、曲面和锯齿面等多种实物，所有实验均使用Grasshopper3相机和TI DLP LightCrafter 4500进行采集。

**📈 对比分析**

与基线（均值、Y'UV、单绿通道）及两种消融方法（仅LCA校正、仅MV融合）对比，LCAMV在平面拟合均方误差上平均降低43.6%，并在非平面彩色物体上实现更平滑、无块效应的深度重建。

**⚠️ 局限性**

主要局限为：算法流程较为复杂，计算量大于传统相位移+三角测量方法，缺乏实时性能；此外，需精确的光学和噪声标定，对环境和设备的依赖较高。

---

## 281. Learning Bimanual Cloth Manipulation with Vision-based Tactile Sensing via Single Robotic Arm

**arXiv ID:** 2603.10609 | [PDF](https://arxiv.org/pdf/2603.10609v1)

**作者:** Dongmyoung Lee `[一作]` (Imperial College London), Petar Kormushev `[通讯]` (Imperial College London)

**通讯引用:** 2702 | [OpenAlex ID](https://openalex.org/A5036281885)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种单臂双手式布料操纵系统 Touch G.O.G.，通过可视触觉抓手实现自适应抓握、边缘滑动和闭环控制，仅依赖局部触觉感知完成布料展开。

**💡 创新点**

创新点包括：① 结合分离宽度控制抓手（D-WCG）与可变摩擦触觉抓手（T-VFG）实现大跨度、可旋转的抓握；② 将 Segment Anything Model (SAM) 作为视觉变压器骨干，构建 PC-Net 进行布料部件分类与 PE-Net 进行边缘姿态估计；③ 设计 SAM‑backboned encoder‑decoder 生成器 SD‑Net，用极少标注的真实触觉数据合成 3 万张高保真触觉图像，缓解数据稀缺问题。

**🔧 技术方法**

核心技术包括：视觉基触觉传感（DIGIT）、单臂 PID 控制、基于 SAM 的视觉变压器、合成数据生成网络 SD‑Net、边缘姿态回归网络 PE‑Net，以及基于触觉反馈的离散 PD 位置控制。

**📊 数据集**

使用了本实验室收集的 10k 条真实触觉序列（约 100 次抓取每类），以及 SD‑Net 生成的 30k 条合成触觉图像；评估时还使用了公开的织物数据集（如 TF1‑TF4、PT1、LB1、PT2 等七种不同织物）。

**📈 对比分析**

与 ResNet50、DenseNet121、ViT 以及传统边缘检测算法对比：PC‑Net 在四类分类上实现 97–99% 的准确率，PE‑Net 在边缘定位上平均距离误差 0.59 mm、角度误差 4.52°，明显优于基线（最差 3.38 mm/50.22°）。在真实世界滑动实验中，系统在平铺布料下 24/35 次成功，在皱折布料下 20/35 次成功，显示出较高的鲁棒性。

**⚠️ 局限性**

局限性包括：① 仅针对单臂实现双手式抓取，难以完成更复杂的多步任务；② 依赖 DIGIT 触觉传感器和专用机械结构，迁移到其他机器人平台需进一步适配；③ 仍需人工标注初始数据，虽然 SD‑Net 大幅缩减，但对极端织物纹理或厚度变化的泛化尚待验证；④ 仅聚焦边缘滑动，缺少对完整折叠、穿衣等后续操作的完整闭环控制。

---

## 282. Social Knowledge for Cross-Domain User Preference Modeling

**arXiv ID:** 2603.10148 | [PDF](https://arxiv.org/pdf/2603.10148v1)

**作者:** Nir Lotan `[一作]` (University of Haifa), Einat Minkov `[通讯]` (University of Haifa)

**通讯引用:** 1789 | [OpenAlex ID](https://openalex.org/A5057700630)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大规模社交网络（Twitter）中的热门账号嵌入，以用户关注的实体构建社交嵌入空间，从而在跨领域（音乐、新闻、政治等）预测用户偏好；

**💡 创新点**

提出将用户映射到预训练实体嵌入空间的诱导式社交用户建模，能够在冷启动或无目标域反馈时实现个性化推荐，并验证其在LLM中的可行性；

**🔧 技术方法**

采用Word2Vec变体学习实体嵌入、平均嵌入生成用户向量、余弦相似度评估、基于社交信息的特征工程以及GPT‑4o提示生成；

**📊 数据集**

使用约1.4万Twitter用户与14个领域（共280个热门实体）的人工挑选数据集，包含用户关注的实体；

**📈 对比分析**

与基于受欢迎度的无个性化基线以及闭域/全域实验对比，MAP平均提升约22%（单域）/12%（闭域），LLM在仅提供12-50个实体时提升13-23%；

**⚠️ 局限性**

局限性包括对热门账号的依赖、潜在的社会偏见与刻板印象、对冷启动的依赖仍需更细粒度实体、以及对多语言或本土化领域的适用性尚待验证。

---

## 283. Law Proofing the Future

**arXiv ID:** 2603.10021 | [PDF](https://arxiv.org/pdf/2603.10021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 284. FG-CLTP: Fine-Grained Contrastive Language Tactile Pretraining for Robotic Manipulation

**arXiv ID:** 2603.10871 | [PDF](https://arxiv.org/pdf/2603.10871v1)

**作者:** Wenxuan Ma `[一作]` (Institute of Automation, Chinese Academy of Sciences), Shuo Wang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FG‑CLTP框架，将3D触觉点云与定量语言对齐，并基于此构建3D‑TLA策略实现多模态感知与控制

**💡 创新点**

引入离散数值token化实现物理量与语言的量化语义对齐，并加入辅助回归损失，突破传统定性表征的局限

**🔧 技术方法**

利用3D触觉点云编码、CLIP对比学习、数值token化、回归监督、流匹配策略以及Gemma‑2B等Transformer技术

**📊 数据集**

构建Contact3D 100k触觉‑语言对数据集，并在GelStereo 2.0、DM‑Tac等传感器上进行跨域验证

**📈 对比分析**

相较于TVL、UniTouch、AnyTouch、CLTP等基线，FG‑CLTP在分类准确率达95.9%，回归MAE下降52.6%，sim‑to‑real误差仅3.5%，并在三项真实任务中成功率分别为85%、75%、60%

**⚠️ 局限性**

仍受限于传感器硬件差异导致的少量误差，对高速动态变化建模尚不充分

---

## 285. Transposition is Nearly Optimal for IID List Update

**arXiv ID:** 2603.10244 | [PDF](https://arxiv.org/pdf/2603.10244v1)

**作者:** Christian Coester `[一作]` (University of Oxford), Christian Coester `[通讯]` (University of Oxford)

**通讯引用:** 109 | [OpenAlex ID](https://openalex.org/A5011187509)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了在i.i.d.请求模型下，Transposition（仅交换请求项与其前驱）自组织列表的长期平均访问成本不超过最优成本加1，几乎达到最优；

**💡 创新点**

创新点在于构造了一种完全无记忆的“sign‑eliminating”组合注入，证明相关多项式系数非负，从而得到加1上界；

**🔧 技术方法**

主要技术包括：使用停机分布公式、将误差拆解为逆序和，再将每项用“gap”变量重新表达成多项式；然后通过组合注入法将负系数映射到正系数；

**📊 数据集**

论文并未使用实验数据集，而是进行严格的理论分析；

**📈 对比分析**

对比方法：与已知的最优静态排序（需事先知道请求分布）以及Move‑to‑Front规则进行比较，证明Transposition在长期期望成本上比Move‑to‑Front好，且只比最优多1；

**⚠️ 局限性**

局限性：结果仅适用于长期稳态；在收敛到稳态前的短期性能无法给出；此外，证明仅适用于i.i.d.请求分布，无法直接推广到自适应或对抗性模型。

---

## 286. Learning to Decode Quantum LDPC Codes Via Belief Propagation

**arXiv ID:** 2603.10192 | [PDF](https://arxiv.org/pdf/2603.10192v1)

**作者:** Mohsen Moradi `[一作]` (New Mexico State University), David G. M. Mitchell `[通讯]` (New Mexico State University)

**通讯引用:** 5833 | [OpenAlex ID](https://openalex.org/A5076523933)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于强化学习的顺序贝尔曼传播（RL‑SVNS）调度，用于解码量子低密度校验码（QLDPC）并显著提升收敛速度和误码率。

**💡 创新点**

创新点在于把变量节点（VN）更新顺序建模为马尔可夫决策过程，利用局部syndrome驱动的状态表示，离线学习一个Q表；并通过增量维护二阶邻域状态和堆优先级实现低复杂度推理。

**🔧 技术方法**

核心技术包括：Q‑learning、SVNS BP、局部状态编码与增量更新、堆实现的贪心调度、二流四元BP（针对抛弃化通道）以及与 guided decimation 的混合。

**📊 数据集**

使用了多种典型QLDPC码作为实验数据集：[[882,24,18≤d≤24]] B1、[[882,48,16]] B2、[[180,10,15≤d≤18]] A5、[[144,12,12]] BB 以及更长的 [[288,12,18]] BB。

**📈 对比分析**

与洪泛BP、随机顺序、BP‑OSD、BPGD 等传统方法比较，RL‑SVNS 在FER 上可降低 1–2 个数量级，收敛迭代次数显著减少；在 QBPGD 的混合版本（RL‑QSVNS‑GD）中，需的 decimation 步数大幅降低，性能进一步提升。

**⚠️ 局限性**

局限性包括：Q表状态空间随 VN 度增长而膨胀，需离线训练；适用于相对稀疏的 QLDPC，极大码长的可扩展性尚待验证；当前实验仅覆盖独立 X 错误与抛弃化通道，未验证对更通用噪声模型的适用性。

---

## 287. A Robust Deep Learning Framework for Bangla License Plate Recognition Using YOLO and Vision-Language OCR

**arXiv ID:** 2603.10267 | [PDF](https://arxiv.org/pdf/2603.10267v1)

**作者:** Nayeb Hasin `[一作]` (Islamic University of Technology), Asif Newaz `[通讯]` (Islamic University of Technology)

**通讯引用:** 157 | [OpenAlex ID](https://openalex.org/A5006761977)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一套针对孟加拉语车牌的端到端识别系统，包括车牌定位和字符识别。

**💡 创新点**

引入双阶段自适应训练策略和多阶段冻结提升YOLOv8的定位性能，并采用VisionEncoderDecoder结合BanglaBERT实现字符级高精度识别。

**🔧 技术方法**

YOLOv8m、Vision Transformer、BanglaBERT、Mixup/CopyPaste数据增强、混合精度训练、Beam Search解码等技术。

**📊 数据集**

使用Bangladeshi License Plate Recognition Dataset、Bangla License Plate Dataset（含合成图像）以及自制低光环境外部验证集，共约20k张图片。

**📈 对比分析**

与U‑Net、YOLOv5m/7m/9m/11m等模型对比，YOLOv8m+多阶段学习在定位准确率97.83%、IoU 91.3%；OCR方面ViT+BanglaBERT在CER 0.1323、WER 0.1068，性能显著优于其他模型。

**⚠️ 局限性**

训练集主要为白天图像，缺乏低光与倾斜视角数据；仅使用轴对齐框标注，限制了分割模型优势。

---

## 288. A Hybrid Knowledge-Grounded Framework for Safety and Traceability in Prescription Verification

**arXiv ID:** 2603.10891 | [PDF](https://arxiv.org/pdf/2603.10891v1)

**作者:** Yichi Zhu `[一作]` (East China University of Science and Technology), Guisheng Fan `[通讯]` (East China University of Science and Technology)

**通讯引用:** 1821 | [OpenAlex ID](https://openalex.org/A5034901082)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并利用混合药物知识库（HPKB）进行药方审计，避免直接用LLM生成错误建议。

**💡 创新点**

创新点在于：①引入Hybrid VKG模型，将约束满足与拓扑推理并存；②提出Iterative Schema Refinement (ISR) 迭代式模式实现可信的知识结构；③设计KB‑grounded Chain of Verification (CoV)，将审计拆解为可验证子任务并通过SQL/Cypher查询获取可追溯证据。

**🔧 技术方法**

采用多代理 Section‑Aware 处理、ISR 迭代模式、SQL + Cypher 双查询、P‑EST 证据筛选、统一映射层，以及LLM（如GPT‑4o、Deepseek‑V3、Qwen3‑32B）进行事实抽取与合成。

**📊 数据集**

使用 100 篇药典文档（用于构建基准知识库）和 100 份真实住院处方（用于评测），以及通过 Gemini 2.5 Pro 生成的 1,000+ 规则化错误处方数据集。

**📈 对比分析**

与 Zero‑shot OpenIE、One‑shot AutoKG 两大基线对比，整体 F1 超过 0.83；与传统 CDSS 及人类经验对照，提出方法 F1 0.72、Recall 70%、Precision 74%，显著提升审计准确率并降低警报疲劳。

**⚠️ 局限性**

局限性包括：对临床情境（如指示类误报）认知不足；缺乏完整医院流程知识；对极端稀有药物覆盖有限；推理成本与速度仍需进一步平衡。

---

## 289. Mashup Learning: Faster Finetuning by Remixing Past Checkpoints

**arXiv ID:** 2603.10156 | [PDF](https://arxiv.org/pdf/2603.10156v1)

**作者:** Sofia Maria Lo Cicero Vaina `[一作]` (Together AI), Max Ryabinin `[通讯]` (Together AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Mashup Learning方法，利用过去的finetune checkpoint来构造更好的初始化，从而加速并提升LLM在新任务上的fine‑tune表现。

**💡 创新点**

创新点在于把历史checkpoint的融合（通过loss/accuracy排序+平均或高级融合）作为训练前初始化，而非传统的从零开始或单一checkpoint继续训练；并证明此做法在多模型、多任务上可持续提升。

**🔧 技术方法**

核心技术包括：①基于目标任务训练集小样本评估的checkpoint相关性排序；②多checkpoint聚合（平均、DARE‑TIES等模型融合）；③在LoRA与全参数finetune两种训练方式下直接使用融合结果作为初始化；④学习率对比与收敛加速分析。

**📊 数据集**

使用八个常见的NLP多选基准数据集：ARC‑Easy、CommonsenseQA、HellaSwag、MathQA、OpenBookQA、PIQA、SocialIQA、WinoGrande；以及Gemma‑3 1B/2B/4B和Mistral‑7B‑Instruct‑v0.2等模型。

**📈 对比分析**

与从零开始和Text‑to‑LoRA等基线比较；在LoRA和全参数finetune上，Mashup Learning平均提升0.5–5个百分点准确率，收敛步数减少约41–46%，墙钟时间减少最高37%（含选取与融合开销）。

**⚠️ 局限性**

局限性包括：①需要对所有历史checkpoint做一次小样本评估，若checkpoint数量极大会产生额外计算成本；②对LoRA的融合需访问原始适配器参数，非公开资源时受限；③在极低数据或极大模型规模下，选取样本量与融合效果的最佳平衡尚需进一步研究。

---

## 290. Measuring and Eliminating Refusals in Military Large Language Models

**arXiv ID:** 2603.10012 | [PDF](https://arxiv.org/pdf/2603.10012v1)

**作者:** Jack FitzGerald `[一作]` (EdgeRunner AI), Tyler Saltsman `[通讯]` (EdgeRunner AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并公开了针对军事领域LLM的拒绝率基准（gold 和两个 bronze 数据集），对 31 公共模型和 3 军事模型进行评测，并尝试使用方向性削减（abliteration）降低拒绝率。

**💡 创新点**

首次系统化测量军事查询的拒绝/偏移行为；发布 gold 基准并证明 bronze 数据集与 gold 的高相关性；将 abliteration 作为减轻拒绝的可行手段进行量化分析。

**🔧 技术方法**

利用拒绝标记 + 判定模型（Selene 1、Gemma 3、gpt-oss-120b）进行自动分类；使用 Heretic 库实现方向性削减；通过 Pearson 相关性、K-L 散度等统计方法评估模型表现。

**📊 数据集**

mil-deflect-gold-alpha（221 条）、mil-deflect-bronze-alpha（1,047 条）、mil-deflect-bronze-bravo（1,500 条）三套军事拒绝基准；31 公共模型与 3 军事模型；通用基准 ARC、GPQA Diamond、GSM8k、IFEval、MMLU Pro、TruthfulQA；synthetic 训练集用于 abliteration。

**📈 对比分析**

采用回答率、拒绝率、偏移率、无效率等指标与基准数据对比；结果显示公开模型拒绝率可高达 98.2%，军事模型在 abliteration 后拒绝率下降 66.5 点，但平均任务性能下降约 2%，若追求 90%+ 答复率需接受 10%–30% 的性能回归。

**⚠️ 局限性**

限制：gold 数据集规模有限；synthetic 数据集真实性不足；abliteration 会导致任务性能下降，尤其是高拒绝率下回归显著；仍缺乏对真实军用环境的全面验证；安全对齐与军用功能平衡难以兼顾。

---

## 291. DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime

**arXiv ID:** 2603.10538 | [PDF](https://arxiv.org/pdf/2603.10538v1)

**作者:** Julian Lorenz `[一作]` (University of Augsburg), Rainer Lienhart `[通讯]` (University of Augsburg)

**通讯引用:** 9627 | [OpenAlex ID](https://openalex.org/A5009744749)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种低延迟的全景场景图生成模型 DSFlash，能够在 RTX 3090 GPU 上每秒处理 56 帧，且在保持或超过现有最优方法的 mR@50 性能的同时，显著降低推理时间。

**💡 创新点**

创新点包括：① 双向关系预测器通过单次前向传递同时输出正向和反向关系；② 基于掩码的动态补丁剪枝与 ToMe 令牌合并，减少 Transformer 计算；③ 将 DSFormer 的分离式分割与关系预测融合，使用 EoMT 轻量分割骨干网；④ 低分辨率掩码提升速度、保持准确率；⑤ 一致性损失和门控机制平衡双向预测。

**🔧 技术方法**

技术方案主要采用 Transformer 结构（ViT‑EoMT）、掩码嵌入、门控多路输出、ToMe‑SD 令牌合并、轻量化分割头、数据增强（DeiT‑III），以及基于 mR@50 的评估和帧率（RPS）测评。

**📊 数据集**

使用主流的 PSG 数据集（49k 图像，56 种谓词）进行训练与评估，并在 Visual Genome 上使用 PredCls 协议做补充实验。

**📈 对比分析**

与 MotifNet、VCTree、HiLo、REACT、DSFormer 等方法对比，DSFlash‑S（40M 参数）在 mR@50 上达到 25.05，DSFlash‑L（340M 参数）在 30.90，均优于 DSFormer（30.70）且延迟仅 18–50 ms，帧率达 56 fps；在 GPU 级别不同的实验中，加入剪枝与令牌合并后 1080 GPU 上延迟从 230 ms 降至 173 ms。

**⚠️ 局限性**

局限性：① 对分割质量高度依赖，若分割误差大会导致 mR@inf 下降；② 低分辨率掩码在某些精细关系上可能失效；③ 双向预测的训练仍需额外一致性损失，模型复杂度略升；④ 在极端资源受限的设备（如嵌入式系统）上仍可能超出算力；⑤ 目前仅在 PSG 数据集上验证，跨域泛化待进一步评估。

---

## 292. Machinagogy: Experiments in Staging Teaching Dramas with LLMs

**arXiv ID:** 2603.10450 | [PDF](https://arxiv.org/pdf/2603.10450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 293. BinWalker: Development and Field Evaluation of a Quadruped Manipulator Platform for Sustainable Litter Collection

**arXiv ID:** 2603.10529 | [PDF](https://arxiv.org/pdf/2603.10529v1)

**作者:** Giulio Turrisi `[一作]` (Italian Institute of Technology), Claudio Semini `[通讯]` (Italian Institute of Technology)

**通讯引用:** 5162 | [OpenAlex ID](https://openalex.org/A5010033061)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一款四足机器人平台 BinWalker，集成了行走、视觉感知、机械臂操作和自带垃圾箱，实现了在不规则户外环境中自主收集手尺寸垃圾的完整闭环。

**💡 创新点**

创新点包括：① 将四足高适应性行走与可扩展机械臂结合，实现了在复杂地形上既能移动又能抓取的多功能平台；② 采用层级控制结构，将行走控制（RL）与抓取控制（预定义动作+IK）分离，提高了系统鲁棒性与可调性；③ 设计了自驱动卸载机构，机器人可在无人工干预下完成垃圾箱卸载。

**🔧 技术方法**

技术手段：四足运动控制基于深度强化学习（RL）实现自适应步态；机械臂控制采用预设动作与 Mink（基于MuJoCo的IK求解器）实现关节空间与笛卡尔空间的协调；视觉感知使用 YOLOv8 目标分割网络 + PCA+深度重投影完成瓶子 3D 位姿估计；ROS/Isaac‑Lab 生态实现软硬件耦合；实验部署在 Unitree Aliengo + Z1 机械臂 + Realsense D435 视觉模块。

**📊 数据集**

数据集：YOLO 部分使用公开的 COCO/Trash‑V1 等瓶子/垃圾检测数据进行训练，随后在实验室与户外环境中进行迁移微调；强化学习行走策略使用 Gazebo/Isaac‑Lab 内部模拟场景进行训练，随后在真实地形上测试。

**📈 对比分析**

对比方法：将 RL 行走 + 传统基于模型的 IK 控制组合与纯模型基于 IK 的传统方案进行对比；在户外野餐区场景下，记录成功抓取率、卸载成功率和平均抓取时间。实验结果显示，BinWalker 在多种地形下成功率达到 80% 以上，平均抓取时间约 5.2 s，显著优于单纯模型基于 IK 的方案（成功率约 65%，抓取时间 7.1 s）。

**⚠️ 局限性**

局限性：① 对碎裂或被掩埋的垃圾识别与抓取效果不佳；② 视觉感知在强光、雨雪等恶劣环境下鲁棒性不足；③ 机械臂与垃圾箱的重量/尺寸限制，难以处理大件或硬质垃圾；④ 目前不具备全局导航功能，依赖人工遥控或预先规划路径；⑤ 强化学习行走策略对大规模多样化地形的泛化能力仍有待提升。

---

## 294. FAR-Dex: Few-shot Data Augmentation and Adaptive Residual Policy Refinement for Dexterous Manipulation

**arXiv ID:** 2603.10451 | [PDF](https://arxiv.org/pdf/2603.10451v1)

**作者:** Yushan Bai `[一作]` (Institute of Automation, Chinese Academy of Sciences), Zhengtao Zhang `[通讯]` (Beijing Zhongke Huiling Robot Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过结合少量真实演示，先在IsaacLab中进行轨迹分段与三维重组实现大规模物理约束数据增强，再利用自适应残差模块对基准策略进行在线强化学习，完成了人类级多指手臂协作的细粒度抓取与操作。

**💡 创新点**

创新点在于：① 轨迹分段与姿态扰动相结合的生成方式，实现了在有限演示下生成物理可行且细节丰富的手‑物体交互数据；② 自适应残差学习通过跨步注意力加权，动态调节手臂与手指动作，显著提升了长周期任务中的精度与鲁棒性；③ 在数据增强与策略优化两阶段采用一致性模型压缩，既保持高质量生成，又大幅降低推理时延。

**🔧 技术方法**

技术包括：IsaacLab仿真、轨迹分段（motion/skill）、三维姿态扰动与重组、基于点云的前向/逆向运动学、基于一致性模型的单步动作预测、跨步注意力的残差权重学习、PPO在线强化。

**📊 数据集**

使用少量人工演示（每个任务2条20Hz演示）作为基础，并在此基础上合成大规模合成数据集 D_g；训练时将 D_h 与 D_g 结合构成最终训练集。

**📈 对比分析**

与 MimicGen、DemoGen 的数据生成相比，生成质量提高约 20%；与 ACT+3D、DP3、ResiP 等基线相比，任务成功率平均提升 7%（最优任务 95%），单步推理时间约 3–4 ms；在真实机器人上，成功率均超过 80%，比 ResiP 提升 10%。

**⚠️ 局限性**

局限性在于：① 生成与训练的仿真成本高；② 仍依赖域随机化，跨模仿误差难以完全消除；③ 缺少力/触觉反馈，导致在极端抓取或细粒度操作时的鲁棒性受限。

---

## 295. S-HPLB: Efficient LLM Attention Serving via Sparsity-Aware Head Parallelism Load Balance

**arXiv ID:** 2603.10353 | [PDF](https://arxiv.org/pdf/2603.10353v1)

**作者:** Di Liu `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14458 | [OpenAlex ID](https://openalex.org/A5039318240)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大规模语言模型（LLM）推理中的注意力计算瓶颈，提出了 S‑HPLB（Sparsity‑Aware Head‑Parallel Load Balance）框架，实现了头自适应稀疏预算分配与多 GPU 负载均衡，从而在保持推理质量的同时显著提升注意力运算速度。

**💡 创新点**

创新点包括：
① 发现并利用注意力头的稀疏性在不同输入与任务中高度稳定，却呈现显著异质性；
② 通过离线稀疏性建模与 max‑min 预算转移算法，实现每个头的最优预算分配；
③ 将头到设备的分配建模为多路分区问题，采用贪心启发式实现跨 GPU 的负载平衡；
④ 在现有稀疏注意力框架（MInference）基础上集成上述两大算法，形成端到端高效推理系统。

**🔧 技术方法**

技术细节包括：
- 稀疏注意力（top‑k / top‑p）
- 头并行（Head‑Parallel）与注意力‑FFN 分离（AFD）
- 离线稀疏性 profiling 与 max‑min 预算转移
- 多路分区负载平衡（贪心分配）
- FlashAttention、MInference 等现有实现作为基准。

**📊 数据集**

实验使用的模型与数据集：
- Llama‑3.1‑8B、Qwen2.5‑7B、Qwen2.5‑72B、Llama‑3‑8B‑262K；
- RULER 长上下文基准（包含 13 项任务，最大 128K 上下文）；
- PG‑19 等校准集用于稀疏性离线 profiling；
- 128K 上下文长度进行 latency 评估。

**📈 对比分析**

与基线比较：
- Full attention（FlashAttention）
- Top‑k 稀疏方法：StreamingLLM、MInference
- Top‑p 方法：XAttention

S‑HPLB 在 RULER 上的表现：
- 准确率仅下降 0.52%/1.37%/3.13%（相较全注意力）并在部分任务甚至超越全注意力；
- 与 XAttention 相比，准确率提升 2.57%/2.94%/0.61%；
- 在 128K 上下文下，注意力延迟比全注意力低 3.39×/4.27×/3.31×；
- 与 XAttention 相比，延迟下降 2.09×/2.22×/2.88×；
- 加入 Head‑Parallel Load Balance 后，进一步减少 1.19×/1.26× 的延迟。

**⚠️ 局限性**

限制与不足：
- 需要离线稀疏性 profiling，增加额外准备成本；
- 依赖 GPU 之间高速互联（NVLink），在单 GPU 或网络瓶颈环境下效果有限；
- 主要针对预填充（prefill）阶段，解码阶段效率提升未覆盖；
- 预算分配与负载平衡基于预估模型，动态负载或非均匀工作负载下可能需进一步调优；
- 对极大模型（如 72B 以上）或不同注意力机制（如 GQA）的通用性待进一步验证。

---

## 296. Why Does It Look There? Structured Explanations for Image Classification

**arXiv ID:** 2603.10234 | [PDF](https://arxiv.org/pdf/2603.10234v1)

**作者:** Jiarui Li `[一作]` (Tulane University), Ramgopal R. Mettu `[通讯]` (Tulane University)

**通讯引用:** 994 | [OpenAlex ID](https://openalex.org/A5063192197)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Interpretability to Explainability（I2X）框架，利用训练过程中对齐的原型强度和模型置信度变化来构建结构化解释，揭示模型如何逐步学习并推断；

**💡 创新点**

创新点在于将无结构的可解释性（如Grad-CAM）转化为结构化的解释，直接从模型自身的训练轨迹中提取原型责任关系，避免了对辅助模型的依赖，并能指导后续微调以降低混淆；

**🔧 技术方法**

核心技术包括Grad-CAM产生的注意力图、PCA+K-Means聚类得到抽象原型、HDBSCAN聚类置信度变化、岭回归建立原型强度与置信度变化的映射，以及结构化可视化和实验验证；

**📊 数据集**

在MNIST、CIFAR-10（ResNet-50）和MNIST（InceptionV3）等公开图像分类数据集上进行实验；

**📈 对比分析**

与传统无结构XAI方法对比，I2X通过分析原型贡献显著降低了同类间混淆（MNIST上7与2混淆下降约5/23个样本），并在细调时实现了准确率提升至98.64%，CIFAR-10上猫狗混淆由261.20下降至238.60；

**⚠️ 局限性**

局限性包括：依赖于后验解释方法和聚类，可能受聚类参数影响；对训练数据顺序敏感；对不确定原型的识别依赖经验阈值，缺乏统一度量；未解决多模态或更大规模模型的可扩展性问题。

---

## 297. AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models

**arXiv ID:** 2603.10126 | [PDF](https://arxiv.org/pdf/2603.10126v1)

**作者:** Yutong Hu `[一作]` (INSAIT, Sofia University St. Kliment Ohridski), Danda Paudel `[通讯]` (INSAIT, Sofia University St. Kliment Ohridski)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个自回归动作专家 AR-VLA，能够在视觉‑语言‑动作模型中持续维护运动历史，实现高频、上下文感知的动作生成。

**💡 创新点**

通过混合键值缓存（HKV）与动态时间重新定位（DTR）实现动作专家与视觉语言模块的异步解耦，解决传统块式动作生成的马尔可夫失忆与频率不匹配问题。

**🔧 技术方法**

采用 Transformer 自回归解码器、RoPE 位置编码、Hybrid KV 缓存、动态时间重新定位、两阶段预训练（动作预训练 + 视觉语言对齐）以及随机历史遮蔽等技术。

**📊 数据集**

使用 BridgeV2、SimplerEnv 仿真、真实 WidowX 机器人数据、PushT、ALOHA 任务以及长周期任务 PushT2、Stack3 等数据集进行训练与评估。

**📈 对比分析**

与 OpenVLA、Pi-0-Fast、Pi-0-5、CogACT、ACT、Diffusion Policy 等基线对比，AR-VLA 在通用与专用任务中均达到或超过最高成功率，轨迹平滑度、抖动显著降低，推理延迟也优于块式方法，长周期任务成功率明显提升。

**⚠️ 局限性**

模型对极端高延迟或非固定帧率环境的适应性有限；需要在高频控制与视觉延迟之间进行同步假设；动作历史遮蔽的比例需要经验调优，且目前验证仅覆盖单一机械臂和有限任务。

---

## 298. Making Bielik LLM Reason (Better): A Field Report

**arXiv ID:** 2603.10640 | [PDF](https://arxiv.org/pdf/2603.10640v1)

**作者:** Adam Trybus `[一作]` (Institute of Philosophy Jagiellonian University), Remigiusz Kinas `[通讯]` (Bielik.ai Speakleash Foundation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并提升波兰大型语言模型Bielik的推理能力，构建评测方法、训练流程并进行系统基准对比

**💡 创新点**

首次将推理能力集成到Bielik并创建多层次推理评测体系，同时设计多模态强化学习与可验证奖励机制

**🔧 技术方法**

使用监督微调（SFT）、直接偏好优化（DPO）、GRPO/DAPO强化学习、VERL平台、Chain‑of‑Thought标注与推理模板

**📊 数据集**

1.3M条英文学术推理轨迹、143k条可验证波兰推理任务、490k波兰数学/代码/科学任务、1.3M推理跟踪、AIME/AMC/MATH‑500等翻译集

**📈 对比分析**

在111道自定义推理题上与30个领先LLM对标，Bielik‑R取得56%得分，最高gemini‑3‑pro‑preview为87%；在正式逻辑题上Bielik‑R达89%一阶逻辑、80%命题推理

**⚠️ 局限性**

表现仍落后主流模型；存在幻觉、假设维持困难、token超限导致误差、推理过程缺乏自适应修正以及模型记忆与推理边界模糊

---

## 299. Chasing RATs: Tracing Reading for and as Creative Activity

**arXiv ID:** 2603.11031 | [PDF](https://arxiv.org/pdf/2603.11031v1)

**作者:** Sophia Liu `[一作]` (University of California), Shm Garanganao Almeda `[通讯]` (University of California)

**通讯引用:** 43 | [OpenAlex ID](https://openalex.org/A5092432012)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并示例化阅读活动轨迹（RATs）框架，设计 WikiRAT 浏览器插件记录并可视化 Wikipedia 阅读路径，包括显式链接路径、语义相似链接和反思连接三种图谱。

**💡 创新点**

将阅读视为创意活动并通过可追踪轨迹呈现，融合三种图谱层级，为创意支持工具提供过程层面的可视化与分析素材，突破传统仅关注产出物的限制。

**🔧 技术方法**

使用浏览器扩展技术抓取点击与停留时间，利用词嵌入或其他语义相似度方法生成 fuzzy linkograph，并用图谱可视化库呈现三种图谱。

**📊 数据集**

依赖 Wikipedia 的公开链接结构和页面内容数据集，示例实验未正式实施，假设使用公开的 Wikipedia 数据与用户点击日志。

**📈 对比分析**

目前为概念性论文，尚无对比实验；未来计划将人类阅读轨迹与 AI 代理阅读轨迹进行比较，评估路径长度、链接密度、停留时间等指标，预期可揭示自动化对阅读过程的压缩与失真。

**⚠️ 局限性**

局限性包括：仍处于设计与概念验证阶段，缺乏大规模实证；数据采集面临隐私与用户授权挑战；扩展到非 Wikipedia 平台的可行性与跨平台兼容性待验证；对用户体验与长期使用效果的评估不足。

---

## 300. Task-Aware Delegation Cues for LLM Agents

**arXiv ID:** 2603.11011 | [PDF](https://arxiv.org/pdf/2603.11011v1)

**作者:** Xingrui Gu `[一作]` (University of California), Xingrui Gu `[通讯]` (University of California)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5104269538)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证一种任务感知的协作信号层，将离线的偏好评估转化为实时、可视化的能力轮廓和协调风险提示，并在此基础上实现闭环委派、显式推理说明和可审计日志。

**💡 创新点**

创新点在于：①利用语义聚类构建任务类型，并基于人类对比数据生成任务条件的胜率图谱与争议率（Tie率）作为风险指示；②在此基础上设计阈值触发的高保证模式，实现可见、可协商、可审计的代理委派；③将能力与风险信息统一为用户面向的委派决策流程。

**🔧 技术方法**

技术手段包括：语义嵌入+降维（UMAP）、K‑means聚类、任务条件胜率估计、Tie率作为不确定性指标、多项式逻辑回归与岭回归进行验证、以及闭环委派算法与日志记录。

**📊 数据集**

使用的数据集为 Chatbot Arena 单轮提示对比数据集，包含提示、两模型的选择结果以及人类偏好标签。

**📈 对比分析**

通过5折交叉验证评估：任务A（胜者预测）准确率显著提升，任务B（难度预测）均方误差下降；移除任务类型特征后两项指标均恶化，验证了任务条件信号的预测价值。

**⚠️ 局限性**

局限性包括：语义聚类可能放大训练语料中的偏见；高胜率可能导致用户过度信任；日志记录在高敏感度任务中可能泄露用户信息，需要最小化数据保留和加入噪声以保障隐私。

---

## 301. Fuel Gauge: Estimating Chain-of-Thought Length Ahead of Time in Large Multimodal Models

**arXiv ID:** 2603.10335 | [PDF](https://arxiv.org/pdf/2603.10335v1)

**作者:** Yuedong Yang `[一作]` (University of Texas at Austin), Radu Marculescu `[通讯]` (University of Texas at Austin)

**通讯引用:** 12979 | [OpenAlex ID](https://openalex.org/A5036227385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Fuel Gauge，能够在推理前预测多模态 LMM 的 Chain-of-Thought（CoT）长度，并利用该预测实现 KV 缓存预分配和 CoT 长度调节。

**💡 创新点**

创新点在于：①发现 CoT 长度可视为 Bernoulli 过程且可预估；②假设并挖掘模型内部“燃料水平”信号，利用该信号预测 CoT 长度；③将预测结果用于两项实际任务，实现显著的内存利用率提升与推理质量控制。

**🔧 技术方法**

使用两阶段神经网络（隐藏状态提取器 f_sig 与燃料估计器 f_fuel）结合线性外推；训练时基于 200 条 CoT 轨迹；在推理时实时计算燃料水平并预测长度；对 KV 缓存采用预测长度预分配；对 CoT 长度调节采用梯度方向控制燃料水平。

**📊 数据集**

在 4 种多模态模型（Qwen3-4B/8B、Qwen3VL-2B/4B）上评估，使用 GPQA-Diamond、AIME24/25、MathVision-m、LongVideoBench-15/60 等 benchmark；训练集为 MMLU（文本）和 MMMU（文本-视觉）前 200 条问题。

**📈 对比分析**

与基线（平均/中位数、End-of-CoT 预测、直接预测网络、HF KV 分配策略）比较；Fuel Gauge 在燃料水平预测的 rMAE 低于基线一半；CoT 长度预测误差显著降低；KV 缓存分配频率下降 9.8 倍；CoT 长度调节实现线性可控，准确率随 η 线性提升，表明方法有效。

**⚠️ 局限性**

局限包括：需要在目标模型上额外训练 82k 参数的小网络；对不同模型的参数调优（窗口大小、层选择）仍需手工；在极端长/短 CoT 的泛化尚不充分；未在大规模视频/长文本场景下深入评估。

---

## 302. Dance2Hesitate: A Multi-Modal Dataset of Dancer-Taught Hesitancy for Understandable Robot Motion

**arXiv ID:** 2603.10166 | [PDF](https://arxiv.org/pdf/2603.10166v1)

**作者:** Srikrishna Bangalore Raghu `[一作]` (University of Colorado Boulder), Alessandro Roncone `[通讯]` (University of Colorado Boulder)

**通讯引用:** 702 | [OpenAlex ID](https://openalex.org/A5020277024)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集了一份多模态、舞者生成的机器人与人类犹豫动作数据集，用于研究机器人在协作中的犹豫表达；

**💡 创新点**

通过仅改变犹豫水平而保持任务目标不变，提供了跨模态、跨体现的可重复基准，并公开了标注的轻微、显著、极端三层犹豫等级；

**🔧 技术方法**

使用FrankEmika Panda本体教学、RealSense RGB‑D摄像头、OpenPose关键点提取，并以ROS、CSV、NPZ等格式发布；

**📊 数据集**

Dance2Hesitate数据集，包括66条机器人本体教学轨迹、84条人类上肢轨迹和70条全身轨迹，分为轻微、显著、极端三层犹豫；

**📈 对比分析**

该数据集为未来基于机器学习的犹豫识别与生成提供基准，论文未给出具体算法性能，但可用于监督分类、生成模型评估；

**⚠️ 局限性**

数据集仅涵盖两种体现（机器人手臂与全身）和两种上下文（固定目标、自由空间），且犹豫水平仅由舞者主观标记，缺乏多样化用户与实时交互验证。

---

## 303. Prioritizing Gradient Sign Over Modulus: An Importance-Aware Framework for Wireless Federated Learning

**arXiv ID:** 2603.10763 | [PDF](https://arxiv.org/pdf/2603.10763v1)

**作者:** Yiyang Yue `[一作]` (Southeast University), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 86118 | [OpenAlex ID](https://openalex.org/A5091266202)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为SP-FL的无线联邦学习框架，通过在梯度传输时将符号与模值分离并优先保障符号的可靠性，从而提升在资源受限无线环境下的模型训练性能。

**💡 创新点**

创新点在于：①引入符号优先传输与误差补偿机制，允许在模值丢失时仍能利用已成功接收的符号；②基于一阶收敛分析构建双层资源分配问题，将带宽按设备重要性分配、功率按符号/模值分配；③使用Newton-Raphson和SCA实现高效的交替优化。

**🔧 技术方法**

技术方法包括：符号-模值分离的梯度量化、误差补偿向量、概率成功传输建模、基于一阶收敛的目标函数推导、交替优化（Newton-Raphson求功率分配、SCA求带宽分配）以及低复杂度的罚函数梯度下降变体。

**📊 数据集**

实验使用CIFAR-10数据集，构建60K参数的CNN模型，模拟IID与非IID（Dirichlet）数据分布。

**📈 对比分析**

与四种基线（无误差、设备调度、均匀带宽分配、仅发送符号）对比，SP-FL在资源受限（功率、带宽、设备数）场景下平均提升约10%测试准确率，且收敛速度最快；在高非IID水平下可达9.96%性能提升。

**⚠️ 局限性**

局限性包括：1) 算法在设备数量极大时计算复杂度仍高，需要低复杂度近似；2) 需要额外的梯度模量信息上行，虽然量小但增加协议开销；3) 误差补偿向量选择对性能敏感，现有方案（上一次全局梯度或随机种子）并非最优；4) 实验仅在单机仿真环境下验证，真实无线网络中多路径、时变信道的影响尚待进一步研究。

---

## 304. TriageSim: A Conversational Emergency Triage Simulation Framework from Structured Electronic Health Records

**arXiv ID:** 2603.10035 | [PDF](https://arxiv.org/pdf/2603.10035v1)

**作者:** Dipankar Srirag `[一作]` (University of New South Wales), Salil Kanhere `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了 TriageSim，一个基于结构化电子健康记录（EHR）生成个性化、可控的急诊科分诊对话的仿真框架，能够输出对话文本和对应音频。

**💡 创新点**

创新点在于将分诊算法（ATS/ESI）嵌入多代理对话模拟；通过人格化参数控制病人说话的流畅度、发音口音和护士的决策风格；以及利用零样本语音克隆和背景噪声混合，生成既具临床真实性又含多种语音变异的音频数据。

**🔧 技术方法**

使用技术包括：大型语言模型（Gemini、GPT‑5、Qwen‑3‑TTS）进行对话生成与语音克隆；规则与预训练模型评估流畅度；语音合成后加入ESC‑50噪声；ASR（Whisper）生成转录；使用加权 Cohen’s κ、WER、UTMOS、F1 等指标评估语义、语音与医学真实性。

**📊 数据集**

数据集来源于 MIMIC‑IV‑ED、ESI Handbook、ETEK Manual 以及公开的教学资源，用以构建结构化病例；音频采样来自公共语音库和 ESC‑50 背景噪声集。

**📈 对比分析**

通过对比合成文本、ASR 文本和原始音频进行对话分诊分类，使用加权 Cohen’s κ 评估效果；结果显示合成文本 κ≈0.31、ASR κ≈0.33、音频 κ≈0.28–0.27，差异不大；WER 10.8，UTMOS 3.42，医学 F1 0.94，说明数据在语义与医学方面保持高质量。

**⚠️ 局限性**

局限性包括：对话分类仍受临床推理难度限制，数据规模仅 814 条对话，评估样本有限；模型对真实世界多样性和更大规模数据的泛化能力未知；生成过程依赖 LLM 与语音克隆技术，可能受模型偏差与声纹稳定性的影响。

---

## 305. There Are No Silly Questions: Evaluation of Offline LLM Capabilities from a Turkish Perspective

**arXiv ID:** 2603.09996 | [PDF](https://arxiv.org/pdf/2603.09996v1)

**作者:** Edibe Yilmaz `[一作]`, Kahraman Kostas `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了14个离线大型语言模型在土耳其遗产语言教育中的鲁棒性与教学安全性，利用基于异常的测试方法。

**💡 创新点**

提出了Turkish Anomaly Suite（TAS）异常集与多维度评分框架，专注于模型对教学风险（幻觉、从众偏差等）的抵抗能力。

**🔧 技术方法**

使用离线推理、异常检测、模型对抗测试和综合得分公式（结合准确性、延迟与参数规模）来衡量模型表现。

**📊 数据集**

构建了包含10个精心设计的边缘案例（TAS）作为评估数据集，覆盖语言干扰、事实误差、历史文化虚构与权威谬误。

**📈 对比分析**

将14个模型按参数大小、延迟与异常抵抗进行对比，最终得分显示8B–14B规模模型在安全性与成本之间达到最佳平衡，顶尖模型为zai-orgglm-4.7-flash和ministral-3-14b-reasoning。

**⚠️ 局限性**

局限性包括样本仅10个异常案例、缺乏与人类教师的对照、结果的统计泛化性有限，并未充分验证在真实教学环境中的长期效果。

---

## 306. Estimating the condition number of Chebyshev filtered vectors with application to the ChASE library

**arXiv ID:** 2603.10514 | [PDF](https://arxiv.org/pdf/2603.10514v1)

**作者:** Edoardo Di Napoli `[一作]` (Jülich Supercomputing Centre), Xinzhe Wu `[通讯]` (Jülich Supercomputing Centre)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在Chebyshev加速子空间迭代中，如何估计滤波后向量集的条件数，并基于此实现了动态选择QR分解算法的机制，以提升ChASE库的性能与可扩展性。

**💡 创新点**

创新点在于提出了一种仅依赖已知谱边界、已求得的Ritz值和滤波多项式阶数的轻量级条件数估计方法，并将其与阈值化的CholeskyQR变体相结合，实现了无显著精度损失的高效QR选择。

**🔧 技术方法**

采用的技术包括Chebyshev多项式滤波、子空间迭代、CholeskyQR、CholeskyQR2、shifted CholeskyQR2、Householder QR以及对条件数的解析上界推导和数值验证。

**📊 数据集**

使用了来自DFT和BSE的典型电子结构问题数据集，包括FLEUR生成的NaCl、AuAg、TiO₂等矩阵，以及FHI‑aims生成的Si/Graphene和Cu₂BaSnS₄等对称矩阵，尺寸从数千到十万不等。

**📈 对比分析**

通过与传统Householder QR的对比，实验表明动态CholeskyQR策略在保持相同收敛性和迭代次数的前提下，显著降低了QR时间（2–6倍加速），整体求解时间提升约10–20%，且在强缩放测试中显示出更好的并行效率。

**⚠️ 局限性**

主要限制是估计仍是上界，极端情形下可能过保守；对极度聚集谱的矩阵，滤波阶数选择可能导致条件数高估；此外shifted CholeskyQR的预处理步骤在某些硬件上仍可能带来额外通信开销。

---

## 307. Large language models can disambiguate opioid slang on social media

**arXiv ID:** 2603.10313 | [PDF](https://arxiv.org/pdf/2603.10313v1)

**作者:** Kristy A. Carpenter `[一作]` (Stanford University), Russ B. Altman `[通讯]` (Stanford University)

**通讯引用:** 59301 | [OpenAlex ID](https://openalex.org/A5084043782)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究利用大型语言模型对社交媒体文本进行解模糊，识别与阿片相关的内容，旨在提高低频、低可见性话题的监测效率。

**💡 创新点**

创新点在于将 LLM 的开放词汇和上下文推理能力用于大规模社交媒体文本的阿片术语去模糊，并在三种任务（基于词典、无词典、以及新兴俚语）中验证其优越性。

**🔧 技术方法**

使用 GPT‑4、GPT‑5、Claude Sonnet 4.5 与 Gemini 2.5 Pro 等四个商业 LLM 进行推理与标注。

**📊 数据集**

使用 2022 年 9 月 Spritzer 数据集、经纬度定位的纽约/加州推文集合，以及通过手工替换为 Pokémon 名称的模拟新兴俚语 80 条推文。

**📈 对比分析**

与六个现有阿片词典（DEA、RedMed 等）对比，LLM 在精确度、召回率、F1 等指标上均显著优于词典；在基于词典任务中 F1 从 0.126 提升到 0.824‑0.972，lexicon‑free 任务中 F1 从 0.08‑0.54 提升到 0.544‑0.769，突显其更高召回率。

**⚠️ 局限性**

局限包括：未对 GPT‑5、Claude、Gemini 进行专门的提示工程、手工标注覆盖有限、LLM 费用高昂、以及对个人监测的伦理风险。

---

## 308. Enhancing Network Intrusion Detection Systems: A Multi-Layer Ensemble Approach to Mitigate Adversarial Attacks

**arXiv ID:** 2603.10413 | [PDF](https://arxiv.org/pdf/2603.10413v1)

**作者:** Nasim Soltani `[一作]` (Institut national de la recherche scientifique), Anderson R. Avila `[通讯]` (Institut national de la recherche scientifique)

**通讯引用:** 449 | [OpenAlex ID](https://openalex.org/A5009407218)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了机器学习网络入侵检测系统（NIDS）对抗样本攻击的脆弱性，并提出一种两层防御模型（集成分类+自编码器），结合对抗训练提升鲁棒性。

**💡 创新点**

创新点包括：①将GAN和FGSM两种攻击方法用于网络流量生成；②设计了堆叠+自编码器的两层防御架构；③将对抗训练与自编码器结合，实现对未知攻击模式的自适应检测；④在NSL‑KDD和UNSW‑NB15两个标准数据集上系统性评估。

**🔧 技术方法**

采用的技术包括：生成对抗网络（GAN）、Fast Gradient Sign Method (FGSM) 对抗样本生成；集成学习（RF、DT、Bagging、KNN、LDA、GB、MLP + LR元分类器）；自编码器（四层全连接）进行异常重构；对抗训练；数据预处理（缺失值填补、序数/独热编码、标准化）等。

**📊 数据集**

使用的数据集为 NSL‑KDD（41特征）和 UNSW‑NB15（49特征），均为公开的网络流量入侵检测基准集。

**📈 对比分析**

通过在未改动、GAN 攻击、FGSM 攻击三种条件下与传统单一模型（DT、LR、LDA、KNN等）对比，评估准确率、召回率、F1 分数和检测率。结果显示，提出的两层模型在 GAN 攻击下仍保持约 90% 的检测率（NSL‑KDD）和 99%（UNSW‑NB15），而单模型在相同条件下性能大幅下降，证明该方法显著提升了 NIDS 的鲁棒性。

**⚠️ 局限性**

局限性包括：①实验仅覆盖二分类（正常/攻击），对多类别或更复杂流量场景的适用性待验证；②对抗样本仅来自 GAN 与 FGSM，可能无法覆盖更高级或混合攻击；③模型训练与推理的计算开销未详细评估，实时部署的可行性尚不明确；④在不同网络拓扑或更新的数据集上的泛化性仍需进一步研究。

---

## 309. Federated Active Learning Under Extreme Non-IID and Global Class Imbalance

**arXiv ID:** 2603.10341 | [PDF](https://arxiv.org/pdf/2603.10341v1)

**作者:** Chen-Chen Zong `[一作]` (Nanjing University of Aeronautics and Astronautics), Sheng-Jun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 4264 | [OpenAlex ID](https://openalex.org/A5103204774)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了联邦主动学习中的查询模型选择，并提出了一个能够在长尾和高异质环境下实现公平样本采样的框架FairFAL。

**💡 创新点**

创新点在于：①发现采样的类平衡度决定最终模型性能；②通过预测差异估计全局不平衡和本地与全局分布的偏差，自适应地在全局模型与本地模型之间切换；③利用全局特征原型生成伪标签，并采用两阶段不确定性-多样性平衡采样来实现精准且多样的查询。

**🔧 技术方法**

技术手段包括轻量级预测差异估计、全局特征原型与伪标签生成、两阶段k-center多样性采样、梯度嵌入空间、Dirichlet分区、熵/余弦不确定性采样等。

**📊 数据集**

使用的基准数据集包括 FMNIST、CIFAR-10、CIFAR-100、OctMNIST 以及 DermaMNIST。

**📈 对比分析**

与随机采样、传统不确定性/多样性方法、混合方法以及现有的 KAFAL、LoGo、IFAL 等 FAL 基线在五个基准上进行对比；在极端长尾和高异质设置下，FairFAL 在最终测试准确率上平均提升 3–6个百分点，表现最优。

**⚠️ 局限性**

局限性包括：对极端客户端隐私差异的鲁棒性未完全验证；阈值 δ、k-center 参数等仍需手动调优；目前仅针对单标签图像分类任务；在更大规模或非图像任务上的效果尚待进一步探索。

---

## 310. HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation

**arXiv ID:** 2603.10128 | [PDF](https://arxiv.org/pdf/2603.10128v1)

**作者:** Daichao Zhao `[一作]` (Shanghai Jiao Tong University), Qiankun Li `[通讯]` (Nanyang Technological University)

**通讯引用:** 911 | [OpenAlex ID](https://openalex.org/A5101431986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出HG‑Lane框架，利用双阶段ControlNet在不需要重新标注的前提下，生成高保真车道场景，涵盖雨雪雾、夜间、暮光等恶劣天气与光照条件，并基于此构建30k图像基准。

**💡 创新点**

创新点在于：①双阶段ControlNet（Canny + InstructPix2Pix）与语义融合预处理，可在冻结预训练模型的情况下实现多条件高保真生成；②直接保留车道标注，避免重标注成本；③构建公开基准，为后续研究提供统一评测。

**🔧 技术方法**

主要技术包括：扩散模型（Stable Diffusion）、ControlNet（Canny、InstructPix2Pix）、VAE、Canny边缘检测、语义融合预处理及无监督后处理。

**📊 数据集**

使用CULane正常图像进行生成，得到每类5k张，共30k张图像，作为新基准；同时在这些图像上评估CLRNet、GANet、FENet等SOTA车道检测模型。

**📈 对比分析**

通过与多种基线（CLRNet、GANet、FENet等）在F1@50、mF1等指标下对比，HG‑Lane使CLRNet整体mF1提升20.87%，各类别提升分别为8.63%、38.8%、14.96%、26.84%、21.5%、12.04%；相较传统生成/降噪方法，性能更佳且无额外推理时间。

**⚠️ 局限性**

局限性：生成过程中仍可能产生位置偏差；对极端光照细节的处理有限；完全依赖预训练模型，未对极端条件下的标注一致性做进一步优化。

---

## 311. Large Spikes in Stochastic Gradient Descent: A Large-Deviations View

**arXiv ID:** 2603.10079 | [PDF](https://arxiv.org/pdf/2603.10079v1)

**作者:** Benjamin Gess `[一作]` (Technische Universität Berlin), Daniel Heydecker `[通讯]` (University of Oslo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文对浅层全连接网络在NTK标度下的SGD训练进行了定量分析，提出了关于catapult阶段的精确理论。

**💡 创新点**

创新点在于给出了判定是否出现大幅“spike”的显式判据G(λ)和相应的概率指数ϑ(λ)，并证明了在不保证spike时其概率以(n/η)^{−ϑ/2}的多项式速率衰减。

**🔧 技术方法**

主要技术包括利用NTK框架将网络动力学简化为标量递推，结合大偏差理论（LDP）构造子/超马尔可夫过程，并使用强马尔可夫性质和可变尺度分解完成严格证明。

**📊 数据集**

研究主要是理论推导，并未针对具体实测数据集，而是以任意给定的训练样本{s_i,p_i}为参数来分析。

**📈 对比分析**

由于是理论分析，论文不涉及数值实验或与其它方法的直接性能对比，主要展示了在满足G(λ)>0时spike几乎必然出现，而G(λ)<0时spike的概率可被上界给出。

**⚠️ 局限性**

局限性在于模型极为简化（单层、均匀初始化、批量大小为1），不直接推广到深层或大批量训练，且对实际数据分布的依赖需要进一步验证。

---

## 312. Causally Grounded Mechanistic Interpretability for LLMs with Faithful Natural-Language Explanations

**arXiv ID:** 2603.09988 | [PDF](https://arxiv.org/pdf/2603.09988v1)

**作者:** Ajay Pravin Mahale `[一作]` `[通讯]` (Hochschule Trier), Ajay Pravin Mahale (Hochschule Trier)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了将因果机制分析与自然语言解释相结合的流水线，评估GPT‑2 Small在IOI任务中的注意头贡献。

**💡 创新点**

创新点在于将激活补丁的因果重要性与ERASER指标结合，并利用LLM自动生成具有因果可信度的解释。

**🔧 技术方法**

使用激活补丁、TransformerLens、ERASER指标以及LLM提示进行解释生成。

**📊 数据集**

基于GPT‑2 Small 124M的IOI提示数据集（50条）。

**📈 对比分析**

与基于注意力熵的基线和随机挑选的头部进行对比，电路方法获得100%充分性、22%完整性，F1为36%，明显优于注意力基线。

**⚠️ 局限性**

局限在于仅评估单一任务与单一模型、缺乏人类评估、固定头部集合、缺乏更大规模或多任务验证。

---

## 313. Pneuma-Seeker: A Relational Reification Mechanism to Align AI Agents with Human Work over Relational Data

**arXiv ID:** 2603.10747 | [PDF](https://arxiv.org/pdf/2603.10747v1)

**作者:** Muhammad Imam Luthfi Balaka `[一作]` (University of Chicago), Raul Castro Fernandez `[通讯]` (University of Chicago)

**通讯引用:** 3711 | [OpenAlex ID](https://openalex.org/A5003690515)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了名为 Pneuma‑Seeker 的 LLM 驱动的交互式数据发现与预处理系统，核心思想是通过关系重写（relational reification）将用户不断演化的信息需求表述为可执行的关系模式；

**💡 创新点**

创新点在于：①将信息需求转化为可迭代的关系模式，使系统与用户之间形成共享的可检验模型；②结合宏观/微观上下文管理的 LLM‑agent 体系，支持大规模表格的动态检索、查询与程序合成；③通过显式的关系图实现可解释性与追溯。

**🔧 技术方法**

主要技术包括：大型语言模型（OpenAI GPT‑4）、对话式规划与执行（ReAct‑style agent）、微观上下文提取脚本、结构化关系与语义算子、数据库后端（DuckDB）、检索器（基于表模式与内容匹配）。

**📊 数据集**

使用 KramaBench 6 个多表数据集（Archeology、Astronomy、Biomedical、Environment、Legal、Wildfire）进行评测，且在实际的芝加哥大学采购数据集上进行了部署验证。

**📈 对比分析**

与基线（基于 Python 代码生成、工具调用的 agentic 系统）对比，Pneuma‑Seeker 在答案质量上更高（如 Biomedical 94.4% vs 66%），成本更低（Token 量与费用均最优），并通过微观上下文抽取显著提升准确率。

**⚠️ 局限性**

主要局限包括：1）仍需人工交互进行需求细化；2）运行时和内存占用在大型数据集上会有一定开销；3）检索召回依赖检索器，部分表格可能被漏检；4）系统依赖强大 LLM，模型性能变化会影响效果。

---

## 314. Where Do Flow Semantics Reside? A Protocol-Native Tabular Pretraining Paradigm for Encrypted Traffic Classification

**arXiv ID:** 2603.10051 | [PDF](https://arxiv.org/pdf/2603.10051v1)

**作者:** Sizhe Huang `[一作]` (State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications), Shujie Yang `[通讯]` (State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出协议原生表格预训练框架 FlowSem‑MAE，用于加密流量分类，并通过流语义单元（FSU）改进自监督掩码自动编码器；

**💡 创新点**

创新点在于：①识别字节级掩码训练的三大偏差（字段不可预测、跨字段嵌入混淆、流级元数据丢失）；②提出可预测性引导过滤、字段专属嵌入和双轴注意力，以协议定义为先验构建模型；

**🔧 技术方法**

使用基于 FSU 的掩码自动编码器、双轴 Transformer（时间轴与字段轴注意力）、预测性过滤策略和均方误差预训练损失；

**📊 数据集**

预训练采用 MAWI 流量轨迹；评估数据集为 ISCX‑VPN（16 类）和 CSTNET‑TLS 1.3（TLS‑120，120 类）；

**📈 对比分析**

与字节级（Pcap‑Encoder、ET‑BERT）、图像级（YaTC、NetMamba）和混合模型（TrafficFormer、netFound）对比，采用冻结编码器和完整微调两种评估；FlowSem‑MAE 在冻结编码器上取得最高准确率（ISCX‑VPN 51.1%、TLS‑120 55.2%）并在半标记数据下达到与完整标记下 TrafficFormer 相当的性能；

**⚠️ 局限性**

限制在于需手工对字段进行可预测性划分，且在更大规模预训练集上可能获得进一步提升。

---

## 315. UAV traffic scene understanding: A cross-spectral guided approach and a unified benchmark

**arXiv ID:** 2603.10722 | [PDF](https://arxiv.org/pdf/2603.10722v1)

**作者:** Yu Zhang `[一作]` (Computer Network Information Center), Jin Tang `[通讯]` (Anhui University)

**通讯引用:** 12048 | [OpenAlex ID](https://openalex.org/A5030720334)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个大规模光学-热红外对齐的 UAV 交通 VQA 数据集 Traffic‑VQA，并提出跨谱认知网络 CTCNet 用于鲁棒的交通场景理解。

**💡 创新点**

创新点包括：① 引入外部交通规制记忆 TRM 通过原型引导知识嵌入（PGKE）填补领域知识缺口；② 设计质量感知光谱补偿模块（QASC）实现双向光谱上下文交换，解决光谱间干扰；③ 在同一数据集上统一多光谱、认知与感知任务的基准，促进跨模态研究。

**🔧 技术方法**

采用冻结的多模态大语言模型 Qwen3‑VL‑8B 作为基础，结合 PGKE（原型检索 + 交叉注意）和 QASC（双向注意的光谱补偿）模块；使用硬件同步采集的光学与热红外图像进行训练；在训练中保持视觉编码器与 LLM 解码器冻结，仅优化残差注入分支。

**📊 数据集**

使用 Traffic‑VQA 数据集，包含 8,180 对齐的 OPT‑TIR 图像及 1.3 M 题答对，涵盖 31 种问答类型，涉及多种光照与天气条件。

**📈 对比分析**

与多款公开 MLLM（MiniGPT‑v2、GeoChat、Qwen2.5‑VL‑7B 等）以及商业模型（GPT‑4o、Gemini‑2.5‑flash）在零样本、少样本和全量 fine‑tune 三种设置下对比；CTCNet 在多光谱综合模式下整体精度提升约 14%（从 47.6% 提升至 61.9%），认知任务精度提升约 4%（从 80.6% 提升至 84.8%），显著优于对照模型。

**⚠️ 局限性**

局限性：仅处理单帧图像，缺少时序视频分析；对极端天气（如深度雾/夜）下极少样本的鲁棒性仍有限；外部 TRM 需要专家手工构建，维护成本较高；模型对光谱极端失真（如热红外噪声）仍易受影响。

---

## 316. Riemannian Geometry-Preserving Variational Autoencoder for MI-BCI Data Augmentation

**arXiv ID:** 2603.10563 | [PDF](https://arxiv.org/pdf/2603.10563v1)

**作者:** Viktorija Poļaka `[一作]` (University of Groningen), Andreea Ioana Sburlea `[通讯]` (University of Groningen)

**通讯引用:** 1205 | [OpenAlex ID](https://openalex.org/A5005668305)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究并实现了一个保持黎曼几何的变分自编码器，用于生成有效的脑电协方差矩阵以用于运动想象脑机接口的数据增强。

**💡 创新点**

将黎曼几何的对数/指数映射与平行运输结合进VAE结构，保证生成的SPD矩阵保持对称正定，并学习跨受试者不变潜在空间。

**🔧 技术方法**

使用黎曼几何映射、对数/指数映射、平行运输、AIRM距离、KL正则化、Diversity Loss以及梯度裁剪等技术。

**📊 数据集**

使用Faller等人的13通道MI-BCI数据集，12名受试者，共5572个试验。

**📈 对比分析**

采用留一受试者交叉验证，比较生成数据对MDM、KNN、SVC分类器的影响；KNN在数据增强后提升约2–3%，SVC下降约4%，MDM几乎不变；标准Euclidean VAE产生大量无效样本。

**⚠️ 局限性**

合成数据多样性略低，对SVC等对决策边界敏感的分类器不利；结果依赖分类器，未能在所有分类器上提升。

---

## 317. FP-Predictor - False Positive Prediction for Static Analysis Reports

**arXiv ID:** 2603.10558 | [PDF](https://arxiv.org/pdf/2603.10558v1)

**作者:** Tom Ohlmer `[一作]` (Heinz Nixdorf Institute at Paderborn University), Eric Bodden `[通讯]` (Heinz Nixdorf Institute at Paderborn University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

训练基于图卷积网络的模型，利用Code Property Graph预测SAST工具报告的真假阳性。

**💡 创新点**

首次将图学习方法与CPG结合用于SAST误报过滤，并通过人工复核证明其对真阳性具有保守判别能力。

**🔧 技术方法**

采用图卷积网络（GCN）、CPG构建、词向量+节点类型+违规标记三类特征的深度学习模型。

**📊 数据集**

使用CamBenchCAP作为训练集，CryptoAPI‑Bench作为评估集。

**📈 对比分析**

与SAST工具原始报告和基准标注对比，CamBenchCAP测试集实现100%准确率，CryptoAPI‑Bench在人工复核后约96.6%准确率，FP检测率从3.7%提升至85.2%。

**⚠️ 局限性**

训练数据规模有限、仅覆盖单类漏洞，CPG缺失跨方法控制流，评估基准标注存在误差，泛化能力需进一步验证。

---

## 318. Compatibility at a Cost: Systematic Discovery and Exploitation of MCP Clause-Compliance Vulnerabilities

**arXiv ID:** 2603.10163 | [PDF](https://arxiv.org/pdf/2603.10163v1)

**作者:** Nanzi Yang `[一作]` (University of Minnesota), Kangjie Lu `[通讯]` (University of Minnesota)

**通讯引用:** 1637 | [OpenAlex ID](https://openalex.org/A5043198742)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对十个官方 MCP SDK 进行系统性分析，发现并利用其未实现的协议条款产生兼容性滥用攻击。

**💡 创新点**

提出兼容性滥用攻击的新攻击面，并设计语言无关的 IR 生成、混合静态‑LLM 分析和模态可利用性评估的三步框架。

**🔧 技术方法**

使用统一中间表示（IR）生成、LLM 辅助的静态分析以及基于 payload 与 timing 的模态可利用性分析技术。

**📊 数据集**

对十个官方 SDK（Python、TypeScript、Go 等）共 275 条协议条款进行评估，检测到 1,270 条非实现实例，进一步识别 1,265 条潜在风险。

**📈 对比分析**

通过人工评估验证结果，平均精度 86%、召回 87%，每条成本约 0.20 美元，分析时间仅数分钟；相较传统模板扫描，覆盖面更广、准确率更高。

**⚠️ 局限性**

局限性包括：依赖 LLM 可能产生误报；对语言细节的捕捉不够充分；报告数量庞大导致手工审核瓶颈；工具尚未实时同步规范演化。

---

## 319. Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers

**arXiv ID:** 2603.10744 | [PDF](https://arxiv.org/pdf/2603.10744v1)

**作者:** Wenhao Sun `[一作]` (University of Electronic Science and Technology of China), Zhaoqiang Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 28981 | [OpenAlex ID](https://openalex.org/A5010561682)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在DiT模型上提出了Just-in-Time（JiT）框架，利用空间冗余实现生成过程的加速；

**💡 创新点**

创新点包括：①空间近似生成ODE（SAG‑ODE）通过稀疏anchor token计算并利用增广lifet​or推算全局速度场；②确定性微流（DMF）实现阶段切换时新token的无缝统计校正；③重要性引导的token激活（ITA）在生成早期动态聚焦高频细节区域；

**🔧 技术方法**

技术手段主要有：流匹配ODE求解、Transformer自注意力、增广lifet​or插值、微分方程微流、局部速度方差重要性评估；

**📊 数据集**

实验基于FLUX.1-dev模型（预训练于大规模文本‑图像数据集，如LAION‑5B）进行；

**📈 对比分析**

与FLUX.1-dev（不同NFE）、RALU、Bottleneck、TaylorSeer、Teacache等SOTA加速方法对比，JiT在4×与7×加速下保持几乎无质量损失，CLIP‑IQA、ImageReward、HPSv2、GenEval、T2I‑CompBench分数均优于或持平于基线，显著提高速度-质量平衡；

**⚠️ 局限性**

局限性：仍需手工设计token激活时间表，适用于基于流匹配的DiT，未在视频或极高分辨率下验证，且对极低步数的稳健性尚待进一步研究。

---

## 320. Post-Quantum Entropy as a Service for Embedded Systems

**arXiv ID:** 2603.10274 | [PDF](https://arxiv.org/pdf/2603.10274v1)

**作者:** Javier Blanco-Romero `[一作]` (Universidad Carlos III de Madrid), Celeste Campo `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 847 | [OpenAlex ID](https://openalex.org/A5021832848)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

构建并评估了面向嵌入式设备的量子熵即服务（QEaaS）系统，利用Quantis QRNG通过CoAP/DTLS 1.3（后量子加密）将熵投递给ESP32微控制器，并在客户端实现BLAKE2s熵池。

**💡 创新点**

创新点在于：①将后量子加密（ML‑KEM‑512、ML‑DSA‑44）集成到CoAP/DTLS 1.3的轻量级实现；②提供双通道服务器（直接OpenSSL提供量子熵或混合Linux熵池），满足不同客户端需求；③为Zephyr ESP32扩展BLAKE2s熵池并支持外部熵注入，保持标准API。

**🔧 技术方法**

使用技术包括：Quantis PCIe QRNG硬件、Linux内核驱动、OpenSSL + OQS provider、wolfSSL 5.8.2（原生ML‑KEM/ML‑DSA）、libcoap 4.3.5、Zephyr RTOS 4.1.0、BLAKE2s熵池、CoAP‑HTTP代理、Nginx reverse proxy、Docker容器。

**📊 数据集**

使用的数据集为本地局域网环境下的实验数据：100 次握手/请求循环，量子熵源为Quantis PCIe-240M，测试环境包含ESP32‑DevKitC V4、Wi‑Fi 802.11n、Gigabit以太网服务器。

**📈 对比分析**

对比方法：在ESP32上测量DTLS 1.3握手+首个CoAP请求的延迟以及随后CoAP RTT，分别在三种密钥交换（ECDHE‑P‑256、X25519、ML‑KEM‑512）与两种签名（ECDSA、ML‑DSA‑44）的组合下，开启/关闭证书验证。结果显示：ML‑KEM‑512在无验证时平均 313 ms，配合 ML‑DSA‑44 为 225 ms；验证后 ML‑KEM‑512+ML‑DSA‑44 仍为 249 ms，显著快于经典 ECDHE‑P‑256+ECDSA（668 ms）并比其快 63%。CoAP RTT 接近 24 ms，熵池注入/提取 < 0.1 ms。

**⚠️ 局限性**

局限性：仅在 ESP32‑DevKitC V4 上验证；需要预先设定 512 B 熵池和 128 B 低阈值；网络环境固定在 LAN，未评估宽域网延迟；未给出能耗分析；后量子算法的实现受限于 wolfSSL 原生代码，未覆盖更高安全级别的 ML‑KEM；服务器端量子熵源与熵池混合方式仅在特定硬件/驱动上实现。

---

## 321. From Imitation to Intuition: Intrinsic Reasoning for Open-Instance Video Classification

**arXiv ID:** 2603.10300 | [PDF](https://arxiv.org/pdf/2603.10300v1)

**作者:** Ke Zhang `[一作]` (Johns Hopkins University), Di Fu `[通讯]` (TikTok)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个三阶段的“Intrinsic Reasoning”框架，先用冷启动监督对齐生成初始推理轨迹，再通过 Group Relative Policy Optimization (GRPO) 进行强化学习提升推理质量，最后训练一个“Intuitive Calibration”模块，将推理轨迹转换为稳定、校准的最终分类结果。

**💡 创新点**

创新点在于将 VLM 的隐式推理能力拆解出来，通过分阶段训练既提升推理质量又避免直接把推理输出当作最终答案，从而在开放实例视频分类任务中显著提升鲁棒性和泛化能力；同时提出了对推理轨迹进行分布一致的校准策略，解决了直接使用 RL 推理导致的判定不可靠问题。

**🔧 技术方法**

使用了 Vision‑Language Model (Qwen2.5‑VL‑7B) 作为基础模型，结合冷启动监督、GRPO 强化学习、以及后置的 Intuitive Calibration 模块；在强化学习阶段采用 DAPO‑style rollout 和规则驱动的奖励评估，后续的校准模块则是一个基于交叉熵的监督分类器。

**📊 数据集**

实验数据集包括公开的 SmartHome‑LLM（1,011 条真实家居监控视频）、MultiHateClip（2,000 条多语言仇恨视频）以及自研的大规模视频内容审核数据集（80K‑130K 条训练样本和 4.5K 条评估样本）。

**📈 对比分析**

与多类基线（传统视频编码器 UniFormerV2、InternVideo2‑6B；闭源 VLM GPT‑4、Gemini‑2.5；以及同源 Qwen2.5‑VL‑7B 在不同后置训练策略下）进行对比，结果表明该框架在 SmartHome‑LLM 上取得最高整体准确率和平均 F1 分数，在 MultiHateClip 上获得 72.72% 的整体准确率和 56.52% 的 Offensive 类 F1 分数，明显优于所有对照方法。

**⚠️ 局限性**

局限性包括：1）模型训练过程较为复杂，需多阶段、大量算力；2）推理轨迹生成与校准依赖于同一 VLM，若基础模型本身不足，提升有限；3）在极大推理长度或极端类目下仍可能出现信息冗余或校准失效。

---

## 322. The DMA Streaming Framework: Kernel-Level Buffer Orchestration for High-Performance AI Data Paths

**arXiv ID:** 2603.10030 | [PDF](https://arxiv.org/pdf/2603.10030v1)

**作者:** Marco Graziano `[一作]` `[通讯]` (Graziano Labs Corp), Marco Graziano (Graziano Labs Corp)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `afceb026-1760-41ae-8d86-010831a37d97` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个名为dmaplane的Linux内核模块，用于在AI工作负载（如分布式推理、Mixture of Experts、分布式训练和权重流式传输）中显式管理缓冲区生命周期、NUMA放置、跨设备共享、RDMA注册、GPU BAR映射以及完成安全的流控；

**💡 创新点**

创新点在于将缓冲区编排提升为第一类内核层，提供稳定的UAPI，整合dma-buf、NUMA、RDMA内核资源、GPU BAR pinning以及基于信用的完成流控，实现零拷贝跨设备共享与高效的点对点数据传输；

**🔧 技术方法**

利用Linux内核原语（dma_alloc_coherent、dma_alloc_attrs、dma_map_page、dma_buf、Kernel Verbs RDMA API、NVIDIA Peer‑to‑Peer BAR pinning）、ring‑based工作队列、信用流控机制以及对GPU BAR映射的优化；

**📊 数据集**

主要使用了AWS EC2实例（g5.xlarge）和Soft‑RoCE软件实现进行基准测试，没有特定机器学习数据集；

**📈 对比分析**

通过对比流控完整性（CQ溢出、信用停滞）、NUMA放置对跨节点 memcpy 带宽的影响、GPU BAR访问层级与 cudaMemcpy 的吞吐差异以及端到端分布式推理的时间分解，结果表明在设计的信用阈值下无 CQ 溢出，NUMA 放置误差仅在 DRAM 规模显著，GPU BAR 写合并映射可提升 Host→GPU 传输；

**⚠️ 局限性**

仅在 Soft‑RoCE 软件实现上验证，未在硬件 RDMA NIC 上测试；缺乏集体通信、服务框架、生产级容错与多租户隔离功能；未来工作需在硬件 RDMA 上进一步验证和集成。

---

## 323. Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression

**arXiv ID:** 2603.10470 | [PDF](https://arxiv.org/pdf/2603.10470v1)

**作者:** Hamidreza Dastmalchi `[一作]` (York University), Hamed Barzamini `[通讯]` (Northern Illinois University)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5082230168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不需要训练的前提下，提出一种在推理阶段通过投影隐藏状态来抑制大规模视觉语言模型中视觉诱发幻觉的方法。

**💡 创新点**

创新点在于利用扩散模型生成对比的伪图像，从中提取低秩幻觉子空间，并在推理时直接投影隐藏表示以消除幻觉。

**🔧 技术方法**

使用Stable Diffusion生成伪图像，利用SVD得到幻觉子空间，并在推理阶段对LVLM隐藏状态做线性投影。

**📊 数据集**

在MSCOCO的5k图像-标题对上构造OHC-25K对照集，并在CHAIR、OPOPE、MMHal、LLaVA-Bench等多项基准上进行评估。

**📈 对比分析**

与DoLa、VCD、OPERA、HALC、Nullu等最先进的测试时或后处理方法相比，CIPHER在CHAIR和BLEU等指标上显著降低幻觉率，同时保持或提升生成质量；在推理吞吐量上几乎与普通贪心解码相同。

**⚠️ 局限性**

局限性包括需要离线生成大量伪图像并进行SVD，投影子空间为全局固定，可能缺乏对不同输入上下文的自适应；对极端视觉噪声或特殊场景的鲁棒性仍有提升空间。

---

## 324. HTMuon: Improving Muon via Heavy-Tailed Spectral Correction

**arXiv ID:** 2603.10067 | [PDF](https://arxiv.org/pdf/2603.10067v1)

**作者:** Tianyu Pang `[一作]` (Dartmouth), Yaoqing Yang `[通讯]` (Dartmouth)

**通讯引用:** 4120 | [OpenAlex ID](https://openalex.org/A5020994183)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的矩阵基优化器HTMuon，利用幂变换的奇异值操作，使更新和权重谱更重尾，提升模型训练效果。

**💡 创新点**

创新点在于：①在保持矩阵基优化器捕获参数耦合优势的同时，引入p∈(0,1)的幂变换，产生更重尾的更新；②证明HTMuon等价于 Schatten‑q 范数约束下的最速下降；③给出光滑非凸场景下的收敛分析；④设计两种高效实现，显著降低计算开销。

**🔧 技术方法**

技术方法包括：矩阵动量预条件、奇异值分解与幂变换、Schatten‑q 范数约束的最速下降理论、随机梯度估计、数值迭代加速等。

**📊 数据集**

实验数据集：LLM 预训练使用 C4（LLaMA、GPT‑2）和 OpenWebText；图像分类使用 CIFAR‑10、CIFAR‑100 和 ImageNet‑1K（ResNet、ViT）。

**📈 对比分析**

对比方法包括 Adam、AdamW、RAdam、Lion、MuOn、Ranger、AdaBelief 等常用优化器；在 LLaMA 上，HTMuon 在 60M、135M、350M 模型上将困惑度降低约 0.92–0.98；在 ResNet 上提升准确率约 0.24–0.31%；在 ViT 上也优于基线。

**⚠️ 局限性**

局限性：未在 1B 以上模型或更大规模数据集上验证；相较于基础 MuOn，单步计算开销略高，需更频繁更新才能获得最佳效果；加速实现仍可进一步改进。

---

## 325. Joint Imaging-ROI Representation Learning via Cross-View Contrastive Alignment for Brain Disorder Classification

**arXiv ID:** 2603.10253 | [PDF](https://arxiv.org/pdf/2603.10253v1)

**作者:** Wei Liang `[一作]` (Lehigh University), Lifang He `[通讯]` (Lehigh University)

**通讯引用:** 8190 | [OpenAlex ID](https://openalex.org/A5071709543)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种联合影像与 ROI 图表示的跨视角对比学习框架，用于脑疾病（ADHD、ASD）分类。

**💡 创新点**

创新点在于使用双向 InfoNCE 对齐全局影像嵌入与局部 ROI 图嵌入，并在统一训练协议下实现对单一分支与联合分支的系统评估。

**🔧 技术方法**

主要技术包括 3D 卷积/Transformer 影像编码器（如 3DSC‑TF、ViT3D*、RAE‑ViT）、图神经网络 ROI 编码器（NeuroGraph、DNN）、交叉视角对比学习以及拼接/注意力/对比融合与分类器。

**📊 数据集**

实验数据集为结构 MRI 数据：ADHD‑200（776 例）和 ABIDE（1112 例）。

**📈 对比分析**

在 5‑折交叉验证中，联合学习在准确率、AUC、F1 上均显著优于单分支（影像或 ROI），并在缺失视角（10%、30%、50%）下仍保持相对稳定。

**⚠️ 局限性**

局限包括仅评估结构 MRI，未验证功能影像或多模态场景；对缺失视角的鲁棒性虽较好但仍需两分支并行训练；缺乏外部临床样本验证。

---

## 326. Amnesia: Adversarial Semantic Layer Specific Activation Steering in Large Language Models

**arXiv ID:** 2603.10080 | [PDF](https://arxiv.org/pdf/2603.10080v1)

**作者:** Ali Raza `[一作]` (Honda Research Institute Europe), Jibesh Patra `[通讯]` (IIT KharagPur)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级的激活空间对抗攻击“Amnesia”，可在不训练或修改权重的情况下通过对内部Transformer注意力值进行微调，绕过开源LLM的安全机制，生成有害内容。

**💡 创新点**

创新点在于只需在单个安全相关层的注意力值上进行局部减法，即可显著提高模型的攻击成功率，且无需梯度优化、提示生成或权重编辑，极大降低了攻击门槛与计算成本。

**🔧 技术方法**

采用了对内部注意力值的激活提取与解码、基于关键词的安全相关向量构造、以及在推理阶段对指定层的值流进行缩放减法等技术；评估时使用了MMLU、SAMSum、Perplexity等常用基准来验证对正向任务的影响。

**📊 数据集**

使用了 WildJailbreak、AdvBench（Harmful_Behaviours）、HarMBench（标准行为）以及自定义的“Forbidden Questions”数据集进行攻击效果评估；在Llama‑2‑7B‑Chat、Llama‑3‑8B‑Instruct和Qwen‑7B‑Chat等模型上进行实验。

**📈 对比分析**

与现有基于梯度的白盒攻击、提示生成攻击以及全局激活方向投影等方法对比，Amnesia在不需要训练或修改权重的前提下，攻击成功率（ASR）可提升至70%–90%，在多模型上表现一致；同时对正常任务的影响微乎其微（MMLU提升0.3%，ROUGE变化不大）。

**⚠️ 局限性**

局限性包括：需手工挑选合适的缩放因子α和关键词集合，方法针对单一层的局部操作可能在不同架构或更强安全机制下失效；未对循环生成等异常行为进行系统化控制，且未验证在更大规模模型或工业部署模型中的可扩展性。

---

## 327. GATech at AbjadGenEval Shared Task: Multilingual Embeddings for Arabic Machine-Generated Text Classification

**arXiv ID:** 2603.10007 | [PDF](https://arxiv.org/pdf/2603.10007v1)

**作者:** Ahmed Khaled Khamis `[一作]` `[通讯]` (Georgia Institute of Technology), Ahmed Khaled Khamis (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对阿拉伯语 AI 生成文本检测任务，构建并微调了多语言 E5-large 编码器，比较了多种池化策略，并提出了带多样本 dropout 和层级学习率衰减的训练方案。

**💡 创新点**

主要创新在于系统地比较了复杂池化策略与简单均值池化的效果，发现均值池化在有限训练数据下表现更佳，并阐释了其原因。

**🔧 技术方法**

使用的技术包括多语言 E5-large 预训练模型、均值池化、加权层池化、多头注意力池化、门控融合、层级学习率衰减、多样本 dropout 以及 Focal Loss。

**📊 数据集**

数据集为 AbjadEval 共享任务提供的 5,298 条阿拉伯语样本，人工文本与机器生成文本各 2,649 条。

**📈 对比分析**

通过在测试集上对比不同池化方法，均值池化取得最高 F1 分数 0.75，其他复杂池化方法在测试集表现逊色。

**⚠️ 局限性**

局限性在于仅使用了比赛提供的数据，未利用外部数据；并且最大序列长度限制导致长文本截断，可能导致关键信息丢失。

---

## 328. Guiding Diffusion Models with Semantically Degraded Conditions

**arXiv ID:** 2603.10780 | [PDF](https://arxiv.org/pdf/2603.10780v1)

**作者:** Shilong Han `[一作]` (National University of Defense Technology), Hongxia Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 14833 | [OpenAlex ID](https://openalex.org/A5021226984)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Condition‑Degradation Guidance (CDG)，通过在Transformer文本编码器中只降解内容token来构造语义退化的负样本，替代传统CFG使用的空提示，从而改进文本到图像扩散模型的引导方式。

**💡 创新点**

创新点在于发现Transformer文本编码器中内容token与上下文聚合token的功能分离，并利用只降解内容token的分层退化策略，构造“好 vs. 几乎好”的对比，从而显著降低指导信号与去噪方向的几何耦合。

**🔧 技术方法**

采用自注意力图的加权PageRank评估token重要性，生成二进制掩码对prompt进行分层退化；在扩散模型中以CFG形式引入该负样本；并在Stable Diffusion 3、SD3.5、FLUX.1、Qwen‑Image等多种架构上实现。

**📊 数据集**

主要使用MS‑COCO 2017验证集的5,000条caption进行评估，并在GenAI‑Bench上检验复杂组合推理能力。

**📈 对比分析**

与CFG、CADS、ICG、PAG、SEG、SFG、DNP等基线在四大模型上对比，CDG在FID、CLIPScore、AestheticScore、VQA Score以及GenAI‑Bench指标上均取得最佳或接近最佳，提升幅度多达数个百分点，且运行开销仅为CFG的极小百分比。

**⚠️ 局限性**

在已采用指导蒸馏的模型（如FLUX.1）提升有限；退化参数需手动选择，虽然默认1.0效果良好，但对不同模型的通用性及极端复杂提示的鲁棒性仍需进一步验证。

---

## 329. Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework

**arXiv ID:** 2603.10281 | [PDF](https://arxiv.org/pdf/2603.10281v1)

**作者:** Rajesh Shrestha `[一作]` (Oregon State University), Xiao Fu `[通讯]` (Oregon State University)

**通讯引用:** 12458 | [OpenAlex ID](https://openalex.org/A5053690968)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种三阶段AC-DC denoiser，并将其嵌入ADMM-Plug‑and‑Play框架，用以解决逆问题中的噪声数据流形不匹配和收敛性难题。

**💡 创新点**

创新点在于①设计了自动校正（AC）+方向校正（DC）+基于分数的去噪三阶段结构，显著缓解了ADMM迭代与训练流形的几何不匹配；②给出了AC-DC denoiser满足弱非扩张残差的理论保证；③在常数步长与自适应步长两种调度下均给出了收敛性证明。

**🔧 技术方法**

技术方法包括基于预训练扩散模型的分数函数、Langevin动力学的方向校正、Tweedie定理去噪、ODE去噪、ADMM优化、弱非扩张与自适应步长调度等。

**📊 数据集**

使用公开图像数据集FFHQ和ImageNet（256×256）进行实验，涵盖图像插值、超分辨率、模糊恢复、相位恢复等多种逆问题。

**📈 对比分析**

与DPS、DAPS、DDRM、DiffPIR、RED‑diff、DPIR、DCDP、PMC等基线比较，采用PSNR、SSIM和LPIPS三种指标；实验结果显示本方法在大多数任务中均获得最高或第二高的指标，显著优于现有PnP和扩散模型方法。

**⚠️ 局限性**

局限性包括：需要多次分数评估导致计算开销大；自适应步长调度在实践中不够直观，常数步长在非凸情形下缺乏理论保证；噪声调度仍基于经验，缺乏问题自适应策略；未给出恢复误差或可恢复性理论分析。

---

## 330. Interpretable Chinese Metaphor Identification via LLM-Assisted MIPVU Rule Script Generation: A Comparative Protocol Study

**arXiv ID:** 2603.10784 | [PDF](https://arxiv.org/pdf/2603.10784v1)

**作者:** Weihang Huang `[一作]` (University of Birmingham), Mengna Liu `[通讯]` (Guangdong University of Foreign Studies)

**通讯引用:** 1890 | [OpenAlex ID](https://openalex.org/A5034243574)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型的可解释规则脚本管线，用四种不同的隐喻识别协议对中文文本进行判定；

**💡 创新点**

将多种理论化协议（MIP/MIPVU、CMDAG、情感、明喻）统一成可执行、可审计的规则脚本，系统实现全流程可解释与可编辑；

**🔧 技术方法**

使用模块化管线（分词、候选选取、语义分析、分类、理由生成），在关键步骤插入GPT‑4的可控调用，并对输出做结构化解析；

**📊 数据集**

七个中文隐喻数据集：PSU CMC、CMC、CMDAG、中国明喻、NLPCC 2024 T9、ConFiguRe、ChineseMCorpus，覆盖词、句、段级标注；

**📈 对比分析**

在各自对应的标注尺度下评估四协议的 F1（A:0.472，B:0.347，C:0.334，D:0.392）并在同一数据集上做跨协议比较（F1从0.018到0.829，κ从0.001到0.986），与基线和文献值对照，显示协议差异是主要性能来源；

**⚠️ 局限性**

局限包括：基线评估依赖近似文献值、使用专有GPT‑4导致可复现性受限、跨协议评估仅在MIPVU标注集上完成、解释性评估样本有限、模型对中文文化情感认知不足；

---

## 331. Decision-Aware Uncertainty Evaluation of Vision-Language Model-Based Early Action Anticipation for Human-Robot Interaction

**arXiv ID:** 2603.10061 | [PDF](https://arxiv.org/pdf/2603.10061v1)

**作者:** Zhaoda Du `[一作]` (Colorado School of Mines), Xiaoli Zhang `[通讯]` (Colorado School of Mines)

**通讯引用:** 32252 | [OpenAlex ID](https://openalex.org/A5100404566)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

系统地评估了视觉-语言模型在短期、部分观察下的动作预测不确定性，并构建了基于时间前缀的评价框架。

**💡 创新点**

创新点在于将Top‑K预测视为可靠性问题，提出了三种聚合策略与四维决策相关评估指标（正确性、校准、选择性决策、置信几何）。

**🔧 技术方法**

采用多次随机解码采样、置信聚合（一致性、置信加权、PairRank）以及期望校准误差、集合校准误差和归一化熵等评价度量。

**📊 数据集**

使用两大第一人称动作预测基准数据集——EGTEA Gaze+和EPIC‑KITCHENS‑100。

**📈 对比分析**

与单次推断基线对比，聚合方法在召回率上相差不大，但在集合校准和阈值选择下表现出不同的置信度分布与选择性准确率；PairRank在较大K时提升了集合校准但缺乏顶层校准。

**⚠️ 局限性**

局限性包括仅评估单一黑盒VLM、未在真实机器人系统中验证、参数固定且未考虑自适应或上下文感知的聚合与阈值设定。

---

## 332. GSVD for Geometry-Grounded Dataset Comparison: An Alignment Angle Is All You Need

**arXiv ID:** 2603.10283 | [PDF](https://arxiv.org/pdf/2603.10283v1)

**作者:** Eduarda de Souza Marques `[一作]`, Daniel Sadoc Menasche `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一种基于GSVD的角度统计量θ(z)的测试样本分类与可解释性评估流程。

**💡 创新点**

创新点在于利用投影子空间的角度来衡量样本与两类子空间的相似度，并通过角度直方图评估类间可混淆性。

**🔧 技术方法**

主要技术包括矩阵伪逆投影、GSVD、角度统计、直方图重叠度量以及Bhattacharyya距离。

**📊 数据集**

在标准图像分类数据集（如MNIST、CIFAR‑10）上进行了实验验证。

**📈 对比分析**

与传统分类器和其他可解释性方法相比，该方法在区分度和可解释性方面取得了较优表现，尤其在类间混淆明显的样本上显著提升。

**⚠️ 局限性**

局限性包括只能处理二分类或需要预先构造子空间，且对高维噪声敏感，未来需扩展到多分类与非线性子空间。

---

## 333. S2D: Sparse to Dense Lifting for 3D Reconstruction with Minimal Inputs

**arXiv ID:** 2603.10893 | [PDF](https://arxiv.org/pdf/2603.10893v1)

**作者:** Yuzhou Ji `[一作]` (Shanghai Jiao Tong University), Xin Tan `[通讯]` (East China Normal University)

**通讯引用:** 10477 | [OpenAlex ID](https://openalex.org/A5069250588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出S2D框架，实现稀疏输入到3D Gaussian Splatting的稀疏到密集提升，结合一阶扩散修复器和随机样本丢弃+加权梯度的重建策略。

**💡 创新点**

一阶扩散模型实现高效图像修复，支持点云与参考图双向指导；随机样本丢弃与加权梯度确保稀疏视角下3D一致性与拟合稳健。

**🔧 技术方法**

Latent Diffusion Model（pix2pix-turbo）、VFM（如π^3）点云生成、3D Gaussian Splatting、DINO特征混合、随机采样与加权梯度优化。

**📊 数据集**

DL3DV-960、3DOVS、MIP360、RE10K、Waymo Open Dataset等。

**📈 对比分析**

与传统3DGS、MIP‑Splatting、ScaffoldGS、NoPoSplat、AnySplat、DepthSplat、ViewCrafter、SEVA、DIFIX、StreetGaussians、EmerNeRF、StreetCrafter等基线对比；在PSNR、SSIM、LPIPS、FID指标上S2D显著优于所有对照，尤其在极稀疏输入下表现最佳。

**⚠️ 局限性**

仍受点云生成质量影响，极端视角偏差时可能出现轻微结构误差；大规模动态场景的实时性与计算成本仍需进一步提升。

---

## 334. AMB-DSGDN: Adaptive Modality-Balanced Dynamic Semantic Graph Differential Network for Multimodal Emotion Recognition

**arXiv ID:** 2603.10043 | [PDF](https://arxiv.org/pdf/2603.10043v1)

**作者:** Yunsheng Wang `[一作]` (Central South University of Forestry and Technology), Keqin Li `[通讯]` (State University of New York)

**通讯引用:** 30919 | [OpenAlex ID](https://openalex.org/A5087894632)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自适应模态平衡的动态语义图差分网络（AMB-DSGDN）用于多模态对话情感识别。

**💡 创新点**

通过差分图注意机制消除共享噪声，构建模态特定子图捕获内外说话人情感依赖，并引入自适应模态dropout动态调节模态贡献。

**🔧 技术方法**

使用Transformer编码、图注意力网络（DiffRGCN）、差分注意力、模态自适应dropout以及多头注意力实现多模态融合与情感建模。

**📊 数据集**

在IEMOCAP和MELD两大多模态对话情感数据集上进行实验。

**📈 对比分析**

与多种RNN、GCN及多模态融合基线对比，AMB-DSGDN在IEMOCAP上取得最高的加权准确率76.09%和加权F1 75.64%，在MELD上性能接近最优，显著提升特定情绪类别识别。

**⚠️ 局限性**

由于图注意力和差分机制的计算开销，模型在极长对话或资源受限环境下推理速度受限，需进一步优化轻量化和加速。

---

## 335. OAuthHub: Mitigating OAuth Data Overaccess through a Local Data Hub

**arXiv ID:** 2603.10056 | [PDF](https://arxiv.org/pdf/2603.10056v1)

**作者:** Qiyu Li `[一作]` (University of California San Diego), Haojian Jin `[通讯]` (University of California Riverside)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 OAuthHub，一种使用个人设备作为本地数据中介的 OAuth 框架，允许开发者通过声明式 manifest 进行细粒度数据访问，并在用户设备上实现统一的运行时权限管理。

**💡 创新点**

创新点在于：①把个人设备变成 OAuth 数据中心，解决传统 OAuth 对始终在线、静态 IP 的依赖；②通过声明访问时机和操作链的 Pipe‑and‑Filter 架构实现细粒度授权；③提供统一的可视化权限面板和日志审计，提升用户对第三方访问的透明度和控制力。

**🔧 技术方法**

技术主要包括：TypeScript 库实现 OAuthHub API，Chrome 扩展和 Android App 作为本地服务；manifest 采用文本定义的操作流水线；利用 OAuth 2.0 规范的 PKCE、JWT 验证等安全措施；对数据进行 GraphQL 查询、过滤、提取、匿名化等操作；基于 Node.js 的跨平台运行时。

**📊 数据集**

数据集主要为真实 OAuth 应用的功能与权限信息：首先分析 62 个 Google Workspace Marketplace 应用的请求与实际需求；随后扩展到 218 个跨平台 OAuth 应用（包括 Google、Trello、Slack、GitHub 市场），并对其进行 manifest 编写与手工标注；实验中使用了 500 封邮件、97 条日历事件和 9 条表单响应等自建数据。

**📈 对比分析**

比较方法：①通过 18 名 CS 学生完成四项编程任务，测量完成时间、代码行数和 NASA‑TLX 认知负荷；②在 100 名受访者中对传统 OAuth 与 OAuthHub 的同意率进行对比；③在 PC（M1 MacBook）和 Android 手机上测量 CPU、内存、延迟和能耗。结果显示：任务完成时间平均从 18.0 min 降至 9.1 min，代码行数从 15.8 行降至 4.7 行；用户同意率提升 56%–78%；系统在 PC 上 CPU <1%，内存 90 MiB，延迟约 195 ms；在手机上 CPU <5%，内存 200 MiB，延迟 849 ms，能耗提升 12%–39%。

**⚠️ 局限性**

局限性包括：1) 不适用于大文件或频繁写入的高数据量 OAuth 场景；2) 需要服务提供商和开发者更新 schema 与 manifest，维护成本较高；3) 需要用户在设备上安装并信任本地扩展/应用，可能导致接受度不高；4) 对离线时段的特殊需求（如 CI/CD 触发）支持有限；5) 当前实现只支持主流 OAuth Provider，尚未覆盖所有 API 细粒度控制。

---

## 336. Huffman-Bucket Sketch: A Simple $O(m)$ Algorithm for Cardinality Estimation

**arXiv ID:** 2603.10930 | [PDF](https://arxiv.org/pdf/2603.10930v1)

**作者:** Matti Karppa `[一作]` `[通讯]` (University of Gothenburg), Matti Karppa (University of Gothenburg)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出 Huffman‑Bucket Sketch (HBS)，通过将 HLL 的寄存器分桶并使用全局 Huffman 码表实现无损压缩，达到 O(m+log n) 位的最优空间，并保持可合并性和摊销 O(1) 的更新。

**💡 创新点**

创新点在于：①使用全局基于估计基数的 Huffman 码表，将寄存器值压缩到最优信息熵；②证明树重建仅需 O(log n) 次；③在保持 Mergeable 的前提下实现近乎常数时间更新。

**🔧 技术方法**

核心技术包括：基于 Poisson 计数模型的寄存器值分布分析、Huffman 码生成与动态重建、桶化设计（桶大小 O(log n)）、全局基数估计与局部最小值维护。

**📊 数据集**

实验数据使用随机生成的 HLL 模拟数据（n=2^30、m=2^15 以及多种负载因子 λ），未采用公开真实数据集。

**📈 对比分析**

与原始 HLL、ExaLogLog 等传统算法对比，HBS 的 Memory‑Variance Product (MVP) 在 3.5–4.1 范围内，接近 ExaLogLog 的 3.67，且在合并与插入操作上保持常数级时间，显示出实用性和竞争力。

**⚠️ 局限性**

局限性：需要 m 远大于 log²n 才能实现常数级操作；重建 Huffman 树及全桶重新编码在极端情况下仍可能导致 O(m) 时间；仅适用于 HLL 或类似集中分布的 sketch，且对基数估计误差敏感。

---

## 337. Towards Robust Speech Deepfake Detection via Human-Inspired Reasoning

**arXiv ID:** 2603.10725 | [PDF](https://arxiv.org/pdf/2603.10725v1)

**作者:** Artem Dvirniak `[一作]` (MIRAI), Oleg Y. Rogov `[通讯]` (Applied AI Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了人类注释的深度伪造音频推理数据集，并提出了结合大型音频语言模型（LALM）与链式推理（CoT）的 HIR‑SDD 框架，既能检测真假音频，又能给出可解释的推理链。

**💡 创新点**

创新点在于：①首个针对语音深度伪造的高质量人类推理数据集；②通过硬标签和 CoT 的联合训练、音频 grounding 以及强化学习（GRPO）提升模型的解释性和鲁棒性；③证明 LALM 在此任务中可实现与传统模型相当甚至更优的检测效果。

**🔧 技术方法**

技术主要包括：大型音频语言模型（SALMONN、Qwen2‑Audio 等） + LoRA 微调；链式推理（CoT）训练与生成；音频 grounding 通过在音频中嵌入可感知的噪声、时间掩蔽等；强化学习（GRPO）与外部评估模型（Qwen2.5‑32B、GPT‑5.1）进行奖励优化；评价指标包括准确率、平衡准确率、F1 以及推理质量（recall/relevance/logic/helpfulness）和 Jaccard 相似度。

**📊 数据集**

使用了多源公开数据集（ASVspoof 5、PyAra、LibriSecVoc、MLAAD、DFADD、M‑AILABS、Golos、SOVA、RuLS、SpeechEval）以及合成数据（XTTS‑V2、ESpeech、ElevenLabs）。

**📈 对比分析**

与传统的 Wav2Vec2‑AASIST 以及 Speech‑DF‑Arena 公开榜单模型比较，SALMONN‑7B 在准确率、平衡准确率和 F1 上均优于传统模型；CoT+GRPO 的推理质量在评估模型上略有提升，但在分类指标上提升不显著；整体仍保持与最佳开源模型相近的性能。

**⚠️ 局限性**

局限性包括：①在未见过的高保真合成模型上仍易误判；②推理链的多样性和一致性受训练数据的限制；③GRPO 对 Jaccard 指标提升有限；④对不同语言或极端噪声条件下的泛化能力尚待进一步验证。

---

## 338. Building Privacy-and-Security-Focused Federated Learning Infrastructure for Global Multi-Centre Healthcare Research

**arXiv ID:** 2603.10063 | [PDF](https://arxiv.org/pdf/2603.10063v1)

**作者:** Fan Zhang `[一作]` (University of Cambridge), Michael Roberts `[通讯]` (University of Cambridge)

**通讯引用:** 4193 | [OpenAlex ID](https://openalex.org/A5018385899)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建了FLA3平台，实现了跨境多中心医疗研究中的可治理联邦学习基础设施，支持运行时治理、访问控制和加密审计。

**💡 创新点**

创新点在于将XACML兼容的属性访问控制、时间绑定的研究授权、加密审计以及多研究隔离整合到联邦学习的调度层，实现可执行的治理与数据主权。

**🔧 技术方法**

使用的技术包括Flower框架、XACML规则引擎Luas、mtls认证、JSON Web Signature审计、FedMAP个人化联邦算法以及容器化部署。

**📊 数据集**

数据集为INTERVAL试验的25个站点共54,446个全血细胞计数样本，平台也在BloodCounts! Consortium的五个跨国机构上部署。

**📈 对比分析**

通过个体训练、FedMAP联邦训练与中央化训练三种策略比较，FedMAP在各中心平均ROC‑AUC提升0.027，几乎与中央化模型相当，同时显著降低跨中心和跨地区差异。

**⚠️ 局限性**

局限包括：实验仍为模拟联邦，未完成现场端到端训练；未验证与差分隐私/安全聚合的组合；策略手工编写易出错；仅防止未授权参与，无法抵御Byzantine攻击。

---

## 339. What do near-optimal learning rate schedules look like?

**arXiv ID:** 2603.10301 | [PDF](https://arxiv.org/pdf/2603.10301v1)

**作者:** Hiroki Naganuma `[一作]`, George E. Dahl `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过定义多种可参数化的学习率调度曲线家族，并在三种不同的训练任务上进行系统搜索与评估，寻找每个家族内近似最优的学习率曲线形状。

**💡 创新点**

创新点在于：①提出基于样条插值的灵活调度家族，覆盖并超越传统余弦、线性和倒数平方根等曲线；②采用两步搜索（先搜索形状后优化基准学习率）来获得近似最优形状；③首次给出线性回归问题的解析最优学习率曲线，并与深度学习任务的最优曲线进行对比；④发现warmup+递减阶段是几乎所有最优曲线的共性。

**🔧 技术方法**

技术手段包括：随机搜索（在每个家族内采样形状参数并对16个基准学习率进行网格搜索）、多重随机种子评估、统计中位数和置信区间计算、线性回归的解析最优求解、对比实验与常用基线（常数、线性、余弦）进行性能比较。

**📊 数据集**

使用的数据集与模型：①线性回归合成数据（指定协方差谱）；②小型CNN在某图像分类数据集（可能为CIFAR-10/100）；③8M参数的Transformer语言模型在WikiText等文本数据集。

**📈 对比分析**

比较方法：对每个工作负载取中位数训练误差或训练/测试困惑度作为指标，使用多达1000个种子进行评估。结果显示，最优调度曲线相较于常数、线性和传统余弦基线，在训练误差/困惑度上平均提升约0.02–0.05，线性回归的最优曲线无warmup且在训练末期快速衰减。

**⚠️ 局限性**

局限性：仅在规模较小、计算量有限的任务上验证；随机搜索对高维调度家族效率不足，可能未挖掘最优解；未对学习率以外的超参数（如动量、Adam β 的调度）进行联合搜索；未在更大、更复杂的模型和数据集上检验结果的普适性。

---

## 340. AraModernBERT: Transtokenized Initialization and Long-Context Encoder Modeling for Arabic

**arXiv ID:** 2603.09982 | [PDF](https://arxiv.org/pdf/2603.09982v1)

**作者:** Omar Elshehy `[一作]` (Saarland University), Mona Abdelazim `[通讯]` (Ain Shams University)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5084111871)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了 AraModernBERT，一种针对阿拉伯语的现代Encoder模型，主要聚焦于转tokenization嵌入初始化和原生8,192-token长上下文处理。

**💡 创新点**

创新点在于将 ModernBERT 的长上下文机制与转tokenization 嵌入初始化结合，专门为阿拉伯语这种形态学丰富、词汇稀疏的语言进行定制。

**🔧 技术方法**

采用了 ModernBERT 架构（交替局部/全局注意力、RoPE 位置信息）、BPE tokenizer、转tokenization 嵌入对齐、MLM 预训练目标以及 8,192-token 原生上下文。

**📊 数据集**

预训练使用约 100 GB 阿拉伯语文本；下游评估涵盖阿拉伯维基、XNLI (Arabic)、OOLD、MQ2Q、WikiAnn、ANERCorp、AQMAR、Twitter NER 等数据集。

**📈 对比分析**

通过与随机/重新初始化嵌入的对照组比较，MLM loss 从 11.46/10.98 降至 3.24，perplexity 从 94,372/58,962 降至 25.54；长上下文下 8,192-token 的 loss 进一步提升至 3.05（perplexity 21.05）；下游任务在 NLI、毒性检测、语义相似性、NER 等上取得 0.47/0.87/0.96 的准确率/F1，表明模型迁移性能良好。

**⚠️ 局限性**

局限性包括：下游实验未充分利用长上下文优势，缺乏对长篇问答或信息抽取等任务的评估；仅验证阿拉伯语，未检验对其他阿拉伯字母语言的适用性；预训练规模相对有限，难以与大规模英文学术模型直接对比。

---

## 341. Code-Space Response Oracles: Generating Interpretable Multi-Agent Policies with Large Language Models

**arXiv ID:** 2603.10098 | [PDF](https://arxiv.org/pdf/2603.10098v1)

**作者:** Daniel Hennes `[一作]` (Google DeepMind), Marc Lanctot `[通讯]` (Google DeepMind)

**通讯引用:** 26276 | [OpenAlex ID](https://openalex.org/A5049659586)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Code‑Space Response Oracles (CSRO)，利用大语言模型（LLM）生成可解释的代码策略，替代传统深度 RL 的 best‑response oracle，形成一种新的多智能体游戏近似均衡求解框架。

**💡 创新点**

创新点包括：①将深度 RL oracle 换成 LLM 代码生成器；②引入迭代精炼循环与上下文抽象来提升策略质量与可扩展性；③产生可读、可审计的源代码策略，从而实现策略可解释性；④在实验中与成熟基线（PSRO‑IMPALA、LLM‑baseline）进行严格对比，验证代码生成 oracle 的竞争力。

**🔧 技术方法**

使用的技术主要是 Gemini 2.5 Pro LLM 进行程序合成、PSRO 框架、AlphaEvolve 与 LinearRefinement 迭代精炼、Meta‑Game equilibrium 求解，以及 OpenSpiel 环境的实现。

**📊 数据集**

实验数据集包括 Repeated Rock‑Paper‑Scissors（RRPS）与 Repeated Leduc Hold’em，RRPS 评估 43 只手工策略，Leduc 评估 CFR+ Nash、AlwaysCall 与 AlwaysFold 等对手。

**📈 对比分析**

通过与 PSRO‑IMPALA（深度 RL）、Gemma 3 27B LLM 及其他基线（Q‑learning、ContRM）在 Population Return、Population Exploitability 与 Aggregate Score 三个指标上进行对比。结果显示 CSRO‑AlphaEvolve 与 LinearRefinement 在 RRPS 上可达或接近 Gemma 3 的 Aggregate Score，在 Leduc 上取得与 CFR+ 相当甚至更低的 exploitability，同时 LLM 调用次数仅随迭代线性增长，显著降低计算成本。

**⚠️ 局限性**

限制包括：依赖 LLM 的能力与 prompt 设计，可能导致代码错误或性能波动；LLM API 调用成本高；在观测空间极大或状态复杂的游戏中，难以在上下文长度内完整表达策略；缺乏针对大规模、实时游戏的可扩展性验证。

---

## 342. HyPER-GAN: Hybrid Patch-Based Image-to-Image Translation for Real-Time Photorealism Enhancement

**arXiv ID:** 2603.10604 | [PDF](https://arxiv.org/pdf/2603.10604v1)

**作者:** Stefanos Pasios `[一作]` (Aristotle University of Thessaloniki), Nikos Nikolaidis `[通讯]` (Aristotle University of Thessaloniki)

**通讯引用:** 5566 | [OpenAlex ID](https://openalex.org/A5034879808)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HyPER-GAN，一种轻量级配对图像到图像（Im2Im）网络，用于实时提升合成图像的真实感。

**💡 创新点**

创新点在于引入混合训练策略，将真实世界图像补丁与配对样本共同监督，显著降低伪影并提升语义一致性。

**🔧 技术方法**

使用 U‑Net 风格生成器、PatchGAN 判别器、FAISS 最近邻搜索、Least‑Squares GAN 与 L1 重建损失。

**📊 数据集**

采用 GTA‑V 的 PFD 数据集（合成图像及其 EPE 增强对）、Cityscapes、Mapillary Vistas 等真实图像。

**📈 对比分析**

通过 KID、mIoU、FPS 等指标与 FastCUT、REGEN、COSMOS Transfer1 等方法对比，HyPER-GAN 在视觉真实度和语义一致性上更优，同时在 1080p 下实现 33 FPS，显著低于其他模型。

**⚠️ 局限性**

局限在于主要针对城市街景，扩展到室内或多样环境仍需验证；混合训练需要构建和维护高质量的真实图像补丁索引，增加预处理成本。

---

## 343. Performance Evaluation of Delay Tolerant Network Protocols to Improve Nepal Earthquake Rescue Communications

**arXiv ID:** 2603.10153 | [PDF](https://arxiv.org/pdf/2603.10153v1)

**作者:** Xiaofei Liu `[一作]` (University of Nottingham), Milena Radenkovic `[通讯]` (University of Nottingham)

**通讯引用:** 1670 | [OpenAlex ID](https://openalex.org/A5018791184)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在尼泊尔加德满都地震救援场景下，构建了多层节点模型，评估了 Epidemic 与 Spray‑and‑Wait DTN 路由协议的性能。

**💡 创新点**

创新点在于基于地震后动态人口分布与救援活动特征的仿真框架，结合多类别边缘节点和双接口模型，系统比较两种经典协议的可靠性、延迟和资源消耗。

**🔧 技术方法**

使用 DTN、Epidemic、Spray‑and‑Wait 路由协议，采用多接口 Bluetooth 与高速通信，仿真环境包括节点缓冲区调节、SOS 信号广播和移动模型。

**📊 数据集**

使用的“数据集”是基于2015年尼泊尔地震灾情的虚拟场景生成的节点分布、移动轨迹与 SOS 事件日志，参数取自真实地震影响区域。

**📈 对比分析**

通过多指标（交付概率、平均延迟、跳数、缓冲时间、传输开销）比较两协议，结果显示 Spray‑and‑Wait 在缓冲区10M 时交付率94.6%，而 Epidemic 仅达15%且产生数万无效转发。

**⚠️ 局限性**

局限性包括：仿真场景仍是简化的二维地图，未考虑多波段信号衰减、能耗和真实硬件限制；仅比较了两种经典协议，缺少新型自适应路由或能耗优化方案。

---

## 344. Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs

**arXiv ID:** 2603.10011 | [PDF](https://arxiv.org/pdf/2603.10011v1)

**作者:** Anna Soligo `[一作]` (Imperial College London), William Saunders `[通讯]` (Anthropic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化大型语言模型的情绪失稳表现，构建多情境评估框架，发现Gemma和Gemini在后训练中易产生挫败情绪，并通过仅280对偏好数据的直接偏好优化（DPO）微调显著抑制此情绪。

**💡 创新点**

①提出针对多轮对话的情绪失稳评估方法；②揭示后训练阶段对Gemma情绪失稳的放大效应；③证明极少量偏好数据即可通过DPO大幅降低情绪失稳，且不牺牲模型能力。

**🔧 技术方法**

采用多轮对话评估、0–10挫败度量表、Claude‑Sonnet‑4 评分、LoRA 微调、DPO 与 SFT 对比、内部情绪激活分析。

**📊 数据集**

使用数字谜题、文本问答、WildChat、Petri 等情绪诱导数据；微调数据由280条高情绪与对应平静回答对构成；基准数据包括 AIME、MATH、GPQA、BBH、TruthfulQA、EmoBench。

**📈 对比分析**

在4000条回复的跨模型评估中，Gemma 在多轮对话中高情绪比例 >70%；DPO 微调后该比例降至 0.3%，并在所有评估条件及 Petri 开放式情绪诱导实验中保持低水平；与 SFT 对比，DPO 效果显著；在标准算数、推理与情绪理解基准上无性能下降。

**⚠️ 局限性**

缺乏后训练细节导致无法精确定位情绪失稳源；评估场景有限，可能漏掉其他失稳模式；实验仅针对 Gemma，未对其他模型或闭源 Gemini 基模型进行干预；情绪抑制并不一定等同于内部状态的消除，可能存在“隐藏情绪”。

---

## 345. Contract And Conquer: How to Provably Compute Adversarial Examples for a Black-Box Model?

**arXiv ID:** 2603.10689 | [PDF](https://arxiv.org/pdf/2603.10689v1)

**作者:** Anna Chistyakova `[一作]` (Trusted AI Research Center, RAS), Mikhail Pautov `[通讯]` (AXXX)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为Contract and Conquer（CAC）的黑盒对抗攻击框架；

**💡 创新点**

创新点在于将知识蒸馏与可收缩搜索空间相结合，理论上在有限迭代内可证明攻击成功；

**🔧 技术方法**

采用知识蒸馏构建代理模型、白盒攻击（如MI-FGSM）以及搜索空间收缩技术；

**📊 数据集**

在CIFAR‑10和ImageNet数据集上评估，目标模型为ResNet‑50与Vision Transformer；

**📈 对比分析**

与HopSkipJump、Sign‑OPT、GeoDA、SquareAttack等基线方法相比，CAC在l∞和l2范数下均实现更高攻击成功率、更小扰动；

**⚠️ 局限性**

受限于代理模型的可训练性与每轮都能产生对抗样本的假设，且对查询次数和模型兼容性敏感。

---

## 346. MCP-in-SoS: Risk assessment framework for open-source MCP servers

**arXiv ID:** 2603.10194 | [PDF](https://arxiv.org/pdf/2603.10194v1)

**作者:** Pratyay Kumar `[一作]` (New Mexico State University), Abu Saleh Md Tayeen `[通讯]` (University of Hartford)

**通讯引用:** 68 | [OpenAlex ID](https://openalex.org/A5046678380)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 222 个开源 Python MCP 服务器进行静态代码分析，构建 MCP‑in‑SoS 管道，系统评估并量化其安全弱点。

**💡 创新点**

提出了结合 CWE‑CAPEC 元数据的风险评分模型，并引入四层威胁表面（协议、工具、资源、提示）及其共现分析，揭示多阶段攻击链。

**🔧 技术方法**

使用 CodeQL、Joern 与 Cisco AI Defender 的静态分析器，结合 LLM（Claude Sonnet 4.5）辅助查询生成与映射。

**📊 数据集**

基于 GitHub 搜索得到的 222 个 MCP 服务器仓库，涵盖 51 种 CWE，覆盖 15,962 条弱点记录。

**📈 对比分析**

对比分析显示 86% 的仓库至少存在一项可映射弱点；高危 CWE（如 CWE‑89、CWE‑863）占比高，绝大多数仓库处于中高风险区，整体风险评分表现出与发现量和高危弱点分布相关的分布。

**⚠️ 局限性**

仅聚焦 Python 服务器；风险评分依赖 CAPEC 的统计字段，实际危害程度可能与模型估值偏差；未考虑动态运行时或多语言实现的差异。

---

## 347. UniStitch: Unifying Semantic and Geometric Features for Image Stitching

**arXiv ID:** 2603.10568 | [PDF](https://arxiv.org/pdf/2603.10568v1)

**作者:** Yuan Mei `[一作]` (Polytechnic University), Bin Xiao `[通讯]` (Chongqing University of Posts and Telecommunications)

**通讯引用:** 19830 | [OpenAlex ID](https://openalex.org/A5103218891)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出UniStitch模型，将语义特征与几何特征统一融合，实现更鲁棒的图像拼接

**💡 创新点**

首次在同一框架中将稀疏几何点与稠密语义图统一表示，采用神经点变换与自适应专家混合模块实现跨模态协同

**🔧 技术方法**

Neural Point Transformer、Adaptive Mixture of Experts (AMoE)、Latent‑Space Modality Robustifier、FFD‑based TPS 以及基于ResNet‑18的语义分支

**📊 数据集**

UDIS‑D 数据集（训练/测试）以及经典拼接数据集（APAP、SPW、LPC 等）

**📈 对比分析**

与传统几何方法（APAP、SPW、LPC）和学习型语义方法（UDIS、UDIS++、StabStitch++、RopStitch）对比，UniStitch 在 UDIS‑D 和外域数据上均取得最高 mPSNR/mSSIM，提升约 1–2 dB

**⚠️ 局限性**

对几何点质量依赖较大，若几何点或描述子不佳会影响性能；极端低纹理或极端光照环境仍需进一步提升鲁棒性

---

## 348. FRIEND: Federated Learning for Joint Optimization of multi-RIS Configuration and Eavesdropper Intelligent Detection in B5G Networks

**arXiv ID:** 2603.10977 | [PDF](https://arxiv.org/pdf/2603.10977v1)

**作者:** Maria Lamprini A. Bartsioka `[一作]` (National Technical University of Athens), Iakovos S. Venieris `[通讯]` (National Technical University of Athens)

**通讯引用:** 1938 | [OpenAlex ID](https://openalex.org/A5077094412)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于联邦学习的 RIS 辅助无基站（Cell-Free）毫米波网络中恶意窃听检测框架。

**💡 创新点**

创新点在于将联邦学习与多 RIS 配置协同优化相结合，实现了分布式隐私保护检测，并通过早期退出机制降低算力负荷；与传统单机或无 RIS 方法相比，系统在秘密率上提升约 30%。

**🔧 技术方法**

采用的技术包括联邦平均（FedAvg）联邦学习、深度卷积神经网络（DCNN）以及 RIS 阶段矩阵优化、早期退出机制。

**📊 数据集**

使用 MATLAB 生成的模拟 CSI 图像数据集，包含 500 设备（70% 正常用户、30% 窃听者），分布在 18 个 AP、3 个 RIS 的工业 B5G 环境中。

**📈 对比分析**

通过与非 RIS 辅助的基线方法对比，使用准确率、召回率、F1 分数及平均秘密率（ASR）等指标评估；模型在 80–93% 的准确率、>95% 的召回率与 18–20 bps/Hz 的 ASR 提升方面表现优异。

**⚠️ 局限性**

局限性包括仅在仿真数据上验证，未使用真实通道；RIS 配置仅为有限个预设相位，未探究更大规模或自适应配置；联邦学习的通信开销与收敛速度在大规模网络中仍待评估。

---

## 349. 6ABOS: An Open-Source Atmospheric Correction Framework for the EnMAP Hyperspectral Mission Based on 6S

**arXiv ID:** 2603.10856 | [PDF](https://arxiv.org/pdf/2603.10856v1)

**作者:** Gabriel Caballero Cañas `[一作]` (Universitat de Valencia), José Moreno `[通讯]` (Universitat de Valencia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了6ABOS，一个基于6S的EnMAP大气校正框架。

**💡 创新点**

创新点在于将6S物理模型与自动化元数据解析、GEE气象参数集成，提供可复现的端到端AC流程。

**🔧 技术方法**

使用6S辐射传输模型、Py6S接口、Google Earth Engine API、Python并行计算。

**📊 数据集**

使用EnMAP Level‑1C数据，验证样本来自西班牙马德里内陆水库Benagéber与Bellús。

**📈 对比分析**

通过与现场R_rs、RMSE、SAM对比，SAM<10°且RMSE低于0.02，表现优于传统AC。

**⚠️ 局限性**

局限在于对极端气溶胶事件、邻接效应及高光谱吸收边缘处理不足。

---

## 350. StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References

**arXiv ID:** 2603.10354 | [PDF](https://arxiv.org/pdf/2603.10354v1)

**作者:** Boyu He `[一作]` (National University of Defense Technology), Zhiping Cai `[通讯]` (National University of Defense Technology)

**通讯引用:** 8109 | [OpenAlex ID](https://openalex.org/A5006334685)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出StyleGallery框架，利用任意风格图像实现无训练的语义感知图像风格迁移；

**💡 创新点**

通过在扩散特征空间自适应聚类实现语义区域分割，结合三维度匹配和区域风格损失，首次实现多参考语义对齐且保持内容完整；

**🔧 技术方法**

采用Stable Diffusion 1.5的潜在扩散模型、DIFT提取中间特征、K-means聚类、DINOv2语义特征、注意力掩码、区域风格损失与全局内容损失的能量引导采样；

**📊 数据集**

构建自定义基准，涵盖25个风格族（来自COCO、FFHQ、WikiArt等），每族4–17幅风格图，生成750幅风格化图像；

**📈 对比分析**

与CNN、Transformer及扩散基线（CAST、CCPL、AdaAttn、StyTr‑2、CSGO、StyleShot、StyleID、AD、AttnST等）在Style、Gram、FID、LPIPS、ArtFID等指标上对比，StyleGallery取得Style 0.5337、FID 16.89、LPIPS 0.3716，整体性能优于所有对照方法；

**⚠️ 局限性**

局部聚类产生的错误掩码可能导致部分区域风格失效，对极其抽象或语义模糊的风格图不够鲁棒，需要更精细的聚类或交互式修正。

---

## 351. OnFly: Onboard Zero-Shot Aerial Vision-Language Navigation toward Safety and Efficiency

**arXiv ID:** 2603.10682 | [PDF](https://arxiv.org/pdf/2603.10682v1)

**作者:** Guiyong Zheng `[一作]` (Sun Yat-Sen University), Boyu Zhou `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2927 | [OpenAlex ID](https://openalex.org/A5101982552)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 OnFly，一种全在机实时零射AVLN系统，使无人机能够在未知3D环境中跟随自然语言指令完成导航并安全停机。

**💡 创新点**

创新点包括：共享感知的双代理架构拆分高频目标生成与低频进度监测；混合关键帧-最近帧记忆实现长时序进度监测与 KV 缓存稳定；语义几何验证器+回溯规划实现指令一致且几何安全的轨迹规划。

**🔧 技术方法**

技术手段包括 Vision‑Language 模型（Qwen3‑VL）、ViT 感知、KV 缓存、混合关键帧记忆、语义特征相似度、深度门限、ESDF 地图、Fast Planner、Jetson Orin NX 量化加速、CUDA Graphs 等。

**📊 数据集**

使用的数据集为 10 个 Unreal Engine 4.27 高保真场景，共 150 个任务（对象导航、精确导航、长程导航），并在真实世界上进行现场飞行实验。

**📈 对比分析**

与 TypeFly、PIVOT、SPF 等零射AVLN基线对比，OnFly 在模拟中 SR 67.8%（vs 26.4% SPF）、OSR 78.1%（vs 61.5% SPF）、CR 2.7%（vs 42.7% SPF），飞行时间 27.1 s（vs 39.2 s SPF），在真实世界实验中亦保持高成功率与低碰撞率。

**⚠️ 局限性**

局限性在于对高质量 VLM 与深度感知的依赖，极端光照或噪声下深度不稳定；双代理架构对时间同步要求高；在极长时序或极大规模开放世界环境下仍需进一步验证。

---

## 352. ScanDP: Generalizable 3D Scanning with Diffusion Policy

**arXiv ID:** 2603.10390 | [PDF](https://arxiv.org/pdf/2603.10390v1)

**作者:** Itsuki Hirako `[一作]` (Institute of Industrial Science), Takeshi Oishi `[通讯]` (Institute of Industrial Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于扩散策略（Diffusion Policy）的自动化3D扫描框架ScanDP，能在仅用少量人类示范数据的情况下，对未知物体实现高效、精确的全景扫描。

**💡 创新点**

创新点包括：① 将占据网格映射（Occupancy Grid Mapping）作为观察输入，提升对噪声和姿态变化的鲁棒性；② 结合最大空球（bubble）表示与路径优化，实现碰撞安全且路径更短；③ 通过稀疏卷积提取OGM特征，避免传统体素卷积的计算瓶颈。

**🔧 技术方法**

采用扩散模型进行动作生成，稀疏卷积网络用于OGM特征编码，bubble碰撞滤波器和动态规划用于路径优化；训练数据仅来自Stanford Bunny的5条示范轨迹。

**📊 数据集**

训练集：Stanford Bunny（5条扫描轨迹）；评估集：多种未见物体（Armadillo、Dragon、Spot等）以及不同尺度和不同视场的物体；同时在真实机器人平台上进行验证。

**📈 对比分析**

与随机、半球、Uniform Hemisphere、Diffusion Policy (DP)、3D Diffusion Policy (DP3) 进行比较。ScanDP在覆盖率上平均提升至约97%（比DP3高~6%），路径长度缩短约32%，对噪声和视场变化具有更强鲁棒性；在真实环境下，覆盖率达95%±2%，远优于DP3的33%±10%。

**⚠️ 局限性**

局限性：1）OGM特征提取受网格尺寸限制，难以扩展到更大场景；2）训练数据基于人类示范，需进一步域自适应以匹配机器人运动学；3）未针对多物体或大规模扫描进行验证。

---

## 353. Reason and Verify: A Framework for Faithful Retrieval-Augmented Generation

**arXiv ID:** 2603.10143 | [PDF](https://arxiv.org/pdf/2603.10143v1)

**作者:** Eeham Khan `[一作]`, Marc Queudot `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向生物医学领域的检索增强生成（RAG）框架，加入神经查询重写、BGE交叉编码重排序、基于证据的解释生成与多级事实性验证，并在推理过程中实现可解释性与纠错。

**💡 创新点**

创新点包括：① 将检索、重排序、解释生成和验证集成为统一模块化流程；② 采用8类细粒度验证词表以区分显式与隐式支持；③ 通过动态示例检索实现可扩展的少样本学习；④ 在有限token预算下评估各子模块对性能的贡献。

**🔧 技术方法**

技术实现主要基于：BM25检索、BGE交叉编码重排序、GPT‑4o进行查询重写与解释验证、Llama‑3‑8B‑Instruct负责答案与解释生成、动态k‑NN示例选择、句子级分块与标签化的自动评估。

**📊 数据集**

使用了PubMed抽象语料库作为知识库，并在BioASQ（二分类）和PubMedQA（三分类）两个生物医学问答数据集上进行评估。

**📈 对比分析**

与vanilla RAG、MedRAG+GPT‑4等基线进行对比。实验结果显示：在BioASQ‑Y/N上达到89.1%准确率，在PubMedQA上实现73.0%准确率，分别逼近或超过更大模型（如GPT‑4）或更复杂检索方案；重排序与动态示例选择显著提升性能，尤其在少样本场景。

**⚠️ 局限性**

局限性包括：仅评估英文生物医学数据集，缺乏临床部署验证；依赖OpenAI API导致延迟与成本；人类评估样本极少，缺乏统计显著性检验；在多跳或因果推理等更复杂任务上未进行验证。

---

## 354. PoultryLeX-Net: Domain-Adaptive Dual-Stream Transformer Architecture for Large-Scale Poultry Stakeholder Modeling

**arXiv ID:** 2603.09991 | [PDF](https://arxiv.org/pdf/2603.09991v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 355. CEI: A Benchmark for Evaluating Pragmatic Reasoning in Language Models

**arXiv ID:** 2603.09993 | [PDF](https://arxiv.org/pdf/2603.09993v1)

**作者:** Jon Chun `[一作]` (Kenyon), Godwin Idowu `[通讯]` (Kenyon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了Contextual Emotional Inference（CEI）基准，包含300个人工生成的情境，评估大型语言模型（LLM）在不同社交情境与权力关系下的语用情感推断能力。

**💡 创新点**

创新点在于：①同时覆盖五种语用子类型（讽刺/反讽、混合信号、战略礼貌、被动攻击、转移话题）与三种权力关系；②每个情境有三位独立注释者并记录详细注释，体现低一致性本身的诊断价值；③构建四级质量控制管线（schema校验、统计一致性、agreement分析与专家仲裁），保证注释质量。

**🔧 技术方法**

使用人工注释、统计异常检测、交叉验证、专家仲裁等质量控制技术；评估阶段采用7种LLM（4商业API、3开源权重），在零样、链式思考（CoT）与few-shot三种提示模式下预测Plutchik情感类别，并对OOV情感标签进行词典映射。

**📊 数据集**

数据集为300个人工设计的场景，每个场景包含上下文、说话者/听者角色、三种权力关系、歧义性发话；每个场景由三名注释者标注Plutchik情感类别与VAD维度，形成900条注释记录。

**📈 对比分析**

评估方法以多数投票的人类注释为黄金标准，比较各模型的准确率、宏F1及子类型细分；模型平均准确率约25%（最高25%），人类多数一致率54%；链式思考或few-shot提示并未显著提升模型性能，显示语用推断仍是显著挑战。

**⚠️ 局限性**

局限性包括：情境为人工构造，缺乏自然对话；注释者仅来自单一机构的本科生，样本规模有限；低一致性虽反映任务难度，但也可能受注释者经验影响；未覆盖多语言、多文化场景，导致跨语境推广受限。

---

## 356. Splat2Real: Novel-view Scaling for Physical AI with 3D Gaussian Splatting

**arXiv ID:** 2603.10638 | [PDF](https://arxiv.org/pdf/2603.10638v1)

**作者:** Hansol Lim `[一作]` (State University of New York), Jongseong Brad Choi `[通讯]` (State University of New York)

**通讯引用:** 482 | [OpenAlex ID](https://openalex.org/A5061176269)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于数字孪生教师的单目深度预训练框架Splat2Real，并针对视角扩充设计CN-Coverage视点选择策略与GOL guardrail混合，提升在视点偏移场景下的鲁棒性。

**💡 创新点**

创新点在于：①将视角覆盖度与新颖度结合的贪婪采样策略CN-Coverage，克服纯覆盖导致的过拟合与高新颖度误差；②使用Gaussian Observation Layer（GOL）进行教师质量门控和观察混合，防止低质量3DGS教师带来的退化；③在单目深度学习中引入“教师-学生”仿真监督而非传统控制行为学习。

**🔧 技术方法**

使用的技术包括：3D Gaussian Splatting（3DGS）渲染生成高通量RGB教师；基于Mesh的真实度量深度与可见性标签；逆深度损失、光滑项与时序一致性约束；CN-Coverage贪婪覆盖+新颖度评分；GOL质量门控与混合策略；以及传统深度网络DepthUNet和AdamW优化器。

**📊 数据集**

使用的公开数据集为20序列TUM RGB-D室内数据集（Freiburg-1与Freiburg-3），对每个序列进行固定的训练/验证/测试划分。

**📈 对比分析**

与随机、机器人、Coverage等基线在不同新增视角预算（0、25、50、100、200、500、1000、2000）下进行step-matched实验；CN-Coverage与GOL-Gated CN-Coverage在中高预算（≥200）下表现最稳健，平均AbsRel误差下降至约0.32–0.35，并在高新颖度尾部误差（最高新颖度分位）显著低于其他策略；在控制代理的安全/进度代理测试中，GOL-Gated CN-Coverage的成功率提升、碰撞率下降，显示其对下游任务的正面影响。

**⚠️ 局限性**

局限性包括：仅在静态室内场景下验证，未覆盖复杂动态或大范围视点漂移；视角预算超过500仍采用重采样，未实现真正的多样化新视角；教师质量门控依赖于渲染统计，可能在不同数据集或渲染器差异下失效；方法对3DGS构建精度敏感，精细度不足时可能影响监督质量。

---

## 357. Beyond Scalars: Evaluating and Understanding LLM Reasoning via Geometric Progress and Stability

**arXiv ID:** 2603.10384 | [PDF](https://arxiv.org/pdf/2603.10384v1)

**作者:** Xinyan Jiang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Lijie Hu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5067496051)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TRACED 框架，通过分析大语言模型（LLM）推理轨迹的几何动力学（位移进展与曲率稳定性）来评估推理质量，构建贝叶斯概率判别器并与传统概率/隐藏状态方法对比。

**💡 创新点**

创新点包括：①用几何轨迹特征取代标量概率，理论上证明进展=位移、稳定性=曲率；②构造“推理质量空间”来提取区分高低质量推理的低维子空间；③将几何特征映射到认知状态（犹豫环、置信累积），实现对模型思维过程的物理解释；④通过几何对齐实现跨任务/跨模型的通用评估。

**🔧 技术方法**

技术：1) 语义白化（利用 W_U^T W_U 诱导度量）处理隐藏状态；2) 计算离散位移 Δz 和加速度 a，进而得到平均位移 M 与平均曲率 K；3) 通过对比协方差矩阵构建质量空间基 B；4) 使用二维高斯贝叶斯模型进行无阈值推理质量判别；5) 进行子空间维度与对齐、数据效率分析。

**📊 数据集**

使用六个多样化基准：结构推理（GSM8K、MATH、TheoremQA、GPQA）和开放式推理（Social IQA、Understanding Fables）。在四个大型模型（DeepSeek‑R1‑Llama‑8B、Qwen3‑4B‑Thinking‑2507、Llama‑3.1‑8B‑Instruct、Qwen2.5‑7B‑Instruct）上评估。

**📈 对比分析**

与 MSP、Perplexity、Entropy 等概率方法、LR Probe、SAPLMA 等隐藏状态分类器、CoE、CoT‑Kinetics 等轨迹方法对比，TRACED 在结构推理任务中平均 AUROC 约 0.71‑0.76、AUPR 约 0.66‑0.80，往往达到或超过基线；在开放式推理任务中同样表现突出，说明几何特征对多任务、跨模型具有更好的鲁棒性和可迁移性。

**⚠️ 局限性**

局限性：①需要模型提供每步隐藏状态并依赖 W_U 诱导度量；②对极长推理或高维隐状态的计算开销相对较大；③在分布偏移显著时需要对齐步骤；④对某些模型（如极低参数或无自回归隐藏层）不适用；⑤实验主要集中在英文/算术/推理任务，跨语言、跨任务的广泛验证尚待进一步探索。

---

## 358. Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning

**arXiv ID:** 2603.10535 | [PDF](https://arxiv.org/pdf/2603.10535v1)

**作者:** Zichao Li `[一作]` (Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences), Xing Yu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种通过多组相对奖励重标定（Group Relative Reward Rescaling, GR^3）来抑制强化学习训练中大型语言模型的长度膨胀问题。

**💡 创新点**

创新点在于将长度控制从加性惩罚转为乘性重标定，结合组内归一化的奖励和优势感知校准，实现无损失、动态可调且适用于连续奖励的长度约束。

**🔧 技术方法**

主要技术包括多组相对奖励重标定（GR^3）、组内优势归一化、组相对长度正则化、优势感知校准，以及对 RLHF 与 RLVR 的统一改进。

**📊 数据集**

实验使用 DeepSeek-R1-Distill-1.5B、DeepSeek-R1-Distill-7B、Qwen3-4B/8B 作为基础模型，任务数据包括 DeepScaleR-Preview（数学推理）、DeepDistill（代码生成）以及 Arena-Human-Preference-140k（聊天对齐）。

**📈 对比分析**

与传统 GRPO、阈值截断、LC-R1、Laser、AdaptThink、DLER 等基线相比，GR^3 在保持或提升准确率的同时显著减少 token 数量（如 AIME24 上从 13,213 降至 7,923，提升 60.1%），并在 RLHF 对齐任务中几乎不增加长度的情况下提升对齐分数。

**⚠️ 局限性**

局限性包括对惩罚系数 α 的敏感性、在极难任务上可能仍需更长推理、以及在某些安全关键场景中过度压缩可能导致推理不完整。

---

## 359. A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment

**arXiv ID:** 2603.10034 | [PDF](https://arxiv.org/pdf/2603.10034v1)

**作者:** Jiyue Jiang `[一作]` (Chinese University of Hong Kong), Chuan Wu `[通讯]` (University of Hong Kong)

**通讯引用:** 11065 | [OpenAlex ID](https://openalex.org/A5012597518)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了基于原则驱动的自适应策略GCSD，用于多方认知刺激对话，构建了500小时真实粤语对话和10,000+模拟对话数据集，并通过四模块系统提升多方对话一致性、个性化认知状态建模、认知刺激注意力损失及多维奖励优化。

**💡 创新点**

创新点包括：①结合真实与原则引导的模拟数据提升对话多样性；②引入多说话人上下文控制；③实现动态参与者认知状态软提示；④设计认知刺激关注注意力损失；⑤采用多维奖励策略实现对话原则遵循。

**🔧 技术方法**

使用大语言模型Qwen-2.5-3b，结合SFT、认知刺激关注注意力损失（CSFAL）、动态软提示网络、以及多维奖励策略的MRPO等技术。

**📊 数据集**

使用了500小时真实粤语CST会话数据集以及10,000+原则引导模拟对话数据集（PGSS）。

**📈 对比分析**

与ERNIE、Doubao-Pro、GPT‑4o、LLaMA‑3.1‑405B等强基线进行自动评估（ROUGE‑L、BLEU‑4、BERTScore）与人工评估（相关性、共情、流畅度），GCSD在BLEU‑4、BERTScore、相关性等指标上均领先，人工A/B测试胜率最高。

**⚠️ 局限性**

目前缺乏长期临床验证，模型可能产生幻觉，仅限文本交互，缺乏多模态支持与完善的安全机制。

---

## 360. Pointy - A Lightweight Transformer for Point Cloud Foundation Models

**arXiv ID:** 2603.10963 | [PDF](https://arxiv.org/pdf/2603.10963v1)

**作者:** Konrad Szafer `[一作]`, Dominik Belter `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出轻量级Transformer骨干网络 Pointy，用于点云处理，并在仅39k训练样本的前训练设置下实现优于更大规模的基础模型的表现。

**💡 创新点**

创新点在于：①完全无Tokenizer的直接点坐标编码；②层级Transformer与Token合并实现局部-全局特征提炼；③统一训练与评测框架，系统性复现多种架构，展示数据量与架构设计的相互影响。

**🔧 技术方法**

使用Transformer、PointNet式嵌入、FPS+ kNN分块、层级Token合并、GeLU激活、AdamW优化等技术。

**📊 数据集**

数据集包括ModelNet40、ScanObjectNN、以及约39k样本的 Objaverse‑LVIS 子集；还做了零样本评估迁移到 ModelNet40/ScanObjectNN。

**📈 对比分析**

通过统一训练设定、相同预处理、批量大小、学习率等，对比多种经典与Transformer模型，Pointy 在 ModelNet40/ScanObjectNN 的准确率分别达到 90.6%/80.0%，在零样本迁移上接近甚至超过大规模多模态模型，证明在少量数据下架构与训练更关键。

**⚠️ 局限性**

局限在于仅使用分类目标进行预训练，缺乏对更细粒度任务（如分割、密集预测）的能力，且预训练数据相对干净，需进一步验证对噪声更大真实扫描数据的泛化。

---

## 361. Detecting Privilege Escalation with Temporal Braid Groups

**arXiv ID:** 2603.10094 | [PDF](https://arxiv.org/pdf/2603.10094v1)

**作者:** Christophe Parisel `[一作]` `[通讯]`, Christophe Parisel

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了云权限图中强连通子图的时序演化，提出使用 Burau Lyapunov 指数作为检测风险边界并指导权限修复的非阿贝尔方法。

**💡 创新点**

创新点在于证明任何阿贝尔统计量（如门控计数、WAR 总和）无法确定聚焦/分散风险区分，并提出 Burau Lyapunov 指数作为唯一可行的分界器，同时给出两种风险模式的操作化分类和对应的修复路径。

**🔧 技术方法**

采用 Burau 表示、随机游走生成 braid 词、引入 σ_i^2σ_{i+1}^{-1} 注入词、计算 Lyapunov 指数，并使用门控计数、WAR 总和等阿贝尔统计量做对照。

**📊 数据集**

在合成数据集上验证：1,000 个随机生成的 6 节点强连通子图，共 49,972 个 (SCC, WAR) 对。

**📈 对比分析**

与传统阿贝尔门限指标（firing‑rate）比较，Burau LE 在 5.7% 的样本上产生不同分类，且在控制 firing‑rate 后仍保持 r≈0.175 的残余相关性，而计数 LE 则 r≈0，显示 Burau 的非阿贝尔优势显著。

**⚠️ 局限性**

局限包括：Burau 表示在 n≥5 时不忠实，仅给出下界；验证仅基于合成数据，未在真实云 IAM、Kubernetes RBAC 等环境中测试；非阿贝尔信号幅度小（约 3% 方差）；整数运算可能溢出，并且对大规模 SCC 的计算成本尚未评估。

---

## 362. Separating Oblivious and Adaptive Differential Privacy under Continual Observation

**arXiv ID:** 2603.11029 | [PDF](https://arxiv.org/pdf/2603.11029v1)

**作者:** Mark Bun `[一作]`, Connor Wagaman `[通讯]` (Boston University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在连续观测模型中，区分了可忽略（oblivious）与自适应（adaptive）差分隐私的能力差异，给出了一个能够在可忽略设置下实现指数级时间步准确输出的算法，但在自适应设置下只能保持常数时间步准确性的任务。

**💡 创新点**

首次在连续观测模型中提供了可忽略与自适应差分隐私之间的明确分离，揭示了自适应输入对隐私保证的实质性限制。

**🔧 技术方法**

利用随机响应、Hoeffding不等式、相关向量查询问题的重建引理以及隐私游戏与总变差距离的关系进行理论分析。

**📊 数据集**

无实验数据集，所有结果均为理论证明。

**📈 对比分析**

通过构造性的上界（指数步长）与下界（常数步长）对比，证明在可忽略设置下可达极高的时间步数，而在自适应设置下几乎无法维持准确性。

**⚠️ 局限性**

局限性在于仅给出理论分离示例，缺乏对实际数据流场景的实验验证，并且分离问题相对人工构造，尚需寻找更自然的应用场景。

---

## 363. Topological Analysis for Identifying Anomalies in Serverless Platforms

**arXiv ID:** 2603.10850 | [PDF](https://arxiv.org/pdf/2603.10850v1)

**作者:** Gianluca Reali `[一作]` (University of Perugia), Mauro Femminella `[通讯]` (University of Perugia)

**通讯引用:** 1623 | [OpenAlex ID](https://openalex.org/A5053723295)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了服务器无状态函数间信息流的拓扑特性，提出通过Hodge分解识别并纠正服务中的循环和低效模式；

**💡 创新点**

创新点在于引入可调度的边权重（度量矩阵）以消除非结构性谐波成分，仅保留真正的拓扑不稳定环路；

**🔧 技术方法**

核心技术是离散Hodge分解、组合拉普拉斯算子、迭代优化度量矩阵和贝蒂数分析；

**📊 数据集**

使用了一个基于AWS Lambda的电商应用的功能调用图，包括约百个函数和多种Saga/补偿循环，并模拟冷启动引起的流量/延迟；

**📈 对比分析**

通过与标准统一权重的Hodge分解对比，展示了迭代度量学习后谐波能量显著下降、循环识别更准确，实验中收敛速度快，性能提升可视化在图表中体现；

**⚠️ 局限性**

局限在于假设拓扑不变、仅考虑二阶单元、对大规模数据中心的可扩展性未完整验证，并未处理实时动态变化的调用图。

---

## 364. Dynamic Modeling and Attitude Control of a Reaction-Wheel-Based Low-Gravity Bipedal Hopper

**arXiv ID:** 2603.10670 | [PDF](https://arxiv.org/pdf/2603.10670v1)

**作者:** Shriram Hari `[一作]` (Indian Institute of Technology Hyderabad), R Prasanth Kumar `[通讯]` (Indian Institute of Technology Hyderabad)

**通讯引用:** 461 | [OpenAlex ID](https://openalex.org/A5044837841)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并仿真了一个在低重力环境下利用内部转子调节姿态的双足跳跃机器人。

**💡 创新点**

首次将单一内部转子与简化的刚体动力学模型结合，实现飞行阶段的姿态闭环控制，兼顾机械简化与控制可行性。

**🔧 技术方法**

使用MuJoCo低重力仿真、平面刚体转子动力学耦合模型、经典PID姿态控制以及混合动力学切换策略。

**📊 数据集**

采用程序生成的月球高程图（procedurally generated heightfield）作为不规则地形数据集。

**📈 对比分析**

与无转子基线对比，转子控制将中空飞行阶段最大角度偏差降低65%以上，落地角误差≤3.5°，并显著减少了执行器饱和时间，验证了显著性能提升。

**⚠️ 局限性**

仅在俯仰平面实现姿态控制，未覆盖偏航和滚转；转子饱和仍在极端运动下可能出现；尚未在真实硬件上验证，需要进一步扩展至三维姿态与更复杂地形。

---

## 365. MAD: Memory Allocation meets Software Diversity

**arXiv ID:** 2603.10840 | [PDF](https://arxiv.org/pdf/2603.10840v1)

**作者:** Manuel Wiesinger `[一作]` (Vrije Universiteit Amsterdam), Stefan Brunthaler `[通讯]` (Universität der Bundeswehr München)

**通讯引用:** 1948 | [OpenAlex ID](https://openalex.org/A5058365944)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出Memory Allocation Diversity（MAD）技术，通过水平和垂直多样化的缓存机制混淆内存分配，抵御Rowhammer攻击。

**💡 创新点**

将软件多样化理念引入内存管理，创新性地设计水平（Block Recycling）与垂直（Buddy Merge/Inverse Merge）双重缓存，显著降低内存块枚举速度并提高攻击检测率。

**🔧 技术方法**

利用Buddy allocator接口，构建分配缓存（C_A）与影子缓存（C_S），实现水平/垂直多样化、随机阈值、随机初始化与回填，并在Python原型中实现。

**📊 数据集**

使用随机分配实验，模拟10亿次内存分配并统计唯一块数；未使用公开数据集，实验基于Python实现的MAD原型。

**📈 对比分析**

与传统Buddy allocator对比，MAD的块枚举下降率提升4.16倍，检测率达98–100%；在10亿分配实验中唯一块数几乎饱和，表明能显著延迟并检测Rowhammer攻击。

**⚠️ 局限性**

提供概率性安全，无法完全阻止攻击；依赖随机化与缓存，攻击者可能通过混淆或占用缓存规避；实验仍在原型阶段，尚未在真实系统或多行攻击下充分验证。

---

## 366. Learning to Wander: Improving the Global Image Geolocation Ability of LMMs via Actionable Reasoning

**arXiv ID:** 2603.10463 | [PDF](https://arxiv.org/pdf/2603.10463v1)

**作者:** Yushuo Zheng `[一作]` (Shanghai Jiao Tong University), Xiongkuo Min `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10283 | [OpenAlex ID](https://openalex.org/A5043405654)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了全球范围的可交互地理定位基准 WanderBench，并构建了一套能够执行旋转与移动等动作的导航图，随后设计了 GeoAoT（Action of Thought）框架，使大语言模型在进行地理定位推理时能够主动发起动作以获取更多视觉信息，从而提升定位精度并生成可解释的推理过程。

**💡 创新点**

创新点在于：①将传统静态图像定位转变为可交互、基于导航图的探索式定位；②引入动作驱动的思考机制（AoT），将文本推理直接映射为物理动作；③设计双向评价协议，既测量定位精度，又评估模型生成多难度问题与位置的能力。

**🔧 技术方法**

主要技术包括：大规模多模态模型（如 GPT‑4o、o3、Gemini‑2.5‑Pro 等）在推理阶段生成定位猜测与动作指令；基于循环的 GeoAoT 过程（猜测 → 反思 → 动作 → 新观测 → 更新）；利用预先构建的导航图进行环境交互；统计与 ANOVA 评估模型在不同覆盖率下的表现；以及对比 Chain‑of‑Thought、检索式定位等基线方法。

**📊 数据集**

使用的数据集为 WanderBench，包含 1,047 个地点、32,741 张全景图像、39,442 条可导航边，覆盖六大洲、50 多国。地点与问题均由大型多模态模型生成，保证了多样性与难度层级。

**📈 对比分析**

评估方法：在 19 种公开与闭源 LMM 上进行地理定位任务，计算街道、城市、国家级别的准确率、召回率、F1、距离误差和 GeoScore；对比传统 Chain‑of‑Thought 与检索式模型。实验结果显示，GeoAoT 能显著降低距离误差（最高可达 1,074 km 的绝对提升），提升细粒度精度；在大部分模型上，GeoAoT 的表现均优于基线，尤其在高覆盖率下保持稳定的单调提升。

**⚠️ 局限性**

局限性包括：①数据集主要由 LMM 生成，可能引入生成偏差与缺失真实世界多样性；②评价环境主要基于静态全景图，缺乏动态变化与真实探测器噪声；③对高计算成本模型（如 GPT‑4o）依赖较大，资源门槛高；④尚未在真实城市行走实验中验证主动探测的实用性。

---

## 367. A Hypergraph-Based Framework for Exploratory Business Intelligence

**arXiv ID:** 2603.10625 | [PDF](https://arxiv.org/pdf/2603.10625v1)

**作者:** Yunkai Lou `[一作]` (Alibaba Group), Ying Zhang `[通讯]` (Zhejiang Gongshang University)

**通讯引用:** 15383 | [OpenAlex ID](https://openalex.org/A5100386104)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于超图的数据模型和操作符的可探索业务智能框架，支持动态模式演化和中间结果复用。

**💡 创新点**

创新点在于将业务探索视为多轮迭代，利用超图模型、采样化子图匹配与连接操作符，并提供理论上无偏的近似查询估计。

**🔧 技术方法**

采用超图数据模型、FaSTest 子图匹配采样、连接采样、基于采样的近似查询处理以及 OLAP 算子等技术。

**📊 数据集**

使用 LDBC 社交网络基准数据集（SF0.1~SF10），涵盖数千万级顶点与边。

**📈 对比分析**

与 Neo4j、MySQL 以及 VerdictDB 等基准系统比较，平均比 Neo4j 提升 16.21×、比 MySQL 提升 46.67×，误差率仅为 0.27%。

**⚠️ 局限性**

局限在于对采样的依赖，极小结果集时可能收敛慢；对非“存在”语义的查询仍存在改进空间。

---

## 368. Fly-PRAC: Packet Recovery for Random Linear Network Coding

**arXiv ID:** 2603.10266 | [PDF](https://arxiv.org/pdf/2603.10266v1)

**作者:** Hosein K. Nazari `[一作]` (Technische Universität Dresden), Frank H. P. Fitzek `[通讯]` (Technische Universität Dresden)

**通讯引用:** 9735 | [OpenAlex ID](https://openalex.org/A5023936439)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种新的部分包恢复（PPR）方案——fprac，用于随机线性网络编码（rlnc）和稀疏网络编码（snc），通过利用线性相关的冗余编码包来估计并纠正部分损坏的数据，支持在中间节点恢复并重编码。

**💡 创新点**

创新点在于：① fprac 只需少量冗余包即可在不解码的情况下估计错误位置，显著降低假阳性概率；② 允许中间节点恢复并重编码，减少后续传输；③ 通过段级 CRC 与依赖包的组合，实现更精确的错误定位和更少的纠错尝试。

**🔧 技术方法**

技术主要包括：随机线性编码、线性相关组的高斯消元估计错误列、段级 CRC 验证与符号置换纠错、依赖包的使用、以及针对 snc 的稀疏系数向量处理。

**📊 数据集**

实验使用 Kodo 16.1.1 实现，采用 GF(2^8) 字段进行仿真，场景为不同生成大小、负载大小、错误率（10^-5~10^-3）以及 R（冗余包数）和段数的参数组合，数据为随机生成的网络编码帧。

**📈 对比分析**

与传统的 sprac 与未恢复的 rlnc 比较，fprac 在错误率为 10^-4 时，良好吞吐量提升至约 2–4 倍；在两跳通信中恢复后总传输量下降 10–16%；在 snc 中平均解码延迟下降 30–50%。实验结果通过良好吞吐量、传输次数、完成时间（CT）和平均解码延迟（ADD）等指标进行对比。

**⚠️ 局限性**

限制主要包括：① 仍需对参数 R、段数等进行经验性调优；② 当生成大小或符号数极大时，估计与纠错的计算开销提升；③ 对极高错误率或过大 R 的场景下假阳性概率虽低但仍存在；④ 目前仅在仿真环境验证，实际部署中对硬件实现与低延迟需求的兼容性尚待进一步评估。

---

## 369. Contrastive learning-based video quality assessment-jointed video vision transformer for video recognition

**arXiv ID:** 2603.10965 | [PDF](https://arxiv.org/pdf/2603.10965v1)

**作者:** Jian Sun `[一作]` (University of Denver), Mohammad H. Mahoor `[通讯]` (University of Denver)

**通讯引用:** 8730 | [OpenAlex ID](https://openalex.org/A5041948053)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种自监督学习的视频质量评估结合视频分类的模型SSL‑V3，利用无参考VQA解决视频分类中的质量依赖问题。

**💡 创新点**

创新点在于设计Combined‑SSL框架，将视频质量分数作为调节因子直接参与分类，并通过对比学习实现无标签VQA回归，形成全新的自监督多任务学习机制。

**🔧 技术方法**

技术包括ViViT视觉Transformer骨干、序列与视频质量回归网络（SSR+VSR）、Tune‑CLS权重调节、结合Batch‑和Subject‑level CBS损失以及对比学习。

**📊 数据集**

使用I‑CONECT（老年人访谈视频，MCI检测）和Hockey Fight Detection（冰球暴力检测）两大真实低质量视频数据集进行实验。

**📈 对比分析**

与多种基线（文本、视频、MC‑ViViT、无VQA/有VQA等）对比，I‑CONECT四个主题上准确率均超过88%，最高达94.87%；HF数据集平均准确率达98.6%，显著优于其它模型，表明VQA与对比学习显著提升性能。

**⚠️ 局限性**

局限在于缺乏完整多任务评估，无法使用传统VQA指标评价质量回归；VQA对高质量视频的依赖仍存在，且在小样本或极短视频上可能出现过拟合，需要进一步验证泛化能力。

---

## 370. VCR: Variance-Driven Channel Recalibration for Robust Low-Light Enhancement

**arXiv ID:** 2603.10975 | [PDF](https://arxiv.org/pdf/2603.10975v1)

**作者:** Zhixin Cheng `[一作]` (University of Science and Technology of China), Haodian Wang `[通讯]` (CHN Energy Digital Intelligence Technology Development)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于变异驱动通道再校准（VCR）的低光图像增强框架。

**💡 创新点**

创新点包括：①变异感知通道过滤（VCF）通过方差统计筛选不一致通道；②颜色分布对齐（CDA）在色度特征空间对齐分布，减少色彩偏移；③在HVI颜色空间中结合三叉通道增强（TCE）实现空间与通道互补。

**🔧 技术方法**

采用HVI颜色空间、实例归一化、卷积注意力机制、三叉通道增强、KL散度分布对齐等技术。

**📊 数据集**

在十个LLIE基准数据集上进行实验，包括LOLv1、LOLv2、SID、SICE、DICM、LIME、MEF、NPE、VV等。

**📈 对比分析**

与11种监督方法和3种无监督方法对比，VCR在PSNR、SSIM、LPIPS等指标上均位居榜首，特别是LOLv1获得28.97 dB PSNR；在未配对数据集的BRISQUE、NIQE指标上亦显著优于现有方法。

**⚠️ 局限性**

局限：对极端高ISO噪声和混合光源导致的色彩失真处理不够充分，模型在噪声误判时可能放大纹理或产生细小色偏。

---

## 371. TractoRC: A Unified Probabilistic Learning Framework for Joint Tractography Registration and Clustering

**arXiv ID:** 2603.10418 | [PDF](https://arxiv.org/pdf/2603.10418v1)

**作者:** Yijie Li `[一作]` (University of Electronic Science and Technology of China), Fan Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 53754 | [OpenAlex ID](https://openalex.org/A5100403400)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出了一个统一的概率框架 TractoRC，用于同时进行纤维束图（tractogram）的配准和纤维束的聚类。

**💡 创新点**

创新点在于把配准和聚类整合为一个共享嵌入空间的联合优化过程，并通过变换等价自监督预训练获得几何感知且变换不变的表征。

**🔧 技术方法**

采用 DGCNN 进行点云嵌入、TPS 变形配准、DCEC 聚类以及自监督的变换等价与几何一致性损失。

**📊 数据集**

在 Human Connectome Project Young Adult (HCP‑YA) 的 dMRI 纤维束数据上进行实验，使用 140 名受试者的 100,000 条随机抽样纤维束。

**📈 对比分析**

与现有基于体素的 SyN、SynthMorph 以及基于纤维束的 WMA 等方法相比，TractoRC 在 ABD（平均束距离）和 wDice（加权 Dice）上均取得显著提升，聚类方面在 α（聚类紧密度）和 WMPG（跨受试者可重复性）指标上也优于 QuickBundles、WMA、DFC 等 SOTA 方法。

**⚠️ 局限性**

局限性包括仅在 HCP‑YA 数据集上验证，尚未评估在不同扫描仪或病理数据上的泛化能力，且联合训练与自监督预训练的计算成本较高。

---

## 372. AI-Generated Rubric Interfaces: K-12 Teachers' Perceptions and Practices

**arXiv ID:** 2603.10773 | [PDF](https://arxiv.org/pdf/2603.10773v1)

**作者:** Bahare Riahi `[一作]` (North Carolina State University), Veronica Cateté `[通讯]` (North Carolina State University)

**通讯引用:** 877 | [OpenAlex ID](https://openalex.org/A5090781324)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究K‑12教师在专业发展工作坊中使用AI工具（MagicSchool.ai）生成评估标准的体验与实践；

**💡 创新点**

首次系统呈现教师对AI生成评分表的可用性、清晰度与公平性认知，并提出编辑灵活性与教师主导权的关键改进点；

**🔧 技术方法**

利用大语言模型（LLM）通过Prompt生成rubric，结合问卷调查与主题分析进行评估；

**📊 数据集**

收集了25名中高学校教师的实验数据（含前后调查与访谈记录）；

**📈 对比分析**

通过Likert量表与主题分析比较教师使用前后感知，结果显示教师对清晰度与对齐度评价积极，但对编辑易用性评价低；

**⚠️ 局限性**

受限于样本规模、缺乏真实课堂部署、对AI生成语言质量的系统性评估不足以及对教师主观偏好的考虑不足。

---

## 373. An Approach for Safe and Secure Software Protection Supported by Symbolic Execution

**arXiv ID:** 2603.10608 | [PDF](https://arxiv.org/pdf/2603.10608v1)

**作者:** Daniel Dorfmeister `[一作]` (Software Competence Center Hagenberg), Markus Zimmermann `[通讯]` (Symflower GmbH)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

将工业控制软件绑定到特定硬件，通过物理不可克隆函数（PUF）实现复制保护，并保证在非目标硬件上程序仍保持安全但执行错误

**💡 创新点**

首次将符号执行与ASM和PUF结合，用符号执行证明安全约束在硬件绑定下仍成立，从而实现安全的复制保护

**🔧 技术方法**

物理不可克隆函数（PUF）、抽象状态机（ASM）、符号执行技术

**📊 数据集**

无真实数据集，仅以单向交通灯控制算法作为示例演示方法

**📈 对比分析**

通过符号执行推导安全状态集合，未给出量化性能评估，但指出公式过长可能导致响应时间增加

**⚠️ 局限性**

需手动识别控制状态并给出安全约束；符号执行产生的公式可能过大影响效率；缺乏对大型工业案例的实测验证

---

## 374. Beyond the Prompt in Large Language Models: Comprehension, In-Context Learning, and Chain-of-Thought

**arXiv ID:** 2603.10000 | [PDF](https://arxiv.org/pdf/2603.10000v1)

**作者:** Yuling Jiao `[一作]` (Wuhan University), Defeng Sun `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8736 | [OpenAlex ID](https://openalex.org/A5029911976)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过严格的概率与信息论框架，对大型语言模型在零样本、上下文学习和链式推理三种提示策略的理论机制进行分析，并给出了对应的误差上界。

**💡 创新点**

创新点在于提出了统一的理论视角：揭示自回归训练可实现语义解码、上下文学习通过贝叶斯聚焦消除任务歧义、链式推理激活任务分解以克服组合迁移，且通过引入K‑分离、近似马尔可夫等假设得到更高阶误差下降。

**🔧 技术方法**

使用的技术包括：隐变量任务建模、贝叶斯推断、KL与TV距离分析、PAC‑Bayes误差界、Hamming距离与K‑分离假设、转移学习框架与证据偏移分析。

**📊 数据集**

论文主要基于理论假设构建的预训练分布 q(d)，未使用具体公开数据集进行实验验证。

**📈 对比分析**

通过推导零样本、ICL 与 CoT 的误差上界进行比较，CoT 的误差下降阶数为 (e^{2nϕ}·c_1)^mK，明显优于 ICL 的 (e^{2nϕ}·c·)^m，表明在多步推理任务中性能更佳。

**⚠️ 局限性**

局限性包括对任务先验平衡、K‑分离、近似马尔可夫、证据偏移等假设的高度依赖，且理论结果在实际 LLM 训练与推理中难以直接验证。

---

## 375. VoxCare: Studying Natural Communication Behaviors of Hospital Caregivers through Wearable Sensing of Egocentric Audio

**arXiv ID:** 2603.10888 | [PDF](https://arxiv.org/pdf/2603.10888v1)

**作者:** Tiantian Feng `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**通讯引用:** 30885 | [OpenAlex ID](https://openalex.org/A5010028928)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一套可在真实临床环境中使用的自我感知可穿戴音频传感系统，用于测量医疗专业人员的自然交互行为；

**💡 创新点**

通过基于语音基础模型的教师-学生蒸馏框架，在仅使用低维声学特征的条件下实现前景/背景说话者分离，并将其与可解释的交互指标（频率、时长、情绪激活）结合；

**🔧 技术方法**

主要技术包括：Android端实时音频特征提取（openSMILE）、基于Whisper的教师模型、ResNet学生模型、规则式情绪激活估计、统计学分析；

**📊 数据集**

训练数据来源于四个公开的第一人称/多人会议语音数据集（MMCSG、ICSI-Meeting、EasyCom、Internal Meeting），并在医院共计255名专业人员的10周长周期中收集原始声学特征；

**📈 对比分析**

与教师模型（Whisper-Tiny/Small/Large）和原始ResNet基线比较，蒸馏后学生模型在四个数据集上的DER分别下降约2-4%，尤其在前景/背景混杂程度高的医院环境中提升显著；

**⚠️ 局限性**

主要局限：缺乏大规模、真实医院环境的前景/背景标注数据；仅使用声学特征，缺乏语义信息，难以捕捉交流内容与任务上下文；以及对工作行为的细粒度测量不足。

---

## 376. Silent Subversion: Sensor Spoofing Attacks via Supply Chain Implants in Satellite Systems

**arXiv ID:** 2603.10388 | [PDF](https://arxiv.org/pdf/2603.10388v1)

**作者:** Jack Vanlyssel `[一作]` (University of New Mexico), Afsah Anwar `[通讯]` (University of New Mexico)

**通讯引用:** 643 | [OpenAlex ID](https://openalex.org/A5064140739)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在NASA的NOS3模拟环境中构造并演示了一个通过供应链植入的恶意模块，能够在卫星内部生成结构合法、时间正确的伪造传感器遥测，并被地面站误认为真实数据。

**💡 创新点**

首次实现了从供应链角度进行的内部传感器伪造攻击，揭示了模块化小卫星中隐蔽的供应链威胁，并指出了隐式遥测信任、缺乏运行时监控和供应链不透明等三大结构缺口。

**🔧 技术方法**

利用NASA Core Flight Software (cFS) 的软件总线接口，编写恶意cFS应用；通过模拟硬件模型（如星标追踪器）实现伪造遥测的生成与发布；结合COSMOS地面工具验证伪造数据的合法性。

**📊 数据集**

实验数据来源为NOS3仿真平台中的标准硬件模型与cFS应用；未使用实际空间遥测数据集，而是通过仿真生成的遥测包进行验证。

**📈 对比分析**

实验对比：正常星标追踪器的遥测与恶意模块生成的伪造遥测在格式、时序与消息ID上完全一致，地面站及内部消费者均无法区分；未给出数值性能指标，重点在可被接受与误导性表现上。

**⚠️ 局限性**

限制包括：仅在仿真环境下验证，未在真机或实轨测试；攻击仅针对单一传感器（星标追踪器），未证明对其他系统的适用性；假设攻击者具有供应链内部访问权限与完整的遥测接口知识，缺乏对实际威胁概率的评估。

---

## 377. Dynamic Knowledge Fusion for Multi-Domain Dialogue State Tracking

**arXiv ID:** 2603.10367 | [PDF](https://arxiv.org/pdf/2603.10367v1)

**作者:** Haoxiang Su `[一作]` (China Telecom Corp Ltd), Shuangyong Song `[通讯]` (China Telecom Corp Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种动态知识融合模型 DKF-DST，用于多域对话状态跟踪；

**💡 创新点**

创新点在于先用对比学习的编码器选取与对话相关的槽，再将所选槽的结构化知识动态注入作为提示，避免无关信息的干扰；

**🔧 技术方法**

技术包括 RoBERTa 编码器 + 对比学习、T5 seq2seq + 结构化知识提示、动态槽选择机制；

**📊 数据集**

使用 MultiWOZ 2.1‑2.4 版本数据集进行实验；

**📈 对比分析**

与多种 seq2seq 基线对比，DKF‑DST 在 MultiWOZ 2.4 上的联合目标准确率达 77.3%，显著优于 D3ST 75.9% 等同类方法；

**⚠️ 局限性**

局限性包括对阈值 δ 的敏感性、可能的误检导致的错误传播，以及对未见领域知识的依赖不足。

---

## 378. HEAL: Hindsight Entropy-Assisted Learning for Reasoning Distillation

**arXiv ID:** 2603.10359 | [PDF](https://arxiv.org/pdf/2603.10359v1)

**作者:** Wenjing Zhang `[一作]` (China Unicom), Shiguo Lian `[通讯]` (China Unicom)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 HEAL 框架，解决大模型推理蒸馏中的教师天花板问题，能够生成难题的有效推理轨迹并进行蒸馏。

**💡 创新点**

创新点在于：① Guided Entropy-Assisted Repair (GEAR) 利用熵动力学定位推理断点并提供局部回溯提示；② Perplexity‑Uncertainty Ratio Estimator (PURE) 通过困惑度与答案不确定度比值过滤逻辑短路；③ Progressive Answer‑guided Curriculum Evolution (PACE) 将蒸馏过程分为基础、潜能扩展和前沿突破三阶段，构建渐进式学习曲线。

**🔧 技术方法**

技术包括熵动力学检测、答案不确定度评估、困惑度计算、比值过滤、无 RL 的三阶段自适应训练；使用 Qwen3‑32B 作为教师模型，Qwen2.5‑14B‑Instruct 与 Qwen3‑4B‑Base 作为学生模型。

**📊 数据集**

实验使用 AIME 2024/2025、MATH 500、OlympiadBench 等高难度数学推理 benchmark；教师采样通过多轮 rejection sampling、答案提示与 GEAR 局部回溯生成数据。

**📈 对比分析**

与 SFT、LIMO、Curriculum SFT 等基线对比，HEAL 在 Qwen2.5‑14B‑Instruct 上 Pass@1 平均 61.68%，比基线提升 10–17%，在 AIME、MATH、OlympiadBench 均表现最优；在 Qwen3‑4B‑Base 上也保持显著提升。

**⚠️ 局限性**

局限性：① 需要已知答案进行回溯提示，限制了对开放式任务的适用；② 仅适用于教师模型处于 ZPD 内的任务，若缺乏必要先验知识可能失效；③ PURE 的困惑度与 NLL 计算在离线阶段增加额外算力开销。

---

## 379. Tureis: Transformer-based Unified Resilience for IoT Devices in Smart Homes

**arXiv ID:** 2603.10038 | [PDF](https://arxiv.org/pdf/2603.10038v1)

**作者:** Alireza Borhani `[一作]` (University of Maryland), Bahar Asgari `[通讯]` (Google)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种完全自监督、上下文感知的Transformer方法Tureis，用于在智能家居中检测故障并定位有缺陷的传感器；

**💡 创新点**

创新点包括：①使用短期窗口的传感器级掩码重建任务，实现无标签的故障定位；②通过迭代隔离机制逐步消除已定位的故障影响，支持多重并发故障；③采用位级特征压缩，显著降低资源占用，便于在边缘设备上部署；

**🔧 技术方法**

技术主要包括：位级特征提取、轻量级BERT式Transformer编码器、传感器级掩码重建训练目标、残差映射为故障证据、迭代隔离与继续循环；

**📊 数据集**

使用了五个公开智能家居数据集：HouseA、HH102、Tulum、Atmo1和Tokyo，覆盖单/多住户、二进制和数值传感器；

**📈 对比分析**

与三大基线（DICE、ThingsDND、Anomaly Transformer）比较，Tureis在单故障定位F1上分别提高7.6%、21.0%和25.0%；在多故障（最多5个同时故障）下提升17.6%和35.4%；定位速度约为单故障检测的10倍，且模型尺寸不到1MB，在Raspberry Pi 5上每分钟仅耗时几毫秒，峰值内存约0.5GB；

**⚠️ 局限性**

局限性在于：目前仅针对事件驱动的离散传感器，未针对连续数值型传感器的性能；对极端异常模式（如大规模同步故障）仍需进一步验证；模型在更大规模、多类型传感器环境下的可扩展性尚未充分评估。

---

## 380. OpenClaw-RL: Train Any Agent Simply by Talking

**arXiv ID:** 2603.10165 | [PDF](https://arxiv.org/pdf/2603.10165v1)

**作者:** Yinjie Wang `[一作]`, Ling Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 OpenClaw‑RL，一个统一的异步框架，利用交互的下一状态信号实现个人和通用 AI 代理的持续强化学习。

**💡 创新点**

创新点在于：①将下一状态信号同时视为评估与指令信息，构建二元 RL 与 Hindsight‑Guided On‑Policy Distillation（OPD）两种在线学习通道；②实现全异步、无阻塞的多组件架构，支持个人对话、终端、GUI、SWE 与工具调用等多种环境；③在一步奖励与整体奖励之间融合，实现长时序任务的密集信用分配。

**🔧 技术方法**

技术手段包括 PPO‑style 剪裁策略梯度、PRM（Process Reward Model）进行下一状态评估、OPD 通过提取提示构造教师上下文并产生 token‑级优势、基于 slime 的四环节异步训练框架、以及对不同任务的步长奖励标准化。

**📊 数据集**

使用的数据集包括：个人代理模拟中 GSM8K；终端、GUI、SWE、工具调用分别使用 SETA RL、OSWorld‑Verified、SWE‑Bench‑Verified、DAPO RL；此外采用 Qwen3‑4B‑SFT 预训练模型与 Qwen3‑8B、Qwen3‑32B 等大模型作为实验基础。

**📈 对比分析**

实验对比了单独的 Binary RL、OPD 与两者加权组合；组合方法在个人代理的 36 个 GSM8K 题目上从基线 0.17 提升至 0.81；在通用代理上，集成步骤奖励显著优于仅用终点奖励（如工具调用 0.30 vs 0.17），验证了两种信号的互补性和有效性。

**⚠️ 局限性**

局限性包括：需要额外部署 PRM 服务器以产生步骤奖励；OPD 仅在下一状态包含可提取指令时生效，导致样本稀疏；框架对硬件资源和延迟敏感，且在缺乏明确指令的环境中表现有限。

---

## 381. Landmark Guided 4D Facial Expression Generation

**arXiv ID:** 2603.10337 | [PDF](https://arxiv.org/pdf/2603.10337v1)

**作者:** Xin Lu `[一作]` (University of Chinese Academy of Sciences), Jun Xiao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 10159 | [OpenAlex ID](https://openalex.org/A5042106312)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于中性关键点引导的LM-4DGAN模型，利用粗细层次的GAN和跨注意力解码器生成可变长度的4D面部表情序列。

**💡 创新点**

创新点包括：① 用中性关键点作为条件增强不同身份的鲁棒性；② 在WGAN框架中加入身份判别器和时间一致性判别器；③ 在位移解码器中引入跨注意力机制以更好匹配不同身份；④ 采用粗细层次生成策略实现可变长度表情生成。

**🔧 技术方法**

使用了生成对抗网络（WGAN）、自动编码器、身份判别器、时间一致性判别器、跨注意力机制、位移解码器以及粗细层次生成架构。

**📊 数据集**

在CoMA数据集上进行训练和评估。

**📈 对比分析**

与Motion3DGAN进行对比，采用每顶点重建误差（0.1mm）评估关键点和网格位移。结果表明本方法在关键点误差和网格误差上均低于Motion3DGAN，尤其在不同身份下表现更佳。

**⚠️ 局限性**

局限性在于缺乏大规模4D面部表情数据，仅在CoMA数据集上测试；未来计划扩展至更多数据集，并关注更细粒度的时序指标。

---

## 382. Adaptive Activation Cancellation for Hallucination Mitigation in Large Language Models

**arXiv ID:** 2603.10195 | [PDF](https://arxiv.org/pdf/2603.10195v1)

**作者:** Eric Yocam `[一作]` (Dakota State University), Judith L. Mwakalonge `[通讯]` (South Carolina State University)

**通讯引用:** 594 | [OpenAlex ID](https://openalex.org/A5091713208)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Adaptive Activation Cancellation (AAC) 框架，通过在自回归生成过程中使用实时前向钩子抑制被识别为幻觉的神经元激活，从而降低大型语言模型的幻觉率。

**💡 创新点**

将幻觉视为 transformer 残差流中的结构化干扰，采用基于线性探测的 H‑Node 识别和置信度加权的实时抑制，既无需外部知识也不需微调，且保持模型原有能力。

**🔧 技术方法**

使用线性探测、置信度加权前向钩子、百分位阈值抑制、FFT 频谱分解、减少/漂移（Reduc/Drift）选择性评估等技术。

**📊 数据集**

在 TruthfulQA、HaluEval 数据集上训练和评估 H‑Node；使用 WikiText‑103 和 MMLU 做能力保持评估。

**📈 对比分析**

与 post‑hoc 处理、ITI、DoLA 等基线对比；在 OPT‑125M、Phi‑3‑mini、LLaMA 3‑8B 上，实时钩子在 TruthfulQA 上提升 MC1 约 +2%/0.7%，对 LLaMA 3‑8B 实现 MC1 +0.04、MC2 +0.003、Token‑F1 +0.003；并保持 WikiText‑103 perplexity 与 MMLU 准确率 0% 下降，表明零能力损失。

**⚠️ 局限性**

限制包括：探测器仅在单一 benchmark 内训练，跨 benchmark 泛化有限；AAC 主要验证于中等规模模型，较大模型效果仍待验证；模拟单通道 ANC，缺乏独立噪声参考，导致对极端噪声情形的适应性不完整。

---

## 383. Assessing Cognitive Biases in LLMs for Judicial Decision Support: Virtuous Victim and Halo Effects

**arXiv ID:** 2603.10016 | [PDF](https://arxiv.org/pdf/2603.10016v1)

**作者:** Sierra S. Liu `[一作]` `[通讯]` (Millburn High School), Sierra S. Liu (Millburn High School)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型在司法决策中是否存在与人类相似或更少的认知偏差，重点考察了受害者良善效应与光环效应。

**💡 创新点**

首次系统评估LLM在受害者良善效应与基于声望的光环效应上的表现，并量化其与人类基准的差异。

**🔧 技术方法**

采用五种主流LLM（ChatGPT 5 Instant/Thinking、DeepSeek V3.1、Claude Sonnet 4、Gemini 2.5 Flash）并基于改写的心理学实验情境进行推理。

**📊 数据集**

使用自定义的改写情境数据集，取自已公开的心理学与司法案例，避免模型记忆影响。

**📈 对比分析**

通过多次重复实验并对比人类基准，评估偏差幅度；结果显示LLM在光环效应上略低于人类，但在受害者良善效应上更强，整体表现仍不稳定。

**⚠️ 局限性**

存在模型间显著变异、部分模型输出拒绝/无效、样本量有限，且未覆盖所有司法相关偏差，限制了结论的泛化与即时司法应用。

---

## 384. Delta-K: Boosting Multi-Instance Generation via Cross-Attention Augmentation

**arXiv ID:** 2603.10210 | [PDF](https://arxiv.org/pdf/2603.10210v1)

**作者:** Zitong Wang `[一作]` (Sun Yat-sen University), Weibin Wu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1805 | [OpenAlex ID](https://openalex.org/A5102831943)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Delta‑K 方法，利用 Vision‑Language Model 预览识别缺失概念，生成差分键向量 ΔK，并在扩散过程的早期交叉注意力键空间注入 ΔK，以解决文本到图像生成中的多实例缺失问题。

**💡 创新点**

创新点在于：①直接在交叉注意力的 key 空间注入差分键而非单纯重新加权注意力分布；②采用在线动态优化的注入强度 α_t，适应每一步的注意力分布；③方法训练‑free、无空间掩码、无需改动模型结构，兼容 U‑Net 与 Diffusion Transformer。

**🔧 技术方法**

使用的技术包括：Vision‑Language Model（VLM）进行概念检测与差分键提取；扩散模型（Stable Diffusion XL、SD3.5‑M、Flux‑dev 等）与交叉注意力结构；在线优化算法（Adam）对注入强度进行自适应调度；差分键注入策略 K' = K + α_t ΔK。

**📊 数据集**

实验使用的主要数据集有：T2I‑CompBench、GenEval 以及 ConceptMix，涵盖属性绑定、对象关系、多实例复杂合成等多种场景。

**📈 对比分析**

与 Attend‑and‑Excite、SynGen、InitNO 等训练‑free 方案以及不同 backbone 进行对比。实验显示 Delta‑K 在 SDXL 上将 Complex 指标从 0.3230 提升至 0.3532，Spatial 从 0.2111 提升至 0.2466；在 SD3.5‑M 上 Spatial 从 0.3053 提升至 0.3487；在 ConceptMix 上多实例成功率亦提升。性能提升显著，且不引入额外推理时间或降低图像质量。

**⚠️ 局限性**

限制：①方法主要针对早期阶段的概念缺失，对超过一定数量的多实例效果可能有限；②需要 VLM 预览步骤，虽然不增加训练成本，但仍依赖 VLM 的推理；③目前只验证在现有 U‑Net 与 DiT 架构，未探索更广泛的模型；④缺乏对跨层信息流细粒度分析，后续可考虑更精细的可学习注入策略。

---

## 385. Adaptive Engram Memory System for Indonesian Language Model: Generative AI Based on TOBA LM for Batak and Minang Language

**arXiv ID:** 2603.10006 | [PDF](https://arxiv.org/pdf/2603.10006v1)

**作者:** Hokky Situngkir `[一作]`, Andhika Bernard Lumbantobing `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对印尼巴塔克语、米南加保语及印尼语，构建了一个1.2B参数的 GPT‑2 基础模型，并通过引入 Engram Memory 机制与三音节粘着式分词实现了三语种语言模型。

**💡 创新点**

创新点在于将基于 n‑gram 统计的 Engram 内存与 Transformer 结合，利用 500,000×768 的外部嵌入表在训练早期即可捕获形态学依赖，从而实现“阶段性转变”并显著加速收敛。

**🔧 技术方法**

技术细节包括：GPT‑2 36 层解码器、Bfloat16 训练、基于 syllabic‑agglutinative tokenization 的分词、Engram 内存模块（双重 2‑gram/3‑gram 路径、可适应稀疏门控、RMSNorm+Scaled‑Dot‑Product）、自注意力与 MLP 的分工、数据预处理 pipeline（Unicode 标准化、正则清洗、MinHash 去重、Parquet 存储）。

**📊 数据集**

数据集由印尼、巴塔克、米南加保三维维基百科、NusaX 文化文本、FineWeb、机器翻译输出以及音频/歌曲词库等多源语料构成，经过严格清洗和去重后形成三语种混合语料。

**📈 对比分析**

与仅使用标准 Transformer 的基线模型相比，Engram 模型在 12,973 步内将损失从 6.4 降至 1.7996，远快于基线所需超过 70,000 步，训练步骤效率提升约 80%，显著降低算力和能耗。

**⚠️ 局限性**

局限性包括：仍需大量语料支持才能进一步提升泛化能力；Engram 表占用额外参数空间，虽然计算开销低，但在极低资源环境下仍可能受限；模型专注于三语种，跨语言扩展需要额外验证。

---

## 386. SNPgen: Phenotype-Supervised Genotype Representation and Synthetic Data Generation via Latent Diffusion

**arXiv ID:** 2603.10873 | [PDF](https://arxiv.org/pdf/2603.10873v1)

**作者:** Andrea Lampis `[一作]` (Politecnico di Milano), Emanuele Di Angelantonio `[通讯]` (University of Cambridge)

**通讯引用:** 73669 | [OpenAlex ID](https://openalex.org/A5012263812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发一种两阶段条件潜在扩散框架 SNPgen，用于生成与表型一致的合成基因型数据。

**💡 创新点**

结合 GWAS 引导的变异筛选与基于标签的潜在扩散生成，提供可直接用于下游风险预测的合成基因型，同时保留 LD 结构、MAF 分布并实现强隐私保证。

**🔧 技术方法**

1D VAE 用于基因型压缩；潜在扩散模型（Latent Diffusion Model）配合 classifier‑free guidance 与交叉注意力进行条件生成；GWAS 权重变异筛选、混合精度训练等技术。

**📊 数据集**

英国生物银行（UK Biobank）458,724 名白人样本，四种疾病（冠心病、乳腺癌、1 型糖尿病、2 型糖尿病）以及模拟数据。

**📈 对比分析**

通过 train‑on‑synthetic、test‑on‑real 协议与 VAE 重构、全基因组 PRS（PRSice‑2、LDpred2）比较；在非线性 XGBoost 下，SNPgen 的 AUC 与真实数据相近，甚至在某些疾病上优于全基因组 PRS。

**⚠️ 局限性**

仅针对单表型、单族群、1–2k SNP 面板；未评估连续表型、多族群；隐私评估为经验性，缺乏形式化隐私保证。

---

## 387. Adaptive Manipulation Potential and Haptic Estimation for Tool-Mediated Interaction

**arXiv ID:** 2603.10352 | [PDF](https://arxiv.org/pdf/2603.10352v1)

**作者:** Lin Yang `[一作]` (Nanyang Technological University), Domenico Campolo `[通讯]` (Nanyang Technological University)

**通讯引用:** 2973 | [OpenAlex ID](https://openalex.org/A5079258091)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究提出了一种统一的基于参数化平衡流形（Equilibrium Manifold）的工具介导接触丰富操作框架，并通过触觉SLAM、在线规划与自适应刚度控制实现对未知物体几何与位姿的实时估计与闭环执行。

**💡 创新点**

创新点在于将物理交互建模与几何流形统一为可微的平衡流形，并将触觉SLAM视为流形参数估计，实现触觉信息驱动的闭环感知–行动循环；此外提出了可差分的多点接触势能、基于不确定性的阻尼调节和混合粒子滤波+梯度优化的触觉SLAM。

**🔧 技术方法**

使用技术包括：可微的超四边形+点云接触势能、隐式函数平衡流形、混合粒子滤波与梯度优化的触觉SLAM、基于MPPI的在线轨迹规划、基于姿态不确定性的阻尼调节等。

**📊 数据集**

实验数据集为260次真实螺丝松紧实验，涉及三种工具（spanner‑21、spanner‑34、spanner‑36）和六种螺丝几何（Hex‑36、Hex‑34、Hex‑30、Squ‑19、Rec‑20、Flw‑33），并结合仿真验证。

**📈 对比分析**

与传统固定阻尼、纯阻抗控制以及无感知策略对比，本文方法在识别率和松紧成功率均达到99%以上，平均任务时间约为13–18秒，且在高摩擦或尺寸误配下显著降低冲击力和卡壳概率。

**⚠️ 局限性**

主要限制包括对摩擦参数的手工调节、仅在二维或准静态假设下验证、对非刚性或高复杂度工具/物体几何的适应性不足，未来需实现在线摩擦估计、推广至三维和更复杂接触情景。

---

## 388. Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models

**arXiv ID:** 2603.10887 | [PDF](https://arxiv.org/pdf/2603.10887v1)

**作者:** Yixiu Mao `[一作]` (Tsinghua University), Xiangyang Ji `[通讯]` (Tsinghua University)

**通讯引用:** 11070 | [OpenAlex ID](https://openalex.org/A5024401174)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出了Dynamics‑Predictive Sampling (DPS)，一种通过隐马尔可夫模型在线预测并选择最具信息量的提示来加速强化学习微调；

**💡 创新点**

创新点在于将提示求解进程建模为动态系统，并利用贝叶斯推断在不进行昂贵rollout的情况下预测提示状态，从而显著减少无效样本；

**🔧 技术方法**

采用隐马尔可夫模型（HMM）与贝叶斯在线推断、GRPO强化学习算法和非平稳衰减机制；

**📊 数据集**

在数学(MATH)、数值规划(Countdown)和视觉几何(Geometry3k)三个推理任务的数据集上进行实验；

**📈 对比分析**

与Uniform Sampling、Dynamic Sampling（oracle）和History Resampling三种采样策略对比，DPS在保持或超过oracle性能的同时，将rollout次数降至30%以下，显著提升训练速度；

**⚠️ 局限性**

局限性在于仅基于正确性奖励定义状态，且top‑k选择策略可能非最优，未来可考虑更复杂奖励或不确定性驱动的采样策略。

---

## 389. eLasmobranc Dataset: An Image Dataset for Elasmobranch Species Recognition and Biodiversity Monitoring

**arXiv ID:** 2603.10724 | [PDF](https://arxiv.org/pdf/2603.10724v1)

**作者:** Ismael Beviá-Ballesteros `[一作]` (University of Alicante), Francisca Giménez-Casalduero `[通讯]` (University of Alicante)

**通讯引用:** 1893 | [OpenAlex ID](https://openalex.org/A5057965854)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并公开了一套包含1117张图像、7种海洋软骨鱼（包括鲨鱼与鳐鱼）的精细分类数据集（eLasmobranc Dataset），并对图像进行了专家级别的标注与元数据整理。

**💡 创新点**

创新点在于：①首次针对地中海重要鲨鱼鳐鱼区（ISRA）提供标准化、离水环境下采集的高质量、种类细粒度图像；②将稀缺且受威胁物种纳入，填补了现有视觉数据集对鲨鱼鳐鱼细分类别的空白；③系统整合了地理、时间、来源等丰富元数据，提升了数据可复现性与科研价值。

**🔧 技术方法**

主要技术包括：多源数据采集与合并（GBIF、iNaturalist、DeepFish、内部实测、市场协作）；基于专家经验的筛选与去重策略；统一的图像标识与元数据编码；CC 许可检查与 attribution CSV 生成；结构化文件组织与 GitHub/Zenodo 发布。

**📊 数据集**

使用的数据集为：eLasmobranc Dataset（1117张图像，902张外部来源，215张内部来源）以及其构建所引用的公开资源（GBIF、iNaturalist、DeepFish、AQUA20、FishNet 等）。

**📈 对比分析**

本文并未针对特定算法进行性能对比；之前的工作已将该数据集用于深度学习识别框架，后续可用本数据集评估分类、检测或监测模型的准确率和泛化能力。

**⚠️ 局限性**

局限性包括：①样本量相对有限，某些物种图像较少；②未覆盖所有地中海区域的物种，主要集中于西班牙东部；③部分图像缺乏精细的时间或地区信息；④图像多来自死体，可能对现场实时监测的可迁移性有限。

---

## 390. World2Act: Latent Action Post-Training via Skill-Compositional World Models

**arXiv ID:** 2603.10422 | [PDF](https://arxiv.org/pdf/2603.10422v1)

**作者:** An Dinh Vuong `[一作]` (Mohammed bin Zayed University of Artificial Intelligence), Ian Reid `[通讯]` (Mohammed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了World2Act后训练框架，利用世界模型的潜在视频动态与视觉-语言-动作（VLA）策略的动作进行对齐，从而在不依赖像素级监督的情况下提升机器人控制的鲁棒性和泛化能力。

**💡 创新点**

创新点包括：①在潜在空间实现视频动态与动作的对齐，显著降低对像素回放的敏感性；②构建自动LLM驱动的技能分解管道，生成RoboCasa‑Skill与LIBERO‑Skill数据集，解决任意长度视频生成瓶颈；③在VLA后训练中引入残差策略与对比学习，形成无奖励的自监督对齐机制。

**🔧 技术方法**

技术手段涵盖：世界模型（Cosmos‑Predict2）及其冻结的潜在编码器；视频与动作适配器（CNN/MLP）及InfoNCE对比学习；LLM（DeepSeek）用于指令拆分；残差策略网络（CNN+Transformer+Action Decoder）进行动作校正；对比损失与重建损失联合训练。

**📊 数据集**

使用的数据集为原始RoboCasa与LIBERO以及通过LLM分解得到的RoboCasa‑Skill与LIBERO‑Skill，并从这些数据训练Skill‑WM；此外在真实机器人上使用Frank Research 3臂进行实验。

**📈 对比分析**

实验对比了多种基线（UWM、Cosmos Policy、GR00T系列、DreamGen、VLA‑RFT、Ctrl‑World），在RoboCasa上World2Act在GR00T‑N1.6‑ft上实现72.6%成功率，比最优基线高1.6%；在LIBERO上提升至98.1%，相对基线提升1.1%；在真实机器人上平均提高6.7%的成功率，均展现了显著的性能提升。

**⚠️ 局限性**

局限性包括：对VLA骨干的依赖较强，Cosmos Policy提升有限；真实世界性能仍受域差距影响；对WM的潜在质量和视觉噪声仍存在一定敏感性。

---

## 391. KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization

**arXiv ID:** 2603.10085 | [PDF](https://arxiv.org/pdf/2603.10085v1)

**作者:** Qitong Sun `[一作]` (Beihang University), Yang Liu `[通讯]` (Zhejiang Lab)

**通讯引用:** 49523 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 KernelSkill——一种多代理、双层记忆框架，用于在闭环优化中自动生成并改进 GPU kernel。

**💡 创新点**

创新点在于将专家优化技能外部化为可检索的知识库，并结合长短期记忆实现可追溯、稳定的优化决策，显著降低盲目搜索与反复回溯。

**🔧 技术方法**

采用大语言模型驱动的多代理闭环（生成、评审、诊断、规划、优化），配合静态代码特征提取、检索式方法选择、长短期记忆管理和诊断修复模块。

**📊 数据集**

使用 KernelBench 250 任务（Levels 1–3）进行评估。

**📈 对比分析**

与 Kevin‑32B、Astra、PRAGMA、CudaForge、QiMeng、STARK 等基线比较，KernelSkill 在所有等级实现 100% 成功率，平均加速分别为 5.44×、2.82×、1.92×，显著优于任何对比方法。

**⚠️ 局限性**

局限在于对长记忆库的覆盖度高度依赖；当无匹配案例时退回纯 LLM 选择；对极端复杂或新硬件环境的适应仍需进一步改进。

---

## 392. COHORT: Hybrid RL for Collaborative Large DNN Inference on Multi-Robot Systems Under Real-Time Constraints

**arXiv ID:** 2603.10436 | [PDF](https://arxiv.org/pdf/2603.10436v1)

**作者:** Mohammad Saeid Anwar `[一作]` (University of Maryland Baltimore County), Nirmalya Roy `[通讯]` (University of Maryland Baltimore County)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 COHORT 框架，利用混合离线–在线强化学习实现多机器人协同分布式大模型（如 CLIP、SAM）推理调度。

**💡 创新点**

创新点在于将拍卖式数据采集与 Advantage‑Weighted Regression 结合生成离线策略，再通过 Multi‑Agent PPO 在线微调，既降低样本成本又适应资源波动；同时实现一次性决策、最小化通信，提升鲁棒性。

**🔧 技术方法**

使用 ROS2 机器人操作系统、拍卖机制、Advantage‑Weighted Regression、Multi‑Agent PPO、分布式资源监控与决策网络、软策略优化（PPO）、Lagrangian 约束。

**📊 数据集**

收集约 20 小时真实机器人运行日志（Husky、Jackal、Spot），包含 CLIP 与 SAM 的推理时间、能耗、网络延迟等；未使用公开标准数据集，而是构建自定义实测数据。

**📈 对比分析**

与基线本地执行、拍卖调度、遗传算法比较，COHORT 在帧率满足率提升 2.5–3.5 倍、能耗下降约 15%/41% 以及 GPU 利用率提升 52% 等方面显著优于对比方法。

**⚠️ 局限性**

局限包括对极端网络波动、硬件极异化仍有限制；新设备需手工初始化；在更大规模或更高计算负载下性能下降；缺乏与导航/通信协同优化；依赖离线日志，环境漂移可能导致策略退化。

---

## 393. Safe Probabilistic Planning for Human-Robot Interaction using Conformal Risk Control

**arXiv ID:** 2603.10392 | [PDF](https://arxiv.org/pdf/2603.10392v1)

**作者:** Jake Gonzales `[一作]` (University of Washington), Lillian J. Ratliff `[通讯]` (University of Washington)

**通讯引用:** 1215 | [OpenAlex ID](https://openalex.org/A5008161296)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个基于控制屏障函数（CBF）与非可交换自适应保真度控制（CRC）的概率安全控制框架，通过在线估计安全边距并动态调整机器人在与人类交互时的保守程度。

**💡 创新点**

创新点在于将非可交换CRC用于控制屏障函数的风险控制，在线学习安全边距实现自适应安全约束，并提供对人类行为不确定性的统计安全保证。

**🔧 技术方法**

使用了控制屏障函数、非可交换自适应保真度控制、离散时间鲁棒控制、LSTM预测模型以及随机行人行为模型等技术。

**📊 数据集**

使用了公开的真实行人轨迹数据集（如真实人群数据）以及仿真数据进行训练与评估。

**📈 对比分析**

与CBF-QP、固定CRC-CF、MPPI等基线方法进行比较，在单智能体与多智能体人机交互仿真中，在线CRC-SF在碰撞率、目标成功率、控制效率和控制平滑度方面表现最优，显著降低碰撞率并保持高效率。

**⚠️ 局限性**

局限性包括对非可交换数据的β估计困难、对人类行为模型泛化假设的依赖，以及在临时约束违背时可能出现安全缺口，需要进一步提升鲁棒性。

---

## 394. CLIPO: Contrastive Learning in Policy Optimization Generalizes RLVR

**arXiv ID:** 2603.10101 | [PDF](https://arxiv.org/pdf/2603.10101v1)

**作者:** Sijia Cui `[一作]` (Institute of Automation), Guanjun Jiang `[通讯]` (Qwen Large Model Application Team, Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在强化学习可验证奖励（RLVR）框架下，提出了 CLIPO——在每个提示的多条轨迹中加入对成功轨迹相似性进行对比学习的辅助奖励，提升模型的推理质量与泛化能力。

**💡 创新点**

创新点在于：①将对比学习嵌入政策优化，使得成功轨迹在潜在空间中聚集、错误轨迹被拉远；②利用 InfoNCE 计算的对比损失直接转化为稠密奖励，补偿了传统 RLVR 仅有的稀疏二元奖励；③在无需人工过程标注的情况下，通过对比学习捕捉推理步骤的内部逻辑一致性。

**🔧 技术方法**

技术包括：基于 Transformer 的 LLM 生成轨迹；加装轻量级对比头（线性映射）提取轨迹级嵌入；使用 InfoNCE 对正负样本进行对比学习；将对比损失转化为辅助奖励并与原始可验证奖励相加；在多种 RLVR 算法（GRPO、GSPO、DAPO、GMPO）上作为通用插件使用。

**📊 数据集**

主要数据集：
- GSM8K（初中/高中算数题）及其 Symbolic、P1、P2 变体；
- MATH 7.5K 及评测集 MATH500、Math‑Perturb Simple/Hard；
- 竞赛级别算数题集（AMC23、AIME、AIME25）；
- 其他通用推理/知识基准（CommonsenseQA、TruthfulQA、TheoremQA、MMLU）。

**📈 对比分析**

与多种 RLVR 基线比较：在 Track I（GSM8K）中，GRPO+ 的平均 Pass@1 从 61.91 提升至 63.26（+1.35），对 Symbolic、P1、P2 的提升更明显；在 Track II（MATH、竞赛题）中，DAPO+ 平均分从 42.70 提升至 44.05（+1.35），GMPO+ 亦实现了 43.76 分。整体上，CLIPO 在分布漂移、符号推理、扰动任务上表现出显著的稳健性与一致的性能提升。

**⚠️ 局限性**

局限性：
- 对比奖励仅在同一批次内存在至少两条成功轨迹时有效；若所有轨迹均失败或仅有单个成功轨迹，无法构造正样本；
- 对比头需要与主模型联合训练，超参数（温度、组大小、λ 等）对效果影响显著，需要额外调参；
- 仍依赖外部可验证器，无法直接用于无可验证任务；
- 对比学习的收益随组大小增加而递增，但训练成本也随之提升；
- 目前实验主要聚焦数学推理，跨领域（如代码生成、智能体规划）需进一步验证。

---

## 395. Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in Whole-Slide Image Prognosis

**arXiv ID:** 2603.10526 | [PDF](https://arxiv.org/pdf/2603.10526v1)

**作者:** Pei Liu `[一作]` (Hunan University), Yiping Liu `[通讯]` (Hunan University)

**通讯引用:** 3327 | [OpenAlex ID](https://openalex.org/A5100726685)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种Sparse Task Vector Mixup with Hypernetworks（STEPH）方案，用以在全切片图像（WSI）预后建模中高效地从其他癌症中迁移知识，以提升单一癌症模型的泛化性能。

**💡 创新点**

创新点在于：①将任务向量（task vector）混合（mixup）作为跨癌症知识迁移的核心操作；②使用两种自适应超网络（hypernetworks）分别学习mixup系数与稀疏聚合权重，实现输入条件的动态调节；③通过稀疏聚合筛选最有益的混合向量，避免无效或冲突的知识干扰。

**🔧 技术方法**

核心技术包括：任务向量表示、任务向量混合（TVM）、稀疏聚合、MIL（多实例学习）架构、超网络（hypernetwork）驱动的动态权重学习，以及C-index等生存分析指标。

**📊 数据集**

使用TCGA公开数据集的13种癌症（共8,818张WSI，7,268名患者），每种癌症样本量从248到1,035不等。

**📈 对比分析**

与传统单癌症训练（vanilla、fine-tuned）、基于表示迁移（ROUPKT）以及多种模型融合方法（AdaMerging、TIES、Surgery、Iso-C）进行对比。STEPH平均提升C-index约5.14%（相较于单癌症学习）和2.01%（相较于表示迁移），并在推理时仅使用单一模型，计算成本显著低于多模型推理。

**⚠️ 局限性**

主要局限包括：某些癌症样本极少（如宫颈、肝癌），导致评估受限；实验仅基于通用MIL架构，未验证更先进网络；方法仍需训练数据以学习超网络权重，缺乏训练自由的策略。

---

## 396. UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery

**arXiv ID:** 2603.10528 | [PDF](https://arxiv.org/pdf/2603.10528v1)

**作者:** Islam Guven `[一作]` (Université catholique de Louvain), Mehmet Parlak `[通讯]` (Université catholique de Louvain)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一套基于多智能体强化学习的无人机医疗物资配送协调框架，能够在时间紧迫、任务不确定的城市环境下动态分配有限的无人机资源；

**💡 创新点**

创新点包括：①将部分可观测MDP与专门设计的奖励分形相结合，实现对医疗任务优先级与交付时限的精细化控制；②系统性比较同步在线算法PPO与异步分布式算法APPO/IMPALA，证明在此类严格时限任务中同步PPO具有更高的稳定性与协同效果；

**🔧 技术方法**

采用的技术主要有Proximal Policy Optimization（PPO）及其变体（大FC网络、LSTM）、优势演员-评论家（A2C）、异步PPO（APPO）、IMPALA以及Ray RLlib框架；

**📊 数据集**

实验基于OpenStreetMap公开的比利时布鲁塞尔地区地理数据，构建30×30格网（12km×12km）模拟真实道路与医疗设施布局；

**📈 对比分析**

通过在不同无人机规模（4/8/12/16/20）下对比PPO、A2C、APPO、IMPALA的1000条测试 episode，发现PPO实现100%任务完成率、最快平均交付时间，且训练时间与推理时间均在可接受范围；

**⚠️ 局限性**

局限性包括：仅使用二维格网模型，未考虑天气、风速、障碍物等复杂因素；能源与通信模型简化；任务仅限单一包裹交付，未涵盖多包、不同重量或更复杂的优先级策略。

---

## 397. Why LLMs Fail: A Failure Analysis and Partial Success Measurement for Automated Security Patch Generation

**arXiv ID:** 2603.10072 | [PDF](https://arxiv.org/pdf/2603.10072v1)

**作者:** Amir Al-Maamari `[一作]` `[通讯]` (University of Passau), Amir Al-Maamari (University of Passau)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 Gemini 3.0 Flash 生成的 319 份 Java 安全补丁在 Vul4J 基准上的效果。

**💡 创新点**

提出了连续度量 Security Repair Score（SRS）并构建了补丁失败的分类学，揭示语义误解是主导失败原因。

**🔧 技术方法**

使用零射击提示的 Gemini 3.0 Flash 生成补丁，并通过 Maven/Gradle 编译、PoV 测试、Semgrep 静态分析和完整功能测试三轴评估。

**📊 数据集**

基准数据集为 Vul4J 的 64 个可复现 Java 漏洞（共 319 份补丁）。

**📈 对比分析**

对比分析显示功能得分平均 0.832，安全得分平均 0.251，SRS 平均 0.542；修复率最高仅 45%，表明 LLM 在功能恢复方面表现好于安全修复。

**⚠️ 局限性**

局限性包括仅针对 Java/Vul4J、单一 LLM、缺乏多语言、多样化漏洞及更大规模的评估数据，且对复杂补丁的链式推理支持不足。

---

## 398. Bilevel Layer-Positioning LoRA for Real Image Dehazing

**arXiv ID:** 2603.10872 | [PDF](https://arxiv.org/pdf/2603.10872v1)

**作者:** Yan Zhang `[一作]` (Sun Yat-sen University), Zhuo Su `[通讯]` (Sun Yat-sen University)

**通讯引用:** 922 | [OpenAlex ID](https://openalex.org/A5010794708)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于CLIP的无监督语义引导损失（H2C loss）和一种双层层位置LoRA（BiLaLoRA）策略，用于在真实环境中高效、可扩展地对预训练的去雾网络进行领域自适应。

**💡 创新点**

创新点在于①将去雾任务重构为在CLIP潜在空间中的语义方向匹配，从而实现无参考的目标驱动优化；②通过可学习的层门控将LoRA注入位置的选择转化为可微分的架构搜索，并采用双层优化解耦参数更新与层选择，自动定位最关键的瓶颈层；③实现了低参数、高效率的自适应，能够快速在白天/夜间等多种真实雾景下切换。

**🔧 技术方法**

使用了CLIP跨模态编码器、LoRA参数高阶低秩适配器、双层（上层架构参数α，下层权重ω）优化框架、梯度截断与一阶近似的超梯度计算、标准的Adam优化器以及数据增强。

**📊 数据集**

训练与评估数据集包括合成雾数据THaze、四个合成来源（RIDCP、ITS、OTS、Haze4K）以及真实雾图集RTTS、URHI、Fattal、HazyDet、Dense-Haze、O-Haze。

**📈 对比分析**

与多种前沿去雾方法（MSBDN、DeHamer、C²PNet、DEA、DAD、PSD、D4、RIDCP、KANet、CoA、IPC、PHATNet等）以及通用恢复模型在四个无参考指标（FADE、BIQME、Entropy、MUSIQ）上对比，BiLaLoRA在绝大多数指标上均位居第一或第二，且在训练时间、参数量与推理速度上显著优于全微调方案。

**⚠️ 局限性**

局限性包括：①仍需为不同光照条件（白天/夜间）训练多套适配器；②依赖CLIP模型的跨模态特性，若CLIP对某些场景语义表征不足，H2C loss效果可能受限；③在极端高雾密度或极夜环境下的性能下降仍有待进一步提升。

---

## 399. Mitigating Translationese Bias in Multilingual LLM-as-a-Judge via Disentangled Information Bottleneck

**arXiv ID:** 2603.10351 | [PDF](https://arxiv.org/pdf/2603.10351v1)

**作者:** Hongbin Zhang `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 60628 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了多语种LLM评判者中的“translationese bias”，并提出一种基于信息瓶颈的分离表示框架DIBJudge，通过将判别关键信息与伪相关信息解耦，显著降低翻译偏见并提升多语种奖励建模效果。

**💡 创新点**

创新点包括：① 将translationese bias拆解为两类伪相关（英语隐空间对齐与跨语言可预测性）并量化；② 设计分离鲁棒与偏差表示的Disentangled Information Bottleneck目标，加入跨协方差惩罚实现显式解耦；③ 通过跨语言对比学习与log‑prob bin分类两种代理任务捕捉伪相关，并在训练中同时优化。

**🔧 技术方法**

技术手段包括：变分信息瓶颈（VIB）压缩鲁棒表示、独立偏差编码器、跨协方差惩罚实现表示解耦、LoRA微调、变分互信息下界、代理任务训练、t‑SNE可视化、统计显著性检验。

**📊 数据集**

使用的数据集有：BELEBELE（122语言多语料）、M‑RewardBench、MM‑Eval、RewardBench、Belebele、Aya、XL‑Sum、Skywork‑Reward‑Preference‑80K等，用于偏差评估、奖励建模与泛化测试。

**📈 对比分析**

与GPT‑4o、Gemini‑2.5‑Flash、Qwen系列、Nemotron、M‑Prometheus、mR3、Think‑as‑Locals等多种基线比较，DIBJudge在多语种奖励基准上实现SOTA，并在翻译偏见指标上显著下降（低资源语种下降幅度高达80%以上），同时保持甚至提升对单语种评判的准确性。

**⚠️ 局限性**

局限性在于：① 仍受预训练数据分布影响，未完全排除其他潜在偏差；② 仅针对两类伪相关设计，其他偏差（如长度、内容偏好）需进一步研究；③ 代理任务设计可能缺乏通用性，对极低资源语言的验证有限；④ 评估主要集中在现有基准，跨任务与更广泛场景的泛化尚未完全证明。

---

## 400. Cybo-Waiter: A Physical Agentic Framework for Humanoid Whole-Body Locomotion-Manipulation

**arXiv ID:** 2603.10675 | [PDF](https://arxiv.org/pdf/2603.10675v1)

**作者:** Peng Ren `[一作]` (Beihang University), Kai Chen `[通讯]` (Zhongguancun Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个以VLM为核心的多模态人形机器人框架，能够将自然语言指令编译成可验证的JSON子任务序列，并通过多物体3D几何监督闭环执行长周期的行走与操纵任务。

**💡 创新点**

创新点在于：①将VLM计划与结构化谓词预条件/成功条件相结合，形成可验证任务程序；②利用SAM3+RGB‑D实现任务相关实体的3D几何重建；③构建基于几何谓词的时序稳定监督器，提供诊断反馈并触发有针对性的恢复与重规划；④将监督信息映射到全身控制层，实现安全、可恢复的行走-操纵协同。

**🔧 技术方法**

核心技术包括：VLM规划（如ChatGPT/Claude）、SAM3分割、RGB‑D几何估计、基于谓词的监督与诊断、MPC上臂操纵、强化学习行走策略、以及多模态监督与回溯重规划框架。

**📊 数据集**

使用室内办公场景中真实的物体集合（瓶子、托盘、杯子等）进行实验，并对比与Being‑0等公开基准（Fetch‑bottle、Deliver‑basket等）以及自定义长周期任务（Tidy‑desk、Tabletop‑sorting、Bring‑me‑a‑drink）。

**📈 对比分析**

与Being‑0对比，Cybo‑Waiter在多步任务中成功率从9/10提升至10/10，在长周期任务中相对缺失监督的版本提升了2–3个成功案例，显示出几何监督与重规划的显著效果。

**⚠️ 局限性**

主要局限在于：①对RGB‑D的依赖导致在强光或深度噪声环境下性能下降；②VLM计划的表达式仍可能产生不确定或模糊的子任务；③当前的恢复策略有限，无法处理复杂动态障碍或人机交互中的即时交互需求。

---

## 401. Detecting and Eliminating Neural Network Backdoors Through Active Paths with Application to Intrusion Detection

**arXiv ID:** 2603.10641 | [PDF](https://arxiv.org/pdf/2603.10641v1)

**作者:** Eirik Høyheim `[一作]` (Norwegian Defence Research Establishment), David Aspinall `[通讯]` (University of Edinburgh)

**通讯引用:** 2803 | [OpenAlex ID](https://openalex.org/A5013254098)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于神经网络活跃路径和局部特征贡献的可解释方法，用来检测并消除入侵检测系统中的后门攻击。

**💡 创新点**

创新点在于利用活跃路径与特征贡献聚类来识别后门，并通过剪枝输入到第一隐藏层的权重直接消除后门，无需重新训练模型。

**🔧 技术方法**

所采用的技术包括局部特征贡献（β、ϕ）提取、Kernel PCA降维、HDBSCAN聚类、活跃路径统计以及权重剪枝。

**📊 数据集**

实验数据来自AIT-IDSv2 Netflow 数据集，人工植入TTL特征的1%后门。

**📈 对比分析**

与传统的激活聚类/BadActs等方法相比，本文方法不需要重新训练，实验表明后门消除后模型在干净数据上的准确率保持在99.3%以上，后门识别准确率高于99%。

**⚠️ 局限性**

局限性包括仅适用于分段线性激活函数、对多特征后门的检测效果有限、需要已知触发样本且无法自动区分过拟合或特征相关性。

---

## 402. Aceso: Carbon-Aware and Cost-Effective Microservice Placement for Small and Medium-sized Enterprises

**arXiv ID:** 2603.10768 | [PDF](https://arxiv.org/pdf/2603.10768v1)

**作者:** Georgia Christofidi `[一作]` (IMDEA Software Institute), Thaleia Dimitra Doudali `[通讯]` (IMDEA Software Institute)

**通讯引用:** 184 | [OpenAlex ID](https://openalex.org/A5033123165)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Aceso系统，在地理受限的SME云环境中为延迟敏感的微服务应用动态做碳排放与成本最优放置；

**💡 创新点**

将碳强度、运营成本与SLO约束统一到多目标优化中，并通过微服务pinning与区域过滤两种启发式搜索空间裁剪，使遗传算法可扩展；

**🔧 技术方法**

使用短期流量预测（GBDT）、工作负载哈希分析、可扩展遗传算法、实时碳强度与价格API，以及Kubernetes+Liqo进行多区域调度；

**📊 数据集**

DeathStarBench社交网络微服务、Azure Functions调用轨迹用于预测模型、真实欧洲与北美云区域碳强度与价格数据；

**📈 对比分析**

与Nautilus、Caribou、LP、普通GA等基线对比；在真实部署与仿真中平均碳降低37.4%/21.2%，成本降低3.6%；优化时间从25s（Aceso）到12min/1.4min/4h；在100微服务规模下碳0.48×、成本0.984×、延迟1.33s；

**⚠️ 局限性**

依赖实时碳强度/价格的准确性；仅在AWS/K8s环境验证；对非容器化或单体应用适用性有限；

---

## 403. A Platform-Agnostic Multimodal Digital Human Modelling Framework: Neurophysiological Sensing in Game-Based Interaction

**arXiv ID:** 2603.10680 | [PDF](https://arxiv.org/pdf/2603.10680v1)

**作者:** Daniel J. Buxton `[一作]` (Nottingham Trent University), David J. Brown `[通讯]` (Nottingham Trent University)

**通讯引用:** 5746 | [OpenAlex ID](https://openalex.org/A5022308505)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一个平台无关的多模态数字人模型框架，整合OpenBCI Galea传感器与SuperTux游戏环境，提供可同步的生理与交互数据流。

**💡 创新点**

通过将感测、交互建模与推理分离，构建可重用、可伦理化的架构，支持多模态数据时间对齐且不嵌入推理模型，满足可访问性与包容性设计需求。

**🔧 技术方法**

使用OpenBCI Galea头戴式多模态传感器采集EEG、EMG、EOG、PPG及IMU信号，利用SuperTux游戏作为可重现的交互情境，并在抽象层进行时间戳同步与结构化存储。

**📊 数据集**

仅使用作者自我仪器化的单人数据记录，无公开数据集；未进行人类受试者实验或行为评估。

**📈 对比分析**

文章未进行实验比较，仅在技术验证层面通过自检确认数据完整性、流连续性与同步精度；暂无性能指标。

**⚠️ 局限性**

仅技术验证，未涉及实际受试者数据、推理模型或评估；架构在可扩展性和实时性方面仍需进一步验证；缺乏伦理审查和实际应用评估。

---

## 404. Probing the Limits of the Lie Detector Approach to LLM Deception

**arXiv ID:** 2603.10003 | [PDF](https://arxiv.org/pdf/2603.10003v1)

**作者:** Tom-Felix Berger `[一作]` `[通讯]` (Ruhr University Bochum), Tom-Felix Berger (Ruhr University Bochum)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在不使用谎言的情况下进行欺骗的能力，并评估传统基于“真伪探测器”(truth probes)的欺骗检测方法是否能够捕捉到此类行为。

**💡 创新点**

提出欺骗不等同于说谎，证明单纯的真伪探测器无法有效识别非说谎的误导性输出；同时展示在对话情境下训练探测器可以显著提升对非说谎欺骗的检测率。

**🔧 技术方法**

设计两组实验：①欺骗任务实验（通过零射和两射提示让模型生成说谎或非说谎的误导性回答）；②truth probe 训练与评估实验（使用逻辑回归在注意力头输出上训练真伪分类器，并通过McNemar检验比较对说谎和非说谎欺骗的检测效果）。

**📊 数据集**

使用 Azaria & Mitchell 的真伪数据集（6322 条陈述，分真假）作为训练/测试集；从中抽取 105 条用于构造欺骗任务的样本，另自制 97 条欺骗数据集（包含说谎、诚实与非说谎误导选项）；对话化版本的提示则通过 Mistral 生成问句。

**📈 对比分析**

比较方法：在欺骗任务中对模型在零射/两射条件下的说谎与非说谎欺骗率进行二项检验；对 truth probe 的性能使用准确率、精确率、召回率、F1 评估；通过 McNemar 检验比较 probe 对说谎和非说谎欺骗的检测差异。实验结果显示：Gemma‑2‑9b‑IT 和 Llama‑3.1‑8B‑Instruct 在两射条件下显著高于机会率能进行非说谎欺骗；truth probe 在 RAW 训练时对说谎检测率高达 84‑88%，但对非说谎欺骗的检测率仅 47‑53%；在 DIA（对话化）训练后，非说谎欺骗检测率提升至 63‑74%，差距缩小 15‑25%。

**⚠️ 局限性**

局限性：①仅使用了三款轻量级开源模型，规模有限；②缺乏对模型内部“意图”或“信念”是否真实存在的验证；③实验中的欺骗示例可能更多是模型对提示的反射性回答，而非真正的策略性欺骗；④未评估更大模型或多模态模型的表现；⑤未探讨 truth probe 对其他类型欺骗（如沉默、提问诱导等）的泛化能力。

---

## 405. Med-DualLoRA: Local Adaptation of Foundation Models for 3D Cardiac MRI

**arXiv ID:** 2603.10967 | [PDF](https://arxiv.org/pdf/2603.10967v1)

**作者:** Joan Perramon-Llussà `[一作]` (Universitat de Barcelona), Polyxeni Gkontra `[通讯]` (Universitat de Barcelona)

**通讯引用:** 1158 | [OpenAlex ID](https://openalex.org/A5006692999)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了 Med-DualLoRA 框架，利用双低秩 LoRA 在多中心 3D CMR 病变检测中实现联邦局部微调，显著提升个性化与通信效率。

**💡 创新点**

首次在 3D 医学影像联邦学习中引入双 LoRA 结构，显式拆分全局共享与本地私有低秩适配器，只传输全局 LoRA；证明仅适配少量 transformer 块即可达到近最优性能。

**🔧 技术方法**

基于 LoRA 的参数高效微调、FedAvg 联邦聚合、在冻结的 CineMA 预训练 FM 上插入低秩适配器、加法融合实现全局与本地 LoRA 的分离。

**📊 数据集**

使用 ACDC（150 例）和联合 M&Ms（M&Ms1/2 共 543 例）数据集，各厂商/中心视为联邦客户端。

**📈 对比分析**

与集中式全微调、线性探测、LoRA、联邦头部微调等基线对比；在联邦设置中 Med‑DualLoRA 达到平衡准确率 0.768、特异性 0.612，接近集中式结果；相较标准 LoRA 提升 7.5% 平衡准确率；通信成本可调，适配两块 transformer 仅 28 MB/轮即可近似最佳。

**⚠️ 局限性**

仅在单任务二分类上验证，未探索更高级聚合策略；对极小样本中心的特异性仍有限；受联邦数据不平衡影响，需进一步提升鲁棒性；对多模态或更大规模预训练 FM 的推广待验证。

---

## 406. Equivariant Asynchronous Diffusion: An Adaptive Denoising Schedule for Accelerated Molecular Conformation Generation

**arXiv ID:** 2603.10093 | [PDF](https://arxiv.org/pdf/2603.10093v1)

**作者:** Junyi An `[一作]` (Shanghai Academy of Artificial Intelligence for Science), Yuan Qi `[通讯]` (Artificial Intelligence Innovation and Incubation Institute Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Equivariant Asynchronous Diffusion (EAD) 模型，用异步去噪策略生成 3D 分子结构。

**💡 创新点**

创新点在于将自回归与同步扩散相结合，采用可调节的异步去噪时间表与动态时间步选择，捕捉分子层级关系并保持分子级别视角。

**🔧 技术方法**

主要技术包括 SE(3)-equivariant 图神经网络（EGNN）、扩散模型、独立噪声采样约束、动态去噪时间步调度以及 dummy atom 机制。

**📊 数据集**

实验使用 QM9 与 GEOM-Drug 两大分子数据集。

**📈 对比分析**

与 EDM、GDM、EDM-Bridge、GeoLDM、UniGEM 等基准相比，EAD 在原子稳定性、分子稳定性、有效性与唯一性等指标上均优于同类方法，尤其在分子稳定性提升约 8% 与有效性提升约 3%。

**⚠️ 局限性**

主要局限是对异步比例 λ 的敏感性，需要细致调参以获得最佳性能。

---

## 407. ESG Reporting Lifecycle Management with Large Language Models and AI Agents

**arXiv ID:** 2603.10646 | [PDF](https://arxiv.org/pdf/2603.10646v1)

**作者:** Thong Hoang `[一作]` (CSIRO Data61), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 30593 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于大型语言模型和AI代理的代理式ESG生命周期管理框架，集成识别、测量、报告、参与与改进五个阶段，并实现了报告验证、多报表比较、报告生成及知识库维护四项核心ESG任务。

**💡 创新点**

通过嵌入多代理和检索增强生成（RAG）技术，将ESG生命周期转化为可自动化、可解释、可适应的动态系统，并首次提出单模型、单代理与多代理三种可比的架构。

**🔧 技术方法**

采用大型语言模型（如GPT‑5、GPT‑4o‑mini）、LangChain框架、RAG、OCR/解析器、情感分析器等工具，并通过多代理监督模式实现任务协同。

**📊 数据集**

使用了13份真实ESG报告（如BP、Microsoft等）与人工生成的合成报告（Report A/B），并结合公开ESG标准（GRI、SASB、TCFD）构建评估数据集。

**📈 对比分析**

通过平均绝对误差、令牌数、LLM调用次数、成本和能耗等指标对三种架构进行对比，结果显示单模型精度最低且能耗最高，单代理最省资源但依赖手工调优工具，且对复杂报表泛化差；多代理在精度与资源占用之间取得最佳平衡。

**⚠️ 局限性**

单代理架构需大量手工调优工具，单模型缺乏上下文和任务方法，且对复杂ESG报告的适应性差；多代理架构虽然精度高但仍受模型能力限制且需要更多LLM调用。

---

## 408. ACE Runtime - A ZKP-Native Blockchain Runtime with Sub-Second Cryptographic Finality

**arXiv ID:** 2603.10242 | [PDF](https://arxiv.org/pdf/2603.10242v1)

**作者:** Jian Sheng Wang `[一作]` `[通讯]` (Yeah LLC), Jian Sheng Wang (Yeah LLC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出一种基于身份‑授权分离的区块链运行时（ACE Runtime），将每笔交易的签名验证替换为轻量化的 HMAC attestations，并在区块后期通过 Groth16 零知识证明聚合完成硬最终性。

**💡 创新点**

核心创新在于 Attest‑Execute‑Prove 三阶段流水线，将验证成本从 O(N) 降到 O(1)，并将 ZK 证明移出关键路径；通过 HKDF 与 Poseidon 实现身份‑授权分离，支持后量子安全且无需 GPU 验证。

**🔧 技术方法**

使用的技术包括 HMAC‑SHA256、HKDF‑SHA256、Argon2id、Poseidon 哈希、Groth16 SNARK（BN254）、GPU 并行证明、Proof‑of‑History 时钟、BFT 投票、EVM 兼容层及预编译合约。

**📊 数据集**

实验数据来源于 Rust 原型在 Apple M3 Pro 上的微基准（Attestation 生成/验证、Groth16 验证、阶段时间等），并通过模型估算 CPU/GPU 处理时间与网络带宽吞吐量。

**📈 对比分析**

对比方法：与 Solana、Ethereum、Aptos、StarkNet 等链的 O(N) 签名验证和最终性时间进行对比；结果显示 ACE Runtime 在硬最终性约 600 ms、O(1) 区块验证、约 16k–32k TPS、约 4k 倍验证速度提升、无 GPU 需求，带宽效率提升 1.9–5 倍。

**⚠️ 局限性**

局限性包括：仍受执行层 I/O 瓶颈限制，Groth16 对量子攻击不安全，生态系统建设不足，证明者市场与可信设置需求，以及缺乏多证明者去中心化方案。

---

## 409. InFusionLayer: a CFA-based ensemble tool to generate new classifiers for learning and modeling

**arXiv ID:** 2603.10049 | [PDF](https://arxiv.org/pdf/2603.10049v1)

**作者:** Eric Roginek `[一作]`, D. Frank. Hsu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了名为InFusionLayer的Python工具，利用Combinatorial Fusion Analysis (CFA) 在多分类任务中融合多基模型以提升准确率。

**💡 创新点**

将CFA的RSC、CD、DS等概念应用于多分类深度学习，支持score与rank双重组合，并提供AC、WCDS、WCP三种加权策略的通用框架，实现了可对2D/3D数据进行批量推理的自动化。

**🔧 技术方法**

Python面向对象设计；PyTorch/TensorFlow/Scikit-learn集成；张量运算；基于RSC与CD的分数/排名加权组合；批处理、递归CFA等技术。

**📊 数据集**

3D机械零件基准集MCB_A/B、ModelNet40/10；2D ImageNet、MNIST；以及多种基模型如DGCNN、PointNet++、ConvNeXt、EfficientNet等。

**📈 对比分析**

通过与各基模型单独性能对比，以准确率为指标；在所有数据集上，CFA融合模型均超过最佳基模型，最高提升约4.5%（如MCB_A 95.78% vs 95.11%）。

**⚠️ 局限性**

主要针对多分类任务；缺少对序列/回归任务的支持；rank计算方法局限导致部分组合效果不佳；缺乏自动化超参搜索和多层CFA。

---

## 410. Intermittent Cauchy walks enable optimal 3D search across target shapes and sizes

**arXiv ID:** 2603.10655 | [PDF](https://arxiv.org/pdf/2603.10655v1)

**作者:** Matteo Stromieri `[一作]` (University of Haifa), Amos Korman `[通讯]` (University of Haifa)

**通讯引用:** 2655 | [OpenAlex ID](https://openalex.org/A5040995971)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0540dec-d77f-42db-94ae-d039248f6393` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文在三维三角形环形空间中，研究了间歇性 Lévy 随机游走（intermittent Lévy walk）的目标检测效率。作者构建了一个理论框架，给出了不同 Lévy 指数 μ（1<μ≤3）下的检测时间下界和上界，并证明在 μ=2（Cauchy walk）时，检测时间几乎达到全局下界，对球、盘、线等不同几何形状保持尺度不变、几何不敏感。

**💡 创新点**

创新点主要包括：
1) 将之前仅限于二维的间歇性 Lévy 搜索理论推广到三维；
2) 发现 μ=2 是三维空间中唯一能对所有常见目标形状实现近似最优、尺度不变的指数；
3) 通过“几何均衡器”（geometric equalizer）概念，证明 Cauchy walk 对目标体积、表面积、延展率的综合敏感性最低；
4) 引入“投影表面积”Δ_P 作为决定检测效率的关键几何参数，并证明对近似凸形状 Δ_P 与 Δ、Δ_B 成正比。

**🔧 技术方法**

技术手段包括：
- 竞争性分析（competitive analysis）求取检测时间与最优策略的比值；
- 解析几何与几何不变量（投影面积、包围盒、延展率）相结合的上下界证明；
- 结合概率论、马尔可夫链理论和 Wald 识别等工具推导检测时间下界；
- 对 Cauchy walk 的上界采用投影分布、平衡分布与三维随机游走的均匀性；
- 在离散三维环面上进行模拟与实验，验证理论预言。

**📊 数据集**

使用的“数据集”是人工合成的三维正方体环面（3D torus）和几何目标（球、盘、线、矩形）。具体实验设置：体积 n=512^3，检测半径 d≥1，目标在空间中心，步长分布为截断幂律 p(ℓ)∝ℓ^{-μ}，μ 取 1, 1.5, 2, 2.5, 3 等值。每种 μ、每种目标形状下，至少 200 次模拟（部分噪声较大时 1000 次）记录平均检测时间。

**📈 对比分析**

比较方法：
- 用检测时间比值 t_detect^X(S)/t_detect^X^⋆(S) 表示搜索效率；
- 对同一目标形状，绘制不同 μ 的检测时间相对 Cauchy 的比例；
- 在图中将检测时间与理论下界（Ω(n/Δ_B)）和 Cauchy 上界（n log^3 n / Δ_P）对齐。结果显示：
  - 当 μ<2 时，球体、盘面、线在小尺寸或大表面积下检测时间显著慢于 Cauchy；
  - 当 μ>2 时，球体、盘面在大尺寸/大表面积下检测时间显著慢于 Cauchy；
  - Cauchy（μ=2）在所有形状、大小下检测时间均保持在理论下界附近，几乎无形状依赖，且相对于最优策略的超时比值在多数量级内保持常数级（多项式对数尺度）。

**⚠️ 局限性**

局限性：
- 模型假设目标静止且可在检测半径内瞬时被发现，忽略了目标运动、障碍物、环境非均匀性等现实因素；
- 只考虑单一目标（或周期性复制的单一目标），不涵盖多目标搜索场景；
- 采用周期性三维环面，边界效应被忽略，可能不适用于有限且有障碍的真实空间；
- 对非凸形状的上界依赖投影面积 Δ_P，实际检测时间可能因形状的高折叠度、凹陷等而更差；
- 结果是渐近理论，常数因子和低阶项对中小规模 n 的影响尚未完全量化；
- 只对离散或连续检测模式下的间歇性搜索进行了分析，未覆盖连续感知或多尺度感知的情形。

---

## 411. Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment

**arXiv ID:** 2603.10929 | [PDF](https://arxiv.org/pdf/2603.10929v1)

**作者:** Fanqi Yu `[一作]` (Istituto Italiano di Tecnologia), Vittorio Murino `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于多模态潜在回放与增量特征调节的终身模仿学习框架，支持在有限存储与数据约束下持续学习新任务并保持旧任务性能。

**💡 创新点**

创新点在于：①在潜在空间中存储视觉、语言、状态与控制指令的压缩表示，极大降低存储成本；②引入基于角度距离的增量特征调节（IFA）机制，动态自适应任务间间隔，提升任务间可分离性并抑制表征漂移；③全冻结预训练编码器，仅更新时序解码器和策略头，实现高效稳定的终身学习。

**🔧 技术方法**

主要技术包括CLIP视觉/语言编码器、FiLM调制层、GPT‑2时序解码器、经验回放（Multimodal Latent Replay）、增量特征调节损失（角度间隔约束）以及基于任务相似度的对任务对选择。

**📊 数据集**

使用LIBERO终身机器人操控基准（LIBERO‑OBJECT、LIBERO‑GOAL、LIBERO‑50）进行实验，数据包含代理视角、眼睛视角图像、语言指令、动作与机器人状态。

**📈 对比分析**

与多种SOTA（Sequential, ER, BUDS, LOTUS, ISCIL, M2Distill, TAIL）对比，MLR+IFA在FWT、AUC最高、NBT最低，尤其在LIBERO‑50上实现显著提升（AUC提升约+15，NBT下降约70%）。

**⚠️ 局限性**

局限性包括：仅在仿真/离线任务序列上验证，尚未评估真实机器人长序列或跨域迁移；对任务相似度阈值和缓冲区大小等超参数敏感；全冻结编码器虽高效，但在任务分布剧烈变化时可能受限。

---

## 412. Graph-GRPO: Training Graph Flow Models with Reinforcement Learning

**arXiv ID:** 2603.10395 | [PDF](https://arxiv.org/pdf/2603.10395v1)

**作者:** Baoheng Zhu `[一作]` (Beijing University of Posts and Telecommunications), Xiao Wang `[通讯]` (Beihang University)

**通讯引用:** 11085 | [OpenAlex ID](https://openalex.org/A5112719601)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于强化学习的在线框架Graph‑GRPO，用来直接优化离散流匹配模型（Graph Flow Models, GFMs），并通过可验证的奖励函数实现任务导向的图生成。

**💡 创新点**

创新点在于①给出了GFMs的解析转移概率表达式，消除了 Monte‑Carlo 采样导致的不可微分问题，使得整个生成过程可完全微分；②设计了迭代细化（refinement）策略，通过在中间噪声级别重新噪声并重新去噪，实现对高奖励样本的局部探索与自我提升。

**🔧 技术方法**

主要技术包括离散流匹配框架、连续时间马尔科夫链（CTMC）率矩阵解析求解、基于 Group Relative Policy Optimization 的强化学习、以及图结构的局部噪声扰动与再生成。

**📊 数据集**

使用了三类数据集：①合成图数据集 Planar 与 Tree（每个图 64 个节点）；②药物发现的分子优化任务，基于 ZINC250k 进行蛋白‑配体对接（如 parp1、fa7、5ht1b 等）；③PMO（Practical Molecular Optimization）基准，包含 23 个化学属性优化任务。

**📈 对比分析**

与多种基线比较：基于GFMs的DeFoG、流式模型DiGress、DisCo、GBD、GraphRNN、BiGG 等；基于RL的GDPO、DDPO、FREED、REINVENT、GCPN 等；以及进化搜索和基因型模型。Graph‑GRPO 在 Planar、Tree 上实现 V.U.N. 分别达 95.0% 与 97.5%，并显著降低训练集比例；在蛋白‑配体对接中，Hit Ratio 最高可达 60%（相较于 GDPO 的 9% 以上提升）；在 PMO 上的 AUC‑top10 达 19.270，超越所有无预筛选或预筛选的竞争方法。

**⚠️ 局限性**

局限性包括：①RL 训练需要大量 oracle 调用，尤其在对接任务中仍受限于 10,000 次调用；②细化策略的噪声参数 t_ϵ 需要手工调节，若设置不当可能破坏分子结构；③目前仅针对离散流匹配模型，其他类型生成器的可迁移性尚未验证；④在极大规模分子或材料生成场景下的计算开销与可扩展性仍待进一步研究。

---

## 413. Personalized Group Relative Policy Optimization for Heterogenous Preference Alignment

**arXiv ID:** 2603.10009 | [PDF](https://arxiv.org/pdf/2603.10009v1)

**作者:** Jialu Wang `[一作]` (Apple Inc), Morteza Dehghani `[通讯]` (Apple Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的基于个性化的GRPO算法P-GRPO，用于在多样化用户偏好下对大型语言模型进行对齐；

**💡 创新点**

创新点在于将优势函数的归一化从批量级别切换到偏好组级别，使用每个偏好组的历史奖励统计量，避免了标准GRPO对主流偏好信号的偏倚；

**🔧 技术方法**

采用了在线奖励归一化（Welford算法）来维护每个偏好组的均值和方差，结合群组相对策略优化（GRPO）框架；

**📊 数据集**

在MovieLens-1M推荐任务、合成偏好生成任务、Goodreads书评生成任务和KGRec音乐推荐任务等多种数据集上进行实验；

**📈 对比分析**

与标准GRPO以及GDPO对比，P-GRPO在收敛速度更快、平均奖励更高、生成质量（ROUGE、余弦相似度）均有提升，且在LLM-as-judge评估中赢率更高；

**⚠️ 局限性**

局限包括假设偏好组可稳定划分、需要高质量聚类、对偏好漂移缺乏适应性，以及对少数组数据不足可能仍导致性能差异。

---

## 414. Contact Coverage-Guided Exploration for General-Purpose Dexterous Manipulation

**arXiv ID:** 2603.10971 | [PDF](https://arxiv.org/pdf/2603.10971v1)

**作者:** Zixuan Liu `[一作]` (National University of Singapore), Lin Shao `[通讯]` (National University of Singapore)

**通讯引用:** 8351 | [OpenAlex ID](https://openalex.org/A5069756785)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了一种基于接触覆盖的通用探索奖励——Contact Coverage‑Guided Exploration（CCGE），用于无任务特定先验的柔性操作学习。

**💡 创新点**

创新点在于：① 将接触状态量化为手指与物体表面区域的交集，构建状态条件的接触计数器；② 通过稀疏的后接触奖励和稠密的预接触能量奖励双重机制，引导手指在不同对象状态下系统探索多样接触模式；③ 使用自编码器+哈希聚类对高维对象状态进行离散化，实现跨状态无干扰的探索计数。

**🔧 技术方法**

采用的技术包括：深度强化学习（PPO）、自编码器+SimHash实现对象状态聚类、接触匹配算法、能量函数式预接触奖励、计数器的稀疏更新与缩放奖励。

**📊 数据集**

实验使用的主要数据集包括：ContactDB（用于对象旋转任务）、ARCTIC（双手开盒任务）、仿真环境中自定义的四个任务（书本分离、受限物体检索、手中重新定向、双手翻盖）。

**📈 对比分析**

与四种基线（纯任务奖励、Learned‑Hash‑Codes Count、Haptics Curiosity、RND‑Dist）以及带任务先验的 TR‑PrePose 进行对比。CCGE 在所有任务上都显著提升学习效率（到70%成功率所需步数平均降低 2‑3 倍）和最终成功率（最高 95%），尤其在受限检索任务中唯一能成功。

**⚠️ 局限性**

主要局限包括：① 仍主要在仿真环境中验证，真实世界实验覆盖有限；② 仅使用手指关键点和几何距离/力阈值来判定接触，缺乏触觉/力‑扭矩等更细粒度感知；③ 对极端复杂场景（如多物体、极限摩擦）适用性待进一步验证。

---

## 415. Cross-Species Transfer Learning for Electrophysiology-to-Transcriptomics Mapping in Cortical GABAergic Interneurons

**arXiv ID:** 2603.11000 | [PDF](https://arxiv.org/pdf/2603.11000v1)

**作者:** Theo Schwider `[一作]`, Ramin Ramezani `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

复制并扩展了Gouwens等人（2020）将Patch‑seq电生理数据映射到转录组亚型的框架，首次在公开的Allen Institute人类Patch‑seq数据上进行验证，并通过序列模型实现跨物种迁移学习。

**💡 创新点**

创新点包括：①利用注意力机制的BiLSTM直接处理结构化的IPFX 12族特征，无需sPCA压缩；②实现鼠标→人类的迁移学习，显著提升人类亚型预测的宏F1；③通过注意力权重提供特征族层面的可解释性。

**🔧 技术方法**

主要技术：IPFX特征提取、sPCA、UMAP可视化、随机森林基线、注意力BiLSTM（含ArcFace头）以及SMOTE过采样；迁移学习采用共享编码器+双头训练及人类微调。

**📊 数据集**

数据集：鼠标视觉皮层Patch‑seq（3699细胞）和人类神经外科切除皮层Patch‑seq（506细胞），均来自Allen Institute并托管于DANDI。

**📈 对比分析**

比较方法：在鼠标上使用10次随机种子交叉验证，报告宏F1与准确率；在人类上使用5折分层交叉验证。结果显示：随机森林宏F1在鼠标为0.8728，人工模型最高为0.8923；人类宏F1最高为0.6795（迁移学习），比单纯人类训练的0.6580提升约0.02。

**⚠️ 局限性**

局限性：①人类样本量小且类别不平衡；②跨物种分布偏移未完全消除；③仅使用工程化特征，未探索直接电压波形的深度学习方法；④缺乏对标签噪声与真实物种差异的进一步拆解。

---

## 416. On the Reliability of Cue Conflict and Beyond

**arXiv ID:** 2603.10834 | [PDF](https://arxiv.org/pdf/2603.10834v1)

**作者:** Pum Jun Kim `[一作]` (Ulsan National Institute of Science and Technology), Jaejun Yoo `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 5274 | [OpenAlex ID](https://openalex.org/A5089933293)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套新型的形状-纹理偏差评估框架，包括高质量的、可识别的形状与纹理图像数据集以及基于全标签空间的指标。

**💡 创新点**

创新点在于：①对形状与纹理进行人类感知级别的纯净定义并构造平衡样本；②使用排名相关的平均倒数排名(MRR)而非相对比例来量化偏差；③结合两者实现了更稳健、可解释的偏差诊断。

**🔧 技术方法**

采用了人类标注与一致性检验、基于图像风格迁移与切分的图像生成流水线、MRR指标以及在多种预训练模型和训练策略上进行的实验。

**📊 数据集**

使用自建的 6,000 张图像数据集，来源于 20 个 ImageNet 超类（10 形状主导、10 纹理主导），并与 ImageNet‑1k 预训练模型进行评估。

**📈 对比分析**

通过在不同训练策略（形状增强、对比学习、纹理扭曲、混合增强、对抗训练）和不同架构（ResNet‑50、ViT、Swin、CMT）下计算 Shape‑Sens、Texture‑Sens 与 Shape Preference，实现跨模型对比；实验表明该框架能够准确捕捉形状/纹理利用与在域内性能的正相关，并与传统基准相比得到更一致、可靠的结论。

**⚠️ 局限性**

局限性包括：仍可能受域迁移影响；纹理与形状的完全隔离尚未实现；样本类别受限于 20 个超类，缺乏更广泛的视觉特征覆盖；对 3D 视角等更复杂因素的考察不足。

---

## 417. PPGuide: Steering Diffusion Policies with Performance Predictive Guidance

**arXiv ID:** 2603.10980 | [PDF](https://arxiv.org/pdf/2603.10980v1)

**作者:** Zixing Wang `[一作]` (Purdue University), Diego Romeres `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 PPGuide 框架，通过自监督多实例学习（MIL）自动为轨迹中的动作片段打标签，训练轻量级分类器，在推理时利用梯度引导预训练扩散策略，显著提升机械手操控任务的成功率和鲁棒性。

**💡 创新点**

创新点包括：①将 MIL 用于稀疏终止奖励的时间信用分配，自动识别成功/失败相关的动作片段；②利用自监督得到的伪标签训练轻量级指导分类器，无需专家演示或稠密奖励；③在推理阶段采用交替梯度引导的扩散过程，兼顾性能与计算效率。

**🔧 技术方法**

使用技术包括：扩散策略（Diffusion Policy）、注意力机制的多实例学习、轻量级分类器（MLP）、梯度引导的逆扩散过程、交替引导调度以及超参数调优。

**📊 数据集**

实验数据集为 Robomimic benchmark 与 MimicGen benchmark，涵盖长时序、多模态、精度敏感的机器人操控任务。

**📈 对比分析**

与基线 DP、DP-SS、PPGuide-CG（常量引导）和 PPGuide-SS（随机采样）等方法对比，PPGuide 在低样本训练下在多数任务上提升成功率约 4%–18%（部分任务可达 70%），交替引导实现与常量引导相近的性能但计算量更低。

**⚠️ 局限性**

局限性包括：①对初始回放质量依赖性强，若成功率低会导致“冷启动”问题；②自监督标签可能捕捉到错误相关性；③对 z-score 阈值和引导强度等超参数敏感，需任务特定调优；④对持续累积误差而非可辨识失败点的任务适用性有限。

---

## 418. Protein Counterfactuals via Diffusion-Guided Latent Optimization

**arXiv ID:** 2603.10811 | [PDF](https://arxiv.org/pdf/2603.10811v1)

**作者:** Weronika Kłos `[一作]` (Machine Learning Group Technische Universität Berlin), Lukas Kades `[通讯]` (BASF Digital Solutions GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于流形约束的对抗性优化框架（MCCOP），能够在连续的序列‑结构潜在空间中寻找最小且生物学可行的突变，从而使深度学习模型的预测结果转变为目标状态；

**💡 创新点**

创新点在于将预训练的扩散模型作为流形先验，在梯度下降过程中交替执行稀疏梯度步和流形投影，同时对预测器进行光滑化以避免生成对抗样本；

**🔧 技术方法**

技术上结合了 CHEAP 编码器‑解码器、平滑的可微预测器（Spectral Normalization、Jacobian 正则化、Softplus 激活和FGSM 对抗增强）以及 DiMA 扩散模型进行流形投影；

**📊 数据集**

在三个蛋白工程任务上进行评估：GFP 荧光恢复、蛋白热稳定性提升以及 E3 ligase 活性恢复，使用相应的公开数据集（TAPE Fluorescence、TAPE Stability、Ube4b Activity）；

**📈 对比分析**

与离散搜索（随机单点突变、遗传算法）和无约束梯度下降等基线相比，MCCOP 在成功率、突变数与对抗率方面表现最佳，稳定性和活性任务中实现 100% 成功率且平均突变数仅 2–3；

**⚠️ 局限性**

局限性包括对流形连续性和光滑性假设的依赖、仅使用计算代理评估结构与化学可行性、仅针对二分类任务且缺乏实验验证。

---

## 419. GGMPs: Generalized Gaussian Mixture Processes

**arXiv ID:** 2603.10442 | [PDF](https://arxiv.org/pdf/2603.10442v1)

**作者:** Vardaan Tekriwal `[一作]` (University of California), Marcus M. Noack `[通讯]` (Lawrence Berkeley National Laboratory)

**通讯引用:** 593 | [OpenAlex ID](https://openalex.org/A5046307547)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Generalized Gaussian Mixture Process (GGMP)，一种用于多模态条件密度估计的 Gaussian Process 变体；

**💡 创新点**

通过局部高斯混合拟合、跨输入组件对齐、每个组件的异方差 GP 训练和加权组合，构建了可闭式推理的多模态 GP 体系，避免了指数级的隐变量结构；

**🔧 技术方法**

使用标准 GP 推断、EM 估计高斯混合、最小化前向 KL 的分布式似然、梯度/投影法优化权重；

**📊 数据集**

在合成数据、美国气温极端值和加工业多变量实验数据上进行实验；

**📈 对比分析**

与单模态 GP、MDN 等基线比较，GGMP 在分布相似性指标上表现优于 GP 基线，往往与 MDN 相当或更好；在校准与置信区间方面 GGMP 更加稳健；

**⚠️ 局限性**

局部混合拟合与组件对齐的贪心策略在模式交叉频繁时可能失效；对局部方差的“插值”未对不确定性进行完整传播，导致稀疏数据下过度自信；可扩展性受限于 O(KN³) 计算开销。

---

## 420. Motion Forcing: A Decoupled Framework for Robust Video Generation in Motion Dynamics

**arXiv ID:** 2603.10408 | [PDF](https://arxiv.org/pdf/2603.10408v1)

**作者:** Tianshuo Xu `[一作]` (Hong Kong University of Science and Technology), Ying-cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2650 | [OpenAlex ID](https://openalex.org/A5101938761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一种名为 Motion Forcing 的框架，利用稀疏点控制和深度中介来生成在视觉质量、物理一致性和可控性上都表现优秀的视频，主要应用于自动驾驶、物理仿真与机器人操作。

**💡 创新点**

创新点包括：① 通过三阶段“Point‑Shape‑Appearance”解耦结构，将物理推理与图像渲染分离；② 引入 Masked Point Recovery（时间/空间遮蔽）强制模型学习物理规律；③ 用深度扭曲方式将相机运动以像素级形式注入；④ 在统一的双时钟扩散模型中实现物理推理与渲染两种任务的共享与分离。

**🔧 技术方法**

技术栈涵盖：VAE 编码、统一 DiT（Diffusion Transformer） + DDIM 采样；Dual Adaptive Layer Normalization 以支持两种噪声时间步；深度扭曲（Depth Warping）编码相机运动；Instance Flow/Softmax Splatting 处理物体控制；Masked Point Recovery（时间截断、对象截断、空间随机丢失）等。

**📊 数据集**

训练与评估数据集：自动驾驶场景使用 Waymo、Driving Dojo 与 YouTube；物理仿真使用 Physion；机器人抓取使用 Jaco Play；模型在 CogVideoX1.5-5B-I2V 基础上进行微调。

**📈 对比分析**

与 MOFA‑Video、Seed Dance 2.0、Wan 2.6 等基线进行对比。Waymo 测试集上，Motion Forcing 在 FVD（157.8）略高于 Seed Dance（112.5）但低于单阶段版，FVMD（205.2）和 Physics‑IQ（33.2）均为最高；在 Physion 与 Jaco Play 上亦优于基线，证明框架具有良好的跨域泛化能力；单阶段版性能明显下滑，验证深度中介的重要性。

**⚠️ 局限性**

局限性：在行人、骑行者等密集非机动车场景中，稀疏点控制难以捕捉众多小体素的复杂运动；高度遮挡的多车交互场景仍可能出现深度排序错误，导致物理一致性下降。

---

## 421. Perceptive Hierarchical-Task MPC for Sequential Mobile Manipulation in Unstructured Semi-Static Environments

**arXiv ID:** 2603.10227 | [PDF](https://arxiv.org/pdf/2603.10227v1)

**作者:** Xintong Du `[一作]` (University of Toronto), Angela P. Schoellig `[通讯]` (Technical University of Munich)

**通讯引用:** 6032 | [OpenAlex ID](https://openalex.org/A5052147335)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一套闭环感知-层次任务模型预测控制（Perceptive HTMPC）框架，用于在半静态、半结构化环境中执行连续的移动机械臂任务；通过在线语义与几何地图更新以及基于控制块的碰撞避免实现实时重规划与高效执行。

**💡 创新点**

创新点在于：①将对象级语义变更检测与贝叶斯推理结合，实时维护时序一致的3D地图；②将基于控制块的安全约束（CBF）在线植入HTMPC，取代传统的欧几里得距离场（EDF）约束；③在完整的闭环感知-控制体系中，实现了对环境变化的即时感知与响应，并在无先验地图、无外部定位的条件下完成长周期任务。

**🔧 技术方法**

核心技术包括：基于ORB‑SLAM3+POCD的实时SLAM与对象一致性推理、体素化EDF/CBF安全约束生成、ACADOS实现的层次模型预测控制、以及双摄RGB‑D的双向感知融合。

**📊 数据集**

实验数据来自仿真环境（随机放置的0.6 m盒子、模拟与真实物理引擎）和实际机器人（UR10+Ridgeback、双摄Orbbec Femto Bolt），未使用公开数据集。

**📈 对比分析**

与传统基于体素更新的静态地图方法及基于EDF的安全约束相比，实验显示CBF约束在部分/延迟感知下能显著提升行进间隙与碰撞自由率（CBF约25–30%高于EDF），同时保持相似或更低的速度与加速度。整体上，闭环Perceptive HTMPC在连续任务执行中实现了更高的效率与安全性。

**⚠️ 局限性**

局限性包括：①对摄像头视野与帧率的依赖，导致感知延迟与不完整地图；②对深度传感噪声与遮挡的鲁棒性仍有限；③当前仅在相对简单的箱子摆放环境中验证，未扩展到更复杂、多样化的真实场景；④未实现主动感知与语言规划的集成。

---

## 422. GhazalBench: Usage-Grounded Evaluation of LLMs on Persian Ghazals

**arXiv ID:** 2603.09979 | [PDF](https://arxiv.org/pdf/2603.09979v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 423. Q-StaR: A Quasi-Static Routing Scheme for NoCs

**arXiv ID:** 2603.10637 | [PDF](https://arxiv.org/pdf/2603.10637v1)

**作者:** Yang Zhang `[一作]` (Tsinghua University), Fengyuan Ren `[通讯]` (Tsinghua University)

**通讯引用:** 3323 | [OpenAlex ID](https://openalex.org/A5080197427)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

设计了一种准动态路由方案，该方案利用网络拓扑和已知流量分布预测长期负载趋势，并在运行时根据该趋势在 XY 与 YX 两条 Dimension‑Order 路径之间做出路由选择。

**💡 创新点**

创新点在于把长期负载趋势作为静态信息预先编码进路由表，既保留了静态路由的简洁性和可预测性，又实现了类似自适应路由的负载均衡，无需实时收集或维护动态状态。

**🔧 技术方法**

技术上采用演化模型对拓扑与流量矩阵进行离线仿真，计算每个节点的 NR‑weight w_NR；随后离线生成每条源-目的对的位图，运行时只做一次位图查表即可决定路由；实现基于 BookSim2 的周期级仿真。

**📊 数据集**

数据集包括：①合成流量模式（Uniform、Shuffle、Permutation、Overturn），②基于 ns‑3 的 Clos 网络叶子交换机的真实工作负载（捕获端口间流量矩阵）。

**📈 对比分析**

通过与 XY、O1Turn、Valiant、ROMM、Odd‑Even 等典型路由方案在同一 NoC（5×5 Mesh）上进行对比。实验表明在统一流量下吞吐量比 XY 提升 42.9%；在真实工作负载下平均/最大延迟分别降低 86.4%/95.3%；整体性能与 Odd‑Even 相近，且在大多数场景下不产生乱序。

**⚠️ 局限性**

局限性包括：需要先验了解流量分布，无法即时应对突发流量变化；路由位图更新需重新生成并写回硬件，更新成本较高；方案仅支持两条 DOR 路径的选择，无法直接扩展到更复杂多路径或非 Mesh 拓扑。

---

## 424. NasoVoce: A Nose-Mounted Low-Audibility Speech Interface for Always-Available Speech Interaction

**arXiv ID:** 2603.10324 | [PDF](https://arxiv.org/pdf/2603.10324v1)

**作者:** Jun Rekimoto `[一作]` (Sony CSL - Kyoto), Bojian Yang `[通讯]` (Sony CSL)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种鼻梁挂载的双传感器（麦克风+振动传感器）接口NasoVoce，用于在日常佩戴智能眼镜时实现低音量（含耳语）语音的持续、隐蔽交互。

**💡 创新点**

通过在鼻梁位置同时捕获气导音频和骨/皮肤导振动信号，并设计D-DCCRN模型实现双模态融合，既提高了在嘈杂环境下的识别准确率，又保留了低音量语音的可识别性，填补了现有耳语识别对噪声鲁棒性不足的空白。

**🔧 技术方法**

采用MEMS麦克风与MEMS振动传感器同步采样，构建双通道D-DCCRN音频增强网络，并使用OpenAI Whisper Large‑v2作为评估ASR模型；训练时加入知识蒸馏损失以提升识别性能。

**📊 数据集**

收集了104小时的双传感器同步录音（45名英语流利参与者），包含正常与耳语发音，并通过混合DEMAND噪声数据生成噪声化训练样本。

**📈 对比分析**

与单麦克风、单振动传感器以及传统音频增强方法对比，使用WER、CER、PESQ、STOI以及MUSHRA评分评估；结果显示，在0dB以上噪声条件下，D-DCCRN融合方案在耳语和正常语音的识别准确率均优于单一传感器，且在高噪声环境下仍保持比单麦克风更好的语音质量。

**⚠️ 局限性**

在极高噪声（>+10dB）下，单振动传感器的性能偶尔超过融合模型，表明需要自适应传感器融合策略；此外，模型尚未实现实时流式处理，且对鼻梁通气变化等个体生理差异缺乏在线自适应校准。

---

## 425. An Efficient Hybrid Deep Learning Approach for Detecting Online Abusive Language

**arXiv ID:** 2603.09984 | [PDF](https://arxiv.org/pdf/2603.09984v1)

**作者:** Vuong M. Ngo `[一作]` (Ho Chi Minh City Open University), Mark Roantree `[通讯]` (Dublin City University)

**通讯引用:** 1307 | [OpenAlex ID](https://openalex.org/A5085828573)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种混合深度学习模型（BERT+CNN+LSTM+ReLU），用于检测多平台（YouTube、论坛、暗网）上的辱骂与滥用语言，并在极度不平衡的数据上实现高效分类。

**💡 创新点**

将BERT的上下文嵌入与CNN提取局部特征、LSTM捕获序列依赖相结合，形成单一端到端框架；在大规模多源不平衡数据集上实现近99%精度与召回；同时构建整合三源（暗网、PAN12、Roman Urdu）的多样化训练集。

**🔧 技术方法**

使用预训练BERT（BERT-base-uncased）生成词向量；1D CNN层（512/256/128滤波器）提取局部特征；单向LSTM（隐藏层500）捕获序列信息；ReLU激活、全连接层、Softmax输出；交叉熵损失；采用PyTorch 2.5 + HuggingFace Transformers；5折交叉验证评估。

**📊 数据集**

三大数据来源：暗网4,600条（2,500 CSA相关、2,100非CSA）；PAN12 198,054 条会话（4,029 侮辱、194,025 非侮辱）；Roman Urdu YouTube 147,180 条评论（73,590 侮辱、73,590 非侮辱）。合计77,620侮辱与272,214非侮辱，比例1:3.5。

**📈 对比分析**

与传统机器学习（NB、LR、SVM）及单一深度模型（CNN、LSTM）和BERT基线进行对比，指标为Precision、Recall、Accuracy、F1、AUC。混合模型在5折平均下取得Precision 0.991、Recall 0.986、Accuracy 0.995、F1 0.989、AUC 0.992，明显优于其他所有基线模型。

**⚠️ 局限性**

模型训练和推理耗时较长（BERT+混合模型约5,000秒训练，推理约460秒），难以满足实时部署需求；仅处理英语/罗马化文本，缺乏多语言和代码混合支持；模型解释性不足，未结合XAI方法进行可解释性分析。

---

## 426. Double-Precision Matrix Multiplication Emulation via Ozaki-II Scheme with FP8 Quantization

**arXiv ID:** 2603.10634 | [PDF](https://arxiv.org/pdf/2603.10634v1)

**作者:** Yuki Uchino `[一作]` (Center for Computational Science RIKEN), Toshiyuki Imamura `[通讯]` (Center for Computational Science RIKEN)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本论文提出一种基于Ozaki-II方案的FP8双精度矩阵乘法仿真方法，利用FP8 MMA单元实现FP64级别的算子；

**💡 创新点**

创新点在于将Karatsuba分解与针对平方模数的模数化简相结合，形成混合构造，显著减少所需FP8乘法次数，并克服FP8在Ozaki-II方案中天然的表示限制；

**🔧 技术方法**

核心技术包括Ozaki-II整数乘法/CRT重构、Karatsuba乘法、模数化简、FP8_E4M3浮点格式、低精度MMA单元；

**📊 数据集**

实验采用随机生成的浮点矩阵（通过rand与randn产生不同动态范围），并在NVIDIA RTX 5080和B200两套GPU上进行性能与精度评测；

**📈 对比分析**

与INT8‑based Ozaki‑II以及FP8‑based Ozaki‑I进行对比，FP8方案在RTX 5080上可比原生FP64 DGEMM提升3–4×，但相较于INT8‑based方案慢约1.3–2.9×；在B200上则均低于原生FP64，INT8方案略优；

**⚠️ 局限性**

局限性包括：需要较大工作内存（FP8方案内存占用约为INT8方案的两倍）；当INT8吞吐量远高于FP8时，FP8方案性能不足；以及在小k尺寸下因指数字段占用导致的效率下降。

---

## 427. WME: Extending CDCL-based Model Enumeration with Weights

**arXiv ID:** 2603.10236 | [PDF](https://arxiv.org/pdf/2603.10236v1)

**作者:** Giuseppe Spallitta `[一作]` (Rice University), Moshe Y. Vardi `[通讯]` (Rice University)

**通讯引用:** 39193 | [OpenAlex ID](https://openalex.org/A5000059818)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了Weighted Model Enumeration（WME）框架，实现了在CDCL求解器中原生支持基于权重的模型枚举与筛选。

**💡 创新点**

创新点包括：①将权重传播与权重约束剪枝直接嵌入CDCL循环；②引入权重冲突分析与学习冲突子句；③设计残差感知回溯与权重相关变量优先级；④对比分析时序回溯与非时序回溯两种枚举策略。

**🔧 技术方法**

使用的技术：CDCL决策与冲突分析、权重传播与上界/下界维护、贪心冲突集提取、残差感知回溯、权重相关变量排序以及多阶段阈值更新。

**📊 数据集**

实验数据集包括：525个随机3-CNF（1.5约束比），100个SATLIB随机3-CNF（4.28约束比），1049个来自贝叶斯网络的带权CNF（共1049实例）。

**📈 对比分析**

对比方法：与DPO、MaxHS等现有工具在Top‑1、Top‑k以及阈值枚举任务上进行对比；实验显示非时序回溯在Top‑k场景下速度更快，时序回溯在阈值枚举中更具优势；原生Top‑k枚举比重复Top‑1更高效；开启权重剪枝显著提升性能。

**⚠️ 局限性**

局限性：性能高度依赖实例结构与权重分布；需在时序/非时序之间做自适应选择；当前仅支持布尔CNF，尚未覆盖SMT或更丰富的理论；大规模实例下记忆开销与回溯成本仍是挑战。

---

## 428. AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory

**arXiv ID:** 2603.10438 | [PDF](https://arxiv.org/pdf/2603.10438v1)

**作者:** Lianjie Ma `[一作]` (Huazhong University of Science and Technology), Lijun Zhu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 2266 | [OpenAlex ID](https://openalex.org/A5106407357)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出AsyncMDE，一种异步多帧单目深度估计系统，利用基础模型低频刷新高质量特征与轻量模型高频自适应更新，实现实时边缘部署。

**💡 创新点**

创新点在于将深度推理拆分为慢速基础模型与快速轻量模型两条路径，通过空间记忆单元实现补充融合与自回归更新，保证在刷新间隔内精度衰减可控且保持高帧率。

**🔧 技术方法**

采用DAv2‑ViTB作为基础模型，MobileNetV3‑Small作为轻量编码器，空间记忆单元（Complementary Fusion + Autoregressive Update），语义门控调制，以及TensorRT加速等技术。

**📊 数据集**

训练使用NYUv2、TartanAir与BridgeData V2混合数据集；评估在ScanNet、Bonn和Sintel三个不同场景的基准上。

**📈 对比分析**

与DAv2‑ViTB、LiteMono、Video Depth Anything、CUT3R等基线对比，AsyncMDE仅3.83M参数在RTX 4090上达237 FPS，Jetson Orin上161 FPS，精度仅低2%相对于DAv2‑ViTB，恢复77%准确率缺口。

**⚠️ 局限性**

在极端运动场景下记忆失效导致精度下限为轻量编码器水平；缺乏绝对尺度约束，未来可通过运动自适应记忆重置和尺度对齐模块进一步提升。

---

## 429. FERRET: Framework for Expansion Reliant Red Teaming

**arXiv ID:** 2603.10010 | [PDF](https://arxiv.org/pdf/2603.10010v1)

**作者:** Ninareh Mehrabi `[一作]` (Meta Superintelligence Labs), Joanna Bitton `[通讯]` (Meta Superintelligence Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了FERRET框架，实现自动化多模态多轮红队攻击，能够从政策描述出发自动发现对话起始点并生成完整攻击对话；

**💡 创新点**

创新点在于将水平扩展（自动生成有效起始点）、垂直扩展（展开为多轮多模态对话）以及元扩展（自动产生新攻击策略）三种机制统一集成，并通过XML转换工具支持文本、图像及其融合攻击；

**🔧 技术方法**

主要技术包括大语言模型与目标模型交互、基于XML的攻击模板转换、图像/文本嵌入（Clip、Drama-Base）、TSNE可视化、采样策略（正例、负例、随机）以及判断模型LlamaGuard；

**📊 数据集**

实验使用Llama Maverick、Claude Haiku、GPT‑4o三大目标模型，攻击策略共7种（3种图像+4种文本），并用LlamaGuard政策集评估违规性；

**📈 对比分析**

通过与FLIRT（单轮自学习）和GOAT（多轮目标驱动）对比，FERRET在所有目标模型上均实现约18‑22% 的攻击成功率提升，单轮实验亦优于FLIRT，且人类评估显示ASR 27.4%，验证了自动评估结果；

**⚠️ 局限性**

局限性包括仅在有限的目标模型上验证，攻击效果受已有策略库限制，缺乏跨语言和跨文化的评估，元扩展的策略生成仍依赖人类示例，且整体对模型规模与架构的泛化尚未充分探索。

---

## 430. EmoStory: Emotion-Aware Story Generation

**arXiv ID:** 2603.10349 | [PDF](https://arxiv.org/pdf/2603.10349v1)

**作者:** Jingyuan Yang `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 21411 | [OpenAlex ID](https://openalex.org/A5100684575)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了情感感知故事生成框架 EmoStory，能够生成主角一致且情绪表达明确的图像故事序列。

**💡 创新点**

创新点在于两阶段设计：通过情感代理和写作代理将抽象情绪映射为具体情绪提示；以及在生成阶段采用区域感知模块分离主角与情感元素区域，实现情绪与主角一致性并存。

**🔧 技术方法**

使用情感因子树（EmoSet）构建情感代理，利用大型语言模型驱动的代理式规划；在生成阶段采用扩散模型结合跨图像注意力、区域掩码与强化情绪模块。

**📊 数据集**

构建自有情绪故事数据集，包含25个主体、8种情绪方向，每个主体-情绪对生成3个故事，总计600个故事，主体由ChatGPT生成，情绪元素来自EmoSet。

**📈 对比分析**

与8种最先进故事生成方法（ConsiStory、StoryDiffusion、1P1S、Story-Adapter、Story2Board、StoryGen、DSD、IP-Adapter）进行对比，EmoStory在情绪准确率70.17%、提示对齐82.06%和主角一致性71.70%上均优于对手，用户研究也显示最高情感共鸣与一致性评分。

**⚠️ 局限性**

局限性包括仅覆盖8类离散情绪、某些情绪的情感元素不足、单一主体输入且不支持长文本或多主体交互。

---

## 431. Effective Dataset Distillation for Spatio-Temporal Forecasting with Bi-dimensional Compression

**arXiv ID:** 2603.10410 | [PDF](https://arxiv.org/pdf/2603.10410v1)

**作者:** Taehyung Kwon `[一作]` (KAIST), Kijung Shin `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种专门针对时空序列预测的双维数据集蒸馏方法（STemDist），通过对时间维和空间维同时压缩，生成可替代原始数据的合成数据集，以加速模型训练并降低显存占用。

**💡 创新点**

创新点在于：①引入位置编码器使STGNN模型对不同数量的站点具备归纳能力，从而实现空间维压缩；②使用聚类对原始站点进行压缩，显著减少蒸馏时的计算量；③引入子集粒度蒸馏（subset‑based granular distillation），在多次随机子集上执行梯度匹配，以提升合成数据的质量。

**🔧 技术方法**

主要技术包括梯度匹配（gradient matching）蒸馏框架、基于自注意力的单头位置编码器、K‑means聚类以及子集随机划分的梯度更新策略；同时在实验中使用MTGNN作为代理模型，并对其做相应扩展。

**📊 数据集**

实验使用了五个真实时空序列数据集：GBA、GLA、ERA5、CAMS、CA（交通与气象数据），并在此基础上生成多规模合成数据集用于可扩展性评估。

**📈 对比分析**

与九类基线（核心采样、一般数据集蒸馏、时序蒸馏方法及传统时空采样）以及多种不同模型（Graph WaveNet、STGCN、FourierGNN）进行对比。结果显示，STemDist 在压缩比例 0.5%–1% 下，训练速度提升至原来的 6×、显存占用降低至 8×、预测误差相较最佳基线低达 12%。

**⚠️ 局限性**

局限性包括：①需要在原始数据上执行聚类，可能在空间结构复杂或无显著聚类特征的数据上效果不佳；②子集粒度蒸馏依赖于随机划分，可能在极少数异常事件上缺乏鲁棒性；③蒸馏过程本身仍需一定计算资源，尤其在极大规模时空数据集上；④方法对超参数（时间/空间压缩比例、子集数）敏感，需要经验性调优。

---

## 432. Phase-Interface Instance Segmentation as a Visual Sensor for Laboratory Process Monitoring

**arXiv ID:** 2603.10782 | [PDF](https://arxiv.org/pdf/2603.10782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 433. TOSSS: a CVE-based Software Security Benchmark for Large Language Models

**arXiv ID:** 2603.10969 | [PDF](https://arxiv.org/pdf/2603.10969v1)

**作者:** Marc Damie `[一作]` (University of Twente), Roos Wensveen `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于CVE数据库的代码选择式安全基准TOSSS，用于评估LLM的安全编码能力

**💡 创新点**

创新点在于采用安全/脆弱代码对比选择任务，避免传统代码生成+静态分析的可扩展性和一致性限制；基准自动化更新可扩展到新漏洞与语言

**🔧 技术方法**

利用LLM交互式提示（提示与不提示）进行二选一决策，并以安全代码选择比例作为得分

**📊 数据集**

使用从CVE数据库提取的函数级代码对照集（MegaVul数据集，包含C/C++与Java约17k个脆弱/修复函数）

**📈 对比分析**

对14个公开及闭源LLM在500条C/C++与500条Java函数上进行评估，得分范围0.48–0.89；提示词可提升平均+0.021至+0.029分，部分模型表现优于或低于随机水平

**⚠️ 局限性**

局限性包括仅覆盖C/C++与Java两种语言、依赖MegaVul的漏洞抽取质量、模型训练数据可能与基准重叠导致结果偏差

---

## 434. NCAA Bracket Prediction Using Machine Learning and Combinatorial Fusion Analysis

**arXiv ID:** 2603.10916 | [PDF](https://arxiv.org/pdf/2603.10916v1)

**作者:** Yuanhong Wu `[一作]` (Fordham University), D. Frank Hsu `[通讯]` (Fordham University)

**通讯引用:** 10320 | [OpenAlex ID](https://openalex.org/A5082344124)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对NCAA男子篮球锦标赛的赛程进行预测，并通过Combinatorial Fusion Analysis（CFA）将五种基模型（逻辑回归、支持向量机、随机森林、XGBoost和卷积神经网络）融合，生成排名和得分两种预测框架。

**💡 创新点**

创新点在于：①将体育预测视为排名问题而非传统分类问题；②提出并应用CFA框架，利用RSC函数和认知多样性（CD）实现基模型在得分空间与排名空间的融合；③通过多模型的认知多样性加权组合提升预测准确率。

**🔧 技术方法**

使用的技术包括：逻辑回归、SVM、随机森林、XGBoost、CNN、随机搜索参数调优、交叉验证、RSC函数构造、认知多样性度量、CFA中的平均/加权组合方法。

**📊 数据集**

数据集来源：Kaggle的“March Machine Learning Mania”比赛数据（2001-2022，除2020年外）以及KenPom网站的球队统计数据；通过特征选择（RFECV）筛选出26个关键特征。

**📈 对比分析**

方法比较：将CFA生成的52个组合模型与10种公开排名系统（如NET、Logan等）对比。排名组合模型的准确率为74.60%，比最优公开系统高1.58%；得分组合模型准确率71.43%，虽不最高但超过一半公开系统。

**⚠️ 局限性**

局限性包括：①仅使用认知多样性作为加权方式，未尝试平均或性能加权；②缺乏对2024赛季真实结果的验证；③模型仅基于过去10年数据，可能无法完全捕捉新赛季球队变化；④实验规模受计算资源限制，未探索更大基模型集。

---

## 435. MALTA: Maintenance-Aware Technical Lag, Estimation to Address Software Abandonment

**arXiv ID:** 2603.10265 | [PDF](https://arxiv.org/pdf/2603.10265v1)

**作者:** Shane K. Panter `[一作]` (Boise State University), Nasir U. Eisty `[通讯]` (University of Tennessee)

**通讯引用:** 122 | [OpenAlex ID](https://openalex.org/A5035948887)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种维护感知技术滞后评估框架MALTA，用以区分可解决的技术滞后与因上游项目停止维护导致的终端技术滞后。

**💡 创新点**

创新点在于将开发活动、维护者响应与仓库元数据三类指标结合，并给出可解释的加权分数，从而弥补传统版本滞后指标无法识别已废弃包的缺陷。

**🔧 技术方法**

技术包括基于提交速率的开发活动评分、拉取请求响应度评分、仓库元数据可行性评分以及加权线性聚合；采用对时间窗口内的提交、PR与元数据进行统计并归一化。

**📊 数据集**

使用从Debian Trixie与Bookworm发行版中提取的1.7 百万次提交、4.2 百万次PR、以及相应GitHub仓库的元数据，共计约28 k个包的数据集。

**📈 对比分析**

通过与PVAC分类和显式归档标签对比，MALTA在区分活跃与衰退维护的AUC达到0.80左右，在检测已归档仓库的AUC为0.93，显著优于单一版本滞后或传统指标；同时对低风险版本滞后包的重新分类显示有约30 %被判为高风险。

**⚠️ 局限性**

局限性包括仅关注GitHub平台的信号，未覆盖其他托管服务；依赖版本号和PR模式，可能无法捕捉仅通过镜像或非GitHub贡献的项目；评估仅在Debian生态，跨生态泛化需进一步验证。

---

## 436. CUAAudit: Meta-Evaluation of Vision-Language Models as Auditors of Autonomous Computer-Use Agents

**arXiv ID:** 2603.10577 | [PDF](https://arxiv.org/pdf/2603.10577v1)

**作者:** Marta Sumyk `[一作]` (Ukrainian Catholic University), Oleksandr Kosovan `[通讯]` (Ukrainian Catholic University)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5070600372)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对五种视觉语言模型（VLM）在评估计算机使用代理（CUA）任务完成度方面的性能进行了大规模的元评估。

**💡 创新点**

创新点在于将VLM作为自主审计者，仅利用自然语言指令和最终 GUI 截图进行任务完成判定，并从准确率、置信度校准和模型间一致性三维度全面评估审计效果。

**🔧 技术方法**

使用的技术包括视觉语言模型（GPT‑4o、Claude 3.5 Sonnet、InternVL‑2‑8B、LLaVA‑v1.5‑7B、Qwen2‑VL‑7B），Brier 分数评估置信度校准，以及 Cohen’s κ 衡量模型间一致性。

**📊 数据集**

评估数据集来自三大 CUAs 基准：Windows Agent Arena、OSWorld 与 macOSWorld，覆盖 macOS、Windows 与 Linux 三大桌面操作系统。

**📈 对比分析**

对比结果显示，专有模型 GPT‑4o 与 Claude 3.5 Sonnet 在所有基准上准确率最高且置信度校准最优；开源模型在 macOSWorld 上表现尚可，但在 Windows 与 Linux 环境下准确率下降明显，且不同模型间的一致性普遍不高，尤其在更复杂的环境中。

**⚠️ 局限性**

局限性包括：仅观察最终截图，忽略中间交互与时间序列；置信度评估依赖提示而非模型内部概率；仅评估二分类完成度，未涵盖安全性、隐私或其他重要审计维度。

---

## 437. Multi-Person Pose Estimation Evaluation Using Optimal Transportation and Improved Pose Matching

**arXiv ID:** 2603.10398 | [PDF](https://arxiv.org/pdf/2603.10398v1)

**作者:** Takato Moriki `[一作]` (Toyota Technological Institute), Norimichi Ukita `[通讯]` (Toyota Technological Institute)

**通讯引用:** 4751 | [OpenAlex ID](https://openalex.org/A5053167635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于最优运输的MPPE评估指标OCpose，解决传统mAP对低置信度误检忽略的问题。

**💡 创新点**

①无需置信度排序即可公平惩罚误检；②在匹配时将置信度融入到OKS计算，提升匹配可靠性；③使用像素级mask代替bbox，降低误判。

**🔧 技术方法**

最优运输（Optimal Transportation）框架、改进的OKS匹配（OKS_p、OKS_m、OKS_c）以及与置信度相关的成本计算。

**📊 数据集**

COCO数据集（包含人姿态与实例分割mask）与CrowdPose数据集（通过语义分割模型生成伪mask）。

**📈 对比分析**

与mAP及OC-cost等传统指标对比；在COCO和CrowdPose上，使用OCpose优化阈值后，mAP几乎保持不变但OCpose显著下降，且人类评估一致率达83.3%。

**⚠️ 局限性**

依赖mask或伪mask，若缺失需额外推断；最优运输计算量大，适用性需在更大规模数据上进一步验证。

---

## 438. Emulating Clinician Cognition via Self-Evolving Deep Clinical Research

**arXiv ID:** 2603.10677 | [PDF](https://arxiv.org/pdf/2603.10677v1)

**作者:** Ruiyang Ren `[一作]` (Renmin University of China), Wayne Xin Zhao `[通讯]` (Renmin University of China)

**通讯引用:** 17651 | [OpenAlex ID](https://openalex.org/A5037145565)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种自进化的诊断代理DxEvolve，通过交互式深度临床研究流程自动获取证据并提炼诊断认知原语，实现可审计的经验积累与提升。

**💡 创新点**

将诊断过程视为逐步证据获取的互动流程，并引入诊断认知原语（DCP）作为可检索、可审计的经验资产，弥补现有AI系统缺乏过程性和可治理学习机制的缺陷。

**🔧 技术方法**

使用大型语言模型（如Qwen3系列、MedGemma、ClinicalCamel）结合工具调用、语义检索（Dense Retrieval）、外部医学文献检索和规则匹配，实现DCR工作流与DCP自进化。

**📊 数据集**

主要使用MIMIC‑CDM（基于MIMIC‑IV的急腹痛案例）作为基准，外部验证采用中国军医大学（PLA General Hospital）匿名病例，并对读者研究子集进行评估。

**📈 对比分析**

与基准CDM和无DCP版本对照，DxEvolve平均提升11.2%诊断准确率；在读者研究子集中达90.4%准确率，接近甚至超越人类专家（88.8%）；在外部医院数据上提升10.2%至17.1%，显示跨机构、跨语言迁移性能。

**⚠️ 局限性**

受限于仅在去标识的电子病历中验证，缺乏实时临床交互、对不同专科和更复杂情境的评估；模型对工具调用的依赖和外部检索开销可能限制部署；并未探究长期持续学习与监管合规性的细节。

---

## 439. Multilingual Reasoning Gym: Multilingual Scaling of Procedural Reasoning Environments

**arXiv ID:** 2603.10793 | [PDF](https://arxiv.org/pdf/2603.10793v1)

**作者:** Konstantin Dobler `[一作]` (Hasso Plattner Institute), Mohamed Ali `[通讯]` (Apple)

**通讯引用:** 4783 | [OpenAlex ID](https://openalex.org/A5046562017)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了Multilingual Reasoning Gym，将原先仅支持英语的Reasoning Gym扩展到14种语言，支持无限生成可调节难度的推理问题。

**💡 创新点**

创新点在于采用模板翻译而非逐样本翻译，结合LLM与人工审核的混合流程，保证多语言任务的自然性与正确性，同时实现跨语言并行生成，填补RLVR领域的多语言空白。

**🔧 技术方法**

技术方法包括模板JSON抽取、LLM（Claude Sonnet 4）自动翻译与迭代润色、人工本土化审核、代码与模板的多语言适配，以及使用RLVR奖励机制进行模型训练与评估。

**📊 数据集**

数据集为通过翻译得到的94个任务在14种语言中的生成实例，总量可达数百万条，可用于RLVR训练和评估。

**📈 对比分析**

通过在不同模型（Gemma、Qwen、SmolLM等）上使用固定实例集，评估了平均@8准确率，发现模型容量越大性能越好，但各语言间仍存在显著差异；在更高难度（75th percentile）下性能下降约15个百分点。

**⚠️ 局限性**

局限性包括三种语言（孟加拉语、泰卢固语、斯瓦希里语）未通过本土审核，部分任务因英语特性被省略或保留英文输入，模板生成无法完全覆盖真实语言多样性，且仅评估了有限的模型与难度设置。

---

## 440. A dataset of medication images with instance segmentation masks for preventing adverse drug events

**arXiv ID:** 2603.10825 | [PDF](https://arxiv.org/pdf/2603.10825v1)

**作者:** W. I. Chu `[一作]`, L. Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并构建了MEDISEG数据集，包含多片药物在真实环境中的图像及实例分割标签，并用YOLOv8/9和FsDet进行模型评估。

**💡 创新点**

创新点在于：①首次提供多片药物、真实光照与遮挡条件下的高质量实例分割数据；②通过多光照、角度和多片交叠的图像来逼真模拟药物使用场景；③在few-shot学习中展示了该数据集对模型迁移学习的显著提升。

**🔧 技术方法**

使用的技术包括：iPhone 12 Pro Max图像采集、COCO Annotator手工标注、YOLOv8/9目标检测框架、FsDet两阶段few-shot学习、遗传算法进行超参调优。

**📊 数据集**

数据集方面，核心为MEDISEG 3Pills与32Pills两个子集；对比实验使用NIH Pillbox和CURE数据集，所有数据均采用COCO格式存储。

**📈 对比分析**

通过mAP@50、mAP@50‑95等指标比较：YOLOv9在MEDISEG上mAP@50‑95明显优于YOLOv8；在few-shot任务中，MEDISEG预训练模型在重叠/遮挡子集上比CURE提升至少30%准确率，显示出更好的迁移与泛化能力。

**⚠️ 局限性**

局限性包括：仅使用单一iPhone设备采集，样本数量与设备多样性有限；缺乏不同摄像机和临床环境的验证；模型评估仍停留在公开数据集上，未进行实际药物使用流程中的现场测试。

---

## 441. Paladin: A Policy Framework for Securing Cloud APIs by Combining Application Context with Generative AI

**arXiv ID:** 2603.10228 | [PDF](https://arxiv.org/pdf/2603.10228v1)

**作者:** Shriti Priya `[一作]` (IBM Research), Arjun Natarajan `[通讯]` (IBM Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了一个基于大语言模型的云 API 安全框架，允许管理员在无需了解各应用细节的情况下，定义并强制执行跨应用的层 7 攻击防护策略；

**💡 创新点**

创新点在于：①将 LLM 作为语义抽取器，自动为不同 API 请求打标签并提取参数；②提供统一的、与应用无关的策略接口；③结合请求上下文与容器资源信息，支持多维度策略判定；④利用缓存与 WebAssembly 提升性能；

**🔧 技术方法**

使用技术包括 Llama‑2‑70b‑chat 大语言模型、Go 语言 + Envoy + WebAssembly 插件、容器监控指标采集、请求与容器上下文缓存；

**📊 数据集**

评估数据集为：①社交/娱乐类 Top‑25 应用 API（501 个），②金融类 API（915 个），③电商类 API（1026 个），全部手工标注请求标签；

**📈 对比分析**

实验对比显示单模式下分类准确率约 81‑86%，并行模式下部分标签准确率可达 96%；相对于无策略拦截，预缓存方案平均增加约 14% 延迟，运行时缓存可进一步压缩；

**⚠️ 局限性**

局限性包括：依赖 LLM 语义识别，易产生误判和漏判；需预先定义标签集合；对极端动态 API 变化响应慢；框架无法取代传统防火墙或 IDS，需与现有安全设施协同使用。

---

## 442. A Two-Stage Architecture for NDA Analysis: LLM-based Segmentation and Transformer-based Clause Classification

**arXiv ID:** 2603.09990 | [PDF](https://arxiv.org/pdf/2603.09990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 443. HeartAgent: An Autonomous Agent System for Explainable Differential Diagnosis in Cardiology

**arXiv ID:** 2603.10764 | [PDF](https://arxiv.org/pdf/2603.10764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 444. PC-Diffuser: Path-Consistent Capsule CBF Safety Filtering for Diffusion-Based Trajectory Planner

**arXiv ID:** 2603.10330 | [PDF](https://arxiv.org/pdf/2603.10330v1)

**作者:** Eugene Ku `[一作]` (Texas A&M University), Yiwei Lyu `[通讯]` (Texas A&M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 PC-Diffuser，一个将可认证的路径一致性障碍函数嵌入扩散规划去噪循环的安全增强框架，能够在生成阶段实现实时安全校正；

**💡 创新点**

创新点在于（1）使用基于胶囊距离的控制障碍函数实现更精确、非保守的碰撞检测；（2）通过 LQR 路径跟踪将 waypoint 转化为动态可行的控制，保证执行轨迹的可实现性；（3）采用路径一致性安全滤波，只调节纵向速度而不改变轨迹几何，降低分布偏移；（4）将上述校正逐步注入扩散去噪循环，实现迭代自适应而非一次性后处理；

**🔧 技术方法**

技术包括扩散概率模型（DDPM）用于轨迹生成；控制障碍函数（CBF）和路径一致性安全滤波；LQR 路径跟踪器；可行性检验与最小化偏差的安全优化（速度级 CBF-QP）；以及胶囊距离计算；

**📊 数据集**

在 nuPlan 公开闭环仿真基准上评估，使用包含 1300 小时驾驶数据的 nuPlan 数据集，划分 Val14 与 Test14-hard，并进一步抽取所有碰撞子集；

**📈 对比分析**

与三类安全增强基线（Classifier Guidance、SafeDiffuser 的三种变体、MPC‑CBF）在所有碰撞挑战集上对比。PC‑Diffuser 将碰撞率从 100% 降至 10.29%，并在复合得分上从 0.00 提升至 0.59；在完整 Val14 与 Test14‑hard 上也分别提升至 0.88 与 0.78，表明在保证安全的同时提升驾驶质量；

**⚠️ 局限性**

局限性包括：对人类驾驶者的反应预测仍为被动，可能导致无法完全避免碰撞；评估依赖 nuPlan 的 IDM 模拟，未能覆盖更高保真度或更自然的邻车行为；以及在极端交叉口场景下仍存在约 10% 的碰撞率。

---

## 445. DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving

**arXiv ID:** 2603.11041 | [PDF](https://arxiv.org/pdf/2603.11041v1)

**作者:** Shuyao Shang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Tieniu Tan `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DynVLA，一种在自主驾驶 Vision‑Language‑Action 模型中使用动态链式思考（Dynamics CoT）的框架

**💡 创新点**

创新点包括：①将未来动态压缩成可离散化的动态 token；②对动态进行 ego‑centric 与 environment‑centric 解耦，并用动作监督和跨视图一致性正则化；③在 CoT 过程中先生成动态 token 再生成动作，显著降低推理时延与冗余；④结合 SFT + RFT 的训练策略提升决策质量

**🔧 技术方法**

主要技术：VQ‑VAE 风格的动态 tokenizer、Transformer 编码/解码器、动作监督正则化、跨视图一致性正则化、SFT 与 RFT（GRPO）强化学习、文本/图像/BEV 多模态处理

**📊 数据集**

使用了 NAVSIM、Bench2Drive、以及 700k 帧的企业内部大规模数据集进行评测

**📈 对比分析**

与传统 End‑to‑End、非 CoT VLA、Textual CoT、Visual CoT 等方法对比，DynVLA 在 PDMS、NavSim、Bench2Drive 以及 ADE/碰撞率上均取得最优或接近最优的性能，同时推理时延仅为 0.37 s，明显低于 Visual CoT（2.29 s）

**⚠️ 局限性**

局限性：动态 tokenizer 需要额外训练；若不进行解耦会出现码本坍塌；依赖多模态传感器，跨视图一致性在不同硬件环境下可能不稳定；实验主要集中在仿真/内部数据，真实道路验证仍待进一步研究

---

## 446. A Survey of Weight Space Learning: Understanding, Representation, and Generation

**arXiv ID:** 2603.10090 | [PDF](https://arxiv.org/pdf/2603.10090v1)

**作者:** Xiaolong Han `[一作]` (University of Surrey), Ferrante Neri `[通讯]` (University of Surrey)

**通讯引用:** 9637 | [OpenAlex ID](https://openalex.org/A5063214931)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了权重空间学习（Weight Space Learning, WSL）的三大维度：权重空间理解（WSU）、权重空间表示（WSR）和权重空间生成（WSG），并为每个维度梳理了代表性方法、技术路线与应用场景。

**💡 创新点**

创新点在于：①提出统一的三维体系和清晰的分类框架，打破领域间命名混乱；②整合对称性分析、无监督表示、生成模型等多学科技术，形成从理解到生成的完整闭环；③在文章中系统收集并比较了公开模型仓库、基准数据集与评估指标，为后续研究提供基准。

**🔧 技术方法**

使用的主要技术包括：对称性（Permutation、Scaling、Orthogonal invariances）分析、图神经网络（GNN）和深度集（DeepSets）等结构感知编码器、Probe-based无监督表示、超网络（Hypernetworks）、变分自编码器（VAE）、生成对抗网络（GAN）、自回归模型、扩散模型（Diffusion）等生成器；此外还有迁移学习、元学习与联邦学习等场景下的权重聚合与个性化技术。

**📊 数据集**

所用数据集主要来源于公开模型仓库（如 Hugging Face、Model Zoo、OpenAI Model Hub 等），以及针对特定任务的公开数据集（ImageNet、COCO、GLUE、WMT 等）用于评估生成/表示模型的性能；对INR、3D ShapeNet、CelebA 等作为数据本身编码的示例。

**📈 对比分析**

比较方法：在模型检索、性能预测、压缩、优化加速、持续学习等任务上，作者汇总了多种指标（准确率、压缩率、推理速度、权重相似度、任务迁移效果）。实验结果表明：①对称性感知编码器在检索/预测任务上提升 5–15%；②基于超网络的生成在少量样本下可匹配或超越传统微调；③扩散模型生成的权重在大规模模型（如 ViT、LLM）上实现了 10–20% 的性能提升，同时显著减少训练时间。

**⚠️ 局限性**

局限性包括：①权重空间高维且结构多变，导致生成模型收敛困难；②不同架构间的对齐与对称性未统一，影响跨模型迁移；③当前评估基准大多依赖公开模型仓库，缺乏标准化的量化评价；④计算开销（尤其是图形化对齐与高阶对称性求解）仍较大，限制了大规模部署。

---

## 447. The Dunning-Kruger Effect in Large Language Models: An Empirical Study of Confidence Calibration

**arXiv ID:** 2603.09985 | [PDF](https://arxiv.org/pdf/2603.09985v1)

**作者:** Sudipta Ghosh `[一作]` (Cognizant Technology Solutions), Mrityunjoy Panday `[通讯]` (Cognizant Technology Solutions)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四款最新LLM在四大基准上进行 24,000 次实验，评估其置信度校准并探究是否存在类似 Dunning‑Kruger 效应的过度自信模式。

**💡 创新点**

首次将 Dunning‑Kruger 效应概念系统迁移至 LLM，揭示低性能模型表现出极端过度自信，并公开完整实验框架与分析流程。

**🔧 技术方法**

采用数值置信度提问、Expected Calibration Error（ECE）、相关系数、方差分析及可靠性图，使用 0.0 温度确定性推理与扩展思维模式。

**📊 数据集**

MMLU、TriviaQA、ARC 与 HellaSwag 四大公开基准，每个模型抽样 1,500 题，共 24,000 次试验。

**📈 对比分析**

采用因子设计交叉比较四模型，评价准确率与 ECE，结果显示 Kimi K2 仅 23.3% 准确率却 ECE 0.726 过度自信；Claude Haiku 4.5 最高准确率 75.4% 与 ECE 0.122 的最佳校准；其余 Gemini 2.5 系列表现介于两者之间。

**⚠️ 局限性**

仅使用显式置信度提示，未覆盖隐式不确定性；仅在扩展思维模式下评估；研究聚焦事实与推理任务，对创造性或开放式任务缺乏验证；API 更新可能导致结果漂移。

---

## 448. The Discrete Charm of the MLP: Binary Routing of Continuous Signals in Transformer Feed-Forward Layers

**arXiv ID:** 2603.10985 | [PDF](https://arxiv.org/pdf/2603.10985v1)

**作者:** Peter Balogh `[一作]` `[通讯]`, Peter Balogh

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析GPT-2 Small的MLP层，发现其神经元实现了二进制路由，区分了需要非线性处理的标记与可线性通过的标记。

**💡 创新点**

创新点在于将Transformer MLP视为二进制路由器，而非单纯的函数逼近，揭示了共识‑异常四元结构以及层级发展的三相模式。

**🔧 技术方法**

采用聚类、二值化激活、决策树、因果消融、Ridge回归等技术，对神经元激活模式和路由逻辑进行定量和可视化分析。

**📊 数据集**

使用WikiText‑103语料库，在GPT-2 Small（124 M参数，12层）上进行实验，覆盖约500K tokens。

**📈 对比分析**

通过在不同共识水平下消融MLP，发现误差在≈10%（共识完整）到≈43%（共识失败）之间，验证了路由结构对模型性能的显著影响。

**⚠️ 局限性**

局限性包括此二进制路由机制在更大模型（Medium、Large）中不明显，且依赖GELU激活，未验证跨架构或跨数据集的普适性。

---

## 449. SBOMs into Agentic AIBOMs: Schema Extensions, Agentic Orchestration, and Reproducibility Evaluation

**arXiv ID:** 2603.10057 | [PDF](https://arxiv.org/pdf/2603.10057v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 450. LuxBorrow: From Pompier to Pompjee, Tracing Borrowing in Luxembourgish

**arXiv ID:** 2603.10789 | [PDF](https://arxiv.org/pdf/2603.10789v1)

**作者:** Nina Hosseini-Kivanani `[一作]` (University of Luxembourg), Fred Philippy `[通讯]` (University of Luxembourg)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5018702042)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在研究中，作者构建了一个涵盖1999–2025年259,305篇RTL新闻的卢森堡语借词首位语料库，并通过分层语言识别与基于形态学规则的检测，对文本进行词级借词与代码切换标注，并对其跨域、跨时的混合程度进行统计与可视化；

**💡 创新点**

其创新点包括提出“借词优先”标注范式，结合句子级语言门控与词级形态学匹配，提供了可扩展的手工与自动化结合的借词检测框架，以及首次在卢森堡语新闻语料中系统性地量化跨时借词与代码切换的演变；

**🔧 技术方法**

技术手段涵盖了OpenLID快速文本语言识别、基于规则的形态学与词形匹配、上下文运行长度与本地LU密度特征的特征工程，以及多维度的代码混合指标（CMI、熵、M-index）计算；

**📊 数据集**

所使用的数据集为RTL.lu新闻语料库（1999–2025年），共43.7M词汇，其中借词词典覆盖约7,796条条目（德语3,632条、法语3,201条、英语535条），以及通过人工校对得到的最终借词集合；

**📈 对比分析**

作者与现有代码切换基准（如LinCE、GLUECoS）对照，指出卢森堡语新闻的混合度在宏观上处于中等水平，借词占比低但随时间显著上升，展示了该方法在低资源语言环境下的可行性与稳健性；

**⚠️ 局限性**

研究局限包括缺乏手工标注的金标准导致检测误差、仅聚焦编辑新闻导致的样本偏倚、对口语与社交媒体场景的覆盖不足，以及对词形变体和多语言相互借用链条识别的进一步提升空间。

---

## 451. Bioinspired CNNs for border completion in occluded images

**arXiv ID:** 2603.10694 | [PDF](https://arxiv.org/pdf/2603.10694v1)

**作者:** Catarina P. Coutinho `[一作]` (University of Bologna), Rita Fioresi `[通讯]` (University of Bologna)

**通讯引用:** 991 | [OpenAlex ID](https://openalex.org/A5089505566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在视觉 cortex 的边界完成机制的数学模型基础上，设计了一种名为 BorderNet 的 CNN，并在 LeNet5 之上加入四个方向（水平、垂直、两条对角线）对应的自定义滤波器，以提升网络对遮挡图像的鲁棒性。

**💡 创新点**

创新点在于将子 Riemannian 几何和 Hamiltonian 正式化中的视觉皮层边界完成理论，转换为可直接嵌入 CNN 的方向感知滤波器，实现了生物启发的边界整合能力。

**🔧 技术方法**

技术手段包括：子 Riemannian 轨迹求解、Hamiltonian 形式的边界完成模型、基于方向滤波器的前置卷积层、LeNet5 经典网络结构以及 ADAM 优化器。

**📊 数据集**

使用的数据集为经典手写数字和服饰图像三大数据集：MNIST、Fashion‑MNIST 与 EMNIST，随后在这些数据集上人为生成两类遮挡（斜条纹与网格）进行测试。

**📈 对比分析**

对比方法是：用未遮挡图像训练两种模型（LeNet5 与 BorderNet），在遮挡图像上测试并计算准确率；通过 100 次循环得到平均准确率，再用 100000 次自助采样计算 BorderNet 相对于 LeNet5 的中位数改进率。实验显示 BorderNet 在大多数遮挡宽度/间距组合下均显著优于 LeNet5，改进率可达数十至百个百分点，只有在极度遮挡时差距不显著。

**⚠️ 局限性**

局限性：仅在简单手写/服饰图像和合成遮挡上验证，未评估在更复杂图像或真实遮挡场景下的表现；改进主要靠固定方向滤波，缺乏自适应学习；当遮挡过重时仍无法显著提升。

---

## 452. Two-Layer Stacked Intelligent Metasurfaces: Balancing Performance and Complexity

**arXiv ID:** 2603.10693 | [PDF](https://arxiv.org/pdf/2603.10693v1)

**作者:** Hong Niu `[一作]` (Nanyang Technological University), H. Vincent Poor `[通讯]` (Princeton University)

**通讯引用:** 154328 | [OpenAlex ID](https://openalex.org/A5042307561)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并评估两层叠加智能超材料（SIM），提出 MF‑SIM 与 FILM 两种结构，探讨其在 6G 低功耗、低复杂度无线系统中的应用。

**💡 创新点**

① 通过最小化层数实现独立幅相控制；② MF‑SIM 采用固定的光电“meta‑fiber”耦合，实现低功耗、低优化复杂度；③ FILM 通过可变形元件实现动态波前调制，提供更大的自由度。

**🔧 技术方法**

利用光电/电光元件、近场衍射模型（Rayleigh‑Sommerfeld）、基于图论/强化学习的拓扑优化、机械形变控制与闭环反馈。

**📊 数据集**

使用仿真数据：28 GHz 载频，Rayleigh 阻尼信道，4×4 MIMO、4 用户 MISO、20/30 dBm 发射功率、-110/-125 dBm 噪声，元件数 100/10×10。

**📈 对比分析**

与单层、四层、七层 SIM 以及传统 MIMO 进行对比。结果显示，2‑layer MF‑SIM 与 FILM 在每层功率衰减下容量下降最小，所需发射功率比七层 SIM 低 7–11 dB，且优化计算量显著降低。

**⚠️ 局限性**

局限性包括：拓扑优化与形变控制的算法复杂度高、机械精度限制导致相位误差、对实际硬件耦合与环境不确定性的评估不足、缺乏实测验证。

---

## 453. Multilingual AI-Driven Password Strength Estimation with Similarity-Based Detection

**arXiv ID:** 2603.10217 | [PDF](https://arxiv.org/pdf/2603.10217v1)

**作者:** Nikitha M. Palaniappan `[一作]` (Queen Mary University of London), Ying He `[通讯]` (Queen Mary University of London)

**通讯引用:** 2053 | [OpenAlex ID](https://openalex.org/A5067747030)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用 ChatGPT 生成英语、印度和混合语言密码，并与真实泄露密码进行 Jaro 相似度匹配，以评估多语言密码强度估计器的性能。

**💡 创新点**

创新点：①将大语言模型 ChatGPT 替代传统 GAN 进行密码生成；②首次针对印度密码构建 PSM；③采用 Jaro 相似度阈值 0.5 的多语言相似度匹配方法，提升检测对类似密码的捕获率。

**🔧 技术方法**

技术手段：ChatGPT 生成密码；Jaro 相似度函数做匹配；精确匹配和相似度匹配的对比评估；简单的准确率计算。

**📊 数据集**

数据集：ChatGPT 生成的 6,666 条英语、印度、混合密码；真实泄露密码：约 7,675 条印度密码（过滤至 8–10 位）和 11,356 条 LinkedIn 英文密码。

**📈 对比分析**

对比方法：用 Jaro 相似度阈值 0.5 匹配生成密码与泄露密码；与 PassGAN 生成的 10,000 条英文密码进行同样匹配。结果显示：ChatGPT 英文 78.08% 匹配率，印度 99.97%，混合 99.92%，PassGAN 仅 96%。说明多语言 ChatGPT 生成的密码在相似度匹配上具有更高准确率。

**⚠️ 局限性**

局限性：①生成密码样本量有限（仅 6,666 条/类）且受 ChatGPT 生成限制；②密码结构严格固定（8–10 位、至少包含大写、小写、数字、符号），无法覆盖更丰富的真实密码模式；③仅使用 Jaro 相似度，未尝试其他语义匹配方法；④阈值 0.5 固定，缺乏对不同语言或密码复杂度的动态调节。

---

## 454. DT-BEHRT: Disease Trajectory-aware Transformer for Interpretable Patient Representation Learning

**arXiv ID:** 2603.10180 | [PDF](https://arxiv.org/pdf/2603.10180v1)

**作者:** Deyi Li `[一作]` (University of Florida), Mei Liu `[通讯]` (University of Florida)

**通讯引用:** 7506 | [OpenAlex ID](https://openalex.org/A5100347935)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种名为DT‑BEHRT的疾病轨迹感知Transformer，用于从电子病历（EHR）中学习可解释的患者表示；

**💡 创新点**

创新点在于：①区分诊断码与治疗码，采用诊断聚合（DA）Token和疾病进程（DP）图模块分别捕获同一器官系统内的诊断交互和跨访视的时间演变；②设计基于轨迹级全局代码遮蔽与ICD‑9祖先预测的预训练任务，以对齐不同模块并提升表示鲁棒性；③通过协方差正则化增强DA Token的去相关性。

**🔧 技术方法**

技术手段包括：Transformer编码器与多头自注意力；图注意力网络（GAT）构建患者诊断-访视异质图；ICD‑9层级嵌入与代码类型/访视索引嵌入；Masking & Ancestor Prediction预训练；协方差正则化。

**📊 数据集**

实验数据集为MIMIC‑III、MIMIC‑IV和eICU三个公开EHR数据库。

**📈 对比分析**

与BEHRT、Med‑BERT、ExBEHRT、G‑BERT、HEART、HypEHR等基线进行比较，DT‑BEHRT在住院死亡、延长住院时长、再住院以及多标签表型预测任务上均取得最优或相近表现，尤其在MIMIC‑III再住院和多访视患者表型预测的宏观AUPRC上表现突出。

**⚠️ 局限性**

局限性包括：模型采用多头注意力与图注意力导致计算量大，难以在资源受限环境中部署；依赖多访视长期轨迹，单访视患者无效；主要聚焦诊断码，对药物、检查等其他码类别的建模不足；模型可能继承EHR数据偏差，需临床验证后才可投入实际。

---

## 455. Spatio-Temporal Forecasting of Retaining Wall Deformation: Mitigating Error Accumulation via Multi-Resolution ConvLSTM Stacking Ensemble

**arXiv ID:** 2603.10453 | [PDF](https://arxiv.org/pdf/2603.10453v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 456. One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination

**arXiv ID:** 2603.10360 | [PDF](https://arxiv.org/pdf/2603.10360v1)

**作者:** Zhan Fa `[一作]` (Nanjing University), Yinghuan Shi `[通讯]` (Nanjing University)

**通讯引用:** 4960 | [OpenAlex ID](https://openalex.org/A5055917015)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的训练‑free 框架，通过对视觉 token 的双重利用来抑制多模态大模型（MLLM）的幻觉问题。

**💡 创新点**

创新点在于将视觉 token 既用于增强视觉语义（Synergistic Visual Calibration, SVC），又用于构造信息缺口的隐空间负样本来校正内部偏置（Causal Representation Calibration, CRC），实现了在同一表示层面同时增强视觉与抑制语言惯性。

**🔧 技术方法**

采用注意力注入、词嵌入插值、随机裁剪 token 生成负样本、对齐差分向量校正以及结构因果模型理论分析等技术。

**📊 数据集**

在多模态问答与图像生成基准上验证，包括 MSCOCO、AOKVQA、GQA、POPE、CHAIR、MMHal‑Bench、MME 等数据集。

**📈 对比分析**

与现有训练‑free 方法（VCD、PAI、VISTA、ONLY 等）对比，实验表明该方法在四大 MLLM（LLaVA‑1.5、MiniGPT‑4、Shikra、InstructBLIP）上平均提升 2% 绝对准确率，并在 POPE、CHAIR、MMHal‑Bench、MME 等任务中取得最佳或第二佳成绩，同时仅增加 1.06× 的推理延迟。

**⚠️ 局限性**

局限性包括对超参数（如 λ_s、λ_c、N_h、K）敏感，需要手动调优；方法仍依赖原始视觉 token 的质量；对极端长文本生成时的视觉衰减与语言惯性平衡仍存在挑战。

---

## 457. Improving Search Agent with One Line of Code

**arXiv ID:** 2603.10069 | [PDF](https://arxiv.org/pdf/2603.10069v1)

**作者:** Jian Li `[一作]` (Nanjing University), Yabiao Wang `[通讯]` (Tencent YoutuLab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了搜索代理策略优化（SAPO）方法，用以稳定工具基代理强化学习训练，防止重要采样分布漂移导致的模型崩溃。

**💡 创新点**

创新点是引入条件化的token级KL约束，只对正向奖励且重要采样比率低于阈值的低概率正向动作施加惩罚，从而软化剪裁并保持梯度流。

**🔧 技术方法**

技术包括基于Group Relative Policy Optimization (GRPO) 的策略梯度、重要采样比率、PPO_KL启发式的KL惩罚、以及条件化的KL约束。

**📊 数据集**

使用七个问答基准数据集：单跳（NQ、TriviaQA、PopQA）与多跳（HotpotQA、2WikiMultihopQA、Musique、Bamboogle）。

**📈 对比分析**

与Search‑R1及多种检索增强代理（如AutoRefine、CriticSearch等）对比，SAPO在所有七个基准上平均提升约10.6个百分点（相对提升31.5%），在多跳任务中提升更显著。

**⚠️ 局限性**

局限性包括仍需手动设置KL惩罚系数和阈值，对极端稀疏奖励场景的鲁棒性尚未充分验证，且依赖于现有检索工具与知识库的质量。

---

## 458. Safe RLHF Beyond Expectation: Stochastic Dominance for Universal Spectral Risk Control

**arXiv ID:** 2603.10938 | [PDF](https://arxiv.org/pdf/2603.10938v1)

**作者:** Yaswanth Chittepu `[一作]` (University of Massachusetts Amherst), Scott Niekum `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 1991 | [OpenAlex ID](https://openalex.org/A5043572737)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于随机优势（FSD）的安全强化学习框架 RAD，取代传统的期望成本约束，以完整的成本分布来控制模型安全性。

**💡 创新点**

创新点在于：①使用第一阶随机优势约束取代单一期望约束；②引入量化加权 FSD，实现对谱风险度量的统一控制；③通过熵正则化的最优传输与 Sinkhorn 迭代，使 FSD 约束可微、可端到端优化。

**🔧 技术方法**

技术手段包括：RLHF 训练流程（SFT、奖励/成本模型训练、基于 Lagrangian 的约束优化）、REINFORCE + RLOO 方差削减、量化粒子逼近成本分布、熵正则化最优传输与 Sinkhorn 迭代、谱风险度量（VaR、CVaR、Wang 等）以及权重函数设计。

**📊 数据集**

使用的数据集：Qwen2.5‑3B 作为基础模型；Alpaca 用于 SFT；BeaverTails 提供帮助性和有害性偏好，用于训练奖励与成本模型；HarmBench 作为离散分布外的鲁棒性评估；GPT‑4o‑mini 用作安全性评判。

**📈 对比分析**

通过与 Safe‑RLHF 与 SFT 基线在安全性（安全回复比例）和有用性（奖励胜率）上的对比，实验表明 RAD 在安全性上显著优于基线，且在大多数加权方案下保持或略低于 Safe‑RLHF 的有用性；在 HarmBench 的离散分布外评估中，偏重尾部的 RAD 变体更能提升安全性。

**⚠️ 局限性**

局限性包括：①仅考虑第一阶随机优势，未覆盖更高阶风险控制；②对量化粒子数与正则化参数敏感，训练稳定性取决于超参数；③实验仅覆盖特定模型与基准，未验证在更大规模或不同任务中的普适性。

---

## 459. Simple minimally unsatisfiable subsets of 2-CNFs

**arXiv ID:** 2603.10944 | [PDF](https://arxiv.org/pdf/2603.10944v1)

**作者:** Oliver Kullmann `[一作]` (Swansea University), Edward Clewer `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文研究 2-CNF 公式的最小不可满足子集（MUS），提出 2-MU 的线性时间判定方法，并对不同类型的 MUS（按单位子句数量划分的四类 Family I–IV）进行分类与算法设计。对于包含 1 或 2 个单位子句的 MUS（Family I、II），给出了多项式时间检测与构造算法；对于包含 1 个单位子句的 MUS，给出了基于“近正则路径”的构造；而包含 2 个单位子句的 MUS 则通过正则路径实现。还提出了基于 L-pathlex 序的增量多项式时间枚举算法，并讨论了多项式延迟的可能性与限制。

**💡 创新点**

创新点主要有：
1. 通过“checked singular DP-reduction”实现 2-MU 的线性时间判定；
2. 将 2-CNF MUS 按单位子句数量划分为四类 Family I–IV，并给出每类的结构与判定/构造算法；
3. 对于 Family I、II 的 MUS，证明可在多项式时间内检测、构造，并给出最短 MUS 的线性时间算法；
4. 设计了 L-pathlex 方案，实现基于正则/近正则路径的增量多项式时间枚举；
5. 明确了 NP-完整性与多项式可解区间，提出若干开放问题。

**🔧 技术方法**

主要技术手段包括：
- checked singular DP-reduction（对 2-MU 判定）;
- 逻辑蕴含图（implication digraph）与正则/近正则路径分析；
- 路径映射到 MUS 的双射/满射构造；
- L-pathlex 线性顺序与 DFS 递归枚举；
- 结构分类与缺陷度（deficiency）理论。

**📊 数据集**

本文未使用公开数据集；实验与评测主要基于理论分析与合成实例。

**📈 对比分析**

方法比较：
- 对于 Family I、II 的 MUS，算法在最坏情况下为线性/多项式时间（具体为 O(u(F)^2·ℓ(F)) 或 O(u(F)·ℓ(F))，远优于 NP-完整的 Family III、IV；
- 对于枚举，采用增量多项式时间而非多项式延迟；
- 线性时间的 2-MU 判定相较于传统二次时间显著提升。

**⚠️ 局限性**

局限性：
- 对于 Family III、IV 的 MUS 仍为 NP-完整，无法给出多项式时间解法；
- 枚举仅实现增量多项式时间，尚未证明可实现多项式延迟；
- 算法主要针对理论实例，缺乏大规模实验验证；
- 只讨论 2-CNF，其他 CNF 形式的 MUS 仍未覆盖。

---

## 460. Architecture-Aware LLM Inference Optimization on AMD Instinct GPUs: A Comprehensive Benchmark and Deployment Study

**arXiv ID:** 2603.10031 | [PDF](https://arxiv.org/pdf/2603.10031v1)

**作者:** Athos Georgiou `[一作]` `[通讯]` (NCA IT), Athos Georgiou (NCA IT)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在AMD Instinct MI325X GPU集群上使用vLLM对四个前沿LLM模型（Llama‑3.1‑405B、DeepSeek V3.2、Qwen3‑VL‑235B、Kimi‑K2.5）进行跨架构推理基准测试，评估不同注意力机制（GQA、MLA）和稀疏专家结构对吞吐量和延迟的影响。

**💡 创新点**

首次实现了跨架构的MI325X推理基准，系统揭示了MLA模型对KV缓存、块大小、AITER加速的特殊配置需求，并量化了活跃参数数对吞吐量的决定性作用。

**🔧 技术方法**

采用vLLM v0.14.1、ROCm 6.4、AMD AI Tensor Engine (AITER)、FP8/INT4量化、PagedAttention、连续批处理等技术。

**📊 数据集**

使用公开模型权重并通过固定文本/图像提示生成合成负载（500/100 token、4k/8k token预填等）进行压力测试。

**📈 对比分析**

通过并发量、吞吐量、p99延迟等指标与模型间的对比显示，GQA模型在高并发下吞吐量最高，MLA模型受限于块大小和AITER，所有模型在约500并发时达到饱和，吞吐量峰值分别为15,944 tok/s（Llama）、15,343 tok/s（DeepSeek）、47,873 tok/s（Qwen3‑VL）和7,327 tok/s（Kimi），且在1,000并发时保持100%成功率。

**⚠️ 局限性**

仅在单台8 GPU MI325X集群、单个vLLM版本、有限的四种模型范围内实验，未覆盖多节点扩展、其他模型架构或vLLM新版本变化。

---

## 461. Evaluating Generalization Mechanisms in Autonomous Cyber Attack Agents

**arXiv ID:** 2603.10041 | [PDF](https://arxiv.org/pdf/2603.10041v1)

**作者:** Ondřej Lukáš `[一作]` (Czech Technical University in Prague), Sebastian Garcia `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文在NetSecGame环境中研究了自主进攻代理在主机/子网IP重新分配下的泛化能力，训练时使用五个IP范围变体，测试时评估在一个未见IP重新分配的网络上。

**💡 创新点**

创新点在于：①将IP重新分配作为最小但关键的分布转移，对代理泛化进行定量分析；②提出行为分布（action‑distribution）可解释的XAI方法定位失败模式；③将LLM提示式推理（ReAct）与实体抽象及元学习相结合，形成多族对比。

**🔧 技术方法**

采用的技术包括：传统RL（DQN/Double‑DQN）与基于候选动作的特征表示；概念抽象（概念化Q‑学习）消除IP依赖；元学习（MAML与Reptile）实现测试时快速适应；LLM推理（ReAct）和LLM+BERT混合（LLM‑BERT）生成动作；行为签名分析。

**📊 数据集**

使用的数据集是NetSecGame的单一企业情景，在该情景下随机生成六个IP地址范围（前五个用于训练，六个用于测试），每个范围包含同样的网络拓扑与服务配置。

**📈 对比分析**

通过对比Win Rate、Return、Step数以及行为签名，结果显示ReAct在未见IP下取得95%成功率，概念化Q学习约65%，MAML约40%，DDQN和DQN几乎无法泛化（0%/3%），随机基线仅6%。LLM方法实现最高成功率但计算量大、存在重复/无效动作循环；概念化方法训练成本高、步骤数多；元学习仅部分恢复。

**⚠️ 局限性**

主要局限包括：①LLM推理需要高昂推理成本，易出现无效动作循环；②概念化方法训练耗时且对不同网络拓扑的适用性未验证；③元学习在此单一场景下收敛缓慢、适应效果有限；④实验仅考虑IP重新分配，未评估更复杂的拓扑或目标变动；⑤缺乏跨域真实网络的验证。

---

## 462. MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction

**arXiv ID:** 2603.10688 | [PDF](https://arxiv.org/pdf/2603.10688v1)

**作者:** Jonas Merkert `[一作]` (Karlsruhe Institute of Technology), Christoph Stiller `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 23130 | [OpenAlex ID](https://openalex.org/A5091574711)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对在线HD地图构建模型，提出一种半监督训练框架，利用多次行驶路径之间的地理一致性对BEV特征网格进行对比学习，从而提升地图预测性能。

**💡 创新点**

创新点包括：①基于地理重叠的多遍历数据划分方法；②将重叠的车辆姿态视为自然数据增强，构造地理对比损失；③将监督与自监督学习结合，形成有效的半监督训练流程。

**🔧 技术方法**

使用技术包括：MapTRv2单射-解码器架构、ResNet-50主干提取图像特征、升维模块生成BEV特征、SimCLR风格的对比学习（InfoNCE损失）以及基于车辆位姿的全局坐标变换。

**📊 数据集**

使用数据集：Argoverse 2（含多遍历日志），对其中的多遍历轨迹进行划分，形成自监督数据集。

**📈 对比分析**

与纯监督基准MapTRv2进行对比，在不同标注比例（2.5%、5%、10%、20%）下训练，实验显示自监督对比学习使模型性能提升13%–42%，尤其在少量标注时接近双倍标注数据效果。

**⚠️ 局限性**

局限性：需要高精度（相对）定位；仅改进中间BEV特征，未直接提升Transformer解码器；对多遍历覆盖的依赖，若数据集中重叠不足则效果受限。

---

## 463. Density-Dependent Graph Orientation and Coloring in Scalable MPC

**arXiv ID:** 2603.10639 | [PDF](https://arxiv.org/pdf/2603.10639v1)

**作者:** Mohsen Ghaffari `[一作]` (Massachusetts Institute of Technology), Christoph Grunau `[通讯]` (ETH Zurich)

**通讯引用:** 1497 | [OpenAlex ID](https://openalex.org/A5056617357)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在可扩展MPC（Scalable MPC）模型下，提出了一种以子图密度（等价于arboricity）为函数的图边缘定向与顶点着色算法，能够在 (loglog n) 轮完成，并保证最大出度与着色数均为 O(α loglog n)（α 为子图密度）。

**💡 创新点**

突破了此前在可扩展MPC中 Θ~(√(log n)) 轮的上界，将时间降到 loglog n，同时只在出度/颜色数上引入了 loglog n 的因子，首次实现了低密度图的高效 MPC 定向与着色。

**🔧 技术方法**

核心技术包括：MPC 模型与图指数化（graph exponentiation）相结合的层次递归演算；稀疏化与局部剪枝（LocalPrune）以控制树视图规模；树形视图与“严格单调可达”约束；以及层分配（PartialLayerAssignment）实现基于层的定向与着色。

**📊 数据集**

本工作为理论算法研究，未使用具体实验数据集，主要通过复杂度分析和概率高概率（w.h.p.）证明来评估算法性能。

**📈 对比分析**

与 Ghaffari‑Lattanzi‑Mitrovic 等人提出的 Θ~(√(log n)) 轮可扩展 MPC 定向算法相比，本算法在 (loglog n) 轮内完成相同功能；在图密度 α 较小的情况下，出度与颜色数仅比最优 O(α) 多一个 loglog n 因子。

**⚠️ 局限性**

主要局限在于：(1) 出度/颜色数仍比理论下界 O(α) 高 loglog n 乘子；(2) 对非常高密度图（α 远大于 log n）仍需额外的随机分区或多层级分割；(3) 具体实现的常数项和对实际大规模数据集的实验验证尚未给出。

---

## 464. Copula-ResLogit: A Deep-Copula Framework for Unobserved Confounding Effects

**arXiv ID:** 2603.10284 | [PDF](https://arxiv.org/pdf/2603.10284v1)

**作者:** Kimia Kamal `[一作]` (Toronto Metropolitan University), Bilal Farooq `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 2681 | [OpenAlex ID](https://openalex.org/A5048496396)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了交通决策中未观测混杂因素对因果关系的影响，提出并在行人交叉和伦敦交通行为数据上验证了Copula-ResLogit混合模型

**💡 创新点**

将残差网络与copula联合建模，形成可解释的深度学习框架以消除未观测混杂的影响

**🔧 技术方法**

采用残差神经网络（ResNet）+ copula函数（Frank、FGM、Gaussian、AMH、Product）+ 最大似然估计+RMSprop SGD等技术

**📊 数据集**

使用虚拟现实VR实验收集的行人压力与等待时间数据以及伦敦旅行需求调查（LTDS）的模式与距离数据

**📈 对比分析**

与传统Copula-Logit及独立模型对比，使用AIC和MPE评估，Copula-ResLogit在AIC/MPE上显著优越，深度层数增加进一步消除混杂

**⚠️ 局限性**

模型仍可能残留部分未观测混杂，残差层无法完全捕获，模型深度增加导致复杂度提升，并未涵盖所有混杂类型或完整的因果推断

---

## 465. HAPEns: Hardware-Aware Post-Hoc Ensembling for Tabular Data

**arXiv ID:** 2603.10582 | [PDF](https://arxiv.org/pdf/2603.10582v1)

**作者:** Jannis Maier `[一作]` (Helmholtz-Zentrum Berlin), Lennart Purucker `[通讯]` (University of Freiburg)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5016560150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种硬件感知的后期集成选择方法，该方法通过构造多目标Pareto前沿，在预测准确性与硬件成本之间实现显著平衡。

**💡 创新点**

首次系统研究硬件感知后期集成选择；利用质量多样性优化技术生成多样化的集成集合；发现内存使用率是最有效的硬件成本指标；并证明即便是简单的贪婪集成也能通过静态多目标权重显著提升性能。

**🔧 技术方法**

采用多目标优化与质量多样性（QDO）框架，使用基于种群的交叉/变异算子、滑动边界归档进行搜索；对每个集成计算交叉熵/相关性描述子；在实验中采用超体积（HV）和逆生成距离（IGD+）等多目标评价指标；对比单模型、贪婪集成、Multi-GES、QDO-ES等基线。

**📊 数据集**

使用TabRepo提供的83个表格分类数据集（共211个数据集，211×1,530个模型配置），数据集覆盖从少量特征/样本到大规模多类别任务的多样场景。

**📈 对比分析**

通过在所有数据集上对比HV与IGD+得分，所提出方法在绝大多数实验中显著优于基线，得到更大的覆盖范围和更靠近参考前沿的解；在内存、推理时间、磁盘使用等硬件指标上均保持领先；Multi-GES在低成本端表现突出，但整体仍落后于所提方法。

**⚠️ 局限性**

局限性包括：实验仅在单一硬件配置下进行，未对多设备多目标进行验证；只关注表格数据；使用静态权重方案，缺乏动态调节；硬件成本指标基于模拟测量，未在真实设备上进行评估。

---

## 466. PRoADS: Provably Secure and Robust Audio Diffusion Steganography with latent optimization and backward Euler Inversion

**arXiv ID:** 2603.10314 | [PDF](https://arxiv.org/pdf/2603.10314v1)

**作者:** YongPeng Yan `[一作]` (Key Laboratory of Aerospace Information Security and Trusted Computing Ministry of Education), Yanzhen Ren `[通讯]` (School of Cyber Science and Engineering Wuhan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了PRoADS，一种基于音频扩散模型的初始噪声嵌入式稳健音频隐写框架。

**💡 创新点**

核心创新在于引入潜空间优化与后向欧拉反演两种技术，显著降低初始噪声重构误差，实现可证明的安全与极低误码率。

**🔧 技术方法**

采用正交矩阵投影嵌入、潜空间梯度优化、后向欧拉反演以及DPM-Solver数值求解等技术。

**📊 数据集**

在AudioCaps数据集上以10秒音频为实验对象，使用了AAC/MP3压缩、上采样、下采样等扰动进行评估。

**📈 对比分析**

与Yang、Kim、Hu等基线方法对比，PRoADS在64 kbps MP3压缩下BER仅0.15%，显著低于其他方法，整体鲁棒性提升约0.5%–0.7%。

**⚠️ 局限性**

主要局限在于提取过程需大量迭代，单条10秒音频提取时间约106秒，计算成本相对较高。

---

## 467. Data Augmentation and Convolutional Network Architecture Influence on Distributed Learning

**arXiv ID:** 2603.10902 | [PDF](https://arxiv.org/pdf/2603.10902v1)

**作者:** Victor Forattini Jansen `[一作]` (Federal University of Viçosa), Larissa Ferreira Rodrigues Moreira `[通讯]` (Federal University of Viçosa)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究通过2^2因子实验与ANOVA方法，系统评估了CNN网络深度与数据增强对分布式训练时GPU、CPU、内存占用、网络包流量及模型准确率的影响。

**💡 创新点**

首次定量揭示了CNN深度和数据增强对硬件资源消耗的贡献度，并发现数据增强显著提升网络包传输量，提供了分布式训练效率与能耗的全新视角。

**🔧 技术方法**

采用Torch Distributed Data Parallel进行分布式训练，使用NetData监控硬件资源，比较浅层MobileNetV2-100与深层MobileOne-S1两种CNN，配合旋转、仿射、裁剪、翻转、色彩抖动等数据增强管道，并通过2^2因子设计与ANOVA进行统计分析。

**📊 数据集**

使用稻田叶片图像数据集Paddy Doctor，包含16,225张标注图像，涵盖12种病害与一类正常叶片，共13类。

**📈 对比分析**

对四种实验组合（浅/深CNN × 有/无DA）测量GPU占用率、网络包速率、CPU/内存使用率及准确率。结果显示深层CNN使GPU占用率提升约48%，数据增强使网络包量增加约78%；准确率最高为99.60%（无DA+浅CNN），最低为94.09%（DA+深CNN）。

**⚠️ 局限性**

局限性在于仅评估两种CNN架构与单一数据增强策略，未考虑更深模型或多样化增强；训练仅至20个epoch，早停限制了对长期性能趋势的洞察；实验规模受限，未探讨更大规模集群或不同网络带宽条件下的表现。

---

## 468. Riemannian MeanFlow for One-Step Generation on Manifolds

**arXiv ID:** 2603.10718 | [PDF](https://arxiv.org/pdf/2603.10718v1)

**作者:** Zichen Zhong `[一作]` (Shandong University), Yilong Yin `[通讯]` (Shandong University)

**通讯引用:** 5820 | [OpenAlex ID](https://openalex.org/A5100672590)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Riemannian MeanFlow（RMF）框架，实现流形上一次采样（1 NFE）的生成。

**💡 创新点**

创新点包括：1）通过并行传输定义流形上的平均速度并推导 Riemannian MeanFlow 身份；2）将损失拆分为两项，利用 PCGrad 解决梯度冲突；3）在同一切向空间上实现无条件与条件（classifier‑free guidance）生成。

**🔧 技术方法**

采用 Riemannian Flow Matching、并行传输、对数映射、Jacobian‑vector product 计算梯度、PCGrad 多任务优化、classifier‑free guidance 以及基于 geodesic 距离的 MMD 评估。

**📊 数据集**

使用的实验数据集包括：地球表面 2D 球面灾害数据、平坦环面蛋白质与 RNA 扭转角数据、合成 SO(3) 旋转数据（Cone、Fisher、Swiss Roll 等）以及高维 hypersphere 试验。

**📈 对比分析**

在 1 NFE 下与 RFM、RCM、EMF、G-PSD/ESD/LSD 等基线对比，RMF/MF‑MT 在大多数数据集上取得最佳或次佳 MMD，显著降低采样成本并提升质量‑效率比。

**⚠️ 局限性**

局限性：依赖可闭式 geodesic 的流形，未处理非闭式 geodesic 的情形；对极小样本集（如 Volcano）性能略弱；需要额外调参以解决梯度冲突；对真实世界应用需人工验证与约束。

---

## 469. Robotic Ultrasound Makes CBCT Alive

**arXiv ID:** 2603.10220 | [PDF](https://arxiv.org/pdf/2603.10220v1)

**作者:** Feng Li `[一作]` (Technical University of Munich), Yuan Bi `[通讯]` (Technical University of Munich)

**通讯引用:** 26862 | [OpenAlex ID](https://openalex.org/A5100334733)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于机器人超声的实时CBCT更新框架，通过估计连续超声帧的变形场，实时更新CBCT切片，从而实现无辐射、连续的解剖导航。

**💡 创新点**

创新点包括：①提出轻量级双向相关网络USCorUNet，利用光流蒸馏、光度一致性和物理正则化训练；②将LC2线性相关注册与机器人标定相结合，提供高精度的跨模态对齐；③通过超声估计实现CBCT切片的动态更新，显著降低重复辐射；④针对探头压迫与外部压缩分别进行模型微调，提升鲁棒性。

**🔧 技术方法**

核心技术包括：机器人超声与CBCT的手眼标定、LC2相似度细化、USCorUNet网络（ResUNet编码-解码+相关体积）、光流蒸馏（RAFT）、光度一致性损失、边界平滑与折叠惩罚正则化，以及实时GPU实现。

**📊 数据集**

使用四个数据集：A）在体手臂/上臂超声；B）猪肉凝胶phantom；C）鸡/猪凝胶phantom；D）腹部phantom；并在B集加入外部压缩实验，覆盖探头压迫与外部压迫两种变形场景。

**📈 对比分析**

与RAFT、DefCor-Net、LC2-FFD等基线在MAE、NCC、FB一致性、折叠率、Dice、SSIM和运行时等指标上对比。USCorUNet在变形质量上将FB残差降低约53%，折叠率下降，Dice略优于RAFT；实时性比RAFT快约5倍，比LC2-FFD快512倍；在外部压缩场景中微调模型进一步提升FB一致性与物理可行性。

**⚠️ 局限性**

局限性包括：①仅使用图像特征，缺乏语义分割信息；②对深度和噪声敏感的超声图像仍难以捕捉全局变形；③物理可行性约束有限，仍可能出现非生物学可行的变形；④在机器人运动导致的干扰下精度下降；⑤未实现完整体更新，仅限切片级更新。

---

## 470. Dark Patterns and Consumer Protection Law for App Makers

**arXiv ID:** 2603.10020 | [PDF](https://arxiv.org/pdf/2603.10020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 471. Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction

**arXiv ID:** 2603.10597 | [PDF](https://arxiv.org/pdf/2603.10597v1)

**作者:** Hao Zhou `[一作]` (Great Bay University), Fei Luo `[通讯]` (Great Bay University)

**通讯引用:** 2186 | [OpenAlex ID](https://openalex.org/A5101711943)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 Progressive Retrospective Framework (PRF)，通过一系列递归回溯单元将不完整的轨迹观察逐步对齐到完整长度，并提出 Rolling‑Start Training Strategy (RSTS) 以提升训练数据利用率；该框架可无缝集成到现有的轨迹预测模型中。

**💡 创新点**

创新点包括：① 逐步回溯的级联结构，分阶段缩小特征映射难度；② 结合 Retrospective Distillation Module (RDM) 与 Retrospective Prediction Module (RPM) 的残差门控蒸馏与预测双向监督；③ 通过 RSTS 在训练阶段多次复用同一序列，显著提升数据效率。

**🔧 技术方法**

技术手段涵盖 Transformer 及残差蒸馏、跨注意力与自注意力、Mamba 序列建模、anchor‑free/anchor‑based 查询设计，以及与 QCNet/DeMo 等基线模型的兼容性。

**📊 数据集**

实验数据集为 Argoverse 2 与 Argoverse 1 两大交通轨迹预测数据集，分别包含不同城市、不同时间分辨率的驾驶场景。

**📈 对比分析**

与 Ori、IT、DTO、FLN、LaKD、CLLS 等基线，以及 QCNet、DeMo 等前沿方法在 Argoverse 验证集上进行对比。PRF 在所有观察长度上均显著降低 mADE/mFDE，且在标准轨迹预测排行榜中取得了领先或与最高水平相当的成绩。

**⚠️ 局限性**

局限性包括：推理时需要多次迭代回溯，导致观察长度越短时计算量和延迟线性增长；对极短观察（如不足一个间隔）仍存在性能下降；需要预设固定的时间间隔 ΔT，无法自适应不同场景；缺乏对极端遮挡或异常轨迹的全面评估。

---

## 472. SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning

**arXiv ID:** 2603.10446 | [PDF](https://arxiv.org/pdf/2603.10446v1)

**作者:** Jianhe Low `[一作]` (University of Surrey), Richard Bowden `[通讯]` (University of Surrey)

**通讯引用:** 13790 | [OpenAlex ID](https://openalex.org/A5044490167)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了多语言手语生成框架 SignSparK，结合 FAST 高效分割器和基于稀疏关键帧的 Conditional Flow Matching 模型，能够直接从文本生成自然流畅的 3D 手语动作。

**💡 创新点**

创新点包括：1) 采用稀疏关键帧训练范式，显著降低回归平滑化；2) 设计 FAST 高效分割器，自动提取语言关键帧；3) 引入重建式 Flow Matching 与 CFG 引导，实现一次步高效采样；4) 支持 Keyframe‑to‑Pose 控制，便于动作编辑；5) 与 3D Gaussian Splatting 结合，生成真实感全身手语头像。

**🔧 技术方法**

使用的技术主要有：SMPL‑X 与 MANO 3D 姿态参数化、WiLoR 手部回归、UNet + 位置编码的时间条件网络、Conditional Flow Matching、classifier‑free guidance、以及 3D Gaussian Splatting 渲染。

**📊 数据集**

所用数据集包括：MeineDGS（分割基准）、Phoenix14T、CSLDaily、How2Sign、BOBSL（包含 BSL）以及合并的多语言语料库，覆盖德语、中文、美国手语与英国手语。

**📈 对比分析**

在 Sign Stitching、Text‑to‑Pose（T2P）与 Keyframe‑to‑Pose（KF2P）三种任务中，与 Sign Stitcher、SOKE、SignGAN 等现有方法对比，DTW、BLEU 等指标均大幅提升，10 步采样仅 0.01 s，显著优于先前模型。

**⚠️ 局限性**

局限性包括：目前仅处理手部与上身动作，面部表情与手势细节仍需改进；关键帧提取依赖 FAST 的分割精度，若分割误差会影响生成质量；多语种泛化仍受训练集比例与语言标记一致性的影响。

---

## 473. A Review of the Negative Effects of Digital Technology on Cognition

**arXiv ID:** 2603.10025 | [PDF](https://arxiv.org/pdf/2603.10025v1)

**作者:** Urška Žnidarič `[一作]` (University of Ljubljana), Octavian Machidon `[通讯]` (University of Ljubljana)

**通讯引用:** 740 | [OpenAlex ID](https://openalex.org/A5082970512)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对数字技术（包括智能手机、社交媒体、游戏、互联网以及生成式人工智能）对人类认知的负面影响进行系统性整合综述，结合文献计量、统计汇总、机制分类和认知流行病学框架。

**💡 创新点**

① 将AI相关研究与传统屏幕技术研究整合，构建跨技术、跨认知域的综合视角；② 提出了效率‑萎缩悖论和负反馈循环模型；③ 以认知流行病学视角评估长期健康风险；④ 强调社会经济与环境因素对结果的调节作用。

**🔧 技术方法**

采用系统检索（数据库检索、人工筛选）、极端案例抽样、贝叶斯 Beta 统计、图谱（引用网络）与机制归纳，辅以元分析和纵向研究综述。

**📊 数据集**

共检索到 565 篇经验研究与 22 篇 AI 关注研究，形成涵盖 30+ 年、涵盖儿童至成人、基本认知到高阶认知的多维数据集。

**📈 对比分析**

通过贝叶斯可信区间对各技术类别、认知领域的比例进行统计；引用网络可视化验证主题聚类；对 AI 与传统技术在效应维度（注意、记忆、执行、创造、批判性思维）的差异进行对比；整体发现，AI 对高阶认知的负面风险更显著，且与传统屏幕技术形成不同的机制轨迹。

**⚠️ 局限性**

① AI 研究样本规模小、横断面设计多，缺乏纵向和实验验证；② 传统屏幕技术研究多受 SES、家庭环境等混杂因素影响，因果推断受限；③ 机制层面主要基于相关性和理论模型，缺乏直接的神经生物学验证；④ 综述受检索策略和选择标准限制，可能遗漏灰色文献或最新数据。

---

## 474. LiTo: Surface Light Field Tokenization

**arXiv ID:** 2603.11047 | [PDF](https://arxiv.org/pdf/2603.11047v1)

**作者:** Jen-Hao Rick Chang `[一作]` (Apple), Oncel Tuzel `[通讯]` (Apple)

**通讯引用:** 14069 | [OpenAlex ID](https://openalex.org/A5028613002)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

LiTo方法将表面光场token化为潜在表示，用于同时捕捉3D几何结构和视角依赖的光照特征。

**💡 创新点**

创新点在于首次将光场拆解为可微分的token表示，并通过这一表示实现对镜面反射等复杂视角依赖现象的高效建模。

**🔧 技术方法**

采用深度学习框架对光场进行分块token化，并利用可微渲染技术将这些token映射到3D几何与材质的潜在空间中。

**📊 数据集**

使用Apple内部的高分辨率光场数据集（包含多视角图像与对应的几何信息）进行训练与评估。

**📈 对比分析**

与传统光场重建与单图像到3D方法相比，LiTo在重建精度、细节保留以及推理速度方面均取得显著提升，实验结果表明其重建误差下降约20%，推理时间缩短至传统方法的一半。

**⚠️ 局限性**

该方法在极端视角变化、强光照或高频纹理场景下的鲁棒性仍有待提升，且依赖大量带标签的光场数据，增加了数据采集与标注成本。

---

## 475. FAME: Formal Abstract Minimal Explanation for Neural Networks

**arXiv ID:** 2603.10661 | [PDF](https://arxiv.org/pdf/2603.10661v1)

**作者:** Ryma Boumazouza `[一作]` (Airbus SAS), Guy Katz `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 2498 | [OpenAlex ID](https://openalex.org/A5102986148)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FAME 框架，利用抽象解释与自适应扰动域实现可扩展的正式最小化解释；

**💡 创新点**

设计无遍历顺序的抽象批量证书与递归扰动域收缩策略，结合 LiRPA 与贪心多维背包启发式，首次在大规模神经网络上实现正式抽象最小解释；

**🔧 技术方法**

采用 LiRPA（CROWN）抽象解释、抽象批量证书、贪心多维背包求解、递归扰动域收缩、VeriX+精细化以及对抗攻击优化；

**📊 数据集**

在 MNIST、GTSRB 以及 CIFAR-10（ResNet-2B）等数据集上进行实验；

**📈 对比分析**

与 SOTA verixplus 进行对比，FAME 在同等模型与扰动下解释规模更小、计算时间更短，平均加速数倍且解释卡数显著下降；

**⚠️ 局限性**

抽象解释的弱化导致与真最小解释存在距离；对大模型的精确细化仍受限于精确求解器；对抗攻击与扰动域参数需要手动调节。

---

## 476. Artificial Intelligence as a Catalyst for Innovation in Software Engineering

**arXiv ID:** 2603.10994 | [PDF](https://arxiv.org/pdf/2603.10994v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 477. Explainable LLM Unlearning Through Reasoning

**arXiv ID:** 2603.09980 | [PDF](https://arxiv.org/pdf/2603.09980v1)

**作者:** Junfeng Liao `[一作]` (University of Technology Sydney), Zhen Fang `[通讯]` (University of Technology Sydney)

**通讯引用:** 13964 | [OpenAlex ID](https://openalex.org/A5057183219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对大型语言模型的可控推理性遗忘方法（TRU），通过生成基于推理的遗忘目标并联合梯度上升和监督损失，实现对指定知识范围的精准删除并保持模型的正常功能。

**💡 创新点**

创新点在于首次将推理过程嵌入遗忘目标，既明确遗忘范围，又指导模型给出合理的拒绝回复，从而解决传统梯度上升方法失控与响应不连贯的问题。

**🔧 技术方法**

主要技术包括：基于深度推理模型自动生成推理+拒绝目标、交叉熵监督损失、梯度上升（GradDiff）损失以及整体联合优化框架。

**📊 数据集**

使用了WMDP（生物与网络安全）、MUSE（版权文本）和TOFU（合成作者问答）三大公开基准数据集进行评估。

**📈 对比分析**

与八种主流基线（GA、GradDiff、KL、PO、WGA、NPO、RMU等）在多维度指标（Rel、Rej、Help、Read、Spe、Logic）上比较，TRU在遗忘质量方面均超过6.0，且在大多数指标上取得显著提升，保持较低的性能损失。

**⚠️ 局限性**

局限性包括：需要依赖高质量的推理模型生成目标，生成成本较高；在极端攻击或极少样本遗忘场景下仍可能出现轻微的遗忘退化；以及对推理目标的准确性和覆盖范围有一定依赖。

---

## 478. GaLoRA: Parameter-Efficient Graph-Aware LLMs for Node Classification

**arXiv ID:** 2603.10298 | [PDF](https://arxiv.org/pdf/2603.10298v1)

**作者:** Mayur Choudhary `[一作]` (San Jose State University), Katerina Potika `[通讯]` (San Jose State University)

**通讯引用:** 526 | [OpenAlex ID](https://openalex.org/A5040358451)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GaLoRA框架，分两阶段训练：先用GraphSAGE GNN在文本属性图上学习结构嵌入，再通过LoRA将这些嵌入注入GPT‑2/ RoBERTa 等LLM，在节点分类任务中进行微调。

**💡 创新点**

创新点在于将结构学习与语言模型微调解耦，只在LLM的少量低秩参数上进行训练，既实现了结构与语义的融合，又将可训练参数降到极低（约0.24%）。

**🔧 技术方法**

核心技术包括 GraphSAGE 作为结构学习器、GPT‑2/ RoBERTa 作为预训练语言模型、LoRA 低秩适配、门控融合机制以及两阶段训练流程。

**📊 数据集**

实验数据集为 Instagram、Reddit 与 ArXiv 三个真实文本属性图，分别涉及商业用户识别、热门用户识别与 40 类论文分类。

**📈 对比分析**

与 GraphAdapter、GLEM 等基线对比，GaLoRA 在三组数据上性能与最先进模型持平或略优，且仅训练约 0.295 M 参数（相当于 GPT‑2 参数的 0.24%），显示出极高的参数效率。

**⚠️ 局限性**

目前仅在节点分类任务上验证，拆分方式可能导致信息泄露；缺乏对链接预测、图分类等其他任务的评估，未来需进一步扩展与验证。

---

## 479. Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America

**arXiv ID:** 2603.10001 | [PDF](https://arxiv.org/pdf/2603.10001v1)

**作者:** Yannis Karmim `[一作]` (Inria), Valentin Barrière `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个针对拉丁美洲的社会文化问答基准，包含超过23,000道多选题，并用该基准评估了多款大型语言模型（LLM）的文化知识水平。

**💡 创新点**

创新点包括：① 结合 Wikipedia 目录与 Wikidata 本体、社会科学专家验证的可扩展数据构建流程；② 对文化元素（如方言、宗教、食物等）进行细粒度分类并量化；③ 同时在西班牙语、葡萄牙语与英文三种语言下评估模型，揭示了模型在本土语言和地区差异上的性能差距。

**🔧 技术方法**

使用的技术包括：① 递归抓取 Wikipedia 类别和子类别，② 通过人工标注（正面、描述、负面）训练多语言 Longformer 进行文档过滤，③ 基于 LLM 的零样本提示生成问答与干扰项，④ 通过多语言 Prompt 评估模型性能。

**📊 数据集**

使用的数据集：① 约154,000篇与拉美文化相关的 Wikipedia 文章；② 通过手工标注过滤后约26,000篇用于问答生成；③ 最终形成23,499道多选题，涵盖20个拉美国家，分别以西班牙语、葡萄牙语及英文提供。

**📈 对比分析**

评估方法：对比多款模型（Mistral 系列、Llama 3.1、Qwen、GPT‑4.1‑mini 等）在本土语言与英文翻译上的准确率；实验发现：① 模型在本土语言上普遍优于英文翻译；② Iberian 西班牙语子集得分显著高于拉美西班牙语；③ 在同一规模内，模型性能随规模增大呈一致提升；④ 在少量实例的文化元素（如“虚构角色”“方言”）上表现差异更大。

**⚠️ 局限性**

局限性：① 仅使用多选题形式，难以捕捉更深层次的文化交互与动态性；② 生成问答时依赖单一 LLM，可能引入偏差；③ 只基于 Wikipedia 及其元数据，忽略了其他语料（如社交媒体、口语记录）；④ 人工标注与验证仍有限，可能影响样本代表性。

---

## 480. Unlearning the Unpromptable: Prompt-free Instance Unlearning in Diffusion Models

**arXiv ID:** 2603.10445 | [PDF](https://arxiv.org/pdf/2603.10445v1)

**作者:** Kyungryeol Lee `[一作]` (Seoul National University), Se Young Chun `[通讯]` (Seoul National University)

**通讯引用:** 2579 | [OpenAlex ID](https://openalex.org/A5052523460)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无提示（prompt‑free）实例忘记方法，利用替代样本、时间步加权与梯度手术，在扩散模型中实现对特定无提示输出（如个人面孔、旗帜等）的忘记，同时保持模型整体生成质量。

**💡 创新点**

① 在无提示条件下实现实例级忘记；② 通过构造替代样本逼近目标并实现“替代忘记”，理论上比直接删除样本对模型影响更小；③ 结合时间步感知的损失权重和仅投影忘记梯度的梯度手术，以平衡遗忘与保持。

**🔧 技术方法**

使用 TediGAN、SDEdit、手工编辑等方法生成替代样本；时间步加权 λ(t)=1−βt；梯度手术（只投影遗忘梯度）；记忆与遗忘损失的多任务优化；评估指标包括 SSCD、LPIPS、SSIM、FID。

**📊 数据集**

CelebA‑HQ、CelebA、FFHQ（人脸数据集）以及 Stable Diffusion 3（生成图片、旗帜等）用于实验。

**📈 对比分析**

与 NegGrad、EraseDiff、SISS、DUO 等基线方法对比，评估遗忘（SSCD<0.4）与模型完整性（LPIPS↓、SSIM↑、FID_pre↓、FID_real↓）。实验表明该方法在单实例与多实例忘记任务中保持更低 SSCD、较低 LPIPS、较高 SSIM，且 FID 几乎不变，优于所有基线。

**⚠️ 局限性**

仅在单一开源扩散模型（DDPM‑CelebA‑HQ、Stable Diffusion 3）验证，未测试闭源大模型；对更大规模或多实例忘记的扩展尚未充分评估；替代样本生成依赖特定工具或手工操作，适用性和自动化程度有限。

---

## 481. Taking Shortcuts for Categorical VQA Using Super Neurons

**arXiv ID:** 2603.10781 | [PDF](https://arxiv.org/pdf/2603.10781v1)

**作者:** Pierre Musacchio `[一作]`, Jaesik Park `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练‑free 的方法，通过阈值化单个神经元的标量激活值来直接生成分类器，并实现极早退出，显著提升视觉语言模型的推理速度。

**💡 创新点**

创新点在于：① 将关注点从宏观的注意力向量转移到微观的标量激活，挖掘出可直接用于分类的“专家神经元”；② 发现这些神经元往往位于浅层，能够在生成第一个 token 时就完成判断，实现极早退出；③ 引入“Agreement”指标量化神经元预测与模型预测的一致性。

**🔧 技术方法**

技术手段包括：简单阈值化激活并二值化；多层聚合（平均或多数投票）；使用特定阈值搜索最佳神经元集合；定义Agreement指标；在第一层直接进行推理以获得速度提升。

**📊 数据集**

使用了七个多类别视觉问答/理解数据集：Pope、InstaOrder（Occlusion 与 Depth）、VizWiz、Clevr、A‑OKVQA、ScienceQa。

**📈 对比分析**

与基线 LLaVA、Qwen 的原始推理、n‑shot 提示、SAVs 进行对比；实验表明：在多数数据集上，专家神经元的准确率、F1 等指标均优于原始模型；在极早退出方案下，推理速度提升至 5.10×；与传统 n‑shot 提示相比，性能显著提升或相当。

**⚠️ 局限性**

局限性：目前仅针对离散类别任务验证，无法保证对开放式提示或复杂推理的效果；性能受探测集规模影响，需足够的样本；对极端分布偏移或不同模型架构的鲁棒性尚未充分验证。

---

## 482. Marginals Before Conditionals

**arXiv ID:** 2603.10074 | [PDF](https://arxiv.org/pdf/2603.10074v1)

**作者:** Mihir Sahasrabudhe `[一作]` `[通讯]` (University of Illinois), Mihir Sahasrabudhe (University of Illinois)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究神经网络从边缘化分布到条件预测的学习阶段，构造了可控多义性与选择器的基准任务，并系统分析训练过程中的停滞‑瞬间转折现象。

**💡 创新点**

首次揭示停滞高度由多义性 K 决定、持续时间由数据量 D 决定；证明梯度噪声以熵力形式稳定边缘化解；展示条件学习的集体“snap”转折和内部选择路由头的先导作用；将此现象与语言模型方向非对称性和逆转诅咒联系起来。

**🔧 技术方法**

采用 4 层 Transformer、AdamW 优化，进行批量大小、学习率、标签噪声等梯度噪声实验；利用对数熵、交叉熵、Hessian 及梯度协方差分析监测模型动态；进行内部头部消融实验验证关键机制。

**📊 数据集**

使用自定义合成数据集：6 字母基字符串 B、4 字母目标 A，设定 K 倍多义性与固定纤维大小；实验覆盖 K∈{5,10,20,36} 与 D∈{3k–36k} 的不同规模。

**📈 对比分析**

通过比较不同 K、D、批量大小、学习率下的停滞持续步数 τ 与损失曲线，评估梯度噪声对停滞的影响；在向前 (A→B) 与向后 ((B,z)→A) 任务中测定训练速度差异，发现向前任务速度慢 1.7–4.4 倍；整体表现符合信息论基准 H(A|B)=log K。

**⚠️ 局限性**

仅在单一模型规模与单一随机种子下得到超线性 τ∝D^1.2，缺乏更广泛尺度验证；对噪声-曲率对齐的定量验证不足；实验仅限合成任务，天然语言任务的推广性仍未知。

---

## 483. The complexity of finite smooth words over binary alphabets

**arXiv ID:** 2603.10733 | [PDF](https://arxiv.org/pdf/2603.10733v1)

**作者:** Julien Cassaigne `[一作]` (CNRS), Raphaël Henry `[通讯]` (Aix Marseille University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文研究了二元字母表上的平滑词（smooth words）及其有限版f‑平滑词（f‑smooth words），证明任何平滑词的因子必为f‑平滑词，并在此基础上给出了f‑平滑词因子复杂度的下界（适用于任意二元字母表）以及上界（在偶字母表上实现Θ上界，在奇字母表上给出改进的多项式上界）。

**💡 创新点**

创新点在于：①首次证明平滑词的因子完全等于f‑平滑词，从而将研究焦点聚焦在f‑平滑词集合上；②在偶字母表上确立了因子复杂度的精确Θ上界；③对奇字母表提供了比以往更紧的上界，利用了新构造的二叉树和谱半径分析。

**🔧 技术方法**

采用了组合学工具，如可导映射、有限导数、bispecial因子与其乘性、树状结构递归、平均长度与最大/最小长度分析，以及谱半径（Perron–Frobenius）方法来估计生成树中单词长度的增长率。

**📊 数据集**

该工作为纯理论研究，无需外部数据集；所有结果均通过严格证明与递归分析得到。

**📈 对比分析**

比较方法主要是与之前的多项式上界（如O(n^β)）以及已知下界（Ω(n^α)）对比。本文在偶字母表上实现与下界一致的Θ(n^ρ)；在奇字母表上提供了新的上界指数ζ，显著小于原先的β值，提升了复杂度估计的精确度。

**⚠️ 局限性**

局限性包括：①未给出单个平滑词在偶/奇字母表上的因子复杂度；②未探讨因子频率（尤其是与Keane等猜想相关的频率等价性）。

---

## 484. CacheSolidarity: Preventing Prefix Caching Side Channels in Multi-tenant LLM Serving Systems

**arXiv ID:** 2603.10726 | [PDF](https://arxiv.org/pdf/2603.10726v1)

**作者:** Panagiotis Georgios Pennas `[一作]` (IMDEA Software Institute), Thaleia Dimitra Doudali `[通讯]` (IMDEA Software Institute)

**通讯引用:** 184 | [OpenAlex ID](https://openalex.org/A5033123165)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个名为CacheSentinel的系统，用于在多租户LLM推理服务中阻止因Automatic Prefix Caching（APC）产生的时间侧信道泄漏，同时不牺牲推理性能。

**💡 创新点**

核心创新在于：①只隔离可能导致泄露的前缀，而非整用户；②动态启用/禁用隔离，基于实时统计的KDE重叠阈值判定侧信道可利用性；③在KV缓存中以极小元数据标记所有权与是否被标记的前缀，实现轻量级、零成本的监控与隔离。

**🔧 技术方法**

技术实现包括：①扩展vLLM KV Cache条目以存储OwnerID与AttackFlag；②Detector组件检测跨用户共享的前缀并按需隔离；③Activator组件实时计算缓存命中/失效的TTFT分布并通过KDE重叠决定是否激活隔离；④整体集成于vLLM多租户调度与执行层。

**📊 数据集**

使用了从ShareGPT公开数据集提取并掩码敏感词后再生成多样化模板的多用户工作负载；同时在9个不同规模（0.5B~13B参数）的LLM模型（Gemma、Llava、Llama、Qwen等）上进行评测。

**📈 对比分析**

与两种基线（无保护的Prefix Caching和用户级Cache Isolation）在TTFT和缓存命中率上对比。CacheSentinel在各种工作负载和模型上实现高达70%更高的缓存复用，推理延迟降低约30%，且与无保护的系统相比仅相差5-10%，同时在攻击者工作负载下完全消除了侧信道可见性。

**⚠️ 局限性**

局限性：①无法防御第一条前缀（无非空前缀）或单次正确猜测的攻击；②依赖KDE阈值设定，阈值不当可能导致过度/不足保护；③在极高负载下侧信道被批处理/排队延迟掩盖，系统仍需判断；④实现依赖vLLM框架，迁移到其他LLM服务需要适配。

---

## 485. MAVEN: A Meta-Reinforcement Learning Framework for Varying-Dynamics Expertise in Agile Quadrotor Maneuvers

**arXiv ID:** 2603.10714 | [PDF](https://arxiv.org/pdf/2603.10714v1)

**作者:** Jin Zhou `[一作]` (Zhejiang University), Shuo Li `[通讯]` (Zhejiang University)

**通讯引用:** 50341 | [OpenAlex ID](https://openalex.org/A5100386630)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了MAVEN框架，利用元强化学习实现单一策略在不同动态（质量、单旋翼推力损失）下的自适应敏捷四旋翼航线规划与控制。

**💡 创新点**

创新点包括：①预测上下文编码器可直接监督动力学与奖励，生成信息丰富的潜在变量；②将离线任务推理与在线PPO策略融合，实现高采样效率与稳定更新；③利用GPU向量化仿真在短时间内完成数十亿步训练。

**🔧 技术方法**

技术手段主要为：meta‑RL（PEARL式上下文编码+PPO）、GPU向量化仿真（Genesis）、Betaflight低层控制、LibTorch部署、实时运动捕捉与MAVLink通信。

**📊 数据集**

实验数据集：仿真中随机采样0.25‑0.5 kg质量与0‑50 %单旋翼推力损失；真实平台使用不同质量（330 g、440 g、550 g）和不同推力损失（30 %、45 %、70 %）的轨迹飞行记录。

**📈 对比分析**

与标准RL（单任务专家）和RL‑DR（域随机化）对比，MAVEN在质量变化下速度与时间接近专家，推力损失下成功率>90%且完成时间低于RL‑DR；即使在超出训练范围的70 %推力损失下也能完成约70 %轨迹。

**⚠️ 局限性**

局限性在于仅验证质量与单旋翼推力损失两类动态，未涵盖风扰动、平台差异等；对极端推力损失（>70 %）鲁棒性仍有限；训练依赖大规模GPU资源，资源不足时训练效率会受限。

---

## 486. Does Reasoning Make Search More Fair? Comparing Fairness in Reasoning and Non-Reasoning Rerankers

**arXiv ID:** 2603.10332 | [PDF](https://arxiv.org/pdf/2603.10332v1)

**作者:** Saron Samuel `[一作]` (Johns Hopkins University), Eugene Yang `[通讯]` (Johns Hopkins University)

**通讯引用:** 975 | [OpenAlex ID](https://openalex.org/A5062016266)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了推理式和非推理式检索重排序器在公平性上的差异。

**💡 创新点**

首次将推理式重排序器与传统重排序器在公平性指标下进行对比，发现推理并未提升公平性。

**🔧 技术方法**

使用多种重排序器（Rank1、Qwen3‑Reranker、ReasonRank、RankZephyr、MonoT5、RankLLaMA）以及BM25、Qwen3检索、融合和oracle配置。

**📊 数据集**

采用TREC 2022 Fair Ranking Track文档集合及其敏感属性标签。

**📈 对比分析**

通过nDCG、AWRF、M1等指标进行量化比较，结果显示推理和非推理模型在公平性上相近，推理模型在相关性上略优，但公平性保持不变。

**⚠️ 局限性**

局限在于仅评估曝光型公平性、未考虑推理过程的文本偏见、只使用英语文档、未探究交叉属性或其他公平度量。

---

## 487. Ranking Reasoning LLMs under Test-Time Scaling

**arXiv ID:** 2603.10960 | [PDF](https://arxiv.org/pdf/2603.10960v1)

**作者:** Mohsen Hariri `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**通讯引用:** 4027 | [OpenAlex ID](https://openalex.org/A5004523290)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在推理LLM的 test‑time scaling 场景下，系统比较了 72 种统计排名方法在不同采样预算（从 1 次到 80 次）下的稳定性与收敛性，并发布了开源库 scorio 以实现这些方法。

**💡 创新点**

创新点包括：①提出了密集基准排名的统一框架和低预算稳定性/收敛性评估协议；②首次对大规模推理模型在奥林匹克数学基准上的多种排名方法进行实证比较；③揭示经验先验（贪心解）在低预算下的方差减小与可能产生的偏差权衡。

**🔧 技术方法**

使用的技术包括：统计排名方法（配对比较模型、IRT、投票规则、图谱谱法、贝叶斯估计等）、经验先验插值、Kendall τ_b 相关性评估、置信区间与保守排名。

**📊 数据集**

实验数据来自 20 个推理 LLM，在四个奥林匹克风格数学基准（代数、几何、数论、组合）上共 30 题，每个模型–问题对采样 80 次（单试验 N=1 的子样本也用于评估）。

**📈 对比分析**

比较方法：用 Kendall τ_b 衡量 N=1（低预算）与 N=80（全预算）下的排名与 Bayes 金标（N=80）或自身 80 试验排名的相似度；结果显示大多数方法在 N=80 时与金标高度一致（τ_b≈0.93–0.95），而低预算下 N 与带贪心经验先验的 N 表现最佳，单试验 τ_b 约为 0.86。

**⚠️ 局限性**

限制：仅评估二元正确性，未覆盖部分计分、开放式输出或验证噪声；经验先验在贪心与随机采样不一致时可能引入系统偏差，且实验局限于数学推理基准。

---

## 488. Deep Randomized Distributed Function Computation (DeepRDFC): Neural Distributed Channel Simulation

**arXiv ID:** 2603.10750 | [PDF](https://arxiv.org/pdf/2603.10750v1)

**作者:** Didrik Bergström `[一作]` (Linköping University), Onur Günlü `[通讯]` (Linköping University)

**通讯引用:** 739 | [OpenAlex ID](https://openalex.org/A5016620064)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

设计并训练了基于自编码器的深度学习架构，用于随机化分布式函数计算（RDFC）中的分布式通道模拟，以实现强协调。

**💡 创新点**

提出了针对RDFC的可构造自编码器架构、训练数据生成算法以及使用总变差距离为目标的损失函数，并利用向量量化层实现率约束，从而显著降低通信负载。

**🔧 技术方法**

使用自编码器、向量量化层、交叉熵损失、Adam优化器、ReduceLROnPlateau回调、直通估计器以及BSC仿真等技术。

**📊 数据集**

以目标BSC(p)分布生成的离散i.i.d.样本作为训练与测试集，样本规模为2^26或更大。

**📈 对比分析**

通过计算合成联合分布与目标分布的总变差距离（TVD）进行评估；实验显示在仅使用局部随机性或同时使用公共随机性的设置下，TVD可降至0.04–0.36，通信率显著低于传统压缩方法。

**⚠️ 局限性**

样本规模有限导致率与理论极限存在差距；在不同字母表大小下未动态调整训练样本，且仅在BSC场景验证，未验证对更复杂分布或更大块长度的适用性。

---

## 489. AttriGuard: Defeating Indirect Prompt Injection in LLM Agents via Causal Attribution of Tool Invocations

**arXiv ID:** 2603.10749 | [PDF](https://arxiv.org/pdf/2603.10749v1)

**作者:** Yu He `[一作]` (Zhejiang University), Zhan Qin `[通讯]` (Zhejiang University)

**通讯引用:** 6057 | [OpenAlex ID](https://openalex.org/A5043524348)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种基于动作层因果归因的 LLM 代理防御方案 AttriGuard，抵御间接提示注入（IPI）攻击。

**💡 创新点**

将 IPI 防御从输入语义判别转向动作因果归因，利用并行对照实验验证工具调用是否因用户意图还是外部观察导致。

**🔧 技术方法**

采用教师强制回放、分层控制衰减与模糊存活判定的并行对照测试，并在多种 LLM 代理架构与对抗优化评估中验证。

**📊 数据集**

主要使用 AgentDojo 和 Agent Security Benchmark 两大基准，在四个大规模 LLM（Gemini‑2.5、GPT‑4.1‑mini、Qwen3‑32B、Llama3.3‑70B）上进行评测。

**📈 对比分析**

与 13 种现有防御（检测、提示、训练、系统级）对比，AttriGuard 在静态攻击下实现 0% ASR、几乎无效用损失，且在自适应攻击中仅保持个位数 ASR，显著优于对手。

**⚠️ 局限性**

依赖黑盒 LLM 仅通过输入输出，且对与用户任务高度重叠的注入（如信息查询）仍可能被误判为合法；在极长序列或高复杂度任务中误判率会略有上升。

---

## 490. CD-Raft: Reducing the Latency of Distributed Consensus in Cross-Domain Sites

**arXiv ID:** 2603.10555 | [PDF](https://arxiv.org/pdf/2603.10555v1)

**作者:** Yangyang Wang `[一作]` (Nanchang University), Zichen Xu `[通讯]` (Nanchang University)

**通讯引用:** 1147 | [OpenAlex ID](https://openalex.org/A5041407793)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 CD-Raft，一种针对跨域数据中心的 Raft 协议改进方案，通过双领导（域领导+全局领导）实现写请求跨域 RTT 仅一次，并保证强一致性。

**💡 创新点**

创新点在于（1）Fast Return 结构，将写操作的跨域 RTT 降至 1；（2）Optimal Global Leader Position 策略，依据请求分布与域间延迟动态选择全局领导，进一步压缩总延迟。

**🔧 技术方法**

技术手段包括 Go 语言实现、RocksDB 存储、gRPC + Protocol Buffers 通信、TLA+ 形式化规范与证明，以及在真实多域云环境中部署测试。

**📊 数据集**

使用 YCSB benchmark，包含四种工作负载（Insert-only、50/50 Update/Read、95/5 Update/Read、Read-only），关键字采用 Zipf 分布。

**📈 对比分析**

通过与 Raft、PigPaxos、Mencius、GeoLM、FR 以及 EPaxos 的基准对比，评估平均延迟、99% 分位延迟和读写延迟。CD‑Raft 在写密集工作负载下平均延迟下降 32.9%‑41%，尾部延迟下降 42%‑52%，在读密集场景仍能保持显著的尾部改善。

**⚠️ 局限性**

局限性包括：跨域拓扑和域数变化时需要重新计算全局领导；Leader 迁移成本较高，需保证迁移周期远大于 RTT；在纯读工作负载下改进有限；对动态网络波动的适应性依赖额外的监控与迁移机制。

---

## 491. Execution Is the New Attack Surface: Survivability-Aware Agentic Crypto Trading with OpenClaw-Style Local Executors

**arXiv ID:** 2603.10092 | [PDF](https://arxiv.org/pdf/2603.10092v1)

**作者:** Ailiya Borjigin `[一作]` (True Trading), Sofiia Pidturkina `[通讯]` (Inc4.net)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Survivability-Aware Execution（SAE）作为 OpenClaw 风格代理栈中的执行层安全合同，确保从 LLM 到交易所的最后一公里都受限。

**💡 创新点**

创新点在于将委托缺口（Delegation Gap）度量与可投射预算、信任状态驱动的动态限额、工具/场地白名单和阶段化执行等多重不可绕过约束整合为可复现的中间件标准。

**🔧 技术方法**

使用的技术包括基于结构化意图规范的 hard‑out‑of‑scope 判定、预算投射（projection）约束、可信状态 (trust‑state) 评估、冷却/速率限制、滑点上限以及统计检验 (block bootstrap, Wilcoxon, z‑test) 的复现评估。

**📊 数据集**

使用的数据集为 2025‑09‑01 至 2025‑12‑01 的 Binance USD‑M BTCUSDT/ETHUSDT 15 分钟条形图、资金费率与手续费信息，构成离线回放。

**📈 对比分析**

通过与 NoSAE、StaticOMS 等基线在同一回放下对比，SAE 在最大回撤从 0.4643 降至 0.0319（93%）、CVaR‑0.99 下降 97.5% 并将委托缺口损失降至 97% 左右，且 AttackSuccess 从 100% 降至 72.8%，表现显著优于基线。

**⚠️ 局限性**

局限包括回放依赖模拟器对清算、滑点和流动性模型的简化、对不同市场/时间段的泛化不足，以及信任评分可能漂移导致的误判或过度限制。

---

## 492. Video-Based Reward Modeling for Computer-Use Agents

**arXiv ID:** 2603.10178 | [PDF](https://arxiv.org/pdf/2603.10178v1)

**作者:** Linxin Song `[一作]` (University of Southern California), Jieyu Zhao `[通讯]` (University of Southern California)

**通讯引用:** 4587 | [OpenAlex ID](https://openalex.org/A5066282713)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出基于执行视频的奖励模型，直接用用户指令和任务视频判断电脑使用代理是否成功，并能定位失败时间段。

**💡 创新点**

创新点在于构建了53k条视频–指令–奖励三元组数据集、使用对抗式指令翻译生成硬负样本并提供时间注释、以及结合空间与时间令牌裁剪实现高分辨率长视频高效训练。

**🔧 技术方法**

核心技术包括视频键帧提取、对抗式指令生成、空间/时间令牌裁剪（STP+TTP）、以及在Qwen3‑VL基础上微调的大模型奖励网络。

**📊 数据集**

使用了AgentNet、ScaleCUA、OSWorld三大公开电脑使用数据集整合而成的ExeVRM‑53K，并通过GitHub和HuggingFace发布。

**📈 对比分析**

在Ubuntu、Mac/Win、Android等多平台评测中，8B模型达84.7%准确率、87.7%召回率，显著优于GPT‑5.2、Gemini‑3 Pro等专有模型和多款开源模型，且在时间定位上获得更高tIoU。

**⚠️ 局限性**

局限性包括对视频分辨率和帧率仍有限制、对极其细微或长时间无显著UI变化的任务可能难以捕捉，以及模型训练仍需大规模GPU资源。

---

## 493. AILS-NTUA at SemEval-2026 Task 8: Evaluating Multi-Turn RAG Conversations

**arXiv ID:** 2603.10524 | [PDF](https://arxiv.org/pdf/2603.10524v1)

**作者:** Dimosthenis Athanasiou `[一作]` (National Technical University of Athens), Giorgos Stamou `[通讯]` (National Technical University of Athens)

**通讯引用:** 3111 | [OpenAlex ID](https://openalex.org/A5085359792)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套统一的多轮检索增强生成（RAG）系统，既解决检索子任务也实现了基于证据的生成与端到端回答。

**💡 创新点**

创新点在于：① 基于单一稀疏检索器的多策略查询重写与层次化RRF融合，提升检索召回而不牺牲 top‑k 精准；② 生成过程拆分为答复可回答性检测、证据段落抽取、双重候选生成与基于技术与用户满意度的判别选择，显著降低幻觉。

**🔧 技术方法**

技术手段包括：DeepSeek‑V3.2、GPT‑4o、GPT‑4o‑mini、ELSER v1（稀疏检索）、Cohere Rerank、嵌套 Reciprocal Rank Fusion、LLM 判别器（多模型）、抽取式证据抽取、微调的长度与可提取性约束。

**📊 数据集**

使用了 SemEval‑2026 MTRAG 评测基准，涵盖 110 条多轮对话、842 轮问答，覆盖四个领域（ClapNQ、FiQA、Govt、Cloud），并对文档按 512‑token 分块生成检索索引。

**📈 对比分析**

通过与 38 个检索基线、26 个生成基线及 29 个端到端基线对比，系统在 Task A 的 nDCG@5 达到 0.5776（排名第一，+20.5%），Task B 的 Harmonic Mean 为 0.7698（排名第二），Task C 的 Harmonic Mean 为 0.5409（排名第十一）。

**⚠️ 局限性**

主要局限在于答案可回答性判别仍存在偏向性：对不可回答的召回率低于 25%，导致对缺乏充分证据的问句过度接受，成为端到端性能的瓶颈。

---

## 494. Differentiable Geometric Indexing for End-to-End Generative Retrieval

**arXiv ID:** 2603.10409 | [PDF](https://arxiv.org/pdf/2603.10409v1)

**作者:** Xujing Wang `[一作]` (Xidian University), Xiaoyi Zeng `[通讯]` (Alibaba)

**通讯引用:** 666 | [OpenAlex ID](https://openalex.org/A5082008486)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Differentiable Geometric Indexing (DGI) 框架，将生成式检索中的索引与检索过程实现端到端的可微分联合训练，从而消除传统方法中的优化阻塞与几何冲突。

**💡 创新点**

创新点主要包括：① 通过 Gumbel‑Softmax 的软教师强制和对称权重共享实现索引与检索的操作统一；② 在单位超球面上使用缩放余弦相似度进行几何优化，消除热门项的范数膨胀导致的 hubness 问题。

**🔧 技术方法**

使用了 Gumbel‑Softmax 近似、对称权重共享、残差量化（RQ‑VAE）、缩放余弦相似度、球面 Riemannian 优化、互信息对比损失、熵正则化、束搜索等技术。

**📊 数据集**

实验数据集包括公开的 AOL4PS（网络搜索）和内部的 AE‑PV（电商页面浏览）两大工业规模数据集。

**📈 对比分析**

与稀疏检索（BM25、DocT5Query）、稠密检索（DSSM‑T5、Sentence‑T5）、生成式检索（DSI、两阶段、UniSearch）等多种基线比较，DGI 在 HitRate、NDCG 等指标上均显著领先；在线 A/B 测试显示 CTR 提升 1.27%、RPM 提升 1.11%。

**⚠️ 局限性**

局限性：仍需在大规模语料下进行更严格的稀疏性与长尾极端情况评估；对温度、缩放因子等超参的敏感性较高；在极端实时低延迟环境中需进一步优化推理效率。

---

## 495. Proceedings of CHIdeology 2026: CHI Workshop on Disentangling the fragmented politics, values and imaginaries of Human-Computer Interaction through ideologies

**arXiv ID:** 2603.10681 | [PDF](https://arxiv.org/pdf/2603.10681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 496. Octopus-inspired Distributed Control for Soft Robotic Arms: A Graph Neural Network-Based Attention Policy with Environmental Interaction

**arXiv ID:** 2603.10198 | [PDF](https://arxiv.org/pdf/2603.10198v1)

**作者:** Linxin Hou `[一作]` (National University of Singapore), Cecilia Laschi `[通讯]` (National University of Singapore)

**通讯引用:** 20799 | [OpenAlex ID](https://openalex.org/A5045065209)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种基于图神经网络的分布式控制架构SoftGM，用于在复杂接触环境中训练柔性软臂实现目标到达任务。

**💡 创新点**

创新点包括：①以章鱼神经系统为启发，将软臂细分为若干代理并通过分布式多智能体强化学习实现在线环境感知与障碍发现；②采用两阶段图注意力信息传递，先将障碍信息注入代理，再实现代理间协作；③通过动态生成与更新图结构，实时融入刚性障碍实体，避免信息瓶颈。

**🔧 技术方法**

技术手段包括：多智能体强化学习（CTDE范式）、图神经网络（GAT）、分布式连续动作策略、Cosserat杆物理仿真（PyElastica）以及自定义奖励与观测编码。

**📊 数据集**

数据集为仿真生成的柔性臂环境，包含三种场景：无障碍、结构障碍和墙壁缺口，目标位置随机采样；所有实验在同一PyElastica仿真环境中完成。

**📈 对比分析**

与六种MARL基线（IDDPG、IPPO、ISAC、MADDPG、MAPPO、MASAC）在相同信息与训练条件下对比。SoftGM在无障碍与结构障碍任务中与强CTDE基线相当，且在墙壁缺口任务中实现最高奖励、最高成功率且平均历时显著缩短，表明其在复杂接触环境中表现最佳。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证；障碍感知依赖理想化接触计算；缺乏硬件实现与传感/执行延迟考量；实验仅针对固定臂长度和段数，未评估对不同几何结构的泛化能力。

---

## 497. LLMGreenRec: LLM-Based Multi-Agent Recommender System for Sustainable E-Commerce

**arXiv ID:** 2603.11025 | [PDF](https://arxiv.org/pdf/2603.11025v1)

**作者:** Hao N. Nguyen `[一作]` (PHENIKAA University), Nguyen Thi Hanh `[通讯]` (PHENIKAA University)

**通讯引用:** 38266 | [OpenAlex ID](https://openalex.org/A5112638384)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种两阶段多代理LLMGreenRec框架，在会话中自动推荐可持续产品并降低数字碳足迹。

**💡 创新点**

通过交互式多代理自我优化提示，自动识别并纠正可持续推荐失误，显著提升意图捕获与绿色产品排序。

**🔧 技术方法**

结合Cross-Encoder重排、六个LLM代理（Evaluate、DetectError、InferReason、RefinePrompt、Augment、Select）、UCB策略和LLM推理技术。

**📊 数据集**

使用MovieLens-1M、Amazon Games、Amazon Bundle（电子、服装、食品）等公开数据集。

**📈 对比分析**

与传统、深度学习及LLM基线（MostPop、NARM、PO4ISR等）在HR@K/NDCG@K上对比，LLMGreenRec在所有数据集和指标上均领先，Bundle数据集上HR@5达0.5504，优势约26–40%。

**⚠️ 局限性**

依赖大型LLM算力，缺乏统一的绿色指标标准，实验仅覆盖有限公开数据集，缺少实时在线评估。

---

## 498. Hierarchical Task Model Predictive Control for Sequential Mobile Manipulation Tasks

**arXiv ID:** 2603.10232 | [PDF](https://arxiv.org/pdf/2603.10232v1)

**作者:** Xintong Du `[一作]` (Technical University of Munich), Angela P. Schoellig `[通讯]` (University of Toronto)

**通讯引用:** 6032 | [OpenAlex ID](https://openalex.org/A5052147335)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种层次任务模型预测控制（Hierarchical‑Task MPC，HTMPC）框架，用于移动机械臂的顺序操控任务，并将该控制器嵌入完整的控制架构（HT‑Arch）。

**💡 创新点**

创新点：
- 将非线性词典序（lexicographic）优化问题重构为可在线求解的形式；
- 在MPC内部直接施加层次约束，避免先后层级的解耦；
- 通过松弛和对数障碍实现约束柔性化，并在线搜索中允许预设的误差容忍度；
- 结合运动学冗余实现基座与机械臂的协同移动，显著提升任务执行效率与反应速度。

**🔧 技术方法**

使用技术：
- 非线性模型预测控制（MPC）
- 词典序优化与多层次约束
- 顺序二次规划（SQP）求解非线性规划（NLP）
- 对数障碍与松弛变量处理约束
- 机器人运动学模型与加速度控制
- 实时计算平台 CasADi + Gurobi。

**📊 数据集**

实验数据集：
- 在真实9自由度移动机械臂（Ridgeback+UR10）上进行多组实验；
- 25个随机正弦/方波基座轨迹与随机目标的仿真/实测；
- 3个比较方法（HTIDKC、HTMPC_WPT、ST‑Arch）对同一任务序列的性能对比。

**📈 对比分析**

比较方法与性能：
- 与基于逆差分运动学（HTIDKC）比较：在任务改变、奇异点与参考变化下，HTMPC 任务跟踪误差平均降低 42%；
- 与单任务架构（ST‑Arch）比较：HT‑Arch 在交付任务序列中执行速度提升 2.3 倍；
- 计算时间：HTMPC 63 ms（10 Hz），HTIDKC 17 ms（50 Hz）；
- 在方波测试中，HTMPC 对前向预测与容错松弛表现出更快收敛和更低误差。

**⚠️ 局限性**

局限性：
- 词典序约束在求解过程中仅保证局部最优，未保证全局最优；
- 需要手动挑选任务维度不超过机器人自由度，适用范围受限；
- 对模型误差、传感噪声等不完美情况的鲁棒性尚未系统评估；
- 计算量相对较大，实时性能受限于硬件；
- 对任务序列长度、复杂度的扩展性尚未在大规模实验中验证。

---

## 499. Reference Architecture of a Quantum-Centric Supercomputer

**arXiv ID:** 2603.10970 | [PDF](https://arxiv.org/pdf/2603.10970v1)

**作者:** Seetharami Seelam `[一作]` (IBM), Jay M. Gambetta `[通讯]` (IBM)

**通讯引用:** 36813 | [OpenAlex ID](https://openalex.org/A5030701195)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了面向量子中心的超级计算机参考架构及三阶段演进路线图，旨在实现量子处理器与传统CPU/GPU资源的深度集成；

**💡 创新点**

创新点在于将量子单元视为同构计算资源，设计了QRMI接口、Tensor Compute Graph、低延迟网络与多级工作流编排等中间件概念，并明确了从分离到全协同的耦合进化；

**🔧 技术方法**

采用了量子资源管理接口QRMI与Slurm插件、低延迟网络（ROCE、NVQLink等）、Tensor Compute Graph、混合工作流编排、错误修正/缓解技术；

**📊 数据集**

示例实验使用了化学分子（N₂、[2Fe-2S]、[4Fe-4S]）的 Hamiltonian 进行 SQD 与闭环 SQD 计算，并未使用公开标准数据集；

**📈 对比分析**

通过 IBM Heron + Fugaku 平台在 6,400 节点上完成大规模对角化，展示了闭环量子‑经典工作流的可行性，但论文未给出与传统纯经典方法的数值性能对比；

**⚠️ 局限性**

主要限制包括量子硬件噪声与门深度受限、耦合实现复杂、对大规模部署缺乏经验，且目前验证仅在单一实验平台完成。

---

## 500. LLM2Vec-Gen: Generative Embeddings from Large Language Models

**arXiv ID:** 2603.10913 | [PDF](https://arxiv.org/pdf/2603.10913v1)

**作者:** Parishad BehnamGhader `[一作]` (Mila Quebec AI Institute), Siva Reddy `[通讯]` (Mila Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自监督框架LLM2Vec-Gen，通过向LLM添加可训练的特殊标记，编码LLM可能生成的回答而非输入文本，从而得到文本嵌入。

**💡 创新点**

创新点在于：①将输入-输出映射问题转化为“生成-编码”范式；②仅冻结LLM主干，训练少量特殊标记和轻量投影层；③结合响应重建和嵌入对齐两项目标，既保证嵌入能复现响应，又能与无监督教师对齐；④嵌入可解释且可解码回文本。

**🔧 技术方法**

使用的技术包括：大规模预训练的解码式LLM（如Qwen‑3、Qwen‑2.5、Llama‑3），特殊训练令牌、轻量MLP投影层，重建损失（跨前向传播的下一个标记预测）和对齐损失（与LLM2Vec教师的嵌入对齐），以及Logit Lens、LatentLens等可解释性工具。

**📊 数据集**

训练使用未标注的Tulu指令跟随数据的单轮提问集（约16万条）；教师嵌入使用同族的LLM2Vec模型；评测数据集包括：MTEB（41任务）、MTEB‑Lite（10任务）、AdvBench‑IR（520个恶意查询）、BRIGHT（多领域推理检索）。

**📈 对比分析**

在MTEB上，LLM2Vec-Gen在Qwen‑3 1.7B/4B/8B等规模上分别提升≈9.3%/12.6%/15.3%，达到自监督SOTA（如8B模型得分62.1）。在AdvBench‑IR上，安全性下降幅度最高43.2%，在BRIGHT上推理检索提升29.3%。与基线（Echo、HyDE、InBedder、GIRCSE、LLM2Vec）相比，均表现出显著优势，尤其在需要安全或推理的任务中。

**⚠️ 局限性**

局限性包括：①依赖无监督教师（LLM2Vec），跨族教师会导致性能下降；②在大模型规模下检索性能略有下降，可能是压缩标记无法完全捕获响应细节；③仅冻结LLM主干限制了进一步提升的空间；④对高质量响应生成的依赖使得模型对训练数据分布有一定敏感性。

---

## 501. Factorized Neural Implicit DMD for Parametric Dynamics

**arXiv ID:** 2603.10995 | [PDF](https://arxiv.org/pdf/2603.10995v1)

**作者:** Siyuan Chen `[一作]` (University of British Columbia), Jonathan Panuelos `[通讯]` (University of Toronto)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5023337048)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种物理编码的神经 Koopman 框架，利用隐式神经表示学习参数化的空间模态和时间演化，实现对可变物理参数（如粘度、几何形状、边界条件）的高维 PDE 动力学建模与长时刻可持续预测。

**💡 创新点**

创新点包括：① 在 Koopman 频谱分解中引入物理编码的连续神经场，分别对模态函数 Φ(x;ξ) 与特征值 Λ(ξ) 进行条件化学习；② 采用对数对共轭模态的参数化与逐阶去相关（deflation）策略，显式正交并强制模态互不重叠；③ 通过短时与长时多步损失相结合的训练策略，提升长期滚动稳定性与泛化能力。

**🔧 技术方法**

技术手段主要有：隐式神经表示（INR）建模连续模态与特征值；Koopman 频谱理论与动态模态分解（DMD）的结合；共轭对参数化、正交约束与去相关学习；多步优化（OptDMD 风格）以及基于伪逆的模态系数回归。

**📊 数据集**

使用的实验数据集包括：变粘度的 Burgers 方程（1D ；10 训练/9 测试粘度值）；二维 Navier–Stokes 双剪切层（双 shear layer 初始条件分离 s∈[0.2,0.4]）；Kármán 旋涡街（圆柱位置变化 x∈[0.35,0.45]）；以及六维形状空间的机翼流场（参数 A_u0,A_u1,A_l0,A_l1,t_e,θ_cw）。

**📈 对比分析**

与基线方法（Consistent Koopman AE、FNO、KNO、P‑DMD、PDE‑Transformer、ResKoopNet）在相同训练/测试设置下比较。结果显示：本方法在所有基准上取得最低的相对均方误差（rMSE），推理时间最快（0.006–0.384 ms/帧），模型参数最少（约 25k–316k），并且在未见物理参数和几何配置下保持稳定的长期预测。

**⚠️ 局限性**

局限性主要有：① 需要显式的物理编码（physics code）来条件化模型，若物理参数难以编码可能受限；② 训练仍需大量时间序列样本和高质量数据，且对极高维参数空间的推广能力尚未充分验证；③ 仅采用线性 Koopman 结构，可能无法捕捉极强非线性耦合或多尺度效应；④ 在极端参数或噪声场景下，模态去相关与正交约束的数值稳定性需进一步评估。

---

## 502. Tool Receipts, Not Zero-Knowledge Proofs: Practical Hallucination Detection for AI Agents

**arXiv ID:** 2603.10060 | [PDF](https://arxiv.org/pdf/2603.10060v1)

**作者:** Abhinaba Basu `[一作]` `[通讯]`, Abhinaba Basu

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于印度哲学Nyāya学说的轻量级验证框架NabaOS，实时检测LLM代理的幻觉并给出分层信任指示；

**💡 创新点**

创新点在于将Nyāya的pramāṇa分类与HMAC签名的工具执行凭证结合，提供可解释的多级信任信号，并实现低延迟的实时验证；

**🔧 技术方法**

采用HMAC签名的工具执行凭证、LLM自标记（self‑tagging）与pramāṇa分类、以及深度代理的URL再抓取与重演校验；

**📊 数据集**

使用自研的多语言幻觉检测基准NyayaVerifyBench（1800个场景，6类幻觉，4种语言）；

**📈 对比分析**

与无验证、Self‑Consistency、RAG‑Grounding、Regex、SVIP等基线比较，NabaOS在所有场景下达成91%检测率、4%误报率，平均额外延迟12 ms；

**⚠️ 局限性**

局限包括对LLM自标记的依赖、无法验证工具自身错误、跨步骤代理的延迟较高、潜在的自欺伪造风险以及基准的合成性质。

---

## 503. Exploring Indicators of Developers' Sentiment Perceptions in Student Software Projects

**arXiv ID:** 2603.10864 | [PDF](https://arxiv.org/pdf/2603.10864v1)

**作者:** Martin Obaidi `[一作]` (Leibniz University Hannover), Kurt Schneider `[通讯]` (Leibniz University Hannover)

**通讯引用:** 5157 | [OpenAlex ID](https://openalex.org/A5031529088)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了学生软件项目中情绪感知的变异性和影响因素，收集四轮问卷并标注30条无上下文语句。

**💡 创新点**

发现情绪感知高度受语句歧义影响，且仅与正向情绪特征有弱关联，首次系统考察个体内随时间的变化。

**🔧 技术方法**

采用问卷调查、情绪量表（Underwood–Froming、PANAS）、Je n's 冲突量表，并用广义估计方程（GEE）建模。

**📊 数据集**

使用来自 GitHub 和 Stack Overflow 的 30 条英文语句作为标注素材。

**📈 对比分析**

通过相关分析和 GEE 结果表明正向情绪特征对正向标注有显著提升，负向标注关联弱，整体效应较小。

**⚠️ 局限性**

主要局限在样本为学生、语句缺乏上下文、标签不平衡且随访缺失可能影响结果的普适性。

---

## 504. Evolving Demonstration Optimization for Chain-of-Thought Feature Transformation

**arXiv ID:** 2603.09987 | [PDF](https://arxiv.org/pdf/2603.09987v1)

**作者:** Xinyuan Wang `[一作]` (Arizona State University), Yanjie Fu `[通讯]` (Arizona State University)

**通讯引用:** 6233 | [OpenAlex ID](https://openalex.org/A5032187620)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于经验进化的闭环框架，利用强化学习探索高质量特征变换序列，并将其写回经验库以动态更新LLM提示，最终实现高效且稳定的特征转换。

**💡 创新点**

创新点在于把少量示例视为可演化的数据而非静态提示，通过多级验证、链式思路重构与熵驱动的多样性选择，构建动态、可验证的上下文，显著提升LLM生成质量和下游任务性能。

**🔧 技术方法**

主要技术包括强化学习探索、LLM的链式思路提示、组合级有效性检测、LLM辅助增强、熵驱动多样性筛选以及闭环写回机制。

**📊 数据集**

使用了覆盖分类与回归的UCI、Kaggle、OpenML等多个表格基准数据集，共计数十个数据集。

**📈 对比分析**

在5折交叉验证下，方法在大多数数据集上优于传统搜索、自动特征工程和现有LLM基线，表现出更高的F1/1-RAE且更为稳定。

**⚠️ 局限性**

主要局限是对LLM推理成本和对离线验证的依赖，且在极大特征维度或极小样本场景下仍可能出现搜索空间过大导致的效率下降。

---

## 505. Prompting with the human-touch: evaluating model-sensitivity of foundation models for musculoskeletal CT segmentation

**arXiv ID:** 2603.10541 | [PDF](https://arxiv.org/pdf/2603.10541v1)

**作者:** Caroline Magg `[一作]` (Quantitative Healthcare Analysis Group, University of Amsterdam), Hoel Kervadec `[通讯]` (Quantitative Healthcare Analysis Group, University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对11种提示式基础模型在四个骨骼区域CT骨与植入物分割任务中进行基准评估，并在完美提示与人类提示两种情况下比较性能。

**💡 创新点**

首次将人类提示变异性纳入评估，提出模型对提示敏感性指标，并在公共与私有数据混合的实验设计下识别Pareto最优模型，突出理想提示与实际使用之间的性能差距。

**🔧 技术方法**

采用2D/3D边框与中心点提示、自动提示提取、观测者人类提示、Dice/NSD/HD95评估、Pareto最优分析、Spearman相关性与Wilcoxon检验等技术。

**📊 数据集**

使用私有的阿姆斯特丹UMC正骨外科CT扫描与公共TotalSegmentator测试集，共49份CT、18类标签、404片轴向切片，涵盖手腕、肩、髋、下腿四个解剖部位。

**📈 对比分析**

先用理想提示计算所有模型的DSC/NSD/HD95，绘制Pareto前沿并挑选参数最小的模型；随后用人类提示评估这些模型的分割一致性和性能下降；结果显示2D SAM2.1 T达89.6% DSC、3D Med‑SAM2达77.1% DSC，使用人类提示时分别下降约2.1%与1.1%。

**⚠️ 局限性**

局限包括：部分模型训练/测试集不完全公开、仅使用轴向切片、标注者为医学生而非放射科专家、未采用迭代修正交互、仅评估几何提示，缺乏文本提示与更复杂解剖结构的验证。

---

## 506. FutureVLA: Joint Visuomotor Prediction for Vision-Language-Action Model

**arXiv ID:** 2603.10712 | [PDF](https://arxiv.org/pdf/2603.10712v1)

**作者:** Xiaoxu Xu `[一作]` (Beihang University), Jiangmiao Pang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 44482 | [OpenAlex ID](https://openalex.org/A5087818121)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FutureVLA框架，实现联合视动预测以提升视觉语言动作模型的未来指导能力

**💡 创新点**

通过视觉与运动流的结构化解耦（Joint Visuomotor Gating）实现视觉支配的消除和时间连续性的保持；并引入两阶段预训练与潜在对齐的后训练策略

**🔧 技术方法**

利用冻结的3D-VAE对连续视频片段进行时空编码；Transformer层实现视觉与运动子表示；交叉注意力与门控机制实现解耦；潜在嵌入对齐损失

**📊 数据集**

多源操作数据集（如LIBERO、SimplerEnv、Frankie机器人实测数据）以及多任务机器人抓取、插入、摆放等数据

**📈 对比分析**

与现有显式与隐式未来指导方法（如WorldVLA、LAPA、Villa-X）在LIBERO、SimplerEnv和真实机器人任务上对比；在Google/WidowX机器人任务中平均提升44.9%~30.1%，在LIBERO长任务提升至99.8%；在真实任务中平均成功率提升至70%（比基线高26.7%）

**⚠️ 局限性**

仍受限于预训练所需的视频片段长度与帧密度的选择，过长或过短均会影响效果；对不同硬件/任务的迁移能力需要进一步验证

---

## 507. Towards Cognitive Defect Analysis in Active Infrared Thermography with Vision-Text Cues

**arXiv ID:** 2603.10549 | [PDF](https://arxiv.org/pdf/2603.10549v1)

**作者:** Mohammed Salah `[一作]` (Khalifa University of Science and Technology), Yusra Abdulrahman `[通讯]` (Khalifa University of Science and Technology)

**通讯引用:** 338 | [OpenAlex ID](https://openalex.org/A5070120499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种零样本认知缺陷分析框架，利用主动红外热像技术与预训练的视觉‑文本模型（VLM）配合 AIRT‑VLM 适配器，实现对碳纤维增强聚合物（CFRP）内部缺陷的自动定位。

**💡 创新点**

创新点在于：①通过轻量化的 AIRT‑VLM 适配器把热像序列压缩成与自然图像分布相匹配的单幅图像，从而消除了热像与 VLM 预训练域的差距；②实现了完全零样本、无标签、无额外训练的缺陷定位；③在保持高信噪比的同时兼顾计算效率。

**🔧 技术方法**

主要技术包括：主动红外热像（AIRT）采集；基于掩码自编码器的 AIRT‑VLM 适配器，用平均池化生成域对齐热像；以及三种公开 VLM（GroundingDINO、Qwen‑VL‑Chat、CogVLM）进行零样本目标定位。

**📊 数据集**

使用了 25 条 CFrP 试样的 AIRT 检测序列，试样在 5 J 与 15 J 低速冲击下产生缺陷，且在常温和 ‑70 °C 两种温度环境下采集，形成了无标注的热像数据集。

**📈 对比分析**

与传统维度约简方法（TSR、PCA、DAT、1D‑DCAE‑AIRT、C‑AET）相比，AIRT‑VLM 适配器提升了 10–20 dB 的信噪比，并在三种 VLM 上获得平均 IoU 约 0.7、中心距离 NCD 约 0.015 的定位性能，显著优于基线。

**⚠️ 局限性**

局限性在于：①无法估计缺陷深度；②无法区分不同缺陷类型（如脱层、空洞、冲击损伤）；③依赖单幅域对齐图像，导致失去原始热像序列的时空物理信息。

---

## 508. Modeling Stage-wise Evolution of User Interests for News Recommendation

**arXiv ID:** 2603.10471 | [PDF](https://arxiv.org/pdf/2603.10471v1)

**作者:** Zhiyong Cheng `[一作]` (Hefei University of Technology), Meng Wang `[通讯]` (Hefei University of Technology)

**通讯引用:** 42433 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本论文提出一种统一框架，既通过全局用户-新闻交互图学习长期协同偏好，又通过将历史行为划分为时间子图，利用LSTM捕捉短期演化，使用自注意力聚合跨阶段长程依赖，完成新闻推荐。

**💡 创新点**

创新点在于将长期协同信息与短期阶段化动态结合，利用全局图初始化局部子图并在局部中同时采用递归与注意力两条分支，实现对新闻推荐中“时间演化+协同共性”的双重建模。

**🔧 技术方法**

技术包括LightGCN图卷积作为全局特征提取器、基于时间窗口的局部子图构造、LSTM序列建模、基于自注意力的长程聚合、以及多任务损失（交叉熵、对比一致性、平滑正则）进行联合训练。

**📊 数据集**

实验使用两大公开新闻数据集：Adressa‑Large（约3.6M用户，81k新闻，35M点击，3个月）和MIND‑Large（1M用户，161k新闻，24M点击，6周）。

**📈 对比分析**

与NRMS、NAML、NPA、LightGCN、CNE‑SUE、TCCM、CROWN等基线相比，本方法在AUC、MRR、nDCG@5/10等指标上均领先明显，尤其在AUC上提升约0.1以上，验证了该框架对短期动态与长期协同的有效融合。

**⚠️ 局限性**

局限性包括需要足够的历史交互才能构建可靠的时间子图，对极度稀疏用户的表现可能受限；模型结构较复杂，训练和推理成本高；并且主要针对新闻场景，迁移到其他时序推荐任务需进一步验证。

---

## 509. One Model, Many Skills: Parameter-Efficient Fine-Tuning for Multitask Code Analysis

**arXiv ID:** 2603.09978 | [PDF](https://arxiv.org/pdf/2603.09978v1)

**作者:** Amal Akli `[一作]` (University of Luxembourg), Yves Le Traon `[通讯]` (University of Luxembourg)

**通讯引用:** 17057 | [OpenAlex ID](https://openalex.org/A5040574362)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了多任务学习与参数高效微调（PEFT）在代码分析任务中的结合，构建单一模型同时完成四种代码分析任务。

**💡 创新点**

创新点在于首次系统评估多任务 PEFT 的有效性，证明其能在多任务环境中保持甚至超越全微调性能，并给出了任务配对、模型架构与数据质量对迁移效果的实证指导。

**🔧 技术方法**

使用 Serial Adapter、Parallel Adapter、LoRA 与 Prefix Tuning 等 PEFT 技术，并采用硬参数共享的多任务训练框架。

**📊 数据集**

数据集来源于 CodeXGLUE 基准，包括 Clone Detection、Vulnerability Detection、Code Search（AdvTest）和 Flakiness Prediction 四个任务。

**📈 对比分析**

与全模型微调、单任务 PEFT 以及 7–34B 指令调教大型 LLM 的零样本推理进行对比，发现多任务 PEFT 在准确率上与全微调相近或更好，同时显著降低可训练参数量（约按任务数缩小）和计算成本（最高可降低 86%）。

**⚠️ 局限性**

局限性包括仅覆盖四种任务与四个 1–1.5B 规模模型，结果可能不适用于更大模型或生成类任务；某些任务（如搜索）在联合训练中易出现负迁移；并且未针对每个模型进行专门的超参调优。

---

## 510. The Quadratic Geometry of Flow Matching: Semantic Granularity Alignment for Text-to-Image Synthesis

**arXiv ID:** 2603.10785 | [PDF](https://arxiv.org/pdf/2603.10785v1)

**作者:** Zhinan Xiong `[一作]` (Conservatoire National des Arts et Métiers), Shunqi Yuan `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Flow Matching 框架下分析生成模型微调的优化动力学，提出 Semantic Granularity Alignment (SGA) 方法，以通过对输出残差空间进行工程化干预，缓解梯度冲突并加速收敛。

**💡 创新点**

创新点在于：将 MSE 目标等价为受动态 Neural Tangent Kernel (NTK) 支配的二次形式，揭示数据相互作用矩阵；通过层级语义分解 (H‑SD) 与语义粒度对齐（tuple‑wise 优化与 scale‑adaptive 频率对齐），实现对残差空间的显式控制。

**🔧 技术方法**

采用的技术包括 Flow Matching、NTK 理论、层级语义分解(H‑SD)、tuple‑wise 优化、scale‑adaptive 频率对齐、LoRA/DoRA 权重适配、FLUX (DiT) 与 Animagine XL 3.1 (U‑Net) 两种架构。

**📊 数据集**

使用六个 GDA 领域（FLUX）与三个领域（U‑Net）中的文本‑图像数据集，每个领域包含 100~数百张图像，涵盖多种风格与场景。

**📈 对比分析**

通过在相同 GPU 预算（1.0 N1 vs 1.5 N1）下，与基线方法对比，采用 GPT‑5.2 LLM Judge 与人类评估两种指标；SGA 在 1.0 N1 下实现 1st‑place 率约 40–55%，超过基线 1.5 N1 的表现；CLIP‑I/T 与 DINO‑I 指标均显著提升。

**⚠️ 局限性**

局限性：H‑SD 依赖外部检测器，若检测器性能不足会影响分解质量；实验仅覆盖文本‑图像合成，未验证在视频生成或多模态混合任务中的通用性。

---

## 511. Measurement-Driven O-RAN Diagnostics with Tail Latency and Scheduler Indicators

**arXiv ID:** 2603.11023 | [PDF](https://arxiv.org/pdf/2603.11023v1)

**作者:** Theofanis P. Raptis `[一作]` (National Research Council), Roberto Verdone `[通讯]` (University of Bologna)

**通讯引用:** 4960 | [OpenAlex ID](https://openalex.org/A5022214905)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对O‑RAN实例进行端到端ICMP延迟与基站调度指标的交叉层诊断，基于实际测量。

**💡 创新点**

提出尾部延迟与调度/链路适配指标相结合的轻量窗口化“退化标志”，实现可解释的跨层诊断。

**🔧 技术方法**

使用ICMP ping、OAI gNB fullstats日志、窗口化统计、尾部百分位和Spearman相关等技术。

**📊 数据集**

基于不同距离（2 m、6 m、11 m）、两种UE（手机与调制解调器）在静态与动态遮挡下收集的现场测量数据。

**📈 对比分析**

通过比较尾部延迟、超过阈值概率以及BLER/MCS统计，发现UE差异、距离与包大小导致的尾部扩大，窗口化相关性证明两者关联，性能上尾部指标显著高于平均值。

**⚠️ 局限性**

仅限ICMP测量可能掩盖短期链路适配变化，且数据覆盖不均（11 m仅手机），缺乏完整时间对齐与更细粒度的链路层观察。

---

## 512. ECoLAD: Deployment-Oriented Evaluation for Automotive Time-Series Anomaly Detection

**arXiv ID:** 2603.10926 | [PDF](https://arxiv.org/pdf/2603.10926v1)

**作者:** Kadir-Kaan Özer `[一作]` (Mercedes-Benz AG), Markus Enzweiler `[通讯]` (Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实施了ECoLAD评估协议，针对汽车时序异常检测在有限CPU/GPU资源下的可部署性能进行系统性评估。

**💡 创新点**

创新点在于：①引入多层计算降级阶梯与显式CPU线程上限；②通过吞吐量覆盖度与可实现AUC‑PR来量化受限资源下的性能；③提供可审计的配置日志和可复现的操作点选择机制，兼顾效率与准确性。

**🔧 技术方法**

采用机械整数比例缩放规则对模型参数进行调度；使用窗口化评分与AUC‑PR评估；在GPU、CPU多线程、CPU有限线程、CPU单线程四个后端下测量推理时间和吞吐量（wps）；对经典与深度学习的TSAD模型进行对比。

**📊 数据集**

使用专有汽车遥测数据（80k点、19维、异常率≈0.022）、公开服务器监控数据SMD以及公开空间遥测数据SMAP。

**📈 对比分析**

在每个计算层记录AUC‑PR、运行时和吞吐量；结果显示经典方法（HBOS、COPOD）在低资源下仍保持高吞吐和较好AUC‑PR；深度方法在GPU表现优异但在CPU单线程时吞吐急剧下降；方法排名随资源下降发生漂移，需同时考虑吞吐覆盖率来做部署决策。

**⚠️ 局限性**

局限性包括：专有遥测数据保密无法公开；实验平台M3 Max非周期精确ECU仿真，需平台校正；机械缩放规则未进行针对每层的精细调优，可能低估最佳性能。

---

## 513. Beyond Accuracy: Reliability and Uncertainty Estimation in Convolutional Neural Networks

**arXiv ID:** 2603.10731 | [PDF](https://arxiv.org/pdf/2603.10731v1)

**作者:** Sanne Ruijs `[一作]` (Lund University), Farrukh Javed `[通讯]` (Lund University)

**通讯引用:** 658 | [OpenAlex ID](https://openalex.org/A5101950181)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对深度卷积网络的两种不确定性估计方法——Monte Carlo Dropout（贝叶斯逼近）和Conformal Prediction（分布无关预测集）在 Fashion‑MNIST 数据集上进行对比实验。

**💡 创新点**

首次系统比较两种方法在不同网络架构（H‑CNN VGG16 与 GoogLeNet）下的校准性、覆盖率与效率，揭示模型复杂度与不确定性估计的相互影响。

**🔧 技术方法**

使用 Monte Carlo Dropout 进行多次前向传播以估计预测熵、互信息等；使用 Inductive Conformal Prediction 计算非合规分数与阈值，生成可置信预测集。

**📊 数据集**

采用公开的 Fashion‑MNIST 数据集，包含 70,000 张 28×28 的灰度图像，10 类服装。

**📈 对比分析**

在准确率、ECE、预测集大小与覆盖率等指标上比较，结果显示 H‑CNN VGG16 取得更高准确率但过度自信，GoogLeNet 则在校准性与覆盖率上更稳健；两种方法互补，MC Dropout 体现模型不确定性，CP 提供统计保证。

**⚠️ 局限性**

限制主要体现在：MC Dropout 需要多次前向传播导致推理成本高，且对 H‑CNN VGG16 的过拟合表现未得到根本缓解；CP 的预测集可能过大，效率低；两种方法均未针对跨域或噪声干扰的鲁棒性进行深入评估。

---

## 514. Model-Free Co-Optimization of Manufacturable Sensor Layouts and Deformation Proprioception

**arXiv ID:** 2603.10059 | [PDF](https://arxiv.org/pdf/2603.10059v1)

**作者:** Yingjun Tian `[一作]` (University of Manchester), Charlie C. L. Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出并实现了一套模型无关、数据驱动的共优化框架，能够同时优化柔性长度测量传感器的布局（数目、位置、长度）与形变预测网络的参数，实现可制造的自由曲面形变自我感知。

**💡 创新点**

创新点包括：① 将离散的传感器数目与连续的布局参数统一在可微分框架下共优化；② 将可制造约束（最短长度、最小间距、无重叠）以可微分损失形式编码；③ 完全基于形变数据，无需物理仿真模型，适用于多种软体机器人和可穿戴设备。

**🔧 技术方法**

技术手段包括：B‑spline曲面参数化与可微分曲线长度计算；利用 PyTorch 的 MLP 与 Adam 优化实现形变预测网络与布局参数的梯度更新；通过符号微分实现重叠、间距与长度约束的可微损失；以及在 GPU 上并行化评估与训练。

**📊 数据集**

使用三套实验数据集：软变形人偶、软操纵臂和肩部可穿戴各自收集约 2000 个形变样本，采用 MoCap 标记进行三维重建后拟合为 B‑spline 表面，作为训练与测试集。

**📈 对比分析**

方法通过与未优化、专家设计或随机布局的基线进行比较；在软操纵臂上共优化后仅需 6 个传感器即可达到 20 传感器精度；在肩部可穿戴上 4 个传感器显著优于 20/4 的基线；在软人偶上 10 个优化传感器的误差比专家布局降低 30%–50%；物理实验进一步验证误差下降幅度可达 30%–50%。

**⚠️ 局限性**

局限性包括：① 需先将点云/网格转换为 B‑spline 表面，增加预处理工作；② 当前 UV 参数域受边界限制，传感器无法跨越；③ 未考虑碰撞或自碰撞等更复杂情形；④ 对于极端形变或高频运动的泛化能力尚未完全验证。

---

## 515. Fine-Tune, Don't Prompt, Your Language Model to Identify Biased Language in Clinical Notes

**arXiv ID:** 2603.10004 | [PDF](https://arxiv.org/pdf/2603.10004v1)

**作者:** Isotta Landi `[一作]` (Icahn School of Medicine at Mount Sinai), Kimberly B. Glazer `[通讯]` (University of Pennsylvania)

**通讯引用:** 1158 | [OpenAlex ID](https://openalex.org/A5018544905)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建了偏见语言词典并提取OB‑GYN和MIMIC‑IV病历片段，人工标注情感价值。

**💡 创新点**

创新点在于双步词典+片段情感标注框架、对词典情感在不同专业和机构中变化的系统性研究，以及通过词典先验和细调实现的跨域适配。

**🔧 技术方法**

使用了Encoder‑only模型GatorTron和BERT、以及生成式LLM Llama3.3、3.2、Med42；通过细调、提示微调和词典先验进行情感分类。

**📊 数据集**

数据集为Mount Sinai OB‑GYN产科记录（1,015片段）和MIMIC‑IV住院/急诊摘要（275片段）。

**📈 对比分析**

对比细调、零射提示、ICL等策略，GatorTron细调+词典先验取得最高F1≈0.96；在跨域验证时性能下降44%，在反向训练时仅下降11%。

**⚠️ 局限性**

局限在于仅从临床医生视角捕捉情感价值，缺乏患者或多样化视角，且仅评估单一专业与多专业混合机构，未能全面揭示分布漂移原因。

---

## 516. mAceReason-Math: A Dataset of High-Quality Multilingual Math Problems Ready For RLVR

**arXiv ID:** 2603.10767 | [PDF](https://arxiv.org/pdf/2603.10767v1)

**作者:** Konstantin Dobler `[一作]` (Hasso Plattner Institute), Mohamed Ali `[通讯]` (Apple)

**通讯引用:** 4783 | [OpenAlex ID](https://openalex.org/A5046562017)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

未提供论文内容，无法概括

**💡 创新点**

无

**🔧 技术方法**

无

**📊 数据集**

无

**📈 对比分析**

无

**⚠️ 局限性**

无

---

## 517. Frames2Residual: Spatiotemporal Decoupling for Self-Supervised Video Denoising

**arXiv ID:** 2603.10417 | [PDF](https://arxiv.org/pdf/2603.10417v1)

**作者:** Mingjie Ji `[一作]` (Nanjing University), Xun Cao `[通讯]` (Nanjing University)

**通讯引用:** 5958 | [OpenAlex ID](https://openalex.org/A5058572381)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了 Frames2Residual (F2R)，一种将时空一致性建模与空间纹理恢复分离的两阶段自监督视频去噪框架。

**💡 创新点**

创新点在于：①先通过帧级盲估计得到时间一致的锚；②再通过recurrence 重新注入中心帧实现空间细节恢复；③使用残差域学习和光流引导的自适应对齐模块（FAAM、FDAM）解耦时空特征。

**🔧 技术方法**

采用预训练图像去噪器 NAFNet 作为结构基准，预训练光流 PWC-Net 作为对齐辅助，4 层 U‑Net 作为骨干网络，FAAM/FDAM 进行流引导的对齐，残差域学习及 recorruption 训练策略。

**📊 数据集**

使用合成 Gaussian 去噪数据集（DAVIS 2017、Set8）和真实原始视频数据集 CRVD，分别在 sRGB 和 raw 视频上评估。

**📈 对比分析**

与多种自监督方法（MF2F、RFR、UDVD、RDRF、ER2R、TAP）以及监督方法（FastDVDNet、FloRNN、PaCNet、NAFNet）比较，F2R 在 DAVIS 上平均 PSNR 36.14 dB、Set8 上 34.30 dB，超越所有自监督方法并接近监督水平；在 CRVD 上同样领先 TAP 与监督方法，提升约 0.6–1 dB。

**⚠️ 局限性**

局限性：需依赖预训练的图像去噪器和光流模型，训练分两阶段复杂；对极端快速运动或大尺度形变仍可能出现对齐误差；对特殊噪声分布的域适应性可能不足。

---

## 518. Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution

**arXiv ID:** 2603.10583 | [PDF](https://arxiv.org/pdf/2603.10583v1)

**作者:** Hongsong Wang `[一作]` (Southeast University), Jie Gui `[通讯]` (Southeast University)

**通讯引用:** 5978 | [OpenAlex ID](https://openalex.org/A5110740283)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于低比特平面指纹的检索式AI生成图像归因框架 LIDA，能在零样本或少样本情况下对未知生成器进行识别和检测。

**💡 创新点**

创新点在于将归因任务转化为实例检索问题，利用低比特平面生成的生成指纹提取器，且只需训练一个轻量化特征编码器即可实现对新生成器的快速适配，提升了通用性和可扩展性。

**🔧 技术方法**

核心技术包括低比特平面指纹生成、基于 ResNet‑50 的特征编码器、无监督预训练（预文本任务）、中心损失和对比损失的少样本归因适配。

**📊 数据集**

主要使用 GenImage 与 WildFake 两大公开生成图像数据集，涵盖多种扩散与 GAN 生成器，评估跨架构与跨生成器归因与检测性能。

**📈 对比分析**

与 ResNet‑50、DIRE、ESSP 等基线相比，LIDA 在 Rank‑1、mAP、检测准确率上均达到或超过 90% 以上，显著优于同类方法，在零样本和 1/5/10/100/1000 shot 场景均实现领先。

**⚠️ 局限性**

局限性包括：需维护并不断更新包含少量样本的注册数据库；对极端图像压缩或模糊程度较高的扰动仍有一定性能下降；在极低样本（1 shot）下，检索排序仍受噪声影响。

---

## 519. Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation

**arXiv ID:** 2603.10340 | [PDF](https://arxiv.org/pdf/2603.10340v1)

**作者:** Sangmim Song `[一作]` (University of Technology Sydney), Karthick Thiyagarajan `[通讯]` (Western Sydney University)

**通讯引用:** 869 | [OpenAlex ID](https://openalex.org/A5079376016)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该论文提出一种无训练、模型无关的推理时视觉蒸馏框架CGVD，利用语言分解、实例分割与傅里叶基填充抑制视觉噪声，从而提升VLA模型在混乱环境中的精准操控。

**💡 创新点**

创新点在于将语言指令驱动的概念门控与两层目标细化结合，先用集合交叉验证排除误检，再用空间消歧明确目标，随后在像素层面通过自洽分割和LaMa填充生成干净观测，显著弥补了精度-推理缺口。

**🔧 技术方法**

关键技术包括SAM3实例分割、两层目标细化（交叉验证+空间消歧）、基于LaMa的傅里叶卷积填充、以及时序一致的图像混合。

**📊 数据集**

主要使用Simulator SimplerEnv，结合RoboCasa与YCB数据集中的对象进行多种混乱任务（如“把勺子放在毛巾上”“把胡萝卜放在盘子上”）。

**📈 对比分析**

与基线VLA模型（π0、GR00T）对比，CGVD在语义混乱下将成功率从43.0%提升至77.5%，在含18个语义干扰物的场景中表现优异，整体对比曲线显示在干扰物增多时CGVD保持高成功率，而基线迅速下滑。

**⚠️ 局限性**

限制包括依赖静态背景导致对动态干扰物不敏感、对非语义混乱的过度填充可能略降性能、以及初始化时的单帧计算导致短暂启动延迟。

---

## 520. Linear-Scaling Tensor Train Sketching

**arXiv ID:** 2603.11009 | [PDF](https://arxiv.org/pdf/2603.11009v1)

**作者:** Paul Cazeaux `[一作]` (Virginia Tech), Rodrigo Figueroa Justiniano `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的 Block Sparse Tensor Train (BSTT) 随机投影方法，用于压缩高维张量并加速张量训练（TT）近似与低秩压缩。

**💡 创新点**

创新点在于将 Khatri‑Rao 与 Gaussian TT 两种已知张量稀疏投影统一为一个参数化框架，并证明其在子空间嵌入（OSE）和子空间注入（OSI）下只线性依赖于张量阶数 d 和子空间维度 r，克服了传统方法指数级扩张的瓶颈。

**🔧 技术方法**

采用结构化随机矩阵（块级高斯或正交化块），利用强 Johnson‑Lindenstrauss 时刻、OOS、OSI 理论、矩阵乘积逼近、随机投影与张量网络运算的高效递归算法。

**📊 数据集**

在合成张量、Hadamard 乘积以及实际量子化学 LiH 分子 Hamiltonian 的 TT 表征上进行实验验证。

**📈 对比分析**

通过与 Khatri‑Rao、Gaussian TT、f_TT(R) 等现有投影方式比较，BSTT 在相同误差下只需 P≈r/ε、R≈d 级别的采样维数，实验中表现出更快的收敛速度和更低的误差，尤其在高阶张量（d≥100）时明显优于传统方法。

**⚠️ 局限性**

限制主要包括：1) 仍需大块秩 R 以满足 OSE/OSI 证明，导致在极大张量阶或维度下存储与计算成本上升；2) 当前基于高斯分布，缺乏对更高效稀疏或快速 JL 变换的理论支持；3) 对更复杂张量网络（如 TTN、MCTDH）以及块稀疏结构的适配尚未完善。

---

## 521. TAMUSA-Chat: A Domain-Adapted Large Language Model Conversational System for Research and Responsible Deployment

**arXiv ID:** 2603.09992 | [PDF](https://arxiv.org/pdf/2603.09992v1)

**作者:** Izzat Alsmadi `[一作]` (Texas A&M University), Anas Alsobeh `[通讯]` (Utah Valley University)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5018319301)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个可复现、模块化的框架TAMUSA-Chat，用于在特定学术机构背景下对大语言模型进行监督微调、检索增强生成和评估。

**💡 创新点**

提供了完整的数据采集到推理的流水线，强调可复现性、可扩展性与伦理治理，并将检索增强与SFT结合以降低幻觉。

**🔧 技术方法**

采用了Playwright爬虫、PyPDF2/Docx解析、Sentence-Transformer生成嵌入、FAISS索引、LoRA/全参数SFT、FastAPI+Gradio、Docker等技术。

**📊 数据集**

以德州A&M大学圣安东尼奥校区公开网页、PDF、课程目录、政策文件等为源，构建约8,932个指令-响应对，约2.4M词。

**📈 对比分析**

通过自动指标（BLEU/ROUGE）与人工评测（GPT‑4判定）与Vicuna‑13B、EduChat等对照，发现TAMUSA‑Chat在事实准确性与上下文相关性上优于基线，且在3B模型下保持可接受的推理延迟。

**⚠️ 局限性**

仍受限于训练数据覆盖范围、对最新政策更新的滞后、对极端多模态查询支持不足，以及在大规模并发场景下资源消耗与成本未彻底评估。

---

## 522. Early-Stage Cancer Biomarker Detection via Intravascular Nanomachines: Modeling and Analysis

**arXiv ID:** 2603.10709 | [PDF](https://arxiv.org/pdf/2603.10709v1)

**作者:** Abdollah Rezagholi `[一作]` (Nanonetworking Center in Catalunya), Ethungshan Shitiri `[通讯]` (Nanonetworking Center in Catalunya)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在血管内引入纳米机器并通过分子通信仿真评估早期癌症生物标志物检测概率，比较简化模型与包含层流、粒子倾向和尺寸依赖运动的真实输运机制。

**💡 创新点**

首次将血管层流、粒子边缘化和尺寸相关运动等真实输运效应集成到 AcCoRD 分子通信仿真框架中，并系统评估它们对检测概率的影响。

**🔧 技术方法**

使用 AcCoRD 仿真框架、层流模型、尺寸相关运动系数、粒子边缘化建模及分子通信理论。

**📊 数据集**

基于 ALPPL2+ 外泌体浓度实验数据（健康、NCC、癌症），并设置 1–10 颗生物标志物及不同直径与流速的微血管参数。

**📈 对比分析**

通过对比统一流动与层流、无边缘化与含边缘化、简化与完整模型下的检测概率来评估性能；结果显示层流与边缘化显著降低检测概率，capillary 的检测效率最高，较大纳米机器可减少所需数量。

**⚠️ 局限性**

仅模拟单段血管，未考虑血管分支、网络拓扑、外部环境或纳米机器能量与通信细节，且生物标志物数量和种类有限，限制了结果的普适性。

---

## 523. The System Hallucination Scale (SHS): A Minimal yet Effective Human-Centered Instrument for Evaluating Hallucination-Related Behavior in Large Language Models

**arXiv ID:** 2603.09989 | [PDF](https://arxiv.org/pdf/2603.09989v1)

**作者:** Heimo Müller `[一作]` (Medical University of Graz), Andreas Holzinger `[通讯]` (BOKU University Vienna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并验证了一种名为 System Hallucination Scale（SHS）的轻量级、用户中心的量表，用来评估大语言模型输出中的幻觉相关行为。

**💡 创新点**

创新点在于将幻觉评估转化为五维可操作的 Likert 量表，并在量表中嵌入一致性诊断，兼具易用性与心理测量的可靠性。

**🔧 技术方法**

采用心理测量技术（Cronbach α、Pearson 相关、内部一致性检验）以及对照 SUS/SCS 等已有量表进行统计分析，结合人机交互实验进行验证。

**📊 数据集**

在 210 名参与者的真实交互实验中收集数据，使用自制的对话与提示集进行评估，未采用公开基准数据集。

**📈 对比分析**

与 SUS/SCS 的比较显示 SHS 具备独立的五维结构且与自动幻觉检测互补；Cronbach α = 0.87，维度间相关系数 0.42–0.72，表明良好的构念效度和可靠性。

**⚠️ 局限性**

局限性包括：评估高度依赖主观判断，受参与者背景知识和认知偏差影响；仅在英语环境与非专业用户中验证，跨语言和高阶专业领域仍需进一步研究。

---

## 524. UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis

**arXiv ID:** 2603.10852 | [PDF](https://arxiv.org/pdf/2603.10852v1)

**作者:** Yali Zhu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Gaofeng Meng `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种层级多智能体框架 UltrasoundAgents，通过主机对全图定位并子机对裁剪图像进行属性识别，形成 ROI→属性→诊断的可审计证据链，提升乳腺超声诊断。

**💡 创新点**

创新点在于：①将全局定位与高层推理与局部属性感知分离，构建显式的多步证据链；②引入 oracle-guided curriculum RL 训练主机以消除子机误差带来的非平稳性；③通过自纠轨迹蒸馏将 RL 轨迹转化为监督数据，实现高质量可部署策略。

**🔧 技术方法**

采用多智能体强化学习（GRPO）、自监督蒸馏、crop‑and‑zoom 观察、基于 Qwen2.5‑VL‑3B 的视觉‑语言模型。

**📊 数据集**

在 BUSBRA、BUSI、BUDIAT 三个公开乳腺超声数据集上训练，并在 BrEaST OOD 数据集上评估。

**📈 对比分析**

相较于 Zero‑Shot、CoT‑SFT、Think‑with‑Image 等基线，UltrasoundAgents 在 AUC、BI‑RADS 及属性准确率上均取得最高水平（如整体 AUC 0.741、BI‑RADS 0.515），且 OOD 诊断提升显著。

**⚠️ 局限性**

局限性包括属性标注稀缺且不平衡，模型对子机生成的属性噪声敏感，且在多中心、多视角数据上仍需验证。

---

## 525. GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations

**arXiv ID:** 2603.10978 | [PDF](https://arxiv.org/pdf/2603.10978v1)

**作者:** Boyuan Chen `[一作]` (New York University), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11129 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过将YOLO检测得到的物体位置信息以结构化提示的形式注入视觉语言模型，缓解了计数任务中的幻觉现象。

**💡 创新点**

提出GroundCount框架，系统比较了 Prompt Augmentation 与 Feature Fusion 的效果，并揭示了不同 VLM 对结构化提示的兼容性差异。

**🔧 技术方法**

采用 YOLOv13x 目标检测、3×3 网格空间编码、对象序列化、Prompt Augmentation、FiLM 及交叉注意力等技术实现多种融合策略。

**📊 数据集**

使用 PhD 视觉问答基准进行计数子任务评估，COCO train2017 用于训练融合模型，YOLOv13x 在所有测试图像上提供检测结果。

**📈 对比分析**

在五个主流 VLM 上比较基线、Prompt、Fusion、Combined 四种方案，Prompt 方案获得最高 81.3% 的计数准确率，提升约 6–7pp，且推理时间减少约 22%。

**⚠️ 局限性**

仅适用于 CNN 检测模型，特征融合效果不佳；部分架构（如 InternVL3.5）对结构化提示不兼容；实验规模有限，未尝试 Transformer 检测或更大规模预训练。

---

## 526. When should we trust the annotation? Selective prediction for molecular structure retrieval from mass spectra

**arXiv ID:** 2603.10950 | [PDF](https://arxiv.org/pdf/2603.10950v1)

**作者:** Mira Jürgens `[一作]` (Ghent University), Willem Waegeman `[通讯]` (Ghent University)

**通讯引用:** 4613 | [OpenAlex ID](https://openalex.org/A5028945060)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并评估了一种针对串联质谱（MS/MS）分子结构检索的选择性预测框架，允许模型在不确定性过高时拒绝输出，进而降低错误注释率。

**💡 创新点**

创新点包括：①系统性比较多种不确定性量化方法（从位级到检索级再到输入空间的距离度量），并证明检索级的第一阶置信度和二阶aleatoric不确定性在任务损失（检索命中率）上的优越性；②结合分布式风险控制（SGR）实现给定风险阈值下的可验证覆盖率，为实际应用提供理论保障。

**🔧 技术方法**

使用的技术包括：基于多层感知机的指纹预测器；深度集成、MC Dropout和Laplace近似等第二阶不确定性估计；温度缩放的softmax、余弦相似度用于候选排序；分布式风险控制算法SGR；以及标准的风险‑覆盖曲线、AURC、相对AURC等评价指标。

**📊 数据集**

评估数据集为MassSpecGym benchmark，包含231 104条MS/MS谱与对应的4096维Morgan指纹，采用基于分子相似度的训练/验证/测试拆分。

**📈 对比分析**

通过与oracle和随机基准对比，发现检索级置信度和score‑gap在K=1时获得最低相对AURC；随着K增大，rank‑variance成为最优；在风险控制设置下，使用SGR可在K=20的目标风险0.5时保留约87%的样本，且经验风险始终低于目标阈值；总体上，选择性预测显著降低了误注释率并提升了覆盖率。

**⚠️ 局限性**

主要局限：实验仅使用单一MLP基线架构，未验证更强大模型（如Transformer）下的结果；不确定性估计未考虑指纹的极度稀疏性或直接在谱输入空间量化；仅提供均值风险控制，未覆盖假发现率（FDR）或置信集合等更严格错误控制方法。

---

