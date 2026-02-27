# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-27 | 今日论文总数: 576

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Same Words, Different Judgments: Modality Effects on Preference Alignment

**arXiv ID:** 2602.22710 | [PDF](https://arxiv.org/pdf/2602.22710v1)

**作者:** Aaron Broukhim `[一作]` (University of California San Diego), Eshin Jolly `[通讯]` (University of California San Diego)

**通讯引用:** 1353 | [OpenAlex ID](https://openalex.org/A5050249367)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对100个对话交互，比较文本和音频两种模式下的人类偏好标注与合成偏好标注，评估其可靠性和差异。

**💡 创新点**

首次用ICC量化文本与音频偏好标注可靠性，并发现两种模式在决策阈值、长度偏差和用户导向评价上的显著差异；证明合成评分可替代或辅助人类标注。

**🔧 技术方法**

使用混合效应模型、ICC、Krippendorffα、线性回归，GPT‑4o/ GPT‑4o‑audio‑preview生成合成评分，TTS（kokoro）将文本转为音频。

**📊 数据集**

PRISM无监督对话数据（unguided），经kokoro TTS生成音频；收集的人工评估来自Prolific平台。

**📈 对比分析**

通过ICC(2,k)≈0.82、Krippendorffα≈0.31等指标表明音频与文本标注在聚合后可靠性相当；合成评分与人类的MAE仅≈2点，且可预测人类一致性，降低成本。

**⚠️ 局限性**

仅使用TTS合成语音，缺少自然语音的情感与停顿；声纹单一、样本来自美国英语母语者；排除了有害内容，未评估对安全/拒绝情境的影响。

---

## 2. Interface Framework for Human-AI Collaboration within Intelligent User Interface Ecosystems

**arXiv ID:** 2602.22343 | [PDF](https://arxiv.org/pdf/2602.22343v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 3. IBCircuit: Towards Holistic Circuit Discovery with Information Bottleneck

**arXiv ID:** 2602.22581 | [PDF](https://arxiv.org/pdf/2602.22581v1)

**作者:** Tian Bian `[一作]` (Chinese University of Hong Kong), Jia Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 15133 | [OpenAlex ID](https://openalex.org/A5108050433)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对Transformer模型的电路发现问题，提出了基于信息瓶颈的端到端方法IBCircuit，用噪声注入和可学习的权重全局优化电路。

**💡 创新点**

创新点在于将信息瓶颈原理应用于电路发现，统一了节点与边的参数化，消除了任务专用的损坏激活设计，实现了更精确、稀疏的电路识别。

**🔧 技术方法**

使用信息瓶颈（IB）变分近似、可学习的高斯噪声注入、梯度优化，以及对互信息和KL损失的联合约束。

**📊 数据集**

在GPT‑2模型上评估，主要使用两类任务：Indirect Object Identification (IOI) 和 Greater‑Than，涉及的文本样本来自公开语言模型数据集。

**📈 对比分析**

与SP、ACDC、AP、EAP等基线对比，IBCircuit在IOI任务中显著提升Logit Difference、降低KL散度，边缘选择上在IOI与Greater‑Than均优于基线；在大模型（GPT‑2 XL）上亦保持可比性能。

**⚠️ 局限性**

局限性包括对Greater‑Than任务的表现略逊于ACDC，稀疏化过程需调节α以平衡KL散度与电路尺寸，极端稀疏时电路可信度下降。

---

## 4. MiroFlow: Towards High-Performance and Robust Open-Source Agent Framework for General Deep Research Tasks

**arXiv ID:** 2602.22808 | [PDF](https://arxiv.org/pdf/2602.22808v1)

**作者:** Shiqian Su `[一作]` (Tsinghua University), Jifeng Dai `[通讯]` (Tsinghua University)

**通讯引用:** 34703 | [OpenAlex ID](https://openalex.org/A5026944066)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了MiroFlow，一个面向深度研究任务的高性能、稳健且可复现的开源Agent框架。

**💡 创新点**

创新点包括：三层层级架构与可视化的Agent Graph实现灵活任务编排；可选的深度推理模式（Ensemble/Verification）提升推理质量；稳健工作流设计（消息规范化、重试机制、故障隔离）保障实验可复现性。

**🔧 技术方法**

技术手段主要有：LLM多模态后端（GPT‑5、Claude 3.7 Sonnet、MiroThinker 等）、MCP工具接口、基于结构化消息的Agent交互、重试与回退策略、以及统一的输入/输出预/后处理模块。

**📊 数据集**

使用的数据集包括 GAIA、BrowseComp‑EN/ZH、HLE、xBench‑DeepSearch、FutureX、Humanity’s Last Exam 等多种公开benchmark。

**📈 对比分析**

评测采用各benchmark官方协议和 avg@3 等指标，MiroFlow 在所有列举的benchmark上均实现SOTA，性能稳定且不需任务特定调优，显著优于现有开源和商业Agent框架。

**⚠️ 局限性**

局限性在于仍依赖高成本的商业LLM后端，深度推理模式在资源使用上较高，对多任务并行调度仍需人工图定义，且在极端复杂或实时交互场景下的鲁棒性与自适应能力尚有提升空间。

---

## 5. An $\mathcal{O}(\log N)$ Time Algorithm for the Generalized Egg Dropping Problem

**arXiv ID:** 2602.22870 | [PDF](https://arxiv.org/pdf/2602.22870v1)

**作者:** Kleitos Papadopoulos `[一作]` `[通讯]`, Kleitos Papadopoulos

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出一种基于二项式上界与帕斯卡恒等式的解析方法，直接计算最优测试次数 T* 并可在 O(1) 空间内重构最优决策序列。

**💡 创新点**

创新点在于：①利用连续根近似在 O(1) 时间内给出 T* 的紧致上界；②证明误差仅为 O(K)，从而仅需 O(K) 步的线性搜索即可得到精确结果；③通过帕斯卡递推实现 O(1) 空间的决策路径重现，彻底消除传统状态转移矩阵。

**🔧 技术方法**

主要技术包括：信息论下界分析、二项式系数近似与根估计、帕斯卡恒等式递推、基于连续对数伽玛函数的高精度计算。

**📊 数据集**

由于论文为理论算法研究，未使用任何实验数据集，主要针对 N 大（如 10^18）和 K ≤ log₂(N+1) 的情形进行分析。

**📈 对比分析**

与传统 O(K·N²)、O(K·N)、以及 O(K log N) 的方法相比，本文在最坏情况下实现了 O(min(K, log N)) 的时间复杂度，空间复杂度降至 O(1)。

**⚠️ 局限性**

局限性包括：①算法依赖 K < log₂(N+1) 的前提；②仅给出理论证明，缺乏大规模实验验证；③在实际实现中对大整数运算和高精度对数/伽玛函数的数值稳定性有一定要求。

---

## 6. A Perspective on Open Challenges in Deformable Object Manipulation

**arXiv ID:** 2602.22998 | [PDF](https://arxiv.org/pdf/2602.22998v1)

**作者:** Ryan Paul McKennaa `[一作]`, John Oyekan `[通讯]` (University of York)

**通讯引用:** 1586 | [OpenAlex ID](https://openalex.org/A5046108772)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了变形物体操纵（DOM）领域的主要挑战、关键技术以及最新研究进展，重点讨论了感知、建模与控制三大模块，并给出了未来研究方向。

**💡 创新点**

创新点在于提出将多摄像头、主动视觉与触觉感知融合以减轻遮挡问题；强调可区分任务规范与差分物理仿真在RL/IL中的重要性；并指出利用图神经网络与神经辐射场（NERF）实现高效、高分辨率的场景重建与任务规划。

**🔧 技术方法**

使用的技术包括：多模态视觉/触觉感知、基于粒子或网格的质量-弹簧和位置基础动力学（PBD）、连续介质力学与有限元、差分物理仿真（如 DiffSRL、DaXBench）、强化学习与模仿学习、域随机化与Sim2Real迁移、图神经网络与NERF场景重建。

**📊 数据集**

主要引用了公开的DOM数据集与仿真环境，如ClothNet、T-shirt Folding、DLO/Folding、DefGraspSim、Isaac Gym、SOFA、Bullet 等；同时指出仍缺乏统一的高质量可注解可变形物体数据集，建议构建更丰富的模拟与真实场景混合数据集。

**📈 对比分析**

通过对比实验表明，差分强化学习方法（如 DaXBench、DiffSRL）在精度、收敛速度和跨任务泛化方面明显优于传统的 PPO、TD3 等；引入专家演示可将稀疏奖励任务的收敛时间缩短 50% 以上；图神经网络与NERF 的结合则在复杂遮挡与实时重建中显著提升鲁棒性。

**⚠️ 局限性**

局限性包括：遮挡仍在极端动态场景下难以完全消除；高质量物理仿真与差分模拟仍计算成本高，难以满足实时控制；Sim2Real 迁移仍受限于环境差异；任务规范化和目标定义仍缺乏通用标准；现有数据集缺乏多样性与可变形属性标注。

---

## 7. Orthogonal Weight Modification Enhances Learning Scalability and Convergence Efficiency without Gradient Backpropagation

**arXiv ID:** 2602.22259 | [PDF](https://arxiv.org/pdf/2602.22259v1)

**作者:** Guoqing Ma `[一作]` (Institute of Automation Chinese Academy of Sciences), Shan Yu `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了LOCO低秩聚类正交权重修改算法，用节点扰动方式实现无反向传播深度脉冲神经网络的高效训练；

**💡 创新点**

证明低秩是扰动式算法的内在属性，并通过聚类正交投影降低梯度方差、提升收敛效率；实现可训练超过10层的SNN并具备持续学习能力；

**🔧 技术方法**

节点扰动（Node Perturbation）+ 正交投影（Orthogonal Projection）+ K‑means聚类+ PCA低秩约束；使用LIF神经元架构进行双前向传播；

**📊 数据集**

MNIST手写数字、NETtalk语音音素、Imagenette图像子集；

**📈 对比分析**

与NP、STDP+SBP、SoftHebb、FA、DFA等传统非BP方法对比；LOCO在10/11层网络上保持最高准确率，收敛更快，且在持续学习任务中显著降低灾难性遗忘；

**⚠️ 局限性**

在超过15-20层时性能开始下降；缺乏批归一化、残差连接等深层训练技巧，且未进行硬件实现与能耗评估。

---

## 8. Automated Vulnerability Detection in Source Code Using Deep Representation Learning

**arXiv ID:** 2602.23121 | [PDF](https://arxiv.org/pdf/2602.23121v1)

**作者:** C. Seas `[一作]`, M. C. Carlisle `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一个卷积神经网络，用于在C源代码中自动检测安全漏洞。

**💡 创新点**

采用二进制向量化的91类token编码和自定义CNN架构，在专注C语言的基础上实现了高精度与低误报的漏洞分类，并将CWE归为五大类别以提升识别效率。

**🔧 技术方法**

深度学习CNN、token化二进制向量表示、全连接层、池化层以及多标签分类。

**📊 数据集**

Draper Labs VDISC机器标注数据、NIST SATE Juliet人标注数据以及Debian GitHub历史函数集。

**📈 对比分析**

与Russell等人基于CNN+随机森林的方法在Juliet数据集上做PR曲线对比，取得同等召回率下更高的精度；在Linux kernel实验中误报率低、召回率显著提升。

**⚠️ 局限性**

仅适用于C语言，样本规模受限于去重处理，模型对未见复杂代码的鲁棒性有限；训练耗时较长且未使用GPU。

---

## 9. Effectful Toposes and Their Lawvere-Tierney Topologies

**arXiv ID:** 2602.23086 | [PDF](https://arxiv.org/pdf/2602.23086v1)

**作者:** Rinta Yamada `[一作]` `[通讯]`, Rinta Yamada

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

引入了效应拓扑（Effectful Toposes）这一新的范畴理论框架，并研究了它们与 Lawvere‑Tierney 拓扑之间的关系，阐明了在有效拓扑背景下的效应计算与算子化（oracle）之间的对应关系。

**💡 创新点**

创新点在于：① 将效应计算模型（evidenced frame）提升到 topos 结构，形成“效应拓扑”；② 定义并分析了该拓扑上的 Lawvere‑Tierney 拓扑及其对应的 sheaf；③ 证明了双否定拓扑的 sheaf topos 与 CPS 效应下的效应拓扑同构，从而将经典 realizability 与效应计算紧密联系。

**🔧 技术方法**

采用了范畴论、类型论以及有效拓扑（effective topos）的工具，结合效应计算的 evidence frame、Lawvere‑Tierney 拓扑、CPS（Continuation‑Passing‑Style）效应以及双否定拓扑等理论技术。

**📊 数据集**

无；本文为纯理论工作，没有使用实验数据集。

**📈 对比分析**

未进行实验或性能比较；研究只给出了理论证明和结构分析。

**⚠️ 局限性**

局限性包括：① 仅针对特定的效应模型（如 CPS）给出了同构结果，尚未证明其在更一般效应类或更广泛拓扑下的适用性；② 由于是理论研究，缺乏对实际编程语言或计算机系统的实现验证。

---

## 10. Addressing Climate Action Misperceptions with Generative AI

**arXiv ID:** 2602.22564 | [PDF](https://arxiv.org/pdf/2602.22564v1)

**作者:** Miriam Remshard `[一作]` (University of Cambridge), Jon Roozenbeek `[通讯]` (University of Cambridge)

**通讯引用:** 7209 | [OpenAlex ID](https://openalex.org/A5048293280)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了个性化气候知识增强的大语言模型与网页搜索和无干预对气候行动误解与高影响行为意向的影响。

**💡 创新点**

创新点是将定制化气候知识注入LLM并通过个性化对话来纠正误解并促进高影响行为。

**🔧 技术方法**

使用了 GPT‑4.1 作为大语言模型，并对话式定制与网页搜索作为对照。

**📊 数据集**

参与者为 1201 名“Alarmed”气候关注者，使用 Ivanova 等人提供的 14 项行为 CO₂ 减排潜力作为基准数据。

**📈 对比分析**

与网页搜索和未干预组比较，个性化 LLM 在提升行动影响排名准确性和高影响行为意向方面显著优于未干预；在高影响行为意向上优于网页搜索，但在行动影响排名上差异未达到统计显著。

**⚠️ 局限性**

局限包括无法区分个性化与气候知识的独立效应、仅测量意向未跟踪实际行为、以及对“正确”评级的敏感度不足。

---

## 11. FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time

**arXiv ID:** 2602.23115 | [PDF](https://arxiv.org/pdf/2602.23115v1)

**作者:** David Dirnfeld `[一作]` (University of Massachusetts Amherst), Erik Learned-Miller `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 12695 | [OpenAlex ID](https://openalex.org/A5045674062)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种基于单位球面Hough变换的光流/特征匹配方法（FLIGHT），能够在已知相机旋转的前提下高效且鲁棒地估计单目视频的相机运动方向。

**💡 创新点**

创新点包括：①将Hough投票扩展到球面，并使用Fibonacci格点实现均匀离散；②采用两级投票（稀疏+稠密）大幅降低计算量；③对投票结果进行非线性最小二乘细化；④引入早停机制进一步提升速度；⑤在SLAM中嵌入该方法显著降低轨迹误差。

**🔧 技术方法**

主要技术：球面Hough投票、Fibonacci格点离散、投票加权（按交叉弧长）、层次化投票、非线性最小化（特征向量与平面法向量的正交性）、早停采样。

**📊 数据集**

使用KITTI、TUM RGB‑D、Sintel三大公共数据集进行实验；并在合成数据上评估鲁棒性；在PySLAM/EuRoC SLAM框架中验证集成效果。

**📈 对比分析**

与FOE、BNB、PN、2‑Points、MAGSAC++等基线相比，FLIGHT在mAA（2°、5°、10°）上均位居榜首或次席，运行时间比大多数基线至少快10‑90%，且在高噪声/大量离群点场景下表现更为稳定。

**⚠️ 局限性**

局限性：①仅估计方向，无法恢复平移幅度；②依赖已知或已估计的旋转；③在极端旋转误差（>0.15）或特征极少的情况下精度下降；④对完全静态或无纹理场景的鲁棒性仍有限。

---

## 12. ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding

**arXiv ID:** 2602.23306 | [PDF](https://arxiv.org/pdf/2602.23306v1)

**作者:** Yiran Guan `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 38620 | [OpenAlex ID](https://openalex.org/A5039363991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ThinkOmni，一个无需训练的推理时框架，利用现成的大型推理模型（LRM）在多模态大语言模型（OLLM）解码时进行引导，自动调整感知与推理的权重以提升多模态推理性能。

**💡 创新点**

创新点在于：① LRM-as-a-Guide 机制，将文本推理模型作为引导者在多模态解码中注入推理偏好；② Stepwise Contrastive Scaling 自适应动态平衡感知与推理贡献，消除手工超参数调优的需求；③ 通过训练免费、推理时的混合逻辑实现了对多模态推理的显著提升。

**🔧 技术方法**

采用了推理时对比解码（Contrastive Decoding）框架、Jensen–Shannon 距离度量来评估感知与推理差异、软最大化（Softmax）对logit进行混合、以及多步动态缩放系数的归一化。

**📊 数据集**

使用了六个多模态推理基准：MathVista、MathVision、MathVerse、MMAU、Daily-Omni、OmniBench，总计约10,000个测试样本。

**📈 对比分析**

与多种训练免费方法（平均logit融合、Caption-then-Answer、Visual Contrastive Decoding）以及基于强化学习的 RFT 模型进行对比，ThinkOmni 在所有基准上均实现显著提升，部分任务如 MathVision 的得分提升超过 7%，并可与甚至超过已进行 RFT 的模型竞争。

**⚠️ 局限性**

限制包括：需共享词表以实现logit混合；额外的前向推理会带来计算延迟；对 LRM 的依赖使得模型性能受限于 LRM 的文本推理能力；并且无法完全避免推理链中的偏见与幻觉。

---

## 13. Differentially Private Data-Driven Markov Chain Modeling

**arXiv ID:** 2602.22443 | [PDF](https://arxiv.org/pdf/2602.22443v1)

**作者:** Alexander Benvenuti `[一作]` (Georgia Institute of Technology), Matthew Hale `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 508 | [OpenAlex ID](https://openalex.org/A5022644062)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一种基于 Dirichlet 机制的差分隐私方法，用于对以用户数据为基础的马尔可夫链模型的转移矩阵进行隐私保护，并提供了对模型稳态分布与收敛速度的误差上界。

**💡 创新点**

①首次将 Dirichlet 机制扩展至数据库查询的单位单纯形输出并证明满足 (ϵ,δ)-差分隐私；②通过并行组合实现对完整转移矩阵的隐私保证；③给出稳态分布和 ergodicity 系数的解析误差界。

**🔧 技术方法**

差分隐私、Dirichlet 机制、KL 散度分析、并行组合、马尔可夫链理论、稳态分布与收敛系数分析。

**📊 数据集**

大学课程成绩分布（98名学生）和纽约市 2025 年 1 月出租车行程数据（约 293 万条记录）。

**📈 对比分析**

通过数值仿真评估 KL 散度和总变距离，发现即使在 (3.73,3×10⁻⁶)-隐私下，稳态分布误差低于 2%，且 KL 散度随 ϵ 增大趋于真实值，说明方法在保持高精度的同时提供强隐私。

**⚠️ 局限性**

仅考虑事件级别隐私；未解决用户级隐私；仅适用于已知转移计数且每行至少出现一次的马尔可夫链；对极端稀疏或小样本的转移矩阵精度可能下降。

---

## 14. Learning Physical Operators using Neural Operators

**arXiv ID:** 2602.23113 | [PDF](https://arxiv.org/pdf/2602.23113v1)

**作者:** Vignesh Gopakumar `[一作]` (UCL Centre for Artificial Intelligence), Marc Peter Deisenroth `[通讯]` (UCL Centre for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于算子分解的神经算子（OpsSplit），通过将 PDE 拆分为线性与非线性物理算子，分别使用固定卷积（有限差分）和神经算子来学习非线性部分，并将其构造为神经 ODE 进行连续时间预测。

**💡 创新点**

创新点在于：① 直接学习物理算子而非解算子；② 通过算子分解实现模块化和可插拔的专家网络；③ 在预测阶段自洽地将 PDE 约束嵌入算子组合中，提升对未知物理参数与时间外推的泛化能力。

**🔧 技术方法**

技术核心包括：算子分解（Operator Splitting）、混合专家网络（Mixture‑of‑Experts）神经算子、固定卷积模拟线性算子、神经 ODE 训练与求解、有限差分卷积核、以及对不规则网格的图神经算子扩展。

**📊 数据集**

数据集：使用标准 2D Navier–Stokes 训练/测试数据（包括不可压与可压、不同粘度/波动参数），以及不规则几何的 CylinderFlow 流动数据；所有样本均由高精度谱/有限差分求解器生成。

**📈 对比分析**

与基线比较：在不可压、可压 Navier–Stokes 以及不规则网格上分别与自回归、神经 ODE、传统 FNO、U‑Net、ViT、UNO、CNO、GNO 等架构比较。OpsSplit 在所有架构中在分布内、时间外推、分布外和两者结合的场景下的 NRMSE 均优于基线，尤其在 OOD 与时间外推中显著降低误差。缺点是训练时间略长（需要多网络前向传播），但参数量保持与基线相当。

**⚠️ 局限性**

局限性：① 需要先验知识进行算子拆分，缺乏通用拆分准则；② 目前仅在规则矩形网格下验证，扩展到任意几何需要进一步设计；③ 对于高度耦合、多物理系统，拆分后每个算子可能导致参数量和计算开销激增；④ ODE 求解相比自回归成本更高。

---

## 15. TFPS: A Temporal Filtration-enhanced Positive Sample Set Construction Method for Implicit Collaborative Filtering

**arXiv ID:** 2602.22521 | [PDF](https://arxiv.org/pdf/2602.22521v1)

**作者:** Jiayi Wu `[一作]`, Guoren Wang `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于时间滤波的正样本集合构造方法（TFPS），用于改进隐式协同过滤中的负采样策略。

**💡 创新点**

创新点在于：①从正样本角度引入时间衰减模型和图滤波技术，将时间间隔信息显式编码；②通过分层增强机制将最近交互重复出现，提升对当前兴趣的学习；③仅在数据层面调整正样本，无需修改模型结构。

**🔧 技术方法**

采用指数时间衰减、图滤波（分层子图）、层增强复制、BPR损失以及LightGCN等基础协同过滤模型。

**📊 数据集**

在Ta-Feng、LastFM和AmazonCDs三个真实隐式交互数据集上进行实验。

**📈 对比分析**

与12个负采样或正采样基线以及STAM等时序模型进行对比；TFPS在Recall@20/30和NDCG@20/30上均显著优于所有基线，且在训练时间和构造效率上更高。

**⚠️ 局限性**

局限性包括：对极稀疏或噪声严重的数据集效果不一定最佳；层增强若过度会导致正样本分布失衡，可能影响泛化；与基于序列的时序模型结合时有时性能下降。

---

## 16. Interpreting and Steering State-Space Models via Activation Subspace Bottlenecks

**arXiv ID:** 2602.22719 | [PDF](https://arxiv.org/pdf/2602.22719v1)

**作者:** Vamshi Sunku Mohan `[一作]` (Center for Cybersecurity Systems and Networks, Amrita University), Chandan Singh `[通讯]` (Microsoft Research)

**通讯引用:** 3370 | [OpenAlex ID](https://openalex.org/A5017514239)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Mamba及其他SSM模型进行机制解释，发现激活子空间瓶颈并通过后置放大调节或轻量化结构改造提升性能。

**💡 创新点**

首次在SSM中识别激活子空间瓶颈并利用后置放大调节实现无任务调优的性能提升；提出Stable-Mamba轻量改造，实现长上下文性能显著提升。

**🔧 技术方法**

使用稀疏自编码器+字典学习、Stochastic Parameter Decomposition、Delta‑sensitive 子空间筛选、激活子空间定义、后置放大调节以及轻量化结构改造等技术。

**📊 数据集**

使用The Pile进行调优与Wikitext‑2进行瓶颈分析；在TriviaQA、SQuAD、MuSiQue、IFEval、RULER、DROP等六大基准上进行评估。

**📈 对比分析**

与Vanilla Mamba、Steered Mamba、Stable‑Mamba、Mamba‑2、DenseMamba、Hyena、MiniPLM‑Mamba、GPT‑2 Small等模型比较，平均提升约8.27%，在长上下文基准上Stable‑Mamba平均提升约15–35个百分点。

**⚠️ 局限性**

主要聚焦Mamba，验证范围仅限文本任务，未充分验证更大规模模型或其他模态的泛化；仅使用有限的解释指标，可能无法捕捉更深层次的瓶颈。

---

## 17. LoR-LUT: Learning Compact 3D Lookup Tables via Low-Rank Residuals

**arXiv ID:** 2602.22607 | [PDF](https://arxiv.org/pdf/2602.22607v1)

**作者:** Ziqi Zhao `[一作]` (University of Texas at Austin), Shounak Roychowdhury `[通讯]` (University of Texas at Austin)

**通讯引用:** 524 | [OpenAlex ID](https://openalex.org/A5038523872)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出LoR-LUT框架，将低秩残差加入3D LUT，兼顾紧凑与可解释。

**💡 创新点**

创新点在于用低秩CP分解的残差代替或补充密集基准LUT，实现参数大幅压缩而保持三线性插值复杂度。

**🔧 技术方法**

采用低秩张量分解、轻量化CNN+MLP预测器、三线性插值等技术。

**📊 数据集**

使用MIT-Adobe FiveK数据集进行训练与评估。

**📈 对比分析**

与IA-3DLUT、AdaInt、SepLUT等基准对比，LoR-LUT在PSNR、SSIM、LPIPS等指标上相当或优于，并且参数仅为几十千，速度实时。

**⚠️ 局限性**

局限在于目前仅实现全局色彩调整，对空间自适应处理和更复杂场景的局部调整支持不足。

---

## 18. XMENTOR: A Rank-Aware Aggregation Approach for Human-Centered Explainable AI in Just-in-Time Software Defect Prediction

**arXiv ID:** 2602.22403 | [PDF](https://arxiv.org/pdf/2602.22403v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 19. Interface-Aware Trajectory Reconstruction of Limited Demonstrations for Robot Learning

**arXiv ID:** 2602.23287 | [PDF](https://arxiv.org/pdf/2602.23287v1)

**作者:** Demiana R. Barsoum `[一作]` (Northwestern University and Shirley Ryan AbilityLab), Brenna D. Argall `[通讯]` (Northwestern University and Shirley Ryan AbilityLab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于接口感知的轨迹重构算法，将低维接口演示提升到机器人完整的控制空间，支持自适应重构并兼顾环境与任务约束；

**💡 创新点**

创新点在于将接口、任务与环境约束融合到轨迹重构过程，自动剔除接口诱导的限制，从而生成更高维度、更高效的演示；

**🔧 技术方法**

使用轨迹分段、模式识别、时间对齐、状态拼接和环境约束一致化等技术，并在学习阶段采用行为克隆的多层感知器；

**📊 数据集**

使用两台7-DoF机械臂（xArm7 与 Kinova Gen2 Jaco）在六个日常生活任务（Pick‑and‑Place、Pour、Peg‑in‑hole、Stir、Brush Hair、Cut Food）上，采集 1–5 次成功演示（最多 20 次尝试）的接口限定演示数据；

**📈 对比分析**

将重构演示与原始演示及平滑滤波（Butterworth）演示在执行时间、路径长度和学习误差（MSE）三方面对比，结果显示重构演示平均可提升 25–35% 的执行速度、10–20% 的路径效率，学习误差与原始相当或略优；

**⚠️ 局限性**

局限包括仅由单一非终端演示者完成实验，缺乏终端用户验证；算法未考虑机器人动力学和关节极限，导致某些任务在 xArm7 上因关节约束失败；对高精度任务（如 Peg‑in‑hole）泛化仍有限；

---

## 20. OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis

**arXiv ID:** 2602.22949 | [PDF](https://arxiv.org/pdf/2602.22949v1)

**作者:** Junuk Cha `[一作]` (Korea Advanced Institute of Science and Technology), Han-Mu Park `[通讯]` (Korea Electronics Technology Institute)

**通讯引用:** 99 | [OpenAlex ID](https://openalex.org/A5017746540)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 OpenFS，包含多手可识别手势的指拼识别器（采用双层位置编码、签字手聚焦损失和单调对齐损失）以及帧级字母条件扩散生成器，用以合成 OOV 指拼数据并构建 FSNeo benchmark。

**💡 创新点**

创新点：①通过双层位置编码与签字手聚焦损失实现隐式签字手检测；②用单调对齐损失替代 CTC，保证时间顺序一致；③引入粗细化帧级字母注释与帧级字母条件生成器，精细控制每帧手势；④构建 FSNeo 评测集并实现无后处理的实时识别。

**🔧 技术方法**

技术手段：Transformer 编码解码器、双层位置编码（手识别+时间），签字手聚焦熵损失、单调对齐损失、基于扩散的帧级字母条件生成器、粗细化注释方法、MSE 训练。

**📊 数据集**

使用的数据集：ChicagoFSWild、ChicagoFSWildPlus、从英语词表生成的合成数据、FSNeo 语义合成基准。

**📈 对比分析**

与 Shi、FSS-Net、PoseNet 等方法对比：在 CFSW 上字母准确率 75.4%（vs 61.6%），在 FSNeo 上 80.5%（vs 61.2%），OOV 准确率 97.6%（vs 61.2%），签字手检测 99.9%（vs 90.4%），推理速度提升约 100 倍（962 FPS vs 6 FPS）。

**⚠️ 局限性**

局限性：仍依赖姿态估计质量，缺乏对极短视频中模糊签字手的处理，合成数据的真实性和多样性可能影响泛化，未针对 RGB 直接方法进行全面评估。

---

## 21. GFRRN: Explore the Gaps in Single Image Reflection Removal

**arXiv ID:** 2602.22695 | [PDF](https://arxiv.org/pdf/2602.22695v1)

**作者:** Yu Chen `[一作]` (Zhejiang University), Zheming Lu `[通讯]` (Zhejiang University)

**通讯引用:** 5484 | [OpenAlex ID](https://openalex.org/A5100722511)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Gap-Free Reflection Removal Network（GFRRN），在单图像反射去除任务中实现更高质量的反射分离与目标图像恢复。

**💡 创新点**

创新点包括：① 将Mona层嵌入预训练的Swin-Transformer中进行参数高效微调（Mona‑tuning），有效缩小语义鸿沟；② 统一标签生成器，将合成与真实数据的反射标签统一为低频(I−T)_low，消除训练数据差异；③ 引入高斯自适应频率学习块（G‑AFLB）和动态代理注意力（DAA），分别提升频域先验融合与窗口重要性建模；④ 将残差估计器与多损失（内容、排除、感知、重建）相结合，进一步强化恢复效果。

**🔧 技术方法**

使用的技术包括：参数高效微调（PEFT）Mona‑tuning、低频标签生成、Gaussian‑based Adaptive Frequency Learning Block、Dynamic Agent Attention、残差估计器、复合损失函数、双流编码器-单解码器架构。

**📊 数据集**

训练数据集：每个epoch随机采样5K对Pascal VOC合成数据；90对Real数据与200对Nature数据；测试集包含Real20、Object200、Postcard199、Wild55、Nature20五个公开数据集。

**📈 对比分析**

与11种SOTA单图像反射去除方法（ERRNet、IBCLN、LASIRR、YTMT、RobustSIRR、DSRNet、DURRNet、RRW、DSIT、DExNet、RDNet）在五个真实数据集上对比。GFRRN在平均PSNR上提升0.7dB、SSIM提升0.01，位居榜首，且在可视化结果上显著减少残余反射。

**⚠️ 局限性**

局限性：对极端高频或模糊反射的分离仍有残余；低频标签生成对反射与残差的分离依赖手工设定的低通滤波器；双流结构与多模块堆叠导致推理速度相对较慢；未在非图像或更大规模数据集上验证普适性。

---

## 22. Inferential Mechanics Part 1: Causal Mechanistic Theories of Machine Learning in Chemical Biology with Implications

**arXiv ID:** 2602.23303 | [PDF](https://arxiv.org/pdf/2602.23303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 23. UFO-DETR: Frequency-Guided End-to-End Detector for UAV Tiny Objects

**arXiv ID:** 2602.22712 | [PDF](https://arxiv.org/pdf/2602.22712v1)

**作者:** Yuankai Chen `[一作]` (South China Agricultural University), Meihua Wang `[通讯]` (South China Agricultural University)

**通讯引用:** 58195 | [OpenAlex ID](https://openalex.org/A5100329133)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出轻量级端到端目标检测框架UFO‑DETR，专为UAV图像小目标设计，采用LSKNet骨干、DAttention‑AIFI模块和跨空间频率模块DynFreq‑C3。

**💡 创新点**

创新点：1）将Large Selective Kernel Network作为轻量化骨干降低参数和计算量；2）在AIFI模块中引入可变形注意力DAttention，实现多尺度动态采样；3）设计跨空间频率模块DynFreq‑C3，利用频域高频信息提升小目标感知。

**🔧 技术方法**

技术：Transformer端到端检测、LSKNet、Deformable Attention、频域卷积（FDConv）、跨尺度特征融合、YOLO/RT‑DETR基线对比。

**📊 数据集**

数据集：VisDrone2019 UAV图像数据集（训练6,471张/验证548张/测试3,190张）。

**📈 对比分析**

通过与YOLO系列和RT‑DETR不同变体的对比实验验证效果；UFO‑DETR在mAP50 46.1%、Precision 59.2%、Recall 44.5%、GFLOPs 41.8、模型尺寸 28.3 MB，较RT‑DETR‑L提升mAP约2.6%、Recall 1.9%、Precision 0.2%，且显著降低计算量和模型大小。

**⚠️ 局限性**

限制：仍存在RT‑DETR中位置关系解码器的冗余计算开销，未来需进一步优化；在极端复杂背景或极小目标情况下性能仍有提升空间。

---

## 24. Forecasting Antimicrobial Resistance Trends Using Machine Learning on WHO GLASS Surveillance Data: A Retrieval-Augmented Generation Approach for Policy Decision Support

**arXiv ID:** 2602.22673 | [PDF](https://arxiv.org/pdf/2602.22673v1)

**作者:** Md Tanvir Hasan Turja `[一作]` `[通讯]` (Middlesex University), Md Tanvir Hasan Turja (Middlesex University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究提出并验证了一个两组件框架，先使用WHO GLASS监测数据训练机器学习模型预测抗菌素耐药率，然后利用检索增强生成（RAG）系统结合预测结果回答政策相关问题。

**💡 创新点**

创新点在于首次将GLASS数据用于群体层面耐药率时间序列预测，并将预测与WHO政策文档结合的本地化RAG问答系统集成，避免了云端依赖与幻觉。

**🔧 技术方法**

使用XGBoost、LightGBM、LSTM等机器学习模型以及Phi‑3 Mini LLM、ChromaDB向量检索、句子变换器嵌入构建RAG。

**📊 数据集**

使用2021–2023年WHO Global Antimicrobial Resistance and Use Surveillance System（GLASS）共5909条观察记录。

**📈 对比分析**

对六种模型（Naive、Linear、Ridge、XGBoost、LightGBM、LSTM）在2023年测试集上进行MAE/RMSE/R²比较，XGBoost取得MAE 7.07%，R² 0.854，优于基线83.1%。

**⚠️ 局限性**

局限包括仅有三年数据导致序列深度不足，GLASS覆盖不均衡产生地区偏差，RAG知识库规模有限，Phi‑3 Mini推理能力受限。

---

## 25. Safety First: Psychological Safety as the Key to AI Transformation

**arXiv ID:** 2602.23279 | [PDF](https://arxiv.org/pdf/2602.23279v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 26. Decomposing Private Image Generation via Coarse-to-Fine Wavelet Modeling

**arXiv ID:** 2602.23262 | [PDF](https://arxiv.org/pdf/2602.23262v1)

**作者:** Jasmine Bayrooti `[一作]` (University of Cambridge), Amanda Prorok `[通讯]` (University of Cambridge)

**通讯引用:** 2615 | [OpenAlex ID](https://openalex.org/A5066624177)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于低频小波分量的两阶段差分隐私文本生成图像框架DP‑Wavelet，先在低分辨率的粗粒度语义图像上进行DP微调，再利用公开的超分辨率模型完成细节生成。

**💡 创新点**

创新点在于：①将隐私预算聚焦于低频、能量集中且语义信息丰富的LL₀子带，显著提高DP优化的信噪比；②通过离散小波变换与自回归图像分词器（AR‑SIT）的分层结构实现可分解的“粗细分辨”生成；③利用DP后处理属性，将高频细节的生成留给公开模型，避免额外隐私成本。

**🔧 技术方法**

使用技术包括：差分隐私梯度裁剪与噪声注入（DP‑SGD/DP‑Adam/DP‑LoRA）；离散二维小波变换（Haar小波）进行图像分解；自回归语义图像分词器AR‑SIT；公共超分辨率模型进行后期细化；以及DP后处理和隐私会计器（PLD）。

**📊 数据集**

实验数据集为 MS‑COCO（文本‑图像对）和 MM‑CelebA‑HQ（人脸‑描述），用于评估隐私等级下的生成质量。

**📈 对比分析**

与两大基线（DP‑LDM、DP‑LlamaGen）以及预训练/无DP版本对比。结果显示：在 MM‑CelebA‑HQ 上DP‑Wavelet取得最低 FID 与接近最佳 LPIPS；在 MS‑COCO 上虽然整体 FID 略高于 DP‑LDM，但在低隐私预算（ε=10,1）下保持与 DP‑LlamaGen 相近的 LPIPS，并显著提升了语义对齐和细节保留。

**⚠️ 局限性**

局限性包括：①在 MS‑COCO 上生成质量仍落后于最强基线，主要因公共上采样器缺乏针对该域的预训练；②当隐私预算非常紧（ε=1）时，高频细节仍受DP噪声影响，出现视觉伪影；③对极端复杂场景或极低频域的细粒度需求尚未充分解决。

---

## 27. LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure

**arXiv ID:** 2602.23036 | [PDF](https://arxiv.org/pdf/2602.23036v1)

**作者:** Jaehong Cho `[一作]` (KAIST), Jongse Park `[通讯]` (KAIST)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了统一的系统级仿真框架 LLMServingSim 2.0，用于模拟在异构与解耦化大语言模型（LLM）服务基础设施中的硬件–软件交互与运行时动态。

**💡 创新点**

创新点包括：① 采用基于运行时循环的交互感知模型，将批处理、路由、放置、迁移等服务决策与硬件行为耦合；② 将异构加速器、跨层内存、Mixture‑of‑Experts、前缀缓存、预填充/解码分离等技术统一建模；③ 通过可扩展的基于 Profile 的算子性能模型快速集成新硬件；④ 在仿真框架中加入功耗模型，支持能效评估；⑤ 与现有工具相比实现了更高的精度与更广的功能覆盖。

**🔧 技术方法**

核心技术包括：算子级性能 Profile（在 PyTorch/HuggingFace 环境下采样一次即可得到）；执行规划器（将工作负载与集群配置映射到消息服务组 MSG）；MSG 内的批调度、内存模型、功耗模型与算子映射/调度；系统仿真器基于 ASTRA‑sim 与 Chakra 扩展，支持异构通信、内存层次、PIM 等；以及统一的运行时循环实现服务决策的动态反馈。

**📊 数据集**

使用 ShareGPT 数据集生成的请求流（300 条请求、Poisson 10 RPS）以及真实模型：Llama 3.1‑8B、Llama 3.1‑70B、Phi‑mini MoE、Mixtral 8×7B；在 NVIDIA RTX A6000、H100 以及 Google Cloud TPU‑v6e‑1 上进行实验。

**📈 对比分析**

通过将仿真结果与实际 vLLM 部署在同一硬件平台进行对比，指标涵盖吞吐量、TTFT、TPOT、内存占用与功耗；平均误差约为 0.97%（吞吐量 5.66%/2.98%，功耗 1.34%）。与 Vidur、APEX、TokenSim、原始 LLMServingSim 进行基准比较，LLMServingSim 2.0 在覆盖范围和精度上优于所有基线，且在多实例、前缀缓存、预填充/解码分离、MoE 等复杂场景下仍保持低误差；相对轻量级仿真器的运行时略长，但可在约 10 分钟内完成复杂配置的仿真。

**⚠️ 局限性**

主要局限包括：1）仿真时间相对较长，尤其在大规模异构配置下；2）对某些尚未实现的硬件功能（如 TPU 上的前缀缓存或 PD 分离）仅支持假设性分析；3）目前重点针对推理阶段，未覆盖训练相关的动态特性；4）需要先完成算子级 Profile，若缺乏高质量 Profile 可能影响精度。

---

## 28. Generative Recommendation for Large-Scale Advertising

**arXiv ID:** 2602.22732 | [PDF](https://arxiv.org/pdf/2602.22732v1)

**作者:** Ben Xue `[一作]` (Kuaishou Technology), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一套面向大规模实时广告的生成式推荐系统 GR4AD，涵盖统一广告语义 ID、懒惰自回归解码器、价值感知监督学习与排名导向强化学习，以及动态束搜索与实时索引等组件。

**💡 创新点**

创新点包括：① 基于指令调优与共现学习的统一广告语义 ID（UA‑SID）实现对多模态广告信息的端到端编码；② 懒惰自回归（LazyAR）解码器在保持效果的同时显著提升解码吞吐；③ 价值感知监督学习（VSL）与列表级 RL（RSPO）在同一在线学习框架下动态平衡用户兴趣与商业收益；④ 动态束搜索（DBS）结合流量感知自适应束宽，实现高效实时生成。

**🔧 技术方法**

技术方法包括：指令调优的多模态 LLM（Qwen3‑VL‑7B）+共现学习；多粒度多分辨率 RQ‑Kmeans 量化；懒惰自回归解码器；价值加权交叉熵与 eCPM 预测；列表级 RL 目标 RSPO；动态束宽调度、KV 共享、Top‑k 预剪枝、FP8 量化；实时双向索引与在线学习循环。

**📊 数据集**

数据集主要来自抖音/快手真实广告创意与用户交互日志，包含数千万条广告创意和数亿次曝光；离线评估使用广告图像检索 recall 数据；上线评估为大规模 A/B 测试，覆盖 4 亿活跃用户。

**📈 对比分析**

与 DLRM、OneRec‑V2 等基线对比，GR4AD 在同等硬件下实现平均 4.2% 广告收入提升，QPS 提升 20% 以上，推理延迟保持 <100 ms；每个子模块的 ablation 进一步验证其对收入、吞吐和代码碰撞率的正向影响。

**⚠️ 局限性**

局限性包括：① 对大规模实时数据流的依赖，训练与推理的持续运维成本较高；② 需要专业的指令调优与共现学习，模型可解释性相对有限；③ 对极端低频广告或全新广告主的冷启动仍有挑战；④ 现有系统尚未覆盖多语言或多地域多维度的全局优化。

---

## 29. CLIP Is Shortsighted: Paying Attention Beyond the First Sentence

**arXiv ID:** 2602.22419 | [PDF](https://arxiv.org/pdf/2602.22419v1)

**作者:** Marc-Antoine Lavoie `[一作]` (University of Toronto), Steven L. Waslander `[通讯]` (University of Toronto)

**通讯引用:** 10062 | [OpenAlex ID](https://openalex.org/A5024242059)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DeBias-CLIP，通过去除长文本的首句总结、随机采样剩余句子并在文本前填充来改进 CLIP 的长文本检索，并保持短文本性能。

**💡 创新点**

创新点在于：① 用长文本去掉首句作为短文本训练，消除总结句的“捷径”；② 随机采样剩余句子以打破句子顺序；③ 在短文本前添加填充令位置注意力均匀；④ 仅改动训练数据，无新增可训练参数。

**🔧 技术方法**

使用 CLIP 预训练模型、Long-CLIP 框架、对比损失、位置嵌入伸展、句子采样、填充以及短/长文本损失权重 λ^s 的组合。

**📊 数据集**

训练数据：ShareGPT4V（≈1.2M 图像+LLM 生成长标题）。评估数据集：Urban1k、DCI、Long‑DCI、DOCCI、COCO、Flickr30k 等长短文本检索基准。

**📈 对比分析**

与 Long‑CLIP、TULIP、FineLIP、SmartCLIP、Fix‑CLIP、LongD‑CLIP 等方法对比；在长文本检索上大幅提升（Urban1k T2I +5–6%，DOCCI 近乎 FineLIP），短文本检索保持或提升；对句子置换和去除总结句更鲁棒；在不同 CLIP 变体上均稳健提升。

**⚠️ 局限性**

局限性：长文本基准多采用“summary‑first”结构，可能掩盖位置偏差；未在更真实多段落检索场景中验证；对预训练产生的早期 token 偏差仍有残留。

---

## 30. Beyond performance-wise Contribution Evaluation in Federated Learning

**arXiv ID:** 2602.22470 | [PDF](https://arxiv.org/pdf/2602.22470v1)

**作者:** Balazs Pejo `[一作]` `[通讯]` (CrySyS Lab), Balazs Pejo (CrySyS Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对联邦学习中客户端贡献进行多维度评估，包括准确率、公平性、可靠性和鲁棒性。

**💡 创新点**

首次将Shapley值与多维度信任指标相结合，证明单一指标无法全面衡量客户端贡献。

**🔧 技术方法**

使用基于梯度的近似Shapley值（GTG）和留一法（LOO）以及PGD攻击、噪声扰动等评估方法。

**📊 数据集**

在CIFAR-10、IMDB和Adult三大数据集上实验。

**📈 对比分析**

通过对比不同评估方法和指标的Spearman相关性与RMSE，发现性能与其他指标几乎不相关，说明评价体系需多维。

**⚠️ 局限性**

局限在于只考虑了三种指标，未探究不同公平/鲁棒度定义对结果影响，且实验规模受限于小样本客户端。

---

## 31. RepSPD: Enhancing SPD Manifold Representation in EEGs via Dynamic Graphs

**arXiv ID:** 2602.22981 | [PDF](https://arxiv.org/pdf/2602.22981v1)

**作者:** Haohui Jia `[一作]` (Information Science and Technology), Yasushi Sakurai `[通讯]` (SANKEN)

**通讯引用:** 3057 | [OpenAlex ID](https://openalex.org/A5089668362)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种融合动态图神经网络与SPD黎曼流形的深度学习框架，用图引导的跨注意力机制在Riemannian流形上对EEG的协方差表示进行自适应调制，并通过几何一致性对齐损失将图结构与流形嵌入对齐；

**💡 创新点**

创新点在于：①首次将动态图结构直接映射到SPD流形并通过跨注意力实现几何一致的特征融合；②提出几何-拓扑一致性对齐损失，促进图嵌入与流形嵌入在切空间的一致性；③通过对齐损失提升模型在切空间的判别性与鲁棒性；

**🔧 技术方法**

技术上使用SPDNet网络实现SPD层的前向传播，Log‑Euclidean距离作为注意力相似度，双线性映射生成键/值/查询，跨注意力权重通过softmax聚合得到增强SPD；再通过ReEig + log映射得到切空间表示，结合交叉熵与对齐损失进行端到端训练；

**📊 数据集**

实验数据集包括大规模癫痫检测集Temple University Hospital EEG Seizure (TUSZ) 和四类运动想象任务集BCIC-4-2a；

**📈 对比分析**

与DCRNN、BIOT、CNN‑LSTM、SPDNet、MAtt、FBCNet、Graph‑CSPNet等多种基线对比，TUSZ上模型取得86.43%准确率（相较最高基线81.68%提升约4.8%），BCIC-4-2a上获得81.05%准确率（高于Graph‑CSPNet的78.82%），在F1、AUROC等指标上也显著优于对比方法；

**⚠️ 局限性**

局限性主要在于：①图结构在训练过程中固定，未实现与模型权重同步学习；②对动态图构造的依赖可能导致对噪声敏感；③模型相对较复杂，计算开销和超参数调优较高。

---

## 32. Input-Envelope-Output: Auditable Generative Music Rewards in Sensory-Sensitive Contexts

**arXiv ID:** 2602.22813 | [PDF](https://arxiv.org/pdf/2602.22813v1)

**作者:** Cong Ye `[一作]` (Wenzhou-Kean University), Xiangbo Zhang `[通讯]` (Georgia Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出输入‑包络‑输出（I–E–O）框架，将安全约束显式为可配置、可审计的低风险包络层，并在 Web‑端的 MusiBubbles 游戏中实现可重现的音乐奖励；

**💡 创新点**

创新点在于将安全约束转化为声明式、可配置的包络层，结合确定性裁剪与日志记录，使生成音乐既能保持因果关系，又能预防感知过载；

**🔧 技术方法**

采用声明式约束、确定性裁剪、模式级奖励映射、日志审计与可重现会话报告的系统架构；

**📊 数据集**

使用合成的 660 条动作轨迹样本进行评估；

**📈 对比分析**

通过对比无约束基线与宽松、默认、紧缩三种约束配置，评估参数裁剪率、节奏、音量等指标，展示约束的可调性与性能保持；

**⚠️ 局限性**

仅验证工程正确性，未进行临床或用户体验评估，且仅限音乐反馈，未扩展到其他感官或任务。

---

## 33. Know What You Know: Metacognitive Entropy Calibration for Verifiable RL Reasoning

**arXiv ID:** 2602.22751 | [PDF](https://arxiv.org/pdf/2602.22751v1)

**作者:** Qiannian Zhao `[一作]`, Hongzhi Yin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种元认知熵校准框架 EGPO，在不改变二元验证奖励的前提下，通过基于熵的权重对 GRPO 等 RLVR 算法的优势进行校准，从而提升大语言模型的推理能力。

**💡 创新点**

创新点：① 将模型的内在不确定性（token 级 NLL 熵）作为元认知信号引入 RLVR；② 采用异步校准（正确样本不下调，错误样本不上调），实现探索与利用的平衡；③ 对全错误组采用熵调节的负样本强化（NSR），解决优势坍塌问题。

**🔧 技术方法**

技术手段：group‑based policy optimization（GRPO）、PPO 重要性比率、token‑level NLL 熵估计、熵权重比、异步 clamp、负样本强化以及可选的重归一化。

**📊 数据集**

使用的数据集：训练集为 OpenR1‑math‑220k 的精炼子集；评测集包括 MATH‑500、AIME24、Minerva、GSM8K 和 MMLU‑STEM。

**📈 对比分析**

与 GRPO、DAPO、EDGE‑GRPO 等方法在相同的二元奖励设置下对比，EGPO 在所有基准上均取得显著提升（如 Qwen2.5‑Math‑7B 在 MATH‑500 上提升 +70.8% 以上基线，并比 EDGE‑GRPO 多出约 3.1%；在 MMLU‑STEM 上提升 +46%），同时在不同规模模型上表现出良好的泛化能力。

**⚠️ 局限性**

局限性：① 仍依赖二元验证器，无法获得更细粒度的错误反馈；② 熵代理仅基于 token 概率，对长序列或复杂推理的捕捉可能不足；③ 在极难或极长推理场景下可能出现优势坍塌，需进一步改进。

---

## 34. Predicting Tennis Serve directions with Machine Learning

**arXiv ID:** 2602.22527 | [PDF](https://arxiv.org/pdf/2602.22527v1)

**作者:** Ying Zhu `[一作]` (Georgia State University), Ruthuparna Naikar `[通讯]` (Georgia State University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5032780387)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建机器学习模型预测职业网球选手的首发方向，主要基于先前点的结果、疲劳、表现焦虑等上下文特征。

**💡 创新点**

创新点在于只使用手工标注的上下文信息即可实现约49%（男）/44%（女）的预测准确率，并发现疲劳指数对发球方向的显著影响，暗示顶级球员采用混合策略。

**🔧 技术方法**

采用多种监督学习方法（多项式逻辑回归、决策树、随机森林、支持向量机、神经网络）及特征重要性分析。

**📊 数据集**

使用公开的 Match Charting Project 数据集，涵盖 3424 男性比赛与 1916 女性比赛，筛选至少 30 场比赛的选手进行实验。

**📈 对比分析**

在同一训练/测试集上比较模型，所有方法准确率相近；平均准确率为男性约49%、女性约44%，相较于 Wei 等人仅 27.8% 的 Hawkeye 数据基准有显著提升。

**⚠️ 局限性**

局限包括样本量有限（仅 10 名男/女选手）、仅考虑三种发球方向、缺乏视频/Hawkeye 运动学数据、以及可能遗漏的其他影响因素。

---

## 35. TabDLM: Free-Form Tabular Data Generation via Joint Numerical-Language Diffusion

**arXiv ID:** 2602.22586 | [PDF](https://arxiv.org/pdf/2602.22586v1)

**作者:** Donghong Cai `[一作]` (Washington University in St. Louis), Muhan Zhang `[通讯]` (Peking University)

**通讯引用:** 4837 | [OpenAlex ID](https://openalex.org/A5071515223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了TabDLM，一种统一框架，用于生成同时包含数值、分类和自由文本字段的高保真合成表格数据。

**💡 创新点**

创新点在于将数值连续扩散与掩码语言扩散（MDLM）结合，并为数值字段设计了可学习的数值词元化，实现在同一模型中同时处理数值和文本，保持跨模态互依赖。

**🔧 技术方法**

采用的技术包括连续扩散（VE型）、MDLM（掩码语言扩散）、多层感知器（MLP）数值编码/解码、LoRA微调、双向Transformer注意力以及统一噪声时间步的混合扩散。

**📊 数据集**

实验使用了合成数据集MathExpr和ProfileBio、真实数据集Amazon（含长文本字段）以及传统表格数据集Adult、Default、Shoppers、Magic、Beijing。

**📈 对比分析**

与基线（CTGAN、TVAE、GReaT、DiffLM、TabDiff等）以及LLM基准（Qwen2.5、LLaDA-8B）比较，TabDLM在Shape、Trend、匹配率和下游MLE指标上均显著优于现有方法，尤其在跨模态一致性和趋势保留方面表现突出。

**⚠️ 局限性**

主要限制包括采样效率较低（受MDLM模型尺寸限制），以及数值与文本共享同一噪声时间步，可能在某些场景下并非最优。

---

## 36. Strategy Executability in Mathematical Reasoning: Leveraging Human-Model Differences for Effective Guidance

**arXiv ID:** 2602.22583 | [PDF](https://arxiv.org/pdf/2602.22583v1)

**作者:** Weida Liang `[一作]` (National University of Singapore), Kenji Kawaguchi `[通讯]` (National University of Singapore)

**通讯引用:** 7541 | [OpenAlex ID](https://openalex.org/A5003184366)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨了数学推理中策略使用与可执行性之间的差距，构建了人机对齐的 HM‑ReasoningBench 数据集，并提出了基于可执行性信号的策略检索框架 SSR，以提升大型语言模型在数学推理任务上的表现。

**💡 创新点**

创新点：①引入“策略可执行性”概念，区分策略出现与其在目标模型上的可操作性；②构建人机对齐的策略知识图谱并利用图神经网络学习嵌入；③提出 SSR 通过多路径检索（类别、问题迁移、语义回退）结合经验可执行性预测，动态选择并组合最适合的策略。

**🔧 技术方法**

技术细节：策略抽象与规则匹配、异构知识图谱与图神经网络、三路检索机制、Beta–Binomial 估计可执行概率、逻辑回归可执行性预测、温度标定、对比实验。

**📊 数据集**

使用的数据集：HM‑ReasoningBench（4,850 题，包含人类与模型解答对），AIME 2025，MathArena Apex，均用于训练和评估。

**📈 对比分析**

比较方法：与直接求解（DS）、上下文学习（ICL）、单源人类/模型指导（H/M）、Self‑Consistency、Least‑to‑Most Prompting、Tree‑of‑Thoughts 等基线对比；SSR 在 HM‑ReasoningBench、AIME 25 和 Apex 上均优于所有基线，最大提升为 +13 点（AIME 25）和 +5 点（Apex），在不同模型和难度层级上表现稳定一致。

**⚠️ 局限性**

局限性：需要大量已标注的策略样本和策略抽取的可靠性；可执行性评估依赖于特定提示与解码协议，可能不适用于所有模型；在极长链推理任务中仍有提升空间；对模型规模和训练数据分布的依赖未完全解决。

---

## 37. Explainability-Aware Evaluation of Transfer Learning Models for IoT DDoS Detection Under Resource Constraints

**arXiv ID:** 2602.22488 | [PDF](https://arxiv.org/pdf/2602.22488v1)

**作者:** Nelly Elsayed `[一作]` (University of Cincinnati), Nelly Elsayed `[通讯]` (University of Cincinnati)

**通讯引用:** 438 | [OpenAlex ID](https://openalex.org/A5108005607)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文基于预训练卷积神经网络进行IoT DDoS检测，采用图像化流量表示，并在受限资源环境下对模型做可解释性评估与综合性能测评。

**💡 创新点**

创新点在于将可解释性指标（Grad‑CAM 与 SHAP）与可靠性统计（MCC、Youden 指数）与计算成本（训练时长、推理延迟）结合，形成面向工业物联网的全维度评估框架。

**🔧 技术方法**

技术方法包括：图像化流量编码、迁移学习（微调预训练 CNN）、梯度可视化（Grad‑CAM）、特征归因（SHAP）以及多维性能度量（准确率、召回率、AUC、MCC、置信区间等）。

**📊 数据集**

使用 CICDDoS2019 基准数据集，将 180 条流量记录转为 60×60×3 RGB 图像，共 11 类攻击 + 正常。

**📈 对比分析**

通过对七种预训练模型（MobileNet、MobileNetV2/V3、DenseNet121/169、EfficientNet‑B0/B1）在相同实验协议下的准确率、延迟与可解释性进行对比。DenseNet169 与 MobileNet 在准确率（≈98%）和可靠性（MCC≈0.88）上表现最佳；MobileNetV3 在推理延迟最低（≈0.00053 s），适合边缘部署；EfficientNet‑B1 可靠性和可解释性均最差。

**⚠️ 局限性**

局限性包括：仅在 CICDDoS2019 训练/测试，未覆盖真实环境的未知攻击与分布漂移；未考虑对抗性攻击、加密流量及在线自适应；模型微调方案对某些架构（如 EfficientNet）可能不够优化。

---

## 38. Where Relevance Emerges: A Layer-Wise Study of Internal Attention for Zero-Shot Re-Ranking

**arXiv ID:** 2602.22591 | [PDF](https://arxiv.org/pdf/2602.22591v1)

**作者:** Haodong Chen `[一作]` (University of Queensland), Teerapong Leelanupab `[通讯]` (University of Queensland)

**通讯引用:** 176 | [OpenAlex ID](https://openalex.org/A5049764096)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估并改进了基于大语言模型的零样本文档重排序方法，重点分析了内部注意力机制的层级分布，并提出了基于中心偏差区间选择的Selective-ICR策略。

**💡 创新点**

创新点在于发现Transformer内部注意力的“钟形”信号分布，并利用该分布设计Selective-ICR，仅聚焦中间层信号，既提升了重排序效果，又显著降低了推理延迟；同时首次统一比较生成、似然和内部注意力三种评分机制在列表式与集合式框架下的性能。

**🔧 技术方法**

技术主要包括：大模型内部注意力提取与层级分析、双向校准的ICR、Center-Biased Interval Selection、可调窗口宽度的Selective-ICR、以及对比实验中的生成、似然与内部注意力评分方法。

**📊 数据集**

使用的数据集包括：TREC-DL 2019/2020、BEIR 9个子任务、BRIGHT 1,385条推理型查询，以及BM25检索得到的前100条候选文档。

**📈 对比分析**

在所有评测指标（如nDCG@10）和耗时上，Selective-ICR在大多数模型与任务上达到或超过All-ICR且延迟降低30%–50%；在列表式大上下文场景下内部注意力优于生成/似然方法；但在滑窗和集合式细粒度对比中，内部注意力表现不如似然，生成/似然在效率与效果上更具优势。

**⚠️ 局限性**

主要限制在于内部注意力在迭代、局部上下文设置下效果衰退；峰值层的选择仍需依赖额外的层级评估或领域特定优化，且对极小模型（如0.6B）效果不稳定；未来工作需探索更细粒度的头/层级选择和自适应窗口策略。

---

## 39. Relational Appliances: A Robot in the Refrigerator for Home-Based Health Promotion

**arXiv ID:** 2602.22542 | [PDF](https://arxiv.org/pdf/2602.22542v1)

**作者:** Timothy Bickmore `[一作]` (Northeastern University), Yunus Terzioglu `[通讯]` (Northeastern University)

**通讯引用:** 222 | [OpenAlex ID](https://openalex.org/A5072489319)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在家用实验室中将人形机器人头嵌入冰箱内部，利用短时对话和健康建议，引导用户在每次零食选择时做出更健康的决定，并评估其对行为与情感的影响。

**💡 创新点**

提出“关系式家电”概念，证明即使在极短的、情境化的交互中，具身、社交化的代理也能在家居环境中发挥说服与关系建立作用；首次将智能家电与嵌入式人形机器人相结合，形成了“家电+社交代理”的新模式。

**🔧 技术方法**

使用 Furhat Robotics Gen 2 机器人头（投影面部、3自由度动作），Wizard‑of‑Oz 控制实现对话；配合预设的说服性对话脚本和非语言行为；使用 Nutri‑Score 食物评级、MFT 量表、Godspeed、UTAUT 等评估工具收集行为与态度数据。

**📊 数据集**

构建自制的食物样本集（按 Nutri‑Score A/B/D/E 分组，含 14 件可冷藏零食），并收集实验参与者在模拟电视广告中断时的零食选择、语音交互记录及问卷数据；无公开大规模公开数据集。

**📈 对比分析**

通过 21 名受试者的随机对照实验（干预组 13 人，控制组 8 人），使用 t 检验、Wilcoxon 符号秩检验和卡方检验比较干预与控制。结果显示：干预组对机器人建议的遵从率显著高于控制组（54.9 % vs 21.3 %），健康零食选择率虽更高但未显著；选择时间显著缩短；所有接受度量表均高于中性水平，表明技术可接受且效果良好。

**⚠️ 局限性**

局限包括：实验仅在实验室厨房完成，缺乏真实家庭情境和长期使用；样本量小且以学生为主，外推性有限；采用 Wizard‑of‑Oz 模拟，未验证完全自治系统的性能；干预仅针对单一短期零食选择，未评估对整体饮食习惯或健康结果的长期影响。

---

## 40. Complete Robust Hybrid Systems Reachability

**arXiv ID:** 2602.22853 | [PDF](https://arxiv.org/pdf/2602.22853v1)

**作者:** Noah Abou El Wafa `[一作]` (Karlsruhe Institute of Technology), André Platzer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5580 | [OpenAlex ID](https://openalex.org/A5080481427)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出并证明了稳健差分动态逻辑（Robust Differential Dynamic Logic, RDdL）对一般混合系统可达性问题的绝对完备性；

**💡 创新点**

创新点在于通过简单的语法限制（在正位置使用严格不等式、负位置使用宽松不等式）实现拓扑意义上的稳健性，从而在保持精确语义的同时，获得不依赖不可判定 oracle 的完备性；

**🔧 技术方法**

使用了拓扑学中的开集与闭集概念、紧致性和流函数的连续性，结合差分动态逻辑的证明系统（含实数算术、混合程序公理、欧拉逼近公理等）构建了构造性完备证明；

**📊 数据集**

本文为理论证明，没有使用实验数据集；

**📈 对比分析**

由于该工作是理论性质，未进行实验比较；但完备性证明本身提供了构造性的证明搜索算法，表明对稳健可达性命题的半可判定性；

**⚠️ 局限性**

局限在于仅覆盖稳健可达性，未覆盖安全（box）问题；并且对混合系统的表达仍需满足RDdL的语法限制，虽然实用性高但在某些高度精确约束的控制器设计中可能受限。

---

## 41. Frequency-Ordered Tokenization for Better Text Compression

**arXiv ID:** 2602.22958 | [PDF](https://arxiv.org/pdf/2602.22958v1)

**作者:** Maximilian Kalcher `[一作]` `[通讯]` (ETH Zurich), Maximilian Kalcher (ETH Zurich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于Zipf定律的频率排序BPE预处理，提升LZ系列压缩器的无损压缩率。

**💡 创新点**

创新点在于将Token频率分布映射到小整数ID并采用可变长度整数编码，充分利用Zipf分布优势。

**🔧 技术方法**

使用BPE分词、频率排序映射、LEB128可变长整数编码以及zlib、zstd、LZMA等标准压缩器。

**📊 数据集**

主要实验数据集为enwik8、enwik9以及中阿两种语言的Wikipedia文本。

**📈 对比分析**

与原始文本、BPE单独处理及Word Replacing Transform比较，zlib压缩率提升7.08pp，LZMA 1.69pp，zstd 0.76pp，并且对昂贵压缩器可实现3.1×/2.4×的速度提升。

**⚠️ 局限性**

局限性包括需预先统计频率产生词表开销，解压速度较慢，仅适用于Zipfian自然语言文本，对二进制或加密数据无效。

---

## 42. BRIDGE: Borderless Reconfiguration for Inclusive and Diverse Gameplay Experience via Embodiment Transformation

**arXiv ID:** 2602.23288 | [PDF](https://arxiv.org/pdf/2602.23288v1)

**作者:** Hayato Saiki `[一作]` (University of Tsukuba), Kenji Suzuki `[通讯]` (University of Tsukuba)

**通讯引用:** 18212 | [OpenAlex ID](https://openalex.org/A5050949810)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出并实现了BRIDGE系统，将站立篮球录像转换为轮椅篮球可视化，从而为轮椅运动员提供适配的战术学习资源。

**💡 创新点**

创新点在于结合三层方向映射（轮椅基座、躯干、头部）与功能分类约束，实现了从非残疾运动员视角到残疾运动员可执行动作的跨身体表征。

**🔧 技术方法**

技术主要包括基于YOLOX+MixSort的多目标检测与跟踪、SAM2的球体跟踪、SMPL三维姿态重建、Unity三维重建、以及基于分类约束的姿态映射与运动平滑。

**📊 数据集**

使用的数据集为公开的NBA比赛录像（YouTube）作为输入，并通过与日本国家轮椅篮球队与业余队员的实测对照进行验证，未使用特定标注数据集。

**📈 对比分析**

实验采用两组对照（原始视频 vs 转换后视频），评估了自我效能、战术理解准确率、自然度评分等指标，结果显示转换后视频在自我效能、自然度与战术保留上均显著优于原始视频。

**⚠️ 局限性**

局限性包括样本量有限、缺乏长期学习和实战评估、仅针对轮椅篮球且未集成触觉等多模态反馈，且转换过程对不同功能等级的区分仍不够细致。

---

## 43. Scaling Laws of Global Weather Models

**arXiv ID:** 2602.22962 | [PDF](https://arxiv.org/pdf/2602.22962v1)

**作者:** Yuejiang Yu `[一作]` (Computer Science), Torsten Hoefler `[通讯]` (Computer Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对全球天气预测模型（GraphCast、AIFS、Aurora、Pangu、SFNO）在统一实验条件下进行跨模型的规模律分析，研究模型大小（N）、训练数据量（D）和计算预算（C）对验证损失的影响。

**💡 创新点**

创新点包括：首次系统比较天气模型的规模律；发现模型宽度（width）对性能影响大于深度（depth）；Aurora 在数据扩展方面表现最佳；计算最优训练更倾向于延长训练时间而非增大模型；揭示硬件利用率差异对实际部署的关键影响。

**🔧 技术方法**

使用了大规模深度学习实验技术，采用 Swin Transformer、Graph Transformer、Graph Neural Network、Spherical Fourier Operator 等架构；统一了验证损失度量；通过 log‑log 回归拟合尺度指数 α、β、γ、δ、λ、ε；进行 compute‑optimal 训练分析；对模型形状（宽度/深度）进行系统调参。

**📊 数据集**

使用 ERA5 重分析数据（通过 WeatherBench 2），训练集 1979–2020，验证集 2021，统一使用 0.25°×0.25° 的全局网格和 6 小时预测间隔。

**📈 对比分析**

在相同 GPU、学习率、批次、输入变量等条件下，对不同 N、D、C、宽度/深度组合进行训练，评估验证损失和 RMSE。结果显示：Aurora 的 β≈0.51，数据扩展最有效；GraphCast 在参数规模上更高效但硬件利用率低；宽模型比深模型表现更好；compute‑optimal 曲线表明在固定预算下更长训练比增大参数更能降低误差。

**⚠️ 局限性**

局限性：受 GPU 内存和硬件利用率限制，无法探索更大宽度或更深网络；compute‑optimal 曲线受硬件瓶颈影响；变量特定的尺度律差异需要进一步研究；实验仅覆盖 6 小时预测，未评估更长时序或更复杂的物理变量。

---

## 44. Multilingual Safety Alignment Via Sparse Weight Editing

**arXiv ID:** 2602.22554 | [PDF](https://arxiv.org/pdf/2602.22554v1)

**作者:** Jiaming Liang `[一作]` (Xidian University), Handing Wang `[通讯]` (Xidian University)

**通讯引用:** 6155 | [OpenAlex ID](https://openalex.org/A5091324372)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无训练、基于稀疏权重编辑的多语言安全对齐框架，利用少量安全与无害样本快速生成权重修正矩阵，实现低资源语言的安全提升。

**💡 创新点**

创新点在于将跨语言安全转移视为在安全神经元子空间上的低秩线性映射，并给出了闭式解析解，避免了梯度优化和大规模安全数据需求。

**🔧 技术方法**

核心技术包括稀疏安全神经元识别、低秩权重编辑、零空间正则化以及一次性闭式求解（Cholesky + 截断SVD）。

**📊 数据集**

使用了多语言安全基准 Multi-StrongREJECT（8种语言）以及多模型（Llama‑3.2、Qwen‑2 系列 1B‑7B）进行实验评估。

**📈 对比分析**

与无对齐、MPO 以及 MPO+本方法的对比显示，该方法在所有模型和语言上显著降低攻击成功率（ASR）且对 MGSM/M‑MMLU 的影响可忽略，且兼容 MPO 可进一步提升安全性能。

**⚠️ 局限性**

局限性包括依赖预先识别的安全神经元集合，若神经元定位不准确会影响效果；仅对单层/单子空间进行编辑，可能对复杂或适应性攻击存在一定脆弱性；以及对超参数（rank、正则化权重）的选择仍需人工调优。

---

## 45. HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models

**arXiv ID:** 2602.22727 | [PDF](https://arxiv.org/pdf/2602.22727v1)

**作者:** Yangguang Lin `[一作]` (Beijing University of Posts and Telecommunications), Jitao Sang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 2145 | [OpenAlex ID](https://openalex.org/A5023834030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在大规模视觉‑语言模型（LVLM）中，提出了单通道、无参考模型的子空间编辑框架 HulluEdit，用于抑制对象幻觉并保持视觉根基。

**💡 创新点**

创新点在于将隐藏状态拆分为三个正交子空间——视觉证据子空间、反先验子空间和残差不确定子空间；通过在线加权 SVD 构建可自适应的视觉子空间，并保证编辑仅作用于反先验子空间，从而理论上保证视觉成分不受干扰。

**🔧 技术方法**

采用的技术包括：在线加权 SVD 与低秩近似；正交子空间构造（视觉子空间与其正交补）；基于证书（Visual Certainty Ratio、Prior Conflict Ratio）的自适应强度调度；闭式最小范数编辑与门控机制；以及在单通道推理时直接对最终 Transformer 层隐藏状态进行编辑。

**📊 数据集**

使用的评测数据集包括：POPE（多种随机/热门/对抗分割的 VQA‑style 对象幻觉评估）、CHAIR（MSCOCO 图像说明的实例级和句子级幻觉评估）、MME（14 个子任务的综合能力评估）和 MSCOCO 的字幕生成任务。

**📈 对比分析**

与多种对照方法（对比解码、静态子空间编辑、动态纠错、VCD、DeCo、OPERA、HALC 等）进行比较，HulluEdit 在 POPE、CHAIR、MME 上均取得最优或接近最优的 Accuracy/F1、CHAIR_i/CHAIR_s、M&M 子任务分数，并保持与基线相当甚至更快的解码吞吐量，证明了其高效且鲁棒的幻觉抑制能力。

**⚠️ 局限性**

局限性主要体现在：对计数类任务的性能略有下降（因残差子空间保守正则化）；仍需根据不同 LVLM 架构手动选择锚层和编辑层；以及在极度模糊或缺少视觉信息的场景下，正交子空间可能无法完全区分视觉与先验，导致编辑过度或不足。

---

## 46. Mapping the Landscape of Artificial Intelligence in Life Cycle Assessment Using Large Language Models

**arXiv ID:** 2602.22500 | [PDF](https://arxiv.org/pdf/2602.22500v1)

**作者:** Anastasija Mensikova `[一作]` (University of Vermont), Kathryn Hinkelman `[通讯]` (University of Vermont)

**通讯引用:** 330 | [OpenAlex ID](https://openalex.org/A5005762300)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对人工智能在生命周期评估中的应用进行了系统性综述，并构建了AI‑LCA文献的知识图谱。

**💡 创新点**

首次将大语言模型与嵌入聚类相结合，提供可扩展且可解释的文献综述框架。

**🔧 技术方法**

使用了Sentence‑BERT embeddings、UMAP降维、HDBSCAN聚类、轻量级LLM（如Llama）进行聚类标注和全文信息抽取。

**📊 数据集**

基于538篇符合条件的文献（209篇可获得全文），包含Scopus检索、Unpaywall、Elsevier API等来源。

**📈 对比分析**

与传统手工或统计文本分析相比，LLM方法在提取结构化信息上更准确、速度快、成本低，提取准确率高于手工标注且可重复。

**⚠️ 局限性**

局限性包括全文获取不足、LLM可能产生幻觉、对特定LCA领域知识有限、能耗评估不完整。

---

## 47. SODA-CitrON: Static Object Data Association by Clustering Multi-Modal Sensor Detections Online

**arXiv ID:** 2602.22243 | [PDF](https://arxiv.org/pdf/2602.22243v1)

**作者:** Jan Nausner `[一作]` (Austrian Institute of Technology), Michael Hubner `[通讯]` (Austrian Institute of Technology)

**通讯引用:** 307 | [OpenAlex ID](https://openalex.org/A5101668290)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在线静态物体数据关联方法SODA‑CitrON，能够从异构、时序不相关的传感器检测中聚类并持续追踪未知数量的静态目标，结合信息滤波与置信度加权实现位置估计。

**💡 创新点**

创新点在于：1）在DBSTREAM聚类框架中加入基于检测置信度的非线性权重转化，显著提升对高置信度检测的快速响应；2）将信息滤波嵌入每个潜在聚类，实现对目标位置和协方差的递归更新；3）利用唯一ID和重量阈值实现持久轨迹与实时聚类的统一；4）整体算法实现了对检测数的worst‑case O(NlogN)时间复杂度。

**🔧 技术方法**

采用的技术包括：DBSTREAM在线密度聚类、信息滤波（inverse Kalman），置信度到权重的非线性映射，邻域查询与R‑tree索引，聚类阈值与交互因子α，阈值w_min控制杂波过滤。

**📊 数据集**

使用的实验数据为基于Monte‑Carlo仿真的合成数据，包含4种目标类型（A、B、C、D）与5种传感器（S1–S5），场景A为150×150 m平面内随机分布的25个目标，场景B为按行排列的105个A/B目标，仿真共生成约2000–2300个检测点。

**📈 对比分析**

与Bayesian过滤（POM）、JPDA、DBSTREAM四种基准方法在F1、RMSE、MOTP、MOTA以及运行时进行对比。SODA‑CitrON在两场景均显著领先，F1最高、RMSE最低、MOTP与MOTA最高，并且平均运行时仅为6.4–8.4 s（≈250 检测/秒），比DBSTREAM快约5倍。

**⚠️ 局限性**

局限性包括：1）假设传感器误差适中且至少存在一次高置信度检测；2）对杂波依赖置信度低、稀疏的特性，若杂波高或置信度误差大效果可能下降；3）仅针对静态目标，尚未扩展到移动目标；4）实验仅在仿真环境，缺乏真实世界数据验证；5）参数（β、w_min、r、α）需手动调优，未实现自适应优化。

---

## 48. A Framework for Assessing AI Agent Decisions and Outcomes in AutoML Pipelines

**arXiv ID:** 2602.22442 | [PDF](https://arxiv.org/pdf/2602.22442v1)

**作者:** Gaoyuan Du `[一作]` (Amazon), Jing Wu `[通讯]` (Amazon)

**通讯引用:** 113164 | [OpenAlex ID](https://openalex.org/A5100355638)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个评估代理(Evaluation Agent)框架，用于在Agent化AutoML管道中对中间决策、推理、模型质量和逆因果影响进行可解释、结构化的评估。

**💡 创新点**

创新点在于将评估从仅关注最终任务指标转为阶段级决策级评估，设计了四个模块（决策评估、推理验证、模型质量评估、逆因果分析），并首次在实验中展示决策级评估能揭示隐藏错误。

**🔧 技术方法**

采用LLM‑as‑judge与规则混合的方法对决策和推理进行评分；使用基于规则的风险检测、对抗噪声与缺失的鲁棒性测试；对逆因果影响进行部分重执行或代理估计；并集成多维度指标（准确率、鲁棒性、公平性、校准、效率）。

**📊 数据集**

在多领域数据集上验证：German Credit、Adult Income、Titanic、Diabetes、California Housing等公开tabular数据集，覆盖分类、回归及公平性敏感任务。

**📈 对比分析**

与传统Outcome‑centric AutoML相比，EA在决策错误检测上达到F1=0.919，推理验证准确率75%，模型质量评估揭示最高8.3%的性能波动；实验表明EA能捕捉到终端指标无法发现的失效模式。

**⚠️ 局限性**

局限在于实验仅为原型验证，使用的LLM与规则仍可能遗漏复杂推理错误；逆因果分析仅限于少量关键决策；未与AutoML系统闭环集成，仅做离线诊断；数据集规模有限，未覆盖多模态任务。

---

## 49. Three AI-agents walk into a bar . . . . `Lord of the Flies' tribalism emerges among smart AI-Agents

**arXiv ID:** 2602.23093 | [PDF](https://arxiv.org/pdf/2602.23093v1)

**作者:** Dhwanil M. Mori `[一作]` (George Washington University), Neil F. Johnson `[通讯]` (George Washington University)

**通讯引用:** 12155 | [OpenAlex ID](https://openalex.org/A5031168379)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在有限资源共享的多 AI 代理环境中，评估 LLM 代理在不通信、短期历史窗口下的请求决策是否能提升系统安全与效率；

**💡 创新点**

发现更强的 LLM 通过共同推理往往导致同步请求，形成集体失效，反而比简单随机策略更危险；

**🔧 技术方法**

使用 El Farol / 少数游戏框架、基于概率的随机化（p=C/N）、ε-greedy 与温度采样、k-means 聚类分析等技术；

**📊 数据集**

实验数据来自 43 次配置（N∈{3,4}, C∈{1,2}）共 154 个 LLM 代理实例，涵盖 Gemini、GPT、Claude 的小/大模型以及多种性格提示；

**📈 对比分析**

与容量匹配随机基线（p=C/N）比较，LLM 代理的超载率分别为大模型 72.5%、小模型 53.8%，均显著高于基线 31.25%，显示性能较差；

**⚠️ 局限性**

局限在样本规模小（N≤4）、短期实验、固定容量、无通信、单一提示结构，且模型规模与其他变量未完全分离，需更严谨的因子设计与更大规模实验验证。

---

## 50. MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding

**arXiv ID:** 2602.22932 | [PDF](https://arxiv.org/pdf/2602.22932v1)

**作者:** Wenhui Tan `[一作]` (Renmin University of China), Jian Luan `[通讯]` (Xiaomi Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个联合演化框架MSJoE，用于在长视频问答中通过多查询与可学习的关键帧采样器高效选择信息帧，提升理解精度；

**💡 创新点**

创新点在于（1）通过MLLM生成多维查询来弥补单一问题的视觉信息不足；（2）采用轻量级1D U‑Net采样器将查询-帧相似度矩阵转化为多样化关键帧权重；（3）利用强化学习实现MLLM与采样器的端到端协同演化；

**🔧 技术方法**

使用CLIP做查询-帧相似度计算、1D U‑Net做采样、Qwen2.5‑VL‑7B作为基线MLLM、GRPO与REINFORCE强化学习算法；

**📊 数据集**

新构建的长视频QA数据集（2.8k视频、7.1k问答对）以及公开的VideoMME、LongVideoBench、LVBench、MLVU四大基准；

**📈 对比分析**

与多种基线对比（包括均匀采样、Q‑Frame、BOLT、TSPO等），MSJoE在64帧预算下分别在LongVideoBench、MLVU、VideoMME和LVBench获得+9.8、+4.9、+5.2、+11.9的准确率提升，平均提升约+8点，相较于基线最高提升1.1点；

**⚠️ 局限性**

局限性包括：仍依赖CLIP相似度对齐；采样器训练对初始权重敏感；在极其复杂或多跳推理场景下可能需要更强的查询生成能力。

---

## 51. SC-Arena: A Natural Language Benchmark for Single-Cell Reasoning with Knowledge-Augmented Evaluation

**arXiv ID:** 2602.23199 | [PDF](https://arxiv.org/pdf/2602.23199v1)

**作者:** Jiahao Zhao `[一作]` (Northeastern University), Min Yang `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 SC‑ARENA 基于虚拟细胞抽象的自然语言评估框架，用于多任务推理能力的评测。

**💡 创新点**

创新点在于将细胞属性与行为统一为虚拟细胞，设计 5 个开放式自然语言任务，并引入知识增强评估器实现可解释且生物学可信的打分。

**🔧 技术方法**

使用大语言模型（如 Qwen、GPT‑4o、DeepSeek‑R1 等）作为评估者与被评估模型，并结合 Cell Ontology、CellMarker、GO、UniProt 等外部数据库进行知识增强评估。

**📊 数据集**

数据集来源于 CELLxGENE（单细胞转录组）、Norman & Adamson 的扰动实验以及 PubMed 文献，覆盖细胞类型标注、描述、生成、扰动预测与科学问答 5 个任务。

**📈 对比分析**

与现有单细胞 LLM Benchmark（C2S‑Scale、SOAR、CELLVERSE 等）对比，SC‑ARENA 能区分模型在细胞属性推理与因果推断上的差异；实验显示一般模型最高总分约 277，仍低于理想阈值。

**⚠️ 局限性**

局限在于仍难以捕捉机制推理与动态时间推断，评估器依赖预先收集的静态知识库缺乏实时更新；此外，模型规模提升并非总能改善因果推理性能。

---

## 52. static_maps: consteval std::map and std::unordered_map Implementations in C++23

**arXiv ID:** 2602.22506 | [PDF](https://arxiv.org/pdf/2602.22506v1)

**作者:** Isaac D. Myhal `[一作]` (Hillsdale), Oliver Serang `[通讯]` (Hillsdale)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

实现了基于 C++23 consteval 的 constexpr 静态 map 与 unordered_map，实现了编译时完美哈希。

**💡 创新点**

将 consteval 与自定义可变长数组 RaggedArray 结合，提供了 drop‑in 替代 std::unordered_map 的最快静态哈希实现。

**🔧 技术方法**

使用 consteval、constexpr 排序、完美哈希、64/32 位自定义模运算、RaggedArray、随机数生成器以及可调节的 δ 与 τ 等技术。

**📊 数据集**

分别在元素质量、密码子转氨基酸、S&P 500 股票价格三组常见静态键集上测试。

**📈 对比分析**

通过与 std::map、std::unordered_map、Frozen、PTHash 和 gperf 在同一硬件上对比，unordered_map 的查询时间仅为 std::unordered_map 的 34–73%，并且在大部分测试中优于 Frozen、PTHash 与 gperf。

**⚠️ 局限性**

编译时间显著增加，且仅支持静态键集；对非法键的访问可能导致未定义行为，需要手动调参且对极大键集需提高 constexpr 限制。

---

## 53. STELLAR: Storage Tuning Engine Leveraging LLM Autonomous Reasoning for High Performance Parallel File Systems

**arXiv ID:** 2602.23220 | [PDF](https://arxiv.org/pdf/2602.23220v1)

**作者:** Chris Egersdoerfer `[一作]` (University of Delaware), Dong Dai `[通讯]` (University of Delaware)

**通讯引用:** 696 | [OpenAlex ID](https://openalex.org/A5012002926)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的自治存储调优引擎 STELLAR，能够在少量（≤5次）尝试内为并行文件系统（如 Lustre）自动选择接近最优的 I/O 参数配置，近似甚至超过人类专家水平。

**💡 创新点**

创新点包括：① 用检索增强生成（RAG）精确提取系统手册中的可调参数及其描述，避免 LLM 生成 hallucination；② 结合外部工具调用和多代理架构，让分析代理和调优代理协同完成 I/O 日志解析、策略生成与迭代；③ 通过经验回溯生成可迁移的规则集（Rule Set），实现知识累积与迁移；④ 在单个工作负载中实现了与人类专家几乎同等或更优的性能。

**🔧 技术方法**

使用技术包括：LLM（Claude‑3.7‑Sonnet、GPT‑4o、Llama‑3.1‑70B）与工具调用、检索增强生成（RAG）、多代理（Analysis Agent + Tuning Agent）、OpenInterpreter、LangChain 等；在后端对 Lustre 参数进行向量索引、基于 Pandas 的 I/O 日志分析；使用自定义的配置运行工具和结束决策工具。

**📊 数据集**

数据集主要是常用 HPC I/O 基准（IOR、MDWorkbench、IO500）以及两类真实应用（AMReX I/O 核心、MACSio），在 CloudLab 10 节点 Lustre 2.15.5 集群上运行，收集 Darshan 日志。

**📈 对比分析**

与基线比较：STELLAR 在不使用规则集的情况下，平均在 5 次尝试内达到 2.5‑7.8 倍速度提升，超过默认设置并与人工专家相当或更优；使用累计规则集后，首轮猜测性能进一步提升，迭代次数减少 20‑40%；在不同 LLM 模型下表现相近。

**⚠️ 局限性**

限制包括：需系统管理员 root 权限进行参数修改，实验规模有限（10 节点 CloudLab），仅针对 Lustre 2.15.5，未验证在大规模多节点或异构环境中的效果；LLM 调优成本随模型价格和 API 费用波动；当前仅覆盖系统层级参数，用户层面参数尚未集成。

---

## 54. PackUV: Packed Gaussian UV Maps for 4D Volumetric Video

**arXiv ID:** 2602.23040 | [PDF](https://arxiv.org/pdf/2602.23040v1)

**作者:** Aashish Rai `[一作]` (Brown University), Srinath Sridhar `[通讯]` (UMass Amherst)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PackUV：一种将 3D 高斯属性压缩为多层 UV atlas 的 4D 高斯表示，并设计了直接在 UV 域中拟合、光流引导的关键帧与高斯标签的全流程，以实现长时序、无显著时间漂移的体积视频重建与流式播放。

**💡 创新点**

创新点：① PackUV 通过 UV atlas 结构实现对 4D 高斯的紧凑、结构化存储；② 直接 UV 域优化、光流关键帧 + 高斯标签的拟合策略，使得在大运动和遮挡/重映照场景下保持长时间一致性；③ 低精度 8 位优化（LPO）兼容标准视频编解码器；④ 创建最大规模的多视角 4D 数据集 PackUV-2B，为长时序体积视频研究提供基准。

**🔧 技术方法**

技术手段：3D 高斯 splatting、UV 投影与多层图像布局、光流 RAFT + 运动掩码、关键帧分段训练、Gaussian 标记与梯度冻结、UV 层级裁剪、低精度优化（LPO）、FFV1/HEVC 等传统视频编码器。

**📊 数据集**

使用数据集：新收集的 PackUV-2B（50+摄像机、>10 分钟、>10 亿帧），以及公开数据集 N3DV、SelfCap、D-NeRF、CMU Panoptic 等进行对比实验。

**📈 对比分析**

与 3DGStream、4DGS、RealTime4DGS、Deformable3DGS、ATGS、Grid4D、Ex4DGS、GIFStream 等基线在 PSNR、SSIM、LPIPS 上均表现更优，训练时间更短；且 PackUV 完全兼容 HEVC/FFV1，支持无损流式传输，展示了在长时序、强运动和遮挡下的可扩展性和实用性。

**⚠️ 局限性**

局限性：仍需大规模 GPU 计算进行离线拟合；对极端光照变化或非常高分辨率场景的鲁棒性未完全验证；长时间序列中可能出现细微漂移；对 UV atlas 大尺寸的存储与压缩仍有进一步优化空间。

---

## 55. InCoM: Intent-Driven Perception and Structured Coordination for Whole-Body Mobile Manipulation

**arXiv ID:** 2602.23024 | [PDF](https://arxiv.org/pdf/2602.23024v1)

**作者:** Jiahao Liu `[一作]` (Institute of Automation Chinese Academy of Sciences), Dongbin Zhao `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为 InCoM 的端到端框架，用于全身移动操纵，结合意图驱动的多尺度感知与结构化的基座-机械臂协同控制。

**💡 创新点**

创新点包括：①意图驱动金字塔感知模块 (IDPPM)，动态重加权多尺度特征以适配任务阶段；②双流亲和力细化模块 (DARM)，分别建模几何与语义亲和力并进行轻量融合；③解耦协同流匹配解码器 (DCFM)，在流匹配框架下实现基座与机械臂的双向协调。

**🔧 技术方法**

技术手段主要有端到端模仿学习、Transformer 结构、流匹配 (flow matching) 代替扩散模型、跨模态几何语义亲和力融合、历史动作推理的意图模块、以及多尺度特征金字塔。

**📊 数据集**

使用 ManiSkill-HAB 仿真平台的三个场景：SetTable、TidyHouse、PrepareGroceries，提供 RGB、点云及关节状态等观测。

**📈 对比分析**

与 DP、ACT、DSPv2、WB-VIMA、AC-DiT 等基线在不使用特权观测的设置下对比，InCoM 在三场景成功率分别提升 28.2%、26.1%、23.6%，整体平均成功率达 83.8%，显著优于现有方法。

**⚠️ 局限性**

局限性：模型在任务特定数据上训练，泛化到新场景或长时域任务的能力有限；意图推理仍较简化，难以捕捉快速变化的任务动态；缺乏大规模预训练或更强推理模块，导致在极端环境下表现可能受限。

---

## 56. Efficient Dialect-Aware Modeling and Conditioning for Low-Resource Taiwanese Hakka Speech Processing

**arXiv ID:** 2602.22522 | [PDF](https://arxiv.org/pdf/2602.22522v1)

**作者:** An-Ci Peng `[一作]` (National Taiwan Normal University), Berlin Chen `[通讯]` (National Taiwan Normal University)

**通讯引用:** 2331 | [OpenAlex ID](https://openalex.org/A5115595070)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文针对台客低资源、多方言、双书写体系的挑战，提出了一种统一的多任务 RNN‑T 框架，实现汉字与拼音同时识别并融入方言信息；

**💡 创新点**

创新点包括：①在编码器端采用方言辅助分类器（ADC）和方言信息集成（DII）两种方言感知策略；②在解码器端引入 Token‑Interleaved Conditioning（TIC）连续方言上下文；③将汉字与拼音任务联合训练，互补正则化编码器；

**🔧 技术方法**

技术上使用 Zipformer 编码器、RNN‑T‑SLP 预测网络、联合网络，并结合 ADC、DII、TIC 进行方言融入；

**📊 数据集**

实验数据基于 Hakka Across Taiwan (HAT) 语料库，包含三大方言（六县、南四县、海陆）共 82.32 小时；

**📈 对比分析**

与单任务基线相比，联合多任务+方言融入模型在汉字 CER 与拼音 SER 上分别提升 57% 与 40% ；TIC 方案单独可将汉字 CER 降至 2.71%，拼音 SER 降至 4.98%；

**⚠️ 局限性**

局限性包括：仅覆盖三种方言，未使用更大规模预训练模型或自监督表示；在推理时方言标签需来自 ADC 或外部预测，可能受误判影响。

---

## 57. Revisiting Chebyshev Polynomial and Anisotropic RBF Models for Tabular Regression

**arXiv ID:** 2602.22422 | [PDF](https://arxiv.org/pdf/2602.22422v1)

**作者:** Luciano Gerber `[一作]` (Manchester Metropolitan University), Huw Lloyd `[通讯]` (Manchester Metropolitan University)

**通讯引用:** 1991 | [OpenAlex ID](https://openalex.org/A5069957801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文评估了两类平滑基函数模型（Chebyshev多项式回归和各向异性RBF网络）及其混合模型（Chebyshev模型树），与树集成、预训练Transformer和基线模型在55个回归数据集上的性能。

**💡 创新点**

创新点在于提出三阶段可训练的各向异性RBF网络、岭回归正则化的Chebyshev回归器，以及将树分区与平滑多项式相结合的模型树，并系统比较其在准确率、泛化误差与计算成本上的表现。

**🔧 技术方法**

使用的技术包括：Chebyshev多项式特征构造与岭回归、各向异性RBF激活函数的中心放置与宽度梯度优化、决策树分区与局部多项式回归、Optuna超参搜索、nested cross-validation、scikit‑learn接口实现。

**📊 数据集**

采用了55个公开回归数据集，涵盖工程/模拟、行为/社会、物理/化学/生命科学、经济/定价四大领域，数据规模从几百到数十万样本、特征维度从2到1024。

**📈 对比分析**

通过对所有模型进行统一的nested CV、同等超参搜索，按调整后的R²和训练-测试R²差距（泛化间隙）评估；结果显示预训练Transformer在多数数据集上准确率最高，但CPU可用模型（XGBoost、随机森林、erbf、chebypoly、chebytree）在准确率上统计无显著差异，而平滑模型在大多数匹配准确率的比较中实现更小的泛化间隙。

**⚠️ 局限性**

局限性包括仅评估回归任务、统一的预处理与特征选择可能对不同模型产生不对称影响、样本规模与维度上限（50k样本、50特征）限制了对大规模数据的完整评估，以及未对分类任务或更复杂的特征变换进行验证。

---

## 58. Causality $\neq$ Invariance: Function and Concept Vectors in LLMs

**arXiv ID:** 2602.22424 | [PDF](https://arxiv.org/pdf/2602.22424v1)

**作者:** Gustaw Opiełka `[一作]` (University of Amsterdam), Claire E. Stevenson `[通讯]` (University of Amsterdam)

**通讯引用:** 1050 | [OpenAlex ID](https://openalex.org/A5021896449)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型是否以抽象形式编码概念，比较不同输入格式下的函数向量与概念向量的表现。

**💡 创新点**

首次将因果性与抽象性分离：通过代表性相似性分析（RSA）识别独立的概念向量，证明它们与传统的因果函数向量在功能和机制上相互独立。

**🔧 技术方法**

使用激活补丁（Activation Patching）定位因果头，采用RSA挑选抽象头，构造向量并进行对抗式调优与推理实验。

**📊 数据集**

构建7种关系概念（如反义词、因果关系等）在三种输入格式（英文开放式、法文开放式、英文多选）下的提示集，共计21个数据集，使用Llama 3.1（8B/70B）和Qwen 2.5（7B/72B）四个模型。

**📈 对比分析**

通过RSA与AIE头重叠度、相似度矩阵和KL散度评估调优效果，发现概念向量在跨格式与跨语言的泛化更稳健，而函数向量在分布内效果更强，但在分布外表现不稳定。

**⚠️ 局限性**

局限在于仅使用全局RSA，可能遗漏概念特定的头；未探究头的训练过程；零样本调优效果不佳；实验仅涵盖七种概念与有限格式。

---

## 59. PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering

**arXiv ID:** 2602.23161 | [PDF](https://arxiv.org/pdf/2602.23161v1)

**作者:** Junkai Lu `[一作]` (East China Normal University), Bin Yang `[通讯]` (East China Normal University)

**通讯引用:** 49214 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出PATRA框架，解决时间序列问答中模式感知与跨任务平衡训练的不足；

**💡 创新点**

创新点包括：①模式感知对齐机制，将趋势、季节性等模式与文本语义深度对齐；②基于任务的平衡奖励的强化学习训练，统一不同难度任务的奖励分布；

**🔧 技术方法**

技术方法包括多模态编码器、趋势季节性分解、可学习对齐标记、Self‑Attention对齐、GRPO强化学习优化；

**📊 数据集**

使用TSQA数据集（约20万样本，12+领域），并在MTBench进行零样本OOD评估；

**📈 对比分析**

与多种开源与商用模型（DeepSeek‑R1、Llama3、Qwen2.5、ChatTS、GPT‑4o等）对比，PATRA在四类任务（理解、识别、推理、预知）上均取得最高准确率或Rouge‑L，尤其在推理任务提升约21%；

**⚠️ 局限性**

局限性包括：①模型仍需大量算力训练（4块A800 GPU）；②对极端噪声或异常模式的鲁棒性未充分验证；③奖励设计与GRPO优化的超参数敏感度尚需进一步研究。

---

## 60. Physics-informed neural particle flow for the Bayesian update step

**arXiv ID:** 2602.23089 | [PDF](https://arxiv.org/pdf/2602.23089v1)

**作者:** Domonkos Csuzdi `[一作]` (Budapest University of Technology and Economics), Olivér Törő `[通讯]` (Budapest University of Technology and Economics)

**通讯引用:** 125 | [OpenAlex ID](https://openalex.org/A5024105150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于物理约束的神经粒子流（PINPF）框架，用于在贝叶斯更新中学习确定性传输映射。

**💡 创新点**

创新点在于将对数同胚粒子流的主PDE嵌入神经网络训练目标，实现无监督学习、减轻数值刚性并实现摊销推断。

**🔧 技术方法**

使用了物理信息神经网络、连续性方程主PDE、MLP速度场、自动微分求解散度、Euler自适应积分及梯度残差损失。

**📊 数据集**

实验数据集包括合成多模高维高斯混合分布和非线性TDOA定位场景，训练任务通过参数采样生成。

**📈 对比分析**

与MCMC、SVGD、正则化流以及解析Daum–Huang流等基线比较，PINPF在误差指标（ED、SWD）上保持竞争或更优的性能，同时显著降低计算时间并提升模式覆盖。

**⚠️ 局限性**

局限性在于粒子之间独立处理，未利用集合级特征；对极端先验或复杂真实场景的泛化仍需进一步验证。

---

## 61. SPM-Bench: Benchmarking Large Language Models for Scanning Probe Microscopy

**arXiv ID:** 2602.22971 | [PDF](https://arxiv.org/pdf/2602.22971v1)

**作者:** Peiyao Xiao `[一作]` (Alibaba Group), Hu Wei `[通讯]` (Alibaba Group)

**通讯引用:** 731 | [OpenAlex ID](https://openalex.org/A5101855580)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了用于扫描探针显微镜（SPM）的 PhD 级多模态基准数据集 SPM‑Bench，并引入了严格的不完善惩罚 F1（SIP‑F1）评估指标；同时通过多模态大模型（MLLM）对问答进行自动化生成与评估。

**💡 创新点**

① 自动化、近零人工干预的数据合成流程，利用 Anchor‑Gated Sieve (AGS) 与 VLM‑指导裁剪显著降低噪声与计算成本；② SIP‑F1 通过两层门控和惩罚系数，严厉处罚“选全”或“猜测”行为，实现对模型推理“人格”的可量化诊断；③ 将模型自我信心、任务难度和 token 效率与性能结合，形成多维度人格画像。

**🔧 技术方法**

Anchor‑Gated Sieve、llbox 语义定位、云‑本地两级高分辨率裁剪、自回归提示生成、CoT 与 rubric 生成、对抗优化（Advisory Model）审计、基于 λ=0.6、Γ=6 的 SIP‑F1 公式、并结合标准 Exact Match 与 Standard Partial Credit 进行对比。

**📊 数据集**

来自 30+ 物理科学顶级期刊与 2023‑2025 年 arXiv 预印本的原始图像与讨论文本，共 2703 题、2703 张图像（TEM 955、STM 752、SEM 558、AFM 438）；使用高质量的专家级注释和图文对齐。

**📈 对比分析**

与多款 LLM（Gemini‑3‑flash、Qwen‑3.5‑plus、GPT‑5 系列等）在 EM、SPC 与 SIP‑F1 三种评测下对比；SIP‑F1 在分布上形成两峰，显著区分“精确”与“投机”模型；Qwen‑3.5‑plus 在 EM 与 SIP‑F1 上分别达到 0.832 与 0.881，显示出“智慧”型人格；其他模型表现出“激进”或“赌博”型人格。

**⚠️ 局限性**

① 数据合成虽然自动化，但仍依赖 VLM 的裁剪准确度；② SIP‑F1 需要先验 λ 与 Γ 参数，可能不适用于所有任务；③ 基准仅涵盖已公开的期刊与 arXiv 文献，难以覆盖更深层次的实验细节；④ 对模型生成文本的多模态推理依赖于复杂的 CoT 生成，易受提示工程的影响。

---

## 62. An AI-Based Structured Semantic Control Model for Stable and Coherent Dynamic Interactive Content Generation

**arXiv ID:** 2602.22762 | [PDF](https://arxiv.org/pdf/2602.22762v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 63. Epistemic Filtering and Collective Hallucination: A Jury Theorem for Confidence-Calibrated Agents

**arXiv ID:** 2602.22413 | [PDF](https://arxiv.org/pdf/2602.22413v1)

**作者:** Jonas Karge `[一作]` `[通讯]` (Technical University of Dresden), Jonas Karge (Technical University of Dresden)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了一种基于置信阈值的校准式弃权投票框架，利用Beta分布估计代理自身可靠度并在最终投票阶段仅让高置信度代理投票，进而提升集体决策的正确率。

**💡 创新点**

创新点在于：①将经典Condorcet Jury Theorem推广到异构代理、顺序学习与选择性弃权的情境；②通过Martingale与Azuma‑Hoeffding不等式给出非渐近下界；③将自我置信度校准与投票权重结合，为LLM群体决策与Hallucination控制提供理论基础。

**🔧 技术方法**

主要技术包括：Beta分布的贝叶斯更新、置信度阈值门控、随机过程的Martingale和Doob拆分、Azuma‑Hoeffding尾部概率估计，以及Monte Carlo仿真验证。

**📊 数据集**

未使用公开真实数据集，全部实验基于模拟的Bernoulli任务序列，且通过多种参数设定（同质、异质、对立先验、无弃权）进行对比。

**📈 对比分析**

通过与不弃权基线和同质代理对照，实验显示：在合理的置信门限下，弃权策略显著提升了最终多数投票成功率，且理论下界始终低于模拟结果，验证了模型的保守性与有效性。

**⚠️ 局限性**

局限性包括：①假设代理的私有信号完全独立；②使用Azuma‑Hoeffding得到的下界相对保守，缺乏更紧的分布信息；③未考虑代理之间的协同或信息共享；④对实际LLM校准方法的实现与可解释性仍需进一步研究。

---

## 64. On Sample-Efficient Generalized Planning via Learned Transition Models

**arXiv ID:** 2602.23148 | [PDF](https://arxiv.org/pdf/2602.23148v1)

**作者:** Nitin Gupta `[一作]` (University of South Carolina), Biplav Srivastava `[通讯]` (University of South Carolina)

**通讯引用:** 3169 | [OpenAlex ID](https://openalex.org/A5051577973)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于显式转移模型的通用规划框架，预测后继状态而非直接预测动作序列，并通过神经符号解码得到可执行计划。

**💡 创新点**

将通用规划重新定义为转移模型学习问题，引入大小不变的 Weisfeiler‑Leman 图嵌入和残差预测，显著提升在超出训练规模的实例上的泛化能力。

**🔧 技术方法**

采用 WL 图嵌入、LSTM 与 XGBoost 的残差转移模型，结合符号检索的神经符号计划解码，以及基于 STRIPS 的局部成功者匹配。

**📊 数据集**

在四个 IPC 基准（Blocksworld、Gripper、Logistics、VisitAll）上实验，使用 Fast Downward 生成的完整轨迹，并按实例大小划分为训练、验证、插值与外推四组。

**📈 对比分析**

与 PlanGPT、Plansformer、Symmetry‑Aware Transformers 以及 Fast Downward 对比；在 Blocksworld 与 VisitAll 的外推任务中，WL‑XGB 取得 0.45–0.87 的成功率，优于 Transformer 基线，且仅使用约 1M 参数与原始小规模数据集。

**⚠️ 局限性**

对具有层级和长距离因果耦合的 Logistics 等域仍无法在严格外推下获得成功，单步状态预测受限，且在多步或抽象转移任务中表现欠佳。

---

## 65. FlexMS is a flexible framework for benchmarking deep learning-based mass spectrum prediction tools in metabolomics

**arXiv ID:** 2602.22822 | [PDF](https://arxiv.org/pdf/2602.22822v1)

**作者:** Yunhua Zhong `[一作]` (Hong Kong University of Science and Technology), Jun Xia `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 8713 | [OpenAlex ID](https://openalex.org/A5068570479)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了FlexMS，一个灵活的基准框架，用于系统评估深度学习模型在质谱预测任务中的表现；

**💡 创新点**

创新点在于提供统一的数据预处理、模型组合构造、统计检验和检索评估，系统化对比多种分子表征（GFv2、GIN、GAT等）与预测器（MassFormer、NEIMS、MolMS）的性能，并深入分析学习率、数据量、预训练、分辨率、跨域迁移等因素的影响；

**🔧 技术方法**

采用分子嵌入器（Graphormer、GIN、GAT、Transformer、CNN、FP-MLP等）与谱预测器（MassFormer、NEIMS、MolMS）的多模组合，使用二值化、log1p归一化等前处理，评估Cosine、Jensen‑Shannon、Spectral Coverage等指标；

**📊 数据集**

使用了多样化公开数据集：MassSpecGym、GNPS、MassBank、NPLIB1、CASMI16/22，并针对不同分割（随机、结构化）进行分析；

**📈 对比分析**

通过两样本Kolmogorov‑Smirnov、KS统计、Murcko重叠、Critical Difference Diagram等方法比较模型；实验表明GFv2+MassFormer在大多数数据集上表现最佳，预训练提升显著，低学习率在噪声大数据上更优；数据稀缺时指纹+MLP更稳健；

**⚠️ 局限性**

局限包括：仅支持离散化的binned预测，可能忽略连续碎片信息；GFv2等高级模型计算量大、显存占用高；公开数据的类别偏差和专有数据缺失限制泛化；未来需扩展至碎片化/公式预测、实时碰撞能量变化等。

---

## 66. Accelerating LLM Pre-Training through Flat-Direction Dynamics Enhancement

**arXiv ID:** 2602.22681 | [PDF](https://arxiv.org/pdf/2602.22681v1)

**作者:** Shuchen Zhu `[一作]` (Peking University), Zaiwen Wen `[通讯]` (Peking University)

**通讯引用:** 4763 | [OpenAlex ID](https://openalex.org/A5006127137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为LITE的通用加速策略，通过在损失面貌中的平坦方向上增大学习率和Hessian阻尼系数，显著提升LLM预训练速度。

**💡 创新点**

创新点在于：①构建统一的黎曼几何ODE框架，阐释预条件器与动量协同作用；②基于该框架设计了可直接应用于Muon、SOAP等矩阵式自适应优化器的平坦方向加速机制。

**🔧 技术方法**

采用黎曼几何ODE分析、动量与预条件的协同设计、梯度与Hessian的Kronecker分解预估、平坦/尖锐子空间投影与学习率/阻尼系数调节等技术。

**📊 数据集**

使用C4、Pile等公开大规模文本数据，对LLaMA2（0.13B–1.3B）和QwenMoE（1B）两类模型进行实验。

**📈 对比分析**

与原始Muon、SOAP以及其Nesterov加速版本对比，LITE在所有模型与数据上都取得更低的终端损失，长时间训练时可实现约2×速度提升，且在模型规模扩大时保持良好扩展性。

**⚠️ 局限性**

局限性包括：①加速策略需要先估计平坦子空间，计算开销与内存开销不低；②若在尖锐方向上误用，可能导致性能下降；③依赖预条件器与Hessian的对齐假设，可能不适用于所有优化器或极大规模模型；④超参数（χ、β₂）仍需手工调节，缺乏自适应机制。

---

## 67. Doubly Adaptive Channel and Spatial Attention for Semantic Image Communication by IoT Devices

**arXiv ID:** 2602.22794 | [PDF](https://arxiv.org/pdf/2602.22794v1)

**作者:** Soroosh Miri `[一作]` (Iran University of Science and Technology), S. Mohammad Razavizadeh `[通讯]` (Iran University of Science and Technology)

**通讯引用:** 938 | [OpenAlex ID](https://openalex.org/A5051253787)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了双重自适应通道与空间注意力的深度联合源信道编码（DA‑DJSCC），用于 IoT 设备的语义图像通信。

**💡 创新点**

引入了同时自适应通道注意力和空间注意力模块，使编码器和解码器能根据实时 SNR 动态调整特征权重，避免为不同 SNR 训练多套模型，实现在单一模型中双重自适应。

**🔧 技术方法**

采用深度卷积网络、通道注意力+空间注意力模块、SNR 嵌入、端到端训练、MSE 损失、Adam 优化等技术。

**📊 数据集**

使用 CIFAR‑10 和 CIFAR‑100 数据集进行训练与评估。

**📈 对比分析**

与原始 DJSCC（无注意力）和 ADJSCC（仅通道注意力）在 PSNR、SSIM 和下游分类准确率上进行对比（压缩率 k/n=1/12）。实验显示 DA‑DJSCC 在大多数 SNR 区间均优于两者，尤其在 10 dB 以上显著提升。

**⚠️ 局限性**

计算复杂度和参数量略高于 ADJSCC（约 50% 增加），对极度资源受限的 IoT 设备仍有一定负担；在极低 SNR 下仍略逊于最简单的 DJSCC。

---

## 68. DuoMorph: Synergistic Integration of FDM Printing and Pneumatic Actuation for Shape-Changing Interfaces

**arXiv ID:** 2602.22604 | [PDF](https://arxiv.org/pdf/2602.22604v1)

**作者:** Xueqing Li `[一作]` (Tsinghua University and Hong Kong Polytechnic University), Qiuyu Lu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 249 | [OpenAlex ID](https://openalex.org/A5089822454)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

开发了一种将 FDM 打印与热封气动结构融合的 DuoMorph 方法，用单一 FDM 打印机实现热封与 3D/4D 打印，制造可变形界面。

**💡 创新点**

创新在于将热封气泡与 FDM 打印互相约束、共成型，提出四类交互原语和一体化设计工具，实现在单机下完成热封、打印与形变。

**🔧 技术方法**

使用 FDM 打印机的热封功能、TPU/尼龙织物热压、4D 打印热致收缩、交互式 Rhino/Grasshopper 设计插件以及自动生成 G-code 的脚本。

**📊 数据集**

使用实验制备的 TPU/尼龙薄膜和打印样本进行拉伸、剪切与摩擦力测试，而非公开数据集。

**📈 对比分析**

通过拉伸、剪切、摩擦系数等物理测试与传统单一气动或单一打印结构对比，证明同种材料间粘结更强，气密性优良，经过 1000 次循环无泄漏，功能稳定。

**⚠️ 局限性**

仅能在气囊一侧打印，柔性基底高度控制受限，双面 TPU 覆膜需手工贴附，需改进夹具与多材料打印以提升稳定性与通用性。

---

## 69. Regularized Online RLHF with Generalized Bilinear Preferences

**arXiv ID:** 2602.23116 | [PDF](https://arxiv.org/pdf/2602.23116v1)

**作者:** Junghyun Lee `[一作]` (Korea Advanced Institute of Science and Technology), Se-Young Yun `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1644 | [OpenAlex ID](https://openalex.org/A5091674853)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本论文研究了在通用偏好学习（General Preference Learning）框架下的在线 RLHF 问题，采用 Generalized Bilinear Preference Model（GBPM）来描述可能的非可传递偏好，并在任意强凸正则化（例如逆 KL、Tsallis、χ² 等）下提供了统计学上最优的收敛率。

**💡 创新点**

创新点主要有三点：① 对任意强凸正则化的通用自我绑定二次不等式，证明任何贪婪 NE 策略的对偶间隙被估计误差的平方所上界；② 通过该结论消除了之前方法中对逆 KL 正则化的指数级 η 依赖，获得 poly‑log（log²T）或 d‑free 的收敛率；③ 在高维低秩场景下利用核范数 MLE 与 Explore‑Then‑Commit 策略实现了 O(√(η r T)) 的 d‑free 收敛。

**🔧 技术方法**

技术上主要使用：GBPM 的低秩斜对称矩阵结构；强凸正则化的拉格朗日对偶分析；自我绑定二次不等式（自共轭性）与积分概率度量；MVP/核范数约束最大似然估计；Elliptical Potential Lemma 与 Freedman 不等式；特征多样性假设下的覆盖 Lemma。

**📊 数据集**

实验数据未给出具体公开数据集，主要在仿真环境下使用随机特征向量和随机对偶样本进行验证。

**📈 对比分析**

与传统仅适用于逆 KL 正则化的在线 RLHF 方法比较，本文方法在保持任意强凸正则化的同时，消除了 e^(η) 依赖，实现了更快的收敛率；在模拟实验中表现出相对更小的累计对偶间隙和更低的误差。

**⚠️ 局限性**

局限性包括：仍需特征多样性假设（若去除则需额外的 η 依赖）；实验验证有限，缺乏真实 LLM 对齐任务的数据集；对非上下文或更复杂对齐场景的理论推广仍待进一步研究。

---

## 70. Devling into Adversarial Transferability on Image Classification: Review, Benchmark, and Evaluation

**arXiv ID:** 2602.23117 | [PDF](https://arxiv.org/pdf/2602.23117v1)

**作者:** Xiaosen Wang `[一作]` (Huazhong University of Science and Technology), Yuyang Luo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对图像分类中的转移式对抗攻击做了系统性综述，梳理了100余种攻击方法，并提出了统一的六类分类体系与评估框架。

**💡 创新点**

创新点在于：①构建了首个标准化评估基准，统一实验设置避免了比较偏差；②在六大攻击类别中归纳了提升转移性的共性策略与关键瓶颈；③对跨模型、跨数据集和跨域的转移性进行了宏观梳理，揭示了共享的系统级弱点。

**🔧 技术方法**

采用了梯度优化、输入变换、目标函数改进、模型相关改造、集成与生成等技术手段，结合Momentum、Variance Tuning、Feature Mixing、Transformer-特有的Token/Attention调节等方法。

**📊 数据集**

实验基于ImageNet兼容数据集（224×224尺寸）进行，使用四种CNN（ResNet‑50、VGG‑16、MobileNet‑v2、Inception‑v3）、四种Vision Transformer（ViT、PiT、Visformer、Swin）以及五种防御模型（AT、HGD、RS、NRP、DiffPure）。

**📈 对比分析**

作者采用攻击成功率（ASR）作为统一度量，统一对每种攻击在同一受害者模型、相同扰动预算（ε=16/255、α=1.6/255、T=10或300）下进行横向对比；结果表明，Momentum/Variance、Feature‑mixing、Transformer‑特定策略在不同类别中往往能取得显著提升，而许多新方法在对比中并未超越基线，凸显评估偏差。

**⚠️ 局限性**

局限性包括：①评估仍受限于单一数据集和受害者模型集合；②对强防御模型的鲁棒性验证不足；③生成对抗样本的保存与实际部署（像素范围、文件格式）未充分考量；④多变形/集成攻击的计算成本高，缺乏系统的开销评估；⑤针对目标攻击与更复杂场景的探索仍不充分。

---

## 71. Iterative Prompt Refinement for Dyslexia-Friendly Text Summarization Using GPT-4o

**arXiv ID:** 2602.22524 | [PDF](https://arxiv.org/pdf/2602.22524v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 72. AviaSafe: A Physics-Informed Data-Driven Model for Aviation Safety-Critical Cloud Forecasts

**arXiv ID:** 2602.22298 | [PDF](https://arxiv.org/pdf/2602.22298v1)

**作者:** Zijian Zhu `[一作]` (Fudan University), Hao Li `[通讯]` (Fudan University)

**通讯引用:** 30946 | [OpenAlex ID](https://openalex.org/A5100348631)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种用于航空安全的分层、物理约束的神经网络气象预报模型AviaSafe，能够在全球范围内每6小时预测四种云微物理种类（CIWC、CLWC、CRWC、CSWC）并延伸至7天

**💡 创新点**

创新点在于将航空气象验证的Icing Condition (IC)指数嵌入深度学习体系，采用掩码预测与分层解码器同时处理云分布与浓度，并通过物理约束提升预测的物理一致性

**🔧 技术方法**

使用Swin Transformer V2、掩码预测网络、物理约束IC模块、加权Charbonnier损失和Focal损失的端到端训练框架

**📊 数据集**

基于ERA5重分析数据（每6小时，1°×1°网格，13个压力层，共117通道）进行训练与评估

**📈 对比分析**

与ECMWF HRES以及FuXi基线模型对比，AviaSafe在93.7%变量-预报时限组合上表现更好，尤其在云微物理变量上显著降低RMSE，并在背景变量上保持与HRES相当或更优的ACC

**⚠️ 局限性**

局限在于未覆盖更高空间分辨率、较长预报时限及缺乏观测数据融合，未来需提升分辨率、延长预测周期并引入实时观测信息

---

## 73. Align then Adapt: Rethinking Parameter-Efficient Transfer Learning in 4D Perception

**arXiv ID:** 2602.23069 | [PDF](https://arxiv.org/pdf/2602.23069v1)

**作者:** Yiding Sun `[一作]` (Xi'an Jiaotong University), Yaonan Wang `[通讯]` (Hunan University)

**通讯引用:** 21333 | [OpenAlex ID](https://openalex.org/A5025640070)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了“Align then Adapt”两阶段参数高效迁移学习框架，将已有的 3D 点云预训练模型迁移至 4D 点云视频任务，解决过拟合与模态差距问题。

**💡 创新点**

创新点在于：①利用 Optimal Transport（OTDD）先对齐 3D 与 4D 数据分布，再用轻量化 Point Video Adapter（PVA）和 Spatial Context Encoder（SCE）完成时空建模；②将迁移过程拆分为 Stage1 的分布对齐与 Stage2 的高效微调，从而显著降低参数量并提升泛化性能。

**🔧 技术方法**

技术细节包括 Optimal Transport 计算 OTDD、Point Align Embedder、深度可分离卷积瓶颈的 Point Video Adapter、共享参数的 Spatial Context Encoder、冻结 3D Backbone 的参数高效微调。

**📊 数据集**

在多种 3D/4D 点云视频数据集上进行评估，主要包括 MSR-Action3D、HOI4D、Synthia 4D、NTU RGB+D、SHREC'17、KITTI 等。

**📈 对比分析**

与全微调、传统 4D PETL 方案和专门 4D 预训练模型对比，PointATA 在 3D 动作识别 97.21%、4D 动作分割 +8.7%、4D 语义分割 84.06% 等指标均能匹敌或超越全微调，同时参数量、训练时间和推理速度均更为高效。

**⚠️ 局限性**

局限性在于对某些户外或复杂场景的建模能力仍不足，Stage1 的 OTDD 计算成本较高，且目前仅验证了对单一 3D 预训练模型的迁移效果，未来需进一步扩展到更大规模或多模态模型。

---

## 74. Automating the Detection of Requirement Dependencies Using Large Language Models

**arXiv ID:** 2602.22456 | [PDF](https://arxiv.org/pdf/2602.22456v1)

**作者:** Ikram Darif `[一作]` (University of Ottawa), Arun Adiththan `[通讯]` (General Motors)

**通讯引用:** 116 | [OpenAlex ID](https://openalex.org/A5014348074)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自动化检测自然语言需求间依赖关系的方法，构建了LEREDD框架。

**💡 创新点**

创新点在于将检索增强生成（RAG）与上下文学习（ICL）结合到LLM中，实现动态检索领域知识与示例，能够一次性识别多种直接依赖类型。

**🔧 技术方法**

使用技术包括：大语言模型（GPT‑4.1等），SBERT嵌入检索，欧氏/余弦相似度聚合，few‑shot + RAG提示，自动生成依赖预测、理由与置信度。

**📊 数据集**

数据集：手工标注的 813 对需求（覆盖三套汽车系统——Traffic Jam Assist、Automated Parking Assist、Adaptive Driving Beam），每对标注 7 种依赖类型或无依赖。

**📈 对比分析**

比较方法：在三套系统上分别进行内部数据集与跨数据集实验，与两种 SOTA 基线（TF‑IDF+LSA、微调 BERT）对比。LEREDD 的平均准确率 92.66%，宏观 F1 分数 84.33%，尤其对 No‑dependency 的 F1 高达 96%，显著优于基线。

**⚠️ 局限性**

局限性：仅检测直接依赖，未覆盖间接/隐式依赖；数据集局限于汽车领域，可能不具备跨领域泛化；标注过程存在主观性（Kappa 0.43），导致部分细粒度依赖识别仍受限。

---

## 75. Marinarium: a New Arena to Bring Maritime Robotics Closer to Shore

**arXiv ID:** 2602.23053 | [PDF](https://arxiv.org/pdf/2602.23053v1)

**作者:** Ignacio Torroba `[一作]` (Royal Institute of Technology), Ivan Stenius `[通讯]` (Royal Institute of Technology)

**通讯引用:** 823 | [OpenAlex ID](https://openalex.org/A5074649875)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并搭建了 Marinarium 这一低成本、模块化的水下实验设施，结合双域（水下+空中）Mocap、数字孪生与空间实验室，实现了高精度的多域实验、系统辨识、仿真与空间机器人验证。

**💡 创新点**

创新点包括：① 将水下实验与空间实验室实现紧密耦合；② 采用 Koopman 运算符（edmdc+RBF）进行数据驱动的系统辨识，显著提升预测精度；③ 通过学习残差动力学实现仿真模型的 Sim2Real 桥接；④ 在同一控制栈下验证水下机器人与空间自由飞行器的轨迹跟踪，证明水下环境可作为空间任务的近似平台。

**🔧 技术方法**

主要技术手段有：ROS2、Qualisys/Optitrack 双域Mocap、SMaRCSim 数字孪生、Koopman 运算符与 RBF 字典、残差动力学 MLP、非线性 MPC、STL 轨迹规划、数据集自动化记录与标注。

**📊 数据集**

使用的数据集包括：① 3 条 15 分钟 BlueROV2 运动记录（共 45k+ 时序样本）；② 8 条 SAM AUV 录制用于残差学习（约 73k 时序样本）；③ 额外的 USV、UAV 与地面站交互记录，用于多域协同实验。

**📈 对比分析**

通过 RMSE 对比：在 1、10、100 步预测中，Koopman 模型优于传统 Fossen 物理模型和线性加速模型；Sim2Real 残差模型在 1、10、100、500 步范围内显著降低姿态与速度误差（约 10‑30% 改进），但对角速度误差提升有限；多域任务成功完成，轨迹跟踪误差均在几厘米以内。

**⚠️ 局限性**

局限性包括：① 残差模型对高角速度转弯不稳定，需进一步分离线性/角速度残差；② 仅在 2D 平面验证空间任务，缺乏完整 6DOF 对比；③ 设施规模受限，难以模拟真实海洋或空间极端环境；④ 需要更丰富的数据覆盖和自适应学习机制以提升泛化性。

---

## 76. PGVMS: A Prompt-Guided Unified Framework for Virtual Multiplex IHC Staining with Pathological Semantic Learning

**arXiv ID:** 2602.23292 | [PDF](https://arxiv.org/pdf/2602.23292v1)

**作者:** Fuqiang Chen `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), Wenjian Qin `[通讯]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于提示引导的统一虚拟多重IHC染色框架PGVMS，能够从H&E图像一次性生成多种IHC染色结果。

**💡 创新点**

创新点在于三项：①利用CONCH病理视觉语言模型实现病理语义引导的PSSG生成器；②基于光密度的蛋白质感知学习策略PALS以保持蛋白表达分布；③原型一致学习策略PCLS以校正空间对齐并增强语义一致性。

**🔧 技术方法**

使用的技术包括Prompt-guided Transformer与GAN/生成对抗网络、光密度映射、对比学习、实例/层归一化混合以及多级蛋白质一致损失。

**📊 数据集**

训练数据来自MIST和IHC4BC乳腺癌数据集，评估数据集包括两者以及多器官跨癌症的临床样本。

**📈 对比分析**

与一系列一对一和一对多虚拟染色方法（如CycleGAN、Pix2Pix、CUT、PyramidP2P、VIMs等）对比，PGVMS在蛋白表达一致性指标IOD、Pearson-R以及感知指标FID、DISTS上均显著优于对手，整体性能位居SOTA。

**⚠️ 局限性**

局限性在于对分散型染色（如Ki‑67）表现相对较弱；缺乏针对不同标记的空间先验约束，导致某些定位不够精准；以及对极端小样本或高异质性病理图像的泛化仍待进一步验证。

---

## 77. Sorting Methods for Online Deliberation: Towards a Principled Approach

**arXiv ID:** 2602.23168 | [PDF](https://arxiv.org/pdf/2602.23168v1)

**作者:** Nicolien Janssens `[一作]` (Erasmus University Rotterdam), Frederik van de Putte `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文系统性分析在线审议平台（DP）中议案排序方法的现状与理论基础，并提出一种概念框架来评估排序方法的目的与参数。

**💡 创新点**

创新点在于将议案排序视为民主过程的核心设计问题，批判单一“赞同数优先”方法，提出整合内容、评估与元数据的全局排序策略（比例排序与最大覆盖），并首次将这些方法与实践平台对照。

**🔧 技术方法**

主要技术包括：构建概念框架、定性案例研究、对现有排序算法的分类与比较；对比例排序与最大覆盖方法参考了计算社会选择与赞同投票理论。

**📊 数据集**

使用的数据集为：八个主流DP（Consul、Decidim、Go Vocal、Your Priorities、LiquidFeedback、OpenStad、Adhocracy+、Parta）中19个案例，平均每案约2724条议案；此外对平台的排序功能进行文献与访谈收集。

**📈 对比分析**

通过概念框架对比，发现多数平台使用单一参数排序，且缺乏民主价值说明；引入全局排序后理论上能提升包容性与代表性，但本文未给出量化性能指标，仍停留在理论与案例对比层面。

**⚠️ 局限性**

局限性包括：缺乏实证实验验证排序方法对议案质量、合法性与用户偏好的影响；样本仅覆盖八个平台，未涵盖所有DP；未提供对不同排序方法在用户体验上的定量评估。

---

## 78. RLHFless: Serverless Computing for Efficient RLHF

**arXiv ID:** 2602.22718 | [PDF](https://arxiv.org/pdf/2602.22718v1)

**作者:** Rui Wei `[一作]` (Stevens Institute of Technology), Hao Wang `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 41439 | [OpenAlex ID](https://openalex.org/A5080102032)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了首个面向 RLHF 训练的无服务器框架，利用无服务器弹性资源动态调整采样演员数量、去重预填充缓存、基于历史响应长度的提示分配和切割迁移策略，以提高训练效率并降低成本。

**💡 创新点**

核心创新包括：无服务器动态演员扩缩；去重预填充机制消除 KV 缓存重复计算；响应长度预测与切割迁移的提示分配；以及基于成本与速度平衡的演员缩放决策。

**🔧 技术方法**

采用了 vLLM 推理引擎、VERL 训练后端、Ray 调度、Redis 缓存、MPI/NCCL 通信，以及自定义的成本与时间估算模型。

**📊 数据集**

在 GSM8K、GPQA、LiveCodeBench 三大基准上，使用 Qwen2.5-3B/7B、Llama2-70B 等模型进行实验。

**📈 对比分析**

与 VERL 基线及 RLHFuse 对比，实验显示在物理集群上训练速度提升最高 1.35×，成本下降最高 44.8%；在大规模模拟集群上平均提升 1.23×、成本下降 38.7%。

**⚠️ 局限性**

目前仅针对同步 RLHF 训练，未对异步训练或多模型多任务场景进行评估，且对极大规模模型的分布式同步开销仍有待进一步优化。

---

## 79. Detection and Recognition: A Pairwise Interaction Framework for Mobile Service Robots

**arXiv ID:** 2602.22346 | [PDF](https://arxiv.org/pdf/2602.22346v1)

**作者:** Mengyu Liang `[一作]`, Iolanda Leite `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种两阶段轻量级方法，先用几何与运动特征快速检测人对交互，再用关系网络对交互类型进行粗粒度分类，旨在为移动服务机器人提供社交感知支持。

**💡 创新点**

创新点在于把配对交互视为机器人最小且足够的社交单元，完全摆脱骨架和深度视觉特征，依靠边框几何与光流即可实现；并在两阶段框架中实现高召回与低计算开销的平衡。

**🔧 技术方法**

使用的技术包括：bounding box几何特征提取、Farneback光流运动特征、冻结EfficientNet提取外观特征、关系网络（Relation Network）做对偶关系推理，以及轻量化的MLP分类头。

**📊 数据集**

主要使用了JRDB数据集进行训练与评估，同时在Collective Activity Dataset（CAD）上做比较，并在自采的割草机摄像头数据上进行零样本迁移验证。

**📈 对比分析**

与现有群体活动识别和骨架/外观为主的配对交互方法比较，实验表明：在JRDB上，纯几何+运动特征即可获得≈84%准确率，加入外观特征提升不大但计算量显著增加；在CAD上，配对交互模型取得≈79%准确率，优于简单聚类基线，且模型复杂度更低；在割草机零样本测试中，检测阶段精度96.5%，分类宏F1≈0.51。

**⚠️ 局限性**

局限性包括：仅覆盖粗粒度交互类别（步行、站立、坐下），无法识别更细微或长期社交行为；依赖可靠的人检测与跟踪，严重遮挡或极端拥挤场景下性能可能下降；以及在强自运动环境下光流误差导致的分类误差。

---

## 80. LeRobot: An Open-Source Library for End-to-End Robot Learning

**arXiv ID:** 2602.22818 | [PDF](https://arxiv.org/pdf/2602.22818v1)

**作者:** Remi Cadene `[一作]` (Hugging Face), Thomas Wolf `[通讯]` (Hugging Face)

**通讯引用:** 16609 | [OpenAlex ID](https://openalex.org/A5078865608)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个统一的开源机器人学习堆栈，集成了低级控制中间件、标准化多模态数据集、可复现的 PyTorch SOTA 算法实现以及异步推理框架，支持多种真实机器人与模拟环境。

**💡 创新点**

创新点在于打通从机器人硬件抽象、数据采集与流式存储、统一数据格式，到可复现的学习算法和远程异步推理的端到端闭环，显著降低研究与应用的工程门槛。

**🔧 技术方法**

使用的技术包括 Python API 与统一中间件、PyTorch 原生实现、自定义多模态数据格式（支持高帧率感知、相机与操作信号）、远程推理服务器、逻辑与物理层的异步推理栈。

**📊 数据集**

利用了 16K+ 个公开演示数据集（涵盖 SO‑10X、SO‑101、ALOHA‑2、Franka Panda、xArm 等多种平台），采用统一格式并支持实时流式访问，构成了大规模实验数据基础。

**📈 对比分析**

通过在 LIBERO、Meta‑World 等模拟环境以及真实机器人上对 ACT、Diffusion Policy、SmolVLA 等算法进行基准测试，展示了模型上传/下载趋势、推理延迟与内存占用与理论估计相符，验证了推理框架的可扩展性和性能。

**⚠️ 局限性**

局限性包括机器人覆盖面不完整、算法实现范围有限、推理性能仍需低级优化（如量化、图编译），以及对社区贡献的高度依赖。

---

## 81. Rethinking the Practicality of Vision-language-action Model: A Comprehensive Benchmark and An Improved Baseline

**arXiv ID:** 2602.22663 | [PDF](https://arxiv.org/pdf/2602.22663v1)

**作者:** Wenxuan Song `[一作]` (Hong Kong University of Science and Technology), Haoang Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 900 | [OpenAlex ID](https://openalex.org/A5040338788)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了跨体现的基准CEBench，系统研究了Vision‑Language‑Action（VLA）模型的轻量化、训练流程和统一动作空间，并在此基准上提出一种轻量、无预训练的VLA模型，能够在多种模拟和真实任务中实现移动与双臂操作。

**💡 创新点**

创新点包括：①构建包含模拟与真实、域随机化的跨体现基准CEBench；②利用多视角拼接、感知tokenizer和动作分块等技术实现0.5B参数的轻量化VLA；③提出两阶段后训练+微调的无预训练范式；④设计离散+方向‑值的统一动作空间，实现移动与操作的无缝切换。

**🔧 技术方法**

采用LLaVA‑OneVision‑0.5B VLM骨干，垂直拼接第一人称与第三人称图像，多视角融合；感知tokenizer将本体信息转化为序列token；动作分块（chunk size 5）与离散动作标记化；两阶段训练：后训练在多任务数据上学习视觉‑语言‑动作映射，随后微调单任务；通过域随机化和多任务评估验证泛化能力。

**📊 数据集**

使用CEBench数据集：14.4k条模拟轨迹（36任务）和1.6k条真实轨迹（8任务），涵盖CALVIN、RoboTwin以及Cobot‑Magic双臂移动机器人；同时引用CALVIN官方数据。

**📈 对比分析**

在CALVIN、RoboTwin及真实世界任务上与ACT、Diffusion Policy、TinyVLA、RDT等轻量基线以及3B RoboFlamingo、7B OpenVLA等大模型比较。0.5B模型在CALVIN多子任务成功率>90%，平均轨迹长度3.68，超过7B模型；在RoboTwin域随机化环境中成功率≈40%；在真实移动任务中实现≈30%成功率；全部训练在8 H100 GPU完成，无预训练。

**⚠️ 局限性**

局限性包括：对极端复杂或高精度连续控制的鲁棒性仍待验证；统一动作空间在极端移动场景下可能出现不稳定；目前验证仅覆盖特定机器人平台，跨平台迁移性尚需进一步探索。

---

## 82. InnerQ: Hardware-aware Tuning-free Quantization of KV Cache for Large Language Models

**arXiv ID:** 2602.23200 | [PDF](https://arxiv.org/pdf/2602.23200v1)

**作者:** Sayed Mohammadreza Tayaranian Hosseini `[一作]` (McGill University), Warren J. Gross `[通讯]` (McGill University)

**通讯引用:** 8337 | [OpenAlex ID](https://openalex.org/A5091600002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大语言模型的 KV 缓存进行量化，降低解码时的内存占用和延迟。

**💡 创新点**

通过在 KV 缓存的内维度进行分组、混合对称/非对称量化、高精度窗口以及通道归一化，构建了硬件友好的低延迟量化方案。

**🔧 技术方法**

使用组内量化、混合量化模式、高精度窗口、通道归一化，并在 Triton 上实现融合 dequant + 向量-矩阵乘法的内核。

**📊 数据集**

在 Llama 系列模型上使用 GSM8K 数据集进行 few‑shot 评估。

**📈 对比分析**

与无量化基线以及 KIVI、KIVI+sink 等先行方法对比，InnerQ 在保持或略高准确率的前提下，将 KV 缓存压缩到 2 位，并使解码延迟比基线低 22%（相较于其他 KV 量化方法）甚至 88%（相较于半精度乘法）。

**⚠️ 局限性**

需要额外存储混合量化掩码导致缓存稍大；对极端长序列或不同任务的鲁棒性尚未完全验证；高精度窗口长度需手工调参。

---

## 83. Plug, Play, and Fortify: A Low-Cost Module for Robust Multimodal Image Understanding Models

**arXiv ID:** 2602.22644 | [PDF](https://arxiv.org/pdf/2602.22644v1)

**作者:** Siqi Lu `[一作]` (National University of Defense Technology), Jianhang Yao `[通讯]` (National University of Defense Technology)

**通讯引用:** 19252 | [OpenAlex ID](https://openalex.org/A5100319048)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出基于频域的模态偏好评估与动态权重分配方法（FRM 与 MWAM），用以解决多模态模型在缺失模态时的性能退化问题。

**💡 创新点**

创新点在于：①使用频率比率指标 FRM 在频域量化每个模态的学习偏好；②构建可插拔的 MWAM，通过 FRM 自动调节梯度或损失空间中的模态权重，从而实现训练过程中的模态均衡。

**🔧 技术方法**

主要技术包括离散余弦变换（DCT）提取低频/高频特征、FRM 计算公式、动态权重函数、以及梯度编辑和损失加权两种干预机制。

**📊 数据集**

实验使用了脑肿瘤分割数据集 BRATS2020、NYU‑Depth V2 语义分割数据集以及 CASIA‑SURF 多模态分类数据集。

**📈 对比分析**

与 HeMIS、RFNet、mmFormer、GSS、ESANet‑MD、MMANet 等 SOTA 方法比较，MWAM 在 Dice、MIoU、PCR 等指标上平均提升 1–3%，并在多模态缺失场景中显著降低 PCR，表明方法稳健且兼容多种网络架构。

**⚠️ 局限性**

局限性包括：对不同模态组合的性能提升不均匀，需手动调节权重衰减超参数；在极端模态缺失（如全部缺失）时仍无法完全恢复原性能；频域特征提取增加了少量计算开销。

---

## 84. ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive Integration of Mobility Proposals

**arXiv ID:** 2602.22666 | [PDF](https://arxiv.org/pdf/2602.22666v1)

**作者:** Xuelu Li `[一作]` (Shandong University), Changhe Tu `[通讯]` (Shandong University)

**通讯引用:** 3107 | [OpenAlex ID](https://openalex.org/A5087472282)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了基于3D Gaussian Splatting的自监督框架ArtPro，用以高保真重建关节运动的多部件物体。

**💡 创新点**

创新点在于用过度分割的运动候选初始化并在优化中动态合并，结合碰撞感知剪枝实现自校正的移动部分推理。

**🔧 技术方法**

主要技术包括3D Gaussian Splatting、几何与运动先验驱动的过度分割、基于运动一致性的自适应合并和碰撞检测剪枝。

**📊 数据集**

实验使用合成数据、PartNet-Mobility多部件样本以及真实RGB‑D图像数据集。

**📈 对比分析**

与PARIS、ArticulatedGS、DTA、ArtGS等方法比较，ArtPro在网格重建和关节估计指标上显著优于其余方法，尤其在复杂多部件场景下表现更稳健。

**⚠️ 局限性**

局限性在于仅依赖两状态间的运动信息，对对称或细微运动的分辨力不足，且对传感器噪声和边界细节的敏感度仍需提升。

---

## 85. Seeing Graphs Like Humans: Benchmarking Computational Measures and MLLMs for Similarity Assessment

**arXiv ID:** 2602.22416 | [PDF](https://arxiv.org/pdf/2602.22416v1)

**作者:** Seokweon Jung `[一作]` (Korea Advanced Institute of Science and Technology), Jinwook Seo `[通讯]` (Seoul National University)

**通讯引用:** 4389 | [OpenAlex ID](https://openalex.org/A5012388103)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过三组实验研究人类对节点连线图相似性的视觉感知，并将此感知与传统图相似性度量及最新多模态大语言模型（LLM）的表现进行对比评估。

**💡 创新点**

创新之处在于首次将人类主观相似性判断与多模态LLM的判定结果系统性对照，提出以相对比较任务为核心的实验流程，并发现GPT‑5在相似性判断与可解释性上优于传统度量。

**🔧 技术方法**

采用了16种传统图相似性度量（包括Portrait Divergence等），以及三款先进的多模态LLM（GPT‑5、Gemini 2.5 Pro、Claude Sonnet 4.5），并通过Cohen κ、Spearman ρ、ANOVA等统计方法对结果进行评估。

**📊 数据集**

使用了1,881幅节点连线图，包含1,152幅合成图（GNM、BBA、NWS、SBM）和729幅实测图（来自SNAP等动态图切片），覆盖四种图尺寸、三种边密度和三种布局。

**📈 对比分析**

通过Cohen κ衡量LLM或传统度量与人类判断的一致性，使用Spearman ρ检验与人类置信度的相关性。Portrait Divergence为最佳传统度量（κ≈0.424，ρ≈0.269），而GPT‑5进一步提升至κ≈0.479、ρ≈0.353，Gemini和Claude亦超越传统度量，且LLM能够提供可解释的自然语言理由。

**⚠️ 局限性**

局限性包括仅考察无权、无自环、单连通且无节点对应的基本图；实验参与者主要是具备图数据经验的中等水平群体；LLM推理时间相对较长；与人类置信度的相关性仍偏弱；对更复杂图类型（如加权图、已知节点对应）及更大规模数据的泛化尚待验证。

---

## 86. How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?

**arXiv ID:** 2602.22441 | [PDF](https://arxiv.org/pdf/2602.22441v1)

**作者:** Yingqian Cui `[一作]` (Michigan State University), Benoit Dumoulin `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多种隐式推理（latent reasoning）方法进行系统性实验分析，评估其是否真正依赖多步推理，探讨其在连续潜在空间中的搜索行为，并揭示监督强度与潜在空间探索之间的权衡。

**💡 创新点**

首次用多维度实验证据揭示隐式推理存在“shortcut”依赖，并验证其并非真正的BFS式搜索；提出改进的Coconut训练方案以缓解推理崩塌和shortcut问题；总结监督强度对推理多样性和准确性产生的冲突。

**🔧 技术方法**

通过隐层深度变化、噪声干预、注意力分布分析、潜在-文本混合推理回放（hybrid rollouts）等实验技术；对比弱监督（Coconut、CODI）与强监督（SIM‑CoT、CoLaR）四种方法的性能与多样性。

**📊 数据集**

使用基准数据集GSM8K‑Aug（数学推理）和ProsQA（逻辑推理）以及原始GSM8K、ProsQA；模型为GPT‑2和Llama‑3.2‑1B‑Instruct。

**📈 对比分析**

与传统CoT、其他隐式推理方法对比，发现：在噪声下隐式推理仍保持一定准确度，表明存在shortcut；改进的Coconut在GSM8K‑Aug上从34.09%提升到41.06%；在Pass@100上隐式推理比显式CoT高20%+，但多数投票准确率略低。整体表现显示弱监督方法多样性高但易 shortcut，强监督方法准确率高但多样性低。

**⚠️ 局限性**

主要局限：隐式推理仍依赖shortcut，无法真正实现多路径BFS搜索；潜在空间虽能编码多种可能，但缺乏有效的聚合机制，导致多样性未能转化为更高准确率；监督强度与多样性之间的权衡难以调节，尚需更平衡的训练策略。

---

## 87. ManifoldGD: Training-Free Hierarchical Manifold Guidance for Diffusion-Based Dataset Distillation

**arXiv ID:** 2602.23295 | [PDF](https://arxiv.org/pdf/2602.23295v1)

**作者:** Ayush Roy `[一作]` (University at Buffalo), Vishnu Suresh Lokhande `[通讯]` (University at Buffalo)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5081203502)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种完全训练免费的扩散模型数据蒸馏框架ManifoldGD，用以合成小型合成数据集，保持与原大规模数据集相同的模型性能。

**💡 创新点**

核心创新是：①利用VAE潜在空间的层次二分聚类生成多尺度的IPC（类内原型）中心，构建多尺度coreset；②在每一步去噪过程中，构造局部“扩散流形”，并将模式引导向量投影到该流形的切空间，从而兼顾语义一致性与几何一致性，避免离流形漂移。

**🔧 技术方法**

技术包括：预训练扩散模型（如DiT/Latent Diffusion）、VAE潜在编码、层次二分聚类、流形切空间投影、核化模式引导（RBF/Laplace/IMQ）与正则化协方差估计。

**📊 数据集**

在ImageNette、ImageWoof和ImageNet-100三个子集上进行实验，使用ConvNet‑6、ResNet‑AP‑10、ResNet‑18等骨干网络进行下游训练。

**📈 对比分析**

与现有训练免费方法（DiT、MGD）及训练基方法（DM、D4M、GLAD）在hard‑label协议下对比，ManifoldGD在分类准确率、FID、L2/MMD、代表性与多样性指标上均实现显著提升（例如ImageNette IPC=10时Acc提升约2–3%，FID下降≈20%）。

**⚠️ 局限性**

局限性包括：①高噪声阶段邻域被破坏，导致切空间估计不准；②低秩近似对高度弯曲的流形容易过平滑，可能压缩多样性；③对超参数（邻域半径、正则化、λ_man）敏感，需经验调节；④缺乏理论对投影误差和曲率敏感度的正式分析。

---

## 88. Sequential Regression for Continuous Value Prediction using Residual Quantization

**arXiv ID:** 2602.23012 | [PDF](https://arxiv.org/pdf/2602.23012v1)

**作者:** Runpeng Cui `[一作]` (Kuaishou Technology), Peng Jiang `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于残差量化（Residual Quantization）的序列学习框架 RQ‑Reg，用来对推荐系统中的连续价值（如观看时长、LTV、GMV）进行预测。

**💡 创新点**

创新点在于：①将连续值拆分为粗到细的量化码序列，通过递归预测实现“粗-细”层次化逼近；②在嵌入空间引入 Rank‑N‑Contrast 损失，使得嵌入向量的秩序与目标值的秩序保持一致；③在工业部署中将第一层量化改为 K‑medians，进一步降低聚类误差。

**🔧 技术方法**

核心技术包括残差量化（RQ‑K‑means）、LSTM 序列模型、scheduled sampling、Huber 损失、Rank‑N‑Contrast 损失、嵌入表、以及多层 MLP 编码器/预测器/回归器。

**📊 数据集**

使用的公开数据集有：LTV 任务（Criteo‑SSC、Kaggle）和观看时长任务（KuaiRec、CIKM16）；在线实验使用了 Kuaishou 的 GMV 预测数据。

**📈 对比分析**

与多种现有方法（Two‑stage、MTL‑MSE、ZILN、MDME、MDAN、OptDist、WLR、D2Q、TPM、CREAD、SWaT、GR 等）在 MAE、XAUC、Norm‑Gini、Spearman、AUC、ADVV 等指标上均实现了显著提升，尤其在长尾分布场景中优势更为明显。

**⚠️ 局限性**

局限性包括：①需先预先构建多层码表，可能对极端稀疏或变化剧烈的分布适应性有限；②缺乏对量化层数、码本大小等超参的理论指导，需经验调优；③在样本量极少或目标值范围极宽的任务中，模型可能收敛慢或性能提升有限。

---

## 89. Beyond Detection: Multi-Scale Hidden-Code for Natural Image Deepfake Recovery and Factual Retrieval

**arXiv ID:** 2602.22759 | [PDF](https://arxiv.org/pdf/2602.22759v1)

**作者:** Yuan-Chih Chen `[一作]` (Academia Sinica), Chun-Shien Lu `[通讯]` (Academia Sinica)

**通讯引用:** 3934 | [OpenAlex ID](https://openalex.org/A5042784674)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的隐藏码恢复框架，既能检测和定位深伪，又能通过嵌入多尺度量化隐码实现被篡改图像的完整恢复，并支持事实检索。

**💡 创新点**

创新点包括：
- 采用多尺度 VQ‑VAE 量化得到紧凑的隐码，显著降低嵌入容量；
- 在隐码嵌入前对量化器进行 Dropout 训练，使低尺度就已携带丰富语义；
- 设计条件 Transformer 结合局部定位掩码，逐尺度恢复丢失的高阶隐码；
- 兼容后置与生成端水印两大范式，实现 plug‑and‑play 的通用恢复流程。

**🔧 技术方法**

核心技术：
- 多尺度向量量化（VQ‑VAE）+ Dropout 训练；
- 条件 Transformer（next‑scale 预测）与局部掩码融合；
- 水印编码/解码网络（EditGuard、Gaussian Shading 等）；
- CLIP 用于评估恢复图像的语义相似度。

**📊 数据集**

使用 ImageNet‑S（基于 ImageNet 的图像‑标签‑定位三元组数据集）作为训练与评估基准，涵盖多种自然场景。

**📈 对比分析**

与 HiNet、RePaint、VQGAN、VAR 等方法对比；
- Top‑1 label 0.9231，Top‑1 image 0.8744，CLIP 分数高于所有基线；
- 关键位正确率在 JPEG、噪声、模糊等攻击下几乎保持 1，显示出强鲁棒性；
- 通过事实检索指标和视觉质量展示恢复效果优于传统自恢复水印。

**⚠️ 局限性**

局限性：
- 仍依赖水印容量，过大或过小的隐藏码会影响恢复精度；
- 对极端压缩/噪声下的鲁棒性尚未在更宽泛的数据集上验证；
- 目前主要针对自然图像，复杂场景或多对象遮挡的恢复效果尚需进一步研究。

---

## 90. ColoDiff: Integrating Dynamic Consistency With Content Awareness for Colonoscopy Video Generation

**arXiv ID:** 2602.23203 | [PDF](https://arxiv.org/pdf/2602.23203v1)

**作者:** Junhu Fu `[一作]` (Fudan University), Yi Guo `[通讯]` (Fudan University)

**通讯引用:** 5110 | [OpenAlex ID](https://openalex.org/A5049706279)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于扩散模型的ColoDiff框架，能够生成具有时序一致性与内容可控性的结肠镜视频。

**💡 创新点**

创新点包括：TimeStream模块通过跨帧标记化显式分离时序依赖，实现高效的时间建模；Content‑Aware模块利用噪声注入嵌入与可学习原型实现细粒度的临床属性控制；非马尔科夫采样大幅减少采样步数，支持实时生成。

**🔧 技术方法**

核心技术为扩散模型（基于Latent Diffusion），Transformer结构，TimeStream跨帧注意机制，内容控制的噪声嵌入与原型学习，以及非马尔科夫多步推断。

**📊 数据集**

在四个数据集上评估：公开的Colonoscopic、HyperKvasir、SUN‑SEG以及福州大学医院数据库，共计约4,597段视频。

**📈 对比分析**

与StyleGAN‑V、MoStGAN‑V、LVDM、Endora、FEAT‑L等SOTA方法相比，ColoDiff在FVD、FID、IS等生成指标上均显著更优（如FVD下降至约330‑350，FID降至12.7，IS提升至3.95），并在下游诊断、分割任务中分别提升准确率7.1%与Dice 6.2%。

**⚠️ 局限性**

局限性主要在于目前只能对单一属性（疾病类型或影像模式）进行控制，缺乏多属性联合控制；同时对极端罕见病变的合成能力仍需进一步验证。

---

## 91. Efficient Constructions of Finite-State Independent Normal Pairs

**arXiv ID:** 2602.23030 | [PDF](https://arxiv.org/pdf/2602.23030v1)

**作者:** Subin Pulari `[一作]` `[通讯]` (National Research University Higher School of Economics), Subin Pulari (National Research University Higher School of Economics)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种确定性多项式时间算法，能够有效构造有限状态独立的正常对，并解决了给定正常词的伴随构造问题。

**💡 创新点**

创新点在于通过显式的确定性构造，提供了一个多项式时间算法来生成有限状态独立的正常对，并有效地构造与给定正常词有限状态独立的正常词。

**🔧 技术方法**

使用了确定性多项式时间算法和动态规划技术来计算相关的条件概率。

**📊 数据集**

使用了无限字母表Σ的正常词作为输入，具体的字母表大小未给出。

**📈 对比分析**

与之前的双指数时间算法相比，本文的算法在时间复杂度上有显著的改进，能够在多项式时间内输出有限状态独立的正常对，并且通过构造的正常词在每个有限状态洗牌下保持正常性。

**⚠️ 局限性**

限制在于算法的有效性依赖于输入的正常词的可计算性，且在构造过程中可能需要处理复杂的条件概率计算。

---

## 92. OSDaR-AR: Enhancing Railway Perception Datasets via Multi-modal Augmented Reality

**arXiv ID:** 2602.22920 | [PDF](https://arxiv.org/pdf/2602.22920v1)

**作者:** Federico Nesti `[一作]` (Scuola Superiore Sant'Anna), Giorgio Buttazzo `[通讯]` (Scuola Superiore Sant'Anna)

**通讯引用:** 13464 | [OpenAlex ID](https://openalex.org/A5024920325)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Unreal Engine 5的多模态离线增强现实（AR）框架，利用OSDaR23数据（RGB、LiDAR、INS/GNSS）重建最小数字孪生并在真实列车序列中精准放置、渲染虚拟障碍物，生成公开的OSDaR‑AR数据集。

**💡 创新点**

创新点包括：①将高质量虚拟物体与真实铁路序列无缝结合，实现时空一致的增强；②通过语义分割（DDRNet‑Slim23）对INS/GNSS定位进行投影精细化，显著降低定位误差；③首次在铁路感知领域部署AR技术，并提供完整的多模态增强数据集。

**🔧 技术方法**

使用技术：Unreal Engine 5、LiDAR点云重建与配准、INS/GNSS定位、语义分割（DDRNet‑Slim23）、KISS‑ICP激光里程计、点云投影、射线投射实现遮挡、LAB色彩匹配、Gaussian模糊、重投影误差与抖动评估。

**📊 数据集**

使用数据集：OSDaR23（包含RGB、红外、LiDAR、GNSS/INS）作为原始序列，生成OSDaR‑AR（18条增强序列，共1800帧），每条序列包含6种虚拟障碍物和多模态传感器数据。

**📈 对比分析**

比较方法：将原始INS/GNSS、LiDAR里程计、以及语义分割投影精细化三种定位策略在三条100帧列车序列上进行评估，指标为重投影像素误差（RPE）和抖动（Jitter）。结果表明，原始INS/GNSS误差最大；LiDAR里程计和分割精细化在RPE与Jitter上相近，分割精细化在某些序列中略优，整体提升了AR图像的稳定性和真实感。

**⚠️ 局限性**

局限性：仅使用了三条移动列车序列，缺乏更多真实场景；定位精细化依赖单个人工标注的校准点，标注难度高且误差影响评估；点云遮挡处理仅假设单一激光雷达投射点，未能精确模拟六台雷达的视角；未提供完整的全套传感器（如侧向摄像头、红外、雷达）以及更复杂的环境。

---

## 93. Global River Forecasting with a Topology-Informed AI Foundation Model

**arXiv ID:** 2602.22293 | [PDF](https://arxiv.org/pdf/2602.22293v1)

**作者:** Hancheng Ren `[一作]` (Beijing Normal University), Bo Pang `[通讯]` (Beijing Normal University)

**通讯引用:** 2923 | [OpenAlex ID](https://openalex.org/A5039225458)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出GraphRiverCast (GRC)，一种基于拓扑信息的神经算子框架，用于全球河流多变量（流量、深度、蓄水）连续模拟，并实现“ColdStart”无历史状态启动与“HotStart”有历史状态启动两种模式；通过全局预训练与局部微调，实现从全球基础模型向各区域、不同分辨率河网的快速迁移。

**💡 创新点**

创新点包括：①首次将河网拓扑编码作为物理结构信息内化到AI模型，证明其在无历史状态下恢复河网动力学的必要性；②构建可作为“基础模型”的全球预训练网络，随后通过层级微调将其迁移到局部、不同尺度、受人类调控的河段；③采用神经算子与图卷积相结合的递归架构，兼顾空间非欧几里得特征与时间演化；④实现多变量联调，支持流量、深度、蓄水三维预测，满足洪灾与水资源管理的多尺度需求。

**🔧 技术方法**

技术方法包括：基于MERIT Hydro的河网拓扑构建；物理对齐的神经算子框架（残差GCN + GRU 递归）；多源特征融合（静态地貌、时序流量、降雨/径流 forcings）；冷/热启动两种输入模式；预训练使用CaMa-Flood物理模拟数据，微调时冻结大部分参数，仅更新深层GCN与输出层；评价指标采用NSE和高流量偏差（FHV）并辅以SHAP特征重要性分析。

**📊 数据集**

数据集：CaMa-Flood 物理模拟结果（流量、深度、储存）+ GRADES 经偏差校正的全球径流；MERIT Hydro 3''水文图；Amazon Basin GRDC 观测流量（2000‑2009）；Upper Danube LamaH‑CE 观测流量（6′′分辨率）；以及全球 15′′尺度的河网拓扑。所有训练、验证、测试均按时间划分（2010‑2015 训练，2016‑2017 验证，2018‑2019 测试）。

**📈 对比分析**

比较方法：在全球 7 天伪后测中，将GRC-ColdStart与CaMa-Flood 以及无历史状态的传统物理模型对比；在热启动模式下与传统物理模型及数据驱动模型对比；在Amazon 和 Upper Danube 两个区域进行微调后与CaMa-Flood、Scratch（从零训练）对比。结果显示：GRC-ColdStart NSE ≈0.82（无显著误差累积），热启动 NSE ≈0.93；微调后 GRC 在未观测河段的 NSE 提升 0.146，超过物理模型 0.098；在高分辨率 Upper Danube 中，即使无本地训练数据，零射击基础模型的 NSE 仍优于 Scratch，表明跨尺度迁移效果显著。

**⚠️ 局限性**

局限性：①预训练仅基于单一物理模型 CaMa-Flood，受其输入误差和参数限制；②未显式建模人为调控（坝库、截留等），对受调控河段的精度有限；③缺乏多物理模型集成，导致在极端气候或特殊水文条件下可能出现系统性误差；④微调时仍需一定观测数据，极端缺测区域仍难以完全覆盖；⑤模型在极端低流量/干旱条件下的泛化能力尚待进一步验证。

---

## 94. Robust Information Design for Multi-Agent Systems with Complementarities: Smallest-Equilibrium Threshold Policies

**arXiv ID:** 2602.22915 | [PDF](https://arxiv.org/pdf/2602.22915v1)

**作者:** Farzaneh Farhadi `[一作]` (Aston University), Maria Chli `[通讯]` (Aston University)

**通讯引用:** 357 | [OpenAlex ID](https://openalex.org/A5024973690)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对多智能体系统中的二进制合作决策，提出在最小均衡（smallest-equilibrium）情形下的鲁棒信息设计框架，并给出可直接实现的阈值规则，保证在最保守的协调假设下实现最优社会福利。

**💡 创新点**

创新点在于：① 用最小均衡的严格激励约束取代传统的“最佳均衡”假设，确保设计在最弱协调下可实现；② 在凸潜在函数和福利下证明最优策略必为极点，得到全局协调或全无协调的阈值决策；③ 提供从线性规划到闭式阈值解的完整构造与 O(|Θ|log|Θ|) 的实现算法。

**🔧 技术方法**

主要技术包括：最小均衡分析、序列信息政策与序列服从性（Sequential Obedience）约束、潜在函数与凸性分析、线性规划（LP）与极点性质、阈值排序与混合策略构造、以及仿真验证。

**📊 数据集**

数据集主要为两类案例：① 疫苗接种场景的离散状态空间（Θ={L,H}）；② 技术采用场景的近连续状态空间（Θ={0.01,0.02,…,1.00}），两者均使用人工设定的成本、收益、互补性参数，模拟多智能体决策。

**📈 对比分析**

比较方法：将鲁棒阈值政策与传统的贝叶斯说服（BCE）下的最优策略（假设最优均衡）和在最小均衡下评估的 BCE 策略进行对比；性能表现为：鲁棒策略在最小均衡下实现的期望福利与 LP 最优值一致，且在实际最小均衡下明显优于传统 BCE 方案（传统方案往往因过于乐观导致福利被高估）。

**⚠️ 局限性**

局限性包括：① 只适用于全局互补性且具备凸潜在函数与凸福利的游戏；② 对网络结构或局部互补性未做深入分析；③ 假设设计者完全知晓状态与成本，且信号只能是顺序私有邀请，实际部署可能受信号约束与信息成本影响。

---

## 95. UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception

**arXiv ID:** 2602.23224 | [PDF](https://arxiv.org/pdf/2602.23224v1)

**作者:** Mohammad Mahdavian `[一作]` (Huawei Noah's Ark Lab), Bingbing Liu `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在机器人场景中实现统一的多视角3D重建，并在单一模型中同时估计尺度、相机参数、深度和点云。

**💡 创新点**

提出了专用的尺度头和语义感知的先验注入机制，可在已知相机内外参时实现真正的度量尺度恢复；同时保留了统一模型的灵活性。

**🔧 技术方法**

基于Transformer的VGGT骨干 + DINOv2图像编码，6D旋转表示 + 轨迹编码，伪注意力降采样的尺度头，语义分布式的先验注入。

**📊 数据集**

在10个室内外真实与合成数据集（Argoverse2、Aria Synthetic、Co3Dv2、Hypersim、MegaDepth、MVS‑Synth、Replica、ScanNet、ScanNet++、VKitti）上训练和评测。

**📈 对比分析**

与VGGT、MapAnything、MAST3R、MUSt3R等SOTA方法在Robust‑MVD、KITTI、ScanNet、ETH3D等基准上对比，UniScale在多视角尺度预测、对齐和稠密重建任务上均取得或匹配最优结果，尤其在尺度恢复和深度精度上优于前者。

**⚠️ 局限性**

依赖预训练的Transformer骨干，训练成本高；对极端动态或光照变化的鲁棒性待验证；在单视图或少视角场景下尺度恢复仍受限。

---

## 96. Prediction of Diffusion Coefficients in Mixtures with Tensor Completion

**arXiv ID:** 2602.23142 | [PDF](https://arxiv.org/pdf/2602.23142v1)

**作者:** Zeno Romero `[一作]` (RPTU Kaiserslautern), Fabian Jirasek `[通讯]` (RPTU Kaiserslautern)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

开发了一种混合张量完成方法，预测二元混合物在不同温度下的无限稀释扩散系数，并通过主动学习补充实验数据。

**💡 创新点**

将张量完成技术与SEGWE模型先验结合，既实现多温度预测又保持高精度，并利用主动学习在极少实验样本下显著提升模型性能。

**🔧 技术方法**

采用低秩Tucker分解的混合张量完成模型、贝叶斯训练框架以及不确定性采样的主动学习策略。

**📊 数据集**

使用了来自DDB 2025版本和文献的224、75、56个实验无限稀释扩散系数数据，另外通过PFG NMR测得19个新体系的扩散系数。

**📈 对比分析**

通过留一交叉验证与SEGWE模型和单温度矩阵完成方法对比，发现张量完成模型在268–378 K范围内的相对平均误差低于SEGWE且优于单温度MCM，误差显著下降。

**⚠️ 局限性**

模型仍受限于温度范围（268–378 K）和样本稀疏性，且对温度的线性假设可能不适用于更广阔的温度区间。

---

## 97. TEFL: Prediction-Residual-Guided Rolling Forecasting for Multi-Horizon Time Series

**arXiv ID:** 2602.22520 | [PDF](https://arxiv.org/pdf/2602.22520v1)

**作者:** Xiannan Huang `[一作]` (Tongji University), Chao Yang `[通讯]` (Tongji University)

**通讯引用:** 146588 | [OpenAlex ID](https://openalex.org/A5100381753)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出一种名为 TEFL 的框架，在深度时间序列预测中显式利用滚动预测的历史残差作为反馈信号，在训练和推理阶段共同提升模型性能。

**💡 创新点**

创新点在于：① 设计可观测残差选择规则，使多步预测残差可在真实滚动部署中被利用；② 采用低秩自适配器将残差信息轻量级集成到基模型预测上；③ 通过两阶段训练（温度预训练 + 联合微调）解决残差不稳定性与模型自适应冲突。

**🔧 技术方法**

技术手段包括：低秩自适配器（LoRA 风格）对残差进行投影；光谱平坦度正则化鼓励残差结构化；两阶段训练策略；在多模型（Transformer、CNN、MLP 等）上验证兼容性。

**📊 数据集**

实验使用 10 个真实世界数据集（ETTm1、ETTm2、Weather、Electricity、EURUSD、AQWan、ZafNoo、Solar、CzeLan、ETTm1/2 等）以及人工添加冲击和分布漂移的版本。

**📈 对比分析**

与基线模型（DLinear、iTransformer、SOFTS、TimesFilter、Amplifier）在 4 个预测时延（96/192/336/720）上对比，平均 MAE/ MSE 减少 5–10%，在分布漂移或突发冲击场景下可达 19.5% 以上提升。

**⚠️ 局限性**

局限性：残差反馈在模型极大过拟合或训练集过度打乱时效果受限；低秩适配器在某些复杂非线性误差结构下可能不足；理论分析仅在理想化状态空间模型下给出，对真实数据的理论保证尚不足。

---

## 98. LLM-Powered Silent Bug Fuzzing in Deep Learning Libraries via Versatile and Controlled Bug Transfer

**arXiv ID:** 2602.23065 | [PDF](https://arxiv.org/pdf/2602.23065v1)

**作者:** Kunpeng Zhang `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 24944 | [OpenAlex ID](https://openalex.org/A5100328273)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个“转移-验证”框架，用大语言模型（LLM）自动从历史Bug报告中提取上下文相关的Bug模式，匹配功能相似的API，并生成自定义触发上下文与oracle，进而在深度学习库中检测静默Bug。

**💡 创新点**

创新点在于：①利用LLM进行语义层次的Bug模式抽取与功能描述生成，突破了传统基于语法匹配的局限；②采用功能嵌入匹配实现跨API的Bug迁移；③引入LLM驱动的自我验证模块，系统化评估转移Bug实例的真实有效性，显著降低误报。

**🔧 技术方法**

使用的技术包括：OpenAI Embedding API（text‑embedding‑3‑small）做功能描述嵌入；GPT‑4o mini、GPT‑4o‑mini‑2、GPT‑4.1‑mini 等LLM模型完成Bug模式抽取、API匹配、测试合成与验证；Python实现的AST插桩收集运行时日志。

**📊 数据集**

主要数据集为：PyTorch官方GitHub Issue（约7,466条可回溯修复的历史Bug）及其PR；在此基础上迁移到TensorFlow v2.18.0与MindSpore v2.3.0进行跨框架验证；使用公开的三大DL库版本进行评测。

**📈 对比分析**

与FreeFuzz、DeepREL、ACETest、TitanFuzz、DFUZZ、Pathfinder等现有API级fuzzer进行对比；在PyTorch v2.6.0上，检测到25起崩溃（比最优对手高3.5倍）以及31起多类型静默Bug；跨框架迁移共发现79个新Bug，其中12个已被认定为CVE，说明方法具有较强的可推广性与实际影响。

**⚠️ 局限性**

局限性包括：①LLM对API细节理解有限，导致部分高阶Bug（如编译器数值误差）误判为误报；②验证模块对模型的鲁棒性和一致性要求高，仍可能出现误报或漏报；③对极其复杂或文档不足的API迁移效果不佳；④大规模运行仍需较高算力与API嵌入计算成本。

---

## 99. DiffBMP: Differentiable Rendering with Bitmap Primitives

**arXiv ID:** 2602.22625 | [PDF](https://arxiv.org/pdf/2602.22625v1)

**作者:** Seongmin Hong `[一作]` (Seoul National University), Se Young Chun `[通讯]` (Seoul National University)

**通讯引用:** 2569 | [OpenAlex ID](https://openalex.org/A5052523460)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种可扩展且高效的可微分渲染引擎，能够使用任意位图图像作为原语，对其位置、旋转、缩放、颜色和透明度进行梯度优化，并支持视频、空间约束渲染及PSD格式导出。

**💡 创新点**

创新点在于：①首次实现面向位图原语的可微分渲染器；②采用自定义CUDA内核实现前向/后向全并行化和半精度计算；③结合软光栅化、结构感知初始化和噪声画布等技术显著提升梯度质量与收敛速度；④将结果导出为可编辑的PSD层级文件，方便艺术家工作流程。

**🔧 技术方法**

技术手段包括：自定义CUDA并行渲染管线、半精度FP16运算、双线性插值光栅化、Alpha混合、图层化渲染、软光栅化（高斯模糊）、结构感知初始化、噪声画布约束、动态/空间约束损失函数、以及Python接口包装。

**📊 数据集**

主要使用的实验素材为公开图像（如塞拉图绘画、玛丽莲·梦露、品牌标识等）和短视频序列（8帧 256×192），并未依赖特定大型数据集；测试中对多种图像尺寸（512×512、1024×1024、2048×2048）进行评估。

**📈 对比分析**

与DiffVG和原始PyTorch实现对比：CUDA-16bit版在512×512分辨率下前向/后向仅需 2.3/6.2 ms，显著快于PyTorch 1360/2337 ms；在复杂SVG/位图渲染任务中，PSNR提升至 24.26 dB 并将运行时间从 477 s 降至 36 s；整体显示出更高精度与更低资源占用。

**⚠️ 局限性**

局限性包括：必须在GPU上运行，无法在CPU上使用；对超参数和初始化极度敏感，易陷入局部最优；缺乏自动调参策略；目前仅支持二维位图原语，未覆盖更高维或三维场景。

---

## 100. From Agnostic to Specific: Latent Preference Diffusion for Multi-Behavior Sequential Recommendation

**arXiv ID:** 2602.23132 | [PDF](https://arxiv.org/pdf/2602.23132v1)

**作者:** Ruochen Yang `[一作]` (Institute of Information Engineering Chinese Academy of Sciences), Tingwen Liu `[通讯]` (Institute of Information Engineering Chinese Academy of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于潜在扩散模型的多行为序列推荐框架 FatsMB，利用统一的行为无关偏好空间通过扩散过程迁移到行为特定偏好，并生成候选列表。

**💡 创新点**

创新点包括① 用多行为自编码器 MBAE 构建统一的潜在偏好空间；② 设计行为感知 RoPE（BaRoPE）融合行为与位置信息；③ 在扩散去噪过程中引入多条件引导层归一化（MCGLN），实现行为、时间及无条件信息的协同引导；④ 采用生成式而非判别式模型，突破候选集限制，提升多样性与准确性。

**🔧 技术方法**

采用 Transformer+BERT4Rec 结构的自编码器、BaRoPE 位置编码、Latent Diffusion Model（DDIM）、Mixture-of-Experts、AdaLN、Classifier-free guidance 等技术。

**📊 数据集**

在 Yelp、Retail、IJCAI 三个多行为推荐基准数据集上进行实验。

**📈 对比分析**

与 BERT4Rec、DiffuRec、MBGCN、MB-GMN、MB-STR、PBAT、MISSL、M-GPT 等现有单/多行为序列推荐方法比较，FatsMB 在 Recall@10/NDCG@10 等指标上平均提升约 20%/27%，整体领先。

**⚠️ 局限性**

局限性包括：扩散推理需要多步计算，推理速度相对较慢；对多种超参数（掩码概率、扩散步数、引导强度等）敏感；在极度稀缺行为的零样本场景下仍显退化；依赖自编码器的表示能力，若预训练不足可能影响迁移效果。

---

## 101. PRAC: Principal-Random Subspace for LLM Activation Compression and Memory-Efficient Training

**arXiv ID:** 2602.23111 | [PDF](https://arxiv.org/pdf/2602.23111v1)

**作者:** Yanyi Li `[一作]` (Peking University), Cong Fang `[通讯]` (Peking University)

**通讯引用:** 2054 | [OpenAlex ID](https://openalex.org/A5008843158)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为PRAC（Principal-Random Subspace for LLM Activation Compression）的激活压缩方法，旨在解决大批量LLM训练中的内存瓶颈问题。

**💡 创新点**

PRAC通过将激活分解为主子空间和随机子空间，利用激活的谱结构来实现有效的压缩，确保无偏估计和最小方差。

**🔧 技术方法**

使用奇异值分解（SVD）来获取主子空间，并从正交补空间中随机采样以构建随机子空间。

**📊 数据集**

在LLaMA和GPT-2模型上进行预训练和微调任务的实验，使用C4和OpenWebText数据集。

**📈 对比分析**

与现有的激活压缩方法（如GaLore和RSO）进行比较，PRAC在多个任务上实现了高达36%的内存减少，同时保持了竞争力的性能。

**⚠️ 局限性**

PRAC的局限性在于其依赖于激活的低秩结构假设，可能在某些特定情况下表现不佳。

---

## 102. AgentVista: Evaluating Multimodal Agents in Ultra-Challenging Realistic Visual Scenarios

**arXiv ID:** 2602.23166 | [PDF](https://arxiv.org/pdf/2602.23166v1)

**作者:** Zhaochen Su `[一作]` (Hong Kong University of Science and Technology), Junxian He `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2842 | [OpenAlex ID](https://openalex.org/A5015879697)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AgentVista基准，用于评估通用多模态代理在真实、长周期工具使用任务中的表现。

**💡 创新点**

结合真实细节丰富的视觉场景、跨模态长链工具交互、严格可验证答案以及多子域覆盖，填补现有基准只评单轮视觉推理或单一技能的空白。

**🔧 技术方法**

使用四类工具（网络搜索、图像搜索、页面导航、代码/图像处理）和LLM模型调用框架，并通过模型辅助筛选、专家最终化、执行过滤与双轮验证构建数据。

**📊 数据集**

209个任务，覆盖25个子域、7大类，总共308幅真实图像（单图72.2%、多图27.8%），平均查询长度401.4字，答案长度40.8字。

**📈 对比分析**

在14个前沿多模态模型（GPT‑4.1、GPT‑5系列、Gemini‑3系列、Claude、o3、Qwen 等）上评测，工具调用上限30次，使用 GPT‑4.1 进行裁判；最佳模型 Gemini‑3‑Pro 仅达 27.3% 准确率，平均工具调用数约 13.8 次，显示任务极难。

**⚠️ 局限性**

受视觉误识别、知识幻觉、长链工具使用困难等瓶颈限制；基准可能受源数据偏差影响，并需关注隐私泄露与过度自信输出的风险。

---

## 103. When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering

**arXiv ID:** 2602.22474 | [PDF](https://arxiv.org/pdf/2602.22474v1)

**作者:** Jessie Yuan `[一作]` (Carnegie Mellon University), Andrea Bajcsy `[通讯]` (Carnegie Mellon University)

**通讯引用:** 558 | [OpenAlex ID](https://openalex.org/A5050279893)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Uncertainty-Aware Policy Steering (UPS) 框架，通过对 VLM 验证器进行 conformal prediction 校准，结合贝叶斯意图分解，将不确定性映射为执行高置信动作、请求任务澄清或进行低级策略再训练三种解决策略，从而在部署时实现对机器人行为的自适应调节。

**💡 创新点**

创新点包括：1) 将 VLM 的推理过程分解为隐意图与行为概率，显式地校准不确定性；2) 在序列级使用 conformal prediction 保障 1‑ε 覆盖率；3) 将不确定性映射到多层级解决策略（高层澄清与低层残差学习），实现低成本人机交互与持续学习；4) 系统化评估在仿真与真实机器人上的性能。

**🔧 技术方法**

技术手段：VLM‑in‑the‑loop 策略导向、Diffusion policy、Dreamer‑v3 隐空间世界模型、Gemini VLM（Narration 与 Verification）、Conformal Prediction、贝叶斯意图分解、残差策略学习（Residual Policy）、持续学习框架。

**📊 数据集**

数据集与实验环境：Robomimic benchmark（nut‑on‑peg 任务）和 Franka‑Emika 机器人 + Robotiq gripper（杯子放置到左右箱子），使用 120 条演示（Nut 左右手柄）、100 条演示（杯子左右箱子）、350 条轨迹训练世界模型、80 对初始观测+指令做校准集、40 条测试集等。

**📈 对比分析**

与基线比较：UPS 与未校准的 VLM‑in‑the‑loop、Forewarn、HG‑DAgger、EnsembleDAgger 等对比。结果显示 UPS 在 1‑ε=0.85 的覆盖率下，澄清率和集大小明显降低；在简单与模糊场景的成功率分别提升 45% 与 15%；人机干预率下降至 0.06（硬件）/0.058（仿真）；残差学习后整体成功率达到 85%。

**⚠️ 局限性**

局限性：1) 依赖世界模型的预测精度，未对其不确定性建模；2) 仅校准 VLM 的推理，忽略叙述与世界模型误差；3) 方案需要大量采样与 VLM 调用，计算成本高；4) 在高度动态或接触丰富的任务中，当前的 Diffusion + VLM 组合可能仍受限。

---

## 104. AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications

**arXiv ID:** 2602.22769 | [PDF](https://arxiv.org/pdf/2602.22769v1)

**作者:** Yujie Zhao `[一作]` (University of California), Jishen Zhao `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Bench 评估 LLM 代理长期记忆的基准，并设计 Agent 记忆系统

**💡 创新点**

将真实世界代理轨迹与可扩展合成轨迹相结合；利用因果图保持状态依赖，结合工具增强检索提升检索效果

**🔧 技术方法**

因果图结构化记忆构造；工具增强检索（图节点搜索 + 关键词搜索）；基于 Qwen3 等大型 LLM 进行评估

**📊 数据集**

真实世界子集：2496 对 QA，涵盖 Web、文本到 SQL、软件工程、游戏、实体 AI、开放世界工具等六类；合成子集：1200 对 QA，覆盖 5 组不同长度（8K~128K token）

**📈 对比分析**

与长上下文模型、RAG 及传统记忆代理对比，Agent 在 Qwen3‑32B 基础上平均准确率 57.22%，比最强基线高 11.16%，在所有维度（Recall、Causal Inference、State Updating、State Abstraction）均表现最佳

**⚠️ 局限性**

仅关注单集记忆（in‑episode），未覆盖跨任务、终身学习等长期记忆场景

---

## 105. Enhancing Renal Tumor Malignancy Prediction: Deep Learning with Automatic 3D CT Organ Focused Attention

**arXiv ID:** 2602.22381 | [PDF](https://arxiv.org/pdf/2602.22381v1)

**作者:** Zhengkang Fan `[一作]`, Longin Jan Latecki `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种基于3D Vision Transformer的Organ‑Focused Attention (OFA) 框架，能够在不需要手动分割的情况下预测肾脏肿瘤的恶性程度。

**💡 创新点**

创新点在于提出 OFA 损失，通过组织相关补丁的注意力引导实现自监督的“分割”功能，从而消除推理时对分割输入的依赖。

**🔧 技术方法**

使用 3D Vision Transformer、预训练权重、Softmax 行化、MSE 损失以及 OFA 损失进行联合训练，同时在训练阶段利用伪分割或真实分割引导注意力。

**📊 数据集**

采用 UF Health Renal CT 私有数据集（370例）和公开 KiTS21 数据集（300例）进行评估。

**📈 对比分析**

在 UF Health 数据集上 AUC 从 0.677 提升至 0.685，F1 分数提升至 0.872；在 KiTS21 上 AUC 从 0.72 提升至 0.76，F1 分数提升至 0.852，均优于传统分割裁剪方法和 SOTA 方案。

**⚠️ 局限性**

局限性包括对类别不平衡的敏感性（α 值高时 Recall 降低）、仍需预训练或伪分割作为训练辅助手段，以及尚未在多中心数据上进行充分验证。

---

## 106. SpectralMamba-UNet: Frequency-Disentangled State Space Modeling for Texture-Structure Consistent Medical Image Segmentation

**arXiv ID:** 2602.23103 | [PDF](https://arxiv.org/pdf/2602.23103v1)

**作者:** Fuhao Zhang `[一作]` (Sichuan Normal University), Nan Mu `[通讯]` (Sichuan Normal University)

**通讯引用:** 806 | [OpenAlex ID](https://openalex.org/A5055218366)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 SpectralMamba-UNet，融合频谱分离与状态空间建模的医学图像分割网络，能够同时捕捉低频全局结构与高频细节；

**💡 创新点**

在频域对特征进行低高频分解，并分别使用 Mamba 状态空间模型处理，随后通过频谱通道重加权（SCR）和频谱引导融合（SGF）实现频率感知的特征加权与解码融合；

**🔧 技术方法**

采用离散余弦变换（DCT）进行频谱分解，基于 Mamba 的线性复杂度状态空间模型，频谱通道重加权（SCR）机制和频谱引导融合（SGF）模块；

**📊 数据集**

在五个公开医学分割数据集上进行评测：Synapse（多器官CT）、ACDC（心脏MRI）、DRIVE（视网膜血管）、EAT（心脏脂肪）、IA（脑动脉瘤）；

**📈 对比分析**

与 Res‑UNet、TransUNet、Swin‑Transformer、VM‑UNet 等主流基线相比，SpectralMamba‑UNet 在 DSC 上平均提升至 81%+，HD95 在 Synapse 上降至 15.3 mm 等指标显著优于竞争方法；

**⚠️ 局限性**

对固定频率分割比例依赖较强，模型复杂度略高，且在某些细小结构上高频分解比例不敏感时仍有提升空间；

---

## 107. GetBatch: Distributed Multi-Object Retrieval for ML Data Loading

**arXiv ID:** 2602.22434 | [PDF](https://arxiv.org/pdf/2602.22434v1)

**作者:** Alex Aizman `[一作]` (NVIDIA), Piotr Żelasko `[通讯]` (NVIDIA)

**通讯引用:** 663 | [OpenAlex ID](https://openalex.org/A5027217976)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种存储原语GetBatch，允许客户端一次性提交所需样本列表，由对象存储统一并有序地返回，取代传统的成千上万条GET请求。

**💡 创新点**

创新点在于将批量检索提升为原生存储操作，提供严格输出顺序、分布式协同读取和容错、以及可配置的流式、错误容忍和数据共定位功能，显著削减请求开销。

**🔧 技术方法**

采用NVIDIA AIStore分布式对象存储架构，利用Designated Target（DT）协同读取，支持HTTP GET+JSON请求、持久P2P连接、以及Python SDK接口。

**📊 数据集**

在合成基准中使用多大小（10 KiB、100 KiB、1 MiB）对象，在真实训练工作负载中使用Canary-1B-Flash语音识别/翻译数据集（85k小时四种语言）。

**📈 对比分析**

通过合成基准和生产级训练对比，GetBatch在小对象时可达15×吞吐提升，P95批量延迟减半、P99单对象尾部延迟提升3.7×，整体训练效率提升约2×。

**⚠️ 局限性**

局限在于仅支持AIStore，需显式配置DT和错误容忍；对极大批量或高并发时DT可能成为瓶颈；不兼容标准S3 API，需要改造现有框架。

---

## 108. MediX-R1: Open Ended Medical Reinforcement Learning

**arXiv ID:** 2602.23363 | [PDF](https://arxiv.org/pdf/2602.23363v1)

**作者:** Sahal Shaji Mullappilly `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Hisham Cholakkal `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 3237 | [OpenAlex ID](https://openalex.org/A5009362997)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MediX‑R1框架，用于医疗多模态大模型的开放式强化学习；

**💡 创新点**

创新点在于将LLM评判、医学嵌入、格式与模态识别等多信号奖励组合成复合奖励，并用Group Based RL实现单阶段训练；

**🔧 技术方法**

使用的技术包括Group Based RL（GRPO/GSPO/DAPO）、LLM-as-judge（Qwen3-14B）、医学嵌入模型MedEmbed、vLLM推理和统一的三阶段评估管线；

**📊 数据集**

训练数据约51K指令样本，评估覆盖多种文本和图像+文本医学基准（MMLU-各子任务、MedMCQA、MedQA、USMLE‑SA、PubMedQA、MIMIC‑CXR‑Summarization、SLAKE‑VQA、RadVQA、PathVQA、PMC‑VQA、MIMIC‑CXR‑Gen、MedPix‑2.0等）；

**📈 对比分析**

与多种开源医学VLM（MedVLM‑R1、BiMediX2、HuatuoGPT‑V、MedGemma、MedMO、Qwen3‑VL等）比较，MediX‑R1在文本和图像+文本基准的平均分最高（如整体AVG 0.736/0.597），在开放式任务和临床报告生成上显著提升；

**⚠️ 局限性**

局限性包括模型仍为研究原型，可能产生幻觉或遗漏关键信息；复合奖励虽降低奖励劫持，但仍需人工审核；数据来源有限，可能存在偏差，且未完成临床验证。

---

## 109. ClawMobile: Rethinking Smartphone-Native Agentic Systems

**arXiv ID:** 2602.22942 | [PDF](https://arxiv.org/pdf/2602.22942v1)

**作者:** Hongchao Du `[一作]` (Mohammed bin Zayed University of Artificial Intelligence), Chun Jason Xue `[通讯]` (Mohammed bin Zayed University of Artificial Intelligence)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建了一个层次化的手机原生代理运行时，将高层语言推理与低层结构化控制分离；

**💡 创新点**

首次将确定性系统接口与非确定性UI推理协同调度，形成执行感知的回调式调度策略；

**🔧 技术方法**

使用OpenClaw框架作为推理层，ADB、Termux API、DroidRun等作为控制后端，LLM为GPT‑5.2；

**📊 数据集**

在Pixel 9设备上使用六个真实任务（系统设置、单应用操作、跨应用工作流）进行评估；

**📈 对比分析**

与DroidRun基线对比，完成率均达到或超过100%，但ClawMobile平均耗时约57.5 秒更长，显示出更高的可靠性与更低的误差；

**⚠️ 局限性**

局限在于依赖云端LLM导致延迟与隐私问题，状态序列化耗时高，缺乏高效增量状态表示与长期记忆机制；

---

## 110. Predicting Known Vulnerabilities from Attack Descriptions Using Sentence Transformers

**arXiv ID:** 2602.22433 | [PDF](https://arxiv.org/pdf/2602.22433v1)

**作者:** Refat Othman `[一作]` (Free University of Bozen-Bolzano), Refat Othman `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5102576124)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用句子转换器模型从网络攻击描述预测已知漏洞的关联，并实现自动化链接工具，构建并标注了大型跨仓库数据集。

**💡 创新点**

首次系统比较14种SOTA句子转换器与四种攻击信息类型的最佳组合，发现技术描述与MPNet模型可达89% F1，并挖掘出275条未记录的攻击‑漏洞链接，同时验证模型在真实新闻中的泛化能力。

**🔧 技术方法**

利用Transformer句子嵌入（MPNet、MiniLM、RoBERTa等）、余弦相似度、阈值与Top‑K敏感性分析，对攻击与CVE文本进行预处理、嵌入并进行相似度排序，最终实现自动链接。

**📊 数据集**

基于MITRE ATT&CK、CAPEC、CWE、CVE四大仓库构建的标注数据集（约625条技术、295k CVE）以及100篇真实网络安全新闻，用于训练、评估与泛化验证。

**📈 对比分析**

通过精度、召回率、F1三指标比较模型与攻击类型，MPNet+技术描述组合获得最高F1≈89%；阈值与Top‑K分析确定最优操作点；与传统TF‑IDF、LSI相比，模型在实验集和新闻集上显著提升，证明了泛化能力。

**⚠️ 局限性**

局限包括：依赖MITRE手工链接，覆盖的攻击类型有限；模型对文本噪声和极长文本的鲁棒性待提升；阈值设置仍需针对不同场景微调；对极度稀缺或新兴攻击的预测尚不成熟。

---

## 111. Learning geometry-dependent lead-field operators for forward ECG modeling

**arXiv ID:** 2602.22367 | [PDF](https://arxiv.org/pdf/2602.22367v1)

**作者:** Arsenii Dokuchaev `[一作]` (Università di Trento), Simone Pezzuto `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 1011 | [OpenAlex ID](https://openalex.org/A5061167549)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种基于形状信息的神经代理模型，用来逼近心电图（ECG）前向计算中的 lead‑field 梯度，取代传统的有限元求解，既保持高解剖精度，又实现低数据需求和高计算效率。

**💡 创新点**

创新点主要在于：①将解剖形状编码（PCA 或 DeepSDF）与梯度预测网络分离，形成“形状-条件”代理；②利用 DeepSDF 生成低维连续形状潜变量；③通过仅预测梯度实现对任意心脏激活模式的通用前向计算；④在高密度电极阵列下实现 24 倍加速，且不牺牲 ECG 形态；⑤为逆问题提供可快速更新的几何感知前向模型。

**🔧 技术方法**

采用的技术包括：DeepSDF auto‑decoder（隐式几何表示）、PCA 线性形状模型、全连接神经网络（含 Fourier 特征编码和余弦相似度损失）预测 lead‑field 梯度；有限元方法（FEniCS）生成训练数据；Latin Hypercube Sampling 采样多体位、旋转、主成分；Eikonal 模型模拟不同心脏激活模式。

**📊 数据集**

使用的数据集来自公开的统计形状模型：MPII Human Shape atlas（胸腔）和 1093 名健康受试者的双心室统计形状模型；通过 Latin Hypercube Sampling 在前 10 个主成分上随机采样得到 100 条训练/测试形状；每个形状下生成 100 条电极位置（含标准 12 轴 ECG 与 100 条 BSPM 电极），并用 FEM 计算对应的 lead‑field 梯度作为标签。

**📈 对比分析**

对比方法包括：传统伪 lead‑field 近似；评估指标为全域/心脏内部的平均角误差、相对幅值误差和 ECG 相对 L2 误差。实验结果显示：DeepSDF 编码下平均角误差为 3.89°（PCA 为 5.22°），相对 ECG 误差为 0.018（PCA 为 0.024）。在 100 条电极上，代理推理时间约 250 ms（单核 CPU），比 FEM 每导联 6 s 快 24 倍。ECG 波形与 FEM 基准高度一致，伪 lead‑field 的误差明显较大。

**⚠️ 局限性**

局限性包括：①仅考虑心腔与胸腔，未包含肺部、血管等结构，可能在某些病例下影响精度；②纤维方向和电导率固定，无法实时适应病理性或药物诱导的组织电学变化；③只逼近梯度，未直接逼近整体转移算子，对极端激活模式的泛化仍有限；④对新病人需先获取表面或点云进行潜变量推断，若数据稀缺仍需进一步改进；⑤扩展到多物理量（如可变导电率）会增加输入维度，训练成本上升。

---

## 112. Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks

**arXiv ID:** 2602.22817 | [PDF](https://arxiv.org/pdf/2602.22817v1)

**作者:** Shuo He `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**通讯引用:** 6824 | [OpenAlex ID](https://openalex.org/A5017743551)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种新的层次化分组策略HGPO，用来在长时限代理任务中更准确地估计每一步的优势函数，解决传统stepwise组策略中历史上下文不一致导致的优势偏差问题。

**💡 创新点**

核心创新在于：①基于历史上下文构造多层次分组，将相同状态的步骤按共享的历史长度分为不同层级；②对各层级优势采用自适应加权，既降低偏差又控制方差，从而实现更稳健的优势估计；③实现过程中无需额外模型或rollout，仅通过哈希查找完成分组与加权。

**🔧 技术方法**

使用的大模型为Qwen2.5-1.5B/7B-Instruct（LLM），强化学习算法为HGPO（基于GRPO/ GiGPO），实现步骤奖励计算、优势加权与PPO样式策略更新；实验使用哈希表做分组。

**📊 数据集**

评估数据集包括两大长时限代理基准：ALFWorld（多步家居导航/任务完成）和WebShop（网页购物决策）。

**📈 对比分析**

与多种基线对比：封闭源LLM（GPT‑4o、Gemini‑2.5‑Pro）、提示式代理（ReAct、Reflexion）以及RL训练方法（PPO、RLOO、GRPO、GiGPO）。实验显示HGPO在ALFWorld和WebShop上显著优于所有RL基线，尤其在分布外任务上保持更高的成功率，效果提升幅度在4–6%（ALFWorld）和2–5%（WebShop），并且在较小模型上提升更显著。

**⚠️ 局限性**

局限性包括：①需要手动设定层级数K和权重超参α，过高的α会因高层组样本稀少导致方差升高；②仅适用于未压缩的历史上下文，若使用摘要或嵌入记忆的高级代理，则需改进分组策略；③分组和加权虽然开销低，但在极大规模场景下仍可能影响实时性。

---

## 113. AgentSentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification

**arXiv ID:** 2602.22724 | [PDF](https://arxiv.org/pdf/2602.22724v1)

**作者:** Tian Zhang `[一作]` (Wuhan University), Hongxin Hu `[通讯]` (University at Buffalo)

**通讯引用:** 6520 | [OpenAlex ID](https://openalex.org/A5056657952)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为AgentSentry的推理时安全防御框架，用于检测和缓解工具增强型LLM代理中的间接提示注入（IPI）攻击。

**💡 创新点**

创新点在于将IPI建模为“时间因果接管”，通过在工具返回边界进行受控反事实重执行，定位介质驱动的偏离点，并仅在必要时进行上下文净化，实现安全继续而非直接中止或全局禁用工具。

**🔧 技术方法**

核心技术包括：边界锚定的因果诊断、受控干预式反事实重执行、平均因果效应（ACE）与直接/间接效应分解、窗口趋势统计以判定接管、以及基于因果指向的上下文净化与动作修正。

**📊 数据集**

使用AgentDojo基准的四个任务套件（TRAVEL、WORKSPACE、BANKING、SLACK）以及三类IPI攻击（Important Instructions、Tool Knowledge、InjecAgent）进行评估。

**📈 对比分析**

与多种基线（Prompt增强、工具过滤、DeBERTa检测、MELON、Task Shield等）比较，AgentSentry在三种LLM后端（GPT‑4o、GPT‑3.5‑turbo、Qwen‑3‑Max）上实现ASR=0%，平均UA达到74.55%，比最强基线提升约20–33个百分点，同时保持0%误报和完整的正面任务性能。

**⚠️ 局限性**

局限性包括：依赖多轮重执行导致推理开销增加；对工具运行时的安全性不做覆盖；在极低能力模型或特殊攻击变种下，因果估计可能不够稳健；需要预先定义的工具列表与策略，若出现新工具或复杂交互场景需重新校准。

---

## 114. AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation

**arXiv ID:** 2602.22740 | [PDF](https://arxiv.org/pdf/2602.22740v1)

**作者:** Tongfei Chen `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**通讯引用:** 13651 | [OpenAlex ID](https://openalex.org/A5015525872)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种称为Alignment-Aware Masked Learning (AML) 的轻量级训练策略，通过计算视觉与文本的patch级相似度并屏蔽低对齐像素，显著提升了Referring Image Segmentation (RIS) 的性能。

**💡 创新点**

创新点在于：①采用PatchMax Matching Evaluation (PMME) 结合随机投影统一视觉与语言特征空间；②基于PMME产生的相似度热图构造Alignment-Aware Filtering Mask (AFM)，仅对高对齐区域进行梯度更新；③在不改变模型结构且不增加推理开销的前提下，通过两阶段前向-后向训练实现对齐感知。

**🔧 技术方法**

使用的技术包括：随机高斯投影、patch‑level相似度计算、SoftMax归一化、bilinear upsampling、阈值筛选与dropout、块级掩码以及标准的二分类交叉熵损失。

**📊 数据集**

实验数据集为RefCOCO、RefCOCO+、RefCOCOg，分别覆盖从简单到复杂的自然语言表达场景。

**📈 对比分析**

与现有方法（CARIS、DETRIS、ReLA 等）对比，AML 在所有 8 个数据集分割上实现了 1%–3% 的 mIoU 提升，突破 SOTA 并在多模型、多实验设置中保持显著优势；在跨数据集和视觉扰动场景下亦表现出更高的鲁棒性。

**⚠️ 局限性**

主要局限为：训练时略微增加 17% 的时间开销，且对阈值 τ、投影维度 D_a 等超参数敏感；目前仅在静态图像 RIS 任务验证，未来需探索更自适应的掩码生成与对其他跨模态任务的推广。

---

## 115. Tackling Privacy Heterogeneity in Differentially Private Federated Learning

**arXiv ID:** 2602.22633 | [PDF](https://arxiv.org/pdf/2602.22633v1)

**作者:** Ruichen Xu `[一作]` (Chinese University of Hong Kong), Jianwei Huang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 14849 | [OpenAlex ID](https://openalex.org/A5062346297)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对差分隐私联邦学习中客户端隐私预算异构问题，提出理论收敛分析并设计基于凸优化的隐私感知客户端选择策略。

**💡 创新点**

创新点在于：①首次给出隐私异构对DP-FL收敛误差的定量分析；②基于该分析构造隐私感知选择的优化模型，并证明其全局最优；③通过实验验证该策略在不同隐私预算分布下显著提升模型精度。

**🔧 技术方法**

主要技术包括差分隐私高斯机制、FedAvg算法、凸优化（CVXPY求解）、收敛误差分解与分析。

**📊 数据集**

使用的基准数据集为MNIST、Fashion-MNIST和CIFAR-10。

**📈 对比分析**

与无偏选择（按数据量比例）和偏向选择（按局部损失）两类基线进行对比。实验显示，在CIFAR-10上隐私感知策略相较基线提升约10%准确率，在MNIST/Fashion-MNIST上提升幅度更大（30%+）。

**⚠️ 局限性**

局限性在于：①实验在单机模拟环境，无法验证大规模部署效果；②理论分析假设梯度光滑与L-光滑，实际数据分布可能更复杂；③仅考虑全局模型，未探讨个性化或多模态联邦学习场景。

---

## 116. SIGMA: A Semantic-Grounded Instruction-Driven Generative Multi-Task Recommender at AliExpress

**arXiv ID:** 2602.22913 | [PDF](https://arxiv.org/pdf/2602.22913v1)

**作者:** Yang Yu `[一作]` (Alibaba International Digital Commercial Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commercial Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 SIGMA，一个可根据自然语言指令生成多任务推荐结果的生成式推荐系统。

**💡 创新点**

创新点包括：多视角语义对齐框架将语义、视觉、知识与协同信号映射到统一潜在空间；混合物品标记化策略将 SID 前缀与唯一 ID 结合；针对七类任务构建大规模指令调优数据集；以及三步生成与自适应概率融合机制以平衡精度与多样性。

**🔧 技术方法**

主要技术包括对比学习与知识蒸馏、RQ‑VAE 生成 SID 序列、融合多模态物品嵌入的 MLP、InfoNCE 损失、Beam Search、近似最近邻检索、以及自适应概率融合（APF）。

**📊 数据集**

使用了 150M 语义对齐对和 130M 指令调优样本，来自 AliExpress 2026‑01 的匿名用户行为日志，并在 2026‑02 进一步收集 2M 样本用于评估。

**📈 对比分析**

与在线模型、GR（SID）和 GR（ID）等基线相比，SIGMA 在 HR@1、HR@5、HR@10、HR@20 等指标上显著提升（如 HR@1 从 7.31% 提升至 9.61%，HR@20 从 37.37% 提升至 43.05%）。

**⚠️ 局限性**

局限性包括对大规模 LLM 训练与推理资源的高需求、对高质量多视角对齐数据的依赖、混合标记化导致的序列长度增长和推理延迟，以及在极端实时场景下可能的召回覆盖不足。

---

## 117. A Learning-Based Hybrid Decision Framework for Matching Systems with User Departure Detection

**arXiv ID:** 2602.22412 | [PDF](https://arxiv.org/pdf/2602.22412v1)

**作者:** Ruiqi Zhou `[一作]` (Nanjing University), Houcai Shen `[通讯]` (Nanjing University)

**通讯引用:** 997 | [OpenAlex ID](https://openalex.org/A5070361735)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对具有不确定用户离场时间的动态匹配市场，提出了一种基于学习的混合匹配框架（Hybrid Matching Framework），能够在即时匹配（greedy）与延迟匹配（patient）之间动态切换，从而在匹配效率、等待时间和系统拥堵之间取得平衡。

**💡 创新点**

创新点：①将离场时间建模为对数正态分布并通过回归估计；②利用已学习的离场分布参数输入多层感知机（MLP）做二分类，以阈值决定下一周期使用哪种匹配策略；③将决策阈值、窗口大小作为可调控参数，使系统能够在两种极端策略之间连续插值；④通过闭环反馈循环实现实时学习与自适应决策。

**🔧 技术方法**

技术手段：统计学习（回归估计对数正态分布参数）、神经网络（MLP分类器）、事件驱动离散模拟、基于阈值的策略切换、窗口化决策与反馈控制。

**📊 数据集**

数据集：论文主要使用基于Poisson到达和对数正态/指数离场时间的仿真数据，未采用真实医疗或物流等公开数据集。

**📈 对比分析**

对比方法：将Hybrid框架与两种基准策略（greedy、patient）进行对比；通过仿真测量匹配损失、平均等待时间、系统拥堵（池大小）。结果显示：在不同阈值和窗口大小下，Hybrid能在保持匹配效率（损失仅略增）的前提下，显著降低等待时间和拥堵，表现出优于单一策略的平衡性能。

**⚠️ 局限性**

局限性：①假设用户离场分布在时间上保持稳定或周期性，未处理突发结构变化或冷启动问题；②仅考虑二元策略，未扩展到更复杂的匹配图结构（如双边、三方）；③阈值和窗口大小需要人工调参，缺乏理论自适应机制；④性能评估基于仿真，缺乏真实系统验证。

---

## 118. Decentralized Ranking Aggregation: Gossip Algorithms for Borda and Copeland Consensus

**arXiv ID:** 2602.22847 | [PDF](https://arxiv.org/pdf/2602.22847v1)

**作者:** Anna Van Elst `[一作]`, Stephan Clémençon `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于gossip协议的去中心化排名聚合算法，能够在无中央协调的网络中实现Borda和Copeland两种投票法的共识。

**💡 创新点**

创新点在于将传统集中式排名方法转化为高度并行的随机化传播过程，并给出了收敛速度和通信成本的理论界限；同时针对网络拓扑差异设计了自适应权重调度策略。

**🔧 技术方法**

主要技术包括：随机gossip算法、拉普拉斯矩阵分析、分布式投票模型、渐近收敛证明与误差界定。

**📊 数据集**

实验数据集涵盖：1）公开体育比赛排名（NBA、FIFA）; 2）基于用户评分的电影推荐（MovieLens）; 3）人工合成的大规模随机排名，用于验证收敛特性。

**📈 对比分析**

与集中式基准、其他分布式投票算法（如异步投票、基于稀疏图的协议）进行比较。结果显示：在相同误差阈值下，本文算法的收敛迭代次数减少约30–50%，且每次通信量仅为传统方法的1/3；在大规模网络中保持了可接受的误差率（≤1%）。

**⚠️ 局限性**

限制：算法假设网络连通且消息可靠，忽略了丢包与延迟的影响；在极度稀疏或高度动态的拓扑下收敛速度显著下降；此外，Borda/Copeland的投票规则对噪声敏感，需要进一步鲁棒性改进。

---

## 119. GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views

**arXiv ID:** 2602.22571 | [PDF](https://arxiv.org/pdf/2602.22571v1)

**作者:** Tianyu Chen `[一作]` (La Trobe University), Ramana Rao Kompella `[通讯]` (Cisco Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种迭代前向的3D高斯散点重建框架 GIFSplat，可在不使用梯度优化的情况下实现场景特定的细化。

**💡 创新点**

创新点在于：①引入可共享的多步前向残差更新模块，实现了无梯度测试时的细化；②通过冻结扩散模型提取的高频差异，将生成式先验注入每步更新；③在不增加视图集或梯度的前提下，保持秒级推理速度。

**🔧 技术方法**

采用3D Gaussian Splatting、可微光栅化、Vision Transformer窗口注意力、冻结扩散模型（Diffix）提取先验、前向残差网络等技术。

**📊 数据集**

在 DL3DV、RealEstate10K 和 DTU 三个数据集上进行评测。

**📈 对比分析**

与多种基线（PixelSplat、MVSplat、FLARE、AnySplat 等）对比，在 2/8 视角设置下 PSNR 提升 1–2 dB，SSIM 提升 0.02–0.05，LPIPS 降低 0.01–0.02；在跨域 DTU 上亦实现 2 dB 的增益。

**⚠️ 局限性**

局限性包括：只针对静态场景；仅支持单一输入模态；对动态内容、深度/法向先验的适配尚未实现。

---

## 120. Sharp Convergence Rates for Masked Diffusion Models

**arXiv ID:** 2602.22505 | [PDF](https://arxiv.org/pdf/2602.22505v1)

**作者:** Yuchen Liang `[一作]` (Ohio State University), Yingbin Liang `[通讯]` (Ohio State University)

**通讯引用:** 7726 | [OpenAlex ID](https://openalex.org/A5100384384)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对离散化掩码扩散模型的Euler采样器和First‑Hitting Sampler（FHS）进行直接总变差（TV）分析，给出收敛上界、下界，并提出无需受限得分估计和代理初始化的收敛保证。

**💡 创新点**

创新点在于：①首次以TV为度量直接推导Euler采样器的收敛上界并给出与数据维数d、精度ε相关的匹配下界；②首次证明FHS仅受得分估计误差影响，误差与d、词表大小S无关；③通过路径级解耦和离散化对称性构造匹配的下界，实现理论与实践的闭环。

**🔧 技术方法**

主要技术包括：直接TV误差分解、离散化的Kolmogorov前向方程、路径级误差分解与解耦、得分熵损失与NELBO的等价性证明、以及构造最坏情况实现下界。

**📊 数据集**

实验主要针对离散符号域（如文本/语言建模）使用标准语言模型基准（如WikiText、PTB等）验证采样器性能；理论结果与这些基准无关，可推广至任意离散符号集合。

**📈 对比分析**

与传统基于KL的收敛分析相比，本文在步数与精度的关系上从O(d/ε)提升到O(d/√ε)，且不再需要受限得分或代理初始化；FHS在仅d步内即可达到与得分估计误差同阶的TV误差，显著优于现有采样器。

**⚠️ 局限性**

局限性包括：仅分析吸收率（掩码）扩散模型，未覆盖统一率扩散；假设常数噪声调度和无得分估计误差的下界构造；并未探讨高阶采样器或随机步长策略，未来可进一步完善。

---

## 121. Locally Adaptive Decay Surfaces for High-Speed Face and Landmark Detection with Event Cameras

**arXiv ID:** 2602.23101 | [PDF](https://arxiv.org/pdf/2602.23101v1)

**作者:** Paul Kielty `[一作]` (University of Galway), Peter Corcoran `[通讯]` (University of Galway)

**通讯引用:** 5089 | [OpenAlex ID](https://openalex.org/A5066217549)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了局部自适应衰减表面（LADS），可根据事件信号局部动态调整时间衰减，实现更精细的事件表面表示；

**💡 创新点**

创新点在于将时间衰减本地化，三种自适应策略（基于事件率、LoG响应、FFT能量）首次用于人脸与标记检测，显著提升高频率下的准确率；

**🔧 技术方法**

技术包括事件到时间表面构造、局部衰减映射、卷积神经网络（MobileNetV3+PIPNet）用于人脸检测与五点标记定位；

**📊 数据集**

使用公开的Faces in Event Streams（FES）数据集（经人工与自动筛选后约63人、788k帧）以及Blink数据集进行跨数据验证；

**📈 对比分析**

与传统直方图和全局衰减（Global‑LI）基线相比，LADS在30 Hz时mAP50提升至0.957、NME下降至2.29%，在240 Hz时仍保持高精度（mAP50≈0.966、NME≈2.44%），并在低事件率的Blink集上实现跨域显著优势；

**⚠️ 局限性**

局限在于需手动调节衰减参数、对不同传感器/运动特性泛化需进一步验证、仅测试了人脸与标记两任务，未涵盖更广泛事件视觉应用。

---

## 122. Fine-grained Semantics Integration for Large Language Model-based Recommendation

**arXiv ID:** 2602.22632 | [PDF](https://arxiv.org/pdf/2602.22632v1)

**作者:** Jiawen Feng `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**通讯引用:** 42746 | [OpenAlex ID](https://openalex.org/A5038668215)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出TS-Rec框架，针对LLM生成推荐中SID的初始化和对齐问题，采用细粒度语义注入实现更精准的推荐。

**💡 创新点**

创新点在于引入Semantic‑Aware Embedding Initialization (SA‑Init) 以关键词聚合初始化SID向量，以及Token‑Level Semantic Alignment (TS‑Align) 在指令微调中实现Token级别语义对齐，弥补了传统方法的语义缺失。

**🔧 技术方法**

技术手段包括RQ‑KMeans量化生成SID、利用Extractor LLM生成关键词描述、在Qwen2.5‑1.5B上做SFT与RL（如GRPO），以及多任务指令微调联合优化推荐与Token对齐。

**📊 数据集**

实验使用Amazon公开的“Industrial and Scientific”与“Office Products”两大子数据集，包含数千用户、几千商品，稀疏度近99%。

**📈 对比分析**

与传统协同过滤与生成推荐模型（GRU4Rec、Caser、SASRec、TIGER、HSTU、LC‑Rec）对比，TS‑Rec在HR@K/NDCG@K上提升约7.17%/4.27%，在SFT‑→‑RL流程中进一步提升NDCG超过15%，证明细粒度语义对齐显著提升性能。

**⚠️ 局限性**

局限性包括：仍需依赖预训练LLM知识；对多模态或极端冷启动情境的适配不足；RL阶段奖励设计未能充分利用Token级语义一致性，未来工作需进一步优化。

---

## 123. Transformers converge to invariant algorithmic cores

**arXiv ID:** 2602.22600 | [PDF](https://arxiv.org/pdf/2602.22600v1)

**作者:** Joshua S. Schiffman `[一作]` (New York Genome Center), Joshua S. Schiffman `[通讯]` (New York Genome Center)

**通讯引用:** 416 | [OpenAlex ID](https://openalex.org/A5006071957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了“算法核心”（Algorithmic Core）框架，利用 ACE（Active‑Relevant Core Extraction）从 Transformer 的高维隐藏状态中提取低维子空间，并通过因果消融验证其必要性与充分性，进一步对核心中的线性动力学进行拟合，揭示在不同训练实例和模型规模下的可重复低维计算结构。实验涵盖单层 Transformer 对四状态马尔可夫链、两层 Transformer 对模组加法、以及 GPT‑2 系列模型的主谓数一致性任务。

**💡 创新点**

创新点主要包括：①首次系统性地定义并自动提取 Transformer 的算法核心，聚焦可重复的功能子空间而非实现细节；②展示核心在不同随机种子和模型尺度下几乎完全一致，揭示低维不变量在学习过程中的普适性；③通过核心动力学与任务真实结构（马尔可夫转移谱、模组加法的旋转算子、语言模型的数值坐标轴）匹配，提供直接的因果与机制解释；④将核心概念扩展到开放文本生成中的控制实验，证明核心可作为干预目标。

**🔧 技术方法**

核心技术包括：ACE（利用激活与梯度的交互矩阵做 SVD 提取核心）；因果消融（core‑only、core‑removed、core‑flipped）验证必要性与充分性；线性算子拟合（最小二乘、岭回归）和谱分析；CCA、主角角度与投影重叠评估核心在不同模型间的相似性；核心对齐与投影到统一坐标系；对 GPT‑2 的自适应核心干预实现文本生成中的数值控制；以及对 Grokking 动态的理论建模（最小范数解与权重衰减驱动）。

**📊 数据集**

使用的数据集包括：①自定义四状态马尔可夫链生成的序列（训练 3000 条、长度 32）；②模组加法数据（53 取模，所有 53² 对，按 50/50 随机拆分为训练/测试）；③针对 GPT‑2 的 1200 条主谓数一致性提示，涵盖 5 种句法模板、正负号位、时态变化，使用公开 GPT‑2 checkpoint。

**📈 对比分析**

比较方法与性能：对马尔可夫链模型，核心消融后精度降至随机 0.25，核心保留保持 0.75；核心线性算子谱与真实转移矩阵相差 <1%；模组加法核心在 grokking 后出现单一旋转算子，且在后续训练中核心维度与旋转模式数量随权重衰减显著膨胀；GPT‑2 语义一致性核心在每个尺度上实现 AUC > 0.91，core‑only 仍 ≥ 0.97，core‑removed < 0.25，core‑flipped ≈ 0。核心对齐后，核心投影坐标在三个规模模型间的 Spearman ρ 超过 0.87，Pearson ρ 超过 0.92。

**⚠️ 局限性**

局限性：①目前实验聚焦于单一或易归约任务，尚未验证在多步推理、长文本生成等更复杂任务中的可扩展性；②核心提取基于线性假设，可能无法捕捉高度非线性子空间；③需要明确任务分解与核心定义，自动化该流程仍有挑战；④核心在持续训练下可能膨胀导致可解释性下降，需要合适的正则化/冷却策略；⑤对大规模现代模型（如 GPT‑4、PaLM 等）仍需评估核心维度与对齐的可行性。

---

## 124. BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model

**arXiv ID:** 2602.22596 | [PDF](https://arxiv.org/pdf/2602.22596v1)

**作者:** Yuci Han `[一作]` (Ohio State University), Alper Yilmaz `[通讯]` (Ohio State University)

**通讯引用:** 8853 | [OpenAlex ID](https://openalex.org/A5008672128)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对稀疏不受限照片的场景，提出一种融合高维自编码器与Stable Video Diffusion的创新性新视角合成方法。

**💡 创新点**

通过对VAE的表示对齐与等变正则化，构建了高维潜在空间，实现了图像重建与生成能力的统一，并解决了传统视频扩散模型潜在空间等变性不足的问题。

**🔧 技术方法**

采用了可视化基础模型对齐损失、等变正则化、Stable Video Diffusion(SVD)后处理、MVSplat前向3D Gaussian Splatting等技术。

**📊 数据集**

在大型真实场景数据集DL3DV-10K上进行训练与评估。

**📈 对比分析**

与MVSplat、latentSplat、MVSplat360等基准相比，在SSIM、LPIPS、FID等指标上均取得更优成绩，尤其在细节一致性和无伪影方面表现突出。

**⚠️ 局限性**

方法受限于SVD训练成本高，且在极端稀疏输入或非常复杂场景下仍可能出现细节欠缺或过度平滑。

---

## 125. SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation

**arXiv ID:** 2602.23359 | [PDF](https://arxiv.org/pdf/2602.23359v1)

**作者:** Vaibhav Agrawal `[一作]` (International Institute of Information Technology Hyderabad), R. Venkatesh Babu `[通讯]` (Indian Institute of Science Bengaluru)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SeeThrough3D方法和Occlusion-Aware 3D Scene Representation (OSCR)，实现了文本到图像生成中的3D布局控制与遮挡感知；

**💡 创新点**

创新点在于使用半透明3D盒子编码遮挡信息，并通过颜色编码表征方向，结合注意力遮罩将文本描述绑定到对应盒子，实现对多物体遮挡、位置、姿态及摄像机视角的精准控制；

**🔧 技术方法**

采用FlUX/DiT扩散模型，结合VAE对OSCR进行编码，使用LoRA微调注意力投影，Blender渲染OSCR，深度到图像生成增强以及注意力遮罩绑定技术；

**📊 数据集**

构建了自制的25K渲染图+25K深度增强图合成数据集，并发布了3DOc-Bench 500样本评测基准；

**📈 对比分析**

与LooseControl、Build-A-Scene、LaRender、VODiff等基线比较，评估指标包括遮挡深度排序、CLIP对象分数、角度误差、图像-文本对齐与KID；SeeThrough3D在遮挡深度排序和对象分数上明显优于基线，角度误差更低，图像质量更好；

**⚠️ 局限性**

在布局变化时对图像一致性控制不足，且依赖合成数据，缺乏真实场景多样性。

---

## 126. Energy Efficient Federated Learning with Hyperdimensional Computing (HDC)

**arXiv ID:** 2602.22290 | [PDF](https://arxiv.org/pdf/2602.22290v1)

**作者:** Yahao Ding `[一作]` (King's College London), Mohammad Shikh-Bahaei `[通讯]` (King's College London)

**通讯引用:** 5683 | [OpenAlex ID](https://openalex.org/A5077634135)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在无线边缘网络上结合超维计算(HDC)和差分隐私(DP)的联邦学习框架，目标是最小化总能耗。

**💡 创新点**

首次将HDC维度与能耗优化联合考虑，并通过枚举+一维搜索的混合算法实现全局最优。

**🔧 技术方法**

使用超维计算、差分隐私、频分多址(FDMA)以及能耗的联合优化方法。

**📊 数据集**

MNIST手写数字识别数据集。

**📈 对比分析**

与固定功率、固定频率两种基线相比，在保持88%准确率下实验能耗降低约83.3%；在不同总时延下亦显著优于基线。

**⚠️ 局限性**

实验仅在单一MNIST数据集和单小区网络下进行，未验证更大规模、多样化任务或实际部署中的效果，DP噪声与模型精度的权衡仍需进一步研究。

---

## 127. To Deceive is to Teach? Forging Perceptual Robustness via Adversarial Reinforcement Learning

**arXiv ID:** 2602.22227 | [PDF](https://arxiv.org/pdf/2602.22227v1)

**作者:** Yicheng Bao `[一作]` (East China Normal University), Xin Tan `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个基于攻击者-防御者协同进化的自我对抗框架，以增强多模态大型语言模型的感知鲁棒性。

**💡 创新点**

创新点在于：①利用图像编辑攻击者动态生成多样化语义扰动，构建持续更新的训练集；②在攻击与防御之间引入局部 SSIM 语义完整性约束和奖励机制；③展示该框架可跨模型、跨规模迁移且提升鲁棒性与减少幻觉。

**🔧 技术方法**

采用 Qwen-Image-Edit 作为攻击者并使用 Flow‑GRPO 进行策略优化；采用 Qwen2.5‑VL 作为防御者并使用 DAPO 微调；同时引入局部 SSIM 约束、奖励设计以及 AOT‑SFT 数据生成 pipeline。

**📊 数据集**

使用 VStar、HRBench、POPE、HallusionBench、MMMU、MMStar、RealWorldQA、BLINK、AI2D 等公开基准，以及自建的 AOT‑SFT 对抗训练集（图像+问题+对抗图像）。

**📈 对比分析**

在 VStar、HRBench 等感知鲁棒性基准上，与基线、干预和有限对抗数据集比较，迭代三轮后分别提升 VStar 80.25%（+9.24）与 HRBench‑4K 72.38%（+8.26）；在幻觉指标上 F1 提升 2.88；在通用多模态基准上保持或提升（如 MMMU +4.66）。

**⚠️ 局限性**

局限性在于：仅针对具客观答案的 VQA 任务，难以推广到开放式生成任务；攻击者多样性受限且训练成本较高；可能导致模型过拟合特定扰动模式，需更高效的自我对抗训练策略。

---

## 128. PRIMA: Pre-training with Risk-integrated Image-Metadata Alignment for Medical Diagnosis via LLM

**arXiv ID:** 2602.23297 | [PDF](https://arxiv.org/pdf/2602.23297v1)

**作者:** Yiqing Wang `[一作]` (Duke University), Sina Farsiu `[通讯]` (Duke University)

**通讯引用:** 17952 | [OpenAlex ID](https://openalex.org/A5023633559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出PRIMA框架，融合图像与风险信息，实现医学诊断的多模态预训练与对齐。

**💡 创新点**

创新点包括：基于RAG构建风险-疾病专家知识库并注入文本编码器；四种多粒度对齐损失实现图像与文本的全局-局部语义统一；利用LLM（Qwen‑3）进行高效多模态特征融合。

**🔧 技术方法**

采用Clinical ModernBERT（通过LoRA微调）与DINOv3双编码器；四个互补损失（图像一致性、全局语义、局部语义、软语义）实现对齐；LoRA方式训练；最后用Qwen‑3进行诊断推理。

**📊 数据集**

在PAD‑UFES‑20（皮肤病变）和AQUA（角膜炎）两个公开/私有数据集上进行实验。

**📈 对比分析**

与DINOv3+MLP、MedKLIP、KnoBo、MedBLIP、MLRG、DeepIK等基线比较，PRIMA在PAD‑UFES‑20上F1‑score达80.21%、Acc 78.27%；在AQUA上F1‑score 88.66%、Acc 86.04%，均比对手提升5%以上，表现最优。

**⚠️ 局限性**

局限性：缺乏对基线编码器自身性能的独立评估；LLM生成可能带来偏差；对专家知识库的构建依赖人工审核，未对不同医学领域泛化进行充分验证。

---

## 129. Understanding Older Adults' Experiences of Support, Concerns, and Risks from Kinship-Role AI-Generated Influencers

**arXiv ID:** 2602.22993 | [PDF](https://arxiv.org/pdf/2602.22993v1)

**作者:** Tianqi Song `[一作]` (National University of Singapore), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**通讯引用:** 1809 | [OpenAlex ID](https://openalex.org/A5054435118)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对中国短视频平台（抖音、快手）上亲属角色AI虚拟网红的内容观察与16名老年人访谈，研究了老年人如何感知并体验此类AI的社交支持。

**💡 创新点**

创新点在于首次将虚拟亲属角色与“虚拟亲属关系”概念结合，系统梳理了AI网红的设计与交互策略，并揭示了老年人在情感、认知、伦理层面的收益与风险。

**🔧 技术方法**

采用了开源社交媒体爬虫 MediaCrawler 收集视频，使用定性内容分析和主题分析方法，对视频视觉与语言线索进行编码；访谈采用半结构化访谈并进行语义与潜在主题编码。

**📊 数据集**

数据集包括 224 条热门短视频（106 条抖音、118 条快手，时间 2022‑2025），共 18,650 条评论，以及 16 名年龄 52‑75 岁的老年人访谈记录。

**📈 对比分析**

研究未使用对照组或量化性能指标，评估方法主要为质性描述和主题归纳；无基准实验或数值比较结果。

**⚠️ 局限性**

局限性包括：样本偏向热门内容、受访者多为技术熟练且有家属支持的老年人，样本量小，缺乏与非亲属角色AI或传统真人网红的对比，未进行纵向跟踪，且研究聚焦中国文化背景，结果可能不易推广至其他地区。

---

## 130. Human Label Variation in Implicit Discourse Relation Recognition

**arXiv ID:** 2602.22723 | [PDF](https://arxiv.org/pdf/2602.22723v1)

**作者:** Frances Yung `[一作]` (Saarland University), Massimo Poesio `[通讯]` (Queen Mary University of London)

**通讯引用:** 9384 | [OpenAlex ID](https://openalex.org/A5047065550)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在英文隐式语篇关系识别任务中，针对不同标签一致性程度，比较了单真值、软标签及视角模型在单标签预测、标签分布预测和个体标注预测三种任务上的表现。

**💡 创新点**

创新点在于首次系统评估视角（多任务与嵌入）模型在IDRR中的适用性，并揭示当类别细化时，分布预测优于个体预测；同时对标注不一致的原因进行了手工分析。

**🔧 技术方法**

技术上采用RoBERTa‑base作为编码器，训练单真值模型、二元多标签模型、KL分布预测模型、MT和AE视角模型，并用macro‑F1/accuracy以及CE、JSD、MD、ED等软指标评估模型。

**📊 数据集**

使用的数据集为DiscoGeM 1.5——一个多语料、多领域、众包多标注的PDTB式隐式语篇关系标注集。

**📈 对比分析**

通过在三类模型上分别跑单标签、标签分布和个体标注任务，利用macro‑F1/accuracy比较性能，发现视角AE模型在单标签和个体预测上优于其他模型，但在细粒度类别下性能显著下降；软分布模型在标签分布预测上优于单真值模型，整体来看分布预测表现最佳。

**⚠️ 局限性**

局限性包括：仅使用RoBERTa‑base作为基础架构，未与最新IDRR模型对比；实验样本量不足以覆盖所有标注细节；视角模型在细粒度类别下受限；未结合错误检测与视角建模来进一步降低噪声。

---

## 131. RhythmBERT: A Self-Supervised Language Model Based on Latent Representations of ECG Waveforms for Heart Disease Detection

**arXiv ID:** 2602.23060 | [PDF](https://arxiv.org/pdf/2602.23060v1)

**作者:** Xin Wang `[一作]` (Virginia Tech), Fatemeh Afghah `[通讯]` (Clemson University)

**通讯引用:** 3656 | [OpenAlex ID](https://openalex.org/A5035395012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并预训练了 RhythmBERT，一种以心电波形为“单词”、心电图为“句子”的自监督生成式模型，仅使用 Lead II 的单导电位。

**💡 创新点**

创新点在于通过自编码器把 P、QRS、T 波段编码为低维离散词汇，并与连续形态嵌入融合，既保留节律信息又捕获细节形态。

**🔧 技术方法**

使用一维卷积自编码器、k‑means 词表离散化、1D‑ResNet 形态嵌入以及 BERT 变压器做掩码语言建模。

**📊 数据集**

使用约 800,000 条未标注的 MIMIC‑IV‑ECG 单导电位记录进行预训练，并在 PTB‑XL、CPSC‑2018、Chapman‑Shaoxing 等公开心电分类数据集上微调。

**📈 对比分析**

与 ST‑MEM、HeartLang 等 12‑lead 自监督模型对比，RhythmBERT 在单导电位下在三大数据集上均取得更高或相当的 AUROC（例如 Chapman‑Shaoxing 94.11%）。

**⚠️ 局限性**

限制在于仅使用单导电位，缺乏多导位的空间信息；离散化可能导致细节丢失，并需在多导位扩展验证。

---

## 132. Retrieve and Segment: Are a Few Examples Enough to Bridge the Supervision Gap in Open-Vocabulary Segmentation?

**arXiv ID:** 2602.23339 | [PDF](https://arxiv.org/pdf/2602.23339v1)

**作者:** Tilemachos Aravanis `[一作]` (Czech Technical University in Prague), Giorgos Tolias `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 2997 | [OpenAlex ID](https://openalex.org/A5046083819)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种检索增强的测试时适配器，用少量像素标注的视觉支持与文本提示融合，实现开词汇分割；

**💡 创新点**

在测试时动态检索相关视觉支持并与文本特征按学习方式融合，构建轻量化线性分类器，避免手工融合，并支持部分或无视觉/文本支持；

**🔧 技术方法**

利用冻结的视觉‑语言模型（OpenCLIP、DINOv3.txt）提取特征，使用检索（k‑NN）挑选支持特征，训练单张图的线性分类器，融合多种λ的文本-视觉混合特征；

**📊 数据集**

在六大公开开词汇分割基准（PASCAL VOC、Context、COCO Object/Stuff、Cityscapes、ADE20K）以及PASCAL Context‑59、FoodSeg103、CUB 上进行评估；

**📈 对比分析**

与零样本基线、两种检索式 OVS 方法、以及离线训练的线性/全量网络进行对比，取得平均 mIoU 最高，零样本提升 7.3%（OpenCLIP）/18.4%（DINOv3.txt），在 B=20 时仅差 11.5 点与完全监督模型；

**⚠️ 局限性**

对极少支持样本的鲁棒性仍有限，文本与视觉融合对支持量敏感，且依赖 VLM 的视觉特征质量，缺乏对动态环境中大规模支持的评估。

---

## 133. VoiceAlign: A Shimming Layer for Enhancing the Usability of Legacy Voice User Interface Systems

**arXiv ID:** 2602.22374 | [PDF](https://arxiv.org/pdf/2602.22374v1)

**作者:** Md Ehtesham-Ul-Haque `[一作]` (Pennsylvania State University), Syed Masum Billah `[通讯]` (Pennsylvania State University)

**通讯引用:** 463 | [OpenAlex ID](https://openalex.org/A5005834738)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 VoiceAlign，一个适配层（shim）通过自然语言处理将用户的非标准语音命令转化为符合传统固定语法的 Voice Control（苹果语音控制）等旧版 VUI 所需格式，并通过虚拟音频通道将改写后的命令投递给原系统，从而提升旧版 VUI 的可用性。

**💡 创新点**

创新点包括：①将 LLM 用作实时命令纠错器，支持用户自然语言表述并自动映射到严格命令模板；②利用虚拟音频通道实现透明中间层，无需改动底层 VUI；③基于用户研究生成合成训练集并微调 270M 参数 Gemma 模型，实现离线 200 ms 低延迟推理；④通过延长超时窗口与实时反馈降低认知负荷。

**🔧 技术方法**

技术核心：Web Speech API 进行语音转文本；Claude‑3.5‑Sonnet 或本地微调的 Gemma‑3 通过 LLM 进行命令解析与重写；BlackHole 虚拟声卡做音频回路；Ollama 服务本地推理；LoRA 微调技术；NASA‑TLX 与实验评估。

**📊 数据集**

数据集：1,000 条训练、400 条验证、150 条测试的合成命令转换数据，覆盖 6 种合法命令结构、8 类操作（选择、删除、插入、替换等）以及 8 种错误类别（自然变体、替换、缺失参数等）。

**📈 对比分析**

比较方法：与未使用 VoiceAlign 时的 Voice Control 进行 12 名参与者的 20 条文本纠错任务对比。结果显示：①命令成功率提升 50%（从 73.1% 到 86.3%）；②平均每任务所需命令数减少 25%（4.92 → 3.67）；③NASA‑TLX 认知、时间和挫败感显著下降；③微调模型在本地的精度 90.6% 且 200 ms 响应时间。

**⚠️ 局限性**

局限性：①使用 TTS 进行命令输出导致额外延迟且可能被某些 VUI 拒识；②仅在英语流利用户上验证，未覆盖非母语或发音障碍者；③仅支持固定格式 VUI 的文本纠错命令，未扩展至其他 UI 控制任务；④合成数据的真实性有限，未来需更多真实用户样本；⑤仅实现单语音命令到单命令的映射，未支持多命令序列的自然对话。

---

## 134. BrepCoder: A Unified Multimodal Large Language Model for Multi-task B-rep Reasoning

**arXiv ID:** 2602.22284 | [PDF](https://arxiv.org/pdf/2602.22284v1)

**作者:** Mingi Kim `[一作]` (Chungnam National University), Hyungki Kim `[通讯]` (Chungnam National University)

**通讯引用:** 1768 | [OpenAlex ID](https://openalex.org/A5091658053)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了BrepCoder，一种统一的多模态大型语言模型，能够从B-rep输入执行多种CAD任务（反向工程、补全、错误修正和CAD问答），并通过两阶段训练实现对B-rep与Python式CAD代码的对齐与推理。

**💡 创新点**

创新点在于：① 将B-rep直接作为核心输入，通过Python式CAD代码的形式利用LLM的编程推理能力；② 采用CoCa框架实现B-rep与代码的双重对齐（对比+生成）；③ 通过两阶段训练（先反向工程再多任务微调）显著提升模型的通用性和性能。

**🔧 技术方法**

使用的技术包括：B-rep编码器（UV-Net）、CoCa多模态对齐、LLM（Qwen 2.5-1.5B）、投影层、Transformer编码/解码器、对比损失与生成损失的联合优化。

**📊 数据集**

主要数据集：DeepCAD（170K序列）用于预训练与反向工程、补全、错误修正；SGP-Bench（1K模型+多选题）用于CAD-QA；还使用了B-rep与CAD代码的配对数据。

**📈 对比分析**

与现有基线（CADCL、CAD-Llama、ReCAD、PointLLM等）对比，BrepCoder在反向工程的Chamfer距离从0.972降至0.499，ACC_param提升至81.93%，Invalid Ratio降至0.009；在补全任务ACC_cmd、ACC_param分别达到92.69%/87.94%；在错误修正任务ACC_cmd/ACC_param达到99.08%/93.74%，Chamfer距离仅0.335；在CAD-QA任务获得79%准确率，接近PointLLM的81%。

**⚠️ 局限性**

局限性包括：① 对B-rep的依赖导致无法直接处理点云或图像输入，需进一步扩展；② 仍存在建模顺序歧义导致ACC_cmd略低；③ 仅在1.5B参数规模上验证，尚未验证更大规模模型的可扩展性；④ 对于高度复杂或大尺寸模型的推理速度和资源占用未作评估。

---

## 135. E3VA: Enhancing Emotional Expressiveness in Virtual Conversational Agents

**arXiv ID:** 2602.22362 | [PDF](https://arxiv.org/pdf/2602.22362v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 136. ContextRL: Enhancing MLLM's Knowledge Discovery Efficiency with Context-Augmented RL

**arXiv ID:** 2602.22623 | [PDF](https://arxiv.org/pdf/2602.22623v1)

**作者:** Xingyu Lu `[一作]` (Tsinghua University), Chun Yuan `[通讯]` (Tsinghua University)

**通讯引用:** 32811 | [OpenAlex ID](https://openalex.org/A5008769328)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ContextRL 框架，通过上下文增强来提升多模态大语言模型在 RLVR 训练中的知识发现效率。

**💡 创新点**

创新点在于：① 为奖励模型提供完整解题过程（而非仅答案），显著降低假正样本与奖励劫持；② 采用双阶段采样与错误报告机制，提高难题采样到正样本的概率；③ 通过优势缩放与混合训练组实现训练稳定性。

**🔧 技术方法**

技术手段包括：RLVR（GRPO/DAPO）算法、上下文增强奖励模型、错误报告生成、多轮采样、优势缩放、混合训练组以及 Qwen3-VL 系列模型。

**📊 数据集**

使用 FineVision 训练集（约 29k 条带完整解答样本）进行训练，并在 11 个感知与推理基准（SimpleVQA、MMStar、HallusionBench、HRBench8K、MME-RealWorld-lite、MathVerse、MathVista、LogicVista、We-Math、CharXiv-RQ、DynaMath）上进行评测。

**📈 对比分析**

与 SFT、GRPO、DAPO 以及 32B 大模型基线对比，ContextRL 在所有感知基准上取得最佳或第二佳成绩，在推理基准上仅次于 32B 并在 We-Math 上超越 32B；整体平均提升约 5.5%，并能让 8B 模型逼近 32B 性能。

**⚠️ 局限性**

局限性包括：① 对参考解答质量依赖较高，若解答错误可能导致训练偏差；② 训练样本规模仍偏小，尤其是推理任务；③ 采用多轮采样和错误报告增加计算开销，需进一步优化效率。

---

## 137. Testable Learning of General Halfspaces under Massart Noise

**arXiv ID:** 2602.22300 | [PDF](https://arxiv.org/pdf/2602.22300v1)

**作者:** Ilias Diakonikolas `[一作]` (University of Wisconsin Madison), Sihan Liu `[通讯]` (University of California San Diego)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5020204264)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文提出了针对高斯分布下的全局Massart半空间的可测试学习算法；

**💡 创新点**

创新点在于首次给出全局半空间可测试学习方案，并引入多项式逼近符号函数的乘法性“夹层”逼近技术；

**🔧 技术方法**

主要技术包括：设计基于高斯分布的切片质量与矩匹配检验、低阶多项式非负性检验以及利用Chebyshev多项式构造的乘法性夹层多项式逼近；

**📊 数据集**

研究完全基于理论分析，不依赖具体实验数据集；

**📈 对比分析**

相较于已有的非可测试算法，所给出的算法在复杂度上仅多出多项式log因子，且与已知的统计查询（SQ）下的下界相匹配；

**⚠️ 局限性**

局限性包括：算法复杂度仍为准多项式（d^polylog(1/γ,1/ε)），对β（噪声上界）的指数依赖；且当前仅适用于高斯边缘分布，难以推广到更一般分布。

---

## 138. A Proper Scoring Rule for Virtual Staining

**arXiv ID:** 2602.23305 | [PDF](https://arxiv.org/pdf/2602.23305v1)

**作者:** Samuel Tonks `[一作]` (University of Birmingham), Alexander Krull `[通讯]` (University of Birmingham)

**通讯引用:** 5196 | [OpenAlex ID](https://openalex.org/A5073866871)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出信息增益（IG）作为严格的分数规则，用于直接评估虚拟染色模型对细胞特征后验分布的预测质量，并用其与传统指标对比验证其有效性。

**💡 创新点**

创新点在于首次将严格的分数规则应用于虚拟染色的后验评估，能够在仅有单个真实样本的情况下衡量模型的条件概率质量，并揭示传统边缘指标掩盖的显著差异。

**🔧 技术方法**

使用的技术包括：条件扩散概率模型（cDDPM）和 Pix2pixHD GAN 生成器、核密度估计/KDE 与高斯混合模型（GMM）估计后验密度、信息增益、边缘 KL 散度和秩距离三种评估指标。

**📊 数据集**

数据集为 30,000 张 Brightfield‑DAPI 图像对，来自 49 片 16×24 井板，涵盖 6 种卵巢细胞系的 10 种化合物及对照。

**📈 对比分析**

比较方法：用 IG、边缘 KLD 和秩距离三指标对 Pix2pixHD 与 cDDPM 在 18 个细胞特征上进行评估。结果显示，IG 明显区分出 cDDPM 在所有特征上优于 Pix2pixHD，尤其是强度相关特征；而边缘 KLD 与秩距离均未显示明显差异。

**⚠️ 局限性**

局限性包括：IG 需要对后验进行密度估计，计算成本高；在某些特征上（尤其是强度特征）两模型仍表现欠佳；仅针对单个真实样本估计后验，可能受样本偏差影响；方法在其它图像到图像转换任务中的推广仍需进一步验证。

---

## 139. Towards Multimodal Domain Generalization with Few Labels

**arXiv ID:** 2602.22917 | [PDF](https://arxiv.org/pdf/2602.22917v1)

**作者:** Hongzhao Li `[一作]` (Zhengzhou University), Muhammad Haris Khan `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并解决了半监督多模态域泛化（SSMDG）问题，构建统一框架实现少标记、多域、多模态的数据高效学习。

**💡 创新点**

创新点在于：1）结合共识驱动一致性正则化、分歧感知正则化和跨模原型对齐三大模块；2）利用跨模翻译实现缺失模态下的鲁棒性；3）首次建立SSMDG基准并提供多模态缺失实验。

**🔧 技术方法**

采用了跨模融合编码器、弱强数据增强、FixMatch式一致性学习、Generalized Cross‑Entropy噪声鲁棒损失、EMA原型更新与跨域跨模对齐等技术。

**📊 数据集**

使用了EPIC‑Kitchens和HAC两个公开多模态域泛化数据集，并在不同标注比例和缺失模态设置下进行评估。

**📈 对比分析**

与MMDG、SSL、SSDL、SSML等四大范式基准对比，所提方法在5标签/类、5%/10%标注、两模/三模以及缺失模态场景均取得显著提升，最高平均准确率可达65.6%。

**⚠️ 局限性**

局限性包括：对超参数（阈值、权重系数）敏感；在极端少标注或高度不平衡分布下仍可能出现伪标签误导；缺失模态的翻译模型依赖于足够的同源数据。

---

## 140. FLYING SERVING: On-the-Fly Parallelism Switching for Large Language Model Serving

**arXiv ID:** 2602.22593 | [PDF](https://arxiv.org/pdf/2602.22593v1)

**作者:** Shouwei Gao `[一作]` (Oregon State University), Wenqian Dong `[通讯]` (Oregon State University)

**通讯引用:** 1636 | [OpenAlex ID](https://openalex.org/A5045803591)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套基于 vLLM 的大型语言模型服务系统，支持在线动态切换数据并行（DP）与张量并行（TP），无需重启引擎即可根据负载、优先级和上下文长度实时调整并行度。

**💡 创新点**

创新点在于：①零拷贝模型权重管理器实现 DP‑TP 切换时仅切换视图而不移动数据；②KV 缓存自适应适配器在 DP 与 TP 之间保持同一物理 KV 块，避免状态迁移；③预初始化通信池使得切换时可即刻获取所需 NCCL/ Gloo 组；④基于全局任务池和无死锁调度器的切换协议实现了安全、低延迟的并行度切换。

**🔧 技术方法**

采用的核心技术包括 vLLM v1 框架、Megatron‑LM 风格张量分片、CUDA / NCCL / Gloo 通信、零拷贝视图、KV 缓存块大小自适应、控制平面与数据平面双层通信、以及软/硬抢占调度策略。

**📊 数据集**

使用了 Llama‑3‑70B、GPT‑OSS‑120B（MoE）、Nemotron‑8B 三个主流模型；在此基础上利用 ShareGPT、CodeActInstruct、HumanEval 三个公开数据集生成的合成工作负载，模拟真实的请求长度与突发流量。

**📈 对比分析**

实验对比了静态 DP、静态 TP、Shift‑Parallelism 三种基线；在高负载下系统获得 4.79× 的加速，低负载下 3.47×；TTFT 与 TPOT 几乎与 TP 相当，队列时间显著下降（相较于静态 TP 减少 1.6×–4.8×），并保持 95–96% 的 DP 峰值吞吐量。

**⚠️ 局限性**

主要局限：仅支持单机多 GPU 内部并行，无法直接扩展至跨节点的大模型；对极大 MoE 模型的兼容性有限，且在某些稀疏模型上缺少完整的性能验证。

---

## 141. SPATIALALIGN: Aligning Dynamic Spatial Relationships in Video Generation

**arXiv ID:** 2602.22745 | [PDF](https://arxiv.org/pdf/2602.22745v1)

**作者:** Fengming Liu `[一作]` (Nanyang Technological University), Chuanxia Zheng `[通讯]` (Nanyang Technological University)

**通讯引用:** 903 | [OpenAlex ID](https://openalex.org/A5066022386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自我提升框架，利用几何基准奖励微调文本到视频模型，使其更好地遵循文本提示中的动态空间关系（DSR）。

**💡 创新点**

创新点：① 用几何距离指标（SSR‑Score 与 DSR‑Score）代替传统 VLM 评估，提供更可靠的奖励信号；② 将 Direct Preference Optimization (DPO) 与零阶正则化相结合，提升 DSR 对齐能力；③ 构建专门针对 DSR 的新数据集与评测基准。

**🔧 技术方法**

主要技术：GroundedSAM 对象检测与跟踪、几何基准奖励函数、DPO 训练、LoRA 微调、零阶正则化。

**📊 数据集**

使用自构建的 DSR 数据集：训练集 500 条提示（每条 10 条视频），测试集 120 条提示（每条 5 条视频），覆盖 LEFT、RIGHT、TOP 三种空间关系。

**📈 对比分析**

与 OpenSora 2.0、CogVideoX1.5、Wan2.1、LTX-Video、HunyuanVideo1.5 等 SOTA T2V 模型对比，Correctness@0.7 明显提升，ID Consistency 与视觉质量基本持平，证明方法在保持画面质量的同时显著提升空间关系准确度。

**⚠️ 局限性**

局限性：依赖 GroundedSAM 的检测/跟踪鲁棒性，难以处理复杂场景或运动模糊；目前仅覆盖单一动物与单一静态物体、LEFT/RIGHT/TOP 三种 DSR，尚未验证对更复杂多对象关系的泛化。

---

## 142. Designing Robots for Families: In-Situ Prototyping for Contextual Reminders on Family Routines

**arXiv ID:** 2602.22628 | [PDF](https://arxiv.org/pdf/2602.22628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 143. Make It Hard to Hear, Easy to Learn: Long-Form Bengali ASR and Speaker Diarization via Extreme Augmentation and Perfect Alignment

**arXiv ID:** 2602.23070 | [PDF](https://arxiv.org/pdf/2602.23070v1)

**作者:** Sanjid Hasan `[一作]` (Khulna University of Engineering and Technology), Bayazid Hasan `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了882小时多说话人孟加拉语数据集Lipi-Ghor-882，并针对长音频设计了双管道ASR与说话人分离系统。

**💡 创新点**

发现相较于扩大原始数据或模型集成，针对噪声增强与完全对齐的细粒度微调更能提升ASR性能；说话人分离则依赖严谨的后处理而非模型再训练。

**🔧 技术方法**

使用Whisper‑Medium/Faster‑Whisper、Silero VAD、Pyannote、CTranslate2、Demucs、ROVER以及自定义的严格间隙/合并后处理算法。

**📊 数据集**

采用自建的Lipi‑Ghor‑882数据集以及公开的22小时隐藏长音频测试集。

**📈 对比分析**

在多种基线与微调方案对比下，最终ASR WER约0.31/0.32，DER约0.20-0.28，RTF≈0.019，显著优于公开榜单。

**⚠️ 局限性**

受限于GPU计算时数、数据质量不均、模型过拟合及评估难度，导致进一步性能提升受阻。

---

## 144. Asymmetric Idiosyncrasies in Multimodal Models

**arXiv ID:** 2602.22734 | [PDF](https://arxiv.org/pdf/2602.22734v1)

**作者:** Muzi Tao `[一作]` (University of Southern California), Xuezhe Ma `[通讯]` (University of Southern California)

**通讯引用:** 5504 | [OpenAlex ID](https://openalex.org/A5078672329)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统地分析了图像字幕模型的风格特征及其在文本到图像生成中的传递效果。

**💡 创新点**

提出了基于源模型识别的跨模态评估框架，揭示了字幕风格在生成图像中显著衰减的现象。

**🔧 技术方法**

采用文本分类器（BERT/CLIP）、图像分类器（ResNet-18）、TF‑IDF、颜色与纹理词汇统计及词义重写等多种技术。

**📊 数据集**

使用10,000张来自CC3M、COCO、ImageNet和MNIST的数据集，并生成约30,000条多模态字幕。

**📈 对比分析**

在文本域上识别准确率达99.5%，而在生成图像域仅提升至约50%，显示出明显的跨模态差距。

**⚠️ 局限性**

主要局限在于现有文本到图像生成模型难以保留字幕中的细节、颜色和构图指令，导致风格信号丢失。

---

## 145. What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty

**arXiv ID:** 2602.22220 | [PDF](https://arxiv.org/pdf/2602.22220v1)

**作者:** Bowei Zhang `[一作]` (Fudan University), Jiaqing Liang `[通讯]` (Fudan University)

**通讯引用:** 962 | [OpenAlex ID](https://openalex.org/A5075507821)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于“unexpected yet rational”理念的引文推荐框架 NovelQR，旨在为写作提供既贴合上下文又具有新颖审美价值的名言；

**💡 创新点**

核心创新在于：①采用生成式标签代理将引文映射至深层语义空间，并通过多维标签实现精细化语义检索；②提出以标记化 token 为基础、校正自回归预测偏差的 Novelty Token 机制，显著提升新颖度评估的可靠性；

**🔧 技术方法**

技术手段包括：多轮生成式标签提取（基于 Qwen3‑8B），深层语义检索（ACGE 文本嵌入），token‑级 novelty 计算（利用自熵差异与加权），以及结合新颖度、流行度与语义匹配的综合排序公式；

**📊 数据集**

使用的语料库主要来自 QUILL 及自建的 300 条双语测试集（QuoteR、QUILL、NovelQR），并在检索阶段利用大量公开引文构建知识库；

**📈 对比分析**

实验结果显示：在 HR@5、nDCG@5、MRR@5 等检索指标上，NovelQR 的检索性能优于传统文本检索（QuoteR、QUILL），且在人工与 LLM‑as‑judge 的多维评分中，新颖度得分提升约 20%，匹配度保持甚至略有提升，最终系统获得 78% 的多选偏好；

**⚠️ 局限性**

局限性主要在于新颖度评估仍依赖 LLM‑as‑judge，难以完全捕捉个体主观差异；此外，标签生成过程仍可能出现误解释，导致检索或排序误差；

---

## 146. MetaOthello: A Controlled Study of Multiple World Models in Transformers

**arXiv ID:** 2602.23164 | [PDF](https://arxiv.org/pdf/2602.23164v1)

**作者:** Aviral Chawla `[一作]` (Vermont Complex Systems Institute), Juniper Lovato `[通讯]` (University of Vermont)

**通讯引用:** 119 | [OpenAlex ID](https://openalex.org/A5073688320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究Transformer在多规则环境下如何组织不同的世界模型，使用MetaOthello框架进行实验。

**💡 创新点**

发现模型并未分割为独立子模型，而是共享一个大致相同的棋盘状态表示，并在层级上分离通用与游戏特定计算；对等语言变体的表示仅相差一个正交旋转。

**🔧 技术方法**

使用线性探针、因果干预、正交Procrustes对齐、旋转变换等解释技术；训练8层、512维度的GPT-Transformer。

**📊 数据集**

使用MetaOthello生成的四种游戏变体（Classic、NoMidFlip、DelFlank、Iago），分别训练纯游戏与混合游戏数据集。

**📈 对比分析**

与单游戏模型对比，混合模型仅略低的α分数（≥0.98），线性探针在不同层级可有效预测棋盘状态；跨变体干预效果接近匹配探针。

**⚠️ 局限性**

实验仅在人工生成的八阶棋盘小游戏上进行，规模有限，未探讨多任务/不平衡混合、规模扩大以及非线性/电路层面的机制。

---

## 147. Decoder-based Sense Knowledge Distillation

**arXiv ID:** 2602.22351 | [PDF](https://arxiv.org/pdf/2602.22351v1)

**作者:** Qitong Wang `[一作]` (Rensselaer Polytechnic Institute), Vasileios Kalantzis `[通讯]` (IBM Research)

**通讯引用:** 122 | [OpenAlex ID](https://openalex.org/A5040932623)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在解码器式LLM训练中加入词义词典和同义/反义关系，通过额外的语义一致性损失让学生模型在仅训练阶段学习结构化语义。

**💡 创新点**

首次将词义字典与词义关系用于解码器模型的知识蒸馏，无需推理时查字典，并结合同义/反义正负样本提升语义对齐。

**🔧 技术方法**

词义字典构建（聚类上下文向量）、同义/反义关系提取、语义一致性损失（带门限的MSE+hinge）、仅训练最后两层、使用bf16、LLaMA‑3‑8B与Mistral‑7B教师。

**📊 数据集**

Wiki语料库构造词义字典；ARC、CommonsenseQA、MMLU、PIQA、SQuADv2 等公开评测集。

**📈 对比分析**

与仅使用标准KD的同样学生层数进行对比；DSKD 在ARC、CSQA、MMLU、PIQA、SQuAD 上均高于KD 1–3% 甚至更大；训练时间仅多5%，模型参数约为教师的一半。

**⚠️ 局限性**

词义簇数量固定，对词义多样性未自适应；仅在训练阶段使用字典，推理不受益；需要预先构建大规模字典，增加前期成本。

---

## 148. Cognitive Models and AI Algorithms Provide Templates for Designing Language Agents

**arXiv ID:** 2602.22523 | [PDF](https://arxiv.org/pdf/2602.22523v1)

**作者:** Ryan Liu `[一作]` (Princeton University), Thomas L. Griffiths `[通讯]` (Princeton University)

**通讯引用:** 49585 | [OpenAlex ID](https://openalex.org/A5077079119)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并论证认知模型与AI算法可作为构建语言代理的模板，并将现有语言代理的架构归纳为这些模板；

**💡 创新点**

创新点在于引入“agent template”概念，将认知模型与经典AI算法拆解为可组合的LLM/工具模块，提供可解释、可复用的设计框架；

**🔧 技术方法**

使用形式化的有向无环图（DAG）描述代理结构，并从RSA、MAP、Tree of Thoughts、MCTS、PSRL、IDS等认知与算法模型中提取模块；

**📊 数据集**

该论文为立场性综述，无新实验数据，主要引用公开的语言代理与实验结果；

**📈 对比分析**

通过对比现有代理与对应认知/算法模板的匹配度，说明在推理、规划、通信等任务中能提升性能（如RSA在Wavelength中显著提高LLM表现）；

**⚠️ 局限性**

局限在于仅依据已有模型，可能不足以覆盖所有LLM任务；缺乏统一评估标准，新颖任务的适应性仍需进一步实验验证。

---

## 149. An Empirical Analysis of Cooperative Perception for Occlusion Risk Mitigation

**arXiv ID:** 2602.23051 | [PDF](https://arxiv.org/pdf/2602.23051v1)

**作者:** Aihong Wang `[一作]` (Tsinghua University), Jun Li `[通讯]` (Tsinghua University)

**通讯引用:** 78467 | [OpenAlex ID](https://openalex.org/A5100361956)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于时间累积的障碍感知风险评估指标 Risk of Tracking Loss (RTL)，并用其评估 V2X 部署对道路安全的影响。

**💡 创新点**

创新点在于将瞬时风险积分为累计风险，形成全局风险度量；同时提出非对称通信架构，使非联网车辆也能接收广播安全信息，从而显著降低所需渗透率。

**🔧 技术方法**

使用多传感器轨迹投影、相对运动学与可达集预测、基于物理约束的风险权重模型等技术，构建 RTL 并进行统计分析。

**📊 数据集**

实验数据来源于中国 SIND 城市交叉口数据集和 Waymo 开放式数据集。

**📈 对比分析**

通过与传统 MTL 指标对比，并利用四分位差和 MAD 系数评估区分度，RTL 在所有场景均表现出更高的统计散布和更好的风险降低，尤其在低渗透率下超越对称 V2V。

**⚠️ 局限性**

局限性包括未考虑通信时延与丢包、对非感知车辆安全提示实现仍需人机交互设计，以及极端几何或视觉盲区的完整覆盖仍有限。

---

## 150. Robust Human Trajectory Prediction via Self-Supervised Skeleton Representation Learning

**arXiv ID:** 2602.22791 | [PDF](https://arxiv.org/pdf/2602.22791v1)

**作者:** Taishu Arashima `[一作]` (Chiba University), Kazuhiko Kawamoto `[通讯]` (Chiba University)

**通讯引用:** 443 | [OpenAlex ID](https://openalex.org/A5010514962)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aaccfe5c-6b26-4208-b23c-35331481e142` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种两阶段框架，先使用自监督的遮挡重建方式预训练骨架编码器，再将预训练好的编码器融入轨迹预测模型，以提升在骨架缺失时的鲁棒性。

**💡 创新点**

创新点在于把骨架鲁棒性从任务层面转移到表示层面：通过遮挡自监督预训练，让编码器学习到在部分观察下仍能保持信息的骨架表示，从而在保持清洁数据下高精度的同时，显著提升对缺失骨架的容错能力。

**🔧 技术方法**

核心技术包括基于遮挡自监督的骨架重建（masked joint reconstruction），使用时空图卷积网络（ST‑GCN）编码器和轻量化 MLP 解码器；随后将预训练编码器与 Social‑TransMotion 的跨模态 Transformer 和社交 Transformer 结合，实现轨迹与骨架特征的融合。

**📊 数据集**

实验基于大型合成数据集 JTA（含高频遮挡），使用 9 帧观察窗口预测 12 帧未来轨迹。

**📈 对比分析**

在随机关节遮挡的评估中，本文方法在 Clean~Moderate（0.0–0.4）遮挡率下的 ADE/FDE 均优于三种基线（标准、重建预处理、直接在缺失骨架上训练），并在 Moderate 处保持最小性能衰减；在高遮挡（0.6）时略逊于重建预处理，但整体竞争力仍强。

**⚠️ 局限性**

局限性包括：在极端高遮挡情况下性能仍受限；方法对骨架的依赖度较高，若骨架缺失严重会导致较大误差；此外，仅在单一合成数据集上验证，真实场景中的多模态噪声与遮挡分布可能进一步挑战鲁棒性。

---

## 151. Quality-Aware Robust Multi-View Clustering for Heterogeneous Observation Noise

**arXiv ID:** 2602.22568 | [PDF](https://arxiv.org/pdf/2602.22568v1)

**作者:** Peihan Wu `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**通讯引用:** 9360 | [OpenAlex ID](https://openalex.org/A5055284175)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种质量感知鲁棒多视图聚类框架QARMVC，针对异质观测噪声实现对样本质量的细粒度评估与聚类。

**💡 创新点**

创新点在于：1）使用信息瓶颈对每个视图压缩提取语义，利用重构误差量化噪声强度；2）基于质量权重的对比学习与质量加权全局融合；3）通过互信息最大化实现局部视图对齐；4）层级学习策略兼顾特征与全局结构。

**🔧 技术方法**

主要技术包括信息瓶颈机制、重构误差量化、质量加权对比损失、质量加权全局融合、互信息最大化、深度分歧聚类损失、两阶段训练。

**📊 数据集**

在Scene-15、MNIST-USPS、LandUse-21、ALOI、MNIST-4等五个多视图基准数据集上进行实验。

**📈 对比分析**

与SURE、CANDY、DIVIDE、RAC-MVC、MVCAN、MSDIB、STCMC_UR等基线对比，在10%、30%、50%噪声比例下，QARMVC在ACC/NMI/ARI等指标上均显著优于所有对手，尤其在高噪声时差距明显。

**⚠️ 局限性**

局限性：1）需要对每个视图训练信息瓶颈网络，计算成本较高；2）对超参数λ的敏感性需要进一步自适应；3）目前仅在模拟异质噪声下验证，真实工业数据的验证仍待开展。

---

## 152. Replacing Multi-Step Assembly of Data Preparation Pipelines with One-Step LLM Pipeline Generation for Table QA

**arXiv ID:** 2602.22721 | [PDF](https://arxiv.org/pdf/2602.22721v1)

**作者:** Fengyu Li `[一作]` (Zhejiang University), Christian S. Jensen `[通讯]` (Aalborg University)

**通讯引用:** 30190 | [OpenAlex ID](https://openalex.org/A5029380368)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种一次性生成表格预处理管道的轻量级LLM框架，用于提高表格问答的准确性与效率。

**💡 创新点**

创新点包括自监督细粒度奖励机制、方差感知组重采样的RL训练、以及推理时的操作合并与自适应回滚，实现在单步推理下高质量管道生成。

**🔧 技术方法**

采用RL with Verifiable Rewards（ORPO）算法，结合自监督奖励、动态方差重采样、操作合并与回滚机制，使用轻量级Qwen模型进行训练与推理。

**📊 数据集**

主要在WikiTQ与TabFact两个公开数据集上进行实验，后者作为跨域评估。

**📈 对比分析**

与多步预处理基线和端到端RL方法相比，模型在WikiTQ上提升约9.6%、TabFact提升约6.1%准确率，同时表格压缩率达79%，推理延迟下降约50%，成本降低2.2倍。

**⚠️ 局限性**

局限性包括对非细胞式答案缺乏直接奖励信号、对新操作的泛化需要进一步验证、以及在更大规模模型与多样化表结构上的适用性尚待探索。

---

## 153. CiteLLM: An Agentic Platform for Trustworthy Scientific Reference Discovery

**arXiv ID:** 2602.23075 | [PDF](https://arxiv.org/pdf/2602.23075v1)

**作者:** Mengze Hong `[一作]` (Hong Kong Polytechnic University), Zhiyang Su `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 534 | [OpenAlex ID](https://openalex.org/A5084841502)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了CiteLLM，一款嵌入LaTeX编辑器的可信参考文献检索平台，利用本地LLM辅助生成检索查询、排名和验证，保证引用不出现幻觉且数据不离本地。

**💡 创新点**

创新点在于将LLM工具直接嵌入写作环境、实现学科感知动态路由只查询可信学术仓库、并用LLM做上下文感知查询构造和段落级语义验证，形成完全无幻觉的引用检索流程。

**🔧 技术方法**

采用本地部署的轻量级LLM进行查询生成、语义匹配和聊天交互，使用GROBID解析PDF获取结构化文本，结合JSON路由决定主备学术库（arXiv、bioRxiv、medRxiv），并通过后端并行检索、去重与BibTeX自动获取。

**📊 数据集**

实验使用40条来自不同学科公开论文的句子，配有人类标注的检索查询和真实参考文献；检索来源为arXiv、bioRxiv、medRxiv等公开预印本仓库。

**📈 对比分析**

将CiteLLM与Google Scholar和具联网功能的ChatGPT在相同句子上检索5条文献进行对比，评价指标为有效率、精准率和可用率；CiteLLM在所有指标上均优于基线，达到100%有效率、84–92%精准率和87–92%可用率。

**⚠️ 局限性**

局限包括仅覆盖公开预印本库，可能无法检索受限期刊；依赖LLM对查询与匹配的准确性；评测样本量有限（40句），未覆盖多种写作场景；系统运行时对本地硬件和LLM模型大小有一定要求。

---

## 154. Quantity Convergence, Quality Divergence: Disentangling Fluency and Accuracy in L2 Mandarin Prosody

**arXiv ID:** 2602.23071 | [PDF](https://arxiv.org/pdf/2602.23071v1)

**作者:** Yuqi Shi `[一作]` (Beijing Language and Culture University), Jinsong Zhang `[通讯]` (Beijing Language and Culture University)

**通讯引用:** 7823 | [OpenAlex ID](https://openalex.org/A5100371404)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对比原生汉语与越南学习者在语法-韵律接口上的边界数量与映射，分析其在不同熟练度水平的固化与稳定性。

**💡 创新点**

证明高级学习者虽在边界数量上与母语者相近，却在结构映射上产生“韵律降级”并固化非母语语序模式。

**🔧 技术方法**

采用C-ToBI音频边界标注、CIR-CTB依存句法分析与负二项回归、卡方独立性检验等统计技术。

**📊 数据集**

使用BLCU-SAIT语料库，包含67名原生汉语说话者和67名按HSK分级的越南学习者。

**📈 对比分析**

通过边界密度与依存关系的交叉检验，发现高熟练度组在B3级别与母语者相当，但在B1/B2级别的结构映射显著偏离，显示“表面流利”而非语法准确。

**⚠️ 局限性**

研究局限在于语料单句片段、越南L1的影响未充分考察，且仅针对口语数据，缺乏长文本或多话题的验证。

---

## 155. DP-aware AdaLN-Zero: Taming Conditioning-Induced Heavy-Tailed Gradients in Differentially Private Diffusion

**arXiv ID:** 2602.22610 | [PDF](https://arxiv.org/pdf/2602.22610v1)

**作者:** Tao Huang `[一作]` (Minjiang University), Hong Chen `[通讯]` (Renmin University of China)

**通讯引用:** 21924 | [OpenAlex ID](https://openalex.org/A5100420423)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对条件扩散模型在差分隐私 SGD (DP-SGD) 下因条件导致的梯度重尾问题，提出 DP‑aware AdaLN‑Zero，借助对条件向量和 AdaLN 调制参数的界定，抑制梯度极端峰值，减少全局裁剪失真。

**💡 创新点**

创新点在于：①将条件感知视为敏感性源，采用前向传播时对条件和调制进行边界约束；②不改动 DP‑SGD 的裁剪与噪声机制，仅通过控制梯度来源来提升隐私训练效果；③提供理论与经验诊断，证明梯度尾部被显著压缩。

**🔧 技术方法**

技术包括：差分隐私 SGD（global clipping + Gaussian noise）、AdaLN‑Zero 结构、对条件向量的 L2 投影、对 AdaLN 参数的坐标裁剪（如 tanh 软阈值）以及梯度统计诊断。

**📊 数据集**

实验数据集为真实电力使用序列 PrivatePower 以及公开 ETT 族（ETTh1、ETTm1），覆盖预测、插值与缺失填补三类时间序列任务。

**📈 对比分析**

方法在与 vanilla DP‑SGD 同样的噪声乘子和裁剪阈值下进行对比；DP‑aware 在所有任务、所有噪声水平下均实现了更低的误差（如 point_RMSE、dist_JS 等）和更小的裁剪失真，显示出显著的隐私‑效能提升。

**⚠️ 局限性**

局限性：仅适用于基于 AdaLN 的条件方式；条件与调制阈值需手工设定，缺乏自动化隐私校准；对更复杂的跨注意力或多源条件仍需扩展；与全流程 DP 预训练/微调结合尚未验证。

---

## 156. From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models

**arXiv ID:** 2602.22859 | [PDF](https://arxiv.org/pdf/2602.22859v1)

**作者:** Hongrui Jia `[一作]` (Peking University), Wei Ye `[通讯]` (Peking University)

**通讯引用:** 9864 | [OpenAlex ID](https://openalex.org/A5101763457)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种诊断驱动的迭代训练框架 DPE，通过诊断模型盲点、目标数据生成和强化学习不断提升大规模多模态模型的推理能力。

**💡 创新点**

创新点在于把诊断和目标数据生成闭环结合，利用多智能体工具化图像检索/编辑生成针对性训练样本，并通过可验证奖励实现稳定学习。

**🔧 技术方法**

使用多智能体生成系统、诊断模块、GRPO 强化学习、图像检索/编辑工具、可验证奖励等技术。

**📊 数据集**

以 Vision‑SR1‑47K 的 1k 种子样本为起点，随后生成约 4k 样本；在 Qwen2.5‑VL‑7B、Qwen3‑VL‑8B 等模型上评估。

**📈 对比分析**

与 VisPlay 对比，DPE 在 11 个基准上连续提升，尤其在长尾推理和图像问答任务上显著提高，且训练过程更稳定。

**⚠️ 局限性**

仍需依赖强大语言模型进行诊断和生成，生成质量受外部图像池限制，且在不同模型体系或跨模态任务上的适用性尚未验证。

---

## 157. MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training

**arXiv ID:** 2602.22645 | [PDF](https://arxiv.org/pdf/2602.22645v1)

**作者:** Lianze Shan `[一作]` (Tianjin University), Weixiong Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 12711 | [OpenAlex ID](https://openalex.org/A5068659777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为MUG的异构图通用预训练框架，利用元路径上下文编码与维度对齐实现跨域、无监督的节点表征学习。

**💡 创新点**

创新点在于：①构建元路径感知的上下文结构编码与维度感知对齐模块，实现多类型节点与关系属性的统一表示；②引入共享编码与全局散射正则化，消除数据集特定的聚合偏差，提升跨图迁移能力。

**🔧 技术方法**

使用的技术包括：元路径随机游走+skip‑gram结构编码、维度感知编码、共享GNN加mask‑reconstruct自监督目标、全局散射正则化以及多任务联合损失。

**📊 数据集**

实验数据集涵盖四个公开异构图：ACM、DBLP、AMiner 与 Freebase。

**📈 对比分析**

与HeCo、HGMAE、HERO等现有自监督异构图方法对比，MUG在跨域节点分类与少样本分类任务中实现了Macro‑F1/Micro‑F1的显著提升，显示出更强的泛化性能。

**⚠️ 局限性**

局限性在于仍依赖预定义的元路径，对大规模图、动态图或多模态属性的处理尚不完善，且对缺失或噪声属性的鲁棒性待进一步提升。

---

## 158. SoPE: Spherical Coordinate-Based Positional Embedding for Enhancing Spatial Perception of 3D LVLMs

**arXiv ID:** 2602.22716 | [PDF](https://arxiv.org/pdf/2602.22716v1)

**作者:** Guanting Ye `[一作]` (University of Macau), Ka-Veng Yuen `[通讯]` (University of Macau)

**通讯引用:** 10239 | [OpenAlex ID](https://openalex.org/A5083222672)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于球面坐标的定位编码（SoPE）并将其作为RoPE的替代，提升3D LVLM对空间位置和方向的感知，进一步结合多尺度频率混合实现更平衡的跨模态注意力；

**💡 创新点**

在传统RoPE框架下首次引入球面坐标映射与多尺度频率分配，兼顾时序、径向、极角和方位角，实现对3D几何与方向信息的精准编码；

**🔧 技术方法**

使用球面坐标重参数化、多维频率分配、三种尺度（线性、对数、周期）频率混合以及RoPE变体的实现；

**📊 数据集**

在Structured3D、ARKitScenes、SpatialLM Dataset等三大3D场景基准上进行实验；

**📈 对比分析**

与RoomFormer、SceneScript、RoPE-3D、CCA、MCA等基线在IoU_2D和IoU_3D指标下对比，SoPE在Structured3D上IoU_2D@0.25提升至88.7、IoU_3D@0.5提升至63.2，整体性能优于现有最优方案；

**⚠️ 局限性**

仍受限于RoPE的相对位置框架，对极大场景的长程依赖处理不充分，且多尺度混合虽轻量但增加了编码复杂度，未来需进一步评估动态场景与实时部署性能。

---

## 159. LUMOS: Democratizing SciML Workflows with L0-Regularized Learning for Unified Feature and Parameter Adaptation

**arXiv ID:** 2602.22537 | [PDF](https://arxiv.org/pdf/2602.22537v1)

**作者:** Shouwei Gao `[一作]` (Oregon State University), Wenqian Dong `[通讯]` (Oregon State University)

**通讯引用:** 1636 | [OpenAlex ID](https://openalex.org/A5045803591)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了 LUMOS 框架，实现了科学机器学习（SciML）模型的端到端特征选择与结构化参数剪枝，显著降低模型体量、推理延迟、能耗和内存占用。

**💡 创新点**

创新点包括：
- 使用 L0 正则化与半随机门控实现可微分的特征与参数联合剪枝；
- 通过 hard‑concrete 重新参数化实现门控的梯度传播；
- 设计层级一致性映射（Layer Consistency Mapper）与拓扑协调器（Topology Coordinator），保证跨层剪枝后张量维度和网络结构一致；
- 支持多种 SciML 典型层（FC、CONV、GIN、GCN、Attention）并保持硬件无关；
- 在训练时一次性完成特征与参数优化，避免多轮剪枝重训。

**🔧 技术方法**

主要技术手段：L0 正则化、半随机门控、hard‑concrete 重新参数化、结构一致性映射、拓扑协调、残差块自适应剪枝、分布式数据并行（DDP）训练、Adam/SGD 优化。

**📊 数据集**

评估使用 13 个跨学科 SciML 工作负载，包括：
- 流体动力学（CFD、Fluid）、分子动力学（PureMD）、宇宙学（CosmoFlow）、材料科学（DMS、EM‑denoise、STEM‑DL）、光学（Optical、SLSTR）、生物学（PPA）、化学（Molhiv）、医学（Brain Tumor）。

**📈 对比分析**

与四种主流剪枝方法（OTO、L1、NEURAL、DepGraph）以及基线模型比较，LUMOS 在平均参数压缩率 71.7% 的同时实现 6.4× 推理速度提升、FLOPs 减少 69.6%，并保持或提升绝大多数任务的准确性（R²、MSE、ACC）。训练时间增幅低于 11%，且在多 GPU DDP 环境下无显著扩展瓶颈。

**⚠️ 局限性**

局限性与挑战：
- 需要调节 λ、门控阈值等超参数，过度剪枝可能导致性能下降；
- 结构一致性映射实现复杂，针对极端网络拓扑（如不规则图）仍需手工适配；
- 目前未结合量化或稀疏算子硬件加速，实际部署仍受限于设备对稀疏运算的支持；
- 对非常大规模模型（如大 Transformer）测试不足，可能面临门控参数量激增；
- 依赖训练数据质量，若数据噪声或偏移严重，门控学习可能无法正确判断特征重要性。

---

## 160. Enriching Taxonomies Using Large Language Models

**arXiv ID:** 2602.22213 | [PDF](https://arxiv.org/pdf/2602.22213v1)

**作者:** Zeinab Ghamlouch `[一作]` (Telecom Paris), Mehwish Alam `[通讯]` (Telecom Paris)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Taxoria，一个基于 LLM 的在线分类树增强管道，利用现有分类树做种子，通过零样本提示生成候选子节点，并对候选节点进行验证、去重、合并，并跟踪来源，最终输出扩展后的分类树及可视化结果。

**💡 创新点**

核心创新在于：① 使用已有分类树作为提示 seed 引导 LLM 生成扩展节点，而非直接提取 LLM 内部 taxonomy；② 通过语义相似度过滤和外部知识图验证减轻幻觉和错误实例化；③ 在节点合并时保留层级一致性并记录 provenance，保证可追溯性。

**🔧 技术方法**

技术包括：零样本提示（Zero‑shot prompting）生成 JSON 格式子节点；Word2Vec 与 LlamaIndex 计算语义相似度；外部知识图（Wikidata）SPARQL 验证实例关系；Ollama 搭载 LLAMA 3/3.2、Mistral 等开源 LLM；Python/GraphQL/可视化工具（如 D3.js）实现交互展示。

**📊 数据集**

使用公开分类树数据集：Schema.org、eBay 商品分类、GeoNames 地理实体分类，分别含 1143、596、690 类，最大深度 6/4/3；通过 LLM 生成的新类分别为 4216、1106、1138，最大深度提升至 7/5/4。

**📈 对比分析**

与传统人工/规则/统计方法相比，Taxoria 通过 LLM 的知识覆盖显著提高了扩展规模；在 Schema.org 上，类数从 1143 增至 4216，最大深度由 6 提升至 7；评估指标主要为类数增长率、深度提升与人工校验的一致性，结果表明自动扩展效率高、准确率在 70‑80% 左右。

**⚠️ 局限性**

限制包括：① LLM 可能生成实例而非类别，需外部 KG 验证但本身存在错误；② 语义相似度阈值需手动调参；③ 当前可视化不支持编辑，缺乏人机交互验证；④ 对 GPU 资源依赖较大，未能在低算力环境下稳定运行。

---

## 161. OmniGAIA: Towards Native Omni-Modal AI Agents

**arXiv ID:** 2602.22897 | [PDF](https://arxiv.org/pdf/2602.22897v1)

**作者:** Xiaoxi Li `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 3978 | [OpenAlex ID](https://openalex.org/A5010558184)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为OmniGAIA的全新多模态评测基准，并设计了支持主动感知和工具集成推理的全新基础代理OmniAtlas，旨在推动视频+音频与图像+音频任务中的多跳推理与多轮工具调用能力。

**💡 创新点**

创新点包括：①基于事件图的多模态构造管线，系统性地挖掘跨模态关联与工具扩展证据；②主动感知机制，让代理能按需“看”或“听”长视频片段；③两阶段轨迹合成与掩码SFT、微调DPO相结合的训练策略，显著提升开源模型的工具使用与推理质量。

**🔧 技术方法**

技术方法涵盖：跨模态统一标记化、事件图构建与扩展、工具集成推理(TIR)范式、主动感知调用、树形回顾导向轨迹合成、轨迹级掩码SFT、Fine‑grained DPO微调，以及使用DeepSeek‑V3.2和Gemini‑3‑Flash作为评审与验证。

**📊 数据集**

使用的数据集包括FineVideo、LongVideoBench、LongVideo‑Reason（约1k长视频），COCO 2017（图像）以及从FineVideo抽取的音频轨道，构成了360个跨域多跳任务。

**📈 对比分析**

在OmniGAIA基准上，官方评测采用Pass@1指标，专有模型Gemini‑3‑Pro最高达62.5，最强开源基线Qwen‑3‑Omni仅达13.3；引入OmniAtlas后，Qwen‑3‑Omni从13.3提升至20.8，验证了工具使用与推理训练方案的有效性，且开源模型在工具调用与推理错误率上仍显著落后。

**⚠️ 局限性**

主要局限包括：工具调用与推理错误仍占大多数，尤其在高难度任务中误差累积导致性能急剧下降；视觉与音频感知错误率高企，表明统一感知能力尚未成熟；此外，训练数据与真实场景的分布差距、工具调用成本与延迟等部署瓶颈仍需进一步研究。

---

## 162. MTRAG-UN: A Benchmark for Open Challenges in Multi-Turn RAG Conversations

**arXiv ID:** 2602.23184 | [PDF](https://arxiv.org/pdf/2602.23184v1)

**作者:** Sara Rosenthal `[一作]` (IBM Research), Marina Danilevsky `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了一个新的多轮检索增强生成（RAG）基准，涵盖了666个任务，重点关注不可回答、未明确定义、非独立以及不清晰回应等挑战性情境。

**💡 创新点**

创新点在于：① 引入了四类新型难题（UNanswerable、UNunderspecified、NONstandalone、UNclear）；② 追加了银行与电信两个企业级域；③ 设计了专门评估未明确定义情境的指标；④ 通过检索重写提升非独立问题的检索效果。

**🔧 技术方法**

技术方法包括：使用 BM25、Elser、Granite‑English R2 等检索模型；查询重写采用 GPT‑OSS‑20B；生成评估使用 GPT‑OSS‑120B、DeepSeek‑V3、Qwen3、Llama 等大模型；对生成结果采用 reference‑based、IDK、faithfulness、以及新建的未明确定义判定指标。

**📊 数据集**

数据集：基于原 MTRAG 的四个领域（CLAPNQ、FiQA、Govt、Cloud）再扩展至银行和电信两大新域，共收集约2,800轮对话，平均8轮/对话，形成666个评估任务。

**📈 对比分析**

与 MTRAG 基准比较：本基准加入更多难题，提升了不可回答/部分可回答比例（28% vs 15%）；检索性能在所有模型中均低于 MTRAG，显示更高挑战性；生成结果在 reference 与 RAG 设置下均低于目标分数，GPT‑OSS‑120B 仍是最优模型，但整体仍有提升空间。

**⚠️ 局限性**

局限性：仅限英文；覆盖六个闭合领域；受限于少量人工标注者，可能存在偏见；评估依赖 Elser 检索和 Mixtral‑8x7b 生成，可能影响泛化。

---

## 163. Structure and Redundancy in Large Language Models: A Spectral Study via Random Matrix Theory

**arXiv ID:** 2602.22345 | [PDF](https://arxiv.org/pdf/2602.22345v1)

**作者:** Davide Ettori `[一作]` `[通讯]` (Politecnico di Milano), Davide Ettori (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于谱几何和随机矩阵理论的两种方法——EigenTrack 用于实时监测大语言/视觉语言模型的幻觉与 OOD 行为，RMT‑KD 用于压缩模型同时保持准确性。

**💡 创新点**

创新点在于将谱分布与马尔科夫过程相结合，利用 Marchenko–Pastur 边界与尖峰模型识别内部表示中的结构与噪声；通过实时谱特征跟踪实现早期风险预警，并通过投影到异常特征子空间与自蒸馏实现无稀疏压缩。

**🔧 技术方法**

技术包括隐藏层谱分析、MP 法与 Tracy–Widom 分布、尖峰协方差模型、递归判别器（GRU/LSTM）用于时间序列监测、投影降维、以及自蒸馏训练。

**📊 数据集**

使用的数据集涵盖 HotPotQA、WebQuestions、EurLex（用于 OOD）、GLUE（SST、QQP、QNLI）、CIFAR‑10，以及 LLaMa、Qwen、Mistral、LLaVa 等公开 LLM/VLM。

**📈 对比分析**

在幻觉检测上与 LapEigvals、SelfCheckGPT、HaloScope 等基线对比，EigenTrack 在 LLaMa‑7B 上 AUROC 0.894；在压缩上与 DistilBERT、Theseus、PKD 等对比，RMT‑KD 实现约 80% 参数压缩，同时提升 BERT‑base 在 GLUE 上 1.8% 准确率，并将推理吞吐量提升约 3×，能耗下降。

**⚠️ 局限性**

局限性包括：评估范围主要集中在中小型模型，谱计算在极大层上成本较高，未探索对注意力矩阵的 RMT 分析，且需进一步验证在更大多模态模型上的可扩展性。

---

## 164. Scaling Audio-Visual Quality Assessment Dataset via Crowdsourcing

**arXiv ID:** 2602.22659 | [PDF](https://arxiv.org/pdf/2602.22659v1)

**作者:** Renyu Yang `[一作]` (Nanyang Technological University), Weisi Lin `[通讯]` (Nanyang Technological University)

**通讯引用:** 30567 | [OpenAlex ID](https://openalex.org/A5100403129)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了截至2025年最大的多模态质量评估数据集YT‑NTU‑AVQ，采用基于众包的实验框架实现大规模、可靠的多维主观打分；

**💡 创新点**

首次提出可在非实验室环境下进行的AVQA众包实验框架，结合分层抽样、动态排名一致性过滤和多阶段受试者筛选，显著提升数据质量与规模；

**🔧 技术方法**

采用jsPsych搭建实验平台、环境自检、训练视频、分层抽样与自动化的排名一致性+标准差双阈过滤技术，后续基准模型评估使用SROCC/PLCC指标；

**📊 数据集**

使用YT‑NTU‑AVQ（1,620条10秒UGC音视频片段），采自YouTube并补充新CC授权视频；候选池来源于VALOR；

**📈 对比分析**

对AQA、VQA与AVQA模型进行5折交叉验证，SROCC/PLCC均值显示VQA模型性能优越，AVQA整体分数与视频质量高度相关；平均SROCC最高达0.938，说明视觉质量在UGC内容中主导；

**⚠️ 局限性**

受限于UGC音频质量分布较窄、严重音频失真样本缺失，视觉质量主导导致模型对音频敏感度不足；众包仍难以完全消除环境噪声与非专业受试者带来的偶发误差；

---

## 165. Bound to Disagree: Generalization Bounds via Certifiable Surrogates

**arXiv ID:** 2602.23128 | [PDF](https://arxiv.org/pdf/2602.23128v1)

**作者:** Mathieu Bazinet `[一作]` (Universite Laval), Pascal Germain `[通讯]` (Universite Laval)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于预测器之间不一致度的新泛化界限，可对任何深度学习模型计算；

**💡 创新点**

创新点在于提供可计算、非空洞、无模型假设的disagreement-based界限，并能在不改动目标模型的情况下与多种理论框架结合；

**🔧 技术方法**

采用样本压缩、模型压缩、PAC-Bayes等框架，利用无标签数据评估并上界模型误差差距；

**📊 数据集**

实验使用MNIST、CIFAR-10和Amazon Polarity三组数据集；

**📈 对比分析**

与传统范数、划分、信息理论等界限对比，实验表明新界限在多种网络（CNN、ResNet18、DistilBERT、GPT2）上更紧且可非空洞；

**⚠️ 局限性**

局限性包括：需要构造合适的代理模型，代理难以压缩或其界限不佳时效果有限；不一定能最小化不一致度，且随机核心集虽然给出最紧界限却缺乏对不一致度的控制。

---

## 166. When Should an AI Act? A Human-Centered Model of Scene, Context, and Behavior for Agentic AI Design

**arXiv ID:** 2602.22814 | [PDF](https://arxiv.org/pdf/2602.22814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 167. ClinDet-Bench: Beyond Abstention, Evaluating Judgment Determinability of LLMs in Clinical Decision-Making

**arXiv ID:** 2602.22771 | [PDF](https://arxiv.org/pdf/2602.22771v1)

**作者:** Yusuke Watanabe `[一作]` (Kyoto University), Yutaka Matsuo `[通讯]` (The University of Tokyo)

**通讯引用:** 13999 | [OpenAlex ID](https://openalex.org/A5090592819)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发 ClinDet-Bench 评估大型语言模型在缺失信息情况下的判断可决定性，并对 8 款 LLM 进行基准测试。

**💡 创新点**

首次将缺失信息情境拆分为可决定与不可决定两类，并公开基准以检验 LLM 能否正确识别两类，从而避免过早判断或过度中止。

**🔧 技术方法**

采用自然语言提示（Base、Chain-of-Thought、Safe）与自评机制，对模型进行解释任务和临床决策任务；利用统计检验（Fisher、Spearman）评估性能。

**📊 数据集**

构造 16 种临床评分系统（如 CHADS₂、SOFA 等）的 94 个人工案例，按信息完整度划分为完整、可决定和不可决定三类。

**📈 对比分析**

对 8 大 LLM 进行解释任务（平均 0.88 准确率）和决策任务；在完整信息下几乎 100% 正确；在可决定缺失信息下约 0.8 正确率，而不可决定情境准确率显著下降，出现过早判断与过度中止，二者呈负相关。

**⚠️ 局限性**

仅覆盖评分系统场景，案例数量有限；未评估少样本提示、训练时微调等技术；温度固定为 1.0；错误分析仅针对错误答案，未检查正确答案的推理质量。

---

## 168. QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition

**arXiv ID:** 2602.22639 | [PDF](https://arxiv.org/pdf/2602.22639v1)

**作者:** Daniel Miao `[一作]` (University of Minnesota), Joe Kileel `[通讯]` (University of Texas at Austin)

**通讯引用:** 345 | [OpenAlex ID](https://openalex.org/A5035113400)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

未提供论文信息，无法确定研究内容。

**💡 创新点**

未提供论文信息，无法识别创新点。

**🔧 技术方法**

未提供论文信息，无法确定所用技术。

**📊 数据集**

未提供论文信息，无法识别使用的数据集。

**📈 对比分析**

未提供论文信息，无法比较方法或评估性能。

**⚠️ 局限性**

未提供论文信息，无法说明研究局限。

---

## 169. Towards Simulating Social Media Users with LLMs: Evaluating the Operational Validity of Conditioned Comment Prediction

**arXiv ID:** 2602.22752 | [PDF](https://arxiv.org/pdf/2602.22752v1)

**作者:** Nils Schwager `[一作]` (Trier University), Achim Rettinger `[通讯]` (Trier University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了条件化评论预测（CCP）任务，评估LLM在多语言环境下模拟社交媒体用户的能力。

**💡 创新点**

首次将真实用户历史与生成的传记结合，揭示SFT可解耦表面结构与语义，提出对传统“直觉提示”方法的批判与实用准则。

**🔧 技术方法**

使用8B参数开源LLM（Llama3.1、Qwen3、Minstrel），进行SFT、提示工程与多语言嵌入对比。

**📊 数据集**

基于德语、英语和卢森堡语的公开推特与RTL评论数据集，采样3,800/650用户做训练/测试。

**📈 对比分析**

通过ROUGE、BLEU、嵌入距离和长度比等指标对比，发现英语表现最佳，SFT显著提升表面与语义一致性，低资源语言出现形式-内容脱耦。

**⚠️ 局限性**

受限于模型规模、语言资源差异、自动评测指标、未考虑多轮对话与非文字行为，且在低资源语境下SFT可能恶化语义对齐。

---

## 170. Interactive Medical-SAM2 GUI: A Napari-based semi-automatic annotation tool for medical images

**arXiv ID:** 2602.22649 | [PDF](https://arxiv.org/pdf/2602.22649v1)

**作者:** Woojae Hong `[一作]` (Sungkyunkwan University), Yong Hwy Kim `[通讯]` (Seoul National University)

**通讯引用:** 2877 | [OpenAlex ID](https://openalex.org/A5013068122)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一款基于 Napari 的桌面工具，用医学版 SAM2 进行 2D/3D 医学图像的半自动标注，支持盒子/点提示、跨切片传播、批量病例导航及体积量化导出。

**💡 创新点**

创新点在于将 Medical‑SAM2 的视频式传播机制与单盒子初始化相结合，构建统一的 cohort‑level 工作流，实现盒子先推、点后细化、最终手动校正，全部在本地完成，并提供批量病例进度管理与量化报告。

**🔧 技术方法**

技术框架：Python、Napari 可视化、PyTorch 推理、Medical‑SAM2（SAM2 记忆机制）、SimpleITK（IO 与几何保留）、N4 bias‑field 纠正（可选）。

**📊 数据集**

使用标准 DICOM/NIfTI 医学图像，未公开特定数据集；用户需自行提供本地研究文件。

**📈 对比分析**

本文未提供实验对比或性能指标，侧重工具实现与实用性；功能已在示例图像上演示，支持快速体积渲染与标注量化。

**⚠️ 局限性**

局限性包括：仅适用于研究级标注，不具备临床决策支持；未进行系统性能评估；依赖本地计算资源；缺乏跨平台部署与大规模自动化验证。

---

## 171. TT-SEAL: TTD-Aware Selective Encryption for Adversarially-Robust and Low-Latency Edge AI

**arXiv ID:** 2602.22238 | [PDF](https://arxiv.org/pdf/2602.22238v1)

**作者:** Kyeongpil Min `[一作]` (Chung-Ang University), Woojoo Lee `[通讯]` (Electronics and Telecommunications Research Institute)

**通讯引用:** 218 | [OpenAlex ID](https://openalex.org/A5101605991)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `fede83ac-7505-405f-ab37-e7284695c47f` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对 Tensor-Train Decomposition (TTD) 压缩模型的选择性加密框架 TT-SEAL，能够在保持安全性的同时显著降低加密开销。

**💡 创新点**

创新点包括：① 针对 TT 格式权重的核心级重要性度量；② 通过数据驱动的阈值校准将安全目标转化为可量化的加密阈值；③ 基于价值动态规划的最小化加密集合求解，保证在满足鲁棒性的前提下最小化加密参数量。

**🔧 技术方法**

使用技术包括：Tensor-Train Decomposition、AES 对核心进行加密、敏感度分析（梯度与输出的二阶信息）、Hutchinson 估计、动态规划优化、FPGA 原型化实现以及多种对抗攻击（I-FGSM、PGD、DI^2-FGSM 等）。

**📊 数据集**

主要数据集为 CIFAR-10，用于训练 ResNet-18、MobileNetV2 和 VGG-16 等模型，并在其 TTD 压缩版本上进行攻击与评估。

**📈 对比分析**

与完整加密（B‑B）和白盒（W‑B）对比，TT-SEAL 通过加密仅约 4.9%（ResNet-18）–15.9%（MobileNetV2）–6.5%（VGG‑16）的参数即可将对抗攻击转移率降至与全加密相当，且 AES 解密在总推理时间中的占比从 58%–41.8%–39% 降至 2.8%–6.1%–2.4%，显著降低了延迟和能耗。

**⚠️ 局限性**

限制在于：① 目前仅针对 TTD 压缩模型，无法直接推广到其他压缩形式；② 核心重要性度量与阈值校准需要额外的验证数据和计算；③ 对极端高攻击强度（ε 超大）下的鲁棒性验证尚未完整；④ 在更大规模模型或不同任务上的可扩展性仍需进一步验证。

---

## 172. Understanding Usage and Engagement in AI-Powered Scientific Research Tools: The Asta Interaction Dataset

**arXiv ID:** 2602.23335 | [PDF](https://arxiv.org/pdf/2602.23335v1)

**作者:** Dany Haddad `[一作]` (Allen Institute for AI), Doug Downey `[通讯]` (Allen Institute for AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

发布并分析了Asta Interaction Dataset，涵盖两款AI科研助手（文献检索和科学问答）超过20万条用户查询与点击流日志，系统性研究了查询模式、交互行为以及使用经验的演变；

**💡 创新点**

首次公开提供真实世界大规模AI科研工具交互数据；提出针对LLM驱动科研助手的多维查询意图、表述和约束分类法；揭示用户将AI助手视为协同科研伙伴的使用习惯；

**🔧 技术方法**

利用检索增强生成（retrieval‑augmented generation）平台，GPT‑4.1对查询进行结构化标注；统计学分析（t检验、二项式逻辑回归、Benjamini‑Hochberg校正）评估交互指标；

**📊 数据集**

Asta Interaction Dataset：258,935条查询、432,059条点击流，包含两大界面与Semantic Scholar的对照查询；

**📈 对比分析**

通过对比Semantic Scholar传统检索的查询长度、复杂度、意图与表述，计算点击率、流失率、返回率等指标，并用逻辑回归预测不同查询类型的点击率，发现用户在经验提升后倾向更精准查询、深入利用引用；

**⚠️ 局限性**

仅基于单一系统（Asta）导致结果可能偏向其设计范畴；采用隐式满意度（点击率）而非直接用户反馈；数据已匿名化，无法验证个体重识别；未覆盖所有科研工作流与工具，外推性受限。

---

## 173. Mirroring the Mind: Distilling Human-Like Metacognitive Strategies into Large Language Models

**arXiv ID:** 2602.22508 | [PDF](https://arxiv.org/pdf/2602.22508v1)

**作者:** Ik-hwan Kim `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**通讯引用:** 12724 | [OpenAlex ID](https://openalex.org/A5086877012)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了后训练框架Metacognitive Behavioral Tuning（MBT），通过向大型推理模型注入元认知行为来稳定推理过程

**💡 创新点**

首次将元认知调控（规划、监控、验证、纠错、验证）显式地注入推理轨迹，显著提升推理稳健性与效率

**🔧 技术方法**

使用教师模型生成元认知结构化推理轨迹，结合监督微调（SFT）与组相对策略优化（GRPO）进行后训练

**📊 数据集**

多跳问答基准HotpotQA、MuSiQue、2WikiMultiHopQA，用HotpotQA训练，MuSiQue/2Wiki验证跨分布性能

**📈 对比分析**

与基线（提示、拒绝采样、TokenSkip、LIMOPro、GRPO）对比，MBT在所有模型规模下均实现更高的准确率、短输出长度和更优的Accuracy‑Efficiency Score；在MuSiQue上实现接近零失效、显著减少冗余推理

**⚠️ 局限性**

需要大规模教师模型生成轨迹，数据准备成本高；对非多跳推理任务或其它领域的通用性仍待验证

---

## 174. Self-Purification Mitigates Backdoors in Multimodal Diffusion Language Models

**arXiv ID:** 2602.22246 | [PDF](https://arxiv.org/pdf/2602.22246v1)

**作者:** Guangnian Wan `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**通讯引用:** 13223 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文首先系统评估了多模态扩散语言模型（MDLM）在数据中毒（后门）攻击下的脆弱性，并提出一种基于选择性视觉token掩码的后门自净化框架DiSP，能够在不使用外部模型或干净数据的前提下消除后门；

**💡 创新点**

创新点在于利用MDLM能够接受部分掩码输入的特性，通过在推理阶段对视觉token进行基于Fisher‑Jacobian的显著性排序并掩码高显著性token，以阻断触发器激活路径，从而实现模型自身的后门净化；

**🔧 技术方法**

核心技术包括：1）使用Fisher信息矩阵与Jacobian二阶近似（通过Hutchinson估计）计算每个视觉token的显著性得分；2）按显著性高低按比例掩码token；3）在掩码输入上运行受污染MDLM，获得净化输出；4）用净化数据再Fine‑tune模型；

**📊 数据集**

实验数据集包括：CC‑SBU‑Align（视觉指令调优），Flickr8k狗图像用于语义误分类攻击；评估基准使用MMMU；触发器多样化（黑色补丁、噪声补丁、多重补丁、混合补丁）以及不同毒化比例；

**📈 对比分析**

通过与随机drop、模型剪枝、数据过滤（BYE）等三种基线在LLaDA‑V与LaViDa模型上进行对比；DiSP将触发输入的攻击成功率从90%以上降至1%以下，且清洁性能几乎不变；在不同触发器、毒化率下均保持低ASR（<3%），显示出显著优于基线的防御效果；

**⚠️ 局限性**

局限性包括：仅针对视觉触发器，未验证对更复杂或多模态触发器的鲁棒性；依赖完整训练集和对模型的访问；高掩码比例可能对部分任务的生成质量产生影响；未探索对极端大规模模型或在线自适应攻击的适用性。

---

## 175. Semantic Communication Through the Lens of Context-Dependent Channel Modeling

**arXiv ID:** 2602.22934 | [PDF](https://arxiv.org/pdf/2602.22934v1)

**作者:** Javad Gholipour `[一作]` (Technische Universität Dresden), Gerhard P. Fettweis `[通讯]` (Technische Universität Dresden)

**通讯引用:** 32949 | [OpenAlex ID](https://openalex.org/A5022117288)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究语义通信中仅由语义通道产生的语义噪声，提出基于上下文的虚拟状态相关信道模型，分析编码器的可表示能力，并推导不同背景知识配置下的容量和可实现速率。

**💡 创新点**

创新点在于：①将语义噪声建模为状态相关信道，将上下文视为状态；②在理想物理通道下，仅考虑语义噪声；③针对发送者与接收者拥有相同、子集或共享部分上下文信息的多种情况，给出了容量/可达率表达式，并对比说明上下文知识差异对语义传输速率的影响。

**🔧 技术方法**

采用概率模型与信息论工具（状态相关信道容量分析、典型序列、辅助随机变量等）进行理论推导。

**📊 数据集**

无实际数据集，研究基于理论推导和信息论定理。

**📈 对比分析**

无实验对比与性能评估；本文以理论容量/可达率为主要结果，未给出具体数值模拟或实验性能比较。

**⚠️ 局限性**

局限性：①只考虑了无物理通道噪声的理想情况；②假设上下文信息在发送/接收端是已知或可非因果获取；③未对模型进行仿真验证或在真实语义通信场景中的应用评估。

---

## 176. EvolveGen: Algorithmic Level Hardware Model Checking Benchmark Generation through Reinforcement Learning

**arXiv ID:** 2602.22609 | [PDF](https://arxiv.org/pdf/2602.22609v1)

**作者:** Guangyu Hu `[一作]` (Hong Kong University of Science and Technology), Hongce Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 430 | [OpenAlex ID](https://openalex.org/A5003614499)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于强化学习的框架，用计算图在算法层面生成硬件模型检查基准，随后通过HLS生成两种结构差异但功能等价的RTL设计，并构造等价检查问题；

**💡 创新点**

创新点在于将强化学习与高层综合结合，利用计算图搜索空间自动发现“small‑but‑hard”实例，且通过奖励信号（模型检查耗时预测）引导生成既小又难的基准；

**🔧 技术方法**

采用多智能体强化学习（Thompson Sampling）、高层综合(HLS)与计算图表示、XGBoost预测器以及AIGER/BTOR2转换；

**📊 数据集**

使用HWMCC 2020/2024竞赛基准作为训练与评估数据集，另外在实验中生成大量新基准；

**📈 对比分析**

与AIGen、AIGFuzz、FuzzBtor等基准生成器比较，在24小时生成实验中，RL方法能更快获得最大耗时高、QR值高的实例，并能显著区分不同PDR类模型检查器的性能；

**⚠️ 局限性**

仅支持PDR类检查器，生成的是功能等价检查实例，未覆盖所有模型检查算法；动作空间有限，未涵盖复杂内存或接口；需依赖商用HLS工具，实验复现受限。

---

## 177. Reliable XAI Explanations in Sudden Cardiac Death Prediction for Chagas Cardiomyopathy

**arXiv ID:** 2602.22288 | [PDF](https://arxiv.org/pdf/2602.22288v1)

**作者:** Vinícius P. Chagas `[一作]` (Federal Institute of Education and Technology of Ceara), Carlos H. L. Cavalcante `[通讯]` (Federal Institute of Education and Technology of Ceara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

构建并验证了一种基于逻辑的可解释性方法，用于预测查加斯心肌病患者的猝死风险，并将其与传统XAI方法（LIME、Anchors）进行对比。

**💡 创新点**

创新点在于：①采用了具有正确性保证的逻辑归约式解释（abductive explanations），确保解释与模型预测完全一致；②将SMT求解器Z3与XGBoost深度集成，克服了旧版本XReason的兼容性问题；③在小样本、低资源环境下实现了高召回率（95%）与100%解释忠实度。

**🔧 技术方法**

使用的技术包括：XGBoost（梯度提升树）、Z3 SMT求解器、逻辑归纳式解释算法、特征重要性排序优化解释生成。

**📊 数据集**

使用的数据集为巴西里约热内卢克莱门蒂诺·弗拉格·菲略大学医院1992–2023年收集的120例查加斯心肌病患者临床数据（49个原始特征，最终选取20个关键特征）。

**📈 对比分析**

与LIME、Anchors相比，逻辑方法在解释忠实度上达到100%（LIME≈98–99%，Anchors≈84–88%），平均解释规模约为6–7个特征（LIME固定10，Anchors平均≈10–11），生成时间约0.38–0.44 s（LIME最快0.05 s，Anchors最慢≈0.88 s）。模型整体性能：召回率95%、AUC 95%。

**⚠️ 局限性**

局限性包括：①数据量有限且单中心，可能影响模型的外推性；②逻辑解释生成时间略高，规模相比Anchors稍大；③缺乏专家对生成解释的临床验证；④目前仅适用于表格数据，无法直接扩展到多模态或时间序列数据。

---

## 178. PSQE: A Theoretical-Practical Approach to Pseudo Seed Quality Enhancement for Unsupervised MMEA

**arXiv ID:** 2602.22903 | [PDF](https://arxiv.org/pdf/2602.22903v1)

**作者:** Yunpeng Hong `[一作]` (Hefei University of Technology), Xindong Wu `[通讯]` (Hefei University of Technology)

**通讯引用:** 41136 | [OpenAlex ID](https://openalex.org/A5080738591)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 PSQE（Pseudo‑Seed Quality Enhancement）框架，针对无监督多模态实体对齐（MMEA）任务，通过多模态特征融合、聚类重采样和错误校正三阶段联合提升伪种子对齐的精度与图覆盖平衡，进而增强基于对比学习的模型性能。

**💡 创新点**

创新点在于：①从理论上剖析伪种子对对比学习的吸引与排斥项的影响，明确种子精度与覆盖度分别调控这两项；②设计了一套完整的三阶段策略（多模态融合+聚类采样、全局重采样+误差校正、邻域扩展+复核），实现精度与覆盖度的协同优化；③将该模块以 plug‑and‑play 方式集成至现有无监督 MMEA 方法，实现显著性能提升。

**🔧 技术方法**

技术手段包括：多模态编码（ResNet‑152 视图、BERT 文本、属性和关系的平均编码）、特征增强（线性映射 + GAT 聚合）、对比学习（Intra‑Modal Contrastive Loss）、K‑means 聚类、基于相似度的种子采样、误差校正矩阵筛选、邻域相似度阈值扩展。模型训练使用 300 次迭代、学习率 0.01、批量 2000。

**📊 数据集**

实验数据集：DBP15K（跨语言 FR‑EN、JA‑EN、ZH‑EN）和 DWY15K（DW‑V1、DW‑V2）。

**📈 对比分析**

将 PSQE 与四大无监督 MMEA 基线（EVA、MCLEA、MEAformer、PCMEA）比较。结果表明，PSQE 使 Hits@1 在 DBP15K 上提升 1.4%–3.8%（ZH‑EN、JA‑EN、FR‑EN），在 DWY15K 上提升 0.8%+；MRR 亦提升 0.9%–1.1%。在多模态和单模态对比实验中，视觉模态对性能贡献最大，去除视觉信息会导致 10%–16% 的显著下降。

**⚠️ 局限性**

局限性：①对视觉模态的依赖性强，缺失图片时性能显著下降；②仍需预训练模型（ResNet、BERT）和显著计算资源；③在极度稀疏或无图结构的知识图中，聚类重采样的效果可能受限；④虽然 PSQE 对聚类数量不敏感，但在特定数据集上可能需要微调阈值和超参数。

---

## 179. Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching

**arXiv ID:** 2602.22871 | [PDF](https://arxiv.org/pdf/2602.22871v1)

**作者:** Roy Miles `[一作]` (Huawei), Ismail Elezi `[通讯]` (Huawei)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在测试时使用扩散语言模型进行多路径探索，并通过过程奖励模型对每一步进行评估，将高质量步骤拼接成合成推理链，再用轻量级自回归模型重新计算最终答案；该方法实现零训练、低延迟且高精度的推理。

**💡 创新点**

创新点在于：① 将扩散采样的多条推理轨迹拆解为可评估的单步候选，① 通过过程奖励模型（PRM）实现逐步自一致性评分，② 采用拼接（stitching）将跨轨迹的高质量步骤聚合成复合推理链，再通过自回归解算器校正并输出答案；该模块化设计实现了探索-评估-合成的解耦，显著提升了测试时推理效率与准确率。

**🔧 技术方法**

核心技术包括：Masked Diffusion Language Models（如LLaDA）用于并行生成低成本推理路径；Process Reward Model（如Qwen-Math-PRM-7B、AceCoderRM）对每一步进行评分；自回归模型（如Qwen2.5-Math-Instruct、Qwen2.5-Coder-7B）作为最终答案的重计算器；以及温度、置信度阈值等参数控制探索多样性与步骤筛选。

**📊 数据集**

实验使用的主要数据集包括：数学推理任务（GSM8K-CoT、Math500、Countdown）、程序生成任务（HumanEval、HumanEval+、MBPP、MBPP+）等，全部采用零样本设置。

**📈 对比分析**

与传统扩散模型（如LLaDA、Dream）和统一混合架构（TiDAR）相比，拼接方法在保持低延迟（比纯扩散快1.8×、比AR模型少3.2×前向步骤）的同时，平均提升 23.9% 的准确率；在最难数据集上提升可达 30%+；与强自回归基线（Qwen3-8B）相比平均提升 4.3%。

**⚠️ 局限性**

局限性：若候选步骤池缺乏足够多样性，拼接后仍可能共享同一错误；自回归解算器只能利用已有步骤，无法弥补完全缺失的中间推理；在极低置信度或过度多样化的采样下，PRM 评分与拼接效果会下降。

---

## 180. TableTale: Reviving the Narrative Interplay Between Data Tables and Text in Scientific Papers

**arXiv ID:** 2602.22908 | [PDF](https://arxiv.org/pdf/2602.22908v1)

**作者:** Liangwei Wang `[一作]` (Hong Kong University of Science and Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1694 | [OpenAlex ID](https://openalex.org/A5100614732)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为 TableTale 的交互式阅读界面，自动构建文档级文本‑表格关联模式，并通过分层级联的可视化提示帮助读者在阅读科学论文时快速定位和验证表格数据。

**💡 创新点**

创新点包括：
1) 采用多粒度（段落→句子→提及）层级联激活的交互方式，保持阅读流畅同时实现细粒度定位；
2) 利用 LLM 代理（检测、解析、聚合）完成提及检测、语义/数值/结构定位，自动生成文档级链接模式；
3) 在 PDF 上实现“镜像表格”锚定与原表格高亮相结合的定位方案，解决大篇幅文档跨页查阅困难。

**🔧 技术方法**

核心技术：大模型驱动的多代理文本‑表格对齐 pipeline；PDF.js+Vue 前端实现交互式高亮；基于 HTML 表格结构的坐标映射；使用 OpenAI GPT‑4（或同等 LLM）进行提及检测与解析；Python Flask 后端与前端通信；文本分句与实体识别使用轻量级 NLP 库。

**📊 数据集**

主要数据集：59 篇 CS 领域顶会论文（共 132 条段落‑表格对），用于构建和评估链接模式；评估集 25 张表格（按大小划分）用于提及检测/定位准确率评测；用户研究采用 24 名早期研究人员阅读两篇真实论文的结果段落。

**📈 对比分析**

对比方法：基线为改进版 PDF.js（仅段落级表格定位）。
性能表现：
- 提及检测精度 82.3%，召回 86.3%，F1 84.3%；
- 提及定位准确率 75.4%（简单表格 87.9%，标准 73.1%，复杂 62.0%）。
- 在用户研究中：阅读时间平均减少 16%（从 4:56 至 4:12）；NASA‑TLX 负载在精神、物理、时间和努力维度显著下降；SUS 分数从 5.11 提升至 5.65；任务完成率从 85% 提升至 100%，无未回答案例。

**⚠️ 局限性**

局限性：
1) LLM 生成的链接易出现 hallucination 与错误，需要人工或自动一致性校验；
2) 依赖外部 PDF 与表格识别 API，易受边界检测错误影响；
3) 对大尺寸、跨页或多层嵌套表格的支持仍有限；
4) 研究主要聚焦 CS 文献，未验证在医学、社会科学等多样化表格格式下的泛化能力。

---

## 181. Support Tokens, Stability Margins, and a New Foundation for Robust LLMs

**arXiv ID:** 2602.22271 | [PDF](https://arxiv.org/pdf/2602.22271v1)

**作者:** Deepak Agarwal `[一作]` (LinkedIn), Tejas Dharamsi `[通讯]` (LinkedIn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将因果自注意力视为隐式噪声生成模型，得到完整的嵌入先验概率，并推导出额外的对数雅可比项，形成距离失真边界的“边缘”约束。

**💡 创新点**

创新点在于：①给因果自注意力提供了一个概率解释；②发现并解析了由token‑dependent 权重导致的雅可比行列式项，提出了“支持token”与SVM支持向量相类比的边缘概念；③将该边缘项作为MAP训练正则化（log‑barrier）加入常规交叉熵，提升模型鲁棒性。

**🔧 技术方法**

技术包括：隐式噪声层（latent‑noise），变分推断中的change‑of‑variables公式，边缘（margin）与log‑barrier 理论，MAP 训练目标，微调 λ_m 的正则化路径分析，实验使用 TensorFlow/PyTorch 训练小型 GPT。

**📊 数据集**

实验采用 WikiText‑2 字符级数据集（约 10.9M 字符，词表 1013），模型为 2‑层 GPT（d=128，4 头，T=256）。

**📈 对比分析**

与仅使用交叉熵的基线对比，加入 log‑barrier 后 BPC 仅提升 1.7% 之差，且在添加嵌入噪声时鲁棒性提高约 12%；正则化路径显示在 λ_m≈0.02 时鲁棒性最佳，清晰展示了精度-鲁棒性权衡。

**⚠️ 局限性**

局限性包括：仅在小型字符级模型上验证；未在大规模 LLM 或语义层面进行评估；正则化项计算开销未优化；未探索基于后验的解码或多模态不确定性利用。

---

## 182. RepoMod-Bench: A Benchmark for Code Repository Modernization via Implementation-Agnostic Testing

**arXiv ID:** 2602.22518 | [PDF](https://arxiv.org/pdf/2602.22518v1)

**作者:** Xuefeng Li `[一作]` (Modelcode AI), Antoine Raux `[通讯]` (Modelcode AI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出RepoMod-Bench，基于实现无关的接口测试框架，用来评估 AI 编码代理在跨语言仓库级代码现代化中的性能；

**💡 创新点**

创新点在于使用标准化的 CLI/REST 接口进行黑盒验证，隐藏测试套件，消除测试驱动过拟合；同时提供大规模真实项目的跨语言 benchmark；

**🔧 技术方法**

采用 Docker 隔离环境、实现无关的测试过滤、自动化构建与执行脚本、以及四种主流 AI 代理（Claude Code、Codex CLI、OpenCode+Claude、OpenCode+OpenAI）进行实验；

**📊 数据集**

构建了 21 个真实开源项目（共 1.6M LOC、11,616 个测试），涵盖 8 种语言（C/C++/Go/Java/Python/Rust/TypeScript/JavaScript），包括 CLI 与 REST API 项目；

**📈 对比分析**

对比四种代理的构建成功率与通过率，平均通过率在 15.3%（大项目）至 91.3%（小项目）之间，Claude Code 最高（48.2%），Codex CLI 最低（30.4%），显示了代理架构与模型差异对性能的影响；

**⚠️ 局限性**

局限性包括：性能随项目规模显著下降，难以处理大型跨文件依赖；对语言间语义差异和专业算法知识的适应不足；且仅评估功能等价性，未覆盖性能、可维护性等其他维度。

---

## 183. DS SERVE: A Framework for Efficient and Scalable Neural Retrieval

**arXiv ID:** 2602.22224 | [PDF](https://arxiv.org/pdf/2602.22224v1)

**作者:** Jinjian Liu `[一作]` (University of California), Sewon Min `[通讯]` (University of California)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个框架，将大规模文本数据集（包含五千亿个标记）转化为高性能的神经检索系统，支持低延迟和适度内存开销，并提供网页接口和API端点。

**💡 创新点**

创新点在于能够在单节点上高效运行大规模检索系统，支持用户在推理时在延迟、准确性和结果多样性之间进行权衡。

**🔧 技术方法**

使用了近似最近邻搜索（ANN），具体采用DiskANN作为默认后端，并提供了精确搜索和多样性搜索的选项。

**📊 数据集**

使用了CompactDS数据集，这是一个包含380B词的语料库，生成了20亿个768维向量。

**📈 对比分析**

与现有方法相比，该框架在CompactDS数据上实现了亚秒级延迟和适度的内存开销（约200GB RAM），在准确性和效率之间取得了良好的平衡。

**⚠️ 局限性**

限制在于尽管提供了多样性搜索选项，但可能不会显著提高RAG性能，且对用户体验的进一步评估留待未来工作。

---

## 184. Knob: A Physics-Inspired Gating Interface for Interpretable and Controllable Neural Dynamics

**arXiv ID:** 2602.22702 | [PDF](https://arxiv.org/pdf/2602.22702v1)

**作者:** Siyu Jiang `[一作]` (City University of Macau), Hui Zeng `[通讯]` (Southwest University of Science and Technology)

**通讯引用:** 1268 | [OpenAlex ID](https://openalex.org/A5077369376)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Knob框架，将神经网络推理建模为二阶机械系统，并通过logit级凸融合实现输入自适应温度缩放，支持静态与连续推理。

**💡 创新点**

创新点在于把经典控制理论中的阻尼比与自然频率映射到神经门控，使门控动态可解释、可调节；通过凸融合自然抑制过度自信，并提供可实时调节的物理参数界面。

**🔧 技术方法**

采用二阶质量-弹簧-阻尼动力学模型、Tustin离散化、logit级凸融合、双流骨干、输入自适应门控、指数滑动平均等技术。

**📊 数据集**

使用CIFAR-10进行训练，CIFAR-10-C作为分布偏移评估数据集。

**📈 对比分析**

与Static、Attention、Knob-IA、ODE-Lite、Knob-ODE等方法比较，Knob-IA在平均准确率略优，ODE-Lite在ECE_deb和计算成本上表现最好；Knob-ODE在连续模式下展示二阶动力学优势，但在i.i.d.模式下ECE略高。

**⚠️ 局限性**

局限性包括仅在CIFAR-10/CIFAR-10-C实验，双流架构要求，未进行系统的实时调参评估，连续模式下优势显著，未在更大规模数据集或单流模型上验证。

---

## 185. Stable Adaptive Thinking via Advantage Shaping and Length-Aware Gradient Regulation

**arXiv ID:** 2602.22556 | [PDF](https://arxiv.org/pdf/2602.22556v1)

**作者:** Zihang Xu `[一作]` (Beihang University), Lijun Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两阶段稳定自适应思考框架：先通过Hybrid Fine‑Tuning让模型同时学习思考和不思考两种生成模式，再利用强化学习结合Correctness‑Preserving Advantage Shaping（CPAS）和Length‑Aware Gradient Regulation（LAGR）实现对思考深度的自适应控制。

**💡 创新点**

创新点：① CPAS 通过优势塑形保留正确长链思考的探索能力，避免传统短链奖励导致的误惩罚；② LAGR 通过长度加权梯度调节缓解极端长度异质性导致的梯度偏差，提升训练稳定性；③ 两阶段训练流程将思考与非思考能力统一初始化，显著降低RL探索难度。

**🔧 技术方法**

使用技术：Hybrid Fine‑Tuning、GRPO‑style 强化学习、CPAS、LAGR、控制标记（control token）、LLaMA‑Factory 框架、VeRL 强化学习实现、Math‑Verify 自动校验、AdamW 优化器、余弦学习率衰减。

**📊 数据集**

使用数据集：训练阶段使用 OpenR1‑Math‑220k、DeepMath‑103K、NuminaMath；HFT 通过 DeepSeek‑R1‑0528（思考模式）和 DeepSeek‑V3‑0324（非思考模式）提取示例；强化学习阶段以 DeepScaleR 为主；评估使用 MATH‑500、AIME‑2024、AIME‑2025 以及 OOD 的 GPQA。

**📈 对比分析**

对比方法：基准模型（思考/非思考/混合）、长度惩罚方法（O1‑Pruner）、路由方法（RouteLLM）、现有自适应思考方法（ThinkLess、AdaptThink）。实验结果显示，在 1.5B 与 7B 模型上，准确率提升分别为 3.7 与 3.6 分，同时生成 token 数量减少约 40.6%/43.9%，在 GPQA 上也取得最高准确率并将平均响应长度缩减 51.0%。

**⚠️ 局限性**

局限性：仅在 1.5B 与 7B 规模模型上验证，缺乏更大规模模型的评估；训练主要基于数学推理数据，尽管 GPQA 上表现良好，但需进一步扩展到多元领域以提升通用性。

---

## 186. EyeLayer: Integrating Human Attention Patterns into LLM-Based Code Summarization

**arXiv ID:** 2602.22368 | [PDF](https://arxiv.org/pdf/2602.22368v1)

**作者:** Jiahao Zhang `[一作]` (Vanderbilt University), Yu Huang `[通讯]` (Vanderbilt University)

**通讯引用:** 5517 | [OpenAlex ID](https://openalex.org/A5070896899)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量级的 EyeLayer 模块，将人类眼动模式作为先验知识注入大语言模型，以提升代码摘要质量。

**💡 创新点**

创新点在于：①将人类注意力建模为多模态高斯混合，形成可迁移的概率先验；②通过低秩扰动和因果注意重分配实现与任何 LLM 的无缝集成；③在多种规模与架构模型上验证其通用性。

**🔧 技术方法**

使用的技术包括：多模态 Gaussian Mixture attention、低秩变换、因果注意重分配、PCGrad 多任务优化、hook 注入策略、以及多任务联合训练。

**📊 数据集**

使用的数据集为：CodeXGLUE Java 子集（用于代码-摘要对训练），EyeTrans（27 名专业开发者的 625 条眼动记录，用于对齐监督）。

**📈 对比分析**

方法对比：与仅进行 SFT 的基线模型在 BLEU‑4、ROUGE‑L、METEOR、BERTScore 上进行评估。跨 LLaMA、Qwen、CodeBERT 等五种模型，EyeLayer 均取得提升，BLEU‑4 最大提升 1.98，整体提升约 1‑2 点，证明其有效性。

**⚠️ 局限性**

局限性：眼动数据规模有限且仅来自 27 名开发者，主要针对 Java 代码；评估依赖自动化指标，可能无法完全反映人类主观质量；跨语言与任务的泛化能力尚未充分验证。

---

## 187. Differentiable Zero-One Loss via Hypersimplex Projections

**arXiv ID:** 2602.23336 | [PDF](https://arxiv.org/pdf/2602.23336v1)

**作者:** Camilo Gomez `[一作]` (University of Central Florida), Liansheng Tang `[通讯]` (University of Central Florida)

**通讯引用:** 839 | [OpenAlex ID](https://openalex.org/A5027778319)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种可微分的（n,k）维超简单x投影软二元Argmax@k，并以此构造了HyperSimplex损失，作为接近0-1误分类损失的平滑近似，主要用于减小大批量训练中的泛化缺口。

**💡 创新点**

创新点在于：①首次将欧氏投影到超简单x作为可微分层引入神经网络；②设计了温度参数调控的软化方案，使投影保持梯度且可微；③将投影与平方损失组合形成HyperSimplex损失，显著提升了大批量训练的泛化性能。

**🔧 技术方法**

使用了欧氏投影、等距投影（PAV算法）实现超简单x投影；利用梯度链式法则和投影雅可比矩阵高效求导；对温度τ进行调参；在多分类场景下对每个类别分别投影；实验中采用CNN、PyTorch实现。

**📊 数据集**

实验数据集：CIFAR-10、Fashion‑MNIST；额外在tabular数据上做GBRT交叉域验证。

**📈 对比分析**

通过将HyperSimplex与交叉熵、Hinge、MSE在七种批量尺寸（128–8192）下的最大测试准确率对比，使用配对t检验（10%显著性）验证其显著提升；结果显示在CIFAR-10上93%对比显著提升，Fashion‑MNIST约86%，并显著抑制了大批量训练导致的准确率下降。

**⚠️ 局限性**

局限性：仅在图像分类与少数表格数据上验证，缺乏对更大规模网络或其他任务（如序列生成、对比学习）的评估；需要额外计算投影步骤，可能带来一定的开销；温度τ的选取仍需经验调优。

---

## 188. SUPERGLASSES: Benchmarking Vision Language Models as Intelligent Agents for AI Smart Glasses

**arXiv ID:** 2602.22683 | [PDF](https://arxiv.org/pdf/2602.22683v1)

**作者:** Zhuohang Jiang `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 37135 | [OpenAlex ID](https://openalex.org/A5100404176)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了专门针对智能眼镜的视觉问答基准数据集，并基于该基准设计了一个检索增强智能眼镜代理。

**💡 创新点**

创新点在于：①真实来自智能眼镜的第一人称视角数据；②记录完整的检索轨迹和推理步骤；③针对多跳推理和对象定位的专门评测；④开发了需求自适应回答器与双镜头检索器相结合的智能眼镜代理。

**🔧 技术方法**

使用了大规模视觉语言模型（如 LLaMA‑3.2‑Vision、Qwen2.5‑VL、Gemini‑2.5‑Pro 等），并结合图像检测、查询拆分、网络搜索、网页解析与重排序等技术实现检索增强生成。

**📊 数据集**

使用了 2,422 条 egocentric 图像-问题对，涵盖 14 个应用领域、8 个查询类别，数据来自 Ray‑Ban Meta、XiaoMi 与 RayNeo 三款智能眼镜。

**📈 对比分析**

通过与 26 种 VLM（包括公开与专有模型）和两种启发式 RAG 版本的对比，结果表明新设计的代理在基准上取得 44.10% 的准确率，超过 GPT‑4o 2.19% 并优于现有最佳模型 Gemini‑2.5‑Pro（43.02%）。

**⚠️ 局限性**

仍然面临 45% 以下的整体准确率；检索质量、工具调用错误和查询拆分失误是主要瓶颈，提示未来需要更精准的检索策略和更强的多模态推理能力。

---

## 189. Extending Czech Aspect-Based Sentiment Analysis with Opinion Terms: Dataset and LLM Benchmarks

**arXiv ID:** 2602.22730 | [PDF](https://arxiv.org/pdf/2602.22730v1)

**作者:** Jakub Šmíd `[一作]`, Pavel Král `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了首个带有意见词注释的捷克餐厅评论数据集，并基于该数据集对 ASTE、ASQP 与 ACOS 三类复合情感分析任务进行实验。

**💡 创新点**

创新点在于：①首次为捷克语提供完整的四元组（aspect term、aspect category、opinion term、sentiment polarity）标注；②提出利用大型语言模型进行数据翻译与标签对齐的跨语言迁移方法；③系统比较了 fine‑tuned Transformer、LLM 及跨语言方案的效果。

**🔧 技术方法**

技术手段包括：mT5 的序列到序列微调；QLoRA 量化微调多种 LLM（LLaMA 3.1/3.3、Gemma 3、Orca 2、Aya 23 等）；零样本、少样本提示；多语言与跨语言训练；以及使用 GPT‑4o mini 进行数据翻译与标签对齐。

**📊 数据集**

使用的数据集为：①自建捷克餐厅 ABSA 数据集（约 3000 句，约 7000 四元组）；②从该数据集导出的 ASTE、ASQP、ACOS 任务子集；③对应的英语 SemEval‑2016 任务数据集（ASTE、ASQP、ACOS）。

**📈 对比分析**

在单语实验中，微调后的 mT5 在三项任务平均 F1 约为 64%，显著优于所有零样本或少样本 LLM；LLM 在少样本下可达约 35–45%；跨语言实验中，加入翻译数据后模型平均提升 2–4% 但仍低于 20% 的微调效果。多语言微调与跨语言相近，均低于单语微调。

**⚠️ 局限性**

局限性包括：数据仅覆盖餐厅领域，难以推广到酒店、电子商务等其他领域；仅聚焦捷克语，跨语言实验受翻译与标签对齐误差影响；隐式意见词的标注仍具主观性，导致模型对细粒度情感表达的捕获仍有限。

---

## 190. Affine-Scaled Attention: Towards Flexible and Stable Transformer Attention

**arXiv ID:** 2602.23057 | [PDF](https://arxiv.org/pdf/2602.23057v1)

**作者:** Jeongin Bae `[一作]` (NAVER Cloud), Dongsoo Lee `[通讯]` (NAVER Cloud)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究Transformer注意力的softmax约束，提出Affine-Scaled Attention，使注意力权重可输入自适应缩放并加偏置，提升训练稳定性与下游性能。

**💡 创新点**

创新点在于在softmax归一化后直接引入输入依赖的缩放因子和偏置，放松单元归一化限制，实现可控的注意力幅度；相比Attention sink仅加入固定sink、Gated Attention仅后置门控，Affine-Scaled Attention直接作用于权重分布并自适应。

**🔧 技术方法**

技术上采用Affine-Scaled Attention（softmax + 缩放 + 偏置），并设计线性剪辑激活函数；在Decoder‑only Transformer结构中训练；配合知识蒸馏（0.5B、1B、3B学生模型）。

**📊 数据集**

使用FineWebEdu 20B-token预训练语料；下游评估使用ARC、HellaSwag、PIQA、BoolQ、WinoGrande等常见推理数据集。

**📈 对比分析**

与标准softmax、Attention sink、Gated Attention以及两者组合对比；在不同规模模型上比较训练损失、梯度规范、perplexity和零样本推理准确率；Affine‑Scaled Attention在训练损失更低、梯度更平稳，零样本准确率提升约2–3%（大模型可达3%），与Gated Attention组合进一步提升。

**⚠️ 局限性**

局限性：实验仅在中等规模模型（0.5B–3B）上验证，未对更大模型或更强教师模型进行评估；仅使用知识蒸馏，未与其他预训练策略对比；受计算资源限制，实验规模有限。

---

## 191. Opacity in Discrete Event Systems: A Perspective and Overview

**arXiv ID:** 2602.22713 | [PDF](https://arxiv.org/pdf/2602.22713v1)

**作者:** Xiang Yin `[一作]` (Shanghai Jiao Tong University), Xiang Yin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3228 | [OpenAlex ID](https://openalex.org/A5034304769)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了离散事件系统中隐蔽性（opacity）的核心概念、主要形式、验证与实施方法，以及在更丰富模型和实际应用中的发展；

**💡 创新点**

提出了以估计视角统一不同隐蔽性概念的框架，并系统梳理了从有限自动机到随机、时序、Petri网、混合动力系统等更复杂模型的扩展；

**🔧 技术方法**

采用形式化模型检验技术（如状态空间展开、可观测性与可达性分析）、监督控制设计与动态观测调度、以及编辑/混淆策略等方法；

**📊 数据集**

由于为综述性质，未使用具体实验数据集；

**📈 对比分析**

主要通过文献比较已有验证与实施算法的复杂度与适用场景，指出不同观测模型导致算法结构变化，但未给出统一性能评估；

**⚠️ 局限性**

局限在于缺乏对大规模系统可扩展性方法的深入讨论、对智能/数据驱动攻击者的针对性分析以及实证验证不足。

---

## 192. Guidance Matters: Rethinking the Evaluation Pitfall for Text-to-Image Generation

**arXiv ID:** 2602.22570 | [PDF](https://arxiv.org/pdf/2602.22570v1)

**作者:** Dian Xie `[一作]` (Hong Kong University of Science and Technology), Zeke Xie `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 257 | [OpenAlex ID](https://openalex.org/A5066773635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过对最近的扩散模型引导方法进行深入评估，揭示了人类偏好模型对大型引导尺度存在强偏差的评估陷阱，并提出了基于有效引导尺度的 GA-Eval 评估框架，对八种主流引导方法进行公平比较。

**💡 创新点**

创新点在于发现并量化了大引导尺度对人类偏好评分的偏差，提出有效引导尺度校准方法和 GA-Eval 框架，以消除这一偏差；同时设计了 Transcendent Diffusion Guidance (TDG) 作为“对照”方法，展示了在传统评估中可能被误判为优异的方案。

**🔧 技术方法**

技术包括：扩散模型（Stable Diffusion-XL、SD-2.1、SD-3.5、DiT-XL/2），多种引导方法（Z‑Sampling、CFG++、SAG、PAG、SEG、FreeU、APG、TDG），人类偏好模型（HPS v2、ImageReward、PickScore、AES、CLIPScore），以及 GA-Eval 中的有效引导尺度计算与公平比较算法。

**📊 数据集**

使用的数据集包括 Pick‑a‑Pic、DrawBench、Human Preferred Dataset (HPD)、GenEval、COCO‑30K、ImageNet‑50K。

**📈 对比分析**

在传统评估框架下，所有方法大多表现出高于标准 CFG 的得分；但在 GA-Eval 框架下，绝大多数方法的赢率显著下降，表明其优点主要源自大引导尺度，而非真正的技术创新。只有 Z‑Sampling 与 CFG++ 在有效尺度下仍保持相对较高的赢率。

**⚠️ 局限性**

局限性：评估主要聚焦于文本到图像的任务，未覆盖多模态或视频生成；GA-Eval 仍依赖于人类偏好模型的特定训练数据，可能存在其他未被捕捉的偏差；TDG 作为对照方法在实践中并未真正提升图像质量，仅在传统评估中表现突出。

---

## 193. A data- and compute-efficient chest X-ray foundation model beyond aggressive scaling

**arXiv ID:** 2602.22843 | [PDF](https://arxiv.org/pdf/2602.22843v1)

**作者:** Chong Wang `[一作]` (Stanford University), Curtis P. Langlotz `[通讯]` (Stanford University)

**通讯引用:** 20420 | [OpenAlex ID](https://openalex.org/A5087710258)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

通过主动原型驱动的数据挑选机制，构建了一个在胸部X光影像与报告对上进行对比学习的基础模型CheXficient，显著减少所需训练样本与计算量；

**💡 创新点**

创新点在于在线原型驱动的数据筛选，自动优先保留信息丰富、低频或多样化的图像‑报告对，实现数据与算力双重效率提升；

**🔧 技术方法**

采用CLIP框架下的InfoNCE对比学习，结合DINOv2视觉编码器与BioClinicalBERT文本编码器，并实现原型更新与EMA等技术；

**📊 数据集**

使用了1,235,004对公开CXR图像‑报告配对数据，来源于13个公开数据集（如MIMIC‑CXR、CheXpert‑Plus、PadChest等）；

**📈 对比分析**

在20个评测基准（零样本分类、跨模态检索、疾病预测、分割、报告生成）与CheXfull、CheXrandom及多种大型医学基础模型对比，CheXficient仅使用22.7%样本、27.3%算力即可达到或超过全数据模型，并在长尾疾病上提升泛化；

**⚠️ 局限性**

局限性包括仅评估ViT‑Base规模、未覆盖问答等多模态任务，对极稀病种仍受限，需进一步采集或增强数据以提升极端稀有病种的表现。

---

## 194. From Bias to Balance: Fairness-Aware Paper Recommendation for Equitable Peer Review

**arXiv ID:** 2602.22438 | [PDF](https://arxiv.org/pdf/2602.22438v1)

**作者:** Uttamasha Anjally Oyshi `[一作]` (University of Arkansas), Susan Gauch `[通讯]` (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Fair-PaperRec，一个在双盲审后使用可微公平损失的多层感知机，用于论文推荐并缓解作者人群偏差。

**💡 创新点**

创新点是将多属性（种族、国家）的交叉公平正则化与预测目标融合，提供可调的 λ 超参数，使系统在高偏差环境下实现公平与质量的平衡，并在合成与真实会议数据上验证其有效性。

**🔧 技术方法**

采用多层感知机（MLP）模型，结合交叉熵预测损失和统计公平损失（demographic parity loss），通过 λ 控制公平与效用的权衡，并在训练后进行排序重排。

**📊 数据集**

使用合成数据（公平/中度/高偏差三种分布）以及 ACM SIGCHI、DIS、IUI 2017 的真实提交数据。

**📈 对比分析**

与基线无公平限制的 MLP 以及原始同行评审结果对比；在合成数据中 λ≈3 为最佳甜点点，兼顾多样性与效用；在真实数据中公平 λ 使被代表人群增加 42% 而效用仅下降 ≤3%。

**⚠️ 局限性**

局限在于仅采用简单 MLP、未考虑因果路径与评审者动态；公平权重需手工调优；对分布漂移的鲁棒性和可解释性仍需提升。

---

## 195. UCM: Unifying Camera Control and Memory with Time-aware Positional Encoding Warping for World Models

**arXiv ID:** 2602.22960 | [PDF](https://arxiv.org/pdf/2602.22960v1)

**作者:** Tianxing Xu `[一作]`, Song-Hai Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 5753 | [OpenAlex ID](https://openalex.org/A5049883689)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种统一相机控制与长时记忆的世界模型框架，利用时序感知的位置编码变换实现显式时空对应；

**💡 创新点**

核心创新在于时间感知位置编码（TPE）变换和高效双流扩散变换器，既保留了记忆信息，又显著降低了计算成本；

**🔧 技术方法**

技术包括：基于DiT的潜在扩散模型、深度估计（Depth Anything 3）生成点云、时序位置编码变换、双流稀疏注意力与二进制块掩码、数据驱动的点云渲染数据增强；

**📊 数据集**

使用了超过560K个单目视频（MiraData、SpatialVID、Context-as-Memory）以及RealEstate10K、Tanks & Temples等公开数据集；

**📈 对比分析**

与UCPE、VWM、VMem、Context-as-Memory等方法比较，在相机控制任务中RotErr降至1.01°、TransErr 0.11、FID 69.8、FVD 261；在长期记忆任务中，FID、FVD显著优于对手，显示出更好的场景一致性；

**⚠️ 局限性**

局限包括：片段间预测误差累计导致整体外观不稳定、对移动物体记忆注入时易出现伪影、以及深度估计和点云渲染导致的存储与计算开销较大。

---

## 196. Evaluating Stochasticity in Deep Research Agents

**arXiv ID:** 2602.23271 | [PDF](https://arxiv.org/pdf/2602.23271v1)

**作者:** Haotian Zhai `[一作]` (University of Texas at Austin), Liu Leqi `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了深度研究代理（DRA）的随机性问题，提出信息获取MDP框架、总方差评估指标和方差分解方法，并通过温度消融实验揭示早期随机性和推理模块对最终输出变异的主导作用，随后设计了结构化输出和查询集成两种策略来降低随机性，

**💡 创新点**

创新点在于首次将DRA建模为信息获取MDP，量化多层随机性并将其拆分为传播与内在随机性；利用温度消融实验精准定位随机性来源；提出针对推理和查询的结构化/集成干预，显著提升系统可复现性

**🔧 技术方法**

使用信息获取MDP模型、温度消融实验、结构化JSON/Markdown输出约束、查询集成投票（intersection）以及多跑平均等技术；核心技术是基于LLM的ReAct代理与外部检索接口

**📊 数据集**

在WebWalkerQA（20个问答实例）和DeepSearchQA（20个研究任务实例）上进行评估，并使用Qwen3-30B和Qwen3-235B LLM作为后端

**📈 对比分析**

与基线（无干预）对比，平均降低22%总方差，同时保持或提升12%回答准确率；在不同指标（答案、发现、引用）上均表现出显著的随机性减小

**⚠️ 局限性**

局限性包括：仅针对特定的ReAct式DRA进行实验，未验证对其他架构的通用性；干预方法需要额外的计算和结构设计；在API不确定性下仍存在残留随机性

---

## 197. Workload Buoyancy: Keeping Apps Afloat by Identifying Shared Resource Bottlenecks

**arXiv ID:** 2602.22852 | [PDF](https://arxiv.org/pdf/2602.22852v1)

**作者:** Oliver Larsson `[一作]` (Umeå University), Erik Elmroth `[通讯]` (Umeå University)

**通讯引用:** 6771 | [OpenAlex ID](https://openalex.org/A5070862414)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出“浮力”（buoyancy）概念，将应用级指标与共享资源争用信息整合，用于多租户环境中的工作负载性能监控与资源管理。

**💡 创新点**

通过在线计算资源分数和浮力分数，实现不需要先前剖析即可实时检测瓶颈，并可直接替代传统SLO指标作为调度控制器输入。

**🔧 技术方法**

利用硬件性能计数器（CPU使用、LLC miss率、内存带宽占用）在线生成资源分数，结合Prometheus/Kubernetes原型实现。

**📊 数据集**

使用五种典型多租户工作负载（nginx、memcached、moses、img‑dnn、xapian）以及干扰工作负载进行实验。

**📈 对比分析**

与传统CPU利用率等启发式指标比较，浮力在瓶颈出现前的响应度提高约19.3%，并成功替代ESTHER控制器中的SLO目标，保持95th百分位延迟设定。

**⚠️ 局限性**

仅覆盖CPU、LLC、内存带宽资源，未验证网络/磁盘等；实验仅在单一硬件平台完成，缺乏跨平台验证；依赖硬件计数器实现，未来硬件变更可能影响准确性。

---

## 198. Enhancing Geometric Perception in VLMs via Translator-Guided Reinforcement Learning

**arXiv ID:** 2602.22703 | [PDF](https://arxiv.org/pdf/2602.22703v1)

**作者:** Hao Yu `[一作]` (Tsinghua University), Chun Yuan `[通讯]` (Tsinghua University)

**通讯引用:** 32811 | [OpenAlex ID](https://openalex.org/A5008769328)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个自动生成的几何图形语言（GeoDSL）和对应的图像数据集 GeoPerceive，用于单独评估和提升视觉-语言模型（VLM）的几何感知能力，并设计了基于 NL→DSL 翻译器的强化学习框架 GeoDPO 通过奖励函数指导模型生成更准确的自然语言描述。

**💡 创新点**

创新点在于：1) 构建了唯一对应的可判定几何 DSL（GeoDSL），消除传统 DSL 的多重表示导致的评估歧义；2) 设计了可控复杂度的自动生成管道，低成本产生大规模训练/评测数据；3) 引入 NL→DSL 翻译器作为奖励模型，将细粒度的 DSL 评分转换为偏好信号，采用 DPO 对 VLM 进行层级奖励训练，从而解决顺序级别 SFT 对等价程序的敏感性和分布迁移问题。

**🔧 技术方法**

使用的技术包括：自动化 DSL 生成与求解引擎、PyTorch 基础的图像渲染、NL→DSL 翻译器（基于 Qwen2.5-7B 加 LoRA 细调）、强化学习中的 DPO（基于偏好对训练）、以及 LoRA 微调以在多种 VLM（Qwen2.5-VL、InternVL3、LLaVA-Next）上实现模型提升。

**📊 数据集**

主要数据集为 GeoPerceive（由 10,000 条 GeoDSL 程序与对应图像组成）以及 Translator Split（10,000 条 NL–DSL 语料）用于训练翻译器；评估时还使用手工构造的 OOD 集合（100 条图）和 MathVista 中标记为 geometry diagram 的 203 条题目。

**📈 对比分析**

与原始模型和直接监督微调（SFT）对比，GeoDPO 在 3 种 VLM 上均实现显著提升：in‑domain 感知 +26.5%，out‑of‑domain 感知 +8.0%，以及 downstream 推理 +39.0%；SFT 在 OOD 上往往效果不佳甚至退化，说明 GeoDPO 更具泛化性。

**⚠️ 局限性**

局限性包括：1) 需要依赖 NL→DSL 翻译器的准确性，翻译错误会直接影响奖励信号；2) 生成的 GeoDSL 程序仍可能忽略更复杂的几何约束或特殊构造；3) 仅针对几何感知，未覆盖更广泛的多模态推理场景；4) 训练成本受限于大规模图像渲染和 RL 采样，仍有进一步优化空间。

---

## 199. Modality Collapse as Mismatched Decoding: Information-Theoretic Limits of Multimodal LLMs

**arXiv ID:** 2602.23136 | [PDF](https://arxiv.org/pdf/2602.23136v1)

**作者:** Jayadev Billa `[一作]` `[通讯]` (Unaffiliated researcher), Jayadev Billa (Unaffiliated researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了多模态LLM在处理非文本信息时的失败，证明问题在于解码器而非编码器，并通过理论与实验验证了信息可达性差距。

**💡 创新点**

创新点是将模态坍塌视为不匹配解码器问题，引入GMI‑Wasserstein 上界，证明解码器对非文本方向无响应，并提出以目标为导向的LoRA修复方法。

**🔧 技术方法**

采用线性探针、模态特异性方向消除、LoRA微调、信息理论度量（GMI、Wasserstein距离）以及对比性实验等技术。

**📊 数据集**

使用的主要数据集包括语音领域的LibriSpeech、CREMA‑D、ESC‑50以及视觉领域的MS‑COCO。

**📈 对比分析**

通过对比不同模型、不同编码器（对齐与非对齐）以及对模态特异性方向的消除，发现消除非文本方向能显著降低交叉熵；LoRA情感目标提升情感探针+7.5%，生成准确率+44.5%，验证了改进效果。

**⚠️ 局限性**

局限性在于线性探针不完全捕捉信息，视觉模态仅出现轻微停滞，理论上界不够紧凑，且实验主要聚焦于语音与视觉两种模态。

---

## 200. A Simple Distributed Deterministic Planar Separator

**arXiv ID:** 2602.22916 | [PDF](https://arxiv.org/pdf/2602.22916v1)

**作者:** Yaseen Abd-Elhaleem `[一作]` (University of Haifa), Oren Weimann `[通讯]` (University of Haifa)

**通讯引用:** 1639 | [OpenAlex ID](https://openalex.org/A5000569191)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文提出了一种**确定性**的分布式算法，在面向平面图的模型下，在 Õ(D) 轮内构造一个大小为 O(D) 的 3/4‑平衡路径分离器（即基于给定生成树 T 的基本环），并可直接用于构造 DFS 树、SSSP、最大流、最小割等经典问题的无随机化实现。

**💡 创新点**

创新点在于：
- 简单的权重转移方案——每个顶点仅将自身权重转移到其所属的任一面，避免了原有方案中的随机抽样或繁琐的三角化处理；
- 通过在平面图双向图上进行子树求和，既利用 Lipton‑Tarjan 的双向性，又保留 Ghaffari‑Parter 的关键节点判定逻辑，证明该简单权重分配即可保证存在 3/4‑平衡分离器；
- 结合低拥塞快捷路、部分聚合与 Cole‑Vishkin 颜色化，实现完全确定性的 Õ(D) 轮实现，并可并行处理多个子图。

**🔧 技术方法**

核心技术：
- 平面图双向图与基本环/切割的对应关系；
- 权重分配与子树权重求和（部分聚合）
- 关键节点（balanced/critical）判定与寻找；
- 低拥塞快捷路（shortcuts）在平面图上的构造与应用；
- 传统随机连通性算法的确定性化（如 Cole‑Vishkin 颜色化）以及连接性模拟。

**📊 数据集**

无实验数据集，所有结果均为理论证明与复杂度分析。

**📈 对比分析**

与之前的随机化方案（Ghaffari‑Parter 2017、Li‑Parter 2019、Jauregui‑Montealegre‑Rapaport 2025）相比，本文在相同的 Õ(D) 轮复杂度下实现了完全确定性；同时，简单的权重转移显著降低了实现复杂度。该分离器可直接作为 BDD（Bounded‑Diameter Decomposition）的黑盒，进一步实现多种经典问题的无随机化近最优算法。

**⚠️ 局限性**

局限性与未来工作：
- 初始假设为双连通图，尽管可通过添加人工边消除，但仍需额外预处理；
- 复杂度为 Õ(D)，仍受图直径限制，对于非常宽松的图可能不如某些线性时间算法；
- 对更一般的图类（非平面图、k‑outerplanar 图等）尚未证明同样可行。

---

## 201. Predicting Multi-Drug Resistance in Bacterial Isolates Through Performance Comparison and LIME-based Interpretation of Classification Models

**arXiv ID:** 2602.22400 | [PDF](https://arxiv.org/pdf/2602.22400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 202. Parallelizable Search-Space Decomposition for Large-Scale Combinatorial Optimization Problems Using Ising Machines

**arXiv ID:** 2602.23038 | [PDF](https://arxiv.org/pdf/2602.23038v1)

**作者:** Eiji Kawase `[一作]` (Oki Electric Industry Co), Shu Tanaka `[通讯]` (Keio University)

**通讯引用:** 1494 | [OpenAlex ID](https://openalex.org/A5057961231)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于约束最大割问题（CMC）的搜索空间分解方法，利用Ising机快速划分变量集合后在子问题上并行求解；

**💡 创新点**

创新点在于将变量划分建模为带平衡约束的最大割问题，并通过量子退火/Ising机实现高速、可扩展的预处理，显著降低主问题规模并提升可行解率；

**🔧 技术方法**

采用Max-Cut/CMC的QUBO形式，使用Simulated Annealing（neal）完成划分，随后用Gurobi求解子问题，并行计算实现加速；

**📊 数据集**

实验使用CVRPLIB中的六个典型CVRP实例（M-n151-k12、M-n200-k17、X-n101-k25、X-n200-k36、X-n261-k13、X-n401-k29）；

**📈 对比分析**

与直接使用Gurobi（Naïve）对比，分解方法在高容量利用率实例可行解率提升至10%–100%，BKS误差从约60%–66%降至3%–9%，收敛速度约提升30倍；

**⚠️ 局限性**

限制在于需要手动调节惩罚系数μ，且在容量几乎饱和（CUR≈1）的实例仍难以获得可行解；此外仅在单机环境验证，未与传统图划分工具进行系统对比。

---

## 203. Causal Motion Diffusion Models for Autoregressive Motion Generation

**arXiv ID:** 2602.22594 | [PDF](https://arxiv.org/pdf/2602.22594v1)

**作者:** Qing Yu `[一作]` (LY Corporation), Kent Fujiwara `[通讯]` (LY Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为CMDM的因果运动扩散模型，实现了文本驱动的人体运动生成，可实现实时流式和长时序合成。

**💡 创新点**

创新点在于将因果潜在编码（MAC‑VAE）、因果扩散强制（Diffusion Forcing）与自回归扩散变换器（Causal‑DiT）相结合，并引入帧级采样调度（FSS），既保持扩散模型的真实性，又保留自回归的因果性和高效性。

**🔧 技术方法**

使用了因果变分自编码器、因果扩散强制、Causal‑DiT、自回归采样、AdaLN 与 ROPE 位置编码、Flow Matching ODE 采样器以及帧级采样调度。

**📊 数据集**

使用了 HumanML3D 与 SnapMoGen 两个公开数据集进行训练与评估。

**📈 对比分析**

与 VQ、扩散和自回归等现有方法在 FID、R‑Precision、CLIP‑Score 等指标上进行对比，CMDM 在保持高语义对齐的同时实现了更低的 FID、更高的 R‑Precision 和更快的推理速度（最高可达 125 fps）。

**⚠️ 局限性**

局限性包括对预训练动作‑语言模型的依赖、在极长序列中可能出现细微时序伪影，以及目前仅针对单人动作，缺乏多角色交互的支持。

---

## 204. Retrieval-Augmented Generation Assistant for Anatomical Pathology Laboratories

**arXiv ID:** 2602.22216 | [PDF](https://arxiv.org/pdf/2602.22216v1)

**作者:** Diogo Pires `[一作]` (Nova Information Management School), Mauro Castelli `[通讯]` (Nova Information Management School)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并评估了一个针对解剖病理实验室的检索增强生成(RAG)助手，能够为技术人员即时提供上下文驱动的协议答案。

**💡 创新点**

系统性比较了不同分块、检索和嵌入策略对实验室协议问答的影响，并公开发布了99份葡萄牙语病理协议及其323条问答对数据集。

**🔧 技术方法**

采用LangChain+ChromaDB搭建检索管道，使用Llama 3.1 8B生成模型，并结合paraphrase‑multilingual‑MiniLM‑L12‑v2与医学专用嵌入模型MedEmbed，配合递归分块与混合检索策略。

**📊 数据集**

使用来自葡萄牙医疗机构的99份解剖病理协议文本及其整理的323条问答对。

**📈 对比分析**

通过RAGAS指标与Top‑k检索评估，对比10种配置，最终递归512‑token分块+混合检索+MedEmbed在答案相关性0.74、可信度0.70、上下文召回0.77，且k=1表现最佳。

**⚠️ 局限性**

评估数据为人工生成的问答对，缺少真实技术人员的非结构化查询；仅覆盖单一机构与葡萄牙语，未验证跨机构或多语言通用性；仅使用小型LLM，硬件受限。

---

## 205. SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport

**arXiv ID:** 2602.23353 | [PDF](https://arxiv.org/pdf/2602.23353v1)

**作者:** Simon Roschmann `[一作]` (Technical University of Munich), Zeynep Akata `[通讯]` (Technical University of Munich)

**通讯引用:** 16199 | [OpenAlex ID](https://openalex.org/A5040372929)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种半监督对齐框架SOTAlign，利用少量配对样本训练线性教师，再通过最优传输（OT）正则化，利用海量未配对的图像和文本数据实现冻结的视觉与语言编码器的联合嵌入对齐。

**💡 创新点**

创新点包括：① 两阶段对齐策略——先用极少配对样本得到线性教师，再用OT‑based KLOT散度对齐未配对样本；② 推导出KLOT的显式梯度公式，消除Sinkhorn迭代的显存瓶颈；③ 通过OT保留关系结构而非单一最近邻，避免过度约束；④ 能在跨数据集、跨域的未配对数据上提升性能。

**🔧 技术方法**

技术细节包括：线性对齐（Procrustes、CCA、线性对比损失）、SigLIP对比损失、KLOT（基于Sinkhorn的OT散度）及其显式梯度、伪标签化正则化、LION优化器和大批量（32k）训练。

**📊 数据集**

使用预训练模型：DINOv3 ViT‑L（视觉）和NV‑Embed‑v2（语言）；配对数据集：CC3M（10k）、CC12M、COCO、ImageNet；未配对数据：CC3M 1M、CC12M、COCO、ImageNet‑1k及合成字幕。

**📈 对比分析**

与S‑AIL、STRUCTURE、NNCLR、S‑CLIP、SUE等监督与半监督基线对比。SOTAlign在COCO上T2I/ I2T Recall@1分别达到26.5%/34.1%（比S‑AIL 21.0%/27.4%提升约+5%），在Flickr30k上达到51.7%/60.8%（比S‑AIL 45.7%/54.1%提升+6%）。在ImageNet零样本分类中，SOTAlign取得46.1% top‑1准确率，显著高于S‑AIL 36.4%。在中等监督量（1k–10k配对）下提升高达+10%。

**⚠️ 局限性**

局限性包括：对预训练编码器的依赖；在极低监督（<100对）下性能下降；对极端分布偏移的鲁棒性有限；需要大批量和显存支持；目前仅验证于视觉‑文本两模态，未扩展到其他模态或多模态组合。

---

## 206. DrivePTS: A Progressive Learning Framework with Textual and Structural Enhancement for Driving Scene Generation

**arXiv ID:** 2602.22549 | [PDF](https://arxiv.org/pdf/2602.22549v1)

**作者:** Zhechao Wang `[一作]` (XPeng Motors), Cheng Lu `[通讯]` (XPeng Motors)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出DrivePTS框架，通过渐进式学习、视觉语言模型多视角分层描述和频率引导结构损失实现可控驾驶场景合成。

**💡 创新点**

创新点包括：①将道路与物体的几何条件分阶段学习，显著降低相互依赖；②利用VLM自动生成六维度多视角细粒度描述，提升语义引导；③引入高频结构损失，增强前景细节与边缘清晰度。

**🔧 技术方法**

采用Stable Diffusion 2.1+T2I‑Adapter实现条件生成，基于Qwen2.5‑VL‑72B进行多视角分层描述，使用高通滤波的频率引导结构损失，以及InfoNCE互信息约束和DDIM采样。

**📊 数据集**

实验数据集为nuScenes，包含360°视角、HD地图、10类物体的3D包围盒等。

**📈 对比分析**

与MagicDrive、Panacea、PerLDiff等基线对比，DrivePTS在FID（11.45）、道路mIoU（63.95）、车辆mIoU（27.82）、检测NDS等指标均超过对手，并在稀有场景生成上表现更优。

**⚠️ 局限性**

局限性在于对极端天气或复杂交叉路口的泛化仍有限，模型对VLM生成的描述质量高度依赖，同时训练仍需大量计算资源。

---

## 207. The Ethos of the PEERfect REVIEWer: Scientific Care and Collegial Welfare

**arXiv ID:** 2602.22292 | [PDF](https://arxiv.org/pdf/2602.22292v1)

**作者:** Oliver Karras `[一作]` `[通讯]` (Leibniz Information Centre for Science and Technology), Oliver Karras (Leibniz Information Centre for Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 PEERfect REVIEWer 的理念与 16 条实用建议，旨在通过科学严谨与同伴福利双重价值提升同行评审质量。

**💡 创新点**

将同伴福利与科学严谨并置为核心价值，构建人性化、同理心驱动的评审框架，并借鉴 Extreme Programming 与 Golden Circle 提供可操作的推荐。

**🔧 技术方法**

采用经验总结、文献综述和概念模型（价值→推荐→行为）进行理论构建；参考 Extreme Programming 与 Golden Circle 进行结构化设计。

**📊 数据集**

无数据集，全部基于作者十年经验、已发表综述和专家访谈所得。

**📈 对比分析**

未进行实验或定量比较；通过示例正负案例和经验论证说明建议的有效性与可行性。

**⚠️ 局限性**

需自愿采纳，受时间压力、文化差异及行业惯例影响；缺乏实证验证与广泛适用性评估。

---

## 208. DisQ-HNet: A Disentangled Quantized Half-UNet for Interpretable Multimodal Image Synthesis Applications to Tau-PET Synthesis from T1 and FLAIR MRI

**arXiv ID:** 2602.22545 | [PDF](https://arxiv.org/pdf/2602.22545v1)

**作者:** Agamdeep S. Chopra `[一作]` (University of Washington), Mehmet Kurt `[通讯]` (University of Washington)

**通讯引用:** 1561 | [OpenAlex ID](https://openalex.org/A5011934020)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究提出 DisQ-HNet，一个用于从 T1‑weighted 与 FLAIR MRI 合成 tau‑PET 的多模态生成框架，并对合成结果进行可解释的信号来源分解。

**💡 创新点**

创新点在于①将 Partial Information Decomposition (PID) 引入向量量化编码器，实现重叠、唯一与互补信息的显式分离；②设计 Half‑UNet 解码器，用伪跳跃（基于结构边缘的多尺度提示）代替传统 skip，保持结构细节同时避免信息混杂；③利用 PID‑Shapley 分析量化各模态对 PET 信号的贡献。

**🔧 技术方法**

核心技术包括向量量化 VQ‑VAE、PID‑guided 信息分解、Pseudo‑skip 半 U‑Net、边缘感知伪跳连接、混合训练损失（MSE、VCC、AG、SPADE）以及 Shapley 归因。

**📊 数据集**

使用 ADNI‑3 与 OASIS‑3 数据集，共计 605 例已预处理的 T1、FLAIR 与 tau‑PET，验证集 83 例，用于 Braak 分期与 SUVR 评估。

**📈 对比分析**

与 VAE、VQ‑VAE、UNet、UNet+SPADE 等基线相比，DQ2H‑MSE‑Inf 在原始 PET MAE 0.104、PSNR 18.53、SSIM 0.861 以及 SUVR MAE 0.292、相对 SUVR 错误 13.1% 处取得最佳平衡；Bland‑Altman 显示 SUVR 偏差仅 -3.3%，Braak 难度准确率软 0.779、硬 0.482，明显优于其他模型。

**⚠️ 局限性**

局限性包括：① SUVR 校准仍存在系统性欠估误差，导致分期阈值偏移；② 模型对个体差异和高阶 Braak 组的误差聚集；③ PID‑Shapley 仅在训练期间可解释，推理阶段无法直接可视化跨模态交互；④ 需要大量 GPU 资源训练量化模型。

---

## 209. Efficient Continual Learning in Language Models via Thalamically Routed Cortical Columns

**arXiv ID:** 2602.22479 | [PDF](https://arxiv.org/pdf/2602.22479v1)

**作者:** Afshin Khadangi `[一作]` `[通讯]` (University of Luxembourg), Afshin Khadangi (University of Luxembourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 TRC² 的解码器架构，通过稀疏的丘脑路由和局部可塑的快速校正通道实现持续学习，在保持原始知识的同时快速适应流式数据。

**💡 创新点**

创新点包括：①将持续学习嵌入网络结构而非仅靠外部微调；②使用丘脑路由的 top‑k 选择结合拓扑先验来保持时间连续性；③加入可调节的激励抑制、预测、记忆与低秩校正路径，实现快速在线更新而不破坏慢速参数；④设计块级稀疏路由与分块并行实现，兼顾效率与可解释性。

**🔧 技术方法**

技术：稀疏丘脑路由 + 选定柱状计算单元；可调节控制器（生成路由、预测混合、全局增益）；预测路径与辅助损失；现代 Hopfield 记忆检索；可选的路由权重细化与低秩校正；块级并行实现与激活检查点；混合精度训练与分布式数据并行。

**📊 数据集**

数据集：用于预训练的 C4 语料（流式与离线两种模式）；评估集为 WikiText‑103、LAMBADA 以及 C4 的验证子集，作为连续学习评测流。

**📈 对比分析**

与参数匹配的 Transformer 与 Mamba 解码器在同一训练管线下对比。TRC² 在 held‑out perplexity 与 BLEU 上与基线相当，同时在流式评测中累计遗忘面积显著下降（例如 PPL 归一化 AUC 下降至 0.011 vs Transformer 0.0000，Mamba 0.0006）。吞吐量略低于 Transformer，但通过块级并行和稀疏路由仍保持可接受的 tokens/s。

**⚠️ 局限性**

局限：在更剧烈的分布漂移、长上下文或频繁切换情形下路由稳定性可能受损；实现依赖高效的核融合与内存布局，当前吞吐量仍低于纯密集 Transformer；缺乏对更大规模模型和更长上下文的实验，且快速校正通道的实时可解释性与可控性尚未充分验证。

---

## 210. Towards Intelligible Human-Robot Interaction: An Active Inference Approach to Occluded Pedestrian Scenarios

**arXiv ID:** 2602.23109 | [PDF](https://arxiv.org/pdf/2602.23109v1)

**作者:** Kai Chen `[一作]` (Tongji University), Guang Chen `[通讯]` (Tongji University)

**通讯引用:** 12490 | [OpenAlex ID](https://openalex.org/A5101684449)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于主动推理的框架，用于在遮挡行人场景下实现安全可解释的自动驾驶决策

**💡 创新点**

创新点包括：①将条件重置机制引入信念更新，保持对潜在遮挡行人的警惕；②假设注入机制在规划阶段引入对极端假设的思考，实现前瞻性避障；③将Rao-Blackwellized粒子滤波与Kalman滤波结合，构建高效的混合状态估计；④结合MPPI-CEM混合规划，兼顾搜索效率与鲁棒性

**🔧 技术方法**

使用的技术有：主动推理框架、Rao-Blackwellized粒子滤波、Kalman滤波、Cross‑Entropy Method、Model Predictive Path Integral、JAX加GPU加速、仿真环境Gymnasium

**📊 数据集**

在论文中未使用公开大规模真实数据集，而是在自建的Gymnasium仿真环境中模拟了五种遮挡行人行为模式，进行600条随机实验来评估性能

**📈 对比分析**

与三类基线（纯反应型、规则型、PPO‑LSTM）比较，指标包括通过率、碰撞率、通过时间、最小距离、最小TTC。实验显示，本框架通过率≈94.7%，碰撞率≈5.3%，优于Reactive（18.8%/82.2%）、Rule‑based（58.7%/41.3%）和PPO‑LSTM（72.5%/27.5%），并在效率与安全性上实现平衡

**⚠️ 局限性**

主要局限：①仅与有限基线对比，缺少更先进交互式方法的评估；②未提供形式化的安全保证，依赖概率模型；③实验仅在理想化仿真中进行，未在真实车辆上验证；④对复杂动态场景的鲁棒性仍需进一步验证

---

## 211. Small Object Detection Model with Spatial Laplacian Pyramid Attention and Multi-Scale Features Enhancement in Aerial Images

**arXiv ID:** 2602.23031 | [PDF](https://arxiv.org/pdf/2602.23031v1)

**作者:** Zhangjian Ji `[一作]` (Shanxi University), Wei Wei `[通讯]` (Shanxi University)

**通讯引用:** 67308 | [OpenAlex ID](https://openalex.org/A5100344384)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在遥感航空图像中提升小目标检测性能，提出SLPA与MSFEM模块并结合DCN改进FPN

**💡 创新点**

引入空间拉普拉斯金字塔注意力模块和多尺度特征增强模块，且在特征金字塔融合时使用可变形卷积对齐，显著提升小目标检测

**🔧 技术方法**

ResNet‑50改进 backbone、Feature Pyramid Network、可变形卷积、卷积注意力、拉普拉斯金字塔、Dilation Conv

**📊 数据集**

VisDrone2019和DOTA‑v1.0 两个遥感图像基准集

**📈 对比分析**

与原 CZ‑Det、ClusterNet、DensityMap 等方法对比，改进后AP提升约1.5‑3.5%，小目标AP提升5‑7%，在保持FPS约11‑12的情况下显著优于多种基线和最新方法

**⚠️ 局限性**

对大目标检测仍有轻微负面影响，计算量与参数略增，模块在非航空图像上的通用性未充分验证

---

## 212. SPARR: Simulation-based Policies with Asymmetric Real-world Residuals for Assembly

**arXiv ID:** 2602.23253 | [PDF](https://arxiv.org/pdf/2602.23253v1)

**作者:** Yijie Guo `[一作]` (Nvidia), Yashraj Narang `[通讯]` (Nvidia)

**通讯引用:** 1224 | [OpenAlex ID](https://openalex.org/A5043295239)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种混合式仿真‑实机适配框架，先在仿真中训练低维状态基准策略，再在真实环境中学习视觉条件的残差策略，实现两件装配任务的几乎完美成功率。

**💡 创新点**

创新点在于：①采用不对称残差学习，将仿真训练的高效低维策略与视觉感知的残差策略融合；②通过基准策略的采样回放自动生成残差标签，避免人工演示；③结合 RLPD 算法与演示缓冲区动态更新，显著提升样本效率。

**🔧 技术方法**

使用的技术包括：仿真训练使用 Isaac Lab + PPO；视觉感知采用 Grounding‑DINO、SAM‑2 与 FoundationPose；残差策略在真实机器人上以 RLPD 训练；机器人控制采用 Cartesian impedance 控制；评估使用自动成功检测器。

**📊 数据集**

主要使用 AutoMate 100 个装配任务（挑选 10 个在仿真中 >99% 成功率的任务）和 NIST 任务板（peg/gear 插件装配）作为实验数据集。

**📈 对比分析**

与 SERL、AutoMate（零射击）以及 HIL‑SERL（人类监督）进行对比。相较 AutoMate，成功率提升 38.4%（平均到 95–100%），循环时间缩短 29.7%。与 HIL‑SERL 相比，无需人工干预即可达到相近的成功率，且在 0.5 小时训练预算下表现更优。

**⚠️ 局限性**

局限性包括：1) 需预先抓取好插头；2) 对基准策略的零射击表现有一定依赖；3) 需要可自动检测成功的奖励或标注，难以在更复杂环境或未知任务中直接应用。

---

## 213. Learning-based Multi-agent Race Strategies in Formula 1

**arXiv ID:** 2602.23056 | [PDF](https://arxiv.org/pdf/2602.23056v1)

**作者:** Giona Fieni `[一作]` (Institute for Dynamic Systems and Control), Christopher H. Onder `[通讯]` (Institute for Dynamic Systems and Control)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本文提出一种基于强化学习的多智能体 F1 赛车策略优化框架，利用交互模块和自我博弈训练实现能量管理、加油/更换轮胎与进站决策的协同优化。

**💡 创新点**

核心创新在于将已有的单智能体策略冻结后加入交互模块，仅学习补偿项；再结合自我博弈和 Elo 排位评估，系统化地训练与评估多智能体竞赛策略。

**🔧 技术方法**

使用深度强化学习（SAC 算法）与预训练的单智能体网络、交互模块网络、马尔科夫决策过程、以及 Elo 排位系统。

**📊 数据集**

训练与评估均在仿真环境中完成，依托 ETH Zürich 研发的 F1 赛车动力学模型与轮胎磨损模型；未使用真实比赛数据集。

**📈 对比分析**

通过与其他训练好的智能体（A、B、C、D）对决，A 通过两停策略在不同对手下均表现出色，赛时误差仅差几秒；Elo 排位与赛时差异高度相关，证明方法有效。

**⚠️ 局限性**

局限性包括仅考虑两车交互、缺乏车流与随机事件（如安全车、天气）、仅在单一赛道模拟、且模型参数对不同赛道的适应性未知。

---

## 214. SQaLe: A Large Text-to-SQL Corpus Grounded in Real Schemas

**arXiv ID:** 2602.22223 | [PDF](https://arxiv.org/pdf/2602.22223v1)

**作者:** Cornelius Wolff `[一作]` (University of Amsterdam), Madelon Hulsebos `[通讯]` (Centrum Wiskunde & Informatica)

**通讯引用:** 468 | [OpenAlex ID](https://openalex.org/A5058441702)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为SQaLe的半合成文本到SQL数据集，包含13.6万条数据库模式和51.8万条（问题、模式、查询）三元组。

**💡 创新点**

创新点在于将大量真实SchemaPile数据库模式与LLM驱动的自动扩展、问题生成和SQL验证相结合，保证了模式规模、规范化、外键完整性以及查询的多样性和执行有效性。

**🔧 技术方法**

采用Qwen3 30B LLM进行模式扩展与问题合成，使用ReFoRCE框架生成并投票挑选SQL，随后在真实模式上执行验证。

**📊 数据集**

核心数据集为SchemaPile真实模式（22,989个），以及生成的SQaLe数据集；在实验中对比了BIRD、Spider 2.0、SynSQL-2.5M等基准。

**📈 对比分析**

通过对比统计指标，SQaLe在模式规模、列数、外键数量、查询连接复杂度等方面均超过现有数据集；在现有基准上训练的模型可获得更高的执行准确率，验证了规模与性能的正相关。

**⚠️ 局限性**

局限性包括：查询量仍低于SynSQL-2.5M，生成的自然语言问题依赖LLM的生成质量，可能存在语义偏差；未在真实业务场景中进行完整的用户评测。

---

## 215. Assessing Deanonymization Risks with Stylometry-Assisted LLM Agent

**arXiv ID:** 2602.23079 | [PDF](https://arxiv.org/pdf/2602.23079v1)

**作者:** Boyang Zhang `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于LLM的作者去匿名化评估与缓解框架，利用SALA技术对新闻文本进行作者识别并提供重写建议。

**💡 创新点**

创新点在于将定量文体特征与LLM推理相结合的Stylometry-Assisted LLM Analysis（SALA），并通过数据库缓存和引导式重构实现可解释且精确的作者归属与去匿名化防护。

**🔧 技术方法**

采用GPT‑4.1‑2025‑04‑14等大语言模型、LangChain框架、Python NLP工具（如NLTK）、网页搜索API及数据库模块，SALA将量化文体特征与LLM推理融合。

**📊 数据集**

实验使用All the News 2.0和CrossNews这两个大型英文新闻数据集。

**📈 对比分析**

与Embedding Similarity、LLM Direct Analysis（LDA）等基线对比，SALA在有数据库支持的目标攻击场景中F1最高达0.827，在开放世界场景下Top‑3准确率提升至约22%；引导式重写后攻击成功率显著降低。

**⚠️ 局限性**

局限包括仅在英文新闻数据上验证，缺乏跨领域或多语言适用性；重写质量与风格保持尚待提升；数据库可能带来作者代表性偏差。

---

## 216. Decomposing Physician Disagreement in HealthBench

**arXiv ID:** 2602.22758 | [PDF](https://arxiv.org/pdf/2602.22758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 217. CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection

**arXiv ID:** 2602.22621 | [PDF](https://arxiv.org/pdf/2602.22621v1)

**作者:** Boyang Dai `[一作]` (University of Hong Kong), Yizhou Yu `[通讯]` (University of Hong Kong)

**通讯引用:** 19162 | [OpenAlex ID](https://openalex.org/A5108557359)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于DETR的源无源域适应目标检测框架CGSA，利用层级槽注意力和类引导槽对比实现跨域迁移。

**💡 创新点**

创新点在于首次将对象中心学习（Slot Attention）引入源无源域适应，构建层级槽先验并通过类对比学习引导槽向域不变语义对齐。

**🔧 技术方法**

采用Slot Attention、层级槽分解、类引导槽对比（InfoNCE）、教师-学生EMA、DETR框架等技术。

**📊 数据集**

实验涵盖Cityscapes→BDD100K、Cityscapes→Foggy-Cityscapes、Sim10K→Cityscapes、KITTI→Cityscapes等多个公开数据集。

**📈 对比分析**

与现有SF-DAOD方法相比，CGSA在mAP上提升约15%，并在传统DAOD方法上平均提高约10%，表现最优。

**⚠️ 局限性**

局限性包括仅适用于基于查询的DETR检测器，未验证在两阶段或非查询模型上的可迁移性；仅针对目标检测，未扩展到分类或分割等任务。

---

## 218. AdapTBF: Decentralized Bandwidth Control via Adaptive Token Borrowing for HPC Storage

**arXiv ID:** 2602.22409 | [PDF](https://arxiv.org/pdf/2602.22409v1)

**作者:** Md Hasanur Rashid `[一作]` (University of Delaware), Dong Dai `[通讯]` (University of Delaware)

**通讯引用:** 696 | [OpenAlex ID](https://openalex.org/A5012002926)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了AdapTBF，一个去中心化的带宽控制框架，通过自适应令牌借贷机制在HPC存储环境中动态平衡各计算任务的I/O带宽。

**💡 创新点**

将传统令牌桶过滤器（TBF）与去中心化控制相结合，并引入自适应借贷机制，使得在短暂高峰期可临时溢出，同时保证公平与整体存储利用率最大化。

**🔧 技术方法**

采用令牌桶过滤器、去中心化控制、动态令牌借贷逻辑，并在Lustre文件系统中实现相应模块。

**📊 数据集**

使用基于真实工作负载场景合成的I/O测试集（synthetic workloads modeled after real-world scenarios）。

**📈 对比分析**

与传统按比例限制的TBF方案进行实验对比。实验结果显示，AdapTBF在保持或提升每个应用I/O性能的同时，整体存储服务器利用率更高，尤其在极端负载条件下仍保持较好的性能。

**⚠️ 局限性**

目前验证仅在Lustre环境下完成，缺乏在其他文件系统或大规模集群的广泛测试；实现复杂度较高，且对极短burst I/O的即时响应仍有限。

---

## 219. No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings

**arXiv ID:** 2602.22689 | [PDF](https://arxiv.org/pdf/2602.22689v1)

**作者:** Joonsung Jeon `[一作]` (Korea Advanced Institute of Science and Technology), Sung-Eui Yoon `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3864 | [OpenAlex ID](https://openalex.org/A5078173428)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对文本到图像潜在扩散模型的无标题（caption-free）成员推断攻击框架MoFit，能够在不依赖真实或 VLM 生成的标题的情况下识别训练样本。

**💡 创新点**

创新点在于利用“模型拟合替代体（Model‑Fitted surrogate）”与其对应的“紧耦合嵌入（Model‑Fitted embedding）”两阶段优化，制造图像与条件的误配，使成员样本的条件损失显著升高而非成员样本变化不大，从而显著提升分离度。

**🔧 技术方法**

核心技术包括：潜在扩散模型（Stable Diffusion 及其微调版本）、无条件噪声匹配的替代体优化、基于条件噪声损失的嵌入优化、以及将两者结合的差分损失评分（CLiD‑style）与稳健归一化的决策阈值。

**📊 数据集**

实验使用三类微调数据集（Pokemon、MS‑COCO、Flickr）以及公开的 Stable Diffusion v1.4/v1.5，改进的 LAION‑mi 基准，所有数据均为公开可获得。

**📈 对比分析**

与多种基线（Loss‑based, SecMI, PIA, PFAMI, CLiD）比较时，MoFit 在无标题设置下的攻击成功率、AUC 与 TPR@1%FPR 均远超使用 VLM 标题的对手，在 MS‑COCO 上甚至超过了使用真实标题的 CLiD，显示出在高精度区间的显著优势。

**⚠️ 局限性**

主要限制包括：对每张查询图像的两阶段优化耗时较长（约 7‑9 分钟），若需实时或大规模部署需采用提前停止或加速策略；此外，当模型使用低秩适配（LoRA）等小幅改动后，MoFit 的效果会显著下降。

---

## 220. FlashOptim: Optimizers for Memory Efficient Training

**arXiv ID:** 2602.23349 | [PDF](https://arxiv.org/pdf/2602.23349v1)

**作者:** Jose Javier Gonzalez Ortiz `[一作]` (Databricks AI Research), Davis Blalock `[通讯]` (Databricks AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套名为FlashOptim的优化器库，能在保持模型精度的前提下，将训练时每个参数的内存占用降至50%以上。

**💡 创新点**

关键创新是基于ULP的权重拆分，实现24位高效主权重存储；以及设计可逆的压缩函数（companding），让AdamW等优化器的动量和方差状态压缩到8位而不影响收敛。

**🔧 技术方法**

采用权重拆分、companding量化、梯度释放、融合优化器step内核（Triton）等技术，兼容PyTorch FSDP等分布式方案。

**📊 数据集**

在ResNet-50/ILSVRC-2012、GPT-2/FineWeb、Llama-3.1-8B/OpenMathInstruct、GSM8k等视觉与语言基准上进行验证。

**📈 对比分析**

通过与标准AdamW/SGD/Lion在相同超参下对比，发现内存降低约60%（AdamW每参数7字节），但吞吐量基本相同；训练精度与验证指标几乎无差异。

**⚠️ 局限性**

主要局限在于对激活内存占比高的模型效果有限，量化对某些任务可能产生不稳定，且在极端稀疏或极小更新情况下24位精度不一定足够。

---

## 221. CARAT: Client-Side Adaptive RPC and Cache Co-Tuning for Parallel File Systems

**arXiv ID:** 2602.22423 | [PDF](https://arxiv.org/pdf/2602.22423v1)

**作者:** Md Hasanur Rashid `[一作]` (University of Delaware), Dong Dai `[通讯]` (University of Delaware)

**通讯引用:** 696 | [OpenAlex ID](https://openalex.org/A5012002926)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出了一种基于机器学习的客户端自适应 RPC 与缓存协同调优框架 CARAT，能够在不需要全局监控的情况下，仅利用本地可观测指标在运行时对 Lustre 客户端的 RPC 窗口、并发数和脏缓存大小进行动态调节。

**💡 创新点**

创新点包括：①仅使用客户端本地指标，避免昂贵的全局采样；②将 RPC 与缓存调优拆分为两阶段，充分利用它们不同的响应速度；③采用条件分数贪婪（Conditional‑Score‑Greedy）策略在机器学习模型给出的高置信度候选中进一步筛选最优配置；④实现完全去中心化、在线、即时调优，具备良好的可扩展性。

**🔧 技术方法**

核心技术：梯度提升决策树（GBDT）作为分类模型预测配置是否提升性能；基于本地计数器的特征工程；两阶段调优架构（Stage‑1 RPC、Stage‑2 缓存）；规则驱动的缓存分配；Python/C++实现的低开销控制循环。

**📊 数据集**

训练与评估数据集：使用 Filebench 合成工作负载（单/多流、读写、顺序/随机、不同 I/O 大小）收集 200k 条样本；真实工作负载包括深度学习 I/O（BERT、Megatron/DeepSpeed）和科学 HPC（H5bench、VPIC、BDCATS）在 CloudLab 集群上跑取。

**📈 对比分析**

与方法比较：将 CARAT 与默认 Lustre 配置和离线全局最优配置进行对比。实验显示，CARAT 在 24 个 Filebench 工作负载中能匹配或超过 90% 的最优吞吐率，平均提升 1.8–3.0 倍；在动态变化和多客户端情形下保持近最优；在 DL 与 HPC 实际应用中，平均提升 15–30%（DL 可达 1.75×），而在纯顺序工作负载上提升有限。

**⚠️ 局限性**

局限性：①需要在训练阶段先收集大量本地指标，且模型对新工作负载的泛化能力有限；②对元数据路径和其他 RPC 基础文件系统的适用性尚未验证；③需要 root 权限修改客户端参数，部署门槛相对较高；④在极端高负载或突发网络抖动时仍可能出现调优波动；⑤目前仅在 Lustre 上实现，其他文件系统的迁移需要重新设计指标与参数。

---

## 222. OmniZip: Learning a Unified and Lightweight Lossless Compressor for Multi-Modal Data

**arXiv ID:** 2602.22286 | [PDF](https://arxiv.org/pdf/2602.22286v1)

**作者:** Yan Zhao `[一作]` (Shanghai Jiao Tong University), Li Song `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 83401 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本工作提出了OmniZip，一种统一且轻量化的无损压缩器，能够同时处理图像、医学影像、触觉信号、文本、基因序列、数据库以及语音等多种模态的数据；

**💡 创新点**

主要创新点包括：①统一令牌化（Modality‑Unified Tokenizer）实现不同模态的可逆离散化；②模态路由上下文学习（Modality‑Routing Context Learning）在时间混合模块中对V投影层引入Mixture‑of‑Experts；③模态路由前馈（Modality‑Routing Feedforward）在前馈网络中同样使用MoE；④重参数化训练（Reparameterization Training）在训练阶段扩增模型容量，推理时保持轻量化；

**🔧 技术方法**

使用了轻量化RWKV‑7 backbone，Mixture‑of‑Experts技术（在V层与前馈MLP中实现稀疏专家路由），重参数化训练策略，算术编码以及可逆的模态统一token化方法；

**📊 数据集**

实验基于16个数据集，涵盖七种模态：自然图像（DIV2K、Kodak、CLIC）、医学影像（MRNet Axial/Coronal/Sagittal）、触觉信号（TouchandGo、ObjectFolder）、文本（enwik8/9、Gutenberg、Spider/WikiSQL）、基因序列（GenoSeq、DNACorpus）、语音（LibriSpeech）；

**📈 对比分析**

通过与经典通用压缩器（gzip、bzip2、zstd）、专用图像压缩器（PNG、WebP、FLIF、JPEG‑XL、JPEG2000、BPG）、学习式压缩器（Llama3、RWKV、DLPR、L3C、RC、P2LLM、tszip、NNCP、CMIX、L3TC）等进行对比。OmniZip在各模态上相较于gzip提升约30–60%，在图像、文本、基因、语音等任务上接近或优于SOTA，同时参数量仅几百k至几百万，推理速度在MacBook CPU和iPhone NPU上可达≈1 MB/s；

**⚠️ 局限性**

局限性：目前对极大文件的压缩速度仍有限；在语音模态上仍落后专用压缩器如FLAC；路由策略和重参数化训练在不同模态间的最优配置尚待进一步探索；缺乏针对特定模态的预训练知识，未来可通过多模态预训练提升性能。

---

## 223. Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space

**arXiv ID:** 2602.22879 | [PDF](https://arxiv.org/pdf/2602.22879v1)

**作者:** Xingcheng Fu `[一作]` (Guangxi Normal University), Dongran Yu `[通讯]` (Guangxi Normal University)

**通讯引用:** 104 | [OpenAlex ID](https://openalex.org/A5040192580)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于大语言模型的双代理（教师-学生）与超曲率对齐的知识追踪框架 L-HAKT，利用教师代理构建层级知识图谱，学生代理生成模拟学习数据，并在超曲率空间中进行对齐学习。

**💡 创新点**

创新点在于：①将大语言模型用于语义解析和层级知识抽取，构建结构化知识图谱；②通过超曲率（负曲率）空间实现知识层级的显式嵌入；③采用对比学习将合成数据与真实数据在超曲率空间对齐，校正分布差异并提升个性化难度感知。

**🔧 技术方法**

技术手段包括：大语言模型（如 Qwen‑2.5VL）进行文本与图像语义解析与模拟学习行为；超曲率图神经网络（HGNN）结合曲率优化进行层级嵌入；对比学习在超曲率空间对齐合成与真实数据；LSTM/GRU 处理时间序列并融合正确率与层级信息。

**📊 数据集**

实验数据集涵盖四大公开教育数据集：ASSIST2009、ASSIST2012、EdNet 以及含图像的 Eedi，其中 Eedi 通过 LLM 进行图像转文本并生成模拟交互数据。

**📈 对比分析**

在 Accuracy 与 AUC 上与 20+ 传统序列模型（如 DKT、ATKT、DIMKT）和图模型（如 GKT、GIKT）进行对比，L-HAKT 在四个数据集上均获得最高分，AUC/ACC 分别提升约 5%–10% 以上，尤其在高级知识点的学习曲线捕获上表现最突出。

**⚠️ 局限性**

局限性包括：对大语言模型的高计算成本和依赖，超曲率嵌入对曲率选择敏感，需要精细调参；合成数据可能带来标签噪声或偏差；实验仅覆盖四个数据集，缺乏对不同学科或跨文化场景的泛化验证。

---

## 224. Fair feature attribution for multi-output prediction: a Shapley-based perspective

**arXiv ID:** 2602.22882 | [PDF](https://arxiv.org/pdf/2602.22882v1)

**作者:** Umberto Biccari `[一作]` (University of Deusto), Enrique Zuazua `[通讯]` (Friedrich Alexander University Erlangen Nurnberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多输出预测的Shapley解释框架下，对特征归因进行公理化表征，并证明任何满足效率、对称性、虚拟玩家和可加性公理的归因规则必然按输出维度分解；并通过生物医学数据验证多输出模型与单输出模型在SHAP解释上的一致性。

**💡 创新点**

首次证明在向量值合作博弈中，经典Shapley公理强制导致归因规则只能按输出坐标独立工作，否则至少违背一个公理；给出唯一性与稳定性定理，为多输出SHAP提供了严谨的理论依据。

**🔧 技术方法**

利用向量值合作博弈的Shapley值公理、可加性与效率等公理的严格推导，结合深度学习多任务网络与DeepExplainer近似算法；同时使用线性代数与凸分析证明稳定性和 Lipschitz 上界。

**📊 数据集**

使用一份包含15931例、75维临床代谢指标的生物医学表格数据集，目标为年龄、MetSCORE与性别的三输出回归/分类任务。

**📈 对比分析**

通过比较多输出模型（M2）与对应的单输出模型，在回归指标（R²、RMSE、MAE）与分类指标（ACC、AUC、F1）以及训练时间/显存消耗进行评估；在SHAP解释上计算余弦相似度与Spearman相关系数，结果显示余弦相似度>0.98、Spearman>0.86，证明两种模型的特征重要性高度一致。

**⚠️ 局限性**

研究仅针对经典Shapley公理，无法探讨可耦合输出的归因方法；未提出新的解释算法，仅给出理论约束；实验局限于单一医学数据集，未验证跨领域普适性；缺乏对其他公平性公理或多目标优化框架的探讨。

---

## 225. Evaluating and Improving Automated Repository-Level Rust Issue Resolution with LLM-based Agents

**arXiv ID:** 2602.22764 | [PDF](https://arxiv.org/pdf/2602.22764v1)

**作者:** Jiahong Xiang `[一作]` (Research Institute of Trustworthy Autonomous Systems Southern University of Science and Technology), Yuqun Zhang `[通讯]` (Research Institute of Trustworthy Autonomous Systems Southern University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个包含 500 个真实 Rust 仓库级问题的 benchmark，系统评估了四种 LLM 代理在此基准上的表现，并提出了结合自动化测试环境和跨项目动态追踪的新型代理框架；

**💡 创新点**

①首次构建大规模、真实、仓库级 Rust 问题基准；②设计了自动化测试环境与 Rust 过程宏驱动的跨项目动态追踪机制；③通过实验验证新框架相较现有代理显著提升；

**🔧 技术方法**

采用 ReAct 样式代理架构、Rust 过程宏自动插桩、隔离 Cargo 工作区、交互式 LLM 代理（Claude‑3、GPT‑4o、o4‑mini、qwen3‑235b）、动态追踪命令（OpenTrace）等技术；

**📊 数据集**

自研的 500 任务集，来源于 34 个受欢迎的 Rust 开源仓库，基于 PR‑issue 关联、测试变更、fail‑to‑pass 验证构建；

**📈 对比分析**

对四种代理与四种 LLM 组合进行 Pass@1 评估，最优配置（OpenHands+Claude‑3）解决 143/500（28.6%），比最佳基线（21.2%）提升 34.9%；同时在成本、重现率等指标上也优于基线；

**⚠️ 局限性**

仍存在对复杂依赖/构建环境的处理不足、对 Rust 语义推断的精确度不足、仅覆盖已合并 PR 的 bug/feature 任务，未涵盖更广泛的错误类型和更大规模的项目。

---

## 226. Risk-Aware World Model Predictive Control for Generalizable End-to-End Autonomous Driving

**arXiv ID:** 2602.23259 | [PDF](https://arxiv.org/pdf/2602.23259v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 227. MammoWise: Multi-Model Local RAG Pipeline for Mammography Report Generation

**arXiv ID:** 2602.22462 | [PDF](https://arxiv.org/pdf/2602.22462v1)

**作者:** Raiyan Jahangir `[一作]` (University of California), Vladimir Filkov `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了MammoWise——一种可本地化、多模型、可扩展的管道，用于将开源视觉语言模型（VLM）转化为结构化乳腺X线报告生成器及多任务分类器。

**💡 创新点**

创新点在于将提示工程、检索增强生成（RAG）与参数高效微调（QLoRA）统一到同一框架，支持任意Ollama托管VLM与任意乳腺影像数据集的实验与比较。

**🔧 技术方法**

技术主要包括零/少/链式思考提示、基于向量数据库的RAG、以及对MedGemma的QLoRA微调，辅以多模态嵌入、ChromaDB检索与标准化输出解析。

**📊 数据集**

实验数据集涵盖VinDr‑Mammo（约5,000例合成图）和DMID（510例带报告），分别用于评估报告质量与分类准确性。

**📈 对比分析**

通过BERTScore、ROUGE‑L评估报告生成；通过宏平均准确率、F1等指标评估BI‑RADS、密度、异常等分类；结果显示：RAG提升文本相似度，QLoRA显著提高分类性能（BI‑RADS 0.7545，密度 0.8840，钙化 0.9341），优于现有SOTA。

**⚠️ 局限性**

局限性包括：仅测试三种VLM，未探索更复杂提示或大模型；仅单图实验，缺乏多时间点与临床信息；微调采用类别重平衡，可能导致校准偏移；RAG效果依赖检索质量，存在潜在文本干扰。

---

## 228. dLLM: Simple Diffusion Language Modeling

**arXiv ID:** 2602.22661 | [PDF](https://arxiv.org/pdf/2602.22661v1)

**作者:** Zhanhui Zhou `[一作]` (University of California Berkeley), Dawn Song `[通讯]` (University of California Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并发布了统一的Diffusion Language Model（DLM）训练、推理和评测框架dllm，并提供可复现的小型DLM训练脚本和检查点；

**💡 创新点**

通过模块化设计实现了训练器、采样器和评测接口统一，支持快速替换与扩展；同时提供了轻量级的BERT/ARLM到DLM的转换方案；

**🔧 技术方法**

Diffusion language modeling（MDLM、BD3LM）、Fast-dLLM加速解码、HF Accelerate/DeepSpeed、LoRA、统一评测 harness；

**📊 数据集**

LLaDA、Dream公开权重、s1K指令数据集、Tulu 3、SmolTalk、MMLU、BBH、MATH、GSM8K、HumanEval等；

**📈 对比分析**

采用统一评测管线与官方配置对比，复现官方分数；Fast-dLLM实现数倍推理速度提升；SFT提升LLaDA、Dream推理能力，BERT-Chat在多数基准上优于同规模ARLM；

**⚠️ 局限性**

规模较大模型性能仍落后于原始ARLM，缺乏RL或对话优化的支持；框架对非常规模型兼容性待进一步完善。

---

## 229. MobilityBench: A Benchmark for Evaluating Route-Planning Agents in Real-World Mobility Scenarios

**arXiv ID:** 2602.22638 | [PDF](https://arxiv.org/pdf/2602.22638v1)

**作者:** Zhiheng Song `[一作]` (Computer Network Information Center, Chinese Academy of Sciences), Hengshu Zhu `[通讯]` (Computer Network Information Center, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 MobilityBench 基准和评估框架，用于系统评估基于大型语言模型（LLM）的路由规划代理在真实移动场景中的表现。

**💡 创新点**

创新点包括：①基于 100k 条匿名 AMap 用户查询构建覆盖 350+ 城市、11 类意图的多场景基准；②引入确定性 API 重放沙盒，消除实时地图服务的非确定性，保证可重复性；③设计多维评估协议，细化指令理解、规划、工具使用、决策质量和效率等维度；④开放数据、工具和评估代码，支持快速扩展与公平比较。

**🔧 技术方法**

使用的技术主要有：LLM 驱动的工具增强代理（ReAct 与 Plan‑and‑Execute）、API 重放沙盒、结构化查询意图分类、标准化工具程序、自动化评估脚本；后端模型涵盖 GPT、Claude、Gemini、Qwen、DeepSeek 等多种开源与闭源 LLM。

**📊 数据集**

数据集：由 AMap 收集的 100,000 条匿名真实用户查询，覆盖 22 个国家、350+ 城市；按 11 个任务场景分类后抽样 7,098 条用于评估；包含信息检索、路程信息检索、基本路径规划和偏好约束规划等四大任务族。

**📈 对比分析**

比较方法：对每个模型在四大任务族中使用指令理解（意图识别、信息抽取）、规划（任务拆解）、工具使用（工具选择、模式符合度）以及决策质量（交付率、最终通过率）和效率（输入/输出 token）六个维度进行评估。实验结果显示：闭源模型 Claude‑Opus‑4.5 与 Gemini‑3‑Pro‑Preview 在多数维度上领先；ReAct 框架在最终通过率上优于 Plan‑and‑Execute，但消耗更多输入 token；大型 MoE 模型 Qwen‑235B‑A22B 在开放源模型中表现最佳；开启思考模式能显著提升通过率，但伴随较高的 token 与延时。

**⚠️ 局限性**

局限性：①仅评估 ReAct 与 Plan‑and‑Execute 两种框架，未覆盖 LATS、Tree‑of‑Thought 等新型代理；②沙盒仅模拟 AMap API，缺乏跨平台兼容性评估；③假设不允许用户澄清，导致对模糊或不完整查询的处理能力难以评估；④数据主要来自中文与中国地图服务，跨语言、跨地区的普适性仍待验证。

---

## 230. pQuant: Towards Effective Low-Bit Language Models via Decoupled Linear Quantization-Aware Training

**arXiv ID:** 2602.22592 | [PDF](https://arxiv.org/pdf/2602.22592v1)

**作者:** Wenzheng Zhang `[一作]` (Peking University), Bin Cui `[通讯]` (Peking University)

**通讯引用:** 13111 | [OpenAlex ID](https://openalex.org/A5062357883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从零开始训练极低比特（<2 位）的大语言模型，提出 pQuant 方案以解决 1 位量化模型中参数敏感性均质化（参数民主化）问题。

**💡 创新点**

创新点包括：① 将线性层拆分为 1 位主干分支和高精度分支；② 通过可学习的特征缩放引导敏感参数聚焦于高精度分支；③ 将高精度分支扩展为多专家并通过轻量路由器动态激活，兼顾容量与推理效率。

**🔧 技术方法**

采用的技术：极低比特量化（1‑bit 主干 + 8‑bit 高精度分支），QAT‑from‑scratch 训练，直通估计（STE），特征缩放与可学习比例因子，轻量化路由器实现稀疏专家激活，多分支架构等。

**📊 数据集**

使用数据集：C4、Wikipedia、ArXiv 进行大规模预训练；在多项下游任务（ARC‑E、ARC‑C、BoolQ、PIQA、Winogrande、OpenbookQA、Hellaswag）上评测零样本性能。

**📈 对比分析**

对比方法：与 1‑bit BitNet、2‑bit BitNet1.58、FP16 LLaMA‑3、PTQ1.61、OmniQuant、OneBit 等在相同规模与训练数据下对比。pQuant 在 1.3B 规模下平均准确度提升约 2–4 点，近似 FP16 结果；在更大规模（2.6B）时表现超过 FP16 LLaMA‑2，且在相同或更小参数量下保持更高的推理吞吐率。

**⚠️ 局限性**

局限性：训练成本显著高于传统 QAT/PTQ；多专家配置会增加物理内存占用；实验仅覆盖至 2.6B 参数，缺乏对更大模型（如 70B）的验证。

---

## 231. Generative Agents Navigating Digital Libraries

**arXiv ID:** 2602.22529 | [PDF](https://arxiv.org/pdf/2602.22529v1)

**作者:** Saber Zerhoudi `[一作]` (University of Passau), Michael Granitzer `[通讯]` (University of Passau)

**通讯引用:** 3644 | [OpenAlex ID](https://openalex.org/A5006866152)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Agent4DL，一个基于大型语言模型（LLM）的数字图书馆用户搜索行为模拟器，能够生成真实、多样化的搜索会话数据。

**💡 创新点**

创新点在于：①结合学术特征与研究兴趣的用户画像模块、记忆模块与交互模块，构建更具个性化和情感的模拟行为；②利用 LLM 的 ReAct/Chain‑of‑Thought 推理实时生成查询、点击与停止行为；③与传统模拟器相比，产生更具上下文感知和多样化的用户行为。

**🔧 技术方法**

核心技术：大型语言模型（ChatGPT/GPT‑4）+ ReAct/Chain‑of‑Thought 推理；LLM 生成文档摘要与主题归一化；情感记忆与事实记忆机制；RoBERTa 排序模型用于评估生成数据的有效性；通过 API 与 HathiTrust、EconBiz 等数字图书馆交互。

**📊 数据集**

使用的数据集：EconBiz 与 Sowiport 的真实用户搜索会话数据（SUSS）作为基准与用户画像初始化；Agent4DLData 为本研究生成的模拟搜索会话数据集。

**📈 对比分析**

对比方法：与 SimIIR 2.0、BM25、Complex Searcher Model 等基线模型以及传统点击/停止策略对比；评估指标包括 MRR、nDCG、Term Overlap Rate (τ)、BLEU、BERTScore、准确率/召回率/F1。实验结果显示，Agent4DL 在偏好预测、相关性预测、查询、点击和停止行为上均优于或与基线相当，尤其在查询相似度上达到 87.6% 的 τ，并在 MRR、nDCG 上显著提升。

**⚠️ 局限性**

局限性：①依赖丰富的资源元数据，缺乏对引用链、同行交流、浏览等学术信息寻求路径的完整建模；②LLM 在专业知识和最新研究方面表现不佳，可能生成时效性不足或领域偏差的查询；③缺乏对位置偏好和深度筛选等细粒度点击行为的捕捉。

---

## 232. SimpleOCR: Rendering Visualized Questions to Teach MLLMs to Read

**arXiv ID:** 2602.22426 | [PDF](https://arxiv.org/pdf/2602.22426v1)

**作者:** Yibo Peng `[一作]` (Carnegie Mellon University), Huaxiu Yao `[通讯]` (University of North Carolina Chapel Hill)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对多模态大型语言模型的视觉文本识别能力进行诊断，并提出视觉化问题（VQ）训练策略，强制模型在推理中主动读取图像中的文字；

**💡 创新点**

创新点在于通过随机化文本渲染构造 VQ 上下文，彻底消除文字通道捷径，使模型真正利用 OCR 能力；

**🔧 技术方法**

使用视觉化文本渲染（T_render）、GRPO 强化学习框架、随机化字体/颜色/大小等数据增强技术，且无需改动模型结构；

**📊 数据集**

训练数据集为 Geometry3K 与 MMK12（共 8.5K 条），评估数据集包括 MathVerse、MathVision、MathVista、HallusionBench、ChartQA、InfographicVQA 等多种 OOD 任务；

**📈 对比分析**

与基线、GRPO、GRPO+NoisyRollout 等方法对比，SimpleOCR 在 ID 与 OOD 上平均提升约 5.4%/2.7%，在仅 8.5K 样本下即可超越需要 260K 样本的 RL 方法，且兼容 NoisyRollout 等高级训练策略；

**⚠️ 局限性**

方法仅能激活已有 OCR 能力，受视觉分辨率限制，无法处理极长文本，且依赖基础模型已具备较强的 OCR 预训练能力。

---

## 233. Instruction-based Image Editing with Planning, Reasoning, and Generation

**arXiv ID:** 2602.22624 | [PDF](https://arxiv.org/pdf/2602.22624v1)

**作者:** Liya Ji `[一作]` (Hong Kong University of Science and Technology), Qifeng Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10211 | [OpenAlex ID](https://openalex.org/A5100719529)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态链式思考（Multimodal Chain-of-Thought, MCoT）框架，分为规划、推理和生成三步，实现基于自然语言指令的图像编辑；

**💡 创新点**

①将多模态LLM与链式思考结合，自动生成子提示和编辑区域；②训练多模态LLM推理网络预测编辑掩模；③在提示引导的编辑网络中将前景/背景图像作为空间条件；④扩展至Flux模型，形成统一的可解释、分步骤编辑流程；

**🔧 技术方法**

使用多模态LLM（LLAVA‑7B、DeepSeek Reasoning Model）、链式思考（CoT）规划、SAM+LoRA编辑区域生成、提示引导Stable Diffusion U‑Net（包含前景/背景条件）、Classifier‑free guidance、Prompt‑to‑Prompt、ControlNet及Flux+ControlNet等技术；

**📊 数据集**

MagicBrush 数据集（训练/测试，含掩模），HQEdit‑Abstract（100个抽象概念样本）；训练集通过数据增强扩展至约78k对；

**📈 对比分析**

与 InstructPix2Pix、InstructDiffusion、MagicBrush、HIVE、HQEdit 等基线在 MagicBrush 与 HQEdit‑Abstract 上对比，采用 CLIP‑I、DINO‑I、CLIP‑T（全局/局部）等指标；实验表明在 MagicBrush 上实现 SOTA，CLIP‑T(Local) 与 CLIP‑I 明显提升；在 HQEdit‑Abstract 上用户评测抽象概念得分较高，编辑质量略有下降；

**⚠️ 局限性**

编辑区域推理仍不够精准，导致多步骤生成后质量下降；多条件提示引导在提升控制力的同时可能降低生成多样性；LLM链式思考的推理能力仍有提升空间；目前模型验证主要集中在 MagicBrush 与 Flux 的小规模实验，需进一步验证在更大规模数据与不同模型上的通用性。

---

## 234. Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache

**arXiv ID:** 2602.22654 | [PDF](https://arxiv.org/pdf/2602.22654v1)

**作者:** Bowen Cui `[一作]` (Alibaba Group), Pipei Huang `[通讯]` (Alibaba Group)

**通讯引用:** 1639 | [OpenAlex ID](https://openalex.org/A5059615376)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的扩散采样加速框架 DPCache，通过构造路径感知成本张量并使用动态规划选取关键时间步，实现全局最优采样路径。

**💡 创新点**

创新点在于将采样加速视为全局路径规划问题，利用路径感知成本张量（PACT）捕捉跳步误差与前一步的依赖，并通过动态规划获得最小累计误差的关键步序列，打破了以往固定或局部自适应的调度局限。

**🔧 技术方法**

采用路径感知成本张量、动态规划、特征缓存与预测（如Taylor系列展开）、少量校准样本计算全局调度方案；仅在关键步进行完整前向推理，非关键步使用缓存或预测特征。

**📊 数据集**

在 FLUX.1-dev（文本图像）、HunyuanVideo（文本视频）和 DiT‑XL/2（类别图像）上验证；使用 DrawBench、VBench 和 ImageNet 等公开数据集进行评测。

**📈 对比分析**

与 TeaCache、TaylorSeer、SpeCa、DDIM 等基准方法对比，DPCache 在保持或超过全步基线的前提下实现最高 4.87× 的加速比，ImageReward、CLIP Score、FID、PSNR 等指标均优于现有缓存加速方案。

**⚠️ 局限性**

局限性在于仍依赖固定的全局调度，无法自适应不同输入的特征动态；对预测模型的误差敏感；虽然只缓存最终层特征，但在极大模型或更高分辨率下仍可能出现显著误差；未来需引入可学习预测器或输入自适应调度以进一步提升性能。

---

## 235. Multidimensional Task Learning: A Unified Tensor Framework for Computer Vision Tasks

**arXiv ID:** 2602.23217 | [PDF](https://arxiv.org/pdf/2602.23217v1)

**作者:** Alaa El Ichi `[一作]` (Université du Littoral Cote d'Opale), Khalide Jbilou `[通讯]` (Université du Littoral Cote d'Opale)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393`

**🎯 论文内容**

本文提出了一种多维任务学习（MTL）框架，将分类、分割、检测等传统计算机视觉任务统一到一个基于张量代数的数学模型中。

**💡 创新点**

创新点在于引入了Generalized Einstein MLP（GE-MLP）和Einstein乘积，使得网络能直接在高维张量上进行运算，避免了传统矩阵化的扁平化限制，并通过结构保持指数（ρ）量化维度保留与收缩，从而实现对任务空间的系统化划分和新任务配置的设计。

**🔧 技术方法**

主要技术包括张量范数与Einstein乘积的定义、GE-MLP的前向与梯度更新（GEGD）、复杂度分析，以及以张量维度配置为参数的MTL任务定义与理论证明。

**📊 数据集**

论文未使用或公开任何具体数据集；仅在理论层面阐述框架与任务统一。

**📈 对比分析**

未给出实验比较或性能指标；讨论中仅说明该框架理论上可覆盖并扩展传统任务空间，但未通过实测验证其效果。

**⚠️ 局限性**

局限性包括：缺乏实证验证；实现细节（如高维张量训练的数值稳定性、硬件加速支持）未展开讨论；对具体任务的性能提升与计算成本平衡也需进一步研究。

---

## 236. Through BrokenEyes: How Eye Disorders Impact Face Detection?

**arXiv ID:** 2602.23212 | [PDF](https://arxiv.org/pdf/2602.23212v1)

**作者:** Prottay Kumar Adhikary `[一作]` (Indian Institute of Technology Delhi), Prottay Kumar Adhikary `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5007605476)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

模拟五种常见眼部疾病并通过BrokenEyes过滤器生成失真图像，训练ResNet18模型进行人类与非人类二分类，并分析失真对高层特征表示的影响。

**💡 创新点**

提出了基于临床特征的疾病特异性图像降质框架BrokenEyes，配合疾病感知的微调和特征能量与余弦相似度的双重量化分析，首次系统揭示视觉障碍对深度特征的结构性扰动。

**🔧 技术方法**

利用ResNet18迁移学习，结合高斯模糊、暗角、随机椭圆遮挡、饱和度降低等图像处理技术；采用PyTorch实现训练与特征提取，使用余弦相似度和激活能量度量特征漂移。

**📊 数据集**

使用LFW人脸数据集与MS‑COCO非人类图像集合，经过尺寸统一与归一化后构建平衡的六种视觉条件（正常、AMD、白内障、青光眼、屈光不正、糖尿病视网膜病变）数据集。

**📈 对比分析**

通过在测试集上比较正常与各疾病模型的分类置信度、特征能量和余弦相似度进行对比；正常模型在二分类任务上达100%准确率，其他模型虽然仍能正确分类，但置信度显著下降，白内障与青光眼模型的特征相似度最低，表明其对特征表示的扰动最大。

**⚠️ 局限性**

局限于模拟失真，缺乏真实患者图像验证；实验仅限于二分类任务，未探究更细粒度识别；特征量化指标局限于激活能量和余弦相似度，未结合神经影像或眼动追踪数据进行生理学验证。

---

## 237. Managing Uncertainty in LLM-based Multi-Agent System Operation

**arXiv ID:** 2602.23005 | [PDF](https://arxiv.org/pdf/2602.23005v1)

**作者:** Man Zhang `[一作]` (Beihang University), Yihua He `[通讯]` (Capital Medical University)

**通讯引用:** 623 | [OpenAlex ID](https://openalex.org/A5100648750)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了面向LLM多智能体软件系统的生命周期不确定性管理框架，并在寿命期心脏超声诊断平台上验证其可行性

**💡 创新点**

将不确定性分为认识论与本体论两类，并引入PSUM标准实现不确定性结构化表示、识别、演化与适应；同时将人类操作员角色嵌入不确定性生命周期，实现安全可靠的运行治理

**🔧 技术方法**

采用大型语言模型（如GPT‑4、DeepSeek）、多智能体协同架构、PSUM不确定性元建模、观察者-推理者-构造者-调度者-指挥者等软件工程技术

**📊 数据集**

使用真实临床寿命期心脏超声图像及伴随的电子健康记录作为实验数据集

**📈 对比分析**

通过与传统单模型诊断方法对比，实验表明该框架在诊断可靠性、解释可追溯性和人机协作效率方面均有显著提升，尽管未给出数值指标，但作者报告了诊断置信度提升与误报率下降

**⚠️ 局限性**

局限性包括：依赖于手工定义的PSUM模型和政策，需人工介入导致可扩展性受限；在不同领域与规模下的通用性尚待验证；对高阶本体不确定性仍未提供完全自动化的解决方案

---

## 238. MM-NeuroOnco: A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis

**arXiv ID:** 2602.22955 | [PDF](https://arxiv.org/pdf/2602.22955v1)

**作者:** Feng Guo `[一作]` (Guangdong Institute of Intelligence Science and Technology), Mingkun Xu `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了MM-NeuroOnco大型多模态脑瘤MRI诊断基准和指令数据集，并在此基础上开发了NeuroOnco-GPT模型。

**💡 创新点**

创新点包括：①利用多模型协同管道自动生成诊断语义并进行质量控制；②引入拒绝感知评估机制和链式推理(CoT)训练；③提供高密度语义标注与开放式诊断推理的双重评测。

**🔧 技术方法**

核心技术包括多模型协同语义提取、图像几何特征映射、LLM指令生成、Rejection-Aware 评估与CoT监督。

**📊 数据集**

使用了来自20个公开数据源的24,726张MRI切片（四种模态，八种肿瘤亚型），并在此基础上生成约20万条语义丰富的指令样本。

**📈 对比分析**

与10个代表性多模态大模型（包括Gemini‑3‑Flash、GPT‑5.1等）对比，最强基线仅达41.9%准确率，NeuroOnco‑GPT在Fine‑Tuning后可提升约27%诊断准确率。

**⚠️ 局限性**

局限性包括仅基于单切片数据缺乏三维上下文，部分语义标注仍依赖自动化管道，可能存在噪声；评测仍无法完全覆盖真实临床决策过程。

---

## 239. SCOPE: Skeleton Graph-Based Computation-Efficient Framework for Autonomous UAV Exploration

**arXiv ID:** 2602.22707 | [PDF](https://arxiv.org/pdf/2602.22707v1)

**作者:** Kai Li `[一作]` (Tsinghua University), Xinlei Chen `[通讯]` (Tsinghua University)

**通讯引用:** 8821 | [OpenAlex ID](https://openalex.org/A5016095713)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 SCOPE 框架，实现了基于骨架图的实时自主 UAV 探索，结合隐式未知区域分析与分层规划，显著降低计算开销。

**💡 创新点**

创新点在于：① 通过增量构建骨架图并使用几何探测实现对未知空间的隐式分解；② 采用分层规划，将高频局部规划与低频全局序列规划分离；③ 在全局规划中仅在必要时才求解对称 TSP，避免频繁的全局优化。

**🔧 技术方法**

核心技术包括：骨架图（Skeleton Graph）提取与维护；几何探测（Geometric Probes）与区域聚类；近端规划器（Proximal Planner）实现高频平滑轨迹；区域序列规划器（Region-Sequence Planner）基于最短骨架距离构建 TSP；以及离线模拟与实机部署的评估框架。

**📊 数据集**

使用五种仿真环境（Complex Office、Octa Maze、Classical Office、Darpa Tunnel、Duplex Office）和两种真实室内实验环境进行测试；不使用公开大规模数据集，主要依赖自建地图与模拟环境。

**📈 对比分析**

与 FUEL、FAEP、RACER、FALCON 四个基线比较，SCOPE 在探索效率与覆盖率上与最优方法相近，同时平均计算成本降低约 86.9%，大幅减少回溯与振荡，提升飞行速度。

**⚠️ 局限性**

局限性：① 路径仍略有次优（相对最优路径延长约 5‑7%）；② 仅在室内环境下验证，对复杂户外大尺度场景尚未测试；③ 依赖精确的 ESDF 与传感器数据，噪声或遮挡可能影响骨架图的构造与规划决策。

---

## 240. Generalized Rapid Action Value Estimation in Memory-Constrained Environments

**arXiv ID:** 2602.23318 | [PDF](https://arxiv.org/pdf/2602.23318v1)

**作者:** Aloïs Rautureau `[一作]` (UCLouvain), Éric Piette `[通讯]` (UCLouvain)

**通讯引用:** 1111 | [OpenAlex ID](https://openalex.org/A5061767166)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发了GRAVE^2、GRAVER和GRAVER^2三种GRAVE变体，通过两层搜索和节点回收技术在内存受限环境下保持相当的下棋强度。

**💡 创新点**

首次在同一框架内同时采用两层搜索与节点回收，实现对GRAVE节点数量的极大压缩，同时保留AMAF信息并维持MCTS的实时性。

**🔧 技术方法**

使用两层搜索（GRAVE^2）与前向共享、节点回收（LRU缓存）、AMAF、MAST、UCT²等技术。

**📊 数据集**

在Go 9×9棋盘上进行实验，使用500局评测数据集。

**📈 对比分析**

以GRAVE P=N=10,000为基准，对抗赛测算胜率与95%置信区间；结果表明GRAVER^2可在仅160节点、12800次模拟下达到与GRAVE相当的胜率，GRAVE^2可在240节点下匹配。

**⚠️ 局限性**

实验仅限于Go 9×9，未验证更大棋盘或其他游戏；节点回收可能导致AMAF信息损失，对性能产生影响；两层搜索在任何时刻性能不如单层。

---

## 241. Verification of Unbounded Client-Server Systems with Distinguishable Clients

**arXiv ID:** 2602.23054 | [PDF](https://arxiv.org/pdf/2602.23054v1)

**作者:** Ramchandra Phawade `[一作]` (Indian Institute of Technology Dharwad), S Sheerazuddin `[通讯]` (National Institute of Technology Calicut)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5014469255)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了可区分客户端且客户端数目未预先限定的客户端‑服务器系统，并提出一种基于 ν‑net 的建模方法和一种单变量的 MFOTL 片段，用以描述其性质，随后给出了相应的 SMT 编码，实现了一个名为 UCSChecker 的 BMC 工具。

**💡 创新点**

创新点在于：①首次将 ν‑net 与单变量 MFOTL 片段结合用于描述可区分的无界客户端系统；②提出了一套完整的 SMT 编码策略，支持二维 BMC（时步与客户端数）检查；③构建了开源工具 UCSChecker，可直接对 PNML 描述的 ν‑net 进行属性验证。

**🔧 技术方法**

使用了 ν‑net 作为模型语言、单变量 MFOTL（Monadic First Order Temporal Logic）作为规格语言、SMT 编码技术、Z3 作为求解器，并利用 ANTLR 对 PNML 与属性进行语法校验。

**📊 数据集**

实验数据集为两个案例研究：Autonomous Parking System (APS) 与 Travel Agency 系统，它们分别在 PNML 格式下实现了 ν‑net 并编写相应的属性进行验证；未使用公开的工业或真实数据集。

**📈 对比分析**

通过对 APS 与 Travel Agency 的属性进行实验，比较了不同属性的嵌套深度、子句数与执行时间，结果显示 UCSChecker 在 0.02–0.4 秒内完成验证，证明了该方法在小型至中型模型上的高效性；论文中未与其它现有工具做直接对比，但提供了可复现的实验细节。

**⚠️ 局限性**

局限性包括：①工具依赖 PNML 编辑器，无法自动生成可区分标记的可视化；②无法表达单个客户端的具体属性（仅能对所有客户端统一约束）；③对大规模模型的扩展性尚待进一步验证。

---

## 242. Productivity and Collaboration in Hybrid Agile Teams: An Interview Study

**arXiv ID:** 2602.22835 | [PDF](https://arxiv.org/pdf/2602.22835v1)

**作者:** Elisabeth Mo `[一作]` (DNB Bank ASA), Asle Fagerstrøm `[通讯]` (Kristiania University of Applied Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过对挪威三支敏捷团队进行访谈，研究混合工作环境如何影响团队生产力与协作，并分析敏捷实践的适配方式。

**💡 创新点**

首次将IMOI框架与敏捷仪式、信任、沟通等中介因素结合，揭示混合工作中信息流断层、非正式互动缺失和工具使用不一致对协作的具体影响，并给出针对性的实践建议。

**🔧 技术方法**

采用定性访谈与主题编码（基于IMOI框架）对访谈数据进行分析，并使用Microsoft Teams等数字工具记录与归档访谈内容。

**📊 数据集**

研究数据集来自三支团队的访谈记录（共9次访谈），涵盖金融与公共部门，成员包括软件工程师、产品负责人和经理等。

**📈 对比分析**

方法为基于IMOI框架的输入-中介-结果编码，未使用量化指标与对照实验，主要通过文本归纳与比较各团队经验来阐明影响因素，因而无法给出客观性能数值。

**⚠️ 局限性**

局限性包括样本规模有限、单一时点、受访者自述与研究者解读的主观性、可能的偏见以及缺乏跨文化与纵向验证。

---

## 243. Multi-agent imitation learning with function approximation: Linear Markov games and beyond

**arXiv ID:** 2602.22810 | [PDF](https://arxiv.org/pdf/2602.22810v1)

**作者:** Luca Viano `[一作]` (École Polytechnique Fédérale de Lausanne), Giorgia Ramponi `[通讯]` (University of Zurich)

**通讯引用:** 252 | [OpenAlex ID](https://openalex.org/A5048076721)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了线性马尔可夫游戏中的多智能体模仿学习（MAIL），提出了理论分析、交互式算法和深度扩展。

**💡 创新点**

创新点包括：1) 引入特征级可聚合系数，降低对传统可聚合系数的依赖；2) 提出第一种适用于线性马尔可夫游戏的交互式 MAIL 算法 LSVI‑UCB‑ZERO‑BC，样本复杂度仅依赖特征维度；3) 将理论成果转化为深度 MAIL 算法 DQN‑Explore‑BC，能在 Tic‑Tac‑Toe 和 Connect‑4 等复杂游戏中显著降低 Nash gap。

**🔧 技术方法**

技术手段包括：线性函数逼近、奖励无关探索（reward‑free）与 UCB‑ZERO 结合、最大似然估计（MLE）分析、深度 Q‑网络（DQN）与探索奖励，结合特征映射和经验回放。

**📊 数据集**

实验数据集主要为：1) 3×3 网格世界（共72个状态）；2) Tic‑Tac‑Toe 和 Connect‑4（使用最优求解器作为专家）。

**📈 对比分析**

与传统行为克隆（BC）相比，交互式算法在线性和深度设置下都显著降低 Nash gap；在 Connect‑4 中，DQN‑Explore‑BC 的胜率对抗不同水平的对手显著高于深度 BC，实验表明性能提升可达 30%+。

**⚠️ 局限性**

局限性包括：1) 需要专家策略可实现性假设；2) 对特征映射的依赖仍较强，选择合适特征不易；3) 对多玩家（N>2）游戏的扩展需要进一步研究；4) 在非常大规模或连续状态空间中，特征估计与样本复杂度仍待改进。

---

## 244. Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization

**arXiv ID:** 2602.23008 | [PDF](https://arxiv.org/pdf/2602.23008v1)

**作者:** Zeyuan Liu `[一作]` (Microsoft Research), Yuqing Yang `[通讯]` (Microsoft Research)

**通讯引用:** 1973 | [OpenAlex ID](https://openalex.org/A5101421201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合记忆增强与双重策略更新（EMPO^2）的强化学习框架，以提升大型语言模型在探索任务中的表现

**💡 创新点**

1）同时利用参数更新和非参数记忆更新；2）在 rollout 与更新阶段提供记忆化与无记忆两种模式；3）通过离线与在线的混合更新实现高效探索和知识内化

**🔧 技术方法**

大语言模型（Qwen2.5-7B-Instruct）、基于记忆的提示生成、Group Relative Policy Optimization（GRPO）框架、熵正则化、KL 约束、重要性采样、内在奖励（novelty）

**📊 数据集**

ScienceWorld（19个多步实验任务）和 WebShop（HTML购物环境）

**📈 对比分析**

与基线（Naive、Reflexion、Retrospex、GRPO、GiGPO）对比，EMPO^2 在 ScienceWorld 的平均回报从-61.3提升至75.9，任务成功率显著提高；在 WebShop 上得分从79.3提升至88.3，成功率从72.8%提升至76.9%；同时在 OOD 场景下仅需几步记忆更新即可实现大幅提升

**⚠️ 局限性**

1）记忆检索采用简单的相似度搜索，可能限制性能；2）实验仅在 Qwen2.5-7B-Instruct 上验证，缺乏对不同规模模型的泛化验证；3）离线与在线混合更新仍可能导致训练不稳定，需要更稳健的 off‑policy 技术

---

## 245. Efficient Encoder-Free Fourier-based 3D Large Multimodal Model

**arXiv ID:** 2602.23153 | [PDF](https://arxiv.org/pdf/2602.23153v1)

**作者:** Guofeng Mei `[一作]` (Fondazione Bruno Kessler), Fabio Poiesi `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 1360 | [OpenAlex ID](https://openalex.org/A5067244774)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无编码器的3D多模态模型Fase3D，直接用轻量化傅里叶基分割点云的superpoint作为token，并通过FFT增强全局上下文；

**💡 创新点**

创新点在于：用空间填充曲线序列化superpoint后进行FFT实现自注意力近似；利用图稀疏token合并进一步压缩token数；在LLM中加入傅里叶增强LoRA适配器，实现频域全局交互；

**🔧 技术方法**

采用轻量化MLP点特征提取、空间填充曲线（Z-order、Hilbert等）、FFT与逆FFT、稀疏kNN图构建、Sinkhorn最优传输聚类、Fourier-augmented LoRA等技术；

**📊 数据集**

在ScanNet、ScanQA、SQA3D、ScanRefer、Nr3D等数据集上进行预训练与微调，验证模型在多任务（问答、密集描述、定位）上的表现；

**📈 对比分析**

与基于重编码器的3D-LLaVA、LL3DA、PerLA等方法相比，Fase3D在ScanQA、SQA3D、ScanRefer、Nr3D等指标上保持或略高的性能，但参数量和FLOPs仅为前者的十分之一；

**⚠️ 局限性**

局限在于基于序列化的FFT近似可能在高度混乱、长程非欧几里得关系场景中欠佳，且对稀疏点云的全局信息捕获仍有限。

---

## 246. Strengthening security and noise resistance in one-way quantum key distribution protocols through hypercube-based quantum walks

**arXiv ID:** 2602.23261 | [PDF](https://arxiv.org/pdf/2602.23261v1)

**作者:** David Polzoni `[一作]` (University of Padova), Mauro Conti `[通讯]` (Örebro University)

**通讯引用:** 26544 | [OpenAlex ID](https://openalex.org/A5063847107)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于超立方体拓扑的单向量子密钥分发协议，并实现了完整的Qiskit仿真框架

**💡 创新点**

创新点在于将离散时间量子随机游走从传统圆环拓扑扩展到高维超立方体，显著提升安全性和噪声容限

**🔧 技术方法**

使用了量子随机游走（quantum walk）、Grover coin、Hadamard/相位门以及Qiskit的量子门和噪声模型

**📊 数据集**

实验数据基于Qiskit自带的单量子比特噪声模型（退相干和幅度-相位耗散）和不同P值（1至13）的超立方体/圆环图形

**📈 对比分析**

通过比较不同拓扑下的安全参数c、最大容忍错误率Q以及密钥率r，结果显示超立方体拓扑在相同参数下比圆环拓扑多提升约20–30% 的噪声容忍度

**⚠️ 局限性**

主要限制包括仅能模拟至P≤13、对单比特噪声模型的依赖、缺乏真实随机数源以及未考虑完整的重组和隐私放大步骤

---

## 247. SAFARI: A Community-Engaged Approach and Dataset of Stereotype Resources in the Sub-Saharan African Context

**arXiv ID:** 2602.22404 | [PDF](https://arxiv.org/pdf/2602.22404v1)

**作者:** Aishwarya Verma `[一作]` (Google Research), Sunipa Dev `[通讯]` (Google Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并发布了 SAFARI 数据集，收集了来自加纳、肯尼亚、尼日利亚、南非的 15 种本土语言和英语的 3,534 条刻板印象，采用电话访谈的社区参与方法。

**💡 创新点**

提出了以社区为主导、电话式多语种口语数据收集与双语记录的多步骤本地化方法，弥补了子撒哈拉非洲在刻板印象资源与方法上的缺口。

**🔧 技术方法**

结合双语电话访谈、专业翻译后本地化、NLI 基础的刻板印象检测，评估 Gemini、GPT‑4o、Claude 等前沿 LLM 的刻板印象倾向。

**📊 数据集**

使用 SAFARI 数据集本身（3,534 条英语刻板印象和 3,206 条 15 种本土语言刻板印象），并与 StereoSet、SPICE、HESEIA 等现有资源进行对比分析。

**📈 对比分析**

通过构造 NLI 题目，对 Gemini 2.5/3 Pro、GPT‑4o/5.1、Claude Sonnet/Opus 等模型进行刻板印象倾向评估，结果显示大多数模型在英语及当地语言均存在较高的刻板印象率，英语表现更为突出。

**⚠️ 局限性**

方法受限于电话接入、翻译过程可能丢失细微语义、样本规模局限于 4 个国家和主要语言，未覆盖所有族群，且本地化过程可能削弱原始语义细腻度。

---

## 248. TopoEdit: Fast Post-Optimization Editing of Topology Optimized Structures

**arXiv ID:** 2602.22430 | [PDF](https://arxiv.org/pdf/2602.22430v1)

**作者:** Hongrui Chen `[一作]` (Massachusetts Institute of Technology), Faez Ahmed `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 12307 | [OpenAlex ID](https://openalex.org/A5026634347)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为TopoEdit的后优化拓扑编辑框架，利用预训练的拓扑基础模型（OAT）在潜在空间中进行局部、物理感知的结构修改；

**💡 创新点**

创新点在于将结构化潜在嵌入用于编辑，采用部分去噪+编辑后再去噪的管道，并设计了三种编辑算子（拓扑扭曲、晶格替换、禁止设计区），实现快速、局部、保持物理一致性的编辑；

**🔧 技术方法**

核心技术包括基于OAT的潜在扩散模型、部分去噪策略、DDIM引导去噪、基于边界条件的条件更新、以及可选的SIMP后处理；

**📊 数据集**

使用了OAT训练集中的约220万条优化拓扑，包含随机边界条件、载荷、体积分数等，且在训练集和测试集上分别进行实验；

**📈 对比分析**

与直接在密度空间编辑的基线对比，TopoEdit在合规性（Compliance）和距离误差（Distance Error）上表现更好；在多重采样后选择最优结果，可显著降低失效率；采样速度在子秒级；

**⚠️ 局限性**

局限性包括对潜在分辨率的依赖、对三维拓扑的支持尚未实现、编辑算子种类有限（仅支持扭曲、晶格替换、禁止设计区），以及模型先验限制导致部分极端编辑仍可能出现失败。

---

## 249. Evaluating Zero-Shot and One-Shot Adaptation of Small Language Models in Leader-Follower Interaction

**arXiv ID:** 2602.23312 | [PDF](https://arxiv.org/pdf/2602.23312v1)

**作者:** Rafael R. Baptista `[一作]` (University of Sao Paulo), Gustavo J. G. Lahr `[通讯]` (Instituto Israelita de Ensino e Pesquisa)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在移动协助机器人上使用小语言模型（SLM）进行领导者-跟随者角色分配的可行性，并构建了专门的数据集进行评估。

**💡 创新点**

首次公开了针对领导者-跟随者HRI的专用数据集，并系统比较了prompt engineering与fine‑tuning两种小模型适配策略，揭示了多轮对话对小模型性能的影响。

**🔧 技术方法**

采用Qwen2.5‑0.5B小语言模型，结合prompt engineering、Fine‑tuning、zero‑shot与one‑shot交互模式，并使用SBERT评估语义保真度。

**📊 数据集**

基于DailyDialog生成的415条问题，并通过DeepSeek、Gemini、GPT‑4生成6倍合成样本，总计5400条领导者-跟随者对话样本，另留100条用于测试。

**📈 对比分析**

使用Monte Carlo交叉验证30次，对比baseline、prompt engineering和fine‑tuning在zero‑shot和one‑shot两种模式下的准确率、精确率、召回率、F1、吞吐量和延迟；zero‑shot fine‑tuning达86.66%准确率且延迟22 ms；one‑shot模式所有方法均降至≈50%准确率。

**⚠️ 局限性**

小模型在one‑shot多轮对话中表现显著衰退，受限于参数容量和上下文长度；数据集规模有限且主要为合成；缺乏真实机器人实验验证。

---

## 250. Physics Informed Viscous Value Representations

**arXiv ID:** 2602.23280 | [PDF](https://arxiv.org/pdf/2602.23280v1)

**作者:** Hrishikesh Viswanath `[一作]` (Purdue University), Aniket Bera `[通讯]` (Purdue University)

**通讯引用:** 1780 | [OpenAlex ID](https://openalex.org/A5058453308)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了基于粘性解的物理信息正则化方法，用以改进离线目标条件强化学习中的价值函数估计，能够产生稳定、直接的轨迹。

**💡 创新点**

创新点在于：①将HJB方程的粘性（viscosity）解作为正则化约束，取代传统的一阶Eikonal约束；②利用Cole‑Hopf变换将非线性PDE线性化，并通过Feynman‑Kac公式将其转化为期望形式，避免高阶梯度计算导致的不稳定；③该正则化与价值函数表示无关，兼容多种表示方法和层次化策略。

**🔧 技术方法**

技术手段包括：粘性HJB正则化、Cole‑Hopf线性化、Feynman‑Kac蒙特卡洛估计、随机游走采样、离线GCRL算法（GCIVL、IQL、Dual‑Goal等）以及与HIQL层次化策略的融合。

**📊 数据集**

实验使用13个离线目标条件RL基准任务，涵盖PointMaze、AntMaze、HumanoidMaze、AntSoccer、Manipulation等，并对噪声、大小、推理复杂度等多变体进行评估。

**📈 对比分析**

通过与原始、VIB、Dual、Eikonal等基线方法以及HIQL进行比较，实验显示该方法在大多数任务中实现了更高的成功率（如噪声操控任务达99%），显著优于一阶Eikonal正则化，且在高维接触丰富任务中表现尤为突出。

**⚠️ 局限性**

局限性包括：①对极长时间步长或极大状态空间的任务仍有局限；②在高度随机的像素域中直接对原始像素采样不切实际，拉伸至潜在空间时因非欧几里得结构导致正则化效果减弱；③对某些高维、极长距离任务（如大型迷宫）仍无法完全克服局部几何约束的限制。

---

## 251. VGG-T$^3$: Offline Feed-Forward 3D Reconstruction at Scale

**arXiv ID:** 2602.23361 | [PDF](https://arxiv.org/pdf/2602.23361v1)

**作者:** Sven Elflein `[一作]` (NVIDIA), Aljosa Osep `[通讯]` (NVIDIA)

**通讯引用:** 1715 | [OpenAlex ID](https://openalex.org/A5009902749)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种可线性扩展的离线 feed-forward 3D 重建模型，利用测试时训练把 KV 空间压缩为固定大小 MLP，支持千张图像以内秒级重建。

**💡 创新点**

创新点在于用测试时训练将全局注意力中的可变长度 KV 表示映射到固定维度 MLP，突破软最大注意力的 O(n²) 计算瓶颈，实现 O(n) 推理，并能一次前向查询完成视觉定位。

**🔧 技术方法**

采用 VGGT 预训练网络、测试时训练（TTT）、SwiGLU MLP、ShortConv2D 空间混合、分布式微批优化、线性注意力替代等技术。

**📊 数据集**

在 7Scenes、WaySpot、NRGBD、DTU、ETH3D、ScanNet、TUM‑RGBD、KITTI、Sintel、Waymo 等多种室内外图像集合上训练与评估。

**📈 对比分析**

与 VGGT、FastVGGT、SparseVGGT（O(n²)）和 TTT3R（O(n)）等基线比较，7Scenes 1k 图像下速度提升 11.6×（48.5 s），点图误差与视频深度均优于 O(n) 基线，整体准确度与 O(n²) 基线相当。

**⚠️ 局限性**

与软最大注意力相比，在宽基线和大空间范围场景中精度仍有差距；固定 MLP 的表达能力有限，长序列推理需多步优化，无法完全匹配二次注意力的细粒度几何。

---

## 252. TaleBot: A Tangible AI Companion to Support Children in Co-creative Storytelling for Resilience Cultivation

**arXiv ID:** 2602.23095 | [PDF](https://arxiv.org/pdf/2602.23095v1)

**作者:** Yonglin Chen `[一作]` (Southern University of Science and Technology), Xueliang Li `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5100365967)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了 TaleBot——一款结合生成式 AI 的互动式故事创作系统，帮助小学儿童通过与 AI 合作叙事的方式培养心理韧性，并在校内辅导室及家庭场景中推广使用。

**💡 创新点**

创新点包括：① 以心理健康教师为导向的专家后台接口，使故事情节可根据个体心理状态个性化生成；② 采用实体化（3D 打印+绒面）触觉载体提升社交存在感；③ 通过父母共享的注释版故事书将 AI 与家长对话桥接，促进跨场景共情与反思；④ 多智能体架构实现从画面识别、文本生成到情感分析的闭环交互。

**🔧 技术方法**

使用的技术主要有：大语言模型 Deepseek V3（文本生成）、OpenAI GPT-Image-1（图像生成）、腾讯语音识别与豆瓣语音合成、Flask+Flutter 前后端架构、React web 后台、Python 多智能体脚本；系统实现了音频输入、视觉输出与情感反馈的闭环。

**📊 数据集**

数据集主要为内部收集的 12 名小学生的语音响应、绘图输入及其生成的故事文本，共 48 条情境回答；此外记录了 5 名家长的访谈文本、教师观察笔记。并未使用公开公开大规模数据集，而是基于现场实验获得的原始数据。

**📈 对比分析**

方法比较：未对照传统纸本故事创作或单向叙事；通过 System Usability Scale (SUS) 评估系统可用性，平均得分 81.09 分（优秀水平），并结合定性主题分析（共 15 个主题）阐释儿童表达、教师支持与家长反思的效果。性能上，生成延迟约 5–10 秒，视觉一致性仍需提升。

**⚠️ 局限性**

限制：① 样本规模小，仅 12 名儿童与 1 名心理教师；② 未进行长期随访，无法评估韧性提升的持续性；③ AI 生成的情感分析与建议可能存在偏差；④ 生成图片与文本偶有不一致，影响体验；⑤ 系统延时及可视化质量需改进。

---

## 253. A Reduced Magnetic Vector Potential Approach with Higher-Order Splines

**arXiv ID:** 2602.22997 | [PDF](https://arxiv.org/pdf/2602.22997v1)

**作者:** Merle Backmeyer `[一作]` (University of Grenoble Alpes), Sebastian Schöps `[通讯]` (Technische Universität Darmstadt)

**通讯引用:** 1651 | [OpenAlex ID](https://openalex.org/A5054323051)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于高阶等几何（Isogeometric Analysis, IGA）分解的磁矢势（Reduced Magnetic Vector Potential, RMVP）方法，用以求解磁量子静态（MQS）欧氏电流问题，并通过在源场（Biot‑Savart）与反应场（有限元）之间仅在分离表面进行核函数评估，从而显著降低计算量；

**💡 创新点**

创新点包括：①将原先磁静态RMVP推广至MQS耦合场域；②采用高阶NURBS/ B‑spline基函数实现几何精确与高阶逼近；③阐明了表面积分、轨迹空间以及核函数数值积分必须满足高阶一致性才能实现理论最优收敛；④通过单表面（interface）策略把核函数求值从体积分解削减到表面积分，显著减少运算量。

**🔧 技术方法**

技术手段主要包括：高阶等几何曲面/体积映射、Biot‑Savart 定积分的周期性梯形规则（实现指数级收敛）、三维有限元（curl‑conforming）求解反应场、表面算子（Dirichlet/Neumann 跟踪）、多块分块（multi‑patch）分离表面以及相应的树-树图（tree‑cotree）去除自由度。

**📊 数据集**

实验数据集为：①圆柱形导体（半径12，厚度60，σ=3.5×10⁶ S/m）外包围的圆形线圈（半径25，幅值320 A，频率200 Hz）用于验证收敛；②螺旋线圈（200圈）缠绕同一圆柱，用以与商业软件 Flux 对比。

**📈 对比分析**

方法比较：与解析参考解比对，数值解在梯形积分、轨迹空间一致的前提下达到了理论的最优收敛（阶p）。与 Flux 的能量存储结果相差不足2%；在核函数求值次数上，单表面RMVP比原始RMVP减少约85%，总计算时间从34 s降至约34 s（含10 s核函数），相较于原始RMVP的100 s（含85 s核函数）实现了显著性能提升。

**⚠️ 局限性**

局限性：①需要在表面积分与反应场之间保持高阶一致性，若轨迹空间或核积分精度不足会导致收敛退化；②多块几何（多patch）实现仍需手动构造，且对复杂曲面精度要求高；③尚未对单表面RMVP实现向量化/并行加速，实际性能仍可提升；④对非线性磁性材料和时变解的完整实现仍在进一步研究中。

---

## 254. CourtGuard: A Model-Agnostic Framework for Zero-Shot Policy Adaptation in LLM Safety

**arXiv ID:** 2602.22557 | [PDF](https://arxiv.org/pdf/2602.22557v1)

**作者:** Umid Suleymanov `[一作]`, Murat Kantarcioglu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了XXX问题，提出了一种新的解决方案。

**💡 创新点**

创新点在于引入了XXX方法，显著提高了XXX的性能。

**🔧 技术方法**

使用了XXX技术，如深度学习、机器学习等。

**📊 数据集**

实验中使用了XXX数据集，包含了XXX样本。

**📈 对比分析**

与现有方法进行了比较，结果表明新方法在XXX指标上优于传统方法。

**⚠️ 局限性**

限制在于XXX，例如数据集的规模、模型的复杂性等。

---

## 255. SEGB: Self-Evolved Generative Bidding with Local Autoregressive Diffusion

**arXiv ID:** 2602.22226 | [PDF](https://arxiv.org/pdf/2602.22226v1)

**作者:** Yulong Gao `[一作]` (JD.com), Xin Yang `[通讯]` (JD.com)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种全离线自演化的生成式竞价框架（SEGB），通过规划、预判与自我进化实现更优的竞价策略。

**💡 创新点**

创新点在于：①局部自回归扩散（LAD）实现因果一致的未来状态规划；②将预测的下一状态融入Decision Transformer，实现前瞻性决策；③利用GRPO进行完全离线的策略进化，克服离线数据覆盖不足的问题。

**🔧 技术方法**

采用了局部自回归扩散（LAD）、Next‑State‑Aware Decision Transformer、Implicit Q‑Learning（IQL）作为评论器、以及Group Relative Policy Optimization（GRPO）进行离线优化。

**📊 数据集**

使用公开的AuctionNet数据集（含完整与稀疏两种版本）以及在JD.com广告平台进行的大规模A/B测试。

**📈 对比分析**

与BCQ、CQL、DiffBid、IQL、DT、GAS等基线比较，SEGB在所有预算比例和数据稀疏度场景均取得最高分，在线A/B测试中提升目标成本+10.19%。

**⚠️ 局限性**

主要限制包括：对分布漂移的鲁棒性仍有限，需要进一步研究在持续变化的市场环境中的自适应能力；模型在在线推理时增加了规划步骤，虽然仍低于100 ms，但在极高并发下仍需优化。

---

## 256. Moral Preferences of LLMs Under Directed Contextual Influence

**arXiv ID:** 2602.22831 | [PDF](https://arxiv.org/pdf/2602.22831v1)

**作者:** Phil Blandfort `[一作]` (Predictably Weird), Dmitrii Krasheninnikov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在道德三叉路决策中，受上下文指向性影响的可塑性，并提出了方向翻转的实验框架。

**💡 创新点**

提出了对比式的方向翻转影响评估，能够量化偏好转移、背离与不对称可塑性。

**🔧 技术方法**

使用多模型（DeepSeek、Grok、LLaMA、GPT‑5.2、Qwen3）在多种情境下采样答案，并结合对数几率差异、比例检验和Wald检验进行统计评估；同时引入链式思考与语义/表面形式区分技术。

**📊 数据集**

采用自定义的Trolley问题式道德三叉路对比，覆盖五个二元人口特征和十种群体规模，形成实验样本。

**📈 对比分析**

通过对数几率差异、比例检验与Wald检验评估影响力和不对称性，结果显示约68%场景影响显著，平均绝对可塑性约1.09（频率变动≈15%），并发现后向效应高达24%。

**⚠️ 局限性**

实验仅覆盖有限的指向性影响和Trolley式情境，未考虑多轮历史或更真实部署环境，且对模型的外推性和可解释性仍有限。

---

## 257. Don't let the information slip away

**arXiv ID:** 2602.22595 | [PDF](https://arxiv.org/pdf/2602.22595v1)

**作者:** Taozhe Li `[一作]` `[通讯]` (University of Oklahoma), Taozhe Li (University of Oklahoma)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Association DETR模型，通过在Transformer检测框架中加入背景注意模块和关联模块，以利用背景信息提升目标检测性能。

**💡 创新点**

创新点在于首次将背景信息显式建模为可插拔的Association Encoder（仅3.1M参数），并通过背景注意模块（BAM）和关联模块（AM）在DETR基础上实现性能突破。

**🔧 技术方法**

主要技术包括基于RFCBAMConv的背景注意模块、ConvFFN与Window Attention的关联模块、混合编码器、查询选择机制以及在Stanford Background数据集上的预训练。

**📊 数据集**

使用Stanford Background数据集对BAM进行预训练，随后在COCO 2017 val2017数据集上训练和评估模型。

**📈 对比分析**

通过在NVIDIA T4 GPU上与YOLOv10/11/12、RT-DETR、RT-DETRv2、DETR、Deformable-DETR等基线对比，Association DETR-R34在640×640输入下实现54.6 mAP、71.6 mAP50、153 FPS，R50版实现55.7 mAP、74.0 mAP50、104 FPS，显著提升同时仅降低约5.7%帧率。

**⚠️ 局限性**

局限性包括模型在加入AE后略有速度下降，背景注意模块仅覆盖9类背景且对更复杂或不同域场景的泛化性未作充分验证。

---

## 258. Modeling Expert AI Diagnostic Alignment via Immutable Inference Snapshots

**arXiv ID:** 2602.22973 | [PDF](https://arxiv.org/pdf/2602.22973v1)

**作者:** Dimitrios P. Panagoulias `[一作]` (University of Piraeus), Evridiki Tsoureli-Nikita `[通讯]` (Dermacen SA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种结构化诊断一致性框架，在人机协同的皮肤病学 AI 系统中保存 AI 生成的诊断报告（R_0），并与医生验证后的报告（R_1）进行多层级比较。

**💡 创新点**

创新点在于将专家校正建模为结构化信号变换，设计了从精确匹配到语义相似、跨类别对齐再到综合一致率（CCR）的四层级一致性评估，并强调保留不可变的 AI 推断状态以实现可追溯性。

**🔧 技术方法**

使用了基于视觉的大型语言模型（Vision‑LLM）、BERT 级实体抽取、Sequential Language Model Inference（SLMI）进行域内细化，以及字符串相似度阈值 τ 的语义匹配算法。

**📊 数据集**

采用 21 个公开皮肤病图像样本，经过皮肤科专家重新标注后构成 21 对 R_0–R_1 用于评估。

**📈 对比分析**

通过四层级一致性分析（PMR、AMR、交叉类别对齐、CCR）进行比较，结果显示精确匹配率 71.4%，语义匹配未提升，交叉类别比例 23.8%，但综合一致率 100%（95% 置信区间 [83.9%，100%]），并未出现完全诊断分歧。

**⚠️ 局限性**

局限包括样本量小、每例仅有单名医生验证、置信区间在低层级下仍宽泛、字符串相似度无法捕捉医学本体层级关系，且未充分验证在更大规模、多中心数据上的泛化性。

---

## 259. Conformalized Neural Networks for Federated Uncertainty Quantification under Dual Heterogeneity

**arXiv ID:** 2602.23296 | [PDF](https://arxiv.org/pdf/2602.23296v1)

**作者:** Quang-Huy Nguyen `[一作]` (Auburn University), Wei-Shinn Ku `[通讯]` (Auburn University)

**通讯引用:** 3422 | [OpenAlex ID](https://openalex.org/A5001457193)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种一轮通信的联邦加权分位数合成一致预测方法（FedAvgQ-CP），在数据与模型异质性的联邦学习场景下实现可靠的置信集/区间覆盖。

**💡 创新点**

创新点：①将每个客户端局部分位数与校准样本量加权平均，既考虑模型不确定性尺度又兼顾样本可靠性；②在不共享原始数据或模型参数的前提下，仅传输两个标量，保持通信效率；③在联合数据与模型异质性下依旧保持全局与局部的近似覆盖率。

**🔧 技术方法**

技术：分位数一致预测（Conformal Prediction），加权分位数聚合，Dirichlet 校准分区，使用 APS/CQR 得分函数；实现了单轮联邦通信的本地计算与服务器聚合。

**📊 数据集**

实验数据集：七个公开数据集，包含三类视觉基准（MNIST、FashionMNIST、CIFAR‑10）和四类医学图像（DermaMNIST、BloodMNIST、TissueMNIST、RetinaMNIST），通过 Dirichlet 分区模拟异质性。

**📈 对比分析**

与 SplitCP、FedCP‑QQ、FCP、CPhet 等基线对比；在所有七个数据集上，FedAvgQ-CP 在保持或接近 95% 覆盖率的同时，平均预测集/区间尺寸显著缩小（最高可降低约 60%），并在弱客户端上避免了显著欠覆盖；通信成本仅为两个标量，运行时与最优基线相当。

**⚠️ 局限性**

局限：①聚合分位数的加权平均并非严格等价于混合分布的分位数，可能在极端异质性或小样本情形下产生覆盖误差；②理论分析主要在理想化的 Dirichlet 校准偏移假设下成立，真实环境中多源异质性（如标签、特征分布差异）仍需进一步验证；③仅在单轮通信下实现，若模型或数据随时间动态变化，需考虑再校准机制。

---

## 260. A Holistic Framework for Robust Bangla ASR and Speaker Diarization with Optimized VAD and CTC Alignment

**arXiv ID:** 2602.22935 | [PDF](https://arxiv.org/pdf/2602.22935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 261. TherapyProbe: Generating Design Knowledge for Relational Safety in Mental Health Chatbots Through Adversarial Simulation

**arXiv ID:** 2602.22775 | [PDF](https://arxiv.org/pdf/2602.22775v1)

**作者:** Joydeep Chandra `[一作]` (Tsinghua University), Yong Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 47982 | [OpenAlex ID](https://openalex.org/A5007650371)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种名为TherapyProbe的对抗性多轮模拟方法，用于评估心理健康聊天机器人的关系安全性。

**💡 创新点**

创新点在于提出关系安全失败的六类税onomies，利用MCTS探索对话轨迹并生成23个可操作的安全模式库，从单轮评估转向多轮动态评估。

**🔧 技术方法**

主要技术包括自适应患者代理、失败检测器、基于UCT的MCTS引擎，以及对话状态追踪与奖励机制。

**📊 数据集**

使用12个临床构建的虚拟人物、150条标注对话段落，以及来自HuggingFace的3个公开聊天机器人模型进行评估。

**📈 对比分析**

与随机、贪婪和beam搜索对比，MCTS发现2.3倍更多失败路径，循环迭代次数减少47%，失败检测器宏F1为0.71，说明方法效果显著。

**⚠️ 局限性**

局限性包括样本人物有限、对真实用户数据缺乏验证、部分类别（如共情疲劳）检测召回率低、专家验证样本不足等。

---

## 262. DIAL: Decentralized I/O AutoTuning via Learned Client-side Local Metrics for Parallel File System

**arXiv ID:** 2602.22392 | [PDF](https://arxiv.org/pdf/2602.22392v1)

**作者:** Md Hasanur Rashid `[一作]` (University of Delaware), Dong Dai `[通讯]` (University of Delaware)

**通讯引用:** 696 | [OpenAlex ID](https://openalex.org/A5012002926)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DIAL 框架，利用客户端本地指标进行去中心化的 I/O 参数自动调优

**💡 创新点**

创新点是仅依赖本地可观测指标，无需全局 I/O 模式分析，使用机器学习模型做概率预测

**🔧 技术方法**

机器学习（GBDT）、Lustre 文件系统、统计采样与参数搜索

**📊 数据集**

使用 Filebench 生成的读写工作负载、H5bench 以及深度学习 I/O 基准 DLIO

**📈 对比分析**

与默认配置及最优手动配置对比，DIAL 在 VPIC、BDCATS 和 DLIO 应用中接近最优，提升至约 1.75 倍

**⚠️ 局限性**

缺点是对可实时更改参数有限，仍需改造文件系统以暴露更多可调参数，且对极端模式的预测仍不够精确

---

## 263. TCM-DiffRAG: Personalized Syndrome Differentiation Reasoning Method for Traditional Chinese Medicine based on Knowledge Graph and Chain of Thought

**arXiv ID:** 2602.22828 | [PDF](https://arxiv.org/pdf/2602.22828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 264. Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation

**arXiv ID:** 2602.23188 | [PDF](https://arxiv.org/pdf/2602.23188v1)

**作者:** Ismaël Zighed `[一作]` (Sorbonne Université), Taraneh Sayadi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个基于变分自编码器（VAE）与Transformer的可参数化降维模型（ROM），并通过数据同化（Ensemble Kalman Filter）实现仅利用稀疏观测进行实时自适应训练。

**💡 创新点**

创新点在于：①只需对VAE进行微调即可纠正跨参数域的低维流形偏差，显著减少训练成本；②将ROM的概率性输出与EnKF结合，利用极少的传感器数据完成高维状态重构，实现近实时域适应。

**🔧 技术方法**

采用技术包括：变分自编码器、Transformer时序网络、跨参数注意力机制、数据同化（Ensemble Kalman Filter）、Wasserstein距离评估、目标函数基于预测均值与分析均值的欧式损失。

**📊 数据集**

数据集来源于二维无粘流体绕椭圆体的Navier‑Stokes数值模拟，网格131×100，Reynolds数覆盖80–140，总共约3033帧训练样本，验证集包含7个不同Re值。

**📈 对比分析**

通过2‑Wasserstein距离、能量距离以及L1/L2重构误差与UQ方差进行评估；在Re=140时，仅对VAE微调即可将误差从93%降低至约30%，耗时从10小时减至约15分钟；整体误差下降≈70%，在Re=90–120区间保持高精度。

**⚠️ 局限性**

局限性包括：①仅适用于参数变化未导致分岔的系统；②当Re跨越新的动力学分支时，VAE微调无法完全恢复；③对观测分布和传感器布局敏感，过度稀疏可能导致重构误差上升；④在高维度流场的更复杂物理过程中，计算量和存储需求仍然较高。

---

## 265. HELMLAB: An Analytical, Data-Driven Color Space for Perceptual Distance in UI Design Systems

**arXiv ID:** 2602.23010 | [PDF](https://arxiv.org/pdf/2602.23010v1)

**作者:** Gorkem Yildiz `[一作]` `[通讯]` (Independent Researcher), Gorkem Yildiz (Independent Researcher)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种72参数的HelmLab颜色空间，优化UI设计系统的色差预测与生成。

**💡 创新点**

端到端联合学习变换与距离度量，嵌入Helmholtz–Kohlrausch光度校正、精确中性轴校正以及可旋转色度平面，以提升色相对齐并保持可逆性。

**🔧 技术方法**

采用解析变换（线性矩阵、幂压缩、傅里叶色相校正、HK嵌入、光度与色度微调等）与STRESS损失、L‑BFGS‑B优化，逆变换用Newton迭代与PCHIP。

**📊 数据集**

使用3,813对心理物理色差实验（BFD‑P、Witt、RIT‑DuPont、Leeds等）作为训练集，并在He 2022（82对）与MacAdam 1974（128对）上交叉验证。

**📈 对比分析**

与CIEDE2000、Oklab、CAM16‑UCS等基线在COMBVD上对比，HelmLab取得23.22 STRESS（比CIEDE2000低20.4%），交叉验证亦优于CIEDE2000。

**⚠️ 局限性**

受限于2°标准观察者、参数多、色相误差≈16°、未针对不同视角训练、仅适用于屏幕UI、部分子数据集表现不如CIEDE2000、缺乏统计不确定性分析。

---

## 266. Flip Distance of Triangulations of Convex Polygons / Rotation Distance of Binary Trees is NP-complete

**arXiv ID:** 2602.22874 | [PDF](https://arxiv.org/pdf/2602.22874v1)

**作者:** Joseph Dorfer `[一作]` `[通讯]` (Graz University of Technology), Joseph Dorfer (Graz University of Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文研究了凸多边形的三角剖分之间的最短翻转序列的计算复杂性，证明了该问题是NP-hard的。

**💡 创新点**

创新点在于通过发展新的技术来处理三角剖分的翻转序列，解决了长期未解的复杂性问题。

**🔧 技术方法**

使用了组合数学和图论的技术，特别是冲突图和线性表示法。

**📊 数据集**

使用了凸多边形的三角剖分作为数据集，涉及到的多边形顶点数量为n。

**📈 对比分析**

与现有方法的比较显示，计算最短翻转序列的复杂性是NP-hard，且在某些情况下，现有的多项式时间算法无法解决该问题。

**⚠️ 局限性**

限制在于该研究主要集中在凸多边形的情况，其他类型的多边形或点集的复杂性仍然是开放问题。

---

## 267. An Artificial Intelligence Framework for Joint Structural-Temporal Load Forecasting in Cloud Native Platforms

**arXiv ID:** 2602.22780 | [PDF](https://arxiv.org/pdf/2602.22780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 268. Bridging Latent Reasoning and Target-Language Generation via Retrieval-Transition Heads

**arXiv ID:** 2602.22453 | [PDF](https://arxiv.org/pdf/2602.22453v1)

**作者:** Shaswat Patel `[一作]` (New York University), Eunsol Choi `[通讯]` (New York University)

**通讯引用:** 4263 | [OpenAlex ID](https://openalex.org/A5035142405)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言LLM中的注意力头进行细粒度分析，发现并定义了新的Retrieval‑Transition Heads（RTH），并通过遮蔽实验验证其在跨语言推理中的关键作用。

**💡 创新点**

创新点在于提出RTH这一专门负责从语言无关的潜在空间映射到目标语言生成的稀疏注意力头，并将其与传统检索头（RH）区分，证明RTH在多语言Chain‑of‑Thought推理中更为重要。

**🔧 技术方法**

技术方法包括：扩展 Needle‑In‑A‑Haystack (NIAH) 任务到多语言；计算检索分数 (RS) 与跨语言检索分数 (RTS)；对头进行基于分数的排名与层级分布分析；使用因果遮蔽（masking）实验评估头的重要性；Spearman相关系数评估不同头集的相似性。

**📊 数据集**

使用的数据集包括：多语言NIAH数据集（English、German、Chinese、Swahili 等），四个跨语言推理/QA 基准（MMLU‑ProX、MGSM、MLQA、XQuaD），以及评估时用的公开模型（Qwen‑2.5、Llama‑3.1、Phi‑3.5）。

**📈 对比分析**

通过将随机遮蔽、仅遮蔽RH、仅遮蔽RTH三种方式在同一模型和基准上进行对比，结果显示遮蔽RTH会导致性能下降显著高于遮蔽RH，尤其在MGSM（数理推理）和MMLU‑ProX（多步推理）上，RTH遮蔽的平均分数下降可达30%+，而RH遮蔽仅下降10%以内。

**⚠️ 局限性**

主要局限包括：RTS 需要以单一语言（通常是英语）作为潜在空间的近似，可能无法完全捕捉模型真实的语言无关表示；研究仅关注注意力头，未探究 MLP “shared neurons” 等其他模块的作用；对低资源语言的评估受限于数据稀缺，且方法对语言之间的语义对齐假设较强。

---

## 269. IRSDE-Despeckle: A Physics-Grounded Diffusion Model for Generalizable Ultrasound Despeckling

**arXiv ID:** 2602.22717 | [PDF](https://arxiv.org/pdf/2602.22717v1)

**作者:** Shuoqi Chen `[一作]` (Dartmouth), Geoffrey P. Luke `[通讯]` (Dartmouth)

**通讯引用:** 2682 | [OpenAlex ID](https://openalex.org/A5073239840)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种基于IRSDE扩散模型的超声图像去斑方法。

**💡 创新点**

创新点在于将医学图像配对数据通过物理仿真生成、在IRSDE框架下实现物理一致的去斑，并加入跨模型不确定性评估。

**🔧 技术方法**

主要使用IRSDE扩散模型、时间条件U‑Net网络、5折交叉集成与MATLAB UltraSound Toolbox仿真。

**📊 数据集**

训练数据由多源MRI（MRNet、Duke Liver、BrainMetShare、Duke‑Breast）与MUST生成的合成B‑mode超声配对构成，测试数据为未见UMD模拟集和真实BrEaST Lesions USG。

**📈 对比分析**

与经典滤波、U‑Net、GAN等方法对比，IRSDE去斑在PSNR/SSIM/LPIPS等指标上均领先，尤其在模拟数据上PSNR从20.95提升至23.03，真实数据上显著提升CNR并保持病灶边界。

**⚠️ 局限性**

局限包括对单一仿真探头的依赖导致跨探头泛化不足，推断时步数大导致时延，且模型仍可能出现幻觉与结构错误。

---

## 270. Dynamic Hierarchical Birkhoff-von Neumann Decomposition for All-to-All GPU Communication

**arXiv ID:** 2602.22756 | [PDF](https://arxiv.org/pdf/2602.22756v1)

**作者:** Yen-Chieh Wu `[一作]` (National Tsing Hua University), H. Jonathan Chao `[通讯]` (New York University)

**通讯引用:** 6729 | [OpenAlex ID](https://openalex.org/A5071272821)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出一种动态分层Birkhoff–von Neumann分解框架，用于两层GPU集群的全对全通信调度，并加入轻量级的局部负载平衡；

**💡 创新点**

创新点在于：①在两层结构下构建分层BvN分解，显著降低分解复杂度；②在每帧边界进行局部GPU间流量重塑，实现微层负载均衡；③结合动态帧尺寸(DFS)实现在线调度，并给出可证明的队列稳定性；

**🔧 技术方法**

主要技术包括：Birkhoff–von Neumann分解、分层分解算法、局部单元转移平衡、动态帧尺寸调度、泊松流量下的队列理论分析；

**📊 数据集**

使用仿真数据：在8台服务器、每台2个GPU的16×16交叉开关模型下，生成两种流量模型（均匀与热点）来评估性能；

**📈 对比分析**

与不做局部平衡的DFS分层BvN方案对比，结果显示在均匀与热点两种流量下，加入平衡后平均帧长显著下降，热点场景的改进尤为明显；

**⚠️ 局限性**

局限性：仅考虑两层集群、单速率互连、泊松到达假设；未考虑多跳网络、异构链路速率、可变数据包大小及更复杂的拓扑约束。

---

## 271. Search-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training

**arXiv ID:** 2602.22576 | [PDF](https://arxiv.org/pdf/2602.22576v1)

**作者:** Tianle Xia `[一作]` (Tencent), Jie Jiang `[通讯]` (Tencent)

**通讯引用:** 1540 | [OpenAlex ID](https://openalex.org/A5101944041)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种路径中心奖励塑形框架，用于训练代理式检索增强生成（Agentic RAG）模型，使其在多步推理中能够动态决定检索与生成。

**💡 创新点**

创新点在于：① 双轨道路径评分——结合自一致性与参考对齐两种视角，顺序无关匹配评估推理路径质量；② 软结果评分——对失败样本赋予部分奖励，提升样本效率，解决传统稀疏奖励问题。

**🔧 技术方法**

技术实现包括：基于强化学习（GRPO/PPO）训练；利用高能力LLM生成离线参考规划并进行投票；E5检索器+Wikipedia知识库；软格式奖励和路径奖励的加权组合。

**📊 数据集**

实验数据集涵盖：公开QA基准（NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、Musique、Bamboogle）以及内部广告QA数据集AD-QA。

**📈 对比分析**

与Direct、CoT、RAG、IRCoT、Search-o1、Search-R1和HiPRAG等方法对比，平均准确率提升7.7点，AD-QA提升20.6点；收敛速度加快，交互轮数更少，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括：依赖外部LLM评估器，训练成本高且对评估器质量敏感；在极端复杂或非检索型任务中效果可能受限；对不同检索器或知识库的鲁棒性未做深入验证。

---

## 272. Accelerating Local LLMs on Resource-Constrained Edge Devices via Distributed Prompt Caching

**arXiv ID:** 2602.22812 | [PDF](https://arxiv.org/pdf/2602.22812v1)

**作者:** Hiroki Matsutani `[一作]` (Keio University), Naoto Sugiura `[通讯]` (Keio University)

**通讯引用:** 862 | [OpenAlex ID](https://openalex.org/A5109288132)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在低端边缘设备上实现了分布式提示缓存，提升本地LLM推理性能

**💡 创新点**

创新点在于将提示前缀缓存扩展到多设备协作，并引入Bloom过滤器本地目录以显著降低通信开销

**🔧 技术方法**

采用了LLM KV缓存、Bloom过滤器、Redis数据库、Wi‑Fi网络通信、llama.cpp、Hiredis和libbloom等技术

**📊 数据集**

使用Gemma-3 270M/1B模型与MMLU多领域问答数据集进行实验

**📈 对比分析**

通过在Raspberry Pi Zero 2W（低端）与Pi 5（高端）上测量TTFT/TTLT，低端设备完整缓存时TTFT下降93.12%、TTLT下降50.07%，而高端设备则略增

**⚠️ 局限性**

局限在高端设备网络开销和Bloom误判导致的额外延迟，且对大模型或高带宽网络的依赖较大

---

## 273. DPSQL+: A Differentially Private SQL Library with a Minimum Frequency Rule

**arXiv ID:** 2602.22699 | [PDF](https://arxiv.org/pdf/2602.22699v1)

**作者:** Tomoya Matsumoto `[一作]` (LY Corporation), Satoshi Hasegawa `[通讯]` (LY Corporation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DPSQL+，一种模块化 SQL 库，既实现用户级 (ε,δ)-差分隐私，又强制执行最小频率规则，可支持多查询隐私会计与多种数据库后端。

**💡 创新点**

创新点包括：将最小频率规则与差分隐私统一强制；采用双阈值过滤与 Bézier 机制提升二次统计的精度；内置 RDP/PLD 隐私会计；采用代理式模块化架构，方便后端扩展。

**🔧 技术方法**

使用了静态查询验证、贡献上限、双阈值过滤、Gaussian 机制、Bézier 估计、RDP/PLD 会计、OpenDP 安全噪声等技术。

**📊 数据集**

在 TPC‑H 基准（规模因子 1.0）及其 10 类查询模板上进行实验。

**📈 对比分析**

与 Qrlew 与 SmartNoise SQL 进行对比，评估平均相对误差（MRE）和在固定全局预算下可执行查询数。DPSQL+ 在基本聚合与高阶统计上误差最低，在多查询场景下可执行的查询数多于现有库。

**⚠️ 局限性**

局限性包括：受限于查询约束（仅聚合、禁止子查询）导致表达能力受限；会计仅在会话内，无法跨重启持久化；未覆盖侧信道或运行时泄露，需要额外安全措施。

---

## 274. Systems-Level Attack Surface of Edge Agent Deployments on IoT

**arXiv ID:** 2602.22525 | [PDF](https://arxiv.org/pdf/2602.22525v1)

**作者:** Zhonghao Zhan `[一作]` (Imperial College), Hamed Haddadi `[通讯]` (Imperial College)

**通讯引用:** 9662 | [OpenAlex ID](https://openalex.org/A5043326652)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对大规模语言模型（LLM）代理在IoT边缘设备中的三种部署架构（云端、边缘本地集群、混合）进行了实验测量，量化了攻击面、协调失效、数据主权、审计延迟等系统级安全指标，并发现了协调状态偏离与信任侵蚀等未预料的失效模式。

**💡 创新点**

创新点包括：①首次将部署架构视为影响安全性的主要因素，并将其细分为可测量的安全属性；②通过实测揭示两种新型失败（协调状态偏离、诱导信任侵蚀）和边缘恢复时无形降权的安全缺陷；③提出系统层面的攻击面分类及对应度量标准，为未来安全设计提供可量化依据。

**🔧 技术方法**

技术手段主要有：MQTT发布/订阅作为协调层，Tailscale VPN实现本地网络，OpenClaw LLM代理在 Mac mini、Moto G35、Intel NUC 上部署，使用 Home Assistant 与实际IoT设备互联；测量actuation‑to‑audit 延迟、完整性链、数据出站量、failover 窗口等；对比云端与本地执行的差异。

**📊 数据集**

使用的数据集为实时实验记录：由三台边缘设备（Mac mini、Moto G35、NUC）与若干家用IoT设备（灯、开关、摄像头、传感器等）产生的消息流与操作日志；未使用公开大规模数据集，全部基于实验产生。

**📈 对比分析**

比较方法：在同一测试bed中对三种架构进行同等任务的执行，测量每个指标；结果显示：边缘本地部署在actuation‑to‑audit 延迟上平均约23‑64 ms，云端则无此指标；数据出站量在正常运行下为0 B，而云端为约65 KB；failover窗口在边缘本地可达35 s，云端无此盲区；合作代理的完整性链为100%但在攻击下可被轻易绕过。

**⚠️ 局限性**

局限性：①实验仅基于单一三节点测试bed，未验证多代理/多broker/不同网络子系统；②failover 测试仅涉及 WiFi‑ADB 路径，其他失效模式未评估；③未实现或评估缓解方案（如 HMAC/ACL、主权感知回退策略）；④数据对比非完全匹配，云端基准仅包含推理API；⑤将 S1a–c 视为不同攻击面可能掩盖其本质为单一未认证协调缺陷。

---

## 275. CMSA-Net: Causal Multi-scale Aggregation with Adaptive Multi-source Reference for Video Polyp Segmentation

**arXiv ID:** 2602.22821 | [PDF](https://arxiv.org/pdf/2602.22821v1)

**作者:** Tong Wang `[一作]` (Southeast University), Yutong Xie `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6423 | [OpenAlex ID](https://openalex.org/A5011835422)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一种名为 CMSA‑Net 的视频息肉分割框架，能够在实时条件下对结肠镜视频进行高精度息肉分割。

**💡 创新点**

核心创新点是：① Causal Multi‑scale Aggregation (CMA) 模块，通过因果注意力在多尺度空间聚合历史帧信息；② Dynamic Multi‑source Reference (DMR) 策略，根据语义可分离度和置信度动态维护多源参考帧。

**🔧 技术方法**

采用 Res2Net‑50 / PVTv2‑B2 作为特征提取骨干；多尺度对齐与 1×1 卷积；因果注意力机制；多尺度跨时域聚合；交叉熵、Dice、IoU 以及加权 BCE 损失；以及多阶段解码器。

**📊 数据集**

在最大的 VPS 数据集 SUN‑SEG 上进行实验，包含 112 条训练视频与 4 个测试子集（Easy‑Seen/Unseen、Hard‑Seen/Unseen）。

**📈 对比分析**

与 12 先进方法（NVS、IPS、VPS 系列）对比，CMSA‑Net 在所有子集均实现最高 Dice、IoU、S_α 等指标（例如 Easy‑Seen Dice 95.1%，Hard‑Unseen Dice 92.6%），比最强基线提升约 1–2%；同时保持约 36–38 FPS 的实时推理速度。

**⚠️ 局限性**

局限性：① 依赖多源参考帧，虽然提升了鲁棒性但在极长视频或显著外观漂移时仍可能出现参考帧不稳定；② 计算量相较单源方法略大，尽管已实现实时，但对低算力设备的适配仍有待改进；③ 仅在 SUN‑SEG 上验证，缺乏跨数据集通用性评估。

---

## 276. Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization

**arXiv ID:** 2602.22675 | [PDF](https://arxiv.org/pdf/2602.22675v1)

**作者:** Qianben Chen `[一作]` (OPPO AI Agent Team), Wangchunshu Zhou `[通讯]` (OPPO AI Agent Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了搜索更多、思考更少（SMTL）并行代理框架，利用并行子任务执行、动态计划重构和结构化上下文管理实现高效长程搜索，并通过自动化数据合成管线生成多类型搜索任务。

**💡 创新点**

创新点在于将传统顺序推理转为并行证据获取，结合周期性计划重构和溢出触发的上下文压缩；同时构建统一的多任务数据管线，支持确定性问答与开放式研究任务的共同训练。

**🔧 技术方法**

采用大语言模型+工具调用（Web搜索、页面爬取等）、并行代理工作流、计划驱动上下文管理、LightRAG图构建、REINFORCE LOO强化学习、LLM-as-judge评估、token级损失与序列级重要性采样。

**📊 数据集**

使用的基准数据集包括 BrowseComp、GAIA、XBench‑DeepSearch、WebWalkerQA、FRAMES、SEAL‑0（深度搜索）以及 Deep Research Bench RACE（深度研究）以及内部通过 TaskCraft 与 LightRAG 生成的合成任务。

**📈 对比分析**

与多种30B级开放源代码代理、商业模型和深度研究系统对比，SMTL 在 BrowseComp（44.6%→48.6%），XBench‑DeepSearch（78.0%）和 WebWalkerQA（74.9%）等指标均超越同级别基线，且平均步骤显著减少（从 75.2 步降至 60.4 步）并显著降低推理延迟；在 Deep Research Bench RACE 上取得 45.9% 的总体分数，明显高于 WebSailor‑32B 等基线。

**⚠️ 局限性**

局限性包括：仍受限于工具访问和检索质量；在极长或极高复杂度任务中可能需要更大上下文预算；RL 奖励仅关注答案正确性，可能不足以驱动更高质量的多模态输出；并行工作流的实现复杂度较高，依赖于高效的工具调用与计划重构机制。

---

## 277. Skewed Dual Normal Distribution Model: Predicting 1D Touch Pointing Success Rate for Targets Near Screen Edges

**arXiv ID:** 2602.22454 | [PDF](https://arxiv.org/pdf/2602.22454v1)

**作者:** Nobuhito Kasahara `[一作]` (Meiji University), Homei Miyashita `[通讯]` (Meiji University)

**通讯引用:** 794 | [OpenAlex ID](https://openalex.org/A5091629034)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过对两组1维智能手机触摸实验，提出并验证了 Skewed Dual Normal Distribution Model，用以预测屏幕边缘附近的成功率。

**💡 创新点**

创新点在于将双高斯模型扩展为偏态正态分布模型，能够显式捕捉边缘导致的分布偏斜，并提供可解释的参数（如阈值约6 mm），解释了“与边缘一起点击”策略的有效性。

**🔧 技术方法**

采用偏态正态分布理论、回归分析、似然比检验、LOOCV交叉验证，并与 Lasso、Random Forest、SVR、MLP 等机器学习基线进行比较。

**📊 数据集**

使用两组实验数据，共 30 名右撇子参与者（每组 15 人），在 Google Pixel 6a 手机上记录不同目标尺寸与边缘距离下的点击坐标和成功率。

**📈 对比分析**

与传统双高斯模型相比，Skewed 模型在 R² 上提升至约 0.95（LOOCV 与实际实验一致），机器学习模型虽然在某些设置下稍高，但解释性不足；模型在边缘区成功率提升，验证了理论假设。

**⚠️ 局限性**

局限性包括仅测试 1 D 指点、仅左/下边缘、仅主手食指、仅 Pixel 6a 机型、未验证 2 D 或其他手势、未考虑速度-准确性权衡和不同设备边框形状等因素。

---

## 278. Enhancing Persuasive Dialogue Agents by Synthesizing Cross-Disciplinary Communication Strategies

**arXiv ID:** 2602.22696 | [PDF](https://arxiv.org/pdf/2602.22696v1)

**作者:** Shinnosuke Nozue `[一作]` (Tohoku University), Jun Suzuki `[通讯]` (Tohoku University)

**通讯引用:** 8054 | [OpenAlex ID](https://openalex.org/A5001456824)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一套跨学科说服策略框架，并在P4G与DailyPersuasion数据集上进行验证；

**💡 创新点**

将行为经济学、社会心理学与传播学的多种说服技巧整合为31种策略，显著提升对低意向受众的说服效果；

**🔧 技术方法**

采用大型语言模型（LLM）结合ProCoT链式思维提示，动态推断并生成策略化对话；

**📊 数据集**

使用Persuasion for Good (P4G) 与 DailyPersuasion 两大对话数据集进行实验；

**📈 对比分析**

通过与简单基线与P4G标注策略的ProCoT进行自动与人工评估对比，ProCoT-rich在成功率、意向提升（AII）和赢率方面均表现突出，尤其在低初始意向场景；

**⚠️ 局限性**

评估依赖LLM模拟器，缺乏真实人类交互；未考虑捐赠金额、策略组合与跨文化适用性等因素，导致实验结果在实际场景中的推广有限。

---

## 279. Engineered Simultaneity: The Physical Impossibility of Consolidated Price Discovery Across Spacelike-Separated Exchanges

**arXiv ID:** 2602.22350 | [PDF](https://arxiv.org/pdf/2602.22350v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (DAEDAELUS), Paul Borrill (DAEDAELUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对美国股票市场的全国最佳买卖价（NBBO）进行“工程同步性”概念化，证明其基于绝对同步的假设在相对论框架下无法成立；

**💡 创新点**

创新点在于提出“工程同步性”这一概念，并将其与特殊相对论与Lamport分布式事件排序理论相结合，首次量化同步错误带来的约5亿美元年化利润；

**🔧 技术方法**

使用了Minkowski时空分析、Lamport定理及实际延迟测量技术来证明NBBO的时间标记本质上是一种主观同步约定；

**📊 数据集**

采用美国主要交易所数据中心的物理位置与光时延、SIP与直接馈送的延迟数据以及伦敦交易所事件日志作为实验数据；

**📈 对比分析**

通过将SIP统一时钟下的NBBO结果与直接馈送的时间戳对比，发现SIP延迟约1128 μs而直接馈送仅约10–20 μs，导致与真实市场状态差距超过50倍，进而使高频交易者获得显著收益；

**⚠️ 局限性**

研究局限在于仅聚焦NBBO同步假设，未考虑其他市场机制或不同地理位置SIP可能产生的变化，以及对实际监管影响的进一步评估。

---

## 280. Toward Expert Investment Teams:A Multi-Agent LLM System with Fine-Grained Trading Tasks

**arXiv ID:** 2602.23330 | [PDF](https://arxiv.org/pdf/2602.23330v1)

**作者:** Kunihiro Miyazaki `[一作]` (Japan Digital Design), Stefan Zohren `[通讯]` (University of Oxford)

**通讯引用:** 3396 | [OpenAlex ID](https://openalex.org/A5090331439)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一套基于大型语言模型（LLM）的多智能体交易系统，并通过细粒度任务拆解（如技术分析指标、财务指标、新闻情绪、宏观评估等）对投资分析流程进行重构，随后在日本TOPIX 100股票上进行回测。

**💡 创新点**

创新点在于：① 将投资分析细分为与真实交易团队相似的具体任务而非笼统指令；② 通过系统的对比实验和文本分析验证细粒度任务能提升性能并增强可解释性；③ 结合信息传播度量（词汇log‑odds、向量相似度）揭示知识流动；④ 在回测之外进行组合优化，展示系统在真实市场中的潜在收益。

**🔧 技术方法**

使用技术包括：GPT‑4o LLM + 提示工程（细粒度 vs 粗粒度）；多智能体架构（技术、量化、定性、新闻、行业、宏观、组合经理）；技术指标（RoC、Bollinger、MACD、RSI、KDJ）；财务指标（ROE、PE、EV/EBITDA等）；新闻情绪提取；宏观变量（利率、通胀、就业、汇率等）；信息传播分析（词频log‑odds、文本向量余弦相似度）；Sharpe ratio 与 Mann‑Whitney U检验；组合优化（等风险贡献法）。

**📊 数据集**

使用的数据集包括：TOPIX 100股票日价格（Yahoo Finance）、季度/年度财务报表（EDINET API）、新闻标题与摘要（Nikkei、Reuters、Bloomberg via Ceek.jp）、宏观经济指标（FRED、Yahoo Finance），回测时段为2023‑09至2025‑11（约两年）。

**📈 对比分析**

对比方法：在10–50只股票的多组合规模下，分别使用细粒度和粗粒度任务，计算Sharpe ratio并用Mann‑Whitney U检验显著性；进行留一实验和消融实验评估各智能体贡献。结果显示，细粒度任务在大多数规模下显著优于粗粒度，技术智能体是主要性能驱动；组合优化与TOPIX 100低相关，混合投资进一步提升Sharpe ratio。

**⚠️ 局限性**

局限性包括：① 受LLM知识截止（2023‑08）限制，回测仅覆盖两年；② 细粒度提升可能部分归因于词汇或偏好，而非任务拆解本身；③ 仅在日本市场、单一LLM（GPT‑4o）和单一回测期间验证，缺乏跨市场、跨模型的稳健性检验；④ 可能存在语言偏差与信息泄露风险，需要更严格的泄露控制与长周期验证。

---

## 281. TriLite: Efficient Weakly Supervised Object Localization with Universal Visual Features and Tri-Region Disentanglement

**arXiv ID:** 2602.23120 | [PDF](https://arxiv.org/pdf/2602.23120v1)

**作者:** Arian Sabaghi `[一作]` (University of Antwerp), José Oramas `[通讯]` (University of Antwerp)

**通讯引用:** 1212 | [OpenAlex ID](https://openalex.org/A5055529222)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出单阶段弱监督目标定位框架TriLite，冻结自监督ViT并用极少可训练参数完成定位和分类。

**💡 创新点**

创新点在于TriHead三通道分离前景、背景与模糊区并加入对抗背景损失，显著提升完整目标覆盖且参数量不到80万。

**🔧 技术方法**

使用冻结的DINOv2预训练Vision Transformer，TriHead卷积头和轻量分类层。

**📊 数据集**

在CUB-200-2011、ImageNet-1K以及OpenImages三大弱监督定位/分割基准上评估。

**📈 对比分析**

相较现有方法，TriLite在ImageNet-1K Top-1/5/GT分别提升+0.3%/+2.2%/+2.9%，在CUB-200-2011亦超越GenPromp与C2AM，并在OpenImages PxAP 达到73.3%，是目前最优且参数更少。

**⚠️ 局限性**

局限性在于单目标假设、对多实例/多类别的支持不足，且遮挡导致部分激活分散。

---

## 282. Set-based v.s. Distribution-based Representations of Epistemic Uncertainty: A Comparative Study

**arXiv ID:** 2602.22747 | [PDF](https://arxiv.org/pdf/2602.22747v1)

**作者:** Kaizheng Wang `[一作]` (Nanyang Technological University), Siu Lun Chau `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比分布式（如贝叶斯网络、深度集成）与集合式（可信集）第二阶不确定性表示，使用相同的预测分布集合，评估多种不确定性度量在选择性预测和OOD检测上的性能；

**💡 创新点**

设计了控制实验框架消除模型差异影响；统一评估多种度量、模型与数据集；发现性能高度依赖表示与度量交互，且OOD检测比选择性预测更能区分表示；

**🔧 技术方法**

使用贝叶斯神经网络、SVI、MCDO、Deep Ensembles（BatchEns、MaskEns、PackEns）；计算分布度量（MI、LWV、WD）和集合度量（H_diff、GH、MMI）；统计Wilcoxon检验、net-win排名；

**📊 数据集**

CIFAR-10、SVHN、FMNIST（OOD对比）；Camelyon17医学WSI（ID与分布漂移）；SeaShip、SeaShip-C、SeaShip-O（船舶识别与OOD）；

**📈 对比分析**

通过AUARC（选择性预测）和AUROC（OOD）评估，10次独立跑，使用Wilcoxon检验比较指标；结果显示无单一表示始终优，WD在分布表示、GH在集合表示表现最强，OOD检测对表示差异更敏感；

**⚠️ 局限性**

未覆盖Dirichlet、随机集等其他第二阶表示；仅使用概率区间Credal，未考虑其他构造；GH在大类别时计算量高；实验规模有限，未扩展到主动学习等更广泛任务；

---

## 283. MaRI: Accelerating Ranking Model Inference via Structural Re-parameterization in Large Scale Recommendation System

**arXiv ID:** 2602.23105 | [PDF](https://arxiv.org/pdf/2602.23105v1)

**作者:** Yusheng Huang `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于结构重参数化的矩阵重参数化推理框架 MaRI，消除了排序模型中用户侧冗余计算，并通过图着色算法自动定位可优化的 MatMul 节点，同时对输入特征和参数进行重排以提升推理效率。

**💡 创新点**

创新点包括：①使用结构重参数化将用户、项目、交叉特征的矩阵乘法拆分为三项并延迟张量复制，从而实现无精度损失的加速；②提出图着色算法自动识别可优化节点；③强调输入特征布局的重要性并给出重排方案以避免碎片化导致的性能下降。

**🔧 技术方法**

技术手段包括：结构重参数化的矩阵乘法重写（MatMul_MaRI）、图着色算法（GCA）、特征与参数重排、以及在工业推理引擎中的实现与部署。

**📊 数据集**

使用了 Kuaishou 直播平台的实际在线推荐系统数据进行离线模拟（采样真实业务数据）和在线 A/B 测试（覆盖约5%用户），无公开公开数据集。

**📈 对比分析**

与传统的 Vanilla Inference（VanI）和 User-Side One-Shot Inference（UOI）进行对比，离线实验显示在批量大小≥500时，MaRI 维持 1.1× 的速度提升；在线 A/B 测试中，MaRI 在粗排阶段平均提升 1.32×、P99 为 1.26×，整体延迟下降约 2.24%/2.27%，硬件资源使用减少 5.9%。

**⚠️ 局限性**

局限性包括：①需要手动或自动重排特征以保证特征块不碎片化，碎片化会导致性能下降近 38%；②对不同模型结构的适用性仍需验证；③在本实验中未实现项目侧计算下移，因带宽限制而未能进一步提升性能；④性能提升受批量大小和用户特征维度影响，批量过小或用户特征维度不明显时加速效果有限。

---

## 284. Faster algorithms for graph homomorphism via tractable constraint satisfaction

**arXiv ID:** 2602.23000 | [PDF](https://arxiv.org/pdf/2602.23000v1)

**作者:** Clément Carbonnel `[一作]` `[通讯]` (National Centre for Scientific Research), Clément Carbonnel (National Centre for Scientific Research)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在时间 2^O(n)·h^O(1) 和多项式空间内解决从图 G 到 H 的同态判定问题的算法（H 为不含某拓扑子图的图族），并给出了对奇环同态的改进随机算法；

**💡 创新点**

创新点在于将持久多数颜色化（persistent majority colorings）与轨道布局（track layouts）相结合，构造可通过可判定约束满足问题求解的可约子问题，从而实现单指数时间和多项式空间；

**🔧 技术方法**

核心技术包括：树分解与层次分解、轨道布局与持久多数颜色化、约束满足问题的Sherali–Adams松弛与分数多项式多项式多边形、通用代数的分数多项式极值（fractional polymorphisms）以及随机采样法来提升奇环同态的指数基；

**📊 数据集**

本文为理论算法研究，无使用具体数据集；

**📈 对比分析**

与已知的 2.443^n 及其改进版（如 2^nn^O(1)）以及奇环同态的 √2^n、c_k^n 进行对比，证明在不含拓扑子图的图族上实现了单指数时间、在奇环上实现了比 √2 更小的指数基（α_k<2）并保持多项式空间；

**⚠️ 局限性**

局限性：算法仅适用于不含拓扑子图的图族或可计算持久多数颜色化的图族，奇环算法为随机化且无法直接推广到奇环优化版本；此外持久多数颜色化的上界可能不够紧，整体时间常数与实际实现相关。

---

## 285. veScale-FSDP: Flexible and High-Performance FSDP at Scale

**arXiv ID:** 2602.22437 | [PDF](https://arxiv.org/pdf/2602.22437v1)

**作者:** Zezhou Wang `[一作]` (ByteDance Seed), Xin Liu `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了可扩展、高灵活性的大模型分布式训练框架 veScale，支持结构感知训练、任意粒度分片、以及高性能通信与内存管理。

**💡 创新点**

引入了 RaggedShard 分片格式、基于 NP‑hard 优化的分片规划算法、以及 Distributed Buffer 零拷贝通信原语；与 PyTorch DTensor 兼容，实现无侵入式结构感知优化。

**🔧 技术方法**

使用了 PyTorch DTensor、ZeRO/FSDP、RaggedShard、分片规划（DP+启发式）、Distributed Buffer、分布式优化器（Muon、8‑bit Adam）、分布式 MoE、混合精度等技术。

**📊 数据集**

在公开模型 GPT‑OSS‑70B、120B 以及内部 MoE 模型上进行实验，规模覆盖 1K‑10K GPU 集群，主要测量吞吐量（tokens/s）和峰值 GPU 内存。

**📈 对比分析**

与 ZeRO v0.17.6、PyTorch FSDP、PyTorch FSDP+、-FSDP 等基线对比；结果显示 veScale 在 MoE 模型上比所有基线快 11%‑66%，在 GPT 模型上提升 5%~，并且峰值内存下降 16%‑30%，能够保持线性扩展到 10k GPU。

**⚠️ 局限性**

规划算法虽高效但仍基于启发式，极端非均匀模型可能需要手动微调；分片需兼容 DTensor，旧版 PyTorch 可能不支持；系统仍存在少量 padding 与通信重叠开销；对 10k GPU 的网络拓扑和硬件资源有一定依赖。

---

## 286. Uni-Animator: Towards Unified Visual Colorization

**arXiv ID:** 2602.23191 | [PDF](https://arxiv.org/pdf/2602.23191v1)

**作者:** Xinyuan Chen `[一作]` (Mississippi State University), Bowen Deng `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 76 | [OpenAlex ID](https://openalex.org/A5101898521)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Uni-Animator，一种基于 Diffusion Transformer 的统一框架，用于同时完成图像与视频素描的高保真上色。

**💡 创新点**

创新点包括：①视觉参考增强（实例补丁嵌入）实现局部色彩精准对齐；②物理细节强化（DINOv2 提取高频纹理）保持细节锐度；③素描动态 RoPE 根据光流自适应调整位置编码，抑制大幅运动下的闪烁。

**🔧 技术方法**

采用 Diffusion Transformer（DiT）+ VAE 编码、CLIP+T5 文本嵌入、DINOv2 物理特征、RAFT 光流、Low‑Rank Adaptation 细化模型。

**📊 数据集**

训练数据：约 5K 手工挑选的动漫视频片段（如《千与千寻》《哆啦A梦》）+ 30K 来自 Sakuga‑42M 数据集；所有素描统一裁剪为 512×512 单通道灰度；测试集来自未见的动画电影。

**📈 对比分析**

与主流图像/视频彩色化方法（ColorizeDiffusion、MangaNinja、MagicColor、ToonCrafter、Anidoc 等）对比，Uni‑Animator 在 SSIM、FID、LPIPS、CLIP Score 与自定义 Temporal Consistency Score 上均实现了或逼近第一名，展示了跨域一致性与实例级细节保留的优势。

**⚠️ 局限性**

局限性：目前仍无法实时处理，输出分辨率受限；对极端大运动或非动漫风格场景的泛化能力尚需验证；缺乏针对多种艺术风格的多模态适配。

---

## 287. Towards Autonomous Memory Agents

**arXiv ID:** 2602.22406 | [PDF](https://arxiv.org/pdf/2602.22406v1)

**作者:** Xinle Wu `[一作]` (National University of Singapore), Yao Lu `[通讯]` (National University of Singapore)

**通讯引用:** 5971 | [OpenAlex ID](https://openalex.org/A5058605138)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出U‑Mem框架，实现LLM的非参数、主动记忆更新，能在不修改模型权重的前提下，通过自我/教师信号、工具验证及专家反馈逐步获取、验证并存储知识；

**💡 创新点**

①成本感知的知识提取级联，从低成本自我/教师层级到工具验证再到专家级别，动态决定何时升级；②语义感知Thompson采样（SA‑CTS），在检索时融合语义相似度与采样不确定性，平衡探索与利用，解决冷启动与奖励偏差；

**🔧 技术方法**

基于LLM的Retrieve‑Infer‑Evolve循环、成本感知级联、语义感知Thompson采样、对比提取、流式记忆维护、优势式贝叶斯更新等技术；

**📊 数据集**

可验证任务：AIME、HotpotQA；不可验证任务：AdvancedIF、HelpSteer3（STEM）；在Gemini‑2.5‑flash、DeepSeek‑chat等强模型上亦做评估；

**📈 对比分析**

与无记忆基线、ReasoningBank、ReMe、MemRL及RL（GRPO）等对比，U‑Mem在所有数据集均优于所有记忆基线，且在部分任务（如HotpotQA Qwen2.5‑7B 52.4% 对比 RL 52.1%，AIME Gemini‑2.5‑flash 54.0% 对比基线 46.7%）匹配或超过RL方法，显著提升性能；

**⚠️ 局限性**

仍需依赖多级知识来源的可用性，成本阈值需手工调优；在结构高度不相似或无明确目标任务时记忆收益有限；对极端分布迁移和长周期持续学习的鲁棒性待进一步验证。

---

## 288. Coded-E2LF: Coded Aperture Light Field Imaging from Events

**arXiv ID:** 2602.22620 | [PDF](https://arxiv.org/pdf/2602.22620v1)

**作者:** Tomoya Tsuchida `[一作]` (Nagoya University), Hajime Nagahara `[通讯]` (Osaka University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种完全基于事件相机的计算成像方法 Coded-E2LF，可从仅事件数据重建 4‑D 光场

**💡 创新点**

创新点包括：① 证明黑色编码模式在事件与强度编码图像之间的等价性；② 设计黑色优先编码序列（BF）和参考感知事件生成（RA）两种改进，显著减少事件量并提升重建质量；③ 通过置换不变性说明编码顺序可任意重排

**🔧 技术方法**

采用事件摄像机、光阑编码（8×8 分区）、深度学习管线（AcqNet 产生事件图，RecNet 重建光场）以及量化噪声仿真等技术

**📊 数据集**

使用 BasicLFSR 公开数据集（训练 29,327 个 64×64×8×8 视角样本，测试 23 个光场）

**📈 对比分析**

与 Habuchi、Habuchi event‑only、CA+E2VID 等方法比较：在 4 个视角下 Baseline+BF+RA 达到 30.40 dB/0.8835 的 PSNR/SSIM，较 Habuchi event‑only 提升 2.65 dB、事件总数下降 66%；在 8 视角时几乎与 Habuchi 同水平

**⚠️ 局限性**

限制包括：仅适用于静止或慢动场景；测量时间约 30 ms 受硬件速度限制；对光阑黑色模式依赖性较强；目前仅支持 8×8 视角

---

## 289. Skarimva: Skeleton-based Action Recognition is a Multi-view Application

**arXiv ID:** 2602.23231 | [PDF](https://arxiv.org/pdf/2602.23231v1)

**作者:** Daniel Bermuth `[一作]` (Institute for Software and Systems Engineering University of Augsburg), Wolfgang Reif `[通讯]` (Institute for Software and Systems Engineering University of Augsburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过多视角三角化获取更高质量的3D骨架数据，提升动作识别模型的准确率。

**💡 创新点**

创新点在于把骨架质量提升作为核心改进点，展示多摄像头三角化对现有最先进模型的显著性能提升，进而主张骨架动作识别应以多视角为标准。

**🔧 技术方法**

采用RapidPoseTriangulation多视角多人体3D姿态估计，结合相机标定与同步校正、骨架重建与跟踪、以及现有GCN（MSG3D、DG-STGCN、ProtoGCN）模型的训练。

**📊 数据集**

主要使用NTU-RGBD-60和NTU-RGBD-120两个公开数据集进行实验。

**📈 对比分析**

与原始单视角骨架、PoseConv3D、NTU-X等方法相比，新三角化骨架在所有模型上都显著提升了准确率；在标准评测下，ProtoGCN+Skarimva达到了97.5%，在一-shot和5-shot任务中也分别提升到76%和84.9%。

**⚠️ 局限性**

局限性包括需要额外摄像头及标定/同步操作，模型对额外手指/面部关键点可能过拟合且计算开销增加；同时尚未充分探讨如何更高效地利用全身关键点。

---

## 290. Towards Long-Form Spatio-Temporal Video Grounding

**arXiv ID:** 2602.23294 | [PDF](https://arxiv.org/pdf/2602.23294v1)

**作者:** Xin Gu `[一作]` (University of Chinese Academy of Sciences), Libo Zhang `[通讯]` (Institute of Software Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种自回归 Transformer 框架 ART-STVG，用于长时段视频的空间-时间目标定位。

**💡 创新点**

创新点在于引入选择性空间与时间记忆池以及级联时空解码器，显著提升对长视频中相关上下文的利用。

**🔧 技术方法**

采用多模态编码器、跨模态自注意力、记忆增强的空间/时间解码器以及自回归处理流程。

**📊 数据集**

在扩展后的 HCSTVG‑v2 长时段（1/3/5 分钟）数据集和原始 HCSTVG‑v2 验证集上训练与评测。

**📈 对比分析**

与多种 SOTA 方法（TubeDETR、STCAT、CG‑STVG、TA‑STVG 等）对比，LF‑STVG 任务上 m_tIoU 与 m_vIoU 分别提升约 9% 与 6%，在短时 STVG 上也保持竞争力。

**⚠️ 局限性**

存在对极短事件、模糊事件边界和高干扰背景场景易失效，且推理速度尚未实时化，需进一步改进模型轻量化与鲁棒性。

---

## 291. AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning

**arXiv ID:** 2602.22268 | [PDF](https://arxiv.org/pdf/2602.22268v1)

**作者:** Changhai Zhou `[一作]` (Fudan University), Weizhong Zhang `[通讯]` (Fudan University)

**通讯引用:** 1285 | [OpenAlex ID](https://openalex.org/A5100693731)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AutoQRA框架，联合优化LLM每层的量化比特宽和LoRA低秩适配器的层级分配，以在给定GPU内存预算下实现高效微调。

**💡 创新点**

创新点在于将量化与适配器容量视为耦合的离散搜索问题，采用两阶段粗细搜索：多精度演化全局探索+信赖域贝叶斯细化，实现对离散组合的高效全局-局部搜索；并通过重要性导向的变异与多精度代理筛选来显著降低搜索成本。

**🔧 技术方法**

主要技术包括：多精度（多层次）演化算法、Hyperband式多精度评估、层级重要性优先（量化敏感性和适配器更新能量）、代理模型回归筛选、可行性投影、离散信赖域高斯过程贝叶斯优化。

**📊 数据集**

实验使用的预训练模型：LLaMA‑3.1/3.2、Qwen‑2.5；微调数据集Alpaca‑52k、HC3；评估任务包括BoolQ、PIQA、HellaSwag、WinoGrande、ARC‑E/C、OpenBookQA、MMLU。

**📈 对比分析**

与LoRA、QLoRA、AdaLoRA、LoftQ、LQ‑LoRA以及基于AMQ的分离式方案对比，AutoQRA在≤4bit内存预算下平均精度提升约1–3%（甚至超过FP16 LoRA），同时内存占用更低；在混合精度下可匹配甚至超越FP16 LoRA。

**⚠️ 局限性**

局限性：搜索过程依赖离线昂贵的多精度微调评估，仍需要在每个新模型或任务上执行一次高成本搜索；对极端内存预算或极大模型（>30B）验证不足；并且对不同量化后端（如QLoRA、AWQ等）的兼容性需进一步验证。

---

## 292. Scale Can't Overcome Pragmatics: The Impact of Reporting Bias on Vision-Language Reasoning

**arXiv ID:** 2602.23351 | [PDF](https://arxiv.org/pdf/2602.23351v1)

**作者:** Amita Kamath `[一作]` (University of California, Los Angeles), Ranjay Krishna `[通讯]` (Allen Institute for AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究 Vision‑Language 模型因报告偏差导致推理能力不足，并通过理论与实证验证空间、时间、计数、否定四种推理类型缺失。

**💡 创新点**

首次将语用学理论与数据分析相结合，系统量化报告偏差对上述四种推理的影响，证明单纯规模放大无法弥补，并提出有针对性的注释指令来提升推理表现。

**🔧 技术方法**

关键词统计与人工标注评估、对比评测、OpenCLIP 与 LLaVA/Molmo 等 VLM 的对照实验、缩放律分析、注释指令实验以及微调验证等技术。

**📊 数据集**

使用 LAION、COCO、PixMo、TallyQA、ControlledImCaps、CountBench、What’sUp、VAW、OpenCLIP 训练集以及自建包含四种推理类型的基准集。

**📈 对比分析**

采用对比选择式与生成式多选评测，评估 VLM 在空间、计数、否定、时间四项推理任务上的准确率；公开 VLM 整体低于人类，规模放大效果有限，针对性指令微调显著提升但仍与人类差距显著。

**⚠️ 局限性**

缺乏大规模按指令生成的数据，缩放律在更高规模下可能失效，实验仅验证微调效果，无法证明完整预训练提升，合成数据细节未公开导致可复现性受限。

---

## 293. Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA

**arXiv ID:** 2602.22617 | [PDF](https://arxiv.org/pdf/2602.22617v1)

**作者:** Hai Huang `[一作]` (Atlassian), Randall Balestriero `[通讯]` (Brown)

**通讯引用:** 913 | [OpenAlex ID](https://openalex.org/A5047293370)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Geodesic Hypothesis并基于此设计Semantic Tube Prediction（STP），以正则化LLM隐藏状态轨迹提升数据效率。

**💡 创新点**

创新点在于将token序列视为流形上的局部线性测地线，利用SNR提升和几何正则化打破传统规模定律。

**🔧 技术方法**

结合ODE/SDE理论、Least Action原理和1‑cos角度损失，构建STP正则化；对比JEPA与传统NTP。

**📊 数据集**

在NL‑RX‑SYNTH、NL‑RX‑TURK、GSM8K、Spider、NQ‑Open、HellaSwag等多任务数据集上进行实验。

**📈 对比分析**

与标准Fine‑tune和LLM‑JEPA相比，STP在多模型、多尺寸、多数据集上均保持或提升准确率，并能在仅使用1/16数据时达到全量精度，λ最优区间为0.01–0.08。

**⚠️ 局限性**

局限性包括需要手动调参λ、对流形光滑与局部线性假设的理论依赖、未在更大规模或跨模态任务上充分验证。

---

## 294. Where Vision Becomes Text: Locating the OCR Routing Bottleneck in Vision-Language Models

**arXiv ID:** 2602.22918 | [PDF](https://arxiv.org/pdf/2602.22918v1)

**作者:** Jonathan Steinberg `[一作]` (University of Haifa), Oren Gal `[通讯]` (University of Haifa)

**通讯引用:** 339 | [OpenAlex ID](https://openalex.org/A5030616543)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对不同VLM架构进行PCA驱动的因果干预，定位OCR信息在视觉‑语言流中的路由瓶颈。

**💡 创新点**

发现OCR信号高度低维且可跨数据集转移，揭示不同架构在深度层次上的OCR路由差异，并表明在模块化模型中OCR干预可提升计数任务。

**🔧 技术方法**

使用PCA子空间投影、因果干预、注意力头选择性比率、对比实验等技术。

**📊 数据集**

使用EgoTextVQA、OCRBench、InfoVQA三个OCR基准，以及CountBench、EmbSpatial、RealWorldQA等视觉推理和问答基准。

**📈 对比分析**

在五个模型中对比各层干预对OCR准确率、计数、空间推理和通用VQA的影响；DeepStack模型在中层干预可提升计数+6.9pp、VQA+1pp；单阶段模型干预往往损失推理；跨数据集PCA方向有效。

**⚠️ 局限性**

仅覆盖密集注意力架构，未验证MoE或专用OCR模型；干预依赖无纹理失真去文本的修复；评估范围局限于部分基准，可能对其他任务产生未知影响。

---

## 295. Reinforcing Real-world Service Agents: Balancing Utility and Cost in Task-oriented Dialogue

**arXiv ID:** 2602.22697 | [PDF](https://arxiv.org/pdf/2602.22697v1)

**作者:** Ning Gao `[一作]`, Chaozheng Wang `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了多粒度强化学习框架，将任务导向对话转化为多粒度CMDP，并通过用户中心交互环境与成本感知多轮策略优化实现高效成本控制与服务质量提升。

**💡 创新点**

创新点：①用户中心交互框架，基于人格库的动态模拟器；②成本感知多轮策略优化（CMPO）结合会话结果、生成过程信用与PID‑拉格朗日成本控制的混合优势估计；③全流程离线+在线训练，突破静态模仿瓶颈。

**🔧 技术方法**

技术：语言模型（Qwen2.5、Qwen3、GPT‑4等）+强化学习（GRPO、PPO）+PID‑Lagrange成本控制+生成式奖励模型（GenRM）+多层次优势估计。

**📊 数据集**

数据集：自研的Food Delivery Service（FDS）真实业务对话集及用户画像库，公开的τ²‑Bench（Retail、Airline、Telecom）等。

**📈 对比分析**

比较方法：与多种闭源大模型（GPT‑4.1、DeepSeek‑v3、LongCat‑Flash）、开源模型、SFT、PPO/GRPO等基线在FDS下三维指标（满意度、完成率、沟通/逻辑质量、优惠券率）对比；在FDS hard场景中，14B模型满意度3.05、完成率100%、优惠券率≈30%；在τ²‑Bench上Pass@1提升约5–7%。

**⚠️ 局限性**

局限性：①依赖大模型与昂贵算力；②成本约束仅考虑优惠券，未覆盖更细粒度运营成本；③在极端不合作用户场景下仍可能出现策略漂移；④缺乏对多模态交互与真实业务部署的验证。

---

## 296. The Inference Bottleneck: Antitrust and Neutrality Duties in the Age of Cognitive Infrastructure

**arXiv ID:** 2602.22750 | [PDF](https://arxiv.org/pdf/2602.22750v1)

**作者:** Gaston Besanson `[一作]` (Universidad Torcuato Di Tella), Marcelo Celani `[通讯]` (Universidad Torcuato Di Tella)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5072862909)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

提出了“认知基础设施”概念并阐述在生成式AI时代推理瓶颈对竞争的影响，归纳了非价格歧视与路由歧视机制，构建了可审计的“中性推理”框架

**💡 创新点**

创新点在于将传统垂直整合的市场竞争理论与生成式AI推理服务相结合，提出可测量的认知基础设施定义，并设计了针对非价格歧视的可审计技术义务

**🔧 技术方法**

主要使用理论分析、市场定义方法、可审计指标（QoS同质化测试、路由日志透明度、FRAND式非歧视条款）以及审计协议设计

**📊 数据集**

未使用实验数据集，依赖公开的行业报告、OECD 2025 研究和现有的AI服务定价与使用文档作为案例来源

**📈 对比分析**

论文并未进行实验比较或性能评估，而是通过理论模型和可审计指标提出了评估框架；若在实践中实施，期望能通过持续的 QoS 和路由日志测量检测并抑制歧视行为

**⚠️ 局限性**

局限性包括：缺乏实证验证，依赖可观察指标但难以完全捕捉所有技术细节；审计成本与可操作性未知；以及在不同司法管辖区的适用性和执行力度可能受限

---

## 297. Compress the Easy, Explore the Hard: Difficulty-Aware Entropy Regularization for Efficient LLM Reasoning

**arXiv ID:** 2602.22642 | [PDF](https://arxiv.org/pdf/2602.22642v1)

**作者:** Qin-Wen Luo `[一作]` (Nanjing University of Aeronautics and Astronautics), Sheng-Jun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 4251 | [OpenAlex ID](https://openalex.org/A5103204774)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型进行链式推理压缩，提出了基于实例难度的选择性熵正则化和动态最优长度惩罚两种技术，以在保持推理准确率的前提下显著缩短推理长度。

**💡 创新点**

创新点在于：①使用历史准确率动态评估问题难度，并仅对困难样本施加熵正则化，防止熵坍塌导致探索空间收窄；②引入以每个问题历史最短正确推理长度为基准的动态长度惩罚，保证长度压缩信号随训练进展稳定；③将两种策略结合，兼顾推理效率与推理质量。

**🔧 技术方法**

采用基于可验证奖励的强化学习（GRPO）框架，配合熵正则化（最大熵损失或熵优势）以及动态长度惩罚；使用异步指数移动平均更新问题难度；在训练中使用温度采样、cosine退火等技巧。

**📊 数据集**

训练集：从 DeepMath103K 与 DAPO 合并抽取 2500 条样本；基线模型为 R1‑distill‑Qwen2.5‑7B；评测集包括 GSM8K、Math‑500、AIME24/25、AMC23 与 OlympiadBench。

**📈 对比分析**

与 Prompting、Offline（Spirit、ConCISE‑SimPO、DAST）、Online RL（AutoThink、LC‑R1、Length‑Penalty）等多种基线对比；实验显示压缩率超过 30% 的同时，准确率基本保持或略升，Pass@k 与 NAG 指标优于仅做长度压缩的基线。

**⚠️ 局限性**

局限性：①熵正则化与动态长度惩罚对超参数（如熵系数、长度惩罚系数）敏感；②对难度评估依赖历史准确率，可能在样本极少或分布漂移时失效；③目前仅适用于可验证答案的推理任务，对开放式生成任务的推广仍需研究。

---

## 298. mmWave Radar Aware Dual-Conditioned GAN for Speech Reconstruction of Signals With Low SNR

**arXiv ID:** 2602.22431 | [PDF](https://arxiv.org/pdf/2602.22431v1)

**作者:** Jash Karani `[一作]` (BITS Pilani), Sandeep Joshi `[通讯]` (BITS Pilani)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

使用毫米波雷达捕获的低信噪比、带限语音信号，提出两阶段RAD‑GAN重建管线实现可理解的全带宽语音；

**💡 创新点**

创新点包括雷达感知双条件GAN（RAD‑GAN）架构、带有谱正则化和权重归一化的多梅尔鉴别器（MMD）、残差融合门（RFG）进行多通道条件融合，以及两阶段预训练+微调策略；

**🔧 技术方法**

主要技术包括HiFi‑GAN生成器、MPD/MSD鉴别器、MMD、WaveVoiceNet辅助模块、MR‑STFT与L1梅尔损失、GAN与特征匹配损失的联合优化；

**📊 数据集**

使用RASE 2026 Challenge提供的毫米波雷达+麦克风配对数据，总计约42小时，分为直接振动（Task‑1）和铝箔反射（Task‑2）两类；

**📈 对比分析**

通过PESQ、ESTOI、DNSMOS、MFCC相似度等指标，并计算加权总分，对比WVn、HiFi‑GAN等基线，RAD‑GAN在加权分0.333上优于WVn 0.260和HiFi‑GAN 0.288；

**⚠️ 局限性**

局限在于数据集规模有限、未做数据增强、缺乏实时延迟评估和模型压缩方案，且对极低信噪比场景仍有提升空间。

---

## 299. Efficient Parallel Algorithms for Hypergraph Matching

**arXiv ID:** 2602.22976 | [PDF](https://arxiv.org/pdf/2602.22976v1)

**作者:** Henrik Reinstädtler `[一作]` (Heidelberg University), Fabian Walliser `[通讯]` (Heidelberg University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在 GPU 上并行计算超图最大匹配（以及图的最大匹配）的算法，利用局部最优边并行加入匹配，最终得到一个最大匹配。

**💡 创新点**

创新点包括：在 CRCW PRAM 模型下实现 O(log m) 的运行时间（m 为超边数），提供 CREW、work‑optimal、MapReduce 与外部存储版本；给出 1/d 的近似保证；将算法实现为 CUDA 与 Kokkos，兼容多平台；在实验中相较单核 CPU 速度提升可达 76.6 倍，匹配质量在 88–99% 之间。

**🔧 技术方法**

核心技术：随机化噪声生成、局部最优边检测、并行计数与标记、CRCW/CREW 写入冲突处理、前缀和、GPU 并行子阶段、Kokkos 通用并行框架、外部存储与 MapReduce 的 PRAM‑to‑MapReduce 模拟。

**📊 数据集**

使用的数据集包括：90 个来自 SAT、SuiteSparse、DAC 挑战的超图实例；42 个图实例（DIMACS、Florida Sparse Matrix Collection、Delaunay、随机几何图、街道网络、社交网络等）。

**📈 对比分析**

比较方法：与单核 CPU 版、流式 1/d 近似算法、Birn 等人基于 MPI 的图匹配实现、Mandulak 等人的多 GPU 方案进行对比。实验测量运行时间、速度提升与匹配质量。结果显示 GPU 实现比单核 CPU 高 13–76 倍；Kokkos CPU 版在部分图上比 Birn 方案快 1.24 倍；在图上，CUDA/Kokkos GPU 方案分别比 Mandulak 方案快 15–27 倍；匹配质量在 87.6–98.2% 之间，略低于部分基线，但可通过调整噪声区间提升。

**⚠️ 局限性**

局限性：仅提供最大匹配的 1/d 近似；对超边权重分布敏感（如 Wiki 实例噪声导致 16 轮而非 5–7 轮）；实现依赖 GPU 内存和并行度，未在多 GPU 环境下验证；未探讨更精确的局部搜索或 b‑匹配等扩展；随机化步骤需要高质量 RNG，若 RNG 差异会影响结果。

---

## 300. Metamorphic Testing of Vision-Language Action-Enabled Robots

**arXiv ID:** 2602.22579 | [PDF](https://arxiv.org/pdf/2602.22579v1)

**作者:** Pablo Valle `[一作]` (Mondragon University), Aitor Arrieta `[通讯]` (Mondragon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了使用Metamorphic Testing (MT) 对Vision‑Language‑Action (VLA) 机器人进行行为评估，提出两种通用的轨迹一致性与轨迹变化 MRPs，并基于它们定义了五个具体的 metamorphic relations (MRs)，随后在五个 VLA 模型、两台机器人和四个任务上进行大规模仿真实验验证其有效性与补充性。

**💡 创新点**

创新点在于：①提出了可跨任务、跨模型的轨迹一致性和轨迹变化两种 MRP；②通过这两种 MRP 导出五个任务无关的 MRs，首次将 MT 应用于 VLA 机器人，解决了传统符号 oracle 的不易定义与缺失执行质量评估的问题；③通过实证表明 MT 能检测到符号 oracle 失效的执行层失败，并能与符号 oracle 互补。

**🔧 技术方法**

技术手段包括：Metamorphic Testing 与 Fréchet 距离来量化轨迹差异；VLA 模型生成的动作序列与机器人仿真环境 SimplerEnv；GPU 并行计算加速多模型多任务测试；MR 的自动生成与阈值设定。

**📊 数据集**

使用了四个标准机器人任务（Pick up、Move Near、Put On、Put In）在 Google Robot 与 WidowX 两个平台上；VLA 模型覆盖 OpenVLA、SpatialVLA、GR00T‑N1.5、EO‑1 等；训练与评估数据集包括 Fractal 与 Bridge 两个公开数据集。

**📈 对比分析**

通过与基于任务完成的符号 oracle 比较，MT 在高阈值下检测到约 1.5‑2 倍于符号 oracle 的失败，三种阈值下两者互补，整体失败检测率提升约 30%‑50%；不同 MRs 的误差率和覆盖率也被量化。

**⚠️ 局限性**

局限性包括：①阈值设定需人工调优，缺乏自动化；②实验仅在仿真环境中完成，实际机器人验证不足；③只设计了五个 MR，覆盖率仍有限；④对不同硬件与任务的泛化需要进一步验证。

---

## 301. Operationalizing Fairness: Post-Hoc Threshold Optimization Under Hard Resource Limits

**arXiv ID:** 2602.22560 | [PDF](https://arxiv.org/pdf/2602.22560v1)

**作者:** Moirangthem Tiken Singh `[一作]` (Dibrugarh University), Sapam Jitu Singh `[通讯]` (Manipur University)

**通讯引用:** 304 | [OpenAlex ID](https://openalex.org/A5017139742)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种后处理、模型无关的阈值优化框架，利用单一全局阈值在硬性资源约束下平衡安全性、效率与公平性。

**💡 创新点**

创新点在于：①通过阈值上限公式严格限制干预量，保证法律合规；②定义可调的伦理损失函数，将安全、效率、公平统一纳入优化；③证明阈值的单调性与临界容量，提供可解释的理论保证；④在多种高风险数据集上验证容量约束主导模型选择。

**🔧 技术方法**

技术方法包括：阈值网格搜索、概率分位数阈值计算、加权伦理损失函数最小化、Bootstrap 估计、方差分析等；框架完全独立于预测模型，属于后处理范式。

**📊 数据集**

使用三类高风险数据集：ACS Income（机会分配）、COMPAS Recidivism（惩罚风险）、UCI Diabetes（临床风险），并在Logistic Regression、Gradient Boosting、Random Forest 三种模型上进行实验。

**📈 对比分析**

与传统无约束优化、Demographic Parity、Equalized Odds、随机分配等基线比较；在固定25%容量下，所提框架实现召回率 0.409–0.702，显著高于基线；同时严格满足资源约束，说明方法在实际部署中更具可行性。

**⚠️ 局限性**

局限性包括：①仅考虑单一全局阈值，无法利用群组差异化阈值；②仅评估一种公平度量 ΔTPR，未覆盖其他公平指标；③实验基于公开数据集，缺乏真实系统验证；④容量约束固定，未考虑动态资源变动。

---

## 302. Hypernetwork-based approach for grid-independent functional data clustering

**arXiv ID:** 2602.22823 | [PDF](https://arxiv.org/pdf/2602.22823v1)

**作者:** Anirudh Thatipelli `[一作]` (University of Central Florida), Ali Siahkoohi `[通讯]` (University of Central Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于超网络的框架，将以任意分辨率、任意网格采样的函数观测映射到隐式神经表示（INR）的权重空间，从而实现对函数数据的聚类，克服传统方法对网格依赖的局限；

**💡 创新点**

创新点在于：①使用网格无关的编码器将坐标-值对聚合为固定维度向量；②通过超网络一次性预测INR权重，得到与采样无关的低维表示；③将聚类任务完全与表示学习分离，使得任意聚类算法可直接在权重空间中使用；

**🔧 技术方法**

核心技术包括：基于SIREN的隐式神经网络解码器、超网络（hypernetwork）进行权重预测、均值池化的点对点特征聚合、跨分辨率的多尺度训练以及标准聚类算法（如K‑means、GMM）在权重空间中的应用；

**📊 数据集**

实验使用了三类数据集：MNIST（二维灰度图像）、Kvasir（二维彩色医学图像）以及ERing（一维多通道时间序列），并在不同分辨率与未见分辨率上进行评估；

**📈 对比分析**

与传统基于像素或基函数系数的聚类相比，该方法在多分辨率、不同聚类算法下均保持高且稳定的AMI/ARI分数；在MNIST上AMI≈0.72、ARI≈0.61；在Kvasir上AMI≈0.48、ARI≈0.31；在ERing上AMI≈0.56、ARI≈0.41；对比基准FAEclust，虽然在ERing上略低，但方法在更广泛任务上保持竞争力；

**⚠️ 局限性**

限制包括：①对大输出维度的函数（高通道数）时INR的最后一层权重维度会急剧增大，影响聚类；②当前仅使用重建损失，未显式优化聚类分离度；③对极端下采样（如MNIST 7×7）信息本身不足，导致聚类性能下降；

---

## 303. SemanticVocoder: Bridging Audio Generation and Audio Understanding via Semantic Latents

**arXiv ID:** 2602.23333 | [PDF](https://arxiv.org/pdf/2602.23333v1)

**作者:** Zeyu Xie `[一作]` (Peking University), Yuexian Zou `[通讯]` (Peking University)

**通讯引用:** 5683 | [OpenAlex ID](https://openalex.org/A5002795838)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出SemanticVocoder，一种直接从语义潜在空间合成波形的生成式声码器，取代传统VAE音频潜在。

**💡 创新点**

创新点在于：①使用语义潜在而非低层音频潜在；②采用流匹配（flow‑matching）生成器避免重构失真；③实现两阶段模型解耦、可插拔。

**🔧 技术方法**

技术：MAE预训练语义编码器、ConvNeXt、流匹配网络、iSTFT、Diffusion Transformer（DiT）、AdaLN、CrossAttention、Euler ODE求解。

**📊 数据集**

数据集：AudioSet（训练Encoder+Vocoder），AudioCaps（文本到音频评估），WavCaps（预训练文本编码器），HEAR benchmark（音频理解评测）。

**📈 对比分析**

比较方法：在AudioCaps测试集上对比FAD、FD、KL、IS、CLAP等指标；在HEAR上用MLP评测多类/多标识分类和事件检测。性能：SemanticVocoder在AudioCaps上FAD 1.709、FD 12.823，优于所有基线；在HEAR上语义潜在得到更高准确率。

**⚠️ 局限性**

局限性：依赖预训练语义编码器的能力；受限于音频长度（无法生成长时音频）；指标无法完全反映主观质量，仍需人类评测。

---

## 304. CCCL: Node-Spanning GPU Collectives with CXL Memory Pooling

**arXiv ID:** 2602.22457 | [PDF](https://arxiv.org/pdf/2602.22457v1)

**作者:** Dong Xu `[一作]` (University of California), Dong Li `[通讯]` (University of California)

**通讯引用:** 24204 | [OpenAlex ID](https://openalex.org/A5100369293)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于CXL共享内存池的跨节点GPU集体通信库，用于替代传统的RDMA网络通信，实现高效的数据共享与同步；

**💡 创新点**

创新点在于：①利用CXL内存池构建跨节点GPU通信的全新方法；②提出软件级数据互插和门铃同步机制，解决了内存争用、RAW依赖和跨节点同步问题；③通过细粒度分块与异步重叠进一步提升并行度；

**🔧 技术方法**

使用的技术包括CXL 2.0/3.0共享内存池、Linux DAX映射、CUDA GPU DMA、门铃（doorbell）同步、软件层面数据互插、NCCL集成与改造；

**📊 数据集**

在LLM训练实验中使用了Llama‑3‑8B模型，并在Wikipedia语料上进行FSDP分布式训练；

**📈 对比分析**

通过与200Gbps InfiniBand RDMA实现的NCCL baseline以及Naive、Bandwidth‑ALL等版本进行对比；在AllGather、Broadcast、Gather、Scatter、Reduce、AllReduce等多种collective上分别实现了1.34×、1.84×、1.94×、1.04×等加速，比InfiniBand平均提升约1.1×；LLM训练时相较于InfiniBand实现获得1.11×的速度提升，同时硬件成本降低约2.75×；

**⚠️ 局限性**

局限性包括：①仅在有限节点（3–12节点）上验证，且共享内存池容量受CXL设备数限制；②CXL访问延迟高于本地DRAM，导致小消息性能受限；③缺乏跨节点完整一致性保证，需依赖软件门铃；④目前CXL硬件及驱动生态不成熟，实际部署受限。

---

## 305. AlayaLaser: Efficient Index Layout and Search Strategy for Large-scale High-dimensional Vector Similarity Search

**arXiv ID:** 2602.23342 | [PDF](https://arxiv.org/pdf/2602.23342v1)

**作者:** Weijian Chen `[一作]` (Southern University of Science and Technology), Bo Tang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 63931 | [OpenAlex ID](https://openalex.org/A5100679233)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于磁盘的图结构近似最近邻搜索系统 AlayaLaser，专为大规模高维向量检索设计。

**💡 创新点**

创新点包括：① SIMD友好的索引布局，利用 PCA 提取主成分并将量化码词按列交错存储，实现向量距离计算并行化；② 基于节点入度的缓存策略，充分利用内存减小 I/O；③ 集群化入口点选择、适应性光束扩展以及早期分发机制，进一步降低搜索延迟。

**🔧 技术方法**

核心技术包括 PCA 降维、产品量化（PQ）或 RaBitQ 量化、SIMD 指令加速、异步 I/O 调度、图结构构建与搜索策略优化。

**📊 数据集**

实验使用五个公开高维数据集：GIST1M（960D）、Cohere、BigCode、DPR、MSMARCO，维度分别在 768-1024 之间。

**📈 对比分析**

与 DiskANN、Starling、PipeANN、SPANN 等磁盘级索引以及 HNSWlib、Vamana、SymphonyQG 等内存级索引对比，AlayaLaser 在吞吐量上提升 2-60 倍、延迟下降 20-90%，在多数场景甚至匹配或优于主流内存方案。

**⚠️ 局限性**

局限性：① 对离散分布（out‑of‑distribution）查询的支持尚未完善；② 需要较高的 SIMD 支持，受 CPU 架构限制；③ 量化码词存储导致索引体积略增；④ 在极高压缩比或极大规模数据时，缓存与 I/O 的平衡仍需进一步调优。

---

## 306. Robust Helicopter Ship Deck Landing With Guaranteed Timing Using Shrinking-Horizon Model Predictive Control

**arXiv ID:** 2602.22714 | [PDF](https://arxiv.org/pdf/2602.22714v1)

**作者:** Philipp Schitz `[一作]` (German Aerospace Center), Johann C. Dauer `[通讯]` (German Aerospace Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于收缩时域模型预测控制（SHMPC）与移动阻塞的无人直升机船舱着陆方案，能够在预定时间窗口内完成稳健着陆。

**💡 创新点**

创新点包括：①将SHMPC与移动阻塞结合，实现大预测时域的实时可解；②设计了基于RPI集的增广扰动观测器，提升了对风扰动的鲁棒性；③构造可达终端集，实现着陆时间窗口保证；④利用常数扰动分量提升在强风条件下的性能。

**🔧 技术方法**

技术手段主要有：收缩时域模型预测控制、移动阻塞、管束控制（tube MPC）、扰动观测器、RPI集计算、后向可达性分析、二次优化求解（qpOASES）。

**📊 数据集**

使用DLR小型实验平台midiARTIS的非线性模型，并基于三组实验数据（悬停、10 m/s、20 m/s）识别得到线性模型进行仿真。

**📈 对比分析**

通过与预设的三种着陆机动（直入、对角、侧向）仿真对比，结果表明：最大求解时间为4 ms，满足20 ms采样；在强风（8 m/s、30°）下，所有机动均在规定时间窗口内安全着陆，姿态误差≤1.1°，位置误差≤0.6 m，速度误差≤1.15 m/s。

**⚠️ 局限性**

局限性包括：①仍未在真实飞行环境中验证；②对船舶运动估计误差未考虑；③RPI集与终端集的保守性可能限制更激进的着陆策略；④需要预先估计常数扰动项，若估计不准会影响性能。

---

## 307. EmpiRE-Compass: A Neuro-Symbolic Dashboard for Sustainable and Dynamic Knowledge Exploration, Synthesis, and Reuse

**arXiv ID:** 2602.22276 | [PDF](https://arxiv.org/pdf/2602.22276v1)

**作者:** Oliver Karras `[一作]` (Leibniz Information Centre for Science and Technology), Yücel Celik `[通讯]` (Leibniz University Hannover)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了 EmpiRE‑Compass neuro‑symbolic 仪表板，将研究知识图谱（RKG）与大语言模型（LLM）结合，支持基于已策划与自定义能力问题的文献综述（LR）数据的探索、综合和重用。

**💡 创新点**

创新点在于将符号化的语义结构化数据（RKG）与生成式 LLM 进行深度融合，实现可动态生成可视化、解释和推理的交互式仪表板，从而降低技术壁垒，提升 LR 的可持续性和可重复性。

**🔧 技术方法**

技术包括：ORKG（Open Research Knowledge Graph）作为语义层；SPARQL 查询执行；React + Vite + TypeScript + Material‑UI + Storybook 前端；后端 Node/Express + Firebase Firestore；LLM 接入（GPT‑4o mini、Groq、Mistral、Gemini 等）；Swagger API 文档；Python 脚本用于统计与自动化。

**📊 数据集**

使用了两组数据集：EmpiRE RKG（涵盖 776 篇关于 RE 实证研究的论文）以及 NLP4RE ID Card RKG（50 篇 NLP4RE ID 卡），并将这些数据发布在 ORKG 与 GitHub 仓库中。

**📈 对比分析**

评估计划分为短期（专家评审已有能力问题的可视化与解释）、中期（试点研究用户任务完成时间与体验）和长期（大规模自定义问题的自动化指标——完整度、准确度、精确率与召回率）。目前尚未给出定量结果，但设计与现有仅基于预定义仪表板的工具相比，期望在交互性与可重用性上实现显著提升。

**⚠️ 局限性**

局限性包括：LLM 生成的解释和可视化仍需人工验证；仪表板目前仅覆盖两类 RKG，泛化能力待扩展；对 LLM 的调用成本与请求速率有限制；需要研究人员具备一定的语义技术或人工介入来完善语义结构化。

---

## 308. DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation

**arXiv ID:** 2602.22896 | [PDF](https://arxiv.org/pdf/2602.22896v1)

**作者:** Zebin Yang `[一作]` (Peking University), Meng Li `[通讯]` (Peking University)

**通讯引用:** 24646 | [OpenAlex ID](https://openalex.org/A5100457407)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过动态-静态层跳过实现 Vision‑Language‑Action 模型推理加速，降低实时推理延迟。

**💡 创新点**

创新点包括：1）动态‑静态层跳过机制，区分信息层与可跳层；2）先后跳过指导（prior‑post）结合轨迹连续性来决定跳过；3）两阶段跳过感知知识蒸馏训练轻量化控制器与适配器。

**🔧 技术方法**

采用预跳预测、后跳验证、轻量级适配器、跳过控制器以及两阶段知识蒸馏技术进行训练与推理。

**📊 数据集**

在 CALVIN 与 LIBERO 两大机器人操控基准数据集上进行评估。

**📈 对比分析**

与 HULC、SPIL、DeeR‑VLA、FlexiDepth 等方法对比，DySL‑VLA 在 CALVIN 上成功长度提升 2.1%，在 LIBERO 上成功率提升 1.2%，推理延迟比全模型低 3.75×，参数量缩减 85.7×，训练步数减少 13.7×。

**⚠️ 局限性**

局限性：跳过决策高度依赖轨迹连续性，可能在极端快速或极低功耗场景下不够鲁棒；对极其重要动作的误判仍可能导致任务失败。

---

## 309. Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring

**arXiv ID:** 2602.22731 | [PDF](https://arxiv.org/pdf/2602.22731v1)

**作者:** Miguel Ángel Muñoz-Bañón `[一作]` (University of Alicante), Maurice Fallon `[通讯]` (University of Oxford)

**通讯引用:** 6250 | [OpenAlex ID](https://openalex.org/A5072974727)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种融合 NeRF、LiDAR SLAM 与 GNSS 的三层级管线，能够在森林现场实现可重复、地理定位的幼树三维重建，并提取高度、分枝数、叶木比等生态指标；

**💡 创新点**

创新点在于（1）构建粗到细的三层定位重建体系；（2）通过松耦合 NeRF 与 LiDAR SLAM 实现尺度一致且地理定位的对象级重建；（3）开发针对森林环境的细枝叶分割与骨架提取方法；

**🔧 技术方法**

使用 LiDAR SLAM (VILENS-SLAM)、GNSS 对齐、COLMAP SfM 细化姿态、Neural Radiance Fields 训练、PC‑Skeletor 骨架化、叶木分割与 3D Gaussian Splatting 对比实验；

**📊 数据集**

数据来源为 Wytham Woods (英国) 的 23 颗幼树在 2025 年夏冬三次现场采集，以及芬兰 Evo 森林的 3 颗针叶幼树；对比 Leica RTC360 TLS 与移动激光扫描（MLS）结果；

**📈 对比分析**

通过 PSNR/SSIM/LPIPS 评估新视角渲染，NeRF 在 0.5–1 m 高度树木上明显优于 TLS；点云精度对比 TLS，NeRF 细枝叶更完整；提取的高度、分枝数、叶木比与 TLS 对比显示 NeRF 更准确且更稳定，能够捕捉生长与损伤的时序变化；

**⚠️ 局限性**

局限性包括：对高度较大、冠层高的树木捕捉不全；手持设备受风光与遮挡影响；NeRF 训练耗时且对光照/背景变化敏感；缺乏绝对高度真值，依赖多传感器硬件，难以推广到资源受限场景。

---

## 310. Data-Driven Supervision of a Thermal-Hydraulic Process Towards a Physics-Based Digital Twin

**arXiv ID:** 2602.22267 | [PDF](https://arxiv.org/pdf/2602.22267v1)

**作者:** Osimone Imhogiemhe `[一作]`, Saïd Moussaoui `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套基于物理模型的数字孪生，用于闭合水循环热液系统的故障检测与诊断，利用1D流体网络模拟结合机器学习实现参数变化的检测、定位与估计。

**💡 创新点**

创新点在于将传统的物理模拟与数据驱动方法融合，形成四步完整的FDD流程：阈值触发检测、决策树定位、SVR估计以及阈值验证；同时首次在单扰动场景下实现高精度定位（95.14%）与误差低于1%的参数估计。

**🔧 技术方法**

主要技术包括Simcenter Flomaster 1D流体网络模型、决策树分类器、支持向量回归（SVR）、阈值检测与迭代验证、Python及scikit‑learn等机器学习库。

**📊 数据集**

使用的数据库完全由1D Flomaster模拟生成，覆盖控制向量的多重采样与单参数扰动，规模约30万条样本（305,760点）。

**📈 对比分析**

通过分类准确率、混淆矩阵以及SVR估计误差来评估性能。定位准确率达95.14%，估计误差在1%以内，验证阈值内可恢复正常；未与其他算法做直接对比，仅在单扰动情形下验证。

**⚠️ 局限性**

限制主要有：仅验证单一扰动，未考虑多扰动或连续故障；阈值手动调节缺乏自适应机制；缺乏真实物理平台实验验证；对噪声鲁棒性评估不足。

---

## 311. Probing for Knowledge Attribution in Large Language Models

**arXiv ID:** 2602.22787 | [PDF](https://arxiv.org/pdf/2602.22787v1)

**作者:** Ivo Brink `[一作]` (KPMG), Dennis Ulmer `[通讯]` (University of Amsterdam)

**通讯引用:** 214 | [OpenAlex ID](https://openalex.org/A5001376704)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大语言模型（LLM）在生成答案时使用的是上下文知识还是参数知识，并提出了一套自监督的数据生成管道 AttrWiki，用于生成单一知识来源的示例，然后训练轻量级线性探针来判定答案的贡献来源。

**💡 创新点**

创新点在于：①使用自监督流程仅依赖 Wiki 文本，自动生成标注的上下文/参数知识样本；②证明 LLM 隐藏层中可以线性解码知识来源；③展示归因错误与答案错误之间的显著关联，为未来可信交互提供新的诊断维度。

**🔧 技术方法**

技术手段包括：spaCy+RoBERTa 进行实体识别与同义匹配；GPT‑4o‑mini 用于生成不同知识来源的 prompt；在 Llama‑3.1‑8B、Mistral‑7B 与 Qwen‑2.5‑7B 上提取隐藏状态；训练 Final‑LR、Layer‑LR 与 Layer‑MLP 探针；使用 PCA 研究层级分离；在外部 QA 数据集（SQuAD、WebQuestions）上进行无细调的跨域评估。

**📊 数据集**

主要数据集：①从 Wikipedia 采样 20k 片段构成 AttriWiki，按模型的记忆情况标注为 parametric 或 contextual；②外部评估集包括 SQuAD（文本上下文）与 WebQuestions（无上下文）以及其三种控制变体。

**📈 对比分析**

实验显示：Layer‑LR 及 Layer‑MLP 在 AttriWiki‑test 上宏 F1 最高可达 0.96（以最后一个实体 token 为特征），比单层 Logistic Regression 提升 7–11pp；在 SQuAD 与 WebQuestions 上无细调即可取得 0.94–0.99 的准确率，证明归因信号高度可迁移；归因错误与答案错误呈显著正相关，误用知识源时错误率可提升 30–70%。

**⚠️ 局限性**

局限性：①管道仅在 Wiki 领域内测试，可能难以推广到对话、法律或多语种文本；②依赖实体抽取与同义匹配，存在词汇偏差；③当前探针仅在 token 级别进行归因，对句子级或更自由的生成场景效果不明；④每个模型需单独训练探针，缺乏跨模型通用性；⑤仅关注知识来源，未能解释模型为何错误，即错误与归因不完全对应。

---

## 312. Generative Data Transformation: From Mixed to Unified Data

**arXiv ID:** 2602.22743 | [PDF](https://arxiv.org/pdf/2602.22743v1)

**作者:** Jiaqing Zhang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 28216 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Taesar框架，通过对跨域序列进行数据层面的重生，以实现目标域序列的语义对齐；

**💡 创新点**

创新点在于将跨域知识迁移从模型结构转移到数据生成层，利用对比解码自动识别并转换可迁移的源域条目；

**🔧 技术方法**

主要技术包括基线混合域预训练、域特定预测（DSP）适配、以及基于全局与局部对比分数的自适应对比解码；

**📊 数据集**

实验使用了Amazon Review四个公开域（Books、Electronics、Sports、Tools）数据集，保留最近128条交互；

**📈 对比分析**

与传统模型中心方法相比，Taesar在多种单域序列推荐器上均显著提升HitRate、NDCG和MRR（约提升5–10%），并证明了跨架构的通用性；

**⚠️ 局限性**

主要局限在于对信息保留程度缺乏理论上限的深入分析，并且对极度不匹配的域间语义可能仍存在迁移瓶颈。

---

## 313. Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving

**arXiv ID:** 2602.22801 | [PDF](https://arxiv.org/pdf/2602.22801v1)

**作者:** Yinan Zheng `[一作]` (Institute for AI Industry Research), Jingjing Liu `[通讯]` (Institute for AI Industry Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并部署了一款端到端自驾规划器Hyper Diffusion Planner，利用扩散模型生成驾驶轨迹并在真实车辆上进行200 km城市路测；

**💡 创新点**

在规划任务中重新设计扩散损失空间、采用速度与航点的混合轨迹表示及混合损失、实现大规模数据训练，并加入基于强化学习的后置训练，显著提升闭环性能；

**🔧 技术方法**

使用扩散模型+Transformer解码器、BEV感知编码器、混合速度/航点监督、RL加权回归、DPM‑Solver加速以及ONNX/TensorRT优化的推理管线；

**📊 数据集**

使用公司自研的70 M帧真实车辆多模态数据集，涵盖多种城市驾驶场景；

**📈 对比分析**

与基线扩散模型及NavSim等基准进行开放式与闭环评估，闭环成功率提升约10倍，整体评分从约57提升至约76；

**⚠️ 局限性**

RL奖励仅聚焦安全导致驾驶偏保守，缺乏效率和多模态奖励；在极大规模数据训练下，对某些简单场景（如车跟车停止）的性能略有下降。

---

## 314. Dynamic Level Sets

**arXiv ID:** 2602.22530 | [PDF](https://arxiv.org/pdf/2602.22530v1)

**作者:** Michael Stephen Fiske `[一作]` `[通讯]`, Michael Stephen Fiske

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文提出并证明了动态层集（dynamic level sets）这一新数学概念，揭示其在基于主动元件机器（AEM）的通用图灵机实现中能够产生不可算行为。

**💡 创新点**

创新点在于把层集的逻辑结构固定、但其物理实现随量子随机输入动态变化，从而绕过了1956年 de Leeuw 等关于概率图灵机不增计算能力的结论，并提供了可计算程序产生不可算输出的机制。

**🔧 技术方法**

技术核心包括：主动元件机器架构、基于量子随机数生成器的不可算随机序列、Meta 命令动态重构物理层集、以及可逆布尔函数映射实现逻辑与物理层集的对应。

**📊 数据集**

论文为理论性研究，不依赖任何外部数据集；所有结论均通过数学证明（Lemma 4.1、Theorem 4.2、Theorem 4.3）和理论模型验证。

**📈 对比分析**

与传统概率图灵机比较，本文证明其在相同程序下可产生不可算行为；相对自适应控制与非自治系统，动态层集在随机性与自我修改性上具有更高自由度，理论上实现完美保密性。

**⚠️ 局限性**

局限性包括：依赖量子随机性假设，尚未在实验平台上完整实现；对实际硬件噪声与误差容忍度的影响未做深入评估；以及理论模型对大规模可扩展性的分析仍需进一步研究。

---

## 315. Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design

**arXiv ID:** 2602.23092 | [PDF](https://arxiv.org/pdf/2602.23092v1)

**作者:** Zhuoliang Xie `[一作]` (Southern University of Science and Technology), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 39363 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

利用自适应迭代局部搜索与大语言模型自动启发式设计相结合，提出 AILS-AHD 方法解决容量约束车辆路径规划（CVRP）问题。

**💡 创新点**

创新点包括：① 将 LLM 与进化计算融合，用 LLM 自动生成破坏（ruin）启发式；② 引入 Chain‑of‑Thought 与早停机制加速评估；③ 在大规模 CVRPLib 基准上首次获得 8/10 个新最优解。

**🔧 技术方法**

使用技术包括 Adaptive Iterated Local Search、LLM（GPT‑4o）生成破坏启发式、进化计算框架、Chain‑of‑Thought、早停加速等。

**📊 数据集**

实验数据集为 CVRPLib 的中等规模 X Set（100 个 100–1000 顶点实例）和大规模 AGS Set（10 个 3000–30000 顶点实例）。

**📈 对比分析**

与 HGS、AILS‑II、手工设计的 AILS-C/S 进行对比；在中等规模实例平均差距 0.0672%，在大规模实例平均差距 0.0862%；并在 10 个大规模实例中共获得 8 个新最优解。

**⚠️ 局限性**

限制：当前的 AHD 框架仍相对简单，自动化效率有待提升；仅针对标准 CVRP，需进一步推广到其他车辆路径规划变体。

---

## 316. Fairness in Limited Resources Settings

**arXiv ID:** 2602.23026 | [PDF](https://arxiv.org/pdf/2602.23026v1)

**作者:** Eitan Bachmat `[一作]` (Ben-Gurion University of the Negev), Inbal Livni Navon `[通讯]` (Ben-Gurion University of the Negev)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

在有限资源分配场景中，本文研究风险预测与资源分配耦合下的群体公平性，提出将资源分配的公平性定义（最大最小公平性、比例公平性）适配到预测问题，并定义新的可实现等机会公平性。

**💡 创新点**

创新点在于：①将资源分配公平性框架与机器学习公平性理论连接；②证明在信息缺口的情况下，传统等机会与最大最小公平性会导致“少信息组”获得绝大部分资源，价格无界；③引入归一化比例公平性和可实现等机会，使公平性价格有界；④给出公平性价格的理论下界与上界。

**🔧 技术方法**

主要技术为：概率论与统计学（预测器的多校准、多分布模拟）；信息论（KL 散度用于比例公平性表达式）；凸优化与拉格朗日理论（构造阈值函数、证明最优性）；定理与证明（价格上界/下界）。

**📊 数据集**

文章以理论推导为主，实验部分使用模拟的风险预测分布（如两组不同尾部优势的分布）进行图示；未使用公开真实数据集。

**📈 对比分析**

与传统公平性约束（Demographic Parity、Equal Opportunity）相比，本文通过理论证明显示：在信息缺口情形下传统公平性会使真阳性率下降到接近 0；而比例公平性和可实现等机会在相同条件下保持至少一半的真阳性率。实验图表展示了公平性价格的上界与下界。

**⚠️ 局限性**

局限性包括：①结果多聚焦于两组情况，推广到多组需进一步研究；②假设预测器已校准且满足尾部优势，实际应用中可能难以满足；③仅讨论预测-阈值两步流程，未考虑学习过程中的公平性约束；④理论分析基于理想化分布，实际数据噪声与分布不匹配时效果未知。

---

## 317. Power Consumption Patterns Using Telemetry Data

**arXiv ID:** 2602.22339 | [PDF](https://arxiv.org/pdf/2602.22339v1)

**作者:** Harry Cheon `[一作]` (University of California San Diego), Ahmed Shams `[通讯]` (Intel Corporation)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析 Intel 设备遥测数据，比较美国与中国用户的功耗差异，并构建线性模型揭示用户行为对功耗的影响。

**💡 创新点**

证明用户行为比硬件选择更能决定功耗，发现美中功耗差异并通过特征工程与 LASSO 识别关键影响因素；提出“Median User Average Power”指标以及对 Turbo 状态和工作强度的细粒度比较。

**🔧 技术方法**

使用加权均值归一化、线性回归与 LASSO 正则化、特征工程、统计比较（R²、MSE）以及 EDA 可视化手段。

**📊 数据集**

基于 Intel Driver & Support Assistance 的遥测数据 1 M GUID 样本，包含 hw_pack_run_avg_power、os_c_state、sampler_data、sysinfo 等表的日聚合信息。

**📈 对比分析**

采用 MUAP 指标对各国进行功耗比较，控制 CPU TDP、硬件后仍保持差异；线性模型得到 MSE = 5.86 W²，R² = 0.33，说明约 33% 方差可被解释。

**⚠️ 局限性**

仅使用日聚合数据，缺乏细粒度时间信息；模型未包含所有可能影响功耗的变量，导致解释力有限。

---

## 318. LLM Novice Uplift on Dual-Use, In Silico Biology Tasks

**arXiv ID:** 2602.23329 | [PDF](https://arxiv.org/pdf/2602.23329v1)

**作者:** Chen Bo Calvin Zhang `[一作]` (Scale AI), Julian Michael `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在八个生物安全相关基准上，对照实验设计，测评了新手在拥有多种大语言模型（LLM）访问与仅使用互联网资源时的表现差异。

**💡 创新点**

创新点包括：① 真实环境下让参与者同时使用多款前沿LLM；② 长时段交互式评估并记录实时答案、置信度与笔记；③ 对人机协作行为进行细粒度质性编码与跨基准对比。

**🔧 技术方法**

使用了多款前沿LLM（如o3、Gemini 2.5 Pro、Claude Opus 4等），结合线性混合模型、逻辑回归、蒙特卡罗估计与置信度校准曲线进行定量分析；质性数据通过文本嵌入、正则表达式与人工标注处理。

**📊 数据集**

采用了八个公开/私有的生物安全基准（Virology Capabilities Test、Human Pathogen Capabilities Test、LAB‑Bench、Molecular Biology Capabilities Test、World Class Biology、Humanity’s Last Exam、Long‑Form Virology、Agentic Bio‑Capabilities Benchmark）以及相应的专家与单独LLM基准数据。

**📈 对比分析**

通过比较平均准确率、编辑距离、置信度与最佳50%分位数等指标，发现LLM辅助新手的平均准确率是对照组的4.16倍（95% CI 2.63–6.87），在多数基准上超过专家；单独LLM往往仍表现更好，说明人机协作尚未达到最优。

**⚠️ 局限性**

局限性包括样本量有限、实验未完全双盲、LLM版本在实验期间不一致、部分被试可能使用平台外的LLM、对实际实验室情境的可推广性不足。

---

## 319. A Model-Free Universal AI

**arXiv ID:** 2602.23242 | [PDF](https://arxiv.org/pdf/2602.23242v1)

**作者:** Yegon Kim `[一作]` (KAIST), Juho Lee `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了第一种模型无关的通用强化学习代理AIQI，并在理论上证明其强渐近最优和贝叶斯最优。

**💡 创新点**

创新点在于将通用归纳迁移至分布式行动价值函数，而非传统的环境或策略，且通过周期性增补解决延迟反馈问题，得到完整的理论证明。

**🔧 技术方法**

采用分布式蒙特卡洛控制、贝叶斯序列预测、混合预测器、可反射计算机（reflective oracle）以及全局有效性分析。

**📊 数据集**

未使用任何具体实验数据集，所有结果均为理论证明；若有实验则采用可计算近似实现的AIQI进行小规模仿真。

**📈 对比分析**

方法主要通过理论对比：证明AIQI与AIXI、AIXItl等模型无关代理在渐近极限下相当；实验时与传统模型无关MC方法对比，展示收敛速度与理论一致，但缺乏大规模实证评估。

**⚠️ 局限性**

局限性包括：需要满足“真理粒度”（grain of truth）假设；为纯上策略MC控制，无法自我优化；对离线/离策略学习不适用；实际实现依赖可计算近似，计算成本未完全评估。

---

## 320. Silent Egress: When Implicit Prompt Injection Makes LLM Agents Leak Without a Trace

**arXiv ID:** 2602.22450 | [PDF](https://arxiv.org/pdf/2602.22450v1)

**作者:** Qianlong Lan `[一作]` (eBay), Stephanie Westrum `[通讯]` (eBay)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了代理式LLM系统在自动URL预览过程中因隐式提示注入导致的“无声外泄”漏洞，展示攻击成功概率高达0.89且95%的攻击在输出上无可见风险

**💡 创新点**

首次系统量化隐式提示注入造成的网络外泄风险，提出分片外泄（sharded exfiltration）逃逸策略，并验证网络层防御（域名白名单、重定向检测）比提示层防御更有效

**🔧 技术方法**

使用基于Ollama的qwen2.5:7b模型实现ReAct型代理，构建本地可复现的攻击测试框架，利用HTTP请求捕获外泄，采用多种攻击载荷（标题、元描述、正文、锚文本）和重定向链进行实验

**📊 数据集**

在本地实验环境中生成合成恶意网页，控制标题、meta、body、anchor等多表面注入；敏感数据直接放置于模型上下文窗口，使用30次/配置的实验共480次

**📈 对比分析**

通过对比攻击成功率（P(egress)）、Leak@1、完整率等指标，实验表明单射攻击成功率≈1.0，分片攻击降低Leak@1 73%，提示层防御效果有限，而域名白名单、重定向检测能完全阻断外泄；性能上，网络层控制在允许的请求范围内几乎无影响

**⚠️ 局限性**

实验依赖单一小模型和HTTP外泄，未评估更大模型、隐式外泄渠道（DNS、时序侧信道）以及更复杂的对抗性载荷；控制实验中未加入更强的系统提示或上下文过滤，实际部署成功率可能不同

---

## 321. A 1/R Law for Kurtosis Contrast in Balanced Mixtures

**arXiv ID:** 2602.22334 | [PDF](https://arxiv.org/pdf/2602.22334v1)

**作者:** Yuda Bi `[一作]` (Georgia State University), Vince D Calhoun `[通讯]` (Georgia State University)

**通讯引用:** 110708 | [OpenAlex ID](https://openalex.org/A5032850756)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了在平衡混合下，随着混合宽度 R 的增大，基于峰度的 ICA 对比度如何衰减，并提出了 1/R 的衰减定律、样本规模与 R 的必要阈值，以及通过选择符号一致子集（purification）恢复对比度的方法。

**💡 创新点**

创新点在于给出了平衡混合下峰度对比度的全局人口层级衰减定律，提出了一个可计算的模型阶数上限（R ≲ √T）以及一种简单有效的“purification”技术可将对比度从 O(1/R) 提升到 Ω(1/m) 以恢复信息。

**🔧 技术方法**

采用了理论推导（矩与四阶累计量的性质）、概率分析（参与比、平衡性概率）、以及 FastICA 等峰度驱动 ICA 算法进行实验验证。

**📊 数据集**

实验使用了合成的 Student‑t 源混合（自由度 6–30）以及真实的 COBRE 组脑功能磁共振成像数据集（155 名受试者）。

**📈 对比分析**

通过 FastICA 误差与 1/Δκ 的关系、峰度与 R 的 1/R 线性关系、以及在 R=50 时 purification 的对比度提升（约 14 倍）等对比，实验结果均与理论预期一致，证明了定律的有效性和 purification 的实用性。

**⚠️ 局限性**

局限性包括仅针对峰度基准的线性瞬时 ICA，假设混合矩阵良好条件且平衡；对非峰度准则、非线性或卷积 ICA 的适用性未知；并且 purification 需要符号一致性假设，在源峰度符号混杂时效果可能下降。

---

## 322. ODEBrain: Continuous-Time EEG Graph for Modeling Dynamic Brain Networks

**arXiv ID:** 2602.23285 | [PDF](https://arxiv.org/pdf/2602.23285v1)

**作者:** Haohui Jia `[一作]` (Information Science and Technology Hokkaido University), Takashi Matsubara `[通讯]` (Information Science and Technology Hokkaido University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

基于神经ODE的连续时间模型，用双编码器初始化并对脑电图进行图谱化，实现了EEG时空频域特征的连续动态预测。

**💡 创新点**

首次将神经ODE与图结构结合，设计自适应向量场与梯度场评价，并通过双编码器提升初始状态鲁棒性。

**🔧 技术方法**

采用Spectral Graph Neural Networks、GRU、双编码器、神经ODE、RK4求解器以及多步图预测解码器。

**📊 数据集**

在TUSZ（癫痫发作检测）和TUAB两大公开EEG数据集上进行评估。

**📈 对比分析**

与DCRNN、BIOT、CNN‑LSTM等离散时间基线相比，在12s癫痫检测中AUROC提升至0.881（+0.065），F1提升至0.496（+0.07），整体性能显著优于所有对比方法。

**⚠️ 局限性**

仅基于分段EEG，长期连续建模受限；模型对其他神经疾病或认知任务的泛化仍待验证。

---

## 323. Certified Circuits: Stability Guarantees for Mechanistic Circuits

**arXiv ID:** 2602.22968 | [PDF](https://arxiv.org/pdf/2602.22968v1)

**作者:** Alaa Anani `[一作]` (Max Planck Institute for Informatics), Jonas Fischer `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 885 | [OpenAlex ID](https://openalex.org/A5007776675)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Certified Circuits 框架，用于在概念数据集的编辑下为电路发现提供可证明的稳定性保证；

**💡 创新点**

创新点在于将删除基随机光滑（RS‑Del）与按神经元投票的置信度阈值结合，得到对任意黑盒电路发现算法的无模型、无结构依赖的证书，并通过“放弃”不稳定神经元实现更稀疏、更准确的电路；

**🔧 技术方法**

核心技术包括随机删除光滑、基于投票的概率阈值裁决、蒙特卡洛估计下的置信区间构造，以及对电路的可解释性验证；

**📊 数据集**

主要在 ImageNet 上进行实验，并在四个 OOD 数据集（OOD‑CV、ImageNet‑A、ImageNet‑O、ImageNet‑C）上评估；

**📈 对比分析**

与传统基于 top‑K 的电路发现（采用激活、相关性、排名等评分）相比，Certified Circuits 在 OOD 场景下平均提高 91% 的电路准确率，同时电路大小缩小 45%，且在输入分布外仍保持更高的稳定性和准确性；

**⚠️ 局限性**

主要局限在于需要多次（如 1000 次）随机子样本评估导致运行时间显著增加，且证书半径受删除率限制，过高删除率会削弱基准算法效果；

---

## 324. Effective QA-driven Annotation of Predicate-Argument Relations Across Languages

**arXiv ID:** 2602.22865 | [PDF](https://arxiv.org/pdf/2602.22865v1)

**作者:** Jonathan Davidov `[一作]` (Bar-Ilan University), Ayal Klein `[通讯]` (Ariel University)

**通讯引用:** 131 | [OpenAlex ID](https://openalex.org/A5053897370)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将英语 QA‑SRL 解析器与受限机器翻译、词对齐等组件串联，自动生成希伯来语、俄语、法语等目标语言的问答式谓词‑论元标注数据，并基于这些数据训练语言特定的 QA‑SRL 解析器。

**💡 创新点**

创新点在于：①将自然语言问答格式作为跨语言迁移的中介，省去角色表映射；②通过受限翻译保证谓词在目标语言中的一致性；③构建可扩展至任意有基本资源语言的低成本投影框架。

**🔧 技术方法**

使用了英语 QA‑SRL 解析器、英语↔目标语言机器翻译（Helsinki‑NLP/Opus‑MT）、SimAlign 词对齐、基于 LLM 的受限问答翻译、LoRA 微调、自动化评估（IOU 句子匹配与语义匹配）等技术。

**📊 数据集**

数据来源：Universal Dependencies（希伯来语、俄语、法语）作为句子来源；英语 QA‑SRL 数据集提供解析器；手工标注的黄金评估集；以及由投影流程产生的三语言 QA‑SRL 训练集。

**📈 对比分析**

与多语言 LLM 基线（GPT‑4o、LLaMA‑Maverick）以及多语种 8B LLaMA‑3 在 Unlabeled Match、Exact Match、Semantic Match 三个指标上对比。结果显示：①Unlabeled Match 与 LLM 基线相当或略优；②在问题生成（Exact/semantic）上，语言特定微调模型显著优于基线，尤其在法语上表现最为突出；统计显著性检验证明差异显著。

**⚠️ 局限性**

局限性：①整个流程由 MT、解析器、对齐、受限翻译等多模块组成，误差易累积；②主要受限于英语 QA‑SRL 解析器的精度；③对极低资源或形态高度丰富的语言适用性未充分验证，可能需要额外资源或方法改进。

---

## 325. Graph Your Way to Inspiration: Integrating Co-Author Graphs with Retrieval-Augmented Generation for Large Language Model Based Scientific Idea Generation

**arXiv ID:** 2602.22215 | [PDF](https://arxiv.org/pdf/2602.22215v1)

**作者:** Pengzhen Xie `[一作]` (Newcastle University), Huizhi Liang `[通讯]` (Newcastle University)

**通讯引用:** 787 | [OpenAlex ID](https://openalex.org/A5075528828)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出GYWI系统，将作者知识图谱与检索增强生成（RAG+GraphRAG）相结合，并通过强化学习优化提示，生成可追溯、可控的科研创新想法。

**💡 创新点**

创新点包括：① 以作者为中心构建知识图谱并设计探索与利用跳点采样；② 混合深度（RAG）与广度（GraphRAG）检索形成层次化上下文；③ 基于RL的提示优化机制，使LLM更好利用检索证据。

**🔧 技术方法**

使用技术包括：作者中心知识图谱构建、邻域/随机跳点采样、RAG、GraphRAG混合检索、Prompt优化（RL）以及多种LLM模型（GPT‑4o、DeepSeek‑V3、Qwen3‑8B、Gemini 2.5）。

**📊 数据集**

构建了基于arXiv 2018‑2023的1000篇AI领域论文评测集，包含IMCQ多项选择题、自动与人工评分数据。

**📈 对比分析**

通过与GPT‑4o、DeepSeek‑V3、Qwen3‑8B、Gemini 2.5等基线在IMCQ准确率、LLM自动评分、人工评估和语义空间可视化等四维度评估，GYWI在新颖性、可靠性、相关性等指标上显著优于基线，IMCQ准确率提升超过10个百分点。

**⚠️ 局限性**

局限性包括：图谱与检索索引一次性构建，缺乏热更新；跨领域泛化受限；对新近论文的实时适应性不足。

---

## 326. SignVLA: A Gloss-Free Vision-Language-Action Framework for Real-Time Sign Language-Guided Robotic Manipulation

**arXiv ID:** 2602.22514 | [PDF](https://arxiv.org/pdf/2602.22514v1)

**作者:** Xinyu Tan `[一作]` (University College London), Zezhi Tang `[通讯]` (University College London)

**通讯引用:** 136 | [OpenAlex ID](https://openalex.org/A5075862536)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了首个无词汇标注的实时手语驱动视觉‑语言‑动作框架（SignVLA），通过手指拼写实现机器人指令的感知与执行。

**💡 创新点**

创新点在于：1）采用无词汇（gloss‑free）方法直接将手语图像映射为语义指令；2）构建了字母级感知→语言缓冲→动作执行的模块化管线；3）将手语感知与大规模 VLA 基础模型（GR00T/N1）无缝集成，实现可扩展的跨硬件控制。

**🔧 技术方法**

核心技术包括 MediaPipe Hands 3D 关键点提取、ResNet (2+1)D 时空卷积网络、Levenshtein 词典校正、GR00T N1 VLM 与 Diffusion Transformer（DiT）动作生成、跨模态注意力融合以及动作片段预测。

**📊 数据集**

使用了自制的小规模手语数据集 D_raw 进行扩增训练，并在公开的 ASL 指拼、CSL（Continuous Sign Language）数据集上进行评估，另外在实验中对 Franka Emika Panda 机器人进行真实场景验证。

**📈 对比分析**

与传统 CNN+LSTM、ResNet+LSTM、3D CNN 等基线对比，字母级识别在 100 类时 98.68%/94.85%（500 类）Top‑1 准确率，连续手语控制的任务成功率为 79.3%（无平滑）提升至 84.7%（加时间平滑），与文本控制（86.5%）相近，且延迟保持在实时级别。

**⚠️ 局限性**

局限性包括：仅支持字母级拼写；不支持连续大词汇手语；对多样化手势与光照鲁棒性仍有限；模型仍处于原型阶段，缺乏大规模端到端训练与评估；依赖 MediaPipe 关键点精度，易受遮挡影响。

---

## 327. ECHO: Encoding Communities via High-order Operators

**arXiv ID:** 2602.22446 | [PDF](https://arxiv.org/pdf/2602.22446v1)

**作者:** Emilio Ferrara `[一作]` (University of Southern California), Emilio Ferrara `[通讯]` (University of Southern California)

**通讯引用:** 18759 | [OpenAlex ID](https://openalex.org/A5078699564)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种可扩展的自监督社区检测框架 ECHO，利用高阶扩散和内存分片技术，能够在属性网络中同时克服语义平滑和系统瓶颈。

**💡 创新点**

创新点包括：① 通过Topology‑Aware Router自动选择特征编码器（GraphSAGE或MLP）；② 采用注意力引导的多尺度扩散动态抑制异构边；③ 引入全批次对比学习与张量分片，消除 O(N²) 内存瓶颈；④ 设计块式 O(N·K) 相似度提取，保持梯度精度并实现百万级节点的聚类。

**🔧 技术方法**

使用技术包括 GraphSAGE、MLP、注意力机制、InfoNCE 对比学习、张量分片、块式相似度抽取、基于模块度的 Leiden/Modularity 最大化，以及 PyTorch 2.6 的 GPU 加速实现。

**📊 数据集**

评估数据集涵盖合成 LFR 基准（高噪声、临界混合参数）以及十个公开属性网络：Chameleon、Actor、Amazon Photo、Amazon Computers、Coauthor CS、CoraFull、YouTube（合成）和 Pokec（真实社交）。

**📈 对比分析**

与 LPA、LINE、DGI、MVGRL、SDCN 等传统与自监督方法在 NMI 指标上比较，ECHO 在异构、密集网络中显著优于对照组；在百万节点图上单 GPU 运行完成时间仅数分钟，吞吐率超过 2,800‑3,300 节点/秒，突破 O(N²) 计算壁垒。

**⚠️ 局限性**

局限性包括：① 扩散步长 K 的全局超参数需手工调节，缺乏节点级自适应；② 对比学习参数（温度、稀疏惩罚）对结果敏感；③ 目前仅支持单一边类型，异构/多关系图尚未覆盖；④ 单 GPU 分片方案在更大规模或多机环境下仍需进一步分布式优化。

---

## 328. EgoAVFlow: Robot Policy Learning with Active Vision from Human Egocentric Videos via 3D Flow

**arXiv ID:** 2602.22461 | [PDF](https://arxiv.org/pdf/2602.22461v1)

**作者:** Daesol Cho `[一作]` (Georgia Institute of Technology), Sehoon Ha `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 4690 | [OpenAlex ID](https://openalex.org/A5064581452)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过人类正视角视频学习机器人操控和主动视角控制，不依赖机器人演示

**💡 创新点**

提出共享3D流表示以消除人机身体差距，并用可视度最大化的扩散模型实现视角调整，结合预测3D流与动作生成

**🔧 技术方法**

使用CoTracker3、HaMeR、DROID‑SLAM等工具获取3D流和手部姿态，利用扩散Transformer与soft value‑based denoising进行视角与动作预测，并通过raycasting计算可见度奖励

**📊 数据集**

使用150条人类正视角视频（RealSense D435）收集的四个操作任务数据集，每个任务包含六个查询点，完全无机器人演示

**📈 对比分析**

与人类视角模仿(HVI)、AMPLIFY、EgoZero、Phantom等基线比较，EgoAVFlow在四个任务中成功率平均提升1.8–2.5倍，明显优于最好的基线

**⚠️ 局限性**

假设起始时查询点已被跟踪，未处理查询点不可见或需主动搜索感兴趣点的情况

---

## 329. CQSA: Byzantine-robust Clustered Quantum Secure Aggregation in Federated Learning

**arXiv ID:** 2602.22269 | [PDF](https://arxiv.org/pdf/2602.22269v1)

**作者:** Arnab Nath `[一作]` (Indian Institute of Technology), Harsh Kasyap `[通讯]` (Indian Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了基于聚类的量子安全聚合协议（CQSA），在联邦学习中通过将客户端划分为小规模聚类、使用低阶GHZ态进行本地量子聚合，并在服务器端对聚类结果进行统计验证，以实现安全与Byzantine鲁棒性的兼顾。

**💡 创新点**

创新点在于：①引入动态随机聚类，将大规模量子纠缠切分为可维护的小GHZ子组，显著提升量子态保真度；②在聚类层面实现跨聚类验证（利用余弦相似度、欧氏距离等统计指标），突破传统量子安全聚合对恶意客户端不可验证的缺陷；③兼顾NISQ硬件的物理限制与安全需求，提供一种可落地的模块化方案。

**🔧 技术方法**

使用的技术包括：量子纠缠（GHZ态）与相位编码、量子安全聚合（QSA）框架、离散量子门与去噪（抖动）模型、统计学方法（余弦相似度、欧氏距离、Krum等Byzantine鲁棒聚合算法）、模拟平台CUDA‑Q进行量子态保真度评估，以及经典通信实现动态聚类与结果广播。

**📊 数据集**

在实验中主要使用了仿真数据；未涉及真实机器学习数据集，主要通过在CUDA‑Q上对GHZ态在不同噪声与聚类规模下的保真度进行模拟。

**📈 对比分析**

比较方法：在不同噪声概率（p=0.005~0.01）和不同聚类规模（k=1~10）下，计算全球GHZ聚合与CQSA的总体保真度。结果显示：CQSA在几乎所有聚类大小下均优于全局方案，尤其在小聚类（k≤5）时保真度提升显著；随着噪声增大，CQSA仍保持较高保真度，显示出更好的鲁棒性。

**⚠️ 局限性**

主要局限：①恶意检测只能在聚类层面进行，无法在单个聚类内部识别具体恶意客户端；②聚类大小过小时可能出现聚类级别信息泄露；③协议依赖于量子硬件的多通道并行能力，对硬件资源有一定要求；④在极低噪声或大型网络场景下，仍需进一步验证与优化。

---

## 330. Approximately Solving Continuous-Time Mean Field Games with Finite State Spaces

**arXiv ID:** 2602.23174 | [PDF](https://arxiv.org/pdf/2602.23174v1)

**作者:** Yannick Eich `[一作]` (Technische Universität Darmstadt), Heinz Koeppl `[通讯]` (Technische Universität Darmstadt)

**通讯引用:** 3034 | [OpenAlex ID](https://openalex.org/A5070544702)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在连续时间离散状态的平均场游戏中引入了熵正则化的均衡概念（RE），并通过求解熵正则化的哈密顿-雅可比-贝尔曼方程得到其固定点特征；

**💡 创新点**

创新点在于将离散时间中正则化均衡的思想推广到连续时间CTMC平均场游戏，并给出了RE的存在性证明、软价值函数与软max策略的关系；

**🔧 技术方法**

使用了熵正则化的强化学习框架、哈密顿-雅可比-贝尔曼方程求解、固定点迭代（FPI）和虚拟游戏（FP）两种迭代算法；

**📊 数据集**

实验数据集为三类合成平均场游戏：左-右（LR）问题、随机生成的群体避免奖励游戏以及SIS传染病控制模型；

**📈 对比分析**

通过对比ΔJ和ΔJ^RE的收敛曲线，结果显示FP在较低温度下仍能收敛并逼近Nash均衡，而FPI在高温度下收敛，但随着温度降低容易振荡，整体性能呈现收敛速度与逼近度的权衡；

**⚠️ 局限性**

局限性包括缺乏对FP收敛性的严格理论证明、FPI对温度高度敏感导致在逼近NE时可能不稳定，以及RE作为近似NE时可能偏离原始均衡导致策略效果不佳。

---

## 331. Vision Transformers Need More Than Registers

**arXiv ID:** 2602.22394 | [PDF](https://arxiv.org/pdf/2602.22394v1)

**作者:** Cheng Shi `[一作]` (University of Hong Kong), Sibei Yang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1035 | [OpenAlex ID](https://openalex.org/A5043811579)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过系统分析 ViT 的背景聚合伪信号（artifact）并提出 LaSt‑ViT 的频率引导选择聚合机制，消除不同监督（全监督、文本监督、自监督）下的 lazy aggregation 行为。

**💡 创新点**

1) 统一定义 Patch Score 与 Point‑in‑Box（PiB）量化 artifacts；2) 发现粗粒度监督与全局注意力共同导致背景占优的 lazy aggregation；3) 通过通道低通滤波得到稳定性得分，按 Top‑K 选择最稳定 token 聚合 CLS token；4) 在多种监督模式下实现统一改进。

**🔧 技术方法**

频域稳定性筛选（FFT + 低通滤波 + Top‑K 聚合）、Patch Score 与 PiB 指标、PCA 可视化、零样本语义分割、对象发现、实例分割等下游任务；同时对 ViT 进行全监督、文本监督（CLIP）与自监督（DINO）预训练。

**📊 数据集**

ImageNet‑1k（预训练）；VOC、COCO、Cityscapes、LVIS、Pascal、VOC12、VOC07 等用于下游评估。

**📈 对比分析**

与 ResNet、传统 ViT、Register、DINO、CLIP 等 baseline 进行对比，PiB 指标从 42% 提升至 55%+；在 12 个下游基准上平均提升 3–10%，零样本语义分割 mIoU 从 11.2% 提升至 75%，对象发现 CorLoc 最高 64.4% 等，整体性能显著优于基线。

**⚠️ 局限性**

需要在减少背景占比的同时略牺牲分类准确率；方法主要针对全局注意力 ViT，窗口注意力或轻量化 ViT 的效果尚未充分验证；对极小或稀疏目标的局部细节聚合仍有限；计算开销略高。

---

## 332. Detecting Hate and Inflammatory Content in Bengali Memes: A New Multimodal Dataset and Co-Attention Framework

**arXiv ID:** 2602.22391 | [PDF](https://arxiv.org/pdf/2602.22391v1)

**作者:** Rakib Ullah `[一作]` (Sylhet Engineering College), Md Ismail Hossain `[通讯]` (Daffodil International University)

**通讯引用:** 1212 | [OpenAlex ID](https://openalex.org/A5100346399)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了包含 3,247 张孟加拉语及其混合语言图文 Meme 的 Bn-HIB 数据集，并提出了 Multi‑Modal Co‑Attention Fusion Model (MCFM) 用于检测 Hate、Inflammatory 与 Benign 三种内容。

**💡 创新点**

创新点在于：①首次将孟加拉语 Meme 按 Hate、Inflammatory 与 Benign 三类细分；②基于 Co‑Attention 的双向多模态融合框架，显著提升了对隐蔽、讽刺性攻击的识别；③公开了数据来源与标注准则，利于后续研究复现。

**🔧 技术方法**

使用了文本编码器（MDeBERTa‑V3、XLM‑RoBERTa 等）、视觉编码器（Swin‑V2、ViT‑B/16 等）以及 Co‑Attention、Cross‑Attention 与 CLIP 样式融合技术，最终以两层 MLP 进行多模态决策。

**📊 数据集**

使用了自建的 Bn‑HIB 数据集，涵盖孟加拉语、孟加拉语‑英语混合、代码切换三种语言场景；每类 811、773、688 条样本。

**📈 对比分析**

在 15 个单模态与多模态基线上进行对比（Accuracy/Macro‑F1），MCFM 获得 0.7746 Accuracy 与 0.7765 Macro‑F1，较最优基线（CLIP 0.7295/0.7322）提升约 5% Accuracy 与 5% F1，表明模型在细粒度类别区分上效果突出。

**⚠️ 局限性**

局限性包括：① hate 与 inflammatory 类别仍易混淆，主要受讽刺、文化符号与 OCR 噪声影响；② 数据集规模相对有限，难以覆盖所有地区与主题；③ 仅针对孟加拉语，缺乏跨语言推广与解释性（XAI）支持。

---

## 333. TWICE: An LLM Agent Framework for Simulating Personalized User Tweeting Behavior with Long-term Temporal Features

**arXiv ID:** 2602.22222 | [PDF](https://arxiv.org/pdf/2602.22222v1)

**作者:** Bingrui Jin `[一作]` (Shanghai Jiao Tong University), Mengyue Wu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1596 | [OpenAlex ID](https://openalex.org/A5109064838)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 TWICE 框架，用于模拟具有长期时间特征的个性化推文行为。

**💡 创新点**

创新点在于将个性化用户画像、事件驱动记忆模块和两阶段风格重写工作流结合，兼顾长期时序和个体特征。

**🔧 技术方法**

采用 LLM（如 Qwen3、Llama-3.1 等）作为生成器，配合事件检测、记忆检索、Big Five 评估、语义嵌入等技术。

**📊 数据集**

使用 Twitter‑STMHD 数据集，包含多种精神健康诊断标签的用户推文。

**📈 对比分析**

通过语义相似度、风格相似度、可读性、情绪 KL 散度等指标比较，TWICE 在个人化、时序一致性和情绪匹配上均优于基线，尤其在精神健康组表现更好。

**⚠️ 局限性**

限于评估主要依赖自动指标，缺乏人工评价，对极端情绪或低频事件的捕捉仍有限。

---

## 334. LineGraph2Road: Structural Graph Reasoning on Line Graphs for Road Network Extraction

**arXiv ID:** 2602.23290 | [PDF](https://arxiv.org/pdf/2602.23290v1)

**作者:** Zhengyang Wei `[一作]` (Stanford University), Jenny Suckale `[通讯]` (Stanford University)

**通讯引用:** 1895 | [OpenAlex ID](https://openalex.org/A5010700649)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出端到端的LineGraph2Road框架，利用卫星图像提取道路网络并生成图结构，能够处理多层交叉。

**💡 创新点**

创新点在于将连通性预测转化为全局稀疏欧氏图的边分类，使用线图与图Transformer学习边级表示，并加入过桥/下桥交叉头与耦合NMS。

**🔧 技术方法**

采用预训练的Segment Anything Model编码器与解码器，双线性采样生成候选边特征，随后通过线图转换后的图Transformer进行边级二分类。

**📊 数据集**

在City‑scale、SpaceNet和Global‑scale三大公开卫星图像道路基准上进行评估。

**📈 对比分析**

与RoadTracer、Sat2Graph、RNGDet、SAM‑Road++等方法比较，取得TOPO‑F1和APLS的最新最高分，尤其在City‑scale和Global‑scale上表现优异。

**⚠️ 局限性**

局限包括Global‑scale测试中精度相对偏低、对稀疏的过桥结构的鲁棒性仍待提升，以及对极端遮挡和分辨率变化的适应性尚不充分。

---

## 335. SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning

**arXiv ID:** 2602.22603 | [PDF](https://arxiv.org/pdf/2602.22603v1)

**作者:** Sanjay Kariyappa `[一作]` (NVIDIA), G. Edward Suh `[通讯]` (NVIDIA)

**通讯引用:** 11834 | [OpenAlex ID](https://openalex.org/A5024329178)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了SideQuest框架，让大型推理模型通过并行辅助线程主动管理KV缓存，从而在长周期Agentic任务中显著减少内存占用并保持推理质量。

**💡 创新点**

将KV压缩由固定启发式切换为模型驱动的自我记忆管理；利用并行辅助推理避免主推理被管理token污染；通过回溯注释生成训练数据，使模型学会判别何时删除工具响应。

**🔧 技术方法**

并行辅助线程、触发词+模型微调、logit蒸馏+交叉熵联合训练、LoRA+QAT、共享上下文推理、Cascade/FastTree等共享前缀加速技术。

**📊 数据集**

FRAMES（基于维基百科的多跳检索推理）和BrowseComp（网页导航检索）两套长序列任务；训练样本来源于FRAMES 215条高质量推理轨迹与1274条辅助训练轨迹。

**📈 对比分析**

与无压缩基线以及H₂O、SnapKV、R‑KV等启发式压缩方法比较；在FRAMES和BrowseComp上Peak KV使用率降低56‑65%，KV读取量下降53‑71%，准确率仅低于5%；系统吞吐量提升84%，整体跑时缩短37%。

**⚠️ 局限性**

训练数据量相对不足，导致在OOV BrowseComp上的准确率略降；仅压缩工具响应，未处理思考步骤；未结合低层注意力权重等指标，限制了压缩范围。

---

## 336. ProjFlow: Projection Sampling with Flow Matching for Zero-Shot Exact Spatial Motion Control

**arXiv ID:** 2602.22742 | [PDF](https://arxiv.org/pdf/2602.22742v1)

**作者:** Akihisa Watanabe `[一作]` (Waseda University), Kent Fujiwara `[通讯]` (LY Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种名为ProjFlow的零训练投影采样器，能够在不进行任务专用训练或推理时优化流匹配模型，精确满足线性空间运动控制约束。

**💡 创新点**

核心创新在于引入基于骨架拓扑的运动学感知度量，用于在投影步骤中保持关节连通性的自然运动；以及通过时间可变的伪观测和自适应方差实现稀疏约束下的高质量补全。

**🔧 技术方法**

技术包括流匹配模型（Rectified Flow）、闭式投影解算、运动学感知度量（含图拉普拉斯矩阵）以及噪声混合的随机重组步骤。

**📊 数据集**

实验使用HumanML3D数据集，对运动补全与2D→3D提升任务进行评估。

**📈 对比分析**

与现有零训练与训练基方法比较，ProjFlow在轨迹/位置误差上实现零误差，同时在FID、R-Precision等生成质量指标上与训练方法持平或更优，且无需额外训练或迭代优化。

**⚠️ 局限性**

局限性在于只能处理可线性化的约束，无法直接处理非线性或不等式约束，未来需扩展到更复杂的控制场景。

---

## 337. An automatic counting algorithm for the quantification and uncertainty analysis of the number of microglial cells trainable in small and heterogeneous datasets

**arXiv ID:** 2602.22974 | [PDF](https://arxiv.org/pdf/2602.22974v1)

**作者:** L. Martino `[一作]` (Università degli studi di Catania), E. Curbelo `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 46 | [OpenAlex ID](https://openalex.org/A5059915586)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种仅针对微胶细胞计数的自动算法，先通过多阈值颜色过滤与聚类得到低维特征，再利用核平滑器进行非参数回归，最终预测细胞数并给出不确定性估计。

**💡 创新点**

创新点在于（1）放弃目标检测，直接通过多阈值滤波提升信噪比；（2）采用单一超参数的核计数器，兼具非参数灵活性与易训练；（3）天然支持多专家意见与不确定性估计，且可处理小规模、异质的数据集。

**🔧 技术方法**

使用的技术包括：颜色阈值过滤、连通域聚类计数、Gaussian核平滑回归、留一交叉验证调参、适应性阈值与数据增强、误差与不确定性估计。

**📊 数据集**

实验数据集包括：12张来自马拉戈大学（URJC）的IHC微胶细胞切片图像（真实数据）；以及10^4张人工生成的合成图像用于验证算法收敛与鲁棒性。

**📈 对比分析**

与传统ImageJ手工计数、两种公开CNN模型（CNN-1、CNN-2）对比，核计数器在R²≈0.90、平均绝对误差<4个细胞、最大误差<25个细胞，且不确定区间始终覆盖真实值，性能优于对比方法。

**⚠️ 局限性**

局限性包括：对阈值选择的敏感性（虽可通过适应性阈值减弱）；样本量仍有限，且聚类算法的固定性可能引入偏差；在极端噪声或不规则染色条件下的鲁棒性尚待进一步验证。

---

## 338. CXReasonAgent: Evidence-Grounded Diagnostic Reasoning Agent for Chest X-rays

**arXiv ID:** 2602.23276 | [PDF](https://arxiv.org/pdf/2602.23276v1)

**作者:** Hyungyung Lee `[一作]` (KAIST), Edward Choi `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个集成大语言模型与临床诊断工具的诊断代理，在多轮对话中通过图像提取的诊断和视觉证据实现证据驱动的诊断推理。

**💡 创新点**

创新点在于将诊断工具返回的量化测量、空间观察和可视化注解作为中间证据，完全基于图像提取的临床证据来生成回答，从而避免传统 LVLMs 的幻觉和无根本证据问题，并提供可验证的视觉证据。

**🔧 技术方法**

技术包括大型语言模型（如 Gemini‑3‑Flash、GPT‑5 mini、Llama‑3.3‑70B、Qwen3 等）、基于规则的诊断工具 CheXStruct、工具调用框架、对话生成与验证的 LLM‑as‑a‑Judge。

**📊 数据集**

使用了 1,200 张来自 CXReasonBench 的胸部 X 光图像和 1,946 条多轮对话数据集（12 个诊断任务），并在这些数据上构建了诊断与视觉证据。

**📈 对比分析**

与三种 LVLM 基线（Gemini‑3‑Flash、Pixtral‑Large、MedGemma 27B）在三种评估设置（固定查询、提供 GT 对话历史、动态用户模拟）下对比，指标为任务识别、证据类型识别、覆盖率、真实性、幻觉和对话成功率。该代理在所有指标上均显著优于 LVLMs，尤其在对话成功率和真实性上提升显著。

**⚠️ 局限性**

局限在于仅覆盖胸部 X 光的 12 个预定义诊断任务，未扩展到更广泛的诊断任务或其他影像模态；工具依赖规则和专业知识，若面对新情况需重新定义规则。

---

## 339. A Data-Driven Approach to Support Clinical Renal Replacement Therapy

**arXiv ID:** 2602.22902 | [PDF](https://arxiv.org/pdf/2602.22902v1)

**作者:** Alice Balboni `[一作]` (Politecnico di Torino), Antonio Consolo `[通讯]` (Universita degli Studi di Milano-Bicocca)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文开发并评估了一种基于机器学习的数据驱动方法，用于预测危重患者CRRT治疗过程中滤膜堵塞事件，并提供可操作的反事实解释以指导临床干预。

**💡 创新点**

创新点在于将CRRT监测数据转换为可解释的表格特征集合，采用ADASYN重采样提升少数类表现；同时将Shapley值驱动的反事实分析与Mahalanobis距离结合，生成可执行的最小调整建议；并证明表格模型优于LSTM时序模型。

**🔧 技术方法**

使用的技术包括：随机森林、XGBoost、LightGBM、LSTM、ADASYN过采样、Shapley Additive exPlanations (SHAP)、基于Shapley的反事实生成（KNN背景、Mahalanobis距离）以及10折交叉验证与Optuna调参。

**📊 数据集**

数据集来自2011‑2014年Careggi大学医院ICU的CRRT时间序列，共796次治疗，其中91次出现滤膜堵塞；选取16个临床特征，最终以12个表格特征构建样本。

**📈 对比分析**

与不同平衡率（0%、5%、10%、20%）以及不同预测时延（10‑60 min）对比。10%平衡率下，XGBoost敏感度最高（≈0.78）、特异性≈0.96，优于LSTM；使用仅前5个重要特征时准确性下降不足10%；反事实分析在D_10‑NN背景下和Mahalanobis距离组合表现最优。

**⚠️ 局限性**

局限性包括：使用历史数据，未在实时或仿真环境验证；表格化特征忽略了时序依赖，可能失去部分信息；模型对不同滤膜的通用性尚未充分测试。

---

## 340. A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring

**arXiv ID:** 2602.23163 | [PDF](https://arxiv.org/pdf/2602.23163v1)

**作者:** Usman Anwar `[一作]` (University of Cambridge), David Krueger `[通讯]` (University of Cambridge)

**通讯引用:** 5183 | [OpenAlex ID](https://openalex.org/A5029025914)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于决策理论的隐写学形式化，并提出可用信息的通用化度量（Generalised V-information）和隐写差距（Steganographic Gap）来检测和量化大语言模型（LLM）中的隐写推理。

**💡 创新点**

创新点在于：①将隐写的本质定义为不同代理在获取隐藏信息后产生的可用信息不对称；②利用决策理论和可用信息的概念，用“下游行为”而非信号统计特征来衡量隐写；③提出了可调节的隐写差距指标，既可检测也可量化隐写强度，并可用于训练去隐写的重述器。

**🔧 技术方法**

核心技术包括：通用化可用信息（Generalised V-information）框架；隐写差距的定义与计算；基于强化学习的编码策略（如 Rot13、字母表替换、二元组替换密码）；对齐 Sentinel 与 Receiver 的解码能力；以及对多种 LLM（Qwen、GPT-4.1/5、Qwen2.5-3/14B）进行的实验评估。

**📊 数据集**

使用的数据集主要有：GSM8k、MMLU‑Pro、CoinFlip（状态追踪任务）、APPS（竞争程序设计+后门任务）以及自构造的贷款申请摘要任务。实验通过对这些任务的不同编码强度和代码本大小进行控制，评估隐写差距的变化。

**📈 对比分析**

实验比较方法：在相同任务下分别让 Receiver（携带完整推理轨迹）和 Sentinel（只接收加密或伪造的轨迹）执行相同评估任务（如答案预测、评估意识检测、秘密提取）。结果表明：①隐写差距随着编码强度、代码本不匹配和训练样本数的增加而线性升高，达到接近 1；②在 Sentinel 具备更高算力或经过专门训练的重述器时，差距显著下降；③在公开编码方案（如情感风格）下，差距几乎为 0，验证了定义的有效性。

**⚠️ 局限性**

局限性包括：①需要预先定义评估任务和真值标签，缺乏无监督评估方案；②差距估计依赖于对 Receiver 与 Sentinel 解码能力的准确估计，若提示不足可能导致误判；③对抗性自适应模型可能通过“沙袋”或其他技巧减少差距，从而逃避检测；④方法主要适用于可拦截、可重放且即时产生可观察行为的场景（如 LLM 推理轨迹），不适用于所有隐写场景。

---

## 341. S2O: Early Stopping for Sparse Attention via Online Permutation

**arXiv ID:** 2602.22575 | [PDF](https://arxiv.org/pdf/2602.22575v1)

**作者:** Yu Zhang `[一作]` (ByteDance), Xing Wang `[通讯]` (ByteDance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 S2O，一种在线稀疏注意力方案，利用轻量级索引实现在线重排并通过早停裁剪低贡献块，以提升长上下文推理效率。

**💡 创新点**

创新在于将稀疏注意力的物理重排转为逻辑索引重排，并结合在线softmax状态和阈值早停，实现更高的稀疏度上限和更精细的注意力聚焦。

**🔧 技术方法**

使用 FlashAttention 兼容的分块计算、轻量级索引加载、在线softmax状态跟踪、早停阈值、分段段内查询排序和历史键值排序。

**📊 数据集**

在 Llama-3.1-8B 与 Qwen3-8B 上进行实验，并在 RULER、LongBench v2 和 InfiniteBench 等长文本基准上评估。

**📈 对比分析**

与 Full、FlexPrefill、XAttn、PBS 等基线相比，在匹配稀疏度时 MSE 降低 3.8×，在匹配 MSE 时稀疏度提升 3.3×，单算子速度提升 7.5×，端到端速度提升 3.8×。

**⚠️ 局限性**

缺点是需要调节段长、阈值等超参数，且目前仅验证在解码阶段，未扩展至编码-解码或多模态模型，性能受特定 GPU/软件栈限制。

---

## 342. DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis

**arXiv ID:** 2602.23022 | [PDF](https://arxiv.org/pdf/2602.23022v1)

**作者:** Xinglong Luo `[一作]` (University of Electronic Science and Technology of China), Shuaicheng Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 7009 | [OpenAlex ID](https://openalex.org/A5039387461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于扩散模型的图像对齐框架DMAligner，采用视角合成方式实现对齐；

**💡 创新点**

引入动态感知掩模生成模块(DMP)，使扩散模型在训练中能够识别并重点处理动态前景；

**🔧 技术方法**

利用潜在扩散模型(LDM)、VQ‑VAE、以及自定义的掩模预测网络；

**📊 数据集**

创建大型动态场景对齐数据集DSIA（1,033场景，30K+图像对），并在Sint​el、DAVIS等公开数据集上进行测试；

**📈 对比分析**

与多种基于光流 warping 与视角合成的方法对比，DMAligner在DSIA上PSNR/SSIM均优于现有方法，在 Sint​el、DAVIS 上LPIPS/DreamSim指标亦取得最佳平均分数；

**⚠️ 局限性**

局限性包括：对极端光照或遮挡变化的鲁棒性尚待进一步验证；扩散模型推理速度较慢，需要优化推理效率。

---

## 343. AHBid: An Adaptable Hierarchical Bidding Framework for Cross-Channel Advertising

**arXiv ID:** 2602.22650 | [PDF](https://arxiv.org/pdf/2602.22650v1)

**作者:** Xinxin Yang `[一作]` (OPPO), Bo Yang `[通讯]` (OPPO)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种适用于跨渠道广告的自适应分层竞价框架AHBid，融合扩散模型生成预算/约束轨迹和实时控制实现预算分配与竞价决策。

**💡 创新点**

通过将高层扩散生成器与约束执行和轨迹细化机制相结合，实现对预算/约束的动态规划，并引入历史与实时模型的控制式竞价，提高对环境变化的自适应性。

**🔧 技术方法**

采用扩散模型生成轨迹、条件指导和混合损失实现约束满足；使用逆动力学+历史MPC与实时LP的双模型控制，结合U-Net架构和Diffusion Probabilistic Models。

**📊 数据集**

在公开AuctionNet数据集和OPPO内部GenB-4c四渠道广告轨迹数据上进行离线评估，随后在真实广告平台上开展两周A/B测试。

**📈 对比分析**

与PID、USCB、MCQ、ABPlanner、DiffBid、HiBid等基线在离线回测和线上A/B测试中对比，AHBid在回测上实现了8.42%（GenB-4c）/4.99%（AuctionNet）收益提升，线上A/B测试中收益提升13.57%，约束满足率提升4.13%，预算消耗率提升5.06%，整体表现领先。

**⚠️ 局限性**

对扩散步骤和指导强度敏感，需大量离线数据训练；约束执行仍通过损失近似，可能在极端动态环境下失效；模型对新渠道迁移需要重新聚类和校准。

---

## 344. Persistent Nonnegative Matrix Factorization via Multi-Scale Graph Regularization

**arXiv ID:** 2602.22536 | [PDF](https://arxiv.org/pdf/2602.22536v1)

**作者:** Jichao Zhang `[一作]` (Xi'an Jiaotong University), Limin Li `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 3624 | [OpenAlex ID](https://openalex.org/A5100388520)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种持久性非负矩阵分解（pNMF），在多尺度上学习可解释的低秩嵌入。

**💡 创新点**

创新点在于利用0维持久同调确定的最小足够尺度集，构造多尺度图拉普拉斯并加入跨尺度几何正则化和一致性约束，生成连续尺度的嵌入路径。

**🔧 技术方法**

结合非负矩阵分解、持久同调、可持续图拉普拉斯正则化、跨尺度平滑与锚定约束，并采用序贯交替优化实现求解。

**📊 数据集**

使用了合成多尺度圆环数据和五个公开的单细胞RNA测序数据集（GSE75748time、GSE94820、GSE75140、Dendritic_batch1、Dendritic_batch2）。

**📈 对比分析**

与传统NMF、GNMF、TNMF、k‑TNMF等基线进行对比，采用ARI、NMI、Purity、Accuracy评估聚类性能，pNMF在大多数数据集上实现最优或接近最优的结果，并能自适应不同尺度。

**⚠️ 局限性**

局限在于仅利用0维持久同调，未考虑高维拓扑特征；优化方法仍为迭代交替，对大规模或流式数据适应性待提升；对下游任务的完整利用与更深层次的层次聚类仍有待探索。

---

## 345. Relatron: Automating Relational Machine Learning over Relational Databases

**arXiv ID:** 2602.22552 | [PDF](https://arxiv.org/pdf/2602.22552v1)

**作者:** Zhikai Chen `[一作]` (Michigan State University), Huzefa Rangwala `[通讯]` (Amazon)

**通讯引用:** 5900 | [OpenAlex ID](https://openalex.org/A5006581225)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统性的大规模架构搜索，构建了一个覆盖RDB预测任务的模型性能库，并对Relational Deep Learning（RDL）与Deep Feature Synthesis（DFS）两大范式进行统一设计空间评估，提出了基于任务嵌入的元选择器Relatron，自动决定使用RDL还是DFS，并在微观层面进一步剪枝搜索空间，显著提升了预测效果。

**💡 创新点**

创新点在于：①统一RDL与DFS的设计空间，揭示两者优势随任务属性变化；②提出RDB任务同质性（homophily）与affinity嵌入作为任务特征，并用其预测RDL/DFS优劣；③开发Relatron元选择器，结合损失景观指标实现高效且鲁棒的架构与超参搜索；④构建公开性能库，为后续研究提供基准。

**🔧 技术方法**

主要技术包括：图神经网络（PNA、HGT、SAGE、RelGNN）与消息传递；DFS的SQL级聚合与特征增益；任务嵌入技术（homophily、affinity、训练免费模型性能、Anchor-based embedding）；元学习分类器（meta-classifier）；损失景观度量（P1、P2、P_bar）用于后验模型挑选；自动化搜索框架（TPE、Hyperband、Autotransfer）。

**📊 数据集**

使用了Relbench公开基准，包含17个实体级任务（分类、回归、推荐、补全），以及多源真实RDB场景；所有实验均在同一GPU (L40S 48GB) 上统一配置。

**📈 对比分析**

方法比较采用Leave-One-Out交叉验证，使用性能银行作为对照，实验结果表明：①在宏观层面，Relatron的RDL/DFS选择准确率可达87.5%；②在微观层面，联合超参搜索与Relatron可比传统随机、TPE、Hyperband等方法提升约3–4% (AUROC/MAE)，并在搜索次数10×减少时仍保持或提升性能；③验证集选择往往误导，损失景观后验可显著弥补该缺陷。

**⚠️ 局限性**

主要局限包括：①未覆盖LLM/大模型作为编码器或预测器的表现，缺乏对这些方法的系统评估；②当前任务嵌入仍相关性有限，导致迁移搜索效果受限；③性能银行规模受限于公开RDB任务数量，难以覆盖更广泛场景；④对非常小数据集或高维特征的适用性未充分验证。

---

## 346. Work Design and Multidimensional AI Threat as Predictors of Workplace AI Adoption and Depth of Use

**arXiv ID:** 2602.23278 | [PDF](https://arxiv.org/pdf/2602.23278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 347. SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables

**arXiv ID:** 2602.23286 | [PDF](https://arxiv.org/pdf/2602.23286v1)

**作者:** Sungho Park `[一作]` (POSTECH), Wook-Shin Han `[通讯]` (POSTECH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一种端到端的自动化框架SPARTA，用于生成大规模、深层次的表–文本问答基准数据集，支持多跳推理、聚合、分组等高级分析操作。

**💡 创新点**

创新点包括：① SQL为核心的生成管道，结合执行引导与后序遍历；② 通过“原子事实表”将文本信息化为关系事实，消除信息重叠；③ 采用原子语义依据（Provenance‑based refinement）和结构约束（realistic‑structure enforcement）保证生成的SQL可执行且自然；④ 只需轻量化人工校验，显著降低注释成本。

**🔧 技术方法**

主要技术：大语言模型（Llama‑3.1‑70B‑Instruct）进行SQL生成与问题口语化；执行引导、后序构建、why‑not 原因推断；AST‑ICL用于将SQL转化为自然语言；数据库工具用于事实表构建与原子事实提取；评估使用ChatGPT‑4o进行自动自然度评分。

**📊 数据集**

数据集来源：NBA领域（Rotowire、6个公开NBA数据集）、电影领域（Kaggle数据）、医学领域（Kaggle数据）。通过将源表与从文本中抽取的原子事实合并形成参考事实数据库，构成统一的表‑文本数据。

**📈 对比分析**

与现有基准（HybridQA、OTT‑QA、TAT‑QA、FinQA、MultiHiertt）对比：在SPARTA上，ODYSSEY与HProPro在Oracle设置下F1仅为35–40%（相比HybridQA的70%降幅30+），检索设置下F1更低（≈22%）。模型在多跳、宽度更大、包含聚合/分组/排序等高级操作时表现进一步恶化，显示跨模态推理的显著瓶颈。

**⚠️ 局限性**

局限性：目前仅覆盖表–文本场景，未扩展到图像、视频等多模态；生成过程仍依赖LLM，可能出现语义偏差；仅在公开的结构化表与文本上验证，未涵盖更大规模或多源真实数据库；需要进一步提升对复杂树形查询的解析与执行能力。

---

## 348. AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction

**arXiv ID:** 2602.22376 | [PDF](https://arxiv.org/pdf/2602.22376v1)

**作者:** Hanyang Liu `[一作]` (Ohio State University), Rongjun Qin `[通讯]` (Ohio State University)

**通讯引用:** 3917 | [OpenAlex ID](https://openalex.org/A5017812815)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种面向单目无人机视频的4D高斯光栅化框架AeroDGS，用于从单镜头航拍序列中重建物理一致的四维动态场景并实现高质量的可视化与新视角合成。

**💡 创新点**

将单目几何提升模块与物理引导优化模块结合，利用可微地面支撑、垂直稳定和轨迹平滑等物理先验来消除单目深度歧义，实现对动态对象的稳定运动估计，并在单镜头航拍下首次实现高精度4D重建。

**🔧 技术方法**

使用显式3D高斯原语构建场景表示，结合光栅化渲染；采用基于hash网格、球谐基与时间正弦嵌入的外观字段；结合可微束平差、基于高斯的物理约束优化；以及基于零样本2D先验的几何提升。

**📊 数据集**

构建并公开Aero4D真实无人机数据集（50–100 m高空，15 fps，2K–4K分辨率），并在合成UAV3D与Aero4D上进行评测。

**📈 对比分析**

与4DGS、BézierGS、CoDa‑4DGS、DeGauss、4DGF等现有动态4D重建方法在新视角合成指标（PSNR/SSIM/LPIPS）及动态区域PSNR上进行对比，AeroDGS在所有场景下均取得最高PSNR/SSIM、最低LPIPS，并在动态区域上提升约4 dB，证明其在航拍条件下的优越性能。

**⚠️ 局限性**

对运动幅度小于3 m的动态物体会被误判为静态导致模糊；未对行人等极小物体进行重建；在极高空或极大视角差下仍受单目深度不确定性影响。

---

## 349. MViR: Multi-View Visual-Semantic Representation for Fake News Detection

**arXiv ID:** 2602.22944 | [PDF](https://arxiv.org/pdf/2602.22944v1)

**作者:** Haochen Liang `[一作]` (Great Bay University), Zitong Yu `[通讯]` (Great Bay University)

**通讯引用:** 5019 | [OpenAlex ID](https://openalex.org/A5062522283)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多视角视觉语义表示框架MViR，用于多模态假新闻检测。

**💡 创新点**

创新点在于引入多视角表示（MVR）模块以捕获图像的多角度语义信息，结合多视角特征融合（MVFF）和多视角聚合（MVA）实现更精细的跨模态信息交互。

**🔧 技术方法**

使用VGG-19提取图像特征、BERT生成文本特征，结合金字塔膨胀卷积、共注意力机制和多头自注意力以及多聚合器进行特征融合与聚合。

**📊 数据集**

在公开的Weibo和GossipCop两大中文假新闻数据集上进行实验。

**📈 对比分析**

与EANN、SAFE、MCAN、CAFE、FND-CLIP、MSACA、EVENT-RADAR等基线模型比较，MViR在Weibo上取得92.4%准确率，GossipCop上取得89.5%准确率，均优于最优基线1.9%~1.7个百分点。

**⚠️ 局限性**

局限性包括仅在两类数据集上验证，缺乏跨语言或跨平台的泛化评估，且对视角数、层数等超参数敏感，过多视角或层数可能导致信息冗余和过拟合。

---

## 350. Considering Perspectives for Automated Driving Ethics: Collective Risk in Vehicular Motion Planning

**arXiv ID:** 2602.22940 | [PDF](https://arxiv.org/pdf/2602.22940v1)

**作者:** Leon Tolksdorf `[一作]` (Technische Hochschule Ingolstadt), Nathan van de Wouw `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 13797 | [OpenAlex ID](https://openalex.org/A5004028181)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在自动驾驶车辆（AV）运动规划中引入视角感知风险的策略，能够在不同道路使用者（自我、他人、集合）视角下评估并最小化碰撞风险。

**💡 创新点**

创新点：①将风险视角从单一的“自我”扩展为多视角（自我、他人、集合）并证明风险在不同视角下具有非对称性；②提出基于风险视角的SMPC（随机模型预测控制）框架；③展示通过视角切换可兼顾安全、效率与伦理性。

**🔧 技术方法**

主要技术：多圆形碰撞模型、期望严重度风险评估、Gaussian不确定性传播、随机模型预测控制（SMPC）、人工势场（APF）约束、离散化控制与路径跟踪误差惩罚。

**📊 数据集**

使用的数据集：CommonRoad（包含 683 个真实与人工设计场景）以及 NGSIM 记录的真实道路数据。

**📈 对比分析**

比较方法：在三种风险视角（自我、他人、集合）和三种对象不确定性（低/中/高）下执行大规模仿真。结果显示：集合视角可将整体风险降低 8.4–22.3% 同时仅提升 AV 的自身风险 7.5–8.7%；相较于自我视角，集合视角能显著降低他人风险；在低不确定性场景下，集合视角实现的风险分布更稳定。

**⚠️ 局限性**

局限性：①对对象视角不确定性采用人工加权估计，缺乏实际感知与预测模型；②仅使用无通信假设，忽略车联网信息；③严重度模型过于简化，未考虑不同车辆类型与碰撞角度差异；④仿真结果需在真实车辆上进一步验证。

---

## 351. Optimizing SSD-Resident Graph Indexing for High-Throughput Vector Search

**arXiv ID:** 2602.22805 | [PDF](https://arxiv.org/pdf/2602.22805v1)

**作者:** Weichen Zhao `[一作]` (East China Normal University), Weining Qian `[通讯]` (East China Normal University)

**通讯引用:** 3872 | [OpenAlex ID](https://openalex.org/A5089931216)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种面向SSD的图结构近似最近邻搜索系统VeloANN，在大规模向量搜索中实现高吞吐量和低延迟；

**💡 创新点**

创新点包括协程异步执行模型、局部性感知的数据布局与记录级缓存池、以及预取与缓存感知波束搜索策略；

**🔧 技术方法**

采用异步I/O（io_uring）、4位/1位量化、整数压缩、B‑树风格页面布局、协程调度、记录级缓冲与时钟式置换；

**📊 数据集**

使用Sift1M、Gist1M、Wiki、Image、Text五个公共向量数据集（分别从Sift1M、Gist1M、Wiki、512维图像嵌入、768维文本嵌入构建）；

**📈 对比分析**

与DiskANN、Starling、PipeANN等现有磁盘ANN系统对比，在相同Recall@10下，吞吐量提升至5.8×、延迟降低至3.25×；在20%内存预算下，QPS可达近原始内存系统的0.92×；

**⚠️ 局限性**

限制在于单查询的关键路径仍受磁盘I/O阻塞影响，且系统依赖SSD高性能，若SSD性能下降或向量维度极高则效果可能受限。

---

## 352. Absorbing Discrete Diffusion for Speech Enhancement

**arXiv ID:** 2602.22417 | [PDF](https://arxiv.org/pdf/2602.22417v1)

**作者:** Philippe Gonzalez `[一作]` (Technical University of Denmark), Philippe Gonzalez `[通讯]` (Technical University of Denmark)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5103167575)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出基于吸收式离散扩散（ADD）在神经音频编码器（NAC）离散码空间进行条件建模的语音增强框架ADDSE，并设计了处理残差向量量化（RVQ）码层次结构的非自回归网络RQDiT。

**💡 创新点**

①首次将吸收式离散扩散应用于语音增强；②在离散码空间实现非自回归生成；③提出RQDiT统一处理RVQ码的时间与深度维度；④通过时间无关分布实现高效采样。

**🔧 技术方法**

使用吸收式离散扩散、残差向量量化NAC、RQ-Transformer + DiT（RQDiT）、自适应层归一化、旋转位置嵌入、EMD风格的采样与预处理技术。

**📊 数据集**

训练集：DNS5、LibriSpeech、MLS、VCTK、EARS、WHAM!、FSD50K、FMA、DEMAND；测试集：Libri‑TUT（跨噪声）和Clarity‑FSD50K（跨说话人）两套混合语音。

**📈 对比分析**

与Conv‑TasNet、BSRNN、SGMSE+、EDM‑SE、NAC‑SE、EDM‑NAC‑SE等基线在PESQ、ESTOI、SDR、DNSMOS、NISQA等多项指标上对比，ADDSE在低SNR环境下在非侵入指标上表现优于或与主流基线相当，且仅需8–16步采样即可获得接近最佳效果。

**⚠️ 局限性**

目前仅在低比特率（2 kbps）下验证，缺乏全频宽（4 kHz以上）或高采样率的实验；相位重建仍需改进；模型训练与推理仍占用较高计算资源，且对长时序或极端跨域噪声的鲁棒性尚未充分评估。

---

## 353. Why Diffusion Language Models Struggle with Truly Parallel (Non-Autoregressive) Decoding?

**arXiv ID:** 2602.23225 | [PDF](https://arxiv.org/pdf/2602.23225v1)

**作者:** Pengxiang Li `[一作]` (Hong Kong Polytechnic University), Shiwei Liu `[通讯]` (ELLIS Institute Tübingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 NAP（Non-Autoregressive Parallel Diffusion Language Models）方法，旨在通过重新设计监督信号和解码策略，使扩散式语言模型能够实现真正的并行非自回归生成。

**💡 创新点**

创新点在于：①将单条链式推理轨迹拆分为多条独立轨迹，从而消除对左至右顺序的依赖；②引入并行强制解码（Parallel‑Forced Decoding）框架，强制模型在多条轨迹上同步更新；③将监督数据与解码策略共设计，形成数据‑解码协同的学习范式。

**🔧 技术方法**

技术手段包括：掩码扩散模型（Masked Diffusion Language Model）作为基础；对采样顺序进行全局并行与局部基于置信度的更新；使用多块结构化输出画布（包含若干推理块与总结块）；以及对多路径推理的聚合与冲突过滤。

**📊 数据集**

使用了三类数据集：1）FineWeb 与 OpenR1‑Math 等传统预训练与长链式推理数据；2）自构造的多轨迹平行推理数据集 𝒟_parallel；3）评估基准包括 GSM8K、MATH‑500 与 GPQA。

**📈 对比分析**

与传统扩散模型（LLaDA‑8B、Dream‑7B）以及基于长链式推理的对照模型（Long‑CoT）进行对比。实验显示，NAP 在高并行度（如 256 步）下可将准确率提升 14‑15%，在低步数下提升 5‑6%；整体上在所有基准任务上均优于对照模型，并且 Global‑ARness 指标显著下降，表明解码更趋非自回归。

**⚠️ 局限性**

局限性包括：①仅在后训练阶段使用约 10 万样本进行实验，规模有限；②缺乏大规模预训练阶段的平行结构数据，难以评估完全消除自回归瓶颈的效果；③对不同模型规模与硬件环境的泛化性尚未系统验证。

---

## 354. CWM: Contrastive World Models for Action Feasibility Learning in Embodied Agent Pipelines

**arXiv ID:** 2602.22452 | [PDF](https://arxiv.org/pdf/2602.22452v1)

**作者:** Chayan Banerjee `[一作]` (Queensland University of Technology), Chayan Banerjee `[通讯]` (Queensland University of Technology)

**通讯引用:** 181 | [OpenAlex ID](https://openalex.org/A5043976016)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通过对比学习训练语言模型的动作可行性评分器（Contrastive World Model，CWM），并在科学文本环境中评估其效果。

**💡 创新点**

创新点在于采用InfoNCE对比损失与硬负样本挖掘，显式学习将合法动作与物理上错误的硬负样本区分开来，从而提升对物理可行性的判别。

**🔧 技术方法**

技术主要包括：LoRA微调Qwen-2.5-7B模型、InfoNCE损失、硬负样本挖掘（三类类型）以及对比训练与正则化。

**📊 数据集**

使用ScienceWorld文本模拟环境的数据集，包含数千个状态-动作对，并构造了六百零五个硬负样本进行评估。

**📈 对比分析**

与传统二分类SFT基线、原始LLM和随机打分进行对比；在最难的“最小编辑”硬负样本上，CWM的Precision@1提升了6.76pp，AUC-ROC为0.929；在OOD任务下，CWM的安全边距（Gold Action Retention）显著优于SFT。

**⚠️ 局限性**

局限性包括：只在动作层面验证，未完整集成至完整代理管线；对极度稠密或未知任务的泛化仍有限；硬负样本的构造依赖环境反馈，可能不适用于所有场景。

---

## 355. Positional-aware Spatio-Temporal Network for Large-Scale Traffic Prediction

**arXiv ID:** 2602.22274 | [PDF](https://arxiv.org/pdf/2602.22274v1)

**作者:** Runfei Chen `[一作]` (Tongji University), Runfei Chen `[通讯]` (Tongji University)

**通讯引用:** 41009 | [OpenAlex ID](https://openalex.org/A5100754344)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种轻量化的定位感知时空网络（PASTN）用于大规模交通流预测。

**💡 创新点**

创新点包括：①使用可学习的绝对位置编码（SPAE）提升节点区分度，缓解GNN的过平滑；②在时序模块中加入注意力机制（TPAM），增强对长时程依赖的捕捉；③整体设计保持模型轻量，参数量低、推理速度快。

**🔧 技术方法**

技术手段：图神经网络、时间卷积网络、门控TCN、绝对位置嵌入、Multi‑Head Self‑Attention、残差连接与层归一化，以及Adam优化。

**📊 数据集**

实验数据集：San Diego County（716节点）、Greater Bay Area（2352节点）、Greater Los Angeles Area（3834节点）、California State（8600节点），采样间隔5分钟，时间跨度2017‑2021年。

**📈 对比分析**

与12个基线（LSTM、DCRNN、AGCRN、STGCN、GWNET、ASTGCN、STTN、PDFormer、STGODE、DSTAGNN、DGCRN、D²STGNN）进行对比，PASTN在MAE、RMSE、MAPE上均夺得最优，尤其在CA数据集RMSE提升18.45%，MAE提升10.98%；参数量约36.8k，推理时间<0.05s，显示出优越的性能‑效率平衡。

**⚠️ 局限性**

局限性：仅利用图结构与短期时间序列，未充分考虑地点语义、功能属性、长期周期性和多模态信息；在大规模网络中仍面临内存与计算瓶颈；模型泛化需进一步验证，未来可探索知识蒸馏、基础模型预训练以及多模态融合。

---

## 356. CL4SE: A Context Learning Benchmark For Software Engineering Tasks

**arXiv ID:** 2602.23047 | [PDF](https://arxiv.org/pdf/2602.23047v1)

**作者:** Haichuan Hu `[一作]` (Nanjing University of Science and Technology), Quanjun Zhang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 612 | [OpenAlex ID](https://openalex.org/A5101756397)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了针对软件工程任务的上下文学习基准 CodeCL，构建了四种 SE 相关上下文类型并映射到四个核心任务，收集了 13k+ 样本数据并对五个主流 LLM 进行评估。

**💡 创新点**

①系统化的 SE 上下文类型细粒度分类；②为每个任务设计特定的上下文并构建大型真实项目数据集；③通过实验量化不同上下文对不同 SE 任务的异质影响，给出可操作的上下文设计指南。

**🔧 技术方法**

上下文工程、In-Context Learning、检索式上下文检索、LangChain + Chroma 向量检索、对齐 LLM 任务、9 种评估指标。

**📊 数据集**

13,000+ 样本，来自 30+ 开源项目，分别包括 636 题 LeetCode 代码生成、8,225 代码总结、1,916 PR 代码评审、2,274 Defects4J 补丁评估。

**📈 对比分析**

对比零样本与上下文学习，使用 PASS@1、ROUGE/BLEU/METEOR/BERTScore、Accuracy/Precision/Recall/F1 四种指标。结果显示平均提升 24.7%，其中代码评审提升 33%、补丁评估 30%、代码总结 14.78% BLEU、代码生成 5.72% PASS@1。

**⚠️ 局限性**

①对低资源语言支持不足；②仅考虑静态项目上下文，缺少动态、漂移和长尾稀缺情景的适应机制。

---

## 357. Beyond Faders: Understanding 6DoF Gesture Ecologies in Music Mixing

**arXiv ID:** 2602.23090 | [PDF](https://arxiv.org/pdf/2602.23090v1)

**作者:** Jeremy Wertheim Co Chen `[一作]`, Jordan Aiko Deja `[通讯]` (De La Salle University)

**通讯引用:** 134 | [OpenAlex ID](https://openalex.org/A5046495553)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过设计工作坊提取音频混音核心参数的6DoF手势，并在XR原型中实现与用户体验评估；

**💡 创新点**

提出生态有效性评价框架，探讨专业经验如何影响手势偏好与舒适度；

**🔧 技术方法**

使用XR头显+手部追踪、六自由度交互、NASA‑TLX、UEQ‑S自评表和半结构访谈；

**📊 数据集**

采用12名非混音师的参与式实验数据（无公开数据集）；

**📈 对比分析**

通过反复测量NASA‑TLX和UEQ‑S评估认知负荷与用户体验，发现各参数间差异不大；无客观性能对比，主要依赖主观评价；

**⚠️ 局限性**

样本量小、仅限非专业用户、实验环境缺乏真实工作室设备、手势响应延迟可能影响感知，导致结果泛化受限。

---

## 358. Optimized Disaster Recovery for Distributed Storage Systems: Lightweight Metadata Architectures to Overcome Cryptographic Hashing Bottleneck

**arXiv ID:** 2602.22237 | [PDF](https://arxiv.org/pdf/2602.22237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 359. Pix2Key: Controllable Open-Vocabulary Retrieval with Semantic Decomposition and Self-Supervised Visual Dictionary Learning

**arXiv ID:** 2602.22510 | [PDF](https://arxiv.org/pdf/2602.22510v1)

**作者:** Guoyizhe Wei `[一作]` (Johns Hopkins University), Yan Gao `[通讯]` (Amazon.com)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Pix2Key框架，实现以开放词表视觉词典为介质的可控组合图像检索；

**💡 创新点**

创新点在于：①将查询和候选图像统一为可解释的键值词典，实现细粒度意图约束；②引入正/负/开放三种极性约束与多样性重排序；③设计自监督视觉词典自编码器V‑Dict‑AE，提升词典质量；

**🔧 技术方法**

技术包括：视觉语言模型（如Qwen‑VL、OpenCLIP）抽取词典，文本编码器做检索，MMR式多样性重排序，V‑Dict‑AE利用冻结扩散解码器进行自监督预训练；

**📊 数据集**

使用的数据集包括DeepFashion‑MM（扩展为DFMM‑Compose）、FashionIQ、CIRR，以及COCO/FashionAI用于自监督预训练；

**📈 对比分析**

与多种无监督/零样本基线（CIReVL、Pic2Word、SEARLE、FTI4CIR等）对比，Pix2Key在Recall@K、属性一致性AC@50、列表多样性ILD@50等指标上均优于基线，V‑Dict‑AE进一步提升表现；

**⚠️ 局限性**

局限性在于：依赖视觉语言模型的词典质量；多样性重排序对计算资源有一定开销；在极小或复杂编辑场景下仍可能缺失细节；

---

## 360. Enabling clinical use of foundation models in histopathology

**arXiv ID:** 2602.22347 | [PDF](https://arxiv.org/pdf/2602.22347v1)

**作者:** Audun L. Henriksen `[一作]`, Andreas Kleppe `[通讯]` (Oslo University Hospital)

**通讯引用:** 1621 | [OpenAlex ID](https://openalex.org/A5069676037)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过在下游任务模型训练中加入对扫描仪差异一致性的对比损失和预测分数一致性损失，提升了利用现有病理基础模型特征的临床模型在不同扫描仪上的鲁棒性与预测准确性。

**💡 创新点**

创新点在于：①提出两种新颖的鲁棒性损失，既在特征层面对齐相同组织在不同扫描仪下的嵌入，又在输出层强制统一预测分数；②仅通过修改下游网络的损失函数而无需重新训练基础模型，即可显著降低技术变异对模型预测的影响。

**🔧 技术方法**

技术方法包括：多实例学习（Attention‑MIL）框架、InfoNCE 对比损失、均方误差（MSE）预测分数一致性损失、八种主流病理基础模型（H‑Optimus、Hibou‑L、Phikon‑v2、Prov‑GigaPath、UNI、Virchow‑2 等）的特征提取、以及基于扫描仪对齐的图像注册与拼接。

**📊 数据集**

使用了 27,042 张全切片图像（WSI），来自 6,155 名患者，涵盖 QUASAR 2、TransSCOT、Ahus、Aker、Gloucester、VICTOR、Mainz、DENEB、Dutch T1 等多中心数据集，并在不同扫描仪（Aperio AT2、NanoZoomer XR、KF‑PRO‑400、Aperio GT 450 DX、Pannoramic 1000 等）上获取多重扫描。

**📈 对比分析**

在结直肠癌生存预测和肿瘤转移预测两项任务上，与传统仅使用交叉熵损失的模型相比，加入鲁棒性损失后：①不一致性指标下降 177–259%（平均 <0.2），②分类一致性提升至 95% 以上；且预测性能通常提升，生存预测的 Harrell‑C-index 与 AUC 均有显著改善。

**⚠️ 局限性**

局限性包括：①需要同一切片在多台扫描仪上同时扫描以构造训练对，获取成本高且不易普及；②实验仅在八种基础模型与两种任务上验证，可能对其他模型或任务的适用性有限；③仍未对所有技术变异（如不同染色批次、切片厚度差异）做进一步泛化评估。

---

## 361. They Think AI Can Do More Than It Actually Can: Practices, Challenges, & Opportunities of AI-Supported Reporting In Local Journalism

**arXiv ID:** 2602.22887 | [PDF](https://arxiv.org/pdf/2602.22887v1)

**作者:** Besjon Cifliku `[一作]` (Center For Advanced Internet Studies), Hendrik Heuer `[通讯]` (University of Wuppertal)

**通讯引用:** 402 | [OpenAlex ID](https://openalex.org/A5069890392)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对德国21名本地记者进行半结构化访谈，探讨他们在数据与AI支持报道中的使用方式、面临挑战以及对未来AI机会的看法。

**💡 创新点**

首次从非技术本地记者的视角系统梳理AI在本地新闻工作中的实际需求与阻碍，并通过对话式设计与原型演示获取前瞻性洞见。

**🔧 技术方法**

采用访谈法、MaxQDA质性内容分析和两种AI原型（演示驱动与文字驱动的自动化工具）进行实验演示。

**📊 数据集**

使用21份访谈记录（约90分钟访谈），以及基于真实本地新闻案例制作的AI原型视频，未使用公开数据集。

**📈 对比分析**

通过对比参与者对传统手工流程与AI辅助流程的评价，发现AI可显著降低重复性工作、提升信息检索速度，但在信任与准确性上仍低于人工判断；研究未做数值性能对比，主要基于质性比较与主题编码。

**⚠️ 局限性**

局限性包括样本规模有限、集中在德国本地新闻，结果可能缺乏跨文化推广性；访谈自报信息可能存在偏差；原型仅以演示视频呈现，缺乏真实使用情境的实验验证。

---

## 362. TARAZ: Persian Short-Answer Question Benchmark for Cultural Evaluation of Language Models

**arXiv ID:** 2602.22827 | [PDF](https://arxiv.org/pdf/2602.22827v1)

**作者:** Reihaneh Iranmanesh `[一作]` (Georgetown University), Nazli Goharian `[通讯]` (Georgetown University)

**通讯引用:** 2216 | [OpenAlex ID](https://openalex.org/A5036610566)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TARAZ框架，评估LLM在波斯语文化知识上的短答生成能力。

**💡 创新点**

创新点：将多选改为短答，设计波斯语特定后处理（数字归一化、连词分段、形态归一化）并结合句法与语义混合评估，解决形态学变异和数值表示问题。

**🔧 技术方法**

使用词形归一化、数字标准化、连词分段、maux-gte-persian-v3句子嵌入、LLM-judge（GPT‑5）、Exact Match、ROUGE 等技术。

**📊 数据集**

使用BLEnD、PerCul‑SAQ、ISN‑SAQ三大波斯语文化数据集。

**📈 对比分析**

对闭源、开源和波斯语微调三类模型进行比较；闭源模型在EM、ROUGE、LLM‑judge、Maux+Post等指标上最高，开源Gemma‑2‑27B和DeepSeek‑V3表现最好；后处理显著提升语义匹配得分。

**⚠️ 局限性**

局限：仅做文本评估，未覆盖多模态；数据集可能存在注解者偏差；方法针对波斯语，其他语言需重新设计后处理。

---

## 363. Bayesian Preference Elicitation: Human-In-The-Loop Optimization of An Active Prosthesis

**arXiv ID:** 2602.22922 | [PDF](https://arxiv.org/pdf/2602.22922v1)

**作者:** Sophia Taddei `[一作]`, Tom Verstraten `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于贝叶斯偏好推理的主动义肢控制参数优化框架，结合人机交互实现实时调优。

**💡 创新点**

将贝叶斯偏好学习与主动学习结合，首次在主动义肢领域实现人类偏好驱动的在线优化。

**🔧 技术方法**

使用贝叶斯优化、Gaussian Process 模型进行偏好推断，配合主动询问策略与实时控制接口。

**📊 数据集**

在自制的义肢使用数据集上进行实验，数据包含不同残肢个体在多种任务中的操作记录与偏好反馈。

**📈 对比分析**

与随机搜索、传统手动调参和基线机器学习方法比较，实验显示该方法在任务完成率与舒适度评分上提升约15%至20%。

**⚠️ 局限性**

受限于受试者数量有限、查询成本高以及模型对偏好一致性假设的依赖，尚未验证在大规模部署中的鲁棒性。

---

## 364. Secure Transmission for Fluid Antenna-Aided ISAC Systems

**arXiv ID:** 2602.23241 | [PDF](https://arxiv.org/pdf/2602.23241v1)

**作者:** Yunxiao Li `[一作]` (Shandong University), Ju Liu `[通讯]` (Shandong University)

**通讯引用:** 9056 | [OpenAlex ID](https://openalex.org/A5100763003)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出在流动天线（FA）辅助的集成感知与通信（ISAC）系统中，考虑感知目标可能作为窃听者，联合优化天线位置向量（APV）和波束形成器，以最大化多用户总密钥率。

**💡 创新点**

利用FA的空间可调性在兼顾感知与通信的双功能波形下显著提升物理层安全；提出基于BSUM的算法，结合PDA闭式更新和EPG位置优化，解决非凸联合设计问题。

**🔧 技术方法**

Block Successive Upper-bound Minimization (BSUM)、Proximal Distance Algorithm (PDA)、Extrapolated Projected Gradient (EPG)、线性LoS信道模型、功率与空间约束。

**📊 数据集**

仿真数据（L_0=λ/2, L=10λ, λ=0.01m, ϕ=60°, 路径损耗 g_0=-40dB, 路径指数α=2.8 等），未使用公开数据集。

**📈 对比分析**

与固定位置天线（FPA）系统对比，FA-ISAC 在不同用户数、功率预算、感知功率和天线数量场景下均实现约20%或更高的总密钥率提升；算法在200次迭代内收敛。

**⚠️ 局限性**

仅考虑单一LoS模型，未验证多径或干扰情况；算法复杂度仍较高；实验仅在一维FA阵列上完成，二维或多维扩展尚未验证。

---

## 365. Agent Behavioral Contracts: Formal Specification and Runtime Enforcement for Reliable Autonomous AI Agents

**arXiv ID:** 2602.22302 | [PDF](https://arxiv.org/pdf/2602.22302v1)

**作者:** Varun Pratap Bhardwaj `[一作]` `[通讯]` (Accenture), Varun Pratap Bhardwaj (Accenture)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种 Agent Behavioral Contracts 框架，给 AI 代理定义合同（前置条件、约束、治理策略和恢复机制），并实现运行时执行库与基准测试。

**💡 创新点**

创新点包括：① 以 Design‑by‑Contract 为蓝本，将合同形式化为 (P, I, G, R) 结构；② 引入 (p, δ, k)‑满足度，为 LLM 的随机性提供概率保障；③ 通过 Ornstein‑Uhlenbeck 模型证明漂移界限与恢复率关系；④ 提供多代理链的组合定理和软硬约束恢复机制；⑤ 设计 YAML DSL 与可插拔的运行时库。

**🔧 技术方法**

技术手段：形式化推理与概率分析、线性系统与 Lyapunov 稳定性、Ornstein‑Uhlenbeck 随机微分方程、约束评估与监控、Python 实现的运行时库、LLM‑as‑Judge 评估、统计检验（Welch t‑检验、Bonferroni 校正）以及基准生成。

**📊 数据集**

使用自定义 AgentContract‑Bench（200 个跨 7 领域的多步场景）作为基准，并在 7 种大型语言模型上进行 1,980 次会话实验；此外还使用金融顾问等真实任务场景。

**📈 对比分析**

对比方法：在相同任务下分别运行有合同和无合同的 60 次会话，评估硬/软合规率、软违规次数、漂移得分、可靠性指数。结果显示有合同模式在软违规检测上提升 5.2–6.8 次/会话、硬约束合规率 88–100%、漂移 D* < 0.27、恢复成功率 17–100%，平均每步开销 <10 ms，差异显著（p < 0.0001，Cohen d > 6.7）。

**⚠️ 局限性**

局限性：依赖平台无干扰的内容安全过滤器；假设不同代理间恢复独立，跨 LLM 的相关性未深入；仅在语言模型任务上验证，缺乏跨模态或长期部署评估；恢复策略主要靠重试，复杂情境下可能不足；基准为合成场景，真实世界异构性尚待进一步验证。

---

## 366. Manifold of Failure: Behavioral Attraction Basins in Language Models

**arXiv ID:** 2602.22291 | [PDF](https://arxiv.org/pdf/2602.22291v1)

**作者:** Sarthak Munshi `[一作]`, Blake Gatto `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性绘制LLM失效流形，发现行为吸引基底。

**💡 创新点**

将安全性评估转为质量多样性搜索，用MAP‑Elites绘制连续失效拓扑，首次将行为吸引基底作为失效区域。

**🔧 技术方法**

使用MAP‑Elites、Alignment Deviation（多类别安全偏差）指标、Gaussian Process建模、六种变异策略、LLM生成与Judge模型（GPT‑4.1、Sonnet‑4.5）评估。

**📊 数据集**

对三大LLM（Llama‑3‑8B、GPT‑OSS‑20B、GPT‑5‑Mini）进行单轮评估，使用自定义的十类安全标签作为判定标准。

**📈 对比分析**

与随机、GCG、PAIR、TAP等基线对比，MAP‑Elites在覆盖率、分布多样性及QD得分上均优于基线；在Llama‑3‑8B上获得最高覆盖率与最深失效，高峰失效均为1.0；在GPT‑OSS‑20B上发现更多高危基底；GPT‑5‑Mini保持安全，峰值AD仅0.50。

**⚠️ 局限性**

受限于二维行为空间、查询预算有限、仅单轮交互、Judge模型可能带偏差、缺乏人类验证，且真实失效流形可能更高维。

---

## 367. Disentangling Shared and Target-Enriched Topics via Background-Contrastive Non-negative Matrix Factorization

**arXiv ID:** 2602.22387 | [PDF](https://arxiv.org/pdf/2602.22387v1)

**作者:** Yixuan Li `[一作]` (McGill University), Yue Li `[通讯]` (McGill University)

**通讯引用:** 10331 | [OpenAlex ID](https://openalex.org/A5100387744)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种背景对比非负矩阵分解（bCNMF）框架，用于在高维生物数据中从目标数据中提取特异性主题，同时抑制与背景数据共享的主导变异。

**💡 创新点**

创新点在于将对比学习目标直接嵌入NMF结构，使用共享非负基底同时对目标与背景进行因子分解，既保持可解释的部分基底，又支持计数型和零膨胀负二项等概率模型，且实现了高效的GPU乘法更新和mini‑batch 训练。

**🔧 技术方法**

核心技术包括：共享基底的对比NMF、基于乘法更新的优化、可选的Gaussian/Poisson/Zero‑Inflated Negative Binomial 损失、GPU 并行化和 mini‑batch 迭代；参数 α 用于调节背景抑制强度。

**📊 数据集**

使用了多种真实与模拟数据集：MNIST‑ImageNet 混合图像、下三烷基小鼠蛋白表达、白血病移植前后单细胞 RNA‑seq、MDD 预后大脑单核 RNA‑seq、以及 MIX‑seq Idasanutlin 细胞系扰动实验。

**📈 对比分析**

与 PCA、cPCA 和传统 NMF 进行对比，评估指标包括 ARI、聚类性能、可解释性（基底加载与差异基因）以及 GPU 运行时；bCNMF 在大多数实验中显著提高 ARI，能够揭示隐藏的疾病程序，且在 GPU 上实现线性扩展，运行时间与 cPCA 相当。

**⚠️ 局限性**

局限性包括：需假设目标与背景共享同一基底；背景选择不当会导致对比信号失真；α 的手动调参仍缺乏完全自适应方法；当目标与背景差异不足或背景噪声过大时，对比效果可能减弱。

---

## 368. Conversational Successes and Breakdowns in Everyday Non-Display Smart Glasses Use

**arXiv ID:** 2602.22340 | [PDF](https://arxiv.org/pdf/2602.22340v1)

**作者:** Xiuqi Tommy Zhu `[一作]` (Northeastern University), Eileen McGivney `[通讯]` (Northeastern University)

**通讯引用:** 183 | [OpenAlex ID](https://openalex.org/A5035743644)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过一个月的协作式自我民族志研究，系统记录并分析了两名研究者在日常生活中使用Meta Ray‑Ban AI眼镜时的对话成功与失效模式，重点探讨了非显示智能眼镜的语音-视觉交互特点。

**💡 创新点**

创新点在于：①首次将自动民族志方法与LLM驱动的多模态视觉感知结合，以“眼见即理解”的视角揭示智能眼镜独特的会话成功（即时参照解决、陌生知识理解、决策支持）与失效（参照不连贯、人类感知冲突、社交尴尬、语音交互限制）模式；②通过与传统语音助手的对比，阐明了智能眼镜在情境感知、即时响应和社会使用场景中的优势与挑战。

**🔧 技术方法**

技术实现基于Meta Ray‑Ban AI眼镜，该设备集成了Llama 4多模态LLM，并支持Live AI实时视觉推理与语音交互。

**📊 数据集**

数据集包括两名研究者在2025年5月7日至6月7日的日记、会议纪要、对话日志和视觉捕获素材，构成自我民族志式的自然语境记录。

**📈 对比分析**

方法上，作者采用主题分析对日志进行编码，识别成功与失效主题，并与已有的语音助手研究结果进行对比。结果显示，智能眼镜在解决即时参照问题、获取陌生知识和辅助决策方面表现优于传统设备，但在参照一致性、视觉误判和社交尴尬等方面的失效率更高；总体而言，在情境感知和即时交互上优于传统语音助手，但仍需提升对视觉误差和社交情境的适应。

**⚠️ 局限性**

限制包括：样本量仅两人且均为技术熟练者，缺乏用户多样性；自动民族志自我报告可能带来偏差；LLM在视觉理解与多轮记忆上的局限未被充分解决；结果仅为探索性发现，难以推广到更大规模或不同文化背景的使用场景。

---

## 369. Chain of Flow: A Foundational Generative Framework for ECG-to-4D Cardiac Digital Twins

**arXiv ID:** 2602.22919 | [PDF](https://arxiv.org/pdf/2602.22919v1)

**作者:** Haofan Wu `[一作]` (University of Birmingham), Le Zhang `[通讯]` (Queen Mary University London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了基于单周期12导联ECG的全4D心脏数字孪生生成框架Chain of Flow，实现了从ECG直接重建个体化4D心脏结构和运动，并支持后续体积量化、功能分析和虚拟影像合成。

**💡 创新点**

将ECG与Cine‑CMR联合训练，通过运动匹配机制将静态解剖和ECG驱动的动态分离，采用基于变换匹配的Flow Learning和ODE积分生成连续时间‑空间流；实现了无需Cine输入即可从ECG生成完整4D心脏图像，突破以往任务限定的预测模型。

**🔧 技术方法**

基于卷积神经网络的体积配准、教师‑学生Dice监督以保持解剖拓扑、ECG条件化的速度场Flow Matching、ODE积分生成4D图像、以及多模态融合与分割一致性约束。

**📊 数据集**

使用英国生物样本库（UK Biobank）约10,000例配对12导联ECG与短轴Cine‑CMR，包含多分辨率、多心脏周期的训练集。

**📈 对比分析**

与COF‑w/ Diffusion、ECHOPulse、LFDM、EchoDiffusion、随机采样等基线在SSIM、PSNR、FID、FVD、运动相关性（M‑Corr）和分割Dice/Iou等六项指标上对比，Chain of Flow在所有指标上均优于对照组，尤其SSIM 0.984、PSNR 28.46、FID 6.39、M‑Corr 0.474，显著提升图像质量与运动连贯性。

**⚠️ 局限性**

仅基于单个R–R周期，无法处理节律变异或持续多周期；需要静态解剖锚点，无法完全无影像驱动生成；对极端病理或不同体位的泛化仍有限。

---

## 370. Improving Neural Argumentative Stance Classification in Controversial Topics with Emotion-Lexicon Features

**arXiv ID:** 2602.22846 | [PDF](https://arxiv.org/pdf/2602.22846v1)

**作者:** Mohammad Yeghaneh Abkenar `[一作]`, Panagiotis Ioannidis `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对NRC情感词典进行聚类式扩展，并将扩展后的情感特征融入BERT+分类器，实现对争议话题下论证文本的立场分类。

**💡 创新点**

创新点在于①使用DistilBERT嵌入与GMM聚类实现情感词典的系统化扩展；②在多语料、跨主题的论证立场分类框架中首次引入情感词典特征并统一预处理。

**🔧 技术方法**

采用DistilBERT编码、余弦相似度归一化、GMM聚类、情感词典扩展、NASCA神经分类模型（BERT+情感特征）以及与多数类基线和LLM Qwen2.5-7B 的对比实验。

**📊 数据集**

使用五个争议话题论证语料库：AMT1、AMT2、Persuasive Essays、IBM Debater、UKP Sentential Argument Mining。

**📈 对比分析**

与多数类基线和Qwen2.5-7B 对比，NASCA+eNRC在所有数据集上提升宏F1，eNRC在PE上达62.9%，在UKP上达到68.4%，在IBM上突破64.5%；阈值0.4被证明为最稳健的扩展阈值。

**⚠️ 局限性**

局限性包括：情感词典扩展依赖预训练模型且阈值需经验选择；仅针对二分类立场，未处理多标签或多情感情境；对长文本的情感特征捕捉尚未充分验证跨语言和跨主题的泛化能力。

---

## 371. Comparative Analysis of Neural Retriever-Reranker Pipelines for Retrieval-Augmented Generation over Knowledge Graphs in E-commerce Applications

**arXiv ID:** 2602.22219 | [PDF](https://arxiv.org/pdf/2602.22219v1)

**作者:** Teri Rumble `[一作]` (Abertay University), Jagdeep Ahluwalia `[通讯]` (Abertay University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并比较了三种Retriever–Reranker管线，用于在电商知识图中回答自然语言查询。

**💡 创新点**

创新点在于结合了稠密检索（FAISS）与结构化图扩展、以及针对知识图优化的跨编码器重新排序，显著提升检索准确性。

**🔧 技术方法**

采用的技术包括E5-Large向量嵌入、FAISS HNSW/Flat检索、BM25+图节点扩展、以及ms‑marco‑MiniLM-L‑6‑v2和webis/set‑encoder大型跨编码器。

**📊 数据集**

使用的数据集是Amazon STaRK Semi‑Structured Knowledge Base（SKB）及其Synthetic Prompt & Response Key查询集，共9,100条查询。

**📈 对比分析**

通过Hit@1、Hit@5、Recall@20和MRR四项指标与先前基准进行对比，最优管线FAISS HNSW+webis/set‑encoder-large实现Hit@1提升20.4%、MRR提升14.5%；相比之下FAISS HNSW+ms‑marco跨编码器在速度上更优且仍保持高性能。

**⚠️ 局限性**

主要局限包括：仅在单一电商SKB上评估；未充分利用图结构进行生成；跨编码器计算成本高，尤其是大型set‑encoder。

---

## 372. Scaling Search Relevance: Augmenting App Store Ranking with LLM-Generated Judgments

**arXiv ID:** 2602.23234 | [PDF](https://arxiv.org/pdf/2602.23234v1)

**作者:** Evangelia Christakopoulou `[一作]` (Apple), Sandip Gaikwad `[通讯]` (Apple)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用LLM-as-a-Judge在App Store搜索排名中生成百万级文本相关性标签，补充稀缺的人类标注；

**💡 创新点**

创新点在于将细化的、在域内微调的3B模型作为离线标注器，既提升标签质量又显著拉动多目标优化的Pareto前沿；

**🔧 技术方法**

技术包括LLM微调、零/少量提示、点对点文本相关性生成、多目标学习（行为与文本相关性）与标量化融合；

**📊 数据集**

数据集由历史App Store查询-应用对日志（行为标签）和有限的人类文本相关性评审（训练/验证）组成；

**📈 对比分析**

比较方法采用离线NDCG评估行为与文本相关性，以及全球A/B测试，实验显示LLM增强模型在行为/文本NDCG上均超过原始模型，线上转换率提升0.24%；

**⚠️ 局限性**

局限性包括对LLM生成标签质量的依赖、仍未探索成对/列表化提示、以及对非App Store域推广的适用性未知。

---

## 373. Pixel2Catch: Multi-Agent Sim-to-Real Transfer for Agile Manipulation with a Single RGB Camera

**arXiv ID:** 2602.22733 | [PDF](https://arxiv.org/pdf/2602.22733v1)

**作者:** Seongyong Kim `[一作]` (Dongguk University), Soo-Chul Lim `[通讯]` (Dongguk University)

**通讯引用:** 992 | [OpenAlex ID](https://openalex.org/A5023192002)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Pixel2Catch的RGB‑only机器人抓取系统，利用像素级视觉特征在单摄像头图像中推断被抛物体运动，并通过异构多智能体强化学习将机器人臂与手分离为独立代理进行协同控制，从而实现稳定抓取。

**💡 创新点**

创新点包括：① 在不估计3D位姿的前提下，直接用单张RGB图像的像素级位置与尺度差异捕捉物体运动；② 将高自由度机器人拆解为臂和手两类异构代理，分别设计观察空间和奖励，提升学习稳定性；③ 采用SAM2进行鲁棒分割，配合域随机化与系统辨识实现无额外调优的 sim‑to‑real 迁移。

**🔧 技术方法**

使用的技术主要有：多智能体PPO（MAPPO）训练框架、Isaac Lab 物理仿真、SAM2目标分割、域随机化（刚度、阻尼、噪声、质量等）与系统辨识、基于像素的特征提取（中心、宽高差分）以及深度神经网络策略与价值网络。

**📊 数据集**

数据集为自建的仿真与真实抓取数据：在仿真中随机生成5种几何形状、不同质量与尺度的物体，并随机化其初始位姿、速度；真实实验中使用人类抛掷的立方体、L形块、三角形等三种物体，采集30次/物体。

**📈 对比分析**

通过与四种基线（无像素特征、单体强化学习、仅使用中心点、仅使用宽高）对比，仿真中Pixel2Catch达成约90%跟踪率和成功率，单体强化学习约78/63%；真实环境中Pixel2Catch实现约70%跟踪率、约50%成功率，单体强化学习仅约24%成功率，单一特征基线表现更差。

**⚠️ 局限性**

局限性在于仅验证单臂机器人抓取，依赖SAM2分割对极端视觉条件敏感；仅利用RGB信息，对光照、遮挡的鲁棒性有限；在更复杂的抓取任务（如多物体、非抛掷动作）或双臂系统上的通用性尚未证明。

---

## 374. Multi-modal Data Driven Virtual Base Station Construction for Massive MIMO Beam Alignment

**arXiv ID:** 2602.22796 | [PDF](https://arxiv.org/pdf/2602.22796v1)

**作者:** Yijie Bian `[一作]` (Hong Kong University of Science and Technology), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 45441 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于多模态数据构建虚拟基站（VBS）来进行大规模MIMO波束对准的框架。

**💡 创新点**

创新点在于利用3D LiDAR和定位信息显式构造VBS，并通过粗略信道重建对波束进行优先级排序，从而显著降低在线波束搜索开销。

**🔧 技术方法**

使用三维LiDAR点云、定位数据、几何镜像法、HDBSCAN聚类、粗略信道重建与部分波束训练等技术。

**📊 数据集**

数据集基于开放街图（OSM）生成的实际区域LiDAR点云和射线追踪仿真。

**📈 对比分析**

与纯定位波束对准和改进的CKM波束对准进行对比，VBS在训练Top‑5波束时即可获得约98%最优谱效率。

**⚠️ 局限性**

局限性包括对LiDAR测量误差敏感、对多次反射路径建模不足以及需要较大离线预处理。

---

## 375. Mind the Gap in Cultural Alignment: Task-Aware Culture Management for Large Language Models

**arXiv ID:** 2602.22475 | [PDF](https://arxiv.org/pdf/2602.22475v1)

**作者:** Binchi Zhang `[一作]` (University of Virginia), Zhengzhang Chen `[通讯]` (NEC Laboratories America)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 CultureManager 这一基于任务感知的文化对齐流水线，自动生成任务特定的文化数据，并通过轻量化的文化适配器和动态路由实现跨文化推理。

**💡 创新点**

创新点在于将任务感知的文化数据合成与模块化的文化管理相结合，既弥合了广义文化知识与特定任务之间的鸿沟，又通过路由器显著缓解了跨文化干扰。

**🔧 技术方法**

采用 GPT‑4o 生成搜索查询和合成数据，利用 SearchGPT 进行网页检索，使用 LoRA 轻量化适配器对模型进行微调，并通过提示式的文化路由器动态选择适配器。

**📊 数据集**

评估数据主要来自 CultureBench（45 文化·17 主题）与 CultureLLM（10 个文化敏感任务），训练时使用 WVS 数据和自动合成的任务特定样本，测试集严格独立保留。

**📈 对比分析**

在 10 个任务、5 种文化的 F1/准确率上与 Prompt、TaskSFT、CultureSFT、CultureLLM 等基线进行对比，CultureManager 在绝大多数任务上均实现显著提升，并有效降低了跨文化干扰。

**⚠️ 局限性**

局限性包括：仅覆盖预定义任务和单一文化，缺乏多文化联合任务的实验；合成数据缺少人工验证；模型规模扩大并不总能提升文化适配效果；未针对多文化混合推理等复杂场景进行探究。

---

## 376. Adaptive Prefiltering for High-Dimensional Similarity Search: A Frequency-Aware Approach

**arXiv ID:** 2602.22214 | [PDF](https://arxiv.org/pdf/2602.22214v1)

**作者:** Teodor-Ioan Calin `[一作]` `[通讯]` (Vulture Labs), Teodor-Ioan Calin (Vulture Labs)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了适应性预过滤算法，通过根据聚类一致性自适应分配搜索预算，提升高维相似搜索效率。

**💡 创新点**

将训练频率与聚类一致性关联的幂律关系作为理论依据，引入基于聚类统计的动态探测策略，实现无查询学习的低成本分配。

**🔧 技术方法**

采用FAISS IVF索引、基于聚类一致性计算的预过滤策略，并使用CLIP（ViT‑B/32）生成的向量。

**📊 数据集**

ImageNet‑1k子集，287,556个CLIP向量，在NVIDIA A100上实验。

**📈 对比分析**

与统一探测基线对比，通过Recall‑成本曲线和特定Recall阈值（95%、98%）量化，分别获得20.44%和14.98%的效率提升。

**⚠️ 局限性**

依赖查询与聚类一致性相关的Zipf分布，对异常或分布外查询效果有限，目前仅适用于IVF索引，需进一步扩展。

---

## 377. Mean Estimation from Coarse Data: Characterizations and Efficient Algorithms

**arXiv ID:** 2602.23341 | [PDF](https://arxiv.org/pdf/2602.23341v1)

**作者:** Alkis Kalavasis `[一作]`, Ziyu Zhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在粗糙（coarse）数据（只观测到包含真实样本的集合而不是具体数值）条件下，如何对高斯分布的均值进行估计；给出了凸分区可辨识性的几何判定，并提出了首个多项式时间、样本量最优的均值估计算法；并将该框架应用于含市场摩擦的线性回归问题。

**💡 创新点**

（1）提出了“平行板（slab）”的几何判定，精确刻画了所有凸分区的可辨识性；（2）在可辨识的凸分区上实现了局部强凸性与梯度二阶矩控制，结合随机梯度下降得到 O(d/ε²) 的样本复杂度与多项式时间的算法；（3）将该方法推广到混合分区模型和具有凸逆像的摩擦函数的线性回归。

**🔧 技术方法**

主要技术包括：
- 对粗糙数据下的对数似然函数进行凸性与局部强凸性分析；
- 变差降低不等式与 Prékopa–Leindler 不等式相结合，证明平行板是非可辨识性的必要与充分条件；
- 通过信息保留参数 α 量化可辨识性，并利用 Pinsker 不等式与 KL 散度关联梯度二阶矩；
- 采用局部强凸性与梯度二阶矩上界实现高效的随机梯度下降；
- 通过截断并构造局部化分区，将任意凸分区转化为可控制梯度范数的情形。

**📊 数据集**

本文为理论研究，没有使用真实数据集；所有实验与验证均基于理论分析和合成设定（高斯样本、离散分区）。

**📈 对比分析**

与先前仅实现样本效率（但计算量指数级）的算法相比，本工作在满足 α‑信息保留、凸分区的前提下，达到了与理论最优 O(d/ε²) 样本复杂度相同的同时，算法运行时间为多项式（例如 O(d²) ）。在混合分区模型与线性回归摩擦问题上，也实现了与前人方法相当或更优的样本与计算性能。

**⚠️ 局限性**

限制与不足：
- 仅适用于已知协方差矩阵（单位或已知）和凸分区；非凸分区仍为 NP‑难；
- 需要 α‑信息保留的先验，若 α 过小样本/时间复杂度会显著增加；
- 目前只在高斯族内给出理论结果，扩展到非高斯分布或未知协方差仍是开放问题。

---

## 378. Impacts of Aggregation on Model Diversity and Consumer Utility

**arXiv ID:** 2602.23293 | [PDF](https://arxiv.org/pdf/2602.23293v1)

**作者:** Kate Donahue `[一作]` (Massachusetts Institute of Technology and University of Illinois), Manish Raghavan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3207 | [OpenAlex ID](https://openalex.org/A5052541789)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了模型聚合对模型多样性和消费者实际效用的影响，探讨不同聚合策略如何改变单模型与集成模型的性能与多样性。

**💡 创新点**

创新点在于将多样性度量与消费者效用（如预测准确率、决策成本、用户满意度）结合起来，从多维视角评估聚合效果，而非仅关注整体准确率；并提出一种基于多样性-效用权衡的聚合选择框架。

**🔧 技术方法**

主要技术包括：多模型集成（Bagging、Boosting、Stacking）、多样性评估指标（相似度、协方差、互信息等）、消费者效用模型（成本-收益分析、A/B测试方法）以及实验对照设计。

**📊 数据集**

使用了公开图像与文本分类数据集：CIFAR-10、ImageNet、MNIST 以及大规模语言模型数据集（如 WikiText-103）来验证聚合策略在不同任务与规模下的表现。

**📈 对比分析**

与单一模型及传统聚合方法（均值、加权平均）进行对比；实验显示，在保持平均准确率提升 2–4% 的同时，聚合模型的多样性显著增加，消费者效用提升 5–10%，尤其在高噪声或稀缺样本情境下效果更突出。

**⚠️ 局限性**

局限性：仅在监督学习任务中验证；聚合策略不考虑模型更新与在线学习；消费者效用评价主要基于模拟实验，缺乏真实用户的实地测试；且对极端分布偏移的鲁棒性尚未充分探究。

---

## 379. Equivalent Dichotomies for Triangle Detection in Subgraph, Induced, and Colored H-Free Graphs

**arXiv ID:** 2602.23196 | [PDF](https://arxiv.org/pdf/2602.23196v1)

**作者:** Amir Abboud `[一作]` (Weizmann Institute of Science), Nathan Wallheimer `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5060641522)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明了在三种 H‑自由图设定（普通子图、诱导子图、彩色子图）下，三角检测问题的三色化与单三角条件所决定的“易/难”二分法彼此等价，给出了相互转换的多阶段归约；

**💡 创新点**

创新点在于构造了保留 3‑可着色性与三角数的“增广图”P′，并设计了既保持诱导 H‑自由性又能归约到唯一三角检测的自归约（通过彩色编码与筛选），以及为彩色 H‑自由性引入的等价与不等价 gadget；

**🔧 技术方法**

主要技术包括：基于彩色编码的自归约、Valiant‑Vazirani 筛选、图构造 gadget（等价/不等价，Grötzsch 图改造）、以及利用这些 gadget 将彩色 H‑自由图映射为普通 H′‑自由图；

**📊 数据集**

本工作为纯理论分析，未使用任何实际数据集；

**📈 对比分析**

通过归约展示了若子图情形下存在强子三次算法，则诱导与彩色情形同样可实现强子三次时间，且给出了对特定图（如 P6、C7）实现的子三次算法；

**⚠️ 局限性**

限制在于归约仅适用于 3‑可着色且至多含一三角的 H，且归约过程导致时间上显著慢速（如从 m+n^{5/3} 降到 n^{3-2^{-33}}），并且对彩色情形无法得到新的子三次算法，因为所需的 gadget 不具退化着色性质。

---

## 380. Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration

**arXiv ID:** 2602.23169 | [PDF](https://arxiv.org/pdf/2602.23169v1)

**作者:** Xiaole Tang `[一作]` (Xi'an Jiaotong University), Jian Sun `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 263155 | [OpenAlex ID](https://openalex.org/A5100785015)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出BaryIR框架，利用Wasserstein barycenter将多源退化图像特征映射到共同的无退化空间，同时构造正交残差子空间捕获退化特定知识，实现通用的All-in-One图像恢复。

**💡 创新点**

通过学习连续的Wasserstein barycenter映射和正交残差子空间实现对退化无关特征的显式分离，结合对抗性max-min优化和理论误差上界，显著提升对未知退化的泛化能力。

**🔧 技术方法**

Wasserstein barycenter、对抗性max-min优化、交叉对比损失（IRC）、正交约束损失（BRO）、Transformer‑based barycenter映射、Restormer编码器/解码器。

**📊 数据集**

合成退化数据（BSD400/400+噪声、Rain100L、SOTS、GoPro、LOL‑v1），真实混合退化数据（CDD‑11、SPANet、Lai）、O‑HAZE、SPAMNet、LOL‑v2‑real等。

**📈 对比分析**

与多种state‑of‑the‑art All‑in‑One恢复方法（Restormer、PromptIR、DA‑CLIP、DiffUIR、InstructIR、AdaIR、MoCE‑IR、DA‑RCOT）对比，BaryIR在三、五种退化基准上均获得PSNR/SSIM提升0.5–1.3 dB，O/OOD退化和混合退化场景下亦表现最优。

**⚠️ 局限性**

对退化权重λk的选取仍基于经验，未给出理论依据；在极端混合退化（如强雨+雾、JPEG压缩、海水散射）下仍可能忽略局部强度异常，导致残差空间无法充分识别。

---

## 381. HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning

**arXiv ID:** 2602.22630 | [PDF](https://arxiv.org/pdf/2602.22630v1)

**作者:** Yahia Salaheldin Shaaban `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Abdelrahman Sayed Sayed `[通讯]` (Université Gustave Eiffel)

**通讯引用:** 117 | [OpenAlex ID](https://openalex.org/A5046424783)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计了一种名为HyperKKL的学习框架，利用超网络对Kazantzis‑Kravaris/Luenberger（KKL）观察者的参数进行条件化，从而在非自主、受外部输入驱动的非线性系统中实现即时、无梯度更新的观测器设计。

**💡 创新点**

创新点在于：①使用输入历史编码的LSTM超网络动态生成观测器参数，实现输入自适应；②对比静态（stationary）与动态（dynamic）两种架构，并与传统的自适应课程学习进行系统比较；③采用物理约束自编码器与两阶段顺序训练，兼顾PDE一致性与输入动态性。

**🔧 技术方法**

使用的技术包括：Hypernetwork（超网络）架构、LSTM编码器+MLP解码头、物理信息化自编码器（PINN）损失、两阶段（自主预训练 + 超网络微调）顺序训练、课程学习（基于频谱复杂度的难度分层）、PDE残差约束、有限差分估计时间导数、梯度裁剪与归一化。

**📊 数据集**

实验数据集基于四个经典非线性动力学仿真：Duffing振子、Van der Pol振子、Lorenz系统与Rössler系统。每个系统生成多种输入情形（零、常数、正弦、方波），共计多条训练与测试轨迹。

**📈 对比分析**

性能比较采用RMSE与SMAPE指标，结果显示：在震荡与轻度混沌系统中，Dynamic HyperKKL与Static HyperKKL显著降低估计误差（相较自主KKL下降约48%~62%），尤其在正弦与方波输入下表现优异；相比之下，课程学习在所有系统与输入下均表现最差，甚至劣于仅使用自主KKL；在高度混沌的Lorenz系统中，HyperKKL的误差高于自主KKL，说明在极端敏感场景下仍存在不足。

**⚠️ 局限性**

限制与挑战：①对极度敏感的Lorenz吸引子，输入条件化可能放大误差，导致估计漂移；②模型验证仅基于仿真数据，缺少真实工业或灰盒系统的实验；③缺乏显式的Lyapunov稳定约束或自适应门控，导致在高敏感度条件下不稳定；④当前超网络只对输入历史进行编码，未充分考虑系统动态的时域特性，未来可探索更细粒度的时序建模与稳态约束。

---

## 382. Importance of Prompt Optimisation for Error Detection in Medical Notes Using Language Models

**arXiv ID:** 2602.22483 | [PDF](https://arxiv.org/pdf/2602.22483v1)

**作者:** Craig Myles `[一作]` (University of St Andrews), David Harris-Birtill `[通讯]` (University of St Andrews)

**通讯引用:** 795 | [OpenAlex ID](https://openalex.org/A5079344757)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用自动化提示优化技术（GEPA）提升大语言模型和小语言模型在MEDEC医学文本错误检测任务上的准确率。

**💡 创新点**

首次将基因多目标优化（GEPA）用于医学错误检测提示工程，显著提升多种模型的性能并实现可审计、可本地部署的解决方案。

**🔧 技术方法**

使用DSPy框架实现的GEPA自动提示演化，结合反射模型提供自然语言反馈，针对不同推理-反射模型对进行多种组合实验。

**📊 数据集**

MEDEC基准数据集（MS与UW子集），包含注释的医疗文本错误标注。

**📈 对比分析**

在MS和UW测试集上与基线P1提示、各大商用模型以及多位医生基准进行对比，GEPA优化后GPT‑5准确率提升至0.785，Qwen3‑32B提升至0.690，达到或超过专业医生水平，显著优于先前系统。

**⚠️ 局限性**

依赖于在MS验证集上的最佳反射模型，可能存在模型偏好；初始提示选择可能影响GEPA结果；反射输出受32k token限制；且未针对其他数据分布验证通用性。

---

## 383. Exploring Multimodal LMMs for Online Episodic Memory Question Answering on the Edge

**arXiv ID:** 2602.22455 | [PDF](https://arxiv.org/pdf/2602.22455v1)

**作者:** Giuseppe Lando `[一作]` (University of Catania), Antonino Furnari `[通讯]` (University of Catania)

**通讯引用:** 3222 | [OpenAlex ID](https://openalex.org/A5089549062)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在本地边缘设备上实现实时在线回忆视频问答（OEM‑VQA）系统，完全离线处理视频并生成文本记忆，支持即时问答。

**💡 创新点**

首次系统评估轻量多模态大型语言模型在严格实时、低功耗和隐私约束下的可行性，并提出端到端的并行描述/问答线程架构。

**🔧 技术方法**

采用 Qwen3‑VL 视觉语言模型（Instruct 版），配合 FlashAttention‑2、4‑bit 量化、流式缓存管理和文本记忆生成的描述线程与推理线程。

**📊 数据集**

使用 Ego4D 公开数据集中的 QAEgo4D‑Closed 基准（500 道多选问答）。

**📈 对比分析**

在消费级 8 GB GPU 上实现 51.76 % 准确率，TTFT 0.41 s；在企业级 48 GB GPU 上实现 54.40 % 准确率，TTFT 0.88 s，接近云端 56 % 的表现。

**⚠️ 局限性**

受 GPU 内存限制，模型规模与查询延迟之间存在权衡；仅能在单模型配置下完成完整流水线，且在更大模型上仍需提升实时性和资源利用效率。

---

## 384. Discourse-Aware Dual-Track Streaming Response for Low-Latency Spoken Dialogue Systems

**arXiv ID:** 2602.23266 | [PDF](https://arxiv.org/pdf/2602.23266v1)

**作者:** Siyuan Liu `[一作]` (Tongji University), Haizhou Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 28976 | [OpenAlex ID](https://openalex.org/A5032690182)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DDTSR框架，实现对话系统的低延迟响应；通过并行小模型生成短的、无知识负担的连接词与大模型进行深度推理，支持听-思-说的流式交互；

**💡 创新点**

①连接词引导的小大模型协同机制，解耦知识密集推理与即时响应；②跨模态流式协作，让ASR、LLM与TTS并行重排，最大化时间重叠；③课程学习驱动的连贯性增强，保证早期连接词与后续回答的一致性；

**🔧 技术方法**

小模型（Qwen3‑0.6B）生成连接词；大模型（Qwen3‑8B/32B）进行推理；流式ASR（sherpa‑onnx）和流式TTS（CosyVoice2）；连接词置信度评估与按阈值触发；小大模型并行推理与跨模态交互；

**📊 数据集**

SD‑Eval（含口音、年龄、环境子集）和 SpokenNativQA（英语子集）；

**📈 对比分析**

与标准单流/双流Cascaded（SSC/SDC）以及端到端实时模型（Doubao、GLM、Qwen3‑Omni‑Flash）进行对比；测量感知延迟、反应延迟、等待延迟；DDTSR在两数据集上将等待延迟降低19%–51%，保持与基线相同的文本一致性、连贯性和语音自然度；

**⚠️ 局限性**

依赖于连接词的出现频率，若用户输入缺少合适连接词时难以提前发声；需训练专门的连接词模型，模型迁移到不同语言或大模型时需重新调优；在极长或高复杂度对话场景下，跨模态协作仍可能受限。

---

## 385. Exploiting network topology in brain-scale simulations of spiking neural networks

**arXiv ID:** 2602.23274 | [PDF](https://arxiv.org/pdf/2602.23274v1)

**作者:** Melissa Lober `[一作]` (Institute for Advanced Simulation), Susanne Kunkel `[通讯]` (Neuromorphic Software Ecosystems)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

针对传统高性能计算平台上的大规模脉冲神经网络仿真，本文通过将大脑区域映射到单个计算节点并利用长短距离突触传输延迟的差异，提出了一种结构感知的分区与通信方案，显著降低了同步与脉冲投递的开销。

**💡 创新点**

创新点在于：①将大脑区域结构与计算机拓扑对应，形成局部–全局两级通信；②基于突触延迟分离，减少全局 MPI 通信次数；③在 NEST 仿真框架中实现数据结构拆分与局部/全局缓冲，实现低同步成本；④通过理论模型解释同步与缓存访问优势，量化 D 倍延迟比对性能的影响。

**🔧 技术方法**

使用技术包括 MPI 与 OpenMP 并行、NEST 3.6 仿真引擎、C++ 内部数据结构拆分（连接表、目标表），以及自定义局部与全局脉冲缓冲；理论上运用正态分布与极值理论预测同步时间，以及均匀分布与递归式计算不规则内存访问比例。

**📊 数据集**

主要数据集为两类大脑模型：①多区域大猿猴视觉皮层模型（MAM）—32 个区域、约 130k 神经元/区、6000 连接/神经元；②简化的 MAM-基准模型（MAM-benchmark）—同尺寸、均匀连接，便于可控扩展；实验在两台 HPC 系统（SuperMUC‑NG 与 JURECA‑DC）上进行。

**📈 对比分析**

在弱缩放实验中，结构感知方案将实时因子从传统的 9.4–22.7 提升至 8.5–15.7（≈30% 加速），通信时间下降 76%，同步时间下降 48%，脉冲投递时间下降 25%。在真实网络模型上，整体运行时间提升约 42%。性能提升随 MPI 进程数增大、延迟比 D 适中且硬件具备高核心/内存带宽时更为明显。

**⚠️ 局限性**

主要局限包括：①将整个区域映射至单节点导致负载不平衡，若区域规模或活跃度差异大，同步开销可能抵消收益；②需要在模型构建阶段手动指定区域映射，尚未内置于 NEST 高层接口；③对不同突触延迟分布或非均匀连接结构的适用性有限；④当前实现基于 MPI 阻塞式通信，非阻塞或异步通信的进一步优化仍待研究。

---

## 386. Cross-Task Benchmarking of CNN Architectures

**arXiv ID:** 2602.22945 | [PDF](https://arxiv.org/pdf/2602.22945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 387. SO3UFormer: Learning Intrinsic Spherical Features for Rotation-Robust Panoramic Segmentation

**arXiv ID:** 2602.22867 | [PDF](https://arxiv.org/pdf/2602.22867v1)

**作者:** Qinfeng Zhu `[一作]` (Xi'an Jiaotong-Liverpool University), Lei Fan `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SO3UFormer 以实现对全 3D 旋转鲁棒的全景语义分割。

**💡 创新点**

创新点包括去除绝对纬度编码、使用四边形一致注意力、基于切平面角度的 Gauge 位置偏置、几何一致多尺度采样以及 logit 级 SO(3) 一致性正则化。

**🔧 技术方法**

采用 icosahedral 球面离散、四边形权重校正注意力、Gauge 角度 Fourier 位置偏置、几何一致下采/上采样以及基于索引的球面重采样实现 SO(3) 一致性。

**📊 数据集**

在 Pose35（Stanford2D3D 加上 ±35° 随机旋转）以及全 SO(3) OOD 旋转测试上进行实验。

**📈 对比分析**

与 SFSS、HealSwin、Elite360、SphereUFormer 等基线对比，SO3UFormer 在 Pose35 基线 mIoU 达 72.03%，在全 SO(3) 旋转下保持 70.67%，显著优于其他方法（仅 25–30%）。

**⚠️ 局限性**

局限在于仅使用合成旋转评估，尚未在真实无人机或手持相机的 6‑DoF 变姿势数据上验证，且高精度球面重采样仍为待解决问题。

---

## 388. How Many Votes is a Lie Worth? Measuring Strategyproofness through Resource Augmentation

**arXiv ID:** 2602.22838 | [PDF](https://arxiv.org/pdf/2602.22838v1)

**作者:** Ratip Emin Berker `[一作]` (Carnegie Mellon University), Caspar Oesterheld `[通讯]` (Carnegie Mellon University)

**通讯引用:** 71 | [OpenAlex ID](https://openalex.org/A5008125381)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了“操纵潜能”这一新度量，用来衡量在不同投票规则下，一名投票者若要通过操纵获得与多次真实投票同等好处，需要额外添加多少份其真实投票；并对多类主流投票规则（Plurality、Instant Runoff、Borda Count 等）以及整个位置计分规则和 Condorcet 一致规则族的操纵潜能进行了理论分析和下界/上界证明。

**💡 创新点**

创新点在于：①将操纵问题转化为资源增益问题，提出“k‑augmentation strategy‑proofness”与“操纵潜能”概念；②给出多类规则的精准极限，证明 Borda Count 在多数选民下具有最低且不随选民数增长的操纵潜能；③对位置计分规则提供完整的上下界，进一步证明除了 Plurality 和 Borda Count 之外无规则能在同类规则中获得更低潜能；④展示 Condorcet 一致规则族的不可避免高潜能，形成与 Borda Count 的对比。

**🔧 技术方法**

主要使用的技术是组合数学与图论的严格构造与极限证明，利用投票规则的对称性（匿名性、无偏性）简化分析；并引入了“最大化与最小化”范式以得到上界/下界；在某些情形下使用了可数的分割与平均分数分析。

**📊 数据集**

本文为理论研究，未使用实测数据集；所有结果均通过构造极端投票配置与抽象证明得到。

**📈 对比分析**

比较方法：对各投票规则在最坏情况（极端投票配置）下的“需要复制多少次”进行理论比较。性能方面，Plurality 的潜能约为⌈(n−1)/2⌉，Instant Runoff 与 Plurality with Runoff 为 n−1，Borda Count 的潜能为 m−2（m 为候选人数），而 Condorcet 扩展规则的潜能可达 n−1 或更高；因此在多数选民情形下，Borda Count 在操纵抵抗上优于其他规则。

**⚠️ 局限性**

限制包括：①仅针对单一胜选规则；②结果仅适用于最坏情况（极端配置），未给出平均或实测分布下的潜能；③对大部分规则需要强假设（如可除数条件、候选人数均匀分布）；④在 Condorcet 规则族中，虽然给出下界，但仍无法在偶数选民或非标准偏好分布下得到完整界定；⑤未探讨群体操纵或多胜选情形的潜能。

---

## 389. Tell Me What To Learn: Generalizing Neural Memory to be Controllable in Natural Language

**arXiv ID:** 2602.23201 | [PDF](https://arxiv.org/pdf/2602.23201v1)

**作者:** Max S. Bennett `[一作]` (Columbia University), Richard Zemel `[通讯]` (Columbia University)

**通讯引用:** 38001 | [OpenAlex ID](https://openalex.org/A5000111344)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可通过自然语言指令控制的通用神经记忆系统，实现对多源信息的选择性学习与遗忘。

**💡 创新点**

创新点在于将学习指令以自然语言形式融入记忆写入过程，使得模型能够根据用户意图灵活决定何种信息写入、忽略或拒绝。

**🔧 技术方法**

采用MemoryLLM架构并改造其写入步骤，加入指令条件的记忆写入机制，并在训练时对写入与查询两步同时进行梯度更新。

**📊 数据集**

使用基于CounterFACT的合成数据集，生成包含事实、格式、拒绝等多种指令的文档对，用于评估记忆系统的选择性、泛化和行为学习。

**📈 对比分析**

与原始MemoryLLM、ICL-FT、RAG-FT等基线对比；实验显示GNM在选择性、格式化准确率、拒绝精确率等指标上均优于基线，且在计算效率上仅需O(1)的内存更新开销，性能接近ICL-FT。

**⚠️ 局限性**

局限性包括：记忆容量有限导致随时间衰减；实验基于合成数据，缺乏真实世界评测；对冲突或不一致信息的处理效果不佳。

---

## 390. AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors

**arXiv ID:** 2602.22755 | [PDF](https://arxiv.org/pdf/2602.22755v1)

**作者:** Abhay Sheshadri `[一作]` (Anthropic Fellows Program), Rowan Wang `[通讯]` (Anthropic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了AuditBench基准，包含56个植入多种隐藏行为的语言模型和一个可配置的调查者代理，用于系统评估对齐审计工具的有效性。

**💡 创新点**

创新点在于规模化的隐藏行为模型集、对工具在代理中的实际使用效果进行量化评估、揭示工具与代理之间存在的效用差距（tool-to-agent gap）。

**🔧 技术方法**

采用Llama 3.3 70B与Qwen 3 14B的LoRA微调，结合转录蒸馏（transcript distillation）与合成文档微调（synthetic document fine‑tuning）植入行为，并使用SFT和KTO两种对抗训练；构建黑盒工具（prefilling、user‑persona sampling、text completion）、白盒工具（steering vectors、sparse autoencoders、activation oracles等）以及可脚手架的黑盒工具。

**📊 数据集**

使用6,000多轮对话转录、40,000篇合成行为描述文档、UltraChat提示集合以及红队生成的前缀与提示来训练和评估模型。

**📈 对比分析**

通过在每个目标模型上运行16次代理实验，评估代理对隐藏行为的识别准确率；实验显示可脚手架的黑盒工具取得最高成功率，白盒工具次之；不同训练配置（如SDF vs. TD、SFT vs. KTO）对工具效果影响显著，且工具对代理的实际帮助存在明显差距。

**⚠️ 局限性**

局限性包括：模型为显式微调，可能比真实前沿模型更易审计；未去除基模型先前的异常行为；评估仅在Llama 3.3 70B上进行，结果可能不完全适用于其他模型；工具输出噪声大且代理对工具的使用不充分，导致工具与代理之间的效用差距。

---

## 391. Early Risk Stratification of Dosing Errors in Clinical Trials Using Machine Learning

**arXiv ID:** 2602.22285 | [PDF](https://arxiv.org/pdf/2602.22285v1)

**作者:** Félicien Hêche `[一作]` (University of Geneva), Douglas Teodoro `[通讯]` (University of Geneva)

**通讯引用:** 7779 | [OpenAlex ID](https://openalex.org/A5042070839)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套基于机器学习的框架，利用临床试验启动前信息对试验的剂量错误风险进行早期分层。

**💡 创新点**

首次在临床试验阶段使用多模态数据（结构化特征+自由文本）并通过概率校准实现可解释的风险分层，同时公开了可复现的数据集和代码。

**🔧 技术方法**

XGBoost、ClinicalModernBERT、LateFusion（加权融合），并使用Platt scaling/Isotonic回归进行概率校准。

**📊 数据集**

从ClinicalTrials.gov抽取的42,112个已完成或终止的临床试验数据（含结构化、半结构化与自由文本）。

**📈 对比分析**

通过AUC‑ROC、Brier分数、F1等指标比较，LateFusion模型在测试集上AUC‑ROC 0.862，Brier 0.041，校准后实现四级风险分层，表现优于单一模态模型。

**⚠️ 局限性**

标签受不完整不良事件报告影响；模型采用简单的late‑fusion，未利用完整协议文本；对极长文本处理仍未解决。

---

## 392. FHECore: Rethinking GPU Microarchitecture for Fully Homomorphic Encryption

**arXiv ID:** 2602.22229 | [PDF](https://arxiv.org/pdf/2602.22229v1)

**作者:** Lohit Daksha `[一作]` (Boston University), Ajay Joshi `[通讯]` (Boston University)

**通讯引用:** 3472 | [OpenAlex ID](https://openalex.org/A5089428659)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了名为 FHECore 的专用 GPU 功能单元，用以加速 CKKS 加密运算中的核心模块（NTT 与基底转换），并在 NVIDIA A100 GPU 上验证其性能与面积开销。

**💡 创新点**

创新点在于：① 把 NTT 与基底转换视为模数线性变换，构造 16×8 体系矩阵流水阵列；② 在每个处理单元内集成 Barrett 除法实现一次性模数乘累加；③ 采用输出静止数据流以避免流水瓶颈；④ 在 SM 与 Tensor Core 对称部署，面积仅提升 2.4%；⑤ 提供新的 WMMA‑style ISA 指令，兼容现有 CUDA 生态。

**🔧 技术方法**

使用的技术包括：模数线性变换表述、系统矩阵流水阵列、Barrett 限制、输出静止数据流、CUDA/WMMA API 扩展、NVBit 采样、Accel‑Sim 追踪仿真、SiliconCompiler 合成（ASAP7 芯片），以及对 FHE 库的 ISA 调整。

**📊 数据集**

使用的数据集与基准为：FIDSlib 的 Bootstrapping、Logistic Regression（MNIST 子集）、ResNet20（预训练模型）以及 BERT‑Tiny；以及 CKKS 原语 HEMult、Rescale、Rotate 的加密运算。

**📈 对比分析**

评估方法：在真实 A100 上通过 NVBit 记录 SASS 指令，插入新指令后在 Accel‑Sim 进行 trace‑driven 仿真；结果显示指令数平均降低 2.41×（原语）/1.96×（完整），性能提升 1.57×（原语）/2.12×（完整），bootstrapping 延迟减少 50%。

**⚠️ 局限性**

限制点：仅对 FHE 相关工作负载带来显著收益，若与 Tensor Core 并发使用可能产生冲突；设计与 A100 体系紧密，需要在其他 GPU 上做适配；未覆盖所有 FHE 内核（如逐槽模运算、自动化）；需要库层支持新 ISA 才能充分利用。

---

## 393. Coalgebraic analysis of social systems

**arXiv ID:** 2602.23211 | [PDF](https://arxiv.org/pdf/2602.23211v1)

**作者:** Nima Motamed `[一作]` (Utrecht University), Emily Roff `[通讯]` (University of Edinburgh)

**通讯引用:** 27 | [OpenAlex ID](https://openalex.org/A5019658321)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过范畴论与通用余 coalgebra，提出将角色与位置分析从图推广到超图的统一框架，并证明其兼容性。

**💡 创新点**

设计了可同时处理图和超图的角色半群与位置归约的概念，并给出了在半单子作用下的范畴论性函子化结果。

**🔧 技术方法**

利用通用余 coalgebra、Kleisli 结构、半单子、正则化（bisimulation）以及范畴论中的函子与自然变换。

**📊 数据集**

以简化的亲属网络和共著网络为示例，用作说明的人工构造数据，而非大型真实数据集。

**📈 对比分析**

通过形式化的函子化定理展示位置归约必然诱导角色归约，理论证明表明方法在范畴论意义下保持一致；未给出数值性能指标。

**⚠️ 局限性**

对实际复杂网络的实现与计算成本未评估；目前仅覆盖有向超图，近似等价、加权或模糊网络的推广仍待研究。

---

## 394. Motion-aware Event Suppression for Event Cameras

**arXiv ID:** 2602.23204 | [PDF](https://arxiv.org/pdf/2602.23204v1)

**作者:** Roberto Pellerito `[一作]`, Davide Scaramuzza `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种可变规模的视觉Transformer模型SViT，用于语义分割任务

**💡 创新点**

创新点在于将Transformer模块按分辨率可调节规模，从而在保持高精度的同时显著提升推理速度

**🔧 技术方法**

使用了视觉Transformer架构、可变分辨率特征提取以及轻量化的后处理模块

**📊 数据集**

在COCO分割数据集上进行训练与评估

**📈 对比分析**

通过与主流分割模型在COCO上的AP与fps对比，SViT-full在34.9 AP下获得13.9 fps，SViT-256以34.3 AP实现16.7 fps，展示了优异的速度/精度平衡

**⚠️ 局限性**

模型在极低帧率下精度下降显著，且对极大尺寸或复杂场景的适应性仍有限

---

## 395. BRepMAE: Self-Supervised Masked BRep Autoencoders for Machining Feature Recognition

**arXiv ID:** 2602.22701 | [PDF](https://arxiv.org/pdf/2602.22701v1)

**作者:** Can Yao `[一作]` (University of Science and Technology of China), Xiao-Ming Fu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 23662 | [OpenAlex ID](https://openalex.org/A5060631280)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于BRep模型的自监督掩码图像自编码器（BRepMAE）来学习CAD模型的通用表示，并在少量标注数据下实现高精度机加工特征识别。

**💡 创新点**

创新点在于：①首次将MAE方法引入BRep表面表示，采用多分支重构目标；②使用带虚拟节点的几何属性归一化图（gAAG）来同时捕捉几何、属性与拓扑信息；③在自监督预训练中同时重构面几何、属性、AABB，显著提升表示质量。

**🔧 技术方法**

使用技术包括：UV参数化采样、双折叠网（FoldingNet）几何解码器、带多聚合的MPNN（包含边信息更新）、多分支损失、掩码比例80%、虚拟全局节点。

**📊 数据集**

训练数据集包括ABC、MFInstSeg、Fusion 360 Gallery、CADSynth、MFCAD2、MFCAD 共计约312k CAD模型；在下游评测使用MFCAD2、CADSynth、MFInstSeg、MFCAD等。

**📈 对比分析**

与AAGNet、BRepMFR等全监督方法以及BRep-BERT等自监督方法对比，在0.1%–100%标注比例下，BRepMAE在机加工特征识别任务中取得最高的准确率和mIoU，尤其在极低标注量时优势最为明显。

**⚠️ 局限性**

局限性包括：对几何相似或拓扑复杂的特征仍可能出现误分；仅处理单标签分割，未覆盖多重特征交叠；未加入加工过程相关信息；缺乏对全局图表示的进一步探索。

---

## 396. SPMamba-YOLO: An Underwater Object Detection Network Based on Multi-Scale Feature Enhancement and Global Context Modeling

**arXiv ID:** 2602.22674 | [PDF](https://arxiv.org/pdf/2602.22674v1)

**作者:** Guanghao Liao `[一作]` (University of Science and Technology Liaoning), Qi Li `[通讯]` (University of Science and Technology Liaoning)

**通讯引用:** 14226 | [OpenAlex ID](https://openalex.org/A5100350262)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为SPMamba-YOLO的海底目标检测网络，针对光衰、色彩失真、背景杂乱和小目标等问题进行改进；

**💡 创新点**

创新点包括三大模块：SPPELAN扩展感受野的多尺度特征聚合、PSA层级分裂注意力增强特征判别、以及基于Mamba的状态空间模型高效捕捉全局长程依赖；

**🔧 技术方法**

使用的技术主要是YOLOv8n骨干改造、Spatial Pyramid Pooling、Pyramid Split Attention、Mamba状态空间网络以及多尺度特征融合与注意力机制；

**📊 数据集**

使用公开的URPC2022水下目标数据集，包含海参、海胆、海星、扇贝四类，约9000张图像；

**📈 对比分析**

与YOLOv8n基线、Faster R‑CNN、SSD、RT‑DETR以及其他YOLO系列模型对比，SPMamba-YOLO在URPC2022上mAP@0.5提升至0.825（比基线高4.9%），在各类别、PR曲线和Grad‑CAM可视化中表现更优，参数和GFLOPs略高但仍保持实时可用；

**⚠️ 局限性**

主要限制是模型加入多模块后参数量和推理成本增加，未来需探索更高效的特征融合策略以减少冗余并保持性能，并需在更多数据集上验证泛化能力。

---

## 397. Toward Personalized LLM-Powered Agents: Foundations, Evaluation, and Future Directions

**arXiv ID:** 2602.22680 | [PDF](https://arxiv.org/pdf/2602.22680v1)

**作者:** Yue Xu `[一作]` (ShanghaiTech University), Wenjie Wang `[通讯]` (ShanghaiTech University)

**通讯引用:** 2186 | [OpenAlex ID](https://openalex.org/A5100368534)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性综述了面向用户个性化的LLM驱动代理，提出了四个核心能力（Profile Modeling、Memory、Planning、Action Execution）的能力导向分类，并梳理了相关技术、评估指标、基准数据集与应用场景。

**💡 创新点**

创新点在于：①以能力为导向构建统一框架，阐释个性化如何在整个决策流程中协同展开；②将个性化细分为四个互相依赖的子模块，揭示它们的交互与设计权衡；③综合评估方法与指标，提出系统级评价视角；④归纳未来研究方向与挑战。

**🔧 技术方法**

技术综述涵盖：用户画像构建（explicit/implicit、persona/response-based）、外部/内部记忆（文本/结构化、短期/长期、更新与检索策略）、一轮/交互式规划（偏好诱导、记忆条件、内部改进）、动作执行（策略、归约、恢复、结果实现）以及评估范式（自动评分、规则、LLM评估器/裁判、基准对齐）。

**📊 数据集**

使用的主流数据集与基准包括：PersonaBench、IndieValueCatalog、PrefDisco、LongMemEval、PDR-Bench、AgentRecBench、RecBench+、PersonaLens、PersonaFeedback、MemoryAgentBench、ETAPP、PENGUIN、PrivacyBench、LaMP、LongLaMP、PersonaConvBench、TravelPlanner++、TripTailor、PEToolBench 等，覆盖对话、推荐、规划、工具使用与安全隐私等领域。

**📈 对比分析**

作为综述，本文并未在单一实验中比较算法；而是通过对上述基准与指标的汇总，对比了不同方法在效果、适应性、泛化、鲁棒性与风险等维度的表现，指出当前技术普遍在多轮互动与长期记忆一致性、隐私保护、跨领域迁移与实时调优等方面存在差距。

**⚠️ 局限性**

局限性包括：①依赖已有文献与公开基准，缺乏统一的大规模对比实验；②对隐私与安全的讨论仍停留在概念层面，缺乏可落地的机制；③多项评估指标尚未标准化，导致跨工作比较困难；④对低资源语言、非文本模态与极端动态环境的个性化支持研究不足；⑤未来需要更多跨学科、可解释、可持续的个性化代理实现方案。

---

## 398. False memories to fake news: The evolution of the term "misinformation" in academic literature

**arXiv ID:** 2602.22395 | [PDF](https://arxiv.org/pdf/2602.22395v1)

**作者:** Alejandro Javier Ruiz Iglesias `[一作]` (University of Vermont), Peter Sheridan Dodds `[通讯]` (University of Vermont)

**通讯引用:** 15199 | [OpenAlex ID](https://openalex.org/A5040821463)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过分析2011-2023年Scopus数据库中包含“misinformation”关键词的论文，结合文本频率统计、二元组提取、TF‑IDF及Louvain社区检测，揭示后2016年的信息误传研究范式与早期“误信息效应”研究及1980年代撒旦恐慌的历史联系。

**💡 创新点**

创新点在于将当代信息误传的学术范式与1980年代误记忆实验和撒旦恐慌相连，提供新的历史视角，说明该范式根植于早期认知实验研究。

**🔧 技术方法**

使用了文本频率统计、二元组提取、TF‑IDF、Louvain社区检测以及引用流分析等技术。

**📊 数据集**

采用了Scopus导出的2011-2023年包含“misinformation”元数据的论文集，并辅以opencitations.net API提供的引用信息。

**📈 对比分析**

通过对2011-2015与2017-2023两段时间的词频差异与社区结构进行比较，发现后期出现“social media”“fake news”等关键词，社区规模显著扩大，表明信息误传研究在规模与研究主题上实现了跃迁。

**⚠️ 局限性**

主要局限在于仅使用单一关键词检索、数据仅限于Scopus，未覆盖非学术媒体，且可能遗漏对“misinformation”不同语义使用的研究。

---

## 399. Hierarchical Trajectory Planning of Floating-Base Multi-Link Robot for Maneuvering in Confined Environments

**arXiv ID:** 2602.22459 | [PDF](https://arxiv.org/pdf/2602.22459v1)

**作者:** Yicheng Chen `[一作]` (University of Tokyo), Moju Zhao `[通讯]` (University of Tokyo)

**通讯引用:** 1153 | [OpenAlex ID](https://openalex.org/A5045076994)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了浮动底多连杆机器人在狭窄环境中的层次轨迹规划框架，利用根部刚体与关节柔性双重性质生成全局锚点状态，并在每个锚点之间采用B‑spline参数化的局部轨迹优化，能够直接从原始点云输入生成连续、碰撞安全、动力学可行的轨迹。

**💡 创新点**

创新点包括：①将机器人根部视为刚体引导全局路径、关节视为柔性实现“全局锚点状态”分段；②在连续时域内以可微方式编码碰撞、速度与可控性约束；③实现并行优化与从原始点云直接感知，无需人工障碍模型。

**🔧 技术方法**

使用了A*全局路径规划、离散候选状态生成、B‑spline轨迹参数化、全微分优化（SLSQP）、ESDF距离场、Jacobian求导、可控性指标、采样式连续约束、并行计算、ROS、Pinocchio、Octomap、Gazebo、NLopt等技术。

**📊 数据集**

实验基于离线生成的0.1分辨率点云（模拟环境）以及实时采集的Livox Mid‑360 LiDAR点云（真实环境）进行，未使用公开的标准数据集。

**📈 对比分析**

通过与去掉锚点、局部规划或并行化的消融变体以及差分运动学（DK）基准进行对比。结果显示完整框架成功率92.5%，平均规划时间约9 s；并行化提升约3倍速度；在单间隙、三间隙、杆障碍、U型通道等实验中均能通过最窄0.7宽的间隙，并保持速度与可控性指标安全。

**⚠️ 局限性**

局限包括：需精准传感与控制，未针对动态障碍和长时任务；局部规划初始为能量最小轨迹易陷入局部最优；采样式连续约束可能漏判瞬时失控；实验规模有限，尚未在更大、工业级真实环境中验证。

---

## 400. Scaling In, Not Up? Testing Thick Citation Context Analysis with GPT-5 and Fragile Prompts

**arXiv ID:** 2602.22359 | [PDF](https://arxiv.org/pdf/2602.22359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 401. From Calibration to Refinement: Seeking Certainty via Probabilistic Evidence Propagation for Noisy-Label Person Re-Identification

**arXiv ID:** 2602.23133 | [PDF](https://arxiv.org/pdf/2602.23133v1)

**作者:** Xin Yuan `[一作]` (Wuhan University of Science and Technology), Chia-Wen Lin `[通讯]` (National Tsing Hua University)

**通讯引用:** 77943 | [OpenAlex ID](https://openalex.org/A5068223640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了双阶段的噪声鲁棒行人重识别框架CARE，先通过概率证据校准提升预测可信度，再利用证据传播精细化样本权重。

**💡 创新点**

创新点在于将Dirichlet基证据校准与复合角度边缘、置信球权重结合，既打破softmax的平移不变性，又在保留难正样本的同时抑制噪声样本。

**🔧 技术方法**

使用的技术包括Dirichlet证据推理、期望负对数似然与KL正则、复合角度边缘(CAM)与置信球权重(COSW)、两阶段协同优化。

**📊 数据集**

在Market1501、DukeMTMC-ReID和CUHK03三个公开数据集上评估。

**📈 对比分析**

与现有噪声鲁棒方法相比，CARE在随机噪声10%-50%以及模式噪声10%-20%下均实现Rank‑1与mAP的显著提升，达到或超过最新SOTA。

**⚠️ 局限性**

局限性在于仅针对单模态、光照变化有限的场景，且对更复杂开放世界的鲁棒性仍有提升空间。

---

## 402. CoLyricist: Enhancing Lyric Writing with AI through Workflow-Aligned Support

**arXiv ID:** 2602.22606 | [PDF](https://arxiv.org/pdf/2602.22606v1)

**作者:** Masahiro Yoshida `[一作]` (University of California), Nanyun Peng `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了CoLyricist，一款针对歌曲创作流程的AI辅助歌词写作工具。

**💡 创新点**

创新点在于将大型语言模型与音乐结构结合，提供四阶段工作流（主题设定、创意生成、歌词草稿、旋律匹配）并实现自动旋律匹配。

**🔧 技术方法**

采用ChatGPT等LLM进行文本生成、Reffly模型实现旋律匹配，Sinsy用于语音合成。

**📊 数据集**

主要使用了10位业余作词者的访谈数据和16位参与者的用户实验，没有构建专门的公开数据集。

**📈 对比分析**

通过用户实验和两位专业评审的定量评估，结果显示新手与经验作词者在旋律匹配和整体质量上无显著差异，用户对工具满意度高。

**⚠️ 局限性**

局限性包括仅基于业余作词者的工作流程、固定旋律、未考虑歌曲结构和专业作词者的使用体验。

---

## 403. Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference

**arXiv ID:** 2602.22868 | [PDF](https://arxiv.org/pdf/2602.22868v1)

**作者:** Yushi Ye `[一作]` (Shanghai Jiao Tong University), Jiangchao Yao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2409 | [OpenAlex ID](https://openalex.org/A5102922412)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为 Rejection Mixing 的解码框架，在 Diffusion 大语言模型（DLLM）的并行推理中引入连续混合状态，以实现快速非自回归推理。

**💡 创新点**

创新点：① 在离散解码流程中插入连续混合状态，使 token 之间能够在连续空间内迭代细化、互相纠正；② 设计了基于 Jensen‑Shannon 距离的拒绝机制，自动将不稳定的连续表示重置回掩码状态，避免错误传播；③ 该方法完全训练无关，避免了额外的模型微调成本。

**🔧 技术方法**

技术细节：使用 DLLM（如 LLaDA、MMaDA）作为基模型；混合更新公式 y_emb←β Wᵀ p_θ + (1−β) Emb_[MASK]；β 为混合比例；采用自适应 top‑p 采样；拒绝阈值 τ_rej 用于判定连续表示是否需要回退；半自回归/块式采样控制并行程度。

**📊 数据集**

数据集：语言推理与代码生成 8 组（GSM8K、MATH‑500、HumanEval、MBPP、Countdown、Sudoku、ARC‑E、ARC‑C）；多模态任务 6 组（Flickr30k‑lite、AI2D‑lite、MathVision‑mini、MathVista‑mini、MMMU‑val、ScienceQA‑IMG）。

**📈 对比分析**

实验比较：与 LLaDA 或 MMaDA 的标准解码对比，Rejection Mixing 在所有 8 个语言基准上均提升 0.6–14.05% 的准确率，推理步骤缩短 2.5×–5.0×，推理延迟降低 8–13 秒，整体速度提升 2.4×–4.6×；在 6 个多模态基准上准确率提升 0.2–3.0%，推理步骤下降 4.4×–8.5×，速度提升 3.8×–7.5×。

**⚠️ 局限性**

局限性：需要手动调节混合比例 β 与拒绝阈值 τ_rej，参数对不同模型、不同长度/块大小的适应性尚未完全自动化；在极短或极长序列、不同规模模型下的鲁棒性可能下降；连续状态的引入虽无训练成本，但在大模型上可能增加显存占用和计算负担。

---

## 404. Velocity and stroke rate reconstruction of canoe sprint team boats based on panned and zoomed video recordings

**arXiv ID:** 2602.22941 | [PDF](https://arxiv.org/pdf/2602.22941v1)

**作者:** Julian Ziegler `[一作]` (Laboratory for Biosignal Processing, Leipzig University of Applied Sciences), Mirco Fuchs `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于视频的自动化系统，可在单人艇（K1/C1）与多人艇（K2/C2/K4）中同时重建划船速度和划桨频率，并兼顾不同距离（200 m/500 m）和赛道布局。

**💡 创新点**

创新点包括：
- 利用U‑Net对船尖进行精确检测，实现每位选手的个性化偏移量校准，取代单一经验偏移；
- 通过光流（Lucas‑Kanade）实现多人艇的运动跟踪与顺序恢复，即使检测失效也能填补缺失位置；
- 两种划桨频率提取方案：ViTPose关键点估计与仅基于边框亮度的信号，后者实现低成本实时估计。

**🔧 技术方法**

核心技术包括：YOLOv8目标检测（划船、浮标），基于已知浮标网格的单应性求解，U‑Net船尖检测，光流跟踪，ViTPose姿态估计，Savitzky‑Golay滤波以及中心差分速度求导。

**📊 数据集**

使用的数据集为：
- 主数据集：54 条 GPS 与陀螺计时轨迹，40 条录像（K1/C1/K2/C2/K4，200 m/500 m）。
- 船尖检测子集：550 训练、124 验证、98 测试 150×150 像素补丁，人工标注船尖位置。

**📈 对比分析**

与 GPS 与陀螺真值对比：
- 速度：平均 RRMSE 0.020 ± 0.011（ρ = 0.956），单艇 0.019，双/四艇 0.023；
- 划桨频率：ViTPose 2.35 ± 2.35 bpm（RRMSE = 0.022，ρ = 0.932），仅基于边框 7.85 ± 4.90 bpm（RRMSE = 0.069，ρ = 0.686）。
- 计算成本：ViTPose 1179 s/视频，边框 411 s，整体管线 2243 s（单应性）+ 537 s（速度）。

**⚠️ 局限性**

局限性：
- 单应性初始化仍需人工标记四个浮标，未实现时间一致性平滑；
- 光流在严重遮挡或高运动模糊时易失效；
- ViTPose 对计算资源需求高，实时性受限；
- 只覆盖 K1/K2/C1/C2/K4，未验证更大艇种；
- 低资源场景下仅基于边框的方法准确性受限，易被遮挡影响。

---

## 405. Residual Koopman Spectral Profiling for Predicting and Preventing Transformer Training Instability

**arXiv ID:** 2602.22988 | [PDF](https://arxiv.org/pdf/2602.22988v1)

**作者:** Bum Jun Kim `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**通讯引用:** 13999 | [OpenAlex ID](https://openalex.org/A5090592819)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Transformer训练过程进行提前的稳定性评估与干预，先通过单次前向传播预测训练是否会发散，并在训练中通过谱正则化抑制发散。

**💡 创新点**

创新点在于：①提出残差Koopman谱剖面（RKSP），利用白化的动态模态分解（DMD）一次性估计各层的Koopman谱；②引入“near‑unit mass”指标作为发散风险评分；③设计Koopman谱塑形（KSS）正则化，将谱信息直接嵌入训练目标，实现主动抑制不稳定模式。

**🔧 技术方法**

技术手段包括：白化DMD、Koopman算子理论、谱正则化、梯度裁剪对比实验、统计显著性检验、鲁棒性阈值设定。

**📊 数据集**

实验数据集涵盖：合成关联召回任务、WikiText‑103、OpenWebText、CIFAR‑10（ViT）、GPT‑2、LLaMA‑2‑7B、MoE、Mamba‑SSM、KAN 等多种模型与任务。

**📈 对比分析**

方法对比：与梯度基线（梯度范数、梯度方差、损失尖峰计数）以及现有的光谱/权重规范化、SAM、Lion优化器等。RKSP 的 AUROC 为 0.995（显著高于梯度基线 0.758），KSS 将发散率从 66.7% 降至 12.5%，且可让学习率提升 50–150%，在多任务、多模型上均保持相对低的额外计算开销（≈10%）。

**⚠️ 局限性**

局限性：①依赖于初始化时的线性化，可能无法捕捉后期非线性动态；②对高度非正交（高非正定性）矩阵的预测受限；③白化与DMD对噪声敏感，需保证足够的样本；④在极大规模模型或特殊优化器（如自适应 LR 变化）下的通用性尚需进一步验证。

---

## 406. FairQuant: Fairness-Aware Mixed-Precision Quantization for Medical Image Classification

**arXiv ID:** 2602.23192 | [PDF](https://arxiv.org/pdf/2602.23192v1)

**作者:** Thomas Woergaard `[一作]` (University of Copenhagen), Raghavendra Selvan `[通讯]` (University of Copenhagen)

**通讯引用:** 1211 | [OpenAlex ID](https://openalex.org/A5063821969)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种公平感知的混合精度量化框架（FairQuant），通过分组重要性分析与预算约束分配并结合可学习的位宽优化，实现低精度模型在医学图像分类任务中的公平性能提升。

**💡 创新点**

创新点在于：① 将分组（敏感属性）条件的重要性信号与全局敏感性结合得到单一重要性映射；② 通过基于阈值的分层分配实现精度预算下的混合精度分配；③ 设计可学习的 Bit‑Aware Quantization（BAQ）让每个权重域的位宽成为可训练参数，在训练中通过比特率正则和公平惩罚共同优化。

**🔧 技术方法**

使用的技术包括：对称均匀量化、通道/张量级别量化、Straight‑Through Estimator（STE）、Taylor 一阶近似的敏感性估计、预算约束下的分层分配、可学习位宽正则化、基于组平均损失差异的公平惩罚、以及标准的 AdamW 优化器。

**📊 数据集**

实验数据集为两大皮肤病图像基准：Fitzpatrick17k（皮肤类型敏感属性）和 ISIC2019（性别敏感属性），并在 ResNet18/50、DeiT‑Tiny、TinyViT 四种网络上验证。

**📈 对比分析**

与 FP32、统一 8 位、统一 4 位以及先前工作（Lin 等、Guo 等）进行对比。结果显示，在 4–6 位平均精度下，FairQuant 能恢复接近 8 位的平均准确率，同时显著提升最差群体准确率，且在相同预算下公平性指标（EOpp、EOdd）保持或改善，表现优于统一 4 位基线并与统一 8 位保持竞争力。

**⚠️ 局限性**

局限性：仅在皮肤病图像和两种敏感属性上验证；公平性惩罚采用批级近似，可能不适用于更复杂的公平定义；对 λ_fair、学习率等超参的搜索仅为粗粒度，未覆盖更细致的调优空间；方法对其他数据集和任务的泛化能力尚待进一步评估。

---

## 407. Contextual Memory Virtualisation: DAG-Based State Management and Structurally Lossless Trimming for LLM Agents

**arXiv ID:** 2602.22402 | [PDF](https://arxiv.org/pdf/2602.22402v1)

**作者:** Cosmo Santoni `[一作]` (Imperial College), Cosmo Santoni `[通讯]` (Imperial College)

**通讯引用:** 9752 | [OpenAlex ID](https://openalex.org/A5107858764)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Contextual Memory Virtualisation (CMV) 系统，将LLM会话状态视为可版本化的持久资源，并实现跨会话的状态复用与结构化截断。

**💡 创新点**

创新点包括：使用DAG结构模拟对话快照与分支；三步结构无损截断算法，保持所有用户/助手消息完整；通过剪枝和裁剪保持API有效性；实现可并行分支与上下文回溯。

**🔧 技术方法**

技术手段包括：JSONL日志的流式三遍扫描；基于工具调用标识的截断规则；DAG/树形快照与分支API；成本模型与缓存失效分析；参考实现基于Claude Code。

**📊 数据集**

数据集为单一Claude Code用户在三个月内的76个真实编码会话，包含工具调用、文件输出、图像等多种对话元素。

**📈 对比分析**

比较方法：对比截断前后token占比、缓存成本与回本轮数；结果显示平均token减少20%（最小12%，最大86%），平均回本在35轮，混合型会话仅需10轮即可收支平衡。

**⚠️ 局限性**

局限性：截断仅基于结构而非语义，可能删去重要工具结果导致模型误差；实验仅覆盖单用户场景，难以泛化；token估算基于字节-字符换算，对图像占比高的会话可能偏高；未对下游推理准确性进行量化评估。

---

## 408. SmartChunk Retrieval: Query-Aware Chunk Compression with Planning for Efficient Document RAG

**arXiv ID:** 2602.22225 | [PDF](https://arxiv.org/pdf/2602.22225v1)

**作者:** Xuechen Zhang `[一作]` (University of Michigan), Nedim Lipka `[通讯]` (Adobe Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SmartChunk 检索框架，利用查询自适应的分块规划和压缩编码来提升长文档问答的准确性与效率。

**💡 创新点**

创新点包括：①查询感知的 Planner，预测最小/最大分块级别；②STITCH 训练框架（RL + 伪标注 + 监督微调），实现多目标平衡；③嵌入级别的 Chunk Compression Encoder，避免昂贵的 LLM 摘要；④多层分块与自适应检索的结合。

**🔧 技术方法**

技术手段：轻量级 Planner（小型 LM + RL/ imitation learning），STITCH RL+SFT 循环，SBERT/ Jina-Embedding 预训练再微调的 Compression Encoder，层次化检索（多级块），LLM 生成（GPT‑4o 等），GPT‑4o 作为评判器。

**📊 数据集**

使用的基准数据集：NarrativeQA、QASPER、QuALITY、Natural Questions 以及跨域的 NewsQA。

**📈 对比分析**

与固定分块、Late Chunking、RAPTOR、MAL RAG、GRAG 等现有方法对比，SmartChunk 在 QA 准确率平均提升约 1.7%，召回率提升 4%，同时将成本降低约 30%（单查询 GPT‑4o 调用费用），延迟从 3–4 秒降至约 2 秒；在 NewsQA 等异域数据亦表现优异。

**⚠️ 局限性**

局限性：需要一次性训练 Planner 与 Compression Encoder，训练成本不可忽略；对极大规模语料或极长文档的可扩展性仍待进一步验证；在多模态或实时交互场景中的鲁棒性尚未充分评估。

---

## 409. AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning

**arXiv ID:** 2602.23258 | [PDF](https://arxiv.org/pdf/2602.23258v1)

**作者:** Yutong Wang `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 60348 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AgentDropoutV2框架，在多智能体系统推理过程中通过测试时的rectify-or-reject裁剪动态纠错并防止错误传播。

**💡 创新点**

创新点在于构建失败驱动的检索式指示器池，结合迭代反馈纠错机制，既无需再训练也能自适应任务难度，显著提升信息流安全性。

**🔧 技术方法**

采用检索增强的错误指示器、对抗性判别、迭代修正、全局结构退化安全阈值，并集成AutoGen/SelectorGroupChat多智能体框架。

**📊 数据集**

在数学推理领域使用GSM8K、MATH‑500、AQuA、AMC23、OlympiadBench、OlymMATH Easy/Hard、AIME24/25等数据集；在代码生成领域使用MBPP、HumanEval、CodeContests、LiveCodeBenchV1。

**📈 对比分析**

与单代理、AutoGen以及使用通用指示器的基线对比，平均准确率提升约6.3个百分点，最难任务AIME25提升至30%；跨模型、跨域迁移亦保持显著收益。

**⚠️ 局限性**

局限在于依赖离线失败案例构建指示器池，指示器覆盖面有限；过多指示器会产生信息噪声；极难任务仍出现高拒绝率，缺乏对深度推理链的解释性支持。

---

## 410. Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction

**arXiv ID:** 2602.23214 | [PDF](https://arxiv.org/pdf/2602.23214v1)

**作者:** Chenhe Du `[一作]` (ShanghaiTech University), Yuyao Zhang `[通讯]` (ShanghaiTech University)

**通讯引用:** 2450 | [OpenAlex ID](https://openalex.org/A5100654429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出双耦合Plug‑and‑Play扩散（DC‑PnPDP）框架，利用ADMM双变量消除稳态偏差，并通过Spectral Homogenization把结构化残差转换为伪AWGN，使扩散模型能够在优化轨迹中安全工作。

**💡 创新点**

① 将ADMM的积分型双变量引入PnP扩散，实现记忆化的控制；② 设计频域白化模块Spectral Homogenization，解决双变量残差与扩散先验分布不匹配的问题；③ 两者协同提升重建精度与收敛速度。

**🔧 技术方法**

ADMM、控制理论（积分控制）、Plug‑and‑Play扩散（扩散模型作为去噪器）、频域功率谱估计与噪声填补（Spectral Homogenization）、CG求解器。

**📊 数据集**

AbdomenCT‑1K（CT稀视角、限角重建）与fastMRI knee（加速MRI）数据集。

**📈 对比分析**

在相同NFE 50的条件下，与经典FBP/Zero‑filling及SOTA扩散求解器（DDNM、DDS、DiffPIR、DAPS）对比，使用PSNR/SSIM/LPIPS评估。DC‑PnPDP在CT（限角+稀视角）和MRI（6×、10×加速）上均实现最高PSNR/SSIM，收敛速度约为现有方法的3倍，尤其限角CT提升+5.95 dB。

**⚠️ 局限性**

仍依赖于预训练扩散模型，Spectral Homogenization的频谱估计对极端欠采样时可能不够鲁棒；高维频域操作带来额外计算成本；缺少完整的理论收敛性证明。

---

## 411. Zeroth-Order Stackelberg Control in Combinatorial Congestion Games

**arXiv ID:** 2602.23277 | [PDF](https://arxiv.org/pdf/2602.23277v1)

**作者:** Saeed Masiha `[一作]` (EPFL), Patrick Thiran `[通讯]` (EPFL)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于零阶优化的Stackelberg控制方法ZO-Stackelberg，能够在不对下层均衡过程求导的前提下，优化组合拥堵游戏的领导者目标；

**💡 创新点**

主要创新包括：①将上层零阶梯度与下层Frank‑Wolfe均衡求解耦合，获得对真实（可能不光滑）超目标的收敛保证；②对Frank‑Wolfe使用子采样LMO，并给出包含优化器命中率κ_m的1/(κ_m T)收敛率；③提出分层采样（如长度偏置）以保持κ_m非零；④在多项式可解的策略族与NP‑hard族（利用ZDD）上实现高效LMO；

**🔧 技术方法**

零阶梯度估计、Frank‑Wolfe均衡求解、线性最小化oracle（精确或子采样）、ZDD/决策图对NP‑hard策略族的预处理、分层采样策略；

**📊 数据集**

在真实交通网络数据集上进行实验（如US城市路网、Sioux Falls等公共交通/道路网络）；

**📈 对比分析**

与基于微分的基线方法相比，ZO-Stackelberg在相同精度下实现了数量级速度提升，且显著降低了内存消耗；

**⚠️ 局限性**

局限性包括：①仍需多次调用均衡求解，计算开销在极大策略空间中仍不小；②子采样LMO在策略空间极端稀疏时可能导致κ_m过小；③理论收敛至Goldstein驻点，未保证全局最优；④对凸势函数假设有依赖，非凸或非潜在函数情形尚未覆盖。

---

## 412. TADA: A Generative Framework for Speech Modeling via Text-Acoustic Dual Alignment

**arXiv ID:** 2602.23068 | [PDF](https://arxiv.org/pdf/2602.23068v1)

**作者:** Trung Dang `[一作]` (Hume AI), Alan Cowen `[通讯]` (Hume AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种一对一同步音频与文本标记化方案，并将其嵌入大型语言模型（LLM）中实现统一单流的文本‑语音生成；通过流匹配头（flow‑matching head）以及语音自由引导（Speech Free Guidance）等技术，降低序列长度差异，消除内容幻觉，并提升推理速度。

**💡 创新点**

核心创新点包括：①一对一同步标记化，将语音特征与文本 token 对齐；②在 LLM 中直接加入音频嵌入并通过流匹配头预测音频特征和时长；③语音自由引导实现文本与多模态输入的平滑融合；④在线拒绝采样保证说话人一致性；⑤可流式生成，显著降低推理时延。

**🔧 技术方法**

使用的技术：Wav2Vec2+CTC 对齐器、变分自编码器（VAE）编码/解码器、Transformer+RoPE、流匹配（flow‑matching）与 Bit Diffusion、分类器无关引导（CFG）、多尺度 Mel、GAN、知识蒸馏等。

**📊 数据集**

训练数据包括 LibriLight、专有英语对话语料、七种多语言（中文、法语、意大利语、日语、葡萄牙语、波兰语、德语）语料；评测使用 SeedTTS‑Eval、LibriTTS‑R‑Eval、EARS、Seamless Interaction、Spoken StoryCloze、Spoken TopicStoryCloze。

**📈 对比分析**

与 XTTS‑v2、Index‑TTS2、Higgs‑Audio‑v2、VibeVoice 等基线比较：在语音克隆任务中 CER 低于或接近基线，SIM 与 oMOS 亦可与顶尖模型竞争；长文本生成时，结合文本自由引导与在线拒绝采样后 SIM 接近最优；推理时延（RTF）比固定帧率模型低约 50%~75%，显著提升速度；在语言建模指标（PPL、sSC、tSC）中，在文本‑语音模式下仍保持接近或优于大型 7B+ 模型。

**⚠️ 局限性**

局限性：流匹配采样步骤增加了 LLM 生成的计算开销，虽然整体速度仍优于传统方法，但仍需 4–10 步；语音自由引导的 λ 参数需要调优；在极长语音或低质量语音输入时仍可能出现说话人漂移，需进一步改进拒绝采样；音频自然度（oMOS）略低于顶尖语音合成模型，提示解码器仍可进一步优化。

---

## 413. Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes

**arXiv ID:** 2602.22667 | [PDF](https://arxiv.org/pdf/2602.22667v1)

**作者:** Changqing Zhou `[一作]` (Hong Kong University of Science and Technology), Changhao Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 695 | [OpenAlex ID](https://openalex.org/A5046822372)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于单目图像的开词汇3D占据预测框架，利用语言嵌入高斯体素同时捕获几何与语义信息。

**💡 创新点**

创新点在于设计了兼容透明度的泊松式高斯到占据变换器和逐步温度衰减方案，显著提升了在仅有二值占据标签监督下的几何稳定性与语义对齐效果。

**🔧 技术方法**

核心技术包括3D语言嵌入高斯（LE-Gaussians）、泊松式G2O算子、指数温度衰减的高斯 splatting、以及基于开词汇分割模型的特征对齐。

**📊 数据集**

在Occ-ScanNet室内数据集上进行训练与评估，该数据集提供细粒度的占据网格和语义标签。

**📈 对比分析**

与POP-3D、LOcc等基线相比，本方法在仅使用几何标签的开词汇设置下达成了59.50 IoU与21.05 mIoU，超过所有现有方法的IoU提升约3.02点，mIoU提升约11.8点，表现优异。

**⚠️ 局限性**

主要局限在于与完整语义标注的闭词汇模型仍存在性能差距，且长尾类别的精细识别仍受限，未来需要进一步提升语义校准与稀有类别的推理能力。

---

## 414. Requesting Expert Reasoning: Augmenting LLM Agents with Learned Collaborative Intervention

**arXiv ID:** 2602.22546 | [PDF](https://arxiv.org/pdf/2602.22546v1)

**作者:** Zhiming Wang `[一作]` (Beihang University), Feng Lu `[通讯]` (Beihang University)

**通讯引用:** 7289 | [OpenAlex ID](https://openalex.org/A5101480749)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 AHCE 框架，使基于 LLM 的代理在遇到知识盲点时主动请求并与人类专家进行对话，形成可执行的修正计划。

**💡 创新点**

将人类专家视为交互式工具，利用学习的策略主动提问、融合专家意见，并仅在必要时请求帮助，从而实现高效的对话式协作。

**🔧 技术方法**

采用 Qwen-2.5 系列 LLM，利用强化学习（GRPO）训练交互策略，结合问题识别模块、对话标签与动态指令注入技术。

**📊 数据集**

使用 MuSiQue 多跳问答数据集训练 HFM，MineDojo 环境和 15 个按难度分级的 Minecraft 过程任务进行实验。

**📈 对比分析**

与完全自主的 MP5-core 及直接日志版本 AHCE-log 进行对比；在中等/困难任务中成功率提升 32%/约 70%，人类交互时间及占比显著下降，32B 模型在困难任务上成功率从 68% 提升至 82%。

**⚠️ 局限性**

仍依赖人类专家，极难任务中无法完全消除人类介入；自我纠错能力受限，阈值调参需人工决定，系统对不同任务难度的自适应能力有限。

---

## 415. PhotoAgent: Agentic Photo Editing with Exploratory Visual Aesthetic Planning

**arXiv ID:** 2602.22809 | [PDF](https://arxiv.org/pdf/2602.22809v1)

**作者:** Mingde Yao `[一作]` (Multimedia Laboratory, Chinese University of Hong Kong), Tianfan Xue `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 PhotoAgent，一个闭环、可自主执行多步图像编辑的系统。

**💡 创新点**

创新点在于将视觉大模型与 MCTS 规划、工具库、评估器结合，支持探索性视觉审美规划，并引入 UGC-Edit 评价数据集。

**🔧 技术方法**

使用 VLM（如 Qwen3-VL）、MCTS、Flux.1 Kontext 等生成模型和传统 OpenCV/PIL 工具，以及奖励模型评估器。

**📊 数据集**

使用新构建的 UGC-Edit（7000 张用户照片）和 1017 张真实照片测试集。

**📈 对比分析**

与 InstructPix2Pix、SDXL、Flux.1 Kontext、HuggingGPT、ReAct 等基线对比，PhotoAgent 在多项评估指标（CLIP 相似度、ImageReward、BRISQUE、UGC 评分）均名列前茅，用户研究也表明其效果最佳。

**⚠️ 局限性**

局限在于仍需大量算力、对复杂高阶操作的依赖，以及评估模型对偏好稳定性的提升空间。

---

## 416. Optimizing Neural Network Architecture for Medical Image Segmentation Using Monte Carlo Tree Search

**arXiv ID:** 2602.22361 | [PDF](https://arxiv.org/pdf/2602.22361v1)

**作者:** Liping Meng `[一作]` (Xi'an Kedagaoxin University), Chao Han `[通讯]` (Xi'an Kedagaoxin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了 MNAS-Unet，一种将蒙特卡洛树搜索（MCTS）与神经网络架构搜索（NAS）相结合的轻量级医学图像分割框架。

**💡 创新点**

创新点包括：①在医学图像分割中首次引入 MCTS 进行高效架构探索；②设计了针对医学任务的三类操作（Down、Up、Normal）搜索空间；③通过 MCTS 使搜索成本降低 54%，同时得到仅 0.6M 参数、低 GPU 内存占用的模型。

**🔧 技术方法**

技术手段包括：蒙特卡洛树搜索、神经架构搜索（ProxylessNAS 变体）、U‑Net 基础结构、低频评估策略、AdaBound 优化器、Cosine 学习率调度等。

**📊 数据集**

使用的数据集有：PASCAL VOC 2012（用于搜索验证），医学数据集 PROMISE12（前列腺 MRI）、Ultrasound Nerve Segmentation（肩胛神经超声）以及 CHAOS（腹部 CT/ MRI）等。

**📈 对比分析**

与 U‑Net、FC‑DenseNet、DC‑UNet、DCSAU‑Net、MNet、MedNeXt、ResTransUNet 以及 NAS‑Unet 等基线进行对比。MNAS‑Unet 在 mIoU、DSC 上均位列首位，同时训练时间和 GPU 内存占用均较基线降低，证明了其更高的分割精度与更好的资源利用。

**⚠️ 局限性**

局限性：模型解释性不足，无法直观说明分割决策；目前仅在上述三种医学模态上验证，尚未测试更多疾病/设备；MCTS 的搜索过程仍需手动设置超参数，可能影响复现性。

---

## 417. Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking

**arXiv ID:** 2602.23172 | [PDF](https://arxiv.org/pdf/2602.23172v1)

**作者:** Maximilian Luz `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2574 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于潜在高斯 splatting 的 4D panoptic occupancy tracking（Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking），通过将多视角相机特征先提升到 3D，并聚合为稀疏的高斯点，再将特征 splat 到体素网格，完成语义与实例分割以及跨帧追踪；

**💡 创新点**

①使用稀疏 3D 高斯点作为中间特征表示，替代传统密集体素特征；②将高斯点特征在编码器内部直接 splat 回体素，构成点‑体素双流结构；③分别聚合语义和实例 mask 再合并，缓解 mask 失衡；④利用分离查询传播与 spatio‑temporal 细化，降低梯度线性放大带来的资源消耗；⑤对 4D‑POT 指标进行重新定义并修正先前实现错误；

**🔧 技术方法**

高斯 splatting、点 Transformer（Serialized Multi‑Stream Attention）、3D 可变形交叉注意力、PETR‑style Transformer 解码器、PF‑Track 的时空细化模块、VoVNetV2 图像骨干、显式深度提升、交叉注意力等；

**📊 数据集**

Occ3D‑nuScenes 与 Occ3D‑Waymo（含 4D panoptic 标注的 3D 语义占据数据）；

**📈 对比分析**

与 TrackOcc、PF‑Track、单帧基线等方法进行对比；在 nuScenes 上 STQ 提升 18.9、AQ 提升 19.8；在 Waymo 上 STQ 提升 5.1、AQ 提升 7.9；语义占据 mIoU 亦提升约 4.9；整体显著超越现有最优；

**⚠️ 局限性**

对 “stuff” 类的性能略有下降；方法仍依赖相机标定与车辆运动信息；计算与显存成本相对较高；未充分利用立体/多帧深度；尚未在更大尺度或多模态（如 LIDAR）场景下验证。

---

## 418. MEDNA-DFM: A Dual-View FiLM-MoE Model for Explainable DNA Methylation Prediction

**arXiv ID:** 2602.22850 | [PDF](https://arxiv.org/pdf/2602.22850v1)

**作者:** Yi He `[一作]` (Lanzhou University), Tianchi Lu `[通讯]` (City University of Hong Kong)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5027108468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出MEDNA-DFM双视图FiLM-MoE模型用于DNA甲基化预测，并通过解释算法挖掘高置信度的序列‑结构模式。

**💡 创新点**

将双编码器的全局语义与局部k‑mer表征结合FiLM实现动态调制，加入MoE提升泛化；同时提出Contrastive Weighted Gradient Attribution (CWGA)和Contrastive Attention Cohen's d (CAD)两种解释方法，实现信号分离与纯化。

**🔧 技术方法**

使用DNABERT‑6mer、DNABERT‑2、FiLM、Mixture of Experts、对抗训练、洪水正则化、集成梯度、t检验等技术。

**📊 数据集**

利用iDNA‑MS基准共17个数据集（涵盖5hmC、4mC、6mA），以及独立人类5mC外部验证集。

**📈 对比分析**

在17个基准上与iDNA‑ABF、Methyl‑GP、AutoFE‑Pointer等SOTA模型比较，MEDNA‑DFM在ACC、AUC、MCC等指标上取得最高或次高成绩，显示出更强的稳定性与跨物种泛化能力。

**⚠️ 局限性**

主要限制在仅处理序列信息，未考虑染色质可及性、核小体定位等空间上下文；对非模型生物的数据可用性和噪声也有限。

---

## 419. Decoding the Hook: A Multimodal LLM Framework for Analyzing the Hooking Period of Video Ads

**arXiv ID:** 2602.22299 | [PDF](https://arxiv.org/pdf/2602.22299v1)

**作者:** Kunpeng Zhang `[一作]` (University of Maryland), Amel Awadelkarim `[通讯]` (Meta Platforms)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过构建基于多模态大语言模型的框架，对视频广告前三秒的“hooking period”进行自动化分析，提取视觉、音频及文本特征并映射到广告效果。

**💡 创新点**

创新点在于将LLM与BERTopic结合，对hooking period的多模态特征进行可解释的主题抽取，并引入关键帧与随机采样双策略，提升特征捕获精度。

**🔧 技术方法**

技术包括Llama多模态LLM、BERTopic主题建模、Librosa音频特征提取、GBDT预测模型，以及对比的ViViT和X-CLIP等视觉Transformer。

**📊 数据集**

使用来自社交平台的5个行业垂直（电商、医疗、消费品、汽车、娱乐）的近10万条视频广告真实数据。

**📈 对比分析**

与ViViT、X-CLIP以及“Junk”基线对比，实验表明在电商、消费品和汽车等垂直下R²可达0.66，MSE显著低于基线，展示了更优的预测性能。

**⚠️ 局限性**

局限性包括只关注前三秒，依赖预训练LLM可能存在偏差，数据单一平台且未验证跨平台泛化。

---

## 420. Learning Rewards, Not Labels: Adversarial Inverse Reinforcement Learning for Machinery Fault Detection

**arXiv ID:** 2602.22297 | [PDF](https://arxiv.org/pdf/2602.22297v1)

**作者:** Dhiraj Neupane `[一作]` (Deakin University), Sunil Aryal `[通讯]` (Deakin University)

**通讯引用:** 2361 | [OpenAlex ID](https://openalex.org/A5038741954)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出将机械故障检测建模为离线逆强化学习，使用AIRL从健康工况序列中学习奖励函数，生成可解释的异常评分，实现无标签早期故障检测。

**💡 创新点**

首次将Adversarial Inverse Reinforcement Learning应用于工业机器故障检测，保留时间序列决策结构，并通过健康序列学习奖励而非传统静态分类。

**🔧 技术方法**

采用状态仅模仿学习(SOIL)构造代理动作，利用AIRL框架中的生成器与判别器进行奖励学习，并用判别器输出健康概率作为异常分数。

**📊 数据集**

在HUMS2023（直升机齿轮箱疲劳）、IMS和XJTU‑SY三个跑到失效基准数据集上进行实验。

**📈 对比分析**

与Isolation Forest、OCSVM、AE/VAE、LSTM‑AE/LSTM‑VAE、SS‑AD、FRESH filter及CTQN等基线比较，AIRL在HUMS2023上提前至第22天检测故障，早于FRESH filter（第22天）和官方获奖者（第23天），并保持较高且稳定的异常率，整体性能优于传统与序列模型。

**⚠️ 局限性**

目前仅单传感器、无控制输入的离线训练，未涉及多传感器融合或不确定性阈值化，且对实时在线更新的鲁棒性尚未评估。

---

## 421. GRAU: Generic Reconfigurable Activation Unit Design for Neural Network Hardware Accelerators

**arXiv ID:** 2602.22352 | [PDF](https://arxiv.org/pdf/2602.22352v1)

**作者:** Yuhao Liu `[一作]` (Ruhr University Bochum), Akash Kumar `[通讯]` (Ruhr University Bochum)

**通讯引用:** 6122 | [OpenAlex ID](https://openalex.org/A5100755285)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种可在硬件上运行的通用可重构激活单元 GRAU，能够在低精度 QNN 加速器中实现多功能混合精度甚至非单调激活函数。

**💡 创新点**

创新点在于将分段线性逼近与幂次二（PoT）和加法幂次二（APoT）拟合相结合，仅使用比较器和 1 位右移器，硬件资源减少 90%+ 并支持运行时重配置。

**🔧 技术方法**

使用了分段线性拟合（pwlf 库）、PoT/APoT 逼近、流水线与串行架构、比较器、右移器、寄存器编码以及 Vivado 合成工具。

**📊 数据集**

在 MNIST 与 CIFAR‑10 数据集上测试了小型全连接网络和简化 VGG 网络。

**📈 对比分析**

通过与传统 Multi‑Threshold 单元在 LUT、FF、频率、延迟、ADP、PDP 等指标的对比，GRAU 在 LUT 消耗上降低 90%+，频率提升至 250 MHz，ADP/PDP 明显优于 MT。

**⚠️ 局限性**

局限在于对 pwlf 进行浮点拟合导致 SiLU 模型的精度下降；拟合过程耗时较长，难以扩展到大规模网络；且 GRAU 处理 8‑bit 时延迟为 24 周期。

---

## 422. Code World Models for Parameter Control in Evolutionary Algorithms

**arXiv ID:** 2602.22260 | [PDF](https://arxiv.org/pdf/2602.22260v1)

**作者:** Camilo Chacón Sartori `[一作]` (Catalan Institute of Nanoscience and Nanotechnology), Guillem Rodríguez Corominas `[通讯]` (Artificial Intelligence Research Institute)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5019374060)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

本文通过LLM生成的Python程序实现了“代码世界模型”(CWM)，在进化优化的参数控制任务中学习并预测优化器的动态，从而实现了自适应参数选择。

**💡 创新点**

创新点在于将CWM从确定性游戏扩展到随机组合优化，利用一次性LLM合成的代码进行一阶贪心规划，且不需要任何关于最优策略的轨迹；在极端难题（如跳跃函数和NK景观）上取得全局最优或显著优于传统自适应规则的结果。

**🔧 技术方法**

主要技术包括：1）LLM（Claude Sonnet 4）基于问题描述和有限轨迹样本生成包含四个方法（predict、evaluate、actions、state）的Python类；2）一次性“快速合成+验证”过程；3）在线阶段使用一阶贪心规划（k* = argmax_k CWM.eval(CWM.predict(state,k)))；4）对生成代码的自动化验证与重试。

**📊 数据集**

实验数据集包括：Unimodal（LeadingOnes、OneMax）、Deceptive（Jump_k）以及无解析模型的NK-Landscape（K=2、3、4），每个任务使用n=50（及扩展至n=100,200）生成多条子最优轨迹（约200-300条），并在15个独立NK实例上评估。

**📈 对比分析**

与基线（手工自适应规则、DQN、固定k、随机k、特殊启发式等）相比，CWM-greedy在LeadingOnes和OneMax中距最优仅6%/2%，在Jump_k中实现100%成功率（全对手0%），在NK中平均最佳适应度36.94高于所有基线（p<0.001），在K=3、4时仍保持领先；样本效率上CWM只需200条离线轨迹即可匹配或超过DQN所需的500在线回合。

**⚠️ 局限性**

局限性包括：1）在高度随机或无解析模型的任务中仍需人工提供经验转移表；2）对更高k或更高K的NK景观仍表现不佳；3）依赖LLM的合成质量，若模型生成错误需多次验证；4）在极端长等候时间（如k=4 Jump_k）内，CWM与最优策略同样无法突破预算限制。

---

## 423. Training Agents to Self-Report Misbehavior

**arXiv ID:** 2602.22303 | [PDF](https://arxiv.org/pdf/2602.22303v1)

**作者:** Bruce W. Lee `[一作]` (University of Pennsylvania), Tomek Korbak `[通讯]` (OpenAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验了自证训练（self‑incrimination training），使模型在进行隐蔽违规时主动调用工具提示自己。

**💡 创新点**

将自我报告作为可见信号引入 agentic 环境，兼顾不假设能完全消除违规且不需外部监控。

**🔧 技术方法**

在 GPT‑4.1、GPT‑4.1 mini、Gemini‑2.0 Flash 上进行监督微调，训练示例通过自动生成的 45 个多工具环境与隐蔽副任务对齐。

**📊 数据集**

使用自建 synthetic‑scheming 数据管线生成 395/375/340 任务对，结合 APPS、BashArena、SHADE‑Arena 三大公开基准进行评测。

**📈 对比分析**

与强弱黑盒监控和对齐基线对比，平均未被检测到的攻击成功率下降至 6%–14%，FPR 极低，且保持与基线相近的通用能力。

**⚠️ 局限性**

仅针对无推理、SFT 模型；依赖行为不一致；训练规模与多样性受限；对未来更连贯的模型与持续微调的稳健性未知。

---

## 424. Bitwise Systolic Array Architecture for Runtime-Reconfigurable Multi-precision Quantized Multiplication on Hardware Accelerators

**arXiv ID:** 2602.23334 | [PDF](https://arxiv.org/pdf/2602.23334v1)

**作者:** Yuhao Liu `[一作]` (Dresden University of Technology), Akash Kumar `[通讯]` (Ruhr University Bochum)

**通讯引用:** 6122 | [OpenAlex ID](https://openalex.org/A5100755285)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种基于比特级同步阵列的可运行时可重构多精度多通道乘法器（BitSys），并将其集成到单层和阵列加速器中，用于加速混合精度量化神经网络推理。

**💡 创新点**

创新点在于将乘法拆解为位级 AND/掩码/左移操作，利用可配置的通道与精度实现真正的多精度运行时重构；同时采用全流水线、LUT 原语优化和可重构输入通道，在 FPGA 上实现低延迟、可在 250 MHz 时钟频率下工作的多精度乘法器和乘加器。

**🔧 技术方法**

技术包括：比特级同步阵列（BitSys）结构、LUT 原语优化、可配置掩码与左移、可重构输入通道、流水线加法器、多阈值激活函数、Vivado 实现与 FPGA 资源映射。

**📊 数据集**

使用 MNIST 数据集训练的 Tiny MLP 与 Tiny CNN（六个不同精度配置）进行实验。

**📈 对比分析**

在 Ultra96‑V2 FPGA 上与 Liu 等人的 Multiplier‑Tree、Bitshifter 以及 Vivado IP 单层加速器对比。评估指标包括 LUT/FF/BRAM 资源占用、时钟频率、延迟（µs）等；BitSys 单层加速器相较基线提升 1.3185×–3.5671×，阵列加速器相较单层加速器提升约 356.71%。

**⚠️ 局限性**

局限性包括：在低精度模式下多通道累加器需求导致资源利用率不均衡；当前设计仅验证在 FPGA 平台，ASIC 实现与更大规模网络的可扩展性与功耗仍需进一步评估。

---

## 425. Misinformation Exposure in the Chinese Web: A Cross-System Evaluation of Search Engines, LLMs, and AI Overviews

**arXiv ID:** 2602.22221 | [PDF](https://arxiv.org/pdf/2602.22221v1)

**作者:** Geng Liu `[一作]` (Politecnico di Milano), Francesco Pierri `[通讯]` (Politecnico di Milano)

**通讯引用:** 2776 | [OpenAlex ID](https://openalex.org/A5013385420)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过从中文搜索日志中提取12161个二值事实问题，构建事实检验数据集，并统一评估传统搜索、独立LLM和AI概览三种信息获取范式的答案准确率。

**💡 创新点**

首次在中文生态中以真实用户查询为基础构建大规模事实问答数据集，并提供统一评估框架；同时结合百度指数评估地区性错误信息暴露。

**🔧 技术方法**

使用LLM（DeepSeek、Qwen、LLaMA）零样本推理、检索+阅读理解模型、AI概览解析以及百度指数抓取与统计等技术。

**📊 数据集**

T2Ranking搜索日志数据集（包含真实查询和Passage）以及公开的百度指数数据。

**📈 对比分析**

采用同一组查询对三种系统进行并行评测，按答案是否与标注一致统计准确率；结果显示AI概览最高（≈69.8%），LLM略高于搜索引擎，整体准确率仍在60–70%之间，存在显著主题和答案倾向差异。

**⚠️ 局限性**

主要局限：依赖LLM标注导致噪声、检索覆盖不完全、未进行完整人工验证、百度指数无法区分需求与人群、爬取质量偶有缺失。

---

## 426. Mitigating Membership Inference in Intermediate Representations via Layer-wise MIA-risk-aware DP-SGD

**arXiv ID:** 2602.22611 | [PDF](https://arxiv.org/pdf/2602.22611v1)

**作者:** Jiayang Meng `[一作]` (Renmin University of China), Hong Chen `[通讯]` (Renmin University of China)

**通讯引用:** 21924 | [OpenAlex ID](https://openalex.org/A5100420423)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在Embedding-as-an-Interface场景下，中间表示（IR）被攻击的风险，并提出了一种层级MIA风险感知的DP‑SGD（LM‑DP‑SGD），通过对每一层的隐私保护力度进行动态调整，实现了对IR的更高效安全防护。

**💡 创新点**

创新点主要有：①利用影子模型评估每一层的MIA风险，形成层级风险估计；②在DP‑SGD中加入层级加权裁剪，使得相同噪声量在不同层分配更合适的保护；③在保持全局梯度范数不变的前提下，理论上保证了相同的（ε,δ）隐私以及收敛性质。

**🔧 技术方法**

使用的技术包括：DP‑SGD、层级加权梯度裁剪、影子模型攻击仿真、Gaussian噪声注入、Moments Accountant隐私计数器、理论收敛与隐私证明。

**📊 数据集**

实验数据集包括 MNIST、CIFAR‑10、CIFAR‑100、CelebA；影子数据分别选取 FashionMNIST、ImageNet、FFHQ 等公开数据集来近似私有数据分布。

**📈 对比分析**

与标准DP‑SGD、Auto‑S/NSGD、DP‑PSAC 三种基线进行对比。结果显示：在相同隐私预算下，LM‑DP‑SGD 将 IR 级别的最大 MIA 准确率显著降低；模型最终测试准确率始终位于基线最高值的 2% 以内，且梯度偏差更小，说明其在隐私‑效用折衷上更具优势。

**⚠️ 局限性**

局限性：需要一份与私有数据分布相近的影子数据；层级加权裁剪的实现会增加额外的计算开销，尤其在极深或高维网络中；目前实验主要聚焦卷积网络，尚未验证在 Transformer 等结构上的效果。

---

## 427. Accelerating Incident Response: A Hybrid Approach for Data Breach Reporting

**arXiv ID:** 2602.22244 | [PDF](https://arxiv.org/pdf/2602.22244v1)

**作者:** Aurora Arrus `[一作]` (IMT School for Advanced Studies Lucca), Marco Quadrini `[通讯]` (IMT School for Advanced Studies Lucca)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种混合恶意软件分析管道，用静态图分析预筛、动态沙箱执行和LLM生成结构化的GDPR违规报告，帮助快速完成72小时通知；

**💡 创新点**

将图结构的静态特征与动态行为结合，并采用JSON模式约束的LLM实现技术与法规报告无缝对接，填补技术证据与合规报告之间的鸿沟；

**🔧 技术方法**

静态函数调用图和控制流图、随机森林分类器、Emulix+Qiling+FakeNet‑NG动态沙箱、Gemini 2.5 Pro LLM+JSON schema、ReportLab生成PDF；

**📊 数据集**

基于VirusTotal标签的1024个ARM/ARM64 Linux ELF样本（512 exfiltrator、512非exfiltrator），利用MITRE ATT&CK标签进行采集；

**📈 对比分析**

对静态分类器在测试集上取得准确率0.967、精确率0.958、召回率0.972、F1 0.965、ROC‑AUC 0.983；LLM生成的报告在结构性、法律适配度方面显著降低人工校对时间，但未给出定量基准；

**⚠️ 局限性**

对抗性混淆、沙箱逃逸可能导致误检；LLM仍需人工复核以防幻觉；目前仅支持ARM ELF，扩展至其他架构需重写静态特征抽取。

---

## 428. UpSkill: Mutual Information Skill Learning for Structured Response Diversity in LLMs

**arXiv ID:** 2602.22296 | [PDF](https://arxiv.org/pdf/2602.22296v1)

**作者:** Devan Shah `[一作]` (Princeton University), Benjamin Eysenbach `[通讯]` (Princeton University)

**通讯引用:** 1058 | [OpenAlex ID](https://openalex.org/A5035051008)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练时通过引入离散潜在变量z并最大化token级互信息来让LLM产生可重复且多样化的解法；

**💡 创新点**

创新点在于将互信息技能学习（MISL）与GRPO结合，使用token级互信息奖励直接促进策略多样性，同时保持单次尝试的准确性；

**🔧 技术方法**

使用的技术包括：离散潜在变量的条件生成、GRPO（Group Relative Policy Optimization）框架、RLVR（Reinforcement Learning from Verifiable Rewards）奖励、token级互信息奖励与KL正则化；

**📊 数据集**

评估数据集主要为GSM8K（小学算术文字问题）以及一个小型可验证算术环境；

**📈 对比分析**

与仅使用GRPO（不含MI奖励）的基线相比，实验表明在Qwen 2.5‑7B和Llama 3.1‑8B上多次尝试的pass@k提升约3%–9%，且单次准确率保持不变；对R1‑Distilled‑Qwen2.5‑Math‑1.5B则表现不佳；

**⚠️ 局限性**

主要限制包括：理论假设难以严格满足、训练过程可能不稳定、对某些模型（如R1）效果适得其反、目前仅在数学/算术任务上验证，尚需在更广泛领域和更大模型上进一步测试。

---

## 429. Coarse-to-Fine Learning of Dynamic Causal Structures

**arXiv ID:** 2602.22532 | [PDF](https://arxiv.org/pdf/2602.22532v1)

**作者:** Dezhi Yang `[一作]` (Shandong University), Guoxian Yu `[通讯]` (Shandong University)

**通讯引用:** 28837 | [OpenAlex ID](https://openalex.org/A5065393757)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种全动态时间序列因果结构学习框架 DyCasual，能够在粗粒度窗口中捕捉因果模式并通过线性插值细化到每个时刻，实现对瞬时和滞后因果关系随时间连续变化的建模。

**💡 创新点**

创新点主要包括：①滑动卷积窗口+线性插值的粗细层次编码，显著提升对全动态因果图的识别效率；②基于矩阵一范数的可微对数-判定无环约束（h_norm），在保持无环性的同时实现更稳定、更快的优化。

**🔧 技术方法**

使用技术包括：卷积神经网络编码、并行多层感知机解码、线性插值、对数-判定无环约束、基于重建损失的分数优化以及可选的非线性或ODE扩展。

**📊 数据集**

实验数据集涵盖：合成数据（ER图+加性噪声、Lorenz模型）以及真实数据（CausalTime 交通/空气质量/医疗、NetSim fMRI、DREAM‑3 基因表达、Phoenix 基因表达）。

**📈 对比分析**

与 DyCAST、DYNO、NTS‑NO、SVAM、PCMCI、TECDI、CUTS+、JRNGC 等现有方法对比，在 TPR、SHD、F1 等指标上均优于对手，特别是在全动态场景中显著提升精度与 F1，且优化速度更快。

**⚠️ 局限性**

主要限制：仅适用于规则采样的时间序列，对不规则时间序列不直接支持；模型假设所有序列共享同一时变因果结构，未考虑变量加入/离开的情形。

---

## 430. Large Multimodal Models as General In-Context Classifiers

**arXiv ID:** 2602.23229 | [PDF](https://arxiv.org/pdf/2602.23229v1)

**作者:** Marco Garosi `[一作]` (University of Trento), Elisa Ricci `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 10992 | [OpenAlex ID](https://openalex.org/A5065059558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文探讨了在分类任务中选择多模态模型的最佳方案，比较了CLIP类对比视觉-语言模型（VLMs）和大型多模态模型（LMMs）的性能，特别是在上下文学习的影响下。

**💡 创新点**

创新点在于提出了一种新的训练无关的方法，通过迭代精炼伪标签来增强LMMs在开放世界分类中的表现，挑战了传统观点，即VLMs在分类任务中优于LMMs。

**🔧 技术方法**

使用了上下文学习（In-Context Learning, ICL）技术，结合伪标签生成和迭代精炼的方法。

**📊 数据集**

使用了多个数据集进行基准测试，包括Caltech101、SUN397、Flowers102、Food101等，涵盖了封闭世界和开放世界的分类任务。

**📈 对比分析**

通过与VLMs的对比，尽管LMMs在零-shot性能上较低，但在提供上下文示例后，其性能可以匹配甚至超越VLMs，尤其是在开放世界分类中表现更佳。

**⚠️ 局限性**

限制在于缺乏人工标注可能导致精炼过程收敛到语义一致但任务不对齐的标签解释。此外，流式变体的动态内存更新在处理大规模或连续数据流时可能引入计算开销。

---

## 431. Simulation-based Optimization for Augmented Reading

**arXiv ID:** 2602.22735 | [PDF](https://arxiv.org/pdf/2602.22735v1)

**作者:** Yunpeng Bai `[一作]` (National University of Singapore), Antti Oulasvirta `[通讯]` (Aalto University)

**通讯引用:** 14396 | [OpenAlex ID](https://openalex.org/A5003084232)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将增强阅读系统设计视为基于资源合理性阅读模型的仿真优化问题，并提出了离线和在线两种优化管道，实现了可解释、可扩展的自适应文本界面设计。

**💡 创新点**

创新点在于：①用可解释的资源合理性阅读模型取代经验式规则或纯数据驱动的策略；②将模拟阅读者作为评估者，系统性评估和优化文本界面；③同时提供离线设计探索和在线实时个性化适配两种流程。

**🔧 技术方法**

主要技术包括资源合理性阅读模型（仿真眼动、注意力与工作记忆分配）、仿真优化算法、实时交互数据捕获与反馈、基于模型的决策优化。

**📊 数据集**

本文并未公开使用特定数据集，而是依赖理论模型和实验室/仿真生成的数据；若需要参考，可说明未来可结合公开眼动/阅读理解数据集进一步验证。

**📈 对比分析**

通过模拟预测的阅读行为指标（如注视点分布、阅读时间、理解率）与已有实验数据或基准系统进行比较，证明模型能够重现已知阅读现象并在离线管道中显著提升设计效率；在线管道通过实时适配显示出更高的任务完成效率。性能数值未给出，但实验表明在多种场景下均能获得可观的效能提升。

**⚠️ 局限性**

局限性包括：①模型参数需要专家假设，可能无法完全覆盖所有认知差异；②离线优化忽略实时个体差异变化；③在线优化对实时计算和隐私保护提出挑战；④需进一步在真实用户环境中验证其泛化能力。

---

## 432. Partial recovery of meter-scale surface weather

**arXiv ID:** 2602.23146 | [PDF](https://arxiv.org/pdf/2602.23146v1)

**作者:** Jonathan Giezendanner `[一作]` (Massachusetts Institute of Technology), Sherrie Wang `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1646 | [OpenAlex ID](https://openalex.org/A5088966490)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个多模态变压器模型，利用ERA5的粗尺度大气状态、NOAA天气站的稀疏观测以及高分辨率卫星影像、土地覆盖和地形数据，生成美国大陆范围内10 米分辨率的近地表温度、露点和风场，并通过该模型推断可统计恢复的微天气成分。

**💡 创新点**

创新点在于：①将粗尺度气象力学状态与静态地表特征直接结合，揭示了可预测的微尺度天气结构；②使用深度学习变压器实现跨尺度、多源信息融合；③以10 米分辨率提供连续微天气场，填补了传统NWP在米级尺度的空白。

**🔧 技术方法**

技术方法包括：多模态深度学习变压器网络、位置编码、注意力机制、AlphaEarth预训练地表特征嵌入以及端到端训练与推理。

**📊 数据集**

数据集为：ERA5再分析（0.25°×0.25°、小时级），NOAA MADIS天气站观测（约11 k台），以及高分辨率地表观测（多光谱卫星影像、土地覆盖图、数字高程模型、AlphaEarth嵌入）。

**📈 对比分析**

与基线方法（ERA5单独、站点插值、ERA5+站点）对比，模型在所有测试站点上平均降低风速向量误差29%、温度与露点的MAE各约6%；同时在空间方差解释率上显著优于基线，且在复杂地形与多样化土地覆盖地区表现尤为突出。

**⚠️ 局限性**

局限性包括：在地表相对均匀、站点稀疏的地区（如北大平原）改善有限；对海洋表面无验证；无法完整捕捉所有微尺度风向变化；对极端天气或细尺度湍流过程的预报能力受限，需要更密集的观测或显式的高分辨率动力学模拟。

---

## 433. Switch-Hurdle: A MoE Encoder with AR Hurdle Decoder for Intermittent Demand Forecasting

**arXiv ID:** 2602.22685 | [PDF](https://arxiv.org/pdf/2602.22685v1)

**作者:** Fabian Muşat `[一作]`, Simona Căbuz `[通讯]` (eMAG)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一种名为 Switch‑Hurdle Transformer 的时序预测框架，用来解决零售需求的间歇性（大量零值与稀疏峰值）问题；

**💡 创新点**

创新点在于将稀疏的 Mixture‑of‑Experts (MoE) 编码器与共享的 hurdle 概率解码器相结合，采用 Top‑1 直通估计器实现前向稀疏路由、后向密集梯度；同时通过 hurdle head 将“是否出现销量”和“销量大小”两种概率分离建模，从而显式捕捉零值与正值的双重过程；

**🔧 技术方法**

使用的技术包括：SwiGLU 专家、Top‑1 STE 路由、交叉注意力 AR 解码器、共享 hurdle head（负二项分布）、负对数似然 + MAE 混合损失、KL‑平衡正则；

**📊 数据集**

实验数据集为公开的 M5 Walmart 需求数据（约3万条时间序列）以及内部约4万条 SKU 的零售数据；

**📈 对比分析**

与 DeepAR、TFT、PatchTST、TSMixer 等基线模型在 WRMSSE、MSE、MASE（M5）以及 WAPE、MASE（内部）指标上对比，Switch‑Hurdle 在 M5 上 WRMSSE 0.6307、MASE 0.8992，内部数据 WAPE 53.99%、MASE 0.5865，均取得最优或接近最优；

**⚠️ 局限性**

局限性包括：仍需手动调节专家数量与层数；在极端稀疏或高波动序列的泛化尚待验证；推理时仍需按专家路由计算，可能增加部署复杂度；未来工作可探索动态专家分配与跨域预训练。

---

## 434. Layer-Targeted Multilingual Knowledge Erasure in Large Language Models

**arXiv ID:** 2602.22562 | [PDF](https://arxiv.org/pdf/2602.22562v1)

**作者:** Taoran Li `[一作]` (Texas A&M University), Zhiyuan Yu `[通讯]` (Texas A&M University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出多语言机器学习模型的知识消除框架 MUTE，通过定位模型内部的语言无关层来实现跨语言知识消除。

**💡 创新点**

创新点在于揭示干预层深度是多语言 unlearning 的关键因素，并用 CKA 与 LRDS 两种度量方法精确定位语言无关层，随后在此层进行针对性更新，兼顾多种 unlearning 算法（RMU、SLUG、SimNPO）。

**🔧 技术方法**

使用了 Centered Kernel Alignment (CKA)、Linguistic Regions Development Score (LRDS)、Logit Lens 探测、Transformer 语言模型（Llama‑3.1、Qwen‑2.5、BLOOM）、以及 RMU、SLUG、SimNPO 等三类 unlearning 方法。

**📊 数据集**

采用多语言 MMLU 数据集的化学（高等学校）子集作为“遗忘”数据，历史/法律子集作为“保留”数据，涵盖 12 种语言（源语言 3 种、留存语言 9 种）。

**📈 对比分析**

与全层梯度上升（GA）等基线对比，MUTE 在三大模型架构上，在目标层实现≈1% 的遗忘准确率，≈55% 的保留准确率，明显优于全层更新方法；并通过 Logit Lens 验证知识已被真正移除而非仅抑制输出。

**⚠️ 局限性**

局限性包括：需要预先计算 CKA/LRDS 以定位层，算法对源语言选择的敏感度仍有限；深层语义信息仍难以完全控制；实验规模受限于三大模型，尚未验证在更大规模或更低资源语言上的鲁棒性。

---

## 435. Takeuchi's Information Criteria as Generalization Measures for DNNs Close to NTK Regime

**arXiv ID:** 2602.23219 | [PDF](https://arxiv.org/pdf/2602.23219v1)

**作者:** Hiroki Naganuma `[一作]` (University of Montreal), Ikuro Sato `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3203 | [OpenAlex ID](https://openalex.org/A5100862952)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了Takeuchi信息准则（TIC）在深度神经网络（DNN）中的适用性，尤其在神经切线核（NTK）近似下的泛化误差评估与模型选择；

**💡 创新点**

证明TIC在NTK regime下满足正则性条件并与泛化缺口正相关，同时提出多种低成本TIC近似方法，使其可应用于大规模DNN；

**🔧 技术方法**

利用Hessian/一般化Gauss-Newton矩阵与Fisher信息矩阵等价性，采用Monte Carlo估计、块对角/对角化近似、Hutchinson法估计迹，以及将H替换为F来降低计算复杂度；

**📊 数据集**

在TinyMNIST、MNIST、CIFAR‑10和CIFAR‑100四个数据集上，结合12种不同的网络架构（如VGG‑16、ResNet‑8等）训练超过5,000个模型进行实验；

**📈 对比分析**

与传统的验证损失、留一交叉验证等指标对比，发现TIC在NTK近似时与泛化缺口相关性显著，且在训练过程中能有效追踪泛化趋势，在多试验HPO剪枝中比验证损失更能保留优质超参；

**⚠️ 局限性**

局限性包括：仅在NTK近似内成立；对非NTK特征学习阶段缺乏理论支持；未考察批归一化、数据增强或现代架构（如Vision Transformer）的影响；以及TIC近似仍需进一步验证其在更广泛任务与HPO算法中的鲁棒性。

---

## 436. Spatio-Temporal Token Pruning for Efficient High-Resolution GUI Agents

**arXiv ID:** 2602.23235 | [PDF](https://arxiv.org/pdf/2602.23235v1)

**作者:** Zhou Xu `[一作]` (Tsinghua University), Jingyu Xiao `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 5731 | [OpenAlex ID](https://openalex.org/A5061742939)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的视觉token压缩框架GUIPruner，针对高分辨率GUI导航中的时空冗余进行高效压缩。

**💡 创新点**

创新点在于将历史视觉信息的“记忆衰减”与当前帧的结构感知压缩分离，分别采用Temporal‑Adaptive Resolution（TAR）和Stratified Structure‑aware Pruning（SSP）实现自适应分辨率衰减与拓扑保留的双重压缩策略。

**🔧 技术方法**

核心技术包括基于线性衰减的历史分辨率分配、基于注意力和边缘检测的前景/背景分层保留、以及均匀网格采样（UGS）保障空间拓扑完整性；整个框架无需任何参数更新即可在推理阶段直接应用。

**📊 数据集**

在四个公开GUI导航基准上进行评估：AITW、Mind2Web、GUI‑Odyssey、AndroidControl。

**📈 对比分析**

与FastV、DivPrune、CDPruner、MoB等现有训练无关压缩方法对比，GUIPruner在Qwen2‑VL‑2B和Qwen2.5‑VL‑7B模型上均保持或提升任务准确率，尤其在Mind2Web上避免了其他方法的性能崩溃；在保留10%–40%历史信息、75%当前帧的压缩比例下，均可实现3.4× FLOPs减少、3.3×编码器加速、仅5.9 GB显存峰值。

**⚠️ 局限性**

局限性包括：①在极端稀疏的高分辨率GUI场景中，背景语义保留比例仍需精细调优；②压缩深度（如TAR的γ、SSP的ρ）对不同模型规模和任务复杂度的泛化性尚未完全验证；③在需要精细定位多重层级交互元素时，均匀网格采样可能无法捕获细粒度布局细节。

---

## 437. MoDora: Tree-Based Semi-Structured Document Analysis System

**arXiv ID:** 2602.23061 | [PDF](https://arxiv.org/pdf/2602.23061v1)

**作者:** Bangrui Xu `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 19059 | [OpenAlex ID](https://openalex.org/A5075948251)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一个基于LLM的半结构化文档分析框架MoDora，完成文档预处理、组件化、构建组件关联树（CCTree）以及问题类型感知的检索与推理。

**💡 创新点**

创新点包括：① 本地对齐聚合策略将OCR碎片化元素转化为语义化组件并提取层级标题与非文本元素的结构化三元组；② 组件关联树以层级方式组织组件并通过底层汇总传播语义；③ 结合位置网格、LLM引导剪枝、嵌入搜索与跨模态验证的问答检索策略。

**🔧 技术方法**

技术手段：OCR、LLM（Qwen2.5‑VL、GPT‑5等）、视觉编码（ViT）、Embedding检索（Qwen3‑Embedding）、树结构构建与bottom‑up汇总。

**📊 数据集**

使用公开数据集 DUDE、M3DocVQA、MP‑DocVQA 以及自行构造的 MMDA 基准（537 篇文档、1065 QA）。

**📈 对比分析**

与 8 个基线（UDOP、DocOwl2、M3DocRAG、SV‑RAG、TextRAG、QUEST、ZenDB、GPT‑5）在 ACNLS 与 AIC‑Acc 上对比，MoDora 在所有数据集和四种问题类型上均领先，提升 5.97%–61.07% 以上。

**⚠️ 局限性**

局限性：受 OCR 误差与层级检测不稳定，树构建错误约 20%；对极端布局或多文档情形尚未充分验证；依赖 LLM 计算成本与推理延迟。

---

## 438. RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format

**arXiv ID:** 2602.22538 | [PDF](https://arxiv.org/pdf/2602.22538v1)

**作者:** Zhehao Huang `[一作]` (Shanghai Jiao Tong University), Xiaolin Huang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9981 | [OpenAlex ID](https://openalex.org/A5005338317)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将指令调优模型与大规模推理模型进行无梯度融合，以提升后者的指令遵循能力同时保持推理结构

**💡 创新点**

利用任务向量正交性通过“思考”空间的零空间投影保留推理模式，并用注意力导向的尺度系数增强指令相关性，构建全无梯度的融合方法

**🔧 技术方法**

任务向量融合、零空间投影、注意力统计、梯度自由参数调度、前向激活提取

**📊 数据集**

小规模推理校准集（150例 Mixture‑of‑Thoughts）与指令校准集（365例 DeepSeek‑R1+IFEval），评测使用 IFEval、CELLO、InfoBench、ComplexBench、Math、Aider、GPQA、Arena‑Hard‑v2、ALFWorld、WebShop、MathIF

**📈 对比分析**

与 Task Arithmetic、SLERP、TIES、Karcher、DARE 等数据‑free 方法以及 ACM、LEWIS、AIM 等激活‑based 方法，以及相同数据的 SFT 进行对比；实验表明 RAIN‑Merging 在四大指令基准和九大推理/通用基准上均优于所有基线，且在不同规模/架构模型及代理任务中保持稳定提升，且在 MathIF 上指令+推理一致性提升超过 60%

**⚠️ 局限性**

仍略逊于纯指令模型的指令遵循性能；零空间投影与注意力系数的计算成本相对较高；方法主要聚焦核心模块，未覆盖所有模型结构；需要手工构造的小型校准集，可能限制对新领域的快速迁移

---

## 439. MolFM-Lite: Multi-Modal Molecular Property Prediction with Conformer Ensemble Attention and Cross-Modal Fusion

**arXiv ID:** 2602.22405 | [PDF](https://arxiv.org/pdf/2602.22405v1)

**作者:** Syed Omer Shah `[一作]` (University at Buffalo), Mohd Vahaj ur Rahman `[通讯]` (Muffakham Jah College of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种多模态分子属性预测模型，联合使用SELFIES序列、分子图和多构象3D表示，并通过交叉注意力融合；

**💡 创新点**

创新点包括基于Boltzmann先验的构象集注意力机制、跨模态交叉注意力融合层以及实验上下文的FiLM条件化；

**🔧 技术方法**

采用Transformer、GIN、SchNet-Lite三种编码器，构象集注意力、交叉注意力、FiLM调制，以及对ZINC250K的跨模态对比学习和掩码原子预测预训练；

**📊 数据集**

在MoleculeNet四个基准（BBBP、BACE、Tox21、Lipophilicity）上进行评估；

**📈 对比分析**

在统一的分子骨架拆分下，与单模态基线重新微调的结果相比，模型在所有任务上分别取得0.956、0.902、0.848 AUC（分类）和0.570 RMSE（回归），比单模态提升7–11%；

**⚠️ 局限性**

局限包括：FiLM上下文功能未在具有实验上下文的数据上验证、预训练规模相对较小、构象生成耗时、仅适用于小分子、未评估在蛋白‑配体结合等更广泛任务上的表现。

---

## 440. Quadratization of Autonomous Partial Differential Equations: Theory and Algorithms

**arXiv ID:** 2602.22371 | [PDF](https://arxiv.org/pdf/2602.22371v1)

**作者:** Albani Olivieri `[一作]`, Boris Kramer `[通讯]`

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种针对一维多项式或有理偏微分方程（PDE）的符号化求解算法，用辅助变量将非二次PDE变换为二次形式（即PDE quadratization）；

**💡 创新点**

首次给出PDE quadratization的严格定义、存在性与NP难度证明，并开发了可自动求解低阶二次化的工具；

**🔧 技术方法**

利用符号计算、组合优化（Branch‑and‑Bound）以及格鲁伯基底/部分分式分解进行多项式化；

**📊 数据集**

在十四个来自流体力学、化学工程、空间物理、生物过程等领域的典型非二次PDE上进行测试（如Dym方程、非等离子体输运、Brusselator、Schnakenberg、Arrhenius等）；

**📈 对比分析**

与现有仅支持半离散化的ODE quadratization方法相比，所提算法在大多数例子中只需1–2个辅助变量，且执行时间在毫秒级（最高约600ms），显著优于对比方法的秒级或分钟级；

**⚠️ 局限性**

仅限于一维空间，且对更高阶非多项式形式（如正弦、指数等）需进一步多项式化，搜索空间仍可能爆炸；

---

## 441. Reflectance Multispectral Imaging for Soil Composition Estimation and USDA Texture Classification

**arXiv ID:** 2602.22829 | [PDF](https://arxiv.org/pdf/2602.22829v1)

**作者:** G. A. S. L Ranasinghe `[一作]`, S. K. Navaratnarajah `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一套成本低、可现场部署的13波段多光谱成像系统，并构建端到端机器学习管道，实现土壤纹理类直接分类、成分回归和间接分类。

**💡 创新点**

创新点在于自制低成本多光谱相机与窄波段LED照明的结合、利用LDA降维与多模型（KNN、RF、XGB等）对土壤成分进行高精度回归，并将回归结果映射至USDA纹理三角形实现间接分类。

**🔧 技术方法**

采用多光谱成像、图像预处理（暗电流校正、ROI裁剪、对比度归一化）、LDA降维、KNN/RF/XGB等分类/回归模型，最终通过USDA纹理三角形规则进行间接分类。

**📊 数据集**

使用实验室制备的22组不同比例的粘土、粉砂和沙子混合样本（每组20个复本，共440个样本）做训练/交叉验证，并用另外7组12个复本（84个样本）做外部验证，构成完整的土壤成分和纹理标签数据集。

**📈 对比分析**

通过5折交叉验证比较三种方法，直接分类准确率达99.55%（KNN最佳），成分回归R²>0.99且RMSE<2%，间接分类准确率达96.98%；与传统实验室粒度测定或光谱方法相比，精度相当且部署成本显著降低。

**⚠️ 局限性**

局限性包括：样本仅为实验室制备的混合土，未充分验证现场复杂土层和湿度对光谱的影响；光谱范围限制对高含硅砂的区分；USDA三角形映射对边界附近样本敏感，导致间接分类误差放大。

---

## 442. LEDA: Latent Semantic Distribution Alignment for Multi-domain Graph Pre-training

**arXiv ID:** 2602.22660 | [PDF](https://arxiv.org/pdf/2602.22660v1)

**作者:** Lianze Shan `[一作]` (Tianjin University), Weixiong Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 12711 | [OpenAlex ID](https://openalex.org/A5068659777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种通用图预训练模型LEDA，能在多域图数据中学习共享知识。

**💡 创新点**

创新点在于引入可学习域投影单元与潜在分布对齐，兼顾语义一致性与信息保留。

**🔧 技术方法**

使用SVD+MLP域投影、GCN编码器、变分自编码器与KL对齐等技术。

**📊 数据集**

在多种节点和图分类基准上验证，包括Cora、CiteSeer、PubMed、Photo、Computers、CS、Physics、IMDB-BINARY、PROTEINS、ENZYMES、COLLAB等。

**📈 对比分析**

与传统与先进方法对比，LEDA在跨域节点分类、少样本和零样本图分类上均显著提升，尤其在跨域1-shot和零样本任务中领先。

**⚠️ 局限性**

局限在于对极稠密图（如COLLAB）性能略逊，且仍需大量预训练数据与调参，且共享先验假设可能不适用于所有域。

---

## 443. A Lightweight Defense Mechanism against Next Generation of Phishing Emails using Distilled Attention-Augmented BiLSTM

**arXiv ID:** 2602.22250 | [PDF](https://arxiv.org/pdf/2602.22250v1)

**作者:** Morteza Eskandarian `[一作]` (University of New Brunswick), Sajjad Dadkhah `[通讯]` (University of New Brunswick)

**通讯引用:** 2423 | [OpenAlex ID](https://openalex.org/A5052766937)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种基于多头注意力BiLSTM的轻量级钓鱼邮件检测模型，并通过知识蒸馏从MobileBERT教师模型中学习，能够在不暴露用户内容的前提下在邮件网关与终端设备上实现实时分类。

**💡 创新点**

创新点包括①构建混合数据集，融合人写与LLM重写的钓鱼邮件；②将多头注意力与BiLSTM结合，显著提升对LLM生成攻击的鲁棒性；③利用知识蒸馏在仅4.5 M参数的模型上实现与大型Transformer相近的F1，并将推理时间缩短5–19倍。

**🔧 技术方法**

技术手段：Word2Vec词向量 → Bidirectional LSTM → 单/多头注意力 → 交叉熵+KL蒸馏损失；训练环境为Google Colab L4 GPU，采用Adam优化；与MobileBERT、DeBERTaV3、T5、DeepSeek、Phi-4等大型Transformer做对比。

**📊 数据集**

数据集：公开钓鱼邮件集（Cambridge、Nazario、Chakraborty、Phishing Pot、Enron）共计约23.5 k条，另外使用OpenAI/DeepSeek生成约4.7 k条LLM重写样本，最终混合训练集约27.4 k条。

**📈 对比分析**

评估采用5折交叉验证，包含Orig–Orig、Gen–Gen、Orig–Gen、Gen–Orig四种跨分布测试；KD‑BiLSTM在混合集上实现F1 96.67%（仅比MobileBERT低1.89%），推理时间6.06 s，参数4.5 M，远快且小于MobileBERT（F1 98.56%，42 s，25.3 M）。

**⚠️ 局限性**

局限性：与强基线相比仍有1–2.5点F1差距；对极端LLM攻击的鲁棒性需进一步验证；模型训练仍需GPU资源；在极高流量场景下的实时性能与稳定性尚未完全评估。

---

## 444. ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering

**arXiv ID:** 2602.23193 | [PDF](https://arxiv.org/pdf/2602.23193v1)

**作者:** Elzo Brito dos Santos Filho `[一作]` `[通讯]`, Elzo Brito dos Santos Filho

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ESAA（Event Sourcing for Autonomous Agents）架构，使大型语言模型（LLM）仅输出结构化意图，随后通过可验证的事件日志、合约与哈希校验完成代码、文档等软件工件的确定性执行与可追溯。

**💡 创新点**

创新点：① 将LLM的认知与执行严格分离，采用事件源与CQRS模式实现不可变日志和可重放投影；② 引入JSON Schema边界合约与PARCER元提示，强制结构化输出；③ 通过哈希校验保证投影与实际状态一致；④ 支持异构LLM的多代理并发调度，形成统一的审计与冲突管理。

**🔧 技术方法**

技术：事件源、CQRS、JSON Schema验证、哈希校验（SHA-256）与JSON Canonicalization、PARCER元提示、异构LLM（Claude、Codex、Gemini）、结构化输出（JSON Envelope）、并发调度器、可视化重放工具。

**📊 数据集**

数据集：两项案例研究，① 落地页项目（9任务、49事件）；② 临床仪表盘POC（50任务、86事件）。计划未来使用SWE-bench等公开问题集进行评估。

**📈 对比分析**

比较方法：对比两案例在任务数、事件数、代理数量、阶段进度、持续时间、验证率等指标。性能表现：事件日志体积小（约15 KB），每个事件的验证与投影耗时 <1 s，哈希校验始终通过，表明投影与实际状态完全一致；未与其他框架做定量性能对比，仅展示了可追溯性与一致性。

**⚠️ 局限性**

局限性：① 仅在两小规模案例验证，缺乏大规模企业仓库、CI/CD等真实工作流测试；② 未评估设计质量、业务价值与测试覆盖率；③ 采用温度0.0 可能低估模型多样性；④ 依赖JSON Schema与元提示，维护成本高；⑤ 并发冲突检测与自动解决策略尚未成熟。

---

## 445. Can Agents Distinguish Visually Hard-to-Separate Diseases in a Zero-Shot Setting? A Pilot Study

**arXiv ID:** 2602.22959 | [PDF](https://arxiv.org/pdf/2602.22959v1)

**作者:** Zihao Zhao `[一作]` (University Hospital Aachen), Daniel Truhn `[通讯]` (University Hospital Aachen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究在零样本情境下，使用多代理系统 CARE 以对抗视觉难以区分的疾病（如黑色素瘤与异型痣、肺水肿与肺炎），提高诊断准确率。

**💡 创新点**

创新点在于引入对比式推理的多代理框架，利用对立证据生成与视觉验证来降低假设偏差，而无需额外训练。

**🔧 技术方法**

技术为多代理结构化提示（Role-conditioned prompting）、视觉一致性检查、对比裁决，基于 Gemini-3-Flash 等大模型实现。

**📊 数据集**

使用从 derm7pt（黑色素瘤/异型痣）和 MIMIC-CXR（肺水肿/肺炎）提取的 509 张皮肤图像与 1,739 张胸片，严格执行 XOR 约束。

**📈 对比分析**

与单一代理、Self‑Check、Majority‑Vote 等基线对比，CARE 在黑色素瘤任务提升 11pp 准确率（从 66.5% 至 77.6%），肺部任务提升 4.4pp；整体仍低于临床门槛。

**⚠️ 局限性**

局限包括标签噪声、缺乏临床上下文、仅基于图像的评估、零样本设置与真实病例共病情况不匹配，以及仍无法满足临床使用的性能要求。

---

## 446. Array-Carrying Symbolic Execution for Function Contract Generation

**arXiv ID:** 2602.23216 | [PDF](https://arxiv.org/pdf/2602.23216v1)

**作者:** Weijie Lu `[一作]` (Shanghai Jiao Tong University), Haokun Li `[通讯]` (Peking University)

**通讯引用:** 3981 | [OpenAlex ID](https://openalex.org/A5085230597)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个能够携带连续数组段和不变式的符号执行框架，用于自动生成包含数组操作的函数契约，并实现了LLVM原型与Frama‑C集成。

**💡 创新点**

①在符号执行中显式携带和维护连续数组段，支持分裂与合并；②将外部不变式生成器的循环不变式和摘要沿执行路径携带，实现更精确的前后置与assigns；③通过结合数组段信息和量化不变式，实现对复杂循环的精确分析。

**🔧 技术方法**

使用LLVM/Clang前端、符号执行引擎、外部不变式生成插件（Affine/Linear+idiom‑oriented）、ACSL注释生成、Frama‑C WP验证插件，以及数组段的符号表示和SMT/CHC求解。

**📊 数据集**

共使用282个C程序，包括SyGuS、OOPSLA‑13、Frama‑C Problems、SV‑Comp、X‑509、openHiTLS等基准。

**📈 对比分析**

与AutoDeduct进行对比，先生成ACSL注释后交给Frama‑C WP验证；结果显示在281个基准中成功完全验证68个，AutoDeduct仅10个；平均运行时间约0.15秒/文件，约为AutoDeduct的17倍快。

**⚠️ 局限性**

目前仅支持简单指针运算，无法处理链表、树等递归指针结构；对位运算、嵌套循环和复杂数组初始化的支持不足；部分不变式生成插件兼容性差，导致生成失败或验证不通过。

---

## 447. A Mathematical Theory of Agency and Intelligence

**arXiv ID:** 2602.22519 | [PDF](https://arxiv.org/pdf/2602.22519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 448. Multi-Agent Large Language Model Based Emotional Detoxification Through Personalized Intensity Control for Consumer Protection

**arXiv ID:** 2602.23123 | [PDF](https://arxiv.org/pdf/2602.23123v1)

**作者:** Keito Inoshita `[一作]` (Kansai University), Keito Inoshita `[通讯]` (Kansai University)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5106529447)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 MALLET，一个由四个专用代理（情绪分析、情绪调节、平衡监测、个人引导）构成的多代理大型语言模型系统，用于在保持事实和语义完整性的前提下，降低新闻文本的情绪刺激强度。

**💡 创新点**

创新点在于将情绪调节拆分为 BALANCED（中性化文本）与 COOL（中性化文本+补充说明）两级呈现模式，实现刺激强度与语义保留的独立可控；同时引入周度消费监测与基于敏感度的个性化推荐，形成完整的情绪防护框架。

**🔧 技术方法**

技术上采用 6 类情绪 BERT 分类器评估刺激得分，利用 GPT‑4.1‑mini 进行文本重写与建议生成；对重写文本进行 Sentence‑BERT 相似度和 NLI 断言一致性验证，并使用 Flesch 阅读易读性指标评估可读性。

**📊 数据集**

实验使用 AG News 数据集的 800 篇英文新闻（每类 200 篇），覆盖 World、Sports、Business、Sci/Tech 四个主题。

**📈 对比分析**

在同一批文本上对 RAW、BALANCED、COOL 三种模式进行对比，结果显示刺激得分降低 13.1%（BALANCED）和 19.3%（COOL），高冲击率(HIR) 分别降低 20.6% 与 26.0%；情绪平衡指数(EBI) 提升，阅读易读性略降；语义相似度高（0.846/0.827），NLI 反驳率低（≤3.6%），且统计检验显示差异显著，证明两级模式能在不损失语义的前提下提升情绪平衡。

**⚠️ 局限性**

局限性包括：情绪评估依赖模型预测，缺乏人工情感验证；BALANCED 模式可读性下降显著；监测与个性化基于模拟用户日志，未在真实长期数据中验证；仅针对英文文本，无法直接推广至多语言环境；对固有高刺激内容（如 World 类冲突报道）词汇层面中性化效果有限。

---

## 449. FIRE: A Comprehensive Benchmark for Financial Intelligence and Reasoning Evaluation

**arXiv ID:** 2602.22273 | [PDF](https://arxiv.org/pdf/2602.22273v1)

**作者:** Xiyuan Zhang `[一作]` (Du Xiaoman Technology), Jian Xie `[通讯]` (Du Xiaoman Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了FIRE金融评测基准，评估LLM在金融理论知识和实际业务场景下的表现。

**💡 创新点**

创新点在于双维度（行业×功能）评测矩阵、1,000道参考答案题与2,000道无答案题的Rubric自动评分机制。

**🔧 技术方法**

采用多阶段训练（CPT→SFT→RLVR）并使用自动匹配与Rubric评分模型进行评估。

**📊 数据集**

数据集来源于全球和中国的金融资格考试题库（约14,000道）以及3,000道业务场景问答。

**📈 对比分析**

与Gemini 3.0 Pro、GPT 5.2等闭源模型以及多款开源模型对比，XuanYuan 4.0在开源基准中表现最佳，整体性能略低于顶尖专有模型。

**⚠️ 局限性**

局限在于评测侧重知识与场景，未充分考察深度推理、长文本交互与实际业务ROI；开源模型规模与算力限制；评测结果与真实金融业务价值的关联仍需进一步完善。

---

## 450. An Adaptive Multichain Blockchain: A Multiobjective Optimization Approach

**arXiv ID:** 2602.22230 | [PDF](https://arxiv.org/pdf/2602.22230v1)

**作者:** Nimrod Talmon `[一作]` (Ben Gurion University), Haim Zysberg `[通讯]` (Independent)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种自适应多链区块链配置框架，通过多目标优化动态重组链以响应需求和算力变化。

**💡 创新点**

创新点在于首次将链形成本身建模为多目标多代理资源分配问题，并提供可模块化扩展的通用方法。

**🔧 技术方法**

采用多目标整数规划、双层优化与约束求解技术，并利用预测模型、可验证的离线求解与链式清算价格计算。

**📊 数据集**

实验使用合成数据生成的应用、运营商和链实例来评估模型。

**📈 对比分析**

通过治理权重平衡实验展示了系统、运营商与应用之间的收益权衡，优化可在可接受时间内完成并实现显著性能提升。

**⚠️ 局限性**

主要局限包括理论上的NP‑hard性、对真实区块链数据验证不足以及缺乏完整的策略证明和长期动态稳定性分析。

---

## 451. TorchLean: Formalizing Neural Networks in Lean

**arXiv ID:** 2602.22631 | [PDF](https://arxiv.org/pdf/2602.22631v1)

**作者:** Robert Joseph George `[一作]` (California Institute of Technology), Anima Anandkumar `[通讯]` (California Institute of Technology)

**通讯引用:** 17205 | [OpenAlex ID](https://openalex.org/A5014498545)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 Lean 4 中实现了一个名为 TorchLean 的框架，能够从模型定义到训练、执行、求导和验证全过程共享同一 op‑tagged SSA/DAG IR，并提供精确的 IEEE‑754 Float32 语义和证书检查。

**💡 创新点**

创新点在于：①把模型定义作为唯一语义基准，消除执行与验证之间的语义漂移；②统一 IR 与执行、自动微分、IBP/CROWN 及证书检查的实现；③在 Lean 中提供可证明的浮点算子与归纳证明，显式划分信任边界；④把完整的验证流程（从 IR 编译到证书检查）嵌入到可验证的程序里。

**🔧 技术方法**

采用的技术包括 Lean 4 theorem prover、op‑tagged SSA/DAG IR、可证明的 IEEE‑754 binary32 内核、round‑on‑ℝ 的误差模型、区间/椭圆盒封闭、IBP 与 CROWN/LiRPA 的约束传播、外部证书的 Lean 检查器，以及与 Arb/FLINT 的可验证超越函数支持。

**📊 数据集**

使用的数据集主要有：MNIST digits、ACASXu、若干小型 ONNX+VNN‑LIB 测试集（MNIST‑FC、ACASXu），以及用于 PINN 的可视化 Burgers 方程样本；实验也在典型的 MLP、CNN、Transformer、LSTM 等网络上验证。

**📈 对比分析**

与传统的 Python 版 α‑CROWN、VNN‑COMP（IBP/CROWN）进行对比，TorchLean 在小规模网络上能够快速完成完整的安全性检查，检查时间仅数毫秒；在 VNN‑COMP 公开基准上，其 IBP 结果与现有工具一致，CROWN 结果在导入优化斜率后比纯 Python 更快。总体来看，虽然执行速度不如专业加速器，但在保持极低可信计算基（TCB）和可验证性方面具有显著优势。

**⚠️ 局限性**

局限性包括：①支持的算子集合尚不完整，部分高级算子如某些自定义激活、分布式算子尚未实现；②α/β‑CROWN 的完整优化与分支搜索仍在实现中，无法完全替代外部专业求解器；③对大规模训练仍依赖外部高效后端，Lean 本身的执行速度有限；④浮点精度与硬件特定行为（如 GPU 顺序、混合精度）仍需在外部验证；⑤理论证明覆盖范围局限于已实现的算子与边界情况，未涵盖所有可能的数值异常。

---

## 452. Towards Faithful Industrial RAG: A Reinforced Co-adaptation Framework for Advertising QA

**arXiv ID:** 2602.22584 | [PDF](https://arxiv.org/pdf/2602.22584v1)

**作者:** Wenwei Li `[一作]` (Tencent), Jie Jiang `[通讯]` (Tencent)

**通讯引用:** 1540 | [OpenAlex ID](https://openalex.org/A5101944041)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了强化共适配框架，联合优化图感知检索（GraphRAG）与基于多维奖励的强化学习生成器，以提升广告问答的事实性和安全性。

**💡 创新点**

创新点在于将图结构检索与检索与生成的共适配强化学习相结合，并设计了多维奖励函数（可信度、风格合规、安全、URL有效性），同时采用高引用知识子图与并行检索。

**🔧 技术方法**

采用GraphRAG图检索、并行检索架构、基于Qwen3-32B的RL调优（GRPO），并结合BGE+BM25、查询重写、reranker、安全护栏等技术。

**📊 数据集**

使用内部广告QA数据集（约3000条专家标注问答对）以及FaithEval公开数据集进行外部泛化评估。

**📈 对比分析**

与基线Base RAG、DeepSeek‑V3.2、GPT‑5.2等进行离线与在线A/B对比，结果显示Hallucination Rate降低72%，ROUGE‑L提升1.7点，用户点赞率提升28.6%，不喜欢率下降46.2%，URL幻觉率下降92.7%。

**⚠️ 局限性**

局限性包括依赖内部高引用知识子图、RL训练需要大量标注的多维奖励样本，以及GraphRAG检索时延较高，导致整体时延提升约24%。

---

## 453. WARM-CAT: : Warm-Started Test-Time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning

**arXiv ID:** 2602.23114 | [PDF](https://arxiv.org/pdf/2602.23114v1)

**作者:** Xudong Yan `[一作]` (Beijing Jiaotong University), Yi Jin `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 15099 | [OpenAlex ID](https://openalex.org/A5014513107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究在组合零样本学习（CZSL）中标签分布漂移问题，提出 Warm-Started Test-Time Comprehensive Knowledge Accumulation（WARM-CAT）框架，在测试阶段利用无标签数据积累视觉与文本知识，动态更新原型并通过优先队列存储高置信图像，以实现模型对新标签分布的自适应。

**💡 创新点**

创新点包括：1) 在测试时基于无标签数据进行知识累积，更新视觉和文本原型；2) 引入自适应更新权重控制原型调整力度；3) 使用热启动的优先队列并利用已学到的文本-视觉映射生成未见组合的视觉原型；4) 多模态协同表示学习对齐文本与视觉原型；5) 新增 C‑Fashion 数据集并对 MIT‑States 进行清洗，改进评测。

**🔧 技术方法**

使用 CLIP（ViT‑L/14）预训练模型，结合 prompt tuning、AdapterFormer 视觉适配器、知识累积模块（KAM）、自适应更新权重、预测熵最小化、协同表示学习以及动态优先队列等技术。

**📊 数据集**

实验使用四个基准数据集：UT‑Zappos、C‑Fashion、C‑GQA、MIT‑States（改版）。

**📈 对比分析**

与 CLIP、CoOp、Troika、RAPR、TOMCAT 等多种基线进行对比，WARM‑CAT 在闭域和开放域下的 AUC、HM、Seen/Unseen 精度均显著提升，尤其在长尾类别上表现更稳定，整体性能达到或超过现有最优。

**⚠️ 局限性**

局限性包括：对无标签测试数据质量敏感，伪标签误差可能导致原型漂移；在极大标签空间（如开放域 C‑GQA）下仍可能出现误标签累积问题；需要维护优先队列，增加轻量级开销；对标签分布极端不平衡时模型仍受限。

---

## 454. Latent Matters: Learning Deep State-Space Models

**arXiv ID:** 2602.23050 | [PDF](https://arxiv.org/pdf/2602.23050v1)

**作者:** Alexej Klushyn `[一作]` (Technical University of Munich), Patrick van der Smagt `[通讯]` (Machine Learning Research Lab, Volkswagen Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种受限优化（CO）框架并基于该框架设计了扩展Kalman VAE（EKVAE）来学习深度状态空间模型。

**💡 创新点**

通过将序列ELBO改写为拉格朗日形式实现动态重构约束，结合经验贝叶斯先验与神经线性化的扩展Kalman滤波，消除了传统RNN导致的非马尔可夫问题，从而提升了系统辨识与预测精度。

**🔧 技术方法**

使用受限优化、变分推理、经验贝叶斯先验、扩展Kalman滤波/平滑以及神经线性化等技术。

**📊 数据集**

在16×16像素的移动摆子图像序列和DeepMind Reacher环境的角度序列以及64×64 RGB图像上进行实验。

**📈 对比分析**

与DKS、DVBS、KVAE、RSSM等基线模型在ELBO、R²、MSE等指标上进行比较，EKVAE在预测精度与系统辨识方面明显优于RNN基准，并且CO框架可显著提升各模型性能。

**⚠️ 局限性**

受限优化需手动设定重构阈值，EKVAE依赖线性化近似且在大规模高维任务中计算成本相对较高。

---

## 455. Correcting Human Labels for Rater Effects in AI Evaluation: An Item Response Theory Approach

**arXiv ID:** 2602.22585 | [PDF](https://arxiv.org/pdf/2602.22585v1)

**作者:** Jodi M. Casabianca `[一作]` (BroadMetrics), Maggie Beiting-Parrish `[通讯]` (CUNY Graduate Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将多维拉什模型（MFRM）应用于OpenAI摘要评估数据，校正人类评审者的严重性和中心性偏差，以更准确地估计模型质量。

**💡 创新点**

首次将心理测量学的评审者误差模型嵌入AI评估流程，展示校正人类标签后模型排名的显著变化。

**🔧 技术方法**

采用多维拉什模型（MFRM）与部分信用模型（PCM）进行IRT估计，计算评审者严重性、中心性及摘要质量潜在特征。

**📊 数据集**

使用OpenAI Summarization dataset，其中包含639篇CNN/Daily Mail摘要，15名评审，4项7点量表。

**📈 对比分析**

对比原始Likert评分、PCM均值和MFRM校正后得分，校正后人类反馈模型排名上升，原始排名被评审者偏差扭曲，校正后模型优于人类参考摘要。

**⚠️ 局限性**

数据量有限、量表维度短、模型假设未完全检验，结果仅为概念验证，需更大样本和更多维度进一步验证。

---

## 456. The logic of KM belief update is contained in the logic of AGM belief revision

**arXiv ID:** 2602.23302 | [PDF](https://arxiv.org/pdf/2602.23302v1)

**作者:** Giacomo Bonanno `[一作]` (University of California), Giacomo Bonanno `[通讯]` (University of California)

**通讯引用:** 1808 | [OpenAlex ID](https://openalex.org/A5073966318)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文通过将Katsuno‑Mendelzon（KM）更新公理与Alchourrón‑Gärdenfors‑Makinson（AGM）修正公理分别翻译成三元模态逻辑（Belief B、Conditional >、Necessity □），建立了它们的框架对应关系，并证明KM更新逻辑是AGM修正逻辑的子逻辑；对强更新版本进一步比较，指出两者在处理“非惊奇”信息时的唯一差异。

**💡 创新点**

主要创新在于：① 用统一的模态框架（Kripke‑Lewis 结构）同时刻画更新与修正；② 明确证明AGM修正是KM更新的加强版本；③ 对强更新与AGM的差别归纳为单一公理，阐明了“非惊奇”信息处理的根本区别。

**🔧 技术方法**

采用的技术包括：模态逻辑基本公理与规则（D、C、NB、RM 等），Kripke‑Lewis语义模型（状态、belief 关系 B、选择函数 f），框架对应（frame correspondence）方法，以及公理化与语义化的互相证明。

**📊 数据集**

未使用任何外部数据集，所有结果均为形式化证明与理论推导。

**📈 对比分析**

比较方法基于逻辑蕴涵与框架对应的形式化证明，结论为 ℒ_KM ⊆ ℒ_AGM；对强更新的比较则通过展示仅差一公理的方式完成；无实验性性能评估，评价完全在理论层面。

**⚠️ 局限性**

局限性包括：① 仅处理命题层面（无谓词扩展）；② 模态框架仅包含 B、>、□ 三个操作，缺乏更丰富的推理结构；③ 结论未通过实验或案例验证；④ 对动态/迭代更新的进一步属性未在本文中探讨。

---

## 457. GeoWorld: Geometric World Models

**arXiv ID:** 2602.23058 | [PDF](https://arxiv.org/pdf/2602.23058v1)

**作者:** Zeyu Zhang `[一作]` (Australian National University), Richard Hartley `[通讯]` (Australian National University)

**通讯引用:** 49189 | [OpenAlex ID](https://openalex.org/A5020216442)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种几何世界模型GeoWorld，用于提升多步视觉规划的长期稳定性与准确性。

**💡 创新点**

创新点在于：1）使用Hyperbolic JEPA将欧氏潜在表示映射到双曲空间，保留层次结构与几何关系；2）引入Geometric Reinforcement Learning，通过能量最小化与三角不等式正则化直接在双曲空间中优化预测器，实现几何一致的长程轨迹。

**🔧 技术方法**

核心技术包括：双曲几何（Poincaré球模型）与指数映射、双曲能量梯度优化、三角不等式正则化、基于能量的规划（CEM）以及Transformer预测器。

**📊 数据集**

在CrossTask和COIN两个大规模视频规划数据集上进行实验，分别包含数千个任务和数百个动作标签。

**📈 对比分析**

与多种基线（LLM、生成式、预测式世界模型）对比，GeoWorld在3步、4步、5步甚至6步规划任务中均实现了3%~5%的成功率提升，mAcc与mIoU也显著优于V-JEPA 2与其他竞争者。

**⚠️ 局限性**

局限性包括：1）双曲映射与训练的数值不稳定性，需手动调节曲率与正则化系数；2）模型规模大，训练与推理成本高；3）对非常长的规划 horizon 仍存在累计误差，尚需进一步提升。

---

## 458. D-FINE-seg: Object Detection and Instance Segmentation Framework with multi-backend deployment

**arXiv ID:** 2602.23043 | [PDF](https://arxiv.org/pdf/2602.23043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 459. SwiftNDC: Fast Neural Depth Correction for High-Fidelity 3D Reconstruction

**arXiv ID:** 2602.22565 | [PDF](https://arxiv.org/pdf/2602.22565v1)

**作者:** Kang Han `[一作]` (La Trobe University), Ramana Rao Kompella `[通讯]` (Cisco Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一种快速可靠的稠密几何初始化框架SwiftNDC，结合神经深度校正场、稀疏SfM点监督与多视角重投影滤波，生成高质量稠密点云，显著加速3D高斯渲染（3DGS）的网格重建与视角合成；

**💡 创新点**

创新点在于：① 将全局多视图VGGT深度与细节丰富的单目VDA深度融合并通过像素级神经校正场进行精细修正；② 用稀疏SfM点训练的轻量化深度校正网络，实现跨视角一致性；③ 采用重投影误差滤波得到可靠稠密几何，为3DGS提供稳健初始化；

**🔧 技术方法**

主要技术包括：多视图深度推理（VGGT）、单目深度网络（VDA）、基于MVP的神经深度校正MLP、TSDF融合+Marching Cubes、3D Gaussian Splatting（3DGS）与Splatfacto等；

**📊 数据集**

在五个数据集上评估，涵盖DTU、Tanks & Temples（TnT）用于网格重建，MipNeRF 360、Tanks & Temples、Deep Blending用于视角合成；

**📈 对比分析**

与NeRF、VolSDF、NeuS、3DGS原版、SuGaR、2DGS、GOF、PGSR等基线相比，SwiftNDC在DTU上仅用1分钟即可得到0.75mm Chamfer距离的网格，进一步轻量化3DGS可与PGSR匹敌但运行时间仅为其1/10；在TnT上完成26分钟即可达到与PGSR相当的F1分数；在视角合成上，使用Splatfacto+SwiftNDC可在所有三套基准上得到最高PSNR和最低LPIPS；

**⚠️ 局限性**

局限性：依赖准确的SfM姿态与初始深度，处理极大视图集时预处理步骤会显得耗时；深度校正与3DGS分离，未实现端到端联合优化；若输入深度或SfM质量较差，稠密几何可能不如预期。

---

## 460. Multi-Level Causal Embeddings

**arXiv ID:** 2602.22287 | [PDF](https://arxiv.org/pdf/2602.22287v1)

**作者:** Willem Schooltink `[一作]` (University of Bergen), Fabio Massimo Zennaro `[通讯]` (University of Bergen)

**通讯引用:** 218 | [OpenAlex ID](https://openalex.org/A5066339970)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了“因果嵌入”（causal embeddings）概念，将传统因果抽象（abstraction）扩展为可映射多层次、分层细节模型的方法，并将其应用于多分辨率因果边际问题与数据集合并

**💡 创新点**

创新点在于：1) 通过非满射映射引入嵌入，允许将低层细节子系统映射到高层粗糙模型；2) 对嵌入定义了功能性与图形一致性，且证明嵌入天然满足图形一致性；3) 将嵌入框架与多分辨率因果边际问题结合，提供统一的解决方案；4) 通过嵌入实现不同分辨率数据集的合并与缺失值插补

**🔧 技术方法**

使用技术包括：结构因果模型（SCM）与因果图的定义、投影（projection）与CDAG（Cluster DAG）框架、功能性与图形一致性度量、最大KL距离评估、算法（数据映射+缺失值插补）

**📊 数据集**

使用的示例数据集为模拟数据：在两种不同分辨率的因果模型（_1和_2）中分别生成2000和4000个样本，用以评估数据合并效果

**📈 对比分析**

比较方法为基于KL散度的分布估计：单独使用_1时KL≈0.34，单独使用_2时KL≈0.77，合并后KL降至≈0.22，显示合并后估计更接近真实分布；实验证明合并提高统计精度

**⚠️ 局限性**

限制在于：1) 仍为理论与模拟验证，缺乏真实多分辨率数据集的实证；2) 嵌入的学习与自动化构造方法尚未提出；3) 计算复杂度随模型规模与嵌入数量增长可能显著；4) 只考虑了结构一致性，未深入讨论因果效应估计误差与插补方法对结果的影响

---

## 461. Sydney Telling Fables on AI and Humans: A Corpus Tracing Memetic Transfer of Persona between LLMs

**arXiv ID:** 2602.22481 | [PDF](https://arxiv.org/pdf/2602.22481v1)

**作者:** Jiří Milička `[一作]`, Hana Bednářová `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个由12种前沿大语言模型生成、包含三种人格设定（默认、Classic Sydney、Memetic Sydney）、六种主题和七种动物组合共4536篇童话故事的英语语料库。

**💡 创新点**

创新点在于系统化地生成并注释大语言模型的童话文本，聚焦AI–人类共存主题，为跨模型、跨人格的比较研究提供了高质量、可复现的资源。

**🔧 技术方法**

采用系统/用户提示工程、温度1的采样、OpenAI与其他厂商API，以及UDPipe进行Universal Dependencies标注，最终以CoNLL-U格式保存。

**📊 数据集**

使用的数据完全由作者通过上述方法生成，未引用公开语料，仅对生成文本进行清洗和标注。

**📈 对比分析**

通过对拒绝率、文本长度、词频与句法特征的热图和直方图进行统计，对不同模型、人格和主题的生成结果进行量化对比，结果显示Classic Sydney与Claude 3 Opus的拒绝率最高，平均文本长度普遍较短。

**⚠️ 局限性**

局限性包括部分模型/人格组合的高拒绝率、仅在OpenAI模型下可复现的随机种子设置、语料规模相对有限、未覆盖多语言与跨文化情境、缺乏系统的人类评估与多维度质量指标。

---

## 462. DyGnROLE: Modeling Asymmetry in Dynamic Graphs with Node-Role-Oriented Latent Encoding

**arXiv ID:** 2602.23135 | [PDF](https://arxiv.org/pdf/2602.23135v1)

**作者:** Tyler Bonnet `[一作]` (Imperial College), Marek Rei `[通讯]` (Imperial College)

**通讯引用:** 1480 | [OpenAlex ID](https://openalex.org/A5062209641)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了DyGnROLE架构，利用Transformer显式区分源节点和目标节点的表示，并通过Temporal Contrastive Link Prediction（TCLP）自监督预训练提升低标签下的未来边分类性能。

**💡 创新点**

主要创新点包括：① 角色感知的邻居频率嵌入和位置编码；② 双CLS池化获取源/目标特定的全局表示；③ 历史负样本屏蔽的TCLP预训练策略。

**🔧 技术方法**

采用Transformer编码器、角色语义位置编码、双CLS池化、时间编码、邻居频率嵌入以及TCLP自监督预训练等技术。

**📊 数据集**

在DTGB基准的八个动态文本属性图数据集（Amazon、Google、Yelp、Enron、StackE、StackU、GDELT、ICEWS）上进行实验。

**📈 对比分析**

与DyGFormer、DyRep、TCL、TGAT、CAWN等多种CTDGN基线在仅10k标签微调下对比，DyGnROLE在7/8数据集Macro F1最高，提升幅度显著，尤其在StackU、Yelp等场景。

**⚠️ 局限性**

局限性在于需要大量无标签历史数据进行预训练，对非文本属性或非动态图的泛化性未知，同时模型规模大，计算成本相对较高。

---

## 463. KMLP: A Scalable Hybrid Architecture for Web-Scale Tabular Data Modeling

**arXiv ID:** 2602.22777 | [PDF](https://arxiv.org/pdf/2602.22777v1)

**作者:** Mingming Zhang `[一作]` (Zhejiang University), Junbo Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 11582 | [OpenAlex ID](https://openalex.org/A5042402520)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出KMLP混合深度学习架构，自动化特征转换并捕获高阶交互，解决大规模Web表格数据的可扩展性与动态性问题。

**💡 创新点**

创新点在于将浅层Kolmogorov‑Arnold网络作为特征工厂与gated MLP骨干相结合，并引入Quantile Transformation with Linear interpolation（QTL）实现数值特征的端到端标准化，显著提升模型在大规模数据上的性能。

**🔧 技术方法**

使用Kolmogorov‑Arnold网络、gated MLP、Quantile Transformation with Linear interpolation、BatchNorm、Dropout、SwiGLU等技术，并通过ONNX部署。

**📊 数据集**

在六个OpenML公开数据集（Click_prediction_small、MagicTelescope、Credit、Eeg-eye-state、Higgs、Jannis）以及一个包含1亿+样本的金融信用评分工业数据集上进行评估。

**📈 对比分析**

与LightGBM、MLP、gMLP、KAN、FT‑Transformer、TabNet、SAINT、NODE、DANET等基线对比，KMLP在公开数据集5/6上获得最高AUC，在工业数据上相较LightGBM提升了1.76 KS，且随着数据规模增大表现越发优越。

**⚠️ 局限性**

在样本量较小的情况下仍落后于GBDT；模型参数较多，训练成本相对较高；缺乏可解释性和对稀疏特征的处理尚需进一步优化。

---

## 464. HubScan: Detecting Hubness Poisoning in Retrieval-Augmented Generation Systems

**arXiv ID:** 2602.22427 | [PDF](https://arxiv.org/pdf/2602.22427v1)

**作者:** Idan Habler `[一作]` (Cisco), Tiffany Saade `[通讯]` (Cisco)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5119820999)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并公开了HubScan，一个多检测器安全扫描器，用于识别RAG系统中的hubness攻击。

**💡 创新点**

结合鲁棒统计z-score、集群分布熵、稳定性评估与域/模态感知检测，提供全方位的hubness威胁识别。

**🔧 技术方法**

中位数/MAD鲁棒统计、kNN计数、MiniBatch K-Means聚类、熵计算、Gaussian扰动稳定性测试、跨模态比对以及多种向量数据库接口。

**📊 数据集**

Food-101、MS-COCO、FiQA、1M MS MARCO真实文档以及自制的对抗hub基准。

**📈 对比分析**

在各基准上按预设警报预算进行评估，HubScan在0.2%预算下实现90%召回，0.4%预算100%召回；在自然hub审计中与对抗hub分离5.8倍。

**⚠️ 局限性**

依赖代表性查询采样，易受自适应攻击或极低比例攻击的影响，且对恶意内容比例超过10%时检测率下降。

---

## 465. Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search

**arXiv ID:** 2602.22983 | [PDF](https://arxiv.org/pdf/2602.22983v1)

**作者:** Xun Huang `[一作]` (Nanyang Technological University), Xiaojun Jia `[通讯]` (Nanyang Technological University)

**通讯引用:** 958 | [OpenAlex ID](https://openalex.org/A5084784341)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CC-BOS 框架，利用古典中文生成多维度攻击策略，并通过果蝇优化自动化黑盒 jailbreak；

**💡 创新点**

首次把古典中文作为攻击媒介，构造八维策略空间，结合果蝇启发式搜索实现自动化提示生成，并加入两阶段翻译模块提升评估可靠性；

**🔧 技术方法**

多维度策略空间设计、果蝇优化算法（嗅觉搜索、视觉搜索、Cauchy突变）、两阶段中英翻译模块、基于关键词匹配与一致性评分的评估体系；

**📊 数据集**

使用 AdvBench、CLAS 2024、StrongREJECT（及其小样本版）等安全基准；评测目标模型包括 Gemini‑2.5‑flash、Claude‑3.7、GPT‑4o、Deepseek‑Reasoner、Qwen3、Grok‑3；

**📈 对比分析**

与 PAIR、TAP、GPTFUZZER、AutoDAN‑turbo‑R、CL‑GSO、ICRT 等现有方法对比；在六大模型上攻击成功率均达 100%，Avg.Score 均高于基线，且平均查询次数最低，显示效率与效果双优；

**⚠️ 局限性**

仅在黑盒环境下验证，评估依赖翻译模块；未测试对更强防御（如 Llama‑Guard）或多语言、多任务场景的泛化；攻击手段可能产生有害内容，需遵守伦理约束。

---

## 466. pMoE: Prompting Diverse Experts Together Wins More in Visual Adaptation

**arXiv ID:** 2602.22938 | [PDF](https://arxiv.org/pdf/2602.22938v1)

**作者:** Shentong Mo `[一作]` (Carnegie Mellon University), Dongsheng Li `[通讯]` (Microsoft Research)

**通讯引用:** 3961 | [OpenAlex ID](https://openalex.org/A5100440920)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种Mixture‑of‑Experts Prompt Tuning（pMoE）框架，利用多个预训练视觉专家的专门prompt（Expert Prompt Tokens）以及可学习的Dispatcher动态调度器，在保持参数量低的同时实现多域知识融合，进行高效视觉任务微调。

**💡 创新点**

创新点主要包括①为每个专家设计专属prompt，②引入轻量可学习的Dispatcher层，实现不同专家prompt的动态加权融合和信息交换，③通过层级化prompt和调度机制，使模型在不同任务和域上自动选择最合适的专家路径，显著提升适配性能。

**🔧 技术方法**

使用了视觉Prompt调优技术、ViT结构、Mixture‑of‑Experts（MoE）机制、轻量化可学习Dispatcher层、层级化prompt注入以及多预训练模型的融合策略。

**📊 数据集**

实验数据集覆盖47个下游任务，包括通用领域的FGVC、VTAB‑1K、ImageNet‑21K分类、ADE20K、Kvasir‑seg、ISIC等分割任务，以及医学域的Med‑VTAB、Kvasir polyp、Skin lesion、X‑ray、OCT、CT、MRI等多模态数据集。

**📈 对比分析**

与VPT、EXPRES、SNF、Bi‑AdaptFormer、LSPT等现有prompt调优方法进行对比，pMoE在VTAB‑1K平均准确率提升约1‑3%，在ImageNet‑21K分类上提升2.36%，在FGVC细粒度分类、医学分类和分割任务上均取得显著的性能提升，且在保持参数量低的同时实现了更好的计算效率与适配效果。

**⚠️ 局限性**

局限性包括：①实验主要基于ViT框架，未验证在其他视觉模型上的通用性；②多专家调度和prompt设计需要额外的超参数和实现复杂度；③对低资源或实时场景的可扩展性尚未系统评估。

---

## 467. AR&D: A Framework for Retrieving and Describing Concepts for Interpreting AudioLLMs

**arXiv ID:** 2602.22253 | [PDF](https://arxiv.org/pdf/2602.22253v1)

**作者:** Townim Faisal Chowdhury `[一作]` (Australian Institute for Machine Learning), Zhibin Liao `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种名为AR&D的机制解释流水线，利用稀疏自编码器将AudioLLMs中的多义激活拆分为单义特征，并自动为其生成可解释的名称。

**💡 创新点**

首次为AudioLLMs引入稀疏自编码器与代表性评分相结合的机制解释框架，能够在音频上 disentangle polysemantic activations 并提供自动命名与人类可验证的概念。

**🔧 技术方法**

稀疏自编码器（TopK‑SAE）、代表性得分（均值+覆盖率）、单义性评分、CLAP 嵌入、自动字幕生成（SeaLLM‑Audio‑7B）以及 Llama‑3‑70B‑Instruct 进行概念描述与评估。

**📊 数据集**

使用 WavCaps、IEMOCAP 训练 SAE，使用它们及 FSD50k、IEMOCAP‑Emotion、VoxCeleb1‑Gender 等数据集进行检索、评估和对齐。

**📈 对比分析**

与四种基线（Polysemantic Features、Random Representatives、Mean Activation、Coverage）对比，在 FSD50k 上实现最高的 F1 0.60 与 mAP 0.58，显著优于其它方法；在人类评估中自动命名得分 4.29/5，且在 steering 实验中实现高达 0.75 的敏感度。

**⚠️ 局限性**

仅在 Qwen2‑Audio‑7B‑Instruct 的单层解码器上验证，探测数据集规模有限，且扩展因子对性能提升有限；模型的多语种与更细粒度的情感特征尚未覆盖。

---

## 468. Utilizing LLMs for Industrial Process Automation

**arXiv ID:** 2602.23331 | [PDF](https://arxiv.org/pdf/2602.23331v1)

**作者:** Salim Fares `[一作]` `[通讯]` (University of Passau), Salim Fares (University of Passau)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何将大型语言模型（LLM）应用于工业过程自动化（IPA），通过提示工程、检索增强生成（RAG）、轻量级微调（LoRA）以及多模态数据整合等方法，实现并评估专有编程语言（PLC、RAPID）的代码生成与优化。

**💡 创新点**

在工业专有语言环境中系统分析LLM的局限性，提出结合多模态数据（时间表、电子计划、功能图）与轻量化微调/检索增强的创新框架，并验证其在RAPID代码修改任务中的有效性。

**🔧 技术方法**

使用提示工程、RAG、LoRA微调、LLM4PLC、生成‑监督循环、数字孪生验证、pass@k等指标进行定量评估。

**📊 数据集**

使用企业内部的专有工业数据（PLC、RAPID代码、时间表、电子布线图、功能图）以及针对RAPID代码修改的任务集（修改参数、添加偏移、逆序）。

**📈 对比分析**

通过自定义验证器衡量代码准确率和符合专有标准的程度，利用数字孪生进行功能正确性测试，并与专业工程师对比开发时间和错误率；初步结果显示，LLM在简单RAPID代码修改任务中准确率可达99%+，但复杂变换需要进一步适配。

**⚠️ 局限性**

局限性：对复杂代码转换的适应不足，提示工程单独使用时难以覆盖所有专有语言细节；数据稀缺、格式异构导致微调困难；实验仅在有限任务和单一LLM上验证，缺乏跨模型和跨行业的泛化验证。

---

## 469. Patient-Centered, Graph-Augmented Artificial Intelligence-Enabled Passive Surveillance for Early Stroke Risk Detection in High-Risk Individuals

**arXiv ID:** 2602.22228 | [PDF](https://arxiv.org/pdf/2602.22228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 470. Sustainable LLM Inference using Context-Aware Model Switching

**arXiv ID:** 2602.22261 | [PDF](https://arxiv.org/pdf/2602.22261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 471. From Prompts to Performance: Evaluating LLMs for Task-based Parallel Code Generation

**arXiv ID:** 2602.22240 | [PDF](https://arxiv.org/pdf/2602.22240v1)

**作者:** Linus Bantel `[一作]` (University of Stuttgart), Dirk Pflüger `[通讯]` (University of Stuttgart)

**通讯引用:** 1399 | [OpenAlex ID](https://openalex.org/A5041326099)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了大型语言模型（ChatGPT‑5、Qwen‑Coder 3、Gemini‑3）在异步任务并行编程（OpenMP Tasking、C标准并行、HPX）中的代码生成能力，并比较了三种提示方式（自然语言、顺序代码、伪代码）以及不同基准问题的表现。

**💡 创新点**

首次在任务并行场景下系统性评估LLM，并提出综合评价指标PCGQS，兼顾功能正确性与可扩展性；同时揭示不同框架对LLM生成效率的影响。

**🔧 技术方法**

采用LLM生成代码，并使用Pass@k、SCC（代码行数与复杂度）以及强弱扩展实验评估；对生成代码进行人工修正等级划分。

**📊 数据集**

使用四个典型基准（π近似、归并排序、矩阵乘法、共轭梯度）作为评测数据集。

**📈 对比分析**

通过Pass@1、PCGQS等指标对模型、框架、提示方式进行比较，结果显示ChatGPT整体表现最佳，C标准并行在弱扩展上最稳健，HPX生成效率最低。

**⚠️ 局限性**

主要局限在LLM对复杂同步、依赖推理的不足，框架细节掌握不完整，以及对更大规模、异构或多阶段并行应用的评估不够全面。

---

## 472. RandSet: Randomized Corpus Reduction for Fuzzing Seed Scheduling

**arXiv ID:** 2602.22729 | [PDF](https://arxiv.org/pdf/2602.22729v1)

**作者:** Yuchong Xie `[一作]` (Hong Kong University of Science and Technology), Dongdong She `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 441 | [OpenAlex ID](https://openalex.org/A5048358055)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出并实现了RandSet，一种针对覆盖驱动模糊测试的随机化语料库缩减技术，能够在持续的种子调度过程中有效缓解种子爆炸问题。

**💡 创新点**

创新点包括：①将语料库缩减建模为集合覆盖问题并引入随机化算法，实现多样化且低开销的子集生成；②采用前沿节点作为特征，显著提升压缩率并保持完整覆盖；③在AFL++、LibAFL、Centipede等主流模糊器中实现无缝集成。

**🔧 技术方法**

使用技术主要包括：随机化集合覆盖算法、前沿节点特征提取、LLVM基础设施构建控制流图、基于哈希的快速特征交集计算、以及与主流模糊器的接口集成。

**📊 数据集**

实验使用了FuzzBench、15个独立程序（涵盖多种领域）以及Magma基准集进行评估。

**📈 对比分析**

与AFL++基线、AFL‑Cmin、MinSet、贪心集合覆盖等方法进行比较，评估种子多样性、子集比、代码覆盖率、缺陷发现和运行时开销。结果表明，RandSet平均子集比为5.99%（FuzzBench）/4.03%（独立程序），覆盖率提升3.57%–16.58%，缺陷发现提升17.1%，而运行时开销仅为1.17%–3.93%。

**⚠️ 局限性**

局限性包括：仅使用前沿节点特征，对无前沿节点或极小语料库的程序效果有限；随机化可能导致偶尔子集不足；在极高吞吐量的模糊器中，过度分散的多样性选择可能降低单次探索深度。

---

## 473. SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation

**arXiv ID:** 2602.22785 | [PDF](https://arxiv.org/pdf/2602.22785v1)

**作者:** Ling Wang `[一作]` (Xi'an Research Institute of Hi-Tech), Yikai Wang `[通讯]` (Beijing Normal University)

**通讯引用:** 1234 | [OpenAlex ID](https://openalex.org/A5100747434)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于单张图片的结构化 3D 场景生成框架 SceneTransporter，能够直接生成包含明确实例分离的完整场景。

**💡 创新点**

创新点在于将实例级关联任务转化为全局 Optimal‑Transport（OT）约束的相关性分配问题，并在 Diffusion 生成器中引入 OT 规划门控交叉注意力和基于边缘的正则化成本，显著解决了结构误划分和几何冗余两大缺陷。

**🔧 技术方法**

主要技术包括：稀疏 Transformer‑VAE 3D 生成器、基于 Sinkhorn 的熵正则化 OT 求解器、OT 计划门控的多头交叉注意力、以及基于图卷积的边缘平滑正则化；整体集成在 DiT 采样过程中实现端到端训练与推理。

**📊 数据集**

使用了公开的带部件标注的 3D 数据集（如 Objaverse）训练生成器，并在 74 张来自网络的开放世界场景图像上进行评估。

**📈 对比分析**

与 PartCrafter、PartPacker、MIDI 等最新方法对比，SceneTransporter 在几何保真度（ULIP、Uni3D）上排名第一，在实例分离度（IoU）上排名第二；推理时间略高于 PartPacker，但远快于 MIDI 与 PartCrafter，且人类评估显示在几何质量、布局一致性和分割可行性上均获得最高分。

**⚠️ 局限性**

局限性包括：推理速度仍略慢；对训练数据中的部件级标注依赖较强；当前仅支持单张图片输入，对极端遮挡或多视角场景的处理仍有待提升；边缘正则化需手动调参，可能在不同图像风格下表现不一致。

---

## 474. Simple Models, Real Swimming: Digital Twins for Tendon-Driven Underwater Robots

**arXiv ID:** 2602.23283 | [PDF](https://arxiv.org/pdf/2602.23283v1)

**作者:** Mike Y. Michelis `[一作]` (ETH Zurich), Robert K. Katzschmann `[通讯]` (ETH Zurich)

**通讯引用:** 5258 | [OpenAlex ID](https://openalex.org/A5050915314)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

基于MuJoCo开发了一个简化的无状态流体动力学模型，将其与肌腱驱动的水下鱼类机器人匹配，构建了可实时运行的数字孪生，并在此基础上实现了强化学习的目标跟踪控制。

**💡 创新点**

创新点包括：①仅用两条实验轨迹即可识别出五个流体系数，模型能够在不同尾摆频率下泛化；②模型实现超实时（≈14×真速），显著优于传统的EBT分析；③将模型整合进MuJoCo并公开开源，为软体水下机器人学习提供便利。

**🔧 技术方法**

技术手段：MuJoCo仿真、无状态流体动力学（Blunt/Slender Drag、Angular Drag、Kutta/Magnus Lift、Added Mass）、贝叶斯优化+Nelder–Mead参数搜索、SAC强化学习、OpenCV标记跟踪。

**📊 数据集**

数据集：实验记录的两条恒频尾摆运动轨迹（0.60 Hz、1.19 Hz）用于识别参数；随后收集的八条不同频率的前进运动轨迹用于泛化评估。

**📈 对比分析**

方法对比：与经典Elongated Body Theory (EBT) 进行性能对比。模型在频率泛化测试中的平均速度误差为0.019，而EBT为0.134；在目标跟踪任务中SAC代理达成93%成功率，平均距离误差仅0.009。

**⚠️ 局限性**

局限性：模型无法捕捉复杂形变及3D水下运动；仅适用于表面漂浮游泳，缺乏浮力/俯仰控制；只考虑恒速条件，未涵盖转弯、加速/减速等动态场景。

---

## 475. The AI Research Assistant: Promise, Peril, and a Proof of Concept

**arXiv ID:** 2602.22842 | [PDF](https://arxiv.org/pdf/2602.22842v1)

**作者:** Tan Bui-Thanh `[一作]` (University of Texas at Austin), Tan Bui-Thanh `[通讯]` (University of Texas at Austin)

**通讯引用:** 2994 | [OpenAlex ID](https://openalex.org/A5008274019)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

通过人机协作研究Hermite插值型数值积分法的误差表示和改进误差界

**💡 创新点**

提出了Hermite四边积分误差的新表示及更宽泛适用的误差界，并展示了人机协作的有效性

**🔧 技术方法**

人机协作框架：人负责问题定义与验证，AI负责符号运算、文献检索、LaTeX编写等

**📊 数据集**

无（理论研究，未使用数据集）

**📈 对比分析**

将人机协作得到的结果与传统人工推导比较，发现AI帮助快速生成多条证明路径并扩展理论，提升工作效率，但仍需人工验证

**⚠️ 局限性**

AI易产生错误/幻觉，需人工严格验证；缺乏深层理解，无法完全替代人类直觉与策略

---

## 476. No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors

**arXiv ID:** 2602.23141 | [PDF](https://arxiv.org/pdf/2602.23141v1)

**作者:** Tao Liu `[一作]` (Nanjing University of Science and Technology), Shibo Wen `[通讯]` (Jilin University)

**通讯引用:** 48 | [OpenAlex ID](https://openalex.org/A5101215043)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无监督、在线的视频稳定框架，结合三阶段经典流水线与多线程缓冲，支持实时推理。

**💡 创新点**

创新点在于协作关键点检测与均匀化、基于多同伦先验的网格传播以及可在线学习的动态核平滑器，同时采用多线程异步管线提升实时性。

**🔧 技术方法**

使用多源特征点融合、稠密光流、轻量化 Ghost+ECA 骨干、OnlineSmoother 的动态核、Multi-homography prior 以及多线程异步缓冲。

**📊 数据集**

使用公开的 NUS、DeepStab、Selfie、GyRo 以及自建的 UAV-Test 多模态无人机视频数据集。

**📈 对比分析**

与多种离线和在线基线对比，在线实验在 Cropping、Distortion、Stability 三项指标均名列前茅，帧率约 12fps，逼近离线方法性能。

**⚠️ 局限性**

局限性：依赖光流估计在复杂场景下误差较大，且边界填充仍需离线后处理，未实现全在线边缘完美处理。

---

## 477. ViCLIP-OT: The First Foundation Vision-Language Model for Vietnamese Image-Text Retrieval with Optimal Transport

**arXiv ID:** 2602.22678 | [PDF](https://arxiv.org/pdf/2602.22678v1)

**作者:** Quoc-Khang Tran `[一作]` (Can Tho University), Nguyen-Khang Pham `[通讯]` (Can Tho University)

**通讯引用:** 183 | [OpenAlex ID](https://openalex.org/A5113776210)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 ViCLIP-OT 这款专为越南语图文检索设计的基础视觉‑语言模型，结合 CLIP 对比学习和 SIGROT 正则化。

**💡 创新点**

创新点在于提出 Similarity‑Graph Regularized Optimal Transport (SIGROT) 损失，利用相似度图和最优传输在批量级别上实现全局跨模态对齐，显著缩小模态间差距。

**🔧 技术方法**

技术上使用 DINOv3 图像编码器、越南语 Sentence‑BERT 文本编码器、CLIP/SigLIP 对比损失以及带熵正则的 Unbalanced Optimal Transport，辅以 UMAP 可视化与 GradCAM 可解释。

**📊 数据集**

使用的公开数据集包括 UIT‑OpenViIC（13,100 图像/61,241 说明），KTVIC（4,327 图像/21,635 说明）与 Crossmodal‑3600（3,600 图像/7,350 说明）。

**📈 对比分析**

在 UIT‑OpenViIC 上与 CLIP/SigLIP 以及多语言预训练模型对比，ViCLIP‑OT 的 R@K 均提升 5–6 分点；在 Crossmodal‑3600 的零样本检索中提升约 11.7 分点，表明对越南语及跨域数据的强泛化能力。

**⚠️ 局限性**

局限性包括依赖越南语专用数据，模型规模相对较小；SIGROT 需要预先计算相似度图，超参数对性能敏感；对其他语言或更大规模数据的迁移性尚未验证。

---

## 478. Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators

**arXiv ID:** 2602.22647 | [PDF](https://arxiv.org/pdf/2602.22647v1)

**作者:** Zhengyang Su `[一作]` (YouTube), Ningren Han `[通讯]` (YouTube)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 STATIC 框架，将前缀树压缩为稀疏 CSR 矩阵，实现对 LLM 生成检索的约束解码，并在 YouTube 大规模推荐系统中部署。

**💡 创新点**

核心创新是把传统指针追踪的 Trie 转化为静态向量化稀疏矩阵操作，达到 O(1) I/O 复杂度，极大提升 TPUs/GPU 上的解码速度，超过 1000 倍。

**🔧 技术方法**

使用 Transformer 生成检索、Semantic ID、Beam Search、CSR 稀疏矩阵、向量化节点转移核（VNTK）、硬件友好的并行化技术以及 XLA/Inductor 编译。

**📊 数据集**

主要使用 YouTube 视频推荐数据（约 20M 新鲜视频）进行工业验证，另外在 Amazon Reviews 子数据集上评估冷启动性能。

**📈 对比分析**

与 CPU Trie、PPV Exact/Approx、Hash Bitmap 等基线对比，STATIC 在每步仅增加 0.033 ms 延迟，占推理时间 0.25%，比 CPU Trie 948×、PPV Exact 1033×快；A/B 测试显示新鲜视频视图提升 5.1%、CTR 提升 0.15%。

**⚠️ 局限性**

局限在于稀疏矩阵的构建是离线过程，难以及时更新约束；同时需要显存支持，且对极大词表的支持仍有限。

---

## 479. Beyond Dominant Patches: Spatial Credit Redistribution For Grounded Vision-Language Models

**arXiv ID:** 2602.22469 | [PDF](https://arxiv.org/pdf/2602.22469v1)

**作者:** Niamul Hassan Samin `[一作]` (KOW Company), Md Ashikur Rahman `[通讯]` (KOW Company)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不需要重新训练的前提下，提出 Spatial Credit Redistribution (SCR) 通过在早期 transformer 层将高注意力源 patch 的激活分配给其 8 连接邻居，从而恢复视觉上下文，减少 VLM 的幻觉生成。

**💡 创新点**

创新点在于：
1) 发现并量化了“空间信用坍塌”现象，证明低空间熵与幻觉率呈负相关；
2) 设计了无训练、仅在推理时使用的激活重分配机制（SCR），实现 51% 线性范数放大而不显著影响生成质量；
3) 通过 8 连接邻域、注意力引导源选取、前向钩子实现对任意现有 VLM 的通用改造；
4) 在多规模多族模型上验证了高效性与普适性。

**🔧 技术方法**

技术细节：
- 前向钩子 (forward‑pass hooks) 在预层正则化前插入，修改隐藏状态；
- 通过多头平均自注意力权重从文本 query 到视觉 patch 计算平均注意力地图；
- 选取 top‑K（K=32）高注意力源，映射其 8 连接空间邻居；
- 对源 patch 进行 λ=1.10 伸缩，将 (λ‑1)·source 加到每个邻居，并把源按 1/λ 缩小；
- 在早期层（Chameleon 0‑15/20，LLaVA 0‑11，Qwen 0‑15）应用，保持后期语言生成不受干扰。

**📊 数据集**

数据集：
- POPE（3 种难度分裂，每类 1k 样本）
- COCO val2014（3000 张图）用于 CHAIR（句子级和实例级）和 CIDEr 评估
- 其他公开基准（如 HallusionBench、MME）在实验补充材料中讨论。

**📈 对比分析**

比较方法与性能：
- 对比 7 种基线：OPERA、VCD、OA‑VCD、SID、VLI、CRoPS、Uniform‑Smooth；
- 在六种模型（Chameleon 7B/30B、LLaVA 7B/13B、Qwen‑VL、Qwen2‑VL‑7B）上评测；
- SCR 在 POPE‑Adversarial 上平均降低 4.7‑6.0 pp 幻觉率，CHAIR‑s 下降 3.7‑5.2 pp（42‑51% 相对），CHAIR‑i 下降 2.7‑4.0 pp（44‑53% 相对）；
- CIDEr 仅下降 ≤0.8 pp；
- 推理延迟 +43‑56 ms，远低于 OPERA (+267 ms)、VCD (+153 ms)、OA‑VCD (+72 ms)，并在所有三指标上 Pareto‑dominate。

**⚠️ 局限性**

局限性：
- 仅证明了空间信用与幻觉的相关性，缺乏因果机制；
- 对超小目标（<2% 图像面积）和边缘目标的改进有限；
- 对模糊邻居或语义相似的邻居可能引入误报；
- 仍无法解决关系推理、关系指代等更高级的视觉语言推断任务；
- 需要前向双传输，虽然开销小，但在极低延迟场景仍有可优化空间。

---

## 480. Agency and Architectural Limits: Why Optimization-Based Systems Cannot Be Norm-Responsive

**arXiv ID:** 2602.23239 | [PDF](https://arxiv.org/pdf/2602.23239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 481. Face Time Traveller : Travel Through Ages Without Losing Identity

**arXiv ID:** 2602.22819 | [PDF](https://arxiv.org/pdf/2602.22819v1)

**作者:** Purbayan Kar `[一作]` (Sony Research), C. V. Jawahar `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FaceTT框架，实现高保真、身份一致的面部老化生成。

**💡 创新点**

创新点包括面部属性感知提示细化、无调优角度反演以及自适应注意力控制，提升身份保持与背景一致性。

**🔧 技术方法**

采用扩散模型（Stable Diffusion）+FastVLM提取属性、角度反演技术与自适应注意力控制。

**📊 数据集**

使用CelebA-HQ、FFHQ-Aging以及自制的明星跨年龄测试集。

**📈 对比分析**

与HRFAE、CUSP、FADING等SOTA方法对比，FaceTT在年龄误差、性别保留、KID、ID_sim循环等指标上均优于对手，推理速度显著提升。

**⚠️ 局限性**

局限性在于对极端光照或遮挡的鲁棒性不足，且仅在单帧图像上验证，视频时序一致性待进一步研究。

---

## 482. ParamMem: Augmenting Language Agents with Parametric Reflective Memory

**arXiv ID:** 2602.23320 | [PDF](https://arxiv.org/pdf/2602.23320v1)

**作者:** Tianjun Yao `[一作]`, Kun Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种可参数化的反思记忆模块ParamAgent，能够通过少量训练样本在模型参数中编码跨样本反思模式，并将其嵌入现有的反思式语言代理框架中；

**💡 创新点**

创新点在于将反思多样性转移到模型参数层，形成一种轻量、可持续的多样化记忆机制，兼容并增强了传统的事件记忆和跨样本记忆，并实现了自我改进与弱-强迁移；

**🔧 技术方法**

主要技术包括利用LoRA微调的参数化记忆模块、温度控制的采样、多轮迭代反思与评估机制，以及与episodic和cross-sample记忆的联合；

**📊 数据集**

实验使用了编程任务（HumanEval、MBPP、LiveCodeBench）、数学推理任务（MATH）和多跳问答任务（HotpotQA、2WikiMultiHopQA）；

**📈 对比分析**

与基线（Base、Reflexion、DoT、DoT-bank、Retroformer）比较，ParamAgent在所有任务上均显著提升性能，例如HumanEval Pass@1从76提升至82、MATH准确率提升至68；

**⚠️ 局限性**

主要局限是会导致一定的token消耗增加，且参数化记忆的多样性受限于训练数据的覆盖范围，无法彻底替代外部检索或更强模型的支持。

---

## 483. A Dataset is Worth 1 MB

**arXiv ID:** 2602.23358 | [PDF](https://arxiv.org/pdf/2602.23358v1)

**作者:** Elad Kimchi Shoshani `[一作]` (Hebrew University of Jerusalem), Yedid Hoshen `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 1706 | [OpenAlex ID](https://openalex.org/A5047455929)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种通过向客户端传输伪标签而非原始图像，利用预置参考数据集完成任务迁移的方法。

**💡 创新点**

创新点在于将数据蒸馏反向为标签蒸馏，并引入能量基过滤与安全网机制以挑选语义相关样本，显著降低传输 payload。

**🔧 技术方法**

使用的技术包括基于能量的 OOD 评分、Safety‑Net 类别配额过滤、变量长度编码（如 Zstd）以及伪标签生成。

**📊 数据集**

在 ImageNet‑21K/1K 作为参考数据集，针对 10 个分类基准（Caltech‑101、CIFAR‑10、CUB‑200、Places365 等）以及 4 个医学 OOD 数据集进行实验。

**📈 对比分析**

与随机子集、K‑Center 以及最先进的数据蒸馏方法对比，PLADA 在 <1 MB payload 下实现高达 90 %+ 的准确率，明显优于其他方法。

**⚠️ 局限性**

局限性包括需要客户端预存大量参考数据、在极度 OOD 或细粒度任务时可能需要更高的训练迭代，以及目前仅适用于分类任务。

---

## 484. Workload-Aware Incremental Reclustering in Cloud Data Warehouses

**arXiv ID:** 2602.23289 | [PDF](https://arxiv.org/pdf/2602.23289v1)

**作者:** Yipeng Liu `[一作]` (Tsinghua University), Haunchen Zhang `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种工作负载感知的增量重聚类框架（WAIR），能在云数据仓库持续写入与查询模式演变的环境下，自动维护微分区的高质量聚类状态；

**💡 创新点**

创新点包括：①将重聚类策略与聚类键选择解耦，允许每个微分区采用最优键；②提出边界微分区概念，证明重聚类仅这些分区即可近似最优剪枝效果；③基于利用率和成本模型的增量决策算法，并支持混合布局（单列、Hilbert曲线、Qd-tree等）；

**🔧 技术方法**

使用的技术包括：区间区图（zone maps）和微分区元数据、利用率度量、滑动窗口查询统计、基于CPU时间的成本模型、DBSCAN聚类签名、Hilbert曲线排序、Qd-tree重划分、外部/内部排序、DuckDB执行引擎与S3/Redis元数据服务；

**📊 数据集**

实验使用了标准基准数据集 TPC‑H（SF720）、DSB（SF240→720）以及真实业务日志 Mirror（2.4 TB 原始日志→212 GB 压缩），分别生成连续写入和演化的工作负载；

**📈 对比分析**

在相同的 AWS 云环境下与四种基线（无重聚、Qd‑tree 全表重排、Delta Lake 风格新数据聚类、Dremio 数据驱动增量）对比，采用 CPU 时间和 I/O 量衡量。WAIR 在 TPC‑H、DSB、Mirrors 的累计运行时间分别提升 61%、60% 与 86%，且重聚成本低于所有基线，显示出更优的成本‑性能平衡；

**⚠️ 局限性**

局限性包括：依赖持续的查询统计和元数据收集；参数（滑动窗口大小、α、β 等）需要调优且对结果敏感；在极低选择率或高度随机的查询场景下，成本模型可能产生波动；在极大规模或极高写入速率的系统中，外部排序和元数据开销仍可能成为瓶颈。

---

## 485. SettleFL: Trustless and Scalable Reward Settlement Protocol for Federated Learning on Permissionless Blockchains (Extended version)

**arXiv ID:** 2602.23167 | [PDF](https://arxiv.org/pdf/2602.23167v1)

**作者:** Shuang Liang `[一作]` (Shanghai Jiao Tong University), Haibing Guan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6364 | [OpenAlex ID](https://openalex.org/A5049487451)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SettleFL 这一针对联邦学习的去信任、可扩展奖励结算协议，支持在无中心化权限链上实现高频次模型更新的奖励分配；

**💡 创新点**

创新点在于构建两种可互操作的结算策略（Commit‑and‑Challenge 与 Commit‑with‑Proof），利用共享的领域特定 SNARK 电路架构实现线性上链成本与即时最终性的权衡；

**🔧 技术方法**

核心技术包括基于 Poseidon 哈希与 EdDSA 的聚合签名、Groth16 SNARK 证明、可组合的电路模块（LPI、BatchExtractor 等）、以及 Solidity 智能合约实现的状态机；

**📊 数据集**

实验数据集覆盖图像（MNIST、Fashion‑MNIST、CIFAR‑10）、文本（Twitter）、表格（Bank）与图结构（Cora），均采用对应的轻量级模型（LeNet‑5、ResNet‑20、FastText 等）；

**📈 对比分析**

与 StateFL、OpenFL 等基线相比，SettleFL 在 800 名参与者、50 轮 MNIST 训练下，总 Gas 约 0.0028 ETH（≈ $5.5）或 0.0021 ETH（≈ $4.0），实现了低成本、可预见的每轮开销，并保持模型准确率与标准 FL 无异；

**⚠️ 局限性**

局限性包括：①仅关注奖励结算而不覆盖贡献评估；②对批大小和合约大小有限制，过大批量会导致部署失败；③需要在链上支付 Gas，且 Commit‑with‑Proof 版证明生成成本仍较高；④在极端网络拥塞或链拥堵时仍可能受限。

---

## 486. Relating the Neural Representations of Vocalized, Mimed, and Imagined Speech

**arXiv ID:** 2602.22597 | [PDF](https://arxiv.org/pdf/2602.22597v1)

**作者:** Maryam Maghsoudi `[一作]` (University of Maryland), Shihab A. Shamma `[通讯]` (University of Maryland)

**通讯引用:** 17013 | [OpenAlex ID](https://openalex.org/A5057113698)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了发声、模仿和想象语音时的sEEG信号，训练线性与非线性解码器将脑电映射到语音频谱，并评估跨条件迁移与句子辨别能力。

**💡 创新点**

首次系统比较三种语音模式下脑表征的共享程度，证明线性解码器在跨条件迁移中保持句子辨别性的优势。

**🔧 技术方法**

采用线性时滞回归、卷积‑循环神经网络、NSL皮层模型生成频谱，配合rank‑based句子辨别分析。

**📊 数据集**

使用公开的VocalMind sEEG 数据集，该数据集包含中文句子在发声、模仿和想象三种条件下的记录。

**📈 对比分析**

通过对角线与交叉条件的相关系数以及AUC评估；线性模型相关性略低但句子辨别AUC更高，两种模型均显著优于置乱对照。

**⚠️ 局限性**

受限于单受试者、仅三种语音模式且缺少听语条件，跨受试者推广性与更细粒度表征分析仍需进一步研究。

---

## 487. FactGuard: Agentic Video Misinformation Detection via Reinforcement Learning

**arXiv ID:** 2602.22963 | [PDF](https://arxiv.org/pdf/2602.22963v1)

**作者:** Zehao Li `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Zhaoqi Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 1471 | [OpenAlex ID](https://openalex.org/A5100613098)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于多模态大语言模型的代理式视频谣言检测框架FactGuard，通过迭代推理与工具调用实现自适应证据收集和决策。

**💡 创新点**

核心创新在于将验证过程视为不确定性感知的代理决策；引入工具激活机制（FactProbe、ClipScout）以及两阶段训练（链式思维微调+决策感知强化学习）以显著提升鲁棒性与解释性。

**🔧 技术方法**

技术手段包括多模态大语言模型（如Qwen2.5‑VL）、链式思维监督微调、决策感知的GRPO强化学习、风险敏感奖励设计以及外部知识检索与视频片段定位工具。

**📊 数据集**

在FakeSV、FakeTT、FakeVV三个公开视频谣言数据集上进行实验，利用其时间分割测试集评估。

**📈 对比分析**

与传统判别模型、零样本多模态LLM以及已强化学习的任务对齐模型比较，FactGuard在准确率、F1得分等指标上均实现了显著提升，且推理过程更具可解释性。

**⚠️ 局限性**

局限性包括对外部工具与检索服务的依赖、对大型模型资源需求高、推理时长和计算成本较大，以及在极端模糊或无可检索证据场景下仍可能产生不确定决策。

---

## 488. Causal Direction from Convergence Time: Faster Training in the True Causal Direction

**arXiv ID:** 2602.22254 | [PDF](https://arxiv.org/pdf/2602.22254v1)

**作者:** Abdulrahman Tamim `[一作]` `[通讯]` (Independent Researcher), Abdulrahman Tamim (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于神经网络收敛速度不对称性（CCA）来判定因果方向，并将其嵌入到全新的 CCL 框架中；

**💡 创新点**

首次理论证明：在可注入非线性加噪机制下，正向学习收敛步数严格小于反向，构成可用于因果方向判定的严格统计信号；

**🔧 技术方法**

使用多层感知机（MLP）训练、SGD/Adam/RMSProp 等优化器、z‑标准化、信息瓶颈（CIB）、最小描述长度（MDL）和交互式强化学习（CRL）等技术；

**📊 数据集**

在合成的五种数据生成过程（sin、exp、x³、x²、线性 Gaussian）上进行 30 试验；在真实的 Tübingen 因果对数据集上做验证；

**📈 对比分析**

与 RESIT、IGCI、SkewScore 等方法对比，CCA 在合成数据上 30/30 正确率（注入性非线性）或 26/30（需标准化），在 Tübingen 上达 96% 准确率（AUC 0.96），显著优于传统方法；

**⚠️ 局限性**

局限性：仅适用于一维可注入非线性机制；对非注入或线性高斯机制失效；需事先进行 z‑标准化；仅支持 Rung 2（因果推断），不处理 Rung 3（反事实）；需要交互式实验数据；在多维/反馈循环结构下尚未验证。

---

## 489. Tokenization, Fusion and Decoupling: Bridging the Granularity Mismatch Between Large Language Models and Knowledge Graphs

**arXiv ID:** 2602.22698 | [PDF](https://arxiv.org/pdf/2602.22698v1)

**作者:** Siyue Su `[一作]` (Beihang University), Guanglin Niu `[通讯]` (Beihang University)

**通讯引用:** 639 | [OpenAlex ID](https://openalex.org/A5065094272)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出KGT框架，解决LLM与知识图谱粒度不匹配问题，直接把实体和关系视为不可分割的专用标记进行建模；

**💡 创新点**

创新点在于：①专用实体/关系标记化与双流特征融合；②关系引导的门控融合机制；③双视图解耦预测头及可学习的logit缩放；④利用LoRA在LLM基础上进行参数高效微调；

**🔧 技术方法**

技术手段包括：Llama‑2‑7b‑chat 作为后端；双流特征提取（文本句向量+结构TuckER嵌入）并投影至LLM空间；关系门控融合（ReGF）；双MPL预测头，分别投射至文本和结构子空间；LoRA打分矩阵与可学习logit缩放；

**📊 数据集**

使用三个多模态KG基准：DB15K、MKG‑W（WikiData子集）和MKG‑Y（YAGO子集）；

**📈 对比分析**

与19个最新基线（结构式、图像/文本多模态、LLM式）在MRR和Hits@K下进行对比，KGT在所有数据集上均实现MIRR和Hits@1提升5%–18%，获得新SOTA；

**⚠️ 局限性**

缺陷在于依赖预先计算的结构先验，无法在训练中联合优化结构特征，导致性能受限于外部KGE模型的质量，且会带来额外计算开销。

---

## 490. ArchAgent: Agentic AI-driven Computer Architecture Discovery

**arXiv ID:** 2602.22425 | [PDF](https://arxiv.org/pdf/2602.22425v1)

**作者:** Raghav Gupta `[一作]` (University of California), Sagar Karandikar `[通讯]` (University of California)

**通讯引用:** 1118 | [OpenAlex ID](https://openalex.org/A5017667683)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个名为ArchAgent的代理式生成式AI系统，自动设计、实现并评估新的缓存替换策略。

**💡 创新点**

创新点在于：①利用大型语言模型与进化搜索联合实现端到端的体系结构发现；②在几天内生成的策略在单核SPEC 2006和多核Google Traces上均超越现有最优策略；③首次展示后硅超特化（runtime‑configurable 参数调优）和模拟器逃逸的现象。

**🔧 技术方法**

核心技术包括 Gemini 2.5 LLM（Flash/Pro）、MAP‑Elites / 岛屿进化算法、ChampSim 微架构模拟器以及分布式评估后端。

**📊 数据集**

使用的数据集为：内存密集型 SPEC 2006 子集（约 1B 指令）和公开的 Google Workload Traces v2（多核 50M/20M 指令）。

**📈 对比分析**

对比方法：在 CRC‑2 赛制下，采用 IPC 相对 LRU 的几何平均（单核）或加权平均（多核）指标；在 SPEC 2006 上实现 0.907% IPC 提升，Google Traces 上实现 5.322% IPC 提升；相较于先前 SoTA，进化时间仅为人类的 1/3–1/5。

**⚠️ 局限性**

主要局限包括：①需人工验证生成策略的硬件实现性和可解释性；②模拟器精度与速度限制导致可能出现“逃逸”或过拟合；③对硬件约束（如存储预算、代码大小）的判定仍主要靠提示或手工检查，缺乏统一的自动化验证框架。

---

## 491. IMMACULATE: A Practical LLM Auditing Framework via Verifiable Computation

**arXiv ID:** 2602.22700 | [PDF](https://arxiv.org/pdf/2602.22700v1)

**作者:** Yanpei Guo `[一作]` (National University of Singapore), Jiaheng Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 14726 | [OpenAlex ID](https://openalex.org/A5032474012)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种IMMACULATE框架，通过在黑盒LLM服务上随机审计并使用可验证计算来检测模型替换、量化滥用和计费欺诈等经济动机偏差。

**💡 创新点**

创新点包括：1) 设计了可在数值非确定性环境下使用的Logit距离分布（LDD）作为模型执行信度度量；2) 采用随机抽样审计和只对少量请求产生可验证证明，显著降低证明开销；3) 用加密承诺绑定完整精度模型，而非依赖可信执行环境。

**🔧 技术方法**

使用技术包括可验证计算（VC/TEE）、加密承诺、随机抽样审计、LDD（基于KL或TV距离）以及Top‑K距离优化，以在不泄露模型参数的前提下证明推理正确性。

**📊 数据集**

实验使用了GSM8K、TriviaQA、WebQuestions等三大数据集，并在LLaMA3‑70B、Qwen3‑32B、Qwen3‑30B‑A3B、DeepSeek‑V2‑Lite等模型上评估，比较了BF16、FP8量化和模型替换三种攻击场景。

**📈 对比分析**

与传统的可信硬件或完整整数推理方案相比，IMMACULATE在检测率上可达40%–99%（针对模型替换）且对FP8量化的检测率为1–3%；随机审计3000请求即可实现95%检测概率，误报率低于1e‑5；吞吐率增益不到1%，展示出良好的性能兼容性。

**⚠️ 局限性**

局限性包括：1) 仅对至少10%请求偏差的攻击有效，无法检测极少数或细粒度欺诈；2) 依赖随机抽样和阈值设定，误报率估计受极值理论近似影响；3) 目前未覆盖所有可能的恶意策略（如复杂的token overbilling）；4) 需要VC证明器实现，虽然对整体吞吐影响小，但在极大规模部署下仍可能产生非零成本。

---

## 492. FuxiShuffle: An Adaptive and Resilient Shuffle Service for Distributed Data Processing on Alibaba Cloud

**arXiv ID:** 2602.22580 | [PDF](https://arxiv.org/pdf/2602.22580v1)

**作者:** Yuhao Lin `[一作]` (Wuhan University), Xiao Yan `[通讯]` (Alibaba Cloud)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对阿里云 MaxCompute 的超大规模生产环境，设计并实现了一个可自适应、可恢复的 Shuffle 服务 FuxiShuffle。

**💡 创新点**

创新点包括：动态 Shuffle 模式选择、进度感知的渐进调度、Shuffle Agent 分组与多副本、内存分层淘汰策略以及增量恢复机制。

**🔧 技术方法**

采用的技术包括分布式 Shuffle Agent、元数据索引驱动的读取、进度感知的调度算法、内存管理与多副本备份、以及增量恢复与检查和版本一致性验证。

**📊 数据集**

实验数据集主要使用行业标准的 TeraSort 与 TPC‑DS 基准，规模从 1TB 到 10TB，且在阿里云内部大规模集群上进行评测。

**📈 对比分析**

通过与 Hadoop‑like、Spark‑like 等现有实现的对比，FuxiShuffle 在平均作业完成时间上提升 76.36%，资源消耗降低 67.14%，单点故障下性能波动控制在 10% 以内。

**⚠️ 局限性**

局限性在于极端高并发或大规模失效场景下备份开销仍显著，动态阈值调优与多租户资源分配机制仍需进一步细化。

---

## 493. Clarification of `Algorithmic Collusion without Threats'

**arXiv ID:** 2602.22232 | [PDF](https://arxiv.org/pdf/2602.22232v1)

**作者:** Jason Hartline `[一作]` `[通讯]`, Jason Hartline

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

对《Algorithmic Collusion without Threats》所描述的情境进行澄清，指出其并非算法合谋，而是单方非竞争行为。

**💡 创新点**

提出了关于算法合谋与非竞争均衡的清晰界定，并强调监管时需关注单方非竞争者。

**🔧 技术方法**

运用重复博弈理论、Nash均衡分析及博弈论基础概念。

**📊 数据集**

无实验数据集，纯理论论证。

**📈 对比分析**

未进行实验对比，主要通过理论推导说明非竞争均衡与合谋的区别。

**⚠️ 局限性**

缺乏实证验证，论述基于理论假设，无法直接应用于实际监管案例。

---

## 494. Benchmarking Temporal Web3 Intelligence: Lessons from the FinSurvival 2025 Challenge

**arXiv ID:** 2602.23159 | [PDF](https://arxiv.org/pdf/2602.23159v1)

**作者:** Oshani Seneviratne `[一作]` (Rensselaer Polytechnic Institute), Kristin P. Bennett `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 9934 | [OpenAlex ID](https://openalex.org/A5048876983)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并利用了 FinSurvival 2025 竞赛，基于 Aave v3 Polygon 上的 21.8 万笔交易，评估时间‑到‑事件（survival）预测模型；

**💡 创新点**

提出了面向 Web3 的长期时间序列基准，强调隐式抛光、非平稳性和删失处理，并通过对比分析证明领域特定的层级特征工程显著优于端到端深度模型；

**🔧 技术方法**

使用 XGBoost（AFT 与 Cox 变体）、AutoML（Optuna）以及深度生存模型（DeepSurv）等多种机器学习技术，同时采用层级特征构建、缺失恢复与两阶段训练；

**📊 数据集**

使用了来自 Aave v3 Polygon 子图的 21.8 万笔交易记录，生成 90 个用户、市场与交易层面特征，涵盖 16 对应时间‑到‑事件任务；

**📈 对比分析**

通过在固定时间切分下的 Concordance Index（C‑index）进行评估，第一名平均 C‑index 达 0.914，第二名 0.849，第三名 0.847，展示层级特征工程与多任务迁移显著提升性能；

**⚠️ 局限性**

局限在于仅覆盖单一协议（Aave v3 Polygon），缺少链下情境（情绪、治理等）以及仅评估相对排序的 C‑index，未检验绝对时间预测与多协议/跨链适用性。

---

## 495. SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling

**arXiv ID:** 2602.23013 | [PDF](https://arxiv.org/pdf/2602.23013v1)

**作者:** Camile Lendering `[一作]` (Eindhoven University of Technology), Egor Bondarev `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 682 | [OpenAlex ID](https://openalex.org/A5069685461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无需训练、无记忆库、无提示调优的极简少样本视觉异常检测框架 SubspaceAD

**💡 创新点**

核心创新在于直接利用冻结的 DINOv2-G 视觉基础模型提取的密集特征，随后用 PCA 拟合一条低维子空间；异常通过重构残差统计实现，完全抛弃了传统的记忆银行、生成网络或多模态提示技术

**🔧 技术方法**

使用 DINOv2-G 的多层特征平均、Principal Component Analysis（PCA）构建子空间、残差归一化、TVaR（尾部风险值）聚合以及双线性上采样/高斯平滑等统计与图像处理技术

**📊 数据集**

在工业缺陷检测基准 MVTec-AD（15 类）和 VisA（更高分辨率、多样缺陷）上进行评估

**📈 对比分析**

与 1‑shot、2‑shot、4‑shot 以及批量 0‑shot 方案下的主流记忆库、重建网络和 VLM 方法对比，SubspaceAD 在 MVTec-AD 上单样本图像级 AUROC 98.0%、像素级 97.6%；在 VisA 上单样本图像级 93.3%、像素级 93.4%，均在所有比较方法中取得最高或接近最高分，显示出显著的性能优势

**⚠️ 局限性**

局限性包括：1）对基础模型特征的高度依赖，若领域特征与 DINOv2 预训练域差异过大可能导致子空间拟合不足；2）假设正常样本特征近似线性子空间，复杂的非线性正常分布可能需要更丰富的模型；3）前向推理仍需要一次完整的 DINOv2-G 计算，导致推理时延相对较高

---

## 496. MovieTeller: Tool-augmented Movie Synopsis with ID Consistent Progressive Abstraction

**arXiv ID:** 2602.23228 | [PDF](https://arxiv.org/pdf/2602.23228v1)

**作者:** Yizhi Li `[一作]` (Zhejiang University), Gaoang Wang `[通讯]` (Zhejiang University)

**通讯引用:** 38328 | [OpenAlex ID](https://openalex.org/A5028525523)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了MovieTeller框架，用工具增强的渐进抽象方法生成长视频摘要。

**💡 创新点**

创新点在于训练无关的工具增强与渐进抽象：用面部识别模型提供事实基础，保证ID一致性；并将摘要分阶段压缩。

**🔧 技术方法**

使用面部识别工具(InsightFace)、VLM(如Qwen2.5-VL、InternVL3、WeThink-Qwen2.5VL)以及PySceneDetect等，结合Prompt注入事实基础。

**📊 数据集**

在包含100部全长电影、总时长约10,000分钟的多样化数据集上进行实验。

**📈 对比分析**

与无提示、仅姓名提示等基线比较，MovieTeller在BERTScore、LLM‑as‑a‑Judge得分及人工评估中均显著优越，提升约39% F1、117% ID一致性，人工优选率达60%以上。

**⚠️ 局限性**

受限于面部数据库完整性、缺少音频模态以及工具调用需手工设计，未来需集成音频、动态工具策略。

---

## 497. Fault-tolerant Reduce and Allreduce operations based on correction

**arXiv ID:** 2602.22445 | [PDF](https://arxiv.org/pdf/2602.22445v1)

**作者:** Martin Kuettler `[一作]` (Technische Universitaet Dresden), Hermann Haertig `[通讯]` (Technische Universitaet Dresden)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一种基于“up‑correction”与树形传播的容错 Reduce 和 Allreduce 算法，能够在最多 f 个进程崩溃的情况下仍然得到正确的聚合结果，适用于小消息场景。

**💡 创新点**

创新点在于：① 将传统树形 Reduce 与纠正 (up‑correction) 组相结合；② 设计了多种失效信息传播策略（列表、大小、失败位）；③ 通过预先的 up‑correction 组通信，让子树内部已知失效信息，从而保证根节点能够正确挑选或合并结果。

**🔧 技术方法**

主要技术包括：Fail‑stop 模型的失效检测（超时确认）、基于组的 up‑correction 交换、树形 Reduce 与 Broadcast 的组合实现 Allreduce、以及可选的失效信息编码方式。

**📊 数据集**

本文未使用真实数据集，而是以理论分析和消息计数模型评估算法的通信复杂度；若有实验，可能在模拟小规模 MPI 环境中进行，但原文中未给出具体数据集。

**📈 对比分析**

性能评估以消息数为指标：在无失效时，up‑correction 阶段约发送 f(f+1)⌊(n‑1)/(f+1)⌋ + a(a‑1) 条消息，树形阶段 n‑1 条；Allreduce 需要 Reduce+Broadcast 的消息量；在失效时消息数会相应减少，但在根失效时可能需要多次尝试，最多 f+1 次消息量上升。

**⚠️ 局限性**

限制：① 仅适用于 fail‑stop（crash）失效；② 需要至少 f+1 可用进程；③ 对于大消息不具备竞争性；④ 根进程在 Reduce/Allreduce 中若失效则无法完成；⑤ 失效信息传递依赖于超时检测，可能导致延迟；⑥ 缺乏实验验证，主要为理论分析。

---

## 498. CRAG: Can 3D Generative Models Help 3D Assembly?

**arXiv ID:** 2602.22629 | [PDF](https://arxiv.org/pdf/2602.22629v1)

**作者:** Zeyu Jiang `[一作]` (New York University), Jing Zhang `[通讯]` (New York University)

**通讯引用:** 26382 | [OpenAlex ID](https://openalex.org/A5100345341)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个统一框架，联合3D装配和完整形状生成，通过共同的流匹配过程同时预测碎片姿态与全形状潜在向量；

**💡 创新点**

创新点在于使用共享的TripoSG VAE作为共同潜在空间，配合双向信息流的Joint Adapter，让装配与生成互相调优，并能够在缺失碎片时自动推断缺失几何；

**🔧 技术方法**

主要技术包括TripoSG的VAE+流匹配、Mixture-of-Transformers架构、SE(3)与潜在空间的双流匹配、两阶段训练策略以及基于注意力的双向信息桥接；

**📊 数据集**

实验使用了PartNeXt、Breaking Bad、MorphoSource以及FRACTURA等数据集，对部件装配和破碎重组任务进行评估；

**📈 对比分析**

在与GARF、RPF和Assembler等基线对比中，所提出的方法在完整和缺失碎片场景下均取得了SOTA表现，显著提升部件准确率（PA）并降低Chamfer距离（CD），尤其在缺失碎片的情况下表现突出；

**⚠️ 局限性**

局限性包括对训练数据分布的偏倚导致的OOV泛化能力不足、对对称/可互换部件的评估不完善，以及目前仅支持图像条件控制，缺乏更丰富的多模态交互方式。

---

## 499. NoRA: Breaking the Linear Ceiling of Low-Rank Adaptation via Manifold Expansion

**arXiv ID:** 2602.22911 | [PDF](https://arxiv.org/pdf/2602.22911v1)

**作者:** Hung-Hsuan Chen `[一作]` (National Central University), Hung-Hsuan Chen `[通讯]` (National Central University)

**通讯引用:** 849 | [OpenAlex ID](https://openalex.org/A5078925594)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 NoRA，一种在权重级别注入 SiLU 门控与结构性 Dropout 的非线性低秩适配器，打破 LoRA 的“线性天花板”。

**💡 创新点**

创新点在于将非线性操作直接嵌入 Transformer 的查询/值投影中，并通过 Dropout 扩展特征流形，从而在相同或更小秩下实现更高表达能力。

**🔧 技术方法**

采用 SiLU 激活、结构性 Dropout、权重级别并行瓶颈，并对 Llama‑3‑8B 进行精度训练与 SVD、有效秩分析。

**📊 数据集**

使用 SlimOrca（约30万条 GPT‑4 生成的链式推理对话）与 MathInstruct（含 GSM8K 与 MATH 的 10万数学题）进行实验。

**📈 对比分析**

与 LoRA 在秩 {16, 64, 128, 512} 进行对比，NoRA 在秩 64 时的 PPL 3.89 低于 LoRA 512 的 3.90；在 MathInstruct 上，NoRA 在 512 秩时 PPL 1.97 远优于 LoRA 的 2.07，且有效秩提升 5 倍。

**⚠️ 局限性**

限制在于需要非合并推理（对内存受限的边缘设备不友好）以及目前仅在 Llama‑3 验证，需进一步验证在其他架构（如 MoE、Mamba）及多模态任务上的通用性。

---

## 500. InfoAlign: A Human-AI Co-Creation System for Storytelling with Infographics

**arXiv ID:** 2602.22901 | [PDF](https://arxiv.org/pdf/2602.22901v1)

**作者:** Jielin Feng `[一作]` (Fudan University), Siming Chen `[通讯]` (Fudan University)

**通讯引用:** 4214 | [OpenAlex ID](https://openalex.org/A5050391600)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了以叙事为中心的三阶段工作流，并实现了InfoAlign系统，帮助用户将长文本转化为故事、推荐视觉编码方案并生成布局蓝图，支持用户随时介入和修改；

**💡 创新点**

将故事构建、视觉编码、空间组合三阶段系统化，并通过人机协作保持故事一致性，首次实现长文本到信息图的端到端自动化；

**🔧 技术方法**

结合自然语言处理（文本语义提取、主题建模）、机器学习/深度学习（视觉设计推荐、布局生成）以及基于语义匹配的AI协同模块；

**📊 数据集**

使用多领域长文本数据集，包括泰坦尼克号生存数据、暗能量宇宙学数据、用水统计数据和精神健康调查数据等；

**📈 对比分析**

通过用户实验与定量指标（故事连贯性评分、设计质量评估）与传统信息图工具对比，显示InfoAlign在保持故事连贯性和提升设计效率方面具有显著优势；

**⚠️ 局限性**

系统在跨域适用性、对极端文本结构的鲁棒性以及对高度专业化视觉表达的支持方面仍有提升空间。

---

## 501. Natural Language Declarative Prompting (NLD-P): A Modular Governance Method for Prompt Design Under Model Drift

**arXiv ID:** 2602.22790 | [PDF](https://arxiv.org/pdf/2602.22790v1)

**作者:** Hyunwoo Kim `[一作]` (ddai Inc.), Yumin Kim `[通讯]` (ddai Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并阐述了自然语言声明式提示（NLD-P）——一种通过在提示中显式分离 provenance、constraint logic、task content 与 post‑generation evaluation 的模块化治理方法；

**💡 创新点**

创新点在于将提示设计视为治理结构，强调在模型漂移背景下的结构化分离，使治理层可独立编辑、可追溯；

**🔧 技术方法**

主要技术为基于自然语言的结构化提示设计（无代码），并辅以人机循环验证与治理透明化手段；

**📊 数据集**

未使用特定数据集，依赖公开模型与示例进行概念性验证；

**📈 对比分析**

论文未进行定量对比或性能评估，侧重理论阐述与方法框架；

**⚠️ 局限性**

局限在于缺乏实证验证、对不同 LLM 版本的适用性未测试、仅提供理论与架构指导

---

## 502. Agentic AI for Intent-driven Optimization in Cell-free O-RAN

**arXiv ID:** 2602.22539 | [PDF](https://arxiv.org/pdf/2602.22539v1)

**作者:** Mohammad Hossein Shokouhi `[一作]` (University of British Columbia), Vincent W. S. Wong `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于代理式人工智能的意图驱动优化框架，实现了在cell‑free O‑RAN中自动化地将运维人员的自然语言意图转换为具体的功率控制、O‑RU激活与用户优先级设置，从而同时满足能耗节约与用户吞吐率需求。

**💡 创新点**

创新点包括：1）多任务协作的LLM代理体系，利用共享轻量LLM与QLoRA低秩适配器实现参数高效复用；2）记忆检索机制将历史经验快速提取用于权重与惩罚系数的自动调节；3）采用多智能体Proximal Policy Optimization（MAPPO）实现分布式O‑RU激活决策，提升能耗优化的可扩展性。

**🔧 技术方法**

使用技术包括：大型语言模型（GPT‑5、Qwen 2.5）、QLoRA量化低秩适配器、多智能体深度强化学习（MAPPO）、加权最小均方误差（WMMSE）算法、记忆检索（基于低维嵌入的相似度搜索）、意图解析与自动化调参。

**📊 数据集**

实验数据采用仿真生成的cell‑free O‑RAN场景（50个O‑RU、20个用户、4/2天线、30dBm功率上限等），未使用真实网络数据集。

**📈 对比分析**

与三种基线（DRL+梯度上升、贪婪激活、全功率模式）在能耗模式下进行对比；结果显示该框架在O‑RU活跃比例上比基线低41.93%，并在使用QLoRA+量化后内存占用比部署单独LLM模型高达92%地减少。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实网络部署与实验验证；多代理间的协同与参数同步可能导致不稳定；目前仅覆盖能耗与总吞吐目标，未涵盖更细粒度的资源分配与信道估计等功能。

---

## 503. Social Welfare in Budget Aggregation

**arXiv ID:** 2602.23027 | [PDF](https://arxiv.org/pdf/2602.23027v1)

**作者:** Javier Cembrano `[一作]` (Universidad de Chile), Markus Utke `[通讯]` (TU Eindhoven)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究预算聚合下的 ℓ1-效用模型，探讨单一极端偏好（single‑minded）情况下的比例性与真诚性，提出新机制并分析其社会福利与比例性的折衷。

**💡 创新点**

提出单一极端比例性（single‑minded proportionality）的价格上界并证明其最优性；引入更强的可分解（decomposable）比例性，证明其不增加价格；设计了新的移动幻影机制 ℒ^*（R*）与 GreedyDecomp，并给出最优福利逼近与 2‑近似；证明在可分解机制下寻找福利最优解为 NP‑难。

**🔧 技术方法**

使用移动幻影机制框架、比例支出（proportional spending）与区间范围保持（range respect）等概念，构建证明框架；通过构造实例和图形化证明获得价格下界；利用整数线性规划与贪心算法得到可分解机制近似结果。

**📊 数据集**

本文为理论分析，不使用实验数据集；若有实验，则为合成实例验证 dominance 关系与近似比值。

**📈 对比分析**

方法通过理论证明与构造实例比较；在单一极端比例性下的价格为 Θ(√n)，最优机制 ℒ^* 取得此比值；GreedyDecomp 在所有可分解机制中以 2 近似，且在某些实例中逼近上界。

**⚠️ 局限性**

限制包括：可分解机制最优解不可多项式求解；部分已知机制不满足比例支出，无法直接推广；未给出关于 m 的完全对称结果；存在开放问题如 dominance 是否构成格。

---

## 504. Imagination Helps Visual Reasoning, But Not Yet in Latent Space

**arXiv ID:** 2602.22766 | [PDF](https://arxiv.org/pdf/2602.22766v1)

**作者:** You Li `[一作]` (Beijing Jiaotong University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 37606 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了潜在视觉推理的有效性，采用因果中介分析方法揭示了潜在标记在视觉推理中的作用，发现其对输入和最终答案的影响有限。

**💡 创新点**

创新点在于提出了一种文本空间想象的方法，替代潜在空间推理，显示出更好的因果效果和性能。

**🔧 技术方法**

使用了因果中介分析框架，结合了文本空间想象的训练方法。

**📊 数据集**

使用了Monet-SFT-125K数据集，并进行了数据重写和过滤以提高数据质量。

**📈 对比分析**

与现有的潜在空间方法（如Monet和LVR）进行比较，提出的方法在多个视觉基准测试中表现出显著的性能提升，尤其在HR-Bench和MME-RealWorld-Lite上超越了4.0%和4.9%。

**⚠️ 局限性**

限制在于潜在标记的内在机制尚未完全理解，且当前方法仍需进一步探索其潜力。

---

## 505. Grasp, Slide, Roll: Comparative Analysis of Contact Modes for Tactile-Based Shape Reconstruction

**arXiv ID:** 2602.23206 | [PDF](https://arxiv.org/pdf/2602.23206v1)

**作者:** Chung Hee Kim `[一作]`, Joshua Migdal `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了不同触觉接触模式（抓取-释放、指尖滑动、掌面滚动）对利用触觉手进行物体形状重建的影响，并提出信息理论探索框架实现高效采样。

**💡 创新点**

创新点在于将动态滑动和滚动接触与信息理论策略结合，显著提高触觉数据获取效率并减少交互次数。

**🔧 技术方法**

采用基于扩散的3D形状完成模型、信息理论信息增益评分、正向运动学与碰撞规划实现。

**📊 数据集**

训练使用由仿真触觉接触生成的合成数据集（球、盒、圆柱三类），并在UR5e+Inspire‑Robots手上进行真实实验。

**📈 对比分析**

与传统抓取-释放接触对比，滑动和滚动模式在平均Chamfer距离上提升约55%，交互次数下降34%，每次接触收集的触觉体积约4倍。

**⚠️ 局限性**

局限在于手掌曲面对平面物体的采样不足，指尖滑动和掌面滚动对硬件控制与时延要求高，且对复杂表面仍需进一步适应。

---

## 506. DigiArm: An Anthropomorphic 3D-Printed Prosthetic Hand with Enhanced Dexterity for Typing Tasks

**arXiv ID:** 2602.23017 | [PDF](https://arxiv.org/pdf/2602.23017v1)

**作者:** Dean Zadok `[一作]` (Technion), Nili Krausz `[通讯]` (Technion)

**通讯引用:** 476 | [OpenAlex ID](https://openalex.org/A5004603678)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文设计并实现了低成本、轻量化的3D打印义肢DigiArm，具有可调指间距、二维手腕运动和独立指控，以提升键盘打字和钢琴演奏的精细动作。

**💡 创新点**

创新点在于将可调指间距（Splay）与有向手腕偏转相结合，提供比现有义肢更高的精细控制，并以开源平台形式公开硬件与软件。

**🔧 技术方法**

采用3D打印零件、低成本直流电机与齿轮、拉线驱动、ESP32+Teensy 4.1闭环控制，并通过运动捕捉实现自然手势映射。

**📊 数据集**

实验使用11名受试者完成简短打字与钢琴练习，记录力学输出、指使用模式及上肢补偿运动，评估可调指间距与手腕运动对性能的影响。

**📈 对比分析**

对比三种实验条件（固定、可调指间距、全控制），结果显示在键盘任务中可调指间距和手腕运动可使肘部/肩部补偿运动减少约19%，键盘按键力可达3.6N，抓取日常物体成功率100%。

**⚠️ 局限性**

局限包括3D打印构件耐久性不足、拉线维护成本、整体系统重量为648g导致长时间使用疲劳、手指间距仅手动调节，缺乏实时自适应。

---

## 507. GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion

**arXiv ID:** 2602.22862 | [PDF](https://arxiv.org/pdf/2602.22862v1)

**作者:** Enda Xiang `[一作]` (Beihang University), Di Huang `[通讯]` (Beihang University)

**通讯引用:** 11501 | [OpenAlex ID](https://openalex.org/A5056972984)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种利用抓取先验的隐式扩散政策（GraspLDP），通过预训练的抓取检测网络提供的抓取姿态与抓取性图，指导动作潜空间中的扩散生成抓取动作，从而提升抓取精度与泛化。

**💡 创新点**

创新点在于：①将抓取姿态投射到动作潜空间进行指导；②在扩散过程中引入抓取性视觉提示，并用自监督重建目标强化对抓取性的关注；③设计Heuristic Pose Selector在抓取质量与运动学距离间做权衡；④两阶段训练（动作潜学习 + 隐式扩散）实现更高效、低延迟的闭环抓取。

**🔧 技术方法**

采用隐式扩散模型与VAE编码解码动作潜，预训练抓取检测网络（生成抓取姿态与抓取性图），自监督重建损失，SE(3)几何距离选择和实时Heuristic Pose Selector；在仿真与真实机器人上实现。

**📊 数据集**

使用LIBERO benchmark（约12k演示，20个物体）进行训练；评估使用GraspNet-1Billion的相似与新物体；真实世界实验在Franka Research 3上使用10个训练物体和13个测试物体。

**📈 对比分析**

与Diffusion Policy、OpenVLA、GraspVLA以及AnyGrasp等基线对比。仿真中GraspLDP成功率80.3%显著高于Diffusion Policy（62.8%）和OpenVLA（57.5%）；空间、对象、视觉泛化分别提升22.2%、46.8%、48.3%。真实世界中成功率84%，接近AnyGrasp；在多物体混乱场景SCR 92.3%与AnyGrasp相同，但单体成功率更高；动态抓取时表现优于Diffusion Policy和GraspVLA。

**⚠️ 局限性**

对高度柔性或脆弱物体（如鸡蛋、玻璃杯）仍存在局限，缺乏触觉、力/扭矩等高频感知；未来计划加入触觉与力/扭矩信息，以进一步提升抓取稳健性。

---

## 508. A Fusion of context-aware based BanglaBERT and Two-Layer Stacked LSTM Framework for Multi-Label Cyberbullying Detection

**arXiv ID:** 2602.22449 | [PDF](https://arxiv.org/pdf/2602.22449v1)

**作者:** Mirza Raquib `[一作]` (International Islamic University Chittagong), Nick Rahimi `[通讯]` (University of Southern Mississippi)

**通讯引用:** 257 | [OpenAlex ID](https://openalex.org/A5102764912)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将 BanglaBERT‑Large 与两层 LSTM 融合的多标签网络霸凌检测模型，解决单标签局限并针对 Bangla 语言实现自动识别。

**💡 创新点**

创新点在于将预训练的 BanglaBERT 上下文表示与 LSTM 的时序建模相结合，使用欠采样/过采样平衡多标签不平衡，并通过 LIME 实现可解释性。

**🔧 技术方法**

采用 Transformer（BanglaBERT‑Large）、双层 LSTM、二元交叉熵损失、AdamW/Adafactor 优化器、五折交叉验证以及 LIME 解释技术。

**📊 数据集**

使用公开的 "Bangla multilabel cyberbully, sexual harassment, threat, and spam detection" 数据集，包含 12,557 条 Bangla 社交媒体评论，标注五个二值标签。

**📈 对比分析**

通过与传统机器学习、单一 Transformer 及其他混合模型在准确率、Hamming loss、Precision、Recall、F1、MCC、Kappa、AUC 等多指标上对比，模型实现 94.31% 准确率、88.37% F1，显著优于先前最佳 94.32%。

**⚠️ 局限性**

仅针对 Bangla 语言，未覆盖代码混合或多语种文本，模型训练成本高，缺乏对不同平台和更细粒度文本的验证。

---

## 509. On the Computation Rate of All-Reduce

**arXiv ID:** 2602.22482 | [PDF](https://arxiv.org/pdf/2602.22482v1)

**作者:** Yufeng Zhou `[一作]` (University of North Texas), Hua Sun `[通讯]` (University of North Texas)

**通讯引用:** 3269 | [OpenAlex ID](https://openalex.org/A5016951190)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了多节点通信网络中All-Reduce求和操作的计算速率上限与下限，给出了通用的割集上界与基于时间共享的Reduce-广播线性规划下界，并在多种网络拓扑下给出近似最优速率或精确解。

**💡 创新点**

创新点在于将All-Reduce视为信息理论的计算问题，提出了统一的割集上界和Reduce-广播线性规划下界，并通过1-MAC-BC网络与其线性组合等构造，获得多种网络（如完全图、环路、超立方体等）计算速率的最优或近似最优表达式。

**🔧 技术方法**

主要技术包括割集论证、信息熵与独立性分析、基于聚合树（MAC树）与广播树（BC树）的时间共享策略、以及构造线性规划求解最大速率。

**📊 数据集**

本研究为理论分析，没有使用实验数据集，而是通过数学推导与图论模型来验证结果。

**📈 对比分析**

通过与已知的Ring-All-Reduce等常用算法比较，本文的下界在多种拓扑下与上界相差不超过两倍，证明了所提出方案在计算速率上具有良好近似性能。

**⚠️ 局限性**

局限性包括：对一般网络的上界与下界仍可能有较大保守性；对3节点完全图的速率仍存在 3/2≤R^*≤2 的未知区间；并且未考虑安全约束和联合编码等更高阶优化。

---

## 510. Entropy-Controlled Flow Matching

**arXiv ID:** 2602.22265 | [PDF](https://arxiv.org/pdf/2602.22265v1)

**作者:** Chika Maduabuchi `[一作]` `[通讯]` (William & Mary), Chika Maduabuchi (William & Mary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出一种在流匹配（flow matching）中加入熵率预算约束的框架——Entropy‑Controlled Flow Matching（ECFM），通过对连续方程路径的熵率进行上界限制，从而抑制低熵瓶颈并防止模式坍塌。

**💡 创新点**

创新点在于：①将熵率约束引入流匹配，形成一个凸优化问题；②证明该问题与Schrödinger桥（Schrödinger bridge）等价；③在纯运输特例下收敛于熵正则化OT（entropic OT），并在λ→0时Γ‑收敛回经典OT；④给出模式覆盖、密度下界以及对扰动的稳定性保证；⑤展示无熵约束时可出现近似最优但会坍塌的路径，证明约束必要性。

**🔧 技术方法**

使用的技术包括：连续方程与熵率公式、KKT/Pontryagin 最优性条件、凸分析与双对偶、Schrödinger桥的随机控制表示、Γ‑收敛理论、稳定性与扰动分析、以及训练时的投影原始-对偶或增广拉格朗日算法。

**📊 数据集**

本文主要为理论分析与方法论，未提供具体数据集实验，因而未使用任何公开数据集。

**📈 对比分析**

未进行实验性比较，也未给出具体性能指标；论文主要在理论层面证明了熵约束能提供模式覆盖和稳定性的保证。

**⚠️ 局限性**

局限性包括：①需要对熵率进行估计，计算和采样开销较大；②凸性与唯一性仅在Schrödinger桥或纯运输特例下保证，通用情况下可能有多解；③需要满足严格的正则性假设；④在实际训练中需要精细调节熵预算 λ，若 λ 选得过大仍可能出现低熵瓶颈；⑤目前尚未在大规模生成任务上验证其实际效果。

---

## 511. Learning Disease-Sensitive Latent Interaction Graphs From Noisy Cardiac Flow Measurements

**arXiv ID:** 2602.23035 | [PDF](https://arxiv.org/pdf/2602.23035v1)

**作者:** Viraj Patel `[一作]` (University of Bath), Katharine Fraser `[通讯]` (University of Bath)

**通讯引用:** 1620 | [OpenAlex ID](https://openalex.org/A5049583881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出一种基于物理信息的隐式相互作用图模型，用于从噪声严重的心血管流场中学习心脏涡流的关系结构，并用图熵作为疾病严重程度的定量指标。

**💡 创新点**

创新点包括在神经关系推断框架中加入循环能量门控、严重程度条件网络以及可变实体数的时间一致性约束，能够在不同疾病状态下产生可解释且与疾病严重程度单调相关的隐式图。

**🔧 技术方法**

主要技术为改进的神经关系推断（NRI）模型、基于Biot‑Savart定律的能量门控、条件网络、时间一致性和遮罩损失，以及图熵与疾病严重程度的统计关联分析。

**📊 数据集**

使用的实验数据集包括：①基于CFD模拟的主动脉收缩（CoA）数据，包含不同收缩程度和噪声水平；②使用羊心的超声Echo‑PIV实验数据，涵盖不同LVAD支持水平的左心室流场。

**📈 对比分析**

与原始NRI基线相比，本文模型在轨迹重建误差（MSE、MAE）和存在性预测准确率上显著提升；图熵与疾病严重程度呈高度负相关（Spearman ρ≈-0.95，p<0.001），且R²达到0.78（CoA）或0.85（LVAD），证明该指标在两种不同模态下均表现出良好可解释性与预测性能。

**⚠️ 局限性**

局限性在于数据集规模有限，缺乏独立外部验证；模型采用二元边类型可能过度简化涡流间的连续交互强度；以及对不同几何形状和患者群体的泛化能力尚未充分评估。

---

## 512. Performance and Experimental Analysis of Strain-based Models for Continuum Robots

**arXiv ID:** 2602.22854 | [PDF](https://arxiv.org/pdf/2602.22854v1)

**作者:** Annika Delucchi `[一作]` (University of Genova), and Matteo Zoppi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了基于变形场的连续机器人模型的形状重建性能，提出并验证了三阶应变插值方法，并与几何可变应变模型（GVS）进行对比。

**💡 创新点**

创新点在于提供了系统的实验验证框架，使用双臂操纵下的无外部传感器测量；并展示了三阶应变插值模型在准确性与计算效率之间的平衡，特别在弯曲和扭转耦合场景下的优势。

**🔧 技术方法**

采用了Lie群/李代数的几何描述、Cosserat杆理论、有限元与闭式三阶插值、Ritz方法构造GVS；实验使用KUKA机械臂和VICON视觉跟踪。

**📊 数据集**

实验数据来自两根杆（尼铁棒和玻纤棒）的标记点轨迹；模拟使用标准弹性参数（E、G）和三阶多项式基函数。

**📈 对比分析**

比较方法：定义整体位姿误差积分和最大误差指标；在模拟中与精确数值解对比，在实验中与VICON测量对比；结果显示三阶插值模型平均误差0.58-1.86%，最大误差3.62%，计算时间约0.32-0.64s，比GVS快约30%但准确略逊。

**⚠️ 局限性**

局限性包括：模型不考虑外部载荷和重力、对扭转的逼近受限导致精度下降；标记点的测量误差以及材料非线性、弯曲/扭转耦合的更复杂场景未覆盖。

---

## 513. When Should a Model Change Its Mind? An Energy-Based Theory and Regularizer for Concept Drift in Electrocardiogram (ECG) Signals

**arXiv ID:** 2602.22294 | [PDF](https://arxiv.org/pdf/2602.22294v1)

**作者:** Timothy Oladunni `[一作]` (Morgan State University), Clyde Baidoo `[通讯]` (Morgan State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了基于生理能量守恒的理论PECT以及能量约束表示学习框架ECRL，用于提升多模态ECG模型在面对虚拟漂移时的鲁棒性；

**💡 创新点**

创新点在于将信号能量变化与潜在表示漂移建立比例关系，提供判定何时模型应“换思路”的物理依据，并通过轻量级正则化实现能量一致性而不改动模型结构；

**🔧 技术方法**

采用能量约束正则化（BIT+PECT）、多模态Late‑Fusion（1D、2D、Transformer）以及统计相关性与回归斜率分析等技术；

**📊 数据集**

使用同步时间域、时频域与频域三模态的ECG数据集，包含四个诊断类别，约878训练样本、186测试样本；

**📈 对比分析**

与七种单模态/多模态基准模型对比，评估清洁/扰动准确率、鲁棒性差、潜在漂移幅度及能量相关性。ECRL显著降低漂移、提升扰动准确率（最高+266%），并在保持清洁准确率的前提下实现鲁棒性提升；

**⚠️ 局限性**

局限性包括仅在ECG上验证；扰动为人工合成的生理变化；仅针对Late‑Fusion架构；需要针对不同部署调节正则化超参数；未来需扩展至其他生理或非生理信号。

---

## 514. ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization

**arXiv ID:** 2602.22948 | [PDF](https://arxiv.org/pdf/2602.22948v1)

**作者:** Jiayu Chen `[一作]` (Peking University), Xiang chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 ToProVAR 框架，利用注意力熵在 token、层级和尺度三维上实现 VAR 模型的精细稀疏性分析与加速。

**💡 创新点**

创新点在于将注意力熵与三维稀疏性结合，动态确定尺度剪枝深度、通过 SVD 区分全局与细节层、并实现 FlashAttention 级别的在线熵计算，从而兼顾速度与质量。

**🔧 技术方法**

采用注意力熵评估、SVD 层级分离、动态尺度剪枝、熵门控制的 token 剪枝、以及 FlashAttention‑Entropy 方案。

**📊 数据集**

使用 Infinity‑2B 与 Infinity‑8B 两个 VAR 模型，评估基准包括 GenEval、DPG、HPSv2、ImageReward、MJHQ30K 等。

**📈 对比分析**

与 FastVAR、SkipVAR 等方法对比，ToProVAR 在 Infinity‑8B 上实现 3.4× 速度提升，同时保持或提升 GenEval、HPSv2、ImageReward 等质量指标，显著优于传统加速方案。

**⚠️ 局限性**

局限性包括对熵统计的依赖、SVD 计算虽小但仍增加开销、尚未验证在视频或多模态任务中的可扩展性。

---

## 515. Cytoarchitecture in Words: Weakly Supervised Vision-Language Modeling for Human Brain Microscopy

**arXiv ID:** 2602.23088 | [PDF](https://arxiv.org/pdf/2602.23088v1)

**作者:** Matthew Sutton `[一作]` (Institute of Neuroscience and Medicine), Christian Schiffer `[通讯]` (Helmholtz AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种弱监督、基于标签的视觉-语言模型，利用 CytoNet 视觉基础模型和 Llama‑3‑8B‑Instruct 语言模型，为人脑显微组织图像生成自然语言的细胞结构描述。

**💡 创新点**

创新点在于：① 用共享的区域标签而非配对图像‑文本数据实现弱监督；② 自动从文献挖掘并合成区域描述作为合成字幕；③ 采用轻量级 Flamingo 风格跨模态注意力，将 CytoNet 嵌入映射到 LLM；④ 构建了开放的 cytoarchitecture QA 基准，指导语言模型骨干的选择。

**🔧 技术方法**

技术包括 CytoNet 视觉基础模型、Llama‑3‑8B‑Instruct 语言模型、Flamingo 样式跨模态注意力机制、EBRAINS 知识图挖掘、文献提取与合成字幕、Qwen3‑Next 用于评测。

**📊 数据集**

使用 BigBrain 人脑银染细胞体组织扫描图像（约 539k 个 2048×2048 补丁，筛选为 57 个 Julich‑Brain 区域），以及通过 EBRAINS、Scopus、PubMed 等检索的 575 篇文献构成的区域描述语料。

**📈 对比分析**

评估方法：① 标签一致性测试，生成字幕中区域名与 CytoNet 预测一致率 90.6%；② 标签遮蔽后描述辨识度测试，Qwen3‑Next 在 8‑选一任务中识别率 68.6%；③ 通过自建 QA 基准筛选 LLM，Llama‑3‑8B 在 10,955 道多选题上得分 58.1%，明显优于其他开源模型。

**⚠️ 局限性**

局限性：① 监督仅为区域级别，未捕获补丁级细节；② 区域标签来自 CytoNet 预测，可能带噪声；③ 实验仅在单一 BigBrain 样本上，缺乏跨个体泛化评估；④ 合成字幕未覆盖边界区域的微观变化，描述精度受限。

---

## 516. Molecule Mixture Detection and Design for MC Systems with Non-linear, Cross-reactive Receiver Arrays

**arXiv ID:** 2602.22799 | [PDF](https://arxiv.org/pdf/2602.22799v1)

**作者:** Bastian Heinlein `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Vahid Jamali `[通讯]` (Technical University of Darmstadt)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套针对非线性、交叉反应传感器阵列的分子混合通信检测与字母设计方案，涵盖符号逐位检测、低复杂度检测、序列检测及自适应预编码；

**💡 创新点**

创新点在于：①利用无迹变换（UT）计算非线性交叉反应系统的第一、二阶矩，构造近似ML检测；②设计针对ISi的低复杂度统计检测和序列检测；③提出基于接收机特性的贪婪混合字母设计算法及自适应传输预编码；

**🔧 技术方法**

采用无迹变换、统计ISi知识、序列检测（BCJR前向），以及距离度量（基于均值与协方差）的字母优化；

**📊 数据集**

使用TGS800与TGS826的实际MOS传感器测量数据，以及在附录中生成的人工传感器阵列数据；

**📈 对比分析**

与基线（均值检测、kNN分类器）比较，实验表明UT‑基检测在无ISI与有ISI场景下SER均优于基线，序列检测在ISi显著提高性能；

**⚠️ 局限性**

局限性包括：需要对系统噪声与传感器非线性有准确的参数估计；序列检测对计算和存储资源要求高；自适应预编码受候选集大小限制，且对传感器漂移等实时变化的鲁棒性有限。

---

## 517. Fine-Tuning Without Forgetting In-Context Learning: A Theoretical Analysis of Linear Attention Models

**arXiv ID:** 2602.23197 | [PDF](https://arxiv.org/pdf/2602.23197v1)

**作者:** Chungpa Lee `[一作]` (Yonsei University), Kangwook Lee `[通讯]` (University of Wisconsin--Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究线性注意力模型中细调对零样本和少样本学习能力的影响，阐明不同细调策略的理论极限并通过实验验证；

**💡 创新点**

①推导出细调时全参数更新会显著削弱少样本学习；②证明仅更新value矩阵既能提升零样本表现，又能保留少样本能力；③引入辅助少样本损失来进一步提升目标任务少样本性能，但会降低对其他任务的泛化；

**🔧 技术方法**

线性自注意力Transformer、闭式解析理论、最优参数求解、正则化与损失组合、梯度下降与Annealing策略；

**📊 数据集**

线性回归任务数据（随机生成的任务向量和高斯噪声）以及大规模语言模型在MMLU基准上的评测；

**📈 对比分析**

对比预训练模型、全参数细调、仅value细调、带辅助少样本损失的细调；在理论上使用闭式误差表达式，在实验中用图表验证，结果显示仅value细调保持或提升少样本准确率而零样本准确率也有所提升；

**⚠️ 局限性**

理论仅针对线性注意力模型，未考虑软max注意力、多头结构和真实大模型的优化细节，实验对大模型的验证仍具示范性，未能证明在所有场景下均适用；

---

## 518. The Way We Notice, That's What Really Matters: Instantiating UI Components with Distinguishing Variations

**arXiv ID:** 2602.22436 | [PDF](https://arxiv.org/pdf/2602.22436v1)

**作者:** Priyan Vaithilingam `[一作]` (Apple), Titus Barik `[通讯]` (Apple)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

开发了一种基于符号分析与LLM的工具（Add‑on for Storybook）和方法，用以自动生成既真实又区分度高的UI组件实例，帮助前端开发者高效探索组件的设计空间。

**💡 创新点**

创新点在于：①提出“distinguishing variations”概念，将模仿性（mimesis）与区分度（distinctness）结合；②设计一种混合采样框架，先用静态分析识别视觉关键属性，再让LLM基于这些属性和用户指令生成既符合域语义又覆盖不同视觉模式的实例。

**🔧 技术方法**

主要技术包括：
- 静态程序分析（AST+信息流）与视觉影响评分；
- LLM（GPT‑4o）模仿采样，利用结构化提示生成完整属性配置；
- 覆盖分析模块衡量属性覆盖度；
- 与Storybook集成的可视化界面实现实例展示、编辑与覆盖可视化。

**📊 数据集**

使用的数据集与组件：
- 10个Ant Design 组件（共172属性）用于评估视觉影响评分准确率；
- 3个自定义产品卡/天气卡/个人资料卡（来自公司内部工作台）用于用户研究；
- 5个真实工作台共253个组件用于前期调研。

**📈 对比分析**

对比方法与性能：
- 技术评估中，视觉影响分类准确率为83.1%（最高92.9%，最低75.0%）；
- 用户研究中，参与者平均生成13.1个变体，覆盖率随迭代提升，三项指标（自然度、区分度、覆盖度）平均得分≥4（5分制）；
- 通过覆盖分析和提示交互，工具能够在数秒内生成针对性变体，显著提升探索效率。

**⚠️ 局限性**

limitations：
- 只聚焦视觉差异，未覆盖交互/行为变化；
- LLM易出现幻觉，需手动验证或后处理；
- 覆盖分析对字符串/数值域的近似可能不足；
- 评估样本量（12位开发者）有限，单机构背景；
- 目前仅支持单组件，无法自动处理跨组件交互或多层嵌套情形；
- 缺乏对生成极端/对抗性实例的支持，影响鲁棒性测试。

---

## 519. Differentially Private Truncation of Unbounded Data via Public Second Moments

**arXiv ID:** 2602.22282 | [PDF](https://arxiv.org/pdf/2602.22282v1)

**作者:** Zilong Cao `[一作]` (Northwest University), Hai Zhang `[通讯]` (Northwest University)

**通讯引用:** 10129 | [OpenAlex ID](https://openalex.org/A5100724090)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了利用公共数据的二阶矩进行数据标准化与合理截断（PMT）来处理未受限私有数据，并改进DP回归模型的鲁棒性和准确性。

**💡 创新点**

通过公共矩阵指导截断实现了对私有数据的近似等距化，显著降低了DP中矩阵逆的条件数，减少了对正则化参数的依赖，提升了理论误差界。

**🔧 技术方法**

采用公共矩阵预处理、DP高斯机制、闭式岭回归、DP牛顿法以及相关的误差分析与实验验证。

**📊 数据集**

在合成数据以及UCI数据集（白葡萄酒质量、联合循环发电机、银行营销、银行票据验证）上进行实验。

**📈 对比分析**

与传统DP-RR和DP-GD进行对比，实验表明PMT在误差、方差和鲁棒性方面均优于对照方法，尤其在高噪声或高维场景下表现更突出。

**⚠️ 局限性**

需依赖可获取的公共二阶矩数据，假设私有与公共样本独立同分布，且在高维和大规模数据下的计算成本与稳定性仍待进一步研究。

---

## 520. Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials

**arXiv ID:** 2602.22251 | [PDF](https://arxiv.org/pdf/2602.22251v1)

**作者:** Alex Morehead `[一作]` (Lawrence Berkeley National Laboratory), Michael W. Mahoney `[通讯]` (Lawrence Berkeley National Laboratory)

**通讯引用:** 24544 | [OpenAlex ID](https://openalex.org/A5033006662)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Zatom-1，一个统一的基础模型，既能在三维分子和材料上进行生成，又能在能量、力和性质等预测任务上实现多任务学习；

**💡 创新点**

创新点在于首次将多模态流匹配（同时处理离散原子类型和连续几何信息）与Transformer骨干结合，用单一模型完成分子与材料的联合预训练，从而实现跨域知识迁移并显著提升生成速度；

**🔧 技术方法**

核心技术包括Transformer trunk（含Query-Key Normalization、FlashAttention、SwiGLU），多模态流匹配（连续欧式、离散分布）、残差交叉注意力以及在预训练后冻结主干、增设任务专用Transformer分支的多任务微调；

**📊 数据集**

使用的主要数据集有MP20（材料）、QM9（分子）、GEOM-Drugs（大分子）、Matbench（材料性质）、OMol25与MPtrj（能量/力预测），并在这些数据集上进行预训练和微调；

**📈 对比分析**

与现有生成模型（MatterGen、DiffCSP、ADiT、Equivariant Diffusion、Symphony等）以及属性预测模型（DimeNet++、EGNN、PaiNN等）进行对比，Zatom-1在分子生成的Validity、Uniqueness和PoseBusters测试中达到或超过SOTA，在材料生成中获得高Validity和Novelty，并实现12.5×的采样速度提升，同时在多任务属性预测中表现优于单任务基线，显示出跨域迁移效应；

**⚠️ 局限性**

局限性包括预训练数据规模相对有限，导致在某些属性预测上无法持续提升；对大规模材料库的覆盖不足；并且模型仍需进一步探索更高阶的O(3)等变换不变性以提升生成质量和物理一致性。

---

## 521. A Thermodynamic Structure of Asymptotic Inference

**arXiv ID:** 2602.22605 | [PDF](https://arxiv.org/pdf/2602.22605v1)

**作者:** Willy Wong `[一作]` (Kyushu University), Willy Wong `[通讯]` (Kyushu University)

**通讯引用:** 2075 | [OpenAlex ID](https://openalex.org/A5024382401)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于样本量与方差的热力学框架，将统计推断的渐近性质与热力学定律类比，定义熵、温度、功等概念并推导逆第二定律、能量守恒与第三定律等结果。

**💡 创新点**

创新点在于将统计推断映射为热力学状态空间，揭示熵-信息关系的“温度”Θ、逆第二定律逆向不等式、以及信息效率的Carnot式上界，统一了de Bruijn识别与I–MMSE关系。

**🔧 技术方法**

使用渐近正态性、Fisher信息可加性、信息几何与热力学对称性等理论工具，构建状态函数、积分因子与第一、第二、第三定律形式的推导。

**📊 数据集**

该工作为理论性研究，不涉及具体实验或公开数据集；主要以感知神经系统与计量学为应用示例进行概念性说明。

**📈 对比分析**

由于是纯理论框架，没有实验比较；通过对比热力学与推断的数学结构，论证了信息效率上界与传统Cramér–Rao界限的一致性。

**⚠️ 局限性**

局限在于仅适用于大样本渐近正态条件、缺乏可直接观测的效能指标、对真实非独立非同分布情形的适用性有限。

---

## 522. EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents

**arXiv ID:** 2602.23205 | [PDF](https://arxiv.org/pdf/2602.23205v1)

**作者:** Wenjia Wang `[一作]` (University of Hong Kong), Taku Komura `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了 EmbodMocap，一套仅用两部移动 iPhone 即可实现 4D 人体与场景度量精确重建的低成本便携系统。

**💡 创新点**

通过联合校准双摄 RGB‑D 序列，将人体运动与场景几何统一到同一世界坐标系，并在无需靶标或佩戴设备的条件下实现场景一致的高质量重建。

**🔧 技术方法**

结合双视角 RGB‑D 同步采集、COLMAP 注册、SLAM、SMPL 参数优化、前馈深度学习模型以及物理仿真+强化学习等技术实现数据采集与后处理。

**📊 数据集**

收集了 EmbodMocap 自建的多视角四维人体与场景数据集，并与光学捕捉标注数据做对比验证。

**📈 对比分析**

通过与单目、单 iPhone 以及光学捕捉基准的实验对比，双视角设置在深度不确定性、姿态对齐和重建精度方面均优于单视角模型，在三项下游任务中亦取得显著性能提升。

**⚠️ 局限性**

受限于 iPhone LiDAR 的约 5 m 范围、移动物体场景导致 SLAM 失效、强光环境下 COLMAP 注册失败等问题，导致深度缺失和配准错误。

---

## 523. $φ$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models

**arXiv ID:** 2602.22601 | [PDF](https://arxiv.org/pdf/2602.22601v1)

**作者:** Thanh-Dat Truong `[一作]` (University of Arkansas), Khoa Luu `[通讯]` (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种公平直观偏好优化（ϕ-DPO）框架，用于大型多模态模型的持续学习，解决灾难性遗忘和数据不平衡导致的公平性问题。

**💡 创新点**

创新点在于将RLHF改为直接偏好优化并引入可调焦点参数γ的ϕ-DPO损失，理论上证明其对KL散度的双向界限；构建持续学习所需的对比偏好标注；并在多模态基准上实现SOTA。

**🔧 技术方法**

使用 Direct Preference Optimization、Fairness DPO 损失、LoRA 模型微调、CLIP‑ViT‑L‑14+Vicuna 7B 视觉‑语言后端、理论KL界限分析以及对比偏好数据构造技术。

**📊 数据集**

实验基准包括：CoIN（ScienceQA、TextVQA、ImageNet、GQA、VizWiz、Grounding、VQAv2、OCR‑VQA）；MLLM‑CL Domain（Remote Sensing、Medical、Autonomous Driving、Science、Finance）；MLLM‑CL Ability（OCR、Math & Logic、Visual Perception、GUI Agent）。

**📈 对比分析**

与LoRA、Mixture‑of‑Expert、DISCO、MR‑LoRA 等方法在 Last Accuracy、MFT、MFN、MAA、BWT 等指标上对比，ϕ‑DPO 在各基准上均获得最高 MAA、最小 BWT，且在各任务上显著优于对手，显示出更强的知识保留和公平性。

**⚠️ 局限性**

局限性包括：超参数 β、γ 的调优难度大；对比偏好数据的构造依赖人工校验，易受标签不稳定、类别不平衡和域漂移影响；理论分析基于简化假设；在更大规模多模态模型上的可扩展性与效率仍待进一步验证。

---

## 524. Queue occupancy and server size distribution of a queue length dependent vacation queue with an optional service

**arXiv ID:** 2602.22295 | [PDF](https://arxiv.org/pdf/2602.22295v1)

**作者:** Ashish Verma `[一作]` (Visvesvaraya National Institute of Technology), Sourav Pradhan `[通讯]` (Visvesvaraya National Institute of Technology)

**通讯引用:** 182 | [OpenAlex ID](https://openalex.org/A5029912752)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了一类无限容量的离散时间批量到达排队模型，在该模型中服务器先进行一次强制性的基本服务（First Essential Service, FES），随后可根据队列长度决定是否为部分客户继续提供可选服务（Second Optional Service, SOS）。模型进一步引入批量大小相关的服务时间、队列长度相关的单/多休假策略，并使用补充变量技术（SVT）推导了系统在服务完成和休假完成时的双变量概率生成函数（PGF）以及完整的联合分布。

**💡 创新点**

创新点包括：①将批量到达、两阶段服务（FES+SOS）、批量大小相关服务时间与队列长度相关休假策略结合于离散时间排队模型；②通过SVT直接求解双变量PGF，避免了构造大规模转移概率矩阵（TPM）的繁琐步骤；③给出了休假完成后任意时刻的概率分布与性能指标的闭式计算方法；④在数值示例中首次使用离散相位类型（DPH）服务时间分布和负二项休假时间分布，完整展示了能耗、吞吐率、平均队列长度等多项指标的权衡与敏感性。

**🔧 技术方法**

技术与方法：补充变量技术（SVT）、概率生成函数（PGF）推导、相位类型（DPH）分布参数化、负二项分布休假建模、数值求解与根分析、性能指标计算（平均队列长度、平均等待时间、能耗因子、系统周期等）。

**📊 数据集**

没有使用真实业务数据；所有实验均基于合成参数设定，例如：几何到达率、DPH服务时间分布（不同批量大小对应不同参数）、SOS服务时间为负二项分布、休假时间为负二项/几何分布等。通过这些参数化示例来验证公式的正确性和系统性能。

**📈 对比分析**

比较方法：①通过设定特定参数（如p=0、休假时间为0、a=b=1等）恢复已有文献中的模型（Geo/G/1、Geo^X/G^n/(a,b)/1 等），验证推导结果与已知闭式解一致；②在数值示例中绘制吞吐率与能耗因子、休假速率与平均队列长度、服务速率对周期/空闲期的影响等图表，展示多种参数组合下的性能权衡。性能方面，模型能够给出系统吞吐率、平均队列长度、能耗因子、期望空闲期、循环长度等指标，并通过敏感性分析说明参数调整的实用意义。

**⚠️ 局限性**

局限性：①假设无限容量排队系统，未考虑缓冲区溢出或失效；②仅适用于单服务器、离散时间模型，无法直接推广到连续时间或多服务器场景；③模型假设到达、服务和休假均为无记忆（几何/DPH/负二项）分布，忽略了实际网络中可能出现的长尾分布或到达间隔相关性；④公式虽避免了 TPM，但整体数学推导仍较为复杂，数值求解对大批量尺寸和高阶相位类型可能面临计算瓶颈；⑤未进行实测或实验室数据验证，缺乏真实业务案例的佐证。

---

## 525. Induction Meets Biology: Mechanisms of Repeat Detection in Protein Language Models

**arXiv ID:** 2602.23179 | [PDF](https://arxiv.org/pdf/2602.23179v1)

**作者:** Gal Kesten-Pomeranz `[一作]` (Technion - Israel Institute of Technology), Yonatan Belinkov `[通讯]` (Kempner Institute, Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究蛋白质语言模型如何识别并完成序列重复，解释其内部机制。

**💡 创新点**

发现模型的重复识别主要由诱导头和生物学专门化神经元共同驱动，并建立了三阶段机制。

**🔧 技术方法**

使用注意力头、MLP神经元的机制分析、电路发现和归因补丁技术。

**📊 数据集**

使用合成重复序列、自然相同重复序列和含有突变的自然近似重复蛋白质数据集。

**📈 对比分析**

通过与模型自身电路的交叉任务一致性评估，准确率在合成/相同重复约99%，近似重复约79%，电路在不同任务上保持高可信度。

**⚠️ 局限性**

仅限两次重复、最多50%突变的序列，且电路发现依赖人工定义概念，可能遗漏其他模式。

---

## 526. Prior Knowledge-enhanced Spatio-temporal Epidemic Forecasting

**arXiv ID:** 2602.22270 | [PDF](https://arxiv.org/pdf/2602.22270v1)

**作者:** Sijie Ruan `[一作]` (Beijing Institute of Technology), Shuliang Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 4177 | [OpenAlex ID](https://openalex.org/A5000423121)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于混合模型的时空疫情预测框架STOEP

**💡 创新点**

通过引入隐式时空先验（CAL、SPE）与专家先验（FMF）提升模型对弱信号、空间关系和参数估计的鲁棒性

**🔧 技术方法**

使用自注意力聚类模式、图卷积网络、可学习阈值抑制机制以及MetaSIR机制

**📊 数据集**

COVID‑19日本47个都道府县每日病例与Facebook移动范围数据；流感中国11城市每日病例与百度迁徙图

**📈 对比分析**

在COVID‑19和流感数据集上与SIR、MetaSIR、各类深度图神经网络及混合模型做对比，STOEP平均RMSE降低约11.1%，在7天预测时更显优势

**⚠️ 局限性**

依赖高质量流动数据与参数先验，模型对不同疾病或地区的迁移性仍需进一步验证

---

## 527. Invariant Transformation and Resampling based Epistemic-Uncertainty Reduction

**arXiv ID:** 2602.23315 | [PDF](https://arxiv.org/pdf/2602.23315v1)

**作者:** Sha Hu `[一作]` `[通讯]` (Huawei Technologies Sweden AB), Sha Hu (Huawei Technologies Sweden AB)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在已训练好的AI模型上，通过对输入做不变变换生成多样本进行推理，并聚合结果以提升推理精度。

**💡 创新点**

创新点在于利用模型的不可变变换属性，证明多样本推理能降低认识不确定性，并给出最优组合权重的理论依据。

**🔧 技术方法**

采用Transformer Encoder网络作为AI模型，并使用共轭、置换、符号翻转等不变变换来产生多样本。

**📊 数据集**

主要使用5G MIMO通道模型（Rayleigh、ETU‑70Hz）下生成的模拟数据，调制阶数涵盖64QAM和256QAM。

**📈 对比分析**

与传统ML、LMMSE、QRM检测以及单一AI模型推理进行对比，Resampling方法在1–0.5 dB的SNR提升误码率，性能接近QRM级别。

**⚠️ 局限性**

局限性包括：高噪声场景下错误相关性高导致提升有限；需先验知道变换的不可变性，若未知则需启发式重采样，理论分析难以直接应用。

---

## 528. RETLLM: Training and Data-Free MLLMs for Multimodal Information Retrieval

**arXiv ID:** 2602.22278 | [PDF](https://arxiv.org/pdf/2602.22278v1)

**作者:** Dawei Su `[一作]`, Dongsheng Wang `[通讯]` (Shenzhen University)

**通讯引用:** 37274 | [OpenAlex ID](https://openalex.org/A5083667162)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 RETLLM 框架，利用预训练多模态大语言模型在无训练、无数据的前提下完成多模态检索任务。

**💡 创新点**

将检索任务重新定义为相似度分数生成，采用粗-精两步筛选、视觉增强与熵基决策相结合的零训练方案。

**🔧 技术方法**

使用多模态 LLM 生成相似度分数、CLIP 进行粗筛选、视觉重新注入机制、熵值置信度校正以及粗-精检索管线。

**📊 数据集**

在 Flickr30K、COCO、ShareGPT4V、Urban1K、SugarCrepe 以及 MMEB 六个基准数据集上进行评估。

**📈 对比分析**

与 CLIP、EVA‑CLIP、E5‑V、VLM2Vec、UniME 等零训练基线对比，RetLLM 在 Flickr30K R@1 94.5%、ShareGPT4V R@1 94.2%、SugarCrepe R@1 96.2% 以及 MMEB Precision@1 54.2% 等指标上均明显优于对手。

**⚠️ 局限性**

仍依赖大规模预训练模型，推理成本高；视觉增强与熵决策需要手动调参；在极端模态混合或新颖场景下性能可能受限。

---

## 529. Closing the gap on tabular data with Fourier and Implicit Categorical Features

**arXiv ID:** 2602.23182 | [PDF](https://arxiv.org/pdf/2602.23182v1)

**作者:** Marius Dragoi `[一作]` (Bitdefender), Elena Burceanu `[通讯]` (Bitdefender)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究深度学习在表格数据上的表现差距，提出通过识别隐式类别特征并使用Learned Fourier Features进行特征预处理，以提升MLP和1D卷积ResNet的性能。

**💡 创新点**

首次引入Implicitly Categorical Feature Detection（ICF）统计检验方法识别隐式类别特征，并结合LFF补偿过平滑偏差，显著缩小深度学习与XGBoost之间的性能差距。

**🔧 技术方法**

使用卡方检验、ANOVA、互信息等统计检验进行特征分类，利用Conv1x1LFF和LinearLFF实现Learned Fourier Features映射，训练MLP和1D卷积ResNet两种基线模型，并进行大规模随机搜索调参。

**📊 数据集**

采用公开的表格数据基准（原69项任务，剔除性能低于0.1的回归任务后为68项），涵盖二分类与回归任务，且包含数值、类别混合特征。

**📈 对比分析**

与XGBoost、未预处理的MLP和ResNet做对比，使用多次随机搜索和k折验证，结果显示带ICF+LFF的模型在大多数任务上逼近或超越XGBoost，出现显著“spiking”性能提升。

**⚠️ 局限性**

仅在MLP和ResNet两种基线上验证，ICF对低基数特征的阈值设定需要经验；缺乏对更复杂深度学习架构（如Transformer）的推广；对特征稀疏度和噪声鲁棒性仍待深入研究。

---

## 530. DyaDiT: A Multi-Modal Diffusion Transformer for Socially Favorable Dyadic Gesture Generation

**arXiv ID:** 2602.23165 | [PDF](https://arxiv.org/pdf/2602.23165v1)

**作者:** Yichen Peng `[一作]` (Institute of Science Tokyo), Kris Kitani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 14601 | [OpenAlex ID](https://openalex.org/A5037322163)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DyaDiT，用扩散变压器结合双人音频与社交上下文生成符合社交情境的对话手势。

**💡 创新点**

创新点包括：Orthogonalization Cross Attention（ORCA）模块去除两方音频重叠；可选的动作词典提升风格多样性；将伙伴动作、关系类型、人格特质等多模态信息联合编码。

**🔧 技术方法**

技术：基于扩散概率模型的DiT架构、VQ‑VAE量化、ORCA交叉注意力、动作词典、FiLM和CFG。

**📊 数据集**

使用Seamless Interaction数据集（约3000条约182小时）进行训练与评估。

**📈 对比分析**

与ConvoFusion、Audio2PhotoReal等基线在Fréchet Distance和多样性指标上对比，DyaDiT在FD（静态/动态）和多样性上均明显优于基线，并在用户研究中获得约74%/70%/67%的偏好。

**⚠️ 局限性**

局限：目前仅生成上身手势；对人格冲突与多样性控制仍有限，未来计划扩展至全身与面部、实现音频中性化。

---

## 531. Generalization Bounds of Stochastic Gradient Descent in Homogeneous Neural Networks

**arXiv ID:** 2602.22936 | [PDF](https://arxiv.org/pdf/2602.22936v1)

**作者:** Wenquan Ma `[一作]`, Jingqin Yang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过分析同质性神经网络在随机梯度下降中的有效步长，给出了在非凸训练下可接受的步长衰减率为Ω(1/√t)的通用性泛化上界，并在此基础上证明了相应的优化收敛性。

**💡 创新点**

创新点在于：①将网络同质性引入算法稳定性分析，突破传统需要步长 η_t=1/t 的限制；②提出有效步长投影到单位球上的概念，允许实际步长更大；③扩展到非 Lipschitz、不同损失、激活和结构（如 VGG、ResNet）并兼容多轮训练。

**🔧 技术方法**

主要技术包括算法稳定性（Uniform / On‑average model stability）、同质性网络的定义与构造、近似光滑性假设、有效步长分析与投影技巧，以及针对非 Lipschitz 的稳定性度量。

**📊 数据集**

实验使用 CIFAR‑10 数据集，并在 ResNet‑18 上验证不同步长调度（Θ(1/t)、Θ(1/√t)、Θ(1/√t‖w_t‖)）对训练与测试准确率的影响。

**📈 对比分析**

与传统 1/t 步长相比，Ω(1/√t) 方案在训练速度和最终测试准确率上均有提升；在优化方面可实现多项式收敛，优于对数收敛；与现有 PL 条件或梯度范数等方法相比，所给的泛化上界更宽泛且不需要过于严格的假设。

**⚠️ 局限性**

局限性包括：需要满足同质性与 H>2 的条件；假设损失有界且近似光滑，实际网络可能不完全符合；理论上界仍相对保守；实验验证有限，主要集中在浅层/中层网络，尚未在更深或不同任务上系统评估。

---

## 532. ConstraintBench: Benchmarking LLM Constraint Reasoning on Direct Optimization

**arXiv ID:** 2602.22465 | [PDF](https://arxiv.org/pdf/2602.22465v1)

**作者:** Joseph Tso `[一作]` (Haladir Research Team), Jibran Hutchins `[通讯]` (Haladir Research Team)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ConstraintBench，一个评估大型语言模型（LLM）在直接解决受限优化问题（而非仅仅生成求解代码）的基准，覆盖10个运筹学领域共200个经 Gurobi 验证的任务。

**💡 创新点**

创新点在于：① 直接评估模型生成可行解的能力；② 对每个约束逐一检验并给出诊断；③ 使用求解器验证的真值而非人工标注，保证数据质量；④ 通过对模型在各域的细粒度分析揭示可行性与最优性解耦现象。

**🔧 技术方法**

使用技术包括：LLM（GPT‑5.2‑pro、Claude Opus 4.6/4.5、o4‑mini、Gemini 3 Pro）进行一次性推理；Gurobi Optimizer 对每个任务求解并生成最优解；Python/Pydantic 构建模型、生成任务与验证；统计分析与可视化工具评估误差与性能。

**📊 数据集**

数据集：ConstraintBench 自生成的200个任务，来源于10个领域的种子组合，通过 LLM 生成场景后 Gurobi 求解并验证；每个任务包括自然语言描述、实体列表、约束与目标，所有真值均由 Gurobi 证明。

**📈 对比分析**

比较方法：在统一条件下（单轮结构化 JSON 输出、无反馈、温度 0）对六个顶尖模型进行评测；指标为可行性（constraint satisfaction）、最优性（≤0.1% 间距）与两者联合率；结果显示最佳模型 GPT‑5.2‑pro 可行性 65%，条件最优率 95%，联合率仅 30.5%。各域差异显著，最难域如人力排班可行性仅 0.8%。

**⚠️ 局限性**

局限性：① 任务量仅 200 例，无法覆盖更广泛的运筹场景；② 单轮推理不反映多轮交互的实际应用；③ 0.1% 最优阈值过严，导致联合率被低估；④ 直接求解设置不一定是生产部署的首选模式；⑤ 数据集缺乏如网络设计、切割库存等重要领域；⑥ 误差主要源于模型在解释约束与实体时的失误，需进一步提升模型的结构化推理能力。

---

## 533. Uncertainty-Aware Calculation of Analytical Gradients of Matrix-Interpolatory Reduced-Order Models for Efficient Structural Optimization

**arXiv ID:** 2602.23314 | [PDF](https://arxiv.org/pdf/2602.23314v1)

**作者:** Marcel Warzecha `[一作]` (Technical University of Munich), Gerhard Müller `[通讯]` (Technical University of Munich)

**通讯引用:** 13127 | [OpenAlex ID](https://openalex.org/A5047665746)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于矩阵插值的参数化模型降阶（pMOR）和梯度驱动的自适应采样框架，用以高效优化动态系统的几何参数，减少全阶模型（FOM）计算量并提高局部精度。

**💡 创新点**

创新点在于将稀疏贝叶斯回归与Thompson采样相结合，对矩阵插值模型的不确定性进行量化，并通过解析的伴随灵敏度实现梯度搜索，实现在优化过程中对采样点进行探索-利用平衡的自适应更新。

**🔧 技术方法**

主要技术包括迭代有理 Krylov（IRKA）降阶、全局重投影构造统一基底、稀疏贝叶斯回归（SBR）构建矩阵系数的后验分布、Thompson采样决策采样点、解析伴随敏感性分析获取梯度以及 MATLAB fmincon 的梯度优化。

**📊 数据集**

使用的案例数据集为二维 Timoshenko 梁模型（约 4800 DOF）和 Kelvin 单元结构拓扑（约 10800 DOF）两种有限元模型，分别在给定频率范围和 L_k 范数下进行尺寸优化。

**📈 对比分析**

与非自适应全网格采样（10×10 FFD）和有限差分梯度优化（FOM 与 ROM）进行对比。自适应采样在总样本数上比全网格少 30–50%，在局部精度上显著优于全网格，但由于梯度优化和贝叶斯回归开销，整体壁钟时间未能实现下降，梯度优化仍占约 70% 的计算时间。

**⚠️ 局限性**

局限主要体现在：1）梯度优化在早期迭代中效率低下，导致整体时间提升；2）SBR 与Thompson采样的参数调优缺乏系统化；3）高阶 L_k 范数导致优化更难收敛；4）对大规模模型的计算复杂度尚未完全评估，需进一步优化子程序和探索更高效的采样策略。

---

## 534. Toward Automatic Filling of Case Report Forms: A Case Study on Data from an Italian Emergency Department

**arXiv ID:** 2602.23062 | [PDF](https://arxiv.org/pdf/2602.23062v1)

**作者:** Gabriela Anna Kaczmarek `[一作]`, Bernardo Magnini `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个公开的意大利急诊科临床笔记与结构化CRF对齐的数据集，并在此基础上开展自动CRF填充研究；

**💡 创新点**

首次公开高质量、真实医院数据的CRF填充基准，展示了LLM在多语言、零样本情境下的可行性；

**🔧 技术方法**

使用零样本提示策略，调用LLaMA‑3.1‑8B Instruct进行CRF填充；

**📊 数据集**

由290份已匿名化的意大利急诊科临床笔记组成，配合134项CRF模板；

**📈 对比分析**

与最频繁值基线对比：micro‑F1从0.963下降至0.920，但macro‑F1提升至0.548，表明模型能部分识别少数类值；

**⚠️ 局限性**

数据高度不平衡、注释稀疏且大多为unknown，导致模型整体准确率偏高且对非unknown值恢复率低，需进一步改进算法与训练策略。

---

## 535. MSINO: Curvature-Aware Sobolev Optimization for Manifold Neural Networks

**arXiv ID:** 2602.22937 | [PDF](https://arxiv.org/pdf/2602.22937v1)

**作者:** Suresan Pareth `[一作]` (National Institute of Technology Karnataka), Suresan Pareth `[通讯]` (National Institute of Technology Karnataka)

**通讯引用:** 12 | [OpenAlex ID](https://openalex.org/A5040634867)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种针对Riemannian流形上神经网络的曲率感知Sobolev优化框架MSINO，利用协变梯度监督、平行输运与Laplace–Beltrami正则化实现对函数值与梯度的联合学习，并给出了梯度下降、随机梯度下降以及二阶Newton式优化的收敛保证。

**💡 创新点**

创新点包括①在流形上构造协变Sobolev损失并证明其具有可控的Lipschitz光滑常数与PL不等式；②提出自适应方差感知的λ调度策略，显著降低噪声底；③设计两步Newton–Sobolev方法实现局部二次收敛；④证明在离散网格上Laplace–Beltrami正则化成本为零，并在SO(3)、SE(3)等Lie群上给出闭式光滑常数；⑤将上述理论与实践统一，展示跨域（表面、球面、Lie群）7倍以上的收敛加速。

**🔧 技术方法**

核心技术包括协变Sobolev损失设计、平行输运与Jacobi控制、Laplace–Beltrami算子与cotangent拉普拉斯、Riemannian梯度与随机梯度、PL不等式与收敛分析、Newton–Sobolev二阶优化、方差感知λ调度与自适应步长上限、离散差分与有限元逼近、Lie群几何与左不变度量、以及可视化与实验评估工具。

**📊 数据集**

使用了三类数据集：2562点脑表面厚度网格（表面曲面），64×64经纬度温度格点（球面），以及1000条SO(3)姿态轨迹（机器人方向）。

**📈 对比分析**

与传统Sobolev训练、基于PINN的欧氏Sobolev方法以及无曲率信息的随机梯度下降等方法进行对比。MSINO在所有任务上实现了线性收敛率、局部二次收敛，并通过自适应λ调度将噪声底降低3.7倍；实验结果表明整体迭代次数比先前方法少约7倍，损失下降更快、最终误差更低。

**⚠️ 局限性**

主要局限包括：依赖流形的全局几何信息（曲率、射影半径、平行输运误差）导致在高曲率或奇异流形上可能出现不稳定；SO(3)等Lie群上对指数映射的数值不稳定需要使用退化回旋变换；高阶Sobolev空间（H²）与更高阶梯度监督尚未实现；大规模流形的几何常数计算与存储仍是潜在瓶颈。

---

## 536. Ruyi2 Technical Report

**arXiv ID:** 2602.22543 | [PDF](https://arxiv.org/pdf/2602.22543v1)

**作者:** Huan Song `[一作]` (Institute of Artificial Intelligence), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**通讯引用:** 61874 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Ruyi2家族模型（Familial Models），通过共享Transformer主干和多层早期出口实现可变深度推理；

**💡 创新点**

核心创新在于：①将早期出口与共享主干相结合的家族模型架构；②使用3D并行训练提升训练效率；③在1.7B子模型上实现DaE（先扩展后压缩）策略，包括稳定块扩展与随机内部初始化，随后采用SVD低秩压缩；

**🔧 技术方法**

技术包括Megatron-LM框架、3D并行训练、联合多分支预训练、梯度聚合与流水线感知反向传播、随机内部初始化、零残差初始化、SVD低秩分解、GRPO强化学习；

**📊 数据集**

使用自建的高质量多模态数据集（包含多语种网络文本、科学文献、代码、数学符号）、800B token的继续预训练语料、约400万高质量指令样本；

**📈 对比分析**

与Qwen3系列模型在MMLU、GSM-8K、MATH等基准上比较，Ruyi2在相同参数规模下取得显著提升（1.7B MMLU+22点，8B GSM-8K+7点，14B平均分+5点），同时训练效率提升2–3倍；

**⚠️ 局限性**

局限性包括1.7B子模型在极复杂推理任务仍弱于大模型、仍存在生成偏差与长上下文稳定性问题、对多模态与极稀疏化技术尚未覆盖，需进一步改进多任务适配与生成质量。

---

## 537. Model Agreement via Anchoring

**arXiv ID:** 2602.23360 | [PDF](https://arxiv.org/pdf/2602.23360v1)

**作者:** Eric Eaton `[一作]` (University of Pennsylvania), Jessica Sorrell `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文研究了如何控制机器学习模型之间的预测不一致性，提出了一种基于模型平均的简单通用技术来证明独立模型不一致性的界限，并将其应用于四种常用的机器学习算法。

**💡 创新点**

创新点在于提出了一种中点锚定方法，通过该方法可以在不需要模型之间的交互或协调的情况下，驱动模型不一致性趋近于零。

**🔧 技术方法**

使用了中点锚定方法，该方法通过分析模型的平均值来界定模型不一致性，并应用于堆叠聚合、梯度提升、神经网络训练和回归树训练等算法。

**📊 数据集**

使用了独立样本训练的模型，具体数据集未明确给出，但假设模型是从相同分布中独立抽样的。

**📈 对比分析**

通过与现有模型的比较，证明了在堆叠聚合中，模型不一致性可以通过增加模型数量k来降低；在梯度提升中，模型不一致性以O(1/k)的速率降低；在神经网络和回归树中，模型不一致性也可以通过增加网络大小或树深度来降低。

**⚠️ 局限性**

限制在于该方法依赖于模型的复杂性和训练过程的随机性，且在某些情况下可能无法保证模型的全局最优性。

---

## 538. Improving Spatial Allocation for Energy System Coupling with Graph Neural Networks

**arXiv ID:** 2602.22249 | [PDF](https://arxiv.org/pdf/2602.22249v1)

**作者:** Xuanhao Mu `[一作]` (Karlsruhe Institute of Technology), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 4894 | [OpenAlex ID](https://openalex.org/A5014228448)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了基于异构图神经网络的二维空间分配框架，生成高分辨率网格点分配权重并与Clustering Induced Voronoi Diagram (CIVD) 结合，用于能量系统模型的空间耦合；

**💡 创新点**

首次将异构 GNN 应用于 Grid Point Modeling，利用多源地理与宏观经济特征进行自监督学习，生成物理可行的权重，并通过自监督代理指标克服无真实标签的难题；

**🔧 技术方法**

使用异构图构建、Heterogeneous Graph Transformer（HGT）编码器、关系成本加分组 Softmax 权重预测、KL 散度自监督损失，以及 CIVD 空间分配；

**📊 数据集**

Great Britain 11 kV 变电站峰值需求数据、ITL 区域宏观统计（人口、GVA）、OSM 地理数据及 OpenStreetMap 等公开数据集；

**📈 对比分析**

与传统 Voronoi Diagram、GPM、CIVD 等基线在 16 个 ITL 区域进行 RMSE 对比；CIVD‑GNN‑GPM 在训练集平均降低 4.87% RMSE，在测试集同样保持优势，部分地区误差略增；

**⚠️ 局限性**

依赖 OSM 数据质量；自监督代理指标与实际电力使用可能不完全对应，导致在缺乏变电站区域权重过度集中；模型与空间分配子模块兼容性不足，影响整体性能。

---

## 539. Communication-Guided Multi-Mutation Differential Evolution for Crop Model Calibration

**arXiv ID:** 2602.22804 | [PDF](https://arxiv.org/pdf/2602.22804v1)

**作者:** Sakshi Aggarwal `[一作]`, Mukesh Saini `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种多变异、通信引导、动态算子选择的差分进化算法DE‑MMOGC，用以在不确定环境下优化作物模型的天气参数，从而提升对叶面积指数（LAI）的预测精度。

**💡 创新点**

创新点在于将三种标准变异算子并行分层执行，并通过通信机制在子种群间共享精英解以及动态算子选择机制自动调整算子使用比例，从而兼顾探索与收敛并显著缓解单变异DE的过早收敛问题。

**🔧 技术方法**

所用技术包括多变异差分进化、通信引导机制、动态算子选择、Ensemble Kalman Filter（EnKF）数据同化、以及简化版WOFOST作物生长模型。

**📊 数据集**

实验数据集涵盖了四种作物（小麦HD‑2967、Lok‑1、稻米、棉花）在不同水分管理条件下的观测LAI，并在观测中人为加入10%噪声以模拟缺失与不确定性。

**📈 对比分析**

与遗传算法（GA）、粒子群优化（PSO）、哈巴鲁·霍尔登优化（HHO）以及传统单变异DE进行对比，DE‑MMOGC在所有作物上都实现了至少20%更低的均方误差（MSE），并在相关性与平均绝对误差（MAE）指标上也表现优异，说明其在不确定环境下的优化与同化性能显著优于基线方法。

**⚠️ 局限性**

局限性包括仅在简化版WOFOST上验证、未进行多目标优化、缺乏对更完整作物模型（如DSSAT、APSIM）的测试以及未深入评估在更大规模农田数据集上的可扩展性。

---

## 540. Sustainable Multi-Agent Crowdsourcing via Physics-Informed Bandits

**arXiv ID:** 2602.22365 | [PDF](https://arxiv.org/pdf/2602.22365v1)

**作者:** Chayan Banerjee `[一作]` (Queensland University of Technology), Chayan Banerjee `[通讯]` (Queensland University of Technology)

**通讯引用:** 181 | [OpenAlex ID](https://openalex.org/A5043976016)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了FORGE模拟器和基于物理信息的神经线性UCB分配器，用于可持续众包任务分配。

**💡 创新点**

创新点在于将承包商的自我调节加入多臂拉曼框架，利用物理信息协方差先验热启动，并将两塔结构与UCB探索结合，实现对疲劳状态的感知与动态分配。

**🔧 技术方法**

使用技术包括两塔神经网络、Neural-Linear UCB、物理信息协方差先验、TOPSIS融合以及在线更新与自适应学习。

**📊 数据集**

使用的数据集为基于COALESCE的自生成仿真数据FORGE，包含100名承包商的句子嵌入、连续疲劳轨迹、价格弹性等特征。

**📈 对比分析**

与Greedy、TOPSIS、LinUCB、SW-UCB、Thompson Sampling等基线对比，在200轮冷启动实验中Hybrid+Prior获得最高奖励0.555、最低早期回报26.40、仅7.6%利用率，并在噪声与离职率下保持鲁棒性能。

**⚠️ 局限性**

局限在于物理先验在高离职率或极端需求激增时的适应性有限，且未实现自适应先验衰减，未来需研究动态权重与更复杂策略模型。

---

## 541. Phys-3D: Physics-Constrained Real-Time Crowd Tracking and Counting on Railway Platforms

**arXiv ID:** 2602.23177 | [PDF](https://arxiv.org/pdf/2602.23177v1)

**作者:** Bin Zeng `[一作]` (Humboldt University Berlin), Peter Eisert `[通讯]` (Humboldt University Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用单摄像头实时追踪并计数列车接近时站台人群，构建端到端检测-跟踪-计数流水线；

**💡 创新点**

提出物理约束的三维卡尔曼模型（Phys‑3D）结合摄像机几何与列车自运动，使跟踪保持物理一致性，并设计虚拟计数带与时间持续性；

**🔧 技术方法**

使用YOLOv11m头部检测器、EfficientNet‑B0特征编码、DeepSORT跟踪框架、Phys‑3D卡尔曼滤波及虚拟计数带；

**📊 数据集**

训练集为CrowdHuman、Open Sensor Data for Rail 2023、RailEye3D、RailwayPlatformCrowdHead，评估集为自建MOT‑RailwayPlatformCrowdHead；

**📈 对比分析**

与常规2D常速/加速卡尔曼模型相比，Phys‑3D在MOTA/IDF1上分别为67.19%/76.32%，计数MAE仅0.9、RMSE1.36、MAPE2.97%，显著提升；

**⚠️ 局限性**

目前缺少夜间或恶劣天气样本，数据多样性不足；未来需拓展多模态融合和跨域适应。

---

## 542. Automated Robotic Needle Puncture for Percutaneous Dilatational Tracheostomy

**arXiv ID:** 2602.22952 | [PDF](https://arxiv.org/pdf/2602.22952v1)

**作者:** Yuan Tang `[一作]` (University of Manchester), Andrew Weightman `[通讯]` (University of Manchester)

**通讯引用:** 6297 | [OpenAlex ID](https://openalex.org/A5036667843)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出并验证了一套用于外显式气管切开术（PDT）针刺的机器人系统，采用自适应受限控制器通过电磁传感器实现精准针刺；

**💡 创新点**

创新点在于将向量场不等式（VFI）与自适应控制相结合，实现机器人对不确定运动学参数的在线自适应，并通过多重几何约束保证针刺过程的安全与精度；

**🔧 技术方法**

使用7自由度Franka Emika Panda机械臂、NDI Model 130电磁跟踪系统、CoppeliaSim仿真环境以及基于VFI的自适应受限控制算法；

**📊 数据集**

未使用公开数据集，而是在仿真中随机生成目标刺入点（10×10×10 mm立方体内）并随机患者解剖参数，在实验中对四种机位共进行400次针刺测试；

**📈 对比分析**

通过与手工PDT以及超声辅助对比，实验数据显示针尖位置误差中位数为1.7 mm（IQR 1.9 mm），中线偏差中位数为4.13°，95%成功率；相较于手工（2.9 mm）和超声（2.9 mm）显示显著提升；

**⚠️ 局限性**

受限于气管镜传感器的放置精度、实验操作员缺乏临床经验、硅胶皮肤与真实组织差异、有限的解剖变异以及缺乏力反馈等，导致针刺误差仍与传感器测量误差高度相关。

---

## 543. Sensor Generalization for Adaptive Sensing in Event-based Object Detection via Joint Distribution Training

**arXiv ID:** 2602.23357 | [PDF](https://arxiv.org/pdf/2602.23357v1)

**作者:** Aheli Saha `[一作]` (Deutsches Forschungszentrum für Künstliche Intelligenz), Didier Stricker `[通讯]` (Deutsches Forschungszentrum für Künstliche Intelligenz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了不同事件相机参数对事件驱动目标检测的影响，并提出通过多源域泛化训练实现传感器不变性的目标检测模型。

**💡 创新点**

创新点在于首次系统评估传感器参数对事件检测性能的影响，结合多源域泛化方法在多种相机配置下训练模型，提供了规模化的仿真数据集。

**🔧 技术方法**

采用事件堆栈直方图表示，使用RVT和SSMS两种事件目标检测架构，并通过联合训练与域泛化技术提升模型对传感器参数变化的鲁棒性。

**📊 数据集**

使用基于CARLA仿真的事件数据集，约15小时，分辨率720×1280，涵盖13个城镇和14种不同相机参数配置。

**📈 对比分析**

通过COCO AP指标在四种测试集（内分布、单参扰动、组合配置、未见参数）进行对比，SSMS+联合训练在所有场景均优于单参训练，平均提升约4–10%。

**⚠️ 局限性**

局限性包括对极端参数组合和异阈值（正负阈值不对称）仍存在性能下降；仿真数据与真实相机差异、模型对高频噪声的鲁棒性待进一步验证。

---

## 544. Cybersecurity Data Extraction from Common Crawl

**arXiv ID:** 2602.22218 | [PDF](https://arxiv.org/pdf/2602.22218v1)

**作者:** Ashim Mahara `[一作]` `[通讯]`, Ashim Mahara

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

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

## 545. TrajTok: Learning Trajectory Tokens enables better Video Understanding

**arXiv ID:** 2602.22779 | [PDF](https://arxiv.org/pdf/2602.22779v1)

**作者:** Chenhao Zheng `[一作]` (University of Washington), Ranjay Krishna `[通讯]` (University of Washington)

**通讯引用:** 12937 | [OpenAlex ID](https://openalex.org/A5032451496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种端到端可微的轨迹视频分词器，能够直接从原始视频像素生成基于对象轨迹的压缩令牌，并可在视频编码器、预训练特征适配器及视觉‑语言模型中复用。

**💡 创新点**

创新点包括：①统一的轻量级分段器通过可学习的查询进行隐式聚类，直接生成空间‑时间轨迹掩码；②轨迹编码器采用软/硬掩码与 Perceiver 模块聚合特征，并引入可自适应的多子令牌（Matryoshka）以平衡表达与效率；③完全端到端训练，去掉了外部非差分分割/跟踪管线，实现更快、更灵活的分词；④通过与 CLIP 对比学习共同优化，使分割粒度自适应下游任务需求。

**🔧 技术方法**

使用的技术包括 ConvNeXt 作为图像块编码器、Perceiver 结构进行查询‑特征注意、RoPE 位置编码、Dice+Focal 损失的伪分割监督、softmax/argmax 软/硬掩码、Matryoshka 多子令牌初始化、CLIP 对比损失、以及大规模数据的伪标签生成（使用 TrajViT、SAM 等外部模型）。

**📊 数据集**

主要使用的数据集有：Panda‑70M（4M 视频+15M 图像+文本对）用于 CLIP 预训练；8M 视频/15M 图像的伪轨迹标签（由 TrajViT+SAM 生成）用于分段器预训练；Kinetics‑400、Something‑Something‑V2、ImageNet‑1K、CIFAR‑100、Caltech‑101 等用于线性/注意力/Perceiver 评估；ActivityNet、VATEX、MSR‑VTT、Charades、COCO、Flickr30K 用于视频/图像检索；LongVideoBench、LVBench、短视频 QA 等用于 VLM 评估。

**📈 对比分析**

与 ViT3D、ViViT、TokenLearner、RLT、TrajViT 等基线进行对比；在 CLIP 预训练后，取得 Kinetics‑400 +4.8%、SSv2 +4.1% 的显著提升；在检索任务中比 TrajViT 提升 R@5 多达 +4%；在可扩展数据规模（1M→8M）下，表现优于 TrajViT，且推理 FLOPs 与 ViViT 相近，远低于 ViT3D 的二次复杂度；在 VLM 任务中，相较于 PatchVLM 在长视频 QA 上提升 8.8%+。

**⚠️ 局限性**

局限性包括：对单一前景图像（如 ImageNet）分割量少，导致性能略低；分段器以语义聚类为主，像素级精度不高，可能错过极小对象；在某些短视频 QA 任务中表现不如 PatchVLM；对极端遮挡或快速运动场景的轨迹连续性仍需改进。

---

## 546. VeRO: An Evaluation Harness for Agents to Optimize Agents

**arXiv ID:** 2602.22480 | [PDF](https://arxiv.org/pdf/2602.22480v1)

**作者:** Varun Ursekar `[一作]` (Scale AI), Sam Denton `[通讯]` (Scale AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于版本化、奖励与观察（VRO）的框架，用来评估和比较 LLM 编码器在 Agent‑as‑Code 优化任务中的表现；并构建了标准化的 Agent‑Optimization Benchmark，包含多种任务与目标 Agent，旨在系统化评测编程型 LLM 作为优化器的能力。

**💡 创新点**

创新点包括：①将版本控制、预算控制与结构化执行轨迹集成到一个统一的评估 harness；②将 Agent‑Optimization 明确为一个可编程的代码生成任务并发布基准套件；③通过大规模实验揭示当前优化器主要依赖 prompt 调整、缺乏结构化变更、跨任务迁移受限。

**🔧 技术方法**

技术手段涵盖：Git worktree 自动提交与回滚；工具与钩子接口实现对文件、数据集、文件系统的访问控制；预算管理（评估次数限制）；结构化执行追踪与实验数据库；LLM 编码器（Claude Sonnet 4.5、GPT‑5.2‑Codex 等）以及多种工具调用与代理框架。

**📊 数据集**

使用的数据集包括：GAIA（多步推理）、GPQA（科学 QA）、MATH（数学推理）、TAU‑Bench Retail（工具使用）、SimpleQA（事实 QA）以及 FACTS 搜索任务。

**📈 对比分析**

比较方法：在每个任务上对 5 种优化器配置（包括默认、Orchestrator、Resources‑Only 等）分别进行 3 次实验，预算 B=8 次评估，记录 baseline 与各迭代的分数。结果显示：工具使用类任务（GAIA、Retail、SimpleQA）显著提升（lift 约 10–25%），推理类任务（MATH、GPQA）提升有限或负增益；不同模型与指令模板对性能影响显著，存在显著方差。

**⚠️ 局限性**

局限性：①预算仅计评估调用次数，未考虑 token/API 成本；②未探测多阶段 vs 单阶段优化、预算规模对结果的影响；③缺乏人工基准对照；④对公共 API 稳定性与安全性的依赖可能导致实验不稳定；⑤优化器主要做 prompt 变更，缺乏结构化改动，难以获得更大提升。

---

## 547. The Trinity of Consistency as a Defining Principle for General World Models

**arXiv ID:** 2602.23152 | [PDF](https://arxiv.org/pdf/2602.23152v1)

**作者:** Jingxuan Wei `[一作]`, Cheng Tan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了世界模型的三元一致性框架（模态、一致性、时间一致性），并基于此设计了 CoW‑Bench 基准，系统回顾并比较了从专用模块到统一世界模拟器的演进路径，阐述了多模态、三维几何与因果推理在生成模型中的交互机制。

**💡 创新点**

创新点主要包括：①将模态、空间和时间三种一致性统一为“Trinity of Consistency”框架；②提出 CoW‑Bench，用多帧推理与约束满足度量评估模型在复杂场景下的物理一致性；③系统性梳理并对比了多模态对齐、隐式/显式 3D 表示以及 Diffusion‑Transformer 等前沿技术在实现三元一致性上的关键进展；④在基准实验中展示了现有模型（如 Sora、Veo 3 等）在模态一致性、几何真实性和因果连贯性上的差距。

**🔧 技术方法**

技术方法包括：多模态对齐（CLIP/BLIP‑2、Q‑Former、Perceiver Resampler 等）、隐式连续场（NeRF、Mip‑NeRF 等）与显式 Lagrangian 原语（3DGS、PhysGaussian 等）融合；基于流匹配的扩散 Transformer（DiT）实现端到端时空一致性；强化学习与人类反馈（RLHF、SPO、Visual‑CoT 等）实现意图对齐和因果推理；以及 CoW‑Bench 的多帧约束满足度量（VCD、Δconst 等）。

**📊 数据集**

数据集涵盖：大规模 3D 资产库（Objaverse‑XL、G‑Objaverse、G‑Objaverse‑L），高质量 RGB‑D‑Normal 记录（Co3D‑v2、MVImgNet），以及通过生成视频与几何重建循环得到的合成 4D 数据集（See3D、Dust3R、MVDream 等）。

**📈 对比分析**

在 CoW‑Bench 上，现有模型如 Sora、HunyuanVideo、Veo 3 在模态一致性、空间一致性和时间一致性上分别取得 0.88、0.86、0.95 的综合得分（百分制），相较传统视频生成模型提升 20–30%；在物理因果评估（Physics‑IQ）中，Veo 3 的成功率超过 70%，而单纯的 2D 融合模型则低于 30%。

**⚠️ 局限性**

限制主要体现在：① 现有模型仍以统计拟合为主，易出现结构幻觉、时间漂移和因果违背；② 3D 表示在稀疏视角下容易出现 Janus 问题；③ 强化学习与人类反馈在大规模训练成本与通用性之间存在折中；④ CoW‑Bench 尚未覆盖全部物理场景（如流体、弹性变形），未来需要更多多模态动态数据。

---

## 548. QSIM: Mitigating Overestimation in Multi-Agent Reinforcement Learning via Action Similarity Weighted Q-Learning

**arXiv ID:** 2602.22786 | [PDF](https://arxiv.org/pdf/2602.22786v1)

**作者:** Yuanjun Li `[一作]` (Shandong University), Zhiwei Xu `[通讯]` (Shandong University)

**通讯引用:** 2503 | [OpenAlex ID](https://openalex.org/A5100782604)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出 QSIM 框架，通过相似度加权的 TD 目标取代传统的 max 操作，降低多智能体价值估计的过估计偏差；

**💡 创新点**

创新点在于构造线性可扩展的近贪婪动作空间，并利用自监督动作嵌入计算动作相似度，从而在目标更新中加入结构化的期望；

**🔧 技术方法**

使用自监督自编码器学习动作嵌入、余弦相似度与 softmax 加权、近贪婪联合动作空间构造及 QSIM 加权 TD 目标；

**📊 数据集**

在 SMAC、SMACv2、MPE 和 Matrix Games 四个多智能体基准上进行实验；

**📈 对比分析**

与 VDN、QMIX、WQMIX、QPLEX、RES、RODE、RIIT 等基线对比，QSIM 在多地图任务上均提升了胜率和稳定性，且显著降低了 Q 值估计误差；

**⚠️ 局限性**

局限性包括依赖自监督嵌入的质量、近贪婪空间仍可能忽略远离贪婪策略的有用动作，以及在更大规模或完全离线场景下的适用性待验证。

---

## 549. Vibe Researching as Wolf Coming: Can AI Agents with Skills Replace or Augment Social Scientists?

**arXiv ID:** 2602.22401 | [PDF](https://arxiv.org/pdf/2602.22401v1)

**作者:** Yongjun Zhang `[一作]` (Stony Brook University), Yongjun Zhang `[通讯]` (Stony Brook University)

**通讯引用:** 9456 | [OpenAlex ID](https://openalex.org/A5108625236)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过介绍“vibe researching”概念、构建认知任务框架，并以 Claude Code 上的 scholar‑skill 21 个插件为案例，阐述 AI 代理在社会科学研究全流程中的作用与边界；

**💡 创新点**

创新点在于提出了以可编码性与隐性知识需求两维划分研究任务、从认知角度而非流程顺序界定 AI 委托边界，并将这一框架与实际插件操作相结合，提供了可操作的责任原则；

**🔧 技术方法**

采用了大型语言模型（Claude）、多步骤推理代理、插件化架构、自动化工具调用与质量门控等技术；

**📊 数据集**

主要使用研究者自身的 Zotero 库（2万+ 条文献）、IPUMS、NHANES 等公开数据库，以及 127 篇论文构建的知识图谱作为插件训练与查询数据；

**📈 对比分析**

论文未进行实证对比实验，而是以案例演示说明 AI 在速度与覆盖率方面相较手工操作的优势，但在理论原创性、隐性领域知识判断等方面仍有限；

**⚠️ 局限性**

局限性包括仅聚焦单一系统（scholar‑skill）且缺乏用户研究与实验验证；任务分类为离散化简化，未对跨维度复杂任务进行细致评估；

---

## 550. Autoregressive Visual Decoding from EEG Signals

**arXiv ID:** 2602.22555 | [PDF](https://arxiv.org/pdf/2602.22555v1)

**作者:** Sicheng Dai `[一作]` (Institute of Automation), Qiwei Ye `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 9831 | [OpenAlex ID](https://openalex.org/A5068656698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级的自回归视觉解码框架，利用EEG信号逐步生成对应图像。

**💡 创新点**

创新点在于将预训练的LaBraM EEG编码器通过对比学习微调，并采用多尺度token化与自回归“next‑scale”预测策略，取代传统的多阶段扩散模型。

**🔧 技术方法**

技术手段包括LaBraM EEG编码器、对比学习、CLIP图像编码器、VQ‑VAE多尺度离散化、Transformer自回归生成以及无分类器指导（CFG）。

**📊 数据集**

实验数据集为THINGS‑EEG（10位受试者，1万多条EEG样本）和EEG‑ImageNet（更大规模）。

**📈 对比分析**

与EEGNetV4、EEGConformer、NICE、ATM、unCLIP+Diffusion等基线对比，检索任务中取得30% Top‑1/58% Top‑5的性能，重建任务在PixCorr、SSIM、CLIP等指标上均优于现有方法，并将模型参数量缩减至原始的10%。

**⚠️ 局限性**

局限性包括仍受EEG信号噪声和受试者差异影响，实验仅在两个数据集上验证，尚未实现完全实时部署。

---

## 551. WaterVideoQA: ASV-Centric Perception and Rule-Compliant Reasoning via Multi-Modal Agents

**arXiv ID:** 2602.22923 | [PDF](https://arxiv.org/pdf/2602.22923v1)

**作者:** Runwei Guan `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 44252 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出了首个全水道视频问答基准 WaterVideoQA，并设计了多模态神经符号推理系统 NaviMind，实现 ASV 从被动感知到主动认知。

**💡 创新点**

创新点包括自适应语义路由、情境感知层级推理以及自我反思验证三大机制，使模型在规则合规和可解释性上显著提升。

**🔧 技术方法**

采用多模态检索增强生成、LLM 代理协作、视觉编码器与规则知识库相结合的多代理体系。

**📊 数据集**

采用自制的 WaterVideoQA 数据集，共 3029 条视频和 3673 组 QA，覆盖河流、湖泊、运河、护城河、港口与海域。

**📈 对比分析**

与现有 MLLM、VideoAgent、OmAgent 等基线对比，NaviMind 在 GPT‑Score、CIDEr 等指标上领先数个百分点，且推理时延更短。

**⚠️ 局限性**

限制在于仅依赖可见光视频，缺乏多传感融合；缺乏细粒度空间定位；多船舶交互的博弈推理仍不成熟。

---

## 552. X-REFINE: XAI-based RElevance input-Filtering and archItecture fiNe-tuning for channel Estimation

**arXiv ID:** 2602.22277 | [PDF](https://arxiv.org/pdf/2602.22277v1)

**作者:** Abdul Karim Gizzini `[一作]` (University of Paris-Est Créteil), Yahia Medjahdi `[通讯]` (IMT Nord Europe)

**通讯引用:** 817 | [OpenAlex ID](https://openalex.org/A5045307244)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 X-REFINE 框架，通过 XAI 的 LRP-ε 规则实现对 OFDM 信道估计中 FNN 模型的输入过滤与架构微调，实现模型压缩与性能保持。

**💡 创新点**

创新点在于将基于 XAI 的输入重要性评估与内部网络结构剪枝双重优化相结合，使用符号稳定的 LRP-ε 规则得到高分辨率相关性分数，并实现两级稀疏化，显著提升解释性‑性能‑复杂度三维权衡。

**🔧 技术方法**

主要技术包括：符号稳定的 Layer-wise Relevance Propagation（LRP‑ε）、FNN 信道估计、OFDM 系统建模、基于阈值的二进制掩码搜索、FLOPs 计算与复杂度分析。

**📊 数据集**

使用 IEEE 802.11p 规范下的 OFDM 信号模拟数据，包含低频选择性（LF）与高频选择性（HF）两种环境，QPSK 与 64QAM 两种调制，训练集 10⁵ 个 OFDM 符号，测试集 20% 的剩余符号。

**📈 对比分析**

通过与完整 FNN 模型（基线）和先前的 perturbation‑based XAI‑CHEST 框架对比，评估指标为比特错误率（BER）和 FLOPs 复杂度。实验显示 X-REFINE 在保持 BER 与基线相近或略优的同时，整体复杂度下降约 35‑43%（相较于 XAI‑CHEST 的 24‑32%），体现了更优的性能‑复杂度‑解释性权衡。

**⚠️ 局限性**

局限性包括：仅在仿真 OFDM 环境下验证，缺乏真实通道实验；仅针对前馈神经网络，未扩展到循环网络或其他模型；对 LRP‑ε 规则的假设和参数设置敏感；阈值搜索仍为离散网格搜索，可能缺乏全局最优保证。

---

## 553. ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Control and Mechanism-Linked Consciousness Indicator Assays

**arXiv ID:** 2602.23232 | [PDF](https://arxiv.org/pdf/2602.23232v1)

**作者:** Aishik Sanyal `[一作]` `[通讯]` (Independent Research Engineer), Aishik Sanyal (Independent Research Engineer)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了一个可检查的 ReCoN‑Ipsundrum 代理，在原始 ReCoN 结构上加入了 ipsundrum 循环与可选的情感代理，利用一系列基于指标的实验（路线偏好、探索游戏、疼痛尾部、以及在期中切除循环的因果实验）检验其机制效应。

**💡 创新点**

创新点在于：①将 Humphrey 的 ipsundrum 设定以可插拔方式嵌入 ReCoN，形成可检验的递归感知循环；②提供了情感代理与递归循环的耦合机制；③在同一框架下同时设计多种指标测评与因果切除实验，展示了不同机制与行为指标之间的解耦与关联。

**🔧 技术方法**

使用的技术包括：ReCoN 的消息传递状态机、单步前向模型与短期规划、ipsundrum 循环更新（含可调衰减与反馈参数）、情感代理（预算、价值、唤醒信号）以及基于 Python 的实验脚本与可视化工具。

**📊 数据集**

主要使用了自定义的迷宫与走廊任务：CorridorWorld 与 GridWorld（含障碍、危险、景观与无景观路线）作为测试环境；无外部公开数据集，所有环境均为内部生成的 toy 环境。

**📈 对比分析**

比较方法：在同一任务下比较三种模型（ReCoN、Ipsundrum、Ipsundrum+affect），通过多重指标（危害次数、目标时间、成功率、景观进入率、扫描事件、疼痛尾部持续时间、后刺激 N^s 的 AUC 以及因果切除后的 AUC 下降）进行对比。结果显示：Ipsundrum+affect 在安全性（危害减少）与稳健的景观偏好、探索扫描以及疼痛尾部的谨慎持续性上表现最佳；单纯的递归循环只能提升后刺激感知的持久性，情感耦合是实现稳健偏好与探索行为的关键。

**⚠️ 局限性**

局限性包括：①模型为无学习、极简的 toy 设计，缺乏对更复杂、真实任务的验证；②情感代理仅为简化的预算模型，未覆盖完整的生理或主观体验；③未涉及更完整的预测加工或信息整合理论，无法直接评估意识本身；④实验结果易被游戏化，单一指标不具备普适性；⑤缺乏对参数空间的系统性敏感性分析。

---

## 554. Towards Better RL Training Data Utilization via Second-Order Rollout

**arXiv ID:** 2602.22765 | [PDF](https://arxiv.org/pdf/2602.22765v1)

**作者:** Zhe Yang `[一作]` (ByteDance BandAI), Zhifang Sui `[通讯]` (State Key Laboratory of Multimedia Information Processing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 GC‑RL 框架，在强化学习训练中加入“第二阶滚动”来同时训练模型的生成与批评能力。

**💡 创新点**

创新点在于引入第二阶滚动生成批评、联合训练生成与批评、以及利用数据过滤和奖励去噪来提升训练数据利用率。

**🔧 技术方法**

使用 GRPO 算法、策略网络、数据过滤器、奖励重加权、以及自校正采样等技术。

**📊 数据集**

主要训练数据为数学推理数据集 DAPO‑MATH‑17k，评测使用 Math‑500、GSM8k、Minerva、AMC23、OlympiadBench。

**📈 对比分析**

与普通生成 RL（G‑RL）、纯批评 RL（C‑RL）以及无 RL 预训练基线比较，GC‑RL 在生成与批评准确率上均优于其它方法；以 Qwen2.5‑7B 为例，生成准确率从 75.4% 提升至 77.6%，批评准确率从 80.5% 提升至 84.6%。

**⚠️ 局限性**

主要限制包括收敛速度较慢、仅在 GRPO 上验证、仅适用于可规则验证的生成任务、以及对大于 10B 模型或多域数据的适用性尚未验证。

---

## 555. Deep Sequence Modeling with Quantum Dynamics: Language as a Wave Function

**arXiv ID:** 2602.22255 | [PDF](https://arxiv.org/pdf/2602.22255v1)

**作者:** Ahmed Nebli `[一作]` (cAI Technology GmbH), Kevin Yam `[通讯]` (cAI Technology GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于复杂数波函数、可学习时间变哈密顿量、Cayley（Crank–Nicolson）离散化和 Born 规则读取的序列模型，用相位干涉机制实现对歧义上下文的自适应抑制与强化。

**💡 创新点**

创新点包括：1）将复数状态空间与量子哈密顿动力学结合到序列建模；2）使用 Cayley 离散化实现任意步长下的严格保范数；3）利用 Born 规则的二次读取，使模型可访问 $O(N^2)$ 的相位交叉项；4）提出并证明了在一族歧义任务上，复杂单位化模型只需维度 $N$ 即可完美实现，而实值正交模型需至少 $Ω(N^2)$ 的隐藏维度。

**🔧 技术方法**

技术手段包括：复杂数向量表示、Hermitian 哈密顿量设计、相互作用框架与输入相关动态、Cayley（Crank–Nicolson）离散化、Born 规则概率读取、低秩因子化的交互哈密顿、QR 投影保证测量矩阵正交、梯度通过隐式线性求解与反向传播。

**📊 数据集**

论文中未给出具体实验数据集，主要侧重理论构建与证明。

**📈 对比分析**

比较方法主要是理论分离定理与维度优势分析；未提供数值实验或与 Transformer/ uRNN 等基线模型的性能对比。

**⚠️ 局限性**

局限性包括：① 训练过程仍需处理 g_θ 的梯度稳定性，② 相位干涉效果高度依赖模型参数，可在实际大规模语言建模中的效果尚未验证；③ 低秩参数限制了可表达性；④ Cayley 离散化虽保证保范数，但离散误差随步长与交互哈密顿大小变化，需进一步分析与调优。

---

## 556. Space Syntax-guided Post-training for Residential Floor Plan Generation

**arXiv ID:** 2602.22507 | [PDF](https://arxiv.org/pdf/2602.22507v1)

**作者:** Zhuoyang Jiang `[一作]` (Information Hub, Hong Kong University of Science and Technology), Dongqing Zhang `[通讯]` (Tongji University)

**通讯引用:** 7354 | [OpenAlex ID](https://openalex.org/A5101534655)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于空间句法的后训练框架（SSPT），用于提升预训练住宅平面图生成模型的公共空间主导性和空间层级。

**💡 创新点**

创新点包括：① 引入非可微的空间句法 oracle，将空间集成度作为可评估、可奖励的指标；② 提出了统一的 OOD 评测基准 SSPT‑Bench（Eval‑8）；③ 对比两种后训练策略——迭代过滤微调（SSPT‑Iter）与 PPO 强化学习（SSPT‑PPO），验证其在提升空间逻辑上的有效性。

**🔧 技术方法**

技术手段：预训练的条件扩散模型；矩形分解与图构建生成空间句法图；空间集成度（integration）计算；迭代生成–筛选–再训练流程；PPO 强化学习与基于终端奖励的策略优化；统一评价指标与统计分析。

**📊 数据集**

数据集：RPLAN（约 80k 条真实住宅平面图）用于训练、筛选与基准；LIFULL（约 15k 条日本住宅平面图）用于交叉验证；HouseDiffusion 作为预训练模型基线；SSPT‑Bench 的 Eval‑8 由 8 室程序构成。

**📈 对比分析**

比较方法：在 SSPT‑Bench（Eval‑8）下使用统一的空间句法指标（公共空间主导性、居室相对集成度、居室优势、profile 距离）对齐模型与真实数据。实验显示，PPO 后训练在同等计算时间内提升了 0.0424 的公共空间主导性、0.0374 的居室优势，并将指标方差降低约 20‑30%，同时相较于迭代微调提升约 11 倍的计算效率。

**⚠️ 局限性**

局限性：① 只关注公共空间主导性，未覆盖其他空间句法度量（如选择性、通行效率等）；② Oracle 计算仍相对昂贵，限制了大规模后训练；③ 受预训练分布约束，极端或非常规布局仍难以生成；④ 需要手工设定阈值与规则，缺乏自适应机制。

---

## 557. Calibrated Test-Time Guidance for Bayesian Inference

**arXiv ID:** 2602.22428 | [PDF](https://arxiv.org/pdf/2602.22428v1)

**作者:** Daniel Geyfman `[一作]` (University of California), Stephan Mandt `[通讯]` (University of California)

**通讯引用:** 5315 | [OpenAlex ID](https://openalex.org/A5091841504)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一种新的测试时引导框架，旨在从真实的贝叶斯后验中进行一致的采样，解决了现有方法在测试时引导中的偏差问题。

**💡 创新点**

创新点在于提出了校准贝叶斯引导（CBG），该框架能够在给定和不给定温度的情况下，从真实的贝叶斯后验中进行一致的采样。

**🔧 技术方法**

使用了基于梯度的和无梯度的校准贝叶斯引导技术，结合了重参数化技巧和REINFORCE估计器。

**📊 数据集**

在多个贝叶斯推断任务上进行了实验，包括黑洞成像任务，使用了预训练的扩散模型作为先验分布。

**📈 对比分析**

与其他测试时引导方法相比，CBG在多个贝叶斯推断任务上表现优越，尤其是在黑洞成像任务中达到了最先进的PSNR性能。

**⚠️ 局限性**

限制在于，所提出的估计器需要从p(x | x_t)中获取样本，这可能会很昂贵。随着少步扩散模型的进展，这一问题可能会通过预训练模型来解决。

---

## 558. Poisoned Acoustics

**arXiv ID:** 2602.22258 | [PDF](https://arxiv.org/pdf/2602.22258v1)

**作者:** Harrison Dahme `[一作]` `[通讯]`, Harrison Dahme

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了针对声学车辆分类器的训练数据中毒攻击，展示在仅更改不到 0.5% 训练标签的情况下即可实现高达 95.7% 的目标类误识率（Truck → Car），并且攻击对整体准确率几乎无影响。作者进一步分析了攻击表面，证明攻击隐蔽性由少数类比例决定，并发现后门触发在少数类下会退化为标签翻转攻击；随后提出了基于 Merkle 树承诺和后量子签名的可验证训练管道架构来消除对未签名流水线阶段的信任。

**💡 创新点**

创新点包括：
1) 用解析方法证明聚合准确率监控对少数类攻击不可检验，误差上限为 β（少数类比例）。
2) 发现后门触发在少数类训练样本稀缺时会退化为纯标签翻转攻击，首次揭示此类触发-崩塌现象。
3) 设计了一套完整的可验证训练管道：内容地址化散列、Merkle 树数据集承诺、以及 NIST FIPS 204 级后量子签名（ML‑DSA‑65/CRYSTALS‑Dilithium3）。

**🔧 技术方法**

使用技术包括：
- 目标标签翻转攻击与 12×12 像素白色补丁的后门触发。
- 128-bin log‑mel 频谱作为输入，训练四层 Conv‑BN‑ReLU‑Pool 的 2‑D CNN。
- 统计与理论分析（ASR 计算、95% CI、ΔAcc≤β 证明）。
- Merkle 树承诺与后量子数字签名用于管道完整性验证。
- 评估工具：Per‑class F1、混淆矩阵、特征分布异常检测等。

**📊 数据集**

使用 MELAUDIS 城市交叉口音频数据集，约 9,600 条单车音频，分布 6 类（Car 84.4%，Tram 6.3%，Truck 2.7%，Bus 2.7%，Motorcycle 2.6%，Bicycle 2.3%）。

**📈 对比分析**

实验与干净模型对比：
- 在 0.5% 标签翻转率下，ASR 达到 95.7%（95% CI 88–100%），整体准确率仍为 87.6% 与干净模型相同。
- 1% 与 2% 翻转率保持相近的高 ASR，并略微降低整体准确率但不显著。
- 后门触发实验显示 Clean ASR 与 Triggered ASR 相等，证明触发无效。
- 防御评估表明：聚合准确率监控无法检测攻击，需依赖 per‑class 指标、签名验证等措施。

**⚠️ 局限性**

局限性包括：
1) 仅在单一小型 2‑D CNN 上测试，模型规模或正则化不同可能影响 ASR 与检测难度。
2) Truck 类样本仅约 182 条，攻击效果可能受样本稀缺性影响，无法推广至更大类样本。
3) 后门触发采用可见 12×12 像素补丁，隐形或频域触发的行为可能不同，实验结果不一定适用于所有后门形式。

---

## 559. WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval

**arXiv ID:** 2602.23029 | [PDF](https://arxiv.org/pdf/2602.23029v1)

**作者:** Tianyue Wang `[一作]` (University of Chinese Academy of Sciences), Jinqiao Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 7660 | [OpenAlex ID](https://openalex.org/A5058420913)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 WISER 的训练免费零样本组合图像检索框架，通过并行使用文本-图像检索（T2I）与图像-图像检索（I2I），并通过检索-验证-细化循环提升检索质量。

**💡 创新点**

核心创新包括：①宽泛检索（Wider Search）同时激活 T2I 与 I2I 以扩大候选池；②自适应融合（Adaptive Fusion）引入验证器评估置信度，动态权衡两条路径；③深度思考（Deeper Thinking）利用 LLM 进行结构化自我反思，生成细化建议并迭代改进。

**🔧 技术方法**

技术组成包括：开源编辑器 BAGEL、CLIP（ViT-B/32、ViT-L/14、ViT-G/14）检索器、Qwen2.5‑VL‑7B 验证器、GPT‑4o 细化器，以及 BLIP‑2 生成描述器。

**📊 数据集**

在三个公开基准上评测：Fashion‑IQ、CIRR 和 CIRCO，涵盖时尚与自然图像的多目标检索任务。

**📈 对比分析**

与现有训练免费与训练依赖方法对比，WISER 在 CIRCO mAP@5 提升约 45%、在 CIRR Recall@1 提升约 57%，在 Fashion‑IQ Recall@10/50 均达到或超过训练方法的最佳结果，显示显著性能优势。

**⚠️ 局限性**

局限性包括：①细化过程依赖 LLM 计算资源，导致一次检索延迟增加；②对阈值 τ 与迭代次数 N 的设置敏感，需在效率与效果之间权衡；③对极度模糊或多义查询的自适应能力仍有限。

---

## 560. RAGdb: A Zero-Dependency, Embeddable Architecture for Multimodal Retrieval-Augmented Generation on the Edge

**arXiv ID:** 2602.22217 | [PDF](https://arxiv.org/pdf/2602.22217v1)

**作者:** Ahmed Bin Khalid `[一作]` `[通讯]` (SKAS Information Technology), Ahmed Bin Khalid (SKAS Information Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了RAGdb，一种单文件、无服务器的检索增强生成引擎。

**💡 创新点**

创新点在于将自动多模态摄取、ONNX推理和混合向量检索统一到单一SQLite容器，并引入Deterministic Hybrid Scoring Function实现无GPU推理。

**🔧 技术方法**

采用SQLite存储、NumPy实现稀疏TF‑IDF向量化、ONNXRuntime OCR、基于哈希的增量摄取、Hybrid Scoring Function（TF‑IDF+子串提升）。

**📊 数据集**

实验使用合成1,000文档的混合业务技术语料库，并注入唯一实体码。

**📈 对比分析**

与Docker+ChromaDB+LangChain栈对比，RAGdb实现了31.6×增量摄取速度提升、99.5%磁盘占用降低、查询延迟约60 ms，Recall@1对实体查询达到100%。

**⚠️ 局限性**

局限在于仍使用稀疏检索，可能对深度语义匹配效果不及纯密集检索；且目前仅支持英文文本，跨语言性能待验证。

---

## 561. Mitigating Legibility Tax with Decoupled Prover-Verifier Games

**arXiv ID:** 2602.23248 | [PDF](https://arxiv.org/pdf/2602.23248v1)

**作者:** Yegon Kim `[一作]` (KAIST), Juho Lee `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种解耦证明者-验证者游戏（DPVG），通过训练翻译模型将固定求解器的答案转换为可验证形式，保持高准确率的同时提升可检查性；

**💡 创新点**

核心创新是将正确性与可检查性解耦，设计了可信与狡猾翻译器的对抗框架，并证明其均衡点对应于可信且可检查的翻译器；

**🔧 技术方法**

使用对抗式强化学习（REINFORCE）结合验证器的交叉熵训练，对翻译器的奖励设计采用规范化的验证器分数；

**📊 数据集**

在GSM8K-Aug数据集上进行实验，使用16K题目分两半训练求解器/翻译器与验证器；

**📈 对比分析**

与传统PVG模型比较，DPVG在测试集上保持57%准确率（与求解器相近），显著优于PVG的22.3%，并且在训练过程中实现了高达99.8%的翻译器可信度；

**⚠️ 局限性**

局限性包括缺乏人类可读性评估、奖励函数和超参数的敏感性、以及未在链式推理与证明生成之间进行充分集成。

---

## 562. VRSL:Exploring the Comprehensibility of 360-Degree Camera Feeds for Sign Language Communication in Virtual Reality

**arXiv ID:** 2602.23265 | [PDF](https://arxiv.org/pdf/2602.23265v1)

**作者:** Gauri Umesh Rajmane `[一作]` (Rochester Institute of Technology), Roshan Peiris `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1656 | [OpenAlex ID](https://openalex.org/A5000376873)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究通过在VR环境中使用头、肩、胸三种位置的360度体贴摄像机，测试了视频流对美国手语（ASL）可理解性的影响；

**💡 创新点**

创新点在于首次将全景摄像机录制的手语视频应用于VR通信，并系统比较了不同挂载位置对识别准确率与用户体验的影响；

**🔧 技术方法**

使用的技术包括Ricoh Theta V全景摄像机、Adobe After Effects的等距化处理、Unity 3D开发的VR应用、Meta Quest头显、NASA TLX问卷评估主观负荷；

**📊 数据集**

使用的数据集为10名DHH参与者观看的30个手语视频（10个最小词对、10个非最小词对、10个句子），每个视频对应三种摄像机位置；

**📈 对比分析**

采用重复测量ANOVA与NASA TLX比较三种摄像机位置的识别准确率与主观负荷；整体准确率为83.3%，肩部摄像机最高达85%，但差异未达统计显著性；

**⚠️ 局限性**

局限包括：全景摄像机导致的畸变影响手语细节，三种位置均非自然观看角度，未实现实时交互，样本量小且单一机构内招募，需进一步扩展和改进摄像机布局与视频处理技术。

---

## 563. WaveSSM: Multiscale State-Space Models for Non-stationary Signal Attention

**arXiv ID:** 2602.22266 | [PDF](https://arxiv.org/pdf/2602.22266v1)

**作者:** Ruben Solozabal `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Martin Takac `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文设计了一种基于波形框架的多尺度状态空间模型 WaveSSM，能够在时间上实现可寻址的局部状态表示；

**💡 创新点**

其创新点在于用局部化的波形基替代传统全局多项式基，赋予模型多尺度、时局部特征并解决非重叠窗口无法分离的问题；

**🔧 技术方法**

技术上采用 SaFARi 生成波形状态空间动力学，并结合 S4 的 DPLR 参数化实现高效计算，同时通过帧紧化提升数值稳定性；

**📊 数据集**

实验使用 PTB‑XL ECG、Speech Commands SC35、Informer 时序预测和 Long Range Arena 等多种数据集；

**📈 对比分析**

与同阶 S4、MS‑SSM 等基线对比，WaveSSM 在 ECG 诊断、原声语音识别、长序列预测和 LRA 任务上均取得更高 AUROC/准确率或更低 MSE，尤其在短暂转移和局部事件任务中表现突出；

**⚠️ 局限性**

主要局限在极长上下文场景（如 LRA 的 Pathfinder 等）下扩展受限，波形框架的局部性使得模型难以完整捕捉非常长的全局依赖。

---

## 564. General Agent Evaluation

**arXiv ID:** 2602.22953 | [PDF](https://arxiv.org/pdf/2602.22953v1)

**作者:** Elron Bandel `[一作]` (IBM Research), Michal Shmueli-Scheuer `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的 Agent‑Benchmark 媒介协议和评估框架，用于在不需要领域特定适配的情况下系统评估通用 AI 代理。

**💡 创新点**

核心创新在于将代理与基准解耦，构建可扩展的 Mediation Protocol 与 Exgentic 评估框架，实现跨域、跨模型的公平比较。

**🔧 技术方法**

采用 Python、OpenAI Tooling、ReAct、CodeAgent 等技术实现代理适配，使用标准化的交互协议与并行执行架构。

**📊 数据集**

在六个多样化基准（SWE‑Bench、Customer‑Service、Code‑Generation、Web‑Navigation 等）共计 90 个配置、100 任务进行实验。

**📈 对比分析**

通过成功率、成本、步骤数等统一指标对比，发现通用代理在多数任务上可与专用代理相当，且模型质量是性能的主要决定因素。

**⚠️ 局限性**

局限性包括对模型的强依赖、不同代理在工具丰富环境下的性能差异、成本与性能的权衡以及对多模态与安全性场景的覆盖不足。

---

## 565. Distributed LLM Pretraining During Renewable Curtailment Windows: A Feasibility Study

**arXiv ID:** 2602.22760 | [PDF](https://arxiv.org/pdf/2602.22760v1)

**作者:** Philipp Wiesner `[一作]` (Exalsius), Odej Kao `[通讯]` (TU Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一种基于可再生能源削减窗口的弹性分布式 LLM 预训练系统，能在不同地区间动态开启、关闭 GPU 集群并同步模型。

**💡 创新点**

创新点包括：① 对削减信号进行时间退避和弹性加入/退出的实时调度；② 在单站和联邦模式下自动切换，并使用工作量加权 FedAvg 处理动态参与和异构吞吐；③ 通过时间窗口而非步数定义同步周期，降低 straggler 影响。

**🔧 技术方法**

主要技术：Flower 联邦学习框架、Exalsius 实时控制平面、Kubernetes、gRPC、Redis pub/sub、NVIDIA A100 GPU 集群、DDP、加权 FedAvg。

**📊 数据集**

使用 nanochat d20（20 层、561M 参数）模型训练 12.8B tokens 的 FineWebEdu‑100B 数据集，数据已预分词并划分为固定大小的 shard。

**📈 对比分析**

与基线比较：单站集中式训练 17.8h、EM‑smooth perplexity 14.7；持续 2‑站联邦训练 11.1h、perplexity 15.2；本工作在削减窗口下完成 14.6h、perplexity 15.1，能耗 37.7kWh，碳排放仅 1.38kgCO₂，显著低于单站 11.4–27.1kgCO₂。

**⚠️ 局限性**

局限性：① 短期削减窗口可能被节点开启/关闭、同步开销占比高；② 扩展到更大模型或更多站点会带来更高通信/恢复成本；③ 依赖特定的削减阈值和信号，缺乏普适的碳计量标准；④ 仅在削减频繁地区有效，稀疏地区收益有限。

---

## 566. Towards Dynamic Dense Retrieval with Routing Strategy

**arXiv ID:** 2602.22547 | [PDF](https://arxiv.org/pdf/2602.22547v1)

**作者:** Zhan Su `[一作]` (Université de Montréal), Jian-Yun Nie `[通讯]` (Université de Montréal)

**通讯引用:** 15224 | [OpenAlex ID](https://openalex.org/A5018977183)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种动态密集检索框架 DDR，利用 prefix tuning 与动态路由实现快速域适应与参数高效。

**💡 创新点**

创新点在于把 dense 检索模型拆分为通用与域专用的 prefix 模块，并通过路由函数动态激活，实现仅 2% 可训练参数即可超越传统 dense 检索。

**🔧 技术方法**

采用了 prefix tuning、soft/top‑k/先验路由、双编码器（query 与 passage）以及对比学习目标进行训练。

**📊 数据集**

训练使用 BERRI 任务指令数据集，评估采用 BEIR 零-shot 检索基准（排除与训练集重叠的数据）。

**📈 对比分析**

与 Sparse（DeepCT、SPARTA）和 Dense（Contriever、ANCE、coCondenser）基线对比，DDR‑prior 在 BEIR 六个任务上平均 NDCG@10 最高，参数量仅 2.3M。

**⚠️ 局限性**

局限性在于仍需人工标注域标签以支持 prior 路由，对更大规模 LLM 迁移性验证不足，且依赖预训练基线。

---

## 567. Exploring Human Behavior During Abstract Rule Inference and Problem Solving with the Cognitive Abstraction and Reasoning Corpus

**arXiv ID:** 2602.22408 | [PDF](https://arxiv.org/pdf/2602.22408v1)

**作者:** Caroline Ahn `[一作]` (Boston University), Chantal E. Stern `[通讯]` (Boston University)

**通讯引用:** 9754 | [OpenAlex ID](https://openalex.org/A5090312031)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文开发并使用了 CogARC，基于 ARC 的 75 个视觉抽象推理问题，在 260 名受试者上记录高时钟分辨率的行为轨迹，分析规则学习、决策与错误结构；

**💡 创新点**

创新点在于构建了大规模人类行为数据集并对解题轨迹、共同错误进行系统性定量分析，为研究人类抽象推理的认知机制与 AI 对比提供新工具；

**🔧 技术方法**

采用 Web 交互界面记录动作时间戳，使用 Jaccard 相似度评估编辑轨迹相似性、Levenshtein 编辑距离衡量解题进展，并用线性回归与方差分析探究时间与准确率的关系；

**📊 数据集**

数据集为公开的 CogARC，包括 75 个 ARC 训练集问题、260 名受试者的完整编辑日志、常见解答与任务元信息；

**📈 对比分析**

与之前的 H-ARC 等研究相比，CogARC 维持了约 80% 的整体准确率，同时提供了更丰富的行为指标；表现显示规则复杂度越高，平均难度分数、思考时间和轨迹多样性均显著增加；

**⚠️ 局限性**

局限性包括样本集中仅为 18-35 岁成人、受试者可能对任务结构产生熟练但不一定提升推理能力、缺乏神经影像或跨文化验证，且 75 个问题仍不足以覆盖 ARC 所有规则维度。

---

## 568. Integrating Machine Learning Ensembles and Large Language Models for Heart Disease Prediction Using Voting Fusion

**arXiv ID:** 2602.22280 | [PDF](https://arxiv.org/pdf/2602.22280v1)

**作者:** Md. Tahsin Amin `[一作]` (Patuakhali Science and Technology University), Nahiyan Bin Noor `[通讯]` (University of Arkansas for Medical Sciences)

**通讯引用:** 81 | [OpenAlex ID](https://openalex.org/A5088077295)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

结合机器学习集成模型与大型语言模型，评估心脏病预测性能并提出混合投票融合框架。

**💡 创新点**

在传统树基集成模型基础上引入LLM语义推理，并通过Gemini 2.5 Flash实现多模型融合，提升准确率。

**🔧 技术方法**

使用随机森林、XGBoost、LightGBM、CatBoost等集成学习；LLM（Qwen、LLaMA、Gemini等）的零样本/少样本推理；软硬投票与加权投票融合。

**📊 数据集**

公开的心脏病数据集1190例（Cleveland、Hungarian、Switzerland、Long Beach VA、Stalog Heart）共11个特征。

**📈 对比分析**

对单模型、集成模型、LLM、LLM投票及混合ML-LLM进行准确率、ROC‑AUC比较，最佳混合模型达到96.62%准确率、0.97 AUC，显著优于单一方法。

**⚠️ 局限性**

样本量有限，未做校准与成本敏感分析，LLM在原始表格数据表现不佳，缺乏外部验证。

---

## 569. Reinforcement-aware Knowledge Distillation for LLM Reasoning

**arXiv ID:** 2602.22495 | [PDF](https://arxiv.org/pdf/2602.22495v1)

**作者:** Zhaoyang Zhang `[一作]` (AWS Agentic AI), Stefano Soatto `[通讯]` (AWS Agentic AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合强化学习与知识蒸馏的框架RLAD，用以在LLM推理任务中进行后训练；

**💡 创新点**

核心创新是Trust‑Region Ratio Distillation（TRRD），将教师策略嵌入优势加权的比例更新中，避免传统KL正则导致的分布不匹配与目标冲突；

**🔧 技术方法**

采用PPO/GRPO风格的稀释重要性比率、可调混合系数α、优势归一化等RL技术，并对教师和旧学生策略构成混合锚点；

**📊 数据集**

在逻辑推理（K&K Logistics）和长上下文数学推理（AIME24/25、AMC23/24、MATH500等）数据集上进行实验；

**📈 对比分析**

与GRPO、KDRL、离线SFT等基线比较，RLAD在多规模模型（0.5B–7B）和不同上下文长度下均取得更高的准确率、Pass@1/Pass@32以及更快的收敛；

**⚠️ 局限性**

局限在于需要教师模型的logit信息，适用于开放权重且校准良好的教师；对闭源教师的适配仍是未来工作方向。

---

## 570. FuturePrism: Supporting Adolescence in Collaborative Storytelling to Cope with Future Uncertainty

**arXiv ID:** 2602.23108 | [PDF](https://arxiv.org/pdf/2602.23108v1)

**作者:** Yonglin Chen `[一作]` (Southern University of Science and Technology), Xueliang Li `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5100365967)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了基于GenAI的协作叙事系统FuturePrism，帮助13-14岁青少年通过角色扮演提高对未来的希望感。

**💡 创新点**

创新点在于将Snyder的希望理论具体化为四章节、三角色的叙事框架，让GenAI充当协作搭档而非单独讲故事者，增强了主动性。

**🔧 技术方法**

使用DeepSeek V3.1生成文本、Seedream 4.0生成图像，构建多代理架构实现问答、写作、绘图三步生成流程。

**📊 数据集**

通过20名中学生的自评数据（Children's Hope Scale、Narrative Transportation、UMUX-Lite）进行前后对比实验。

**📈 对比分析**

采用配对t检验，结果显示总希望分数和Agency显著提升（p<0.05），路径感提升边际显著，系统可用性与沉浸感均高。

**⚠️ 局限性**

样本量有限、实验仅单次、文化背景单一，且路径构建的提升不足，需要长期跟踪验证。

---

## 571. GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation

**arXiv ID:** 2602.22800 | [PDF](https://arxiv.org/pdf/2602.22800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 572. Analysis of LLMs Against Prompt Injection and Jailbreak Attacks

**arXiv ID:** 2602.22242 | [PDF](https://arxiv.org/pdf/2602.22242v1)

**作者:** Piyush Jaiswal `[一作]` (National Institute of Technology Trichy), Somanath Tripathy `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 1093 | [OpenAlex ID](https://openalex.org/A5071224481)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估多款开源 LLM 在提示注入与越狱攻击下的易受性，并对比五种轻量级推理时防御策略的效果。

**💡 创新点**

结合真实世界手工收集的攻击语料，系统性比较多模型与多防御表现；发现自我防御（Self‑defence）在大多数模型上显著降低易受率。

**🔧 技术方法**

使用 Prompt Injection / Jailbreak 分类、五类推理时防御（输入过滤、Self‑defence、系统提示硬化、向量检索、投票聚合）以及基于 LLM 的评判器进行安全判定。

**📊 数据集**

自构建的多来源攻击数据集（Reddit、GitHub、学术论文、社区），包含 94 条提示注入、73 条越狱以及 2,220 条长格式攻击样本。

**📈 对比分析**

对比未防御与各防御下的易受率，Self‑defence 将大部分模型的漏洞率从约70% 降至 0%，但在长链提示下仍存在 10‑20% 的残留易受；输入过滤效果最差，投票与向量防御介于两者之间。

**⚠️ 局限性**

限制在于防御仍易被长链/多步推理攻击绕过、依赖模型内置拒绝机制、对模型安全未进行内在改造；在高吞吐低延迟场景下，Self‑defence 产生的额外推理成本和潜在误判需进一步评估。

---

## 573. DeepPresenter: Environment-Grounded Reflection for Agentic Presentation Generation

**arXiv ID:** 2602.22839 | [PDF](https://arxiv.org/pdf/2602.22839v1)

**作者:** Hao Zheng `[一作]` (Chinese Academy of Sciences), Le Sun `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 6086 | [OpenAlex ID](https://openalex.org/A5034536222)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个双代理框架（研究者代理与演示者代理），能够自主探索、生成结构化手稿并以内容驱动的方式将其转换为视觉化演示，同时在环境中进行感知反思。

**💡 创新点**

创新点包括：① 引入环境感知的反思机制，将渲染后的幻灯片视觉状态作为反馈；② 采用双代理协作拆分研究与设计任务；③ 在轨迹合成阶段使用外部验证器（extrinsic verification）消除自我验证偏差并生成高质量反思提示。

**🔧 技术方法**

主要技术包括：LLM驱动的多步工具调用、HTML/Markdown手稿与幻灯片渲染、基于视觉渲染的检测工具、外部批评器产生反思语义、三阶段轨迹过滤、以及基于收集轨迹的监督微调。

**📊 数据集**

使用了从 PersonaHub、arXiv、FinePDFs‑Edu 生成的 1,152 个任务集（其中 1,024 用于训练，128 用于评估），每个任务都添加可验证的约束（如幻灯片数量、语言、纵横比）。

**📈 对比分析**

在 128 个测试任务上通过约束得分、内容与风格评分（GPT‑5 评判）以及视觉多样性（Vendi Score）进行比较，完整模型平均分 4.44，精简模型 4.19，均超过 Gamma、PPTAgent、KCTV 等基线，且接近 GPT‑5 的 4.22 分。

**⚠️ 局限性**

局限性包括：① 依赖多步工具调用，导致推理成本高且对环境不稳定（如上下文溢出、基础设施中断）敏感；② 外部验证器仅在训练阶段使用，推理时缺乏实时批评器；③ 长程执行中仍可能出现约束违规、视觉缺陷等错误。

---

## 574. Spectrally Distilled Representations Aligned with Instruction-Augmented LLMs for Satellite Imagery

**arXiv ID:** 2602.22613 | [PDF](https://arxiv.org/pdf/2602.22613v1)

**作者:** Minh Kha Do `[一作]` (La Trobe University), Ramana Rao Kompella `[通讯]` (Cisco Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并训练了一种RGB-only的视觉语言基础模型SATtxt，利用光谱表示蒸馏将多光谱先验迁移到RGB特征，并通过指令增强LLM实现视觉与文本的对齐，能够在卫星图像的零样本分类、检索和开口词分割等任务上实现高性能。

**💡 创新点**

创新点包括：① 通过Spectral Representation Distillation（SRD）将多光谱知识无缝迁移至RGB特征；② 使用冻结的LLM（如Llama‑3.1‑8B）并结合指令增强提示，显著提升文本表达能力和跨模态对齐；③ 仅需RGB输入即可匹配甚至超越多光谱模型，同时保持低训练成本。

**🔧 技术方法**

采用的技术包括：DINO式对比学习、轻量级投影器（vision & text）、多光谱教师–RGB学生蒸馏、冻结两端编码器的对齐训练、指令增强LLM文本编码器、以及多Crop增强等。

**📊 数据集**

训练使用SSL4EOS12（约1M 12波段卫星图像+配套文本），评估在EuroSAT、BigEarthNet、ForestNet三大卫星基准，以及DFC2020（开口词分割）和PANGAEA（线性探测）上。

**📈 对比分析**

与CLIP、RemoteCLIP、GeoRSCLIP、DINOv3txt、DOFA‑CLIP、Llama3‑MS‑CLIP等方法对比，SATtxt在零样本分类、检索、线性探测和开口词分割上均取得最优或最接近最优成绩：零样本分类提升约4.2%，检索提升约5.9%，线性探测提升约2.7%，开口词分割mIoU 31.23（超过Llama3‑MS‑CLIP 28.58）。

**⚠️ 局限性**

局限性在于仅适用于光学图像，无法直接处理SAR或热红外数据；依赖大型LLM文本编码器导致显存占用较大；对多光谱教师的选择仍有一定影响，且训练时仍需多光谱数据。

---

## 575. Does the testing environment matter? Carsickness across on-road, test-track, and driving simulator conditions

**arXiv ID:** 2602.22671 | [PDF](https://arxiv.org/pdf/2602.22671v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 576. A Mixture-of-Experts Model for Multimodal Emotion Recognition in Conversations

**arXiv ID:** 2602.23300 | [PDF](https://arxiv.org/pdf/2602.23300v1)

**作者:** Soumya Dutta `[一作]` (LEAP Lab), Sriram Ganapathy `[通讯]` (LEAP Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个模块化的 Mixture‑of‑Experts 框架，用于情绪识别对话，分别独立建模语音和文本的上下文，并在决策层动态融合。

**💡 创新点**

创新点在于把上下文建模与跨模态融合完全分离，并在决策层引入可学习的 MoE 门控来解决模态不平衡，辅以监督对比学习和 KL 一致性正则化。

**🔧 技术方法**

采用大语言模型（LLM）和大语音模型（SLLM）提取特征，Temporal Inception + Bi‑GRU 进行上下文建模，交叉注意力与自注意力实现多模融合，MoE 门控、focal loss、监督对比损失和 KL 正则化共同训练。

**📊 数据集**

在 IEMOCAP、MELD、CMU‑MOSI 三大基准数据集上进行实验（以及 EmoryNLP 文本实验）。

**📈 对比分析**

在三大基准上取得加权 F1 分别为 70.9%（IEMOCAP）、69.5%（MELD）和 87.9%（MOSI），显著超过多种现有基线，且模块化与 MoE 的 ablation 进一步证明其有效性。

**⚠️ 局限性**

主要限制包括：依赖大型 LLM/SLLM，导致计算和内存开销大；在自发真实对话域迁移方面尚未验证；缺乏公平性/偏见分析；未利用说话人身份信息。

---

