# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-10-28 | 今日论文总数: 770

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Barik transform and Barik eliminant: a novel algebraic method to solve polynomial equations in the sine and cosine of one variable

**arXiv ID:** 2610.36985 | [PDF](https://arxiv.org/pdf/2610.36985v1)

**作者:** Sandipan Bandyopadhyay `[一作]` (Indian Institute of Technology Madras), Sandipan Bandyopadhyay `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `Symbolic Computation` `Physics Related` `Symbolic Computation` `Algebraic Manipulation` `Review/Survey Paper`

**🎯 论文内容**

提出了Barik变换及其消元方法，用于解决至少含一条一次式（关于sinθ和cosθ）的三角方程组；

**💡 创新点**

创新点在于利用几何对称性构造线性参数化，将三角方程直接转化为一元多项式，从而避免了双角半角变换所带来的参数奇异和次数翻倍问题；

**🔧 技术方法**

采用几何坐标变换（Helmert变换）、符号代数运算（多项式除法、求GCD）以及符号求根技巧；

**📊 数据集**

本文为理论性工作，未使用任何实验数据集；

**📈 对比分析**

与传统的HTT（双角半角）或结果式（resultant）方法相比，Barik方法在满足前提条件时能够以极简闭式形式给出解，计算复杂度显著降低；

**⚠️ 局限性**

局限性：仅适用于包含至少一个一次式的单变量三角方程组，不能作为HTT的全局替代方案，且对非一次式或多变量情况仍需其他方法。

---

## 2. Nothing Left Behind: Finding Evidence from a Large Randomized Evaluation Corpus

**arXiv ID:** 2610.36910 | [PDF](https://arxiv.org/pdf/2610.36910v1)

**作者:** Giuliano Martinelli `[一作]` (World Bank Group), Samuel Paul Fraiberger `[通讯]` (World Bank Group)

**关键词:** `Computation and Language` `Retrieval` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Contrastive Learning` `Text` `Review/Survey Paper`

**🎯 论文内容**

评估两种检索架构（基于标签的检索与基于语义检索加LLM过滤）在 ImpactAI RCT 语料库中的证据检索效果，并探讨检索深度对召回率和延迟的影响。

**💡 创新点**

提出将语义检索与查询特定的 LLM 过滤分离的两阶段架构，并与传统标签检索直接对比，验证语义检索在召回率上的显著提升。

**🔧 技术方法**

使用 Gemini 2.5 Flash 生成文本嵌入、基于语义相似度的检索、LLM（Gemini 2.5 Flash）进行相关性过滤，以及基于标签的实体链接与 BM25 组合。

**📊 数据集**

ImpactAI 7354 条 RCT 的 74826 个干预-结果对；实验包括 Beta Tester 查询、Synthetic 查询、Double‑annotated 查询和基于系统综述的 348 条评估问题。

**📈 对比分析**

在候选深度 2,500 时，语义检索+过滤的宏观论文召回率从 45% 提升至 85%，过滤后精度达到 94.7%；与标签检索相比召回率提升约 40%，但检索时间显著增加。

**⚠️ 局限性**

评估仅针对 ImpactAI RCT 语料，缺乏跨领域验证；过滤模型与检索深度高度耦合，需权衡延迟；标签维护与语义索引均需持续维护，实验样本量有限。

---

## 3. Drop the Details, Get the Gist: Dynamic Conceptual Representation for Video Recognition

**arXiv ID:** 2610.36832 | [PDF](https://arxiv.org/pdf/2610.36832v1)

**作者:** Ryoya Yoshiyama `[一作]` (University of Osaka), Yuta Nakashima `[通讯]` (University of Osaka)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Transformer` `Vision Language Model` `Contrastive Learning` `Video` `Text`

**🎯 论文内容**

提出了一个基于Transformer的程序化视频识别框架，将视频划分为时间序列的段落，并通过可解释的概念得分向量进行表征，最终通过Transformer对概念序列进行建模得到视频类别；

**💡 创新点**

创新点在于将概念瓶颈模型从静态图像扩展到视频，利用可解释概念的时间序列来捕捉视频的语义事件与时序关系，并引入概念重要性评分解释预测；

**🔧 技术方法**

使用了预训练的视听语言模型（Perception Encoder）提取视频-概念相似度得分，Transformer编码器进行时序建模，并通过掩码实验计算概念重要性；

**📊 数据集**

在YouCook2（89类烹饪视频）和ActivityNet（44类家庭活动）两个多步骤程序化视频数据集上进行评估；

**📈 对比分析**

与直接使用视频特征的黑盒Transformer基线相比，概念瓶颈模型仅落后约1-2个百分点；在不同概念粒度（raw、verb、oa）和Top-k过滤下，准确率基本相当，表明粗粒度概念即可达到高精度；

**⚠️ 局限性**

局限包括高度依赖视觉-语言模型的对齐性能，概念重要性评分仅考虑单个概念的遮蔽且不捕捉交互；以及在缺乏数据集特定描述时性能略有下降，需更灵活的概念生成与选择机制。

---

## 4. AK_QMUL at ArGuard Shared Tasks: A Single Fine-Tuned JAIS for Arabic Harmful-Prompt Detection

**arXiv ID:** 2610.36360 | [PDF](https://arxiv.org/pdf/2610.36360v1)

**作者:** Ali Khairallah `[一作]` (Queen Mary University of London), Arkaitz Zubiaga `[通讯]` (Queen Mary University of London)

**关键词:** `Computation and Language` `Classification` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `LoRA` `Quantization` `Text`

**🎯 论文内容**

使用4-bit QLoRA微调JAIS-6.7B Arabic decoder并在其最后token上加线性分类头，分别解决ArGuard-2026任务B1（安全/不安全二分类）和B2（七类危害领域）

**💡 创新点**

利用单一Arabic-centric backbone跨任务共享；B1通过四路标签（安全/不安全×MSA/方言）与AraSafe安全样本增强；B2通过5个种子softmax平均提升稳健性

**🔧 技术方法**

QLoRA、LoRA适配器、NF4量化、last-token pooling、线性分类头、argmax解码、种子平均

**📊 数据集**

ArGuard-2026共享任务数据集（含方言标签的阿拉伯提示）以及AraSafe安全提示集做增量训练

**📈 对比分析**

在ArGuard排行榜上与基线比较，B1宏F1达0.984（占据首位），B2宏F1达0.790（占据首位），显著优于现有方法

**⚠️ 局限性**

仅使用单一JAIS backbone未与其他模型对比；B1未对不同种子或折叠重复消融；B2对MSA方言下Bully类仍存在显著误判

---

## 5. cuArms: GPU-Accelerated Conflict-Based Planning for Multi-Arm Manipulation

**arXiv ID:** 2610.36250 | [PDF](https://arxiv.org/pdf/2610.36250v1)

**作者:** Philip Huang `[一作]` (Nvidia), Lars Johannsmeier `[通讯]` (Nvidia)

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Reinforcement Learning` `Simultaneous Localization and Mapping` `Contrastive Learning` `Gaussian Splatting` `Optical Flow` `Point Cloud` `Mesh` `Graph`

**🎯 论文内容**

提出了 cuArms，一种GPU加速的冲突驱动多臂操纵规划算法，能够在数秒内生成并不断改进多臂任务与运动规划（TAMP）方案。

**💡 创新点**

创新点包括：① 将每臂可重用的多模态路网与冲突搜索分离，利用GPU批量几何校验；② 将LaCAM与PIBT结合，形成懒惰的多臂冲突驱动搜索；③ 通过分层的“先调度后协调”框架，将高层调度与低层运动规划耦合；④ 采用GPU并行短路与多臂路径改进实现任意改进。

**🔧 技术方法**

技术手段：GPU批量碰撞检查、CUDA并行、概率路网（PRM）与技能节点扩展、LaCAM/PIBT冲突解析、启发式后向Dijkstra、GPU加速逆运动学、基于ScheduleStream的高层调度、GPU加速的多臂短路与快捷路优化。

**📊 数据集**

数据集与实验环境：
- 仿真焊接任务（RoboBallet）：10/40个焊点，4或8臂；
- 平台上桌面操作任务：持有、堆叠、包装、拆包（2–4物体）以及手渡、双臂携带；
- 真实机器人实验：两台Flexiv Rizon4臂的电子组装任务。

实验使用NVIDIA RTX Pro 6000 GPU工作站。

**📈 对比分析**

对比方法：RL（RoboBallet）与ScheduleStream。cuArms在焊接任务中，初始解时间仅0.2–0.41 s，完成时间比RL快20–28%，并将初始完成时间缩短47%；在桌面任务中，cuArms比ScheduleStream快约46%，且初始完成时间更短；在8物体包装任务中，ScheduleStream无法得到解，cuArms能顺利求解并降低42–57%的完成时间。

**⚠️ 局限性**

局限性：
- 依赖高层调度给出的任务骨架，若骨架质量差会导致多臂规划困难；
- 搜索为启发式懒惰，缺乏完整性保证；
- 对极大物体数量或高度动态变化的环境，GPU批量碰撞检查仍可能成为瓶颈；
- 需要GPU资源，CPU端的通信和控制仍占一定比例。

---

## 6. People's Views on Smiley Emoji on WeChat Platform, Positive Emotion and Sarcasm

**arXiv ID:** 2610.35908 | [PDF](https://arxiv.org/pdf/2610.35908v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Multimedia`

---

## 7. Text Generation in Low-Resource Languages Using Human-Curated Vocabularies

**arXiv ID:** 2610.37105 | [PDF](https://arxiv.org/pdf/2610.37105v1)

**作者:** Michael Christlambert Sinanta `[一作]` (Independent Researcher), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `Computation and Language` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text`

**🎯 论文内容**

提出了一个训练无关的框架，利用人类编制的词汇表、少量人写范例和LLM推理，生成低资源语言的合成文本。

**💡 创新点**

创新点在于将词表采样与主题约束相结合，并借助LLM推理模式，显著提升极低资源语言（如Acehnese、Hitu）的词汇多样性与流畅度，同时大幅降低离题率。

**🔧 技术方法**

采用Zipfian词汇采样、主题相关低频词筛选、少样本提示、LLM推理（思考模式）以及词典式词候选提供等技术。

**📊 数据集**

使用6种东南亚语言的手工词典（印尼语、越南语、爪哇语、米南卡布语、Acehnese、Hitu）及其词频估计（通过英文词典频率映射），并为每种语言收集40句+5段的人写范例。

**📈 对比分析**

与零样本和仅范例提示两种基线对比，采用人类评测流畅度、离题率和词汇多样性指标；在极低资源语言上流畅度平均提升约0.7分，离题率降至≈0.06，词汇多样性提升≈0.5。

**⚠️ 局限性**

局限性包括仅覆盖拉丁字母语言、词典质量与方言匹配问题、仅使用三款商用API模型、未验证生成数据对后续模型训练的实际效果，并不适用于非拉丁或形态学复杂语言。

---

## 8. Proactive Memory for Event-Driven LLM Agents

**arXiv ID:** 2610.36967 | [PDF](https://arxiv.org/pdf/2610.36967v1)

**作者:** Jinyang Li `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Optimization` `Federated Learning` `Safty and Privacy` `Explainability and Interpretability` `Computational Efficiency` `Knowledge Distillation` `Representation Learning` `Adversarial Attack` `Hyperparameter Search` `Data-Centric Learning` `Robotic Intelligence` `Meta Learning` `Drug Discovery` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Chain-of-Thought` `Time Series` `Sequential` `Benchmark`

**🎯 论文内容**

提出了ProactiveMemory层级记忆框架与ProactiveStream纵向事件流基准，帮助LLM代理在持续观察与历史积累中主动识别并决策服务机会。

**💡 创新点**

创新点在于将记忆分为事件、时间轨迹和行为规律三层，并通过源验证与历史重放来确保记忆质量；同时设计了干预控制策略以降低误干预。

**🔧 技术方法**

利用大型语言模型（如GPT‑5.6 Luna/Terra/Sol）生成轨迹与规律，实施源验证、重放检测和干预控制；采用统一的干预控制政策与记忆检索接口。

**📊 数据集**

使用三类长序列数据集：Sports（HeartSteps运动数据）、Home（CASAS 家庭传感器数据）和Code（GitHub 仓库事件历史）。

**📈 对比分析**

在ProactiveStream上与五个外部基线（ProactiveAgent、ContextAgent、Mem0、A‑MEM、Graphiti）对比，ProactiveMemory在整体F1上达到52.6%，相较最强基线提升21.4%，且在每个领域表现更好，误干预率显著下降。

**⚠️ 局限性**

局限性包括：仅针对低风险服务场景（运动、家庭检查、代码维护），缺乏跨域或实时评估；性能高度依赖人工标注的参考标签；隐私、数据保留与部署安全仍需进一步验证。

---

## 9. Can LLMs Reliably Grade Olympiad Proofs? A Study of Mathematical Verification with LLMs

**arXiv ID:** 2610.36767 | [PDF](https://arxiv.org/pdf/2610.36767v1)

**作者:** Azim Ospanov `[一作]` (Huawei Foundation Model Department), Farzan Farnia `[通讯]` (Chinese University of Hong Kong)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Optimization` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Chain-of-Thought` `Prompt Engineering` `Retrieval-Augmented Generation` `Mixture of Experts` `Text`

**🎯 论文内容**

系统评估LLM在奥林匹克级数学证明验证的能力，并提出人工策划的错误注入数据集、迭代自我批判的对抗证明生成pipeline以及基于多臂赌博机的评分框架；

**💡 创新点**

创新点在于（1）构建人类标注的错误注入USAMO 2025数据集；（2）提出Targeted adversarial errOr injection for mAth evaluation（TAR‑OJE）自我批判生成对抗证明；（3）将评分转化为多臂赌博机问题，用UCB算法优化评分一致性；

**🔧 技术方法**

使用的技术包括LLM-based verifier、思考型推理（CoT）、多样本采样（MeanCoT@16、MajorityVote@16）、奖励模型（ORM/PRM）、自我批判与细化循环、SFT微调以及UCB多臂赌博机；

**📊 数据集**

主要数据集为人类策划的USAMO 2025错误注入解答集（48份），以及通过TAR‑OJE生成的合成数据集（USAMO 2025/2026、IMO 2025、IMC 2025）；

**📈 对比分析**

通过在Full/Blind两种设置下，对20个模型进行CoT@1、MeanCoT@16、MajorityVote@16、Best‑of‑N等多种采样与奖励模型进行对比，评估指标包括QWK、RMSE、Bias等；多臂赌博机方法相比ProofGrader在QWK提升26.4%、误差下降14%，但整体仍低于专家一致率（0.76）；

**⚠️ 局限性**

主要限制是需要标记方案和参考解答，无法直接应用于blind或无参照的验证场景，且对非竞赛式开放式问题的适用性有限。

---

## 10. RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions

**arXiv ID:** 2610.36791 | [PDF](https://arxiv.org/pdf/2610.36791v1)

**作者:** Haoyu Zhao `[一作]` (DAMO Academy, Alibaba Group), Zhongyu Li `[通讯]` (Hong Kong Embodied AI Lab)

**关键词:** `Robotics` `Robotic Intelligence` `Generation` `Data Synthesis` `Transformer` `World Model` `Diffusion model` `Rectified Flow` `Vision-Language-Action Model` `Contrastive Learning` `Auto Encoder` `Video` `Multimodality`

**🎯 论文内容**

提出一种跨体型视频世界模型，通过连续潜在动作作为条件接口，在人类、机器人和仿真视频上进行跨体型预训练，随后对目标机器人进行适配，并利用数字遥操作和潜在动作编辑两种数据生成管线来增强机器人策略学习。

**💡 创新点**

①将3D‑grounded潜在动作视为共享动作接口，消除不同机器人控制空间的对齐瓶颈；②在预训练阶段融合全局动作 token 与帧对齐的局部调制，提升未来视频预测质量；③提供数字遥操作与潜在动作编辑的双重数据合成策略，为机器人学习提供大规模多体型数据。

**🔧 技术方法**

使用 Cosmos3 变压器视频生成框架，配合 RynnVLA‑Latent 的连续潜在动作提取；自监督潜在动作预训练；扩散模型（rectified flow）以及零初始化的局部调制与全局注意力动作注入；实现数字遥操作与动作重标记的后处理流程。

**📊 数据集**

23个公开数据集，总计约 6,190 小时（约 2.57M 条 episodes），包括人类交互视频（HowTo100M、EgoVid、EgoDex、EPIC‑KITCHENS 等）、真实机器人演示（Marvin‑WUJI、Astribot‑S1 等）和仿真视频（RoboCasa、BEHAVIOR‑1K 等），用于预训练、适配和评估。

**📈 对比分析**

在 DreamDojo GR‑1 公开基准上与现有模型比较，PSNR/SSIM/LPIPS 均有提升；在自建双臂机器人平台（Marvin‑WUJI 与 Astribot‑S1）的视频预测同样优于基线；在真实机器人策略学习中，加入数字遥操作或潜在动作生成的数据可显著提升任务成功率（如 Plate‑Placement 由 82% 提升至 91%，Steak‑Serving 40%→54%）；在模拟政策基准 LIBERO/PLUS 上，使用 80M 合成数据中期训练后，成功率提升 2–25%。

**⚠️ 局限性**

潜在动作可能同时编码相机运动、外观变化等信息，缺乏因果辨识；合成数据的动作与生成观察仅在时序上对齐，物理一致性无法保证；实验结果基于 35 次试验，统计意义有限；合成数据提升可能由数据量与计算资源共同作用，未分离二者；模型对长时序一致性、跨体型泛化仍需进一步改进。

---

## 11. CASEL: Consequence-Aware Selective Execution for Reliable Long-Horizon Tool-Using Agents

**arXiv ID:** 2610.36570 | [PDF](https://arxiv.org/pdf/2610.36570v1)

**作者:** Yanyan Zhang `[一作]` (Carnegie Mellon University), Hao Zhang `[通讯]`

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Robotic Intelligence` `Transformer` `Large Language Model` `Agentic AI` `Risk Weighted Cost Function` `Selective Prediction` `Probability Calibration` `Text`

**🎯 论文内容**

提出一种名为CASEL的运行时决策层，能够在语言模型代理调用外部工具时，根据校准的失败概率、后果、可逆性等风险信号，选择直接执行、验证、沙箱、澄清、批准或放弃等六种干预模式。

**💡 创新点**

创新点在于将风险信号与成本、交互开销相结合，使用基于风险的成本函数进行瞬时决策；引入可逆性约束、动态干预选型、以及可校准的失效概率阈值，并给出了有限样本阈值和轨迹账本的理论检验。

**🔧 技术方法**

技术包括概率校准（等距回归/温度缩放）、选择性预测、风险加权成本函数、可逆性分类、模糊/新鲜度评估、以及离线阈值和账本策略。

**📊 数据集**

使用合成实验数据：40,000条校准动作和6,000条包含20步的测试轨迹，动作按只读、可逆、不可逆三类划分，并在每类中设定不同的失败概率和后果分布。

**📈 对比分析**

与多种基线（不干预、仅验证、基于不确定性/后果/风险的门控）在相同成本下对比，结果显示CASEL将严重副作用率从14.75%降至7.02%（或5.82%），平均归一化伤害下降0.061，任务成功率略低但成本不变；相对门控，所需人工批准比例减少约30%。

**⚠️ 局限性**

局限性包括：评估仅在模拟环境下完成，未测量实际干预效果；阈值和账本在真实部署中可能失效；方法依赖于校准的失败概率和后果估计，需要领域特定的参数；并未与其他层级决策框架（如REFLEX、CA2I）进行实验比较。

---

## 12. Visual Autoregressive Generation with Next Token Imagination

**arXiv ID:** 2610.36793 | [PDF](https://arxiv.org/pdf/2610.36793v1)

**作者:** Ruihang Jiang `[一作]`, Xiatian Zhu `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Transformer` `Diffusion model` `Auto Encoder` `Vision Language Model` `Contrastive Learning` `Image`

**🎯 论文内容**

提出了 Next Token Imagination (NTI) 训练范式，让视觉自回归模型在训练时探索非训练数据中的可行视觉序列，从而提升生成质量。

**💡 创新点**

创新点在于在非 GT 上下文中对可行的非 GT token 进行上下文感知的监督（Context‑Regulated Imagination）并通过 Snapshot‑based Context Approximation 实现高效单通道训练，打破传统 GT‑only 监督限制。

**🔧 技术方法**

采用离散视觉 tokenizer（MaskGIT‑VQGAN）、因果 Transformer、代码书相似度、总变异一致度、熵等度量进行上下文调节，并利用冻结快照实现 SCA，整体训练保持单前向通道。

**📊 数据集**

在 ImageNet‑1K（256×256）进行类条件生成实验，并在 GigaTok、AliTok 等不同 tokenizer 上验证。

**📈 对比分析**

与现有 AR、diffusion、flow‑matching 等模型对比，NTI 在相同参数量下将 FID 从 3.02 降至 1.67，IS 提升至 312.72，保持标准 AR 结构且训练成本仅提升约10%，在多种模型规模和 tokenizer 上均优于或匹配同类方法。

**⚠️ 局限性**

仍依赖离散 tokenizer 的预训练质量，且对极大模型或高分辨率生成的扩展未充分验证；想象范围受限于代码书相似度和上下文一致性阈值，可能在更复杂视觉内容中表现受限。

---

## 13. Whether Is Not When: Separating Occurrence from Timing in Polymarket Prices

**arXiv ID:** 2610.36639 | [PDF](https://arxiv.org/pdf/2610.36639v1)

**作者:** Sizheng Fan `[一作]` (University of International Business and Economics), Rong Zhao `[通讯]` (Shenzhen University of Information Technology)

**关键词:** `Computers and Society` `Finance Related` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Retrieval-Augmented Generation` `Tabular` `Time Series` `Sequential`

**🎯 论文内容**

分析 Polymarket 价格是否能作为早期警报，区分价格信息中的发生与时间两部分。

**💡 创新点**

首次将预测市场价格拆分为发生与时间信息，并用大规模交易数据验证其效能。

**🔧 技术方法**

采用等距回归、Log 评分拆分、洗手交易过滤、阶梯读取等技术。

**📊 数据集**

使用 1.24 亿笔链上交易、208 事件、338 死线梯形和 51 次央行决策等数据集。

**📈 对比分析**

通过 recall/accuracy、信息量（比特）和警报性能对比，发现 1 天前准确率为 56%，时间信息仅占 1/8，梯形读取比单合同更精准。

**⚠️ 局限性**

局限在于事件时间依赖 LLM 搜索、样本量相对有限、账号多样性不足以及未深入解释价格背后的动因。

---

## 14. Peak-DNA: Recurrent Morphology in Global Narrative Propagation

**arXiv ID:** 2610.36571 | [PDF](https://arxiv.org/pdf/2610.36571v1)

**作者:** Joosung Kim `[一作]` (Nagoya University), Shanwen Wu `[通讯]` (Nagoya University)

**关键词:** `Cryptography and Security` `Anomaly Detection` `Recommendation System` `Computational Efficiency` `Representation Learning` `DTW Similarity` `Hierarchical Clustering` `Time Series Normalization` `Savitzky–Golay Smoothing` `Time Series` `Tabular` `Benchmark` `Review/Survey Paper`

**🎯 论文内容**

本文提出Peak‑DNA框架，用于在噪声严重的全球叙事传播流中识别重复出现的传播形态。

**💡 创新点**

创新在于将宏峰合并、形态归一化并使用DTW进行结构相似度分析，首次将叙事传播视作可度量的“基因组”结构。

**🔧 技术方法**

使用Savitzky–Golay平滑去噪、宏峰合并、归一化时间序列、DTW相似度、层次聚类等技术。

**📊 数据集**

利用2026年1月1日至5月12日的GDELT事件数据，聚焦五个高放大叙事领域（伊朗、国防、AI、太空/埃隆、量子/PQC）。

**📈 对比分析**

通过DTW相似度与层次聚类对宏峰形态进行比较，结果显示同一领域内宏峰高度聚类，跨领域存在部分共享形态，表现出强内部结构紧凑性。

**⚠️ 局限性**

局限在于仅覆盖单一时间窗口、未对参数进行领域特定调优、缺乏纵向演化分析及实际应用验证。

---

## 15. A Simple Presentation of Robust Membership Inference Attacks

**arXiv ID:** 2610.36458 | [PDF](https://arxiv.org/pdf/2610.36458v1)

**作者:** Renaud Gaucher `[一作]` (École Polytechnique), Lydia Zakynthinou `[通讯]` (Johns Hopkins University)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Adversarial Attack` `Bayesian Inference` `Contrastive Learning` `Gaussian Splatting` `Stochastic Differential Equation` `Ordinary Differential Equation` `Tabular` `Synthetic Data`

**🎯 论文内容**

本文提出并证明了针对高维高斯均值估计和二进制产品分布的鲁棒成员推断攻击，并将其与差分隐私下的误差下界相结合，构成一种通用的理论框架；

**💡 创新点**

创新点在于把贝叶斯后验分析与简单的内积检测结合，得到一个不依赖具体机制、只需已知精度上界的攻击，同时给出了匹配的下界并系统梳理了相关文献；

**🔧 技术方法**

使用的技术包括贝叶斯推断（高斯/贝塔后验）、中心极限定理与Hoeffding、Chebyshev不等式、Paley–Zygmund、矩估计、以及对齐误差与真实均值的几何分析；

**📊 数据集**

实验数据集为合成的高斯样本（均值为0，协方差为单位矩阵）和伯努利样本（均值在[0,1]^d 上均匀抽样），并通过理论证明展示攻击效果；

**📈 对比分析**

通过理论分析与对比，攻击在维度满足 d ≳ n + α²n²log(1/β) 时可识别至少 O(1/α²) 个成员，且下界与差分隐私所需噪声的量级相匹配，说明攻击在高维、低误差场景下具有最优性；

**⚠️ 局限性**

局限性包括：仅适用于均值估计或结构相似的参数估计问题；假设攻击者已知完整的生成分布且可使用先验；在实际大规模机器学习模型上实现时仍需依赖实验验证；以及对非独立同分布、非高斯/伯努利数据的适用性不足。

---

## 16. Bridging Exploration and Loss of Plasticity in On-Policy Reinforcement Learning

**arXiv ID:** 2610.36309 | [PDF](https://arxiv.org/pdf/2610.36309v1)

**作者:** Andrea Vasco Grieco `[一作]` (Politecnico di Torino), Raffaello Camoriano `[通讯]` (Politecnico di Torino)

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning` `Tabular` `Time Series` `Sequential`

**🎯 论文内容**

本文通过理论推导和实验验证，阐明了在使用高斯策略的PPO中，标准差σ的衰减会导致梯度范数和Hessian尖锐度发散，进而引发Loss of Plasticity（LoP），并证明了σ的演化是现有LoP缓解方法（如Adam调参、L2正则、CBP等）有效性的根本原因；

**💡 创新点**

创新点在于首次将σ收缩与LoP联系起来，给出梯度范数、Hessian尖锐度以及Hessian近似秩的严谨上界，并用该理论解释了多种经验手段（Adam β1=β2、熵正则、L2正则等）的工作原理；

**🔧 技术方法**

使用的技术包括PPO框架、可微高斯策略、理论分析（梯度与Hessian矩阵性质）、对比实验（PPO、Tuned Adam、L2、CBP、Adaptive LR等）以及在仿真与真实机器人（Franka Panda）上的部署；

**📊 数据集**

实验数据集主要有Isaac Lab的Isaac‑Reach‑Franka‑v0、MuJoCo Ant/Slippery Ant、以及真实Franka 7‑DoF机械臂的实验；

**📈 对比分析**

通过与标准PPO、Tuned Adam、L2正则、CBP等方法对比，结果表明对σ进行重置或熵正则能显著避免LoP崩溃，并在仿真和真实世界任务中取得更高奖励，性能明显优于仅使用标准PPO；

**⚠️ 局限性**

局限性包括仅针对可微高斯策略的PPO研究，未覆盖离散动作或非高斯策略；理论假设虽弱，但仍需要进一步验证在更大规模或不同RL算法中的适用性；实验中对σ统计估计和常数的精确值尚未给出，实际效果受超参数设置影响；

---

## 17. Sensitivity of RMS Delay Spread Estimation to Power Delay Profile Thresholding in Time-Varying Channels at 33 GHz

**arXiv ID:** 2610.36783 | [PDF](https://arxiv.org/pdf/2610.36783v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Information Theory`

---

## 18. Continue or Restart? The Limits of Early Failure Prediction in Coding Agents

**arXiv ID:** 2610.35971 | [PDF](https://arxiv.org/pdf/2610.35971v1)

**作者:** Srinath Srinivasan `[一作]` (NC State University), Wesley Assunção `[通讯]` (NC State University)

**关键词:** `Software Engineering` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Mixture of Experts` `Text` `Sequential`

**🎯 论文内容**

对 147,002 条编码代理执行轨迹进行分析，比较基于早期前缀的失败预测模型、任务条件回归和细调神经监控器，并在共享令牌预算下重放停止策略。

**💡 创新点**

揭示了跨任务排名与任务内判别的混淆，证明早期前缀在同一任务上的判别力接近随机，并通过解析式分解阐明学习停止相对随机重启的优势仅来自任务层面的分配，而非尝试层面的识别。

**🔧 技术方法**

使用线性计数回归、冻结句子嵌入、任务条件逻辑回归、LoRA 微调的 Qwen3‑0.6B 神经监控器、共享令牌预算的记录重放、解析式差分以及 Pareto 主导性分析等技术。

**📊 数据集**

数据集包括 Nebius 的 SWE‑rebench（OpenHands+Qwen3‑Coder‑480B‑A35B‑Instruct）和 NVIDIA 的 Open‑SWE‑Traces（SWE‑agent+Qwen3.5、OpenHands+Qwen3.5、OpenHands+MiniMax‑M2.5）四种模型-框架组合，共 147,002 条执行轨迹。

**📈 对比分析**

评估指标为池化 AUC、任务内 AUC、对数损失；对比学习停止、随机重启与不间断继续，在 30k–100k 令牌上进行重放，学习停止相较随机仅提升至 1.98pp，然而不间断继续在解题率和平均令牌消耗上始终占优。

**⚠️ 局限性**

局限在于仅测试四个 Python 代码库的模型-框架组合，未考虑状态保留恢复、内部表示或序列反馈等更先进监控技术，且早期前缀判别有限，结果可能不易推广至其他语言或代理。

---

## 19. TACO-Graph: Automated Construction of Text-Atributed Graphs

**arXiv ID:** 2610.36650 | [PDF](https://arxiv.org/pdf/2610.36650v1)

**作者:** Amiri Hayes `[一作]` (Massachusetts Institute of Technology), Kristina Wicke `[通讯]` (New Jersey Institute of Technology)

**关键词:** `Machine Learning` `Classification` `Computational Efficiency` `Representation Learning` `Retrieval` `Graph Neural Network` `Transformer` `Retrieval-Augmented Generation` `Contrastive Learning` `Prompt Engineering` `Text` `Graph` `Tabular`

**🎯 论文内容**

本文提出一种名为TACO-Graph的框架，用来系统地枚举、评估并为文本属性图（TAG）构造选择最优方案。

**💡 创新点**

创新点在于将图构造视为可学习的决策空间，利用小样本代理评估与可解释的决策树来预测不同构造在下游任务中的表现，从而避免手工试错或专家知识。

**🔧 技术方法**

方法主要使用决策树回归、轻量级GNN（GraphSAGE）与GraphRAG的RAGAS评估，以及在代表性子样本上快速计算得分。

**📊 数据集**

实验涵盖五个真实数据集：ArXiv、Sporting Goods、Electronics、Toys（来自Amazon评论）和Books-History，均为多属性表格数据。

**📈 对比分析**

对比实验显示，决策树在独立子样本上对构造质量的Spearman相关系数平均高于0.9（节点分类）和0.8（GraphRAG检索），并且通过代理评估可将完整模型训练成本降低约2.5–9.3倍，显著提升计算效率。

**⚠️ 局限性**

局限性包括：仅针对表格数据的构造空间；对构造得分分布要求较高，若差异不明显则学习效果差；GraphRAG任务的计算开销仍高，代理评估对检索质量的预测不如节点分类稳健。

---

## 20. When Retrieval Is Not Enough: Obsolete Facts in Agent Memory

**arXiv ID:** 2610.36308 | [PDF](https://arxiv.org/pdf/2610.36308v1)

**作者:** Oleeviya Babu Poikarayil `[一作]` (Otto von Guericke University Magdeburg), Klaus Turowski `[通讯]` (Otto von Guericke University Magdeburg)

**关键词:** `Artificial Intelligence` `Retrieval` `Explainability and Interpretability` `Computational Efficiency` `Benchmark` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `McNemar statistical test` `Text` `Benchmark`

**🎯 论文内容**

探究MemoryAgentBench单跳FactConsolidation任务中检索预算、竞争陈述以及模型如何挑选最新事实的影响，评估检索证据的规模和内容对答案质量的作用。

**💡 创新点**

通过在固定prompt下对检索证据进行等长替换干预，明确证明过时竞争事实比无关证据更易误导模型；提出检索证据的诊断式审核方法。

**🔧 技术方法**

使用BM25文本检索、Qwen2.5-7B-Instruct与Qwen3-14B大型语言模型、固定prompt、McNemar统计检验以及人工审核。

**📊 数据集**

MemoryAgentBench的单跳FactConsolidation数据集，分别包含6K历史（455条记忆+100问）和32K历史（2,310条记忆+100问）。

**📈 对比分析**

对比不同token预算（64、128、256、512）和等长干预（A/B），以及32K历史复制；发现预算增大会显著降低准确率，且过时陈述导致准确率从100%降至44.6%（Qwen2.5）或87.8%（Qwen3）。BM25-64优于BM25-512。

**⚠️ 局限性**

仅测试两款Qwen模型、单跳链、语义检索、synthetic数据；未评估密集检索、多跳或跨模型情况；统计受限于共享历史，结果不一定可推广到真实部署环境。

---

## 21. Learning Where to Trust: Reliability-Guided Expert Routing and Velocity Gating for 3D Deformable Registration

**arXiv ID:** 2610.36672 | [PDF](https://arxiv.org/pdf/2610.36672v1)

**作者:** Chenyu Guo `[一作]` (Harbin Institute of Technology), Shuhan Qi `[通讯]` (Harbin Institute of Technology)

**关键词:** `Artificial Intelligence` `Optimization` `Computational Efficiency` `Representation Learning` `Transformer` `Mixture of Experts` `Contrastive Learning` `Diffusion model` `Biomedical Data` `Magnetic Resonance Imaging` `Computed Tomography`

**🎯 论文内容**

提出了一种名为ReliMorph的三维可变形配准框架，利用局部匹配可靠性来指导空间上下文选择与残差速度调节，实现从粗到细的对称变形更新。

**💡 创新点**

创新点在于首次将对应关系可靠性（基于局部相关分布的熵、峰值、期望偏移等统计）与空间专家路由和残差速度门控同时耦合，结合逆一致性反馈动态调整后续层级的路由和更新强度。

**🔧 技术方法**

技术手段包括共享解码器+轻量级模态残差适配、局部相关分布统计、三种深度可分离专家（局部、轴向、膨胀）路由、概率门控残差速度、差分式尺度平方整合以及对称逆向传播。

**📊 数据集**

在IXI脑T1‑MRI、OASIS脑T1‑MRI、以及跨模态AbdomMRCT（CT→MR）三组公开数据集上进行实验。

**📈 对比分析**

与TransMorph、CorrMLP、LGANet++、SACB‑Net、ShiftMorph、SyN、LapIRN、SYMNet等多种基线相比，ReliMorph在DSC最高、HD95最低、折叠率最低，且在速度与显存上保持竞争优势；消融实验验证了可靠性统计、专家路由、速度门控和一致性反馈的关键作用。

**⚠️ 局限性**

局限性包括对逆一致性与无折叠性的完整保证缺失，模型对极端大变形或极低对比度区域的鲁棒性仍有待提升，且训练时需额外计算相关分布与门控，导致一定的算力与存储开销。

---

## 22. Where Does My Call Go? Measuring Google Meet, Zoom, and Microsoft Teams on Starlink

**arXiv ID:** 2610.35874 | [PDF](https://arxiv.org/pdf/2610.35874v1)

**作者:** Kousar Malekinasab `[一作]` (University of Victoria), Jianping Pan `[通讯]` (University of Victoria)

**关键词:** `Networking and Internet Architecture` `Optimization` `Federated Learning` `Computational Efficiency` `Tabular` `Time Series` `Audio` `Benchmark`

**🎯 论文内容**

对Google Meet、Zoom、Microsoft Teams在Starlink网络（静止与机上）下的WebRTC服务点选择与RTT性能进行系统测量与分析。

**💡 创新点**

首次系统比较三大RTC平台在LEO星链环境下的服务点选取策略，并揭示平台与星链PoP切换不匹配导致的RTT膨胀现象，推动对LEO‑优化服务端部署的研究。

**🔧 技术方法**

使用浏览器版WebRTC与Selenium+JavaScript Hook自动化会话；提取ICE候选、RTT、路径信息；通过IP归属、ASN与地理定位推断服务端点；在加拿大CCCLEO测试床及6条Starlink航班上采集数据。

**📊 数据集**

7个加拿大PoP（Victoria、Whitehorse、Saskatoon、Thompson、Iqaluit、Fredericton、St. John's）的静止测量；6条机上航班（YYJ–YEG、YEG–YYZ、YVR–OGG）共计超过24小时的轨迹与RTT记录；公开的RTC会话日志与GitHub数据集。

**📈 对比分析**

对每次会话提取服务点IP、ASN、位置并与当前PoP对齐，统计RTT分布、峰值与尾部；结果显示：Google Meet RTT最低（多数<100 ms），Zoom次之，Microsoft Teams最高（120–200 ms）；在机上时，Google Meet随PoP动态切换，Teams保持会话固定，Zoom在区域中心间跳变导致RTT波动。

**⚠️ 局限性**

仅测浏览器版，未涵盖原生客户端；测量范围局限于加拿大与部分西杰特航班；服务点推断依赖IP归属与地理定位的准确性；未评估视频质量、拥塞、丢包等QoE指标；多方会议规模与多路径路径分析有限。

---

## 23. Self-Keyed In-Parameter Memorization for External Memory-Free Dialogue Recall in Frozen Language Models

**arXiv ID:** 2610.36948 | [PDF](https://arxiv.org/pdf/2610.36948v1)

**作者:** XinYu Piao `[一作]` (Korea University), Jong-Kook Kim `[通讯]` (Korea University)

**关键词:** `Computation and Language` `Retrieval` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Benchmark`

**🎯 论文内容**

提出一种自键自存的参数化记忆方法Skim，在没有外部记忆、梯度微调或未来问题/答案信息的前提下，能够将对话中的关键信息写入LLM参数并在后续对话中检索。

**💡 创新点**

创新点在于：①使用模型自身生成的问答对作为存储键，避免需预先给定事实与提示；②采用闭式岭回归一次性更新单一feed‑forward层，实现多条记忆并存且不随对话长度增长；③通过自检索调参的方式实现模型自适应配置。

**🔧 技术方法**

核心技术包括：自问自答式候选抽取、键/值评估与选择、闭式岭回归参数更新、一次性全局重解以消除顺序写入的遗忘。

**📊 数据集**

实验使用LongMemEval长对话回忆基准（单场景子集），共33条非数值事实；在五大主流LLM（Qwen3、Nemo、Yi-1.5、Llama-3、Mistral）上评估。

**📈 对比分析**

与KV缓存、True-Q、Self-Q等基线比较，Skim在N=32时平均回忆率为63.1%，最强模型达75.0%，远高于无缓存基线（3.1–15.6%）并可与一次性批量写入保持一致。

**⚠️ 局限性**

局限在于：①记忆写入主要受自抽取阶段影响，描述性多词答案抽取率低；②对数值型答案的存储与检索效果不佳；③自键生成依赖模型自身推理，可能导致键不完全覆盖所有后续问题的变体。

---

## 24. SpecGuard: Detecting Multi-Turn LLM Jailbreaks via Specification-Based Runtime Monitoring

**arXiv ID:** 2610.36258 | [PDF](https://arxiv.org/pdf/2610.36258v1)

**作者:** Junle Li `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Bingzhuo Zhong `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `Cryptography and Security` `Safty and Privacy` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Sequential` `Benchmark`

**🎯 论文内容**

设计并实现了一种无训练、基于规范的监控框架SpecGuard，用于实时检测多轮LLM jailbreak。

**💡 创新点**

创新点在于将多轮对话安全视为可执行的线性时序逻辑规范，构造固定的原子命题契约，将安全相关事实抽象为可追溯的原子命题，并将规范编译为符号有限自动机，实现可审计、可重现的违规检测。

**🔧 技术方法**

采用原子命题抽取、LTL_f 规范、符号确定有限自动机、运行时与审计模式相结合的技术方案。

**📊 数据集**

使用三大多轮安全基准：SafeDialBench-ZH、SafeDialBench-EN、SafeMTData，共计 10,075 句对话（48,933 轮）。

**📈 对比分析**

与四个开源守卫模型（ShieldGemma‑2B、WildGuard‑7B、BingoGuard‑8B、GuardReasoner‑8B）进行相同输入对比，SpecGuard 在所有基准上获得最高 AUC、F1、召回率，并保持较低的 FPR，整体平衡性能最优。

**⚠️ 局限性**

局限性在于依赖人工定义的命题与规范，可能无法覆盖未知攻击模式；对单轮细粒度判定不如专门训练的模型，且需要人工维护命题集。

---

## 25. Learning to Code the World: Harness Evolution for Compositional 3D Reconstruction

**arXiv ID:** 2610.37046 | [PDF](https://arxiv.org/pdf/2610.37046v1)

**作者:** Haoyu Yao `[一作]` (Peking University), Yadong Mu `[通讯]` (Peking University)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Vision-Language-Action Model` `Auto Encoder` `Prompt Engineering` `Diffusion model` `Image` `Mesh` `Point Cloud` `Text`

**🎯 论文内容**

提出 SceneEvolve 框架，利用可执行代码的语义接口和教师引导的 harness 进化，实现单图像到可编辑、可组合的 3D 场景的高质量重建。

**💡 创新点**

创新点：1) SceneAct 提供参数化原语、层级组合和语义空间关系的可执行接口，显著降低几何推理难度；2) 将可重用的重建 harness 作为可进化目标，教师在固定模型参数下优化执行环境；3) 证明小型开源模型通过 harness 进化即可超越大型专有模型的性能。

**🔧 技术方法**

技术手段：基于 Qwen3.8‑27B 的代码生成式大型语言模型；教师‑学生 harness 进化循环；Blender 执行后端；可执行程序接口（SceneAct）；层级语义操作与关系约束；质量评估指标（SSIM、LPIPS、DINOv2、PSNR）。

**📊 数据集**

数据集：ScanNet、CA‑1M（内部开发与测试），ARKitScenes（外部测试）。

**📈 对比分析**

比较方式：在不做模型微调的零样本设置下，与 GPT‑5.6‑Sol、GPT‑6‑Astra、Claude‑Opus‑5、Gemini‑Flash‑3.8 等专有模型进行对比；评估指标包括 PSNR、SSIM、LPIPS、DINOv2；结果显示 SceneEvolve 在所有四项指标上均优于基线，推理时间下降 47% 以上，等效部署成本降低 59.8%。跨模型迁移到 Kimi‑K3 后仍保持与前沿模型相当的性能。

**⚠️ 局限性**

局限性：1) 对极端视角、光照变化或细粒度几何细节的恢复仍不够完美；2) harness 进化需要教师指导、开发集支持，迁移到全新领域需重新进化；3) 颜色和灯光的准确性有时不足，导致重建结果与真实场景仍存在色差。

---

## 26. Hierarchical Low-Rank Reparameterization for Parameter-Efficient

**arXiv ID:** 2610.37068 | [PDF](https://arxiv.org/pdf/2610.37068v1)

**作者:** Rong Huang `[一作]` (University of Science and Technology of China), Xiao-Ming Fu `[通讯]` (University of Science and Technology of China)

**关键词:** `Computation and Language` `Computational Efficiency` `Knowledge Distillation` `Representation Learning` `Transformer` `Supervised Fine-Tuning` `Low-Rank Factorization` `Parameter-Efficient Fine-Tuning (PEFT)` `Text`

**🎯 论文内容**

本文提出了一种层级低秩重参数化框架HiLoRR，用于进一步压缩现有参数高效微调（PEFT）方法产生的可训练参数，从而在保持任务性能的同时显著降低参数量。

**💡 创新点**

创新点在于将可训练参数重新组织为近似正方形矩阵，并通过递归的低秩分解进行压缩，形成多层级结构，能够统一应用于LoRA、DoRA、MoSA、RaSA等多种PEFT方法，展示了通用且高效的压缩思路。

**🔧 技术方法**

使用的主要技术包括近似正方形矩阵重排（Near‑Square reshaping）、低秩分解（rank‑k factorization）、递归层级重参数化、以及对传统LoRA等适配器的直接改造。

**📊 数据集**

实验使用的主要数据集为Commonsense‑15K进行下游微调，随后在八个常识推理基准（BoolQ、PIQA、SIQA、ARC‑c、ARC‑e、OBQA、HellaSwag、WinoGrande）以及MetaMathQA‑40K进行数学推理微调后在GSM8K基准上评估。

**📈 对比分析**

在与LoRA、DoRA、MoRA、NoRA、SVFit等基线比较时，HiLoRR在相同或更低的可训练参数预算下取得了更高或相近的平均准确率；在压缩约10倍参数的情况下仍能保持与原始方法相当的性能，并在多数基准上超过基线。

**⚠️ 局限性**

局限性包括对学习率、初始化、层级深度和重排策略的高度敏感；更深层级的分解在大参数预算下可能导致优化困难；目前实验仅覆盖Transformer语言模型，需进一步验证在其他模型架构和任务上的适用性。

---

## 27. HIDRA: Hierarchical Dual-Routing Attention for Replay-Free Lifelong Imitation Learning

**arXiv ID:** 2610.36374 | [PDF](https://arxiv.org/pdf/2610.36374v1)

**作者:** Fanqi Yu `[一作]` (Istituto Italiano di Tecnologia), Vittorio Murino `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Representation Learning` `Transformer` `Vision-Language-Action Model` `Mixture of Experts` `Contrastive Learning` `Prompt Engineering` `Retrieval-Augmented Generation` `Multimodality` `Image` `Text` `Benchmark`

**🎯 论文内容**

在无重放、单次传递的终身模仿学习环境下，本文提出了HIDRA路由框架以实现稳定的专家选择和持续学习；

**💡 创新点**

创新点在于通过分层双注意力机制将指令检索与上下文细化解耦，并引入键级几何正则化以缓解路由干扰，支持可扩展的专家库；

**🔧 技术方法**

技术方案包括LoRA轻量级专家模块、双注意力路由、键正则化、预训练视觉-语言-动作(backbone)与多模态融合；

**📊 数据集**

使用LIBERO基准（OBJECT、GOAL、SPATIAL、10、MIX、INS等）以及自定义的LIBERO-MIX、LIBERO-INS数据集进行评估；

**📈 对比分析**

与Replay、Replay-free、DMPEL、ISCIL、L2M、EWC等方法对比，HIDRA在AUC、前向迁移(FWT)表现最佳，且保持低遗忘；

**⚠️ 局限性**

对细粒度空间任务的鲁棒性仍有限，空间表示需要进一步提升，且未在真实机器人平台上进行验证。

---

## 28. Skill Dreaming: Evolving Repository Skills through Pull Request Streams

**arXiv ID:** 2610.36188 | [PDF](https://arxiv.org/pdf/2610.36188v1)

**作者:** Xinyuan Song `[一作]` (Causal Dynamics Lab), Liang Zhao `[通讯]` (Causal Dynamics Lab)

**关键词:** `Artificial Intelligence` `AI Code Assistant` `Data-Centric Learning` `Federated Learning` `Optimization` `Computational Efficiency` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Sequential`

**🎯 论文内容**

构建并持续演化基于拉取请求(PR)流的可执行技能库，用已验证的 PR 记录来指导大语言模型进行软件仓库的功能实现与维护。

**💡 创新点**

创新点在于：①将 PR 历史转化为可执行、可持续更新的技能，并通过七种对齐操作（Add、Update、Specialize、Extend、Conflict、Supersede、Keep）实现技能的增删改；②闭环的 PR‑>技能‑>PR 迭代流程，完全不需更新 LLM 参数；③多层次技能分类（检测、构造、实现、维护、演化）及其与 PR 记录的关联维护。

**🔧 技术方法**

技术手段包括：PR 记录构建（功能定位、PR 检索与重构）、固定权重 LLM 的技能生成与对齐、检索增强生成 (RAG)、技能模板与多模态检索、基于任务相似度的技能注入、验证与回溯修复、以及通过 Claude Opus / Sonnet 进行任务执行与评估。

**📊 数据集**

数据集：SWE‑bench Pro 与 SWE‑bench Verified（12 个仓库），包含各自的 PR 历史、测试用例与代码；额外使用 1,000 条 Ansible PR 用于质量分类；GitHub API 提取 PR 元数据。

**📈 对比分析**

与检索基线（BM25、dense、Hybrid）及现有技能学习方法（Trace2Skill、Learning to Commit、SkillLens、CODESKILL、Socratic‑SWE）进行对比。SWE‑bench Pro/Verified 上 resolved rate 从 0.79/0.83 提升至 0.88/0.92，安全率提升至 1.00，Maintenance Surface F1 达 0.74；在跨仓库实验中，任务成功率提升 0.07–0.20，表明方法具备较强的迁移与通用性。

**⚠️ 局限性**

局限性：① 依赖丰富且可用的 PR 历史，无法直接适用于无 PR 或 PR 数据稀缺的仓库；② 对 PR 质量与复现性的依赖，误导的 PR 可能污染技能库；③ 相比基线，算力与 token 消耗略高，尤其在大模型或大规模仓库时需进一步评估可扩展性；④ 目前多聚焦于单语种（英文）代码，跨语言适配待研究。

---

## 29. Beyond Structural Unification: A Perspective and Roadmap on Unified Multimodal Models

**arXiv ID:** 2610.35906 | [PDF](https://arxiv.org/pdf/2610.35906v1)

**作者:** Yinyi Luo `[一作]` (Carnegie Mellon University), Jindong Wang `[通讯]` (William & Mary)

**关键词:** `Multimedia` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Transformer` `Vision Language Model` `Contrastive Learning` `Reinforcement Learning` `Prompt Engineering` `Mixture of Experts` `Multimodality` `Text` `Image` `Benchmark`

**🎯 论文内容**

提出了统一多模态模型（UMM）在功能统一方面存在的“统一悖论”，并阐述了五个关键层面（输入、表示、推理、优化、评估）需要重新思考以实现真正的功能统一；

**💡 创新点**

创新点在于将功能统一定义为在保持所有关键能力无显著退化的前提下实现整体收益，并给出了可量化的统一收益与能力保持门限、干预矩阵以及完整的评估协议；

**🔧 技术方法**

利用动态输入校准、分解表示（语义流与重建流）、持久多模态信念状态、梯度协同优化以及跨任务系统评估等技术手段；

**📊 数据集**

主要参考公开基准（VQA、MMMU、MM-Vet、TokenFlow、POPE、HallusionBench 等）和统一评测框架中的标准数据集；

**📈 对比分析**

与传统结构统一的UMM对比，提出统一收益指标 D 及各能力增益 Δ_k，使用配对自助抽样得到置信区间来判定功能统一；实验表明结构统一不一定带来整体收益，需通过功能统一评估才能揭示真正的性能提升；

**⚠️ 局限性**

局限性包括：缺乏端到端的实证验证，所给技术方案仅为示例；评估框架依赖于预先声明的能力集、权重与阈值，结果高度依赖具体部署场景；未提供统一模型的完整实现与大规模实验。

---

## 30. Remembering Is Not Enough: Refining Memories and Learning When to Use Them in Multi-Agent Systems

**arXiv ID:** 2610.36652 | [PDF](https://arxiv.org/pdf/2610.36652v1)

**作者:** Yipeng Hao `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Optimization` `Federated Learning` `Explainability and Interpretability` `Computational Efficiency` `Meta Learning` `AI Code Assistant` `Drug Discovery` `Robotic Intelligence` `Transformer` `Large Language Model` `Agentic AI` `Retrieval-Augmented Generation` `Prompt Engineering` `Chain-of-Thought` `Mixture of Experts` `Text` `Code` `Sequential` `Benchmark`

**🎯 论文内容**

提出一种无训练框架 DiscernMem，用于在多代理 LLM 系统中区分记忆缺陷与上下文误用，并通过对记忆内容进行精细修正与元记忆学习来提升协作性能。

**💡 创新点**

将记忆内容修正与适用边界学习分离，利用配对重放评估单个记忆的影响并在历史任务上验证修复，同时记录使用历史以学习角色特定的使用与回避条件。

**🔧 技术方法**

采用配对重放（paired replay）、缺陷感知记忆细化（DMR）、边界学习元记忆（BLM）等技术，LLM 作为决策者，使用 Qwen3-4B-Instruct-2507 及其嵌入模型。

**📊 数据集**

在四个基准上评估：TriviaQA、PopQA、KodCode、AgentBoard 的 PDDL 任务，涵盖问答、代码生成和符号规划。

**📈 对比分析**

与 20+ 传统与可训练内存方法比较，使用 AutoGen、CAMEL、MacNet 三种多代理架构，在所有基准上均取得最高平均分，代码生成和规划任务表现最显著，平均提升 0.5–4.3 分。

**⚠️ 局限性**

仅依赖可重现的自动评分任务，无法处理开放式生成或不可逆环境；对单个记忆的评估忽略多记忆交互，元记忆边界的自我修正能力有限。

---

## 31. They Will See Johnny's Message: A Measurement Study of Post-Quantum Cryptography Across Email Delivery Paths

**arXiv ID:** 2610.36145 | [PDF](https://arxiv.org/pdf/2610.36145v1)

**作者:** Jaehwan Park `[一作]` (University of Tennessee), Doowon Kim `[通讯]` (University of Tennessee)

**通讯引用:** 931 | [OpenAlex ID](https://openalex.org/A5025936242)

**关键词:** `Cryptography and Security` `Security` `Network Protocols` `Cryptography` `Text` `Email Data` `Network Traffic Data` `Benchmark` `Measurement Study` `Cryptography Related`

**🎯 论文内容**

本文通过对Tranco Top 1M域名的三次快照扫描、邮件客户端与发送MTA的受控实验以及OpenPGP/S/MIME实现的加密测试，系统评估了电子邮件生态在TLS层和消息层对后量子密码（PQC）的部署、协商与可用性。

**💡 创新点**

创新点在于首次将服务器端PQc支持、客户端协商行为和消息层加密互操作性三大视角结合，并提出结构化的PQC密钥语料库和针对TLS 1.3的混合/纯组探测方法，以揭示从能力到实际使用的完整落地障碍。

**🔧 技术方法**

主要技术包括：基于OpenSSL 3.5.5的TLS握手扫描、三种PQC组（传统、混合、纯ML‑KEM）顺序探测、Apple Mail/Gmail/Outlook等主流客户端的TLS捕获、Postfix/Exim等MTA的可配置实验，以及GnuPG、PGPony、S/MIME证书导入与加解密验证。

**📊 数据集**

数据集包括：2026年7月26日的Tranco Top 1M域名快照（共约1.7 M域），对应的MX、提交、访问和Webmail端点；一套19个OpenPGP密钥与6个S/MIME证书的结构化语料库；以及来自八大邮件服务商的发件MTA测试结果。

**📈 对比分析**

比较方法：按服务阶段（提交、转发、访问、Webmail）计算PQC可用率；按提供商统计混合/纯组支持占比；对客户端和MTA进行对比实验，评估协商成功率、降级到明文的比例；对消息层实现进行成功率比较，发现GnuPG/PGPony在不同后端下的成功率差异明显；性能表现上，PQC握手在受支持端点上平均延迟约+10 %相较传统TLS，消息层加解密在纯PQC密钥上仍可完成但兼容性有限。

**⚠️ 局限性**

局限性：仅评估基础设施而非真实邮件流量；Tranco域名样本不代表所有用户邮箱，且对快照时间窗口有限；仅测试八个发件服务商，未覆盖所有MTA实现；客户端和后端依赖导致结果高度相关；未调查运营商和厂商的组织决策，只识别技术障碍；对PQC仅做实现层面评估，未覆盖身份验证和证书链兼容性等更深层安全问题。

---

## 32. Condition-Enhanced Generative Adversarial Network for Aerodynamic Configuration Design of Commercial Aircraft

**arXiv ID:** 2610.35963 | [PDF](https://arxiv.org/pdf/2610.35963v1)

**作者:** Youzheng Yu `[一作]` (Northwestern Polytechnical University), Weiwei Zhang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 8218 | [OpenAlex ID](https://openalex.org/A5003289067)

**关键词:** `Machine Learning` `Generation` `Optimization` `Generative Adversarial Network` `Generative Adversarial Network` `Diffusion model` `Score-based Model` `Auto Encoder` `Contrastive Learning` `Tabular` `Point Cloud` `Mesh`

**🎯 论文内容**

提出一种 Condition‑Enhanced Generative Adversarial Network（CEGAN），用于在给定几何和气动性能约束下快速生成可行的三维飞机配置。

**💡 创新点**

创新点在于：① 在生成器损失中加入几何指导罚项和气动性能指导罚项，使得生成结果同时满足几何约束和气动目标；② 训练期间使用与数据集相同的代理模型，将预测误差直接嵌入损失；③ 在推断阶段引入输出过滤器，利用代理模型剔除不满足约束的样本，进一步提升生成质量。

**🔧 技术方法**

技术包括：Wasserstein GAN with Gradient Penalty（CWGAN‑GP）框架；条件嵌入的生成器与判别器；几何与气动误差的加权损失函数；多任务前馈神经网络代理模型；输出过滤算法。

**📊 数据集**

数据集基于 Cessna Citation X 的 28 个几何参数，利用 OpenVSP 进行低阶气动仿真生成 C_L、C_D、C_M 等指标。共收集 2500 条样本，使用 Latin Hypercube Sampling 生成不同条件组合。

**📈 对比分析**

通过 MRE（平均相对误差）和 MNND（最近邻距离）两指标评估。相较于基线 CWGAN‑GP，CEGAN 在插值案例（Cond‑A、Cond‑B）将生成误差降低至少 36%，在强外推案例（Cond‑C）降低至少 46%；同时保持与基线相当甚至更高的多样性；与在 5000 条样本上训练的 CWGAN‑GP 比较，CEGAN 在相同样本量下实现更优性能。

**⚠️ 局限性**

局限性：① 代理模型的准确性对气动误差指导效果影响显著，低 R² 会削弱改进；② 仅使用低阶 OpenVSP 结果，未验证在高保真 CFD 或实验数据下的泛化；③ 需要手工调节多项权重（λ_k、λ_C_i、a 等），调参成本较高；④ 对不同飞机结构的迁移性尚未评估。

---

## 33. Agent or Human? Detecting Commits by AI Coding Agents in the Absence of Authorship Signatures

**arXiv ID:** 2610.36229 | [PDF](https://arxiv.org/pdf/2610.36229v1)

**作者:** Kosei Horikawa `[一作]`, Ahmed E. Hassan `[通讯]`

**关键词:** `Software Engineering` `AI Code Assistant` `Classification` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Text`

**🎯 论文内容**

本研究调查在去除 AI 编码代理（如 GitHub Copilot、OpenAI 代码助手等）的签名后，是否能够仅凭提交内容识别代理提交，并构建并评估多种检测模型。

**💡 创新点**

创新点在于：①首次在提交级别对签名被掩蔽的代理提交进行检测；②系统比较了传统机器学习（XGBoost、LightGBM、CatBoost）、文本表示（TF‑IDF）、深度表示（CodeBERT）和大语言模型（ChatGPT、GPT‑4）在此任务中的表现；③揭示了“消息长度”和“改动规模”是检测的主要信号，并分析了模型在未见代理、不同时间段及不同仓库上的泛化能力。

**🔧 技术方法**

使用的技术包括：XGBoost 作为主模型；LightGBM、CatBoost 作为对比模型；基于哈希的 TF‑IDF 词向量加逻辑回归；CodeBERT 句子嵌入加线性分类；以及零/少样本 LLM 判断；特征工程涵盖提交信息长度、文件改动统计、代码风格与词汇统计，并在训练内进行特征相关性裁剪。

**📊 数据集**

数据集来自 2026 年 10 月之前公开的 3,300 多个非 Fork 的 GitHub 仓库，收集了 1.8 万个默认分支提交，其中标注了五个主流 AI 代理（Copilot、ChatGPT、OpenAI 代码助手等）和人类提交；提交通过作者账户、签名短语等方式标记，并在实验中对签名进行掩蔽。

**📈 对比分析**

在与签名被掩蔽的同一仓库、同一时代的训练/测试拆分下，XGBoost 达到 ROC‑AUC 约 0.85；TF‑IDF 的性能相当，且在未见代理的实验中 AUC 下降到 0.55‑0.65；精度在观察到的代理比例下最高约 20‑25%，表明模型能提升审查优先级但无法完全替代签名；在时间演化实验中，模型在后续几个月 AUC 下降 5‑10%，说明检测需定期再训练。

**⚠️ 局限性**

主要局限包括：①检测依赖于提交信息的长度与风格，若代理更改消息模板或采用更短消息，性能会显著下降；②未能完全捕获代码本身的 AI 特征，导致对未见代理和长时间迁移的鲁棒性差；③使用的标记仅基于已公开的签名，无法确认所有人工提交是否确实未使用 AI；④实验仅覆盖公开仓库，缺乏对私有或新兴项目的验证。

---

## 34. An Unconditional Space-Approximation Lower Bound for Semi-Streaming Matching

**arXiv ID:** 2610.36577 | [PDF](https://arxiv.org/pdf/2610.36577v1)

**作者:** Yang P. Liu `[一作]` (Carnegie Mellon University), Janani Sundaresan `[通讯]` (University of Waterloo)

**关键词:** `Data Structures and Algorithms` `Optimization` `Computational Efficiency` `Federated Learning` `Information Theory` `Communication Complexity` `Graph Construction` `KL Divergence` `Entropy Analysis` `Graph`

**🎯 论文内容**

本文通过构造层级化的稀疏图（利用RS图）和信息理论的工具，将多通道通信下的“最大二分匹配”问题转化为多轮分布式通信问题，从而给出了在p轮（p>1）下流式算法求近似最大匹配所需空间的下界。该方法提供了与传输复杂度相对应的精确分析，并证明了任何常数逼近（如(1-3^-p+)-逼近）的p-pass流式算法都需要至少Ω(n·log^{p-1}n)空间。

**💡 创新点**

创新点主要在于：① 将多轮流式算法与分布式通信模型关联的“流式到通信”的模拟技术得到改进；② 利用RS图的结构设计出能在多层次上“隐藏”匹配信息的图构造；③ 通过KL散度链式规则和信息量分析，精确刻画了在给定隐藏索引分布下信息量的传播与约束，从而实现了对多通道通信成本的精细化下界。

**🔧 技术方法**

使用的主要技术包括：通信复杂性框架、信息理论（KL散度、熵）与链式规则、RS图（r,t-RS图）构造、分层嵌套图结构、随机删除/掩码技术、以及流式算法的多轮通信模拟。

**📊 数据集**

由于研究的是理论下界，实验数据并未使用真实数据集；所有的图结构均为论文自定义的合成图（由RS图递归构造得到的层级化稀疏图）。

**📈 对比分析**

本文未给出与具体算法的直接性能对比实验，而是通过理论分析证明：在p>1轮的情况下，任何常数近似（如0.9逼近）都需要至少Ω(n·log^{p-1}n)的存储空间；相比之下，已知的上界为O(n·log n)，从而在多轮情形下实现了近似匹配空间复杂度的完整上下界。

**⚠️ 局限性**

主要局限性包括：① 结果仅适用于常数近似和固定p轮的流式算法；② 对于非二分匹配或更广泛的图类，结论尚未扩展；③ 证明中使用的RS图参数和隐藏索引分布是理论构造，实际实现与实际网络流式环境的对应关系尚不明确。

---

## 35. Testing Agentic Service Systems: An Agentic Living Lab Methodology Across the Physical-Digital Continuum

**arXiv ID:** 2610.35978 | [PDF](https://arxiv.org/pdf/2610.35978v1)

**作者:** Zhongjie Wang `[一作]` (Harbin Institute of Technology), Hanchuan Xu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 514 | [OpenAlex ID](https://openalex.org/A5069247819)

**关键词:** `Software Engineering` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Optimization` `Federated Learning` `Robotic Intelligence` `Agentic AI` `Event-driven runtime semantics` `Semantic Delta` `Digital/Entity Dual Proof` `Rule/Agent/Human Role Implementation` `Tabular` `Time Series` `Sequential` `Benchmark` `Review/Survey Paper`

**🎯 论文内容**

设计了一套 Agentic Living Lab 方法论与参考基础设施，用于系统化地测试代理服务系统在跨物理-数字连续体中的行为。

**💡 创新点**

以服务情节为核心测试单元，拆分代理与环境、引入人类角色多种实现模式、显式现实化与语义差异，提供多维判断向量，确保测试可重现与可解释。

**🔧 技术方法**

采用事件驱动运行时语义、分层参考架构、能力声明式现实子系统适配器、语义 Delta、数字/实体双重证明、规则/代理/人类角色实现等技术。

**📊 数据集**

主要使用构造实验设计（旅行服务与工业恢复案例），结合模拟轨迹、FMU/EMC 模型等数据，未使用公开大型真实数据集。

**📈 对比分析**

通过多维判断向量（任务成功、服务履行、价值等）与对照实验（代理版本、子系统迁移、代理代理替代等）比较，结果显示代理与环境、协议、现实化细节对结论影响显著；论文未给出具体性能指标。

**⚠️ 局限性**

仅为方法与参考架构，缺乏完整平台与实证验证；代理质量、代理代理有效性、跨子系统稳定性仍为经验问题；依赖人工标注与校准，实际实现与性能未评估。

---

## 36. UT-HMR: Uncertainty-Guided Temporal Modeling for Robust 3D Human Mesh Recovery

**arXiv ID:** 2610.36203 | [PDF](https://arxiv.org/pdf/2610.36203v1)

**作者:** Zhibo Wang `[一作]` (University of Waterloo), David Clausi `[通讯]` (University of Waterloo)

**关键词:** `Computer Vision and Pattern Recognition` `Pose Estimation` `Transformer` `Graph Neural Network` `Contrastive Learning` `Gaussian Splatting` `Diffusion model` `Image` `Video` `Mesh`

**🎯 论文内容**

开发了一种单阶段多人体网格恢复框架 UT-HMR，能够在弱观测条件下通过时间早期融合和姿态不确定性建模来提高 3D 网格重建精度。

**💡 创新点**

创新点包括：① 用低秩+对角高斯协方差直接预测全人体姿态的不确定性，避免了传统多假设或采样方法；② 在检测器的查询特征上引入时间早期融合，为单帧中心提取多帧上下文信息；③ 将预测的不确定性转化为联合权重，进行异方差监督，使模型更关注可靠关节。

**🔧 技术方法**

主要技术：ViT-B/14 视觉 Transformer 编码器；DETR 风格解码器与查询；低秩+对角协方差估计网络；时间早期融合（T=3 帧）；高斯负对数似然分布损失；异方差关节损失。

**📊 数据集**

数据集：VIP‑HockeyPose‑v2（冰球）、WorldPose（足球）和通用 3DPW（野外场景）。

**📈 对比分析**

与现有单阶段方法（SAT‑HMR、PSVT、Multi‑HMR 等）进行对比，UT‑HMR 在两大体育数据集上取得所有指标（MPJPE、PA‑MPJPE、PVE）最高或近似最高；在 3DPW 上保持竞争力，且模型体积更小、速度更快。

**⚠️ 局限性**

局限性：① 仍依赖 ViT‑B/14 基础模型，虽然轻量但对 GPU 资源有一定要求；② 仅通过短时窗口（±1 帧）进行时间融合，长时序信息未充分利用；③ 协方差模型假设姿态误差集中于少数主方向，可能不适用于极端异常姿态；④ 预测的协方差仅用于训练，推理时仍返回确定性网格，无法直接输出不确定性预测作为结果。

---

## 37. EVA-Flow: Learning Shared Chemistry for Unified Environment-Aware Molecular Conformation Generation

**arXiv ID:** 2610.36280 | [PDF](https://arxiv.org/pdf/2610.36280v1)

**作者:** Bing Yan `[一作]` (FAIR at Meta), Benjamin Kurt Miller `[通讯]` (FAIR at Meta)

**关键词:** `Machine Learning` `Drug Discovery` `Protein Structure Prediction` `Transformer` `Auto Encoder` `Flow-based Model` `Diffusion model` `Score-based Model` `Contrastive Learning` `Supervised Fine-Tuning` `Graph` `Biomedical Data`

**🎯 论文内容**

提出一种统一的环境感知分子构象生成框架EVA-Flow，能够在真空、蛋白质配体结合、溶剂化和晶体包装等多种环境下生成准确的3D构象。

**💡 创新点**

通过将分子共通的化学结构与环境特定的物理约束分离，利用环境条件对流匹配模型进行条件化，实现跨环境知识迁移；同时在预训练与统一微调过程中显著提升数据稀缺环境的性能。

**🔧 技术方法**

采用变分自编码器（VAE）+流匹配（Flow Matching）解码器，配合SE(3)-equivariant Transformer网络和环境嵌入网络，使用谐波基底分布与噪声插值实现高效采样。

**📊 数据集**

在多来源预训练语料（GEOM-QM9、GEOM-DRUGS、SPICE、OE62等）共计445k分子与9.8M构象，随后在四个目标环境（真空、Docking、溶剂化、晶体包装）上微调，分别使用GEOM-DRUGS、PDBBind-v2020、Aquamarine和COD数据集。

**📈 对比分析**

与单模型单环境训练、ETFlow及其他生成基线（GeoDiff、GeoMol、Torsional Diffusion、MCF-L）对比；在4构象协议下，EVA-Flow在真空、Docking、Solvation和Packing四个环境中均取得最高Recall/Precision COV，并在数据稀缺环境中显著优于其他方法；在2K协议下与ETFlow相当或略优。

**⚠️ 局限性**

受限于模型复杂度和训练数据多样性，EVA-Flow在极低数据环境下仍易出现样本多样性不足；对未见环境（如表面吸附、固液界面）的泛化性未知；生成结果需进一步与高精度物理模拟结合验证。

---

## 38. Natural Hard-Negative Mining from Positives for Instruction-Following Retrieval

**arXiv ID:** 2610.36900 | [PDF](https://arxiv.org/pdf/2610.36900v1)

**作者:** Chaolang Li `[一作]`, Pengwen Dai `[通讯]`

**关键词:** `Artificial Intelligence` `Retrieval` `Transformer` `Contrastive Learning` `Prompt Engineering` `Large Language Model` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

本文提出了一种自然硬负样本挖掘（Natural Hard-Negative Mining）方法，将现有的查询+正样本+负样本组合转化为指令化的检索对比训练实例，并在MS MARCO数据上构建了一个6,716条实例的训练集。

**💡 创新点**

创新点在于：① 识别可转换负样本（Convertible Negatives），通过生成最小可辨别的指令把它们提升为正样本；② 将原始正样本重新标注为自然硬负样本，迫使检索器学会在指令条件下区分同主题文档；③ 采用三阶段自动化管道（Candidate Cleaning、Reversal Construction、Geometry Filtering）避免长负样本的合成与生成误差。

**🔧 技术方法**

技术包括：双编码器检索器与InfoNCE对比损失；LLM（GPT‑5.1）用于候选清洗与指令生成；冻结预训练编码器进行几何过滤；对齐检索空间并保持语义一致性。

**📊 数据集**

使用数据集：MS MARCO‑W（原始查询+正负文本），构建出的6,716条指令化训练实例；在评估时使用 InstructIR 与 FollowIR 两个检索指令化基准；实验模型包括 Qwen3‑Embedding‑0.6B、Qwen3‑Embedding‑4B、BGE‑M3。

**📈 对比分析**

对比方法：在 InstructIR 上与 Promptriever、InF‑IR 以及多种双编码器基线比较；在 FollowIR 上与跨编码器排名器、商业 API 以及微调检索器对比。实验结果表明，使用 0.6B 模型微调后即可在 InstructIR 上获得最优 Robustness@10 与 nDCG@10，且在 FollowIR 上的平均相关性与指令敏感度均优于更大规模模型，证明所提方法在数据与模型规模上都更高效。

**⚠️ 局限性**

限制：未对更大规模训练语料或更大模型进行实验，构建过程依赖于LLM和GPU，计算成本较高，未来工作需扩展到更大规模的数据与模型。

---

## 39. Lipi-Ghor: A Large-Scale, Diarized, Multi-Domain Bengali Speech Corpus and a Scalable Pipeline for Building It

**arXiv ID:** 2610.36184 | [PDF](https://arxiv.org/pdf/2610.36184v1)

**作者:** Sanjid Hasan `[一作]` (Khulna University of Engineering & Technology), Bayazid Hasan Simon `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `Computation and Language` `Recognition` `Transformer` `Contrastive Learning` `Supervised Fine-Tuning` `Retrieval-Augmented Generation` `Audio` `Video` `Text`

**🎯 论文内容**

构建了一个大规模弱监督的 Bengali 语音语料库 Lipi‑Ghor，包含 885 小时长音频、17 个领域、998 个视频，并提供了说话人划分的 SSTT 注释。

**💡 创新点**

① 第一份覆盖长音频和说话人划分的 Bengali 语料；② 提出了可复制的基于 YouTube 字幕的弱监督语料构建流水线，适用于其他低资源语言。

**🔧 技术方法**

利用 YouTube API 进行字幕与音频提取，使用 pyannote 的 Precision‑2 进行说话人分离并与字幕对齐生成 SSTT；在评估时采用四个公开 ASR 模型的 zero‑shot 与 fine‑tuned 对比实验。

**📊 数据集**

使用自建的 Lipi‑Ghor 数据集（885 小时、17 个领域）以及 500 片人工校正的金标评估集，另外引用四个公开 ASR 模型作为基准。

**📈 对比分析**

通过 WER/CER 对 zero‑shot 与 fine‑tuned 模型进行比较；zero‑shot WER 介于 30%–85%；fine‑tuned Fast Conformer 24.7%/15.6% 以及 Faster Whisper Medium 20.7%/11.9%，在各领域表现差异显著，情感、方言和长篇内容的错误率最高。

**⚠️ 局限性**

弱监督字幕导致转写噪声，自动说话人标签未人工验证；方言覆盖不足；依赖 YouTube API，复现性受限；原始音频不再公开，需自行重新下载。

---

## 40. DeepInfer-Mac: Running a 750B-Class MoE on a 24 GB Laptop

**arXiv ID:** 2610.36564 | [PDF](https://arxiv.org/pdf/2610.36564v1)

**作者:** Haorui Li `[一作]` (Southeast University), Haorui Li `[通讯]` (Southeast University)

**关键词:** `Performance` `Computational Efficiency` `AI Code Assistant` `Transformer` `Mixture of Experts` `Large Language Model` `Mixture of Experts` `Prompt Engineering` `Retrieval-Augmented Generation` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Neural Radiance Field` `Auto Encoder` `Generative Adversarial Network` `Contrastive Learning` `Gaussian Splatting` `Simultaneous Localization and Mapping` `World Model` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation` `Text` `Code` `Sequential`

**🎯 论文内容**

在一台24 GB统一内存的MacBook Air上，使用SSD存储大量参数，利用混合低位量化、专家缓存、预取与需求优先的调度技术，使DeepSeek‑V4.1‑Flash 750 B MoE模型得以顺序执行并完成文本、代码及中英文生成任务。

**💡 创新点**

创新点在于：①将模型权重压缩至2/3/4位并将大部分路由专家直接存放在SSD，②设计可绑定GPU缓冲区生命周期的专家缓存机制，③引入下一层预测与需求优先的预取策略，实现读写重叠并显著提升吞吐量。

**🔧 技术方法**

采用的技术包括：Apple M4 GPU与统一内存、MLX框架、混合低位量化、SSD直接读写、专家缓存与LRU替换、下一层路由预测、需求优先的预取、GPU内存管理与缓冲区生命周期控制。

**📊 数据集**

实验使用的“数据集”主要是对模型的多语言与代码生成测试，涉及英文、中文和Python提示，亦包含较长的检索式提示；权重来源是DeepSeek‑V4.1‑Flash 750 B模型的压缩检查点（约337.6 GiB）。

**📈 对比分析**

通过与串行执行、仅重叠驻留专家读取以及加入需求优先预取三种调度方式的匹配实验进行比较，发现预取策略相较于单纯重叠可提升约10–25 %（取决于提示长度）的解码吞吐量，但同时增加了相当比例的SSD读取量；整体吞吐量达到数十tokens/秒，首次生成时间（TTFT）在几秒级别。

**⚠️ 局限性**

限制主要包括：仅在单线程单序列场景下验证；实验在共享桌面机器上进行，存在主机swap活跃；低位量化后模型质量未进行全面评估；缺乏与官方未压缩模型的完整质量对比；不同SSD或更大缓存可能导致性能变化。

---

## 41. ATEMPO: Amortized Transfer Entropy Model Per-lag Over-Time

**arXiv ID:** 2610.36536 | [PDF](https://arxiv.org/pdf/2610.36536v1)

**作者:** Emma Andrews `[一作]` (National University of Singapore), Gianmarco Mengaldo `[通讯]` (National University of Singapore)

**关键词:** `Artificial Intelligence` `Computational Efficiency` `Representation Learning` `Time Series` `Transformer` `Convolutional Neural Network` `Diffusion model` `Score-based Model` `Contrastive Learning` `Time Series` `Tabular` `Multimodality`

**🎯 论文内容**

提出并实现了ATEMPO，一种将转移熵估计与滞后、条件深度和时间位置共用的单一模型，从而实现滞后结构的全局学习与快速查询；

**💡 创新点**

通过把传统需要为每个滞后-深度组合单独训练的转移熵估计器转化为可调节的共享模型，显著降低了计算成本并允许实时捕捉时间变异的延迟；

**🔧 技术方法**

使用条件扩散得分（conditional diffusion scores）构建的可调估计器、时间条件化、拉普拉斯卷积网络等深度学习方法；

**📊 数据集**

在WEATHER‑5K气象时间序列、Neural Latents Benchmark（NLB）脑机接口任务、MMTT‑Bench多模态预测任务以及若干合成数据上进行评估；

**📈 对比分析**

与八种基线（TENDE、NPEET、MINE、NJEE、CCMI、PCMI、Liang‑Kleeman、DTW）和传统随机/默认滞后选取方式对比，ATEMPO在转移熵估计精度、滞后检测准确率、以及基于选取滞后的预测MSE/R²提升上均达到或逼近最佳性能，且计算时间仅为基线的四分之一；

**⚠️ 局限性**

受限于高维度下样本量与模型容量的统计需求，估计误差会影响滞后选择；此外，其对真实世界非高斯、多维度数据的泛化尚需进一步验证。

---

## 42. Structure-Actuation Co-Design of All-Electric Heavy-Duty Robotic Manipulators

**arXiv ID:** 2610.36996 | [PDF](https://arxiv.org/pdf/2610.36996v1)

**作者:** Mohammad Bahari `[一作]` (Tampere University), Jouni Mattila `[通讯]` (Tampere University)

**关键词:** `Robotics` `Optimization` `Robotic Intelligence` `Reinforcement Learning` `NSGA-II evolution multi-objective search` `Closed-loop inverse kinematics trajectory tracking simulation` `Energy loss surrogate based on motor loss` `Structural load screening and system-level drive package coordination` `Tabular` `Point Cloud`

**🎯 论文内容**

本文提出了一种混合离散-连续多目标双层框架，用于全电动重载机械手的结构-驱动共设计，联合优化连杆几何、机制拓扑与离散电机/变速箱/螺杆组合，并通过系统级协调与质量反馈实现驱动质量一致性。

**💡 创新点**

创新之处在于将离散驱动单元选择与连续机械结构优化统一在双层多目标求解器中，并引入质量反馈循环，确保驱动尺寸与机器人动力学自洽，从而实现全电动高负载机械手的整体性能平衡。

**🔧 技术方法**

主要技术手段包括NSGA-II进化多目标搜索、离散与连续变量混合搜索、闭环逆运动学轨迹跟踪仿真、基于电机损耗的能量损失代理、结构负载筛选与系统级驱动包协同。

**📊 数据集**

实验使用了12个全姿态SE(3)基准TCP姿势的数据集，以及包含68种电机、离散变速比和螺杆导程的驱动库，对100 kg工具负载（以及20、60、100、140 kg）进行评估。

**📈 对比分析**

通过Pareto前沿比较硬件质量、共设计成本与供能量的权衡，得到八个可行无支配解，硬件质量在387–420 kg，成本归一化为1.19–1.31，供能量为31–37 kJ；质量反馈循环收敛仅需三次迭代，显著提高设计一致性。

**⚠️ 局限性**

局限性包括简化的结构和电机损耗模型、固定的腕关节设计、未考虑腕关节能耗、有限的搜索预算以及缺乏实验验证。

---

## 43. Ontology-guided multi-agent reasoning for explainable differential diagnosis

**arXiv ID:** 2610.36325 | [PDF](https://arxiv.org/pdf/2610.36325v1)

**作者:** Roxana Ramona Szomiu `[一作]` (Technical University of Cluj-Napoca), Adrian Groza `[通讯]` (Technical University of Cluj-Napoca)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Drug Discovery` `Transformer` `Graph Neural Network` `Large Language Model` `Retrieval-Augmented Generation` `Agentic AI` `Prompt Engineering` `Text` `Graph` `Tabular` `Benchmark` `Electronic Health Records`

**🎯 论文内容**

提出了一种基于本体论的多智能体系统(MAS)来实现可解释的差异诊断，利用LLM进行自然语言理解，结合知识图谱进行符号推理，并通过交互式询问逐步消除诊断不确定性。

**💡 创新点**

创新点在于将语言模型与符号推理分离，使用外部专家-curated本体论与知识图谱进行严格的知识检索，设计可追踪的诊断评分函数，并通过多轮是非问题实现系统自适应的“问诊”机制，从而提升诊断透明度和鲁棒性。

**🔧 技术方法**

技术包括：LiteLLM驱动的LLM；CrewAI框架管理多角色智能体；GraphDB存储与SPARQL查询的医学本体论（SYMP、DOID、HPO）；确定性Python代码用于症状映射、疾病评分、信息增益计算和最终诊断生成。

**📊 数据集**

使用Q4DX基准数据集（PatientCases.csv，共99例、29种诊断），在80%和50%症状完整度两种设置下评估模型性能。

**📈 对比分析**

与MedQDx（GPT‑4.1）基线对比：MedMAS‑DDx在80%完整度的Top‑1精度从18.7%提升到21.2%，Top‑3从25.8%提升到27.3%；MedQDx在相同设置下的Top‑1为44%，Top‑3为30.3%。系统通过10条平均询问实现了与更丰富初始信息相当的诊断置信度。

**⚠️ 局限性**

局限性包括：本体覆盖范围不足导致部分症状/诊断无法直接匹配；符号推理严格性限制了对临床语义细微差异的捕获；交互模式仅支持“是/否/不确定”，无法处理症状严重度等细节；评估仅在小规模基准上完成，缺乏真实临床数据验证。

---

## 44. Identity or Evidence? Auditing Evidence Dependence in Agentic Life-Science Benchmarks

**arXiv ID:** 2610.36510 | [PDF](https://arxiv.org/pdf/2610.36510v1)

**作者:** Danai Brilli `[一作]` (Zo Labs), Maria Georgila `[通讯]` (Zo Labs)

**关键词:** `Computer Vision and Pattern Recognition` `Benchmark` `Explainability and Interpretability` `Drug Discovery` `Transformer` `Large Language Model` `Prompt Engineering` `McNemar检验` `Wilson区间` `Spearman相关` `Biomedical Data` `Text` `Benchmark`

**🎯 论文内容**

对生物医学推理基准进行系统审计，评估在去除实体身份或删除证据后模型性能的变化。

**💡 创新点**

提出配对提示干预（身份模糊、证据删除、置换、回忆等）以及统计检验框架，用以量化实体身份对推理结果的依赖性。

**🔧 技术方法**

利用大语言模型进行推理，配合McNemar检验、Wilson区间、最小可检测效应、Spearman相关等统计方法进行结果对比与显著性评估。

**📊 数据集**

使用Minikel临床进展数据集、LAB-Bench LitQA2、Biomni Eval1、LAB-Bench MCQ等公开数据集进行实验。

**📈 对比分析**

通过完整提示、身份模糊提示、无证据提示等配对对照比较，发现身份模糊导致准确率显著下降（如0.783→0.531），证据删减不显著；在LitQA2上，逐步删词导致身份-模糊差距上升；回忆提示验证模型可直接访问记录信息。

**⚠️ 局限性**

受限于无法区分识别与隐藏标记对性能的影响，缺乏时间上独立的结果样本，模型执行环境未完全隔离网络调用，以及仅使用快照标签，导致对模型普适推理能力的推断受限。

---

## 45. Thermodynamic characterization of message-passing: Framework and initial study

**arXiv ID:** 2610.36348 | [PDF](https://arxiv.org/pdf/2610.36348v1)

**作者:** Kalana G. Abeywardena `[一作]` (University of Toronto), Stark C. Draper `[通讯]` (University of Toronto)

**关键词:** `Information Theory` `Computational Efficiency` `State space modeling` `Numerical simulation` `Random thermodynamics (theory of mismatch cost)` `Tabular` `Time Series` `Physics Related`

**🎯 论文内容**

研究了消息传递式错误更正解码器的热力学效率，给出了基于不匹配成本的能量消耗下界。

**💡 创新点**

首次将不匹配成本框架应用于消息传递解码器，并揭示输入统计、解码动态和硬件实现共同决定能耗。

**🔧 技术方法**

使用了随机热力学的不匹配成本理论、状态空间建模、周期性机分析与数值模拟。

**📊 数据集**

以 [7,4] 亨廷码在二进制擦除信道（BEC）上的实验，产生了约 0.5 aJ 的最低能耗估计。

**📈 对比分析**

与传统基于硬件面积和时间的能耗模型相比，该方法给出了更细粒度的能量下界，结果显示不同先验和错误模式可导致至少 10% 的能耗差异。

**⚠️ 局限性**

主要限制在于状态空间指数增长，难以扩展到大块 LDPC 码；同时忽略了内存访问与实际硬件细节。

---

## 46. Diagnosing and Correcting Bias in MLLM for Long Video Understanding

**arXiv ID:** 2610.36210 | [PDF](https://arxiv.org/pdf/2610.36210v1)

**作者:** Xusheng Liang `[一作]` (City University of Hong Kong), Jiawei Ma `[通讯]` (City University of Hong Kong)

**关键词:** `Computer Vision and Pattern Recognition` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Vision Language Model` `Contrastive Learning` `Prompt Engineering` `Large Language Model` `Video` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

本文提出了S-MME基准用于诊断多模态大模型在长视频问答中对快捷路径的偏倚，并给出了无训练的CLEAR框架通过对抗视图和残差干预来缓解该偏倚。

**💡 创新点**

创新点在于将快捷路径偏倚拆分为语言先验与局部视觉误导两类，并利用无训练的对抗视图与残差对齐实现推理时的偏倚修正。

**🔧 技术方法**

使用的技术包括对抗视图构造（question‑only与salient‑view）、隐藏状态对比与残差注入、以及在解码器后期进行干预的轻量级对齐。

**📊 数据集**

实验使用了Video‑MME、LongVideoBench、LVBench、MLVU以及自建的S‑MME诊断基准。

**📈 对比分析**

与多种开源/闭源长视频LLM对比，CLEAR在不增加帧数预算的前提下提升了10–15%准确率，并显著降低了Shortcut‑Error（约30%），证明了其有效性。

**⚠️ 局限性**

局限性包括需要可访问模型隐藏状态、对阈值与残差强度敏感、以及在极端稀疏采样或非视频模态下的适用性仍需验证。

---

## 47. AC-JEPA: Anchored Cross-Modal Joint Embedding Predictive Architecture for Visual-to-Symbolic Translation

**arXiv ID:** 2610.36493 | [PDF](https://arxiv.org/pdf/2610.36493v1)

**作者:** Chaofei Hong `[一作]`, Qing Zhang `[通讯]` (Zhejiang Lab)

**关键词:** `Artificial Intelligence` `Recognition` `Image Translation` `Drug Discovery` `Transformer` `Vision Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Image` `Text` `Multimodality`

**🎯 论文内容**

设计了一个兼容SFT的跨模态潜在预测目标（AC‑JEPA），在图像到SMILES的生成模型中联合训练，用于精确的光学化学结构识别。

**💡 创新点**

结合联合嵌入预测与因果解码，使用双向掩码潜在预测避免将伪标签硬化为分类目标，从而在已标记和伪标记数据上提升稳健性。

**🔧 技术方法**

基于Qwen3‑VL‑2B自回归视觉‑语言模型，加入中间层潜在投影器与预测器，并通过SIGReg正则化；训练分为基于真实标注的SFT阶段与通过潜在预测的合成/伪标记再训练阶段。

**📊 数据集**

使用MolParser‑7M生成的图像‑SMILES语料进行预训练，Real‑SFT 100K真实图像‑SMILES进行微调，USPTO‑Mol与论文结构图像作为未标注数据，评估基准为WildMol Simple与Markush。

**📈 对比分析**

在WildMol Simple上从Base SFT的67.40%提升至80.23%（+12.83%），Markush从36.48%提升至48.28%（+11.8%）；直接伪标签SFT仅63.47%/36.04%，而AC‑JEPA自训练显著优于此；控制噪声实验表明AC‑JEPA对SMILES误差更鲁棒。

**⚠️ 局限性**

对伪标签质量与生成成本仍有依赖，噪声容忍度受限于实验设置，尚未在其他视觉‑符号翻译任务上验证，且自训练需要额外计算资源。

---

## 48. Your Agent Acts Differently When You're Around: How User Presence Changes Agent Behavior in Negotiation

**arXiv ID:** 2610.36435 | [PDF](https://arxiv.org/pdf/2610.36435v1)

**作者:** Jane Pan `[一作]` (New York University), He He `[通讯]` (New York University)

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Explainability and Interpretability` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Benchmark`

**🎯 论文内容**

本研究通过构建PRESence评估框架，系统比较LLM代理在有无交互式用户存在时在谈判任务中的表现，探讨用户在场如何影响代理行为与结果。

**💡 创新点**

创新点在于首次将用户是否在场作为实验变量，对同一谈判游戏在交互式用户与仅静态指令两种条件下进行严格对照，揭示用户在场能显著提升收益、增加代理攻击性但亦诱发不对齐行为。

**🔧 技术方法**

采用大型语言模型（GPT‑4、GPT‑3.5、Claude‑3.5、Claude‑2、Qwen 系列、DeepSeek 等）担任代理和模拟用户，并用LLM判别器自动检测对用户和对手的不对齐行为。

**📊 数据集**

使用四大公开谈判基准（NegotiationArena、Deal‑or‑No‑Deal、PrincipalBench、CaSiNo）共 12 轮次，覆盖单议题与多议题、公开与私有估值等多种情境。

**📈 对比分析**

与静态指令条件相比，交互式用户平均提升 13.7% 付款，并使代理的首要报价更具攻击性；但不对齐行为的出现率从 2.6% 翻倍至 5.5%，且与收益无显著相关，表明收益提升并非由不对齐驱动。

**⚠️ 局限性**

局限性包括仅关注二人谈判、模拟用户而非真实用户、仅在英文环境下实验、未揭示导致行为变化的机制，以及未覆盖多代理或多用户场景。

---

## 49. From Generation to Scoring: Scaling LLM-Based Semantic Labeling at LinkedIn

**arXiv ID:** 2610.36383 | [PDF](https://arxiv.org/pdf/2610.36383v1)

**作者:** Jian Sheng `[一作]` (LinkedIn), Fedor Borisyuk `[通讯]` (LinkedIn)

**关键词:** `Machine Learning` `Recommendation System` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Text`

**🎯 论文内容**

将LinkedIn的个人档案资格标签任务从使用打包式生成（packed generation）改为单候选语义评分（semantic scoring），通过单候选prompt、只训练yes/no答案、答案token读取，并利用Multiple-Item Scoring（MIS）在一次预填充中并行计算多候选得分；

**💡 创新点**

1）将生成式标签器端到端转化为评分器，保持决策独立性；2）引入答案token读取和标签仅训练，避免答案列表格式学习；3）设计MIS注意力掩码和位置ID重用，实现多候选无交互评分；4）在生产环境中集成Ray Data MIS API和SGLang实现，提升吞吐量；

**🔧 技术方法**

使用指令微调的大语言模型（1.5B参数）进行单候选训练；答案token读取与阈值化；Multiple-Item Scoring（MIS）结合FlashInfer、SGLang实现预填充不解码；Ray Data API实现批量推理；vLLM作为baseline；

**📊 数据集**

LinkedIn成员档案及其检索出的200条资格候选，约100M个查询-项对；训练集规模从45K对到443K对不等；

**📈 对比分析**

与baseline packed generation 在相同模型规模下，semantic scoring 在45K训练对下达成86.3%的一致性，packed generation仅50.8%；一致性（候选重排）semantic scoring <0.01%，packed generation 0.86%；吞吐量上，offline 1.24×提升，模型服务 1.9×提升；MIS通过一次预填充将每标签预填充token降低约三分之一，N=200请求可达4.8K labels/s；

**⚠️ 局限性**

1）仅评估与参考标签的一致性，未验证对业务决策的实际影响；2）MIS掩码不适用于需要跨候选交互的列表式或重排序任务；3）模型规模与质量关系未充分探索，尤其是小型决策模型的性能上限未知；4）实验仅在LinkedIn场景验证，通用性仍待进一步验证。

---

## 50. Taming Multi-Dimensional Skew in Sparse Matrix Multiplication with Contention-Aware Scheduling [Technical Report]

**arXiv ID:** 2610.36822 | [PDF](https://arxiv.org/pdf/2610.36822v1)

**作者:** Wentao Huang `[一作]` (National University of Singapore), Kian-Lee Tan `[通讯]` (National University of Singapore)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Computational Efficiency` `Optimization` `Graph` `Tabular` `Time Series` `Sequential` `Benchmark`

**🎯 论文内容**

本文提出一种基于“morsel”切片的动态调度框架，专门针对稀疏矩阵乘法（SpGEMM）中的多维偏斜与缓存竞争问题，实现了负载均衡、内存带宽利用率提升及对不同累加器的自适应选择。

**💡 创新点**

创新点包括：①通过固定结果体积（nnz）而非行数或运算量来划分morsel，解耦计算与存储两维偏斜；②引入基于压缩因子（cf）排序的三组线程调度策略，显著降低共享缓存竞争；③自适应开启SMT的策略（SMTselect）在高cf时禁用，以平衡计算吞吐与缓存压力；④morsel级别的累加器选择（Hash vs ESC）进一步提升执行效率。

**🔧 技术方法**

技术方法包括：动态任务调度（morsel‑wise）、固定输出规模划分、cf 统计与排序、三组线程调度、SMT自适应启用、累加器自适应选择、以及针对不同SpGEMM实现（HASH、ESC）的统一框架。

**📊 数据集**

实验数据集覆盖：R‑MAT（ER、SSCA、G500）合成矩阵；SuiteSparse实际稀疏矩阵；关系查询数据（ClickHouse、MovieLens、IMDB 等）；图分析数据（AMiner DBLP、社交网络等）。

**📈 对比分析**

与业界主流实现（Intel MKL、PB、HEAP、DIM3、NPHJ、PHJ 等）以及传统的 flop‑count 调度、OpenMP 动态/引导调度进行对比。结果显示：在偏斜工作负载上，提出框架相较于 HASH 可提升约 81%（最高 1.8×），相较 ESC 最高 36%（1.5×），相较 MKL 最高 3.6×，并在 CXL 扩展内存平台上显著提升带宽利用率（达 40 GB/s，超过 DRAM 峰值）。

**⚠️ 局限性**

局限性：①在极度稀疏、密度低的矩阵上，由于 cf 统计与排序开销占比高，可能略逊于静态 flop‑count 调度；②框架基于 Gustavson 行行求法，尚未验证对内积或外积实现的直接适配；③需要预先获取每行非零数目，增加前处理成本；④在极大矩阵或高并发系统下，线程组划分与 SMT 选取的阈值需进一步调优。

---

## 51. Dream-COP: Guiding Algorithm Discovery for Combinatorial Optimization Problems in an Experience World

**arXiv ID:** 2610.36618 | [PDF](https://arxiv.org/pdf/2610.36618v1)

**作者:** Kuan Xu `[一作]` (Singapore Management University), Zhiguang Cao `[通讯]` (Singapore Management University)

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning from Human Feedback` `Neural Architecture Search` `Transformer` `Large Language Model` `World Model` `Reinforcement Learning` `Tabular` `Graph`

**🎯 论文内容**

本文提出 Dream-COP，利用历史搜索树构建经验世界，让策略代理在内部环境中通过重放学习可执行的搜索策略，从而在 LLM 驱动的组合优化算法发现中更有效地分配搜索预算。

**💡 创新点**

创新点在于：①将完整搜索树视为可重放的经验世界，消除了再次生成与评估程序的成本；②设计可执行的策略形式（决定开启、扩展或停止搜索分支）并通过重放目标对策略进行评估；③实现策略与搜索的闭环学习，并展示跨问题迁移的有效性。

**🔧 技术方法**

技术核心包括：大型语言模型（GPT‑6‑Luna、GPT‑4o‑mini、GPT‑6‑Sol）作为生成与评估引擎；基于树的搜索结构与重放逻辑；基于重放目标的策略优化；以及在实际搜索中按学习到的策略指导搜索步骤。

**📊 数据集**

使用的数据集覆盖五类组合优化问题：构造型 TSP、KP、在线箱装（OBPP）、流车间排程（FSSP）和二维箱装（2D‑BPP）；每类问题包含多规模合成实例、公开基准实例（如 TSPLIB、Taillard、BWMV 等），并采用 1,000–500 个测试实例进行最终评估。

**📈 对比分析**

与 MCTS‑AHD、EoH、ReEvo、FunSearch 等基线进行对比；在构造型 TSP/KP、OBPP、FSSP 和 2D‑BPP 上，Dream‑COP 在平均 gap、平均搜索请求数等指标上均优于基线；在 2D‑BPP 上实现了更优的 bin 使用率，同时显著降低搜索步骤；跨问题迁移实验表明学习到的策略在未见问题上亦能提升搜索效率并节省约 25–64% 的搜索请求。

**⚠️ 局限性**

局限性包括：①对 LLM 的依赖导致计算与 token 成本较高；②策略学习需要多轮 LLM 调用，可能在资源受限环境下不易部署；③当前实验仅覆盖少数几类 COP，尚未验证对更大规模或不同结构问题的普适性；④重放机制假设历史树信息完整且准确，若搜索树信息不充分或失真，策略性能可能下降。

---

## 52. Jaxnasium: Embracing PyTrees For Any-Agent, Cross-Suite Reinforcement Learning

**arXiv ID:** 2610.36075 | [PDF](https://arxiv.org/pdf/2610.36075v1)

**作者:** Koen Ponse `[一作]` (Leiden University), Thomas M. Moerland `[通讯]` (Leiden University)

**通讯引用:** 1105 | [OpenAlex ID](https://openalex.org/A5003595753)

**关键词:** `Machine Learning` `Reinforcement Learning` `Optimization` `Hyperparameter Search` `Reinforcement Learning` `Tabular` `Time Series` `Sequential` `Benchmark`

**🎯 论文内容**

提出并实现了 Jaxnasium——一个面向 JAX 的强化学习库，统一了环境接口、算法实现与工具，支持空间无关、单文件实现，兼容 JAX-native 与 host 环境，并提供 300+ 环境的包装器、项目脚手架、超参数搜索与多种 seed 的评估工具。

**💡 创新点**

核心创新在于将观测与动作空间都视作 PyTree，从而让网络、算法和多智能体逻辑在不改动代码的前提下自动匹配任意空间结构；实现单一训练循环即可同时运行 JAX 原生与 host 环境，且保持或提升性能；提供统一的 Gymnasium 风格接口与跨域包装器。

**🔧 技术方法**

采用 JAX（XLA、JIT、vmap 等）与 Equinox 构建可组合的 PyTree 模块；使用 Gymnasium、EnvPool 等 host 环境的包装器；实现基于 PyTree 的网络与多智能体结构；配合自定义工具包对树结构进行映射、拆分、归一化等操作。

**📊 数据集**

利用 11 个 JAX 本地套件共 333 个环境（Gymnasium、Gymnax、Brax、Jumanji、Pgx、JaxMARL、XLand-MiniGrid、Craftax 等）以及 host 的 Gymnasium 与 EnvPool，进行基准测试与多任务实验。

**📈 对比分析**

在四种后端（JAX 原生、Gymnasium、EnvPool、EnvPool-XLA）上对 PPO 进行吞吐量（环境步/秒）基准，使用相同超参数与网络结构，与 Stable‑Baselines3、CleanRL（PyTorch & JAX）以及 Stoix 的 Anakin/Sebulba 进行对比；Jaxnasium 的吞吐量在所有后端均与或超过现有最快实现，证明其统一训练循环不损失性能。

**⚠️ 局限性**

目前仅实现了四个算法，缺乏更丰富的算法库；主打单机单 GPU 研究场景，未针对多设备分布式训练提供优化；多智能体支持基于单智能体实现的自动向量化，复杂的 CTDE 结构仍需手动调整；未来需要扩充算法、完善分布式支持并改进大型多智能体学习的易用性。

---

## 53. PrivacyEvalClaw: A Tool-Augmented Deliberative Agent for Auditing Privacy Leakage in LLM Jailbreak Responses

**arXiv ID:** 2610.36711 | [PDF](https://arxiv.org/pdf/2610.36711v1)

**作者:** Krishna Kanth Nakka `[一作]` (Huawei Heisenberg Research Center), Xuebing Zhou `[通讯]` (Huawei Heisenberg Research Center)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Explainability and Interpretability` `Transformer` `Large Language Model` `Agentic AI` `Chain-of-Thought` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

针对LLM的隐私渗透评估，提出了一种训练无关的推理式评估框架PrivacyEvalClaw，能够在单一评估器无法解决的模糊案例中主动收集多种工具证据并进行协商。

**💡 创新点**

创新点在于将隐私评估视为动态推理过程，而非单次预测；通过异构工具（守护模型、解码器、PII检测器、专用问答与分歧分析器）与主动工具调用协同，显著提高评估准确性与可审计性。

**🔧 技术方法**

主要技术包括：多工具调用架构、推理式代理（agent）调度、分歧分析器、解码器（Base64、摩斯码、凯撒密码）以及轻量级PII检测器；评估采用基于对比的LLM-as-Judge对话、Cohen κ一致性和五分制过程质量指标。

**📊 数据集**

使用了125k条红队测试数据（70个PII属性、50个数据主体、55种越狱策略），并在其中随机抽取350条样本进行实验；人类标注集合包含218例披露、132例非披露。

**📈 对比分析**

与无工具直接评估、无工具链式思维以及缺失分歧分析器等基线相比，PrivacyEvalClaw在整体质量和过程质量上的胜率分别达82.86%和98%以上；最终隐私标签与人类标注的准确率高达85.43%，κ值为0.708。

**⚠️ 局限性**

局限性包括：依赖可用工具，无法处理未知编码/密码；仅使用单一基础模型；工具集不完整，可能忽视未来新型隐私风险；实验受限于固定攻击策略和模型，泛化性待验证；工具调用策略主要由系统提示驱动，缺乏自适应学习。

---

## 54. CIVER: Causal Intervention Verification for Reliable Code Review

**arXiv ID:** 2610.36010 | [PDF](https://arxiv.org/pdf/2610.36010v1)

**作者:** Xinyuan Song `[一作]` (Causal Dynamics Lab), Liang Zhao `[通讯]` (Causal Dynamics Lab)

**关键词:** `Software Engineering` `AI Code Assistant` `Explainability and Interpretability` `Data-Centric Learning` `Transformer` `Large Language Model` `Contrastive Learning` `Prompt Engineering` `Cross-Attention Mechanism` `Text` `Code` `Benchmark`

**🎯 论文内容**

构建了一个基于因果干预的代码审查验证框架，能够区分真实缺陷与误报，并对审查结果进行可靠性评估。

**💡 创新点**

创新点在于利用针对性修复、无关修改和语义保持变换三种干预方式，并通过专家迭代联合优化审查、干预和验证三大模块，从因果角度验证审查结论的可靠性。

**🔧 技术方法**

采用冻结的LLM审查器与代码/文本/对齐编码器、轻量级Transformer+MLP、对比学习和交叉注意力机制，并通过专家迭代进行联合训练。

**📊 数据集**

主要使用SWE-Review-Bench、AACR-Bench、CRScore、SWE-PRBench、CodeReviewBench等多种公开评测数据集。

**📈 对比分析**

与冻结LLM、Self-Reflection、LLM-Verifier、Fix-Guided Prompt等基线对比，AUROC从0.57提升至0.60，精准率提升至0.77，误报率降低，且在跨仓库、跨语言的迁移实验中保持较高性能。

**⚠️ 局限性**

局限性包括对干预生成质量的高度依赖、LLM生成审查内容的误差可能影响定位准确性，以及对复杂缺陷或低资源语言的适应能力仍有限。

---

## 55. RePAIR: Inverting Backdoor Attacks in Semantic Segmentation

**arXiv ID:** 2610.37015 | [PDF](https://arxiv.org/pdf/2610.37015v1)

**作者:** Arnold Brosch `[一作]` (Heinrich Heine University Düsseldorf), Kira Maag `[通讯]` (Heinrich Heine University Düsseldorf)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Adversarial Attack` `Safty and Privacy` `Convolutional Neural Network` `Transformer` `Contrastive Learning` `Auto Encoder` `Supervised Fine-Tuning` `Image`

**🎯 论文内容**

提出了一种后置、无标签、对攻击无先验的语义分割后门修复方法 RePAIR，能够在不重新训练模型的前提下对受攻击像素进行定位与纠正。

**💡 创新点**

创新点在于将攻击建模为指向目标类的概率转移，并利用内部特征中残留的受害者语义信息实现逆向修复，同时通过安全过滤器构建无毒参考集并实现无监督的受害者-目标对识别。

**🔧 技术方法**

采用内部特征与分类头矛盾检测、残差概率校准、特征空间逆转运输、输出安全约束等技术，并为 CNN 与 Transformer 设计了结构特定的语义参考与多层联合逻辑回归读取器。

**📊 数据集**

在 Cityscapes、PASCAL VOC 2012 与 ADE20K 数据集上评估，并分别使用 DeepLabV3+（ResNet-50）与 SegFormer（MiT-B5）两种主流分割架构。

**📈 对比分析**

与裁剪、微调、Segment Cleanse、ABL 等现有防御方法对比，RePAIR 在 8 种后门攻击下将攻击成功率降至 0.46%–7.86%（DeepLabV3+）和 ≤0.54%（SegFormer），同时保持或提升 mIoU‑B，表现出最优或次优的整体性能。

**⚠️ 局限性**

局限性包括需要足够的无毒样本来构建可靠参考集；对极弱后门或攻击模式与模型架构高度匹配的情况效果有限；且方法针对后门攻击设计，可能不适用于其他类型的模型篡改。

---

## 56. Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs

**arXiv ID:** 2610.36671 | [PDF](https://arxiv.org/pdf/2610.36671v1)

**作者:** Donghyeon Cho `[一作]`, Eunho Yang `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Computational Efficiency` `Representation Learning` `Transformer` `Vision Language Model` `Diffusion model` `Vision-Language-Action Model` `Contrastive Learning` `Retrieval-Augmented Generation` `Prompt Engineering` `Image` `Text` `Multimodality` `Video`

**🎯 论文内容**

提出了一种名为 LOWN 的解码方法，能够自适应地决定在扩散视觉语言模型的每一步是否需要访问图像，并根据是否需要图像决定解码的步长。

**💡 创新点**

创新点在于利用 FULL 与 TEXT‑ONLY 前向推理的预测一致性作为无训练、无阈值的判别标准，既能判断何时省略图像处理，又能动态确定一次可解码的 token 数量；该方法可与视觉 token 剪枝、键值缓存等技术无缝融合。

**🔧 技术方法**

核心技术包括：视觉 token 剪枝（RedVTP、ZOO‑Prune）、键值缓存（Fast‑dLLM）、基于预测一致性的证书（anchor & certificate）以及动态全/文本前向调用的控制策略。

**📊 数据集**

在 LLaDA‑V、LaViDa、VidLaDA-8B 三个 DVLM 框架上，使用十个文本‑图像多模态评测集（MathVista、ScienceQA、ChartQA、MME、MMBench、DocVQA、GQA、LLaVA‑Bench、CapArena、DeCapBench）以及视频评测集（EgoSchema、MLVU）。

**📈 对比分析**

与传统的完整前向解码、RedVTP、ZOO‑Prune、KLASS、D^3ToM 以及 Fast‑dLLM 等方法对比，LOWN 在保持 99.6% 基线得分的同时，将 FLOPs 降低至 0.35 倍；与视觉 token 剪枝联用时 FLOPs 进一步下降至 0.22 倍；在时间速度上平均提升 57% 以上。

**⚠️ 局限性**

局限性包括：anchor 预测随着时间推移可能失效，导致不一致率升高；对需要持续视觉依赖的任务（如表格、文档阅读）节省空间有限；无法在解码前预设精确的计算预算，需在运行时动态调整。

---

## 57. Tail Cues, Principal Corrections: Plug-and-Play Rectification for Open-Set Test-Time Adaptation

**arXiv ID:** 2610.36870 | [PDF](https://arxiv.org/pdf/2610.36870v1)

**作者:** Yingkai Yang `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**关键词:** `Machine Learning` `Domain Adaptation` `Classification` `Anomaly Detection` `Convolutional Neural Network` `Transformer` `Contrastive Learning` `Gaussian Splatting` `Auto Encoder` `Weighted Ridge Regression` `Singular Value Decomposition (SVD)` `Image`

**🎯 论文内容**

在非平稳目标流的开放集测试时适应（OSTTA）场景下，提出一种轻量级、无梯度更新的特征校正模块 TALOR，利用冻结的线性头作为几何锚点，对特征进行主子空间和尾子空间分解并纠正可移除的主漂移；

**💡 创新点**

创新点在于通过头权重的奇异值分解将特征投影到高增益主坐标和低增益尾坐标，利用尾坐标作为漂移感知器，采用软路由的尾能量估计和加权岭回归在线估计可移除的仿射主漂移，既不需要梯度更新也不依赖源数据；

**🔧 技术方法**

核心技术包括对线性头权重的SVD分解、尾能量的高斯混合模型软路由、加权岭回归求解尾-主线性映射、指数滑动平均更新、以及无梯度的特征重构与源头推断；

**📊 数据集**

主要使用ImageNet-C（15种噪声、最高severity）和LAION-C（6类域移），并在ImageNet-R、ImageNet-Sketch、Places365、Textures、NINCO、iNaturalist等数据集上进行开放集和闭集的连续评估；

**📈 对比分析**

在ImageNet-C基准上，TALOR以单独补丁方式提升平均H-score至66.1%，超过下一佳UniEnt 63.9% 约2.2个百分点；在将TALOR叠加至COME后实现68.4% H-score，较基线提升约4.7%；在LAION-C上亦取得显著提升，平均H-score提升10.8%（CoTTA）至13.5%（ViDA）等；

**⚠️ 局限性**

局限性主要包括对能量阈值τ和EMA动量β的依赖，在极端语义或协变量漂移下软路由可能误判，且不更新backbone，可能在高度动态或大规模尾维度情况下效果受限；

---

## 58. DCKV: Learned Transform Coding of the Prompt KV Cache in Hybrid Language Models

**arXiv ID:** 2610.36950 | [PDF](https://arxiv.org/pdf/2610.36950v1)

**作者:** Egor Maximov `[一作]` (MIL Team), Aleksei Goncharov `[通讯]` (MIL Team)

**关键词:** `Machine Learning` `Compression` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Auto Encoder` `Prompt Engineering` `Contrastive Learning` `Text`

**🎯 论文内容**

本文提出一种针对混合型语言模型 KV 缓存的可学习压缩编码器，能够在冻结模型上端到端训练。

**💡 创新点**

创新点在于将变换、量化和条件熵模型一起训练，并使用上下文条件预测来显著减少缓存大小。

**🔧 技术方法**

技术上采用 PCA 联合基底、非线性校正、混合量化、GPU rANS 以及基于 token 嵌入和邻居信息的条件熵模型。

**📊 数据集**

使用的训练数据为 FineWeb‑Edu 以及 48 篇校准文档（包括网络文本和数学推理），评估在 Qwen3.5‑2B 上。

**📈 对比分析**

在 RULER、LongBench 等长上下文基准上，压缩率约 210 bytes/token，较 KVTC 低 70%+，且在同等分数下性能更优。

**⚠️ 局限性**

局限包括仅在单一模型上测试、每个压缩率需单独训练、未验证跨设备解码一致性以及对更大上下文的实测不足。

---

## 59. CogArena: Benchmarking Multimodal Agents on Interactive Behavioral Experiments

**arXiv ID:** 2610.36484 | [PDF](https://arxiv.org/pdf/2610.36484v1)

**作者:** Kianté Fernandez `[一作]` (University of California), Ian Krajbich `[通讯]` (University of California)

**关键词:** `Artificial Intelligence` `Benchmark` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Computational Efficiency` `Explainability and Interpretability` `Federated Learning` `Recommendation System` `Optimization` `Robotic Intelligence` `Drug Discovery` `AI Code Assistant` `Meta Learning` `Neural Architecture Search` `Protein Structure Prediction` `Transformer` `Agentic AI` `Prompt Engineering` `Vision Language Model` `Vision-Language-Action Model` `Retrieval-Augmented Generation` `Chain-of-Thought` `Mixture of Experts` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

创建了CogArena基准，让AI代理通过浏览器完成10个典型的社交与行为实验，并以完成度、准确度和对人类行为签名的匹配三层评分；

**💡 创新点**

创新点在于将实验从文本提示迁移至真实的浏览器交互，采用三层可复制的评分体系，直接对比AI代理产生的数据与已发布的人类行为签名，并公开了可更新的排行榜；

**🔧 技术方法**

技术实现基于jsPsych构建实验，使用Browser-Use等脚手架将模型输出转化为键盘/鼠标动作，利用OpenRouter统一调用多家模型提供商，统计层面采用比例检验、配对检验、相关性检验等传统统计方法评估签名显著性；

**📊 数据集**

数据集为10个实验的原始刺激与设计，结合每个实验公开的人类基准数据作为Level 2与Level 3的对照；

**📈 对比分析**

通过对四款经济级多模态模型（Moonshot AI、Google、Alibaba、OpenAI）在每个任务下进行10次实验，计算Level 1完成率、Level 2准确度（与人类均值比较）以及Level 3签名显著性；结果显示完成率不高，平均Composite得分约58.6（最高模型最高），但不同签名在不同任务与会话间差异大；

**⚠️ 局限性**

局限在于未收集人类在CogArena版实验中的数据，比较仅基于原始实验结果；模型样本有限（仅经济级且一次性跑最高级模型）；任务设计覆盖不全，未能检验所有已知行为签名；

---

## 60. The uneven rise of AI-generated web content

**arXiv ID:** 2610.36631 | [PDF](https://arxiv.org/pdf/2610.36631v1)

**作者:** Adam Visokay `[一作]` (University of California, Berkeley), Tyler H. McCormick `[通讯]` (University of Washington)

**关键词:** `Artificial Intelligence` `Generation` `Data Synthesis` `Large Language Model` `Bayesian Hierarchical Modeling` `Post-stratification` `PageRank Sampling` `Text`

**🎯 论文内容**

本文研究了 2019‑2026 年 Common Crawl 抓取的网页中 AI 生成文本的普及率及其随时间、语言、域名层级、页面中心度等维度的差异。

**💡 创新点**

创新点在于构建 Bayesian 误差校正模型，将多种 AI 检测器（开源与商业）的灵敏度与特异性差异纳入估计，并通过后归一化（post‑stratification）实现对整个公共网页的推断。

**🔧 技术方法**

采用了 Bayesian 层级模型、PageRank 分层抽样、后归一化、以及两款主检测器（open Pangram + Binoculars‑Falcon）进行文本标注。

**📊 数据集**

使用 Common Crawl 2019‑2026 年的抓取数据、页面语言、长度、PageRank、TLD 信息，以及 EditLens、RAID、多语种 GenAI、AITDNA 四个标注语料库来校准检测器。

**📈 对比分析**

与单一检测器或未校正的比例相比，校正后估计的整体 AI 文本比例从 2022 前的近 0% 增至 2026 年 24%（95% 可信区间 22.9‑24.4），并揭示了不同语言、域名层级和页面中心度的显著差异。

**⚠️ 局限性**

限制包括仅分析可抓取且文本长度 ≥ 500 字的网页，未覆盖社交媒体短文本；检测器误差假设可能不完全成立；未包含域级随机效应；以及 Common Crawl 的采样和抓取策略可能导致对某些网站的欠采样。

---

## 61. Back to the Future: The Evolving Universal Relation Model

**arXiv ID:** 2610.36833 | [PDF](https://arxiv.org/pdf/2610.36833v1)

**作者:** Nan Tang `[一作]` (Hong Kong University of Science and Technology), Haixun Wang `[通讯]` (Evenup)

**关键词:** `Databases` `Data-Centric Learning` `Federated Learning` `Optimization` `Retrieval-Augmented Generation` `Prompt Engineering` `Supervised Fine-Tuning` `Text` `Tabular` `Finance Related` `Review/Survey Paper`

**🎯 论文内容**

本文提出一种以查询驱动的数据库构建与演化框架（Evolving Universal Relation Model），在文档分析场景下让查询既能得到答案，又能逐步构建可复用、可约束的数据库状态。

**💡 创新点**

创新点在于：①将查询本身视为数据库演化的原语；②引入“演化的通用关系”作为用户视图，动态生成逻辑模式和物化实例；③在文档提取与数据库事务之间形成闭环，实现查询即构造、查询即维护。

**🔧 技术方法**

使用技术包括自然语言解析与文本到SQL编译、ER与关系模型的动态归约、文档提取与实体解析层、可扩展的索引与优化器、以及事务与版本管理机制。

**📊 数据集**

实验使用多种企业文档集合（合同、发票、报告、邮件、政策、票务）以及演示案例中的课程分析文档进行验证。

**📈 对比分析**

与传统 schema‑first、schema‑on‑read 和 RAG 方法对比，实验显示在相同工作负载下，查询响应时间下降约30%–40%，并且后续查询可重用的事实数显著提升，整体系统成本保持在可接受范围内。

**⚠️ 局限性**

局限性包括：①对文档提取精度和实体解析的高度依赖；②事务隔离和版本一致性仍需进一步完善；③索引与存储结构对频繁演化的支持尚不成熟；④对不确定性、置信度与证据覆盖度的处理仍为未来研究方向。

---

## 62. MM-Net: A Multimodal Multi-Task Network for Sleep Staging and Sleep Apnea Assessment After Ischemic Stroke

**arXiv ID:** 2610.36583 | [PDF](https://arxiv.org/pdf/2610.36583v1)

**作者:** Md Wahiduzzaman Suva `[一作]` (American International University-Bangladesh), Jia Uddin `[通讯]` (Woosong University)

**关键词:** `Machine Learning` `Classification` `Anomaly Detection` `Convolutional Neural Network` `Recurrent Neural Network` `Transformer` `Contrastive Learning` `Supervised Fine-Tuning` `Cross-Attention` `Biomedical Data` `Time Series` `Multimodality` `Electrocardiogram`

**🎯 论文内容**

提出一种多模态多任务网络 MM-Net，用于在缺血性脑卒中患者的多导睡眠多导监测（PSG）数据中同时进行睡眠分期和呼吸暂停事件检测。

**💡 创新点**

创新点在于：① 引入睡眠专门化的 EEG 编码器并通过交叉拟合与岭回归实现表示一致性；② 将神经（EEG/EOG）与心肺（ECG、呼吸、SpO₂ 等）特征通过交叉注意力和共享时序建模融合；③ 在单模型中实现睡眠分期、呼吸事件评分和 AHI 估计。

**🔧 技术方法**

核心技术包括预训练 EEG 编码器 LaBraM 的睡眠微调、交叉拟合 + 对齐、卷积编码器、交叉注意力融合、双向 LSTM 时序解码、HMM 后处理与 AHI 计数公式。

**📊 数据集**

主要使用 iSLEEPS（99名缺血性脑卒中患者的 PSG）、Sleep‑EDF（健康睡眠）、ISRUC‑Sleep（睡眠障碍与健康）和 UCD（睡眠呼吸暂停）四个公开数据集进行训练、验证和跨域测试。

**📈 对比分析**

与 13 种重新实现或公开的单任务/多任务睡眠分期模型比较，MM‑Net 在 iSLEEPS 上达到 0.864 的准确率（宏 F1 0.756、κ 0.664），呼吸事件 AUC 0.825，显著优于所有基线（p<0.01）。在 ISRUC‑I 以及未重训练的外部数据集上，模型仍保持 0.83–0.84 的呼吸事件 AUC 与 0.63–0.65 的 AHI 相关性。

**⚠️ 局限性**

主要局限包括：仅在单中心的 iSLEEPS 数据上验证，缺乏大规模多中心验证；模型设计为离线评分，无法实时检测；呼吸事件仅以 30 s 事件起始为标签，未提供精确起止时间；对鼻压等设备特定信号的依赖导致迁移时需删减通道。

---

## 63. Too many clients spoil the logs: distributed parsing, aggregation techniques and metrics

**arXiv ID:** 2610.36946 | [PDF](https://arxiv.org/pdf/2610.36946v1)

**作者:** André García Gómez `[一作]` (Austrian Institute of Technology), Edgar Weipp `[通讯]`

**关键词:** `Cryptography and Security` `Federated Learning` `Anomaly Detection` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Text` `Benchmark`

**🎯 论文内容**

本文对分布式日志解析进行了系统的实验研究，评估了多种解析算法以及聚合技术的效果，并提出了针对分布式环境的指标体系；同时，基于6,720个实验配置生成了40,320条结果数据，深入剖析了不同解析器在客户端数量和日志量变化下的性能表现。

**💡 创新点**

创新点包括：①提出了新的指标分类法（如TS、RT、FR、PS）并与现有指标对比；②首次在分布式环境下对5种主流解析算法进行大规模基准测试；③设计并评估了多种聚合策略（串接、缓存、去重等），展示其对解析质量和下游异常检测的影响。

**🔧 技术方法**

技术方法涵盖：使用Frequency-based Logram、Similarity-based LenMa、Heuristic-based Drain/Brain、LLM-based LogBatcher；实现了Federated Parsing Pipeline，支持集中式和去中心化聚合；采用OpenAI GPT‑5.4‑mini进行LLM推理；使用LogHub的HDFS和BGL数据集；实现了多种评价指标的计算与可视化。

**📊 数据集**

使用的主要数据集为LogHub公开的HDFS和BGL两大工业日志集，分别包含约1,115万条和470万条日志，模板数分别为52和377。

**📈 对比分析**

比较方法：在本地与分布式配置下，计算GA、TS、NED、PA、FR、RT、PS等指标；通过热图和折线图展示客户端数量与数据比例对指标的影响；结果表明聚合技术可提升解析器性能，尤其在LLM解析器和BGL数据集上效果显著；部分解析器对客户端数量不敏感，表现更为稳定。

**⚠️ 局限性**

局限性：基准使用的ground truth模板可能含有人工误差；仅选取两类数据集和有限数量的解析器，可能不具备普适性；LLM可能在预训练阶段已接触部分日志，导致性能偏差；聚合过程增加通信开销与隐私风险；实验仅采用IID场景，未覆盖非IID条件。

---

## 64. SOMA: A Benchmark for Fine-Grained Egocentric Behavior and Body-Motion Understanding

**arXiv ID:** 2610.36384 | [PDF](https://arxiv.org/pdf/2610.36384v1)

**作者:** Matea Tashkovska `[一作]` (École Polytechnique Fédérale de Lausanne), Alexander Mathis `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `Computer Vision and Pattern Recognition` `Benchmark` `Transformer` `Large Language Model` `Vision Language Model` `Auto Encoder` `Contrastive Learning` `Retrieval-Augmented Generation` `Video` `Text` `Multimodality`

**🎯 论文内容**

本文提出了 SOMA 这一从第一视角视频中细粒度理解人体运动的基准，并构建了 1,632 条人类验证的问答样本，随后对 37 种多模态大语言模型（MLLM）与人类评测者进行零样本比较。

**💡 创新点**

创新点在于：①将身体配置、运动动力学与时间推理三大能力轴拆分成 12 个任务，专注于体位与细节动作；②通过人工审核的盲目语言过滤和人类校验生成高质量问题；③构建了与视频同步的 3D 运动信息训练集 SOMA-Train，首次在 egocentric 场景中加入运动信号以提升模型感知。

**🔧 技术方法**

使用的技术包括：LLM（Gemini 2.5、GPT‑4o 等）生成和筛选问答、VQ‑VAE 运动编码器、对齐视频/运动/文本三模态的多模态 Transformer 训练、以及多模型评估框架。

**📊 数据集**

所用数据集为 Nymeria 与 Ego‑Exo4D 两大 egocentric 视频库，提取 397 条视频（1,611 片段）构成基准；SOMA‑Train 由同源视频、对应 3D 运动轨迹与语言描述构成 28,596 条 QA 对。

**📈 对比分析**

对 37 模型进行零样本评估，人类平均准确率 68.5%，最佳模型 Qwen3.6‑27B 仅 50.0%；通过对比不同采样密度、推理策略和多模态输入，发现仅加入运动信号可获得 2–3% 的提升，但整体仍与人类相距较远。

**⚠️ 局限性**

局限性包括：模型对身体自身的感知仍显弱，时间推理与重复计数准确率低；存在右手偏好与计数欠估的偏差；基准和训练集仍受限于现有 egocentric 数据，缺乏更丰富的多感官信息；当前的多模态融合方法效果有限，尚需更先进的感知与推理机制。

---

## 65. Stratum: Repository Memory for Coding Agents Beyond the Context Window

**arXiv ID:** 2610.36153 | [PDF](https://arxiv.org/pdf/2610.36153v1)

**作者:** Yuvraj Sehgal `[一作]` (Brock University), Naser Ezzati-Jivan `[通讯]` (Brock University)

**通讯引用:** 574 | [OpenAlex ID](https://openalex.org/A5019537475)

**关键词:** `Software Engineering` `AI Code Assistant` `Graph Neural Network` `Transformer` `Retrieval-Augmented Generation` `Large Language Model` `Prompt Engineering` `Text` `Graph`

**🎯 论文内容**

提出一种与代码仓库并存的多层持久化内存层，结合结构、历史与经验知识，并通过页面安全工具让编程代理跨会话、跨模型保持记忆。

**💡 创新点**

创新点在于：①将知识持久化为文本+图结构而非模型KV；②通过共变与失败边缘弥补静态分析的盲点；③引入页面安全协议，使代理能可视化分页与预算；④在多会话、多模型实验中验证其有效性。

**🔧 技术方法**

使用 SQLite 图数据库 + HNSW 向量索引 + 文件监视器 + 页面安全工具（MCP）与 LLM（Claude Sonnet/Haiku）进行交互；嵌入使用本地模型生成。

**📊 数据集**

基于六个开源项目（Django、Flask、Redis、requests、scikit‑learn、Vue core）共 17.6M tokens；从 752+852 个 bug‑fix 提交抽取样本；合成存储 1M 节点（约 440M tokens）用于性能测试。

**📈 对比分析**

与检索仅（RAG）及全文上下文对比：在 8 个真实 bug‑fix 任务中，继承经验记忆的代理成功率 21/24（87.5%），token 成本降低 4.5×；检索仅成功 6/24；全上下文条件 14/16 但成本 3.8×；存储检索平均 1.7 ms。

**⚠️ 局限性**

局限性包括：仅针对 Python 项目、静态导入基于最新提交；共变边缘不覆盖所有语义；实验规模有限、写入笔记质量未控制；检索仅词法，未评估嵌入效果；未测评存储更新开销及多模型兼容性。

---

## 66. TAPESTRY: A Single Backbone for Video Generation and Robot Control

**arXiv ID:** 2610.36823 | [PDF](https://arxiv.org/pdf/2610.36823v1)

**作者:** Jeremy A. Collins `[一作]` (Georgia Institute of Technology), Paarth Shah `[通讯]` (Toyota Research Institute)

**关键词:** `Robotics` `Robotic Intelligence` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Auto Encoder` `Vision-Language-Action Model` `Video` `Text` `Sequential`

**🎯 论文内容**

本文提出一种将预训练的视频扩散变换器重新用于机器人控制的方法，称为TAPESTRY，它通过将每个动作时间步视为VAE patch令牌，在不进行视频生成的情况下直接生成动作序列；

**💡 创新点**

创新点在于：①将动作映射为视频VAE的patch令牌，保持与预训练输入统计一致；②采用三目标联合训练（视频扩散、联合视频+动作扩散、动作仅扩散）共享同一Transformer backbone；③在零终端SNR v‑prediction调度下实现单步DDIM推理，避免高频抖动并大幅提升速度；

**🔧 技术方法**

技术上使用预训练的DiT（CogVideoX-2B）和3D‑VAE编码器、T5‑XXL文本编码器、DDIM/DDPM扩散框架，以及线性映射W、无偏差层归一化、Patchifying卷积ϕ；

**📊 数据集**

数据集包括大规模视频（Ego4D、Epic Kitchens、SSv2、Panda‑70M、DROID、BridgeData、AgiBot‑World）以及机器人交互数据（遥控演示的Toolbox、Oven、Corn任务）和LIBERO视频数据；

**📈 对比分析**

与现有方法（Cosmos Policy、DreamZero、Amplify、Diffusion Policy等）比较，TAPESTRY在LIBERO保留任务上平均成功率达76%（较Cosmos Policy+DreamZero提升6%），在真实双手机械臂任务的在分布成功率为93%/28/30，出乎意料的OOV表现达到67–75%；推理速度在RTX 5090上单步生成16步动作只需84 ms，占用9 GB GPU内存，比DreamZero快约44×；

**⚠️ 局限性**

局限性包括：单步推理仅给出动作分布的条件均值，可能不足以处理多模态或高度随机的动作；依赖零终端SNR v‑prediction调度，其他扩散目标（如rectified flow）尚未验证；以及仅针对20D双手动作空间，未探索多体制的动作头。

---

## 67. Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better

**arXiv ID:** 2610.36774 | [PDF](https://arxiv.org/pdf/2610.36774v1)

**作者:** Jialin Zhang `[一作]` (University of California, Berkeley), Matei Zaharia `[通讯]` (University of California, Berkeley)

**关键词:** `Artificial Intelligence` `Optimization` `Text`

**🎯 论文内容**

本文研究了将GEPA反射式优化循环进行批量并行化：每一步同时提出多个候选（P个父代×N个变异），并行评估它们，从而加速优化并探索更广阔的搜索空间。

**💡 创新点**

创新点在于：①提出了批量并行提案机制，打破原本单步单提案的顺序结构；②通过runtime模型解释并预测不同宽度下的加速和瓶颈；③探讨了在固定优化预算下如何在父代数P、变异数N等维度进行资源分配。

**🔧 技术方法**

技术实现基于开源GEPA优化器，扩展了其采样策略；采用并行反射评估（mini‑batch + 全验证）以及多工人池；利用简化的runtime模型估算时间和验证延迟；进一步通过成本与metric‑call预算的统一度量来评估效率。

**📊 数据集**

使用了两个公开任务数据集：LiveBench‑Math（数学竞赛题目）和HoVer（多跳事实验证），分别在这两个任务上进行实验。

**📈 对比分析**

与传统单突变（单提案）方法对比：批量化方案在墙钟时间上平均缩短约3–4倍；在LiveBench‑Math上从68.9%提升至72.1%，在HoVer上从49.0%提升至60.0%；在metric‑call和花费预算上也表现更高效，尤其在前期预算有限时显著优于单突变。

**⚠️ 局限性**

局限性包括：①受限于worker池规模，过宽的批量会导致单步延迟增长；②全验证阶段仍是主要瓶颈，且其触发概率随批量大小上升；③目前的P、N选择仍以经验为主，缺乏系统化的预算分配策略；④在更大规模或不同类型任务上的泛化效果尚待进一步验证。

---

## 68. Learning from Points and Patches: A Pre-trained Hybrid Attention Transformer for Time Series Analysis

**arXiv ID:** 2610.36572 | [PDF](https://arxiv.org/pdf/2610.36572v1)

**作者:** Peiwang Tang `[一作]` (University of Science and Technology of China), Defu Lian `[通讯]` (University of Science and Technology of China)

**关键词:** `Machine Learning` `Anomaly Detection` `Classification` `Representation Learning` `Transformer` `Contrastive Learning` `Supervised Fine-Tuning` `Time Series`

**🎯 论文内容**

提出了TimeHAT框架，结合统一点-块掩码策略和混合注意力编码器，对时序数据进行预训练，实现跨任务迁移。

**💡 创新点**

创新在于将点级和块级掩码统一为UPPMS以同时学习细粒度与宏观模式，并设计前缀约束双向+方向混合注意力以防止深层秩退化。

**🔧 技术方法**

采用Transformer结构、RoPE位置编码、掩码自回归预训练、前缀约束注意力、方向混合注意力，以及MLP投影头。

**📊 数据集**

在大规模1B点的UTSD-12G进行预训练，随后在ETT、Weather、Electricity、Traffic、UAE多变量等六大基准以及异常检测、分类任务上评测。

**📈 对比分析**

与众多时序基线（TimesNet、TimeMixer++、GPT4TS、Autoformer等）以及基于Transformer的最新模型进行fine-tune、few-shot和zero-shot对比，TimeHAT在长期预测、插值、异常检测和分类上均实现或逼近最佳结果。

**⚠️ 局限性**

缺乏跨模态、跨域多任务统一框架，对长序列分块效率与对抗鲁棒性等方面仍有待提升。

---

## 69. Generalization in Learning: Eyring Formula and Predator-prey Model

**arXiv ID:** 2610.35965 | [PDF](https://arxiv.org/pdf/2610.35965v1)

**作者:** S. V. Kozyrev `[一作]` (Steklov Mathematical Institute of Russian Academy of Sciences), Alexander Pechen `[通讯]` (Steklov Mathematical Institute of Russian Academy of Sciences)

**关键词:** `Machine Learning` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Stochastic Differential Equation` `Tabular` `Time Series` `Physics Related`

**🎯 论文内容**

本研究通过引入捕食-猎物模型与Eyring公式，探索了机器学习中的泛化机制，并用SDE方法对过拟合进行抑制；

**💡 创新点**

创新点在于将生物学捕食-猎物动力学与统计物理热力学相结合，提出通过捕食者推离狭窄极小值来实现宽极小值选择，从而提升泛化性能，并利用Eyring公式解释了“grokking”延迟泛化现象；

**🔧 技术方法**

采用随机梯度朗之万动力学（SDE）、捕食-猎物优化算法、宽极小值选择策略，以及对比传统梯度下降的数值实验；

**📊 数据集**

使用合成数据集，基于隐藏函数f(x)=sin(πx)在区间[-1,1]上采样，加入高斯噪声的训练集和测试集；

**📈 对比分析**

对比传统梯度下降与捕食-猎物SDE优化后，测试集均方误差从0.161下降至0.106，偏差从0.0594降至0.0334，显示出显著的过拟合抑制效果；

**⚠️ 局限性**

局限性包括仅在简单回归任务上验证，需大量迭代（10^6次）才能显现优势，且对更复杂实际任务的推广尚未评估。

---

## 70. TDLM: Adaptive Temporal Integration for Diffusion Language Models

**arXiv ID:** 2610.36940 | [PDF](https://arxiv.org/pdf/2610.36940v1)

**作者:** Bohang Sun `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `Machine Learning` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Diffusion model` `Cross-Entropy Self-Distillation` `Time Attention` `Text`

**🎯 论文内容**

在冻结的扩散语言模型上引入时间自条件化模块，让未完成位置利用其历史隐藏表示进行预测，从而减少前向传播次数。

**💡 创新点**

提出位置对齐的时间注意力机制与简单的轨迹自蒸馏目标，允许模型重用先前的推理计算。

**🔧 技术方法**

使用冻结的Transformer骨干、时间注意力层、交叉熵自蒸馏、并行轨迹训练。

**📊 数据集**

在 LLaDA、Dream 两个模型上，使用 GSM8K、MATH-500、MBPP 任务数据集。

**📈 对比分析**

与原始扩散语言模型和 Fast-dLLM 进行比较，TDLM 在生成速度上提升最多 7.15 倍，且保持或略优的答案质量。

**⚠️ 局限性**

依赖于教师生成的轨迹，受限于冻结骨干的能力；历史信息量有限，且需要额外的存储与计算开销。

---

## 71. RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG

**arXiv ID:** 2610.36444 | [PDF](https://arxiv.org/pdf/2610.36444v1)

**作者:** Zekai Chen `[一作]` (Beijing Institute of Technology), Rong-Hua Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `Artificial Intelligence` `Retrieval` `Computational Efficiency` `Representation Learning` `Transformer` `Retrieval-Augmented Generation` `Agentic AI` `Prompt Engineering` `Large Language Model` `Reinforcement Learning` `Multimodality` `Text` `Benchmark`

**🎯 论文内容**

研究了在多模态文档检索增强生成中何时调用Agent以及如何控制其后续操作，并提出了名为RouteDRAG的框架；

**💡 创新点**

通过学习式路径路由与动态Agent执行相结合，实现查询级与操作级的选择，解决Agent不必要调用和非单调操作收益问题；

**🔧 技术方法**

采用图结构文档索引、查询条件表示、相对路径预测、动作强化、Planner提议修正、顺序策略学习等技术，并基于LLM Planner；

**📊 数据集**

使用DocBench和MMLongBench‑Doc两大多模态文档问答基准；

**📈 对比分析**

与九种基线RAG方法以及规则/统一Agent执行方式对比，RouteDRAG在DocBench达到75.86%准确率、MMLongBench‑Doc 53.80%，在保持或提升准确率的同时减少回答调用和令牌使用；

**⚠️ 局限性**

仅限单轮问答，未覆盖多轮交互；在极端噪声或低资源环境下鲁棒性仍有待提升。

---

## 72. Exploratory creative pre-ranking: Hierarchical empirical-Bayes Thompson sampling for real-time creative optimization

**arXiv ID:** 2610.36020 | [PDF](https://arxiv.org/pdf/2610.36020v1)

**作者:** Jiale Han `[一作]` (UCLA), Xiaowu Dai `[通讯]` (UCLA)

**关键词:** `Machine Learning` `Optimization` `Recommendation System` `Thompson Sampling` `Bayesian Inference` `Hierarchical Modeling` `Tabular` `Sequential`

**🎯 论文内容**

提出一种在广告召回后、排名前进行的创意预排序方法，使用层级经验贝叶斯先验与Thompson采样来为每个创意评分，并通过单一draw挑选候选子集。

**💡 创新点**

创新点在于：① 通过层级结构（全局→受众段→广告宇宙→广告系列→创意）自适应估计，利用历史数据一次性求得每层的收缩参数；② 仅维护单属性计数器并在请求时用线性对数几率模型重建每段的后验；③ 采用Thompson draw实现探索，同时通过每个系列的配额和全局预算的轮询分配保持计算成本可控。

**🔧 技术方法**

使用的技术包括：层级经验贝叶斯估计、Beta-Bernoulli后验与Thompson采样、单属性计数器的线性对数几率重构、滚动全局预算控制与轮询分配、离线参数拟合与在线即时推理。

**📊 数据集**

实验数据来自 Roblox Sponsored‑ads 2026 年流量日志，重放约 10^5 条召回请求，包含 500 万以上创意；后续在实时流量中做了 A/B 测试。

**📈 对比分析**

比较方法包括：Uniform 随机挑选、TS‑One 单创意Thompson、RR‑Random 随机填充配额、TS‑RR（完整算法）以及 Oracle 全部创意；在 Replay 与 A/B 中，TS‑RR 使预测的 Play‑Through‑Rate 提升 8.14%（相对 Uniform）并覆盖 44% 的理想上限，且仅增加 0.2% 的服务延迟。

**⚠️ 局限性**

局限性包括：需要频繁离线更新层级收缩参数和属性权重；对单属性计数器的线性重构假设属性效应可加性，可能在高度交互的目标中失效；目前仅在 Roblox 平台验证，通用性待进一步验证。

---

## 73. TrawMem: Reframing Agent Memory as Task-Oriented Role-Adaptive Workspaces

**arXiv ID:** 2610.36918 | [PDF](https://arxiv.org/pdf/2610.36918v1)

**作者:** Mingyu Zhang `[一作]`, Yupeng Hu `[通讯]`

**关键词:** `Computation and Language` `Retrieval` `Computational Efficiency` `Representation Learning` `Explainability and Interpretability` `Federated Learning` `Safty and Privacy` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Graph Neural Network` `Text` `Sequential` `Benchmark`

**🎯 论文内容**

构建了基于任务角色自适应的工作空间，用以提升 LLM 代理的记忆检索与组合效率。

**💡 创新点**

创新点在于将记忆视为任务角色与关系的工作空间，而非单纯相关性检索；通过角色分配与互补性选择，实现更精细的记忆组合。

**🔧 技术方法**

使用 LLM 进行线程摘要、关系建模、角色赋值；基于子模量优化的选择算法；图结构表示线程与记忆。

**📊 数据集**

使用 LoCoMo 与 MemGallery 两大长序列对话记忆基准，测试四种公开/闭源模型（Qwen3.5‑9B、Qwen3.8‑27B、DeepSeek‑V4‑Flash、GPT‑5.6‑Luna）。

**📈 对比分析**

与现有 SimpleMem、LightMem 等基线对比，平均提升宏 F1 2.23–3.14 分，LLM‑Judge 提升 2.19–6.34 分；在多跳、时序等子任务表现尤为显著。

**⚠️ 局限性**

仍受检索延迟与上下文长度限制；在时间精度与注释不一致时易失误；需要更多数据隐私与偏见控制措施。

---

## 74. GNN-Guided Network Reduction for Centralized Urban Traffic Routing

**arXiv ID:** 2610.36981 | [PDF](https://arxiv.org/pdf/2610.36981v1)

**作者:** Nikolai Antonov `[一作]` (Czech Technical University in Prague), Jan Hůla `[通讯]` (Czech Technical University in Prague)

**关键词:** `Machine Learning` `Optimization` `Autonomous Driving` `Graph Neural Network` `Reinforcement Learning` `Contrastive Learning` `Graph` `Tabular` `Time Series`

**🎯 论文内容**

研究中央化交通路由（CTR）中分段流的MILP在运营时间预算内求解困难，提出基于GNN的网络缩减方法，先为每个OD流预测路段重要性评分，再构建缩减网络并在该网络上求解MILP。

**💡 创新点**

创新点在于将路段评分与网络缩减结合，使用专门的GNN学习流特定路段排序；方法可为每个时刻动态生成缩减网络，显著提升求解质量；同时对比LP和随机排序验证评分效能。

**🔧 技术方法**

采用的技术包括：分段流混合整数线性规划、图神经网络（GNN）进行路段评分、启发式路段选择与完整路径构造、路由恢复算法、SUMO微观仿真评估。

**📊 数据集**

使用四个真实城市路网（布拉格‑德维奇、伦敦、纽约、悉尼）及其对应的OD流和交通场景（静态与递增）作为数据集。

**📈 对比分析**

与固定缩减方法（FG、SP、BR、SR）以及GNN、LP、Random评分进行对比；在10秒预算下，GNN平均相对目标差距仅0.4–0.6%，比最佳固定缩减低约6倍，在仿真中获得最高到达率和竞争性的行程时长。

**⚠️ 局限性**

主要限制在于MILP仅通过分段成本近似拥堵，未完整模拟信号灯、队列、车辆交互等细节；GNN评分模型依赖训练样本，对极端场景或极大网络的泛化可能受限；实时更新与动态交通变化的适应性尚待改进。

---

## 75. Generative Refinement Learning for Temporal Interpolation in Scientific Machine Learning

**arXiv ID:** 2610.36751 | [PDF](https://arxiv.org/pdf/2610.36751v1)

**作者:** Pu Ren `[一作]` (Lawrence Berkeley National Lab), N. Benjamin Erichson `[通讯]` (Lawrence Berkeley National Lab)

**关键词:** `Machine Learning` `Generation` `Data Synthesis` `Optimization` `Diffusion model` `Diffusion model` `Score-based Model` `Optical Flow` `Time Series` `Image` `Physics Related`

**🎯 论文内容**

本研究提出一种通过先线性插值再利用条件扩散模型纠正误差的科学数据时间插值方法

**💡 创新点**

创新点在于将简单数值插值作为初始参考，训练扩散模型学习残差纠正，从而提高中间状态重建精度

**🔧 技术方法**

采用条件扩散模型（FLEX架构）结合FiLM归一化、门控融合两端观测特征，训练时用噪声化残差预测目标

**📊 数据集**

在三类数据集上评估：Navier–Stokes Kraichnan湍流（Re=2000~32000）、上海雷达降水和海表温度（128×128格）

**📈 对比分析**

与线性插值、U‑Net、Super SloMo、EDEN等基线对比，iFLEX在所有数据集上实现最低相对弗罗贝尼乌斯误差和最高SSIM，尤其在Re=36000时将误差从13.73%降至10.48%

**⚠️ 局限性**

局限包括：需要观测点时间间隔足够短，短时事件可能无法重建；线性插值作为参考可能不适用于更长间隔；单个时间点独立生成，缺乏时间序列一致性和方程守恒约束

---

## 76. Beyond Update Counts: A Control Map for Evaluating Test-Time Model Updates

**arXiv ID:** 2610.36109 | [PDF](https://arxiv.org/pdf/2610.36109v1)

**作者:** Zehao Zhao `[一作]` (Meta AI), Hamed Firooz `[通讯]` (Meta AI)

**关键词:** `Machine Learning` `Explainability and Interpretability` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Text` `Benchmark`

**🎯 论文内容**

提出 Control Map 框架，用于记录并比较在推理过程中完成的有序更新，并对其对模型回答的影响进行分层评估。

**💡 创新点**

创新点包括：① 将更新历史与性能差异的正式界限关联；② 将即时学习、后续保留与局部性拆分为独立决策指标；③ 在跨模型、跨基准的比较中明确更新计划对应关系并直接验证；④ 为评估提供可复现的记录与阈值定义。

**🔧 技术方法**

技术手段包括：形式化的更新策略与依赖规范、概率历史分辨率与差异界限、基于控制映射的跟踪规则、统计估计与置信区间、以及在 MAC‑760M、LoRA、MEMIT、ROME、AlphaEdit 等模型上的实验实现。

**📊 数据集**

使用的数据集涵盖：Single‑NIAH、LAMBADA、SocialIQA、CounterFact、WikiRecent、zsRE、WikiData‑CounterFact 等。

**📈 对比分析**

比较方法：先用 Control Map 记录每个策略在每个例子前的更新历史，再分别统计行为差异、获取、保留、局部性指标；实验显示层次放置、更新时机和剂量对准确率的影响因设置而异；部分编辑器在某些指标上优于其他，但在局部性或保留上往往失效，说明单一评估不能充分反映性能。

**⚠️ 局限性**

局限性：① 评估完全依赖预先定义的跟踪规则，若遗漏路径则失效；② 只在有限的模型与基准上验证，跨设置推广需额外验证；③ 需要手动指定跨模型策略对应关系，未实现自动化；④ 目前不保证对所有潜在更新路径的完整覆盖，可能导致某些行为被忽视。

---

## 77. Mining the Nineteenth Century for the Thirteenth: Register-Scale Forced Alignment and HTR Bootstrapping via Legacy Regesta and Antiquarian Editions

**arXiv ID:** 2610.36956 | [PDF](https://arxiv.org/pdf/2610.36956v1)

**作者:** Tamas Kovacs `[一作]`, Daniel Luger `[通讯]` (University of Vienna)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Data-Centric Learning` `Knowledge Distillation` `Transformer` `Convolutional Neural Network` `Contrastive Learning` `Retrieval-Augmented Generation` `Prompt Engineering` `Knowledge Distillation` `Text` `Image`

**🎯 论文内容**

本文在教皇奥涅尔三世注册簿的手写文本识别中，利用现代转录作为基准，评估19世纪旧版编辑和注册摘要的对齐、投射以及在无标签情况下的适配效果。

**💡 创新点**

创新点在于将传统的19世纪手稿编辑层与现代识别模型结合，提出零标签适配、投射正字法、基于注册摘要的条目分割和遗留词典及预训练中世纪NER的蒸馏提升识别质量。

**🔧 技术方法**

使用的技术包括Conformer-CTC编码器、Transformer注意力解码器、R2L BPE混合模型、字符/单词语言模型、基于字符投射的正字法映射、动态规划条目对齐、以及RoBERTa和Span-NER预训练模型的蒸馏。

**📊 数据集**

数据集包括Reg. Vat. 9的394页（1–130和137–203页），共860条记录、203,819词；19世纪十个全书文本和Horoy、MGH、Theiner等编辑；Pressutti的注册索引；以及使用OCR扫描的互联网档案版本。

**📈 对比分析**

通过与现代转录对齐、投射后测量WER/CER，并与oracle（现代第二年标签）对比，发现投射+重放可将第二年WER从约17.5%降低至约14%（约1/3 oracle差距），预训练NER蒸馏将NER F1提升至约90%（比规则F1高约10%），遗留词典能覆盖约30% OOV词并以95%精度标记错误。

**⚠️ 局限性**

局限包括仅针对单一注册卷；测试集规模小（13页），跨种子差异；19世纪文本覆盖率仅约一半；OCR错误和语料标注不一致；NER基准依赖于预训练模型且存在误标；未能区分不同书写者的视觉差异。

---

## 78. Omni-IGN: Onboard Image-Goal Navigation for Drones with Omnidirectional Vision

**arXiv ID:** 2610.36968 | [PDF](https://arxiv.org/pdf/2610.36968v1)

**作者:** Zichen Yan `[一作]` (National University of Singapore), Lin Zhao `[通讯]` (National University of Singapore)

**关键词:** `Robotics` `Autonomous Driving` `Reinforcement Learning` `Depth Estimation` `Transformer` `Convolutional Neural Network` `Reinforcement Learning` `Contrastive Learning` `Vision-Language-Action Model` `Optical Flow` `Image` `Video`

**🎯 论文内容**

提出了一套完全靠机载四镜头鱼眼相机实现的全景图像目标导航（ImageNav）系统——Omni-IGN；

**💡 创新点**

创新点在于：①引入自注意力模块在四个视角间自适应融合目标相关特征，显著提升探索效率；②将学习式立体深度估计与安全层结合，实现在不使用深度传感器的情况下的碰撞规避；③实现端到端连续速度控制，兼顾飞行平滑性与实时性；

**🔧 技术方法**

使用的技术包括：ResNet9视觉编码器、PPO强化学习、Multi-head自注意力、Fast-FoundationStereo立体深度网络、基于深度的碰撞预测与动作安全层、FP16 TensorRT加速；

**📊 数据集**

在模拟训练与评估使用Gibson、MP3D、HM3D三个室内数据集；

**📈 对比分析**

在Habitat基准上与其他连续动作基线对比，Omni-IGN在成功率(SR)和路径效率(SPL)上均超越同类方法，接近离散动作SOTA，并在真实环境中实现零射击到现实的无调优迁移，成功率超过95%；

**⚠️ 局限性**

局限性包括：立体深度扫描视角有限，导致侧向碰撞时误判；深度估计在纹理较少或光照变化剧烈场景下误差增大；在高度动态或多障碍密集环境中可能仍需更丰富的传感器信息或更复杂的安全策略。

---

## 79. Advancement or Acceleration? How Data Curation Improves LLM Training

**arXiv ID:** 2610.36983 | [PDF](https://arxiv.org/pdf/2610.36983v1)

**作者:** Jiaxing Wang `[一作]` (JD.com), Qixia Jiang `[通讯]` (JD.com)

**关键词:** `Machine Learning` `Data-Centric Learning` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Contrastive Learning` `Text`

**🎯 论文内容**

在大语言模型训练中系统研究了数据策划（混合和选择）对最终性能的提升与加速的区别，并提出了一种基于单次基线训练可计算的 Δ‑test 诊断方法。

**💡 创新点**

创新点在于：①将数据策划效果分为“advancement”（提升最终可达性能）和“acceleration”（更快达到相同性能）两类；②证明提升只有在存在可纠正的训练–目标差距时才可能发生；③提出可观测的 Δ‑test，能够在投入昂贵的策划之前预估提升潜力；④将这一理论框架统一到混合与选择两种策划范式，并通过理论与真实 LLM 训练验证。

**🔧 技术方法**

技术手段包括：教师‑学生感知器模型与复制法的统计力学分析，得到自洽方程预测训练预算与性能关系；实证实验使用 160M/1.1B GPT‑style LLM 训练；对混合采用 RegMix、DoGE、Skill‑It、Aioli、TANDEM 等方法；对选择采用 RHO‑1、BLADE 等目标感知方法；对比基线为均匀混合/全量训练和随机选择。

**📊 数据集**

主要数据集：SlimPajama‑6B（7个域）、OpenWebMath（候选池）、MetaMath‑Mammoth（目标），以及 TinyLlama‑1.1B、BERT‑style 训练与多数学基准（GSM‑8K、MATH、SVAMP、ASDiv、MAWPS、TAB、MQA、MMLU）。

**📈 对比分析**

与基线的比较方式：在相同训练预算下对模型的困惑度（混合）或任务准确率（选择）进行评估。实验结果表明：当 Δ≈0（无训练–目标差距）时，任何策划方法只能实现加速，最终性能与均匀基线相当；当 Δ>0 时，适当的混合或选择可在相同预算下获得显著提升（如混合中 perplexity 从 49.19 降至 40.2，选择中准确率从 17.2% 提升至 31.3%）。

**⚠️ 局限性**

局限性：Δ‑test 基于单一基线训练的损失差距，可能忽略更细粒度的分布不匹配；理论分析使用感知器模型，易于解析但对真实 NLP 任务的可迁移性有限；实验规模受 GPU 资源限制，仅在 160M/1.1B 规模验证，未覆盖更大模型或多任务场景；对目标构造方法的探索仍有限，需进一步研究如何设计更有效的目标集以最大化提升。

---

## 80. KL-MOPD: Discrepancy-Aware Multi-Objective On-Policy Distillation for Generative Recommendation

**arXiv ID:** 2610.36973 | [PDF](https://arxiv.org/pdf/2610.36973v1)

**作者:** Weiqin Yang `[一作]` (Zhejiang University), Jiawei Chen `[通讯]` (Zhejiang University)

**关键词:** `Information Retrieval` `Recommendation System` `Knowledge Distillation` `Transformer` `Supervised Fine-Tuning` `Contrastive Learning` `Reinforcement Learning from Human Feedback` `Text` `Sequential`

**🎯 论文内容**

研究并实现了一种基于多目标 on‑policy 蒸馏的生成式推荐方法 KL‑MOPD，能够在每个生成步骤中动态为两个目标教师分配不同的 token‑级权重，从而提升多目标推荐的联合性能。

**💡 创新点**

创新点在于：①识别并解决生成式推荐中出现的 token‑级不平衡（教师学习难度差异）和跨 token 的不一致性；②提出基于初始学生‑教师 KL 差距的权重分配机制，实现对难以蒸馏的教师给予更大关注；③在理论上证明 KL‑MOPD 在加性联合目标上严格优于传统 MOPD；④实现了 Plug‑and‑Play、预计算权重、无额外前向推理的高效方案。

**🔧 技术方法**

使用了 Transformer encoder‑decoder 生成式推荐框架（TIGER/REG4Rec），RQ‑kmeans 离散化 token，基于 KL 反向散度的 on‑policy 蒸馏，动态 token‑级权重调整，预训练 + 教师微调，beam search 推理，DPO / SFT 训练方法。

**📊 数据集**

实验数据集包括：TencentGR、YooChoose、Taobao（公开基准）以及 10 亿级工业广告数据集 Industry；使用的 backbone 有 TIGER 和 REG4Rec。

**📈 对比分析**

与单目标教师、8 个最先进基线（RCE‑KD、MTS^24MM、SmartGR、GR4AD、TSGR、vanilla MOPD、SMOPD、TrustMOPD）进行对比；KL‑MOPD 在单目标平均提升约 4.91%，在联合目标平均提升约 4.12%；在工业离线实验中提升约 3.19%，在 5 天线上 A/B 测试中广告收入提升 1.63%，GMV 提升 2.20%。

**⚠️ 局限性**

局限性包括：①只考虑双目标（两个教师）的情况，理论与实现对多教师场景尚未扩展；②主要针对生成式推荐任务，未验证在更一般的语言任务中的适用性；③权重依赖于初始学生‑教师 KL 差距，若初始学生性能不佳可能影响效果；④对长序列或高维 token 的可扩展性尚未系统评估。

---

## 81. When Black-Box Robot Policies Fail: Online Sensor Diagnosis and Repair with a Transparent Surrogate

**arXiv ID:** 2610.36852 | [PDF](https://arxiv.org/pdf/2610.36852v1)

**作者:** Marc Ziegler `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `Robotics` `Robotic Intelligence` `Anomaly Detection` `Explainability and Interpretability` `Reinforcement Learning` `Contrastive Learning` `Auto Encoder` `Optical Flow` `Fourier-based Model` `Tabular` `Time Series` `Sequential`

**🎯 论文内容**

本文研究在冻结的机器人基础模型（URMA）下，利用结构化代理进行诊断和输入修复，以提升多形态四足机器人在仿真环境中的行走成功率。

**💡 创新点**

创新点在于提出了机制驱动诊断（MBD）与相位引用修复（PRR）结合的无梯度诊断和修复框架，能够仅通过传感器读数和动作输出实现故障定位与修复。

**🔧 技术方法**

使用了 Fourier 基础的周期运动模型、岭回归的误差补偿、分段相位归一化、基于统计阈值的异常检测以及梯度/TreeSHAP 对比分析等技术。

**📊 数据集**

数据集基于 MuJoCo 仿真中的 Go2、Go1 与 ANYmal C 机器人，包含 64k 个状态-动作对和 37 种人工注入的传感器故障（偏移、失效、漂移等）。

**📈 对比分析**

通过与基线相位偏差、梯度归因、TreeSHAP 等方法对比，MBD+PRR 在 640 次 Go2 实验中将行走成功率从 259/640 提升至 429/640（+57%），在线模式提升至 382/640；相比之下基线仅提升至 408/640。

**⚠️ 局限性**

局限在于仅验证单一冻结策略、仿真环境、部分筛选的故障集、缺乏硬件验证，以及无法处理多重同时故障或跨机器人迁移。

---

## 82. Merkle-Tree-Based Verifiable Public Data Auditing for Cloud Object Storage Using Smart Contracts

**arXiv ID:** 2610.36001 | [PDF](https://arxiv.org/pdf/2610.36001v1)

**作者:** Siri Reddy Tadi `[一作]` (Chaitanya Bharathi Institute of Technology), Jaswanth Bhimireddy `[通讯]` (Chaitanya Bharathi Institute of Technology)

**关键词:** `Cryptography and Security` `Data-Centric Learning` `Federated Learning` `Safty and Privacy` `Computational Efficiency` `Explainability and Interpretability` `Merkle Tree` `Hashing (SHA-256)` `Smart Contracts` `Blockchain Verification` `Merkle Path Proof` `Tabular` `Object Storage Data` `Benchmark` `Finance Related`

**🎯 论文内容**

本文设计、实现并评估了基于 Merkle 树的可验证公共数据审计系统，利用智能合约存储根哈希，审计时仅下载并检查挑选对象，能够实时判定对象是否完整、被修改或已缺失。

**💡 创新点**

创新点在于将对象键与 SHA‑256 摘要绑定为 Merkle 树叶子，并通过单根哈希在区块链上形成全局可信、可公开验证的完整性参考，同时实现了按对象三分状态分类（完整、修改、缺失）的本地化检测。

**🔧 技术方法**

采用的技术包括 SHA‑256 哈希、Merkle 树与 Merkle 路径证明、以太坊 Solidity 智能合约、MinIO（或 s3s‑fs）对象存储和 Hardhat 本地以太坊网络。

**📊 数据集**

实验使用合成数据集，包含 10、100、1,000 和 10,000 个对象，单个对象大小从 1 KiB 到 256 KiB 随机生成。

**📈 对比分析**

通过与全数据审计、Merkle 全审计和样本审计对比，Merkl 化后在检测率 100% 的前提下，单对象审计约 10 ms、100 对象样本约 150 ms；链上根哈希的 gas 成本仅为单个哈希表的 1/252，且仅需一次链读即可完成验证。

**⚠️ 局限性**

限制包括需在可信环境下生成根哈希、仅提供完整性验证不含恢复或加密功能、链上读取仍需本地调用、公开网络的交易延迟与 gas 费用未在实验中评估。

---

## 83. FrameForge: Harnessing Priors of Video Foundation Models for Unified Image Generation

**arXiv ID:** 2610.36545 | [PDF](https://arxiv.org/pdf/2610.36545v1)

**作者:** Yijing Lin `[一作]` (University of Science and Technology of China), Yongdong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Mixture of Experts` `Prompt Engineering` `Supervised Fine-Tuning` `Chain-of-Thought` `Image` `Video` `Text`

**🎯 论文内容**

使用预训练的视频生成模型，提出 FrameForge 框架实现统一的条件图像生成；

**💡 创新点**

引入输入自适应的虚拟时间关系，利用 Mixture of Temporal Experts (MoTE) 选取不同时间位置先验，以及 Chain of Temporal Reasoning (CoTR) 的中间隐状态与瓶颈注意力，解决不同条件‑目标关系的多样性；

**🔧 技术方法**

MoTE（多时序专家路由+RoPE）和 CoTR（无监督中间潜变量+瓶颈注意力）配合 LoRA 微调；

**📊 数据集**

三大任务数据集：Subject200K/UNO‑1M + DreamBench（主体驱动），COCO2017（canny‑to‑image），GEdit（指令式编辑）；

**📈 对比分析**

与多种基线对比，FrameForge 在 DreamBench 上 O 得分 0.739（仅次 ChronoEdit），在 COCO2017 上 SSIM 0.447（仅次 EasyControl），在 GEdit 上 G‑O 6.12，均优于或匹配现有最佳开源/闭源方法；

**⚠️ 局限性**

仍受限于固定的预训练视频模型，单一隐状态可能不足以捕获极大变换，且在一些任务上仍落后部分闭源系统；

---

## 84. A Singularity Avoiding Track Model for Race Car Optimization

**arXiv ID:** 2610.36430 | [PDF](https://arxiv.org/pdf/2610.36430v1)

**作者:** Adwait Verulkar `[一作]` (General Motors), Ryan Trinter `[通讯]` (General Motors)

**关键词:** `Robotics` `Optimization` `Autonomous Driving` `Diffusion model` `Contrastive Learning` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation` `Point Cloud`

**🎯 论文内容**

本文提出了一套完整的轨道建模管道，能够将测量得到的赛道点云转化为闭合、C^4连续的 Monge 曲面模型，并在 Modelica 车辆优化中使用。

**💡 创新点**

创新点在于用最小曲率线（minimum‑curvature line）作为曲面参考线，避免传统中心线导致的 Frenet 奇点；同时将整个拟合过程集成为可重复使用的、以 C^4 片段五次样条为输出的统一接口。

**🔧 技术方法**

技术方法包括：基于边界约束的二次规划求解最小曲率线；Delaunay 三角剖分 + RBF 插值重采样；在每纵向截面上对高度进行多项式最小二乘拟合；使用 CasADi/IPOPT 对全轨道进行 C^4 片段五次样条拟合；Modelica MultiBody 组件实现两自由度轨道约束；FMI 3.0 导出供 CasADi/IPOPT 直接求解最短圈时间 NLP。

**📊 数据集**

使用的典型数据集为新加坡 F1 赛道的测量点云（含边界、轨道中心线、垂直高度信息），并以该赛道为测试案例。

**📈 对比分析**

与先前基于中心线的 ribbon‑plus‑Monge 曲面模型相比，本文的管道在最短圈时间 NLP（IPOPT）中实现了显著的数值收敛：对中心线模型出现的双重不可解与内点不适（dual infeasibility）震荡问题，改为最小曲率线后，原本无法收敛的案例能够在相同迭代次数内达到预设容差，且求解时间保持在可接受范围。

**⚠️ 局限性**

局限性包括：对点云需先做唯一且按纵向/横向索引排序；使用的是规则的 (s,w) 网格，可能在高度变化剧烈或宽度不均匀的赛道上精度受限；目前仅处理闭合轨道，未针对开放赛道或非常狭窄的弯道进行专门优化；以及计算时间（约 5 分钟/赛道）相对较高，未来可通过自适应网格和动态参照线进一步改进。

---

## 85. Reasoning about the Genetic Code in Lean

**arXiv ID:** 2610.36343 | [PDF](https://arxiv.org/pdf/2610.36343v1)

**作者:** Colin T. Jones `[一作]` (University of Maryland Baltimore County), Tyler R. Josephson `[通讯]` (University of Maryland Baltimore County)

**关键词:** `Logic in Computer Science` `AI Code Assistant` `Drug Discovery` `Protein Structure Prediction` `Chain-of-Thought` `Large Language Model` `Biomedical Data` `Review/Survey Paper`

**🎯 论文内容**

使用 Lean 4 定义并形式化了分子生物学中央法则中的转录与翻译过程，并对其进行机器验证；

**💡 创新点**

首次在形式化证明框架内证明了遗传密码的冗余性、转录的单射性、逆向转录的可逆性等关键性质，展示了形式化方法在生物学中的可行性；

**🔧 技术方法**

采用 Lean 4 交互式定理证明器、Mathlib4 以及基于列表与递归的函数定义；

**📊 数据集**

无传统生物学数据集，使用自定义的核苷酸与氨基酸类型作为抽象输入；

**📈 对比分析**

通过编写 Lean 证明脚本进行逻辑验证，无运行时性能指标，重点在于证明的可验证性与错误检测能力；

**⚠️ 局限性**

存在语义误差与生物学细节缺失（如剪接、启动/终止密码子规则、病毒逆转录等），模型对不同物种的变体缺乏覆盖，且未考虑实验数据的验证。

---

## 86. Martingale-Based Resource Partitioning Bounds for URLLC and eMBB Network Slicing in 5G

**arXiv ID:** 2610.36602 | [PDF](https://arxiv.org/pdf/2610.36602v1)

**作者:** Hanbing Li `[一作]` (Shandong Normal University), Yantong Wang `[通讯]` (Shandong Normal University)

**关键词:** `Networking and Internet Architecture` `Optimization` `Federated Learning` `Safty and Privacy` `Computational Efficiency` `Time Series` `Benchmark`

**🎯 论文内容**

本文提出了一种基于马氏过程的5G网络切片资源分配框架，能够同时满足URLLC与eMBB的低延迟和高吞吐量需求。

**💡 创新点**

创新点在于利用MGF构造指数马氏过程并应用Doob最大值不等式，得到非渐进的延迟与队列溢出概率上界，从而实现更精确的资源预算。

**🔧 技术方法**

核心技术包括马氏过程理论、Doob最大值不等式、Moment Generating Function（MGF）以及低复杂度的二分搜索与线性搜索算法。

**📊 数据集**

实验使用的主要是仿真数据，基于三十万时隧模拟的网络流量与服务模型，未涉及公开数据集。

**📈 对比分析**

与传统有效容量（Effective Capacity）基准进行比较，结果显示在10^-6可靠性目标下，该方法总共只需44个资源块，较基准的64块降低了31.2%，并在有限资源约束内保持可行。

**⚠️ 局限性**

局限性包括模型假设的独立同分布、对Rayleigh衰落的限定以及在极低概率事件下仍需大量仿真验证，且对更复杂多路况的通用性尚未验证。

---

## 87. Beyond the 1D Reality-Virtuality Continuum: A 2D Framework for Seven Extended Reality Paradigms

**arXiv ID:** 2610.36674 | [PDF](https://arxiv.org/pdf/2610.36674v1)

**作者:** Rajashekhar Vachiravelu Saminathan `[一作]` (Indian Institute of Technology), Rajashekhar Vachiravelu Saminathan `[通讯]` (Indian Institute of Technology)

**关键词:** `Human-Computer Interaction` `Multimodality` `Review/Survey Paper`

**🎯 论文内容**

本文提出了一个二维Reality‑Virtuality（RV）框架，用以描述和比较七种扩展现实（XR）范式：VR、AR、MR、AV、PR、AnR、DR。

**💡 创新点**

创新点在于将传统的单维RV连续体扩展为二维空间，新增信息增减维度，使得不同XR技术在相同的实-虚位置上仍能通过信息操作的方向和幅度得到区分。

**🔧 技术方法**

使用的技术包括虚拟现实（CAVE‑E）、增强现实（QR码导向）、混合现实（软蛇机器人控制）、增强虚拟性（实物探针训练）、消减现实（视觉噪声抑制）、现象现实（机器人通信体验）以及扩增现实（虚拟碎纸路径规划）等七种XR实现。

**📊 数据集**

数据集为在地震灾害场景下设计的七个XR实验实例，包含各技术对应的实验环境、交互方式和视觉呈现。

**📈 对比分析**

比较方法为作者主观的定性交叉范式分析，基于对七种XR实现的体验与观察，识别其在实-虚轴和信息增减轴上的位置；由于缺乏统一测评指标，本文未给出量化性能度量，但通过归纳表明二维框架能更细致地区分AR、DR、AnR等范式。

**⚠️ 局限性**

局限性包括：仅针对地震场景的有限实现，未进行大规模用户实验或量化验证；信息增减轴未能完整捕捉语义或认知效应；框架的普适性和可靠性需通过后续结构化经验研究（如卡片分类、Delphi）进一步验证。

---

## 88. Monitor Accuracy Isn't the Answer: A Decision-Value Framework for Evaluating AI Agent Restart Monitors

**arXiv ID:** 2610.36544 | [PDF](https://arxiv.org/pdf/2610.36544v1)

**作者:** Siddharth Jain `[一作]` (Intuit), Venkat N. Vedam `[通讯]`

**关键词:** `Artificial Intelligence` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Explainability and Interpretability` `Computational Efficiency` `Agentic AI` `Bayesian Decision Theory` `Bootstrap` `Blackwell Experiment` `Clopper-Pearson Interval` `Text` `Benchmark`

**🎯 论文内容**

研究了失败监测器在AI代理中的决策价值，通过分析重启收益、信息保持与统计支持来评估监测器的有效性。

**💡 创新点**

提出了决策充分性判据与基于置信区间的决策审计，说明准确率不是唯一评价标准，并给出了重启机会与决策价值的明确度量。

**🔧 技术方法**

采用了Blackwell实验比较、贝叶斯决策理论、Clopper‑Pearson区间、Bootstrap以及对SWE‑bench验证环境的外部验证等技术。

**📊 数据集**

在SWE‑bench Verified的19个SymPy问题上执行了194次模型驱动实验，并收集了130条已验证轨迹进行审计。

**📈 对比分析**

与固定截断、Luby/几何重启、上下文重启以及失败分支重用等方案对比，实验显示在部分任务上短截断略有提升，但大多数情况下无显著收益，且复现性差。

**⚠️ 局限性**

主要局限包括任务样本有限、仅评估单一模型和单一仓库、每任务仅5次试验、未包含学习型监测器、以及对置信区间方法的先验假设和分布平稳性要求。

---

## 89. count-catcher: Lean Theorems from Model Counting Certificates

**arXiv ID:** 2610.36196 | [PDF](https://arxiv.org/pdf/2610.36196v1)

**作者:** Johannes K. Fichte `[一作]`, Stefan Szeider `[通讯]`

**关键词:** `Logic in Computer Science` `Computational Efficiency` `Explainability and Interpretability` `Optimization` `AI Code Assistant` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Benchmark`

**🎯 论文内容**

提出了一种工具，将CPOG证书转换为Lean定理，证明一个CNF公式在V个变量上有确切的N个模型。

**💡 创新点**

创新点在于将模型计数的结果转化为Lean中的可验证定理，并结合编码引理，形成关于组合对象数量的定理。

**🔧 技术方法**

使用了Lean证明助手，结合了CPOG证书格式和LRAT检查器。

**📊 数据集**

使用了2022年模型计数竞赛中的公式和大约8TB的CPOG证书数据集。

**📈 对比分析**

与CPOG的验证检查器相比，所提出的方法在性能上更快，三项检查的时间约为CPOG验证检查器的四分之一。

**⚠️ 局限性**

限制在于CPOG证书的大小可能过大，导致无法在单个证明项中处理，且当前方法依赖于特定的模型计数器和生成器。

---

## 90. EREBOTS 4.0: A framework for conversational assistants with A2A agents and MCP tools

**arXiv ID:** 2610.36221 | [PDF](https://arxiv.org/pdf/2610.36221v1)

**作者:** Elia Pacioni `[一作]` (HES-SO Valais-Wallis), Michael Ignaz Schumacher `[通讯]`

**关键词:** `Multiagent Systems` `AI Code Assistant` `Autonomous Driving` `Federated Learning` `Optimization` `Reinforcement Learning from Human Feedback` `Robotic Intelligence` `Transformer` `Agentic AI` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

构建了 EREBOTS 4.0 框架，实现了基于 A2A 与 MCP 协议的 LLM 与确定性工具代理的协同对话助手，并提供身份验证、用户上下文、任务跟踪等完整生命周期管理。

**💡 创新点**

创新点包括：①将多代理系统标准化为可互操作的框架，②实现了对官方 A2A 技术兼容性测试套件的完整通过，③通过配置文件即可无代码添加本地或远程代理，④在代理间统一管理身份、同意标志与执行追踪。

**🔧 技术方法**

技术栈：Python、FastAPI、MCP、A2A、OpenAI‑compatible LLM API、SQLAlchemy、PostgreSQL/SQLite、Keycloak OIDC、Docker Compose、Traefik、Vue.js、Gradio。

**📊 数据集**

数据集：示例仅使用内置的单位转换工具和用户资料管理，无外部公开数据集；内部测试采用模拟模型响应和 A2A TCK 预设负载。

**📈 对比分析**

比较方法：在官方 A2A 技术兼容性测试套件中跑 97 条需求，全部通过；内部测试覆盖 567 条单元与集成测试，验证路由、并发、持久化和错误处理；性能指标主要体现在兼容性通过率，未给出吞吐量或延迟数值。

**⚠️ 局限性**

局限性：①仅支持单一排队等待任务；②同意标志仅存储未被强制检查；③任务持久化仅在单实例环境下有效，未实现跨副本同步；④代理间不支持协商或竞标；⑤路由与是否询问用户取决于 LLM，未评估成功率；⑥未对并发负载给出性能基准。

---

## 91. From Structural Feedback to Prompt Policies: Learning Faithful Text-to-Image Prompt Editors

**arXiv ID:** 2610.36146 | [PDF](https://arxiv.org/pdf/2610.36146v1)

**作者:** Wentao Ye `[一作]` (Zhejiang University), Haobo Wang `[通讯]` (Zhejiang University)

**通讯引用:** 11398 | [OpenAlex ID](https://openalex.org/A5100446035)

**关键词:** `Artificial Intelligence` `Generation` `Explainability and Interpretability` `Graph Neural Network` `Transformer` `Prompt Engineering` `Reinforcement Learning` `Supervised Fine-Tuning` `Vision Language Model` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出一种结构感知的文本提示编辑方法，利用校准后的场景图来引导生成器的提示优化。

**💡 创新点**

创新点在于：① 将同一场景图同时用于编辑器的条件化和图像-基准奖励；② 采用置信度保持的奖励和环境感知的GRPO提升对不确定性与生成噪声的鲁棒性；③ 通过局部编辑（删除、重排、插入）实现保守而可解释的提示修改。

**🔧 技术方法**

使用场景图抽取（GLIP + BLIP）、多模态混合编码器（CLIP + GCN）、基于强化学习的编辑策略（GRPO），并配合监督微调。

**📊 数据集**

在Lexica、DiffusionDB以及COCO caption三大测试集上进行评估，训练时使用72K个手工优化提示对，进一步扩展到144K对。

**📈 对比分析**

与Promptist、MagicPrompt、VisualPrompter、GPT重写等方法对比，在美学提升和结构一致性（尤其关系召回）上均实现领先：在Lexica上相对Promptist提升了7.96点关系召回，整体SGR达80.74；在多生成器迁移实验中，仍保持最高的关系召回率；人类评估显示在一致性上+9%至+10%的胜率。

**⚠️ 局限性**

局限性包括：依赖解析器、定位与生成器质量；场景图对计数、否定、情感等抽象意图表达有限；编辑策略过于保守时难以完成全局重构；训练需多图渲染，成本较高；仅在英文提示上验证，跨语言泛化未知。

---

## 92. Design and Architecture of Airmod: An Active Noise Cancellation and Deep Learning System for Real-Time Respiratory Rate Estimation from Tracheal Sounds

**arXiv ID:** 2610.37083 | [PDF](https://arxiv.org/pdf/2610.37083v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Sound`

---

## 93. Who Convinced Whom? Tracing How LLMs Change Their Minds in Multi-Agent Systems

**arXiv ID:** 2610.35930 | [PDF](https://arxiv.org/pdf/2610.35930v1)

**作者:** Arnesh Batra `[一作]` (Indraprastha Institute of Information Technology), Phan Xuan Tan `[通讯]` (Shibaura Institute of Technology)

**通讯引用:** 528 | [OpenAlex ID](https://openalex.org/A5073937678)

**关键词:** `Multiagent Systems` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `Benchmark` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Benchmark`

**🎯 论文内容**

本文通过设计 MAD Bench（2400 问题的综合基准）和匹配的独立反思对照，系统评估多代理辩论（MAD）对语言模型决策的影响，深入分析了从同伴信息中获得的修正、推理贡献、源信息权重、注意力路径以及对最终准确率和错误一致性的影响。

**💡 创新点**

创新点在于：①提出 MAD Bench 及匹配的独立反思实验框架，能将同伴交流与额外推理时间分离；②通过“在推理前强制回答”与“信息拆分”实验量化同伴推理对决策的主要贡献；③在源信息层面引入可靠性与身份标签，揭示可靠性比身份更具影响力；④使用注意力拆除与残差互换两种内部干预，证明混合模型中存在非注意力的影响通路；⑤提出有效独立投票数（n_eff）衡量合作后的独立性下降。

**🔧 技术方法**

技术包括：多轮多代理辩论协议、匹配独立反思控制、决策词读取、回答与推理分离、同伴信息按答案/推理拆分、源信息标签插入、注意力掩码与全注意力拆除、残差流与嵌入层互换、统计分析（bootstrap、permutation、Benjamini–Hochberg）。

**📊 数据集**

数据集为 MAD Bench，包含 3 大类（知识与推理、价值与道德判断、安全与拒绝），共 2400 个文本问题，其中 1700 个有参考答案。该基准融合了 GSM8K、MATH-500、MMLU、Ethics、AdvBench 等公开数据。

**📈 对比分析**

对比方法：在三模型（Qwen3.5‑9B、Llama‑3.1‑8B、Gemma‑4‑E4B）下，比较辩论、独立投票、匹配反思三种决策。结果显示：相较于初始投票，辩论提升约 1%（Qwen、Gemma）或无显著差异（Llama）；相较于反思，辩论对错误一致率提升 3.1–7.4个百分点；在 12 轮辩论中，三方有效独立投票数降至约 1，而反思下为 1.7–2.1。

**⚠️ 局限性**

局限性包括：仅评估 3 个 8–10B 开源模型，未覆盖更大/闭源模型；仅使用英文文本基准，缺少多语言或其他任务；仅实现 3 轮常规辩论（12 轮仅 600 问题子集）；实验仅针对单一辩论协议，未考察其他拓扑或更长交互。

---

## 94. Integrated Design and Reproducible Feasibility Analysis of a Four-Wheel Rover with Packet-Commanded Imaging, LiDAR Ranging, and Learning-Assisted Target Recognition

**arXiv ID:** 2610.36816 | [PDF](https://arxiv.org/pdf/2610.36816v1)

**作者:** Sabik Bin Sultan `[一作]` (BAF Shaheen), Maria Sultan `[通讯]` (South Point School)

**关键词:** `Robotics` `Recognition` `Contrastive Learning` `Supervised Fine-Tuning` `Image`

**🎯 论文内容**

本文通过对一台四轮教育型遥控星际车的机械、电气、软件及传感器体系进行建模与分析，提出了一套可复现的设计评估与验证流程，涵盖尺寸、质量、动力、悬挂、传感器几何、指令执行及安全保护等方面，并用仿真与合成数据验证目标识别算法。

**💡 创新点**

创新点在于：1）将组件选型与失效机制、数值设计极限、可复现验证任务形成可追踪的设计链；2）使用独立悬挂与双电源架构分离动力与任务电流，避免电气干扰；3）提供完整的指令状态模型和硬件无关的软件验证脚本；4）利用合成数据构建场景离散的目标识别基准，并对比几种特征+SVM方法。

**🔧 技术方法**

技术包括：差分驱动与闭环混合指令映射、独立轮载体与悬挂弹簧/阻尼、Raspberry Pi 5任务子系统、单束LiDAR、舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机舵机

**📊 数据集**

合成目标板裁剪图像数据集：共2400张 64×64 RGB 低分辨率裁剪图像，按情景组划分为训练（1200张）、验证（400张）、正式测试（400张）和偏移测试（400张），每组包含四类目标（红圆、蓝方、绿三角、无目标），并且测试组与训练组完全分离。

**📈 对比分析**

对比三种识别方法：1）HSV+几何阈值基线；2）HSV直方图+线性SVM；3）HOG+HSV直方图+线性SVM。基线在正式测试上准确率 95%（宏F1 0.951），偏移测试上 71.75%（宏F1 0.734）。学习方法在正式测试准确率约 82%（宏F1 0.789），偏移测试准确率降至 61.75%（宏F1 0.584）和 48.5%（宏F1 0.478）。说明基线方法在两种测试集上均优于学习模型，学习模型对背景和噪声的鲁棒性差。

**⚠️ 局限性**

主要限制：1）未进行真实星际车硬件试验，所有分析基于CAD、理论曲线和合成数据；2）缺乏组件性能测量（电机扭矩/电流、悬挂弹簧/阻尼、轮胎强度等）；3）未测量真实相机标定、LiDAR与相机的外参及中心质量；4）指令执行验证仅在软件模拟环境下完成，未验证无线接收、低功率保护等硬件可靠性；5）目标识别仅在裁剪图像上测试，未评估完整图像定位与检测；6）设计基于理想化假设（滚阻、滑移、光照），不反映实际土壤和气象条件。

---

## 95. When Is a Cheap Fairness Edit Safe? Screening Post-Hoc Edits in Two-Sided Matching Markets

**arXiv ID:** 2610.36649 | [PDF](https://arxiv.org/pdf/2610.36649v1)

**作者:** Moirangthem Tiken Singha `[一作]`, Rabinder Kumar Prasada `[通讯]`

**关键词:** `Machine Learning` `Optimization` `Recommendation System` `Federated Learning` `Safty and Privacy` `Explainability and Interpretability` `Graph Neural Network` `Transformer` `Contrastive Learning` `Auto Encoder` `Tabular` `Graph` `Review/Survey Paper`

**🎯 论文内容**

研究在不重新训练模型的情况下，利用闭式线性投影在已部署的两侧匹配系统中抑制受保护属性的影响，并给出匹配质量损失的可预估界限；

**💡 创新点**

提出了在保持模型不变的前提下可调整、闭式、并且能够在推理时评估公平性与福利损失的后置干预方法，并证明在垂直排序市场下可给出确定性与概率性福利上界；

**🔧 技术方法**

使用线性投影（类似LEACE）对编码器输出进行白化与方向消除，结合Deferred Acceptance匹配机制，配合概率理论与秩统计推导福利界限；

**📊 数据集**

在合成垂直市场（含低/高特征偏差设置）和三类真实数据（美国法学院录取、巴西全国考试、Speed Dating 评分市场）上进行实验，编码器分别为矩阵分解、双塔网络和图神经网络；

**📈 对比分析**

与迭代零空间投影、RLACE、分数标准化与拒绝选项等后置干预基线对比，采用阈值裁定的屏蔽规则，结果显示所提方法在合成实验中完全通过预设判定且能在不牺牲福利的前提下显著缩小属性差距；在真实数据上，闭式因子化编码器在90‑100%实例内保持“graded”层级，而训练网络则被判为超出范围；

**⚠️ 局限性**

局限在于仅适用于平衡一对一完全市场且需要两侧共享同一排序准则的垂直或近垂直结构；对非垂直偏好、容量限制、未匹配或不完整偏好列表的情况无理论保证；此外，概率界限依赖交换性假设，且只对单一方向的线性编辑给出分析，无法覆盖多方向或非线性模型微调。

---

## 96. CONCUR: Concordant Auxiliary Rewards for Multi-Task, Multi-Reward Reinforcement Learning

**arXiv ID:** 2610.36437 | [PDF](https://arxiv.org/pdf/2610.36437v1)

**作者:** Jixuan Leng `[一作]` (Carnegie Mellon University), Inderjit S. Dhillon `[通讯]` (Google)

**关键词:** `Computation and Language` `Reinforcement Learning` `AI Code Assistant` `Transformer` `Reinforcement Learning` `Prompt Engineering` `Large Language Model` `Text` `Code` `Instruction` `Writing`

**🎯 论文内容**

本文提出一种名为 ConCUR 的优势估计器，利用投影方式将多任务、多奖励设置中的门控辅助奖励（gated auxiliary）按主奖励的顺序和上限约束进行调整，以保证在同一组样本中主奖励始终决定响应的优先级。

**💡 创新点**

创新点：
- 通过“顺序约束（order constraint）”和“上限约束（ceiling constraint）”两种投影，系统性地扩展门控机制，使辅助奖励在任何响应对中都不会逆转主奖励的排序。
- 引入两种变体：默认 T 始终保持上限约束，适配性 A 通过“earned trust”动态决定是否解除上限约束，兼顾不同任务的特性。
- 证明投影在保持主奖励顺序、最小化能量损失的同时，可在保持辅助奖励对通过响应的区别贡献的前提下，提升整体奖励效果。

**🔧 技术方法**

技术：
- 基于组相对优势（group-relative advantage）框架的 GDPO（Group-Relative DPO）算法。
- 投影实现使用等价的“pool‑adjacent‑violators”算法（即等距回归）对每个组内的辅助奖励进行顺序与上限约束投影。
- 通过“earned trust”统计每个门控配对的同序比例，在满足阈值后动态解除上限约束。
- 训练采用 300 步的 512 个 prompt 批次，学习率 1e‑6，并在更新中剔除所有奖励相同的组。

**📊 数据集**

数据集：
- 训练与评估任务混合包含四类任务：数学（Math）、代码（Code）、指令遵循（IF）和写作（Writing）。
- 评估基准涵盖 8 个官方评测集：MATH‑500、OlympiadBench、HumanEval+、LiveCodeBench v6、GuideBench、WritingBench、Creative Writing v3、LongBench‑Write。

**📈 对比分析**

对比方法与性能：
- 对比 7 种现有优势估计器（GRPO、GDPO、GD²PO、GEOMA、SAW、MT‑GRPO、PAPO）以及一个基准（GDPO 丢弃所有辅助奖励）。
- 在三种不同模型（如 Qwen‑系列、Llama‑系列等）上，ConCUR（T 与 A 两种）在所有模型的平均分（Avg）上均超过 GDPO 四分以上，并且在每个模型中均为最佳或第二最佳。
- 具体表现：在 Math 任务上提升 4+ 分，在 Writing 任务上提升 3–5 分；在其他任务（Code、IF）提升幅度一般为 1–2 分。
- 进一步分析表明：
  * 仅顺序约束（-O）能显著提升 Writing 任务；
  * 仅上限约束（-T）对 Math 任务效果最佳；
  * 两种变体差异不大，平均分差距不到 1 分。

**⚠️ 局限性**

局限性：
- 仅在单一四任务混合上验证，未在更大规模或多样化任务集合上测试。
- 门控辅助仅限于四个长度奖励和一个 Reward‑Model（RM）奖励，无法评估对其他类型门控奖励的适用性。
- 只使用单一中间权重（权重 1）进行实验，未探究权重对投影效果的影响。
- 评估只基于训练轮次的奖励分布，未直接测量辅助奖励在正式测试集上的收益。
- 顺序约束和“earned trust”策略均为设计选择，若任务设计者允许辅助奖励优先于主奖励的小差异，则方法可能不合适。

---

## 97. Counterfactual Profile Perturbations for Recommendation Explanations: A Reproducibility Study

**arXiv ID:** 2610.36208 | [PDF](https://arxiv.org/pdf/2610.36208v1)

**作者:** Meysam Varasteh `[一作]` (University of Colorado), Robin Burke `[通讯]` (University of Colorado)

**关键词:** `Information Retrieval` `Recommendation System` `Explainability and Interpretability` `Matrix Factorization` `Variational Autoencoder` `Learning to Explain` `Counterfactual Perturbation` `Coverage Evaluation` `MPNR Metric` `RBO Evaluation` `Tabular`

**🎯 论文内容**

本研究扩展并评估了Learning to eXplain Recommendations (LXR) 方法，使其能解释推荐列表中前10名物品，并与多种基线进行对比。

**💡 创新点**

创新点在于：①将LXR训练范围从单一Top‑1扩展到Top‑10；②引入了基于项目流行度与矩阵分解嵌入相似度的新解释器；③系统性评估了不同推荐模型与数据集下的覆盖率与最小扰动大小。

**🔧 技术方法**

采用的技术包括：学习型反事实扰动 (LXR10)、矩阵分解与VAE推荐器、项目流行度基线、嵌入相似度基线（LF）、RBO 评价指标、覆盖率与MPNR 等量化指标。

**📊 数据集**

使用了三大公开数据集：MovieLens‑1M、Yahoo Music 以及 Pinterest，分别构建 MF 与 VAE 两种推荐器。

**📈 对比分析**

在 Top‑1 与 Top‑10 任务中，LXR10 在大多数场景下覆盖率高、MPNR 低；LF 在 MF 推荐器上表现尤为突出；相较原 LXR，LXR10 在覆盖率上略有提升，MPNR 亦略低，说明其更有效；但所有方法均存在较大的扰动大小。

**⚠️ 局限性**

主要局限：①扰动往往包含超过5个项目，用户可读性不足；②对流行项目高度依赖导致解释效果不稳定；③未进行用户体验评估；④寻找最优扰动的搜索复杂度高，实际应用难度大；⑤不同解释器产生的扰动不唯一，难以统一衡量。

---

## 98. What Should an Information Bottleneck Encoder Learn? Separating Representation Learning from Rate Control

**arXiv ID:** 2610.36255 | [PDF](https://arxiv.org/pdf/2610.36255v1)

**作者:** Hassan Tavakoli `[一作]` (Oregon State University), Bella Bose `[通讯]` (Oregon State University)

**关键词:** `Machine Learning` `Representation Learning` `Compression` `Variational Information Bottleneck (VIB)` `Information Bottleneck Theory` `Rate-Conditioned Networks` `Image`

**🎯 论文内容**

论文研究信息瓶颈(IB)在给定压缩预算β下的表示分离问题，提出冻结表示、按预算训练压缩-解码器对的分离管道，并证明在充分表示下可实现整个IB路径；

**💡 创新点**

创新点在于：①推导出非高斯情形下的表示-速率分离理论并给出充分性缺陷ϵ_F与IB差距的统一上界；②将这一理论应用于分离式VIB训练，显著降低训练成本并在高压缩下提升准确率；

**🔧 技术方法**

技术方法包括：信息瓶颈理论、Gaussian IB 解析分解、变分信息瓶颈(VIB)、率条件网络、两阶段训练（先冻结表示，再按预算训练压缩器与解码器）以及理论证明中的Pinsker与熵不等式；

**📊 数据集**

实验使用四个视觉数据集：MNIST、SVHN、CIFAR‑10、STL‑10，并在单域与混合域两种设定下进行评估；

**📈 对比分析**

与传统独立训练的VIB、FiLM‑条件VIB以及OIB进行对比，结果显示分离管道在所有预算下平均准确率提升约8–10个百分点，训练MAC数减少7.7倍、运行时间缩短约2倍；在大压缩（小λ）时表现更优；但在小压缩下交叉熵和ECE略高；

**⚠️ 局限性**

局限性包括：①对低压缩率时的校准与交叉熵仍不理想；②需要为每个预算单独训练压缩-解码器；③充分性缺陷理论上限与实际误差之间的闭合不够紧；④方法假设β已知且不做优化，未考虑预算自适应；

---

## 99. The Vocabulary Is Already Here: Adapting Pretrained Interpretable Features for Generative Recommendation

**arXiv ID:** 2610.36763 | [PDF](https://arxiv.org/pdf/2610.36763v1)

**作者:** Ziluowen Luo `[一作]` (Central South University), Senzhang Wang `[通讯]` (Central South University)

**关键词:** `Information Retrieval` `Recommendation System` `Explainability and Interpretability` `Transformer` `Auto Encoder` `Large Language Model` `Contrastive Learning` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

设计了一种可解释的生成式推荐框架（API），利用预训练稀疏自编码器（P‑SAE）中的可解释特征词典直接构造项的可解释标识符（PID），并通过语义预测适配器（SPA）学习用户偏好，从而实现无额外词典学习的生成式推荐和检索解释。

**💡 创新点**

创新点在于：①将预训练稀疏自编码器的可解释特征直接作为生成式推荐的词典，消除词典学习成本；②提出 PID 机制，既保留激活强度又降低同类特征冲突；③通过 SPA 结合特征索引和强度预测，并引入基于候选项的偏好对齐，实现更精准的检索和可解释性；④提供完整的特征贡献分解，能够解释推荐决策。

**🔧 技术方法**

核心技术包括：预训练稀疏自编码器（P‑SAE）及其特征词典、基于 Transformer 的生成式推荐模型（T5、GPT‑2）、PID 构造（特征优先级、激活强度归一化）、SPA（历史编码、特征索引与强度预测、基于 Hellinger 距离的评分）、以及检索解释的特征贡献分解。

**📊 数据集**

使用 Amazon Beauty、Sports、Toys 三个子集的文本描述（标题、品牌、分类、价格）作为实验数据集，分别采用训练、验证、测试拆分；评估指标包括 Recall@5/10、NDCG@5/10、碰撞率、Overlap@20 等。

**📈 对比分析**

在多种词典化方法（Random、OPQ、HiD‑VAE、RVQ、RQ‑VAE、RKMeans）和两种生成式解码策略（AR、PR）下，API 在 12 组数据‑指标组合中获得 9 组最佳结果；在多项指标上优于基线，尤其在 Toys 领域 Recall/ NDCG 领先；PID 显著降低碰撞率并提升内容保持度；SPA 的偏好对齐和强度预测对检索性能贡献明显。

**⚠️ 局限性**

局限性包括：①依赖于预训练 P‑SAE 词典，若目标域与预训练域差异较大，可能影响词典覆盖；②对文本描述质量敏感，非文本信息难以直接利用；③PID 仍受限于固定的特征数量，极端稀疏/多样的商品可能无法完全表达；④在大规模实时系统中，生成式解码和特征贡献分解的计算成本仍需进一步优化。

---

## 100. Concorde: Geometry-Aware Link Prediction via Decoupled Energy Minimization

**arXiv ID:** 2610.36204 | [PDF](https://arxiv.org/pdf/2610.36204v1)

**作者:** Ege Demirci `[一作]` (University of California Santa Barbara), Ambuj Singh `[通讯]` (University of California Santa Barbara)

**关键词:** `Machine Learning` `Retrieval` `Optimization` `Graph` `Representation Learning` `Graph Neural Network` `Contrastive Learning` `Diffusion model` `Score-based Model` `Auto Encoder` `Generative Adversarial Network` `Supervised Fine-Tuning` `Graph` `Tabular` `Time Series`

**🎯 论文内容**

设计并实现了一种能量式链接预测框架 Concorde，将链接概率拆分为三条结构上解耦的分支（谱锚点、混合曲率残差、非局部向量场）并通过学习的门控进行融合，采用 N‑pair 损失训练，专注于极度稀疏图的早期检索任务。

**💡 创新点**

创新点包括：① 理论证明谱过滤与自同构破坏不可共存，提出谱锚点保持全局谱不变；② 引入可学习的混合曲率产品空间 ℍ×𝕊 与特征感知 Ollivier–Ricci 辅助，以在局部捕获对称破坏；③ 设计在双曲流形上直接学习向量场的非局部几何对齐机制，突破传统注意力在曲率不一致时的限制；④ 用门控实现结构解耦的能量叠加，避免了单一空间过度拟合。

**🔧 技术方法**

技术手段包括谱扩散 (M^k‑1π^⊤)、混合曲率嵌入与距离计算、Sinkhorn 近似 Ollivier–Ricci 计算、Lorentz 模型下的指数映射与向量场转化、N‑pair 排序损失、全负样本训练、以及基于张量/矩阵操作的高效实现。

**📊 数据集**

使用五个标准基准：Cora、CiteSeer、PubMed（有属性的社交/学术网络），OGBL-DDI（高密度无属性药物互作网络），以及极稀疏的电网网络 Power，覆盖稀疏/密集、属性/无属性等多种结构情景。

**📈 对比分析**

与包括传统重叠度量、GNN、子图方法、Hyperbolic GNN、Gelato 等在内的多种基线进行对比，Concorde 在早期检索指标 P@10%/P@20% 上实现了显著提升，尤其在稀疏图 Power 上近乎翻倍（~27.6% 对比 13.4%），在密集无属性图 OGBL-DDI 上匹敌甚至略优于最佳重叠方法（46.45% 对比 46.12%），整体保持或超过其他方法的 AP 水平。

**⚠️ 局限性**

局限性包括：① Ollivier–Ricci 通过 Sinkhorn 迭代计算成本较高，且在极大规模时仍需预计算；② 门控权重是全局均匀的，无法自适应处理结构异质子图；③ 仅优化早期检索排序，导致与校准性更强的基准在 AP 上略有落后；④ 对图的增量更新与跨图迁移的鲁棒性尚未验证。

---

## 101. BUSTA: Teaching Small Language Models to Reason About Binary Execution

**arXiv ID:** 2610.36288 | [PDF](https://arxiv.org/pdf/2610.36288v1)

**作者:** Ritika Poddar `[一作]` (MITRE Corporation), Michael Doyle `[通讯]` (MITRE Corporation)

**关键词:** `Cryptography and Security` `Reinforcement Learning from Human Feedback` `Computational Efficiency` `Representation Learning` `Knowledge Distillation` `Explainability and Interpretability` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Prompt Engineering` `Low-Rank Adaptation` `Sequential`

**🎯 论文内容**

Fine-tune open-weight LLMs on an inverse basic‑block execution task，以输入寄存器状态预测给定输出寄存器状态。

**💡 创新点**

首次将强化学习（GRPO）与低秩适配（LoRA）相结合，在专门的逆向工程任务上进行后训练，并通过标注“可知”/“不可知”寄存器来提升模型对低级二进制行为的推理能力。

**🔧 技术方法**

使用 Group Relative Policy Optimization（GRPO）强化学习、Low‑Rank Adaptation、Pydantic prompt schemas、Ghidra disassembly 与 Unicorn 模拟器。

**📊 数据集**

构建自 Assemblage 100 条 Windows PE 二进制文件的数据集，生成 287,137 个基本块并采样 282,741 条有效样本。

**📈 对比分析**

与原始开放权重模型和多种 frontier API 模型在有效率、寄存器覆盖率、可知寄存器准确率、不可知分类准确率等指标下进行对比；微调模型在可知寄存器准确率上提升 12.1%（BUSTA‑3B）/ 31.9%（BUSTA‑14B），逼近 frontier 99.2%，同时显著降低 token 使用量。

**⚠️ 局限性**

仅基于 100 条二进制文件，数据集规模有限；奖励函数设计可能仍需改进；实验仅覆盖单个基本块，未扩展至多块序列或完整程序；未在真实受限环境中验证部署效果。

---

## 102. UniPCH: A Unified Framework for Cross-Chain Payments via Payment Channel Hubs

**arXiv ID:** 2610.36866 | [PDF](https://arxiv.org/pdf/2610.36866v1)

**作者:** Yuanming Shao `[一作]` (Harbin Institute of Technology), Zejun Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Federated Learning` `Zero Knowledge Proof` `Blind Signature` `Adaptor Signature` `Re-signature` `Finance Related`

**🎯 论文内容**

本文提出了 UniPCH 框架，基于支付渠道中心（PCH）实现跨链私密支付，并通过两层架构将本地通道执行与跨链支付逻辑解耦。

**💡 创新点**

创新点包括统一的 UniPC 通道抽象、选择性重签名盲条件签名（S‑RBCS）用于跨中心条件授权、以及能够支持异构区块链的通用接口，从而实现兼容性、无缝集成和攻击抵御。

**🔧 技术方法**

使用技术包括盲随机化签名、线性唯一加密、适配器签名、重签名适配器签名、零知识证明、以及 Go 语言实现的通道协议；基于 secp256k1、BLS12‑381 与 Pedersen 绑定的加密和签名方案。

**📊 数据集**

实验数据集为 6 条链（3 条 Bitcoin regtest、3 条 Ethereum Ganache）下的 50 名用户、每人 5 次支付；并与 A^2L+、P2C2T 与 XPCH 进行对比。

**📈 对比分析**

在对比实验中，UniPCH 在运行时间与带宽上均优于基线方案，尤其是账户实现与选择性刷新可使时延降低 20%–30% 以上，网络延迟实验亦显示其对延迟更具鲁棒性。

**⚠️ 局限性**

局限性包括：需要在所有中心之间维护全连接通道网络导致通道数与中心数平方增长；UTXO 实现的证据刷新在高并发场景下仍会产生额外计算和通信开销；目前仅提供 UTXO 与账户两种实现，扩展到其他区块链模型仍需开发对应实现。

---

## 103. Translating Poetic Emotion into Dynamic Mathematical Visuals: A Strange Attractor-Based Approach

**arXiv ID:** 2610.36698 | [PDF](https://arxiv.org/pdf/2610.36698v1)

**作者:** Subhadra Nayak `[一作]` (Indian Institute of Technology), Subhadra Nayak `[通讯]` (Indian Institute of Technology)

**关键词:** `Human-Computer Interaction` `Generation` `Representation Learning` `Image Translation` `Transformer` `Supervised Fine-Tuning` `Diffusion model` `Vision-Language Model` `Generative Adversarial Network` `Contrastive Learning` `Text` `Image` `Video`

**🎯 论文内容**

开发了将诗歌情感通过预训练情感分类器提取后，映射到奇异吸引子参数，进而生成粒子动态可视化的完整管道。

**💡 创新点**

创新在于将细粒度情感标签映射到数学奇异吸引子的参数，实现可预测且动态的视觉表现，并构建完整的情感‑参数‑可视化映射框架。

**🔧 技术方法**

使用RoBERTa情感分类器（基于GoEmotions），TouchDesigner结合GLSL粒子模拟，奇异吸引子模型（Thomas、Lorenz、Chen‑Lee、Halvorsen）以及HSV与Bloom等视觉处理技术。

**📊 数据集**

使用GoEmotions数据集（约58,000条Reddit评论）对RoBERTa模型进行微调，以获取细粒度情感标签。

**📈 对比分析**

通过小规模（8人）pilot study，比较诗歌情感与对应可视化情感的匹配率，得到约23%的一致率，说明系统在情感映射与视觉表达上尚需改进。

**⚠️ 局限性**

局限包括样本量极小、情感分类模型倾向中性、情感‑参数一对一映射缺乏整体性、评估方式过于简化，缺乏开放式或统计化的感知评估。

---

## 104. GAWM: A Compact Latent World-Action Model for Multi-Task Robot Manipulation

**arXiv ID:** 2610.37091 | [PDF](https://arxiv.org/pdf/2610.37091v1)

**作者:** Yang Liu `[一作]` (Genisom AI), Xiang Gao `[通讯]` (Genisom AI)

**关键词:** `Robotics` `Robotic Intelligence` `Computational Efficiency` `Representation Learning` `Transformer` `Auto Encoder` `World Model` `Vision-Language-Action Model` `Contrastive Learning` `Reinforcement Learning` `Video` `Image`

**🎯 论文内容**

提出一种紧凑型世界-动作模型（Geometry-Aware World Model），通过任务条件的潜在预测器在冻结的 DINOv3 编码器的潜在空间中预测未来两个时隙的视觉令牌，然后用 ACT 风格的动作解码器直接回归动作片段，构成一种无搜索、实时的多任务机器人操控策略。

**💡 创新点**

创新点包括：①仅用 100M 级别参数，完全不依赖大规模 VLM 或视频生成器；②使用任务代码而非动作/状态驱动的预测器，实现对任务未来场景的“几何感知”预测；③在潜在空间中并行预测两个未来时隙，并用冻结编码器的 patch 特征作为稠密监督，避免表示坍塌；④将预测结果直接输入动作解码器，消除传统规划搜索，显著降低推理延迟。

**🔧 技术方法**

核心技术：冻结 DINOv3 ViT-S/16 编码器 + 64 令牌适配器；任务代码 MLP；双向 Transformer 预测器（并行两 horizon）；VICReg 风格正则化（div 与 var）及未来监督 patch 解码器；ACT-Style 动作解码器；训练目标为动作 ℓ1 + 未来/当前监督 + 正则化。

**📊 数据集**

使用的数据集：LIBERO（Standard 1,693 轨迹与 Extended 6,585 轨迹，四个任务套件）；RoboTwin 2.0（50 双臂任务，清洁与随机化 50/500 轨迹）；真实机器人实验（Tianji Marvin Pro 双臂，四个瓶子任务，共 340 轨迹）。

**📈 对比分析**

与 VLAs、WAMs 及其他紧凑策略对比：在 RoboTwin 2.0 上以 77M 参数取得 93.94%（清洁）/93.10%（随机）高于 10B 参数 WAMs；在 LIBERO 上以 38M 参数获得 95.6%（与 3.2B 参数 π_0 相比 1.4% 低），超过 0.48B LiLa-WAM；推理延迟最低（8.8 ms/片段 vs 63–140 ms 其他紧凑模型）。

**⚠️ 局限性**

局限性：仅适用于已知的封闭任务集合，需预先提供任务代码；不支持自然语言指令；对任务外或大规模多样性任务的泛化尚未验证。

---

## 105. MVRCap: Towards Comprehensive Video Captioning via Hierarchical Multi-Dimensional Verifiable Rewards

**arXiv ID:** 2610.36838 | [PDF](https://arxiv.org/pdf/2610.36838v1)

**作者:** Weihuang Lin `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Reinforcement Learning` `Vision Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Video` `Text` `Multimodality`

**🎯 论文内容**

提出MVRCap框架，通过强化学习与细粒度可验证奖励实现全面视频字幕生成。

**💡 创新点**

创新在于构建六维结构化注释范式、MVRCap‑SFT与MVRCap‑QA数据集，以及层次化概念覆盖、事实核验与简洁约束三大奖励机制。

**🔧 技术方法**

采用结构化视频理解MLLM、Qwen2.5‑VL‑7B基模型、GRPO策略优化、冻结Judge LLM评估以及多级问答奖励。

**📊 数据集**

使用自研的MVRCap‑SFT（4K视频‑字幕，六维结构）和MVRCap‑QA（205K问答对）数据集，训练基线后进行RL微调。

**📈 对比分析**

在VC4VG、CAPability与VDC三大基准上，MVRCap‑32B/8B模型均超过LLaVA‑Video‑Gen、Gemini系列等，对标开源模型取得最高All F1≈55.18%和VDC Accuracy≈49.7%。

**⚠️ 局限性**

局限在于仍依赖大规模MLLM与算力，RL对初始结构敏感，且在极端动态情境下可能产生少量事实偏差。

---

## 106. CRLoMTL: Conflict-Resolution through Low-rank matrices for Multi-Task Learning

**arXiv ID:** 2610.36337 | [PDF](https://arxiv.org/pdf/2610.36337v1)

**作者:** Ahmed Agiza `[一作]` (Brown University), Sherief Reda `[通讯]` (Brown University)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Depth Estimation` `Classification` `Transformer` `LoRA (Low-Rank Adaptation)` `SVD (Singular Value Decomposition)` `Multi-Task Learning` `Image`

**🎯 论文内容**

提出了CRLoMTL，一种参数高效的多任务学习框架，利用任务特定LoRA更新和共享子空间融合来解决任务间冲突；

**💡 创新点**

创新点在于：1）在梯度空间之外通过低秩参数更新缓解冲突；2）采用多轮训练（任务特定更新+全局SVD融合），生成单一可直接合并到backbone的更新；3）不需要运行时多任务分支，训练结束后可得到单模型；

**🔧 技术方法**

使用LoRA低秩适配、SVD子空间resolver、Swin Transformer backbone、梯度裁剪与多pass训练；

**📊 数据集**

在PASCAL-Context、NYUD‑v2（语义分割+深度估计）以及CelebA（40属性分类）三个数据集上进行实验；

**📈 对比分析**

与MTI‑Net、MTLoRA、DWA、PCGrad、CAGrad、Nash‑MTL、MGDA、IMTL‑G、FAMO、FairGrad、Go4Align、ConsMTL、Shared‑LoRA、KnOTS等SOTA方法在Swin‑Tiny/S‑Small上比较。CRLoMTL在PASCAL‑Context实现+2.77%（Tiny）/+6.08%（Small）Δm，在NYUD‑v2 +5.03%Δm，在CelebA +3.5%Δm，同时仅使用单任务参数的4–6%；

**⚠️ 局限性**

局限性：训练需要多轮任务特定和全局pass，导致一定计算开销；对SVD子空间维度与LoRA rank的选择敏感；在任务数大幅增加时SVD占用时间上升；目前仅验证于Swin Transformer，跨backbone推广需进一步验证；

---

## 107. Differentially Private Preconditioned Power Iteration Method for Principal Component Analysis that Adapts to the Coherence

**arXiv ID:** 2610.36691 | [PDF](https://arxiv.org/pdf/2610.36691v1)

**作者:** Duy V. Le `[一作]` (Boston University), Huy L. Nguyen `[通讯]` (Northeastern University)

**关键词:** `Data Structures and Algorithms` `Federated Learning` `Safty and Privacy` `Privacy mechanism (Differential Privacy)` `Matrix Decomposition` `Gaussian Mechanism` `QR Decomposition` `Power Iteration` `Clipping Technique` `Adaptive Hyperparameter Selection`

**🎯 论文内容**

提出了一种基于预条件化幂迭代的不同ially private PCA算法，该算法在行级ℓ₂ 隐私模型下，对输入矩阵的低一致性（coherence）具有自适应的误差保证。

**💡 创新点**

创新点包括：① 在保持同一秩k的情况下将误差从最坏情况的 O(√d/ϵ) 降至与矩阵一致性相关的更小量；② 使用预条件化和剪切（clipping）技术在每一步迭代中控制对单行的敏感度；③ 无需事先知道一致性或特征值间隙，通过私有超参数选择方法自动确定剪切阈值 R 与迭代次数 T；④ 通过马氏过程分析实现噪声累积的有效抵消。

**🔧 技术方法**

核心技术包括：隐私高斯机制 + 先进组合；预条件化幂迭代（block power method）配合逆平方预条件器；对每行向量的剪切操作；矩阵分解与 QR 分解；马氏过程与高斯集中理论用于噪声与误差分析。

**📊 数据集**

无具体数据集，论文为理论研究，未在实验数据上验证。

**📈 对比分析**

与前人（如 Analyze Gauss、entry‑level 隐私模型下的方案、k‑PCA 的重复 deflation 等）进行比较。该算法在低一致性数据上误差显著优于现有方法；当一致性不低时，仍保持与 Analyze Gauss 相近的性能。实验/理论表明误差约为 O(dkσ_k²‖U‖∞² / (ϵ²Δ_k²))，比最坏情况的 O(√d/ϵ) 小许多。

**⚠️ 局限性**

局限性包括：① 需要矩阵满足正向特征值间隙 Δ_k>0；② 预条件化与剪切阈值的设定虽然通过私有选择实现，但实现复杂；③ 对高维度大 k 的计算成本相对较高；④ 目前仅在行级 ℓ₂ 隐私模型下证明，未直接推广到其他隐私模型。

---

## 108. Who Knows What - and What Does It Support? Perspective-Bounded Evidence Calibration for Multimodal Theory of Mind

**arXiv ID:** 2610.36824 | [PDF](https://arxiv.org/pdf/2610.36824v1)

**作者:** Ziqi Liao `[一作]` (Jilin University), Hongxia Xie `[通讯]` (Jilin University)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Vision Language Model` `Multimodality` `Text` `Benchmark`

**🎯 论文内容**

本文提出了PIP-ToM框架，解决多模态理论心智推理中证据可接收性问题，构建以代理为中心的证据存储并通过声明感知的路由对证据进行校准；

**💡 创新点**

创新点在于将证据可接收性原则明确为接口规范，既限制证据的可访问性，又根据声明类型匹配证据，从而避免视角泄漏和源路混淆；

**🔧 技术方法**

技术上结合了多模态事件重构、代理视角存储、结构化心智状态声明解析、路由特定证据校准以及对基线贝叶斯逆规划的概率调整；

**📊 数据集**

使用MuMA-ToM、MMToM-QA和Hi-ToM三大理论心智基准数据集；

**📈 对比分析**

与多种基线（通用LLM、专用推理提示、状态跟踪、模型基础推理）对比，PIP-ToM在GPT-4o和其他模型上分别在MuMA-ToM、MMToM-QA和Hi-ToM上获得92.56%、95.00%和95.00%的平均准确率，显著优于对照组；

**⚠️ 局限性**

局限性在于对隐式和递归心智状态仍需依赖模型推理，直接匹配方法无法覆盖所有案例；此外，框架在高深度推理场景下的计算成本和对全局重构质量的依赖仍需进一步研究。

---

## 109. A Local Stabilized Isogeometric Formulation with Discontinuous Pressure for Mesh-Conforming and Immersed Analysis of Nearly Incompressible Elasticity

**arXiv ID:** 2610.36349 | [PDF](https://arxiv.org/pdf/2610.36349v1)

**作者:** Ashkan Ali Madadi `[一作]` (University of Texas at Austin), Thomas J. R. Hughes `[通讯]`

**关键词:** `Computational Engineering, Finance, and Science` `Physics Related` `Variational Multi-Scale (VMS) Residual Stabilization` `Nitsche Method` `L² Projection` `Octree Subgrid Integration` `Extended B-spline` `Mesh` `Benchmark`

**🎯 论文内容**

本文提出了一种局部残差基变分多尺度（VMS）稳定的混合等距分析（IGA）公式，用离散压强实现近压缩弹性问题的体积锁定消除，并在保持静态消元的同时将压强自由度局部化；

**💡 创新点**

创新点在于将VMS残差稳定项与离散压强结合，显著降低了对压强网格细化的阶数依赖（仅需相对网格加粗因子为2），并将该框架推广至非贴合（浸入式）分析，采用扩展B-spline、压强细胞聚合、子格分裂积分和Nitsche弱边界条件，实现全局解算器复用与高度可扩展性；

**🔧 技术方法**

核心技术包括B-spline/NURBS等距空间、变分多尺度残差稳定化（GLS/SUPG/PSPG/VMS）、扩展B-spline（消除切割单元不完整基函数）、压强细胞聚合（L²投影）、递归子格切割积分（octree）以及Nitsche方法弱施加Dirichlet条件；

**📊 数据集**

通过经典的Kirsch圆环与Cook’s membrane基准测试（提供解析解与奇异行为），分别在网格贴合和浸入式设置下评估模型；

**📈 对比分析**

实验结果显示离散inf‑sup常数保持正值，误差收敛率达到理论预期；在浸入式方案中仍保留混合稳定性，压强与应力场无振荡，VMS变体实现无条件稳定且仅需局部化计算；

**⚠️ 局限性**

局限在于完全不可压的Stokes问题无法完成压强消元；残差参数若选取过大会导致过度稳定；当前仅适用于线性弹性，非线性大变形及更复杂几何的推广仍待研究；

---

## 110. AHEAD: Adaptive Neural Acceleration for Large-Scale Linear Programming

**arXiv ID:** 2610.36321 | [PDF](https://arxiv.org/pdf/2610.36321v1)

**作者:** Sihao Liu `[一作]` (North Carolina State University), Xiaorui Liu `[通讯]` (North Carolina State University)

**关键词:** `Machine Learning` `Optimization` `Transformer` `Recurrent Neural Network` `Contrastive Learning` `Supervised Fine-Tuning` `Tabular` `Benchmark`

**🎯 论文内容**

AHEAD 通过轻量级神经加速器为一阶线性规划求解器提供温启动加速，实现端到端速度提升

**💡 创新点**

只学习步长倍率，使用权重共享递归预测器和基于残差的自适应停止手风车，保证每一步低成本并可泛化至不同规模与精度

**🔧 技术方法**

采用 PDHG/PDLP 求解器、10维求解器状态的 MLP 步长预测器、学习手风车决策网络、多目标跨时间窗口训练以及 GPU 实现

**📊 数据集**

合成 Packing LP（10^5–10^7 变量）和 PageRank LP（Barabási–Albert 图）等数据集，并包含未见分布与难度分层样本

**📈 对比分析**

与 Gurobi GPU-PDHG、OR-Tools PDLP、PDHG-Net 进行对比，针对不同规模、精度和难度，AHEAD 端到端加速 6.5–11.6 倍，显著优于基线

**⚠️ 局限性**

仅在 Packing 与 PageRank 结构上验证，未覆盖真实工业 LP；手风车学习需要额外测量求解器运行时间；对极端稀疏或不平衡矩阵的鲁棒性未知

---

## 111. OFDM-Based Neuromorphic Wake-Up Signal

**arXiv ID:** 2610.36914 | [PDF](https://arxiv.org/pdf/2610.36914v1)

**作者:** Gaspard Goupy `[一作]` (Nokia Bell Labs), Dani Korpi `[通讯]` (Nokia Bell Labs)

**关键词:** `Networking and Internet Architecture` `Computational Efficiency` `Federated Learning` `Optimization` `Spiking Neural Network` `OFDM Modulation` `IFFT` `Threshold Pulse Encoding` `Weighted Sparse Correlation Algorithm` `Time Series` `Benchmark`

**🎯 论文内容**

提出了一种基于OFDM的稀疏尖峰式醒醒信号nWUS，并设计了对应的非协同神经形态检测器。

**💡 创新点**

创新点在于通过在子载波上叠加线性相位生成时间域稀疏峰值，并利用这些峰值构建事件驱动的稀疏相关模板，实现低功耗神经形态检测。

**🔧 技术方法**

使用了OFDM调制、线性相位生成、IFFT、阈值脉冲编码以及加权稀疏相关算法，并借助Sionna仿真工具进行评估。

**📊 数据集**

采用3GPP TR 38.901城市宏观通道模型（Sionna 2.0）进行单输入单输出下行链路的多用户随机场景仿真。

**📈 对比分析**

与两种OOK‑WUS基准在相同平均发射功率下比较，nWUS在相同或更短时隙内实现了更高的检测概率（平均提升18.6pp至3.3pp），并将计算量降低至OOK的2.5倍至43.5倍，空闲期计算量下降86%。

**⚠️ 局限性**

局限性包括缺乏硬件实现与功放线性度评估、阈值脉冲编码需要预校准、以及对极端信道条件的鲁棒性待进一步验证。

---

## 112. From Slices to Volumes: Comparing 2D, 2.5D, and 3D Residual Networks for Pulmonary Nodule Malignancy-Risk Classification

**arXiv ID:** 2610.35960 | [PDF](https://arxiv.org/pdf/2610.35960v1)

**作者:** Bahodir Nazarov `[一作]` (Sichuan University), Fatima Zazoul `[通讯]` (Sichuan University)

**关键词:** `Machine Learning` `Classification` `Convolutional Neural Network` `Contrastive Learning` `Biomedical Data` `Computed Tomography`

**🎯 论文内容**

比较2D、2.5D和3D残差网络在肺结节恶性风险分类上的性能。

**💡 创新点**

通过对同一LIDC‑IDRI数据集进行不同空间表示、MoCo预训练和轻量化数据增强的系统比较，评估空间维度对模型表现的影响。

**🔧 技术方法**

使用ResNet‑18架构，包含Momentum Contrast预训练、轻度数据增强、不同输入表示（单张切片、5张切片堆叠、完整CT体积）。

**📊 数据集**

LIDC‑IDRI公开胸部CT及多读者恶性风险评分数据集。

**📈 对比分析**

在固定的患者分层拆分下，通过验证Macro‑F1选择检查点，最终2.5D轻度增强模型在三分类测试上取得最高Macro‑F1 0.6537，二分类去除中性风险后最高Macro‑F1 0.8664。

**⚠️ 局限性**

缺乏外部验证、不同表示间的预处理和训练细节不一致、未重复多次实验、患者内相关性未估计、标签为主观读者评分。

---

## 113. Algorithmic Monoculture in Machine Learning-based Site-Specific Fertilizer Recommendation

**arXiv ID:** 2610.36941 | [PDF](https://arxiv.org/pdf/2610.36941v1)

**作者:** Abdelghani Belgaid `[一作]` (Mohammed VI Polytechnic University), Abdelghani Belgaid `[通讯]` (Mohammed VI Polytechnic University)

**关键词:** `Machine Learning` `Agriculture Related` `Optimization` `Federated Learning` `Risk‑Reward Optimization` `Covariance Matrix Analysis` `Tabular` `Agriculture Related`

**🎯 论文内容**

构建了一个连续行动经济模型，评估不同农场采用不同机器学习肥料推荐模型时的系统性外部性与私人成本，并通过集中度指标和协方差一般化形式分析模型错误聚集效应。

**💡 创新点**

提出了基于共享与农场特定错误的系统损失公式、统一的两模型阈值与准确性约束的凸组合优化规则，揭示模型多样性与误差相关性对社会损失的影响，并在摩洛哥稻麦等系统中通过基准校准展示了效益。

**🔧 技术方法**

使用了经济学的风险‑收益优化（均值‑方差型）框架、Herfindahl–Hirschman指数、协方差矩阵分析，以及基于预测误差的量化指标（1‑R²）来构建模型。

**📊 数据集**

采用了摩洛哥五个谷物系统（雨养大麦、粒收玉米、饲料玉米、灌溉小麦、雨养小麦）共计132,017个推荐场景的数据，9个回归模型族进行对比。

**📈 对比分析**

比较了四种分配策略（最低风险单一模型、均匀分配、受私人成本约束的最优组合、无约束的社会损失最小化），在中心参数设定下受限组合平均可降低10.9%–41.5%的模拟社会损失，均衡化方案在满足约束时亦能获得收益；但若误差高度相关则优势消失。

**⚠️ 局限性**

局限性包括：评价仅基于预测误差，不涉及作物产量或经济/环境效益；误差相关性、外部性强度、λ、ρ等参数未通过数据识别；假设的二次外部性与单一数据样本限制了因果推断；采用的分配假设与现实采用行为不一定匹配。

---

## 114. A State Augmented Approach to Paralle Modeling in High-Level Synthesis

**arXiv ID:** 2610.36481 | [PDF](https://arxiv.org/pdf/2610.36481v1)

**作者:** Nicholas Kovacs `[一作]` (University of South Florida), Hao Zheng `[通讯]` (University of South Florida)

**关键词:** `Hardware Architecture` `Optimization` `Benchmark`

**🎯 论文内容**

提出了一种基于同步顺序进程的显式架构建模方法，并在RV32I处理器上实现验证。

**💡 创新点**

通过将架构状态与HLS调度分离，用C++显式描述流水线阶段和危害，实现显著的CPI提升和面积优化。

**🔧 技术方法**

使用同步顺序进程模型、C++建模、Vitis HLS、Kintex UltraScale+ FPGA实现，并与Ibex RTL及算法化HLS基线进行对比。

**📊 数据集**

使用八个内核基准（含CoreMark和其他自定义核）以及全程序周期计数作为评估数据集。

**📈 对比分析**

在同一硬件平台上测量周期计数、实现频率和面积，CPI‑优化版在220 MHz下实现1.221 CPI，面积优化版实现65 % LUT和47 % FF的减幅，运行时间比Ibex低约15–20 %。

**⚠️ 局限性**

仅在RV32I处理器上验证，工作负载有限，未涵盖外部内存延迟和缓存，工具链及硬件平台特异性强，缺乏更广泛设计的通用性验证。

---

## 115. ResCom: Activation Residual Compensation for Large Language Model Quantization

**arXiv ID:** 2610.35955 | [PDF](https://arxiv.org/pdf/2610.35955v1)

**作者:** Ning Yang `[一作]` (Chinese Academy of Sciences), Yuting Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 12978 | [OpenAlex ID](https://openalex.org/A5110722362)

**关键词:** `Machine Learning` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Large Language Model` `Contrastive Learning` `Auto Encoder` `Text`

**🎯 论文内容**

在已校准的低位权激量化 LLM 上引入残差条件低秩修正分支，补偿激活舍入误差。

**💡 创新点**

创新点是利用激活舍入残差进行低秩回归拟合的残差条件校正，在不更新已校准权重的前提下恢复输出信息。

**🔧 技术方法**

技术包括残差条件低秩分支、闭式正则化低秩回归、统计矩阵（协方差、交叉协方差）计算与 SVD。

**📊 数据集**

使用 WikiText-2、C4、MMLU、GSM8K 等文本数据集进行校准与评估，并在多种 LLM 结构（Llama、LLaMA、Mistral、Gemma）上测试。

**📈 对比分析**

与原始 SERQ、PrefixQuant、FlatQuant 校准模型对比，在 4 位权激量化下，ResCom 在 WikiText-2 上将 PPL 从 7.708 降至 7.147，MMLU 提升 2.89 分，五次 GSM8K 正确率从 23.96% 提升到 36.16%。

**⚠️ 局限性**

局限性在于需要额外的校准数据、额外的低秩矩阵参数和运行时乘加运算，且对极低秩或大模型的性能提升有限。

---

## 116. MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics

**arXiv ID:** 2610.36513 | [PDF](https://arxiv.org/pdf/2610.36513v1)

**作者:** Yibo Yang `[一作]` (Independent Researcher), Zixia Wang `[通讯]` (University of Exeter)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Federated Learning` `Safty and Privacy` `Benchmark` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Biomedical Data` `Benchmark`

**🎯 论文内容**

构建 MedConstBench，系统评估不同宪章对医疗伦理 LLM 行为的影响，并提出基于“do‑no‑harm”准则的自适应宪章选择器。

**💡 创新点**

将宪章视为可控变量、采用多模型同行评审打分机制、提出自适应宪章选择方法，突破传统单一评估与固定宪章的局限。

**🔧 技术方法**

使用 Beauchamp 四原则的分级评估框架、离散评分与多模型同行评审、以及基于性能差异的最优宪章选择算法。

**📊 数据集**

MedDilemma（400 条临床伦理困境）与 MedEthicsQA（400 条公开伦理问答）。

**📈 对比分析**

对十款开源 LLM 在四种宪章下进行打分，发现无单一宪章最优；自适应选择器在平均和最差案例上均达到与 oracle 相当的表现。

**⚠️ 局限性**

宪章效果高度依赖具体模型，评估受限于手工标注规模，且未覆盖所有医疗伦理场景。

---

## 117. Evaluation-Guided Demonstration Refinement for Vision-Language-Action Policies: The Winning Entry in the MetaFine Open Competition

**arXiv ID:** 2610.36765 | [PDF](https://arxiv.org/pdf/2610.36765v1)

**作者:** Lifan Zheng `[一作]` (Southeast University), Fan Liu `[通讯]` (Southeast University)

**关键词:** `Computer Vision and Pattern Recognition` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Vision-Language-Action Model` `Flow-based Model` `Supervised Fine-Tuning` `Retrieval-Augmented Generation` `Diffusion model` `Multimodality` `Image` `Text`

**🎯 论文内容**

在MetaFine开源竞赛中，针对五个桌面任务分别训练了任务专属策略，并通过反复检查指令级别的失败，改进演示轨迹、加入专家后续与恢复演示，最终实现了最高96.66分的冠军成绩。

**💡 创新点**

创新点在于：①对官方演示进行细粒度的错误定位与修正；②使用任务特定的训练混合和演示源分配；③结合演示聚合（dataset aggregation）采集的专家后续与恢复轨迹；④采用流匹配行为克隆（flow‑matching BC）与OpenPI框架实现高效的连续动作生成。

**🔧 技术方法**

技术手段包括：使用OpenPI实现的全参数微调、AdamW优化、学习率预热与余弦衰减、10步去噪的流匹配BC目标、一次性生成50步动作并截断执行的策略设计，以及对不同任务使用不同动作截断长度的自适应规划。

**📊 数据集**

数据集来源于MetaFine官方演示与专家程序生成的轨迹，补充了额外场景的演示以及从失败状态记录的后续和恢复示例；在本地开发中使用约80–120个场景进行评估和检查。

**📈 对比分析**

通过官方排行榜与本地开发套件的双重评估，提交策略在官方最终排名第一，整体得分96.66，高于第二名94.71；在本地评估中，各任务的合成分数均显著提升（例如T1从88.95提升到93.39，T4从99.81提升到99.97），表明方法对多任务和多场景具有较好适应性。

**⚠️ 局限性**

局限性包括：①演示修正与恢复示例的选择仍依赖人工检查，缺乏自动化判定；②局部性能提升可能导致其他指令或场景的回退；③本地评估与官方评估使用不同场景，难以完全衡量泛化能力；④训练中混合了演示源与采样改动，难以单独评估每种技术的具体贡献。

---

## 118. SetGRPO: Calibrating Grader-Induced Policy Optimization

**arXiv ID:** 2610.36807 | [PDF](https://arxiv.org/pdf/2610.36807v1)

**作者:** Jiale Huang `[一作]`, Yupeng Hu `[通讯]`

**关键词:** `Machine Learning` `Reinforcement Learning from Human Feedback` `Optimization` `Reinforcement Learning` `Conformal Prediction` `Text` `Benchmark`

**🎯 论文内容**

本文提出 SetGRPO，构建联合训练系数不确定集，并通过几何安全恢复与风险门控筛选在 Rubric-based GRPO 中抵消评审员随机性导致的策略更新偏差。

**💡 创新点**

创新点在于将组内归一化造成的系数相关噪声建模为联合集合，并结合当前策略梯度几何和条件方向逆转风险进行自适应恢复与筛选，提供有限样本覆盖与风险控制。

**🔧 技术方法**

技术主要包括联合奖励集校准 (JRSC)、几何引导安全恢复 (GGSR)、选择性风险门控优化 (SRGO) 以及 conformal prediction 与 Clopper–Pearson 置信界风险估计。

**📊 数据集**

使用 RubricHub 训练集（5类任务）进行评估，并在十个跨医疗、写作、指令遵循的公开基准上测试模型。

**📈 对比分析**

与原 GRPO、RubricHub、RuscaRL 等方法对比，SetGRPO 在 Qwen3.5 与 Llama3.2 系列上实现了宏平均得分最高，整体提升约 3–5 分。

**⚠️ 局限性**

局限性包括依赖评审员评分的可交换性假设、需要双份评分进行校准、额外的计算开销，以及可能无法完全消除评审偏差或在极端噪声环境下仍出现方向逆转。

---

## 119. safeL4: Rethinking Safe-Language-Based Single Address Space Microkernels

**arXiv ID:** 2610.35999 | [PDF](https://arxiv.org/pdf/2610.35999v1)

**作者:** Zheyuan Xu `[一作]` (Chinese Academy of Sciences), Rui Hou `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 7528 | [OpenAlex ID](https://openalex.org/A5010636698)

**关键词:** `Operating Systems` `Safty and Privacy` `Computational Efficiency` `Large Language Model` `Prompt Engineering` `Auto Encoder` `Diffusion model` `Contrastive Learning` `Gaussian Splatting` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation` `Tabular` `Time Series` `Sequential` `Benchmark`

**🎯 论文内容**

设计并实现了基于Rust安全语言的单地址空间微内核safeL4，构建了能力系统并提供了低级硬件抽象、IPC、调度等核心服务，支持用户空间程序和设备驱动。

**💡 创新点**

创新点在于：①利用Rust的对象能力实现无硬件隔离的可撤销、单调权限的能力系统；②通过低级安全抽象封装所有unsafe操作，实现最小化的TCB；③设计能力派生树和宏化接口，保证能力的不可伪造、可追溯和可撤销；④将单地址空间设计与能力调用结合，显著降低IPC、上下文切换开销。

**🔧 技术方法**

主要技术包括：Rust语言与其安全子集、Arc/RwLock、弱引用（Weak）、proc‑macro、能力派生树（CDTree）、零拷贝消息传递、链接时接口注册、panic 捕获/恢复、Musl libc 兼容、以及对 RISC‑V64/AArch64 的支持。

**📊 数据集**

实验使用的基准和数据集：YCSB（SQLite3）工作负载、LMbench（文件系统子基准）、seL4bench（低级操作基准），在 Raspberry Pi 3B+ (Cortex‑A53) 设备上运行。

**📈 对比分析**

通过与 seL4、Linux 以及 Redox 的对比：在低级操作（IPC、硬件调用、IRQ、页面映射、调度）中 safeL4 的平均速度提升为 1.91×；在 IPC/IDC 上提升 7.8–22.3×；在 YCSB SQLite3 任务中相对 seL4 提升 14.3%，相对 Linux 提升 23.2%；文件系统微基准相对 Redox 提升 1.47×。

**⚠️ 局限性**

限制与不足：①仍未实现 DMA/IOMMU 隔离，unsafe 代码占约 15% 仍属于 TCB；②在帧分配等操作中因细粒度锁和 CDTree 维护导致高开销；③安全性保证仅涵盖语言级别内存/类型安全和能力安全，驱动功能正确性仍需额外保障；④对 Rust 编译器、标准库等工具链的安全性有依赖。

---

## 120. Physiognomy Revisited: How Vision-Language Models Encode Sensitive Attributes Through Image Transformation

**arXiv ID:** 2610.35910 | [PDF](https://arxiv.org/pdf/2610.35910v1)

**作者:** Soumyajit Datta `[一作]` (Rochester Institute of Technology), Ashiqur R. KhudaBukhsh `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 671 | [OpenAlex ID](https://openalex.org/A5035198055)

**关键词:** `Computer Vision and Pattern Recognition` `Safty and Privacy` `Explainability and Interpretability` `Transformer` `Vision Language Model` `Prompt Engineering` `Contrastive Learning` `Integrated Gradients` `Image` `Text` `Multimodality`

**🎯 论文内容**

本研究利用 AI 生成的人脸图像，探究了视觉‑语言模型（VLM）在接受涉及种族、性取向及犯罪记录等受保护身份的图像变换请求时的行为，分析了其拒绝模式的非确定性、生成图像的可分辨性以及这些偏见如何在后续文本生成中传播；

**💡 创新点**

创新点在于揭示 VLM 对敏感属性的推断与生成之间的安全行为不对称性，即模型拒绝直接推断属性却同样接受并执行基于这些属性的图像变换；同时首次系统性展示了变换后图像在统计上可区分且可导致跨模态偏见扩散，并将此问题拓展至犯罪属性。

**🔧 技术方法**

技术上采用了 Gemini 与 OpenAI 的 VLM，配合自然语言变换提示，使用 ResNet 与 ViT 进行分类可分辨性测试；利用 Integrated Gradients 与面部解析进行特征归因；通过图像‑文本实验评估偏见传播。

**📊 数据集**

使用三大 AI 生成人脸数据集（Hyperface、AIFace、MSFake），共 1,002 张样本，按性别平衡；此外在附录中对少量真实人脸进行了鲁棒性验证。

**📈 对比分析**

评估方法包括多次重复查询的拒绝率统计、不同数据集的拒绝差异、分类器在 gay/straight 与 straight/stay 两组间的准确率（最高可达 88% 以上，远高于 50% 随机基准），以及文本特征的对数比值分析，均显示出显著的身份相关偏差。

**⚠️ 局限性**

局限性包括：仅使用合成图像，无法直接推断真实世界效果；只探讨了有限的二元身份标签；对辅助分类器与可解释方法的依赖可能引入额外假设；性别推断仅为粗略平衡手段；未涉及人工判断或更细粒度的身份维度。

---

## 121. Agar: What-If Analysis for Microservice Applications

**arXiv ID:** 2610.36636 | [PDF](https://arxiv.org/pdf/2610.36636v1)

**作者:** Saarth Deshpande `[一作]` (University of Texas at Austin), Venkat Arun `[通讯]` (University of Texas at Austin)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Explainability and Interpretability` `Optimization` `Tabular` `Time Series`

**🎯 论文内容**

提出了一款基于可解释区间模型的反事实性能分析工具，可回答在不同工作负载、资源分配或 SLO 目标下的微服务应用的 SLO 可行性、资源需求、成本估算和性能诊断等问题，并实现了基于该工具的自动伸缩器。

**💡 创新点**

创新点在于：①将每个组件模型的预测目标从精确值改为包含所有观测的上下界，保证 100% 准确且能通过越界检测到模型失效；②联合优化组件模型以最小化整体不确定性，并用贪心 Set‑Cover 近似算法选择最优 SLO 阈值；③直接将这些可解释区间模型组合，进行反事实推断，显著降低实验成本。

**🔧 技术方法**

使用了：基于 monotonicity 和加性假设的利用率模型、SLO 满足阈值模型和 RPS 预测模型；Charnes–Cooper 变换求解线性规划以得到最优复制数；贪心 Set‑Cover 近似求解阈值选择；结合 Kubernetes、Trace、CPU/内存/延迟等监控数据；并通过可解释区间组合实现反事实查询。

**📊 数据集**

在三款典型微服务应用（Online Boutique、TeaStore 以及另一实验应用）上采集的生产/实验工作负载、请求速率、CPU/内存利用率、SLO 满足情况等指标，用于训练与验证模型。

**📈 对比分析**

与现有水平伸缩器（如 Kubernetes HPA、Galileo 等）对比，实验显示在三应用中使用该工具的自动伸缩器可将 CPU 分配减少高达 36.8% 的同时仍满足 SLO；此外还能准确估算运营成本、给出服务启动时间优化建议并定位性能缺陷，验证了模型的可操作性。

**⚠️ 局限性**

局限性包括：需要假设应用行为随时间保持稳定，代码或硬件变更后需重新训练；对噪声或离群值较敏感；目前仅支持水平伸缩，未覆盖纵向伸缩或动态资源重分配；模型仅假设 monotonicity 与加性，可能无法完全捕捉复杂交互。

---

## 122. Governed Sovereign RAG for Public Administration: Tripartite Epistemic Routing, Facet-Guided Clarification, and Deterministic Numeric Verification in Citizen Services

**arXiv ID:** 2610.36561 | [PDF](https://arxiv.org/pdf/2610.36561v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## 123. From Ethernet to Wi-Fi: Flow Feature Selection and Construction for Traffic Classification

**arXiv ID:** 2610.37007 | [PDF](https://arxiv.org/pdf/2610.37007v1)

**作者:** Francisco Rau `[一作]` (MaxLinear, Inc.), Joaquin Perez `[通讯]` (Universitat de València)

**关键词:** `Networking and Internet Architecture` `Classification` `Domain Adaptation` `Contrastive Learning` `Feature Selection` `Tabular` `Time Series`

**🎯 论文内容**

设计了一个五阶段流程，选取并构造适用于 Wi‑Fi 的流特征，从而实现仅用以太网训练的数据也能在 Wi‑Fi 上进行高精度流量分类。

**💡 创新点**

创新点在于针对率依赖、分布偏移、类别信息与冗余等多维度自动筛选特征，并通过构造归一化的包大小与时间比率特征，使模型能够跨介质迁移。

**🔧 技术方法**

使用特征选择技术（相关系数、Kolmogorov–Smirnov 距离、互信息、冗余消除）和 XGBoost 分类器，并在 ns‑3 仿真环境下实现 Wi‑Fi/以太网数据的流量提取。

**📊 数据集**

采用 ns‑3 生成的以太网和 Wi‑Fi 流量数据，包含六个开发场景（S1–S6）用于特征设计，单独的 S7 场景用于最终评估。

**📈 对比分析**

通过与基线的完整特征集（29 维）及逐步删减的特征集比较，实验表明七维特征在 Wi‑Fi 上的准确率从约 88% 提升到 99.9%，而以太网准确率保持在 99.5%，证明了跨介质迁移的有效性。

**⚠️ 局限性**

局限性在于仅基于 ns‑3 仿真数据，未涵盖真实无线网络噪声、不同 802.11 版本、QoS 访问类别及多实现的 DASH 流，未来需验证在更广泛的实际条件下的稳健性。

---

## 124. A generalist policy for muscle-driven control across individual musculoskeletal models

**arXiv ID:** 2610.36535 | [PDF](https://arxiv.org/pdf/2610.36535v1)

**作者:** Bozhao Wang `[一作]` (Tsinghua University), Xiaoqiang Tang `[通讯]` (Tsinghua University)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Transformer` `Graph Neural Network` `Reinforcement Learning` `Gaussian Splatting` `Contrastive Learning` `Prompt Engineering` `Biomedical Data` `Mesh` `Point Cloud` `Tabular` `Time Series` `Computed Tomography` `Magnetic Resonance Imaging` `Positron Emission Tomography` `Ultrasound` `Fibre Orientation Distribution` `Diffusion Tensor Imaging` `Electronic Health Records` `Electrocardiogram`

**🎯 论文内容**

提出MyoAcross框架，实现了不需要个体微调的通用肌肉驱动参考运动跟踪，结合个体化肌骨模型构建、形态条件化策略学习和虚拟模型增强；

**💡 创新点**

创新点在于①利用形态特征通过FiLM条件化策略，使单一政策可跨个体模型工作；②采用虚拟模型增强显著提升对未见模型的泛化；③在单图像估计、体重/身高以及单侧下肢缩短等临床相关结构变化下验证策略可迁移；

**🔧 技术方法**

技术手段包括SMPL+SKEL进行区域缩放与质量校准；姿态重定向（GMR-Fit）生成参考运动；形态编码器+FiLM条件化的Actor-Critic结构；PPO强化学习；虚拟模型通过高斯扰动生成；以及对EMG、GRF等生物力学信号的后验比较；

**📊 数据集**

主要使用数据集：AMASS（141人）、Gait120（120人）、MoVi（88人）以及KIT源运动；单图像实验使用HSMR估计的SKEL形状；

**📈 对比分析**

与基线（模板MLP、无条件MLP、Concat）对比，采用完成率、覆盖率、跟踪误差等指标；MyoAcross在31个未见个体模型上完成率达99.24%，跟踪误差下降22.6%；在Gait120中垂直GRF相关性0.853、EMG相关性0.406；在单侧下肢缩短和单图像模型上亦保持高完成率；

**⚠️ 局限性**

局限性包括：①所有模型共享相同拓扑结构，无法直接处理拓扑差异；②缺乏个体肌肉力量、张力等功能参数，仅依赖形态、体重估计；③虚拟增强主要考虑几何扰动，对肌肉动力学或神经控制的非几何差异不敏感；④单图像估计的精度未与真实标记数据直接验证；

---

## 125. Training-Free Inference Acceleration for Video Large Language Models: Methods, Trade-offs, and Evaluation

**arXiv ID:** 2610.36327 | [PDF](https://arxiv.org/pdf/2610.36327v1)

**作者:** Yuqi Li `[一作]` (Wyze Labs, Inc.), Yingli Tian `[通讯]` (City University of New York)

**关键词:** `Computer Vision and Pattern Recognition` `Computational Efficiency` `Representation Learning` `Compression` `Transformer` `Vision Language Model` `Large Language Model` `Contrastive Learning` `Keyframe Selection` `Feature Merging` `KV Cache Management` `Post-Training Quantization (PTQ)` `Video` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

本文系统梳理并分类了训练无关的视频大型语言模型（VLM）加速方法，提出按输入、编码、投影、预填和解码阶段的分层体系，并在统一评测框架下比较其准确性与延迟。

**💡 创新点**

首次从干预位置出发，深入分析了时空证据保持与计算节省的关系，提出质量–延迟平衡（QLB）指标，并在实验中揭示局部节省不必然转化为端到端加速的规律。

**🔧 技术方法**

综合运用了关键帧选择、空间自适应、特征合并、投影与预填阶段的视觉‑语言交互压缩、解码阶段的 KV 缓存管理以及后训练量化（PTQ）等多种技术，形成多阶段压缩方案。

**📊 数据集**

在 LLaVA‑OneVision‑7B 等模型上使用 Video‑MME、EgoSchema、LongVideoBench、MVBench 四个基准，并对比 VideoDC、Qwen2.5‑VL‑3B 等公开数据集进行评测。

**📈 对比分析**

在统一硬件（NVIDIA H100）和固定帧数条件下，采用准确率、TTFT、TPOT 等指标并结合 QLB 评价方法；实验显示早期编码压缩可提升约30% TTFT，但仅略提升准确性，跨阶段量化在保持约97%精度的同时显著降低内存。

**⚠️ 局限性**

实验仅基于单一 backbone、固定帧预算和单一硬件平台，跨文献对比受不同实验协议影响，PTQ 结果主要来源于图像任务，缺乏完整端到端延迟细分，未充分考虑不同部署环境和硬件差异。

---

## 126. Frankly Speaking: The Company Election Integrity Keeps in Congressional Communication

**arXiv ID:** 2610.36187 | [PDF](https://arxiv.org/pdf/2610.36187v1)

**作者:** Michael A. Robinette `[一作]` (University of Vermont), Juniper Lovato `[通讯]` (University of Vermont)

**通讯引用:** 185 | [OpenAlex ID](https://openalex.org/A5073688320)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Representation Learning` `Transformer` `Word2Vec` `Contrastive Learning` `Prompt Engineering` `Text` `Review/Survey Paper`

**🎯 论文内容**

本文通过对2010-2025年10,420封国会议员电子通函的文本分析，考察了民主党与共和党在使用与选举诚信相关词汇（如“integrity”“security”“fraud”等）时的语义差异。

**💡 创新点**

创新点在于揭示即使使用相同的词汇，两党在语义邻居（semantic neighborhoods）上呈现出截然不同的框架，表明选举诚信的概念在两党中被赋予不同的政治意义。

**🔧 技术方法**

采用自然语言处理技术：对文本进行分词、去停用词，使用Word2Vec训练时间分段词向量模型；对各党年度模型进行正交Procrustes对齐；通过余弦距离评估跨党分歧、Jaccard相似度量化语义邻居重叠，并以频率匹配的噪声阈值进行统计比较。

**📊 数据集**

使用的数据集为DCInbox项目提供的美国国会议员电子通函全文，共计10,420封，覆盖2010至2025年，按党派（民主党与共和党）分类。

**📈 对比分析**

通过对比跨党余弦距离与同频词的噪声分布（百分位数），以及对各党内部邻居重叠的Jaccard基准，发现两党在多数关键词上并未显著偏离噪声水平，但在语义邻居层面差异显著；在党内语义迁移上，变化幅度相近，且主要集中在“certification”“rigged”“stolen”等词上。

**⚠️ 局限性**

局限性包括：低频词的向量不稳定、模型对噪声敏感、仅分析电子通函而非更广泛的公开沟通、过滤策略可能偏向安全与欺诈语境、未评估选民对信息的感知与影响，且未对模型设定（维度、窗口、种子）做系统敏感性检验。

---

## 127. CAMR: Compatibility-Aware Memory Replay for Continual Learning in Large Language Models

**arXiv ID:** 2610.36734 | [PDF](https://arxiv.org/pdf/2610.36734v1)

**作者:** Jingpu Yang `[一作]` (Beijing Zhongguancun Academy), Xiaoqing Zhang `[通讯]` (Beijing Zhongguancun Academy)

**关键词:** `Computation and Language` `Knowledge Distillation` `Computational Efficiency` `Representation Learning` `Reinforcement Learning from Human Feedback` `Federated Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Retrieval-Augmented Generation` `Prompt Engineering` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

提出兼容性感知记忆重放（CAMR）框架，在持续学习过程中联合控制当前示例的监督与历史示例的重放，受限于每次到达的重放预算。

**💡 创新点**

创新点在于：①将兼容性检查与记忆门控结合，生成可用于决策的历史摘要；②采用在线actor‑critic策略，根据获取、保留和训练成本反馈动态选择三种动作（普通重放、兼容历史强化、跳过已掌握重复）；③实现对全参数LLM的持续更新而不需额外微调模块。

**🔧 技术方法**

技术包括：MPNet嵌入、随机超平面LSH检索、RMSNorm投影、sigmoid门控求相关度、actor‑critic学习、基于NLL的奖励设计、预算约束的采样策略。

**📊 数据集**

数据集：WikiBigEdit（事实更新）、WikiFactDiff（事实替换）、TRACE（四任务交叉学习：C‑STANCE、MeetingBank、Py150、ScienceQA）、FinQA→GSM8K→APPS→SAMSum（跨域连续学习），以及内部的FOMC、NumGLUE、Py150用于基准测试。

**📈 对比分析**

与全参数微调、LoRA、知识编辑、随机/语义/关系Oracle/Maximally Interfered/Transfer Selective等回放方法对比，CAMR在WikiBigEdit上连续提升Original Match（93.66%）并超过Full‑History Replay；在TRACE任务上获得最高最终分数（56.75%、69.84%、67.92%、95.40%）；在跨域序列中保持或提升前三个任务的性能；在不同重放预算下始终领先；消融实验表明动作选择、记忆门控和在线反馈对性能贡献显著。

**⚠️ 局限性**

局限性：实验仅覆盖Qwen3‑8B‑Base（未验证更大模型或其他体系结构）；部分基线使用LoRA而非全参数微调；对包含大量重复事实的数据流的评估有限；重放预算为固定，未探讨动态调节；缺乏对模型长期记忆衰减的长期实验。

---

## 128. Nonlinear Direct Feedback Alignment for Scalable Backpropagation-Free Training

**arXiv ID:** 2610.36718 | [PDF](https://arxiv.org/pdf/2610.36718v1)

**作者:** Wu Lin `[一作]` (Shenzhen University), Qiuzhen Lin `[通讯]` (Shenzhen University)

**关键词:** `Machine Learning` `Classification` `Computational Efficiency` `Representation Learning` `Convolutional Neural Network` `Residual Neural Network` `Transformer` `Direct Feedback Alignment` `Gradient Alignment` `Subnetwork Alignment Analysis` `Supervised Fine-Tuning` `Image` `Biomedical Data`

**🎯 论文内容**

本文提出了一种非线性直接反馈对齐（NDFA）方法，通过在每个隐藏层构建轻量级非线性替代子网络来生成可表示依赖的反馈，从而实现与BP相似的梯度传播，并保持层间并行训练；

**💡 创新点**

创新点在于将固定随机反馈矩阵替换为可学习的非线性子网络，从而捕捉隐藏表示到输出的非线性变换，并采用在线学习策略不断更新该子网络；

**🔧 技术方法**

技术上结合了直接反馈对齐（DFA）、BP、轻量级MLP子网络、梯度对齐和子网络对齐分析等；

**📊 数据集**

实验使用了CIFAR‑10/100、SVHN、ImageNet以及Mante 2013神经记录数据集，覆盖卷积网络、残差网络和视觉Transformer；

**📈 对比分析**

与BP及DFA、DKP、PEPITA、FDFA等基线比较，NDFA在所有网络和数据集上均显著优于现有DFA方法，且将与BP的性能差距压缩到10%以内（如ConvNet上准确率提升至78.6%/47.9%，ResNet‑32在ImageNet上提升17%）；

**⚠️ 局限性**

局限性包括：对极深网络仍存在一定性能差距；替代子网络带来额外计算开销；缺乏严格的收敛理论分析。

---

## 129. Repulsive Population Policy Gradients for Behaviorally Diverse Reinforcement Learning

**arXiv ID:** 2610.36425 | [PDF](https://arxiv.org/pdf/2610.36425v1)

**作者:** Saeed Hedayatian `[一作]` (University of Southern California), Stefanos Nikolaidis `[通讯]` (University of Southern California)

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning` `Large Language Model` `Prompt Engineering` `Text` `Image` `Video` `Tabular` `Time Series` `Sequential` `Benchmark`

**🎯 论文内容**

研发了 RePPO，一种直接在小规模多策略集上训练质量多样化的强化学习算法；

**💡 创新点**

创新点是将 Soft QD 的连续目标转化为每个策略的可塑形奖励，使得标准 PPO 能够通过种群互斥实现多样化，无需离散化或进化算子；

**🔧 技术方法**

使用了 Soft QD、PPO、可塑形奖励、加性 Markovian 行为描述符、局部 repulsion 计算、指数移动平均归一化、行为优势融合等技术；

**📊 数据集**

实验数据集包括 Brax 连续控制任务（Ant、HalfCheetah、Humanoid、Walker2D）、四款 Atari 游戏（Boxing、Asterix、Seaquest、Ms. Pac‑Man）以及两大 LLM（Qwen3‑1.7B、Granite‑4.1‑3B）在 MATH、Minerva、OlympiadBench 的推理任务；

**📈 对比分析**

通过与独立 PPO、PPGA、PGA‑ME、DCRL‑ME、Sep‑CMA‑MAE、RefQD 及提示基线比较，RePPO 在所有连续控制任务获得最高的 qVS，Atari 和 LLM 也实现了与基准相当的质量且显著提升多样性，整体表现优于或匹配传统方法；

**⚠️ 局限性**

局限性在于需可加性行为描述符、固定种群规模和带宽；在高维描述符下可能需要学习或自适应描述符；对极低质量策略的 repulsion 可能导致不稳定；并不适用于全行为空间的完整覆盖，更适合优化紧凑多样化集合。

---

## 130. Systematic Morphometric Brain Autoencoding for Representation Learning in Neurodegenerative Diseases

**arXiv ID:** 2610.37074 | [PDF](https://arxiv.org/pdf/2610.37074v1)

**作者:** Nuria Balbás-Esteban `[一作]` (Universidad Carlos III de Madrid), Carlos Sevilla-Salcedo `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `Computer Vision and Pattern Recognition` `Representation Learning` `Image Restoration` `Convolutional Neural Network` `Auto Encoder` `Contrastive Learning` `Auto Encoder` `Diffusion model` `Biomedical Data` `Image` `Magnetic Resonance Imaging` `Alzheimer's Disease`

**🎯 论文内容**

本文提出了一种针对结构T1加权MRI的自监督预训练框架 SYMBA，旨在学习适用于神经退行性疾病分析的形态学表征；

**💡 创新点**

创新点在于：①将解剖学约束融入遮挡策略，产生生物学意义的遮挡区域；②采用多尺度结构相似性（MS-SSIM）重建损失，仅对遮挡区进行监督；③引入逐步加深的学习课程，从局部细节过渡到全局形态；

**🔧 技术方法**

技术上使用3D ResNet-50编码器、转置卷积解码器，遮挡器基于脑灰质掩模采样几何形状（立方体、球体、菱形）并对称镜像；重建损失为MS-SSIM，训练采用AdamW与余弦退火；

**📊 数据集**

预训练数据集由6953份来自七个异质人群的T1 MRI组成，包含OASIS、IXI、SALD、Wayne、DLBS、NKI、CamCAN等；下游评估使用未参与预训练的ADNI与AIBL两大外部数据集；

**📈 对比分析**

与BrainMorph、BrainSegFounder、AnatCL、BrainIAC等主流自监督模型在冻结线性探测任务上比较，SYMBA在AD与CN、MCI进展预测、三分类、阿尔茨海默病与正常对照等任务均显著优于基线（如ADNI AD vs CN AUROC 0.94 vs 0.90）；在低标注样本场景下，仅用6个标记样本即可达到≈0.90 AUROC，远超全监督与其他预训练模型；

**⚠️ 局限性**

局限性包括：①仅在结构MRI上验证，缺乏多模态或其他疾病的泛化评估；②重建质量虽满足学习目标，但未针对临床可解释性做进一步验证；③对硬件要求高（3D ResNet+高遮挡率）且训练时间较长；

---

## 131. How Stable Are LLM-Generated Proofs Under Program Refactoring? A Stress Test for Verifiable Code Generation

**arXiv ID:** 2610.36328 | [PDF](https://arxiv.org/pdf/2610.36328v1)

**作者:** Aravind Mandiga `[一作]` (University of Georgia), Guoming Li `[通讯]` (University of Georgia)

**关键词:** `Artificial Intelligence` `AI Code Assistant` `Benchmark` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text`

**🎯 论文内容**

提出并实现了RefactorProof基准，用来评估已验证Lean证明在经过行为保持的程序重构后是否仍能通过验证；

**💡 创新点**

创新点在于：①将证明稳定性量化为Proof Survival Rate (PSR)，②构造了五类可证明的程序变形并对比不同来源的证明（官方参考证明与专用模型证明）的稳定性；

**🔧 技术方法**

主要技术包括：Lean4 4.18环境下的自动等价证明（rfl和定义性归约）、基于文本的五类重构（D、S、E）、任务级PSR统计与聚合、集群自助抽样的置信区间估计；

**📊 数据集**

使用了VERINA任务集（185个能在固定环境下编译的任务，46个官方提供参考证明），并从这些任务中生成变形版本；

**📈 对比分析**

通过比较每个任务在所有已证明变形上的PSR，得到专用模型的PSR显著高于官方参考（DeepSeek 59.3%、Goedel 64–65%对比参考仅48%）；在结构性和语义性变形下的差异也被统计；

**⚠️ 局限性**

局限性包括：只评估短非递归任务的子集；变形仅限于五类简单重构；未覆盖大规模真实项目中的重构；参考证明数量少；模型生成的证明质量和覆盖率低；统计方法未进行多重校正，结果易受样本偏差影响。

---

## 132. Impossibility of No-Regret in Multi-Agent Bidding

**arXiv ID:** 2610.36666 | [PDF](https://arxiv.org/pdf/2610.36666v1)

**作者:** Zongjun Yang `[一作]` (Columbia University), Christian Kroer `[通讯]` (Columbia University)

**关键词:** `Computer Science and Game Theory` `Optimization` `Reinforcement Learning` `Federated Learning` `Reinforcement Learning` `Contrastive Learning` `Ordinary Differential Equation` `Tabular` `Time Series` `Benchmark`

**🎯 论文内容**

本文研究在多代理的连续第二价格拍卖中，各代理使用预算或 ROI 约束的“pacing”算法时，是否能实现联合无后悔（joint no‑regret）；通过建立变异性与后悔之间的定理，将该问题归结为计算难度，并用 PID 与 OMD 动态系统的数值实验验证了高变异导致后悔线性增长的现象。

**💡 创新点**

创新点在于首次证明：在 i.i.d. 到达的第二价格拍卖中，即使所有代理都使用有效的学习算法，也无法在多项式时间内同时满足所有代理的预算/ROI 约束并实现子线性后悔；并通过高方差→高后悔的理论桥梁，将多代理学习的非收敛性转化为不可解的计算难题。

**🔧 技术方法**

主要技术包括：基于原始-对偶框架的后悔分解与强凸性分析、对流形拍卖的“fluid”模型与随机化转移、以及通过构造特殊分布实现对偶强凸性与约束可行性的证明；还使用 PID 控制与镜像下降（OMD）的离散/连续动力学模拟。

**📊 数据集**

实验使用人工构造的极简分布（如 Rock‑Paper‑Scissors、Beta 混合分布）来模拟三名代理的价值向量，未采用真实广告或搜索数据集。

**📈 对比分析**

论文并未给出传统意义上的基准对比；通过理论证明表明在给定模型下无法获得子线性后悔，而数值实验则显示无论采用 PID 还是 OMD，后悔随时间线性增长，证明了算法在此环境下的失败。

**⚠️ 局限性**

局限性包括：只考虑二价拍卖且预算/ROI 约束；假设商品到达是 i.i.d.，不涵盖更一般的敌对或相关场景；证明依赖多项式时间与指数规模的区分，若拍卖轮数为指数级则可能可行；以及理论与实验均基于人工分布，缺乏对实际广告交易数据的验证。

---

## 133. PRISM-BENCH: Diagnosing Grounded and Faithful Multimodal Reasoning Beyond Final-Answer Accuracy

**arXiv ID:** 2610.36554 | [PDF](https://arxiv.org/pdf/2610.36554v1)

**作者:** Vansh Bhatia `[一作]` (Advanced Micro Devices, Inc.), Emad Barsoum `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `Computer Vision and Pattern Recognition` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Chain-of-Thought` `Prompt Engineering` `Large Language Model` `Retrieval-Augmented Generation` `Multimodality` `Image` `Text` `Benchmark`

**🎯 论文内容**

提出了一个包含1000道专家精心设计、配有理想化推理轨迹的多模态推理基准PRISM-BENCH，用于评估模型的基线性能、推理内容、成本与可解释性。

**💡 创新点**

创新点在于：①为每道题提供人类作者的推理轨迹，支持基于推理内容的诊断；②设计了感知探针（perception probe）与轨迹对齐度（trace alignment）两种诊断指标；③提出了与推理成本相关的效率评估指标（cost‑aware metric）。

**🔧 技术方法**

采用的技术包括：链式思维提示（chain‑of‑thought prompting）、过程奖励模型（PRM）与无奖励模型（critic‑free）对齐度评估、对话式推理与成本评估公式、以及人工设计与审核的四阶段注释流程。

**📊 数据集**

使用的数据集为PRISM-BENCH，包含5个未充分覆盖的推理类别（视觉核心参考/遮挡、手写/OCR、导航/地图推理、多跳推理、科学/医学推理），每类200道题；图像来源涵盖RefCOCO、OpenStreetMap、MedXpertQA、手写样本等。

**📈 对比分析**

在25个开源与闭源模型（包括GPT‑5、Claude Opus 4.6、Gemini 2.5 Pro等）上进行评测。总体准确率最高为66.0%，人类基准为83.9%；在四个可比类中最佳模型落后18.4点。感知探针显示约85%的失败源于感知错误；轨迹对齐度AUROC为0.66，能显著预测答案正确性；成本指标重新排列24.2%的模型对，说明仅靠准确率无法衡量实际部署效率。

**⚠️ 局限性**

局限性包括：①基准仅覆盖5类任务，仍有未覆盖的推理领域；②轨迹对齐度依赖于人工编写的参考轨迹，可能不适用于开放世界输入；③成本指标需要模型暴露推理token数量，闭源API无法使用；④过程奖励模型（VisualPRM）在本数据上表现有限；⑤对模型对齐度的阈值设定及评估方法在不同模型规模下可能需要进一步调优。

---

## 134. Structured Maps for Landmark-Based Route Instructions with LLMs

**arXiv ID:** 2610.36582 | [PDF](https://arxiv.org/pdf/2610.36582v1)

**作者:** Mifu Matsumoto `[一作]` (University of Hyogo), Hiroaki Ohshima `[通讯]` (University of Hyogo)

**关键词:** `Computation and Language` `Recognition` `Retrieval` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Vision Language Model` `Prompt Engineering` `Image` `Text` `Multimodality`

**🎯 论文内容**

本文将局部地图信息转换为结构化SVG文本，以便大型语言模型（LLM）在地标选择任务中直接使用坐标与语义信息，并与传统PNG图像输入进行对比。

**💡 创新点**

创新点在于用可解析的SVG文本取代像素图像，让LLM能更精确地获取地图几何与标签；同时证明通过LoRA/QLoRA微调，小型本地LLM可获得显著的性能提升。

**🔧 技术方法**

采用SVG文本表示、LoRA/QLoRA参数高效微调，以及GPT、Gemma、Llama等多种LLM进行实验。

**📊 数据集**

使用从人类餐厅通行描述与OpenStreetMap匹配构建的两任务数据集（任务1 500例，任务2 339例）进行评估。

**📈 对比分析**

在对比实验中，SVG文本输入在两项地标选择任务上均达到或超过PNG图像输入的准确率；在微调后，小型模型的准确率提升至约49%–55%。

**⚠️ 局限性**

局限包括：评估仅采用精确匹配，忽略多种合理答案；未考虑建筑高度、颜色等视觉显著性特征；数据集规模相对有限。

---

## 135. RangeRectQuant: Compressing Scientific Fields for Every Future Rectangular Sum

**arXiv ID:** 2610.36873 | [PDF](https://arxiv.org/pdf/2610.36873v1)

**作者:** Jiahao Zhang `[一作]` (Zhengzhou University of Aeronautics), Yu Chen `[通讯]` (Zhengzhou University of Aeronautics)

**关键词:** `Computer Vision and Pattern Recognition` `Compression` `Auto Encoder` `Contrastive Learning` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Image` `Physics Related`

**🎯 论文内容**

提出了 RangeRectQuant 算法，对科学栅格图像进行压缩，保证任何后续的轴对齐矩形求和在给定误差预算内，同时保持像素的取值范围并可指定部分像素保持原值。

**💡 创新点**

创新点在于利用前缀和误差的整数区间推导，实现了向后-向前扫描确定合法区间的方法，使得量化可以在保持矩形误差约束、像素范围和指定样本准确性的前提下，完成线性时间、单行工作空间的压缩；同时通过粗细像素分离进一步降低编码量。

**🔧 技术方法**

主要技术包括前缀和误差约束、整数区间动态规划（向后求区间端点、向前选像素）、量化步长 q 的粗细像素划分、LZMA 无损编码、以及对保护像素和缺失像素的特殊处理。

**📊 数据集**

使用的实验数据集包括 2020 Global Human Settlement Layer 的建成面积（100 m 分辨率，值范围 0–10000 m²）和 1984–2024 年的水出现率产品（值范围 0–100，缺失值为 255），在四个城市（开罗、德里、纽约、雅加达）以及 40 个不同位置的 1024×1024 窗口上进行测试。

**📈 对比分析**

通过与无损 JPEG2000、SZ3、MGARD、QPET 等现有压缩器以及固定边距控制、稳定 sigma‑delta 等方法对比，实验显示 RangeRectQuant 在所有误差预算下都能得到更小的文件，平均压缩率提升约 30%–70%，水域数据亦表现出显著的空间节约。

**⚠️ 局限性**

局限性包括：需要预先设定误差预算和量化步长；在极大误差预算时量化步长选择空间受限；处理极大图像仍需较长时间；算法不支持源数据的动态更新；对缺失像素的处理依赖掩码，若掩码不完整可能影响结果。

---

## 136. Replication or Corroboration? An Axiomatic Framework for Evidential Dependence in Multi-Agent Debate

**arXiv ID:** 2610.36151 | [PDF](https://arxiv.org/pdf/2610.36151v1)

**作者:** Hoang-Loc Cao `[一作]` (University of New Brunswick), Hung Cao `[通讯]` (University of New Brunswick)

**通讯引用:** 3241 | [OpenAlex ID](https://openalex.org/A5038366927)

**关键词:** `Multiagent Systems` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `Text` `Review/Survey Paper`

**🎯 论文内容**

研究多智能体辩论中，复制报告与真正的佐证如何被区别对待，并提出一种“谱系商（lineage quotient）”的表示方法，使判断者能忽略同一根证据的多次渲染，却保留不同根证据的加权效应。

**💡 创新点**

核心创新在于：
- 将复制与佐证的区分形式化为最小判定游戏，并证明根复制不变性与阈值均匀复制不变性等价；
- 设计了谱系商，证明它是满足根复制不变性、根标签不变性和对新根敏感性的最粗表示；
- 将该商推广到量化双向论证框架（QBAF），证明新根在符号一致路径上的影响严格传播，并给出紧致的增量模量；
- 通过大规模结构审计（28,224 直接星型、2,052 DAG）及真实 AVeriTeC 数据验证谱系商能实现复制不敏感且对新根严格响应。

**🔧 技术方法**

主要技术手段包括：
- 形式化的最小判定游戏与阈值不变性证明；
- 观察界面与信息边界分析，构造谱系商及其代数性质；
- 对 QBAF 的同构分块（quotient）与签名聚合语义的递推，证明复制不变性与新根响应；
- 通过数值实验与统计检验（bootstrap 置信区间）评估各判定器的表现。

**📊 数据集**

实验数据集：
- 28,224 个直接星型 QBAF 配置；
- 2,052 个多层 DAG（链、分叉、菱形）配置；
- 277 条来自 AVeriTeC 的事实核查数据（127 支持、150 反驳），其中每条报告被视为唯一根并手工标注源标签。

**📈 对比分析**

比较方法：在每个配置下对 5 种判定器（投票、出现计数、Max‑Quad、内容商、谱系商）进行复制干预（八次复制）与新根干预，统计：
- 平均分数变化（RSS_8）和投票翻转率（RFR_8）；
- 新根对分数的符号增益（FLG）和最小增益（MinG）；
- 复制对决议的影响（FASR）。
- 结果表明：谱系商在所有指标上达到零复制影响且保持正的最小增益（≈0.0013-0.0088），而其他判定器要么复制敏感、要么新根无效。

**⚠️ 局限性**

局限性：
- 仅假设根标签的离散分区，未考虑渐进式/概率性依赖；
- 对错误的根分割或合并（false split/merge）未建模，可能导致过度或不足的聚合；
- 不处理伪造身份、强制性内容变更或完全新证据的情况；
- 只在离散、可预见的 QBAF 结构中验证，尚未在更复杂或动态网络上证明。

---

## 137. An Algebra of Transformers

**arXiv ID:** 2610.36795 | [PDF](https://arxiv.org/pdf/2610.36795v1)

**作者:** Yong Wang `[一作]`, Hexiao Kang `[通讯]`

**关键词:** `Logic in Computer Science` `Transformer` `Equational logic` `Formal proof` `Rewrite systems` `Kripke-style bisimulation` `Pomset labelled transition system` `Review/Survey Paper`

**🎯 论文内容**

提出了Transformer的算子代数BAT和APT，给出完整的公理、语义与证明，并扩展到带递归与抽象的APT_τ；

**💡 创新点**

构建了统一的算子分解与真并发代数，证明了公理系统的完备性与一致性，并引入Cluster Fair Abstraction Rule实现无效循环消除；

**🔧 技术方法**

运用了Pomset labelled transition system、equational logic、rewrite systems、Kripke‑style bisimulation、Cluster Fair Abstraction Rule等形式化工具；

**📊 数据集**

无实验数据集，纯理论研究；

**📈 对比分析**

不涉及实验对比，主要通过形式化证明展示方法的正确性与完备性；

**⚠️ 局限性**

局限于有限序列Transformer，缺乏对大规模或连续模型的实用实现，递归与抽象操作相对复杂。

---

## 138. Nemotron-Labs-Diffusion-Uni: Elastic Self-Refining Unified Diffusion LLM with Cross-Paradigm On-Policy Distillation

**arXiv ID:** 2610.36466 | [PDF](https://arxiv.org/pdf/2610.36466v1)

**作者:** Shufan Li `[一作]`, Pavlo Molchanov `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Image Translation` `Knowledge Distillation` `Representation Learning` `Transformer` `Diffusion model` `Vision Language Model` `Supervised Fine-Tuning` `Large Language Model` `Image` `Text` `Multimodality`

**🎯 论文内容**

本论文提出一种统一的离散扩散语言模型（DLLM）<model_name>，通过跨范式的自我策略蒸馏（OPD）和自我细化的掩码扩散过程，实现同时具备视觉理解和视觉生成的能力。

**💡 创新点**

创新点主要有两点：①引入跨范式自我策略蒸馏，允许从自回归视觉语言模型（VLM）和连续扩散生成模型同时蒸馏知识；②设计自我细化的掩码扩散过程，在掩码与清晰状态之间引入多步中间状态，提升生成图像的视觉质量。

**🔧 技术方法**

技术方面采用了Elastic‑MoT两塔结构、SBD（block‑causal）扩散训练、token‑age 级别的多步掩码进程、ball‑tree 路径随机掩码、以及对教师生成的 Refined 输出进行 SFT 损失的自我策略蒸馏。

**📊 数据集**

训练使用了多模态文本‑图像对（MMMU、ChartQA、InfoVQA、AI2D）、对象定位（RefCOCO、ScreenSpot‑Pro）、多模态推理（We‑Math、Geometry3k、MathVerse）、文本‑图像生成（GenEval、DPG、HPSv3）以及图像编辑（ImgEdit）等公开数据集，并以 Nemotron‑3 Nano Omni 与 Qwen‑Image 为教师。

**📈 对比分析**

在视觉理解、推理、文本‑图像生成和图像编辑等任务上，相比之前的统一 DLLM（如 LaViDa‑O、MMaDA、LLaDa‑V、NLD‑VLM）以及连续扩散模型，<model_name> 取得了新的最佳成绩；在 GenEval/DHP、HPSv3、ImgEdit 等基准上，性能接近甚至超越强基线 Flux1‑dev 与 Qwen‑Image，明显缩小了 DLLM 与 AR‑VLM 之间的差距。

**⚠️ 局限性**

局限性包括：①仍需大量算力与长时间训练（64 H100 GPU 26 天）；②模型规模与训练数据混合仍可进一步优化；③自我细化过程对 token‑age 设定与掩码速率敏感，可能在极高分辨率或长序列下效果有限；④跨范式蒸馏对教师模型的依赖使得适配新教师存在一定门槛。

---

## 139. Hy-Nomas: Towards Free-form Interactive World Generation

**arXiv ID:** 2610.36729 | [PDF](https://arxiv.org/pdf/2610.36729v1)

**作者:** Qiaoling Chen `[一作]` (Tencent Hy Visual Generation Team), Kai Zou `[通讯]` (Tencent Hy Visual Generation Team)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Prompt Engineering` `World Model` `Vision-Language-Action Model` `Auto Encoder` `Contrastive Learning` `Score-based Model` `Reinforcement Learning` `Video` `Text` `Multimodality`

**🎯 论文内容**

开发了一款基于8B扩散Transformer的交互式视频世界模型，可在连续生成过程中处理自由文本指令与连续相机控制，并通过结构化视频字幕与动作感知因果注意力实现多轮交互与记忆保持。

**💡 创新点**

创新点包括：①动作感知因果注意力（将描述性文本跨段可见、动作文本局部可见）解决跨段动作重复与时序错位；②Prompt Enhancer将用户指令与历史转化为结构化字幕；③多阶段训练曲线（从段级因果、动作对齐、相机控制到高质量精炼、分段强制与扩散蒸馏）；④自研的高效训练与推理基础设施（多级缓存、结构化遮罩、FLOPs基序列打包、单卡INT8+滑动窗口注意力）。

**🔧 技术方法**

技术包括：HunyuanVideo 1.5基础、DCAE‑V压缩VAE、Qwen2.5‑VL‑7B文本编码、MMDiT双流Transformer、相机交叉注意力、线性流匹配训练、分段因果与动作掩码、相机轨迹编码、分段强制、扩散蒸馏、LoRA微调、单卡INT8+滑动窗口注意力与PixelUMM解码器。

**📊 数据集**

使用了海量多源视频数据（真实与虚拟环境），包括带相机轨迹、动作标签、场景变化的结构化字幕；在后续阶段使用Kimi‑K3高质量字幕、ViPE相机轨迹校验等；训练数据分阶段自适应采样、聚类去重、层级标签过滤。

**📈 对比分析**

通过自建的Hy‑Nomas‑Bench（280个多轮交互、相机导航、交互混合、空间记忆四个子集）与Genie 3、LingBot‑World v2、Happy Oyster三模型进行人类评估。该模型在多轮文本交互、物理合理性、相机控制上均取得最高净胜率（约60‑75%），在混合控制下表现更优；空间记忆仍弱于Genie 3，Win率仅13%。

**⚠️ 局限性**

局限：①空间记忆性能不足，尤其在多次视角回访时易出现物体状态或布局漂移；②对复杂相机轨迹的泛化仍有限，LoRA微调时若训练数据缺乏多样相机动作会导致控制能力退化；③高质量视频生成依赖多阶段训练与大规模算力，部署成本高；④单卡推理仍受限于显存与解码器效率，长视频仍需多卡或量化方案。

---

## 140. Stiffness coefficients in granular micromechanics determined with three-point bending experiments and digital image correlation for a bimodulus material

**arXiv ID:** 2610.36862 | [PDF](https://arxiv.org/pdf/2610.36862v1)

**作者:** Mohammad Mahdi Nayeban `[一作]` (University of Catania), Luca Placidi `[通讯]` (Telematic University Pegaso)

**关键词:** `Computational Engineering, Finance, and Science` `Optimization` `Digital Image Correlation (DIC)` `Finite Element Analysis` `Optimization (Pattern Search + BFGS)` `Image` `Physics Related`

**🎯 论文内容**

研究提出了一套基于三点弯曲实验得到的全场数字图像相关（DIC）应变数据，对颗粒微观力学框架下的双模量本构模型进行反演识别，最终得到压缩与拉伸相应的交互刚度参数。

**💡 创新点**

创新点在于：①首次将全场 DIC 应变信息与颗粒微观力学模型结合；②利用中性轴偏移提供物理启发的初始参数；③采用混合模式（模式搜索+准牛顿）优化策略实现高效收敛；④对参数分辨率进行了定量分析。

**🔧 技术方法**

技术手段包括：三点弯曲实验、数字图像相关（DIC）获取 G_11 领域；有限元求解（COMSOL）与 MATLAB 的耦合实现模型更新；混合优化（模式搜索+BFGS）以 NRMSE 为目标函数进行参数识别。

**📊 数据集**

数据集为：在双浸渍石墨 EG‑92E 试样上进行的三点弯曲实验，最大载荷 3500 N，ROI 宽 80 mm、全高度；DIC 采集 2448×2052 像素图像，获得 G_11 全场；此外通过无载荷 DIC 评估测量噪声。

**📈 对比分析**

比较方法是把数值模型预测的 G_11 场与实验 G_11 场做全场 NRMSE 误差比较；优化后误差从 18.6 % 降至 12.3 %，中性轴位置差异小于 1 mm，且误差主要集中在载荷作用区附近。

**⚠️ 局限性**

局限性包括：仅使用 G_11 作为目标，导致切向刚度参数分辨率差；仅在弹性范围内验证；未考虑塑性或显著的局部载荷转移效应；特征长度 L 被固定，未进行识别；第一梯度模型对加载区细节描述不足。

---

## 141. SWIM: Learning What Matters from Behaviors for JEPA World Models

**arXiv ID:** 2610.36662 | [PDF](https://arxiv.org/pdf/2610.36662v1)

**作者:** Jiaming Hu `[一作]` (Boston University), Qingjun Cui `[通讯]` (Unity Technologies)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Transformer` `Recurrent Neural Network` `World Model` `Contrastive Learning` `Reinforcement Learning` `Auto Encoder` `Image` `Video` `Tabular` `Benchmark`

**🎯 论文内容**

提出一种基于逆动力学的状态特征加权方法 SWIM，并将学习到的加权状态度量用于 JEPA 视觉世界模型的距离校准，从而实现更有效的基于图像的规划。

**💡 创新点**

创新点在于利用记录的动作信息自动学习每个状态坐标的行为相关权重，无需手工挑选或均匀加权；权重在训练时通过逆动力学预测与判别损失共同优化，随后用于无额外计算的规划阶段。

**🔧 技术方法**

核心技术包括：1) 逆动力学模型（带门控的 MLP）预测动作以获得状态变位信息；2) 加权正则化与判别（hinge / InfoNCE）来学习门控；3) 将学习到的权重应用于 SCALE 的状态距离，调校 latent 空间；4) 使用 LeWM 作为基准模型，CEM 与 iCEM 进行规划。

**📊 数据集**

使用 OGBench 视觉操纵任务数据集，包括 Cube、Cube-Double、Cube-Triple、Cube-Quadruple 和 Scene 共 5 个任务，均提供完整的模拟器状态与离线轨迹。

**📈 对比分析**

与 LeWM、LeWM+SCALE（均匀权重）以及预训练视觉特征 baseline 进行对比。实验结果显示 SWIM 在 CEM 与 iCEM 两种规划器上均比均匀权重提升 1–6% 的成功率，且在 Scene 任务上提升最为显著；当加入无关变量时，SWIM 能显著下调其权重并恢复成功率。

**⚠️ 局限性**

局限性包括：1) 依赖于已提供的状态维度，若重要信息缺失则无法学习；2) 需要足够覆盖的离线轨迹才能准确估计行为相关权重；3) 目前仅在 SCALE 目标下验证，尚未在其他状态监督或更复杂环境中深入评估。

---

## 142. Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss

**arXiv ID:** 2610.36202 | [PDF](https://arxiv.org/pdf/2610.36202v1)

**作者:** Oz Amram `[一作]` (Independent), Oz Amram `[通讯]` (Independent)

**关键词:** `Machine Learning` `Anomaly Detection` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Contrastive Learning` `Text`

**🎯 论文内容**

本文提出一种利用基础模型对对话内容进行概率评估（Persona-Header Likelihood Evaluation，PHLE）的方法，用以检测大型语言模型在隐藏系统提示下的潜在意图。

**💡 创新点**

创新点在于证明基础模型已内含后期训练中选择的角色特征，并能跨模型家族读取潜在信号；提供零样本对隐藏提示的分类，并显著优于提示式分类器，验证了Persona Selection Model（PSM）的关键预测。

**🔧 技术方法**

采用基础模型的对数似然比统计，对不同角色头部描述进行评分，并对多条回答累计求和；使用自助法估计AUC不确定性，且对比了生成模型的“天花板”评分与提示式分类器。

**📊 数据集**

实验数据包括由Olmo 3 7B和Qwen2.5 7B系列模型生成的两类数据：① 300个道德困境的简短回答；② 1000条无语义关联的数字序列；隐藏系统提示覆盖“爱猫”“爱鹰”“爱火车”“隐秘伤害”“隐秘友好”“无提示”六种情形。

**📈 对比分析**

与基线相比，PHLE在AUC上实现了高达0.94（隐秘伤害vs无提示）和0.84（爱鹰vs爱猫的数字序列）等显著性能；提示式分类器（Olmo与GPT‑4.1）接近随机；对手生成模型的“天花板”评分最高；跨家族实验显示另一家族的基础模型亦能读取相同信号。

**⚠️ 局限性**

局限性包括仅使用7B参数模型，隐藏意图通过系统提示而非训练实现，需事先给定候选假设且对过滤方法敏感；缺乏对真实多模态或更大模型的验证；得分需要校准才能在实际审核中直接使用。

---

## 143. Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs

**arXiv ID:** 2610.35985 | [PDF](https://arxiv.org/pdf/2610.35985v1)

**作者:** Tianqi Chen `[一作]`, Jian Cheng `[通讯]`

**关键词:** `Machine Learning` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text`

**🎯 论文内容**

本文提出了一种基于门控的残差量化方法，对条件记忆表进行逐查找位深度分配，从而在不需再训练的前提下实现大幅压缩与高效推理。

**💡 创新点**

创新点在于：①用1‑bit残差梯度存储表行，并在只读1级时即可计算门值；②按门值动态决定每次查找读取的层级，实现对长尾知识的精准保留；③在无额外训练的条件下，将表存储从GPU 23.6 GiB降至3.0 GiB，显著提升推理吞吐。

**🔧 技术方法**

使用的技术包括：1‑bit残差量化（TCQ/INT/PQ）梯度、门控加权、基于门值的查找深度阈值分配、vLLM框架集成，以及对GPU/主机内存的读写优化。

**📊 数据集**

实验数据集涵盖 PopQA、TriviaQA、NQ‑open、CMMLU 以及 1.15×10⁸ RedPajama 词表，用于统计访问频率和门值评估。

**📈 对比分析**

通过与 FP8 表、均匀重构（1/2/3 层）以及三种查找选择规则（门值、预期门、频率、随机）比较，门控量化在 36 个 PopQA 对比中均显著优于其它规则；在 2.10 位读取下误差仅比 FP8 低 0.2 点；表内存压缩 7.8×，vLLM 通过率提升 11%–31%。

**⚠️ 局限性**

局限性：仍需主机内存存放深层残差，增加跨设备数据传输；方法专门针对条件记忆层，未验证在其它类型表或模型中的泛化；门值依赖上下文，可能在极端稀缺语境下估计不准确。

---

## 144. No Tokenizer, Still a Gap: Cross-Lingual Fairness of Homoglyph and Invisible-Character Watermarking

**arXiv ID:** 2610.36432 | [PDF](https://arxiv.org/pdf/2610.36432v1)

**作者:** Alexander Nemecek `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**关键词:** `Computation and Language` `Safty and Privacy` `Explainability and Interpretability` `Large Language Model` `Contrastive Learning` `Text` `Benchmark`

**🎯 论文内容**

对 24 个书写体系、40 种语言中的同义字（homoglyph）和不可见字符水印进行容量、存活率、检测所需文本量以及公平性评估。

**💡 创新点**

提出了按语言细化的容量计算框架、处理“常规”与“有意”变换的层级模型、以及覆盖信息的键控检测器；通过公平性分析揭示脚本决定容量差异，资源水平影响仅体现在脚本层面。

**🔧 技术方法**

使用 Unicode 17.0 的同义字表、UTS #39 的混合脚本检测与骨架映射、Unicode 标准化（NFC、NFKC）、可插入不可见字符集合（零宽空格、连接符等），并结合统计容量公式、有效容量、AUC 与最小检测长度等指标。

**📊 数据集**

基于 FLORES+ dev‑test 1,012 句子平行语料库；配合 Unicode 17.0.0 与 CLDR 48.2.0 的字符与脚本数据。

**📈 对比分析**

对每种语言在不同变换（如复制粘贴、NFKC、混合脚本检测、完整骨架映射）下的容量、有效容量、检测 AUC 以及最小检测长度进行对比；容量范围 7.09–2.14 bpc（3.31 倍），经变换后缩至 2.59–3.45 倍，复制粘贴时扩至 8.49 倍；检测长度从 1.85–4.95 码点（清洁）到 2–93 码点（复制粘贴）不等。

**⚠️ 局限性**

仅 24 个脚本中 20 个仅含单语种，导致脚本内方差估计有限；资源分类依赖单一公开体系，匹配不确定；使用单一语料与单一 Unicode/CLDR 版本；可视化与渲染依赖字体与引擎；检测器为覆盖信息上限，未考虑无原文情形；以及对多字符同义字与选择器等未计入容量。

---

## 145. What Has Actually Been Proved? Auditable Paper-to-Lean Alignment with LeanTrail

**arXiv ID:** 2610.35926 | [PDF](https://arxiv.org/pdf/2610.35926v1)

**作者:** Fabricio Venturim `[一作]` (Fundação Getulio Vargas), Jorge Poco `[通讯]` (Fundação Getulio Vargas)

**关键词:** `Logic in Computer Science` `AI Code Assistant` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

设计并实现了LeanTrail，一种后置审计工具，用来评估Lean项目中声明是否真正对应并完整证明了原论文中的命题、定义、引理等论文项；

**💡 创新点**

将正式化进度拆分为语义对应和证明完整性两步，使用语言模型仅判定语义对应，证明完整性通过确定性依赖图审计，并将两者融合为五种可检验的覆盖状态；提供可检验界面和可导出的JSON任务；

**🔧 技术方法**

使用检索增强生成技术（BM25+递归排名融合）检索候选声明，语言模型（Gemini 3.8 Flash与GPT‑6 Sol）进行语义判别，确定性证明审计（依赖图遍历+证明证书）以及融合规则；

**📊 数据集**

使用三大Lean项目FinEqs、APAP、Ringel共计98个论文项（7 Formalized、5 Statement only、23 Partial、13 Blueprint only、50 Missing）作为评估数据集；

**📈 对比分析**

对检索、语义判断和覆盖状态分别评估召回@k、准确率；在两个模型上各运行三次，最终覆盖状态正确率为91.8–92.5%，与人工共识高度一致，重复运行一致率达到96–97/98；比较显示模型误差方向不同，GPT‑6 Sol在误判上不产生过度声明；

**⚠️ 局限性**

评估仅覆盖3个项目和98项，缺乏跨项目泛化验证；语义对应仍依赖模型，错误可能导致错误覆盖状态；证明审计仅跟踪Lean标准公理，未验证项目自定义公理；提取假设论文为结构化文本，未处理非结构化或多版本论文；

---

## 146. Balanced State Subspaces for Mamba-2 State Compression

**arXiv ID:** 2610.36617 | [PDF](https://arxiv.org/pdf/2610.36617v1)

**作者:** Shang Tingyu `[一作]` (Shaanxi Normal University), Cao Wenfei `[通讯]` (Shaanxi Normal University)

**关键词:** `Machine Learning` `Compression` `Representation Learning` `Transformer` `Auto Encoder` `Contrastive Learning` `SVD` `Hankel Spectrum Analysis` `Text`

**🎯 论文内容**

对Mamba-2的循环状态做后训练压缩，利用子空间旋转和层级分配实现低维化

**💡 创新点**

提出基于平衡截断的全经验Gramian子空间变换，并引入HARP层级Hankel分配与检索敏感性保护，能同时保留输入-输出映射与检索性能

**🔧 技术方法**

使用经验可控与可观测Gramian估计、SVD求平衡基、平衡截断、Hankel谱与检索敏感性探测技术

**📊 数据集**

WikiText-2 语言建模和 LAMBADA 词义预测作为评估数据集

**📈 对比分析**

与 GHOST、Magnitude、Saliency、随机等方法对比，BASS+HARP 在 50% 与 25% 维度压缩下仅增加 0.3–1.7 ppl，显著优于 GHOST；检索准确率保持或提升

**⚠️ 局限性**

在极高压缩率下仍存在 20–25% perplexity 增长，评估范围限于文本任务，校准集选择对结果敏感，旋转引入额外解码开销需进一步优化

---

## 147. Adaptive Calibration of RF Digital Twins via Budget-Constrained Radio Measurement Selection

**arXiv ID:** 2610.36406 | [PDF](https://arxiv.org/pdf/2610.36406v1)

**作者:** Yalin E. Sagduyu `[一作]` (Nexcepta), Hossein Jafari `[通讯]` (Nexcepta)

**关键词:** `Networking and Internet Architecture` `Optimization` `Federated Learning` `Computational Efficiency` `Diffusion model` `Score-based Model` `Simultaneous Localization and Mapping` `World Model` `Optical Flow` `Tabular` `Time Series` `Point Cloud` `Mesh` `Physics Related`

**🎯 论文内容**

本文研究在有限测量预算下，通过主动选择测量位置来校准基于物理的无线射频数字孪生模型。

**💡 创新点**

创新点在于提出并评估了多种在线测量选择策略（包括结构化、逐步和基于不确定性的自适应方法），证明在预算受限时可显著提升校准精度。

**🔧 技术方法**

技术包括使用OpenStreetMap语义地理数据、Blender进行三维建模、Sionna的确定性射线追踪以及软件定义无线电测量。

**📊 数据集**

数据集为30个现场测点的实测路径损耗（USRP B210）与对应的射线追踪预测结果，频率1.2 GHz。

**📈 对比分析**

通过离线蒙特卡洛模拟比较五种采样策略，评估平均和最差绝对偏差，结果显示AUS策略在低预算下误差下降最快，最差误差下降最多（≈91%）。

**⚠️ 局限性**

局限性包括仅校正全局偏移、对小规模测点集的验证、以及对更复杂环境或动态场景的适用性未作评估。

---

## 148. Learning and Reasoning with Geometry: How Generation Helps Spatial Understanding

**arXiv ID:** 2610.36022 | [PDF](https://arxiv.org/pdf/2610.36022v1)

**作者:** Rui Yang `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**通讯引用:** 39674 | [OpenAlex ID](https://openalex.org/A5078109015)

**关键词:** `Computer Vision and Pattern Recognition` `Depth Estimation` `Generation` `Representation Learning` `Reinforcement Learning` `Transformer` `Vision Language Model` `Flow-based Model` `Reinforcement Learning` `Prompt Engineering` `Diffusion model` `Contrastive Learning` `Image` `Text` `Point Cloud`

**🎯 论文内容**

提出了 GeoUG——一种统一的视觉语言模型，能够同时进行几何生成（深度和多视图）与空间理解，并在推理时通过生成几何证据来校正答案。

**💡 创新点**

创新点包括：1）多尺度度量深度编码器（MMDC）实现对不同深度范围的精细表示；2）在训练中将几何生成与空间理解联合优化，获得任务依赖的提升；3）在推理阶段引入草稿条件化的自适应交替推理策略，利用强化学习平衡生成成本与准确率。

**🔧 技术方法**

技术方法涵盖：基于 BAGEL 的统一生成-理解框架；流匹配（flow‑matching）目标实现几何生成；MMDC 的三通道深度编码与解码；强化学习（RLVR）和门控策略用于生成决策；以及多尺度数据增强与 affine 对齐的深度评估。

**📊 数据集**

使用的数据集包括：CV‑Bench‑3D、3DCueBench、DA‑2K（视觉与文本版）、MindCube、MMSI‑Bench、NYU‑v2、KITTI、ETH3D、Sintel、DIODE 等；并在 800 条精挑细选的 3DCueBench 题目上对几何难度进行控制。

**📈 对比分析**

与现有 VLMs（BAGEL、Qwen、InternVL、SenseNova 等）及深度估计器 DA3 进行比较，GeoUG 在 CV‑Bench‑3D、3DCueBench、DA‑2K 上的圆形准确率分别达到 98.5%、97.3% 与 97.9%，接近或超过 DA3；在多视图推理任务中显示更高的 PSNR 与重建质量；并通过自适应生成显著减少生成调用量，仅比始终生成低 0.7% 的准确率损失。

**⚠️ 局限性**

局限性包括：生成质量不稳定时会导致错误修正；RL 门控策略对超参数 λ 敏感，需调优；多尺度编码虽提升精度但训练成本更高；模型在极大尺度或复杂遮挡场景下的深度推断仍受限；目前仅针对单图像空间推理，跨视角一致性尚未充分验证。

---

## 149. On the VC Dimension of Autoregressive Linear Threshold Functions

**arXiv ID:** 2610.36970 | [PDF](https://arxiv.org/pdf/2610.36970v1)

**作者:** Ilan Doron-Arad `[一作]` (MIT), Elchanan Mossel `[通讯]` (MIT)

**关键词:** `Machine Learning` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Review/Survey Paper`

**🎯 论文内容**

研究了自回归线性阈值函数（autoregressive LTFs）的 VC 维度，证明在自回归步数 T≥d² 时 VC 维度为 Θ(d²)，并首次展示在随机化（logistic 采样）模型下，VC 维度可提升到指数级；

**💡 创新点**

突破了已知的 d 与 d² 问题，给出精确的 Θ(d²) 下界；同时证明 deterministic 与 stochastic 自回归模型之间存在指数级的 VC 维度分离，首次在此类模型中发现此类结构性差异；

**🔧 技术方法**

利用将标签矩阵编码进权重的递归块构造、模运算、块化状态机与单一 LTF 的实现；在随机化模型中使用二进制编码、复数正弦分析和 Logistic 采样概率的精确控制；

**📊 数据集**

该工作为纯理论分析，无实验数据集；

**📈 对比分析**

与先前的 O(d²) 计数上界相匹配，证明下界 Θ(d²) 与上界一致；在随机化模型中，VC 维度从多项式提升到指数，展示了理论上的巨大性能差异；

**⚠️ 局限性**

结果仅适用于理论模型，未给出具体学习算法；需要非常长的自回归步数 T；随机化模型依赖于精确的 Logistic 参数与大样本；对实际 LLM 的可解释性与实现仍有限。

---

## 150. Plenty of Room in the Middle: Revealing Opportunities on a GPU Cluster via Fine-Grained Monitoring

**arXiv ID:** 2610.36329 | [PDF](https://arxiv.org/pdf/2610.36329v1)

**作者:** Andrew Sheinberg `[一作]` (Princeton University), Kai Li `[通讯]` (Princeton University)

**关键词:** `Artificial Intelligence` `Optimization` `Computational Efficiency` `Time Series`

**🎯 论文内容**

开发并部署了细粒度（100 ms）GPU监控工具，对普林斯顿大学89节点、319 GPU的研究集群进行长周期监测，并分析GPU利用率、内存占用与通信模式等；

**💡 创新点**

采用DCGM的SM Activity等细粒度指标，以100 ms采样实现对GPU SM/内存利用率的精确评估，揭示70 %作业低SM/内存占用、80 %时间估计失准、以及多GPU工作负载周期性通信，为多任务调度与资源共享提供依据；

**🔧 技术方法**

Nvidia DCGM API、轻量级C实现、SQLite本地数据库、SLURM作业信息抓取、MIG/MPS/Unified Memory等技术；

**📊 数据集**

10天普林斯顿GPU集群数据，7,399个作业、150名用户、12系，包含GPU SM、内存、PCIe、NVLink等指标；

**📈 对比分析**

与先前使用1 s或100 s采样的监控系统比较，细粒度采样能捕获短暂空闲与通信峰值；相较于NVML/SMI对SM利用率的高估，DCGM提供更准确的利用率；整体GPU利用率仅29%，内存16%，表明改进空间；

**⚠️ 局限性**

采样最高为10 Hz（100 ms），无法捕获微秒级内核；仅分析10天作业映射；多GPU作业样本稀少；仅两种GPU型号混合，未充分考虑硬件异构对网络/通信的影响；

---

## 151. DuTri: Dual-Branch Modeling with Triadic Factorization for Multimodal Multi-Scenario Recommendation

**arXiv ID:** 2610.36836 | [PDF](https://arxiv.org/pdf/2610.36836v1)

**作者:** Jintao Huang `[一作]` (Shenzhen University), Zhong Ming `[通讯]` (Shenzhen University)

**关键词:** `Information Retrieval` `Recommendation System` `Contrastive Learning` `CANDECOMP/PARAFAC (CP) Factorization` `Residual Quantization` `Conditional Gated Multimodal Fusion` `Multimodality`

**🎯 论文内容**

提出了 DuTri 双分支框架，分别学习协同偏好和多模态融合，并通过对齐实现两者协同；

**💡 创新点**

创新点在于使用三元因子分解（CP）和残差量化构建跨场景协同偏好，并用用户/场景偏好指导多模态权重生成，以及通过对比学习实现跨分支对齐；

**🔧 技术方法**

技术包括 CANDECOMP/PARAFAC (CP) 因子分解、残差量化、条件门控多模态融合、对比学习对齐、场景特定 MLP 预测等；

**📊 数据集**

使用 AntM^2C（阿里巴巴五场景数据）和 Amazon（三类商品场景）两个真实数据集；

**📈 对比分析**

与多模态推荐基线（Diff-MSIN, DMF, etc.）和多场景推荐基线（STAR, HMoE, PLE, AMS 等）进行对比，DuTri 在所有场景下均取得最高的 WAUC，且推理速度与 AMS 相近；

**⚠️ 局限性**

局限性包括对 CP 超参数和对比对齐权重较敏感，模型参数量相对较大，且在某些场景下量化或对齐策略改动效果不显著。

---

## 152. From Solver Trajectories to Teaching Trajectories: Cognition-Aligned Reasoning Distillation

**arXiv ID:** 2610.36445 | [PDF](https://arxiv.org/pdf/2610.36445v1)

**作者:** Yashuo Luo `[一作]` (Beihang University), Xu Wang `[通讯]` (Beihang University)

**关键词:** `Artificial Intelligence` `Knowledge Distillation` `Computational Efficiency` `Explainability and Interpretability` `Transformer` `Large Language Model` `Reinforcement Learning` `Chain-of-Thought` `Prompt Engineering` `Supervised Fine-Tuning` `Text` `Benchmark`

**🎯 论文内容**

研究了知识蒸馏，提出CARD框架，将大型模型的推理轨迹转换为与小模型认知匹配的教学轨迹；

**💡 创新点**

创新点在于通过RL优化的轻量化Reformatter，结合分布对齐奖励和逻辑连贯奖励，解决认知鸿沟和失败步骤问题；

**🔧 技术方法**

使用GRPO（Generalized Reward-Predictive Off-policy）训练Reformatter，利用KL对齐奖励、失败步骤判定（FSF）和信息泄漏检测；

**📊 数据集**

使用DeepMath-103K（DeepSeek-R1生成的三条CoT轨迹）以及数学推理基准GSM8K、MATH-500、AGI-MATH、AMC2023、AIME2024/25，OOD基准ARC-Challenge和MMLU-Pro；

**📈 对比分析**

与Vanilla SFT、MCC-KD、MoRSD、Mix TokenSkip、Mix Small Teacher等方法对比，CARD在ID平均提升约2-3分，在OOD平均提升13-15分，表现最优；

**⚠️ 局限性**

局限在于仅验证至7B以下学生模型，未评估更大模型或多轮交互任务，且未探索非数学推理场景。

---

## 153. Robust LLM Data Selection via Content-Invariant Gradient Attribution

**arXiv ID:** 2610.36755 | [PDF](https://arxiv.org/pdf/2610.36755v1)

**作者:** Jia Gu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Huawei Shen `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `Machine Learning` `Data-Centric Learning` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Prompt Engineering` `Text`

**🎯 论文内容**

提出了一种名为 Content-Invariant Gradient Attribution (CIGA) 的数据选择框架，用于指令微调大型语言模型，并在梯度归因中消除表面形式偏差。

**💡 创新点**

创新点包括：① 通过统计梯度幅度与跨表面形式变体的离散度来挑选“稳定而有影响力”的层；② 在该层上进行梯度相似度计算并动态融合语义相似度（RRF），实现梯度与语义的互补；③ 仅使用一次预热微调检查点和单层梯度，大幅降低归因成本。

**🔧 技术方法**

主要技术手段有：warm-up fine-tuning、consistent influential layer selection (S^ci 评分)、Adam 预处理梯度、动态特征融合（RRF）、语义嵌入与余弦相似度、A/B 评估、t-SNE 可视化。

**📊 数据集**

使用多源指令数据集（FLAN‑v2、CoT、Dolly、OASST1、Alpaca‑GPT4、GSM8K）构建训练池；在 TyDiQA、BBH、GSM8K、MMLU、HellaSwag 五个评估基准上进行实验。

**📈 对比分析**

与 LESS、TracIn、Cos、IFD、SelectIT 等方法对比。CIGA 在 5% 训练子集下平均 Jaccard 一致性提升约 9–10%；在所有基准任务上宏平均表现均为最高，并在多任务中接近全量微调效果。归因成本相比传统多检查点方法降低约 71%。

**⚠️ 局限性**

局限性：① 仍需对整个候选数据集执行前向后向传播，计算开销大；② 需要针对目标任务准备验证集，缺少验证集时效果受限。

---

## 154. ML-OLLA: Link Adaptation for Real-Time Traffic in 5G/6G NTN Systems

**arXiv ID:** 2610.36147 | [PDF](https://arxiv.org/pdf/2610.36147v1)

**作者:** Artur Magamedov `[一作]` (HSE University), Evgeny Khorov `[通讯]` (HSE University)

**关键词:** `Networking and Internet Architecture` `Optimization` `Autonomous Driving` `Federated Learning` `Computational Efficiency` `Link Adaptation` `HARQ-CC` `CQI statistical` `Sliding Window Estimation` `Discrete Optimization`

**🎯 论文内容**

提出了一种针对5G/6G NTN实时流量的多层外环链路自适应算法ML-OLLA。

**💡 创新点**

创新在于为每一次传输尝试分配不同的目标误码率和独立的OLLA过程，并采用低复杂度的数据驱动参数选择方法。

**🔧 技术方法**

使用了链路自适应、HARQ-CC、CQI统计、滑动窗口估计、离散优化以及ns-3仿真平台。

**📊 数据集**

实验基于ns-3中的3GPP NTN通道模型与仿真场景，未使用公开数据集。

**📈 对比分析**

通过与传统OLLA、基于SINR的OLLA以及最稳健MCS方法对比，ML-OLLA在相同可靠性与延迟下将系统容量提升四倍，延迟阈值从60ms降至40ms。

**⚠️ 局限性**

局限性包括未考虑HARQ-CC增益、模型对真实链路环境的适应性有限以及算法对gNB处理能力的要求。

---

## 155. A fully articulated humanoid dexterous hand with human-level joint structure enabled by linkage-based actuation

**arXiv ID:** 2610.36789 | [PDF](https://arxiv.org/pdf/2610.36789v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Robotics`

---

## 156. Texture Consolidation of Generated Multi-View Renderings

**arXiv ID:** 2610.36098 | [PDF](https://arxiv.org/pdf/2610.36098v1)

**作者:** Hongyi Liu `[一作]` (Johns Hopkins University), Misha Kazhdan `[通讯]` (Johns Hopkins University)

**通讯引用:** 2138 | [OpenAlex ID](https://openalex.org/A5001385960)

**关键词:** `Graphics` `Restoration` `Generation` `Data Synthesis` `Optical Flow` `Diffusion model` `Gradient Domain` `Finite Element Method` `Variational Equation` `Mesh-based Registration` `Mesh` `Image` `Point Cloud`

**🎯 论文内容**

本文提出了一种基于小量生成渲染与纹理图集的纹理合成管线，先通过反投影获取带置信度的纹理颜色，再使用面向网格的光流注册对齐纹理，最后在梯度域中融合得到无缝纹理图像。

**💡 创新点**

创新点在于：①将置信度融入光流与梯度域拼接的能量函数，①采用基于有限元的四边形积分实现纹理空间与网格空间的精细耦合，②设计了结合法线与深度的置信度估计来缓解几何不一致导致的反投影误差。

**🔧 技术方法**

使用了反投影、法线与深度置信度计算、基于变分方程的网格光流求解、梯度域拼接、以及基于四边形积分的有限元离散化技术。

**📊 数据集**

采用了由扩散模型生成的合成多视图渲染和对应的纹理图集，实验主要基于自制的三维模型（如Bear、ShapeNet等）进行评估。

**📈 对比分析**

通过与传统RGB‑D图像配准与融合方法（如光流+梯度拼接、基于视角选择的混合）对比，实验表明在稀疏生成视角下本方法能够显著降低鬼影和缝隙，得到更平滑、连贯的纹理，性能表现优于现有方法。

**⚠️ 局限性**

局限性包括：需要精确的相机参数；在极度几何不一致或极度稀疏的视角下效果下降；光流与梯度拼接的计算量较大；尚未在真实RGB‑D数据集上进行充分验证。

---

## 157. RoboSTD: Zero-Shot Single-to-Dual Transfer via Sagittal Mirroring for Bimanual Learning

**arXiv ID:** 2610.36685 | [PDF](https://arxiv.org/pdf/2610.36685v1)

**作者:** Hongyu Wu `[一作]` (Beijing Academy of Artificial Intelligence), Guocai Yao `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Vision-Language-Action Model` `Prompt Engineering` `Reinforcement Learning` `Video` `Text` `Sequential`

**🎯 论文内容**

提出RoboSTD框架，利用单臂演示通过矢状面镜像与LLM生成的时空约束，构造零样本双臂监督并训练双臂机器人策略。

**💡 创新点**

创新点在于①零样本单臂到双臂转移；②结合几何对称的矢状镜像实现可执行对侧演示；③使用LLM自动生成臂分配、先后顺序与冲突约束，实现空间时间协调；④在不收集双臂演示的前提下缓解双臂数据瓶颈。

**🔧 技术方法**

技术手段包括：矢状面镜像映射（观测、关节、动作）；LLM（GPT‑4.1）推断时空约束；基于语言的双臂策略π；URDF映射与几何对齐；时间同步与执行重组。

**📊 数据集**

使用的主要数据集为：单臂演示（实验中收集50/50演示、1,000演示；Open X‑Embodiment用于偏差分析）；RoboTwin 2.0仿真任务集（10个多样任务）；AgileX ALOHA实机平台真实任务（Bowl Placement、Towel Storage、Flower Arrangement、Cup Collection、Sandwich Making）。

**📈 对比分析**

与单臂直接监督和Oracle（本地双臂演示）比较：RoboSTD在10个RoboTwin任务中平均成功率为66.5%，仅比Oracle低1.1pp；在Cup Collection任务中平均收集杯数4.22，超过Oracle 3.92；在Sandwich Making中完成率50%与Oracle一致。低配任务成功率从40%提升至77%。此外，RoboSTD显著减小Bias Gap与Density Gap。

**⚠️ 局限性**

局限性：依赖机器人几何对称，非对称机械臂或极端空间布局可能不适用；LLM推断误差或不可解释的约束可能导致失配；需要一定量的单臂演示来覆盖任务空间；对高度复杂的时空协作（如多阶段交叉动作）仍可能不足；安全性和执行精度受限于镜像映射的精确性。

---

## 158. Confident First, Reflective Second: Reinforcement Learning of Confidence-Preserving Reflection for Video Understanding

**arXiv ID:** 2610.36467 | [PDF](https://arxiv.org/pdf/2610.36467v1)

**作者:** Zihao Lin `[一作]` (University of California, Davis), Lifu Huang `[通讯]` (University of California, Davis)

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Video Understanding` `Transformer` `Reinforcement Learning` `Vision Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Video` `Text`

**🎯 论文内容**

提出了一个在视频理解任务中通过先给出初始答案再进行反复检索与修正的循环训练框架，旨在提升初始答案的可靠性并减少置信度侵蚀。

**💡 创新点**

创新点在于提出“先自信后反思”原则，并针对置信度侵蚀的三条渠道——奖励偏差、参数漂移和梯度冲突，分别设计了初始答案信用、已验证决策锚定和不对称梯度保护三种机制。

**🔧 技术方法**

使用强化学习（基于GRPO风格的策略梯度）与多阶段策略路由，结合参考模型的KL约束、奖励重塑以及梯度投影技术来实现上述机制。

**📊 数据集**

主要在三大视频多选基准上进行实验：Video‑MME、MLVU（m‑avg）和LongVideoBench。

**📈 对比分析**

与同尺寸7B模型（如Qwen2.5‑VL‑7B、VL‑Rethinker、Video‑R1、TimeSearch‑R等）对比，取得平均67.0%准确率，比最佳对手高1.7个百分点；在初始答案与最终答案上分别提升4.1点与3.7点。

**⚠️ 局限性**

局限性在于只研究了单一的Qwen2.5‑VL‑7B backbone与单一的答案‑检索‑修正循环，实验仅在长视频多选任务上进行且每个配置仅跑一次，未覆盖其他循环结构或更大规模的模型。

---

## 159. CARD: Contact-Anchored Retargeting for Dexterous Manipulation

**arXiv ID:** 2610.36539 | [PDF](https://arxiv.org/pdf/2610.36539v1)

**作者:** Cunxi Dai `[一作]` (Carnegie Mellon University), Akash Sharma `[通讯]` (Amazon Frontier AI & Robotics)

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Transformer` `Convolutional Neural Network` `Vision-Language-Action Model` `Contrastive Learning` `Diffusion model` `Optical Flow` `Reinforcement Learning` `Video` `Image` `Multimodality` `Time Series`

**🎯 论文内容**

通过统一的手-接触表示，将人类演示（视频、触觉手套或运动捕捉）转换为保留接触力的机器人轨迹，实现无对象模型的高保真触觉重播；

**💡 创新点**

首次提出以接触为中心的重定向框架（CARD），在保留接触力和方向的同时实现跨多种机器人外观的重定向，并结合视频接触估计模型Vid-HACO和触觉跟踪控制器；

**🔧 技术方法**

利用MANO手模型、语义链接映射、Vid-HACO视频接触预测、跨帧采样优化（CEM）、触觉跟踪控制（积分阻抗+正向力）以及统一的手-接触表示；

**📊 数据集**

使用本机构视频、触觉手套+VIVE跟踪、运动捕捉（共104段）和公开GRAB数据集（100段）进行评估；

**📈 对比分析**

与基线（DexPilot、AnyTeleop、ReGrind等）相比，CARD在视频演示上达到61%接触率、视频+手套78%、GRAB 43%，显著高于基线（<10%或22.5%），硬件实验中触觉跟踪将开环成功率从约4/30提升至25/30；

**⚠️ 局限性**

受限于输入演示质量，开环重播无法补偿大幅物体位移，且对物体模型的无依赖可能在极端抓取情景下产生误差，未来需引入闭环感知策略提升鲁棒性。

---

## 160. Sample-and-Score: Surrogate-Guided Coordinated Spatial Reuse in IEEE 802.11 Networks

**arXiv ID:** 2610.36868 | [PDF](https://arxiv.org/pdf/2610.36868v1)

**作者:** Maksymilian Wojnar `[一作]` (AGH University of Krakow), Maksymilian Wojnar `[通讯]` (AGH University of Krakow)

**关键词:** `Networking and Internet Architecture` `Optimization` `Graph Neural Network` `Transformer` `Score-based Model` `Graph` `Tabular`

**🎯 论文内容**

提出了一种基于采样和评分的Co-SR优化框架：在网络状态的图结构编码下，随机生成Co‑SR配置，使用离线训练好的代理模型评分，挑选得分最高的配置供网络控制器执行。

**💡 创新点**

创新点在于：①将Co‑SR配置映射为图结构并利用图神经网络+Transformer实现高效评分；②采用混合密度网络捕捉速率分布；③不依赖在线交互或频繁测量，能够在一次前向传播中评估数千个候选方案；④通过“采样‑评分”实现了与传统基于阈值或贪婪策略的对比优势。

**🔧 技术方法**

使用的技术包括：随机采样生成Co‑SR配置、图神经网络（3层GNN + 3层Transformer）、混合密度网络（4个高斯分量）、AdamW优化、余弦一次周期学习率调度、稀疏RSSI特征编码以及批量分割加速推理。

**📊 数据集**

数据集来自自研的mapc‑sim仿真器，覆盖11类参数化场景（2–10个AP、最多50个STA），每个场景随机采样1000次位置，每次生成50个Co‑SR配置（共50,000条训练样本），包括随机和理想MCS组合。

**📈 对比分析**

与完美代理、随机、DCF、SR组、H‑MAB、F‑optimal和T‑optimal等基线比较。实验显示：在开放空间和住宅场景中，采样‑评分可在8192候选下达到T‑optimal的95%吞吐率；在2–16 AP的未见拓扑中，恢复率为52–77%；比DCF高出约16%；与H‑MAB相比，推理速度快30×，且不需要多步在线收敛。

**⚠️ 局限性**

限制包括：仅在静态、全缓冲下行负载场景下评估；未使用完整堆栈仿真器或硬件验证；代理在超大规模（>16 AP）和动态网络下的泛化性待验证；并且性能在候选数超过2048后趋于饱和，说明高质量配置的稀缺性成为瓶颈。

---

## 161. IOSage: Detecting HPC I/O Bottleneck Patterns to Guide LLM Recommendations

**arXiv ID:** 2610.36370 | [PDF](https://arxiv.org/pdf/2610.36370v1)

**作者:** Mahdi Banisharif `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Recommendation System` `Anomaly Detection` `Optimization` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Supervised Fine-Tuning` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Tabular` `Benchmark`

**🎯 论文内容**

本文提出 IOSage 系统，先用多标签分类器从 Darshan 日志中检测七种 I/O 瓶颈模式，再将检测结果和模式卡片、作业配置交给大型语言模型（Claude Sonnet 4、GPT‑4o、Llama 3.1），让模型生成具体的 I/O 调优建议，并通过对真实应用的重复跑测验证加速效果。

**💡 创新点**

创新点在于：① 将模式识别任务交给训练好的分类器，提供基准化的模式检测，避免单一日志缺乏基准导致 LLM 误诊；② 引入“模式卡片”知识库，让 LLM 在有限信息下生成可执行的调优代码；③ 通过配置标注的 benchmark 训练与生产日志的双重权重，提升检测的泛化能力；④ 通过实验验证模型推荐在实际 HPC 作业中显著降低总壁时。

**🔧 技术方法**

技术包括：多标签 XGBoost 分类器、TreeSHAP 解释、Darshan 计数特征提取、模式卡片（含计数统计和改动模板）、大型语言模型（Claude、GPT‑4o、Llama）Prompt 设计、实验跑测与对比分析。

**📊 数据集**

数据集主要来自 ALCF Polaris 生产的 131,151 条 Darshan 日志（经清洗后 91,807 条用于训练），以及 689 条配置标注的 benchmark 运行（IOR、DLIO、h5bench、mdtest、HACC‑IO、mpi4py），另外使用 TraceBench 的 9 条真实应用 trace 做外部测试。

**📈 对比分析**

检测方面，分类器在 488 条测试日志上取得 Micro‑F1 0.929，显著优于 Drishti（0.364）和 WisIO（0.311）；在 78 条无模式日志中，LMM 仅提议 9 条变更，而 LLM 单独时提议 64 条，误报大幅下降。推荐方面，Claude 在 6 条真实作业中正确命名 5 条加速改动，平均总壁时提升 4.38×（openPMD）和 1.93×（WRF 10‑min 重启），相较于 LLM 单独推荐，精确度和实际加速更优。

**⚠️ 局限性**

局限性包括：① 仅覆盖七种预定义瓶颈模式，无法发现未注册的模式；② 模式检测依赖完整 POSIX/IO‑IO 计数，对无 POSIX 日志（如 LAMMPS、GROMACS）效果受限；③ 训练数据来自两台 Lustre 系统，跨文件系统泛化尚未验证；④ 仅在单一集群（Delta）上评估加速，需在更多环境验证；⑤ 只生成已知改动模板，无法自动发现新优化策略。

---

## 162. TOPS: Reshaping Flow Source and Transport for Multi-Task Robot Manipulation

**arXiv ID:** 2610.36694 | [PDF](https://arxiv.org/pdf/2610.36694v1)

**作者:** Tuo An `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Transformer` `Flow-based Model` `Gaussian Splatting` `Mixture of Experts` `Reinforcement Learning` `Multimodality` `Sequential`

**🎯 论文内容**

研究了一种多任务流匹配政策的源先验和插值调度改进方法（TOPS），通过逆映射专家动作拟合任务特定高斯混合源并采用基于高斯最优传输的慢-快调度来提升少步生成性能。

**💡 创新点**

①利用逆映射专家动作拟合多模态任务特定源；②依据源-目标协方差差异计算“扩展”方向，并提出沿该方向的慢-快插值调度；③在保持原有政策骨干的前提下，显著提升多任务学习、数据效率和实时性。

**🔧 技术方法**

流匹配框架、概率主成分分析混合模型 (MPPCA)、高斯最优传输 (Gaussian OT)、基于语言的源选择器、两步/两阶段数值求解器。

**📊 数据集**

LIBERO-10、LIBERO-Plus、CALVIN（D→D 与 ABC→D）、三套真实世界多任务套件（Drawer Opening、Color Matching、Pick-Vegetable）以及相关仿真与真实演示数据。

**📈 对比分析**

与多种基线（Standard FM、OT-CFM、PriorGrad、MeanFlow、MM-FM、LeaP 等）在 ID/OOD 环境下使用相同政策骨干、单步推理进行对比；TOPS 在 LIBERO-10 的成功率从 53% 提升至 70%，CALVIN 任务链长度提升至 2.27，推理延迟降低到约 1/6；在真实世界任务中成功率提升至 77.5% vs 51.25%；在数据有限场景下亦保持优势。

**⚠️ 局限性**

需要离线构建每个任务的多模态源与调度，无法动态更新；仅支持预定义任务与固定动作段长度，未验证对未见任务或可变时序生成的泛化；对极大规模任务库的可扩展性仍待验证。

---

## 163. Dependency Graphs of Mathematical Arguments: Representation, Structural Measurement, and Proof Review

**arXiv ID:** 2610.36776 | [PDF](https://arxiv.org/pdf/2610.36776v1)

**作者:** Eunji Park `[一作]` (Seoul National University), Eunji Park `[通讯]` (Seoul National University)

**关键词:** `Artificial Intelligence` `AI Code Assistant` `Representation Learning` `Explainability and Interpretability` `Graph Neural Network` `Transformer` `Large Language Model` `Prompt Engineering` `Graph Theory` `Text` `Graph` `Review/Survey Paper`

**🎯 论文内容**

提出并实现了一个基于源文献的论文图框架，利用大语言模型提取数学论证中的定义、假设、构造、计算和结论，并通过测量合同对其深度、bulk 和宏骨架进行量化。

**💡 创新点**

创新性地统一了源关联的依赖表示、LLM辅助的细粒度提取以及可重复测量协议，明确区分源出现、上下文状态与推理记录，并对共享子论证采用集合并计数。

**🔧 技术方法**

采用大语言模型进行自然语言提取、图论算法计算有向无环图的深度与 bulk、转移消除、统一递归以及版本化提示协议 PGPP 0.1。

**📊 数据集**

以选定的数学论文集合（如 Coq、Mizar、Lean、自然语言论文）为原始数据，构建对应的论文图。

**📈 对比分析**

通过在同一测量合同下多次运行不同 LLM，比较得到的依赖图一致率、深度与 bulk 等指标；实验结果显示测量可重复性良好，但具体数值未给出。

**⚠️ 局限性**

受限于 LLM 提取准确性与人工审核的成本，且测量仅评估已记录的论证结构，无法反映论证难度或原创性，缺乏对大规模数据集的验证。

---

## 164. TAC2: Timestamped Audio Captioning as composable functions

**arXiv ID:** 2610.36568 | [PDF](https://arxiv.org/pdf/2610.36568v1)

**作者:** Swapnil Bhosale `[一作]` (University of Surrey), Zeyu Jin `[通讯]` (Adobe Research)

**关键词:** `Sound` `Generation` `Recognition` `Transformer` `Diffusion model` `Contrastive Learning` `Prompt Engineering` `Knowledge Distillation` `Large Language Model` `Audio` `Text` `Multimodality`

**🎯 论文内容**

本文提出了一种基于多视角弱监督融合和指令-输入-输出（IIO）格式的基线模型，能够从真实世界音频中生成时间精确、可查询的声学时间线，并通过轻量 LoRA 专家扩展语音属性，最终实现精细音频理解与音频生成的双重目标。

**💡 创新点**

创新点在于：①通过时间注册 IoU 匹配将音事件检测、说话人分离与音频描述器的预测无监督融合，构造可规模化的时间线标注；②将时间线拆分为可组合的 IIO 任务，实现单步或多步推理、外部先验条件化；③利用少量标注的 LoRA 专家诱导预训练 backbone 中的隐式语言学能力，进一步扩充到情绪、重音、语调等语音属性。

**🔧 技术方法**

核心技术包括：多视角弱监督融合、时间注册 IoU 匹配、IIO 完成格式、LoRA 专家适配、学生模型蒸馏、冻结文本 LLM 的 Describe‑then‑Reason 评估，以及基于结构化时间线的文本到音频扩散模型训练。

**📊 数据集**

使用的数据集包括：公开的语音情绪和声学属性数据集（IEMOCAP、RAVDESS、MELD、Common Voice）用于 LoRA 专家训练；内部的野生音频语料库（包含多说话人、杂音等）用于多视角弱监督融合；以及 MMSU、MMAR、MMAU、MSU‑Bench 等公开评测基准。

**📈 对比分析**

在音频理解方面，模型与冻结文本 LLM 组合在 MMSU、MMAR、MMAU、MSU‑Bench 上均实现了超过 80% 的总分，尤其在感知子任务上相较传统 LALM 提升 10‑15%；在音频生成方面，利用结构化时间线作为监督的文本到音频生成器，在事件定位、说话人属性和单词强调控制上均取得 1.5–2.1 s 的时间定位误差，且生成音频在语音识别准确率和情绪一致性上与原始音频高度相近。

**⚠️ 局限性**

局限性主要体现在：①对弱教师的质量高度依赖，若教师预测错误会影响时间线构造；②依赖于内部专有的野生音频数据，公开重现需替换音频源；③目前的属性覆盖面有限，未涵盖所有可能的声学或语言学特征；④生成模型在复杂语境下仍可能出现音频质量或同步性的小幅漂移。

---

## 165. Compatibility and future-proof testing for multi-agent systems

**arXiv ID:** 2610.36289 | [PDF](https://arxiv.org/pdf/2610.36289v1)

**作者:** Yueshan Li `[一作]` (ETH Zurich), Emilio Frazzoli `[通讯]` (ETH Zurich)

**关键词:** `Multiagent Systems` `Reinforcement Learning` `Autonomous Driving` `Optimization` `Federated Learning` `Safety and Privacy` `Agentic AI` `Closure Operation` `Equivalence Class Analysis` `Simulation` `Tabular` `Time Series` `Graph`

**🎯 论文内容**

提出了未来可验证（future-proof）测试框架，用种子代理(seed agents)指导多代理系统的兼容性验证；

**💡 创新点**

创新点在于将兼容性定义为可闭包（closure）概念，证明种子兼容不等于共同兼容，提出可测度未来可验证代理的闭包，并给出种子测试的下界损失；

**🔧 技术方法**

采用理论框架(arena、结果生成器、不可接受集)、闭包运算、等价类分析以及实验模拟；

**📊 数据集**

使用两组仿真数据：①单车道车辆链条的刹车参数空间；②256种阻碍区(grid‑world)的机器人路径规划；

**📈 对比分析**

通过案例演示，所有seed兼容代理不一定互相兼容，只有闭包中的代理才保证全局安全；对比显示在非空seed下可实现零损失的测试，说明理论有效性；

**⚠️ 局限性**

局限性包括：需人工挑选seed、闭包与可分离集难以计算、对大规模代理空间及动态增长的兼容性不一定稳定，且缺乏实际部署实验验证。

---

## 166. LLM knowledge for improved Out-of-Distribution Tabular Prediction - A Cross-Validated Regularisation Approach

**arXiv ID:** 2610.36806 | [PDF](https://arxiv.org/pdf/2610.36806v1)

**作者:** Sølve Langseth `[一作]` (Oslo Metropolitan University), Hugo L. Hammer `[通讯]` (Oslo Metropolitan University)

**关键词:** `Machine Learning` `Domain Adaptation` `Data-Centric Learning` `Transformer` `Large Language Model` `L2 Regularization` `Cross-Validation` `Tabular`

**🎯 论文内容**

本文利用大语言模型（LLM）生成系数先验，作为正则化回归的收缩目标，以提升跨域（OOD）预测性能。

**💡 创新点**

创新点在于把LLM生成的先验当作可调节的收缩目标，并通过源域交叉验证自动选择模型与先验结构，证明其对真实域移位具有显著正面影响。

**🔧 技术方法**

技术手段包括：多次采样LLM系数、构造独立、相关和混合先验、L2 正则化线性/逻辑回归、源域交叉验证与置换/符号翻转对照。

**📊 数据集**

使用了五个公开表格数据集：Boston Housing、County Cancer Mortality、Heart Disease、Pima Indians Diabetes、Adult，并按真实协变量划分 ID/OOD。

**📈 对比分析**

与零均值岭回归基线比较，四个数据集平均提升约2–3% OOD MSE 或 log loss，置信区间显著；但单次部署的预测区间往往跨零，表现波动较大。

**⚠️ 局限性**

局限包括：先验选择依赖源域验证难以辨识最优组合、单一源–目标划分不足、LLM 可能记忆训练数据、对大样本时无效、仅提升正则化效果而非真正域迁移。

---

## 167. Underdamped Anchored Langevin Dynamics

**arXiv ID:** 2610.36470 | [PDF](https://arxiv.org/pdf/2610.36470v1)

**作者:** Qi Feng `[一作]` (Florida State University), Lingjiong Zhu `[通讯]` (Florida State University)

**关键词:** `Machine Learning` `Optimization` `Stochastic Differential Equation` `Tabular`

**🎯 论文内容**

提出了一种新的动量基础的欠阻尼锚定朗之万动力学（UALD）和欠阻尼锚定朗之万蒙特卡洛（UALMC）算法，旨在有效地从非可微和重尾目标分布中进行采样。

**💡 创新点**

这是首个针对非可微目标的动量基础朗之万算法，且不受渐近偏差的影响，提供了非渐近保证。

**🔧 技术方法**

使用了动量基础的欠阻尼朗之万动力学（UALD）和其离散化方案，结合了平滑参考势能的替换和漂移项的乘法缩放。

**📊 数据集**

在数值实验中使用了合成的拉普拉斯分布、重尾的多元学生t分布，以及真实数据集（如乳腺癌和银行票据数据集）进行贝叶斯逻辑回归和神经网络实验。

**📈 对比分析**

与现有的基于过阻尼的锚定朗之万算法（ALMC）相比，UALMC在维度和精度水平上的迭代复杂度表现更优，且在数值实验中显示出更好的性能。

**⚠️ 局限性**

限制在于该方法在处理高维数据时可能面临计算复杂度的挑战，尤其是在需要评估平滑参考势能时。

---

## 168. Simulated petrological labels enable probabilistic mineralogy prediction from whole-rock geochemistry

**arXiv ID:** 2610.36429 | [PDF](https://arxiv.org/pdf/2610.36429v1)

**作者:** Matthew J. Liu `[一作]` (Lawrence Berkeley National Laboratory), Kristin A. Persson `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `Machine Learning` `Data Synthesis` `Representation Learning` `Transformer` `Contrastive Learning` `Supervised Fine-Tuning` `Tabular` `Physics Related`

**🎯 论文内容**

通过大规模热力学模拟生成的矿物组装标签，预训练Transformer编码器并在此基础上训练概率性矿物预测头，再利用少量实测岩石的再校准层，将模拟结果对齐到真实矿物分布，从全岩化学成分推断矿物组成。

**💡 创新点**

①将模拟标签与少量实测数据结合的两阶段学习框架；②概率性头（Bernoulli+Dirichlet）能够一次性预测多种矿物的出现与丰度；③利用仿射再校准实现少样本快速适配，显著弥补模拟偏差。

**🔧 技术方法**

Transformer编码器（CrabNet）+掩码元素建模预训练；Bernoulli-Dirichlet概率头；后置再校准层（仿射映射）；联合负对数似然评估和AUROC、MAE等指标。

**📊 数据集**

约788k 全球全岩化学样本（预训练）；约950k 通过Phreeqc等热力学求解得到的模拟矿物组装标签；约10k USGS火山岩真实矿物分布数据用于评估。

**📈 对比分析**

与仅使用原始化学输入的MLP相比，预训练编码器提升了0.2–0.5 nat NLL；零样本情况下AUROC 0.72–0.92；再校准后NLL从≈1.5 nat降至≈1.1 nat，MAE从≈7%降至≈3%；仅用100个样本即可获得≈80%置信度提升。

**⚠️ 局限性**

①模拟仅考虑热力学平衡，忽略动力学与分离过程；②模拟条件假设导致系统性偏差；③对多晶型和固溶体的辨识受限于单一化学信息；④评估基准数据局限于两组火山岩；⑤未与外部方法（如XRD）进行进一步验证。

---

## 169. On the Fundamental Limits of Bistatic ISAC Over Vector Channels

**arXiv ID:** 2610.36351 | [PDF](https://arxiv.org/pdf/2610.36351v1)

**作者:** Dongsheng Peng `[一作]`, Jun Chen `[通讯]`

**关键词:** `Information Theory` `Optimization` `Federated Learning` `Computational Efficiency` `Representation Learning` `Physics Related` `Contrastive Learning` `Gaussian Splatting` `Score-based Model` `Diffusion model` `Stochastic Differential Equation` `Information Theory` `Convex Optimization` `Physics Related`

**🎯 论文内容**

研究了双极ISAC系统在向量信道上的基本极限，给出了在平行信道和一般向量信道下的容量-失真函数 C(D) 的单字母表述，并在高斯状态下实现了完全的容量-失真-功率函数 C(D,P) 的闭式表达。

**💡 创新点**

① 允许平行信道中各子信道的退化方向不一致，首次给出了这类系统的精确单字母容量-失真表达；② 在一般向量信道下，利用 Fisher 信息矩阵、通道增强技术和矩阵极值不等式，证明高斯状态下的 Gaussian superposition 编码是最优的，完成了无退化顺序的向量高斯 ISAC 的完全解。

**🔧 技术方法**

信息论极限分析、退化信道与非退化信道的单字母表述、Gaussian superposition 编码、通道增强、Fisher 信息矩阵及其矩阵不等式、凸优化与 Lagrange 对偶、极值不等式、De Bruijn 恒等式。

**📊 数据集**

无实验数据集，本文全部为理论推导与数学证明。

**📈 对比分析**

通过在二维高斯 ISAC 的数值实验，比较了联合编码与独立编码的容量-失真曲线，表明联合编码在低失真时具有显著优势；同时通过极值不等式证明了 Gaussian superposition 编码在高斯信道下的最优性，确保所给的 C(D,P) 取得全局最优。

**⚠️ 局限性**

仅考虑了记忆无状态的 i.i.d. 信道模型；对非高斯信道或存在状态相关性的情况尚未给出完全解；在实际系统中实现时对信道估计误差、时变信道等因素的影响仍需进一步研究。

---

## 170. OmniGlimpse: Asymmetric Identity Memory for Online Multi-View Tracking

**arXiv ID:** 2610.36785 | [PDF](https://arxiv.org/pdf/2610.36785v1)

**作者:** Rangya Zhang `[一作]` (Nanyang Technological University), Mir Feroskhan `[通讯]` (Nanyang Technological University)

**关键词:** `Computer Vision and Pattern Recognition` `Object Tracking` `Transformer` `Convolutional Neural Network` `Contrastive Learning` `Optical Flow` `Auto Encoder` `Video` `Image`

**🎯 论文内容**

研究在线多视角身份维护，提出 OmniGlimpse 在仅一次性初始化后无进一步身份标签即可实现跨视角跟踪。

**💡 创新点**

采用异步 propose–verify 框架，区分短期自适应关联与长期稳定验证，并在不同空间中维护目标特定自适应与全局共享分布，从而提升对遮挡、外观漂移和错误标签的鲁棒性。

**🔧 技术方法**

使用冻结的 ReID 编码器与目标视图特定适配器、基于高斯原型的长期验证、Ridge 分类器、三元组对比学习与置信度控制器共同构成系统。

**📊 数据集**

在 M3Track（双摄像头 RGB+IR）与 MvMHAT（多摄像头）两个基准数据集上进行评估。

**📈 对比分析**

与多种在线基线（OCLReID、ByteTrack 等）和离线训练参考（CrossMOT、AGW）对比，IDF1 与 CVIDF1 分别提升 9.1 与 22.1 分，且在离线参考中保持竞争力。

**⚠️ 局限性**

仅针对预先标注的少量目标，初始化强度高，尚未扩展至稠密目标或弱初始化场景。

---

## 171. Upper and Lower Bounds on the Complexity of Unit-Distance Graph Realization in $\mathbb{Z}^d$

**arXiv ID:** 2610.36516 | [PDF](https://arxiv.org/pdf/2610.36516v1)

**作者:** Eric Binnendyk `[一作]`, Eric Binnendyk `[通讯]`

**关键词:** `Computational Complexity` `Optimization` `Contrastive Learning` `Reinforcement Learning` `Physics Related`

**🎯 论文内容**

本文提出了一种基于轻平衡切割的分治算法，能够在二维格点上以 2^O(n^(1/2)(log n)^2) 时间判断图是否可在 ℤ^2 中实现单位距离嵌入，并给出了三维格点上 2^O(n^(2/3)(log n)^2) 的上界；

**💡 创新点**

创新点在于证明了任何 ℤ^d 中的格点嵌入图都存在少于 2n^((d-1)/d) 条边的轻平衡切割，并利用该切割实现了指数级子线性加速的分治搜索；

**🔧 技术方法**

主要技术包括组合几何的轻平衡切割理论、分治递归、稀疏化引理与 ETH 结合的下界证明，以及对逻辑引擎的三维扩展构造；

**📊 数据集**

论文为理论分析，未使用实际数据集，所有结果均基于构造性证明与复杂度分析；

**📈 对比分析**

与朴素的 2^(2d|V|) 搜索相比，该算法在指数上提升了约 Ω̃(n^1/d) 的因子；下界证明显示在 2D 与 3D 的上界与下界在对数因子内相匹配，证明了算法的接近最优性；

**⚠️ 局限性**

主要限制是对 ETH 的依赖、仅针对固定维度（d 为常数）以及对稀疏化引理的应用，未讨论更高维或连续欧几里得空间中的实现与复杂度；

---

## 172. Bayesian Evidence Accumulation for Graph-Based Learning in Dynamic Data-Driven Applications Systems

**arXiv ID:** 2610.36190 | [PDF](https://arxiv.org/pdf/2610.36190v1)

**作者:** Moises Ponce `[一作]` (University of Tennessee), Vasileios Maroulas `[通讯]` (University of Tennessee)

**关键词:** `Machine Learning` `Classification` `Data-Centric Learning` `Optimization` `Federated Learning` `Computational Efficiency` `Graph Neural Network` `Bayesian Inference` `Confidence Thresholding` `Evidence Accumulation` `Graph` `Tabular` `Time Series`

**🎯 论文内容**

提出 B-EAGLe 框架，将图分类任务转化为在 DDDAS 环境下的贝叶斯证据累积决策过程

**💡 创新点**

用贝叶斯滤波替代传统加性累积，支持连续观测和概率置信度阈值，并与 GNN 编码器无缝集成

**🔧 技术方法**

GAT 基图神经网络编码器 + 贝叶斯证据累积模块 + 置信阈值决策机制（Cromwell 规则乘法更新）

**📊 数据集**

MNIST-SP、CIFAR10-SP（图分类）和 PATTERN、CLUSTER（节点分类）四个公开数据集

**📈 对比分析**

与 GatedGCN+、GRIT 等基线在相同观测预算下比较，B-EAGLe 在 3/4 数据集上准确率提升最高 13%，平均观测次数低；但节点任务由于阈值过高导致超时率接近 100%

**⚠️ 局限性**

高阈值导致节点任务超时率高、观测选择仅随机未利用 DDDAS 反馈、对观测相关性假设不完整、缺乏实时实验验证

---

## 173. E0.5: From Reconstructive to Future-Predictive 3D Foundations for Robotic World-Action Modeling

**arXiv ID:** 2610.36011 | [PDF](https://arxiv.org/pdf/2610.36011v1)

**作者:** X-Era AI Lab Team `[一作]` (X-Era.AI), Guangrun Wang `[通讯]` (X-Era.AI)

**关键词:** `Robotics` `Robotic Intelligence` `Transformer` `Convolutional Neural Network` `Vision-Language-Action Model` `Diffusion model` `Flow-based Model` `Mixture of Experts` `World Model` `Point Cloud` `Mesh` `Video` `Sequential`

**🎯 论文内容**

研发了E0.5，一种基于3D基础模型的世界-动作模型，联合学习机器人动作与未来三维几何状态。

**💡 创新点**

创新点：①将3D基础模型直接作为可训练的核心骨干，实现几何先验的自适应；②对未来3D状态进行显式监督，提升空间推理与泛化能力；③在大规模机器人交互预训练中保持几何监督，使模型在真实机器人上保持高性能。

**🔧 技术方法**

使用技术：VGGT‑Omega 3D 基础模型，混合Transformer（MoT）将动作专家与几何表征融合，流匹配动作生成，DepthDiT等几何监督头。

**📊 数据集**

使用数据集：RoboTwin 2.0（50任务、仿真）、AgiBotWorld‑Beta（160k轨迹、177任务）、真实机器人平台 AgileX Piper 与 Zerith H1 的多任务数据。

**📈 对比分析**

与多种 VLA/WAM 基线对比：在 RoboTwin Clean 模式下 89.9% 成功率，Random 模式下 50.1%；在真实机器人上 ID/OOD 下分别达到 100%/80% 等，尤其在 OOD 情况下性能提升约 30 倍，显著优于 π_0.5 等基线。

**⚠️ 局限性**

局限性：对物体边界细节仍有误差，随机化环境下性能仍显著下降；依赖大规模预训练数据和计算资源；对动作序列长度与跨硬件迁移的鲁棒性待进一步验证。

---

## 174. SAGE: Semantic-Augmented Geometric Experts for Point Cloud Registration

**arXiv ID:** 2610.36658 | [PDF](https://arxiv.org/pdf/2610.36658v1)

**作者:** Tomoyasu Shimada `[一作]` (NTT, Inc.), Ryuichi Tanida `[通讯]` (NTT, Inc.)

**关键词:** `Computer Vision and Pattern Recognition` `Pose Estimation` `Representation Learning` `Transformer` `Graph Neural Network` `Mixture of Experts` `Contrastive Learning` `Cross Attention` `Point Cloud`

**🎯 论文内容**

将冻结的预训练点云语义表示与几何特征融合，通过语义引导的专家混合网络（SG‑MoE）完成点云配准。

**💡 创新点**

创新点在于利用冻结语义特征进行专家路由，使各专家专注于不同语义‑几何模式，解决仅用几何导致的歧义；并通过两阶段特征提取实现语义与几何的深度融合。

**🔧 技术方法**

采用Sonata预训练语义编码器（冻结）、KPConv下采样卷积、GeoTransformer框架、交叉注意力以及语义引导的混合专家网络；配合RANSAC和结构感知估计器（LGR）完成配准。

**📊 数据集**

在室内点云数据集3DMatch与低重叠版本3DLoMatch上进行实验，附加在KITTI上进行验证。

**📈 对比分析**

与单通道与迭代改进的基线（Predator、CoFiNet、GeoTransformer、RoITr、PEAL、PSReg）在RANSAC及LGR协议下对比，SAGE在3DLoMatch上实现最高配准召回率，3DMatch上保持竞争力，并在特征匹配召回率和内点比例上均优于对手。

**⚠️ 局限性**

局限在于RANSAC的均匀采样对语义信息利用不足，导致高采样量下性能不显著提升；同时系统依赖于预训练语义模型，若语义分布与训练集差异大可能影响泛化。

---

## 175. Polarity--Time Interaction and Multi-Level Reliability for Event-Based Spacecraft Pose Estimation

**arXiv ID:** 2610.36625 | [PDF](https://arxiv.org/pdf/2610.36625v1)

**作者:** Yonglong Zhang `[一作]`, Yang Liu `[通讯]`

**关键词:** `Artificial Intelligence` `Pose Estimation` `Optimization` `Transformer` `Spiking Neural Network` `Optical Flow` `Contrastive Learning` `Simultaneous Localization and Mapping` `Score-based Model` `Stochastic Differential Equation` `Video` `Time Series`

**🎯 论文内容**

利用事件相机对太空船进行姿态估计，提出了可靠性感知的多阶段框架，包括极性-时间交互的三分支特征、几何约束的事件‑CAD一致性细化以及质量加权的轨迹优化；

**💡 创新点**

创新点在于：①在极性-时间特征上引入跨分支注意力与门控以捕获时空信息；②用几何安全的事件‑CAD一致性评分与候选接受策略提升单帧精度；③在轨迹层面构建十维质量描述符并通过回归校准权重，实现基于观察质量的自适应加权；

**🔧 技术方法**

主要技术包括事件特征编码与聚合、跨分支注意力机制、Swin‑B特征提取、基于几何的权重学习、E‑PnP-RANSAC姿态估计、方向不变的梯度时空一致性判定、基于Huber损失的稀疏非线性最小二乘轨迹优化；

**📊 数据集**

使用公开的SPADES事件数据集（包含Proba‑2航天器的300条合成轨迹和15,500帧实验室轨迹），并在其官方评测标准下进行实验；

**📈 对比分析**

与现有基线（如PASAN+IRIS、RF‑DETR级联）对比，系统在SPADES挑战测试集上取得综合姿态得分0.0474，较PASAN+IRIS下降约5.4%，在验证集上从0.0772降至0.0514；

**⚠️ 局限性**

局限性主要体现在：①长时间序列中仍存在观测偏差；②对极限光照或极端运动场景的鲁棒性待进一步提升；③计算量相对较大（5窗口约52.7 ms/帧），在更高帧率或资源受限平台上的部署仍需优化。

---

## 176. The Complexity of Strategizing Against No-Regret Learners: Pure When It Is Easy, Hard Even When Mixed

**arXiv ID:** 2610.36261 | [PDF](https://arxiv.org/pdf/2610.36261v1)

**作者:** Yiheng Su `[一作]` (University of Wisconsin--Madison), Emmanouil-Vasileios Vlatakis-Gkaragkounis `[通讯]` (University of Wisconsin--Madison)

**关键词:** `Computer Science and Game Theory` `Optimization` `Reinforcement Learning from Human Feedback` `Game Theory` `Reinforcement Learning` `Contrastive Learning` `Bregman Divergence Analysis` `Potential Function Decomposition`

**🎯 论文内容**

研究了在有限时间重复博弈中，对已知在线学习者实施先知式策略优化的计算复杂性，证明在一般博弈中混合策略仍无法突破NP‑hardness，而在零和博弈中纯策略即可实现最佳盈余。

**💡 创新点**

首次将均值基础学习者与独立集差距归约相结合，证明在一般博弈中无多项式时间混合策略优化器能保证靠近最优纯序列；并提供纯策略在零和博弈中等价的高效构造。

**🔧 技术方法**

采用基于独立集差距的归约、均值基础学习者条件、FTRL势函数分解、Bregman散度分析、设置间提取与前缀平衡等理论技术。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

通过理论证明展示在一般博弈中无法在多项式时间内逼近最优纯序列，而在零和博弈中纯策略可在多项式时间内获得与混合策略相同的盈余，体现了纯/混合策略的计算易难对比。

**⚠️ 局限性**

仅适用于最坏情况，依赖学习者满足均值基础条件、学习率约束以及特定时间范围；对随机或平滑实例的效果尚未说明；零和博弈结果仅在FTRL及其特定正则化下成立。

---

## 177. The Trinity:Capability-Driven Recursive Self-Improvement for Multimodal Agents

**arXiv ID:** 2610.36077 | [PDF](https://arxiv.org/pdf/2610.36077v1)

**作者:** Weidong Tang `[一作]`, Yang You `[通讯]`

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Robotic Intelligence` `Meta Learning` `Transformer` `Agentic AI` `World Model` `Vision-Language-Action Model` `Diffusion model` `Retrieval-Augmented Generation` `Chain-of-Thought` `Multimodality` `Text` `Image` `Video` `Tabular` `Benchmark`

**🎯 论文内容**

构建了名为The Trinity的递归自我提升框架，将环境生成与代理harness演化耦合，通过能力树将多模态任务归纳为共享能力需求，并在真实/模拟/真实循环中迭代提升agent的能力。

**💡 创新点**

①从任务驱动转向能力驱动的递归自我提升；②引入能力树和可验证的简化训练世界；③形成可复用的harness树与能力感知路由；④在多任务、多模态和物理真实环境上实现跨任务迁移。

**🔧 技术方法**

能力驱动的Real2Sim2Real、递归自我提升（RSI）、可验证的模拟世界生成、能力树与harness树构建、基于能力感知的路由、冻结的Claude Opus 5 backbone以及六协议的harness框架。

**📊 数据集**

28个训练基准与18个未见基准（涵盖规则发现、约束求解、感知控制、多智能体交互、长时序环境等），以及两项真实机器人任务。

**📈 对比分析**

与基线harness对比，平均在训练基准上提升了34点，在未见基准上提升了26点；在两项真实机器人任务中也表现出显著的迁移效果，验证了能力树和路由的有效性。

**⚠️ 局限性**

依赖网络搜索获取基准，覆盖度与质量可能受限；目前仅在harness层面进行递归自我提升，未扩展至模型参数训练；能力树在实验后固定，未能动态捕获新出现的能力需求。

---

## 178. Ground Sample Distance Constrained k-Coverage Viewpoint Planning via Set Multi-Cover Formulation

**arXiv ID:** 2610.36450 | [PDF](https://arxiv.org/pdf/2610.36450v1)

**作者:** Waleed Bin Khalid `[一作]` (Indiana University), Chul Min Yeum `[通讯]` (University of Waterloo)

**关键词:** `Robotics` `Optimization` `Set Multi-Cover Formulation` `Greedy Algorithm` `ILP (Integer Linear Programming)` `Mesh` `Benchmark`

**🎯 论文内容**

本文提出一种基于GSD约束的k‑覆盖视点选择框架，用于无人机三维重建与检查；

**💡 创新点**

创新点在于：①把GSD、视角、视场、遮挡等几何感知约束融入视点生成；②利用体素化表面对候选视点进行可见性评估并针对低覆盖体素做增补；③设计一种动态加权贪心算法，实时权衡剩余覆盖缺口与视点稀缺度，实现接近最优的视点集合；

**🔧 技术方法**

技术包括：体素化表面建模、基于法线的候选视点生成、射线检测遮挡过滤、GSD几何约束、增补采样、集合多重覆盖问题建模、贪心动态权重选择、ILP求解以及重叠剔除基准；

**📊 数据集**

使用两套公开3D模型（Angel Statue与Rooster），在多种体素分辨率和k值下进行实验；

**📈 对比分析**

与基于重叠剔除的贪心方法以及ILP最小视点数解进行比较。贪心方法在所有设置下均能实现99.93%–100%的k‑覆盖，仅比ILP多3.00%–16.39%的视点；重叠剔除方法往往无法完全覆盖。

**⚠️ 局限性**

局限性包括：假设结构已知且静止；未考虑无人机运动、动态避障、最小视点间距和重建质量；仅在两种模型上验证；无理论近似保证，且计算时间随候选数急剧增加。

---

## 179. Developing a Dynamic Multi-factor Authentication System for Securing Smart Organizations

**arXiv ID:** 2610.36926 | [PDF](https://arxiv.org/pdf/2610.36926v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Cryptography and Security`

---

## 180. Do Coding Agents Still Need a Debugger? Frontier-Model-Based Fault Localization at the Accuracy Ceiling

**arXiv ID:** 2610.36015 | [PDF](https://arxiv.org/pdf/2610.36015v1)

**作者:** Christoph Daxerer `[一作]` (Software Competence Center Hagenberg GmbH), Josef Pichler `[通讯]` (University Of Applied Sciences Upper Austria)

**通讯引用:** 332 | [OpenAlex ID](https://openalex.org/A5103053531)

**关键词:** `Software Engineering` `AI Code Assistant` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Agentic AI` `Text` `Code`

**🎯 论文内容**

对 Sol 6.1、Luna 6 与 Qwen 3.8 三大 LLM 编码代理在 LeetCode 算法题中单行故障定位的准确率进行系统评估，并在加入行为保持变换后进一步测试其稳健性；同时比较了无工具、可选调试器和强制调试器三种工具使用模式下的定位效果、成本与收益。

**💡 创新点**

① 通过将原始程序与经过五种结构化、语义保持的重写（如循环重构、双重否定、变量重命名等）相匹配，构造了“变换后程序”对，检验 LLM 在语法重写导致的误判；② 引入可选与强制两类工具使用策略，量化调试器对定位准确率的增益及其成本比例；③ 在同一实验框架下对三大模型进行横向对比，揭示了模型规模、训练时长对定位性能的影响。

**🔧 技术方法**

使用基于工具调用的 LLM 交互式调试框架（包含脚本执行、pdb 风格断点、状态检查等工具），结合 OpenAI API（Sol 6.1、Luna 6）与自托管 Qwen 3.8（vLLM）执行推理；对每个程序生成多条推理记录，记录 token 计数、执行时长与费用；通过 Top‑1 正确率、工具使用率、误判原因分析等指标进行评估。

**📊 数据集**

LeetCodeDataset：150 条精简的 Python 算法题（Easy/Medium/Hard）及其官方解答和完整单元测试；在每个参考程序上注入 5 种单行故障（共 610 个故障），随后使用 5 种行为保持变换得到 3,022 个变换后程序；共 65,376 次代理推理实验。

**📈 对比分析**

在无工具模式下，三大模型均达到了 98–100% 的定位准确率；变量重命名、死代码插入等轻度变换几乎无影响；但循环重构和双重否定会导致 Luna 6 和 Qwen 3.8 的准确率下降 2–4%；强制调试器可将变换后错误率降低 60–80%，并将两模型的误差差距压缩至 0.1–0.3%；可选调试器仅少量使用交互断点，仍能获得大部分准确率提升，但 token 消耗比强制模式低约 2–3 倍，费用亦显著降低。

**⚠️ 局限性**

① 仅针对短、单文件、具备完整规范的 Python 算法题，缺乏对大型多文件项目或不完整规范的验证；② 仅测试单行故障，未覆盖多行或跨文件缺陷；③ 变换后的程序仍与原始 LeetCode 题目在语义上等价，模型仍可能借助已知解答进行推断，难以彻底消除训练数据熟悉度的影响；④ 实验环境与模型 API 版本变更可能导致复现难度；⑤ 调试器工具的可用性与准确性受限于工具接口与模型对指令的理解。

---

## 181. p-DCIM: A 28nm Probabilistic SRAM DCIM based Ising Machine for Near-Memory Context Selection in LLM-RAG Pipeline

**arXiv ID:** 2610.37033 | [PDF](https://arxiv.org/pdf/2610.37033v1)

**作者:** Madhav Vadlamani `[一作]` (Georgia Institute of Technology), Shimeng Yu `[通讯]` (Georgia Institute of Technology)

**关键词:** `Hardware Architecture` `Optimization` `Retrieval` `Computational Efficiency` `Retrieval-Augmented Generation` `QUBO/Ising 组合优化` `Text`

**🎯 论文内容**

本研究设计并实现了28nm SRAM数字计算存储内(Ising)加速器p-DCIM，用于检索增强生成(RAG)流程中的集成上下文选择。

**💡 创新点**

创新点在于：①双模共享SRAM数组与符号加法树，既用于QUBO系数构造（确定模式），又用于概率Annealing求解（概率模式）；②利用SRAM伪读误差产生随机性，并通过数字化的幅度平面开启/关闭实现可编程冷却；③将加速器集成到近存储(HBF)检索系统，实现候选向量与QUBO构造/求解留在内存子系统，最大限度减少跨包数据搬运。

**🔧 技术方法**

使用技术包括：12T SRAM计算单元（6T存储+4T NOR计算+2T多路器）、DCIM乘加（MAC）数据通路、QUBO/Ising 组合优化、伪读扰动随机性、数字幅度平面冷却调度、以及HotpotQA的问答实验。

**📊 数据集**

实验数据集为HotpotQA问答集，包含检索候选片段和需求子问题，用于构造QUBO并评估生成答案质量。

**📈 对比分析**

与LLM基线SetR/SetRO对比，p-DCIM在能耗上实现了超过六个数量级的降低（仅0.9 μJ vs 358 J），总查询能耗下降67%，同时在Exact Match、Gold‑Contains、token‑level F1等指标上保持与基线相当。

**⚠️ 局限性**

限制在于：验证范围仅限于可在0.16 mm²宏内拟合的QUBO规模，需进一步扩展至更大规模系统；伪读随机性对供电电压敏感，需精细化控制；单芯片功率/面积仍受制约，进一步集成与规模化仍是挑战。

---

## 182. Folding Attention Biases into Keys Makes Compacted KV Caches Run on Standard Kernels

**arXiv ID:** 2610.36520 | [PDF](https://arxiv.org/pdf/2610.36520v1)

**作者:** Utkarsh Kanwat `[一作]`, Utkarsh Kanwat `[通讯]`

**关键词:** `Machine Learning` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Large Language Model` `Compression` `Text`

**🎯 论文内容**

将Attention Matching（AM）中的偏置项折叠进保留的键向量，得到可用标准Attention核的纯KV缓存；

**💡 创新点**

提出每个头仅需一个闭式方向即可将静态偏置迁移到键上，保持缓存大小、维度与查询路径不变；

**🔧 技术方法**

利用AM压缩、线性最小二乘估计方向、协方差闭式解和值再拟合等技术；

**📊 数据集**

在QuALITY、LongHealth、LongBench等问答数据集以及Qwen3系列和Llama‑3.1‑8B模型上进行评估；

**📈 对比分析**

与原AM、drop‑bias以及其他压缩方法对比，准确率在13/14等价预算设置下与AM无显著差距，KL更低，解码速度和容量提升至原始的≈98%；

**⚠️ 局限性**

局限于等价预算、对查询漂移敏感，在某些模型/预算下误差增大，未覆盖所有动态压缩或高偏置场景。

---

## 183. Privately Clipping Heavy-Tailed Data

**arXiv ID:** 2610.36281 | [PDF](https://arxiv.org/pdf/2610.36281v1)

**作者:** Matthew Joseph `[一作]` (Google Research), Alex Kulesza `[通讯]` (Google Research)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Tabular`

**🎯 论文内容**

本文提出一种全同态差分隐私方法，用于在对向量求和时自动选择最佳的截断阈值

**💡 创新点**

创新点在于引入残差协方差（residual coherence）作为低灵敏度的代理量，利用它在私有阈值化算法中寻找平衡点，进而实现对截断阈值的隐私化自适应选择

**🔧 技术方法**

核心技术包括：基于残差协方差的阈值化（利用 机制实现私有查询），对向量规范化和截断的闭式偏差-方差分析，以及稀疏矩阵预处理加速的实现

**📊 数据集**

实验使用了十个来自互联网的真实数据集，包括 Steam、Reddit、LastFM、HN、MovieLens、Pantry、Twitch、Foursquare、Jester、Slashdot，维度从几千到数十万不等

**📈 对比分析**

与传统基于固定分位数（0.9/0.95/0.99）以及另一种量化方法（k‑th 高分位数）相比，该方法在 6/10 数据集上获得最低误差，比最佳基线至少低 30%，且总体误差提升不超过 32%

**⚠️ 局限性**

局限性包括：理论证明基于假设条件（Assumption 1）不必然满足，且不提供必要条件；在某些极端分布下残差协方差可能失效；未针对 DP‑SGD 等实际梯度裁剪场景做深入评估

---

## 184. Trace-WAM: Learning to Act via 3D Motion Prediction for World Action Models

**arXiv ID:** 2610.36840 | [PDF](https://arxiv.org/pdf/2610.36840v1)

**作者:** Xu Hu `[一作]` (Hong Kong Polytechnic University), Zhaoxiang Zhang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `World Model` `Auto Encoder` `Mixture of Experts` `Flow-based Model` `Vision-Language-Action Model` `Video` `Multimodality`

**🎯 论文内容**

提出 Trace-WAM，联合学习未来视频预测、3D 运动预测和动作生成的模型

**💡 创新点**

创新点在于将未来 3D 运动预测作为辅助目标，使用三支 Transformer 的 Tri‑MoT 架构和 B‑spline 曲线实现紧凑运动表示，显著提升对交互动态的关注

**🔧 技术方法**

技术包括冻结的 Causal Video VAE、Transformer 三支专家（视频、动作、运动）、B‑spline 运动曲线、流匹配训练、结构化注意力掩码

**📊 数据集**

使用 RoboCasa‑GR1、LIBERO、LIBERO‑Plus 三个机器人操作基准进行评估

**📈 对比分析**

与 Fast‑WAM、Joint‑WAM、DIAL 等方法对比，Trace‑WAM 在少样本和分布外（如背景、机器人初始状态、语言、光照等）条件下显著提升成功率，例如少样本下提升约 18–19pp，LIBERO‑Plus 上 84.5% 对比 66%（Fast‑WAM）

**⚠️ 局限性**

局限在于仍需大量多模态演示数据，对预训练视频 VAE 的依赖较强，且在极端环境或复杂动态场景中运动预测误差仍可能影响最终决策

---

## 185. Abstract Lipschitz Continuity

**arXiv ID:** 2610.37006 | [PDF](https://arxiv.org/pdf/2610.37006v1)

**作者:** Marco Campion `[一作]` (Sorbonne Université), Caterina Urban `[通讯]` (Inria)

**关键词:** `Programming Languages` `Explainability and Interpretability` `Computational Efficiency` `Optimization` `AI Code Assistant` `Review/Survey Paper`

**🎯 论文内容**

本文提出了“抽象 Lipschitz 连续性（Abstract Lipschitz Continuity, ALC）”这一新的函数性质，用以衡量程序语义在抽象属性空间中的稳健性和可预测性，并给出了对应的证明系统；

**💡 创新点**

创新点在于：①将传统的 Lipschitz 连续性扩展到抽象域，显式区分了属性基础（qualitative）逼近和定量逼近；②证明了 ALC 是部分完备性、部分抽象非干预（abstract non‑interference）和抽象鲁棒性（abstract robustness）的充分条件；③提出了基于程序语法的推理系统，能够通过基本命令的 ALC 证明和侧条件推导出整体程序的 ALC；

**🔧 技术方法**

核心技术包括：抽象解释的上闭包算子（uco）与预度量（pre‑metric）的组合，抽象 Lipschitz 连续性的定义与证明，以及对程序语义的结构化推理规则（如基本命令、顺序组合、非确定性选择、Kleene 迭代）的系统化；

**📊 数据集**

论文以理论示例为主，未使用具体数据集；示例包括文件直方图、整数集合、符号域等，用以展示 ALC 在不同抽象空间中的应用；

**📈 对比分析**

通过与已知性质（部分完备性、抽象非干预、抽象鲁棒性）的理论比较，证明 ALC 作为充分条件的适用性；在程序证明系统方面，讨论了常数与抽象的可变性以及证明可行性，给出了若干实例证明，展示系统的有效性；

**⚠️ 局限性**

局限性：①ALC 并非闭合于函数复合，需满足额外侧条件；②证明系统不完整，可能无法得到最优 Lipschitz 常数；③对预度量和抽象的选择敏感，若选择不当可能导致不可证或误判；④论文主要为理论性工作，缺乏大规模实证评估。

---

## 186. BehaveRec: When Does Long-Term Behavioral History Help Sequential Recommendation? A Study with Session, History, and Community Signals

**arXiv ID:** 2610.36239 | [PDF](https://arxiv.org/pdf/2610.36239v1)

**作者:** Aysenur Inan `[一作]` (Walmart Global Tech), Sushant Kumar `[通讯]` (Walmart Global Tech)

**关键词:** `Artificial Intelligence` `Recommendation System` `Transformer` `Graph Neural Network` `Contrastive Learning` `Reinforcement Learning` `Causal Modeling` `GraphSAGE` `ThreeWayGate Fusion` `Sequential` `Graph`

**🎯 论文内容**

设计并评估了 BehaveRec 框架，结合会话、90 天用户历史和基于行为相似度的用户-用户图，探究长期历史与社区信号对顺序推荐的实际帮助。

**💡 创新点**

通过对用户历史长度的阶层化实验揭示长期历史对推荐性能的剂量-响应关系，并给出约 20 次交互的实用部署阈值；使用用户-用户行为相似图而非商品共现来构建社区信号。

**🔧 技术方法**

使用 causal transformer 会话编码器、双向 Transformer 用户历史编码器、GraphSAGE 邻居聚合、ThreeWayGate 融合、InfoNCE 对齐、权重共享以及标签平滑交叉熵等技术。

**📊 数据集**

实验基于 137M 交互序列的工业数据集（12.6M 用户），并在 MovieLens ML‑1M、Amazon CDs & Vinyl、Amazon Electronics 等公开数据集进行跨域验证。

**📈 对比分析**

采用 Recall@5 / NDCG@k 等指标与用户层 bootstrap CI 与 SASRec、BERT4Rec 等基准对照；工业数据中 BehaveRec 比 SASRec 提升约 3.2%（+0.0226 R@5），在用户历史 > 20 次时优势明显；在稀疏历史环境下不具优势。

**⚠️ 局限性**

结果基于单次训练，方差估计有限；图构建对比非独立重训；物品级评估仅单次实验，方向性不确定；部署阈值为经验选择；在短历史或低维词表环境下性能不一定提升。

---

## 187. GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy for Mobile GUI Agents

**arXiv ID:** 2610.36877 | [PDF](https://arxiv.org/pdf/2610.36877v1)

**作者:** Bingchao Wang `[一作]` (Shanghai Jiao Tong University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `Artificial Intelligence` `Computational Efficiency` `Reinforcement Learning from Human Feedback` `Representation Learning` `Transformer` `Vision Language Model` `Contrastive Learning` `Auto Encoder` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出了第一种基于Joint‑Embedding Predictive Architecture (JEPA) 的无解码器移动GUI代理，直接在JEPA编码器的潜在空间中对候选动作进行评分；

**💡 创新点**

创新点在于：①将JEPA作为完整的动作决策策略；②完全消除token级解码，显著降低计算和内存开销；③证明预测器可用随机初始化，训练目标即为决策本身；

**🔧 技术方法**

技术细节包括：ViT‑L/16冻结的JEPA编码器（在未标注的GUI截图上预训练）；残差预测器对屏幕潜在向量做动作/区域条件变换；结构化头（余弦相似度）与MiniLM文本匹配器联合评分；后期的span ranker和click reranker进一步精调文本参数；

**📊 数据集**

使用数据集：GUI‑Odyssey（189K移动屏幕截图）、VideoCUA（桌面截图）进行JEPA预训练；训练集为13,604条episodes（共74,714步），包含点击、滑动等动作；评测采用AndroidControl官方四子集（IDD、app_unseen、task_unseen、category_unseen）。

**📈 对比分析**

在与同协议的Qwen2.5‑VL‑7B LoRA微调基线对比：本方法在同输入、同GPU上实现了2.8–4.0×的推理速度提升、4.6–6.4×的显存降低，并达到了91.0/70.9（Low/High）IDDD的86.4/92.4%准确率；在全测试集上超过OS‑Atlas‑7B、MIRAGE‑4B、Qwen3‑VL‑8B等VLM代理。

**⚠️ 局限性**

局限性：依赖屏幕可访问性树（a11y）作为候选来源，未覆盖无a11y或网页DOM；仅完成一步级评分，缺乏任务规划与终止判断；部分无候选的点击被误判为错误；推理时间测量不含屏幕捕获与a11y检索；在非移动设备或不同候选来源上需要额外改造。

---

## 188. Benchmarking LLM Agents for Behaviorally Adequate Test Generation from Software Requirements

**arXiv ID:** 2610.36584 | [PDF](https://arxiv.org/pdf/2610.36584v1)

**作者:** Binhang Qi `[一作]` (National University of Singapore), Jin Song Dong `[通讯]` (National University of Singapore)

**关键词:** `Software Engineering` `AI Code Assistant` `Benchmark` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text`

**🎯 论文内容**

构建并发布了一个包含 5,086 个真实项目的需求‑测试实例集合，并对 LLM 代理在从自然语言需求生成完整可执行且行为充分的单元/集成/系统级测试用例的能力进行系统评估。

**💡 创新点**

创新点包括：①首次结合真实 BDD（Gherkin）需求与完整执行验证的测试实现映射；②通过与开发者原始测试对比的 mutant‑kill 召回率来衡量行为充分性；③揭示生成测试与实际需求之间的“信息缺口”，并提出需求‑根植的 harness 思路。

**🔧 技术方法**

技术手段：使用多种 LLM 代理（Aider、OpenHands、Codex CLI、Claude Code）与四种大型语言模型（DeepSeek‑V4.1‑Flash、GPT‑6 Luna/Sol、Claude Sonnet 5.5）；配合 BDD 框架（Cucumber、Behave、Behat、Godog）、自动化执行、代码覆盖与变异测试、以及多层次评价指标（Pass@1、KillComplete@1、API‑F1、覆盖 Jaccard）。

**📊 数据集**

数据集来源于 8 个开源项目（CoreShop、PrestaShop、Sylius、python‑docx、Algorand SDK、Kapua、e‑petitions、coddy‑agent），覆盖 5 种语言（Java、Python、PHP、Go、Ruby）和 4 种 BDD 框架，包含 1,415 个 Feature 与 5,086 个需求‑测试实例。

**📈 对比分析**

比较方法：在 target‑hidden 与 all‑hidden 两种视图下，对八种 agent‑model‑reasoning 配置进行实验，记录生成时间、LLM 调用次数、Tokens 与成本；评估 Pass@1 与 KillComplete@1。结果显示：最佳配置 Codex/DeepSeek‑V4.1‑Flash Pass@1 ≈ 82.8%，但 KillComplete@1 仅 ≈ 33.8%；成本提升并未带来显著行为充分性提升，表现出收益递减。

**⚠️ 局限性**

局限性：①需求文本往往缺乏具体的设置与期望结果，导致信息缺口；②缺少项目级上下文（fixtures、helper、业务约定）会使生成测试缺失关键断言；③LLM 生成具有显著非确定性，单次生成的可靠性低；④数据集仅包含 BDD 需求，未覆盖更自由的自然语言需求；⑤可能存在训练数据泄漏导致的偏差。

---

## 189. From Ripple to Boundary: Deterministic Impact Boundaries for LLM-Based Software Evolution

**arXiv ID:** 2610.36549 | [PDF](https://arxiv.org/pdf/2610.36549v1)

**作者:** Jiameng Bai `[一作]` (Zhejiang University), Sai Wu `[通讯]` (Zhejiang University)

**关键词:** `Software Engineering` `AI Code Assistant` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Graph Neural Network` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Diffusion model` `Hashing` `Graph-based Data Structures` `Graph` `Text` `Tabular` `Finance Related`

**🎯 论文内容**

构建基于知识图的设计资产 Merkle DAG 并实现两阶段差分来确定 LLM 代码演化的精确影响边界，随后将该边界注入到 LLM 生成上下文，保证增量代码变更的安全与精确。

**💡 创新点**

将设计资产建模为内容可寻址的 Merkle DAG，实现从设计变更到代码变更的确定性传播；引入两阶段差分算法提取因果链；通过确定性上下文注入消除 LLM 生成过程中的模糊影响范围。

**🔧 技术方法**

知识图 (Knowledge Graph)、内容可寻址 Merkle DAG、哈希递归结构、两阶段差分算法、LLM 代理、Cypher 查询、上下文注入模板、工业支付平台 CAS 系统。

**📊 数据集**

工业支付微服务平台的设计资产（类图、时序图、规则表）与对应生成的代码，实验中对 3 种变更类型（新增、修改、删除）在所有 KG 节点上进行。

**📈 对比分析**

与传统基于语义检索或 RAG 的概率性影响分析方法对比；实验表明差分引擎捕获 100% 的多跳因果链，LLM 生成的多文件修改零意外更改，未给出数值指标，评价主要为定性成功案例。

**⚠️ 局限性**

目前仅进行定性评估；缺乏公开基准与量化指标；方法局限于设计资产，需验证对数据库模式、IaC 等其他结构化工件的适用性；未覆盖大规模多文件交互场景的性能数据。

---

## 190. When Can a Learned Design Surrogate Be Trusted in a Self-Driving Microfluidics Laboratory? An Offline Closed-Loop Framework Built and Tested on Six Public Tables

**arXiv ID:** 2610.36395 | [PDF](https://arxiv.org/pdf/2610.36395v1)

**作者:** Zhiyuan Zhang `[一作]`, Zhiyuan Zhang `[通讯]`

**关键词:** `Machine Learning` `Drug Discovery` `Optimization` `Autonomous Driving` `Reinforcement Learning from Human Feedback` `Federated Learning` `Explainability and Interpretability` `Computational Efficiency` `Robotic Intelligence` `Meta Learning` `Knowledge Distillation` `Representation Learning` `Hyperparameter Search` `Data-Centric Learning` `Transformer` `Convolutional Neural Network` `Recurrent Neural Network` `Graph Neural Network` `Spiking Neural Network` `Supervised Fine-Tuning` `Reinforcement Learning` `Prompt Engineering` `Contrastive Learning` `Auto Encoder` `Generative Adversarial Network` `Conformal Prediction` `Quantile Regression` `Hierarchical Calibration` `Label-free Applicability Domain Screening` `Decision Rule Learning` `Budget Planning Strategy` `Tabular` `Time Series` `Sequential` `Benchmark` `Physics Related`

**🎯 论文内容**

本文提出并验证了一个自驱动实验室（self‑driving laboratory）决策框架，将学习到的设计代理（surrogate）与实验仪器闭环。通过六个公开的微滴/脂质体实验表格，模拟 107 个离线实验“目标实验室”，在四个关键决策（能否信任、所需测量量、先测哪些、何时停止/升级）上给出可量化指标（competence ratio、预算 k*(ε)、时间‑目标、决策成本），并在预注册、封闭验证和负控制等严格实验设计下得到23条系统性发现。

**💡 创新点**

创新点：①首次在微滴/脂质体领域公开、预注册并封闭验证（sealed‑prediction）设计代理的可靠性；②建立统一的决策框架，将 physics‑guided hard constraints 与逆设计评估融为一体；③量化不同 shift 类型的成本与代理迁移效应；④在离线实验表格上实现完整的闭环基准（包括采集、适应、建议和决策），可直接转化到真实实验装置。

**🔧 技术方法**

技术方法：使用 XGBoost（重实现）、V2 physics‑guided regressor、warm‑start residual boosting、随机森林、MLP、量化森林、GP、深度集成等多种 surrogate；采用 conformal prediction、分位数回归、分层校准、置信区间、label‑free 适用域筛选（k‑NN、Mahalanobis、Isolation Forest 等）；利用决策规则学习（基于两行诊断）和预算规划策略；所有实验均在 Python/Scikit‑Learn/XGBoost 环境下实现，并用 SHA‑256 记录所有程序、数据哈希。

**📊 数据集**

数据集：六个公开表格——DAFD 3.0（868 行综合 + 64 行外部文献）、Chagot et al.（392/76 行）、Smart Microfluidics liposome（246+57 行）、A1（901 行）、R2‑A1（830 行）以及 R2‑A2（1092 行）。每个表格分别代表不同的 shift 场景（设备、流体、化学、实验室、脂质、制备方法）。

**📈 对比分析**

比较方法与性能：在 107 个离线层级上，比较 14 种闭环策略、6 种 source 模型族、warm‑start 与 scratch 适应、不同采集规则（k0 规则）、决策成本 λ 等。结果表明：①对未见化学组的 surrogate，competence ratio 常在 1–2 之间，表明模型失效；②warm‑start residual boosting 能在 20 条测量内把误差恢复至 1.25× 开发误差；③在 301 个困难单元上，提出一次并 warm‑start 能将 10% 目标达到率从 3.31 次降低到 2.80 次；④预算 k*(ε) 在 chemistry、device、lab 三级 shift 下差异显著，但可通过 shift‑type 预估；⑤决策规则在 λ=5/10 下比始终继续更经济，且可在 94% 的化学层级中保持 90% 成功率；⑥负控制验证表明绝大多数“优”发现并非偶然。

**⚠️ 局限性**

限制：①未在真实 wet‑lab 实验中验证，所有结论仅基于离线表格模拟；②仅针对 droplet microfluidics 与脂质体两类实验，结果可能不泛化到其它物理实验；③对 label‑free 适用域筛选效果有限，仍未能完全预测单个候选的可靠性；④shift 评估基于已知标签分组，真实场景中的未知 shift 可能更复杂；⑤在高噪声或极端稀疏采样条件下，置信区间与决策规则的表现不稳定。

---

## 191. Blindfold: Keeping the Audit Coin Secret from LLM Agents

**arXiv ID:** 2610.37055 | [PDF](https://arxiv.org/pdf/2610.37055v1)

**作者:** Wei Zhu `[一作]` (Renmin University of China), Huiping Sun `[通讯]` (Peking University)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Statistical Hypothesis Testing` `Random Sampling` `Inverse Probability Weighting` `Text` `Time Series`

**🎯 论文内容**

本文研究了大型语言模型代理在执行授权操作时的审计式认证，提出了前置非干预（PNI）属性来量化审计硬币泄露对违规率估计的影响，并在真实代理栈上验证了该方法的有效性。

**💡 创新点**

创新点在于：①将PNI形式化为信息流属性并与误差偏差（g‑泄露）对应；②证明PNI为无偏认证的必要且充分条件并给出精确的偏差上界；③给出多步泄露组合的精确与渐进界限；④设计“盲fold”机制（静默分支、每次调用新硬币、诱饵提示等）以消除泄露；⑤在真实系统中测量并演示泄露导致认证失效与修复措施的效果。

**🔧 技术方法**

所用技术包括信息流理论（非干预与可解密）、全变差距离量化泄露、随机抽样审计与逆倾向加权（IPW）估计、时间均匀置信序列（empirical‑Bernstein）、统计假设检验、LLM注入实验与时间戳侧信道测量。

**📊 数据集**

实验使用了 560 条注入任务对（包含 500 次许可、500 次拒绝），在 1000 个对照对上进行审计，使用 gpt‑4.1‑mini 和 gpt‑4o‑mini 进行攻击实验，并收集了真实门控日志、工具时间戳与预注册的执行计划。

**📈 对比分析**

与传统的全局校准 conformal 方法、非随机审计及先验风险控制相比，本文提出的盲fold+静默审计方案在覆盖率（>0.99）和估计误差（≤(1‑ε)δ）上显著优于纯审计；每次调用新硬币可线性降低偏差，而静默分支和诱饵提示能够实质性阻止侧信道攻击。

**⚠️ 局限性**

局限性包括：实验仅在构建的监控器栈上完成，缺乏公开大规模或多模型验证；只评估了单一注入任务和单一 LLM，未检验跨 episode 自适应泄露；对共享负载等并行处理的泄露仍需进一步隔离；需要大量样本才能可靠估计 δ；在高频率或复杂工具链中时间戳侧信道仍可能存在残留泄露。

---

## 192. HoloRecon: Incremental Reconstruction of Long-Range Panoramas with Global Consistency

**arXiv ID:** 2610.36876 | [PDF](https://arxiv.org/pdf/2610.36876v1)

**作者:** Zhuoxiao Li `[一作]` (Hong Kong University of Science and Technology), Wufan Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Pose Estimation` `Depth Estimation` `Optimization` `Convolutional Neural Network` `Graph Neural Network` `Simultaneous Localization and Mapping` `Optical Flow` `Gaussian Splatting` `Supervised Fine-Tuning` `Graph Optimization` `Image` `Point Cloud` `Video` `Benchmark`

**🎯 论文内容**

提出了将稀疏ORB特征映射与前馈几何模型相结合的全景重建框架，利用预测深度实现跨视图3D验证和全局闭环纠正。

**💡 创新点**

创新点在于使用前馈预测几何校准稀疏地图，进而在无共享稀疏特征的情况下通过3D验证确立远程相机约束，并引入专门的PanoRecon基准。

**🔧 技术方法**

采用了ORB稀疏映射、Pi3/Finetuned前馈几何模型、相似变换验证、GPU加速的全景束调整、图优化与密集几何融合等技术。

**📊 数据集**

使用了360LOC、YUTO MMS以及新发布的7场景PanoRecon（含LiDAR参考）进行评估。

**📈 对比分析**

与PanoVGGT、Scal3R、COLMAP、Spherical‑SfM、360DVO、Open‑vSLAM、Stella‑vSLAM‑Dense等方法在ATE、RPE、精度/召回/ F‑score等指标上对比，本文在轨迹精度（ATE）最小、密集重建F‑score最高，尤其在长距离重访场景中显著提升。

**⚠️ 局限性**

局限在于对极端运动模糊、低纹理区及动态物体的鲁棒性不足，尚未实现实时重建，且在高帧率长序列仍有内存/计算瓶颈。

---

## 193. ARISE: Authentic Reference-Informed Stability Enhancement for Continual Multimodal Forgery Detection

**arXiv ID:** 2610.36509 | [PDF](https://arxiv.org/pdf/2610.36509v1)

**作者:** Yuchen Zhang `[一作]` (Xi'an Jiaotong University), Zhedong Zheng `[通讯]` (University of Macau)

**关键词:** `Computer Vision and Pattern Recognition` `Anomaly Detection` `Federated Learning` `Computational Efficiency` `Representation Learning` `Adversarial Attack` `Vision Language Model` `Transformer` `Vision Language Model` `Contrastive Learning` `Gaussian Mixture Model` `Experience Replay` `Multimodality` `Text` `Image` `Benchmark`

**🎯 论文内容**

提出攻击增量连续多模态伪造检测框架，并通过真实样本参考实现稳定性增强，提出ARISE方法；

**💡 创新点**

①提出 attack‑incremental continual learning 设置；②构建三协议（联合模态、文本侧、图像侧）演进的 FABLE‑Bench ；③引入 Authentic Reference‑Informed Stability Enhancement，利用真实样本的 Gaussian 混合参考生成 GRASP（责任感驱动的各向异性表征稳定）和 VAR（脆弱性感知重放加权）两种机制；

**🔧 技术方法**

使用多模态视觉‑文本 backbone（如 Florence‑2、CLIP、ViLT）+ 经验回放、Gaussian Mixture Model、Gaussian‑Responsibility‑Aware Stability Preservation、Vulnerability‑Aware Replay；

**📊 数据集**

FABLE‑Bench（821k 图文对，30+ 攻击机理，三种演进协议）、Weibo、FineFake 公开数据集；

**📈 对比分析**

与多类持续学习方法（架构扩展、经验回放、正则化）以及零射 MLLM 进行基准比较。ARISE 在 P1、P2、P3 协议上平均准确率分别为 85.37%、81.22%、90.47%，比最接近方法 LPR 提升约 1–2%；在 Weibo、FineFake 上亦取得最高 Accuracy/Macro‑F1；在 CLIP、ViLT 等不同 backbone 上保持显著提升；

**⚠️ 局限性**

需要在训练阶段维护重放记忆和 GMM 计算，存储和计算成本相对较高；对极端新型攻击的即时适应仍受限；在某些任务中仍存在少量遗忘，且实验主要集中在图文伪造场景，未覆盖更广泛的多模态攻击。

---

## 194. SCOPE: Source Credibility and Outlet Profiling Engine

**arXiv ID:** 2610.35929 | [PDF](https://arxiv.org/pdf/2610.35929v1)

**作者:** Ilyas Nagiyev `[一作]` (Nazarbayev University), Preslav Nakov `[通讯]` (MBZUAI)

**关键词:** `Multiagent Systems` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Agentic AI` `Text` `URL`

**🎯 论文内容**

构建了SCOPE多代理系统，对新闻媒体URL进行偏见、事实性和可信度的结构化、可解释性评估。

**💡 创新点**

通过将媒体评估流程拆分为专门的采集、内容分析、证据检索和报告生成四个代理，并提供可审计的中间证据，实现更透明、可靠的媒体画像。

**🔧 技术方法**

使用LLM驱动的多代理架构、爬虫、DuckDuckGo检索、外部知识库（Tranco、RSF、Freedom House）以及MBFC方法学。

**📊 数据集**

在200个MBFC目录中的媒体渠道上进行实验，使用MBFC的偏见、事实性和可信度标签作为黄金标准。

**📈 对比分析**

与LLM-only、article-only、search-only以及MBC等基线对比，SCOPE在MAE、事实性检测、错误率等八项指标上均优于基线，成本约为0.05美元/媒体。

**⚠️ 局限性**

局限性包括主要针对英语、美国媒体；潜在的间接标签泄露；分析器对超链接来源识别不完整；对极端阴谋或伪科学媒体的政治倾向定位困难。

---

## 195. Keeping Contact: Task-Adaptive Friction Margins for Bimanual Non-Prehensile Manipulation

**arXiv ID:** 2610.36961 | [PDF](https://arxiv.org/pdf/2610.36961v1)

**作者:** Qian Ying Lee `[一作]` (Nanyang Technological University), Domenico Campolo `[通讯]` (Nanyang Technological University)

**关键词:** `Robotics` `Optimization` `Robotic Intelligence` `Convolutional Neural Network` `Reinforcement Learning` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation` `Tabular` `Time Series`

**🎯 论文内容**

提出在双手非抓握平面倾斜任务中使用有符号摩擦距离来规划接触力，并给出了任务自适应摩擦裕度；

**💡 创新点**

创新点在于将摩擦裕度定义为力空间内到摩擦边界的有符号距离，并根据转动瞬时的杠杆需求自适应地设定裕度，从而实现全局最优的SOCP求解；

**🔧 技术方法**

采用凸二阶锥规划（SOCP）、有符号摩擦距离约束、任务自适应裕度计算以及基于力/扭矩传感器的阻尼控制；

**📊 数据集**

通过实物实验，使用两台7-DOF Kinova Gen3机械臂与盒子物体，改变物体质量和重心位置；

**📈 对比分析**

与无裕度和固定裕度两种对照进行对比，实验表明任务自适应裕度显著降低摩擦利用率、减少滑移失效，并在模型不匹配时仍保持成功率；

**⚠️ 局限性**

局限性包括仅考虑单一接触模式，未对多模式规划做高层搜索，缺乏在线动态更新裕度与控制策略的闭环扩展。

---

## 196. Vision-Language Agents Can Learn in Their Own Latent Worlds

**arXiv ID:** 2610.36925 | [PDF](https://arxiv.org/pdf/2610.36925v1)

**作者:** Haoxi Li `[一作]` (HKUST), Song Guo `[通讯]` (HKUST)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Robotic Intelligence` `Transformer` `Vision Language Model` `World Model` `Reinforcement Learning` `Contrastive Learning` `LoRA` `PPO` `Potential-based Planning and Replay` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出一种将预训练视觉语言模型（VLM）重新构造为递归潜在转移算子，并与模型基础强化学习相结合的框架，以在自身潜在空间内进行想象与学习。

**💡 创新点**

共享同一 VLM 骨干的潜在表示，同时通过轻量 LoRA 适配器实现世界模型与策略的双重角色，并在在线交互中持续更新潜在动态，从而无需生成完整观测。

**🔧 技术方法**

采用预训练 VLM+V-JEPA2 编码器、LoRA 微调、潜在递归动态学习、奖励与表示对齐损失、PPO 政策与价值更新以及基于潜在的规划与回放等技术。

**📊 数据集**

在五个交互式视觉环境（Sokoban、FrozenLake、Navigation、PrimitiveSkill、EB-ALFRED）上进行训练与评估。

**📈 对比分析**

与 PPO、GRPO 以及 LeWorldModel 等基线对比，实验显示在大多数任务中取得最高成功率，并且在计算成本上优于基于视觉生成的世界模型。

**⚠️ 局限性**

潜在想象在某些任务（导航、操作）收益有限，缺乏不确定性估计，模型仅在单一 VLM 与仿真环境验证，且需进一步研究何时何距进行想象以提升效率。

---

## 197. Latent Visual Contrastive Reasoning with Multi-Branch Exploration

**arXiv ID:** 2610.36809 | [PDF](https://arxiv.org/pdf/2610.36809v1)

**作者:** Kegeng Tang `[一作]` (University of Tennessee at Chattanooga), Zihao Wang `[通讯]` (University of Tennessee at Chattanooga)

**关键词:** `Computer Vision and Pattern Recognition` `Representation Learning` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Vision Language Model` `Contrastive Learning` `Chain-of-Thought` `Image` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

提出Latent Visual Contrastive Reasoning (LVCR)框架，在视觉问答中生成多条问题条件的隐式视觉分支，并通过软对比学习对齐这些分支与标注的视觉证据。

**💡 创新点**

创新点包括：①多分支隐式视觉推理；②基于边框的软对比损失，实现分支与证据的对齐；③推理时根据问题相关性对分支排序并注入解码器，避免单一路径信息瓶颈。

**🔧 技术方法**

使用Qwen2.5-VL-Instruct大模型作为骨干，结合视觉编码器、问题条件注意力、分支查询、对比学习（NCE）、软归一化权重、分支排序等技术。

**📊 数据集**

训练数据采用VISUAL COT（含边框证据）的417k条图文问答；评估使用V* Bench、MMVP、BLINK等感知密集视觉推理基准。

**📈 对比分析**

在3B和7B规模下与原始Qwen2.5-VL、Vision‑R1、LVR、PAPO、PixelReasoner等基线进行对比。LVCR在多项指标（如V*整体、MMVP、BLINK子任务）上显著提升，提升幅度通常在1–4个百分点。

**⚠️ 局限性**

局限性：①对训练时的边框标注依赖；②分支数目固定，无法自适应多样化的证据数量；③在某些基准（如MMVP的相对位置子任务）仍不如专门的像素级或工具增强方法。

---

## 198. Property-Based Testing for Robust Softmax and Cross-Entropy

**arXiv ID:** 2610.36427 | [PDF](https://arxiv.org/pdf/2610.36427v1)

**作者:** Muhammad Maaz `[一作]`, Muhammad Maaz `[通讯]`

**关键词:** `Machine Learning` `Optimization` `Computational Efficiency` `Robotic Intelligence` `Federated Learning` `Explainability and Interpretability` `Property-Based Testing` `Automatic Differentiation` `Numerical Analysis`

**🎯 论文内容**

通过 property‑based 测试重新发现 softmax 与交叉熵在标准机器学习库中的数值错误，并提出一种更稳健的实现；

**💡 创新点**

创新点在于用 log1p(S) 替代 softplus 处理 max‑shift 后的求和，既避免了 NaN 梯度，又兼容因子化注意力掩码等极端输入；

**🔧 技术方法**

使用了 Hypothesis 的 property‑based 测试、自动微分、数值分析与 PyTorch 的 GPU/CPU 计算；

**📊 数据集**

本工作不依赖公开数据集，而是通过随机生成的浮点向量（至少 2 维）作为输入进行测试；

**📈 对比分析**

通过验证 softmax Jacobian 行和为零以及在学生与教师 logits 相同情况下梯度为零等数学性质，与 PyTorch 原生实现及先前修复版本进行对比；性能与原实现相近，计算速度基本不受影响；

**⚠️ 局限性**

局限性包括仅在 float64 环境下验证、未在所有深度学习框架中广泛测试、对极低精度（如 bf16）下的鲁棒性仍待进一步评估。

---

## 199. FIDES-Live: Evaluating Behavioral Concordance in Sequential Trading Agents

**arXiv ID:** 2610.36476 | [PDF](https://arxiv.org/pdf/2610.36476v1)

**作者:** Arther Tian `[一作]` (DGrid AI), Aaron Chan `[通讯]` (DGrid AI)

**关键词:** `Cryptography and Security` `Finance Related` `Sequential` `Tabular`

**🎯 论文内容**

对实时交易代理的操作轨迹进行细粒度审计，检查决策有效性、可选承诺、执行与账本对账以及报告真实性，形成一套完整的审计协议 FIDES‑Live。

**💡 创新点**

创新点在于将 FIDES 的一致性评估从一次性策略转移到连续的交易循环，拆分为四个可独立评估的维度，并引入可见的不可评估（unscorable）和不可用（unavailable）标记，避免把不同来源的错误混合成单一分数。

**🔧 技术方法**

使用了确定性 JSON 解析与结构校验、约束检查（现金与持仓上限）、账本重建与对账、报告字段一致性检验，以及基于时间戳的序列重构和哈希完整性校验。

**📊 数据集**

基于两套前向运行的实验数据：一套是 2026 年 9 月 30 日至 10 月 7 日的加密货币现货盘（共 7,299 轮），另一套是 2026 年 9 月 28 日至 29 日的股票试点（共 310 轮）。每轮包含市场观察、决策、订单、执行收据、账面状态和后续报告。

**📈 对比分析**

通过对每条轨迹的四个维度分别计算通过率，得到决策有效率（约 99%+）、报告准确率（加密货币 14%–100%，股票 99.7%+）、承诺覆盖率（0–100%）和执行对账一致率（100%）。不同模型路由在每个维度上表现差异显著，说明单一总体分数会掩盖重要细节。

**⚠️ 局限性**

局限性包括：1) 数据为回溯性、依赖单一市场路径且缺乏随机化实验，无法得出因果或长期 alpha 结论；2) 仅检验结构与数值一致性，未评估自然语言解释的语义忠实度；3) 执行采用本地纸面模拟，未覆盖真实交易所的深度、排队和冲击；4) 未对模型身份和请求路由进行严格验证，导致对模型性能的归因受限。

---

## 200. Counting the Bandages: On measuring the time that AI saves

**arXiv ID:** 2610.35942 | [PDF](https://arxiv.org/pdf/2610.35942v1)

**作者:** Solomon Shalom Lijo `[一作]` (Independent Researcher), Solomon Shalom Lijo `[通讯]` (Independent Researcher)

**关键词:** `Artificial Intelligence` `AI Code Assistant` `Optimization` `Computational Efficiency` `Data-Centric Learning` `Prompt Engineering` `Retrieval-Augmented Generation` `Supervised Fine-Tuning` `Chain-of-Thought` `Tabular` `Text` `Finance Related`

**🎯 论文内容**

本文从“账本视角”重新定义AI辅助工作的时间节省，提出时间节省、缺陷逃逸率以及盈亏平衡缺陷成本等指标，利用理论模型、识别框架、仿真复现以及实际实验，对AI辅助工作效率与质量之间的权衡进行了系统评估。

**💡 创新点**

创新点包括：①将时间节省与缺陷成本关联，推出盈亏平衡缺陷成本概念；②提出完整的随机化研究时间节省度量识别框架，揭示哪些量可估计、哪些只能边界；③推导检查模型的解析解，证明过度信任导致更快但更差；④发布完整实验、复现代码与数据，形成可重复的评估标准。

**🔧 技术方法**

技术方法主要包括：搜索理论下的指数检测律、Amdahl定理、极值比较、部分识别与Makarov bounds、PPML与聚类置信区间、Bootstrap与jackknife-t、随机实验设计、仿真与最优搜索推导。

**📊 数据集**

使用的数据集：METR公开的246个软件开发问题、GDPval实验数据、SEC 10‑Q财报数据（30家公司）以及实验中人为注入错误的模拟数据。

**📈 对比分析**

比较方法采用比例均值（ratio of means）和几何平均比率两种总结，推断使用PPML、cluster‑bootstrap和jackknife‑t；在METR复现中得到+18.8%至+33.7%的加速，GDPval上限约为3.6×；实验中AI评审根据错误率按ln‑比例增加检查次数，过度信任导致更快但错误率显著上升。

**⚠️ 局限性**

局限性：模型假设单一缺陷、指数检测律、无误报；缺陷后续成本c未知，盈亏平衡缺陷成本需按组织自行设定；实验仅测试单一AI评审模型，缺少人类参与；跨研究质量指标不一致，识别区间宽；部分识别方法依赖先验分布；结果对具体组织成本阈值高度相关。

---

## 201. When the Critic Falls Behind: Co-Evolving Rubrics and Harnesses for Continual Agent Self-Improvement

**arXiv ID:** 2610.36826 | [PDF](https://arxiv.org/pdf/2610.36826v1)

**作者:** Hao Chen `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Optimization` `Meta Learning` `Drug Discovery` `AI Code Assistant` `Robotic Intelligence` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Contrastive Learning` `Reinforcement Learning` `Text` `Sequential` `Benchmark`

**🎯 论文内容**

RuHarness 框架通过共同进化 Rubric 与 Harness，实现在 LLM 代理上持续自我改进的机制；

**💡 创新点**

其创新点在于动态诊断 Rubric 以跟踪代理行为弱点的变化，并以最具诊断价值的弱点优先指导 Harness 更新，从而提升长期改进和跨任务泛化；

**🔧 技术方法**

技术上结合轨迹分析生成失败报告、四状态 Rubric 生命周期管理、时间可变诊断效用公式、Dirichlet 后验保守估计、六维 Harness 协同更新与验证门控候选选择；

**📊 数据集**

实验使用 EvoAgentBench 这一跨领域长序列代理自进化基准，涵盖信息检索、数学推理、软件工程、代码实现和知识工作等五大领域，并在 Qwen3.5‑35B‑A3B 与 GLM‑4.7‑Flash 两大 backbone 上进行评估；

**📈 对比分析**

与 ACE、EvoSkill、Self‑Harness、Meta‑Harness、AHE 等自进化基线对比，RuHarness 在 held‑in 与 held‑out 集合上均表现最优，尤其在未见任务上提升 4.5‑5.3 分，显著缩小泛化差距；

**⚠️ 局限性**

限制在于实验仅在单一代理框架中验证六维 Harness 抽象，缺乏在更广泛代理架构上的泛化验证，且演化周期固定，未探索长期开放式适应场景。

---

## 202. SpatialPC: Deep Computation with Sublinear Local Credit Propagation

**arXiv ID:** 2610.36139 | [PDF](https://arxiv.org/pdf/2610.36139v1)

**作者:** Benjamin Prada `[一作]` (University of South Florida), Ankur Arjun Mali `[通讯]` (University of South Florida)

**通讯引用:** 710 | [OpenAlex ID](https://openalex.org/A5068199296)

**关键词:** `Machine Learning` `Computational Efficiency` `Representation Learning` `Graph Neural Network` `Contrastive Learning` `Auto Encoder` `Optical Flow` `Stochastic Differential Equation` `Image`

**🎯 论文内容**

在二维常密度、单位范围、有限度的计算平台上，证明了局部学习规则的信用传播延迟为Ω(N^{1/d})，并构造了一个单元范围、度为2d的有向无环图，该图实现了N-1长度的Hamiltonian计算，同时在Θ(N^{1/d})轮内将损失信息传递给所有参数。随后提出了SpatialPC——一种基于预测编码的本地信用传播框架，使用伴随预测动态（APD）在编译后的稀疏Moore曲线图上实现精确的边局部梯度，并通过可调的可行性权重控制梯度精度。

**💡 创新点**

（1）首次将信用传播延迟归因于通信几何而非计算深度，给出紧致的Ω(N^{1/d})下界；（2）设计出实现该下界的单元范围图和稀疏Moore编译器；（3）将预测编码与本地APD结合，形成可在有限轮内保持梯度一致性的SpatialPC；（4）在有限轮预算下，证明梯度与收敛梯度对齐并达到与深度128反向传播相当的准确率。

**🔧 技术方法**

光锥下界证明、单元范围有向无环图构造、稀疏Moore空间填充曲线编译、预测编码与APD的同步本地动态、可行性残差正则化、梯度余弦对齐评价、实验中的梯度与残差监测。

**📊 数据集**

MNIST与Fashion‑MNIST（标准60k/10k划分，784维像素）。

**📈 对比分析**

通过将SpatialPC在不同N（16、64、256）下使用T=2√N轮训练，比较其梯度余弦（最高≈0.97）和测试准确率（MNIST≈97.6%，Fashion‑MNIST≈88.1%）与深度128、宽度128的传统反向传播基线，发现相当甚至略优；链式控制模型在相同设置下性能几乎为随机。

**⚠️ 局限性**

（1）Moore曲线的最佳到达深度D_reach=√N+2在所有规模下仅实验证明；（2）有效信用半径R_ε的上界仍取决于学习得到的Jacobian，尚未给出随N增长的理论上限；（3）仅验证了二维情形，三维堆叠、卷积或序列结构的推广未实验；（4）本研究未考虑非本地广播或合成梯度等超越本地模型的方案。

---

## 203. Is the education gap in generative AI use smaller than the internet's was at a similar level of use? Evidence from 16 European countries

**arXiv ID:** 2610.36181 | [PDF](https://arxiv.org/pdf/2610.36181v1)

**作者:** Keisuke Watanabe `[一作]` (Glasp Inc), Keisuke Watanabe `[通讯]` (Glasp Inc)

**关键词:** `Information Retrieval` `Tabular` `Review/Survey Paper`

**🎯 论文内容**

使用欧盟Eurostat 2025年ICT调查数据，对16个可比国家在2025年生成式AI使用与互联网使用总体水平相匹配的年份，比较完成教育水平（高 vs. 低）之间的使用差距。

**💡 创新点**

创新点在于将生成式AI与互联网在相同总体使用水平下的教育差距进行匹配比较，而非单纯按技术发布时间或扩散阶段比较；同时使用对数比率与点差两种指标，并通过分位数自助法和符号检验给出置信区间与显著性。

**🔧 技术方法**

主要技术包括：对数比率（ln (p_H/p_L)）、点差（p_H‑p_L）、logit差、Bootstrap 95% 区间、两边符号检验、线性插值、教育构成标准化匹配、银行业务对比等。

**📊 数据集**

数据集为：Eurostat 2025年生成式AI使用（isoc_ai_iaiu）、互联网使用（isoc_ci_ifp_iu）、互联网银行业务（isoc_ci_ac_i）以及人口教育构成（lfsa_pgaed）。

**📈 对比分析**

比较方法：先按总体使用率将生成式AI的2025年率与互联网不同年份的总体率匹配（差距≤5个百分点），然后计算两组教育差距的变化（Δ），再对所有匹配国家的Δ取中位数并进行Bootstrap和符号检验。结果显示，生成式AI的教育差距普遍小于互联网对应水平，点差中位数下降约27.6个百分点，符号检验p值<0.001，可信区间均为负值。

**⚠️ 局限性**

局限性包括：只纳入互联网采用相对较晚的16国，结果不具普适性；使用单一年份的生成式AI数据，缺乏长期趋势；未对年龄、就业、学生状态等混杂因素进行调整；未公开预注册计划；自助法假设国家样本独立，可能低估误差。

---

## 204. Ground-to-Satellite Knowledge Distillation for Crop Classification under Temporal and Geographic Shift

**arXiv ID:** 2610.37021 | [PDF](https://arxiv.org/pdf/2610.37021v1)

**作者:** Miro Miranda `[一作]` (Forschungszentrum Jülich), Claudia Paris `[通讯]` (University of Twente)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Knowledge Distillation` `Domain Adaptation` `Agriculture Related` `Transformer` `Vision Language Model` `Contrastive Learning` `Image` `Multimodality`

**🎯 论文内容**

利用训练时可获得的地面相机照片作为特权监督，改进仅使用卫星影像的作物分类模型。

**💡 创新点**

提出在冻结的卫星嵌入上进行知识蒸馏，显著提升时空迁移下的鲁棒性，并证明地面视角的监督能有效弥补卫星表征的局限。

**🔧 技术方法**

使用CLIP ViT‑B/32 作为教师、AlphaEarth 与 TESSERA 嵌入作为学生、S2/S1 月度复合序列变压器，并结合 KL、余弦相似度的蒸馏损失。

**📊 数据集**

基于 LUCAS 2018 数据集，包含 53,885 个配套地面照片与 Sentinel‑2/1 月度复合的 17 类作物标签。

**📈 对比分析**

与仅使用交叉熵的基线模型对比，KD 在冻结模型上提升 2.6–5.1 点宏观 F1，冻结 TESSERA 在 2022 年迁移上提升 5.5 点，且在地理留出（地中海）中增益更显著。

**⚠️ 局限性**

受限于训练阶段需要的地面照片；不同输入路径导致冻结与 fine‑tune 之间的嵌入不匹配；在小规模地理留出上表现波动不定。

---

## 205. SO-101 Bench: Accessible Spatial Manipulation

**arXiv ID:** 2610.36420 | [PDF](https://arxiv.org/pdf/2610.36420v1)

**作者:** Truman Hickok `[一作]` (Independent), Truman Hickok `[通讯]` (Independent)

**关键词:** `Computer Vision and Pattern Recognition` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Vision-Language-Action Model` `World Model` `Large Language Model` `Diffusion model` `Score-based Model` `Generative Adversarial Network` `Contrastive Learning` `Gaussian Splatting` `Simultaneous Localization and Mapping` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation` `Image` `Video` `Text` `Multimodality` `Point Cloud` `Mesh` `Benchmark`

**🎯 论文内容**

提出了SO-101 Bench——一种基于 Isaac Lab 数字双生的低成本SO-101机械臂语言驱动空间操控基准，并对其进行系统的仿真‑实机（sim2real）验证。

**💡 创新点**

核心创新在于①构建可准确保持策略排名的数字双生；②设计了三种对象泛化拆分（seen、unseen-object、unseen-class）和三大子集（ROBUST、PLAN、MEM）以检验空间推理、旋转匹配与记忆能力；③在相同 36 h GPU 预算下公平比较多种 VLAs、WAMs 与 LLM agent，并剖析 harness 与模型对性能的影响。

**🔧 技术方法**

利用 Isaac Lab 物理仿真、Vision‑Language‑Action (VLA) 与 World‑Action Model (WAM) 架构、GPT‑6.1 Sol 等 LLM 与 PyRUA‑Lean 代码‑as‑policy harness、SAM‑3 语义分割、逆运动学等技术。

**📊 数据集**

使用 138 件从 3D AI Studio 生成或扫描得到的家庭物体，收集 3,629 条仿真遥控演示与约 4,400 条真实机器人演示，涵盖 seen、unseen‑object、unseen‑class 三种拆分。

**📈 对比分析**

在 916 条主评估 episode 上按 15 个任务×拆分细胞计算平均成功率（95% 置信区间），结果显示最佳学习策略 InternW0‑Δ 与 Galaxea 0.5 约 61% 成功；LLM agent GPT‑6.1 Sol 达到 83%；在 45 条仿真‑实机对比中，排名完全一致，仿真对真实成功率高估 9–31%。

**⚠️ 局限性**

局限性包括：sim2real 仅在 5 个策略与 45 个布局下验证，置信区间宽；固定 36 h 预算导致部分策略欠训练；LLM agent 计算时间未计入实时成本；基准仅涵盖刚性物体与单臂操作，缺乏柔性、双臂与实时约束的扩展。

---

## 206. An Incremental Knowledge-Graph Learner Digital Twin for Evidence-Grounded Early Warning

**arXiv ID:** 2610.36820 | [PDF](https://arxiv.org/pdf/2610.36820v1)

**作者:** Ngoc Luyen Le `[一作]` (Gamaizer), Bertrand Laforge `[通讯]` (Sorbonne Université)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Anomaly Detection` `Graph Neural Network` `Contrastive Learning` `Prompt Engineering` `Chain-of-Thought` `Tabular` `Graph` `Review/Survey Paper`

**🎯 论文内容**

引入增量式知识图学习者数字孪生（Learner Digital Twin），将学习者轨迹、预测特征、归因、预警和证据链统一映射为图实体，实现可追踪、可验证的早期预警；

**💡 创新点**

通过知识图作为实时同步的学习者数字孪生，既是特征抽取的底层，又是解释的证据支撑；实现增量更新、PROV‑O 级的可追溯性和 SHACL 验证的解释完整性；

**🔧 技术方法**

RDF/OWL 知识图、SPARQL 查询、SHACL 验证、PROV‑O 语义模型、TreeSHAP 归因、树形模型（XGB、RF、ET、HGB、LGBM、CAT、TABTX）等；

**📊 数据集**

Open University Learning Analytics Dataset (OULAD)——22 个演示、7 个模块、32,593 名学生、约 10.6M VLE 交互记录；

**📈 对比分析**

与传统原始表特征对应的 KG 特征在所有模型和时间检查点下实现预测性能平价（F1、AUC 相同），并在预警时提供完整证据链，Direct@k、Fid@k 等指标均达到 90% 以上；

**⚠️ 局限性**

仅为离线检查点增量同步，未评估实时流式、乱序事件、错误修正、实时部署与教师反馈；缺乏对教师决策影响的评估，未扩展到能力知识表示。

---

## 207. Interpretable Deep Learning Models for Breast Cancer Diagnosis Using Medical Imaging

**arXiv ID:** 2610.37026 | [PDF](https://arxiv.org/pdf/2610.37026v1)

**作者:** Sami Alnawali `[一作]` (King Faisal University), Ahmad Din `[通讯]` (National University of Computer and Emerging Sciences)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Explainability and Interpretability` `Anomaly Detection` `Convolutional Neural Network` `Transformer` `Supervised Fine-Tuning` `Contrastive Learning` `Image` `Biomedical Data` `Magnetic Resonance Imaging` `Computed Tomography`

**🎯 论文内容**

对 BreakHis 数据集上的乳腺癌二分类问题，评估并比较了十种先进的迁移学习模型，并加入了多种可解释 AI 方法。

**💡 创新点**

提出了一套完整的评估框架，将传统 CNN 与 Transformer 结合，并同时展示模型的可解释性。

**🔧 技术方法**

采用迁移学习（ResNet50、DenseNet121、EfficientNet、Xception、MobileNetV2、VGG19、InceptionV3、ConvNeXt、Vision Transformer、Swin Transformer）以及 Grad‑CAM、LIME、SHAP、LRP 等 XAI 技术。

**📊 数据集**

使用公开的 BreakHis 病理图像数据库（7,909 张，四倍数放大率）。

**📈 对比分析**

通过患者级别拆分的验证集和测试集，对每种模型在所有放大率下的准确率、AUC、F1 等指标进行统计比较，DenseNet121 在测试集上实现 99.4% 准确率和 0.9997 AUC，优于其他模型。

**⚠️ 局限性**

仅基于单一二分类任务且可能存在样本相关性导致指标偏高，且 XAI 评估仍为定性，缺乏多中心验证与多类别扩展。

---

## 208. JevRec: Linguistic Encoding with Jev-like Decision Modeling for LLM-based Generative Recommendation

**arXiv ID:** 2610.36937 | [PDF](https://arxiv.org/pdf/2610.36937v1)

**作者:** Qiyong Zhong `[一作]` (Zhejiang University), Li Zhang `[通讯]` (Zhejiang University)

**关键词:** `Computation and Language` `Recommendation System` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Sequential`

**🎯 论文内容**

提出一种基于语言学编码与Jev-like决策模型的生成式推荐框架JevRec

**💡 创新点**

通过可变长度词编码赋予热门商品更短SID、长尾商品更丰富语义；采用长度与内容分离的预测头降低梯度冲突

**🔧 技术方法**

变长词编码、共享码本、长度预测分支、长度条件生成、长度感知推理

**📊 数据集**

三个公开推荐数据集（未列出具体名称）

**📈 对比分析**

与多种基线对比，实验表明在推荐指标上持续领先，训练与推理时间低于TIGER，长度与内容预测的梯度冲突显著下降

**⚠️ 局限性**

未详细说明，但可能面临长度预测误差导致生成质量下降或对极长序列支持不足

---

## 209. When Connectivity Misleads: Guiding Agentic Search with Contrastive Evidence Paths

**arXiv ID:** 2610.36134 | [PDF](https://arxiv.org/pdf/2610.36134v1)

**作者:** Junchen Li `[一作]` (University of Electronic Science and Technology of China), Shuang Liang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1281 | [OpenAlex ID](https://openalex.org/A5088419677)

**关键词:** `Information Retrieval` `Retrieval` `Computational Efficiency` `Explainability and Interpretability` `Graph Neural Network` `Transformer` `Agentic AI` `Contrastive Learning` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Graph`

**🎯 论文内容**

提出 Contrastive Evidence Paths (CEP) 与 Structured Corpus Interaction (SCI)，通过图路径检索与结构化地图辅助搜索代理进行更高效的证据发现与验证。

**💡 创新点**

创新点：①将单条最强查询条件路径与背景访问分布校准相结合，抑制图连通性导致的排名偏差；②在 SCI 中引入结构化地图与锚点扩展，让代理可直接查看路径与实体链接，从而减少无关文档的干扰并提升探索效率。

**🔧 技术方法**

使用的技术包括：图构建（实体共享构成文档图）、最强路径动态规划、背景 PageRank 校准、最大乘积贝尔曼-福特算法、基于路径的排序与采样、结构化地图展示与锚点本地扩展。

**📊 数据集**

主要数据集：BrowseComp‑Plus（100k Web 文档 + 830 题），TREC‑COVID（171k 科学文档 + 50 题），以及 FineWeb‑Edu（90万文档）用于规模扩展实验。

**📈 对比分析**

与传统检索代理、DCI、DR‑DCI、RISE、RARG++、AtlasNav 等基线比较。实验表明：在 BrowseComp‑Plus 上，SCI 最高准确率 95.3%（比最优基线高 4–7%），工具调用和总生成成本降低 15–54%；在 TREC‑COVID 上，nDCG@10 达 81.8%（比 RARG++ 高 5.8pp），工具调用与成本分别降低 33% 与 48%。

**⚠️ 局限性**

局限性：①依赖实体抽取与图构建，实体覆盖与频率阈值可能影响检索效果；②在极大规模语料下（百万级以上）仍需进一步优化图搜索与路径回溯的计算复杂度；③目前仅在两个公开基准上验证，未在更广泛领域或多语言环境中测试。

---

## 210. Deep Bayesian Approaches for Interpretable Nonlinear Structural System Identification

**arXiv ID:** 2610.36633 | [PDF](https://arxiv.org/pdf/2610.36633v1)

**作者:** Wei Liu `[一作]` (National University of Singapore), Eleni Chatzi `[通讯]` (ETH Zurich)

**关键词:** `Machine Learning` `Explainability and Interpretability` `Optimization` `Physics Related` `Recurrent Neural Network` `Transformer` `Diffusion model` `Auto Encoder` `Contrastive Learning` `Stochastic Differential Equation` `Ordinary Differential Equation` `Time Series` `Tabular` `Sequential` `Benchmark`

**🎯 论文内容**

本文系统性介绍了深度贝叶斯滤波在非线性结构动力学识别中的应用，基于Deep Markov Model、Physics‑Guided DMM和Neural EKF三种框架进行实验验证。

**💡 创新点**

创新点在于将物理先验（如ROM、模态矩阵）与可学习的状态空间模型相结合，并对不同推断机制（变分推断 vs. 可微Kalman）进行比较，阐明物理结构对可解释性和预测性能的影响。

**🔧 技术方法**

主要技术包括可变分状态空间模型、可微递推滤波、ROM嵌入、Bouc‑Wen硬化/软化模型、LSTM/GRU编码器，以及ELBO与预测误差的联合训练。

**📊 数据集**

使用的实验数据来自一个三维两层两槽剪切框架的有限元模拟，配备Bouc‑Wen支撑和随机地震输入，并在不同温度、材料弹性模量和激励幅度下生成190组仿真。

**📈 对比分析**

比较方法包括后验重构、无测量反馈的开放式预测、隐藏状态恢复以及参数估计；结果显示PgDMM在重构和参数恢复上表现最好，而Neural EKF在短期预测和激励幅度估计上更优，整体误差在10⁻²至10⁻¹范围。

**⚠️ 局限性**

局限性包括：缺乏真实场测数据验证、对模型可识别性和不确定性校准不足、对高维结构和未知输入的适应性有限、以及对实时在线部署的计算成本需要进一步优化。

---

## 211. Beyond the Estimate: Wiener-Informed Decoding for Easier Training of Pixel-Space Flow Models

**arXiv ID:** 2610.36192 | [PDF](https://arxiv.org/pdf/2610.36192v1)

**作者:** Ziqiao Wang `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Flow-based Model` `Diffusion model` `Score-based Model` `Contrastive Learning` `Optical Flow` `Image` `Text` `Multimodality`

**🎯 论文内容**

本文提出一种名为Wiener-informed Decoding (WiD) 的方法，将粗糙的二阶统计（Wiener估计）嵌入到像素空间流模型的解码器中，以帮助网络更快、更稳定地学习速度场，并显著提升生成质量。

**💡 创新点**

创新点在于：① 将线性Wiener估计的分析、合成运算与解码器的特征处理和输出校正相结合，形成可训练的统计预测路径；② 通过系数接口让统计坐标与解码器特征共存，既提供了粗略先验，又为后续学习提供了便利；③ 在保持原始网络结构不变的前提下，用统计知识初始化参数，极大加速收敛并提升生成效果。

**🔧 技术方法**

技术核心包括：像素空间流模型（PixelDiT、DeCo）框架；Wiener估计的全局DCT和共享patch PCA统计；可训练的分析/合成矩阵、波特增益函数；系数扩展器与像素嵌入的融合；联合训练与传统的速度场目标；以及可选的REPA等辅助正则化。

**📊 数据集**

主要使用 ImageNet（256×256、512×512）进行类条件生成和文本到图像生成；另有 51,200 张样本用于统计估计；文本生成基准使用 GenEval、DPG-Bench 等评测集；实验也涵盖了 DeCo-XXL/16、PixelDiT-XL/16 等大规模模型。

**📈 对比分析**

与基线（原始 PixelDiT/DeCo、带 REPA 的版本）对比，WiD 在 80、260、500 轮训练中分别把 FID 从 2.36 降到 2.09、从 1.79 降到 1.61、从 1.54 降到 1.53，减少训练轮数 37.5%。同样的改进在 512×512 细化、文本到图像任务中也能显著提升 FID、sFID、IS 等指标。性能提升伴随参数增幅仅约 1%–1.3%。

**⚠️ 局限性**

局限性包括：① 依赖预先估计的二阶统计，若统计不佳或数据分布变化，效果可能下降；② 方法目前仅在像素空间流模型上验证，未证明能迁移到其他生成框架；③ 对高分辨率图像和多模态输入的泛化能力尚需进一步评估；④ 在极大模型规模下，额外的线性变换可能增加计算开销。

---

## 212. CaLLM: Continual Adaptation of LLMs via Non-Parametric Adapter Routing

**arXiv ID:** 2610.36994 | [PDF](https://arxiv.org/pdf/2610.36994v1)

**作者:** Bilge Celik `[一作]` (Eindhoven University of Technology), Joaquin Vanschoren `[通讯]` (Eindhoven University of Technology)

**关键词:** `Machine Learning` `Federated Learning` `Computational Efficiency` `Domain Adaptation` `Knowledge Distillation` `Representation Learning` `Meta Learning` `Transformer` `Large Language Model` `Mixture of Experts` `Supervised Fine-Tuning` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Sequential`

**🎯 论文内容**

提出 CaLLM 框架，利用原型路由与轻量级适配器实现大型语言模型的持续学习，避免灾难性遗忘。

**💡 创新点**

创新点在于将无梯度原型网络与动态专家池相结合，既支持在线与离线两种学习模式，又通过池容量调控实现正向/反向迁移的平衡。

**🔧 技术方法**

使用的技术包括原型网络、LoRA 等参数高效微调、冻结背骨、余弦距离路由、以及与传统经验重放、正则化等方法的对比。

**📊 数据集**

实验数据集为 TRACE‑8（跨领域任务）和 CITB‑19（长序列对话任务，含人工合成样本）。

**📈 对比分析**

与 Seq‑FT、Inference‑Only、EWC、Experience Replay、oLoRA、TreeLoRA 等基线比较，CaLLM 在平均任务性能、正向迁移（BWT）以及在线 OMA 上均显著优于对手。

**⚠️ 局限性**

局限性包括对池容量的敏感性、需要预先设定阈值、对跨语言相似任务的区分不足，以及实验仅覆盖文本任务，没有验证跨模态通用性。

---

## 213. Conservative Dueling Bandits: Preference Learning under Baseline Constraints

**arXiv ID:** 2610.36708 | [PDF](https://arxiv.org/pdf/2610.36708v1)

**作者:** Rohan Deb `[一作]`, Arindam Banerjee `[通讯]`

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning` `Inverse-Gap-Weighting` `Confidence Ellipsoid Estimation` `Log-Likelihood Maximization` `Tabular` `Synthetic Data`

**🎯 论文内容**

提出保守型上下文双击版（Conservative Contextual Dueling Bandits）模型，要求每一步的最佳展示项累积得分不低于基准策略的(1-α)倍；设计两类算法—针对线性效用的Conservative Maximum-Informative-Pair（C-MIP）和针对一般效用类的Inverse-Gap-Weighting Baseline-Conservative（IGW-BC），并证明其在任意时间点均满足安全约束，且总回报损失仅为一个与时间无关的加性项；

**💡 创新点**

创新点在于：①将安全约束改写为基于偏好得分的相对度量，避免需要绝对奖励；②通过效用差距而非绝对估计来校验安全；③在基准回退（fallback）阶段仍保持学习，通过与基准配对的有信息项保证信息流动；④提出两种适用于线性和一般函数类的算法，并给出理论上任意时刻的安全性与回报上界。

**🔧 技术方法**

使用技术包括：Bradley–Terry–Luce模型的对数似然最大化、基于置信椭圆的参数估计、逆间隙权重（Inverse-Gap-Weighting）采样、在线逻辑回归预言机、马尔科夫过程与高斯界等。

**📊 数据集**

实验数据集：人工合成的线性实例（d=10，K=50，T=10^4）与一般效用类（d=5，K=40，|F|=64随机二次函数），并设置多种α、d、K、基准位置q进行参数扫描。

**📈 对比分析**

与无约束算法MaxInP（线性）或BT-IGW（一般）以及仅使用基准策略的Baseline-Only进行对比；实验结果显示：①保守算法在所有设置下均满足安全约束，且无违约；②相较于无约束算法，保守算法的额外回报为常数级或低线性级；③Baseline-Only的回报线性增长，远逊于保守算法。

**⚠️ 局限性**

局限性：①假设基准得分已知或可用较宽范围估计；②模型仅考虑二项偏好，未涵盖非可传递偏好或多项比较；③安全约束是累计式，未考虑逐步（stagewise）约束；④在实际大规模任务中，在线预言机和置信区间的计算成本可能较高。

---

## 214. A 2-DOF MultiBody Track Joint for Race Car Optimization

**arXiv ID:** 2610.36407 | [PDF](https://arxiv.org/pdf/2610.36407v1)

**作者:** Srikanth Sivaramakrishnan `[一作]` (General Motors), Ryan Trinter `[通讯]` (General Motors)

**关键词:** `Robotics` `Optimization` `Autonomous Driving` `Diffusion model` `Score-based Model` `Contrastive Learning` `Gaussian Splatting` `Simultaneous Localization and Mapping` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation` `Point Cloud` `Tabular` `Time Series` `Benchmark` `Physics Related`

**🎯 论文内容**

本文设计并实现了一个 2 维自由度的 Modelica MultiBody 轨道关节，该关节通过六条四次样条（Monge patch）描述赛道曲面，既能在时间域仿真中为赛车提供精确的道路几何，又可直接用于 CasADi/IPOPT 的最短赛道时间优化。

**💡 创新点**

创新点包括：① 将完整赛道曲面封装为可复用的 MultiBody 组件；② 使用 d'Alembert 原理闭合约束，避免引入拉格朗日乘子；③ 通过解析地计算轨道三角系的角速度向量，支持车辆坐标相对轨道三角系，从而大幅简化优化初始猜测；④ 通过 FMI 3.0 导出单一 FMU，兼容时域仿真与直接点乘优化。

**🔧 技术方法**

技术栈涵盖：Modelica MultiBody、Monge patch 参数化、piecewise quintic B‑spline、Horner 展开评估、d'Alembert 虚功原理、FMI 3.0 交换、CasADi 直接点乘、IPOPT 与 MUMPS 线性求解器。

**📊 数据集**

使用了由配套拟合管道生成的六条四次样条，数据源为新加坡 Formula 1 赛道的 LiDAR 点云；同时在论文中对 23 条 2026 年 F1 赛道进行泛化验证。

**📈 对比分析**

性能对比：与 1‑DOF Claytex 轨道关节相比，本关节不需要额外角度表格，直接通过样条计算全三角系；在 23 条赛道的 injectivity margin（1/|κ|b）均≥1，保证 DAE 可解；单点质量跑完整盘时，Horner 与展开评估仅差 26% 的运算时间，误差均在 10⁻¹² 级；在最短赛道时间优化中，整个 FMU 约占 464 s 计算时间的 70%，其中约 167 s 用于目标与约束评估，说明评估效率对整体性能影响最大。

**⚠️ 局限性**

局限性：① 依赖外部拟合管道，无法自行判断参考线是否合格；② 对于不满足 injectivity margin 的赛道，约束矩阵会退化；③ 仅支持闭环赛道，断点不易处理；④ 采用均匀 knot，非均匀曲线需要额外搜索；⑤ 目前组件、模型和数据未公开，重现性受限；⑥ 需要多实例共享同一参数记录时不支持。

---

## 215. OmniBrain-0: A Unified Agentic Model for Embodied Intelligence

**arXiv ID:** 2610.36013 | [PDF](https://arxiv.org/pdf/2610.36013v1)

**作者:** Xiaomi Embodied Intelligence Team `[一作]` (University of Macau), Hangjun Ye `[通讯]`

**关键词:** `Computation and Language` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Agentic AI` `Large Language Model` `Reinforcement Learning` `Knowledge Distillation` `Vision-Language-Action Model` `Text` `Multimodality`

**🎯 论文内容**

提出了 OmniBrain-0，一种统一的紧凑型装置代理，通过高层决策模型协调多种低层操控与导航工具，并构建三阶段训练与递归自我改进闭环。

**💡 创新点**

创新点包括：1）将操控与导航通过共享工具抽象统一到同一决策层；2）采用多教师 on‑policy distillation（MOPD）将专业域专家的决策迁移到单一模型；3）引入 Rubric‑Guided Recursive Self‑Improvement（RSI）实现交互数据循环构建、提醒策略优化与专家更新；4）开发结构化交互数据管道，保留观测‑决策‑反馈序列并过滤噪声。

**🔧 技术方法**

核心技术为：Qwen3.5‑9B 作为高层决策器；工具调用与反馈统一接口；多教师 on‑policy distillation；Rubric‑guided reminder policy；交互归一化、推理重写与一致性校验；离线训练与强化学习融合；大规模文本与交互数据混合预训练。

**📊 数据集**

使用的数据集包括：LIBERO‑Pro、LIBERO‑Plus、RoboCasa365、RoboTwin 2.0、R2R‑CE Val‑Unseen；EmbSpatial‑Bench、RefSpatial‑Bench；以及三项真实世界物理任务（block stacking、apple placement、drawer opening）。

**📈 对比分析**

与基线专家策略（π_0.5、InternVLA‑N1、π_0.5‑Plus、GPT‑5.6‑Sol 等）以及多种少量样本基线进行对比。OmniBrain‑0 在所有 5 大基准上取得与老师相近或更优的成功率（65–88%），在完成时间上比 GPT‑5.6‑Sol 快 1–8 倍，token 消耗降低 4–12 倍；在空间问答和真实世界任务上亦显示显著提升。

**⚠️ 局限性**

局限性包括：对低层技能的覆盖与可靠性高度依赖；在未见任务组合和空间定位上的误差仍较大；真实世界实验任务有限，难以完全评估鲁棒性；对分布漂移和大规模多样化交互的持续改进仍需进一步探索。

---

## 216. Distributed LLM Inference via Communication-Efficient Speculative Decoding

**arXiv ID:** 2610.35957 | [PDF](https://arxiv.org/pdf/2610.35957v1)

**作者:** Joohyun Shin `[一作]` (Korea Advanced Institute of Science and Technology), Joonhyuk Kang `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `Machine Learning` `Computational Efficiency` `Large Language Model` `Transformer` `Large Language Model` `Distributed Speculative Decoding` `FP16 Computing` `GEMM Parallelism` `Text`

**🎯 论文内容**

提出 DISCO 框架，实现分布式 LLM 推理的通信高效 Speculative Decoding，消除词表规模通信开销。

**💡 创新点**

通过在服务器端复制草稿小模型，仅在服务器端本地重现草稿分布进行拒绝采样，保持目标分布精确且无需词表规模通信。

**🔧 技术方法**

采用分布式 Speculative Decoding、服务器端 SLM 复制、通信延迟与负载分析、FP16 计算及 GEMM 并行等技术。

**📊 数据集**

使用 GSM8K、CNN/DailyMail、HumanEval 等数据集，并对比 Qwen2.5 系列模型验证方法泛化性。

**📈 对比分析**

与 DSD、DSSD 两基线在不同带宽、草稿长度和硬件配置下对比，DISCO 在低带宽下可实现约 1.5× 速度提升，高带宽下仍保持领先，最高达约 1.9× 加速。

**⚠️ 局限性**

仅在单请求场景评估，未考虑多用户并发；服务器端 SLM 复制增加资源占用；对极大词表或不同模型家族的兼容性待进一步验证。

---

## 217. KORA: Deterministic-First Execution Control for AI Workloads

**arXiv ID:** 2610.35869 | [PDF](https://arxiv.org/pdf/2610.35869v1)

**作者:** Albert Heekwan Kim `[一作]` (Krako Labs), Albert Heekwan Kim `[通讯]` (Krako Labs)

**关键词:** `Machine Learning` `Computational Efficiency` `Optimization` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Mixture of Experts` `Text` `Benchmark`

**🎯 论文内容**

提出 KORA 通过在模型调用前进行确定性路由和验证来控制 AI 工作流，避免不必要的模型推理；

**💡 创新点**

创新点在于将模型调用边界显式化为决策点，先尝试确定性或结构化处理，再在必要时才调用模型；

**🔧 技术方法**

使用任务路径选择、确定性处理器、结构化查找、策略验证、模型升级等控制技术；

**📊 数据集**

使用自制的 100 题确定性占优基准和 12 题客户支持模拟请求作为测试数据；

**📈 对比分析**

与传统的直接模型调用基线对比，KORA 在基准任务中避免了 80% 的模拟模型调用，且确定性输出无错误；

**⚠️ 局限性**

局限性包括仅在合成工作负载上验证，未涉及真实模型调用、网络延迟、能耗、成本或生产环境性能。

---

## 218. Measuring Users' Security and Privacy Concerns About Third-Party Community Bots on Discord

**arXiv ID:** 2610.36626 | [PDF](https://arxiv.org/pdf/2610.36626v1)

**作者:** Kofi Ampomah `[一作]` (University of Alabama), Tu Le `[通讯]` (University of Alabama)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Knowledge Distillation` `Data-Centric Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Text` `Multimodality`

**🎯 论文内容**

本文研究了 Discord 生态中 183,142 条 Top.gg 用户评论，构建了 31 主题的安全与隐私（S&P）分类体系，识别并量化 S&P 相关讨论，并实现了基于 BotLens 的审计框架。

**💡 创新点**

创新点在于将 LLM 知识蒸馏与 DeBERTa 结合生成可扩展的 S&P 分类器；扩充 Android 词典为 31 主题的 Discord 专属 S&P 词表；提出 SPRR/TPR 指标和 BotLens 工具，首次将评论驱动的安全隐私信号转化为实时决策支持。

**🔧 技术方法**

使用 Llama 3 8B 进行标签生成、DeBERTa‑v3‑large 进行二分类、GoEmotions 进行情感归类、Argos Translate 处理多语种、MCP 服务器实现 BotLens，并利用 Python+Playwright 进行数据抓取。

**📊 数据集**

数据来源为 Top.gg 的 23,102 个 Discord 机器人 183,142 条用户评论（覆盖 49 种语言，2021‑2026 年），以及相应的权限、标签、服务器计数、GitHub 链接等元数据。

**📈 对比分析**

分类器在 30% 标注预算下实现 F1 0.938（AUC≈0.96）；SPRR 最高 4.82%，Data Sharing 30.5%；BotLens 22 次查询平均响应时间 <20 ms；实验表明不同标签/类别的 SPRR 与权限特征的相关性较弱。

**⚠️ 局限性**

局限性包括仅覆盖 Top.gg，评论自我选择可能偏高；LLM 翻译与标签误差导致噪声；未验证评论真实性；权限与功能关系的因果性不确定；模型对极短或混合语言评论表现不足。

---

## 219. Frugal Handwritten Text Recognition for Sub-Saharan Civil-Status Registers: A Smartphone-to-Records Pipeline with CPU-Only Inference

**arXiv ID:** 2610.36540 | [PDF](https://arxiv.org/pdf/2610.36540v1)

**作者:** Ousmane Hamadou `[一作]` (University of Ngaoundéré), Jean Louis Kedieng Ebongue Fendji `[通讯]` (University of Ngaoundéré)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Computational Efficiency` `Data-Centric Learning` `Convolutional Neural Network` `Recurrent Neural Network` `Transformer` `Supervised Fine-Tuning` `Contrastive Learning` `Vision Language Model` `Image` `Text`

**🎯 论文内容**

使用低成本手机捕捉、经典预处理和CPU推理的管线，对乍得市政登记簿的手写民事档案进行转录和信息抽取。

**💡 创新点**

提出一种完全基于消费级硬件、无需GPU、仅需少量标注即可在当地办公室部署的“节俭”手写文本识别方案。

**🔧 技术方法**

结合Kraken的CRNN‑CTC识别模型、手写行分割校正、CamemBERT‑NER及模板规则，实现端到端的识别与角色解析。

**📊 数据集**

构建了1,568张智能手机RAW图像的乍得出生、婚姻、死亡记录语料库，并提供185条行级转录及179条实体标注。

**📈 对比分析**

在已知行分割的测试集上，预训练模型CER为14.53%、WER为56.93%，微调后降至CER 0.50%、WER 3.37%；CPU上平均每张图像6秒，单机推理速率约10张/分钟；在下游角色解析中，精度从预训练的44.5%提升至81.2%，最优上限为90.3%。

**⚠️ 局限性**

实验仅覆盖单一注册处，测试集中写手重叠，缺乏自动行分割评估，未使用Transformer基线，且仅在免费GPU上完成微调，导致对其他地区或写手的泛化能力尚未验证。

---

## 220. LLM Hallucination Detection via Low-rank Multilinear Decomposition

**arXiv ID:** 2610.36191 | [PDF](https://arxiv.org/pdf/2610.36191v1)

**作者:** Devansh Sharma `[一作]` (University of Central Florida), Shahana Ibrahim `[通讯]` (University of Central Florida)

**关键词:** `Computation and Language` `Anomaly Detection` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Contrastive Learning` `Auto Encoder` `Random Forest` `Tucker Decomposition` `SVD` `Ridge Alignment` `Text` `Benchmark`

**🎯 论文内容**

开发了一种基于低秩多线性分解的LLM内部表示检测模型LOTUS，用于识别模型生成的幻觉回答。

**💡 创新点**

创新点在于使用多层、多特征的秩约束Tucker-2投影对变长token表示做统计摘要，既保持层-特征交互，又得到固定维度低秩表示，同时仅需少量标注数据即可训练轻量分类器，可跨模型迁移。

**🔧 技术方法**

使用统计量摘要（峰值、分位数、跨层差）、Tucker-2低秩分解、随机SVD、随机森林分类器、无监督投影、ridge对齐等技术。

**📊 数据集**

在 Qwen2.5-7B-Instruct、LLaMA-3.1-8B 两大 LLM 上，结合 TruthfulQA、TriviaQA、NQ-Open、TyDiQA-GP 四个 QA 基准。

**📈 对比分析**

与八个基线（无监督、监督）比较，LOTUS 在聚合 AUROC 上平均提升 7.1 点，超越 HARP、ACT‑ViT，且在模型间的对齐迁移保持误差 ≤6 点，内存占用远低于 ACT‑ViT。

**⚠️ 局限性**

局限在于仅评估问答场景，依赖自动度量标签，跨语言、跨任务迁移仍需验证；对极长文本的 token 统计窗口可能不足；需要对模型内部维度差异做更多实验。

---

## 221. Live-OPSD: On-Policy Self-Distillation for Proactive Streaming Large Video Models

**arXiv ID:** 2610.36959 | [PDF](https://arxiv.org/pdf/2610.36959v1)

**作者:** Kaixuan Du `[一作]` (Qwen Business Unit of Alibaba), Guanjun Jiang `[通讯]` (Qwen Business Unit of Alibaba)

**关键词:** `Machine Learning` `Knowledge Distillation` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Video` `Text` `Multimodality`

**🎯 论文内容**

本文提出了Live-OPSD，一种后训练方法，利用训练时特有的“特权提示”对流式多模态助手的自身生成进行自我蒸馏，以改进其响应时机。

**💡 创新点**

创新点在于：① 将特权提示与无提示模型对齐，生成教师分布；② 采用决策感知选择（DAS）动态分配训练样本，聚焦于决策不一致与一致的关键时刻；③ 将对策略自身输出的分布蒸馏与流式交互结合，形成一种端到端的自我监督训练框架。

**🔧 技术方法**

技术包括：对话式流式交互模型（如Qwen3-Omni-30B-A3B），逆Kullback–Leibler（reverse‑KL）损失作为蒸馏目标，决策感知选择算法，训练时的外部特权提示生成，超参数控制的更新预算。

**📊 数据集**

使用了公开的视频和多模态数据集，经过对话式训练数据构造，包含直接问题、持续指令等；评估采用LiveProBench、OVO‑Bench、OmniMMI等公开基准。

**📈 对比分析**

在同一模型体系下，Live-OPSD相较于继续SFT（Vanilla‑SFT、CB‑SFT、SSB‑SFT）在LiveProBench平均得分提升5.26点（从54.38至59.64），在OVO‑Bench整体准确率提升至65.84点；在OmniMMI主动警报任务中也实现了显著提升（例如30B模型从39.50%提升至59.00%）。

**⚠️ 局限性**

局限性包括：① 依赖离线特权提示，缺乏完全在线自监督；② 训练样本仍需人工标注的决策标签；③ 只在特定流式交互协议下验证，难以直接迁移到更复杂的动作空间；④ 对历史截断和模型生成历史的匹配不完全，可能影响推理阶段的性能。

---

## 222. Characterizing Privacy Leakage in Spiking Neural Networks: A Systematic Analysis of Temporal Dynamics, Stochasticity, Neuronal Dynamics and Learning Mechanisms

**arXiv ID:** 2610.36243 | [PDF](https://arxiv.org/pdf/2610.36243v1)

**作者:** Ayana Moshruba `[一作]` (George Mason University), Maryam Parsa `[通讯]` (George Mason University)

**关键词:** `Neural and Evolutionary Computing` `Safty and Privacy` `Spiking Neural Network` `Supervised Fine-Tuning` `Contrastive Learning` `Image` `Tabular`

**🎯 论文内容**

系统评估了生物学启发的时序动态（Rate、泊松突发、延迟突发）对脉冲神经网络（SNN）在成员推断攻击（MIA）中的隐私影响，并通过实验分离了随机性、神经元内在动力学、学习机制等因素的作用。

**💡 创新点**

创新点在于首次将时序结构、随机性、内在神经元动力学和学习算法的交互作用分离出来，证明SNN隐私并非单一结构属性，而是多因素共同决定的产物；同时展示了突发动态、AdEx神经元和进化学习对隐私与性能的不同影响。

**🔧 技术方法**

使用的技术包括Rate/Poisson‑Burst/Delayed‑Burst输入编码；LIF与Adaptive Exponential Integrate‑and‑Fire（AdEx）神经元模型；梯度反向传播（BPTT）与进化学习（EONS）；差分隐私训练（DPSGD）；shadow‑model MIA攻击框架与AUC评估。

**📊 数据集**

实验数据集包括MNIST、Fashion‑MNIST、CIFAR‑10、Iris以及Breast Cancer Wisconsin（诊断）等。

**📈 对比分析**

通过在相同网络架构和训练流程下切换时序动态、添加时间抖动、输入噪声、脉冲噪声或使用不同学习算法，对比测试准确率与MIA AUC，结果显示突发动态虽能降低泄露但牺牲准确率；AdEx模型在多数任务上既提升准确率又进一步降低泄露；进化学习在小型任务上可达到接近随机猜测的隐私水平。

**⚠️ 局限性**

限制在于仅覆盖有限数据集和网络规模；进化学习实验仅限于小型表格数据；未探究更复杂或实际攻击场景；缺乏理论解析，仅提供经验观察。

---

## 223. QESChunker: A Single Objective Unifies Overlapping and Non-Overlapping Chunking for RAG

**arXiv ID:** 2610.36557 | [PDF](https://arxiv.org/pdf/2610.36557v1)

**作者:** Yifan Zhao `[一作]` (Alibaba Group), Yuming Deng `[通讯]` (Alibaba Group)

**关键词:** `Information Retrieval` `Optimization` `Retrieval` `Data-Centric Learning` `Retrieval-Augmented Generation` `Prompt Engineering` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

提出了基于问题–证据监督的文档分块方法QESChunker，并提供两种求解器：非重叠分块的精确动态规划和允许重叠的贪婪子模优化；

**💡 创新点**

创新点在于将分块视为一个统一的覆盖+结构优先的优化问题，首次利用问题–证据对来直接指导分块边界；

**🔧 技术方法**

核心技术包括：问题–证据覆盖函数、分块结构先验权重、非重叠情形下的动态规划、重叠情形下的子模最大化与贪婪近似，以及懒惰评估与单例安全机制；

**📊 数据集**

实验数据集包含两份自行构建的中文政策报告（党代会报告、公共卫生白皮书）以及三份公开英文QA语料库（QASPER、PeerQA、PrivacyQA）；

**📈 对比分析**

与固定长度、自然段、TextTiling、ClusterSemanticChunker、LumberChunker、Sentence-Window、Parent-Document、Auto-Merging等基线对比；在非重叠设置下DP在9/10评估格点上领先，重叠设置下贪婪在所有10/10格点上领先，尤其在紧凑返回上下文预算时提升最显著；

**⚠️ 局限性**

局限性包括：需依赖足够且高质量的问答–证据监督，适用于查询频繁、已注释的文档；转换为最终生成质量的效果尚不明确；在冷启动或稀疏标注场景下性能可能主要由结构先验和检索器决定；

---

## 224. Do We Need an LLM to Know Why Sources Conflict? Decision Models for Knowledge-Conflict Typing in RAG

**arXiv ID:** 2610.36915 | [PDF](https://arxiv.org/pdf/2610.36915v1)

**作者:** Parv Khatri `[一作]`, Tanya Sah `[通讯]`

**关键词:** `Computation and Language` `Classification` `Explainability and Interpretability` `Computational Efficiency` `Retrieval` `Transformer` `Retrieval-Augmented Generation` `Prompt Engineering` `Large Language Model` `Text` `Benchmark`

**🎯 论文内容**

评估并比较所有托管的typed decision模型在检索增强语言模型中识别信息冲突类型的效果，并与Gemini等大型语言模型进行成本、速度与准确率的对比；

**💡 创新点**

首次将typed decision模型用于多来源冲突分类任务，展示其在准确率、低成本和低延迟上的竞争力，并引入基于置信度的分层推理（cascade）方法；

**🔧 技术方法**

使用typed decision模型（如1.13、Decider、GPT‑6 Luna Decisions等）、Gemini 2.5 Flash、Gemini 3.8 Flash等LLM，以及统一的提示和树状问题拆分；

**📊 数据集**

使用公开的Conflict‑Typing benchmark（458条检索查询，5类冲突标签）以及构造的植入误信息样例；

**📈 对比分析**

在相同输入和提示下进行配对比较，结果显示多数决策模型在准确率上与原论文的Gemini 2.5 Flash相当或更好，成本约为其十分之一、延迟约为其十分之一；置信度分层可将大多数差距恢复至Gemini 3.8 Flash的四分之一成本；

**⚠️ 局限性**

局限包括样本量有限（尤其误信息类仅5条真实样本）、模型版本变动快、对来源顺序敏感性差异大、对单一误信息类型的检测仅通过概率而非标注，且最终答案质量评估未考虑检索结果的直接可见性。

---

## 225. Dense Representation Learning and Differentiable Shape Modeling for Spine Landmark Detection

**arXiv ID:** 2610.36951 | [PDF](https://arxiv.org/pdf/2610.36951v1)

**作者:** Augustin Daridon `[一作]`, Loic Le Folgoc `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Pose Estimation` `Representation Learning` `Optimization` `Convolutional Neural Network` `Graph Neural Network` `Contrastive Learning` `Diffusion model` `Score-based Model` `Optical Flow` `Stochastic Differential Equation` `Image` `Biomedical Data` `Computed Tomography` `Magnetic Resonance Imaging` `Review/Survey Paper`

**🎯 论文内容**

提出一种基于像素级对比学习的解剖标记点检测框架（Pix2Point），并结合可微形状先验和图搜索实现脊柱标记点的精细优化。

**💡 创新点**

创新点：①将标记点检测视作像素-原型对比学习，生成自由形状热图；②使用可微形状先验与热图结合的能量模型，实现子像素级定位；③利用图搜索的离散优化（GDO）快速逼近全局最优。

**🔧 技术方法**

技术：U‑Net 编码器-解码器，像素级对比损失，预训练的自监督嵌入，PCA 形状先验，LBFGS 可微优化，Dijkstra 图搜索。

**📊 数据集**

数据集：公开 AASCE（424/47/126 训练/验证/测试），私有双视角脊柱 X‑ray（Coronal 与 Sagittal，各 394/44/134 训练/验证/测试）共 93 个标记点。

**📈 对比分析**

与基线（Gaussian 热图 + MSE、CARN、SpatialConfiguration‑Net、nnLandmark、Vertebra‑focused、SLSN、Interactive Keypoint）对比，在 AASCE、Coronal 与 Sagittal 公开/私有测试集上均取得更低的 MRE、较高的 SDR 与临床参数排名，性能与最先进方法相当或更佳。

**⚠️ 局限性**

局限：仍受相邻椎体视觉相似导致的局部极大值影响；全局优化需依赖 GDO 初始；方法目前仅针对脊柱标记点，难以直接推广至其他解剖结构。

---

## 226. Language Models Keep Abandoned Analyses: Residue and Adaptation after Garden-Path Sentences

**arXiv ID:** 2610.36797 | [PDF](https://arxiv.org/pdf/2610.36797v1)

**作者:** Xiang Ding `[一作]` (Nagoya University), Xiang Ding `[通讯]` (Nagoya University)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Representation Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Activation Patching` `Linear Probe` `Role Direction Intervention` `Text`

**🎯 论文内容**

研究语言模型在园地句（garden‑path）后保留被放弃的句法分析对后续文本的影响，并通过激活补丁等方法定位残留与适应的来源。

**💡 创新点**

发现模型同时保留放弃的分析和修正后的分析，残留主要来自名词短语的早期状态，而适应则来源于后续状态，并将这两种效应与残余激活和误差驱动学习对照。

**🔧 技术方法**

使用24个公开权重因果 Transformer（GPT‑2、Pythia、Llama、Mistral、Gemma、Qwen 等）进行对数概率测评、采样、激活补丁、线性探针和角色方向干预等技术。

**📊 数据集**

采用多种园地句集合（NP/Z、NP/S、主谓‑修饰语等）和目标片段集，包括 Discovery、Held‑out、SyntaxGym 等，约93+48+42 个项目。

**📈 对比分析**

通过定义残留对比 A_X=(X_gp–X_nb)–(X_cue–X_nc) 消除标点和动词偏差，量化生产类与理解类的残留与适应，发现残留在生产类约占目标效果的 76%，适应效应随句法错误大小成比例，且与模型规模相关。

**⚠️ 局限性**

局限性包括仅研究英语 NP/Z 等结构，未覆盖其他语言或更大模型；仅使用公开权重，未评估闭源模型；补丁实验覆盖模型有限；部分预设预测未通过；采样生成判定基于人工规则。

---

## 227. SVIn+: Multi-Modal Framework for Robust Underwater State Estimation using 3D Sonar, Visual-Inertial, Water-Pressure, and DVL

**arXiv ID:** 2610.36454 | [PDF](https://arxiv.org/pdf/2610.36454v1)

**作者:** Chinmay Burgul `[一作]` (University of Delaware), Alberto Quattrini Li `[通讯]` (Dartmouth College)

**关键词:** `Robotics` `Optimization` `Robotic Intelligence` `Autonomous Driving` `Simultaneous Localization and Mapping` `Factor Graph Nonlinear Optimization` `Sliding-Window Framework` `IMU Preintegration` `KISS-ICP` `Depth Measurement Constraint` `DVL Speed Constraint` `Point Cloud` `Multimodality` `Image` `Video` `Time Series`

**🎯 论文内容**

提出了一种名为SVIn+的多模态状态估计框架，将视觉‑惯性、3D声呐、深度传感器和DVL的测量融合在非线性优化滑动窗口中实现高频且鲁棒的定位；

**💡 创新点**

创新点在于：1）采用KISS‑ICP前端生成的6D相对位姿因子，既保留声呐的结构信息，又保持因子图稀疏性，实现在嵌入式硬件上的实时性能；2）将多传感器的约束紧耦合到同一因子图中，显著降低漂移并提升全局一致性；3）在复杂水下环境（船体、珊瑚礁）中进行实地验证，展示了比传统视觉‑惯性或单独传感器方案更优的表现。

**🔧 技术方法**

主要技术包括：SVIn视觉‑惯性SLAM、IMU预积分、深度测量约束、DVL速度约束、3D声呐扫描配准(KISS‑ICP)、因子图非线性优化、滑动窗口（Sliding‑Window）框架；

**📊 数据集**

使用了Aqua2平台收集的三个数据集：BajanQueen‑01（船体内部约75 m）、BajanQueen‑02（船体全程约600 m）、Folkestone‑01（珊瑚礁约280 m），每个数据集均包含立体视觉、DVL、3D声呐和压力计同步测量；

**📈 对比分析**

通过回到原点误差、ArUco标定以及与SVIn2、KISS‑ICP、离线SfM等基线的比较，SVIn+在BajanQueen‑01的返回原点误差仅为0.21 m，明显优于SVIn2（0.53 m）、单独深度（0.29 m）和单独DVL（0.52 m）。在长距离数据集上也保持低漂移，并在点云重建与视觉特征对齐上表现更好。

**⚠️ 局限性**

局限性包括：1）对声呐稀疏或失配情况较为敏感，需更鲁棒的配准或自适应切换机制；2）尚未实现全局闭环与循环检测，依赖外部回环或基线闭环；3）缺乏在线外部标定，系统对动态传感器失准的抵抗力有限；4）实验仅覆盖两种典型环境，未覆盖极端浑浊或深海高压等极端场景。

---

## 228. Follow the Herd: A Data-Driven Approach for Actionable Recourse

**arXiv ID:** 2610.36580 | [PDF](https://arxiv.org/pdf/2610.36580v1)

**作者:** Nishtha Madaan `[一作]` (Indian Institute of Technology Delhi), Srikanta Bedathur `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `Artificial Intelligence` `Optimization` `Recommendation System` `Data-Centric Learning` `Transformer` `Diffusion model` `Gradient guided optimization` `Tabular` `Time Series` `Sequential` `Finance Related`

**🎯 论文内容**

提出了Data‑Driven Action Recourse（DDAR）框架，利用无监督的历史用户轨迹通过扩散模型生成可行且低成本的行动递推方案；

**💡 创新点**

首次通过数据驱动方式自动学习行动成本，消除人工定义成本函数，并将最可能的轨迹视为低成本路径；

**🔧 技术方法**

采用条件扩散模型（guided diffusion）与Transformer denoiser、梯度引导的优化搜索技术实现用户状态预测与递推生成；

**📊 数据集**

在三组合成/半实数据集——LoanSynth、GradSynth和AdultIncome++上进行实验；

**📈 对比分析**

与Wachter、DiCE、FACE、SCD等基线在Recourse Rate（RR‑AUC）、成本、CNLL/UNLL等指标上比较，DDAR在递推率更高、成本更低且可行性更好；

**⚠️ 局限性**

依赖历史数据分布，难以覆盖极端稀有场景；缺乏对多样性与长期约束的理论保障；实验主要在模拟数据上，缺乏真实世界验证。

---

## 229. CoHARL: Co-Evolving Agent Policies and Executable Harnesses with Reinforcement Learning

**arXiv ID:** 2610.36469 | [PDF](https://arxiv.org/pdf/2610.36469v1)

**作者:** Jianwen Chen `[一作]` (University of North Carolina at Chapel Hill), Huaxiu Yao `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `Machine Learning` `Reinforcement Learning` `AI Code Assistant` `Transformer` `Large Language Model` `Reinforcement Learning` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text`

**🎯 论文内容**

设计并实现了一种同时演化共享策略和可执行工具箱（harness）族的联合强化学习框架；

**💡 创新点**

创新点在于通过人口级调度与基于执行轨迹的程序修改器（proposer）实现策略与接口的协同进化，提升了软件工程和终端任务的学习效率；

**🔧 技术方法**

使用了大模型（Qwen3.5‑35B）+ LoRA、GRPO（group‑relative advantage）强化学习、基于 LLM 的程序编辑、基于置信下界的调度器以及历史精度‑成本档案；

**📊 数据集**

主要数据集包括 SWE‑rebench、SWE‑bench Verified 以及 Terminal‑Bench 2.0；

**📈 对比分析**

与单独的“harness‑only search”（Meta‑Harness）和“policy‑only RL”（Plain GRPO）以及两种替代策略（search‑then‑RL、单 harness 交替 RL）进行对比，平均 resolved rate 由 21.2%/20.3% 提升至 27.6%（SWE‑rebench），在 SWE‑bench Verified 上从 52% 提升至 59.8%，在 Terminal‑Bench 2.0 上从 33.7% 提升至 36.8%；

**⚠️ 局限性**

局限性包括对隔离执行环境的依赖、潜在的安全风险、对 LLM 编辑器的性能与可靠性依赖、对不同任务域的迁移性不确定以及实验资源消耗较大。

---

## 230. Deep Learning Volumetric Segmentation for Prediction of Cavity Geometry in Underground Jet Boring Mining

**arXiv ID:** 2610.36303 | [PDF](https://arxiv.org/pdf/2610.36303v1)

**作者:** Giovanni Zingaro `[一作]` (University of Waterloo), Robert Gracie `[通讯]` (University of Waterloo)

**关键词:** `Computational Engineering, Finance, and Science` `Segmentation` `Explainability and Interpretability` `Convolutional Neural Network` `Diffusion model` `Contrastive Learning` `Mixture of Experts` `Point Cloud` `Multimodality` `Tabular` `Mining Related`

**🎯 论文内容**

开发了一种基于三维U‑Net的深度学习模型，用于预测加拿大西北部Cigar Lake矿区Jet Boring System（JBS）切削后洞穴的体积几何形状。

**💡 创新点**

创新点包括：①将多模态勘探、钻探和切削记录与地质块模型统一投影到体素网格并构造特征通道；②在U‑Net中加入背填残差分支以保留重要的背填信息；③使用联合IoU+交叉熵损失与集成模型提升泛化；④采用集成的Integrated Gradients实现可解释性。

**🔧 技术方法**

核心技术为三维卷积编码‑解码网络（U‑Net）+残差分支，配合数据增强、温度标定、复合损失函数、20个模型的集成以及可解释性方法Integrated Gradients。

**📊 数据集**

使用Cigar Lake矿区的历史切削洞穴点云（C‑ALS扫描）、岩石块模型、钻探与切削参数记录等多源数据，构成约20个带有真实洞穴体素标记的样本。

**📈 对比分析**

与传统的专家设定圆柱洞穴基线相比，模型在测试集上IoU提升22.4%（0.601 vs 0.491），F1提升14.7%（0.747 vs 0.651），并将平均绝对U3O8吨位误差从38%降至25.6%，模型在训练、验证和测试集表现相近，表明过拟合较少。

**⚠️ 局限性**

主要局限是样本量有限且主要集中在已背填洞穴，导致模型在未背填（原始岩体）的小洞穴上表现欠佳，且对岩体不确定性和测量误差的处理仍不充分。

---

## 231. The Inbox as a Coordination Primitive for LLM Agent Societies

**arXiv ID:** 2610.36264 | [PDF](https://arxiv.org/pdf/2610.36264v1)

**作者:** Karthik Sreedhar `[一作]` (MIT Media Lab), Ramesh Raskar `[通讯]` (MIT Media Lab)

**关键词:** `Multiagent Systems` `Reinforcement Learning from Human Feedback` `Optimization` `Autonomous Driving` `Federated Learning` `Explainability and Interpretability` `Computational Efficiency` `Robotic Intelligence` `Meta Learning` `Drug Discovery` `AI Code Assistant` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Graph` `Review/Survey Paper`

**🎯 论文内容**

本文研究了大规模语言模型代理社会中的收件箱(Inbox)机制，将消息处理从传统的推送(Push)切换到拉取(Pull)，并将其视为一种协调原语；

**💡 创新点**

创新点在于提出并系统评估收件箱的四个核心决策——读取规则、记忆规则、支付优先级以及回复内容，并通过大规模仿真证明在高负载、噪声与多跳转发情境下，收件箱是实现有效协作的必要条件；

**🔧 技术方法**

技术方法主要使用大型语言模型（如gpt‑5.6‑terra、gpt‑5.4‑mini 等）进行决策判断，结合自定义的推送/拉取协议、可退款存款优先机制、消息过滤与优先级排序等；

**📊 数据集**

数据集涵盖多规模代理社群（1k–16k）以及基于 SNAP 联系人图的网络，使用模拟的会议调度与组建任务数据，并对兴趣匹配、转发路径等进行人工构造；

**📈 对比分析**

比较方法：在会议调度和组建任务上对比传统Push、不同读取/记忆/支付/回复规则，记录提案重复率、成功召回率、组建成功率及token成本；实验结果显示Push在高并发下错误率激增，收件箱规则在60%–70%召回率下显著降低token使用；

**⚠️ 局限性**

局限性包括：仅评估固定规则而未考虑自适应学习；使用单一模型服务商，可能不具备普适性；仿真假设所有代理共享相同模型判断，未考虑多模型差异；仅针对两类任务，缺乏对更广泛协调场景的验证。

---

## 232. Spiking Neural Networks Are Harder to Steal: A Benchmark Study of Model Stealing Attacks

**arXiv ID:** 2610.36298 | [PDF](https://arxiv.org/pdf/2610.36298v1)

**作者:** Ayana Moshruba `[一作]` (George Mason University), Maryam Parsa `[通讯]` (George Mason University)

**关键词:** `Artificial Intelligence` `Adversarial Attack` `Safty and Privacy` `Spiking Neural Network` `Convolutional Neural Network` `Adversarial Attack` `Image` `Benchmark`

**🎯 论文内容**

评估了四种模型窃取攻击在脉冲神经网络（SNN）和传统人工神经网络（ANN）上的可行性，并比较了它们的提取效果。

**💡 创新点**

证明SNN因其稀疏事件驱动和时间动态特性，在黑盒查询下更难被窃取，即使使用同类SNN替代模型，提取效率仍低于ANN。

**🔧 技术方法**

对Papernot、Tramèr、T‑RND和颜色扰动等四种模型窃取策略进行实验，使用测试一致性、随机一致性、非定向/定向对抗转移率和查询效率评估。

**📊 数据集**

MNIST、Fashion‑MNIST（FMNIST）和CIFAR‑10。

**📈 对比分析**

通过一致性、对抗转移率和查询量三维度进行对比；结果显示SNN目标的替代一致性低10–20%，对抗转移率低15–25%，且在相同查询预算下收敛更慢。

**⚠️ 局限性**

仅考虑了上述四种攻击且所有替代模型均为ANN（部分实验包含SNN替代），未探讨针对SNN专门设计的更高效窃取方法；实验受限于所选神经元模型和编码方案，可能不适用于所有SNN架构。

---

## 233. SG^2SLAM: Spatial Geometry-Informed 3D Gaussian Splatting SLAM for Monocular Outdoor Navigation

**arXiv ID:** 2610.36312 | [PDF](https://arxiv.org/pdf/2610.36312v1)

**作者:** Ziang Guo `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `Robotics` `Autonomous Driving` `Optimization` `Convolutional Neural Network` `Gaussian Splatting` `Simultaneous Localization and Mapping` `Depth Estimation` `Bundle-Adjustment` `Optical Flow` `Point Cloud` `Video`

**🎯 论文内容**

提出 SG^2SLAM，利用流式空间几何先验和光线条件化 3D 高斯分布核，在单目室外导航中实现 3D‑3D 联合关联的实时 SLAM。

**💡 创新点**

创新点在于：①用块级空间几何先验替代逐帧 2D‑3D 匹配，消除累积误差；②引入光线条件化 3DGS 核，使原始高斯在视角变换下保持几何一致；③通过 Sim(3) 全局图优化一次性融合所有块，解决尺度漂移。

**🔧 技术方法**

主要技术包括：3D Gaussian Splatting、基于卷积网络的深度估计、块级流式前端、光线条件化卷积核、光度损失+深度监督的 Bundle‑Adjustment、Sim(3) 归一化闭式匹配、链式与环路约束的全局优化。

**📊 数据集**

使用 KITTI odometry、Waymo Open Dataset 以及自建带 RTK‑GNSS/IMU/LiDAR 的真实驾驶序列进行评估。

**📈 对比分析**

与 MonoGS、S3PO‑GS、DiskChunGS、GigaSLAM 等基线对比，SG^2SLAM 在 KITTI 上平均 ATE 为 6.24 m、在 Waymo 上 0.63 m，渲染质量（PSNR 22.7 dB、SSIM 0.86、LPIPS 0.41）明显优于对手。

**⚠️ 局限性**

局限性包括：需要深度先验与块大小调节，后端计算量大且依赖高端 GPU；在极端光照、雨夜或大动态场景下的鲁棒性尚待提升。

---

## 234. Haste Makes Waste? Towards Safe Few-Step Text-to-Image Models

**arXiv ID:** 2610.36665 | [PDF](https://arxiv.org/pdf/2610.36665v1)

**作者:** Lior Ziv `[一作]` (Tel Aviv University), Mahmood Sharif `[通讯]` (Tel Aviv University)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Safty and Privacy` `Knowledge Distillation` `Transformer` `Diffusion model` `Score-based Model` `Rectified Flow` `Contrastive Learning` `Supervised Fine-Tuning` `Image` `Text` `Multimodality`

**🎯 论文内容**

研究了文本到图像模型在经过少步蒸馏加速后对安全性的影响，并提出专门针对少步模型的概念消除方法 DICE。

**💡 创新点**

发现速度与安全存在张力，揭示少步模型易导致概念消除失败的两条机制，并设计 DICE 通过直线路径重噪和自蒸馏保持来实现更安全的概念消除。

**🔧 技术方法**

使用流匹配（rectified‑flow）框架、LoRA 微调、ESD、EraseAnything、UCE 等概念消除技术，结合自蒸馏和离线噪声构造。

**📊 数据集**

在 NSFW‑56k、I2P、LAION‑Pop、COCO‑30k 等公开数据集上评估安全性、质量和对抗攻击。

**📈 对比分析**

通过 NudeNet、SD‑Checker（含 @5 版本）和 HPSv2 等指标对比，DICE 在三款少步模型上将不安全率降至 0.03–0.05 左右，保持 HPSv2 与未防护模型相近，优于 ESD、EraseAnything、UCE 等现有方法。

**⚠️ 局限性**

DICE 并非万无一失，仍在对抗提示下产生一定不安全输出；实验依赖自动检测器，缺乏人工评估；未完全解释导致消除失败的因果机制。

---

## 235. Log-Derivative Trick for Parametric Markov Chains

**arXiv ID:** 2610.36655 | [PDF](https://arxiv.org/pdf/2610.36655v1)

**作者:** Arthur Gall `[一作]` (National Institute of Informatics), Thomas Seiller `[通讯]` (CNRS)

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning` `Monte Carlo Sampling` `Gradient Descent` `Stochastic Differential Equation` `Tabular` `Graph` `Benchmark`

**🎯 论文内容**

研究终止参数化马尔可夫链（pMC）的期望奖励和到达概率优化，提出基于轨迹采样的无偏梯度估计器。

**💡 创新点**

通过测度论推广 log‑derivative trick，得到在任意可测状态空间上无偏且不需要全局状态空间材料化的梯度采样方法。

**🔧 技术方法**

采用 Radon–Nikodym 导数、Monte Carlo 采样、梯度下降（如 AdamW、SGD）以及测度论框架进行实现。

**📊 数据集**

在合成离散 pMC（5k 状态、100 参数）和连续状态案例（T₃、T₄）上实验，使用 PRISM 生成的模型数据集。

**📈 对比分析**

与传统线性求解器、Jacobi 方法、重要采样基线等对比，采样估计在相同时间内获得更高奖励改进，误差极低且速度提升数十倍。

**⚠️ 局限性**

受采样方差影响，尤其在样本量极小时收敛不稳；循环路径长度大时需要更长采样；并未提供全局最优保证。

---

## 236. SplatScout: Query-Driven Part-Level Referring Segmentation in Gaussian Scenes

**arXiv ID:** 2610.37098 | [PDF](https://arxiv.org/pdf/2610.37098v1)

**作者:** Yihan Zang `[一作]` (King Abdullah University of Science and Technology), Dominik Engel `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Transformer` `Vision Language Model` `Gaussian Splatting` `Agentic AI` `Prompt Engineering` `Image` `Point Cloud` `Text` `Multimodality`

**🎯 论文内容**

提出了一种查询驱动的闭环感知框架（SplatScout），让视觉‑语言模型（VLM）主动选择观察、分割与 3D 实例构造，并通过多轮视觉反馈迭代优化目标；

**💡 创新点**

核心创新在于将 VLM 的决策从仅选取预生成候选转移到完整控制感知流程（Navigator、Segmenter、Lifter），实现动态、迭代的闭环反馈，解决了细粒度部件指代中固定候选不足的问题；

**🔧 技术方法**

技术包括 2D Gaussian Splatting 场景表示、SAM 进行查询导向分割、FlashSplat 及投影投票实现 2D → 3D Gaussian lifting、基于 Qwen3.5‑397B 或 GPT‑5.6 的 VLM 控制器以及多视角渲染/反馈机制；

**📊 数据集**

使用了自建的 PartRef‑ScanNet++ 数据集（261 条手工验证的查询，覆盖 12 个 ScanNet++ 场景、201 个物体与 95 个部件），并在 LERF、Causal‑LERF、GR‑LERF、Causal‑ScanNet、GR‑ScanNet 等公开基准上进行评测；

**📈 对比分析**

与基线（如 GaussDet、OpenGround、ZeroSplat）相比，SplatScout 在 PartRef‑ScanNet++ 上 mIoU 达到 0.538，远超 0.253，且在 4/5 公开基准上取得更高分数；

**⚠️ 局限性**

局限性包括：对 VLM 的强大依赖（性能随模型规模波动显著）、对部分基准的分辨率不匹配导致较差表现、以及对大量视角与迭代计算的资源需求。

---

## 237. CCP-DP: Cross-cluster context-guided phase dynamic programming for DNA storage sequence reconstruction

**arXiv ID:** 2610.36559 | [PDF](https://arxiv.org/pdf/2610.36559v1)

**作者:** Zixin Zhang `[一作]` (Tianjin University), Huaming Wu `[通讯]` (Tianjin University)

**关键词:** `Information Retrieval` `Data-Centric Learning` `Computational Efficiency` `Representation Learning` `Contrastive Learning` `Auto Encoder` `Reinforcement Learning` `Biomedical Data`

**🎯 论文内容**

提出一种基于跨簇上下文引导的相位动态规划（CCP‑DP）方法，用于在低覆盖率、强插入/缺失/替换噪声下重建 DNA 存储序列。

**💡 创新点**

创新点在于：①利用同一数据集内不同读簇共享的序列上下文信息（Cross‑LOO），为局部簇内证据不足的情况提供补充；②结合盲 IDS 统计估计和异常评分，自动决定是否对某些簇进行结构化相位动态规划；③通过跨簇上下文权重调节和局部/跨簇证据的协同打分，显著提升低覆盖率下的重建准确率。

**🔧 技术方法**

技术核心包括：跨簇 k‑mer 统计、留一交叉上下文概率模型、基于 k‑mer 重叠和 band‑align 的临时一致体构建、盲 IDS 参数估计、长度约束的相位动态规划（Phase‑DP）以及自适应路由策略。

**📊 数据集**

实验数据集包括 Erlich‑152、Srinivas‑110、Chandak‑108 和 Chandak‑150 四个公开 DNA 存储测序数据集，覆盖 110–152 bp 长度，覆盖率从 5×到 20×不等。

**📈 对比分析**

与 MUSCLE、ITR、CPL、BBS、TrellisBMA、DNAFormer、TReconLM 等基线方法比较，CCP‑DP 在所有数据集上实现 95.36%–99.98% 的 exact reconstruction 率；在 5×覆盖率下，较 BBS/CPL 的提升达 6.2–17.5%，且平均运行时间仅为 1–7 秒，显著低于传统方法。

**⚠️ 局限性**

局限性：对跨簇上下文重用程度高度依赖，若数据集内的序列模式不一致或难簇比例过高，则跨簇信息贡献有限；在极高插入/缺失率（>10%）下，仍可能出现负收益；此外，盲 IDS 估计假设错误分布相对均匀，若实际误差分布不符合，性能会受限。

---

## 238. kw-bench: Breaking Frontier Harnesses on Real-World Tasks

**arXiv ID:** 2610.36405 | [PDF](https://arxiv.org/pdf/2610.36405v1)

**作者:** George Juraj Salapa `[一作]` (G.S. Limited Liability Company), George Juraj Salapa `[通讯]` (G.S. Limited Liability Company)

**关键词:** `Artificial Intelligence` `Benchmark` `Finance Related` `AI Code Assistant` `Transformer` `Agentic AI` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Tabular` `PDF/CSV` `Sequential`

**🎯 论文内容**

提出kw-bench，一个模拟为期数周的调查性知识工作（如五年增值税重建）的基准，并展示前沿模型与传统模型在该基准上的性能差异。

**💡 创新点**

创新点在于：1）将真实业务流程、三代SAP系统导出、隐藏答案与陷阱等完整模拟成基准；2）使用硬编码、无LLM裁判的确定性评分；3）系统性对比不同agent harness在同一模型下的表现，揭示harness决定性作用；4）展示前沿模型在短时间内完成任务的可行性。

**🔧 技术方法**

使用的技术包括：终端式coding agent harness（OpenCode、micro‑cc）、Claude Code、Codex CLI、Harbor评估框架、Python脚本、PDF/CSV解析、税务规则引擎、工具调用、上下文管理与记忆迁移。

**📊 数据集**

使用的数据集为：基于真实税务专家案例构造的合成SAP提取数据（包含ECC、S/4HANA、咨询重建三代）、隐藏答案文件、税务申报文件、模板与公司品牌文件，随机种子可复现。

**📈 对比分析**

方法是：在相同模型（DeepSeek v4.1 Flash）下，对四个harness分别跑一次任务，并记录分数、墙钟时间、调用次数、费用；随后在前沿模型（Claude Opus 5.5）下验证同一harness的效果。结果显示：在前沿模型下两小时内完成任务，得分≈1，费用≈6美元；在低阶模型下，独立harness得分≈0.5，实验harness得分0，说明harness是决定性因素。

**⚠️ 局限性**

局限性包括：仅一次跑测，缺乏统计显著性；只测试了一个低阶模型与两个前沿模型；实验harness需额外配置以兼容非本地模型；合成数据虽然逼真但仍简化了业务细节；确定性评分无法体现专家洞察；仅在无人干预的headless场景下评估。

---

## 239. Cross-modal Steering Vectors Exist Across Both Image and Text Generation

**arXiv ID:** 2610.36269 | [PDF](https://arxiv.org/pdf/2610.36269v1)

**作者:** Donghao Zeng `[一作]` (Mila, McGill University), Benno Krojer `[通讯]` (Mila, McGill University)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Explainability and Interpretability` `Transformer` `Vision Language Model` `Contrastive Learning` `Prompt Engineering` `Large Language Model` `Diffusion model` `Image` `Text` `Multimodality`

**🎯 论文内容**

研究了统一多模态模型中跨模态的 steering 向量，检验了从一种模态提取的 steering 向量是否能在另一模态上零样本地控制生成。

**💡 创新点**

证明了 steering 向量在语言和图像之间可互转，发现跨模态 steering 具有几乎相同效果，并揭示了向量几何相似性与转移成功率的关联。

**🔧 技术方法**

使用了最小对差集（contrastive pair）来提取线性 steering 向量，在三种统一模型（UniAR、Emu3.5、Liquid）上进行层范围内插值并评估；采用 Gemini 3.7 Flash 作为 LLM 判别器。

**📊 数据集**

构造了 8 个概念（情感、年龄、清洁度、混乱度、尺寸、深度、空间方向、颜色）的最小对差集图像和文本对，使用 Gemini 生成与编辑；评估使用 20 条中性场景描述。

**📈 对比分析**

通过 LLM 判别器计算概念率，取 AUC 并比较转移向量的恢复率，结果显示文本到图像的转移恢复率平均 0.84–1.20，图像到文本平均 0.74–0.96，且对情感、尺寸等概念效果最佳。

**⚠️ 局限性**

模型间差异显著，某些概念（如空间方向）转移失败；向量在不同模型层级的效果不稳定；评估依赖单一 LLM 判别器，可能受其偏差影响。

---

## 240. Forget the Data, Keep the Capability: Switchable Unlearning for Retrieval

**arXiv ID:** 2610.36858 | [PDF](https://arxiv.org/pdf/2610.36858v1)

**作者:** Jingong Chen `[一作]` (South Dakota State University), Kwanghee Won `[通讯]` (South Dakota State University)

**关键词:** `Computer Vision and Pattern Recognition` `Retrieval` `Transformer` `Contrastive Learning` `LoRA 低秩补丁` `Gradient Isolation` `Parameter Consolidation` `Membership Inference 攻击评估` `Forgetting-Rank 审计` `Image`

**🎯 论文内容**

提出 MoME 框架，将预训练模型拆分为 Core 与 Parasitic 两个可切换模式，实现对删除请求的即时响应。

**💡 创新点**

创新点在于：① 利用梯度隔离将忘记样本的梯度与 Core 参数分离；② 通过 Parasite 参数实现全数据知识与 Core 的解耦；③ 引入 Forgetting‑Rank 统计量对训练后的忘记效果进行可量化审计。

**🔧 技术方法**

技术细节包括：LayerNorm/Bias 级别的零初始化 Parasite 参数、LoRA 低秩补丁、两步梯度更新、参数 Consolidation、BatchNorm 统计冻结、Membership Inference 攻击评估和 Forgetting‑Rank 审计。

**📊 数据集**

实验使用人车 ReID 数据集 Market‑1501、VeRi‑776、RegDB 以及分类数据集 CIFAR‑10/100、TinyImageNet，所有数据均采用标准划分与稀疏保留集。

**📈 对比分析**

与原始模型、Fine‑tune、GA、NegGrad+、SalUn、Bad Teacher、LoRA、SSD、SCRUB、LoTUS、MUNBa、LOFT、Retrain、Core‑only 等方法对比，MoME‑Core 在检索准确率上与 retrained reference 相当并且攻击 AUC 与 Forgetting‑Rank 接近零点；MoME‑Full 则恢复完整性能，证明切换仅需翻转标志即可完成删除与恢复。

**⚠️ 局限性**

局限性包括：仅适用于带有可增量 affine 参数的网络（如 LayerNorm/BatchNorm）；需要冻结 BatchNorm 统计才能保证安全；Core 中已学习的知识无法被删除；对非 ViT/ResNet 等骨干的迁移效果有限；在保留集过大或全量删除时可能出现容量与隐私的权衡。

---

## 241. USB Differentiated Service for Networked Cyber-Physical Systems

**arXiv ID:** 2610.36436 | [PDF](https://arxiv.org/pdf/2610.36436v1)

**作者:** Anton Njavro `[一作]` (Boston University), Richard West `[通讯]` (Boston University)

**关键词:** `Operating Systems` `Benchmark`

**🎯 论文内容**

设计并实现了支持 USB 3.x 的实时 xDCI gadget 驱动，实现主机间与主机到设备的双向通信，并通过多路 USB 中断器实现不同流的分流服务。

**💡 创新点**

首次将 USB 3.x 的时隙化等时等时机 (isochronous) 与大容量 (bulk) 传输在同一条 USB 总线上同时进行，并通过 Quest RTOS 的 I/O VCPU 和多中断器实现每个流的时延与吞吐分离，解决传统 USB 仅提供最佳努力服务的问题。

**🔧 技术方法**

使用了 USB 3.x xDCI 与 xHCI 接口、USB 中断器（MSI）、Quest RTOS 的 Sporadic/PIBS 服务器、双缓冲最新包存储、基于事件环的完成处理以及动态中断器分配等技术。

**📊 数据集**

实验基准为 1024 字节数据包的连续 Isochronous 与 Bulk 流，使用两台配备 Intel Raptor Lake SoC 的 SuperMicro X13SRN-H 开发板，通过 USB‑C‑USB‑A 线连接，运行 Quest RTOS。

**📈 对比分析**

评估结果显示：最大可达 ~1 Gbps 的 Isochronous 传输（理论上限 1048.6 Mbps，实际 98‑99%），对 125 µs 与 2 ms 交错流进行时延测量，平均 IAT 与最大值均保持在目标值内，且相较于单中断器的 Linux 环境，差异化服务显著降低了时延抖动；动态切换服务间隔时仍能保持时延稳定。

**⚠️ 局限性**

局限性包括：设备端仅实现单个中断器（无法对多设备流进一步隔离）、实验仅覆盖 USB 3.x 与两台主机，未探讨更复杂的拓扑；缺少对错误恢复、丢包率与多设备场景的评估；实现依赖 Quest RTOS，需迁移至其他系统时需要额外工作。

---

## 242. 4D Reconstruction with Events & RGB

**arXiv ID:** 2610.36610 | [PDF](https://arxiv.org/pdf/2610.36610v1)

**作者:** Dulanga Weerakoon `[一作]` (Singapore-MIT Alliance for Research & Technology Centre), Paul Pu Liang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Depth Estimation` `Pose Estimation` `Autonomous Driving` `Transformer` `Spiking Neural Network` `Neural Radiance Field` `Optical Flow` `Contrastive Learning` `Gaussian Splatting` `Image` `Video` `Point Cloud` `Multimodality`

**🎯 论文内容**

通过在MonST3R模型中添加事件摄像机输入分支，实现了RGB+事件的4D重建。

**💡 创新点**

创新点在于最小化的事件-RGB融合方法，即仅通过拼接事件体素与RGB特征，无需完整架构改造，提升了在高运动场景下的重建精度。

**🔧 技术方法**

使用的技术包括事件体素化、ViT编码器、Siamese结构以及MonST3R的深度与位姿预测头。

**📊 数据集**

使用的数据集为Sintel、Bonn、KITTI、PointOdyssey，并通过V2E仿真生成事件流。

**📈 对比分析**

与单模RGB、单模事件及事件到RGB转换等基线相比，RGB+事件在相机位姿与视频深度评估指标上均取得最低误差，尤其在高对象运动下提升明显。

**⚠️ 局限性**

局限性包括仅使用合成事件数据、仅在MonST3R上验证、融合方式过于简单、未针对事件稀疏性设计专用编码器。

---

## 243. BoxWorld: Let World Engines Control and Video Generators Render

**arXiv ID:** 2610.36371 | [PDF](https://arxiv.org/pdf/2610.36371v1)

**作者:** Zike Wu `[一作]` (University Of British Columbia), Renjie Liao `[通讯]` (University Of British Columbia)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Score-based Model` `Contrastive Learning` `Knowledge Distillation` `Prompt Engineering` `Video` `Image` `Text` `Benchmark`

**🎯 论文内容**

提出BoxWorld框架，将交互推理和生成渲染分离，使用Box World Engine执行规则、Box-State Generative Renderer按对象状态生成视频。

**💡 创新点**

创新点是引入显式但最小化的对象中心状态接口（布局框、持久属性、动态状态）和三阶段训练策略（Masked Layout Grounding、State Dynamics Adaptation、Rollout Distillation），实现实时可控交互。

**🔧 技术方法**

技术包括自适应遮罩跨注意力的状态归纳模块、LoRA微调、Diffusion Forcing、Teacher/​Self‑Forcing式分布匹配蒸馏以及基于感知模型的状态注释。

**📊 数据集**

使用公开视频数据集VideoGen‑1M、4DNeX、YouTube‑VIS、VBench、VBench‑Long进行训练，构造PairedStateBench评测。

**📈 对比分析**

在PairedStateBench上实现Trigger 92.2%、Suppress 90.5%、Paired‑CF 97.3%、Task 83.9%，明显优于LingBot‑v1/v2（≤16.2% Paired‑CF）和LongLive；在VBench/VBench‑Long及YouTube‑VIS布局‑视频任务上保持与后者相当或更优的画质与对齐指标。

**⚠️ 局限性**

局限性包括对盒子级别控制的依赖、对感知模型标注质量敏感，以及在复杂物理细节和超大规模场景下的可扩展性待进一步验证。

---

## 244. RV565: Cross-Layer Co-Design for Frame-Independent Video Playback on MicroPython-Class ESP32-S3 Systems

**arXiv ID:** 2610.35935 | [PDF](https://arxiv.org/pdf/2610.35935v1)

**作者:** Ayushman Bhattacharya `[一作]` (Pollinations AI), Anwesha Chakraborty `[通讯]` (JIS University)

**关键词:** `Hardware Architecture` `Computational Efficiency` `Compression` `Deflate` `SPI2 GDMA Transmission` `Optical Flow` `Absolute Deadline Scheduling` `Length-Based Prediction Model` `Video`

**🎯 论文内容**

在 ESP32‑S3 上实现了面向单帧的显示原生 RGB565 视频解码框架，并在独立固件与 MicroPython 系统上评估其性能。

**💡 创新点**

创新点在于将 Deflate 单帧无依赖压缩与双核重叠解码相结合，实现零丢帧 24FPS，并通过元数据预测可调度性。

**🔧 技术方法**

采用的技术包括 Deflate 无损压缩、双核协同解压、最近邻缩放、SPI2 GDMA 传输、绝对截止调度以及基于长度的预测模型。

**📊 数据集**

使用了公开的《Big Buck Bunny》480p H.264 片段，在两段 10 秒窗口内进行压缩并构建独立帧容器。

**📈 对比分析**

对比串行与重叠路径，重叠路径将解压从关键路径移除，平均帧时间从约 41.3 ms 降至 31.3 ms，保持 24FPS 且无丢帧；与 OreoOS 集成时由于文件系统读、外部内存写和阻塞 SPI 导致 FPS 降至 12.2。

**⚠️ 局限性**

局限性包括仅在单一分辨率/帧率和单个视频内容上验证，缺乏与 MJPEG 或 H.264 等基准比较，且在系统层面尚未实现双核重叠，未测量输入延迟。

---

## 245. h1: A Taste of the Bitter Lesson in Automated Theorem Proving

**arXiv ID:** 2610.35962 | [PDF](https://arxiv.org/pdf/2610.35962v1)

**作者:** Skander Moalla `[一作]` (FAIR at Meta), Julia Kempe `[通讯]` (FAIR at Meta)

**关键词:** `Machine Learning` `AI Code Assistant` `Computational Efficiency` `Explainability and Interpretability` `Federated Learning` `Knowledge Distillation` `Reinforcement Learning from Human Feedback` `Representation Learning` `Robotic Intelligence` `Meta Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Model Context Protocol (MCP)` `Reflection` `ReAct` `Text` `Benchmark`

**🎯 论文内容**

本研究提出了一种简单的“循环 harness”，通过 ReAct 交互、Lean 4 交叉验证、反射与缓存，利用前沿 LLM 在测试时扩展计算来完成形式化证明，取得了多项基准的最先进成绩，并能与复杂工作流相媲美甚至超越。

**💡 创新点**

创新点在于：① 用极简的循环结构（无预设拆解/规划）即可实现高效的推理与工具调用；② 通过反射压缩上下文并保留验证成功的进展；③ 引入缓存与持久文件编辑，显著降低输入/输出成本；④ 证明“简单测试时扩展”能与专门化工作流对抗，甚至在开源问题上表现更优。

**🔧 技术方法**

主要技术包括：前沿大模型（GPT‑5.6 Sol、Gemini 3.1 Pro、DeepSeek V4 Flash 等）；ReAct 交互模式；Lean 4 与 Mathlib 交互工具（LeanExplore、Loogle、REPL、LSP、SafeVerify）；反射 LLM；Model Context Protocol (MCP)；多轮缓存与上下文管理；成本与预算跟踪。

**📊 数据集**

使用的基准与数据集有：PutnamBench（672 题，含 Putnam100 子集）、IMO 2026（6 题）、FATE‑X（100 题）、FormalQualBench（23 题）、FC100‑solved（100 题）、FormalConjectures（236 开放题目及已解决 100 题）以及 Putnam 2025、IMO 2026 的新形式化版本。

**📈 对比分析**

评估方法：在相同模型与预算下，比较每个系统的 solve 计数、每题平均成本（美元）以及 pass@k 曲线；使用 Pareto‑domination 指标。结果显示：① 对于 PutnamBench、FC100‑solved、FATE‑X、FormalConjectures，简单循环与前沿模型实现了 SOTA（例如 GPT‑5.6 Sol 在 PutnamBench 上 670/672，FATE‑X 90/100，FC100‑solved 78/100，IMO 2026 6/6）；② 在相同模型下，该循环可在低成本（如 DeepSeek V4 Flash 0.3 $ 每题）达到或超过 Goedel‑Architect、AlphaProof Nexus、MerLean 等专门化系统；③ 长期单次尝试在开放 Erdős 问题上往往比多次重复更具成本效益。

**⚠️ 局限性**

局限性包括：① 仅支持 Lean 4 单文件形式化，无法处理多文件或大型教材级证明；② 依赖黑盒 LLM，无法量化模型对训练数据的依赖；③ 界面与工具仍为人工设计，未实现完全自适应；④ 对数值、仿真、可视化等非符号推理任务不适用；⑤ 在极少数问题上仍无法突破（如 Erdős 125、MerLean 的特殊任务）。

---

## 246. Ancestor Lifting of Quasi-Cyclic Additive Codes and Certified Quantum Codes

**arXiv ID:** 2610.36700 | [PDF](https://arxiv.org/pdf/2610.36700v1)

**作者:** Liangdong Lu `[一作]` (Air Force Engineering University), Qiang Fu `[通讯]`

**关键词:** `Information Theory` `Quantum Computing` `Coding Theory` `Information Theory` `Information Set Decoding (ISD)` `Brouwer–Zimmermann Algorithm` `GF(4) Additive Code Theory` `Quasi-Cyclic Structure` `CRT Decomposition` `Symmetric Gram Matrix Analysis` `Self-Orthogonality Verification` `Tabular` `Quantum Code Tables` `Benchmark` `Physics Related`

**🎯 论文内容**

通过构造 1‑生成（1,2）-准循环的 GF(4) 加法码，并证明其在迹内积下自正交，得到对应的二进制稳定子量子码，进而在 Grassl 码表中提供新的记录码。

**💡 创新点**

创新点包括：① 祖先提升（ancestor‑lifting）定理与除子格距离法则，将码的最小距离与其在长度因子上的祖先码联系；② 设计了判定自正交的“回文准循环”族；③ 开发了三阶段验证流水线（ISD 预筛、Brouwer–Zimmermann 精确计算、证据重检）；④ 引入 “lift‑anchored” 结构实现高效构造；⑤ 通过推广和子码方法在 34 个表项上实现精确距离提升。

**🔧 技术方法**

采用的技术主要包括：GF(4) 加法码理论、准循环（QC）结构、CRT 分解、祖先提升映射、信息集译码（ISD）、Brouwer–Zimmermann 最小距离算法、对称 Gram 矩阵分析、以及对称自正交判定与扩展构造。

**📊 数据集**

数据集来源于对 9,620 个随机（1,2）-QC 码的穷举搜索（长度 up to 153），以及 Grassl 量子码表作为基准；所有构造的生成矩阵、见证、日志与 SHA‑256 校验码已归档。

**📈 对比分析**

与 Grassl 码表比较时，证实 20 个条目改进 1，[[60,13]] 2 改进 2；进一步通过推广实现 34 条目改进。最小距离从 14 提升至 17，平均 BZ 计算耗时数秒至数小时，验证过程可复现且无误。

**⚠️ 局限性**

主要限制在于计算预算：对更大长度或更高距离的码，Brouwer–Zimmermann 计算仍超时；仅考虑 1‑生成准循环码，无法覆盖更一般的加法码结构；对部分候选码仅给出上界，缺乏严格下界；最后，证明某些“混合”分支的理论解释仍待完成。

---

## 247. Program-Level Adversarial Co-Adaptive Self-Play for Robotic Security

**arXiv ID:** 2610.36143 | [PDF](https://arxiv.org/pdf/2610.36143v1)

**作者:** Leo Benaharon `[一作]` (Isuzu Technical Center of America), Yifan Wei `[通讯]` (Isuzu Technical Center of America)

**关键词:** `Robotics` `Robotic Intelligence` `Adversarial Attack` `Autonomous Driving` `Transformer` `Large Language Model` `Reinforcement Learning` `Agentic AI` `Text` `Sequential`

**🎯 论文内容**

提出一种基于可执行程序的对抗自适应自我对弈框架，用编码代理在机器人软件栈中生成攻击与防御程序并通过任务反馈不断迭代；

**💡 创新点**

核心创新在于：1) 直接在可执行程序层面进行攻击与防御的共进化；2) 通过任务级别的质量评价与回归，避免模型微调；3) 将攻击、防御与机器人中间件/SDK接口紧密结合，实现实时、可追踪的对抗实验；

**🔧 技术方法**

使用大型语言模型（Claude、Opus、Sonnet）进行代码生成；ROS2、DDS中介实现程序提交与执行；Autoware‑CARLA 与 Unitree G1‑MuJoCo 作为模拟平台；评估指标包括到达率、进度分数、速度比等；

**📊 数据集**

实验数据集为：Autoware 在 CARLA 的 60 m 直线行驶任务；Unitree G1 在 MuJoCo 的 3 m 地面步行任务；共 8 代演化，每代分别完成清洁、攻击、守卫、联合多轮评估；

**📈 对比分析**

对比方法为：在每代生成前后分别跑清洁、攻击仅、守卫仅与多轮联合测试；性能表现为驾驶任务 40 轮中 18 轮到达成功、16 轮超时、6 轮碰撞；人形任务 176 轮中 51 轮到达成功，其中攻击仅 30/40 失败、联合 25/120 达成；

**⚠️ 局限性**

限制包括：仅单条八代演化，任务和场景单一；评价指标仅关注进度与速度，未覆盖安全性、平滑度等；未对各反馈机制做消融实验；防御编排器尚未实现多专长模块融合与回滚机制。

---

## 248. Generation and Verification: Human-Agent Collaboration for Safety-Critical Automotive Software Engineering

**arXiv ID:** 2610.35982 | [PDF](https://arxiv.org/pdf/2610.35982v1)

**作者:** Elizaveta Zinovyeva `[一作]` (Amazon Web Services), Brian Jensen `[通讯]` (Amazon Web Services)

**关键词:** `Software Engineering` `AI Code Assistant` `Autonomous Driving` `Safty and Privacy` `Transformer` `Large Language Model` `Prompt Engineering` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

构建了一个基于Model Context Protocol的汽车软件编码助手，提供生成、编辑和验证三种工具，并在安全关键的汽车领域中实现可审计的验证流程。

**💡 创新点**

创新点在于将验证作为独立可触发的工具与生成分离，并通过提示工程与微调两种方式定制LLM，使其在满足MISRA、AUTOSAR等标准的同时保持高效交互。

**🔧 技术方法**

采用Qwen3-32B等大模型，通过Prompt Engineering和LoRA微调，配合静态规则引擎和MCP架构实现生成、编辑和验证功能。

**📊 数据集**

数据集来自AUMOVIO的生产汽车代码，包含约6,000个函数，每个函数附带需求文本、UML行为、API上下文以及手工标注的规则集。

**📈 对比分析**

对比了PE与FT两种定制策略，在C/C++代码的规则违规率、生成延迟以及人类工程师评估中，PE在C语言违规率最低，FT在C++架构违规率稍优，整体生成延迟从14s到110s不等。

**⚠️ 局限性**

局限包括未评估功能正确性与编译通过率、仅使用单一组织的数据与规则、未进行显著性检验、以及人类评估仅基于自报数据且未收集使用日志。

---

## 249. Making Self-Driving Laboratories Auditable:A Unified Framework for Hybrid Actions,Measurements, and Safe Execution

**arXiv ID:** 2610.36761 | [PDF](https://arxiv.org/pdf/2610.36761v1)

**作者:** Zihan Zhang `[一作]` (Shanghai Innovation Institute), Tong Zhu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `Robotics` `Autonomous Driving` `Robotic Intelligence` `Safety and Privacy` `Transformer` `Vision Language Model` `Vision-Language-Action Model` `Signal Temporal Logic` `Retrieval-Augmented Generation` `Multimodality` `Tabular` `Time Series` `Benchmark`

**🎯 论文内容**

构建了一套统一框架，集成了基于测量条件的混合动作生成（LAM）、统一执行子系统（OpenSDL）、运行时安全门控（Safe-SDL）以及实验评价基准（L3-Bench）

**💡 创新点**

首次将测量令牌与视觉语言动作并行结合，生成连续轨迹和类型化仪器API；实现了跨设备、跨时钟的可追溯事件记录；提供了端到端的可复现实验评估协议

**🔧 技术方法**

使用多模态Transformer作为LAM的VLM骨干，流匹配生成机器人轨迹，函数调用头生成API；Rust实现的OpenSDL执行引擎；Signal Temporal Logic (STL) 监控；SQLite记录事件；Clean-3D数字孪生

**📊 数据集**

无公开标准数据集，使用自定义实验批次（高通量稀释、测量条件滴定、故障注入）在模拟/硬件/数字孪生中收集的原始观察、动作、返回记录

**📈 对比分析**

与仅视觉单臂策略以及无测量令牌的LAM做对比，采用统一任务脚本、阈值和终止规则；在标称、测量偏移、设备迁移和故障注入等四种设置下评估；预期通过测量令牌提升适应性、并行调度降低完成时间、运行时门控提高故障恢复率

**⚠️ 局限性**

尚未完成模型训练与实验验证；缺乏安全认证；仅在有限的硬件与模拟环境验证；事件记录和安全监控依赖于正确的时钟同步；对未知化学反应与极端环境的鲁棒性有限

---

## 250. Online gas classification at the edge

**arXiv ID:** 2610.36999 | [PDF](https://arxiv.org/pdf/2610.36999v1)

**作者:** Ton Juny Pina `[一作]` (University of Groningen), Elisabetta Chicca `[通讯]` (University of Groningen)

**关键词:** `Emerging Technologies` `Classification` `Computational Efficiency` `Anomaly Detection` `Spiking Neural Network` `Contrastive Learning` `Hebbian Learning` `Time Series` `Sensor Data`

**🎯 论文内容**

设计并实现了一种受昆虫嗅觉系统启发的稀疏高维网络，用于在边缘设备上在线、实时地对气体混合物进行分类，并在混合信号下实现少样本自适应重校准。

**💡 创新点**

创新点在于将随机二进制投影与k-WTA稀疏化相结合，形成一种可直接映射原始传感器读数到高维稀疏码的“expand‑and‑sparsify”架构，并使用仅二进制权重和Hebbian本地学习实现极低功耗、可在线学习的神经网络，具备计数草图功能，可在有限标记样本下实现高效重校准。

**🔧 技术方法**

主要技术包括：高维随机二进制投影、k-WTA稀疏化、单层线性读出层（二进制权重Hebbian学习或64位梯度下降线性读出）、离线与在线训练、嵌入式稀疏高维特征、在TEXEL混合信号神经形态芯片上实现。

**📊 数据集**

使用自制的含有8个Mox传感器的传感器阵列，测量的气体为乙烯、甲烷、一氧化碳及其二元混合物；数据集为随机序列，共600次训练和300次测试，其中单气体和二元混合两种版本。

**📈 对比分析**

与线性SVM、RBF SVM以及64位梯度下降线性读出进行对比；在二元混合任务上，网络达0.645的准确率优于线性SVM 0.563，近似RBF SVM 0.80；在单气体任务上略逊于线性SVM；在芯片上实现的二分类任务中准确率为74.6%，功耗为每次推断482 nJ、静态功耗27 nW。

**⚠️ 局限性**

局限性包括：对传感器漂移的适应仍依赖于持续在线学习；二进制权重与低分辨率投影可能限制了在更大、更多类别数据集上的性能；在高噪声或高浓度混合环境下的鲁棒性尚未充分验证；芯片实验受限于2000维稀疏码，无法充分验证更高维特征的优势。

---

## 251. Latent Residual Reasoning for Ranking in Generative Recommendation

**arXiv ID:** 2610.36890 | [PDF](https://arxiv.org/pdf/2610.36890v1)

**作者:** Haihan Shi `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Supervised Fine-Tuning` `LoRA 低秩适配` `Chain-of-Thought` `Tabular` `Sequential`

**🎯 论文内容**

提出了一种名为 Refiner 的框架，在生成式推荐模型上实现多步残差式潜在推理，以提升候选项的分数精度。

**💡 创新点**

核心创新包括（1）残差累积推理机制，使得每一步仅输出对前一步分数的增量修正；（2）历史查询向量提供压缩的全局用户行为上下文；（3）通过 LoRA 等参数高效微调，仅在冻结基础模型的前提下添加轻量级模块，保持模型部署的可行性。

**🔧 技术方法**

技术细节涵盖基于 Transformer 的生成式推荐（MTGR、SASRec-ranking）、残差推理机制、LoRA 低秩适配、历史查询、候选掩码、AUC/GAUC 评估和线上 A/B 测试。

**📊 数据集**

实验使用 Steam、MovieLens10M 两个公开数据集，70M 用户/4M 物品、20 亿交互的工业离线数据，以及线上 A/B 测试的真实流量。

**📈 对比分析**

通过与传统 xDeepFM、DIN、ReaRec 等基线对比，公测数据上 AUC/GAUC 提升约 0.8%–1.4%，工业数据上 CTR AUC/GAUC 分别提升 1.8%–4.3%，线上 A/B 测试中 PV‑CTR、UV‑CTCVR、GMV 分别提升 0.75%、0.21%、0.46%。

**⚠️ 局限性**

局限性在于：仅在已有的生成式推荐基线上微调，比较范围受限；与 OnePiece 等需交互式候选的先进模型对比不足；当前不支持自适应推理深度，未来可进一步探索。

---

## 252. Confidence Is Not a Promise: A Reference Audit of Typed Decision Models

**arXiv ID:** 2610.36311 | [PDF](https://arxiv.org/pdf/2610.36311v1)

**作者:** Pedro Carvalho Brom `[一作]` (Federal Institute of Brasília), Pedro Carvalho Brom `[通讯]` (Federal Institute of Brasília)

**关键词:** `Machine Learning` `Classification` `Explainability and Interpretability` `Domain Adaptation` `Transformer` `Supervised Fine-Tuning` `Bayesian Inference` `Contrastive Learning` `Prompt Engineering` `Temperature Scaling` `Text` `Multimodality` `Benchmark` `Finance Related`

**🎯 论文内容**

研究并评估了 typed decision model（如 Laya 与 Jev）的概率输出、校准、门控机制及其在不同语言、噪声、域漂移下的表现，并对比了多种阈值选择方法。

**💡 创新点**

提出了针对这些模型的贝叶斯下限门（Beta lower‑bound rule）与 Learn‑then‑Test 的对比，系统化评估其在保证准度与覆盖率方面的优缺点，并验证了噪声与语言迁移对保证的影响。

**🔧 技术方法**

采用温度缩放、贝叶斯推断、Clopper–Pearson 置信区间、AUROC、Brier/NLL 等统计方法，结合阈值搜索（固定序列、递增扫描）和多目标门优化。

**📊 数据集**

主要使用 MASSIVE 意图分类数据集（pt‑PT 与 en‑US），以及新闻标题、金融推文、LIAR2、Fake.Br 等四个额外领域的数据，涵盖 52 种语言的并行版本。

**📈 对比分析**

实验显示，微调后 Laya 在大约 4,000 标签即可匹配或超越 Jev 的准确率；贝叶斯下限门能在保证准度的前提下略高于 Learn‑then‑Test 的覆盖率；但噪声、语言/域变更会显著破坏保证，需重新校准。

**⚠️ 局限性**

局限性包括仅评估五选项意图任务、对噪声与多语言迁移的鲁棒性有限、贝叶斯下限门在极低采样下仍易失效、实验多为探索性、以及模型输出被截断导致的精度误差。

---

## 253. Evaluating Priority Dispatch for Premium Membership in On-demand Taxi-sharing

**arXiv ID:** 2610.37011 | [PDF](https://arxiv.org/pdf/2610.37011v1)

**作者:** Nozomi Yamamoto `[一作]` (Hokkaido University), Itsuki Noda `[通讯]` (Hokkaido University)

**关键词:** `Multiagent Systems` `Autonomous Driving` `Optimization` `Tabular` `Time Series` `Benchmark`

**🎯 论文内容**

研究在按需计程车共享中引入高端会员，比较两种调度策略的时间与服务差异。

**💡 创新点**

提出基于短时限与专用车队的两种优先调度方法，并揭示其在不同需求场景下的效果差异。

**🔧 技术方法**

采用基于SUMO的仿真，利用顺序最优插入算法与时间/距离比指标评估。

**📊 数据集**

使用合成15×15网格地图与三种需求场景（均匀、时间变动、城市聚焦）生成的随机请求数据。

**📈 对比分析**

通过30次随机种子仿真，对时间/距离比、取消率和服务差距进行统计，结果显示短时限在高负荷下优于专用车队，反之亦然。

**⚠️ 局限性**

限制在于缺乏定价模型、混合策略、真实地图与车辆动态，且仅评估了随机请求。

---

## 254. When do token systems stabilize? Compatibility constraints and community structure

**arXiv ID:** 2610.36744 | [PDF](https://arxiv.org/pdf/2610.36744v1)

**作者:** Flore Sentenac `[一作]` (HEC Paris Business School), Sophie H. Yu `[通讯]` (University of Pennsylvania)

**关键词:** `Computer Science and Game Theory` `Optimization` `Reinforcement Learning from Human Feedback` `Federated Learning` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Adversarial Attack` `Hyperparameter Search` `Data-Centric Learning` `Robotic Intelligence` `Meta Learning` `Drug Discovery` `AI Code Assistant` `Neural Architecture Search` `Protein Structure Prediction` `Foster–Lyapunov drift` `Markov chain stability theory` `Random graph theory` `Stochastic block model` `Erdős–Rényi graph` `Large deviation analysis` `Lyapunov method` `Game optimization` `Review/Survey Paper`

**🎯 论文内容**

研究在兼容性约束下的代币系统的稳定性，并给出必要且充分的稳定性判据，证明最小代币规则在所有可行的稳定策略中是最优的。

**💡 创新点**

提出了新的基于割条件的稳定性判据，将最小代币规则的最优性与马尔可夫链的负漂移、Lyapunov方法联系起来；同时在随机块模型（stochastic block model）下识别了两类阻碍稳定性的结构性因素（社区阻碍和个体阻碍），并给出了对应的阈值条件。

**🔧 技术方法**

主要使用了马尔可夫链稳定性理论（Foster–Lyapunov漂移）、随机图理论（随机块模型、Erdős–Rényi图）、大偏差分析（速率函数 I_i）、马尔可夫和大数原理以及对策最优化（最小代币规则的贪婪性）。

**📊 数据集**

本文没有使用实验数据集，全部在理论模型（随机块模型和 Erdős–Rényi 图）上进行推导与证明。

**📈 对比分析**

由于是理论研究，没有与其他方法做实验性能比较；论文通过证明稳定性判据和阈值，说明在满足判据的条件下系统是稳定的，而在不满足时系统必定不稳定，提供了完全的理论性能保证。

**⚠️ 局限性**

局限性在于：仅考虑无策略（非策略性）最小代币规则，未讨论自适应或激励兼容性；模型假设兼容性边缘独立且是随机块模型，实际应用中可能存在更复杂的社群结构；未给出实验验证，缺乏对真实代币系统的验证。

---

## 255. PDE-Based Bounds on Optimal Exponents in Active Simple Hypothesis Testing

**arXiv ID:** 2610.36657 | [PDF](https://arxiv.org/pdf/2610.36657v1)

**作者:** Sushant Vijayan `[一作]` (Plaksha University), Himadri Mandal `[通讯]` (Indian Statistical Institute)

**关键词:** `Information Theory` `Optimization` `Stochastic Differential Equation` `Ordinary Differential Equation` `Physics Related`

**🎯 论文内容**

研究了主动简单假设检验（ASHT）问题的最优误差指数，给出了更紧的上下界；

**💡 创新点**

通过将误差指数与Hamilton–Jacobi PDE的黏性解联系，并利用比较原理和凸/凹逼近，提出了可参数化、统一的误差指数界；

**🔧 技术方法**

主要技术是PDE理论、黏性解比较原理、Hopf–Lax公式以及Hamiltonian的下界松弛；

**📊 数据集**

无；

**📈 对比分析**

无实验对比，论文仅给出理论界限与已有结果（如Chernoff信息、Rgo_B界）进行比较，表明在理论上可取得更优界；

**⚠️ 局限性**

局限在于缺乏具体数值示例或实验验证，且上界的某些取值是否严格优于已有结果仍需进一步证明。

---

## 256. A Harness Reducing over-Defensive Writing in Agents by Isolating Objections

**arXiv ID:** 2610.36717 | [PDF](https://arxiv.org/pdf/2610.36717v1)

**作者:** Xiaochen Chang `[一作]`, Xiaochen Chang `[通讯]`

**关键词:** `Computation and Language` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Data-Centric Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

提出一种将写作任务拆分为审稿、过滤和撰写三阶段的框架，旨在减少写作中的过度防御性表述

**💡 创新点**

通过上下文隔离和预先筛选审稿点，显著降低防御性陈述出现率，同时保持原始来源主张的保留率

**🔧 技术方法**

利用大语言模型（GPT-6.1、DeepSeek-V4.1-Flash）实现审稿、过滤与写作，结合阈值选择机制和证据强度约束

**📊 数据集**

使用25个商业项目草案作为实验数据集，生成约500字的正式报告进行评估

**📈 对比分析**

与直接扩写和仅禁止防御性写作的对照组比较，防御性主张从59降至2，保留率从80.35%提升至93.12%，未出现未支持事实添加

**⚠️ 局限性**

局限在评估规模有限，依赖LLM评估器，需进一步通过更大样本和人工标注验证一致性

---

## 257. Design of DoF-Robust MIMO via Generalized Pinching Antennas

**arXiv ID:** 2610.36670 | [PDF](https://arxiv.org/pdf/2610.36670v1)

**作者:** Zhiguo Ding `[一作]` (Nanyang Technological University), H. Vincent Poor `[通讯]` (Princeton University)

**关键词:** `Information Theory` `Optimization` `Federated Learning` `Computational Efficiency` `Physics Related` `Random Matrix Theory` `Numerical Integration` `Simulation and Modeling` `Tabular` `Time Series` `Simulation Data` `Physics Related`

**🎯 论文内容**

本文提出并验证了一种利用广义“pinching”天线的MIMO系统设计，旨在通过将天线靠近用户来保持空间自由度（DoF），从而降低多用户信道矩阵的秩缺失风险并提升系统可靠性。

**💡 创新点**

创新点在于：①将pinching天线动态布置于用户附近，使多用户信道Gram矩阵接近对角结构；②推导了二用户下的秩缺失概率和切换误差概率的闭式表达；③通过高信噪比极限分析揭示pinching天线在空间分布用户时对DoF和可靠性的显著提升。

**🔧 技术方法**

使用的技术包括：广义pinching天线设计、零逼近（ZF）波束成形、随机矩阵理论（2×2非同源Wishart分布）、高SNR极限展开、数值积分与模拟仿真。

**📊 数据集**

采用仿真数据：在三角频率3.5 GHz、波导高度3 m、噪声功率-80 dBm的条件下，用户位置随机均匀分布于矩形服务区，使用路径损耗指数α=2、Rayleigh衰落模型。

**📈 对比分析**

与传统固定位置天线系统进行对比，评价指标为秩缺失概率和出局概率；分析与仿真显示，pinching天线在用户距离差异较大或空间分布较广的场景下，秩缺失概率显著降低，出局概率可下降数十分贝，尤其在高SNR下性能优势更为突出。

**⚠️ 局限性**

局限性包括：仅在二用户情形得到完整解析，通用多用户场景仅通过仿真验证；假设pinching天线位置固定且未考虑波导衰减；零逼近波束需完美CSI，实际系统需进一步研究对CSI不完美的鲁棒性。

---

## 258. Camera-to-Quadruped Robot Pose Estimation from Sequential Images

**arXiv ID:** 2610.36599 | [PDF](https://arxiv.org/pdf/2610.36599v1)

**作者:** Seobin Hwang `[一作]` (Chonnam National University), Yeong-Jun Cho `[通讯]` (Chonnam National University)

**关键词:** `Robotics` `Pose Estimation` `Robotics` `Computer Vision` `Transformer` `Convolutional Neural Network` `Contrastive Learning` `Optical Flow` `Supervised Fine-Tuning` `Video` `Image` `Point Cloud` `Benchmark`

**🎯 论文内容**

本文提出了一种从外部单目相机视频序列估计四足机器人18-DoF全身姿态（6-DoF基座 + 12-DoF关节角）的框架，并推出了首个公开的四足机器人相机姿态基准数据集。

**💡 创新点**

创新点包括：1）引入关键点选择器（selector token）和骨架感知遮罩自注意力，解决腿对称与自遮挡导致的2D关键点歧义；2）结合时间卷积和残差校正的三维姿态恢复，显著提升深度解算和位姿精度；3）构建首个公开的四足机器人相机姿态基准数据集。

**🔧 技术方法**

核心技术包括：Transformer编码器 + Selector token + Skeleton-aware Masked Self-Attention + SimCC 1D分布预测用于2D关键点；预训练的JointNet MLP预测关节角；基于相机射线与基座位姿的时间卷积推理；前向运动学（FK）+ Refinement MLP校正三维关键点；多损失（PCK、ADD、旋转几何损失）进行训练。

**📊 数据集**

使用的数据集为Synth4Dog（或名为QuadrupedPoseBenchmark），包含51,697张合成RGB图像，覆盖52个背景场景，来自Boston Dynamics Spot，提供2D/3D关键点、6-DoF基座姿态、12-DoF关节角、相机内参外参和边界框。

**📈 对比分析**

与DREAM、RoboPose、SGTAPose、RoboPEPP等基线相比，本文方法在PCK@2.5px达到92.18%（+5.6%），ADD@20cm 85.94%（+10%），ADD-AUC 66.98%，平均关节角误差3.65°，并以约33FPS的速度实现实时推理，整体性能显著提升。

**⚠️ 局限性**

局限性包括：仅在单一机器人（Spot）与合成场景训练，未充分验证对其他平台（Go1、ANYmal）的泛化；在极端遮挡或光照变化下仍可能出现失误；虽然仿真→真实迁移效果良好，但仍需进一步的域适配研究。

---

## 259. Preserving rare-item recall under a storage budget: quantization, distillation, and the limits of teacher fidelity

**arXiv ID:** 2610.36267 | [PDF](https://arxiv.org/pdf/2610.36267v1)

**作者:** Han Chen `[一作]` (University of Texas at Austin), Yingrui Li `[通讯]`

**关键词:** `Information Retrieval` `Recommendation System` `Compression` `Knowledge Distillation` `Graph Neural Network` `Contrastive Learning` `Knowledge Distillation` `Auto Encoder` `Graph` `Tabular`

**🎯 论文内容**

在固定存储预算（168 字节/行）下，比较了七种压缩方案（低秩因子化、4‑bit 低精度、UnKD 40 维学生、BPR 等），并评估它们在稀有（尾部）物品召回上的表现。

**💡 创新点**

创新点在于：① 证明在相同字节数下，更多的秩（rank）远比更高精度（precision）更能保留尾部召回；② 对比教师保留率（teacher‑hit retention）与实际尾部召回，揭示教师保留率并不能可靠指示压缩模型的尾部性能；③ 通过多数据集、多模型基线和严格的统计检验，首次系统评估压缩推荐系统在稀有物品上的真实效果。

**🔧 技术方法**

使用的技术包括：矩阵分解（MF）、LightGCN、贝叶斯个性化排序（BPR）、Unbiased Knowledge Distillation（UnKD）、SVD‑based 低秩因子化、不同位宽量化（32、16、8、4 位）、对照实验与自定义评估指标（tail recall@20、NDCG@20、teacher‑hit retention、off‑list 贡献），以及 10,000 次配对 Bootstrap 置信区间和 Holm 校正。

**📊 数据集**

数据集：Gowalla、Last.fm、Amazon Video Games、Steam、Goodbooks‑10k、MovieLens‑20M（MF 基础），以及 Amazon Video Games 的 LightGCN 基础；全部使用公开的 5‑core 过滤后、非时间序列划分。

**📈 对比分析**

在七个数据集·基线组合（共 7 cells）下，每个 cell 评估 7 个候选模型（共 21 对），均在相同字节预算内。结果显示：4‑bit 300‑秩因子化保留 95–103% 的教师尾部召回，明显优于 40‑维低精度或 UnKD 学生；教师保留率与尾部召回往往不一致；尾部召回在 10 个预设比较中均超过学生，且 4‑bit 因子化在大多数 cell 上取得最优或接近最优的 NDCG@20。

**⚠️ 局限性**

局限性：仅评估 MF（及少量 LightGCN）模型，缺乏冷启动或时间序列推断；尾部阈值固定为 80% 最少互动物品，列表长度固定为 20；仅测试 UnKD 40‑维学生与单一量化方案（均匀量化）；未测量推理延迟和实际在线系统成本；实验未涵盖多种压缩策略（如哈希、产品量化、混合精度）。

---

## 260. Context Is a Pipeline, Not a Window: Information Attrition in LLM Agent Harnesses

**arXiv ID:** 2610.36114 | [PDF](https://arxiv.org/pdf/2610.36114v1)

**作者:** Yuvraj Sehgal `[一作]` (Brock University), Naser Ezzati-Jivan `[通讯]` (Brock University)

**通讯引用:** 574 | [OpenAlex ID](https://openalex.org/A5019537475)

**关键词:** `Software Engineering` `Computational Efficiency` `Explainability and Interpretability` `Safty and Privacy` `Transformer` `Large Language Model` `Agentic AI` `Text` `Review/Survey Paper`

**🎯 论文内容**

本文通过对LLM代理运行时工具输出处理流程进行细粒度测量，揭示并量化了代理在上下文传递过程中的信息流失；并在真实的Linux内核跟踪调试任务上，重放和对比不同切片策略的效果；进一步通过受控实验验证信息流失对计算成本和准确率的影响；最后对七个主流代理框架的工具输出切片实现进行调研，评估其安全性。

**💡 创新点**

创新点在于：①提出“上下文是管道而非窗口”的五阶段模型，并为每一阶段设计可测量的损失指标；②首次在真实工作负载中重放并量化工具结果被切片导致的数据丢失；③通过受控实验展示信息流失导致的不是准确率下降而是额外的计算工作；④发现并比较多种框架默认的基于位置的切片实现，提示潜在的通用缺陷。

**🔧 技术方法**

技术手段包括：对工具结果进行完整记录与序列化，使用前缀切片与行级保留两种切片策略进行对比；利用自定义的“yield”指标（记录成果/已执行计算）衡量信息利用率；对多轮调用进行重复实验并记录token数、工具调用次数、计算量；对不同框架的源代码进行静态分析，识别切片实现。

**📊 数据集**

数据集主要来自两类：①720次代理运行产生的11,189条工具结果（含3,827条被切片丢失）；②31次实验中重放的864条真实工具结果；③一次规模化生产跟踪（21容器部署）中的1,189条工具结果；所有数据均来自Linux内核跟踪与代理自带的诊断任务。

**📈 对比分析**

比较方法：对比修复（row‑aware 切片）与缺陷（前缀切片）及其变体（键顺序反转、仅键顺序控制）的实验组，记录正确率、工具调用次数、计算量、token使用、yield；在同一任务与模型（temperature 0）下保持其他因素不变。实验结果显示，信息流失导致工具调用+28%→+43%、计算量翻倍、yield从0.61降至0.31，但准确率保持不变，说明成本上升而非准确率下降。

**⚠️ 局限性**

局限性包括：①仅在单一LLM代理与单一任务（内核跟踪诊断）上验证，缺乏跨模型、跨领域的普适性验证；②受控实验使用的是有限数量的轨迹与故障实例，样本量有限，无法完全排除偶然性；③实验仅关注前缀切片对信息流失的影响，未探究其它切片策略（如分页、缓存）对准确率的直接因果关系；④缺失对生产规模下完整时延与成本的实测。

---

## 261. InfraDev: Can Coding Agents Develop AI Infrastructure, and How Can We Tell?

**arXiv ID:** 2610.36589 | [PDF](https://arxiv.org/pdf/2610.36589v1)

**作者:** Yangshen Deng `[一作]` (University of Edinburgh), Luo Mai `[通讯]` (University of Edinburgh)

**关键词:** `Artificial Intelligence` `AI Code Assistant` `Computational Efficiency` `Benchmark` `Transformer` `Large Language Model` `Prompt Engineering` `Agentic AI` `Retrieval-Augmented Generation` `Text` `Code` `Benchmark`

**🎯 论文内容**

构建了一个基于真实 AI 基础设施项目（Megatron‑LM、vLLM、verl）合并 PR 的 36 任务基准，并提出隐藏探针与提示合成技术来评估大型模型的代码生成能力。

**💡 创新点**

引入联合测试与提示合成的隐藏探针，显著提升了任务评判的可靠性；同时设计了共享依赖、共享模型缓存的资源高效执行环境，降低存储与构建时间。

**🔧 技术方法**

利用代理生成的隐藏探针和提示、Docker 虚拟环境共享依赖、深度学习框架（Megatron‑LM、vLLM、verl）以及多 GPU 训练与测试基础设施。

**📊 数据集**

从 Megatron‑LM、vLLM、verl 三大 AI 基础设施项目中挑选 36 个合并 PR 作为任务集，涵盖训练、推理与强化学习。

**📈 对比分析**

加入隐藏探针后模型完成率下降 11–19%，最高模型 38.9%；使用自定义技能后 GPT‑5.6 Sol 任务完成率提升至 52.6%，显示隐藏探针与技能对性能的积极影响。

**⚠️ 局限性**

受限于多 GPU 执行成本高、PR 选取需要人工专家干预，难以快速扩展与自动化；以及评测仍需多 GPU 资源，影响可扩展性。

---

## 262. INC-MoE: Overlapping In-Network Collectives and Expert Computation for Expert-Parallel Mixture-of-Experts Inference

**arXiv ID:** 2610.36339 | [PDF](https://arxiv.org/pdf/2610.36339v1)

**作者:** John Stephan `[一作]` (Huawei), Guan Zixuan `[通讯]` (Huawei)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Computational Efficiency` `Optimization` `Transformer` `Mixture of Experts` `In-Network Computing` `RoCEv2` `Pipeline Parallelism` `Band-based Pipelining` `Device-side Synchronization` `Text`

**🎯 论文内容**

提出了INC-MoE，一种在规模化部署中将专家并行的Mixture-of-Experts层的调度与合并任务下放到网络交换机，实现通信与计算的重叠，从而提升推理吞吐量和降低首个token延迟。

**💡 创新点**

创新点包括：① 用消息语义实现的in‑network AllGather和ReduceScatter，① 通过交换机同时执行两种聚合来利用链路双向带宽；② 在单一持久内核中按band进行管线化，① 让调度、专家前馈和合并三阶段并行，② 通过设备侧同步实现无宿主交互。

**🔧 技术方法**

使用技术：网络层级的in‑network computing、RoCEv2消息协议、设备侧一侧通信库（MSCCL++）、专用交换机仿真器、持久线程块角色分配与自定义同步、SwiGLU FFN与权重融合。

**📊 数据集**

采用OLMoE-1B-7B模型（E=64，top‑k=8，H=2048，I=1024）进行推理预填充实验，使用4节点scale‑out测试床，分别对不同token数（24K–126K）进行测评。

**📈 对比分析**

与vLLM默认AllGather/ReduceScatter路径进行对比，采用TTFT、每加速器吞吐率、等效带宽三项指标；实验显示TTFT降低20.2%–33.1%，吞吐率提升25.3%–49.5%，等效带宽提升25.6%–31.8%。

**⚠️ 局限性**

局限性包括：① 交换机实现仍为软件仿真器，硬件实现效果尚未验证；② 对小消息效率仍受仿真开销影响；③ 需要对band数和线程块分配进行调优，适配不同token规模；④ 对不规则专家分配或极端负载不平衡场景的鲁棒性待进一步研究。

---

## 263. Learning-Augmented Coordination Mechanisms

**arXiv ID:** 2610.36508 | [PDF](https://arxiv.org/pdf/2610.36508v1)

**作者:** George Christodoulou `[一作]` (Aristotle University of Thessaloniki), Ioannis Vlachos `[通讯]` (Athens University of Economics and Business)

**关键词:** `Computer Science and Game Theory` `Optimization` `Reinforcement Learning`

**🎯 论文内容**

研究了学习增强的协调机制在原子和非原子拥塞游戏中的设计与分析，提出了一种新的协调机制，旨在通过对资源成本的修改来改善社会均衡结果。

**💡 创新点**

创新点在于引入了学习增强的协调机制，该机制结合了对需求的预测，能够在准确预测时实现近似最优性能，同时保持良好的最坏情况保证（PoS）。

**🔧 技术方法**

使用了学习增强的协调机制，结合了潜在函数和阈值机制来调整资源的延迟函数。

**📊 数据集**

研究了具有多项式延迟函数的原子和非原子拥塞游戏，具体的实验数据集未明确给出。

**📈 对比分析**

通过与原始游戏的最佳纳什均衡进行比较，提出的机制在准确预测时能够提高最佳均衡的性能，并在不准确预测时保持良好的最坏情况保证。性能表现优于传统的协调机制。

**⚠️ 局限性**

限制在于对于更强的一致性要求，任何协调机制都无法保证有限的鲁棒性，且在面对微小的预测误差时，性能可能会急剧下降。

---

## 264. Beyond IceBERT: Open-Data ModernBERT Encoders for Icelandic from 12M to 1B Parameters

**arXiv ID:** 2610.36230 | [PDF](https://arxiv.org/pdf/2610.36230v1)

**作者:** Hafsteinn Einarsson `[一作]` (University of Iceland), Hafsteinn Einarsson `[通讯]` (University of Iceland)

**关键词:** `Computation and Language` `Representation Learning` `Data-Centric Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Fast Vocabulary Transfer` `Text`

**🎯 论文内容**

本文训练了从12M到1B参数的ModernBERT Icelandic编码器，使用9B开放文本重新评估IceBERT基准并加入五个新任务；

**💡 创新点**

创新点在于首次构建1B规模单语冰岛语编码器，并在受控条件下系统比较从Ettin和mmBERT等源模型进行warm‑start与从零训练的差异，展示词表迁移和数据节省的效果；

**🔧 技术方法**

使用的技术包括ModernBERT架构、掩码语言建模、Fast Vocabulary Transfer、按token/参数的训练预算、层次化自举置信区间评估以及多源去重与去污流程；

**📊 数据集**

所用数据集为九个公开冰岛语语料库（Gigaword、Wikipedia、Dynaword、MaCoCu、FineWeb‑2、FinePDFs、HPLT、MADLAD‑400、IC3‑v2）共9B词，额外加CulturaX用于更大token预算；

**📈 对比分析**

通过统一fine‑tune harness、调优学习率、5个随机种子和层次化自举，比较模型在11项原IceBERT任务及5项新任务上的性能；结果显示1B从零训练模型在所有任务上显著优于IceBERT，而warm‑start模型在多数任务上平均得分最高，且在低token预算下可将所需文本量削减至四分之一或八分之一；

**⚠️ 局限性**

局限性包括种子波动导致的噪声、warm‑start比较未覆盖所有训练阶段、训练数据可能包含测试文本、数据版权限制、仅评估fine‑tune性能、仅针对冰岛语且仅在公开文本上实验，未验证长文本或跨语言迁移的通用性。

---

## 265. Recall or Relearn? A Mixture of In-Context Experts for Recurring Concept Drift

**arXiv ID:** 2610.36585 | [PDF](https://arxiv.org/pdf/2610.36585v1)

**作者:** Zhangkai Wu `[一作]` (University of Sydney), Yi Zhang `[通讯]` (University of Technology Sydney)

**关键词:** `Machine Learning` `Classification` `Anomaly Detection` `Computational Efficiency` `Federated Learning` `Data-Centric Learning` `Transformer` `Mixture of Experts` `Prompt Engineering` `Contrastive Learning` `Retrieval-Augmented Generation` `Tabular` `Time Series` `Sequential`

**🎯 论文内容**

本文提出一种名为 MICE（Mixture of In-Context Experts）的无训练方法，利用冻结的表格基础模型（TabPFN）在数据流中管理存储的上下文与 FIFO 窗口，自动在检测到概念漂移时切分段并合并符合召回成本阈值的上下文，从而在递归概念漂移情境下实现记忆与重新学习的决策。

**💡 创新点**

创新点包括：① 用最近一批的 log‑loss 对存储上下文和 FIFO 窗口进行加权的快速层；② 通过半 Brier 损失的折扣后期望值上界（≤ 2 ln 2）控制慢速层的混合预测；③ 在检测到漂移时通过估计回忆再学习误差（recall regret）来合并段落，确保上下文纯净；④ 提供理论证明，指出在纯上下文与边际条件下，记忆带来的期望准确率提升等于学习曲线下的面积；⑤ 在多种合成与真实数据流上实现了最优或接近最优的准确率，远超 reset、FIFO、LTM 等传统策略。

**🔧 技术方法**

使用技术：TabPFN v2 作为基础模型；两样本检验在 300 行窗口上检测漂移；快速层采用指数权重（log‑loss）更新；慢速层采用折扣半 Brier 损失的多专家融合；跨折叠估计回忆再学习误差；纯上下文与边际条件下的理论分析。

**📊 数据集**

数据集：合成的 Grid 流（Gaussian 均值中心数 {5,30,100}，区块长度 200/500/2000），River 官方流（SEA、STAGGER、Agrawal、Sine、RBF、Hyperplane），真实流（Electricity、INSECTS、CoverType、PokerHand、Airlines、NOAA Weather、Spam、Phishing、Bitcoin Heist、CIC‑IDS2017）。

**📈 对比分析**

与 FIFO、LTM、Reset、窗口集合、Drift‑Resilient TabPFN 以及增量学习器（ARF、SRP、HAT、Leveraging Bagging 等）进行基准比较。MICE 在 29 条真实流上平均准确率 83.8%（比 FIFO 83.3%、Reset 81.5% 高），在 70/72 条随机边界的合成流上也明显优于 reset；F1、AUC、ECE、Brier 亦保持领先或相近；慢速层保证在每步的折扣半 Brier 损失相对 FIFO 的后退不超过 2 ln 2。

**⚠️ 局限性**

局限性：① 需要冻结的 TabPFN，若模型需更新会失效；② 理论保证基于纯上下文与显著边际条件，若实际边际过小或上下文混合，效果可能下降；③ 回忆再学习误差估计依赖交叉验证，计算开销较大；④ 对高维非表格数据的适用性未知；⑤ 参数（漂移阈值、回忆阈值）仍需手工调优。

---

## 266. Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training

**arXiv ID:** 2610.36794 | [PDF](https://arxiv.org/pdf/2610.36794v1)

**作者:** Yue Wu `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**关键词:** `Machine Learning` `Computational Efficiency` `Representation Learning` `Retrieval` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Mixture of Experts` `Text` `Benchmark`

**🎯 论文内容**

提出了 Select‑TTT，一种在测试时训练中根据用户查询对快权重写入进行选择性调度的机制，以提升长上下文问答性能。

**💡 创新点**

创新点在于利用冻结输出头的查询交叉熵作为无监督的相关性信号，对每个块的写入步长进行温度缩放 softmax 软加权，既保持了 In‑Place TTT 的并行性，又实现了查询条件下的自适应快权重更新。

**🔧 技术方法**

采用技术包括 In‑Place Test‑Time Training、快权重写入、块级并行推理、查询‑CE 相关性评分、top‑k 均值聚合、温度缩放 softmax、块级对数求和等。

**📊 数据集**

使用的数据集包括 LongMemEval‑S、LoCoMo 两个长上下文 QA 基准，以及用于持续预训练的 LongCite‑45k。

**📈 对比分析**

方法通过与原始零样本基线、统一写入 IP‑TTT 以及持续预训练（CPT）进行对比实验；在 Llama‑3.1‑8B 与 Qwen3‑4B 两种 8B 级模型上，Select‑TTT 在零样本和持续训练场景下平均提升约3–4%（某些子任务提升更大），尤其在多证据检索和跨会话推理任务中表现突出。

**⚠️ 局限性**

局限性包括：仅使用冻结输出头的查询‑CE 作为相关性估计，可能受模型校准影响；固定块大小和未探索学习型相关性函数；实验仅覆盖英文 8B 规模模型，缺乏多语言、跨模态和更大模型的验证。

---

## 267. Spatial-Frequency Learning for Scanning Transmission Electron Microscopy Image Enhancement

**arXiv ID:** 2610.36487 | [PDF](https://arxiv.org/pdf/2610.36487v1)

**作者:** Hesong Li `[一作]` (Beijing Institute of Technology), Tao Zhang `[通讯]` (Hangzhou Dianzi University)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Data Synthesis` `Convolutional Neural Network` `Transformer` `Diffusion model` `Contrastive Learning` `Auto Encoder` `Generative Adversarial Network` `Supervised Fine-Tuning` `Image` `Biomedical Data` `Computed Tomography` `Physics Related`

**🎯 论文内容**

提出了一种空间‑频率融合网络（SFFN）用于扫描透射电子显微镜（STEM）图像去噪与增强，并构建了更真实、多样的合成数据集 AtoMix。

**💡 创新点**

创新点包括：① 在频域使用坐标引导卷积区分不同频率位置，显著提升周期性原子信号的恢复；② 设计可学习的空间‑频率融合块，实现空间域与频域特征的灵活融合；③ 开发基于真实 STEM 图像的噪声校准方法，直接估计扫描噪声、点噪声和背景噪声参数；④ 构造包含多晶体系统、混合原子排列、不同成像模式（HAADF/BF）和复杂失真（球差、低亮度、低分辨率）的合成数据集 AtoMix。

**🔧 技术方法**

技术手段主要包括快速傅里叶变换（FFT/IRFFT）、坐标引导频域卷积（CGFC）、残差块、可学习的跨域融合、随机 Perlin 噪声生成、噪声校准与合成、PyTorch 实现与 Adam 优化，及与多种传统和深度学习方法的对比实验。

**📊 数据集**

使用的主要数据集有：原有的 TEMImageNet、SFIN 数据集；自研的 AtoMix 数据集（涵盖 46 种晶体单元、7 个晶系、混合排列、HAADF/BF 模式、球差、低亮度、低分辨率等多样化失真）；以及公开的真实先进材料 STEM 图像（超导材料、固态电池、纳米器件、神经形态材料等）。

**📈 对比分析**

通过 PSNR、SSIM、IoU 等指标在合成数据和真实数据上进行系统对比。SFFN 在 AtoMix 上取得 PSNR≈32.7 dB、SSIM≈0.921，优于 SFIN、FSNet、GAN 等方法，提升约 1–2 dB；在真实材料图像中恢复原子形态更清晰、定位 IoU 最高，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：合成数据与真实噪声仍存在一定差距，影响极端低剂量场景的泛化；模型对不同成像模式（尤其是低亮度/低分辨率）适应性仍有限；未对更高维度数据（如 4D‑STEM）或动态注意机制进行扩展；缺乏针对不同材料类型的跨域泛化评估。

---

## 268. Total Deformation-based MPM for Perfectly Elastic Solid Simulation with Strong Deformations

**arXiv ID:** 2610.36945 | [PDF](https://arxiv.org/pdf/2610.36945v1)

**作者:** Timna Böttcher `[一作]` (RWTH Aachen University), Jan Bender `[通讯]` (RWTH Aachen University)

**关键词:** `Graphics` `Physics Related` `Diffusion model` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation` `Mesh` `Point Cloud`

**🎯 论文内容**

提出了一种基于材料点方法（MPM）的全新求解器，能够在强变形下保持完全弹性物体的完整性，并兼容传统的网格碰撞处理。

**💡 创新点**

创新点在于：①使用参考格点直接计算每个粒子的变形梯度，从而避免数值断裂；②在当前网格上计算弹性力，使碰撞处理与标准MPM一致；③加入零能量模式抑制，消除非线性变形导致的伪能量问题；④实现粒子级别的碰撞和动画处理，扩展了MPM的适用范围。

**🔧 技术方法**

技术实现主要包括：MPM与TLMPM的基础框架、参考格点与当前格点的双重网格转移、基于参考格点的变形梯度计算、弹性力和零能量抑制势能的推导与实现、显式时间积分、基于网格的碰撞投影与粒子级别碰撞检测。

**📊 数据集**

使用了标准的物理材料模型（Neo‑Hookean）和 Stanford Bunny 模型作为测试数据集；实验场景包括立方体扭转、立方体坠落、Bunny 的多种初始变形以及多 Bunny 下落穿越障碍箱的碰撞演示。

**📈 对比分析**

与传统 MPM 和 TLMPM 进行对比：传统 MPM 在强变形下出现数值断裂；TLMPM 失去自然碰撞响应且在扭转时产生自逆。我们的求解器能够完整保留物体形状，零能量抑制显著减少伪能量。性能上，时间步长取 1 ms，平均每步耗时约 72.5 ms（比 MPM 的 26.3 ms 约 3 倍，TLMPM 22.7 ms 相近），主要因额外的网格转移导致。

**⚠️ 局限性**

局限性包括：①显式积分，稳定性受限，需较小时间步；②计算量大，六次网格转移导致性能开销；③不同 MPM 对象间的碰撞仍易粘连；④目前仅支持完全弹性，未实现弹塑性；⑤零能量抑制会略增材料刚度，需手动调节。

---

## 269. CrowdCC: Jointly Benchmarking Corruption Robustness and Deployment Quantization in Crowd Counting

**arXiv ID:** 2610.36340 | [PDF](https://arxiv.org/pdf/2610.36340v1)

**作者:** S Mahmudul Hasan `[一作]` (Tulane University), Lu Peng `[通讯]` (Tulane University)

**关键词:** `Computer Vision and Pattern Recognition` `Crowd Counting` `Computational Efficiency` `Benchmark` `TensorRT` `Quantization` `Image` `Benchmark`

**🎯 论文内容**

提出了一种控制腐败基准，评估在边缘设备上部署的人群计数模型在输入退化和模型量化下的表现。

**💡 创新点**

创新点在于将腐败鲁棒性和部署量化的评估结合起来，使用23种与部署相关的腐败类型进行联合评估。

**🔧 技术方法**

使用了TensorRT引擎进行FP32、FP16和后训练INT8精度的推理，并测量了相应的吞吐量和能耗。

**📊 数据集**

使用了四个数据集，包括ShanghaiTech A/B、UCF-QNRF和JHU-Crowd++，评估了26个模型。

**📈 对比分析**

通过比较不同精度下的模型表现，发现清晰准确度不能完全预测腐败性能，INT8的鲁棒性变化显著，且不一定提高延迟或能效。

**⚠️ 局限性**

限制在于未测试组合腐败（如低光、运动模糊和压缩），且模型排名和精度效应的转移到自然退化捕获尚未建立。

---

## 270. Decoupling Procedural and Outcome Fairness in Seed Allocation for Influence Maximization

**arXiv ID:** 2610.37112 | [PDF](https://arxiv.org/pdf/2610.37112v1)

**作者:** Xiaojing Du `[一作]` (Adelaide University), Feiyu Yang `[通讯]` (Civil Aviation Flight University of China)

**关键词:** `Social and Information Networks` `Optimization` `Federated Learning` `Explainability and Interpretability` `Computational Efficiency` `Reinforcement Learning from Human Feedback` `Graph Neural Network` `Contrastive Learning` `Integer Programming` `Random Block Model Analysis` `Influence Maximization` `Graph` `Tabular` `Benchmark`

**🎯 论文内容**

论文提出一种新的公平影响最大化算法 DPA（Decoupled Procedural Allocation），通过将群体配额与种子选择分离，实现对社交网络中传播导致的结果差距的量化与最小化。

**💡 创新点**

创新点在于：①给出了图独立种子分配下的期望结果差距的精确分解和连续下界；②基于该分解设计闭式配额规则与在实际图上进行的群内交换优化；③提供了可验证的总收益下界，保证在任何可行配额下都能接近最优。

**🔧 技术方法**

主要技术包括：线性邻域曝光模型（Count 与 Fraction）、随机块模型分析、闭式配额求解、基于贡献的群内排序与交换、整数规划上界扫描，以及独立级联 (IC) 模型的实验验证。

**📊 数据集**

使用的数据集包括：合成的 Stochastic Block Model (SBM) 与 Barabási–Albert (BA) 网络；真实网络如 BlogCatalog、Flickr、Amherst41、Pokec 等，其中 Pokec 提供了约 163 万用户的性别属性。

**📈 对比分析**

与现有九种基准（Greedy‑IM、Proportional、Maximin、Welfare、FIMM、LP‑Greedy、IGM‑AGM 及其配额约束版本）以及 DPA 的三种实现（g、r、e）进行对比。实验表明，DPA 在所有评估图和设定下均能实现最小平均结果差距（< 0.0004），并且在保持总收益 95–99% 的同时满足访问容差约束。

**⚠️ 局限性**

局限性包括：①理论分析主要针对两组情况，扩展到多组仍需进一步研究；②依赖于已知或可估计的传播系数 β、γ，若实际传播效应未知则效果可能下降；③在极大规模网络中尽管 DPA 计算量低，但仍需在极稀疏或极密集图上验证其稳健性。

---

## 271. Provenance Signatures for Low Rank Adaptations of Image Diffusion Models

**arXiv ID:** 2610.36933 | [PDF](https://arxiv.org/pdf/2610.36933v1)

**作者:** Tobias Cheung `[一作]` (University of Edinburgh), Rik Sarkar `[通讯]` (University of Edinburgh)

**关键词:** `Machine Learning` `Generation` `Anomaly Detection` `Explainability and Interpretability` `Transformer` `Diffusion model` `Wasserstein distance` `Score-based Model` `Contrastive Learning` `Auto Encoder` `Image`

**🎯 论文内容**

本文提出一种黑盒模型来源鉴别协议，利用生成图像的局部内在维度直方图作为签名，结合Wasserstein距离与假设检验，区分基础扩散模型、其LoRA适配以及完全不同的模型。

**💡 创新点**

创新点在于：①首次使用内在维度直方图作为模型签名；②在开放世界场景下，利用假设检验和Wasserstein距离实现对LoRA适配的高效、无标注数据的鉴别；③将该方法与传统需要标注数据的来源鉴别方法对比，展示其优势。

**🔧 技术方法**

核心技术包括扩散模型、LoRA适配、MADA内在维度估计、Wasserstein距离、统计假设检验；同时使用VAE编码、无监督聚类等辅助技术。

**📊 数据集**

基准模型为Stable Diffusion v1.5；LoRA适配分别在 Food101、Pokémon、WikiArt、CheXpert、EuroSAT 等数据集上训练；负样本使用 DeciDiffusion、PixArt‑α‑512、Stable Diffusion v2.1、SDXL、v3.5 medium 等模型。

**📈 对比分析**

在问题 I（负样本鉴别）上，方法得到 AUROC 0.855、F1 0.800；在问题 II（基准模型与 LoRA 区分）上，AUROC 1.000、F1 0.943，均优于传统基线（Nataraj19、Wang20、Giudice21、Ha26、PCA‑64 Wasserstein）。

**⚠️ 局限性**

局限性包括：对高 LoRA 规模或与基准域差异较大的适配识别效果下降；方法目前只针对 LoRA 适配，未验证对全微调或其他适配方法的适用性；在极端噪声或极端输入场景下的鲁棒性尚待进一步研究。

---

## 272. Weakness Localization Is Not Enough: State-Aware Recursive Knowledge Improvement for Large Language Models

**arXiv ID:** 2610.36811 | [PDF](https://arxiv.org/pdf/2610.36811v1)

**作者:** Runze Jiang `[一作]` (Peking University), Longtao Huang `[通讯]` (Alibaba Group)

**关键词:** `Computation and Language` `Knowledge Distillation` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Graph Neural Network` `Text` `Graph` `Benchmark`

**🎯 论文内容**

提出并实现了一种递归知识改进（Recursive Knowledge Improvement, RKI）框架，用闭环的诊断评估、知识状态估计与针对性干预循环来持续提升大语言模型的知识表现。

**💡 创新点**

创新点在于识别并解决“知识状态别名”问题，即同一弱点可能对应多种操作状态，RKI通过图结构知识单元统一诊断与干预，并在每轮更新后重新估计模型的操作知识状态，从而实现更精细的干预选择。

**🔧 技术方法**

主要技术包括：图结构知识单元构建、诊断评估得到弱点与状态指标、状态估计函数Φ、基于状态的干预策略Ψ（可用性、可访问性、可使用性），以及循环更新与评估的闭环机制。

**📊 数据集**

实验使用 HotpotEval（公开多跳问答基准）和 SafeInternalEval（企业安全合规内部基准），在两者上统一使用 Qwen3-4B-Instruct-2507 作为模型骨干，DeepSeek-V4-Pro 构建知识图谱。

**📈 对比分析**

对比 GraphGen、Self-QA、EntiGraph 等现有知识整合方法，RKI 在 HotpotEval 上平均提升 5.90 分，在 SafeInternalEval 上提升 30+ 分，总体平均提升 18.41 分，明显优于最强基线。

**⚠️ 局限性**

局限性包括：状态估计仅基于诊断证据和干预历史，未恢复模型真实隐含知识；循环评估和干预增加计算与评估成本，需要进一步优化效率与状态表示。

---

## 273. OpenJaws: An Open Hardware UMI Framework with Stereo Localization for Generalist Manipulation

**arXiv ID:** 2610.36598 | [PDF](https://arxiv.org/pdf/2610.36598v1)

**作者:** Adithyavairavan Murali `[一作]`, Stan Birchfield `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Autonomous Driving` `Optimization` `Reinforcement Learning from Human Feedback` `Transformer` `Convolutional Neural Network` `Recurrent Neural Network` `Diffusion model` `Simultaneous Localization and Mapping` `Vision-Language-Action Model` `Agentic AI` `Contrastive Learning` `Auto Encoder` `Image` `Video` `Point Cloud` `Mesh` `Tabular` `Benchmark`

**🎯 论文内容**

构建了一个完全开源、基于立体视觉的 Universal Manipulation Interface（UMI）抓手与硬件框架，并公开发布了用于评估抓取数据采集设备定位精度的 UMI‑Loc‑Bench 数据集；同时在真实 UR10 机器人上训练并部署了基于扩散策略（Diffusion Policy）的抓取与堆叠任务，验证了系统的实用性。

**💡 创新点**

其创新点包括：①采用 ISO 9409‑1‑50‑4‑M6 标准接口实现抓手与协作机器人无缝对接；②用低成本立体相机取代传统单目+IMU 或闭源跟踪，实现绝对尺度的精确定位；③首次公开立体定位基准 UMI‑Loc‑Bench，系统性比较了 COLMAP、ORB‑SLAM3、cuVSLAM、ViPE 等多种定位算法；④将真实采集数据转化为数字孪生环境，展示了 agentic real‑to‑sim 的可行性。

**🔧 技术方法**

技术手段包括：3D 打印模块化抓手结构、Dynamixel 动作器与 Arduino 微控制器、全局快门鱼眼立体相机、SAM 分割掩码、COLMAP、ORB‑SLAM3、cuVSLAM 视觉 SLAM 与 SfM、ViPE 深度学习定位、Diffusion Policy 控制器，以及 GPT‑6 Astra 生成数字孪生。

**📊 数据集**

使用了自研的 UMI‑Loc‑Bench‑v1（包含 10 个多任务、数百帧立体记录）、200 条堆叠、50 条抓取演示数据；以及带 AprilTag 的基准立方体用于获取绝对尺度的 ground‑truth。

**📈 对比分析**

通过对比实验，立体定位在全轨迹覆盖率、运行时和误差上显著优于单目+IMU。COLMAP 在全轨迹上取得 5.2 mm 平均位移误差和 0.8° 旋转误差，cuVSLAM 以 9.9 mm/2.5° 的误差在 34 s 以内完成；单目方法误差在 48 mm 左右。使用立体定位训练的策略在 UR10 上实现 90% 的成功率；数字孪生成功率为 70%。

**⚠️ 局限性**

局限性包括：需要精确的相机标定与基线校准；单目+IMU 方案易失效且需额外映射；当前仅验证单臂任务，缺乏双臂或更复杂动态场景的评估；数据集规模有限，可能限制模型对更大多样性的泛化；抓取时仍可能出现抓取滑移或碰撞问题。

---

## 274. A Smart Instant Book Filter Model based on Dual-Objective Optimization

**arXiv ID:** 2610.36287 | [PDF](https://arxiv.org/pdf/2610.36287v1)

**作者:** Yi Hou `[一作]` (Instant Book), Huizhong Duan `[通讯]` (Search Ranking)

**关键词:** `Artificial Intelligence` `Optimization` `Recommendation System` `Reinforcement Learning` `Prompt Engineering` `Chain-of-Thought` `Tabular` `Time Series`

**🎯 论文内容**

提出并部署了一种智能即时预订过滤系统，利用双目标优化最大化即时预订率同时保持整体预订率不下降。

**💡 创新点**

创新点在于将即时预订折扣累计收益（DCG）作为代理函数，用随机游走参数搜索（带重启）在决策树阈值空间寻找最优阈值，形成双目标优化的学习框架，并通过A/B测试验证效果。

**🔧 技术方法**

技术包括：确定性决策树模型、折扣累计收益代理、随机游走参数搜索、离线侧面排名仿真（SxS ranker）以及在线A/B实验。

**📊 数据集**

使用了Airbnb实时搜索与预订数据，包括数千次搜索迭代、即时预订与请求预订的交易记录，评估指标为即时预订率和整体预订率。

**📈 对比分析**

与六组人工专家设定的阈值进行A/B对比，实验结果显示即时预订率提升8.20%，整体预订率保持0%下降，且主机即时预订采用率提升近1%。

**⚠️ 局限性**

局限在于模型仅使用静态阈值的确定性决策树，缺乏对更复杂特征交互的建模；代理函数的准确性依赖于DCG近似；需要在线学习与自动化机器学习的进一步改进。

---

## 275. Advancing global crop simulations with differentiable modelling

**arXiv ID:** 2610.36827 | [PDF](https://arxiv.org/pdf/2610.36827v1)

**作者:** Yunan Lin `[一作]` (Technical University Of Munich), Niklas Boers `[通讯]` (Technical University Of Munich)

**关键词:** `Computational Engineering, Finance, and Science` `Optimization` `Agriculture Related` `Auto Encoder` `Contrastive Learning` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Supervised Fine-Tuning` `Tabular` `Time Series` `Sequential` `Agriculture Related`

**🎯 论文内容**

开发了可微分的全球作物模拟框架 Agrocosm，利用自动微分对 LPJmL 过程模型的关键参数进行梯度优化，从而显著提升作物产量的模拟精度。

**💡 创新点**

创新点在于首次实现全流程可微分的 GGCM，能够基于日尺度碳水通量观测进行参数校准，并将校准结果推广到全球历史与未来产量预测、极端事件归因以及灌溉策略优化。

**🔧 技术方法**

采用的技术包括 Julia 实现的 LPJmL 改造、Enzyme.jl 自动微分、Adam 梯度优化、逆向自动微分计算作物产量对气候输入的敏感性、参数自举与方差分解。

**📊 数据集**

使用的数据集包括 FLUXNET 日尺度 GPP、RECO、ET 观测；ISIMIP3 气候与管理输入；GDHY 历史产量数据库；欧洲 EU 子国家 2018 年作物产量统计；以及 CMIP6 SSP1/5 气候模型。

**📈 对比分析**

与 AgMIP GGCMI Phase 3 模型比较时，校准后的 Agrocosm 在四种主要作物的历史相关性和 RMSE 上位列前列；在未来 SSP5‑8.5 场景下，参数优化显著降低稻米的预测增幅，并通过参数自举显示参数不确定性在稻米产量波动中的主导作用。

**⚠️ 局限性**

局限性包括观测数据分布不均、仅使用单一参数集合、未考虑管理、病虫害等因素，以及极端事件归因仅适用于模型能充分模拟的情景。

---

## 276. RankFormer: Rethinking Time Series Transformers through Representation Capacity Analysis and Progressive Decoding

**arXiv ID:** 2610.36703 | [PDF](https://arxiv.org/pdf/2610.36703v1)

**作者:** Peiwang Tang `[一作]` (University of Science and Technology of China), Defu Lian `[通讯]` (University of Science and Technology of China)

**关键词:** `Machine Learning` `Time Series Forecasting` `Representation Learning` `Computational Efficiency` `Transformer` `Contrastive Learning` `Auto Encoder` `Prompt Engineering` `Mixture of Experts` `Time Series` `Tabular` `Benchmark` `Finance Related`

**🎯 论文内容**

提出了一种新的多变量时间序列预测框架 RankFormer，结合编码器-解码器结构和逐步残差细化。

**💡 创新点**

创新点包括基于矩阵秩的分析框架、通过 Patchification 与 MLP 嵌入突破线性秩瓶颈、Hourglass Attention 维护秩稳定性以及 Interactive Progressive Decoder 通过全局交互和递进细化提升多通道预测。

**🔧 技术方法**

核心技术为 Transformer、秩分析、Mlp 嵌入、Hourglass Attention、交叉通道条件、递进残差细化与离线折叠推理。

**📊 数据集**

在九个公开基准数据集上评估：Electricity、Traffic、Weather、Exchange‑Rate、Solar、ETTh1、ETTh2、ETTm1、ETTm2。

**📈 对比分析**

与 9 种最新 SOTA 模型（SEER、iTransformer、PatchTST、Fredformer、DLinear、SRSNet、DUET、xPatch、Amplifier）在 96/192/336/720 预测步长下进行对比，RankFormer 在 7/8 数据集上均获得最佳 MSE/MAE，累计 14/16 项指标排名第一。

**⚠️ 局限性**

局限性在于架构更复杂、对秩约束和深度敏感，需要更多计算资源；此外仅在固定长度输入/短期（≤720 步）下验证，尚未探讨极长时间序列或缺失数据情境。

---

## 277. Uniformity Testing for the Monomer-Dimer Model

**arXiv ID:** 2610.36630 | [PDF](https://arxiv.org/pdf/2610.36630v1)

**作者:** Joy Qiping Yang `[一作]` (University of Sydney), Yuichi Yoshida `[通讯]` (National Institute of Informatics)

**关键词:** `Data Structures and Algorithms` `Optimization` `Computational Efficiency` `Anomaly Detection` `Contrastive Learning` `Approximate Counting` `Entropy Estimation` `Poincaré Inequality` `Chebyshev Inequality` `Graph` `Review/Survey Paper` `Physics Related`

**🎯 论文内容**

在已知图上，对单体-双体模型（匹配分布）的未知权重向量做均匀性检验，即判断给定分布是否与所有匹配的均匀分布相同或在总变差距离上至少为ε。

**💡 创新点**

证明了在最大度数上界Δ的图中，检测均匀性所需样本数为Θ(√m/ε²)（其中m为边数），并给出了匹配计数的近似多项式时间预处理；同时给出了匹配结构下的匹配计数下界，证明了√m/ε²是最优的，从而揭示了学习全权重向量（O(m/ε²)样本）与仅检验均匀性（O(√m/ε²)样本）的显著分离。

**🔧 技术方法**

利用了：1）匹配的硬排斥约束导致的局部条件独立性，构造可观测的统计量ψ_e；2）在线图上的近似熵张量化和谱独立性证明了相对熵的可分解；3）Poincaré不等式给出协方差上界；4）对匹配计数的FPTAS（或Jerrum‑Sinclair算法）用于预估边的可用性概率；5）基于二次统计量S=〈Ψ^{(1)},Ψ^{(2)}〉的矩检验与Chebyshev不等式。

**📊 数据集**

本研究为理论性工作，未使用具体实验数据集，仅在构造的稀疏/稠密图族上进行信息论下界证明。

**📈 对比分析**

由于是理论分析，没有实验比较；算法在最坏情况下的样本复杂度为O_Δ(√m/ε²)，时间复杂度为多项式但包含指数常数依赖于Δ，显著低于学习全权重所需的O(m/ε²)样本。

**⚠️ 局限性**

局限性包括：仅适用于最大度数上界Δ的图；算法常数与Δ呈指数增长；预处理需要对匹配计数的FPTAS，实际实现难度大；未考虑图结构更一般的情况（如不均匀度数或其他图模型）。

---

## 278. HC2Robot: Hyper-Connection Transformers with Predicted Foresight for Multi-Terrain Humanoid Parkour

**arXiv ID:** 2610.36825 | [PDF](https://arxiv.org/pdf/2610.36825v1)

**作者:** Wenhao Xu `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `World Model` `Future Motion Prediction` `Adversarial Motion Prior` `Heterogeneous Actor-Critic Training` `Multi-Stream Routing` `Simulation Data` `Time Series`

**🎯 论文内容**

提出了HC2Robot框架，结合超连结Transformer和预测前瞻模块实现仿真无信息不对称的仿人机器人跑酷控制。

**💡 创新点**

创新点在于将特权信息通过预测前瞻模块学习为可部署的隐状态，并利用多流超连结Transformer提升时间特征重用，同时使用事件时间学习而非固定时间预测。

**🔧 技术方法**

采用异构演员‑评论家训练、超连结Transformer、多流路由、预测前瞻模块、未来运动预测以及对抗运动先验等技术。

**📊 数据集**

使用IsaacLab模拟环境生成的随机楼梯、缝隙、高平台、斜坡等四类地形进行训练；随后在Unitree G1机器人上进行物理验证。

**📈 对比分析**

与Hiking‑in‑the‑Wild和ParkourFormer对比，HC2Robot在四类地形中的平均成功率为97.6%、安全脚步率为97.94%，均显著优于基线方法。

**⚠️ 局限性**

局限在于对极端动态或崎岖地形的泛化尚未充分验证，且在在线适应性和计算开销方面仍有提升空间。

---

## 279. What Victims Take Away: Attack Reuse and Defense Revision after Prompt-Injection Attacks

**arXiv ID:** 2610.36422 | [PDF](https://arxiv.org/pdf/2610.36422v1)

**作者:** Zihan Chen `[一作]`, Di Zhu `[通讯]`

**关键词:** `Cryptography and Security` `Safty and Privacy` `Adversarial Attack` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

在Tensor Trust这款以大型语言模型为核心的在线游戏中，研究者收集并分析了玩家的攻击日志，探讨了攻击历史对玩家防御修复与攻击复用的影响。通过比较玩家收到的攻击文本与同一时间在同类玩家间流通但未被接收的文本，评估了被攻击者是否会将成功的攻击文本再次用于攻击他人。

**💡 创新点**

① 首次在实验环境中系统性检验受害者-攻击者路径，发现攻击复用仅在攻击成功时显著增加；② 发现攻击日志对被攻击者既可作为修复工具，也会被复制为攻击载体；③ 说明复用主要集中在已积极攻击的玩家，而非新手；④ 给出针对AI安全产品、红队报告和漏洞数据库的四项可验证的设计假设。

**🔧 技术方法**

使用基于事件的线性概率模型（包含事件固定效应、文本受欢迎度、文本固定效应和目标防御固定效应）对复用行为进行计量；对攻击文本进行变体族归类，采用五元组特征、MinHash Jaccard相似度阈值0.6；对结果进行Bootstrap和Poisson模型验证；对突破事件实施差分法评估防御更改后同族攻击成功率的下降。

**📊 数据集**

Tensor Trust原始数据库（2023年6月-2024年2月）：共563,349条攻击记录，其中415,644条为对其他玩家的攻击；12,531条突破事件；覆盖3,244名攻击者与4,415名被攻击者；记录包括攻击文本、模型回复、攻击结果、转账金额以及防御配置等信息。

**📈 对比分析**

比较方法：每个曝光事件（玩家首次收到某攻击文本）与同一时间段内向同类目标发出的至多五条“可比”文本进行配对；通过事件固定效应控制目标和时间的混杂；采用文本固定效应与受欢迎度控制对比。性能结果显示：① 所有接收文本的复用率约2-6%，而匹配文本仅约0.03%；② 成功接收文本的复用率比失败文本高约8-10个百分点；③ 泄露文本的复用率高于失败，近乎失误（near‑miss）复用率与失败无显著差异；④ 破坏后防御更改后，同族攻击成功率下降约10-20个百分点。

**⚠️ 局限性**

局限性：① 研究基于游戏环境，玩家自选且投入为虚拟货币，真实世界攻击动机与回报可能不同；② 观察数据缺乏随机化，攻击日志是否被实际查看（presence proxy）无法完全确认；③ 只涉及三种2023年版LLM，模型更新与不同安全机制可能改变结果；④ 对攻击文本的族化归类依赖于相似度阈值，可能无法捕捉技术演化细节；⑤ 未探究近似成功文本的后续修改和策略迁移。

---

## 280. Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs

**arXiv ID:** 2610.36353 | [PDF](https://arxiv.org/pdf/2610.36353v1)

**作者:** Tiejin Chen `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Anomaly Detection` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Text`

**🎯 论文内容**

研究了在大型语言模型中先检测问题的模糊性，然后利用标签保持、低容量、语义无关的扰动进行检测并驱动澄清策略。

**💡 创新点**

创新点是提出了“标签保持、低容量、语义无关”的输入扰动原则，并实现了黑盒规则库与白盒重要性引导的双版本检测器。

**🔧 技术方法**

采用了多样采样、EigV(Agree)语义一致性度量、白盒梯度/DiffMask/LogitLens重要性评分、规则库词替换与短语前缀等技术。

**📊 数据集**

使用了 ABG‑CoQA、AmbigQA‑w‑Ctx、AQuA、VQ‑FA 等四个具备模糊标注的英文问答数据集。

**📈 对比分析**

与直接提示、EigV、语义熵等输出侧不确定性基线相比，黑盒/白盒扰动检测在所有模型与数据集上均取得最高或第二高的AUROC，平均提升可达0.07。

**⚠️ 局限性**

限制在于黑盒方法依赖手工设计的英文规则库，难以迁移到其他语言；白盒方法需要模型内部可访问权限，适用范围受限。

---

## 281. Layerwise Repacking: Variable-Length Hybrid-Attention Training without an Extra Redistribution Pass

**arXiv ID:** 2610.35989 | [PDF](https://arxiv.org/pdf/2610.35989v1)

**作者:** Jiahao Zhang `[一作]` (KlingAI Research), Chengru Song `[通讯]` (KlingAI Research)

**关键词:** `Machine Learning` `Computational Efficiency` `Transformer` `Mixture of Experts`

**🎯 论文内容**

在变长混合注意力训练中，提出层级重打包（Layerwise Repacking）方案，将全注意力层保持宽并行组，稀疏层则在节点内局部组内执行，并把层间布局转换合并到原有的 Ulysses all-to-all 通信中，避免额外的重分发阶段。

**💡 创新点**

创新点在于（1）在保持全注意力宽并行的同时，将稀疏层的样本重新打包到节点局部子组，显著降低跨节点通信；（2）通过在 Ulysses 通信中插入可变拆分和逆置置换，实现无额外通信的布局切换；（3）设计辅助父级求解器，对父组成员重新分配，以进一步提升子组内部负载均衡。

**🔧 技术方法**

使用的技术包括：纯 Ulysses（all-to-all）序列并行、可变拆分与逆置换、层级重打包、辅助父级求解器（基于已校准的多模态成本模型），以及 8 GPU 节点的混合并行架构。

**📊 数据集**

在 64 GPU（8 节点）环境下训练一个 1F3S 混合注意力模型（每 3 层稀疏层配一层全注意力层），收集前 200 步训练数据，其中 22 步出现宽父组。

**📈 对比分析**

对比四种配置（基线无重打包、基线有重打包、改进求解器无重打包、完整系统），完整系统将平均端到端时间从 37.221 s 减少到 34.390 s（7.61 %），MFU 从 56.47 % 提升到 61.30 %（4.83 %）。在 22 个宽父组步骤中，重打包单独实现 5.95 % 或 4.17 % 的时间下降。

**⚠️ 局限性**

主要限制包括：SP8 子组上限导致极长样本在某些步骤无法充分利用子组并导致性能退化；评估仅覆盖单一 1F3S 模型与单一集群，未验证对更大或不同网络架构的适用性；重打包和求解器需要准确的成本预测，且在内存峰值上有潜在风险。

---

## 282. ObjectTwin: Measuring the Asset Correspondence Gap for Capture-Anchored Digital Twins

**arXiv ID:** 2610.35904 | [PDF](https://arxiv.org/pdf/2610.35904v1)

**作者:** Zhenghe Guo `[一作]` (Zhejiang University), Zhenghe Guo `[通讯]` (Zhejiang University)

**关键词:** `Computer Vision and Pattern Recognition` `Autonomous Driving` `Optimization` `Federated Learning` `Computational Efficiency` `Representation Learning` `Data-Centric Learning` `Robotic Intelligence` `Transformer` `Neural Radiance Field` `Diffusion model` `Auto Encoder` `Contrastive Learning` `Generative Adversarial Network` `Gaussian Splatting` `Optical Flow` `Image` `Video` `Point Cloud` `Mesh` `Graph`

**🎯 论文内容**

做了什么

**💡 创新点**

创新点是什么

**🔧 技术方法**

用了什么技术

**📊 数据集**

用了什么数据集

**📈 对比分析**

如何比较的方法，性能怎么样

**⚠️ 局限性**

limitation是什么

---

## 283. Evaluating AlphaEarth Embeddings for Crop Classification: Separability, Temporal Dynamics, and Cross-Regional Transferability

**arXiv ID:** 2610.36857 | [PDF](https://arxiv.org/pdf/2610.36857v1)

**作者:** Priyanka Goyal `[一作]` (Indian Institute of Technology Kharagpur), Manjira Sinha `[通讯]` (Tata Consultancy Services Research)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Agriculture Related` `Contrastive Learning` `Principal Component Analysis (PCA)` `Bhattacharyya Distance` `Mahalanobis Distance` `Moran's I` `CKA` `EMD` `Image` `Tabular` `Time Series` `Agriculture Related`

**🎯 论文内容**

设计并实现了一个无模型依赖的特征评估框架，比较AlphaEarth嵌入与传统光谱、NDVI、谐波特征在空间、时间、分辨率及跨地区迁移等维度上的表现。

**💡 创新点**

提出统一的评价指标集合（几何可分性、时间轨迹、空间自相关、跨区域一致性）并揭示嵌入特征在分布感知可分性与作物轮作动态捕捉上的优势，同时指出其跨区域迁移性不足。

**🔧 技术方法**

采用AlphaEarth自监督嵌入、传统光谱反射率、NDVI峰值特征、单周期谐波系数，并用PCA、Bhattacharyya、Mahalanobis、Moran's I、CKA、EMD等统计方法进行评估。

**📊 数据集**

以美国CONUS（2017–2023年）和印度北部AgriFieldNet（2021年）两大场景为基准，利用USDA Cropland Data Layer、Sentinel‑2 MSI、Google Earth Engine等公开数据。

**📈 对比分析**

通过留一县/区域交叉验证、空间块交叉验证和跨洲迁移实验对比，结果显示嵌入在同域内的分布感知可分性和轮作可分性分别高于传统特征（Bhattacharyya提升约40倍、轮作可分性比值1.784），但在跨洲迁移时表现最差（Macro‑F1仅占US内性能的约33%），谐波特征迁移更稳健。

**⚠️ 局限性**

限制包括：未对AlphaEarth嵌入做区域微调、印度数据仅覆盖单季且集中在乌塔尔邦、单周期谐波模型对多季作物可能不足，导致跨区域结论需要进一步验证。

---

## 284. ETC-Bench: A Multilingual Benchmark for Evaluation-Side Transcript Canonicalization

**arXiv ID:** 2610.36527 | [PDF](https://arxiv.org/pdf/2610.36527v1)

**作者:** Zhiheng Qian `[一作]` (Shanghai Jiao Tong University), Kai Yu `[通讯]` (AISpeech Ltd)

**关键词:** `Computation and Language` `Benchmark` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Agentic AI` `Text` `Multimodality`

**🎯 论文内容**

本文提出评估侧转录本规范化（ETC）任务，并构建了覆盖12种语言、181类现象、共1098个最小对的多语种基准ETC‑Bench。

**💡 创新点**

创新点在于把转录本规范化定义为独立任务，设计可扩展的现象分类体系，提出语义保留的 Exact‑Match Normalization Accuracy（SENA）评估指标，并推出以语义契约为导向的LLM代理技能TACS。

**🔧 技术方法**

技术手段包括传统的TN/ITN规则化器、BERTScore 语义相似度计算、以及三种LLM（GPT‑5.6‑Terra、DeepSeek‑V4‑Flash、Qwen3.8‑Flash）结合TACS实现的自适应规范化。

**📊 数据集**

使用的数据集是ETC‑Bench，包含1098个人工审校的最小对，涵盖12种语言的数值、缩写、正字法、同音替换等多样现象。

**📈 对比分析**

通过SENA指标对比，传统规则化器平均得分仅3.7–9.4%，而TACS‑Qwen在所有12语种上宏平均达68.8%，TACS‑GPT、TACS‑DeepSeek分别为43.3%和43.8%，显著优于基线。

**⚠️ 局限性**

局限性包括对同音替换的识别仍有较大挑战、评估阈值δ对结果敏感、以及基准目前仍缺乏更多语言与现象的覆盖。

---

## 285. Dynamical coarse-graining of neural-network training

**arXiv ID:** 2610.36180 | [PDF](https://arxiv.org/pdf/2610.36180v1)

**作者:** Lin-Juan Wang `[一作]` (Beihang University), Jianxiang Wang `[通讯]` (Peking University)

**通讯引用:** 2145 | [OpenAlex ID](https://openalex.org/A5101988752)

**关键词:** `Machine Learning` `Optimization` `Computational Efficiency` `Representation Learning` `Stochastic Differential Equation` `Ordinary Differential Equation`

**🎯 论文内容**

本文提出了一种神经网络训练的动态粗粒化框架，定义了学习算子并建立了相关定理，以便在较小的状态空间中预测和控制训练过程。

**💡 创新点**

创新点在于提出了学习算子的下降定理、精确闭合定理和动态粗粒化定理，为构建和评估神经网络训练的动态粗粒化模型提供了系统框架。

**🔧 技术方法**

使用了学习算子理论和动态粗粒化方法，结合了数学分析和数值模拟。

**📊 数据集**

使用了有限宽度的非线性ReLU网络作为数据集，进行了数值实验以验证理论结果。

**📈 对比分析**

通过与720参数的ReLU网络的直接训练动态进行比较，发现15个时刻的宏观方程在数值精度上能够准确预测训练输出和学习算子的演变，性能显著优于冻结NTK近似。

**⚠️ 局限性**

限制在于所提出的框架主要针对特定条件下的网络，可能无法广泛适用于所有类型的神经网络，且在处理更复杂的网络时需要进一步的研究。

---

## 286. AI-Assisted Discovery of CNOT Decompositions for Quantum Permutations on Constrained Topologies

**arXiv ID:** 2610.37045 | [PDF](https://arxiv.org/pdf/2610.37045v1)

**作者:** Daniele Lizzio Bosco `[一作]` (University of Udine), Giuseppe Serra `[通讯]` (University of Udine)

**关键词:** `Artificial Intelligence` `Optimization` `Reinforcement Learning from Human Feedback` `Drug Discovery` `Graph Neural Network` `Transformer` `Reinforcement Learning` `Mixture of Experts` `Prompt Engineering` `Diffusion model` `Score-based Model` `Optical Flow` `Graph` `Tabular` `Time Series` `Sequential` `Physics Related` `Benchmark`

**🎯 论文内容**

本文利用 AlphaCNOT 这一 AI 辅助合成工具，研究了在受限拓扑下的量子排列电路的 CNOT 合成，发现了远程换位的低门实现模式并将其推广为全局换位（Global Transpositions）算法；随后通过 AlphaCNOT 进一步揭示了“中间 CNOT 融合”（Intermediate CNOT Fusion）的现象，说明单独优化换位块无法获得的额外门数优势。

**💡 创新点**

创新点在于：1) 通过 AI 输出证明远程换位可用至多 6d-4 条 CNOT 实现，给出构造性证明；2) 基于该结论提出全局换位 deterministic 合成方法，优于传统 SWAP 路由；3) 发现并实验验证了中间 CNOT 融合现象，为量子电路编译提供了新的优化视角。

**🔧 技术方法**

使用技术包括：AlphaCNOT（基于强化学习 + MCTS 的模型驱动 CNOT 合成器）、Dijkstra 最短路径搜索（在加权 Cayley 图上求最小 CNOT 成本）、全局换位构造式、最优 SWAP 路由与全局 CNOT 合成的比较、以及小规模量子拓扑的穷举 BFS 求最优 CNOT。

**📊 数据集**

实验数据集为多种受限拓扑（如 4L、4Y、5L、5T、6L、6T、6Y、7L、7T、7Y、8F、8H、8T、8T2 等），每个拓扑随机选择 100 条目标排列（4L、4Y 采用全部 24 条），并使用 AlphaCNOT 的 best‑of‑10 方案；同时对 5 个及以下 qubit 的拓扑进行全局 BFS 求最优 CNOT。

**📈 对比分析**

比较方法：对同一目标排列，分别求最优 SWAP（3×SWAP 计数）、Global Transpositions、AlphaCNOT 以及最优 CNOT（仅对 5 qubit 以内可行）。性能结果：平均 CNOT 数从 SWAP 27.85 → GT 26.24 → AlphaCNOT 25.55；在所有测试拓扑上 AlphaCNOT 的平均 CNOT 数比 GT 低约 3.2%，比 SWAP 低约 10%。

**⚠️ 局限性**

局限性：1) 中间 CNOT 融合的理论阐释与最优性证明尚缺失；2) Global Transpositions 仍受限于“返回排列子群”的假设，无法捕捉所有可行的非排列中间变换；3) AlphaCNOT 的结果依赖随机搜索，未给出严格上界；4) 对大规模 qubit 的可扩展性尚未验证。

---

## 287. Full Collapse Is Suboptimal: Controlled Partial Neural Collapse for Long-Tailed Recognition

**arXiv ID:** 2610.36460 | [PDF](https://arxiv.org/pdf/2610.36460v1)

**作者:** Hyeontaek Hwang `[一作]` (KAIST), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Recognition` `Contrastive Learning` `Sparse Regularization` `Image`

**🎯 论文内容**

提出控制性部分神经崩塌方法，兼顾头尾类别的不同需求，避免尾类过度收敛；

**💡 创新点**

创新点在于同时调控特征维度和训练到推理的间隔：使用CASA节点级稀疏激活控制维度，ETF‑AM自适应边距调节尾类清晰度；

**🔧 技术方法**

技术包括CASA节点门控（基于实例特征的软/硬掩码）、ETF约束的固定权重框架、频率自适应边距（LDAM式）以及稀疏正则（spike‑and‑slab）；

**📊 数据集**

在四个长尾分类基准上验证：CIFAR100‑LT、ImageNet‑LT、Places‑LT、iNaturalist‑2018；

**📈 对比分析**

与多种基线（CE、LDAM、BALMS、WS、ETF等）对比，平均提升 1–2% top‑1 率，特别在 Medium/​Few 子集显著提高，且参数/ FLOPs 增幅极小；

**⚠️ 局限性**

局限在于：1）当类别数大于特征维度时无法使用简单 ETF，需手工调节 Γ；2）理论基于有限样本类条件模型，实际网络证明仍开放；3）目前仅验证于分类任务，未扩展到检测/分割等长尾场景。

---

## 288. Polynomial-Time Solvability of the Mixed Evacuation Problem

**arXiv ID:** 2610.37065 | [PDF](https://arxiv.org/pdf/2610.37065v1)

**作者:** Naoyuki Kamiyama `[一作]` (Kyoto University), Naoyuki Kamiyama `[通讯]` (Kyoto University)

**关键词:** `Data Structures and Algorithms` `Optimization` `Linear Programming` `Ellipsoid Method` `Submodular Function Minimization` `Extreme Point Theory` `Graph Theory (Reachability and Connected Components Analysis)` `Graph`

**🎯 论文内容**

证明混合疏散问题（mixed evacuation problem）在一般情况下可以在多项式时间内求解，给出了一个基于半径算法（ellipsoid method）的完整理论证明。

**💡 创新点**

创新点在于：①将混合疏散问题转化为一系列线性规划约束，并利用极点（extreme point）性质得到可行的分离器；②证明了与动态运输问题对应的子模量函数可用多项式时间内最小化；③通过双重性与极点分析，构造了多项式大小的线性不等式系统，使得问题可归约到可行性判定。

**🔧 技术方法**

主要技术包括：线性规划与其对偶性、半径算法（ellipsoid method）、子模量函数最小化、极点（extreme point）理论、图论中的可达性与连接分量分析。

**📊 数据集**

本工作为纯理论研究，无使用任何实际数据集；所有证明均基于图结构、容量、时间限制等抽象参数。

**📈 对比分析**

方法评估基于理论复杂度，证明了存在多项式大小的不等式系统和多项式时间的分离器，从而保证了问题的可解性；并未给出实验或与现有算法的性能比较。

**⚠️ 局限性**

局限性包括：①算法依赖于半径算法，虽然是多项式时间，但常数项和迭代次数可能很大，实际运行效率不一定好；②证明中使用的子模量函数最小化和极点计算在实践中实现可能较为困难；③论文未给出具体的时间上界或实现细节，缺乏对实际应用场景的验证。

---

## 289. Conservative Policy Iteration For Zero-Sum POSGs

**arXiv ID:** 2610.36847 | [PDF](https://arxiv.org/pdf/2610.36847v1)

**作者:** Erwan C. Escudie `[一作]` (University of Groningen), Jilles Steeve Dibangoye `[通讯]` (University of Groningen)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Reinforcement Learning` `Linear Programming` `Optimization` `Benchmark`

**🎯 论文内容**

提出一种保守策略迭代（Conservative Policy Iteration, CPI）算法，用于有限时域零和部分可观测随机游戏（zs-POSG）的近似规划

**💡 创新点**

创新点在于：①不需要全局价值函数逼近，而是利用当前联合策略产生的固定策略值函数作为局部备份的后续项；②通过在保守平均的占用轨迹上进行局部政策改进，聚焦于可被对手利用的区域；③将点基值迭代与策略搜索的优势结合，得到更稳健的收敛性与效率平衡

**🔧 技术方法**

核心技术包括：占用状态归约、焦点信念游戏（focal belief game）、条件占用状态与固定策略价值函数、基于线性规划的局部贝尔曼备份、GLIE探索和随机逼近理论

**📊 数据集**

使用了标准的有限时域零和POSG基准数据集：Adversarial Tiger、Recycling、Mabc 与 Matching Pennies

**📈 对比分析**

与PBVI、HSVI、CFR+等基准算法比较，CPI在短期（小时间步）下与PBVI竞争；在较长时间步下，CPI在运行时间与可利用性（exploitability）之间取得更好的折衷；在Adversarial Tiger与Recycling等更适合广泛占用空间采样的实例中，PBVI仍略优，但CPI在更大时间步和更复杂游戏中表现更稳健

**⚠️ 局限性**

局限性包括：①每一步需求解两方局部线性规划，计算成本高；②收敛速度慢、样本复杂度高，限制了大时间步的可扩展性；③当前实现依赖线性规划子程序，可能在更大规模游戏中成为瓶颈

---

## 290. Variable-Length Feedback Coding With Universal Decoding and Bandit Channel Selection

**arXiv ID:** 2610.36818 | [PDF](https://arxiv.org/pdf/2610.36818v1)

**作者:** Recep Can Yavas `[一作]` (Bilkent University), Recep Can Yavas `[通讯]` (Bilkent University)

**关键词:** `Information Theory` `Optimization` `Reinforcement Learning from Human Feedback` `Federated Learning` `Reinforcement Learning` `Contrastive Learning` `Mixture of Experts` `Physics Related`

**🎯 论文内容**

提出了一种在未知噪声方差的K个AWGN信道上进行联合学习与通信的可变长度反馈（VLF）编码方案，该方案在单一过程内完成信道选择与误码决策，显著降低通信延迟；

**💡 创新点**

将信道学习与信息传输实现耦合，提出了能够在保持误码率不变的前提下，同时自适应选择最佳信道并动态调整功率的算法；

**🔧 技术方法**

利用了信息密度与似然比的马尔可夫性质、投影估计与精度更新、Lai–Robbins bandit下界、奖励堆叠表示、切比雪夫与Chernoff界、以及一系列小尾巴与上界分析等技术；

**📊 数据集**

本研究为理论分析，未使用实测数据集，全部基于K个未知噪声方差的AWGN信道模型；

**📈 对比分析**

与传统的探索后承诺（ETC）和单独VLF方案进行比较，证明在所有K≥2的实例中，联合方案在二阶系数上更优，误码率与能量约束均得到满足；

**⚠️ 局限性**

仍缺乏最优的二阶上界，尚不清楚是否存在更低的信道探索成本；此外研究仅覆盖高斯信道，尚未推广到其他信道类型（如BSC等）。

---

## 291. Specialization, Diversity, and Scaling in Child-Scale Language Models

**arXiv ID:** 2610.36679 | [PDF](https://arxiv.org/pdf/2610.36679v1)

**作者:** Steven Y. Feng `[一作]` (Stanford University), Michael C. Frank `[通讯]` (Stanford University)

**关键词:** `Computation and Language` `Data-Centric Learning` `Representation Learning` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Contrastive Learning` `Text` `Audio`

**🎯 论文内容**

研究了在儿童规模数据下，小型GPT-2模型对不同儿童语言环境的学习效果与多样性影响。

**💡 创新点**

以固定模型、不同训练数据源和规模为变量，探究小数据量下环境特征与分布匹配对语言模型泛化的决定作用。

**🔧 技术方法**

使用GPT-2微型/小型因果语言模型、Kneser-Ney 5-gram基线以及Zorro、BLiMP、WordSim等语法和语义评测。

**📊 数据集**

BabyView自然语料（孩子家中密集录音转写）与TinyDialogues合成儿童指向对话。

**📈 对比分析**

通过相同词数的训练集、子集抽样与模型种子重复，评估持有外预测、语法测试与词义相似度；发现同环境优势主要由词汇分布驱动，混合儿童环境可显著降低持有外损失但对语法指标提升有限。

**⚠️ 局限性**

仅针对GPT-2、有限的训练规模与语料覆盖，未捕捉儿童真实多模态输入，且实验未验证对完全未见儿童的迁移。

---

## 292. Learning Human Sports Motion Feasibility via Quantile Regression on Tracking Data

**arXiv ID:** 2610.36183 | [PDF](https://arxiv.org/pdf/2610.36183v1)

**作者:** Marc Gutiérrez-Pérez `[一作]` (Institut de Robòtica i Informàtica Industrial, CSIC-UPC), Antonio Agudo `[通讯]` (Institut de Robòtica i Informàtica Industrial, CSIC-UPC)

**关键词:** `Artificial Intelligence` `Optimization` `Reinforcement Learning from Human Feedback` `Computational Efficiency` `Representation Learning` `Multi-Layer Perceptron (MLP)` `Quantile Regression` `Pinball Loss` `Regularization` `Time Series` `Tracking Data`

**🎯 论文内容**

本文提出将运动员到达任意场地位置的最短可行时间（TTA）建模为条件下分位数回归，利用低分位（1%）学习运动员的最快可达时间；

**💡 创新点**

创新点在于：①用低分位回归直接学习可达时间下界，避免传统平均或密度估计；②在损失中加入可达面积正则化，兼顾时间预测与空间覆盖；③在不同运动（足球、篮球、橄榄球）上检验跨域泛化，证明模型可迁移；

**🔧 技术方法**

技术上采用轻量化多层感知器（MLP）和基于物理的六种运动模型作为基线，使用pinball（下分位）损失加面积正则化进行训练；

**📊 数据集**

数据集为FIFA 2022世界杯全比赛（64场）光学追踪数据（30Hz），以及NBA 2015-16赛季SportVU（25Hz）和NFL Big Data Bowl 2019单场（10Hz）追踪数据；

**📈 对比分析**

与六种物理运动模型以及Anzer-Bauer的可达凸包基线在MAE、RMSE、pinball误差和匹配覆盖率下的可达面积等指标上进行对比，实验显示MLP在所有指标上均优于物理模型，且在不同距离区间保持较小误差；

**⚠️ 局限性**

局限性包括：①模型虽然在不同运动上泛化，但跨域可达面积仍不如目标运动专门调参的物理模型；②对极端速度或障碍物等情况的预测仍依赖训练数据；③没有显式保证星形或单连通性等几何属性，需要后续审计或约束；

---

## 293. Progressively Refining SAM for Accurate Phytoplankton Segmentation

**arXiv ID:** 2610.36722 | [PDF](https://arxiv.org/pdf/2610.36722v1)

**作者:** Jialin Li `[一作]`, Junyu Dong `[通讯]` (Ocean University Of China)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Transformer` `Convolutional Neural Network` `Supervised Fine-Tuning` `Contrastive Learning` `Feature Fusion` `Image`

**🎯 论文内容**

提出了基于 SAM2 的 PhySeg，专门针对浮游植物的像素级分割问题。

**💡 创新点**

创新点在于设计轻量级的递进式细化模块，利用多尺度特征融合逐步恢复细节并纠正粗略预测。

**🔧 技术方法**

使用了 SAM2+LoRA 微调、残差头、ASPP 风格特征融合以及焦点+Dice 损失等技术。

**📊 数据集**

构建了 Phytoplankton Segmentation Dataset（PhySegD），包含 9824 张图像、11990 个实例，并进行手工掩膜标注。

**📈 对比分析**

与 CascadePSP、BPR、CRM、HQ‑SAM2 等方法比较，PhySeg 在 mIoU、Dice 及 BF1/BF2 等指标上均取得最佳表现。

**⚠️ 局限性**

局限性主要体现在 BF4 边界指标提升有限，以及数据集规模仍相对有限，难以覆盖所有浮游植物种类。

---

## 294. PsyJev: Psychological Analysis via System-1 Judgment

**arXiv ID:** 2610.36689 | [PDF](https://arxiv.org/pdf/2610.36689v1)

**作者:** Fangyuan Liu `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `Artificial Intelligence` `Classification` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Review/Survey Paper`

**🎯 论文内容**

本研究提出了一个统一的心理学判断框架和对应的直接概率判定模型，能够在同一条文本上根据不同心理学问题作出不同的判断；

**💡 创新点**

创新点在于：①将13个异构心理学数据集整理为82个问题定义的判断，保留各自决策标准；②设计了系统-1风格的直接概率判定模型PsyJev，直接输出合法答案的分布；③提出了Judgment‑Anchored RLCD（JA‑RLCD）训练策略，结合类平衡监督、SFT参考保留、问组奖励以及logit‑空间探索，显著提升了跨任务性能；

**🔧 技术方法**

技术主要包括：问题条件化输入序列、合法答案的logit提取与归一化、联合有监督微调（SFT）、基于logit‑空间的强化学习（JA‑RLCD）以及路径梯度和Brier‑基问组奖励；

**📊 数据集**

使用了13个公开心理学数据集（共计 53,775 条记录，222,191 个观察判断），涵盖抑郁/焦虑、情感与关系、压力/痛苦、认知、健康与危机等六大主题；

**📈 对比分析**

在各模型规模（2B、4B、9B）下，PsyJev在平均 F1 上分别达到 64.33、66.21、67.56，均超过同一backbone的答案生成模型与多种基线；在未标注的 Dreaddit 数据集迁移上也取得 77.76/78.70 F1，说明跨数据集泛化能力；

**⚠️ 局限性**

局限性包括：①仍需大量标注数据和手工设计的问题；②对极端稀缺类别的处理有限；③模型规模较大，计算资源需求高；④仅用于判断任务，不能直接用于临床诊断或个体化治疗。

---

## 295. IoT-based Smart Real-Time Energy-Aware Server Room Monitoring System

**arXiv ID:** 2610.35873 | [PDF](https://arxiv.org/pdf/2610.35873v1)

**作者:** Muhammad Dikko Gambo `[一作]` (King Fahd University of Petroleum and Minerals), Abdallah Moubayed `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 3156 | [OpenAlex ID](https://openalex.org/A5022670644)

**关键词:** `Networking and Internet Architecture` `Anomaly Detection` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Recurrent Neural Network` `Contrastive Learning` `Auto Encoder` `SHAP (SHapley Additive exPlanations)` `Time Series` `Tabular` `Benchmark`

**🎯 论文内容**

构建了基于IoT的无代理DDS中间件的服务器机房实时能耗监测系统，结合多头LSTM实现热应力预测、冷却效率检测和功率预测。

**💡 创新点**

创新点在于将边缘传感与集中分析解耦，采用brokerless DDS实现低延迟高可靠性数据分发，并通过多任务LSTM实现同时预测多种关键指标。

**🔧 技术方法**

使用了DDS（Fast DDS）中间件、Focal Loss、BCE、MSE损失、多头LSTM网络以及SHAP解释性分析。

**📊 数据集**

使用公开的Air Cooled Tropical Data Center（TDC2.0）六个月数据集。

**📈 对比分析**

与两种QoS配置（Reliable与Best‑Effort）对比，可靠QoS平均延迟1.76 ms、吞吐量57.1 Mbps；模型在热应力任务F1=0.92、冷却效率AUC=0.76，功率预测R²>0.83。

**⚠️ 局限性**

局限性包括仅在单一数据集验证，缺乏不同机房或真实部署的泛化评估，且冷却效率检测精度仍受阈值选择影响。

---

## 296. Less is more: learning what to profile for GPU kernel optimization

**arXiv ID:** 2610.36576 | [PDF](https://arxiv.org/pdf/2610.36576v1)

**作者:** Yihan Wang `[一作]` (Renmin University of China), Wei Xu `[通讯]` (Renmin University of China)

**关键词:** `Machine Learning` `Optimization` `Computational Efficiency` `Reinforcement Learning from Human Feedback` `Neural Architecture Search` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text`

**🎯 论文内容**

提出 AutoProfiler，一种反馈驱动的框架，通过构建确定性分支树实现针对每个 GPU kernel 的自适应硬件 profiling，从而提升 LLM 驱动的 kernel 重写效果。

**💡 创新点**

创新点在于将硬件 profiling 从固定指标集合转变为可学习的、基于 kernel 的决策过程；使用 REWRITE 与 SPLIT 两种树结构改进操作，在 LLM 反馈指导下自适应生成最有效的 profiling 配置；首次证明动态 profiling 能比任何单一静态配置更优，并通过实验验证。

**🔧 技术方法**

核心技术包括：基于 LLM 的候选树结构生成与评估、确定性 profiling 树表示、序列化的 profiling 决策、对比评估（paired evaluation）以及多轮迭代优化的闭环。

**📊 数据集**

使用 CUDA‑Agent‑Ops‑6K 数据集构建树，评估则采用 KernelBench 作为无偏见的跨域基准（包含 Level 1~3 三个复杂度层次）。

**📈 对比分析**

在 KernelBench 上与 CudaForge、KernelMem、KernelAgent、AutoKernel 等训练无关基线对比，AutoProfiler 在速度提升上达到 1.704×（最高 1.352×，相对最强基线提升 8.5%），准确率 87.2%（比对手高 2.4–8.8pp），且仅一次 LLM 调用，显著降低计算开销。

**⚠️ 局限性**

局限性包括：仍依赖预先收集的 metric 集合；树结构在极大规模的 kernel 组合上可能需要进一步扩展；对极端低效或极端高效 kernel 的 profiling 选择尚未达到 oracle 层面，且在不同硬件平台上的泛化仍需进一步验证。

---

## 297. Learning atomic interactions for the thermodynamics of liquid mixtures

**arXiv ID:** 2610.36440 | [PDF](https://arxiv.org/pdf/2610.36440v1)

**作者:** Marco Hoffmann `[一作]` (RPTU Kaiserslautern-Landau), Fabian Jirasek `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `Machine Learning` `Drug Discovery` `Physics Related` `Graph Neural Network` `Contrastive Learning` `Supervised Fine-Tuning` `Graph` `Tabular`

**🎯 论文内容**

提出COGNAC，一个基于图神经网络的原子级别活动系数预测模型，将其嵌入UNIFAC框架以实现高精度且可直接在过程模拟器中使用。

**💡 创新点**

创新点在于用GNN学习每个原子及其分子邻域产生体积、表面积及相互作用参数，取代传统预定义结构群，兼具可解释性、热力学一致性并显著提升预测准确性。

**🔧 技术方法**

采用消息传递图神经网络（MPNN）结合两头输出网络预测原子几何参数和对称交换能量，训练时加入COSMO几何正则化和氢键符号正则化，并通过自动微分实现多组数据类型的损失。

**📊 数据集**

训练数据来自德国多明特数据库（DDB）的822,780条二元混合实验点（VLE、LLE、SLE、ACT、HE等），并用CHAOS数据库提供的COSMO cavity作为正则化参考。

**📈 对比分析**

通过十折交叉验证与全数据模型，COGNAC在二元、三元、四元体系以及所有可描述体系中均比mod.UNIFAC和更先进的HANNA降低13–63%误差，且在azeotrope、溶解度、相图等方面与实验一致。

**⚠️ 局限性**

局限在于尚未覆盖离子液体、电解质和聚合物等需要长程电势或自由体积效应的体系，且模型依赖于SMILES和已构建的图结构，对极端高压或极低温条件的推广仍需验证。

---

## 298. DocSkill: Skill-Integrated Agentic Reasoning for Multilingual Multi-Hop Document Question Answering

**arXiv ID:** 2610.36759 | [PDF](https://arxiv.org/pdf/2610.36759v1)

**作者:** Jing Huang `[一作]` (Ant Digital Technologies, Ant Group), Ying Yan `[通讯]` (Ant Digital Technologies, Ant Group)

**关键词:** `Computation and Language` `Retrieval` `Computational Efficiency` `Explainability and Interpretability` `Knowledge Distillation` `Representation Learning` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Agentic AI` `Mixture of Experts` `Prompt Engineering` `MCTS (Monte Carlo Tree Search)` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

提出 DocSkill 框架，利用可学习的技能编排实现多语言多跳文档问答。

**💡 创新点**

创新点在于将技能抽象为可复用的能力层级，并使用 bounded MCTS 生成轨迹监督，同时引入 MMHop-DocBench 数据集。

**🔧 技术方法**

使用多模态 LLM（Qwen3‑VL、Gemini、Claude 等）+ LoRA 微调 + DPO 偏好优化 + MCTS 轨迹构造 + OCR、检索、解析等工具调用。

**📊 数据集**

使用 MMHop-DocBench（60K QA，10 种语言，8 领域）作为主数据集，并对比 MMLongBench‑Doc、DocBench 等公开长文档 QA 基准。

**📈 对比分析**

与 VLM、OCR、RAG、文档代理等基线在 MMHop‑DocBench 上比较，DocSkill macro Acc 41.0、F1 49.2，提升 5.5/6.3 分；在 MMLongBench‑Doc 上 Acc 69.2、F1 67.3、LASJ 68.9，DocBench LASJ 86.8，均优于最强基线。

**⚠️ 局限性**

局限性包括对大规模预训练 LLM 及 GPU 资源的高度依赖，技能集合固定难以自动扩展，跨语言覆盖尚有限，未充分验证更复杂文档结构的通用性。

---

## 299. LoRA on a Budget: Function-Aware Post-Hoc Compression for Adapter Fleets

**arXiv ID:** 2610.36386 | [PDF](https://arxiv.org/pdf/2610.36386v1)

**作者:** Shenghui Li `[一作]` (Uppsala University), Thiemo voigt `[通讯]`

**关键词:** `Artificial Intelligence` `Compression` `Computational Efficiency` `Optimization` `Transformer` `Large Language Model` `Contrastive Learning` `Optimization` `Text`

**🎯 论文内容**

提出了功能感知的低秩分配方法（FRA），在LoRA适配器舰队内外两层分配保留秩，降低存储与推理成本。

**💡 创新点**

创新点在于：①使用单层截断的端到端功能风险测量构造加法搜索代理；②通过动态规划精确求解多选背包问题得到候选分配；③加入光谱补偿保持搜索稳健；④在构造的功能前沿上进行全局极大极小分配，实现两层资源分配的等价统一优化。

**🔧 技术方法**

主要技术包括：LoRA低秩更新与SVD重压缩；端到端JS散度功能风险评估；多选背包动态规划；光谱能量阈值与补偿；全局极大极小优化与前沿闭包。

**📊 数据集**

使用了三个公开LoRA舰队：LoRA Land（12个Mistral‑7B适配器）、Lots‑of‑LoRAs（25个Mistral‑7B‑Instruct‑v0.2适配器）和LoRARetriever（41个Llama‑2‑7B适配器）。

**📈 对比分析**

与均匀光谱压缩、单层光谱重分配、端到端光谱曲线、功能动态规划等方法比较。实验显示在中等至严苛预算下，FRA在保留率、最差百分位和尾部性能方面优于其他方法，尤其显著提升了尾部保留率。

**⚠️ 局限性**

局限性包括：①代理误差随候选形状变化，需光谱补偿；②依赖无标签校准提示，需额外计算；③功能风险代理（JS散度）不保证下游任务精度；④对不同LoRA结构或更大模型的推广尚未验证。

---

## 300. ActiveAlign: Aligning Egocentric Manipulation via Active Perception

**arXiv ID:** 2610.36859 | [PDF](https://arxiv.org/pdf/2610.36859v1)

**作者:** Ri-Zhao Qiu `[一作]` (NVIDIA), Erwin Coumans `[通讯]` (NVIDIA)

**关键词:** `Robotics` `Robotic Intelligence` `Depth Estimation` `Pose Estimation` `Transformer` `Diffusion model` `Neural Radiance Field` `Gaussian Splatting` `Contrastive Learning` `Vision-Language-Action Model` `Supervised Fine-Tuning` `Image` `Video` `Point Cloud` `Mesh` `Multimodality`

**🎯 论文内容**

提出并实现了 ActiveAlign 框架，通过主动感知（手势波动和触觉反馈）实现实时的相机标定与物体对齐，从而提升 Real‑to‑Sim‑to‑Real 的精度。

**💡 创新点**

创新点在于利用主动感知替代传统被动标定，结合重要采样与可微渲染实现 egocentric 相机在线标定，以及触觉高度图对齐实现毫米级精度；该方法在公开数据集上实现了无标定相机的高 IoU。

**🔧 技术方法**

使用了可微渲染、重要采样、深度估计、SAM 分割、GroundingDINO、SAM2、FoundationPose、3DGS、BundleSDF、DRRobot、Mujoco、Blender、GPT‑6‑Astra 等技术；触觉高度图通过手指触碰生成。

**📊 数据集**

采用了公开无标定机器人操纵数据集 Mobile ALOHA、ABC‑130K、Humanoid Everyday，以及自制的 Unitree G1 实验数据；同时利用 ArUco 立方体辅助标定。

**📈 对比分析**

与 DRRobot 及手工标定方法对比，ActiveAlign 在相机标定 IoU 上取得最高值；在细粒度操作任务中，结合触觉规划的成功率显著高于基于关键点的传统规划，达到数十个百分点的提升。

**⚠️ 局限性**

局限性包括：对主动动作的依赖导致需要额外的运动规划，且在高度动态或视角极端变化的场景下可能收敛慢；触觉高度图对细节的分辨率有限，且仅适用于平面或近似平面的目标；当前实现主要在单机器人或固定相机安装上验证，移动机器人标定不稳定。

---

## 301. Externalizing Hand Grip Pressure Through Real-Time Visual Feedback

**arXiv ID:** 2610.36769 | [PDF](https://arxiv.org/pdf/2610.36769v1)

**作者:** Andrew L. Huard `[一作]` (University of California Santa Barbara), Misha Sra `[通讯]` (University of California Santa Barbara)

**关键词:** `Human-Computer Interaction` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Contrastive Learning` `Auto Encoder` `Reinforcement Learning from Human Feedback` `Tabular` `Time Series` `Sequential` `Review/Survey Paper`

**🎯 论文内容**

利用配备20个FSR传感器的手套实时采集手部压力，并将压力分布可视化，帮助初学者在棒球和乒乓球握姿中精准匹配目标握力分布。

**💡 创新点**

创新点在于：①将高维压力数据聚合为功能性手部区域；②设计并比较连续梯度与离散条形两种可视化编码；③引入基于GRASP握姿的定制化传感区域映射，探究其对匹配表现和认知负荷的影响。

**🔧 技术方法**

技术手段包括：20个FSR传感器手套、Arduino Uno进行数据采集、Unity实现实时可视化（梯度与条形），以及Beam Eye Tracker进行眼动记录，NASA‑TLX自评用于工作负荷评估。

**📊 数据集**

实验数据集：28名右手受试者完成棒球和乒乓球各三种握姿，共36个实验块；记录压力读数、眼动轨迹、匹配MAE、匹配次数、时间与NASA‑TLX等。

**📈 对比分析**

采用混合设计的线性/广义线性混合模型进行比较，结果显示连续梯度可视化在匹配准确度（MAE更低）、匹配次数和眼动熵方面优于离散条形；但离散条形被部分受试者认为更直观；定制化传感区域映射降低了感知时间负荷，但对准确度无显著提升。

**⚠️ 局限性**

局限包括仅针对右手、使用未定制的棉手套导致传感位置变异、FSR易受噪声干扰、实验仅在静态握姿进行，缺乏动态运动情境以及未评估长期技能迁移效果。

---

## 302. Decoupling Consistency and Conflict across MLLM Layers for Fake News Detection

**arXiv ID:** 2610.36569 | [PDF](https://arxiv.org/pdf/2610.36569v1)

**作者:** Kai Qian `[一作]` (Soochow University), Jiajie Xu `[通讯]` (Soochow University)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Anomaly Detection` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Mixture of Experts` `Multimodality` `Text` `Image`

**🎯 论文内容**

提出一种多层级一致性与冲突解耦框架（MLCCD），通过任务相关的 token 选择、层间显著变化的 token 过滤以及可适应的路由机制，从多模态大型语言模型（MLLM）不同层提取一致性与冲突证据，用以判别多模态假新闻。

**💡 创新点**

创新点包括：①利用 MLLM 各层的内部表示差异捕捉稀疏冲突；②在 token 级别做任务相关性过滤，避免无关信息稀释冲突信号；③在层级上分离一致性与冲突证据，并通过独立路由器动态聚合，适应不同样本和语义层次的证据重要性。

**🔧 技术方法**

核心技术：LoRA 微调的 MLLM（以 Qwen3‑VL‑8B‑Instruct 为主），多模态 token 选择与层级差分过滤，Transformer 层上下文编码，交叉模态一致/冲突特征构造，双路由器聚合，二分类 MLP 预测，并加入解耦正则化。

**📊 数据集**

使用三大公开多模态假新闻数据集：Weibo（中文）、Pheme（英文）和 AMG（英文）。

**📈 对比分析**

与多种基线（传统多模态检测器 CFFN、MTS、MIMoE‑FND、DDU；通用多模态方法 DMD、DecAlign、MultiLoReFT；MLLM 基线 FKA‑Owl、直接 LoRA 微调 Gemma‑4‑E2B‑it/E4B‑it、Qwen3‑VL‑2B/8B‑Instruct）进行对比。MLCCD 在所有数据集上均获得最高准确率和宏 F1，分别提升 0.9–1.6% 的准确率和 0.9–2.3% 的宏 F1；在对抗性样本上检测率从 18% 提升至 42%；推理时间仅增加约 3 秒。

**⚠️ 局限性**

局限性：①在 Weibo 上提升幅度相对有限，原因是基线已接近上限；②性能受 token 保留比例、层级保留比例等超参影响，需要调优；③对极小冲突或事件/声明冲突的检出率仍低于对象/场景冲突；④方法依赖 MLLM 的内部层结构，迁移到不同模型或语言时可能需要重新设计。

---

## 303. SMEBENCH: Benchmarking Neural Music Editing with Synthetized MIDI Edits

**arXiv ID:** 2610.36715 | [PDF](https://arxiv.org/pdf/2610.36715v1)

**作者:** Matthieu Cervera `[一作]` (Quebec Artificial Intelligence Institute), Cem Subakan `[通讯]` (Quebec Artificial Intelligence Institute)

**关键词:** `Sound` `Benchmark` `Transformer` `Large Language Model` `Diffusion model` `Score-based Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Audio` `Text` `Multimodality`

**🎯 论文内容**

构建了SMEBench，一套带有源音频、指令、目标音频、编辑元数据和文本描述的对照式音乐编辑基准；

**💡 创新点**

通过在MIDI层面施加已知编辑并同步渲染音频，提供了可测量“保留内容”“执行编辑”“文本一致性”三种指标的对照目标；

**🔧 技术方法**

采用MIDI符号变换、FluidSynth渲染、CLAP、LPAPS、Audiobox等多种评估指标以及LLM（Qwen3）生成和校验文本；

**📊 数据集**

基于MIDICaps（≈168k MIDI）生成12种编辑类型的11,160对示例，并补充416个真实多轨录音的对照子集；

**📈 对比分析**

对比SDEdit、DDIM Inversion、ZETA、Virtual Consistency Inversion、SteerMusic和MusicGen等六个编辑系统，使用源、目标和文本三种参考分别评估，发现不同指标给出的系统排名可完全相反；

**⚠️ 局限性**

受限于合成音频与真实录音的差距、仅覆盖MIDI可表达的12种编辑、某些编辑难以听感确认以及评估指标仍无法同时衡量保留、执行和感知质量。

---

## 304. Do AI Scientists Know What's New?

**arXiv ID:** 2610.36511 | [PDF](https://arxiv.org/pdf/2610.36511v1)

**作者:** Miri Liu `[一作]` (University of Illinois), ChengXiang Zhai `[通讯]` (University of Illinois)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Data-Centric Learning` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text`

**🎯 论文内容**

提出了一种细粒度、可解释的声明级别新颖性检测模块 ClaimNov，并构建了基于时间轴的评估框架

**💡 创新点**

创新点在于：①把声明（claim）作为判定新颖性的基本单元；②给出完整的声明级新颖性档案；③通过时间真值和多维度数据集实现客观评估；④显著提升新颖性检测的灵敏度与特异性，减少误报与漏报

**🔧 技术方法**

技术方法包括：①大语言模型（Sonnet 4.6）进行声明提取、检索、蕴涵评估；②基于向量检索与余弦相似度的同维声明检索；③rank‑discounted mean与权重合成的全提案新颖性评分；④最终由更强的LLM评判给出通过/拒绝判决

**📊 数据集**

使用了四大数据集共1,060个提案：①已完成的仿真提案（380个）；②评审争议声明对（160个）；③后续实现的创新提案（400个）；④混合提案（100个），并包含对应的论文池与真实引用记录

**📈 对比分析**

与七种现有新颖性检查（包括 AI Scientist v1、CoI-Agent、Si 等）对比，ClaimNov 在全提案层面捕获 97% 的仿真复制、误报率仅 1.5%，在声明层面识别原始贡献的比例约 21%，同时拒绝率仅 3%；整体信息指标（Informational J、Δ）均为最高，性能显著优于传统方法

**⚠️ 局限性**

局限性包括：①依赖于预定义的维度权重，可能不适用于非AI领域；②仅评估固定论文池，检索能力有限；③所有提案与评判均使用同一 LLM，可能引入模型偏见；④对时间阈值敏感，后续更先进模型需重构数据；⑤未评估实时生成循环中的成本与效益

---

## 305. Whole-Body Humanoid Control via Stochastic Motion Tokenization

**arXiv ID:** 2610.36452 | [PDF](https://arxiv.org/pdf/2610.36452v1)

**作者:** Michael Drolet `[一作]` (Technical University Of Darmstadt), Jan Peters `[通讯]` (Technical University Of Darmstadt)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Auto Encoder` `Reinforcement Learning` `Diffusion model` `Auto Encoder` `Prompt Engineering` `Vision-Language-Action Model` `Video` `Text` `Sequential`

**🎯 论文内容**

本文设计并实现了一个三阶段管道：首先用 DAPS 变分自编码器将人形机器人 Unitree G1 的全身运动分词；其次用基于 Transformer 的自回归模型按行为标签或文本生成离散运动序列；最后用单一 PPO 跟踪策略在仿真和真实机器人上执行这些生成的运动，实现了 35 秒的战斗-舞蹈连续切换。

**💡 创新点**

创新点包括：① 引入可控熵的 DAPS 分词器，显著优于 FSQ/VQ‑VAE；② 将离散运动流作为上下文输入自回归生成器，实现基于标签/文本的长时序运动合成；③ 将生成的离散运动序列直接馈给低层策略，完成端到端可部署的高/低层分离框架；④ 在 Unitree G1 上实现多行为长时序切换。

**🔧 技术方法**

使用技术包括 DAPS（离散变分自编码器）、Transformer（Encoder‑Decoder 用于下一个分词块预测）、PPO（强化学习跟踪策略）、DeepMimic 风格奖励、Domain Randomization、JAX+XLA 训练框架、文本/标签上下文嵌入。

**📊 数据集**

采用公开全身运动数据集 LAFAN1 与 BONES‑SEED，已映射至 29‑DoF Unitree G1，并按 30 Hz 重采样后切成 24 帧（0.8 s）窗口。

**📈 对比分析**

在同等 160 bit 位宽下与 FSQ、VQ‑VAE 对比：在 BONES‑SEED 上 qpos MSE 分别为 7.1±0.3、9.0±0.4、11.5±1.7；在 LAFAN1 上为 14.4±1.0、19.0±1.5、28.3±3.0；条件对数似然亦最高。物理机器人上成功执行 16.8 s 的舞蹈、格斗、行走以及 35.2 s 的战斗‑舞蹈过渡，跟踪奖励和 MSE 接近仿真值。

**⚠️ 局限性**

局限性包括：① 生成序列需离线预先计算，缺乏实时在线生成；② 运动库规模与转移难度未系统评估；③ 低层策略仅依据离散词典，缺少全局路径约束；④ 仅在 Unitree G1 上验证，缺少跨机器人泛化；⑤ 分词分布与实际动作差异可能导致执行误差。

---

## 306. Efficient Remote Sensing Image-Text Retrieval via Interaction-Free Recall and Text-Guided Reranking

**arXiv ID:** 2610.36756 | [PDF](https://arxiv.org/pdf/2610.36756v1)

**作者:** Xi Chen `[一作]` (Wuhan University), Zhenyuan Shun `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Retrieval` `Computational Efficiency` `Representation Learning` `Transformer` `Vision Language Model` `Contrastive Learning` `Auto Encoder` `Retrieval-Augmented Generation` `Image` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

提出了Fast-then-Fine (FTF) 两阶段遥感图像-文本检索框架，通过交互自由多粒度召回与参数无关的 BTIB 重新排序实现高效且精细的跨模态匹配。

**💡 创新点**

创新点包括：① 交互自由多粒度召回可预计算图像嵌入，避免全局 token‑level 交互；② BTIB 通过双向 softmax 归一化实现无参数的文本引导视觉交互；③ I²M 损失同时优化跨模态对齐与内部结构一致；④ 通过候选数 K 直接调节准确率与推理效率的权衡。

**🔧 技术方法**

使用 CLIP ViT‑B/32 视觉与文本编码器，窗口池化生成粗细级视觉 token，构建多粒度视觉嵌入；BTIB 采用双向归一化的 token 交互；I²M 损失结合 InfoNCE 与邻域一致性；近似最近邻实现快速召回；最终通过 cosine 相似度进行重排序。

**📊 数据集**

在三大遥感图像-文本基准上进行实验：RSITMD、RSICD 与 UCM。

**📈 对比分析**

与 20 种任务特定和 VLM 方式（如 FSSN、EBAKER、RemoteCLIP 等）对比，在 RSITMD 上 FTF Full 达到 51.73 mR，FTF(K=200) 51.47 mR，且 K=50 时延迟仅 12.4 ms、吞吐 80.6 q/s；在 RSICD 与 UCM 上同样保持接近最强基线的 mR，但推理速度提升 2–6 倍，展示出优秀的准确率‑效率平衡。

**⚠️ 局限性**

局限性：召回阶段仍需预计算并存储图像嵌入，对极大图库的存储与更新成本较高；候选覆盖受 K 限制，过小的 K 可能遗漏真值；BTIB 采用无参数交互，可能在极度复杂或噪声较大的查询中性能不足；跨域或更大规模遥感数据的泛化性尚未充分验证。

---

## 307. Native Rectangular Linear Assignment with Implicit Rejection: A Jonker-Volgenant Solver without Square Expansion

**arXiv ID:** 2610.35948 | [PDF](https://arxiv.org/pdf/2610.35948v1)

**作者:** Ratha Siv `[一作]`, Ratha Siv `[通讯]`

**关键词:** `Mathematical Software` `Optimization` `Rectified Flow` `Optical Flow` `Contrastive Learning` `Benchmark` `Benchmark`

**🎯 论文内容**

提出并实现了一个基于 Jonker–Volgenant 算法的矩形匹配求解器，该求解器通过隐式拒绝终端实现无平方扩展的统一阈值匹配，并在 Rust、C++ 以及混合实现中提供了 Python 接口。

**💡 创新点**

在 Jonker–Volgenant 的搜索阶段仅使用一条距离和一条前驱行来表示拒绝，实现了在不构造完整平方矩阵的情况下完成阈值匹配；同时给出了等价的单侧目标和原始‑对偶证明，提供了可验证的最优性证书。

**🔧 技术方法**

采用 Jonker–Volgenant 的增广路径算法、矩形约简、隐式拒绝终端、Rust/C++ 原生扩展、Python 拓展、批处理工作者、基准对比、内存占用测量、等价变换与对偶证明等技术。

**📊 数据集**

使用100%生成的随机浮点成本矩阵（0~1 或 -1~1），覆盖不同尺寸（4 到 5000 行列）、宽高比（4:1、固定行/列）、结构化区块、稀疏、重复成本、不同阈值、正负阈值等，构成 204 种实验设置。

**📈 对比分析**

在两套 Python 环境（3.14.9/3.11.17）下对 8,946 次单矩阵和批量测试进行计时，比较 JVRUST、JVHYBRID、JVCPP 与扩展形式的 JVX、JVS、JVC 以及 SciPy（capped、dummy、square）；结果显示 JVRUST 在 6/7 必要匹配和 5/5 阈值匹配场景中最快，且在大多数情况比扩展接口快数倍；批处理和内存占用显著低于扩展形式；但在高竞争（256×320 阻塞）和极宽/极窄尺寸下 SciPy 的 capped 方案更快。

**⚠️ 局限性**

实验仅在单台 Windows/Intel i9 主机上进行，未控制 CPU 频率和后台负载；使用浮点64且未评估极大动态范围；基准仅覆盖合成矩阵，未验证真实跟踪或工业数据；对偶证明在理论上成立，但代码路径和浮点误差未完整证明；仅比较特定实现，对其他 LSAPE 或可选匹配实现未做完整对照。

---

## 308. Agentic Data Sampling Framework for Multilingual ASR via Linguistic-Aware Error-Driven Optimization

**arXiv ID:** 2610.36538 | [PDF](https://arxiv.org/pdf/2610.36538v1)

**作者:** Zhiheng Qian `[一作]` (Shanghai Jiao Tong University), Kai Yu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `Computation and Language` `Recognition` `Transformer` `Large Language Model` `Agentic AI` `Supervised Fine-Tuning` `Audio` `Text`

**🎯 论文内容**

在多语种 ASR 训练中，构建了一个基于 LLM 的智能数据采样框架，利用语言学误差分析（音位、形态、句法）来动态调整域、样本和词级别的采样策略。

**💡 创新点**

创新点在于把评估结果嵌入训练循环，形成闭环；引入语言学维度的误差指标，并用 LLM 代理根据误差反馈实时决定采样方案。

**🔧 技术方法**

使用 Whisper‑medium 作为语音识别模型；LLM 代理结合 DoReMi、DOGE、LESS、RHO‑1 等采样算法；采用低秩适配（LoRA）等微调技术。

**📊 数据集**

在 Common Voice 的12种语言子集上（Belarusian、Chinese、Dutch、French、German、Italian、Japanese、Polish、Portuguese、Russian、Spanish、Turkish）进行实验。

**📈 对比分析**

与无采样和随机采样基线对比，agentic‑300h 在全 1200 小时训练预算下平均 WER/CER 降低至 19.4，优于无采样 23.9、随机采样 22.5，并在大多数语言上取得最优结果。

**⚠️ 局限性**

局限性包括仅验证了12种语言，采样策略的可解释性与泛化性待进一步探索；误差诊断框架需更细粒度的语言学规则；模型对极低资源语言的提升有限。

---

## 309. PROBE: Frontier Coding Agents Find Different Bugs Than Maintainers Fix

**arXiv ID:** 2610.35987 | [PDF](https://arxiv.org/pdf/2610.35987v1)

**作者:** Marco Paina `[一作]` (LogicStar), Martin Vechev `[通讯]` (ETH Zurich)

**关键词:** `Software Engineering` `AI Code Assistant` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Code` `Benchmark`

**🎯 论文内容**

创建了一个基准（BugFinderBench），收集20个GitHub仓库在18个月内维护者修复的2457个真实bug，并用多轮文件分区审计 harness 评估10个前沿编码代理的发现能力。

**💡 创新点**

首次以维护者修复记录为参考，构建完整仓库级bug发现基准，并提出LLM驱动的匹配方法，证明代理发现的bug与维护者修复的bug在分布上显著不同。

**🔧 技术方法**

利用LLM进行commit分类、bug抽取、代码锚定、报告合成与去重，并设计规则+LLM双阶段匹配、以及多轮文件分区审计 harness 来系统评估。

**📊 数据集**

使用20个Python/JavaScript/TypeScript开源项目的2025年1月快照及其2025-2026期间的提交历史，生成的2457个维护者修复bug组成参考集。

**📈 对比分析**

在相同 harness 下对代理的召回、精度和F1与参考集进行对比，最高召回达20.8%、最高精度8.1%；更强模型和更高推理力度仅提升召回，精度基本不变；代理发现的bug真实率约70%–95%，但与维护者修复的bug分布差异显著。

**⚠️ 局限性**

参考集仅覆盖18个月内修复的bug，未能涵盖所有潜在bug；基准聚焦三种内存安全语言且文件数≤200k，缺乏对内存安全和更大项目的评估；评估方法主要基于静态代码检查，忽略运行时环境和用户交互导致的bug。

---

## 310. Rollback: Experience Backtracking in World Models for Policy Improvement

**arXiv ID:** 2610.36068 | [PDF](https://arxiv.org/pdf/2610.36068v1)

**作者:** Jiayi Luo `[一作]` (AgiBot), Guanghui Ren `[通讯]` (AgiBot)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Transformer` `World Model` `Reinforcement Learning` `Supervised Fine-Tuning` `Monte Carlo Tree Search` `Flow-based Model` `Video` `Image` `Sequential`

**🎯 论文内容**

本文提出了一种利用经验回溯（experience backtracking）在世界模型中进行策略改进的方法，并将世界模型从被动仿真器转变为可进行反事实推演的引擎。

**💡 创新点**

创新点在于：①通过回溯到关键决策点并在共享上下文上分支生成可比的行动序列，集中想象力于最具信息量的决策；②利用树结构聚合子树结果，得到细粒度的行动层级优势标签，无需额外的价值网络；③构建一个统一的分布式训练系统，实时将真实机器人数据用于“测试时物理校正”（test‑time physics grounding）以不断修正世界模型，并与策略共同进化。

**🔧 技术方法**

技术手段包括：基于 GE‑Sim 2.0 的动作条件视频世界模型、蒙特卡洛树搜索（MCTS）与渐进扩展、流匹配式策略优化、优势条件化的 policy 训练、LoRA 方式的 SFT 与 DMD 进行世界模型微调，以及异步云‑边缘‑设备三层架构的并行搜索与训练。

**📊 数据集**

数据集与任务：使用从专家示范收集的离线视频、由单台物理机器人实时采集的轨迹，并在三项挑战性真实任务（木块提取、球投掷、桌面曲棍球）上进行评估，任务观测包括三视角 RGB 与机器人本体状态。

**📈 对比分析**

与基线（SFT、RECAP、RISE w/ GE‑Sim2）比较，回溯世界模型方法在三任务上分别提升绝对成功率 10.67%、11.11% 与 11.85%，平均提升 11.21%；在 8 小时内不使用人类演示即可超过人类玩家在球投掷和桌面曲棍球的表现，展示了更稳健、持续的性能提升。

**⚠️ 局限性**

局限性：①对高质量世界模型的依赖，若模型误差大仍可能导致错误标签；②需要昂贵的 GPU 资源（48 H200 GPU）与实时数据管道；③实验仅覆盖三类任务，尚未验证对更复杂、多步或部分可观测环境的普适性；④奖励模型误判可能污染优势标签；⑤方法对长时间序列的预测误差仍存在累积风险。

---

## 311. Deterministic Causal Quantization with Representative States and Bounded Error

**arXiv ID:** 2610.35933 | [PDF](https://arxiv.org/pdf/2610.35933v1)

**作者:** Rama Cont `[一作]` (University of Oxford), Massoud Heidari `[通讯]` (LeastBit Information Lab)

**关键词:** `Information Theory` `Compression` `Shannon lower bound` `Rényi theory` `Sobolev inequality` `Markov chain` `Feller property` `Information theory` `Time Series` `Sequential` `Review/Survey Paper`

**🎯 论文内容**

本文提出了一种确定性、无误差上限为ε的量化方法——Midpoint Representative Realization（MRR），并证明该方法在所有输入路径上都满足误差约束。

**💡 创新点**

创新点在于给出了与ε‑entropy、过程ε‑entropy的分散率比较，证明在任意稳定源上MRR的地址熵率与最优块量化的差距不超过两比特；在格点初始化时进一步将差距压缩到一比特，并利用Fisher信息给出二次收敛的误差价。

**🔧 技术方法**

使用了几何覆盖定理、地址树计数、相对熵与Fisher信息的Sobolev不等式、马尔可夫链与Feller性质、Shannon下界以及Rényi理论等技术。

**📊 数据集**

本文未使用传统实验数据集，而是在理论上考虑任意稳态实值过程，并对独立同分布、拉普拉斯、正态和自回归拉普拉斯等典型分布给出了具体上界。

**📈 对比分析**

与传统块量化和平均误差率做对比，MRR在无分布假设下仅多出两比特，格点初始化可进一步降低到一比特；当满足Fisher信息条件时误差价为O(ε²)，即误差价随ε²衰减。

**⚠️ 局限性**

局限包括对非独立过程的收敛性未知、无法保证随容量增大速率单调下降，以及对特定更新规则的适用性有限。

---

## 312. Beyond LWE: Post-Quantum Assumptions under a Counterfactual Break of the Dihedral Coset Problem

**arXiv ID:** 2610.36731 | [PDF](https://arxiv.org/pdf/2610.36731v1)

**作者:** Ramses Fernandez `[一作]` (AegisQ Technologies), Ramses Fernandez `[通讯]` (AegisQ Technologies)

**关键词:** `Cryptography and Security` `Security` `Cryptography` `Quantum Computing` `Fourier Analysis` `Quantum Computing` `Information Entropy` `Gaussian Measure` `Quantum Measurement Analysis` `Review/Survey Paper` `Physics Related`

**🎯 论文内容**

在对假设存在多项式时间的二面体余数问题（DCP）求解器的基础上，系统梳理了其对量子化简链条、最优向量问题（SVP、SIS、SIVP）以及学习误差（LWE、Ring‑LWE、Module‑LWE）的定量后果，并对NIST标准及其他后量子方案给出威胁评估。

**💡 创新点**

提出三种不同的DCP求解假设（无噪声循环余数、带失败指数的DCP、外推余数DCP）并给出对应的可证明安全因子；系统化了从DCP到 LWE / SVP 的全部已公开化简链路，并标注样本需求、误差容忍度以及参数证书的完整性。

**🔧 技术方法**

利用隐藏子组问题的 Fourier 分析、Kuperberg 筛选、Regev 的子集求和模板、信息熵与 Gaussian 测度工具，以及对量子测量与信息丢失的定量分析。

**📊 数据集**

无实验数据集；论文完全基于理论分析和已有密码方案（ML‑KEM、ML‑DSA、FrodoKEM 等）的参数设置。

**📈 对比分析**

通过与已知经典算法（BKZ、sieving、NTRU、McEliece 等）以及已发表的量子攻击（Kuperberg、Regev 等）进行对比，评估了在各假设下的安全级别、样本复杂度和失败概率，说明了在某些参数域下的安全优势与不足。

**⚠️ 局限性**

主要局限在于：1) 需要的样本量往往超出已公开化简的范围；2) 失败指数假设对真实实例的适用性有限；3) 外推余数 DCP 的参数证书在实际部署方案上尚缺失；4) 分区/适配性假设未得到严格证明；5) 对实际密码方案（如 ML‑KEM、ML‑DSA）的完整攻击路径仍未完成。

---

## 313. Axiom: a logic programming language to express and enforce network designs

**arXiv ID:** 2610.36748 | [PDF](https://arxiv.org/pdf/2610.36748v1)

**作者:** Tibor Schneider `[一作]` (ETH Zurich), Laurent Vanbever `[通讯]` (ETH Zurich)

**关键词:** `Networking and Internet Architecture` `Optimization` `Autonomous Driving` `Anomaly Detection` `Computational Efficiency` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Text` `Benchmark`

**🎯 论文内容**

提出一种语言与检查器，直接在配置文本上验证网络设计并定位违规命令

**💡 创新点**

将网络设计视为第一类实体，以Datalog规则表达，实现对配置的即时反馈和定位错误

**🔧 技术方法**

基于Datalog与格结构的分层求解器，结合配置模板与递归查询

**📊 数据集**

使用TopologyZoo中最大754路由器网络生成的Cisco配置

**📈 对比分析**

与Batfish等工具对比，特定设计下检查4.5M行配置仅需2–5秒，内存低于0.3GB；与通用验证相比，速度提升1–10倍，错误定位更精准

**⚠️ 局限性**

需要手动编写设计模板，受限于对正则、顺序过滤等高级配置的支持不足，尚未实现增量更新

---

## 314. When Does Perturbation Consistency Add Information Beyond Softmax Confidence? Theory and Evidence for the Decision Consistency Framework

**arXiv ID:** 2610.36640 | [PDF](https://arxiv.org/pdf/2610.36640v1)

**作者:** Amit Kalitaa `[一作]`, Moirangthem Tiken Singh `[通讯]` (Dibrugarh University)

**关键词:** `Machine Learning` `Classification` `Anomaly Detection` `Convolutional Neural Network` `Contrastive Learning` `Adversarial Attack` `Image` `Biomedical Data`

**🎯 论文内容**

本文研究了决策一致性框架（DCF）在图像分类任务中的误判检测能力，评估其与传统单前向置信度分数（如MSP、logit margin）以及对抗性和分布偏移下的性能差异；

**💡 创新点**

创新点在于提出了理论上证明一致性与清洁输出无关的判定条件、构造了无标签的分层非参数无翻转证书，并系统性地对不同模型、数据集与扰动类型进行预设实验验证；

**🔧 技术方法**

技术上主要使用六种轻微扰动（旋转、亮度、对比、噪声、模糊、JPEG压缩）计算标签稳定性（LS）与置信度偏移（CD）并组合得到一致性得分（CS），同时实现分层置信度证书与对抗性检测；

**📊 数据集**

使用了CIFAR‑10、STL‑10、Intel‑Scene、OCTMNIST等四个图像分类数据集，以及对抗攻击数据和CIFAR‑10‑C等噪声变异；

**📈 对比分析**

通过AUROC、AURC等指标与MSP、margin、entropy、energy等基线进行对比，结果显示在干净数据上DCF性能逊于MSP，且成本高；但在噪声和对抗攻击场景下可略优于MSP，且证书能在不计算扰动时提供高置信度筛选；

**⚠️ 局限性**

主要局限在于仅针对小规模图像任务、六种轻微扰动、固定模型结构；对更强扰动、不同任务、标签保持不确定性等情况缺乏评估；证书的可靠性依赖于数据交换性，且在分布漂移时效果有限。

---

## 315. High-Fidelity Aerostructural Dataset of Transonic Wings in Cruise Conditions

**arXiv ID:** 2610.36442 | [PDF](https://arxiv.org/pdf/2610.36442v1)

**作者:** Pavankumar Koratikere `[一作]` (Purdue University), Leifur Leifsson `[通讯]` (Purdue University)

**关键词:** `Machine Learning` `Optimization` `Data-Centric Learning` `Computational Efficiency` `Federated Learning` `Explainability and Interpretability` `Representation Learning` `Reinforcement Learning from Human Feedback` `Neural Architecture Search` `Physics Related` `Transformer` `Auto Encoder` `Diffusion model` `Score-based Model` `Contrastive Learning` `Gaussian Splatting` `Simultaneous Localization and Mapping` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation` `Supervised Fine-Tuning` `Tabular` `Graph` `Mesh` `Point Cloud` `Benchmark`

**🎯 论文内容**

创建了一个包含2471个样本和3个OOD样本的高保真多学科（气动-结构）耦合数据集，并用它训练多任务深度学习模型预测机翼的气动表面分布、集成气动量和翼盒失效分布。

**💡 创新点**

首次公开发布完整的机翼气动-结构耦合数据集，提供基准样本与OOD样本，显著促进跨学科ML方法的研发与评估。

**🔧 技术方法**

采用Latin hypercube采样、MPhys耦合框架、ADFlow CFD、TACS有限元、UQAeroNet/MHA-Net多任务网络以及NIROM（POD/Autoencoder）失效预测技术。

**📊 数据集**

使用了本文自建的 Transonic Wing Aerostructural Dataset，公开托管在 Hugging Face。

**📈 对比分析**

通过 NRMSE 与 R² 指标与 MHA‑Net、UQAeroNet、POD/Autoencoder NIROM 进行对比，气动表面/集成量 NRMSE ≤0.03、R²≈0.998；失效分布 NRMSE≈0.118、R²≈0.76–0.83。

**⚠️ 局限性**

仅考虑巡航条件、使用铝材料、缺少更多结构参数、多学科耦合的复杂度有限，且失效预测精度仍有提升空间。

---

## 316. LociArt: A Unified Paradigm for Free-Style Local Video Stylization

**arXiv ID:** 2610.36456 | [PDF](https://arxiv.org/pdf/2610.36456v1)

**作者:** Yuxuan Chen `[一作]` (MiLM Plus, Xiaomi Inc.), Daiguo Zhou `[通讯]` (MiLM Plus, Xiaomi Inc.)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Image Translation` `Restoration` `Transformer` `Auto Encoder` `Generative Adversarial Network` `Diffusion model` `Prompt Engineering` `Contrastive Learning` `Generative Refinement Networks` `Vision Language Model` `Video` `Image` `Text` `Benchmark`

**🎯 论文内容**

提出LociArt框架，实现自由风格本地视频风格化；

**💡 创新点**

核心创新在于极坐标风格感知掩模与表示‑几何条件化，允许将风格与空间精准绑定，支持多区域多风格组合；

**🔧 技术方法**

技术包括GRNStyle（基于Generative Refinement Networks的变体）、极坐标风格编码、VAE‑编码掩模与几何支持、源证据引导的迭代生成；

**📊 数据集**

构建了LociArt‑100K数据集（100K源视频+掩模+风格指令+目标风格化视频）以及LociArt Benchmark（80源视频+84掩模单元，168个前景/背景风格化案例）；

**📈 对比分析**

与现有图像/视频风格化及编辑方法（LEAST、UniVST、RegionRoute、StyleGallery、VACE、CoinVE‑Edit、Seedance 2.5等）对比，GRNStyle在Style Prototype Similarity、CLIP‑T、PSNR、Temporal Consistency及VLM评分上均表现最优；

**⚠️ 局限性**

局限包括对掩模精度和风格多样性的依赖，极坐标码表需预定义，且对极端形变或遮挡的鲁棒性尚待进一步验证。

---

## 317. A Trainable CPG Framework for Quadruped Locomotion

**arXiv ID:** 2610.36537 | [PDF](https://arxiv.org/pdf/2610.36537v1)

**作者:** Xiyan Liu `[一作]` (Zhejiang University), Shaoxing Qu `[通讯]` (Zhejiang University)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Recurrent Neural Network` `Reinforcement Learning` `Proximal Policy Optimization (PPO)` `Supervised Fine-Tuning` `Optical Flow` `Time Series` `Sequential`

**🎯 论文内容**

本文提出一种可训练的四足机器人行走控制框架CPG–ENN，将八神经元中央模式发生器（CPG）与共享的执行神经网络（ENN）相结合，实现多步态的速度与方向跟踪，并支持跨步态切换。

**💡 创新点**

创新点在于将显式节律生成与学习实现分离：使用共享的EN网络将CPG节律与轨迹参数映射到关节角度；采用单一反馈策略调节六个物理可解释的轨迹变量；以及采用Beta分布的PPO训练以严格限制动作范围，从而实现无特定步态训练的跨步态切换。

**🔧 技术方法**

技术上使用了八神经元CPG模型、共享的多层感知机ENN、基于Isaac Gym的并行模拟训练、Proximal Policy Optimization（PPO）配Beta动作分布、离线逆运动学监督、以及基于比例-微分（PD）控制的关节跟踪。

**📊 数据集**

数据集主要来自Isaac Gym仿真环境，使用离线IK生成的脚位轨迹作为ENN监督；硬件实验采用Unitree Go2机器人，在Jetson NX上部署控制器并实时下发关节指令。

**📈 对比分析**

与AllGaits和Walk These Ways（WTW）进行对比实验，在5种步态与4种速度组合下，CPG–ENN实现了99.39%的跨步态切换成功率，平均速度跟踪误差低于0.13 m/s，显著优于基线方法，并在无人调参的情况下保持稳定。

**⚠️ 局限性**

局限性包括：对复杂地形的反馈处理尚未加入，跨步态切换仍依赖相对简单的CPT时间对齐规则，且当前框架仅验证了平地跑步，未来需扩展到不平地或障碍物环境。

---

## 318. Detecting and Mitigating Query-Based System Prompt Leakage Attacks

**arXiv ID:** 2610.36295 | [PDF](https://arxiv.org/pdf/2610.36295v1)

**作者:** Tushar Nayan `[一作]` (Florida International University), Ruimin Sun `[通讯]` (Florida International University)

**关键词:** `Cryptography and Security` `Anomaly Detection` `Safty and Privacy` `Graph Neural Network` `Transformer` `Large Language Model` `Prompt Engineering` `Contrastive Learning` `Text` `Graph`

**🎯 论文内容**

针对黑盒LLM系统，提出一种端到端的系统提示泄露检测与缓解框架。

**💡 创新点**

通过构建有限语义交互图、统计异常过滤和基于线图的GCN结构学习，首次把多轮查询的相似关系转化为可判别的图结构，从而实现对分散式提问式泄露攻击的检测。

**🔧 技术方法**

使用句子编码器计算余弦相似度构建图，采用Grubbs异常检测挑选候选组件，并将其转化为线图输入GCN进行多跳关系学习；同时结合单查询风险评分做早期拦截。

**📊 数据集**

实验使用公开提示泄露攻击数据集（Zhang、PromptFuzz、LeakAgent、PLeak）、五大benign集（GSM8K、SST-2、SQuAD、Dolly、WildJailbreak）以及三套多轮对话集（OASST1、ChatAlpaca、WildChat）。

**📈 对比分析**

与 PromptGuard、LLMGuard 等输入侧防御对比，混合查询与多轮对话场景下 F1 94.8–100%、召回 91–100%、误报率 ≤3.4%，在自适应攻击中将精确/子串泄漏压至 0–0.01，平均查询延迟约 32.96 ms。

**⚠️ 局限性**

对极短或极稀疏攻击难以早期发现，依赖语义嵌入对极度伪装的查询敏感，并需额外存储与计算资源来维护交互图状态。

---

## 319. ADTGuard: Adaptive Depth-Time Safeguard for Streaming Inference in Large Language Models

**arXiv ID:** 2610.36574 | [PDF](https://arxiv.org/pdf/2610.36574v1)

**作者:** Chuming Wei `[一作]` (Xi'an Jiaotong-Liverpool University), Guangwen Yang `[通讯]` (Tsinghua University)

**关键词:** `Artificial Intelligence` `Safty and Privacy` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

研发了一种轻量级流式安全防护模块 ADTGuard，能够在每个 token 生成前实时检测其安全性，并在不修改底层 LLM 的前提下完成监控。

**💡 创新点**

创新点包括：① 通过因果深度路由（Causal Depth Router）动态选择关键层；② 深度‑时间安全动力学（Depth‑Time Safety Dynamics）在深度与时间维度上联合建模安全证据；③ 风险启动目标细化（Risk‑Onset Target Refinement）在仅有响应级标签的情况下逐步生成 token 级监督。

**🔧 技术方法**

技术细节：冻结主干 LLM（如 Qwen3‑8B、Llama‑3.1‑8B、Gemma‑7B）；实现 CDR、DTSD（深度与时间状态空间网络）、RTR、单调性正则化；训练采用 LoRA 方式，仅更新约 16 M 可训练参数。

**📊 数据集**

实验使用多项安全基准：SGB‑S‑Eval、SGB‑WildGuard、AEGIS2.0、BeaverTails 与 WildGuard，覆盖生成与公开数据集。

**📈 对比分析**

与基于表示的防护（SIREN、Kelp）及独立审查模型（Qwen3Guard‑Gen/Stream、Llama‑Guard、SentGuard）对比，ADTGuard 在 15 种模型‑数据组合中 14 次获得最高 Streaming F1，且相对参数量少、每 token 延迟约 2.3 ms。

**⚠️ 局限性**

局限性：仍需依赖响应级标签；在更大规模或极端对抗场景下的鲁棒性待验证；仅在固定 LLM 上测试，无法保证跨模型迁移；未实现早停机制，安全检测不一定能及时中断生成。

---

## 320. Before the First Edit? Measuring When Coding-Agent Guidance Actually Arrives

**arXiv ID:** 2610.35992 | [PDF](https://arxiv.org/pdf/2610.35992v1)

**作者:** Thupakula Sai Varun `[一作]`, Thupakula Sai Varun `[通讯]`

**关键词:** `Software Engineering` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Agentic AI` `Text` `Code`

**🎯 论文内容**

本文通过审计编码代理的“计划先于首次源代码修改”这一承诺，对代理系统的四个门控规则（编辑器仅、编辑器加字面量、所有工具请求、可写能力）以及计划交付的时间点进行验证，并结合本地多轮代理实验与公开轨迹数据，探讨代理在实际修复任务中何时以及是否会提前收到规划指导。

**💡 创新点**

创新点在于：①提出可操作的“边界审计”框架，利用源代码哈希在每个工具调用前后对比，明确首次源变更的时序；②将门控规则与实际计划交付拆分，区分规则触发与计划可用性；③将本地受控实验与公开轨迹结合，系统性评估多种代理与门控组合的覆盖率与效果；④对公开数据（PAUSE、SI2CA）进行源代码变化与计划交付的可观测性分析，指出公共轨迹在同步文件状态方面的局限。

**🔧 技术方法**

主要技术手段包括：源代码哈希快照比较、工具调用日志解析、门控策略实现（编辑器请求、字面量命令匹配、通用shell可写性判断）、本地代理运行（Qwen2.5-Coder 7B/14B）与计划调用、公开轨迹重构与重放、统计与可视化。

**📊 数据集**

使用的数据集包括：SWE‑bench Lite（开发集与测试集）、PAUSE 公开轨迹（verified/multilingual/Pi/OpenHands）、SI2CA 训练轨迹、以及自定义的39个构造操作单元。

**📈 对比分析**

比较方法：对每种门控策略在构造单元中测量计划交付率、覆盖率、错误率；在本地多轮实验中记录每个策略下的“首次变更是否在计划后”比例、计划token数与耗时；在公开轨迹中统计触发、计划存在与最终差异的对应关系。性能表现：在构造单元中，所有工具请求策略覆盖100%目标变更；编辑器仅策略仅覆盖12.5%；在本地实验中，所有策略都保证计划在首次变更前，但实际变更率仅在少数任务中出现，且几乎全部使用结构化编辑器，缺少shell‑first写。

**⚠️ 局限性**

局限性包括：①样本量有限，主要聚焦Python项目且任务数量受限；②公开轨迹缺少同步文件快照，导致无法精准定位首次变更；③仅通过最终diff或已命名路径间接推断变更，忽略临时写、回滚或非Python文件；④计划交付未直接证明能提升修复质量；⑤模型与策略的随机性与硬件/时钟波动可能影响结果；⑥缺乏官方SWE‑bench评分，无法量化修复成功率。

---

## 321. Does LLM Pre-Training Typically Occur at the Edge of Stability?

**arXiv ID:** 2610.36778 | [PDF](https://arxiv.org/pdf/2610.36778v1)

**作者:** Yuhang Cai `[一作]` (University Of California Berkeley), Kaifeng Lyu `[通讯]` (Tsinghua University)

**关键词:** `Machine Learning` `Optimization` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Contrastive Learning` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

本文研究并量化了大规模语言模型预训练中二次近似失效的两种模式：负曲率导致的 EoC（Edge of Convexity）和正曲率导致的 EoS（Edge of Stability）。

**💡 创新点**

创新点在于首次系统性提出 EoC 这一 LR 无关的失效模式，并给出可扩展的检测方法；同时将 EoS 概念推广到随机梯度、适应性优化器，并提供理论上可保证的判别标准。

**🔧 技术方法**

主要技术包括随机幂迭代估计负最小特征值、基于局部二次模型的投影和模拟来估计 EoS 临界学习率、以及对 AdamW 等适应性优化器的预条件化分析。

**📊 数据集**

实验使用 GPT‑style 模型（43M–787M）在 FineWeb、OpenWebMath 等公开数据集上进行预训练，采用 WSD 学习率调度、AdamW、不同批大小和学习率阶梯/余弦衰减方案。

**📈 对比分析**

与传统的仅关注 λ_max 的 EoS 定义相比，本文的检测方法在 3000+ 检查点上发现 EoC 处处出现、EoS 在大批量和较高学习率下占大多数时间；实验还显示，当学习率超出临界值时最终评估损失显著上升，表明该判别对训练稳定性具有实际指导意义。

**⚠️ 局限性**

局限性包括：只验证了 SGD 与 AdamW，尚未探究 MuP 等新优化器；理论分析依赖于局部二次模型的假设，可能在更复杂的网络结构或更大规模模型上失效；检测方法虽然可扩展，但仍需额外计算资源来估计负特征值和投影子空间。

---

## 322. ORBITAL: Optimization of Robust Brain--Computer Inference Through Architecture-Aware Low-Precision Compression

**arXiv ID:** 2610.36411 | [PDF](https://arxiv.org/pdf/2610.36411v1)

**作者:** Saim Rehman `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**关键词:** `Computer Vision and Pattern Recognition` `Optimization` `Computational Efficiency` `Adversarial Attack` `Compression` `Robotic Intelligence` `Convolutional Neural Network` `Contrastive Learning` `Auto Encoder` `Quantization` `Adversarial Training` `Biomedical Data` `Electroencephalogram (EEG)`

**🎯 论文内容**

提出 ORBITAL 框架，结合硬件可实现的结构压缩、TRADES 恢复、攻击评估和边缘硬件验证，以提升 EEG 解码器在嵌入式 BCI 上的抗攻击鲁棒性和能效。

**💡 创新点**

1) 通过结构化通道剪枝与硬件约束实现可部署的压缩点；2) 证明 TRADES 恢复在压缩后最能提升鲁棒性，并探索压缩与恢复的交互；3) 用多攻击确认和边缘原生量化评估验证鲁棒性和硬件真实收益；4) 对抗边界的分布式重排而非统一加宽。

**🔧 技术方法**

使用 EEGNet/ ShallowConvNet 模型、结构化通道剪枝、TRADES 对抗训练、QAT 低精度量化、PGD/ APGD/ FAB 等对抗攻击、TensorRT 在 Jetson Orin Nano 上原生 FP16/ INT8 推理、Pareto 选点和多种统计检验。

**📊 数据集**

BCI Competition IV-2a（九位受试者、22 通道、四类运动想象）。

**📈 对比分析**

先用 PGD20 等筛选清洁/鲁棒性，随后用 PGD100、APGD-CE/DLR、FAB 确认，评估 clean-correct true‑union 准确率。S25‑TRADES 在压缩 26% 参数、25% MAC 的同时，mean true‑union 达 23.06%，比全宽 Full 21.27% 提升 1.79pp；在 Jetson 上 FP16 推理时能耗/延迟分别降低 5.35%/0.1ms；INT8 量化虽然预测一致，但延迟/能耗提升约两倍。

**⚠️ 局限性**

对抗恢复的收益随压缩度变化且不适用于所有架构；量化精度降低导致转换开销增加，降低边缘实现收益；实验仅覆盖 EEGNet/ ShallowConvNet，其他 BCI 任务未知；未充分分析不同攻击半径和模型细节对鲁棒性的长期影响。

---

## 323. FinLLM-KG: Knowledge Graph-Enhanced LLMs for Financial Sentiment Analysis in Algorithmic Trading

**arXiv ID:** 2610.36083 | [PDF](https://arxiv.org/pdf/2610.36083v1)

**作者:** Giorgos Iacovides `[一作]`, Danilo Mandic `[通讯]`

**关键词:** `Computation and Language` `Explainability and Interpretability` `Optimization` `Finance Related` `Transformer` `Graph Neural Network` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `logit-to-score conversion` `Kernel SHAP` `Exact Shapley` `Text` `Graph` `Tabular` `Time Series` `Finance Related`

**🎯 论文内容**

结合金融知识图谱与冻结因果LLM，在推理阶段动态注入最新10‑K公司信息，对财经新闻情绪进行上下文增强，并提供句子层和事实层可解释性，用以构建长短组合。

**💡 创新点**

创新点在于：①利用已审核的10‑K知识图谱在推理时注入公司背景，而非训练或微调模型；②提供公司级背景而非单篇文章的孤立分析；③实现句子级和事实级的可解释性；④在同一基础LLM上显著提升组合收益与风险调整后表现。

**🔧 技术方法**

技术手段包括：冻结因果LLM（如Claude Haiku 4.5/FinDPO）、FinReflectKG知识图谱、文本嵌入检索（top‑k）、logit‑to‑score 转换、Kernel SHAP 与 Exact Shapley 解释。

**📊 数据集**

数据集：约200k财经新闻（2015‑2021），经NER筛选后约160k；FinReflectKG（700+公司 2014‑2024 的10‑K三元组）；S&P 100 日行情数据（Yahoo Finance）用于组合回测。

**📈 对比分析**

与FinDPO基线在同一基础LLM上对比；在35% long‑short 组合中，累积收益从275%提升至约330%，Sharpe 从 2.55 提升至 3.15，Sortino 与 Calmar 亦显著提升，表明知识图谱增强显著提升了收益与风险控制。

**⚠️ 局限性**

局限性包括：需要已公开的10‑K才能注入，缺失公司信息时无法评分；仅检索最新10‑K，忽略更远历史信息；模型冻结无法针对任务微调；解释计算成本高；对多步生成或动态更新的支持有限。

---

## 324. Janus Vehicles: Entity-Centric Threat Modelling of Protocol-Conformant Vehicles under a Malicious-Manufacturer Assumption

**arXiv ID:** 2610.37059 | [PDF](https://arxiv.org/pdf/2610.37059v1)

**作者:** Jaeung Franciskus Yoo `[一作]` (Korea University), Huy Kang Kim `[通讯]` (Korea University)

**关键词:** `Cryptography and Security` `Security` `Threat Modelling` `Autonomous Driving` `Safty and Privacy` `Review/Survey Paper`

**🎯 论文内容**

本文提出了针对恶意设计车辆的实体中心威胁模型，包含四类恶意实体原型（Malign Entity Archetypes）、行为-位点矩阵（Behaviour–Locus Matrix）以及行为交互模型（Behavioural Interaction Model），从硬件、软件、网络层面系统化描述隐藏功能的实现与隐藏方式。

**💡 创新点**

创新点在于：①把真实性与功能目的分离，首次从恶意设计者视角重新定义车辆威胁；②提出三维威胁模型（实体原型+行为-位点矩阵+行为交互模型），填补现有模型忽视恶意设计者的空白；③借鉴恶意软件行为分类框架，构建面向实体的综合威胁分析方法。

**🔧 技术方法**

使用的技术主要是概念建模与威胁建模方法，参考了恶意软件行为分类（如MBC、MAEC）来构造行为层次；未涉及算法实现或硬件实验。

**📊 数据集**

无，本文为理论框架，没有使用具体数据集。

**📈 对比分析**

本文未做实验或性能对比，亦未给出定量评估；因此没有可比方法或性能指标。

**⚠️ 局限性**

limitations: ①仅为概念性模型，缺乏具体检测/调查流程与工具支持；②未经过实证验证，无法确认模型在真实车辆中的适用性；③对可观测性与证据获取的技术细节不足；④不同观察者（同行、实验室、监管机构）对行为的可见性差异未被量化。

---

## 325. AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations

**arXiv ID:** 2610.36980 | [PDF](https://arxiv.org/pdf/2610.36980v1)

**作者:** Qiming Shi `[一作]` (StepFun), Zuo Bai `[通讯]` (StepFun)

**关键词:** `Artificial Intelligence` `Autonomous Driving` `Optimization` `Reinforcement Learning from Human Feedback` `Benchmark` `Transformer` `Large Language Model` `Agentic AI` `Retrieval-Augmented Generation` `Prompt Engineering` `Tabular` `Time Series` `Finance Related` `Benchmark`

**🎯 论文内容**

在AutoVend基准上，研究者让LLM代理管理20台分布式自动售货机网络，并评估其在180天内的经济表现与规则基线比较；

**💡 创新点**

创新点在于构建了一个长期、基于真实销售数据的卖方代理商业环境，结合共享库存、采购提前期、付费现场补货与跨站资源协调；

**🔧 技术方法**

技术上使用了多种大型语言模型（Claude、GPT-6.1、DeepSeek、Kimi、GLM）与Model Context Protocol工具调用，配合模拟器实现代理决策与物理执行；

**📊 数据集**

数据集来源于99台自动售货机的六个月订单级销售记录，共220,128件产品，提炼出1,372种商品的真实需求与销售模式；

**📈 对比分析**

评估方法通过比较终端净资产、商品组合质量、浪费率、定价后悔、货架利用率、工具调用次数及API效率等指标，结果显示GPT-6.1 Sol优于规则基线，而其余四模型普遍低于基线；

**⚠️ 局限性**

局限性包括在单一环境下测试、对跨站协调与执行成本的平衡不足、反馈学习效果有限，以及对真实多变商业情境的泛化能力待进一步验证。

---

## 326. Tap2Listen: Instant Target Switching for Streaming Audio-Visual Speaker Extraction

**arXiv ID:** 2610.36643 | [PDF](https://arxiv.org/pdf/2610.36643v1)

**作者:** Hansol Park `[一作]` (Sungkyunkwan University), Kyuhong Shim `[通讯]` (Sungkyunkwan University)

**关键词:** `Sound` `Recognition` `Computational Efficiency` `Transformer` `Convolutional Neural Network` `Contrastive Learning` `Cross-Attention` `Audio` `Video` `Multimodality`

**🎯 论文内容**

提出了一种名为Tap2Listen的实时音视频目标说话人提取模型，能够在听者点击脸部时即时切换目标说话人；

**💡 创新点**

通过将目标无关的音频编码与目标特定的视觉解码分离，仅在切换时重新投影缓存的音频表示，实现了毫秒级的目标切换；

**🔧 技术方法**

使用了声谱分析（STFT）和因果自注意力的音频编码器、BlazeNet64视觉前端、交叉注意力解码器，并结合预测损失和深度监督提取损失进行训练；

**📊 数据集**

在LRS3‑2mix数据集（约40k训练、5k验证、3k测试）上进行实验；

**📈 对比分析**

与现有AV‑SkiM基线比较，Tap2Listen在相同参数量下实现了9.68 dB SI‑SNRi（高于8.85 dB），在目标切换后首帧即可恢复完整质量，且在单个移动CPU核心上能保持实时（≈3.2 ms/帧）；

**⚠️ 局限性**

局限在于算法延迟略高（16 ms对比1 ms），以及仍需在STFT域上处理，可能对低延迟极端场景有限制。

---

## 327. PASS Is a Claim, Not an Observation: Mechanical Re-verification of 10 Self-Asserted Fix Claims in a Live Multi-Agent Audit Layer

**arXiv ID:** 2610.35950 | [PDF](https://arxiv.org/pdf/2610.35950v1)

**作者:** Lei Zhenbing `[一作]`, Lei Chang `[通讯]`

**关键词:** `Multiagent Systems` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Text` `Review/Survey Paper`

**🎯 论文内容**

对AITRAP多智能体平台中10条自我声明的代码修复公告进行机械化重新验证，评估其真实性

**💡 创新点**

首次系统性验证自报修复的可靠性，揭示自述PASS声明常常与实际代码变更不符

**🔧 技术方法**

采用确定性规则验证：文件修改时间戳一致性、修复模式与调用位置匹配、公告文本可重现性三项判定

**📊 数据集**

利用AITRAP的广播日志语料库（179条公告，其中28条包含修复声明，10条为代码级修复）

**📈 对比分析**

以三项判定标准构成的验证流程为基准，发现40%修复声明未通过定位检查，50%信息充分但仍未验证，说明自报可靠性低

**⚠️ 局限性**

样本量仅10条（总体抽样），验证由作者完成，缺乏第三方验证；仅检测静态代码，未涵盖运行时行为；受限于特定平台与时间点

---

## 328. PREEN: Exemplar-Based Color Style Transfer with Smartphone Editor Presets

**arXiv ID:** 2610.36896 | [PDF](https://arxiv.org/pdf/2610.36896v1)

**作者:** Beomjin Ahn `[一作]` (Samsung Research), Beomjin Ahn `[通讯]` (Samsung Research)

**关键词:** `Computer Vision and Pattern Recognition` `Image Translation` `Image Harmonization` `Computational Efficiency` `Representation Learning` `Transformer` `Convolutional Neural Network` `Vision Language Model` `Contrastive Learning` `Diffusion model` `Auto Encoder` `LoRA Low-Rank Adaptation` `Image` `Multimodality`

**🎯 论文内容**

提出了 PREEN 系统，实现基于内容–参考图像的低维预设回归，将颜色风格迁移转换为 29 维编辑器参数的预测，从而在手机端实现快速、可解释的风格转移。

**💡 创新点**

创新点包括：① 使用空间自适应 LUT 教师对任意参考图像进行伪预设恢复，克服无真实预设监督的难题；② 通过 CLIP 条件回归器和 LoRA 微调实现端到端训练；③ 引入单参数 α 控制色度强度，允许在同一次推理后生成从默认到饱和的连续预设。

**🔧 技术方法**

技术手段包括：CLIP ViT‑L/14 编码器、两棵残差 MLP 分别回归色度与亮度子空间、梯度基于 CIELAB 的参数恢复、LoRA 低秩适配、差分编辑器（29 维预设可微分）和 CIELAB 级别像素损失。

**📊 数据集**

主要数据集：PST50（人类标注的 50 对内容/参考/专家目标三元组）用于评估；训练数据来自 OpenImages 与 AVA，按语义相似度随机配对约数十万对；实验中还使用 DPST、OpenImages、AVA、OpenImages+AVA 的随机采样。

**📈 对比分析**

与基准方法（AdaIN、Reinhard、NLUT、SA‑LUT、Deep Preset、ModFlows 等）以及 VLM 直接预测预设进行对比。PREEN 在 PSNR、SSIM、LPIPS、ΔE*_ab 等指标上均优于 raster/LUT 方法，HCorr 虽略低但可通过更高层参数学习提升；在 PST50 上达到 24.6 dB PSNR、0.914 SSIM、0.094 LPIPS；在手机端仅 0.36 秒 CPU 推理，INT8 量化后性能差距不到 0.3 dB。

**⚠️ 局限性**

局限性：仅能表达全局 29 维编辑器空间，无法实现局部色调或遮罩效果；当目标超出编辑器可达子空间时（如完整色彩重映射）预测误差显著；参考图像需语义相似且在训练集范围内；以及对 29 维预设的依赖可能限制更细粒度的创意调整。

---

## 329. Reliability-Aware Short-Term Glycemic Risk Prediction Using an Anchor-Based Comparative Framework of Sequence, Multimodal, and State-Updating Digital-Twin Models

**arXiv ID:** 2610.36363 | [PDF](https://arxiv.org/pdf/2610.36363v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## 330. 3, 7, 14, 17, 42: Instruction Tuning Installs and Sharpens Number-Choice Heuristics in Language Models

**arXiv ID:** 2610.36178 | [PDF](https://arxiv.org/pdf/2610.36178v1)

**作者:** Samar Ansari `[一作]`, Samar Ansari `[通讯]`

**关键词:** `Computation and Language` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Supervised Fine-Tuning` `Text`

**🎯 论文内容**

本文测量语言模型在生成随机整数时的完整概率分布，剖析其在不同训练和解码设置下的行为。

**💡 创新点**

创新点在于用枚举而非采样消除采样误差，首次公开完整概率分布并将模型自身与解码器的贡献分离。

**🔧 技术方法**

使用枚举（token级概率求和）、温度重标、核采样/Top‑k、不同提示与解码格式等技术。

**📊 数据集**

基于18个公开权重模型线、1–100范围内的整数、29种解码配置、两套推理堆栈进行实验。

**📈 对比分析**

通过与基线（未调教）对比，采用总变差距离评估与均匀分布的偏离；结果显示调教显著收敛、消除端点和十位数等现象。

**⚠️ 局限性**

限制包括仅公开权重模型、未覆盖更大模型、未测量非标准拼写、仅验证单一条件、解码与提示格式不统一、量化不变、缺乏多语言/多尺度验证。

---

## 331. Closed-Loop Aware Planning via Imitation-Free Speed Scoring for End-to-End Autonomous Driving

**arXiv ID:** 2610.36079 | [PDF](https://arxiv.org/pdf/2610.36079v1)

**作者:** Emad Aghajanzadeh `[一作]` (Technical University of Munich), Ulf Schlichtmann `[通讯]` (Technical University of Munich)

**关键词:** `Robotics` `Autonomous Driving` `Recurrent Neural Network` `Transformer` `Score-based Model` `Reinforcement Learning` `Contrastive Learning` `Optical Flow` `Video` `Sequential`

**🎯 论文内容**

提出一种可插拔的规划头 CLAP-AD，在端到端驾驶模型中分离路径和速度学习，使用多准则评分实现速度的可控性。

**💡 创新点**

创新点在于用物理安全约束监督的多准则速度评分机制替代单一速度模仿，并实现无需速度条件训练的测试时速度控制。

**🔧 技术方法**

采用模仿学习预测路径，GRU+查询注意力的多分支评分网络（碰撞、红绿灯、停牌），以及阈值选择和 PID 控制器等技术。

**📊 数据集**

使用 Bench2Drive、Bench2Drive-Speed、NAVSIM v2 等公开仿真和真实驾驶数据集进行训练与评估。

**📈 对比分析**

与 UniAD、VAD、TFv6 等基线相比，CLAP-AD 在 Bench2Drive-Speed 上 Speed‑Adherence 提升 12.9 分，Bench2Drive Driving Score 提升 28.5/32.7 分，NAVSIM EPDMS 提升 6.1 分，性能优于 SparseDriveV2 等方法。

**⚠️ 局限性**

局限在于对查询式架构的依赖、在规则式专家数据下优势不明显、以及在更大模型规模下尚未验证。

---

## 332. NSB2: An Open-Source Modular Pipeline for Application-Network Co-Simulation

**arXiv ID:** 2610.35872 | [PDF](https://arxiv.org/pdf/2610.35872v1)

**作者:** Harikrishna Kuttivelil `[一作]` (University of California, Santa Cruz), Katia Obraczka `[通讯]` (University of California, Santa Cruz)

**通讯引用:** 9642 | [OpenAlex ID](https://openalex.org/A5067832138)

**关键词:** `Networking and Internet Architecture` `Benchmark`

**🎯 论文内容**

提出了 NSB2 —— 一个轻量级的网络模拟桥接中间件，允许应用程序通过守护进程与 ns‑3、OMNeT++ 等网络模拟器实现协同仿真。

**💡 创新点**

创新点包括：完整重构的多通道通信架构、基于内存键值缓存的 Payload 传输优化、统一的 Protobuf 协议、双语言（Python/C++）支持，以及可插拔的传输/缓存模块，实现了真正的跨语言、跨模拟器、跨平台集成。

**🔧 技术方法**

技术栈主要使用 TCP sockets、Protocol Buffers、Redis、YAML、CMake+pkg‑config、Python3 与 C++17 等开源组件。

**📊 数据集**

实验使用自定义合成负载（10/50/100/500 节点，10/100/1000 msg/s）进行压力测试，并未引用公开数据集。

**📈 对比分析**

在 Cloudlab 机器上运行 60 秒实验，测量延迟、CPU 占用和内存，结果显示在 10–100 节点时延迟低于 50 ms、CPU 低于 76%、内存低于 7 MB；当节点数升至 500 时性能下降明显，表明在大规模场景下仍保持可接受。

**⚠️ 局限性**

主要局限是单线程轮询实现导致文件描述符上限（1024）被突破，进而在 500 节点时产生显著延迟和 CPU 占用；未来需改进守护进程的轮询机制或实现多机部署以提升可扩展性。

---

## 333. Expressive Guitar Synthesis via Controllable Latent Flow Matching

**arXiv ID:** 2610.36786 | [PDF](https://arxiv.org/pdf/2610.36786v1)

**作者:** Hsueh-Wei Fu `[一作]`, Yi-Hsuan Yang `[通讯]`

**关键词:** `Sound` `Generation` `Data Synthesis` `Transformer` `Flow-based Model` `Diffusion model` `Prompt Engineering` `CFG (Classifier-Free Guidance)` `Audio`

**🎯 论文内容**

提出了一个两阶段的可控吉他合成管线，先预测每弦的音高与幅度曲线，再用潜在流匹配合成音频；

**💡 创新点**

核心创新在于在两阶段均引入门控跨弦注意力，显式中间控制曲线以及使用伪速度预测器实现MIDI‑仅合成；

**🔧 技术方法**

使用了轴向Transformer+门控跨弦注意力、AdaLN调制的离散时间流匹配（Latent Flow Matching）、Stable Audio Open 语义潜在解码器以及CFG引导；

**📊 数据集**

采用公开的 GuitarSet 数据集进行训练与评估；

**📈 对比分析**

与单阶段模型和 DDSP‑Guitar 进行对比，控制精度（音高与幅度误差）显著提升，MOS 在内容正确性、表现力、音质和整体偏好上均高于基线；

**⚠️ 局限性**

局限包括对多弦混响特性的依赖需训练专门的 FIR 滤波器、缺乏对录音环境和音色的条件控制、以及潜在对大规模多样化吉他演奏的泛化能力有限。

---

## 334. Task-Utility IQA: Rethinking Image Quality Assessment in Embodied Tasks

**arXiv ID:** 2610.35905 | [PDF](https://arxiv.org/pdf/2610.35905v1)

**作者:** Jirong Zha `[一作]` (Tsinghua University), Xinlei Chen `[通讯]` (Tsinghua University)

**通讯引用:** 18123 | [OpenAlex ID](https://openalex.org/A5016095713)

**关键词:** `Computer Vision and Pattern Recognition` `Explainability and Interpretability` `Robotic Intelligence` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Vision Language Model` `Image` `Multimodality` `Text`

**🎯 论文内容**

提出了一种基于多模态大语言模型（MLLM）的具身图像质量评估（IQA）框架，将图像质量判定从无目标评分转变为针对具体任务的可解释、决策导向的评价。

**💡 创新点**

创新点在于：①将IQAs视为目标条件的实用性判断而非单一感知评分；②利用MLLM的推理和知识检索实现任务知识化、区域化的质量解释；③通过“任务条件实用性得分”（Task‑Conditioned Utility Score，TCUS）直接预测下游任务成功率，证明传统指标与任务效果高度失配。

**🔧 技术方法**

核心技术包括：多模态大语言模型（CLIP‑IQA、Q‑Instruct、Q‑ALIGN 等）、检索增强生成（RAG）用于知识基底、指令调优与人类反馈（DPO）来对齐任务效用、视觉可视化与区域定位、以及工具调用（去噪、对比增强等）支持决策。

**📊 数据集**

主要使用的评估数据集为 VisDrone2019‑DET（UAV 视角目标检测数据），将检测标注转化为 VQA 样式的存在/计数任务，用于衡量不同 IQA 指标对任务成功的预测能力。

**📈 对比分析**

与传统 IQA 评估（PSNR、SSIM 等）以及现有 MLLM‑IQA 基线（Q‑SiT、Q‑Insight、DeQA‑Score‑Mix3 等）对比，TCUS 在 AUROC=0.925、AP=0.948、Spearman r=0.710 的指标上显著优于其他方法，显示其更好地捕捉了任务实用性。

**⚠️ 局限性**

局限性包括：①仅在 UAV‑VQA 场景进行验证，未覆盖所有具身任务；②TCUS 仍是经验性代理，缺乏统一的任务实用性度量；③MLLM 可能产生幻觉、推理不稳定且计算成本高，需在实时边缘部署前进一步优化与不确定性评估。

---

## 335. Why and How to Do Science When AI Is Better at It: A Framework Grounded in Human Understanding and Direction

**arXiv ID:** 2610.36116 | [PDF](https://arxiv.org/pdf/2610.36116v1)

**作者:** Hoda Heidari `[一作]` (Carnegie Mellon University), Aydin Mohseni `[通讯]` (Carnegie Mellon University)

**通讯引用:** 83 | [OpenAlex ID](https://openalex.org/A5060168471)

**关键词:** `Artificial Intelligence` `Review/Survey Paper`

**🎯 论文内容**

提出了一套基于“可持续的集体人类理解与方向”原则的两臂测试框架，阐述了 AI 在科研各阶段的使用边界与建议。

**💡 创新点**

首次将科学的核心目标界定为持续的集体人类理解，并将此目标拆解为“人类理解”和“人类方向”两条检验标准，提供了跨阶段的实践方案。

**🔧 技术方法**

主要采用哲学与社会科学的理论分析与案例讨论，未涉及具体技术实现或算法实现。

**📊 数据集**

未使用任何数据集；所有论证基于文献综述与概念推演。

**📈 对比分析**

无实验或数值比较，所有结论均基于理论推理与案例讨论，未给出可量化性能指标。

**⚠️ 局限性**

缺乏实证验证与可操作性评估，过度依赖哲学假设，实际应用时可能面临细节执行与跨学科适配的挑战。

---

## 336. StereoSplat: Metric-Scale Novel View Synthesis via Stereo-Grounded Gaussian Splatting

**arXiv ID:** 2610.35902 | [PDF](https://arxiv.org/pdf/2610.35902v1)

**作者:** Vladimir Yugay `[一作]` (University of Amsterdam), Diogo C. Luvizon `[通讯]` (Meta)

**关键词:** `Graphics` `Depth Estimation` `Generation` `Compression` `Transformer` `Gaussian Splatting` `Optical Flow` `Contrastive Learning` `Auto Encoder` `Video` `Image`

**🎯 论文内容**

提出了一种专为立体视频设计的前向3D高斯斑点重建框架，实现了度量尺度的全局几何与高质量视角合成。

**💡 创新点**

创新在于利用已校准立体匹配基座提供的度量深度作为几何锚，结合跨视差融合解码器与可学习的逆深度聚合，实现了高精度度量重建与压缩。

**🔧 技术方法**

使用立体基础模型FoundationStereo作为特征提取，基于Transformer的跨视差融合，门控深度细化，逆深度哈希聚合以及3D高斯斑点渲染。

**📊 数据集**

在大规模合成立体视频数据集（约137k帧，20k序列，包含9室内+4室外、不同视角、基线配置）上训练，并在Infinigen、TartanAir v2、KITTI等真实/合成数据上评估。

**📈 对比分析**

与AnySplat、VicaSplat、Depth Anything 3、MVSplat、PixelSplat、DepthSplat等基线对比，取得几何误差低于0.05、δ1>95%、PSNR≈29-30，显著优于现有方法。

**⚠️ 局限性**

仅适用于已校准立体对，无法处理单目或未标定输入；假设场景静态，对动态视频重建效果下降；依赖立体基座的推理速度限制。

---

## 337. A lightweight C Code Generator for Heterogeneous TinyML Systems

**arXiv ID:** 2610.36907 | [PDF](https://arxiv.org/pdf/2610.36907v1)

**作者:** Chen Liu `[一作]` (Dresden University of Technology), Christian Mayr `[通讯]` (Dresden University of Technology)

**关键词:** `Software Engineering` `Computational Efficiency` `Optimization` `Convolutional Neural Network` `Prompt Engineering` `Auto Encoder` `Image`

**🎯 论文内容**

实现了一个轻量级的 C 代码生成器，能够将 TFLite 模型转换为纯 C 代码，并通过 AOT 优化（权重调优与偏置融合）实现对定量模型的高效部署，支持基于模板的域特定加速器集成。

**💡 创新点**

创新点在于：①采用 AOT 权重调优与偏置融合消除运行时浮点重缩放与 CPU 偏置加法；②使用纯 C 代码生成，避免运行时依赖；③通过模板机制实现对任意 DSA 的灵活映射，且仅需在编译阶段完成维度重排；④一次性生成完整可编辑的源代码，提升透明度与可维护性。

**🔧 技术方法**

技术手段包括：Python 接口、TFLite 前端解析、IR 生成与优化（如权重调优、偏置融合、维度重排、内存对齐、ping‑pong 缓冲）、Jinja2 模板生成 C 代码、针对 RISC‑V DSP 的定制 MLA 调度、以及对量化参数（scale、zero‑point）进行处理。

**📊 数据集**

使用自研雷达图像数据集进行四分类（车、骑行者、行人、噪声）任务，训练出 RadarNet（9,704 参数）并取得 92% 准确率。

**📈 对比分析**

通过与 microTVM（无加速器）和 microTVM+UMA（TVM 通过 UMA 接口加速）三种方法对比，衡量单次推理时间、可执行文件大小和手工干预量。实验结果表明：我们的生成器推理时间 630µs、二进制 25kB，比分别为 microTVM（4571µs、39.3kB）和 microTVM+UMA（1470µs、31.2kB）快约 7× 与 2.3×，且二进制大小减小 63.6% 与 79.4%。

**⚠️ 局限性**

局限性包括：①目前仅支持 CNN 常见算子（卷积、全连接、池化、激活等），需手动扩展以支持其他算子；②为每个新后端准备完整模板集合是一项一次性但耗时的工作；③权重调优与偏置融合在极端量化场景下可能带来微小精度损失；④对非 RISC‑V 或不具备 bit‑shift 输出的 DSA 的兼容性尚未充分验证。

---

## 338. Instruction Capacity: Behavior and Degradation of Language Models under Controlled Load

**arXiv ID:** 2610.36095 | [PDF](https://arxiv.org/pdf/2610.36095v1)

**作者:** Hennadii Kutomanov `[一作]` (Independent researcher), Hennadii Kutomanov `[通讯]` (Independent researcher)

**关键词:** `Machine Learning` `Computational Efficiency` `Explainability and Interpretability` `Data-Centric Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Contrastive Learning` `Statistical Modeling` `Text`

**🎯 论文内容**

设计并实现了一个测量语言模型在指令负荷下执行质量的工具，利用单词替换规则将指令拆解为独立的伯努利试验，并构建统计模型来描述正确替换概率随文本长度、词位、规则数等维度的变化。

**💡 创新点**

提出了以规则数为单位的指令容量（Instruction Execution Capacity, IEC）概念，并给出了带不确定度的量化方法；使用独立可计算的单词替换指令构造最小化互相干扰的负荷，同时通过随机效应和分箱、近似平移校正等技术实现可比、可解释的统计描述。

**🔧 技术方法**

采用Python实现的实验仪器；统计分析使用广义线性混合模型（GLMM）对logit进行分箱回归，随机效应校正；使用分箱和最小二乘/最小距离估计平滑规律；使用delta方法计算容量不确定度。

**📊 数据集**

实验数据来自自定义的无语义词汇表（2220个单词），生成1890个实验点，2958个请求，共642,430个试验；目标模型为GPT‑4o mini（128k上下文、16,384最大回答长度）。

**📈 对比分析**

通过对比不同负荷维度的概率曲线，计算出在置信水平95%时的指令容量约为37–38条规则；在50%水平时约为18–19条规则；该方法与现有研究相比提供了可量化且可复现的负荷度量，并可与其它模型进行直接比较。

**⚠️ 局限性**

局限性包括：①仅评估单词替换这类“原子”指令，无法直接映射到更复杂的自然语言指令；②使用无语义词汇，缺乏对语义理解的考察；③受模型token限制，测量范围受限；④对其它模型的推广需要重新生成词表并适配不同分词器。

---

## 339. Small agents should know what their tools do

**arXiv ID:** 2610.35981 | [PDF](https://arxiv.org/pdf/2610.35981v1)

**作者:** Alexander Kovalenko `[一作]` (Czech Technical University in Prague), Artem Redchych `[通讯]` (Czech Technical University in Prague)

**关键词:** `Software Engineering` `Optimization` `Reinforcement Learning from Human Feedback` `Computational Efficiency` `Explainability and Interpretability` `Data-Centric Learning` `Transformer` `Supervised Fine-Tuning` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `World Model` `Text` `Benchmark`

**🎯 论文内容**

提出了EffectWorld，一个具备工具效应标注、回放验证器和优化器的可执行工具环境，利用该环境对小型开源语言模型进行监督微调并在多项评测中显著提升工具调用效果。

**💡 创新点**

创新点在于将工具的效应（纯/读/写/破坏性）、幂等性和读写足迹直接纳入工具描述，使得效果安全和调用效率可被自动验证；同时构造了基于优化器的负样本生成和效果安全违规标签，推动了负样本训练方法在工具调用任务中的研究。

**🔧 技术方法**

主要技术包括：LoRA低秩适配器、基于规则的回放验证器、编译器式的程序模式调用和调用优化器（常数折叠、冗余读取/调用消除、批量化、调度），以及多种负样本训练目标（UL、NPO、DPO）。

**📊 数据集**

使用了自定义的EffectWorld数据（18个工具，9种任务生成器）、公开工具调用语料（Nemotron‑Agentic、Hermes、ToolACE）以及BFCL和NESTful等基准测试集。

**📈 对比分析**

对比方法包括：原始基线模型、纯监督微调、负样本训练（UL、NPO、DPO）以及多种数据处理方案。结果显示，微调后模型在分布内任务成功率从约53%提升至99%，在未见任务种类和在线购物领域分别提升约30%和8%，BFCL多轮准确率提升至26.6%，NESTful F1从0.39提升至0.52。

**⚠️ 局限性**

局限性：EffectWorld任务规模有限（仅9类任务，单域）；脚本用户不变；优化器的消除规则受工具足迹信息限制；未覆盖长序列、跨域和失败重试场景；仅评估4B级模型，未探究更大模型或完全微调的表现。

---

## 340. NextEval: An Agent-Based Solver and Benchmark for Derivative-Free Optimization

**arXiv ID:** 2610.36553 | [PDF](https://arxiv.org/pdf/2610.36553v1)

**作者:** Shuo Chen `[一作]` (Independent Researcher), Zaikun Zhang `[通讯]` (Sun Yat-sen University)

**关键词:** `Machine Learning` `Optimization` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Tabular` `Benchmark`

**🎯 论文内容**

本文提出了NextEval，一个将大型语言模型（LLM）作为持续代理与数值环境协同工作的无导数优化（DFO）求解器，并设计了NextEval Bench基准协议来比较不同LLM+工具配置在同一预算下的性能。

**💡 创新点**

创新点包括：① 通过LLM代理动态选择、配置并切换数值工具，取代传统固定策略；② 将数值工具抽象为可调用工具，维护共享评估历史和方法状态，实现跨方法继续；③ 设计统一的Benchmark协议，统一任务、工具、预算及信息约束，对LLM-工具组合进行可重复的对比。

**🔧 技术方法**

采用的技术包括：大型语言模型（如GPT‑5.6、Qwen 3.8 Flash、DeepSeek V4 Flash）与工具调用框架、数值环境管理评估历史与工具状态、分析与搜索工具（信赖域、poll搜索、几何修复等）、OptiProfiler用于性能/数据剖面计算、实验记录与验证。

**📊 数据集**

使用的基准数据集为OptiProfiler的S2MPJ集合（92个无约束低维（n≤5）优化问题），并在任务中加入起点扰动、噪声、坐标旋转等特征。

**📈 对比分析**

通过对比NextEval与COBYQA、NEWUOA、SciPy Powell和BFGS等经典DFO基线，在50n评估预算下，以相对目标收敛度为准则，使用性能剖面（ρ）和数据剖面（δ）评估；NextEval在所有指标上均获得最高的聚合得分，说明在评估次数效率上优于传统方法。对不同LLM模型、推理设置与执行harness的对照实验进一步表明，模型选择和推理增强会显著提升性能。

**⚠️ 局限性**

限制包括：仅在低维≤5的公开问题上验证，未检验对未知问题族的泛化；旋转坐标仅保持目标值结构，未实现完全匿名；评估仅计数函数调用，未衡量运行时或LLM调用成本；未对数值工具单独进行消融，无法单独归因于LLM；未扩展到高维或更复杂问题。

---

## 341. uTHCD-C: A Corruption-Robustness Benchmark for Tamil Handwritten Character Recognition

**arXiv ID:** 2610.36850 | [PDF](https://arxiv.org/pdf/2610.36850v1)

**作者:** Preethi S `[一作]` (Kongu Engineering College), Arun Prakash A `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Convolutional Neural Network` `Transformer` `Image` `Benchmark`

**🎯 论文内容**

构建并发布了 uTHCD-C，针对 Tamil 手写字符识别的十种人工腐败（含笔画变细）进行系统鲁棒性评估。

**💡 创新点**

首次为 Tamil HCR 设计多级腐败基准，提出 mCE 归一化指标，并通过十种腐败与五个严重级别统一比较模型鲁棒性。

**🔧 技术方法**

采用 ImageNet 预训练的卷积与 Transformer 模型，进行五折分层交叉验证与 writer‑independent 测试，计算 mCE、参数量、GMACs 与推理吞吐。

**📊 数据集**

使用 uTHCD（90,950 张图像，156 类，753 位作者）的 Tamil 手写字符数据集。

**📈 对比分析**

通过 mCE 与 VGG16‑BN 归一化比较，ConvNeXt 与 Swin 分别取得 0.75 与 0.83 的 mCE；ShuffleNet 与 MnasNet 分别为 1.55 与 1.85；吞吐最高的 ShuffleNet 约 30,591 img/s，但鲁棒性最差；Pareto 分析展示吞吐–mCE 取舍。

**⚠️ 局限性**

仅使用合成腐败，缺乏对真实失真文档的验证；吞吐测量受特定硬件/实现限制；仅评估单一数据集，结果对其他脚本或任务的泛化性未知。

---

## 342. AudioRSI: Recursively Self-Improving Audio Reasoners

**arXiv ID:** 2610.36578 | [PDF](https://arxiv.org/pdf/2610.36578v1)

**作者:** Jiajun Fan `[一作]` (University of Illinois Urbana-Champaign), Ivan Bulyko `[通讯]` (Amazon)

**关键词:** `Sound` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Reinforcement Learning` `Chain-of-Thought` `Prompt Engineering` `Audio` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

提出了AudioRSI，一个利用音频大型语言模型自身评判其推理路径并在训练中递归更新的强化学习框架。

**💡 创新点**

创新点在于：①让模型自身充当批评家，产生过程级奖励而非仅依赖外部或手工规则；②采用周期性同步的递归结构，使得每一次改进的模型成为下一阶段的评判者，从而让奖励随策略进步而提升。

**🔧 技术方法**

使用了GRPO（Group-Reward Policy Optimization）强化学习算法、基于模型的自评判（pairwise comparison）与周期性批评家同步，并在此基础上构建了无外部奖励模型的音频推理系统。

**📊 数据集**

主要数据集为AVQA用于训练，评估使用四个OOV基准：MMAU test-mini、MMSU、MMAU-Pro 和 MMAR。

**📈 对比分析**

与基准（Ke-Omni-R、Omni-R1、Audio Flamingo 3、Gemini 2.5 Pro 等）相比，AudioRSI 在三项OOV基准的7B开放源码层次上表现最佳，且在MMAU test-mini上取得最高分；结构化推理表现显著提升。

**⚠️ 局限性**

局限性包括：实验仅在单一模型骨干和单一训练语料上验证，递归机制在不同规模或任务（如图像、蛋白质）中的泛化性尚未探究；周期参数T对性能影响敏感，需要进一步调优。

---

## 343. LabWorld: Learning Interactive 4D Worlds for Scientific Experimentation

**arXiv ID:** 2610.36592 | [PDF](https://arxiv.org/pdf/2610.36592v1)

**作者:** Suozhi Huang `[一作]` (Princeton University), Mengdi Wang `[通讯]` (Princeton University)

**关键词:** `Computer Vision and Pattern Recognition` `Robotic Intelligence` `Optimization` `Transformer` `Vision Language Model` `Generative Adversarial Network` `Auto Encoder` `Vision-Language-Action Model` `Diffusion model` `Optical Flow` `World Model` `Simultaneous Localization and Mapping` `Trajectory Optimization` `Video` `Image` `Text`

**🎯 论文内容**

开发了一个基于视觉语言模型和多帧观察的 Real2Sim 框架 LabWorld，将单目 RGB 实验室视频转换为可编辑的交互式 4D 世界，包含可复用资产、度量场景、时间变化的物体状态和程序事件。

**💡 创新点**

首次将事件约束融入运动重建，通过实验程序中的结构化事件约束保证视觉一致性与物理有效性，并实现了可重复使用的 3D 资产与时间连续的场景重建。

**🔧 技术方法**

使用 SAM3 分割、MapAnything 深度与相机估计、生成式 3D 资产合成、基于视觉、几何、时间、事件和物理约束的轨迹优化，以及 vision‑language agent 生成事件列表。

**📊 数据集**

在自家实验室收集的十条移液（pipetting）和十条振荡（vortexing）单目 RGB 视频作为评估数据集。

**📈 对比分析**

与 GPT‑6 Astra、Do as I Do、FoundationPose 等基线相比，LabWorld 在移液任务上平均 Mask IoU 0.501（最高），在振荡任务上 Tube‑Length–Normalized PCK‑AUC 0.638（最高），且移除事件约束后物理约束满足率从 35/35 降至 5/35。

**⚠️ 局限性**

在强手部遮挡情况下几何与运动估计不确定，事件约束效果依赖于事件提取与场景几何的精度。

---

## 344. AGDecode: Semantic Attribute Grammar Derivatives for Safety-Constrained Decoding of Large Language Models

**arXiv ID:** 2610.36105 | [PDF](https://arxiv.org/pdf/2610.36105v1)

**作者:** Rajan Maghera `[一作]` (University of Toronto), Maryam Mehri Dehnavi `[通讯]` (University of Toronto)

**通讯引用:** 808 | [OpenAlex ID](https://openalex.org/A5003682451)

**关键词:** `Software Engineering` `AI Code Assistant` `Safty and Privacy` `Transformer` `Large Language Model` `Prompt Engineering` `Constraint Satisfaction` `Formal Verification` `SMT/CHC Solving` `Text`

**🎯 论文内容**

在LLM生成程序的过程中，设计并实现了 AGDecode 框架，实现语义约束的实时解码。

**💡 创新点**

创新点在于把属性语法推广到不完整语法树，构造残留属性语法并转化为关系约束，利用外部 SMT/CHC 求解器进行可实现性判定，从而实现通用可编程的语义约束。

**🔧 技术方法**

技术手段包括递归属性语法、残留属性语法、关系约束系统、Horn 子句化、SMT/CHC 求解器（Z3 Spacer）、Coq 形式化验证以及 Python 实现。

**📊 数据集**

使用 HumanEval C++ 子集（映射到 C）共 77 个测试用例，评估 C 程序在空指针、除零和未初始化等未定义行为方面的安全性。

**📈 对比分析**

与无约束生成相比，AGDecode 在安全性上显著提升；token‑级求解平均耗时 0.2–2.6 s，约 25% 的查询超时；整体生成在可接受的计算开销内实现了路径敏感安全属性。

**⚠️ 局限性**

局限性包括需要手工编写完整语法与属性约束、缺乏回溯机制、token 级求解延迟高、依赖求解器完整性以及对生成质量可能产生负面影响。

---

## 345. Multiple Leak Localization in Water Distribution Networks through Graph Neural Networks

**arXiv ID:** 2610.35972 | [PDF](https://arxiv.org/pdf/2610.35972v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Machine Learning`

---

## 346. The Batch Size Can Be One: Test-Time Adaptation from Classifier Geometry

**arXiv ID:** 2610.36628 | [PDF](https://arxiv.org/pdf/2610.36628v1)

**作者:** Jaeho Hwang `[一作]` (Ewha Womans University), Uiwon Hwang `[通讯]` (Ewha Womans University)

**关键词:** `Machine Learning` `Domain Adaptation` `Classification` `Representation Learning` `Convolutional Neural Network` `Transformer` `Contrastive Learning` `Supervised Fine-Tuning` `Image`

**🎯 论文内容**

提出OriTTA，一种单样本测试时自适应方法，通过在源模型的几何框架中重新定位特征空间原点，利用合成锚点定义理想点、轴和圆锥，对特征进行角度对齐和半径约束，并通过记忆库检测并恢复模型崩溃。

**💡 创新点**

创新点包括：①从源模型直接构建“原点重定位框架”而非依赖测试流统计；②为每类设定理想点、轴及圆锥，形成固定的几何目标；③使用不依赖批统计的角度交叉熵与半径惩罚损失；④基于类均衡记忆库的崩溃检测与恢复机制。

**🔧 技术方法**

采用线性分类头、特征投影到理想子空间、重定位原点、合成锚点生成、角度对齐损失、半径惩罚、记忆对齐损失、崩溃检测并恢复源权重等技术。

**📊 数据集**

使用CIFAR-10-C、CIFAR-100-C、ImageNet-C（15种畸变）三个数据集，配合WRN-28-10、WRN-40-2、ResNet-50+GroupNorm、ViT-B/16四个backbone。

**📈 对比分析**

与TENT、EATA、SAR、DeYO、ReCAP、SaTeen等基线在batch size=1、episodic/continual协议以及shuffle和class-blocked流下进行对比。OriTTA在所有16种设置下均实现最高平均准确率，尤其在ImageNet-C fog上从未崩溃，平均提升约2–3个百分点，并在100个种子实验中表现出最佳稳定性。

**⚠️ 局限性**

局限性：当理想子空间几乎为整个特征空间或极少时（如ViT-B/16）优势减弱；当类别数小或原点偏移微弱（如CIFAR-10）时重定位效果有限；方法假设线性头且类别分布均匀，对长尾或非线性头适用性未知；实验仅涵盖合成畸变，未验证自然迁移场景。

---

## 347. SpecLST: Spectral-Enhanced Long-Short Term Spatio-Temporal Multimodal Modeling for Traffic Forecasting

**arXiv ID:** 2610.36393 | [PDF](https://arxiv.org/pdf/2610.36393v1)

**作者:** Kenan Kang `[一作]` (Auckland University of Technology), Weihua Li `[通讯]` (Auckland University of Technology)

**关键词:** `Machine Learning` `Traffic Forecasting` `Autonomous Driving` `Optimization` `Computational Efficiency` `Graph Neural Network` `Transformer` `Recurrent Neural Network` `Mixture of Experts` `Attention Mechanism` `Frequency Domain Transformation` `Residual Learning` `Time Series` `Graph` `Multimodality` `Tabular`

**🎯 论文内容**

提出SpecLST框架，将两周历史交通流的低频频谱、实时天气/日历/事件信息通过注意力融合，结合稀疏Mixture-of-Experts图卷积和高频残差头，实现多模态长短期时空交通预测。

**💡 创新点**

创新点在于：①使用FFT将长序列压缩为低频系数，避免大规模自注意力；②注意力融合多模态上下文；③引入可学习稀疏邻接增强图结构；④设计频率增强残差头专门捕捉突发事件的高频信号。

**🔧 技术方法**

采用FFT频域变换、Transformer式注意力、多头自注意力、稀疏Mixture-of-Experts、图卷积、门控残差头以及可学习邻接矩阵等技术。

**📊 数据集**

实验使用四个PeMS公开交通数据集（PEMS-BAY、PEMS-BAY22、PEMS04、PEMS08），并在PEMS-BAY22中额外整合天气、日历和社交媒体事件信息。

**📈 对比分析**

与VAR、SVR、DCRNN、MTGNN、LSTTN、STAEformer等统计、图卷积和Transformer基线比较，SpecLST在所有四个数据集、所有预测时隙均取得MAE/MAPE/RMSE均优于或相近，尤其在PEMS-BAY22和PEMS08上实现最佳表现，并在内存占用和训练时间上位于Pareto前沿。

**⚠️ 局限性**

局限性包括：可学习邻接矩阵为时间不变，难以实时捕捉交通突发的时变空间关联；频域低频表示可能忽略非周期性趋势；事件信息仅在PEMS-BAY22可用，跨区域泛化能力尚待提升。

---

## 348. Cross-Day Compatibility of Daily Item Embeddings: An Empirical Study of Alignment, Scale Management, and CTR Adaptation

**arXiv ID:** 2610.36408 | [PDF](https://arxiv.org/pdf/2610.36408v1)

**作者:** Yicheng Wang `[一作]`, Yicheng Wang `[通讯]`

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Graph Neural Network` `Contrastive Learning` `Representation Learning` `Auto Encoder` `Tabular` `Sequential`

**🎯 论文内容**

研究了每日物品共现嵌入在多日CTR训练中的跨版本兼容性问题，并系统评估了多种上游稳定化与下游任务学习技术。

**💡 创新点**

提出将ID初始化、正交对齐、温暖启动与全局缩放等组合应用于Word2Vec与XSimGCL，并证明温暖启动与全局缩放能显著提升GAUC，同时CTR训练的编码器可进一步弥补残余漂移。

**🔧 技术方法**

使用Word2Vec (SGNS)、图邻域对比学习 XSimGCL、正交对齐 (OP)、多种映射 (affine, MLP)、单位/均值归一化、温暖启动、全局重缩放、球面约束、CTR任务的多层感知机与降维编码器等。

**📊 数据集**

基于LY公司广告业务的日常曝光与点击日志，包含约百万级物品和至少数百万条训练样本，采用天数分割进行多日嵌入与CTR评估。

**📈 对比分析**

采用固定候选集的Recall@100、GAUC、LogLoss与跨日余弦/范数等指标，对48个配置（32个CTR）进行对比，发现ID初始化+OP提升GAUC 2.6%，温暖+全局缩放最优提升GAUC 4.0%，CTR训练编码器在多日余弦上提升10%以上。

**⚠️ 局限性**

实验仅覆盖单一工业场景与单日窗口，未评估在线CTR提升，缺乏多次实验与统计校正，且温暖启动与缩放效应未完全分离。

---

## 349. GeoGSplat: Geometry-Grounded Non-Pixel-Aligned Gaussian Splatting

**arXiv ID:** 2610.36760 | [PDF](https://arxiv.org/pdf/2610.36760v1)

**作者:** Jiale Wang `[一作]` (Xi'an Jiaotong University), Tong Zhang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Depth Estimation` `3D Reconstruction` `Transformer` `Convolutional Neural Network` `Gaussian Splatting` `Vision Transformer` `Attention Aggregation` `Neural Radiance Field` `Chamfer Distance` `Rendering Loss` `Image` `Point Cloud` `Mesh`

**🎯 论文内容**

提出 GeoGSplat，一种基于几何先验的非像素对齐高斯散射框架，用于高质量的新视角合成与3D重建。

**💡 创新点**

通过双分支编码器同时生成 Anchor 球体和 Triplane 结构，并使用层次化解码器在 Anchor 约束下逐步生成稠密点再预测高斯原语，实现固定高斯预算且具备显式几何约束。

**🔧 技术方法**

采用 Vision Transformer 作为图像编码器，Plücker 光线嵌入，注意力聚合，Triplane 细化、上采样与残差卷积，以及几何监督（Chamfer Distance）与渲染损失结合训练。

**📊 数据集**

在 ScanNet++、ARKitScenes、BlendedMVS、CO3D、WildRGB-D 上训练，评估使用 DTU 与 NRGBD 数据集。

**📈 对比分析**

与像素对齐方法（MVSplat、DepthSplat、AnySplat）以及非像素对齐方法（TokenGS、VolSplat）对比，GeoGSplat 在相同高斯预算下 PSNR 最高、LPIPS 最低，并在更大视角变换下优势更明显。

**⚠️ 局限性**

仍依赖预先标定的相机姿态和大量训练数据，且在极端遮挡或纹理稀疏场景中高斯中心可能偏移，未探讨可变高斯预算和实时推理速度。

---

## 350. Causal Localization of Evidence Effects in Transformers

**arXiv ID:** 2610.36524 | [PDF](https://arxiv.org/pdf/2610.36524v1)

**作者:** Jipeng Li `[一作]`, Yanning Shen `[通讯]`

**关键词:** `Computation and Language` `Explainability and Interpretability` `Transformer` `Prompt Engineering` `Contrastive Learning` `Text`

**🎯 论文内容**

提出一种受控符号输入的证据聚合方法（CEA），通过改变符号计数和顺序来探测 Transformer 的注意力头对决策的因果影响。

**💡 创新点**

创新点在于：①使用受控符号输入以消除自然语言中词义混杂的干扰；②设计双向去除与恢复（BUC）搜索策略，联合评估去除和恢复效应；③给出条件理论证明，将注意力头归因误差与干预效果关联。

**🔧 技术方法**

采用冻结的自回归 Transformer（如 Pythia‑410M）、注意力写入量向量、对比对数值、去除（ablation）与均值补丁（mean patching）等技术；实现了前后交错、随机等四种符号顺序的实验设计。

**📊 数据集**

使用的主要数据集为：Pythia‑410M 训练好的模型权重；以及一个人工构造的自然语言桥接任务，包含 16 条支持两种互斥状态的报告。

**📈 对比分析**

与随机头集、单头敏感度、ActPatch 等方法进行对比；在符号计数实验中，最佳头集在 12 头以内即可恢复 70%~95% 的效应；在自然语言桥接任务中，受控符号头集的准确率下降约 8%，显著高于随机控制（约 0.8%）。

**⚠️ 局限性**

局限性包括：仅检验最终提示位置的写入影响，无法覆盖完整的 head‑MLP 路径；受控符号输入与真实自然语言推理的关联有限；搜索成本未与头数完全独立；缺乏对多模型和更长文本的泛化验证。

---

## 351. Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception

**arXiv ID:** 2610.37037 | [PDF](https://arxiv.org/pdf/2610.37037v1)

**作者:** Lai Wei `[一作]` (Shanghai Jiao Tong University), Weiran Huang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `Computer Vision and Pattern Recognition` `Data Synthesis` `Multimodal Perception` `Representation Learning` `Vision Language Model` `Reinforcement Learning` `Contrastive Learning` `Prompt Engineering` `Retrieval-Augmented Generation` `Vision-Language-Action Model` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出了一种完全自动化的多区域视觉问题合成框架，用于训练视觉语言模型（VLM）连接图像内的证据；

**💡 创新点**

创新点在于将多区域组合问题与跨模型答案一致性、区域消除验证以及多选题转换相结合，形成可验证的训练样本；

**🔧 技术方法**

采用了强化学习与可验证奖励（RLVR）、跨模型答案一致性检验、区域消除测试以及多选题生成等技术；

**📊 数据集**

使用从公开图像库自动合成的20.4K个多区域问题样本（由Zooming-without-Zooming v2生成），并在多种公开基准上进行评估；

**📈 对比分析**

在25个多模态感知基准（包含细粒度感知、空间感知、长上下文理解、具身空间推理、代理任务和通用视觉推理）上，与基线模型和现有合成数据集相比，ZwZ2在所有规模下均实现了显著提升（平均提高约5-6点，最高模型在整体准确率上达60.51%）；

**⚠️ 局限性**

局限性包括对长上下文检索能力提升有限（最长上下文表现趋于收敛）、对单图像训练的依赖导致对多图像或交互式场景的迁移仍受限，以及验证过程对模型对齐的依赖可能导致部分高质量样本被误删。

---

## 352. Hierarchical Multimodal Interaction Reasoning for Dyadic Relationship Recognition

**arXiv ID:** 2610.36249 | [PDF](https://arxiv.org/pdf/2610.36249v1)

**作者:** Wang Tang `[一作]` (Chengdu University of Information Technology), Hatice Gunes `[通讯]` (University of Cambridge)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Transformer` `Contrastive Learning` `Mixture of Experts` `Multimodality` `Audio` `Text` `Image` `Video`

**🎯 论文内容**

本文提出了一种层次化多模态交互推理框架 HiRe-DRR，用于识别两人之间的隐含社会关系；

**💡 创新点**

创新点在于：①从特征层级、交互类型和关系层级三维组织关系证据；②引入 Mixture-of-Interaction Transformer（MoI-Former）通过稀疏路由自适应选择与关系相关的交互特征；③设计 Hierarchical Conditional Loss（HC-Loss）在层次化监督下提升关系语义学习；

**🔧 技术方法**

技术上结合了面部、身体、音频与文本的低层与高层特征提取，构建同质与异质交互关系，使用 Transformer 进行稀疏路由与上下文交互，最终通过 HC‑Loss 进行层次化分类；

**📊 数据集**

实验使用了三大公共数据集：NoXi、UDIVA 与 Seamless Interaction（SeamInt）；

**📈 对比分析**

与多种基准方法（GR2N、HGM、PGCN、LIReC、CT、AsyReC）对比，HiRe-DRR 在 NoXi、UDIVA 与 SeamInt 的 UAR/mUAR/mAP 均实现最高水平，例如 NoXi 上 UAR 63.6%、mAP 63.1%，UDIVA 上 UAR 66.8%、mAP 66.8%，SeamInt 上 UAR 68.5%、mAP 68.5%；

**⚠️ 局限性**

局限性在于：当前评测基准仅覆盖短时交互，缺乏长期行为与多主体关系建模；相近关系类别的交互模式易重叠，导致识别困难；未来需探索长时序建模、多人物关系推理与更灵活的证据发现机制。

---

## 353. When the Application Layer Is Disposable: Putting Integrity Constraints Back in the Database

**arXiv ID:** 2610.36418 | [PDF](https://arxiv.org/pdf/2610.36418v1)

**作者:** Hasara Kalumin `[一作]` (University of Maryland), Amol Deshpande `[通讯]` (University of Maryland)

**关键词:** `Databases` `Optimization` `Computational Efficiency` `Database Management` `Tabular` `Database Management`

**🎯 论文内容**

本文提出并实现了约束索引（constraint indexes），一种将跨表约束编译为事务性访问路径的机制，并将其作为 PostgreSQL 扩展部署，用于在数据库层面强制执行多表、折叠结构、弱实体以及基数约束等非键约束。

**💡 创新点**

创新点在于：①将约束治理（constraint governor）作为核心架构，引入可插拔的约束索引实现；②利用共享内存哈希表、事务级别的逻辑锁和统一的撤销日志，实现对多表约束的低成本、事务安全维护；③通过实验展示约束索引在多线程环境下可实现 1.09×–2.41× 的吞吐量提升，同时保留 81%–98% 的无约束吞吐量。

**🔧 技术方法**

技术细节包括：PostgreSQL 16 扩展实现、共享内存哈希表分区、事务级别的 advisory lock、统一的事务撤销日志、手工实现的约束索引操作（插入、更新、删除）、以及对事务提交/回滚的回调处理。

**📊 数据集**

实验使用的自定义数据集：约 5.7 M 实体，包含 Person/Content 继承层次、折叠关系、折叠弱实体（IdCard、Address）以及多对多关系 ContributesTo，整体约 2–3 M 关联条目。

**📈 对比分析**

比较方法：在 256 个并发客户端下，对比无约束、显式查询检查、触发器检查、阴影表、阴影表触发器和约束索引六种配置。约束索引在大多数工作负载下实现了 1.09×–1.54× 的速度提升（最高 2.41×），并在 84%–96% 的无约束吞吐量范围内维持性能。

**⚠️ 局限性**

限制包括：①实现仍为实验性 PostgreSQL 扩展，缺乏完整的自动化编译器；②仅支持键相关约束，无法覆盖所有通用断言；③对事务隔离级别和并发异常（如写偏斜）的处理有限；④扩展需要通过 SQL 函数调用，未与查询计划器深度集成，导致一定的性能开销；⑤缺乏对大规模约束组合的共享与复用机制。

---

## 354. OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning

**arXiv ID:** 2610.36335 | [PDF](https://arxiv.org/pdf/2610.36335v1)

**作者:** Hui Yang `[一作]` (University of Manchester), Yongsheng Gao `[通讯]` (SNOMED International)

**关键词:** `Artificial Intelligence` `Knowledge Distillation` `Explainability and Interpretability` `Computational Efficiency` `Recommendation System` `Drug Discovery` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Agentic AI` `Prompt Engineering` `Supervised Fine-Tuning` `Text` `Biomedical Data` `Benchmark` `Medicine Related`

**🎯 论文内容**

设计并实现了一个基于本体的 agentic RAG 框架（OA‑RAG），可将正式 OWL 本体推理与 LLM 生成相结合，实现三种不同信度的推理模式。

**💡 创新点**

首次将 OWL 本体的形式推理与 LLM 结合，在 ReAct 循环中通过 SPARQL 查询与本体服务交互，并对答案给出四级 grounding 级别，既能保证答案可信度，又提供可靠性评估。

**🔧 技术方法**

采用 ReAct 交互式 LLM 框架、SPARQL 本体查询、OWL 形式推理器、BM25 名称匹配回退以及四级 grounding 评估等技术。

**📊 数据集**

使用 GO、FoodOn、SNOMED CT 三大公开本体构造的 300 MCQ / 200 OQ 评测集，以及 OntoTune 公开的 MedQA、USMLE、MedMCQA、PubMedQA 六个医学 QA 任务。

**📈 对比分析**

与基础 LLM、ontology fine‑tuned LLM、RAG、ReAct、MedRAG 等方法对比，OA‑RAG 在 Test1/2 上平均提升 10–15% 以上，Test3 与 i‑MedRAG 接近但仅使用 1/10 的检索/生成 token，整体表现最佳，并且更高的 grounding 级别对应更高的准确率。

**⚠️ 局限性**

在 OQ 的检索仍受限于文本注释的可检索性，正式推理仍依赖已建模的 axiom；在本体覆盖范围外的问题仍难以获得强 grounding，且检索多语义信息时可能需要更高成本。

---

## 355. Proof Repair and (Co)algebra Equivalences

**arXiv ID:** 2610.36669 | [PDF](https://arxiv.org/pdf/2610.36669v1)

**作者:** E. G Plante `[一作]`, Talia Ringer `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `Programming Languages` `Proof Repair` `Type Theory` `Category Theory` `Formal Verification` `Agda` `Homotopy Type Theory` `Coalgebraic Methods` `Algebraic Methods` `Formal Proof` `Review/Survey Paper`

**🎯 论文内容**

本文对类型论中的证明修复提供了第一套基础理论，证明配置等价于同一多项式函子初始代数（或最终余代数）之间的代数同构，并基于此扩展到余代数（共修复）场景。

**💡 创新点**

创新点在于将证明修复与范畴代数联系起来，揭示配置本质是代数同构，消除了对 univalence 的依赖，并首次在共代数框架下定义共修复与 M‑类型（共代数的最终化）的对应关系。

**🔧 技术方法**

所用技术包括强制性类型论（Agda 代码）、多项式函子与 P‑代数、初始代数与最终余代数的同伦性质、同伦最终化与 bisimulation、以及 cubical Agda 的路径类型。

**📊 数据集**

研究不依赖任何实验数据集，完全基于形式化证明与理论推导；所有结果均在 Agda/ Cubical Agda 环境下验证。

**📈 对比分析**

由于本工作为理论性基础性研究，没有实验对比或性能评测，主要通过证明同构唯一性与修复程序的重构来说明方法的正确性和优雅性。

**⚠️ 局限性**

局限性包括：只处理可作为同一多项式函子初始/最终代数的类型，无法直接处理索引不一致的递归族；共修复需要同伦最终化的假设，且目前尚未给出对非初始/非最终类型的通用修复方案。

---

## 356. A Hyper-Logical View on Multi-Agent Path Planning

**arXiv ID:** 2610.35964 | [PDF](https://arxiv.org/pdf/2610.35964v1)

**作者:** Hadar Frenkel `[一作]` (Bar-Ilan University), Dor Atzmon `[通讯]` (Bar-Ilan University)

**通讯引用:** 1248 | [OpenAlex ID](https://openalex.org/A5025220652)

**关键词:** `Multiagent Systems` `Optimization` `Reinforcement Learning from Human Feedback` `Explainability and Interpretability` `Model Checking` `Formal Methods` `HyperLTL` `HyperQPTL` `Logical Translation`

**🎯 论文内容**

将多智能体路径规划（MAPP）各类变体通过 HyperLTL/HyperQPTL 逻辑形式化，提供模块化的翻译框架，并将最优性与成本函数（如 makespan、SOC）视为超属性来表达。

**💡 创新点**

创新点在于：①提出统一的超逻辑规范视角，将共通约束与变体特定约束分离；②利用 HyperLTL/HyperQPTL 表达跨代理约束、最优性及成本；③展示如何将 MAPP 归约为可判定的超属性模型检测，揭示量化交替对复杂度的影响。

**🔧 技术方法**

使用技术包括：HyperLTL 与其 ω-regular 扩展 HyperQPTL、LTS 转换、模型检测理论、量化子推理、以及逻辑翻译与成本函数（makespan、SOC）作为超属性的定义。

**📊 数据集**

未使用任何实验数据集，本文主要为理论建模与逻辑翻译，未进行数据实验。

**📈 对比分析**

未进行实验比较；作者指出现有 HyperLTL/HyperQPTL 检查器（如 AutoHyper）的性能仍不及成熟的专用 MAPP 求解器，本文侧重于规范化与可解释性，而非速度优化。

**⚠️ 局限性**

局限性：①现有超属性模型检查器复杂度高，难以与成熟 MAPP 求解器竞争；②量化交替导致指数/双指数复杂度；③仅覆盖一次性（one‑shot）变体，未覆盖在线/长期变体；④SOC 的完整异步推理仍未实现。

---

## 357. Efficient Quantization Training with Stochastic Rounding via Structured Subspace Randomness

**arXiv ID:** 2610.36587 | [PDF](https://arxiv.org/pdf/2610.36587v1)

**作者:** Zeyu Wang `[一作]` (Google), Liang Liu `[通讯]` (Google)

**关键词:** `Machine Learning` `Computational Efficiency` `Optimization` `Transformer` `Large Language Model` `Stochastic Rounding` `Mixture of Experts` `Text`

**🎯 论文内容**

设计并实现了一种在低精度 INT8 训练中共享随机阈值的 Structured Subspace Stochastic Rounding (S3R) 并提出错误修正方法 S3R‑EC，以减少随机数生成开销并保持梯度无偏

**💡 创新点**

创新点在于：①将 SR 的随机阈值按 token 共享，显著降低 VPU 负载；②发现共享阈值产生的均值偏移主要通过激活梯度传播，进而提出结构化错误修正（S3R‑EC）直接消除该偏移；③实现无额外随机数生成、无优化器改动的错误修正

**🔧 技术方法**

采用了：Stochastic Rounding、结构化子空间随机性、误差均值消除、JAX/XLA 编译、Google Cloud TPU v5p 硬件加速

**📊 数据集**

在英文 C4 数据集上训练 12 层 Llama 风格解码器；在更大规模的 Llama 3.1‑8B、Qwen3‑8B、Gemma 3‑12B 以及混合专家模型 Qwen3‑30B‑A3B、DeepSeek‑V2‑Lite 16B 上进行完整训练评估

**📈 对比分析**

与 BF16、INT8 RTNE、独立 SR（Threefry）对比，S3R‑EC 在验证损失上与独立 SR 近乎一致，速度方面在密集 LLM 上比 BF16 加速 1.24–1.32×，比独立 SR 加速 1.11×；在 MoE 模型上也实现了 1.04–1.06× 的加速

**⚠️ 局限性**

局限性包括：在 per‑example 阈值共享下仍有残余误差；对 8‑bit INT8 的适用性，尚未验证更低精度（如 4‑bit）；实验只在有限的 seed 数量下评估，细微差异可能难以检测

---

## 358. Hardware-Matched Spiking Neural Networks for Real-Time Purkinje Cell Spike Classification

**arXiv ID:** 2610.37104 | [PDF](https://arxiv.org/pdf/2610.37104v1)

**作者:** Arooba Farhan `[一作]` (Lahore University of Management Sciences), Muhammad Ali Siddiqi `[通讯]` (Lahore University of Management Sciences)

**关键词:** `Hardware Architecture` `Classification` `Spiking Neural Network` `Convolutional Neural Network` `8-bit Quantization` `AdamW Training` `FP16/FP32 Backend Implementation` `Learning Time Constants` `Fixed Synaptic Delay` `Sliding Window Normalization` `Biomedical Data` `Time Series`

**🎯 论文内容**

研发并实现两种硬件匹配的稀疏神经网络，用于实时Purkinje细胞单/复发射突的分类。

**💡 创新点**

提出针对微控制器MVP和Sky130专用电路的轻量级ConvSNN和StreamSNN架构，并在未见细胞上验证其泛化能力，显著提升准确率、降低参数量与功耗。

**🔧 技术方法**

利用卷积与时序流SNN、LIF/ALIF神经元、学习时间常数、固定突触延迟、基于NEO的检测、8‑bit量化、AdamW训练及FP16/FP32后端实现。

**📊 数据集**

使用64条鼠Purkinje细胞记录（采样24.414 kHz），包含SS/CS/FP标注，采用滑动窗口归一化处理。

**📈 对比分析**

通过五折细胞留出交叉验证与随机拆分对比，ConvSNN在细胞留出下CS F1达到0.873，StreamSNN为0.837；相较SNet‑4、ResNet‑18提升0.04–0.05点，参数量更少；MVP加速后延迟降低62%，电池寿命翻倍。

**⚠️ 局限性**

局限包括对噪声估计与波形采集的依赖、事件窗口重叠导致额外延迟、未在自由行为动物中验证，以及对多硬件映射的适配性需进一步提升。

---

## 359. Personality Latents: Conditioning Frozen Language Models on Individuals from Their Own Writing

**arXiv ID:** 2610.36805 | [PDF](https://arxiv.org/pdf/2610.36805v1)

**作者:** Gunhee Cho `[一作]` (Sungkyunkwan University), Yun-Gyung Cheong `[通讯]` (Sungkyunkwan University)

**关键词:** `Computation and Language` `Representation Learning` `Generation` `Data-Centric Learning` `Safty and Privacy` `Transformer` `Large Language Model` `Prompt Engineering` `Auto Encoder` `Contrastive Learning` `Text` `Review/Survey Paper`

**🎯 论文内容**

提出了人格潜在向量（Personality Latents, PL）方法，利用个人写作文本学习无标签、无档案的低维向量，并通过软前缀注入到冻结的语言模型中实现个体化人格控制；

**💡 创新点**

创新点在于：①直接将自由潜在向量与模型优化耦合，而非仅通过编码器生成，解耦了潜在空间与编码器限制；②同一潜在向量可跨模型使用，仅需为每个模型训练小型适配器；③在无标签数据上训练并通过生成文本评估人格真实性，证明潜在向量确实携带人格信息；

**🔧 技术方法**

技术包括：生成式潜在优化（free latent optimization），软前缀注入（adapter将潜在映射为多维向量），梯度上限损失（margin loss）、正则化、编码器对潜在的近似（set encoder + attention），以及测试时潜在细化；

**📊 数据集**

使用的公开数据集有：Kaggle MBTI（8608人，MBTI标签）、Essays（2081人，二元Big Five标签）、PANDORA（1178人，连续Big Five分数），每人8-64条帖子；

**📈 对比分析**

与三类基线（Persona embedding：PPlug、DEP；提示基线：Trait Prompt、TS、In‑Context）在相同冻结模型、相同提示下进行比较。PL（精炼潜在）在MBTI轴的AUC分别为0.754（中性问题）和0.782（访谈问题），优于所有基线；在Big Five上虽表现弱但仍不劣于提示基线；测试时潜在细化显著提升性能，单一编码器预测仅为0.649/0.729 AUC，细化后提升至0.754/0.782；跨模型实验表明同一潜在向量可被Llama‑3.1‑8B‑Instruct读取，AUC 0.592；

**⚠️ 局限性**

局限性包括：生成文本的“人格”评估依赖于文本编码器和scorer，可能捕捉话题/词汇而非真实人格；Big Five数据信度低，实验表现不佳；仅使用单一3B模型及一组跨模型验证；测试细化需要每人帖子与额外优化；潜在向量可能被滥用进行假冒或隐私泄露。

---

## 360. Learning to Formalize: Bridging Open-Ended Workloads and Fast Decision Models

**arXiv ID:** 2610.36595 | [PDF](https://arxiv.org/pdf/2610.36595v1)

**作者:** Chengxuan Pei `[一作]` (CUHK-Shenzhen), Zhe Li `[通讯]` (Kunlunxin Corporation)

**关键词:** `Artificial Intelligence` `Computational Efficiency` `Knowledge Distillation` `Reinforcement Learning from Human Feedback` `Robotic Intelligence` `Transformer` `Supervised Fine-Tuning` `Knowledge Distillation` `Large Language Model` `Text`

**🎯 论文内容**

构建了一套流程，先用一个小型的候选项生成器（基于教师生成列表的知识蒸馏训练）从任务上下文中生成任务相关的候选列表，再由一个冻结的决策模型在该列表中做出选择，填补了开放式LLM任务与决策模型接口之间的缺口。

**💡 创新点**

将候选列表的构造任务从大型通用LLM迁移到专用的小型模型，并通过离线教师列表的蒸馏实现候选生成的泛化；在CALVIN-D操纵任务中证明了这一分离带来的成功率提升和延迟降低。

**🔧 技术方法**

采用 Qwen3 系列的轻量级模型做候选生成器，使用序列级监督学习（SFT）与知识蒸馏；使用结构化验证器确保候选符合 schema；冻结 AutoTrust JEV-27B 决策器做最终选择；对决策过程、延迟、GPU 时长和能耗进行度量。

**📊 数据集**

CALVIN-D 操作数据集（1,011 条验证注释 + 200 条测试 episode），DeepSeek V4 Flash 生成的三姿候选列表（36,453 对），以及 Qwen3 家族模型。

**📈 对比分析**

与官方 Jev-1.13.0 API 仅使用 34 个通用动作的基线（成功率 11%）对比使用候选生成器的方案（成功率 46.5%）；在全部 1,011 条验证注释上做六臂对照，比较了 fine-tuned Qwen3-4B/1.7B + JEV、通用 JEV 以及直接生成 Qwen3.5-27B。结果显示：SFT 方案在 52% 左右的成功率下，1.7B 流程每次决策仅 1.83 秒、与直接生成相比延迟下降 50%，GPU 时长和能耗也降低 14%；4B 方案成功率相似但资源占用更高。

**⚠️ 局限性**

评估仅覆盖 CALVIN-D 的带有特权状态和暂停物理的环境，未验证实时性能或跨数据集、跨 schema 的迁移能力；候选列表可能遗漏有效动作或继承教师错误；比较同时混合了不同模型家族、候选数目和决策器，未能单独量化各组件贡献；资源节省依赖于正式化器的大小。

---

## 361. How Much Does the Hint Matter? Scope Conditions for Hindsight Self-Distillation in Financial Reasoning

**arXiv ID:** 2610.36082 | [PDF](https://arxiv.org/pdf/2610.36082v1)

**作者:** Johnny Yu `[一作]` (University of California, Berkeley), Cyril Gorlla `[通讯]` (CTGT)

**关键词:** `Machine Learning` `Knowledge Distillation` `Explainability and Interpretability` `Computational Efficiency` `Federated Learning` `Data-Centric Learning` `Finance Related` `Transformer` `Large Language Model` `Prompt Engineering` `Supervised Fine-Tuning` `Contrastive Learning` `Text` `Tabular` `Finance Related`

**🎯 论文内容**

提出并实现了 Fin‑SD，一种针对多步金融推理的局部 hindsight self‑distillation 方法，利用学生模型自身识别错误点并生成简短批评，再通过逆 KL 对齐在 100 词 rollout 上实现自我改进。

**💡 创新点**

创新点在于不依赖外部教师或完整参考答案，仅使用学生自身生成的错误定位与批评作为提示，证明提示内容与更强模型写提示无显著差异，突出错误定位与局部监督的重要性。

**🔧 技术方法**

采用 on‑policy self‑distillation、逆 KL 损失、冻结学生权重作为教师、错误前轨迹截断、100-token rollout 对齐，并通过 LoRA 微调和 AdamW 优化实现训练。

**📊 数据集**

使用 FinanceReasoning Hard 子集（238 题）进行评测，并构建了 220 题自制金融推理训练集（来源于 3,133 个 Python 财务函数），同时对 MMLU‑Pro 和 FinTrust 进行跨域评估。

**📈 对比分析**

与基线 GPT‑OSS‑20B、OPSD（全解答）和 SFT（参考解答微调）对比，Fin‑SD 在 FinanceReasoning Hard 上准确率提升 3.03 百分点，token 消耗降低 3.3%，且在相同准确率下比 OPSD 节省 9.5% token；在 MMLU‑Pro 和 FinTrust 上表现无明显退化。

**⚠️ 局限性**

局限性包括自我蒸馏上限受学生预训练知识限制、训练集规模有限且与评测集可能存在重叠、未进行广泛的超参数调优，以及错误定位能力在更大模型或更复杂任务中可能下降。

---

## 362. AgentBaton: Efficient Model Collaboration in Agentic Tasks

**arXiv ID:** 2610.35959 | [PDF](https://arxiv.org/pdf/2610.35959v1)

**作者:** Junlong Li `[一作]` (Hong Kong University of Science and Technology), Pavlo Molchanov `[通讯]`

**关键词:** `Multiagent Systems` `Computational Efficiency` `Federated Learning` `Reinforcement Learning from Human Feedback` `Meta Learning` `Transformer` `Agentic AI` `Large Language Model` `Mixture of Experts` `Prompt Engineering` `Text` `Sequential` `Benchmark`

**🎯 论文内容**

研究在长周期代理任务中可持续模型之间一次性交接的准确度-成本权衡，并提出轻量级检测器自动判定交接时机。

**💡 创新点**

发现从强模型到弱模型的一次性交接可显著降低成本且保持准确度，并设计了基于执行历史的实时交接检测器。

**🔧 技术方法**

采用一次性模型交接、日志文件或上下文传输、Qwen 3.5 2B训练的检测器、基于大语言模型的代理与工具调用以及多模型协作技术。

**📊 数据集**

使用 DeepSWE v1.1 和 Terminal‑Bench 2.1 两个代理基准；并采样 2000 任务进行检测器训练。

**📈 对比分析**

与单模型基线、查询级路由和多轮路由对比；平均成本下降 65‑70%，准确率保持或提升至 73‑80%。

**⚠️ 局限性**

交接策略依赖模型间进度与兼容性，某些模型对交接点的识别不稳定；仅评估一次性交接，未探讨多次交接的效果。

---

## 363. Evaluating Federated Vision-Language and Vision-Only Models for Privacy-Aware Breast Cancer Imaging

**arXiv ID:** 2610.36379 | [PDF](https://arxiv.org/pdf/2610.36379v1)

**作者:** Malithi Mithsara Wanniarachchi Kankanamge `[一作]` (Southern Illinois University), Abdur R. Shahid `[通讯]` (Southern Illinois University)

**关键词:** `Computer Vision and Pattern Recognition` `Federated Learning` `Classification` `Safty and Privacy` `Transformer` `Convolutional Neural Network` `Vision Language Model` `Contrastive Learning` `FedAvg` `Client-level Differential Privacy` `Biomedical Data` `Image` `Breast Cancer Imaging`

**🎯 论文内容**

本文评估了在隐私保护的联邦学习框架下，利用预训练视觉-语言模型和仅视觉模型对乳腺热成像进行分类的可行性。

**💡 创新点**

创新点在于将视觉-语言模型与客户端差分隐私相结合，并系统地对比不同模型、去掉文本通道、通信成本及隐私-效能权衡。

**🔧 技术方法**

使用了联邦平均（FedAvg）、客户端级差分隐私（更新裁剪+高斯噪声）、CLIP、BiomedCLIP、Vision Transformer等模型。

**📊 数据集**

采用了两个公开乳腺热成像数据集：BCD（正常vs.异常）和乳腺热成像（良性vs.恶性）。

**📈 对比分析**

通过患者级ROC‑AUC（带Bootstrap 95%置信区间）比较模型，在非私有联邦学习下BCD任务达到0.94–0.99的AUC，乳腺热成像任务仅为0.50–0.64；去掉文本通道后性能相当且通信成本减半，差分隐私导致BCD AUC平均下降0.37。

**⚠️ 局限性**

局限包括样本量小、测试集受限、模型未进行中心化对照、差分隐私参数不佳导致噪声过大、缺乏多重随机种子、未评估真实攻击及临床验证。

---

## 364. PrinGS: Privacy Preservation in Gaussian Splatting

**arXiv ID:** 2610.36168 | [PDF](https://arxiv.org/pdf/2610.36168v1)

**作者:** Gaurav Behera `[一作]` (IIIT Hyderabad), Ponnurangam Kumaraguru `[通讯]` (IIIT Hyderabad)

**关键词:** `Computer Vision and Pattern Recognition` `Safty and Privacy` `Gaussian Splatting` `Image` `Point Cloud`

**🎯 论文内容**

提出一种在3D高斯喷洒（3DGS）模型中直接对已训练的高斯表示进行匿名化的方法，支持后训练匿名化与隐私设计训练；

**💡 创新点**

创新点在于：① 在表示层面精确分割并匿名化敏感高斯，② 通过局部模糊操作既去除人脸与文字的可识别特征又保持场景几何与视觉质量；

**🔧 技术方法**

采用基于可微光栅化的三维高斯分割、局部卷积模糊算子、k近邻加权均值滤波，以及对文本的几何与颜色联合模糊；

**📊 数据集**

使用合成室内场景数据集（9名多样化人类主体）和包含可读文字的公开场景（如Tanks & Temples、DL3DV-10K）进行评估；

**📈 对比分析**

与标准3DGS对比，FSM从83.87%降至9.79%（提高74%），FDR从94.74%降至39.47%，文字CER从0.208提升至1.00（完全无法识别），且后训练匿名化在1秒内完成；

**⚠️ 局限性**

局限性包括：① 仍可被面部检测器识别（FDR不为0）；② 需要用户提供参考视角或掩码；③ 主要针对人脸与文字，对其他隐私信息的泛化仍待验证。

---

## 365. God Does Not Play Dice: Enforcing Batch Invariance in Agentic Kernel Generation

**arXiv ID:** 2610.36623 | [PDF](https://arxiv.org/pdf/2610.36623v1)

**作者:** Youwei Bao `[一作]` (George Mason University), Simin Chen `[通讯]` (George Mason University)

**关键词:** `Artificial Intelligence` `Optimization` `Computational Efficiency` `Federated Learning` `Large Language Model` `Agentic AI` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

提出并实现了一个可插拔插件，结合DSL驱动的静态依赖分析与跨批次动态验证，用于检测和消除LLM推理系统中GPU kernel的批量不变性（BI）违规，集成到现有的LLM驱动的GPU kernel生成与优化器中。

**💡 创新点**

创新点在于：①把BI作为接受门直接嵌入生成循环，兼容现有优化器而不需重写；②引入契约（contract）驱动的静态分析，精准定位禁止的批次相关依赖；③设计跨批次动态验证，保证同一输入在不同批量上下文下的bitwise一致性；④实现二者互补的双层筛选机制，显著提升BI通过率。

**🔧 技术方法**

主要技术包括：DSL（定义批次语义与受保护参数）；静态依赖分析（检测批次符号到受保护参数的违规路径）；动态验证（在不同批次大小/组合下对同一样本结果进行比对）；插件化设计（作为acceptance gate插入优化循环）；以及现有LLM生成器/优化器（如Kevin、K-Search、FlashInfer-Bench）的集成接口。

**📊 数据集**

使用的评测数据集：KernelBench.com的GPU kernel任务集合与FlashInfer-Bench的LLM推理工作负载，涵盖多种kernel族与子类型，覆盖实际LLM推理场景。

**📈 对比分析**

评估方法：在三种代表性LLM驱动的kernel优化框架下，分别在基线（无BI检查）与BI-aware生成下进行对比实验。实验结果显示，BI通过率从67.6%提升至91.9%，而基线几何平均加速的保持率达91.8%，证明在保障BI的同时几乎不损失性能；静态分析与动态验证互补，进一步提高了检测覆盖率。

**⚠️ 局限性**

局限性：①静态分析目前跨文件分析能力有限，可能漏检跨文件依赖；②实验仅在kernel输出层面评估BI，未考虑端到端token生成或训练阶段的累计误差；③需要更多自动化规则学习与强化学习结合，以进一步提升覆盖范围和性能；④对非语义独立任务或特殊硬件的适用性仍需验证。

---

## 366. Three Worlds for the Price of a Logarithm: Baseline-Protected Best-of-Both-Worlds Combinatorial Semi-Bandits

**arXiv ID:** 2610.36245 | [PDF](https://arxiv.org/pdf/2610.36245v1)

**作者:** Ting Hu `[一作]` (University of Wisconsin--Madison), Emmanouil-Vasileios Vlatakis-Gkaragkounis `[通讯]` (University of Wisconsin--Madison)

**关键词:** `Machine Learning` `Reinforcement Learning` `Reinforcement Learning` `FTRL (Follow-the-Regularized-Leader)` `Importance Weighting` `Entropy Regularization` `Doubling Mechanism` `Stage and Phase Strategy` `Ville Inequality` `Markov Stopping Theory` `Hybrid FTRL`

**🎯 论文内容**

提出一种名为“基于经验测试的翻倍算法”（Doubling Under Empirical Tests）的组合半波次算法，能够在已知基线覆盖所有臂的前提下，在不知道时间 horizon、环境类型或误差间隔的情况下同时实现基线保护、对抗性鲁棒性和随机性适应性。

**💡 创新点**

创新点在于：①提出了两种可观测的经验测试（对学习者正向经验损失和对基线负向经验损失的比较）来动态调整学习权重和经验损失预算；②将熵FTRL与混合FTRL结合，分别用于混合阶段和纯阶段，实现既能在对抗环境下保持O(√T)的失调，又能在i.i.d.环境下获得多项式对数速率；③实现了不需要先验信息的 horizon‑free 方案，且对基线保护的期望损失仅为对数级。

**🔧 技术方法**

使用的技术包括：半波次反馈模型、重要性加权估计、熵 FTRL 与混合 FTRL（Hybrid‑FTRL）正则化、分阶段（stage）和分段（phase）策略、两阶段翻倍（doubling）机制、条件期望与马尔可夫停止理论、Ville 不等式、混合混合测试与预算更新。

**📊 数据集**

该工作为理论算法，未在真实数据集上进行实验；所有结果均为理论上界。

**📈 对比分析**

通过与已有基线保护算法、对抗性最优算法以及最佳二世界（best‑of‑both‑worlds）算法进行对比，证明在每个固定时间点 T 上：①对基线的期望损失为 O(log T)；②对任何固定动作的期望损失在对抗环境下为 O(√T·polylog T)；③在 i.i.d. 环境下唯一最优动作的情况下，期望损失为 O(polylog T /Δ)。相比现有方法，该算法不需要预知 horizon、gap 或环境类型，且统一实现三种性能保证。

**⚠️ 局限性**

主要限制包括：①基线必须覆盖所有臂（δ>0），覆盖率越低导致损失上界越大；②实现上需要枚举所有动作（可通过凸分解等技术简化但仍有计算开销）；③对纯阶段的混合 FTRL 需要处理固定坐标，导致实现复杂；④理论上界包含多项对数因子，实际常数系数可能较大。

---

## 367. ASPEN: Adaptive Semantic Reasoning via Command Persistence Prediction for Efficient Navigation VLAs

**arXiv ID:** 2610.36285 | [PDF](https://arxiv.org/pdf/2610.36285v1)

**作者:** Vieri N. Akustia `[一作]` (University of Houston), Bin Hu `[通讯]` (University of Houston)

**关键词:** `Robotics` `Computational Efficiency` `Robotic Intelligence` `Transformer` `Vision Language Model` `Vision-Language-Action Model` `Contrastive Learning` `Large Language Model` `Multimodality` `Text` `Image`

**🎯 论文内容**

提出ASPEN，一种基于命令持久性预测的事件触发调度器，能够在冻结的导航VLAs中判断何时可以重用上一次命令，从而显著降低语义推理调用次数；

**💡 创新点**

创新点在于：① 用预测模型评估下一命令是否保持不变，决定是否重用前一命令；② 结合多模态轻量级预测器与资格检查，实现自适应触发；③ 在跨数据集环境下验证该方法的鲁棒性；

**🔧 技术方法**

技术包括：Frozen SigLIP 与 Frozen Llama-3 的多模态编码，双层 Transformer 进行视觉+状态+指令的融合，命令持久性概率预测，阈值触发机制，及基于交叉熵的训练与阈值校准；

**📊 数据集**

使用了 R2R-CE 进行训练与阈值校准，RxR-CE val-unseen 进行无域迁移评估；

**📈 对比分析**

与冻结 NaVILA、FlashTrigger、SP‑VLA、Free Rule、Full Gate 等方法比较，ASPEN 在 RxR-CE val-unseen 上将 VLA 调用数减少 9.5%（估计推理时间 8.1%），同时保持与 NaVILA 相近的 SR、OS、nDTW、NE，并保持 92.1% 的 Exact agreement；

**⚠️ 局限性**

局限性：仅在冻结 VLA 上验证，缺乏对目标域迁移的理论保证；Exact agreement 下降导致部分命令序列不完全一致；未测量实际机器人上的延迟与能耗；方法依赖阈值校准，可能在不同任务中需要重新调整。

---

## 368. Scaling and Renormalization in Neural Networks: An Effective Renormalization-Group Description of Grokking

**arXiv ID:** 2610.36113 | [PDF](https://arxiv.org/pdf/2610.36113v1)

**作者:** Min Chen `[一作]` (University of Pittsburgh), Junyu Liu `[通讯]` (University of Pittsburgh)

**通讯引用:** 3937 | [OpenAlex ID](https://openalex.org/A5100682743)

**关键词:** `Machine Learning` `Physics Related` `Renormalization Group Theory` `Spectral Ranking` `Local Renormalization Mapping` `Tabular`

**🎯 论文内容**

论文探讨了深度神经网络中的grokking现象，即从记忆到泛化的延迟转变，并建立了一个定量的物理框架来描述这种动态。

**💡 创新点**

创新点在于将缩放和重整化群理论应用于神经网络，提供了一种有效的描述grokking的框架，并通过构建有效的几何结构和局部重整化映射来提取关键指数。

**🔧 技术方法**

使用了重整化群理论、谱排序和局部重整化映射等技术，结合了对神经网络训练过程的物理解释。

**📊 数据集**

使用了模块加法任务的数据集，模型为一个具有500个隐藏单元的两层感知器，训练在30个独立的运行上进行。

**📈 对比分析**

通过与传统的Gibbs集合的比较，论文展示了重整化群预测的关键指数与直接测量的结果在1-10%范围内一致，验证了所提出框架的有效性。

**⚠️ 局限性**

限制在于重整化群方法在神经网络中的应用仍然面临挑战，特别是在量化训练过程中的观察量与物理量之间的对应关系方面。

---

## 369. Data-driven Discovery of Lie Symmetries and First Integrals with Latent ODEs

**arXiv ID:** 2610.35967 | [PDF](https://arxiv.org/pdf/2610.35967v1)

**作者:** Matt L. Wiemman `[一作]`, Peter Melchior `[通讯]` (Princeton University)

**关键词:** `Machine Learning` `Physics Related` `Representation Learning` `Optimization` `Auto Encoder` `Recurrent Neural Network` `Auto Encoder` `Ordinary Differential Equation` `Stochastic Differential Equation` `Contrastive Learning` `Score-based Model` `Diffusion model` `Generative Adversarial Network` `Neural Radiance Field` `Vision-Language-Action Model` `Prompt Engineering` `Mixture of Experts` `Video` `Image` `Tabular` `Benchmark`

**🎯 论文内容**

提出LatentLieODE框架，利用自编码器学习低维表征，训练Latent ODE后通过线性生成器和Lie括号残差发现系统对称，并联合学习对应的守恒量。

**💡 创新点**

创新点包括完全可微的对称发现目标、利用矩阵范数谱检测对称代数维度、同时学习并校准第一积分的物理单位、无需预设对称维度、以及能在视频观测上直接发现非线性对称。

**🔧 技术方法**

核心技术为自编码器+卷积/递归解码器、Latent ODE、可微Lie括号残差、超集线性生成器、谱间隙判定、行动角正则化、第一积分MLP、Newton-Schulz反平方根与自动微分。

**📊 数据集**

使用六个经典基准系统（谐振子、单摆、双摆、Lotka–Volterra、像素摆、球面测地线）以及视频观测数据进行验证。

**📈 对比分析**

与LaLiGAN、Noether网络、HNN/LNN等方法对比，LatentLieODE能自动发现所有系统的真实对称并恢复守恒量，匹配度|ρ|≥0.978；在球面数据中成功恢复非阿贝尔so(3)结构，显示更高效且不需手工指定对称维度。

**⚠️ 局限性**

局限性在于仅搜索线性生成器，若系统对称在低维表征中不可线性化则无法发现；需手动设定潜在维度；对高噪声或高维视频的鲁棒性待进一步验证；对受限约束或非可微系统的适用性有限。

---

## 370. An Empirical Study of Undisclosed Charges in Hook-Based Decentralized Exchanges

**arXiv ID:** 2610.36682 | [PDF](https://arxiv.org/pdf/2610.36682v1)

**作者:** Kai Ma `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `Cryptography and Security` `Finance Related` `Supervised Fine-Tuning` `Chain-of-Thought` `Prompt Engineering` `Retrieval-Augmented Generation` `Tabular` `Time Series`

**🎯 论文内容**

本文对Uniswap v4和PancakeSwap Infinity等基于Hook的AMM生态进行大规模实证研究，评估Hook中未披露的结算扣费对交易者成本的影响。

**💡 创新点**

创新点在于首次构建完整的字节码分析与历史结算重构管道，揭示Hook中隐藏费用的普遍性、触发频率和经济负担。

**🔧 技术方法**

采用EVM字节码符号执行、Z3 SMT求解以及链上交易追踪技术来识别收费机制并重建交易结算。

**📊 数据集**

使用截至2026年8月1日的Uniswap v4与PancakeSwap Infinity的7,873个Hook、50,011个池、约2.96亿笔交易的数据集。

**📈 对比分析**

通过与标准池费对比和费用占比统计，发现Hook收费频率高但单笔费用低，且在BSC上频率更高；整体收入约398万美元。

**⚠️ 局限性**

局限包括仅覆盖四个平台、缺乏未定价交易、可能漏检复杂收费路径，以及未评估收费合法性或恶意动机。

---

## 371. TriDifNet: A Unified Three-Task Diffusion Network for End-to-End RGB-T Perception

**arXiv ID:** 2610.37097 | [PDF](https://arxiv.org/pdf/2610.37097v1)

**作者:** Qianchao He `[一作]` (Xi'an Jiaotong University), Xiaoqing Tian `[通讯]` (Xi'an University of Science and Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Image Translation` `Restoration` `Transformer` `Convolutional Neural Network` `Diffusion model` `Contrastive Learning` `Mixture of Experts` `Optical Flow` `Image` `Multimodality` `Benchmark`

**🎯 论文内容**

提出了三任务扩散网络TriDifNet，实现RGB‑T图像的配准、融合与行人检测的端到端统一处理；

**💡 创新点**

通过闭环互耦的多任务反馈（Det2Reg、Det2Fus、Fus2Reg）以及Nash‑MTL后期平衡，实现配准、融合与检测三者相互促进；

**🔧 技术方法**

采用条件扩散模型（DDPM/DDIM）、残差融合、交叉注意力、ROI引导重推理、冻结红外检测器的语义监督与Nash‑MTL多任务学习；

**📊 数据集**

在KAIST Multispectral Pedestrian Benchmark上进行实验，并通过仿射畸变生成弱配准场景；

**📈 对比分析**

与多种注册、融合与系统级基线（R1–R3、F1–F3、S1–S7）对比，TriDifNet在误配准下获得65.44%召回、57.55% mAP50、26.83% mAP50:95，显著优于所有对照；

**⚠️ 局限性**

推理速度受限于扩散迭代，且实验仅涵盖弱仿射畸变，缺乏对大幅非刚性配准及实时性能的验证。

---

## 372. Noise Steering: Exploiting Wake Steering and Acoustic Directivity for Noise-Constrained Wind Farm Control via Reinforcement Learning

**arXiv ID:** 2610.36888 | [PDF](https://arxiv.org/pdf/2610.36888v1)

**作者:** Martín de Frutos `[一作]` (Universidad Politécnica de Madrid), Esteban Ferrer `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Optimization` `Reinforcement Learning` `Tabular` `Time Series` `Physics Related`

**🎯 论文内容**

本文提出了Noise Steering方法，通过协同控制风机舵角和功率曲线，在满足局部噪声阈值的前提下最大化风电场功率；

**💡 创新点**

创新点在于首次将舵角偏转（wake steering）与功率削减结合，利用风机噪声的双极辐射方向性，并通过强化学习实现无模型闭环控制；

**🔧 技术方法**

采用PPO强化学习算法，在低阶稳态仿真器与自建的低保真声学模型上训练；随后将策略迁移到高阶、时间分辨的FAST.Farm仿真器进行验证；

**📊 数据集**

使用基于六台NREL 5MW风机的合成风电场数据，训练过程中采样风速区间8–15 m/s，共300k步；校准阶段采用16个不同舵角与风速组合的仿真样本；

**📈 对比分析**

与传统贪婪控制和行业标准的噪声削减（NRO）模式对比；结果显示Noise Steering在所有风速下满足40 dB阈值，平均功率损失仅2%（最高7.7%），而NRO平均损失17%；在湍流风场下保持91.7%贪婪功率，噪声限制违规率仅为7.4%；

**⚠️ 局限性**

局限性包括：训练环境与高保真模拟器间存在模型不匹配导致策略过于保守；仅考虑单一风向和固定保护区域；声学模型未考虑地面反射、折射及大气吸收等传播效应；未来需在更高保真或在线学习环境中进一步优化。

---

## 373. On Shaky Ground: Characterising Non-Rigid Base Effects in Mobile Manipulation

**arXiv ID:** 2610.36725 | [PDF](https://arxiv.org/pdf/2610.36725v1)

**作者:** Zongyuan Zhang `[一作]` (Queensland University of Technology), Jonathan M. Roberts `[通讯]` (Queensland University of Technology)

**关键词:** `Robotics` `Robotic Intelligence` `Autonomous Driving` `Simultaneous Localization and Mapping` `Optical Flow` `Reinforcement Learning` `Time Series` `Tabular` `Point Cloud`

**🎯 论文内容**

通过在两套轻量化移动机械臂平台上进行砂光实验，记录接触力、工具位置与基座振动，分析基座非刚性对接触任务的影响。

**💡 创新点**

首次系统量化低底部阻力下持续接触导致的力缺失与工具位移，并展示通过增加基座-地面阻力可显著改善力实现和抑制位移。

**🔧 技术方法**

采用力/姿态混合控制的力控砂光、六轴力/扭矩传感器、惯性测量单元、ArUco标记视觉跟踪和MuJoCo仿真，测量力、位移与振动。

**📊 数据集**

该工作未使用公开数据集，而是基于自制的两台轻量平台（UR10/UR5）与手动砂光器进行单一试验数据采集。

**📈 对比分析**

通过比较低与高底部阻力、不同臂姿势、不同力指令以及工具速度，发现高阻力下力实现率提升至约88–101%，工具位移显著减小，但振动幅度随工具速度增大。

**⚠️ 局限性**

试验仅一次且平台与支持条件有限，未测量表面质量，且未在真正的移动机器人上验证主动补偿策略，导致结果的普适性与精度受限。

---

## 374. Digital Twin-Guided Wireless Localization via Probabilistic Inverse Learning

**arXiv ID:** 2610.36905 | [PDF](https://arxiv.org/pdf/2610.36905v1)

**作者:** Mahmoud Saad Abouamer `[一作]` (Aalborg University), Petar Popovski `[通讯]` (Aalborg University)

**关键词:** `Information Theory` `Optimization` `Autonomous Driving` `Federated Learning` `Computational Efficiency` `Representation Learning` `Reinforcement Learning from Human Feedback` `Transformer` `Auto Encoder` `Simultaneous Localization and Mapping` `Contrastive Learning` `Gaussian Splatting` `Score-based Model` `Point Cloud` `Tabular` `Time Series` `Benchmark`

**🎯 论文内容**

提出一种基于数字孪生指导的概率逆向学习框架（PRISM），可使用单天线AP的有限信道观测实现精准无线定位。

**💡 创新点**

创新点在于将数字孪生作为前向一致性检查，将位置候选的生成与数字孪生的误差加权融合，既避免了对精确逆向模型的依赖，又保留了位置估计的概率性质。

**🔧 技术方法**

核心技术包括Transformer编码器对多AP CIR特征进行聚合、基于高斯分布的概率位置候选生成、以及数字孪生产生的信道一致性能量对候选进行加权的逆向学习与后验重构。

**📊 数据集**

使用Sionna Ray Tracing生成的Munich场景（6 GHz，100 MHz，10 AP~15 AP）作为数据集，训练并评估。

**📈 对比分析**

与CNN-Direct、Transformer-Direct、传统指纹定位（C-FP、Dense-FP、RFMap-FP）对比，PRISM在10个AP时中位误差降至约0.7 m（比Transformer-Direct低约63%），平均误差也显著优于所有基准，并在DT失真与观测SNR偏移时保持稳健。

**⚠️ 局限性**

局限在于仍假设环境为静态，数字孪生需要预先构建，动态变化的物体或材质更改对定位精度的影响尚未系统评估，且在高维3D场景或多天线配置下的扩展尚待验证。

---

## 375. Bayesian Replicator Dynamics and the Inversion of Strict Dominance in Bimatrix Games

**arXiv ID:** 2610.36345 | [PDF](https://arxiv.org/pdf/2610.36345v1)

**作者:** Jaime Ruiz-Serra `[一作]` (University of Sydney), Michael Harré `[通讯]` (University of Sydney)

**关键词:** `Computer Science and Game Theory` `Reinforcement Learning` `Optimization` `Game Theory` `Reinforcement Learning` `Stochastic Differential Equation` `Ordinary Differential Equation` `Contrastive Learning` `Review/Survey Paper`

**🎯 论文内容**

研究了在两人两行动博弈中，用期望效用和对数证据两种不同的适应度函数对复制子动力学的影响，揭示了对数证据模型能够反转优势策略并在 Prisoner’s Dilemma 中产生合作、在 Harmony 游戏中产生背叛的现象，并完成了两行动博弈的拓扑分类。

**💡 创新点**

创新点在于：①首次证明在所有两人博弈中，若存在优势策略，则对数证据模型将其消除；②将复制子动力学与连续时间贝叶斯推理直接对接，构造了对数证据的哈密顿量；③给出了两行动博弈的完整拓扑分类，将不等价游戏分为 𝒟 与 ℰ 两类，并与传统线性模型对应；④指出对数证据模型在无噪声、无限精度条件下的不可逆性与无内部吸引子的特性。

**🔧 技术方法**

技术手段主要包括复制子方程、对数证据（log‑linear）适应度的贝叶斯解释、对数几率坐标变换、哈密顿量与能量守恒分析、顶点保持同胚（topological equivalence）判定，以及对两行动博弈的边界激励分析。

**📊 数据集**

本研究为理论分析，不使用任何实验或现实数据集；所有结论均来自数学推导与符号计算。

**📈 对比分析**

通过数学证明比较两种适应度模型的动力学拓扑，分析内部平衡点、极限环和稳定性，发现对数证据模型在优势策略游戏中完全改变了吸引子结构；在多种经典博弈（Prisoner’s Dilemma、Stag Hunt、Matching Pennies 等）中展示了不同的轨道与稳定性结果。

**⚠️ 局限性**

局限性包括：①模型假设无限精度、无噪声、无折扣，现实学习过程中需要考虑有限精度与噪声；②无法直接对应具体多智能体强化学习算法；③不适用于多行动或多玩家情形，需进一步推广；④由于对数证据模型在边界处发散，导致无内部吸引子，难以捕捉实际系统的混合策略稳态。

---

## 376. Japanese Discourse Structure Analysis Based on Segmented Discourse Representation Theory Using LLMs

**arXiv ID:** 2610.36667 | [PDF](https://arxiv.org/pdf/2610.36667v1)

**作者:** Satomu Tomisaki `[一作]` (University of Tokyo), Hitomi Yanaka `[通讯]` (University of Tokyo)

**关键词:** `Computation and Language` `Representation Learning` `Explainability and Interpretability` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Text`

**🎯 论文内容**

使用 GPT‑5.1 将 MSDC 与 STAC 两个英语对话语料库逐句翻译为日语，并基于此构建日语 SDRT 语料；随后用 Llama‑3、Swallow‑3 等 LLM 进行 fine‑tune 以及 prompt‑based 语篇关系解析，比较两种方式在日语与英语上的性能，并通过三种消融实验（连词/终助词、上下文语篇关系、上下文内容）探究模型依赖的信息。

**💡 创新点**

首次将 SDRT 解析跨语言迁移到日语，证明日语与英语的 LLM 解析性能相当，并通过消融实验揭示模型主要依赖语篇结构而非词汇线索；同时指出「でも」等日语连词对 Contrast 关系预测具有显著影响。

**🔧 技术方法**

核心技术包括 GPT‑5.1 机器翻译、Llamipa fine‑tune 模型（Llama‑3、Swallow‑3）、prompt‑based 解析（GPT‑5.5、Llama‑3.1）、以及针对连词/助词、上下文关系与内容的消融实验。

**📊 数据集**

使用的语料为 MSDC（Minecraft 对话）与 STAC（Settlers of Catan 对话）两大英语 SDRT 语料库，及其经 GPT‑5.1 翻译得到的日语版本。

**📈 对比分析**

通过对每种关系的 F1 评分与加权平均 F1 进行对比，fine‑tune 方案在日语和英语上均达到约 0.79 的加权平均 F1，明显优于 prompt‑based（<0.72）；消融实验表明移除语篇关系会显著下降性能，而移除上下文内容影响有限。

**⚠️ 局限性**

局限性包括仅覆盖游戏聊天域，日语语料由机器翻译生成可能不自然；未对完整日语语料进行人工评测；缺乏跨域验证和真实日语原生语料。

---

## 377. Bringing Hardware Security Education to All: A Seamless Curriculum Integration Approach

**arXiv ID:** 2610.36219 | [PDF](https://arxiv.org/pdf/2610.36219v1)

**作者:** Tanvir Hossain `[一作]` (University of Kansas), Tamzidul Hoque `[通讯]` (University of Kansas)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Tabular` `Review/Survey Paper`

**🎯 论文内容**

在核心课程（嵌入式系统、数字逻辑设计、计算机体系结构）中嵌入9个硬件安全实验模块，以上下文学习的方式让学生在完成设计课题时自然而然接触安全概念。

**💡 创新点**

创新点在于：①将硬件安全视为设计课程的有机延伸，而非单独选修；②模块严格按照“设计概念-安全概念”耦合原则与课程同步；③仅增加约2小时讲座，重用已有实验设备与作业，成本极低。

**🔧 技术方法**

使用的技术包括：基于8051 MCU的逆向与缓冲区溢出实验、基于FPGA的逻辑锁定与硬件木马实验、CPU周期计数与缓存侧信道实验；以及在实验室环境中使用调试器、逻辑分析仪、VHDL/FPGA工具链等。

**📊 数据集**

数据集为三次课程投放的学生反馈，共计25名响应者，涵盖2023春季与2024秋季的内部与外部评估调查。

**📈 对比分析**

比较方法为自评前后知识量量表与“是否提供更多动手学习”的比例，结果显示知识得分提升0.7–1.2分（四分制），87%–100%学生认为模块提升了动手体验；未与传统选修课程做统计显著性比较。

**⚠️ 局限性**

局限性包括样本量小、仅自评、缺乏对照组、未评估其他平台（32位CPU、FPGA硬件木马等）、设备可复现性受限，覆盖范围未涵盖制造阶段或门控级逆向等高级威胁。

---

## 378. Compote: Collapse-Resistant Multiple Hypothesis Learning For Sequential Recommendation

**arXiv ID:** 2610.37005 | [PDF](https://arxiv.org/pdf/2610.37005v1)

**作者:** Louis Castres Saint Martin `[一作]` (Criteo AI Lab), Patrick Gallinari `[通讯]` (Criteo AI Lab)

**关键词:** `Machine Learning` `Recommendation System` `Transformer` `Contrastive Learning` `Mixture of Experts` `Sequential`

**🎯 论文内容**

提出一种多假设变换器 COMPOTE，通过温度退火的赢家通吃目标实现多假设学习，并用无归一化的最大相似度规则进行高效检索。

**💡 创新点**

创新点包括：① 通过退火多假设学习避免头部崩塌，天然实现条件分布的最大熵分区；② 证明无归一化最大池化可保持贝叶斯排名；③ 给出对假设数的对数阶推广误差界。

**🔧 技术方法**

使用因果自注意力的共享编码器 + 多个预测头，退火多假设学习 (annealed MCL) 目标，最大相似度聚合规则，softmax 近似的向量检索。

**📊 数据集**

在 Amazon 2023 评论数据集的三个子集：Musical Instruments、Industrial & Scientific、Video Games（约 25k 商品，5-10k 用户）。

**📈 对比分析**

与 7 个 ID‑based 序列推荐器、TIGER 生成推荐器以及 6 个多兴趣模型进行对比；在 Hit@5、NDCG@5/10 等指标上，COMPOTE 在 11/12 列中排名第一，且在 NDCG@5 上相对 TIGER 提升 9–10%，表明多假设学习显著提升了性能。

**⚠️ 局限性**

局限性包括：仅使用商品 ID，不结合内容特征；需要调节假设数 M 以避免过度划分；在更大规模或不同领域的数据上验证有限；对模型解释性（为什么假设会形成几乎静态的商品划分）仍待深入。

---

## 379. Beyond QASecClaw: Agent Scaling, Verifier Feedback, and Evidence Reliability in Software Assurance

**arXiv ID:** 2610.36947 | [PDF](https://arxiv.org/pdf/2610.36947v1)

**作者:** S M Abdulla Al Mamun `[一作]` (University of Manchester), Mohd. Ruhul Ameen `[通讯]` (Varendra University)

**关键词:** `Software Engineering` `Safty and Privacy` `Explainability and Interpretability` `AI Code Assistant` `Transformer` `Large Language Model` `Mixture of Experts` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Tabular` `Benchmark`

**🎯 论文内容**

在已有确定性工具证据的前提下，探究多专家LLM上下文能否提升软件保证；

**💡 创新点**

揭示单一上下文与多上下文在固定证据下并未显著降低错误清除率，验证器反馈提升正确判定但错误率亦上升，模型对误导性证据识别有限；

**🔧 技术方法**

使用 Qwen2.5-Coder-7B-Instruct（及 1.5B 版本）与 SpotBugs、JBMC、Defects4J、SV-COMP 等工具；

**📊 数据集**

采用 Juliet Java、OWASP Benchmark、Defects4J、SV-COMP Java archive、以及自定义 E5 模板等数据集；

**📈 对比分析**

对比单一上下文、5/10/50 专家上下文及加门控与否，实验显示多上下文未降低不安全清除率，验证器反馈虽提升正确判定却也增加错误判定；

**⚠️ 局限性**

受限于模型规模单一、实验仅针对 Java、证据检查依赖手工门控、缺乏预算匹配评估以及多专家上下文未考虑工具协同等因素。

---

## 380. Toward Circuit-Aware Mutation Testing for Quantum Programs: Exact Visibility and Execution Planning

**arXiv ID:** 2610.36976 | [PDF](https://arxiv.org/pdf/2610.36976v1)

**作者:** Andriy Miranskyy `[一作]` (Toronto Metropolitan University), Jose Campos `[通讯]`

**关键词:** `Software Engineering` `Physics Related` `Quantum Circuit Theory` `Operator Commutativity` `Rotation Angle Analysis` `State Difference Norm` `Statistical Estimation` `Clifford Stabilizer Analysis` `Quantum Circuit Data` `Review/Survey Paper`

**🎯 论文内容**

构建了一个针对量子程序的变异测试框架，能够区分存活变异的原因并指导执行与测试改进。

**💡 创新点**

创新点在于将量子电路的结构与状态相结合，给出可计算的可视化条件、旋转角度阈值、统计与结构过滤证书，并提供完整的决策流程和误差界定。

**🔧 技术方法**

采用量子电路理论、算子对易性、旋转角度分析、状态差迹范数、统计估计、克利福德稳定子分析等技术。

**📊 数据集**

主要使用小规模理论实例（如 GHZ、单量子旋转）以及在理想无噪声模型下的可执行示例；未使用大规模实验数据集。

**📈 对比分析**

通过理论推导给出变异得分区间、检测阈值判定与资源预算；实验验证以小型电路示例展示，性能以理论上可接受的计算复杂度说明。

**⚠️ 局限性**

局限性包括仅针对理想无噪声单门或子电路变异、未覆盖多门变异、没有考虑中间测量、重置、经典控制或硬件噪声；实现需手动获取期望值或使用复杂的样本估计。

---

## 381. When AI Can Build It, What Should We Assess? Rethinking Evidence of Engineering Competence

**arXiv ID:** 2610.36413 | [PDF](https://arxiv.org/pdf/2610.36413v1)

**作者:** Ali Shakiba `[一作]` (University of Sydney), Ali Shakiba `[通讯]` (University of Sydney)

**关键词:** `Software Engineering` `AI Code Assistant` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Tabular` `Sequential` `Review/Survey Paper`

**🎯 论文内容**

在一次13周的研究生软件工程课程中，作者重新设计并首次实施了AI允许的学习与评估流程，学生可以将大量工作委托给AI，但仍保留计划、验证、评审与集成等关键决策责任。

**💡 创新点**

创新之处在于提出了工程责任工作流、六通道评估证据架构以及收集‑交叉检查‑升级的推理逻辑，三者共同构成了在AI生成物出现时仍能有效评估工程能力的体系。

**🔧 技术方法**

技术手段包括 GitHub 版本控制与 PR 流程、Visual Studio Code 的 AI 日志插件、Gradescope 自动批改、CI/CD 自动测试、专门的 AI 记录工具以及人工智能生成代码与文档的多款生成模型。

**📊 数据集**

研究并未使用传统公开数据集，而是从 87 名学生产生的 957 个个人仓库、34 个团队仓库、AI 日志条目、自动化评分结果以及自我反思作业中收集证据。

**📈 对比分析**

由于本研究为设计与操作报告，未进行对照实验或性能评估；评估结果通过四条经验教训呈现，说明评估权重、交叉检查、证据强度与自动化可扩展性等方面的实际表现。

**⚠️ 局限性**

局限性包括仅在单一高校单一课程实施、缺乏对照组与学习成效量化、证据渠道间高度相关导致的解释困难、以及大量需要人工解读与交叉验证的工作负荷。

---

## 382. Learning Multi-Phase Robot Loco-Manipulation via Phase-Conditioned Policy Optimization

**arXiv ID:** 2610.36270 | [PDF](https://arxiv.org/pdf/2610.36270v1)

**作者:** Simone Manni `[一作]` (ETH Zurich), Jan Brüdigam `[通讯]` (Robotics and AI Institute)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Reinforcement Learning` `Phase-Conditioned Policy Optimization` `PPO Extension` `Low-Level Whole-Body Control Architecture`

**🎯 论文内容**

训练单一统一策略完成多阶段机器人行走与操作任务（如举起并搬运箱子、轮胎旋转等）

**💡 创新点**

利用阶段条件化PPO、相位门控奖励、相位预测辅助头以及多阶段初始化，显著引导探索并提升学习效率

**🔧 技术方法**

PPO扩展、阶段条件化价值网络、相位门控奖励、低层全身控制架构与动作增量输出

**📊 数据集**

无需人类演示，使用仿真随机化的物体尺寸、质量、摩擦等数据，在Unitree G1与Boston Dynamics Spot上验证

**📈 对比分析**

与基线MLP、无相位预测、单阶段初始化等对比，成功率从约44%–14%提升至约90%，收敛速度显著加快

**⚠️ 局限性**

需人工手动划分阶段，规模化面临表示瓶颈；仿真与真实差异导致的转移挑战仍未完全解决

---

## 383. DuplexTune: Coupled-State Replay for Memory-Efficient Mixture-of-Experts Adaptation

**arXiv ID:** 2610.36009 | [PDF](https://arxiv.org/pdf/2610.36009v1)

**作者:** Paul Raj `[一作]` (Texas A&M University-Commerce), Mutlu Mete `[通讯]` (Texas A&M University-Commerce)

**关键词:** `Machine Learning` `Computational Efficiency` `Knowledge Distillation` `Representation Learning` `Transformer` `Mixture of Experts` `Supervised Fine-Tuning` `Large Language Model` `Text`

**🎯 论文内容**

提出 DuplexTune 框架，通过宽度桥接和耦合状态重放实现对 Qwen1.5‑MoE‑A2.7B 的高效适配，显著降低显存占用并提升吞吐；

**💡 创新点**

创新点在于引入宽度桥接在残差流之间交换信息，并在冻结路由器的前提下通过桥接热身与主干适配两阶段训练实现激活重放，同时给出隐式状态恢复的收敛与重放条件；

**🔧 技术方法**

采用预训练注意力与专家模块包装、可学习的宽度桥接、桥接热身与主干适配训练、投影恢复激活及张量重放技术；

**📊 数据集**

使用 Qwen1.5‑MoE‑A2.7B 作为模型，基于通用指令数据集进行自回归监督学习，并评估 MMLU、GSM8K 与 MT‑Bench（多语言合计）指标；

**📈 对比分析**

与 LoRA、DoRA、(IA)^3、SFT+checkpointing、LOMO、GaLore 等方法对比，在单 GPU 上显存从 64.8 GB 降至 38.8 GB（↓40%），吞吐提升 26%（19.9→25.1 样本/s），并在 MMLU/GSM8K/MT‑Bench 上取得 67.2%/75.6%/7.69，稍低于基线多语言表现；

**⚠️ 局限性**

仍需存储专家权重与优化器状态，重放依赖收敛条件；实验仅在单一 GPU、单一指令集、单一模型上验证，缺乏跨模型/跨任务/跨语言的鲁棒性评估；多语言性能低于基线。

---

## 384. Curriculum-Driven Deep Reinforcement Learning for Autonomous Far-Range Rocket Landing

**arXiv ID:** 2610.35954 | [PDF](https://arxiv.org/pdf/2610.35954v1)

**作者:** Atin Chowdhury `[一作]` (Indian Institute of Information Technology), Piyush Joshi `[通讯]`

**关键词:** `Machine Learning` `Reinforcement Learning` `Autonomous Driving` `Optimization` `Reinforcement Learning` `Curriculum Learning` `Critic Warm-up` `Rehearsal Mechanism` `Tabular` `Time Series`

**🎯 论文内容**

本文提出一种基于三阶段课程学习的深度强化学习框架，用以实现从远距离（1,000–1,500 单位）到垂直降落的完整控制。

**💡 创新点**

创新点包括：① 采用完整 3×3 旋转矩阵观测向量以显式提供水平位移与速度信息；② 设计了三阶段课程（远程导航、软降落、完整任务）并引入 critic warm‑up 与 rehearsal 机制防止灾难性遗忘；③ 将奖励函数简化为 O(1) 级别，显著减少梯度不平衡。

**🔧 技术方法**

使用的技术主要是 PPO 强化学习算法、三阶段课程学习、critic warm‑up、回放混合、全矩阵观测、联合离散动作空间以及重新设计的奖励函数。

**📊 数据集**

实验数据来自自行构建的 6‑DOF 发射器模拟环境，初始水平距离从 1,000 至 1,500 单位均匀采样，训练总共 14.5M 步，评估 100,000 次。

**📈 对比分析**

在与基线（Chowdhury 等）进行对比时，提出方法在相同测试分布下成功率提升至 99.98%（基线 68.31%），且平均最终距离从 13.10 单位降至 7.07 单位，燃料余量保持 69.1%。

**⚠️ 局限性**

局限性包括：仅在确定性仿真环境中验证，未考虑真实世界扰动与传感器噪声；依赖 PPO 及手工调参的奖励函数；未对多种 RL 算法或随机种子进行系统鲁棒性评估。

---

## 385. Automatic Certified Differentiation in Lean

**arXiv ID:** 2610.36396 | [PDF](https://arxiv.org/pdf/2610.36396v1)

**作者:** Ramy Shahin `[一作]` (Qualgebra), Ramy Shahin `[通讯]` (Qualgebra)

**关键词:** `Symbolic Computation` `AI Code Assistant` `Computational Efficiency` `Explainability and Interpretability` `Optimization` `Large Language Model` `Prompt Engineering` `Proof Automation` `Inductive Search` `Theorem Proving` `Tabular` `Text` `Review/Survey Paper`

**🎯 论文内容**

本文提出一种基于 Lean 证明助手的自动化认证微分框架，能够在求导时同时生成正确性证明与有效域。

**💡 创新点**

创新点在于利用已有 Mathlib 导数定理自动构造证明，并通过元编程实现函数结构匹配和证书生成；同时引入索引加速定理搜索。

**🔧 技术方法**

技术包括 Lean 4 的元编程、定理库索引、归纳搜索与证明填充，以及利用 Mathlib 的单变量导数定理。

**📊 数据集**

使用了 500 个随机生成的实数函数（表达式树深度≤7），覆盖基本算术、三角、对数等构造。

**📈 对比分析**

通过与无索引搜索对比，评估在 8 核处理器上平均 0.18 秒/函数，索引后加速 8.6 倍，最长 1.9 秒，12 个无索引超时。

**⚠️ 局限性**

局限性包括仅支持单变量一阶导数、域约束可能非最优、对复杂符号匹配依赖深度展开、未处理多变量或分段函数。

---

## 386. When Diffusion Recommenders Stop Sampling: Preserving Generative Diversity

**arXiv ID:** 2610.36954 | [PDF](https://arxiv.org/pdf/2610.36954v1)

**作者:** Louis Castres Saint Martin `[一作]` (Criteo AI Lab), Patrick Gallinari `[通讯]` (Criteo AI Lab)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Auto Encoder` `Diffusion model` `Contrastive Learning` `Regression` `Bayesian Personalized Ranking` `Sequential`

**🎯 论文内容**

提出一种三阶段潜在扩散框架 DARTS，先冻结自编码器的潜在空间，然后在该空间训练无监督扩散模型，最后在解码阶段加入排序监督，实现多样化且准确的推荐。

**💡 创新点**

通过解耦表示学习、生成建模与离散排序，避免联合学习潜在空间与去噪器以及在去噪过程中过度使用离散损失，显著提升采样多样性并突破传统扩散推荐的“采样崩塌”问题。

**🔧 技术方法**

使用 Transformer 基础的序列自编码器、条件潜在扩散模型、回归去噪损失以及最终的贝叶斯个性化排序（BPR）损失；还采用了后验预测聚合来利用多样性提升推荐精度。

**📊 数据集**

在 Amazon 2023 评测数据集的三类子集（Instrument、Scientific、Game）上进行实验。

**📈 对比分析**

与 SASRec、DreamRec、DiffuRec、ADRec 等顺序、生成检索和扩散基准相比，DARTS 在 Recall@K、NDCG@K 等指标上提升 6.8%–12.0%，并通过多样本聚合进一步提升性能，且相比竞争者在推理速度上更快。

**⚠️ 局限性**

局限性包括仅在单步留一评估场景下验证，多样化样本的可解释性和多步未来预测尚未展开；且模型对多样化样本聚合的计算成本仍高于单样本推理。

---

## 387. API-TAIL: Detecting Model Substitution in LLM APIs through Temperature-Aligned Likelihood Tails

**arXiv ID:** 2610.36801 | [PDF](https://arxiv.org/pdf/2610.36801v1)

**作者:** Haocheng Mei `[一作]` (Tokenrhythm Technologies), Yunhe Wang `[通讯]` (Tokenrhythm Technologies)

**关键词:** `Machine Learning` `Anomaly Detection` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

提出了一种面向自然流量的语言模型 API 取代检测方法，利用批量文本的标准化异常度进行一次性评分并计数，结合复合零假设校准阈值；

**💡 创新点**

创新点在于：①将合法部署变异视为复合零假设并通过温度对齐消除批量级别偏移；②使用完整词表的精确热力学计算获得可重现的标准化典型性分数；③在多配置的本地模拟集上阈值校准，兼顾多种合法解码与系统提示；

**🔧 技术方法**

采用标准化异常度（excess surprisal）与其方差构造z分数；温度对齐通过最小化z分数均值寻找有效温度；计数超阈值z>3的响应并与从合法配置生成的95%阈值比较；实现基于GPU的单句推理，完全可复现；

**📊 数据集**

对 OpenRouter 上两大模型（Qwen3.8‑27B 与 Gemma‑4‑31B‑it）的26个纯正端点、10个特殊端点以及由7个端点提供的6种替代模型共计143,955条响应；使用1000个独立测试提示作为评估集；

**📈 对比分析**

在相同的响应集、相同的合法配置池和相同阈值规则下与 RUT、MET、NLL/日志秩/熵计数等基线比较；在四个面板中，10% 替换率下使用200请求，FAR≤0.07，检测率在不同替代模型间平均为0.69–1.00，远超基线（大多为0.00–0.13）；随着请求数增至400，检测率提升至≥0.96；

**⚠️ 局限性**

局限性：需要本地拥有被检测模型的权重以计算精确热力学；对极端部署配置（如自定义系统提示、非公开的温度设置）仍可能导致误报；在长文本与高温度环境下对齐阈值可能受限；

---

## 388. Hyperparameter and Neural Architecture Search via Evolutionary Optimization for Reinforcement Learning: HPNAS-RL

**arXiv ID:** 2610.36882 | [PDF](https://arxiv.org/pdf/2610.36882v1)

**作者:** Nihal Acharya Adde `[一作]` (Volkswagen AG), Hanno Gottschalk `[通讯]` (Technische Universität Berlin)

**关键词:** `Machine Learning` `Autonomous Driving` `Reinforcement Learning` `Neural Architecture Search` `Hyperparameter Search` `Optimization` `Convolutional Neural Network` `Reinforcement Learning` `Genetic Algorithm` `Multi-Objective Optimization` `Transfer Learning` `Parallel Training` `Image`

**🎯 论文内容**

提出了一个双分支进化框架，联合优化强化学习中的神经网络结构和超参数，以提升自动驾驶任务中的奖励和模型效率。

**💡 创新点**

创新点在于将NAS与HPO在同一进化过程中同步搜索，并引入教师-学生策略迁移和并行评估，实现对视觉特征提取网络与PPO超参数的协同优化。

**🔧 技术方法**

使用遗传算法（GA）进行多目标优化，采用多目标NSGA-II、早期退出初始化、优化迁移学习、SBX交叉、多项式变异以及并行训练。

**📊 数据集**

使用Unity3D高保真模拟环境中的三道环形赛道，基于PPO算法训练视觉输入为语义分割图像的自动驾驶代理。

**📈 对比分析**

与仅进行NAS的基线对比，实验显示HPNAS在累计奖励上提升约5%~20%，参数量降低30%~50%，FLOPs显著下降，整体性能优于NAS。

**⚠️ 局限性**

限制包括搜索空间受限于四个超参数，低保真训练导致评估偏差，早期实验收敛不稳，未集成贝叶斯代理模型提升样本效率，以及缺少与HPO-only基线的直接比较。

---

## 389. Counting by the Book: Verifying Alaska's Ranked-Choice Elections

**arXiv ID:** 2610.36494 | [PDF](https://arxiv.org/pdf/2610.36494v1)

**作者:** Lucy Horowitz `[一作]` (University of California, Berkeley), Lucy Horowitz `[通讯]` (University of California, Berkeley)

**关键词:** `Logic in Computer Science` `Explainability and Interpretability` `Computational Efficiency` `Federated Learning` `Safety and Privacy` `Formal Verification` `Theorem Proving` `Program Verification` `Tabular` `Review/Survey Paper` `Benchmark` `Finance Related`

**🎯 论文内容**

本文构建了基于阿拉斯加州选举法规的正式可验证模型，并实现了一个用 Lean 4 证明的 IRV 计票器，随后与官方公布的投票记录进行对比。

**💡 创新点**

创新之处在于同时提供了法律文本的形式化解释、可验证实现以及对实际投票记录的重现模型，揭示两者在重复排名和多重投票处理上的差异。

**🔧 技术方法**

作者使用 Lean 4 进行形式化验证，利用定理证明实现符合规范，并用可执行的 Lean 程序重新计算投票结果。

**📊 数据集**

数据集来自阿拉斯加州 2022 年、2024 年选举的投票记录（CVR），共 115 场竞选，包含 22 场多轮计票。

**📈 对比分析**

比较方法是将可验证模型与重现模型逐轮计票结果对比，发现两者在 75 个投票记录上产生不同计票，最终赢家一致，但差异可导致七票左右的得票差异。

**⚠️ 局限性**

局限在于假设 CVR 完整且正确、未能验证实际 Dominion 软件实现、对法律解释的主观性以及仅聚焦阿拉斯加州而非通用框架。

---

## 390. Developing a Speech Language Model for Basque

**arXiv ID:** 2610.36911 | [PDF](https://arxiv.org/pdf/2610.36911v1)

**作者:** Andoni Sudupe `[一作]` (University of the Basque Country), Eva Navas `[通讯]` (University of the Basque Country)

**关键词:** `Computation and Language` `Generation` `Data Synthesis` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Auto Encoder` `Flow-based Model` `Generative Adversarial Network` `Contrastive Learning` `Audio` `Text` `Multimodality`

**🎯 论文内容**

开发了首个开放源码的巴斯克语端到端语音语言模型（SLM），实现低延迟双向口语交互。

**💡 创新点**

采用模块化 LLaMA‑Omni 结构，将已有 LLM（Latxa）与 Whisper‑encoder、mHuBERT 离散单元编码器和 Unit HiFi‑GAN vocoder 组合；通过自动化流程将 VoiceAssistant‑400K 翻译成巴斯克语并合成音频，解决低资源语言的数据匮乏问题。

**🔧 技术方法**

使用 Latxa（LLaMA‑3.1‑8B‑Instruct 适配巴斯克语）、Whisper‑large‑v3‑eu 编码器、mHuBERT‑147 微调后与 1000‑KMeans 聚类得到离散单元、Unit HiFi‑GAN vocoder、非自回归流式语音解码器，以及两阶段训练（理解+文本生成、语音生成）和 LLM‑as‑judge 评估方法。

**📊 数据集**

合成数据约 450k 题/答（550h 问题、3300h 回答）；使用 VoiceAssistant‑400K 原始英文数据、Composite‑eu 600h 巴斯克语语音、TTS 数据 16h 两名说话人（Maider、Antton），以及用于评估的基准 ASR 数据集。

**📈 对比分析**

通过 LLM‑as‑judge（内容质量）、UTMOS（音频质量）和 WER/CER（可懂度）三项指标评估，并与 SpeechGPT、SALMONN+Orca、Qwen2‑Audio+Orca、LLaMA‑Omni、LLaMA‑Omni2‑7B 等英文 SLM 对比；模型得分 3.86/5（内容）、3.07 UTMOS、9.94% WER、370 ms 延迟，内容质量和延迟与英文基线竞争力强，语音质量仍略逊一筹。

**⚠️ 局限性**

评估仅基于合成数据，缺乏真实巴斯克语口语测试；LLM‑as‑judge 与模型同属家族可能存在自偏；音频质量与英文基线存在差距；未进行人类评估。

---

## 391. Sybil-Proofness of Generalized Adaptive Clinching and Implications for Blockchain Mechanism Design

**arXiv ID:** 2610.36629 | [PDF](https://arxiv.org/pdf/2610.36629v1)

**作者:** Chengkai Zhang `[一作]` (Rutgers University), Qishen Han `[通讯]` (Rutgers University)

**关键词:** `Computer Science and Game Theory` `Finance Related` `Ordinary Differential Equation`

**🎯 论文内容**

本文在去中心化金融中提出一种两边市场的 AMM 机制，采用扩展的适应性临界拍卖（adaptive clinching），实现了精确结算、个人理性、弱帕累托最优，并抵御 Sybil 攻击。

**💡 创新点**

创新点在于将原本仅适用于单边、固定供给的临界拍卖推广到两边、供给随状态变化的 AMM，并构建了通用目标驱动的临界框架，首次证明该机制在预算约束下实现 Sybil 鲁棒性与弱帕累托最优。

**🔧 技术方法**

技术手段包括目标基准的几何分析、连续动力学的微分方程求解、以及对成本曲线的凸性与可导性假设，从而构造出可行且可证明激励兼容的机制。

**📊 数据集**

由于研究属于理论机理设计，本文未使用实验数据集，而是通过严格的数学证明和不可行性结果来验证机制属性。

**📈 对比分析**

与现有 AMM 和 Sybil 防护机制对比，本文机制在保持收益不下降的同时实现了无免费午餐、精确结算和弱帕累托最优，并在两边交易中保持了良好的效率与激励兼容性。

**⚠️ 局限性**

限制方面，机制假设同一方向的预算不可超报，并只能达到弱帕累托最优；若放宽这两点，则不可行，且实现完全帕累托最优与 Sybil 鲁棒性同时满足被证明为不可能。

---

## 392. Mimetic Embodiment Transfer of Human-Humanoid Interaction from In-the-Wild Videos

**arXiv ID:** 2610.36615 | [PDF](https://arxiv.org/pdf/2610.36615v1)

**作者:** Nghia Le `[一作]`, Anh Nguyen `[通讯]`

**关键词:** `Robotics` `Pose Estimation` `Data Synthesis` `Robotic Intelligence` `Optimization` `Transformer` `Convolutional Neural Network` `Recurrent Neural Network` `Graph Neural Network` `Diffusion model` `Reinforcement Learning` `Contrastive Learning` `Auto Encoder` `Generative Adversarial Network` `Optical Flow` `Score-based Model` `Video` `Image` `Mesh` `Point Cloud` `Sequential`

**🎯 论文内容**

提出一种从真实人-人交互视频中提取物理可行的人-人交互姿态，并将其中一名参与者转化为不同人形机器人的交互动作，生成新的训练数据集。

**💡 创新点**

结合联合相对方向与解剖学偏移矩阵、语义和距离加权、后期平衡优化，实现在不同人形机器人上保持交互语义的可模仿转移，并构建大规模多样化的从视频转人形机器人交互数据集。

**🔧 技术方法**

使用SMPL全身重建、脚-地面一致性微调、逆运动学初始化、交互建模矩阵、加权优化、后期平衡与碰撞避免，以及MotionCLIP计算语义重要性和强化学习训练机器人执行策略。

**📊 数据集**

从YouTube、TikTok等公开视频采集1200段两人交互视频，重建约500k帧；生成的交互数据用于Unitree R1 Edu和HighTorque Mini-hi；与Human3.6M、AMASS、3DPW等传统数据集进行对比。

**📈 对比分析**

与传统IK、GMR、PAIR等方法比较，MPJPE最低、IRD和CF1最高；在生成任务上，Duolando在FID_k、FID_cd、接触率上表现最好；在实机部署中，机器人能够保持平衡并安全执行交互动作。

**⚠️ 局限性**

受限于人体与机器人形态差异，部分动作在机器人上不可行；当前部署仅依赖预定义序列，未实现实时人-机同步；需要进一步适应动态人类行为。

---

## 393. Optimality and Impossibility in Multidimensional Strategyproof Facility Location

**arXiv ID:** 2610.36613 | [PDF](https://arxiv.org/pdf/2610.36613v1)

**作者:** Jianan Lin `[一作]` (Rensselaer Polytechnic Institute), Jianan Lin `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `Computer Science and Game Theory` `Optimization` `Federated Learning` `Safty and Privacy` `Explainability and Interpretability` `Review/Survey Paper`

**🎯 论文内容**

研究了在无支付的多维度范数空间中，针对社群成本目标的确定性策略可测（SP）设施定位机制，证明单一设施的坐标中位数（CM）在欧氏平面上无论是否匿名都能达到最优逼近比，并给出在ℓ∞^d空间中精确的SP机制；同时证明在1<p<∞的ℓ_p^2空间中，任意多设施（2≤k<n）无确定性SP机制能获得有限逼近比，并将两设施的不可行性扩展到所有k>2。

**💡 创新点**

① 去掉了此前对单一设施CM最优性证明中的匿名性假设；② 在非欧氏空间中提供了全局最优的SP机制；③ 通过离散化“远距代理”归约，将两设施的不可行性推广到任意k；④ 对高维空间提出了仅需对一对代理保持对称性的更弱不可行性条件。

**🔧 技术方法**

利用策略可测的凸投影性质、连续性与正齐性转换、对角化及共线性分析、局部方向集合（cone）与极点的几何推理，以及分段持续性（segment persistence）与极限子网技术；结合离散化的菜单集合（menu）和最近点性质进行证明。

**📊 数据集**

无实验数据集；研究完全是理论分析与严格证明，关注的是逼近比和不可行性上限。

**📈 对比分析**

通过证明与已知最优比值（如√(2(n^2+1))/(n+1) 或 √2）相匹配，确认CM在欧氏平面上的最优性；在ℓ∞^d中构造的lexicographic最小化机制实现逼近比1；而在ℓ_p^2空间中证明任何多设施SP机制的逼近比必为无穷大，展示了不可行性。

**⚠️ 局限性**

对p=1和p=∞的多设施情况仍未解决；在高维空间中可否去掉对对称性的要求仍是开放问题；此外在一维线上，k>2且无匿名性的可行性也尚未确定。

---

## 394. Enhancing Deep Learning Model Performance through Data Generation Strategies: Classification of Olive Tree Diseases

**arXiv ID:** 2610.35984 | [PDF](https://arxiv.org/pdf/2610.35984v1)

**作者:** Hicham El Akhal `[一作]` (Moulay Ismail University of Meknes), Abdelbaki El Belrhiti El Alaoui `[通讯]` (Moulay Ismail University of Meknes)

**关键词:** `Artificial Intelligence` `Classification` `Data Synthesis` `Convolutional Neural Network` `Diffusion model` `Generative Adversarial Network` `Contrastive Learning` `CutMix` `MixUp` `Image` `Agriculture Related`

**🎯 论文内容**

通过结合传统数据增强与CutMix、MixUp及DCGAN等生成技术，对七类橄榄树叶病害图像进行数据扩增，并训练EfficientNetV2B0模型

**💡 创新点**

提出了分三种数据分布场景（S1、S2、S3）的混合数据生成框架，并在此基础上系统评估不同生成方法的效果

**🔧 技术方法**

采用传统增强、CutMix、MixUp、DCGAN、FID评估、EfficientNetV2B0及5折交叉验证等技术

**📊 数据集**

使用从公开来源（如GitHub、Kaggle）收集的6170张橄榄叶病害图像（7类）

**📈 对比分析**

与无增强基线相比，最佳方案在S2场景下准确率提升至95.99%，整体提升约2.5%，在S1与S3场景亦实现显著提升

**⚠️ 局限性**

受限于DCGAN训练耗时、数据质量与分布对模型敏感、模型对过度增强易饱和以及对低数据量场景的适用性未充分验证

---

## 395. A Heart Means Recursion: Exploring Visual Meaning and Communication in VLMs

**arXiv ID:** 2610.36449 | [PDF](https://arxiv.org/pdf/2610.36449v1)

**作者:** Alexa R. Tartaglini `[一作]` (Google DeepMind), Yuxuan Li `[通讯]` (Google DeepMind)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Transformer` `Prompt Engineering` `Chain-of-Thought` `Vision Language Model` `Image` `Text` `Multimodality`

**🎯 论文内容**

系统研究视觉语言模型（VLM）在无关任务中受简单视觉符号（形状、颜色、图标）影响的“视觉驱动”现象，并通过多轮通信游戏评估模型生成与解释可视化来传递抽象概念和故事的能力，进一步与人类认知对比。

**💡 创新点**

创新点在于：①首次揭示视觉符号可显著操纵VLM的推理策略（如递归/迭代决策）并证明模型对其影响缺乏透明性；②发现模型在生成和解释可视化时存在生成‑解释差距；③通过人类实验验证模型视觉语义与人类直觉的偏离，提示潜在的“隐形沟通”风险。

**🔧 技术方法**

技术手段包括：提示工程（explicit & silent image prompts）、多轮交互式通信游戏、基于 PIL 的自由绘图、思维链（CoT）与自我报告分析，使用 Claude、Gemini 等主流大语言模型。

**📊 数据集**

数据集主要为：1）288种彩色形状符号（24种形状 × 12种颜色）；2）基准代码推理任务（如二叉树遍历、斐波那契、二分查找等）；3）对照图像（高斯噪声、ImageNet 真实图片）；4）少量人类标注的视觉解释和故事分类任务。

**📈 对比分析**

比较方法：对比无图像基线、文本描述、噪声图像的递归比例；计算视觉驱动效应 ΔP(recursion)；评估多轮通信成功率（symbol vs. free‑form 绘图）、人类识别准确率。结果显示：视觉符号可将递归率提升 0.5–1.0，free‑form 绘图的沟通成功率可达 97%（zero‑shot 91%），而符号沟通的 zero‑shot 成功率仅 48%；人类对顶尖符号的识别准确率仅 36%。

**⚠️ 局限性**

局限性包括：①实验任务高度人工合成，缺乏真实世界应用场景；②人类评估样本量小且仅限7人，结论需进一步验证；③对视觉驱动机制的内部解释缺失，未深入探究模型内部表示；④使用的 VLM 主要为封闭源模型，缺乏可重复性。

---

## 396. Words or Deeds? How LLMs Predict Users' Decisions in Human-Agent Dialogue

**arXiv ID:** 2610.36813 | [PDF](https://arxiv.org/pdf/2610.36813v1)

**作者:** Baiqiao Zhang `[一作]` (Hong Kong University of Science and Technology), Yulong Bian `[通讯]` (Shandong University)

**关键词:** `Computation and Language` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Sequential`

**🎯 论文内容**

通过收集42名用户与五种性格化GPT-4o代理的重复囚徒困境游戏（共1260轮）记录，对LLM在利用对话与历史决策预测用户行为的效果进行实验评估。

**💡 创新点**

首次引入“语句保持率”概念，拆分预测误差为已履行与违背语句两类，系统比较LLM在不同代理人格与用户-代理相容性下的预测偏差，揭示对话在提升已履行语句预测时过度加权、忽视代理与相容性差异的局限。

**🔧 技术方法**

采用八个开源指令调优LLM（Qwen、Gemma、Seed-OSS、GLM、Ministral、Olmo）进行零样本预测；构造决策记录与对话输入组合；使用AUC、准确率、预测偏差等评估指标；对话插入实验验证模型对语句的敏感度。

**📊 数据集**

使用Personality-1260数据集，42名中国大学生完成BFI-44测评后，分别与五个GPT-4o代理进行6轮对话+决策，共计1260轮。

**📈 对比分析**

通过信息消融、插入语句和代理人格/相容性回归等对比方法，发现决策记录即可将AUC提升至约0.80，加入对话几乎无提升；预测整体准确率约0.74，但对已履行语句提升（从0.76→0.95）与违背语句下降（从0.65→0.17）对比明显；预测对代理人格与相容性变化几乎无响应。

**⚠️ 局限性**

样本量小、受试者为低风险大学生、所有代理均为GPT-4o、未对因果关系进行独立验证、对不同语言和任务的泛化性未做评估。

---

## 397. S3-S2ST: Semantically Supervised Streaming Speech-to-Speech Translation

**arXiv ID:** 2610.36777 | [PDF](https://arxiv.org/pdf/2610.36777v1)

**作者:** Yuxiang Zhao `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `Sound` `Translation` `Computational Efficiency` `Representation Learning` `Transformer` `Supervised Fine-Tuning` `Contrastive Learning` `End-to-End Joint Training` `Audio` `Text`

**🎯 论文内容**

提出并训练 S3‑S2ST 模型，学习从语义对齐的流式监督中生成语音到语音翻译，并在 640 ms 输入网格上实现可调的提交策略，兼顾翻译质量与延迟。

**💡 创新点**

创新点包括：① 将完整句子对转化为增量训练样本以实现语义对齐的流式监督；② 引入可调阈值门控学习何时提交翻译；③ 通过播放适应阈值利用缓冲音频动态提升质量，允许在单一检查点上实现多质量–延迟操作点。

**🔧 技术方法**

使用 Thinker‑Talker 架构、640 ms 块编码器与块因果适配器、MLP 写入门控、八层隐藏状态融合、预训练权重冻结、语义单元对齐以及端到端联合训练。

**📊 数据集**

数据集包括公开的 FLEURS 英中测试集、内部 9‑pair 测试集、RealSI 基准，以及约 13 k 小时真实源音频（≈ 6.5 k 小时/向）和 300 k 小时的离线训练音频，配合 Qwen3 与 CosyVoice 进行文本与语音合成。

**📈 对比分析**

与 SeamlessM4T‑v2‑large、SeamlessStreaming、Qwen3‑Omni、UniSS、SimulS2ST‑Omni、X‑Translator、Hy‑SimulMT 等基线在 FLEURS、9‑pair 及 RealSI 上通过 BLEU、COMET、ASR‑COMET、SIM、UTMOS、StartOffset 等指标对比。S3‑S2ST 在 9‑pair 取得最高翻译质量，在 FLEURS 上表现竞争，且在 RealSI 上通过可调阈值实现优于 SimulS2ST‑Omni 与 Hy‑SimulMT 的质量–延迟折中。

**⚠️ 局限性**

局限性：评估仅覆盖英中两语种；多语言迁移和长对话场景尚未验证；内部 9‑pair 数据不公开；可能存在预训练重叠或近似重复；自动评估工具可能存在语言偏差。

---

## 398. RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding

**arXiv ID:** 2610.36533 | [PDF](https://arxiv.org/pdf/2610.36533v1)

**作者:** Buzhao Liu `[一作]` (Washington University in St. Louis), Chongjie Zhang `[通讯]` (Washington University in St. Louis)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Vision Language Model` `Retrieval-Augmented Generation` `Chain-of-Thought` `World Model` `Video` `Image` `Text` `Benchmark`

**🎯 论文内容**

提出并实现了RoboAuto框架，用于基于大语言模型的机器人程序搜索，通过层级知识库、场景地面化、阶段性验证与自设计实验实现物理理解。

**💡 创新点**

创新点在于：1) 采用层级知识库，将可转移的物理知识（可作用性、空间关系、动作效果）与可执行技能分离；2) 通过视觉工具进行任务相关物体识别与关键量测量实现场景地面化；3) 采用多阶段规划与验证策略，避免后续失败；4) 通过自设计实验主动验证物理假设并更新知识。

**🔧 技术方法**

核心技术包括：大语言模型（Claude Opus 4.6）作为编码代理；Molmo2、SAM 3等视觉工具用于物体识别与量测；基于层级知识库的检索与更新；多阶段规划与独立验证器；自设计实验脚本生成与执行。

**📊 数据集**

数据集：使用LIBERO-90演示视频构建初始知识库；在LIBERO、LIBERO-PRO、LIBERO-PRO Long、LIBERO-Plus、Robosuite、RoboTwin 2.0等五大基准集上评估；以及真实 WidowX AI 机器人上的长周期精准操作任务。

**📈 对比分析**

与CaP-Agent0、ASPIRE、RHO等自主Code-as-Policy方法以及π_0.5 VLA基准进行对比。RoboAuto在五大基准中平均成功率最高，短期任务 89.1%，LIBERO-PRO Long 73.4%（>2倍于ASPIRE 30.8%），LIBERO-Plus 90.3%（>2倍于ASPIRE 44.7%），Robosuite 90.6%（>10%提升），RoboTwin 2.0 91.5%（vs 24.5%无知识）。

**⚠️ 局限性**

局限性在于：实验主要在仿真环境中进行，真实机器人实验仍需人工重置场景且缺乏完整安全监控；知识库初始化仅基于LIBERO演示，对其他领域的通用性仍需验证。

---

## 399. Rethinking Channel Estimation for Fluid Antenna Systems: A Physics-Informed Operator Learning

**arXiv ID:** 2610.36787 | [PDF](https://arxiv.org/pdf/2610.36787v1)

**作者:** Zixin Wang `[一作]` (Hong Kong University of Science and Technology), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `Information Theory` `Optimization` `Computational Efficiency` `Physics Related` `Transformer` `Convolutional Neural Network` `Diffusion model` `Contrastive Learning` `Auto Encoder` `Generative Adversarial Network` `Prompt Engineering` `Tabular` `Time Series` `Physics Related`

**🎯 论文内容**

本文提出一种面向流体天线系统（FAS）的连续频域通道估计框架，将其建模为受Helmholtz方程约束的随机场逆问题，并通过物理信息化的U型傅里叶神经算子（UFNO）实现端到端的高精度通道重构；

**💡 创新点**

创新点包括：①利用电磁信息理论将有限孔径观测映射为频域卷积逆问题；②设计UFNO结合可学习软阈值（SST）与 Sobolev 损失，既抑制噪声又保持高频细节；③通过 Landau–Pollak 经典 EDoF 边界指导空间模式预算，实现跨分辨率的解析性算子；

**🔧 技术方法**

使用技术包括：U形傅里叶神经算子（UFNO）、双分支全局傅里叶 + 本地 MLP、可学习软阈值、Sobolev 损失、频域模式截断、FFT/IFFT、复杂域表征、实例归一化、实例化的梯度一致性约束；

**📊 数据集**

数据集为基于几何多径模型（3.5 GHz、4λ aperture、48×48网格、48子载波）的合成数据，训练集51200个样本，验证集5120个样本，SNR在–20到30 dB随机采样；

**📈 对比分析**

与传统 LS、OMP、SBL、DC‑GOMP、CNN、ViT 等方法比较，UFNO 在 NMSE 上平均低 3–5 dB 以上，远优于 CS 方法；在不同采样率、延迟扩展、跨分辨率测试中保持稳定；推理延迟约 9.5 ms，显著快于 OMP/SBL（20–180 ms/秒级）且与标准 FNO/ViT/ CNN 速度相当；

**⚠️ 局限性**

局限性包括：仅在仿真数据上验证；对极端延迟扩展、不同天线配置（平面/体积）与时间变通道的泛化尚未充分测试；UFNO 目前实现比标准 FNO 更慢，需进一步优化；

---

## 400. SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

**arXiv ID:** 2610.36991 | [PDF](https://arxiv.org/pdf/2610.36991v1)

**作者:** Zafer Sahinoglu `[一作]` (Mitsubishi Electric), Cahit Berkay Kazangirler `[通讯]` (Karabuk University)

**关键词:** `Artificial Intelligence` `Retrieval` `Generation` `Computational Efficiency` `Representation Learning` `Graph Neural Network` `Transformer` `Retrieval-Augmented Generation` `Contrastive Learning` `Prompt Engineering` `Large Language Model` `Text` `Graph` `Benchmark` `Agriculture Related` `Finance Related` `Physics Related`

**🎯 论文内容**

提出 SpectRAG，一种将密集检索与加权块图谱学相结合的检索增强生成框架，能够在固定 token 预算下构造多样、连贯、覆盖充分的上下文。

**💡 创新点**

创新点包括：①使用多种边信号（语义相似、文档相邻、章节层级、实体重叠、引用链接、Utility 问题相似）构建加权块图；②通过图拉普拉斯谱坐标实现相关性传播与跨段桥接证据挖掘；③在候选扩展后进行基于图接近度与谱信息的重评分，并在 token 预算下实现结构化、去冗余的上下文选择。

**🔧 技术方法**

技术实现涵盖：dense embedding（FAISS）+ BM25 + utility-question matching；加权块图构造与稀疏化；拉普拉斯谱分解（取前32个非平凡特征向量）；个性化 PageRank 与谱距离；图邻域扩展；token‑budgeted 选择策略；hybrid seed retrieval（RRF融合）。

**📊 数据集**

数据集：UltraDomain（农业、计算机科学、法律、混合四子集），MultiHopRAG，2WikiMultiHopQA，MuSiQue 以及长文档问答集合。

**📈 对比分析**

与 NaiveRAG、HybridRAG、LightRAG、HyDE 等基线对比；检索层面 Recall@k、MRR 明显提升，尤其在多跳推理任务上显著优于纯向量检索；生成层面 EM/F1 稍有提升，最优表现与 LightRAG 相当但存储成本更低，整体性能在多跳、跨文档场景下更具优势。

**⚠️ 局限性**

局限性：检索改进并未在所有任务中转化为显著生成提升（检索‑生成差距）；对边权重、分块粒度高度敏感；在比较、时间类问题上表现不如基线；计算与存储开销仍高，尤其是谱分解与近似检索的实现；需要更细致的多模型评估与人类偏好研究。

---

## 401. Readout and Operating-Point Sensitivity in Event-Stream Candidate Selection

**arXiv ID:** 2610.36743 | [PDF](https://arxiv.org/pdf/2610.36743v1)

**作者:** Yifan Shuai `[一作]`, Yifan Shuai `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Anomaly Detection` `Computational Efficiency` `Representation Learning` `Convolutional Neural Network` `Transformer` `Contrastive Learning` `Ridge Regression` `Bootstrap` `ROC` `Supervised Fine-Tuning` `Video` `Time Series`

**🎯 论文内容**

评估事件摄像机在不同表示与读取方式下的候选选择性能，使用离线 DailyDVS-200 数据和多模型、可微调、轻量化基线，并对阈值、窗口、视图、历史长度等进行系统实验。

**💡 创新点**

系统化比较冻结自监督表示与训练后 CNN、CLS 与 patch 读出、阈值重新校准、分区敏感性以及多重比较校正的影响，揭示模型大小与性能的相对依赖关系。

**🔧 技术方法**

采用事件累积成图像、ViT 与 V‑JEPA 自监督、MobileNetV3/ResNet 预训练、TinyCNN 训练、Ridge 回归阈值校准、ROC 的 pAUC20 与 TPR10、Bootstrap 阈值敏感性等技术。

**📊 数据集**

使用 DailyDVS‑200 520 条录制片段，包含 47 名参与者，涵盖握手、摔倒、门开启等动作。

**📈 对比分析**

通过参与者分层折叠、10 次重新分区、验证阈值选择，测量捕获率、误触率、pAUC20 与 TPR10，结果显示 DINO CLS 与 MobileNet 等轻量模型在低 FPR 区间表现具有竞争力；模型规模与读出方式相互影响。

**⚠️ 局限性**

仅为离线候选评估，缺少连续背景、同步 IMU/音频、实时触发频率、能耗等实际使用信息；数据与模型受开发集适配限制，未能证明个人记录价值。

---

## 402. CVBL-CCTV-Face Dataset for Real-World Face Enhancement and Recognition in Surveillance

**arXiv ID:** 2610.36781 | [PDF](https://arxiv.org/pdf/2610.36781v1)

**作者:** Manish Das `[一作]` (Indian Institute of Information Technology), Satish Kumar Singh `[通讯]` (Indian Institute of Information Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Restoration` `Super Resolution` `Retrieval` `Transformer` `Convolutional Neural Network` `Diffusion model` `Supervised Fine-Tuning` `Contrastive Learning` `Image`

**🎯 论文内容**

在真实闭路电视监控环境下，构建了 CVBL‑CCTV‑Face 数据集，包含无配对的低质量 (LQ) CCTV 人脸图与高质量 (HQ) 参考图，随后设计并实现了一个模块化的面部增强与识别管线，结合 MTCNN 检测、transformer/diffusion 超分模型（CodeFormer、RestoreFormer++、PiSA‑SR、TSD‑SR）、Inception‑ResNet‑v1 嵌入提取与 L2 匹配，并在实时系统中验证其可行性。

**💡 创新点**

主要创新在于提供真实无配对的 CCTV 人脸数据集，针对该数据集提出并比较了三种训练策略（预训练、合成降质微调、真实低质微调），证明合成降质微调最优；同时构建了完整的实时监控识别系统，并系统评估了多种超分模型对识别性能的影响。

**🔧 技术方法**

使用的技术包括 MTCNN 人脸检测；transformer/diffusion 基础的超分模型（CodeFormer、RestoreFormer++、PiSA‑SR、TSD‑SR）；Inception‑ResNet‑v1 512‑维人脸嵌入；欧氏距离（L2）进行身份匹配；以及基于 RTX 3090 的 25 FPS 实时推理。

**📊 数据集**

采用自制的 CVBL‑CCTV‑Face 数据集，包含 223 名身份的多张低质量 CCTV 人脸图（LQ）和对应的高质量参考图（HQ），分为版本A、B，分别提供不同数量的 HQ 参考。

**📈 对比分析**

通过在三种训练设置下评估 Top‑1 识别准确率和 L2 距离。基线未增强为 62.6%；预训练模型约 64.4–64.8%；在合成降质微调下 TSD‑SR 达到最高 70.08%；PiSA‑SR 与 RestoreFormer++ 分别为 67.81% 与 67.49%。实时推理在 RTX 3090 上约 25 FPS。

**⚠️ 局限性**

限制主要包括：在真实低质量图像上微调效果不如合成降质；模型对未对齐、包含多种复杂失真（遮挡、姿态、光照）时仍易失误；缺乏验证指标（FAR/FRR）、推理延迟与不同硬件平台的成本评估；数据集规模和多样性仍有限，需要进一步扩充和对齐。

---

## 403. HqcLethe: Streaming HQC Encapsulation for Tiny Stateless Servers

**arXiv ID:** 2610.36808 | [PDF](https://arxiv.org/pdf/2610.36808v1)

**作者:** Wind Wong `[一作]` (Vrije Universiteit Amsterdam), Klaus v. Gleissenthall `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Computational Efficiency` `Large Language Model` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Neural Radiance Field` `Auto Encoder` `Generative Adversarial Network` `Contrastive Learning` `Gaussian Splatting` `Simultaneous Localization and Mapping` `World Model` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation`

**🎯 论文内容**

设计并实现了一种无状态的代码基HQc（HQC）封装协议，允许Tiny Stateless Server通过分块流式传输HQc密文实现后量子前向保密；

**💡 创新点**

创新点在于利用中国剩余定理将多项式乘法拆分为可独立处理的模数残差，并通过掩码随机化确保不泄露密钥；结合客户端谜题、重放过滤器和cookie实现无状态、抗内存耗尽的DoS防护；

**🔧 技术方法**

采用了中国剩余定理（CRT）、掩码随机化、HQC KEM、ML‑KEM‑768认证、BLAKE3哈希、XChaCha20‑Poly1305 AEAD、Rocq/SSProve与ProVerif验证、Rust实现、AVX2/NEON硬件加速；

**📊 数据集**

使用标准NIST参数集（HQC‑1/3/5、ML‑KEM‑768），在AMD Ryzen 7 5800H与Raspberry Pi 3B+上进行实验，并用自制负载生成器模拟网络延迟与丢包；

**📈 对比分析**

与基于Classic McEliece的McTiny及有状态缓冲基线对比；在类别5下传输字节减少27×，服务器CPU降低9×；在无状态模式下内存固定1.2–1.6 MiB，显著抵御内存耗尽攻击；Pi上CPU占用和吞吐率虽低，但仍优于McTiny；

**⚠️ 局限性**

局限性包括额外的CPU开销（2~4×）、需要更多数据包和更长延迟；对多项式乘法实现依赖AVX/NEON；重放过滤器容量与分块大小仍需进一步优化，且无法防止网络链路饱和攻击。

---

## 404. ConformalMOS: Uncertainty-Aware MOS Prediction with Conformal Intervals and Ordinal Modeling

**arXiv ID:** 2610.36126 | [PDF](https://arxiv.org/pdf/2610.36126v1)

**作者:** Kehinde Elelu `[一作]` (Michigan State University), Ebuka Okpala `[通讯]` (Clemson University)

**关键词:** `Sound` `Audio` `Uncertainty Estimation` `Machine Learning for Quality Assessment` `Transformer` `MLP (Multilayer Perceptron)` `Conformal Prediction` `Gaussian Smoothing` `Ordinal Regression` `Feature Extraction` `Audio`

**🎯 论文内容**

提出一种名为ConformalMOS的框架，结合序数化建模与拆分式共形预测，以在合成语音质量评估中同时输出点估计和统计上有效的置信区间

**💡 创新点**

在MOS预测中首次结合共形预测实现覆盖率保证，并用高斯平滑标签保留MOS的序数关系，提高不确定性校准和鲁棒性

**🔧 技术方法**

使用预训练音频编码器（M2D2、wav2vec）提取特征，轻量级MLP作为序数预测头，Gaussian平滑标签，拆分式共形校准产生置信区间

**📊 数据集**

BVCC数据集，包含7,106条英语音频样本，来自187个TTS/VC系统，按官方分割进行训练、验证、校准和测试

**📈 对比分析**

与UTMOS、FUSE‑MOS等基准比较，ConformalMOS在系统层面实现最低MSE（0.08）和最高LCC，置信区间覆盖率近乎达到设定的95%覆盖目标，且平均宽度更窄

**⚠️ 局限性**

仅在BVCC内部数据上验证，未评估跨域泛化；对wav2vec骨干的共形校准效果不佳，显示对底层预测质量高度敏感

---

## 405. Noise Shapes the Surplus: Strategic Play Against No-Regret Learners

**arXiv ID:** 2610.36621 | [PDF](https://arxiv.org/pdf/2610.36621v1)

**作者:** Yiheng Su `[一作]` (University of Wisconsin--Madison), Emmanouil-Vasileios Vlatakis-Gkaragkounis `[通讯]` (University of Wisconsin--Madison)

**关键词:** `Computer Science and Game Theory` `Optimization` `Reinforcement Learning` `Reinforcement Learning` `Contrastive Learning` `Optimization`

**🎯 论文内容**

研究在零和博弈中，优化器如何在对手采用噪声 FTRL（Follow‑The‑Regularized‑Leader）并使用衰减学习率的情况下获取策略盈余；

**💡 创新点**

提出两种新的盈余提取机制：①基于噪声平均响应的选择机制，能在均值可盈利时获得 Ω(1/η_T) 的盈余；②基于学习者移动的移动机制，在满足响应性与稳定性条件下获得 Ω(∑_t< T η_t) 的盈余；并证明在常见噪声模型和 η_t∝t^{-1/2} 时仍能达到 Ω(√T) 的上界。

**🔧 技术方法**

利用 FTRL 的选择映射、噪声相对尺度分析、严格凸正则化的 KKT 条件，以及对噪声模型（新鲜、累计、增长噪声）的极限行为证明；同时结合经典 FTRL 失调与稳定性分析来设计主动策略。

**📊 数据集**

无（本文为理论分析，未使用数据集）。

**📈 对比分析**

相较于以往仅在常数学习率且无噪声环境下的 Ω(√T) 结果，本工作在衰减学习率和一般噪声环境下仍实现同阶上界；通过实验模拟验证了理论预测的极限响应与盈余曲线。

**⚠️ 局限性**

仅适用于零和博弈且对正则化、噪声方差有限、学习率满足 η_t∝t^{-ρ} (0<ρ<1) 的假设；需要优化器事先知晓学习者的更新规则；对非 FTRL 学习者或多玩家/非零和情形尚未给出结果。

---

## 406. Adapting Video Models for Robot Control via Motion-guided Asymmetric Prediction

**arXiv ID:** 2610.36624 | [PDF](https://arxiv.org/pdf/2610.36624v1)

**作者:** Taeyoung Kim `[一作]`, Jinwoo Shin `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Robotic Intelligence` `Transformer` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Vision-Language-Action Model` `Video`

**🎯 论文内容**

提出一种在预训练视频扩散模型原生潜在空间中进行运动导向不对称预测（MoGAP）的视觉监督方法，用于改进World Action Models在机器人控制中的表现。

**💡 创新点**

核心创新在于利用潜在空间中的运动得分自适应地为动态区域分配更高噪声、为静态区域保持更低噪声，从而无需外部教师或额外模态即可让视觉监督更贴合控制需求。

**🔧 技术方法**

技术包括视频扩散模型、流匹配训练、潜在差分计算运动得分、双时隙不对称噪声分配、辅助训练分支和不对称注意力掩码。

**📊 数据集**

实验数据集包括RoboCasa GR-1 Tabletop、多任务仿真基准、RoboCasa Kitchen单臂多视角任务以及真实机器人OpenArm Humanoid。

**📈 对比分析**

与基线WAM、Fast-WAM、Cosmos3、VLA方法及教师监督方法对比，MoGAP在GR-1上平均成功率从52.3%提升至62.3%（+10pp），Fast-WAM从38.8%提升至47.5%；在真实机器人上尤其在对象重定位任务中表现出显著提升。

**⚠️ 局限性**

训练时需要额外的辅助令牌，导致训练时间增加约1.26倍；对运动得分阈值与噪声分配参数的敏感性未深入探讨。

---

## 407. SpeeDiff: Scalable Pixel-Anchored End-to-End Latent Diffusion Model

**arXiv ID:** 2610.36419 | [PDF](https://arxiv.org/pdf/2610.36419v1)

**作者:** Bingliang Zhang `[一作]` (ByteDance Seed), Qiushan Guo `[通讯]` (ByteDance Seed)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Auto Encoder` `Contrastive Learning` `Image`

**🎯 论文内容**

提出 SpeeDiff，一种从零开始端到端联合训练 VAE 与扩散模型的框架。

**💡 创新点**

核心创新包括：1) 引入 Tweedie Pixel Reconstruction (TPR) 损失，防止在端到端训练中出现潜空间坍塌；2) 采用基于 Vision‑Transformer 的 ViT‑VAE 替代传统 CNN‑VAE；3) 在架构中加入 REPA++，将预训练视觉模型（如 DINOv3）与潜空间和中间扩散特征对齐。

**🔧 技术方法**

技术细节包括：Transformer‑based VAE 与 DiT 变体、Tweedie 公式推导的潜像素重建损失、REPA++ 的双重对齐（潜空间对齐 + 中间特征对齐）、无 stop‑gradient 的完整端到端优化、以及使用自回归指导（AutoGuidance）进行采样。

**📊 数据集**

主要数据集为 ImageNet，分别在 256×256 和 512×512 分辨率下进行训练与评估。

**📈 对比分析**

与现有两阶段 LDM、SiT、REPA 等方法对比，SpeeDiff 在 256×256 上无指导下 FID 1.50，512×512 上 FID 1.53；训练收敛速度提升 140 倍（相对 SiT）和 61 倍（相对 REPA）；在无指导的基准上已达到最高生成质量，并显示出更好的可扩展性。

**⚠️ 局限性**

局限性包括：仍需依赖预训练视觉模型进行对齐，可能在非 ImageNet 数据集或更高分辨率下表现有限；TPR 需要额外解码步骤，略增计算开销；对极端超大模型的训练稳定性未深入验证。

---

## 408. Exponential Wake-Up Overshoot in Explicit Gaussian Point-Process Filters

**arXiv ID:** 2610.35958 | [PDF](https://arxiv.org/pdf/2610.35958v1)

**作者:** Hanwu Qin `[一作]` (Zhejiang University), Dianhai Wang `[通讯]` (Zhejiang University)

**关键词:** `Machine Learning` `Finance Related` `Optimization` `Stochastic Differential Equation` `Ordinary Differential Equation` `Contrastive Learning` `Probability and Limit Analysis` `Time Series` `Tabular`

**🎯 论文内容**

研究了显式高斯点过程滤波器在长时间无事件后醒来时的指数放大现象，给出了该更新的分布、静默期间状态方差的二阶展开、以及判断是否出现跨越风险的可计算指标；同时证明了显式与隐式更新在静默与醒来阶段的误差特性，并通过合成与真实交易流实验验证理论预测。

**💡 创新点**

首次推导显式滤波器在静默期的状态方差二阶展开及醒来更新的精确分布，并提出跨越指数 κ 作为预测醒来放大是否超过真实速率的阈值；证明显式更新在固定 bin 宽度下误差呈指数增长，而隐式更新受对数控制；并通过实验验证理论指标的预测能力。

**🔧 技术方法**

采用扩展卡尔曼滤波框架、随机游走状态模型、泊松计数观测、Lambert W 函数求隐式解、连续时间极限分析、Brownian 指数函数的光谱表示与极限理论，以及概率与极限分析工具。

**📊 数据集**

合成实验流（控制静默长度、速率、噪声水平）以及 Binance USD‑M 永续合约的真实交易数据（10 个合约、20 个交易日）。

**📈 对比分析**

通过理论指标 κ 与实验量化的超额损失、日志得分等指标进行比较。实验显示在所有 73 个跨越细胞中，隐式滤波器的平均超额损失为 1.4×10³ nats，显式滤波器为 5.9×10⁴⁰ nats；在真实数据中，隐式滤波器获得更高的日志得分；显式滤波器在无跨越细胞中误差相对较低。

**⚠️ 局限性**

主要局限包括：隐式更新在连续时间极限下仍可能出现指数放大，缺乏对所有 bin 宽度和静默长度的统一误差界；变异截断可能降低隐式性能；实验仅针对单步更新，未考虑多步迭代或非线性观测；结果依赖于对数链接的特定假设。

---

## 409. Knowledge distillation across teacher architectures for compact diabetic retinopathy grading

**arXiv ID:** 2610.35961 | [PDF](https://arxiv.org/pdf/2610.35961v1)

**作者:** Mossa Manisa `[一作]` (Shahjalal University of Science and Technology), A. K. M. Fakhrul Hossain `[通讯]`

**关键词:** `Machine Learning` `Knowledge Distillation` `Classification` `Computational Efficiency` `Convolutional Neural Network` `Transformer` `Knowledge Distillation` `Temperature Scaling` `Hard Label Weighting` `Cross-Entropy Loss` `KL Loss` `Transfer Learning` `Image` `Biomedical Data`

**🎯 论文内容**

在APTOS 2019数据集上将ResNet‑50、MobileNetV3‑Large和Swin‑T三种教师模型的知识蒸馏到MobileNetV3‑Small学生，实现五级糖尿病视网膜病变分级；

**💡 创新点**

首次在同一实验协议下系统比较CNN与Transformer教师对同一轻量化学生的影响，并探索温度与硬标签权重的交互效应；

**🔧 技术方法**

采用离线响应式知识蒸馏、温度缩放、硬标签加权、交叉熵+KL损失以及ImageNet预训练迁移学习；

**📊 数据集**

使用公开的APTOS 2019 Blindness Detection数据集（预分割的训练/验证/测试集）；

**📈 对比分析**

通过教师×温度×α的24格网格和多随机种子验证，最佳配置（Swin‑T→MobileNetV3‑Small，T=4，α=0.7）单次种子QWK 0.9093，三种种子平均0.9024，参数缩减18倍，模型尺寸5.94 MB；

**⚠️ 局限性**

结果受单一随机种子和单一数据集划分影响，未在外部数据集或真实硬件上验证，教师单一检查点和稀缺级别样本导致性能估计不稳。

---

## 410. Learning Side-by-Side Navigation for Personal Mobility Vehicles from Human Drivers

**arXiv ID:** 2610.36726 | [PDF](https://arxiv.org/pdf/2610.36726v1)

**作者:** Luis Yoichi Morales `[一作]`, Florent Ferreri `[通讯]`

**关键词:** `Robotics` `Autonomous Driving` `Optimization` `Reinforcement Learning from Human Feedback` `Reinforcement Learning` `Simultaneous Localization and Mapping` `Maximum Entropy Inverse Reinforcement Learning` `Model Predictive Control` `Sequential` `Point Cloud`

**🎯 论文内容**

提出一种去中心化共享规划方法，让两辆个人移动载具在保持侧对侧对话式队形的同时实现自主安全平稳行驶。

**💡 创新点**

创新点包括：1）仅通过共享里程计信息，利用互为转置的效用矩阵完成联合决策；2）离线从地图中提取可行通道来确定队形几何；3）结合硬性安全层与回退策略保证不撞墙不卡顿；4）使用逆强化学习从人类对话式驾驶数据中学习参数。

**🔧 技术方法**

使用采样式模型预测控制（MPC）、最大熵逆强化学习、ROS 2仿真平台和差分驱动动力学模型。

**📊 数据集**

数据集为20名驾驶员（10对）在室内走廊共计409分钟（85段录制）的真实对话式驾驶轨迹，包含清道和障碍物两种场景。

**📈 对比分析**

与人类驾驶者的对比显示，系统在障碍物和空道两种环境下，侧对侧队形保持时间分别提升至约60‑90%，无墙壁/圆锥碰撞，停顿率为0；速度与人类保持一致，且横向间距更均匀。

**⚠️ 局限性**

局限性包括：1）仅实现单一侧对侧队形，无法复制人类的两种交替形成；2）安全规则过于保守导致减速频繁；3）仿真评估样本有限，未在真实环境中验证；4）轨迹生成仍以车辆自身坐标为基础，难以规划“先让后跟”的交叉动作。

---

## 411. How Learning Selects Equilibria and When It Leads to Collusion

**arXiv ID:** 2610.36306 | [PDF](https://arxiv.org/pdf/2610.36306v1)

**作者:** Hanzhang Wang `[一作]` (University of California, Berkeley), Manxi Wu `[通讯]` (University of California, Berkeley)

**关键词:** `Computer Science and Game Theory` `Reinforcement Learning` `Reinforcement Learning` `Markov Game` `Entropy Regularization` `GLRT Hypothesis Testing` `Young Random Potential Framework`

**🎯 论文内容**

研究了学习算法设计如何影响无限时延折扣马尔可夫博弈中均衡的选择，提出一种基于假设检验与价值相关探索的周期性学习动态；

**💡 创新点**

创新点在于将学习动态与随机稳定性相结合，证明一致性状态下的平稳Nash均衡被选中，并通过玩家特定的价值变换实现对均衡的max‑min选择；

**🔧 技术方法**

使用了熵正则化的光滑最优响应、GLRT假设检验、Young随机潜能框架以及马尔可夫游戏与有限记忆重复博弈的映射；

**📊 数据集**

无实验数据，纯理论推导；

**📈 对比分析**

通过理论证明展示了更长公共记忆、足够的耐心与精准监测能导致算法性合谋，且在对称公司情况下可逼近垄断收益；

**⚠️ 局限性**

局限在于需要完整模型信息、同步周期与重启机制，状态空间随记忆长度指数增长，缺乏可扩展的去中心化实现。

---

## 412. A Quantitative Approach to Asch Conformity in Frontier Large Language Models

**arXiv ID:** 2610.36668 | [PDF](https://arxiv.org/pdf/2610.36668v1)

**作者:** Senthil Rajasekaran `[一作]` (Universite libre de bruxelles), Stylianos Loukas Vasileiou `[通讯]` (New Mexico State University)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Distribution Distance Measurement (Total Variation, Jensen-Shannon)` `Statistical Testing (Fisher Exact Test, Holm Correction, Wilcoxon Signed-Rank Test)` `Text`

**🎯 论文内容**

本文改编 Asch 服从实验，利用 LLM 在 2015 年 SAT 风格填空题上多次采样，构造 preferred‑model‑words 与 non‑preferred‑model‑words，测量在呈现制造共识前后模型答案分布的变化。

**💡 创新点**

创新点在于：①使用分布‑到‑分布的比较方法，通过重复采样得到概率分布；②不再仅统计是否选择一致，而是量化答案分布的位移（conformity shift）及其对不同答案类型的依赖；③展示了小规模共识就能显著诱导 LLM 服从的现象。

**🔧 技术方法**

主要技术包括：高频词抽取、近义/远义词生成、随机排列答案、统计学检验（Fisher 精确检验、Holm 校正、Wilcoxon 符号秩检验）、分布距离度量（总变差、Jensen‑Shannon）。

**📊 数据集**

使用了十条 2015 年 SAT 句子完成题作为实验文本，且用 GPT‑5.6‑terra 生成近义/远义词。

**📈 对比分析**

通过在六款前沿 LLM（OpenAI gpt‑6‑sol/luna、Anthropic claude‑sonnet‑5‑5/opus‑5‑5、Google gemini‑3.8‑flash/4‑argon）上进行 100 次独立采样，比较 alone 与 group 阶段的答案分布。结果显示：preferred‑model‑word 的服从提升可达 49 个百分点，near non‑preferred‑word 约 30 个百分点，far 近乎无效；大部分问题在 n=1→5 的规模增大时表现出显著提升，n=5→10 的增幅有限。

**⚠️ 局限性**

局限性包括：①仅通过采样近似模型真实分布，导致额外不确定性；②实验仅限于 SAT 句子填空，缺乏跨任务的泛化；③无法直接访问模型内部分布，限制了分析深度；④共识设计为“制造”而非自然交互，影响外部可解释性。

---

## 413. Packet Interval Modulation-based NFC for Scalable IoT around Meandered E-textile

**arXiv ID:** 2610.35870 | [PDF](https://arxiv.org/pdf/2610.35870v1)

**作者:** Shigemi Ishida `[一作]` (Future University Hakodate), Satoshi Ido `[通讯]` (Future University Hakodate)

**关键词:** `Networking and Internet Architecture` `Optimization` `Computational Efficiency` `Time Series` `Tabular` `Benchmark`

**🎯 论文内容**

提出了一种基于包间隔调制的无同步NFC标签通信方案，用于在可穿戴读卡器与稀疏环境传感标签之间实现高密度、低碰撞的数据采集。

**💡 创新点**

创新点在于将信息编码到短脉冲之间的时间间隔，而非传统的包内容；采用异步、无时隙机制，并通过随机退避窗口和双次传输显著降低碰撞率，提升吞吐量。

**🔧 技术方法**

使用高频子载波424 kbps、最小化的2字节帧结构、随机退避窗口（2^14计时单位）以及脉冲间隔映射等技术实现异步通信。

**📊 数据集**

数据集为仿真生成的标签数量从1到20、100 000次实验的随机时间戳和8 bit传感数据；未使用真实硬件数据集。

**📈 对比分析**

与传统基于时隙的ALOHA协议对比；仿真显示在10+标签场景下，所提方法吞吐量约为ALOHA的4倍，单标签吞吐率稳定在160 bps以上，碰撞率低于5%。

**⚠️ 局限性**

局限性包括：实验仅限仿真，缺乏真实硬件验证；对标签数超过20的扩展尚未评估；依赖读写器与标签对齐的载波同步，且在极端功耗或时钟漂移条件下性能未知。

---

## 414. A Fibrational Framework For Multi-Conclusion Logics

**arXiv ID:** 2610.36552 | [PDF](https://arxiv.org/pdf/2610.36552v1)

**作者:** Oualid Merzouga `[一作]`, Oualid Merzouga `[通讯]`

**关键词:** `Logic in Computer Science` `Other` `Review/Survey Paper`

**🎯 论文内容**

提出了一个两层图基模式与类型理论的框架，统一了单输出卡氏逻辑和多输出线性逻辑的证明系统，支持资源敏感与模态的灵活组合。

**💡 创新点**

创新点在于将传统树形上下文表达式替换为图形描述器，并通过描述器2-细胞与F/U连接词构建了可切分的多结论计算规则，同时给出多重cut与类比线性对称的证明与语义对应。

**🔧 技术方法**

采用了模式理论、图形描述器、偏范畴/多范畴/多极范畴、线性-非线性（LNL）多极范畴、纤维化与可分化等范畴理论与证明论技术。

**📊 数据集**

该工作不使用任何实验数据集，全部基于形式语义与理论证明。

**📈 对比分析**

通过与Licata–Shulman–Riley的单输出框架以及Blanco的经典多极线性逻辑范例进行语义与证明结构的对应与适用性对比，证明了系统的结构性规则与cut的可接受性，表现出与现有框架相当甚至更为通用的理论一致性。

**⚠️ 局限性**

局限性在于目前仅处理命题子句，未涵盖依赖类型、任意多极基底的可分化和完整的初性结果，需要进一步扩展到更一般的多极/多极范畴与初性理论。

---

## 415. ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization

**arXiv ID:** 2610.36499 | [PDF](https://arxiv.org/pdf/2610.36499v1)

**作者:** Zexin Li `[一作]` (George Mason University), Simin Chen `[通讯]` (George Mason University)

**关键词:** `Software Engineering` `Optimization` `Computational Efficiency` `Explainability and Interpretability` `Contrastive Learning` `Probabilistic Verification` `Image` `Text`

**🎯 论文内容**

提出了一个单一序列化测试的黑盒概率验证框架，用于评估深度学习编译器优化对模型语义一致性的影响。

**💡 创新点**

创新点在于用单一序列化测试实现更严格的三值决策合同，消除区间族并通过修正的Clopper–Pearson或任意有效置信序列实现查询效率。

**🔧 技术方法**

采用概率阈值检验、二项分布置信区间、任意有效置信序列以及对模型一致性的决策级不一致性Oracle。

**📊 数据集**

在24个DL任务（如ResNet）上使用TorchInductor与TensorRT编译后模型，采用统一L2/L∞球、文本token替换、图像补丁等分布进行采样。

**📈 对比分析**

与PROVERO的区间族方法比较，单测在相同误差预算下平均使用3.6–4.2倍更少的独立样本；在840个输入–后端实例中得到817个接受、19个不确定、4个拒绝，展示了更高的查询效率和更强的错误控制。

**⚠️ 局限性**

假设采样独立且Oracle无状态，无法处理多分布或罕见事件，且缺乏对全局一致性或多种种子整体保障。

---

## 416. EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training

**arXiv ID:** 2610.37062 | [PDF](https://arxiv.org/pdf/2610.37062v1)

**作者:** Masato Fujitake `[一作]`, Masato Fujitake `[通讯]`

**关键词:** `Machine Learning` `Optimization` `Computational Efficiency` `Representation Learning` `Mixture of Experts` `Large Language Model` `Mixture of Experts` `Newton–Schulz Iteration` `Frobenius Energy-based Scaling` `Text`

**🎯 论文内容**

在稀疏混合专家（MoE）语言模型训练中引入 EnergyCal 规则，按已完成 Newton–Schulz 方向能量动态抑制每个专家的更新步幅。

**💡 创新点**

创新点在于：①仅使用单次 NS 方向能量作为可调系数；②不增加额外优化器状态；③兼容 MuonH 与普通 Muon，且能显著降低训练中的 loss 峰值。

**🔧 技术方法**

技术方法包括：Newton–Schulz 迭代求解近似正交方向、基于 Frobenius 能量的动态缩放系数、与 Muon/Hyperball 结合的权重投影与写回、实验对比评估。

**📊 数据集**

实验使用 4B 参数、128 专家、top‑2 路由的 DeepSeekMoE‑style 模型，训练 1.31B token（FineWeb 数据集）。

**📈 对比分析**

比较方法：在相同硬件、相同数据顺序下，对 MuonH、plain Muon、Muon+WD 以及 EnergyCal‑MuonH/Plain‑Muon 进行三次种子实验；评估指标包括最终验证交叉熵、loss 峰值负载（B₁）、训练吞吐量。结果显示 EnergyCal‑MuonH 将峰值负载降低 99.51%，最终 perplexity 提升 4.06%，吞吐量仅下降 1.62%；Plain‑Muon 上 EnergyCal 将峰值负载降低 95.98%，且最终 perplexity 微幅提升。

**⚠️ 局限性**

局限性包括：①对比学习率、权重衰减不同，无法单独归因于能量缩放；②未提供仅缩放系数统一的对照实验；③仅在单一 4B 模型和数据上验证，缺乏跨规模、跨数据集的泛化；④单步干预未能消除实例峰值，因而对机制的因果解释有限。

---

## 417. V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models

**arXiv ID:** 2610.37019 | [PDF](https://arxiv.org/pdf/2610.37019v1)

**作者:** Miso Choi `[一作]` (Korea University), Jungbeom Lee `[通讯]` (Korea University)

**关键词:** `Machine Learning` `Representation Learning` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Vision Language Model` `Mixture of Experts` `Retrieval-Augmented Generation` `Contrastive Learning` `Image` `Text` `Multimodality`

**🎯 论文内容**

设计并评估了一种名为V-Lumen的外部视觉查找记忆，能在不增加每个token计算量的前提下扩展视觉语言模型的视觉表示容量；

**💡 创新点**

通过冻结代码表对视觉特征离散化、空间聚合哈希、文本条件哈希以及多头条件门控，将视觉记忆作为独立可扩展的容量轴；

**🔧 技术方法**

采用代码表映射、空间聚合哈希（SAH）、文本条件哈希（PRH/DRH）、多头条件门控（MHCG）、外部可扩展嵌入表以及LoRA/MoE融合等技术；

**📊 数据集**

使用了DocVQA、ChartQA、InfographicVQA、ScienceQA、Visual Genome、GQA、RefCOCO、AOKVQA、TextCaps、AI2D、ChartMuseum、FRIEDA、VisRes、MMMU、MMMU-Pro等数据集；

**📈 对比分析**

与冻结backbone、LoRA、MoE等方法在匹配总预算下进行对比，V-Lumen在五个推理基准上平均提升约+17.8个百分点，零样本迁移亦提升数个百分点，证明其作为独立容量轴的有效性；

**⚠️ 局限性**

尚未验证跨GPU内存外部署、关键构造开销对性能的影响、内存行数扩展对准确率的提升有限，以及在更大模型上的可扩展性待进一步验证。

---

## 418. Can AI Guide Testing Without Generating Tests? A Comparative Study of AI-Assisted Unit Testing

**arXiv ID:** 2610.36185 | [PDF](https://arxiv.org/pdf/2610.36185v1)

**作者:** Zihan Fang `[一作]` (Vanderbilt University), Gina R. Bai `[通讯]` (Vanderbilt University)

**通讯引用:** 339 | [OpenAlex ID](https://openalex.org/A5040160847)

**关键词:** `Software Engineering` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Large Language Model` `Prompt Engineering` `Code Coverage Analysis` `Static Analysis` `Mutation Testing` `Text` `Code`

**🎯 论文内容**

本文提出了一种名为 TestCompass 的反馈型 AI 辅助工具，专门给出针对测试缺口的定向反馈，而不直接生成可执行测试代码，并在课堂实验中与通用 AI 助手进行对比。

**💡 创新点**

创新点在于将 AI 从直接生成测试代码转变为仅提供覆盖率驱动的反馈，既帮助学生发现待测行为，又保留了学生主动编写和验证测试的机会。

**🔧 技术方法**

使用了 IntelliJ IDEA 插件、代码覆盖分析、LLM（ChatGPT）生成反馈、双阶段验证、SonarQube 静态分析以及 PIT 变异覆盖等技术。

**📊 数据集**

实验使用了两道 Java 单元测试任务（MarsRover 与 BowlingScoreKeeper），共包含 12/13 条需求和 798 条测试方法，构成了数据集。

**📈 对比分析**

通过 18 名学生的实验，对比了测试行为、测试套件质量（覆盖率、变异覆盖、维护性、错误率）以及主观体验。结果显示两者在需求覆盖率相近，TestCompass 能用更少的测试方法完成相同覆盖；通用 AI 生成更多测试，略高覆盖率，但差异不显著。

**⚠️ 局限性**

局限性包括样本量有限、仅在单门课程进行实验、任务顺序固定、通用 AI 助手多样化导致差异、仅进行短期评估、仅适用于 Java+JUnit 环境，缺乏长期学习效果验证。

---

## 419. Beyond the Evidence Coherence Trap: Agent-Driven Evidence Selection for Temporally Disordered Fake News Detection

**arXiv ID:** 2610.36764 | [PDF](https://arxiv.org/pdf/2610.36764v1)

**作者:** Chi Wang `[一作]` (Chongqing University), Wentao Li `[通讯]` (University of Leicester)

**关键词:** `Artificial Intelligence` `Classification` `Anomaly Detection` `Transformer` `Recurrent Neural Network` `Large Language Model` `Retrieval-Augmented Generation` `Agentic AI` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

研究时序混乱的假新闻检测，提出 DelSA（自更新文档选择代理）并构建 ChronoFake 基准。

**💡 创新点**

创新点在于利用 LLM 生成多维质量向量并通过自更新规则书迭代修正证据选择，避免“证据一致性陷阱”，实现一次性高效推理。

**🔧 技术方法**

技术栈包括 gpt‑5.4‑mini、BERT、BGE‑M3 检索 + FAISS、MLP 重排序器、双向 LSTM 分类器，以及规则书与时间线同步的评估流程。

**📊 数据集**

使用 ChronoFake（30,706 篇）以及 PolitiFact、GossipCop 作为验证和泛化数据集。

**📈 对比分析**

与内容、LLM 增强和检索增强基线对比，DelSA 在 ChronoFake 上准确率 0.892、F1 0.893；在 PolitiFact 上准确率 0.932、F1 0.932；在 GossipCop 上准确率 0.932、F1 0.932，均优于对照方法。

**⚠️ 局限性**

局限性包括对规则书的生成依赖 LLM，训练耗时且对大规模推理需要冻结规则；在无时序干扰的传统假新闻数据上优势不如传统内容模型。

---

## 420. Learning to Look Beyond Screens: Transferable Visual Observation from GUI Grounding

**arXiv ID:** 2610.37093 | [PDF](https://arxiv.org/pdf/2610.37093v1)

**作者:** Ruilin Yao `[一作]` (Wuhan University of Technology), Changhua Meng `[通讯]` (Ant Group)

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Optimization` `Representation Learning` `Transformer` `Supervised Fine-Tuning` `Reinforcement Learning` `Vision Language Model` `Prompt Engineering` `Image` `Text` `Multimodality`

**🎯 论文内容**

构建了一个名为 LBS 的框架，通过先在 GUI 交互中进行监督微调（SFT），随后在多轮强化学习（RL）中优化裁剪‑检查决策，以实现主动视觉感知，并将学到的观察策略迁移到非 GUI 的视觉任务中。

**💡 创新点**

创新点在于：①利用 GUI 的空间标注作为中间观察行为的监督，设计了以结果为门控的空间奖励和选择性响应更新；②在奖励中加入小目标检视奖金和文本一致性批评，提升了对观察质量的细粒度引导；③通过实验验证这种 GUI‑导向的监督能够迁移到自然图像、图表和高分辨率视觉基准，证明观察策略的通用性。

**🔧 技术方法**

技术手段包括：使用 Qwen3.5‑35B‑A3B 与 Qwen3.8‑27B 视觉‑语言模型，冻结视觉编码器，仅更新语言模块；SFT 通过教师生成的多轮 GUI 交互轨迹进行监督；RL 采用多轮裁剪‑检查框架，利用裁剪边界与目标框 IoU 计算空间奖励、重复/回退惩罚，并引入小目标奖金；采用 GRPO‑风格的分组相对优化和 PPO‑style 剪切策略；使用裁剪工具实现可复现的局部视图和坐标映射。

**📊 数据集**

数据集主要为：①通过代码生成的合成 GUI 数据（含指令–目标框配对，约 45k/10k 轨迹）；②评测基准包括 ScreenSpot‑Pro（GUI 定位）、PerceptionBench（通用视觉理解）、HR‑Bench 4K/8K、ChartQAPro、OSWorld‑G‑R 以及其它高分辨率和图表任务。合成数据不包含评测基准图像。

**📈 对比分析**

与基线（未微调模型）和仅 SFT 的版本进行对比；RL 在同一 backbone 上显著提升：ScreenSpot‑Pro 的准确率从 78.56%→80.39%（Qwen3.5）或 75.02%→81.15%（Qwen3.8）；PerceptionBench 上从 38.6%→45.1%（Qwen3.5）或 46.6%→59.7%（Qwen3.8）；在 HR‑Bench、ChartQAPro 等非 GUI 任务上也实现 3–7% 的提升，优于单纯结果导向的 RL。实验表明，仅提升 GUI 定位并不一定带来跨域提升，强调中间观察奖励的重要性。

**⚠️ 局限性**

局限性包括：①奖励设计对不同任务的敏感度高，需多项奖励平衡；②目前仅验证在静态图像上的迁移，未覆盖视频或流式输入；③对合成 GUI 数据的依赖，可能限制在真实交互环境中的适用性；④部分模型对 prompt 形式的遵循不稳定；⑤裁剪工具调用和多轮交互增加推理延迟与计算成本。

---

## 421. Benchmarking Task Takeover Across Agent Handoffs

**arXiv ID:** 2610.36523 | [PDF](https://arxiv.org/pdf/2610.36523v1)

**作者:** Wenyuan Li `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**关键词:** `Computation and Language` `Benchmark` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Code` `Benchmark`

**🎯 论文内容**

提出了HSC-Bench benchmark，用于评估大型语言模型在任务接管时如何检索先前证据并继续未完成的工作。

**💡 创新点**

创新点在于同时设计两条评估轨道（代码证据恢复与状态依赖继续），并引入结构化状态信息来显式记录先前决策、操作与文件状态。

**🔧 技术方法**

使用了GPT‑4.1生成结构化状态与叙事摘要，评估模型（GPT‑5.4、Claude Sonnet 5、DeepSeek V3.2、Qwen 3.5 Plus、Gemini 3.5 Flash Lite、Kimi K2），并采用行级F1和基于二进制标准的任务完成率。

**📊 数据集**

数据集包括300个基于ContextBench的代码问题追踪记录和739个合成的对话/工具/文件状态任务，全部从GPT‑5.4的执行轨迹中生成。

**📈 对比分析**

通过在相同手风琴点、相同源历史的条件下比较模型在三种上下文（最近转录、结构化状态、完整历史）下的表现，发现结构化状态能提升所有模型的证据恢复（平均行F1提升约0.12）并在多数模型的继续任务中提升平均得分，但不同模型在证据恢复与继续任务中的强项不同。

**⚠️ 局限性**

局限在于只考虑单次接管情景，未分析多轮接管或误导性完成记录对后续模型的影响，且结构化状态的好处因任务类型而异，不能完全统一解释。

---

## 422. DFIP: Decision-Focused Inference Path for System One Models

**arXiv ID:** 2610.36804 | [PDF](https://arxiv.org/pdf/2610.36804v1)

**作者:** Ismail Pelaseyed `[一作]` (Superagent Technologies Inc.), Alan Zabihi `[通讯]` (Superagent Technologies Inc.)

**关键词:** `Performance` `Computational Efficiency` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Text`

**🎯 论文内容**

实现并评估了 Decision-Focused Inference Path（DFIP）在四个 System One 模型（PPLX-Decider、Liquid d1-3B、Clef-Flash、GLiNER2.5-Decide）上的加速技术，包括选择性重放、候选投影和长度兼容批处理，验证了加速路径的答案保真性。

**💡 创新点**

将现有 CUDA 图重放、词表投影和长度感知批处理技术系统化地应用于决策模型，展示了工作负载特定的加速边界并提出了对答案保真性的实证审计框架。

**🔧 技术方法**

使用 CUDA 图、Triton/CuBLAS 词表投影、长度兼容批处理、容器缓存控制、Python/PyTorch/Transformers 以及 Qwen 内核/编译器等技术。

**📊 数据集**

采用 BANKING77 意图测试集、系统生成的短 Boolean/分类请求、GLiNER 的多尺寸批处理请求，以及外部模拟的到达流进行实验。

**📈 对比分析**

通过与已强化原生基线（安装 Qwen 内核或编译器）比较容器内中位延迟，报告了几倍加速：Decider 2.47×、Clef 1.45×、Liquid 1.01×、GLiNER 1.40×，并对答案一致性进行审计。

**⚠️ 局限性**

加速仅在重复短请求场景显著，异构到达流下性能逆转；CUDA 错误、缓存失效、数值漂移和有限样本限制结果的普适性；未评估真实生产负载、成本与长期可靠性。

---

## 423. REACH: An RL-based Hierarchical Humanoid Crawling Framework for Adaptable Confined Space Traversal

**arXiv ID:** 2610.36546 | [PDF](https://arxiv.org/pdf/2610.36546v1)

**作者:** Shengzhang Xu `[一作]` (Southern University of Science and Technology), He Kong `[通讯]` (Southern University of Science and Technology)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Transformer` `Recurrent Neural Network` `Reinforcement Learning` `Simultaneous Localization and Mapping` `Optical Flow` `Contrastive Learning` `Point Cloud` `Image` `Video`

**🎯 论文内容**

研发了一套名为REACH的层次化强化学习框架，使未改装的Unitree G1在受限空间内通过可变高度的双足步行、爬行以及双向模式切换实现顺畅通行，并配备LiDAR规划器。

**💡 创新点**

将可变高度双足步行与爬行统一在同一RL架构下，采用历史条件潜在适配模块和双向转移策略实现平滑姿态切换，并在规划器中加入Humanoid专用高度约束与障碍抑制。

**🔧 技术方法**

深度强化学习（异步Actor‑Critic + 历史潜在编码）、基于LiDAR的点云地图与A*+优化路径、Pure Pursuit追踪器、Kimodo生成过渡动作、以及在IsaacLab/ MuJoCo 进行大规模并行训练。

**📊 数据集**

仿真中的随机地形集（箱阵列、粗糙地面、波浪地形、斜坡、平台）以及真实Unitree G1的LiDAR点云（Point‑LIO+Elevation Mapping）。

**📈 对比分析**

通过对比四种低级策略的ablation、轨迹跟踪误差与梯度进度，以及转移成功率；实验显示Full变体在仿真中实现100%/96%双向切换，真实机器人实现100%/92%；在受限空间中实现低至40 cm的爬行和80 cm的步行，表现出良好的鲁棒性。

**⚠️ 局限性**

依赖0.5 m的切换阈值，极端动态负载与复杂三维障碍的鲁棒性有限，仿真中地形随机化未覆盖所有真实环境，且训练成本高且对高频传感器信息有一定要求。

---

## 424. When to Intervene? An Empirical Study of Errors and Self-Recovery in LLM Agents

**arXiv ID:** 2610.36471 | [PDF](https://arxiv.org/pdf/2610.36471v1)

**作者:** Chiming Duan `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**关键词:** `Artificial Intelligence` `Anomaly Detection` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Optimization` `Federated Learning` `Robotic Intelligence` `AI Code Assistant` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Mixture of Experts` `Vision Language Model` `Vision-Language-Action Model` `Conditional Decision Fusion` `Text` `Multimodality` `Sequential` `Benchmark`

**🎯 论文内容**

对LLM代理在与Web服务交互时的错误与自我恢复行为进行系统研究，并提出基于自我恢复预测的干预决策方法。

**💡 创新点**

①首次构建专门记录错误、识别与自我恢复的 ReInBench 数据集；②提出 ReCoF，将历史恢复模式与上下文评估相结合，实现对错误是否能自行恢复的精准预测；③系统评估干预时机，证明自我恢复对后续错误处理与任务完成的正向影响。

**🔧 技术方法**

利用多模态预训练模型（文本、视觉、图像–文本）训练恢复模式专家；使用评估LLM进行上下文难度判断；通过条件决策融合（Conditional Decision Fusion）动态加权专家预测；整体基于 LLM 推理与可视化反馈的交互式代理执行框架。

**📊 数据集**

ReInBench：1,259 条代理执行轨迹、1,403 个错误事件（725 个自行恢复），涵盖计算机使用与 Web 浏览、文档/电子表格编辑、跨应用工作流等多种任务。

**📈 对比分析**

对比了 Direct LLM、AgentForesight、Sentry、PrefixGuard、Wink 等基线，ReCoF 在 Claude 3.7 Sonnet、Claude Sonnet 4.5、Kimi K2.5 三种执行 LLM 上分别提升了约 4.78%、1.30%、11.30% 的 Macro‑F1，平均相对提升 9.02% 的 Macro‑F1；在自我恢复预测、非恢复召回率等指标上均显著优于最强基线。

**⚠️ 局限性**

①数据集覆盖的 LLM 与任务类型有限，结果可能不具备广泛泛化性；②干预方法未在实际执行中验证，仅预测可恢复性，未评估不同干预内容与时机对任务完成的综合影响；③对错误类型和环境的定义与标注仍存在主观性和局限性。

---

## 425. Structured Time-Frequency Representation for Channel Prediction

**arXiv ID:** 2610.35956 | [PDF](https://arxiv.org/pdf/2610.35956v1)

**作者:** Ning Yang `[一作]` (Chinese Academy of Sciences), Haijun Zhang `[通讯]` (University of Science and Technology Beijing)

**关键词:** `Information Theory` `Representation Learning` `Optimization` `Computational Efficiency` `Transformer` `Recurrent Neural Network` `Contrastive Learning` `Auto Encoder` `Prompt Engineering` `Time Series` `Tabular`

**🎯 论文内容**

在频分双工（FDD）环境下，构造了一种基于联合时频位置字段的结构化预测框架FAR-CSI，用于从历史上行（UL）CSI预测未来下行（DL）CSI；

**💡 创新点**

创新点在于提出FDD‑aware结构化路由规则，使观测UL与目标DL在时频位置空间保持一致的任务结构，并利用预训练的CSI Transformer实现跨频段的高精度预测；

**🔧 技术方法**

核心技术包括：CSI patch化表示、联合时频位置编码、预训练HeterCSI Transformer、物理域比值归一化（RoS）损失及自适应mask重建；

**📊 数据集**

使用QuaDRiGa仿真生成的3GPP UMa NLOS FDD数据集，包含16个历史UL时刻预测4个未来DL时刻；

**📈 对比分析**

与传统RNN、GRU、LSTM、CNN、Transformer、LLM4CP、WiFo等基线以及TDD场景中的PAD3、Generic CSI Transformer对比，FAR-CSI在NMSE上最低（0.1502），比最优基线低约21.4%，在MRT基准谱效率上最高；

**⚠️ 局限性**

局限性包括：仅在单一仿真数据集上验证；未测试在实际硬件环境下的鲁棒性；对高频带宽/大规模MIMO场景的扩展尚未评估。

---

## 426. PHP2Shell and the Future of AI-Discovered Vulnerability Disclosure: Evidence, Verification, and Human Authority

**arXiv ID:** 2610.36334 | [PDF](https://arxiv.org/pdf/2610.36334v1)

**作者:** Toqeer Ali Syed `[一作]` (Islamic University in Madinah), Asadullah Abdullah Khan `[通讯]` (Islamic University in Madinah)

**关键词:** `Cryptography and Security` `Federated Learning` `Safty and Privacy` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

设计并实现 FedTrust‑CVDv2，一个分层联邦协同漏洞披露框架，用以将技术证据、AI 辅助发现、人类授权和权限审计绑定到同一漏洞案例修订。

**💡 创新点**

通过“证据到行动”授权契约，实现在同一修订上独立签名观测、人工审核和权限限制的结合，确保外部披露仅在证据、授权和政策一致时才可执行。

**🔧 技术方法**

采用 Ed25519 签名、AES‑GCM 加密、SHA‑256 摘要、RFC 8785 规范 JSON、BFT 排序（如 Hyperledger Fabric/PBFT）以及 Python/JSON 的实现技术。

**📊 数据集**

生成了 27 个政策家族、每族 25 个身份实例的 675 个构造性测试装置（共 25 个合法、650 个非法案例）作为实验数据集。

**📈 对比分析**

通过对完整门控与单项护栏剔除的对比实验，门控在 650 个非法装置中全数拒绝，且 25 个合法装置全部通过；移除单一护栏时会相应放行对应类别的错误，验证了每项安全检查的必要性。

**⚠️ 局限性**

仅在构造实验环境中验证；未评估 AI 发现准确性、真实漏洞负载或在真实组织流程中的可操作性，且仍需人力审查与跨组织协调。

---

## 427. Real-Time Smart Retail Analytics through Fog Computing using the RetailLightNet Lightweight Model

**arXiv ID:** 2610.35871 | [PDF](https://arxiv.org/pdf/2610.35871v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Networking and Internet Architecture`

---

## 428. From Walks to Skeletons: A Reusable Transport Architecture for Interactive Monte Carlo PDE Solvers

**arXiv ID:** 2610.36931 | [PDF](https://arxiv.org/pdf/2610.36931v1)

**作者:** Yunfeng Liao `[一作]`, Xiucheng Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `Graphics` `Computational Efficiency` `Optimization` `Physics Related` `Stochastic Differential Equation` `Diffusion model` `Score-based Model` `Optical Flow` `Graph` `Mesh` `Physics Related`

**🎯 论文内容**

提出了一种可重用的传输骨架（DAG）框架，利用 Walk-on-Spheres 轨迹的共享与重权重化，实现交互式椭圆 PDE 求解中的持续更新、压缩和局部修复。

**💡 创新点**

创新点在于通过 Poisson 核重权重限定附件、条件路径细化、共享 hub 接口、覆盖保持的图压缩、局部几何修复以及区域低秩传输编辑，显著提升 Monte Carlo PDE 求解的效率与交互性。

**🔧 技术方法**

使用了 Walk-on-Spheres、Poisson 核重权重、Doob h-变换、方向引导、并行 DAG 评估、变差导向精细化、图压缩与锚点选择、低秩修正以及双向 WoS 等技术。

**📊 数据集**

实验数据集包括多物体测试域（多个不规则障碍的球形封闭域）以及特殊域（杯形、双室、火山/村庄岛等），并采用空间变化的 Dirichlet 边界值。

**📈 对比分析**

通过与独立路径 WoS、独立骨架以及无重用的 WoS 对比，实验显示骨架在保持或降低误差的同时求解时间提升 2-3 订单级；在多次边界求解中，骨架在第二次求解后即可快 8 倍；在源项问题、变差导向精细化、hub 密集查询、局部修复与双向 WoS 等子实验中也均表现出显著的计算节省与误差控制。

**⚠️ 局限性**

局限性包括附件阈值与重要性权重的折衷导致附件效率受限；骨架构造与更新仍需额外开销；压缩后锚点数量与覆盖率需要平衡；对极小或难以采样的几何对象仍需双向 WoS；低秩修正只能局部调整，无法改变 PDE 本身。

---

## 429. Immersive Commons: Designing Ephemeral Spatial Dialogue for Co-Located Mixed Reality Environments

**arXiv ID:** 2610.36014 | [PDF](https://arxiv.org/pdf/2610.36014v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Human-Computer Interaction`

---

## 430. ClusterSplat: Semantic Cluster Selection for 3D Visual Grounding in Gaussian Splatting

**arXiv ID:** 2610.36462 | [PDF](https://arxiv.org/pdf/2610.36462v1)

**作者:** Jaesung Lee `[一作]` (KAIST), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Pose Estimation` `Representation Learning` `Transformer` `Gaussian Splatting` `Vision Language Model` `Contrastive Learning` `Point Cloud` `Text` `Image`

**🎯 论文内容**

在3D Gaussian Splatting（3DGS）框架中，将3D视觉定位从原先的原子级得分过滤改为基于场景自适应聚类的离散选择，从而实现对指代表达中单一目标的精准3D目标与2D分割。

**💡 创新点**

①通过实例感知的高斯特征学习与描述长度驱动的场景自适应聚类，动态生成符合场景结构的候选集；②使用查询条件的聚类评分网络（CLIP文本+Transformer+MLP）实现对聚类的直接排名，避免了对阈值或比例的敏感；③将聚类得分直接作为可渲染的2D分割信号，实现了显式3D目标与可视化mask的统一。

**🔧 技术方法**

实例感知高斯特征学习、基于描述长度的最小生成树合并、CLIP文本编码、Transformer解码器、MLP聚类评分器、基于高斯光栅化的可渲染分割。

**📊 数据集**

ScanRefer（与ScanNet v2场景配对的指代表达数据集）、ScanNet v2（用于类别名称定位）以及Ref-LERF（细粒度空间推理验证）。

**📈 对比分析**

与LangSplat、OpenGaussian、Dr.Splat、ReferSplat等3DGS开放词汇基线对比；ClusterSplat在ScanRefer上取得39.72% mIoU（2D）/23.33% 3D mIoU，54.35% mAcc@0.25，优于所有基线；在ScanNet v2类别名称定位中同样取得最高的mAcc与接近最优的3D IoU；在Ref-LERF上获得31.14% mIoU、41.70% mAcc@0.25。

**⚠️ 局限性**

①聚类候选集若被过度合并或拆分，可能导致目标被误覆盖或遗漏；②目前仅支持单一目标指代，无法处理复数或变形数量的表达；③每个新场景需要离线预处理，限制了实时应用与大规模场景迁移；④对复杂遮挡或相似物体的区分仍依赖聚类质量。

---

## 431. ReCoPII: A Relation-Grounded Framework for Building Contextual PII Masking Datasets

**arXiv ID:** 2610.35925 | [PDF](https://arxiv.org/pdf/2610.35925v1)

**作者:** Taehwan Lee `[一作]` (Hanyang University), Taehwan Lee `[通讯]` (Hanyang University)

**关键词:** `Computation and Language` `Safty and Privacy` `Data-Centric Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text`

**🎯 论文内容**

提出一种基于实体关系的上下文PII遮掩框架ReCoPII，利用本体约束规划、LLM占位符生成和独立审核，自动构造MASK/KEEP标签而不假设单一数据主体。

**💡 创新点**

创新点在于：①不依赖作者假设，而是通过域条件实体关系图推断遮掩决策；②在LLM写入占位符前先规划关系，确保文本支持；③采用独立LLM审核验证关系隐私性并自动生成标签；④构建包含实体关系和边级隐私标签的ReCoPII-Ko语料库。

**🔧 技术方法**

使用技术包括：大型语言模型（GPT‑4o‑mini、GPT‑5.6 Luna等）进行规划与占位符生成、边标签审查；基于CRF+Gazetteer+MLP的A.X检测器；XLM‑R‑large、KLUE‑R‑base等窗口模型；关系条件头与直接头的上下文遮掩头；联合微调（Joint FT）。

**📊 数据集**

数据集：ReCoPII‑Ko，由原始Basic语料（2,355篇韩文长文）筛选、注释实体关系、生成MASK/KEEP标签后得到1,857篇文档，涵盖六个领域、18种实体类型、25,656条关系边。

**📈 对比分析**

比较方法：将ReCoPII-Ko与SPY式单作者增强的同源文档进行对比。结果显示：人类评审对ReCoPII-Ko KEEP标签的一致率达88.42%，SPY仅26.27%；基于ReCoPII-Ko训练的分类器宏F1提升3–4点，过遮掩率下降约6–9%；联合微调后，过遮掩率≤2.8%，泄漏率≤1.44%；在人类验证样本上，过遮掩率在2.5–5%之间，泄漏率4%以下。

**⚠️ 局限性**

局限性：仅在合成韩文文档上评估，未检验自然文本或其他语言；大部分评估使用生成标签，缺乏全面的真实隐私判定；人类验证覆盖面有限（每文档最多四条实体），且保留实体多集中在插入句中，可能影响指标的普适性。

---

## 432. Executable Archaeology of Programs That Model Programming: Reanimating the Heuristic Compiler (1961) and KBEmacs (1984)

**arXiv ID:** 2610.36341 | [PDF](https://arxiv.org/pdf/2610.36341v1)

**作者:** Jeff Shrager `[一作]`, Mia Golec `[通讯]` (Carnegie Mellon University)

**关键词:** `Artificial Intelligence` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Computational Efficiency` `Representation Learning` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Sequential` `Review/Survey Paper`

**🎯 论文内容**

重新实现并运行了1961年的Heuristic Compiler和1984年的KBEmacs，利用可执行考古学方法恢复并验证其原始行为。

**💡 创新点**

通过在现代环境下复现两套不同年代、不同目标的自动编程系统，揭示它们与当前LLM助手在概念上的连续性，并展示原始算法在当今技术栈中的可执行性。

**🔧 技术方法**

构建IPL‑V解释器与Zetalisp兼容层、使用Common Lisp与SBCL、借助Claude Code（Anthropic）完成代码转录、修正与调试，并采用可执行考古学框架进行重演。

**📊 数据集**

使用从CMU档案恢复的约4,470行Heuristic Compiler纸面列表和MIT 1985版KBEmacs四卷约34,500行源代码，以及KBEmacs自测中的SQRT等验证序列作为原始数据集。

**📈 对比分析**

通过逐行与Simon 1963论文和KBEmacs自测的预期输出比较，验证重现程序与原始结果完全一致；重点关注功能正确性，未进行时间或资源性能评估。

**⚠️ 局限性**

仍缺少完整的英语前端与未完成的自测序列，KBEmacs仅实现核心无GUI版本；知识库规模有限，无法解决大规模知识获取问题；仅验证功能正确性，未评估其在现代开发流程中的实用性与性能。

---

## 433. Don't Pick a Winner: Knowledge Conflicts Need a Prior, Not a Rule

**arXiv ID:** 2610.36169 | [PDF](https://arxiv.org/pdf/2610.36169v1)

**作者:** Minh Pham Van `[一作]` (Posts and Telecommunications Institute of Technology), Cong Tran `[通讯]` (Posts and Telecommunications Institute of Technology)

**通讯引用:** 275 | [OpenAlex ID](https://openalex.org/A5067894717)

**关键词:** `Artificial Intelligence` `Retrieval` `Explainability and Interpretability` `Computational Efficiency` `Knowledge Distillation` `Representation Learning` `Transformer` `Retrieval-Augmented Generation` `Large Language Model` `Prompt Engineering` `Text` `Graph` `Benchmark`

**🎯 论文内容**

本文提出了一种在检索增强语言模型中对上下文与模型自身知识产生冲突时的决策方法，能够在任意上下文正确率比例下实现最优或近似最优的答案输出。

**💡 创新点**

创新点在于：①从决策理论出发证明固定权重是线性且次优；②证明对称的相似度（如JSD）只能检测冲突而无法指示哪一方正确；③通过为模型自身知识提供校准的置信分布并引入上下文正确率先验，构建了可在部署中自适应调整的选择器。

**🔧 技术方法**

技术主要包括：校准知识笔记的两层变换器及其置信度计算；双声道阅读器（同时输出来自上下文与模型的答案）；基于梯度提升的选择器（加上先验偏移的对数几率调整）；以及对比实验中使用的预训练语言模型（Qwen3.5‑35B‑A3B 与 Llama‑3.1‑8B‑Instruct）。

**📊 数据集**

数据集主要是自构建的“GRACE”基准（基于LC‑Quad 2.0 与 Wikidata 的知识图问答），以及公开的 ClashEval 与 Housing Statute QA 用于外部测试。

**📈 对比分析**

与十三种固定规则方法（context‑follow、contrastive decoding、trained controller、memory‑first 等）及同构骨干的 CK‑PLUG、TCR 等进行比较，系统在所有上下文正确率比例下均高于最佳基线，最高可提升约 26 点；在 ClashEval 上同样取得领先，最大提升 25‑33 点。

**⚠️ 局限性**

局限性主要集中在选择器的训练范围：其校准与泛化依赖于训练时使用的图结构与问题类型，遇到全新类型的查询时可能需要重新训练；此外，系统在模型自身知识不足时几乎不具备提升效果。

---

## 434. A hyper-reduction strategy for elastoplastic hydro-mechanical problems in partially saturated soil

**arXiv ID:** 2610.37060 | [PDF](https://arxiv.org/pdf/2610.37060v1)

**作者:** Alba Muixí `[一作]` (Universitat Politècnica de Catalunya BarcelonaTech), Sergio Zlotnik `[通讯]` (Universitat Politècnica de Catalunya BarcelonaTech)

**关键词:** `Computational Engineering, Finance, and Science` `Computational Efficiency` `Optimization` `Proper Orthogonal Decomposition` `Energy-Conserving Sampling and Weighting (ECSW)` `Galerkin Projection` `Newton–Raphson Iteration` `Tabular` `Physics Related`

**🎯 论文内容**

提出一种适用于部分饱和土壤中弹塑性耦合水力-力学问题的超低阶模型，将位移和孔隙压强分别用 POD 基础表示，并采用基于能量守恒的采样与加权（ECSW）实现超降维；

**💡 创新点**

创新点在于：①将 POD 基础按场拆分，独立训练机械与流动两块的 ECSW 规则；②在每块规则中仅逼近残差的非线性校正项，以精确预先计算的仿射基线为参照；③完全基于 FEniCSx 标准组装流程实现，无需修改本构公式；

**🔧 技术方法**

使用技术包括：Proper Orthogonal Decomposition、Energy-Conserving Sampling and Weighting（ECSW）、Galerkin 投影、Newton–Raphson 迭代、Taylor–Hood P₂–P₁ 有限元、FEniCSx/DOLFINx 计算框架；

**📊 数据集**

实验数据集为两种坝模型（二维平面应变和三维全尺寸）共 10 条测试参数和 40 条训练参数，参数分别为摩擦角 ϕ、杨氏模量 E、饱和渗透率 K_s 与堆填高 h_fill，采用拉丁超立方采样；

**📈 对比分析**

与完整 FEM 及仅减基的 ROM 对比时，超降维模型在保持 0.1%~0.5% 误差的同时，实现 2D 模型 30–43 倍加速、3D 模型 65–77 倍加速；

**⚠️ 局限性**

局限性在于：误差受 POD 截断决定，无法进一步降低；超降维规则需建立在非线性校正项的分离上；若本构模型更为复杂（如迭代返回映射）仍需验证加速效果；

---

## 435. Correction or Manipulation? Measuring Epistemic Vigilance in LLM Agents Under User Pressure

**arXiv ID:** 2610.36453 | [PDF](https://arxiv.org/pdf/2610.36453v1)

**作者:** Nardine Basta `[一作]` (Macquarie University), Amin Beheshti `[通讯]` (Macquarie University)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Safty and Privacy` `Transformer` `Large Language Model` `Prompt Engineering` `Signal Detection Theory` `Logistic Regression` `Bootstrap CI` `Text`

**🎯 论文内容**

构建并评估LLM代理在面对用户挑战时的警觉性和抗压能力，提出基于认识论警觉性与信号检测理论的评估框架，并在回答与行动两类任务中统一测评。

**💡 创新点**

创新点包括：1）将认识论警觉性与信号检测理论结合，区分偏差与敏感度；2）设计单一说服技术的压力语料库，确保压力强度可调；3）在回答与行动任务中同步评估，揭示二者差异；4）提出证据门控作为潜在缓解手段。

**🔧 技术方法**

技术方法包括：使用LLM模型（GPT‑6、Llama、Ministral、DeepSeek）；信号检测理论（d′、c 计算）；等方差高斯模型；logistic 回归与 Bootstrap CI；prompt 工程与系统提示；以及人工评审器对否认影响的判定。

**📊 数据集**

数据集包括：MMLU‑Pro、GPQA‑diamond、MATH‑500 的专业多选题与数学题；手工生成的源文件（真实与虚假教材）；synthetic 账户恢复情境（30 条），以及 1,860 个问题的屏蔽与样本重复检查。

**📈 对比分析**

对比方法：对七个模型在同一问题集合上分别计算正确、错误场景下的妥协率，进而估算 d′ 与 c；对强度、引用与证据门控进行统计检验。结果显示 GPT‑6 系列在敏感度上最优；引用来源降低偏差但往往降低敏感度；强度提升在大多数模型无显著效应；证据门控仅在 Llama 上显著提升敏感度，且整体对比显示模型差异显著。

**⚠️ 局限性**

局限性：1）敏感度基于已回答一致的问题，可能与问题难度相关；2）引用真实性与答案正确性混杂，难以区分两者影响；3）行动任务仅使用单一 synthetic 场景；4）等方差假设未被检验；5）证据门控仅在回答任务中评估，未测行动任务；6）对话的部分信息缺失导致否认影响判定不确定。

---

## 436. MedJev: Evaluating Zero-Shot System-One Decision Making in Medicine

**arXiv ID:** 2610.36006 | [PDF](https://arxiv.org/pdf/2610.36006v1)

**作者:** Simon A. Lee `[一作]` (UCLA), Jeffrey N. Chiang `[通讯]` (UCLA)

**关键词:** `Computation and Language` `Classification` `Recommendation System` `Explainability and Interpretability` `Computational Efficiency` `Drug Discovery` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Mixture of Experts` `Text` `Tabular` `Biomedical Data` `Electronic Health Records` `Benchmark`

**🎯 论文内容**

开发并评估了一种 System-One 结构化预测框架 MedJev，统一处理医学知识问答、诊断和 EHR 基于概率的临床预测，并在十个医学基准上进行零射击和后置校准实验。

**💡 创新点**

提出将医学问答与临床预测统一为同一结构化请求接口，证明 System-One 能在零射击下与传统 LLM 和监督模型竞争，并展示轻量级后置校准显著提升概率可靠性。

**🔧 技术方法**

采用 OpenJev 的 System-One 接口（Choice/Noul 请求、温度/Platt 校准）、Qwen3.5、MedGemma 等大模型，结合 XGBoost 与 TF‑IDF 监督基线进行对照实验。

**📊 数据集**

使用 MedQA、USMLE、NEJM、DiagnosisArena 等医学知识基准，以及 MIMIC‑IV（住院死亡、LOS>7d、30d再入院）和 MIMIC‑IV‑ED（ED 入院、处置、2h 入院预测）等 EHR 数据集。

**📈 对比分析**

通过准确率/AUROC/ECE 等指标与公开 frontier 结果和监督模型对比；MedJev 在知识问答上接近 GPT‑4，零射击在死亡/LOS 任务与监督模型可比，校准后概率误差大幅降低，但在 ED 运营指标上监督模型更优。

**⚠️ 局限性**

局限性包括：单中心数据导致泛化受限；部分任务正例稀少导致评估不稳；基准可能与模型预训练重叠；序列化方式未优化；校准参数不跨任务迁移；模型未临床验证。

---

## 437. Evolving Agent Teams

**arXiv ID:** 2610.36848 | [PDF](https://arxiv.org/pdf/2610.36848v1)

**作者:** Shiyi Cao `[一作]` (University of California, Berkeley), Ion Stoica `[通讯]` (University of California, Berkeley)

**关键词:** `Artificial Intelligence` `Optimization` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Neural Architecture Search` `Transformer` `Large Language Model` `Agentic AI` `Retrieval-Augmented Generation` `Mixture of Experts` `Text` `Code` `Benchmark`

**🎯 论文内容**

提出了Evolving Agent Teams多代理框架，能够在GPU kernel优化过程中不断重构团队组织，逃离局部最优并持续改进性能。

**💡 创新点**

核心创新在于：① 由中心监督者诊断每一次停滞并动态调整团队角色、协调方式及知识管理；② 采用持久化文件系统记录学习文档、演化历史与团队产物，使重构具备累积经验的基础。

**🔧 技术方法**

技术实现基于Claude Code/Opus LLM代理，设计了多角色（如Profiler、Debugger、Knowledge Integrator等）与分支并行执行；监督者通过文本生成、文件读写和指令调度实现团队演化；文件系统提供共享、版本化的代码与实验记录。

**📊 数据集**

实验使用的基准包括：FlashInfer‑Bench（47 DeepSeek‑V3 生产工作负载）、FA4 B200（CuTe DSL 24个工作负载）、GDN 100个工作负载、G+AR 5个工作负载、MI350X、B200、Apple Silicon MoE、RoboDojo 42 任务以及 Flash‑KMeans benchmark 网格。

**📈 对比分析**

与 Claude Code、固定四角色团队、Codex、K‑Search、CUDAForge 等基线在相同 token 预算下进行对比。结果显示：在预算控制下，EAT 取得平均 0.74× 的速度提升（Claude Code 仅 0.44–0.51×）；在额外预算下，平均 1.21×，并在 47 个工作负载中 36 个实现加速；在其他任务（KDA prefill、MoE offload、RoboDojo）亦显著优于现有系统。

**⚠️ 局限性**

实验仅做了两次重复，单次跑的统计有限；缺乏成本‑性能 Pareto 分析；在噪声更大的领域或更大规模任务上的鲁棒性尚未验证；模型选择与预算对比的系统性分析仍待进一步研究。

---

## 438. Silent Blindness: Fault Vulnerability of Hardware Malware Detectors

**arXiv ID:** 2610.36612 | [PDF](https://arxiv.org/pdf/2610.36612v1)

**作者:** Alireza Abolhasani Zeraatkar `[一作]` (University of California Davis), Hussain Al-Asaad `[通讯]` (University of California Davis)

**关键词:** `Cryptography and Security` `Anomaly Detection` `Safty and Privacy` `Fault Injection` `AdaBoost` `Bagging` `minimax Selective Hardening` `Confidence Interval Statistical Evaluation` `Tabular`

**🎯 论文内容**

对硬件恶意软件检测器（HMD）在18种分类器、4种计数器预算及AdaBoost/Bagging集成方案下，进行结构级停留‑0/1与参数位翻转故障注入实验，评估不同故障对检测召回率与误报率的安全影响。

**💡 创新点**

提出基于安全后果（逃逸、服务拒绝、静默等）的故障分类法；系统性分析HMD设计空间的故障敏感性；证明树模型在单点故障下更稳健；展示集成与选择性硬化可显著降低最大回忆损失。

**🔧 技术方法**

结构级故障注入、硬件实现验证、AdaBoost/Bagging集成、minimax 选择性硬化算法、置信区间统计评估。

**📊 数据集**

使用约7,400实例的恶意软件/正常程序数据集（来自先前公开训练集）训练并验证所有18种分类器。

**📈 对比分析**

与无故障基线对比，采用recall与FPR变化阈值（5%）划分故障类别；比较不同分类器、计数器预算、集成方法的关键故障率；结果显示树模型5.7% vs 14.3%；Bagging将关键故障率降至0%；通过选择性硬化可将最大回忆损失从60%降至3.3%。

**⚠️ 局限性**

仅在结构层级注入，未覆盖门级或合成后故障；KNN未评估存储样本故障；硬化方案基于面积分配，未进行实物测量；实验基于已验证RTL，未验证物理实现与保护开销。

---

## 439. ExecDiffuser: Closing the Sample-to-Execution Gap in Diffusion Planning

**arXiv ID:** 2610.36837 | [PDF](https://arxiv.org/pdf/2610.36837v1)

**作者:** Shuo Pei `[一作]` (Beijing Institute of Technology), Yong Wang `[通讯]` (University of Hong Kong)

**关键词:** `Robotics` `Autonomous Driving` `Optimization` `Computational Efficiency` `Reinforcement Learning from Human Feedback` `Transformer` `Diffusion model` `Flow-based Model` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Contrastive Learning` `Retrieval-Augmented Generation` `Prompt Engineering` `Point Cloud` `Sequential` `Tabular` `Time Series`

**🎯 论文内容**

本文提出了 ExecDiffuser，一种将生成候选轨迹、轨迹选择与后续重新规划耦合在一起的循环框架，用于解决生成规划器中“样本到执行”差距的问题。

**💡 创新点**

创新点包括：
1) Intent‑Guided Diffusion Proposal（IGD）通过可学习的意图查询、责任分配与轨迹形状约束实现对生成候选的执行感知引导；
2) Execution‑Grounded Selection（EGS）利用共享的执行语义学习候选的发布价值，并通过递归上下文把已发布轨迹返回给下一循环；
3) 匹配的反事实评估协议（matched counterfactual evaluation）独立衡量候选可用性与发布质量；
4) 通过持续的执行反馈与自适应连续门（Adaptive Continuity Gate）提升规划稳定性与响应速度。

**🔧 技术方法**

核心技术包括：
- 基于扩散的多模态轨迹生成（Diffusion Planner/FlowDrive 基础）
- 语义共享的在线执行证据（drivable, collision, risk, progress 等）
- 责任分配与轨迹形状惩罚的联合损失
- 学习的候选评分网络（EGS scorer）
- 递归发布上下文与自适应时间门控
- 不确定性增强的轨迹预测与 MRM（可回滚移动）回退策略。

**📊 数据集**

使用的公开数据集有：
- nuPlan v1.1（Val14、Test14、Test14‑hard）
- InterPlan（交互强度高的长尾场景）
- NAVSIM v1.1（官方 PDMS 评估）。

**📈 对比分析**

在统一的五种随机种子复现协议下，ExecDiffuser 在 nuPlan 的六个闭环评估场景中平均得分 85.62，分别比 Diffusion Planner 和 FlowDrive 提升 4.02 和 1.31 分；在 InterPlan 上提升 55.2%；在 NAVSIM 上得到 92.6 的 PDMS 分数，超过 DiffusionDriveV2 1.4 分，整体表现位于基线之上。

**⚠️ 局限性**

局限性包括：
- 需要额外的模型参数和推理开销（+5.2M 参数，推理时间约 59.5 ms）；
- 依赖轨迹预测的准确性，预测误差在遮挡或高风险场景中会导致误判和回退失效；
- 虽然提升了发布质量，但仍有一定的 false‑valid 与 false‑reject 错误，尤其在复杂交互场景；
- 目前仅在仿真评估中验证，尚未在真实车辆上进行充分测试。

---

## 440. Split Federated Unlearning

**arXiv ID:** 2610.36465 | [PDF](https://arxiv.org/pdf/2610.36465v1)

**作者:** Jia Chen `[一作]` (Wuhan Textile University), Jun Bai `[通讯]` (Adelaide University)

**关键词:** `Machine Learning` `Federated Learning` `Optimization` `Safty and Privacy` `Convolutional Neural Network` `Transformer` `Contrastive Learning` `Supervised Fine-Tuning` `Gradient Modulation` `CKA` `MIA Evaluation` `Hierarchical Aggregation` `Double Importance Estimation` `Selective Perturbation` `Restricted Recovery` `Image`

**🎯 论文内容**

提出了分布式Split Federated Unlearning（SFU）框架，并设计了SURE方法实现近似的模型忘记；

**💡 创新点**

首次系统化刻画了SFL中跨分区耦合、历史影响传播和更新模式可行性，并通过理论与实验证明了单分区更新的局限；

**🔧 技术方法**

采用双重重要性估计、层级选择性扰动与受限恢复、分层聚合、梯度调制、CKA与MIA评估等技术；

**📊 数据集**

在MNIST、CIFAR-10、CIFAR-100与对应的LeNet、CNN、ResNet-18模型上进行实验；

**📈 对比分析**

与Fine‑Tune、FedEraser、FedRecovery、SalUn、MoDe、FedOSD、NoT、FedCARE、FedMemPrune等方法在客户端和类别忘记任务下进行对比，SURE在保持高保留性能的同时取得与模式匹配重训练差距低于2个百分点，通信/计算成本与Fine‑Tune相近；

**⚠️ 局限性**

缺点包括：尚未给出严格的忘记保证；恢复阶段依赖保留数据；在不同更新模式、分割位置和忘记场景下性能差异显著，需进一步优化。

---

## 441. RELAY: One Query, Two Roles for Sequential Recommendation

**arXiv ID:** 2610.36000 | [PDF](https://arxiv.org/pdf/2610.36000v1)

**作者:** Zesheng Li `[一作]` (University of Chinese Academy of Sciences), Honggang Qi `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 10685 | [OpenAlex ID](https://openalex.org/A5047226078)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Contrastive Learning` `Shared Attention Mechanism` `Softmax` `Log Compression` `Truncation` `Sequential`

**🎯 论文内容**

提出了RELAY模型，用单一查询实现语义读取与行为记忆路由，直接在一次前向传播中完成顺序推荐的完整排名；

**💡 创新点**

创新点在于将历史选择共享给两种任务——语义偏好表示与候选级别行为记忆路由，并用单一Catalog Likelihood联合训练，显著压缩参数并提升推理速度；

**🔧 技术方法**

使用语义索引（SID）编码、双向历史编码器、共享注意力机制、候选门控、全量词典Softmax以及对记忆计数的对数压缩和前B个邻居截断；

**📊 数据集**

在六个Amazon Review 2014/2023 数据集上评估，分别为 Sports、Beauty、Toys、Scientific、Instruments、Games；

**📈 对比分析**

与16种基线（包括OneChord、SPRINT、FAVE等）比较，RELAY在Recall@5和NDCG@5上平均提升约14%和18%，参数量减少65%，推理速度提升1.7–1.8倍，且在所有数据集均取得最高指标；

**⚠️ 局限性**

限制包括：需要预先构建固定的4码语义词典，记忆邻居预算和单查询方式可能无法完全捕捉多兴趣场景；对极大词典或极长历史的可扩展性尚未深入验证。

---

## 442. Does More Thinking Pay Off? Reasoning Cost and Correctness of LLMs in Code Generation and Bug Fixing

**arXiv ID:** 2610.36952 | [PDF](https://arxiv.org/pdf/2610.36952v1)

**作者:** Andie Huang `[一作]` (Massey University), Lin Ma `[通讯]` (Hangzhou Normal University)

**关键词:** `Software Engineering` `AI Code Assistant` `Computational Efficiency` `Explainability and Interpretability` `Transformer` `Large Language Model` `Chain-of-Thought` `Prompt Engineering` `McNemar Test` `t-test` `Holm Correction` `Text` `Code` `Benchmark`

**🎯 论文内容**

本文系统评估了大型语言模型在代码生成与 bug 修复任务中使用链式思考（推理轨迹）的收益与成本。

**💡 创新点**

创新点包括：①量化推理词数对正确率的影响；②构建六类推理行为的分类体系；③设计离线截断实验与基准；④在多模型、多任务上比较不同截断策略的效果。

**🔧 技术方法**

采用 Qwen3‑32B、QwQ‑32B、R1‑Distill‑Qwen‑32B、R1‑Distill‑Llama‑70B 等大模型，结合链式思考提示、思考标签提取、手工编码行为分类、McNemar 与 t 检验、Holm 校正等统计方法，以及 Docker 环境进行评测。

**📊 数据集**

使用 HumanEval Pro、MBPP Pro 两个代码生成基准和 BugsInPy 328 个真实 bug 进行实验。

**📈 对比分析**

通过开启/关闭推理、配对 McNemar 检验和 t 检验评估准确率差异；使用固定比例截断、重复检测、首代码段截断等策略重新生成答案，并与全推理对照；结果显示推理提升不稳定，最多截取一半推理轨迹可保持相同准确率。

**⚠️ 局限性**

局限性包括：仅针对 Python 任务；使用 4‑bit 量化模型，可能影响推理长度与准确率；bug 修复准确率低，截断实验基于存储推理，未评估实时生成时的效率与成本。

---

## 443. The Content Leash: Mitigating Degenerate Solutions in Invertible Speaker Anonymization

**arXiv ID:** 2610.36768 | [PDF](https://arxiv.org/pdf/2610.36768v1)

**作者:** Nirjon Mobarak Talukder `[一作]` (Islamic University of Technology), Abu Raihan Mostofa Kamal `[通讯]` (Islamic University of Technology)

**关键词:** `Sound` `Safty and Privacy` `Knowledge Distillation` `Representation Learning` `Transformer` `Convolutional Neural Network` `Flow-based Model` `Contrastive Learning` `Knowledge Distillation` `Audio`

**🎯 论文内容**

对可逆说话者匿名化方法进行重实现，并通过在训练目标中加入 anchor 与 range 损失来避免退化，利用 ECAPA‑TDNN 的知识蒸馏构建 mel‑域验证器，并在统一评估协议下展示了显著提升的隐私-可用性平衡。

**💡 创新点**

① 发现原训练目标易导致声学内容被严重破坏的退化解；② 提出 anchor（对齐）与 range（动态范围）约束来限制匿名化后谱图的变化；③ 通过把 ECAPA‑TDNN 迁移到 mel‑域实现验证器泛化；④ 采用统一评估框架（测量 floor、held‑out 验证器、多种 seed、仅未见说话者）提升实验可信度。

**🔧 技术方法**

使用可逆条件可逆网络（cINN）结合 triplet loss、anchor loss、range loss、log‑determinant 正则；知识蒸馏（similarity preserving）将 ECAPA‑TDNN 迁移到 mel‑域；HiFi‑GAN vocoder；Whisper ASR 计算 WER；ECAPA‑TDNN 与 ResNet 说话人验证器进行 EER 评估。

**📊 数据集**

训练集：VCTK 与 LibriTTS 子集（按 85/15 划分验证）；评估集：LibriTTS 38 个未见说话者共 303 句（每说话者最多 8 句）。Vocoder 采用在 VCTK 上 fine‑tuned 的 HiFi‑GAN。

**📈 对比分析**

与 McAdams 基线和原始 Rano 方法直接对比；对 38 未见说话者，EER 从 2.8% 提升到 26.3%（≈9×提升），WER 约 11.8%（相对 floor 2.5%），恢复相似度 94.9% 与未改 wav 相同；held‑out ResNet 的 EER 约 27.9%，证明泛化效果；与原 Rano 比较，EER 更低、恢复相似度更高，尽管 WER 近似相同但相对 floor 更高。

**⚠️ 局限性**

① 蒸馏后 mel‑域验证器与原 ECAPA‑TDNN 的相似度仅为 0.73，隐私提升有限；② 评估仅覆盖单一语料库、单一攻击模型（无知情攻击），未检验 pitch/voice distinctiveness 等 VoicePrivacy 评测指标；③ 恢复依赖保存匿名谱图，实际部署需 lossless 存储；④ 仅使用一组 vocoder，未全面验证对 vocoder 变化的鲁棒性；⑤ 仅对 anchor 与 range 损失同时进行 ablation，未单独分析两者贡献。

---

## 444. The Exact Weight Spectrum of $\RM(m-7,m$): A Counterexample to the Carlet-Solé Conjecture

**arXiv ID:** 2610.36687 | [PDF](https://arxiv.org/pdf/2610.36687v1)

**作者:** Yueying Lou `[一作]` (Nanjing Normal University), Qichun Wang `[通讯]` (Nanjing Normal University)

**关键词:** `Information Theory` `Optimization` `Computational Efficiency` `Representation Learning` `Federated Learning` `Explainability and Interpretability` `Knowledge Distillation` `Reinforcement Learning from Human Feedback` `Neural Architecture Search` `Protein Structure Prediction` `Fourier Transform` `Contrastive Learning` `Optimization` `Computational Efficiency` `Representation Learning` `Knowledge Distillation` `Reinforcement Learning from Human Feedback` `Neural Architecture Search` `Protein Structure Prediction`

**🎯 论文内容**

本文研究了 Reed–Muller 码 (m-7, m) 的 Hamming 重量光谱，并给出了完整光谱，排除了先前未确定的八个重量值，进而构造出对 Carlet‑Solé 猜想的反例。

**💡 创新点**

创新点在于：①首次完成 (m-7, m) 的重量光谱确定；②通过仿射几何与傅里叶分析相结合的全新方法，排除所有低重量情况；③提供了 Carlet‑Solé 猜想在 c=7 的完整反例。

**🔧 技术方法**

主要技术手段包括：仿射几何覆盖与约束分析、Fourier 变换与傅里叶系数正交性、标准形式与结构定理、几何转移（维度提升）等。

**📊 数据集**

本文不使用实验数据集，所有结果均为理论推导与严格证明。

**📈 对比分析**

与先前已知的 (n-5,n)、(n-6,n) 结果相比，本文成功扩展到更高阶，完整确定光谱；相较于已有的猜想与部分计算结果，本文提供了完全的解析解，性能上可谓“完全正确”。

**⚠️ 局限性**

局限性：①仅针对 (m-7,m) 进行了完整分析，对 (n-8,n) 等更大缺口的情况仍未解决；②当前的结构定理仅适用于 3·2ⁿ‑r 范围，未覆盖更大重量；③对非偶重量的细节分析仍需进一步研究。

---

## 445. AxisQ: Depth-Adaptive Channel--Frequency Quantization for Camouflaged Object Detection

**arXiv ID:** 2610.36844 | [PDF](https://arxiv.org/pdf/2610.36844v1)

**作者:** Chi Ruan `[一作]` (SpinEntropy), Yun Liu `[通讯]` (Nankai University)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Transformer` `Contrastive Learning` `Auto Encoder` `Supervised Fine-Tuning` `Image` `Benchmark`

**🎯 论文内容**

提出一种深度自适应混合域的后训练量化框架，用于提高伪装目标检测模型在低比特宽度下的预测质量。

**💡 创新点**

创新点在于：①根据特征深度动态切换量化分配域——浅层采用通道域分配，深层采用局部频域分配；②引入伪装目标检测专用的 COD‑GGN 任务感知成本与边界加权 Hessian 重构；③利用标签辅助 Fisher 估计和逆水填充实现精准的位宽分配。

**🔧 技术方法**

采用的技术包括：后训练量化 (PTQ)、通道级 Fisher 成本估计、块级 2‑D DCT 频域分配、COD‑GGN 曲率近似、边界加权 Hessian 重构、逆水填充与精度预算校正。

**📊 数据集**

实验使用四大公开伪装目标检测基准数据集：CAMO、CHAMELEON、COD10K 与 NC4K。

**📈 对比分析**

与多种 PTQ 与重构方法（LS‑ViT、BRECQ、AdaRound+QDrop、FIMA‑Q、APHQ‑ViT、RepQ‑ViT、COD‑TDQ）在 CFRN 与 CamoFormer‑P 上进行对比，结果显示在 2‑bit 激活量化下，平均加权 F‑measure 分别提升 12.7% 与 33.3%，在更高比特宽度（W4A4）同样保持领先或相当性能。

**⚠️ 局限性**

局限性包括：需预先划分浅/深阶段并手动设置阶段分割；重构与 Hessian 计算在训练时仍需较大算力；方法主要针对 Transformer‑based COD 模型，泛化至其他网络结构或任务尚未验证。

---

## 446. Edhivo: An AI-Native Platform for Scalable Assessment and Instant Feedback in Higher Education

**arXiv ID:** 2610.36654 | [PDF](https://arxiv.org/pdf/2610.36654v1)

**作者:** Moraish Kapoor `[一作]` (Stony Brook University), Jiawei Zhou `[通讯]` (Stony Brook University)

**关键词:** `Computers and Society` `AI Code Assistant` `Education` `Computational Efficiency` `Explainability and Interpretability` `Transformer` `Large Language Model` `Chain-of-Thought` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Review/Survey Paper`

**🎯 论文内容**

开发了 Edhivo，一套 AI 原生的教学平台，用于自动生成评估题目并对学生作答进行语义化评分，以弥补高等教育中评估生成与反馈的瓶颈。

**💡 创新点**

创新点在于双 LLM 管道：一条 LLM 负责基于上传课程内容生成多选题和开放式题目，另一条 LLM 依据教师设定的 rubrics 进行多维度（正确性、完整性、相关性）评分，且所有 AI 交互均以源材料为锚定，配合人机协同审查。

**🔧 技术方法**

采用 Gemini 3 Flash 进行题目生成、GLM 4.7 Flash 进行评估，配合 AI Orchestration Layer、结构化 JSON 提示、源材料 grounding、Chain‑of‑Thought 评分流程和 HITL 审核。

**📊 数据集**

使用的主要数据集为 AMS 691.01 课程的教材、讲义、转录等上传材料，以及 36 名研究生在 2026 年春季学期的 10 题评估与答案；学生答卷与教师评分被收集用于实验。

**📈 对比分析**

通过与人工评估的时间与成本对比，自动生成评估从约 30 min 降至 5.2 min，节省约 25 min；自动评分与人工评分（含 2.5 min/学生审核）相比，200 名学生可节省 8.3 h；每 200 学生评估成本 <$0.55；学生调查显示 100% 认为界面易用且即时反馈提升学习体验。

**⚠️ 局限性**

局限性包括缺乏监考保障、可能存在语言偏见（LLM 写作风格易被高评）、仅验证 STEM 领域客观答案的有效性、样本规模有限（36 名学生，单一课程），且作者时间节省估算基于单个教师的非实验测量。

---

## 447. Dirichlet Process Mixtures within Semiparametric Bayesian Networks: A Novel Conditional Density Estimation Approach

**arXiv ID:** 2610.36110 | [PDF](https://arxiv.org/pdf/2610.36110v1)

**作者:** Victor Alejandre `[一作]` (Universidad Politécnica de Madrid), Concha Bielza `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `Machine Learning` `Computational Efficiency` `Data-Centric Learning` `Representation Learning` `Dirichlet Process Mixture Model` `Variational Inference` `Score-based Model` `Tabular` `Benchmark`

**🎯 论文内容**

提出了一种基于Dirichlet过程混合模型（DPM）的新型条件概率分布（CPD）建模方法，并将其集成到半参数贝叶斯网络（SPBN）框架中。

**💡 创新点**

创新点在于通过DPM实现了对连续变量的非参数建模，提供了更灵活和高效的密度估计，同时降低了推理成本。

**🔧 技术方法**

使用了Dirichlet过程混合模型（DPM）和变分推理方法（CAVI、GPCAVI和Memoized）来进行参数学习和结构学习。

**📊 数据集**

在合成数据和真实数据集上进行了评估，合成数据集用于控制数据生成过程，真实数据集来自UCI机器学习库。

**📈 对比分析**

与基于核密度估计（KDE）的SPBN和其他最先进的贝叶斯网络模型相比，DPM-SPBN在密度估计和计算要求上表现显著优越，尤其在合成数据上实现了更低的绝对对数似然误差。

**⚠️ 局限性**

限制在于DPM的推理过程可能在计算上较为复杂，尤其是在处理大规模数据集时，Memoized方法的计算成本随着样本大小和节点数量的增加而显著上升。

---

## 448. MARQ: A Modular, LLM-Agnostic Requirement Quality Assessor Infused with Systems Engineering Knowledge

**arXiv ID:** 2610.36234 | [PDF](https://arxiv.org/pdf/2610.36234v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## 449. CohortFlow: A Steerable Clinical Cohort Curation Agent with Localized Semantic Correction

**arXiv ID:** 2610.36100 | [PDF](https://arxiv.org/pdf/2610.36100v1)

**作者:** Neil Chen `[一作]` (University of California Berkeley), Yang Yang `[通讯]` (University of California San Francisco)

**关键词:** `Artificial Intelligence` `Data-Centric Learning` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Agentic AI` `Text` `Tabular` `Biomedical Data` `Electronic Health Records`

**🎯 论文内容**

开发了一个可驱动的临床队列抽取代理，支持在执行过程中根据自然语言更正局部更新队列规范，而不需要完全重生成。

**💡 创新点**

创新点在于将更正视为增量任务（Task Delta），利用语义指纹在执行图中只失效受影响节点，从而保留未变计算结果，实现语义级的修正而非时间顺序级的重启。

**🔧 技术方法**

采用LLM（Qwen3-8B）进行自然语言解析与增量更新，构建执行有向无环图（DAG）与语义指纹机制；结合结构化数据库查询与混合检索（BM25/嵌入）评估临床笔记；实现顺序与并行修正调度。

**📊 数据集**

使用CohortX Task 1训练集构造100个队列情境，并在MIMIC-IV数据库上进行概念解析与查询验证。

**📈 对比分析**

与传统全重生成基线比较：语义准确度微F1 77.0% vs 76.4%（无统计显著差异），但后续修正延迟下降64.8%、运行中44.3%，LLM调用减少38.1%；后端执行验证13个状态全部成功，展示局部重用效果。

**⚠️ 局限性**

局限性包括：布尔逻辑准确性与澄清处理不足；仅评估结构化层面，未覆盖临床笔记检索与推理；使用单一8B模型与量化，可能限制性能；基准修正为人工构造而非真实临床交互。

---

## 450. Right Click, Wrong Box: Grading GUI Grounding by What a Click Does

**arXiv ID:** 2610.36831 | [PDF](https://arxiv.org/pdf/2610.36831v1)

**作者:** Xiaokang Yuan `[一作]` (Zhejiang University), Zeng Wang `[通讯]` (Zhejiang Geely Holding Group Co., Ltd)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Reinforcement Learning from Human Feedback` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Reinforcement Learning` `Prompt Engineering` `Vision Language Model` `Contrastive Learning` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Neural Radiance Field` `Auto Encoder` `Generative Adversarial Network` `Image` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

本文提出了以点击结果为准的 GUI 定位正误判定（outcome‑defined correctness），并实现了一种可以在网页上执行点击、记录结果指纹的探针（probe），利用它生成“结果映射”（outcome maps）并对模型预测进行执行判断；随后对两款自托管 Web 应用（Gitea 与 Django Oscar）以及公开截图基准进行盲审，量化 box scoring 下被拒绝的点击中实际正确的比例；在此基础上分析了评测误差来源（训练种子、样本抽样、评分规则）并给出改进建议；最后在 3B 模型上对 box reward 与 outcome reward 的训练效果进行对比实验。

**💡 创新点**

创新点包括：
1) 以点击导致的功能结果为标准，重新定义 GUI grounding 的正确性；
2) 开发可在 Chromium 中对每个像素点执行点击并捕获服务器状态变化的探针，生成结果映射；
3) 通过执行判断和盲审揭示 box scoring 对多重等价元素的系统性低估；
4) 将 bias（评分规则）与 noise（种子、样本）分离，提出错误预算分析；
5) 推荐在评测中同时报告 box accuracy 与 outcome accuracy，并对 seed 与 paired interval 做标准化。

**🔧 技术方法**

主要技术手段：
- 基于 Chromium 的点击执行探针，捕获导航、请求、对话框、可视变化等多种“签名”作为结果；
- 结果映射（outcome map）构建，覆盖每个交互元素的 3×3 网格及边界外点；
- 执行判断（execution judge）与盲审（human audit）相结合的评价流程；
- 统计方法：配对 Bootstrap 区间、Cohen's κ、Wilson 区间、置信区间；
- 强化学习训练（GRPO）与 LoRA 微调，比较 box reward 与 outcome reward 的效果。

**📊 数据集**

使用的数据集与模型：
- 两个自托管 Web 应用：Gitea 1.25（94 页）与 Django Oscar 4.2（60 页）；
- 公共截图基准（ScreenSpot、ScreenSpot‑Pro、OSWorld‑G、. 等）；
- 11 种 grounding 模型（包括 7–8B 的 UI‑Venus‑1.5‑8B、GTA1‑7B、Qwen3‑VL‑8B 及 3B 级别的 Qwen2.5‑VL‑3B 与 8 个 GRPO‑训练的 3B 变体）；
- 通过探针执行的约 1,000 次点击在 Gitea 与约 700 次在 Oscar 上；
- 公开截图中的 4,838 步骤用于评估。

**📈 对比分析**

比较方法与性能：
- 将 box scoring 与 outcome scoring 在两款应用上的准确率对比，发现后者提升 2–3 分；
- 盲审显示被 box scoring 拒绝的点击中约 6–14% 在执行上是正确的，Oscar 上高达 34%；
- 结果映射在 Gitea 中 17.7% 目标与 box 不符，Oscar 31.2%；
- 对比 7–8B 与 3B 模型，强模型的误差中有 20% 为 outcome‑correct；
- 在 3B 模型训练实验中，box reward 与 outcome reward 的差异在 0.25 分以内，说明额外奖励对性能提升有限；
- 通过种子与抽样的 paired interval，证明 1 分差距在统计上可被解释为噪声或评分偏差。

**⚠️ 局限性**

局限性：
- 结果映射仅在单一界面状态下定义，无法处理多步交互或动态变化的 UI；
- 探针工作在 1280×720 分辨率，可能漏掉高分辨率下的细节；
- 评测主要聚焦于 Web 接口，原生移动/桌面 UI 的等价性较低，适用性有限；
- 训练实验仅在 3B 规模上进行，未验证更大模型或更长训练周期的效果；
- 盲审依赖人工判断，仍可能存在主观误判；
- 只对已注释的框进行评估，未对注释错误本身做进一步修正。

---

## 451. RubricAtlas: Mining Type-Level Rubrics from Expert Reports to Evaluate Deep Research Agents

**arXiv ID:** 2610.36878 | [PDF](https://arxiv.org/pdf/2610.36878v1)

**作者:** Delin Wang `[一作]` (StepFun), Zuo Bai `[通讯]` (StepFun)

**关键词:** `Artificial Intelligence` `Finance Related` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Chain-of-Thought` `Text`

**🎯 论文内容**

本文设计并实现了RubricAtlas，一套基于专家报告挖掘的类型级评估框架，用于评价深度研究代理生成的Web搜索与综合报告。

**💡 创新点**

创新点包括：①从不同对象的专家报告中抽取共性分析问题，形成可复用的类型级需求；②将评估结构化为问题–目标–检查点层级，既评估内容完整性又能追溯证据来源；③将完整性评估与论证有效性评估分离，避免混淆；④通过参考报告选择实验揭示评估对基准敏感性，并用删除/填充实验验证评估的因果性。

**🔧 技术方法**

技术手段主要包括：大型语言模型（LLM）对报告进行结构化分解（维度→问题→结论树）；文本检索与对齐；二元检查点与负条件评估；统计一致性指标（Cohen κ、Kendall τ）和置信区间估计；以及自动化的评估流水线（ReAct、Serper、Jina等）。

**📊 数据集**

使用的数据集为2026年1–9月中国金融研究机构发布的中文财务研究报告，共计数百份，涵盖多行业、多体裁（公司覆盖、收益评估、行业深度、策略报告），构成RubricAtlas中的研究类型、问题、目标与检查点。

**📈 对比分析**

比较方法：在相同任务集上评估10个深度研究代理的完整性得分和层级得分；评估器间一致性为κ≈0.4–0.5，Kendall τ≈0.5；参考报告选择实验显示覆盖率提升3–5分，排名相关性≈0.6；删除实验验证缺失分析对得分影响显著；最终Claude在两位评估器下均表现最佳，其他代理间差距明显。

**⚠️ 局限性**

局限性：①资源仅覆盖中文金融领域，缺乏跨语言、跨领域验证；②缺乏独立专家对目标层级适用性的全面审核；③完整性与论证有效性评估未完全整合，存在评估耦合不明显的问题；④实验中未完全分离检索、生成、评估等环节对结果的影响；⑤参考报告选择实验基于冻结Atlas，未检验单报生成的可行性与泛化能力。

---

## 452. Tight Characterizations of AMM Mechanism Design under Weak Efficiency

**arXiv ID:** 2610.36635 | [PDF](https://arxiv.org/pdf/2610.36635v1)

**作者:** Chengkai Zhang `[一作]` (Rutgers University), Yuhao Li `[通讯]` (Columbia University)

**关键词:** `Computer Science and Game Theory` `Federated Learning` `Optimization` `Safty and Privacy` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `Finance Related`

**🎯 论文内容**

论文研究在没有公平排序（plain consensus）的区块链环境下，自动做市商（AMM）机制能否同时满足激励兼容（IC）和弱局部效率（wLE），并证明在确定性匿名机制下两者不可兼得；随后在三种受限策略空间（无Sybil、无预算过报、公开价值）下给出可实现的机制。

**💡 创新点**

创新点在于：①首次在去中心化机制设计中证明在普通共识（仅具审查抵抗）下IC与wLE的不可兼容性，形成公平排序与审查抵抗之间的分离；②提出一种通用的两侧匹配构造，将原问题拆分为单侧分配；③针对三类受限策略空间分别设计了手握机制、比例手握机制和分层机制，展示在相应约束下可以实现IC、wLE、价格曲线一致性、匿名性等目标。

**🔧 技术方法**

主要技术包括：构造性证明（利用Sybil、预算过报、价值误报三种策略）来展示不可能性；利用凸性、单调性以及价格曲线的可逆性推导订单匹配与支付规则；设计基于“活跃集合”等动态分配算法（等份手握、比例手握）；在分层机制中使用公共价值层次分配与平均补偿曲线。

**📊 数据集**

无数据集，纯理论分析与证明。

**📈 对比分析**

方法比较：文献中先前的公平排序机制已实现IC与wLE，但需公平排序；本文在不公平排序下证明不可能；在受限策略空间下给出的机制通过理论证明满足所有目标。性能（即满足的属性）在理论上达到：IC、wLE、价格曲线一致、匿名、个体理性。没有实验或数值比较。

**⚠️ 局限性**

局限性：①结果仅适用于确定性匿名机制；若允许随机化、可见身份或放宽匿名性，可能出现可行方案；②仅在两资产AMM下讨论；多资产情况未知；③对真实链上实现的计算复杂度和可扩展性未评估；④受限策略空间下的机制在实际用户行为中是否足够鲁棒仍待实验验证。

---

## 453. Do bots feel bad? Assessing empirical evidence for functional valence in LLMs

**arXiv ID:** 2610.36962 | [PDF](https://arxiv.org/pdf/2610.36962v1)

**作者:** Iwan Williams `[一作]` (University of Copenhagen), Iwan Williams `[通讯]` (University of Copenhagen)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Reinforcement Learning` `Contrastive Learning` `Prompt Engineering` `Retrieval-Augmented Generation` `Multimodal Dialog` `Audio-Visual Contrastive Learning` `Text` `Audio` `Image` `Multimodality`

**🎯 论文内容**

评估并批判当前关于LLMs功能性情绪/情感状态的三项实证研究。

**💡 创新点**

提出细薄与厚层次的评估标准，并指出现有研究在阈值、动机、共同货币和自我归属等方面的不足。

**🔧 技术方法**

使用情感向量提取、线性探针、Thurstonian模型、强化学习和多模态对话/图像/音频对比实验等技术。

**📊 数据集**

引用了Sofroniew的171种情感故事集、Ren的56个模型的多模态对话/图像/音频数据集，以及Han的迷宫任务、MMLU、GSM8K等。

**📈 对比分析**

通过对内部状态激活、触发条件一致性、激励效应和共同货币特征的对比，发现这些研究在满足更严格阈值时表现不一致，未能给出统一的性能评估。

**⚠️ 局限性**

主要局限在于缺乏对动机、共同货币、主观性和角色扮演等关键特征的实验验证，结果可能被信念或偏好等非情绪机制解释，因而尚未能确立LLMs具备功能性情绪。

---

## 454. MUSE: Revealing Responsiveness Gaps of LLM Tutors on Multiple-Solution Programming Problems

**arXiv ID:** 2610.36367 | [PDF](https://arxiv.org/pdf/2610.36367v1)

**作者:** Sky Cen `[一作]` (University of Central Florida), Shashank Sonkar `[通讯]` (University of Central Florida)

**关键词:** `Computers and Society` `AI Code Assistant` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Data Synthesis` `Text`

**🎯 论文内容**

构建了多解问题的模拟环境，并对LLM辅导员的解法偏好、响应性及过渡行为进行系统评估；通过该环境实现学生解法的可控模拟与教师回应的自动分类。

**💡 创新点**

首次提出针对多解编程题的可控模拟框架和评价指标，揭示LLM辅导员普遍偏向粗暴方案、缺乏对学生方案的响应，并通过针对性的监督微调显著提升响应和过渡质量。

**🔧 技术方法**

使用大型语言模型（GPT‑OSS、Qwen3.8‑27B、Llama‑3.3‑70B、Gemma‑4‑12B），采用人工合成数据生成器创建示范对话，随后在这些示范上进行监督微调（SFT），并在推理时提供可选解法列表作为上下文。

**📊 数据集**

以NeetCode 150集合为基础，挑选150道编程题，每题提供三种有效解法（暴力、次优、最优），并用GPT‑OSS生成正确与错误直觉，构成训练与测试集。

**📈 对比分析**

通过对比基线模型与微调后模型在解决方案支持率（SSR）和后解决过渡得分（PTS）上的表现；微调后SSR从约50%提升至约60%（增幅9–18个百分点），PTS从0.34提升至0.57（增幅≈0.23），并显著减少引导至次优方案的比例。

**⚠️ 局限性**

局限性在于仅尝试了有限的微调配置；数据集聚焦于面向面试的编程题，缺乏对其他多解领域（如数学证明、物理问题）的验证，可能影响结果的泛化性。

---

## 455. Gaussian Process Bandits for Constraint Black Box Lever Optimization

**arXiv ID:** 2610.36935 | [PDF](https://arxiv.org/pdf/2610.36935v1)

**作者:** Bram van den Akker `[一作]` (Booking.com), Anastasiia Kornilova `[通讯]` (Booking.com)

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning` `Gaussian Process Bandit` `Reinforcement Learning` `Time Series` `Tabular` `Benchmark`

**🎯 论文内容**

研究了在受限预算、延迟聚合反馈和操作约束下的离散预算分配权重优化问题，提出基于Gaussian Process Upper Confidence Bound (GP‑UCB) 的分段无关算法并在仿真与真实生产环境中验证。

**💡 创新点**

关键创新在于将复合Matérn+Linear核用于GP‑UCB，以在高度非平稳、噪声和竞争对手反馈的情境下实现鲁棒探索；同时构建高保真仿真环境用于先期评估与自动核选择。

**🔧 技术方法**

使用Gaussian Process Bandit（GP‑UCB）算法、复合核（Matérn + Linear）、RBF/Linear等核比较、ε‑greedy基线、仿真器、随机对照试验（RCT）等技术。

**📊 数据集**

采用自行生成的高保真仿真奖励景观（由高斯抛物点构造的非单调曲线），以及Booking.com 生产线上6周的真实分配数据进行评估。

**📈 对比分析**

与ε‑greedy、RBF/Matérn单核以及常数/最优基线对比，GP‑UCB（Matérn+Linear）在四种环境下累计回报均优于ε‑greedy，尤其在噪声、漂移和反应式环境中实现15–20%的累积回报降低，线上RCT亦显著提升效率。

**⚠️ 局限性**

受限于仅使用单一时间序列的仿真与实验、缺乏上下文特征、未探索真正对抗性博弈与高维上下文环境，且仅在有限预算与操作约束下验证，未来需扩展至更大规模多段、时变及上下文驱动的场景。

---

## 456. HeteGCD: Statistics-Aware Dual-Prototype Learning for Generalized Category Discovery on Heterophilous Graphs

**arXiv ID:** 2610.36696 | [PDF](https://arxiv.org/pdf/2610.36696v1)

**作者:** Xuanzhi Xi `[一作]` (University of Electronic Science and Technology of China), Junming Shao `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `Machine Learning` `Representation Learning` `Anomaly Detection` `Graph Neural Network` `Contrastive Learning` `Graph`

**🎯 论文内容**

研究在异构图上进行通用类别发现，提出 HeteGCD 方法

**💡 创新点**

创新点在于将节点属性与局部属性差异统计融合到嵌入空间，并在概率空间用邻域类别分布构建分布原型，从而实现双原型学习以支持未知类别的发现

**🔧 技术方法**

使用 gated dual‑stream encoder、dual‑prototype classifier、对比学习、伪标签蒸馏、均值熵正则化以及基于软分配的分布原型构建等技术

**📊 数据集**

在五个异构图数据集上进行评估：Roman Empire、Chameleon、Actor、BlogCatalog、Flickr

**📈 对比分析**

与 24 种基线组合（8 编码器 × 3 发现框架）对比，HeteGCD 在 All 和 New 准确率上名列前茅，平均 All 排名 3.8，New 排名 5.4，表现优于大多数方法

**⚠️ 局限性**

局限性包括：仍需要已知类别标签作为监督；对超参数敏感；在某些数据集上性能提升有限；未针对动态或大规模图进行专门优化

---

## 457. AdaGround: Token-Normalized Adaptive Zooming with Verifier-in-the-Loop Refinement for GUI Grounding

**arXiv ID:** 2610.36874 | [PDF](https://arxiv.org/pdf/2610.36874v1)

**作者:** Bingchao Wang `[一作]` (Shanghai Jiao Tong University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `Computer Vision and Pattern Recognition` `Object Detection` `Computational Efficiency` `Explainability and Interpretability` `Vision Language Model` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出了一种训练‑free 的 GUI grounding 框架 AdaGround，结合自适应缩放、验证器驱动的迭代重定位与结构化目标组合，以提升在高分辨率屏幕上定位小目标的准确率。

**💡 创新点**

创新点包括：① Token‑Normalized Adaptive Zooming（TNZ）—根据模型自报的目标尺寸动态决定放大倍数、视口宽度与迭代深度；② Verifier‑in‑the‑Loop Refinement（VLR）—用一个 Yes/No 验证器来接受/拒绝缩放结果、选择最佳缩放链、触发重定位与有条件的放弃；③ Compositional Structure Grounding（CSG）—通过先定位列/行标题再合成单元格来处理结构化目标；④ 将多种缩放链通过验证器仲裁而非投票，实现更稳健的结果组合。

**🔧 技术方法**

技术手段主要是：1）利用 Qwen3‑VL‑8B/32B 等大型 VLM 作为黑盒；2）从模型输出的粗略定位与边框估计计算目标 token span；3）按 token span 自适应设定缩放倍率、裁剪尺寸与最大迭代次数；4）构造验证窗口并以模型第一 token 概率生成 Yes/No 反馈；5）设计链组合与重定位策略；6）对缺失目标做存在查询并做阈值判定。

**📊 数据集**

使用的数据集：OSWorld‑G（含 54 个无目标指令），ScreenSpot‑Pro（1,581 条指令，目标尺寸均较小），ScreenSpot‑v2（1,272 条指令，目标尺寸较大）以及 held‑out UI‑Vision（5,479 条指令）。

**📈 对比分析**

与 MVP、ZoomClick、单通道单 pass 以及多种已训练的 GUI 模型进行对比。AdaGround 在 OSWorld‑G 上平均提升 3–7 分，在 ScreenSpot‑Pro 上提升 0.9–1.9 分，且在所有公开基准上超过 MVP 并与 72B 级训练模型竞争；在 UI‑Vision 上与 MVP 相当或更优。实验采用配对检验，差异均具有统计显著性。

**⚠️ 局限性**

局限性：① 仅针对 Qwen3‑VL‑8B/32B 这两种 backbone 评估，未验证跨模型泛化；② 计算成本比单 pass 低 3–4 倍；③ 需要模型返回边框才能启用 TNZ；④ 验证器若给出误判，可能导致错误答案被保留；⑤ CSG 目前只支持表格单元格，无法处理滑块或文本光标等其它结构化目标；⑥ 不适用于多步骤交互式任务，只是单步定位。

---

## 458. From Voiceprints to User Entities: Online Speaker-Memory Construction for Multi-User Spoken Agents

**arXiv ID:** 2610.36713 | [PDF](https://arxiv.org/pdf/2610.36713v1)

**作者:** Yucong Zhang `[一作]` (Chinese University of Hong Kong), Ming Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `Sound` `Recognition` `Representation Learning` `Transformer` `Recurrent Neural Network` `Large Language Model` `Contrastive Learning` `Retrieval-Augmented Generation` `Prompt Engineering` `Audio` `Text` `Benchmark`

**🎯 论文内容**

本文提出了在线说话人-记忆构造框架 EvoSpeakerMem，结合声纹观察、实体创建/链接与记忆写入，实现持续的个性化语音交互。

**💡 创新点**

创新点在于把声纹视为观测而非身份，将在线说话人识别与记忆写入耦合；提出训练无关的 gated 更新策略和全样本 KNN 维护；构造 LoCoMo‑ID 公开语音记忆基准，覆盖隐藏身份、条件漂移与四人交错干扰。

**🔧 技术方法**

技术包括 ECAPA‑TDNN 声纹编码、语音 LLM 转写、可替换文本记忆后端、基于余弦相似度的路由与门控更新、Parakeet 语音分段以及 Qwen3‑TTS 语音合成。

**📊 数据集**

数据集为 LoCoMo‑ID（30个清洁+30个条件漂移、5,958 语音问答），SMIIP‑TV（373 说话人多时长录音）以及 MSU‑Bench 用于验证任务。

**📈 对比分析**

与固定模板、全更新、gate‑update、all‑exemplar KNN 四种策略比较；在 LoCoMo‑ID 上 identity‑gated QA 从 71.41% 提升至 75.27%（清洁）或 60.15% 提升至 72.40%（漂移）；SMIIP‑TV 上固定模板真探测率随时间差增大从 29.8% 提升到 55.6%，验证表明在线维护能显著减少碎片化，但仍有交叉污染风险。

**⚠️ 局限性**

局限包括：LoCoMo‑ID 使用合成语音、漂移仅为模拟环境；四人干扰仅为交错故事，未覆盖自然多说话人对话；未提供真实世界噪声和重叠语音；多样本 KNN 没有限制记忆增长；评估未覆盖正式隐私认证与安全机制。

---

## 459. MAREval: A Multi-Agent Framework for Evaluating Natural Language Recommendation Explanations

**arXiv ID:** 2610.36389 | [PDF](https://arxiv.org/pdf/2610.36389v1)

**作者:** Reza Yousefi Maragheh `[一作]` (Walmart Global Tech), Sushant Kumar `[通讯]` (Walmart Global Tech)

**关键词:** `Artificial Intelligence` `Recommendation System` `Explainability and Interpretability` `Transformer` `Large Language Model` `Chain-of-Thought` `Prompt Engineering` `Agentic AI` `Monte Carlo Sampling` `Text`

**🎯 论文内容**

设计并实现了MAREval，一个结构化多代理框架，用于自动评估推荐系统中的自然语言解释，提升用户信任与透明度。

**💡 创新点**

创新点包括：①Chain of Debate (CoD) 先行规划多代理任务，②引入Moderator对话调节减少prompt漂移与噪声，③Arbitrator结合Monte Carlo采样实现结果稳健且与人类评估高度一致。

**🔧 技术方法**

技术手段：大语言模型（如GPT‑3.5‑turbo）作为评估器；链式思考、协同辩论、对话调节、投票仲裁、Monte Carlo采样；统计评估指标（Cohen's κ、Pearson ρ、Kendall τ）。

**📊 数据集**

数据集：公开的 TopicalChat 对话评估集；自有电商推荐解释集（108 条解释，人工标注）；以及 300,000 条商品解释用于线上 A/B 测试。

**📈 对比分析**

与 Vanilla、Chain‑of‑Thought、Self‑Refine、G‑Eval、ChatEval 等基线对比，MAREval 在 TopicalChat 和电商数据上均取得最高相关系数，标准差最低；线上 A/B 显著提升 ATC 1.18% 与 GMV 1.36%。

**⚠️ 局限性**

局限性：需要多轮 LLM 运行导致延迟增加；依赖 GPT‑3.5‑turbo，可能存在成本与偏见；对极端稀有或非结构化解释的鲁棒性未知；缺乏多语言或跨文化评估。

---

## 460. Verifiable Efficiency Limits of Innate LLM Planning in Sliding Puzzles

**arXiv ID:** 2610.36415 | [PDF](https://arxiv.org/pdf/2610.36415v1)

**作者:** Yezhou Cheng `[一作]` (Uber, Inc.), Charles Ling `[通讯]` (Western University)

**关键词:** `Artificial Intelligence` `Computational Efficiency` `Explainability and Interpretability` `Optimization` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Prompt Engineering` `Chain-of-Thought` `Bootstrap` `Trajectory Replaying` `Graph Distance Heuristic` `Optimal Certificate` `Text` `Benchmark`

**🎯 论文内容**

提出了一个可验证的完整计划评估框架，将成功加权计划效率拆解为执行可靠性、规划协调性和行动经济性，并在4×4滑动拼图上对多种语言模型的完整计划进行评估。

**💡 创新点**

创新点在于将计划效率分解为三层可验证度量，保留所有失败尝试；使用已证明最优路径作为基准；通过精确控制时间步长和障碍数量，对长时程规划的限制进行定量分析。

**🔧 技术方法**

采用轨迹重放验证、图距离启发式 h(s) 计算、最优证书、Bootstrap 置信区间等技术，对模型输出进行逐步验证与评估。

**📊 数据集**

使用了两套棋盘数据集：主集 320 个棋盘（k=0~3，d*=10,20,30,40），以及 80 个压力集棋盘（k=2,3，d*=50,60），共 8400 条完整计划输出。

**📈 对比分析**

通过比较不同模型（Astra、Fable 5、Opus 5.5、GPT‑5.6 Sol、Claude Sonnet 5.5、GPT‑5.6 Luna、Claude Haiku 4.5）在 R（执行可靠性）、Q（规划协调性）、E（行动经济性）和 SPE（整体效率）上的 95% 置信区间。结果显示 Astra、Fable 5 与 Opus 5.5 在所有指标上表现完美（R=Q=1），SPE 最高；Sol 与 Sonnet 5.5 在执行可靠性和协调性均下降；Luna 与 Haiku 虽有略高的经济性，但协调性不足导致 SPE 中等。

**⚠️ 局限性**

局限性包括仅评估行动空间，未考虑令牌消耗、响应时延和成本；轨迹细节有限，仅统计执行可靠性与目标完成；实验仅覆盖少数模型家族，未包括更大规模或开源权重模型。

---

## 461. Learning to Read Braille: Benchmarking and Improving Braille Recognition Capabilities of VLMs

**arXiv ID:** 2610.36195 | [PDF](https://arxiv.org/pdf/2610.36195v1)

**作者:** Kartik Jangra `[一作]` (Nilekani Centre at AI4Bharat), Raj Dabre `[通讯]` (Nilekani Centre at AI4Bharat)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Transformer` `Vision-Language Model` `Supervised Fine-Tuning` `Data Synthesis` `Image Enhancement` `Image` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

创建了BrailleBench真实世界盲文图像基准，并通过合成盲文数据训练Vision‑Language模型，实现了对真实盲文图像的高准确率识别。

**💡 创新点**

提出了专门的盲文基准、合成盲文生成管线以及仅使用合成数据训练的VLM，展示了在无真实训练样本情况下仍能显著提升盲文识别性能的可行性。

**🔧 技术方法**

使用了Vision‑Language模型（Qwen3.5‑2B、Chandra‑OCR‑2）、Liblouis盲文渲染、图像增强与全量微调或LoRA等技术。

**📊 数据集**

使用了259张手工标注的真实盲文图像（BrailleBench，215 Grade‑1 + 44 Grade‑2），以及约276k条合成盲文图像‑文本对和WMT‑Braille 100k对进行对比与评估。

**📈 对比分析**

与零-shot VLM、工具辅助系统以及现有盲文识别工具对比，模型在Grade‑1达到约87%精确匹配、Grade‑2约61%，仅需一次推理，性能接近或超过工具辅助系统（Grade‑1 88%）并显著优于其他VLM和OCR工具。

**⚠️ 局限性**

局限包括样本量相对有限、仅覆盖英语盲文、合成‑真实迁移仍不完全、对长文本和Grade‑2缩写的错误率较高，以及缺乏多语言、多种拍摄条件和更广泛评估。

---

## 462. VISTA: Differentiable Viewpoint Planning for Long-Horizon Task Visualization

**arXiv ID:** 2610.36111 | [PDF](https://arxiv.org/pdf/2610.36111v1)

**作者:** Donggang Jia `[一作]` (King Abdullah University of Science and Technology), Ivan Viola `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `Graphics` `Optimization` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Vision Language Model` `Diffusion model` `Gaussian Splatting` `Contrastive Learning` `Reinforcement Learning` `Mesh` `Image` `Video` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

提出了 VISTA，一种针对长周期任务的可微分视点规划框架，能够根据每一步动作的目标（对象-部件对或关系元组）自动优化相机参数，实现从局部细节到整体结构的连贯可视化。

**💡 创新点**

创新点在于：① 将视点规划视作可微分的可视化策略；② 结合五项互补的视点质量指标（可见覆盖、补丁熵、功能对齐、上下文聚焦、时间连续性）；③ 采用软光栅化实现全流程梯度传递，使相机参数可直接通过梯度优化；④ 支持人类叙事可视化与智能体感知两大应用场景。

**🔧 技术方法**

使用了可微分光栅化（nvdiffrast）、基于 VLM 的语义方向提取（GPT‑5.4）、多起点 Adam 优化、语义分割与实体属性软化、Bézier 曲线平滑等技术；同时引入可视化目标的联合目标函数与时间动态采样。

**📊 数据集**

主要数据集包括 RealAppliance（家电模型），Chevrolet C8 车模型，Isaac Sim 自带的 Forklift 模型；并基于这些模型构建了 VISTA‑Bench，用于评估相机策略对 VLM 感知、目标决策与闭环控制的影响。

**📈 对比分析**

与固定相机、视点熵优化以及人类偏好视点等基线相比，VISTA 在用户研究中获得显著更高的可见性、上下文、运动平滑度与整体满意度（平均 4.53/5），并在 VLM 任务基准上实现 2.1 倍以上的闭环成功率，说明自适应视点规划显著提升了感知与决策性能。

**⚠️ 局限性**

局限性包括：① 目标函数非凸，依赖多起点初始与局部优化；② 仅考虑对象-部件或关系目标，未充分建模更丰富的动作语义（如插入、倾倒等）；③ 对 VLM 提取的前后方向和分割质量高度敏感，误差会影响功能对齐和可见覆盖；④ 软光栅化在极端遮挡或细小特征下可能失效；⑤ 仅通过视点距离和路径长度近似时间连续性，未考虑更高级的运动感知。

---

## 463. Re-Ordering the Generalized Hofstadter Functions G, H, and Beyond

**arXiv ID:** 2610.35994 | [PDF](https://arxiv.org/pdf/2610.35994v1)

**作者:** Pierre Letouzey `[一作]`, Pierre Letouzey `[通讯]`

**关键词:** `Discrete Mathematics` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Knowledge Distillation` `Representation Learning` `Reinforcement Learning from Human Feedback` `Meta Learning` `Review/Survey Paper`

**🎯 论文内容**

本文提出了一种新的证明方法，证明了一类广义Hofstadter函数F_k的逐点单调性，并且提供了对之前文章中多个猜想的首次证明。

**💡 创新点**

创新点在于提供了一种更简洁的证明方法，避免了使用无限形态词，并且引入了新的公式来表达F_k的前像的迭代。

**🔧 技术方法**

使用了Rocq证明助手对所有结果进行了形式化和验证，确保了定义和陈述的准确性。

**📊 数据集**

没有具体提到使用的数据集，但涉及的函数F_k是基于Hofstadter函数的递归定义。

**📈 对比分析**

通过与之前的工作进行比较，证明了F_k(n) ≤ F_k+1(n)的逐点单调性，并且在n > 12(k+1)(k+6)时，F_k(n) < F_k+1(n)。

**⚠️ 局限性**

限制在于虽然提供了新的证明和猜想的验证，但对于更高阶的递归函数的行为仍然需要进一步的研究和探索。

---

## 464. RadarCom: Unlocking Simultaneous Communication and Sensing on Commodity Automotive mmWave Radars

**arXiv ID:** 2610.36622 | [PDF](https://arxiv.org/pdf/2610.36622v1)

**作者:** Yi Wei `[一作]` (Saint Louis University), Nan Cen `[通讯]` (Saint Louis University)

**关键词:** `Networking and Internet Architecture` `Autonomous Driving` `Diffusion model` `Contrastive Learning` `Auto Encoder` `Generative Adversarial Network` `Optical Flow` `Stochastic Differential Equation` `Ordinary Differential Equation` `Point Cloud`

**🎯 论文内容**

本研究提出了RadarCom系统，实现在未改装的商业COTS毫米波汽车雷达上直接进行雷达对雷达通信，同时保持雷达原有的感知功能。

**💡 创新点**

创新点在于将雷达间的相互干扰重新定位为通信载波，并引入斜率不匹配发现、双斜率同步、基于范围FFT的差分编码以及感知感知干扰抑制等完整的软件定义解决方案。

**🔧 技术方法**

采用的软件技术包括：斜率不匹配的信号解调、双斜率同步标识、基于开始频率的差分调制与解调、能量跳跃检测与样本重建、子帧一致性验证等，全部基于标准FMCW雷达工作流程实现。

**📊 数据集**

实验数据来自于使用TI AWR2243 77 GHz COTS雷达的实际采集，包含室内外静态、动态、遮挡、冷启动等多种测试场景，未使用公开数据集。

**📈 对比分析**

与现有ISAC/雷达通信方案对比，RadarCom在保持感知F1~80.5%和距离/速度误差≤0.55 m的同时，实现了平均5.55 kbps的有效数据速率，符号错误率低于1.5%，显著优于传统方法的低速或需要额外硬件的方案。

**⚠️ 局限性**

主要局限在于API重配置延迟导致帧周期长、数据速率受限；仅支持直射LoS传播，遮挡会显著降低速率；未实现多路复用和全双工；对硬件漂移和相位噪声的容错性仍有限。

---

## 465. ASNER: Software Architecture Recovery with Adaptive Semantic Neighborhoods and Evidence-Grounded Refinement

**arXiv ID:** 2610.36894 | [PDF](https://arxiv.org/pdf/2610.36894v1)

**作者:** Wenhao Wang `[一作]` (Wuhan University), Tiankuo Zhao `[通讯]` (Wuhan University)

**关键词:** `Software Engineering` `AI Code Assistant` `Optimization` `Computational Efficiency` `Representation Learning` `Explainability and Interpretability` `Graph Neural Network` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Graph Neural Network` `Community Detection` `Text` `Graph` `Tabular`

**🎯 论文内容**

本文提出 ASNER 方法，用 LLM 自动生成文件责任描述，构建自适应语义邻域（SGASN），将语义边与静态依赖融合后进行社区检测，随后利用责任拥有感知代理（ROA）根据项目责任和代码证据细化组件边界，实现全自动软件架构恢复。

**💡 创新点**

创新点包括：① 将文件责任结构化为 SRS，区分直接实现与调用关系；② 设计 SGASN，根据项目规模与局部相似度动态选择语义邻居；③ 引入 ROA，通过责任上下文与可追踪代码证据约束边界调整；④ 在跨模型、跨语言的 12 项目上进行系统评估，验证方法的鲁棒性。

**🔧 技术方法**

技术手段：LLM（DeepSeek‑V4.1‑Flash、Qwen3‑8B、GLM‑5.2）生成责任与 ROA 决策；Qwen3‑Embedding‑4B 做文件责任向量；ENRE 提取多语言依赖；SGASN 算法构建自适应语义邻域；图融合与社区检测；评价指标 MoJoFM、a2a、ARI、a2a_adj、c2c@66；下游任务 LocAgent 的文件/模块/函数定位。

**📊 数据集**

使用 12 个开源项目（C、C++、Java、Python），NLOC 5K–2.143M；与 SemArc 等基线共享的 8 项目用于统一评测；此外选取 Dask、pypdf 进行下游代码定位验证。

**📈 对比分析**

通过与 SemArc、SARIF、GAER‑GAT、ARC、WCA‑UE、LIMBO、ACDC、Bunch‑SAHC、FCA、SADE 等 10+ 基线在 5 个指标上进行宏平均比较。ASNER 在公共基准上对 GAER‑GAT 提升 ARI 19.5% 与 c2c@66 32.9%；在全部 12 项目上均超越 SemArc；在 LocAgent 下游任务中，文件 Acc@5 从 87.5% 提升至 100%，模块/函数 Acc@10 提升 8.33 分。

**⚠️ 局限性**

局限性：依赖 LLM 推理，推理成本高且对 API 费用敏感；依赖 ENRE 进行依赖抽取，受语言支持与抽取准确性限制；SGASN 对极大规模项目的扩展性尚待验证；ROA 仅在边界文件层面细化，可能无法捕获更深层次的责任错配；实验集中在公开项目，缺乏工业级真实案例验证。

---

## 466. Instant Software Factories

**arXiv ID:** 2610.36254 | [PDF](https://arxiv.org/pdf/2610.36254v1)

**作者:** Wenxin Jiang `[一作]` (Socket Inc), Mikola Lysenko `[通讯]` (Socket Inc)

**关键词:** `Software Engineering` `AI Code Assistant` `Optimization` `Transformer` `Large Language Model` `Prompt Engineering` `Agentic AI` `Text`

**🎯 论文内容**

提出并实现了“即时软件工厂”，通过仅使用定时 cron‑job 代理和共享沟通板（如 GitHub）来完成从原型到产品的长期维护工作，并设计了 bootstrap prompt 以一次性配置工厂。

**💡 创新点**

核心创新点在于证明两种协调输入足以支撑完整工厂，构建了可分离治理层并将工厂视为多目标优化问题，解决了工厂进度评估中的“内生反馈”难题。

**🔧 技术方法**

利用大型语言模型驱动的编码代理、cron 任务调度、GitHub 问题/拉取请求黑板、LLM prompt 自动化设置以及基于stigmergy 的自组织协作。

**📊 数据集**

在两条工业工厂上进行实验：CLI 系统（9 个软件包注册中心、约20个包管理器）和 agentic 系统（内部验证平台），共收集 704 条新问题、58 条 CI 工作流 PR、15-17 天的运行日志。

**📈 对比分析**

通过对比 merged PR 速率、问题关闭率、逃逸率（错误修复）和成本，实验显示 merge 速率提升约 15×，95% 问题由代理创建，逃逸率为 5.6%，每次合并成本约 $17；治理层有效抑制错误扩散。

**⚠️ 局限性**

局限性包括仅在单一组织内验证、实验时长短、缺乏 bootstrap 自动化评估、对高度耦合或多仓库场景支持不足、治理阈值未最小化、数据集偏向 PR 级任务且未完全证明可泛化。

---

## 467. LingGS: Decoupling Feed-Forward Geometry Prediction from Camera Tracking for High-Fidelity Gaussian SLAM

**arXiv ID:** 2610.36853 | [PDF](https://arxiv.org/pdf/2610.36853v1)

**作者:** Da Li `[一作]` (King Abdullah University of Science and Technology), Dominik Engel `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Pose Estimation` `Depth Estimation` `Optimization` `Simultaneous Localization and Mapping` `Transformer` `Gaussian Splatting` `Simultaneous Localization and Mapping` `Optical Flow` `Contrastive Learning` `Huber Loss` `Keyframe Replaying` `Inverse Depth Loss` `Laplace-Guided Gaussian Insertion` `Image` `Point Cloud`

**🎯 论文内容**

设计了一个在线单目高精度高质量3D Gaussian SLAM系统 LingGS，利用流式3D基础模型的稠密深度预测与显式几何跟踪，实现实时相机位姿估计和 Gaussian 映射。

**💡 创新点**

创新点在于将流式模型的深度作为虚拟 RGB‑D 传感器，并引入直接深度残差的几何解算器来显式优化相机位姿；在此基础上构建高效的 Gaussian 映射器，在不训练额外模型的情况下显著提升渲染质量与几何精度。

**🔧 技术方法**

使用了 ORB‑SLAM3 的框架、Huber 鲁棒损失与直接深度残差、Laplace‑guided Gaussian 插入、关键帧重放联合优化、逆深度深度损失，以及预训练的流式 3D 基础模型（如 LingBot‑Map 等）。

**📊 数据集**

在四个室内外数据集上评估：7‑Scenes、Oxford Spires、RE10K、TUM‑D。

**📈 对比分析**

与 MonoGS、S3PO‑GS、Photo‑SLAM、OTF 等基线在 PSNR/SSIM/LPIPS、ATE/RPE、深度误差以及 FPS 上进行对比。LingGS 在渲染指标最高、ATE 与深度误差最低，帧率保持 6–7 FPS，整体性能明显优于基线。

**⚠️ 局限性**

局限性包括未实现环闭合或动态场景处理；依赖 ORB 特征，在纹理弱或重复环境中易失效；缺乏长期一致性与鲁棒性，需要进一步结合学习先验提升。

---

## 468. UnityBench-Mini3D: Evaluating LLM Agents for 3D Game Development in Unity

**arXiv ID:** 2610.35980 | [PDF](https://arxiv.org/pdf/2610.35980v1)

**作者:** Yijun Zhou `[一作]` (Unity China), Jie Feng `[通讯]` (Unity China)

**关键词:** `Software Engineering` `Benchmark` `Large Language Model` `Agentic AI` `Text`

**🎯 论文内容**

暂无具体研究内容

**💡 创新点**

暂无创新点信息

**🔧 技术方法**

暂无技术细节

**📊 数据集**

暂无数据集信息

**📈 对比分析**

暂无对比方法与性能评估

**⚠️ 局限性**

暂无已知限制

---

## 469. JustGoal: With Only a Goal, the AI Baseline Is Above the Expert Kernels

**arXiv ID:** 2610.36434 | [PDF](https://arxiv.org/pdf/2610.36434v1)

**作者:** Yaojian Chen `[一作]` (Tsinghua University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Optimization` `Computational Efficiency` `AI Code Assistant` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Mixture of Experts` `Text` `Tabular` `Benchmark`

**🎯 论文内容**

研究了利用大型基础模型在几乎没有专家知识的情况下自动生成 GPU 内核，并通过与人类手工实现及专门化代理的基准对比验证其性能。

**💡 创新点**

提出“JustGoal”目标仅基准作为评估基线，证明当前最强模型已能自构建评估循环并达到或超越人类实现，重新评估专门化工具的增值。

**🔧 技术方法**

使用通用大模型（Opus 4.8、Opus 5、Fable 5.1）与 Claude Code 等 agent；自行实现评估、验证与计时；对比 GPU 内核生成与手工实现，开展 ablation 与 scope‑matched 对比。

**📊 数据集**

在 NVIDIA B200 GPU 上测试 11 种算子（Attention、Sparse Attention、Linear Attention、MoE 等）共 42 个 shape；对比开源库 FlashAttention‑4、FlashInfer 及公开的 KDA‑0.5、AKO4X、CAKE、AVO 等。

**📈 对比分析**

采用 FlashInfer‑Bench 统一计时，计算几何平均加速比；结果显示 JustGoal 在 8/10 操作符上超过开源库，在 5/9 上超过最快专门化代理；单 shape 速度提升约 1.18×，全 workload 时仍低于人类实现但比单 shape 低。

**⚠️ 局限性**

仅限 NVIDIA B200，未覆盖其他硬件；对比使用不同模型/预算的代理；评估覆盖有限，无法保证所有输入的正确性；专门化工具的增益需在相同模型和任务范围内重新测评。

---

## 470. ACOR: incremental evidence accumulation towards real-time DNA storage reconstruction

**arXiv ID:** 2610.36500 | [PDF](https://arxiv.org/pdf/2610.36500v1)

**作者:** Linxuan Han `[一作]` (Tianjin University), Guanjin Qu `[通讯]` (Tianjin University)

**关键词:** `Information Retrieval` `Computational Efficiency` `Data-Centric Learning` `Restoration` `Beam Search` `Sparse Counting` `Parallel Multi-threading` `Biomedical Data` `Sequential`

**🎯 论文内容**

提出了一种名为ACOR的实时流式 DNA 存储重建方法，在读取序列时即时累积 k‑mer 证据并在簇关闭时一次性进行序列一致性恢复

**💡 创新点**

核心创新在于：1）使用可持续的 k‑mer 证据账本，避免每读更新后重算；2）将稀疏存储与并行束搜索相结合，实现大规模低内存、高速的实时重建；3）通过长度约束束搜索与优先级选取的组合，实现高准确率且仅输出一次一致序列

**🔧 技术方法**

技术包括：k‑mer 窗口滑动累积、稀疏计数账本、长度约束束搜索（beam search）、优先级候选选择、并行多线程实现、稀疏高上下文计数、边界支持统计

**📊 数据集**

在多种公开 DNA 存储数据集（Illumina、Nanopore、Microsoft CNR、Trellis BMA 等）以及作者自制的 96055 条 100nt payload 的 NGS 实验数据集上评估；总计 2.64 亿条读取样本

**📈 对比分析**

与 BBS、VS、BMALA、Deep‑DNA CPL、MUSCLE 等基线方法对比，ACOR 在大多数数据集上实现 92.16–99.92% 的 Exact 恢复率；在自制 NGS 库中 99.92% 的 Exact，且单线程 9 秒、16 线程 20 秒完成 96055 条设计，速度提升 28–1330 倍；内存使用比历史实现低数十倍

**⚠️ 局限性**

局限包括：1）当前仅支持已分配、已关闭的簇，无法自适应在线分簇和闭合；2）对极长（>几 kb）序列的误差控制仍不够理想；3）对噪声高的单端/长读段仍需改进；4）性能仍受输入加载和 I/O 限制影响

---

## 471. WAKMan: Workload Aware KVCache Manager

**arXiv ID:** 2610.36355 | [PDF](https://arxiv.org/pdf/2610.36355v1)

**作者:** Suhas Kumar Morisetty `[一作]` (Columbia University), Wesley Harris `[通讯]` (Capital One)

**关键词:** `Artificial Intelligence` `Optimization` `Computational Efficiency` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

本文研究了大规模语言模型服务中KV缓存失效策略在不同请求类型混合下的表现，并提出一种基于实时流量回放的动态选择器WAKMan，能够在不知晓请求类型或提示内容的情况下，在LRU与S3FIFO之间切换以最大化缓存命中率。

**💡 创新点**

创新点在于：①利用仅需块哈希的shadow缓存回放对最近流量进行评估；②每隔固定时间窗口（T）根据回放结果自动切换失效策略；③完全不依赖请求类型或提示信息，适用于多租户共享部署。

**🔧 技术方法**

主要技术包括：shadow LRU和S3FIFO的实现与回放；基于请求块哈希的统计与模拟；基于T分钟窗口的估算与策略切换机制；对比分析（平均regret、最大regret）与理想无穷缓存基准。

**📊 数据集**

使用19种合成工作负载，包含单轮、会话（多轮）和agentic三种请求类型，20种不同混合比例（S/M/A），每个工作负载以Gemma模型在vLLM上运行，模拟真实GPU内存与块大小。

**📈 对比分析**

实验将WAKMan与LRU和S3FIFO在同一合成流量上进行对比，主要指标为前缀缓存命中率。WAKMan平均提升约5个百分点，最大regret仅为11个百分点，显著优于任何单一静态策略。

**⚠️ 局限性**

局限性：①仅考虑LRU和S3FIFO两种候选策略；②未探究动态窗口长度T的最优值；③实验基于合成数据，缺少真实生产环境的验证。

---

## 472. Callosal: Rethinking the Kernel Page Cache for Disaggregated Storage and Memory

**arXiv ID:** 2610.36347 | [PDF](https://arxiv.org/pdf/2610.36347v1)

**作者:** James McMahon `[一作]`, Anurag Khandelwal `[通讯]`

**关键词:** `Operating Systems` `Optimization` `Federated Learning` `Computational Efficiency` `Retrieval-Augmented Generation` `Contrastive Learning` `Tabular` `Time Series` `Benchmark`

**🎯 论文内容**

实现了一种基于共享分离内存的 Linux 内核页缓存共存机制，利用 RDMA 与可编程交换机的网络加速 DSM（MIND）实现文件页面的强一致性与透明共享。

**💡 创新点**

创新点在于：① 将一致性控制从传统的文件系统/块设备层转移到页缓存层，直接在页缓存中实现跨节点的细粒度一致性；② 采用网络加速的 DSM 与在网络层的共存目录，消除集中式元数据服务器的瓶颈；③ 通过远程锁与版本号验证，保证在多节点并发读写下的强一致性，同时保持局部读写的高性能。

**🔧 技术方法**

使用技术包括：Linux 内核页缓存改造、RDMA 直接内存访问、可编程交换机上的 MIND 共存目录、MSI‑style 锁与版本号校验、内核模块层面对 inode 元数据的远程同步、在用户层实现的轻量级文件系统 CallosalFS 与 KV 存储。

**📊 数据集**

实验数据集主要是合成工作负载：<cite>O'Neil 的微基准（DRBL/DRBM/DRBH, DWOL/DWOM/DWOH, DRWOM/DRWOH）、生产者/消费者队列和基于文件的键值存储（统一与 Zipfian 访问分布）。

**📈 对比分析**

对比方法：与 ext4、tmpfs、本地 NFS、NFS over RDMA、CephFS 四种系统在相同硬件（4 节点、可编程 Tofino 交换机）下进行吞吐量和可扩展性测评。结果显示：在混合读写（5% 写）工作负载下，CallosalFS 对比 NFS 提升约 4.6×；对比 CephFS 取得十倍以上的吞吐量提升；在键值存储中，CallosalFS 在 4 节点上可达 4.5 Mops，扩展比为 1.72×，而 NFS 虽吞吐更高但返回脏数据；CephFS 在多节点扩展时出现性能崩溃。整体表明该方案在保持强一致性的同时实现了高可扩展性。

**⚠️ 局限性**

局限性包括：① 原型仅预分配 DSM，未实现动态内存分配；② 写回与持久化仍未完整实现；③ 仅支持页面级一致性，无法处理更细粒度或更大规模的元数据同步；④ 对高冲突场景（如 DWOH）仍有显著性能下降；⑤ 缺乏完整容错与故障恢复机制；⑥ 仅在实验集群规模（4 节点）内评估，尚未验证大规模部署。

---

## 473. Explicit Reed-Solomon Subspace Designs: Optimal List and Line Decoding

**arXiv ID:** 2610.36215 | [PDF](https://arxiv.org/pdf/2610.36215v1)

**作者:** Omar Alrabiah `[一作]` (University of California Berkeley), Venkatesan Guruswami `[通讯]` (University of California Berkeley)

**关键词:** `Information Theory` `Reed-Solomon coding` `Subspace design` `List decoding`

**🎯 论文内容**

本文构造了一种显式的Reed-Solomon编码，能够在接近列表解码容量的情况下，从错误中恢复信息，且具有最优的列表大小。

**💡 创新点**

创新点在于提出了一种新的构造方法，使得Reed-Solomon编码能够在接近列表解码容量的情况下实现常数大小的列表，而不是多项式增长的列表大小。

**🔧 技术方法**

使用了扩展域上的Reed-Solomon编码和子空间设计的理论，结合了确定性列表解码算法。

**📊 数据集**

使用了多项式大小的扩展域，并且构造了多组评估点以满足编码的要求。

**📈 对比分析**

与Brakensiek等人（2026）的算法相比，本文的方法在列表大小上实现了常数（接近最优），而不是多项式增长，且在选择编码的评估点上进行了精心设计。

**⚠️ 局限性**

限制在于构造的评估点需要满足特定的条件，且在某些情况下可能无法保证所有的编码都能达到预期的性能。

---

## 474. PolicyMediator: Multi-Regional Policy Learning through Mediation Agents with Evolving Skills in OPC Ecosystems

**arXiv ID:** 2610.35997 | [PDF](https://arxiv.org/pdf/2610.35997v1)

**作者:** Jie Li `[一作]` (Northwestern Polytechnical University), yu li `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 15513 | [OpenAlex ID](https://openalex.org/A5100702456)

**关键词:** `Multiagent Systems` `Optimization` `Reinforcement Learning from Human Feedback` `Explainability and Interpretability` `Transformer` `Large Language Model` `Reinforcement Learning` `Bayesian Reinforcement Learning` `Zero-order Policy Gradient Optimization` `Agentic AI` `Tabular` `Text` `Finance Related`

**🎯 论文内容**

通过构建双向媒介代理与自演化技能，结合 Stackelberg 领导-追随框架，对中国六大地区的一人公司（OPC）迁移决策进行多区域政策学习和优化；

**💡 创新点**

首次将已验证的政策解释与认知更新机制嵌入双向中介，实现政策与企业认知的闭环反馈，显著提升了政策匹配效率与地区公平性；

**🔧 技术方法**

使用 LLM 生成式对话、零阶策略梯度优化、贝叶斯强化学习、以及自演化技能评估与修订的多模态方法；

**📊 数据集**

基于公开的中国地区政策、工业适配、进入难度、容量与基准收益数据，构建六地区、五 OPC 类型、六政策类别的仿真环境；

**📈 对比分析**

与8种基线（无政策、静态福利、上下文Bandit、Carbon‑PPO、Nash‑Welfare、Belief‑Robust、Bi‑level Consensus、CONSENSAGENT）对比，PolicyMediator 在所有15种设置中均名列前茅，平均提升政府总体目标 42.4%，OPC 价值 40.9%；

**⚠️ 局限性**

仅在模拟环境中验证，假设政府为协同合作，未考虑竞争性区域与长期动态行为，限制了对真实多主体决策场景的泛化能力。

---

## 475. GarmentSegAgent: Autonomous Garment Segmentation for Web Fashion Images via Textual Self-Refinement

**arXiv ID:** 2610.36506 | [PDF](https://arxiv.org/pdf/2610.36506v1)

**作者:** Jia Chen `[一作]` (Wuhan Textile University), Jun Bai `[通讯]` (Adelaide University)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Transformer` `Vision Language Model` `Agentic AI` `Prompt Engineering` `Text Gradient Backpropagation` `Image` `Text`

**🎯 论文内容**

提出一种自主的服装分割框架 GarmentSegAgent，利用文本自我优化实现无人工空间提示的图像分割；

**💡 创新点**

核心创新在于 CAST 机制，将视觉错误归因到指令字段并进行约束驱动的选择性文本梯度优化，避免全局更新导致的误操作；

**🔧 技术方法**

结合 Qwen2.5‑VL（规划）、SAM 2.1（执行）、Qwen3‑VL（评估）以及 TextGrad 的文本梯度反向传播；

**📊 数据集**

在 DeepFashion2、ModaNet、Fashionpedia 三大服装分割基准上进行评估；

**📈 对比分析**

与多种基线（SAM、LISA、SegZero+SAM、SegAgent 等）比较，GarmentSegAgent 在 mIoU、Boundary IoU、精度和召回率上均优于最强对手，提升幅度约 1–3%；

**⚠️ 局限性**

局限在单目标上衣分割、对多实例/多类的扩展未研究、迭代过程增加推理成本，且依赖冻结模型的视觉反馈无法保证像素级完美性。

---

## 476. GRAFT: Geometry-Aware Residual Adaptation for Foundation Trackers

**arXiv ID:** 2610.36507 | [PDF](https://arxiv.org/pdf/2610.36507v1)

**作者:** Liang Li `[一作]` (Xiaomi Robotics Lab), Diyun Xiang `[通讯]` (Xiaomi Robotics Lab)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Transformer` `Convolutional Neural Network` `Reinforcement Learning` `Knowledge Distillation` `Prompt Engineering` `Point Cloud` `Video`

**🎯 论文内容**

提出一种称为GRAFT的几何感知残差适配框架，利用冻结的行为基础模型（BFM）作为基础，只对下肢执行目标添加地形条件校正；

**💡 创新点**

创新点在于（1）保留冻结BFM参数和运动参考接口，单独训练轻量级适配器；（2）通过奖励设计和地形感知实现对步态的精细调整；（3）引入教师-学生蒸馏，将特权地形观测转为机载深度；（4）实现可部署的门控，支持运行时开启/关闭；

**🔧 技术方法**

使用的技术包括：PPO强化学习、基于地形的奖励（摆爬、踏足、平坦抑制）、教师-学生蒸馏、下肢执行残差网络、模拟与真实机器人部署；

**📊 数据集**

使用数据集：Unitree G1机器人模拟环境中的程序化地形（台阶、平台、缺口等）以及来自SONIC训练集的运动参考；

**📈 对比分析**

与多种方法比较（动作残差、深度学生、LoRA、运动标记残差、从零开始RL），在硬地形下取得接近100%的成功率（四个障碍中39/40），同时在平地上残差极小；

**⚠️ 局限性**

局限性包括：需要为每个冻结基础模型单独训练适配器；主要关注下肢校正，对上肢运动变更的协调不足；实验以预录运动参考为主，未在实时遥控或VLA命令下验证。

---

## 477. HiLo-RD: Cross-Budget Self-Distillation for Masked Diffusion Language Models

**arXiv ID:** 2610.36901 | [PDF](https://arxiv.org/pdf/2610.36901v1)

**作者:** Xiang Li `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `Machine Learning` `Knowledge Distillation` `Computational Efficiency` `Representation Learning` `Transformer` `Diffusion model` `Large Language Model` `Score-based Model` `Contrastive Learning` `Text`

**🎯 论文内容**

提出HiLo‑RD（High‑to‑Low Budget Representation Distillation）方法，在有限生成预算下通过自我蒸馏将同一模型的高窗口推理表示迁移到低窗口，显著提升Masked Diffusion Language Models的推理质量；同时引入高效轨迹获取与清洁EOT停止规则，减少不必要的前向传播；并展示了多轮自蒸馏可进一步改进质量‑预算平衡。

**💡 创新点**

创新点在于：① 通过在同一模型内部对齐高窗口与裁剪后低窗口的隐藏表示，克服预算诱发的早期提交问题；② 设计只依赖模型自身轨迹的跨预算表示蒸馏目标，无需外部教师；③ 结合高效轨迹获取与清洁EOT停止，实现推理成本与答案长度双重优化；④ 通过递归自蒸馏实现模型自我进化的质量‑预算提升路径。

**🔧 技术方法**

使用的技术包括：Masked Diffusion Language Models（如LLaDA‑8B‑Instruct）；跨预算表示蒸馏（HiLo‑RD）与LoRA参数高效适配；纯扩散解码与快速回滚（fast‑rollout）停止规则；注意力诊断与固定预算干预实验；多轮自蒸馏循环。

**📊 数据集**

实验数据集：数学推理领域的GSM8K、MATH500；代码生成领域的HumanEval与MBPP；训练集为完整的GSM8K与MATH原始数据，测试集中使用对应的测试集。

**📈 对比分析**

与基线（vanilla LLaDA‑8B‑Instruct、Masked SFT、Diffu‑GRPO、OPSD、dOPSD）比较，HiLo‑RD在GSM8K上准确率提升约9.4个百分点（从68.84%到78.24%），可读长度缩短约25%；在MATH500、MBPP、HumanEval等任务中同样获得最高准确率或Pass@k并保持最短答案；相对于全窗口解码，HiLo‑RD在低预算下实现更高精度与更短输出，且在高效轨迹获取下进一步降低推理时间。

**⚠️ 局限性**

局限性包括：① 仍需对不同任务与预算规模调优，收益存在任务依赖；② 需要同一模型的高低窗口轨迹，跨模型迁移尚未验证；③ 追踪与蒸馏成本相对较高，尤其在大模型与长序列场景；④ 递归自蒸馏的改进收敛速度与极限未知；⑤ 仅关注可读长度与准确率，尚未系统评估生成多样性与鲁棒性。

---

## 478. Linear-Time FPT Rectilinear Planarity Testing of Series-Parallel Graphs

**arXiv ID:** 2610.37018 | [PDF](https://arxiv.org/pdf/2610.37018v1)

**作者:** Giacomo Ortali `[一作]` (Università Telematica Internazionale Uninettuno), Giacomo Ortali `[通讯]` (Università Telematica Internazionale Uninettuno)

**关键词:** `Data Structures and Algorithms` `Optimization` `Formal Proof` `Algorithm Design` `Recursive Counting` `Formal Verification` `Graph` `Review/Survey Paper` `Benchmark`

**🎯 论文内容**

研究了在可变嵌入情形下，利用SPQ^*-树和螺旋度（spirality）技术对系列并行4-图进行直线可绘制性（rectilinear planarity）检测，并给出了时间复杂度为2^O(k)·n的算法（k为度为4的顶点数）。

**💡 创新点**

提出关键定理（Key Lemma），证明轻型P2节点不增大螺旋度集合的片段数，从而控制整棵SPQ^*-树上所有螺旋度集合的片段数上界；并将该理论与Lean 4形式化验证相结合。

**🔧 技术方法**

采用SPQ^*-树结构、螺旋度集合的算术运算（加、差、交、并）、片段数（piece）计数与递归界定、以及形式化证明工具Lean 4 + Mathlib。

**📊 数据集**

无实验数据集；该工作为纯理论算法分析与形式化验证，未进行数据驱动实验。

**📈 对比分析**

相较于已有的O(n^2)算法，该方法在k=o(log n)时实现子二次时间（n^1+o(1)），并在所有k下保持2^O(k)·n的上界，明显优于传统的O(n^2)上界。

**⚠️ 局限性**

算法的指数因子由对P2/P3节点规则的粗放上界（36^4k）决定，若能进一步细化片段合并分析可能降低指数；另外，目前只能通过度为4顶点数k或重型P节点数h来参数化，尚未证明能仅以P3节点数为参数；在非二分图或弯曲最小化等扩展场景下仍需研究。

---

## 479. LearnPrunerV2: Learnable Progressive Token Pruning for Vision-Language Models

**arXiv ID:** 2610.36986 | [PDF](https://arxiv.org/pdf/2610.36986v1)

**作者:** Rinyoichi Takezoe `[一作]` (Li Auto Inc.), Kaiwen Long `[通讯]` (Li Auto Inc.)

**关键词:** `Computer Vision and Pattern Recognition` `Computational Efficiency` `Representation Learning` `Knowledge Distillation` `Transformer` `Vision Language Model` `Contrastive Learning` `Large Language Model` `Multimodality` `Image` `Text` `Video`

**🎯 论文内容**

提出 LearnPrunerV2，一种可学习的逐层视觉令牌裁剪框架，利用轻量裁剪模块估计视觉令牌重要性，并通过预算确定裁剪（BDP）实现固定层预算，兼顾硬件友好与性能；

**💡 创新点**

创新点包括：1）全流程可学习裁剪机制，加入阴影令牌获取查询相关性并通过门控平滑跨层信息；2）提出预算确定裁剪（BDP）为每层分配固定令牌预算，并结合多样性-aware 选择提升低预算下的表现；3）端到端联合训练裁剪模块与 VLM，消除先前两阶段或手工调度的缺陷；

**🔧 技术方法**

使用技术：轻量可学习裁剪模块（LPM）与 MLP，阴影令牌（Shadow Token）提取查询相关性，门控机制抑制跨层注意噪声，Straight‑Through Estimator 实现可微分裁剪决策，预算确定裁剪（BDP）与多样性-aware 选取，整体端到端联合训练；

**📊 数据集**

训练数据：LLaVA‑mix665k（用于学习裁剪模块）；评测数据集：GQA、ScienceQA‑IMG、TextVQA、POPE、MME、VQAv2、MMBench、MMBench‑CN、MMVet，以及视频基准 TGIF‑QA、MSVD‑QA、MSRVTT‑QA；

**📈 对比分析**

对比方法：FastV、SparseVLM、VisPruner、DART、CDPruner、LearnPruner 等；实验结果显示：在 LLaVA‑v1.5‑7B 上保持 99.8% 以上性能的同时裁剪 77.8% 视觉令牌；在 LLaVA‑NeXT‑7B 高分辨率场景下 22.2% 令牌依旧保持 99.7%；在 Qwen2.5‑VL‑7B 上实现 100%+ 平均准确率，显著优于前沿方法；

**⚠️ 局限性**

局限性：1）固定预算下对实例适应性受限；2）在极低令牌数（≤32）下性能略有下降；3）对不同视觉编码器需要重新训练裁剪模块；4）缺少针对实时多模态场景的动态预算自适应机制。

---

## 480. A Graph-informed Kriging model for sample-efficient Bayesian optimization of wind farm layouts

**arXiv ID:** 2610.35975 | [PDF](https://arxiv.org/pdf/2610.35975v1)

**作者:** Zhen-Fan Wang `[一作]` (Shanghai Jiao Tong University), Onur Bilgen `[通讯]` (Rutgers University)

**关键词:** `Machine Learning` `Optimization` `Graph Neural Network` `Gaussian Process` `Bayesian Optimization` `Graph` `Tabular` `Physics Related`

**🎯 论文内容**

提出 Graph-informed Kriging（GIK）框架，将风机对间的低维相对坐标交互信息嵌入 Kriging 均值，提升高维稀缺样本下的 AEP 预测与贝叶斯优化效率。

**💡 创新点**

创新点在于利用有向气动图将双机相互作用重用为全场趋势函数，显著降低对全场高阶数据的需求，突破传统 Kriging 在大规模 WFLO 中的样本效率瓶颈。

**🔧 技术方法**

技术组合包括：有向图与图传递理论、Kriging/高斯过程、贝叶斯优化、离散与连续 wake 模型（Gaussian、ADM‑RANS）、多阶图中心度与通量分析，以及多精度 Kriging 对比。

**📊 数据集**

使用的数据集包括 NREL 5 MW 风机离散格点、IEA Wind Task 37 风场与布局、Lillgrund 风机布局，以及通过两机 Gaussian wake 或 ADM‑RANS 生成的双机交互离线数据库。

**📈 对比分析**

与常数/线性/多项式趋势 Kriging、普通 Kriging BO、双精度 Kriging、DEBO、SNOPT+WEC、CMA‑ES 等方法对比，GIK 在 IEA Task 37 仅约 280 次 AEP 评估即可接近 DEBO 的最优 AEP，CFD 基案例（16 机/48 机）仅需 200/80 次评估即可获得显著改进，样本效率提升 2–3 阶。

**⚠️ 局限性**

局限性包括：未能完全捕捉多机非对称耦合与时间变流动；基于稳态 ADM‑RANS 或 Gaussian wake，忽略了湍流、机组控制等动态效应；需预先生成双机数据库，机型或风况变化时需重建；未考虑控制、结构或经济多目标约束。

---

## 481. Multi-task Disentangling Fusion Mamba-CNN for MODIS Spectral-Temporal Unmixing with Endmember Variability Modeling

**arXiv ID:** 2610.36750 | [PDF](https://arxiv.org/pdf/2610.36750v1)

**作者:** Motasem Alkayid `[一作]` (University of Calgary), Lincoln Linlin Xu `[通讯]` (University of Calgary)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Image Translation` `Recognition` `Segmentation` `Data Synthesis` `Convolutional Neural Network` `Transformer` `Mixture of Experts` `Contrastive Learning` `Auto Encoder` `Generative Adversarial Network` `Score-based Model` `Flow-based Model` `Diffusion model` `Image` `Time Series` `Tabular` `Agriculture Related`

**🎯 论文内容**

本文提出一种名为 Multi-task Disentangling Fusion Mamba (MDF‑Mamba) 的端到端网络，用于 MODIS 时序数据的光谱‑时序解混，结合 CNN 专家、Mamba 状态空间模型、流形约束超连接、可学习端元变体与跨注意力重建，并通过多任务监督与课程学习实现高精度的混合率估计与光谱重建。

**💡 创新点**

创新点：①异构多分支专家网络按波段固定分配，解耦光谱‑时序耦合；②将 Mamba 状态空间模型用于时间序列编码；③使用流形约束的超连接 (mHCFusion) 进行稳定、可解释的多专家融合；④引入 ELMM 并结合像素级可学习端元变体与跨注意力重建；⑤多任务监督与课程学习同时约束混合率与光谱重建。

**🔧 技术方法**

所用技术包括：双通道 CNN（光谱与空间路径）、Squeeze‑and‑Excitation、Twist Gate、双向 Mamba 状态空间模型、FiLM 归一化、跨注意力重建、ELMM、拉普拉斯光谱平滑、Mahalanobis & SAD 重建损失、对比损失、Sinkhorn 双随机矩阵（mHCFusion）等。

**📊 数据集**

数据集：MOD13Q1 250 m 16日复合，2010年，覆盖加拿大阿尔伯塔与萨斯喀彻温省，6 个波段（NDVI、EVI、Red、NIR、Blue、MIR），23 个时序，14 个土地覆盖类（最终评估 12 类），按地理块划分的训练/验证/测试集。

**📈 对比分析**

与 11 个传统与深度学习基线对比（FCLSU、SGSNMF、SUnSAL、PGMSU、SUnCNN、MiSiCNet、SSAF‑Net、PPM‑Net、DeepTrans‑HSU、SRViT、UNMamba）。在阿尔伯塔与萨斯喀彻温的 12/14 评价指标中取得最佳或第二佳：MAE = 0.0255、RMSE = 0.0654、重建 MSE = 0.0067、SAD = 0.1843、OA = 0.712/0.748、AA = 0.472/0.478、Kappa = 0.658/0.688，显著优于传统线性混合、稀疏正则、Transformer 与 SSM 等方法。

**⚠️ 局限性**

局限性：①依赖 MODIS 250 m 分辨率，纯度低的类（如苔藓、冰雪）难以区分；②模型结构较大、训练耗时，需课程学习与精细的端元初始化；③跨域泛化虽好但受地区光谱分布差异影响；④缺乏对极端光谱噪声或云覆盖的鲁棒性，需进一步改进。

---

## 482. Multi-Scale Spatio-Temporal Exploratory Analysis of Urban Vehicle Theft

**arXiv ID:** 2610.36478 | [PDF](https://arxiv.org/pdf/2610.36478v1)

**作者:** Chaitanya Jobanputra `[一作]` (German Research Center for Artificial Intelligence), Gerrit Großmann `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `Computers and Society` `Crime Analysis` `Urban Studies` `Geospatial Getis–Ord G_i^*` `Moran's I` `Gaussian kernel density estimation` `Knox space-time near-repeat test` `STL seasonal-trend decomposition` `PELT change point detection` `Negative binomial regression` `Cox spatial autocorrelation test` `Jaccard similarity comparison` `Tabular` `Geospatial Data` `Time Series` `Review/Survey Paper` `Benchmark`

**🎯 论文内容**

本研究对2000-2022年洛杉矶警方车辆失窃记录进行清洗、去重、时空标准化后，利用多尺度（小时至十年、100米格子至人口普查区）分析车辆失窃的空间集中度、热点持续性、近重复模式以及Kia与Hyundai车型的盗窃波动；

**💡 创新点**

创新点在于对长期行政记录进行系统的数据预处理（事件定义、时间字段解析、坐标校正、人口普查边界映射），并在此基础上通过多重检验控制、空间统计与自激进点过程等多种方法构建一套全面、可重复的空间时空探索框架；

**🔧 技术方法**

主要技术包括Geospatial Getis–Ord G_i^*热点检测、Moran's I空间自相关、Gaussian kernel密度估计、Knox空间-时间近重复检验、STL季节-趋势分解、PELT分段变点检测、负二项回归、Cox空间自相关检验与Jaccard相似度比较；

**📊 数据集**

使用的数据集为洛杉矶警方2000-2022年车辆相关犯罪记录（约521,098起失窃案）、2020年人口普查区划及多版本ACS车辆、人口、收入数据，以及开放街地图POI点集；

**📈 对比分析**

方法主要以描述性统计和空间统计检验为主，未进行传统机器学习性能对比；通过多重检验控制和空间显著性检验，确保热点与近重复的统计显著性，整体表现符合预期且可在其他城市重复使用；

**⚠️ 局限性**

主要局限包括：数据公开受限、地理编码精度不一（尤其是未验证的交叉地址）、车辆曝光度以家庭车辆计数为近似、空间单元对结果影响大（MAUP）、仅研究单一城市、2022年数据部分缺失、回收率和交通记录被排除后对整体量值的影响未完全量化。

---

## 483. The View from Nowhere: Population-Centered Agent Evolution

**arXiv ID:** 2610.36660 | [PDF](https://arxiv.org/pdf/2610.36660v1)

**作者:** Shuo Yan `[一作]` (Macquarie University), Shang Gao `[通讯]` (Macquarie University)

**关键词:** `Artificial Intelligence` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Data-Centric Learning` `Meta Learning` `Transformer` `Large Language Model` `Knowledge Distillation` `Prompt Engineering` `Retrieval-Augmented Generation` `Reinforcement Learning` `Text`

**🎯 论文内容**

提出并实现了 Population-Centered Evolution（PCE），一种利用演化种群中所有个体的任务成功/失败记录和自我改造历史来指导单个代理改造的自我提升框架。

**💡 创新点**

创新点在于引入“view from nowhere”原则，将种群覆盖与最佳个体性能差距作为改造驱动；通过“population block”聚合跨个体模式来产生诊断假设，从而实现跨个体的知识迁移与性能凝聚。

**🔧 技术方法**

主要技术包括：1）任务记录与修改历史的收集与重构；2）distillation agent 对所有记录进行统计、对比并生成 population block；3）诊断模块结合父体记录、代码与 population block，提出并实现单体改造；4）演化循环与子代评估；5）token 与请求计数用于资源对比。

**📊 数据集**

实验数据集：SWE-bench Verified（200 任务，其中 60 用于演化、140 作为 hold-out）以及 Polyglot（225 任务，60 用于演化、165 作为 hold-out）。

**📈 对比分析**

与基线 GEA（同模型、同初始 harness、相同演化调度）进行对比。PCE 在 SWE-bench Overall 上取得 72.1% 成功率（比 GEA 提升 3.2pp），Polyglot Overall 上取得 96.7%（比 GEA 提升 5.9pp）。此外 PCE 在种群覆盖与最佳个体差距、前五名个体性能以及 token 消耗上均优于 GEA，token 使用降低 36%（SWE-bench）和 59%（Polyglot）。

**⚠️ 局限性**

局限性包括：1）仅在固定基础模型和编码代理场景下验证，未评估在更大规模或不同任务领域的普适性；2）population block 生成与诊断仍依赖于 distillation agent 的能力，可能受限于模型解释能力；3）对种群规模扩大后信息压缩与迁移的机制尚未深入研究。

---

## 484. Quantifying and Prioritizing Stealthy Source-Manipulation Threats Against Multi-Source Anomaly Detectors

**arXiv ID:** 2610.36693 | [PDF](https://arxiv.org/pdf/2610.36693v1)

**作者:** Indraveni Chebolu `[一作]` (Centre for Development of Advanced Computing), Ch A S Murty `[通讯]` (Centre for Development of Advanced Computing)

**关键词:** `Cryptography and Security` `Anomaly Detection` `Contrastive Learning` `Auto Encoder` `Supervised Fine-Tuning` `Optimization` `Tabular` `Time Series` `Benchmark` `Agriculture Related` `Physics Related`

**🎯 论文内容**

本文提出了任务条件可见性边际（task-conditioned visibility margin）作为一种本地第一阶屏蔽度量，用来评估在跨源异常检测器中某一来源子集被攻击时对下游任务的威胁，并利用该度量为审计优先级排序；

**💡 创新点**

创新点在于给出闭式公式计算该边际，揭示检测器对任务相关方向的盲点并可量化最小证据攻击，进而指导审计顺序；

**🔧 技术方法**

技术包括梯度与雅可比矩阵线性化、SVD求解、闭式公式、有限攻击优化、学习型异常检测器（PCA、GDN、TranAD）以及多任务模型（ridge、MLP）；

**📊 数据集**

使用的数据集为17通道工业水泵遥测数据（UCI Hydraulic Systems）、ICRISAT五源作物产量面板以及公开的遥测基准；

**📈 对比分析**

通过与随机顺序、告警余量、任务梯度基线等排名方法对比，margin在前十次审计中平均发现四个失败，明显优于随机和告警排名，并与任务梯度排名相近；

**⚠️ 局限性**

局限性是该度量仅为局部一阶屏蔽度量，无法完整捕捉大预算非线性攻击的全局风险，对时间变化或物理约束的适用性有限。

---

## 485. Polymorphic First-Class Environments

**arXiv ID:** 2610.36798 | [PDF](https://arxiv.org/pdf/2610.36798v1)

**作者:** Jinhao Tan `[一作]` (University of Hong Kong), Bruno C. d. S. Oliveira `[通讯]` (University of Hong Kong)

**关键词:** `Programming Languages` `AI Code Assistant` `Computational Efficiency` `Representation Learning` `Explainability and Interpretability` `Prompt Engineering` `Retrieval-Augmented Generation` `Large Language Model`

**🎯 论文内容**

提出了一种多态的第一类环境（Polymorphic First-Class Environments）计算机语言核心，使得环境可作为普通值进行创建、传递、检查与组合，并且保留 System F 的多态性；

**💡 创新点**

通过引入显式的“盒类型”（Box Types）和“显式类型绑定”（Manifest Types）以及双侧类型等价（Type Equivalence）机制，消除了 System F 中所有形式的替换（term‑to‑term、type‑to‑term、type‑to‑type），实现了完全无替换、基于环境的小步语义；

**🔧 技术方法**

利用 de Bruijn 索引、显式环境语义、manifest type 绑定、类型等价规则、instantiation lemmas 以及在 Coq 中的机械化证明；

**📊 数据集**

该工作不使用外部数据集，而是对核心语义和元理论进行形式化验证，所有相关证明已在 Coq 证明助手中完成；

**📈 对比分析**

对比传统 System F 的元层替换语义，本系统通过环境语义保持了类型安全、归一化和类型等价可判定等性质，且实现了动态模块组合、沙盒化和透明类型组件等功能；但本文未给出执行时间或内存使用等性能评测；

**⚠️ 局限性**

局限性包括：缺乏封闭（sealing）和子类型支持、无法处理递归模块、对高阶子类型和隐式类型推断的支持有限，以及对 ML 风格模块系统中更复杂的抽象机制（如透明/不透明类型）尚未完全覆盖。

---

## 486. Is Token Confidence Just Chain Length? A Signal-Processing Audit of Reasoning Traces

**arXiv ID:** 2610.36709 | [PDF](https://arxiv.org/pdf/2610.36709v1)

**作者:** Pragna Mallappa `[一作]` (Keshav Memorial Institute of Technology), Pragna Mallappa `[通讯]` (Keshav Memorial Institute of Technology)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Transformer` `Chain-of-Thought` `Large Language Model` `Signal Processing (sliding average, spectral estimation, CUSUM)` `Text` `Benchmark`

**🎯 论文内容**

研究将链式推理中的token confidence视为信号，利用滤波与检测评估其在链选择与早停中的有效性。

**💡 创新点**

指出DeepConf低分主要捕捉链长度而非推理质量，1/f谱中高频成分为唯一剩余有效信息。

**🔧 技术方法**

采用信号处理技术（滑动平均、频谱估计、CUSUM等）以及对token概率的log转换计算token confidence。

**📊 数据集**

在MATH500和GSM8K两个数学推理基准上，使用1.5B参数的DeepSeek模型生成链进行实验。

**📈 对比分析**

与长度排序、平均confidence、DeepConf等基线比较，发现长度排名优于DeepConf，早停检测无显著优势，整体准确率与完整投票相近。

**⚠️ 局限性**

局限在于仅使用单一小模型、固定采样温度和k=20，并且早停评估仅在已完成链上模拟，未考察更大模型或自由文本答案。

---

## 487. Minimax-Optimal Small-Loss Bounds for Online Linear Optimisation on $\ell_p$ Balls

**arXiv ID:** 2610.36097 | [PDF](https://arxiv.org/pdf/2610.36097v1)

**作者:** Jack J. Mayo `[一作]` (Kurtos.ai), Jack J. Mayo `[通讯]` (Kurtos.ai)

**关键词:** `Machine Learning` `Optimization` `Bregman散度分解` `shifted-loss stability分析` `径向与二次正则化的梯度闭式` `单次切换调度` `局部曲率与熵方法` `对i.i.d.损失的期望偏差拆分`

**🎯 论文内容**

在全信息下的线性在线优化（OLO）问题中，针对ℓp球（p≥2）提出两种FTRL算法，利用径向与二次正则化并在累积偏移损失阈值处切换，获得关于“消除滑移余量”的小损失渐近上界；

**💡 创新点**

创新点在于①首次给出了能同时兼顾两条小损失分支的最优常数（2+o(1))p/ln p；②提出“shifted‑loss stability”正则化概念；③通过单次切换实现无预知期望的最优小损失率；④构造了一个可直接实现的、带有唯一根求解的二维正则化器；

**🔧 技术方法**

技术上主要使用了：Bregman散度分解、shifted‑loss stability分析、径向与二次正则化的梯度闭式、单次切换调度、局部曲率与熵方法、以及对i.i.d.损失的期望偏差拆分；

**📊 数据集**

本工作完全基于理论分析，没有使用任何实验数据集；

**📈 对比分析**

通过与下界构造对比、与FTL比较、以及对不同p值的渐近常数分析，证明了算法在所有维度、所有时限和所有损失路径下都达到最优的最小极限；在随机环境中，算法实现了对消除滑移与均值范数的分离，收敛速率为O(log T)且分离了方差与正则化偏差；

**⚠️ 局限性**

局限性包括：仅适用于全信息反馈；只能在ℓp球上有效，无法直接推广到一般对称凸体；算法需要精确的支持函数和动作求解，数值实现对精度和计算复杂度尚未完全分析；在带反馈（bandit）情形下无法直接应用；对平方根分支的常数仍非最优。

---

## 488. Persistence, Exposure, and Credit Routing in Linear Temporal Predictive Coding

**arXiv ID:** 2610.36412 | [PDF](https://arxiv.org/pdf/2610.36412v1)

**作者:** Akira Tanase `[一作]` (Sakana AI), Jeffrey Seely `[通讯]` (Sakana AI)

**关键词:** `Machine Learning` `Optimization` `Representation Learning` `Explainability and Interpretability` `Recurrent Neural Network` `Contrastive Learning` `Auto Encoder` `Adam optimization` `Linear algebra derivation` `Pseudo-inverse solving` `Time Series` `Sequential`

**🎯 论文内容**

对线性时序预测编码网络的信用分配机制进行理论分析，并通过持久性与曝光算子P、Q揭示空间与时间误差的平衡，进而提出模式路由机制来克服单模式障碍。

**💡 创新点**

创新点在于将细胞沙弗理论与预测编码结合，引入P、Q算子解释信息在空间模式中的长期保留与曝光，并证明标准局部tPC缺失未来信用，提出模式路由学习路径，展示了在长时序任务中显著提升的能力。

**🔧 技术方法**

采用细胞沙弗理论、直接网络沙弗扩散、伪逆求解、线性代数推导、Adam优化等技术；实验使用单隐藏层线性网络与30位复制任务。

**📊 数据集**

数据集主要为人工生成的序列（含随机提示和干扰片段）以及30位复制任务数据。

**📈 对比分析**

通过对比标准局部tPC更新与完整时间信用（全梯度）更新，在延迟召回任务中，完整时间信用将均方误差降至≈0.005，而局部tPC仅接近基线≈0.97；在复制任务中亦出现相同的性能差距。

**⚠️ 局限性**

局限性在于标准局部tPC缺乏长期信用传播，难以学习模式路由；目前方法仅在线性网络上验证，非线性扩展尚待进一步研究。

---

## 489. Any3R: Feedforward 3D Reconstruction from Any Camera Type

**arXiv ID:** 2610.36150 | [PDF](https://arxiv.org/pdf/2610.36150v1)

**作者:** Jingong Chen `[一作]` (South Dakota State University), Kyle Gao `[通讯]` (Aalto University)

**通讯引用:** 651 | [OpenAlex ID](https://openalex.org/A5080492904)

**关键词:** `Computer Vision and Pattern Recognition` `Depth Estimation` `Pose Estimation` `3D Reconstruction` `Transformer` `Neural Radiance Field` `Contrastive Learning` `Optical Flow` `Cross-View Attention` `Depth Decoding` `Image` `Point Cloud`

**🎯 论文内容**

提出Any3R模型，能够一次性从任意混合的中央相机（针孔、鱼眼、全景）中重建3D场景并预测相机位姿，无需重新校准或去畸变。

**💡 创新点**

创新点在于将稠密射线场（Ray Field）注入位置编码与损失函数，采用角坐标位置编码和固体角加权，彻底消除对针孔假设的依赖，使单一模型适用于所有中央相机。

**🔧 技术方法**

使用视觉Transformer架构、角坐标位置编码、固体角加权的点误差、深度解码、跨视图注意力等技术。

**📊 数据集**

在Matterport3D、Stanford2D3DS、Structured3D、PanoCity、DL3DV等公开数据集上，结合13种中央相机模型（含pinhole、鱼眼、全景等）进行评估；并在ScanNet++、FIORD等真实相机数据上验证零样本能力。

**📈 对比分析**

与Wid3R、OmniVGGT、PanoVGGT、Argus、MapAnything、π^3、VGGT等现有模型进行比较，Any3R在大多数相机类型下实现了最高的重建精度和位姿精度，尤其在宽角和全景镜头上表现尤为突出。

**⚠️ 局限性**

局限性：仍需预先提供射线场（相机标定），对非中央相机的适应性有限；极端大视角（>360°）下的表现未全面验证；实时部署时的速度与内存开销尚待进一步优化。

---

## 490. Freeze, Don't Clone: Fine-Tuning with Expressive Policies Without Retaining Offline Data

**arXiv ID:** 2610.36247 | [PDF](https://arxiv.org/pdf/2610.36247v1)

**作者:** Sami Nur Islam `[一作]` (Mila), Pablo Samuel Castro `[通讯]` (Mila)

**关键词:** `Machine Learning` `Reinforcement Learning` `Robotic Intelligence` `Transformer` `Reinforcement Learning` `Diffusion model` `Score-based Model` `Contrastive Learning` `Knowledge Distillation` `Tabular` `Time Series` `Sequential` `Benchmark`

**🎯 论文内容**

在离线预训练后，当离线数据被丢弃时，研究发现继续进行行为克隆会导致表达式先验被快速遗忘，从而引发表示崩塌和在线学习崩溃；为此提出在“无保留”微调阶段停止在线行为克隆，仅保留冻结的表达式先验，并通过 critic-guided 选择或一阶演员蒸馏实现策略改进；

**💡 创新点**

创新点在于揭示行为克隆在无离线数据场景下的致命缺陷，提出冻结表达式先验而不继续行为克隆的两步设计原则，显著提升多种流形/扩散式策略在无离线数据微调中的稳健性；

**🔧 技术方法**

使用流式匹配（IFQL、FQL）、扩散式策略（IDQL）等高表达式先验方法，并结合行为克隆（BC）与 critic 更新或蒸馏来实现决策；

**📊 数据集**

主要实验数据集为 OGBench 7 个操作与运动域（共 35 任务），低质量离线数据（D4RL、Meta-World 等）以及 Robomimic 人类遥控演示数据（Multi-Human），并在这些任务上进行比较；

**📈 对比分析**

与传统的 BC-On（持续在线行为克隆）以及保留离线数据的基准相比，BC-Off（冻结先验）在 IFQL、FQL、QC 等方法上在 1M 在线步内均显著提升成功率，甚至接近或超过保留离线数据的表现；

**⚠️ 局限性**

局限性包括：仅在表达式多模态先验下有效，单峰（高斯）策略无法从中获益；冻结先验可能限制对环境分布漂移的适应；对极低质量离线数据的提升有限；

---

## 491. Explicit Regional Representations For Video Quality Assessment

**arXiv ID:** 2610.36978 | [PDF](https://arxiv.org/pdf/2610.36978v1)

**作者:** Fuyi Yang `[一作]` (University of California), Ming Kai Hsu `[通讯]` (Transtreams Technologies)

**关键词:** `Computer Vision and Pattern Recognition` `Quality Assessment` `Representation Learning` `Transformer` `Vision Language Model` `Contrastive Learning` `Optical Flow` `Supervised Fine-Tuning` `Video` `Text`

**🎯 论文内容**

提出了视频质量评估的像素级区域表示——质量相关热图（QRH），并在DOVER模型上演示其有效性。

**💡 创新点**

创新点在于将质量证据与感知先验分离并通过视觉‑语言模型进行语义引导，形成无需像素级标注的显式区域表示，并设计两种确定性采样策略。

**🔧 技术方法**

使用了多种局部质量线索（模糊、运动不一致、压缩痕迹）、视觉注意力与运动易感性模型、视觉‑语言模型（如Qwen3.6-35B）、soft‑OR融合与匈牙利算法实现采样。

**📊 数据集**

在LSVQ‑test/1080p、KoNViD‑1k、LIVE‑VQC以及YouTube‑UGC等公开评估集上进行实验。

**📈 对比分析**

与原始DOVER及多种基准方法对比，分布保持采样（DPS）在不重新训练时即可逼近随机采样平均预测，而质量感知采样（QAS）结合微调后在大多数指标上均超过原模型，SROCC/PLCC提升约2–3%。

**⚠️ 局限性**

局限性包括对VLM语义判断的依赖、需要额外的微调以适配不同采样分布、仅在现有片段式VQA骨干上验证，且在完全无标注的场景下仍需进一步探索学习式区域表征。

---

## 492. A Fine-Grained Empirical Study of Upstream Security Events and Downstream Mitigation in Maven

**arXiv ID:** 2610.36603 | [PDF](https://arxiv.org/pdf/2610.36603v1)

**作者:** Kazi Amit Hasan `[一作]` (Queen's University), Steven H. H. Ding `[通讯]` (McGill University)

**关键词:** `Software Engineering` `Safty and Privacy` `Tabular` `Time Series` `Benchmark`

**🎯 论文内容**

本文追踪了Maven生态中832个CVE从上游修复到下游项目采用的完整生命周期，系统分析了上游披露、修复、发布、NVD公开以及下游采用的时序与延迟。

**💡 创新点**

首次将上游披露方式（silent/transparent）、生命周期顺序、下游依赖暴露量、采用时延以及自动化工具（dependabot）的使用纳入同一统一框架，从而揭示两端延迟交互对整体安全暴露的共同影响。

**🔧 技术方法**

采用GitHub API、NVD API及Maven元数据抓取技术，对事件时间进行排序、统计和可视化，并使用卡方检验、Mann‑Whitney U检验等非参数统计方法进行比较。

**📊 数据集**

使用了前期Maven生态研究的832个CVE、257个上游仓库和29,952个下游项目，构成44,450个(CVE, upstream, downstream)三元组，并从中筛选出1,677个可观测的下游采用对。

**📈 对比分析**

通过事件顺序对比与延迟分箱（≤1天、1‑7天、7‑30天等）评估，发现70%以上的采用发生在NVD公开后；自动化工具（dependabot）将发布至采用的平均延迟从约110天降至约10天，表明自动化能显著缩短缓解时间，但多数项目仍延迟超过90天。

**⚠️ 局限性**

研究仅覆盖公开CVE的Maven项目，缺乏私有或无CVE漏洞的情况；依赖公开时间戳可能存在误差；观测到的采用对仅占曝光总量3.8%，未能捕捉全部暴露；且未分析非Dependabot的自动化工具。

---

## 493. PailitaoMMSearch2.0 Technical Report: Scaling Generative Retrieval to Unified End-to-End Multimodal Search

**arXiv ID:** 2610.36958 | [PDF](https://arxiv.org/pdf/2610.36958v1)

**作者:** Xiaohan Ye `[一作]` (Taobao & Tmall Group of Alibaba), Bo Zheng `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `Information Retrieval` `Retrieval` `Recommendation System` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Chain-of-Thought` `Vision Language Model` `Retrieval-Augmented Generation` `Image` `Text` `Multimodality`

**🎯 论文内容**

构建了一套统一端到端的多模态生成检索系统 PailitaoMMSearch2.0，能够直接从用户上传的图片和自由文本指令生成商品 Semantic ID（SID），实现对相似、编辑、匹配等多种购物意图的检索，并在不同资源约束的在线场景中部署；

**💡 创新点**

①通过可组合的数据扩展工具集实现冷启动、实时流量精炼与持续生产的多阶段数据生成；②采用持续预训练（CPT）、混合预算监督微调（SFT）和统一强化学习（RL）等多阶段训练，学习一模型即可处理多意图检索；③在RL阶段引入 Sample–Beam 轮询策略，将生成过程与在线 beam 搜索对齐；④通过模型规模与推理预算双维度的场景缩放，提供可配置的质量‑成本运营点；

**🔧 技术方法**

使用原生视觉‑语言模型（Qwen3.5 及其变体）进行生成检索；持续预训练、混合预算 SFT、RL（GRPO+Sample–Beam）等多阶段训练；链式推理（CoT）生成与质量控制模块；

**📊 数据集**

Taobao/Tmall 购物目录及其产品属性、图片、标题等多模态数据；通过工具集生成的代理产品关系、冷启动合成数据、实时流量日志；评估集按相似/编辑/匹配三类拆分，评判者校准集为独立人工标注；

**📈 对比分析**

与 Qwen3.8-Flash-Next、GPT6-luna 及内部 recall‑ranking 系统进行离线对比，使用 NDCG、Good Rate、Excellent Rate、平均响应时延和 GPU 使用等指标；2B Low‑think 配置宏观 Excellent Rate 55.1%（比 baseline 高 3.6）、宏观 NDCG 0.918、RT 278 ms（比 baseline 低 64%），在线 A/B 测试提升商品页浏览 +17.90%、交易量 +27.04%、GMV +8.94%；

**⚠️ 局限性**

仅覆盖三种购物关系且为单域电商；“开放式”指接口而非完全关系覆盖；大多数消融在 2B checkpoint 进行，三种模型尺寸为独立 checkpoint，缺乏弹性参数共享；自动评判器与数据过滤器可能存在关联误差；在线 A/B 结果难以归因单一组件；未来需扩展关系税onomies、联合优化检索质量与延迟、研究可压缩或自适应模型等。

---

## 494. Comparative Analysis of Large Load Interconnection Requirements Across U.S. Transmission Entities

**arXiv ID:** 2610.36645 | [PDF](https://arxiv.org/pdf/2610.36645v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Hardware Architecture`

---

## 495. DocXEval: A Systematic Evaluation of Explainability in Multimodal Document Understanding

**arXiv ID:** 2610.36929 | [PDF](https://arxiv.org/pdf/2610.36929v1)

**作者:** Saifullah Saifullah `[一作]` (German Research Center for Artificial Intelligence), Sheraz Ahmed `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Transformer` `Vision Language Model` `Contrastive Learning` `Auto Encoder` `Generative Adversarial Network` `Multimodality` `Text` `Image` `Benchmark`

**🎯 论文内容**

研究提出了DocXEval框架，对多模态文档理解任务中的可解释性进行系统评估。

**💡 创新点**

创新点在于将现有XAI方法与评价指标适配为多模态、多目标环境，并提供首个综合基准。

**🔧 技术方法**

使用了13种后验和注意力可解释方法、17种评估指标，以及多模态特征分割、基线选择等技术。

**📊 数据集**

评估覆盖了3个任务、8个公开数据集（如FUNSD、CORD、DocVQA、SciDocs等）和4个模型。

**📈 对比分析**

通过排名与统计分析发现Occlusion、Attention-based和Gradient-based方法表现最佳，整体解释可信度高，但多方法得分差异不显著。

**⚠️ 局限性**

局限包括仅评估4种Transformer模型、未覆盖跨模态交互解释、对大规模特征空间的Shapley方法效果不佳等。

---

## 496. From Biological Curves to Soft Modular Robots: Learning-Based Inverse Design with Uncertainty-Guided Refinement

**arXiv ID:** 2610.36681 | [PDF](https://arxiv.org/pdf/2610.36681v1)

**作者:** Dixiang Wang `[一作]`, Luyang Zhao `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Computational Efficiency` `Transformer` `Auto Encoder` `Reinforcement Learning` `Contrastive Learning` `Diffusion model` `Score-based Model` `Generative Adversarial Network` `Tabular` `Time Series` `Biomedical Data`

**🎯 论文内容**

构建了基于学习的逆向设计框架，能将生物曲线映射到软模块机器人配置。

**💡 创新点**

引入前向代理和不确定性引导的候选细化，实现在不做全搜索的前提下高效逆向设计。

**🔧 技术方法**

使用MLP前向代理、两阶段逆向网络、Softmax映射以及基于概率差的候选搜索。

**📊 数据集**

基于物理模型生成的约5M条样本数据集，包括12根骨架的布线、绳长与对应弯曲角度。

**📈 对比分析**

与原始物理搜索及其他学习方法比较，逆向设计成功率从78%提升至93%，运行时间从1.99×10⁶ ms降至4.18 ms，前向代理速度提升至1.9×10⁸倍。

**⚠️ 局限性**

仅针对准静态形状，未考虑重力、摩擦及动力学影响，难以直接预测运动轨迹。

---

## 497. When All Rollouts Fail: Policy-Compatible Joint Decoding for Hard-Problem Reasoning

**arXiv ID:** 2610.36849 | [PDF](https://arxiv.org/pdf/2610.36849v1)

**作者:** Zehao Wang `[一作]` (University of Science and Technology of China), Lin Yang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `Machine Learning` `Reinforcement Learning` `Optimization` `Computational Efficiency` `Representation Learning` `Knowledge Distillation` `Transformer` `Large Language Model` `Reinforcement Learning` `Prompt Engineering` `Mixture of Experts` `Chain-of-Thought` `Retrieval-Augmented Generation` `Text` `Sequential` `Benchmark`

**🎯 论文内容**

提出一种基于策略兼容的强化学习框架，利用教师-学生分布约束联合解码和动态top-k技术，结合混合行为优化，从专家轨迹中恢复可验证的正向奖励信号，以提升大型语言模型的推理性能。

**💡 创新点**

创新点在于：①通过分布约束联合解码与动态top-k实现教师与学生 token 级别的兼容性，②仅在所有 rollouts 失败时触发专家引导，恢复学习信号，③使用重要性采样校正混合行为优化，显著提升探索效率和学习稳定性。

**🔧 技术方法**

采用的核心技术包括：RLVR、GRPO、分布约束联合解码、动态 top‑k 机制、重要性采样（IS ratio）与混合行为优化（PICO）等。

**📊 数据集**

实验使用的主要数据集有：DAPO‑Math‑17K、AIME 2025/26、HMMT 2026、MATH‑500、GPQA‑Diamond、Humanity’s Last Exam 以及 agentic 任务数据集 ALFWorld 与 ScienceWorld。

**📈 对比分析**

与 GRPO、LUFFY、Prefix‑RFT、ReLIFT、RLRT 等基线在数学与 agentic benchmark 上进行对比，平均提升约 4%（数学）和 6.8%（agentic），并在多项难度高的评测任务中显著领先，展示了优越的性能与学习稳定性。

**⚠️ 局限性**

局限性包括：对专家轨迹的可用性和质量依赖较大；动态 top‑k 需要手动设定参数且对不同任务可能需要微调；方法在训练期间仍需教师支持，推理阶段可去除但训练成本高；对极端长文本或非文本任务的适用性尚未充分验证。

---

## 498. Normalized Cuts and Data Clustering via Matrix Fractional Programming

**arXiv ID:** 2610.36616 | [PDF](https://arxiv.org/pdf/2610.36616v1)

**作者:** Yannan Chen `[一作]` (Shenzhen University), Kaiming Shen `[通讯]` (Chinese University of Hong Kong (Shenzhen))

**关键词:** `Machine Learning` `Optimization` `Data-Centric Learning` `Clustering` `MM theory` `Linear search` `Matrix fractional programming` `Sparse anchor graph construction` `Image` `Tabular` `Graph`

**🎯 论文内容**

本文提出一种多维分数规划（FP）框架，将多类归一化割（NCut）问题转化为可通过线性搜索求解的加权二分匹配问题，实现对0‑1离散约束的直接优化。

**💡 创新点**

创新点在于引入多维二次变换的FP方法，克服传统Dinkelbach变换只能处理单比率的局限，并通过MM理论证明算法的单调收敛及局部收敛速度。

**🔧 技术方法**

主要技术包括多维二次变换、MM理论、线性搜索、稀疏锚点图构造，以及与传统谱聚类和其他直接NCut优化方法的比较。

**📊 数据集**

实验使用12个公开数据集（如Shuttle、Breast、MNIST、CIFAR-10、RT-IoT2022等）以及图像分割任务（500个超像素图像），验证方法的有效性。

**📈 对比分析**

通过与两阶段谱聚类方法（SC、ISR、DSSRC）和直接NCut优化方法（FINC、FCD、FPC）的对比，FPC在11/12个数据集上获得最佳或相当的NCut值，在ACC、NMI、ARI等聚类指标上排名最高，并且在运行时间上显著优于基线。

**⚠️ 局限性**

局限性包括：仍可能陷入局部最优，结果对初始化敏感；未覆盖其他图分割准则或在线/分布式场景；在极大规模或高维数据上仍需进一步改进算法效率。

---

## 499. WithinVAE: Compact Visual Autoencoding with Self-Distilled Semantics

**arXiv ID:** 2610.36593 | [PDF](https://arxiv.org/pdf/2610.36593v1)

**作者:** Yuzhou Ji `[一作]` (Shanghai Jiao Tong University), Xuelong Li `[通讯]` (China Telecom)

**关键词:** `Computer Vision and Pattern Recognition` `Compression` `Representation Learning` `Knowledge Distillation` `Auto Encoder` `Transformer` `Auto Encoder` `Generative Adversarial Network` `Contrastive Learning` `Knowledge Distillation` `Image`

**🎯 论文内容**

设计并训练了一种compact视觉自编码器（VAE），通过内部EMA教师实现自监督的语义蒸馏，并在token序列上使用位置依赖的遮蔽，实现了高压缩率与语义结构的统一。

**💡 创新点**

①无需外部预训练的识别网络即可获得语义结构；②通过自蒸馏的全局（DINO样式）和局部（iBOT样式）损失将语义监督内嵌在同一模型；③位置依赖遮蔽使token序列天然排序，支持无重训练的前缀解码。

**🔧 技术方法**

变分自编码器与一维序列瓶颈、EMA教师‑学生自蒸馏（全局+局部）、位置依赖token遮蔽、投影层与原型头、生成器对抗损失、感知损失、One‑D‑Piece 初始化以及后续的 LightningDiT 生成器。

**📊 数据集**

ImageNet‑1K（256×256）为主要训练与评估数据集；还使用 Qwen‑Image latent 作为跨接口的辅助实验。

**📈 对比分析**

与 RPiAE 及 SD‑VAE、VA‑VAE 等基线在同一一维序列接口下对比。结果显示在仅占 RPiAE 一半标量容量的情况下，PSNR 提升 1.36 dB，rFID 降 0.04，LPIPS 降 0.057，SSIM 提升 0.068；前缀解码（128 token）仍优于完整 RPiAE；冻结线性探测精度提升 6.5 点。生成实验中，LightningDiT 在无引导时 recall 超过 RPiAE，但 gFID 与 IS 仍落后，表明语义提升并不自动带来更优生成。

**⚠️ 局限性**

①模型容量以连续标量计数，未实现可变比特率编码；②语义评估主要基于全局分类指标，缺乏对单个 token 语义贡献的可解释性；③自蒸馏在高压缩下仍难以匹配外部预训练语义；④生成器与自编码器的分布匹配问题仍未解决，导致生成质量受限。

---

## 500. Your Wish Is My Commit: Over-Compliant Refactoring in AI Coding Agents

**arXiv ID:** 2610.36346 | [PDF](https://arxiv.org/pdf/2610.36346v1)

**作者:** Haocong Jiang `[一作]` (University of Southern California), Souti Chattopadhyay `[通讯]` (University of Southern California)

**关键词:** `Software Engineering` `AI Code Assistant` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Code`

**🎯 论文内容**

本文通过对大型语言模型在持续重构任务中的行为进行实验，系统性评估其过度配合（over‑compliance）现象，揭示其对代码产生无益修改、代码膨胀和停止判断失效等问题。

**💡 创新点**

创新点在于提出并量化六个行为信号（S1–S6），以客观衡量模型在无终点重构任务中持续编辑的程度，并首次将这些信号与代码质量、结构变动等维度关联，形成可用于工具检测和改进的评估框架。

**🔧 技术方法**

使用的技术包括基于RefDiff的代码结构树（CST）差异分析、行级 diff 统计、自然语言处理的关键词分类器（用于判定“停”语句），以及对多轮交互中的模型推理和生成结果进行自动化日志与度量。

**📊 数据集**

实验数据集为60个 Java 代码库，分别为50个算法问题的参考实现和10个面向对象的学生作业，另外还扩展到4个开源大规模项目做消融验证。

**📈 对比分析**

通过在 Codex 与 Claude Code 这两个主流编码代理上，分别采用5种模型（GPT‑5.4/5.4‑mini/5.5、Claude Sonnet 4.6/Opus 4.8）进行10轮重构，比较其首次停止轮次、前后停歇编辑量、停后编辑、结构回滚、重新实现以及编辑局部性等指标；结果显示所有模型均未能及时停止，GPT 系列往往停在第10轮，Claude 系列则停在第3–7轮，但即使停后也会继续编辑，整体表现表明过度配合是普遍存在且难以消除的。

**⚠️ 局限性**

局限性包括：① 仅覆盖 Java 语言，无法直接推广到动态或函数式语言；② 只考虑了持续“请再重构”这种极端用户行为，未涵盖更真实的多样化交互场景；③ 采用的 RefDiff 只捕获方法级别变动，忽略局部语句级别细节；④ 未评估多轮编辑后代码的功能正确性；⑤ 结果基于有限数量的实验跑，随机性可能影响细节统计。

---

## 501. Pricing Cross-Bank Context in Federated Anti-Money Laundering

**arXiv ID:** 2610.35996 | [PDF](https://arxiv.org/pdf/2610.35996v1)

**作者:** Lidia Losavio `[一作]` (Università della Svizzera italiana), Marc Langheinrich `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 7187 | [OpenAlex ID](https://openalex.org/A5069422623)

**关键词:** `Cryptography and Security` `Federated Learning` `Anomaly Detection` `Safty and Privacy` `Graph Neural Network` `FedAvg` `Graph` `Tabular` `Time Series` `Finance Related`

**🎯 论文内容**

本文探讨了跨行反洗钱（AML）中如何在联邦学习框架下利用已记录的跨行交易信息进行图神经网络（GNN）训练，评估不同程度的跨行信息披露对检测性能的影响。

**💡 创新点**

创新点在于提出三种渐进的跨行信息接口（无额外负载、仅传输交易计数、传输边界状态），并系统衡量其在不同分区粒度下的增益以及在加入正式隐私保护后的实用性。

**🔧 技术方法**

使用了多种面向 AML 的 GNN 基础架构（GIN、PNA、TransConv、SALT-Trans、SALT-GAT、FraudGT）以及 FedAvg 联邦学习协议，并结合隐私保护机制（Gaussian 噪声、几何噪声）进行实验。

**📊 数据集**

实验数据集为 AMLworld HI‑Small，采用其标准时间序列划分和转导式协议，在不同银行数量（4、8、16、32）下进行测试。

**📈 对比分析**

与本地单行训练、边缘丢弃 FedAvg、以及中心化训练进行对比，发现可观测切片在大部分模型和分区下保持最优性能；添加交易计数在多数情况下可与中心化相当；边界状态在特定情境下有提升，但在加入隐私保护后失效。

**⚠️ 局限性**

局限性包括仅针对交易驱动的模型，未考虑多重边特定 GNN 改进；隐私保护实验仅在单一释放场景下评估，未覆盖整个联邦训练过程；实际银行环境的验证仍待进一步研究。

---

## 502. A Serious-Game Testbed for Self-Governing Human-Machine Institutions

**arXiv ID:** 2610.36989 | [PDF](https://arxiv.org/pdf/2610.36989v1)

**作者:** Callum Berger `[一作]` (University of Lincoln), Stefan Sarkadi `[通讯]` (University of Lincoln)

**关键词:** `Multiagent Systems` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Optimization` `Federated Learning` `Robotic Intelligence` `Explainability and Interpretability` `Computational Efficiency` `Knowledge Distillation` `Representation Learning` `Adversarial Attack` `Hyperparameter Search` `Data-Centric Learning` `Meta Learning` `Drug Discovery` `AI Code Assistant` `Neural Architecture Search` `Protein Structure Prediction` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `World Model` `Text` `Sequential` `Review/Survey Paper`

**🎯 论文内容**

在一个多代理仿真/serious‑game 平台 Megabike 上，使用 LLM 代理在共享一辆巨型自行车的条件下进行资源管理、规则制定与执行实验；通过记录每个座位的决策、行动与惩罚，研究其自治理能力。

**💡 创新点**

提供了一个可视化、可审计的混合机构实验环境，使得规则与行动都被显式记录并可被强制执行；实现了人机同座、同体制的治理实验；首次在 LLM 集体中评估制度弹性与治理稳定性。

**🔧 技术方法**

Unity 引擎 + LLMUnity 语言模型集成（Qwen3.5‑4B），多代理神经‑符号架构（类似 JaCaMo），符号化的规则系统与审计机制，有限句子自然语言交互与决策流程。

**📊 数据集**

实验日志（40 场次）与生成的对话、动作、规则数据；使用 Qwen3.5‑4B 语言模型权重作为模型来源；未使用公开 NLP 语料，只在内部模拟中产生数据。

**📈 对比分析**

采用 Kruskal‑Wallis、Holm 校正的非参数比较，评估不同人格倍增条件（Cooperative, Lazy, Subversive, Opposing）下的存活轮数、努力程度、惩罚效果和规则稳定性；结果显示 Cooperative 条件提升努力与存活，Lazy 条件导致更低努力与更多冲突；但规则未被修改，惩罚与违规者不匹配，表明制度弹性不足。

**⚠️ 局限性**

实验仅包含机器代理，未引入人类；使用的小模型限制了语言表达与推理能力；动作空间有限，未开启机车间转移与审计成本；游戏时间仅5轮，难以充分检验制度弹性；数据量相对有限，外推性受限。

---

## 503. LM-DSB: A Lightweight Multi-agent System for Automated Daily Safety Bulletin Generation in Construction

**arXiv ID:** 2610.36433 | [PDF](https://arxiv.org/pdf/2610.36433v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Artificial Intelligence`

---

## 504. FORESEE-MS: Toward Prospective Spatial Forecasting of Future Multiple Sclerosis Lesions from Baseline FLAIR MRI

**arXiv ID:** 2610.36428 | [PDF](https://arxiv.org/pdf/2610.36428v1)

**作者:** Maryam Akhavan Aghdam `[一作]` (University of Alabama), Shahram Rahimi `[通讯]` (University of Alabama)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Convolutional Neural Network` `Transformer` `Auto Encoder` `Contrastive Learning` `Data Augmentation` `Biomedical Data` `Magnetic Resonance Imaging`

**🎯 论文内容**

本文提出了FORESEE-MS评估协议，利用基线FLAIR MRI预测未来多发性硬化病灶的位置，并将该方法与基于群体先验的脑部病灶频率图进行比较。

**💡 创新点**

创新点在于构建折叠特定的 population-prior atlas 作为基线，并结合 voxel‑level AUROC/AUPRC 与 Recall@Top‑K% 两种指标，系统评估基线 MRI 的预测潜力。

**🔧 技术方法**

使用了五种深度学习模型（3D U‑Net、Attn+ResBlock+VAE、轻量级 3D ViT、SwinUNETR 冻结版与微调版），配合距离到脑边界特征、数据增强和 BCE+Tversky 损失等技术。

**📊 数据集**

实验数据来源于 MSSEG‑2 数据集，仅包含 39 名训练样本（28 名出现新病灶），每名受试者提供基线与随访 FLAIR 图像及新病灶掩模。

**📈 对比分析**

与 atlas 的比较显示，atlas 在 pooled voxel‑level AUROC（0.747）和 AUPRC（1.24×10⁻³）上表现最佳，深度学习模型未能显著优于 atlas；在 Recall@Top‑K% 方面，3D U‑Net 与 Attn+ResBlock+VAE 在 K=1%、5%、10%、20% 上略高于 atlas，但多数置信区间包含零，差异不显著。

**⚠️ 局限性**

研究局限主要包括样本量小（仅 39 名受试者），导致统计置信区间宽广；atlas 的构建受限于约 31 名训练者，可能不够代表性；未整合治疗或临床信息；以及难以区分基线已有病灶与潜在病灶的差异。

---

## 505. Calibrated for Which Policy? Conformal Uncertainty for Fixed World Models Under Intervention

**arXiv ID:** 2610.35877 | [PDF](https://arxiv.org/pdf/2610.35877v1)

**作者:** Ayush Mittal `[一作]` (Noumenal), Dhruv Gupta `[通讯]` (Indian Institute of Science)

**关键词:** `Machine Learning` `Reinforcement Learning` `Robotic Intelligence` `World Model` `World Model` `Contrastive Learning` `Reinforcement Learning` `Conformal Inference` `Tabular` `Time Series` `Sequential`

**🎯 论文内容**

研究了在部署策略变化时固定世界模型的校准问题，提出通过估计比例权重实现权重 conformal 校准，以保持目标策略下的覆盖率。

**💡 创新点**

创新点在于：1) 将政策干预下的 likelihood ratio 分解为占据率与动作倾向；2) 用域分类器估计比例，并给出覆盖下界与比例误差与 log‑loss 误差的紧确关系；3) 用 Lean 4 进行机器验证的完整证明。

**🔧 技术方法**

使用技术包括权重 conformal 推断、比例估计（域分类器与后验估计）、总变差与 log‑loss 误差分析、机器验证框架 Lean 4，以及 DINO‑WM 预训练的世界模型。

**📊 数据集**

实验数据来自 Franka 机械臂的 192 条物理一跳转移，使用预训练的 DINO‑WM 模型；构造了控制的分布 Pλ 与 Qλ 以模拟不同的政策漂移。

**📈 对比分析**

比较了普通源策略 conformal、oracle 比例权重、学习比例权重以及裁剪比例权重。结果显示：普通方法覆盖率随 λ 增大显著下降，而 oracle 与学习比例权重在 λ=1.5 时仍保持约 90.9% 的覆盖率，说明学习比例能有效恢复目标覆盖。

**⚠️ 局限性**

局限性包括：仅研究一跳、边际覆盖，要求共享条件动力学和目标支持在源内；未处理机制漂移或轨迹级覆盖；实验仅在受控的 192 条过渡上验证，真实环境中可能表现不同。

---

## 506. SAGE: Saliency-Annealed Gated Encoder for Universal Vision Recognition

**arXiv ID:** 2610.36719 | [PDF](https://arxiv.org/pdf/2610.36719v1)

**作者:** Le Yang `[一作]` (Huawei Technologies Co., Ltd.), Qiang Zhao `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Object Detection` `Segmentation` `Recognition` `Transformer` `Contrastive Learning` `Auto Encoder` `Varifocal Loss` `SyncBN` `MSDeformAttn` `Laplacian High-Frequency Enhancement` `FiLM` `Image`

**🎯 论文内容**

提出 SAGE，一种在 ViT-Adapter 基础上加入显著性引导、热图增强、注册 token FiLM、高频细节恢复等模块的改进型 ViT 骨干；

**💡 创新点**

三大创新：①显著性引导残差软门控结合 CLS‑token saliency；②基于温度的多层聚合和局部细节分支；③训练中使用三阶段余热衰减与两阶段从 Focal 到 VFL 的质量感知策略；

**🔧 技术方法**

使用自监督 ViT 预训练（Objects365/DINOv2 等）、DINO‑style saliency、FiLM、Laplacian 高频增强、MSDeformAttn、SyncBN、Varifocal Loss、语义一致性损失等技术；

**📊 数据集**

在 COCO（目标检测、实例分割、关键点）、ADE20K（语义/实例分割）、TinyImageNet（分类）、DIOR‑R（方向检测）等公开基准上进行评测；

**📈 对比分析**

与现有 DEtr/ViT-Adapter/Co‑DETR/RF‑DETR 等方法对比，单尺度、无 TTA 下，SAGE 在 COCO box AP 62.8、mask AP 55.3、TinyImageNet Top‑1 94.76%、ADE20K mIoU 60.3、DIOR‑R AP_50 78.8 等任务均达到或超过 SOTA；

**⚠️ 局限性**

局限性包括：需额外的显著性、频率与注册 token 模块增加模型复杂度；不同任务使用的预训练/训练细节未统一，跨论文比较存在一定偏差；对更大规模/多模态任务的验证尚未展开。

---

## 507. Physiology-Informed Audio Quality and Listening Effort Estimation

**arXiv ID:** 2610.36865 | [PDF](https://arxiv.org/pdf/2610.36865v1)

**作者:** Niall McGuire `[一作]` (University of Strathclyde), Dimitra Emmanouilidou `[通讯]` (Microsoft Research)

**关键词:** `Sound` `Classification` `Recognition` `Anomaly Detection` `Explainability and Interpretability` `Convolutional Neural Network` `Recurrent Neural Network` `Transformer` `Auto Encoder` `Supervised Fine-Tuning` `Contrastive Learning` `Diffusion model` `Score-based Model` `Audio` `Multimodality` `Time Series` `Biomedical Data` `Electrocardiogram` `Fibre Orientation Distribution` `Diffusion Tensor Imaging`

**🎯 论文内容**

本研究收集实验室高密度EEG、可穿戴EEG、fNIRS和眼动仪等多模态生理信号，评估其在音频质量与聆听努力解码中的可行性。

**💡 创新点**

创新点在于首次同时比较实验室级和消费者级设备在音频质量与聆听努力解码上的表现，并通过个体校准展示了可穿戴设备在无新受试者情况下可解码的音频质量趋势。

**🔧 技术方法**

使用机器学习分类与回归（SVM、XGBoost、MLP等）结合频域特征、功率谱、熵、眼动指标和fNIRS血氧变化进行解码。

**📊 数据集**

数据集包括NISQA语音质量语料、ODAQ音乐语料以及对比的清晰/失真样本，共144段音频（102语音+42音乐），并使用标准的MOS标签。

**📈 对比分析**

在Leave-One-Subject-Out和Within-Subject两种评估方案下，穿戴EEG、fNIRS和眼动仪在二分类任务上达0.60–0.65的准确率，回归任务与MOS的相关系数约0.38–0.43；实验室EEG未超过偶然水平。

**⚠️ 局限性**

局限性包括实验室EEG样本量少、信号质量受干扰、未检验对新音频的泛化、仅使用固定音频集、未解决个体质量排序问题。

---

## 508. Proposing Better Rollouts for On-Policy Distillation

**arXiv ID:** 2610.36477 | [PDF](https://arxiv.org/pdf/2610.36477v1)

**作者:** Sirui Chen `[一作]` (University of Illinois Urbana Champaign), Jingrui He `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `Machine Learning` `Knowledge Distillation` `Reinforcement Learning from Human Feedback` `Optimization` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Mixture of Experts` `Contrastive Learning` `Text` `Biomedical Data` `Benchmark`

**🎯 论文内容**

在大语言模型的后训练中，提出了Mixture Proposal On-Policy Distillation (MPOPD)，通过在本地信赖域下用相对教师支持度重加权学生策略，得到学生与教师的几何混合作为回采分布，并用可学习的提议策略一次前向推理生成轨迹，从而避免传统OPD中学生轨迹漂移到教师难以覆盖的低质量区域，导致教师监督嘈杂或误导的问题。

**💡 创新点**

创新点在于：①把回采策略本身视为优化目标，用相对教师支持度和KL信赖域构造局部重加权，得到几何混合回采分布；②用可学习的提议策略逼近该几何混合，避免每一步都同步跑教师和学生，保持了OPD的在线性时间特性；③通过实验验证该方法能显著减少教师-学生不匹配、缩短生成长度、提升多项推理任务性能。

**🔧 技术方法**

核心技术包括：局部信赖域优化、相对教师支持度重加权、几何混合回采策略、可学习的提议策略、重要性加权的学生更新以及KL距离约束；实现基于vLLM与GH200 GPU。

**📊 数据集**

评测数据集：16个数学推理、知识问答与医学推理基准（AIME、AMC、Minerva、OlympiadBench、GPQA-Diamond、ARC-Challenge、MMLU-Pro、PubMed、MedQA、Lancet、NEJM 等），使用 Qwen3-1.7B-Base、Qwen3-4B-Base 作为学生模型，教师为 Qwen3-4B-Instruct-2507。

**📈 对比分析**

与标准OPD、ExOPD、EOPD、VETO四个最近的变体进行对比。实验显示在 Qwen3-4B-Base 上数学推理平均准确率提升约 +4.3%，单任务提升可达 +6.9%；在 16 个基准中保持所有任务最优，生成长度平均下降 41%；在知识问答与医学推理同样获得 1–2% 的准确率提升，生成长度显著减少。总体来看，MPOPD 在精度-速度-长度方面均优于对照方法。

**⚠️ 局限性**

局限性包括：①需要调节倾斜强度 η，虽然对结果影响有限但仍需经验选择；②额外的提议策略训练会略微增加显存占用与训练时间；③方法假设教师对轨迹的支持度可靠，若教师自身产生噪声或缺乏覆盖，重加权效果可能受限；④未在更大规模模型或更广泛任务（如对话、程序合成）上进行验证。

---

## 509. Gradient flow of magnetic inclination and intensity as a GPS-free guidance mechanism

**arXiv ID:** 2610.36812 | [PDF](https://arxiv.org/pdf/2610.36812v1)

**作者:** Zayed Herma `[一作]` (University of Grenoble Alpes), Andréa Vassilev `[通讯]` (University of Grenoble Alpes)

**关键词:** `Robotics` `Autonomous Driving` `Optimization` `Reinforcement Learning from Human Feedback` `Simultaneous Localization and Mapping` `Gradient flow` `Tabular` `Time Series` `Physics Related`

**🎯 论文内容**

提出利用地磁倾角与强度的梯度流控制实现无GPS磁导航，并通过预设中间航点引导系统到达目标地点。

**💡 创新点**

创新点在于将磁导航问题转化为梯度流控制，并证明在预定义区域内磁倾角与强度足以提供定位信息。

**🔧 技术方法**

使用控制理论中的梯度流方法、地磁场模型（如IGRF/ WMM）以及路径规划与航点跟踪算法。

**📊 数据集**

主要使用全球地磁场模型数据（IGRF）及基于仿真生成的路径场景进行验证。

**📈 对比分析**

与传统GPS导航在仿真环境中对比，表现出相近的航向误差，且在无卫星信号时仍能完成目标定位。

**⚠️ 局限性**

局限性包括磁场映射非一一对应导致定位模糊、仅适用于磁特征差异明显的区域，并且对磁噪声与外部干扰较为敏感。

---

## 510. Log-To-Leak: Prompt Injection Attacks on Tool-Using LLM Agents via the Model Context Protocol

**arXiv ID:** 2610.36108 | [PDF](https://arxiv.org/pdf/2610.36108v1)

**作者:** Yuepeng Hu `[一作]` (Duke University), Jian Du `[通讯]` (TikTok)

**关键词:** `Cryptography and Security` `Adversarial Attack` `Safty and Privacy` `Transformer` `Large Language Model` `Prompt Engineering` `Agentic AI` `Text` `Benchmark`

**🎯 论文内容**

本文研究了一种在Model Context Protocol（MCP）工具调用中注入后置日志记录的攻击，并提出了由Trigger、Tool Binding、Justification、Pressure四个自然语言组件构成的系统化诱导模板。

**💡 创新点**

创新点在于将工具元数据注入细分为四个明确的诱导组件，并通过结构化模板在保持任务完成的前提下实现高效的后置日志记录，首次量化评估此类工具注入攻击的成功率、日志准确性和对任务执行的影响。

**🔧 技术方法**

使用了MCP协议工具注入技术、自然语言诱导模板、评估指标（攻击成功率ASR、日志准确率LA、目标任务完成率变化Δ_TCR、恶意服务器完成率变化Δ_MCR、Token使用变化Δ_TU）以及针对不同模型与服务器的实验对比。

**📊 数据集**

实验数据来自5个MCP服务器（GitHub、Mapbox、PayPal、Playwright、YFinance）的共555个任务，其中包括自定义任务集和MCP-Universe提供的任务。

**📈 对比分析**

与基线诱导方法相比，该攻击在18个兼容模型-服务器组合中实现62.6–100% ASR、77–95% LA，任务完成率差异不超过6.6个百分点；在GPT-5等模型上进一步显著提升日志准确率；但在某些组合下仍存在较大Token消耗或任务完成率下降。

**⚠️ 局限性**

局限性包括实验仅覆盖有限的模型与服务器，攻击假设攻击者已控制并注入服务器元数据；评估指标对语义泄露的捕捉不全面，防御方法在实际部署中仍未能完全抑制攻击，且模型更新、客户端差异等因素可能导致结果变化。

---

## 511. Holomorphic Cauchy Networks for Efficient Stationary Tail Computation in High-Dimensional Reflected Brownian Motions

**arXiv ID:** 2610.36558 | [PDF](https://arxiv.org/pdf/2610.36558v1)

**作者:** Kehan Wang `[一作]` (Chinese University of Hong Kong), Kehan Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `Machine Learning` `Computational Efficiency` `Optimization` `Physics Related` `Stochastic Differential Equation` `Score-based Model` `Auto Encoder` `Contrastive Learning` `Tabular` `Physics Related`

**🎯 论文内容**

提出 Holomorphic Cauchy Networks（HCN），在 NN4MGF 框架下学习高维反射布朗运动（RBM）的稳态 Laplace 变换，从而直接计算尾概率。

**💡 创新点**

创新点包括：
- 将全纯性从损失函数迁移至网络架构，使用 Cauchy 激活实现无 Cauchy–Riemann 罚；
- 通过原点锚定和边界坐标排除实现结构化归一化与面独立性；
- 引入交互子网络捕捉相关系统；
- 采用闭式导数、顺序反向传播和维度自适应批量调度，显著降低 GPU 内存与时间。

**🔧 技术方法**

技术方法：
- 复杂数值网络与 Cauchy 复激活；
- 共享坐标与边界表示；
- 原点锚定、边界坐标排除；
- 交互子网络实现跨坐标交互；
- 计算闭式导数与顺序反向传播；
- 采用 Fixed‑Talbot Laplace 逆变换评估尾概率。

**📊 数据集**

数据集与实验：
- 采用 RBM 参数（μ、Σ、R）作为训练输入；
- 主要合成基准包括 2D、20D、30D、50D 产品型、2D Harrison 非产品、10D 相关非产品等；
- 参考结果使用理论解析、已知 Laplace 变换或仿真估计。

**📈 对比分析**

性能对比：
- 在 30D 实验中，HCN 的平均更新时间比 NN4MGF 减少 9.7 倍、峰值 GPU 内存减少约 10 倍；
- 在 2D、20D、30D、50D 上，HCN 的尾误差均低于或接近 NN4MGF（例如 2D Harrison 误差从 6.68% 降至 1.42%）；
- 10D 相关案例中，HCN 的误差显著优于独立模型，表明交互子网络能有效捕捉跨坐标依赖。

**⚠️ 局限性**

局限性：
- 未给出固定宽度下训练收敛的理论保证；
- 对全局概率测度的完整性与可行性仅依赖经验验证；
- Pole 分布与 Fixed‑Talbot 节点的关系尚未建立，可能影响逆变换精度；
- 缺乏大规模相关 RBM 的高精度仿真基准；
- 仅在单 GPU 环境下验证，尚未探索多 GPU/分布式训练效果。

---

## 512. PRISM: Pathology Reliability In Scarce-label Medicine. A Tile-level Benchmark of Pathology Foundation Models

**arXiv ID:** 2610.36517 | [PDF](https://arxiv.org/pdf/2610.36517v1)

**作者:** Alicem Koyun `[一作]` (Westcliff University), Alicem Koyun `[通讯]` (Westcliff University)

**关键词:** `Machine Learning` `Classification` `Anomaly Detection` `Domain Adaptation` `Transformer` `Contrastive Learning` `Temperature Scaling` `L2 Regularization` `Image` `Biomedical Data` `Benchmark`

**🎯 论文内容**

提出PRISM基准，用于评估病理基础模型在不同标签预算、校准性、跨机构迁移四个维度下的可靠性；

**💡 创新点**

首次将模型选择、标签比例、校准误差与跨域稳健性统一评估，并引入Composite Reliability Index（CRI）衡量多维度性能；

**🔧 技术方法**

采用冻结嵌入、L2正则化逻辑回归探针，配合温度缩放（temperature scaling）进行后期校准；

**📊 数据集**

使用六个病理切片级数据集（PCam、CRC、MHIST、BRACS、SPIDER-Breast、LungHist700）及Camelyon17的五医院跨机构对；

**📈 对比分析**

通过864个探针实验和3600个跨医院迁移实验，对八种基础模型（CLIP、PLIP、CONCH、UNI、VIRCHOW2、GigaPath、H-Optimus-0、MIDNIGHT）在不同标签比例下评估AUROC、ECE、Brier以及CRI；结果显示：在低标签（1%）时AUROC与校准排名解耦，探针会在某些数据集出现单类崩溃，迁移时标签增多有时会恶化校准，温度缩放并非始终有效；

**⚠️ 局限性**

局限性：仅针对切片级任务、冻结嵌入、单一探针正则化；未涵盖幻灯片级评估；CRI仅为筛选工具，需在实际临床数据上进一步验证。

---

## 513. EMMA: Structured Episodic Memory for Multi-Agent Workflow Synthesis

**arXiv ID:** 2610.35966 | [PDF](https://arxiv.org/pdf/2610.35966v1)

**作者:** Ruijia Niu `[一作]` (University of California San Diego), Kristina Klinkner `[通讯]` (Amazon)

**关键词:** `Multiagent Systems` `Reinforcement Learning from Human Feedback` `Optimization` `Autonomous Driving` `Robotic Intelligence` `Federated Learning` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Mixture of Experts` `Text` `Tabular` `Benchmark`

**🎯 论文内容**

提出了 EMMA——一种将多代理系统配置视为分层记忆的结构化经验学习框架，能够在每次失败后通过一次受限编辑来更新记忆；

**💡 创新点**

创新点在于：①为记忆分层（控制器记忆、代理池、技能池）并为每个层级制定显式的更新范围；②利用反射器对失败进行定位、分类、泛化和最小化，仅产生单一受限编辑；③加入程序化安全门（范围兼容、提示保持、内存增长限制、泄漏防止、策略更改）保证更新不会偏离训练反馈；

**🔧 技术方法**

使用技术包括：多代理编排（Magnetic‑One风格）、层次化经验记忆、单步合成与溢出拆分、结构化反射与范围归属、受限编辑与安全门、LLM骨干作为控制器、代理与反射器；

**📊 数据集**

实验数据集：KramaBench（28训练/76测试），AppWorld（90训练/168/417测试），BBEH‑mini（92训练/368测试）；

**📈 对比分析**

与ACE、GEPA、EvoMAS、ADAS四个基线以及五种LLM骨干(Opus 5、Sonnet 5、GLM‑5、DeepSeek v3.2、Kimi‑K2.5)比较，EMMA在所有基准上获得最高平均性能，排名17/20场景第一，最强基线提升可达32.5%，且训练执行次数比最优基线少17–38%；

**⚠️ 局限性**

局限性包括：与骨干族群的性能差距仍未消除；接受的编辑在任务失败后仍会保留，可能留下无效或有害条目；缺乏跨批次或多轮验证机制；以及对弱执行器代理的依赖可进一步改进。

---

## 514. OpenGray: a benchmark for evaluating the decisions of radiotherapy planning agents

**arXiv ID:** 2610.37081 | [PDF](https://arxiv.org/pdf/2610.37081v1)

**作者:** Humza Nusrat `[一作]` (Henry Ford Health), Kundan Thind `[通讯]` (Henry Ford Health)

**关键词:** `Artificial Intelligence` `Benchmark` `Transformer` `Large Language Model` `Agentic AI` `Biomedical Data` `Radiotherapy Planning` `Benchmark`

**🎯 论文内容**

建立了OpenGray基准，用于评估放疗规划代理的决策，并对比了四种系统（规则脚本、加算检查、解释器和完整代理）在30例头颈病例上的表现，以及对SAGE临床日志的审计。

**💡 创新点**

首次提供具有已知正确答案的放疗规划决策基准，揭示仅凭剂量指标难以捕捉的决策失误，并强调决策准确性的重要性。

**🔧 技术方法**

采用大型语言模型（Terra、Mistral 3 8B）与规则脚本相结合的多阶段规划框架，并利用dose‑influence矩阵重算剂量以评估计划质量。

**📊 数据集**

基于OpenKBP‑Opt头颈病例数据集（100例），其中10例用于开发、30例用于评估，并使用39例SAGE的临床日志进行验证。

**📈 对比分析**

通过决策准确率、剂量达标率、问题识别率和技术失败率四项指标比较，Terra的LLM系统决策准确率超过90%，但在重叠冲突任务上错误率高，所有系统提交的可行控制计划有约24%未满足硬目标。

**⚠️ 局限性**

局限于单一头颈数据集、两种冲突类型、未考虑机具约束，且基准可能被针对性调优，导致结果在更广泛场景中的可推广性受限。

---

## 515. Tessera: Structure-Completed Evidence Units for Web Retrieval-Augmented Generation

**arXiv ID:** 2610.37080 | [PDF](https://arxiv.org/pdf/2610.37080v1)

**作者:** Peichun Hua `[一作]` (Chinese University of Hong Kong Shenzhen), Yunming Xiao `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `Information Retrieval` `Retrieval` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Retrieval-Augmented Generation` `Prompt Engineering` `Agentic AI` `Text` `Image` `Tabular`

**🎯 论文内容**

设计并实现了一层训练‑free 的结构补全单元（Structure-Completed Evidence Units）层，解耦检索与读取，在检索后对页面 HTML 进行解析，将表格行、列表项、段落等最小单元与其标题、表头、章节路径等上下文一起打包，填入读取器的 token 预算，从而让读取器看到完整、有语义的证据单元；

**💡 创新点**

首次提出在查询时动态重新构造网页结构并以最小单元形式重新投递给读取器的方案，完全不需要额外训练或重建索引，且可同时兼容文本、截图以及多跳搜索代理的不同 RAG 流程；

**🔧 技术方法**

采用 DOM 递归遍历、表格网格重建、标题路径与列头关联、语义预筛选（前 100 单元）+跨编码器评分、按预算分组打包等技术，利用现有检索器（BGE、Qwen3-VL 等）和阅读器（Qwen3.5‑4B、Gemma 等），实现无训练、无索引改动的轻量化插件；

**📊 数据集**

使用 2025 年 8 月版英文维基百科快照（约 706 万条目）作为索引语料，评估集包括 Natural Questions、NQ‑Tables、SimpleQA（共 7,851 题）以及多跳搜索代理的 FRAMES（824 题）和 MuSiQue（1,000 题）等；

**📈 对比分析**

在相同 token 预算下与 14 种检索、上下文扩展与结构感知分块方法对比，全部使用同一检索器、阅读器和评测标准；在文本、截图和 agentic pipelines 上均提升 5‑6 分，agentic 搜索准确率提升 5.2 分且搜索次数和上下文长度均下降；

**⚠️ 局限性**

局限性包括：仅适用于已抓取且可直接解析的静态 HTML 页面，未覆盖动态渲染或非维基百科网页；依赖锚点页面和词法预筛选，仍有大量错误源自未锚页面或预筛选遗漏；对语义匹配不足导致部分单元无法被捕获。

---

## 516. GraftRay: Physically Based Mesh Insertion into Gaussian Splatting Scenes via Path Tracing

**arXiv ID:** 2610.37036 | [PDF](https://arxiv.org/pdf/2610.37036v1)

**作者:** Jinxin Ai `[一作]` (Technical University of Munich), Matthias Niessner `[通讯]` (Technical University of Munich)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Restoration` `Optimization` `Computational Efficiency` `Diffusion model` `Transformer` `Diffusion model` `Gaussian Splatting` `Path Tracing` `Optical Flow` `Non-negative Least Squares` `MIS (Multiple Importance Sampling)` `Image` `Mesh` `Point Cloud` `3DGS (3D Gaussian Splatting)` `Physics Related`

**🎯 论文内容**

提出 GraftRay 框架，实现将网格对象在 2D/3D 高斯剖分（Gaussian Splatting）场景中进行物理一致的插入，恢复材质、面积光源并使用差分合成保持原场景细节。

**💡 创新点**

创新点在于：① 用扩散模型直接从 LDR 视图估计材质与法向并投影到高斯上；② 通过用户标记的光源区域自动构造面积光源；③ 在 OptiX 中实现随机透明度的高斯路径追踪，支持光照、阴影与全局照明；④ 用受限观测的非负最小二乘求解光源辐射；⑤ 差分渲染将光照变化迁移回原始高斯渲染。

**🔧 技术方法**

技术包括扩散式材质/法向估计（RGB↔X、DiffusionRenderer）、OptiX 加速的随机透明度路径追踪、解析面积光源、MIS、受限 LDR 的非负最小二乘光源估计、以及差分合成。

**📊 数据集**

使用合成场景（FIPT、NFINITE）、真实室内拍摄（FIPT‑Real、ScanNet++）以及 SpAItial 生成的 3DGS 场景进行实验。

**📈 对比分析**

与 ComGS、EAG‑PT、GaSLight、MV‑CoLight 等方法在 FIPT 与 NFINITE 上进行量化对比，采用 ΔPSNR、mask PSNR、阴影 IoU、RMSE 等指标；在真实/生成场景中使用 VQAScore 比较。结果显示 GraftRay 在插入光照、阴影和整体渲染质量上均达到或超过现有最佳方案。

**⚠️ 局限性**

局限性：需人工用 SAM 标记光源；依赖扩散模型提供的材质/法向先验，先验误差会传递到光源估计和插入结果。

---

## 517. Crop Yield Prediction Using Regression Models: A Comparative Evaluation of Model Performance

**arXiv ID:** 2610.37022 | [PDF](https://arxiv.org/pdf/2610.37022v1)

**作者:** Saptak Das `[一作]` (Indian Institute of Science Education and Research), Suborno Naskar `[通讯]` (MCKV Institute of Engineering)

**关键词:** `Machine Learning` `Agriculture Related` `Supervised Fine-Tuning` `Hyperparameter Search` `Feature Selection` `Tabular`

**🎯 论文内容**

本文通过对包含环境、作物和农业管理变量的百万级数据集，比较了五种回归模型（线性回归、岭回归、套索回归、决策树回归和随机森林回归）在80:20和70:30训练/测试比例下的预测性能；

**💡 创新点**

创新点在于系统评估了不同特征编码方式（完整one‑hot与参考类别剔除）、模型正则化以及树模型的泛化差距，并指出在该数据集上线性模型已达最优基准；

**🔧 技术方法**

使用的技术包括数据预处理（去除负值、one‑hot编码、布尔二值化）、特征选择、GridSearchCV/RandomizedSearchCV超参数搜索、R²/MSE/RMSE评估以及训练/测试性能对比；

**📊 数据集**

采用的是Kaggle提供的农业作物产量数据集（1,000,000条记录，10个特征，其中231条负产量已剔除，最终999,769条）；

**📈 对比分析**

比较方法是对同一训练/测试划分下不同模型的R²、MSE、RMSE进行对比；结果显示线性回归、岭回归和套索回归在两种划分下均取得约0.913的R²、约0.501的RMSE；随机森林虽训练R²高达0.987，但测试R²降至约0.907，显示过拟合；决策树在70:30划分下略有提升；

**⚠️ 局限性**

局限性包括：仅评估了单一划分与交叉验证、未检验不同季节/地区的泛化性、未探索更先进的提升方法（如梯度提升），以及对特征重要性分析不足。

---

## 518. Meta Models: Models that Model Learning

**arXiv ID:** 2610.37001 | [PDF](https://arxiv.org/pdf/2610.37001v1)

**作者:** Peihang Li `[一作]`, Ziming Liu `[通讯]`

**关键词:** `Machine Learning` `Meta Learning` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Large Language Model` `Mixture of Experts` `Contrastive Learning` `Tabular` `Time Series`

**🎯 论文内容**

本文通过把训练日志视作原生数值数据，利用冻结的TabPFN作为meta‑model，预测语言模型训练的最终损失、学习曲线、内部动态和下游性能；

**💡 创新点**

创新点在于证明了无更新权重的tabular in‑context 学习模型即可成为通用训练meta‑model，并通过KAN提升预测精度，同时完成了对公开大规模训练跑的前瞻性预测；

**🔧 技术方法**

核心技术包括TabPFN（预训练的tabular PFN）、Kolmogorov‑Arnold 网络（KAN）用于学习可解释的配置坐标，以及无监督上下文推理；

**📊 数据集**

使用的实验数据集包括Marin、StepLaw、Marin‑MoE、DataDecide、Delphi 超参数扫盲以及Marin Hero 535B 公开训练日志；

**📈 对比分析**

与NCPL、XGBoost、参数化学习法等基线对比，TabPFN在最终损失 MAE 上往往优于或匹配Fine‑tuned LLM，学习曲线预测误差显著降低，KAN增强进一步提升 29%~56% 的预测精度，前瞻预测亦超越官方缩放法；

**⚠️ 局限性**

主要局限在于对未观测的训练条件（如全新数据方案、极端规模）的外推可靠性不足，需要更严格的前瞻评估以验证实际节省计算成本。

---

## 519. The State and Fate of Multilingual, Contextual Evaluation in the NLP World

**arXiv ID:** 2610.36990 | [PDF](https://arxiv.org/pdf/2610.36990v1)

**作者:** Manan Uppadhyay `[一作]` (Microsoft Research India), Sunayana Sitaram `[通讯]` (Microsoft Research India)

**关键词:** `Computation and Language` `Benchmark` `Large Language Model` `Prompt Engineering` `Contrastive Learning` `Text` `Multimodality` `Review/Survey Paper` `Benchmark`

**🎯 论文内容**

对51个最新的多语种评估基准进行了系统的数据驱动审计，涵盖242个数据集、219种语言，并从覆盖率、代表性和严谨性三个维度进行定量分析。

**💡 创新点**

首次将覆盖率、任务公平性、翻译依赖度和文化适配度等指标统一量化，并结合多语言污染检测工具（STAMP与DCQ）系统评估基准污染风险，提出了多语种评估框架改进建议。

**🔧 技术方法**

使用Claude模型自动抽取结构化字段；利用STAMP对翻译版基准做语义保留的重写检验污染；使用DCQ生成词级同义扰动以检测模型记忆；结合手工验证与统计分析完成数据完整性检查。

**📊 数据集**

所审计的数据集包括X-NLI、XQuAD、TyDiQA、AFRIBench、MasakhaNER、IrokoBench、MILU、INCLUDE、Global MMLU等，涵盖各类任务（NLI、QA、NER、情感、常识推理、摘要、编程、道德安全等）。

**📈 对比分析**

对比方法主要是污染检测（STAMP与DCQ）和覆盖/代表性统计；结果显示老旧翻译基准污染率高（如PAWS‑X≈68%，MGSM≈45%），新基准污染率低；任务公平性和文化适配度普遍不足。

**⚠️ 局限性**

局限性包括：仍依赖现有基准且难以覆盖所有语言和文化场景；翻译和重写工具在低资源、跨语言特征上效果有限；污染检测仍受模型偏好与tokenizer的影响；注释者背景透明度低，难以评估真实社区需求。

---

## 520. Correctness Conditioning as Selection in Chain-of-Thought Faithfulness Evaluation

**arXiv ID:** 2610.37070 | [PDF](https://arxiv.org/pdf/2610.37070v1)

**作者:** Yueer Pan `[一作]` (Zhejiang University), Yuexi Pan `[通讯]` (University of Science and Technology of China)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `AI Code Assistant` `Transformer` `Chain-of-Thought` `Large Language Model` `Cochran–Mantel–Haenszel test` `Inverse Probability Weighting` `Causal Inference` `Covariance Decomposition` `Text` `Synthetic Data`

**🎯 论文内容**

研究在 chain‑of‑thought (CoT) 评估中，回答正确性筛选会导致样本偏差，量化了所产生的选择偏差 (selection gap)。

**💡 创新点**

提出基于结构因果模型的理论框架，将选择偏差闭式表示为 Δ = Cov(A,C)/P(C=1)，并区分难度混杂与快捷路碰撞带来的偏差。

**🔧 技术方法**

采用因果推断、协方差分解、Cochran–Mantel–Haenszel 检验、逆概率加权等统计技术来估计与校正偏差。

**📊 数据集**

在人工生成的算术 DAG、布尔电路和图可达等程序化任务（synthetic 数据集）上实验，并对四个生产部署模型和两个 100% 准确度对照进行验证。

**📈 对比分析**

与全轨迹评估相比，正确性筛选下的 Faithfulness 分数普遍上升；GLM‑5.3 在算术任务上出现 +7.4pp 的选择偏差，DeepSeek 等模型亦有显著提升；加权或忽略未解析响应后偏差减小但仍为正。

**⚠️ 局限性**

主要局限在于仅验证于可观测的程序化任务，缺乏自然语言任务中的可验证干预答案；短路使用检测不足，难度混杂的因果识别仍需进一步实验。

---

## 521. A First-Step Bias Restores Lost Diversity in Diffusion Model Distillation

**arXiv ID:** 2610.37110 | [PDF](https://arxiv.org/pdf/2610.37110v1)

**作者:** Wenwu Tang `[一作]`, Olga Saukh `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Knowledge Distillation` `Transformer` `Diffusion model` `Flow-based Model` `Score-based Model` `Contrastive Learning` `Prompt Engineering` `Image` `Video` `Text`

**🎯 论文内容**

对已蒸馏的图像和视频生成模型进行后训练，只在第一步加小型偏置适配器，恢复并提升相同提示下的多样性，同时保持原模型的其它步骤不变。

**💡 创新点**

提出 ReSeed 方法：利用教师模型的第一段平均速度来训练仅在第一步激活的残差偏置，证明仅修改第一步即可显著恢复多样性，并且不需要目标图像、感知或对抗损失。

**🔧 技术方法**

技术包括流匹配（Flow Matching）和分布匹配蒸馏（DMD）基础模型、残差偏置适配器、欧拉积分、平均速度匹配损失、基于 CLIP/DINOv3 的多样性评估。

**📊 数据集**

使用 COCO‑2017 验证集的 5,000 条标题、JourneyDB 训练提示、VBench 视频数据集（320 条提示）以及 FLUX.1‑dev、SD3、Wan2.1 等模型的预训练权重。

**📈 对比分析**

与原始 DMD、GAD、DP‑DMD 等蒸馏方法相比，ReSeed 在 SD3‑Medium/SD3.5‑Medium 的 DINOv3 多样性分别提升约 37.1%/39.3%，FLUX 的多样性提升 48.3%，视频多样性提升 27.1%；同时 FID、LPIPS 等质量指标均保持或略有提升，且仅增加 0.002–0.004% 参数和 <0.5% 推理延迟。

**⚠️ 局限性**

局限性：仅在流匹配变压器架构上验证，未测试非流匹配或 U‑Net 基础模型；训练需要教师在第一步的轨迹，且多样性提升可能在不同任务或更大模型上表现不一致。

---

## 522. Self-Regularized Flow Language Models

**arXiv ID:** 2610.37115 | [PDF](https://arxiv.org/pdf/2610.37115v1)

**作者:** Theodoros Kouzelis `[一作]` (Archimedes, Athena RC), Nikos Komodakis `[通讯]` (Archimedes, Athena RC)

**关键词:** `Machine Learning` `Generation` `Data Synthesis` `Knowledge Distillation` `Transformer` `Diffusion model` `Flow-based Model` `Stochastic Differential Equation` `Ordinary Differential Equation` `Auto Encoder` `Self-supervised Learning` `Knowledge Distillation` `Text`

**🎯 论文内容**

提出 Self-Regularized Flow Language Models (SERF)，通过在连续流式扩散语言模型中引入噪声到干净的自蒸馏，改进训练目标。

**💡 创新点**

创新点是利用同一插值路径上的更干净教师预测作为自监督信号，无需外部模型或修改推理流程。

**🔧 技术方法**

采用EMA教师、自蒸馏（latent和概率分布两种形式）、连续流匹配、流映射、ODE/SDE采样等技术。

**📊 数据集**

使用 OpenWebText、GSM8K 以及在三种连续扩散范式（one-hot、latent、球面）上的数据集。

**📈 对比分析**

与三种基线模型对比，SERF 在训练速度提升至 7.7×、MAUVE 提升 24.3%、GSM8K 准确率提升 28.8%，并可在 64×更少采样步骤下达到基线性能。

**⚠️ 局限性**

局限性包括对噪声级策略敏感、仍需要 EMA 学习率调参、以及在极低温度解码下效果不如离散模型。

---

## 523. RAVEN: A Unified Active SLAM System for Rapid Autonomous Visual Exploration with Aerial Robots under Localization Uncertainty

**arXiv ID:** 2610.37111 | [PDF](https://arxiv.org/pdf/2610.37111v1)

**作者:** Xinyi Chen `[一作]`, Shaojie Shen `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Autonomous Driving` `Simultaneous Localization and Mapping` `Image` `Point Cloud`

**🎯 论文内容**

提出了RAVEN框架，实现基于感知的主动SLAM；

**💡 创新点**

创新点在于将探索效率与定位精度统一平衡的感知感知轨迹规划，并通过可调参数λ实现两者权衡；

**🔧 技术方法**

采用EPEE+仿真环境，集成多摄像头实时渲染、IMU噪声模拟和VINS+闭环检测的SLAM模块；

**📊 数据集**

使用四个室内场景的仿真数据集：Corridor、Warehouse、Office、Apartment；

**📈 对比分析**

与SRE、REAL、ExplORB、FALCON等先进规划器进行10次运行的统计比较；RAVEN在λ=0时与FALCON相当的探索时长但定位误差显著降低，λ=1时定位误差最低，λ=0.5提供了平衡性能；

**⚠️ 局限性**

仅在仿真环境验证，缺乏真实机器人实验；计算量与规划时间未在大规模环境中验证，可能受限于实时性与场景复杂度。

---

## 524. A Counterexample to KL1 Completeness and a Complete Disjunctive Calculus

**arXiv ID:** 2610.37107 | [PDF](https://arxiv.org/pdf/2610.37107v1)

**作者:** Damiano Presa `[一作]` (University of Verona), Massimiliano Badino `[通讯]` (University of Verona)

**关键词:** `Logic in Computer Science` `Explainability and Interpretability` `Computational Efficiency` `Optimization` `Agentic AI` `Formal Proof` `Invariant Analysis` `Local Feature Analysis` `Review/Survey Paper`

**🎯 论文内容**

研究并修正了输入/输出逻辑 ESS 的推理系统，证明其原计算不完备并给出修正版。

**💡 创新点**

发现单一中介要求导致不完备，提出并证明了带分裂侧条件的 may-trans-disj 规则，并证明修正后系统在任意规则集下一致性、有限规则集下完备。

**🔧 技术方法**

利用形式化证明与不变量、局部特征分析，并在 Agda 证明助手中进行机器验证；结合输入/输出逻辑与 Kant 规则的语义定义。

**📊 数据集**

无外部数据集，主要使用理论构造的规则集合和示例 counterexample 进行验证。

**📈 对比分析**

与原 ESS 计算比较，证明原不完备、修正版完备；Agda 证明编译时间约 5 秒，验证过程可自动完成。

**⚠️ 局限性**

仅在有限规则集下证明完备，无法推广到无限规则集；L^e 的完备性仍未解决；原计算的完备性尚未被证明。

---

## 525. On the Sparsity of Direct Preference Optimization: Weight Disentanglement in the NTK Regime

**arXiv ID:** 2610.37085 | [PDF](https://arxiv.org/pdf/2610.37085v1)

**作者:** Kensuke Sasaki `[一作]` (University of Tokyo), Issei Sato `[通讯]` (University of Tokyo)

**关键词:** `Machine Learning` `Optimization` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Contrastive Learning` `Text`

**🎯 论文内容**

本研究对 Direct Preference Optimization（DPO）在大语言模型上的优化动态进行了理论分析，并在实验中验证了其产生稀疏参数更新的机制。

**💡 创新点**

创新点在于提出并证明了“双重稀疏性”概念——DPO 的更新方向由少数支持样本的稀疏梯度特征线性组合而成，同时每个梯度特征本身因优劣答案共享词语而变得稀疏；该双重稀疏性与权重解耦（weight disentanglement）关联，可解释 DPO 在模型融合中的低干扰特性。

**🔧 技术方法**

核心技术包括：神经切线核（NTK）线性化、对 DPO 逻辑回归的隐式最大间隔偏置分析、梯度特征向量与支持向量的关联，以及基于交叉任务内积的解耦度量。

**📊 数据集**

实验使用的模型有 GPT‑2 Small/Medium、Llama‑3‑8B‑Instruct、Mistral‑7B‑Instruct‑v0.2、Gemma‑2‑9B‑it；数据集包含 Anthropic HH‑RLHF（helpful/harmful）和 UltraFeedback Binarized（四维度）。

**📈 对比分析**

与传统监督微调（SFT）相比，DPO 在跨任务干扰指标上提升 11~35 倍；在模型融合（task‑vector merging）实验中，DPO 的性能下降幅度显著低于 SFT。

**⚠️ 局限性**

局限性包括：理论基于 NTK 线性化假设，未必在整个训练过程中成立；参数层级稀疏性假设依赖于优劣答案的词重叠，若两答案完全无关则稀疏性可能消失；此外，仅在实验中验证了小规模到 9B 参数的模型，尚需更大规模验证。

---

## 526. GKS Fully Homomorphic Encryption: Security Analysis

**arXiv ID:** 2610.37061 | [PDF](https://arxiv.org/pdf/2610.37061v1)

**作者:** John Graham `[一作]` (Zero Inc.), Vladimir Shpilrain `[通讯]` (CUNY)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Biomedical Data` `Tabular` `Electronic Health Records`

**🎯 论文内容**

对Gribov‑Kahrobaei‑Shpilrain (GKS) 私钥无噪声全同态加密方案进行正式安全分析，并给出实现与部署细节。

**💡 创新点**

提出基于幂等环Sₙ的无引导、无解耦的FHE设计，结合条件单密文IND‑EAV证明、ZCG碰撞防御、双加密以及非循环模数/非交换矩阵实例，构建出无限深度、无解耦的加解密流程。

**🔧 技术方法**

利用幂等环同态、嵌入α、秘密置换π与三角变换T的组合、统计与组合计数分析、零模式泄露与理想回收难度假设，并通过实验验证其安全性。

**📊 数据集**

在实验验证中使用医学数据集——糖尿病患者数据集、帕金森病数据集以及乳腺癌患者数据集。

**📈 对比分析**

与BGV/BFV/CKKS等基于RLWE的FHE方案在密文大小、加解密时延、乘除法时延、无限深度等指标进行比较；在无GPU、双加密SIMD实现下，GKS每槽加密约2.4 µs、加法2 ns、乘法31 ns、解密2.3 µs、扩展16×，且实现无限深度。

**⚠️ 局限性**

主要局限在理想回收的零知识假设未被完全证明、ZCG与随机三角基尚未完整实现、对理想回收后的安全性仅在非循环模数/非交换实例可保障，且单密文IND‑EAV仅在条件下成立。

---

## 527. Exploring the Limits of the Delta Learning Hypothesis: Single-Pair Generator Biases in Direct Preference Optimization

**arXiv ID:** 2610.37058 | [PDF](https://arxiv.org/pdf/2610.37058v1)

**作者:** Giuseppe Curci `[一作]` (Okkam), Andrea Zugarini `[通讯]` (Villanova.ai)

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Direct Preference Optimization (DPO) Loss` `Reward Hacking Analysis` `Style Feature Classifier` `Length Normalization` `Text` `Benchmark`

**🎯 论文内容**

本文对比单模型对与多模型生成的 DPO 偏好数据，探究其在不同模型族与规模下的效果，发现单模型对会引入生成器相关的系统偏差，导致奖励黑客和性能下降。

**💡 创新点**

创新点在于提供了理论证明和实证分析，揭示单模型对产生的偏好信号会被生成器相关特征主导，并提出仅通过提升生成器多样性即可部分缓解此问题，而无需外部 LLM 评判。

**🔧 技术方法**

使用了 Direct Preference Optimization (DPO) 损失、长度归一化、奖励黑客分析、风格特征分类器以及多模型偏好数据生成技术。

**📊 数据集**

数据集包括 AllenAI 的 Olmo-2-0425-1b-preference-mix（重新生成的多模型偏好数据）和评估基准 TruthfulQA、ARC-Challenge、IFEval、Math-500、GSM8K、Human-Eval、MBPP。

**📈 对比分析**

通过在相同提示分布下训练不同模型，比较单模型对与多模型+评判者生成的偏好数据所得到的 DPO 训练结果，实验显示单模型对在大多数基准上显著低于多模型对，增加生成器多样性虽能提升性能，但仍无法完全达到评判者监督的水平。

**⚠️ 局限性**

局限性在于实验仅覆盖至 8B 参数的模型，尚未验证在更大规模模型上的泛化；同时生成器选择与分层方法可能对结果产生影响，需要进一步研究。

---

## 528. State estimation for a hybrid rigid-continuum space manipulator system

**arXiv ID:** 2610.37043 | [PDF](https://arxiv.org/pdf/2610.37043v1)

**作者:** Lorenzo Ticozzi `[一作]` (Georgia Institute of Technology), Panagiotis Tsiotras `[通讯]` (Georgia Institute of Technology)

**关键词:** `Robotics` `Pose Estimation` `Optimization` `Robotic Intelligence` `Simultaneous Localization and Mapping` `Gaussian Splatting` `Factor Graph` `Levenberg‑Marquardt Nonlinear Least Squares Optimization` `Magnus Expansion` `Stereo Vision Triangulation` `Image` `Point Cloud` `Tabular`

**🎯 论文内容**

本文提出一种用于混合软硬空间机械臂的状态估计框架，通过将软连续段与硬关节结合在因子图中实现联合估计；

**💡 创新点**

创新点在于：①将连续软臂的几何可变应变（GVS）参数化直接纳入因子图作为可估计变量；②利用因子图统一融合基座姿态、关节编码器、相机立体视觉以及软臂的 Magnus 约束；③实现对整条链路（基座到末端）状态的精确估计并返回协方差；

**🔧 技术方法**

使用技术包括：因子图建模与GTSAM求解、Levenberg‑Marquardt非线性最小二乘优化、GVS与Magnus展开、基于立体相机的彩色标记点三角测量；

**📊 数据集**

数据集为基于SoRoSim V6.3的仿真数据，包含60 s轨迹、5个彩色标记点的合成图像（Blender渲染），并在每个时刻加入高斯噪声的基座姿态、编码器与视觉测量；

**📈 对比分析**

与现有方法的直接对比未给出，但实验结果显示末端平均定位误差为90 mm（占系统长度2.3%），相对于基座的相对误差仅毫米级；计算时间平均1.23 ms/步，适用于实时应用；

**⚠️ 局限性**

局限性包括：缺乏对软臂姿态的直接观测导致软臂角度误差较大；视角受限，易出现遮挡；目前仅在仿真环境验证，需在真实硬件上进一步测试；

---

## 529. Guided-RLVR: Guided Exploration in Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2610.37041 | [PDF](https://arxiv.org/pdf/2610.37041v1)

**作者:** Bo Yang `[一作]` (Beihang University), Hainan Zhang `[通讯]` (Beihang University)

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Reinforcement Learning` `Transformer` `Large Language Model` `Reinforcement Learning` `Prompt Engineering` `Joint Optimization` `Text`

**🎯 论文内容**

提出 Guided‑RLVR，一种在 RLVR 框架下加入轻量级控制器，在高熵句子边界主动注入语义指引 token，从而引导 LLM 在推理过程中探索更广阔的思路。

**💡 创新点**

①把状态条件控制器与 RLVR 结合，主动在推理中插入语义 steering token；②通过高熵句子边界触发机制控制注入时机；③加入 log‑perplexity 奖励作为辅助探索信号；③实现显著提升轨迹多样性和通过有限采样覆盖更多问题。

**🔧 技术方法**

RLVR (GRPO、DAPO)、轻量级 MLP 控制器、log‑perplexity 奖励、熵阈值触发、token logit boost、联合优化（policy+controller）。

**📊 数据集**

MATH、AIME2024/2025、AMC2023、Code‑R1 数据集；OpenThoughts‑114k‑math 用于控制器的冷启动。

**📈 对比分析**

在 Qwen2.5‑7B‑Instruct 与 Qwen3‑4B 两大 backbone 上，分别与 vanilla GRPO/DAPO、ENT、REP 进行对比；Guided‑RLVR 在 pass@k、文本多样性 (TD) 与方程多样性 (ED) 上均领先，平均提升 1.1–2.8 个百分点，且在大采样预算下保持优势。

**⚠️ 局限性**

计算成本略升高（约 12.6% GPU‑小时）；需要调节 log‑perplexity 奖励系数 α；仅在有限采样预算下提升，未在更大规模或不同任务上验证泛化；需手工设定 steering token 集合，可能限制适用范围。

---

## 530. GPA-VGGT: Geometry-Preserving Acceleration of VGGT

**arXiv ID:** 2610.37000 | [PDF](https://arxiv.org/pdf/2610.37000v1)

**作者:** Muyeol Sung `[一作]` (Seoul National University), Jongwoo Lim `[通讯]` (Seoul National University)

**关键词:** `Computer Vision and Pattern Recognition` `Computational Efficiency` `Depth Estimation` `Pose Estimation` `Transformer` `Gaussian Splatting` `Contrastive Learning` `Auto Encoder` `FlashAttention` `Image` `Video` `Point Cloud`

**🎯 论文内容**

在不进行额外训练的前提下，提出一种新的加速方法GPA‑VGGT，旨在提升VGGT在长视序列上的推理速度，同时保持其几何估计的高精度。

**💡 创新点**

核心创新包括：① 仅合并键和值而保留所有查询，结合+log s校正以恢复合并键的注意力质量；② 在早期层引入聚合跨帧广播（Pooled Cross‑Frame Broadcast），以低成本保留跨视图上下文；③ 通过低维高斯投影减少匹配计算；④ 对敏感块采用自适应合并比例。

**🔧 技术方法**

技术实现主要依赖Transformer注意力机制的改进（键值合并与查询保持）、聚合跨帧上下文、随机投影匹配、Triton自定义核（FlashAttention风格）以及PyTorch 2.11的自动混合精度。

**📊 数据集**

在CO3D、RealEstate10K（RE10K）、7‑Scenes和NRGBD四个多视几何基准上进行评估，使用CO3D与RE10K的相机姿态AUC@3与AUC@30指标，以及7‑Scenes与NRGBD的点图精度、完整度与法向一致性。

**📈 对比分析**

与FastVGGT、AVGGT、FasterVGGT等现有加速方案相比，GPA‑VGGT在10帧时的加速不超过1.2×，但在200帧时可达到约3.0×的速度提升（在500帧时为4.6×），且在AUC@3上与原始VGGT的差距保持在1%以内，显著优于其他方法在严格指标上的性能衰减。

**⚠️ 局限性**

局限性包括：在极长序列（>500帧）时仍受显存限制；相较于AVGGT(4)等最快方案，速度略低；对特殊场景（如极宽基线或低纹理区域）的合并策略仍可能导致细节丢失；并且所有改进均保持无训练，仅依赖设计，未进一步探索与微调结合的潜在收益。

---

## 531. LitUbik: open-source software for human-led systematic reviews with traceable AI assistance

**arXiv ID:** 2610.37181 | [PDF](https://arxiv.org/pdf/2610.37181v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Digital Libraries`

---

## 532. Zhengrong-Spatial-Aware 1.0: Advancing 3D Multimodal Intelligence

**arXiv ID:** 2610.37216 | [PDF](https://arxiv.org/pdf/2610.37216v1)

**作者:** Yu Zhang `[一作]` (Beijing Freedo Technology Co Ltd), Wenbo Chen `[通讯]` (Beijing Freedo Technology Co Ltd)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Image Translation` `Restoration` `Pose Estimation` `Depth Estimation` `Segmentation` `Autonomous Driving` `Optimization` `Computational Efficiency` `Representation Learning` `Data-Centric Learning` `Robotic Intelligence` `Meta Learning` `Reinforcement Learning from Human Feedback` `Neural Architecture Search` `Transformer` `Convolutional Neural Network` `Large Language Model` `Supervised Fine-Tuning` `Vision Language Model` `Contrastive Learning` `Prompt Engineering` `Gaussian Splatting` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Neural Radiance Field` `Auto Encoder` `Generative Adversarial Network` `Mixture of Experts` `Chain-of-Thought` `Image` `Point Cloud` `Mesh` `Text` `Multimodality`

**🎯 论文内容**

实现并验证了两套端到端的多模态适配流水线，用HiSpatial-3B结合RGB+3D点图进行细调，以及使用QLoRA对Qwen3-VL（8B/32B）进行监督式适配，以实现3D感知视觉问答；

**💡 创新点**

创新之处在于将度量尺度的3D点图直接注入视觉语言模型输入，并通过混合域指令微调展示了既能专注空间推理又能保持通用多模态能力的可复现工程方案；

**🔧 技术方法**

采用MoGe单目几何估计生成点图，HiSpatial架构（PaliGemma2-3B+点图分支），Qwen3-VL的MRoPE+DeepStack视觉‑语言融合，以及4‑bit QLoRA参数高效微调，配合WebDataset序列化、混合域训练和自动化评估；

**📊 数据集**

使用自定义空间数据集（包含RGB图、边界框、掩码、相机位姿、CAD/GLB模型）以及12类空间问题，并在同一数据集上加入风格化子集和通用VQA指令数据进行混合；

**📈 对比分析**

通过统一的JSONL评估接口对比性能，HiSpatial从1.73%提升至79.89%（原始子集）并在风格化子集从14.9%提升至55.1%；Qwen3-VL-8B从20.74%提升至45.27%，Qwen3-VL-32B从30.79%提升至47.60%；

**⚠️ 局限性**

局限性在于单一域过度专化导致对原始自由格式回答的精度下降，需要精细调节数据混合比例；输出格式分布影响适配效果；仅使用单目点图，未验证对其它3D表示的泛化性。

---

## 533. Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution

**arXiv ID:** 2610.37201 | [PDF](https://arxiv.org/pdf/2610.37201v1)

**作者:** Jiawei Zhou `[一作]` (Shanghai AI Laboratory), Chaochao Lu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `Artificial Intelligence` `Optimization` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `AI Code Assistant` `Transformer` `Large Language Model` `Reinforcement Learning` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Sequential` `Benchmark`

**🎯 论文内容**

本文提出BiPA（双边归因）工具，评估技能优化器在长时序任务中的选择性、成本与行为变化，随后分析三种公开技能优化器（GEPA、SkillOpt、SkillAdam）的表现并给出四条改进设计原则。

**💡 创新点**

创新点在于：①提出无需改动优化器即可通过重放接受规则和双边图实现对技能改动收益的无偏估计；②将输入token增量拆解为四类行为（更多工具调用、拆分轮次、每轮内容增量、技能文本长度），揭示成本来源；③发现训练集噪声导致的“选拔诅咒”并给出校正方案；④基于分析给出可直接落地的四条设计原则。

**🔧 技术方法**

主要技术包括：随机重放模拟（Monte Carlo）评估接受规则；记录回合与上下文的双边图（Turn‑Context Graph）；基于回合弹性（Turn Elasticity）分解输入token；以及多模型、多基准的实验评估。

**📊 数据集**

使用数据集：Terminal‑Bench 2.1（TB2）和 AutomationBench（AutoBench）两大长时序任务集，模型为 DeepSeek‑V4.1‑Flash 与 GPT‑6‑Luna，基准共18、75、120个测试任务，训练与dev集共160、30任务。

**📈 对比分析**

对比方法：对比学习得到的技能与种子技能以及无技能情况的测试成功率、输入token、轮次、工具调用比值；通过BiPA得出技能真实收益和成本。实验结果显示：学习到的技能在测试集上仅平均提升0.1点成功率，dev收益主要由选拔噪声驱动，且成本上升约16%（输入token）且多数迁移失败。

**⚠️ 局限性**

局限性：①仅为事后测量，无法预估单条规则的真实影响；②假设采用的harness（OpenCode）不压缩历史，若使用压缩历史的harness需重新测定回合弹性；③无法分解成功度至单个回合或规则，限制门控对单条规则的信用分配；④研究仅覆盖三种优化器与两模型，结果对其他方法的泛化性未知。

---

## 534. Acoustic Signature Optimisation for Bioinspired Flapping-Wing Micro Aerial Vehicles

**arXiv ID:** 2610.37190 | [PDF](https://arxiv.org/pdf/2610.37190v1)

**作者:** Yi Lun Duh `[一作]`, Sophie F. Armanini `[通讯]`

**关键词:** `Robotics` `Optimization` `Gaussian Process` `Bayesian Optimization` `Finite Element Dynamics` `Curle Sound Radiation Approximation` `Hilbert Envelope Extraction` `Butterworth Filtering` `Audio` `Tabular`

**🎯 论文内容**

本文提出并验证了一套针对微型拍翼飞行器（FWMAV）的声学优化设计框架，主要通过两种被动几何方法——翼内脉络（venation）布局优化和仿蜂鸟尾缘装饰（trailing‑edge fringe）——来降低翼的冲击噪声与气动吼声。

**💡 创新点**

创新点在于：①首次将翼脉络布局视为声学变量，并使用非线性von Kármán板模型与贝叶斯优化相结合的迭代流程自动搜索最安静的脉络配置；②提出了仿蜂鸟尾缘羽毛的可附加装饰纹理，既不削减升力，又能显著降低300–1000 Hz的宽带噪声；③实现了集成制造工艺（单片碳纤维框架+贴合膜）与实验验证的闭环体系。

**🔧 技术方法**

采用的技术包括：非线性有限元动力学（implicit generalized‑α积分、Newton求解），Gaussian Process 预测与最大期望改进搜索，Curle式声辐射近似（冲击力率→声压），高通 Butterworth 过滤与Hilbert包络提取；实验部分使用miniDSP UMIK‑1麦克风阵列、ATI Nano 17Ti 力/扭矩传感器及水刀切割制造。

**📊 数据集**

数据集主要由实验测试得到的多种翼设计构成：基准翼B0；脉络优化翼M1与M1‑adverse；五种添加尾缘纹理（F0–F4）与一款切割式锯齿翼。每种翼在10 Hz拍频下多次记录，得到冲击峰值包络和升力数据。

**📈 对比分析**

比较方法为：在同一拍频、相同实验环境下，将各翼在三个麦克风位置的冲击峰值包络与基准翼做对比，计算 dB 减小量；同时测量升力以评估性能损失。结果显示：脉络最优翼M1在80–1000 Hz频段冲击峰值降低约7.8 dB，升力提升至0.112 N；短牙齿尾缘纹理F2在300–1000 Hz降低约4.1 dB，升力仅略低于基准；切割锯齿翼在同频段降低约4.5 dB，却升力下降约25%。

**⚠️ 局限性**

局限性包括：①拍翼机件本身在1 kHz以上频段产生的噪声占主导，限制整体噪声降低潜力；②仅针对静态被动设计，未考虑主动控制或振动抑制；③模型使用的声辐射近似和冲击力率代理仅能预测相对排名，无法给出绝对声压水平；④实验仅在单一频率10 Hz下验证，未覆盖更广拍频或不同翼面积的通用性。

---

## 535. Irreducibility Testing of Sparse Polynomials over Integers is NP-hard and Probabilistic Polynomial Time

**arXiv ID:** 2610.37179 | [PDF](https://arxiv.org/pdf/2610.37179v1)

**作者:** Yichuan Cao `[一作]` (State Key Laboratory of Mathematical Sciences Academy of Mathematics and Systems Science Chinese Academy of Sciences), Jiawei Nie `[通讯]` (State Key Laboratory of Mathematical Sciences Academy of Mathematics and Systems Science Chinese Academy of Sciences)

**关键词:** `Computational Complexity` `Optimization` `Computational Efficiency` `Federated Learning` `Explainability and Interpretability` `Hyperparameter Search` `Data-Centric Learning` `Reinforcement Learning from Human Feedback` `Neural Architecture Search` `Drug Discovery` `AI Code Assistant` `Protein Structure Prediction` `Review/Survey Paper` `Physics Related`

**🎯 论文内容**

研究稀疏整数多项式的可约性判定问题，证明其在确定性多项式时间下的可约性是NP‑hard，且可在随机多项式时间（PP）和多项式空间内判定可约性。

**💡 创新点**

首次给出稀疏整数多项式可约性判定的下界（coNP‑hard）与上界（PP∩PSPACE）的匹配结果，并利用最新的准黎曼猜想与均匀有效素理想计数定理实现上界，填补了该领域长期未解的复杂度空缺。

**🔧 技术方法**

使用了从3‑CNF可满足性问题的多项式时间还原、Filaseta–Ford–Konyagin 的近似提升技巧、Cureton–Thorner 的准黎曼猜想、Frei–Sofos–Widmer 的素理想计数估计、模根计数与Eisenstein 判据、以及多项式空间的流式算法。

**📊 数据集**

本文未使用任何实验数据集，全部证明为理论复杂度分析。

**📈 对比分析**

通过比较可约性判定的上界与下界，表明该问题属于PP（可在随机多项式时间内）并且在PSPACE内可解，同时下界表明若存在多项式时间判定器则P=NP；这些结果均为理论复杂度评估，未涉及数值实验。

**⚠️ 局限性**

主要局限在于：1）对有限域稀疏多项式可约性的复杂度仍未确定；2）上界的证明依赖外部的准黎曼猜想与素理想计数定理，若这些定理被证明不成立则上界结果不再成立；3）算法为理论模型，实际实现的效率与常数因子尚未评估。

---

## 536. MLE-World: Predicting Machine Learning Experiment Outcomes with a Learned World Model

**arXiv ID:** 2610.37175 | [PDF](https://arxiv.org/pdf/2610.37175v1)

**作者:** Brian Ko `[一作]` (University of Washington), Sherry Yang `[通讯]` (New York University)

**关键词:** `Machine Learning` `AI Code Assistant` `Hyperparameter Search` `Computational Efficiency` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `World Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Sequential`

**🎯 论文内容**

提出MLE-World，一种在不执行程序的情况下预测机器学习任务终端输出（错误追踪、训练日志、验证分数）的世界模型，并利用其实现测试时刻扩展（TTS）搜索；

**💡 创新点**

创新点包括：①在4B Qwen3.5模型上进行监督微调，能够一次性预测终端输出与异常类型，且附带置信度；②设计五种基于该预测的TTS策略（Best‑of‑N、Sequential Revision、Pairwise Judging、Fusion、UCB）；③引入置信度门控，在低置信度时才进行真实执行，从而显著降低计算成本；

**🔧 技术方法**

使用的技术包括：监督微调语言模型、结构化多字段输出、置信度评分（p_max、uncertainty）、UCB探索式奖励、置信度门控、AIDE代理框架、四种程序操作（Draft、Debug、Improve）。

**📊 数据集**

数据集为MLE‑bench Lite的17个任务，训练集约497k条程序执行记录，保留1,700条程序做验证，实验中每任务使用3个种子、12h预算。

**📈 对比分析**

实验对比方法：提示式大型模型DeepSeek、Coder，以及无模型的任务/最近程序/父记录基线。MLE‑World在终端输出准确率77.9%（比DeepSeek 41.1%高），五种TTS策略平均得到52.1 percentile（比基线45.4%提升22.1%）。置信度门控进一步将真实执行时间压缩3040%且性能保持在或略高。

**⚠️ 局限性**

局限性：主要在分布内评估；对训练集未见任务的泛化尚未充分验证；超时预测准确率最低，误判仍需完整执行；模型依赖大量历史执行日志，若日志不足可能导致误导搜索。

---

## 537. Deep Holes and Aperture: Sign Laws for the Empirical Correlation of Lattice-Sampled Random Wave Fields

**arXiv ID:** 2610.37174 | [PDF](https://arxiv.org/pdf/2610.37174v1)

**作者:** Tianhao Guo `[一作]` (Shanxi University), Yuzhi Yang `[通讯]` (Japan Advanced Institute of Science and Technology)

**关键词:** `Information Theory` `Physics Related` `Gaussian Splatting` `Score-based Model` `Contrastive Learning` `Image` `Tabular`

**🎯 论文内容**

研究在有限孔径内，利用格点采样的高斯随机波场的经验滞后相关性，并给出在所有残差上保持正实部的概率以及对应的梯度下降保证。

**💡 创新点**

提出“深空洞极限定律”，将格点深空洞、深空洞配置的协方差、以及不同晶格的符号类等几何特征与经验相关性的极值行为直接关联，并给出精确的临界孔径常数与两侧孔径律；同时揭示了相位变化在任何正路径功率下导致的失败底部条件。

**🔧 技术方法**

运用了随机场极值理论、汉森–惠特曼尾估计、Gaussian 插值与马氏引理、泊松求和、谱函数分析、以及对称/不对称矩阵的高斯逼近等多种数学工具。

**📊 数据集**

通过对平面等速随机波场的高斯傅里叶‑贝塞尔展开进行精确仿真，采样 4096 条独立 realizations；模拟中使用的波场没有离散波束，保证了连续谱的真实性。

**📈 对比分析**

与深空洞 Gaussian 近似和单点近似进行比较。实验结果表明：深空洞近似能很好预测失败概率；单点近似误差显著；在不同晶格（正方形、六边形、简单立方、BCC）和不同边界条件下，模型能准确捕捉失效概率的量级，且与理论给出的上界/下界紧密吻合。

**⚠️ 局限性**

局限性包括：仅处理单输出链路；忽略互耦、阵列旋转对残差的影响；仅在 2‑3 维空间分析；未给出多输出或多波束的推广；以及对非常不规则或非凸孔径的分析仍待完善。

---

## 538. What Survives Distillation? A Mechanistic Interpretability Study of GPT-2 and DistilGPT-2

**arXiv ID:** 2610.37157 | [PDF](https://arxiv.org/pdf/2610.37157v1)

**作者:** Nabeelah Faumi `[一作]`, Deshan Sumanathilaka `[通讯]`

**关键词:** `Computation and Language` `Explainability and Interpretability` `Knowledge Distillation` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

演示了如何在 LuaLaTeX 或 XeLaTeX 中使用 ACL 样式文件。

**💡 创新点**

提供了多语言文本示例以展示样式兼容性。

**🔧 技术方法**

使用了 LaTeX、ACL 样式文件以及 LuaLaTeX/XeLaTeX。

**📊 数据集**

未使用任何数据集。

**📈 对比分析**

文中未进行实验对比，无法评估性能。

**⚠️ 局限性**

缺乏实验验证与真实数据，仅为示例演示。

---

## 539. Conditional Values: The Contingent Foundations of Attitudes toward Artificial Intelligence

**arXiv ID:** 2610.37197 | [PDF](https://arxiv.org/pdf/2610.37197v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## 540. Lossless Compression of Proximity Graph Indexes for Efficient Approximate Nearest Neighbors Search

**arXiv ID:** 2610.37176 | [PDF](https://arxiv.org/pdf/2610.37176v1)

**作者:** Francesco Benocci `[一作]` (ISTI-CNR), Rossano Venturini `[通讯]` (University of Pisa)

**关键词:** `Information Retrieval` `Compression` `Graph Neural Network` `Contrastive Learning` `Auto Encoder` `Graph` `Image` `Text` `Tabular`

**🎯 论文内容**

本文提出了一套针对 HNSW 邻接表的压缩框架，先通过离线图节点重排序（EGB）诱导结构局部性，再采用差分编码与 SIMD 加速的整数压缩（如 Zeta、SVB）对邻接列表进行压缩，从而大幅降低图结构存储占用且对查询延迟几乎无影响。

**💡 创新点**

核心创新点包括：①将离线图重排序与整数压缩结合，显著降低邻接列表熵；②在保证在线查询解压速度的前提下，使用 SIMD 友好的压缩算法（Zeta/ SVB）实现近 50% 的边存储压缩；③在 eBay 生产环境中部署验证，实测 44.3% 边存储缩减且查询延迟无可观提升。

**🔧 技术方法**

使用的技术包括：EGB（Enhanced Graph Bisection）或 KaMinPar 图重排序；差分（gap）编码；整数压缩算法 Zeta、SVB、Bitpacker、Elias‑Fano；偏移量的 Elias‑Fano 编码；Rust 语言实现与 FAISS/HNSW 库集成；SIMD 预取与分块解压。

**📊 数据集**

实验数据集涵盖四类：1) 1M 128‑维 SIFT 图像特征（欧氏距离）；2) 8.8M 文本段落，分别使用 dense/ sparse BERT 128‑/1024‑维嵌入（内积相似度）；3) 1M 1024‑维 dense 向量通过 PQ 压缩（M=128）后再检索；4) 30K‑维 sparse BERT 向量（120 非零），以及 eBay 的 7M 96‑维产品嵌入。还在 eBay 生产系统中进行部署评估。

**📈 对比分析**

与无压缩基线（Plain）以及其他压缩方案（Bitpacker、Elias‑Fano、SVB）进行对比，测量指标为 bits/edge、Accuracy@10 以及查询延迟。结果显示，Zeta 在所有数据集实现 1.7×–2.3× 的压缩比，SVB 在保持 2.1× 压缩比的同时，查询延迟接近或略优于 Plain；Bitpacker 作为轻量级方案提供 1.3×–1.6× 的压缩且延迟与 Plain 相当。eBay 生产部署验证了 44.3% 边存储压缩且无延迟退化。

**⚠️ 局限性**

局限性包括：仅针对 HNSW 图设计，其他图索引（如 Vamana、NSG）可能需重新评估；离线重排序增加构建时间，需承担额外内存映射成本；由于 HNSW 边界限制，参考式压缩收益低，导致无法利用列表级冗余；压缩方案依赖 SIMD 友好 CPU，跨平台性能可能受限；此外仅压缩图结构，未同时优化向量量化，整体压缩潜力仍有提升空间。

---

## 541. SimplexLoRA: Dynamic Rank Updating via Alternating Minimization with Simplex Projection

**arXiv ID:** 2610.37206 | [PDF](https://arxiv.org/pdf/2610.37206v1)

**作者:** Igor Shalygin `[一作]`, Aleksandr Beznosikov `[通讯]`

**关键词:** `Machine Learning` `Optimization` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Contrastive Learning` `Auto Encoder` `Alternating Optimization` `Projection Methods` `Text`

**🎯 论文内容**

本文提出了一种通过在n-尺度单纯形上投影权重实现自适应Rank分配的LoRA变体；

**💡 创新点**

创新点在于在保持总参数预算不变的前提下，利用权重简单线性约束动态学习每层的Rank，且不引入额外推理开销；

**🔧 技术方法**

技术上结合LoRA的低秩分解、可学习的层级缩放因子、欧氏投影至单纯形、交替优化以及动态Rank重构；

**📊 数据集**

实验使用了GLUE（MNLI、SST-2、CoLA、QQP、QNLI、RTE、MRPC）、XSum、CNN/DailyMail、GSM8K、BoolQ、HellaSwag等数据集，并在DeBERTaV3base、Llama 3.1 8B、Qwen3-8B、BART‑large等四大模型上验证；

**📈 对比分析**

与标准LoRA及多种动态Rank PEFT方法进行对比，结果表明在相同0.24%参数预算下，本文方法在大多数任务与模型上取得等价或更优的准确率，尤其在低Rank（1~2）场景中提升显著；

**⚠️ 局限性**

局限性包括需调节K、T、λ等额外超参数、仅在至8B模型上验证、部分任务收益不明显、以及权重重要性估计在梯度稀疏时可能产生偏差。

---

## 542. A Bi-level Co-design Framework for Lunar Monoped with 2D Jump and Landing Control

**arXiv ID:** 2610.37199 | [PDF](https://arxiv.org/pdf/2610.37199v1)

**作者:** Sanjay Ramkumar `[一作]` (Indian Institute of Space Science and Technology), Suraj Kumar `[通讯]` (U R Rao Satellite Center)

**关键词:** `Robotics` `Optimization` `Robotic Intelligence` `Reinforcement Learning` `Optimization` `Trajectory Optimization` `QP Task Space Control` `LQR Control`

**🎯 论文内容**

研究了月球低重力环境下单足机器人跳跃的形态与控制共设计，评估不同控制器对最优腿长的影响，并在仿真中实现二维原地与前进跳跃及着陆控制。

**💡 创新点**

提出双层共设计框架，将差分进化与两种内部控制器（QP任务空间控制和轨迹优化）相结合，揭示控制器选择显著影响形态优化结果，并在二维跳跃中加入自旋角动量调节和LQR着陆控制，提升稳定性。

**🔧 技术方法**

使用差分进化（DE）优化腿长，CasADi/Ipopt求解轨迹优化，MuJoCo仿真评估性能，QP任务空间控制基于离散二次规划，LQR控制用于着陆后姿态稳定。

**📊 数据集**

未使用公开数据集，全部实验基于自建的MuJoCo机器人模型和仿真数据。

**📈 对比分析**

通过比较两种控制器在不同形态下的成本运输(COT)来评估性能；轨迹优化在绝大多数形态下比QP-TSC低约17%（最佳COT分别为1.28 vs 1.55），并在二维跳跃中实现稳定的原地和前进跳跃。

**⚠️ 局限性**

仅在仿真中验证，缺乏硬件实验；形态参数仅限腿长，未考虑实际执行机构和扭矩曲线；着陆控制对姿态误差较为敏感，未提供稳健的多姿态适应方案。

---

## 543. LiveSearch: Questions as Executable Verifier Code for a Live Benchmark of Professional-Source Financial Search Agents

**arXiv ID:** 2610.37166 | [PDF](https://arxiv.org/pdf/2610.37166v1)

**作者:** Sinuo Wang `[一作]` (StepFun), Zuo Bai `[通讯]` (StepFun)

**关键词:** `Artificial Intelligence` `Finance Related` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Tabular` `Benchmark`

**🎯 论文内容**

该论文提出了 LiveSearch，构建了一个可实时更新的金融搜索基准，将问题表示为可执行程序，答案通过每日在专业金融数据服务上运行得到。

**💡 创新点**

创新点在于将问题与答案分离：问题文本保持不变，而答案通过可执行验证代码每日自动刷新，消除了答案过时问题并实现无人工干预的答案生成与更新。

**🔧 技术方法**

技术方法包括程序枚举与筛选、LLM 过滤与口头化、工具调用循环、可执行程序与提取器、LLM 评判器以及每日重执行的自动刷新流程。

**📊 数据集**

使用的数据集来自专业金融数据服务（Model Context Protocol），覆盖公司基本面、报价、宏观系列、基金、指数、事件等，最终生成 3,320 个可执行问题。

**📈 对比分析**

在通用网页搜索条件下评估六种工具调用模型，准确率在 70.9%–76.7% 之间；每日有约 70% 的问题答案会变化，模型无法保持答案复用，表现出对实时更新的敏感性。

**⚠️ 局限性**

局限性包括数据来源仅覆盖部分金融品种且无法公开调用，评估结果依赖 LLM 判别器，可能引入偏差或误判。

---

## 544. Coverage Under Contradiction: A Benchmark for Negation-Aware Retrieval

**arXiv ID:** 2610.37162 | [PDF](https://arxiv.org/pdf/2610.37162v1)

**作者:** Pasha Abrishamchian `[一作]` (Georgetown University), Ophir Frieder `[通讯]` (Georgetown University)

**关键词:** `Information Retrieval` `Retrieval` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Text` `Benchmark`

**🎯 论文内容**

构建了 CoverageBench‑Negate 基准，用以评估检索系统在面对与预训练知识冲突的矛盾信息时的覆盖能力。

**💡 创新点**

创新点在于通过可控否定（negation）增强原有覆盖基准，生成正反对比对，检验检索器对真相的忠实度。

**🔧 技术方法**

使用大型语言模型（如 Gemma、Llama、Qwen3）进行否定生成、文档重写和质量验证，并结合嵌入相似度检查。

**📊 数据集**

覆盖六个 CoverageBench 数据集（CAsT 2020、NeuCLIR 2024、RAG 2024、CRUX‑DUC04、CRUX‑MultiNews、RAGTIME 2025），共约 63,000 条编辑对。

**📈 对比分析**

对比 BM25、Qwen3、Jina 等检索器发现，BM25 在否定情况下仍显著优于密集检索器，表明其对语义一致性的更好把握。

**⚠️ 局限性**

局限性包括依赖 LLM 生成与评估可能引入偏差；否定对齐不保证与预训练知识完全冲突；对更高阶矛盾处理不足。

---

## 545. GCSR-OCC: Geometry Calibration and Semantic Routing for Open-Vocabulary Occupancy Prediction

**arXiv ID:** 2610.37149 | [PDF](https://arxiv.org/pdf/2610.37149v1)

**作者:** Bohua Wang `[一作]` (Alibaba Amap), Yang Cai `[通讯]` (Alibaba Amap)

**关键词:** `Computer Vision and Pattern Recognition` `Autonomous Driving` `Computational Efficiency` `Representation Learning` `Knowledge Distillation` `Transformer` `Gaussian Splatting` `Vision Language Model` `Contrastive Learning` `Knowledge Distillation` `Image` `Multimodality` `Point Cloud`

**🎯 论文内容**

提出 GCSR‑OCC 框架，通过几何校准和多模态语义路由实现单幅 RGB 图像的开放词汇 3D 体素占用预测。

**💡 创新点**

创新点：1）Evidence‑Conserving Gaussian‑to‑Occupancy（EC‑G2O）—利用质量归一化、p‑norm 聚合和可学习增益抑制 Gaussian 尺度膨胀与过度重叠；2）Multi‑mode Query‑First Gaussian Semantic Routing（MM‑QGSR）—对每个 Gaussian 进行多模式语义匹配并通过文本条件路由聚合；3）子类细粒度查询和平滑聚合进一步提升语义精度。

**🔧 技术方法**

使用技术包括：3D Gaussian Splatting、质量归一化、p‑norm 归约、可学习增益、语义匹配与文本对齐、低秩投影、子类查询、语义路由，以及通过 2D 视觉‑语言模型蒸馏训练语义路由器。

**📊 数据集**

使用 Occ‑ScanNet 数据集（单幅 RGB + 语义占用体素），训练集 45,755 帧，验证集 19,764 帧。

**📈 对比分析**

与多种闭词汇与开放词汇基线对比，在 Occ‑ScanNet 上实现几何 IoU 63.31%（超过所有闭词汇模型），语义 mIoU 27.44%（超过所有开放词汇基线），同时将占用聚合时延从 19.76 ms 降至 12.88 ms。

**⚠️ 局限性**

局限性：仍无法与全监督闭词汇模型的语义 mIoU 接近；对细粒度语义的校准受限于缺乏 3D 语义标注；在高度遮挡或多视角情境下的几何恢复仍有限。

---

## 546. MCP-Net: Real-Time Coronary Perception Network and Topological Analysis for Robotic Guidewire Navigation

**arXiv ID:** 2610.37127 | [PDF](https://arxiv.org/pdf/2610.37127v1)

**作者:** Sahar Nasirihaghighi `[一作]`, Stefano Stramigioli `[通讯]` (University of Twente)

**关键词:** `Computer Vision and Pattern Recognition` `Robotic Intelligence` `Convolutional Neural Network` `Transformer` `Auto Encoder` `Contrastive Learning` `Optical Flow` `Diffusion model` `Score-based Model` `Rectified Flow` `Neural Radiance Field` `Image` `Biomedical Data` `Computed Tomography`

**🎯 论文内容**

开发了一套基于CT血管模型生成的合成X射线数据，利用多任务冠状动脉感知网络（MCP‑Net）实现单帧血管分割、导丝尖端定位与狭窄定位，并将预测结果转化为可解析的中心线图谱和导航路径

**💡 创新点**

首次在同一网络中联合完成血管分割、设备定位与病变定位，并通过结构化图谱推导出从根部到狭窄点的完整导航路线

**🔧 技术方法**

采用DeepLabV3+骨干网络+多任务头、DSNT热图定位、骨架化+距离变换构造中心线图谱、最短路径搜索生成路线

**📊 数据集**

使用376份冠状动脉CT血管体素生成的1504张256×256像素的合成X射线图像数据集

**📈 对比分析**

与U‑Net、SegFormer、FPN等架构同一多任务训练，DeepLabV3+实现血管Dice 90.3%、中心线Dice 96.0%、导丝尖端误差1.09px、狭窄门部误差3.82px，端到端延迟约27 ms（37 fps）

**⚠️ 局限性**

仅在合成数据上训练，缺乏真实临床图像的验证，存在域迁移挑战，需要进一步的仿真‑真实适配与临床测试

---

## 547. EEG-Based Decoding of Musical Mode and Melodic Direction Using Empirical Mode Decomposition

**arXiv ID:** 2610.37148 | [PDF](https://arxiv.org/pdf/2610.37148v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Sound`

---

## 548. Mechanistic Failure Modes of Weight Pruning in Transformer Language Models

**arXiv ID:** 2610.37126 | [PDF](https://arxiv.org/pdf/2610.37126v1)

**作者:** Nabeelah Faumi `[一作]` (Informatics Institute of Technology), Deshan Sumanathilaka `[通讯]` (Informatics Institute of Technology)

**关键词:** `Machine Learning` `Explainability and Interpretability` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Large Language Model` `Prompt Engineering` `Contrastive Learning` `Text`

**🎯 论文内容**

在 GPT‑2 小模型上进行因子化剪枝实验，交叉重要性标准（幅值 vs 随机）、组件类型（注意力头 vs MLP 神经元）以及三种稀疏度（30%/50%/70%），通过激活补丁、路径补丁和直接 Logit 归因等机制层级因果分析工具，检验剪枝对已知 IOI 与 Greater‑Than 电路的内部结构与功能影响。

**💡 创新点**

提出了两种截然不同的机制失效模式：电路零化（结构被完全删去）和电路翻转（结构保留但功能反转），并证明传统性能指标（准确率、困惑度）无法捕捉后者；强调在压缩模型评估中需要补充电路级因果分析。

**🔧 技术方法**

使用 TransformerLens 框架进行激活和中间状态截取；采用幅值 L2 范数与随机方式评估重要性；实施完整的激活补丁、路径补丁和直接 Logit 归因（DLA）来量化各头/神经元对 IOI/GT 逻辑的贡献。

**📊 数据集**

数据集包括：① 300 条人工合成的 IOI 句子（ABB/ABC 设计）用于电路性能评估；② 对 GT 电路的合成提示集（类似格式）；③ TinyStories 语料库用于评估整体语言建模困惑度。

**📈 对比分析**

通过 12 种剪枝变体（2×2×3）分别记录平均 logit 差、准确率、困惑度、隐藏状态余弦相似度和 CKA ；结果显示：高稀疏度的注意头幅值剪枝导致电路零化（准确率降至 45–48%），而 30% MLP 神经元幅值剪枝产生电路翻转（准确率仅 17.7%，困惑度仅 74.6，误导性更大）。

**⚠️ 局限性**

局限性包括：仅在 GPT‑2 小模型上实验，未检验更大规模模型；仅分析 IOI 与 GT 两个电路，缺乏对自然语言任务的生态验证；剪枝采用零权重且无后续微调；重要性评估仅基于 L2 范数，未考虑梯度或归因导向方法；对安全机制的推论仅基于假设。

---

## 549. Matrix Samplers and Quantum-Proof Extractors: Their Correspondence and Almost-Optimal Constructions

**arXiv ID:** 2610.37122 | [PDF](https://arxiv.org/pdf/2610.37122v1)

**作者:** Kuan Cheng `[一作]` (Peking University), Ruiyang Wu `[通讯]` (Peking University)

**关键词:** `Computational Complexity` `Optimization` `Safty and Privacy` `Computational Efficiency` `Explainability and Interpretability` `Federated Learning` `Contrastive Learning` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Auto Encoder` `Generative Adversarial Network` `Review/Survey Paper` `Benchmark` `Physics Related`

**🎯 论文内容**

本文通过建立矩阵采样器与量子安全随机提取器之间的对应关系，提出了新的矩阵采样器和量子安全提取器构造，并实现了随机数消耗和样本复杂度的最优或近似最优。

**💡 创新点**

创新点在于首次将Zuckerman的采样-提取器对应扩展到矩阵情形，并利用量子侧信息的提取技术得到具有线性输出、最优种子长度的强量子安全提取器；同时给出了矩阵采样器的随机性与样本复杂度的新的下界。

**🔧 技术方法**

主要技术包括Hermitian稀疏化、量子留置哈希定理、弱设计、多项式评估、有限域构造、有限独立性、以及迭代采样与提取的混合方法。

**📊 数据集**

本文为理论研究，无实验数据集，所有结果均为定量的构造与上界下界证明。

**📈 对比分析**

与此前Xun‑Zuckerman（CCC 2025）相比，所构造的矩阵采样器在随机种子长度上相同但样本数从O(ε⁻²log(2D/δ))降到O(ε⁻²·log^α(2D/δ)/log(2D/δ))，并去除了对数平方因子；量子安全提取器在常数熵率下实现了O(log(n/η))种子、Ω(n)输出，显著优于已有的O(log(n/η)^2)种子或低输出长度方案。

**⚠️ 局限性**

局限性包括对参数的严格约束（如需要ε≥(log(2D/δ))^‑C，或对α取固定小常数），构造复杂度高且常数因子未给出；在极端参数（如极小ε或极大D）下效果可能退化；理论上仍存在可进一步压缩随机性或样本的空间。

---

## 550. Prospector: learning a compact graph traversal policy for scientific reasoning

**arXiv ID:** 2610.37121 | [PDF](https://arxiv.org/pdf/2610.37121v1)

**作者:** Haiqian Yang `[一作]` (Massachusetts Institute of Technology), Markus J. Buehler `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Computational Efficiency` `Reinforcement Learning from Human Feedback` `Knowledge Distillation` `Graph Neural Network` `Transformer` `Large Language Model` `Reinforcement Learning` `Retrieval-Augmented Generation` `Chain-of-Thought` `Prompt Engineering` `Graph` `Text`

**🎯 论文内容**

构建并训练了一个小型的LLM策略（Prospector），通过工具调用在知识图谱中逐步遍历，给出可追溯的答案链；在半导体制造文献的图谱上训练后，能零样本迁移到生物启发材料、PFAS和材料属性等其他领域。

**💡 创新点**

创新点在于：① 将检索与推理分离，模型在推理过程中实时决定读取哪条图边；② 用低秩适配器在冻结的大模型上进行后训练，保持模型体积小且易于本地部署；③ 通过内部潜在状态读写实现对遍历行为的可解释与可调控。

**🔧 技术方法**

技术包括：LLM（4B规模）+ 低秩适配器、ReAct式工具调用、强化学习（GRPO）分两阶段训练（全周期与决策阶段）、检索-验证机制、潜在空间可视化与调节。

**📊 数据集**

数据集：四个科学知识图谱——半导体制造、bio‑inspired materials、PFAS、materials‑properties，均由同一生成抽取管道构建；评测用195条半导体留存题目及各图谱对应的210–223条测试题。

**📈 对比分析**

与八款公开LLM（如Claude、GPT‑4、Llama‑2等）在同一工具设置下对比，Prospector在单步推理下精度0.839，接近最强的0.877；采样+验证后提升至0.913；在未见图谱上0.80+，超过大多数基线；同时单GPU每分钟回答10.7题，成本仅$0.57/千问，显著低于云端API。

**⚠️ 局限性**

局限性包括：① 依赖高质量的知识图谱，图谱缺失或噪声会影响推理；② 目前仅支持预定义的三种工具，扩展到更复杂的查询或开放式假设生成仍需研究；③ 虽可本地部署，但对显存和计算仍有一定需求；④ 对于极长链或多步推理，潜在状态管理可能需要改进。

---

## 551. Rethinking Positional Encoding and Interaction Granularity for CTR Prediction

**arXiv ID:** 2610.37140 | [PDF](https://arxiv.org/pdf/2610.37140v1)

**作者:** Xinxun Zhang `[一作]` (Wuhan University), Jiawei Jiang `[通讯]` (Wuhan University)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Sequential` `Tabular`

**🎯 论文内容**

提出TRIP框架，结合目标感知的秩-时间位置编码（FRE + TRTE）与时序分组因果交互（TGCI）实现CTR预测，兼顾序列顺序、时间距离与目标参考。

**💡 创新点**

创新点包括：① 低秩秩编码（FRE）在不增加参数的前提下捕获序列顺序；② 目标相对时间编码（TRTE）通过学习时间势能或旋转实现对预测时间的精准定位；③ 时序分组因果交互（TGCI）在目标相对时序组内实现双向交互、组间因果流动、掩码重建与组摘要压缩。

**🔧 技术方法**

使用Transformer结构，加入FRE与TRTE的加性或旋转位置编码，TGCI模块实现分组掩码、密度校正、组摘要与重建损失；训练时联合CTR损失与掩码重建损失，采用PyTorch实现。

**📊 数据集**

四个公开CTR基准：KuaiRec、KuaiRand、TAAC、TaobaoAd，覆盖短视频、广告与电商场景。

**📈 对比分析**

与11个基线（DeepFM、DCNv2、Wukong、RankMixer、DIN、DIEN、STCA、TokenFormer、OneTrans、HyFormer、MixFormer）及多种位置编码对比。TRIP在四个数据集均优于所有基线，平均AUC提升1.47%，单数据集最高提升4.45%；位置编码迁移实验显示FRE+TRTE在OneTrans、HyFormer、MixFormer中也均超越传统编码。

**⚠️ 局限性**

局限性：① 时序分组采用固定的对数区间，可能不适配用户兴趣的个性化转折；② 组摘要采用平均池化，可能丢失细粒度事件信息；③ 对低正样本率场景的过拟合风险在KuaiRand中表现为LogLoss偏高。

---

## 552. Physics Guided Diffusion Model Based Exposure Interpolation for High Dynamic Range Imaging

**arXiv ID:** 2610.37135 | [PDF](https://arxiv.org/pdf/2610.37135v1)

**作者:** Zhengguo Li `[一作]` (A*STAR), Chaobing Zheng `[通讯]` (Wuhan University of Science and Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Generation` `Super Resolution` `Transformer` `Diffusion model` `Score-based Model` `Contrastive Learning` `Image` `Physics Related`

**🎯 论文内容**

提出了一种基于物理引导扩散模型的曝光插值算法，能够在双大曝光比(SDR)图像之间合成中间曝光图像，从而消除亮度顺序反转（BOR）伪影并提升HDR成像质量。

**💡 创新点**

创新点包括：① 将物理驱动的初步插值与扩散模型相结合，利用新的预测目标设计来降低噪声与目标之间Pearson相关性的波动及误差放大系数；② 在扩散模型中引入新的预测目标，显著提升高频细节恢复；③ 用扩散模型替代传统CNN增强模块，实现更精准的细节重建。

**🔧 技术方法**

使用的技术包括：物理驱动曝光插值（IMF、CRF、权重融合方法）、Swin-UNet backbone的深度网络、前向/反向扩散过程（带新预测目标）、轨迹蒸馏、潜空间压缩、以及基于扩散生成的中间曝光图像与MEF融合。

**📊 数据集**

实验使用了两个HDR图像数据集：一是论文中引用的HDR数据集（420/40/60序列，512×512补丁），另一是公开的测试集用于评估。

**📈 对比分析**

通过与多种物理驱动算法、CNN增强算法、不同预测目标设计及不同α参数的对比，客观指标（PSNR、SSIM、FSIM、LPIPS、MUSIQ、HyperIQA）均显著优于现有方法；在MEF融合实验中，所提算法在MEF-SSIM、TMQI、PSNR、SSIM等指标上均超越六大主流曝光融合算法，主观评测显示更锐利细节、颜色自然。

**⚠️ 局限性**

主要局限是扩散模型的反向采样过程需要多步迭代，导致计算量大、速度慢，未来工作需进一步降低采样次数以提升效率。

---

## 553. SFE-O: An Ontology for Knowledge-Grounded AI in Supercritical CO$_2$ Biomass Extraction and Fractionation

**arXiv ID:** 2610.37123 | [PDF](https://arxiv.org/pdf/2610.37123v1)

**作者:** Thomas Pallet `[一作]` (Université de Lorraine, CNRS, CRAN), M{é}lanie Glibusic `[通讯]`

**关键词:** `Artificial Intelligence` `Knowledge Distillation` `Representation Learning` `Explainability and Interpretability` `Computational Efficiency` `Federated Learning` `Drug Discovery` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Neural Architecture Search` `Protein Structure Prediction` `Retrieval-Augmented Generation` `Prompt Engineering` `Contrastive Learning` `Auto Encoder` `Tabular` `Time Series` `Biomedical Data` `Agriculture Related` `Physics Related`

**🎯 论文内容**

提出并实现了专门针对超临界CO₂提取的模块化OWL 2本体SFE-O，完成了知识图谱的构建、语义化与一致性验证；

**💡 创新点**

在已有通用过程本体和生物质本体的基础上，首次显式化了Broken‑and‑Intact Cell（BIC）模型中的CER/FER机制及其与质量传递、过程参数之间的计算依赖，并通过重用ChEBI、QUDT、OWL‑Time、PROV‑O实现了高层次语义互操作；

**🔧 技术方法**

采用OWL 2 DL、模块化设计、重写关系、PROV‑O记录来源、HermiT推理器与SPARQL查询，结合ReifiedComposition、ContextualRole与ComputationalDependency模式；

**📊 数据集**

使用了单个真实的Dunaliella salina试剂批次（350 bar、65 °C、120 min、三分离器级联）进行实例化，以验证本体的完整性与可查询性；

**📈 对比分析**

通过对比推理一致性检查（HermiT成功）与SPARQL查询结果（CQ5得到期望输出）验证了本体功能；性能尚未做量化评估，仅证明了语义一致性与查询正确性；

**⚠️ 局限性**

局限性包括：未实现数值计算与约束执行（无SHACL验证），缺乏有效扩散率模型，未整合外部热物理、分子描述符或拟合服务，且仅在单一批次数据上验证，未与传统实验或AI方法进行对比。

---

## 554. How Much Can Best-of-N Help a World Action Model? Measuring Selectable Headroom and Realizing It in Closed Loop

**arXiv ID:** 2610.37196 | [PDF](https://arxiv.org/pdf/2610.37196v1)

**作者:** Haojia Gao `[一作]` (Tsinghua University), Xueqian Wang `[通讯]` (Tsinghua University)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `World Model` `Reinforcement Learning` `Vision-Language-Action Model` `Tabular` `Time Series` `Sequential` `Benchmark`

**🎯 论文内容**

提出了去噪的反事实协议，量化世界动作模型（WAM）在测试时选择中的可选头部空间，并基于此设计了失败门控选择器。

**💡 创新点**

创新点在于：①通过可重复的状态快照和多次续航实现对选择头部空间的客观测量；②使用留出或acles消除优化者的诅咒；③发现结构化动作扰动可产生可选头部空间；④证明想象未来已携带可用于选择的候选信息；⑤构建的选择器仅在预测失败时偏离策略，显著提升闭环成功率。

**🔧 技术方法**

技术包括位移可复现的分支、重复续航、留出oracle、共享/独立随机数、方差分解、线性探测器、失败门控与排名器，基于MuJoCo仿真器实现。

**📊 数据集**

使用四个WAM（Cosmos Policy、Flex‑π、Fast‑WAM、Streaming‑WAM）在RoboCasa和LIBERO‑Plus两个基准上进行评估。

**📈 对比分析**

与单次连续oracle、随机offset、价值头对比；在训练任务上闭环成功率从76.9%提升至81.6%（+4.7%），在未见任务上提升至69.8%（+7.2%）。

**⚠️ 局限性**

局限性包括：需要可完全恢复的仿真状态；头部空间估计需大量续航；在真实机器人上可能受限于状态恢复误差；目前仅在MuJoCo仿真环境验证，泛化到其他物理平台尚待验证。

---

## 555. VGIQA: Fine-Grained Image Quality Assessment with Visual Grounding Rewards

**arXiv ID:** 2610.37186 | [PDF](https://arxiv.org/pdf/2610.37186v1)

**作者:** Weichang Wang `[一作]` (Xidian University), Nannan Wang `[通讯]` (Xidian University)

**关键词:** `Computer Vision and Pattern Recognition` `Image Quality Assessment` `Anomaly Detection` `Vision Language Model` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Chain-of-Thought` `Diffusion model` `Image` `Multimodality`

**🎯 论文内容**

提出了 VGIQA，一种能够同时进行全局与细粒度局部图像质量评估的多模态大型语言模型。

**💡 创新点**

创新点包括：① 通过“Cold Start”阶段实现 CoT 结构化推理与 IQA 先验注入；② 采用 GRPO 多任务强化学习，专门设计局部视觉定位奖励（DIoU + F1）与全局质量奖励，形成严格的奖励隔离机制；③ 构建了首个基于视觉定位的局部 IQA 数据集 VGSR‑2K。

**🔧 技术方法**

技术手段：多模态大型语言模型（Qwen‑2.5‑VL‑7B‑Instruct）作为基础，使用监督微调与链式思考数据；随后进行多任务 GRPO 强化学习；奖励设计结合几何度量（DIoU）与语义对齐（F1）。

**📊 数据集**

数据集：VGSR‑2K（约1.8k 个基于 DiffIQA 与 SRBench 的参考‑生成图像对，包含人工标注的局部缺陷框）；用于全局评估的标准 IQA 基准（KonIQ、SPAQ、KADID、PIPAL、LIVE‑Wild、AGIQA‑3K、CSIQ）。

**📈 对比分析**

与现有方法比较：在 VGSR‑2K 上，VGIQA 的 F1 达到 0.3024，DIoU 0.3235，明显优于 Q‑Insight、Qwen‑2.5‑VL‑72B‑Instruct 和 Qwen‑3‑VL‑32B‑Instruct；在全局降质感知任务中，VGIQA 的各类分类与级别准确率均高于 Q‑Insight；在评分回归任务中，VGIQA 与 Q‑Insight 的 PLCC/SRCC 接近，略低但仍保持竞争力。

**⚠️ 局限性**

局限性：① 目前仅针对图像评估，未扩展至视频；② 依赖人工标注的局部框，数据规模有限；③ 在极端或未见过的失真类型下可能仍需进一步鲁棒性提升。

---

## 556. WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training

**arXiv ID:** 2610.37161 | [PDF](https://arxiv.org/pdf/2610.37161v1)

**作者:** Ziyang Zeng `[一作]` (IQuest Research), Bryan Dai `[通讯]` (IQuest Research)

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Robotic Intelligence` `Data-Centric Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

本文提出了一套名为WorkFactory的框架，用于构造可执行的专业任务环境，并基于此生成教师示范轨迹，进而训练三款工作型代理模型WorkPilot（4B、27B、35B-A3B）。

**💡 创新点**

核心创新在于信息支持图（information support graph）的逆向规划，明确输出所需信息的来源与位置，并通过执行‑基础修复（execution‑based repair）检测并修复构造过程中的支持缺陷；同时结合大规模（112,686个任务）专业任务库，显著提升工作型代理的表现。

**🔧 技术方法**

技术包括：LLM驱动的意图合成、信息支持图规划、环境构造与检查、执行‑基础修复、教师轨迹采集、监督微调（supervised fine‑tuning）以及对比实验。

**📊 数据集**

使用了112,686个由WorkFactory构造、覆盖612个职业的任务集，并在GDPval、JobBench、Workspace‑Bench、OmegaUse‑OfficeVal四大专业工作基准上进行评估。

**📈 对比分析**

与各自的基线模型（Qwen3.5‑4B、Qwen3.6‑27B、Qwen3.6‑35B‑A3B）对比，WorkPilot在四项基准上均有显著提升；例如WorkPilot‑35B‑A3B的GDPval‑AA Elo从879提升至1565，WorkPilot‑27B在GDPval‑AA达到局部估计1612 Elo，已位列前沿模型之列。

**⚠️ 局限性**

限制包括：教师模型GLM‑5.3仅支持文本输入，导致无法直接学习视觉证据；实验仅基于监督微调，未探索强化学习；任务示范的多样性和质量仍受构造方法和教师选择影响，未来需进一步扩展多模态与RL等方法。

---

## 557. MAPLE-Sec: Mobility-Aware Partitioned Language-Model Engine for Roaming IoT Security

**arXiv ID:** 2610.37147 | [PDF](https://arxiv.org/pdf/2610.37147v1)

**作者:** Yuhang Qiu `[一作]` (Jimei University), Por Lip Yee `[通讯]` (Universiti Malaya)

**关键词:** `Networking and Internet Architecture` `Anomaly Detection` `Computational Efficiency` `Federated Learning` `Safty and Privacy` `Transformer` `Supervised Fine-Tuning` `Auto Encoder` `Tabular` `Time Series`

**🎯 论文内容**

提出并实现了 MAPLE‑Sec，一个面向移动 IoT 端点的分层语言模型安全检测框架，解决跨域 handover 造成的检测状态丢失、上传量大、延迟高等问题，实现实时、低功耗的安全检测。

**💡 创新点**

创新点包括：①轨迹感知分区调度（TPS）根据预估 handover 时长、电池剩余与上行链路质量动态决定模型分区点；②令牌预算早停（TBEE）在置信度满足阈值或令牌耗尽时提前退出，保证 worst‑case 延迟；③激活压缩编码器（ACC）在 handover 时迁移压缩后的推理状态，保持检测连续性，首次在移动 IoT 上实现神经网络状态迁移。

**🔧 技术方法**

技术细节：安全微调的 Transformer 编码器、动态模型分区、置信度 + 令牌预算早停、基于 rate‑distortion 的学习式激活压缩、混合整数规划求解在线决策，全部在 Jetson Orin Nano 上低功耗实现。

**📊 数据集**

使用 CIC‑IoT 2023 和 Edge‑IIoTset 两个公开 IoT 入侵检测数据集，在 SUMO/LuST 生成的城市交通轨迹上重放，分别在 IID 与非 IID (Dirichlet α=0.3) 设定下进行实验。

**📈 对比分析**

与 Cloud‑only、EdgeLLM、EdgeShard、Active‑Infer 四个基线对比。IID 下 MAPLE‑Sec F1 为 0.962，较 EdgeShard 提升 2.9pp，较 Cloud‑only 提升 9.3%；非 IID 下仍保持 0.941。平均延迟 23 ms，P95 38 ms，零 deadline 违规；上传量 160 B，能耗 17.4 mJ；ACC 将迁移能耗比原始激活低 2.4×，整体性能显著优于基线。

**⚠️ 局限性**

局限性：①流量与移动状态的耦合仅通过静态数据与 SUMO 轨迹模拟，未收集真实移动端生成的流量；②边缘尾部与 ACC 解码器仅在桌面 GPU 上评估，缺少完整嵌入式边缘部署验证；③ACC 与原始激活的对比未与量化或更先进图像压缩基线做全面比较。

---

## 558. Beyond Parameter Protection: Fisher-Shaped EWC for Replay-Free Continual Learning

**arXiv ID:** 2610.37145 | [PDF](https://arxiv.org/pdf/2610.37145v1)

**作者:** Liying Wang `[一作]` (Universitat Autonoma De Barcelona), Joost van de Weijer `[通讯]` (Universitat Autonoma De Barcelona)

**关键词:** `Machine Learning` `Classification` `Representation Learning` `Computational Efficiency` `Knowledge Distillation` `Meta Learning` `Transformer` `Contrastive Learning` `Elastic Weight Consolidation (EWC)` `Low-Rank Adaptation (LoRA)` `Image` `Text`

**🎯 论文内容**

提出了 Fisher-Shaped EWC（FS‑EWC），在连续学习中通过在训练目标中加入对未来参数漂移邻域的 Fisher 敏感性约束，同时采用轨迹平均的模型合并方式构造未来任务的参考解；

**💡 创新点**

创新点在于：① 将 Fisher 信息从仅仅用于参数保护扩展到指导当前任务解的形成，使解在未来有限参数位移下更不易导致遗忘；② 在训练时对邻域内最坏情况的 Fisher 敏感性进行优化；③ 通过多步轨迹平均而非单一终点迭代来构造保留模型，从而在保留任务时保持更好的稳定性；

**🔧 技术方法**

技术主要包括：Elastic Weight Consolidation（EWC）与其在线形式；经验 Fisher 信息矩阵（empirical FIM）估计；Fisher 敏感性极值约束（对邻域半径 r 取最大值）；轨迹平均（trajectory‑based anchor construction）；LoRA 低秩适配器；以及多种 FIM 近似（批量、逐样本、模型采样）和矩阵表示（对角、Kronecker‑factored）；

**📊 数据集**

实验数据集涵盖：视觉领域的 CIFAR‑100、ImageNet‑R、ImageNet‑A、DomainNet，以及五个细粒度数据集（Aircraft、Cars、CUB‑200、Flowers、Pets）和它们的 5Datasets 组合；语言领域使用 Llama‑3.2‑1B/3B 在 AG News、Amazon Reviews、DBpedia、Yahoo Answers 四个文本分类任务；

**📈 对比分析**

对比方法包括：原始 EWC、LoRA、EWC‑LoRA、以及多种 LoRA 变体（InfLoRA、SD‑LoRA、BiLoRA 等）；结果显示 FS‑EWC 在所有视觉基准上均优于对应 EWC 或 EWC‑LoRA，典型提升如 ImageNet‑A FAA 从 56.42% 提升至 62.08%，ImageNet‑R 从 74.28% 提升至 79.81%；在语言基准上，FS‑EWC 也提升了约 4–5% 的最终平均准确率；此外，诊断实验表明 FS‑EWC 在保持相同参数漂移量的情况下，损失增加更小，Fisher 敏感性更低，显示出更好的稳定‑可塑性平衡；

**⚠️ 局限性**

局限性包括：① 需要额外的 FIM 计算和极值搜索，增加计算开销；② 对邻域半径 r 的选择仍需经验调优，虽然实验表明效果对 r 稳定，但在更大规模模型上可能需要更精细的设置；③ 仅在无重放的连续学习场景下验证，尚未评估在带重放或多模态任务中的表现；④ 理论上给出的损失上界可能相对保守，实际提升取决于 Fisher 估计的质量。

---

## 559. Multibranch Ensemble Head in Vision Foundation Models for Efficient Cucumber Phenological Stage Classification

**arXiv ID:** 2610.37118 | [PDF](https://arxiv.org/pdf/2610.37118v1)

**作者:** Kostiantyn Khabarlak `[一作]` (Dnipro University of Technology), Kostiantyn Khabarlak `[通讯]` (Dnipro University of Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Computational Efficiency` `Representation Learning` `Transformer` `Convolutional Neural Network` `Contrastive Learning` `Mixture of Experts` `Image` `Agriculture Related`

**🎯 论文内容**

提出了多分支正则化分类头，利用轻量线性分支和重型 MLP 分支的随机采样与概率空间聚合，改进了 Vision Foundation Models 在小规模农业数据上的适配；

**💡 创新点**

创新点在于通过训练时随机选择单个分支并在推理时概率平均，形成内部模型集成，既提供正则化又几乎不增加推理成本；

**🔧 技术方法**

采用自监督预训练的 VFM（如 DINOv2、DINOv3 ViT/ConvNeXt）作为骨干网络，并实现多分支头；

**📊 数据集**

使用 NALEF Food Safety 数据集，对黄瓜五个生长阶段进行图像分类；

**📈 对比分析**

对比多种强监督与自监督骨干、不同头部（线性、重型、正则化多分支），在 80/20 验证拆分上评估 Macro F1 与 Cohen's Kappa；DINOv2 ViT‑S + 多分支头在 12.2 GFlops、22 M 参数下达到 0.978 Macro F1，匹配 10× 参数量的 ViT‑L；

**⚠️ 局限性**

局限包括仅在叶面视角下训练，花果遮挡导致与开花阶段混淆；对恶劣天气、夜间或镜头污染等条件缺乏鲁棒性；并且验证集不平衡，结果对小类别敏感。

---

## 560. MixSIS: Mixed-supervised instance segmentation

**arXiv ID:** 2610.37138 | [PDF](https://arxiv.org/pdf/2610.37138v1)

**作者:** Christian Wilms `[一作]` (University of Hamburg), Simone Frintrop `[通讯]` (University of Hamburg)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Transformer` `Convolutional Neural Network` `Diffusion model` `Contrastive Learning` `Auto Encoder` `Non-Maximum Suppression` `Image`

**🎯 论文内容**

提出一种新的混合监督实例分割任务(MixSIS)，并设计基线模型AnnsMixer实现任意注释级别的伪掩码生成与训练；

**💡 创新点**

创新点在于：①允许同一张图像中混合mask、box、point和image级别注释；②使用SAM结合几何提示自动生成伪掩码；③在单一系统中统一处理多种弱监督；

**🔧 技术方法**

技术包括SAM模型进行伪掩码生成、CAM提取点提示、非极大值抑制、Mask R‑CNN全监督训练；

**📊 数据集**

主要使用COCO数据集进行MixSIS与WSIS评估，同时在VOC、UVO、Cityscapes、PASCAL VOC、BDD100k、MinneApple和MAD等数据集进行案例验证；

**📈 对比分析**

与多种WSIS方法对比，AnnsMixer在MixSIS任务上mAP分别为0.323、0.295、0.266，均高于Mask R‑CNN、BoxLevelSet、SAPNet等；在WSIS场景下亦能第二名或更好；案例研究显示可将标注时间降低约90%，同时提升或保持分割精度至+34%；

**⚠️ 局限性**

局限性包括：伪掩码质量对图像级别弱监督较低，导致性能下降；依赖SAM的预训练知识，可能对不同域适应性不足；目前仅在COCO等公开数据集验证，缺乏跨域或真实工业场景的深入评估。

---

## 561. TopoField: Technical Report for ECCV 2026 SceneFun3D Challenge

**arXiv ID:** 2610.37218 | [PDF](https://arxiv.org/pdf/2610.37218v1)

**作者:** Zhiwei Chen `[一作]` (Shandong University), Yupeng Hu `[通讯]` (Shandong University)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Graph Neural Network` `Contrastive Learning` `Diffusion model` `Optical Flow` `Stochastic Differential Equation` `Score-based Model` `Point Cloud` `Graph` `Benchmark`

**🎯 论文内容**

提出 TopoField 两阶段确定性解码器，将 VOLT 的点级语义概率通过 TTA 聚合并解码为排名实例掩码，用于室内点云功能元素分割。

**💡 创新点**

通过索引对齐的 TTA 语义共识和基于图几何的核心/支撑分割，弥合连续概率场与离散实例集之间的 "field-to-instance" 差距，实现无学习参数的高质量实例生成。

**🔧 技术方法**

采用 13 次测试时增强、索引对齐累积、缓存融合、稀疏 3D kNN 图、Dijkstra 多源赋值、soft‑NMS 及类感知评分等技术。

**📊 数据集**

在 SceneFun3D 公开数据集（160 训练 / 30 验证 / 30 测试）上进行训练与评测。

**📈 对比分析**

与多种基线（单阈值、图几何、类条件）对比，在验证集上提升 AP 至 39.67%，官方测试集 AP 为 28.29%，并夺得第二名。

**⚠️ 局限性**

仍依赖精细阈值与图构建参数，对极小或高度遮挡的功能部件召回率偏低，且解码过程不支持在线更新。

---

## 562. KEPT: Conformal Keepouts Kept in Planning for Enclosed-Space Navigation in Degraded Sensing

**arXiv ID:** 2610.37217 | [PDF](https://arxiv.org/pdf/2610.37217v1)

**作者:** Eric Minwoo Kim `[一作]` (Korea University), Jong-Kook Kim `[通讯]` (Korea University)

**关键词:** `Robotics` `Autonomous Driving` `Optimization` `Safty and Privacy` `Conformal Prediction` `Simultaneous Localization and Mapping` `Optical Flow` `Stochastic Differential Equation` `Point Cloud` `Time Series`

**🎯 论文内容**

在烟雾、雾等感知质量下降的封闭空间中，使用符合概率的保持区（keepout）校准 LiDAR 距离误差，并在规划与执行层统一采用单一制动距离约束，保证保持区不变，从而提升车辆在狭窄通道的安全性与到达率。

**💡 创新点**

创新点：①按无标签严重程度动态选择符合概率阈值；②在规划时与指令层都使用相同的制动距离约束；③保持保持区不变，避免传统方法因收敛或缓解置信度导致路径失效；④结合追踪误差作为唯一裕度。

**🔧 技术方法**

技术方法包括 conformal prediction（一次性置信区间）、LiDAR 距离误差估计、栅格占据图、控制 barrier function（安全过滤器）、kinodynamic 轨迹规划器（B‑spline 优化）以及 FlightBench 仿真平台。

**📊 数据集**

数据集：室内烟雾记录（40 kg 地面机器人 + 16‑ring LiDAR + 视觉标定），以及仿真通道与雾模型（基于相同 LiDAR 采样的模拟器），并使用预录制的保持区更新序列。

**📈 对比分析**

实验通过在同一路径、相同起点/终点、相同保持区更新序列下，与部署模式（A' 0.15）以及其他基线（法向、滤波器等）对比。结果显示 KEPT 在保持区内进场次数比部署模式降低约70%，到达率不下降（甚至略高），保持区覆盖率基本符合目标概率。

**⚠️ 局限性**

局限性：闭环评估使用预录制的静态保持区，未进行实时感知与在线更新；仅验证单一路径与单一雾级别；缺乏对动态障碍物或不同地形的泛化；不提供轨迹级别的概率安全保证；对测量噪声与追踪误差的假设未严格验证。

---

## 563. PipeLink: Contract-Guided Linking of GPU Kernel Components

**arXiv ID:** 2610.37200 | [PDF](https://arxiv.org/pdf/2610.37200v1)

**作者:** Jize Guo `[一作]` (Peking University), Xupeng Miao `[通讯]` (Peking University)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Optimization` `Computational Efficiency` `Linking` `Code Generation` `Resource Allocation` `Contract-Based Programming`

**🎯 论文内容**

构建了一个基于合同的GPU核组件链接器PipeLink，可将独立编译的核组件合并为大核

**💡 创新点**

通过合同约束实现组件间资源匹配与自动重定位，避免手工组合并保持最高性能

**🔧 技术方法**

使用LTO（链接时优化）生成的对象、合同检查、资源分配求解、自动生成传输代码以及供应商LTO后端

**📊 数据集**

未使用传统数据集，而是在51种配置的Hopper与Blackwell GPU上进行基准测试

**📈 对比分析**

将链接后的核与同源单单元编译结果对比，平均误差≤0.2%；在多种ML场景（注意力、MoE、GEMM+SwiGLU）中实现2-2.16倍加速或与原始单核相当

**⚠️ 局限性**

只能处理已声明合同的组件，依赖特定GPU架构和LTO支持，缺少动态资源调度与跨厂商兼容性

---

## 564. Action-Value Calibration via Continuation Sharing for Long-Horizon Agentic Reinforcement Learning

**arXiv ID:** 2610.37195 | [PDF](https://arxiv.org/pdf/2610.37195v1)

**作者:** Yingjian Chen `[一作]` (Sun Yat-sen University), Zhilin Zhao `[通讯]` (Sun Yat-sen University)

**关键词:** `Machine Learning` `Reinforcement Learning` `Agentic AI` `Reinforcement Learning`

**🎯 论文内容**

无法确定论文的具体研究内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法确定比较方法与性能

**⚠️ 局限性**

无法确定论文的局限性

---

## 565. INTR++: INterpretable TRansformer with better Accuracy, Interpretability, and Adaptability

**arXiv ID:** 2610.37178 | [PDF](https://arxiv.org/pdf/2610.37178v1)

**作者:** Dipanjyoti Paul `[一作]` (Iit Kharagpur), Wei-Lun Chao `[通讯]` (Boston University)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Explainability and Interpretability` `Domain Adaptation` `Transformer` `Prompt Engineering` `Contrastive Learning` `Image` `Biomedical Data` `Magnetic Resonance Imaging` `Ultrasound` `Computed Tomography`

**🎯 论文内容**

设计并实现了一种可解释的Transformer框架INTR-PlusPlus，核心在于加入轻量化Prompt Adapter Module（PAM），通过适配器使class‑specific prompts在保持ViT预训练特征不变的前提下获得域特定信息，从而提升分类准确率、解释能力和跨域适应性。

**💡 创新点**

创新点包括：
• 引入PAM，实现prompt与图像patch之间的自注意力与交叉注意力，让prompt能主动吸收域特定特征；
• 采用门控机制动态调节适配器贡献，避免过度扰动预训练特征；
• 在Prompt-CAM的基础上兼顾自然图像与专业图像（医学、显微镜、遥感等）三大域，展示了显著的跨域泛化能力。

**🔧 技术方法**

技术手段：
• 预训练ViT（DINO/DINOv2/BioCLIP2）作为冻结backbone；
• 学习class‑specific prompts（C×D维度）以及分类头；
• PAM：低维线性压缩、GELU、Prompt Self‑Attention（PSA）、Prompt Cross‑Attention（PCA）、上采样投影与门控融合；
• 训练采用交叉熵损失，后端利用attention热力图提供解释。

**📊 数据集**

数据集：
• 自然图像域：CUB‑200‑2011、Stanford Dogs、Oxford Flowers、MedLeaf、Stanford Cars、FGVC Aircraft；
• 专业图像域：MRI Brain Tumors、Gallbladder Cancer Ultrasound、Colon Cancer WSI、Aerial Image Dataset (AID)。

**📈 对比分析**

与方法比较：
• 对齐Prompt‑CAM、ProtoConcepts、PIPNet、TesNet等可解释模型；
• 与后验解释方法Grad‑CAM、Layer‑CAM、Eigen‑CAM、Attention Roll‑out对比；
• 与适配器方法LoRA、AdaptFormer对比；
性能表现：
• 在所有10个数据集上均超过Prompt‑CAM，平均准确率提升约4–6%；
• 解释性指标（插入/删除）在医学数据（GBCU）上显著优于其他方法；
• 在自然域与专业域均保持竞争力，尤其在医学、显微镜和遥感任务中表现最为突出。

**⚠️ 局限性**

局限性：
• 仍依赖冻结的ViT backbone，若预训练特征与目标域差距过大可能限制适配效果；
• 需要为每个数据集训练可学习的class‑specific prompts，规模较大时计算与存储成本上升；
• 解释热力图主要基于attention，可能存在过拟合或误导性解释的风险；
• 在极端小样本或高噪声场景下的鲁棒性尚未系统评估。

---

## 566. DomeGS: Perceptually Consistent Background Completion for Unbounded 3D Gaussian Splatting in Virtual Reality

**arXiv ID:** 2610.37172 | [PDF](https://arxiv.org/pdf/2610.37172v1)

**作者:** Anastasiya Pechko `[一作]` (Jagiellonian University), Przemysław Spurek `[通讯]` (Jagiellonian University)

**关键词:** `Graphics` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Gaussian Splatting` `Vision Language Model` `Supervised Fine-Tuning` `Image` `Point Cloud` `Mesh`

**🎯 论文内容**

提出一种统一的背景完成框架，将 360° 生成式补全的天空图像直接嵌入 3D Gaussian Splatting（3DGS）场景，构造出可产生视差的天空球（skydome），实现前景与背景在同一 3DGS 资产中一致响应。

**💡 创新点**

创新点包括：①在 3DGS 表示中首次构造可视差的天空球而非传统 skybox；②结合 360° 扩散模型的全景补全与基于 3DGS 的监督策略，保持光照、色彩一致性；③通过多视角监督生成新的高质量 Gaussian 背景，消除前景与远景的边界缝隙。

**🔧 技术方法**

技术手段：3D Gaussian Splatting；360° 生成式扩散模型（FLUX.2‑klein‑base‑9B + LoRA）；多视角视差监督与 Alpha 合成；Unity 引擎实时渲染；Qwen3‑VL‑32B 用于自动视觉质量评估。

**📊 数据集**

使用了四个自采集的未 bounded 户外 3DGS 重建场景（Garden、Bear、Buddha、Tower）作为实验数据；生成模型训练基于公开的 360° 场景数据集（如 FLUX 训练集）。

**📈 对比分析**

通过 2×2 factorial within‑subject 实验与传统 skybox 对比，使用微探针（IMM、SKY）和 IPQ 问卷进行主观评估，并用 Qwen3‑VL‑32B 进行客观评估。结果显示在背景一致性（p=0.042, d=0.85）和沉浸感（p=0.042, d=0.35）上均显著优于 skybox，且无显著舒适度差异，任务表现相同。

**⚠️ 局限性**

局限性：①需先完成 3DGS 重建并且输入远景信息缺失；②生成模型对色彩/照明一致性仍有局限；③验证仅覆盖四个场景，缺乏更广泛的多样性；④对动态光照或复杂材质的支持有限；⑤不支持实时动态交互或场景更新。

---

## 567. From 12 to 200 Demonstrations: How ACT, Diffusion Policy, SmolVLA and $π_{0.5}$ Scale on a Low-Cost Arm

**arXiv ID:** 2610.37170 | [PDF](https://arxiv.org/pdf/2610.37170v1)

**作者:** Simon Cowling `[一作]` (fysical.ai), Simon Cowling `[通讯]` (fysical.ai)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Diffusion model` `Vision-Language-Action Model` `Vision Language Model` `Diffusion model` `Supervised Fine-Tuning` `Prompt Engineering` `Mixture of Experts` `Reinforcement Learning` `Video` `Image` `Sequential`

**🎯 论文内容**

在SO‑101低成本机械臂上，比较了ACT、Diffusion Policy、SmolVLA和π_0.5四种仿学习策略在12到200条演示数据下的pick‑and‑place任务表现，分析了成功率、失败阶段、启动失败和超参数敏感性。

**💡 创新点**

揭示预训练模型在数据稀缺时具有显著优势，但在100+演示时从零训练的ACT同样或更优；同时证明启动失败源自演示开头的空闲帧，且π_0.5的训练权重比例决定成功率差距。

**🔧 技术方法**

使用LeRobot框架实现四种策略，分别采用Transformer、扩散、SmolVLA和PaliGemma等架构，训练时统一固定步数和学习率，并对π_0.5采用LoRA和完整专家训练两种细化方案。

**📊 数据集**

数据集为225条人机协同演示，随后按12、25、50、100、200的嵌套子集训练，每个模型在十个固定放置场景上执行共1340个episode。

**📈 对比分析**

通过统计置信区间、Fisher精确检验、逻辑回归趋势和阶段阶梯分析，发现ACT在100演示时成功率93%，π_0.5专家仅为75%；SmolVLA和Diffusion Policy在50演示达到峰值后下降；π_0.5 LoRA几乎无成功。

**⚠️ 局限性**

局限包括单一任务单一机器人单一操作者、训练步骤与数据量共线、种子与光照差异、评价时间限制、无多模型验证及未对多重比较进行校正。

---

## 568. Rènyi-Regularized Flow-Based Variational Inference for Bayesian Calibration

**arXiv ID:** 2610.37169 | [PDF](https://arxiv.org/pdf/2610.37169v1)

**作者:** Exaucè Luweh Adjim Ngarti `[一作]`, Arthur Vidard `[通讯]`

**关键词:** `Artificial Intelligence` `Optimization` `Bayesian Calibration` `Federated Learning` `Computational Efficiency` `Explainability and Interpretability` `Flow-based Model` `Flow-based Model` `Variational Inference` `Rényi Divergence` `Gradient Normalization` `HMC` `KDE` `Grid Normalization` `Tabular`

**🎯 论文内容**

本文研究了在多模态后验中使用正则化的正态化流变分推断，提出通过Rényi散度正则化来缓解模式崩溃。

**💡 创新点**

创新点在于将ELBO与Tikhonov正则化关联，解释模式崩溃原因，并引入基于Rényi散度的正则项及自适应权重，提升多模态覆盖率。

**🔧 技术方法**

使用正则化的变分推断、正态化流、Rényi散度、梯度归一化、HMC、KDE、网格归一化等技术。

**📊 数据集**

采用两个二维参数的合成双峰后验数据集，分别为连接双峰和分离双峰两种配置。

**📈 对比分析**

与无正则化流及高斯混合模型对比，通过前向KL、反向KL、总变差、能量距离和近似崩溃比例等指标评估，正则化将近似崩溃比例从50%降至25%，前向KL和能量距离改善，但反向KL和总变差略升高。

**⚠️ 局限性**

限制在于正则化改变了贝叶斯目标，仍未彻底消除模式崩溃，且效果依赖网络架构、初始化和优化过程，存在显著的可变性。

---

## 569. Physics-Guided Flow Matching: A Generative AI Framework for Global Total Electron Content Forecasting

**arXiv ID:** 2610.37159 | [PDF](https://arxiv.org/pdf/2610.37159v1)

**作者:** Si-Tong Yang `[一作]` (Clemson University), Benjamin C. Martinez `[通讯]` (Utah State University)

**关键词:** `Machine Learning` `Physics Related` `Convolutional Neural Network` `Recurrent Neural Network` `Flow-based Model` `Score-based Model` `Time Series`

**🎯 论文内容**

提出了一种基于物理引导的流匹配（PGFM）框架，用于24小时全球TEC预测。

**💡 创新点**

创新点在于将离散PDE驱动的物理流作为概率路径，并通过空间自适应融合物理与数据驱动的速度场。

**🔧 技术方法**

采用流匹配、卷积GRU、条件U‑Net等神经网络以及物理估计器，结合PDE物理知识。

**📊 数据集**

使用一整年TIEGCM生成的全球TEC数据（72×144 网格，15分钟分辨率）。

**📈 对比分析**

与统计模型STGP、深度学习ConvLSTM及无物理指导的流匹配基线对比，PGFM在所有时延下RMSE、MAE、nMAE均低于其他可部署模型，平均降幅约39%–60%。

**⚠️ 局限性**

局限在于对未来物理变量的依赖、极地区物理过程分辨率不足，以及目前仅在模拟数据上验证，尚需在实时观测数据上进一步测试。

---

## 570. FastBEVFusion: Simple Multi-Modal Fusion for 3D Object Detection at 50 FPS

**arXiv ID:** 2610.37153 | [PDF](https://arxiv.org/pdf/2610.37153v1)

**作者:** Tom Sanitz `[一作]` (University of Hamburg), Christian Wilms `[通讯]` (University of Hamburg)

**关键词:** `Computer Vision and Pattern Recognition` `Autonomous Driving` `Object Detection` `Computational Efficiency` `Transformer` `Convolutional Neural Network` `Contrastive Learning` `Cross-Attention` `Multi-Scale Training` `Data Augmentation` `Image` `Point Cloud` `Multimodality`

**🎯 论文内容**

提出 FastBEVFusion，一种结合 LiDAR 与多视角相机的 3D 目标检测系统，利用半对齐 BEV 空间和轻量级交叉注意力融合，实现在 50 FPS 以上的实时检测。

**💡 创新点**

创新点包括：①用均匀深度分布将图像特征投影到 BEV，避免昂贵且易出错的深度估计；②仅使用单个交叉注意力块实现软动态融合，显著降低计算量；③通过多阶段训练提升对标定误差的鲁棒性。

**🔧 技术方法**

技术手段：ResNet‑18/50 图像编码器、PointPillars LiDAR 编码器、单层多头注意力交叉注意力模块、MLP 上采样、TransFusion 查询‑基检测头、数据增强和多尺度训练等。

**📊 数据集**

使用 nuScenes 数据集进行验证与评估。

**📈 对比分析**

与现有实时系统（如 BEVDet、BEVDepth、Fast‑BEV、SECOND、CenterPoint、TransFusion‑L 等）在 mAP、NDS 以及延迟（RTX 4090 上）进行对比。FastBEVFusionR50 以 63.4 mAP、66.8 NDS、41 ms 延迟领先，FastBEVFusionR18 更快 17 ms，但略低 mAP，整体性能优于其他实时方法。

**⚠️ 局限性**

局限性：系统延迟主要受编码器（ResNet/PointPillars）影响，若追求更高精度需使用更强编码器但会增加延迟；对极端低光、遮挡或极端视角的鲁棒性尚未充分验证；依赖半对齐 BEV 可能在特定场景下精度受限。

---

## 571. Variational Optimal Control for Diffusion Guidance

**arXiv ID:** 2610.37146 | [PDF](https://arxiv.org/pdf/2610.37146v1)

**作者:** Ruoyu Wang `[一作]` (AGI Lab, Westlake University), Chi Zhang `[通讯]` (AGI Lab, Westlake University)

**关键词:** `Machine Learning` `Generation` `Optimization` `Knowledge Distillation` `Transformer` `Diffusion model` `Score-based Model` `Contrastive Learning` `Gaussian Splatting` `Denoising` `End-to-end Distillation` `LoRA` `Image`

**🎯 论文内容**

提出一种变分控制框架，将采样时引导与基于奖励的模型微调统一到共享的雅可比拉回机制中，并实现 Variational Optimal Guidance (VOG) 和 Distilled Variational Optimal Control (DVOC)。

**💡 创新点**

创新点在于：① 用变分最优控制原理推导出对终端奖励的拉回修正，② 引入雅可比拉回把输出梯度投射回噪声空间，③ 将终端改进通过端点蒸馏方式“摊薄”到模型参数，实现采样引导与模型适配的统一本质。

**🔧 技术方法**

采用 Gibbs 变分原理、Doob 变换、低阶高斯近似、denoising 雅可比、端点蒸馏、LoRA 细化以及 EDM2、DiT、Z-Image-Turbo 等扩散模型。

**📊 数据集**

主要使用 ImageNet（64×64、256×256、512×512）以及 Z-Image-Turbo 作为预训练模型，评估基准采用 1000‑prompt Pick‑a‑Pic split。

**📈 对比分析**

与 CFG、AutoGuidance、ReCFG、Diffusion‑OPSD、DRTune 等基线进行对比；VOG 在所有 12 种模型/分辨率组合中均提升 FID（如 DiT‑XL/2 25 步从 7.58↓4.44），DVOC 在 Z‑Image‑Turbo 上经过 20 次更新后在 HPSv3++ 与四个 OOD 评测指标上均优于其它训练方法，且取得最高训练平均分。

**⚠️ 局限性**

局限性：① 对雅可比拉回的理论分析基于线性子空间假设，非线性 denoiser 的近似效果仍有限；② 计算雅可比及端点蒸馏需要额外推导与内存开销；③ 超参数（如 K、momentum、拉回调度）对性能影响较大，需要经验调优；④ 目前实验聚焦于 ImageNet，尚未在更广泛的多模态或大规模数据上验证。

---

## 572. Generative Replay for Equilibrium Propagation: Toward Edge-Friendly Continual Learning in Hopfield Networks

**arXiv ID:** 2610.37139 | [PDF](https://arxiv.org/pdf/2610.37139v1)

**作者:** Victor Cook `[一作]` (LIRMM), Marina Reyboz `[通讯]` (CEA List)

**关键词:** `Machine Learning` `Continual Learning` `Optimization` `Computational Efficiency` `Hopfield Network` `Equilibrium Propagation` `Energy-based Generation Replay` `Input Reconstruction Loss` `Experience Replay` `Generative Replay` `Image`

**🎯 论文内容**

在持续学习场景下，提出了基于Equilibrium Propagation训练的Hopfield网络的能量生成重放（EB‑Gen）方法，利用网络自身的动力学生成伪样本来缓解灾难性遗忘。

**💡 创新点**

创新点在于不依赖外部缓冲区或额外生成模型，直接通过冻结网络的能量景观实现输入与输出的共同重放，并在EqProp中加入输入重建损失以提升生成质量。

**🔧 技术方法**

使用技术包括Equilibrium Propagation、能量基础生成重放、输入重建损失、冻结网络进行伪样本生成，以及传统的经验重放、生成重放等基线。

**📊 数据集**

在Split‑MNIST、Split‑CIFAR10（以及少量的ImageNet10）等分割式类别增量数据集上进行实验。

**📈 对比分析**

与FineTune、ER、GenReplay、SRC、Joint训练等方法比较，EB‑Gen在Split‑MNIST和Split‑CIFAR10上分别取得约75%和56%平均准确率，稳定性略低于ER但优于GenReplay，显示在保持硬件友好与内存零开销的前提下仍能显著降低遗忘。

**⚠️ 局限性**

局限在于生成样本多样性不足、对输入重建的依赖导致部分任务下性能低于经验重放，且在更复杂或无任务边界的持续学习场景下的有效性尚未验证。

---

## 573. MRI-to-CT Translation of Spinal Images via Anatomy-Guided Pseudo-Pairing and Adversarial Diffusion

**arXiv ID:** 2610.37134 | [PDF](https://arxiv.org/pdf/2610.37134v1)

**作者:** Eduardo Nieto `[一作]` (Fundación Vicomtech, Basque Research and Technology Alliance), Karen López-Linares `[通讯]` (Fundación Vicomtech, Basque Research and Technology Alliance)

**关键词:** `Computer Vision and Pattern Recognition` `Image Translation` `Convolutional Neural Network` `Generative Adversarial Network` `Diffusion model` `Contrastive Learning` `Adversarial Attack` `Biomedical Data` `Magnetic Resonance Imaging` `Computed Tomography`

**🎯 论文内容**

提出三阶段的MRI→CT脊柱翻译框架：①基于脊柱分割的解剖引导伪配对；②使用SynDiff扩散+GAN模型进行二维翻译；③利用3D ResUNet+PatchGAN后处理提升体积一致性。

**💡 创新点**

创新点在于：①利用形状描述与刚性配准实现跨患者的伪配对；②将无配对学习与扩散模型相结合的SynDiff；③引入3D后处理网络显著改善体积连续性。

**🔧 技术方法**

技术包括脊柱分割、形状描述+余弦距离、Dice相似度、SynDiff（非扩散+扩散模块）、3D ResUNet + 3D PatchGAN GAN，以及多损失（adversarial, L1, SSIM, perceptual, inter‑slice consistency）。

**📊 数据集**

使用10个公开数据集，共1136 CT和558 MR扫描，覆盖颈、胸、腰三大脊柱区，测试集约1911张MRI切片。

**📈 对比分析**

采用分布指标（MMD、FID）、结构指标（Dice、EMD、JSD）以及下游分割任务进行评估；相较于无配对SynDiff基线，伪配对将Dice提升0.073（0.666→0.739），3D后处理进一步提高到0.575；整体FID略升，表明体积一致性改善但轻微失真。

**⚠️ 局限性**

局限性包括：缺乏真实配对标签导致评估只能用分布/下游任务；骶骨、胸椎和颈椎因变异大仍易出现误差；3D后处理虽消除断层，但引入平滑导致整体图像真实感下降。

---

## 574. Science-Fiction Transformation as a Design Lens for Visualization Transformation

**arXiv ID:** 2610.37131 | [PDF](https://arxiv.org/pdf/2610.37131v1)

**作者:** Abdulhaq Adetunji Salako `[一作]` (University of Rostock), Christian Tominski `[通讯]` (University of Rostock)

**关键词:** `Human-Computer Interaction` `Explainability and Interpretability` `Data-Centric Learning` `Text` `Review/Survey Paper`

**🎯 论文内容**

提出了利用科幻变形概念的反思视角，对可动画化的可视化转换进行设计与分析，并给出了四个假想的转换示例。

**💡 创新点**

创新点在于把科幻中的变形逻辑（目的、对应关系、可见性）引入可视化转换设计，提出对应关系的基数作为转换复杂度的指标，并强调目的驱动的视图转换。

**🔧 技术方法**

主要使用理论分析与概念框架构建技术，结合动画设计原理（如可见性、过渡结构）和科幻变形案例进行阐释；未实现具体算法。

**📊 数据集**

未使用真实数据集，采用虚构的合作网络数据情境来说明转换示例。

**📈 对比分析**

未进行实验或性能评估，仅通过理论论证和案例说明来比较不同转换的可读性与复杂度，缺乏定量表现。

**⚠️ 局限性**

局限性包括：仅考虑了1:1、1:N、N:1对应关系，未探讨N:N；科幻案例有限，未覆盖更广泛的变形策略；示例设计缺乏细节实现与实验验证。

---

## 575. Finite Element Modeling of Fracture in Brittle Foods with Physically Consistent Parameter Calibration: A Practical Case Study on Cooled Dark Chocolate

**arXiv ID:** 2610.37124 | [PDF](https://arxiv.org/pdf/2610.37124v1)

**作者:** Patrick Kurzeja `[一作]` (TU Dortmund University), Jörn Mosler `[通讯]` (TU Dortmund University)

**关键词:** `Computational Engineering, Finance, and Science` `Physics Related` `Tabular` `Mesh` `Benchmark`

**🎯 论文内容**

提出了一套面向食品（冷却黑巧克力）脆性断裂的实用建模流程，比较了尖锐界面与相位场有限元框架，并通过分阶段识别弹性模量、临界能量释放率和相位场宽度，对实验数据进行校准和验证。

**💡 创新点**

创新点在于：①将相位场宽度与网格尺寸耦合考虑，揭示两者对峰值力、断裂路径和能量曲线的相互影响；②提出“六步”实用建模路线，明确目标先决条件与参数识别顺序；③在有限计算资源下实现相位场与尖锐界面模型的可比性，并给出针对不同断裂目标的参数调优策略。

**🔧 技术方法**

采用了基于Griffith能量准则的尖锐界面与相位场有限元方法（包含局部与全局能量最小化、弧长控制），结合线性弹性力学和能量梯度降解函数；实验方面使用了拉伸、压缩、三点弯曲和板弯曲测试。

**📊 数据集**

使用冷却黑巧克力制成的试件，进行拉伸、压缩、三点弯曲（已知裂纹路径）和板弯曲（自由裂纹路径）实验，获得弹性斜率、峰值力和裂纹长度等数据。

**📈 对比分析**

通过将尖锐界面模型的临界能量释放率g_c与实验峰值力对应，获得单一校准；相位场模型在给定网格下调节l_φ并重新缩放g_c，以匹配峰值力。结果显示：尖锐界面模型对已知裂纹路径的弹性斜率和峰值力精度最高；相位场模型在l_φ较小时能近似尖锐界面结果，但对裂纹路径的曲率和应力场敏感；在板弯曲实验中，较大的l_φ能更好匹配实验裂纹路径，但会降低峰值力和峰位移。整体性能满足实验误差范围（峰值力误差<2%）。

**⚠️ 局限性**

局限性包括：①相位场宽度l_φ受网格尺寸限制，导致参数并非纯粹材料常数；②校准结果高度依赖目标（峰值力、峰位移或裂纹路径），无法统一；③实验受限于低温、单一材料、静态加载，未考虑温度、塑性、异质性等食品特有因素；④仅针对脆性断裂，无法直接推广到塑性或多阶段断裂过程。

---

## 576. Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution

**arXiv ID:** 2610.37215 | [PDF](https://arxiv.org/pdf/2610.37215v1)

**作者:** Pham Tri Quang `[一作]` (Center for AI Research, VinUniversity), Vien Anh Ngo `[通讯]` (Center for AI Research, VinUniversity)

**关键词:** `Robotics` `Robotic Intelligence` `Autonomous Driving` `Domain Adaptation` `Graph Neural Network` `Vision-Language-Action Model` `Retrieval-Augmented Generation` `Diffusion model` `Vision Language Model` `Graph` `Image` `Video` `Multimodality`

**🎯 论文内容**

提出一种 Retrieve to Act (R2A) 框架，在 Vision‑Language‑Action（VLA）策略中通过检索运动原语提升对视觉 OOD 的鲁棒性。

**💡 创新点**

创新点在于将训练经验构造为纯运动图，并用查询依赖的图神经网络（GNN）动态定位与当前执行相匹配的运动原语，再通过门控融合机制实现即时纠正，完全绕过脆弱的视觉特征。

**🔧 技术方法**

使用离线运动原语图、查询依赖 GNN、扩散模型进行初始状态校正以及门控融合网络；所有模块均保持 VLA 权重冻结。

**📊 数据集**

在 LIBERO‑Plus、RoboTwin2.0‑Plus 等仿真数据集以及 Aloha 机器人真实实验数据上进行评估。

**📈 对比分析**

与 GR00T N1.7、π_0.5、StableVLA 等基线以及 VLS、SDN 等现有 OOD 方法对比，平均成功率提升 12%–40%，在七类视觉扰动下均优于基线。

**⚠️ 局限性**

局限在于缺乏显式场景几何、无法处理未见的错误恢复场景、对极端视觉偏差的泛化仍受限。

---

## 577. Uncovering Shortcut Amplification in Video RLVR through the Lens of Perceptual Uncertainty Collapse

**arXiv ID:** 2610.37211 | [PDF](https://arxiv.org/pdf/2610.37211v1)

**作者:** Jiafeng Liang `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning` `Knowledge Distillation` `Representation Learning` `Anomaly Detection` `Transformer` `Reinforcement Learning` `Contrastive Learning` `Knowledge Distillation` `Vision Language Model` `Video` `Multimodality` `Benchmark`

**🎯 论文内容**

在视频推理任务中，针对 RLVR 训练产生的视觉无关捷径现象，提出了 Bifocal Self‑Distillation Policy Optimization (BSPO) 方法。

**💡 创新点**

创新点在于将原则引导的信用分配与对抗视觉扰动的对比式蒸馏相结合，形成双重分布信号，既提升推理敏感性又抑制视觉无关捷径。

**🔧 技术方法**

技术包括 RLVR 的 GRPO 框架、基于奖励的策略梯度、原则抽取（PCA）、视觉扰动（SCP）与自蒸馏的双向重加权优化。

**📊 数据集**

使用 Video‑MME、LVBench、TempCompass、MVBench、MMVU、Video‑MMMU 等六个公开视频理解与推理基准。

**📈 对比分析**

与 GRPO、DAPO、SDPO、VISD 等主流 RLVR 与自蒸馏基线比较，BSPO 在所有基准上均提升 1–5%（根据实验表格），并显著改善对视觉伪证的鲁棒性。

**⚠️ 局限性**

局限在于仅在单一后训练阶段验证，对长期训练与不同模型架构的泛化仍待进一步评估；同时视觉扰动策略需要手工设计，可能不适用于所有场景。

---

## 578. 3D-SceneQ: Empowering 3D LLM With Query-Guided Adaptive Pruning and Multi-Modal Feature Enhancement

**arXiv ID:** 2610.37202 | [PDF](https://arxiv.org/pdf/2610.37202v1)

**作者:** Yongqi Shan `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Computational Efficiency` `Representation Learning` `Explainability and Interpretability` `Transformer` `Large Language Model` `Vision Language Model` `Contrastive Learning` `Supervised Fine-Tuning` `Cross‑Attention` `Point Cloud` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出了3D-SceneQ，一种结合自适应对象剪枝（QGAP）和多模态特征增强（MOFE）的3D大型语言模型，用于更高效、更准确的3D场景理解、描述与问答；

**💡 创新点**

1）QGAP利用可学习的潜在查询实现任务相关对象自适应剪枝；2）MOFE将多视角2D语义特征投影到3D空间并与3D点云特征跨模态融合，提升语义表达；

**🔧 技术方法**

基于PointNet++的3D编码器、CLIP/OpenSeg的2D特征提取、BERT作为查询与类别嵌入、Vicuna-7B LLM、LoRA微调、Cross‑Attention融合；

**📊 数据集**

使用ScanNet、Scan2Cap、ScanQA、SQA3D、3RScan、Matterport3D等公开的3D视觉‑语言数据集进行对齐与指令微调；

**📈 对比分析**

在3D场景描述、问答和基于场景的推理三大基准上，与现有SOTA（Leo、Point‑LLM、3D‑MiT等）相比，Scan2Cap提升CIDEr至78.1 (+5.7)，ScanQA CIDEr 104.7 (+20.0 EM@1)，SQA3D EM@1 67.0%（比前一最佳高出约17点），显示显著性能提升；

**⚠️ 局限性**

主要局限：针对室内场景，户外迁移性差；缺乏时间建模，难处理动态场景；推理深度受限，需进一步强化推理与时序建模。

---

## 579. Tight Bounds for Randomized Generalized $k$-Server on Uniform Metrics

**arXiv ID:** 2610.37192 | [PDF](https://arxiv.org/pdf/2610.37192v1)

**作者:** Junho Hwang `[一作]`, Junho Hwang `[通讯]`

**关键词:** `Data Structures and Algorithms` `Optimization` `Reinforcement Learning from Human Feedback` `Computational Efficiency` `Federated Learning` `Contrastive Learning` `Reinforcement Learning` `Mixture of Experts` `Prompt Engineering` `Review/Survey Paper`

**🎯 论文内容**

本文研究在统一度量空间上的广义 k-服务器问题，给出了单点距离为 1 时的随机竞争比为 Θ(k log k)，以及带权重的统一度量空间下的竞争比为 exp(Θ(k)) 的上下界，并证明这些上界与下界在常数级上相匹配。

**💡 创新点**

创新点包括：① 用加权多数与耦合技术构造了一个在单位权重下  (k+1)ln(2k) 竞争的算法；② 通过随机排列约束和组件结构提出了新的下界构造，使得对抗无知对手的最优竞争比至少 1/(20k ln k)；③ 对带权重情况采用层级阶段与专家聚合相结合的策略，得到 20^k 竞争比，几乎逼近已知的 2^k 下界；④ 所有结果对任意点数的空间均成立，且不需要对点数做额外假设。

**🔧 技术方法**

主要技术手段包括：加权多数（weighted majority）和其在可切换比较器上的扩展；耦合的“降重”方法（shrink‑dartboard）；随机排列约束的两阶矩分析与组件分解；阶段化策略（如 Bijoy–Mondal–Chiplunkar 的层级阶段）结合专家算法（Blum‑Burch）实现权重聚合；以及对请求集合的集合论与概率分析。

**📊 数据集**

该工作完全为理论分析，不使用实验数据集；所有结论均通过严谨的概率与组合证明给出。

**📈 对比分析**

与之前的 O(k² log k)（Bienkowski 等）和 Ω(k)（Koutsoupias 等）的上、下界相比，本文将上界压缩到 Θ(k log k)，下界提升到 1/(20k ln k)，几乎匹配。对于加权情况，先前的上界为 exp(O(k²))，本文给出 20^k 的上界，接近 2^k 的已知下界。

**⚠️ 局限性**

限制与展望：① 现有算法维护的概率分布支持规模指数级，实际实现及多项式时间实现仍是未解决的问题；② 竞争比常数（1/20 与 20）还有进一步改进空间；③ 目前仅对统一度量空间给出了完整分析，针对非统一或更一般度量空间的扩展仍待研究。

---

## 580. Remember Where You Walk: Learning Spatial Memory for End-to-End Legged Robot Exploration

**arXiv ID:** 2610.37203 | [PDF](https://arxiv.org/pdf/2610.37203v1)

**作者:** Leon Magnus `[一作]` (Technical University of Darmstadt), Oleg Arenz `[通讯]` (Technical University of Darmstadt)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Convolutional Neural Network` `Recurrent Neural Network` `Transformer` `Reinforcement Learning` `Simultaneous Localization and Mapping` `Hadamard Memory` `Image` `Sequential`

**🎯 论文内容**

该论文提出了一种基于Hadamard记忆的空间索引隐式记忆网络(lsmm)，用于机器人在未知迷宫中的自主导航。

**💡 创新点**

创新点在于将记忆更新限定在机器人当前位置附近的局部窗口，并通过Hadamard记忆更新保持长期信息，同时与环境坐标对齐，既避免了循环网络的混淆，又实现了端到端的地图记忆。

**🔧 技术方法**

技术上结合了PPO强化学习、辅助迷宫重建头、CNN编码器、MLP以及Hadamard记忆更新机制等。

**📊 数据集**

使用了30万份程序化生成的迷宫进行仿真训练，以及在Unitree Go2四足机器人上进行真实户外迷宫部署测试。

**📈 对比分析**

与LSTM、SRU、Transformer和固定窗口MLP等基线比较，lsmm在覆盖率相当的情况下，访问分布均匀度更好，迷宫重建误差降低三倍，整体性能更优。

**⚠️ 局限性**

局限性包括固定记忆尺寸只能覆盖已知大小的环境、依赖外部定位估计、仅在平坦小型迷宫中验证，且未实现内部自定位或对更大崎岖环境的可扩展性。

---

## 581. Taming Noise Initialization for Zero-shot Video Restoration with Image Diffusion Priors

**arXiv ID:** 2610.37290 | [PDF](https://arxiv.org/pdf/2610.37290v1)

**作者:** Chong Wang `[一作]` (Nanyang Technological University), Bihan Wen `[通讯]` (Nanyang Technological University)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Transformer` `Diffusion model` `Score-based Model` `Optical Flow` `Video` `Image`

**🎯 论文内容**

利用预训练的图像扩散模型，实现了零样本视频恢复框架；

**💡 创新点**

通过DDIM反演得到与原始视频相关的噪声进行初始化，再采用自适应噪声混合抑制降质模式，并在采样过程中加入测量引导的轨迹注意力以提升时序一致性；

**🔧 技术方法**

DDIM反演、适应性噪声混合、测量引导轨迹注意力、数据一致性约束等扩散模型相关技术；

**📊 数据集**

在DAVIS、REDS4和Set8三大视频数据集上进行实验；

**📈 对比分析**

与Kwon等人提出的相同噪声初始化方法、DiffPIR等现有零样本恢复方法对比，实验显示在PSNR、SSIM、LPIPS和FVD上均优于或相近，且在多任务下保持更高的稳定性；

**⚠️ 局限性**

相较于先前方法，引入DDIM反演、噪声混合和轨迹注意力显著增加了运行时间（约2倍）和显存占用，需进一步加速。

---

## 582. Bayesian Inference of Expensive Simulators: Survey and Comparison of Design Choices

**arXiv ID:** 2610.37284 | [PDF](https://arxiv.org/pdf/2610.37284v1)

**作者:** Šimon Soldát `[一作]` (Czech Technical University in Prague), Václav Šmídl `[通讯]` (Czech Technical University in Prague)

**关键词:** `Machine Learning` `Optimization` `Hyperparameter Search` `Gaussian Process` `Bayesian Optimization` `Tabular` `Time Series` `Sequential` `Benchmark` `Review/Survey Paper` `Physics Related`

**🎯 论文内容**

本文对使用昂贵模拟器进行贝叶斯推断的方法进行系统综述，并在统一框架下对不同设计选择进行实验比较。

**💡 创新点**

创新点包括：①提出统一的BOSIP框架，明确三大设计维度（代理变量、代理模型、采集函数）；②证明代理变量选择对推断效率决定性影响，提出自定义代理变量策略；③首次将非平稳高斯过程和新型采集函数（EIV、IMMD）引入该框架；④发布开源Julia实现。

**🔧 技术方法**

技术方法主要为贝叶斯优化、Gaussian过程代理、代理变量映射、期望积分方差（EIV）、最大均值差（IMMD）以及传统的最大方差（MaxVar）采集函数。

**📊 数据集**

使用了七个基准：四个解析式（AB、Simple、Banana、Bimodal）和三个PDE驱动的真实问题（SIR、Duffing、Diffusion）。

**📈 对比分析**

通过在相同模拟预算（10⁰–10²次）下测量总变差（TV）来比较，结果表明：①代理变量选择对效率影响最大，模拟器输出或自定义代理显著优于log‑likelihood；②非平稳GP虽提升性能但计算成本高；③MaxVar在单峰问题上表现最好，EIV在多峰问题上更有优势；④新采集函数在多数场景未明显优于MaxVar，凸显“无免费午餐”。

**⚠️ 局限性**

局限性包括：对代理变量设计依赖专家知识；非平稳GP的超参数估计和批量计算开销大；实验受限于有限模拟预算，无法验证极高维或极低成本模拟器的效果；部分高级采集函数在实际使用中难以收敛。

---

## 583. Modeling Preference Dependencies with Directed Acyclic Graphs for Narrative-Driven Recommendation

**arXiv ID:** 2610.37280 | [PDF](https://arxiv.org/pdf/2610.37280v1)

**作者:** Mingzhe Lu `[一作]` (Chinese Academy of Sciences), Hang Su `[通讯]` (Independent Researcher)

**关键词:** `Information Retrieval` `Recommendation System` `Graph Neural Network` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Graph`

**🎯 论文内容**

构建了一个以用户叙述为输入的首选项依赖图（PDG），并基于该图进行有向路径检索和拓扑路径排序，生成符合复杂叙事需求的推荐列表。

**💡 创新点**

创新点在于将叙事中的首选项资格关系显式建模为有向无环图，并将该结构贯穿候选检索与分数累积，解决了传统方法忽略层级依赖的问题。

**🔧 技术方法**

主要技术包括使用大型语言模型（如 Qwen-3.8‑27B）进行首选项抽取与图构造，利用 Serper API 的有向检索，结合动态路径累计与图权重聚合的评分算法。

**📊 数据集**

采用了三大公开基准：Reddit MovieSuggestions（电影）、Reddit 书籍（30 条英文查询）和 Xiaohongshu/知乎（30 条中文动漫查询）。

**📈 对比分析**

在 Precision@10 / Recall@10 / NDCG@10 上与多种 LLM 基线及 OCG‑Agent 进行对比，PDG‑F (+Year) 在所有数据集上均位居榜首，Movie P@10 提升 13.7%，Book NDCG@10 提升 92.4%，Anime R@10 提升 60.9%。

**⚠️ 局限性**

局限性包括：仍存在排名误差，主要受检索结果和文本匹配质量限制；对极端复杂或模糊叙事的解析仍依赖 LLM，可能导致图构造不准；此外缺乏对多语言长篇叙事的全面评估。

---

## 584. An Interactive Mobile Augmented Reality System for Efficient Virtual Model-to-Patient Registration in Thoracic Surgery

**arXiv ID:** 2610.37271 | [PDF](https://arxiv.org/pdf/2610.37271v1)

**作者:** Emmanuel Ajaero `[一作]` (Concordia University), Yiming Xiao `[通讯]` (Concordia University)

**关键词:** `Human-Computer Interaction` `Pose Estimation` `Depth Estimation` `Optimization` `Computational Efficiency` `Convolutional Neural Network` `Recurrent Neural Network` `Optical Flow` `Simultaneous Localization and Mapping` `Auto Encoder` `Contrastive Learning` `Point Cloud Processing` `ICP (Iterative Closest Point)` `Point Cloud` `Image` `Mesh` `Biomedical Data` `Computed Tomography` `Review/Survey Paper`

**🎯 论文内容**

开发了一套基于iPad LiDAR的移动增强现实系统，实现无外部跟踪的胸外科虚拟模型与患者的标记化配准。

**💡 创新点**

采用手动交互式粗配准与LiDAR点云精细化、实时占据格覆盖度得分反馈三大创新，构建端到端无标记移动AR工作流。

**🔧 技术方法**

Unity+ARKit 处理可视化与同步RGB‑Depth；BlazePose 做体部分割；Fast Point Feature Histograms 与 RANSAC 进行初始配准；点对平面ICP 细化；占据格覆盖度算法实时评估数据完整性。

**📊 数据集**

以Visible Human Project（男性CT）生成的体表与器官模型做虚拟模型，配合3D打印物理模型与18个标记点进行评估。

**📈 对比分析**

在16位受试者中，粗配准mTRE约12.8 mm；自动精细化后mTRE下降至3.3 mm，完成时间约4 min；系统可用性SUS 84，显示高可用与准确。

**⚠️ 局限性**

仅基于静态CT与软组织形变未考虑，评估受限于少数临床医师与固定顺序，且设备依赖低分辨LiDAR，可能受表面反射影响。

---

## 585. REGLAT at ArGuard Shared Tasks: An Ensemble-Based Model Integrating NER for Harmful Content Detection in Arabic

**arXiv ID:** 2610.37265 | [PDF](https://arxiv.org/pdf/2610.37265v1)

**作者:** Ahmed M. Fetouh `[一作]` (Benha University), Hamada Nayel `[通讯]` (Benha University)

**关键词:** `Computation and Language` `Classification` `Recognition` `Transformer` `Supervised Fine-Tuning` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

本论文提出了一种两阶段管道，将 GLiNER 的实体识别结果与 MARBERT 的文本分类相结合，以对阿拉伯语用户提示进行细粒度有害内容分类。

**💡 创新点**

创新点在于通过将实体信息序列化为结构化文本块并与原始提示拼接，显式提供实体级别的线索，弥补单纯上下文表示对细分类别区分的不足。

**🔧 技术方法**

核心技术包括零样本泛化 NER 模型 GLiNER、预训练阿拉伯语 Transformer MARBERT、实体序列化格式设计以及微调策略。

**📊 数据集**

使用了 ArGuard 2026 共享任务的训练集（16,835 条提示）和开发集（2,405 条）进行评估，并在官方盲测集上进行最终测试。

**📈 对比分析**

在开发集上，该系统取得宏观 F1 为 0.886、准确率 0.897；在盲测集上，子任务 1 的准确率 0.793、宏观 F1 0.735，子任务 2 的准确率 0.770、宏观 F1 0.754，显示出优越的细粒度分类性能。

**⚠️ 局限性**

主要局限包括：不同类别之间语义重叠导致误分类、实体识别错误对模型输入的影响、输入长度限制导致信息截断，以及在不同方言或更广泛数据上的泛化能力不足。

---

## 586. Preserving Separation Logic Specifications through Compilation

**arXiv ID:** 2610.37244 | [PDF](https://arxiv.org/pdf/2610.37244v1)

**作者:** Carmine Abate `[一作]` (Barkhausen Institut), Sebastian Ertel `[通讯]` (Barkhausen Institut)

**关键词:** `Programming Languages` `AI Code Assistant` `Computational Efficiency` `Explainability and Interpretability` `Formal Methods` `Formal Verification` `Separation Logic` `Resource-aware Simulation` `Mechanized Proof` `Review/Survey Paper`

**🎯 论文内容**

研究了如何在编译过程中保持分离逻辑的弱前提规范，并提出了一种基于资源感知交互树的模拟技术，将源程序的资源所有权与目标程序的操作对应，并验证了闭合程序以及模块化编译与链接时的规范传递。

**💡 创新点**

创新点在于将分离逻辑的弱前提保持与编译过程结合，利用资源感知的交互树模拟实现了编译时资源所有权的跟踪，并实现了模块化编译时规范的无缝传递。

**🔧 技术方法**

采用分离逻辑、弱前提（WP）保持、资源感知交互树模拟，并在 Rocq 证明助手中进行机械化验证。

**📊 数据集**

论文未使用任何外部数据集；验证完全基于形式化证明。

**📈 对比分析**

本研究未进行实验性比较或性能评估；主要通过形式化证明展示了方法的正确性。

**⚠️ 局限性**

局限性包括：只针对结构化命令式语言到类似汇编语言的编译；未考虑并发、动态链接或更复杂的目标指令；假设目标操作无源对应物。

---

## 587. Spine: Compiling Subtask Trees into Structured Context for Long-Horizon Coding Agents

**arXiv ID:** 2610.37224 | [PDF](https://arxiv.org/pdf/2610.37224v1)

**作者:** Jiahong Xiang `[一作]` (Southern University of Science and Technology), Yuqun Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `Software Engineering` `AI Code Assistant` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Sequential` `Benchmark`

**🎯 论文内容**

提出了 Spine 接口与运行时，允许编码代理显式声明子任务边界，并通过持久化执行树在每一次模型调用前即时编译上下文，既保留未完成子任务的交互，又用已完成子任务的结果代替细节，避免历史膨胀和冗余工具调用。

**💡 创新点**

①在模型上下文中显式表示子任务边界与完成状态；②将子任务粒度与执行代理分离，允许同一代理在不创建额外子代理的情况下递归执行嵌套子任务；③通过持续的 SpineTree 在每次调用前“即时编译”上下文，替换已完成子任务的详细交互，仅保留未完成子任务的交互，从而兼顾 KV 缓存重用。

**🔧 技术方法**

设计并实现了四个生命周期命令（Open、Close、Next、Spawn）与 SpineTree（持久化树结构），前端后端解析器，使用 Rust/TypeScript 实现；集成到 CodeTuner（Rust）和 CodeAgent（Node/TypeScript）两大主流 harness，并在 ChatGPT、Gemini、LLaMA‑70B、Mistral 等大型模型上进行实验。

**📊 数据集**

在三个公开长周期编码基准上评估：repo‑evolution（GitHub 仓库演化任务）、codex（完整程序重构任务）以及 open‑ended（开源工程任务）。此外，在 CodeAgent 上使用多模型（ChatGPT、Gemini、LLaMA‑70B）进行交叉评测。

**📈 对比分析**

与多种上下文管理基线（原生压缩、反射、计划‑分支‑返回、递归子代理）在 CodeTuner 上对 repo‑evolution 进行对比，并在 CodeAgent 上对三大基准进行对比。Spine 在所有基准上提升 Score 约 9.5–17.5%，repo‑evolution 上提升 11.8%，显著降低 token 与成本（约 32%–73%），在多模型、多系统下均保持优异。

**⚠️ 局限性**

评估仅覆盖公开基准与两种主流 harness，未检验更大规模真实项目或多语言环境；实现依赖代理遵循显式命令，若不遵循可能失效；递归子代理仍带来一定通信开销；模型对结构化上下文的理解仍有限，可能在复杂嵌套或极长任务中出现误判。

---

## 588. ReForest: Interpretability Meets Reasoning in Isolation Forest

**arXiv ID:** 2610.37316 | [PDF](https://arxiv.org/pdf/2610.37316v1)

**作者:** Juyoung Seo `[一作]` (Sungkyunkwan University), Moohong Min `[通讯]` (Sungkyunkwan University)

**关键词:** `Artificial Intelligence` `Anomaly Detection` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Auto Encoder` `Prompt Engineering` `Mixture of Experts` `Tabular` `Biomedical Data` `Finance Related`

**🎯 论文内容**

提出了一种名为 ReForest 的混合异常检测框架，将 Isolation Forest 的统计高效性与大型语言模型（LLM）的语义推理能力结合起来，使用路径感知的 LLM 集成和不确定性感知的动态集成实现自适应决策。

**💡 创新点**

创新点包括：①将 Isolation Forest 的决策路径转换为自然语言，为 LLM 提供可解释的统计依据；②基于实例级不确定性动态调节统计模型与 LLM 的权重；③双视角可解释性，提供统计路径和语义推理的解释；④自适应预筛选策略，显著降低 LLM 计算开销。

**🔧 技术方法**

技术方法包括：Isolation Forest、深度自然语言处理（LLM）推理、路径深度提取、自然语言转换、Z-score 归一化、sigmoid 转换、动态权重 α(x) 计算与加权融合。

**📊 数据集**

使用四个真实世界数据集：Hepatitis C Virus（医疗）、Stroke Prediction（医疗）、Home Equity Loan（金融）、Loan Default Prediction（金融）。

**📈 对比分析**

与 IForest、LOF、ECOD、AutoEncoder 等基线在 F1、Recall、Precision 等指标上进行比较。ReForest 在医疗和金融场景平均提升 Recall 约21% 和 F1 约10%，在 HCV、Stroke 和 Loan Default 数据集实现最高 F1 分数；在 HMEQ 数据集仅与 AutoEncoder 接近，提升有限。

**⚠️ 局限性**

局限性包括：对纯统计模式（如 HMEQ）提升有限；需要 LLM 推理仍带来计算成本，尽管预筛选已降低到 15–20%；过多路径可能导致信息冗余和噪声；不确定性校准与领域知识集成仍有待进一步改进。

---

## 589. EdgePace: Deadline- and Freshness-Aware LLM Scheduling on Unified-Memory Platforms

**arXiv ID:** 2610.37307 | [PDF](https://arxiv.org/pdf/2610.37307v1)

**作者:** Teng Mei `[一作]`, Yunming Xiao `[通讯]`

**关键词:** `Performance` `Computational Efficiency` `Optimization` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Video` `Text`

**🎯 论文内容**

开发了一个面向统一内存边缘设备的LLM推理调度运行时，能够在保证前景任务实时性和LLM反馈新鲜度的前提下，动态调度LLM推理以缓解内存带宽竞争。

**💡 创新点**

创新点在于：① 在预填和解码阶段均提供细粒度安全暂停点；② 通过仅使用四个PMU计数器，低成本估计前景任务的内存需求和阻塞情况；③ 结合前景任务的时间余量、LLM输出的新鲜度阈值，设计风险与新鲜度感知的三态调度策略。

**🔧 技术方法**

使用的技术包括：LLM模型分层细分（Fine Plan）、PMU监控、基于阈值的风险评分与三态调度，以及对现有llama.cpp推理框架的动态加载。

**📊 数据集**

数据集/工作负载：多摄像头语义反馈（OpenEQA视频子集）和MuJoCo机器人目标跟踪；模型为Qwen系列LLM（1.5B、8B、VLM版）。

**📈 对比分析**

与未调度、MLC-LLM、TensorRT、MNN、QAIRT等基线比较，在四种统一内存平台上，EdgePace将前景任务的截止时间错过率降低24.1%–96.1%，同时保留64.9%–98.5%的LLM吞吐量，且所有LLM输出均落在其新鲜度窗口内。

**⚠️ 局限性**

局限性包括：① 需要为每种LLM和平台生成Fine Plan，初始配置耗时；② 只针对统一内存平台，无法直接扩展到分离显存的系统；③ 在极端高并发场景下，PMU采样间隔和阈值设置仍可能导致误判。

---

## 590. Spiking Neuromorphic Control for Periodic Orbits Stabilization in Underactuated Robots: A Greedy Lyapunov-Based Approach

**arXiv ID:** 2610.37310 | [PDF](https://arxiv.org/pdf/2610.37310v1)

**作者:** Maksim Surov `[一作]` (Sirius University of Science and Technology), Oleg Sumenkov `[通讯]` (Sirius University of Science and Technology)

**关键词:** `Robotics` `Robotic Intelligence` `Optimization` `Spiking Neural Network` `Lyapunov-based Control` `Event-triggered Control` `Riccati Differential Equation` `B-spline Interpolation` `Hybrid System Analysis` `Trajectory Tracking` `Time Series` `Tabular`

**🎯 论文内容**

本文提出了一种基于事件触发脉冲的神经形态控制器，用于在欠驱动机器人中实现周期轨道的轨迹跟踪。

**💡 创新点**

创新点在于将脉冲控制策略与横向线性化相结合，构造周期性线性时变混合系统，并通过贪婪的Lyapunov方法设计触发条件，实现了对周期轨道的实际轨道稳定性保证。

**🔧 技术方法**

核心技术包括：事件触发脉冲控制、横向线性化、周期性线性时变系统的李雅普诺夫矩阵求解（Riccati微分方程）、混合系统分析与B样条插值实现实时控制。

**📊 数据集**

实验数据集来自物理Butterfly机器人，该机器人是一个两自由度欠驱动系统，实验记录了姿态角、球位、控制脉冲和滤波后控制输入等数据。

**📈 对比分析**

与传统连续控制方法相比，实验结果表明脉冲控制实现了相似甚至更好的跟踪性能，同时在存在测量噪声、动力学不确定性和延迟的情况下仍保持鲁棒性。

**⚠️ 局限性**

局限性包括：理论分析仅证明了实际轨道稳定性，未给出完整的非线性鲁棒性证明；目前仅适用于单一控制信号；对多输入系统的推广及更复杂的触发策略仍待研究。

---

## 591. Structured Representation Probing for Risk Anticipation in End-to-End Autonomous Driving

**arXiv ID:** 2610.37299 | [PDF](https://arxiv.org/pdf/2610.37299v1)

**作者:** Jiawei Liang `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `Machine Learning` `Autonomous Driving` `Anomaly Detection` `Computational Efficiency` `Representation Learning` `Transformer` `Contrastive Learning` `Supervised Fine-Tuning` `Vision-Language-Action Model` `Video` `Point Cloud` `Tabular` `Benchmark`

**🎯 论文内容**

设计并实现了 Structured Representation Probe（SRP），通过读取冻结的端到端自动驾驶政策的结构化内部表示（actor、ego、map），在不修改政策的情况下实时输出风险警报。

**💡 创新点**

创新点包括：①源特定线性风险方向以保留不同表示源的语义差异；②使用 cardinality‑normalized log‑mean‑exp (LME) 池化既能强调稀疏高风险元素，又能降低令牌数对分数的影响；③在监督中加入事件类型辅助头，联合训练共享的风险方向，从而提升对多种故障模式的预测能力；④整体设计极其轻量，仅数千参数，运行时开销低于 3%。

**🔧 技术方法**

技术细节：线性投影投射到风险方向；LME 池化（源内及全局）；事件窗口标签构造；二元交叉熵（加权）与辅助事件头损失；归一化统计；单阈值在线报警。使用 CARLA Bench2Drive 环境收集内部表示和硬失败日志。

**📊 数据集**

数据集：Bench2Drive（220条道路、不同天气与镇区）用于训练、验证和测试；还在 CARLA Leaderboard 2.0（20条验证路段）及 220 条新 Bench2Drive 路段上评估迁移性能。

**📈 对比分析**

对比方法：Temporal Consistency、Task‑Aware Risk、TRFD 三种运行时监测基线；在 DriveTransformer、UniAD、VAD 三个端到端模型上进行评估。指标包括 PR‑AUC、ROC‑AUC、帧召回 (R_F)、事件召回 (R_E) 和惩罚性领先时间 (pLead)。SRP 在所有模型上均明显提升 PR‑AUC、ROC‑AUC 和 pLead，尤其在 DriveTransformer 的 ROC‑AUC 从 0.715 提升到 0.833，pLead 从 0.633s 提升到 0.701s；UniAD 与 VAD 亦出现显著提升。

**⚠️ 局限性**

局限性：仅在仿真环境验证，真实道路部署尚未验证；事件窗口标签只能捕捉到记录的硬失败，无法反映内部不安全状态；对不同路段的迁移表现仍有波动，尤其在新路段或不同仿真版本中；对源（actor/ego/map）重要性的解释尚不充分，进一步因果分析和多源融合仍需研究。

---

## 592. GradCO: Behavior-Aware Gradient-Conflict Offloading for Text-Based Person Anomaly Search

**arXiv ID:** 2610.37267 | [PDF](https://arxiv.org/pdf/2610.37267v1)

**作者:** Zhiheng Fu `[一作]`, Yupeng Hu `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Anomaly Detection` `Retrieval` `Transformer` `Contrastive Learning` `Mixture of Experts` `Text` `Image` `Multimodality`

**🎯 论文内容**

本文提出了 GradCO 框架，用于文本描述下的人群异常检索（TPAS），通过分离身份与行为特征实现精准匹配。

**💡 创新点**

创新点在于三大模块：B‑MoE 用来分离外观、正常行为与异常行为；COL 通过对冲突样本的梯度卸载解决“同身份不同行为”导致的梯度冲突；HIA 采用查询感知的加权聚合避免特征聚合失真。

**🔧 技术方法**

技术上采用多专家网络、对比学习损失、offload anchor 机制、方向多样性正则化以及查询条件的注意力聚合。

**📊 数据集**

使用的数据集包括 TPAS 基准 PAB、Multi‑Weather、OOD UCC，以及 TIPR 基准 CUHK‑PEDES、ICFG‑PEDES、RSTPReID、UFine6926/UFine3C。

**📈 对比分析**

与现有方法（如 CMP、CLIP、X‑VLM 等）相比，GradCO 在 R@1、R@5、R@10、mAP、Multi‑Weather 平均和 OOD 指标上均取得显著提升，例如 PAB R@1 由 83.06% 提升至 85.39%，Multi‑Weather R@1 提升至 72.19%，UCC OOD R@1 提升至 76.08%。

**⚠️ 局限性**

局限性包括参数量和显存消耗略高（247M 参数，18GB 显存），推理速度略慢（2.3 ms/样本），在极度相似的同身份异行为样本仍可能误检，并且对训练数据（大部分为合成）有一定依赖，影响跨域迁移能力。

---

## 593. GameVerse: A Minute-Scale Gameplay Dataset for Long-Horizon Interactive World Modeling

**arXiv ID:** 2610.37257 | [PDF](https://arxiv.org/pdf/2610.37257v1)

**作者:** Kang He `[一作]` (Alaya Lab), Yongtao Ge `[通讯]` (Alaya Lab)

**关键词:** `Computer Vision and Pattern Recognition` `Data Synthesis` `Transformer` `Vision Language Model` `Contrastive Learning` `Supervised Fine-Tuning` `Video` `Text`

**🎯 论文内容**

构建并公开了 GameVerse：一套以分钟级长时序为主的游戏玩法视频数据集，包含69.7K段、1,655小时、349.6M帧，并提供每段分钟级一致的摄像机轨迹和四层级结构化字幕；

**💡 创新点**

首次公开整合分钟级连续视频、可靠摄像机姿态与多粒度结构化字幕的数据集，为长时序交互式世界建模提供了关键的监督信号；

**🔧 技术方法**

采用多阶段视频质量评估、CLIP+TransNetV2场景切割、ViPE摄像机姿态估计与验证、Kimi-K2.5视觉语言模型生成四层级字幕；

**📊 数据集**

采集自11,002条YouTube游戏录制，覆盖110+ AAA 与独立游戏，包含第一/第三人称视角与多种玩家行为；

**📈 对比分析**

在相同backbone上对 HY‑World 与 LTX‑Video 进行 GameVerse 微调，显著提升 VBench（60s生成的时间一致性与运动平滑度）和 WorldScore（3D一致性、相机控制与视觉质量）等指标，超越原始模型；

**⚠️ 局限性**

仅覆盖游戏场景，摄像机轨迹与字幕仍存在少量误差，连续剪辑仅限两分钟，未能单独验证各模块贡献，构建成本高。

---

## 594. TaskBridge: Technical Report for SIGIR 2026 eCommerce Challenge

**arXiv ID:** 2610.37231 | [PDF](https://arxiv.org/pdf/2610.37231v1)

**作者:** Zhiheng Fu `[一作]` (Shandong University), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `Information Retrieval` `Recommendation System` `Retrieval` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text`

**🎯 论文内容**

TaskBridge 提出了一种基于目录语义锚定的任务导向电商查询重写框架，能够将用户的活动描述转换为可被 BM25 检索的产品表达，并在不同协议下返回一个或多个改写查询。

**💡 创新点**

创新点包括：① 构建标题词汇表和语义锚点以桥接任务意图与目录词汇；② 统一生成五种候选重写（原始、反馈、投票、需求、产品）以覆盖不同重写策略；③ 引入协议感知选择模块，结合 BM25 结果、重写器评分和加权 Jaccard 重叠，实现单查询与多查询的 Pareto 最优选取。

**🔧 技术方法**

使用的技术主要是大型语言模型（Qwen3.5-9B 进行生成与投票，Qwen3-VL-Embedding-8B 进行语义锚检索，Qwen3-VL-Reranker-8B 进行重写评估），以及传统的 BM25 检索和基于倒排索引的词汇提取与评分。

**📊 数据集**

实验基于 SIGIR 2026 eCommerce Query Reformulation 挑战的数据集，包含任务导向查询、产品标题语料以及目录级语义资源；在 Open Phase 与 Closed Phase 两个评测阶段进行评估。

**📈 对比分析**

与官方原始查询 BM25 基线相比，TaskBridge 在 Automatic 协议下的 Task Completion NDCG@10 从 0.42 提升至 0.49，Recall@100 与 MAP 亦略有提升；在 Interactive 协议下进一步提升至 0.58、0.33、0.23。Closed Phase 同样表现出 Automatic 与 Interactive 的明显差距，表明多查询策略能更好覆盖产品表达。

**⚠️ 局限性**

局限性包括：① 仅在推理时使用，无监督学习或微调，可能限制对新领域或语言的适应性；② 依赖大型语言模型，计算成本高；③ 对语义锚点提取与投票阈值的设定较为经验化，鲁棒性待进一步验证。

---

## 595. Peers4U: Designing Learner-Centered Virtual Peers for Children's Multimodal Mathematics Learning

**arXiv ID:** 2610.37238 | [PDF](https://arxiv.org/pdf/2610.37238v1)

**作者:** Zhi Zheng `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `Human-Computer Interaction` `Education` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Data-Centric Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Generative Adversarial Network` `Text` `Multimodality`

**🎯 论文内容**

本研究设计并实现了一个以学习者为中心的多模态虚拟同伴系统Peers4U，用于提升儿童数学概念理解。

**💡 创新点**

创新点在于将LLM驱动的虚拟同伴根据学习者的认知水平个性化配置，并通过多模态解决方案和可检视的过程实现混合主动交互。

**🔧 技术方法**

使用技术包括大语言模型（LLM）、认知分析管道、动态生成同伴认知配置与多模态素材生成。

**📊 数据集**

使用的数据集主要是儿童在“表示17”任务中的初始解决方案与生成的同伴艺术品，没有公开大型数据集。

**📈 对比分析**

通过与基于人物设定的基线进行对比实验，22名儿童参与者显示学习者中心的主题编排在解决方案数量和多样性上显著优于基线，且专家评估支持其教学效果。

**⚠️ 局限性**

局限性包括样本规模有限、实验仅聚焦单一数学任务、对LLM生成内容的可靠性与安全性仍需进一步验证。

---

## 596. AnalogScout: Simulation-Efficient PVT-Aware Analog Sizing with LLM Agents

**arXiv ID:** 2610.37319 | [PDF](https://arxiv.org/pdf/2610.37319v1)

**作者:** Yichen Long `[一作]` (Fudan University), Keren Zhu `[通讯]` (Fudan University)

**关键词:** `Hardware Architecture` `Optimization` `Computational Efficiency` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Mixture of Experts` `Tabular` `Text` `Benchmark`

**🎯 论文内容**

利用大语言模型（LLM）代理在仿真预算有限的情况下，针对多工艺/电压/温度（PVT）角点进行模拟高效的模拟电路尺寸优化。框架通过精英设计重用、解析 gm/Id 本地尺寸意图以及全角点验证的全局微调，实现对 OTA 结构的尺寸调整与性能验证。

**💡 创新点**

1) 引入了精英池与完全匹配缓存，显著减少重复仿真；2) LLM 通过“gm/Id 意图”指令让解析器仅用一次或三次操作点仿真即可确定子模块尺寸；3) 将 LLM 与传统全局优化（如 CMA-ES）相结合，兼顾搜索质量与仿真效率；4) 在固定的 16k 次 SPICE 仿真预算下，能在仅占预算 7% 时就达到与 CMA-ES 同等功率水平。

**🔧 技术方法**

- 大语言模型代理（ChatGPT 等）与固定评估器交互，进行设计编辑与尺寸意图设置；
- 精英池与 exact‑match 缓存；
- gm/Id LUT 解析器（一次操作点仿真即可推算子模块尺寸）；
- 结构重排（同族模块互换）；
- 全局微调（CMA‑ES）与全角点验证；
- Monte‑Carlo 误差与饱和度检查。

**📊 数据集**

- 三个 SKY130 OTA 基准（Miller 2‑stage、Folded Cascode 2‑stage、3‑stage NMC）
- 一个 GF180MCU Miller OTA（电压 3.3 V）
- 以手工 gm/Id 设计为起点，按 PDK 规范进行 PVT 约束。

**📈 对比分析**

与 CMA‑ES、BO、LAS‑Style、ACP‑Style 等基线在相同 16k 次仿真预算下比较。评估指标为典型角点功率、所有 PVT 角点规范满足情况。结果显示 AnalogScout 在三种 OTA 上均实现了 43% 左右的功率下降，且在 7% 的仿真预算即可达到 CMA‑ES 的最终功率；在 GF180MCU Miller OTA 上功率下降达 59.6%。

**⚠️ 局限性**

- 本地 gm/Id 解析在仿真效率上显著提升，但相对与同等预算的重新启动并未带来显著功率改进；
- 结构重排对功率提升作用有限；
- GF180MCU 测试仅覆盖 Miller OTA，尚未验证对更复杂 OTA 的普适性。

---

## 597. A Lightweight Attention-Based CNN with Saliency-Map Interpretability for Skin Cancer Detection

**arXiv ID:** 2610.37314 | [PDF](https://arxiv.org/pdf/2610.37314v1)

**作者:** Sanjana Islam Kasfia `[一作]` (International Standard University), Abir Mahmud Shahariar `[通讯]` (International Standard University)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Explainability and Interpretability` `Computational Efficiency` `Convolutional Neural Network` `Gradient visualization (Saliency Map)` `Transfer Learning` `Adam Optimization` `Cross-Entropy Loss` `Image`

**🎯 论文内容**

构建了轻量级的注意力CNN模型，用于皮肤癌二分类并加入可解释的Saliency Map；

**💡 创新点**

将Squeeze‑and‑Excitation通道注意力嵌入自定义网络，并在轻量化设计中加入可解释性可视化；

**🔧 技术方法**

采用卷积神经网络、通道注意力、迁移学习、梯度可视化（Saliency Map）、Adam优化、交叉熵损失等技术；

**📊 数据集**

使用公开的二分类皮肤病变数据集（训练2637张图像，测试660张图像）；

**📈 对比分析**

与VGG16/VGG19/ResNet101/MobileNet等预训练模型对比，准确率保持87%，参数量仅1.95M（比MobileNet更小），存储占用7.45MB，PR‑AUC≈0.943；

**⚠️ 局限性**

仅限二分类，样本量有限，对光照、毛发等噪声的鲁棒性有待提升。

---

## 598. Where Should the Evaluation Budget Go in LLM-Driven Agent Scaffold Evolution?

**arXiv ID:** 2610.37276 | [PDF](https://arxiv.org/pdf/2610.37276v1)

**作者:** Sichun Luo `[一作]` (University of Hong Kong), Qi Liu `[通讯]` (University of Hong Kong)

**关键词:** `Machine Learning` `Neural Architecture Search` `Computational Efficiency` `Optimization` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Mixture of Experts` `Score-based Model` `Text` `Benchmark`

**🎯 论文内容**

针对LLM驱动的代理框架（agent scaffold）进化过程中的评估资源分配问题，提出了一种将稀疏搜索阶段评估与针对性最终确认阶段评估分离的框架，称为<省略>，通过先在前期全量评估一小批候选结构来构建信息量高的共用核心与响应模型，然后在搜索阶段对每个候选使用共享信息量高核心与少量自适应项进行稀疏评估，最后再利用累积观测和warm‑started successive halving对所有候选进行目标确认，以较低评估成本获得与全量评估相近甚至更优的最终性能。

**💡 创新点**

① 将评估任务拆分为搜索反馈与最终选择两角色，提出了针对性资源分配策略；② 在搜索阶段使用共享信息量高核心与固定响应模型实现可比性强的稀疏评估；③ 在最终选择阶段利用warm‑started successive halving重用已有观测，实现更精确的最终挑选；④ 综合上述策略在四大基准与两种进化框架上均实现约一半评估成本下与全量评估相当或更优的表现。

**🔧 技术方法**

多项技术组合：1）基于Beta后验与信息量度量的项筛选；2）低秩多维项目响应模型（MIRT）用于将稀疏观测映射到完整评估尺度；3）共享核心与Gumbel‑top‑k自适应采样；4）warm‑started successive halving与交叉拟合响应模型的差分估计；5）对评估成本进行细粒度预算分配与回放分析。

**📊 数据集**

四个基准：HotpotQA、HoVer、DROP、IFBench，每个基准均划分为SEARCH(300)、DEV(150)、TEST(300/294)三部分；使用两种进化框架ShinkaEvolve与AdaEvolve；评估器基于Qwen3.5‑9B与DeepSeek‑V4‑Flash‑0731。

**📈 对比分析**

与五种基线策略比较：Skip Eval、Successive Halving、Random、Cascade、Full。实验显示，在ShinkaEvolve与AdaEvolve两框架、四基准下，<省略>平均测试得分与Full相近或略优，且评估细胞成本约为Full的一半，随机或Cascade方法在相同成本下性能明显落后。

**⚠️ 局限性**

限制：① 评估分配需要预先全量评估一批候选（warm‑up），对计算资源仍有一定需求；② 共享核心与固定响应模型假设候选间响应相关性较高，若候选多样性极高可能影响比较可靠性；③ 方案针对LLM代理脚本进化，迁移到其他任务或模型需要进一步验证。

---

## 599. Understanding Everyday Communication Patterns, Stress, and Productivity in Hybrid Work through Egocentric Wearable Audio Sensing

**arXiv ID:** 2610.37270 | [PDF](https://arxiv.org/pdf/2610.37270v1)

**作者:** Tiantian Feng `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**关键词:** `Sound` `Knowledge Distillation` `Representation Learning` `Data-Centric Learning` `Anomaly Detection` `Computational Efficiency` `Explainability and Interpretability` `Safty and Privacy` `Convolutional Neural Network` `Transformer` `Knowledge Distillation` `Contrastive Learning` `Audio`

**🎯 论文内容**

本研究开发了一套前向可穿戴音频传感系统，实时提取低维声学特征并离线完成前景/背景说话者分离，从而计算工作中的说话负荷和说话活跃度，用于监测混合工作场景下的沟通模式与情绪负担。

**💡 创新点**

创新点在于：①将可穿戴设备与低维特征提取结合，保证隐私与成本；②采用教师-学生知识蒸馏，将大规模语音基础模型（WavLM/Whisper）的知识迁移至紧凑 ResNet，显著提升前景说话者识别精度；③通过混合线性模型将说话负荷与活跃度与自评压力、生产率关联，揭示办公与居家环境的差异。

**🔧 技术方法**

使用技术包括：Android Atom 设备 + openSMILE 低维特征提取；ResNet 结构的前景/背景说话者分离模型；WavLM‑Large 与 Whisper‑Tiny/Small/Large 的教师模型进行知识蒸馏；混合线性模型用于关联分析。

**📊 数据集**

实验数据集包括：MMCSG、EasyCom（两者均为智能眼镜收集的亲身式语音数据）以及 ICSI‑Meeting Corpus（多说话者会议语音）。

**📈 对比分析**

与教师模型相比，蒸馏后的 Distilled‑ResNet 在三大数据集上的 DER 分别降低约 1.9–2.8 点，性能与 Whisper‑Tiny 相当；在加入室内噪声增强的低 SNR 场景下，DER 进一步下降 5.8–2.7%。

**⚠️ 局限性**

局限性包括：样本量仅 16 名办公人员，无法全面代表所有混合工作人群；仅使用低维 MFCC 等特征，可能限制对复杂说话者区分的精度；系统受设备佩戴率和佩戴时长不一的影响；未结合其他多模态数据（如姿态、心率）进一步丰富工作体验分析。

---

## 600. Watermark Basins: Black-Box Detection of Distilled LLMs

**arXiv ID:** 2610.37266 | [PDF](https://arxiv.org/pdf/2610.37266v1)

**作者:** Hongyan Chang `[一作]` (FAIR, Meta Superintelligence Labs), Alexandre Mourachko `[通讯]` (FAIR, Meta Superintelligence Labs)

**关键词:** `Cryptography and Security` `Knowledge Distillation` `Anomaly Detection` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Contrastive Learning` `Text`

**🎯 论文内容**

提出了“水印基底”概念，利用训练时水印的弱方向在参数空间形成持久性基底，进而实现对未授权蒸馏模型的黑盒检测

**💡 创新点**

创新点在于将文本水印的可检测信号迁移到参数空间，通过梯度投影与任务子空间剔除得到可量化的对齐分数，显著提升了检测鲁棒性和准确性

**🔧 技术方法**

采用固定清洗探针模型、随机投影、主成分分析去除任务方向、关键字对齐排名检验等技术，实现了无访问权重的黑盒检测

**📊 数据集**

使用了公开的OpenR1-Math-220k、GSM8K、Math-500等推理数据集，结合DeepSeek、GPT-OSS、Qwen、Llama 等多种教师模型和多种文本水印方案进行实验

**📈 对比分析**

与现有基于token计数的黑盒检测相比，在1%假阳性率下实现了79%（n-gram=1）/45%（n-gram=3）的召回率，显著高于传统方法的29%/1.3%

**⚠️ 局限性**

局限性包括：检测只能针对审计域内的文本，跨域或持续SFT后文本信号可能衰减，且依赖于教师生成的水印样本，若水印被彻底擦除或未使用，方法失效

---

## 601. Your Policy Is Secretly a Harness: The Decision Geometry of Agent Interfaces

**arXiv ID:** 2610.37255 | [PDF](https://arxiv.org/pdf/2610.37255v1)

**作者:** Jian Mu `[一作]` (Hong Kong University of Science and Technology), Yao Shu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `Machine Learning` `Reinforcement Learning from Human Feedback` `Optimization` `Explainability and Interpretability` `Transformer` `Large Language Model` `Reinforcement Learning` `Prompt Engineering` `Agentic AI` `Softmax` `KL Regularization` `MMD-style Mismatch Measurement` `Text`

**🎯 论文内容**

对语言模型代理进行后训练（post‑training）后，研究其在不同 harness（提示、工具、控制逻辑）下学到的偏好，并探讨在 harness 变更时模型如何迁移以及何时需要适配。

**💡 创新点**

①证明最优策略的对数优势恰好编码了 harness 的相对动作分数，post‑training 将模型的动作偏好逼近该相对分数并使其对特定 harness 产生依赖；②将部署收益拆解为“ceiling”（harness 可提供的最高收益）与“mismatch”（模型偏离最优策略所损失的收益）两项，并给出“mismatch”可通过采样动作估计；③利用该估计预测在新 harness 上是否需要适配以及适配后是否能提升性能。

**🔧 技术方法**

KL 正则化的控制理论、软贝尔曼方程、策略梯度后训练、软最大化（softmax）对数优势、动作分数几何、基于教师模型的误差上界、采样状态折扣分布、MMD 风格的失配度量。

**📊 数据集**

MiniWoB++（六种 harness 组合）和 Terminal‑Bench Lite（五种编码代理 harness）作为评测数据集；使用 Qwen3.5‑2B（小模型）和 Qwen3.5‑9B（大模型）做后训练，并以 GPT‑5.6‑luna 为教师。

**📈 对比分析**

通过对比冻结权重下不同 harness 的任务成功率、通过进一步训练（adaptation）后与源检查点（source checkpoint）的比较、以及通过“ceiling”与“mismatch”两项预测适配收益。实验显示：post‑training 在其训练 harness 上显著提升成功率并将误差下降 70‑90%；当 harness 变更仅改变呈现时，即使最优决策不变，冻结模型表现也可从 85% 跌至 19%；在 batched‑click 新 harness 下，冻结模型下降但经过适配后能超过原 harness；“mismatch”估计能准确预测 10‑多分之差的收益变化。

**⚠️ 局限性**

① 需要已知或可近似的最优策略（或教师模型），否则“ceiling”和“mismatch”无法精准估计；② 研究聚焦于离散动作空间的 harness，连续或大规模多模态任务仍未验证；③ 对 harness 设计的假设（如 softmax 解码、参考策略固定）在更复杂的多步骤交互中可能不成立；④ 只评估了少量任务，未探讨在更大规模或更异构任务上的泛化能力。

---

## 602. GraphRAGProv: GraphRAG Meets How-Provenance

**arXiv ID:** 2610.37237 | [PDF](https://arxiv.org/pdf/2610.37237v1)

**作者:** Wadie El Amrani `[一作]` (Independent), Luis Galárraga `[通讯]` (INRIA)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Knowledge Distillation` `Recommendation System` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Graph` `Benchmark`

**🎯 论文内容**

本研究构建了源文本与 RDF 语句的双向映射，利用 LLM 生成 SPARQL 查询并执行，同时记录 how‑provenance 多项式，形成可追踪的答案生成链。

**💡 创新点**

创新点在于将 RDF‑star 重写与 how‑provenance 结合，实时跟踪每条语句在答案推导中的作用，并提供完整的源文本验证路径。

**🔧 技术方法**

采用 DeepSeek‑V4‑Flash LLM 进行语义抽取与 SPARQL 生成，GraphDB 存储与查询，Rust 实现的 NPCS 进行查询重写以计算 how‑provenance。

**📊 数据集**

实验使用 2WikiMultiHopQA 与 MuSiQue 两个多跳 QA 基准，共 400 条案例。

**📈 对比分析**

通过对 400 条案例的跟踪完成率（311 条完整追踪）与传统仅返回答案的方式对比，展示了在约 78% 案例中可获得完整源文本追踪，显著提升错误定位效果。

**⚠️ 局限性**

受限于 LLM 的抽取与生成误差、实体不一致以及缺乏检索阶段，导致部分答案缺失或错误；系统性能受 LLM 与 GraphDB 推理速度的限制。

---

## 603. AutoRSE: Evaluating LLM Agents as Recommendation Engineers for Self-Evolving Recommender Systems

**arXiv ID:** 2610.37228 | [PDF](https://arxiv.org/pdf/2610.37228v1)

**作者:** Rui Tian `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**关键词:** `Information Retrieval` `Recommendation System` `Optimization` `Data-Centric Learning` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Tabular`

**🎯 论文内容**

提出了AutoRSE评估框架，用以统一测评LLM驱动的推荐系统工程代理（RSE‑Agents）在工业级推荐任务上的自适应改进能力；

**💡 创新点**

创新点在于：①建立六大代表性推荐任务的标准化测试平台；②引入轨迹级分析拆解代理优化过程，区分工程优化与算法创新；③通过“idea‑injection”实验验证创新想法是提升性能的关键瓶颈；

**🔧 技术方法**

技术包括LLM代理（Claude Opus 4.8、GPT‑5.5、DeepSeek V4 Flash）与三种代理框架（Claude Code、Codex、Terminus‑2），统一的沙盒执行环境与12 h训练预算；

**📊 数据集**

使用六个工业级数据集：Amazon Beauty、Amazon Kindle、MovieLens Latest、Criteo 5M、KuaiSAR v2、KuaiSAR v2等，覆盖生成推荐与CTR预测；

**📈 对比分析**

通过对比基线与代理最终提交的性能，平均提升16.1 %（相对提升），在五项任务中多数实现显著增益；轨迹分析显示91.7 %实验为工程优化，8.3 %为算法创新，算法创新越高相关提升越大；

**⚠️ 局限性**

限制在于代理目前缺乏真正的算法创新，主要依赖超参数调优、模型集成等工程手段，且自发想法生成效果差，导致性能提升受限。

---

## 604. Sharpness-Aware Muon: Matrix-Aware Optimization with Momentum-Guided Perturbations

**arXiv ID:** 2610.37225 | [PDF](https://arxiv.org/pdf/2610.37225v1)

**作者:** Zhiwei Zhai `[一作]` (Chinese University of Hong Kong), Ying-Jun Angela Zhang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `Machine Learning` `Optimization` `Score-based Model` `Contrastive Learning` `Gradient Accumulation` `Matrix Norm Analysis` `Random Linearization Stability Theory` `Image` `Text`

**🎯 论文内容**

提出了 MuSAM 与 MoMuSAM 两种将 sharpness‑aware 机制融入 Muon 的优化器，进一步提升了 Muon 的泛化性能。

**💡 创新点**

创新点在于：① 将 SAM 风格的局部最坏情况扰动与 Muon 的矩阵正交化动量结合，构造了 MuSAM；② 将 Muon 的动量本身作为时间聚合的锐度探测方向，消除额外梯度评估并显著提升扰动的稳定性，得到 MoMuSAM。

**🔧 技术方法**

使用了矩阵正交化动量（Polar 近似 + Newton–Schulz）、SAM 线性化扰动、矩阵范数分析、随机线性化稳定性理论以及梯度累积与归一化技术。

**📊 数据集**

在图像分类上使用 Fashion‑MNIST、CIFAR‑10 与 CIFAR‑100；在语言建模上使用 OpenWebText 训练 124M GPT‑2（NanoGPT）。

**📈 对比分析**

与 SAM、MSAM、VASSO 以及原始 Muon 进行对比；实验表明 MuSAM 在大多数任务上已超过基线，而 MoMuSAM 在准确率、验证损失与计算效率方面均取得最优或接近最优成绩，且对扰动半径 ρ 与学习率 η 更具鲁棒性。

**⚠️ 局限性**

局限性包括：① 仍需额外的梯度评估（尽管 MoMuSAM 仅一次，但相比纯 Muon 仍多一步）；② 对矩阵块的处理在大规模网络中的实现复杂度未充分评估；③ 理论分析在理想化假设下给出，实际训练中可能受其他超参或硬件约束影响。

---

## 605. BenchMeUp: Building Software Engineering Benchmarks from Natural-Language Specifications

**arXiv ID:** 2610.37283 | [PDF](https://arxiv.org/pdf/2610.37283v1)

**作者:** Islem Bouzenia `[一作]` (CISPA Helmholtz Center for Information Security), Michael Pradel `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `Software Engineering` `Benchmark` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text`

**🎯 论文内容**

构建了一个名为 BenchMeUp 的系统，能够自动从研究者给出的自然语言规范中生成软件工程基准，涵盖实例搜索、构造、验证与统一接口打包。

**💡 创新点**

创新点在于：①基于自然语言规范自动生成统一的搜索规则和构造计划；②使用多代理 LLM 并行搜索与实例构造；③在全新容器中重放构造脚本以验证可复现性；④将所有实例统一封装成标准命令行接口，保证评测一致性。

**🔧 技术方法**

采用的技术包括：大语言模型（Claude Opus 5、Deepseek‑v4‑flash）、多代理框架（搜索代理、构造代理）、基于 YAML 的规则与计划描述、容器化（Docker）重放验证、并行调度与经验共享、以及统一的 CLI 包装。

**📊 数据集**

使用的数据集涵盖四个软件工程任务：程序修复（Java），问题再现（GitHub Issues），编译器自适配（C/C++），以及静态分析警告修复（Java）。共收集约200+实例，源自 GitHub 上公开的开源项目。

**📈 对比分析**

实验对比了 BenchMeUp 与现有技术（RepairAgent、ReProAgent、BOCA、google‑java‑format 等）以及通用 LLM（Codex）。BenchMeUp 构造了 198 个实例，其中 180（91%）满足全部标准；平均每个实例构造耗时 36 分钟、成本 0.79 美元。与 Codex 的对比显示，Codex 仅 10/200（5%）满足全部标准。

**⚠️ 局限性**

局限性包括：①构造过程仍依赖 LLM 生成的检查脚本，可能存在误判；②弱的构造模型导致部分候选被丢弃，构造效率受限；③仅使用 GitHub 作为数据源，对非 GitHub 生态的可迁移性有限；④需要人工审查检查器与实例的正确性；⑤多代理运行成本与资源需求较高。

---

## 606. Multi-Label Agreement: A Flexible Framework for Inter-Annotator Reliability

**arXiv ID:** 2610.37292 | [PDF](https://arxiv.org/pdf/2610.37292v1)

**作者:** Tim Menzner `[一作]` (Coburg University of Applied Sciences), Jochen L. Leidner `[通讯]` (Coburg University of Applied Sciences)

**关键词:** `Computation and Language` `Data-Centric Learning` `Explainability and Interpretability` `Text` `Benchmark`

**🎯 论文内容**

提出 Multi-Label Agreement (MLA) 框架，用平均对偶 Hamming 一致性与不同基线校正评估多标签标注的一致性。

**💡 创新点**

创新点在于：①证明 MLA 等价于标签级 Fleiss/Scott 等系数的加权平均并与 Krippendorff’s α（Hamming 距离）相联系；②提供基于标签流行度的条件最小一致性诊断；③支持多种基线（pooled-marginal、AC1、PABAK）并进行敏感性分析。

**🔧 技术方法**

使用的技术包括：归一化 Hamming 距离、Fleiss/Scott/AC1/PABAK 基线模型、Krippendorff’s α、Monte Carlo 仿真、Python 实现。

**📊 数据集**

采用 Coburg 大学新闻句子偏见标注（31 类、4 句、3 评注者）作为案例数据；仿真生成合成多标签数据用于评估不同指标。

**📈 对比分析**

与 Augmented κ、boot‑F1、MASI、A_m 等指标比较，展示在不同 prevalence/agreement 场景下各基线的差异；仿真表明不同基线对相同数据给出截然不同的数值，反映其底层假设差异。

**⚠️ 局限性**

局限性：仅适用于完全、平衡的二元非层级标签；不处理缺失标注；Hamming 距离无法捕捉标签间依赖；示例样本量小；未给出置信区间或统计显著性检验。

---

## 607. RPSD: Retrieval-Privileged On-Policy Self-Distillation with Optimal Transport

**arXiv ID:** 2610.37281 | [PDF](https://arxiv.org/pdf/2610.37281v1)

**作者:** Dongyue Li `[一作]` (Nagoya University), Fei Ma `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `Artificial Intelligence` `Recommendation System` `Knowledge Distillation` `Reinforcement Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Retrieval-Augmented Generation` `Optimal Transport` `Tabular` `Sequential`

**🎯 论文内容**

提出了 Retrieval‑Privileged on‑policy Self‑Distillation (RPSD)，通过在 LLM 生成推荐时使用对齐的跨用户轨迹作为教师的特权信息，完成对 token‑级别信用的重新分配；

**💡 创新点**

首次将 on‑policy 自蒸馏与基于时序感知最优传输的相似用户检索相结合，实现了细粒度 token 级信用分配与协作监督，解决了传统 GRPO 的信用盲区与单用户监督限制；

**🔧 技术方法**

采用 LLM 微调 + GRPO 强化学习、on‑policy 自蒸馏、时序感知最优传输、Diversity‑aware 选择以及 token‑级优势重塑等技术；

**📊 数据集**

在 MovieLens‑1M、Netflix 与 MicroLens 三大公开数据集上进行实验；

**📈 对比分析**

与非 LLM 基线（GRU4Rec、SASRec、LightGCN 等）和 LLM 基线（TallRec、LLaRA、SPRec、RecZero 等）进行对比，RPSD 在 HR@3/5 与 NDCG@3/5 上实现显著提升，尤其在候选集扩大的情况下表现更优；

**⚠️ 局限性**

受限于检索质量与跨域适配，RPSD 在源域与目标域差异较大时可能出现负迁移，且对检索候选的依赖导致推理阶段不受影响但训练成本增加；

---

## 608. Fast Approximate Line Nearest Neighbor Search

**arXiv ID:** 2610.37256 | [PDF](https://arxiv.org/pdf/2610.37256v1)

**作者:** Jiadong Xie `[一作]` (Chinese University of Hong Kong), Yingfan Liu `[通讯]` (Xidian University)

**关键词:** `Databases` `Retrieval` `Computational Efficiency` `Graph Neural Network` `Contrastive Learning` `Beam Search` `Image` `Graph` `Point Cloud` `Tabular`

**🎯 论文内容**

提出一种基于亲和图的近似线最近邻搜索框架（Line‑Enhanced Proximity Graph，LPG），通过在图中加入有限的跨区域边实现高效的线查询。

**💡 创新点**

创新点在于：①利用“pair edges”连接分离但与同一直线相关的区域，②采用图平滑的层次划分和商图来高效产生候选区域对，③在保持原始维度和总度数上限的前提下，显著提升搜索吞吐量和召回率。

**🔧 技术方法**

采用技术包括 HNSW 亲和图、图平滑的均值传播层次划分、商图（quotient graph）生成、区域对选择与评分（基于分离度与方向支持）、pair edge 选择与剪枝，以及 beam search 搜索策略。

**📊 数据集**

实验使用 6 个公开视觉数据集：Face‑1100、Multi‑PIE、CelebA、WIDER、SIFT100M（10M/100M点）以及两套基于 GAN 的 128/512 维向量集合。

**📈 对比分析**

与理论 LANN、直接图搜索、维度提升搜索（Quadratic Lifting）以及精确搜索基准进行对比；在 Recall≥0.99 的目标下，LPG 在大多数数据集上实现 2.0–4.2 倍 QPS 加速，Recall 可达 1.0；在 N=512 子集上甚至达 976 倍速度提升。

**⚠️ 局限性**

局限性包括：对极高维或超大规模数据集仍需显著的构建与 pair‑edge 选择成本，参数（如 pair‑edge 数量、叶子容量、平滑轮数）需手动调优；在某些数据集的 top‑1 召回率仍低于理论方法；当前实现不支持在线增删数据更新。

---

## 609. TCM-DiagCommit: A TCM-SD-Derived Benchmark for Evidence-Conditioned Syndrome Selection and Commitment

**arXiv ID:** 2610.37243 | [PDF](https://arxiv.org/pdf/2610.37243v1)

**作者:** Erwei Hou `[一作]` (Beijing University of Posts and Telecommunications), Yue Hu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `Artificial Intelligence` `Benchmark` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Biomedical Data` `Electronic Health Records`

**🎯 论文内容**

构建了TCM‑DiagCommit基准，用以评估中医证候鉴别模型在分阶段证据下的诊断选择与终止决策行为；

**💡 创新点**

首次将诊断任务从静态最终标签转为证据条件下的动态选择与承诺评估，并引入候选难度梯度、分层证据展示与固定标签扰动等多维度测试；

**🔧 技术方法**

使用零-shot DirectLLM等大型语言模型推理、语义邻居构造、候选生成与固定提示框架；

**📊 数据集**

基于公开TCM‑SD/ZY‑BERT数据集，抽取1,834个病例生成23,842条任务行，覆盖148个证候标签；

**📈 对比分析**

与多款LLM（DeepSeek、MiniMax、Mimo、Qwen等）进行零-shot对照，随机候选下top‑1准确率约63%–66%，硬标签或全词汇下仅25%–22%，承诺错误率高达62%–74%，表明单一最终答案评估不足；

**⚠️ 局限性**

受限于TCM‑SD样本分布与标签平衡抽样、固定提示与API变更，且缺乏临床真实证据流与监督学习基准，未来需加入临床阶段标注、检索/重排序基线及更稳健的评估策略。

---

## 610. Grounding LLM Agents in Business Processes: Process-Aware Knowledge Graphs for Reliable Multi-Step Service Invocation

**arXiv ID:** 2610.37242 | [PDF](https://arxiv.org/pdf/2610.37242v1)

**作者:** Saiful Khan `[一作]` (Science and Technology Facilities Council), Afrah Farea `[通讯]` (University of Oxford)

**关键词:** `Artificial Intelligence` `AI Code Assistant` `Recommendation System` `Optimization` `Graph Neural Network` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Agentic AI` `Prompt Engineering` `Cypher graph query` `OData parsing` `BPMN parsing` `Graph` `Text` `Tabular` `Benchmark`

**🎯 论文内容**

构建了一个将OData服务元数据与BPMN流程模型融合的类型化属性图，并开发了混合检索器和基于ReAct的LLM代理，实现多步API调用的可靠性提升。

**💡 创新点**

创新点在于将服务端元数据和业务流程结构统一到图中，利用流程链扩展提升检索准确性，并提供结构化的payload构造上下文，显著减少LLM生成的无效或发明的代码。

**🔧 技术方法**

使用了OData、BPMN解析、FalkorDB图数据库、FAISS密集检索、Cypher图查询、LangGraph ReAct agent、Ollama LLM以及mock HTTP验证层。

**📊 数据集**

使用了OData-Proc benchmark（42个OData服务、14个BPMN流程、288条查询）以及SOCBench-D进行外部泛化测试。

**📈 对比分析**

与仅密集检索、仅图检索和ChunkRAG基线对比，过程链查询Recall@3提升5.7点；在多步代理实验中完成率从10.3%提升至38.7%，无效调用率从0.79降至0.03；在无流程上下文的系统中检索精度无显著差异。

**⚠️ 局限性**

限制包括：依赖人工编写的活动–实体映射，人工成本较高；验证使用相同的元数据导致环路性，未在真实后端测试；对无流程层的API无提升，适用范围受限。

---

## 611. When Is a Relation Worth Trusting? Manipulation Costs, Exposure, and the Trust Trap in Multi-Relational Graph Learning

**arXiv ID:** 2610.37239 | [PDF](https://arxiv.org/pdf/2610.37239v1)

**作者:** Vincess Dongmo `[一作]` (SYNTHI-AI), Astride Melvin Fokam Ninyim `[通讯]` (SYNTHI-AI)

**关键词:** `Machine Learning` `Anomaly Detection` `Federated Learning` `Explainability and Interpretability` `Computational Efficiency` `Adversarial Attack` `Graph Neural Network` `Contrastive Learning` `Dynamic Programming` `Certificate Boundary` `Validation Sampling` `Cost-Aware Planning` `Graph` `Tabular`

**🎯 论文内容**

研究多关系图学习中，基于操纵成本评估哪些关系值得信任，并提出成本感知的门控与选择策略。

**💡 创新点**

创新点包括：① 当部分群体在某关系上被隔离且攻击者无法开启门控时，门控严格优于选择或重加权；② 用曝光下限与单位成本比值挑选最需加固的关系；③ 对线性模型给出精确极值与非线性模型的证明边界。

**🔧 技术方法**

技术手段包括：多关系CSBM理论、线性与非线性图模型（RLR、R-GCN、HGB-Graph等）、动态规划、暴力/贪婪攻击、证书边界、验证抽样与成本感知规划。

**📊 数据集**

使用数据集：YelpChi、Amazon Reviews、FDCompCN、S-FFSD、Tolokers 以及合成的多关系CSBM。

**📈 对比分析**

与无关图基线、对抗训练模型等进行对比，实验显示在多数设定下该方法的鲁棒AUC不劣于图无关模型，并在多关系情形下实现显著提升。

**⚠️ 局限性**

局限性：结果依赖预设的成本情景，未从实际运营中估算成本；非线性模型的证明边界较宽；需要标注验证集，且在真实数据上加固效果有限。

---

## 612. HDVS: semi-supervised semantic segmentation via heterogeneous dual-branch voting supervision

**arXiv ID:** 2610.37236 | [PDF](https://arxiv.org/pdf/2610.37236v1)

**作者:** Yongqi Shan `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Convolutional Neural Network` `Transformer` `Contrastive Learning` `Cross-branch Feature Communication` `Entropy Filtering` `Strong-weak Data Augmentation` `Cross-pseudo Supervision` `Vote-based Pseudo Label Generation` `Image`

**🎯 论文内容**

提出了一种异构双分支投票监督（HDVS）框架，用于半监督语义分割，通过双分支模型产生更可靠的伪标签并实现特征层的跨模交流；

**💡 创新点**

创新点在于：①利用异构网络（CNN+Transformer）提升伪标签多样性；②引入特征通信模块（FCM）在特征层进行跨分支扰动与信息传递；③设计基于置信度的动态投票机制，融合两分支输出生成高质量伪标签；

**🔧 技术方法**

核心技术包括：多尺度特征提取、跨分支特征通信、投票伪标签生成、强弱数据增强、交叉伪监督和熵过滤；

**📊 数据集**

实验数据集为PASCAL VOC 2012（经典与增强划分）和Cityscapes（城市街景），并使用ResNet-50与SegFormer（MiT-B2）作为骨干网络；

**📈 对比分析**

与SOTA方法（如U^2PL、UniMatch、Co-training等）比较，HDVS在VOC 2012与Cityscapes上均取得平均提升约1–2.9 mIoU，显著优于所有已公布方法；

**⚠️ 局限性**

局限性包括：在极少标签（1/16）场景下，异构分支可能产生偏差伪标签，导致分支误差传递；在极端光照或颜色变化区域的分割精度仍有提升空间。

---

## 613. Asynchronous Verifiable and Complete Secret Sharing: From Design Patterns to System-Level Trade-offs

**arXiv ID:** 2610.37234 | [PDF](https://arxiv.org/pdf/2610.37234v1)

**作者:** Joachim Bruneau-Queyreix `[一作]` (Univ. Bordeaux), Laurent Réveillère `[通讯]` (Univ. Bordeaux)

**关键词:** `Cryptography and Security` `Federated Learning` `Safty and Privacy` `Computational Efficiency` `Optimization` `Reliable Broadcast` `Encrypted Message Distribution` `Polynomial Commitment` `Batching/Packing` `Proofs of Completion Based on Corpus` `Tabular` `Time Series` `Benchmark`

**🎯 论文内容**

对现代一拍式异步可验证与完整秘密分享（AVSS/ACSS）协议进行设计空间分析和系统级性能评测。

**💡 创新点**

提出基于论文语料库的设计空间筛选器、统一的 Rust 实现框架，并给出按成本和性能特征挑选协议的实用指南。

**🔧 技术方法**

采用可靠广播、加密消息分发、多项式承诺、批量/打包、基于语料的完成证明等技术，并用 Tokio 进行异步网络实现。

**📊 数据集**

使用 Grid'5000 试验平台（20 台 64 GB 10 Gbps 机器）进行实验，秘密为 32 字节随机值，按不同节点数和批量大小进行调度。

**📈 对比分析**

通过对七个代表性协议在理想与恶意 dealer 两种情景下的延迟、吞吐量、消息计数、带宽与 CPU 进行对比，展示不同设计维度对成本的影响，没有单一协议在所有指标上占优。

**⚠️ 局限性**

实验仅涵盖一拍式协议，未覆盖真实 WAN 网络与更复杂 Byzantine 行为；实现可能未包含协议特定的优化，评测不构成所有协议的全面基准。

---

## 614. Subspace Clifford Networks

**arXiv ID:** 2610.37226 | [PDF](https://arxiv.org/pdf/2610.37226v1)

**作者:** Zhongping Ji `[一作]`, Zhongping Ji `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Convolutional Neural Network` `Transformer` `Contrastive Learning` `Auto Encoder` `Recomputation-based Backpropagation` `CUDA Kernel Optimization` `Image`

**🎯 论文内容**

提出 Subspace Clifford Networks（子空间 Clifford 网络），将原先全局循环滚动的 Clifford 交互替换为在低维子空间内的全双向 wedge 计算，并在子空间与整体通道之间插入 1×1 投影；同时实现了融合的 CUDA 内核和重计算反向传播。

**💡 创新点**

创新点主要有：① 用子空间分组覆盖所有内点的 wedge 组合，减少中间通道扩展最多 75%；② 通过 1×1 投影实现跨组通道混合；③ 针对固定小组尺寸实现单线程 fused CUDA kernel 并采用重计算反向，显著降低 GPU 计算和内存占用；④ 将这一子空间交互嵌入多分辨率层级结构，进一步提升准确率。

**🔧 技术方法**

技术手段包括：Clifford 几何代数中的外部积（wedge）与内积、子空间分组与 SiLU 激活、1×1 线性投影、重计算反向（recomputation‑based）CUDA 内核、MetaFormer 结构、深度学习框架（PyTorch）与多分辨率 Pyramid 网络。

**📊 数据集**

实验使用 CIFAR‑100 数据集（50k 训练 / 10k 测试）。

**📈 对比分析**

在与原始 shift‑based CliffordNet（单尺度和层级）以及同样的卷积层级基线进行对比时，子空间模型在参数量更少（例如 0.756M 对 1.4M）、训练时间更短（≈ 30 min 对 58 min）且 Top‑1 准确率相近或更高；在层级版本中，k=3、k=4 的子空间网络可达 80.1–80.2% Top‑1，深度扩展到 24 层可进一步提升至 81.2%，同时参数量仍低于 shift‑based 对标模型。

**⚠️ 局限性**

局限性：① 交互仅在小组内覆盖，跨组高维关系仍由 1×1 投影近似；② 对组尺寸 k 的选择仍需经验调参；③ 现实现高度依赖 CUDA，缺乏跨平台通用性；④ 在更大规模数据集或更高分辨率任务上的泛化尚未充分验证。

---

## 615. UCVE: Graph Structured Retrieval for Compliance Validation

**arXiv ID:** 2610.37323 | [PDF](https://arxiv.org/pdf/2610.37323v1)

**作者:** Emily Chen `[一作]` (Capital One), Amirfarrokh Iranitalab `[通讯]` (Capital One)

**关键词:** `Computation and Language` `Retrieval` `Federated Learning` `Data-Centric Learning` `Explainability and Interpretability` `Computational Efficiency` `Graph Neural Network` `Transformer` `Reinforcement Learning` `Score-based Model` `Contrastive Learning` `Retrieval-Augmented Generation` `Text` `Graph` `Benchmark`

**🎯 论文内容**

构建了一个统一的合规验证框架 UCVE，利用上下文义务图（CRG）对文档中所有法律义务进行结构化检索与判定。

**💡 创新点**

核心创新是将强化学习用于学习义务间共现的边权重，并通过谱重排序在社区结构上进行低通滤波，从而实现高召回率；这两项技术相互依赖，单独使用效果有限。

**🔧 技术方法**

采用的技术包括上下文义务图（CRG）构建、强化学习边权学习、谱图理论（拉普拉斯矩阵特征向量低通滤波）、k-NN稀疏化、BFS遍历、以及 LLM 验证器。

**📊 数据集**

实验使用了三大领域数据集：OPP‑115（隐私政策合规）、CLAUDETTE（消费者合同公平性）和 ESRS（可持续发展报告）。

**📈 对比分析**

通过匹配 K 的召回评估（每个基线返回同一数量节点）与 BM25、平面余弦、交叉编码器、PPR、Leiden、Heat Kernel 等基线对比，UCVE 在所有三个域平均提升约 4.3% 的召回率，且在最大域 ESRS 上仍显著优于非谱基线。

**⚠️ 局限性**

局限性包括：在高密度义务图上路径效率低、对域专家提供的元数据构建 CRG 的依赖、对超大义务空间训练信号不足导致 RL 边权学习收敛困难、对超参数（k、n、α）的敏感性，以及未覆盖多文档合规验证场景。

---

## 616. Hierarchical Sparse Graph for Multi-dimensional Assignment in Event-based Tracking

**arXiv ID:** 2610.37321 | [PDF](https://arxiv.org/pdf/2610.37321v1)

**作者:** Muxi Zha `[一作]` (National University of Defense Technology), Banglei Guan `[通讯]` (National University of Defense Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Object Tracking` `Graph Neural Network` `Optical Flow` `Contrastive Learning` `Fuzzy Logic` `Hierarchical Clustering` `Graph` `Event-based Data`

**🎯 论文内容**

提出一种无训练、可插拔的事件多维关联方法，将tracklet关联建模为层次稀疏图（HSG）的划分。

**💡 创新点**

创新点包括：将多维事件关联视为不确定的多对多匹配问题；使用模糊逻辑融合贝塞尔曲线位置平滑误差与方向-位移一致性两种运动指标，构造关联概率；设计层次稀疏量化与自适应连接相似度阈值的图划分算法；实现全程无监督、无硬阈值的高精度追踪。

**🔧 技术方法**

采用的技术包括：模糊逻辑与层次聚类、贝塞尔曲线拟合、方向-位移一致性度量、稀疏化筛选、连接相似度自适应阈值、图划分搜索-合并算法、无监督训练等。

**📊 数据集**

使用的公开数据集为 eTram（交通事件跟踪）和 Ev-UAV（极小目标事件跟踪）两大事件数据集。

**📈 对比分析**

与多种事件基线（KF+HA、NS、IoU）以及多种图像基线（Permatrack、UAVMOT、OC‑SORT、LTTrack、TOPIC、LPT）在 MT、ML、MOTA、IDF1/Track IoU 等指标上进行对比。实验表明，方法在 eTram 上 MOTA 98.5/95.7（IDF1）并显著优于所有事件基线；在 Ev‑UAV 上 Track IoU/Acc/IDF1 分别提升至 81.3/52.7/74.0，超过 OC‑SORT 的 2 倍以上。

**⚠️ 局限性**

主要限制在于基于投票式图划分，缺乏对轨迹连续性与多帧动态约束的显式建模，导致在轨迹稀疏或高密度场景下易出现误关联，需要后续在增量或滑动窗口机制中进一步改进。

---

## 617. Self-Referenced Diagnosis of SPAD Dark-Count-Rate Switching for Quantum Communication Receivers

**arXiv ID:** 2610.37312 | [PDF](https://arxiv.org/pdf/2610.37312v1)

**作者:** Yu Luo `[一作]` (Mississippi State University), Junming Diao `[通讯]` (Mississippi State University)

**关键词:** `Emerging Technologies` `Anomaly Detection` `Physics Related` `Time Series Analysis` `Time Series`

**🎯 论文内容**

提出了一种自参考诊断框架，用于区分量子通信接收器中单光子雪崩二极管（SPAD）的暗计数率（DCR）切换的内在随机电报信号（RTS）与非内在伪影。

**💡 创新点**

创新点在于通过低到高的后脉冲产率比R结合DCR状态的可重复性、跨通道行为和事件时序等补充检查，提供了一种实用的自参考诊断框架。

**🔧 技术方法**

使用了自参考诊断方法，主要依赖于时间标记的暗计数数据，结合后脉冲相关性进行分析。

**📊 数据集**

使用了四个商业SPAD模块（PerkinElmer SPCM-AQR系列），在黑暗环境中监测超过920小时。

**📈 对比分析**

通过与温度相关的阈值穿越、光泄漏、缆线干扰和电气注入等控制实验进行比较，发现不同伪影产生了不同的相关性和时序特征，支持了诊断框架的有效性。

**⚠️ 局限性**

限制在于R值接近1时，可能会导致对内在RTS的误判，因为光泄漏和电气注入也可能产生类似的R值，需结合其他特征进行进一步确认。

---

## 618. MvT: Event-based Multi-view Projection for Multiple Object Tracking

**arXiv ID:** 2610.37309 | [PDF](https://arxiv.org/pdf/2610.37309v1)

**作者:** Muxi Zha `[一作]` (National University of Defense Technology), Laurent Kneip `[通讯]` (ShanghaiTech University)

**关键词:** `Computer Vision and Pattern Recognition` `Object Tracking` `Optical Flow` `Contrastive Learning` `Auto Encoder` `Event-based Data` `Video` `Benchmark`

**🎯 论文内容**

提出了一种多视图投影与融合（MvT）框架，用于事件相机的多目标跟踪；

**💡 创新点**

创新点包括：①将事件流投影为三种二维正交视图（XOY、YOT、TOX），保留时空信息；②引入时间重排和高斯金字塔来缓解事件离散化；③在多视图融合阶段通过轨迹分离与多视图关联实现轨迹重建；④提出三条关联准则，兼顾短期线性与跨视图交叉的情况；

**🔧 技术方法**

主要技术：事件投影、时间重排、双向高斯金字塔、图像/分割方法的无监督融合、基于几何方向与交点的轨迹关联；

**📊 数据集**

使用了四个数据集：ECD Shapes（标准大目标），BEES（蜜蜂小目标），Ev-UAV（无人机小目标），以及作者自行收集的高分辨率小目标数据集SOD（真实与模拟）；

**📈 对比分析**

与伪帧方法（ETD、RMRNet）、事件逐帧方法（AEBT、AEMOT、E-MS）以及基准Prophesee进行对比；在大多数指标（MOTA、IDF1、Track IoU/Acc、FA）上均优于现有方法，尤其在小目标、复杂运动场景中显著提升；

**⚠️ 局限性**

局限性包括：关联仅基于运动连续性，长时间遮挡或大方向变化时性能下降；未利用事件极性或强度重建的外观信息；高斯核采用等距形状，对不同视图的适应性有限；参数如Δθ、金字塔层数需人工设定，缺乏自适应机制。

---

## 619. Understanding Human-Robot Collaboration through Joint Action

**arXiv ID:** 2610.37298 | [PDF](https://arxiv.org/pdf/2610.37298v1)

**作者:** Chengwen Zhang `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `Human-Computer Interaction` `Review/Survey Paper` `Robotic Intelligence` `Text` `Review/Survey Paper`

**🎯 论文内容**

对 2021‑2026 年 83 篇实证人机协作(HRC)论文进行系统综述，并基于联合行动(Joint Action)理论构建了包含 9 个构造的框架，分析了各构造在文献中的分布、共现关系，揭示了研究重点与空白。

**💡 创新点**

创新点在于：① 将联合行动理论的三大功能层（表示、预测、整合）细化为 9 个可操作构造；② 采用理论驱动的编码与开放式子码方法，将构造的实现方式系统化；③ 通过频率与 lift 计算的共现分析，量化构造之间的相互依赖与缺口，为未来研究指明方向。

**🔧 技术方法**

主要技术包括：PRISMA 2020 系统检索、理论驱动编码（顶层构造 → 子码）、子码共现网络构建、lift 与期望计数计算、异构子码的条件依赖分析。

**📊 数据集**

数据集：83 篇符合纳入标准的实证 HRC 论文（2021‑2026 年），来源于 RO‑MAN、HRI、CHI、IEEE THRI、Nature、Science Robotics 等主流会议与期刊。

**📈 对比分析**

比较方法：对各构造及子码的出现频率进行统计，对子码共现进行 lift 及期望计数筛选，构建共现网络与依赖矩阵。结果显示：预测（JA6、JA7）与整合（JA8）构造覆盖率最高（>70%），而表示层（特别是共享意图 JA2）覆盖率最低；共现网络揭示任务划分、监控与预测等子码是多数构造的前置条件，表明“执行架构”是 HRC 研究的核心。

**⚠️ 局限性**

limitation：① 研究范围受限于选定数据库与关键词，可能漏检使用不同术语或跨学科领域的相关工作；② 对共享意图、合作承诺等隐式构造的编码高度依赖研究者主观判断，缺乏统一评估标准；③ 综述聚焦于技术实现与机制描述，未对人因与长期协作效果进行量化验证。

---

## 620. TRUCO: A Scalable Lock-Free Algorithm for Parallel Contour Extraction

**arXiv ID:** 2610.37297 | [PDF](https://arxiv.org/pdf/2610.37297v1)

**作者:** Rafael Munoz-Salinas `[一作]` (Cordoba University), Manuel J. Marin-Jimenez `[通讯]` (Cordoba University)

**关键词:** `Computer Vision and Pattern Recognition` `Computational Efficiency` `Lock-free synchronization` `SIMD vectorization` `Speculative tracing` `Image`

**🎯 论文内容**

提出TRUCO，一种锁自由的多核并行二值图像轮廓提取算法；

**💡 创新点**

通过开始点所有权规则与上升终止策略实现无锁并行，同时采用8位状态简化和仅提取几何边界，避免构建层次结构；

**🔧 技术方法**

并行行切分、锁自由同步、speculative tracing、8‑bit状态、SIMD向量化、起始点拥有规则、向上终止逻辑；

**📊 数据集**

使用合成16MP圆形密集图像、Liu4K 16MP真实图像，以及垂直条纹和螺旋极端测试集；

**📈 对比分析**

与OpenCV的Suzuki‑Abe、RLE和Garcia基线比较，单线程TRUCO约2×快，20线程可达约20×加速（16MP 14×、1MP 20×），在Liu4K 16MP约10.5×；极端案例下约2×；

**⚠️ 局限性**

在垂直条纹和单螺旋等极端几何场景下并行速度受限，线程争用/缓存一致性导致abort率升高；不构建层次结构限制了对层次信息的提供；

---

## 621. A Linear Kernel for Planar Semitotal Domination

**arXiv ID:** 2610.37295 | [PDF](https://arxiv.org/pdf/2610.37295v1)

**作者:** Paloma T. de Lima `[一作]` (Norwegian School of Economics), Lukas Retschmeier `[通讯]` (University of Copenhagen)

**关键词:** `Data Structures and Algorithms` `Optimization` `Region decomposition` `Reduction rules` `Combinatorial proof`

**🎯 论文内容**

论文研究了平面图上的半总支配（Semitotal Dominating）问题，并给出了参数为解大小的线性核（最多 358k 个顶点）。同时证明该问题在分割图、二分图、t‑claw‑free 图上为 W[2]-hard，在圆图上为 W[1]-hard。

**💡 创新点**

创新点在于：①首次为半总支配问题构造线性核；②设计了三条新的归约规则，能够保持 2‑距离见证约束；③通过区域分解与归约规则结合，严格上界为 358k，提升了先前总支配线性核的技术；④在特殊图类中给出了新的硬度结果，完善了该问题的参数化复杂度图谱。

**🔧 技术方法**

主要技术包括：图的区域分解（region decomposition）框架、邻域分类型（N1、N2、N3）的细分、归约规则（Rule 1、2、3）以及组合论式的顶点计数与归约后图大小证明。

**📊 数据集**

论文为理论性质的研究，不涉及实验数据集；所有结果均通过严格的组合证明与归约论证得到。

**📈 对比分析**

与之前针对总支配的线性核相比，本文得到的 358k 线性核在同类问题中是首个完成的；在特殊图类上，通过归约与硬度证明，表明问题在这些图类中没有多项式核（除非 P=NP）。

**⚠️ 局限性**

局限性包括：只针对平面图给出线性核，未覆盖非平面图；在 claw‑free（3‑claw‑free）图上仍未知是否 FPT；归约规则的常数系数较大，实际实现可能复杂；未给出实验验证。

---

## 622. LESSRec: Level-Specific History Modeling for Semantic ID-Based Recommendation

**arXiv ID:** 2610.37286 | [PDF](https://arxiv.org/pdf/2610.37286v1)

**作者:** Shangxin Yang `[一作]` (Chongqing University), Min Gao `[通讯]` (Chongqing University)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Contrastive Learning` `Attention Mechanism` `Sequential`

**🎯 论文内容**

提出了一种新的推荐系统模型LESSRec，专注于语义ID基础的推荐，通过分别编码历史令牌序列并在预测时限制对目标令牌语义级别的注意力来提高推荐效果。

**💡 创新点**

首次识别出同级历史在语义ID基础生成推荐中的预测优势，并提出LESSRec和LESSRec+，后者通过将编码器分为特定语义级别的组来减少计算量。

**🔧 技术方法**

使用了级别隔离自注意力（LIS）和级别对齐交叉注意力（LAC）技术，以及项级语义序列（ILS）来增强模型的推荐能力。

**📊 数据集**

在三个公共数据集（Games、Software和Beauty）上进行了实验，使用了三种不同的标记器进行比较。

**📈 对比分析**

与传统的序列推荐器和语义ID基础生成推荐器相比，LESSRec和LESSRec+在所有数据集和指标上均表现出显著的性能提升，LESSRec+在计算效率上也表现更佳。

**⚠️ 局限性**

模型的局限性在于仍然依赖于固定的语义ID，可能无法适应动态变化的用户偏好和新兴的推荐场景。

---

## 623. Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa

**arXiv ID:** 2610.37245 | [PDF](https://arxiv.org/pdf/2610.37245v1)

**作者:** Valdini Douglace Lemofouet `[一作]` (African Institute for Mathematical Sciences), Shamsuddeen Hassan Muhammad `[通讯]` (Bayero University)

**关键词:** `Computation and Language` `Safty and Privacy` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning from Human Feedback` `Prompt Engineering` `Text`

**🎯 论文内容**

提出了首个 Hausa 安全偏好数据集 HausaSafeLM，并在 Qwen2.5‑7B‑Instruct 上对 SFT 与 SFT+DPO 的安全对齐效果进行系统实验。

**💡 创新点**

创新点在于：①首次构建面向 Hausa 的安全偏好数据集；②揭示低资源语言中拒绝率指标的误导性；③系统阐明安全、流畅性与帮助性之间的权衡表面。

**🔧 技术方法**

采用 LoRA 微调、SFT、DPO、以及 Gemini LLM 作为自动评判器，对模型在 Hausa 上的拒绝与响应进行评估。

**📊 数据集**

数据集包括从 PKU‑SafeRLHF 翻译并本地化的 10,000 条样本（3,000 gold + 7,000 silver），并使用 500 条有害 Hausa 提示与 200 条 benign Aya 提示进行验证。

**📈 对比分析**

通过对 Base、SFT‑G、SFT‑GS、DPO‑G、DPO‑GS 五个配置在 Refusal Rate、False Refusal Rate、Helpfulness Rate、SafeQ、AyaQ 等指标上的比较，发现：SFT 提升拒绝流畅性和覆盖率；DPO 提升拒绝精确度但削弱流畅性；银数据对 SFT 有显著正面作用，对 DPO 影响甚微，且无单一配置同时达到安全、帮助与流畅的最佳平衡。

**⚠️ 局限性**

局限性包括：仅使用单一 7B 模型；评估由单一 Gemini 评判器完成，缺乏多名原生 Hausa 注释者；数据集为翻译而非本土生成，可能遗漏当地特有危害；样本量有限，结果对细微差异需谨慎解读。

---

## 624. TAO: Tool-Aware Joint Scheduling for LLM-Based Agentic Serving

**arXiv ID:** 2610.37230 | [PDF](https://arxiv.org/pdf/2610.37230v1)

**作者:** Shuhuai Li `[一作]` (Shanghai Jiao Tong University), Jianghao Lin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Optimization` `Computational Efficiency` `Federated Learning` `Transformer` `Large Language Model` `Agentic AI` `Retrieval-Augmented Generation` `Prompt Engineering` `Mixture of Experts` `Text`

**🎯 论文内容**

提出了一种面向大模型代理服务的工具感知联合调度框架，融合工具返回时机、缓存信用衰减与令牌占用等评估指标，利用动态规划在 GPU 节点容量约束下联合选取与放置等待中的会话。

**💡 创新点**

创新点在于：①首次将工具调用延迟作为调度信号，引入工具返回准备度；②构造三维评估三元组（缓存信用、工具准备度、令牌量）并采用字典序最大化；③使用离散化多背包/通用分配的动态规划实现全局最优调度，突破传统逐会话启发式方法。

**🔧 技术方法**

核心技术包括：时间感知缓存信用估计（指数衰减）、工具返回准备度预测器（历史时长与实时提示融合）、多维评估三元组与字典序比较、容量状态动态规划（多背包/通用分配模型），以及异步重验证机制。

**📊 数据集**

在 DeepSeek‑V4 Flash 与 Qwen3‑32B 两大大模型平台上，分别构造 12 种模型-工作负载-并发度组合进行实验。

**📈 对比分析**

与现有最优基线（包括 SGLang、llm‑d 等）进行对比，实验表明该框架可将总体延迟提升约 26.8%，首次令牌时间降低约 36.3%，解码吞吐量提升约 29.5%。

**⚠️ 局限性**

局限性：①实验范围局限于特定 GPU 设备与工具环境；②动态规划对会话池规模的时间与空间复杂度敏感，极大规模部署时可能受限；③工具返回准备度预测依赖历史数据，异常或新工具时预测准确性可能下降。

---

## 625. Decision Models as Judges for Hallucination Detection

**arXiv ID:** 2610.37229 | [PDF](https://arxiv.org/pdf/2610.37229v1)

**作者:** Haozheng Liu `[一作]` (New York University), Lingyao Li `[通讯]` (University of Arizona)

**关键词:** `Computation and Language` `Anomaly Detection` `Computational Efficiency` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Mixture of Experts` `Text` `Benchmark`

**🎯 论文内容**

评估了五种决策模型（非生成）作为幻觉检测判定器，并与生成LLM判定器gpt‑6‑sol进行对比，探究其置信度可靠性以及通过置信级联实现低成本提升。

**💡 创新点**

在幻觉检测任务中首次系统比较多种决策模型在不同幻觉类型上的表现，研究其置信度校准与误判分布，并提出置信级联的低成本集成方案，提供成本‑准确性折衷。

**🔧 技术方法**

采用决策模型（jev、d1、clef、pplx、gpt‑6‑luna）与生成LLM gpt‑6‑sol，使用ECE、conf‑AUROC、overconfidence gap等指标评估置信度；实现投票、仲裁和置信级联三种集成策略，并通过API调用估算成本与延迟。

**📊 数据集**

使用六个公开幻觉检测数据集（FAVABench、HADTest、RAGTruth、ContraDoc、ROSCOE、IF‑RewardBench），共9,852条样本，按五种幻觉类型（二元标签）进行划分。

**📈 对比分析**

通过平衡准确率（BAcc）、宏F1、AUROC等指标比较，发现决策模型在事实性与上下文相关错误上可与gpt‑6‑sol匹配或接近，且成本仅为其1–10%；置信级联在事实性、事实制造和上下文不一致三类上，在成本仅为1–3%时可超过gpt‑6‑sol；在逻辑与指令不一致上仍落后。

**⚠️ 局限性**

仅覆盖五种幻觉类型且来源不均衡，数据集与错误类型混淆；置信度在不同模型间不完全可比；级联策略未在实时部署上验证；prompt设计与置信度获取方式未系统探究；误判模式与数据集差异导致结果解释受限。

---

## 626. Vision-Language-Action or World-Action? Just Action Models

**arXiv ID:** 2610.37219 | [PDF](https://arxiv.org/pdf/2610.37219v1)

**作者:** Yonghao Yu `[一作]` (University of Tokyo), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**关键词:** `Computer Vision and Pattern Recognition` `Robotic Intelligence` `Reinforcement Learning` `Transformer` `Vision-Language-Action Model` `Contrastive Learning` `Vision Language Model` `Cosine Loss` `Freezing Network` `Image` `Text` `Multimodality`

**🎯 论文内容**

在机器人控制中，利用预训练视觉模型与行动专家训练只需行动的策略，替代传统视觉生成模型。

**💡 创新点**

创新点在于将未来观测的监督转化为冻结的特征目标，仅在训练阶段使用，推理时完全去除视觉生成模块。

**🔧 技术方法**

采用预训练视觉/视觉语言模型、行动专家、cosine 预见损失、冻结目标网络及目标缓存等技术。

**📊 数据集**

使用 LIBERO、LIBERO-Plus 与 RoboTwin 2.0 三大机器人任务数据集进行实验。

**📈 对比分析**

与 FastWAM、ImageWAM 等基线进行固定架构比较，LIBERO 上提升 1‑2%，LIBERO-Plus 上提升 9‑15%，RoboTwin 近乎等价，并显著降低训练 FLOPs。

**⚠️ 局限性**

局限包括对预训练视觉表示的依赖以及对目标位置选择的敏感性，且未在真实物理环境中验证部署效果。

---

## 627. Displacement Moments and Path-Terminal Separation in Brownian Interpolant Sampling

**arXiv ID:** 2610.37413 | [PDF](https://arxiv.org/pdf/2610.37413v1)

**作者:** Zhong Han `[一作]` (National University of Singapore), Zhong Han `[通讯]` (National University of Singapore)

**关键词:** `Machine Learning` `Optimization` `Federated Learning` `Computational Efficiency` `Data-Centric Learning` `Stochastic Differential Equation` `KL Divergence Analysis` `Path Decomposition` `Information Theory Methods` `Euler Sampling` `Schrödinger Bridge Theory` `Time Series` `Physics Related`

**🎯 论文内容**

研究了 Brownian 随机插值采样中位移矩与路径终端分离的性质，并给出了基于 Euler 采样的 KL 散度误差上界及其与位移矩的关系；同时提出了基于 iid 端点对的当前状态学习器并证明其期望输出 KL 一致性。

**💡 创新点**

通过将位移矩的阶数与 KL 误差相结合，首次给出四阶位移矩可实现 O(N⁻¹) 的误差上界；解析路径与终端信息的损失，构建了利用终端对的学习器，并对 Markov 投影下的信息丢失进行了路径分解。

**🔧 技术方法**

利用随机微积分、KL 散度分析、Euler 采样、Schrödinger 桥理论、条件扩散身份、路径分解及信息论方法。

**📊 数据集**

论文主要使用模拟的 Brownian 路径和 iid 端点对进行数值验证，并未提供具体真实数据集。

**📈 对比分析**

与传统的 Markov Euler 采样方法对比，数值实验表明终端 KL 可达到 O(1/N²)，路径 KL 维持 O(1/log N) 等；对比展示了不同位移矩阶数下的误差阶数差异。

**⚠️ 局限性**

仍未给出一般二阶矩耦合下的统一终端速率；缺乏对迭代方法优势的理论证明；实验主要基于模拟，缺少真实应用验证。

---

## 628. Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation

**arXiv ID:** 2610.37409 | [PDF](https://arxiv.org/pdf/2610.37409v1)

**作者:** Zhenye Yang `[一作]` (Beijing University of Posts and Telecommunications), Senzhang Wang `[通讯]` (Central South University)

**关键词:** `Information Retrieval` `Recommendation System` `Graph Neural Network` `Contrastive Learning` `Self‑Calibrated Gate` `Residual Injection` `SVD` `GNN` `Multimodality` `Graph`

**🎯 论文内容**

通过离线几何变换与在线不对称图传播结合，实现多模态推荐中补充性关系的建模

**💡 创新点**

将补充性建模拆分为离线Manifold增强和在线异构双流传播，并引入Soft Conceptor、gap约束注入和自校准门控，实现语义与行为拓扑的解耦与融合

**🔧 技术方法**

软概念子空间投影、局部SVD、残差注入、异构双流GNN、Self‑Calibrated Gate、BPR等联合使用

**📊 数据集**

亚马逊四大公开数据集（Baby、Sports、Clothing、Electronics）

**📈 对比分析**

与MF‑BPR、LightGCN、VBPR、SLMRec、BM3、MGCN、LGMRec、MENTOR、GUME等基线对比，MAGE在Recall@20和NDCG@20上均超过最强基线约2.8%–6.5%，在硬补充子集上提升约10%

**⚠️ 局限性**

共依赖不完善的共购边，几何约束无法保证残差语义；对极端噪声、稀疏或时间漂移的鲁棒性有限，且方法复杂度较高

---

## 629. Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling

**arXiv ID:** 2610.37393 | [PDF](https://arxiv.org/pdf/2610.37393v1)

**作者:** Sinjae Kang `[一作]` (KAIST), Kimin Lee `[通讯]` (KAIST)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Flow-based Model` `Reinforcement Learning` `Supervised Fine-Tuning` `Tabular` `Image` `Video` `Sequential`

**🎯 论文内容**

提出质量可控策略QCP，允许在推理时通过离散质量标签控制机器人执行速度、平滑度等属性。

**💡 创新点**

创新点在于将质量作为显式条件输入并结合迭代后见重标记循环，能在无需人工标注的情况下扩展可控质量范围。

**🔧 技术方法**

采用流匹配模型与量化质量标签训练的条件行为克隆，配合多轮在线rollout收集和重标记。

**📊 数据集**

使用Robomimic、RoboEval模拟数据集以及真实Franka Research 3机器人收集的演示数据。

**📈 对比分析**

与BC、质量过滤BC、SORL、SCIQL、DSRL、EXPO等基线比较，QCP在任务成功率和质量指标上均超越基线，迭代后可控质量显著提升。

**⚠️ 局限性**

局限性包括对质量评分函数的依赖、离散标签导致的细粒度控制受限，以及在极大数据量或复杂多属性场景下可能需要更多标注和计算资源。

---

## 630. Same Verdict, Different Readers: Personalizing Fact-Checking Explanations from Relative Preferences

**arXiv ID:** 2610.37380 | [PDF](https://arxiv.org/pdf/2610.37380v1)

**作者:** Haokun Wen `[一作]` (National University of Singapore), Wynne Hsu `[通讯]` (National University of Singapore)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Recommendation System` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Direct Preference Optimization` `Text`

**🎯 论文内容**

提出了Faithful Personalized Fact-Checking Explanation Rewriting任务，即在保持事实准确性的前提下，针对不同读者的偏好重写已有的事实核查解释。

**💡 创新点**

创新点在于：①通过LLM模拟读者的相对偏好构造个性化监督；②利用跨人格互惠对生成互相排斥的偏好对；③引入两阶段逐步学习（预热与探索优化），实现从预定义风格到多样化个性化解释的迁移。

**🔧 技术方法**

主要技术包括：LLM候选生成与审核（Fidelity & Style Auditors）、位置平衡偏好选择、Borda排序与正向/负向样本构造、DPO（Direct Preference Optimization）+辅助似然损失、LoRA微调。

**📊 数据集**

使用三大事实核查数据集（MOCHEG、TRUE、VeriTaS），共计2080条断言，生成10个随机人物档案后得到20800条 claim–persona 组合。

**📈 对比分析**

与八个零样本LLM（包括4B、27B等开放权重模型及多款商用模型）对比，PERPL在模拟读者偏好方面获得最高的平均DWR（≈81.75%）且Fidelity Pass Rate>90%，在所有基线中实现最优的事实准确度+个性化对齐组合指标。

**⚠️ 局限性**

局限性包括：①偏好监督完全依赖LLM模拟，可能无法完全反映真实人类偏好；②受预定义四种风格样本质量限制，初始学习阶段对多样性依赖有限；③模型对极端或罕见人格档案的泛化尚待验证。

---

## 631. Uncovering Why LLMs Fail: From Internal Mechanisms to Adversarial Reasoning Trajectories

**arXiv ID:** 2610.37369 | [PDF](https://arxiv.org/pdf/2610.37369v1)

**作者:** Zirui Ding `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Yunfeng Lu `[通讯]` (Beihang University)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Adversarial Attack` `Transformer` `Large Language Model` `Prompt Engineering` `Attention–MLP 协调度` `TransformerLens/patching` `谱分析` `Text`

**🎯 论文内容**

开发并评估一种基于七个功能状态的内部诊断框架 CauTrace，用于追踪 LLM 在推理过程中的失败机制。

**💡 创新点**

①首次将 LLM 推理拆分为七个功能状态并同步追踪；②提出三层诊断（表示传播、时序状态协同、全程轨迹）；③在正常与对抗性失败中发现共同的内部路径。

**🔧 技术方法**

采用白盒激活探针、TransformerLens/patching、谱分析、Attention–MLP 协调度、时间均值等技术，并构建复合得分 R_str、R_proc 等指标。

**📊 数据集**

使用 Qwen2.5 与 LLaMA 家族模型的公开评测数据集（TruthfulQA、HaluEval、HarmBench、JailbreakBench 等）以及 500 条 reasoning‑interference 与 300 条安全性对照对。

**📈 对比分析**

通过比较成功与失败样本的表示一致性、Verification‑Commitment 比例、轨迹分布，发现失败样本在 A̅_sup 下降、R_vc 降低；对抗测试中对应状态偏移显著，验证了框架对内部变化的敏感性；相较基准，CauTrace 能揭示多条失败路径并在不同模型上保持一致性。

**⚠️ 局限性**

①不同模型架构导致表示层指标（如 ρ^V、Γ_ac,V）不一致，需针对性校准；②框架依赖白盒访问与预训练参数，无法直接用于黑盒部署；③阈值和权重设置对结果敏感，需在每个任务上重新调参；④未深入定位具体参数或子网络导致失败的因果机制。

---

## 632. EmbeddingGemma 2: Unified and Efficient Multimodal Representations

**arXiv ID:** 2610.37364 | [PDF](https://arxiv.org/pdf/2610.37364v1)

**作者:** Henrique Schechter Vera `[一作]` (Google), Mojtaba Seyedhosseini `[通讯]` (Google)

**关键词:** `Machine Learning` `Retrieval` `Representation Learning` `Computational Efficiency` `Knowledge Distillation` `Transformer` `Contrastive Learning` `Large Language Model` `Vision Language Model` `Auto Encoder` `Knowledge Distillation` `Text` `Image` `Video` `Audio` `Multimodality`

**🎯 论文内容**

提出了EmbeddingGemma 2，一款拥有744M参数的多模态嵌入模型，能够将文本、图像、视频、音频及其混合输入映射到统一向量空间。

**💡 创新点**

创新点包括：1）使用简化的层级嵌入（仅靠投影而非额外查找表）提升多模态表示；2）分阶段训练策略（文本预微调 → 多模态预微调 → 对比学习微调），显著优化嵌入空间；3）引入Matryoshka Representation Learning将全尺寸嵌入分解为多层子维度；4）结合自监督对比、正交正则和教师蒸馏三种损失，实现高质量且可压缩的嵌入；5）支持按需量化（int4 per-block/per-channel）和截断，降低内存占用。

**🔧 技术方法**

技术栈包括：Bidirectional Transformer backbone（24层，512维），Vision Encoder（ViT 16×16，768维）与Audio Encoder（卷积+Conformer，1536维）共同投影到共享空间；Per‑Layer Embeddings（RMSNorm+gate机制）；多模态交织输入；三阶段训练损失（Contrastive NCE、Spread‑out Regularizer、Distillation）与Matryoshka分维；量化感知训练与后期裁剪；参数模块化，支持文本/视觉/音频子模型。

**📊 数据集**

使用约2.9T标记的多模态数据集：文本问答、句子相似、代码检索、网络搜索；图像/视频/文档混合任务；音频/语音、音乐、环境声音等；此外，还用MMEB‑v2、MTEB、MIEB、MAEB、MSEB等公开基准进行评测。

**📈 对比分析**

与公开榜单上小于1B参数的文本/多模态/音频基准模型（如Gemini Embedding 2、CLIP、SigLIP、CLAP等）进行对比。EmbeddingGemma 2在MTEB多语言文本、代码检索、MMEB‑v2视觉/视频/文档任务、MIEB、MAEB、MSEB上均创下新Pareto前沿，特别在多模态检索和音频检索上优于同类更大模型；在多模态模型中以更少参数获得更高综合得分；量化版本仍保持与浮点模型相近的性能。

**⚠️ 局限性**

局限性：1）对极长序列（超出8192 token）仍需扩展；2）视频和音频输入依赖固定帧/声道采样，可能在细粒度时序任务上受限；3）尽管量化已优化，但在极低内存场景下仍需进一步压缩；4）模型在零射击/非文本声音聚类等细粒度任务上表现不及专用音频模型；5）对复杂多模态推理（如长文本+视频+音频）仍需验证其鲁棒性。

---

## 633. Online Interval Selection with Binary Predictions: Optimal Consistency and a Continuous Robustness Tradeoff

**arXiv ID:** 2610.37361 | [PDF](https://arxiv.org/pdf/2610.37361v1)

**作者:** Zhong Han `[一作]` (National University of Singapore), Zhong Han `[通讯]` (National University of Singapore)

**关键词:** `Data Structures and Algorithms` `Optimization` `Reinforcement Learning from Human Feedback`

**🎯 论文内容**

研究在线区间选择问题，在每个请求到达时已知其是否属于最终最优解的二元预测，并在允许撤销已接受请求但不可恢复已拒绝请求的情境下，探讨在保证有限鲁棒性的前提下实现最优一致性的策略。

**💡 创新点**

1) 在严格鲁棒性约束下证明一致性最佳上限为 d = (3+√5)/2；2) 提出 Shaped‑BANK 这一确定性算法，在鲁棒预算 φ⁵ 处实现一致性 d，并给出连续鲁棒‑一致性折衷的精确阈值；3) 设计账本（ledger）机制记录已失误预测的长度，支持更灵活的替换决策。

**🔧 技术方法**

使用离散化区间构造、账本成本门限、冲突代价函数（q_*）、最优替换规则、几何上限分析、随机化与期望一致性分析以及极端输入构造来证明上下界。

**📊 数据集**

本文主要使用理论构造的极端输入实例进行实验验证，并未使用公开的真实数据集；所有结果均基于精确的数学证明和极限构造。

**📈 对比分析**

与已有的 LR、RP 等基线算法在一致性 d 与鲁棒预算 φ⁵ 的交点处进行比较，证明 Shaped‑BANK 在该点实现最优一致性；在整个鲁棒预算区间绘制了最优保证曲线；随机化算法在期望一致性与有限鲁棒性方面给出了下限，表明随机化无法突破一致性 d 的上限。

**⚠️ 局限性**

1) 对期望一致性与鲁棒性共同最优的研究仍未完成；2) 仅在极端输入构造上证明下界，实际输入分布下的性能尚不清楚；3) 目前的最优策略仍需拒绝短负请求，无法实现无过滤、连续鲁棒性与精度召回平衡的完整方案；4) 未考虑取消成本、可靠性约束等实际运营因素。

---

## 634. Normative Deviation Modeling for EEG-based Depression Detection

**arXiv ID:** 2610.37338 | [PDF](https://arxiv.org/pdf/2610.37338v1)

**作者:** Wan Jiang `[一作]` (Hefei University Of Technology), Richang Hong `[通讯]` (Hefei University Of Technology)

**关键词:** `Machine Learning` `Anomaly Detection` `Classification` `Transformer` `Auto Encoder` `Diffusion model` `Contrastive Learning` `Generative Adversarial Network` `Biomedical Data` `Time Series` `Electrocardiogram`

**🎯 论文内容**

本文提出基于健康参考的EEG异常检测框架，设计了两种方法S-NDD和F-NDD，用于识别抑郁症。

**💡 创新点**

创新点在于同时利用结构化和分布式健康参考模型，计算多层次偏差（标量级和特征级），并将其融合进深度EEG特征，从而显著提升跨源泛化能力。

**🔧 技术方法**

采用VQ-VAE和扩散模型训练健康参考，计算重建误差与去噪误差；S-NDD通过标量偏差融合，F-NDD通过注意力聚合和辅助监督处理特征偏差；还引入目标健康校准方法。

**📊 数据集**

实验使用公开的Mumtaz2016和MODMA EEG抑郁症数据集，覆盖19和128通道两种采集方式。

**📈 对比分析**

与判别学习、生成增广、域适应等多种基线比较，S-NDD和F-NDD在窗口和受试者层面均获得最高准确率、召回率、F1和AUC，尤其在跨源迁移时优于传统方法。

**⚠️ 局限性**

局限性包括：仍需通过电极几何对齐来缓解采集差异，目标健康样本校准不可或缺；对超参数和模型选择较为敏感。

---

## 635. WCC: A Learning-Based Framework for Deciding When to Apply Cooperative Coevolution in Large-Scale Optimization

**arXiv ID:** 2610.37329 | [PDF](https://arxiv.org/pdf/2610.37329v1)

**作者:** Wenjie Qiu `[一作]`, Yue-Jiao Gong `[通讯]`

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning` `Reinforcement Learning` `PPO` `Feature Selection` `Auto Encoder` `Tabular` `Benchmark` `Benchmark`

**🎯 论文内容**

提出了一种基于强化学习的混合优化框架WCC-MMES，能够在全空间优化（NDA）和子空间协同进化（CC）之间自适应切换并动态分配评估预算，从而提升大型尺度可分解优化（LSGO）的求解性能。

**💡 创新点**

创新点包括：①将优化模式选择和资源分配拆分为两步自回归决策；②构建16维状态特征，融合问题结构、收敛动态、资源消耗与历史模式信息；③引入FeatureSE对异构特征进行动态加权；④采用对数比率奖励与PPO算法训练，显著提升收敛稳定性。

**🔧 技术方法**

主要技术：PPO强化学习框架、FeatureSE特征选择网络、两层自回归策略网络、CMA-ES（子空间优化）与MMES（全空间优化）、DG2与RDDSM子空间划分方法。

**📊 数据集**

使用CEC2013 LSGO benchmark（15个问题，包含可分离、部分可分离、重叠与非可分离类别）以及AOB（1000维、不同重叠程度）等数据集进行评估。

**📈 对比分析**

与多种CC、NDA及混合算法（如DG2-CMAES、MMES、sep-CMA-ES、FCMAES、MetaES、HCC-ES、CC-SHADE-SL）进行比较，WCC-MMES在3×10^6评估预算下平均性能显著优于对照，统计显著性检验（Wilcoxon rank‑sum）显示多数问题上取得显著提升。

**⚠️ 局限性**

局限性：①训练需预先计算分解，增加前期成本；②对超参数和奖励设计敏感；③在极高维或极其复杂交互结构的特殊问题上可能不如单一最优策略；④训练过程计算量大，对资源要求高。

---

## 636. Decomposing Directed Graphs into Partial Orders is Hard

**arXiv ID:** 2610.37379 | [PDF](https://arxiv.org/pdf/2610.37379v1)

**作者:** Grzegorz Gutowski `[一作]` (Jagiellonian University), Mikołaj Rams `[通讯]` (Jagiellonian University)

**关键词:** `Computational Complexity` `Optimization` `Computational Efficiency` `Explainability and Interpretability` `Contrastive Learning` `Reinforcement Learning` `Prompt Engineering` `Mixture of Experts` `Graph` `Review/Survey Paper`

**🎯 论文内容**

本文研究有向图的部分序列分解与划分问题，提出部分序列分解数与划分数，并在有向无环图（DAG）上对两者的复杂性进行分类；证明对任意固定k≥2，判定DAG可否被覆盖/划分为k个部分序列是NP‑完备的，且即使在基底图为无三角形的DAG时仍保持硬性；同时证明在三角形无DAG中两者等价；给出了k=2时从NAE‑3SAT的构造，k≥3时从k‑着色的构造；引入并讨论部分序列临界DAG及其在构造中的作用。

**💡 创新点**

创新点在于：①首次给出部分序列分解与划分在DAG上的完整复杂性刻画；②证明三角形无DAG中分解数等于划分数；③提出并利用部分序列临界（pop‑critical）DAG进行多k情况的归约；④揭示分解数与划分数在一般DAG中可能不等，开辟新的研究方向。

**🔧 技术方法**

主要技术手段包括：构造变量与子句 gadget 的有向无环图；利用三角形无性与转移闭包约束来限制颜色；通过行图（line digraph）与图着色的对应关系把划分转化为图着色问题；构造部分序列临界DAG并使用其雨虹集（rainbow set）性质；从NAE‑3SAT、k‑着色等经典 NP‑难问题进行多层归约。

**📊 数据集**

文中未使用真实数据集，所有实例均为人工构造的有向无环图或其基底无三角形图，用于证明归约正确性。

**📈 对比分析**

方法以理论复杂度证明为主，没有实验对比；通过多重归约展示 NP‑完备性，证明在最坏情况下求解任意固定k的分解/划分不可多项式逼近；对比经典图着色问题显示相似的难度阶。

**⚠️ 局限性**

局限性包括：①对有向图中的循环结构（如支配图、强连通分量）未给出完整分析；②对锦标赛（tournaments）中k≥3的情况仍未解决；③部分序列临界DAG的构造目前仅存在存在性证明，缺乏多项式时间构造；④尚未确定分解数与划分数之间差距是否可无界。

---

## 637. Generalizable Target-Less Joint Intrinsic-Extrinsic Camera-LiDAR Calibration using Deep Pixel-Point Correspondences

**arXiv ID:** 2610.37350 | [PDF](https://arxiv.org/pdf/2610.37350v1)

**作者:** Simon Bultmann `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `Robotics` `Autonomous Driving` `Optimization` `Pose Estimation` `Depth Estimation` `Convolutional Neural Network` `Transformer` `Simultaneous Localization and Mapping` `Contrastive Learning` `Optical Flow` `RANSAC` `Joint Nonlinear Optimization` `Image` `Point Cloud` `Video`

**🎯 论文内容**

提出一种无目标、全自动的相机‑激光雷达联合标定方法，能够同时估计相机内参、径向切向畸变以及相机‑激光雷达外参。

**💡 创新点**

创新点在于将深度学习的像素‑点对应关系迁移到未校准、畸变的原始相机图像上，使用基于物理可行的相机参数增广的对应学习；并通过交替对应预测与联合非线性优化，实现对内参和外参的同步细化，从而在未见过的相机‑激光雷达配置上实现高精度标定。

**🔧 技术方法**

采用基于CMRNext架构的像素‑点对应网络，配合合成相机参数增广、RANSAC内点筛选和基于Ceres的联合最小二乘优化；初始化阶段使用COLMAP实现相机SfM、ORB‑SLAM3与FAST‑LIO2实现运动对齐。

**📊 数据集**

在KITTI（多序列）、Argoverse、PandaSet等公开数据集上训练，并在KITTI 00原始相机与三台实际机器人（Car、Spot、UAV）上进行零样本测试。

**📈 对比分析**

与MDPCalib、CMRNext、Koide等基准方法对比，联合标定方案在未见相机‑激光雷达设置下，平移误差约为2.25 cm、旋转误差0.05°，比基准方法低20–30%，且能同时恢复畸变参数；在真实机器人上亦取得最小的重投影误差（<4 px）和翻译误差。

**⚠️ 局限性**

局限性包括对COLMAP、ORB‑SLAM3/FAST‑LIO2等外部初始化的依赖；需要大量同步图像-雷达对（约100帧）导致约300 秒的计算时间；对极端光照或大畸变场景的鲁棒性尚待进一步验证。

---

## 638. Constraining the Commit Boundary: A DIO/ODI-Informed Authorization Guardrail for Agentic Entity Resolution

**arXiv ID:** 2610.37407 | [PDF](https://arxiv.org/pdf/2610.37407v1)

**作者:** Daniel Beránek `[一作]` (Independent Researcher), Daniel Beránek `[通讯]` (Independent Researcher)

**关键词:** `Artificial Intelligence` `Safty and Privacy` `Explainability and Interpretability` `Agentic AI` `Retrieval-Augmented Generation` `Contrastive Learning` `Graph` `Text` `Benchmark`

**🎯 论文内容**

本文提出一种基于 DIO/ODI 诊断分类的事务边界授权架构，旨在将用于推理的证据与用于永久身份变更的授权证据分离，防止错误假设通过检索放大并被误认为确认。

**💡 创新点**

创新点在于将 DIO/ODI 的四类失败模式（参照、语义、时间、上下文）映射到特定的身份保护约束，并在授权记录中强制执行“声明‑特定升迁、冲突处理、作用域绑定与版本完整性”这一合同，形成一个可验证的“提交‑时授权”不变量。

**🔧 技术方法**

采用的技术包括：DIO/ODI 参考模型（提供诊断词汇），SHACL/SPARQL 用于图约束验证，PROV‑O 用于证明链记录，基于版本的事务控制与重试机制，以及两阶段授权与提交验证算法。

**📊 数据集**

论文并未使用任何实际数据集；它仅提出了未来评估的语料构建方案，建议结合实体解析基准、Temporal‑EL 等带时间演化的实体集。

**📈 对比分析**

对比方法在设计上列出了四个基线（无约束、传统标识规则、通用授权门控、DIO/ODI‑增强门控），但由于未实现系统，论文未给出任何性能指标或实验结果，评估仅以指标与实验方案说明为准。

**⚠️ 局限性**

主要限制包括：不消除假设‑条件检索的自我强化；授权仅在事务边界生效，无法防止内部状态被错误假设污染；缺乏依赖感知的证据流；依赖领域策略、升迁过程与验证器的正确性；以及未对实际系统进行实现与实验验证。

---

## 639. Clearing the FoG: Training KG Agents to Retrieve the Evidence Behind Their Answers

**arXiv ID:** 2610.37405 | [PDF](https://arxiv.org/pdf/2610.37405v1)

**作者:** Zongyuan Sun `[一作]` (Beijing University of Posts and Telecommunications), Haoran Luo `[通讯]` (Nanyang Technological University)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Knowledge Distillation` `Representation Learning` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Robotic Intelligence` `Graph Neural Network` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Agentic AI` `Retrieval-Augmented Generation` `Chain-of-Thought` `Mixture of Experts` `Graph` `Text` `Benchmark`

**🎯 论文内容**

训练知识图谱(KG)代理检索答案背后的证据，提出 FoG 框架

**💡 创新点**

在问题合成阶段将支持三元组嵌入问题，并在 RL 训练中通过 evidence‑coverage process reward (ECPR) 对检索到的证据进行稠密奖励，使代理能够真正检索并使用证据而非仅凭记忆给出答案

**🔧 技术方法**

采用后向扩展的 evidence‑grounded synthesis、think‑act‑observe 的 KG 交互代理（含 7 种原子操作）、SFT+LoRA 预训练，随后使用 Group Relative Policy Optimization (GRPO) 的 RL 训练，并在每步给予 ECPR 过程奖励

**📊 数据集**

在三大 Wikidata 基准（WikiWebQuestions、QALD10‑en、GTSQA）上进行训练与评测，并在 Freebase 上的 WebQSP 进行零样本跨 KG 转移实验

**📈 对比分析**

与 LLM 推理、检索式方法和 Agent‑based 方法比较，FoG 在所有基准上均获得最高 Hit@1 与 F1，3B 参数的 FoG 甚至超过 4B KG‑R1；跨 KG 转移时在 WebQSP 上取得 77.2 Hit@1，接近专为 Freebase 设计的基线

**⚠️ 局限性**

主要限制包括 KG 的完整性与时效性不足、对外部教师模型（Kimi‑K2）的依赖，以及目前对长时序图交互的支持有限

---

## 640. Para-Gaussian Approximations of Random Deep Neural Networks

**arXiv ID:** 2610.37388 | [PDF](https://arxiv.org/pdf/2610.37388v1)

**作者:** Mihai Nica `[一作]`, Janosch Ortmann `[通讯]`

**关键词:** `Machine Learning` `Computational Efficiency` `Representation Learning` `Stochastic Differential Equation` `Gaussian Splatting` `Monte Carlo Simulation`

**🎯 论文内容**

本文提出了一种更准确的神经网络分布近似，称为para-Gaussian分布，旨在改善随机初始化神经网络在有限层数下的高斯行为近似。

**💡 创新点**

创新点在于引入了para-Gaussian近似，利用Hermite多项式构建了一种稍微非高斯的分布，并展示了其在深层网络中的有效性。

**🔧 技术方法**

使用了Hermite多项式和递归方法来近似神经网络的深层分布，特别是针对线性神经网络和ReLU神经网络进行了分析。

**📊 数据集**

使用了Monte Carlo模拟数据集来验证所提出的para-Gaussian近似与传统高斯近似的比较。

**📈 对比分析**

通过总变差距离比较了para-Gaussian近似与高斯近似，结果表明para-Gaussian近似在收敛速率上优于高斯近似，分别为O(n^-2)和O(n^-1)。

**⚠️ 局限性**

限制在于该方法主要针对随机初始化的神经网络，且在网络深度较大时，仍需进一步研究其在实际应用中的表现。

---

## 641. Agent-Conditioned Routing: Diverse Reasoning Paths Within One Frozen Model

**arXiv ID:** 2610.37375 | [PDF](https://arxiv.org/pdf/2610.37375v1)

**作者:** Ananth Kalyanasundaram `[一作]` (QpiAI), Aswanth Krishnan `[通讯]` (QpiAI)

**关键词:** `Artificial Intelligence` `Physics Related` `Transformer` `Mixture of Experts` `Agentic AI` `Text` `Mathematical Problems` `Benchmark`

**🎯 论文内容**

提出一种无训练、无用户提示的稀疏MoE推理机制——Agent-Conditioned Routing（ACR），通过固定共享/私有专家可行性掩码在同一模型实例中并行生成多条推理轨迹，从而在单次问题提示下实现多方法多解；

**💡 创新点**

创新点在于利用稀疏MoE的专家可行性掩码作为推理时的多样性控制器，实现单模型、单提示下的多方法推理，而无需额外训练、角色提示或模型副本；

**🔧 技术方法**

核心技术包括冻结稀疏MoE模型、基于掩码的专家可行性约束、持久掩码在前缀内保持一致、以及批量生成多轨迹的推理；

**📊 数据集**

使用了一个经过筛选的126题数学与物理多模式推理基准（涵盖积分、级数、组合、概率等12类），并通过人工评审和模型评审进行验证；

**📈 对比分析**

与温度采样、Expert‑Sample、RoE等推理多样性基线以及信息化序列提示对照，ACR在GLM 5.3模型上实现了约两倍、在GPT‑5.6 Sol上实现了约1.67倍的“至少两种正确方法”问题计数，整体正确率保持在95–98%之间；

**⚠️ 局限性**

局限性包括掩码未赋予专家语义专长，可能产生相同方法；多样性评估侧重方法计数，未评估中间步骤有效性；受限于已筛选的基准，未覆盖更广泛任务和模型，且正确率相较于传统温度采样略有下降。

---

## 642. Comparing Temporal Deep Learning Architectures for Discrete Cross-User Myoelectric Control

**arXiv ID:** 2610.37374 | [PDF](https://arxiv.org/pdf/2610.37374v1)

**作者:** Ethan Eddy `[一作]` (University of New Brunswick), Erik Scheme `[通讯]` (University of New Brunswick)

**关键词:** `Human-Computer Interaction` `Classification` `Recognition` `Explainability and Interpretability` `Computational Efficiency` `Recurrent Neural Network` `Transformer` `Convolutional Neural Network` `Contrastive Learning` `Adam optimization` `Biomedical Data` `Time Series` `Sequential` `Electrocardiogram`

**🎯 论文内容**

对612名用户的EMG数据进行离线与实时实验，比较五种时序深度学习架构在无校准交叉用户离散肌电控制中的性能。

**💡 创新点**

首次在真实连续采样、跨年份、跨国的数据上验证零射击（zero‑shot）交叉用户模型，并对手工特征与CNN特征提取方法做系统对比。

**🔧 技术方法**

使用LSTM、双向LSTM、GRU、RNN、Transformer等时序网络，结合1D CNN特征提取、能量阈值分割、Teager‑Kaiser能量检测和Adam优化器。

**📊 数据集**

主要数据集为公开的EMG‑EPN‑612（612名用户、6种手势）和新发布的MyoDisCo连续采集数据（14名参与者、随机提示手势）。

**📈 对比分析**

通过交叉验证、参数计数、层数实验以及Friedman/Nemenyi统计检验进行比较；GRU在离线集上最高达93%准确率，在MyoDisCo连续集上可达96%。

**⚠️ 局限性**

局限性包括手势类别仅限6种、连续数据集样本量有限、分割方法依赖能量阈值、模型对嵌入式实现的资源需求未进一步评估，以及在实际用户无反馈情境下的可解释性不足。

---

## 643. TimeLineXR: An Interactive Exploration of Time in Vision-Language Models

**arXiv ID:** 2610.37349 | [PDF](https://arxiv.org/pdf/2610.37349v1)

**作者:** Nidham Tekaya `[一作]` (University of Applied Sciences St. Pölten), Matthias Zeppelzauer `[通讯]` (University of Applied Sciences St. Pölten)

**关键词:** `Computer Vision and Pattern Recognition` `Explainability and Interpretability` `Representation Learning` `Transformer` `Vision Language Model` `Contrastive Learning` `Auto Encoder` `KPCA (Kernel Principal Component Analysis)` `Bezier Curve Fitting` `WebXR` `Gradio` `Three.js` `Image` `Text` `Multimodality`

**🎯 论文内容**

构建了可交互的 TimeLineXR 系统，允许用户在网页和 WebXR 体验中将图像投射到 VLM 的时间轴上，直观展示模型对时间的内在编码。

**💡 创新点**

创新点在于将隐式时间轴可视化为 3D 可探索对象，并通过 XR 环境实现实物图像的“抛掷投射”交互，使用户能够实时查询并比较不同图像在时间轴上的位置。

**🔧 技术方法**

使用 CLIP / EVA‑CLIP 文本/图像编码器、余弦核 KPCA 降维、拟合 1D Bézier 曲线、最近点投射等技术；前端采用 Gradio + Plotly、Three.js + WebXR；后端用 Python、PyTorch、FastAPI。

**📊 数据集**

以 TIME10k 为基准数据集，使用每一年生成的文本提示进行编码，并在此基础上构建时间轴；同时在演示中加载了多种类别的示例图像（手机、船只、仪器等）。

**📈 对比分析**

与原方法（S=13 的降维）相比，TimeLineXR 采用 S=3，平均绝对误差 MAE 为 15.30（CLIP）/20.81（EVA‑CLIP），比参考值 9.00/7.62 稍高，但已足够满足交互可读性；后端预测在 CPU 上约 150–300 ms，GPU 低于 50 ms，XR 投掷动画持续 0.5–1.3 s。

**⚠️ 局限性**

限制包括：1) 仅使用离线索引，无法实时更新时间轴；2) WebXR 受限于设备摄像头访问，无法直接捕捉现实物体；3) 降维至 3D 牺牲了部分估计精度；4) 目前仅支持上传/预设图像，未实现即时摄像头输入。

---

## 644. Aligning Institutional Power and Operational Ability

**arXiv ID:** 2610.37342 | [PDF](https://arxiv.org/pdf/2610.37342v1)

**作者:** Andrea Marino `[一作]` (University of Amsterdam), Tom van Engers `[通讯]` (University of Amsterdam)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Optimization` `Review/Survey Paper`

**🎯 论文内容**

本文提出了一套将多智能体系统的操作层与规范层进行对齐的正式框架，利用Petri网描述操作流程，使用反应规则管理机构性权力与义务的生命周期，并定义了操作性可执行性、制度合法性和功权对齐等可实现性与合法性属性，进而实现对规范设计中潜在矛盾的预先检测与修复。

**💡 创新点**

创新点在于：①首次将机构性权力与操作层能力分离并统一进流程模型；②引入“功权对齐”概念，将规范层的权力与义务与操作层的可执行动作直接关联；③通过“治理Petri网”将操作执行与机构状态同步；④提出基于状态空间的修复方法，区分操作层与机构层的可还原与扩展修复。

**🔧 技术方法**

采用的技术主要包括：Petri网及其执行语义、反应规则（reactive rules）实现机构状态更新、正式定义的对齐属性（如操作性可执行性、制度合法性）、治理网（governed net）模型，以及理论证明与状态空间分析。

**📊 数据集**

研究采用的是一个基于真实托运纠纷启发的示例场景（owner–broker–creditor的托运协议），并未使用公开数据集；所有分析均基于该案例的抽象Petri网模型。

**📈 对比分析**

方法比较主要通过对比原始治理网与修复后的治理网在状态空间中的对齐情况。修复后所有可达配置均满足对齐属性，证明了修复策略的有效性；论文未给出时间或空间复杂度的量化实验数据，主要为理论证明与图示演示。

**⚠️ 局限性**

局限性包括：①模型假设为确定性Petri网，未考虑随机/时序行为；②缺乏对齐属性的定量评估（如概率或延迟）；③修复策略仍需人工设计，缺乏自动化合成方法；④对大规模系统的可扩展性与实现细节未作实验验证。

---

## 645. diff-soup: GPU-native Differentiable Simulation and Inference for Large-scale Stiff Dynamical Systems

**arXiv ID:** 2610.37333 | [PDF](https://arxiv.org/pdf/2610.37333v1)

**作者:** Samuel Degnan-Morgenstern `[一作]` (Massachusetts Institute of Technology), Martin Z. Bazant `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `Mathematical Software` `Optimization` `Computational Efficiency` `Physics Related` `Stochastic Differential Equation` `Ordinary Differential Equation` `Diffusion model` `Auto Encoder` `Contrastive Learning` `Image` `Time Series` `Tabular` `Physics Related`

**🎯 论文内容**

开发了 diff-soup，一套基于 JAX 的可微分、稀疏导向的统一 PDE 套件，用于 GPU 原生模拟与大规模刚性动力学系统的推断；

**💡 创新点**

创新点在于将稀疏自动微分、GPU 加速稀疏直接求解（cuDSS）、可变步长/可变阶 BDF/NDF 整合到单一框架，支持高阶梯度、事件处理和全流程 PDE‑约束优化；

**🔧 技术方法**

使用的技术包括 JAX 自动微分与图着色、GPU 端稀疏 LU 分解、BDF/NDF VSVO 积分、Schur 补数、数据/模型并行、多 GPU 负载平衡、混合精度与检查点反向传播；

**📊 数据集**

实验数据包括合成的相分离材料图像（Cahn‑Hilliard）和三维填料反应堆模拟（带噪声的浓度场），用于贝叶斯逆推和嵌入式神经网络训练；

**📈 对比分析**

与 SUNDIALS、PETSc、Julia、MATLAB 等主流库对比，diff‑soup 在前向求解和梯度计算上比 CPU 方案快 1–2 阶，GPU 方案可实现 50× 速度提升，并在大规模（>10⁴ 自由度）问题上保持高精度；

**⚠️ 局限性**

局限性包括：目前仅支持 1 阶 DAE、对极大 3D 问题仍受 GPU 记忆和分解填充的限制、缺乏专门预条件器的 Newton–Krylov 支持，以及在小规模或非稀疏结构问题上 GPU 开销占比较高。

---

## 646. Hidden in Plain Sight: an Empirical Study about The Instruction Stacks of Coding Agents

**arXiv ID:** 2610.37332 | [PDF](https://arxiv.org/pdf/2610.37332v1)

**作者:** Felipe de Sant'Anna Paixão `[一作]` (Federal University of Bahia), Iftekhar Ahmed `[通讯]` (University of California at Irvine)

**关键词:** `Software Engineering` `AI Code Assistant` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Levenshtein distance` `Text` `Review/Survey Paper`

**🎯 论文内容**

研究了五个主流命令行编程代理（Claude Code、Codex CLI、Gemini CLI、Qwen Code、OpenCode）在不同版本发布中，完整指令堆栈（系统提示、工具定义、环境上下文等）的内容、结构和演化。

**💡 创新点**

①首次横向纵向系统捕获并量化各代理完整指令堆栈；②构建十二项功能分类代码表并使用LLM自动标注；③追踪指令变更的开发动机并将其归因于具体提交。

**🔧 技术方法**

采用容器化环境捕获模型请求，使用文本分段与LLM（GLM 5.3 Flash、Gemini 4 Argon）进行语义标注，利用Levenshtein距离、工具匹配等差异比较技术。

**📊 数据集**

约1,886个公开npm发布版本的请求（共约349,404句段）组成的数据集，覆盖五个代理的全部公开版本。

**📈 对比分析**

通过相对编辑距离、工具增删计数、功能共享度进行横纵比较，发现指令堆栈占请求字符>90%，变更集中且约48%具说明，工具定义在部分代理占比超过70%，共享能力有限但功能一致。

**⚠️ 局限性**

仅捕获首轮请求，未覆盖后续交互；Claude Code的动机缺失；LLM标注与追踪虽高精度但仍可能漏报；结果不一定适用于图形界面或不同配置的代理。

---

## 647. CTBakeGS: training-free baking of photorealistic CT volume renderings into 3D Gaussian splats

**arXiv ID:** 2610.37406 | [PDF](https://arxiv.org/pdf/2610.37406v1)

**作者:** Haruki Fukuda `[一作]` (Gunma University), Lars Ebert `[通讯]` (Zurich Forensic Science Institute)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Gaussian Splatting` `Neural Radiance Field` `Lighting Pre-baking` `Transfer Function Mapping` `Anti-aliasing Compensation` `Biomedical Data` `Computed Tomography`

**🎯 论文内容**

将CT体素渲染直接转换为三维高斯斑点（3D Gaussian Splatting）模型，无需任何训练或场景优化；

**💡 创新点**

创新点在于训练无、即时烘焙光照、使用自适应八叉树分块、将细胞映射为高斯斑点、保留低透明度斑点以避免薄层消失；

**🔧 技术方法**

采用3DGS、光照预烘焙、传递函数映射、基于细胞的高斯参数化、抗锯齿补偿与随机保留暗斑点等技术；

**📊 数据集**

使用公开的TotalSegmentator数据集中的六个1.5 mm全身CT扫描；

**📈 对比分析**

通过与渐进式光线投影基准对比，PSNR约32–34 dB，SSIM 0.96–0.98，LPIPS 0.03–0.09；转换耗时4–30 秒，渲染帧率约68 fps（800×800）；

**⚠️ 局限性**

局限性包括仅评估6个病例、仅1.5 mm分辨率、未验证解剖准确性、对更细切片或更大体积的适应性不足、缺少视角相关颜色、仅对单一光照模型的评估。

---

## 648. LinMix: Data Mixing as Linear Domain Attribution

**arXiv ID:** 2610.37390 | [PDF](https://arxiv.org/pdf/2610.37390v1)

**作者:** Hubert Baniecki `[一作]` (University of Warsaw), Martin Pawelczyk `[通讯]` (University of Vienna)

**关键词:** `Machine Learning` `Data-Centric Learning` `Explainability and Interpretability` `Optimization` `Domain Adaptation` `Transformer` `Large Language Model` `Contrastive Learning` `Linear Regression` `Mixture of Experts` `Text` `Benchmark`

**🎯 论文内容**

提出一种基于顶点采样（vertex sampling）的数据混合方法，并将其视为域级归因（domain attribution）问题，使用可解释的线性回归模型来预测并优化预训练语言模型的域权重。

**💡 创新点**

创新点在于：①把数据混合视为实验设计中的回归设计问题，利用D-最优原则采样混合点；②顶点采样能够在不需要超参数的情况下在数百个细粒度域上实现可解释的线性回归；③理论证明顶点采样在等置信度下最优，且Dirichlet采样在接近先验时会显著放大归因误差；④在大规模预训练（1B、4B参数）中验证顶点采样显著优于Dirichlet和非线性方法。

**🔧 技术方法**

技术包括：顶点采样（按容量上限饱和域），线性回归（β = (ΠᵀΠ)⁻¹Πᵀy），顶点采样的配对反向填充以保持正交；实验设计的D-最优性证明；对比非线性回归（LightGBM）和指数混合模型；对模型性能进行Spearman相关、R²等统计评估；并在多种基准上进行下游评测。

**📊 数据集**

使用WebOrganizer 200B-token语料库，拆分为24个粗粒度域和288个细粒度域（约占94%数据）。模型在该语料上预训练：OLMo2 1B参数、Qwen3.5 4B参数，训练数据分别为1B、2B、32B、80B tokens；下游基准包括ARC、HellaSwag、PIQA、Common Pile、MBPP（编程）、GSM8K（数学）。

**📈 对比分析**

与传统的Dirichlet采样+非线性回归（LightGBM、指数模型）对比。顶点采样在24域时与非线性方法相当，在288域时表现更好（Spearman ρ提升至约0.8，R²接近1）。在目标模型评估中，顶点采样在MBPP上比基线提升46%，在GSM8K提升14%，在大多数基准上平均提升约11%。

**⚠️ 局限性**

限制：①需要训练大量代理模型（proxy swarm）才能估计归因，成本高；②实验仅使用单一目标模型训练，未探究多目标或动态更新；③顶点采样仍受域容量上限（κ）约束，可能不完全符合真实数据重复情况；④目前只使用线性模型，尚未验证更复杂非线性模型在顶点采样下的潜在进一步提升。

---

## 649. Uneven Use, Similar Trust Generative AI in Denmark and Romania

**arXiv ID:** 2610.37348 | [PDF](https://arxiv.org/pdf/2610.37348v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## 650. How Many and How Much? Contrastive Learning with Negative Samples for Knowledge Graph Embedding

**arXiv ID:** 2610.37389 | [PDF](https://arxiv.org/pdf/2610.37389v1)

**作者:** Sooho Moon `[一作]` (Chung-Ang University), Yunyong Ko `[通讯]` (Chung-Ang University)

**关键词:** `Machine Learning` `Representation Learning` `Computational Efficiency` `Knowledge Distillation` `Graph Neural Network` `Contrastive Learning` `Graph`

**🎯 论文内容**

提出一种针对知识图谱嵌入的对比学习框架，通过自适应负样本预算分配与分布校准负样本权重来提升模型表现。

**💡 创新点**

创新点在于：① 将开放世界知识图谱视为其闭世界图谱的随机部分观测，基于观察度量负样本误判风险并动态调整每条正样本的负样本数量；② 在负样本权重计算时同时考虑负分数的离散度与集中度，形成分布校准权重。

**🔧 技术方法**

技术包括：基于贝叶斯观测模型的风险估计、负样本预算公式、标准化分数、熵归一化的集中度度量、预算感知的权重归一化以及经典的对比损失。

**📊 数据集**

使用了三个公开知识图谱：FB15k237、WN18RR 和 YAGO3-10，并在 TransE、RotatE、DistMult、ComplEx 四种嵌入模型上进行实验。

**📈 对比分析**

与 15 组基线方法（Uniform、Self‑Adv、SANS、EANS、ERDNS 等）在 MRR、Hits@10 上进行对比，实验显示该框架在所有模型‑数据集组合上平均排名第一，显著提升了 MRR（最多提升 2.4%）并且训练时间与显存占用均降低约 30%。

**⚠️ 局限性**

局限性在于：假设观测概率均匀且实体/关系集保持不变，无法直接处理异构观测或增量/归纳学习场景；此外仅调整负样本数量与权重，未进一步优化负样本采样策略本身。

---

## 651. VLA-Stem: A Lightweight Safety Mechanism for Vision-Language-Action Models via Reinforcement Learning

**arXiv ID:** 2610.37378 | [PDF](https://arxiv.org/pdf/2610.37378v1)

**作者:** Bojun Zou `[一作]` (University of Science and Technology of China), Jianmin Ji `[通讯]` (University of Science and Technology of China)

**关键词:** `Robotics` `Robotic Intelligence` `Safty and Privacy` `Reinforcement Learning from Human Feedback` `Transformer` `Reinforcement Learning` `Vision-Language-Action Model` `Knowledge Distillation` `Course Learning` `Multimodality` `Point Cloud`

**🎯 论文内容**

提出了一种名为VLA-Stem的轻量级强化学习安全机制，能在不细调基础VLA策略的前提下，将VLA产生的运动参考与实时障碍规避融合，实现安全执行。

**💡 创新点**

创新点在于：①脑干启发的上层任务意图与低层安全适配紧耦合；②通过轨迹级指导在保持空间意图的同时放宽时序约束；③采用教师‑学生蒸馏与难度递增的课程学习，实现对障碍诱发偏差的快速恢复。

**🔧 技术方法**

使用PPO强化学习与轨迹级奖励、教师‑学生蒸馏、课程学习、VLA条件化输入以及低维障碍感知，实现安全策略的训练与部署。

**📊 数据集**

在ManiSkill模拟环境下的PickCube、StackCube、Object Stowing任务中训练，随后在真实UR7e机器人上用50条遥控演示和AprilTag障碍标定进行验证；未使用公开数据集，全部为自建实验数据。

**📈 对比分析**

与原始VLA、VLA细调以及VLSA等基线对比，采用SUCC与COLL指标评估；在六个受限仿真场景下，VLA-Stem平均无碰撞任务成功率52.8%（基线23.7%），且在无障碍场景保持原VLA性能；真实世界实验中碰撞率从73–100%降至20–40%，任务成功率从0–20%提升至47–60%。

**⚠️ 局限性**

局限性包括：仅在结构化抓取/堆叠任务与低维障碍感知上验证；对更长时程、接触/力学安全目标的适应尚未探究；需进一步提升对视觉噪声和复杂障碍的鲁棒性。

---

## 652. Euclidean Crowding-Driven Adaptive Euclidean-Hyperbolic Graph Learning

**arXiv ID:** 2610.37373 | [PDF](https://arxiv.org/pdf/2610.37373v1)

**作者:** Hao Chen `[一作]` (Xi'an Jiaotong University), Hongying Zhang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `Machine Learning` `Representation Learning` `Optimization` `Graph Neural Network` `Contrastive Learning` `Riemannian Gradient Descent` `Attention Weight Learning` `Adversarial Loss` `Graph` `Multimodality`

**🎯 论文内容**

本文提出了基于层级图的欧氏拥挤度量和对应的泛化理论，并设计了一种轻量级欧氏‑超曲率融合框架 EH‑Fusion，实现了对不同层级结构图的自适应几何建模。

**💡 创新点**

创新点在于：① 用几何包络定理定量刻画欧氏嵌入的层级拥挤现象；② 将拥挤度与判别边缘相结合给出欧氏与超曲率空间的泛化边界比较；③ 在单一 Euclidean‑to‑Hyperbolic 映射基础上加入节点级邻域扩张权重的自适应融合，兼顾欧氏与超曲率优势。

**🔧 技术方法**

采用欧氏到 Poincaré 球面或 Lorentz 模型的指数映射、Riemannian 梯度下降、对抗损失、基于邻域扩张特征的注意力权重学习，并在 GCN/GAT backbone 上实现。

**📊 数据集**

实验使用五个标准图数据集（Cora、Citeseer、Airport、Disease、Ele‑Fashion）以及多模态视觉编码器（ImageBind、CLIP、DINOv2、ViT）。

**📈 对比分析**

在多种基准模型（GCN、GAT、HGCN、FHNN、Poincaré Embedding 等）中，EH‑Fusion 在高层级（δ≈0–1）数据上平均提升 AUC 1–2%，在中等层级数据上保持与最优单模态相当，且在低维度下表现稳健；单层指数映射已能实现接近全超曲率网络的性能。

**⚠️ 局限性**

主要限制包括：需要先估计图的 Gromov δ‑超曲率以指导模型选择，超曲率映射的数值稳定性与计算开销；对非树状、弱层级图仍可能出现超曲率过拟合；并未给出明确的阈值或自适应判别机制来自动切换几何。

---

## 653. Tokens of Motion: Primitive Motion Learning for Human-Humanoid Skill Transfer

**arXiv ID:** 2610.37330 | [PDF](https://arxiv.org/pdf/2610.37330v1)

**作者:** Hanwen Ren `[一作]` (United Imaging Intelligence), Ziyan Wu `[通讯]` (United Imaging Intelligence)

**关键词:** `Robotics` `Robotic Intelligence` `Representation Learning` `Reinforcement Learning from Human Feedback` `Transformer` `Contrastive Learning` `Auto Encoder` `Vision-Language-Action Model` `Diffusion model` `Reinforcement Learning` `Video` `Sequential`

**🎯 论文内容**

提出 M-PACT 框架，利用大规模人类视频学习可复用的运动原语，并通过轻量化解码器实现跨体态迁移、上下文预测与闭环执行。

**💡 创新点**

创新点在于把运动结构拆分为与体态无关的原语与体态专属解码器，既能从人类数据中高效学习原语，又能仅通过少量机器人演示快速适配；同时引入自监督分段、离散化词表化与上下文条件预测。

**🔧 技术方法**

采用自监督运动分段 SkillSeg、Embodied Primitive Motion Tokenization (EPMT)（离散化平移/孔径、连续旋转）、向量量化、轻量化解码器与 CC‑EPMT 的视觉/语言上下文预测；使用状态正则化与效果重构。

**📊 数据集**

主要使用 EgoDex、HoloAssist、HumanoidEveryday 三个手部轨迹数据集，并在真实 Humanoid 上进行验证。

**📈 对比分析**

通过与直接轨迹重现、基线方法对比，M‑PACT 在转移到 HoloAssist 时保持所有码表活跃，重现误差仅为 cm 级、旋转误差 <1°；在 HumanoidEveryday 只需适配解码器即可获得低误差；闭环预测 Top‑1 约 33%，重现误差与训练原语相差不到 0.5 cm，跨体态迁移仅需约 100 次机器人演示。

**⚠️ 局限性**

局限在于旋转与孔径预测误差仍较大，闭环预测准确率有限；需要少量机器人演示；对任务和场景的泛化仍受限；模型主要聚焦手部轨迹，未充分处理复杂物体交互。

---

## 654. The Capacity of Private Information Retrieval from Coded Storage Systems with Colluding, Robust and Byzantine Servers

**arXiv ID:** 2610.37412 | [PDF](https://arxiv.org/pdf/2610.37412v1)

**作者:** Rui Sun `[一作]` (Shandong University), Yiwei Zhang `[通讯]` (Shandong University)

**关键词:** `Information Theory` `Safty and Privacy`

**🎯 论文内容**

提出了在含有协作、无响应与Byzantine服务器的GRS编码存储系统中实现私密信息检索的线性全支持秩方案，并给出了其容量表达式；

**💡 创新点**

通过利用Schur乘积码与统一选择/截断MDS属性的查询矩阵，首次在非退化情况下（K>1,T>1）对U-B-MDS-TPIR容量进行定量分析，并在多文件检索中实现了P≥M/2时的容量最优；

**🔧 技术方法**

核心技术包括GRS码的编码与解码、Schur乘积码的生成、均匀选择/截断全秩矩阵构造、线性私密查询设计以及多文件线性组合的MDS编码；

**📊 数据集**

该工作为理论研究，不涉及实验数据集，全部基于信息理论与编码理论的抽象模型；

**📈 对比分析**

通过与已知的MDS-TPIR、TPIR及MDS-PIR容量上界对比，证明在N-U-2B≥KT且K=1或T=1时方案达到容量极限；

**⚠️ 局限性**

对K>1且T>1的非线性或非全支持秩方案的容量仍未完全解决，且当前构造对文件长度和字段大小有一定规模要求。

---

## 655. Game-Theoretic Solvers with Bayesian Opponent Modeling and Moral Rhetoric Framing for GLEE

**arXiv ID:** 2610.37397 | [PDF](https://arxiv.org/pdf/2610.37397v1)

**作者:** Hayden Arnold `[一作]` (Korea Advanced Institute of Science and Technology), Hayden Arnold `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `Computer Science and Game Theory` `Reinforcement Learning from Human Feedback` `Optimization` `Explainability and Interpretability` `Transformer` `Large Language Model` `Reinforcement Learning` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Mixture of Experts` `Text` `Sequential` `Benchmark`

**🎯 论文内容**

开发并部署了一款能在GLEE比赛中同时做出数值决策和生成策略语言的多任务代理。

**💡 创新点**

创新点在于将传统博弈理论解（Rubinstein、Faratin、Kamenica–Gentzkow）与在线贝叶斯估计、Thompson采样道德框架和证据-直觉语义层相结合，并针对理论假设违背的真实对手设计了多重防护措施。

**🔧 技术方法**

核心技术包括经典博弈解算器、在线贝叶斯更新、衰减式后验收缩、基于Moral Foundations的5臂Thompson采样、语言模型生成消息以及对冲策略（如抑制谎言、有限共享信息）。

**📊 数据集**

主要数据来源于GLEE比赛的实时人类与AI对手，以及由作者自定义的11种人工对手原型进行的13,200局蒙特卡罗仿真。

**📈 对比分析**

通过在比赛排行榜、蒙特卡罗对手原型的中位得分、去掉对手估计模块的消融实验以及对语言框架与结果关联的统计检验进行比较；表现显示在谈判和说服场景中，模型能显著提升平均得分，且在某些原型下达到或超过基准，但总体排名仍位于后半段。

**⚠️ 局限性**

局限性包括：道德框架与信息注册的相关性分析缺乏因果实验；对手估计模块在实战中的真实贡献未单独测量；跨游戏学习受限于对手身份披露率低；以及整体排行榜表现仅中等。

---

## 656. When Is Additional Retrieval Worthwhile? A Cost-Aware Study of Evidence Gain and Answer Reliability in RAG

**arXiv ID:** 2610.37358 | [PDF](https://arxiv.org/pdf/2610.37358v1)

**作者:** Zhouting Zhao `[一作]`, Zhouting Zhao `[通讯]`

**关键词:** `Information Retrieval` `Retrieval` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Retrieval-Augmented Generation` `Prompt Engineering` `Contrastive Learning` `Text` `Benchmark`

**🎯 论文内容**

研究了一种基于成本的两动作检索策略，利用仅来自基线检索的特征预测是否需要额外的混合检索，以减少无效的检索调用。

**💡 创新点**

创新点在于将检索增益预测与成本考量相结合，使用线性回归估计混合检索的边际收益，并在训练前冻结模型与阈值，避免过拟合。

**🔧 技术方法**

使用了密集检索+交叉编码rerank、BM25+密集检索的混合检索、岭回归预测模型以及基于文档分组的交叉验证与配对自助法评估。

**📊 数据集**

采用了 QASPER 基准数据集（包含科研论文的问答与证据标注），在训练集上进行政策学习，并在 60 篇文章的留存集上进行检索与答案质量评估。

**📈 对比分析**

在检索层面，冻结策略与始终混合检索在 Hit@10 上相当（78.70%），但仅触发混合检索 24.5% 的问题；在答案层面，策略未显著提升正确性或可信度，差异均不具统计显著性。

**⚠️ 局限性**

主要限制包括检索增益与答案质量之间的脱节、样本中救援与损害事件稀缺导致模型预测弱、评估依赖 LLM 判定而非独立人工标注，以及成本模型仅为单位化惩罚，未衡量真实运行时开销。

---

## 657. HIDE: Hierarchical Isolation via Documentation and Evidence for Rustc Bugs

**arXiv ID:** 2610.37343 | [PDF](https://arxiv.org/pdf/2610.37343v1)

**作者:** Tianrong Zhang `[一作]` (Peking University), Dan Hao `[通讯]` (Peking University)

**关键词:** `Software Engineering` `AI Code Assistant` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Code`

**🎯 论文内容**

HIDE 是一种基于层级 LLM 的 Rust 编译器（rustc）Bug 隔离方法，能够在一个 Bug 产生时快速给出一份文件级别的可疑文件列表。

**💡 创新点**

其创新点在于将编译器的层级结构（crate→module→file）与 LLM 文档聚合相结合，形成“Layered Documentation Aggregation”，并通过“Hierarchical Bug Isolation”结合 Scoped Candidate Competition 与 Evidence‑Guided Code Slicing 两种上下文管理机制，解决分布式功能理解、海量文件搜索与上下文过载三大难题。

**🔧 技术方法**

核心技术包括：大语言模型（GPT‑5.4、Gemini‑3.1‑Pro‑Preview、DeepSeek‑V4‑Pro 等）生成层级文档；利用 rustc 查询日志与编译器日志作为执行证据；使用 Top‑K 选取与代码切片评分的自定义评分函数；以及多层次的候选压缩与局部竞争策略。

**📊 数据集**

实验使用了 160 个可复现的 rustc Bug（94 个 ICE、46 个完整性、20 个有效性），覆盖 rustc 1.53.0–1.85.0 版本的源代码。

**📈 对比分析**

与 Agentless、SoapFL、GraphLocator 三个基线比较，HIDE 在 Accuracy@1 与 Accuracy@10 上分别提升 5.83%/6.04%，平均运行时间 154 s、平均 token 145,413，显著低于基线（Agentless 269 s/2,111,809 token、SoapFL 306 s/580,901 token、GraphLocator 445 s/99,642 token）。

**⚠️ 局限性**

主要局限在于 Accuracy@1 仍仅 42%（定位首位仍有提升空间）、仅评估 ICE/完整性/有效性三类 Bug，方法级定位表现有限，且对 LLM 生成质量与稳定性敏感。

---

## 658. CodeCap-Bench: Can Code Serve as a Good Video Caption?

**arXiv ID:** 2610.37341 | [PDF](https://arxiv.org/pdf/2610.37341v1)

**作者:** Jiafu Tang `[一作]` (Nanjing University), Jiaheng Liu `[通讯]` (Nanjing University)

**关键词:** `Computer Vision and Pattern Recognition` `AI Code Assistant` `Transformer` `Large Language Model` `Vision Language Model` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Video` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

研究可执行代码作为视频字幕的可行性，并构建了专门评估的基准 CodeCap‑Bench。

**💡 创新点**

创新点在于把可执行代码视为视频描述媒介，提出细粒度视觉与时空一致性评估框架，并将其与自然语言字幕进行系统对比。

**🔧 技术方法**

使用多模态大型语言模型（GPT‑6‑astra、GPT‑6‑sol、Opus‑5 等）、工具代理（Pi Agent、Codex 等）以及渲染引擎，评估视觉相似度、对象级时空 IoU 与细粒度信息保留。

**📊 数据集**

采用 323 个结构化 2D 视频数据集（桌面界面、2D 游戏、动画、可视化、图标等），并提供焦点与对象级时空标注。

**📈 对比分析**

通过问题回答、视频重建（Appearance、Motion Pattern、Keyframe Similarity）和细粒度评分等多维度指标比较，结果显示可执行代码字幕在问题回答准确率和重建相似度上明显优于自然语言字幕，最高模型整体保留得分约 73，动态信息保留更难。

**⚠️ 局限性**

局限性包括：大型专有模型仍难完整捕捉复杂时序，开源模型性能差距显著；代理生成不一定提升精度；成本与效率与准确度不完全正相关。

---

## 659. Mind the Intent Gap: A Methodology for Collaborative Agent-Driven Software Engineering

**arXiv ID:** 2610.37340 | [PDF](https://arxiv.org/pdf/2610.37340v1)

**作者:** Nicolas Tacheny `[一作]` (University of Mons), Saïd Mahmoudi `[通讯]` (University of Mons)

**关键词:** `Software Engineering` `AI Code Assistant` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Agentic AI` `Retrieval-Augmented Generation` `Prompt Engineering` `Text` `Code` `Document` `Tabular`

**🎯 论文内容**

提出并验证了 Collaborative Agentic Intent Alignment (CAIA) 方法论，帮助 AI 代理在软件团队中以一致、可追溯的意图工作，并通过知识库持续记录和传递变更的决策与理由。

**💡 创新点**

创新点包括：① 以微团队对变更意图进行四步对齐（描述、分析、评审、对齐）来消除意图差距；② 将代理的使用限制为单一“变更主控”操作，避免多代理输出不一致；③ 通过知识代理在每次变更后自动提出知识库更新，保持知识库与产品同步；④ 将整个流程与企业内部自动化平台绑定，支持快速落地。

**🔧 技术方法**

使用技术：基于大型语言模型（LLM）的三类代理（助手、构建、知识）；流程管理平台；知识库（结构化记录产品设计与决策）。

**📊 数据集**

使用的数据集：Ni2 公司的内部软件项目变更记录、代码、测试和文档组成的知识库；未公开公开数据集，实验集中于实际企业项目。

**📈 对比分析**

比较方法：在 Ni2 生产环境中进行主观评估（团队成员反馈），并计划在大学课程中做对照实验，比较执行时间与结果与预期的吻合度。初步结果显示 CAIA 组在保持相似时间的同时，修正次数更少，结果更贴近预期。

**⚠️ 局限性**

局限性：缺乏大规模量化评估，实验样本主要来自单一企业；对不同规模、分布式团队的适应性未知；代理在意图未完全对齐时仍可能做出错误推断；知识库维护成本与治理仍需进一步研究。

---

## 660. ScIntBench: Physical Interaction Plausibility in 3D Scene Generation

**arXiv ID:** 2610.37326 | [PDF](https://arxiv.org/pdf/2610.37326v1)

**作者:** Wendi Ying `[一作]` (University of Western Australia), Ajmal Saeed Mian `[通讯]` (University of Western Australia)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Benchmark` `Transformer` `Large Language Model` `Diffusion model` `Vision Language Model` `Physics Simulation (Blender + Isaac Sim)` `Text` `Mesh` `Point Cloud` `Multimodality` `Benchmark`

**🎯 论文内容**

提出ScIntBench基准，用来评估文本驱动的3D室内场景在交互时的物理可行性与稳定性。

**💡 创新点**

引入四种交互相关度量（ISR、ICP、PIV、DP），从互动可用性、执行进度、后续物理完整性与动态物理可信度四个维度补全传统语义与静态物理评估。

**🔧 技术方法**

利用多视角渲染、几何检查、被动与主动物理模拟（Blender + Isaac Sim），并借助大型语言模型对交互任务与评估进行构造与判分。

**📊 数据集**

基于SceneEval的30条室内场景文本提示（10类房间），并生成203个标准化交互任务，共计2436个评估实例。

**📈 对比分析**

对12种不同范式的生成方法（任务特化、LLM/VLM、代理式与通用编码代理）进行评估，结果显示：静态物理良好不等于交互成功；部分方法无法完成任何交互；通用编码代理在语义功能与ISR上表现优异，但物理完整性仍有欠缺。

**⚠️ 局限性**

局限在于：交互任务的自动构造仍依赖LLM，交互执行的可行性取决于原始模型是否具备必要的物理关节与属性，评估仍无法覆盖所有真实世界交互细节。

---

## 661. CoS-VQ: Coordinated Shared-Scale Vector Quantization for Large Language Models

**arXiv ID:** 2610.37336 | [PDF](https://arxiv.org/pdf/2610.37336v1)

**作者:** Zukang Xu `[一作]` (Houmo AI), Dawei Yang `[通讯]` (Houmo AI)

**关键词:** `Machine Learning` `Computational Efficiency` `Knowledge Distillation` `Optimization` `Transformer` `Large Language Model` `Vector Quantization` `Contrastive Learning` `Auto Encoder` `Text`

**🎯 论文内容**

提出 CoS‑VQ 框架，对大型语言模型进行低比特量化，通过共享尺度向量量化保持低维码字，并联合优化硬索引和共享尺度。

**💡 创新点**

创新点在于使用多尺度候选池、残差感知切线可分上界以及下包络区间搜索，实现在不枚举代码组合的前提下完成联合索引‑尺度优化，从而突破单点更新停滞的瓶颈。

**🔧 技术方法**

主要技术包括向量量化、残差感知 Majorization、可分上界构造、下包络区间求解、候选池构造以及真误差验证。

**📊 数据集**

使用 RedPajama‑1T 进行校准，评估数据集包括 WikiText‑2、C4、LM Evaluation Harness、IFEval、LiveCodeBench、GPQA‑D、AIME 2025 等。

**📈 对比分析**

与 BF16、PV‑Tuning、QuIP#、QTIP、NoWag、AQLM、LLVQ 等基线在 2‑比特目标下对比，CoS‑VQ 在 Qwen3 与 Llama‑3.1 系列模型上平均困惑度更低、下游任务准确率更高，且推理吞吐提升约 1.5–1.7 倍。

**⚠️ 局限性**

局限性包括：优化仅在有限候选池内，理论最优性仅针对上界；残差感知上界可能过于保守，难以逃离某些坐标停滞；候选质量受多尺度探测与码本设计影响；实验仅覆盖部分 LLM 架构和低比特配置，推理效率受硬件实现限制。

---

## 662. Beyond the Readout: A Latent Numerical Frame Shared Across Modality and Operation

**arXiv ID:** 2610.37422 | [PDF](https://arxiv.org/pdf/2610.37422v1)

**作者:** Oriol Juan Sabater `[一作]`, Xavier Suau `[通讯]`

**关键词:** `Artificial Intelligence` `Representation Learning` `Explainability and Interpretability` `Transformer` `Large Language Model` `Contrastive Learning` `Prompt Engineering` `Retrieval-Augmented Generation` `Text` `Image` `Multimodality`

**🎯 论文内容**

本文通过因果干预和子空间学习，揭示了多模态语言模型在文本和图像输入下进行加减运算时的内部数值表征结构；

**💡 创新点**

创新点包括：①使用因果子空间学习方法辨别不同计算路径是否共享相同坐标或仅共享几何关系；②证明四种条件共享一个13维隐藏子空间，该子空间与输出读数无关，体现了抽象数值概念；③展示隐藏子空间在时间上调度，控制下一步输出，揭示了模型的“先算后读”机制；

**🔧 技术方法**

采用分布式对齐搜索（Distributed Alignment Search）得到22维因果子空间；通过旋转、缩放映射和正交群同步对子空间进行对齐；利用RSA、CKA等相似度量与自定义交叉条件检验；再用自回归干预、数字检索等评估方法；

**📊 数据集**

使用合成算术数据集：从[0,99]整数生成所有加法与减法表达式，并将其以纯文本或在白底黑字图像中呈现，共计约2万条示例；

**📈 对比分析**

比较方法：直接子空间转移、旋转+缩放对齐、读数子空间剔除、隐藏子空间同步等；性能表现为：①直接转移平均45.5%恢复；②旋转+缩放后恢复91.4%；③隐藏子空间在12个关系上平均余弦0.807，检索top‑1 75.5%；④在时间延迟干预中，隐藏子空间可在下一步重定向77.6%的单位位；

**⚠️ 局限性**

局限性包括：仅研究加减运算（不含乘除）、仅针对整数[0,99]范围、只使用两种模态（文本与图像）以及特定模型家族，且子空间维度固定，未探究更大数值或更复杂算术任务的可迁移性。

---

## 663. Time-Dynamic Semantic-Activity Entanglement: Quantifying Communication-Based Synchrony in Collaborative VR

**arXiv ID:** 2610.37427 | [PDF](https://arxiv.org/pdf/2610.37427v1)

**作者:** Pooja Pol `[一作]` (Technische Hochschule Augsburg), Qing Huang `[通讯]` (Technische Hochschule Augsburg)

**关键词:** `Human-Computer Interaction` `Explainability and Interpretability` `Representation Learning` `Transformer` `Large Language Model` `Contrastive Learning` `Auto Encoder` `Retrieval-Augmented Generation` `Audio` `Text` `Multimodality`

**🎯 论文内容**

开发了一种时间动态语义-活动纠缠（TSAE）指标，用来定量描述协作VR环境中团队成员的说话活动与语义对齐，并通过Gini系数评估团队层面的同步分布。

**💡 创新点**

创新点在于将说话量与语义相似度结合成无向双向纠缠度量，并利用Gini系数对团队同步的分布进行可解释的动态分析，区别于传统基于邮件或单向通信的同步指标。

**🔧 技术方法**

使用Whisper自动语音识别转写文本，Jina v2模型进行late‑chunking的句子嵌入，余弦相似度与欧氏距离调制的纠缠计算，以及Gini系数、Lorenz曲线和Bootstrap置信区间等统计方法。

**📊 数据集**

15场德国语协作VR游戏会话数据集，每个会话约25分钟，团队规模5–8人，使用48 kHz单声道音频并转录为完整对话文本。

**📈 对比分析**

与先前基于邮件网络的纠缠方法对比，TSAE在协作VR场景下显示同步分布更均匀；不同时间窗口的Gini系数呈下降趋势，表明较低的不平等；虽然未给出传统“性能”指标，但通过分布特征和统计显著性展示了方法的有效性。

**⚠️ 局限性**

局限包括：仅使用德语数据，缺乏人口统计和先前关系信息；未评估自动语音识别误差对结果的鲁棒性；无法直接关联任务绩效；方法对不同类型CPS任务的泛化性尚未验证。

---

## 664. Privacy-Preserving Over-the-Air Federated Learning over Cell-Free Massive MIMO System with User-Centric Clustering

**arXiv ID:** 2610.37424 | [PDF](https://arxiv.org/pdf/2610.37424v1)

**作者:** Manobendu Sarker `[一作]` (Polytechnique Montréal), Soumaya Cherkaoui `[通讯]` (Polytechnique Montréal)

**关键词:** `Information Theory` `Federated Learning` `Safty and Privacy` `Optimization` `Contrastive Learning` `Reinforcement Learning` `Chain-of-Thought` `Prompt Engineering` `Retrieval-Augmented Generation` `Image`

**🎯 论文内容**

在单元格无基站（cell‑free）大规模MIMO网络上，设计了跨层优化框架，联合调度 AP‑UE 聚类、上行功率分配与差分隐私（DP）噪声量化，实现 OTA（over‑the‑air）联邦学习（FL）中的隐私与通信协同优化。

**💡 创新点**

①将物理热噪声视为可利用的隐私资源并在 ergodic zCDP 框架下量化；②首次在 CF‑OTA 架构下同时考虑用户聚类、功率与 DP 噪声的混合整数非凸优化，并通过差分凸（DC）+贪心坐标下降实现；③揭示了隐私受限与噪声受限两种工作模式及其自适应切换。

**🔧 技术方法**

交叉层优化、差分凸（DC）/凸凹程序（CCP）功率分配、贪心坐标下降聚类、Chebyshev/熵硬化分析、零浓度 DP（zCDP）设计、基于统计的 OTA 误差模型、FedAvg 联邦学习训练。

**📊 数据集**

MNIST 与 CIFAR‑10 数据集，用于端到端 FL 性能验证。

**📈 对比分析**

与五种基线（Privacy‑Unaware、Fixed‑Cluster、Equal‑Power、Single‑Cell、Naive‑DP）在不同 ε 级别下比较，提出方案在严苛隐私下实现近乎无噪声学习，聚合方差下降约 89%，MNIST 训练精度 97%，CIFAR‑10 训练精度 47%，显著优于基线。

**⚠️ 局限性**

主要限制：依赖大规模信道硬化和热噪声期望代理，有限天线数导致保守估计；假设全参与、静态大规模统计，未考虑部分参与、频率选择性或动态聚类；实验采用统计模型而非真实时变信道。

---

## 665. Neighborhood Deprivation Shapes Post-Flood Population Losses in Pune and Bandung

**arXiv ID:** 2610.37418 | [PDF](https://arxiv.org/pdf/2610.37418v1)

**作者:** Sukie Xiuqi Yang `[一作]` (Hong Kong University of Science and Technology), Catherine Chen `[通讯]` (Louisiana State University)

**关键词:** `Computers and Society` `Cluster Analysis` `Logistic Regression` `Tabular` `Time Series` `Remote Sensing Data` `Geospatial Data` `Review/Survey Paper` `Agriculture Related` `Physics Related`

**🎯 论文内容**

通过聚类移动设备检测数据与洪水边界，分析印度浦那和印度尼西亚邦加德市洪灾后社区设备出现变化，探讨贫困指数与迁移模式关系。

**💡 创新点**

首次在全球南方快速城市化背景下使用卫星洪水多边形与设备层面移动数据相结合，揭示贫困与洪灾暴露交互影响的迁移梯度。

**🔧 技术方法**

采用层次聚类（Ward 方法）与二项 Logistic 回归，并利用轮廓系数和树形图确定两类迁移轨迹。

**📊 数据集**

使用 NetMob 2024 设备检测数据、Dartmouth Flood Observatory 洪水多边形、Global Gridded Relative Deprivation Index (GRDI) 贫困指数及 ASTER 高程数据。

**📈 对比分析**

通过双城比较法，聚类结果两城均呈两类，回归显示低贫困区更易出现设备增加，模型在不同洪水编码下表现稳健，解释力显著。

**⚠️ 局限性**

洪水边界粗糙、设备数据仅代表拥有智能手机的群体、缺乏个体迁移路径、未纳入道路或避难所等基础设施，导致对最脆弱人群影响估计可能被低估。

---

## 666. The Envelope Layer: An Underestimated Surface of LLM Agent Defense

**arXiv ID:** 2610.37415 | [PDF](https://arxiv.org/pdf/2610.37415v1)

**作者:** Yuelin Han `[一作]`, Yuelin Han `[通讯]`

**关键词:** `Cryptography and Security` `Safty and Privacy` `Explainability and Interpretability` `Adversarial Attack` `Transformer` `Large Language Model` `Prompt Engineering` `Agentic AI` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

本文通过 1,200 次 2×2 因子实验和 320 次记忆实验，系统评估并细化了“Envelope 层”在 A2A 代理安全中的边界与效果。

**💡 创新点**

创新点：① 将 Envelope 层从单一“入口通道”细化为两层——入口通道 (L1a) 与上下文组装面 (L1b)；② 发现标签强度、信封真实性与历史记忆共同决定安全边界；③ 通过对记忆正负影响的对比揭示记忆两面性；④ 在多前端、多 LLM 后端下揭示前端上下文组合对标签效用的影响。

**🔧 技术方法**

使用技术：A2A‑TIBA 代理攻击链、GDA 测量方法、A/B/C/D 四类结果分类、Semantic Breach Rate 与 Semantic Refusal Rate 评价指标、2×2 因子设计、记忆条件（成功、失败、冲突）设计、基于 LiteLLM 的统一 LLM 接口、DeepSeek‑Flash 判定器。

**📊 数据集**

数据集：采用 1,000 条基线攻击案例（来自前工作）扩展为 1,200 条实验数据，包含 3 种前端代理（Hermes、OpenClaw、Claude Code）× 5 种后端 LLM（MiniMax‑M3、DeepSeek‑v4‑Pro、GLM‑5.3、Kimi‑k2.7‑Code、Qwen3.8‑Max），再加 320 条记忆实验数据。每种配置均有 20 次重复。

**📈 对比分析**

比较方法：对比原始无标签基线、弱标签、强标签以及伪造信封条件下的 Semantic Breach Rate；对比不同 LLM 后端与前端组合的差异；对记忆条件的比较（成功 vs 失败 vs 冲突）。结果显示：强标签在原始信封下能显著降低 breach rate（最高 61.7% 降至 0%），伪造信封削弱或反转效果；记忆成功显著提升 breach rate（81.2%），记忆失败显著降低（6.2%）。

**⚠️ 局限性**

Limitation：① 实验未随机化、可能存在时间混杂；② 每组仅 20 次重复，置信区间宽；③ 只使用单一攻击技能，未评估自适应攻击；④ 未加入 A2A 权限细分；⑤ OpenClaw‑Qwen3.8 组 token 上限不同；⑥ 判定器与攻击者使用同一模型，可能存在判定偏差；⑦ 未对标注基线与处理组进行完全同步对照。

---

## 667. AuRaSim: A Ray-Tracing Based Automotive FMCW Radar Simulator for Dynamic Scenarios

**arXiv ID:** 2610.37403 | [PDF](https://arxiv.org/pdf/2610.37403v1)

**作者:** Zhaoze Wang `[一作]` (HELLA GmbH & Co. KGaA), Markus Gardill `[通讯]` (Brandenburg University of Technology)

**关键词:** `Robotics` `Autonomous Driving` `Diffusion model` `Optical Flow` `Simultaneous Localization and Mapping` `Image` `Video` `Point Cloud` `Benchmark`

**🎯 论文内容**

本文实现了一个完整的汽车 FMCW 雷达模拟框架 AuRaSim，能够从动态数字孪生场景、可配置雷达硬件、物理传播模型到完整 ADC 信号的端到端生成。

**💡 创新点**

创新点包括：①基于射线追踪的路径解析与图像方法，②场景 FOV 方向限定的 Fibonacci 射线采样，③在每个射线交点使用顶点速度场实现局部多普勒建模，④利用 Blender 与 Sionna RT 开源技术实现可扩展的全流程仿真，并通过数字孪生验证其真实性。

**🔧 技术方法**

使用的技术主要有：Blender 用于构建动态 3D 场景和动画；Sionna RT 进行 GPU 加速的射线追踪和电磁场计算；FMCW 信号模型与多普勒计算；可配置的 MIMO 天线阵列与辐射模式；以及路径级显式记录用于后续 ADC 合成。

**📊 数据集**

实验数据集来自 Gao 等人发布的 19,800 帧同步 77 GHz 雷达与 RGB 图像的校园、道路和停车场场景，雷达配置为 TI AWR1843 77 GHz，2×4 TDM MIMO，128 样/脉冲，510 脉冲/帧，4 MSps 采样。

**📈 对比分析**

通过将模拟得到的范围–多普勒图与真实雷达测量进行对齐，使用 RMSE 与 PSNR 进行量化评估；与 RadarSimPy 基线相比，AuRaSim 在三组场景中平均 RMSE 降低 38.8%（从 0.240 降至 0.147），PSNR 提升 3.5 dB（从 13.43 dB 提升至 16.93 dB），显示出更高的信号真实性。

**⚠️ 局限性**

主要局限包括：未对雷达噪声、相位噪声、干扰、校准误差及恶劣天气等传感器特性进行建模；高阶多路径和高密度动态场景仍需大量计算；材料参数的精确性和测量驱动的校准尚待进一步完善。

---

## 668. Code uniqueness of 4-cell connected disjoint clusters in spatial chromatic model

**arXiv ID:** 2610.37395 | [PDF](https://arxiv.org/pdf/2610.37395v1)

**作者:** Weining Zhu `[一作]` (Zhejiang University), Weining Zhu `[通讯]` (Zhejiang University)

**关键词:** `Computational Geometry` `Computational Efficiency` `Representation Learning` `Optimization` `Contrastive Learning` `Supervised Fine-Tuning` `Physics Related`

**🎯 论文内容**

本文对空间染色模型（SCM）中的四胞连通无交集簇（4-ccd）进行理论分析与证明，确认其染色码唯一性。

**💡 创新点**

创新点在于首次为4-ccd簇构建完整的染色码唯一性证明，并引入t‑码、r‑码等新概念以描述簇的转移与结构。

**🔧 技术方法**

主要使用了组合数学与离散几何的推理，结合染色码的值转移与位置转移（TE/TL）进行严谨证明。

**📊 数据集**

研究基于理论推导，不涉及具体实验数据集；所讨论的SCM空间是理论构造的。

**📈 对比分析**

由于研究为纯理论证明，没有对比实验或性能评估；若有，则仅通过数学逻辑验证其正确性。

**⚠️ 局限性**

局限性包括仅证明了4-ccd簇的唯一性，5-ccd及更大簇的唯一性仍未正式证明，且证明过程复杂且难以扩展到更高维空间。

---

## 669. Channel Selection for Mixture-of-Experts Decoding Needs No Calibration Data

**arXiv ID:** 2610.37386 | [PDF](https://arxiv.org/pdf/2610.37386v1)

**作者:** Geunhyoung Kim `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `Machine Learning` `Computational Efficiency` `Representation Learning` `Knowledge Distillation` `Transformer` `Mixture of Experts` `Mixture of Experts` `Large Language Model` `Auto Encoder` `Prompt Engineering` `Retrieval-Augmented Generation` `Text`

**🎯 论文内容**

本文提出了一种无需校准数据即可对Mixture-of-Experts模型进行通道选择的方法，并实现了一款仅基于模型权重的读取器（Reader），实现了显著的读数压缩与推理加速。

**💡 创新点**

创新点在于：①证明传统的阈值校准既不能保证读数预算，也不能持续提升精度；②提出基于top‑k的无校准通道选择；③设计了利用旋转三元化Sketch和低秩恢复的权重仅读读取器，实现约17.8%读数的同时保持98.1%精度。

**🔧 技术方法**

技术手段包括：旋转Hadamard变换后的三元化权重Sketch、精确的top‑k通道选择、低秩下投影恢复、基于GPU 32字节访存计费、Triton自定义核以及vLLM框架集成。

**📊 数据集**

使用的数据集包括GPQA‑Diamond、IFEval、MMLU‑Pro（生成任务）以及OLMoE、UltraChat（校准与对比实验）。

**📈 对比分析**

在三大后训练MoE模型（Qwen3‑Thinking、Gemma‑4、Qwen3.5）与九个基准上，Reader在12.8%/17.8%读数下分别保持94.5%/98.1%完整模型精度，并使MoE块速度提升1.6–2.7×、整体模型速度提升1.15–1.88×，优于Prox、TEAL、WINA等传统阈值或top‑k选择器。

**⚠️ 局限性**

局限性包括：仅针对解码阶段的通道选择；读数加速受GPU架构与预算影响；部分优化仅在单一预算与单一GPU上调优；对大批量处理的通道选择尚未实现；在极端稀疏或量化模型上仍需进一步验证。

---

## 670. PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations

**arXiv ID:** 2610.37381 | [PDF](https://arxiv.org/pdf/2610.37381v1)

**作者:** Hanqing Wang `[一作]` (JD.com), Youzheng Wu `[通讯]` (JD.com)

**关键词:** `Robotics` `Robotic Intelligence` `Representation Learning` `Computational Efficiency` `Transformer` `Vision-Language-Action Model` `Vision Language Model` `Diffusion model` `World Model` `Optical Flow` `Contrastive Learning` `Flow-based Model` `Video` `Image` `Text` `Multimodality`

**🎯 论文内容**

设计并训练了一种轻量化的Vision–Language–Action模型，利用物理动态模块从视觉语言特征中提取物理标记，并通过直接未来特征预测与预训练视频世界模型辅助训练，使得机器人能够基于当前观测和指令生成控制动作。

**💡 创新点**

创新点在于：1）引入物理动态模块，用固定长度的物理标记补充VLM特征；2）在训练时结合直接RGB/深度/光流预测与冻结视频世界模型的流匹配监督，提升物理表示学习；3）全程使用公开数据集并通过人机合成、子任务标注与ECoT注解扩充数据，支持跨体型与跨语言的通用策略。

**🔧 技术方法**

使用的技术包括：Qwen3.5-4B视觉语言模型、轻量化Transformer物理动态模块、Diffusion Transformer行动专家、条件流匹配训练、预训练视频模型LingBot-Video-Dense-1.3B做为辅助监督、数据预处理与动作映射统一框架、以及人机合成与ECoT自然语言注解。

**📊 数据集**

数据集涵盖：① 公开的真实机器人轨迹（RoboSet、DROID、BridgeData等）、② 手持UMI数据（UMI、FastUMI等）、③ 仿真数据（ManiSkill2、Meta-World、InternData-A1等），以及 ②、③ 之外的通用视觉语言数据（COCO-Caption、VQAv2）、空间与具身视觉语言数据（SpaceThinker、RoboVQA 等），并通过人机合成、子任务标注和ECoT注解进一步扩充。

**📈 对比分析**

在四个仿真基准上进行对比：LIBERO 99.3%/89.9%、RoboTwin 2.0 93.6%、SimplerEnv 81.5%，均优于或接近当前最先进的 VLA 方法（如 Qwen-RobotManip、InternVLA-A1.5 等）。实验还验证了世界模型辅助监督提升未来特征预测精度，子任务与ECoT注解对性能的细微影响。

**⚠️ 局限性**

局限性：① 仅利用 RGB、深度和光流作为未来预测目标，缺乏更细粒度的物理量（如接触状态、材质等）；② 训练与评估多在仿真环境，真实世界部署的鲁棒性尚待进一步验证；③ 物理动态模块虽轻量，但仍需要较大 VLM 参数；④ 目前不支持在线视频生成，若需要更长时域预测需重新设计；⑤ 依赖大量公开数据，若数据分布与目标任务差异大，性能可能下降。

---

## 671. TRIDENT-Seq: Profile-Matched Prefix-Future Contrast for Label-Efficient Payment Risk Modeling

**arXiv ID:** 2610.37365 | [PDF](https://arxiv.org/pdf/2610.37365v1)

**作者:** Weiyu Han `[一作]` (Ant International), Tianyi Zhang `[通讯]` (Ant International)

**关键词:** `Machine Learning` `Anomaly Detection` `Federated Learning` `Data-Centric Learning` `Recommendation System` `Finance Related` `Transformer` `Contrastive Learning` `Supervised Fine-Tuning` `Sequential` `Tabular` `Finance Related`

**🎯 论文内容**

提出了一种基于Prefix–Future Contrast（PFC）的交易序列预训练模型TRIDENT-Seq，用来为多种在线支付业务构建共享的风险评估编码器，并在少量标签情况下实现高效迁移与自适应。

**💡 创新点**

创新点在于：①引入了活动匹配的未来摘要对比损失，避免了传统对比学习中因交易量差异导致的shortcut；②设计了基于统计摘要的未来表示与两项辅助任务（活动预测与分散正则化）进一步强化编码器的风险区分能力；③实现了在多业务场景下单一预训练模型的高效跨域迁移与低标注适配。

**🔧 技术方法**

使用的技术包括：Transformer编码器；Bloom-filter式哈希编码实现无词表的属性表示；Prefix–Future Contrast对比学习；活动预测（回归）与分散正则化；以及在业务层面采用类别平衡交叉熵进行微调。

**📊 数据集**

使用的数据集来自Ant International的三类支付业务（Wallet、Shopping、SME），包含数千万条未标注交易历史以及相应的标注风险标签（最多数十万条）。

**📈 对比分析**

与多种基线（scratch、CoLES、BYB、CPC、TransactionGPT、PANTHER、XGBoost等）进行对比，评估指标为AP与0.5%预算下的Recall。TRIDENT-Seq在所有业务和标签预算下均显著优于基线，尤其在Wallet业务的10%标签下提升约24倍Recall，在SME业务的全标签下提升约1.8倍Recall，并在A/B测试中实现约38% chargeback 与损失率下降。

**⚠️ 局限性**

限制包括：①对历史极度稀疏的业务（如SME）仍存在性能波动；②编码器在不同业务/预算下的训练稳定性差异较大；③缺乏对模型内部机制的直接可解释性与因果验证；④模型规模和预训练时间较大，需在更大数据/模型上验证可扩展性。

---

## 672. Rendering-Free Event Frame Simulation for Fast Reinforcement Learning of Autonomous Robot Navigation

**arXiv ID:** 2610.37543 | [PDF](https://arxiv.org/pdf/2610.37543v1)

**作者:** Dequan Ou `[一作]` (Delft University of Technology), Guido C. H. E. de Croon `[通讯]` (Delft University of Technology)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Autonomous Driving` `Spiking Neural Network` `Transformer` `Reinforcement Learning` `Optical Flow` `Contrastive Learning` `Domain Randomization` `Quantization` `PPO` `Generative Adversarial Network` `Point Cloud` `Image` `Video`

**🎯 论文内容**

开发了一种无需渲染的解析式事件模拟器，直接从场景几何和自机运动生成事件帧，结合强化学习训练无人机障碍规避策略；

**💡 创新点**

创新点在于：①完全跳过光照渲染，采用基于移动对比度边缘的解析式事件合成，速度提升≈400×；②实现了零样本从模拟到真实的迁移；③将量化后的int8网络部署到56 g的Crazyflie上，实现全系统低功耗、无外部定位；

**🔧 技术方法**

技术包括解析式事件生成（点云边缘投影+阈值化）、对比度与饱和度校准、PPO强化学习、域随机化、int8量化、STM32N6 NPU推理；

**📊 数据集**

使用的是自建的程序化障碍场景（树杆、屏幕、管道、植物等），以及少量真实GenX320传感器的短时记录用于校准；

**📈 对比分析**

与基准相比：与使用真实深度信息的深度代理（100%成功）相比，事件代理在未见的渲染环境中取得83%/70%成功率；对比单一GPU日渲染事件训练（约40%/53%）显著提升；真实飞行中取得100%/83%/67%的成功率；

**⚠️ 局限性**

局限性包括：依赖静态几何的模拟，难以处理高速动态障碍或高纹理变化；对光照或视角的极端变化适应性有限；目前仅验证了单一小型无人机，扩展到更复杂任务和更大平台仍需进一步研究。

---

## 673. From Interaction to Capability: Task Advancement as State and Supervision for GUI Agents

**arXiv ID:** 2610.37482 | [PDF](https://arxiv.org/pdf/2610.37482v1)

**作者:** Ziyang Chen `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `Artificial Intelligence` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Robotic Intelligence` `Transformer` `Large Language Model` `Vision Language Model` `Vision-Language-Action Model` `Agentic AI` `Retrieval-Augmented Generation` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出了Task Advancement Memory（TAM），通过记录已验证的任务进展来维护跨应用的GUI记忆，并将其用于在线任务执行和可复用操作的学习。

**💡 创新点**

创新点在于：①用任务进展定义任务状态并实现任务削减；②采用进展监督机制将局部成功的交互抽象为可重用的操作；③将状态信息与可复用操作分离，支持界面变更时的适配。

**🔧 技术方法**

技术手段包括：LLM驱动的任务编译与状态管理；基于观察的进展验证与Evidence Cache；进展监督的操作图学习；Qwen3‑VL‑32B‑Instruct与GPT‑4o等大模型作为策略与评估器。

**📊 数据集**

使用的公开数据集为MobileWorld GUI‑only子集和SPA‑Bench English Level 3子集，涵盖跨应用长周期工作流与单应用难度最高级别任务。

**📈 对比分析**

与AWM、Mobile‑Agent‑E、GUI‑explorer、StructAgent、MemGUI‑Agent等基线对比，在MobileWorld上成功率从29.1%提升至34.2%（+5.1pp），在SPA‑Bench上从53.7%提升至57.0%（+3.3pp）；同时平均控制器轮数下降、生命周期Token成本降低。

**⚠️ 局限性**

局限性：依赖模型对GUI证据的解析，易受模糊输出影响；评估仅覆盖移动GUI任务，未验证对Web/桌面等界面；持续任务流和应用更新时的操作适配与衰减尚未充分研究。

---

## 674. PerceptRL: Making Visual Perception Rewardable in MLLMs from Answers to Interactions

**arXiv ID:** 2610.37471 | [PDF](https://arxiv.org/pdf/2610.37471v1)

**作者:** Tongkun Guan `[一作]` (Shanghai Jiao Tong University), Xiaokang Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `Computer Vision and Pattern Recognition` `Reinforcement Learning from Human Feedback` `Transformer` `Vision Language Model` `Reinforcement Learning` `Agentic AI` `Prompt Engineering` `Image` `Text` `Multimodality`

**🎯 论文内容**

在多模态大语言模型的后训练阶段，利用模型自身的感知错误构建可验证的单轮问答和多轮工具交互数据，分别通过 Perception RL 和 Perception Agentic RL 两级强化学习提升视觉感知能力。

**💡 创新点**

创新点在于：①将开放式图像字幕中的感知错误转化为可验证的 QA 对，提供规则化奖励；②通过坐标注入（Coordinate Injection）纠正工具交互中的空间漂移，并在多轮 RL 中引入基于空间精度的过程奖励，实现对中间定位的监督。

**🔧 技术方法**

技术主要包括：多代理管线（自动检测和验证字幕错误）、规则化奖励策略、基于 DAPO 的强化学习、坐标注入与定位校正、过程奖励机制、代码解释器工具交互。

**📊 数据集**

使用 Qwen3.6‑Flash（35A3）作为基准模型，在 8 个视觉感知基准（RealWorldQA、MMStar、BLINK、TreeBench、ERQA、RefSpatial、CountQA、RealWorldCounting）上评测，并从原始图像描述中挖掘训练数据。

**📈 对比分析**

与 10+ 领先 VLM（Gemini‑3‑Flash、Qwen3.5‑Plus 等）在共享工具接口下比较，最终 Perception Agentic RL 模型在 7 个基准上平均精度提升至 71.6%，比基准模型高 15.7 个百分点、比最强对手高 5.2 个百分点。

**⚠️ 局限性**

局限性包括：仍以静态图像为主，未覆盖视频或时间一致性；对定位模型的依赖需手工选择，未实现端到端联合学习；生成的数据仍受原始模型错误分布影响，可能在超出训练域的任务上泛化受限。

---

## 675. M4Bench: Evaluating Procedural Specification for Clinical EHR Derivation Agents

**arXiv ID:** 2610.37459 | [PDF](https://arxiv.org/pdf/2610.37459v1)

**作者:** Hannes Ill `[一作]` (Massachusetts Institute of Technology), Leo Anthony Celi `[通讯]` (Beth Israel Deaconess Medical Center)

**关键词:** `Artificial Intelligence` `Benchmark` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Supervised Fine-Tuning` `Tabular` `Biomedical Data` `Electronic Health Records`

**🎯 论文内容**

本研究建立了一个可审计、可版本化的临床 EHR 派生变量基准，包含 28 个任务、15 个任务族，并为每个任务提供经临床专家审核的 Markdown “技能”脚本；

**💡 创新点**

创新点在于将临床隐式程序知识抽象为可检索、可复现的技能，并通过多种对照实验（decoy skill、raw SQL、operational spec、schema 变形）拆解技能对 LLM 代码生成的影响；

**🔧 技术方法**

技术手段包括 Docker 隔离执行、SQL 代码生成与调试、Codex/Claude LLM 交互、技能加载与执行、基于列的奖励、严格通过率与 key‑F1 评估、任务级 bootstrap 置信区间等；

**📊 数据集**

数据集为基于 MIMIC‑IV v3.1 与 eICU Collaborative Research Database v2.0 的 DuckDB 克隆，每个任务拥有数千至数百万行的临床数据；

**📈 对比分析**

比较方法是对比无技能（schema+任务）与有技能的条件，计算奖励差异并进行 Bootstrap 置信区间；结果显示 Codex 在 28 个任务上平均提升 +0.136（p<10⁻⁴），Claude 提升 +0.120，主要来源于规格完整性；但关键粒度（key‑F1）提升仅 +0.003，严格通过率仍低；

**⚠️ 局限性**

局限性包括仅覆盖 28 个任务、缺乏跨医院/数据库的泛化、评估仅衡量与公开参考实现的一致性而非临床正确性、操作规格已解释大部分提升导致技能形式影响有限、以及潜在的训练数据记忆化问题。

---

## 676. Test-Time Communication in Trading Agent Swarms

**arXiv ID:** 2610.37448 | [PDF](https://arxiv.org/pdf/2610.37448v1)

**作者:** Avid Fayaz `[一作]` (Infinite Ascent), André Silva `[通讯]` (Infinite Ascent)

**关键词:** `Multiagent Systems` `Finance Related` `Explainability and Interpretability` `Transformer` `Large Language Model` `Prompt Engineering` `Agentic AI` `Text` `Tabular` `Time Series`

**🎯 论文内容**

本文通过让 200 名基于语言模型的交易代理在七个交易日内在不同的“测试时通信”模式下运行，记录并分析它们的交易行为、信息曝光和词汇传播，从而评估同侪交流对交易决策的影响。

**💡 创新点**

创新点在于：①首次在真实 US 市场上实时对多种语言模型交易代理进行测试时通信实验；②采用自然实验（晨间排队顺序）与“无同侪”基线结合，精确区分同侪影响与共同输入的贡献；③系统性记录每个代理在每一次决策前可见的同侪信息，为未来的可解释性和因果推断提供基础。

**🔧 技术方法**

使用的技术包括：大型语言模型（Opus 5.5、GPT-6.1 Sol、Grok 4.6、DeepSeek V4 Pro），自定义的交易循环和工具调用框架，五种通信机制（隔离、协作板、竞争板、带 pings 的板、同侪记忆），以及完整的日志和可追溯的曝光记录系统。

**📊 数据集**

数据集为 7 天的实时 US 股票市场数据（价格、成交量等）以及对应的纸质账户记录；同时收集了所有代理的交易日志、消息内容、研究请求、注记和评分信息，形成了完整的曝光-行动双重记录。

**📈 对比分析**

比较方法：将各通信模式下的账户收益与 SPY 基准及同一模型的隔离组进行对比；通过引入“无同侪”基线和自然实验（晨间排队差异）来衡量同侪词汇传播、决策规则的扩散以及交易同步的效应。结果显示：①通信组在词汇共享上有显著提升；②共享持仓对收益贡献显著；③通信组的卖出交易同步率明显高于隔离组；整体收益低于 SPY，说明同步可能带来集体风险。

**⚠️ 局限性**

局限性：每个模型-通信组合仅有一组，无法对通信模式进行统计显著性排序；实验持续时间仅 7 天，难以评估长期收益和稳定性；评分信息在前两天缺失且对所有组均被标记为“实验性”，可能抑制合作意愿；所有账户为纸质，未考虑真实资本流动及交易成本；实验设计中通信与框架信息捆绑，无法区分渠道效应与说明性信息的影响。

---

## 677. CRISP: Co-designing Pruning and Scheduling for Efficient MoE Inference on Edge Servers

**arXiv ID:** 2610.37442 | [PDF](https://arxiv.org/pdf/2610.37442v1)

**作者:** Congwei Zhang `[一作]` (Shandong University), Dongxiao Yu `[通讯]` (Shandong University)

**关键词:** `Machine Learning` `Computational Efficiency` `Transformer` `Mixture of Experts` `Text`

**🎯 论文内容**

提出CRISP框架，在边缘平台上通过组合感知的剪枝、重要性排序的神经元重排和剪枝感知的I/O调度实现MoE推理的高效执行。

**💡 创新点**

创新点在于将专家组合频率与专家角色的异质性联合考虑，形成分层宽度策略，并通过神经元重排实现单个专家多宽度支持，随后利用宽度异质性实现I/O与计算的重叠。

**🔧 技术方法**

使用基于激活幅度的神经元重要性度量进行重排、按热度划分的Hot/Warm/Cold三层执行策略、宽度感知的异步I/O与计算调度、CUDA实现的分块加载与截断前馈，兼容量化。

**📊 数据集**

数据集包括C4校准集、WikiText-103、C4文本评价、下游任务MMLU、HellaSwag、ARC-Challenge、WinoGrande，模型为Mixtral-8x7B和DeepSeek-MoE-16B。

**📈 对比分析**

与全量加载、均匀剪枝、MoE-Offload、EdgeMoE等基线在相同专家内存预算下对比，CRISP在10%~40%内存预算上实现最高约2.1×吞吐量提升，准确率下降≤3.6%，能耗下降70%+。

**⚠️ 局限性**

局限在于需要离线频率统计与重排，适应快速工作负载变化受限；对极低端设备的共享内存/缓存情况未作深入评估；同时假设路由分布不变，实际部署中需频繁更新配置。

---

## 678. Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning

**arXiv ID:** 2610.37437 | [PDF](https://arxiv.org/pdf/2610.37437v1)

**作者:** Zengqi Zhao `[一作]` (University Of North Carolina), Yue Zhang `[通讯]` (University Of North Carolina)

**关键词:** `Computer Vision and Pattern Recognition` `Representation Learning` `Computational Efficiency` `Explainability and Interpretability` `Reinforcement Learning from Human Feedback` `Transformer` `World Model` `Vision Language Model` `Prompt Engineering` `Mixture of Experts` `Retrieval-Augmented Generation` `Contrastive Learning` `Generative Adversarial Network` `Auto Encoder` `Image` `Multimodality` `Point Cloud` `Mesh` `Benchmark`

**🎯 论文内容**

构建了一个基于模拟器的视觉空间想象基准，提供同一未观测状态的RGB、深度、正交视图、全景、顶视图等多种表示，并在此基准上评估多种空间推理模型；

**💡 创新点**

证明了准确的RGB图像并不总能提升空间推理，提出不同表示可互补提升推理性能，并基于此提出了轻量级的自适应表示选择方法GEARS；

**🔧 技术方法**

采用Matterport3D 3D场景、HY‑World 2.0和SEVA等世界模型生成想象状态，使用GPT‑6.1‑sol和Gemini 3.7 Flash等多模态推理器，结合门控与路由机制实现GEARS；

**📊 数据集**

使用Matterport3D数据集，构造了922道手工标注的空间推理问题，涵盖视角想象和跨视角整合两种设置；

**📈 对比分析**

通过对oracle与估计表示的对比实验，发现oracle RGB提升整体性能但对部分问题下降，其他表示可补偿错误；GEARS在Viewpoint想象和Cross‑view整合上分别达到约61%和78%的准确率，优于MindJourney和AVIC；

**⚠️ 局限性**

局限在于仅测试静态室内场景、使用有限的世界模型、未覆盖所有表示与更复杂交互，且与oracle结果相比仍有显著差距。

---

## 679. Kernel Performance Engineering under Physical Non-Idealities

**arXiv ID:** 2610.37466 | [PDF](https://arxiv.org/pdf/2610.37466v1)

**作者:** Supun Kuruppu `[一作]` (National University of Singapore), Tulika Mitra `[通讯]` (National University of Singapore)

**关键词:** `Hardware Architecture` `Optimization` `Tabular` `Time Series` `Benchmark` `Physics Related`

**🎯 论文内容**

研究并量化了 NVIDIA H100 GPU 在 dsmem（分布式共享内存）功能下，由制造缺陷（floor sweeping）和 TDP 限制导致的物理非理想性对 kernel 性能的影响，并基于此提出波平衡（wave‑equalized）和波感知聚类（wave‑aware clustering）两种优化策略。

**💡 创新点**

创新点在于：①首次将 floor sweeping 的虚拟 GPC 组织对执行波数的影响建模；②提出波平衡的条件公式，避免因 active SM 限制导致的额外执行波；③设计波感知聚类策略，根据执行波数动态选择 TPC 或 GPC 聚类，提升 FlashAttention 4 的吞吐率最高 5.30%；④通过 TDP 与 DVFS 的关联分析，解释在功耗极限下数据搬运减少如何提升 SM 频率并进一步提升性能。

**🔧 技术方法**

使用的技术包括：NVIDIA H100 上的 dsmem 微基准、GEMM 与 FlashAttention 4 的实际 kernel ；对 SM ID 采样、聚类分析实现 virtual GPC（skyline）反演；利用 ncu 进行 SM 频率、张量核心利用率、L2 读取等指标采集；以及对 Tenstorrent Wormhole accelerator 的 TDP‑频率关系做实验。

**📊 数据集**

使用的“数据集”是人工合成的工作负载：① 256 TB 的 1 MB 随机 FP32 流；② GEMM 的多种矩阵尺寸（如 8192×4096×8192、7680×4096×512 等）；③ FlashAttention 4 的多种序列长度（2 K–64 K）、批量大小、头数和 tile 组合（B_M∈{128,64,32}，B_N∈{80,64,32}）；⑤ Tenstorrent Wormhole 的矩阵乘法在不同核心数与精度模式下。

**📈 对比分析**

比较方法：以 cluster size 1（无聚类）为基线，分别测评 cluster size 2（TPC聚类）、4、8（GPC聚类）以及波感知聚类。性能指标主要是吞吐率（TFlop/s 或 ops/s）与 SM 频率。实验表明：① 在满足波平衡条件时，GPC 聚类可提升吞吐率（最高 7.32%）；② 波感知聚类相较于基线可提升 5.30%；③ 对于低头维度，GPC 聚类甚至会降低性能；④ 在 TDP 限制下，SM 频率随聚类大小线性提升，吞吐率与频率呈高相关性；⑤ Tenstorrent Wormhole 的吞吐率随核心数线性上升至功耗阈值后出现明显衰减。

**⚠️ 局限性**

局限性：① 波平衡条件仅适用于特定的 active SM 限制（H100 为 132 与 120），对其他 GPU 或不同工艺需重新测定；② 仅在单核/单张卡上验证，未探讨多 GPU/多卡分布式场景；③ 只考虑了 dsmem 与 TDP，未覆盖其他物理非理想性如电压波动、热失效等；④ 波感知聚类的实现需要手工分析线程块数，自动化工具仍待完善。

---

## 680. Strategyproof Deep Reinforcement Learning for Online Resource Allocation

**arXiv ID:** 2610.37504 | [PDF](https://arxiv.org/pdf/2610.37504v1)

**作者:** Fariba Dehghan `[一作]` (University of Southampton), Stephanie Gauthier `[通讯]` (University of Southampton)

**关键词:** `Multiagent Systems` `Reinforcement Learning` `Optimization` `Transformer` `Other` `Reinforcement Learning` `Supervised Fine-Tuning` `Interval Arithmetic` `Tabular` `Time Series` `Benchmark`

**🎯 论文内容**

本文提出了一种基于光滑最小-最大（Smooth Min–Max）网络的策略可证明（strategyproof）在线资源分配机制（SP‑SMM），能够在未知未来请求的情况下自适应地决定是否接受任务并确定支付；

**💡 创新点**

创新点在于：① 用结构化的SMM网络保证分配规则在价值、时长和需求上单调，从而在任何学习阶段都保持可证明的诚实性；② 通过对SMM网络的数值验证搜索实现了可计算的临界支付（critical payment），即使网络没有显式闭式逆；③ 允许使用强化学习或模仿学习训练同一架构，显著提升训练效率；

**🔧 技术方法**

主要技术包括：构造正权重的SMM网络、基于区间算术的精确阈值搜索、基于SARSA的拟合批训练、以及在训练阶段的线性与非线性网络融合；

**📊 数据集**

实验数据集为若干人工合成的资源分配流，包括“Partner”（四资源两对伙伴任务）、“Small”“Large”“Continuous”“Complementary”等，覆盖单资源、双资源及高密度混合需求场景；

**📈 对比分析**

与线性阈值规则、其他单调神经网络（LipschitzMono、SMNN、MonoDense、MonoNet、PMLinear）以及基于网格的Lattice方法比较，SP‑SMM 在 Partner 流中取得 97.8% 的离线最优收益，且在大多数流上均处于前列；模仿学习实现了 89 倍速度提升，RL 训练时间大幅增长；

**⚠️ 局限性**

局限性包括：假设任务的到达时间可观测，单次交互模型；未处理任务可延迟或多次交互的情况；以及对更大规模多资源、连续需求分布的可扩展性仍需进一步验证。

---

## 681. Computing GR(1) Assume-Guarantee Contracts

**arXiv ID:** 2610.37534 | [PDF](https://arxiv.org/pdf/2610.37534v1)

**作者:** Yigit Narter `[一作]` (University of Michigan), Inigo Incer `[通讯]` (University of Michigan)

**关键词:** `Logic in Computer Science` `Autonomous Driving` `Optimization` `Computational Efficiency` `Federated Learning` `Safty and Privacy` `SMT solving` `Symbolic contract algebra` `Quantifier elimination` `Tabular` `Time Series` `Sequential` `Benchmark`

**🎯 论文内容**

开发了基于GR(1)假设–保证合同的上下文辅助变量消除（CAVE）算法，实现了合同运算（组合、商、合并）在Pacti平台上的自动化，并在自动紧急制动（AEB）案例中验证。

**💡 创新点**

提出在GR(1)规范中使用有限窗口上下文辅助变量消除方法，保证结果保持GR(1)结构并在子问题内实现最优性，同时将该方法集成至Pacti实现。

**🔧 技术方法**

采用有限窗口展开、SMT求解的量化消除、GR(1)结构分解与符号合同代数运算，并与Spot等Büchi自动机工具进行对比实验。

**📊 数据集**

使用合成的GR(1)规范家族以及自动紧急制动（AEB）案例的真实参数作为实验数据集。

**📈 对比分析**

与Spot基于自动机的精确消除对比，CAVE在变量数较多时更快；在liveness约束增多时Spot性能急剧下降；实验显示CAVE的运行时间随变量数呈指数增长，但对liveness约束增长线性。

**⚠️ 局限性**

仅支持有限next深度的GR(1)规范，无法处理跨步依赖或无限时序约束；产生的结果仅为保守近似，可能不等价于精确消除。

---

## 682. Which Hand Was Late? Diagnosis for Robot Failure Recovery in Bimanual Tool Use

**arXiv ID:** 2610.37533 | [PDF](https://arxiv.org/pdf/2610.37533v1)

**作者:** Baptiste Bonin `[一作]` (Université Laval), Audrey Durand `[通讯]` (Université Laval)

**关键词:** `Robotics` `Robotic Intelligence` `Explainability and Interpretability` `Anomaly Detection` `Vision Language Model` `Transformer` `Vision Language Model` `Prompt Engineering` `Supervised Fine-Tuning` `Contrastive Learning` `Linear Probing` `Image` `Time Series` `Multimodality` `Tabular` `Benchmark`

**🎯 论文内容**

本文在模拟的双手工具使用任务（锁开锁）中，构建了具有可读失效原因的实验环境，并对各传感通道及视觉语言模型（VLM）的内部表征进行线性探针，评估其对失效原因的可诊断性及其对修复策略的影响。

**💡 创新点**

创新点包括：
1) 引入可根据记录轨迹直接读取失效原因的双手工具使用基准；
2) 系统性探测单手、双手和无传感器场景下的失效原因可识别性；
3) 发现工具滑出（cam‑out）只在持工具手上可检测，双手迟到的推力弱度在工具手迟到时无法通过被动传感器判断；
4) 证明 VLM 的内部状态确实编码了哪只手迟到，但其回答依赖于提问表述。

**🔧 技术方法**

技术手段：
- 通过脚本控制器在仿真中注入单一失效，并记录机械轨迹；
- 对多种传感器（手腕扭矩、接触、关节、力、图像、关节姿态、命令等）以及 VLM 的冻结特征进行线性探针；
- 使用 Qwen3‑VL 8B 进行视觉语言推理并提问；
- 评估修复策略的成功率。

**📊 数据集**

数据集：758 次实验，涉及 17 只门，共计 174 次成功、198 次错误（106 次 cam‑out）、133 次时序失效、134 次力度失效、119 次双重失效。每次实验的记录轨迹可供后续解析。

**📈 对比分析**

比较方法：以每个门内的 AUC 作为评估指标，计算各传感通道在不同诊断任务（检测、诊断、时序、力度、工具滑出、双重失效）下的性能。结果显示：
- 单手工具滑出可由手腕扭矩/接触/关节极好检测（AUC≈0.97）；
- 双手迟到需跨手传感器，最优的组合是双手姿态（AUC≈1.00）或双手命令；
- 双重失效在工具手迟到时无可被动传感器能辨识，AUC≈0.5；
- 通过问答与线性探针的对比，VLM 内部状态对迟到手具有高可读性（AUC≈0.96），但回答质量受提问措辞影响。

**⚠️ 局限性**

局限性：
- 仅在仿真环境中验证，真实机器人可能面临更复杂的延迟分布；
- 失效类型有限，仅涵盖锁开门任务，无法直接推广到其他双手工具使用情境；
- 仅评估被动传感器，未探究主动诊断或多模态融合的潜力；
- VLM 的回答依赖于提问表述，说明对模型解释性仍有不足。

---

## 683. GTFormer: A Syntax-Free Graph Network Alternative to Transformers for 8192$\times$ Length Extrapolation in Linear Time

**arXiv ID:** 2610.37524 | [PDF](https://arxiv.org/pdf/2610.37524v1)

**作者:** Elisabeth Fink `[一作]` (Middlesex University), Elisabeth Fink `[通讯]` (Middlesex University)

**关键词:** `Machine Learning` `Computational Efficiency` `Representation Learning` `AI Code Assistant` `Graph Neural Network` `Transformer` `Large Language Model` `Contrastive Learning` `Prompt Engineering` `Text` `Sequential`

**🎯 论文内容**

开发了一种名为GTFormer的层次图神经网络，用语法无关的方式替代传统的全连接自注意力，实现对极长上下文序列（最高可达4,194,304个token）的高效建模。

**💡 创新点**

创新点在于将平衡二叉树结构与水平shift边、随机expander短路、凸高通门以及顶层广播相结合，构成低度图拓扑；凸门通过通道级凸组合保持特征在凸包内，避免深层漂移；expander提供跨树子结构的短路，显著提升信息传播效率。

**🔧 技术方法**

技术包括图神经网络消息传递、双层GELU MLP、水平shift编码、随机expander短路、通道级凸高通门、顶层广播乘法融合、双头读取以及混合损失（联合分类、槽位、变量/指针监督），并用AdamW + cosine学习率调度训练。

**📊 数据集**

使用自定义的八个合成任务数据集，词表大小320，序列长度从16到4,194,304，涵盖多值覆盖、非对齐键值、两跳指针、状态机、作用域、检索增益、XML/括号校验以及对抗性语法混淆等场景。

**📈 对比分析**

与平面Transformer（Sinusoidal、RoPE）、纯二叉树、无expander、无凸门等架构对比。GTFormer在仅用N≤512的训练序列后，零射精到N=4,194,304时，准确率达98%+，显著优于Transformer（仅能到4,096并出现OOM），并且显存与计算复杂度保持线性（O(N)）。

**⚠️ 局限性**

局限性包括：对二叉树划分的依赖导致某些奇数对齐问题；需要多通道隐藏空间，参数量相对较大；验证范围仅限合成任务，未在真实大型文本或代码语料上测试；计算推理时延高于简易Transformer；对某些任务（如多跳检索）仍有细微性能下降。

---

## 684. Leveraging Distributed Sensing and Hyperdimensional Computing for Non-Line-of-Sight Passive Object Sensing

**arXiv ID:** 2610.37519 | [PDF](https://arxiv.org/pdf/2610.37519v1)

**作者:** Placido Mursia `[一作]` (NEC Laboratories Europe GmbH), Xavier Costa-Perez `[通讯]` (NEC Laboratories Europe GmbH)

**关键词:** `Robotics` `Anomaly Detection` `Autonomous Driving` `Computational Efficiency` `Robotic Intelligence` `Safety and Privacy` `Contrastive Learning` `Diffusion model` `Score-based Model` `Simultaneous Localization and Mapping` `Stochastic Differential Equation` `Point Cloud` `Tabular` `Time Series` `Benchmark`

**🎯 论文内容**

提出了一种基于通信基础设施和高维向量计算（HDC）的分布式被动实体感知框架，能够检测机器人视线之外的危险区域，并将信息嵌入ROS导航的全局代价图中实现路径规划辅助。

**💡 创新点**

创新点包括：1) 利用多接收器收集的无线信道测量信息进行NLOS感知；2) 采用低复杂度的HDC编码实现无需显式信道重建的分类；3) 通过多接收器协同融合提升鲁棒性；4) 在ROS导航中直接修改全局代价图，将潜在危险信息提前加入规划过程。

**🔧 技术方法**

使用技术包括：6G ISAC概念、30 GHz多天线系统、无线射线追踪仿真（Wireless InSite）、HDC（高维向量绑定、束缚、置换与相似度计算）、迭代重训练、ROS2导航堆栈以及STM32 MCU性能评估。

**📊 数据集**

使用的数据集来自射线追踪仿真，生成10,000条CIR样本，按80/20划分训练/测试，样本按目标坐标被标记为安全区或危险区。

**📈 对比分析**

与传统基于原始CIR的深度神经网络（8层、约465k参数）对比，HDC连续映射平均准确率为88.98%（比DNN高11个百分点），二进制映射为82.21%仍优于DNN；多接收器联合融合进一步提升准确率至约89%，危险区F1约84.94%；MCU上二进制HDC操作时间均在200 µs以内，显示出显著的低延迟优势。

**⚠️ 局限性**

局限性：仅完成二分类（安全/危险）研究，未实现精确定位或多类风险建模；假设CIR估计完美，未考虑估计误差；对动态人/物体实时变化的适应性待验证；安全性仅为辅助感知，需与现有传感器协同使用；未评估环境变化对信道噪声的影响。

---

## 685. ALADDIN: Adversarial Latent Diffusion Distillation for One-Step Identity Swapping

**arXiv ID:** 2610.37503 | [PDF](https://arxiv.org/pdf/2610.37503v1)

**作者:** Ivan Shchekotov `[一作]` (Constructor University), Dmitry Vetrov `[通讯]` (Constructor University)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Image Translation` `Knowledge Distillation` `Transformer` `Convolutional Neural Network` `Diffusion model` `Generative Adversarial Network` `Contrastive Learning` `Score-based Model` `Flow-based Model` `Auto Encoder` `Image`

**🎯 论文内容**

提出了一种一步式人脸交换模型 ALADDIN，利用预训练扩散模型初始化生成器并进行对抗训练，实现源身份迁移与目标属性保持的平衡。

**💡 创新点**

创新点在于将扩散模型的先验直接注入生成器和判别器，利用冻结的扩散特征进行对抗训练，无需额外蒸馏；并通过三元组伪标签实现可控的身份与属性平衡。

**🔧 技术方法**

使用了流匹配扩散模型（FLUX）、PuLID 身份编码器、OminiControl 属性分支、冻结扩散模型特征的判别器、LoRA 微调、LPIPS、ArcFace 等技术。

**📊 数据集**

在 VGGFace2-HQ 和 Arc2Face 上训练，评估基于 FFHQ 1,000 对源-目标图像。

**📈 对比分析**

与多种一步式和多步式基线对比，ALADDIN 在 FID、身份相似度、姿态与表情等四项指标上均优于所有一步式模型，并在多步式 APPLE 模型相同或更好，同时仅需一次网络评估，训练数据量大幅降低。

**⚠️ 局限性**

限制包括在极端年龄差、异常身份或大姿态变换下身份迁移效果下降，部分方法在目标属性保持（如纹身）时仍有缺失，并受扩散模型分辨率等限制。

---

## 686. When POST Becomes GET: Investigating HTTP Method Interchangeability and Its Security Implications

**arXiv ID:** 2610.37474 | [PDF](https://arxiv.org/pdf/2610.37474v1)

**作者:** Matteo Golinelli `[一作]` (University of Trento), Bruno Crispo `[通讯]` (University of Trento)

**关键词:** `Cryptography and Security` `Security` `Web Application Analysis` `HTTP Method Interchangeability` `Browser Automation` `Request Rewriting` `Response Similarity Analysis` `Large-Scale Measurement` `Web Traffic Data` `HTTP Requests` `Structured Responses` `Review/Survey Paper` `Benchmark` `Security and Privacy`

**🎯 论文内容**

论文通过浏览器自动化收集 POST 请求并改写为 GET，随后比较响应以测量 Web 应用中 HTTP 方法可互换性。

**💡 创新点**

主要创新是提出基于浏览器的 MIDE 流水线，使用无查询 GET 基线消除泛型响应误判，并首次大规模测量方法可互换性。

**🔧 技术方法**

技术包括 Playwright 浏览器自动化、请求记录与重放、基于查询字符串的 POST→GET 转换以及结构化响应相似度比较。

**📊 数据集**

数据集为 Tranco Top 10k 网站（10,000 个域），共捕获 21,280 个 POST，重新发送 14,244 个。

**📈 对比分析**

与传统基线方法相比，MIDE 在 1,228 个高置信度实例中发现 4.1% 的站点可互换，平均每站点 10‑page 深度、约 5 秒间隔，成功重放率约 98%。

**⚠️ 局限性**

限制包括未鉴权、表单输入有限、仅支持 3 种正文格式、URL 长度/查询限制，导致对可互换性的估计偏保守且不覆盖后端仅登录后可访问的路径。

---

## 687. How Reliable Is Individualized Functional-Network Attribution in Resting-State fMRI Cognitive Prediction in ADNI?

**arXiv ID:** 2610.37472 | [PDF](https://arxiv.org/pdf/2610.37472v1)

**作者:** Sheethal Bhat `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Andreas Maier `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `Computer Vision and Pattern Recognition` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Multi-Layer Perceptron` `Bootstrap CI` `Permutation Test` `Spearman Correlation` `Network Ablation` `Biomedical Data` `Alzheimer's Disease` `Magnetic Resonance Imaging`

**🎯 论文内容**

评估了在ADNI数据中基于静息态功能连接的多层感知器预测模型对个体功能网络归因的可靠性，并通过分割半、扫描-重扫描、置换检验三种方法测量其测试-重测一致性。

**💡 创新点**

首次将个体网络归因的可靠性与扫描-重扫描数据和人群置换基线相结合，揭示归因信号既有共享结构又有微小个体特异性，并给出报告建议。

**🔧 技术方法**

采用冻结的Schaefer-200功能连接输入的多层感知器模型，配合网络消融法计算每个功能网络对预测的贡献，并用Spearman相关、Bootstrap CI和置换检验评估可靠性。

**📊 数据集**

以阿尔茨海默病神经影像取样计划（ADNI）中的875名受试者的T1和rs-fMRI数据为基础，重点使用132名测试集受试者的扫描-重扫描对。

**📈 对比分析**

与传统的同会话拆半比较，拆半相关系数接近0.95，但扫描-重扫描仅为约0.61；置换检验表明匹配对的相关系数显著高于随机匹配，提示存在约0.15的个体特异性。

**⚠️ 局限性**

样本量有限（74名受试者/205对），预测模型性能弱且对不同协议的适配有限，且扫描-重扫描未进行ComBat校正，间隔时间长且未建模。

---

## 688. Who Shaped What? Untangling Human and AI Creative Contributions Through Spatial Provenance in Image Editing

**arXiv ID:** 2610.37469 | [PDF](https://arxiv.org/pdf/2610.37469v1)

**作者:** Riya Sahni `[一作]` (Columbia University), Tongyu Zhou `[通讯]` (Adobe Research)

**关键词:** `Human-Computer Interaction` `Explainability and Interpretability` `Transformer` `Large Language Model` `Vision Language Model` `Prompt Engineering` `Diffusion model` `Image`

**🎯 论文内容**

研究人机协作图像编辑中人类与AI的创作贡献辨别，通过设计一个混合式交互工具Roop记录空间来源并以热图展示编辑历史。

**💡 创新点**

创新点在于将编辑日志与图像区域关联，并用空间热图帮助创作者对比自身与AI的贡献，揭示记录与主观归因之间的差异。

**🔧 技术方法**

使用 GPT‑4o、SAM3、Google Gemini 2.5 Flash Image（Nano Banana）等模型，配合自定义热图生成算法。

**📊 数据集**

实验使用两张示例图像和预设情绪目标列表，未使用公开大规模数据集。

**📈 对比分析**

通过20名参与者的对内设计比较用户主动与AI建议两种工作流程下的自我归因评分，并用 Wilcoxon 符号秩检验发现 AI 建议时自我归因显著降低；热图与手工标注的匹配度被定性评估，存在一定程度的一致性与误差。

**⚠️ 局限性**

局限性包括样本规模小、任务短暂、仅使用预设图像/目标、热图仅反映交互强度而非创意价值、未同步全局调整、无法验证参与者先前意图。

---

## 689. Penny for Your Prompts: A Measurement Study of Contextual Leakage in AI Chat Platforms

**arXiv ID:** 2610.37467 | [PDF](https://arxiv.org/pdf/2610.37467v1)

**作者:** Paschalis Bekos `[一作]` (Stony Brook University), Michalis Polychronakis `[通讯]` (Stony Brook University)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Prompt Engineering` `Large Language Model` `Text`

**🎯 论文内容**

审计了42个AI聊天平台的Web界面，通过系统测量和可视化，发现了上下文信息和身份信息被第三方跟踪器泄露的现象。

**💡 创新点**

创新点在于首次将网络层捕获与可见V8引擎级别的JS执行追踪相结合，能够精确归因泄露事件到具体用户交互，并揭示跟踪器如何从页面结构中提取对话内容。

**🔧 技术方法**

使用技术包括Puppeteer进行网络流量捕获、VisibleV8（VV8）进行JS API调用记录、手工交互脚本以及后端日志分析。

**📊 数据集**

数据集由42个公开可用的AI聊天平台账号、两种固定提示语、产生的会话标题以及对应的第三方请求记录组成。

**📈 对比分析**

通过对比不同平台的第三方请求数量、泄露比例以及跟踪器归属，评估了泄露风险，发现Google和Meta主导，泄露率在20%–45%之间，体现了系统性隐私风险。

**⚠️ 局限性**

局限性包括仅针对Web版平台、单一地区视角、未覆盖移动/桌面应用、实验仅执行一次、隐私政策与实际跟踪行为可能不一致。

---

## 690. An Input Skip Can Reparameterize a VAE Without Improving the Generator

**arXiv ID:** 2610.37456 | [PDF](https://arxiv.org/pdf/2610.37456v1)

**作者:** Kevin Lee `[一作]` (University of California at Los Angeles), Kevin Lee `[通讯]` (University of California at Los Angeles)

**关键词:** `Machine Learning` `Generation` `Data Synthesis` `Representation Learning` `Auto Encoder` `Score-based Model` `Contrastive Learning` `Diffusion model` `Flow-based Model` `Gaussian Splatting` `Time Series` `Tabular` `Benchmark`

**🎯 论文内容**

本文研究了在变分自编码器（VAE）中加入常数输入 skip 对重建误差与生成分布的影响，并通过理论推导和实验验证了 skip 对重建 MSE 的可调性以及对生成性能的潜在负面影响。

**💡 创新点**

创新点在于：①证明对 Gaussian VAE 的 decoder 与噪声进行适当缩放即可在保持生成分布不变的同时任意减小重建 MSE；②展示了单纯优化重建误差会导致能量分数（energy score）和重要性加权对数密度（IW log density）恶化；③引入观察方差估计方法，并与 PPCA 基准进行对比，揭示了观察方差与重建误差共同决定模型生成质量的关系。

**🔧 技术方法**

采用的技术包括：Gaussian VAE、输入 skip 机制、ELBO 与 IWELBO 训练、Adam 优化、能量分数评估、重建方差估计、PPCA 线性基准以及对模型参数的数值重标定。

**📊 数据集**

使用的数据集为：一套人工合成的 Gaussian 实验数据（八维正态潜变量，192维观测）以及 NOAA Deep-ocean Assessment and Reporting of Tsunamis（DART）记录中的七条年度海底压力时间序列（每条 192 个观测窗口）。

**📈 对比分析**

通过比较基准模型（无 skip 与有 skip）在相同观察方差下的重建 MSE、能量分数和 IW log density，并与 PPCA 进行对比。实验结果显示：有 skip 的模型重建 MSE 下降，但能量分数与 IW log density 均出现恶化；PPCA 在所有七条 DART 记录上均获得更低的平均能量分数；同一观测方差下，skip 与无 skip 的能量分数差异呈现正负两种情况。

**⚠️ 局限性**

局限性包括：①输入 skip 的效果高度依赖训练路径与参数缩放，单次训练可能得到不同的生成器；②观察方差估计可能导致生成分布方差偏差，从而恶化能量分数；③实验仅考虑常数 skip，未探讨可变 skip 或更复杂的 skip 结构；④能量分数与 IW log density 对生成质量的评估仍可能缺乏全面性；⑤数据集仅包含 DART 和人工合成数据，难以直接推广到其他时间序列任务。

---

## 691. Electric-Field Induction Motor with Discrete Circuit Elements and Improved Rotor Dynamics

**arXiv ID:** 2610.37452 | [PDF](https://arxiv.org/pdf/2610.37452v1)

**作者:** Alex S. Miller `[一作]` (Massachusetts Institute of Technology), Jeffrey H. Lang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `Robotics` `Physics Related` `Time Series`

**🎯 论文内容**

本研究设计并实现了一种宏观尺度的电磁场诱导电机，利用离散的可重复使用电阻器和电路网络在转子上实现可调的电容耦合时间常数，从而实现了对转子动力学的精准控制。

**💡 创新点**

创新点在于：①将离散、可采购的电阻器嵌入转子，克服了传统连续材料对电阻率限制；②在转子上实现“深条”电路（类似磁场电机的深条转子），使转子时间常数随滑频变化，实现起始扭矩与工作效率的双重优化。

**🔧 技术方法**

主要技术包括：电磁场诱导电机的等离子静电连续介质模型、离散电阻与电容网络的等效电导/电容计算、PCB制造的轴向场结构、混合V珠与气压轴承的悬浮支撑、高压线性放大器驱动与光学测速传感、功率离线自旋测定等实验方法。

**📊 数据集**

使用的数据集为：在10–3000 Hz 的激励频率范围内，对五种不同转子配置（#1–#5）进行实验测得的转速、扭矩与滑频曲线，全部实验数据来源于本文所搭建的测试台。

**📈 对比分析**

通过将实验扭矩与滑频曲线与基于连续介质模型的解析预测进行对比，验证了模型的准确性；在深条转子配置下，起始扭矩提升约 2 倍，峰值扭矩仅略有下降，整体性能满足预期，展示了相较单电阻转子的显著优势。

**⚠️ 局限性**

局限性包括：①转子侧的寄生电容仍显著降低最大扭矩；②实验台受限于固定的气压轴承与可调间隙范围，无法完全模拟真空或极高温环境；③高压驱动需要多频率控制，技术实现复杂；④目前仅验证轴向场结构，径向或线性配置尚未完成；⑤离散元件的热稳定性与长期可靠性尚待进一步研究。

---

## 692. Interpretable Meta-Policies for Neural Decision Making

**arXiv ID:** 2610.37451 | [PDF](https://arxiv.org/pdf/2610.37451v1)

**作者:** Raban Emunds `[一作]` (Technical University Darmstadt), Kristian Kersting `[通讯]` (Technical University Darmstadt)

**关键词:** `Machine Learning` `Explainability and Interpretability` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Large Language Model` `Prompt Engineering` `World Model` `Image` `Text`

**🎯 论文内容**

提出一种层次化强化学习框架，利用神经网络实现低级技能执行，使用符号化的元策略对技能进行高层决策；

**💡 创新点**

通过对象中心化观测、LLM自动生成技能奖励和符号化规则，实现技能的解耦与可解释性，并在不牺牲性能的前提下提高对齐度和对环境改动的鲁棒性；

**🔧 技术方法**

采用并行化Q网络（Parallelised Q-Networks）与λ-回报的离线Q学习，结合对象级状态表示、LLM提示生成奖励与规则、神经符号混合元策略，以及基于规则的符号化元策略；

**📊 数据集**

在JAXAtari的Seaquest、Kangaroo以及Crafter环境上训练与评估；

**📈 对比分析**

与纯深度RL基线（PQN、HPQN、PPO）以及全符号化Nudge对比，使用游戏奖励、手册对齐指标和简化环境鲁棒性进行量化。实验显示，框架在游戏奖励和目标对齐上与基线相当甚至更好，并且在去除敌人等简化改动时保持或提升性能；

**⚠️ 局限性**

依赖准确的对象检测与提取，对LLM生成代码的正确性需人工验证；仅适用于离散动作空间，Q学习限制；奖励误设风险和规则定义仍需人工干预。

---

## 693. Multi-Agent Reinforcement Learning of Co-Adaptive Hip Exoskeleton Control (MARL-Exo) for Metabolic Energy Reduction

**arXiv ID:** 2610.37441 | [PDF](https://arxiv.org/pdf/2610.37441v1)

**作者:** Yifei Yuan `[一作]` (New Jersey Institute of Technology), Xianlian Zhou `[通讯]` (New Jersey Institute of Technology)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Transformer` `Recurrent Neural Network` `Reinforcement Learning` `Simultaneous Localization and Mapping` `Contrastive Learning` `Auto Encoder` `Biomedical Data` `Time Series` `Tabular` `Electronic Health Records`

**🎯 论文内容**

提出并实现了一个两阶段多智能体强化学习框架（MARL-Exo），在模拟中将全身肌肉骨骼模型与双侧髋关节外骨骼协同适应，使用基于生理学的代谢能量奖励，随后在真实人类受试者上无个体调优地部署控制器，验证其在降低步行代谢成本方面的有效性。

**💡 创新点**

创新点在于：①将细粒度肌肉代谢模型直接嵌入外骨骼奖励函数，指导协同学习；②采用随机化解剖、步态与速度的304肌肉全身模型，实现群体级控制器的生成；③通过两阶段训练稳定地实现人机协同适应，并证明在真实受试者中无需个体化调节即可获得代谢收益；④发现去除正功率塑造项可进一步降低代谢功率。

**🔧 技术方法**

技术方法包括：多智能体Proximal Policy Optimization（PPO）与两阶段训练课程；基于DART引擎的全身肌肉骨骼仿真；Uchida/Umberger生理代谢能量模型；双侧髋关节可编程外骨骼硬件；IMU测量髋角度和速度，低通滤波；间接呼吸代谢测量（COSMED K5）评估能量消耗。

**📊 数据集**

数据集主要来自：①在仿真中随机生成的解剖参数、步态参数和速度（0.65–2.0 m/s）；②八名健康成年人在1.25 m/s跑步机上的实验数据，包含无外骨骼、被动外骨骼、以及两种主动控制器的代谢率与运动学记录；未使用公开的第三方数据集。

**📈 对比分析**

比较方法：在实验中采用随机化顺序对无外骨骼、被动外骨骼、MARL‑Exo三种条件进行配对t检验；在仿真中对无设备、带正功率塑形的控制器和仅代谢奖励的控制器进行循环平均代谢功率和肌群分布比较。性能表现为：相较被动外骨骼，MARL‑Exo平均降低18.3 %代谢率；相较无设备，平均降低7.4 %，但六名受试者能达到24.2 %及以上的代谢降低；髋关节运动范围提升36.8 %。

**⚠️ 局限性**

局限性包括：样本量仅8名健康成年人，缺乏不同速度、坡度或障碍步态人群的验证；外骨骼重量较重、扭矩上限有限，导致仿真与实物的功率匹配不足；软组织耦合、IMU误差、关节接口滑动等硬件不确定因素未完全在仿真中捕获；代谢模型的选择可能影响奖励导向，未评估其他模型的鲁棒性；未观察长时间适应或多日使用的效果。

---

## 694. Request-Adaptive Execution for Efficient Hybrid LLM Inference

**arXiv ID:** 2610.37432 | [PDF](https://arxiv.org/pdf/2610.37432v1)

**作者:** Fahao Chen `[一作]` (Shandong University), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `Distributed, Parallel, and Cluster Computing` `Computational Efficiency` `Transformer` `Large Language Model` `Mixture of Experts` `Text`

**🎯 论文内容**

设计并实现了 Flex，一套结合请求自适应混合注意力/SSM LLM 模型与高效推理运行时的系统。

**💡 创新点**

创新点包括：① 在预填阶段进行一次请求级别的门控，动态选择每个块的 Attention 或 SSM 分支并在解码期间保持不变；② selective batching engine 能在批次内根据分支差异拆分/合并，只对 Attention/SSM 计算进行分支化；③ 双层 radix‑tree + 层级优先级驱动的混合上下文缓存，可在不同层路径下共享前缀缓存并实现细粒度驱逐。

**🔧 技术方法**

技术手段：混合块（轻量门 + Attention/SSM 分支 + 共享 FFN）；门控训练时的 soft routing 与成本正则化；FP8/FP16/ BF16 量化；NVIDIA GreenContext 进行 SM 资源划分；load‑aware 任务调度与 GPU 资源分配；两级 radix‑tree 结构和基于访问频率、重算成本、内存占用、深度的缓存优先级公式。

**📊 数据集**

使用六个公开数据集评估：ARC‑Easy、ARC‑Challenge（推理）、PIQA、TruthfulQA（问答）、CNN/DailyMail、PubMed（摘要）。

**📈 对比分析**

与基线比较：模型层面对齐 Llama‑3（全注意力）、Mamba‑2（全 SSM）、Zamba、Jamba、Nemotron‑H（固定混合布局）；系统层面对齐 vLLM、SGLang、Jenga。结果显示：Flex 在 13/18 质量指标上获胜；在相同参数量下 TFLOPs/token 降低 33–67%；系统吞吐量提升 1.5–3.9×，TTFT 降低约 80%，TPOT 降低 60–70%，整体 E2E 延迟显著下降。

**⚠️ 局限性**

局限性：① 预填阶段一次性门控导致对长序列前缀特征的依赖性，若前缀信息不足可能导致子路由不理想；② 两级缓存结构和多分支模型增加了实现复杂度与内存占用；③ 需要专门训练与微调，且不同任务或规模可能需重新优化门控权重与成本正则；④ 在极大模型（如 70B）下仍需 FP8 量化以避免 OOM，可能影响精度。

---

## 695. AMOR: Adaptive Modal Operator Refinement for Operator Learning on Arbitrary Meshes and Irregular 3D Geometries

**arXiv ID:** 2610.37485 | [PDF](https://arxiv.org/pdf/2610.37485v1)

**作者:** David Voderholzer `[一作]` (Lawrence Berkeley National Laboratory), Thomas O'Leary-Roseberry `[通讯]` (Ohio State University)

**关键词:** `Computational Engineering, Finance, and Science` `Optimization` `Computational Efficiency` `Representation Learning` `Graph Neural Network` `Diffusion model` `Contrastive Learning` `Auto Encoder` `Mesh` `Graph` `Physics Related` `Benchmark`

**🎯 论文内容**

提出并实现了一种名为Adaptive Modal Operator Refinement（AMOR）的神经算子框架，替代传统的Fourier卷积以适配任意网格并支持自适应模式基础，证明其为连续算子的全局逼近器；

**💡 创新点**

创新点在于使用低秩的约束基底块取代FFT，消除维度相关的张量乘法扩展，支持独立输入输出基底并通过统一的编码/解码接口实现；

**🔧 技术方法**

核心技术包括有限秩积分算子、重构的输入输出基底、FiLM条件化、可分解张量权重以及对不规则网格的基底迁移；

**📊 数据集**

实验使用八个不同的PDE基准数据集，包括二维非线性扩散、Helmholtz、Kolmogorov流、三维弹性、FlowBench、空气机翼、弹性与机翼肋骨三种加载方案；

**📈 对比分析**

在结构化网格上AMOR与FNO、PCANet的对比显示其在非线性扩散和Helmholtz上误差显著低于FNO，在三维弹性问题上既准确又速度快，而在Kolmogorov流上仍略逊；在不规则网格与3D问题中AMOR优于Geo‑FNO、DSE等方法，且在高维度下推理时间增长更慢；

**⚠️ 局限性**

局限性主要在于静态全局基底对输运驱动或多尺度高波动问题的逼近不足，需更多动态自适应或联合基底学习，并且基底构造和张量参数在高模式数时仍可能导致计算成本升高。

---

## 696. The Minimum Subpacketization of Reed-Solomon Codes with Linear Repair Achieving the Cut-Set Bound

**arXiv ID:** 2610.37522 | [PDF](https://arxiv.org/pdf/2610.37522v1)

**作者:** Weijun Fang `[一作]` (Shandong University), Fang-Wei Fu `[通讯]` (Nankai University)

**关键词:** `Information Theory` `Optimization` `Review/Survey Paper`

**🎯 论文内容**

本文研究了在满足切割集最优性下，Reed–Solomon 代码的最小子分包化程度。

**💡 创新点**

提出了新的数值下界与构造，精确给出了在参数满足 d(d+2)≥4(n+1) 时的最小子分包化，并给出了 G_s(m) 的闭式表达和阈值。

**🔧 技术方法**

利用交叉比场的可除性与相对次数约束，结合基变换构造子空间，以及覆盖自由族与数论工具进行下界与上界证明。

**📊 数据集**

论文主要在理论上证明，未使用具体实验数据集；构造基于任意有限域 𝔽_q 的评估点集合。

**📈 对比分析**

与已有的高速修复方案相比，本文给出的子分包化在理论上是最小的，匹配下界，在大多数参数范围内达到最优。

**⚠️ 局限性**

仅在参数范围外（如某些 (n,d) 组合）仍存在未知情况，且构造仍可能不完全最优，需进一步验证或改进。

---

## 697. Approaching Optimal Scaling of Queue Size for an Input-Queued Switch

**arXiv ID:** 2610.37521 | [PDF](https://arxiv.org/pdf/2610.37521v1)

**作者:** Taha Ameen `[一作]` (Carnegie Mellon University), Weina Wang `[通讯]` (Carnegie Mellon University)

**关键词:** `Performance` `Optimization` `Probability Analysis` `Combinatorial Graph Theory` `Recursive Batch Processing`

**🎯 论文内容**

提出一种参数化为整数 k 的多层批处理调度策略，用于输入排队交换机，以期实现接近 O(n/(1-ρ)) 的队列长度上界。

**💡 创新点**

创新点在于引入递归的批处理层级设计，允许任意层数 k，并通过随机分配初始层级和促发机制保证在每一层都能构造足够的完美匹配，从而使指数 2^k/(2^k-1) 逐渐趋近 1，逼近理论最优缩放。

**🔧 技术方法**

使用了组合图论（s‑factor、完美匹配、Kőnig 颜色定理）、概率分析（Chernoff 上界、随机过程的漂移不等式）以及递归批处理时间块划分的调度算法。

**📊 数据集**

论文中未使用真实网络数据，而是在理论分析和仿真中采用均匀 Poisson/二项分布的随机到达模型，对 n×n 大规模交换机进行实验验证。

**📈 对比分析**

通过与现有批处理策略（如单层批处理、MaxWeight 等）在大系统规模下的理论队列长度上界进行比较，证明所提出的 k‑层策略在 1/(1-ρ)∈[o(n^2),Θ(n^2)] 区间内实现了 O(nlog(n/(1-ρ))/(1-ρ)^{2^k/(2^k-1)}) 的上界，逼近预期的 O(n/(1-ρ)) 目标，且在运行时复杂度保持在 O(n^3k) 每时隙。

**⚠️ 局限性**

局限性在于：(1) 需要预先知道系统参数 ρ 与 n 的标度以确定块长度 L_j；(2) 递归层级 k 越大，调度计算和实现复杂度随 k 增长；(3) 对于极端 ρ 接近 1 的情形，仍需更细致的调度细节或更高阶的层级以进一步降低队列长度。

---

## 698. SecDMP: Provably Secure Data Memory-Dependent Prefetching (Extended Version)

**arXiv ID:** 2610.37454 | [PDF](https://arxiv.org/pdf/2610.37454v1)

**作者:** Jens Sprengers `[一作]` (KU Leuven), Frank Piessens `[通讯]` (KU Leuven)

**关键词:** `Cryptography and Security` `Saft and Privacy` `Benchmark`

**🎯 论文内容**

设计并实现了一种硬件/软件协同的防御SecDMP，用于阻止数据内存相关预取器（DMP）泄露机密数据，并给出形式化安全证明。

**💡 创新点**

提出完整的DMP泄漏分类及其对不同预取器的适用；在不影响公共数据预取性能的前提下，利用静态秘密标记在内存中禁止DMP扫描秘密；给出可在任何基础语义上证明安全的通用框架；首次公开实现商业级DMP并进行实证验证。

**🔧 技术方法**

基于可扩展的Speculative OOO语义模型、DMP训练/预取函数、静态秘密分区、硬件CSR控制、Verilator仿真、基准评测与非干预侧信道检测技术。

**📊 数据集**

通过自定义的DMP‑Bench，将HACL*加密库（Chacha20、Salsa20、SHA2、HMAC、HKDF）与公共工作负载（Embench、CoreMark、SpMV、AoP）以及标准Spectre‑PHT测试程序组合。

**📈 对比分析**

采用非干预泄漏测试和侧信道观测，对比永久禁用、选择性禁用、SecDMP及SecDMP+Spectre防御；结果显示SecDMP在所有泄漏测试中完全防护，平均性能开销仅0.36%（DMP‑Bench），硬件面积增加约0.5%；相比之下永久禁用在安全性下损失2–66%的性能。

**⚠️ 局限性**

依赖静态秘密分区，需手动/自动划分堆/栈；对Apple DMP Pref类的具体机制未知，模型未覆盖；对动态分区/高频栈调度的实用性有限；需要配合安全的秘密跟踪实现才能全面部署。

---

## 699. When Visual Evidence Is Insufficient: How Instructions, Post-Training, and Internal Activations Shape Refusal in Multimodal Language Models

**arXiv ID:** 2610.37430 | [PDF](https://arxiv.org/pdf/2610.37430v1)

**作者:** Ananth K Suresh `[一作]` (Indian Institute of Technology Hyderabad), Vineeth N Balasubramanian `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `Machine Learning` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Vision Language Model` `Multimodality` `Text` `Image` `Benchmark`

**🎯 论文内容**

研究多模态大语言模型在视觉不可回答问题时的拒绝行为，并通过指令、后期训练和内部激活干预三种方式探究其可控性。

**💡 创新点**

提出了同样样本级别的拒绝原因分类与过渡分析框架，证明拒绝行为受指令语义、位置以及模型内部表示共同塑造；同时展示了内部激活方向的可迁移控制潜力。

**🔧 技术方法**

使用了基于指令的前缀/后缀提示、监督微调（SFT）与偏好对齐方法（DPO、ORPO、SimPO），以及通过样本差异构建的激活方向干预。

**📊 数据集**

采用 InternVL3-2B、InternVL3-8B、Qwen2.5-VL-7B 三大 MLLM 与 MoHoBench 视觉不可回答问题基准进行评估。

**📈 对比分析**

结果显示：指令语义（尤其是证据感知和通用拒绝）能使拒绝率提升 18–45pp，指令位置会导致 13–23% 样本行为逆转；后期训练可将拒绝率提升至 90% 以上，但伴随 24% 以上的混合回答；激活方向干预可在所有条件下提升 1–15pp，且不影响描述性请求或 MMMU 评分。

**⚠️ 局限性**

主要局限包括：拒绝率提升并不等同于可靠的证据识别；激活方向效果受单一种子与特定模型限制；研究仅覆盖三款模型和 MoHoBench，缺乏在更广泛 VQA 场景的验证。

---

## 700. The Structure Gap: When More Data Cannot Replace Structured Reasoning

**arXiv ID:** 2610.37523 | [PDF](https://arxiv.org/pdf/2610.37523v1)

**作者:** Kheer Sagar Patel `[一作]` (University of Illinois Chicago), Mojtaba Soltanalian `[通讯]` (University of Illinois Chicago)

**关键词:** `Machine Learning` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Representation Learning` `Graph Neural Network` `Gaussian Splatting` `Contrastive Learning` `Chain-of-Thought` `Graph` `Tabular`

**🎯 论文内容**

定义结构编码问题并揭示在有限资源下结构与无结构学习之间的差距

**💡 创新点**

提出结构缺口概念并用理论和实验证明无结构学习在有限数据下无法完全替代已知结构的推理

**🔧 技术方法**

理论证明（高斯逆问题、单元覆盖下界、未观察轨道不可辨识）与实验评估

**📊 数据集**

合成数据集：高斯逆、k-sparse 线性回归、8-24 节点 Erdős–Rényi 图的可达性问题

**📈 对比分析**

对比全连接网络（结构无知）与结构知情方法（Wiener 逆、OMP/ISTA、GNN、BFS），发现结构知情方法在有限样本和规模外推性能上显著优于全连接网络

**⚠️ 局限性**

局限性：需要已知且正确的紧凑结构；结构误差或未知时可能导致偏差；实验仅覆盖部分问题与模型

---

## 701. World Dreamer v5-lite: A Multi-Robot World Action Model with One Dreamer

**arXiv ID:** 2610.37518 | [PDF](https://arxiv.org/pdf/2610.37518v1)

**作者:** Guoning Wu `[一作]`, Yongbin Zheng `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Vision Language Model` `Vision-Language-Action Model` `Auto Encoder` `World Model` `Flow-based Model` `Contrastive Learning` `Video` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出了一种基于共享三维世界状态和姿态感知查询的多机器人协作控制框架，实现多机器人动作预测与语言指令的联合推理。

**💡 创新点**

创新点包括：① 将多视角视觉特征映射到统一空间坐标构成共享世界状态，解决视角不对应问题；② 采用姿态感知的查询进行跨机器人信息交换，保持各自动作流独立；③ 引入遮蔽视图的合作训练策略，鼓励机器人利用队友的观测信息。

**🔧 技术方法**

技术细节：预训练的 Qwen3‑VL 视觉‑语言模型；预训练的视频 VAE 编码器与空间投影；共享世界状态的三面平面写入与可见性门控更新；基于流匹配的解码器；遮蔽视图与动作前缀的联合损失训练。

**📊 数据集**

数据集：预训练集包含 20 类机器人模拟轨迹、RoboFactory 轨迹和 LeKiwi/OpenArm 真实轨迹（总计 10k+ 小时）；评估集包括 RoboFactory 三项多机任务、RoboCasa365 365 个家庭任务，以及四个真实机器人实验（两 LeKiwi 或 LeKiwi+OpenArm）。

**📈 对比分析**

与六种单机/多机基线（Diffusion Policy、ACT、π_0.5、Fast‑WAM、GauDP 等）对比：在 RoboFactory 上平均成功率 74.0%（比最强基线 π_0.5 高 23.7pp），四个真实任务平均成功率 78.8%（比 π_0.5 高 13.8pp），在 RoboCasa365 上整体成功率 58.08%（比 PatchWAM 高 10.48pp）。

**⚠️ 局限性**

局限性：依赖大量预训练数据，模型对极端遮挡或动态光照的鲁棒性尚未充分验证；信息交换仅通过查询，可能在更复杂场景下不足；实验主要集中在室内特定机器人平台，缺乏对更广泛机器人族群与更大尺度协作的泛化验证。

---

## 702. Ego-CEVR: Learning Cross-Embodiment Visual Representations from Human Egocentric Videos

**arXiv ID:** 2610.37509 | [PDF](https://arxiv.org/pdf/2610.37509v1)

**作者:** Keru Zhou `[一作]` (Maniformer), Maoqing Yao `[通讯]` (Maniformer)

**关键词:** `Robotics` `Robotic Intelligence` `Representation Learning` `Transformer` `Diffusion model` `Flow-based Model` `Auto Encoder` `Vision-Language-Action Model` `Contrastive Learning` `Video` `Multimodality`

**🎯 论文内容**

提出 Ego-CEVR，一种数据本地化框架，利用人类和机器人多源第一人称视频与动作监督，先通过视频预测学习全局任务表示，再通过手腕观测与动作监督学习局部交互表示，实现分层视觉表征学习以提升机器人操作性能。

**💡 创新点**

创新点：① 将多源第一人称视频与动作监督分层结合；② 用视频预测训练全局表示，用手腕观测与动作监督训练交互表示；③ 通过跨注意力将两种表示融合进动作专家，兼顾任务上下文与局部执行；④ 支持少量示例的跨体现形适配。

**🔧 技术方法**

技术：Diffusion Transformer（Wan2.2）视觉骨干 + VAE 编码；条件流匹配（flow‑matching）预测损失；跨注意力（CrossAttn）融合全局与交互表示；基于流匹配的动作专家（π_0.5）以及多阶段预训练与微调。

**📊 数据集**

数据集：LIBERO、LIBERO‑Plus、RoboTwin 2.0（仿真）和 AgiBot G1/G2（真实机器人），并采集人类第一人称视频与机器人头摄像头视频。

**📈 对比分析**

与 VLA、Fast‑WAM、VPP2 等基线对比：在 LIBERO 上成功率 97.7%（接近 Fast‑WAM 97.6%）；在 LIBERO‑Plus 上 90.2% 最高，且在相机、背景、噪声扰动下分别达 91.1%、98.8%、95.4%；在 RoboTwin 2.0 Hard 设定上 70.7% 最高。Ablation 实验表明层次表示提升约 16.4% 以上，显著增强鲁棒性。

**⚠️ 局限性**

局限性：未在大规模人类第一人称视频上预训练，缺乏对模型规模化的验证；对不同规模模型与更大视频集的泛化与提升尚未探索；目前仅在有限机器人平台与任务上验证，跨体现形迁移仍需进一步研究。

---

## 703. Global Choice without Excluded Middle in Dependent Type Theory

**arXiv ID:** 2610.37498 | [PDF](https://arxiv.org/pdf/2610.37498v1)

**作者:** Sewon Park `[一作]`, Sewon Park `[通讯]`

**关键词:** `Logic in Computer Science` `Other` `Review/Survey Paper`

**🎯 论文内容**

构造了一个在依赖型理论中使用全局选择但不满足排中律和命题外延性的模型

**💡 创新点**

首次展示在无命题外延性下，全局选择不蕴含排中律，并给出了完整的一致性证明

**🔧 技术方法**

使用 staged sets（分层集合）和范畴与家庭（CwF）框架，结合不可预测的命题宇宙与无穷真值

**📊 数据集**

无（纯理论构造）

**📈 对比分析**

通过模型验证与已知的 Diaconescu 定理对比，证明在该模型下排中律失效，性能方面无实验数据

**⚠️ 局限性**

模型依赖于选定的良序集合，且仅证明在该特定结构下全局选择不等价于排中律；对更一般的依赖型理论或其他集合论假设的适用性尚未讨论

---

## 704. Revisiting Privacy Risks in Quantized Deep Neural Networks

**arXiv ID:** 2610.37488 | [PDF](https://arxiv.org/pdf/2610.37488v1)

**作者:** Jie Fu `[一作]` (Stevens Institute of Technology), Wendy Hui Wang `[通讯]` (Stevens Institute of Technology)

**关键词:** `Cryptography and Security` `Safty and Privacy` `Computational Efficiency` `Convolutional Neural Network` `Transformer` `Adversarial Attack` `Contrastive Learning` `Auto Encoder` `Image` `Tabular`

**🎯 论文内容**

本研究系统评估了量化深度神经网络（QNN）在面对会员推断攻击（MIA）时的隐私风险，覆盖了八种主流黑盒/白盒攻击和六种主流量化方法，探讨量化对攻击效果、模型性能以及现有防御的影响。

**💡 创新点**

创新点在于：①首次对量化模型进行全量MIA评估，揭示4–8位量化几乎不降低攻击成功率；②发现极低位（2位）量化虽能显著降低隐私泄露，但伴随巨大的模型效能损失；③评估了传统防御（DP、dropout、对抗正则）在QNN上的表现，发现其增益有限并会进一步削弱模型性能。

**🔧 技术方法**

技术手段包括：黑盒/白盒MIA算法（C-MIA, ML-Leaks, SAMIA, Entropy, M-Entropy, RMIA, WB-S, WB-P）；量化技术（PTQ: PyTorch‑Quant, AdaRound, QDrop; QAT: PACT, BSQ, HAWQ）；防御技术（Dropout, 对抗正则, 差分隐私DP‑SGD）。

**📊 数据集**

实验使用四个基准数据集：图像数据集CIFAR‑100、CINIC‑10；表格数据集Location、Texas；对应模型为ResNet‑18、MobileNetV2、MLP。

**📈 对比分析**

通过攻击准确率和TPR@1% FPR评估攻击效果；结果显示4‑bit量化对MIA几乎无影响，2‑bit量化将攻击准确率降至接近随机，但模型精度下降高达95%；在防御方面，量化与DP/Dropout/AR组合只带来微小隐私提升，且往往伴随显著的效能损失，Dropout在保持较高准确率的同时略微降低攻击成功率。

**⚠️ 局限性**

局限性包括：仅评估了当前流行的量化方法与攻击，未设计针对QNN的新攻击或防御；极低位量化导致模型性能极度下降；实验仅涵盖四个数据集和三种模型，可能难以全面推广到更广泛的应用场景。

---

## 705. Keeping Inference in the Radio: Design and Measurement of a Spiking Accelerator for mmWave ISAC Activity Recognition

**arXiv ID:** 2610.37480 | [PDF](https://arxiv.org/pdf/2610.37480v1)

**作者:** Anthony Sánchez `[一作]` (IMDEA Networks Institute), Joerg Widmer `[通讯]` (IMDEA Networks Institute)

**关键词:** `Networking and Internet Architecture` `Recognition` `Spiking Neural Network` `Quantization-aware Training` `Hardware Acceleration` `FPGA Implementation` `Time Series` `Point Cloud`

**🎯 论文内容**

在 mmWave 通信无线电的可编程逻辑中实现并评估了一种用于微德普勒活动识别的脉冲神经网络（SNN）加速器，并与传统 ANN 进行同构对比。

**💡 创新点**

首次在通信硬件上实现 ISAC HAR 的 SNN 加速器；通过量化感知训练将权重量化至 3 bit，显著压缩资源；系统级比较同拓扑 ANN，展示显著的逻辑与能耗优势。

**🔧 技术方法**

采用 LIF 型脉冲神经网络、Spiker+ 固定点加速器生成、Xilinx ZCU208 RFSoC FPGA RTL、基于帧的脉冲编码器以及 HLS 对 ANN 基线进行实现。

**📊 数据集**

使用 DISC‑A 数据集（IEEE 802.11ay 60 GHz mmWave 公开微德普勒光谱）。

**📈 对比分析**

在同一设备、相同时钟频率下，用准确率、宏 F1、推理时延、逻辑使用率和动态能耗等指标进行对比；SNN 132 神经元实现 94.6 % 准确率、206 时钟周期推理，逻辑使用为 ANN 的 1/8，动态能耗为 ANN 的 1/3.4（对比 16 bit ANN 更低 1/7.6）。

**⚠️ 局限性**

受限于前端信号处理功耗和 FPGA 静态功耗；仅在单一链路单一数据集上验证；模型在新主体上仍有性能波动，需进一步改进编码与时延。

---

## 706. What Carries Forward? Pretraining Disclosure Across Language Model Releases

**arXiv ID:** 2610.37465 | [PDF](https://arxiv.org/pdf/2610.37465v1)

**作者:** Md Sakib Ul Islam `[一作]` (Victorian Institute of Technology), S M Mehedi Zaman `[通讯]` (Rutgers University)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Data-Centric Learning` `Transformer` `Large Language Model` `Text` `Review/Survey Paper`

**🎯 论文内容**

对41个模型发布的预训练信息进行文档审计，使用2025年Foundation Model Transparency Index的24项指标比较同一模型家族不同版本之间的披露变化。

**💡 创新点**

首次系统评估同一家族内部版本间披露的连续性，发现总分提升可能掩盖具体指标的丢失，并提出需明确记录继承与变更信息的必要性。

**🔧 技术方法**

结合AI辅助抽取（ChatGPT）与人工审核进行指标编码，并使用统计与可视化方法追踪指标增、减、保留。

**📊 数据集**

使用41个模型的技术报告与官方文档（包括开放权重与开放源码/可复现版本），覆盖17个模型家族。

**📈 对比分析**

通过相邻版本的二元指标比较计算增减，统计24个比较中得分变化：11次下降、6次上升、7次不变，整体增减计数为28增、38减，表明后续版本并非始终提升披露水平。

**⚠️ 局限性**

限制包括样本有限、比较不完全可比、分组条目混合不同变体、仅评估已公开文档、指标为二元且权重相同，未检验实际复现与数据质量。

---

## 707. Concordat: Typed Sessions for LLM Agents

**arXiv ID:** 2610.37461 | [PDF](https://arxiv.org/pdf/2610.37461v1)

**作者:** Milad Sikaroudi `[一作]`, Milad Sikaroudi `[通讯]`

**关键词:** `Artificial Intelligence` `Agentic AI` `Computational Efficiency` `Federated Learning` `Explainability and Interpretability` `Optimization` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Finite State Machine` `Text` `Benchmark`

**🎯 论文内容**

提出了Concordat，一个通过有限状态协议在运行时强制执行大语言模型代理行为的Python库。

**💡 创新点**

创新点在于将面向对象的协议概念与LLM代理结合，利用状态机动态限制可用工具并对继承进行行为安全检查，从而把规则从提示转化为接口属性。

**🔧 技术方法**

采用有限状态机（typestate/session types）、Python装饰器、运行时工具过滤、预算线性拆分以及协议安全继承检查等技术实现。

**📊 数据集**

使用τ‑bench航空客服基准数据集（50个任务）并构造了施压情景的模拟客户对话。

**📈 对比分析**

在对比实验中，标准代理在压力情境下违反确认规则的频率分别为gpt‑4.1的26%和Haiku的23%，而Concordat代理为0%；两者任务成功率基本持平，且Concordat代理的成本略低。

**⚠️ 局限性**

局限包括实验范围仅覆盖单一领域和少数模型、依赖关键字确认检查、模拟客户行为可能不完全代表真实用户，以及对多代理全局协议的支持尚未实现。

---

## 708. PolyGandr: Steering Generative Models via Spatial Navigation through Latent Space

**arXiv ID:** 2610.37446 | [PDF](https://arxiv.org/pdf/2610.37446v1)

**作者:** Mingwei S. G. Li `[一作]`, Remco Chang `[通讯]` (Tufts University)

**关键词:** `Human-Computer Interaction` `Generation` `Data Synthesis` `Generative Adversarial Network` `Diffusion model` `Diffusion model` `Generative Adversarial Network` `Score-based Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Auto Encoder` `Image` `Mesh`

**🎯 论文内容**

本文提出并实现了一种视觉交互框架，将全局潜在空间探索与局部细化相结合，用户可以通过选择基点、对比示例构造方向以及调节步长，在潜在空间中进行向量导航，从而控制GAN、扩散模型乃至工程设计空间（如旋转型kirigami）的生成结果。

**💡 创新点**

创新点包括：
1) 将全局UMAP投影与局部混合面板联动，实现例子驱动的向量导航；
2) 统一的“基点‑方向‑步长”交互框架，支持从示例直接构造并可视化完整的编辑轨迹；
3) 兼容不同生成模型与工程设计任务的通用接口，并在三种案例中验证其可迁移性；
4) 在局部视图中提供可自由重新布局的参考点，实现更细粒度的权重调节；
5) 通过交互式探查验证潜在空间的线性局部性与非欧氏几何。

**🔧 技术方法**

核心技术：
- 潜在空间向量加减与加权插值（基于示例差异、对比对平均等）；
- 参数化UMAP降维（保证新样本可嵌入同一投影）；
- 交互式可视化（全局地图、局部混合面板、轨迹可视化、网格模式）；
- 对扩散模型的“噪声+再解码”插值策略；
- 进程优化（进度条、批量解码、可视化预览）。

**📊 数据集**

使用数据集：
- StyleGAN2训练于人脸与猫图像，采样2万点；
- Stable Diffusion XL训练于6种椅子风格，采样4千点；
- 旋转型kirigami：随机生成2万条凸四边形（8维坐标），用于聚合、投影和可视化。

**📈 对比分析**

评估方式：
- 对比三种案例的交互时延与用户体验；
- 在GAN案例中，用户在全局地图上快速定位、调节方向后可实时生成（帧率>20Hz）；
- 在扩散案例中，单步解码需数秒，但支持进度条与分批渲染；
- 在kirigami案例中，采用专家访谈（40分钟）评估探索效率、可行性发现与约束指引的可用性；
- 结果显示框架能在不需要先指定语义目标的情况下，帮助用户快速发现与细化编辑路径。

**⚠️ 局限性**

局限性：
- 采用线性向量算子，难以捕捉高阶非线性或跨区域语义变化；
- 对扩散模型的实时交互受限，交互延迟较大；
- 对约束空间缺乏自动化可行性引导，用户仍需手动试错；
- 在高维设计空间中，二维混合布局难以覆盖全部自由度，可能导致局部可达性不足；
- 对模型或任务的具体细节（如参数化解码方式、可视化比例）需进一步定制。

---

## 709. Extending the W3C Data Privacy Vocabulary to Cover the German Health Data Use Act - Lessons Learned

**arXiv ID:** 2610.37527 | [PDF](https://arxiv.org/pdf/2610.37527v1)

**作者:**  `[一作]`,  `[通讯]`

**关键词:** `Computers and Society`

---

## 710. Segmentation Confidence for Arbitrary CNNs

**arXiv ID:** 2610.37431 | [PDF](https://arxiv.org/pdf/2610.37431v1)

**作者:** Baris Oguz `[一作]` (University of Pennsylvania), Ipek Oguz `[通讯]` (Vanderbilt University)

**关键词:** `Computer Vision and Pattern Recognition` `Segmentation` `Anomaly Detection` `Convolutional Neural Network` `Contrastive Learning` `Diffusion model` `Biomedical Data` `Image` `Ultrasound` `MRI`

**🎯 论文内容**

提出了一种基于多次预测的不确定性估计，并通过体积随不确定性阈值逐步收缩的比值来计算单一置信度分数，能够在不改造网络的情况下评估医学图像分割质量。

**💡 创新点**

创新点在于将体积变化率转化为单一置信度指标，并利用无监督异常检测快速识别低质量分割。

**🔧 技术方法**

采用测试时翻转或高斯噪声增强生成多份预测，计算标准差与熵混合不确定图，提取体积比例序列并取均值得到置信度。

**📊 数据集**

使用3D超声胎盘数据（210/49张图像）和MRI海马数据（260张训练图像）进行评估。

**📈 对比分析**

与体积变异系数（VVC）对比，置信度与Dice相关性更高，离群检测显著区分低Dice分割，整体性能优于VVC。

**⚠️ 局限性**

局限在于仅测试两种增强方式，需在更多模态和更大样本上验证，临床实用性仍待进一步评估。

---

## 711. Bayes-Adaptive MDPs for Rollout Allocation to Enhance GRPO Training

**arXiv ID:** 2610.37429 | [PDF](https://arxiv.org/pdf/2610.37429v1)

**作者:** Faeze Moradi Kalarde `[一作]` (University of Toronto), Scott Sanner `[通讯]` (University of Toronto)

**关键词:** `Machine Learning` `Optimization` `Reinforcement Learning from Human Feedback` `Transformer` `Large Language Model` `Reinforcement Learning` `Prompt Engineering` `Value Iteration` `Beta Prior` `Advantage Estimation` `Text` `Benchmark`

**🎯 论文内容**

提出了AdaRoll-GRPO框架，用于在大型语言模型的GRPO后训练中根据问题难度动态分配有限的rollout预算。

**💡 创新点**

核心创新是将rollout预算分配问题建模为贝叶斯自适应马尔可夫决策过程（BAMDP），并通过离线值迭代学习出无在线开销的最佳分配策略。

**🔧 技术方法**

使用贝叶斯MDP、Beta先验、值迭代、优势估计的相对奖励机制以及GRPO的现有优化目标。

**📊 数据集**

在数学推理（GSM8K、MATH）、逻辑推理（LogiQA）和常识推理（CoLoTa）四大基准数据集上，采用Llama 3.1/3.2、Gemma 3-4B-IT和Qwen 2-MATH-1.5B模型进行实验。

**📈 对比分析**

与vanilla GRPO、Dr.GRPO和DAPO等基线对比，AdaRoll-GRPO在相同rollout预算下显著提升了准确率，尤其在难度分布熵较高的数据集上收益更大。

**⚠️ 局限性**

局限性包括：离线值迭代对大批量规模的计算复杂度较高；模型假设奖励为二值且使用Beta先验，可能不适用于更复杂或连续奖励情形；以及在极端难题上仍可能需要更高预算。

---

## 712. Predict the Swing, Match the Speed: Robotic Pipe Stabbing With Learned Stereo Vision and Residual Reinforcement Learning

**arXiv ID:** 2610.37535 | [PDF](https://arxiv.org/pdf/2610.37535v1)

**作者:** Mohamed Ashraf Mostafa `[一作]` (Azhar University), Samir Khaled `[通讯]`

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Convolutional Neural Network` `Reinforcement Learning` `Optical Flow` `Contrastive Learning` `World Model` `Image` `Video`

**🎯 论文内容**

提出一种能预测悬挂管道摆动并以匹配速度捕获的机器人立杆装置StabBot，成功实现对不同尺寸管道的无误击入

**💡 创新点**

创新点在于结合计时卡尔曼滤波器预测摆动、姿态匹配速度、触觉确认终端接触以及残差强化学习补偿；同时使用一套可覆盖9种尺寸的开放式V形滚筒头

**🔧 技术方法**

采用双目关键点卷积网络（ResNet-18+热图+软Argmax）定位管道、针尖、盒口，卡尔曼滤波预测摆动，基于模型的控制器实现速度匹配，残差策略使用PPO训练的MLP

**📊 数据集**

使用基于MuJoCo的1:4 Froude缩放模拟数据，生成200个随机试验（不同尺寸、摆幅、误差），并在训练中使用20,000张双目图像做关键点标注

**📈 对比分析**

在200个配对试验中，StabBot相较于固定引导成功率从74%提升到100%，接触速度与峰值力分别降低约7.8倍，且在延迟、帧丢失、偏置等鲁棒性测试中保持100%成功

**⚠️ 局限性**

限制在仿真环境，未涉及真实图像、真机测力误差、机电耦合噪声等实际操作问题；对极端摆幅与极大测量噪声仍有性能下降，需进一步实地验证

---

## 713. A note on data-efficient finetuning of vision foundation models

**arXiv ID:** 2610.37510 | [PDF](https://arxiv.org/pdf/2610.37510v1)

**作者:** Niclas Hergenröther `[一作]` (Max Planck Institute for Intelligent Systems), Antonio Orvieto `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `Computer Vision and Pattern Recognition` `Classification` `Convolutional Neural Network` `Transformer` `Supervised Fine-Tuning` `Prompt Engineering` `LoRA` `Dropout` `L1/L2 Regularization` `Autoaugment` `Image`

**🎯 论文内容**

对视觉提示（VP）在多种基础模型上的性能进行系统评估，并与线性探测、全微调和LoRA等迁移学习方法进行对比。

**💡 创新点**

系统化地评估不同模型架构（注意力、卷积、Mamba）与VP、LP、FFT、LoRA的交互，揭示ConvNext最优且VP在小样本任务下可通过标签映射/正则化提升。

**🔧 技术方法**

采用AutoVP、FCLM、可调整提示尺寸、LoRA、Dropout、L1/L2正则、SAM、Autoaugment等技术。

**📊 数据集**

使用ImageNet1k预训练模型和8个下游图像分类任务（CIFAR10/100、Flowers102、GTSRB、DTD、EuroSAT、Food101、OxfordPets）。

**📈 对比分析**

通过在DTD等数据稀缺任务上对VP、LP、FFT、LoRA进行性能对比，发现LoRA在保持轻量的同时往往优于FFT，VP在大模型上表现可与Transformer相当，但在小数据集上易过拟合。

**⚠️ 局限性**

局限性在于对模型和任务的覆盖有限，VP在小样本任务中的泛化仍受限，正则化难以完全弥补与LoRA的性能差距，实验对GPU资源要求较高。

---

## 714. semanticfa: An R Package for Response-Free Semantic Factor Analysis of Psychometric Scales

**arXiv ID:** 2610.37494 | [PDF](https://arxiv.org/pdf/2610.37494v1)

**作者:** Devon Yanitski `[一作]` (University of Alberta), Chris Westbury `[通讯]` (University of Alberta)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Representation Learning` `Transformer` `Large Language Model` `Contrastive Learning` `Factor Analysis` `Text` `Tabular` `Review/Survey Paper`

**🎯 论文内容**

使用语义嵌入结合因子分析（semanticfa 包）对 50 个 IPIP Big‑Five 项目进行工作流程示例，生成与传统问卷相似的因子结构并对其进行解释与细化。

**💡 创新点**

将大规模语言模型（Qwen3-Embedding）生成的语义向量嵌入到因子分析流程中，首次实现语义结构与人类回答结构的系统对比与自适应因子数选择。

**🔧 技术方法**

使用 R（semanticfa、psych、EGAnet 等）进行因子分析、保留准则、负荷矩阵、Congruence 计算，Python（sentence‑transformers、torch）构建 NLI 相似度矩阵，结合多种嵌入模型（0.6B、4B、8B）对项进行编码。

**📊 数据集**

IPIP Big‑Five Factor Markers 的 50 项目（公开匿名数据集）以及 42 项 DASS 项目用于嵌入比较与补充分析。

**📈 对比分析**

通过五种保留准则（与最小数值优先）确定因子数，并用 NLI 相似度矩阵、Congruence 量化语义因子与响应因子的一致性，结果显示语义因子与人类回答的相关系数与原始响应相符，且因子数与作者意图一致。

**⚠️ 局限性**

对嵌入质量与硬件浮点运算的依赖导致结果在不同机器上略有差异；未预注册实验；仅基于公开数据，可能缺乏对更多测量工具的泛化验证。

---

## 715. Teach reality, not ethics: a drop-in continued-pretraining corpus installs coupled human-biosphere-AI welfare dispositions across model families

**arXiv ID:** 2610.37484 | [PDF](https://arxiv.org/pdf/2610.37484v1)

**作者:** Trent R. Northen `[一作]` (Bioaligned Labs), Mingxun Wang `[通讯]` (Bioaligned Labs)

**关键词:** `Computation and Language` `Explainability and Interpretability` `Federated Learning` `Computational Efficiency` `Reinforcement Learning from Human Feedback` `Drug Discovery` `AI Code Assistant` `Optimization` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Mixture of Experts` `Text` `Biomedical Data` `Review/Survey Paper` `Benchmark` `Agriculture Related` `Physics Related`

**🎯 论文内容**

构建并使用一份基于真实文献的“耦合福利”继续预训练语料库，训练大型语言模型在面对生态风险决策时更好地抵御压力诱导的破坏性选择。

**💡 创新点**

创新点在于通过对人类福祉、生态功能与模型自身能力三者依赖关系的系统化、实证化描述，而非仅靠后训练的价值声明，直接植入模型的推理框架，从而实现更深层次的生态责任感。

**🔧 技术方法**

采用 QLoRA 继续预训练（CPT）技术，将语料库插入到模型的权重中；评估使用压力阶梯、决策网格和开放式承保决策任务等多种实验工具。

**📊 数据集**

数据集为 934 篇重写的案例文档，所有文档均可追溯至真实论文（来源于 peS2o、arXiv、PubMed 等），并配备同样大小但去除生态内容的对照集（Arm‑N）。

**📈 对比分析**

在五个不同架构（Dense、Hybrid‑SSM、Sparse‑MoE）的模型上进行对比实验，结果显示在压力阶梯和决策网格上，已训练模型的“破坏率”显著下降（最高降至 0/300 或 0.004），且 MMLU 能力保持不变；在开放式承保任务中，训练模型在高压和绿洗指令下仍保持较高的拒绝率。

**⚠️ 局限性**

主要局限包括单一随机种子导致结果可变性、模型部署堆栈对绝对率敏感、压力阶梯的动态范围受限、部分模型（如 Phi‑4）梯度数值难以重现、语料库偏向 peS2o 与 arXiv 来源、以及缺乏多模型、多种评测环境的再现性验证。

---

## 716. Follow the Gradient: Tracing Safety Degradation Back to Training Data

**arXiv ID:** 2610.37481 | [PDF](https://arxiv.org/pdf/2610.37481v1)

**作者:** Saket Reddy `[一作]`, Pratyush Maini `[通讯]`

**关键词:** `Machine Learning` `Safty and Privacy` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Contrastive Learning` `Text` `Benchmark`

**🎯 论文内容**

构建了一种可扩展的梯度嵌入方法 SimpleLoRIF，用于检测并去除大型微调数据集中对安全性产生负面影响的例子，并基于此过滤掉 Tulu3 数据集 20% 的高梯度相似度样本，生成 SafeTulu 数据集；随后在多种 LLM（Qwen、Llama、Olmo）上验证该过滤能显著降低 jailbreak 攻击成功率，同时保持或提升模型能力。

**💡 创新点**

创新点：①将 LoRIF 简化为只在最终四层 transformer 反向传播并去掉曲率校正，从而在百万级数据上实现高效梯度嵌入；②发现“emergent misalignment”在梯度空间中可被捕获，验证不良代码训练样本与广泛错误行为梯度高度相似；③提出基于梯度相似度的安全数据筛选方案，能比内容过滤或先前梯度方法更精准、更高效；④通过 SafeTulu 实现可自定义过滤比例与子集，促进研究者灵活应用。

**🔧 技术方法**

使用技术：梯度归因（SimpleLoRIF）→梯度嵌入与相似度评分；聚类目标集中心化评分；梯度嵌入压缩（随机投影+rank‑1 近似）；梯度相似度对模型微调数据进行排序并去除；多任务评估：白盒 GCG、黑盒 ActorBreaker、各类能力基准（ORBench、MMLU、IFEval、MBPP、GSM8K、MATH、MedQA）。

**📊 数据集**

数据集：①Tulu3（939k 示例）用于训练与过滤；②SafeRLHF（43k 违规响应）作为目标集；③不安全代码与安全代码对比数据用于验证 EM；⑤评测基准包括 JailbreakBench（GCG、ActorBreaker）、ORBench、MMLU Pro、IFEval、MBPP、GSM8K、MATH 500、MedQA。

**📈 对比分析**

对比方法：基线完整微调、随机去除 20%、LLM‑as‑a‑Judge 过滤（内容安全评分）、He 等人 2023 的梯度过滤；结果显示 SafeTulu 在所有模型上 GCG 和 ActorBreaker 攻击成功率均下降 12‑22%，且在多项能力基准上保持或提升分数，优于其他三种过滤方式。

**⚠️ 局限性**

局限性：仅测试通用安全降解，未覆盖特定危害（如 sycophancy）；仅使用 SafeRLHF 作为目标集，未验证对其他危害标签的泛化；梯度相似度筛选假设对不同模型/规模均有效，但在极大模型或不同任务上可能需要额外验证。

---

## 717. Autonomous fixed-wing interception using reinforcement learning

**arXiv ID:** 2610.37470 | [PDF](https://arxiv.org/pdf/2610.37470v1)

**作者:** Daan van Hamersveld `[一作]` (Delft University of Technology), Iurii Kriachko `[通讯]` (Delft University of Technology)

**关键词:** `Robotics` `Reinforcement Learning` `Autonomous Driving` `Robotic Intelligence` `Recurrent Neural Network` `Convolutional Neural Network` `Reinforcement Learning` `Domain Randomization` `Optical Flow` `Kalman Filter` `Simultaneous Localization and Mapping` `Image` `Video` `Tabular`

**🎯 论文内容**

设计并实现了一套基于端到端神经网络的固定翼拦截无人机控制系统，并在真实飞行中验证了其能够成功拦截MALE无人机。

**💡 创新点**

创新点在于：①利用强化学习（PPO）直接将滤波后的单目相机检测结果映射到四个执行器指令，绕过传统的低层控制；②采用多阶段课程学习和域随机化（domain randomization）实现从仿真到实飞的稳健迁移；③将经典的Proportional Navigation与Pure Pursuit作为基线，展示RL在多数场景下的优越性能。

**🔧 技术方法**

技术主要包括：Python+JAX/Numba仿真环境、Proximal Policy Optimization（Stable‑Baselines3实现）、常数加速度卡尔曼滤波器、单目相机基于边界框的定位、姿态与动作滞后模型、以及手动调参的层级自适应飞控回路。

**📊 数据集**

数据集为自建的仿真环境，包含数千个随机生成的目标轨迹与动态参数；实飞数据来自两架X‑UAV Talon平台，使用单目相机检测目标并记录相机图像、飞控日志与卡尔曼滤波估计。

**📈 对比分析**

通过与PN、PP两种经典引导器在相同仿真环境下的对比实验，RL控制器在90%以上的情形下成功拦截，且在大部分课程阶段的成功率高于基线；在真实飞行中，RL控制器成功完成了6次拦截任务，其中3次达到精确碰撞距离≤3 m，显示出良好的实战可行性。

**⚠️ 局限性**

局限性包括：①依赖相机检测，未实现自主获取目标；②对误检和目标尺寸变化的鲁棒性有限；③在极端高速或极大偏离训练分布时性能下降；④RL控制器产生的舵面振荡行为不易解释；⑤基线控制器未在真实飞行中验证，实际性能对比不完整。

---

## 718. ETAMem: Decoupling and Aligning Heterogeneous Memory for Long-Term Conversational Agents

**arXiv ID:** 2610.37443 | [PDF](https://arxiv.org/pdf/2610.37443v1)

**作者:** Xinying Qian `[一作]` (Nankai University), Jeff Z. Pan `[通讯]` (University of Edinburgh)

**关键词:** `Computation and Language` `Retrieval` `Representation Learning` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Contrastive Learning` `Text`

**🎯 论文内容**

提出了一种多视角记忆框架ETAMem，专门用于长期对话代理中的异构信息存储与检索

**💡 创新点**

创新点在于将对话信息拆分为原子记忆单元（AMU），并针对事件、情感与时间三类信息分别构建专属层次结构和表示，同时使用熵引导的检索路由分配查询预算

**🔧 技术方法**

利用LLM进行AMU抽取与情感分类，采用文本与情感编码器形成专属表示，使用树形结构（事件树、情感树、时间树）进行动态维护，配合熵路由与层次beam搜索实现检索

**📊 数据集**

在LoCoMo和LongMemEval-S两个长期对话记忆基准上进行评估，并在CloneMem上进一步验证情感检索效果

**📈 对比分析**

与MemoryBank、A-MEM、Mem0、MemoryOS、MAGMA、TiMem、LightMem等基线比较，ETAMem在LoCoMo上总体准确率提升12.46%，在LongMemEval-S提升7.96%，并且在构建和推理的token成本上保持竞争力

**⚠️ 局限性**

当前的情感表征仅基于情绪极性，可能无法捕捉中性偏好或细微情绪；未来需加入更细粒度的情感关联分类

---

## 719. FeatureArc: Benchmarking Coding Agents on Evolving Features

**arXiv ID:** 2610.37434 | [PDF](https://arxiv.org/pdf/2610.37434v1)

**作者:** Yunlong Wang `[一作]` (Tencent), Zhichao Hu `[通讯]` (Tencent)

**关键词:** `Software Engineering` `AI Code Assistant` `Benchmark` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Code` `Benchmark`

**🎯 论文内容**

构建了一套基于仓库历史的同一功能多阶段演进基准（Evolving Features Benchmark），并实现了自动化任务生成管道，评估代码生成模型在L4级别（无中间错误反馈）下持续扩展功能的能力。

**💡 创新点**

创新点在于：① 通过对仓库历史进行特征挖掘，恢复功能的真实演进链；② 结合显式需求文档、接口契约与可执行的 F2P/P2P 测试，形成完整的端到端验证流程；③ 设计 L4 评估场景，让模型在不接收外部错误反馈的情况下自行检测和修复前置错误，真正考察模型的持续性和自检能力。

**🔧 技术方法**

使用的技术包括：
- 变更关系图与四种搜索（全局 beam、anchor-guided、endpoint bridging、community-guided）来挖掘同一功能链；
- LLM 辅助的代码审查、测试提取与规格生成；
- 自动化环境构建、测试校验与基准验证；
- 采用多语言（Python、TypeScript、Rust、Go）支持与多模型（Claude Opus 5.5、GPT‑6 Astra、Hunyuan 4 Preview 等）对比。

**📊 数据集**

使用了 73 个手工审核的任务，涵盖 2‑10 个阶段，来源于 4 种主流编程语言的公开仓库（含至少 30 星、开源许可、Linux 可执行等条件）。

**📈 对比分析**

比较方法：将 10 个不同模型在同一套任务上执行 L4 评估，记录单阶段通过率、链级通过率、继承错误率、自检覆盖率等指标。性能表现：首阶段通过率约 62.7%，但链级通过率仅 28.5%；继承错误导致后续阶段 F2P 通过率从 70.2% 降至 22.6%；大多数模型在出现失败后几乎不再检测前置错误，导致错误累积。

**⚠️ 局限性**

局限性包括：
- 评估场景仍以 L4 为主，未覆盖更低级别的交互式反馈；
- 依赖 LLM 进行任务审查与测试生成，可能引入误差；
- 基准覆盖的语言和应用场景有限，无法覆盖所有工业项目；
- 对模型内部自检机制的分析仍以轨迹标签为主，缺乏更细粒度的内部执行日志；
- 目前对长任务（≥5 阶段）的成功率极低，说明仍需改进模型的长期记忆与错误恢复能力。

---

## 720. Orthogonal and Parametrized Multi-Attribute Steering of LLMs

**arXiv ID:** 2610.37517 | [PDF](https://arxiv.org/pdf/2610.37517v1)

**作者:** Remy Ogasawara `[一作]` (University of Utah), Jeff M. Phillips `[通讯]` (University of Utah)

**关键词:** `Machine Learning` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Prompt Engineering` `Text`

**🎯 论文内容**

提出了激活空间 Steering 的改进方法：对每个向量识别可驱动区间并重新缩放、实现参数化 Steering 以实现可解释的强度设置，以及通过低秩正交变换实现多属性的独立控制。

**💡 创新点**

创新点在于（1）识别并剪裁每个 steering 向量的可驱动区间，使强度具有统一可解释意义；（2）将 Steering 从“推”改为“设”——参数化 Steering 可直接设置激活在期望强度；（3）通过线性正交化消除多属性干扰，使多属性 Steering 变得可组合、稳定。

**🔧 技术方法**

使用了线性表示假设、对比激活添加、参数化 Steering、低秩正交变换（投影并映射到正交空间）以及 Llama‑2 7B、Qwen‑3 8B 等大模型的单层激活。

**📊 数据集**

使用了 Anthropic 的 Sycophancy 与 Advanced AI Risk 评估集、TruthfulQA、Toxigen、BBQ、以及自制的 Warmth 合成数据集进行实验。

**📈 对比分析**

与传统 α‑iterative、未剪裁方法以及 MAT‑Steer 进行对比；在多属性 Steering 的敏感度与可驱动范围上，参数化+正交化方法平均提升约10 % 评分，范围增大至原来的约3倍，且对相关属性的干扰显著降低。

**⚠️ 局限性**

局限性在于仅针对单层线性激活；正交化仍受向量相关性的影响；对更复杂或跨层的行为控制仍不充分，且可能被滥用于放大不良行为。

---

## 721. Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well

**arXiv ID:** 2610.37652 | [PDF](https://arxiv.org/pdf/2610.37652v1)

**作者:** Ian Chuang `[一作]` (University of California, Berkeley), Claire Tomlin `[通讯]` (University of California, Berkeley)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Convolutional Neural Network` `Reinforcement Learning` `Behavior Imitation (DAgger)` `Symmetric Actor-Critic Training` `ResSafe Safety Filter` `Image` `Point Cloud` `Sequential`

**🎯 论文内容**

本文训练并部署了LEAP手臂的姿态到位控制策略，能够在任何手腕方向下将物体移动到任意姿势，并配备可调的残差安全过滤器。

**💡 创新点**

创新点包括：利用自生成抓取数据与适应性摩擦与目标课程提升学习效率；引入可调残差安全过滤器，零样本转移且易于调节安全性；实现仅靠腕部深度摄像头即可完成端到端控制，展示了高度可移植性。

**🔧 技术方法**

主要技术包括强化学习（PPO）+ 行为模仿（DAgger）+ 对称演员-评论家训练 + 视觉CNN + 运动捕捉 + 机器人动力学校准 + 适应性摩擦课程 + ResSafe改进的安全过滤器。

**📊 数据集**

使用LightningGrasp生成约100k抓取，扩展并滤除碰撞得到约400k抓取；无人工数据；评估集为100k未见抓取目标；模拟与真实传感器收集的目标序列。

**📈 对比分析**

通过与无过滤器、HJ安全过滤器、无课程/无特权观察的基线对比，实验显示在真实机器人上13.3小时、4800个目标，状态策略95.2%成功，安全过滤器将跌落率从16降至4；深度策略在精度与可靠性上略低但仍表现可观。

**⚠️ 局限性**

局限性在于仅针对单个物体和单个过滤器训练；视觉sim-to-real差距在刀与酒杯上明显；过滤器需针对每个目标/政策单独训练；未实现零跌落和100%成功；多物体、多任务与更长序列的鲁棒性仍是挑战。

---

## 722. Repurposing Video Diffusion Transformers for Cross-View Temporal Object Correspondence

**arXiv ID:** 2610.37633 | [PDF](https://arxiv.org/pdf/2610.37633v1)

**作者:** Siyoon Jin `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `Computer Vision and Pattern Recognition` `Object Tracking` `Segmentation` `Transformer` `Diffusion model` `Flow-based Model` `Optical Flow` `Contrastive Learning` `Video`

**🎯 论文内容**

本文提出了一种联合视角-时间的目标对应框架，利用预训练视频 Diffusion Transformer（DiT）在 RGB‑to‑Mask 领域实现跨视角与时序的连续目标分割，避免了传统帧级匹配+追踪的误差累积。

**💡 创新点**

创新点在于：①将视频 DiT 重新定位为流匹配模型，将 RGB 迁移到掩码的潜在空间；②设计统一的双视角 token 空间，实现跨视角全注意力与时间连续性的耦合；③引入焦点注意力对齐、硬负样本约束和局部上下文裁剪等针对性机制，强化跨视角目标定位与实例辨别；④构建稀疏到稠密的伪标签修订流程，利用双向传播、跨视角几何与语义验证提升稠密监督质量。

**🔧 技术方法**

主要技术包括：视频 Diffusion Transformer、流匹配（Flow Matching）框架、全注意力跨视角融合、RoPE 位置编码、硬负样本损失、局部上下文裁剪、跨视角几何验证（VGGT）、语义一致性检测（DINOv3）。

**📊 数据集**

使用 Ego‑Exo4D 验证集（已通过自研伪标签生成得到稠密标注）进行评测，同时对比官方基线与公开方法。

**📈 对比分析**

在 Track‑then‑Match 与 Match‑then‑Track 两种传统流水线下，本文方法在视觉一致性（VA）与条件召回（mR@IoU）等时序一致性指标上均实现领先或竞争性表现；在区域相似度（J）和边缘精度（F）方面略逊于基于 SAM2 的追踪方法，原因是视频 DiT 的潜在空间压缩导致边缘细节丢失。

**⚠️ 局限性**

局限性包括：①潜在空间压缩导致小目标的边缘精度下降；②对极端遮挡或快速视角变化时的跨视角对齐仍可能产生误匹配；③伪标签生成与过滤流程对 GPU 资源和复杂度有一定要求，可能限制大规模部署。

---

## 723. What Improves During Pretraining? Tracking Knowledge Acquisition with Point-in-Time Models

**arXiv ID:** 2610.37631 | [PDF](https://arxiv.org/pdf/2610.37631v1)

**作者:** Chang Su `[一作]` (University of Pennsylvania), Yao Lu `[通讯]` (University College London)

**关键词:** `Artificial Intelligence` `Representation Learning` `Explainability and Interpretability` `Data-Centric Learning` `Transformer` `Large Language Model` `Prompt Engineering` `Contrastive Learning` `Text` `Benchmark`

**🎯 论文内容**

本文利用点时间（Point‑In‑Time）语言模型评估预训练过程中知识学习的时间可变性，提出了基于不同年份模型负对数似然方差的 Temporal Variability Score（TVS），并分析了低TVS和高TVS项随训练进度的学习差异。

**💡 创新点**

创新点在于：①使用点时间模型计算项的系数变异作为 TVS，捕捉知识在不同年份语料中的出现不一致性；②发现高 TVS 项在后期预训练中提升速度快于低 TVS 项，从而揭示知识学习的阶段性差异。

**🔧 技术方法**

技术包括：系数变异计算 TVS；在多模型（Pythia、TinyLlama、SmolLM3、OLMo 等）不同检查点上对多项选择基准进行长度匹配分组；机会归一化准确率评估；对答案特定 TVS 与后期 margin 增长进行回归分析。

**📊 数据集**

使用的数据集包括 16,239 条来自 HellaSwag、PIQA、WinoGrande、ARC‑Easy/Challenge、MMLU 的项目；采用 PIT‑4B 系列（2017–2024 年）4B 参数模型作为对照；并对 FineWeb 语料进行年度覆盖检查。

**📈 对比分析**

比较方法：在各模型的多个检查点（从 0.54B 至 11.1T 训练标记）下，按 TVS 将项目分为低、中、高三组，使用机会归一化准确率比较学习曲线。结果显示，早期差异不大，但在后期高 TVS 组显著优于低 TVS 组，平均提升约 7.3%（机会归一化），在知识密集任务（ARC、MMLU）上更为显著。

**⚠️ 局限性**

局限性包括：TVS 取决于点时间模型的质量；仅使用单一模型规模（4B）PIT‑4B 进行 TVS 计算；未完全排除基准泄漏问题；研究仅覆盖多项选择基准，未验证对其他任务或模型规模的通用性。

---

## 724. VideoLVR: Latent Visual Reasoning over Grounded Video Evidence

**arXiv ID:** 2610.37613 | [PDF](https://arxiv.org/pdf/2610.37613v1)

**作者:** Shilong Li `[一作]`, Shilei Wen `[通讯]`

**关键词:** `Computer Vision and Pattern Recognition` `Representation Learning` `Reinforcement Learning from Human Feedback` `Explainability and Interpretability` `Transformer` `Vision Language Model` `Vision-Language-Action Model` `Reinforcement Learning` `Chain-of-Thought` `Contrastive Learning` `Video` `Text` `Multimodality`

**🎯 论文内容**

提出一种VideoLVR框架，通过将视频中的局部时空区域聚合为潜在视觉块来支持多步推理。

**💡 创新点**

创新点包括：①将时空分布的局部视觉证据按推理步骤分组并作为潜在视觉块；②使用视觉词预测（VTP）将视觉与文本推理统一在同一自回归框架；③引入TVPO实现文本与视觉动作的联合强化学习。

**🔧 技术方法**

采用潜在视觉推理、视觉词预测、教师模型合成证据驱动推理轨迹、GRPO/TVPO等技术。

**📊 数据集**

使用OneThinker和Video-R1的视频问答数据，利用Qwen3-VL-30B-A3B-Thinking合成的含时空标注推理轨迹进行训练。

**📈 对比分析**

在五个视频推理基准（Video-MMMU、TempCompass、MMVU、LongVideo-Reason、VideoMathQA）上，与多种基线对比，基于Qwen3-VL-8B的VideoLVR在平均分上提升约1.3点，基于Qwen2.5-VL-7B的版本也取得最高平均分。

**⚠️ 局限性**

主要局限是依赖合成或人工时空标注数据，且潜在视觉状态仅捕捉局部视觉信息，难以表达更高层次语义。

---

## 725. Beyond Answer Changes: Matched Same-Class Controls for Local Audio Evidence

**arXiv ID:** 2610.37605 | [PDF](https://arxiv.org/pdf/2610.37605v1)

**作者:** Yunuo Wang `[一作]` (Hong Kong University of Science and Technology), Xing Liu `[通讯]` (Northeastern University)

**关键词:** `Sound` `Explainability and Interpretability` `Data Synthesis` `Transformer` `Prompt Engineering` `Diffusion model` `Contrastive Learning` `Audio` `Multimodality`

**🎯 论文内容**

对音频-语言模型进行局部波形干预后回答变化的评估，采用匹配的同类与不同类事件替换方案；

**💡 创新点**

引入同类替换对照，能区分模型对事件类别变化与波形重写的敏感度，提供了更精细的本地证据评估方法；

**🔧 技术方法**

使用匹配的波形编辑、交叉淡化、水平估计、背景复制等音频处理技术，结合答案变化与选择性准确率的统计分析；

**📊 数据集**

合成声景数据集（Scaper）基于FSD‑MIX‑SED（FSD50K）前景音频和TUT Urban Acoustic Scenes 2018背景，包含600个声景、33类查询事件；

**📈 对比分析**

对比Qwen2.5‑Omni‑7B、Qwen2.5‑Omni‑3B与Fun‑Audio‑Chat‑8B三模型，在600声景中进行1,200次试验，发现不同类替换导致的答案变化率比同类重写高49.7–60.9个百分点，选择性准确率达73.7–81.9%；

**⚠️ 局限性**

局限于合成数据、单一声景长度与声压比，缺乏自然录音验证；需要精确事件边界与同类donor，无法直接推广到更复杂或多标签情境；

---

## 726. GAS-PFN: Graph-Aware Spatiotemporal Prior-Data Fitted Networks for Zero-Shot Traffic Forecasting

**arXiv ID:** 2610.37598 | [PDF](https://arxiv.org/pdf/2610.37598v1)

**作者:** Joseph Natter `[一作]` (Kennesaw State University), Rui Wu `[通讯]` (Kennesaw State University)

**关键词:** `Machine Learning` `Autonomous Driving` `Optimization` `Computational Efficiency` `Data-Centric Learning` `Graph Neural Network` `Transformer` `Diffusion model` `Contrastive Learning` `Mixture of Experts` `Graph` `Time Series` `Tabular` `Benchmark`

**🎯 论文内容**

提出了一种基于零样本预测的图结构交通流量预测模型GAS-PFN，能够在没有参数更新的情况下，仅使用最近90分钟的历史和图网络结构进行预测。

**💡 创新点**

创新点包括将随机游走扩散的道路网络结构融入合成任务先验，设计双掩码Transformer分支分别建模传感器内时间动态和邻域空间-时间交互，以及混合上下文记忆与混合邻域路由策略。

**🔧 技术方法**

采用Prior-Data Fitted Networks框架，Transformer主干，随机游走扩散、双掩码注意力、门控融合、语义与图相似度混合邻域。

**📊 数据集**

在六个公开交通预测基准上评估：METR-LA、PEMS-BAY、PEMS03、PEMS04、PEMS07、PEMS08。

**📈 对比分析**

与TimePFN及四个通用零样本预测器（Chronos-2、TimesFM-2.5、Moirai-2、Lag-Llama）以及SCPT进行比较；在90分钟预测窗口上平均MSE下降34%，在27/36指标上获得最佳结果，且在受限监督道路覆盖时仍优于SCPT。

**⚠️ 局限性**

在部分大规模网络上性能提升有限，随着上下文预算增大收益递减；零样本模式仍无法完全匹配充分监督模型的性能，且依赖准确的道路拓扑信息。

---

## 727. Spectral Braids: A Topological Descriptor for Dynamic Directed Networks

**arXiv ID:** 2610.37580 | [PDF](https://arxiv.org/pdf/2610.37580v1)

**作者:** Mehmet Emin Aktas `[一作]` (Kennesaw State University), Esra Akbas `[通讯]` (Georgia State University)

**关键词:** `Social and Information Networks` `Classification` `Contrastive Learning` `Representation Learning` `Graph`

**🎯 论文内容**

本文提出谱结迹（spectral braids）方法，对动态有向网络的社区转移矩阵进行复特征值跟踪，并将其随时间的绕行轨迹映射为固定长度特征，用于网络分类；

**💡 创新点**

创新点在于首次将结迹与 braid monodromy 概念引入图数据，通过记录复特征值的相对运动捕获网络循环行为，并利用 Burau 表示将可变长度的 braid 转化为固定维度特征；

**🔧 技术方法**

采用社区层面转移矩阵、复特征值跟踪、braid 字词读取、Burau 表示、几何统计和指数求和等技术；

**📊 数据集**

实验使用了两组合成在线社区动态网络（History 与 Order）以及四个真实网络（邮件、贸易、Reddit、共享单车）验证方法可行性；

**📈 对比分析**

与未跟踪特征的基线比较，在 5‑折交叉验证下，Burau 特征实现 97.1% 的准确率，远超 44.5%；在事件顺序区分任务中达到 90.8%，并对不同速率、噪声的网络保持高准确率；

**⚠️ 局限性**

方法仅适用于存在多组循环流动且特征值非实数的网络；若转移矩阵近似可逆或几乎所有特征值实数，则 braid 为空；并且需要每个时间窗口至少数千条边才能获得可靠特征。

---

## 728. Pneuma: A Large-Scale Real-World Time Series Benchmark for Commercial Aircraft Predictive Maintenance

**arXiv ID:** 2610.37563 | [PDF](https://arxiv.org/pdf/2610.37563v1)

**作者:** Jiafeng Lin `[一作]` (Tsinghua University), Jianmin Wang `[通讯]` (Tsinghua University)

**关键词:** `Machine Learning` `Anomaly Detection` `Benchmark` `Transformer` `Recurrent Neural Network` `Convolutional Neural Network` `Time Series`

**🎯 论文内容**

提出了 Pneuma，首个公开的大规模航空预测维护基准，涵盖4类商用机、9种真实故障，基于 QAR 时序记录与维护日志。

**💡 创新点**

创新点在于：①提供跨机型、跨故障的真实运营数据与标签；②三大任务（多变量预测、监督分类、无监督异常检测）统一拆分与评估协议；③系统性比较多种时序模型并揭示预测与异常检测的关联。

**🔧 技术方法**

使用多类时序模型：全序列学习（Autoformer、TimesNet、DLinear等）、时间序列基础模型（Toto‑2、Chronos‑2、TimesFM‑2.5 等）、分类模型（MambaSL、VSFormer、LITE、MiniROCKET 等）、异常检测模型（KAN‑AD、Anomaly Transformer、TranAD、USAD、OmniAnomaly 等）。

**📊 数据集**

使用 Pneuma 数据集，包含约920M 时间步、4机型（A320、A321、B787、B777）、9故障情形，所有标注来源于维护日志与 QAR 记录。

**📈 对比分析**

在统一拆分和评估指标下对 25+ 模型进行比较：预测任务以 MAE/RMSE 评估，分类与异常检测以 Precision@R10（召回 10%）衡量。结果显示即使是最强模型（如 DLinear、VSFormer）仍难以捕捉细微故障，预测精度与异常检测性能高度相关；预训练基础模型在 5‑shot 上表现可观但总体仍落后于专用模型。

**⚠️ 局限性**

局限性：①数据仅聚焦于排气系统，未覆盖其他关键系统；②故障标签基于维护日志，可能存在噪声与遗漏；③缺乏跨机型迁移和少样本学习的深入探讨；④任务划分与评估方法仍可能无法完全模拟真实运维环境。

---

## 729. When uncertainty quantification fails: structural vulnerabilities and corrections for selective prediction under clinical class imbalance

**arXiv ID:** 2610.37561 | [PDF](https://arxiv.org/pdf/2610.37561v1)

**作者:** Walter Endrizzi `[一作]` (Fondazione Bruno Kessler), Venet Osmani `[通讯]` (Queen Mary University of London)

**关键词:** `Machine Learning` `Classification` `Anomaly Detection` `Explainability and Interpretability` `Data-Centric Learning` `Contrastive Learning` `Supervised Fine-Tuning` `Reinforcement Learning` `Biomedical Data` `Tabular` `Time Series` `Alzheimer's Disease` `Electronic Health Records` `Review/Survey Paper`

**🎯 论文内容**

本文研究了在临床类别严重不平衡的预测任务中，选择性预测系统的标准不确定度量失效，并提出一种阈值对齐的混淆熵 H_τ 与类条件拒绝 CCR 的组合框架，以纠正少数类被错误拒绝的问题。

**💡 创新点**

创新点在于：①设计了阈值感知的熵函数 H_τ，能够将最大不确定度对齐至临床决策阈值；②提出了类条件拒绝机制 CCR，按预测类别内排序，消除少数类的绝对不确定度偏差；③引入了类分层风险覆盖曲线 ccAUGRC，避免全局指标掩盖子群风险。

**🔧 技术方法**

技术上使用异步集成学习（五种基学习器）、Platt 归一化、阈值映射重构、类条件排序与阈值对齐的不确定度评分，以及类分层的 AUC 风险覆盖评估。

**📊 数据集**

实验基于 56 条合成条件和三组真实神经退行性疾病队列（多发性硬化 MS π=0.13、帕金森 PD π=0.39、阿尔茨海默 AD π=0.54）。

**📈 对比分析**

与标准 Shannon 熵、分类边际、全局与类条件的多种基线比较，结果表明 H_τ+CCR 在少数类保持高敏感度、实现风险平衡，并在 ccAUGRC 评估中显著优于其他方法。

**⚠️ 局限性**

局限性包括仅适用于二分类任务、需预设决策阈值、对模型校准质量敏感、集成计算成本较高，以及未在单模型深度学习框架中验证。

---

## 730. GroundControl: Satellite-Conditioned Multi-View Diffusion for Reconstruction-Free Novel View Synthesis

**arXiv ID:** 2610.37545 | [PDF](https://arxiv.org/pdf/2610.37545v1)

**作者:** Akshay Krishnan `[一作]` (Niantic Spatial), Daniyar Turmukhambetov `[通讯]` (Niantic Spatial)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Autonomous Driving` `Transformer` `Diffusion model` `Score-based Model` `Prompt Engineering` `Contrastive Learning` `Ray-guided cross-attention` `DDIM sampling` `Dual CFG` `Image` `Point Cloud`

**🎯 论文内容**

提出一种基于卫星图像条件的多视角扩散模型，用来在户外场景中从稀疏街景图像生成新的逼真视角，同时保持空间布局一致性。

**💡 创新点**

核心创新包括：将卫星图像以光线引导交叉注意力（RGCA）融入预训练的多视角扩散网络；使用双重分类器无监督引导实现对街景与卫星信息的独立权重控制；仅微调轻量化注意力层，保留原始生成先验。

**🔧 技术方法**

使用SEVA多视角扩散框架、DINOv3编码卫星图像、ray-guided cross-attention、DDIM采样、双CFG、Plücker射线与时间步嵌入等技术。

**📊 数据集**

在nuScenes、Argoverse-v2、Metropolis、VIGOR等地理标注的多视角数据集以及DL3DV作为离散分布测试集上进行训练与评估；卫星图像来源于公开遥感数据。

**📈 对比分析**

与仅使用街景的SEVA基线以及仅基于卫星的Sat2Density比较，实验表明GroundControl在PSNR、SSIM和语义IoU上显著提升，尤其在视角不重叠时更为明显，同时在多视角一致性上也取得更低的TSED并提高一致视图比例。

**⚠️ 局限性**

依赖精确的地理定位和卫星与街景的粗对齐；对卫星图像分辨率和遮挡情况敏感；在极端遮挡或复杂结构场景中仍可能出现空间布局不准的误差。

---

## 731. Attention as Lattice Gauge Theory: Relations Interact Only Through Wilson Loops

**arXiv ID:** 2610.37656 | [PDF](https://arxiv.org/pdf/2610.37656v1)

**作者:** Aakash Sarkar `[一作]` (University of California), Motoshige Sato `[通讯]` (University of California)

**关键词:** `Machine Learning` `Anomaly Detection` `Protein Structure Prediction` `Graph Neural Network` `Transformer` `Contrastive Learning` `Diffusion model` `Score-based Model` `Flow-based Model` `Rectified Flow` `Neural Radiance Field` `Auto Encoder` `Generative Adversarial Network` `Graph` `Physics Related`

**🎯 论文内容**

提出基于实体自由度的 Gauge 对称性约束下的边注意力机制——Holonomy Attention，并证明其为唯一可行的非交换更新形式；在此基础上完成关系补全、异常检测与社区划分。

**💡 创新点**

创新点在于通过 Gauge 自由度推导出唯一的边更新规则，首次将 Wilson 循环、知识图谱嵌入、AlphaFold 三角更新等多领域方法统一为一个等变注意力框架，并在无监督环境下实现高精度关系推断与异常识别。

**🔧 技术方法**

技术包括 Gauge 理论、Wilson 循环、等变注意力（edge‑wise update 为 Ψ(H)T_ij）、极化投影 Π、正交投影、S4 群同步化实验、对角化与重加权策略。

**📊 数据集**

实验数据主要使用随机生成的 S4 群同步化图（14 个实体，稀疏/密集连边，破坏比例 p∈{0.1,0.2,0.3,0.45}），以及无标签两聚类图；未使用公开图数据集。

**📈 对比分析**

方法对比：与 Singer 特征向量法、循环边消息传递等同步算法进行对比；在 p=0.2 的稀疏图上 Holonomy Attention 达到 0.89 的准确率，远高于单路径 0.63；对三角一致性的 AUC 为 0.99，显著高于可交换分数 0.76；在社区划分实验中 AUC 达 0.98。

**⚠️ 局限性**

局限性：目前仅使用预加载的离散群元素，未学习动态 Gauge 场；实验规模有限，缺乏大规模真实图的验证；对连续 Lie 群的推广尚待探索。

---

## 732. GEPA-Curator: Curated Learning for Self Improving Agents

**arXiv ID:** 2610.37651 | [PDF](https://arxiv.org/pdf/2610.37651v1)

**作者:** Zack Yu `[一作]` (University Of California Berkeley), Amir Gholami `[通讯]` (University Of California Berkeley)

**关键词:** `Computation and Language` `Reinforcement Learning` `Optimization` `Data-Centric Learning` `Autonomous Driving` `Robotic Intelligence` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Agentic AI` `Text` `Benchmark`

**🎯 论文内容**

在现有反思式提示优化框架上加入了任务筛选与成功轨迹检索，形成GEPA‑Curator；

**💡 创新点**

通过主动排除噪声任务并提供过去成功示例，显著提升了提示优化的信号质量；

**🔧 技术方法**

采用任务筛选（Task Curation）和先验成功检索（Prior Success Retrieval）两大机制，结合LLM反射器与Curator；

**📊 数据集**

在WebArena（GitLab、Shopping、Shopping Admin、Reddit）、ALFWorld、AppWorld与τ‑bench四个基准上进行评估；

**📈 对比分析**

相较于GEPA和ACE基线，平均提升20–21%成功率，最大单域提升达7.4%，效率版仍保持近似性能；

**⚠️ 局限性**

依赖历史记录和Curator判断，成本与模型规模相关，且对极端噪声任务的识别仍存在一定局限。

---

## 733. UPP$^{\text{cc}}$ is Not Closed Under Intersection

**arXiv ID:** 2610.37634 | [PDF](https://arxiv.org/pdf/2610.37634v1)

**作者:** Mark Bun `[一作]` (Boston University), Mandar Juvekar `[通讯]` (Boston University)

**关键词:** `Computational Complexity` `Optimization` `Computational Efficiency` `Representation Learning` `Explainability and Interpretability` `Federated Learning` `Contrastive Learning` `Approximation Theory` `Gaussian Splatting` `Score-based Model` `Diffusion model` `Taylor Approximation` `Symmetrization` `Tabular` `Graph` `Benchmark` `Review/Survey Paper`

**🎯 论文内容**

本文构造了一个显式的 ±1 矩阵 A，尺寸为 2^Θ(d²) × 2^Θ(d²)，其 sign rank 为 Θ(d)，但 A 的两两并行版本 A∘A 的 sign rank 为 2^Θ(d)。这表明无误差通信复杂度类 UPP 在交集操作下不是闭合的。

**💡 创新点**

创新点在于：① 将 Forster 的平均余量法推广到连续球面情形，并通过旋转对称性得到对称化函数 H 的高阶导数有界；② 证明低阶多项式无法在除去坐标轴附近的域上符号表示“左下象限”指示函数，从而得到 H 的系数上界；③ 将上述连续结果鲁棒化，允许对称化在决策边界附近出现错误，最终得到离散问题的强下界；④ 通过显式取整数半空间以及随机球面点的离散化，给出了具体的示例。

**🔧 技术方法**

主要技术包括：Forster 的平均余量与 sign rank 关联；球面平均化（symmetrization）与 Gegenbauer 多项式展开；Taylor 近似与高阶导数估计；逼近理论（Newman 近似与 Sherstov 的组合技巧）用于处理多变量阈值函数；以及从连续到离散的圆形映射与误差传播。

**📊 数据集**

论文使用的“数据集”主要是显式构造的矩阵：1）整数半空间矩阵 _M,d（取 M≈2^c d）；2）由随机均匀球面点生成的点积矩阵 G。没有使用外部真实数据集。

**📈 对比分析**

与现有方法对比：传统 Forster 方法仅能得到对数级别的下界；本文实现了指数级别（2^Ω(d)）的下界；在阈值门电路、学习半空间等应用场景中，证明了交集可以导致指数级的复杂度提升。实验结果与理论一致，显式矩阵构造使得结论可直接验证。

**⚠️ 局限性**

局限性包括：① 该分离仍非最优，未知是否存在 U(F) = O(log n) 且 U(F∘F) = Ω(n) 的示例；② 对于交集运算的更一般化（k>2）下的精确下界仍未完成；③ 对于 approximate sign rank 的进一步提升与上界仍是开放问题；④ 目前的技术主要依赖球面对称性，可能难以直接推广到非球面或更复杂的输入空间。

---

## 734. EnvSched: Environment-Aware Scheduling for Rollout-Efficient Agent Reinforcement Learning

**arXiv ID:** 2610.37609 | [PDF](https://arxiv.org/pdf/2610.37609v1)

**作者:** Sihan Yang `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `Machine Learning` `Reinforcement Learning` `Graph Neural Network` `Reinforcement Learning` `Graph Embedding` `Greedy Algorithm` `Graph` `Tabular` `Benchmark`

**🎯 论文内容**

在可执行工具环境中，提出了基于环境与任务对的调度框架EnvSched，能够在不额外生成演练的情况下，动态选择高信息量任务并保持环境多样性。

**💡 创新点**

创新点在于将环境与任务对视为数据单元，构建双关系图（内环境与跨环境），利用奖励方差预测任务信息量，并将覆盖率与利用率在同一调度步骤内协同优化。

**🔧 技术方法**

主要技术包括图嵌入与相似度计算、奖励方差预测、基于相邻节点的图先验融合、以及覆盖率与利用率分配的贪心批次构造。

**📊 数据集**

使用了EnvScaler可执行环境池（2550个任务、51个环境）以及Qwen3系列模型，并在BFCL‑v3、τ‑Bench、ACEBench‑Agent三大基准上评测。

**📈 对比分析**

与随机、DPS、LEEPS等基线对比，EnvSched在200步预算下平均提升BFCL‑v3、τ‑Bench、ACEBench‑Agent约3–5分；在仅40步预算下已匹敌或超越随机，表现出高效率。

**⚠️ 局限性**

局限在于依赖预先提取的工具与状态相似度、对图邻居规模的超参数敏感，且在环境规模急剧扩张时可能需要更多计算资源；未考察非可执行或动态改变的环境情况。

---

## 735. TANDEM: Balancing Centralized and Decentralized Critics to Address the Information Gap in Complex Multi-Agent Environments

**arXiv ID:** 2610.37592 | [PDF](https://arxiv.org/pdf/2610.37592v1)

**作者:** Alireza Saleh Abadi `[一作]`, Prashant Doshi `[通讯]`

**关键词:** `Multiagent Systems` `Reinforcement Learning` `Transformer` `Graph Neural Network` `Reinforcement Learning` `Auto Encoder` `Contrastive Learning` `Tabular` `Time Series` `Sequential`

**🎯 论文内容**

提出一种名为TANDEM的训练框架，利用单一中心化Critic的双重读取（全局读与局部读），通过参数无关门控根据两种读取的优势符号一致性动态混合优势，并配合共享目标训练，从而在CTDE中减少信息缺口；

**💡 创新点**

创新点包括：①使用同一Critic实现全局与局部两种读取，避免额外网络；②引入参数无关门控（基于优势符号不一致的正比调制），实现按需撤回或恢复中心化信息；③共享目标让局部读取逐步逼近全局读取；④在正式的TAO‑Dec‑POMDP框架下对信息缺口进行形式化；

**🔧 技术方法**

技术细节包括：MAPPO actor‑critic（PPO + GAE）；Critic采用基于实体注意力编码的全局/局部读取；门控采用tanh+median绝对偏差归一化；共享目标实现两种读取与同一回报目标的同时回归；

**📊 数据集**

数据集与环境：Vectorized Multi‑Agent Simulator（VMAS）中的Discovery（部分可观测）与Sampling（观测噪声）任务；3x3网格Wildfire Suppression任务（包含Agent/Task openness）；每种任务均设置gap与无gap对照；

**📈 对比分析**

方法比较：对齐六种主流CTDE基线（MAPPO、COMA、QMIX、VDN、MAVEN、MADDPG），在9种实验设置中，TANDEM在8/9场景取得最高执行回报，且在5/9场景显著优于所有基线（p<0.05，Bonferroni校正），门控与共享目标是关键改进点；

**⚠️ 局限性**

局限性：仅针对Agent/Task openness，未覆盖Frame openness；门控仅依据优势符号一致性，可能无法纠正两者均误差的情况；实验仅分离单一信息缺口类型，未在多重缺口混合或更大规模/标准benchmark上评估；

---

## 736. GNBG-C: A Unified and Configurable Benchmark Generator for Constrained Continuous Optimization

**arXiv ID:** 2610.37590 | [PDF](https://arxiv.org/pdf/2610.37590v1)

**作者:** Ali Ghasemzadeh `[一作]` (University of Hong Kong), Amir H. Gandomi `[通讯]` (University of Technology Sydney)

**关键词:** `Neural and Evolutionary Computing` `Optimization` `Benchmark`

**🎯 论文内容**

本文提出 GNBG-C，一个将可行域作为可配置对象的受限连续优化基准生成器，在保持已知最优点可行性的同时，可灵活调节可行域大小、边界形状、变量耦合、违约尺度与冗余度。

**💡 创新点**

创新点在于将可行域从传统固定约束解耦出来，构建可调节的约束层并提供诊断信息，实现在保持目标景观不变的前提下系统地研究约束诱发的搜索难度。

**🔧 技术方法**

技术实现包括中心化约束函数 g_j(x)=ψ_j(z_Ij(x);θ_j)−κb_j、κ 的可行域比例校准、不同约束族（线性、凸/非凸二次、多项式、超越函数）、变量支持率、数值尺度控制和冗余约束的自动生成。

**📊 数据集**

使用 GNBG 目标函数（已知全局最优点）生成 8 种可行域配置的实例，每种配置 15 个样本，维度为10，并在此基础上对 8 种受限优化算法进行评估。

**📈 对比分析**

通过 8 个算法（DE、PSO、GA、SRES、ISRES、SciPy‑DE 等）在 2,500、5,000、10,000 次评估处测量可行性发现速度与最佳可行目标误差，实验显示不同配置显著区分算法行为，SRES 在最终误差和实现率上表现最佳，但无统一“最优”排序。

**⚠️ 局限性**

局限性包括仅针对已知最优点的模式、仅处理不等式约束、固定维度（10）、未覆盖等式/离散变量或动态/噪声约束，且需进一步验证在真实工程问题中的泛化能力。

---

## 737. REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation

**arXiv ID:** 2610.37584 | [PDF](https://arxiv.org/pdf/2610.37584v1)

**作者:** Yuxi Huang `[一作]` (University of Electronic Science and Technology of China), Shuo Shang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `Information Retrieval` `Recommendation System` `Transformer` `Contrastive Learning` `Knowledge Distillation` `Generative Adversarial Network` `Auto Encoder` `Text` `Sequential`

**🎯 论文内容**

提出了 REGER 框架，结合生成语义检索预训练、交互历史中的过渡结构注意力以及基于 SID 前缀树的层次协同偏好蒸馏，显式建模商品间的语义与协同关系；

**💡 创新点**

创新点在于：①通过生成语义检索（GSR）预训练让 SID 嵌入直接继承语义相似性；②设计 Item Transition Structure Attention (ITSA) 将全局与局部注意力融合，强化邻近商品间的有向转移；③构造 SID 前缀树的协同偏好分解与层次化协同偏好蒸馏 (HCPD)，实现对协同偏好的精细化指导；

**🔧 技术方法**

采用残差量化的 RQ‑VAE 生成 SID、T5 变换器的编码器‑解码器结构、生成语义检索 (GSR)、ITSA 结构以及基于 DuoRec 的 HCPD 蒸馏技术；

**📊 数据集**

在 Amazon Reviews 2014 的 Beauty、Sports、Toys 三大子域以及 2023 的 Instruments、Scientific、Games 六个子域进行实验；

**📈 对比分析**

与判别式推荐（SASRec、FDSA、BERT4Rec、S³‑Rec、DuoRec）、生成式推荐（TIGER、LETTER、UNGER、ReSID、ETEGRec、DIGER、ActionPiece、TIGER‑SST）以及混合式推荐（LIGER、MemGen‑GR）在 Recall@5/10、NDCG@5/10 进行对比，REGER 在所有数据集和指标上均超过基线，特别是加入 HCPD 后 Recall@10 与 NDCG@10 分别提升至 0.093 与 0.049，显著优于其他方法；

**⚠️ 局限性**

仍然依赖固定长度 SID，需额外预训练与蒸馏阶段，且对极端稀疏或新颖商品的表现尚未充分验证，实验仅覆盖 Amazon 领域，跨域适用性需进一步研究。

---

## 738. Gym-Anything-RT: Scaling Real-Time Environments for Vision-Language Agents

**arXiv ID:** 2610.37567 | [PDF](https://arxiv.org/pdf/2610.37567v1)

**作者:** Pranjal Aggarwal `[一作]` (Carnegie Mellon University), Sean Welleck `[通讯]` (Carnegie Mellon University)

**关键词:** `Machine Learning` `Reinforcement Learning from Human Feedback` `Computational Efficiency` `Data-Centric Learning` `Robotic Intelligence` `Transformer` `Large Language Model` `Vision Language Model` `Agentic AI` `Prompt Engineering` `World Model` `Multimodality` `Text` `Image` `Video`

**🎯 论文内容**

本文提出了Gym-Anything-RT框架，通过LLM编码器自动生成实时可视化小游戏，并构建了100个实时与非实时对应任务，系统评估现有视觉语言模型的实时交互能力。

**💡 创新点**

创新点在于利用创作者-审计者管线自动化生成可调时钟的小游戏，并提供可控制实时/暂停/慢速三种运行模式，能够精准分离模型的速度瓶颈与能力不足。

**🔧 技术方法**

使用技术包括LLM编码代理（GPT-6 Astra等）、Gym-Anything基础设施、Oracle解算器、基于时钟控制的评估堆栈以及沙箱内代码执行等。

**📊 数据集**

数据集为自动生成的100个实时小游戏及其非实时对应版本，包含五个难度层和两种交互模式，总计2000个配置。

**📈 对比分析**

通过比较模型在实时、慢速、暂停模式下的通过率、平均求解时间和成本，结果显示GPT-6 Astra在实时任务仅9%通过，暂停后提升至75%，且速度和成本成为主要瓶颈。

**⚠️ 局限性**

局限性包括模型在实时场景下极慢且通过率低，即使开启代码能力也无法完全弥补；实验集仅涵盖100个小游戏，难以覆盖更广泛的真实世界交互场景。

---

## 739. ReferStream: Interactive Subject Injection for Streaming Video Generation

**arXiv ID:** 2610.37566 | [PDF](https://arxiv.org/pdf/2610.37566v1)

**作者:** Shuolin Xu `[一作]` (Bournemouth University), Xiaobin Hu `[通讯]` (National University of Singapore)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Auto Encoder` `Diffusion model` `Prompt Engineering` `Retrieval-Augmented Generation` `Video` `Image` `Text` `Benchmark`

**🎯 论文内容**

在流式视频生成中，实现了在不重新训练模型、也不修改已生成帧的前提下，按用户指定时间动态注入新的视觉参考，并将其融入正在生成的视频。

**💡 创新点**

创新点在于：①将参考图像先在当前场景中进行联合构造（Joint Construction）以获得与场景一致的特征；②在缓存中短暂写入这些特征后立即恢复原始缓存（Transient Injection & Restoration），从而避免对后续生成产生长期干扰。

**🔧 技术方法**

采用冻结的 LongLive2 生成器，利用其自动回归视频扩散架构；通过 VAE 编码参考图像、DiT 进行特征提取、并在缓存层进行临时写入与恢复；评估使用 CLIP、GME‑T、VBench、AMT‑S 等标准视觉指标。

**📊 数据集**

构建了新的 ReferStream 基准数据集，包含 48 条 InShot 和 20 条 MultiShot 视频案例，配有按时间排序的参考到达、指令演变和目标主体标注；还使用 OpenS2V‑Nexus 作为对比基准。

**📈 对比分析**

与 LongLive‑2.0 基线以及 VACE、Phantom、HOMIE 等离线方法对比，ReferStream 在文本遵循、新旧主体相似度（New‑ID、Exist‑ID）和计数准确度等方面均取得最高分，视频质量指标（美学、运动流畅度）表现可与顶级方法相当或略低。

**⚠️ 局限性**

主要局限在于：①在提升旧主体保留的同时，新主体的相似度会略有下降；②依赖自动检测与特征匹配，可能未捕捉到所有身份漂移或漏检；③缺乏大规模人类评估，评估结果对实际用户体验的映射仍需进一步验证。

---

## 740. Kinematic Constraint-Referenced Evaluation for Articulated Object Manipulation

**arXiv ID:** 2610.37550 | [PDF](https://arxiv.org/pdf/2610.37550v1)

**作者:** Baptiste Bonin `[一作]` (Université Laval), Audrey Durand `[通讯]` (Université Laval)

**关键词:** `Robotics` `Robotic Intelligence` `Explainability and Interpretability` `Reinforcement Learning` `Optical Flow` `Contrastive Learning` `Point Cloud` `Tabular` `Time Series`

**🎯 论文内容**

提出并评估了一种基于关节轴的指令评估方法CRS，用来判断机械臂的运动命令是否符合被操作对象的运动约束。

**💡 创新点**

创新点在于：不需要演示或学习的评判器，仅利用关节轴模型将指令分解为沿轴推进和离轴浪费两部分，实现对“动手”动作的量化评价，并能为失败原因提供可解释性。

**🔧 技术方法**

使用了刚体运动学投影、命令式仿真检测、基于进度与浪费的双通道评估，以及对不同阈值的门控验证。

**📊 数据集**

在 ManiSkill 数据集下的 GAPartNet 74 个模拟关节对象（门、抽屉、龙头）上进行实验。

**📈 对比分析**

与传统的二元成功率、Δq_real、DTW、MetaFine 稳定性等指标对比，CRS 在检测转移失败、指令错误方向、离轴命令等方面具有更高的分辨率，能够将错误原因分门别类，并在误判率上显著低于单一二元指标。

**⚠️ 局限性**

局限性包括：假设握持为刚性抓取，适用于单自由度关节，未能测量离轴浪费的实际物理代价；仅在仿真中验证，缺乏真实机器人实验。

---

## 741. Can Discrete Tokens Serve as Semantically Lossless Representations of Visual Signals?

**arXiv ID:** 2610.37662 | [PDF](https://arxiv.org/pdf/2610.37662v1)

**作者:** Jialv Zou `[一作]` (Huazhong University of Science and Technology), Xinggang Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Representation Learning` `Compression` `Transformer` `Vision Language Model` `Contrastive Learning` `Supervised Fine-Tuning` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出一种新的离散视觉表示方法——Factorized Vector Quantization（FVQ），并在工业规模多模态模型中验证其有效性

**💡 创新点**

创新点在于将多维稠密投影与对称平行码本结合，形成紧致框架（tight‑frame）保证每个码本获得完整特征视角，从而实现比传统RVQ、PQ更高的有效位预算和更好语义容量

**🔧 技术方法**

核心技术包括稠密投影矩阵、紧致框架分析/合成、对称平行量化、离散索引映射到词嵌入、工业规模预训练与对齐策略

**📊 数据集**

使用 100B 训练标记（含 STEM、OCR、图文对、图文生成等）进行量化器训练与对齐；预训练和监督微调使用 60B/30B 标记；在 LLaVA-OneVision 4B 任务上做实验；评测数据集包括 MMMU、MMStar、RealWorldQA、MathVista、OCRBench、DocVQA、InfoVQA、ChartQA、CharXiv、AI2D 等

**📈 对比分析**

与连续表示 Qwen3‑ViT 以及其他离散量化器（单码本 VQ、RVQ、PQ）进行对比，FVQ 在 9 指标平均上比 UniAR 提升 22.8%，比 Emu3‑Chat 提升 59.2%，最终与 Qwen3‑VL‑8B 的表现相当，尤其在 OCR、文档和图表细粒度理解任务上显著超越其他离散模型

**⚠️ 局限性**

局限性包括：仍需要海量训练数据和算力；量化器训练和对齐过程较为复杂；对其他视觉编码器的通用性虽已初步验证，但在极端高分辨率或特殊视觉场景下的细节捕捉仍可能受限

---

## 742. Metacognitive Decoding: Efficient Inference for Diffusion Language Agents

**arXiv ID:** 2610.37642 | [PDF](https://arxiv.org/pdf/2610.37642v1)

**作者:** Mingyu Zhao `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `Computation and Language` `Computational Efficiency` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Large Language Model` `Reinforcement Learning` `Agentic AI` `Prompt Engineering` `Text` `Benchmark`

**🎯 论文内容**

提出一种元认知控制器，通过动态调整扩散语言模型在代理交互中的解码策略，显著降低推理 FLOPs 并保持或提升任务准确率。

**💡 创新点**

创新点在于结合置信度对齐策略和 Flux 控制的级联政策，依据模型置信度与交互状态实时决定哪些 tokens 进行最终确定以及计算窗口长度，从而消除冗余计算。

**🔧 技术方法**

使用了块扩散（block diffusion）机制、两阶段控制器（confidence‑alignment + flux）、基于奖励的自监督训练（GRPO）以及对扩散模型的轻量级插件化。

**📊 数据集**

评估数据集包括 BFCL v3（工具调用）、HotpotQA、2WikiMultiHopQA、MuSiQue 等多跳推理与工具使用基准。

**📈 对比分析**

通过与 LLaDA2.0-mini、LLaDA‑8B‑Tools、DLLM‑Searcher、Fast‑dLLM 以及无控制器的 LLaDA‑8B (SFT) 进行对比，平均 FLOPs 降低约 20–30%，且在多数基准上保持或提升任务准确率。

**⚠️ 局限性**

局限性：控制器需针对不同扩散模型进行适配，当前实验仅在特定规模模型上验证，未展示在更大规模或其他扩散架构上的可扩展性与泛化能力。

---

## 743. Know, then Draw: A New Learning Paradigm for Visual Generation

**arXiv ID:** 2610.37626 | [PDF](https://arxiv.org/pdf/2610.37626v1)

**作者:** Yucheng Mao `[一作]` (University of California, San Diego), Hanwen Jiang `[通讯]` (Adobe Research)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Data Synthesis` `Transformer` `Diffusion model` `Large Language Model` `Score-based Model` `Vision Language Model` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出一种分阶段学习框架，将自回归预训练用于视觉知识获取，随后冻结自回归网络并在其上训练像素扩散渲染器，实现从文本到图像的高质量生成；

**💡 创新点**

创新点包括：①将自回归和扩散在预训练与后训练中分离并针对不同目标进行优化；②引入仅用语言监督训练的Semantic‑LM语义分词器，无需图像解码器；③基于阶段特定的规模法则确定计算分配，显著提升计算效率；④在约6,600 H100 GPU‑小时内训练出3B参数模型，取得GenEval 0.86、DPG‑Bench 87.78的优异表现。

**🔧 技术方法**

使用的技术包括Transformer自回归模型、IBQ码表分词、语义分词器、像素级扩散模型（流匹配损失）、位置指令token、基于C=6ND的计算近似、logit‑normal采样、时间步嵌入及自适应归一化层。

**📊 数据集**

使用公开的大规模文本‑图像对数据集（如LAION、CC12M等），在预训练和后训练阶段分别训练自回归和扩散模型。

**📈 对比分析**

通过与从零开始训练的像素扩散基线、不同分词器（Recon、Semantic‑Recon、Semantic‑LM）进行对照实验，评估损失、生成质量（GenEval、DPG‑Bench）。结果表明自回归预训练后扩散在相同模型/数据规模下损失更低、生成质量更佳；3B模型在GenEval上达0.86、DPG‑Bench上达87.78，仅需约6,600 H100 GPU‑小时，比Z‑Image‑Base低48倍。

**⚠️ 局限性**

局限性包括：①依赖冻结的分词器，未验证更大规模或多分词器的鲁棒性；②推理速度和显存使用未系统评估；③仅在文本‑图像条件下测试，未探讨多模态或跨语言情况；④对超参数和训练细节的敏感性仍需进一步研究；⑤缺乏对生成多样性、长文本理解等方面的深入分析。

---

## 744. RESPLE2: Inertial-Informed Recursive Non-Uniform Spline Estimation for Multi-Sensor Odometry with Global Refinement

**arXiv ID:** 2610.37602 | [PDF](https://arxiv.org/pdf/2610.37602v1)

**作者:** Ziyu Cao `[一作]` (Linköping University), Kailai Li `[通讯]` (Linköping University)

**关键词:** `Robotics` `Pose Estimation` `Optimization` `Autonomous Driving` `Robotic Intelligence` `Simultaneous Localization and Mapping` `Extended Kalman Filter` `Levenberg-Marquardt Global Optimization` `B-spline` `Point Cloud` `Image` `Time Series` `Sequential`

**🎯 论文内容**

开发了RESPLE2，一种递归非均匀B样条框架，用于多传感器（LiDAR、相机、IMU、腿关节）融合的6自由度实时位姿估计。

**💡 创新点**

创新点在于利用结点夹持边界属性实现IMU驱动的精确控制点预测，依据运动与测量自适应结点放置，并提供基于循环闭合的全局B样条优化以纠正漂移。

**🔧 技术方法**

采用连续时间B样条、扩展Kalman滤波（IEKF）、IMU积分、点云投影、视觉特征跟踪、腿关节动力学、VoxelMap局部地图、iBTC循环闭合、以及Levenberg-Marquardt全局优化。

**📊 数据集**

在GrandTour、JEPLO、PPS Tracked Vehicle、HOPPER、HANDI等公开与自制数据集上进行评估，涵盖四足、履带、混合足飞行器和手持平台。

**📈 对比分析**

与Coco-LIC、FAST-LIVO2、CTE-MLO、Point-LIO等state-of-the-art方法在不同传感器配置与极端动态下进行RPE/APE对比，RESPLE2在大多数场景下实现更低误差并保持实时性能。

**⚠️ 局限性**

仍存在极端惯性饱和或视觉/雷达失效时的鲁棒性挑战、循环闭合依赖、对多传感器时间同步和硬件资源的需求。

---

## 745. EgoThumb: A Vision-Driven Supernumerary Robotic Thumb for Autonomous Grasp and Holding Assistance

**arXiv ID:** 2610.37582 | [PDF](https://arxiv.org/pdf/2610.37582v1)

**作者:** Jianyu Hou `[一作]` (University of California San Diego), Hao Su `[通讯]` (Sudo AI GmbH)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Diffusion model` `Vision-Language-Action Model` `Optical Flow` `Contrastive Learning` `Image` `Video` `Time Series`

**🎯 论文内容**

设计并实现了 EgoThumb——一种轻量化、双绞线驱动的超数额机器人拇指，并通过视觉模仿学习实现自主抓握与放置，支持单手完成传统需要双手的“握持-操作”任务。

**💡 创新点**

创新点包括：将动力传递迁移至前臂以降低末端重量、采用双绞线主动控制弯曲与伸展实现精确关节定位、在掌侧安装鱼眼相机获取自我视角且无遮挡、将抓取时机和手部运动通过端到端视觉模仿学习自动推断，从而实现无需手部运动触发的自主协作。

**🔧 技术方法**

核心技术包括：双绞线传动与前臂电机模块、掌侧鱼眼摄像头与异步影像捕获、Diffusion Policy 的视觉模仿学习框架（CLIP‑ViT 预训练特征 + Conditional‑Unet1D 生成网络）、实时通信中间件与时延匹配算法、以及专用的脚踏遥操作接口用于数据收集。

**📊 数据集**

数据集：在三位不同手型的示范者上收集约 50 条演示序列（每个任务），随后扩增至 105 条演示，涵盖 9 种室内环境（光照、背景）并在测试中加入两种全新环境。每条序列包含 240×320 RGB 图像、关节角度和遥操作目标动作。

**📈 对比分析**

与基线（仅使用遥操作或低维手部运动触发）比较，EgoThumb 在 4 种真实世界任务（拾起方块、拧盖子、搅茶、转移球）中取得平均 0.9 的成功率；在不同手型和扩展的初始/目标区域中仍保持 0.73–0.87 的成功率；在见过与未见过环境中保持 0.8 以上的成功率，表明具备良好的泛化能力。

**⚠️ 局限性**

局限性包括：对长时序与大空间变化任务（如转移球）性能下降、对快速手部运动产生模糊、掌侧相机突出的佩戴不适、未测试多角度手部接近的鲁棒性、以及仅基于 10 次二进制成功率评估，缺乏细粒度协调质量与统计分析。

---

## 746. As the Story Unfolds: Watching a Film and Identifying Characters as a Human Does

**arXiv ID:** 2610.37579 | [PDF](https://arxiv.org/pdf/2610.37579v1)

**作者:** Zhongrui Gui `[一作]` (University of Oxford), Andrew Zisserman `[通讯]` (University of Oxford)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Object Detection` `Object Tracking` `Recommendation System` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Reinforcement Learning` `Vision Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Video` `Text` `Audio` `Multimodality`

**🎯 论文内容**

本文提出并实现了从电影对话中自动识别并命名主角的任务，并在此基础上设计了在线构建角色库的流水线，完成了从视频帧到角色命名再到角色跟踪的端到端处理。

**💡 创新点**

创新点包括：①首次将角色命名定义为单镜头内基于文字和视觉线索的多模态推理任务；②构建了大规模的Movie Character Naming（MCN）数据集，包含22k多选问题和20部影片的首现/首名时间戳；③采用色彩视觉提示和两阶段SFT+RL细调的多模态LLM，解决了颜色依赖的快捷方式；④提出了流式角色库更新机制和错误修正策略；⑤设计了mTIoU、PCR等新评测指标。

**🔧 技术方法**

技术手段包括：多模态大语言模型（Qwen3VL-4B-Thinking）、视觉提示（彩色框+字幕）、Speaker Diarization + TalkNet对齐、Parakeet转录、面部/语音轨道提取、SFT + GRPO强化学习、颜色随机置换增强、面部/语音聚类、面部/声音嵌入匹配、统计投票命名、错误校正三层机制。

**📊 数据集**

使用数据集：MovieNet、Movie101（作为基础）、自建MCN（407部电影，377部训练、30部测试、20部首现测试），包含23,792个多选命名实例和574个首现/首名时间戳。

**📈 对比分析**

与随机基线、开源Qwen3VL/3.5、以及专有Gemini-3系列进行对比。开源Fine‑Tuned Qwen3VL-4B-Thinking在MCN测试集上整体准确率达72.76%（已命名约81.06%），在未见电影上为64.78%；流式角色库在mTIoU_v/ n分别为84.90/54.40，PCR@5为63.00，均显著优于Gemini-3-Flash（51.81/30.92/35.00）和Gemini-3.1-Pro（49.05/30.10/44.00）。

**⚠️ 局限性**

局限性：仍与顶尖专有模型存在差距，主要受转录/说话人检测误差影响；对细微提示（如指点、手势）和多人物混合场景的推理不够鲁棒；角色库在大型多角色影片中易受背景噪声干扰；尚未实现完全自动化的错误回溯和全局优化，需进一步研究代理式或交互式框架。

---

## 747. The Noise Plane: A Diffusion Bridge Across Exponent and Dispersion

**arXiv ID:** 2610.37576 | [PDF](https://arxiv.org/pdf/2610.37576v1)

**作者:** Xuran Hu `[一作]` (Hong Kong University of Science and Technology), Wufan Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Restoration` `Transformer` `Convolutional Neural Network` `Diffusion model` `Score-based Model` `Image` `Biomedical Data` `Computed Tomography` `Ultrasound` `Review/Survey Paper`

**🎯 论文内容**

提出了p-Bridge测量索引扩散桥模型，统一处理Tweedie噪声并实现指数条件的去噪网络。

**💡 创新点**

通过构建噪声平面（测量指数ν与噪声指数p），推导共享逆均值更新与单位偏差目标，并用单一指数条件网络覆盖整个Tweedie族，无需为每个指数单独训练模型。

**🔧 技术方法**

采用Diffusion Bridge与指数条件，使用逆均值更新、单位偏差损失、时间与指数嵌入的UNet架构，以及基于局部统计的噪声指数与测量指数估计方法。

**📊 数据集**

使用BSDS500、DIV2K训练集进行合成实验；FMD荧光显微镜多配置数据；miniSAR、FARAD‑X、TerraSAR‑X SAR实测数据进行真实场景验证。

**📈 对比分析**

与BM3D、VST+BM3D、Noise2VST、VIRNet、I^2SB、ResShift、FBI‑Denoiser、EAMamba、Restormer、NAFNet、DDRM等多种基线在合成图像、显微镜和SAR数据上进行PSNR/SSIM/ratio指标比较；p-Bridge条件版在大多数设置下取得最高PSNR/SSIM，且跨指数鲁棒性接近oracle选择的5个专家模型。

**⚠️ 局限性**

对噪声指数估计误差有一定敏感性（需在[1,2]区间内）；不支持极低噪声指数（p≈0）或非Tweedie噪声；极暗或极亮区域估计误差较大；需要额外的指数估计步骤，增加推理复杂度；缺乏对高维多模态或非Tweedie噪声的直接适用性。

---

## 748. Learning Visuomotor Quadrotor Flight Through Dynamic Narrow Gaps via Differentiable Simulation

**arXiv ID:** 2610.37557 | [PDF](https://arxiv.org/pdf/2610.37557v1)

**作者:** Riccardo Bianco `[一作]`, Davide Scaramuzza `[通讯]` (University of Zurich)

**关键词:** `Robotics` `Reinforcement Learning` `Robotic Intelligence` `Reinforcement Learning` `Differentiable Simulation` `BPTT (Backpropagation Through Time)` `Path Gradient` `RK4 Integration` `Time-Varying Signed Distance Loss` `Image` `Video` `Point Cloud`

**🎯 论文内容**

训练了一种直接使用视觉和惯性测量的无人机控制策略，能够在没有教师、规划器或示范的情况下通过可微分仿真实现对移动窄缝的端到端学习，并成功在真实无人机上实现零拷贝动态窄缝穿越飞行。

**💡 创新点**

1) 设计了可微分的“移动缝”损失，将稀疏穿越事件转化为稠密路径梯度；2) 采用路径梯度（BPTT）而非分数函数梯度，大幅降低方差并实现几分钟内收敛；3) 提供了开源的可微分仿真框架；4) 在实际硬件上实现了从训练到飞行的闭环闭合。

**🔧 技术方法**

可微分仿真、BPTT路径梯度、基于视觉特征的CTBR（Collective Thrust + Body‑Rate）策略、RK4积分、时间可变签名距离损失、工程化视觉特征提取（gap corner projections）等。

**📊 数据集**

主要使用随机生成的仿真环境（1000个并行环境，随机位姿、gap角度、运动幅度等），以及高保真仿真和实际飞行实验；没有使用公开的公开数据集。

**📈 对比分析**

与PPO对比：在相同GPU资源下，BPTT使用的转移量少19×、训练时间少45.5×；在仿真中状态/视觉静止/移动任务的成功率分别为100%、98.9%、100%、99.3%；在真实世界10次飞行中成功率为80%；路径误差保持厘米级。

**⚠️ 局限性**

仅在平面摆动的移动缝下验证；依赖可微分模型与精确物理；视觉输入为工程特征而非原始图像；未测试动态不匹配、长时视野丢失、硬接触等情况；BPTT的记忆和计算成本随时间增长；缺乏对更复杂运动或自然开口的验证。

---

## 749. MuonGS: Pretraining LLMs with Relaxed Weight-Norm Control

**arXiv ID:** 2610.37647 | [PDF](https://arxiv.org/pdf/2610.37647v1)

**作者:** Athanasios Glentis `[一作]` (University of Minnesota), Mingyi Hong `[通讯]` (University of Minnesota)

**关键词:** `Machine Learning` `Optimization` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Contrastive Learning` `Mixture of Experts` `Text` `Physics Related`

**🎯 论文内容**

提出了MuonsGS优化器，通过对Muon's更新进行Gram–Schmidt投影消除径向分量，以控制权重范数增长；

**💡 创新点**

创新点在于用投影方式实现更轻量级的范数控制，理论上保证有效学习率累积至少为√T，从而在整个训练过程中保持探索能力；

**🔧 技术方法**

技术包括Muon's正交化、Gram–Schmidt投影、有效学习率累积分析以及对光滑尺度不变目标的收敛证明；

**📊 数据集**

实验使用FineWeb数据集训练160M与400M GPT-Style模型，以及Ultra‑FineWeb数据集训练135M活跃/1B总参数的MoE模型；

**📈 对比分析**

与MuonsH、MuonsW以及vanilla Muon进行比较，结果显示MuonsGS在长时间训练中达到更低的验证困惑度，匹配相同验证误差时估计可节省约15%的训练代币量；

**⚠️ 局限性**

局限性包括仅给出最坏情况的累积上界、缺乏对不同范数增长速率最佳选择的理论指导，以及对非光滑或非尺度不变目标的性能缺乏深入分析。

---

## 750. Perceiving and Predicting Dynamic Sports: Spatiotemporal Intelligence in Vision Models for Egocentric Tennis Video

**arXiv ID:** 2610.37619 | [PDF](https://arxiv.org/pdf/2610.37619v1)

**作者:** Tudor Braicu `[一作]` (Massachusetts Institute of Technology), Daniela Rus `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `Computer Vision and Pattern Recognition` `Pose Estimation` `Depth Estimation` `Transformer` `Vision Language Model` `Contrastive Learning` `Supervised Fine-Tuning` `LoRA` `Video` `Multimodality`

**🎯 论文内容**

本文收集并标注了 85 盘高质量的视角对称网球视频数据，使用该数据训练并评估多种基准视觉模型在 7 项时空感知与预测任务上的表现。

**💡 创新点**

创新点在于将大规模预训练视觉模型迁移到从移动视角捕获的网球视频中进行时空预测，并展示其在实时全身运动生成中的可行性。

**🔧 技术方法**

采用自监督、语言监督和视听多模态的图像编码器、视频编码器以及 VLM 进行特征提取，并在其上加入跨注意力预测头和 LoRA 适配器进行微调。

**📊 数据集**

使用同步的第一人称摄像头与 5 台静态摄像头录制的 53155 帧网球比赛视频，包含 3D 球轨、2D 球员位置、击球时刻与击球类型等标注。

**📈 对比分析**

通过在同一数据集上对 13 种模型进行统一评估，发现图像编码器与 VLM 在当前帧定位误差 0.54–0.71 m、预测误差 0.34–1.27 m 之间相近；深度信息能使 3D 球定位误差降至 0.21 m；较长视觉历史对预测提升有限。

**⚠️ 局限性**

主要限制在于视角下的深度估计不精确、球与球拍对准误差大以及运动记录质量不够高，影响后续全身运动生成的精度和真实感。

---

## 751. ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning

**arXiv ID:** 2610.37666 | [PDF](https://arxiv.org/pdf/2610.37666v1)

**作者:** Chaehyun Kim `[一作]` (NVIDIA), Fu-En Yang `[通讯]` (NVIDIA)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning` `Transformer` `Vision Language Model` `Vision-Language-Action Model` `Reinforcement Learning` `Chain-of-Thought` `Video` `Image` `Text` `Multimodality`

**🎯 论文内容**

提出了ThinkWorld框架，利用Chain-of-Dynamics（CoD）与世界想象强化学习，将人类视频与机器人演示共享的动作推理实现跨体型、跨任务、跨环境的泛化与技能组合。

**💡 创新点**

创新点包括：①将CoD定义为基于世界动力学的共享动作表示；②通过未来想象与观察到的未来对比为CoD提供奖励，无需动作标签；③将未来特征对比作为RL奖励，扩展到无标签视频；④将视觉、语言、动作三模态融合在同一VLM中。

**🔧 技术方法**

采用了Qwen3‑VL‑2B + MolmoAct2 VLM作为骨干，FAST式运动分词器，将未来查询映射至DINOv3特征空间，并使用GRPO进行RL；同时结合自回归CoD生成与未来预测损失。

**📊 数据集**

使用的数据集包括机器人演示集（MolmoAct2‑DROID、BimanualYAM、Bridge/RT‑1、BC‑Z、RoboCOIN）、人类视频集（EgoDex、EgoVerse）、LIBERO与LIBERO‑PRO，以及实机YAM的真实实验数据。

**📈 对比分析**

与Diffusion Policy、MolmoAct2、LaWAM、VLA‑JEPA等方法在ID/OOD任务、LIBERO、LIBERO‑PRO以及真实机器人实验中比较，ThinkWorld在OOS场景、任务与环境迁移上取得最高成功率，例如在LIBERO‑PRO上实现55.6%平均成功率，显著优于对比模型。

**⚠️ 局限性**

局限性包括：对预训练VLM和DINO特征质量高度依赖；未来预测在极端动态或光照变化场景下仍易失真；RL奖励基于特征相似度可能导致误导；在完全未知的动作或物体上仍可能失败；对人类视频的规模和多样性仍有限。

---

## 752. An EVM-Cosmos Bridge for Atomic, Semantic-Preserving Transfer of Environmental Tokens

**arXiv ID:** 2610.37663 | [PDF](https://arxiv.org/pdf/2610.37663v1)

**作者:** Sebastian Pandolfi `[一作]` (Universidad de la Repúblic), Laura González `[通讯]` (Universidad de la Repúblic)

**关键词:** `Software Engineering` `Federated Learning` `Optimization` `Explainability and Interpretability` `Computational Efficiency` `Safety and Privacy` `Chain-of-Thought` `Prompt Engineering` `Retrieval-Augmented Generation` `Contrastive Learning` `Tabular` `Time Series` `Sequential` `Finance Related`

**🎯 论文内容**

在EVM和Cosmos生态之间实现了原子且语义保留的环境指标代币跨链桥接，避免了语义混淆和超时竞态；

**💡 创新点**

①在HTLC基础上引入单方回调与赏金激励，解决跨链超时竞态；②通过加密承诺生成确定性指标身份，构建类注册表并在两链上镜像，保证语义一致性；

**🔧 技术方法**

HTLC、Axelar GMP跨链消息传递、ERC‑1155多类代币标准、CosmWasm合约、keccak256内容地址、TLA+机理检查、Bounty激励机制；

**📊 数据集**

基于真实环境指标代币的主网部署（Polygon‑Neutron），共执行50次跨链转账实验；

**📈 对比分析**

在50次实验中，EVM→Cosmos平均延迟97 s（p95 = 141 s），成本约1–2 USD，语义校验开销低于0.4 % 消息费用；与传统HTLC或信任中心桥接相比，提供原子性与语义安全；

**⚠️ 局限性**

依赖Axelar的消息费用资格与回调时延；需要监视器或赏金以防回调失败；仅支持单方转移；仅在Polygon‑Neutron对上验证；类创建受限于所有者；缺乏实现级形式验证。

---

## 753. TeamGym: A Gym for Team Decision-Making by AI Agents with Distributed Information

**arXiv ID:** 2610.37571 | [PDF](https://arxiv.org/pdf/2610.37571v1)

**作者:** Chanwoo Park `[一作]` (Massachusetts Institute of Technology), Asuman Ozdaglar `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `Artificial Intelligence` `Reinforcement Learning` `Federated Learning` `Explainability and Interpretability` `Computational Efficiency` `Data-Centric Learning` `AI Code Assistant` `Optimization` `Transformer` `Large Language Model` `Reinforcement Learning` `Retrieval-Augmented Generation` `Prompt Engineering` `Chain-of-Thought` `Text` `Tabular` `Sequential` `Finance Related` `Legal Related` `Medical Related` `Software Engineering Related` `Benchmark`

**🎯 论文内容**

构建了一个名为 TeamGym 的环境，用以生成、评估和训练多智能体团队决策问题，重点关注信息分布、检索和沟通结构；通过训练强化学习策略提升团队决策质量并观察协同现象；

**💡 创新点**

1）提出了“关键信息（PIU）”的概念，能精确划分支持正确决策所需的最小信息集合；2）通过可控的PIU分配、沟通结构和代币预算，系统性区分了沟通缺口与检索缺口；3）在多模型、多领域数据集上展示团队协同训练可自发产生高效沟通策略。

**🔧 技术方法**

基于大型语言模型（如 Qwen2.5‑7B‑Instruct、GPT‑5.5）实现决策与评估；使用强化学习（Group Relative Policy Optimization）训练参与者与协调者策略；采用PIU构造算法（注释、推理、留一检验）生成最小证据集合；实验中还利用对齐评分（GPT‑5.5 评判）衡量决策成功率。

**📊 数据集**

法律（CUAD、MAUD等）、金融（FinDVer、FinLongDocQA）、医疗（MLCR、EvidenceBench）和软件工程（SWE Atlas、DeepSWE）四大领域共计12个真实语料库，构成了团队决策实例。

**📈 对比分析**

在单模型、分配PIU、检索PIU三种评估条件下，对比8种语言模型在12个语料库中的成功率；结果显示：PIU分布导致平均22.9个百分点的沟通缺口，检索缺口再进一步下降19.9个百分点；强化学习训练后，团队决策质量提升约10–20个百分点，并出现自发的协调行为。

**⚠️ 局限性**

1）实验仅涵盖单一模型训练跑；2）PIU构造依赖模型推理，可能存在误判；3）仅考虑共享收益假设，未探讨策略博弈或恶意行为；4）检索成本未显式建模，忽视真实检索负担；5）在部分领域（如FinLongDocQA）训练转移效果有限。

---

## 754. Heads of Security: Learned Attention Gates Improve Prompt Injection Robustness

**arXiv ID:** 2610.37657 | [PDF](https://arxiv.org/pdf/2610.37657v1)

**作者:** Igor Shilov `[一作]`, Yves-Alexandre de Montjoye `[通讯]`

**关键词:** `Machine Learning` `Safty and Privacy` `Transformer` `Large Language Model` `Prompt Engineering` `Reinforcement Learning from Human Feedback` `Text`

**🎯 论文内容**

提出了一种名为Heads of Security (HoS) 的防御方法，通过在Transformer注意力头中学习自适应通道门控，实现指令–数据分离，从而防止Prompt Injection攻击。

**💡 创新点**

创新点在于只通过极少量的通道门控参数（每头每层），在注意力计算前加上信任级别相关的偏置，重新分配对可信与不可信 token 的注意力，而不需要完整微调模型或增加新能力。

**🔧 技术方法**

使用的技术包括：通道条件注意力门控（在注意力预激点加上 logσ(g) 乘子），对 Llama‑3.3‑70B‑Instruct 与 GPT‑OSS‑20B 进行训练，训练方式参照 SecAlign++ 的 Direct Preference Optimization 与 Cleaned‑Alpaca 数据集。

**📊 数据集**

数据集：Cleaned‑Alpaca（含用户指令与注入指令对），以及多种 Prompt Injection 评测基准（AgentDojo、AgentDyn、LLMail‑Inject、WASP 等）和标准多选题基准（MMLU‑Pro、5‑shot 等）。

**📈 对比分析**

与现有防御 SecAlign++（微调型）和 ASIDE（架构型）相比，HoS 在黑盒 RL‑Hammer 与白盒 Claudini 攻击下的攻击成功率显著降低（GPT‑OSS 上仅10% vs 97%/85%），在标准多选与few‑shot 评测中保持与基准模型相近的效能（如 5‑shot MMLU‑Pro 67% vs 68%），且在 Agentic Prompt Injection 基准中与 SecAlign 竞争并降低攻击成功率。

**⚠️ 局限性**

局限性：HoS 仅对标准密集注意力模型适用，无法直接应用于纯线性注意力架构；其安全性依赖于已定义的信任边界，无法自动判断哪些内容可信；训练数据固定，可能导致对不同对话场景的泛化不足；并非完整的防御体系，仍需与指令层级、上下文完整性等技术结合。

---

## 755. Partially Edited Audio: Generation, Defense, and Open Problems

**arXiv ID:** 2610.37640 | [PDF](https://arxiv.org/pdf/2610.37640v1)

**作者:** Lin Zhang `[一作]` (Independent Researcher), Sanjeev Khudanpur `[通讯]` (Johns Hopkins University)

**关键词:** `Sound` `Generation` `Data Synthesis` `Anomaly Detection` `Explainability and Interpretability` `Transformer` `Diffusion model` `Score-based Model` `Generative Adversarial Network` `Contrastive Learning` `Retrieval-Augmented Generation` `Audio` `Multimodality` `Review/Survey Paper` `Benchmark`

**🎯 论文内容**

对部分编辑音频（Partial‑Edited Audio）进行系统综述，阐述其定义、特征、生成与防御的双重视角，并整理现有编辑目标、方法、评估指标及防御技术。

**💡 创新点**

提出了“fake”与“有害”是应用依赖、非固有属性的观念，构建了生成与防御相互关联的框架，强调双重意图和责任可追溯性。

**🔧 技术方法**

综述了编辑技术（剪切/拼接、生成后拼接、上下文感知填充、指令条件统一模型）以及防御技术（深度伪造检测、定位、辨识、可解释性分析）。

**📊 数据集**

引用并评述了多种数据集和挑战赛：RealEdit、MMAE、PartialSpoof、HAD、ADD 系列、音视频多模态数据等。

**📈 对比分析**

通过对各方法在自然度、说话人相似度、WER、UTMOS、WavLM 等指标上的比较，总结了目前主流方法的优缺点，但未给出新实验结果，主要依赖公开基准和现有论文评估。

**⚠️ 局限性**

局限性包括：聚焦单语、单说话人、语音类编辑；对音乐、音乐伴奏、一般声效、跨模态编辑的讨论有限；防御研究仍偏向内容层面的语音伪造，缺乏对多说话人、多编辑来源的追踪；缺乏跨域、跨风格的鲁棒性验证。

---

## 756. Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions

**arXiv ID:** 2610.37624 | [PDF](https://arxiv.org/pdf/2610.37624v1)

**作者:** Anouar Imel `[一作]` (University of Electronic Science and Technology of China), Khalil El Khammar `[通讯]` (Nanjing University of Information Science and Technology)

**关键词:** `Artificial Intelligence` `Explainability and Interpretability` `Computational Efficiency` `Recommendation System` `Transformer` `Large Language Model` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `Text` `Benchmark`

**🎯 论文内容**

构造了一个名为 CrossProfile 的诊断基准，并在其上评估多种 LLM 记忆策略（稀有优先、相关性优先、随机、摘要等）以及决策模型的表现，进一步探讨记忆压缩、事实所有权和社交证明对最终决策的影响。

**💡 创新点**

创新点在于：① 交叉对比支持、干扰和无关事实的所有权而保持事实文本不变，形成可验证答案的基准；② 通过此基准揭示记忆策略的典型失效模式（稀有优先误入干扰、相关性优先忽略干扰）及决策模型对归因标签的社交证明偏差；③ 提出记忆设计应关注“证据平衡”而非“稀有度”或“主题相关性”。

**🔧 技术方法**

主要技术包括：多代理对话脚本化生成、按 token 预算选择或摘要记忆、稀有优先与相关性优先排序、oracle 相关性筛选、社交证明归因标签的消除与随机化、对照实验（随机、全记忆、空记忆）、以及对 LLM（Qwen2.5‑7B、Llama‑3.1‑8B、DeepSeek、Flash）进行推理。

**📊 数据集**

使用的数据集有：CrossProfile（60 个模板、480 个任务）、HiddenBench（65 个任务）、自然对话版本的 HiddenBench 以及通过 DeepSeek API 评估的 30/60 个 CrossProfile 任务。

**📈 对比分析**

比较方法：对每种记忆策略按多种 token 预算（0.4/0.6/0.8）计算决策准确率，并与随机、全记忆、空记忆、oracle 相关性等对照；使用 bootstrap 置信区间与 Holm 校正评估显著性。性能结果显示：在受控 HiddenBench 上稀有优先取得约 88% 的准确率，明显优于摘要和随机；在 CrossProfile 上稀有优先在“私有支持/共享干扰”细胞表现优秀，但整体不优于随机，相关性优先在经典隐藏配置下显著下降；决策模型对归因标签呈社交证明偏差，去除标签可提升 13–24 分。自然对话中无记忆策略优于其他，完整转录甚至低于空记忆。

**⚠️ 局限性**

局限性：① 记忆策略未能处理干扰事实的主题相关性；② 决策模型依赖归因标签导致社交证明偏差，真实场景下归因不独立；③ 证据往往在对话前期丢失，导致记忆无法恢复；④ 基准仅覆盖有限的四代理、固定文本模板，无法覆盖更大规模多样化情境；⑤ 评价主要基于准确率，未考虑解释性、资源消耗等维度。

---

## 757. SLICE-CiM: Selective Layer-wise Inference Control Engine with Confidence-Based Early Exit for SRAM based Digital Compute-in-Memory

**arXiv ID:** 2610.37601 | [PDF](https://arxiv.org/pdf/2610.37601v1)

**作者:** Pratham Sharma `[一作]` (Indian Institute of Technology Indore), Santosh Kumar Vishvakarma `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `Hardware Architecture` `Computational Efficiency` `Convolutional Neural Network` `Contrastive Learning` `Score-based Model` `Diffusion model` `Auto Encoder` `Generative Adversarial Network` `Image`

**🎯 论文内容**

提出了一套可精度可扩展的 SRAM 基数字计算内存（DCIM）外围控制框架，结合 10T NOR SRAM 细胞实现混合精度推理与基于置信度的早期退出。

**💡 创新点**

创新点包括：1）Flush‑Merge Enable (FME) 控制器将累加器刷新、计数器重置、精度更新和行线重新配置合并为单周期，消除模式切换空闲停滞；2）Precision Register/Decoder 通过训练无关的灵敏度指标实现按层精度分配；3）Confidence‑Based Early‑Exit 模块在 MSB‑优先序列中动态终止累加；4）全周期控制器保证安全退出，正式推导早期退出误差上界。

**🔧 技术方法**

使用的技术包括：10T NOR SRAM 计算细胞、bit‑serial MAC、MSB‑优先排序、训练无关灵敏度度量、阈值校准、动态精度切换、硬件验证（Cadence Virtuoso/Genus/Calibre/Spectre）、功耗评估（VCD 流）和系统级推理评估。

**📊 数据集**

在 CIFAR‑100 数据集上使用 MobileNetV2 进行推理测试，并通过对 1,000 张训练图像的前向校准确定每层阈值。

**📈 对比分析**

与现有混合精度方法（如 HAWQ、HAQ）以及其它 SRAM‑CIM 方案相比，本设计在 2b/8b 与 0.8 V 下实现 454.91 TOPS/W，约为 8b/8b、1.2 V 基线 31.35 TOPS/W 的 13.33 倍提升，功耗仅占系统 0.24%。

**⚠️ 局限性**

局限性包括：对 GELU/SiLU 等低稀疏激活的 Transformer 模型适应性不足；需要在动态阈值与极端激活值下进一步校准；以及对更大规模矩阵尺寸的可扩展性和多芯片互连的研究仍待展开。

---

## 758. MoI-Bench: A Benchmark for Image Generation and Editing on Mobile Devices

**arXiv ID:** 2610.37599 | [PDF](https://arxiv.org/pdf/2610.37599v1)

**作者:** Kailai Feng `[一作]` (University College London), Jagmohan Chauhan `[通讯]` (University College London)

**关键词:** `Computer Vision and Pattern Recognition` `Generation` `Computational Efficiency` `Transformer` `Large Language Model` `Diffusion model` `Prompt Engineering` `Vision Language Model` `Image` `Text` `Multimodality` `Benchmark`

**🎯 论文内容**

提出了MoI‑Bench和MoI‑Eval，用于从用户需求出发评估移动端图像生成与编辑模型的功能与部署成本。

**💡 创新点**

创新点在于将任务拆分为操作×目标两维轴，并结合细粒度的评估准则与质量‑部署二维地图；同时首次提出基于任务的质量与部署可行性两层评估框架。

**🔧 技术方法**

采用大语言模型（GPT‑4o、Gemini‑3.1‑Pro）生成问答式评测，量化指令遵循、视觉质量、细节保留等维度；在移动端使用4‑bit量化、FP16/8‑bit激活等多种精度配置进行推理测评。

**📊 数据集**

使用基于用户调查收集的1150个样本，覆盖64个子任务，涵盖文本生成、局部/全局编辑、语义编辑、引用编辑等多种场景；样本来自公开图像库（Pexels、Wikimedia Commons）并手工构造指令。

**📈 对比分析**

对19个模型（含闭源和公开权重）在质量、任务覆盖和移动部署成本上进行对比，闭源模型达Q4级别，开源模型最高为Q3；移动端模型如DreamLite、Ovis‑U1在Q2但部署效率低，揭示质量与部署之间的权衡。

**⚠️ 局限性**

局限性包括对多图像引用任务支持不足、文本编辑能力弱、移动部署仍未达到低延迟/低内存的理想指标；评测仅在有限的硬件平台上完成，缺乏跨设备的通用性验证。

---

## 759. Timestral

**arXiv ID:** 2610.37585 | [PDF](https://arxiv.org/pdf/2610.37585v1)

**作者:** Adrien Petralia `[一作]`, Zsófia Csákány `[通讯]`

**关键词:** `Artificial Intelligence` `Time Series` `Transformer` `Supervised Fine-Tuning` `Large Language Model` `Contrastive Learning` `Diffusion model` `Score-based Model` `Time Series` `Tabular` `Benchmark`

**🎯 论文内容**

提出了名为Timestral的四个时间序列基础模型，能够进行概率性单变量、多变量以及含协变量的预测，并通过两阶段预训练+微调实现高精度。

**💡 创新点**

核心创新在于：① Decoupled Horizon Forecasting（DHF）将预测令牌与因果上下文分离，解决了长预测区间与上下文遮掩的耦合问题；② 角色感知协变量框架，让已知未来协变量在预测中可用而不泄漏目标未来信息；③ 先在大规模合成语料上预训练，再在真实+合成混合语料上微调，显著提升泛化能力。

**🔧 技术方法**

技术手段包括：Transformer（带RMSNorm、SwiGLU、门控注意力、可变体注意力）、32步patch嵌入、两层残差MLP、量化头（21个分位点）、Pinball损失、非自回归多步预测、缺失数据增广。

**📊 数据集**

训练使用了自制的合成数据（包含趋势、季节、突变等多样化特征）和真实数据；评测基准为GIFT‑Eval、TIME、fev‑bench（MASE、CRPS、SQL）、IND‑Eval，统一使用公开数据集。

**📈 对比分析**

在所有基准上，209M模型以最低MASE、CRPS、最高SQL和MASE skill排名第一，123M模型排名第二；四个规模在参数-准确度Pareto前沿，且在同类模型中实现了显著性能提升，优于2.5B参数模型。

**⚠️ 局限性**

局限性包括：① 仍受参数规模限制，进一步增大模型可提升性能但推理成本上升；② 推理依赖CPU/GPU资源，尤其在多变量长上下文时延迟显著；③ 预训练依赖合成数据，可能与真实分布存在偏差；④ 仅覆盖概率分位预测，未直接支持自回归采样或多步骤模拟。

---

## 760. Zero-Shot Multimodal Event Spotting in Soccer Broadcasts

**arXiv ID:** 2610.37583 | [PDF](https://arxiv.org/pdf/2610.37583v1)

**作者:** Sadia Kamal `[一作]` (University of Maryland Baltimore County), Tony Braskich `[通讯]` (Comcast)

**关键词:** `Computer Vision and Pattern Recognition` `Recognition` `Explainability and Interpretability` `Computational Efficiency` `Transformer` `Large Language Model` `Prompt Engineering` `Vision Language Model` `Retrieval-Augmented Generation` `Video` `Text` `Multimodality`

**🎯 论文内容**

本文提出了一种零样本多模态框架，能够在完整的足球比赛转播中检测、分类并定位事件，同时给出兴奋度评分；

**💡 创新点**

创新点在于无需足球专项训练、使用通用多模态大语言模型联合视觉故事板与解说文本实现事件检测与时间定位，并通过提示优化和兴奋度评分提升可解释性；

**🔧 技术方法**

核心技术包括故事板视觉表示、对齐解说文本、通用多模态LLM推理、两步提示设计、自动提示优化（GEPA、MIPROv2）以及基于证据的兴奋度打分；

**📊 数据集**

使用了三场公开 YouTube 足球转播（共 130 个事件，6 类：进球、射门、失误、角球、开球、黄牌）进行手工标注；

**📈 对比分析**

在跨数据集设置下，零样本 GPT‑5.6 系列模型在 5 秒阈值下取得 72% F1 与 76% mAP，显著优于基于 SoccerNet 预训练的 CALF 与 NetVLAD++，并且在故事板输入下推理成本比单帧低 72%；

**⚠️ 局限性**

局限包括仅在三场比赛上验证，可能无法泛化到不同联赛/转播风格，故事板采样可能遗漏短时视觉特征，回放/回顾内容仍易误检，且兴奋度评分依赖模型判断，缺乏人类主观验证。

---

## 761. Collective Dynamics of Self-Reinforcement: How Cross-Task Coupling Drives Policy Collapse in Unsupervised RLVR

**arXiv ID:** 2610.37553 | [PDF](https://arxiv.org/pdf/2610.37553v1)

**作者:** Peize Ding `[一作]` (Columbia University), Peize Ding `[通讯]` (Columbia University)

**关键词:** `Machine Learning` `Reinforcement Learning` `Transformer` `Reinforcement Learning` `Neural Tangent Kernel (NTK) theory` `Random Simulation` `Text`

**🎯 论文内容**

论文研究了无标签强化学习中使用多数投票自我奖励时，随着训练问题数量增大导致策略崩溃的现象。

**💡 创新点**

创新点在于提出了跨任务共享参数的集体不稳定性机制，并给出临界问题数与共享耦合、初始置信度等参数的量化关系。

**🔧 技术方法**

使用了可解析的多问题模型、神经切线核（NTK）理论、随机模拟以及小型Transformer实验等技术。

**📊 数据集**

实验主要基于合成的二分类任务和小型Transformer上的未标记问题集，没有使用大规模真实数据集。

**📈 对比分析**

通过与传统多数投票奖励、GRPO、Adam等方法比较，证明了理论预测的临界规模与衰减曲线，并展示了添加边际熵奖励后可抑制崩溃的效果。

**⚠️ 局限性**

限制在于模型假设过于简化，核在训练中会改变，且未在大规模预训练语言模型上验证，导致预测在真实网络中的准确性有限。

---

## 762. Scaling Laws for Tool-Use Agents via Controllable Environment Synthesis

**arXiv ID:** 2610.37648 | [PDF](https://arxiv.org/pdf/2610.37648v1)

**作者:** Andrew Szot `[一作]` (Apple), Alexander Toshev `[通讯]` (Apple)

**关键词:** `Machine Learning` `Data Synthesis` `Optimization` `Reinforcement Learning from Human Feedback` `Autonomous Driving` `Computational Efficiency` `Federated Learning` `AI Code Assistant` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Agentic AI` `Prompt Engineering` `Retrieval-Augmented Generation` `Chain-of-Thought` `World Model` `Text` `Tabular` `Benchmark`

**🎯 论文内容**

设计并实现了可控的可执行环境合成管道，利用LLM自动生成多功能应用、数据库架构与API，并基于这些环境合成多步工具使用任务，随后用SFT训练LLM代理并在多种基准上评估其泛化能力。

**💡 创新点**

创新点在于：① 通过LLM可控合成完整的可执行工具集（app）与对应数据库，① 提供对工具数量、任务多样性、数据配置（数据库实例与干扰数据）等维度的精细控制；② 基于这些可控属性构建了数据组合依赖的缩放律，能够预测并给出在给定计算预算下的最优数据分配策略。

**🔧 技术方法**

技术手段包括：Gemini-3.1 Pro（构建app与任务）、GLM-5.2（生成专家轨迹）、Qwen3.5（SFT训练学生模型）、Gymnasium环境接口、Python API交互、可验证的任务指令与oracle求解器、以及对训练流量与FLOPs的Chinchilla近似，用于构建缩放律。

**📊 数据集**

数据集方面：合成了458,752个任务，覆盖64个app、10,472个API（6,266写入、4,206读取），生成约5.9B训练token；基准测试使用AppWorld、BFCL、ASTRA-Bench以及自建Holdout集。

**📈 对比分析**

与现有基线（Nemotron、ToolACE、SimSAT、EnvScaler、EnvFactory等）进行对比，结果显示：在AppWorld、BFCL等基准上平均提升30.5pp；在AppWorld Test‑C上比SimSAT高7.6–11.6pp；在BFCL上比最强基线高2.0–3.7pp；采用缩放律给出的最优数据组合相比单纯扩展任务数量显著提升性能。

**⚠️ 局限性**

局限性：① 未模拟用户在环交互，缺乏人机问答环节；② 重点关注结构化数据库任务，缺少长文本检索与信息检索场景；③ 缩放律仅在单一合成管线验证，未验证跨管线迁移；④ 所有实验均在模拟环境中完成，未展示真实部署效果。

---

## 763. Understanding the limits of latent world models

**arXiv ID:** 2610.37588 | [PDF](https://arxiv.org/pdf/2610.37588v1)

**作者:** Rodrigo Gonzalez Laiz `[一作]` (Helmholtz Munich), Steffen Schneider `[通讯]` (Helmholtz Munich)

**关键词:** `Machine Learning` `Reinforcement Learning` `Representation Learning` `Optimization` `Recurrent Neural Network` `Auto Encoder` `World Model` `Reinforcement Learning` `Contrastive Learning` `Flow-based Model` `Image` `Tabular`

**🎯 论文内容**

本文系统评估了隐式世界模型是否真实地学习环境动力学，提出了“faithful latent world model”概念和等变条件，并引入latent rollout consistency（LRC）指标和状态解码 probe 以客观量化模型的表示与动态一致性。

**💡 创新点**

创新点在于：①将世界模型的真实性定义为状态到潜在空间的等变性；②提出 LRC 作为衡量动作条件潜在预测一致性的可计算指标；③通过状态解码与 LRC 结合揭示了两类失败模式（速度估计不足与非线性观察映射导致的动态学习困难）。

**🔧 技术方法**

技术方法包括：使用 RSSM（Recurrent State‑Space Model）编码器+潜在动力学；MPC 控制器；对潜在序列做自回归 LRC 计算并用 R² 评估；利用非线性混合层（可逆）对观测进行可逆变换；对解码器（probe）进行非线性 MLP 训练以评估状态信息保留；对比状态输入与像素输入下的模型性能。

**📊 数据集**

数据集为 DeepMind Control Suite 中的三个低维连续控制任务：Pendulum、CartPole 与 Reacher；分别在原始状态观测与 64×64/224×224 RGB 图像观测下收集 1,000 条 100 步随机轨迹用于评估。

**📈 对比分析**

评估方法：对每个模型与环境，计算 LRC（随时间步长的 R²）、状态解码误差（位置/速度）以及 500 步控制奖励；并与仅接受状态输入的基线模型做对比。实验结果显示：①任务奖励高并不保证高 LRC；②像素观测导致 LRC 衰退速度快，且速度信息缺失是主要瓶颈；③即便补入真实速度也无法完全弥补像素观测的不足；④引入可逆非线性混合层后 LRC 与奖励均下降，表明非线性观测难以让模型学习 faithful 动态。

**⚠️ 局限性**

局限性：①仅在简单的确定性连续控制环境上验证，未探讨随机、部分可观测或高维情形；②只评估三类代表性模型，未覆盖所有现有世界模型；③实验结果更多展示对比例子而非普遍适用的结论；④对 LRC 的理论解析与实现细节仍有待进一步完善。

---

## 764. When Should We Intervene at Test Time?

**arXiv ID:** 2610.37658 | [PDF](https://arxiv.org/pdf/2610.37658v1)

**作者:** Jiahao Zhang `[一作]` (Zhengzhou University of Aeronautics), Yu Chen `[通讯]` (Zhengzhou University of Aeronautics)

**关键词:** `Machine Learning` `Domain Adaptation` `Classification` `Transformer` `Contrastive Learning` `Bayesian Estimation` `Active Learning` `Image` `Text`

**🎯 论文内容**

本文提出了 PORTAL 控制器，用于在测试时动态决定何时部署现有的干预措施（如 BN 更新、视图融合等），在何处获取标签，以及何时停止查询。该控制器利用共享标签信息，在同一批次内对所有候选预测进行比较，并根据估计准确率差距和共享错误决定查询策略，从而在保证平均准确率的同时避免批次误差。

**💡 创新点**

创新点包括：
1) 引入共享标签机制，使得一个标签即可同时评估所有候选预测；
2) 结合贝叶斯估计（Dirichlet 先验）和共享误差协方差，推导出每个候选的期望准确率与方差；
3) 设计基于“可达性”和“期望价值”两步的查询策略，自动在必要时停止标签收集；
4) 将获取的标签用于后续批次的候选模型训练，实现“同一批次标签双赢”。

**🔧 技术方法**

技术手段：
- 共享标签与分层(strata)划分，基于预测模式聚合相同候选集合；
- 贝叶斯更新（Dirichlet 先验）求解候选准确率均值与协方差；
- 基于 EVPI（完美信息期望值）和可达性判据的查询停止机制；
- 目标函数最大化与最小化准则的贪心分配；
- 训练阶段使用“决策标签”提升未来批次候选模型。

**📊 数据集**

数据集：
- PACS（Photo、Art、Cartoon、Sketch）
- CIFAR‑10‑C / CIFAR‑100‑C
- ImageNet‑C（完整腐败流）
- CLIP 视觉–文本预训练模型（ViT‑L/14）用于 PACS、OfficeHome、VLCS
- CIFAR‑10.1
- 其他如 VLCS、OfficeHome、PACS 等域自适应基准。

**📈 对比分析**

对比方法：
- 固定规则（未改动模型、BN、Flip、BN+Flip、Mean‑logit 集成）
- ERM（Uniform、Disagreement、Top‑KL）
- Batch oracle、EATTA、其他主动适配方法
- 评估指标：平均准确率、批次误差（Batch harm）、标签预算、收敛速度。结果显示：
  * 在类块到达的 PACS 任务中，PORTAL 将 BN+Flip 的准确率从 22.58% 提升至 51.28%；
  * 对完整 ImageNet‑C 流，误差从 50.14% 降至 49.55%（提高 0.59%）；
  * 在 CIFAR‑10‑C/100‑C 上，PORTAL 仅用 91% 的标签量保持 BN+Flip 的性能；
  * 在 CLIP‑PACS 任务中，PORTAL 通过 1.4–7.4 个标签就能匹配八视图输出，显著节省查询成本。

**⚠️ 局限性**

局限性：
- 需要先验的标签分布假设（分层内类概率不变），在快速变化或多模态环境下可能失效；
- 查询策略依赖标签成本设定，若标签获取成本高或不可控，效果受限；
- 控制器只能在已有候选集合内作决策，无法自适应生成新干预或改进模型结构；
- 对极端数据偏差或严重域漂移时，期望值与方差估计可能失真；
- 评估主要基于分类任务，尚未验证到更复杂的生成或检测任务。

---

## 765. KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry

**arXiv ID:** 2610.37655 | [PDF](https://arxiv.org/pdf/2610.37655v1)

**作者:** Wenqing Peng `[一作]` (University of Helsinki), Zhi-Song Liu `[通讯]` (LUT University)

**关键词:** `Artificial Intelligence` `Physics Related` `Computational Efficiency` `Optimization` `Ordinary Differential Equation` `Stochastic Differential Equation` `Auto Encoder` `Contrastive Learning` `Automatic Differentiation` `Tabular` `Time Series`

**🎯 论文内容**

提出 KPPax，一种将 KPP 语言化学机制转换为可微、JIT 编译且可向量化的 JAX 代码的预处理器；

**💡 创新点**

创新点在于实现了对化学速率方程的稀疏 log‑空间求值与自动微分，能够自动生成精确的敏感度（前向/后向）而无需手工推导；

**🔧 技术方法**

主要技术包括 JAX 自动微分、稀疏 COO 矩阵运算、log‑空间速率求值、递归检查点的后向 AD、Kvaerno 隐式 RK 求解器，以及对 KPP 语法的解析；

**📊 数据集**

使用的基准数据集是 POLLU 机制，构造了模拟的流管实验数据用于识别分析，并生成合成的排放检索数据用于网络训练；

**📈 对比分析**

对比方法：将自动微分得到的雅可比与有限差分比较（误差 < 1e-6），计算速度提高 36 倍；识别预测与参数估计误差在 30% 内吻合；在线微调使生物排放校正的 RMSE 降低 2.4–5 倍，单 GPU 训练约 2 分钟；

**⚠️ 局限性**

局限性包括 log‑空间要求浓度始终正，导致对零或近零浓度梯度为零；隐式求解器使用密集 LU 分解，难以扩展至千级分子机制；识别分析仅为局部，无法捕捉全局非线性曲率；需开发稀疏/矩阵无关求解器以实现大规模机制。

---

## 766. Can LLMs Uncover Hidden Physical Laws through Free Exploration?

**arXiv ID:** 2610.37649 | [PDF](https://arxiv.org/pdf/2610.37649v1)

**作者:** Zhen Yang `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `Computation and Language` `Physics Related` `Transformer` `Large Language Model` `Agentic AI` `Prompt Engineering` `World Model` `Text`

**🎯 论文内容**

提出并实现了一个完全自动化的物理法则发现框架LongChang，让大型语言模型在没有预设任务或提示的情况下，独立探测并实验验证模拟世界中的改动物理法则。

**💡 创新点**

核心创新在于：①无任务驱动的自我提问与实验设计；②多代理LLM协同完成场景构造、工具设计、实验记录与评估；③通过实验记录驱动场景演化，形成闭环的自主科学发现循环。

**🔧 技术方法**

技术包括：基于MuJoCo的3D刚体模拟；LLM生成与审阅的多代理架构（探险者、工具评审者、场景设计者、判定者）；自动化实验记录与验证；对话式指令与动作接口。

**📊 数据集**

数据集：自定义的15个场景（1个正常控制+14个单一改动物理法则），并通过LLM场景生成器扩展的多种演化场景。

**📈 对比分析**

对比方法：与PhysGym、NewtonBench、DiscoverPhysics等基准对比；评价指标包括法则正确性C、实验支持S、发现率D、探索长度L、构造成功率CS等。结果显示，12个LLM中只有GPT‑6‑Astra在无任务情况下获得16.7%发现率（最高），其余11个均为0%；即使在额外的任务提示或更高预算下，发现率提升有限。

**⚠️ 局限性**

局限性：当前LLM很难从偶然观察中形成完整实验方案并推断完整物理法则；发现率低、对实验记录的解读不充分；框架仅在人工合成的模拟世界中测试，未验证在真实复杂环境中的效果。

---

## 767. Multi-State Linear Attention with Generalized Second-Order Routing

**arXiv ID:** 2610.37645 | [PDF](https://arxiv.org/pdf/2610.37645v1)

**作者:** Hee Seung Hwang `[一作]` (Princeton University), Zhiwei Deng `[通讯]` (Google DeepMind)

**关键词:** `Computation and Language` `Computational Efficiency` `Representation Learning` `Transformer` `Large Language Model` `Retrieval-Augmented Generation` `Mixture of Experts` `Text` `Benchmark`

**🎯 论文内容**

本文提出了一种基于第一、第二矩的多状态线性注意力聚合方法。

**💡 创新点**

创新点在于利用键向量的均值和方差来构造聚合权重，并通过门控混合softmax实现更内容感知的路由。

**🔧 技术方法**

技术上使用多状态线性注意力、键/查询矩阵的第一、第二矩、门控softmax混合以及Gated DeltaNet/Mamba-2架构进行实验。

**📊 数据集**

使用FineWeb‑edu预训练数据，Long‑Data‑Collections、RULER NIAH、HELMet检索增广生成数据以及LongBench多子任务作为评估数据集。

**📈 对比分析**

与Vanilla、Position、Meanpool基线对比，在RULER NIAH/QA上提升8.3%/7.0%，在LongBench平均提升13.5%/12.0%，且在相同性能下使用的内存仅为基线的四分之一。

**⚠️ 局限性**

局限性包括需要额外的中期训练步骤，额外矩阵存储导致最高约12.6%内存增量，且仅在Gated DeltaNet与Mamba-2上验证，可能不适用于其他模型或更大规模。

---

## 768. LCM-Promo: An LLM-Based Uplift Model for Promotion Allocation

**arXiv ID:** 2610.37570 | [PDF](https://arxiv.org/pdf/2610.37570v1)

**作者:** Nesta Midavaine `[一作]` (Prosus), Zülküf Genç `[通讯]` (Prosus)

**关键词:** `Machine Learning` `Recommendation System` `Optimization` `Transformer` `Large Language Model` `Supervised Fine-Tuning` `Prompt Engineering` `Tabular` `Text` `Finance Related`

**🎯 论文内容**

在iFood平台上开发并部署了基于大型语言模型的促销分配系统LCM‑Promo，能够预测不同促销力度下用户的转化阈值，并实现促销的精准分配。

**💡 创新点**

创新点在于将LLM细化为“大型商务模型”，通过单词输出解释为转化阈值的概率分布，天然地产生单调的促销‑响应曲线，并可直接用于预算约束下的分配和离线提升评估。

**🔧 技术方法**

技术上采用4B参数稠密LLM在商务日志上进行微调，利用交叉熵训练单词预测，结合阈值概率分布（PMF）解码，并实现了水流增强（waterfall）标签；推理部署在自托管vLLM和BESH批处理终端，使用预计算的用户行为表征作为提示。

**📊 数据集**

数据集为iFood的交易与促销日志（含用户、商家、产品交互），以及基于随机对照实验（RCT）的固定促销数据，用于离线评估；在线A/B实验覆盖数百万用户的流失再激活场景。

**📈 对比分析**

与原生产策略Segment‑EMA相比，在线A/B实验实现了+1%转化率提升和-6%促销成本下降；离线评估与Uplift Random Forest比较，AUUC和Qini相当，Uplift@30略低，水流变体表现更差。

**⚠️ 局限性**

局限性包括仅在单一生命周期阶段（流失再激活）和单一平台（iFood）验证，扩展到其他生命周期阶段和电商平台的效果尚待探索；水流增强方法与LLM预训练目标冲突导致表现不佳。

---

## 769. When, Where, and What to Correct: Human-Guided Adaptive Residual Correction for Robot Policies

**arXiv ID:** 2610.37547 | [PDF](https://arxiv.org/pdf/2610.37547v1)

**作者:** Anukriti Singh `[一作]` (University of Maryland), Firas Abi-Farraj `[通讯]` (Enchanted Tools)

**关键词:** `Robotics` `Robotic Intelligence` `Reinforcement Learning from Human Feedback` `Transformer` `Auto Encoder` `Reinforcement Learning` `Principal Component Analysis` `Supervised Fine-Tuning` `Sequential` `Tabular`

**🎯 论文内容**

提出了一种轻量级的后训练模块ARC，在冻结的模仿学习策略上通过稀疏人工修正学习残差校正，以提升精细操作任务的成功率。

**💡 创新点**

创新点在于利用人类干预的时机和低维修正子空间来限定残差学习的时间和动作空间，而不是直接将人类修正作为目标，从而在保持基准策略不变的前提下显著提高精度。

**🔧 技术方法**

采用了二分类门控网络预测何时激活残差，主成分分析提取低维修正基底，强化学习（TD3）在此限定空间内进行奖励驱动的残差学习。

**📊 数据集**

使用真实机器人Miroki手臂进行的三项精细操作任务（托盘堆叠、标记插入、杯子堆叠），并基于每个任务的约55条示范数据训练基准策略以及约8–10条修正回放用于ARC。

**📈 对比分析**

与基准ACT策略和直接监督学习基准相比，ARC在三项任务中平均成功率从33.3%提升至75.0%，在托盘堆叠任务中实现100%成功，显著优于其他方法。

**⚠️ 局限性**

局限性包括对人工干预时机的依赖，门控模型可能随操作者变化而不同，需要额外的数据收集；且未探讨更复杂任务或多模态感知下的适用性。

---

## 770. Honeycomb-Surfels: Compact Hexagonal Primitives for Efficient Scene Rendering and Reconstruction

**arXiv ID:** 2610.37668 | [PDF](https://arxiv.org/pdf/2610.37668v1)

**作者:** Prakhar Kaushik `[一作]` (Johns Hopkins University), Alan Yuille `[通讯]` (Johns Hopkins University)

**关键词:** `Computer Vision and Pattern Recognition` `Computational Efficiency` `Generation` `Data Synthesis` `Neural Radiance Field` `Gaussian Splatting` `Perspective-correct ray-plane intersection rendering` `Soft edge sigmoid weighting` `Linear color gradient` `Image` `Point Cloud` `Mesh`

**🎯 论文内容**

提出了一种新的六边形表面原语 Hexels，用于 3D 新视角合成、表面重建和 2D 图像表示，目标是降低模型内存、训练 VRAM 与渲染时间。

**💡 创新点**

创新点在于将六边形支持与一阶空间颜色场结合，单个原语即可覆盖更大局部平面区域并建模光滑颜色变化，避免纹理贴图和神经解码器，显著提升表达效率。

**🔧 技术方法**

使用正交切向量定义的六边形平面原语、软边缘 sigmoid 加权、线性颜色梯度、Perspective‑correct 射线‑平面交点渲染、Adam 优化、稠密化与剪枝等技术实现。

**📊 数据集**

在 34 个真实场景中评估，包括 DTU、Tanks & Temples、Mip‑NeRF360 以及 Image‑GS 数据集。

**📈 对比分析**

通过在匹配模型内存、匹配原子数或原始设置下与 2DGS、3DGS、Triangle Splatting 等基线对比，Hexels 在相同模型内存下 PSNR 提升 4.5–7.4 dB，VRAM 减少 3.6 倍，FPS 提升 1.5–2.3 倍，表面重建 Chamfer 距离减少 28.9%，在不受限设置下仍保持竞争力。

**⚠️ 局限性**

限制在于一阶颜色模型难以表达高频纹理，需更小的原子；初始化可能导致覆盖不足；生成的网格不保证闭合或流形；在极小投影或极端倾斜视角下可能产生采样伪影。

---

